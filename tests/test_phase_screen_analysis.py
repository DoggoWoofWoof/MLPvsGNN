import json
from pathlib import Path

import pytest
import yaml

from scripts import analyze_phase_screen
from scripts.analyze_phase_screen import compile_analysis, select_confirmation_rates

DATASET = "toy"
AXIS = "degree_rewire"
RATES = [0.10, 0.25]


def test_phase_selection_no_crossing_keeps_only_endpoints() -> None:
    points = [(0.0, 0.1), (0.1, 0.08), (0.25, 0.05), (1.0, 0.01)]
    assert select_confirmation_rates(points) == [0.0, 1.0]


def test_phase_selection_keeps_each_crossing_bracket() -> None:
    points = [(0.0, 0.1), (0.1, 0.02), (0.25, -0.01), (0.5, -0.03), (1.0, 0.04)]
    assert select_confirmation_rates(points) == [0.0, 0.1, 0.25, 0.5, 1.0]


def test_phase_selection_exact_zero_keeps_neighbors_and_endpoints() -> None:
    points = [(0.0, 0.1), (0.1, 0.02), (0.25, 0.0), (0.5, -0.03), (1.0, -0.04)]
    assert select_confirmation_rates(points) == [0.0, 0.1, 0.25, 0.5, 1.0]


def _screen(root: Path, rate: float, **contract: object) -> None:
    cell = root / DATASET / AXIS
    cell.mkdir(parents=True, exist_ok=True)
    key = f"{rate:.2f}".replace(".", "p")
    (cell / f"rate_{key}.json").write_text(
        json.dumps(
            {
                "status": "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE",
                "dataset": DATASET,
                "axis": AXIS,
                "rate": rate,
                "screen_contract": {
                    "test_metrics_computed": False,
                    "training_seed": 0,
                    **contract,
                },
                "validation_gnn_minus_qls": {"recall@5": 0.01},
                "models": {
                    "sa_mlp": {"validation_metrics": {"recall@5": 0.50}},
                    "seed_aware_gnn": {"validation_metrics": {"recall@5": 0.51}},
                },
                "intervention": {"axis": AXIS, "rate": rate},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def screen_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A hermetic one-dataset, one-axis screen with its own config and sources."""
    config = {
        "axes": {AXIS: {"rates": RATES}},
        "datasets": {DATASET: {"confirmation": "confirmation.json"}},
        "screen_selection_rule": {"metric": "validation_recall_at_5_gnn_minus_qls"},
        "perturbation_seeds": {AXIS: 0},
        "training": {"epochs": 1},
        "parameter_regime": {"hidden_dim": 64},
        "modal": {"app": "toy"},
    }
    config_path = tmp_path / "phase_screen.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "confirmation.json").write_text(
        json.dumps(
            {
                "models": {
                    model: {
                        "seeds": {"0": {"training": {"best_validation_recall@5": value}}}
                    }
                    for model, value in (("sa_mlp", 0.50), ("seed_aware_gnn", 0.52))
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(analyze_phase_screen, "CONFIG_PATH", config_path)
    monkeypatch.setattr(analyze_phase_screen, "REPO_ROOT", tmp_path)
    root = tmp_path / "phase_screen"
    for rate in RATES:
        _screen(root, rate)
    return root


def test_complete_screen_is_analyzed_without_reading_any_test_metric(
    screen_root: Path,
) -> None:
    analysis, confirmation = compile_analysis(screen_root)
    assert analysis["status"] == "PHASE_SCREEN_VALIDATION_ONLY_ANALYZED"
    assert analysis["test_metrics_computed"] is False
    assert analysis["predictor_training_allowed"] is False
    assert analysis["stopping_point"] == "RATES_SELECTED_REQUIRES_PROTOCOL_COMMIT_BEFORE_TEST"
    assert (
        confirmation["status"]
        == "GENERATED_FROM_LOCKED_VALIDATION_RULE_REQUIRES_COMMIT_BEFORE_TEST"
    )
    assert confirmation["training"]["seeds"] == [0, 1, 2, 3, 4]


def test_a_cell_that_computed_test_metrics_is_refused(screen_root: Path) -> None:
    # The gate that keeps E2 rate selection validation-only. If this ever
    # passes, test outcomes could reach the selection rule.
    _screen(screen_root, RATES[0], test_metrics_computed=True)
    with pytest.raises(ValueError, match="contract failed"):
        compile_analysis(screen_root)


def test_a_cell_trained_on_another_seed_is_refused(screen_root: Path) -> None:
    _screen(screen_root, RATES[0], training_seed=3)
    with pytest.raises(ValueError, match="contract failed"):
        compile_analysis(screen_root)


def test_an_incomplete_screen_is_refused_rather_than_partially_selected(
    screen_root: Path,
) -> None:
    key = f"{RATES[-1]:.2f}".replace(".", "p")
    (screen_root / DATASET / AXIS / f"rate_{key}.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        compile_analysis(screen_root)
