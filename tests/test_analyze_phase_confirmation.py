"""E2's analysis must refuse anything that would let a test outcome pick a rate.

The confirmation is the only package that evaluates test metrics on conditions
chosen by an earlier analysis. Its value depends entirely on that choice having
been made from validation alone, so the compiler checks the assertion on every
cell rather than trusting the launcher that wrote it.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.analyze_phase_confirmation as apc
from scripts.analyze_phase_confirmation import (
    METRICS,
    MODEL_NAMES,
    PRIMARY_METRIC,
    _crossings,
    compile_analysis,
)

AXES = {
    "feature_mask": {"rates": [0.0, 1.0], "perturbation_seed": 14142},
}
SEEDS = [0, 1, 2, 3, 4]
ORDER = "e" * 64
QUERIES = 8


def _npz(path: Path, order: str = ORDER, offset: float = 0.0) -> str:
    arrays = {}
    for index, model in enumerate(MODEL_NAMES):
        block = np.full((QUERIES, len(METRICS)), 0.4 + index * offset, dtype=np.float32)
        for seed in SEEDS:
            arrays[f"{model}_seed_{seed}"] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metric_names=np.asarray(METRICS),
        query_order_sha256=np.asarray(order),
        **arrays,
    )
    return apc._sha256(path)


def _models(offset: float) -> dict:
    return {
        model: {
            "seeds": {
                str(seed): {
                    "metrics": {
                        metric: 0.4 + index * offset + seed * 1e-4 for metric in METRICS
                    }
                }
                for seed in SEEDS
            }
        }
        for index, model in enumerate(MODEL_NAMES)
    }


def _contract(**overrides) -> dict:
    contract = {
        "selected_by_locked_validation_only_rule": True,
        "test_selected_rate": False,
        "seed_zero_validation_checkpoint_reused_without_test_peeking": True,
    }
    contract.update(overrides)
    return contract


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "phase_confirmation"
    sealed = tmp_path / "sa_mlp_confirmation"
    sealed.mkdir(parents=True)

    config = {
        "axes": AXES,
        "datasets": {"webqsp": {"selected_gnn": "gat"}},
        "training": {"seeds": SEEDS},
    }
    config_path = tmp_path / "phase_confirmation.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(apc, "CONFIG_PATH", config_path)

    # Clean origin: the sealed five-seed confirmation, GNN below QLS.
    sealed_sha = _npz(sealed / "webqsp.query_metrics.npz", offset=-0.05)
    (sealed / "webqsp.json").write_text(
        json.dumps(
            {
                "status": apc.SEALED_COMPLETE,
                "dataset": "webqsp",
                "data": {"test_query_order_sha256": ORDER},
                "query_metrics": {"sha256": sealed_sha},
                "models": _models(-0.05),
            }
        ),
        encoding="utf-8",
    )

    # Perturbed cell at rate 1.0, GNN above QLS: a sign change.
    cell = root / "webqsp" / "feature_mask" / "rate_1p00"
    cell_sha = _npz(cell / "query_metrics.npz", offset=0.05)
    (cell / "result.json").write_text(
        json.dumps(
            {
                "status": apc.CELL_COMPLETE,
                "dataset": "webqsp",
                "axis": "feature_mask",
                "rate": 1.0,
                "data": {"test_query_order_sha256": ORDER},
                "query_metrics": {"sha256": cell_sha},
                "confirmation_contract": _contract(),
                "models": _models(0.05),
            }
        ),
        encoding="utf-8",
    )
    return root, sealed


def _compile(package: tuple[Path, Path]) -> dict:
    root, sealed = package
    return compile_analysis(root, sealed_root=sealed, bootstrap_replicates=16)


def _rewrite(package: tuple[Path, Path], **changes) -> None:
    root, _sealed = package
    path = root / "webqsp" / "feature_mask" / "rate_1p00" / "result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_a_conforming_matrix_compiles(package) -> None:
    analysis = _compile(package)
    assert analysis["status"] == "PHASE_CONFIRMATION_ALL_CELLS_ANALYZED"
    assert len(analysis["rows"]) == 2  # clean plus one perturbed rate


def test_the_clean_rate_is_read_from_the_sealed_confirmation(package) -> None:
    # Retraining a clean cell per axis would create several non-identical
    # origins for what the design treats as one shared baseline.
    root, _sealed = package
    assert not (root / "webqsp" / "feature_mask" / "rate_0p00").exists()
    analysis = _compile(package)
    clean = [row for row in analysis["rows"] if row["rate"] == 0.0]
    assert len(clean) == 1
    assert clean[0]["is_clean_origin"] is True


def test_a_rate_chosen_from_test_outcomes_is_refused(package) -> None:
    _rewrite(package, confirmation_contract=_contract(test_selected_rate=True))
    with pytest.raises(ValueError, match="selected without test outcomes"):
        _compile(package)


def test_a_missing_contract_is_refused_rather_than_assumed_clean(package) -> None:
    _rewrite(package, confirmation_contract={})
    with pytest.raises(ValueError, match="selected without test outcomes"):
        _compile(package)


def test_a_cell_that_does_not_assert_the_locked_rule_is_refused(package) -> None:
    _rewrite(
        package,
        confirmation_contract=_contract(selected_by_locked_validation_only_rule=False),
    )
    with pytest.raises(ValueError, match="locked rule"):
        _compile(package)


def test_a_cell_that_does_not_assert_seed_zero_provenance_is_refused(package) -> None:
    _rewrite(
        package,
        confirmation_contract=_contract(
            seed_zero_validation_checkpoint_reused_without_test_peeking=False
        ),
    )
    with pytest.raises(ValueError, match="seed-0 provenance"):
        _compile(package)


def test_an_incomplete_cell_is_refused(package) -> None:
    _rewrite(package, status="PHASE_CONFIRMATION_IN_PROGRESS")
    with pytest.raises(ValueError, match="not complete"):
        _compile(package)


def test_a_cell_filed_under_the_wrong_rate_is_refused(package) -> None:
    # A cell that ran one rate but sits in another rate's directory would place
    # a perturbation level at the wrong point on the phase curve.
    _rewrite(package, rate=0.25)
    with pytest.raises(ValueError, match="different rate"):
        _compile(package)


def test_packed_metrics_that_fail_their_digest_are_refused(package) -> None:
    root, _sealed = package
    _npz(root / "webqsp" / "feature_mask" / "rate_1p00" / "query_metrics.npz", offset=0.2)
    with pytest.raises(ValueError, match="failed SHA-256"):
        _compile(package)


def test_metrics_computed_on_a_different_query_order_are_refused(package) -> None:
    # Pairing depends on both models scoring the same queries in the same order.
    root, _sealed = package
    path = root / "webqsp" / "feature_mask" / "rate_1p00"
    digest = _npz(path / "query_metrics.npz", order="a" * 64, offset=0.05)
    _rewrite(package, query_metrics={"sha256": digest})
    with pytest.raises(ValueError, match="query order changed"):
        _compile(package)


def test_an_incomplete_sealed_clean_baseline_is_refused(package) -> None:
    _root, sealed = package
    payload = json.loads((sealed / "webqsp.json").read_text(encoding="utf-8"))
    payload["status"] = "SA_MLP_CONFIRMATION_IN_PROGRESS"
    (sealed / "webqsp.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        _compile(package)


def test_a_sign_change_is_reported_as_a_crossing(package) -> None:
    analysis = _compile(package)
    crossings = _crossings(analysis)
    assert len(crossings) == 1
    assert crossings[0]["interval"] == [0.0, 1.0]
    assert crossings[0]["gap_before"] < 0 < crossings[0]["gap_after"]


def test_a_crossing_between_two_null_cells_is_not_called_significant() -> None:
    # The failure this guard exists to prevent: two cells that are each
    # indistinguishable from zero straddle zero by noise alone, and the sign
    # change reads as a regime change.
    analysis = {
        "rows": [
            {
                "dataset": "webqsp",
                "axis": "feature_mask",
                "rate": rate,
                "seed_aware_gnn_minus_sa_mlp": {
                    PRIMARY_METRIC: {
                        "seed_effect": {"mean": mean},
                        "holm_pvalue": 0.9,
                    }
                },
            }
            for rate, mean in ((0.0, -0.0001), (1.0, 0.0001))
        ]
    }
    crossing = _crossings(analysis)[0]
    assert crossing["both_endpoints_holm_significant"] is False


def test_holm_is_applied_within_one_axis_and_rate(package) -> None:
    analysis = _compile(package)
    assert analysis["holm_scope"] == "datasets_within_axis_and_rate"
    for row in analysis["rows"]:
        assert "holm_pvalue" in row["seed_aware_gnn_minus_sa_mlp"][PRIMARY_METRIC]


def test_compiling_before_the_gate_is_refused_with_an_explanation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(apc, "CONFIG_PATH", tmp_path / "absent.yaml")
    with pytest.raises(SystemExit, match="analyze_phase_screen.py"):
        compile_analysis(tmp_path, sealed_root=tmp_path)
