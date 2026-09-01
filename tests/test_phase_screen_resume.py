import argparse
import json
from pathlib import Path

import pytest

from scripts.run_phase_screen import completed_screen_cell


def _args(path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "output": path,
        "dataset": "toy",
        "axis": "degree_rewire",
        "rate": 0.25,
        "data_fingerprint_sha256": "abc",
        "training_seed": 0,
        "perturbation_seed": 7,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "status": "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE",
        "dataset": "toy",
        "axis": "degree_rewire",
        "rate": 0.25,
        "data_fingerprint_sha256": "abc",
        "intervention": {"kind": "degree_rewire", "seed": 7},
        "screen_contract": {
            "split_evaluated": "validation_only",
            "test_metrics_computed": False,
            "training_seed": 0,
        },
        "models": {"sa_mlp": {}, "seed_aware_gnn": {}},
        "validation_gnn_minus_qls": {"recall@5": 0.01},
    }
    record.update(overrides)
    return record


def _write(path: Path, record: dict[str, object]) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_missing_output_runs_the_cell(tmp_path: Path) -> None:
    assert completed_screen_cell(_args(tmp_path / "result.json")) is None


def test_partial_cell_is_recomputed(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "result.json", _record(status="PHASE_SCREEN_IN_PROGRESS")
    )
    assert completed_screen_cell(_args(path)) is None


def test_complete_cell_is_reused_without_retraining(tmp_path: Path) -> None:
    path = _write(tmp_path / "result.json", _record())
    reused = completed_screen_cell(_args(path))
    assert reused is not None
    assert reused["validation_gnn_minus_qls"] == {"recall@5": 0.01}


@pytest.mark.parametrize(
    ("record_override", "args_override"),
    [
        ({}, {"data_fingerprint_sha256": "different"}),
        ({}, {"rate": 0.5}),
        ({}, {"axis": "random_add"}),
        ({}, {"dataset": "other"}),
        ({}, {"training_seed": 1}),
        ({}, {"perturbation_seed": 8}),
        ({"screen_contract": {
            "split_evaluated": "test",
            "test_metrics_computed": False,
            "training_seed": 0,
        }}, {}),
        ({"screen_contract": {
            "split_evaluated": "validation_only",
            "test_metrics_computed": True,
            "training_seed": 0,
        }}, {}),
    ],
)
def test_contract_mismatch_never_overwrites(
    tmp_path: Path,
    record_override: dict[str, object],
    args_override: dict[str, object],
) -> None:
    path = _write(tmp_path / "result.json", _record(**record_override))
    with pytest.raises(ValueError, match="different frozen contract"):
        completed_screen_cell(_args(path, **args_override))


def test_complete_cell_missing_a_model_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "result.json", _record(models={"sa_mlp": {}}))
    with pytest.raises(ValueError, match="missing a registered model"):
        completed_screen_cell(_args(path))


def test_complete_cell_without_contrast_is_rejected(tmp_path: Path) -> None:
    record = _record()
    del record["validation_gnn_minus_qls"]
    path = _write(tmp_path / "result.json", record)
    with pytest.raises(ValueError, match="no validation contrast"):
        completed_screen_cell(_args(path))
