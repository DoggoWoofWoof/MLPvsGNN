"""E2 must be monitorable the moment it launches, and gated until it may.

Package E2 is the one package whose registered matrix does not exist until the
screen analysis emits it. The audit therefore has to do two opposite things
correctly: refuse to invent a matrix while the gate is closed, and verify the
validation-only provenance of every cell once it is open.
"""

import json
from pathlib import Path

import pytest
import yaml

import scripts.audit_modal_progress as apm
from scripts.audit_modal_integrity import _verify_payload_contract

PACKAGE = "phase_confirmation"
AXES = {
    "degree_rewire": {"rates": [0.0, 0.25, 1.0], "perturbation_seed": 31415},
    "feature_mask": {"rates": [0.0, 1.0], "perturbation_seed": 14142},
}
TRAINING = {
    "seeds": [0, 1, 2, 3, 4],
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "ks": [1, 5, 20],
}
ORDER = "f" * 64


@pytest.fixture
def generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "configs").mkdir()
    config = {
        "modal": {"app": "message-passing-retrieval-phase-confirmation"},
        "axes": AXES,
        "datasets": {"webqsp": {"selected_gnn": "gat", "confirmation": "sealed.json"}},
        "training": TRAINING,
    }
    (tmp_path / "configs" / f"{PACKAGE}.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    monkeypatch.setattr(apm, "REPO_ROOT", tmp_path)
    return tmp_path


def _context() -> dict:
    return {
        "config": {
            "axes": AXES,
            "datasets": {"webqsp": {"selected_gnn": "gat"}},
            "training": TRAINING,
        },
        "confirmations": {
            "webqsp": {
                "data_fingerprint_sha256": "abc",
                "data": {"test_query_order_sha256": ORDER},
            }
        },
    }


def _payload(**overrides) -> dict:
    payload = {
        "dataset": "webqsp",
        "axis": "degree_rewire",
        "rate": 0.25,
        "data_fingerprint_sha256": "abc",
        "data": {"test_query_order_sha256": ORDER},
        "config": {"selected_gnn": "gat", "perturbation_seed": 31415, **TRAINING},
        "confirmation_contract": {
            "selected_by_locked_validation_only_rule": True,
            "test_selected_rate": False,
            "seed_zero_validation_checkpoint_reused_without_test_peeking": True,
        },
    }
    payload.update(overrides)
    return payload


def _errors(**overrides) -> list[str]:
    payload = _payload(**overrides)
    return [
        error
        for error in _verify_payload_contract(PACKAGE, payload, _context())
        if "candidate" not in error
    ]


def test_e2_is_gated_until_the_screen_analysis_generates_its_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Auditing E2 before the gate would require an expected matrix, and the only
    # way to produce one early is to guess the selected rates.
    monkeypatch.setattr(apm, "REPO_ROOT", tmp_path)
    assert apm._is_gated(PACKAGE) is True
    assert not any(apm._is_gated(other) for other in apm.PACKAGES if other != PACKAGE)


def test_the_gate_opens_once_the_protocol_exists(generated: Path) -> None:
    assert apm._is_gated(PACKAGE) is False


def test_the_clean_rate_is_not_part_of_the_expected_matrix(generated: Path) -> None:
    # The clean condition reuses the sealed confirmation instead of being
    # retrained, so no clean cell is ever written and expecting one would report
    # a permanent phantom shortfall.
    keys = apm._expected_keys(PACKAGE)
    assert keys == {
        "webqsp/degree_rewire/rate_0.25",
        "webqsp/degree_rewire/rate_1.00",
        "webqsp/feature_mask/rate_1.00",
    }


def test_a_conforming_confirmation_cell_passes(generated: Path) -> None:
    assert _errors() == []


def test_a_rate_chosen_from_test_outcomes_is_refused() -> None:
    # The single claim the whole package rests on.
    assert "confirmation rate was selected using test outcomes" in _errors(
        confirmation_contract={
            "selected_by_locked_validation_only_rule": True,
            "test_selected_rate": True,
            "seed_zero_validation_checkpoint_reused_without_test_peeking": True,
        }
    )


def test_an_unasserted_contract_is_refused_rather_than_assumed_clean() -> None:
    # A missing assertion is not evidence of compliance.
    errors = _errors(confirmation_contract={})
    assert "confirmation rate was selected using test outcomes" in errors
    assert "confirmation rate did not come from the locked rule" in errors
    assert "seed-0 checkpoint provenance not asserted" in errors


def test_a_clean_cell_written_into_e2_is_refused() -> None:
    assert "clean rate is not a confirmation cell" in _errors(rate=0.0)


def test_a_partial_seed_set_is_refused() -> None:
    # A confirmation on fewer than the registered five seeds is a different
    # design, not a smaller version of this one.
    config = {"selected_gnn": "gat", "perturbation_seed": 31415, **TRAINING}
    config["seeds"] = [0, 1, 2]
    assert "confirmation seed set differs" in _errors(config=config)


def test_e2_is_not_judged_by_the_screen_clause() -> None:
    # E1 must not compute test metrics; E2 exists to compute them. Applying E1's
    # clause here would mark every valid confirmation cell INVALID.
    assert "validation screen accessed test metrics" not in _errors()
