"""Package D is audited by a different shape than the training packages.

The serving benchmark trains nothing and generates no candidates: it loads
checkpoints from a completed candidate-budget cell and measures latency. Reusing
the training packages' verification would demand artifacts D never writes and
would classify every valid condition INVALID, which is worse than not auditing
it at all -- a false INVALID invites recomputing correct results.
"""

import pytest

import scripts.audit_modal_progress as apm
from scripts.audit_modal_integrity import _checkpoint_records, _verify_payload_contract

PACKAGE = "online_systems"
MODELS = ("sa_mlp", "seed_aware_gnn")


def _context() -> dict:
    return {
        "config": {"datasets": {"webqsp": {"selected_gnn": "gat"}}},
        "confirmations": {"webqsp": {"data_fingerprint_sha256": "abc"}},
    }


def _payload(**overrides) -> dict:
    payload = {
        "status": "UNCACHED_UNSEEN_EMBEDDING_SYSTEMS_COMPLETE",
        "dataset": "webqsp",
        "data_fingerprint_sha256": "abc",
        "config": {"selected_gnn": "gat"},
        "checkpoints": {
            model: {"path": f"/root/ckpt_{model}.pt", "sha256": "d" * 64}
            for model in MODELS
        },
        "boundary": {"query_specific_cache_reads_in_timed_path": False},
        "conditions": {"batch_1": {"models": {}}},
    }
    payload.update(overrides)
    return payload


def test_one_serving_benchmark_per_dataset_is_the_condition(tmp_path, monkeypatch) -> None:
    # D has no family, budget, axis or rate level, so the generic condition key
    # would raise on the fields it does not have.
    assert apm._condition_key(PACKAGE, {"dataset": "webqsp"}) == "webqsp"


def test_the_expected_matrix_is_the_dataset_set() -> None:
    keys = apm._expected_keys(PACKAGE)
    assert keys == {
        "2wiki_clean",
        "musique_clean",
        "webqsp",
        "hotpotqa_clean",
        "squad_clean",
        "metaqa",
    }


def test_d_is_registered_and_not_gated() -> None:
    assert PACKAGE in apm.PACKAGES
    assert apm._is_gated(PACKAGE) is False


def test_the_verified_artifact_is_the_reused_checkpoint() -> None:
    # D records its checkpoints under a different key and shape than the
    # training packages, and has no seed sweep to iterate.
    records = _checkpoint_records(PACKAGE, _payload())
    assert {model for model, _seed, _record in records} == set(MODELS)
    assert all(seed == 0 for _model, seed, _record in records)
    assert all(
        record["checkpoint_path"] and record["checkpoint_file_sha256"]
        for _model, _seed, record in records
    )


def test_a_conforming_benchmark_passes() -> None:
    assert _verify_payload_contract(PACKAGE, _payload(), _context()) == []


def test_d_is_not_asked_for_a_candidate_proof_it_never_writes() -> None:
    # The failure this exemption prevents: demanding candidate_contract would
    # make every valid D condition INVALID, and a false INVALID invites
    # recomputing results that were correct.
    errors = _verify_payload_contract(PACKAGE, _payload(), _context())
    assert not any("candidate" in error for error in errors)


def test_a_timed_path_that_read_a_query_specific_cache_is_refused() -> None:
    # The package exists to measure the uncached boundary. A cached read inside
    # the timed path measures a different system than the one reported.
    errors = _verify_payload_contract(
        PACKAGE,
        _payload(boundary={"query_specific_cache_reads_in_timed_path": True}),
        _context(),
    )
    assert "timed path read a query-specific cache" in errors


def test_an_unasserted_boundary_is_refused_rather_than_assumed_uncached() -> None:
    errors = _verify_payload_contract(PACKAGE, _payload(boundary={}), _context())
    assert "timed path read a query-specific cache" in errors


def test_a_benchmark_without_checkpoint_proof_is_refused() -> None:
    # Without it there is nothing tying the measured system to the models whose
    # effectiveness is reported elsewhere.
    errors = _verify_payload_contract(PACKAGE, _payload(checkpoints={}), _context())
    assert "missing reused checkpoint proof" in errors


def test_a_benchmark_on_the_wrong_fingerprint_is_refused() -> None:
    errors = _verify_payload_contract(
        PACKAGE, _payload(data_fingerprint_sha256="other"), _context()
    )
    assert "dataset fingerprint mismatch" in errors


def test_a_benchmark_on_a_different_gnn_family_is_refused() -> None:
    errors = _verify_payload_contract(
        PACKAGE, _payload(config={"selected_gnn": "gin"}), _context()
    )
    assert "selected GNN differs from protocol" in errors
