"""A killed substrate audit must not start over from zero.

The audit commits to the volume after every family/split, so a run that dies
partway leaves real measurements behind. Nothing adopted them: `completed_audit`
returns None for anything not marked complete, so the next attempt recomputed
all four graph families from scratch.

It is wasteful everywhere and worst on hotpotqa, whose 19,570 validation
queries run against a 507,494-node, 16.2M-edge graph. The Modal function's
ceiling is six hours and no retries are configured, so any run that does not
finish -- timeout, preemption, or a workspace hitting its spend limit, which is
what actually killed the previous attempt mid-audit -- discards everything it
measured, and the next one begins at exactly the same place with exactly the
same budget.

Carrying finished families forward is not a shortcut past a result. A family's
statistics are a deterministic function of the frozen graph and the frozen
candidate pools, and both are pinned by the data fingerprint that the contract
check has already matched, so recomputing one cannot change its value. What must
never be adopted is a family that is *itself* unfinished, and the split between
those two cases is what these tests pin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

audit = pytest.importorskip("scripts.run_graph_substrate_audit")

GRAPHS = ["dataset_default", "structural_only", "knn_only", "baseline_a_simple"]
SPLITS = ["validation"]
FINGERPRINT = "a39ed732e2a8788f"


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    namespace = argparse.Namespace(
        dataset="hotpotqa_clean",
        data_fingerprint_sha256=FINGERPRINT,
        graphs=list(GRAPHS),
        splits=list(SPLITS),
        max_hops=3,
        output=tmp_path / "substrate.json",
        resume_partial=True,
    )
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


def _family(*splits: str) -> dict:
    return {
        "provenance": "frozen",
        "stored_directed_edges": 16_223_058,
        "splits": {name: {"queries": 19_570} for name in splits},
    }


def _payload(*, status: str | None = None, families: dict | None = None, **overrides) -> dict:
    payload = {
        "status": status or audit.IN_PROGRESS_STATUS,
        "dataset": "hotpotqa_clean",
        "data_fingerprint_sha256": FINGERPRINT,
        "graphs_audited": list(GRAPHS),
        "splits_audited": list(SPLITS),
        "diagnostic_contract": {
            "max_hops": 3,
            "candidate_pools_modified": False,
        },
        "graphs": families if families is not None else {},
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "substrate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Which partial files are candidates for reuse at all
# --------------------------------------------------------------------------


def test_a_partial_audit_of_the_same_measurement_is_offered_for_reuse(tmp_path):
    _write(tmp_path, _payload(families={"dataset_default": _family("validation")}))
    assert audit.partial_audit(_args(tmp_path)) is not None


def test_a_finished_audit_is_not_offered_as_partial(tmp_path):
    """`completed_audit` already returns it whole; the two paths must not overlap."""
    _write(tmp_path, _payload(status=audit.COMPLETE_STATUS))
    assert audit.partial_audit(_args(tmp_path)) is None


def test_a_missing_file_is_not_an_error(tmp_path):
    assert audit.partial_audit(_args(tmp_path)) is None


def test_a_truncated_file_is_treated_as_absent_rather_than_as_a_violation(tmp_path):
    """A half-written file is evidence of a dead writer, not of a bad contract.

    Raising here would turn a crash during the final write into a permanent
    refusal that only a manual delete could clear.
    """
    (tmp_path / "substrate.json").write_text('{"status": "GRAPH_SUB', encoding="utf-8")
    assert audit.partial_audit(_args(tmp_path)) is None


def test_resumption_can_be_switched_off(tmp_path):
    _write(tmp_path, _payload(families={"dataset_default": _family("validation")}))
    assert audit.partial_audit(_args(tmp_path, resume_partial=False)) is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("dataset", "squad_clean"),
        ("data_fingerprint_sha256", "0000000000000000"),
        ("graphs_audited", ["dataset_default"]),
        ("splits_audited", ["test"]),
    ],
)
def test_a_partial_audit_of_a_different_measurement_is_refused(tmp_path, field, value):
    """Silently reusing another measurement's families is the worst outcome here.

    It would produce a file that looks complete and mixes two contracts, so the
    mismatch has to raise rather than fall back to recomputation.
    """
    _write(tmp_path, _payload(**{field: value}))
    with pytest.raises(ValueError, match="different diagnostic contract"):
        audit.partial_audit(_args(tmp_path))


def test_a_different_hop_budget_is_refused(tmp_path):
    payload = _payload()
    payload["diagnostic_contract"]["max_hops"] = 2
    _write(tmp_path, payload)
    with pytest.raises(ValueError, match="different diagnostic contract"):
        audit.partial_audit(_args(tmp_path))


def test_a_partial_audit_claiming_modified_pools_is_refused(tmp_path):
    """The read-only guarantee is not negotiable, even for a discarded file."""
    payload = _payload()
    payload["diagnostic_contract"]["candidate_pools_modified"] = True
    _write(tmp_path, payload)
    with pytest.raises(ValueError, match="different diagnostic contract"):
        audit.partial_audit(_args(tmp_path))


# --------------------------------------------------------------------------
# Which families inside such a file may actually be carried
# --------------------------------------------------------------------------


def test_a_family_with_every_populated_split_is_carried(tmp_path):
    payload = _payload(families={"dataset_default": _family("validation")})
    assert set(audit.adoptable_families(payload, _args(tmp_path), ["validation"])) == {
        "dataset_default"
    }


def test_a_family_missing_a_populated_split_is_not_carried(tmp_path):
    """Half a family is exactly the artifact that must never read as a result."""
    payload = _payload(families={"dataset_default": _family("validation")})
    carried = audit.adoptable_families(payload, _args(tmp_path), ["validation", "test"])
    assert carried == {}


def test_a_family_with_no_splits_at_all_is_not_carried(tmp_path):
    """The loop inserts the family entry *before* measuring anything into it.

    So an empty `splits` is the normal state of the family being worked on when
    the process died, and it is the single most likely thing to be found in a
    killed run's file.
    """
    payload = _payload(families={"dataset_default": {"provenance": "frozen", "splits": {}}})
    assert audit.adoptable_families(payload, _args(tmp_path), ["validation"]) == {}


def test_only_the_finished_families_of_a_mixed_file_are_carried(tmp_path):
    """The case this exists for: some families done, one interrupted."""
    payload = _payload(
        families={
            "dataset_default": _family("validation"),
            "structural_only": _family("validation"),
            "knn_only": {"provenance": "frozen", "splits": {}},
        }
    )
    carried = audit.adoptable_families(payload, _args(tmp_path), ["validation"])
    assert set(carried) == {"dataset_default", "structural_only"}


def test_carried_families_follow_the_requested_graph_order(tmp_path):
    """The report reads families positionally, so order is not cosmetic."""
    payload = _payload(
        families={
            "knn_only": _family("validation"),
            "dataset_default": _family("validation"),
        }
    )
    carried = audit.adoptable_families(payload, _args(tmp_path), ["validation"])
    assert list(carried) == ["dataset_default", "knn_only"]


def test_a_family_not_requested_by_this_run_is_ignored(tmp_path):
    payload = _payload(families={"some_other_family": _family("validation")})
    assert audit.adoptable_families(payload, _args(tmp_path), ["validation"]) == {}


def test_a_malformed_family_entry_is_skipped_rather_than_crashing(tmp_path):
    payload = _payload(families={"dataset_default": "not-a-mapping"})
    assert audit.adoptable_families(payload, _args(tmp_path), ["validation"]) == {}


def test_carrying_every_family_leaves_nothing_to_recompute(tmp_path):
    """A file one write short of complete must not trigger a full re-audit."""
    payload = _payload(families={name: _family("validation") for name in GRAPHS})
    carried = audit.adoptable_families(payload, _args(tmp_path), ["validation"])
    assert list(carried) == GRAPHS


# --------------------------------------------------------------------------
# The completed path must keep behaving exactly as before
# --------------------------------------------------------------------------


def test_a_complete_audit_is_still_returned_whole(tmp_path):
    payload = _payload(status=audit.COMPLETE_STATUS, families={"dataset_default": _family("validation")})
    _write(tmp_path, payload)
    assert audit.completed_audit(_args(tmp_path)) == payload


def test_a_complete_audit_of_a_different_contract_still_raises(tmp_path):
    _write(tmp_path, _payload(status=audit.COMPLETE_STATUS, dataset="metaqa"))
    with pytest.raises(ValueError, match="different diagnostic contract"):
        audit.completed_audit(_args(tmp_path))


def test_an_in_progress_file_is_still_never_returned_as_complete(tmp_path):
    """The original guarantee, unchanged: unfinished is not a result."""
    _write(tmp_path, _payload(families={name: _family("validation") for name in GRAPHS}))
    assert audit.completed_audit(_args(tmp_path)) is None
