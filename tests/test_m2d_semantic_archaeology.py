"""The archaeology's instruments, checked on cases whose answer is known.

The script's whole claim is that it measures rather than reads. That claim is
worth exactly as much as the instruments are: a set-dependence probe that
returns an empty list for everything would "prove" S4 has no rank-aware column
while proving nothing at all. So the probe is run here against hand-built heads
whose properties are fixed by construction -- one column that is deliberately
computed against the candidate set, one deliberately constant, the rest
pointwise -- and it has to find exactly those.

The consistency checks inside the script are tested the same way, by feeding
them records that are wrong on purpose and requiring them to refuse. A check
that cannot fail is not a check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.run_artifacts import ArtifactError, verify_artifact_file
from scripts import m2d_semantic_archaeology as arch

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def heads() -> dict:
    return arch.build_heads(arch.PAYLOAD_WIDTH)


@pytest.fixture(scope="module")
def payload() -> dict:
    if not arch.BASELINE_TABLE.exists():
        pytest.skip("the immutable M2B baseline table is not on this machine")
    return arch.build_payload()


# ---------------------------------------------------------------------------
# The instruments, on known answers
# ---------------------------------------------------------------------------


class _KnownHead(torch.nn.Module):
    """Four columns whose properties are fixed by construction.

    ``pointwise`` depends on the candidate alone. ``constant`` is the same for
    every candidate of a query. ``set_relative`` subtracts the candidate set's
    mean, so removing any candidate moves every other candidate's value.
    ``pointwise_twin`` is there so a probe that flags everything fails too.
    """

    feature_names = ("pointwise", "constant", "set_relative", "pointwise_twin")
    rung = "KNOWN"

    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        pointwise = candidates.sum(dim=1)
        constant = torch.full_like(pointwise, float(query.sum()))
        set_relative = pointwise - pointwise.mean()
        return torch.stack([pointwise, constant, set_relative, pointwise * 2.0], dim=1)


def test_the_probe_finds_the_set_dependent_column_and_only_it():
    measured = arch.probe(_KnownHead().to(arch.DTYPE), width=8)
    assert measured["set_dependent_columns"] == ["set_relative"]


def test_the_probe_finds_the_constant_column_and_only_it():
    measured = arch.probe(_KnownHead().to(arch.DTYPE), width=8)
    assert measured["constant_within_query_columns"] == ["constant"]
    assert measured["constant_within_query_column_count"] == 1


def test_the_probe_refuses_a_head_whose_width_disagrees_with_its_names():
    class Mislabelled(_KnownHead):
        feature_names = ("pointwise", "constant")

    with pytest.raises(AssertionError, match="named columns"):
        arch.probe(Mislabelled().to(arch.DTYPE), width=8)


def test_the_probe_reports_row_equivariance():
    measured = arch.probe(_KnownHead().to(arch.DTYPE), width=8)
    assert measured["row_equivariant_under_candidate_permutation"] is True


def test_the_block_summary_round_trips():
    names = [f"state_product[{i}]" for i in range(64)] + ["normalized_state_dot"]
    folded = arch._summarise(names)
    assert folded == ["state_product[0..63]", "normalized_state_dot"]
    assert arch._expand(folded) == set(names)


# ---------------------------------------------------------------------------
# The live rungs
# ---------------------------------------------------------------------------


def test_the_live_heads_hold_the_parameters_the_fits_recorded(heads):
    assert heads["S2"].parameter_count() == 0
    assert heads["S3"].parameter_count() == 2 * arch.PAYLOAD_WIDTH == 3072
    assert heads["S4"].parameter_count() == 2 * arch.PAYLOAD_WIDTH * 64 == 196608


def test_the_live_heads_emit_the_column_counts_the_track_quotes(heads):
    assert len(heads["S2"].feature_names) == 3
    assert len(heads["S3"].feature_names) == 5
    assert len(heads["S4"].feature_names) == 258


def test_s4_has_no_set_dependent_column_at_all(heads):
    """The load-bearing measurement of the whole archaeology.

    Everything in S3_NOT_IN_S4 that says ABSENT rather than RESTRICTED rests on
    this: no column S4 emits changes when a different candidate leaves the set.
    """

    measured = arch.probe(heads["S4"], arch.PAYLOAD_WIDTH)
    assert measured["set_dependent_columns"] == []


def test_dot_qd_pct_is_the_only_set_dependent_column_in_s2_and_s3(heads):
    for rung in ("S2", "S3"):
        measured = arch.probe(heads[rung], arch.PAYLOAD_WIDTH)
        assert measured["set_dependent_columns"] == ["dot_qd_pct"], rung


def test_exactly_the_query_state_block_is_constant_within_a_query(heads):
    """64 of S4's 258 columns cannot reorder anything, and it is worth being
    sure it is those 64 and not some other coincidence of the probe's draw."""

    measured = arch.probe(heads["S4"], arch.PAYLOAD_WIDTH)
    assert measured["constant_within_query_column_count"] == 64
    assert arch._expand(measured["constant_within_query_columns"]) == {
        f"query_state[{index}]" for index in range(64)
    }


def test_no_s2_or_s3_column_is_constant_within_a_query(heads):
    for rung in ("S2", "S3"):
        measured = arch.probe(heads[rung], arch.PAYLOAD_WIDTH)
        # semantic_difference is inert at v = 0, which IS constant; that is a
        # fact about the initialization and the record should show it rather
        # than a test pretending otherwise.
        expected = {"semantic_difference"} if rung == "S3" else set()
        assert arch._expand(measured["constant_within_query_columns"]) == expected


# ---------------------------------------------------------------------------
# The consistency checks refuse when they should
# ---------------------------------------------------------------------------


def test_the_parameter_cross_check_refuses_a_live_count_that_drifted(payload):
    doctored = json.loads(json.dumps(payload["rungs"]))
    doctored["S4"]["parameters"]["counted_with_numel"] = 98304
    with pytest.raises(AssertionError, match="not describing the same model"):
        arch.check_against_baseline_table(doctored)


def test_the_column_accounting_refuses_rows_that_do_not_add_up(monkeypatch, payload):
    real = arch.primitive_rows

    def short(records):
        rows = real(records)
        return [row for row in rows if row["name"] != "normalized_state_dot"]

    monkeypatch.setattr(arch, "primitive_rows", short)
    with pytest.raises(AssertionError, match="the live heads"):
        arch.build_payload()


def test_the_rank_awareness_claim_is_checked_against_the_measurement(monkeypatch, payload):
    """A row that CLAIMED S4 sorts would have to survive the probe, and does not."""

    real = arch.primitive_rows

    def lying(records):
        rows = real(records)
        for row in rows:
            if row["name"] == "normalized_state_dot":
                row["sorts_or_ranks_over_the_candidate_set"] = True
        return rows

    monkeypatch.setattr(arch, "primitive_rows", lying)
    with pytest.raises(AssertionError, match="the probe measured"):
        arch.build_payload()


def test_the_scorer_cost_per_column_predicts_the_rung_it_was_not_derived_from(payload):
    cross = payload["parameter_cross_check"]["scorer_cost_per_semantic_column"]
    assert cross["derived_from"] == "S2 and S3"
    assert cross["parameters_per_column"] == 32
    assert cross["agrees"] is True
    assert cross["predicts_S4_remainder"] == cross["S4_remainder_recorded"] == 8609


def test_the_98304_correction_reproduces_both_counts_live():
    correction = arch.the_98304_correction()
    assert correction["quoted_constant"] == 98304
    assert correction["quoted_constant_is_about_width"] == 768
    assert correction["live_count_at_that_width"] == 98304
    assert correction["live_count_at_this_track_width"] == 196608


# ---------------------------------------------------------------------------
# The two required sets
# ---------------------------------------------------------------------------


def test_every_s3_not_in_s4_entry_says_absent_or_restricted(payload):
    entries = payload["S3_NOT_IN_S4"]
    assert entries, "the set the repairs are drawn from is empty"
    for entry in entries:
        assert entry["in_s4"] in {"ABSENT", "RESTRICTED"}, entry["primitive"]
        assert entry["restricted_how"].strip()
        assert entry["would_cost"].strip()
        if entry["in_s4"] == "RESTRICTED":
            assert entry["s4_analogue"], (
                f"{entry['primitive']} is called restricted with no analogue named; "
                "that is 'absent' with a softer word"
            )


def test_only_dot_qd_pct_is_absent_from_s4_entirely(payload):
    absent = [e["primitive"] for e in payload["S3_NOT_IN_S4"] if e["in_s4"] == "ABSENT"]
    assert absent == ["dot_qd_pct"]


def test_the_absent_entry_agrees_with_what_was_measured(payload):
    """The declaration forbids re-adding a primitive S4 already has. The only
    entry allowed to say ABSENT is the one the probe supports."""

    entry = next(e for e in payload["S3_NOT_IN_S4"] if e["in_s4"] == "ABSENT")
    assert entry["s4_analogue"] is None
    assert payload["rungs"]["S4"]["measured"]["set_dependent_columns"] == []
    assert payload["rank_aware_columns_by_rung"]["S4"] == []
    assert payload["rank_aware_columns_by_rung"]["S3"] == ["dot_qd_pct"]


def test_the_restricted_entries_do_not_silently_become_repairs(payload):
    """Three of the five are cheap raw geometry and two cost 1,536 parameters
    each; the record has to distinguish them, because section 7 asks whether
    cheap raw geometry alone is enough before anything expensive is proposed."""

    cheap = [e["primitive"] for e in payload["S3_NOT_IN_S4"] if e["raw_geometry_only"]]
    assert "cosine_qd at raw 1536" in cheap
    assert "mean_abs_diff at raw 1536" in cheap
    assert "dot_qd_pct" in cheap
    learned = [e for e in payload["S3_NOT_IN_S4"] if not e["raw_geometry_only"]]
    assert len(learned) == 2
    assert all("1,536 parameters" in e["would_cost"] for e in learned)


def test_the_reverse_set_records_which_primitives_cannot_reorder(payload):
    entries = payload["S4_NOT_IN_S3"]
    assert entries
    inert = [e["primitive"] for e in entries if not e["can_reorder_candidates"]]
    assert inert == ["query_state[0..63]"], (
        "the record should name exactly the block the probe measured to be constant "
        "within a query"
    )


def test_the_archaeology_refuses_to_name_a_cause(payload):
    assert "does not identify" in payload["what_this_does_not_say"] or (
        "Nothing here identifies the CAUSE" in payload["what_this_does_not_say"]
    )
    assert "forbade claiming its cause" in payload["what_this_does_not_say"]


# ---------------------------------------------------------------------------
# Writing it
# ---------------------------------------------------------------------------


def test_print_only_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(arch, "RUN_STORE", tmp_path / "runs")
    monkeypatch.setattr(arch, "SELECTED", tmp_path / "semantic_archaeology.json")
    if not arch.BASELINE_TABLE.exists():
        pytest.skip("the immutable M2B baseline table is not on this machine")

    assert arch.main(["--print-only"]) == 0
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "semantic_archaeology.json").exists()
    assert "semantic archaeology" in capsys.readouterr().out


def test_the_artifact_is_written_once_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(arch, "RUN_STORE", tmp_path / "runs")
    monkeypatch.setattr(arch, "SELECTED", tmp_path / "semantic_archaeology.json")
    if not arch.BASELINE_TABLE.exists():
        pytest.skip("the immutable M2B baseline table is not on this machine")

    assert arch.main([]) == 0

    written = list((tmp_path / "runs").rglob("result.json"))
    assert len(written) == 1
    receipt = verify_artifact_file(
        written[0],
        expect_source_commit=arch.source_commit(),
        expect_config_fingerprint=arch.config_fingerprint(),
    )
    assert receipt.identity.phase == "m2d"
    assert receipt.identity.arm == "archaeology"
    assert receipt.identity.seed is None
    assert receipt.row_count == len(receipt_payload(written[0])["primitives"])

    selected = json.loads((tmp_path / "semantic_archaeology.json").read_text(encoding="utf-8"))
    assert selected["source_commit"] == arch.source_commit()
    assert selected["payload"]["column_totals"]["S4"] == 258


def test_a_second_run_at_the_same_run_id_is_refused(tmp_path, monkeypatch):
    """The run id is what makes paths unique, so pinning it is the way to ask
    the writer to collide -- and it must refuse rather than overwrite."""

    monkeypatch.setattr(arch, "RUN_STORE", tmp_path / "runs")
    monkeypatch.setattr(arch, "SELECTED", tmp_path / "semantic_archaeology.json")
    monkeypatch.setattr(arch.run_artifacts, "current_run_id", lambda **_: "pinned-run")
    if not arch.BASELINE_TABLE.exists():
        pytest.skip("the immutable M2B baseline table is not on this machine")

    assert arch.main([]) == 0
    with pytest.raises(ArtifactError, match="Artifacts are immutable"):
        arch.main([])


def receipt_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["payload"]
