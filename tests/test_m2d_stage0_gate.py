"""The M2D advance gate, exercised on results that were built to exercise it.

Every fixture here is synthetic and says so. That is the point: a gate tested
against the real Stage-0 numbers would only ever confirm whatever those numbers
happen to be, and could not show that the gate is capable of returning the other
answer. What these tests establish is that each condition can pass, can fail,
and fails for the reason the declaration gives -- checked before the real
results were fetched.

The asymmetry between ADVANCE and STOP is checked too, because it is the least
obvious part of the rule. The gate needs only one condition, so an ADVANCE on A
or C is sound whatever B would have said. A STOP is not: it claims all three
failed, and one of them was never measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import m2d_stage0_gate as gate

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml").read_text(encoding="utf-8")
)
CELLS = gate.declared_cells(DECLARATION)
FAILURE = CELLS["failure"]
CONTROL = CELLS["control"]

BASE_RECALL_5 = 0.500
BASE_RECALL_1 = 0.300
BASE_MRR = 0.400


def _arm(name: str, *, eligible: bool, at_5: float, at_1: float = 0.0, mrr: float = 0.0) -> dict:
    """One arm's row, expressed as a delta in pp from the shared baseline."""

    return {
        name: {
            "recall@1": BASE_RECALL_1 + at_1 / 100.0,
            "recall@5": BASE_RECALL_5 + at_5 / 100.0,
            "recall@20": 0.700,
            "mrr": BASE_MRR + mrr / 100.0,
            "sources": ["S4"],
            "eligible_as_a_final_model": eligible,
            "role": "synthetic",
        }
    }


def _result(cell: str, arms: dict) -> dict:
    dataset, regime = cell.split("/")
    return {
        "status": "M2D_STAGE0_COMPLETE",
        "cell": cell,
        "dataset": dataset,
        "regime": regime,
        "trained_anything": False,
        "test_split_read": False,
        "source_commit": "a" * 40,
        "panel": {"queries": 1000, "portion": "synthetic"},
        "arms": {
            **_arm(gate.REFERENCE_ARM, eligible=True, at_5=0.0),
            **arms,
        },
    }


def _panel(*, z1: dict[str, float], z4: dict[str, float] | None = None) -> dict:
    """Four cells, each carrying one eligible arm and the diagnostic arm.

    ``z1`` and ``z4`` map a cell to its recall@5 delta in pp. Anything absent
    is flat, which is the neutral case for every condition.
    """

    z4 = z4 or {}
    results = {}
    for cell in FAILURE + CONTROL:
        arms = {
            **_arm("Z1_S4_DENSE", eligible=True, at_5=z1.get(cell, 0.0)),
            **_arm(
                gate.DIAGNOSTIC_ARM,
                eligible=False,
                at_5=z4.get(cell, 0.0),
                at_1=z4.get(cell, 0.0),
            ),
        }
        results[cell] = _result(cell, arms)
    return results


# ---------------------------------------------------------------------------
# The gate is the declaration's, not this file's
# ---------------------------------------------------------------------------


def test_the_thresholds_are_the_declared_ones() -> None:
    text = (REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml").read_text(encoding="utf-8")
    conditions = DECLARATION["advance_gate"]["advance_only_if_at_least_one_holds"]
    assert f"+{gate.IMPROVEMENT_PP:.2f}pp" in conditions["A_fixed_fusion_works"]
    assert f"{gate.CONTROL_REGRESSION_PP:.2f}pp" in conditions["A_fixed_fusion_works"]
    assert "not adjustable" in text or "are_not_adjustable" in text


def test_the_verdict_names_match_the_declaration() -> None:
    assert gate.STOP == DECLARATION["advance_gate"]["if_none_holds"]["verdict"]
    assert DECLARATION["advance_gate"]["filed_before_any_diagnostic_ran"] is True


def test_the_cells_are_the_declared_cells() -> None:
    assert FAILURE == ["squad_clean/R1", "musique_clean/R1"]
    assert set(CONTROL) == {"hotpotqa_clean/R1", "metaqa/R1"}


# ---------------------------------------------------------------------------
# Condition A can pass, and can fail for each of its three reasons
# ---------------------------------------------------------------------------


def test_a_passes_when_both_blockers_gain_and_one_reaches_the_bar() -> None:
    results = _panel(z1={FAILURE[0]: 0.30, FAILURE[1]: 0.10})
    verdict = gate.evaluate(results, DECLARATION)
    a = verdict["conditions"][0]
    assert a["holds"] is True
    assert a["passing_arms"] == ["Z1_S4_DENSE"]
    assert verdict["verdict"] == gate.ADVANCE


def test_a_fails_when_only_one_blocker_improves() -> None:
    """The declaration says BOTH. A large gain on one and a loss on the other
    is exactly the shape an average would hide."""

    results = _panel(z1={FAILURE[0]: 5.00, FAILURE[1]: -0.10})
    a = gate.evaluate(results, DECLARATION)["conditions"][0]
    assert a["holds"] is False
    assert a["rows"][0]["improves_both_failure_cells"] is False


def test_a_fails_when_both_improve_but_neither_reaches_the_bar() -> None:
    results = _panel(z1={FAILURE[0]: 0.20, FAILURE[1]: 0.20})
    a = gate.evaluate(results, DECLARATION)["conditions"][0]
    assert a["holds"] is False
    assert a["rows"][0]["improves_both_failure_cells"] is True
    assert a["rows"][0]["reaches_the_quarter_point_bar"] is False


def test_a_fails_when_a_control_regresses_past_the_allowance() -> None:
    """A repair that breaks a cell S4 already wins is not a repair."""

    results = _panel(z1={FAILURE[0]: 1.00, FAILURE[1]: 1.00, CONTROL[0]: -0.51})
    a = gate.evaluate(results, DECLARATION)["conditions"][0]
    assert a["holds"] is False
    assert a["rows"][0]["holds_both_controls"] is False


def test_a_control_regression_exactly_at_the_allowance_is_admitted() -> None:
    results = _panel(z1={FAILURE[0]: 1.00, FAILURE[1]: 1.00, CONTROL[0]: -0.50})
    a = gate.evaluate(results, DECLARATION)["conditions"][0]
    assert a["rows"][0]["holds_both_controls"] is True
    assert a["holds"] is True


def test_the_diagnostic_arm_can_never_satisfy_a() -> None:
    """Z4 runs two semantic models. However well it does, A must not see it."""

    results = _panel(z1={}, z4={cell: 10.0 for cell in FAILURE})
    a = gate.evaluate(results, DECLARATION)["conditions"][0]
    assert gate.DIAGNOSTIC_ARM not in [row["arm"] for row in a["rows"]]
    assert a["holds"] is False


def test_a_reports_how_many_arms_were_tried_not_just_the_winner() -> None:
    """A pass by one of four is a selection over four, and is reported so."""

    results = _panel(z1={FAILURE[0]: 1.00, FAILURE[1]: 1.00})
    for result in results.values():
        result["arms"].update(_arm("Z2_S4_SPLADE", eligible=True, at_5=-1.0))
        result["arms"].update(_arm("Z3_S4_DENSE_SPLADE", eligible=True, at_5=0.0))
    a = gate.evaluate(results, DECLARATION)["conditions"][0]
    assert a["arms_tried"] == 3
    assert a["arms_passed"] == 1
    assert "1 of 3" in a["selection_note"]


# ---------------------------------------------------------------------------
# Condition B is unmeasured, and the verdict is honest about it
# ---------------------------------------------------------------------------


def test_b_is_reported_unmeasured_rather_than_failed() -> None:
    b = gate.evaluate(_panel(z1={}), DECLARATION)["conditions"][1]
    assert b["measured"] is False
    assert b["holds"] is False
    assert b["cells_it_would_have_to_hold_on"] == FAILURE
    assert "dot_qd_pct" in b["what_would_settle_it"] or "percentile" in (
        b["what_would_settle_it"]
    )


def test_a_stop_is_not_returned_while_b_is_unmeasured() -> None:
    """The asymmetry, and the reason for it: an unmeasured condition might have
    passed, so a STOP resting on it is not a STOP."""

    verdict = gate.evaluate(_panel(z1={}), DECLARATION)
    assert verdict["conditions_holding"] == []
    assert verdict["verdict"] == gate.STOP_PENDING_B
    assert "unmeasured" in verdict["why"]


def test_an_advance_stands_even_though_b_is_unmeasured() -> None:
    """The gate needs one condition. A pass on A does not wait on B."""

    verdict = gate.evaluate(_panel(z1={c: 1.0 for c in FAILURE}), DECLARATION)
    assert verdict["verdict"] == gate.ADVANCE
    assert verdict["conditions"][1]["measured"] is False


def test_a_measured_b_would_allow_a_plain_stop(monkeypatch) -> None:
    """The STOP path has to be reachable, or the gate can never close a phase."""

    monkeypatch.setattr(
        gate,
        "condition_b",
        lambda results, cells: {
            "condition": "B_a_named_primitive_reorders",
            "holds": False,
            "measured": True,
        },
    )
    verdict = gate.evaluate(_panel(z1={}), DECLARATION)
    assert verdict["verdict"] == gate.STOP
    assert verdict["if_none_holds"]["then"].startswith("freeze S3")


# ---------------------------------------------------------------------------
# Condition C
# ---------------------------------------------------------------------------


def test_c_passes_when_the_diagnostic_arm_repairs_both_blockers() -> None:
    results = _panel(z1={}, z4={FAILURE[0]: 1.00, FAILURE[1]: 0.50})
    verdict = gate.evaluate(results, DECLARATION)
    c = verdict["conditions"][2]
    assert c["holds"] is True
    assert c["eligible_as_a_final_model"] is False
    assert verdict["verdict"] == gate.ADVANCE


def test_c_fails_when_only_one_blocker_is_repaired() -> None:
    results = _panel(z1={}, z4={FAILURE[0]: 5.00, FAILURE[1]: -1.00})
    c = gate.evaluate(results, DECLARATION)["conditions"][2]
    assert c["holds"] is False


def test_c_fails_when_recall_at_1_regresses() -> None:
    """recall@1 is the metric the blockers are blocked on. A fusion that lifts
    recall@5 by pushing the right answer out of rank 1 has repaired nothing."""

    results = _panel(z1={}, z4={cell: 1.0 for cell in FAILURE})
    for cell in FAILURE:
        arm = results[cell]["arms"][gate.DIAGNOSTIC_ARM]
        arm["recall@1"] = BASE_RECALL_1 - 0.01
    c = gate.evaluate(results, DECLARATION)["conditions"][2]
    assert c["holds"] is False


def test_c_never_makes_the_diagnostic_arm_eligible() -> None:
    results = _panel(z1={}, z4={cell: 20.0 for cell in FAILURE})
    verdict = gate.evaluate(results, DECLARATION)
    assert verdict["conditions"][2]["holds"] is True
    assert "Never that two semantic models" in verdict["conditions"][2]["what_a_pass_would_mean"]
    assert verdict["conditions"][0]["holds"] is False


# ---------------------------------------------------------------------------
# What the gate refuses to do
# ---------------------------------------------------------------------------


def test_a_verdict_over_fewer_than_four_cells_is_refused() -> None:
    results = _panel(z1={})
    del results[CONTROL[0]]
    with pytest.raises(ValueError, match="all four declared cells"):
        gate.evaluate(results, DECLARATION)


def test_a_result_that_claims_to_have_trained_is_refused(tmp_path: Path) -> None:
    result = _result(FAILURE[0], {})
    result["trained_anything"] = True
    (tmp_path / "cell.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="trained or read the test split"):
        gate.load_results(tmp_path)


def test_a_result_that_is_not_a_completed_stage_0_run_is_refused(tmp_path: Path) -> None:
    (tmp_path / "cell.json").write_text(json.dumps({"status": "PARTIAL"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a completed M2D Stage-0 result"):
        gate.load_results(tmp_path)


def test_an_enveloped_artifact_is_unwrapped(tmp_path: Path) -> None:
    """Results arrive inside run_artifacts' envelope, not bare."""

    payload = _result(FAILURE[0], {})
    (tmp_path / "cell.json").write_text(
        json.dumps({"identity": {}, "payload": payload}), encoding="utf-8"
    )
    assert gate.load_results(tmp_path)[FAILURE[0]]["cell"] == FAILURE[0]


def test_every_delta_is_within_one_panel_and_never_against_m2b() -> None:
    verdict = gate.evaluate(_panel(z1={}), DECLARATION)
    assert "within one cell's own" in verdict["comparison_basis"]
    source = (REPO_ROOT / "scripts" / "m2d_stage0_gate.py").read_text(encoding="utf-8")
    assert "m2b_baseline_table" not in source, (
        "the gate must not reach for M2B's table: it reports the holdout portion "
        "and subtracting it from this panel would subtract two different panels"
    )


def test_the_rendered_gate_states_the_verdict_and_the_arm_count() -> None:
    verdict = gate.evaluate(_panel(z1={c: 1.0 for c in FAILURE}), DECLARATION)
    text = gate.render(verdict)
    assert verdict["verdict"] in text
    assert verdict["conditions"][0]["selection_note"] in text
    assert "UNMEASURED" in text
    for cell in FAILURE + CONTROL:
        assert cell in text
