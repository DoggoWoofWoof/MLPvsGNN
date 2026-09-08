"""M2D Stage 1's report, on the parts that could misstate the result.

A report is where a stage's numbers stop being artifacts and start being
claims, so what is checked here is not that it renders but that it cannot
render something the artifacts do not say.

Three groups.

**It is derived, not typed.** Every figure in the document has to come from a
file. The test for that is not to re-read the document but to perturb the
inputs and require the output to move with them.

**It does not decide.** The verdict is the gate's. A report that computed its
own would be a second gate, applied after the numbers were visible.

**It keeps the distinction the verdict's label loses.** The gate assigned
case 3, whose filed pattern is "both fail", while A3-MINIMAL's only shortfall
was RESOLVABLE. A reader given the label alone would conclude something
stronger than the artifacts support, so the report is required to carry the
three-outcome vocabulary and not flatten it to pass/fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytest.importorskip("yaml")
pytest.importorskip("torch")

from scripts import m2d_stage1_gate as gate
from scripts import m2d_stage1_report as report

REPORT = REPO_ROOT / "docs" / "M2D_STAGE1_REPORT.md"
GATE_DOC = REPO_ROOT / "docs" / "M2D_STAGE1_GATE.md"

pytestmark = pytest.mark.skipif(
    len(gate.load_results()) != 8,
    reason="the eight Stage-1 fits are not present in this checkout",
)


@pytest.fixture(scope="module")
def inputs():
    return report.load()


@pytest.fixture(scope="module")
def text(inputs) -> str:
    return report.render(*inputs)


# ---------------------------------------------------------------------------
# It is derived, not typed
# ---------------------------------------------------------------------------


def test_the_committed_document_is_exactly_what_the_script_writes(text) -> None:
    """A document edited by hand after the fact is a document that can disagree
    with the artifacts it claims to summarise."""

    assert REPORT.is_file()
    assert REPORT.read_text(encoding="utf-8").replace("\r\n", "\n") == text


def test_every_arm_number_moves_when_the_artifact_moves(inputs) -> None:
    """The check that the tables are read rather than transcribed."""

    results, verdict, record, rows = inputs
    before = report.render(results, verdict, record, rows)

    key = ("musique_clean/R1", "A3_MINIMAL")
    mutated = dict(results)
    mutated[key] = json.loads(json.dumps(results[key]))
    mutated[key]["metrics"]["recall@5"] = 0.123456
    after = report.render(mutated, verdict, record, rows)

    assert "0.123456" in after
    assert "0.123456" not in before


def test_the_measured_cost_is_the_records_lines_repriced(inputs) -> None:
    """Not a fresh basis. Doubling a fit's training time must move the measured
    spend and leave the prediction it is compared against alone."""

    results, _verdict, record, _rows = inputs
    cost = report.measured_cost(results, record)

    heavier = dict(results)
    key = ("squad_clean/R1", "A1")
    heavier[key] = json.loads(json.dumps(results[key]))
    heavier[key]["systems"]["train_time_seconds"] *= 2
    dearer = report.measured_cost(heavier, record)

    assert dearer["measured_spend_usd"] > cost["measured_spend_usd"]
    assert dearer["predicted_spend_usd"] == cost["predicted_spend_usd"]
    assert dearer["measured_work_seconds"] > cost["measured_work_seconds"]


def test_the_measured_spend_uses_the_divisor_the_record_was_derived_at(inputs) -> None:
    results, _verdict, record, _rows = inputs
    cost = report.measured_cost(results, record)
    assert cost["utilisation_assumed"] == record["prediction"]["utilisation_assumed"]
    assert cost["usd_per_hour"] == record["container"]["usd_per_hour"]
    expected = (
        cost["measured_work_seconds"] / cost["utilisation_assumed"] / 3600.0
        * cost["usd_per_hour"]
    )
    assert cost["compute_usd"] == pytest.approx(expected)
    assert cost["measured_spend_usd"] == pytest.approx(
        expected + cost["container_overhead_usd"]
    )


def test_a_partial_stage_is_refused_rather_than_reported(monkeypatch) -> None:
    """Section 15 reports a completed stage. Eight fits were paid for; a report
    over seven would be a smaller experiment presented as the declared one."""

    full = gate.load_results()
    short = {key: value for key, value in list(full.items())[:7]}
    monkeypatch.setattr(gate, "load_results", lambda *a, **k: short)
    with pytest.raises(SystemExit, match="7 of 8"):
        report.load()


# ---------------------------------------------------------------------------
# It does not decide
# ---------------------------------------------------------------------------


def test_the_verdict_is_the_gates(text, inputs) -> None:
    _results, verdict, _record, _rows = inputs
    assert verdict["verdict"] in text
    assert verdict["prospective_case"]["case"] in text


def test_the_report_computes_no_verdict_of_its_own() -> None:
    """The gate's vocabulary must not be reachable from this module. A report
    that could name a verdict could name a different one."""

    source = (REPO_ROOT / "scripts" / "m2d_stage1_report.py").read_text(encoding="utf-8")
    for literal in ('"STOP_S4_DEVELOPMENT"', '"ADVANCE_TARGETED_M2D"', '"PASS"', '"FAIL"'):
        assert literal not in source, f"{literal} is decided here rather than read"


# ---------------------------------------------------------------------------
# It keeps the three-outcome vocabulary
# ---------------------------------------------------------------------------


def test_the_resolvable_outcome_is_reported_and_not_flattened(text, inputs) -> None:
    """The one thing a reader could get wrong from the case label alone."""

    _results, verdict, _record, _rows = inputs
    musique = verdict["arms"]["A3_MINIMAL"]["blockers"]["musique_clean/R1"]["outcome"]
    assert musique == "RESOLVABLE", "this test describes the result that was measured"
    assert "RESOLVABLE" in text
    assert "musique_clean/R1 RESOLVABLE" in text
    assert "seed 0" in text


def test_the_report_says_extra_seeds_were_not_authorised(text, inputs) -> None:
    """RESOLVABLE is the one state extra seeds could resolve, and section 14
    authorises them only on a pass. A report naming the state without naming
    the restriction would read as an invitation to run them."""

    _results, verdict, _record, _rows = inputs
    assert verdict["extra_seeds"]["authorised_by_this_gate"] is False
    assert "authorises it: false" in text


def test_the_mandatory_metrics_all_appear(text) -> None:
    for metric in ("R@1", "R@5", "R@20", "MRR", "FullCov@20"):
        assert metric in text
    for percentile in ("p50", "p95", "p99"):
        assert percentile in text


def test_the_parameter_counts_are_the_ones_the_arms_built(text, inputs) -> None:
    """Section 11's numbers, and section 3's bound on what may be added."""

    results, _verdict, _record, _rows = inputs
    parameters = results[("squad_clean/R1", "A3_MINIMAL")]["parameters"]
    assert parameters["added_semantic_parameters"] == 1536
    assert f"{parameters['total']:,}" in text
    assert "1,536" in text


def test_the_gate_document_is_the_gates_own_render(inputs) -> None:
    """Two documents, one source. The gate's render is not paraphrased into the
    report; it is committed as itself."""

    results, _verdict, _record, rows = inputs
    rendered = gate.render(
        gate.evaluate(results, gate.declaration(), rows)
    )
    assert GATE_DOC.is_file()
    assert GATE_DOC.read_text(encoding="utf-8").replace("\r\n", "\n").strip() == (
        rendered.strip()
    )
