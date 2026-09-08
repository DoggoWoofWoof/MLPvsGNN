"""The Stage-0 report: that it reports, and that it does not decide anything.

Section 20 of the declaration lists eleven things a report must carry, then a
verdict from a fixed pair, then a stop. A generated document can satisfy that
list and still be wrong in the one way that matters here -- by stating a number
or a conclusion that no filed artifact holds. So these tests are mostly about
provenance rather than prose:

**The order and the section list are the declaration's.** Renaming or dropping
an item in the report has to be a test failure, not a matter of proofreading.

**The verdict is the gate's.** ``ADVANCE_TARGETED_M2D`` appears in the document
because ``stage0_gate.json`` says so; feed the renderer a different gate and a
different verdict must come out, including the counts in the verdict table. A
report that would print an advance whatever the gate returned is not reporting.

**The numbers are read, not typed.** The Stage-1 cost is summed from M2B's own
measured S4 fit seconds; the sidedness labels and the "best of the raw ones"
figure are computed over the filed rows. Each of the three had a defect that a
first reading of the generated document caught, and each has a test below.
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.qls_v2_semantic import PARAMETER_FREE_FEATURE_NAMES
from scripts import m2d_stage0_report as report

ARTIFACTS_PRESENT = report.REPORT.exists() and report.GATE_JSON.exists()
needs_artifacts = pytest.mark.skipif(
    not ARTIFACTS_PRESENT, reason="the Stage-0 artifacts are not fetched in this checkout"
)


@pytest.fixture(scope="module")
def inputs() -> dict[str, object]:
    return {
        "declared": report.declaration(),
        "archaeology": report._payload(report.ARCHAEOLOGY),
        "results": report.stage0_results(),
        "primitives": report.primitive_results(),
        "gate": report._payload(report.GATE_JSON),
        "record": report._payload(report.COMPUTE_RECORD),
        "rows": report.baseline_rows(),
    }


@pytest.fixture(scope="module")
def rendered(inputs) -> str:
    return report.render(**inputs)


@pytest.fixture(scope="module")
def document() -> str:
    return report.REPORT.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The document is the script's output and nothing else
# ---------------------------------------------------------------------------


@needs_artifacts
def test_the_committed_document_is_exactly_what_the_script_writes(rendered, document) -> None:
    """No hand edit survives. The header says never edited by hand; this is that."""

    assert document.rstrip("\n") == rendered.rstrip("\n")


@needs_artifacts
def test_nothing_scientific_is_computed_here(inputs) -> None:
    for cell in {**inputs["results"], **inputs["primitives"]}.values():
        assert cell["trained_anything"] is False
        assert cell["fits_anything_new"] is False
        assert cell["test_split_read"] is False


# ---------------------------------------------------------------------------
# Section 20's list, in section 20's order
# ---------------------------------------------------------------------------


@needs_artifacts
def test_the_eleven_sections_are_the_declarations_eleven_in_its_order(
    inputs, rendered
) -> None:
    items = inputs["declared"]["stop_condition"]["after_stage_0_or_stage_1_report"]
    assert len(items) == 11
    headings = re.findall(r"^## (\d+)\. (.+)$", rendered, flags=re.MULTILINE)
    assert [int(number) for number, _ in headings] == sorted(items)
    for number, heading in headings:
        declared = items[int(number)]
        assert heading.lower() == declared.lower(), (number, heading, declared)


@needs_artifacts
def test_a_renamed_item_moves_the_heading_with_it(inputs) -> None:
    """The headings are read from the declaration, not typed beside it."""

    mutated = copy.deepcopy(inputs)
    items = mutated["declared"]["stop_condition"]["after_stage_0_or_stage_1_report"]
    items[7] = "a heading no one would write by hand"
    assert "## 7. A heading no one would write by hand" in report.render(**mutated)


@needs_artifacts
def test_the_verdict_and_the_stop_follow_the_eleven(rendered) -> None:
    order = [rendered.index(marker) for marker in ("## 1. ", "## 11. ", "## Verdict", "## STOP_FOR_REVIEW")]
    assert order == sorted(order)


# ---------------------------------------------------------------------------
# The verdict is the gate's
# ---------------------------------------------------------------------------


@needs_artifacts
def test_the_verdict_printed_is_the_verdict_the_gate_returned(inputs, rendered) -> None:
    assert f"**{inputs['gate']['verdict']}**" in rendered


@needs_artifacts
def test_a_different_gate_prints_a_different_verdict(inputs) -> None:
    """The report must not be able to advance a phase the gate stopped."""

    mutated = copy.deepcopy(inputs)
    mutated["gate"]["verdict"] = "STOP_S4_DEVELOPMENT"
    mutated["gate"]["conditions_holding"] = []
    text = report.render(**mutated)
    assert "**STOP_S4_DEVELOPMENT**" in text
    assert "**ADVANCE_TARGETED_M2D**" not in text


@needs_artifacts
def test_the_verdict_tables_counts_come_from_the_gate(inputs) -> None:
    """`1 of 5` is the claim. It has to move when the gate's count moves."""

    condition = next(
        c for c in inputs["gate"]["conditions"] if c["condition"].startswith("B_")
    )
    assert report.verdict_row(condition)[2].startswith(
        f"{condition['primitives_passed']} of {condition['primitives_tried']}: "
    )
    mutated = copy.deepcopy(condition)
    mutated["primitives_passed"] = 3
    mutated["primitives_tried"] = 9
    assert report.verdict_row(mutated)[2].startswith("3 of 9: ")


@needs_artifacts
def test_a_condition_that_holds_is_marked_and_one_that_does_not_is_not(inputs) -> None:
    for condition in inputs["gate"]["conditions"]:
        holds = report.verdict_row(condition)[1]
        assert holds == ("**yes**" if condition["holds"] else "no")


# ---------------------------------------------------------------------------
# What the diagnostics removed
# ---------------------------------------------------------------------------


@needs_artifacts
def test_a2_is_reported_removed_and_a4_is_not_licensed(rendered) -> None:
    matrix = rendered[rendered.index("## The exact minimum next matrix") :]
    a2 = next(line for line in matrix.splitlines() if line.startswith("| A2 "))
    a4 = next(line for line in matrix.splitlines() if line.startswith("| A4 "))
    assert "**removed**" in a2
    assert "not a deferral" in a2
    assert "not licensed" in a4


@needs_artifacts
def test_the_removal_of_a2_is_what_condition_a_returned(inputs) -> None:
    condition = next(
        c for c in inputs["gate"]["conditions"] if c["condition"].startswith("A_")
    )
    assert condition["holds"] is False
    assert condition["arms_passed"] == 0
    for cell, results in inputs["results"].items():
        for arm, deltas in results["deltas_vs_s4_pp"].items():
            if not arm.startswith("Z") or arm == "Z0_S4":
                continue
            if arm.endswith("DIAGNOSTIC_ONLY"):
                continue
            assert deltas["recall@5"] < 0, (cell, arm)


@needs_artifacts
def test_the_diagnostic_only_arm_is_never_offered_as_an_arm(rendered) -> None:
    matrix = rendered[rendered.index("## The exact minimum next matrix") :]
    assert "Z4" not in matrix
    assert "diagnostic only" in rendered[: rendered.index("## The exact minimum next matrix")]


# ---------------------------------------------------------------------------
# The Stage-1 cost is summed from filed measurements
# ---------------------------------------------------------------------------


@needs_artifacts
def test_the_fit_seconds_are_m2bs_own_measured_seconds(inputs) -> None:
    seconds = report.m2b_r1_fit_seconds()
    assert set(seconds) == set(report.STAGE_1_CELL_DATASETS)
    for dataset, value in seconds.items():
        filed = report._payload(next(report.M2B_HEADLINE.glob(f"*{dataset}*.json")))
        assert value == filed["cells"]["R1"]["rungs"]["S4"]["training"]["training_seconds"]


@needs_artifacts
def test_the_per_arm_total_is_the_sum_of_those_seconds(rendered) -> None:
    seconds = report.m2b_r1_fit_seconds()
    matrix = rendered[rendered.index("**Cost.**") :]
    assert f"**{sum(seconds.values()):.1f}**" in matrix
    for value in seconds.values():
        assert f"| {value:.1f} |" in matrix


@needs_artifacts
def test_the_cost_is_labelled_an_estimate_and_not_a_filed_record(rendered) -> None:
    assert "no such record exists for Stage 1" in rendered


@needs_artifacts
def test_the_matrix_is_the_declarations_cells_and_seed_zero(inputs, rendered) -> None:
    matrix = rendered[rendered.index("## The exact minimum next matrix") :]
    cells = inputs["declared"]["stage_1"]["cells"]
    for cell in [*cells["mandatory_blockers"], *cells["controls"]]:
        assert f"`{cell}`" in matrix
    assert inputs["declared"]["stage_1"]["seeds"] == [0]
    assert "Seeds: [0]" in matrix


@needs_artifacts
def test_the_report_does_not_authorise_the_matrix_it_specifies(rendered) -> None:
    """True whatever the declaration's gates say later. A report is not an
    authorisation, and this one specifies a matrix it may not launch."""

    assert "It is not authorised by this document" in rendered
    assert "STOP_FOR_REVIEW" in rendered


@needs_artifacts
def test_the_record_is_pinned_to_the_status_stage_0_closed_at(inputs, rendered) -> None:
    """Stage 0's record must not re-render itself under Stage 1's status.

    The header status is derived from the gate's verdict, so it stays where
    Stage 0 left it; the phase's live status appears only as a forward pointer
    to what happened after the stop.
    """

    gate = inputs["gate"]
    assert f"Stage 0 closed at `{report.stage_0_status(gate)}`" in rendered
    assert report.stage_0_status(gate) == f"M2D_STAGE0_GATE_RETURNED_{gate['verdict']}"
    assert "this document has not, and must not" in rendered

    mutated = copy.deepcopy(inputs)
    mutated["declared"]["status"] = "SOME_LATER_STATUS"
    text = report.render(**mutated)
    assert f"Stage 0 closed at `{report.stage_0_status(gate)}`" in text
    assert "The declaration now reads `SOME_LATER_STATUS`" in text


# ---------------------------------------------------------------------------
# The three defects a first reading of the generated document caught
# ---------------------------------------------------------------------------


def test_an_empty_right_set_means_no_column_is_right_aligned() -> None:
    """`right = right or {...}` cannot express "none of them"; `is None` can.

    Passing `right=set()` marks a table as all prose. Under `or`, an empty set
    is falsy and silently becomes the default, so every prose table in the
    document came out right-aligned.
    """

    rule = report.table(["a", "b", "c"], [["1", "2", "3"]], right=set())[1]
    assert rule == "|---|---|---|"
    assert report.table(["a", "b"], [["1", "2"]])[1] == "|---|---:|"
    assert report.table(["a", "b", "c"], [["1", "2", "3"]], right={2})[1] == "|---|---|---:|"


def test_the_sidedness_label_can_name_either_rung() -> None:
    """A two-way label reading everything under a half as "traded" called the
    panel's most lopsided cell its most balanced one."""

    assert report.sidedness(0.669) == "S4 dominated"
    assert report.sidedness(0.151) == "S3 dominated"
    assert report.sidedness(0.444) == "S3 dominated"


@needs_artifacts
def test_the_best_raw_primitive_quoted_is_the_best_of_them(inputs, rendered) -> None:
    """The bullet says "the best of the three"; it once quoted the second-worst."""

    condition = next(
        c for c in inputs["gate"]["conditions"] if c["condition"].startswith("B_")
    )
    blocker = condition["cells_it_would_have_to_hold_on"][0]
    raw = {
        row["primitive"]: row["share_of_s4_top1_errors_reordered"][blocker]
        for row in condition["rows"]
        if row["primitive"] in PARAMETER_FREE_FEATURE_NAMES
    }
    assert set(raw) == set(PARAMETER_FREE_FEATURE_NAMES)
    assert f"the best of the three reaching {max(raw.values()):.4f}" in rendered
    for value in raw.values():
        assert value < condition["bar"]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


@needs_artifacts
def test_every_table_is_rectangular(rendered) -> None:
    width = None
    for line in rendered.splitlines():
        if not line.startswith("|"):
            width = None
            continue
        cells = len(line.split("|")) - 2
        if width is None:
            width = cells
        assert cells == width, line


@needs_artifacts
def test_the_launch_history_is_reported_in_full(rendered) -> None:
    """Section 18 asks for the spend, not for the spend of the run that worked.

    Every launch gets a row, the totals are summed over the rows rather than
    stated beside them, and the artifacts the history claims landed are the
    artifacts this report actually read.
    """

    containers = sum(entry["containers"] for entry in report.LAUNCH_HISTORY)
    artifacts = sum(entry["artifacts"] for entry in report.LAUNCH_HISTORY)
    assert f"{containers} container starts in total; {artifacts} produced an artifact" in rendered
    assert artifacts == len(report.stage0_results()) + len(report.primitive_results())
    rows = [
        line
        for line in rendered.splitlines()
        if re.match(r"^\| \d+ \| (stage0|primitives) \|", line)
    ]
    assert len(rows) == len(report.LAUNCH_HISTORY)
    for index, entry in enumerate(report.LAUNCH_HISTORY, start=1):
        assert rows[index - 1].startswith(
            f"| {index} | {entry['stage']} | {entry['containers']} | {entry['artifacts']} |"
        )
    landed = {"stage0": len(report.stage0_results()), "primitives": len(report.primitive_results())}
    for stage, count in landed.items():
        claimed = sum(e["artifacts"] for e in report.LAUNCH_HISTORY if e["stage"] == stage)
        assert claimed == count, stage
    # A launch that produced NOTHING leaves nothing on disk, so no assertion here
    # can confirm one happened. That asymmetry is why the history is recorded by
    # hand and labelled as such, and why the spend below it is a bound and not a
    # bill: what is checkable is that every artifact claimed was in fact fetched.


@needs_artifacts
def test_both_frozen_p95_references_are_quoted_not_just_the_flattering_one(
    inputs, rendered
) -> None:
    """The declaration froze two pairs so neither could be quoted on its own.

    The 14-cell mean the report computes is the all-fits pair; the gate reads
    the blocker subset. Printing only the mean would be the selective quote the
    declaration's own note exists to prevent.
    """

    reference = inputs["declared"]["systems_gate"]["reference_values"]
    assert f"S4 {reference['S4_uncached_p95_ms']:.4f} ms" in rendered
    assert f"S3 {reference['S3_uncached_p95_ms']:.4f} ms" in rendered
    assert reference["subset"] in rendered
    assert " ".join(reference["note"].split()) in rendered
    assert " ".join(inputs["declared"]["systems_gate"]["requirement"].split()) in rendered
    assert "M2D measured no systems number" in rendered


@needs_artifacts
def test_no_number_is_stated_as_a_bill(rendered) -> None:
    assert "The billed figure was not read back from Modal" in rendered


@needs_artifacts
def test_s4_is_not_described_as_small(inputs, rendered) -> None:
    """Section 14. The phase arguing for a cheap repair states what it repairs."""

    check = inputs["archaeology"]["parameter_cross_check"]
    assert "**S4 is not small.**" in rendered
    assert f"{check['S4']['total_parameters_recorded']:,}" in rendered
    assert f"{check['S4']['semantic_parameters_live']:,}" in rendered
