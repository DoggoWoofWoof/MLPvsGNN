#!/usr/bin/env python
"""Apply M2D's filed advance gate to the Stage-0 results, and say what it returns.

Written and committed BEFORE the results existed, which is the only way a gate
means anything. It computes nothing new: every figure it reads was produced by
scripts/run_m2d_stage0_probe.py, verified on fetch, and selected by commit.

The gate is section 10's, quoted from configs/m2d_s4_semantic_repair.yaml rather
than paraphrased here, and it advances if at least ONE of three conditions
holds:

``A`` -- a fixed zero-training fusion improves recall@5 on BOTH failure cells,
with at least one improvement >= +0.25pp, and regresses NEITHER control cell by
more than 0.50pp on recall@5.

``B`` -- one missing S3 or raw semantic primitive systematically ranks the
relevant item above S4's wrong top item on BOTH failure cells.

``C`` -- the S3+S4 diagnostic RRF substantially repairs BOTH failure cells.

Three things about how they are applied.

**A is evaluated over every eligible arm, and the count is reported.** The
declaration says so in as many words: "a pass by one of four is a selection over
four". So this script reports how many arms were tried and how many passed, and
never narrows to the arm that happened to work.

**A compares each arm against Z0 within one panel, never against M2B's table.**
The probe runs on the FIT portion of the validation split; M2B's filed numbers
are the HOLDOUT portion. An arm-minus-Z0 delta inside one artifact is a real
comparison. An arm-minus-M2B number would be two different panels subtracted.

**B is not measurable from what Stage 0 produced, and that is recorded rather
than assumed away.** The probe ranks four whole models; B asks about a single
primitive, which needs a per-primitive ranking the probe does not emit. So B is
reported UNMEASURED. That matters asymmetrically and the verdict says so: an
ADVANCE on A or C stands on its own, because the gate needs only one condition.
A STOP does not, because an unmeasured condition might have passed -- so when A
and C both fail, this script returns STOP_PENDING_B rather than STOP_M2D, and
names the measurement that would settle it.

``C`` needs a reading of "substantially repairs" that is not chosen after the
fact. It is taken as the same bar A sets for a real improvement -- recall@5 up
on both failure cells with at least one >= +0.25pp -- applied to the Z4 arm,
plus recall@1 not regressing on either. Z4 remains ineligible as a model: a C
pass is evidence that an integrated representation is worth trying, never a
licence to ship two semantic models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

DECLARATION = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
RESULT_ROOT = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage0"
GATE_JSON = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage0_gate.json"

#: The reference arm every fusion is compared against. S4 alone, inside the
#: same panel, so the comparison is an ordering change and nothing else.
REFERENCE_ARM = "Z0_S4"

#: The S3+S4 arm, which condition C is about and which nothing may select.
DIAGNOSTIC_ARM = "Z4_S4_S3_DIAGNOSTIC_ONLY"

#: Percentage points. Both are the declaration's, and neither is adjustable.
IMPROVEMENT_PP = 0.25
CONTROL_REGRESSION_PP = 0.50

STOP = "STOP_M2D"
STOP_PENDING_B = "STOP_PENDING_B"
ADVANCE = "ADVANCE_TARGETED_M2D"


def declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION.read_text(encoding="utf-8"))


def declared_cells(config: dict[str, Any]) -> dict[str, list[str]]:
    cells = config["stage_0"]["cells"]
    return {
        "failure": list(cells["failure_cells"]),
        "control": [cells["passage_control"], cells["kb_control"]],
    }


def load_results(root: Path = RESULT_ROOT) -> dict[str, dict[str, Any]]:
    """The fetched Stage-0 artifacts, keyed by cell.

    Reads the selected logical result each cell's fetch wrote, and unwraps the
    envelope. A file that is not a completed M2D Stage-0 result is refused
    rather than skipped: a gate that silently ignores an unreadable cell would
    return a verdict over fewer cells than it claims.
    """

    results: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope.get("payload", envelope)
        if payload.get("status") != "M2D_STAGE0_COMPLETE":
            raise ValueError(f"{path} is not a completed M2D Stage-0 result")
        if payload.get("trained_anything") or payload.get("test_split_read"):
            raise ValueError(f"{path} claims to have trained or read the test split")
        results[payload["cell"]] = payload
    return results


def _pp(value: float) -> float:
    """A metric in [0, 1] as percentage points."""

    return float(value) * 100.0


def arm_deltas(result: dict[str, Any], metric: str = "recall@5") -> dict[str, float]:
    """Each arm minus the reference, in percentage points, within one panel."""

    arms = result["arms"]
    reference = _pp(arms[REFERENCE_ARM][metric])
    return {
        name: _pp(row[metric]) - reference for name, row in arms.items() if name != REFERENCE_ARM
    }


def eligible_arms(result: dict[str, Any]) -> list[str]:
    """Fusion arms that could be a final model, in the artifact's own words.

    Read from the artifact rather than from a list here, because the artifact
    records eligibility per arm and a second list could disagree with it.
    """

    return sorted(
        name
        for name, row in result["arms"].items()
        if row["eligible_as_a_final_model"] and name != REFERENCE_ARM
    )


def condition_a(results: dict[str, dict[str, Any]], cells: dict[str, list[str]]) -> dict[str, Any]:
    """Every eligible arm, evaluated on both blockers and both controls."""

    failure, control = cells["failure"], cells["control"]
    candidates = sorted(set.intersection(*(set(eligible_arms(results[c])) for c in failure)))
    rows = []
    for arm in candidates:
        gains = {cell: arm_deltas(results[cell])[arm] for cell in failure}
        regressions = {cell: arm_deltas(results[cell])[arm] for cell in control}
        improves_both = all(value > 0.0 for value in gains.values())
        reaches_bar = any(value >= IMPROVEMENT_PP for value in gains.values())
        holds_controls = all(value >= -CONTROL_REGRESSION_PP for value in regressions.values())
        rows.append(
            {
                "arm": arm,
                "recall_at_5_gain_pp": gains,
                "control_recall_at_5_delta_pp": regressions,
                "improves_both_failure_cells": improves_both,
                "reaches_the_quarter_point_bar": reaches_bar,
                "holds_both_controls": holds_controls,
                "passes": improves_both and reaches_bar and holds_controls,
            }
        )
    passing = [row["arm"] for row in rows if row["passes"]]
    return {
        "condition": "A_fixed_fusion_works",
        "arms_tried": len(rows),
        "arms_passed": len(passing),
        "passing_arms": passing,
        "selection_note": (
            f"{len(passing)} of {len(rows)} eligible arms pass. A pass by one of "
            f"{len(rows)} is a selection over {len(rows)} and is reported as such."
        ),
        "rows": rows,
        "holds": bool(passing),
    }


def condition_b(results: dict[str, dict[str, Any]], cells: dict[str, list[str]]) -> dict[str, Any]:
    """Not measurable from Stage 0 as it was run, and said so before it ran.

    The probe ranks four whole models. B asks whether ONE primitive reorders
    S4's mistakes, which needs a ranking per primitive -- cosine_qd, the raw
    dot, dot_qd_pct's within-query percentile, mean_abs_diff -- over the same
    pool. That is cheap, it is the same embeddings already loaded, and it is
    not what ran.
    """

    return {
        "condition": "B_a_named_primitive_reorders",
        "holds": False,
        "measured": False,
        "why": (
            "the Stage-0 probe emits rankings for four whole models and no ranking "
            "for any single primitive, so no evidence here bears on B either way"
        ),
        "what_would_settle_it": (
            "rank each cell's frozen pool by each raw primitive alone -- cosine_qd, "
            "the raw query-document dot, its within-query percentile, mean_abs_diff "
            "-- and report, on the queries S4 gets top-1 wrong, the share where that "
            "primitive alone ranks a relevant candidate above S4's wrong top item"
        ),
        "cells_it_would_have_to_hold_on": cells["failure"],
    }


def condition_c(results: dict[str, dict[str, Any]], cells: dict[str, list[str]]) -> dict[str, Any]:
    """The S3+S4 diagnostic arm, at the same bar A sets for a real improvement.

    "Substantially repairs" is read as A's bar plus no recall@1 regression,
    fixed here rather than chosen once the number is visible. A pass says an
    integrated representation is worth trying. It never says ship both models.
    """

    rows = {}
    for cell in cells["failure"]:
        result = results[cell]
        rows[cell] = {
            "recall_at_5_gain_pp": arm_deltas(result, "recall@5")[DIAGNOSTIC_ARM],
            "recall_at_1_gain_pp": arm_deltas(result, "recall@1")[DIAGNOSTIC_ARM],
            "mrr_gain_pp": arm_deltas(result, "mrr")[DIAGNOSTIC_ARM],
        }
    at_5 = [row["recall_at_5_gain_pp"] for row in rows.values()]
    at_1 = [row["recall_at_1_gain_pp"] for row in rows.values()]
    holds = all(v > 0.0 for v in at_5) and any(v >= IMPROVEMENT_PP for v in at_5)
    holds = holds and all(v >= 0.0 for v in at_1)
    return {
        "condition": "C_s3_plus_s4_repairs_both",
        "arm": DIAGNOSTIC_ARM,
        "eligible_as_a_final_model": False,
        "by_cell": rows,
        "holds": holds,
        "what_a_pass_would_mean": (
            "that the two representations carry different information, so an "
            "integrated single-model representation is worth trying. Never that "
            "two semantic models may be run at inference."
        ),
    }


def evaluate(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    cells = declared_cells(config)
    missing = sorted(set(cells["failure"] + cells["control"]) - set(results))
    if missing:
        raise ValueError(
            f"the gate needs all four declared cells and {missing} are absent. A "
            "verdict over three cells is not the verdict this gate describes."
        )

    a = condition_a(results, cells)
    b = condition_b(results, cells)
    c = condition_c(results, cells)
    conditions = [a, b, c]
    holding = [item["condition"] for item in conditions if item["holds"]]

    if holding:
        verdict, why = ADVANCE, f"{', '.join(holding)} holds"
    elif b["measured"]:
        verdict, why = STOP, "none of the three conditions holds"
    else:
        verdict, why = (
            STOP_PENDING_B,
            (
                "neither A nor C holds, and B was not measured. A STOP that rests on "
                "an unmeasured condition is not a STOP -- measure B, then re-apply "
                "this gate unchanged"
            ),
        )

    return {
        "status": "M2D_STAGE0_GATE_EVALUATED",
        "phase": "M2D",
        "stage": "stage_0",
        "gate_source": "configs/m2d_s4_semantic_repair.yaml#advance_gate",
        "gate_filed_before_any_diagnostic_ran": config["advance_gate"][
            "filed_before_any_diagnostic_ran"
        ],
        "cells": cells,
        "panel": {
            cell: {
                "queries": results[cell]["panel"]["queries"],
                "portion": results[cell]["panel"]["portion"],
                "source_commit": results[cell]["source_commit"],
            }
            for cell in sorted(results)
        },
        "comparison_basis": (
            "every delta is arm minus " + REFERENCE_ARM + " within one cell's own "
            "panel. Nothing here is compared against M2B's filed table, which "
            "reports the holdout portion and is a different panel."
        ),
        "conditions": conditions,
        "conditions_holding": holding,
        "verdict": verdict,
        "why": why,
        "if_none_holds": config["advance_gate"]["if_none_holds"],
    }


def render(gate: dict[str, Any]) -> str:
    a, b, c = gate["conditions"]
    lines = [
        "# M2D Stage-0 advance gate",
        "",
        f"**{gate['verdict']}** \u2014 {gate['why']}.",
        "",
        gate["comparison_basis"],
        "",
        "## Panels",
        "",
        "| cell | queries | commit |",
        "|---|---:|---|",
    ]
    for cell, row in gate["panel"].items():
        lines.append(f"| {cell} | {row['queries']:,} | `{row['source_commit'][:12]}` |")

    lines += ["", "## A \u2014 does a fixed fusion work?", "", a["selection_note"], ""]
    if not a["rows"]:
        lines.append("No eligible arm was common to both failure cells.")
    else:
        failure = list(a["rows"][0]["recall_at_5_gain_pp"])
        control = list(a["rows"][0]["control_recall_at_5_delta_pp"])
        header = ["arm", *(f"{cell} (blocker)" for cell in failure)]
        header += [f"{cell} (control)" for cell in control]
        header.append("passes")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---|" + "---:|" * (len(failure) + len(control)) + "---|")
        for row in a["rows"]:
            cells = [row["arm"]]
            cells += [f"{row['recall_at_5_gain_pp'][cell]:+.3f}" for cell in failure]
            cells += [f"{row['control_recall_at_5_delta_pp'][cell]:+.3f}" for cell in control]
            cells.append("yes" if row["passes"] else "no")
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            (
                f"Every figure is recall@5 in percentage points against {REFERENCE_ARM}. "
                f"An arm passes on a gain in both blockers, at least one reaching "
                f"+{IMPROVEMENT_PP:.2f}pp, and neither control down more than "
                f"{CONTROL_REGRESSION_PP:.2f}pp."
            ),
        ]

    lines += [
        "",
        "## B \u2014 does one named primitive reorder?",
        "",
        (
            "**UNMEASURED**, and recorded as such before the results existed "
            f"rather than after: {b['why']}."
        ),
        "",
        f"What would settle it: {b['what_would_settle_it']}.",
        "",
        "## C \u2014 does S3+S4 repair both blockers?",
        "",
        (
            f"Holds: **{'yes' if c['holds'] else 'no'}**. Diagnostic only "
            "— the arm runs two semantic models and can never be a result."
        ),
        "",
        "| cell | recall@1 | recall@5 | MRR |",
        "|---|---:|---:|---:|",
    ]
    for cell, row in c["by_cell"].items():
        lines.append(
            f"| {cell} | {row['recall_at_1_gain_pp']:+.3f} | "
            f"{row['recall_at_5_gain_pp']:+.3f} | {row['mrr_gain_pp']:+.3f} |"
        )
    lines += ["", c["what_a_pass_would_mean"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULT_ROOT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    gate = evaluate(load_results(args.results), declaration())
    markdown = render(gate)
    print(markdown)
    if args.print_only:
        return 0
    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATE_JSON.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"wrote {GATE_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
