#!/usr/bin/env python
"""Apply M2D's Stage-1 gate to the eight arm fits, and say what it returns.

Written and committed BEFORE any Stage-1 fit exists, which is the only way a
gate means anything. It computes nothing new: every figure it reads is produced
by scripts/run_m2d_stage1_arms.py, verified on fetch, and selected by commit,
or is a reused row from M2B's immutable baseline table.

The rule is section 12's, unchanged by the section 8b amendment, which says in
as many words that it alters no threshold:

* **Blockers.** On squad_clean/R1 and musique_clean/R1 the arm must be within
  0.50pp of S3 on recall@5.
* **Controls.** On hotpotqa_clean/R1 and metaqa/R1 the arm must not regress
  recall@5 by more than 0.50pp against native S4.
* **Rank 1 is not supporting detail.** Section 12 says a challenger that fixes
  recall@5 and leaves rank 1 where it is has not repaired the failure this
  phase was opened for. So an arm that clears recall@5 without improving
  recall@1 against native S4 on both blockers is reported as NOT a repair.
* **Systems.** Section 13: total uncached p95 must stay below S3's.

Five things about how they are applied, each fixed here rather than after.

**"Within 0.50pp" is a signed bound, not a distance.** ``arm - S3 >= -0.50pp``.
An arm that BEATS S3 is not failed for being more than 0.50pp away from it.

**A miss smaller than the cell's own seed noise is not a pass.** It is reported
RESOLVABLE, which is a third outcome and not a softer second one. The band is
not typed here: it is the measured spread of native S4's recall@5 across M2B's
three filed seeds on that same cell, so a difference seed 0 provably cannot
settle is named as unsettled rather than decided. RESOLVABLE never advances the
phase on its own; it is the one state in which section 15's extra seeds could
change the decision, which is exactly the condition the declaration puts on
running them.

**Every arm is judged and the count is reported.** Two arms are tried. A pass
by one of two is a selection over two and is reported with its denominator, the
way condition B's one-of-five was.

**Rank-1 repair is a direction, not a magnitude.** No magnitude was ever filed
for it, and inventing one now -- after the Stage-0 diagnostic named the primitive
-- would be choosing a threshold with the result in view. So the test is that
recall@1 improves against native S4 on both blockers, and the sizes are
reported beside it for the reader.

**The four prospective readings are the declaration's, applied mechanically.**
Section 8b filed CASE 1-4 before these arms ran. This script picks the case from
the pass/fail pattern; it does not choose a reading after seeing the numbers.

The verdict pair is section 20's: ``STOP_S4_DEVELOPMENT`` or
``ADVANCE_TARGETED_M2D``. Then STOP_FOR_REVIEW, always.
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
RESULT_ROOT = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage1"
GATE_JSON = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage1_gate.json"
BASELINE_TABLE = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)

#: The two arms section 8b reduced the ladder to. A3_MINIMAL is the candidate;
#: A1 is the control that makes a gain from A3_MINIMAL attributable.
ARMS = ("A1", "A3_MINIMAL")
CANDIDATE_ARM = "A3_MINIMAL"
CONTROL_ARM = "A1"

#: The rungs the arms are compared against. Both are REUSED: nothing already
#: fit is refit to produce a comparison row.
INCUMBENT = "S3"
NATIVE = "S4"

#: Percentage points. Section 12's, and not adjustable here.
BLOCKER_BOUND_PP = -0.50
CONTROL_BOUND_PP = -0.50

#: The seed M2D Stage 1 runs. Section 15's extra seeds are not authorised.
SEED = 0

PASS = "PASS"
FAIL = "FAIL"
RESOLVABLE = "RESOLVABLE"

STOP = "STOP_S4_DEVELOPMENT"
ADVANCE = "ADVANCE_TARGETED_M2D"

#: Not a verdict. The two verdicts above are scientific claims about eight
#: fits; this is what the gate returns when it does not have eight fits. M2D
#: Stage 0 learned this once already -- a stop resting on an unmeasured
#: condition is not a stop -- and the same asymmetry applies here, harder: an
#: ADVANCE needs one arm to clear everything, but a STOP claims both arms were
#: tried and both failed. With fits missing, neither claim has been earned.
NOT_MEASURED = "STAGE_1_NOT_MEASURED"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION.read_text(encoding="utf-8"))


def declared_cells(config: dict[str, Any]) -> dict[str, list[str]]:
    cells = config["stage_1"]["cells"]
    return {
        "blocker": list(cells["mandatory_blockers"]),
        "control": list(cells["controls"]),
    }


def baseline_rows() -> list[dict[str, Any]]:
    return json.loads(BASELINE_TABLE.read_text(encoding="utf-8"))["rows"]


def baseline(rows: list[dict[str, Any]], cell: str, rung: str, seed: int = SEED):
    dataset, regime = cell.split("/")
    for row in rows:
        if (row["dataset"], row["regime"], row["rung"], row["seed"]) == (
            dataset,
            regime,
            rung,
            seed,
        ):
            return row
    raise KeyError(f"{cell} {rung} seed {seed} is not in M2B's baseline table")


def seed_spread_pp(rows: list[dict[str, Any]], cell: str, metric: str = "recall@5") -> float:
    """The measured seed-to-seed spread of NATIVE S4 on this cell, in pp.

    The resolvable band. Not a tolerance chosen here: it is how much this
    cell's own recall@5 already moves between M2B's three filed seeds with
    nothing but the seed changed. A shortfall smaller than that is a
    difference seed 0 cannot settle, and saying so is more honest than either
    passing it or failing it.
    """

    dataset, regime = cell.split("/")
    values = [
        row[metric]
        for row in rows
        if (row["dataset"], row["regime"], row["rung"]) == (dataset, regime, NATIVE)
    ]
    if len(values) < 2:
        raise ValueError(f"{cell} has fewer than two seeds of {NATIVE} in the table")
    return (max(values) - min(values)) * 100.0


def load_results(root: Path = RESULT_ROOT) -> dict[tuple[str, str], dict[str, Any]]:
    """The fetched Stage-1 artifacts, keyed by (cell, arm).

    A file that is not a completed M2D Stage-1 arm fit is refused rather than
    skipped: a gate that silently ignored an unreadable fit would return a
    verdict over fewer fits than it claims.
    """

    results: dict[tuple[str, str], dict[str, Any]] = {}
    if root is None or not Path(root).is_dir():
        return results
    for path in sorted(Path(root).glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope.get("payload", envelope)
        if payload.get("status") != "M2D_STAGE1_ARM_COMPLETE":
            raise ValueError(f"{path} is not a completed M2D Stage-1 arm fit")
        if payload.get("test_split_read"):
            raise ValueError(f"{path} claims to have read the test split")
        if payload.get("arm") not in ARMS:
            raise ValueError(f"{path} carries arm {payload.get('arm')!r}, not one of {ARMS}")
        results[(payload["cell"], payload["arm"])] = payload
    return results


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def blocker_outcome(delta_vs_s3_pp: float, band_pp: float) -> str:
    """PASS, RESOLVABLE or FAIL on one blocker cell.

    The bound is signed: beating S3 passes. The band is the cell's own seed
    spread, so RESOLVABLE means "seed 0 cannot settle this", never "close
    enough".
    """

    if delta_vs_s3_pp >= BLOCKER_BOUND_PP:
        return PASS
    if delta_vs_s3_pp >= BLOCKER_BOUND_PP - band_pp:
        return RESOLVABLE
    return FAIL


def control_outcome(delta_vs_s4_pp: float) -> str:
    return PASS if delta_vs_s4_pp >= CONTROL_BOUND_PP else FAIL


def evaluate_arm(
    arm: str,
    results: dict[tuple[str, str], dict[str, Any]],
    rows: list[dict[str, Any]],
    cells: dict[str, list[str]],
) -> dict[str, Any]:
    """One arm, against section 12 and section 13. Nothing is chosen here."""

    missing = [
        cell for cell in cells["blocker"] + cells["control"] if (cell, arm) not in results
    ]
    if missing:
        return {
            "arm": arm,
            "measured": False,
            "cells_not_measured": missing,
            "passes": False,
            "why": "not every declared cell produced a fit; the arm is not judged",
        }

    blockers: dict[str, Any] = {}
    for cell in cells["blocker"]:
        payload = results[(cell, arm)]
        incumbent = baseline(rows, cell, INCUMBENT)
        native = baseline(rows, cell, NATIVE)
        band = seed_spread_pp(rows, cell)
        delta_s3 = (payload["metrics"]["recall@5"] - incumbent["recall@5"]) * 100.0
        delta_s4 = (payload["metrics"]["recall@5"] - native["recall@5"]) * 100.0
        rank1_s4 = (payload["metrics"]["recall@1"] - native["recall@1"]) * 100.0
        blockers[cell] = {
            "recall_at_5_vs_s3_pp": delta_s3,
            "recall_at_5_vs_s4_pp": delta_s4,
            "recall_at_1_vs_s4_pp": rank1_s4,
            "recall_at_1_vs_s3_pp": (
                payload["metrics"]["recall@1"] - incumbent["recall@1"]
            ) * 100.0,
            "mrr_vs_s4_pp": (payload["metrics"]["mrr"] - native["mrr"]) * 100.0,
            "mrr_vs_s3_pp": (payload["metrics"]["mrr"] - incumbent["mrr"]) * 100.0,
            "seed_spread_band_pp": band,
            "outcome": blocker_outcome(delta_s3, band),
            "improves_rank_1_over_native_s4": rank1_s4 > 0.0,
        }

    controls: dict[str, Any] = {}
    for cell in cells["control"]:
        payload = results[(cell, arm)]
        native = baseline(rows, cell, NATIVE)
        delta_s4 = (payload["metrics"]["recall@5"] - native["recall@5"]) * 100.0
        controls[cell] = {
            "recall_at_5_vs_s4_pp": delta_s4,
            "recall_at_1_vs_s4_pp": (
                payload["metrics"]["recall@1"] - native["recall@1"]
            ) * 100.0,
            "mrr_vs_s4_pp": (payload["metrics"]["mrr"] - native["mrr"]) * 100.0,
            "outcome": control_outcome(delta_s4),
        }

    systems = {}
    for cell in cells["blocker"] + cells["control"]:
        payload = results[(cell, arm)]
        incumbent = baseline(rows, cell, INCUMBENT)
        native = baseline(rows, cell, NATIVE)
        systems[cell] = {
            "arm_uncached_p95_ms": payload["systems"]["uncached_p95_ms"],
            "s3_uncached_p95_ms": incumbent["uncached_p95_ms"],
            "s4_uncached_p95_ms": native["uncached_p95_ms"],
            "below_s3": payload["systems"]["uncached_p95_ms"] < incumbent["uncached_p95_ms"],
            "increase_over_native_s4_pct": (
                payload["systems"]["uncached_p95_ms"] / native["uncached_p95_ms"] - 1.0
            ) * 100.0,
        }

    blocker_outcomes = [row["outcome"] for row in blockers.values()]
    rank_1_repaired = all(row["improves_rank_1_over_native_s4"] for row in blockers.values())
    controls_hold = all(row["outcome"] == PASS for row in controls.values())
    systems_hold = all(row["below_s3"] for row in systems.values())

    passes = (
        all(outcome == PASS for outcome in blocker_outcomes)
        and controls_hold
        and systems_hold
        and rank_1_repaired
    )
    return {
        "arm": arm,
        "measured": True,
        "blockers": blockers,
        "controls": controls,
        "systems": systems,
        "parameters": {
            cell: results[(cell, arm)]["parameters"] for cell in cells["blocker"]
        },
        "integration": {
            cell: results[(cell, arm)]["integration"] for cell in cells["blocker"]
        },
        "blockers_pass": all(outcome == PASS for outcome in blocker_outcomes),
        "blockers_resolvable": RESOLVABLE in blocker_outcomes and FAIL not in blocker_outcomes,
        "controls_hold": controls_hold,
        "systems_hold": systems_hold,
        "rank_1_repaired_on_both_blockers": rank_1_repaired,
        "passes": passes,
        "why": _why(blocker_outcomes, controls_hold, systems_hold, rank_1_repaired),
    }


def _why(
    blocker_outcomes: list[str], controls_hold: bool, systems_hold: bool, rank_1: bool
) -> str:
    reasons = []
    if FAIL in blocker_outcomes:
        reasons.append("a blocker misses the 0.50pp bound by more than its own seed spread")
    elif RESOLVABLE in blocker_outcomes:
        reasons.append(
            "a blocker misses the bound by less than its own seed spread, so seed 0 "
            "cannot settle it"
        )
    if not controls_hold:
        reasons.append("a control regresses recall@5 by more than 0.50pp against native S4")
    if not systems_hold:
        reasons.append("uncached p95 is not below S3's, which removes the reason to repair S4")
    if not rank_1:
        reasons.append(
            "recall@1 does not improve against native S4 on both blockers, and section 12 "
            "says a challenger that fixes recall@5 and leaves rank 1 where it is has not "
            "repaired this failure"
        )
    return "; ".join(reasons) if reasons else "every filed condition holds"


def prospective_case(
    control: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any] | None:
    """Which of section 8b's four readings this result is, chosen by pattern.

    The readings were filed before the arms ran. This assigns one; it does not
    write one -- and it assigns none at all unless both arms were measured,
    because every one of the four readings is a statement about what two
    measured arms did.
    """

    if not (control.get("measured") and candidate.get("measured")):
        return None
    cases = config["stage_1_amendment"]["causal_interpretation_filed_in_advance"]
    a1, a3 = control.get("passes", False), candidate.get("passes", False)
    blockers_moved = control.get("blockers_pass", False) or candidate.get(
        "blockers_pass", False
    )
    controls_broken = not control.get("controls_hold", True) or not candidate.get(
        "controls_hold", True
    )

    if blockers_moved and controls_broken:
        name = "case_4"
    elif a1 and a3:
        name = "case_1"
    elif a3 and not a1:
        name = "case_2"
    elif not a1 and not a3:
        name = "case_3"
    else:
        # A1 passes and A3-MINIMAL does not. Section 8b's case 1 covers "A1 is
        # approximately A3-MINIMAL"; this is its stronger form, and it points
        # the same way: prefer the simpler arm.
        name = "case_1"
    return {"case": name, **cases[name]}


def evaluate(
    results: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cells = declared_cells(config)
    arms = {arm: evaluate_arm(arm, results, rows, cells) for arm in ARMS}
    passed = [arm for arm, row in arms.items() if row["passes"]]
    candidate = arms[CANDIDATE_ARM]
    control = arms[CONTROL_ARM]

    expected = {(cell, arm) for arm in ARMS for cell in cells["blocker"] + cells["control"]}
    absent = sorted(f"{cell}/{arm}" for cell, arm in expected - set(results))
    if absent:
        verdict = NOT_MEASURED
    else:
        verdict = ADVANCE if passed else STOP
    return {
        "status": "M2D_STAGE1_GATE_EVALUATED",
        "phase": config["phase"],
        "stage": "stage_1",
        "gate_source": "scripts/m2d_stage1_gate.py",
        "gate_filed_before_any_stage_1_fit": True,
        "seed": SEED,
        "cells": cells,
        "comparison_basis": (
            "Blockers against reused S3, controls against reused native S4, both from "
            "M2B's immutable baseline table on the same held-out portion. Nothing "
            "already fit was refit to produce a comparison row."
        ),
        "bounds_pp": {
            "blocker_vs_s3_recall_at_5": BLOCKER_BOUND_PP,
            "control_vs_s4_recall_at_5": CONTROL_BOUND_PP,
            "resolvable_band": "the cell's own measured S4 seed spread; not a typed tolerance",
        },
        "arms": arms,
        "fits_expected": len(expected),
        "fits_measured": len(expected) - len(absent),
        "fits_absent": absent,
        "arms_tried": len(ARMS),
        "arms_passed": len(passed),
        "passing_arms": passed,
        "selection_note": (
            f"{len(passed)} of {len(ARMS)} arms pass. A pass by one of {len(ARMS)} is a "
            f"selection over {len(ARMS)} and is reported as such."
        ),
        "prospective_case": prospective_case(control, candidate, config),
        "verdict": verdict,
        "why": (
            (
                f"{len(absent)} of {len(expected)} declared fits are not present, so "
                "neither verdict has been earned. A STOP here would claim both arms were "
                "tried and both failed, which is not what the artifacts say."
            )
            if absent
            else (
                "At least one arm clears every filed condition."
                if passed
                else "No arm clears every filed condition. " + candidate.get("why", "")
            )
        ),
        "extra_seeds": _extra_seeds(arms),
        "then": "STOP_FOR_REVIEW",
    }


def _extra_seeds(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Section 15 authorises seeds 1 and 2 only where they could change this.

    RESOLVABLE is the only state in which they could, which is why it is a
    separate outcome rather than a rounded PASS or a rounded FAIL.
    """

    could_change = sorted(
        arm for arm, row in arms.items() if row.get("blockers_resolvable")
    )
    return {
        "could_change_the_decision_for": could_change,
        "authorised_by_this_gate": False,
        "rule": (
            "Extra seeds are proposed only for an arm whose blocker shortfall is inside "
            "that cell's own seed spread. Everything else is already settled by seed 0 "
            "under the filed rule, and a seed that cannot change the decision is not run."
        ),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(gate: dict[str, Any]) -> str:
    lines = [
        "# M2D Stage 1 — gate",
        "",
        (
            f"Applied by `{gate['gate_source']}`, committed before any Stage-1 fit "
            f"existed. Seed {gate['seed']} only. {gate['comparison_basis']}"
        ),
        "",
        f"**Verdict: {gate['verdict']}**",
        "",
        gate["why"] if gate["fits_absent"] else gate["selection_note"],
        "",
    ]
    if gate["fits_absent"]:
        lines += [
            f"Measured {gate['fits_measured']} of {gate['fits_expected']} declared fits. "
            "Absent: " + ", ".join(f"`{name}`" for name in gate["fits_absent"]) + ".",
            "",
        ]
    for arm, row in gate["arms"].items():
        lines += [f"## {arm}", ""]
        if not row["measured"]:
            lines += [f"Not judged: {row['why']}.", ""]
            continue
        lines += [
            f"Passes: **{'yes' if row['passes'] else 'no'}**. {row['why']}.",
            "",
            "| blocker | R@5 vs S3 | band | outcome | R@1 vs S4 | MRR vs S4 |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for cell, blocker in row["blockers"].items():
            lines.append(
                f"| {cell} | {blocker['recall_at_5_vs_s3_pp']:+.3f} | "
                f"{blocker['seed_spread_band_pp']:.3f} | {blocker['outcome']} | "
                f"{blocker['recall_at_1_vs_s4_pp']:+.3f} | {blocker['mrr_vs_s4_pp']:+.3f} |"
            )
        lines += [
            "",
            "| control | R@5 vs S4 | outcome | R@1 vs S4 | MRR vs S4 |",
            "|---|---:|---|---:|---:|",
        ]
        for cell, ctrl in row["controls"].items():
            lines.append(
                f"| {cell} | {ctrl['recall_at_5_vs_s4_pp']:+.3f} | {ctrl['outcome']} | "
                f"{ctrl['recall_at_1_vs_s4_pp']:+.3f} | {ctrl['mrr_vs_s4_pp']:+.3f} |"
            )
        lines += [
            "",
            "| cell | arm p95 ms | S4 p95 | S3 p95 | below S3 | vs native S4 |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for cell, system in row["systems"].items():
            lines.append(
                f"| {cell} | {system['arm_uncached_p95_ms']:.3f} | "
                f"{system['s4_uncached_p95_ms']:.3f} | {system['s3_uncached_p95_ms']:.3f} | "
                f"{'yes' if system['below_s3'] else 'NO'} | "
                f"{system['increase_over_native_s4_pct']:+.1f}% |"
            )
        lines += [
            "",
            "| blocker | S4 top-1 errors | corrected | newly broken | net |",
            "|---|---:|---:|---:|---:|",
        ]
        for cell, integration in row["integration"].items():
            lines.append(
                f"| {cell} | {integration['s4_top1_errors']:,} | "
                f"{integration['corrected']:,} | {integration['newly_broken']:,} | "
                f"{integration['net_top1_corrections']:+,} |"
            )
        lines += [""]

    case = gate["prospective_case"]
    lines += ["## The reading, filed in advance", ""]
    if case is None:
        lines += [
            (
                "None assigned. Every one of section 8b's four readings is a statement "
                "about what two measured arms did, and the arms are not both measured."
            ),
            "",
        ]
    else:
        lines += [f"**{case['case']}** — {case['pattern']}.", "", case["conclusion"], ""]
        if case.get("action"):
            lines += [f"Action: {case['action']}", ""]
    lines += [
        "## Extra seeds",
        "",
        gate["extra_seeds"]["rule"],
        "",
        (
            "Could change the decision for: "
            + (
                ", ".join(gate["extra_seeds"]["could_change_the_decision_for"])
                or "no arm"
            )
            + f". Authorised by this gate: {gate['extra_seeds']['authorised_by_this_gate']}."
        ),
        "",
        f"## {gate['then']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULT_ROOT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    gate = evaluate(load_results(args.results), declaration(), baseline_rows())
    print(render(gate))
    if args.print_only:
        return 0
    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATE_JSON.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"wrote {GATE_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
