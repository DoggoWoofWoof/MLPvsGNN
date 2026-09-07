"""Apply M2C's filed Stage-0 advance gate to the four cell results.

Written and committed before the results were fetched, for the reason the
declaration itself gives: a threshold moved after seeing the number it failed
is not a threshold. Nothing here decides anything -- every condition is read
out of ``configs/m2c_s4_structural_conditioning.yaml`` and every required-repair
value out of the immutable baseline table, and this script only applies them.

Two verdicts come out, and they are independent by declaration:

``STOP_STRUCTURAL_M2C`` / ``ADVANCE_STRUCTURAL_RANKING_M2C``
    whether deterministic graph direction carries enough ranking signal to
    justify training S4-STRUCT-TRANSFORM.

``ADMISSION_CLOSED`` / ``ADMISSION_REMAINS_PLAUSIBLE``
    whether directional candidate admission can, in the oracle limit, recover
    what each blocker needs. This is a statement about a ceiling, not about
    achieved recall, and the declaration is explicit that the two are not the
    same thing.

One honest complication is recorded rather than smoothed over. The gate says
"S4 + direction RRF improves recall@5 on BOTH failure cells" without naming
which of the nine residual x provenance combinations must do the improving. So
this script evaluates all nine, reports how many pass, and states the selection
in the output: a pass by one combination out of nine is a choice among nine and
has to be read as one. It does not narrow the gate to a single combination
after the fact, because that would be picking the comparison from the results.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import shim
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import shim
    sys.path.insert(0, str(REPO_ROOT / "src"))

DECLARATION = REPO_ROOT / "configs/m2c_s4_structural_conditioning.yaml"
BASELINE_TABLE = REPO_ROOT / "outputs/m2c_s4_structural_conditioning/m2b_baseline_table.json"
STAGE0_ROOT = REPO_ROOT / "outputs/m2c_s4_structural_conditioning/stage0"
OUTPUT = REPO_ROOT / "outputs/m2c_s4_structural_conditioning/stage0_gate.json"

#: The gate's own numbers, transcribed here ONLY so the transcription can be
#: checked against the declaration at run time. Every use below reads the
#: declaration; these exist to make a silent edit to either side fail loudly.
DECLARED = {
    "effectiveness_a_recall_at_5_min_pp": 0.25,
    "effectiveness_b_recall_at_1_min_pp": 2.0,
    "effectiveness_b_recall_at_5_floor_pp": -0.25,
    "protection_max_regression_pp": 0.50,
}

FUSION_ARM = "S4_plus_direction_rrf"
DIRECTION_ARM = "direction_only"


def _load(path: pathlib.Path) -> Any:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)


def _check_transcription(gate: dict[str, Any]) -> None:
    """The prose gate and the numbers this script applies must agree.

    The gate is stated in English in the declaration. A script that applies it
    has to turn that English into comparisons somewhere, and this is the seam
    where a wrong number would be invisible -- so the seam is checked rather
    than trusted.
    """

    effectiveness = gate["effectiveness_condition"]
    protection = gate["protection_condition"]
    for value, text, where in (
        (DECLARED["effectiveness_a_recall_at_5_min_pp"], "+0.25pp", effectiveness),
        (DECLARED["effectiveness_b_recall_at_1_min_pp"], "+2.0pp", effectiveness),
        (DECLARED["effectiveness_b_recall_at_5_floor_pp"], "-0.25pp", effectiveness),
        (DECLARED["protection_max_regression_pp"], "0.50pp", protection),
    ):
        if text not in where:
            raise SystemExit(
                f"the declaration no longer states {text!r}; this script applies "
                f"{value} and must not be run until the two agree"
            )
    if gate.get("filed_before_modal_results") is not True:
        raise SystemExit("the gate is not marked as filed before results")


def _cells(declaration: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    block = declaration["stage_0"]["cells"]
    failure = {cell.split("/")[0]: cell for cell in block["failure_cells"]}
    control = {
        block[key].split("/")[0]: block[key]
        for key in ("passage_r3_control", "kb_r3_control")
    }
    return failure, control


def _results(cells: dict[str, str]) -> dict[str, dict[str, Any]]:
    loaded = {}
    for dataset, cell in cells.items():
        path = STAGE0_ROOT / f"{cell.replace('/', '_')}.json"
        result = _load(path)
        if result.get("status") != "M2C_STAGE0_COMPLETE":
            raise SystemExit(f"{path} is not a complete Stage-0 result")
        if result["cell"] != cell:
            raise SystemExit(f"{path} holds {result['cell']}, expected {cell}")
        if result.get("trained_anything") or result.get("test_split_read"):
            raise SystemExit(f"{path} claims training or a test read; Stage 0 does neither")
        loaded[dataset] = result
    return loaded


def _row(result: dict[str, Any], arm: str, residual: str, provenance: str) -> dict[str, Any]:
    for row in result["matrix"]:
        if (row["arm"], row["residual"], row["provenance"]) == (arm, residual, provenance):
            return row
    raise SystemExit(f"{result['cell']} has no {arm}|{residual}|{provenance} row")


def _combinations(result: dict[str, Any]) -> list[tuple[str, str]]:
    return sorted(
        {
            (row["residual"], row["provenance"])
            for row in result["matrix"]
            if row["arm"] == FUSION_ARM
        }
    )


def _delta_pp(result: dict[str, Any], combination: tuple[str, str], metric: str) -> float:
    """Fusion minus the S4 reference, in percentage points, on one cell.

    Against ``s4_reference`` rather than against the S4 matrix rows: the S4 rows
    are the same numbers repeated once per combination so the matrix reads as a
    grid, and differencing a row against its own duplicate would silently give
    zero if the duplication ever broke.
    """

    residual, provenance = combination
    fused = _row(result, FUSION_ARM, residual, provenance)[metric]
    return (float(fused) - float(result["s4_reference"][metric])) * 100.0


def evaluate_effectiveness(
    failure: dict[str, dict[str, Any]], gate: dict[str, Any]
) -> dict[str, Any]:
    _ = gate  # applied through DECLARED, which _check_transcription ties to it
    combinations = _combinations(next(iter(failure.values())))
    rows = []
    for combination in combinations:
        at5 = {d: _delta_pp(r, combination, "recall@5") for d, r in failure.items()}
        at1 = {d: _delta_pp(r, combination, "recall@1") for d, r in failure.items()}
        condition_a = all(v > 0 for v in at5.values()) and any(
            v >= DECLARED["effectiveness_a_recall_at_5_min_pp"] for v in at5.values()
        )
        condition_b = all(
            v >= DECLARED["effectiveness_b_recall_at_1_min_pp"] for v in at1.values()
        ) and all(v >= DECLARED["effectiveness_b_recall_at_5_floor_pp"] for v in at5.values())
        rows.append(
            {
                "residual": combination[0],
                "provenance": combination[1],
                "delta_recall_at_5_pp": at5,
                "delta_recall_at_1_pp": at1,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "passes": condition_a or condition_b,
            }
        )
    passing = [r for r in rows if r["passes"]]
    return {
        "combinations_evaluated": len(rows),
        "combinations_passing": len(passing),
        "passed": bool(passing),
        "selection_note": (
            "The gate does not name which residual x provenance combination must "
            f"improve. All {len(rows)} were evaluated; {len(passing)} passed. A pass "
            "by one of many is a selection over many and must be read as one."
        ),
        "by_combination": rows,
    }


def evaluate_protection(
    control: dict[str, dict[str, Any]], effectiveness: dict[str, Any]
) -> dict[str, Any]:
    """No control cell may lose more than 0.50pp recall@5.

    Checked on every combination, not only the ones that passed effectiveness,
    because a mechanism that damages the controls everywhere it is applied is
    worth knowing about even when it never helps.
    """

    rows = []
    for entry in effectiveness["by_combination"]:
        combination = (entry["residual"], entry["provenance"])
        regressions = {
            dataset: -_delta_pp(result, combination, "recall@5")
            for dataset, result in control.items()
        }
        rows.append(
            {
                "residual": combination[0],
                "provenance": combination[1],
                "recall_at_5_regression_pp": regressions,
                "protected": all(
                    v <= DECLARED["protection_max_regression_pp"] for v in regressions.values()
                ),
                "effectiveness_passed": entry["passes"],
            }
        )
    protected_and_effective = [r for r in rows if r["protected"] and r["effectiveness_passed"]]
    return {
        "passed": bool(protected_and_effective),
        "combinations_protected": sum(1 for r in rows if r["protected"]),
        "combinations_protected_and_effective": len(protected_and_effective),
        "by_combination": rows,
    }


def evaluate_residual_distinctness(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Mechanistic condition 1: R2 must not be R1 wearing a new name.

    Two independent readings, because they can disagree and the disagreement is
    informative: whether the two residuals produce different RANKINGS (the
    matrix rows), and whether they admit different NODES (the +64 diagnostic's
    Jaccard). A pair that ranks identically but admits differently has not done
    nothing; a pair identical on both has.
    """

    ranking = {}
    admission = {}
    for dataset, result in results.items():
        differences = []
        for arm in (DIRECTION_ARM, FUSION_ARM):
            for provenance in sorted({row["provenance"] for row in result["matrix"]}):
                legacy = _row(result, arm, "R1_LEGACY_DIRECTIONAL", provenance)
                subspace = _row(result, arm, "R2_SEED_SUBSPACE", provenance)
                differences.append(
                    max(
                        abs(float(legacy[metric]) - float(subspace[metric]))
                        for metric in ("recall@1", "recall@5", "recall@20", "mrr")
                    )
                )
        ranking[dataset] = {
            "max_absolute_metric_difference": max(differences),
            "identical": max(differences) == 0.0,
        }
        overlap = result["admission_diagnostic"]["pairwise_overlap"].get(
            "legacy_PF64|seed_subspace_PF64"
        )
        admission[dataset] = overlap
    identical_everywhere = all(entry["identical"] for entry in ranking.values())
    return {
        "passed": not identical_everywhere,
        "ranking": ranking,
        "admitted_set_overlap": admission,
        "reading": (
            "identical on every cell, arm and provenance -- the new variable did "
            "nothing and any effect belongs to the legacy mechanism"
            if identical_everywhere
            else "the two residuals are behaviourally distinct somewhere"
        ),
    }


def evaluate_margin(failure: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Mechanistic condition 2: a meaningfully positive directional margin.

    The declaration says "meaningfully positive" and does not quantify it. This
    script therefore applies the WEAKEST defensible reading -- the direction
    prefers the relevant candidate over S4's top mistake more often than not,
    and on average by a positive amount -- and reports the numbers beside the
    verdict. A condition that fails the weakest reading fails under every
    stricter one too, so no threshold choice is doing any work in that case; a
    condition that passes only the weakest one is flagged here as depending on a
    number the declaration never filed, and belongs to review rather than to
    this script.

    Margins are read from ``measurable`` comparisons only. Where coverage is
    low, most comparisons have an uncovered side and there is no margin to
    average; the fraction of the error population that is measurable at all is
    carried through, because a mechanism with a fine margin on 3% of the errors
    is not a mechanism that repairs the cell.
    """

    rows = []
    for dataset, result in failure.items():
        for key, block in sorted(result["error_conditioned"].items()):
            rows.append(
                {
                    "cell": result["cell"],
                    "dataset": dataset,
                    "residual_and_provenance": key,
                    "error_population": block["queries"],
                    "measurable": block["measurable"],
                    "fraction_measurable": block["fraction_of_the_error_population_measurable"],
                    "fraction_positive": block["fraction_positive"],
                    "mean_margin": block["mean_margin"],
                    "median_margin": block["median_margin"],
                    "weakest_reading_positive": bool(
                        block["measurable"]
                        and block["fraction_positive"] is not None
                        and block["fraction_positive"] > 0.5
                        and block["mean_margin"] is not None
                        and block["mean_margin"] > 0.0
                    ),
                }
            )
    positive_cells = sorted({r["cell"] for r in rows if r["weakest_reading_positive"]})
    return {
        "passed": bool(positive_cells),
        "threshold_status": (
            "PASSES_ONLY_THE_WEAKEST_READING -- 'meaningfully positive' was never "
            "quantified in the declaration, so this pass depends on a number that "
            "was not filed and is referred to review"
            if positive_cells
            else "FAILS_THE_WEAKEST_READING -- no stricter reading could pass"
        ),
        "failure_cells_with_a_positive_margin": positive_cells,
        "by_combination": rows,
    }


def evaluate_admission(
    failure: dict[str, dict[str, Any]], declaration: dict[str, Any], table: dict[str, Any]
) -> dict[str, Any]:
    """Per-cell oracle feasibility, against the required repair, not the deficit.

    ``required_repair_to_guard_pp`` comes from the immutable baseline table and
    is cross-checked against the declaration's own transcription of it. It is
    the deficit LESS M2B's per-cell tolerance -- neither the deficit nor the
    tolerance -- and the whole feasibility question turns on using the right one.
    """

    required_declared = declaration["stage_0"]["admission_feasibility"]["required_values"]
    required = {}
    for row in table["multi_seed_margins"]:
        key = f"{row['dataset']}_{row['regime']}"
        if key in required_declared:
            value = float(row["required_repair_to_guard_pp"])
            if f"{value:.4f}pp" != required_declared[key]:
                raise SystemExit(
                    f"{key}: the baseline table says {value:.4f}pp and the declaration "
                    f"says {required_declared[key]}; they must agree before either is used"
                )
            required[row["dataset"]] = value

    cells = []
    for dataset, result in failure.items():
        diagnostic = result["admission_diagnostic"]
        deltas = diagnostic["delta_recall_ceiling_at_5_pp_versus_a64"]
        best_arm = max(
            (arm for arm in deltas if arm != "A64"), key=lambda arm: deltas[arm]
        )
        best = float(deltas[best_arm])
        need = required[dataset]
        cells.append(
            {
                "cell": result["cell"],
                "required_repair_to_guard_pp": need,
                "arms": deltas,
                "best_directional_arm": best_arm,
                "best_delta_recall_ceiling_at_5_pp": best,
                "pairwise_overlap": diagnostic["pairwise_overlap"],
                "unique_relevant_admissions": diagnostic["unique_relevant_admissions"],
                "recall_ceiling_at_5": diagnostic["recall_ceiling_at_5"],
                "verdict": "ADMISSION_REMAINS_PLAUSIBLE" if best >= need else "ADMISSION_CLOSED",
            }
        )
    plausible = [c for c in cells if c["verdict"] == "ADMISSION_REMAINS_PLAUSIBLE"]
    return {
        "verdict": "ADMISSION_REMAINS_PLAUSIBLE" if plausible else "ADMISSION_CLOSED",
        "rule": declaration["stage_0"]["admission_feasibility"]["rule"],
        "roll_up": (
            "The declaration files this verdict per cell. The phase-level verdict "
            "is ADMISSION_CLOSED only when every failure cell is closed, because "
            "one cell where the oracle can still reach the bar leaves the mechanism "
            "open even if the other cannot."
        ),
        "oracle_headroom_is_not_achieved_recall": declaration["stage_0"][
            "admission_feasibility"
        ]["do_not_equate_oracle_headroom_with_achieved_recall"],
        "by_cell": cells,
    }


def build() -> dict[str, Any]:
    declaration = _load(DECLARATION)
    table = _load(BASELINE_TABLE)
    gate = declaration["stage_0"]["advance_gate"]
    _check_transcription(gate)

    failure_cells, control_cells = _cells(declaration)
    failure = _results(failure_cells)
    control = _results(control_cells)

    effectiveness = evaluate_effectiveness(failure, gate)
    protection = evaluate_protection(control, effectiveness)
    distinctness = evaluate_residual_distinctness({**failure, **control})
    margin = evaluate_margin(failure)
    admission = evaluate_admission(failure, declaration, table)

    conditions = {
        "effectiveness": effectiveness["passed"],
        "protection": protection["passed"],
        "mechanistic_1_residuals_are_distinct": distinctness["passed"],
        "mechanistic_2_margin_is_positive": margin["passed"],
    }
    advance = all(conditions.values())
    return {
        "status": "M2C_STAGE0_GATE_EVALUATED",
        "declaration": "configs/m2c_s4_structural_conditioning.yaml",
        "gate_was_filed_before_results": True,
        "thresholds_were_not_adjusted": gate["thresholds_are_not_adjustable"],
        "cells": {
            "failure": sorted(failure_cells.values()),
            "control": sorted(control_cells.values()),
        },
        "source_commits": {
            result["cell"]: result.get("source_commit")
            for result in sorted({**failure, **control}.values(), key=lambda r: r["cell"])
        },
        "conditions": conditions,
        "all_conditions_required": gate["all_conditions_required"],
        "ranking_verdict": "ADVANCE_STRUCTURAL_RANKING_M2C" if advance else "STOP_STRUCTURAL_M2C",
        "if_the_gate_fails": gate["if_the_gate_fails"],
        "admission_verdict": admission["verdict"],
        "verdicts_are_independent": declaration["stop_condition"]["verdicts_are_independent"][
            "why_two"
        ],
        "effectiveness": effectiveness,
        "protection": protection,
        "mechanistic_1_residual_distinctness": distinctness,
        "mechanistic_2_error_conditioned_margin": margin,
        "admission_feasibility": admission,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    args = parser.parse_args()

    verdict = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    headline = ("conditions", "ranking_verdict", "admission_verdict")
    print(json.dumps({key: verdict[key] for key in headline}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
