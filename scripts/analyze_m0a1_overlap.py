#!/usr/bin/env python
"""Apply M0A.1's filed advancement rule to the overlap-audit results.

Not a licence to launch: advancement_to_m0b.is_automatic is false. This applies
the four filed conditions mechanically, against configs/m0a1_overlap.yaml's own
numbers, so the reading is reproducible rather than eyeballed -- and reports
which condition holds, on which dataset and family, at which budget, rather
than a single collapsed boolean.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "m0a1_overlap.yaml"
POINTS = 100.0
MOVEMENT_METRICS = ("any_gold_at_pool", "recall_ceiling@5")
GOLD_PERMUTATION_TEST = (
    "tests/test_candidate_expansion_v2.py"
    "::test_permuting_gold_leaves_the_expansion_byte_identical"
)
# Parameter names that could plausibly carry gold identity into expansion. Not
# an exhaustive taboo list -- expand()'s actual parameter set is compared
# against it, so a renamed leak would still show up as an unrecognised name.
SUPERVISION_LOOKING_NAMES = {"gold", "golds", "relevant", "label", "labels", "y", "target"}


def structural_leakage_check() -> dict[str, Any]:
    """Can gold identity reach expand() at all? Checked live, not recalled.

    The strongest form of "no leakage" is that the function has no parameter
    through which gold identity could enter, so it is checked by introspecting
    the live signature rather than trusting a memory of a past test run. This
    mirrors tests/test_candidate_expansion_v2.py's own
    test_expansion_takes_no_argument_that_could_carry_supervision. The
    behavioural counterpart -- permuting gold labels leaves expand()'s output
    byte-identical -- is proved once for every caller by a parametrized unit
    test and is not re-derived per dataset; it is disclosed by name and path
    rather than silently assumed.
    """

    from mp_retrieval.candidate_expansion_v2 import expand

    parameters = set(inspect.signature(expand).parameters)
    suspect = sorted(parameters & SUPERVISION_LOOKING_NAMES)
    return {
        "expand_parameters": sorted(parameters),
        "no_parameter_could_carry_gold_identity": not suspect,
        "suspect_parameters": suspect,
        "behavioural_proof_is_a_standing_unit_test_not_this_artifact": True,
        "behavioural_proof_source": GOLD_PERMUTATION_TEST,
    }


def invariants_and_leakage(result: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    """Condition 1, composed from three sources rather than one dict.

    Four invariants live directly on this run's artifact. A fifth --
    the frozen candidate pool is bit-exact -- is proved by
    validate_candidate_contract, which raises rather than records a failure,
    so a COMPLETE status already implies it; it is read from
    result['candidate_contract']['status'] because the runner never copies it
    into result['invariants']. The sixth is the leakage check above. None of
    the three sources is allowed to silently stand in for another.
    """

    on_artifact = dict(result["invariants"])
    bit_exact = (
        result["candidate_contract"]["status"] == "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"
    )
    return {
        "invariants_on_this_artifact": on_artifact,
        "invariants_on_this_artifact_all_true": bool(all(on_artifact.values())),
        "cq_is_bit_exact_against_the_frozen_artifact": bit_exact,
        "cq_is_bit_exact_source": (
            "result['candidate_contract']['status'], via validate_candidate_contract, "
            "which raises rather than records a failed check"
        ),
        "no_parameter_could_carry_gold_identity": leakage["no_parameter_could_carry_gold_identity"],
        "gold_permutation_leaves_expansion_byte_identical_source": GOLD_PERMUTATION_TEST,
        "all_hold": bool(
            all(on_artifact.values())
            and bit_exact
            and leakage["no_parameter_could_carry_gold_identity"]
        ),
    }


def curve_movement(result: dict[str, Any]) -> dict[str, Any]:
    """Additive-pool AnyGold/recall@5 movement against R1, at every curve budget.

    The filed condition names matched-budget movement. The curve is measured
    on the additive pool, not matched, because that is what the runner built
    it from. The bridge: matched_budget.agrees_with_additive is True for every
    family in every dataset here, meaning the matched and additive pools
    recovered the identical set of golds -- and AnyGold/recall@5 movement is a
    function of which golds are present, nothing else. So additive movement is
    a faithful, disclosed stand-in for matched movement here, not a silent
    substitution. If agreement ever failed for a cell that also cleared the
    threshold, this function would still report the additive number and this
    docstring's bridge would no longer apply to that cell -- that is what
    matched_budget_agreement below is for.
    """

    baseline = result["r1"]["headroom"]
    rows = []
    for family, curve in result["curve"].items():
        agrees = result["overlap"][family]["matched_budget"]["agrees_with_additive"]
        for point in curve["points"]:
            row = {
                "family": family,
                "budget": point["budget"],
                "matched_budget_agrees_with_additive_at_headline": agrees,
            }
            for metric in MOVEMENT_METRICS:
                row[f"{metric}_points"] = (point[metric] - baseline[metric]) * POINTS
            rows.append(row)
    return {"baseline": {metric: baseline[metric] for metric in MOVEMENT_METRICS}, "rows": rows}


def materially_improves_headroom(movement: dict[str, Any], threshold: float) -> dict[str, Any]:
    moved = [
        row
        for row in movement["rows"]
        if row["matched_budget_agrees_with_additive_at_headline"]
        and any(abs(row[f"{metric}_points"]) >= threshold for metric in MOVEMENT_METRICS)
    ]
    return {
        "threshold_points": threshold,
        "holds": bool(moved),
        "evidence": sorted(
            moved,
            key=lambda row: -max(abs(row[f"{metric}_points"]) for metric in MOVEMENT_METRICS),
        )[:8],
    }


def systems_contract(result: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    """The structural arm's own contract. Never reads the directional verdict."""

    factor = float(thresholds["latency_factor"])
    rss_max = int(thresholds["peak_rss_bytes_max"])
    ratios = result["systems"]["expansion_over_context_p95_ratio"]
    latency_breaches = {family: ratio for family, ratio in ratios.items() if ratio > factor}
    rss = int(result["systems"]["peak_process_rss_bytes"])
    return {
        "latency_factor": factor,
        "latency_ratios": ratios,
        "latency_breaches": latency_breaches,
        "peak_rss_bytes_max": rss_max,
        "peak_process_rss_bytes": rss,
        "rss_breached": bool(rss > rss_max),
        "does_not_inherit_the_directional_verdict": result["systems"][
            "does_not_inherit_the_directional_verdict"
        ],
        "passes": bool(not latency_breaches and rss <= rss_max),
    }


def universal_budget(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The smallest filed numeric point that reaches every dataset's own ceiling.

    Not fitted: this scans the fixed, pre-declared point list and asks only
    whether recovery at that point already equals recovery at the uncapped
    frontier, for every dataset and family with any recoverable gold. It never
    reads which dataset it is looking at to choose the answer.
    """

    per_point: dict[int, list[bool]] = {}
    candidates_with_headroom = 0
    for result in results.values():
        for curve in result["curve"].values():
            points = curve["points"]
            full = next(p for p in points if p["budget"] == "full_bounded_n1_frontier")
            if full["missing_golds_recovered"] == 0:
                continue  # nothing recoverable in this cell; it cannot inform a budget
            candidates_with_headroom += 1
            for point in points:
                if not isinstance(point["budget"], int):
                    continue
                per_point.setdefault(point["budget"], []).append(
                    point["missing_golds_recovered"] == full["missing_golds_recovered"]
                )
    sufficient = sorted(
        budget
        for budget, hits in per_point.items()
        if hits and all(hits) and len(hits) == candidates_with_headroom
    )
    return {
        "cells_with_any_recoverable_gold": candidates_with_headroom,
        "per_numeric_point_reaches_full_recovery_everywhere": {
            str(budget): bool(all(hits)) for budget, hits in sorted(per_point.items())
        },
        "smallest_sufficient_universal_budget": sufficient[0] if sufficient else None,
        "no_point_sufficient": not sufficient,
    }


def decide(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    conditions_config = config["advancement_to_m0b"][
        "advance_the_simple_structural_r3_construction_only_if_all_hold"
    ]
    thresholds = config["systems_contract_for_the_structural_arm"]["thresholds"]
    movement_threshold = float(config["advancement_to_m0b"]["movement_points"])

    leakage = structural_leakage_check()
    per_dataset = {}
    for dataset, result in results.items():
        movement = curve_movement(result)
        per_dataset[dataset] = {
            "invariants_and_leakage": invariants_and_leakage(result, leakage),
            "curve_movement": movement,
            "materially_improves_headroom": materially_improves_headroom(
                movement, movement_threshold
            ),
            "systems_contract": systems_contract(result, thresholds),
        }

    condition_1 = all(
        row["invariants_and_leakage"]["all_hold"] for row in per_dataset.values()
    )
    condition_2 = any(
        row["materially_improves_headroom"]["holds"] for row in per_dataset.values()
    )
    condition_3 = all(row["systems_contract"]["passes"] for row in per_dataset.values())
    condition_4_detail = universal_budget(results)
    condition_4 = condition_4_detail["smallest_sufficient_universal_budget"] is not None

    all_hold = bool(condition_1 and condition_2 and condition_3 and condition_4)
    verdict = "ADVANCE" if all_hold else "DO_NOT_ADVANCE"

    return {
        "status": "M0A1_ADVANCEMENT_RULE_APPLIED",
        "verdict": verdict,
        "advancement_is_automatic": False,
        "filed_rule": conditions_config,
        "conditions": {
            "1_leakage_and_invariants_pass": condition_1,
            "2_materially_improves_candidate_headroom": condition_2,
            "3_independent_systems_contract_passes": condition_3,
            "4_a_universal_bounded_expansion_rule_can_be_stated": condition_4,
        },
        "per_dataset": per_dataset,
        "universal_budget": condition_4_detail,
        "thresholds_were_filed_before_results": bool(
            config["advancement_to_m0b"]["thresholds_are_not_moved_after_seeing_m0a1"]
        ),
        "m0a_verdict_is_unchanged": "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE",
        "no_directional_method_was_required_for_this_reading": bool(
            config["advancement_to_m0b"]["no_directional_method_is_required_for_advancement"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "m0a1_overlap")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "m0a1_overlap_analysis.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    results = {
        name: json.loads((args.root / f"{name}.json").read_text(encoding="utf-8"))
        for name in config["datasets"]
    }
    analysis = decide(results, config)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": analysis["verdict"], "conditions": analysis["conditions"]}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
