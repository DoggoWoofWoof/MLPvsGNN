#!/usr/bin/env python
"""Apply the filed M0A advancement rule to the probe outputs, mechanically.

The thresholds are read from configs/m0a_probe.yaml rather than written here,
so the decision cannot be reached by choosing a number after seeing the result.
The rule has two halves and both are reported: a set of gates that must all
pass, and two mutually exclusive outcomes of which exactly one may hold.

This script decides nothing else. It does not rank the expansion methods, does
not name a winner among edge families, and does not licence M0B on its own --
advancement is not automatic, and a decision is a person's to take on the
evidence this prints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "m0a_probe.yaml"
POINTS = 100.0
MOVEMENT_METRICS = ("any_gold_at_pool", "recall_ceiling@5")


def _points(regime: dict[str, Any], baseline: dict[str, Any], metric: str) -> float:
    """The movement of one metric against R1, in points."""

    return (regime["headroom"][metric] - baseline["headroom"][metric]) * POINTS


def _cells(probe: dict[str, Any], rule: str) -> dict[str, dict[str, Any]]:
    return {
        key: cell
        for key, cell in probe["regimes"].items()
        if key.startswith("R3/") and key.endswith(f"/{rule}")
    }


def dataset_movement(probe: dict[str, Any]) -> dict[str, Any]:
    """Every R3 cell's movement against this dataset's own R1."""

    baseline = probe["regimes"]["R1"]
    rows = {}
    for rule in ("matched", "additive"):
        for key, cell in _cells(probe, rule).items():
            rows[key] = {
                metric: _points(cell, baseline, metric) for metric in MOVEMENT_METRICS
            }
            rows[key]["admitted_per_query_mean"] = cell["admitted_per_query"]["mean"]
    return rows


def gates(probe: dict[str, Any]) -> dict[str, bool]:
    """The conditions that must all hold before either outcome may be read."""

    matched = _cells(probe, "matched")
    return {
        "every_invariant_passed": all(
            value is True for value in probe["invariants"].values()
        ),
        "matched_budget_held_for_every_query": all(
            cell["candidate_count"] == probe["regimes"]["R1"]["candidate_count"]
            for cell in matched.values()
        ),
        "latency_percentiles_recorded": all(
            {"p50", "p95", "p99"} <= set(cell.get(field, {}))
            for cell in matched.values()
            for field in ("expansion_latency_ms", "context_build_latency_ms")
        ),
        "scan_cap_reported": all(
            "neighbour_scan_cap_fired_queries" in cell for cell in matched.values()
        ),
        "nothing_was_trained": probe["trained_anything"] is False,
        "test_split_unread": probe["test_split_read"] is False,
    }


def _thresholds(config: dict[str, Any]) -> dict[str, float]:
    """The filed numbers, read as numbers.

    They are declared beside the sentences that state them, and a test asserts
    the two agree. Parsing the prose here instead would put a second, weaker
    copy of the rule in the analysis, which is exactly the drift the pairing is
    meant to prevent.
    """

    rule = config["advancement_to_m0b"]["and_one_of"]
    return {
        "movement_points": float(rule["movement_points"]),
        "decisive_null_matched_points": float(rule["decisive_null_matched_points"]),
        "decisive_null_additive_points": float(rule["decisive_null_additive_points"]),
    }


def outcomes(
    movements: dict[str, dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Which of the two filed outcomes hold. Zero, one, or -- refused -- both."""

    filed = _thresholds(config)
    move_threshold = filed["movement_points"]
    null_matched = filed["decisive_null_matched_points"]
    null_additive = filed["decisive_null_additive_points"]

    moved = []
    for dataset, rows in movements.items():
        for key, row in rows.items():
            if not key.endswith("/matched"):
                continue
            for metric in MOVEMENT_METRICS:
                if abs(row[metric]) >= move_threshold:
                    moved.append({"dataset": dataset, "cell": key, "metric": metric,
                                  "points": row[metric]})

    inert = all(
        abs(row[metric]) < (null_matched if key.endswith("/matched") else null_additive)
        for rows in movements.values()
        for key, row in rows.items()
        for metric in ("recall_ceiling@5",)
    )
    return {
        "thresholds": {**filed, "read_from": "configs/m0a_probe.yaml"},
        "movement": bool(moved),
        "movement_evidence": moved,
        "decisive_null": bool(inert),
    }


def abort_rule(probe: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """The filed systems condition, applied per cell rather than per dataset.

    The declaration says a dataset is aborted before reporting when its
    expansion p95 per query exceeds its context-build p95 by more than a factor
    of four on the same container. The runner does not implement the abort, so
    it is applied here instead -- after the fact, which is worse, and recorded
    as such rather than passed over because the outcome was convenient.
    """

    factor = float(config["compute"]["abort_rule_latency_factor"])
    breaches: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for key, cell in _cells(probe, "matched").items():
        expansion = cell.get("expansion_latency_ms", {}).get("p95")
        context = cell.get("context_build_latency_ms", {}).get("p95")
        if expansion is None or context is None:
            # The rule cannot be evaluated on this cell. That is a gap in the
            # record, not a pass; the latency gate reports it separately and
            # silently skipping here would let it read as compliance.
            unreadable.append(key)
            continue
        ratio = expansion / context if context else float("inf")
        if ratio > factor:
            breaches.append({"cell": key, "ratio": ratio,
                             "expansion_p95_ms": expansion, "context_p95_ms": context})
    return {
        "factor": factor,
        "breached": bool(breaches),
        "breaches": sorted(breaches, key=lambda row: -row["ratio"]),
        "arms_that_breached": sorted({row["cell"].split("/")[2] for row in breaches}),
        "cells_where_the_rule_could_not_be_evaluated": sorted(unreadable),
    }


def arm_agreement(probe: dict[str, Any]) -> dict[str, Any]:
    """Did the two arms admit the same nodes, and was the cap ever binding?

    Equal ceilings between a method and its control are only a null about the
    method if the two admitted different sets. If the budget was loose enough
    that both took the whole neighbourhood, the comparison never happened.
    """

    rows = {}
    for key, cell in _cells(probe, "matched").items():
        _, family, method, _ = key.split("/")
        rows.setdefault(family, {})[method] = cell
    out = {}
    for family, arms in rows.items():
        directional = arms.get("L1_DIRECTIONAL")
        structural = arms.get("STRUCTURAL_NEIGHBOUR")
        if directional is None or structural is None:
            continue
        left = [set(row) for row in directional["admitted_nodes_per_query"]]
        right = [set(row) for row in structural["admitted_nodes_per_query"]]
        identical = sum(1 for a, b in zip(left, right, strict=True) if a == b)
        overlaps = [
            len(a & b) / len(a | b) if (a | b) else 1.0
            for a, b in zip(left, right, strict=True)
        ]
        out[family] = {
            "queries": len(left),
            "queries_where_the_arms_admitted_identical_sets": identical,
            "mean_jaccard": sum(overlaps) / len(overlaps) if overlaps else 1.0,
            "min_jaccard": min(overlaps) if overlaps else 1.0,
            "seeds_at_the_per_seed_cap": directional["seeds_at_the_per_seed_cap"],
            "queries_with_a_capped_seed": directional["queries_with_a_capped_seed"],
            "the_comparison_happened": bool(
                directional["queries_with_a_capped_seed"] > 0 and identical < len(left)
            ),
            "headroom_is_identical": (
                directional["headroom"] == structural["headroom"]
            ),
        }
    return out


def unconstrained_diagnostic(probe: dict[str, Any]) -> dict[str, Any]:
    """The whole one-hop frontier, uncapped. Never a matched-budget comparison."""

    baseline = probe["regimes"]["R1"]
    out = {}
    for key, cell in probe["regimes"].items():
        if not key.endswith("/UNCONSTRAINED_FRONTIER/diagnostic"):
            continue
        family = key.split("/")[1]
        out[family] = {
            metric: _points(cell, baseline, metric) for metric in MOVEMENT_METRICS
        }
        out[family]["admitted_per_query_mean"] = cell["admitted_per_query"]["mean"]
    return out


def decide(probes: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """The filed rule, applied. Not a licence to launch: advancement is manual."""

    per_dataset = {name: gates(probe) for name, probe in probes.items()}
    aborts = {name: abort_rule(probe, config) for name, probe in probes.items()}
    movements = {name: dataset_movement(probe) for name, probe in probes.items()}
    all_gates = all(all(row.values()) for row in per_dataset.values())
    result = outcomes(movements, config)

    breached = sorted(name for name, row in aborts.items() if row["breached"])

    if not all_gates:
        verdict = "GATES_FAILED"
        why = "a filed gate did not pass, so neither outcome may be read"
    elif result["movement"] and result["decisive_null"]:
        verdict = "CONTRADICTORY"
        why = (
            "movement and decisive_null both hold, which the rule does not allow; "
            "the thresholds describe overlapping regions and this needs diagnosis, "
            "not a choice between them"
        )
    elif result["movement"]:
        verdict = "MOVEMENT"
        why = config["advancement_to_m0b"]["and_one_of"]["movement"]
    elif result["decisive_null"]:
        verdict = "DECISIVE_NULL"
        why = config["advancement_to_m0b"]["and_one_of"]["decisive_null"]
    else:
        verdict = "NEITHER"
        why = config["advancement_to_m0b"]["neither_holds"]

    if breached:
        verdict = f"{verdict}_UNDER_A_BREACHED_ABORT_RULE"
        why = (
            " ".join(why.split())
            + " -- BUT the filed abort rule fired on "
            + ", ".join(breached)
            + ". The runner did not implement the abort, so those datasets were "
            "reported when the declaration says they should have been stopped "
            "before reporting. Whether the reading may be acted on is a decision "
            "about that breach, not a computation over the data."
        )

    return {
        "status": "M0A_ADVANCEMENT_RULE_APPLIED",
        "verdict": verdict,
        "why": " ".join(why.split()),
        "advancement_is_automatic": False,
        "gates": per_dataset,
        "abort_rule": aborts,
        "abort_rule_breached": bool(breached),
        "gates_all_passed": all_gates,
        "outcomes": result,
        "movement_points_against_r1": movements,
        "thresholds_were_filed_before_results": bool(
            config["advancement_to_m0b"]["thresholds_are_not_moved_after_seeing_m0a"]
        ),
        "not_part_of_the_rule": {
            "why": (
                "context for reading the verdict above. Neither block can change "
                "it, and neither was used to reach it."
            ),
            "arm_agreement": {
                name: arm_agreement(probe) for name, probe in probes.items()
            },
            "unconstrained_frontier": {
                name: unconstrained_diagnostic(probe) for name, probe in probes.items()
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "m0a_probe")
    parser.add_argument(
        "--json-output", type=Path, default=REPO_ROOT / "outputs" / "m0a_probe_analysis.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    probes = {
        name: json.loads((args.root / f"{name}.json").read_text(encoding="utf-8"))
        for name in config["datasets"]
    }
    analysis = decide(probes, config)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": analysis["verdict"], "why": analysis["why"]}, indent=2))


if __name__ == "__main__":
    main()
