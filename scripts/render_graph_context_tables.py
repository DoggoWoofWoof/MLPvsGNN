#!/usr/bin/env python
"""Lay out the graph-context pilot's arms, and name the ones the frontier kills.

``run_graph_context_pilot.py`` decides what was measured; this only decides how
it is read. The separation matters more here than it did for Phase -1, because
this pilot's whole output is a *decision* -- which arms carry into Stage C and
which are killed -- and a decision rule buried in a rendering script is a rule
nobody can audit.

The rule is the config's, restated: an arm dies if some other arm is at least as
good on structural recovery AND no more expensive at p95 construction latency,
with at least one of the two strictly better. That is Pareto domination on the
declared axes and nothing else. Three properties of it are deliberate:

*   Ties do not kill. Two arms that measure identically both survive, and the
    cheaper one is preferred later by a human, not silently here.
*   Recovery is retention, not reach. "More graph" is not the goal -- Phase -1
    already showed the whole corpus is three hops away on every dataset -- so an
    arm that recovers no structure is not rescued by touching more nodes.
*   Latency is p95, not median. An arm whose median is fast and whose tail is
    not cannot be served, and the median hides exactly that.

Domination is computed per dataset and an arm is killed only where it is
dominated on **every** dataset measured. One dataset is not a frontier, and the
two Stage B datasets were chosen because they disagree.

Seed distance is reported beside recovery but is not an axis. It answers a
different question -- whether the restored context reaches the candidates that
lost their neighbourhood -- and folding it into the frontier would let an arm
buy its way past the cost axis with reach it does not need.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.graph_context import ARMS, SEED_DISTANCE_HOPS  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "outputs" / "graph_context_pilot"
COMPLETE_STATUS = "GRAPH_CONTEXT_PILOT_COMPLETE"

#: The declared frontier axes, and the direction each improves in.
RECOVERY_AXIS = "retention_median"
COST_AXIS = "build"


def _rows(table: list[list[str]], align: str) -> str:
    header, *body = table
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(f":{'-' * len(name)}" if a == "l" else f"{'-' * len(name)}:"
                                 for name, a in zip(header, align)) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _number(value: Any, places: int = 4) -> str:
    if value is None:
        return "--"
    number = float(value)
    if number != number:  # NaN, which is a real answer here and says so
        return "n/a"
    return f"{number:,.{places}f}"


def dominated(arms: dict[str, dict], name: str) -> list[str]:
    """Arms that are at least as good on both axes and strictly better on one."""
    mine = arms[name]
    my_recovery = float(mine["recovery"][RECOVERY_AXIS])
    my_cost = float(mine["latency_ms"][COST_AXIS]["p95"])
    winners = []
    for other, block in arms.items():
        if other == name:
            continue
        recovery = float(block["recovery"][RECOVERY_AXIS])
        cost = float(block["latency_ms"][COST_AXIS]["p95"])
        if recovery != recovery or my_recovery != my_recovery:
            continue  # a NaN recovery is not evidence of domination either way
        if recovery >= my_recovery and cost <= my_cost and (
            recovery > my_recovery or cost < my_cost
        ):
            winners.append(other)
    return sorted(winners)


def verdicts(per_dataset: dict[str, dict]) -> dict[str, dict[str, Any]]:
    """Kill an arm only where every measured dataset agrees it is dominated."""
    names = [arm for arm in ARMS if all(arm in block["arms"] for block in per_dataset.values())]
    result = {}
    for arm in names:
        by_dataset = {
            dataset: dominated(block["arms"], arm) for dataset, block in per_dataset.items()
        }
        killed = all(by_dataset[dataset] for dataset in by_dataset)
        result[arm] = {
            "dominated_by": by_dataset,
            "verdict": "killed" if killed else "survives",
        }
    return result


def load(root: Path, split: str) -> dict[str, dict]:
    per_dataset = {}
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != COMPLETE_STATUS:
            print(
                f"skipping {path.name}: status {payload.get('status')!r}, not complete",
                file=sys.stderr,
            )
            continue
        if split not in payload.get("splits", {}):
            print(f"skipping {path.name}: no {split} split", file=sys.stderr)
            continue
        per_dataset[payload["dataset"]] = payload["splits"][split]
    return per_dataset


def size_table(block: dict) -> str:
    table = [["arm", "context nodes p50", "p95", "added ratio p50", "share of graph p50",
              "context edges p50"]]
    for arm, data in block["arms"].items():
        size = data["size"]
        table.append([
            arm,
            _number(size["context_nodes"]["median"], 0),
            _number(size["context_nodes"]["p95"], 0),
            _number(size["added_ratio"]["median"], 2),
            _number(size["graph_share"]["median"]),
            _number(size["context_edges"]["median"], 0),
        ])
    return _rows(table, "l" + "r" * 5)


def recovery_table(block: dict) -> str:
    table = [["arm", "retention p50", "retention mean", "boundary cut", "isolated fraction"]]
    for arm, data in block["arms"].items():
        recovery = data["recovery"]
        table.append([
            arm,
            _number(recovery["retention_median"]),
            _number(recovery["retention_mean"]),
            _number(recovery["boundary_cut"]),
            _number(recovery["isolated_fraction"]),
        ])
    return _rows(table, "l" + "r" * 4)


def reach_table(block: dict) -> str:
    hops = list(range(1, SEED_DISTANCE_HOPS + 1))
    table = [["arm", "closer than G[Cq]", *[f"reach<={hop}" for hop in hops],
              f"unreached at {SEED_DISTANCE_HOPS}"]]
    for arm, data in block["arms"].items():
        reach = data["reach"]
        table.append([
            arm,
            _number(reach["distance_improved"]),
            *[_number(reach[f"seed_reach_at_{hop}"]) for hop in hops],
            _number(reach["seed_unreachable"]),
        ])
    return _rows(table, "l" + "r" * (len(hops) + 2))


def stratified_table(block: dict) -> str:
    """Where the improvement lands, which is the question Phase -1 poses."""
    strata = list(next(iter(block["arms"].values()))
                  ["reach"]["seed_distance_improved_by_prior_induced_degree"])
    table = [["arm", *strata]]
    for arm, data in block["arms"].items():
        by = data["reach"]["seed_distance_improved_by_prior_induced_degree"]
        table.append([arm, *[_number(by[name]) for name in strata]])
    return _rows(table, "l" + "r" * len(strata))


def latency_table(block: dict) -> str:
    table = [["arm", "build p50", "build p95", "build p99", "features p95", "total p95"]]
    for arm, data in block["arms"].items():
        latency = data["latency_ms"]
        table.append([
            arm,
            _number(latency["build"]["median"], 1),
            _number(latency["build"]["p95"], 1),
            _number(latency["build"]["p99"], 1),
            _number(latency["features"]["p95"], 1),
            _number(latency["total"]["p95"], 1),
        ])
    return _rows(table, "l" + "r" * 5)


def frontier_table(decision: dict[str, dict[str, Any]], datasets: list[str]) -> str:
    table = [["arm", *datasets, "verdict"]]
    for arm, block in decision.items():
        cells = []
        for dataset in datasets:
            by = block["dominated_by"][dataset]
            cells.append(", ".join(by) if by else "on the frontier")
        table.append([arm, *cells, block["verdict"]])
    return _rows(table, "l" + "l" * len(datasets) + "l")


def render(per_dataset: dict[str, dict], split: str) -> str:
    if not per_dataset:
        return "No complete graph-context pilot results found.\n"
    datasets = sorted(per_dataset)
    decision = verdicts(per_dataset)
    parts = [
        f"Split: `{split}`. Datasets: {', '.join(f'`{name}`' for name in datasets)}.",
        "",
        "Queries measured, and how many were skipped for carrying no retrieval seed:",
        "",
    ]
    counts = [["dataset", "requested", "measured", "skipped, no seeds"]]
    for dataset in datasets:
        block = per_dataset[dataset]
        counts.append([
            dataset,
            str(block["queries_requested"]),
            str(block["queries_measured"]),
            str(block["queries_skipped_without_seeds"]),
        ])
    parts += [_rows(counts, "lrrr"), ""]

    for dataset in datasets:
        block = per_dataset[dataset]
        parts += [
            f"### {dataset}", "",
            "**Size.**", "", size_table(block), "",
            "**Structural recovery**, on Phase -1's own definitions.", "",
            recovery_table(block), "",
            "**Seed reach.** Recomputed through each arm's own context, uncapped "
            f"to {SEED_DISTANCE_HOPS} hops and divided by nothing.", "",
            reach_table(block), "",
            "**Where the improvement lands**, by the induced degree the candidate "
            "had in `G[Cq]` before any context was restored.", "",
            stratified_table(block), "",
            "**Latency, milliseconds per query per arm.**", "",
            latency_table(block), "",
        ]

    parts += [
        "### Frontier", "",
        f"Pareto domination on (`{COST_AXIS}` p95 latency, `{RECOVERY_AXIS}`). An arm "
        "is killed only where every measured dataset agrees it is dominated.", "",
        frontier_table(decision, datasets), "",
    ]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="emit the verdicts as JSON")
    args = parser.parse_args(argv)

    per_dataset = load(args.root, args.split)
    if args.json:
        print(json.dumps(verdicts(per_dataset), indent=2))
        return 0
    text = render(per_dataset, args.split)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
