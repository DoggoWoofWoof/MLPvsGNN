#!/usr/bin/env python
"""Paired five-seed and coverage diagnostics for a confirmation result."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any


METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
T_CRITICAL_95 = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def mean_sd_ci(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    t_critical = T_CRITICAL_95.get(len(values), 1.96)
    half_width = (
        t_critical * sample_std / math.sqrt(len(values))
        if len(values) > 1
        else float("nan")
    )
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": sample_std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _musique_hops(raw_path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            decomposition = item.get("metadata", {}).get("question_decomposition")
            if not isinstance(decomposition, list) or not decomposition:
                raise ValueError(f"MuSiQue item {item.get('id')} lacks question decomposition")
            result[str(item["id"])] = len(decomposition)
    return result


def _query_hops(
    result: dict[str, Any],
    manifest_path: Path,
    musique_raw: Path | None,
) -> tuple[dict[str, int], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_indices = manifest["split_indices"]["test"]
    test_ids = [str(manifest["ids"][index]) for index in test_indices]
    if musique_raw is None:
        return {}, {
            "status": "unavailable",
            "reason": "frozen and processed records contain no per-query hop label",
            "test_queries": len(test_ids),
        }
    raw_hops = _musique_hops(musique_raw)
    prefix = f"{result['dataset']}_q_"
    hops: dict[str, int] = {}
    missing: list[str] = []
    for query_id in test_ids:
        raw_id = query_id.removeprefix(prefix)
        if raw_id not in raw_hops:
            missing.append(query_id)
        else:
            hops[query_id] = raw_hops[raw_id]
    if missing:
        raise ValueError(f"Missing MuSiQue hop metadata for {len(missing)} test queries")
    return hops, {
        "status": "complete",
        "source": "raw question-decomposition length",
        "test_queries": len(hops),
    }


def _group_metrics(
    seed_payload: dict[str, Any],
    group_by_query: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for query_id, row in seed_payload["per_query"].items():
        if query_id in group_by_query:
            grouped[group_by_query[query_id]].append(row)
    return {
        str(group): {
            "queries": len(rows),
            **{metric: statistics.fmean(float(row[metric]) for row in rows) for metric in METRICS},
        }
        for group, rows in sorted(grouped.items())
    }


def _aggregate_grouped_seeds(
    model: dict[str, Any],
    group_by_query: dict[str, int],
) -> dict[str, Any]:
    by_seed = {
        seed: _group_metrics(payload, group_by_query)
        for seed, payload in model["seeds"].items()
    }
    groups = sorted({group for grouped in by_seed.values() for group in grouped}, key=int)
    aggregate: dict[str, Any] = {}
    for group in groups:
        aggregate[group] = {
            metric: mean_sd_ci(
                [float(grouped[group][metric]) for grouped in by_seed.values()]
            )
            for metric in METRICS
        }
        aggregate[group]["queries"] = next(
            grouped[group]["queries"] for grouped in by_seed.values() if group in grouped
        )
    return {"by_seed": by_seed, "aggregate": aggregate}


def _paired_group_gap(
    left: dict[str, Any],
    right: dict[str, Any],
    group_by_query: dict[str, int],
) -> dict[str, Any]:
    left_grouped = _aggregate_grouped_seeds(left, group_by_query)["by_seed"]
    right_grouped = _aggregate_grouped_seeds(right, group_by_query)["by_seed"]
    result: dict[str, Any] = {}
    for group in sorted(set.intersection(*(set(value) for value in left_grouped.values())), key=int):
        result[group] = {}
        for metric in METRICS:
            gaps = [
                float(left_grouped[seed][group][metric])
                - float(right_grouped[seed][group][metric])
                for seed in sorted(left_grouped, key=int)
            ]
            result[group][metric] = mean_sd_ci(gaps)
    r20_means = [float(result[group]["recall@20"]["mean"]) for group in sorted(result, key=int)]
    result["trend"] = {
        "groups": [int(group) for group in sorted(result, key=int) if group != "trend"],
        "r20_gap_means": r20_means,
        "monotonic_nonincreasing": all(
            right <= left for left, right in zip(r20_means, r20_means[1:])
        ),
    }
    return result


def analyze(
    result: dict[str, Any],
    manifest_path: Path,
    musique_raw: Path | None,
) -> dict[str, Any]:
    first_model = next(iter(result["models"].values()))
    first_seed = next(iter(first_model["seeds"].values()))
    answer_count = {
        query_id: int(row["gold_count"])
        for query_id, row in first_seed["per_query"].items()
    }
    hops, hop_provenance = _query_hops(result, manifest_path, musique_raw)
    max_width = max(int(width) for width in result["config"]["hidden_widths"])
    gnn_key = f"{result['frozen_best_gnn']}_h{max_width}"
    comparison_keys = [
        f"plain_mlp_h{max_width}",
        f"offset_mlp_h{max_width}",
        f"offset_mlp_k4_h{max_width}",
        result["capacity_selection_validation_only"]["plain_mlp"]["selected"],
        result["capacity_selection_validation_only"]["offset_mlp"]["selected"],
    ]
    comparison_keys = list(dict.fromkeys(comparison_keys))
    by_answer_count = {
        key: _aggregate_grouped_seeds(result["models"][key], answer_count)
        for key in [*comparison_keys, gnn_key]
    }
    answer_gaps = {
        key: _paired_group_gap(result["models"][key], result["models"][gnn_key], answer_count)
        for key in comparison_keys
    }
    by_hop = (
        {
            key: _aggregate_grouped_seeds(result["models"][key], hops)
            for key in [*comparison_keys, gnn_key]
        }
        if hops
        else {}
    )
    hop_gaps = (
        {
            key: _paired_group_gap(result["models"][key], result["models"][gnn_key], hops)
            for key in comparison_keys
        }
        if hops
        else {}
    )
    collinear = bool(hops) and all(hops[query_id] == answer_count[query_id] for query_id in hops)
    return {
        "status": result["status"],
        "dataset": result["dataset"],
        "frozen_gnn_key": gnn_key,
        "answer_count": {
            "by_model": by_answer_count,
            "paired_minus_gnn": answer_gaps,
        },
        "hop_count": {
            "provenance": hop_provenance,
            "perfectly_collinear_with_gold_count": collinear,
            "by_model": by_hop,
            "paired_minus_gnn": hop_gaps,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--musique-raw", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    analysis = analyze(result, args.manifest, args.musique_raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": analysis["dataset"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
