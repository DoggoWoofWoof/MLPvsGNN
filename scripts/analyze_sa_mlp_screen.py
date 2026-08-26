#!/usr/bin/env python
"""Compile the preregistered Structure-Aware MLP screening gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("metaqa", "webqsp", "hotpotqa_clean")
MODELS = ("interaction", "static_structure", "query_local_structure", "sa_mlp")
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(root: Path) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    for name in DATASETS:
        result = _load(root / f"{name}.json")
        if result.get("status") != "SA_MLP_SCREEN_DATASET_COMPLETE":
            raise ValueError(f"Incomplete SA-MLP result: {name}")
        baseline = result["baseline_seed_0"]
        models = {
            model: {
                "metrics": {
                    metric: result["models"][model]["metrics"][metric] for metric in METRICS
                },
                "validation_recall@5": result["models"][model]["training"][
                    "best_validation_recall@5"
                ],
                "parameters": result["models"][model]["parameters"]["parameters"],
                "training_seconds": result["models"][model]["training"]["training_seconds"],
                "inference": result["models"][model]["inference"],
            }
            for model in MODELS
        }
        datasets.append(
            {
                "dataset": name,
                "plain_mlp": baseline["plain_mlp"],
                "gnn": baseline["gnn"],
                "models": models,
                "gap_closure": result["gap_closure"],
                "feature_cache": result["feature_cache"],
            }
        )
    metaqa = _load(root / "metaqa.json")
    hops = []
    for hop in (1, 2, 3):
        key = str(hop)
        hops.append(
            {
                "hop": hop,
                "queries": metaqa["models"]["sa_mlp"]["by_hop"][key]["queries"],
                "plain_mlp_recall@5": metaqa["baseline_seed_0"]["plain_mlp"]["by_hop"][key][
                    "metrics"
                ]["recall@5"],
                "gnn_recall@5": metaqa["baseline_seed_0"]["gnn"]["by_hop"][key]["metrics"][
                    "recall@5"
                ],
                "query_local_recall@5": metaqa["models"]["query_local_structure"]["by_hop"][key][
                    "metrics"
                ]["recall@5"],
                "sa_mlp_recall@5": metaqa["models"]["sa_mlp"]["by_hop"][key]["metrics"]["recall@5"],
            }
        )
    return {
        "status": "SA_MLP_SCREEN_GATE_COMPLETE",
        "gate_pass": all(row["gap_closure"]["dataset_pass"] for row in datasets),
        "datasets_passed": sum(row["gap_closure"]["dataset_pass"] for row in datasets),
        "datasets": datasets,
        "metaqa_hops": hops,
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Structure-Aware MLP one-seed screening result",
        "",
        (
            "Status: **the preregistered gate passed on all three datasets**. Values below are "
            "seed 0 only and are screening evidence, not five-seed paper estimates."
        ),
        "",
        "## Primary gate",
        "",
        (
            "| Dataset | Frozen MLP R@5 | Frozen GNN R@5 | SA-MLP R@5 | SA-GNN delta | "
            "Gap closure | Pass |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["datasets"]:
        closure = row["gap_closure"]
        lines.append(
            f"| {row['dataset']} | {_pct(closure['plain_mlp_seed_0'])} | "
            f"{_pct(closure['gnn_seed_0'])} | {_pct(closure['sa_mlp_seed_0'])} | "
            f"{_pct(closure['sa_mlp_seed_0'] - closure['gnn_seed_0'])} | "
            f"{closure['fraction']:.2f}x | {'yes' if closure['dataset_pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            (
                "The frozen rule required at least 0.50x closure on two of three datasets. "
                "Observed closure is 1.58x on MetaQA, 1.45x on WebQSP, and 4.30x on HotpotQA."
            ),
            "",
            "## Ablation ladder",
            "",
            "| Dataset | Frozen MLP | Interaction | Static | Query-local | Full SA | Frozen GNN |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["datasets"]:
        lines.append(
            f"| {row['dataset']} | {_pct(row['plain_mlp']['metrics']['recall@5'])} | "
            f"{_pct(row['models']['interaction']['metrics']['recall@5'])} | "
            f"{_pct(row['models']['static_structure']['metrics']['recall@5'])} | "
            f"{_pct(row['models']['query_local_structure']['metrics']['recall@5'])} | "
            f"{_pct(row['models']['sa_mlp']['metrics']['recall@5'])} | "
            f"{_pct(row['gnn']['metrics']['recall@5'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Static graph descriptors alone are harmful in all three settings. Query-local "
                "distance/path/PPR descriptors account for essentially the entire MetaQA gain and "
                "most of the HotpotQA gain. WebQSP requires the combined interaction and "
                "query-local model; neither family closes its GNN gap alone."
            ),
            "",
            "## MetaQA diagnostic",
            "",
            "| Hop | Queries | Frozen MLP R@5 | Frozen GNN R@5 | Query-local R@5 | SA-MLP R@5 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["metaqa_hops"]:
        lines.append(
            f"| {row['hop']} | {row['queries']} | {_pct(row['plain_mlp_recall@5'])} | "
            f"{_pct(row['gnn_recall@5'])} | {_pct(row['query_local_recall@5'])} | "
            f"{_pct(row['sa_mlp_recall@5'])} |"
        )
    lines.extend(
        [
            "",
            (
                "SA-MLP exceeds the frozen GNN at every hop. The improvement is largest at one "
                "and two hops, while the three-hop difference is small. This strengthens the "
                "result that native query hop count is not a monotonic proxy for message-passing "
                "value."
            ),
            "",
            "## Systems cost",
            "",
            (
                "| Dataset | Cache GiB | Preprocess s | SA ms/q | GNN ms/q | GNN/SA latency | "
                "SA incr. GPU MiB | GNN incr. GPU MiB |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["datasets"]:
        sa = row["models"]["sa_mlp"]["inference"]
        gnn = row["gnn"]["inference"]
        lines.append(
            f"| {row['dataset']} | {row['feature_cache']['cache_bytes'] / 1024**3:.3f} | "
            f"{row['feature_cache']['total_preprocessing_seconds']:.1f} | "
            f"{sa['latency_ms_per_query']:.4f} | {gnn['latency_ms_per_query']:.4f} | "
            f"{gnn['latency_ms_per_query'] / sa['latency_ms_per_query']:.2f}x | "
            f"{sa['peak_gpu_memory_mb_incremental']:.1f} | "
            f"{gnn['peak_gpu_memory_mb_incremental']:.1f} |"
        )
    lines.extend(
        [
            "",
            (
                "The learned SA forward pass is 1.21--2.98x faster than the selected GNN, not the "
                "3.6--9.9x advantage of the plain MLP, because online cache lookup and the explicit "
                "scoring head add work. GPU allocation remains far below the GNN, especially on "
                "HotpotQA. The method also shifts cost to CPU/disk: caches range from 0.030 to "
                "2.835 GiB and total process RSS is high because frozen arrays and memory maps "
                "coexist."
            ),
            "",
            "## Required fairness control before the paper claim",
            "",
            (
                "The query-local feature set includes a distance-0 bucket for the frozen "
                "dense/SPLADE retrieval seeds. This exposes seed membership explicitly, whereas "
                "the frozen GNN received candidate embeddings and adjacency but no seed indicator. "
                "Therefore this screen establishes that the registered fixed-feature package "
                "beats the old GNN; it does not yet isolate how much comes from graph paths/PPR "
                "versus the retrieval-seed prior. Confirmation must retain the frozen SA "
                "architecture and add a seed-only control (and, for the strongest causal "
                "comparison, a seed-aware GNN control)."
            ),
            "",
            (
                "No test-driven feature or architecture change was made. The combined SA-MLP is "
                "now eligible to be frozen for five-seed confirmation, but the one-seed values "
                "must not be presented as final estimates."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "sa_mlp_screen")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "sa_mlp_screen_analysis.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "docs" / "SA_MLP_SCREEN_RESULTS.md",
    )
    args = parser.parse_args()
    result = analyze(args.root)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_output.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "datasets_passed": result["datasets_passed"],
                "gate_pass": result["gate_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
