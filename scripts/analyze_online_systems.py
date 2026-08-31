#!/usr/bin/env python
"""Compile the uncached unseen-embedding systems benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "online_systems.yaml"
MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")
STAGES = (
    "fusion_and_seed_ms",
    "topology_induction_ms",
    "query_local_summary_ms",
    "gather_transfer_forward_topk_ms",
    "total_ms",
)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-12)


def compile_analysis(root: Path, budget_root: Path) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = []
    for dataset in config["datasets"]:
        path = root / f"{dataset}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Online systems result is incomplete: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "UNCACHED_UNSEEN_EMBEDDING_SYSTEMS_COMPLETE"
            or result.get("dataset") != dataset
            or result["boundary"].get("query_specific_cache_reads_in_timed_path") is not False
            or result["parity"].get("topology_bit_exact") is not True
            or result["parity"].get("qls_local_float16_bit_exact") is not True
        ):
            raise ValueError(f"Online systems contract failed: {path}")
        budget_path = budget_root / dataset / "budget_400" / "result.json"
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        if (
            budget.get("status") != "CANDIDATE_BUDGET_DATASET_COMPLETE"
            or budget.get("data_fingerprint_sha256")
            != result.get("data_fingerprint_sha256")
        ):
            raise ValueError(f"Budget-400 checkpoint source differs: {dataset}")
        conditions = {}
        for batch_size in config["measurement"]["batch_sizes"]:
            key = f"batch_{batch_size}"
            condition = result["conditions"][key]["models"]
            qls = condition["sa_mlp"]
            gnn = condition["seed_aware_gnn"]
            conditions[key] = {
                "models": condition,
                "qls_total_latency_divided_by_gnn": _ratio(
                    qls["total_latency_ms_per_query"]["mean"],
                    gnn["total_latency_ms_per_query"]["mean"],
                ),
                "gnn_throughput_divided_by_qls": _ratio(
                    gnn["throughput_queries_per_second"],
                    qls["throughput_queries_per_second"],
                ),
            }
        cached = {}
        for model in MODEL_NAMES:
            inference = budget["models"][model]["seeds"]["0"]["inference"]
            cached[model] = {
                "latency_ms_per_query": float(inference["latency_ms_per_query"]),
                "incremental_gpu_memory_mb": float(
                    inference["peak_gpu_memory_mb_incremental"]
                ),
            }
        rows.append(
            {
                "dataset": dataset,
                "sample": result["sample"],
                "boundary": result["boundary"],
                "parity": result["parity"],
                "startup": result["startup"],
                "static_storage": result["static_storage"],
                "uncached": conditions,
                "cached_operator_only_reference": cached,
            }
        )
    return {
        "status": "UNCACHED_UNSEEN_EMBEDDING_SYSTEMS_ALL_DATASETS_ANALYZED",
        "datasets": rows,
        "claims": {
            "post_retrieval_boundary_only": True,
            "raw_text_and_upstream_retrieval_out_of_scope": True,
            "query_specific_cache_reads_in_timed_path": False,
            "candidate_topology_and_qls_parity_gated": True,
            "cached_and_uncached_timings_never_spliced": True,
            "hardware": config["modal"]["gpu"],
        },
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Uncached unseen-embedding systems results",
        "",
        (
            "Each request begins with a held-out query embedding and Dense/SPLADE ranked IDs. "
            "The timed path rebuilds equal-RRF candidates, retrieval seeds, candidate topology, "
            "and QLS local summaries before model inference."
        ),
        "",
        "| Dataset | Batch | Queries | QLS ms/query | GNN ms/query | QLS/GNN | QLS q/s | GNN q/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["datasets"]:
        for condition_name, condition in row["uncached"].items():
            models = condition["models"]
            batch = condition_name.removeprefix("batch_")
            lines.append(
                f"| {row['dataset']} | {batch} | {row['sample']['unique_queries']} | "
                f"{models['sa_mlp']['total_latency_ms_per_query']['mean']:.3f} | "
                f"{models['seed_aware_gnn']['total_latency_ms_per_query']['mean']:.3f} | "
                f"{condition['qls_total_latency_divided_by_gnn']:.3f} | "
                f"{models['sa_mlp']['throughput_queries_per_second']:.1f} | "
                f"{models['seed_aware_gnn']['throughput_queries_per_second']:.1f} |"
            )
    lines.extend(
        [
            "",
            (
                "These are post-retrieval ranker timings, not raw-text end-to-end retrieval "
                "timings. Query encoding, Dense ANN lookup, and SPLADE index lookup are shared "
                "upstream and excluded for both methods. Cached operator-only numbers remain a "
                "separate reference."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "online_systems")
    parser.add_argument(
        "--budget-root", type=Path, default=REPO_ROOT / "outputs" / "candidate_budget"
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "online_systems_analysis.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "docs" / "ONLINE_SYSTEMS_RESULTS.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = compile_analysis(args.root, args.budget_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps({"status": analysis["status"]}, indent=2))


if __name__ == "__main__":
    main()
