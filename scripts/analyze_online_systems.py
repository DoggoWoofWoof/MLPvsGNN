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


# Stages whose products a per-query cache would store. The warm-cache path in
# Package C reads these instead of recomputing them, so their cost is both what
# building one cache entry costs and what reading it saves.
CACHEABLE_PREFIX_STAGES = (
    "fusion_and_seed_ms",
    "topology_induction_ms",
    "query_local_summary_ms",
)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-12)


def _cache_break_even(
    model_condition: dict[str, Any],
    cached_ms_per_query: float,
    batch_size: int,
) -> dict[str, Any]:
    """Compute-only break-even for the per-query cache.

    Derived entirely from quantities the frozen protocol already measures; it
    adds no timing and changes no measurement. Definition: serving one query
    uncached costs ``uncached_ms``. Serving it from cache costs
    ``cached_ms`` but first requires building its entry, which costs the
    cacheable prefix. Caching therefore repays itself once

        build_ms <= repeats * (uncached_ms - cached_ms)

    so ``break_even_additional_servings`` is the smallest number of *further*
    servings of the same query that repay building its entry, and total
    servings is that plus the one that built it.

    This is compute-only. The frozen protocol measures static-asset bytes but
    not the per-query cache footprint, so storage is excluded and the result is
    a lower bound on the true break-even point.
    """
    stages = model_condition["batch_latency_ms"]
    build_ms = sum(
        float(stages[stage]["mean"]) for stage in CACHEABLE_PREFIX_STAGES
    ) / max(batch_size, 1)
    uncached_ms = float(model_condition["total_latency_ms_per_query"]["mean"])
    saving_ms = uncached_ms - float(cached_ms_per_query)
    repays = saving_ms > 0.0
    return {
        "definition": "compute_only_per_query_cache_excluding_storage",
        "cache_build_ms_per_query": build_ms,
        "uncached_ms_per_query": uncached_ms,
        "cached_ms_per_query": float(cached_ms_per_query),
        "saving_ms_per_served_query": saving_ms,
        "cache_ever_repays": repays,
        "break_even_additional_servings": (
            build_ms / saving_ms if repays else None
        ),
        "break_even_total_servings": (
            1.0 + build_ms / saving_ms if repays else None
        ),
    }


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
        for batch_size in config["measurement"]["batch_sizes"]:
            key = f"batch_{batch_size}"
            conditions[key]["cache_break_even"] = {
                model: _cache_break_even(
                    conditions[key]["models"][model],
                    cached[model]["latency_ms_per_query"],
                    batch_size,
                )
                for model in MODEL_NAMES
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
            "cache_break_even_is_compute_only_excluding_storage": True,
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
            "## Cache break-even",
            "",
            (
                "Derived from the measured stage breakdown and the Package C warm-cache "
                "reference; it adds no timing. A per-query cache stores the products of "
                "fusion, seed construction, topology induction, and QLS summaries, so "
                "building one entry costs that prefix and reading it saves the difference "
                "between the uncached and cached paths. The break-even column is the "
                "number of *further* servings of the same query that repay building its "
                "entry."
            ),
            "",
            (
                "This is compute-only: the frozen protocol measures static-asset bytes "
                "but not per-query cache footprint, so storage is excluded and these are "
                "lower bounds."
            ),
            "",
            (
                "| Dataset | Batch | Model | Build ms | Uncached ms | Cached ms | "
                "Saved ms | Break-even repeats |"
            ),
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["datasets"]:
        for condition_name, condition in row["uncached"].items():
            batch = condition_name.removeprefix("batch_")
            for model, entry in condition["cache_break_even"].items():
                repeats = entry["break_even_additional_servings"]
                rendered = "never repays" if repeats is None else f"{repeats:.2f}"
                lines.append(
                    f"| {row['dataset']} | {batch} | {model} | "
                    f"{entry['cache_build_ms_per_query']:.3f} | "
                    f"{entry['uncached_ms_per_query']:.3f} | "
                    f"{entry['cached_ms_per_query']:.3f} | "
                    f"{entry['saving_ms_per_served_query']:.3f} | {rendered} |"
                )
    lines.append("")
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
