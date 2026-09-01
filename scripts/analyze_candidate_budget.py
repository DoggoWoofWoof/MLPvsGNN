#!/usr/bin/env python
"""Compile the frozen equal-RRF candidate-budget experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.analyze_linear_rank_structure import (
    _hierarchical_paired_ci,
    _holm,
    _mean_std_ci,
    _paired_t_pvalue,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "candidate_budget.yaml"
MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_arrays(
    path: Path,
    result: dict[str, Any],
    seeds: list[int],
) -> dict[str, np.ndarray]:
    if _sha256(path) != result["query_metrics"]["sha256"]:
        raise ValueError(f"Candidate-budget query metrics failed SHA-256: {path}")
    with np.load(path) as packed:
        if tuple(map(str, packed["metric_names"].tolist())) != METRICS:
            raise ValueError(f"Candidate-budget metric order changed: {path}")
        if str(packed["query_order_sha256"].item()) != result["data"][
            "test_query_order_sha256"
        ]:
            raise ValueError(f"Candidate-budget query order changed: {path}")
        return {
            model: np.stack(
                [
                    np.asarray(packed[f"{model}_seed_{seed}"], dtype=np.float32)
                    for seed in seeds
                ]
            )
            for model in MODEL_NAMES
        }


def _systems(result: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    output = {
        "feature_cache": result["feature_cache"],
        "topology": result["data"]["topology"],
        "models": {},
    }
    for model in MODEL_NAMES:
        records = result["models"][model]["seeds"]
        output["models"][model] = {
            "parameters": result["models"][model]["parameters"],
            "latency_ms_per_query": _mean_std_ci(
                [
                    float(records[str(seed)]["inference"]["latency_ms_per_query"])
                    for seed in seeds
                ]
            ),
            "incremental_gpu_memory_mb": _mean_std_ci(
                [
                    float(
                        records[str(seed)]["inference"]["peak_gpu_memory_mb_incremental"]
                    )
                    for seed in seeds
                ]
            ),
            "training_seconds": _mean_std_ci(
                [
                    float(records[str(seed)]["training"]["training_seconds"])
                    for seed in seeds
                ]
            ),
        }
    qls = output["models"]["sa_mlp"]["latency_ms_per_query"]["mean"]
    gnn = output["models"]["seed_aware_gnn"]["latency_ms_per_query"]["mean"]
    output["qls_latency_divided_by_gnn"] = float(qls) / max(float(gnn), 1e-12)
    output["warm_cache_only"] = True
    return output


def compile_analysis(
    root: Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seeds = list(map(int, config["training"]["seeds"]))
    budgets = list(map(int, config["candidate_contract"]["budgets"]))
    rows: list[dict[str, Any]] = []
    order_hashes: dict[str, str] = {}
    for dataset_position, dataset in enumerate(config["datasets"]):
        for budget_position, budget in enumerate(budgets):
            condition_root = root / dataset / f"budget_{budget}"
            result_path = condition_root / "result.json"
            packed_path = condition_root / "query_metrics.npz"
            if not result_path.is_file() or not packed_path.is_file():
                raise FileNotFoundError(f"Candidate-budget output is incomplete: {condition_root}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                result.get("status") != "CANDIDATE_BUDGET_DATASET_COMPLETE"
                or result.get("dataset") != dataset
                or int(result.get("budget", -1)) != budget
                or sorted(map(int, result["models"]["sa_mlp"]["seeds"])) != seeds
                or sorted(map(int, result["models"]["seed_aware_gnn"]["seeds"])) != seeds
                or result["comparison_contract"].get("test_selected_budget") is not False
            ):
                raise ValueError(f"Candidate-budget contract failed: {result_path}")
            order_hash = result["data"]["test_query_order_sha256"]
            if dataset in order_hashes and order_hashes[dataset] != order_hash:
                raise ValueError(f"Test query order changed across budgets: {dataset}")
            order_hashes[dataset] = order_hash
            arrays = _load_arrays(packed_path, result, seeds)
            low, high = _hierarchical_paired_ci(
                arrays["seed_aware_gnn"],
                arrays["sa_mlp"],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + dataset_position * 101 + budget_position,
            )
            models = {
                model: {
                    metric: _mean_std_ci(
                        arrays[model][:, :, metric_index].mean(axis=1).tolist()
                    )
                    for metric_index, metric in enumerate(METRICS)
                }
                for model in MODEL_NAMES
            }
            contrast = {}
            for metric_index, metric in enumerate(METRICS):
                by_seed = (
                    arrays["seed_aware_gnn"][:, :, metric_index].mean(axis=1)
                    - arrays["sa_mlp"][:, :, metric_index].mean(axis=1)
                ).tolist()
                contrast[metric] = {
                    "seed_effect": _mean_std_ci(by_seed),
                    "paired_seed_t_pvalue": _paired_t_pvalue(by_seed),
                    "paired_hierarchical_query_ci95_low": float(low[metric_index]),
                    "paired_hierarchical_query_ci95_high": float(high[metric_index]),
                }
            rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "models": models,
                    "seed_aware_gnn_minus_sa_mlp": contrast,
                    "equal_rrf_test": result["equal_rrf_test"],
                    "context": result["data"]["structural_context_all_queries"],
                    "test_candidate_ceiling_mean": result["data"][
                        "test_candidate_ceiling_mean"
                    ],
                    "systems": _systems(result, seeds),
                    "metaqa_hops": (
                        {
                            model: {
                                str(hop): {
                                    metric: _mean_std_ci(
                                        [
                                            float(
                                                result["models"][model]["seeds"][str(seed)][
                                                    "by_hop"
                                                ][str(hop)]["metrics"][metric]
                                            )
                                            for seed in seeds
                                        ]
                                    )
                                    for metric in METRICS
                                }
                                for hop in (1, 2, 3)
                            }
                            for model in MODEL_NAMES
                        }
                        if dataset == "metaqa"
                        else None
                    ),
                }
            )
    for budget in budgets:
        budget_rows = [row for row in rows if row["budget"] == budget]
        for metric in METRICS:
            adjusted = _holm(
                {
                    row["dataset"]: row["seed_aware_gnn_minus_sa_mlp"][metric][
                        "paired_seed_t_pvalue"
                    ]
                    for row in budget_rows
                }
            )
            for row in budget_rows:
                record = row["seed_aware_gnn_minus_sa_mlp"][metric]
                record["holm_adjusted_pvalue_across_datasets"] = adjusted[row["dataset"]]
                record["holm_significant_0.05"] = adjusted[row["dataset"]] < 0.05
    return {
        "status": "CANDIDATE_BUDGET_ALL_DATASETS_ANALYZED",
        "datasets": rows,
        "budgets": budgets,
        "bootstrap": {
            "method": "paired_optimizer_seed_then_shared_query_resampling",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "claims": {
            "all_budgets_reported": True,
            "test_selected_budget": False,
            "candidate_ceiling_is_diagnostic_only": True,
            "latency_is_warm_cache": True,
            "uncached_systems_reported_separately": True,
        },
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Equal-RRF candidate-budget results",
        "",
        (
            "All four preregistered budgets are reported for both matched models. No budget is "
            "selected using test effectiveness."
        ),
        "",
        "| Dataset | Budget | PoolCov | RRF R@5 | QLS R@5 | GNN R@5 | GNN − QLS | Candidates | Edges | Components |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["datasets"]:
        context = row["context"]
        lines.append(
            f"| {row['dataset']} | {row['budget']} | "
            f"{_pct(row['test_candidate_ceiling_mean'])} | "
            f"{_pct(row['equal_rrf_test']['recall@5'])} | "
            f"{_pct(row['models']['sa_mlp']['recall@5']['mean'])} | "
            f"{_pct(row['models']['seed_aware_gnn']['recall@5']['mean'])} | "
            f"{100 * row['seed_aware_gnn_minus_sa_mlp']['recall@5']['seed_effect']['mean']:+.2f} | "
            f"{context['candidate_count_mean']:.1f} | "
            f"{context['stored_directed_edges_mean']:.1f} | "
            f"{context['connected_components_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            (
                "PoolCov is the mean fraction of a query's golds that reached the "
                "candidate pool. It is an oracle diagnostic, is never given to either "
                "model, and is not an achievable Recall@K: for a query with `g` golds "
                "and `p` of them pooled, Recall@K cannot exceed `min(p, K) / g`, which "
                "is below `p / g` whenever `g` exceeds `K`. The per-cut-off ceilings are "
                "reported in CANDIDATE_BUDGET_AND_HEADROOM_RESULTS.md. "
                "The systems fields in this table remain warm-cache measurements; the separate "
                "uncached unseen-embedding benchmark charges fusion, topology induction, QLS "
                "local summaries, transfer, forward, and top-K."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "candidate_budget")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "candidate_budget_analysis.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "docs" / "CANDIDATE_BUDGET_RESULTS.md",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = compile_analysis(
        args.root,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps({"status": analysis["status"]}, indent=2))


if __name__ == "__main__":
    main()
