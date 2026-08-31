#!/usr/bin/env python
"""Compile the frozen edge-provenance message-passing experiment."""

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
CONFIG_PATH = REPO_ROOT / "configs" / "edge_provenance.yaml"
MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
FAMILIES = (
    "sealed_a_multigraph",
    "baseline_a_simple",
    "symbolic_b",
    "knn_only",
    "full_union_c",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_arrays(
    path: Path,
    expected_sha256: str,
    expected_order_sha256: str,
    seeds: list[int],
) -> dict[str, np.ndarray]:
    if _sha256(path) != expected_sha256:
        raise ValueError(f"Edge-provenance query metrics failed SHA-256: {path}")
    with np.load(path) as packed:
        if tuple(map(str, packed["metric_names"].tolist())) != METRICS:
            raise ValueError(f"Edge-provenance metric order changed: {path}")
        if str(packed["query_order_sha256"].item()) != expected_order_sha256:
            raise ValueError(f"Edge-provenance query order changed: {path}")
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
    output: dict[str, Any] = {"models": {}}
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
    output["topology"] = result["data"]["topology"]
    output["feature_cache"] = result["feature_cache"]
    output["warm_cache_only"] = True
    return output


def _family_source(
    root: Path,
    dataset: str,
    family: str,
    confirmation: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    if family == "sealed_a_multigraph":
        return confirmation, root.parent / "sa_mlp_confirmation" / f"{dataset}.query_metrics.npz"
    condition = root / dataset / family
    return (
        json.loads((condition / "result.json").read_text(encoding="utf-8")),
        condition / "query_metrics.npz",
    )


def compile_analysis(
    root: Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seeds = list(map(int, config["training"]["seeds"]))
    rows: list[dict[str, Any]] = []
    graph_audits: dict[str, Any] = {}
    for dataset_position, (dataset, spec) in enumerate(config["datasets"].items()):
        confirmation = json.loads(
            (REPO_ROOT / spec["confirmation"]).read_text(encoding="utf-8")
        )
        if confirmation.get("status") != "SA_MLP_CONFIRMATION_DATASET_COMPLETE":
            raise ValueError(f"Sealed confirmation is incomplete: {dataset}")
        expected_order = confirmation["data"]["test_query_order_sha256"]
        for family_position, family in enumerate(FAMILIES):
            result, packed_path = _family_source(root, dataset, family, confirmation)
            if family == "sealed_a_multigraph":
                expected_status = "SA_MLP_CONFIRMATION_DATASET_COMPLETE"
            else:
                expected_status = "EDGE_PROVENANCE_DATASET_FAMILY_COMPLETE"
            if (
                result.get("status") != expected_status
                or sorted(map(int, result["models"]["sa_mlp"]["seeds"])) != seeds
                or sorted(map(int, result["models"]["seed_aware_gnn"]["seeds"])) != seeds
            ):
                raise ValueError(f"Edge-provenance result is incomplete: {dataset}/{family}")
            if family != "sealed_a_multigraph":
                if (
                    result.get("dataset") != dataset
                    or result.get("family") != family
                    or result["comparison_contract"].get("test_selected_edge_families")
                    is not False
                    or result["data"]["test_query_order_sha256"] != expected_order
                ):
                    raise ValueError(f"Edge-provenance contract failed: {dataset}/{family}")
                if dataset not in graph_audits:
                    graph_audits[dataset] = result["edge_provenance"]
            query_metrics = result["query_metrics"]
            arrays = _load_arrays(
                packed_path,
                query_metrics["sha256"],
                expected_order,
                seeds,
            )
            low, high = _hierarchical_paired_ci(
                arrays["seed_aware_gnn"],
                arrays["sa_mlp"],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + dataset_position * 101 + family_position,
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
            data = result["data"]
            rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "directed_edges": int(data.get("directed_edges", data.get("edges"))),
                    "models": models,
                    "seed_aware_gnn_minus_sa_mlp": contrast,
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
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        for metric in METRICS:
            adjusted = _holm(
                {
                    row["dataset"]: row["seed_aware_gnn_minus_sa_mlp"][metric][
                        "paired_seed_t_pvalue"
                    ]
                    for row in family_rows
                }
            )
            for row in family_rows:
                record = row["seed_aware_gnn_minus_sa_mlp"][metric]
                record["holm_adjusted_pvalue_across_datasets"] = adjusted[row["dataset"]]
                record["holm_significant_0.05"] = adjusted[row["dataset"]] < 0.05
    return {
        "status": "EDGE_PROVENANCE_ALL_DATASETS_ANALYZED",
        "datasets": rows,
        "families": list(FAMILIES),
        "graph_audits": graph_audits,
        "bootstrap": {
            "method": "paired_optimizer_seed_then_shared_query_resampling",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "claims": {
            "all_registered_families_reported": True,
            "sealed_a_reused_not_retrained": True,
            "simple_a_duplicate_normalization_control_reported": True,
            "test_selected_edge_family": False,
            "warm_cache_systems_only": True,
        },
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Edge-provenance results",
        "",
        (
            "All preregistered edge families are shown. Positive gaps mean learned message "
            "passing exceeds QLS-MLP on the same graph family."
        ),
        "",
        "| Dataset | Edge family | Directed edges | QLS R@5 | GNN R@5 | GNN − QLS |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in analysis["datasets"]:
        lines.append(
            f"| {row['dataset']} | {row['family']} | {row['directed_edges']} | "
            f"{_pct(row['models']['sa_mlp']['recall@5']['mean'])} | "
            f"{_pct(row['models']['seed_aware_gnn']['recall@5']['mean'])} | "
            f"{100 * row['seed_aware_gnn_minus_sa_mlp']['recall@5']['seed_effect']['mean']:+.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "The sealed A multigraph is reused from the completed fairness confirmation. "
                "The simple-A row is the mandatory duplicate-normalization control; therefore a "
                "difference between sealed A and simple A cannot be misreported as edge semantics."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "outputs" / "edge_provenance")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "edge_provenance_analysis.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "docs" / "EDGE_PROVENANCE_RESULTS.md",
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
