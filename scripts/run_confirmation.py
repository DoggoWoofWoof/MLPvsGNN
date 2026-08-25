#!/usr/bin/env python
"""Run the frozen multi-seed and capacity confirmation on one complete dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any
import warnings

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.operator_models import build_operator_model, model_parameter_counts  # noqa: E402
from mp_retrieval.protocol import seed_everything, sha256_file  # noqa: E402

from scripts.run_operator_screen import (  # noqa: E402
    _build_local_topologies,
    _hash_tensor_contract,
    _score_queries,
    _train_model,
)


EFFECTIVENESS_KEYS = (
    "recall@1",
    "recall@5",
    "recall@20",
    "mrr",
    "full_coverage@1",
    "full_coverage@5",
    "full_coverage@20",
    "conditional_recall@1",
    "conditional_recall@5",
    "conditional_recall@20",
    "conditional_hit@1",
    "conditional_hit@5",
    "conditional_hit@20",
    "conditional_mrr",
)
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def model_specs(widths: list[int], best_gnn: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for width in widths:
        specs.append({"key": f"plain_mlp_h{width}", "model": "plain_mlp", "hidden_dim": width})
        specs.append({"key": f"offset_mlp_h{width}", "model": "offset_mlp", "hidden_dim": width})
    max_width = max(widths)
    specs.extend(
        [
            {
                "key": f"offset_mlp_k4_h{max_width}",
                "model": "offset_mlp_k4",
                "hidden_dim": max_width,
            },
            {"key": f"{best_gnn}_h{max_width}", "model": best_gnn, "hidden_dim": max_width},
        ]
    )
    return specs


def _mean_std(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = (
        T_CRITICAL_95.get(len(values), 1.96) * sample_std / math.sqrt(len(values))
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


def _aggregate_runs(seed_runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    runs = list(seed_runs.values())
    return {
        "test_metrics": {
            key: _mean_std([float(run["metrics"][key]) for run in runs])
            for key in EFFECTIVENESS_KEYS
        },
        "validation_recall@5": _mean_std(
            [float(run["training"]["best_validation_recall@5"]) for run in runs]
        ),
        "training_seconds": _mean_std(
            [float(run["training"]["training_seconds"]) for run in runs]
        ),
        "latency_ms_per_query": _mean_std(
            [float(run["inference"]["latency_ms_per_query"]) for run in runs]
        ),
        "throughput_queries_per_second": _mean_std(
            [float(run["inference"]["throughput_queries_per_second"]) for run in runs]
        ),
        "peak_gpu_memory_mb_incremental": _mean_std(
            [float(run["inference"]["peak_gpu_memory_mb_incremental"]) for run in runs]
        ),
    }


def _repeated_score(
    model: torch.nn.Module,
    test_queries: list,
    nodes: torch.Tensor,
    query_embeddings: torch.Tensor,
    local_edges: dict,
    device: torch.device,
    *,
    batch_size: int,
    ks: tuple[int, ...],
    repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metrics: dict[str, Any] | None = None
    per_query: list[dict[str, Any]] | None = None
    telemetry_rows: list[dict[str, float]] = []
    for _ in range(repeats):
        current_metrics, current_rows, telemetry = _score_queries(
            model,
            test_queries,
            nodes,
            query_embeddings,
            local_edges,
            device,
            batch_size=batch_size,
            ks=ks,
            timed=True,
        )
        metrics = current_metrics
        per_query = current_rows
        telemetry_rows.append(telemetry)
    assert metrics is not None and per_query is not None
    latency = [row["latency_ms_per_query"] for row in telemetry_rows]
    throughput = [row["throughput_queries_per_second"] for row in telemetry_rows]
    total_memory = [row["peak_gpu_memory_mb_total"] for row in telemetry_rows]
    incremental_memory = [row["peak_gpu_memory_mb_incremental"] for row in telemetry_rows]
    inference = {
        "repeats": repeats,
        "latency_ms_per_query": statistics.median(latency),
        "throughput_queries_per_second": statistics.median(throughput),
        "peak_gpu_memory_mb_total": max(total_memory),
        "peak_gpu_memory_mb_incremental": max(incremental_memory),
        "repeat_telemetry": telemetry_rows,
    }
    return metrics, per_query, inference


def _select_capacity(
    models: dict[str, dict[str, Any]],
    family: str,
    tie_margin_percentage_points: float,
) -> dict[str, Any]:
    candidates = {
        key: value
        for key, value in models.items()
        if value["spec"]["model"] == family
    }
    validation = {
        key: float(value["aggregate"]["validation_recall@5"]["mean"])
        for key, value in candidates.items()
    }
    best_value = max(validation.values())
    margin = tie_margin_percentage_points / 100.0
    eligible = [key for key, value in validation.items() if best_value - value <= margin]
    selected = min(
        eligible,
        key=lambda key: int(candidates[key]["parameters"]["parameters"]),
    )
    return {
        "criterion": "mean validation recall@5 across frozen seeds",
        "tie_margin_percentage_points": tie_margin_percentage_points,
        "validation_means": validation,
        "eligible_within_tie_margin": eligible,
        "selected": selected,
        "selected_parameters": candidates[selected]["parameters"],
        "selected_test_metrics": candidates[selected]["aggregate"]["test_metrics"],
    }


def _paired_seed_gaps(
    models: dict[str, dict[str, Any]],
    left: str,
    right: str,
    seeds: list[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20"):
        values = [
            float(models[left]["seeds"][str(seed)]["metrics"][metric])
            - float(models[right]["seeds"][str(seed)]["metrics"][metric])
            for seed in seeds
        ]
        result[metric] = {"by_seed": dict(zip(map(str, seeds), values)), **_mean_std(values)}
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError(
            f"Expected {args.expected_queries} queries for {args.dataset}, got {len(dataset.queries)}"
        )
    splits = {
        split: dataset.split(split)
        for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST)
    }
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        nodes = torch.from_numpy(np.asarray(dataset.node_array)).to(device=device, dtype=torch.float32)
        query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(
            device=device, dtype=torch.float32
        )
    local_edges, topology_seconds = _build_local_topologies(dataset, dataset.queries)
    specs = model_specs(args.hidden_widths, args.best_gnn)
    model_results: dict[str, Any] = {}
    for spec in specs:
        key = spec["key"]
        seed_results: dict[str, Any] = {}
        parameter_counts: dict[str, int] | None = None
        for seed in args.seeds:
            seed_everything(seed)
            model = build_operator_model(
                spec["model"],
                dataset.feature_dim,
                spec["hidden_dim"],
                layers=args.layers,
                offset_directions=args.offset_directions,
                dropout=args.dropout,
                temperature=args.temperature,
            )
            current_counts = model_parameter_counts(model)
            if parameter_counts is None:
                parameter_counts = current_counts
            elif current_counts != parameter_counts:
                raise RuntimeError("Parameter count changed across seeds")
            trained, training = _train_model(
                model,
                splits[QuerySplit.TRAIN],
                splits[QuerySplit.VALIDATION],
                nodes,
                query_embeddings,
                local_edges,
                device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=seed,
            )
            metrics, rows, inference = _repeated_score(
                trained,
                splits[QuerySplit.TEST],
                nodes,
                query_embeddings,
                local_edges,
                device,
                batch_size=args.batch_size,
                ks=tuple(args.ks),
                repeats=args.inference_repeats,
            )
            seed_results[str(seed)] = {
                "metrics": metrics,
                "training": training,
                "inference": inference,
                "per_query": {
                    query.query_id: {
                        **row,
                        "gold_count": int(query.relevant_global.numel()),
                        "in_pool_gold_count": int(query.relevant_local.numel()),
                    }
                    for query, row in zip(splits[QuerySplit.TEST], rows)
                },
            }
            del trained, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        assert parameter_counts is not None
        model_results[key] = {
            "spec": spec,
            "parameters": parameter_counts,
            "uses_topology": spec["model"] in {"gcn", "sage", "gat", "gin"},
            "seeds": seed_results,
            "aggregate": _aggregate_runs(seed_results),
        }
    capacity_selection = {
        family: _select_capacity(
            model_results,
            family,
            args.capacity_tie_margin_percentage_points,
        )
        for family in ("plain_mlp", "offset_mlp")
    }
    max_width = max(args.hidden_widths)
    gnn_key = f"{args.best_gnn}_h{max_width}"
    paired_gaps = {
        key: _paired_seed_gaps(model_results, key, gnn_key, args.seeds)
        for key in (
            f"plain_mlp_h{max_width}",
            f"offset_mlp_h{max_width}",
            f"offset_mlp_k4_h{max_width}",
            capacity_selection["plain_mlp"]["selected"],
            capacity_selection["offset_mlp"]["selected"],
        )
    }
    source_files = {
        name: {
            "bytes": (args.data / name).stat().st_size,
            "sha256": (
                sha256_file(args.data / name)
                if name != "nodes.npy" or args.nodes_sha256 == "computed-by-launcher"
                else args.nodes_sha256
            ),
        }
        for name in (
            "nodes.npy",
            "queries_all.npy",
            "dense_top200_all.npy",
            "splade_top200_all.npy",
            "query_ids_all.json",
            "graph.pt",
        )
    }
    result = {
        "status": "CONFIRMATION_GATE_NOT_PAPER_FINAL",
        "dataset": dataset.dataset,
        "data": {
            "queries": len(dataset.queries),
            "nodes": dataset.num_nodes,
            "edges": dataset.metadata["num_edges"],
            "feature_dim": dataset.feature_dim,
            "splits": {split.name.lower(): len(values) for split, values in splits.items()},
            "source_files": source_files,
            "candidate_contract_sha256": dataset.metadata["candidate_contract_sha256"],
            "topology_preprocessing_seconds": topology_seconds,
        },
        "comparison_contract": {
            "same_frozen_query_embeddings": True,
            "same_frozen_node_embeddings": True,
            "same_candidate_pool": True,
            "same_labels": True,
            "same_negatives": True,
            "same_multi_positive_listwise_loss": True,
            "same_canonical_splits": True,
            "same_seeds": args.seeds,
            "same_optimizer_and_epoch_budget": True,
            "offset_inference_uses_adjacency": False,
            "gnn_only_privileged_input": "candidate-induced adjacency",
            "sha256": _hash_tensor_contract(dataset.queries),
        },
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "frozen_best_gnn": args.best_gnn,
        "models": model_results,
        "capacity_selection_validation_only": capacity_selection,
        "paired_minus_frozen_gnn": paired_gaps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--best-gnn", choices=["gcn", "sage", "gat", "gin"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--hidden-widths", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--offset-directions", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument("--capacity-tie-margin-percentage-points", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--nodes-sha256", default="computed-by-launcher")
    args = parser.parse_args()
    result = run(args)
    compact = {
        key: {
            "params": value["parameters"]["parameters"],
            "val_R@5_mean": value["aggregate"]["validation_recall@5"]["mean"],
            "test_R@5_mean": value["aggregate"]["test_metrics"]["recall@5"]["mean"],
            "test_R@5_sd": value["aggregate"]["test_metrics"]["recall@5"]["sample_std"],
        }
        for key, value in result["models"].items()
    }
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "capacity_selection": result["capacity_selection_validation_only"],
                "table": compact,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
