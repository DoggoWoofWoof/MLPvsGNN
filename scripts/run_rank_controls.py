from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mp_retrieval.rank_fusion import (
    FrozenRankContract,
    aggregate_metric_arrays,
    load_frozen_rank_contract,
    ranking_metrics,
    rrf_rankings,
    select_dense_weight,
    sha256_file,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_metric_key(name: str) -> str:
    return name.replace("@", "_at_")


def _evaluate_split(
    contract: FrozenRankContract,
    indices: np.ndarray,
    *,
    dense_weights: list[float],
    constant: int,
    batch_size: int,
    save_query_metrics: bool,
) -> tuple[dict[str, dict[str, float | int]], dict[str, np.ndarray]]:
    method_names = ["dense", "splade"] + [f"rrf_dense_{weight:.2f}" for weight in dense_weights]
    metric_names = [
        "candidate_ceiling",
        "candidate_available",
        "recall@1",
        "recall@5",
        "recall@20",
        "mrr",
        "full_coverage@20",
        "conditional_recall@1",
        "conditional_recall@5",
        "conditional_recall@20",
        "conditional_mrr",
    ]
    arrays = {
        method: {name: np.empty(indices.size, dtype=np.float32) for name in metric_names}
        for method in method_names
    }
    gold_count = np.empty(indices.size, dtype=np.int16)
    in_pool_gold_count = np.empty(indices.size, dtype=np.int16)

    for batch_start in range(0, indices.size, batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        dense = np.asarray(contract.dense[batch_indices], dtype=np.int64)
        splade = np.asarray(contract.splade[batch_indices], dtype=np.int64)
        rrf = rrf_rankings(
            dense,
            splade,
            dense_weights=dense_weights,
            constant=constant,
            top_k=20,
        )
        rankings = {
            "dense": dense[:, :20],
            "splade": splade[:, :20],
            **{f"rrf_dense_{weight:.2f}": value for weight, value in rrf.items()},
        }
        for local_index, query_index in enumerate(batch_indices.tolist()):
            output_index = batch_start + local_index
            golds = contract.golds[query_index]
            candidates = np.concatenate((dense[local_index], splade[local_index]))
            available = set(golds) & set(map(int, candidates))
            gold_count[output_index] = len(golds)
            in_pool_gold_count[output_index] = len(available)
            for method, method_rankings in rankings.items():
                row = ranking_metrics(method_rankings[local_index], golds, candidates)
                for metric in metric_names:
                    value = row[metric]
                    arrays[method][metric][output_index] = np.nan if value is None else float(value)

    aggregate = {
        method: aggregate_metric_arrays(metric_arrays) for method, metric_arrays in arrays.items()
    }
    packed: dict[str, np.ndarray] = {}
    if save_query_metrics:
        packed["query_index"] = indices.astype(np.int64, copy=False)
        packed["gold_count"] = gold_count
        packed["in_pool_gold_count"] = in_pool_gold_count
        for method, metric_arrays in arrays.items():
            for metric, values in metric_arrays.items():
                packed[f"{method}__{_safe_metric_key(metric)}"] = values
    return aggregate, packed


def _run_dataset(
    dataset: str,
    dataset_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    contract = load_frozen_rank_contract(dataset_root, dataset=dataset, hash_sources=True)
    weights = [float(value) for value in config["methods"]["weighted_rrf"]["dense_weight_grid"]]
    constant = int(config["methods"]["equal_rrf"]["constant"])
    batch_size = int(config["evaluation"]["batch_size"])

    validation_started = time.perf_counter()
    validation, _ = _evaluate_split(
        contract,
        contract.split_indices["validation"],
        dense_weights=weights,
        constant=constant,
        batch_size=batch_size,
        save_query_metrics=False,
    )
    validation_seconds = time.perf_counter() - validation_started
    weighted_validation = {
        weight: validation[f"rrf_dense_{weight:.2f}"] for weight in weights
    }
    selected_weight = select_dense_weight(weighted_validation, metric="recall@5")

    test_weights = sorted({0.5, selected_weight})
    test_started = time.perf_counter()
    test_all, packed = _evaluate_split(
        contract,
        contract.split_indices["test"],
        dense_weights=test_weights,
        constant=constant,
        batch_size=batch_size,
        save_query_metrics=True,
    )
    test_seconds = time.perf_counter() - test_started
    selected_name = f"rrf_dense_{selected_weight:.2f}"
    equal_name = "rrf_dense_0.50"
    test = {
        "dense": test_all["dense"],
        "splade": test_all["splade"],
        "equal_rrf": test_all[equal_name],
        "weighted_rrf_selected": test_all[selected_name],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    query_metrics_path = output_root / f"{dataset}.query_metrics.npz"
    np.savez_compressed(query_metrics_path, **packed)
    result = {
        "status": "P0_A1_RANK_CONTROLS_COMPLETE",
        "dataset": dataset,
        "source_root": str(dataset_root),
        "source_sha256": contract.source_sha256,
        "query_count": contract.query_count,
        "source_width": contract.source_width,
        "split_counts": {name: int(values.size) for name, values in contract.split_indices.items()},
        "rrf_constant": constant,
        "weighted_dense_grid": weights,
        "selection": {
            "split": "validation",
            "metric": "recall@5",
            "selected_dense_weight": selected_weight,
            "selected_splade_weight": 1.0 - selected_weight,
            "validation_metrics": {
                f"{weight:.2f}": weighted_validation[weight] for weight in weights
            },
        },
        "validation_fixed_controls": {
            "dense": validation["dense"],
            "splade": validation["splade"],
            "equal_rrf": validation[equal_name],
        },
        "test": test,
        "test_access_audit": {
            "weighted_test_weights_computed": test_weights,
            "unselected_weighted_test_results_computed": False,
        },
        "query_metrics": {
            "path": str(query_metrics_path),
            "sha256": sha256_file(query_metrics_path),
            "rows": int(contract.split_indices["test"].size),
        },
        "timing": {
            "validation_seconds": validation_seconds,
            "test_seconds": test_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    result_path = output_root / f"{dataset}.json"
    _atomic_json(result_path, result)
    result["result_path"] = str(result_path)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["status"] != "PROTOCOL_FROZEN_BEFORE_TEST_EVALUATION":
        raise ValueError("Rank-control protocol is not frozen")
    datasets = list(args.datasets or config["datasets"])
    unknown = sorted(set(datasets) - set(config["datasets"]))
    if unknown:
        raise ValueError(f"Datasets are outside the frozen protocol: {unknown}")

    results = []
    for dataset in datasets:
        dataset_root = args.source_root / dataset / config["source_contract"]["dataset_subdirectory"]
        result = _run_dataset(dataset, dataset_root, args.output_root, config)
        results.append(result)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "selected_dense_weight": result["selection"]["selected_dense_weight"],
                    "test_R@5": {
                        method: metrics["recall@5"] for method, metrics in result["test"].items()
                    },
                    "seconds": result["timing"]["total_seconds"],
                }
            ),
            flush=True,
        )

    summary = {
        "status": "P0_A1_RANK_CONTROLS_COMPLETE",
        "protocol": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "tag": "p0-rank-controls-protocol-v1",
        },
        "datasets": {
            result["dataset"]: {
                "result_path": result["result_path"],
                "selected_dense_weight": result["selection"]["selected_dense_weight"],
                "test": result["test"],
                "timing": result["timing"],
            }
            for result in results
        },
    }
    _atomic_json(args.output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p0_rank_controls.yaml")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "p0_rank_controls")
    parser.add_argument("--datasets", nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
