#!/usr/bin/env python
"""Run the frozen P0 A2 training-free structural controls."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mp_retrieval.rank_fusion import (
    FrozenRankContract,
    aggregate_metric_arrays,
    load_frozen_rank_contract,
    ranking_metrics,
    sha256_file,
)
from mp_retrieval.structural_controls import (
    FROZEN_LOCAL_FEATURE_NAMES,
    FrozenStructuralCache,
    candidate_contract_hashes,
    equal_rrf_fusion,
    fixed_structural_scores,
    rank_scores,
    selected_rrf_ranking,
    stable_candidate_union,
)

METHODS = (
    "selected_rrf",
    "distance",
    "ppr",
    "path_connectivity",
    "structural_summary",
    "selected_rrf_plus_ppr",
    "selected_rrf_plus_structural_summary",
)
METRICS = (
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
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_metric_key(name: str) -> str:
    return name.replace("@", "_at_")


def _validate_alignment(
    contract: FrozenRankContract,
    cache: FrozenStructuralCache,
    confirmation: dict[str, Any],
) -> dict[str, Any]:
    metadata = cache.metadata
    sealed = confirmation["feature_cache"]
    if tuple(metadata.get("local_feature_names", ())) != FROZEN_LOCAL_FEATURE_NAMES:
        raise ValueError("Structural cache feature names differ from the sealed QLS contract")
    if metadata.get("source_fingerprint_sha256") != confirmation["data_fingerprint_sha256"]:
        raise ValueError("Structural source fingerprint differs from the confirmation artifact")
    if metadata.get("contract_sha256") != sealed["contract_sha256"]:
        raise ValueError("Structural cache contract differs from the confirmation artifact")
    if metadata.get("candidate_contract_sha256") != sealed["candidate_contract_sha256"]:
        raise ValueError("Structural candidate contract differs from the confirmation artifact")
    if cache.local.ndim != 2 or cache.local.shape[1] != len(FROZEN_LOCAL_FEATURE_NAMES):
        raise ValueError("Structural local feature matrix has an unexpected shape")
    if cache.candidate_ptr.shape != (contract.query_count + 1,):
        raise ValueError("Structural candidate pointers do not cover every query")
    if int(cache.candidate_ptr[-1]) != int(cache.local.shape[0]):
        raise ValueError("Structural candidate pointers do not cover every feature row")
    if int(metadata["candidate_rows"]) != int(cache.local.shape[0]):
        raise ValueError("Structural metadata candidate-row count is inconsistent")
    if cache.query_position.shape != (contract.query_count,) or not np.array_equal(
        cache.query_position,
        np.arange(contract.query_count, dtype=cache.query_position.dtype),
    ):
        raise ValueError("Structural cache is not in canonical query-index order")

    observed_hashes = candidate_contract_hashes(
        contract.query_ids,
        contract.dense,
        contract.splade,
        cache.candidate_ptr,
    )
    comparison = confirmation["comparison_contract"]
    expected_tensor = comparison["sha256"]["candidates"]
    if observed_hashes["candidate_tensor_sha256"] != expected_tensor:
        raise ValueError("Frozen candidate tensor digest differs from the QLS confirmation")
    compatibility = comparison.get("candidate_compatibility_proof")
    if compatibility is not None:
        expected_order = compatibility["candidate_id_order_sha256"]
        if observed_hashes["candidate_id_order_sha256"] != expected_order:
            raise ValueError("Frozen candidate ID/order digest differs from the QLS confirmation")
    return {
        "status": "BIT_EXACT_A2_INPUT_ALIGNMENT",
        "source_fingerprint_sha256": metadata["source_fingerprint_sha256"],
        "structural_contract_sha256": metadata["contract_sha256"],
        "candidate_contract_sha256": metadata["candidate_contract_sha256"],
        **observed_hashes,
        "candidate_order_reference": (
            "candidate_compatibility_proof"
            if compatibility is not None
            else "comparison_contract.sha256.candidates"
        ),
        "queries": contract.query_count,
        "candidate_rows": int(cache.local.shape[0]),
        "feature_names": list(FROZEN_LOCAL_FEATURE_NAMES),
        "graph_loaded_by_A2": False,
        "embeddings_loaded_by_A2": False,
    }


def _evaluate_split(
    contract: FrozenRankContract,
    cache: FrozenStructuralCache,
    indices: np.ndarray,
    *,
    dense_weight: float,
    rrf_constant: int,
    save_query_metrics: bool,
) -> tuple[dict[str, dict[str, float | int]], dict[str, np.ndarray]]:
    arrays = {
        method: {metric: np.empty(indices.size, dtype=np.float32) for metric in METRICS}
        for method in METHODS
    }
    gold_count = np.empty(indices.size, dtype=np.int16)
    in_pool_gold_count = np.empty(indices.size, dtype=np.int16)

    for output_index, raw_query_index in enumerate(indices):
        query_index = int(raw_query_index)
        dense = np.asarray(contract.dense[query_index], dtype=np.int64)
        splade = np.asarray(contract.splade[query_index], dtype=np.int64)
        candidates = stable_candidate_union(dense, splade)
        start = int(cache.candidate_ptr[query_index])
        end = int(cache.candidate_ptr[query_index + 1])
        if end - start != candidates.size:
            raise ValueError(f"Structural row count changed at query {query_index}")
        local = np.asarray(cache.local[start:end], dtype=np.float32)
        structural = fixed_structural_scores(local)
        selected_rrf = selected_rrf_ranking(
            dense,
            splade,
            dense_weight=dense_weight,
            constant=rrf_constant,
        )
        rankings = {
            "selected_rrf": selected_rrf,
            **{
                method: rank_scores(candidates, structural[method])
                for method in (
                    "distance",
                    "ppr",
                    "path_connectivity",
                    "structural_summary",
                )
            },
        }
        rankings["selected_rrf_plus_ppr"] = equal_rrf_fusion(
            selected_rrf,
            rankings["ppr"],
            constant=rrf_constant,
        )
        rankings["selected_rrf_plus_structural_summary"] = equal_rrf_fusion(
            selected_rrf,
            rankings["structural_summary"],
            constant=rrf_constant,
        )
        golds = contract.golds[query_index]
        available = set(golds) & set(map(int, candidates))
        gold_count[output_index] = len(golds)
        in_pool_gold_count[output_index] = len(available)
        for method, ranking in rankings.items():
            row = ranking_metrics(ranking, golds, candidates)
            for metric in METRICS:
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


def _validate_a1_reproduction(
    observed: dict[str, float | int],
    rank_result: dict[str, Any],
) -> dict[str, Any]:
    expected = rank_result["test"]["weighted_rrf_selected"]
    checked = {}
    for metric in METRICS:
        difference = float(observed[metric]) - float(expected[metric])
        if abs(difference) > 2e-7:
            raise ValueError(f"A2 failed to reproduce corrected A1 {metric}: {difference}")
        checked[metric] = difference
    return {
        "status": "CORRECTED_A1_SELECTED_RRF_REPRODUCED",
        "maximum_absolute_difference": max(map(abs, checked.values())),
        "metric_differences": checked,
    }


def run(
    args: argparse.Namespace,
    *,
    rank_result_payload: dict[str, Any] | None = None,
    confirmation_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["status"] != "PROTOCOL_FROZEN_BEFORE_TEST_EVALUATION":
        raise ValueError("A2 protocol is not frozen")
    if args.dataset not in config["frozen_inputs"]["datasets"]:
        raise ValueError(f"Dataset is outside the A2 protocol: {args.dataset}")
    rank_result = rank_result_payload or _load(args.rank_result)
    confirmation = confirmation_payload or _load(args.confirmation)
    if rank_result.get("dataset") != args.dataset or confirmation.get("dataset") != args.dataset:
        raise ValueError("A2 reference artifacts name a different dataset")
    dense_weight = float(rank_result["selection"]["selected_dense_weight"])
    rrf_constant = int(config["methods"]["selected_rrf"]["constant"])

    contract = load_frozen_rank_contract(args.data, dataset=args.dataset, hash_sources=False)
    cache = FrozenStructuralCache.load(args.feature_cache)
    alignment_started = time.perf_counter()
    alignment = _validate_alignment(contract, cache, confirmation)
    alignment_seconds = time.perf_counter() - alignment_started

    validation_started = time.perf_counter()
    validation, _ = _evaluate_split(
        contract,
        cache,
        contract.split_indices["validation"],
        dense_weight=dense_weight,
        rrf_constant=rrf_constant,
        save_query_metrics=False,
    )
    validation_seconds = time.perf_counter() - validation_started
    test_started = time.perf_counter()
    test, packed = _evaluate_split(
        contract,
        cache,
        contract.split_indices["test"],
        dense_weight=dense_weight,
        rrf_constant=rrf_constant,
        save_query_metrics=True,
    )
    test_seconds = time.perf_counter() - test_started
    reproduction = _validate_a1_reproduction(test["selected_rrf"], rank_result)

    args.query_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.query_metrics_output, **packed)
    result = {
        "status": "P0_A2_FIXED_STRUCTURAL_CONTROLS_COMPLETE",
        "dataset": args.dataset,
        "protocol": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "tag": "p0-fixed-structural-controls-a1-v2",
        },
        "input_contract": {
            "data": str(args.data),
            "feature_cache": str(args.feature_cache),
            "rank_result_tag": config["frozen_inputs"]["rank_control_tag"],
            "rank_result_sha256": rank_result.get("query_metrics", {}).get("sha256"),
            "confirmation_data_fingerprint_sha256": confirmation["data_fingerprint_sha256"],
            "selected_dense_weight": dense_weight,
            "rrf_constant": rrf_constant,
            "split_counts": {
                name: int(values.size) for name, values in contract.split_indices.items()
            },
        },
        "alignment": alignment,
        "validation": validation,
        "test": test,
        "a1_reproduction": reproduction,
        "test_access_audit": {
            "validation_selected_A2_weights_or_rules": False,
            "all_locked_methods_reported": list(test) == list(METHODS),
            "test_selected_models_or_features": False,
        },
        "query_metrics": {
            "path": str(args.query_metrics_output),
            "sha256": sha256_file(args.query_metrics_output),
            "rows": int(contract.split_indices["test"].size),
        },
        "timing": {
            "alignment_seconds": alignment_seconds,
            "validation_seconds": validation_seconds,
            "test_seconds": test_seconds,
            "total_seconds": time.perf_counter() - started,
            "service_latency": False,
            "structural_cache_reused": True,
        },
    }
    _atomic_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "p0_fixed_structural_controls.yaml",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--rank-result", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-metrics-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    namespace = parse_args()
    completed = run(namespace)
    print(
        json.dumps(
            {
                "dataset": completed["dataset"],
                "test_R@5": {
                    method: metrics["recall@5"] for method, metrics in completed["test"].items()
                },
                "seconds": completed["timing"]["total_seconds"],
            }
        )
    )
