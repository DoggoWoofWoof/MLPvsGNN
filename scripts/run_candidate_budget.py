#!/usr/bin/env python
"""Run the frozen equal-RRF candidate/context-budget comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.candidate_budget import (
    build_budget_dataset,
    structural_context_statistics,
)
from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.operator_models import model_parameter_counts
from mp_retrieval.protocol import seed_everything
from mp_retrieval.structural_features import (
    build_or_load_structural_features,
)
from scripts.run_confirmation import _aggregate_runs
from scripts.run_edge_provenance import _atomic_json, _sha256
from scripts.run_main_table import (
    _hop_aggregates,
    _load_or_build_topologies,
    _state_sha256,
)
from scripts.run_operator_screen import _aggregate_rows, _metric_row
from scripts.run_sa_mlp_confirmation import (
    PACKED_QUERY_METRICS,
    _build_model,
    _fit,
    _query_order_sha256,
    _repeated_score,
    _rows_to_array,
    validate_candidate_contract,
)

MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")


def _rank_baseline(test_queries, ks: tuple[int, ...]) -> tuple[dict[str, Any], np.ndarray]:
    rows = []
    for query in test_queries:
        size = int(query.candidate_index.numel())
        rows.append(_metric_row(torch.arange(size, 0, -1, dtype=torch.float32), query, ks))
    return _aggregate_rows(rows), _rows_to_array(rows)


def run(
    args: argparse.Namespace,
    checkpoint_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    source = load_complete_dataset(args.data, dataset=args.dataset)
    if len(source.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the frozen protocol")
    source_contract = validate_candidate_contract(
        args.baseline,
        source,
        args.candidate_contract_compatibility,
    )
    if args.baseline["selected_gnn"]["model"] != args.selected_gnn:
        raise ValueError("Frozen selected GNN differs from the budget protocol")
    dense = np.load(args.data / "dense_top200_all.npy", mmap_mode="r")
    splade = np.load(args.data / "splade_top200_all.npy", mmap_mode="r")
    dataset = build_budget_dataset(
        source,
        dense,
        splade,
        budget=args.budget,
        rrf_constant=args.rrf_constant,
        chunk_size=args.fusion_chunk_size,
    )
    observed_hops = sorted({query.hop for query in dataset.queries if query.hop is not None})
    if observed_hops != sorted(args.required_hops):
        raise ValueError(f"Expected hop labels {args.required_hops}, got {observed_hops}")
    splits = {
        split: dataset.split(split)
        for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST)
    }
    if any(not rows for rows in splits.values()):
        raise RuntimeError("Candidate-budget comparison requires all canonical splits")
    topologies, topology_metadata = _load_or_build_topologies(dataset, args.topology_cache)
    feature_fingerprint = hashlib.sha256(
        (
            args.data_fingerprint_sha256
            + dataset.metadata["candidate_contract_sha256"]
            + f"equal_rrf_k{args.rrf_constant}_budget{args.budget}"
        ).encode("utf-8")
    ).hexdigest()
    features = build_or_load_structural_features(
        dataset,
        topologies,
        args.feature_cache,
        source_fingerprint=feature_fingerprint,
        config=args.feature_config,
    )
    if checkpoint_hook is not None:
        checkpoint_hook()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        nodes = torch.from_numpy(np.asarray(dataset.node_array)).to(
            device=device, dtype=torch.float32
        )
        query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(
            device=device, dtype=torch.float32
        )
    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    test_queries = splits[QuerySplit.TEST]
    rrf_metrics, rrf_rows = _rank_baseline(test_queries, tuple(args.ks))
    result: dict[str, Any]
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            existing.get("dataset") != args.dataset
            or int(existing.get("budget", -1)) != args.budget
            or existing.get("data_fingerprint_sha256") != args.data_fingerprint_sha256
            or existing.get("budget_candidate_contract_sha256")
            != dataset.metadata["candidate_contract_sha256"]
        ):
            raise ValueError("Existing partial budget result has a different frozen contract")
        result = existing
    else:
        result = {
            "status": "CANDIDATE_BUDGET_IN_PROGRESS",
            "dataset": args.dataset,
            "budget": args.budget,
            "data_fingerprint_sha256": args.data_fingerprint_sha256,
            "source_candidate_contract": source_contract,
            "budget_candidate_contract_sha256": dataset.metadata[
                "candidate_contract_sha256"
            ],
            "candidate_ordering": {
                "method": "equal_rrf",
                "constant": args.rrf_constant,
                "dense_weight": 0.5,
                "splade_weight": 0.5,
                "tie_break": "ascending_global_node_id",
                "seed_prior": "original_dense_top5_union_splade_top5_intersect_budget",
            },
            "data": {
                "queries": len(dataset.queries),
                "nodes": dataset.num_nodes,
                "global_directed_edges": dataset.metadata["num_edges"],
                "splits": {split.name.lower(): len(rows) for split, rows in splits.items()},
                "test_query_order_sha256": _query_order_sha256(test_queries),
                "topology": topology_metadata,
                "structural_context_all_queries": structural_context_statistics(
                    dataset.queries, topologies
                ),
                "test_candidate_ceiling_mean": float(
                    np.mean([query.candidate_ceiling for query in test_queries])
                ),
            },
            "equal_rrf_test": rrf_metrics,
            "feature_cache": features.metadata,
            "comparison_contract": {
                "same_budget_for_both_models": True,
                "same_equal_rrf_candidates_and_order": True,
                "same_embeddings_labels_splits_loss_seeds_optimizer_epochs": True,
                "same_original_retrieval_seed_prior_intersected_with_pool": True,
                "same_sealed_a_multigraph": True,
                "test_selected_budget": False,
            },
            "models": {name: {"seeds": {}} for name in MODEL_NAMES},
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
                if key not in {"baseline", "feature_config"}
            },
        }
    query_arrays: dict[str, np.ndarray] = {"equal_rrf": rrf_rows}
    if args.query_metrics_output.is_file():
        with np.load(args.query_metrics_output) as packed:
            query_arrays.update(
                {
                    key: np.asarray(packed[key])
                    for key in packed.files
                    if key not in {"metric_names", "query_order_sha256"}
                }
            )

    def checkpoint() -> None:
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    for model_name in MODEL_NAMES:
        for seed in args.seeds:
            seed_key = str(seed)
            if seed_key in result["models"][model_name]["seeds"]:
                continue
            seed_everything(seed)
            model = _build_model(
                model_name,
                dataset,
                features,
                args.selected_gnn,
                target_parameters,
                args,
            )
            counts = model_parameter_counts(model)
            if model_name == "seed_aware_gnn":
                if int(counts["parameters"]) - target_parameters != args.hidden_dim:
                    raise ValueError("Seed-aware GNN parameter count changed")
            elif abs(int(counts["parameters"]) - target_parameters) > args.max_parameter_difference:
                raise ValueError("QLS-MLP parameter match exceeds the frozen tolerance")
            model, training = _fit(
                model_name,
                model,
                splits[QuerySplit.TRAIN],
                splits[QuerySplit.VALIDATION],
                nodes,
                query_embeddings,
                topologies,
                features,
                device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=seed,
            )
            metrics, rows, inference = _repeated_score(
                model_name,
                model,
                test_queries,
                nodes,
                query_embeddings,
                topologies,
                features,
                device,
                batch_size=args.batch_size,
                ks=tuple(args.ks),
                repeats=args.inference_repeats,
            )
            checkpoint_path = args.checkpoint_root / model_name / f"seed_{seed}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            result["models"][model_name]["parameters"] = counts
            result["models"][model_name]["seeds"][seed_key] = {
                "metrics": metrics,
                "training": training,
                "inference": inference,
                "by_hop": _hop_aggregates(test_queries, rows),
                "checkpoint_sha256": _state_sha256(model.state_dict()),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_file_sha256": _sha256(checkpoint_path),
            }
            query_arrays[f"{model_name}_seed_{seed}"] = _rows_to_array(rows)
            args.query_metrics_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.query_metrics_output,
                metric_names=np.asarray(PACKED_QUERY_METRICS),
                query_order_sha256=np.asarray(result["data"]["test_query_order_sha256"]),
                **query_arrays,
            )
            checkpoint()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for model_name in MODEL_NAMES:
        runs = result["models"][model_name]["seeds"]
        if sorted(map(int, runs)) != sorted(args.seeds):
            raise RuntimeError(f"Incomplete seed set for {model_name}")
        result["models"][model_name]["aggregate"] = _aggregate_runs(runs)
    result["query_metrics"] = {
        "path": str(args.query_metrics_output),
        "sha256": _sha256(args.query_metrics_output),
        "format": "npz_float32_test_query_order_v1",
        "metrics": list(PACKED_QUERY_METRICS),
        "arrays": sorted(query_arrays),
    }
    result["status"] = "CANDIDATE_BUDGET_DATASET_COMPLETE"
    checkpoint()
    return result


if __name__ == "__main__":
    raise SystemExit("Use scripts/modal_candidate_budget.py for the frozen execution")
