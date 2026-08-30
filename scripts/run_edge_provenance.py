#!/usr/bin/env python
"""Run the frozen edge-provenance comparison on complete retrieval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.edge_provenance import graph_payload
from mp_retrieval.l2_data import edge_index_to_csr
from mp_retrieval.operator_models import model_parameter_counts
from mp_retrieval.protocol import seed_everything
from mp_retrieval.structural_features import (
    build_or_load_structural_features,
)
from scripts.run_confirmation import _aggregate_runs
from scripts.run_main_table import (
    _hop_aggregates,
    _load_or_build_topologies,
    _state_sha256,
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _family_dataset(data_root: Path, family_graph: Path, dataset_name: str):
    dataset = load_complete_dataset(data_root, dataset=dataset_name)
    edge_index, num_nodes = graph_payload(family_graph)
    if num_nodes != dataset.num_nodes:
        raise ValueError("Edge-family graph node count differs from frozen embeddings")
    rowptr, col, _edge_type = edge_index_to_csr(torch.from_numpy(edge_index), num_nodes)
    metadata = {
        **dataset.metadata,
        "num_edges": int(edge_index.shape[1]),
        "edge_family_graph_sha256": _sha256(family_graph),
    }
    return replace(dataset, rowptr=rowptr, col=col, metadata=metadata)


def run(
    args: argparse.Namespace,
    checkpoint_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    family_metadata = json.loads(args.family_metadata.read_text(encoding="utf-8"))
    if family_metadata.get("format") != "edge_provenance_families_v1":
        raise ValueError("Unsupported edge-family sidecar")
    if family_metadata.get("selected_family") != args.family:
        raise ValueError("Selected edge-family sidecar does not match the requested family")
    dataset = _family_dataset(args.data, args.family_graph, args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the frozen protocol")
    if family_metadata["num_nodes"] != dataset.num_nodes:
        raise ValueError("Edge-family manifest node count differs from the complete artifact")
    candidate_contract = validate_candidate_contract(
        args.baseline,
        dataset,
        args.candidate_contract_compatibility,
    )
    if args.baseline["selected_gnn"]["model"] != args.selected_gnn:
        raise ValueError("Frozen selected GNN differs from the edge-provenance protocol")
    observed_hops = sorted({query.hop for query in dataset.queries if query.hop is not None})
    if observed_hops != sorted(args.required_hops):
        raise ValueError(f"Expected hop labels {args.required_hops}, got {observed_hops}")
    splits = {
        split: dataset.split(split)
        for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST)
    }
    if any(not rows for rows in splits.values()):
        raise RuntimeError("Edge-provenance comparison requires all canonical splits")
    topologies, topology_metadata = _load_or_build_topologies(dataset, args.topology_cache)
    feature_fingerprint = hashlib.sha256(
        (
            args.data_fingerprint_sha256
            + family_metadata["selected_edge_key_sha256"]
            + args.family
        ).encode("utf-8")
    ).hexdigest()
    features = build_or_load_structural_features(
        dataset,
        topologies,
        args.feature_cache,
        source_fingerprint=feature_fingerprint,
        config=args.feature_config,
        graph_path=args.family_graph,
    )
    if checkpoint_hook is not None:
        checkpoint_hook()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    nodes = torch.from_numpy(np.asarray(dataset.node_array)).to(device=device, dtype=torch.float32)
    queries = torch.from_numpy(np.asarray(dataset.query_array)).to(
        device=device, dtype=torch.float32
    )
    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    result: dict[str, Any]
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            existing.get("dataset") != args.dataset
            or existing.get("family") != args.family
            or existing.get("data_fingerprint_sha256") != args.data_fingerprint_sha256
            or existing.get("edge_key_sha256")
            != family_metadata["selected_edge_key_sha256"]
        ):
            raise ValueError("Existing partial result belongs to a different frozen contract")
        result = existing
    else:
        result = {
            "status": "EDGE_PROVENANCE_IN_PROGRESS",
            "dataset": args.dataset,
            "family": args.family,
            "data_fingerprint_sha256": args.data_fingerprint_sha256,
            "edge_key_sha256": family_metadata["selected_edge_key_sha256"],
            "candidate_contract": candidate_contract,
            "data": {
                "queries": len(dataset.queries),
                "nodes": dataset.num_nodes,
                "directed_edges": dataset.metadata["num_edges"],
                "splits": {split.name.lower(): len(rows) for split, rows in splits.items()},
                "test_query_order_sha256": _query_order_sha256(splits[QuerySplit.TEST]),
                "topology": topology_metadata,
            },
            "edge_provenance": family_metadata,
            "feature_cache": features.metadata,
            "comparison_contract": {
                "same_frozen_embeddings_candidates_labels_loss_splits": True,
                "same_frozen_retrieval_seeds": True,
                "same_selected_gnn_family_and_parameters": True,
                "only_changed_input": "global_edge_family",
                "test_selected_edge_families": False,
            },
            "models": {name: {"seeds": {}} for name in MODEL_NAMES},
            "config": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
                if key not in {"baseline", "feature_config"}
            },
        }
    query_arrays: dict[str, np.ndarray] = {}
    if args.query_metrics_output.is_file():
        with np.load(args.query_metrics_output) as packed:
            query_arrays = {
                key: np.asarray(packed[key])
                for key in packed.files
                if key not in {"metric_names", "query_order_sha256"}
            }

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
            difference = (
                int(counts["parameters"]) - target_parameters
                if model_name == "seed_aware_gnn"
                else abs(int(counts["parameters"]) - target_parameters)
            )
            expected_difference = args.hidden_dim if model_name == "seed_aware_gnn" else 0
            if (
                model_name == "seed_aware_gnn" and difference != expected_difference
            ) or (
                model_name == "sa_mlp" and difference > args.max_parameter_difference
            ):
                raise ValueError("Model parameter parity differs from the frozen confirmation")
            model, training = _fit(
                model_name,
                model,
                splits[QuerySplit.TRAIN],
                splits[QuerySplit.VALIDATION],
                nodes,
                queries,
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
                splits[QuerySplit.TEST],
                nodes,
                queries,
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
                "by_hop": _hop_aggregates(splits[QuerySplit.TEST], rows),
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
    result["status"] = "EDGE_PROVENANCE_DATASET_FAMILY_COMPLETE"
    checkpoint()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--family-graph", type=Path, required=True)
    parser.add_argument("--family-metadata", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-metrics-output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--topology-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--selected-gnn", required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--required-hops", nargs="*", type=int, default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-parameter-difference", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    confirmation = json.loads(args.baseline.read_text(encoding="utf-8"))
    args.baseline = confirmation["baseline"]
    screen_config = __import__("yaml").safe_load(
        (REPO_ROOT / "configs" / "sa_mlp_screen.yaml").read_text(encoding="utf-8")
    )
    args.feature_config = {
        "retrieval_seeds": screen_config["retrieval_seeds"],
        "static_features": screen_config["static_features"],
        "query_local_features": screen_config["query_local_features"],
        "preprocessing": {"query_chunk_size": 8192},
    }
    run(args)


if __name__ == "__main__":
    main()
