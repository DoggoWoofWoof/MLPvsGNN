#!/usr/bin/env python
"""Run one preregistered six-dataset main-table experiment."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import sys
import time
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
from mp_retrieval.topology_store import (  # noqa: E402
    PackedLocalTopologies,
    build_packed_topologies,
)
from scripts.run_confirmation import (  # noqa: E402
    _aggregate_runs,
    _mean_std,
    _repeated_score,
)
from scripts.run_operator_screen import (  # noqa: E402
    _aggregate_rows,
    _hash_tensor_contract,
    _train_model,
)


GNN_CANDIDATES = ("gcn", "sage", "gat", "gin")
PRIMARY_METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
COMPACT_QUERY_METRICS = (
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_or_build_topologies(
    dataset,
    cache: Path | None,
) -> tuple[PackedLocalTopologies, dict[str, Any]]:
    if cache is not None and (cache / "metadata.json").is_file():
        started = time.perf_counter()
        store = PackedLocalTopologies.load(cache)
        if store.edge_ptr.size != len(dataset.queries) + 1:
            raise ValueError("Topology cache query count does not match the frozen dataset")
        return store, {
            "cache_hit": True,
            "cache_load_seconds": time.perf_counter() - started,
            "cold_build_seconds": store.build_seconds,
            "candidate_induced_edges": store.num_edges,
            "packed_storage_bytes": store.storage_bytes,
        }
    store = build_packed_topologies(dataset, dataset.queries)
    if cache is not None:
        store.save(cache)
    return store, {
        "cache_hit": False,
        "cache_load_seconds": 0.0,
        "cold_build_seconds": store.build_seconds,
        "candidate_induced_edges": store.num_edges,
        "packed_storage_bytes": store.storage_bytes,
    }


def _compact_per_query(queries, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        query.query_id: {
            **{key: row[key] for key in COMPACT_QUERY_METRICS},
            "gold_count": int(query.relevant_global.numel()),
            "in_pool_gold_count": int(query.relevant_local.numel()),
            "hop": query.hop,
        }
        for query, row in zip(queries, rows)
    }


def _hop_aggregates(queries, rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for query, row in zip(queries, rows):
        if query.hop is not None:
            groups.setdefault(int(query.hop), []).append(row)
    return {
        str(hop): {"queries": len(values), "metrics": _aggregate_rows(values)}
        for hop, values in sorted(groups.items())
    }


def _paired_gaps(models: dict[str, Any], selected_gnn: str, seeds: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        values = [
            float(models["plain_mlp"]["seeds"][str(seed)]["metrics"][metric])
            - float(models[selected_gnn]["seeds"][str(seed)]["metrics"][metric])
            for seed in seeds
        ]
        output[metric] = {"by_seed": dict(zip(map(str, seeds), values)), **_mean_std(values)}
    return output


def _fit(
    model_name: str,
    seed: int,
    dataset,
    splits,
    nodes: torch.Tensor,
    query_embeddings: torch.Tensor,
    local_edges: PackedLocalTopologies,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, int]]:
    seed_everything(seed)
    model = build_operator_model(
        model_name,
        dataset.feature_dim,
        args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        temperature=args.temperature,
    )
    counts = model_parameter_counts(model)
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
    return trained, training, counts


def run(
    args: argparse.Namespace,
    checkpoint_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError(
            f"Expected {args.expected_queries} queries for {args.dataset}, "
            f"got {len(dataset.queries)}"
        )
    if args.required_hops:
        observed = sorted({query.hop for query in dataset.queries if query.hop is not None})
        if observed != sorted(args.required_hops):
            raise ValueError(f"Expected hop labels {args.required_hops}, got {observed}")
    splits = {
        split: dataset.split(split)
        for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST)
    }
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        nodes = torch.from_numpy(np.asarray(dataset.node_array)).to(
            device=device, dtype=torch.float32
        )
        query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(
            device=device, dtype=torch.float32
        )
    local_edges, topology = _load_or_build_topologies(dataset, args.topology_cache)

    source_names = [
        "nodes.npy",
        "queries_all.npy",
        "dense_top200_all.npy",
        "splade_top200_all.npy",
        "query_ids_all.json",
        "graph.pt",
    ]
    if (args.data / "node_ids.json").is_file():
        source_names.append("node_ids.json")
    staged_manifest_path = args.data / "_frozen_source_manifest.json"
    if staged_manifest_path.is_file():
        staged_manifest = json.loads(staged_manifest_path.read_text(encoding="utf-8"))
        source_files = staged_manifest["files"]
        if set(source_files) != set(source_names):
            raise ValueError("Staged source manifest does not match the required source contract")
        for name in source_names:
            if int(source_files[name]["bytes"]) != (args.data / name).stat().st_size:
                raise ValueError(f"Staged source size changed after fingerprinting: {name}")
    else:
        source_files = {
            name: {
                "bytes": (args.data / name).stat().st_size,
                "sha256": sha256_file(args.data / name),
            }
            for name in source_names
        }
    data_fingerprint = hashlib.sha256(
        json.dumps(source_files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    result: dict[str, Any] = {
        "status": "PAPER_MAIN_TABLE_IN_PROGRESS",
        "dataset": dataset.dataset,
        "data_fingerprint_sha256": data_fingerprint,
        "data": {
            "queries": len(dataset.queries),
            "nodes": dataset.num_nodes,
            "edges": dataset.metadata["num_edges"],
            "feature_dim": dataset.feature_dim,
            "splits": {split.name.lower(): len(values) for split, values in splits.items()},
            "source_files": source_files,
            "candidate_contract_sha256": dataset.metadata["candidate_contract_sha256"],
            "node_identity": dataset.metadata["node_identity"],
            "topology": topology,
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
            "gnn_only_privileged_input": "candidate-induced adjacency",
            "sha256": _hash_tensor_contract(dataset.queries),
        },
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "selection_validation_only": {
            "metric": "recall@5",
            "seed": args.selection_seed,
            "models": {},
        },
        "models": {},
    }

    def checkpoint() -> None:
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    selection_states: dict[str, dict[str, torch.Tensor]] = {}
    checkpoint_dir = args.output.parent / f"{args.output.stem}_selection_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for gnn_name in GNN_CANDIDATES:
        trained, training, counts = _fit(
            gnn_name,
            args.selection_seed,
            dataset,
            splits,
            nodes,
            query_embeddings,
            local_edges,
            device,
            args,
        )
        state = {key: value.detach().cpu() for key, value in trained.state_dict().items()}
        selection_states[gnn_name] = state
        state_path = checkpoint_dir / f"{gnn_name}.pt"
        torch.save(state, state_path)
        result["selection_validation_only"]["models"][gnn_name] = {
            "best_validation_recall@5": training["best_validation_recall@5"],
            "history": training["history"],
            "training_seconds": training["training_seconds"],
            "parameters": counts,
            "checkpoint_sha256": _state_sha256(state),
            "test_metrics_computed": False,
        }
        checkpoint()
        del trained
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected_gnn = max(
        GNN_CANDIDATES,
        key=lambda name: float(
            result["selection_validation_only"]["models"][name]["best_validation_recall@5"]
        ),
    )
    result["selection_validation_only"]["selected"] = selected_gnn
    result["selection_validation_only"]["tie_break_order"] = list(GNN_CANDIDATES)
    result["models"] = {
        "plain_mlp": {"uses_topology": False, "seeds": {}},
        selected_gnn: {"uses_topology": True, "seeds": {}},
    }
    checkpoint()

    for seed in args.seeds:
        for model_name in ("plain_mlp", selected_gnn):
            if seed == args.selection_seed and model_name == selected_gnn:
                seed_everything(seed)
                model = build_operator_model(
                    model_name,
                    dataset.feature_dim,
                    args.hidden_dim,
                    layers=args.layers,
                    dropout=args.dropout,
                    temperature=args.temperature,
                ).to(device)
                model.load_state_dict(selection_states[model_name])
                training = {
                    **{
                        key: result["selection_validation_only"]["models"][model_name][key]
                        for key in ("best_validation_recall@5", "history", "training_seconds")
                    },
                    "reused_validation_selected_checkpoint": True,
                    "train_queries": len(splits[QuerySplit.TRAIN]),
                    "train_queries_with_candidate_gold": sum(
                        query.relevant_local.numel() > 0 for query in splits[QuerySplit.TRAIN]
                    ),
                }
                counts = model_parameter_counts(model)
            else:
                model, training, counts = _fit(
                    model_name,
                    seed,
                    dataset,
                    splits,
                    nodes,
                    query_embeddings,
                    local_edges,
                    device,
                    args,
                )
            metrics, rows, inference = _repeated_score(
                model,
                splits[QuerySplit.TEST],
                nodes,
                query_embeddings,
                local_edges,
                device,
                batch_size=args.batch_size,
                ks=tuple(args.ks),
                repeats=args.inference_repeats,
            )
            result["models"][model_name]["parameters"] = counts
            result["models"][model_name]["seeds"][str(seed)] = {
                "metrics": metrics,
                "training": training,
                "inference": inference,
                "by_hop": _hop_aggregates(splits[QuerySplit.TEST], rows),
                "per_query": _compact_per_query(splits[QuerySplit.TEST], rows),
            }
            checkpoint()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for model_name in ("plain_mlp", selected_gnn):
        result["models"][model_name]["aggregate"] = _aggregate_runs(
            result["models"][model_name]["seeds"]
        )
    result["paired_mlp_minus_gnn"] = _paired_gaps(result["models"], selected_gnn, args.seeds)
    result["status"] = "PAPER_MAIN_TABLE_DATASET_COMPLETE"
    checkpoint()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topology-cache", type=Path, default=None)
    parser.add_argument("--required-hops", nargs="*", type=int, default=[])
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "selected_gnn": result["selection_validation_only"]["selected"],
                "paired_mlp_minus_gnn": result["paired_mlp_minus_gnn"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
