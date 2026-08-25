#!/usr/bin/env python
"""Run the gated seven-model clean operator screen on complete frozen data."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any
import warnings

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import (  # noqa: E402
    CompleteQuery,
    CompleteRetrievalDataset,
    load_complete_dataset,
)
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.operator_models import (  # noqa: E402
    SCREEN_MODELS,
    build_operator_model,
    model_parameter_counts,
)
from mp_retrieval.protocol import seed_everything, sha256_file  # noqa: E402
from mp_retrieval.topology_store import (  # noqa: E402
    PackedLocalTopologies,
    build_packed_topologies,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _limited_queries(
    queries: list[CompleteQuery],
    limit_per_split: int | None,
) -> list[CompleteQuery]:
    if limit_per_split is None:
        return queries
    selected: list[CompleteQuery] = []
    for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST):
        selected.extend(
            query for query in queries if query.split == int(split)
        )
        selected = [
            *[query for query in selected if query.split != int(split)],
            *[query for query in selected if query.split == int(split)][:limit_per_split],
        ]
    return sorted(selected, key=lambda query: query.query_index)


def _hash_tensor_contract(queries: list[CompleteQuery]) -> dict[str, str]:
    candidates = hashlib.sha256()
    labels = hashlib.sha256()
    splits = hashlib.sha256()
    for query in queries:
        candidates.update(query.query_id.encode("utf-8"))
        candidates.update(query.candidate_index.numpy().tobytes())
        labels.update(query.query_id.encode("utf-8"))
        labels.update(query.relevant_global.numpy().tobytes())
        splits.update(query.query_id.encode("utf-8"))
        splits.update(int(query.split).to_bytes(1, "little"))
    return {
        "candidates": candidates.hexdigest(),
        "labels": labels.hexdigest(),
        "splits": splits.hexdigest(),
    }


def _build_local_topologies(
    dataset: CompleteRetrievalDataset,
    queries: list[CompleteQuery],
) -> tuple[PackedLocalTopologies, float]:
    edges = build_packed_topologies(dataset, queries)
    return edges, edges.build_seconds


def _prepare_batch(
    batch: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    local_edges: Any,
    device: torch.device,
    *,
    uses_topology: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, list[int]]:
    lengths = [int(query.candidate_index.numel()) for query in batch]
    candidate_index = torch.cat([query.candidate_index for query in batch]).to(device)
    query_index = torch.tensor([query.query_index for query in batch], dtype=torch.long, device=device)
    anchor_index = torch.tensor([query.anchor_global for query in batch], dtype=torch.long, device=device)
    batch_index = torch.repeat_interleave(
        torch.arange(len(batch), device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )
    edge_index: torch.Tensor | None = None
    if uses_topology:
        if hasattr(local_edges, "batch_edge_index"):
            edge_index = local_edges.batch_edge_index(batch, lengths, device)
        else:
            shifted: list[torch.Tensor] = []
            offset = 0
            for query, length in zip(batch, lengths):
                edges = local_edges[query]
                if edges.numel():
                    shifted.append(edges + offset)
                offset += length
            edge_index = (
                torch.cat(shifted, dim=1).to(device)
                if shifted
                else torch.empty((2, 0), dtype=torch.long, device=device)
            )
    return (
        node_embeddings[candidate_index],
        query_embeddings[query_index],
        node_embeddings[anchor_index],
        batch_index,
        edge_index,
        lengths,
    )


def _listwise_batch_loss(
    scores: torch.Tensor,
    batch: list[CompleteQuery],
    lengths: list[int],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    offset = 0
    for query, length in zip(batch, lengths):
        local_scores = scores[offset : offset + length]
        positives = query.relevant_local.to(scores.device)
        if positives.numel():
            target = torch.zeros_like(local_scores)
            target[positives] = 1.0 / positives.numel()
            losses.append(-(F.log_softmax(local_scores, dim=0) * target).sum())
        offset += length
    if not losses:
        raise RuntimeError("Training batch has no in-pool gold candidates")
    return torch.stack(losses).mean()


def _metric_row(
    scores: torch.Tensor,
    query: CompleteQuery,
    ks: tuple[int, ...],
) -> dict[str, float | None]:
    order = torch.argsort(scores, descending=True).tolist()
    relevant = set(query.relevant_local.tolist())
    total_global = int(query.relevant_global.numel())
    total_available = len(relevant)
    first = next((rank + 1 for rank, candidate in enumerate(order) if candidate in relevant), None)
    row: dict[str, float | None] = {
        "candidate_ceiling": query.candidate_ceiling,
        "candidate_available": float(total_available > 0),
        "mrr": 0.0 if first is None else 1.0 / first,
        "conditional_mrr": None if total_available == 0 else (0.0 if first is None else 1.0 / first),
    }
    for k in ks:
        hits = len(relevant & set(order[:k]))
        row[f"recall@{k}"] = hits / total_global
        row[f"full_coverage@{k}"] = float(hits == total_global)
        row[f"conditional_recall@{k}"] = (
            None if total_available == 0 else hits / total_available
        )
        row[f"conditional_hit@{k}"] = None if total_available == 0 else float(hits > 0)
        row[f"conditional_full_coverage@{k}"] = (
            None if total_available == 0 else float(hits == total_available)
        )
    return row


def _aggregate_rows(rows: list[dict[str, float | None]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key in rows[0]:
        values = [float(row[key]) for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else float("nan")
        if key.startswith("conditional_"):
            result[f"{key}_queries"] = len(values)
    return result


def _score_queries(
    model: torch.nn.Module,
    queries: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    local_edges: dict[CompleteQuery, torch.Tensor],
    device: torch.device,
    *,
    batch_size: int,
    ks: tuple[int, ...],
    timed: bool,
) -> tuple[dict[str, float | int], list[dict[str, float | None]], dict[str, float]]:
    model.eval()
    uses_topology = bool(model.uses_topology)
    batches = [queries[start : start + batch_size] for start in range(0, len(queries), batch_size)]

    def forward(batch: list[CompleteQuery]) -> tuple[torch.Tensor, list[int]]:
        prepared = _prepare_batch(
            batch,
            node_embeddings,
            query_embeddings,
            local_edges,
            device,
            uses_topology=uses_topology,
        )
        nodes, query_vectors, anchors, batch_index, edge_index, lengths = prepared
        return model(nodes, query_vectors, anchors, batch_index, edge_index), lengths

    if timed and batches:
        with torch.no_grad():
            forward(batches[0])
        _synchronize(device)
    base_memory = 0
    if timed and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        base_memory = int(torch.cuda.memory_allocated(device))
    started = time.perf_counter()
    stored: list[tuple[list[CompleteQuery], torch.Tensor, list[int]]] = []
    with torch.no_grad():
        for batch in batches:
            scores, lengths = forward(batch)
            stored.append((batch, scores.detach().cpu(), lengths))
    _synchronize(device)
    elapsed = time.perf_counter() - started
    rows: list[dict[str, float | None]] = []
    for batch, scores, lengths in stored:
        offset = 0
        for query, length in zip(batch, lengths):
            rows.append(_metric_row(scores[offset : offset + length], query, ks))
            offset += length
    aggregate = _aggregate_rows(rows)
    peak_total = (
        int(torch.cuda.max_memory_allocated(device))
        if timed and device.type == "cuda"
        else 0
    )
    telemetry = {
        "inference_seconds": elapsed,
        "latency_ms_per_query": elapsed * 1000 / max(len(queries), 1),
        "throughput_queries_per_second": len(queries) / max(elapsed, 1e-12),
        "peak_gpu_memory_mb_total": peak_total / 2**20,
        "peak_gpu_memory_mb_incremental": max(peak_total - base_memory, 0) / 2**20,
    }
    return aggregate, rows, telemetry


def _train_model(
    model: torch.nn.Module,
    train_queries: list[CompleteQuery],
    validation_queries: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    local_edges: dict[CompleteQuery, torch.Tensor],
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    eligible = [query for query in train_queries if query.relevant_local.numel()]
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = -float("inf")
    history: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(epochs):
        order = eligible[:]
        random.Random(seed + epoch * 1_000_003).shuffle(order)
        model.train()
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            prepared = _prepare_batch(
                batch,
                node_embeddings,
                query_embeddings,
                local_edges,
                device,
                uses_topology=bool(model.uses_topology),
            )
            nodes, query_vectors, anchors, batch_index, edge_index, lengths = prepared
            scores = model(nodes, query_vectors, anchors, batch_index, edge_index)
            loss = _listwise_batch_loss(scores, batch, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _rows, _telemetry = _score_queries(
            model,
            validation_queries,
            node_embeddings,
            query_embeddings,
            local_edges,
            device,
            batch_size=batch_size,
            ks=(5,),
            timed=False,
        )
        value = float(validation["recall@5"])
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "validation_recall@5": value,
            }
        )
        if value > best_validation:
            best_validation = value
            best_state = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(best_state)
    return model, {
        "training_seconds": elapsed,
        "peak_training_gpu_memory_mb_total": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        ),
        "best_validation_recall@5": best_validation,
        "history": history,
        "train_queries": len(train_queries),
        "train_queries_with_candidate_gold": len(eligible),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    queries = _limited_queries(dataset.queries, args.limit_per_split)
    splits = {
        split: [query for query in queries if query.split == int(split)]
        for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST)
    }
    if any(not values for values in splits.values()):
        raise RuntimeError("The screen requires non-empty train, validation, and test splits")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        nodes = torch.from_numpy(np.asarray(dataset.node_array)).to(
            device=device, dtype=torch.float32
        )
        query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(
            device=device, dtype=torch.float32
        )
    local_edges, topology_seconds = _build_local_topologies(dataset, queries)
    model_results: dict[str, Any] = {}
    for model_index, name in enumerate(args.models):
        seed_everything(args.seed)
        model = build_operator_model(
            name,
            dataset.feature_dim,
            args.hidden_dim,
            layers=args.layers,
            offset_directions=args.offset_directions,
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
            seed=args.seed,
        )
        metrics, per_query, inference = _score_queries(
            trained,
            splits[QuerySplit.TEST],
            nodes,
            query_embeddings,
            local_edges,
            device,
            batch_size=args.batch_size,
            ks=tuple(args.ks),
            timed=True,
        )
        model_results[name] = {
            "metrics": metrics,
            "parameters": counts,
            "training": training,
            "inference": inference,
            "per_query": {
                query.query_id: row
                for query, row in zip(splits[QuerySplit.TEST], per_query)
            },
            "uses_topology": bool(model.uses_topology),
            "screen_order": model_index,
        }
        del trained, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    gnn_names = [name for name in ("gcn", "sage", "gat", "gin") if name in model_results]
    best_gnn = max(
        gnn_names,
        key=lambda name: float(model_results[name]["training"]["best_validation_recall@5"]),
    )
    gaps: dict[str, dict[str, float]] = {}
    for offset_name in ("offset_mlp", "offset_mlp_k4"):
        if offset_name not in model_results:
            continue
        gaps[offset_name] = {
            metric: float(model_results[offset_name]["metrics"][metric])
            - float(model_results[best_gnn]["metrics"][metric])
            for metric in ("recall@1", "recall@5", "recall@20", "mrr")
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
        "status": "SCREENING_ONLY_NOT_PAPER_FINAL",
        "dataset": dataset.dataset,
        "data": {
            "queries": len(queries),
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
            "same_seed": args.seed,
            "same_hyperparameter_budget": True,
            "offset_inference_uses_adjacency": False,
            "gnn_only_privileged_input": "candidate-induced adjacency",
            "sha256": _hash_tensor_contract(queries),
        },
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "models": model_results,
        "best_gnn_selected_by_validation_recall@5": best_gnn,
        "offset_minus_best_gnn_test": gaps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=SCREEN_MODELS, default=list(SCREEN_MODELS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--offset-directions", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--nodes-sha256", default="computed-by-launcher")
    args = parser.parse_args()
    result = run(args)
    compact = {
        name: {
            "R@1": values["metrics"]["recall@1"],
            "R@5": values["metrics"]["recall@5"],
            "R@20": values["metrics"]["recall@20"],
            "MRR": values["metrics"]["mrr"],
            "params": values["parameters"]["parameters"],
            "latency_ms": values["inference"]["latency_ms_per_query"],
            "memory_mb": values["inference"]["peak_gpu_memory_mb_total"],
        }
        for name, values in result["models"].items()
    }
    print(json.dumps({"status": result["status"], "dataset": result["dataset"], "table": compact}, indent=2))


if __name__ == "__main__":
    main()
