#!/usr/bin/env python
"""Run the preregistered one-seed fixed-structure MLP screen."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import warnings
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import CompleteQuery, load_complete_dataset  # noqa: E402
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.operator_models import (  # noqa: E402
    STRUCTURE_AWARE_MODELS,
    build_explicit_feature_mlp,
    model_parameter_counts,
)
from mp_retrieval.protocol import seed_everything  # noqa: E402
from mp_retrieval.structural_features import (  # noqa: E402
    StructuralFeatureStore,
    build_or_load_structural_features,
)
from scripts.run_main_table import (  # noqa: E402
    _compact_per_query,
    _hop_aggregates,
    _load_or_build_topologies,
    _state_sha256,
)
from scripts.run_operator_screen import (  # noqa: E402
    _aggregate_rows,
    _hash_tensor_contract,
    _listwise_batch_loss,
    _metric_row,
    _synchronize,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


class _RSSSampler:
    def __init__(self, interval_seconds: float = 0.005):
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _RSSSampler:
        self.baseline = int(self.process.memory_info().rss)
        self.peak = self.baseline

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.peak = max(self.peak, int(self.process.memory_info().rss))

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.peak = max(self.peak, int(self.process.memory_info().rss))
        self._stop.set()
        assert self._thread is not None
        self._thread.join()


def _prepare_batch(
    batch: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    features: StructuralFeatureStore,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, list[int]]:
    lengths = [int(query.candidate_index.numel()) for query in batch]
    candidate_index = torch.cat([query.candidate_index for query in batch]).to(device)
    query_index = torch.tensor(
        [query.query_index for query in batch], dtype=torch.long, device=device
    )
    batch_index = torch.repeat_interleave(
        torch.arange(len(batch), device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )
    structural = features.batch_features(
        batch,
        include_static=bool(model.include_static),
        include_local=bool(model.include_local),
        device=device,
    )
    return (
        node_embeddings[candidate_index],
        query_embeddings[query_index],
        batch_index,
        structural,
        lengths,
    )


def _forward_batch(
    model: torch.nn.Module,
    batch: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    features: StructuralFeatureStore,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    nodes, queries, batch_index, structural, lengths = _prepare_batch(
        batch,
        node_embeddings,
        query_embeddings,
        features,
        model,
        device,
    )
    return model.forward_explicit(nodes, queries, batch_index, structural), lengths


def _score_once(
    model: torch.nn.Module,
    queries: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    features: StructuralFeatureStore,
    device: torch.device,
    *,
    batch_size: int,
    ks: tuple[int, ...],
    timed: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    model.eval()
    batches = [queries[start : start + batch_size] for start in range(0, len(queries), batch_size)]
    if timed and batches:
        with torch.no_grad():
            _forward_batch(
                model,
                batches[0],
                node_embeddings,
                query_embeddings,
                features,
                device,
            )
        _synchronize(device)
    base_gpu = 0
    if timed and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        base_gpu = int(torch.cuda.memory_allocated(device))
    stored: list[tuple[list[CompleteQuery], torch.Tensor, list[int]]] = []
    started = time.perf_counter()
    with _RSSSampler() as rss, torch.no_grad():
        for batch in batches:
            scores, lengths = _forward_batch(
                model,
                batch,
                node_embeddings,
                query_embeddings,
                features,
                device,
            )
            stored.append((batch, scores.detach().cpu(), lengths))
        _synchronize(device)
    elapsed = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    for batch, scores, lengths in stored:
        offset = 0
        for query, length in zip(batch, lengths):
            rows.append(_metric_row(scores[offset : offset + length], query, ks))
            offset += length
    aggregate = _aggregate_rows(rows)
    peak_gpu = (
        int(torch.cuda.max_memory_allocated(device))
        if timed and device.type == "cuda"
        else 0
    )
    return aggregate, rows, {
        "inference_seconds": elapsed,
        "latency_ms_per_query": elapsed * 1000 / max(len(queries), 1),
        "throughput_queries_per_second": len(queries) / max(elapsed, 1e-12),
        "peak_gpu_memory_mb_total": peak_gpu / 2**20,
        "peak_gpu_memory_mb_incremental": max(peak_gpu - base_gpu, 0) / 2**20,
        "peak_cpu_rss_mb_total": rss.peak / 2**20,
        "peak_cpu_rss_mb_incremental": max(rss.peak - rss.baseline, 0) / 2**20,
    }


def _repeated_score(
    model: torch.nn.Module,
    queries: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    features: StructuralFeatureStore,
    device: torch.device,
    *,
    batch_size: int,
    ks: tuple[int, ...],
    repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metrics: dict[str, Any] | None = None
    rows: list[dict[str, Any]] | None = None
    telemetry: list[dict[str, float]] = []
    for _ in range(repeats):
        metrics, rows, current = _score_once(
            model,
            queries,
            node_embeddings,
            query_embeddings,
            features,
            device,
            batch_size=batch_size,
            ks=ks,
            timed=True,
        )
        telemetry.append(current)
    assert metrics is not None and rows is not None
    return metrics, rows, {
        "repeats": repeats,
        "latency_ms_per_query": statistics.median(
            item["latency_ms_per_query"] for item in telemetry
        ),
        "throughput_queries_per_second": statistics.median(
            item["throughput_queries_per_second"] for item in telemetry
        ),
        "peak_gpu_memory_mb_total": max(
            item["peak_gpu_memory_mb_total"] for item in telemetry
        ),
        "peak_gpu_memory_mb_incremental": max(
            item["peak_gpu_memory_mb_incremental"] for item in telemetry
        ),
        "peak_cpu_rss_mb_total": max(
            item["peak_cpu_rss_mb_total"] for item in telemetry
        ),
        "peak_cpu_rss_mb_incremental": max(
            item["peak_cpu_rss_mb_incremental"] for item in telemetry
        ),
        "repeat_telemetry": telemetry,
    }


def _fit(
    model: torch.nn.Module,
    train_queries: list[CompleteQuery],
    validation_queries: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    features: StructuralFeatureStore,
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
            scores, lengths = _forward_batch(
                model,
                batch,
                node_embeddings,
                query_embeddings,
                features,
                device,
            )
            loss = _listwise_batch_loss(scores, batch, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _rows, _telemetry = _score_once(
            model,
            validation_queries,
            node_embeddings,
            query_embeddings,
            features,
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
                {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
            )
    _synchronize(device)
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    return model, {
        "training_seconds": time.perf_counter() - started,
        "peak_training_gpu_memory_mb_total": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        ),
        "best_validation_recall@5": best_validation,
        "history": history,
        "train_queries": len(train_queries),
        "train_queries_with_candidate_gold": len(eligible),
    }


def _baseline_contract(baseline: dict[str, Any], dataset) -> None:
    if baseline["dataset"] != dataset.dataset:
        raise ValueError("Frozen baseline dataset does not match the requested dataset")
    if baseline["candidate_contract_sha256"] != dataset.metadata["candidate_contract_sha256"]:
        raise ValueError("Frozen baseline candidate contract does not match the loaded data")
    if int(baseline["queries"]) != len(dataset.queries):
        raise ValueError("Frozen baseline query count does not match the loaded data")


def _gap_closure(
    plain: float,
    gnn: float,
    sa: float,
) -> float:
    gap = gnn - plain
    if gap <= 0:
        raise ValueError("SA-MLP screening requires a positive frozen GNN gap")
    return (sa - plain) / gap


def run(
    args: argparse.Namespace,
    checkpoint_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    seed_everything(args.seed)
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError(
            f"Expected {args.expected_queries} queries for {args.dataset}, "
            f"got {len(dataset.queries)}"
        )
    _baseline_contract(args.baseline, dataset)
    splits = {
        split: dataset.split(split)
        for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST)
    }
    if any(not values for values in splits.values()):
        raise RuntimeError("The SA-MLP screen requires non-empty canonical splits")
    observed_hops = sorted({query.hop for query in dataset.queries if query.hop is not None})
    if sorted(args.required_hops) != observed_hops:
        raise ValueError(f"Expected hop labels {args.required_hops}, got {observed_hops}")
    local_edges, topology = _load_or_build_topologies(dataset, args.topology_cache)
    features = build_or_load_structural_features(
        dataset,
        local_edges,
        args.feature_cache,
        source_fingerprint=args.baseline["data_fingerprint_sha256"],
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
    target_parameters = int(args.baseline["gnn"]["parameters"]["parameters"])
    result: dict[str, Any] = {
        "status": "SA_MLP_SCREEN_IN_PROGRESS",
        "dataset": dataset.dataset,
        "seed": args.seed,
        "data_fingerprint_sha256": args.baseline["data_fingerprint_sha256"],
        "baseline_result_sha256": args.baseline_result_sha256,
        "data": {
            "queries": len(dataset.queries),
            "nodes": dataset.num_nodes,
            "edges": dataset.metadata["num_edges"],
            "splits": {split.name.lower(): len(values) for split, values in splits.items()},
            "candidate_contract_sha256": dataset.metadata["candidate_contract_sha256"],
            "topology": topology,
        },
        "comparison_contract": {
            "baseline_immutable": True,
            "same_frozen_query_embeddings": True,
            "same_frozen_node_embeddings": True,
            "same_candidate_pool": True,
            "same_labels_and_loss": True,
            "same_canonical_splits": True,
            "same_seed_and_epoch_budget": True,
            "retrieval_seeds_use_labels": False,
            "learned_model_receives_adjacency": False,
            "sha256": _hash_tensor_contract(dataset.queries),
        },
        "baseline_seed_0": args.baseline,
        "feature_cache": features.metadata,
        "models": {},
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in {"baseline", "feature_config"}
        },
    }

    def checkpoint() -> None:
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    for name in STRUCTURE_AWARE_MODELS:
        seed_everything(args.seed)
        model = build_explicit_feature_mlp(
            name,
            dataset.feature_dim,
            args.projection_dim,
            static_dim=features.static_dim,
            local_dim=features.local_dim,
            target_parameters=target_parameters,
            dropout=args.dropout,
            temperature=args.temperature,
        )
        counts = model_parameter_counts(model)
        difference = abs(int(counts["parameters"]) - target_parameters)
        if difference > args.max_parameter_difference:
            raise ValueError(
                f"{name} differs from the frozen GNN by {difference} parameters"
            )
        model, training = _fit(
            model,
            splits[QuerySplit.TRAIN],
            splits[QuerySplit.VALIDATION],
            nodes,
            query_embeddings,
            features,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        metrics, rows, inference = _repeated_score(
            model,
            splits[QuerySplit.TEST],
            nodes,
            query_embeddings,
            features,
            device,
            batch_size=args.batch_size,
            ks=tuple(args.ks),
            repeats=args.inference_repeats,
        )
        result["models"][name] = {
            "uses_topology_in_learned_model": False,
            "uses_interactions": bool(model.include_interactions),
            "uses_static_features": bool(model.include_static),
            "uses_query_local_features": bool(model.include_local),
            "head_dim": int(model.scorer[0].out_features),
            "parameters": counts,
            "parameter_difference_from_frozen_gnn": int(
                counts["parameters"] - target_parameters
            ),
            "checkpoint_sha256": _state_sha256(model.state_dict()),
            "training": training,
            "metrics": metrics,
            "inference": inference,
            "by_hop": _hop_aggregates(splits[QuerySplit.TEST], rows),
            "per_query": _compact_per_query(splits[QuerySplit.TEST], rows),
        }
        checkpoint()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plain_r5 = float(args.baseline["plain_mlp"]["metrics"]["recall@5"])
    gnn_r5 = float(args.baseline["gnn"]["metrics"]["recall@5"])
    sa_r5 = float(result["models"]["sa_mlp"]["metrics"]["recall@5"])
    closure = _gap_closure(plain_r5, gnn_r5, sa_r5)
    result["gap_closure"] = {
        "metric": "recall@5",
        "plain_mlp_seed_0": plain_r5,
        "gnn_seed_0": gnn_r5,
        "sa_mlp_seed_0": sa_r5,
        "frozen_gnn_gap": gnn_r5 - plain_r5,
        "recovered_gap": sa_r5 - plain_r5,
        "fraction": closure,
        "threshold": args.gap_closure_threshold,
        "dataset_pass": closure >= args.gap_closure_threshold,
    }
    result["status"] = "SA_MLP_SCREEN_DATASET_COMPLETE"
    checkpoint()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--topology-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--baseline-result-sha256", required=True)
    parser.add_argument("--required-hops", nargs="*", type=int, default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--max-parameter-difference", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument("--gap-closure-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    config = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    args.feature_config = {
        "retrieval_seeds": config["retrieval_seeds"],
        "static_features": config["static_features"],
        "query_local_features": config["query_local_features"],
    }
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "gap_closure": result["gap_closure"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
