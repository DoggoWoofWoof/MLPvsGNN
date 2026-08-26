#!/usr/bin/env python
"""Run the frozen five-seed SA-MLP confirmation and seed-prior controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
import warnings
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import CompleteQuery, load_complete_dataset  # noqa: E402
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.operator_models import (  # noqa: E402
    SEED_ONLY_MODEL,
    build_explicit_feature_mlp,
    build_seed_aware_message_passing,
    model_parameter_counts,
)
from mp_retrieval.protocol import seed_everything  # noqa: E402
from mp_retrieval.structural_features import (  # noqa: E402
    StructuralFeatureStore,
    build_or_load_structural_features,
)
from mp_retrieval.topology_store import PackedLocalTopologies  # noqa: E402
from scripts.run_confirmation import _aggregate_runs, _mean_std  # noqa: E402
from scripts.run_main_table import (  # noqa: E402
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
from scripts.run_sa_mlp_screen import _RSSSampler  # noqa: E402

MODEL_NAMES = ("sa_mlp", SEED_ONLY_MODEL, "seed_aware_gnn")
PACKED_QUERY_METRICS = (
    "recall@1",
    "recall@5",
    "recall@20",
    "mrr",
    "full_coverage@20",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _query_order_sha256(queries: list[CompleteQuery]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        digest.update(query.query_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _legacy_pre_hop_contract_sha256(queries: list[CompleteQuery]) -> str:
    """Reproduce the exact contract used by the legacy 2Wiki/MuSiQue artifacts.

    The only difference from the current contract is that the legacy digest
    predates optional hop metadata. Query IDs, split bytes, candidate IDs in
    stable order, and gold IDs are all still covered bit-for-bit.
    """

    digest = hashlib.sha256()
    for query in queries:
        digest.update(query.query_id.encode("utf-8"))
        digest.update(int(query.split).to_bytes(1, "little"))
        digest.update(query.candidate_index.numpy().tobytes())
        digest.update(query.relevant_global.numpy().tobytes())
    return digest.hexdigest()


def _candidate_order_sha256(queries: list[CompleteQuery]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        digest.update(query.query_id.encode("utf-8"))
        digest.update(int(query.candidate_index.numel()).to_bytes(4, "little"))
        digest.update(query.candidate_index.numpy().tobytes())
    return digest.hexdigest()


def validate_candidate_contract(
    baseline: dict[str, Any],
    dataset,
    compatibility: str | None,
) -> dict[str, Any]:
    """Validate frozen candidates before topology, feature, or training work."""

    expected = baseline["candidate_contract_sha256"]
    current = dataset.metadata["candidate_contract_sha256"]
    mode = "current_with_hop_metadata"
    observed = current
    if current != expected:
        if compatibility != "pre_hop_metadata_v1":
            raise ValueError("Frozen baseline candidate contract does not match")
        observed = _legacy_pre_hop_contract_sha256(dataset.queries)
        mode = compatibility
        if observed != expected:
            raise ValueError(
                "Legacy compatibility check failed: query IDs, split, candidate order, "
                "or gold IDs differ from the frozen artifact"
            )
    proof = {
        "status": "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE",
        "mode": mode,
        "expected_contract_sha256": expected,
        "observed_contract_sha256": observed,
        "current_contract_with_hop_metadata_sha256": current,
        "candidate_id_order_sha256": _candidate_order_sha256(dataset.queries),
        "queries": len(dataset.queries),
        "candidate_rows": sum(int(query.candidate_index.numel()) for query in dataset.queries),
        "covered_fields": [
            "query_id",
            "split",
            "candidate_ids_in_stable_order",
            "gold_ids",
        ],
        "ignored_legacy_field": (
            "hop_metadata_only" if mode != "current_with_hop_metadata" else None
        ),
    }
    proof["proof_sha256"] = hashlib.sha256(
        json.dumps(proof, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return proof


def _seed_indicator(
    batch: list[CompleteQuery],
    lengths: list[int],
    device: torch.device,
) -> torch.Tensor:
    indicator = torch.zeros((sum(lengths), 1), dtype=torch.float32, device=device)
    offset = 0
    for query, length in zip(batch, lengths):
        if query.retrieval_seed_local is None:
            raise ValueError("Frozen retrieval seed positions are absent")
        if query.retrieval_seed_local.numel():
            positions = query.retrieval_seed_local.to(device=device) + offset
            indicator[positions, 0] = 1.0
        offset += length
    return indicator


def _prepare_batch(
    batch: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    topologies: PackedLocalTopologies,
    features: StructuralFeatureStore,
    model_name: str,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    lengths = [int(query.candidate_index.numel()) for query in batch]
    candidate_index = torch.cat([query.candidate_index for query in batch]).to(device)
    query_index = torch.tensor(
        [query.query_index for query in batch], dtype=torch.long, device=device
    )
    batch_index = torch.repeat_interleave(
        torch.arange(len(batch), device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )
    nodes = node_embeddings[candidate_index]
    queries = query_embeddings[query_index]
    if model_name == "sa_mlp":
        structural = features.batch_features(
            batch,
            include_static=True,
            include_local=True,
            device=device,
        )
        return model.forward_explicit(nodes, queries, batch_index, structural), lengths
    seeds = _seed_indicator(batch, lengths, device)
    if model_name == SEED_ONLY_MODEL:
        return model.forward_explicit(nodes, queries, batch_index, seeds), lengths
    if model_name == "seed_aware_gnn":
        edge_index = topologies.batch_edge_index(batch, lengths, device)
        return (
            model.forward_seed_aware(nodes, queries, batch_index, edge_index, seeds),
            lengths,
        )
    raise ValueError(f"Unknown confirmation model: {model_name}")


def _score_once(
    model_name: str,
    model: torch.nn.Module,
    queries: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    topologies: PackedLocalTopologies,
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
            _prepare_batch(
                batches[0],
                node_embeddings,
                query_embeddings,
                topologies,
                features,
                model_name,
                model,
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
            scores, lengths = _prepare_batch(
                batch,
                node_embeddings,
                query_embeddings,
                topologies,
                features,
                model_name,
                model,
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
    metrics = _aggregate_rows(rows)
    peak_gpu = (
        int(torch.cuda.max_memory_allocated(device)) if timed and device.type == "cuda" else 0
    )
    return (
        metrics,
        rows,
        {
            "inference_seconds": elapsed,
            "latency_ms_per_query": elapsed * 1000 / max(len(queries), 1),
            "throughput_queries_per_second": len(queries) / max(elapsed, 1e-12),
            "peak_gpu_memory_mb_total": peak_gpu / 2**20,
            "peak_gpu_memory_mb_incremental": max(peak_gpu - base_gpu, 0) / 2**20,
            "peak_cpu_rss_mb_total": rss.peak / 2**20,
            "peak_cpu_rss_mb_incremental": max(rss.peak - rss.baseline, 0) / 2**20,
        },
    )


def _repeated_score(
    model_name: str,
    model: torch.nn.Module,
    queries: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    topologies: PackedLocalTopologies,
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
            model_name,
            model,
            queries,
            node_embeddings,
            query_embeddings,
            topologies,
            features,
            device,
            batch_size=batch_size,
            ks=ks,
            timed=True,
        )
        telemetry.append(current)
    assert metrics is not None and rows is not None
    return (
        metrics,
        rows,
        {
            "repeats": repeats,
            "latency_ms_per_query": statistics.median(
                item["latency_ms_per_query"] for item in telemetry
            ),
            "throughput_queries_per_second": statistics.median(
                item["throughput_queries_per_second"] for item in telemetry
            ),
            "peak_gpu_memory_mb_total": max(item["peak_gpu_memory_mb_total"] for item in telemetry),
            "peak_gpu_memory_mb_incremental": max(
                item["peak_gpu_memory_mb_incremental"] for item in telemetry
            ),
            "peak_cpu_rss_mb_total": max(item["peak_cpu_rss_mb_total"] for item in telemetry),
            "peak_cpu_rss_mb_incremental": max(
                item["peak_cpu_rss_mb_incremental"] for item in telemetry
            ),
            "repeat_telemetry": telemetry,
        },
    )


def _fit(
    model_name: str,
    model: torch.nn.Module,
    train_queries: list[CompleteQuery],
    validation_queries: list[CompleteQuery],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    topologies: PackedLocalTopologies,
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
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=weight_decay)
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
            scores, lengths = _prepare_batch(
                batch,
                node_embeddings,
                query_embeddings,
                topologies,
                features,
                model_name,
                model,
                device,
            )
            loss = _listwise_batch_loss(scores, batch, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _rows, _telemetry = _score_once(
            model_name,
            model,
            validation_queries,
            node_embeddings,
            query_embeddings,
            topologies,
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


def _build_model(
    model_name: str,
    dataset,
    features: StructuralFeatureStore,
    selected_gnn: str,
    target_parameters: int,
    args: argparse.Namespace,
) -> torch.nn.Module:
    if model_name == "sa_mlp":
        return build_explicit_feature_mlp(
            "sa_mlp",
            dataset.feature_dim,
            args.projection_dim,
            static_dim=features.static_dim,
            local_dim=features.local_dim,
            target_parameters=target_parameters,
            dropout=args.dropout,
            temperature=args.temperature,
        )
    if model_name == SEED_ONLY_MODEL:
        return build_explicit_feature_mlp(
            SEED_ONLY_MODEL,
            dataset.feature_dim,
            args.projection_dim,
            static_dim=0,
            local_dim=1,
            target_parameters=target_parameters,
            dropout=args.dropout,
            temperature=args.temperature,
        )
    if model_name == "seed_aware_gnn":
        return build_seed_aware_message_passing(
            selected_gnn,
            dataset.feature_dim,
            args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            temperature=args.temperature,
        )
    raise ValueError(f"Unknown confirmation model: {model_name}")


def _rows_to_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row[metric]) for metric in PACKED_QUERY_METRICS] for row in rows],
        dtype=np.float32,
    )


def _screen_seed_rows(screen: dict[str, Any], test_queries: list[CompleteQuery]) -> np.ndarray:
    packed = screen["models"]["sa_mlp"]["per_query"]
    return np.asarray(
        [
            [float(packed[query.query_id][metric]) for metric in PACKED_QUERY_METRICS]
            for query in test_queries
        ],
        dtype=np.float32,
    )


def _paired_contrasts(result: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    baseline_gnn = result["baseline"]["selected_gnn"]["seeds"]
    models = result["models"]
    pairs = {
        "sa_mlp_minus_seed_aware_gnn": (
            models["sa_mlp"]["seeds"],
            models["seed_aware_gnn"]["seeds"],
        ),
        "sa_mlp_minus_seed_only": (models["sa_mlp"]["seeds"], models[SEED_ONLY_MODEL]["seeds"]),
        "seed_aware_gnn_minus_frozen_gnn": (models["seed_aware_gnn"]["seeds"], baseline_gnn),
    }
    output: dict[str, Any] = {}
    for contrast, (left, right) in pairs.items():
        output[contrast] = {}
        for metric in PACKED_QUERY_METRICS:
            values = [
                float(left[str(seed)]["metrics"][metric])
                - float(right[str(seed)]["metrics"][metric])
                for seed in seeds
            ]
            output[contrast][metric] = {
                "by_seed": dict(zip(map(str, seeds), values)),
                **_mean_std(values),
            }
    return output


def _reuse_screen_seed(
    args: argparse.Namespace,
    dataset,
    features: StructuralFeatureStore,
    target_parameters: int,
) -> tuple[dict[str, Any], np.ndarray] | None:
    screen = args.screen_seed_0
    if screen is None:
        return None
    if screen.get("status") != "SA_MLP_SCREEN_DATASET_COMPLETE":
        raise ValueError("Reusable screen result is incomplete")
    if screen["dataset"] != dataset.dataset:
        raise ValueError("Reusable screen dataset does not match")
    if screen["data"]["candidate_contract_sha256"] != dataset.metadata["candidate_contract_sha256"]:
        raise ValueError("Reusable screen candidate contract does not match")
    if screen["feature_cache"]["contract_sha256"] != features.metadata["contract_sha256"]:
        raise ValueError("Reusable screen feature contract does not match")
    record = deepcopy(screen["models"]["sa_mlp"])
    if (
        abs(int(record["parameters"]["parameters"]) - target_parameters)
        > args.max_parameter_difference
    ):
        raise ValueError("Reusable SA-MLP parameter match is outside the frozen tolerance")
    rows = _screen_seed_rows(screen, dataset.split(QuerySplit.TEST))
    record.pop("per_query", None)
    record["reused_from_frozen_screen"] = True
    record["source_result_sha256"] = args.screen_result_sha256
    return record, rows


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
    if args.baseline["dataset"] != dataset.dataset:
        raise ValueError("Frozen baseline dataset does not match")
    candidate_contract = validate_candidate_contract(
        args.baseline,
        dataset,
        args.candidate_contract_compatibility,
    )
    if (
        args.candidate_contract_proof_sha256 is not None
        and candidate_contract["proof_sha256"] != args.candidate_contract_proof_sha256
    ):
        raise ValueError("Preflight candidate-contract proof changed before training")
    if args.baseline["selected_gnn"]["model"] != args.selected_gnn:
        raise ValueError("Frozen selected GNN family does not match the confirmation protocol")
    observed_hops = sorted({query.hop for query in dataset.queries if query.hop is not None})
    if observed_hops != sorted(args.required_hops):
        raise ValueError(f"Expected hop labels {args.required_hops}, got {observed_hops}")
    splits = {
        split: dataset.split(split)
        for split in (QuerySplit.TRAIN, QuerySplit.VALIDATION, QuerySplit.TEST)
    }
    if any(not rows for rows in splits.values()):
        raise RuntimeError("Confirmation requires non-empty canonical splits")
    topologies, topology_metadata = _load_or_build_topologies(dataset, args.topology_cache)
    features = build_or_load_structural_features(
        dataset,
        topologies,
        args.feature_cache,
        source_fingerprint=args.data_fingerprint_sha256,
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
    result: dict[str, Any] = {
        "status": "SA_MLP_CONFIRMATION_IN_PROGRESS",
        "dataset": dataset.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "baseline_result_sha256": args.baseline_result_sha256,
        "data": {
            "queries": len(dataset.queries),
            "nodes": dataset.num_nodes,
            "edges": dataset.metadata["num_edges"],
            "splits": {split.name.lower(): len(rows) for split, rows in splits.items()},
            "candidate_contract_sha256": dataset.metadata["candidate_contract_sha256"],
            "test_query_order_sha256": _query_order_sha256(splits[QuerySplit.TEST]),
            "topology": topology_metadata,
        },
        "comparison_contract": {
            "same_frozen_embeddings_candidates_labels_loss_splits": True,
            "retrieval_seed_labels_used": False,
            "sa_architecture_changed_after_screen": False,
            "seed_only_receives_adjacency": False,
            "seed_aware_gnn_extra_input": "binary_retrieval_seed_membership_only",
            "seed_aware_gnn_extra_parameters": args.hidden_dim,
            "sha256": _hash_tensor_contract(dataset.queries),
            "candidate_compatibility_proof": candidate_contract,
        },
        "baseline": args.baseline,
        "feature_cache": features.metadata,
        "models": {name: {"seeds": {}} for name in MODEL_NAMES},
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in {"baseline", "feature_config", "screen_seed_0"}
        },
    }
    query_arrays: dict[str, np.ndarray] = {}

    def checkpoint() -> None:
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    reused = _reuse_screen_seed(args, dataset, features, target_parameters)
    if reused is not None:
        record, query_array = reused
        result["models"]["sa_mlp"]["parameters"] = record["parameters"]
        result["models"]["sa_mlp"]["head_dim"] = record["head_dim"]
        result["models"]["sa_mlp"]["seeds"]["0"] = record
        query_arrays["sa_mlp_seed_0"] = query_array
        checkpoint()

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
            if model_name != "seed_aware_gnn":
                difference = abs(int(counts["parameters"]) - target_parameters)
                if difference > args.max_parameter_difference:
                    raise ValueError(
                        f"{model_name} differs from frozen GNN by {difference} parameters"
                    )
            else:
                frozen_count = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
                if int(counts["parameters"]) - frozen_count != args.hidden_dim:
                    raise ValueError(
                        "Seed-aware GNN must add exactly one scalar weight per channel"
                    )
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
                splits[QuerySplit.TEST],
                nodes,
                query_embeddings,
                topologies,
                features,
                device,
                batch_size=args.batch_size,
                ks=tuple(args.ks),
                repeats=args.inference_repeats,
            )
            result["models"][model_name]["parameters"] = counts
            if model_name != "seed_aware_gnn":
                result["models"][model_name]["head_dim"] = int(model.scorer[0].out_features)
            result["models"][model_name]["seeds"][seed_key] = {
                "metrics": metrics,
                "training": training,
                "inference": inference,
                "by_hop": _hop_aggregates(splits[QuerySplit.TEST], rows),
                "checkpoint_sha256": _state_sha256(model.state_dict()),
                "reused_from_frozen_screen": False,
            }
            query_arrays[f"{model_name}_seed_{seed}"] = _rows_to_array(rows)
            checkpoint()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for model_name in MODEL_NAMES:
        runs = result["models"][model_name]["seeds"]
        if sorted(map(int, runs)) != sorted(args.seeds):
            raise RuntimeError(f"Incomplete seed set for {model_name}")
        result["models"][model_name]["aggregate"] = _aggregate_runs(runs)
    result["paired_contrasts"] = _paired_contrasts(result, args.seeds)
    args.query_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.query_metrics_output,
        metric_names=np.asarray(PACKED_QUERY_METRICS),
        query_order_sha256=np.asarray(result["data"]["test_query_order_sha256"]),
        **query_arrays,
    )
    result["query_metrics"] = {
        "path": str(args.query_metrics_output),
        "sha256": _sha256(args.query_metrics_output),
        "format": "npz_float32_test_query_order_v1",
        "metrics": list(PACKED_QUERY_METRICS),
        "arrays": sorted(query_arrays),
    }
    result["status"] = "SA_MLP_CONFIRMATION_DATASET_COMPLETE"
    checkpoint()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-metrics-output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-result-sha256", required=True)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--selected-gnn", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--topology-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--screen-seed-0", type=Path, default=None)
    parser.add_argument("--screen-result-sha256", default=None)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--candidate-contract-proof-sha256", default=None)
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
    args.baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    args.screen_seed_0 = (
        None
        if args.screen_seed_0 is None
        else json.loads(args.screen_seed_0.read_text(encoding="utf-8"))
    )
    config = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    screen_config = __import__("yaml").safe_load(
        (REPO_ROOT / "configs" / "sa_mlp_screen.yaml").read_text(encoding="utf-8")
    )
    args.feature_config = {
        "retrieval_seeds": screen_config["retrieval_seeds"],
        "static_features": screen_config["static_features"],
        "query_local_features": screen_config["query_local_features"],
        "preprocessing": {"query_chunk_size": 8192},
    }
    del config
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "paired_contrasts": result["paired_contrasts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
