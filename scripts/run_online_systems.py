#!/usr/bin/env python
"""Benchmark cache-disabled post-retrieval serving from unseen embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.online_serving import (
    OnlineBatch,
    build_online_queries,
    fuse_equal_rrf_candidates,
)
from mp_retrieval.structural_features import (
    StructuralFeatureStore,
    compute_query_local_features,
)
from mp_retrieval.topology_store import (
    PackedLocalTopologies,
    build_packed_topologies,
)
from scripts.run_edge_provenance import _atomic_json, _sha256
from scripts.run_sa_mlp_confirmation import (
    _build_model,
    _seed_indicator,
)

MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")
STAGE_NAMES = (
    "fusion_and_seed_ms",
    "topology_induction_ms",
    "query_local_summary_ms",
    "gather_transfer_forward_topk_ms",
    "total_ms",
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _selected_test_indices(dataset, count: int) -> np.ndarray:
    test = dataset.split(QuerySplit.TEST)
    available = np.asarray([query.query_index for query in test], dtype=np.int64)
    if available.size <= count:
        return available
    positions = np.linspace(0, available.size - 1, count, dtype=np.int64)
    return available[positions]


def _transient_batch(
    dataset,
    source_indices: np.ndarray,
    dense_rows: np.ndarray,
    splade_rows: np.ndarray,
    query_rows: np.ndarray,
    *,
    budget: int,
    rrf_constant: int,
) -> tuple[OnlineBatch, dict[str, float]]:
    started = time.perf_counter()
    candidate_rows = fuse_equal_rrf_candidates(
        dense_rows,
        splade_rows,
        budget=budget,
        constant=rrf_constant,
    )
    queries = build_online_queries(candidate_rows, dense_rows, splade_rows)
    fusion_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    topologies = build_packed_topologies(dataset, queries)
    topology_ms = (time.perf_counter() - started) * 1000
    return (
        OnlineBatch(
            source_query_indices=source_indices,
            query_embeddings=query_rows,
            queries=queries,
            topologies=topologies,
        ),
        {
            "fusion_and_seed_ms": fusion_ms,
            "topology_induction_ms": topology_ms,
        },
    )


def _forward_online(
    model_name: str,
    model: torch.nn.Module,
    batch: OnlineBatch,
    node_embeddings: torch.Tensor,
    static_features: torch.Tensor,
    device: torch.device,
    *,
    local_features: np.ndarray | None,
    top_k: int,
) -> None:
    lengths = [int(query.candidate_index.numel()) for query in batch.queries]
    candidate_index = torch.cat([query.candidate_index for query in batch.queries]).to(device)
    batch_index = torch.repeat_interleave(
        torch.arange(len(batch.queries), device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )
    nodes = node_embeddings[candidate_index]
    queries = torch.from_numpy(batch.query_embeddings).to(device=device, dtype=torch.float32)
    if model_name == "sa_mlp":
        if local_features is None:
            raise ValueError("QLS online forward requires freshly computed local summaries")
        local = torch.from_numpy(local_features).to(device=device, dtype=torch.float32)
        structural = torch.cat((static_features[candidate_index], local), dim=1)
        scores = model.forward_explicit(nodes, queries, batch_index, structural)
    else:
        seeds = _seed_indicator(batch.queries, lengths, device)
        edge_index = batch.topologies.batch_edge_index(batch.queries, lengths, device)
        scores = model.forward_seed_aware(nodes, queries, batch_index, edge_index, seeds)
    offset = 0
    for length in lengths:
        torch.topk(scores[offset : offset + length], min(top_k, length), sorted=True)
        offset += length


def _execute(
    model_name: str,
    model: torch.nn.Module,
    dataset,
    source_indices: np.ndarray,
    dense_rows: np.ndarray,
    splade_rows: np.ndarray,
    query_rows: np.ndarray,
    node_embeddings: torch.Tensor,
    static_features: torch.Tensor,
    device: torch.device,
    *,
    budget: int,
    rrf_constant: int,
    ppr_damping: float,
    ppr_iterations: int,
    top_k: int,
) -> dict[str, float]:
    process = psutil.Process(os.getpid())
    _synchronize(device)
    gpu_start = (
        torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rss_samples = [process.memory_info().rss / 2**20]
    total_started = time.perf_counter()
    batch, timings = _transient_batch(
        dataset,
        source_indices,
        dense_rows,
        splade_rows,
        query_rows,
        budget=budget,
        rrf_constant=rrf_constant,
    )
    rss_samples.append(process.memory_info().rss / 2**20)
    if model_name == "sa_mlp":
        started = time.perf_counter()
        local = compute_query_local_features(
            batch.queries,
            batch.topologies,
            damping=ppr_damping,
            ppr_iterations=ppr_iterations,
        )
        # The frozen QLS checkpoint was trained against a float16 local cache.
        local = local.astype(np.float16).astype(np.float32)
        summary_ms = (time.perf_counter() - started) * 1000
        rss_samples.append(process.memory_info().rss / 2**20)
    else:
        local = None
        summary_ms = 0.0
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        _forward_online(
            model_name,
            model,
            batch,
            node_embeddings,
            static_features,
            device,
            local_features=local,
            top_k=top_k,
        )
    _synchronize(device)
    forward_ms = (time.perf_counter() - started) * 1000
    rss_samples.append(process.memory_info().rss / 2**20)
    gpu_peak = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else 0.0
    )
    return {
        **timings,
        "query_local_summary_ms": summary_ms,
        "gather_transfer_forward_topk_ms": forward_ms,
        "total_ms": (time.perf_counter() - total_started) * 1000,
        "peak_process_rss_mb": max(rss_samples),
        "peak_gpu_allocated_mb": gpu_peak,
        "incremental_peak_gpu_allocated_mb": max(gpu_peak - gpu_start, 0.0),
    }


def _parity_audit(
    dataset,
    source_indices: np.ndarray,
    dense: np.ndarray,
    splade: np.ndarray,
    queries: np.ndarray,
    cached_topologies: PackedLocalTopologies,
    cached_features: StructuralFeatureStore,
    *,
    budget: int,
    rrf_constant: int,
    damping: float,
    ppr_iterations: int,
) -> dict[str, Any]:
    dense_rows = np.asarray(dense[source_indices]).copy()
    splade_rows = np.asarray(splade[source_indices]).copy()
    query_rows = np.asarray(queries[source_indices], dtype=np.float32).copy()
    online, _timings = _transient_batch(
        dataset,
        source_indices,
        dense_rows,
        splade_rows,
        query_rows,
        budget=budget,
        rrf_constant=rrf_constant,
    )
    local = compute_query_local_features(
        online.queries,
        online.topologies,
        damping=damping,
        ppr_iterations=ppr_iterations,
    ).astype(np.float16)
    cursor = 0
    topology_equal = True
    local_equal = True
    max_abs = 0.0
    for source_index, query in zip(source_indices, online.queries):
        reference = replace(query, query_index=int(source_index))
        cached_edges = cached_topologies[reference].numpy()
        online_edges = online.topologies[query].numpy()
        topology_equal = topology_equal and np.array_equal(cached_edges, online_edges)
        length = int(query.candidate_index.numel())
        cached_local = np.asarray(cached_features.local_for_query(reference), dtype=np.float16)
        online_local = local[cursor : cursor + length]
        local_equal = local_equal and np.array_equal(cached_local, online_local)
        if cached_local.size:
            max_abs = max(
                max_abs,
                float(
                    np.max(
                        np.abs(
                            cached_local.astype(np.float32)
                            - online_local.astype(np.float32)
                        )
                    )
                ),
            )
        cursor += length
    if not topology_equal or not local_equal:
        raise ValueError("Uncached online path failed cached topology/QLS parity")
    return {
        "queries": int(source_indices.size),
        "candidate_generation": "same locked equal-RRF implementation and budget contract",
        "topology_bit_exact": topology_equal,
        "qls_local_float16_bit_exact": local_equal,
        "qls_local_max_abs_error_after_float16": max_abs,
        "query_specific_caches_used_after_parity_gate": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    cold_started = time.perf_counter()
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    dense = np.load(args.data / "dense_top200_all.npy", mmap_mode="r")
    splade = np.load(args.data / "splade_top200_all.npy", mmap_mode="r")
    query_array = np.load(args.data / "queries_all.npy", mmap_mode="r")
    feature_metadata = json.loads(
        (args.cached_features / "metadata.json").read_text(encoding="utf-8")
    )
    feature_dimensions = SimpleNamespace(
        static_dim=len(feature_metadata["static_feature_names"]),
        local_dim=len(feature_metadata["local_feature_names"]),
    )
    budget_result = json.loads(args.budget_result.read_text(encoding="utf-8"))
    if budget_result.get("status") != "CANDIDATE_BUDGET_DATASET_COMPLETE":
        raise ValueError("Package C budget result is incomplete")
    if (
        budget_result.get("dataset") != args.dataset
        or int(budget_result.get("budget", -1)) != args.budget
        or budget_result.get("data_fingerprint_sha256") != args.data_fingerprint_sha256
    ):
        raise ValueError("Package C result differs from the online systems contract")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        node_embeddings = torch.from_numpy(np.asarray(dataset.node_array)).to(
            device=device, dtype=torch.float32
        )
        static_features = torch.from_numpy(
            np.asarray(np.load(args.cached_features / "static.npy", mmap_mode="r"))
        ).to(
            device=device, dtype=torch.float32
        )
    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    models = {}
    checkpoint_proof = {}
    for model_name in MODEL_NAMES:
        model = _build_model(
            model_name,
            dataset,
            feature_dimensions,
            args.selected_gnn,
            target_parameters,
            args,
        ).to(device)
        checkpoint_path = Path(
            budget_result["models"][model_name]["seeds"][str(args.model_seed)][
                "checkpoint_path"
            ]
        )
        expected_file_hash = budget_result["models"][model_name]["seeds"][
            str(args.model_seed)
        ]["checkpoint_file_sha256"]
        if _sha256(checkpoint_path) != expected_file_hash:
            raise ValueError("Package C checkpoint file failed SHA-256 verification")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models[model_name] = model
        checkpoint_proof[model_name] = {
            "path": str(checkpoint_path),
            "sha256": expected_file_hash,
        }
    _synchronize(device)
    cold_start_seconds = time.perf_counter() - cold_started
    ready_rss_mb = process.memory_info().rss / 2**20
    ready_gpu_mb = (
        torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    )
    test_indices = _selected_test_indices(dataset, args.sample_queries)
    parity_indices = test_indices[: min(args.parity_queries, test_indices.size)]
    feature_config = args.feature_config["query_local_features"]["personalized_pagerank"]
    cached_topologies = PackedLocalTopologies.load(args.cached_topology)
    cached_features = StructuralFeatureStore.load(args.cached_features)
    parity = _parity_audit(
        dataset,
        parity_indices,
        dense,
        splade,
        query_array,
        cached_topologies,
        cached_features,
        budget=args.budget,
        rrf_constant=args.rrf_constant,
        damping=float(feature_config["damping"]),
        ppr_iterations=int(feature_config["iterations"]),
    )
    del cached_topologies, cached_features

    # Compile/warm all CPU/GPU kernels without retaining any query-specific result.
    warm_indices = test_indices[: min(args.warmup_queries, test_indices.size)]
    for model_name in MODEL_NAMES:
        for start in range(0, warm_indices.size, max(args.batch_sizes)):
            selected = warm_indices[start : start + max(args.batch_sizes)]
            _execute(
                model_name,
                models[model_name],
                dataset,
                selected,
                np.asarray(dense[selected]).copy(),
                np.asarray(splade[selected]).copy(),
                np.asarray(query_array[selected], dtype=np.float32).copy(),
                node_embeddings,
                static_features,
                device,
                budget=args.budget,
                rrf_constant=args.rrf_constant,
                ppr_damping=float(feature_config["damping"]),
                ppr_iterations=int(feature_config["iterations"]),
                top_k=args.top_k,
            )

    conditions: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        telemetry = {name: {stage: [] for stage in STAGE_NAMES} for name in MODEL_NAMES}
        memory = {
            name: {
                "peak_process_rss_mb": [],
                "peak_gpu_allocated_mb": [],
                "incremental_peak_gpu_allocated_mb": [],
            }
            for name in MODEL_NAMES
        }
        per_query_totals = {name: [] for name in MODEL_NAMES}
        query_totals = {name: 0 for name in MODEL_NAMES}
        wall_totals = {name: 0.0 for name in MODEL_NAMES}
        for repeat in range(args.repeats):
            for start in range(0, test_indices.size, batch_size):
                selected = test_indices[start : start + batch_size]
                input_dense = np.asarray(dense[selected]).copy()
                input_splade = np.asarray(splade[selected]).copy()
                input_queries = np.asarray(query_array[selected], dtype=np.float32).copy()
                order = MODEL_NAMES if (repeat + start // batch_size) % 2 == 0 else MODEL_NAMES[::-1]
                for model_name in order:
                    row = _execute(
                        model_name,
                        models[model_name],
                        dataset,
                        selected,
                        input_dense,
                        input_splade,
                        input_queries,
                        node_embeddings,
                        static_features,
                        device,
                        budget=args.budget,
                        rrf_constant=args.rrf_constant,
                        ppr_damping=float(feature_config["damping"]),
                        ppr_iterations=int(feature_config["iterations"]),
                        top_k=args.top_k,
                    )
                    for stage in STAGE_NAMES:
                        telemetry[model_name][stage].append(row[stage])
                    for measurement in memory[model_name]:
                        memory[model_name][measurement].append(row[measurement])
                    per_query_totals[model_name].append(row["total_ms"] / selected.size)
                    query_totals[model_name] += selected.size
                    wall_totals[model_name] += row["total_ms"] / 1000
        condition = {}
        for model_name in MODEL_NAMES:
            stage_summary = {
                stage: _percentiles(values)
                for stage, values in telemetry[model_name].items()
            }
            condition[model_name] = {
                "batches": len(telemetry[model_name]["total_ms"]),
                "queries": query_totals[model_name],
                "batch_latency_ms": stage_summary,
                "total_latency_ms_per_query": {
                    **_percentiles(per_query_totals[model_name])
                },
                "throughput_queries_per_second": query_totals[model_name]
                / max(wall_totals[model_name], 1e-12),
                "memory_mb": {
                    measurement: max(values)
                    for measurement, values in memory[model_name].items()
                },
            }
        conditions[f"batch_{batch_size}"] = {"models": condition}

    storage_files = {
        "nodes.npy": args.data / "nodes.npy",
        "graph.pt": args.data / "graph.pt",
        "static.npy": args.cached_features / "static.npy",
        **{
            f"checkpoint_{name}.pt": Path(proof["path"])
            for name, proof in checkpoint_proof.items()
        },
    }
    result = {
        "status": "UNCACHED_UNSEEN_EMBEDDING_SYSTEMS_COMPLETE",
        "dataset": args.dataset,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "boundary": {
            "input": "query_embedding_plus_dense_and_splade_ranked_ids",
            "charged": [
                "equal_rrf_and_seed_construction",
                "candidate_embedding_gather",
                "candidate_induced_topology",
                "qls_query_local_summaries_when_applicable",
                "host_device_transfer",
                "model_forward",
                "top_k_selection",
            ],
            "out_of_scope_shared_upstream": [
                "raw_query_text",
                "query_encoding",
                "dense_ann_retrieval",
                "splade_index_retrieval",
            ],
            "query_specific_cache_reads_in_timed_path": False,
            "corpus_static_assets_allowed": True,
        },
        "sample": {
            "split": "held_out_test_queries_unseen_during_ranker_training",
            "selection": "deterministic_even_spacing_in_canonical_test_order",
            "unique_queries": int(test_indices.size),
            "repeats": args.repeats,
            "batch_sizes": args.batch_sizes,
            "budget": args.budget,
            "rrf_constant": args.rrf_constant,
        },
        "parity": parity,
        "startup": {
            "cold_start_seconds": cold_start_seconds,
            "ready_process_rss_mb": ready_rss_mb,
            "ready_gpu_allocated_mb": ready_gpu_mb,
            "cpu_threads": psutil.cpu_count(logical=True),
            "torch_threads": torch.get_num_threads(),
            "device": str(device),
        },
        "checkpoints": checkpoint_proof,
        "static_storage": {
            name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for name, path in storage_files.items()
        },
        "cached_artifacts_used_only_for_untimed_parity": {
            "topology": str(args.cached_topology),
            "features": str(args.cached_features),
        },
        "conditions": conditions,
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in {"baseline", "feature_config"}
        },
    }
    _atomic_json(args.output, result)
    return result


if __name__ == "__main__":
    raise SystemExit("Use scripts/modal_online_systems.py for the frozen execution")
