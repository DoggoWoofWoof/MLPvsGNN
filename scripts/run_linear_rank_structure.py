#!/usr/bin/env python
"""Run the frozen P0 A3 19-parameter linear rank+structure control."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mp_retrieval.linear_control import (
    LINEAR_FEATURE_NAMES,
    LOCAL_FEATURE_NAMES,
    STATIC_FEATURE_NAMES,
    LinearInputCache,
    PositiveIndex,
    build_or_load_linear_input_cache,
    build_positive_index,
    packed_positive_positions,
    packed_row_indices,
    segmented_listwise_loss,
)
from mp_retrieval.protocol import seed_everything
from mp_retrieval.rank_fusion import (
    FrozenRankContract,
    aggregate_metric_arrays,
    load_frozen_rank_contract,
    ranking_metrics,
    sha256_file,
)
from mp_retrieval.structural_controls import FrozenStructuralCache
from scripts.run_fixed_structural_controls import _validate_alignment

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
PACKED_METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")


class LinearRankStructure(nn.Module):
    """Bias-free scorer frozen by the A3 protocol."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(len(LINEAR_FEATURE_NAMES)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weight


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _validate_references(
    dataset: str,
    rank_result: dict[str, Any],
    structural_result: dict[str, Any],
    confirmation: dict[str, Any],
) -> dict[str, Any]:
    if any(payload.get("dataset") != dataset for payload in (rank_result, structural_result, confirmation)):
        raise ValueError("A3 reference artifacts do not name the requested dataset")
    if rank_result.get("status") != "P0_A1_RANK_CONTROLS_COMPLETE":
        raise ValueError("A3 requires the completed corrected A1 artifact")
    if structural_result.get("status") != "P0_A2_FIXED_STRUCTURAL_CONTROLS_COMPLETE":
        raise ValueError("A3 requires the completed A2 artifact")
    if confirmation.get("status") != "SA_MLP_CONFIRMATION_DATASET_COMPLETE":
        raise ValueError("A3 requires the sealed QLS confirmation artifact")
    reproduction = structural_result.get("a1_reproduction", {})
    if reproduction.get("status") != "CORRECTED_A1_SELECTED_RRF_REPRODUCED":
        raise ValueError("A2 did not record corrected A1 reproduction")
    if float(reproduction.get("maximum_absolute_difference", float("inf"))) > 2e-7:
        raise ValueError("A2 selected-RRF reproduction is outside tolerance")
    if (
        structural_result["alignment"]["structural_contract_sha256"]
        != confirmation["feature_cache"]["contract_sha256"]
    ):
        raise ValueError("A2 and QLS confirmation structural contracts differ")
    return {
        "rank_status": rank_result["status"],
        "structural_status": structural_result["status"],
        "confirmation_status": confirmation["status"],
        "a1_reproduction_maximum_absolute_difference": reproduction[
            "maximum_absolute_difference"
        ],
    }


def _load_static(feature_root: Path, cache: FrozenStructuralCache) -> np.ndarray:
    metadata = cache.metadata
    if tuple(metadata.get("static_feature_names", ())) != STATIC_FEATURE_NAMES:
        raise ValueError("Static feature names differ from the frozen A3 order")
    if tuple(metadata.get("local_feature_names", ())) != LOCAL_FEATURE_NAMES:
        raise ValueError("Local feature names differ from the frozen A3 order")
    static = np.load(feature_root / "static.npy", mmap_mode="r")
    if static.ndim != 2 or static.shape[1] != len(STATIC_FEATURE_NAMES):
        raise ValueError("Frozen static feature matrix has an unexpected shape")
    return static


def _build_feature_batch(
    query_indices: np.ndarray,
    structural: FrozenStructuralCache,
    static: np.ndarray,
    derived: LinearInputCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, lengths, offsets = packed_row_indices(query_indices, structural.candidate_ptr)
    candidate_ids = np.asarray(derived.candidate_ids[rows], dtype=np.int64)
    if candidate_ids.size and (
        int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= static.shape[0]
    ):
        raise ValueError("A3 candidate ID is outside the frozen static feature matrix")
    features = np.concatenate(
        (
            np.asarray(derived.rank_features[rows], dtype=np.float32),
            np.asarray(static[candidate_ids], dtype=np.float32),
            np.asarray(structural.local[rows], dtype=np.float32),
        ),
        axis=1,
    )
    if features.shape[1] != len(LINEAR_FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("A3 feature batch violates the frozen 19-column contract")
    segment_ids = np.repeat(np.arange(query_indices.size, dtype=np.int64), lengths)
    return features, candidate_ids, segment_ids, lengths, offsets


def _training_batch(
    query_indices: np.ndarray,
    structural: FrozenStructuralCache,
    static: np.ndarray,
    derived: LinearInputCache,
    positive: PositiveIndex,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, _ids, segment_ids, _lengths, offsets = _build_feature_batch(
        query_indices, structural, static, derived
    )
    positives = packed_positive_positions(query_indices, positive, offsets)
    return (
        torch.from_numpy(features).to(device=device, non_blocking=device.type == "cuda"),
        torch.from_numpy(segment_ids).to(device=device, non_blocking=device.type == "cuda"),
        torch.from_numpy(positives).to(device=device, non_blocking=device.type == "cuda"),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward_models(
    models: list[LinearRankStructure],
    query_indices: np.ndarray,
    structural: FrozenStructuralCache,
    static: np.ndarray,
    derived: LinearInputCache,
    device: torch.device,
    *,
    batch_size: int,
    store: bool,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, float]]:
    for model in models:
        model.eval()
    batches = [query_indices[start : start + batch_size] for start in range(0, query_indices.size, batch_size)]
    if batches:
        features, _ids, _segments, _lengths, _offsets = _build_feature_batch(
            batches[0], structural, static, derived
        )
        with torch.no_grad():
            tensor = torch.from_numpy(features).to(device=device)
            for model in models:
                model(tensor)
        _synchronize(device)
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    baseline_gpu = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_gpu = int(torch.cuda.memory_allocated(device))
    stored: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            features, candidate_ids, _segments, lengths, _offsets = _build_feature_batch(
                batch, structural, static, derived
            )
            tensor = torch.from_numpy(features).to(
                device=device, non_blocking=device.type == "cuda"
            )
            score_matrix = torch.stack([model(tensor) for model in models], dim=1)
            if store:
                stored.append((batch.copy(), candidate_ids, lengths, score_matrix.cpu().numpy()))
            peak_rss = max(peak_rss, process.memory_info().rss)
        _synchronize(device)
    seconds = time.perf_counter() - started
    peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return stored, {
        "inference_seconds": seconds,
        "latency_ms_per_query": seconds * 1000 / max(query_indices.size, 1),
        "throughput_queries_per_second": query_indices.size / max(seconds, 1e-12),
        "peak_gpu_memory_mb_total": peak_gpu / 2**20,
        "peak_gpu_memory_mb_incremental": max(peak_gpu - baseline_gpu, 0) / 2**20,
        "peak_cpu_rss_mb_total": peak_rss / 2**20,
        "peak_cpu_rss_mb_incremental": max(peak_rss - baseline_rss, 0) / 2**20,
    }


def _rows_from_stored(
    stored: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    contract: FrozenRankContract,
    *,
    model_count: int,
) -> list[list[dict[str, float | None]]]:
    rows: list[list[dict[str, float | None]]] = [[] for _ in range(model_count)]
    for query_indices, candidate_ids, lengths, score_matrix in stored:
        offset = 0
        for query_index, length in zip(query_indices, lengths):
            ids = candidate_ids[offset : offset + length]
            for model_index in range(model_count):
                scores = score_matrix[offset : offset + length, model_index]
                ranking = ids[np.lexsort((ids, -scores))]
                rows[model_index].append(
                    ranking_metrics(ranking, contract.golds[int(query_index)], ids)
                )
            offset += int(length)
    return rows


def _aggregate_rows(rows: list[dict[str, float | None]]) -> dict[str, float | int]:
    arrays = {
        metric: np.asarray(
            [np.nan if row[metric] is None else float(row[metric]) for row in rows],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    return aggregate_metric_arrays(arrays)


def _evaluate_models(
    models: list[LinearRankStructure],
    query_indices: np.ndarray,
    contract: FrozenRankContract,
    structural: FrozenStructuralCache,
    static: np.ndarray,
    derived: LinearInputCache,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[list[dict[str, float | int]], list[list[dict[str, float | None]]], dict[str, float]]:
    stored, timing = _forward_models(
        models,
        query_indices,
        structural,
        static,
        derived,
        device,
        batch_size=batch_size,
        store=True,
    )
    rows = _rows_from_stored(stored, contract, model_count=len(models))
    return [_aggregate_rows(model_rows) for model_rows in rows], rows, timing


def _repeat_timing(
    model: LinearRankStructure,
    query_indices: np.ndarray,
    structural: FrozenStructuralCache,
    static: np.ndarray,
    derived: LinearInputCache,
    device: torch.device,
    *,
    batch_size: int,
    repeats: int,
    first: dict[str, float],
) -> dict[str, Any]:
    records = [first]
    for _ in range(1, repeats):
        _stored, timing = _forward_models(
            [model],
            query_indices,
            structural,
            static,
            derived,
            device,
            batch_size=batch_size,
            store=False,
        )
        records.append(timing)
    return {
        "repeats": repeats,
        "latency_ms_per_query": statistics.median(
            record["latency_ms_per_query"] for record in records
        ),
        "throughput_queries_per_second": statistics.median(
            record["throughput_queries_per_second"] for record in records
        ),
        "peak_gpu_memory_mb_total": max(record["peak_gpu_memory_mb_total"] for record in records),
        "peak_gpu_memory_mb_incremental": max(
            record["peak_gpu_memory_mb_incremental"] for record in records
        ),
        "peak_cpu_rss_mb_total": max(record["peak_cpu_rss_mb_total"] for record in records),
        "peak_cpu_rss_mb_incremental": max(
            record["peak_cpu_rss_mb_incremental"] for record in records
        ),
        "repeat_telemetry": records,
        "includes_cached_feature_gather_and_device_transfer": True,
        "excludes_metric_computation": True,
    }


def _fit_models(
    learning_rates: list[float],
    seed: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    contract: FrozenRankContract,
    structural: FrozenStructuralCache,
    static: np.ndarray,
    derived: LinearInputCache,
    positive: PositiveIndex,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    clip_norm: float,
) -> tuple[list[LinearRankStructure], list[dict[str, Any]], dict[str, float]]:
    seed_everything(seed)
    models = [LinearRankStructure().to(device) for _ in learning_rates]
    if any(sum(parameter.numel() for parameter in model.parameters()) != 19 for model in models):
        raise RuntimeError("A3 model must have exactly 19 trainable parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "lr": learning_rate}
            for model, learning_rate in zip(models, learning_rates)
        ],
        weight_decay=weight_decay,
    )
    eligible = train_indices[
        positive.ptr[train_indices + 1] > positive.ptr[train_indices]
    ]
    histories: list[list[dict[str, float | int]]] = [[] for _ in models]
    best_validation = [-float("inf")] * len(models)
    best_states: list[dict[str, torch.Tensor] | None] = [None] * len(models)
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    baseline_gpu = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_gpu = int(torch.cuda.memory_allocated(device))
    started = time.perf_counter()
    for epoch in range(epochs):
        order = eligible.tolist()
        random.Random(seed + epoch * 1_000_003).shuffle(order)
        for model in models:
            model.train()
        losses: list[list[float]] = [[] for _ in models]
        for start in range(0, len(order), batch_size):
            batch = np.asarray(order[start : start + batch_size], dtype=np.int64)
            features, segments, positives = _training_batch(
                batch, structural, static, derived, positive, device
            )
            optimizer.zero_grad(set_to_none=True)
            model_losses = [
                segmented_listwise_loss(
                    model(features),
                    segments,
                    positives,
                    num_queries=batch.size,
                )
                for model in models
            ]
            torch.stack(model_losses).sum().backward()
            for model in models:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            for values, loss in zip(losses, model_losses):
                values.append(float(loss.detach()))
            peak_rss = max(peak_rss, process.memory_info().rss)
        validation, _rows, _timing = _evaluate_models(
            models,
            validation_indices,
            contract,
            structural,
            static,
            derived,
            device,
            batch_size=batch_size,
        )
        for model_index, (model, metrics) in enumerate(zip(models, validation)):
            value = float(metrics["recall@5"])
            histories[model_index].append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": float(np.mean(losses[model_index])),
                    "validation_recall@5": value,
                }
            )
            if value > best_validation[model_index]:
                best_validation[model_index] = value
                best_states[model_index] = deepcopy(
                    {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
                )
    _synchronize(device)
    seconds = time.perf_counter() - started
    records: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        if best_states[model_index] is None:
            raise RuntimeError("A3 failed to select a validation checkpoint")
        model.load_state_dict(best_states[model_index])
        records.append(
            {
                "learning_rate": learning_rates[model_index],
                "best_validation_recall@5": best_validation[model_index],
                "best_epoch": next(
                    int(row["epoch"])
                    for row in histories[model_index]
                    if row["validation_recall@5"] == best_validation[model_index]
                ),
                "history": histories[model_index],
                "train_queries": int(train_indices.size),
                "train_queries_with_candidate_gold": int(eligible.size),
            }
        )
    peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return models, records, {
        "training_seconds_shared_feature_pass": seconds,
        "peak_training_gpu_memory_mb_total": peak_gpu / 2**20,
        "peak_training_gpu_memory_mb_incremental": max(peak_gpu - baseline_gpu, 0) / 2**20,
        "peak_training_cpu_rss_mb_total": peak_rss / 2**20,
        "peak_training_cpu_rss_mb_incremental": max(peak_rss - baseline_rss, 0) / 2**20,
    }


def _hop_aggregates(
    query_indices: np.ndarray,
    rows: list[dict[str, float | None]],
    hops: np.ndarray | None,
) -> dict[str, Any]:
    if hops is None:
        return {}
    output: dict[str, Any] = {}
    query_hops = hops[query_indices]
    for hop in sorted(set(map(int, query_hops))):
        selected = [row for row, value in zip(rows, query_hops) if int(value) == hop]
        output[str(hop)] = {"queries": len(selected), "metrics": _aggregate_rows(selected)}
    return output


def _pack_query_rows(rows: list[dict[str, float | None]]) -> np.ndarray:
    return np.asarray(
        [[float(row[metric]) for metric in PACKED_METRICS] for row in rows],
        dtype=np.float32,
    )


def _reference_summary(
    rank_result: dict[str, Any],
    structural_result: dict[str, Any],
    confirmation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "a1_test": rank_result["test"],
        "a2_test": structural_result["test"],
        "seed_only_mlp": confirmation["models"]["seed_only"]["aggregate"]["test_metrics"],
        "qls_mlp": confirmation["models"]["sa_mlp"]["aggregate"]["test_metrics"],
        "seed_aware_selected_gnn": confirmation["models"]["seed_aware_gnn"]["aggregate"][
            "test_metrics"
        ],
    }


def run(
    args: argparse.Namespace,
    *,
    rank_result_payload: dict[str, Any] | None = None,
    structural_result_payload: dict[str, Any] | None = None,
    confirmation_payload: dict[str, Any] | None = None,
    checkpoint_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "PROTOCOL_FROZEN_BEFORE_TEST_EVALUATION":
        raise ValueError("A3 protocol is not frozen")
    if args.dataset not in config["frozen_inputs"]["datasets"]:
        raise ValueError(f"Dataset is outside the A3 protocol: {args.dataset}")
    if config["model"]["trainable_parameters"] != 19 or config["model"]["bias"]:
        raise ValueError("A3 must use the bias-free 19-parameter protocol v2")
    rank_result = rank_result_payload or _load(args.rank_result)
    structural_result = structural_result_payload or _load(args.structural_result)
    confirmation = confirmation_payload or _load(args.confirmation)
    references = _validate_references(
        args.dataset, rank_result, structural_result, confirmation
    )
    contract = load_frozen_rank_contract(args.data, dataset=args.dataset, hash_sources=False)
    structural = FrozenStructuralCache.load(args.feature_cache)
    alignment = _validate_alignment(contract, structural, confirmation)
    static = _load_static(args.feature_cache, structural)
    if int(static.shape[0]) != int(confirmation["data"]["nodes"]):
        raise ValueError("Static feature node count differs from the confirmation artifact")
    rrf_constant = int(config["model"]["rrf_constant"])
    derived, derived_run = build_or_load_linear_input_cache(
        contract,
        structural.candidate_ptr,
        args.derived_cache,
        candidate_tensor_sha256=alignment["candidate_tensor_sha256"],
        structural_contract_sha256=alignment["structural_contract_sha256"],
        constant=rrf_constant,
    )
    if checkpoint_hook is not None:
        checkpoint_hook()
    positive_started = time.perf_counter()
    positive = build_positive_index(contract, structural.candidate_ptr, derived.candidate_ids)
    positive_seconds = time.perf_counter() - positive_started
    splits = contract.split_indices
    if any(splits[name].size == 0 for name in ("train", "validation", "test")):
        raise ValueError("A3 requires non-empty canonical train/validation/test splits")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    training = config["training"]
    result: dict[str, Any] = {
        "status": "P0_A3_LINEAR_RANK_STRUCTURE_IN_PROGRESS",
        "dataset": args.dataset,
        "protocol": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "tag": "p0-linear-rank-structure-protocol-v2",
        },
        "input_contract": {
            "data": str(args.data),
            "feature_cache": str(args.feature_cache),
            "derived_cache": str(args.derived_cache),
            "queries": contract.query_count,
            "candidate_rows": int(structural.candidate_ptr[-1]),
            "splits": {name: int(indices.size) for name, indices in splits.items()},
            "identity_source": contract.identity_source,
            "graph_loaded": False,
            "embeddings_loaded": False,
            "labels_persisted_in_derived_cache": False,
        },
        "alignment": {
            **alignment,
            "static_feature_names": list(STATIC_FEATURE_NAMES),
            "linear_feature_names": list(LINEAR_FEATURE_NAMES),
            "reference_validation": references,
        },
        "derived_cache": {**derived.metadata, **derived_run},
        "label_index": {
            "persisted": False,
            "positive_rows": int(positive.local.size),
            "construction_seconds": positive_seconds,
        },
        "model": {
            "name": "linear_rank_structure",
            "trainable_parameters": 19,
            "bias": False,
            "hidden_layers": 0,
            "embeddings": False,
            "adjacency_in_forward": False,
            "message_passing": False,
            "feature_names": list(LINEAR_FEATURE_NAMES),
        },
        "learning_rate_screen": {},
        "selected_learning_rate": None,
        "seeds": {},
        "references_not_retrained": _reference_summary(
            rank_result, structural_result, confirmation
        ),
        "test_access_audit": {
            "test_used_for_learning_rate_selection": False,
            "test_used_for_epoch_checkpoint_selection": False,
            "test_selected_features_or_models": False,
            "test_evaluations_per_seed": 0,
        },
        "config": {
            "device": str(device),
            "epochs": int(training["epochs"]),
            "batch_size": int(training["query_batch_size"]),
            "seeds": list(training["seeds"]),
            "learning_rate_grid": list(training["learning_rate_grid"]),
        },
    }

    def checkpoint() -> None:
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    learning_rates = [float(value) for value in training["learning_rate_grid"]]
    screen_models, screen_records, screen_timing = _fit_models(
        learning_rates,
        int(training["learning_rate_selection"]["seed"]),
        splits["train"],
        splits["validation"],
        contract,
        structural,
        static,
        derived,
        positive,
        device,
        epochs=int(training["epochs"]),
        batch_size=int(training["query_batch_size"]),
        weight_decay=float(training["weight_decay"]),
        clip_norm=float(training["gradient_clip_norm"]),
    )
    for record, model in zip(screen_records, screen_models):
        key = f"{record['learning_rate']:.12g}"
        result["learning_rate_screen"][key] = {
            **record,
            "checkpoint_sha256": _state_sha256(model),
            "weights": model.weight.detach().cpu().tolist(),
        }
    result["learning_rate_screen_timing"] = screen_timing
    selected_index = min(
        range(len(learning_rates)),
        key=lambda index: (
            -float(screen_records[index]["best_validation_recall@5"]),
            learning_rates[index],
        ),
    )
    selected_learning_rate = learning_rates[selected_index]
    result["selected_learning_rate"] = selected_learning_rate
    checkpoint()

    query_arrays: dict[str, np.ndarray] = {}
    canonical_seeds = [int(seed) for seed in training["seeds"]]
    screen_seed = int(training["learning_rate_selection"]["seed"])
    for seed in canonical_seeds:
        if seed == screen_seed:
            model = screen_models[selected_index]
            fit_record = screen_records[selected_index]
            fit_timing = screen_timing
            reused = True
        else:
            models, records, fit_timing = _fit_models(
                [selected_learning_rate],
                seed,
                splits["train"],
                splits["validation"],
                contract,
                structural,
                static,
                derived,
                positive,
                device,
                epochs=int(training["epochs"]),
                batch_size=int(training["query_batch_size"]),
                weight_decay=float(training["weight_decay"]),
                clip_norm=float(training["gradient_clip_norm"]),
            )
            model = models[0]
            fit_record = records[0]
            reused = False
        test_metrics, test_rows, first_timing = _evaluate_models(
            [model],
            splits["test"],
            contract,
            structural,
            static,
            derived,
            device,
            batch_size=int(training["query_batch_size"]),
        )
        timing = _repeat_timing(
            model,
            splits["test"],
            structural,
            static,
            derived,
            device,
            batch_size=int(training["query_batch_size"]),
            repeats=int(training["inference_repeats_for_timing"]),
            first=first_timing,
        )
        rows = test_rows[0]
        result["seeds"][str(seed)] = {
            "metrics": test_metrics[0],
            "by_hop": _hop_aggregates(splits["test"], rows, contract.hops),
            "training": {**fit_record, **fit_timing, "reused_from_lr_screen": reused},
            "inference": timing,
            "checkpoint_sha256": _state_sha256(model),
            "weights": model.weight.detach().cpu().tolist(),
        }
        query_arrays[f"linear_rank_structure_seed_{seed}"] = _pack_query_rows(rows)
        result["test_access_audit"]["test_evaluations_per_seed"] += 1
        checkpoint()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate: dict[str, Any] = {"test_metrics": {}, "validation_recall@5": {}}
    for metric in METRICS:
        aggregate["test_metrics"][metric] = _mean_std(
            [float(result["seeds"][str(seed)]["metrics"][metric]) for seed in canonical_seeds]
        )
    aggregate["validation_recall@5"] = _mean_std(
        [
            float(result["seeds"][str(seed)]["training"]["best_validation_recall@5"])
            for seed in canonical_seeds
        ]
    )
    aggregate["training_seconds"] = _mean_std(
        [
            float(result["seeds"][str(seed)]["training"]["training_seconds_shared_feature_pass"])
            for seed in canonical_seeds
        ]
    )
    aggregate["latency_ms_per_query"] = _mean_std(
        [
            float(result["seeds"][str(seed)]["inference"]["latency_ms_per_query"])
            for seed in canonical_seeds
        ]
    )
    result["aggregate"] = aggregate
    args.query_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.query_metrics_output,
        metric_names=np.asarray(PACKED_METRICS),
        query_index=splits["test"],
        **query_arrays,
    )
    result["query_metrics"] = {
        "path": str(args.query_metrics_output),
        "sha256": sha256_file(args.query_metrics_output),
        "rows": int(splits["test"].size),
        "arrays": sorted(query_arrays),
    }
    if result["test_access_audit"]["test_evaluations_per_seed"] != len(canonical_seeds):
        raise RuntimeError("A3 did not perform exactly one metric evaluation per seed")
    result["status"] = "P0_A3_LINEAR_RANK_STRUCTURE_COMPLETE"
    checkpoint()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "p0_linear_rank_structure.yaml"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--rank-result", type=Path, required=True)
    parser.add_argument("--structural-result", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-metrics-output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    namespace = parse_args()
    completed = run(namespace)
    print(
        json.dumps(
            {
                "dataset": completed["dataset"],
                "selected_learning_rate": completed["selected_learning_rate"],
                "test_R@5": completed["aggregate"]["test_metrics"]["recall@5"],
            }
        )
    )
