"""Low-capacity linear controls over frozen rank and structural features."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .rank_fusion import FrozenRankContract, sha256_file
from .structural_controls import stable_candidate_union

RANK_FEATURE_NAMES = ("dense_reciprocal_rank", "splade_reciprocal_rank")
STATIC_FEATURE_NAMES = (
    "log_out_degree",
    "log_in_degree",
    "log_total_degree",
    "pagerank",
    "hub_degree_percentile",
    "coreness",
    "clustering_wedge_estimate",
)
LOCAL_FEATURE_NAMES = (
    "distance_0",
    "distance_1",
    "distance_2",
    "distance_3_plus_or_unreachable",
    "seed_connections",
    "paths_length_1",
    "paths_length_2",
    "paths_length_3",
    "personalized_pagerank",
    "common_out_neighbors_with_seed_neighborhood",
)
LINEAR_FEATURE_NAMES = RANK_FEATURE_NAMES + STATIC_FEATURE_NAMES + LOCAL_FEATURE_NAMES


@dataclass
class LinearInputCache:
    """Label-free derived A3 inputs aligned to the sealed structural rows."""

    root: Path
    candidate_ids: np.ndarray
    rank_features: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path) -> LinearInputCache:
        root = Path(root)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "linear_rank_structure_inputs_v1":
            raise ValueError(f"Unsupported A3 input cache in {root}")
        candidate_ids = np.load(root / "candidate_ids.npy", mmap_mode="r")
        rank_features = np.load(root / "rank_features.npy", mmap_mode="r")
        if candidate_ids.ndim != 1:
            raise ValueError("A3 candidate IDs must be one-dimensional")
        if rank_features.shape != (candidate_ids.size, len(RANK_FEATURE_NAMES)):
            raise ValueError("A3 rank features do not align with candidate IDs")
        return cls(root, candidate_ids, rank_features, metadata)


@dataclass
class PositiveIndex:
    """In-memory label index; never persisted in the label-free feature cache."""

    ptr: np.ndarray
    local: np.ndarray


def rank_feature_rows(
    dense: np.ndarray,
    splade: np.ndarray,
    candidates: np.ndarray,
    *,
    constant: int,
) -> np.ndarray:
    """Build the two frozen normalized reciprocal-rank features."""

    dense = np.asarray(dense, dtype=np.int64)
    splade = np.asarray(splade, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    if dense.ndim != 1 or splade.ndim != 1 or candidates.ndim != 1:
        raise ValueError("Rank rows and candidates must be one-dimensional")
    if np.unique(dense).size != dense.size or np.unique(splade).size != splade.size:
        raise ValueError("Each frozen source ranking must contain unique IDs")
    if constant < 0:
        raise ValueError("Reciprocal-rank constant must be non-negative")
    position = {int(node_id): index for index, node_id in enumerate(candidates)}
    if len(position) != candidates.size:
        raise ValueError("Stable candidate union must contain unique IDs")
    values = np.zeros((candidates.size, 2), dtype=np.float32)
    numerator = float(constant + 1)
    for source_index, source in enumerate((dense, splade)):
        for zero_based_rank, node_id in enumerate(source):
            try:
                candidate_position = position[int(node_id)]
            except KeyError as exc:
                raise ValueError("A ranked source ID is absent from the candidate union") from exc
            values[candidate_position, source_index] = numerator / (
                constant + zero_based_rank + 1
            )
    return values


def input_contract_sha256(
    *,
    candidate_tensor_sha256: str,
    structural_contract_sha256: str,
    constant: int,
) -> str:
    payload = {
        "format": "linear_rank_structure_inputs_v1",
        "candidate_tensor_sha256": candidate_tensor_sha256,
        "structural_contract_sha256": structural_contract_sha256,
        "constant": int(constant),
        "rank_feature_names": list(RANK_FEATURE_NAMES),
        "candidate_dtype": "int32",
        "rank_dtype": "float16",
        "labels_persisted": False,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_or_load_linear_input_cache(
    contract: FrozenRankContract,
    candidate_ptr: np.ndarray,
    root: str | Path,
    *,
    candidate_tensor_sha256: str,
    structural_contract_sha256: str,
    constant: int,
) -> tuple[LinearInputCache, dict[str, Any]]:
    """Build candidate IDs/rank features once, without graph, embeddings, or labels."""

    root = Path(root)
    expected_contract = input_contract_sha256(
        candidate_tensor_sha256=candidate_tensor_sha256,
        structural_contract_sha256=structural_contract_sha256,
        constant=constant,
    )
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        cache = LinearInputCache.load(root)
        if cache.metadata.get("contract_sha256") != expected_contract:
            raise ValueError("Existing A3 derived cache has a different frozen contract")
        if cache.candidate_ids.size != int(candidate_ptr[-1]):
            raise ValueError("Existing A3 derived cache has a different candidate-row count")
        return cache, {"reused": True, "build_seconds_this_run": 0.0}

    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    rows = int(candidate_ptr[-1])
    candidate_partial = root / "candidate_ids.partial.npy"
    rank_partial = root / "rank_features.partial.npy"
    candidate_ids = np.lib.format.open_memmap(
        candidate_partial, mode="w+", dtype=np.int32, shape=(rows,)
    )
    rank_features = np.lib.format.open_memmap(
        rank_partial,
        mode="w+",
        dtype=np.float16,
        shape=(rows, len(RANK_FEATURE_NAMES)),
    )
    for query_index in range(contract.query_count):
        dense = np.asarray(contract.dense[query_index], dtype=np.int64)
        splade = np.asarray(contract.splade[query_index], dtype=np.int64)
        candidates = stable_candidate_union(dense, splade)
        start = int(candidate_ptr[query_index])
        end = int(candidate_ptr[query_index + 1])
        if end - start != candidates.size:
            raise ValueError(f"A3 candidate count differs at query {query_index}")
        if candidates.size and (
            int(candidates.min()) < np.iinfo(np.int32).min
            or int(candidates.max()) > np.iinfo(np.int32).max
        ):
            raise OverflowError("A3 candidate global ID does not fit the frozen int32 cache")
        candidate_ids[start:end] = candidates.astype(np.int32, copy=False)
        rank_features[start:end] = rank_feature_rows(
            dense,
            splade,
            candidates,
            constant=constant,
        ).astype(np.float16)
    candidate_ids.flush()
    rank_features.flush()
    del candidate_ids, rank_features
    candidate_path = root / "candidate_ids.npy"
    rank_path = root / "rank_features.npy"
    candidate_partial.replace(candidate_path)
    rank_partial.replace(rank_path)
    build_seconds = time.perf_counter() - started
    metadata = {
        "format": "linear_rank_structure_inputs_v1",
        "contract_sha256": expected_contract,
        "candidate_tensor_sha256": candidate_tensor_sha256,
        "structural_contract_sha256": structural_contract_sha256,
        "candidate_rows": rows,
        "query_count": contract.query_count,
        "rrf_constant": int(constant),
        "rank_feature_names": list(RANK_FEATURE_NAMES),
        "candidate_dtype": "int32",
        "rank_dtype": "float16",
        "labels_persisted": False,
        "graph_loaded": False,
        "embeddings_loaded": False,
        "build_seconds": build_seconds,
        "cache_bytes": candidate_path.stat().st_size + rank_path.stat().st_size,
        "files_sha256": {
            "candidate_ids.npy": sha256_file(candidate_path),
            "rank_features.npy": sha256_file(rank_path),
        },
    }
    temporary_metadata = root / "metadata.json.partial"
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    return LinearInputCache.load(root), {
        "reused": False,
        "build_seconds_this_run": build_seconds,
    }


def build_positive_index(
    contract: FrozenRankContract,
    candidate_ptr: np.ndarray,
    candidate_ids: np.ndarray,
) -> PositiveIndex:
    """Map gold global IDs to local positions without writing labels to disk."""

    ptr = np.empty(contract.query_count + 1, dtype=np.int64)
    ptr[0] = 0
    arrays: list[np.ndarray] = []
    for query_index, golds in enumerate(contract.golds):
        start = int(candidate_ptr[query_index])
        end = int(candidate_ptr[query_index + 1])
        ids = np.asarray(candidate_ids[start:end], dtype=np.int64)
        local = np.flatnonzero(np.isin(ids, np.asarray(golds, dtype=np.int64))).astype(
            np.int32, copy=False
        )
        arrays.append(local)
        ptr[query_index + 1] = ptr[query_index] + local.size
    packed = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int32)
    return PositiveIndex(ptr=ptr, local=packed)


def packed_row_indices(
    query_indices: Sequence[int] | np.ndarray,
    candidate_ptr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat cache rows, candidate lengths, and packed query offsets."""

    queries = np.asarray(query_indices, dtype=np.int64)
    lengths = candidate_ptr[queries + 1] - candidate_ptr[queries]
    offsets = np.empty(queries.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    rows = np.empty(int(offsets[-1]), dtype=np.int64)
    for position, query_index in enumerate(queries):
        start = int(candidate_ptr[query_index])
        end = int(candidate_ptr[query_index + 1])
        rows[offsets[position] : offsets[position + 1]] = np.arange(start, end)
    return rows, lengths.astype(np.int64, copy=False), offsets


def packed_positive_positions(
    query_indices: Sequence[int] | np.ndarray,
    positive: PositiveIndex,
    batch_offsets: np.ndarray,
) -> np.ndarray:
    """Translate per-query local positives into positions in a packed batch."""

    queries = np.asarray(query_indices, dtype=np.int64)
    arrays: list[np.ndarray] = []
    for position, query_index in enumerate(queries):
        start = int(positive.ptr[query_index])
        end = int(positive.ptr[query_index + 1])
        arrays.append(
            positive.local[start:end].astype(np.int64, copy=False) + batch_offsets[position]
        )
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)


def segmented_listwise_loss(
    scores: torch.Tensor,
    segment_ids: torch.Tensor,
    positive_positions: torch.Tensor,
    *,
    num_queries: int,
) -> torch.Tensor:
    """Vectorized multi-positive listwise loss over packed variable-size queries."""

    if scores.ndim != 1 or segment_ids.shape != scores.shape:
        raise ValueError("Scores and segment IDs must be aligned vectors")
    if positive_positions.ndim != 1 or positive_positions.numel() == 0:
        raise ValueError("A training batch must contain positive candidate positions")
    maximum = torch.full(
        (num_queries,), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    maximum.scatter_reduce_(0, segment_ids, scores, reduce="amax", include_self=True)
    normalizer = torch.zeros(num_queries, dtype=scores.dtype, device=scores.device)
    normalizer.scatter_add_(0, segment_ids, torch.exp(scores - maximum[segment_ids]))
    log_partition = maximum + torch.log(normalizer)
    positive_segments = segment_ids[positive_positions]
    positive_sum = torch.zeros_like(normalizer)
    positive_sum.scatter_add_(0, positive_segments, scores[positive_positions])
    positive_count = torch.zeros_like(normalizer)
    positive_count.scatter_add_(0, positive_segments, torch.ones_like(scores[positive_positions]))
    if torch.any(positive_count == 0):
        raise ValueError("Every packed training query must have an in-pool positive")
    return (log_partition - positive_sum / positive_count).mean()
