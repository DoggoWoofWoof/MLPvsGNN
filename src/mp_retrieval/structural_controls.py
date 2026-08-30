"""Training-free controls over frozen query-local structural features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rank_fusion import rrf_rankings

DISTANCE_WEIGHTS = np.asarray([1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0], dtype=np.float32)
PATH_CONNECTIVITY_COLUMNS = (4, 5, 6, 7, 9)
PPR_COLUMN = 8
FROZEN_LOCAL_FEATURE_NAMES = (
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


@dataclass
class FrozenStructuralCache:
    """Memory-mapped view of the sealed QLS cache without graph/torch imports."""

    root: Path
    local: np.ndarray
    candidate_ptr: np.ndarray
    query_position: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, root: str | Path) -> FrozenStructuralCache:
        root = Path(root)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "fixed_structural_features_v1":
            raise ValueError(f"Unsupported structural cache format in {root}")
        return cls(
            root=root,
            local=np.load(root / "local.npy", mmap_mode="r"),
            candidate_ptr=np.load(root / "candidate_ptr.npy", mmap_mode="r"),
            query_position=np.load(root / "query_position.npy", mmap_mode="r"),
            metadata=metadata,
        )


def stable_candidate_union(dense: np.ndarray, splade: np.ndarray) -> np.ndarray:
    """Dense order followed by SPLADE candidates not already observed."""

    combined = np.concatenate((dense, splade)).astype(np.int64, copy=False)
    _values, first = np.unique(combined, return_index=True)
    return combined[np.sort(first)]


def rank_scores(candidate_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Rank descending scores with ascending global-node-ID ties."""

    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if candidate_ids.ndim != 1 or scores.shape != candidate_ids.shape:
        raise ValueError("Candidate IDs and scores must be aligned one-dimensional arrays")
    if np.unique(candidate_ids).size != candidate_ids.size:
        raise ValueError("Candidate IDs must be unique")
    if not np.isfinite(scores).all():
        raise ValueError("Structural scores must be finite")
    return candidate_ids[np.lexsort((candidate_ids, -scores))]


def selected_rrf_ranking(
    dense: np.ndarray,
    splade: np.ndarray,
    *,
    dense_weight: float,
    constant: int,
) -> np.ndarray:
    candidates = stable_candidate_union(dense, splade)
    ranking = rrf_rankings(
        np.asarray(dense, dtype=np.int64)[None, :],
        np.asarray(splade, dtype=np.int64)[None, :],
        dense_weights=[dense_weight],
        constant=constant,
        top_k=int(dense.size + splade.size),
    )[float(dense_weight)][0]
    ranking = ranking[: candidates.size]
    if set(map(int, ranking)) != set(map(int, candidates)):
        raise ValueError("Full selected-RRF ranking does not preserve the candidate set")
    return ranking


def equal_rrf_fusion(
    left: np.ndarray,
    right: np.ndarray,
    *,
    constant: int,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("Fusion rankings must be aligned one-dimensional arrays")
    if set(map(int, left)) != set(map(int, right)):
        raise ValueError("Fusion rankings must contain the same candidate set")
    return rrf_rankings(
        left[None, :],
        right[None, :],
        dense_weights=[0.5],
        constant=constant,
        top_k=left.size,
    )[0.5][0]


def fixed_structural_scores(local_features: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the four locked A2 structural scores."""

    local = np.asarray(local_features, dtype=np.float32)
    if local.ndim != 2 or local.shape[1] != 10:
        raise ValueError("Expected the exact ten frozen query-local QLS features")
    distance = local[:, :4] @ DISTANCE_WEIGHTS
    ppr = local[:, PPR_COLUMN]
    path = local[:, PATH_CONNECTIVITY_COLUMNS].mean(axis=1)
    summary = np.column_stack(
        (
            distance,
            local[:, PATH_CONNECTIVITY_COLUMNS],
            ppr,
        )
    ).mean(axis=1)
    return {
        "distance": distance,
        "ppr": ppr,
        "path_connectivity": path,
        "structural_summary": summary,
    }


def candidate_contract_hashes(
    query_ids: Sequence[str],
    dense: np.ndarray,
    splade: np.ndarray,
    candidate_ptr: np.ndarray,
) -> dict[str, str]:
    """Reproduce both sealed candidate digests without graph loading."""

    if dense.shape != splade.shape or dense.shape[0] != len(query_ids):
        raise ValueError("Query IDs and frozen rank arrays are misaligned")
    if candidate_ptr.shape != (len(query_ids) + 1,):
        raise ValueError("Structural candidate pointers are misaligned")
    order_digest = hashlib.sha256()
    tensor_digest = hashlib.sha256()
    for query_index, query_id in enumerate(query_ids):
        candidates = stable_candidate_union(dense[query_index], splade[query_index])
        cached_count = int(candidate_ptr[query_index + 1] - candidate_ptr[query_index])
        if candidates.size != cached_count:
            raise ValueError(
                f"Candidate count differs from structural cache at query {query_index}"
            )
        encoded_query_id = str(query_id).encode("utf-8")
        order_digest.update(encoded_query_id)
        order_digest.update(int(candidates.size).to_bytes(4, "little"))
        order_digest.update(candidates.tobytes())
        tensor_digest.update(encoded_query_id)
        tensor_digest.update(candidates.tobytes())
    return {
        "candidate_id_order_sha256": order_digest.hexdigest(),
        "candidate_tensor_sha256": tensor_digest.hexdigest(),
    }


def candidate_order_sha256(
    query_ids: Sequence[str],
    dense: np.ndarray,
    splade: np.ndarray,
    candidate_ptr: np.ndarray,
) -> str:
    """Reproduce the dedicated candidate-order digest used by legacy confirmations."""

    return candidate_contract_hashes(query_ids, dense, splade, candidate_ptr)[
        "candidate_id_order_sha256"
    ]
