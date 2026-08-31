"""Cache-disabled post-retrieval serving helpers for unseen query embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .complete_data import CompleteQuery, CompleteRetrievalDataset
from .rank_fusion import rrf_rankings
from .topology_store import PackedLocalTopologies, build_packed_topologies


@dataclass
class OnlineBatch:
    """One transient batch built only from upstream embeddings and ranked IDs."""

    source_query_indices: np.ndarray
    query_embeddings: np.ndarray
    queries: list[CompleteQuery]
    topologies: PackedLocalTopologies


def fuse_equal_rrf_candidates(
    dense: np.ndarray,
    splade: np.ndarray,
    *,
    budget: int,
    constant: int = 60,
) -> list[np.ndarray]:
    """Return unique equal-RRF candidates for each upstream ranking pair."""

    dense = np.asarray(dense)
    splade = np.asarray(splade)
    if dense.shape != splade.shape or dense.ndim != 2:
        raise ValueError("Dense and SPLADE online rankings must be aligned matrices")
    maximum = 2 * dense.shape[1]
    if budget <= 0 or budget > maximum:
        raise ValueError(f"Online candidate budget must lie in [1, {maximum}]")
    ranked = rrf_rankings(
        dense,
        splade,
        dense_weights=[0.5],
        constant=constant,
        top_k=maximum,
    )[0.5]
    rows = []
    for index in range(dense.shape[0]):
        unique_count = np.unique(np.concatenate((dense[index], splade[index]))).size
        rows.append(
            np.asarray(ranked[index, : min(budget, unique_count)], dtype=np.int64).copy()
        )
    return rows


def build_online_queries(
    candidate_rows: list[np.ndarray],
    dense: np.ndarray,
    splade: np.ndarray,
) -> list[CompleteQuery]:
    """Build ephemeral, label-free query objects and frozen seed membership."""

    if len(candidate_rows) != dense.shape[0] or dense.shape != splade.shape:
        raise ValueError("Online candidates and source rankings are misaligned")
    queries = []
    for position, candidate_values in enumerate(candidate_rows):
        local = {int(node): index for index, node in enumerate(candidate_values)}
        seeds = np.concatenate((dense[position, :5], splade[position, :5]))
        _values, first = np.unique(seeds, return_index=True)
        stable_seeds = seeds[np.sort(first)]
        seed_local = torch.tensor(
            [local[int(node)] for node in stable_seeds if int(node) in local],
            dtype=torch.long,
        )
        queries.append(
            CompleteQuery(
                query_index=position,
                query_id=f"online:{position}",
                candidate_index=torch.from_numpy(candidate_values.copy()),
                relevant_local=torch.empty(0, dtype=torch.long),
                relevant_global=torch.empty(0, dtype=torch.long),
                anchor_global=int(dense[position, 0]),
                split=-1,
                retrieval_seed_local=seed_local,
            )
        )
    return queries


def build_online_batch(
    dataset: CompleteRetrievalDataset,
    source_query_indices: np.ndarray,
    dense: np.ndarray,
    splade: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    budget: int,
    constant: int = 60,
) -> OnlineBatch:
    """Fuse candidates and induce topology without reading a query cache."""

    candidates = fuse_equal_rrf_candidates(
        dense,
        splade,
        budget=budget,
        constant=constant,
    )
    queries = build_online_queries(candidates, dense, splade)
    topologies = build_packed_topologies(dataset, queries)
    return OnlineBatch(
        source_query_indices=np.asarray(source_query_indices, dtype=np.int64),
        query_embeddings=np.asarray(query_embeddings, dtype=np.float32),
        queries=queries,
        topologies=topologies,
    )
