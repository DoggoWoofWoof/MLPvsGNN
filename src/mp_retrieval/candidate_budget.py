"""Shared post-fusion candidate budgets for the matched QLS/GNN study."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - production Modal image includes Numba
    prange = range

    def njit(*args, **kwargs):
        del args, kwargs

        def decorate(function):
            return function

        return decorate

from .complete_data import (
    CompleteQuery,
    CompleteRetrievalDataset,
    complete_query_contract_sha256,
)
from .rank_fusion import rrf_rankings
from .topology_store import PackedLocalTopologies


def _stable_union(*rows: np.ndarray) -> np.ndarray:
    combined = np.concatenate(rows).astype(np.int64, copy=False)
    _values, first = np.unique(combined, return_index=True)
    return combined[np.sort(first)]


def build_budget_dataset(
    dataset: CompleteRetrievalDataset,
    dense: np.ndarray,
    splade: np.ndarray,
    *,
    budget: int,
    rrf_constant: int = 60,
    chunk_size: int = 4096,
) -> CompleteRetrievalDataset:
    """Truncate every frozen union using one locked equal-RRF ordering."""

    dense = np.asarray(dense)
    splade = np.asarray(splade)
    if dense.shape != splade.shape or dense.ndim != 2:
        raise ValueError("Dense and SPLADE rankings must be aligned matrices")
    if dense.shape[0] != len(dataset.queries):
        raise ValueError("Rank matrices do not align with the complete query contract")
    maximum = 2 * dense.shape[1]
    if budget <= 0 or budget > maximum:
        raise ValueError(f"Candidate budget must lie in [1, {maximum}]")

    queries: list[CompleteQuery] = []
    for start in range(0, len(dataset.queries), chunk_size):
        end = min(start + chunk_size, len(dataset.queries))
        ranked = rrf_rankings(
            dense[start:end],
            splade[start:end],
            dense_weights=[0.5],
            constant=rrf_constant,
            top_k=maximum,
        )[0.5]
        for local_row, source_query in enumerate(dataset.queries[start:end]):
            unique_count = int(source_query.candidate_index.numel())
            candidate_values = np.asarray(
                ranked[local_row, : min(budget, unique_count)], dtype=np.int64
            )
            candidates = torch.from_numpy(candidate_values.copy())
            local = {int(node): index for index, node in enumerate(candidate_values)}
            relevant_local = torch.tensor(
                [
                    local[int(gold)]
                    for gold in source_query.relevant_global.tolist()
                    if int(gold) in local
                ],
                dtype=torch.long,
            )
            seed_global = _stable_union(
                dense[source_query.query_index, :5],
                splade[source_query.query_index, :5],
            )
            seed_local = torch.tensor(
                [local[int(node)] for node in seed_global if int(node) in local],
                dtype=torch.long,
            )
            queries.append(
                CompleteQuery(
                    query_index=source_query.query_index,
                    query_id=source_query.query_id,
                    candidate_index=candidates,
                    relevant_local=relevant_local,
                    relevant_global=source_query.relevant_global.clone(),
                    anchor_global=source_query.anchor_global,
                    split=source_query.split,
                    hop=source_query.hop,
                    retrieval_seed_local=seed_local,
                )
            )
    metadata = {
        **dataset.metadata,
        "parent_candidate_contract_sha256": dataset.metadata[
            "candidate_contract_sha256"
        ],
        "candidate_contract_sha256": complete_query_contract_sha256(queries),
        "candidate_budget": budget,
        "candidate_ordering": "equal_rrf",
        "rrf_constant": rrf_constant,
        "rrf_dense_weight": 0.5,
        "rrf_splade_weight": 0.5,
        "rrf_tie_break": "ascending_global_node_id",
    }
    return replace(dataset, queries=queries, metadata=metadata)


@njit(cache=True, parallel=True)
def _component_counts(
    lengths: np.ndarray,
    edge_ptr: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    output = np.empty(lengths.size, dtype=np.int32)
    for query in prange(lengths.size):
        size = lengths[query]
        parent = np.arange(size, dtype=np.int32)
        for cursor in range(edge_ptr[query], edge_ptr[query + 1]):
            left = edge_index[0, cursor]
            right = edge_index[1, cursor]
            while parent[left] != left:
                parent[left] = parent[parent[left]]
                left = parent[left]
            while parent[right] != right:
                parent[right] = parent[parent[right]]
                right = parent[right]
            if left != right:
                parent[right] = left
        components = 0
        for node in range(size):
            root = node
            while parent[root] != root:
                root = parent[root]
            if root == node:
                components += 1
        output[query] = components
    return output


def structural_context_statistics(
    queries: list[CompleteQuery],
    topologies: PackedLocalTopologies,
) -> dict[str, float | int]:
    """Summarize ceiling, graph size/density, and connectivity at one budget."""

    lengths = np.asarray([query.candidate_index.numel() for query in queries], dtype=np.int32)
    edges = np.diff(topologies.edge_ptr).astype(np.int64, copy=False)
    if lengths.size != edges.size:
        raise ValueError("Topology rows do not align with the budgeted queries")
    ceilings = np.asarray([query.candidate_ceiling for query in queries], dtype=np.float64)
    denominators = np.maximum(lengths.astype(np.int64) * (lengths - 1), 1)
    density = edges / denominators
    components = _component_counts(lengths, topologies.edge_ptr, topologies.edge_index)

    def summary(prefix: str, values: np.ndarray) -> dict[str, float | int]:
        return {
            f"{prefix}_mean": float(values.mean()) if values.size else 0.0,
            f"{prefix}_p50": float(np.percentile(values, 50)) if values.size else 0.0,
            f"{prefix}_p95": float(np.percentile(values, 95)) if values.size else 0.0,
            f"{prefix}_p99": float(np.percentile(values, 99)) if values.size else 0.0,
            f"{prefix}_max": float(values.max()) if values.size else 0.0,
        }

    return {
        "queries": int(lengths.size),
        **summary("candidate_count", lengths),
        **summary("candidate_ceiling", ceilings),
        **summary("stored_directed_edges", edges),
        **summary("stored_directed_density", density),
        **summary("connected_components", components),
    }
