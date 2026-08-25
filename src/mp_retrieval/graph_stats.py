"""Pre-training graph-quality statistics and query-conditioned signal measures."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .data import GraphRetrievalData


def _sample_indices(size: int, limit: int, seed: int) -> torch.Tensor:
    if size <= limit:
        return torch.arange(size)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(size, generator=generator)[:limit]


def gini(values: torch.Tensor) -> float:
    """Gini coefficient for non-negative values."""

    array = values.detach().cpu().double().flatten()
    if array.numel() == 0 or float(array.sum()) == 0.0:
        return 0.0
    array = array.sort().values
    n = array.numel()
    ranks = torch.arange(1, n + 1, dtype=torch.double)
    return float((2 * (ranks * array).sum() / (n * array.sum())) - (n + 1) / n)


def degree_statistics(edge_index: torch.Tensor, num_nodes: int) -> dict[str, float]:
    src, dst = edge_index.cpu()
    out_degree = torch.bincount(src, minlength=num_nodes).float()
    in_degree = torch.bincount(dst, minlength=num_nodes).float()
    total = in_degree + out_degree
    mean = float(total.mean()) if num_nodes else 0.0
    return {
        "degree_mean": mean,
        "degree_std": float(total.std(unbiased=False)) if num_nodes else 0.0,
        "degree_p95": float(torch.quantile(total, 0.95)) if num_nodes else 0.0,
        "degree_max": float(total.max()) if num_nodes else 0.0,
        "degree_gini": gini(total),
        "hubness_max_over_mean": float(total.max() / max(mean, 1e-12)) if num_nodes else 0.0,
        "isolated_fraction": float((total == 0).float().mean()) if num_nodes else 0.0,
    }


def feature_edge_alignment(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    max_edges: int = 200_000,
    seed: int = 0,
) -> dict[str, float]:
    """Cosine similarity on observed edges versus independently paired nodes."""

    selected = _sample_indices(edge_index.shape[1], max_edges, seed)
    src, dst = edge_index.cpu()[:, selected]
    x = F.normalize(features.detach().cpu().float(), dim=-1)
    observed = (x[src] * x[dst]).sum(dim=-1)
    random_dst = torch.randint(
        features.shape[0], (selected.numel(),), generator=torch.Generator().manual_seed(seed + 1)
    )
    null = (x[src] * x[random_dst]).sum(dim=-1)
    return {
        "edge_cosine_mean": float(observed.mean()) if observed.numel() else 0.0,
        "edge_cosine_std": float(observed.std(unbiased=False)) if observed.numel() else 0.0,
        "random_pair_cosine_mean": float(null.mean()) if null.numel() else 0.0,
        "feature_alignment_lift": float(observed.mean() - null.mean()) if observed.numel() else 0.0,
    }


def query_neighborhood_signal(
    edge_index: torch.Tensor,
    relevance: list[torch.Tensor],
    num_nodes: int,
    *,
    query_indices: Iterable[int] | None = None,
    max_queries: int = 1_000,
    seed: int = 0,
) -> dict[str, float]:
    """Measure whether a relevant node's outgoing neighbors are also relevant.

    ``positive_neighbor_lift`` subtracts each query's relevance prevalence from
    its positive-neighbor rate, reducing the trivial effect of answer-set size.
    This is a task-aligned homophily statistic for retrieval, not class-label
    homophily imported from node classification.
    """

    candidates = list(range(len(relevance))) if query_indices is None else list(query_indices)
    if len(candidates) > max_queries:
        chosen = _sample_indices(len(candidates), max_queries, seed).tolist()
        candidates = [candidates[i] for i in chosen]
    src, dst = edge_index.cpu()
    rates: list[float] = []
    lifts: list[float] = []
    coverages: list[float] = []
    for query_idx in candidates:
        positives = relevance[query_idx].cpu()
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[positives] = True
        from_positive = mask[src]
        denom = int(from_positive.sum())
        if denom == 0:
            rates.append(0.0)
            lifts.append(-float(positives.numel() / num_nodes))
            coverages.append(0.0)
            continue
        rate = float(mask[dst[from_positive]].float().mean())
        rates.append(rate)
        lifts.append(rate - float(positives.numel() / num_nodes))
        positive_sources_with_positive_neighbor = src[from_positive & mask[dst]].unique()
        covered = torch.isin(positives, positive_sources_with_positive_neighbor)
        coverages.append(float(covered.float().mean()))
    return {
        "positive_neighbor_rate": float(np.mean(rates)) if rates else 0.0,
        "positive_neighbor_lift": float(np.mean(lifts)) if lifts else 0.0,
        "relevant_neighbor_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "signal_queries": float(len(candidates)),
    }


def graph_statistics(
    data: GraphRetrievalData,
    *,
    query_indices: Iterable[int] | None = None,
    seed: int = 0,
) -> dict[str, float | dict[str, int]]:
    """Compute the pre-registered statistics used by the crossover predictor."""

    data.validate()
    possible_edges = max(data.num_nodes * (data.num_nodes - 1), 1)
    result: dict[str, float | dict[str, int]] = {
        "num_nodes": float(data.num_nodes),
        "num_edges": float(data.num_edges),
        "directed_density": float(data.num_edges / possible_edges),
    }
    result.update(degree_statistics(data.edge_index, data.num_nodes))
    result.update(feature_edge_alignment(data.node_features, data.edge_index, seed=seed))
    result.update(
        query_neighborhood_signal(
            data.edge_index,
            data.relevance,
            data.num_nodes,
            query_indices=query_indices,
            seed=seed,
        )
    )
    if data.edge_type is not None:
        counts = Counter(int(x) for x in data.edge_type.cpu().tolist())
        result["edge_type_counts"] = {str(k): v for k, v in sorted(counts.items())}
        result["num_edge_types"] = float(len(counts))
    return result

