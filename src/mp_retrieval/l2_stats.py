"""Query-conditioned graph-quality statistics for candidate retrieval graphs."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .graph_stats import degree_statistics
from .l2_data import CandidateQuery


def _mean(values: torch.Tensor) -> float | None:
    return float(values.float().mean()) if values.numel() else None


def _cosine_alignment(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    seed: int,
    max_edges: int = 50_000,
) -> dict[str, float | None]:
    edge_count = edge_index.shape[1]
    if edge_count == 0:
        return {
            "edge_feature_cosine": None,
            "random_pair_feature_cosine": None,
            "feature_similarity_lift": None,
        }
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if edge_count > max_edges:
        chosen = torch.randperm(edge_count, generator=gen)[:max_edges]
        edge_index = edge_index[:, chosen]
    x = F.normalize(features.detach().cpu().float(), dim=-1)
    source, target = edge_index.cpu()
    observed = (x[source] * x[target]).sum(dim=-1)
    random_target = torch.randint(x.shape[0], (source.numel(),), generator=gen)
    random_similarity = (x[source] * x[random_target]).sum(dim=-1)
    return {
        "edge_feature_cosine": _mean(observed),
        "random_pair_feature_cosine": _mean(random_similarity),
        "feature_similarity_lift": float(observed.mean() - random_similarity.mean()),
    }


def _edge_type_statistics(edge_type: torch.Tensor | None) -> dict[str, Any]:
    if edge_type is None:
        return {
            "edge_types_available": False,
            "num_edge_types": None,
            "edge_type_entropy": None,
            "normalized_edge_type_entropy": None,
            "edge_type_counts": None,
        }
    counts = Counter(int(value) for value in edge_type.cpu().tolist())
    total = max(sum(counts.values()), 1)
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalized = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
    return {
        "edge_types_available": True,
        "num_edge_types": len(counts),
        "edge_type_entropy": entropy,
        "normalized_edge_type_entropy": normalized,
        "edge_type_counts": {str(key): value for key, value in sorted(counts.items())},
    }


def candidate_graph_statistics(
    query: CandidateQuery,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor | None,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute one query's structural and task-aligned graph statistics."""

    num_nodes = int(query.candidate_index.numel())
    num_edges = int(edge_index.shape[1])
    positives = torch.zeros(num_nodes, dtype=torch.bool)
    positives[query.relevant_local.cpu()] = True
    prevalence = float(positives.float().mean())
    source, target = edge_index.cpu()
    positive_source_edges = positives[source] if num_edges else torch.zeros(0, dtype=torch.bool)
    positive_neighbor_rate = (
        _mean(positives[target[positive_source_edges]]) if positive_source_edges.any() else None
    )
    neighborhood_noise = (
        None if positive_neighbor_rate is None else 1.0 - positive_neighbor_rate
    )
    base_noise = 1.0 - prevalence
    edge_homophily = _mean(positives[source] == positives[target]) if num_edges else None
    result: dict[str, Any] = {
        "query_id": query.query_id,
        "split": int(query.split),
        "num_candidates": num_nodes,
        "num_edges": num_edges,
        "directed_density": num_edges / max(num_nodes * (num_nodes - 1), 1),
        "candidate_ceiling": query.candidate_ceiling,
        "positive_prevalence": prevalence,
        "label_edge_homophily": edge_homophily,
        "positive_neighbor_rate": positive_neighbor_rate,
        "positive_neighbor_lift": (
            None if positive_neighbor_rate is None else positive_neighbor_rate - prevalence
        ),
        "neighborhood_noise": neighborhood_noise,
        "neighborhood_noise_excess_over_random": (
            None if neighborhood_noise is None else neighborhood_noise - base_noise
        ),
    }
    result.update(degree_statistics(edge_index, num_nodes))
    result.update(_cosine_alignment(features, edge_index, seed=seed))
    result.update(_edge_type_statistics(edge_type))
    return result


def aggregate_query_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric query statistics without hiding unavailable fields."""

    if not rows:
        return {"queries": 0}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    aggregate: dict[str, Any] = {"queries": len(rows)}
    for key in keys:
        values = [row.get(key) for row in rows]
        numeric = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None
        ]
        if not numeric:
            continue
        array = np.asarray(numeric, dtype=np.float64)
        aggregate[key] = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "p05": float(np.quantile(array, 0.05)),
            "median": float(np.quantile(array, 0.5)),
            "p95": float(np.quantile(array, 0.95)),
            "available_queries": len(numeric),
        }
    aggregate["edge_types_available"] = all(
        bool(row["edge_types_available"]) for row in rows
    )
    return aggregate
