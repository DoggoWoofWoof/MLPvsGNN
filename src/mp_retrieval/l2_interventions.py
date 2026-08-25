"""Shared query-local interventions for the controlled phase diagram."""

from __future__ import annotations

from typing import Any

import torch

from .l2_data import CandidateQuery
from .perturbations import TopologyPerturbation, degrade_features, remove_edge_types


def apply_intervention(
    queries: list[CandidateQuery],
    features: dict[CandidateQuery, torch.Tensor],
    edges: dict[CandidateQuery, torch.Tensor],
    edge_types: dict[CandidateQuery, torch.Tensor | None],
    *,
    kind: str,
    rate: float,
    seed: int,
    removed_edge_types: set[int] | None = None,
) -> tuple[
    dict[CandidateQuery, torch.Tensor],
    dict[CandidateQuery, torch.Tensor],
    dict[CandidateQuery, torch.Tensor | None],
    dict[str, Any],
]:
    """Apply one intervention identically before any model sees the data."""

    if kind == "clean":
        return features, edges, edge_types, {
            "kind": "clean",
            "rate": 0.0,
            "seed": seed,
            "stage": "none",
        }
    changed_features = dict(features)
    changed_edges = dict(edges)
    changed_types = dict(edge_types)
    before_edges = sum(int(value.shape[1]) for value in edges.values())
    changed_edge_positions = 0
    comparable_edge_positions = 0
    skipped_empty = 0
    for query_idx, query in enumerate(sorted(queries, key=lambda item: item.query_id)):
        query_seed = seed + query_idx * 1_000_003
        if kind in {"feature_mask", "feature_gaussian"}:
            mode = "mask" if kind == "feature_mask" else "gaussian"
            changed_features[query] = degrade_features(
                features[query], rate, seed=query_seed, mode=mode
            )
            continue
        if kind == "typed_edge_removal":
            types = edge_types[query]
            if types is None:
                raise ValueError(
                    "typed_edge_removal is unavailable: this artifact has no edge types. "
                    "Export the canonical typed graph instead of treating untyped edges as one type."
                )
            selected_types = removed_edge_types or set()
            if not selected_types:
                raise ValueError("typed_edge_removal requires --remove-edge-types")
            changed_edges[query], changed_types[query] = remove_edge_types(
                edges[query], types, selected_types
            )
            continue
        if edges[query].shape[1] == 0:
            skipped_empty += 1
            continue
        changed_edges[query], changed_types[query] = TopologyPerturbation(
            kind=kind,
            rate=rate,
            seed=query_seed,
        ).apply(
            edges[query],
            num_nodes=int(query.candidate_index.numel()),
            edge_type=edge_types[query],
        )
        if changed_edges[query].shape == edges[query].shape:
            comparable_edge_positions += int(edges[query].shape[1])
            changed_edge_positions += int(
                (changed_edges[query] != edges[query]).any(dim=0).sum()
            )
    after_edges = sum(int(value.shape[1]) for value in changed_edges.values())
    stage = "shared_model_input" if kind.startswith("feature_") else "gnn_topology"
    algorithm = {
        "degree_rewire": "nested_degree_preserving_switch_v2",
        "add_random": "unique_uniform_directed_addition_v1",
        "hub_injection": "existing_high_degree_target_redirect_v1",
        "feature_mask": "entrywise_bernoulli_mask_v1",
        "feature_gaussian": "column_scaled_gaussian_v1",
        "typed_edge_removal": "relation_id_filter_v1",
        "drop": "uniform_directed_edge_drop_v1",
    }.get(kind)
    return changed_features, changed_edges, changed_types, {
        "kind": kind,
        "rate": rate,
        "seed": seed,
        "algorithm": algorithm,
        "stage": stage,
        "applied_before_model_construction": True,
        "identical_degraded_features_for_all_models": kind.startswith("feature_"),
        "edges_before": before_edges,
        "edges_after": after_edges,
        "changed_edge_positions": changed_edge_positions,
        "achieved_changed_edge_fraction": (
            changed_edge_positions / comparable_edge_positions
            if comparable_edge_positions
            else None
        ),
        "empty_graph_queries_skipped": skipped_empty,
        "removed_edge_types": sorted(removed_edge_types or set()),
    }
