"""Diagnostics for the graph substrate that QLS and the GNN actually receive.

Phase -1 of the QLS-v2 programme. Every frozen Paper-1 result was produced on
the *candidate-induced* graph ``G[C_q]``: an edge survives only when both of its
endpoints are Dense/SPLADE candidates
(:func:`mp_retrieval.complete_data.CompleteDataset.induced_subgraph`). Nothing
here changes that; this module measures how much of the original neighbourhood
that induction destroys, so the frozen comparison can be interpreted correctly.

The functions are deliberately free of dataset objects and take plain CSR arrays,
because the substrate question is the same for every dataset and the answer has
to be reproducible from the frozen graph alone.

Read-only. Nothing here trains, tunes, or admits a feature.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SubstrateCounts",
    "induced_view",
    "connectivity_summary",
    "retention_summary",
    "receptive_field_sizes",
    "hop_distances",
    "path_preservation",
    "bridge_loss",
    "distribution",
]

# A distance of this value means "not reached within the hop budget".
UNREACHED = np.iinfo(np.int32).max


def distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Summarize a per-node quantity the way every Phase -1 table reports it."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"{prefix}_{key}": float("nan") for key in _DISTRIBUTION_KEYS}
    percentiles = np.percentile(values, [10, 25, 50, 75, 90, 95])
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_p10": float(percentiles[0]),
        f"{prefix}_p25": float(percentiles[1]),
        f"{prefix}_median": float(percentiles[2]),
        f"{prefix}_p75": float(percentiles[3]),
        f"{prefix}_p90": float(percentiles[4]),
        f"{prefix}_p95": float(percentiles[5]),
        f"{prefix}_max": float(values.max()),
    }


_DISTRIBUTION_KEYS = ("mean", "p10", "p25", "median", "p75", "p90", "p95", "max")


class SubstrateCounts:
    """The induced graph together with the global context it discarded.

    ``induced_subgraph`` computes each candidate's global degree in order to
    expand its CSR row and then throws that number away. It is the denominator
    of the retention statistic, so this class keeps it.
    """

    __slots__ = (
        "edges",
        "global_degree",
        "induced_out_degree",
        "boundary_edges",
        "self_loops",
        "num_candidates",
    )

    def __init__(
        self,
        edges: np.ndarray,
        global_degree: np.ndarray,
        induced_out_degree: np.ndarray,
        boundary_edges: int,
        self_loops: int,
        num_candidates: int,
    ):
        self.edges = edges
        self.global_degree = global_degree
        self.induced_out_degree = induced_out_degree
        self.boundary_edges = boundary_edges
        self.self_loops = self_loops
        self.num_candidates = num_candidates


def induced_view(
    rowptr: np.ndarray,
    col: np.ndarray,
    candidates: np.ndarray,
    *,
    drop_self_loops: bool = True,
) -> SubstrateCounts:
    """Build ``G[C_q]`` in local indices and retain the discarded global counts.

    Mirrors ``CompleteDataset.induced_subgraph`` exactly -- an edge survives only
    when its target is also a candidate -- but additionally returns the global
    out-degree of every candidate and the number of edges that leave the pool.
    Self-loops are excluded by default because every connectivity statistic in
    the protocol is defined on the non-self graph.
    """

    candidates = np.asarray(candidates, dtype=np.int64)
    size = int(candidates.size)
    if size == 0:
        empty = np.empty((2, 0), dtype=np.int64)
        return SubstrateCounts(empty, np.empty(0, np.int64), np.zeros(0, np.int64), 0, 0, 0)

    starts = rowptr[candidates]
    degrees = rowptr[candidates + 1] - starts
    global_degree = degrees.astype(np.int64, copy=True)
    total = int(degrees.sum())
    if total == 0:
        empty = np.empty((2, 0), dtype=np.int64)
        return SubstrateCounts(empty, global_degree, np.zeros(size, np.int64), 0, 0, size)

    source_local = np.repeat(np.arange(size, dtype=np.int64), degrees)
    group_starts = np.repeat(np.cumsum(degrees) - degrees, degrees)
    positions = np.repeat(starts, degrees) + (
        np.arange(total, dtype=np.int64) - group_starts
    )
    neighbors = col[positions]

    order = np.argsort(candidates, kind="stable")
    sorted_candidates = candidates[order]
    slots = np.searchsorted(sorted_candidates, neighbors)
    in_range = slots < sorted_candidates.size
    keep = np.zeros(neighbors.size, dtype=bool)
    keep[in_range] = sorted_candidates[slots[in_range]] == neighbors[in_range]
    target_local = order[slots[keep]].astype(np.int64, copy=False)
    source_kept = source_local[keep]

    self_loop_mask = source_kept == target_local
    self_loops = int(self_loop_mask.sum())
    if drop_self_loops and self_loops:
        source_kept = source_kept[~self_loop_mask]
        target_local = target_local[~self_loop_mask]

    edges = np.stack((source_kept, target_local))
    induced_out_degree = np.bincount(source_kept, minlength=size).astype(np.int64)
    boundary = int(total - int(keep.sum()))
    return SubstrateCounts(edges, global_degree, induced_out_degree, boundary, self_loops, size)


def _undirected_adjacency(edges: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR of the undirected, de-duplicated, self-loop-free induced graph."""

    if edges.shape[1] == 0:
        return np.zeros(size + 1, dtype=np.int64), np.empty(0, dtype=np.int64)
    source = np.concatenate((edges[0], edges[1]))
    target = np.concatenate((edges[1], edges[0]))
    keep = source != target
    source, target = source[keep], target[keep]
    keys = source * size + target
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    unique = np.ones(keys.size, dtype=bool)
    if keys.size:
        unique[1:] = keys[1:] != keys[:-1]
    source = source[order][unique]
    target = target[order][unique]
    counts = np.bincount(source, minlength=size)
    rowptr = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(counts, out=rowptr[1:])
    return rowptr, target.astype(np.int64, copy=False)


def _components(rowptr: np.ndarray, col: np.ndarray, size: int) -> np.ndarray:
    """Component label per node on the undirected view, by iterative BFS."""

    labels = np.full(size, -1, dtype=np.int64)
    label = 0
    for start in range(size):
        if labels[start] >= 0:
            continue
        labels[start] = label
        stack = [start]
        while stack:
            node = stack.pop()
            for position in range(rowptr[node], rowptr[node + 1]):
                neighbor = int(col[position])
                if labels[neighbor] < 0:
                    labels[neighbor] = label
                    stack.append(neighbor)
        label += 1
    return labels


def connectivity_summary(counts: SubstrateCounts) -> dict[str, float]:
    """Isolates, degree distribution and component structure of ``G[C_q]``.

    Degrees are undirected and self-loop free, so ``degree 0`` genuinely means
    "this candidate has no neighbour inside the pool" -- the condition under
    which a message-passing layer has nothing to aggregate.
    """

    size = counts.num_candidates
    if size == 0:
        return {"candidates": 0.0, "edges": 0.0}
    rowptr, col = _undirected_adjacency(counts.edges, size)
    degree = (rowptr[1:] - rowptr[:-1]).astype(np.int64)
    labels = _components(rowptr, col, size)
    component_sizes = np.bincount(labels)
    ordered = np.sort(component_sizes)[::-1]
    summary = {
        "candidates": float(size),
        "edges_directed_non_self": float(counts.edges.shape[1]),
        "edges_undirected_non_self": float(col.size // 2),
        "self_loops": float(counts.self_loops),
        "isolated_fraction": float((degree == 0).mean()),
        "degree_1_fraction": float((degree == 1).mean()),
        "degree_ge_2_fraction": float((degree >= 2).mean()),
        "mean_degree": float(degree.mean()),
        "median_degree": float(np.median(degree)),
        "components": float(component_sizes.size),
        "largest_component_fraction": float(ordered[0] / size),
        "second_component_fraction": float(ordered[1] / size) if ordered.size > 1 else 0.0,
    }
    summary.update(distribution(degree, "degree"))
    return summary


def retention_summary(counts: SubstrateCounts) -> dict[str, float]:
    """How much of each candidate's real neighbourhood survived retrieval.

    ``retention(v) = induced_degree(v) / global_degree(v)``. A candidate with
    global degree 40 and induced degree 1 has lost 97.5% of the context a
    conventional GNN would aggregate over.
    """

    global_degree = counts.global_degree.astype(np.float64)
    induced = counts.induced_out_degree.astype(np.float64)
    has_neighbors = global_degree > 0
    if not has_neighbors.any():
        return {
            "candidates_with_global_neighbors": 0.0,
            "boundary_cut_ratio": float("nan"),
        }
    retention = np.zeros(global_degree.size, dtype=np.float64)
    retention[has_neighbors] = induced[has_neighbors] / global_degree[has_neighbors]
    observed = retention[has_neighbors]
    total_incident = float(global_degree.sum())
    summary = {
        "candidates_with_global_neighbors": float(has_neighbors.sum()),
        "retention_zero_fraction": float((observed == 0.0).mean()),
        "retention_below_10pct_fraction": float((observed < 0.10).mean()),
        "retention_below_25pct_fraction": float((observed < 0.25).mean()),
        "boundary_cut_ratio": float(counts.boundary_edges / total_incident)
        if total_incident
        else float("nan"),
    }
    summary.update(distribution(observed, "retention"))
    return summary


def receptive_field_sizes(
    counts: SubstrateCounts, *, max_hops: int = 3
) -> dict[str, float]:
    """Distinct non-self nodes each candidate can reach within ``h`` layers.

    This is the quantity a message-passing layer actually aggregates over. A
    candidate whose ``R1`` is 0 receives no message at all, and for a one-layer
    GNN that candidate is scored by an MLP.
    """

    size = counts.num_candidates
    if size == 0:
        return {}
    rowptr, col = _undirected_adjacency(counts.edges, size)
    reachable = np.eye(size, dtype=bool)
    frontier = np.eye(size, dtype=bool)
    summary: dict[str, float] = {}
    for hop in range(1, max_hops + 1):
        nxt = np.zeros_like(frontier)
        rows = np.nonzero(frontier)
        for node, source in zip(rows[0], rows[1]):
            nxt[node, col[rowptr[source] : rowptr[source + 1]]] = True
        frontier = nxt & ~reachable
        reachable |= nxt
        sizes = reachable.sum(axis=1) - 1  # exclude self
        summary[f"R{hop}_median"] = float(np.median(sizes))
        summary[f"R{hop}_mean"] = float(sizes.mean())
        summary[f"R{hop}_zero_fraction"] = float((sizes == 0).mean())
    return summary


def hop_distances(
    rowptr: np.ndarray,
    col: np.ndarray,
    sources: np.ndarray,
    size: int,
    *,
    max_hops: int = 3,
) -> np.ndarray:
    """Multi-source BFS distance to every node, ``UNREACHED`` beyond the budget.

    Works on any CSR, so the same function measures the induced graph and the
    global graph. That is the point: the two must be measured identically for
    their difference to mean anything.
    """

    distance = np.full(size, UNREACHED, dtype=np.int64)
    frontier = np.unique(np.asarray(sources, dtype=np.int64))
    if frontier.size == 0:
        return distance
    distance[frontier] = 0
    for hop in range(1, max_hops + 1):
        starts = rowptr[frontier]
        degrees = rowptr[frontier + 1] - starts
        total = int(degrees.sum())
        if total == 0:
            break
        group_starts = np.repeat(np.cumsum(degrees) - degrees, degrees)
        positions = np.repeat(starts, degrees) + (
            np.arange(total, dtype=np.int64) - group_starts
        )
        neighbors = np.unique(col[positions])
        fresh = neighbors[distance[neighbors] == UNREACHED]
        if fresh.size == 0:
            break
        distance[fresh] = hop
        frontier = fresh
    return distance


def path_preservation(
    global_distance: np.ndarray,
    induced_distance: np.ndarray,
    *,
    max_hops: int = 3,
) -> dict[str, float]:
    """Classify what candidate induction did to each seed-to-target relationship.

    ``P_h = P(d_induced <= h | d_global <= h)``: of the relationships that really
    exist within ``h`` hops, the fraction the induced graph still expresses.
    """

    global_distance = np.asarray(global_distance, dtype=np.int64)
    induced_distance = np.asarray(induced_distance, dtype=np.int64)
    if global_distance.shape != induced_distance.shape:
        raise ValueError("Distance arrays must be aligned")
    if global_distance.size == 0:
        return {}
    connected_globally = global_distance <= max_hops
    connected_induced = induced_distance <= max_hops
    summary = {
        "targets": float(global_distance.size),
        "connected_globally_fraction": float(connected_globally.mean()),
        "connected_induced_fraction": float(connected_induced.mean()),
        "globally_connected_but_induced_disconnected": float(
            (connected_globally & ~connected_induced).sum()
        ),
        "already_disconnected_globally": float((~connected_globally).sum()),
    }
    both = connected_globally & connected_induced
    summary["distance_inflated_fraction"] = (
        float((induced_distance[both] > global_distance[both]).mean()) if both.any() else 0.0
    )
    summary["mean_distance_inflation"] = (
        float((induced_distance[both] - global_distance[both]).mean()) if both.any() else 0.0
    )
    for hop in range(1, max_hops + 1):
        eligible = global_distance <= hop
        preserved = eligible & (induced_distance <= hop)
        summary[f"path_preservation_at_{hop}"] = (
            float(preserved.sum() / eligible.sum()) if eligible.any() else float("nan")
        )
        summary[f"eligible_at_{hop}"] = float(eligible.sum())
    return summary


def bridge_loss(
    global_distance: np.ndarray,
    induced_distance: np.ndarray,
    *,
    max_hops: int = 3,
) -> dict[str, float]:
    """Relationships destroyed because every connecting path left the pool.

    ``bridge_loss@h`` is the complement of ``path_preservation@h``: among targets
    genuinely within ``h`` hops of a seed, the fraction the induced graph cannot
    reach at all within ``h``. It isolates the ``seed -> non-candidate ->
    candidate`` pattern that neither a GNN layer nor a QLS hop feature can see.
    """

    preservation = path_preservation(
        global_distance, induced_distance, max_hops=max_hops
    )
    summary: dict[str, float] = {}
    for hop in range(1, max_hops + 1):
        rate = preservation.get(f"path_preservation_at_{hop}", float("nan"))
        summary[f"bridge_loss_at_{hop}"] = float("nan") if np.isnan(rate) else float(1.0 - rate)
        summary[f"eligible_at_{hop}"] = preservation.get(f"eligible_at_{hop}", 0.0)
    return summary
