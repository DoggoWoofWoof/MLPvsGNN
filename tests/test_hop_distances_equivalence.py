"""`hop_distances` was made faster; it must not have been made different.

Five of the six substrate audits are already complete and their numbers are
reported. This function produced them, so an optimization that changes any
output by one hop silently invalidates finished work rather than failing --
which is why the change is pinned against a reference implementation of the
original algorithm rather than against a handful of expected arrays.

The optimization is an ordering change. The original uniqued every gathered
neighbour and then dropped the already-reached ones; this filters first and
uniques only what survives. The two agree because uniquing a filtered set and
filtering a uniqued set produce the same set, and `np.unique` returns it sorted
either way. On hotpotqa's global graph that reordering is worth 2.2x: by the
third hop nearly every neighbour has already been reached, so the original sorted
millions of entries in order to keep a few thousand.
"""

from __future__ import annotations

import numpy as np
import pytest

from mp_retrieval.graph_substrate import UNREACHED, hop_distances


def reference_hop_distances(rowptr, col, sources, size, *, max_hops=3):
    """The original algorithm, kept verbatim as the thing to agree with."""
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


def _random_csr(rng, nodes, edges):
    src = rng.integers(0, nodes, edges)
    dst = rng.integers(0, nodes, edges)
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    rowptr = np.zeros(nodes + 1, dtype=np.int64)
    np.add.at(rowptr, src + 1, 1)
    return np.cumsum(rowptr), dst.astype(np.int64)


@pytest.mark.parametrize("max_hops", [1, 2, 3, 4])
def test_the_optimized_traversal_matches_the_original_on_random_graphs(max_hops):
    """Random shapes, because the failure would be structural, not numeric.

    Duplicate edges, self-loops, isolated nodes and multiple seeds all arise
    here; each is a case where filter-then-unique could have diverged from
    unique-then-filter if the reasoning were wrong.
    """
    rng = np.random.default_rng(max_hops)
    for _ in range(150):
        nodes = int(rng.integers(2, 60))
        rowptr, col = _random_csr(rng, nodes, int(rng.integers(1, 260)))
        seeds = rng.choice(nodes, size=int(rng.integers(1, min(nodes, 5) + 1)), replace=False)
        assert np.array_equal(
            hop_distances(rowptr, col, seeds, nodes, max_hops=max_hops),
            reference_hop_distances(rowptr, col, seeds, nodes, max_hops=max_hops),
        )


def test_a_dense_graph_where_every_node_is_reached_early():
    """The case the optimization targets: the frontier saturates the graph.

    Once everything is reached, the original still gathered and sorted every
    neighbour of a large frontier to discover that nothing was new.
    """
    nodes = 40
    src = np.repeat(np.arange(nodes), nodes)
    dst = np.tile(np.arange(nodes), nodes)
    rowptr = np.zeros(nodes + 1, dtype=np.int64)
    np.add.at(rowptr, src + 1, 1)
    rowptr = np.cumsum(rowptr)
    assert np.array_equal(
        hop_distances(rowptr, dst.astype(np.int64), np.array([0]), nodes, max_hops=3),
        reference_hop_distances(rowptr, dst.astype(np.int64), np.array([0]), nodes, max_hops=3),
    )


def test_repeated_edges_do_not_shift_a_distance():
    """Multiplicity is exactly what changed position in the pipeline."""
    rowptr = np.array([0, 3, 4, 4], dtype=np.int64)
    col = np.array([1, 1, 1, 2], dtype=np.int64)
    distance = hop_distances(rowptr, col, np.array([0]), 3, max_hops=2)
    assert distance.tolist() == [0, 1, 2]
    assert np.array_equal(distance, reference_hop_distances(rowptr, col, np.array([0]), 3, max_hops=2))


def test_unreachable_nodes_stay_unreached():
    rowptr = np.array([0, 1, 1, 1], dtype=np.int64)
    col = np.array([1], dtype=np.int64)
    distance = hop_distances(rowptr, col, np.array([0]), 3, max_hops=3)
    assert distance[2] == UNREACHED


def test_a_seed_keeps_distance_zero_even_with_a_self_loop():
    rowptr = np.array([0, 2, 2], dtype=np.int64)
    col = np.array([0, 1], dtype=np.int64)
    assert hop_distances(rowptr, col, np.array([0]), 2, max_hops=3).tolist() == [0, 1]


def test_no_sources_reaches_nothing():
    rowptr = np.array([0, 1, 1], dtype=np.int64)
    col = np.array([1], dtype=np.int64)
    distance = hop_distances(rowptr, col, np.array([], dtype=np.int64), 2, max_hops=3)
    assert (distance == UNREACHED).all()


def test_the_hop_budget_is_still_respected():
    """A faster traversal must not reach further, only sooner."""
    nodes = 6
    rowptr = np.arange(nodes + 1, dtype=np.int64)
    col = np.arange(1, nodes + 1, dtype=np.int64) % nodes
    for budget in (1, 2, 3):
        distance = hop_distances(rowptr, col, np.array([0]), nodes, max_hops=budget)
        reached = distance[distance != UNREACHED]
        assert reached.max() <= budget
