"""The optimised traversals must return the *identical* numbers, not close ones.

Phase -1's audit was 3.31 h per graph family, and a profile at hotpotqa's shape
put 74% of that in one call -- the global seed BFS -- and a further 25% in
`expansion_sizes`. Both were rewritten for speed. Neither may move a measured
quantity, because the frozen substrate decision is read off these values.

So every test here compares the fast path against the original algorithm rather
than against a stored expectation: a stored number would drift silently if the
definition changed, whereas a reference implementation fails loudly.

The overflow cases are not decoration. The first version of the mat-vec path
used a `uint8` accumulator, which wraps at 256. What overflows is the number of
frontier *edges* arriving at one node -- 256 of them sum to zero, which reads as
"no frontier edge", so a reached node was silently reported unreachable. Sixty
random graphs passed it, because none accumulated that much at a single node.
Both routes to it are pinned below: parallel edges, and 256 distinct sources
into one target, which is what a hub on the real 507k-node graph looks like.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.graph_substrate import (  # noqa: E402
    UNREACHED,
    _neighbourhood,
    expansion_sizes,
    hop_distances,
    traversal_matrix,
)

HOPS = (1, 2, 3, 4)


def csr(sources: np.ndarray, targets: np.ndarray, size: int):
    order = np.argsort(sources, kind="stable")
    sources, targets = sources[order], targets[order]
    rowptr = np.zeros(size + 1, dtype=np.int64)
    np.add.at(rowptr, sources + 1, 1)
    np.cumsum(rowptr, out=rowptr)
    return rowptr, targets.astype(np.int64)


def random_symmetric(rng, size, edges):
    a = rng.integers(0, size, edges)
    b = rng.integers(0, size, edges)
    return csr(np.concatenate([a, b]), np.concatenate([b, a]), size)


def reference_expansion(rowptr, col, pool, seeds, *, max_hops):
    """The original algorithm: a fresh neighbourhood walk per hop."""

    pool = np.unique(np.asarray(pool, dtype=np.int64))
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    base = float(pool.size)
    summary = {"candidates": base}
    for hop in range(1, max_hops + 1):
        u_seed = np.union1d(pool, _neighbourhood(rowptr, col, seeds, hop))
        u_target = np.union1d(pool, _neighbourhood(rowptr, col, pool, hop))
        summary[f"U_seed_{hop}_nodes"] = float(u_seed.size)
        summary[f"U_target_{hop}_nodes"] = float(u_target.size)
        summary[f"U_seed_{hop}_expansion"] = (
            float(u_seed.size / base) if base else float("nan")
        )
        summary[f"U_target_{hop}_expansion"] = (
            float(u_target.size / base) if base else float("nan")
        )
    return summary


def same_summary(left, right):
    if set(left) != set(right):
        return False
    for key, value in left.items():
        other = right[key]
        if np.isnan(value) and np.isnan(other):
            continue
        if value != other:
            return False
    return True


# --------------------------------------------------------------------------
# hop_distances: the mat-vec path is the same traversal
# --------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(25))
def test_matrix_path_matches_the_gather_path(trial):
    rng = np.random.default_rng(trial)
    size = int(rng.integers(30, 400))
    rowptr, col = random_symmetric(rng, size, int(rng.integers(0, 4 * size)))
    sources = np.unique(rng.integers(0, size, size=int(rng.integers(1, 6))))
    matrix = traversal_matrix(rowptr, col, size)
    for hops in HOPS:
        expected = hop_distances(rowptr, col, sources, size, max_hops=hops)
        actual = hop_distances(
            rowptr, col, sources, size, max_hops=hops, matrix=matrix
        )
        assert np.array_equal(expected, actual)


@pytest.mark.parametrize("degree", [255, 256, 257, 512, 1024])
def test_parallel_edges_past_the_accumulator_still_reach(degree):
    """`uint8` wrapped here and called a reached node UNREACHED.

    The quantity that overflows is the number of frontier *edges* arriving at
    one node, not the number of distinct neighbours. This builds it the cheap
    way: four nodes, with ``degree`` parallel edges all running 0 -> 1. At
    exactly 256 the sum is 0, indistinguishable from "no frontier edge" -- the
    failure is silent and yields a plausible number.
    """
    size = 4
    rowptr = np.array([0, degree, degree, degree, degree], dtype=np.int64)
    col = np.full(degree, 1, dtype=np.int64)
    matrix = traversal_matrix(rowptr, col, size)
    actual = hop_distances(
        rowptr, col, np.array([0]), size, max_hops=2, matrix=matrix
    )
    assert actual[1] == 1
    assert np.array_equal(
        actual, hop_distances(rowptr, col, np.array([0]), size, max_hops=2)
    )


@pytest.mark.parametrize("frontier_size", [256, 512])
def test_distinct_frontier_neighbours_past_the_accumulator_still_reach(frontier_size):
    """The same overflow by the route the real graph actually takes.

    No duplicate edges here: ``frontier_size`` distinct sources each contribute
    one edge to a single target, which is what a hub on a 507k-node graph looks
    like at hop two. A simple-graph invariant elsewhere would not rescue the
    uint8 version -- the sum is over edges either way.
    """
    size = frontier_size + 1
    target = frontier_size
    sources = np.arange(frontier_size, dtype=np.int64)
    rowptr, col = csr(sources, np.full(frontier_size, target, dtype=np.int64), size)
    matrix = traversal_matrix(rowptr, col, size)
    actual = hop_distances(rowptr, col, sources, size, max_hops=2, matrix=matrix)
    assert actual[target] == 1
    assert np.array_equal(
        actual, hop_distances(rowptr, col, sources, size, max_hops=2)
    )


def test_the_matrix_respects_edge_direction():
    """Message flow is directed; a transpose applied the wrong way round would
    silently measure the reverse graph and inflate reachability."""
    size = 3
    rowptr, col = csr(np.array([0, 1]), np.array([1, 2]), size)  # 0 -> 1 -> 2
    matrix = traversal_matrix(rowptr, col, size)
    forward = hop_distances(rowptr, col, np.array([0]), size, max_hops=2, matrix=matrix)
    assert forward[2] == 2
    backward = hop_distances(rowptr, col, np.array([2]), size, max_hops=2, matrix=matrix)
    assert backward[0] == UNREACHED


def test_no_sources_reaches_nothing():
    rowptr, col = csr(np.array([0]), np.array([1]), 2)
    matrix = traversal_matrix(rowptr, col, 2)
    distance = hop_distances(
        rowptr, col, np.array([], dtype=np.int64), 2, max_hops=3, matrix=matrix
    )
    assert (distance == UNREACHED).all()


# --------------------------------------------------------------------------
# expansion_sizes: one accumulating pass, same sizes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(25))
def test_incremental_expansion_matches_the_per_hop_walk(trial):
    rng = np.random.default_rng(1000 + trial)
    size = int(rng.integers(30, 400))
    rowptr, col = random_symmetric(rng, size, int(rng.integers(0, 4 * size)))
    pool = np.unique(rng.integers(0, size, size=int(rng.integers(1, min(size, 60)))))
    seeds = np.unique(rng.integers(0, size, size=int(rng.integers(1, 6))))
    for hops in HOPS:
        assert same_summary(
            expansion_sizes(rowptr, col, pool, seeds, max_hops=hops),
            reference_expansion(rowptr, col, pool, seeds, max_hops=hops),
        )


def test_expansion_is_monotonic_in_the_hop_budget():
    """Accumulating across hops would be wrong if a later hop ever shrank the
    set; this pins the property the rewrite relies on."""
    rng = np.random.default_rng(3)
    rowptr, col = random_symmetric(rng, 200, 400)
    pool = np.unique(rng.integers(0, 200, size=20))
    seeds = np.unique(rng.integers(0, 200, size=3))
    summary = expansion_sizes(rowptr, col, pool, seeds, max_hops=3)
    for tag in ("U_seed", "U_target"):
        sizes = [summary[f"{tag}_{hop}_nodes"] for hop in (1, 2, 3)]
        assert sizes == sorted(sizes)


def test_an_isolated_pool_never_expands():
    rowptr, col = csr(np.array([], dtype=np.int64), np.array([], dtype=np.int64), 10)
    pool = np.array([0, 1, 2])
    summary = expansion_sizes(rowptr, col, pool, np.array([0]), max_hops=3)
    for hop in (1, 2, 3):
        assert summary[f"U_target_{hop}_nodes"] == 3.0
        assert summary[f"U_target_{hop}_expansion"] == 1.0
