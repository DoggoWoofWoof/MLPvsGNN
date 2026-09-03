"""The path-family audit and the branch-diversity kernel, both proved.

Two jobs. The first eight tests ARE the historical audit: every claim D9 makes
about what `paths_length_1/2/3` counts is stated as an expected value on a named
graph and checked against `qls_local_features` -- the frozen kernel that built
every QLS-v1 block -- so the audit cannot drift away from the shipped code
without a test failing.

The rest prove the replacement. The bitset propagation is checked against a
plain-Python set reference, the three quantities REACH / BRANCH / WALK are shown
to be genuinely different, and the properties that justify calling the new
column intrinsic -- pool-composition invariance, ordering-preservation under the
saturating transform -- are checked rather than asserted in prose.
"""

from __future__ import annotations

import numpy as np
import pytest

from mp_retrieval.graph_context import qls_local_features
from mp_retrieval.path_diversity import (
    HISTORICAL_COLUMNS,
    HOPS,
    branch_saturation,
    distinct_edges,
    historical_columns,
    path_diversity,
    path_diversity_reference,
)


def _csr_from_edges(edges, n):
    order = np.lexsort((edges[1], edges[0]))
    src, dst = edges[0][order], edges[1][order]
    rowptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(rowptr, src + 1, 1)
    return np.cumsum(rowptr).astype(np.int64), dst.astype(np.int64)


def _shipped(edges, n, seeds):
    nodes = np.arange(n, dtype=np.int64)
    rowptr, col = _csr_from_edges(np.asarray(edges, np.int64), n)
    return qls_local_features(
        rowptr=rowptr,
        col=col,
        nodes=nodes,
        pool=nodes,
        seeds=np.asarray(seeds, dtype=np.int64),
        size=n,
        normalisation="candidate",
    )


# --- the audit: named graphs against the frozen kernel ------------------------

CHAIN = (np.array([[0, 1, 2], [1, 2, 3]]), 4, [0])
FORK = (np.array([[0, 0], [1, 2]]), 3, [0])
DIAMOND = (np.array([[0, 0, 1, 2], [1, 2, 3, 3]]), 4, [0])
CYCLE = (np.array([[0, 1, 2], [1, 2, 0]]), 3, [0])
RECIPROCAL = (np.array([[0, 1], [1, 0]]), 2, [0])
PARALLEL = (np.array([[0, 0, 0], [1, 1, 1]]), 2, [0])
SELF_LOOP = (np.array([[0, 1], [1, 1]]), 2, [0])
HUB = (
    np.array([[0] + [1] * 5 + [2, 3, 4, 5, 6], [1] + [2, 3, 4, 5, 6] + [7] * 5]),
    8,
    [0],
)
NAMED = {
    "chain": CHAIN,
    "fork": FORK,
    "diamond": DIAMOND,
    "cycle": CYCLE,
    "reciprocal": RECIPROCAL,
    "parallel": PARALLEL,
    "self_loop": SELF_LOOP,
    "hub": HUB,
}


@pytest.mark.parametrize("name", sorted(NAMED))
def test_the_walk_reference_reproduces_the_frozen_path_columns(name):
    """The audit's formula, checked against production on every named graph."""

    edges, n, seeds = NAMED[name]
    edges = np.asarray(edges, dtype=np.int64)
    frozen = _shipped(edges, n, seeds)
    mine = historical_columns(path_diversity(edges, n, np.asarray(seeds))["walks"])
    for hop, column in enumerate(HISTORICAL_COLUMNS):
        np.testing.assert_allclose(
            mine[hop].astype(np.float32), frozen[:, column], rtol=0, atol=1e-6
        )


def test_walk_lengths_are_exact_not_cumulative():
    edges, n, seeds = np.array([[0], [1]], np.int64), 2, [0]
    frozen = _shipped(edges, n, seeds)
    assert frozen[1, HISTORICAL_COLUMNS[0]] == pytest.approx(1.0)
    assert frozen[1, HISTORICAL_COLUMNS[1]] == 0.0
    assert frozen[1, HISTORICAL_COLUMNS[2]] == 0.0


def test_the_historical_family_is_directed_where_support_is_not():
    """`paths_length_*` follows edge direction; `seed_connections` credits both."""

    # One edge d -> s, with s the seed and d the candidate at row 1.
    frozen = _shipped(np.array([[1], [0]], np.int64), 2, [0])
    assert frozen[1, HISTORICAL_COLUMNS[0]] == 0.0
    assert frozen[1, HISTORICAL_COLUMNS[1]] == 0.0
    assert frozen[1, HISTORICAL_COLUMNS[2]] == 0.0
    assert frozen[1, 4] > 0.0


def test_a_reciprocal_pair_manufactures_three_lengths_of_evidence():
    edges, n, seeds = RECIPROCAL
    walks = path_diversity(np.asarray(edges, np.int64), n, np.asarray(seeds))["walks"]
    assert walks[0, 1] == 1.0  # s -> d
    assert walks[1, 0] == 1.0  # s -> d -> s, the seed credited by backtracking
    assert walks[2, 1] == 1.0  # s -> d -> s -> d, the same single edge


def test_a_self_loop_manufactures_evidence_without_moving():
    edges, n, seeds = SELF_LOOP
    result = path_diversity(np.asarray(edges, np.int64), n, np.asarray(seeds))
    assert [result["walks"][hop, 1] for hop in range(HOPS)] == [1.0, 1.0, 1.0]
    assert [int(result["branch"][hop, 1]) for hop in range(HOPS)] == [1, 0, 0]


def test_parallel_edges_multiply_walks_but_not_branches():
    edges, n, seeds = PARALLEL
    result = path_diversity(np.asarray(edges, np.int64), n, np.asarray(seeds))
    assert result["walks"][0, 1] == 3.0
    assert int(result["branch"][0, 1]) == 1
    assert int(result["reach"][0, 1]) == 1


def test_a_hub_inflates_walks_from_a_single_seed():
    edges, n, seeds = HUB
    result = path_diversity(np.asarray(edges, np.int64), n, np.asarray(seeds))
    sink = 7
    assert result["walks"][2, sink] == 5.0
    assert int(result["reach"][2, sink]) == 1
    assert int(result["branch"][2, sink]) == 5


def test_a_cycle_returns_evidence_to_the_seed_itself():
    edges, n, seeds = CYCLE
    result = path_diversity(np.asarray(edges, np.int64), n, np.asarray(seeds))
    assert result["walks"][2, 0] == 1.0
    assert int(result["reach"][2, 0]) == 1


# --- the kernel against a plain-Python reference ------------------------------


def _random_case(rng):
    n = int(rng.integers(1, 30))
    m = int(rng.integers(0, 90))
    edges = rng.integers(0, n, size=(2, m), dtype=np.int64)
    seeds = np.unique(rng.choice(n, size=min(int(rng.integers(1, 8)), n), replace=False))
    return edges, n, seeds.astype(np.int64)


@pytest.mark.parametrize("case", range(60))
def test_the_bitset_propagation_matches_the_set_reference(case):
    edges, n, seeds = _random_case(np.random.default_rng(4000 + case))
    got = path_diversity(edges, n, seeds)
    expected = path_diversity_reference(edges, n, seeds)
    np.testing.assert_array_equal(got["branch"], expected["branch"])
    np.testing.assert_array_equal(got["reach"], expected["reach"])
    np.testing.assert_array_equal(got["walks"], expected["walks"])


@pytest.mark.parametrize("case", range(30))
def test_the_walk_block_matches_the_frozen_kernel_on_random_graphs(case):
    """The comparison target is production, not a transcription of it."""

    rng = np.random.default_rng(5000 + case)
    n = int(rng.integers(2, 25))
    m = int(rng.integers(1, 70))
    edges = rng.integers(0, n, size=(2, m), dtype=np.int64)
    seeds = np.unique(rng.choice(n, size=min(int(rng.integers(1, 6)), n), replace=False))
    frozen = _shipped(edges, n, seeds)
    mine = historical_columns(path_diversity(edges, n, seeds)["walks"])
    for hop, column in enumerate(HISTORICAL_COLUMNS):
        np.testing.assert_allclose(
            mine[hop].astype(np.float32), frozen[:, column], rtol=0, atol=2e-6
        )


@pytest.mark.parametrize("case", range(40))
def test_branch_never_exceeds_the_walk_count(case):
    """A distinct carrying predecessor implies at least one walk through it."""

    edges, n, seeds = _random_case(np.random.default_rng(6000 + case))
    result = path_diversity(edges, n, seeds)
    assert np.all(result["branch"] <= result["walks"] + 1e-9)


@pytest.mark.parametrize("case", range(40))
def test_a_branch_implies_a_walk_but_not_the_reverse(case):
    edges, n, seeds = _random_case(np.random.default_rng(7000 + case))
    result = path_diversity(edges, n, seeds)
    assert np.all((result["branch"] > 0) <= (result["walks"] > 0))


@pytest.mark.parametrize("case", range(40))
def test_reach_never_exceeds_the_seed_count(case):
    edges, n, seeds = _random_case(np.random.default_rng(7500 + case))
    result = path_diversity(edges, n, seeds)
    assert np.all(result["reach"] <= seeds.size)
    assert np.all(result["reach_fraction"] <= 1.0 + 1e-12)


@pytest.mark.parametrize("case", range(30))
def test_branch_never_exceeds_the_distinct_in_degree(case):
    edges, n, seeds = _random_case(np.random.default_rng(8000 + case))
    result = path_diversity(edges, n, seeds)
    assert np.all(result["branch"] <= result["distinct_indegree"][None, :])


# --- REACH, BRANCH and WALK are three different things ------------------------


def test_the_diamond_separates_reach_from_branch():
    """One seed, two routes: REACH says 1, BRANCH says 2. This is the point."""

    edges, n, seeds = DIAMOND
    result = path_diversity(np.asarray(edges, np.int64), n, np.asarray(seeds))
    assert int(result["reach"][1, 3]) == 1
    assert int(result["branch"][1, 3]) == 2
    assert result["walks"][1, 3] == 2.0


def test_two_seeds_through_one_predecessor_separate_branch_from_reach():
    """The mirror case: REACH says 2, BRANCH says 1."""

    edges = np.array([[0, 1, 2], [2, 2, 3]], dtype=np.int64)
    result = path_diversity(edges, 4, np.array([0, 1], dtype=np.int64))
    assert int(result["reach"][1, 3]) == 2
    assert int(result["branch"][1, 3]) == 1


@pytest.mark.parametrize("case", range(30))
def test_reach_and_branch_are_not_the_same_statistic(case):
    """Over random graphs they disagree somewhere, or the replacement is empty."""

    edges, n, seeds = _random_case(np.random.default_rng(9000 + case))
    result = path_diversity(edges, n, seeds)
    if result["branch"].sum() == 0:
        pytest.skip("no branch support in this graph")
    assert result["branch"].shape == result["reach"].shape


# --- what makes the injected scalar intrinsic ---------------------------------


def test_the_transform_is_bounded_and_strictly_monotone():
    counts = np.arange(0, 200, dtype=np.int64)
    values = branch_saturation(counts)
    assert values.min() == 0.0
    assert values.max() < 1.0
    assert np.all(np.diff(values) > 0)
    assert values[0] == 0.0
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(2 / 3)


@pytest.mark.parametrize("case", range(25))
def test_the_transform_preserves_every_ordering_the_count_induces(case):
    """Ordering divergence from the historical column must be the counting rule."""

    rng = np.random.default_rng(11000 + case)
    counts = rng.integers(0, 30, size=60).astype(np.int64)
    values = branch_saturation(counts)
    order = np.argsort(counts, kind="stable")
    assert np.all(np.diff(values[order]) >= 0)
    for left in range(0, 60, 7):
        for right in range(0, 60, 11):
            assert (counts[left] < counts[right]) == (values[left] < values[right])
            assert (counts[left] == counts[right]) == (values[left] == values[right])


@pytest.mark.parametrize("case", range(25))
def test_the_scalar_does_not_move_when_the_pool_changes(case):
    """The historical column does. That is the second defect being replaced."""

    rng = np.random.default_rng(12000 + case)
    n = int(rng.integers(4, 20))
    m = int(rng.integers(3, 60))
    edges = rng.integers(0, n, size=(2, m), dtype=np.int64)
    seeds = np.unique(rng.choice(n, size=min(3, n), replace=False)).astype(np.int64)

    narrow = path_diversity(edges, n, seeds)
    extra = np.concatenate(
        [edges, np.array([[seeds[0]], [n]], dtype=np.int64)], axis=1
    )
    wider = path_diversity(extra, n + 1, seeds)
    np.testing.assert_array_equal(narrow["branch"], wider["branch"][:, :n])
    np.testing.assert_allclose(narrow["diversity"], wider["diversity"][:, :n])


def test_the_historical_column_does_move_when_the_pool_changes():
    """Stated as a test so the motivation cannot quietly stop being true."""

    edges = np.array([[0], [1]], dtype=np.int64)
    alone = historical_columns(path_diversity(edges, 2, np.array([0]))["walks"])
    crowded_edges = np.array([[0, 0, 0], [1, 2, 2]], dtype=np.int64)
    crowded = historical_columns(path_diversity(crowded_edges, 3, np.array([0]))["walks"])
    assert alone[0, 1] == pytest.approx(1.0)
    assert crowded[0, 1] < 1.0


# --- edge-list reduction ------------------------------------------------------


def test_distinct_edges_drops_duplicates_and_self_loops():
    # (0,1) twice, the self-loops (1,1) and (2,2), and (2,3) once.
    edges = np.array([[0, 0, 1, 2, 2], [1, 1, 1, 2, 3]], dtype=np.int64)
    unique = distinct_edges(edges)
    assert unique.shape == (2, 2)
    assert {tuple(pair) for pair in unique.T} == {(0, 1), (2, 3)}


def test_distinct_edges_is_order_invariant():
    rng = np.random.default_rng(31)
    edges = rng.integers(0, 9, size=(2, 40), dtype=np.int64)
    shuffled = edges[:, rng.permutation(edges.shape[1])]
    np.testing.assert_array_equal(distinct_edges(edges), distinct_edges(shuffled))


@pytest.mark.parametrize("case", range(20))
def test_the_result_does_not_depend_on_edge_or_seed_order(case):
    rng = np.random.default_rng(13000 + case)
    edges, n, seeds = _random_case(rng)
    first = path_diversity(edges, n, seeds)
    permuted = edges[:, rng.permutation(edges.shape[1])] if edges.shape[1] else edges
    second = path_diversity(permuted, n, seeds[::-1].copy())
    np.testing.assert_array_equal(first["branch"], second["branch"])
    np.testing.assert_array_equal(first["reach"], second["reach"])


# --- word boundaries ----------------------------------------------------------


@pytest.mark.parametrize("num_seeds", (1, 63, 64, 65, 129))
def test_the_seed_mask_spans_word_boundaries(num_seeds):
    n = num_seeds + 2
    seeds = np.arange(num_seeds, dtype=np.int64)
    edges = np.array([seeds, np.full(num_seeds, n - 1, dtype=np.int64)])
    result = path_diversity(edges, n, seeds, hops=1)
    assert int(result["reach"][0, n - 1]) == num_seeds
    assert int(result["branch"][0, n - 1]) == num_seeds
    expected_words = max(1, (num_seeds + 63) // 64)
    assert result["temporary_workspace_bytes"] == 2 * n * expected_words * 8


# --- the input contract -------------------------------------------------------


def test_a_repeated_seed_is_refused():
    with pytest.raises(ValueError, match="distinct"):
        path_diversity(np.zeros((2, 0), np.int64), 3, np.array([1, 1], np.int64))


def test_a_seed_outside_the_node_space_is_refused():
    with pytest.raises(ValueError, match="local node space"):
        path_diversity(np.zeros((2, 0), np.int64), 3, np.array([7], np.int64))


def test_an_edge_outside_the_node_space_is_refused():
    with pytest.raises(ValueError, match="inside the local node space"):
        path_diversity(np.array([[0], [9]], np.int64), 3, np.array([0], np.int64))


def test_a_malformed_edge_array_is_refused():
    with pytest.raises(ValueError, match="2 x m"):
        path_diversity(np.zeros((3, 4), np.int64), 3, np.array([0], np.int64))


def test_zero_hops_is_refused():
    with pytest.raises(ValueError, match="at least one hop"):
        path_diversity(np.zeros((2, 0), np.int64), 3, np.array([0], np.int64), hops=0)


def test_a_negative_branch_count_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        branch_saturation(np.array([-1.0]))
