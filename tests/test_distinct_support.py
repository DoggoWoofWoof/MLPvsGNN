"""D8 step 4: prove the bitset kernel against a brute-force reference.

The point of these tests is that the replacement column differs from
`seed_connections` in exactly one way -- what it counts -- and in no other way.
So they check both halves: the distinct count against an independent slow
implementation, and the historical count the kernel returns alongside it against
the *frozen shipped kernel itself*, not against a second reading of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mp_retrieval.distinct_support import (
    HISTORICAL_NAME,
    REPLACEMENT_NAME,
    WORD_BITS,
    comparison_summary,
    distinct_seed_support,
    distinct_seed_support_reference,
    historical_column,
    popcount_rows,
    support_masks,
)
from mp_retrieval.graph_context import qls_local_features

SEED_CONNECTIONS_COLUMN = 4


def _random_case(rng, *, n, m, num_seeds):
    edges = rng.integers(0, n, size=(2, m), dtype=np.int64)
    seeds = np.unique(rng.choice(n, size=min(num_seeds, n), replace=False))
    return edges, int(n), seeds.astype(np.int64)


def _csr_from_edges(edges, n):
    """A CSR the frozen `induced_edges` can walk, from the same edge list."""
    order = np.lexsort((edges[1], edges[0]))
    src = edges[0][order]
    dst = edges[1][order]
    rowptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(rowptr, src + 1, 1)
    rowptr = np.cumsum(rowptr)
    return rowptr.astype(np.int64), dst.astype(np.int64)


# --- the kernel against brute force -----------------------------------------


@pytest.mark.parametrize("case", range(60))
def test_bitset_matches_brute_force_reference(case):
    rng = np.random.default_rng(1000 + case)
    n = int(rng.integers(1, 40))
    m = int(rng.integers(0, 120))
    num_seeds = int(rng.integers(1, min(n, 9) + 1))
    edges, n, seeds = _random_case(rng, n=n, m=m, num_seeds=num_seeds)

    counts, fraction, connections, size = distinct_seed_support(edges, n, seeds)
    ref_counts, ref_fraction, ref_connections, ref_size = (
        distinct_seed_support_reference(edges, n, seeds)
    )

    assert size == ref_size
    np.testing.assert_array_equal(counts, ref_counts)
    np.testing.assert_array_equal(fraction, ref_fraction)
    np.testing.assert_array_equal(connections, ref_connections)


@pytest.mark.parametrize("case", range(20))
def test_distinct_count_never_exceeds_the_historical_count(case):
    """The replacement collapses multiplicity; it can never manufacture it."""
    rng = np.random.default_rng(2000 + case)
    edges, n, seeds = _random_case(
        rng, n=int(rng.integers(2, 30)), m=int(rng.integers(0, 100)), num_seeds=4
    )
    counts, _, connections, _ = distinct_seed_support(edges, n, seeds)
    assert np.all(counts <= connections)


@pytest.mark.parametrize("case", range(20))
def test_support_is_bounded_by_the_seed_count(case):
    rng = np.random.default_rng(3000 + case)
    edges, n, seeds = _random_case(
        rng, n=int(rng.integers(2, 30)), m=int(rng.integers(0, 200)), num_seeds=6
    )
    counts, fraction, _, size = distinct_seed_support(edges, n, seeds)
    assert counts.max(initial=0) <= size
    assert np.all(fraction >= 0.0)
    assert np.all(fraction <= 1.0)


@pytest.mark.parametrize("case", range(20))
def test_a_node_supports_exactly_the_seeds_its_mask_names(case):
    """popcount is not checked in isolation: the bits themselves are checked."""
    rng = np.random.default_rng(4000 + case)
    edges, n, seeds = _random_case(
        rng, n=int(rng.integers(2, 25)), m=int(rng.integers(0, 80)), num_seeds=5
    )
    masks, _, _, _ = support_masks(edges, n, seeds)
    expected = [set() for _ in range(n)]
    seed_set = {int(s) for s in seeds}
    for edge in range(edges.shape[1]):
        source, target = int(edges[0, edge]), int(edges[1, edge])
        if source in seed_set:
            expected[target].add(source)
        if target in seed_set:
            expected[source].add(target)
    for node in range(n):
        named = {
            int(seeds[bit])
            for bit in range(seeds.size)
            if masks[node, bit // WORD_BITS] >> np.uint64(bit % WORD_BITS) & np.uint64(1)
        }
        assert named == expected[node]


# --- the exact cases the replacement exists to separate ----------------------


def test_a_reciprocal_pair_is_one_seed_not_two():
    """The defect in one line: one seed pointing both ways scores 2 historically."""
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    counts, fraction, connections, size = distinct_seed_support(edges, 2, [0])
    assert connections[1] == 2.0
    assert counts[1] == 1
    assert size == 1
    assert fraction[1] == 1.0


def test_parallel_edges_from_one_seed_are_one_seed():
    edges = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    counts, _, connections, _ = distinct_seed_support(edges, 2, [0])
    assert connections[1] == 3.0
    assert counts[1] == 1


def test_two_distinct_seeds_are_two():
    """The case the historical column cannot tell from the reciprocal pair."""
    edges = np.array([[0, 1], [2, 2]], dtype=np.int64)
    counts, fraction, connections, size = distinct_seed_support(edges, 3, [0, 1])
    assert connections[2] == 2.0
    assert counts[2] == 2
    assert size == 2
    assert fraction[2] == 1.0


def test_a_self_loop_on_a_seed_scores_two_historically_and_one_distinctly():
    """Both historical branches fire on a seed self-loop. The bitset sets one bit."""
    edges = np.array([[0], [0]], dtype=np.int64)
    counts, _, connections, _ = distinct_seed_support(edges, 1, [0])
    assert connections[0] == 2.0
    assert counts[0] == 1


def test_a_seed_adjacent_to_a_seed_accrues_support_like_the_historical_column():
    """The replacement keeps the historical privilege rule, it does not fix it."""
    edges = np.array([[0], [1]], dtype=np.int64)
    counts, _, connections, _ = distinct_seed_support(edges, 2, [0, 1])
    assert connections[0] == 1.0 and connections[1] == 1.0
    assert counts[0] == 1 and counts[1] == 1


def test_no_edges_gives_no_support():
    edges = np.zeros((2, 0), dtype=np.int64)
    counts, fraction, connections, size = distinct_seed_support(edges, 5, [0, 1])
    assert counts.tolist() == [0] * 5
    assert fraction.tolist() == [0.0] * 5
    assert connections.tolist() == [0.0] * 5
    assert size == 2


def test_a_query_with_no_seeds_yields_a_zero_fraction_not_a_division_by_zero():
    edges = np.array([[0], [1]], dtype=np.int64)
    counts, fraction, connections, size = distinct_seed_support(edges, 2, [])
    assert size == 0
    assert counts.tolist() == [0, 0]
    assert fraction.tolist() == [0.0, 0.0]
    assert connections.tolist() == [0.0, 0.0]


def test_duplicate_seed_positions_are_one_seed():
    counts, fraction, _, size = distinct_seed_support(
        np.array([[0], [1]], dtype=np.int64), 2, [0, 0, 0]
    )
    assert size == 1
    assert counts[1] == 1
    assert fraction[1] == 1.0


# --- the property that makes it a representation fix, not just a recount -----


def test_the_fraction_of_one_candidate_does_not_depend_on_the_other_candidates():
    """The second defect: the historical column is scaled by the pool's best.

    Adding a better-supported candidate to the pool lowers the historical value
    of a candidate whose own topology did not change. The fraction does not move.
    """
    lonely = np.array([[0], [1]], dtype=np.int64)
    counts_a, fraction_a, connections_a, _ = distinct_seed_support(lonely, 2, [0])
    crowded = np.array([[0, 0, 0, 0], [1, 2, 2, 2]], dtype=np.int64)
    counts_b, fraction_b, connections_b, _ = distinct_seed_support(crowded, 3, [0])

    assert counts_a[1] == counts_b[1] == 1
    assert fraction_a[1] == fraction_b[1] == 1.0
    assert historical_column(connections_a)[1] == 1.0
    assert historical_column(connections_b)[1] < 1.0


@pytest.mark.parametrize("case", range(15))
def test_the_fraction_is_invariant_to_seed_ordering(case):
    rng = np.random.default_rng(5000 + case)
    edges, n, seeds = _random_case(
        rng, n=int(rng.integers(2, 25)), m=int(rng.integers(0, 60)), num_seeds=5
    )
    shuffled = seeds.copy()
    rng.shuffle(shuffled)
    left = distinct_seed_support(edges, n, seeds)
    right = distinct_seed_support(edges, n, shuffled)
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])


@pytest.mark.parametrize("case", range(15))
def test_the_count_is_invariant_to_edge_ordering(case):
    rng = np.random.default_rng(6000 + case)
    edges, n, seeds = _random_case(
        rng, n=int(rng.integers(2, 25)), m=int(rng.integers(1, 60)), num_seeds=4
    )
    permuted = edges[:, rng.permutation(edges.shape[1])]
    np.testing.assert_array_equal(
        distinct_seed_support(edges, n, seeds)[0],
        distinct_seed_support(permuted, n, seeds)[0],
    )


# --- bitset mechanics --------------------------------------------------------


def test_one_word_holds_up_to_sixty_four_seeds():
    masks, _, _, size = support_masks(
        np.zeros((2, 0), dtype=np.int64), 70, np.arange(64)
    )
    assert size == 64
    assert masks.shape == (70, 1)


def test_more_than_sixty_four_seeds_spill_into_a_second_word():
    n = 200
    seeds = np.arange(65)
    edges = np.stack([seeds, np.full(65, 199, dtype=np.int64)])
    masks, _, _, size = support_masks(edges, n, seeds)
    assert size == 65
    assert masks.shape == (n, 2)
    counts, fraction, _, _ = distinct_seed_support(edges, n, seeds)
    assert counts[199] == 65
    assert fraction[199] == 1.0


def test_the_word_count_is_the_ceiling_of_seeds_over_sixty_four():
    for num_seeds, words in ((0, 1), (1, 1), (64, 1), (65, 2), (128, 2), (129, 3)):
        masks, *_ = support_masks(
            np.zeros((2, 0), dtype=np.int64), 200, np.arange(num_seeds)
        )
        assert masks.shape == (200, words), num_seeds


@pytest.mark.parametrize("case", range(15))
def test_the_high_bit_of_a_word_survives(case):
    """A shift built with the wrong dtype loses bit 63. Check it explicitly."""
    rng = np.random.default_rng(7000 + case)
    n = 200
    seeds = np.arange(int(rng.integers(64, 130)))
    target = 199
    edges = np.stack([seeds, np.full(seeds.size, target, dtype=np.int64)])
    counts, _, _, size = distinct_seed_support(edges, n, seeds)
    assert counts[target] == size == seeds.size


@pytest.mark.parametrize("case", range(30))
def test_popcount_of_the_masks_equals_the_accumulated_count(case):
    """The two routes to the same number, kept independent and checked.

    The pass counts incrementally, so it never scans the masks. `popcount_rows`
    scans them and counts nothing else. Their agreeing is the evidence that the
    bits and the counter describe the same support set.
    """
    rng = np.random.default_rng(9000 + case)
    edges, n, seeds = _random_case(
        rng, n=int(rng.integers(1, 35)), m=int(rng.integers(0, 150)), num_seeds=7
    )
    masks, counts, _, num_seeds = support_masks(edges, n, seeds)
    np.testing.assert_array_equal(popcount_rows(masks, num_seeds), counts)


def test_popcount_agrees_across_a_word_boundary():
    n = 200
    seeds = np.arange(129)
    edges = np.stack([seeds, np.full(seeds.size, 199, dtype=np.int64)])
    masks, counts, _, num_seeds = support_masks(edges, n, seeds)
    assert masks.shape[1] == 3
    assert counts[199] == 129
    np.testing.assert_array_equal(popcount_rows(masks, num_seeds), counts)


def test_popcount_reads_no_bit_beyond_the_seeds_that_exist():
    """Bits above `num_seeds` cannot be set, so the bounded scan loses nothing."""
    masks = np.zeros((1, 1), dtype=np.uint64)
    masks[0, 0] = np.uint64(0b1111)
    assert popcount_rows(masks, 2).tolist() == [2]
    assert popcount_rows(masks, 4).tolist() == [4]


# --- against the frozen shipped kernel ---------------------------------------


@pytest.mark.parametrize("case", range(25))
def test_the_historical_count_reproduces_the_frozen_column(case):
    """The accumulator is the historical one, proved against the shipped kernel.

    `qls_local_features` calls `_local_feature_chunk` -- the frozen kernel that
    built every QLS-v1 feature block -- so this compares against production, not
    against a transcription of it.
    """
    rng = np.random.default_rng(8000 + case)
    n = int(rng.integers(2, 30))
    m = int(rng.integers(1, 90))
    edges = rng.integers(0, n, size=(2, m), dtype=np.int64)
    seeds = np.unique(
        rng.choice(n, size=int(rng.integers(1, min(n, 7) + 1)), replace=False)
    )
    nodes = np.arange(n, dtype=np.int64)
    rowptr, col = _csr_from_edges(edges, n)

    frozen = qls_local_features(
        rowptr=rowptr,
        col=col,
        nodes=nodes,
        pool=nodes,
        seeds=seeds,
        size=n,
        normalisation="candidate",
    )
    _, _, connections, _ = distinct_seed_support(edges, n, seeds)
    np.testing.assert_allclose(
        historical_column(connections).astype(np.float32),
        frozen[:, SEED_CONNECTIONS_COLUMN],
        rtol=0,
        atol=1e-6,
    )


def test_the_frozen_column_is_reproduced_even_when_it_is_empty():
    """The kernel leaves an all-zero column alone rather than dividing by zero."""
    n = 4
    nodes = np.arange(n, dtype=np.int64)
    edges = np.array([[1, 2], [2, 3]], dtype=np.int64)
    rowptr, col = _csr_from_edges(edges, n)
    frozen = qls_local_features(
        rowptr=rowptr,
        col=col,
        nodes=nodes,
        pool=nodes,
        seeds=np.array([0]),
        size=n,
        normalisation="candidate",
    )
    _, _, connections, _ = distinct_seed_support(edges, n, np.array([0]))
    assert connections.sum() == 0.0
    assert frozen[:, SEED_CONNECTIONS_COLUMN].sum() == 0.0
    np.testing.assert_array_equal(historical_column(connections), np.zeros(n))


def test_historical_column_applies_log1p_then_the_maximum_in_that_order():
    connections = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    column = historical_column(connections)
    np.testing.assert_allclose(column, np.log1p(connections) / np.log1p(3.0))
    assert column[2] == 1.0
    # log1p first is not cosmetic: it is why 3 edges is not worth 3x 1 edge.
    assert column[1] > connections[1] / connections[2]


# --- input contracts ---------------------------------------------------------


def test_a_malformed_edge_array_is_refused():
    with pytest.raises(ValueError, match="2, m"):
        support_masks(np.zeros((3, 4), dtype=np.int64), 5, [0])


def test_an_out_of_range_edge_endpoint_is_refused():
    with pytest.raises(ValueError, match="edge endpoint"):
        support_masks(np.array([[0], [9]], dtype=np.int64), 5, [0])


def test_an_out_of_range_seed_position_is_refused():
    with pytest.raises(ValueError, match="seed position"):
        support_masks(np.zeros((2, 0), dtype=np.int64), 5, [7])


def test_a_negative_seed_position_is_refused():
    with pytest.raises(ValueError, match="seed position"):
        support_masks(np.zeros((2, 0), dtype=np.int64), 5, [-1])


# --- the mechanistic comparison ----------------------------------------------


def test_comparison_summary_counts_the_rows_where_the_two_rules_disagree():
    edges = np.array([[0, 1, 0], [1, 0, 2]], dtype=np.int64)
    counts, _, connections, _ = distinct_seed_support(edges, 3, [0])
    summary = comparison_summary(counts, connections)
    assert summary["rows"] == 3
    assert summary["rows_with_any_support"] == 2
    assert summary["rows_where_edges_exceed_distinct_seeds"] == 1
    assert summary["max_distinct_support"] == 1
    assert summary["max_historical_connections"] == 2.0


def test_comparison_summary_buckets_by_induced_degree():
    counts = np.array([0, 1, 2, 3], dtype=np.int64)
    connections = np.array([0.0, 1.0, 4.0, 3.0], dtype=np.float64)
    degrees = np.array([0, 1, 3, 9], dtype=np.int64)
    buckets = comparison_summary(counts, connections, degrees)["by_induced_degree"]
    names = ("degree_0", "degree_1", "degree_2_to_4", "degree_5_plus")
    assert [buckets[name]["rows"] for name in names] == [1, 1, 1, 1]
    assert buckets["degree_2_to_4"]["rows_where_edges_exceed_distinct_seeds"] == 1
    assert buckets["degree_5_plus"]["rows_where_edges_exceed_distinct_seeds"] == 0
    assert buckets["degree_5_plus"]["mean_distinct_support"] == 3.0


def test_comparison_summary_reports_an_empty_bucket_without_dividing_by_zero():
    summary = comparison_summary(
        np.array([1], dtype=np.int64),
        np.array([1.0]),
        np.array([0], dtype=np.int64),
    )
    empty = summary["by_induced_degree"]["degree_5_plus"]
    assert empty["rows"] == 0
    assert empty["mean_distinct_support"] == 0.0
    assert empty["mean_historical_connections"] == 0.0


def test_comparison_summary_handles_an_empty_node_space():
    summary = comparison_summary(
        np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0, dtype=np.int64)
    )
    assert summary["rows"] == 0
    assert summary["max_distinct_support"] == 0
    assert summary["max_historical_connections"] == 0.0


# --- naming ------------------------------------------------------------------


def test_the_two_statistics_are_named_apart():
    assert HISTORICAL_NAME == "seed_connections"
    assert REPLACEMENT_NAME == "distinct_seed_support"
    assert HISTORICAL_NAME != REPLACEMENT_NAME
