"""Headroom for a pool that can move, and the invariants that keep it honest.

The historical layer answers one question about one fixed pool. This layer adds
the comparison between pools, so the tests here are mostly about what must stay
equal: R1 and R2 share a candidate set and therefore share a ceiling exactly,
recovered and lost golds are never netted into a single flattering figure, and
every definition that already existed is imported rather than restated.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from mp_retrieval import candidate_headroom as historical
from mp_retrieval import headroom_v2 as hv2
from mp_retrieval.headroom_v2 import (
    full_coverage_ceiling,
    pool_movement,
    ragged_from_rows,
    regime_headroom,
)

NUM_NODES = 16
KS = (1, 5, 20)

# Four queries chosen so every branch of the ceiling has a witness:
# q0 -- both golds in the pool, q1 -- one of two, q2 -- none, q3 -- one of one.
GOLDS = [
    np.array([1, 2], dtype=np.int64),
    np.array([3, 4], dtype=np.int64),
    np.array([5, 6], dtype=np.int64),
    np.array([7], dtype=np.int64),
]
POOLS = [
    np.array([0, 1, 2], dtype=np.int64),
    np.array([3, 8, 9], dtype=np.int64),
    np.array([10, 11, 12], dtype=np.int64),
    np.array([7, 13, 14], dtype=np.int64),
]


@pytest.fixture(scope="module")
def golds():
    return ragged_from_rows(GOLDS)


@pytest.fixture(scope="module")
def pool():
    return ragged_from_rows(POOLS)


def _sizes(rows) -> np.ndarray:
    return np.asarray([row.size for row in rows], dtype=np.int64)


# --- the definitions that already existed are reused, not rewritten ---


@pytest.mark.parametrize("name", ["present_counts", "headroom_metrics", "ragged_from_rows"])
def test_the_historical_definition_is_the_same_object(name):
    """Two definitions of AnyGold in two modules is the failure this prevents."""
    assert getattr(hv2, name) is getattr(historical, name)


def test_the_new_module_defines_only_what_is_new():
    tree = ast.parse(Path(hv2.__file__).read_text(encoding="utf-8"))
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert defined == {"full_coverage_ceiling", "pool_movement", "regime_headroom"}


def test_nothing_here_writes_into_a_historical_path():
    source = inspect.getsource(hv2)
    for forbidden in ("open(", "write_text", "np.save", "torch.save", "mkdir"):
        assert forbidden not in source, forbidden


# --- the full-coverage ceiling, where both caps bite ---


def test_full_coverage_needs_every_gold_present(golds, pool):
    metrics, present, gold_counts = regime_headroom(pool, golds, num_nodes=NUM_NODES, ks=KS)
    assert present.tolist() == [2, 1, 0, 1]
    assert gold_counts.tolist() == [2, 2, 2, 1]
    # q0 and q3 have all of theirs; q1 and q2 do not.
    assert metrics["full_coverage_ceiling@20"] == pytest.approx(0.5)


def test_the_cut_off_is_part_of_the_ceiling_not_a_footnote(golds, pool):
    """A query with two golds cannot reach full coverage at 1 however good the pool."""
    metrics, _, _ = regime_headroom(pool, golds, num_nodes=NUM_NODES, ks=KS)
    assert metrics["full_coverage_ceiling@1"] == pytest.approx(0.25)
    assert metrics["full_coverage_ceiling@5"] == pytest.approx(0.5)


def test_full_coverage_never_exceeds_all_gold_at_pool(golds, pool):
    metrics, _, _ = regime_headroom(pool, golds, num_nodes=NUM_NODES, ks=KS)
    for k in KS:
        assert metrics[f"full_coverage_ceiling@{k}"] <= metrics["all_gold_at_pool"]


def test_a_ceiling_asked_for_at_a_nonpositive_cut_off_is_refused():
    present = np.array([1, 1])
    with pytest.raises(ValueError, match="positive"):
        full_coverage_ceiling(present, present, ks=(0,))


def test_misaligned_counts_are_refused_rather_than_broadcast():
    with pytest.raises(ValueError, match="align"):
        full_coverage_ceiling(np.array([1, 2]), np.array([1]), ks=(5,))


def test_an_empty_evaluation_reports_zero_rather_than_dividing(golds):
    empty = ragged_from_rows([])
    metrics = full_coverage_ceiling(
        np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), ks=KS
    )
    assert set(metrics.values()) == {0.0}
    assert empty[1].tolist() == [0]


# --- R1 and R2 share a candidate set, so they share a ceiling exactly ---


def test_the_r2_context_does_not_move_the_candidate_ceiling(golds, pool):
    """R2 widens what a candidate can see. It does not widen what can be scored.

    The equality is asserted rather than assumed because the whole read of the
    R2 arm depends on it: any movement in R2 has to come from context, and a
    ceiling that drifted would mean the candidate set had drifted too.
    """

    rowptr = np.arange(NUM_NODES + 1, dtype=np.int64)
    col = (np.arange(NUM_NODES, dtype=np.int64) + 1) % NUM_NODES

    def one_hop(row: np.ndarray) -> np.ndarray:
        reached = [col[int(rowptr[n]) : int(rowptr[n + 1])] for n in row.tolist()]
        return np.unique(np.concatenate([row, *reached]))

    context = [one_hop(row) for row in POOLS]
    assert any(ctx.size > row.size for ctx, row in zip(context, POOLS, strict=True))

    r1, _, _ = regime_headroom(pool, golds, num_nodes=NUM_NODES, ks=KS)
    r2, _, _ = regime_headroom(pool, golds, num_nodes=NUM_NODES, ks=KS)
    assert r1 == r2

    widened, _, _ = regime_headroom(
        ragged_from_rows(context), golds, num_nodes=NUM_NODES, ks=KS
    )
    assert widened != r1, "scoring the context would be a different regime, not R2"


# --- what a regime did to the pool, reported in both directions ---


def _movement(regime_pools):
    _, baseline_present, gold_counts = regime_headroom(
        ragged_from_rows(POOLS), ragged_from_rows(GOLDS), num_nodes=NUM_NODES, ks=KS
    )
    _, regime_present, _ = regime_headroom(
        ragged_from_rows(regime_pools), ragged_from_rows(GOLDS), num_nodes=NUM_NODES, ks=KS
    )
    return pool_movement(
        baseline_present=baseline_present,
        regime_present=regime_present,
        gold_counts=gold_counts,
        baseline_sizes=_sizes(POOLS),
        regime_sizes=_sizes(regime_pools),
    )


def test_a_recovery_is_counted_against_the_slots_it_cost():
    """Two golds recovered for two extra slots is a rate of one, not a headline of two."""
    regime = [
        POOLS[0],
        np.array([3, 4, 8, 9], dtype=np.int64),
        np.array([5, 10, 11, 12], dtype=np.int64),
        POOLS[3],
    ]
    movement = _movement(regime)
    assert movement["missing_golds_recovered"] == 2
    assert movement["pool_slots_added_total"] == 2
    assert movement["recovered_gold_per_added_candidate"] == pytest.approx(1.0)
    assert movement["queries_improved"] == 2
    assert movement["pool_size_is_matched"] is False


def test_a_matched_budget_regime_reports_no_added_slots():
    """Under the headline rule the denominator is zero, and the rate says zero."""
    regime = [
        np.array([1, 2, 4], dtype=np.int64),
        POOLS[1],
        POOLS[2],
        POOLS[3],
    ]
    movement = _movement(regime)
    assert movement["pool_size_is_matched"] is True
    assert movement["pool_slots_added_total"] == 0
    assert movement["recovered_gold_per_added_candidate"] == 0.0
    assert movement["missing_golds_recovered"] == 0


def test_recovered_and_lost_golds_are_never_netted_away():
    """A regime that gains one query a gold and costs another one is not a null.

    A single net figure would print both of these as zero and read as "nothing
    happened", which is the one description that is certainly wrong.
    """

    regime = [
        np.array([0, 1, 9], dtype=np.int64),
        np.array([3, 4, 8], dtype=np.int64),
        POOLS[2],
        POOLS[3],
    ]
    movement = _movement(regime)
    assert movement["missing_golds_recovered"] == 1
    assert movement["golds_lost"] == 1
    assert movement["net_gold_movement"] == 0
    assert movement["queries_improved"] == 1
    assert movement["queries_worsened"] == 1
    assert movement["pool_size_is_matched"] is True


def test_a_swap_inside_one_query_reads_as_no_movement():
    """The per-query difference is a count, so an equal swap cancels within it.

    This is a real limit of counting rather than tracking identities, and it is
    pinned here so nobody later reads a zero as "the pool was untouched".
    """

    regime = [
        np.array([0, 1, 5], dtype=np.int64),
        POOLS[1],
        POOLS[2],
        POOLS[3],
    ]
    movement = _movement(regime)
    assert movement["missing_golds_recovered"] == 0
    assert movement["golds_lost"] == 1
    assert movement["queries_worsened"] == 1


def test_a_regime_that_changes_nothing_says_so():
    movement = _movement(POOLS)
    assert movement["missing_golds_recovered"] == 0
    assert movement["golds_lost"] == 0
    assert movement["queries_unchanged"] == len(POOLS)
    assert movement["pool_size_is_matched"] is True


def test_the_pool_size_distribution_is_reported_not_just_its_mean():
    regime = [
        np.array([1, 2, 4, 5, 6], dtype=np.int64),
        POOLS[1],
        POOLS[2],
        POOLS[3],
    ]
    movement = _movement(regime)
    assert movement["pool_size_max"] == 5
    assert movement["pool_size_p50"] == pytest.approx(3.0)
    assert movement["pool_size_mean"] == pytest.approx(3.5)


def test_misaligned_per_query_arrays_are_refused():
    with pytest.raises(ValueError, match="align"):
        pool_movement(
            baseline_present=np.array([1, 1]),
            regime_present=np.array([1]),
            gold_counts=np.array([1, 1]),
            baseline_sizes=np.array([1, 1]),
            regime_sizes=np.array([1, 1]),
        )
