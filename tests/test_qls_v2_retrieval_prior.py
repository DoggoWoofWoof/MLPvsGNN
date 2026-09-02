"""Group A's frozen formulas, and the one distinction it exists to restore.

The catalog's rationale for A4-A6 is that "the distinction between 'SPLADE ranked
it 190th' and 'SPLADE never retrieved it' is real and v1 could not express it".
On a 200-deep pool that understates the problem: at rank 200 exactly, A2 is
``1 - 200/200 = 0``, which is bit-for-bit the value A2 reports for a candidate
SPLADE never returned. The indicator columns are not redundant encoding, they are
the only thing separating two genuinely different states, and the test below
pins that they do.

Everything here is graph-free, which is why it is unblocked while Phase −1 runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from mp_retrieval.qls_v2_retrieval_prior import (
    MAX_RRF,
    POOL_DEPTH,
    RETRIEVAL_PRIOR_FEATURE_NAMES,
    MISSING_RANK,
    RRF_CONSTANT,
    rank_lookup,
    retrieval_prior_features,
)

NAMES = RETRIEVAL_PRIOR_FEATURE_NAMES


def _column(matrix: np.ndarray, name: str) -> np.ndarray:
    return matrix[:, NAMES.index(name)]


def _filler(count: int, *, avoid: int = -1) -> list[int]:
    """Unique padding ids that cannot collide with the candidate under test."""
    return [i for i in range(1000, 1000 + count + 1) if i != avoid][:count]


def _features(candidates, dense=(), splade=(), seeds=()):
    return retrieval_prior_features(
        candidates, dense_ranked=dense, splade_ranked=splade, seeds=seeds
    )


# --------------------------------------------------------------------------
# The formulas
# --------------------------------------------------------------------------


def test_rank_percentiles_follow_the_stated_formula():
    features = _features([10, 20, 30], dense=[10, 20, 30])
    assert _column(features, "dense_rank_pct") == pytest.approx(
        [1 - 1 / POOL_DEPTH, 1 - 2 / POOL_DEPTH, 1 - 3 / POOL_DEPTH]
    )


def test_rrf_sums_both_reciprocals_and_tops_out_at_the_stated_maximum():
    features = _features([7], dense=[7], splade=[7])
    assert _column(features, "rrf")[0] == pytest.approx(2.0 / (RRF_CONSTANT + 1))
    assert MAX_RRF == pytest.approx(2.0 / (RRF_CONSTANT + 1))


def test_a_missing_retriever_contributes_nothing_to_rrf_rather_than_one_over_k():
    """Absence must contribute zero, and the tempting alternative is worse.

    The natural-looking shortcut is to leave the sentinel rank 0 in the formula
    and let it fall out. It does not fall out: ``1/(60+0)`` is larger than
    ``1/(60+1)``, the most any *real* rank can contribute, so a candidate one
    retriever never returned would score above one it ranked first. The
    catalog's "missing term contributes 0" is what prevents that inversion.
    """
    only_dense = _features([7], dense=[7], splade=[])
    assert _column(only_dense, "rrf")[0] == pytest.approx(1.0 / (RRF_CONSTANT + 1))

    best_real_contribution = 1.0 / (RRF_CONSTANT + 1)
    sentinel_as_rank = 1.0 / (RRF_CONSTANT + MISSING_RANK)
    assert sentinel_as_rank > best_real_contribution


def test_being_found_by_both_retrievers_outscores_being_found_by_one():
    """The ordering RRF exists to produce, at equal rank."""
    both = _features([7], dense=[7], splade=[7])
    one = _features([7], dense=[7], splade=[])
    assert _column(both, "rrf")[0] > _column(one, "rrf")[0]


def test_rrf_still_prefers_a_strong_single_retriever_to_a_weak_consensus():
    """A rank-1 hit in one retriever beats a rank-100 hit in both.

    This is correct RRF behaviour rather than a defect, and it is pinned so the
    "both is always better" intuition cannot be quietly coded in later.
    """
    top_of_one = _features([7], dense=[7], splade=[])
    mid_of_both = _features(
        [7], dense=_filler(99, avoid=7) + [7], splade=_filler(99, avoid=7) + [7]
    )
    assert _column(top_of_one, "rrf")[0] > _column(mid_of_both, "rrf")[0]


def test_the_source_indicators_are_mutually_exclusive_and_exhaustive():
    features = _features([1, 2, 3], dense=[1, 3], splade=[2, 3])
    dense_only = _column(features, "dense_only")
    splade_only = _column(features, "splade_only")
    both = _column(features, "both_retrievers")

    assert dense_only == pytest.approx([1, 0, 0])
    assert splade_only == pytest.approx([0, 1, 0])
    assert both == pytest.approx([0, 0, 1])
    # Every retrieved candidate falls in exactly one bucket.
    assert (dense_only + splade_only + both) == pytest.approx([1, 1, 1])


def test_a_candidate_neither_retriever_returned_sets_no_indicator():
    features = _features([99], dense=[1], splade=[2])
    for name in ("dense_only", "splade_only", "both_retrievers"):
        assert _column(features, name)[0] == 0.0


def test_a_candidate_a_retriever_never_returned_scores_zero_not_one():
    """The sentinel must never read as a perfect rank.

    Missing is encoded as rank 0, and ``1 - 0/200`` is 1.0 -- the *maximum*.
    Without the guard, every candidate a retriever never returned would present
    as that retriever's top hit, which inverts the feature completely. A range
    check cannot catch it, because 1.0 is a legal value and only its direction
    is wrong; this asserts the value itself, on both retrievers independently.
    """
    features = _features([7], dense=[1, 2, 3], splade=[4, 5, 6])
    assert _column(features, "dense_rank_pct")[0] == 0.0
    assert _column(features, "splade_rank_pct")[0] == 0.0
    assert _column(features, "best_rank_pct")[0] == 0.0
    # Neither retriever has an opinion, so there is nothing to disagree about.
    assert _column(features, "rank_disagreement")[0] == 0.0


def test_one_retriever_missing_does_not_inflate_the_other_columns():
    """Asymmetric absence, checked per retriever rather than only in aggregate."""
    dense_has_it = _features([7], dense=[7], splade=[4, 5, 6])
    assert _column(dense_has_it, "splade_rank_pct")[0] == 0.0
    assert _column(dense_has_it, "dense_rank_pct")[0] == pytest.approx(
        1 - 1 / POOL_DEPTH
    )

    splade_has_it = _features([7], dense=[1, 2, 3], splade=[7])
    assert _column(splade_has_it, "dense_rank_pct")[0] == 0.0
    assert _column(splade_has_it, "splade_rank_pct")[0] == pytest.approx(
        1 - 1 / POOL_DEPTH
    )


def test_disagreement_and_best_rank_are_derived_from_the_two_percentiles():
    features = _features([5], dense=[5], splade=_filler(49, avoid=5) + [5])
    dense_pct = _column(features, "dense_rank_pct")[0]
    splade_pct = _column(features, "splade_rank_pct")[0]
    assert _column(features, "rank_disagreement")[0] == pytest.approx(
        abs(dense_pct - splade_pct)
    )
    assert _column(features, "best_rank_pct")[0] == pytest.approx(
        max(dense_pct, splade_pct)
    )


def test_seed_membership_is_reported_for_candidates_that_are_seeds():
    """Seeds are themselves candidates and are structurally degenerate.

    A seed sits at distance 0 from itself, so downstream structural features
    behave differently on it; A9 is what lets the learner tell it apart.
    """
    features = _features([4, 5, 6], dense=[4, 5, 6], seeds=[5])
    assert _column(features, "is_seed") == pytest.approx([0, 1, 0])


def test_a_seed_that_no_retriever_returned_is_still_marked():
    features = _features([42], dense=[1], splade=[2], seeds=[42])
    assert _column(features, "is_seed")[0] == 1.0


# --------------------------------------------------------------------------
# The collision A4-A6 exist to resolve
# --------------------------------------------------------------------------


def test_last_rank_and_never_retrieved_are_indistinguishable_in_the_percentile():
    """The exact degeneracy the catalog's rationale is about.

    If this ever stopped holding, A4-A6 would be redundant -- so the test is
    written to fail loudly if the percentile silently gains a way to encode
    absence, which would mean the scale had changed underneath the frontier.
    """
    ranked_last = _features([7], splade=_filler(POOL_DEPTH - 1, avoid=7) + [7])
    never = _features([7], splade=[1, 2, 3])

    assert _column(ranked_last, "splade_rank_pct")[0] == 0.0
    assert _column(never, "splade_rank_pct")[0] == 0.0


def test_but_the_indicators_and_rrf_do_separate_them():
    """The distinction survives in the columns built to carry it."""
    ranked_last = _features([7], splade=_filler(POOL_DEPTH - 1, avoid=7) + [7])
    never = _features([7], splade=[1, 2, 3])

    assert _column(ranked_last, "splade_only")[0] == 1.0
    assert _column(never, "splade_only")[0] == 0.0
    assert _column(ranked_last, "rrf")[0] > _column(never, "rrf")[0] == 0.0


# --------------------------------------------------------------------------
# Ranges, shapes and refusals
# --------------------------------------------------------------------------


def test_every_feature_stays_inside_its_declared_range():
    rng = np.random.default_rng(0)
    universe = np.arange(400)
    dense = rng.choice(universe, size=POOL_DEPTH, replace=False)
    splade = rng.choice(universe, size=POOL_DEPTH, replace=False)
    features = retrieval_prior_features(
        universe, dense_ranked=dense, splade_ranked=splade, seeds=universe[:5]
    )

    for name in ("dense_rank_pct", "splade_rank_pct", "rank_disagreement", "best_rank_pct"):
        column = _column(features, name)
        assert column.min() >= 0.0 and column.max() <= 1.0

    rrf = _column(features, "rrf")
    assert rrf.min() >= 0.0 and rrf.max() <= MAX_RRF

    for name in ("dense_only", "splade_only", "both_retrievers", "is_seed"):
        assert set(np.unique(_column(features, name))) <= {0.0, 1.0}


def test_the_matrix_has_one_column_per_named_feature():
    features = _features([1, 2], dense=[1], splade=[2])
    assert features.shape == (2, len(NAMES))
    assert len(set(NAMES)) == 9


def test_a_duplicated_candidate_in_a_ranking_is_refused():
    """A duplicate would give one candidate two ranks and double its RRF term.

    ``rrf_rankings`` refuses this for the same reason; a silent resolution here
    would make the two disagree about the same frozen arrays.
    """
    with pytest.raises(ValueError, match="duplicate"):
        rank_lookup([3, 4, 3])


def test_a_ranking_deeper_than_the_frozen_pool_is_refused():
    with pytest.raises(ValueError, match="frozen pool"):
        rank_lookup(list(range(POOL_DEPTH + 1)))


def test_an_empty_candidate_set_yields_an_empty_matrix_not_an_error():
    assert _features([], dense=[1], splade=[2]).shape == (0, len(NAMES))
