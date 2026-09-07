"""The Stage-0 rules, on cases whose answers are fixed by construction.

Two kinds of check run here. Most are small worked examples: four candidates,
one gold, a ranking chosen so the right answer is obvious by inspection. The
rest are agreement checks against machinery this track already froze -- the M2C
probe's recall, the existing RRF -- because a second definition of recall or a
second fusion in the same track would produce numbers that look comparable with
the declaration's and are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval import m2d_semantic_fusion as fusion
from mp_retrieval.m2c_structural_offset import (
    RRF_CONSTANT,
    rank_positions,
    reciprocal_rank_fusion,
)

#: Four candidates with deliberately non-contiguous ids, so a bug that indexes
#: the pool by node id instead of by position cannot pass.
POOL = np.array([10, 20, 30, 40], dtype=np.int64)


def ranks_from(order: list[int]) -> np.ndarray:
    """Ranks over POOL from a stated best-to-worst node order."""

    position = {int(node): index for index, node in enumerate(POOL.tolist())}
    out = np.empty(POOL.size, dtype=np.int64)
    for rank, node in enumerate(order, start=1):
        out[position[node]] = rank
    return out


# ---------------------------------------------------------------------------
# Metrics agree with the definition the track already froze
# ---------------------------------------------------------------------------


def test_recall_and_mrr_match_the_m2c_probe_definition():
    """If these two ever diverge, a Stage-0 row and a baseline-table row would
    be different quantities under the same column name."""

    from scripts.run_m2c_stage0_probe import _metrics as m2c_metrics

    generator = np.random.default_rng(0)
    for _ in range(200):
        size = int(generator.integers(1, 40))
        scores = generator.normal(size=size)
        pool = generator.permutation(size * 3)[:size]
        relevant = generator.random(size) < 0.2
        ranks = rank_positions(scores, pool)
        assert fusion.metrics(ranks, relevant) == m2c_metrics(ranks, relevant)


def test_recall_is_over_the_relevant_candidates_in_the_pool():
    relevant = np.array([True, False, True, False])
    ranks = ranks_from([10, 20, 30, 40])  # both golds at ranks 1 and 3
    row = fusion.metrics(ranks, relevant)
    assert row["recall@1"] == 0.5
    assert row["recall@5"] == 1.0
    assert row["mrr"] == 1.0


def test_a_query_with_no_relevant_candidate_scores_zero_and_says_so():
    row = fusion.metrics(ranks_from([10, 20, 30, 40]), np.zeros(4, dtype=bool))
    assert row["mrr"] == 0.0
    assert row["scored"] == 0.0


def test_mean_metrics_of_nothing_is_not_a_crash_or_a_nan():
    empty = fusion.mean_metrics([])
    assert empty["queries"] == 0
    assert empty["mrr"] == 0.0


# ---------------------------------------------------------------------------
# The unranked-candidate convention
# ---------------------------------------------------------------------------


def test_a_pool_member_the_source_never_returned_contributes_exactly_zero():
    """The convention that separates this from an arbitrary choice: it is what
    linear_control.rank_feature_rows already does for the same two sources."""

    contribution = fusion.source_contribution(POOL, np.array([30, 10]))
    assert contribution[POOL.tolist().index(30)] == 1.0 / (RRF_CONSTANT + 1)
    assert contribution[POOL.tolist().index(10)] == 1.0 / (RRF_CONSTANT + 2)
    assert contribution[POOL.tolist().index(20)] == 0.0
    assert contribution[POOL.tolist().index(40)] == 0.0


def test_the_source_rank_is_a_position_in_the_source_list_not_in_the_pool():
    """40 is last in the pool and first in the source. Its contribution has to
    be the first-place one; a bug that ranked within the pool would give it the
    last-place value."""

    contribution = fusion.source_contribution(POOL, np.array([40, 30, 20, 10]))
    assert contribution[POOL.tolist().index(40)] == 1.0 / (RRF_CONSTANT + 1)
    assert contribution[POOL.tolist().index(10)] == 1.0 / (RRF_CONSTANT + 4)


def test_a_repeated_candidate_in_a_source_ranking_is_refused():
    with pytest.raises(ValueError, match="must not repeat"):
        fusion.source_contribution(POOL, np.array([10, 10]))


def test_unranked_candidates_are_ranked_last_when_a_source_is_used_alone():
    contribution = fusion.source_contribution(POOL, np.array([30]))
    ranks = fusion.unranked_last(contribution, POOL)
    assert ranks[POOL.tolist().index(30)] == 1
    # The other three are genuinely tied for this source and fall to the frozen
    # tie-break, ascending node id: 10, 20, 40.
    assert list(ranks) == [2, 3, 1, 4]


# ---------------------------------------------------------------------------
# Fusion agrees with the frozen implementation where both apply
# ---------------------------------------------------------------------------


def test_fusion_reproduces_the_existing_rrf_when_every_source_covers_the_pool():
    generator = np.random.default_rng(1)
    for _ in range(100):
        size = int(generator.integers(2, 30))
        pool = generator.permutation(size * 4)[:size]
        a = rank_positions(generator.normal(size=size), pool)
        b = rank_positions(generator.normal(size=size), pool)

        mine = fusion.fuse_ranks(
            [fusion.full_coverage_contribution(a), fusion.full_coverage_contribution(b)], pool
        )
        theirs = rank_positions(reciprocal_rank_fusion([a, b], pool), pool)
        assert np.array_equal(mine, theirs)


def test_fusing_one_source_with_itself_returns_that_sources_order():
    ranks = ranks_from([30, 10, 40, 20])
    fused = fusion.fuse_ranks([fusion.full_coverage_contribution(ranks)], POOL)
    assert np.array_equal(fused, ranks)


def test_a_partial_source_can_only_promote_never_demote():
    """The point of the zero convention. Adding a source that ranked only one
    candidate must lift that candidate and leave the others in their existing
    relative order, rather than reshuffling them by node id."""

    base = ranks_from([10, 20, 30, 40])
    partial = fusion.source_contribution(POOL, np.array([40]))
    fused = fusion.fuse_ranks([fusion.full_coverage_contribution(base), partial], POOL)

    assert fused[POOL.tolist().index(40)] < base[POOL.tolist().index(40)]
    others = [10, 20, 30]
    before = [base[POOL.tolist().index(node)] for node in others]
    after = [fused[POOL.tolist().index(node)] for node in others]
    assert np.argsort(before).tolist() == np.argsort(after).tolist()


def test_a_negative_contribution_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        fusion.fuse_ranks([np.array([-1.0, 0.0, 0.0, 0.0])], POOL)


def test_the_constant_is_the_projects_own_and_is_sixty():
    import yaml

    budget = yaml.safe_load(
        (REPO_ROOT / "configs" / "candidate_budget.yaml").read_text(encoding="utf-8")
    )
    assert fusion.RRF_CONSTANT == 60
    assert budget["candidate_contract"]["rrf_constant"] == 60


# ---------------------------------------------------------------------------
# Section 4: complementarity
# ---------------------------------------------------------------------------


def test_the_two_directions_of_disagreement_are_counted_separately():
    """The count the declaration says the phase turns on: S4-wrong/S3-right
    large with S3-wrong/S4-right near zero means S4 is dominated."""

    relevant = np.array([False, True, False, False])  # node 20 is gold
    s4 = ranks_from([10, 20, 30, 40])  # top-1 is 10: wrong
    s3 = ranks_from([20, 10, 30, 40])  # top-1 is 20: right

    row = fusion.complementarity(s4, s3, relevant, POOL)
    assert row["s4_wrong_s3_right"] is True
    assert row["s3_wrong_s4_right"] is False
    assert row["neither_right_at_1"] is False

    reverse = fusion.complementarity(s3, s4, relevant, POOL)
    assert reverse["s4_wrong_s3_right"] is False
    assert reverse["s3_wrong_s4_right"] is True


def test_the_cross_ranks_report_where_the_other_model_put_this_models_find():
    """Two golds, each model finding a different one first. The interesting
    number is where the OTHER model buried it."""

    relevant = np.array([False, True, False, True])  # 20 and 40 are gold
    s4 = ranks_from([20, 10, 30, 40])  # finds 20 at rank 1; 40 is at rank 4
    s3 = ranks_from([40, 30, 10, 20])  # finds 40 at rank 1; 20 is at rank 4

    row = fusion.complementarity(s4, s3, relevant, POOL)
    assert row["s3_best_relevant_rank_under_s4"] == 4
    assert row["s4_best_relevant_rank_under_s3"] == 4
    assert row["same_best_relevant_node"] is False


def test_the_same_gold_found_at_different_ranks_is_its_own_category():
    relevant = np.array([False, True, False, False])
    s4 = ranks_from([10, 20, 30, 40])  # gold at 2
    s3 = ranks_from([20, 10, 30, 40])  # same gold at 1

    row = fusion.complementarity(s4, s3, relevant, POOL)
    assert row["same_best_relevant_node"] is True
    assert row["both_found_same_relevant_ordered_differently"] is True


def test_top_k_overlap_is_reported_at_all_three_cutoffs():
    s4 = ranks_from([10, 20, 30, 40])
    s3 = ranks_from([20, 10, 40, 30])
    row = fusion.complementarity(s4, s3, np.array([True, False, False, False]), POOL)
    assert row["overlap"][1] == 0
    assert row["overlap"][5] == 4
    assert row["overlap"][20] == 4


def test_a_query_with_no_gold_reports_none_rather_than_zero(caplog):
    """A zero here would be a measurement of something that does not exist."""

    row = fusion.complementarity(
        ranks_from([10, 20, 30, 40]),
        ranks_from([20, 10, 30, 40]),
        np.zeros(4, dtype=bool),
        POOL,
    )
    assert row["has_relevant_in_pool"] is False
    assert row["s3_best_relevant_rank_under_s4"] is None
    assert row["same_best_relevant_node"] is None
    assert row["overlap"][5] == 4  # the overlap is still defined


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError, match="same candidate pool"):
        fusion.complementarity(
            np.array([1, 2, 3]), np.array([1, 2, 3, 4]), np.zeros(4, dtype=bool), POOL
        )


# ---------------------------------------------------------------------------
# Section 6: rescue
# ---------------------------------------------------------------------------


def test_a_query_s4_already_gets_right_is_not_in_the_population():
    relevant = np.array([True, False, False, False])
    assert (
        fusion.rescue_row(ranks_from([10, 20, 30, 40]), {"S3": ranks_from([10, 20, 30, 40])}, relevant)
        is None
    )


def test_a_query_whose_gold_was_never_a_candidate_is_not_in_the_population():
    """An admission failure is not a ranking error, and counting it as one
    would put a floor under every rescuer's failure rate."""

    assert (
        fusion.rescue_row(
            ranks_from([10, 20, 30, 40]),
            {"S3": ranks_from([10, 20, 30, 40])},
            np.zeros(4, dtype=bool),
        )
        is None
    )


def test_the_rescuers_are_those_that_reach_the_cutoff():
    relevant = np.array([False, False, False, True])  # gold is 40
    s4 = ranks_from([10, 20, 30, 40])  # gold at 4: wrong at 1
    others = {
        "Dense": ranks_from([40, 10, 20, 30]),  # gold at 1
        "SPLADE": ranks_from([10, 40, 20, 30]),  # gold at 2
        "S3": ranks_from([10, 20, 30, 40]),  # gold at 4
    }
    row = fusion.rescue_row(s4, others, relevant)
    assert row["s4_best_relevant_rank"] == 4
    assert row["rescued_at"][1]["rescuers"] == ["Dense"]
    assert row["rescued_at"][5]["rescuers"] == ["Dense", "S3", "SPLADE"]
    assert row["rescued_at"][1]["none"] is False


def test_a_query_nothing_rescues_is_recorded_as_such():
    relevant = np.array([False, False, False, True])
    s4 = ranks_from([10, 20, 30, 40])
    others = {"Dense": ranks_from([10, 20, 30, 40]), "S3": ranks_from([20, 10, 30, 40])}
    row = fusion.rescue_row(s4, others, relevant, cutoffs=(1,))
    assert row["rescued_at"][1]["rescuers"] == []
    assert row["rescued_at"][1]["none"] is True


def test_rescue_counts_are_per_rescuer_and_do_not_pretend_to_partition():
    """Two rescuers on one query is the normal case, and forcing a single label
    would need a precedence order nobody declared."""

    relevant = np.array([False, False, False, True])
    rows = [
        fusion.rescue_row(
            ranks_from([10, 20, 30, 40]),
            {"Dense": ranks_from([40, 10, 20, 30]), "SPLADE": ranks_from([40, 30, 20, 10])},
            relevant,
        )
    ]
    table = fusion.rescue_table(
        rows, names=("Dense", "SPLADE"), excluded={"no_relevant_in_pool": 0}, cutoffs=(1,)
    )
    assert table["by_cutoff"][1]["rescued_by"] == {"Dense": 1, "SPLADE": 1}
    assert table["by_cutoff"][1]["rescued_by_any"] == 1
    assert table["by_cutoff"][1]["rescued_by_none"] == 0


def test_the_table_carries_the_measurability_fraction_and_the_excluded_counts():
    """M2C's lesson, and the declaration requires it in advance: a conditioned
    statistic that does not say what fraction it was measurable on can look
    clean on 6% of the errors."""

    relevant = np.array([False, False, False, True])
    rows = [
        fusion.rescue_row(
            ranks_from([10, 20, 30, 40]), {"Dense": ranks_from([40, 10, 20, 30])}, relevant
        )
    ] * 3
    table = fusion.rescue_table(
        rows, names=("Dense",), excluded={"no_relevant_in_pool": 1, "s4_top1_already_right": 7}
    )
    assert table["population"] == 3
    assert table["fraction_of_the_error_population_measurable"] == 1.0
    assert table["queries_whose_gold_was_never_a_candidate"] == 1
    assert table["share_of_all_top1_failures_that_are_admission_failures"] == 0.25
    assert table["excluded"]["s4_top1_already_right"] == 7


def test_an_empty_population_produces_zeros_rather_than_a_division_error():
    table = fusion.rescue_table([], names=("Dense",), excluded={"no_relevant_in_pool": 0})
    assert table["population"] == 0
    assert table["by_cutoff"][1]["share_rescued_by"]["Dense"] == 0.0
    assert table["fraction_of_the_error_population_measurable"] == 0.0
