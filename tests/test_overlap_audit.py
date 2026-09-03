"""The overlap partition must survive the fact that its own classes overlap.

The declaration files three classes that are not disjoint. These tests pin the
2x2 cross-tabulation as the primitive, check that the filed classes are derived
from it unchanged, and force the overlapping cell to be visible in both totals
rather than silently assigned to one.
"""

from __future__ import annotations

import numpy as np
import pytest

from mp_retrieval.overlap_audit import (
    CROSS_TAB_CELLS,
    FILED_CLASSES,
    OVERLAPPING_CELL,
    classify_query_golds,
    marginal_recovery,
    overlap_partition,
    recovery_share,
)


def _classify(golds, pool, u2, expanded):
    return classify_query_golds(
        golds=np.array(golds, dtype=np.int64),
        pool=np.array(pool, dtype=np.int64),
        u2=np.array(u2, dtype=np.int64),
        expanded=np.array(expanded, dtype=np.int64),
    )


def _sizes(row):
    return {cell: int(np.asarray(row[cell]).size) for cell in CROSS_TAB_CELLS}


# --- what the classification is over ---


def test_a_gold_already_scoreable_is_not_a_missing_gold(): 
    """The partition is defined over golds absent from Cq."""
    row = _classify(golds=[7], pool=[7, 9], u2=[7, 9, 11], expanded=[7, 9, 11])
    assert sum(_sizes(row).values()) == 0


def test_a_gold_in_r2_context_and_recovered_lands_in_one_cell():
    row = _classify(golds=[11], pool=[7, 9], u2=[7, 9, 11], expanded=[7, 9, 11])
    assert _sizes(row) == {
        "IN_U2_AND_RECOVERED": 1,
        "IN_U2_NOT_RECOVERED": 0,
        "BEYOND_U2_RECOVERED": 0,
        "BEYOND_U2_NOT_RECOVERED": 0,
    }


def test_a_gold_outside_r2_context_but_recovered_is_the_interesting_case():
    """This cell is the whole reason the stage exists: R3 beyond TARGET_H1."""
    row = _classify(golds=[13], pool=[7, 9], u2=[7, 9, 11], expanded=[7, 9, 13])
    assert _sizes(row)["BEYOND_U2_RECOVERED"] == 1


def test_a_gold_in_context_that_expansion_did_not_promote_is_the_overlap():
    row = _classify(golds=[11], pool=[7, 9], u2=[7, 9, 11], expanded=[7, 9])
    assert _sizes(row)[OVERLAPPING_CELL] == 1


def test_a_gold_nowhere_near_the_graph_is_still_missing():
    row = _classify(golds=[99], pool=[7, 9], u2=[7, 9, 11], expanded=[7, 9, 11])
    assert _sizes(row)["BEYOND_U2_NOT_RECOVERED"] == 1


def test_every_missing_gold_lands_in_exactly_one_cell():
    row = _classify(
        golds=[7, 11, 13, 15, 99], pool=[7, 9], u2=[7, 9, 11, 15], expanded=[7, 9, 11, 13]
    )
    sizes = _sizes(row)
    assert sum(sizes.values()) == 4  # 7 is in the pool and is not missing
    assert sizes == {
        "IN_U2_AND_RECOVERED": 1,
        "IN_U2_NOT_RECOVERED": 1,
        "BEYOND_U2_RECOVERED": 1,
        "BEYOND_U2_NOT_RECOVERED": 1,
    }


# --- the filed classes are derived, and their overlap is visible ---


def test_the_filed_classes_are_the_sums_of_their_cells():
    rows = [
        _classify(golds=[11], pool=[7], u2=[7, 11], expanded=[7, 11]),
        _classify(golds=[12], pool=[7], u2=[7, 12], expanded=[7]),
        _classify(golds=[13], pool=[7], u2=[7], expanded=[7, 13]),
        _classify(golds=[14], pool=[7], u2=[7], expanded=[7]),
    ]
    partition = overlap_partition(rows)
    cells = partition["cross_tabulation"]["gold_instances"]
    for name, members in FILED_CLASSES.items():
        assert partition["filed_classes"][name]["gold_instances"] == sum(
            cells[cell] for cell in members
        ), name


def test_the_overlapping_cell_is_counted_in_two_filed_classes():
    """The defect is disclosed by the data structure, not by a footnote."""
    rows = [_classify(golds=[12], pool=[7], u2=[7, 12], expanded=[7])]
    partition = overlap_partition(rows)
    overlap = partition["the_overlapping_cell"]
    assert overlap["cell"] == OVERLAPPING_CELL
    assert overlap["gold_instances"] == 1
    assert sorted(overlap["counted_in"]) == ["R2_CONTEXT_RECOVERABLE", "STILL_MISSING"]
    classes = partition["filed_classes"]
    assert classes["R2_CONTEXT_RECOVERABLE"]["gold_instances"] == 1
    assert classes["STILL_MISSING"]["gold_instances"] == 1


def test_the_filed_classes_do_not_sum_to_the_total_and_that_is_the_point():
    """If they summed, the overlap would have been silently resolved."""
    rows = [_classify(golds=[12], pool=[7], u2=[7, 12], expanded=[7])]
    partition = overlap_partition(rows)
    filed = sum(row["gold_instances"] for row in partition["filed_classes"].values())
    assert filed == 2
    assert partition["missing_gold_instances_total"] == 1
    assert partition["cross_tabulation_is_a_partition"] is True


def test_query_counts_count_queries_not_gold_instances():
    rows = [
        _classify(golds=[11, 12], pool=[7], u2=[7, 11, 12], expanded=[7, 11, 12]),
        _classify(golds=[13], pool=[7], u2=[7, 13], expanded=[7, 13]),
    ]
    partition = overlap_partition(rows)
    assert partition["cross_tabulation"]["gold_instances"]["IN_U2_AND_RECOVERED"] == 3
    assert partition["cross_tabulation"]["queries"]["IN_U2_AND_RECOVERED"] == 2
    assert partition["queries_with_a_missing_gold"] == 2


def test_one_query_can_contribute_to_several_cells_at_once():
    rows = [_classify(golds=[11, 12], pool=[7], u2=[7, 11], expanded=[7, 11, 12])]
    partition = overlap_partition(rows)
    queries = partition["cross_tabulation"]["queries"]
    assert queries["IN_U2_AND_RECOVERED"] == 1
    assert queries["BEYOND_U2_RECOVERED"] == 1
    assert sum(queries.values()) > partition["queries_with_a_missing_gold"]


def test_an_empty_audit_reports_zero_rather_than_dividing():
    partition = overlap_partition([])
    assert partition["missing_gold_instances_total"] == 0
    share = recovery_share(partition)
    assert share["undefined_because_nothing_was_recovered"] is True
    assert share["already_in_r2_context"] == 0.0


# --- where the recovery came from ---


def test_recovery_share_is_over_recovered_golds_not_all_missing_golds():
    rows = [
        _classify(golds=[11], pool=[7], u2=[7, 11], expanded=[7, 11]),
        _classify(golds=[13], pool=[7], u2=[7], expanded=[7, 13]),
        _classify(golds=[99], pool=[7], u2=[7], expanded=[7]),
    ]
    share = recovery_share(overlap_partition(rows))
    assert share["recovered_gold_instances"] == 2
    assert share["already_in_r2_context"] == pytest.approx(0.5)
    assert share["beyond_r2_context"] == pytest.approx(0.5)


def test_a_recovery_entirely_inside_r2_context_reads_as_one():
    rows = [_classify(golds=[11], pool=[7], u2=[7, 11], expanded=[7, 11])]
    share = recovery_share(overlap_partition(rows))
    assert share["already_in_r2_context"] == pytest.approx(1.0)
    assert share["beyond_r2_context"] == pytest.approx(0.0)


# --- the saturation curve ---


def _point(budget, added, recovered, *, lifts=False):
    return {
        "budget": budget,
        "added_nodes_per_query_mean": added,
        "missing_golds_recovered": recovered,
        "lifts_per_seed_cap": lifts,
    }


def test_marginal_recovery_is_gold_gained_per_node_added():
    rows = marginal_recovery([_point(4, 4.0, 10), _point(8, 8.0, 14)])
    assert rows[0]["golds_gained"] == 4
    assert rows[0]["added_nodes_per_query"] == pytest.approx(4.0)
    assert rows[0]["recovered_gold_per_added_node"] == pytest.approx(1.0)


def test_a_saturated_step_reads_as_zero_not_as_missing():
    rows = marginal_recovery([_point(16, 16.0, 43), _point(32, 30.0, 43)])
    assert rows[0]["golds_gained"] == 0
    assert rows[0]["recovered_gold_per_added_node"] == pytest.approx(0.0)


def test_a_step_that_adds_no_nodes_does_not_divide_by_zero():
    rows = marginal_recovery([_point(32, 30.0, 43), _point(64, 30.0, 43)])
    assert rows[0]["recovered_gold_per_added_node"] is None


def test_the_step_into_the_unbounded_point_is_flagged():
    """It lifts the per-seed cap, so it is not a prefix of the same ordering."""
    rows = marginal_recovery(
        [_point(64, 30.0, 43), _point("full_bounded_n1_frontier", 167.0, 43, lifts=True)]
    )
    assert rows[0]["step_also_lifts_the_per_seed_cap"] is True
    assert rows[-1]["to"] == "full_bounded_n1_frontier"


def test_a_single_point_curve_has_no_steps():
    assert marginal_recovery([_point(4, 4.0, 10)]) == []
