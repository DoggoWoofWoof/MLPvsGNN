"""M0A.1: where do the golds that structural expansion recovers actually come from?

M0A measured that bounded structural expansion recovers missing golds. It did
not measure whether those golds were already sitting in R2's context universe
``U2 = TARGET_H1(Cq)``. The two neighbourhoods are not the same set -- ``U2``
is built from in-neighbours, the expansion frontier is undirected -- so the
question is real rather than definitional.

The declaration files three classes that are **not disjoint**: a gold in ``U2``
but not in ``Cq_struct`` satisfies both ``R2_CONTEXT_RECOVERABLE`` and
``STILL_MISSING``. That defect was found before any M0A.1 number existed. It is
not repaired by editing the class definitions. Instead the primitive measured
here is the 2x2 cross-tabulation of (in ``U2``) x (in ``Cq_struct``); the three
filed classes are derived from it unchanged, and the overlapping cell is named
and counted so neither reading is hidden.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

CROSS_TAB_CELLS = (
    "IN_U2_AND_RECOVERED",
    "IN_U2_NOT_RECOVERED",
    "BEYOND_U2_RECOVERED",
    "BEYOND_U2_NOT_RECOVERED",
)

# The overlapping cell of the filed partition. It is where R2 held the answer
# as context and bounded structural expansion still failed to promote it.
OVERLAPPING_CELL = "IN_U2_NOT_RECOVERED"

FILED_CLASSES = {
    "R2_CONTEXT_RECOVERABLE": ("IN_U2_AND_RECOVERED", "IN_U2_NOT_RECOVERED"),
    "R3_BEYOND_R2": ("BEYOND_U2_RECOVERED",),
    "STILL_MISSING": ("IN_U2_NOT_RECOVERED", "BEYOND_U2_NOT_RECOVERED"),
}


def classify_query_golds(
    *,
    golds: np.ndarray,
    pool: np.ndarray,
    u2: np.ndarray,
    expanded: np.ndarray,
) -> dict[str, np.ndarray]:
    """One query's golds that are absent from ``Cq``, split across the 2x2.

    Golds already in ``pool`` are excluded: the partition is defined over golds
    absent from ``Cq``, and a gold already scoreable was never missing.
    """

    golds = np.asarray(golds, dtype=np.int64)
    pool = np.asarray(pool, dtype=np.int64)
    missing = golds[~np.isin(golds, pool)]
    in_u2 = np.isin(missing, np.asarray(u2, dtype=np.int64))
    recovered = np.isin(missing, np.asarray(expanded, dtype=np.int64))
    return {
        "IN_U2_AND_RECOVERED": missing[in_u2 & recovered],
        "IN_U2_NOT_RECOVERED": missing[in_u2 & ~recovered],
        "BEYOND_U2_RECOVERED": missing[~in_u2 & recovered],
        "BEYOND_U2_NOT_RECOVERED": missing[~in_u2 & ~recovered],
    }


def overlap_partition(per_query: list[dict[str, np.ndarray]]) -> dict[str, object]:
    """Gold-instance counts and query counts, for the 2x2 and the filed classes.

    A gold instance is one (query, gold node) pair, so a node that is gold for
    two queries counts twice -- the partition is about answers, not about nodes.
    Query counts are the number of queries holding at least one gold in a cell,
    and they do not sum to the number of queries: one query can contribute to
    several cells at once.
    """

    instances = {cell: 0 for cell in CROSS_TAB_CELLS}
    queries = {cell: 0 for cell in CROSS_TAB_CELLS}
    for row in per_query:
        for cell in CROSS_TAB_CELLS:
            size = int(np.asarray(row[cell]).size)
            instances[cell] += size
            queries[cell] += int(size > 0)

    total = sum(instances.values())
    classes = {
        name: {
            "gold_instances": sum(instances[cell] for cell in cells),
            "from_cells": list(cells),
        }
        for name, cells in FILED_CLASSES.items()
    }
    for name, cells in FILED_CLASSES.items():
        classes[name]["queries"] = sum(
            1
            for row in per_query
            if any(np.asarray(row[cell]).size for cell in cells)
        )

    return {
        "cross_tabulation": {
            "gold_instances": instances,
            "queries": queries,
        },
        "filed_classes": classes,
        "missing_gold_instances_total": total,
        "queries_with_a_missing_gold": sum(
            1
            for row in per_query
            if any(np.asarray(row[cell]).size for cell in CROSS_TAB_CELLS)
        ),
        "the_overlapping_cell": {
            "cell": OVERLAPPING_CELL,
            "gold_instances": instances[OVERLAPPING_CELL],
            "queries": queries[OVERLAPPING_CELL],
            "counted_in": [
                name
                for name, cells in FILED_CLASSES.items()
                if OVERLAPPING_CELL in cells
            ],
            "why": (
                "R2 held this gold as context and bounded structural expansion "
                "did not promote it. The filed classes are not disjoint, so "
                "this cell is counted inside two of them; it is reported alone "
                "so neither total is mistaken for a partition."
            ),
        },
        "cross_tabulation_is_a_partition": total
        == sum(instances[cell] for cell in CROSS_TAB_CELLS),
    }


def recovery_share(partition: dict[str, object]) -> dict[str, float]:
    """What fraction of recovered golds came from inside R2's context.

    Denominator is recovered golds, not all missing golds, because the question
    is where the recovery came from -- not how much recovery there was.
    """

    cells = partition["cross_tabulation"]["gold_instances"]
    inside = int(cells["IN_U2_AND_RECOVERED"])
    beyond = int(cells["BEYOND_U2_RECOVERED"])
    recovered = inside + beyond
    if recovered == 0:
        return {
            "recovered_gold_instances": 0,
            "already_in_r2_context": 0.0,
            "beyond_r2_context": 0.0,
            "undefined_because_nothing_was_recovered": True,
        }
    return {
        "recovered_gold_instances": recovered,
        "already_in_r2_context": inside / recovered,
        "beyond_r2_context": beyond / recovered,
        "undefined_because_nothing_was_recovered": False,
    }


def marginal_recovery(points: list[dict[str, object]]) -> list[dict[str, object]]:
    """Gold recovered per extra admitted node, between consecutive curve points.

    Saturation analysis, not hyperparameter fitting: this locates where the
    recovery arrives, and no per-dataset budget is selected from it.

    The step into the final unbounded point also lifts the per-seed cap, so it
    is not a prefix of the same ordering as the numeric points. That step is
    flagged rather than quietly averaged in with the others.
    """

    rows: list[dict[str, object]] = []
    for previous, current in pairwise(points):
        added = float(current["added_nodes_per_query_mean"]) - float(
            previous["added_nodes_per_query_mean"]
        )
        gained = int(current["missing_golds_recovered"]) - int(
            previous["missing_golds_recovered"]
        )
        rows.append(
            {
                "from": previous["budget"],
                "to": current["budget"],
                "added_nodes_per_query": added,
                "golds_gained": gained,
                "recovered_gold_per_added_node": (gained / added) if added > 0 else None,
                "step_also_lifts_the_per_seed_cap": bool(
                    current.get("lifts_per_seed_cap", False)
                ),
            }
        )
    return rows
