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

Whether ``A64(q) subset U2`` is a mathematical identity (M0B Safeguard B)
--------------------------------------------------------------------------
It is not one, in general, and this module does not assert it as an
invariant anywhere. ``U2 = TARGET_H1(Cq)`` walks true in-neighbours of the
*raw, directed* ``graph.pt`` (``dataset.rowptr``/``dataset.col`` ->
``build_operators`` -> ``operators.forward``, see ``graph_context.py``).
``A64`` walks the frontier of the *symmetrised* ``structural_only`` edge
family -- a different source (reconstructed from per-document neighbour
lists in ``edge_provenance.py``, not from ``graph.pt``), explicitly passed
through ``_undirected()`` / ``symmetric_csr()`` before ``expand()`` sees it
(see ``scripts/run_m0a1_overlap.py``).

Containment needs two separate, dataset-specific facts to both hold, neither
of which follows from the code's structure alone:

1. ``graph.pt`` (sealed A) is bidirectionally closed for that dataset --
   symmetrising it would change nothing, so every in-neighbour edge ``U2``
   used is also available as an out-neighbour. Verified per dataset in
   ``outputs/edge_provenance_analysis.json``'s ``graph_audits`` block: true
   for ``squad_clean``, ``2wiki_clean``, ``musique_clean``, ``metaqa``, and
   ``webqsp``; **false only for ``hotpotqa_clean``**, whose directed
   receptive field is strictly smaller than its symmetrised one.
2. ``structural_only``'s edges are fully covered by sealed A's edges --
   tracked by ``edge_provenance.reconstruct_edge_families()``'s own
   ``structural_coverage_by_sealed_a`` diagnostic. That diagnostic's mere
   existence is evidence the repo's own authors did not assume full
   coverage; it is measured per dataset, not guaranteed by construction.

Because fact 2 is not independently confirmed to be 1.0 anywhere, containment
is measured as a rate here (``admitted_node_overlap`` /
``aggregate_admitted_node_overlap`` below) on every dataset, including the
five that are bidirectionally closed. "Expected to run high on the closed
five" and "proven to equal 1.0" are different claims; only the weaker one is
filed.
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


# --- containment of A64 in U2 (M0B Safeguard B; see the module docstring) ---


def admitted_node_overlap(*, admitted: np.ndarray, u2: np.ndarray) -> dict[str, np.ndarray]:
    """One query's A64 admissions, split by membership in R2's context U2.

    Unlike ``classify_query_golds``, this is not a gold-recovery measurement:
    it covers every admitted node regardless of gold status, because the
    containment question is about what R3 newly scores, not about which of
    that is gold. See the module docstring for why ``A64 subset U2`` is
    measured here rather than assumed.
    """

    admitted = np.asarray(admitted, dtype=np.int64)
    in_u2 = np.isin(admitted, np.asarray(u2, dtype=np.int64))
    return {
        "ADMITTED_IN_U2": admitted[in_u2],
        "ADMITTED_BEYOND_U2": admitted[~in_u2],
    }


def aggregate_admitted_node_overlap(
    per_query: list[dict[str, np.ndarray]],
) -> dict[str, object]:
    """Containment rate of A64 in U2, aggregated over queries.

    Reports a rate, never a boolean. A rate of 1.0 on some dataset would be
    consistent with containment being a coincidence of that dataset's graph
    rather than a proven identity -- see the module docstring for the two
    facts that would need to hold for it to be provable, and why this
    function does not attempt to distinguish "coincidentally 1.0" from
    "structurally guaranteed" on its own.
    """

    in_u2 = sum(int(np.asarray(row["ADMITTED_IN_U2"]).size) for row in per_query)
    beyond = sum(int(np.asarray(row["ADMITTED_BEYOND_U2"]).size) for row in per_query)
    total = in_u2 + beyond
    if total == 0:
        return {
            "admitted_node_instances_total": 0,
            "admitted_in_u2": 0,
            "admitted_beyond_u2": 0,
            "containment_rate": None,
            "undefined_because_nothing_was_admitted": True,
        }
    return {
        "admitted_node_instances_total": total,
        "admitted_in_u2": in_u2,
        "admitted_beyond_u2": beyond,
        "containment_rate": in_u2 / total,
        "undefined_because_nothing_was_admitted": False,
    }


# --- NODE_ROLE (M0B Safeguard A) ---

ROLE_RETRIEVAL_CANDIDATE = "RETRIEVAL_CANDIDATE"
ROLE_STRUCTURAL_SCORED_CANDIDATE = "STRUCTURAL_SCORED_CANDIDATE"
ROLE_CONTEXT_ONLY = "CONTEXT_ONLY"
NODE_ROLES = (ROLE_RETRIEVAL_CANDIDATE, ROLE_STRUCTURAL_SCORED_CANDIDATE, ROLE_CONTEXT_ONLY)


def node_roles(
    *, cq: np.ndarray, scored: np.ndarray, context: np.ndarray
) -> dict[str, np.ndarray]:
    """Deterministic, zero-training role partition of one query's context universe.

    Exactly one role per node, by construction: ``RETRIEVAL_CANDIDATE`` is
    ``cq``; ``STRUCTURAL_SCORED_CANDIDATE`` is ``scored minus cq`` (empty
    under R1 and R2, where ``scored == cq``; ``A64`` under R3, where
    ``scored == Cq_struct``); ``CONTEXT_ONLY`` is ``context minus scored``
    (empty under R1, where ``context == Cq``).

    Requires ``cq subset scored subset context`` and raises if either
    containment fails -- this doubles as a wiring check for the regime the
    caller believes it built. Given both containments hold, the three
    returned arrays are pairwise disjoint and their union is exactly
    ``context``: ``cq union (scored \\ cq) == scored`` since ``cq subset
    scored``, and ``scored union (context \\ scored) == context`` since
    ``scored subset context``. No training, no thresholds, no gold label.
    """

    cq = np.unique(np.asarray(cq, dtype=np.int64))
    scored = np.unique(np.asarray(scored, dtype=np.int64))
    context = np.unique(np.asarray(context, dtype=np.int64))
    if not np.all(np.isin(cq, scored)):
        raise ValueError("cq must be a subset of scored")
    if not np.all(np.isin(scored, context)):
        raise ValueError("scored must be a subset of context")
    return {
        ROLE_RETRIEVAL_CANDIDATE: cq,
        ROLE_STRUCTURAL_SCORED_CANDIDATE: scored[~np.isin(scored, cq)],
        ROLE_CONTEXT_ONLY: context[~np.isin(context, scored)],
    }


def node_role_counts(roles: dict[str, np.ndarray]) -> dict[str, int]:
    """Per-role node counts for one (query, regime) cell, for distribution reporting."""

    return {role: int(np.asarray(roles[role]).size) for role in NODE_ROLES}


# --- R1/R2/R3 set invariants (M0B Safeguard B) ---


def regime_set_invariants(
    *,
    cq: np.ndarray,
    cq_struct: np.ndarray,
    a64: np.ndarray,
    universal_cap: int = 64,
) -> dict[str, object]:
    """R3-specific set invariants for one query, checked rather than assumed.

    ``scored_R1 == scored_R2`` is true by construction (both regimes score
    exactly ``Cq``, the same array) and ``oracle_R1 == oracle_R2`` bit-exact
    is the existing M0A.1 check -- neither is re-derived here. This function
    covers what is specific to R3's construction: ``Cq`` remains scoreable
    under R3 (``scored_R1 subset scored_R3``), the admission is disjoint from
    ``Cq`` and bounded by the universal cap, and ``Cq_struct`` is exactly the
    union with no separate, potentially-diverging union step.
    """

    cq = np.unique(np.asarray(cq, dtype=np.int64))
    cq_struct = np.unique(np.asarray(cq_struct, dtype=np.int64))
    a64 = np.unique(np.asarray(a64, dtype=np.int64))
    admitted_delta = np.setdiff1d(cq_struct, cq, assume_unique=True)
    return {
        "scored_r1_subset_scored_r3": bool(np.all(np.isin(cq, cq_struct))),
        "a64_disjoint_from_cq": bool(a64.size == 0 or not np.any(np.isin(a64, cq))),
        "admitted_delta_size": int(admitted_delta.size),
        "admitted_delta_within_universal_cap": bool(admitted_delta.size <= universal_cap),
        "cq_struct_equals_cq_union_a64": bool(
            np.array_equal(cq_struct, np.union1d(cq, a64))
        ),
    }
