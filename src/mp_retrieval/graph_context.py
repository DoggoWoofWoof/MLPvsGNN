"""Query-local graph context: which nodes may contribute structural information.

Phase -1 established that the historical construction -- semantic top-K to a
candidate set ``Cq``, then the strict vertex-induced graph ``G[Cq]`` -- deletes
most of each candidate's global neighbourhood (median retention 7.7%-25.0%,
boundary cut 70.0%-90.4%) and most seed-relative multi-hop reach (13.6%-79.1% of
hop-2 reach survives). Both QLS-v1 and the historical one-layer GNN read that
graph.

This module builds candidate *contexts*: supersets ``Uq`` of ``Cq`` drawn from
the true global graph. It never changes what is scored. Two separate objects:

    Cq   the scoring universe -- the frozen Dense/SPLADE candidate set
    Uq   the context -- nodes allowed to contribute structural information

``scored_nodes == Cq`` in every arm, so the candidate ceiling is identical
across arms and none of this is a candidate-generation change.

Naive expansion is not a viable universal definition of "local context": Phase
-1 measured ``Cq u N2(Cq)`` at 93.2% of the hotpotqa graph and ``Cq u N3(Cq)``
at 100.0% of it. Those arms are excluded on that measurement, not on taste.

Three facts about the sealed graphs shape every definition below.

**Five of the six graphs are reachability-symmetric.** On 2wiki the 855,146
stored edges are 521,614 distinct pairs and every pair carries its reverse. So
in-neighbours and out-neighbours coincide, and a bridge rule of the form "``v``
is reachable from a seed *and* has an edge into ``Cq``" is satisfied
automatically -- ``Sq`` is a subset of ``Cq``, so the seed that reached ``v`` is
itself the candidate ``v`` points back at. A first version of ``PATH_H2`` used
exactly that rule and returned a set identical to ``SEED_H1`` on every query. The
path arms therefore require a genuine two-path ``c1 -> v -> c2`` with
``c1 != c2``: a node that reconnects two *different* members of the scored set
is what induction actually destroyed.

hotpotqa's sealed graph is the exception, and the audit measured the gap rather
than assuming it away: its directed message-flow receptive field is strictly
smaller than its symmetrised one at every hop (R2 median 18.36 against 18.77, R3
32.79 against 33.12). Direction is therefore kept explicit throughout. Repairing
what a one-layer GNN lost means restoring *in*-neighbours of ``Cq``; asking what
a seed reaches means *out*-neighbours of ``Sq``. On five datasets the two agree;
on hotpotqa they do not, and the arms must not quietly pick the wrong one.

**Edges are stored with multiplicity, and hotpotqa stores self-loops.**
34.1%-53.6% of stored messages are duplicates, and hotpotqa's sealed graph is
the only one of the 24 audited graph-splits carrying any stored self-loop --
1.55 per candidate against a mean global out-degree of 21.19. Counting "distinct
candidate neighbours" over the raw arrays would count one neighbour several
times and would let a self-loop pose as a bridge, so the path arms run on a
deduplicated, self-loop-free view built once per split.

**Retention has one definition here, and it is Phase -1's.** A stored self-loop
is not context from a neighbour, so it leaves both the numerator and the
denominator; stored multiplicity stays on both sides. ``candidate_structure``
below computes this by calling the audit's own ``induced_view``, so the ``CAND``
arm reproduces the audited numbers rather than resembling them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

__all__ = [
    "ARMS",
    "SERVING_EXCLUDED",
    "DISTANCE_BUCKETS",
    "Operators",
    "build_operators",
    "context_nodes",
    "induced_edges",
    "candidate_structure",
    "seed_distance",
    "SEED_DISTANCE_HOPS",
    "context_report",
    "qls_local_features",
]

#: Arms carried into the pilot. ``CAND`` is the historical control.
ARMS = ("CAND", "SEED_H1", "TARGET_H1", "SEED_H2", "PATH_H2", "BRIDGE_H2")

#: Excluded before any pilot compute, on Phase -1's measured corpus share.
SERVING_EXCLUDED = {
    "SEED_H3": "33.0%-99.7% of the corpus graph",
    "TARGET_H2": "17.9%-93.2% of the corpus graph",
    "TARGET_H3": "72.5%-100.0% of the corpus graph",
}

#: Columns 0-3 of the frozen QLS-v1 local descriptor: a one-hot seed-distance
#: bucket over {0, 1, 2, >=3-or-unreachable}. The only unnormalised columns, so
#: the movement signal no rescaling can manufacture.
DISTANCE_BUCKETS = slice(0, 4)


@dataclass(frozen=True)
class Operators:
    """Traversal operators for one graph, built once per split.

    ``forward[i, j] = 1`` per stored edge ``i -> j``, so ``forward @ 1_X`` is
    nonzero on nodes with an edge *into* ``X`` and ``reverse @ 1_X`` is nonzero
    on its out-neighbours. The ``simple_*`` pair is the same graph with
    duplicate edges and self-loops removed, which is what distinct-neighbour
    counting requires.

    ``edge_source[k]`` is the row that stored edge ``k`` belongs to -- the CSR
    row index expanded, which ``induced_edges`` needs and which depends on the
    graph alone. Building these per query instead of per split would cost more
    than the traversals they serve: MEASURED on webqsp's 13,379,166-edge CSR,
    expanding it costs 122-362 ms, against 0.4-267 ms for every context this
    module builds. That is the lesson the substrate audit already paid for.
    """

    forward: sp.csr_matrix
    reverse: sp.csr_matrix
    simple_forward: sp.csr_matrix
    simple_reverse: sp.csr_matrix
    edge_source: np.ndarray
    size: int


def build_operators(rowptr, col, size) -> Operators:
    rowptr = np.asarray(rowptr, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    forward = sp.csr_matrix(
        (np.ones(col.size, dtype=np.int64), col, rowptr), shape=(size, size)
    )
    degrees = rowptr[1:] - rowptr[:-1]
    src = np.repeat(np.arange(size, dtype=np.int64), degrees)
    keep = src != col
    pairs = np.unique(np.stack([src[keep], col[keep]]), axis=1)
    simple = sp.csr_matrix(
        (np.ones(pairs.shape[1], dtype=np.int64), (pairs[0], pairs[1])),
        shape=(size, size),
    )
    return Operators(
        forward=forward,
        reverse=forward.T.tocsr(),
        simple_forward=simple,
        simple_reverse=simple.T.tocsr(),
        edge_source=src,
        size=int(size),
    )


def _indicator(nodes, size, dtype=np.int64):
    vector = np.zeros(size, dtype=dtype)
    if nodes.size:
        vector[nodes] = 1
    return vector


def _step(matrix, nodes, size):
    """One traversal step.

    ``int64`` throughout: a ``uint8`` accumulator wraps at 256 incoming frontier
    edges and then reports a reached node as unreached, silently.
    """
    if nodes.size == 0:
        return np.zeros(size, dtype=bool)
    return matrix.dot(_indicator(nodes, size)) != 0


def _reach(matrix, sources, size, hops):
    """Nodes within ``hops`` steps of ``sources``, following ``matrix``'s direction."""
    seen = np.zeros(size, dtype=bool)
    if sources.size == 0:
        return seen
    seen[sources] = True
    frontier = sources
    for _ in range(hops):
        fresh = _step(matrix, frontier, size) & ~seen
        if not fresh.any():
            break
        seen |= fresh
        frontier = np.flatnonzero(fresh)
    return seen


def _distinct_two_paths(operators, anchors, pool):
    """Nodes carrying ``a -> v -> c`` for some ``a`` in ``anchors``, ``c`` in
    ``pool``, with ``a != c``.

    The distinctness requirement is the whole point. Without it the rule is
    vacuous on a symmetric graph: ``v`` adjacent to any candidate satisfies both
    halves through that one candidate, and the arm returns the plain
    neighbourhood. Counting runs on the deduplicated view so a parallel edge
    cannot pose as a second endpoint.

    Identity is resolved without enumerating pairs: when exactly one anchor and
    exactly one candidate are involved, their ``id + 1`` sums are equal if and
    only if they are the same node.
    """
    size = operators.size
    anchor_mark = _indicator(anchors, size)
    pool_mark = _indicator(pool, size)
    anchor_id = np.zeros(size, dtype=np.int64)
    anchor_id[anchors] = anchors + 1
    pool_id = np.zeros(size, dtype=np.int64)
    pool_id[pool] = pool + 1

    incoming = operators.simple_reverse.dot(anchor_mark)  # anchors a with a -> v
    outgoing = operators.simple_forward.dot(pool_mark)  # candidates c with v -> c
    incoming_id = operators.simple_reverse.dot(anchor_id)
    outgoing_id = operators.simple_forward.dot(pool_id)

    both = (incoming >= 1) & (outgoing >= 1)
    same_single_node = (incoming == 1) & (outgoing == 1) & (incoming_id == outgoing_id)
    return both & ~same_single_node


def context_nodes(arm, *, operators, pool, seeds):
    """Global node ids of the context ``Uq`` for one query under one arm.

    ``pool`` is ``Cq`` and ``seeds`` is ``Sq``, both as global ids. ``Sq`` is a
    subset of ``Cq`` by construction of the artifact -- the audit computes
    ``seed_global = pool[seed_local]`` -- so ``Cq u Sq == Cq`` and the seed arms
    are not a union of two independent sets. They differ from the target arms
    only in what they expand *from*: roughly ten seeds against 316-361
    candidates.

    Returned ids are sorted, unique, and always contain ``pool``.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    size = operators.size
    pool = np.unique(np.asarray(pool, dtype=np.int64))
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    in_pool = np.zeros(size, dtype=bool)
    in_pool[pool] = True

    if arm == "CAND":
        return pool
    if arm == "SEED_H1":
        extra = _step(operators.reverse, seeds, size)
    elif arm == "SEED_H2":
        extra = _reach(operators.reverse, seeds, size, 2)
    elif arm == "TARGET_H1":
        # In-neighbours: exactly the messages a one-layer GNN lost to induction.
        extra = _step(operators.forward, pool, size)
    elif arm == "PATH_H2":
        extra = _distinct_two_paths(operators, seeds, pool)
    elif arm == "BRIDGE_H2":
        extra = _distinct_two_paths(operators, pool, pool)
    else:  # pragma: no cover - ARMS is exhaustive above
        raise AssertionError(arm)

    return np.union1d(pool, np.flatnonzero(extra & ~in_pool))

#: How far the seed-distance probe looks. The frozen descriptor's own bucket
#: stops at 2 and lumps everything beyond into one class, which is exactly the
#: resolution this exists to recover; four hops is one past Phase -1's deepest
#: measured reach statistic and keeps the traversal bounded.
SEED_DISTANCE_HOPS = 4


def seed_distance(operators, nodes, pool, seeds, *, max_hops=SEED_DISTANCE_HOPS):
    """Hops from ``Sq`` to each candidate, travelling only through ``Uq``.

    Returns ``max_hops + 1`` for a candidate no path of that length reaches, so
    the array is finite and orderable; seeds are 0 because ``Sq`` is inside
    ``Cq``.

    This exists because the frozen QLS-v1 descriptor cannot answer the question
    Stage B asks. MEASURED over 480 comparisons on 120 random graphs: no context
    wider than ``SEED_H1`` moves one of its distance buckets, and none changes a
    candidate's raw seed-incidence count. That is a structural identity rather
    than a property of those graphs -- the bucket is one-hot over
    ``{0, 1, 2, >=3-or-unreachable}``, a candidate reaches bucket 2 exactly when
    some ``seed -> x -> candidate`` path exists, and every such ``x`` is an
    out-neighbour of a seed, which is what ``SEED_H1`` is. Its six remaining
    columns are each divided by a maximum over the whole local node space, so
    they move when the context widens whether or not the candidate's own
    topology did.

    Travel is restricted to the context by intersecting each frontier with it,
    which is the definition of a context: an arm may only use the nodes it
    admitted.
    """
    import numpy as _np

    size = operators.size
    pool = _np.unique(_np.asarray(pool, dtype=_np.int64))
    seeds = _np.unique(_np.asarray(seeds, dtype=_np.int64))
    in_context = _np.zeros(size, dtype=bool)
    in_context[_np.asarray(nodes, dtype=_np.int64)] = True

    distance = _np.full(size, max_hops + 1, dtype=_np.int64)
    if seeds.size == 0:
        return distance[pool]
    distance[seeds] = 0
    seen = _np.zeros(size, dtype=bool)
    seen[seeds] = True
    frontier = seeds
    for hop in range(1, max_hops + 1):
        fresh = _step(operators.reverse, frontier, size) & in_context & ~seen
        if not fresh.any():
            break
        distance[fresh] = hop
        seen |= fresh
        frontier = _np.flatnonzero(fresh)
    return distance[pool]



def induced_edges(rowptr, col, nodes, size, *, edge_source=None):
    """Stored edges of ``G`` with both endpoints in ``nodes``, as ``(src, dst)``.

    Edges come from the global graph and are never invented; an arm that
    produced an edge absent from ``G`` would be measuring a graph nobody has.

    ``edge_source`` is ``Operators.edge_source``. Pass it: without it this
    expands the whole graph's CSR rows on every call, which MEASURED 122-362 ms
    per call on webqsp regardless of how small the context is.
    """
    keep = np.zeros(size, dtype=bool)
    keep[nodes] = True
    if edge_source is None:
        degrees = rowptr[1:] - rowptr[:-1]
        edge_source = np.repeat(np.arange(size, dtype=np.int64), degrees)
    mask = keep[edge_source] & keep[col]
    return edge_source[mask], col[mask].astype(np.int64, copy=False)


def candidate_structure(rowptr, col, nodes, pool, size):
    """Phase -1's structural quantities for ``Cq``, generalised to a context.

    Returns, per candidate, how much of its true neighbourhood survives inside
    ``Uq`` and whether it has any neighbour there at all. Three conventions are
    inherited from the audit rather than re-chosen, because a Stage-B number that
    is not comparable to Phase -1 answers no question:

    *   the denominator is the candidate's stored out-degree with stored
        self-loops removed and stored multiplicity kept;
    *   the numerator counts stored non-self out-messages landing in ``Uq``,
        again with multiplicity;
    *   isolation is the *symmetrised distinct* one-hop notion -- a candidate is
        isolated when it has no undirected non-self neighbour inside ``Uq`` --
        which is what the audit's ``isolated_fraction`` reports.

    They are inherited by *calling* ``induced_view``, the function the audit
    calls, so with ``nodes == pool`` this is the audit's own computation and the
    ``CAND`` arm is exact by construction rather than by agreement.
    ``tests/test_graph_context.py`` pins that against ``retention_summary`` and
    ``connectivity_summary``.

    Only ``induced_view``'s per-node arrays are used. Its ``boundary_edges`` is
    a whole-context total, and the boundary cut wanted here is the candidates',
    so that one is recomputed from the candidate rows.
    """
    from .graph_substrate import _undirected_adjacency, induced_view

    nodes = np.asarray(nodes, dtype=np.int64)
    pool = np.unique(np.asarray(pool, dtype=np.int64))
    counts = induced_view(rowptr, col, nodes)
    position = np.searchsorted(nodes, pool)

    denominator = (counts.global_degree - counts.self_loops_per_node).clip(min=0)
    denominator = denominator[position].astype(np.float64)
    kept = counts.induced_out_degree[position].astype(np.float64)
    measurable = denominator > 0
    retention = np.zeros(pool.size, dtype=np.float64)
    retention[measurable] = kept[measurable] / denominator[measurable]

    undirected_rowptr, _ = _undirected_adjacency(counts.edges, int(nodes.size))
    undirected_degree = (undirected_rowptr[1:] - undirected_rowptr[:-1])[position]

    return {
        "retention": retention,
        "measurable": measurable,
        "kept_messages": kept,
        "incident_messages": denominator,
        "isolated": undirected_degree == 0,
        "induced_degree": undirected_degree.astype(np.int64),
    }


def context_report(arm, *, rowptr, col, operators, pool, seeds):
    """Size and structural-recovery numbers for one query under one arm.

    ``total_ratio`` and ``added_ratio`` are reported separately and never
    interchangeably: ``|Uq| / |Cq|`` is bounded below by 1 because ``Uq``
    contains ``Cq``, while ``(|Uq| - |Cq|) / |Cq|`` starts at 0. Confusing the
    two once understated a 472k-node context as "0.2x the pool".

    Retention, isolation and boundary cut come from ``candidate_structure``,
    which is Phase -1's definition, so ``CAND`` reproduces the audited tables.
    """
    size = operators.size
    pool = np.unique(np.asarray(pool, dtype=np.int64))
    nodes = context_nodes(arm, operators=operators, pool=pool, seeds=seeds)
    src, _ = induced_edges(
        rowptr, col, nodes, size, edge_source=operators.edge_source
    )

    structure = candidate_structure(rowptr, col, nodes, pool, size)
    retention = structure["retention"]
    measurable = structure["measurable"]
    kept = structure["kept_messages"]
    incident = structure["incident_messages"]

    def _stat(function, values):
        return float(function(values)) if values.size else float("nan")

    return {
        "arm": arm,
        "candidates": int(pool.size),
        "context_nodes": int(nodes.size),
        "added_nodes": int(nodes.size - pool.size),
        "total_ratio": float(nodes.size / pool.size) if pool.size else float("nan"),
        "added_ratio": (
            float((nodes.size - pool.size) / pool.size) if pool.size else float("nan")
        ),
        "graph_share": float(nodes.size / size) if size else float("nan"),
        "context_edges": int(src.size),
        "candidate_isolated_fraction": _stat(
            np.mean, structure["isolated"].astype(float)
        ),
        "retention_mean": _stat(np.mean, retention[measurable]),
        "retention_median": _stat(np.median, retention[measurable]),
        "boundary_cut": (
            float(1.0 - kept.sum() / incident.sum())
            if incident.sum()
            else float("nan")
        ),
    }


def qls_local_features(
    *,
    rowptr,
    col,
    nodes,
    pool,
    seeds,
    size,
    damping=0.85,
    ppr_iterations=8,
    edge_source=None,
):
    """Frozen QLS-v1 query-local descriptors over ``nodes``, read off ``pool``.

    Reported, not relied on. See ``seed_distance`` for why: every column of this
    descriptor is either capped at seed-distance 2 -- which ``SEED_H1`` already
    saturates -- or divided by a per-query maximum over the whole local node
    space. There is no column in which "the arm restored structure" and "the
    normaliser moved" can be told apart.

    This calls the *shipped* feature kernel rather than reimplementing it, so an
    arm cannot appear to move features merely because a second implementation
    disagrees with the first. What changes between arms is the node space the
    kernel runs on: ``CAND`` reproduces the historical
    ``frozen_candidate_induced_directed_graph`` exactly, and every other arm
    lets seed signal travel through the noncandidate bridges induction deleted.
    Only the ``pool`` rows are returned -- context nodes are never scored.

    One caveat, stated because it would otherwise read as signal: six of the ten
    columns are normalised by a per-query maximum taken over the whole local node
    space, so widening the context can rescale a candidate whose own topology did
    not change. ``DISTANCE_BUCKETS`` is free of that confound and is the primary
    movement measure for exactly that reason.
    """
    from .structural_features import _local_feature_chunk

    nodes = np.asarray(nodes, dtype=np.int64)
    pool = np.unique(np.asarray(pool, dtype=np.int64))
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    src, dst = induced_edges(rowptr, col, nodes, size, edge_source=edge_source)
    edges = np.stack(
        [np.searchsorted(nodes, src), np.searchsorted(nodes, dst)]
    ).astype(np.int64)
    features = _local_feature_chunk(
        0,
        1,
        np.array([0, nodes.size], dtype=np.int64),
        np.array([0, seeds.size], dtype=np.int64),
        np.searchsorted(nodes, seeds).astype(np.int64),
        np.array([0, edges.shape[1]], dtype=np.int64),
        np.array([0], dtype=np.int64),
        edges,
        float(damping),
        int(ppr_iterations),
    )
    return np.asarray(features)[np.searchsorted(nodes, pool)]
