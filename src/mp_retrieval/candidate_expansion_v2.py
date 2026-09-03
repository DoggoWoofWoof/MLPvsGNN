"""Parameter-free candidate expansion: a new scientific object, not a change.

The historical headroom layer measures ``oracle(Cq)`` over a pool that is fixed
by construction, and ``configs/candidate_headroom.yaml`` prohibits candidate
regeneration, admission and graph expansion in paper 1. That contract is not
edited or relaxed anywhere in this module. It governs a different object.

Regime R3 studies ``oracle(Cq')``, where regeneration is the declared
intervention rather than a violation of anything. A different object needs a
different namespace, so nothing here writes into a historical path, reuses a
historical output directory, or imports a historical module in a way that could
change its behaviour.

What is reconstructed
---------------------
CRAG level 1 (``docs/level1_unified_protocol.md`` in the read-only CRAG
reference) routes with "a shared MLP produces K relational offsets around the
nearest dense document". This repository's ``docs/OFFSET_OPERATOR_PROTOCOL.md``
carries the same idea as ``offset_mlp``: score ``cos(normalize(a + g(q, a)), x)``
with the anchor ``a`` fixed at dense rank 1 and ``g`` a small learned MLP.

Both are learned, and a learned candidate generator is forbidden here. The
shape is kept and the learned part is deleted:

* ``g(q, a)`` asks which direction to move from the anchor. The parameter-free
  residual that already exists in the data is ``r_q = normalize(e_q - x_a)``.
* The directions available are not a learned K. They are the real displacements
  the graph already contains, ``d(u, v) = normalize(x_v - x_u)``, one per edge.
* Compatibility is a cosine between the two. No parameters, no fitting, no
  temperature, no threshold learned from data.

The control, ``STRUCTURAL_NEIGHBOUR``, walks the identical frontier under the
identical caps and admits in ascending node id. It exists so that a movement in
the directional arm can be read against arbitrary bounded expansion of exactly
the same neighbourhood, rather than against no expansion at all.

What this module never reads
----------------------------
Gold nodes, gold relations, supporting facts, the split label, the dataset
identifier, or any test-time quantity. ``expand`` takes embeddings, topology,
seeds and an anchor, and its output is a pure function of those. The leakage
tests assert this by permuting golds and requiring byte-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-12
DIRECTIONAL = "L1_DIRECTIONAL"
STRUCTURAL = "STRUCTURAL_NEIGHBOUR"
EXPANSION_METHODS = (DIRECTIONAL, STRUCTURAL)


@dataclass(frozen=True)
class ExpansionBudget:
    """Every bound is declared before a result exists, in configs/m0a_probe.yaml.

    ``neighbour_scan_cap_per_seed`` bounds work on high-degree nodes. When it
    fires the expansion records it; a cap that is not reported reads as full
    coverage of the neighbourhood.
    """

    hop_cap: int = 1
    per_seed_cap: int = 16
    graph_expansion_cap: int = 128
    neighbour_scan_cap_per_seed: int = 4096

    def __post_init__(self) -> None:
        if self.hop_cap != 1:
            raise ValueError("M0A declares a hop cap of 1; a deeper walk needs a new declaration")
        for name in ("per_seed_cap", "graph_expansion_cap", "neighbour_scan_cap_per_seed"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Expansion:
    """One query's expansion under one method, with both pool rules realised."""

    method: str
    admitted: np.ndarray
    scores: np.ndarray
    matched_pool: np.ndarray
    additive_pool: np.ndarray
    evicted: np.ndarray
    protected: np.ndarray
    neighbours_scanned: int
    scan_cap_fired: bool
    degenerate_residual: bool
    zero_displacement_edges: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _unit(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalise, returning the unit rows and which rows had a usable norm."""

    rows = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=-1)
    usable = norms > EPS
    safe = np.where(usable, norms, 1.0)
    return rows / safe[..., None], usable


def query_residual(query_embedding: np.ndarray, anchor_embedding: np.ndarray) -> np.ndarray:
    """``normalize(e_q - x_a)``, the parameter-free stand-in for a learned offset.

    Returns a zero vector when the residual has no usable norm, which callers
    must treat as "no direction is defined for this query" rather than as a
    direction that happens to be zero.
    """

    residual = np.asarray(query_embedding, dtype=np.float64) - np.asarray(
        anchor_embedding, dtype=np.float64
    )
    norm = float(np.linalg.norm(residual))
    if norm <= EPS:
        return np.zeros_like(residual)
    return residual / norm


def _out_neighbours(rowptr: np.ndarray, col: np.ndarray, node: int) -> np.ndarray:
    start, end = int(rowptr[node]), int(rowptr[node + 1])
    return col[start:end]


def _rank(scores: np.ndarray, nodes: np.ndarray, limit: int) -> np.ndarray:
    """Indices of the best ``limit`` entries by score, ties by ascending node id.

    ``np.lexsort`` takes the last key as primary, so this reads bottom-up:
    order by descending score, break ties by ascending id.
    """

    order = np.lexsort((nodes, -scores))
    return order[:limit]


def _frontier_neighbours(
    *,
    rowptr: np.ndarray,
    col: np.ndarray,
    seeds: np.ndarray,
    in_pool: np.ndarray,
    budget: ExpansionBudget,
) -> tuple[list[tuple[int, np.ndarray]], int, bool]:
    """Per-seed candidate neighbours, already stripped of the pool and the seed.

    Returns one ``(seed, neighbours)`` pair per seed, the total number of
    neighbours scanned, and whether the scan cap fired on any seed. Neighbours
    are unique and ascending, so the cap always keeps the same prefix for the
    same graph.
    """

    per_seed: list[tuple[int, np.ndarray]] = []
    scanned = 0
    fired = False
    for seed in seeds:
        neighbours = np.unique(_out_neighbours(rowptr, col, int(seed)))
        if neighbours.size:
            neighbours = neighbours[~in_pool[neighbours]]
        if neighbours.size:
            neighbours = neighbours[neighbours != int(seed)]
        if neighbours.size > budget.neighbour_scan_cap_per_seed:
            fired = True
            neighbours = neighbours[: budget.neighbour_scan_cap_per_seed]
        scanned += int(neighbours.size)
        if neighbours.size:
            per_seed.append((int(seed), neighbours))
    return per_seed, scanned, fired


def _matched_budget(
    *,
    pool: np.ndarray,
    seeds: np.ndarray,
    anchor: int,
    admitted: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Admit into a pool of unchanged size by displacing its tail.

    ``pool`` is in frozen candidate order -- dense order first, then unseen
    SPLADE -- so its tail is the weakest evidence the frozen retrieval produced.
    Seeds and the anchor are protected from eviction: ``Sq`` is a subset of
    ``Cq`` by construction and every downstream feature assumes it, so evicting
    a seed would change the contract rather than test the expansion.

    Returns the new pool in frozen order with admitted nodes appended, the
    admitted subset actually used, its scores, the evicted tail, and the
    protected ids.
    """

    protected = np.unique(np.concatenate([seeds, np.array([anchor], dtype=np.int64)]))
    protected = protected[np.isin(protected, pool)]
    evictable = ~np.isin(pool, protected)
    room = int(evictable.sum())
    take = min(int(admitted.size), room)
    if take <= 0:
        return pool.copy(), admitted[:0], scores[:0], pool[:0], protected

    used, used_scores = admitted[:take], scores[:take]
    evictable_positions = np.flatnonzero(evictable)
    evicted_positions = evictable_positions[-take:]
    evicted = pool[evicted_positions]
    keep = np.ones(pool.size, dtype=bool)
    keep[evicted_positions] = False
    matched = np.concatenate([pool[keep], used])
    return matched, used, used_scores, evicted, protected


def expand(
    method: str,
    *,
    rowptr: np.ndarray,
    col: np.ndarray,
    node_embeddings: Any,
    query_embedding: np.ndarray | None,
    anchor: int,
    pool: np.ndarray,
    seeds: np.ndarray,
    budget: ExpansionBudget,
    num_nodes: int,
) -> Expansion:
    """One query's parameter-free expansion. A pure function of its arguments.

    ``pool`` must be in frozen candidate order; it is not sorted here, because
    the matched-budget rule reads that order to decide what the tail is.
    """

    if method not in EXPANSION_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {EXPANSION_METHODS}")
    pool = np.asarray(pool, dtype=np.int64)
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    in_pool = np.zeros(num_nodes, dtype=bool)
    in_pool[pool] = True

    per_seed, scanned, fired = _frontier_neighbours(
        rowptr=rowptr, col=col, seeds=seeds, in_pool=in_pool, budget=budget
    )

    degenerate = False
    zero_edges = 0
    nodes_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []

    if method == DIRECTIONAL:
        residual = query_residual(query_embedding, node_embeddings[int(anchor)])
        degenerate = bool(np.all(residual == 0.0))
        if not degenerate:
            for seed, neighbours in per_seed:
                source = np.asarray(node_embeddings[seed], dtype=np.float64)
                targets = np.asarray(node_embeddings[neighbours], dtype=np.float64)
                directions, usable = _unit(targets - source)
                zero_edges += int((~usable).sum())
                scores = directions @ residual
                scores = np.where(usable, scores, -np.inf)
                keep = _rank(scores, neighbours, budget.per_seed_cap)
                nodes_parts.append(neighbours[keep])
                score_parts.append(scores[keep])
    else:
        for seed, neighbours in per_seed:
            keep = neighbours[: budget.per_seed_cap]
            nodes_parts.append(keep)
            score_parts.append(np.zeros(keep.size, dtype=np.float64))

    if nodes_parts:
        nodes = np.concatenate(nodes_parts)
        scores = np.concatenate(score_parts)
        # A node reachable from several seeds keeps its best score, once.
        unique_nodes, inverse = np.unique(nodes, return_inverse=True)
        best = np.full(unique_nodes.size, -np.inf, dtype=np.float64)
        np.maximum.at(best, inverse, scores)
        finite = np.isfinite(best)
        unique_nodes, best = unique_nodes[finite], best[finite]
        keep = _rank(best, unique_nodes, budget.graph_expansion_cap)
        admitted, admitted_scores = unique_nodes[keep], best[keep]
    else:
        admitted = np.zeros(0, dtype=np.int64)
        admitted_scores = np.zeros(0, dtype=np.float64)

    matched, used, used_scores, evicted, protected = _matched_budget(
        pool=pool, seeds=seeds, anchor=int(anchor), admitted=admitted, scores=admitted_scores
    )
    additive = np.concatenate([pool, admitted]) if admitted.size else pool.copy()

    diagnostics: dict[str, Any] = {
        "admitted_used_under_matched_budget": int(used.size),
        "admitted_scores_used": used_scores,
        "seeds_with_neighbours": len(per_seed),
    }
    return Expansion(
        method=method,
        admitted=admitted,
        scores=admitted_scores,
        matched_pool=matched,
        additive_pool=additive,
        evicted=evicted,
        protected=protected,
        neighbours_scanned=scanned,
        scan_cap_fired=fired,
        degenerate_residual=degenerate,
        zero_displacement_edges=zero_edges,
        diagnostics=diagnostics,
    )
