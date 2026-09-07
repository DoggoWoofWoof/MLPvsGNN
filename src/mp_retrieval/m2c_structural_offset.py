"""M2C: deterministic structural displacement geometry. No learned parameters.

What is new here, and what is not
---------------------------------
:mod:`mp_retrieval.candidate_expansion_v2` already implements a parameter-free
directional expansion: a query residual, real edge displacements
``normalize(x_v - x_u)``, a cosine compatibility, per-seed top-k, and a
matched-budget admission rule. M0A measured it against the trivial
lowest-node-id control and found the two **indistinguishable in every one of
nine family cells on three datasets**, including cells where they admitted
almost disjoint sets. That null is not re-litigated by copying the mechanism.

Exactly one thing differs here, and it is the reason this module exists.

``candidate_expansion_v2.query_residual`` is ``normalize(e_q - x_a)`` for a
**single** anchor at dense rank 1. It discards nine of the ten retrieval seeds
and makes the direction a function of whichever document dense retrieval
happened to put first. This module computes instead

    r_q = q - P_span(E_q) q

the component of the query orthogonal to the span of **all** the retrieval
seeds -- "what no seed already explains".

The two are not nested: ``q - x_a`` subtracts the anchor *vector*, while
``q - P_span{x_a} q`` subtracts only the component of ``q`` along it, and they
agree only in the degenerate case where the anchor already equals that
component. So the single-anchor form is a genuinely different residual rather
than a special case, and :func:`single_anchor_residual` is provided so a probe
can **measure** how far apart the two point instead of assuming the change
matters.

The second thing this module does that the existing one does not is **ranking**.
``query_residual`` appears in exactly one module and one test in this
repository, both about admission; directional compatibility has never been used
to order candidates. :func:`directional_statistics` and
:func:`direction_prototype` exist for that.

Leakage
-------
Nothing here reads gold nodes, gold relations, supporting facts, split labels
or dataset identifiers. Every function is a pure function of embeddings,
topology and seeds, which is what makes the permutation test in
``tests/test_m2c_structural_offset.py`` meaningful rather than decorative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Absolute floor for "this vector has a usable norm". Shared with
#: candidate_expansion_v2 so the two modules agree about degeneracy.
EPS = 1e-12

#: Relative singular-value tolerance for deciding the numerical rank of the
#: seed matrix. Singular values below ``RANK_RTOL * s_max`` are dropped, so a
#: near-collinear seed does not contribute a direction that is mostly rounding
#: error. Declared as a constant rather than passed at every call site, because
#: a tolerance that varies per caller is a tuning knob in disguise.
RANK_RTOL = 1e-8

#: Below this ratio of ``||r|| / ||q||`` the residual is treated as degenerate:
#: the query lies (numerically) inside the span of its own seeds, so "what the
#: seeds do not explain" has no reliable direction.
RESIDUAL_RELATIVE_FLOOR = 1e-8


@dataclass(frozen=True)
class Residual:
    """One query's residual, with everything needed to audit how it was made."""

    vector: np.ndarray
    seed_count: int
    subspace_rank: int
    residual_norm: float
    query_norm: float
    relative_norm: float
    degenerate: bool
    fell_back_to_query: bool

    def diagnostics(self) -> dict[str, Any]:
        return {
            "seed_count": self.seed_count,
            "subspace_rank": self.subspace_rank,
            "residual_norm": self.residual_norm,
            "query_norm": self.query_norm,
            "relative_norm": self.relative_norm,
            "degenerate_residual": self.degenerate,
            "fell_back_to_query": self.fell_back_to_query,
        }


def _unit(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalise, returning unit rows and which rows had a usable norm.

    Rows without a usable norm are returned as zeros and flagged, never as a
    direction that happens to point somewhere arbitrary.
    """

    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    norms = np.linalg.norm(rows, axis=-1)
    usable = norms > EPS
    safe = np.where(usable, norms, 1.0)
    return rows / safe[..., None], usable


def seed_subspace_residual(
    query_embedding: np.ndarray,
    seed_embeddings: np.ndarray,
    *,
    rank_rtol: float = RANK_RTOL,
) -> Residual:
    """``normalize(q - P_span(E) q)`` -- what the retrieval seeds do not explain.

    The projector is built from a rank-revealing SVD of the seed matrix rather
    than from normal equations: ``E (E^T E)^-1 E^T`` is exactly the formulation
    that fails when two seeds are near-collinear, which is the common case here
    because seeds are retrieved by similarity to one query and are therefore
    similar to each other.

    Handles, and records: zero seeds, one seed, collinear seeds, rank-deficient
    ``E``, and a query that lies inside the seed span.
    """

    query = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
    query_norm = float(np.linalg.norm(query))

    seeds = np.asarray(seed_embeddings, dtype=np.float64)
    if seeds.size == 0:
        seeds = np.zeros((0, query.shape[0]), dtype=np.float64)
    if seeds.ndim == 1:
        seeds = seeds[None, :]
    if seeds.shape[-1] != query.shape[0]:
        raise ValueError(
            f"seed width {seeds.shape[-1]} does not match query width {query.shape[0]}"
        )

    # Drop unusable seed rows before they can define a spurious basis direction.
    _, usable = _unit(seeds)
    seeds = seeds[usable] if seeds.shape[0] else seeds

    if query_norm <= EPS:
        # No query direction at all. There is nothing to be orthogonal to.
        return Residual(
            vector=np.zeros_like(query),
            seed_count=int(seeds.shape[0]),
            subspace_rank=0,
            residual_norm=0.0,
            query_norm=query_norm,
            relative_norm=0.0,
            degenerate=True,
            fell_back_to_query=False,
        )

    if seeds.shape[0] == 0:
        # Zero seeds: the projection is onto the trivial subspace, so the
        # residual is the query itself. This is not the degenerate case -- the
        # direction is well defined, there is simply nothing subtracted.
        unit_query = query / query_norm
        return Residual(
            vector=unit_query,
            seed_count=0,
            subspace_rank=0,
            residual_norm=query_norm,
            query_norm=query_norm,
            relative_norm=1.0,
            degenerate=False,
            fell_back_to_query=False,
        )

    # Columns of ``basis`` span the seed subspace. full_matrices=False keeps
    # this at (dim, m) rather than (dim, dim) -- m is at most ten.
    left, singular, _ = np.linalg.svd(seeds.T, full_matrices=False)
    if singular.size and singular[0] > EPS:
        keep = singular > (rank_rtol * float(singular[0]))
    else:
        keep = np.zeros(singular.shape, dtype=bool)
    rank = int(keep.sum())
    basis = left[:, keep]

    projected = basis @ (basis.T @ query) if rank else np.zeros_like(query)
    residual = query - projected
    residual_norm = float(np.linalg.norm(residual))
    relative = residual_norm / query_norm

    if relative <= RESIDUAL_RELATIVE_FLOOR or residual_norm <= EPS:
        # The query lies inside the span of its own seeds. Fall back to the
        # normalized query, and say so on the row: a fallback that is not
        # recorded is a silent second method.
        return Residual(
            vector=query / query_norm,
            seed_count=int(seeds.shape[0]),
            subspace_rank=rank,
            residual_norm=residual_norm,
            query_norm=query_norm,
            relative_norm=relative,
            degenerate=True,
            fell_back_to_query=True,
        )

    return Residual(
        vector=residual / residual_norm,
        seed_count=int(seeds.shape[0]),
        subspace_rank=rank,
        residual_norm=residual_norm,
        query_norm=query_norm,
        relative_norm=relative,
        degenerate=False,
        fell_back_to_query=False,
    )


def single_anchor_residual(
    query_embedding: np.ndarray, anchor_embedding: np.ndarray
) -> Residual:
    """The existing repository form, wrapped in the same record type.

    ``normalize(e_q - x_a)``. Present so a probe can measure how far the
    subspace residual actually moves from the residual M0A used, rather than
    assuming the change matters. If the two pick the same directions, M0A's
    null transfers and there is nothing here worth training on.
    """

    query = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
    anchor = np.asarray(anchor_embedding, dtype=np.float64).reshape(-1)
    if anchor.shape != query.shape:
        raise ValueError(f"anchor width {anchor.shape} does not match query {query.shape}")
    residual = query - anchor
    norm = float(np.linalg.norm(residual))
    query_norm = float(np.linalg.norm(query))
    if norm <= EPS:
        return Residual(
            vector=np.zeros_like(query),
            seed_count=1,
            subspace_rank=1,
            residual_norm=norm,
            query_norm=query_norm,
            relative_norm=0.0,
            degenerate=True,
            fell_back_to_query=False,
        )
    return Residual(
        vector=residual / norm,
        seed_count=1,
        subspace_rank=1,
        residual_norm=norm,
        query_norm=query_norm,
        relative_norm=norm / query_norm if query_norm > EPS else 0.0,
        degenerate=False,
        fell_back_to_query=False,
    )


def displacements(
    seed_embedding: np.ndarray, neighbour_embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``delta(s, v) = normalize(e_v - e_s)`` for one seed and its neighbours.

    Identical in definition to the displacement ``candidate_expansion_v2`` uses
    for admission; shared definition, different consumer.
    """

    source = np.asarray(seed_embedding, dtype=np.float64).reshape(-1)
    targets = np.asarray(neighbour_embeddings, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[None, :]
    if targets.shape[-1] != source.shape[0]:
        raise ValueError("neighbour width does not match seed width")
    return _unit(targets - source)


def compatibility(residual: np.ndarray, unit_deltas: np.ndarray) -> np.ndarray:
    """``c(s, v) = cosine(r_q, delta(s, v))``, given already-unit inputs."""

    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    unit_deltas = np.asarray(unit_deltas, dtype=np.float64)
    if unit_deltas.ndim == 1:
        unit_deltas = unit_deltas[None, :]
    return unit_deltas @ residual


def directional_statistics(
    scores: np.ndarray, usable: np.ndarray | None = None
) -> dict[str, float]:
    """The four deterministic statistics Stage 0 audits for one candidate.

    Returned for every candidate, including candidates with no graph-supported
    direction -- those get ``dir_support_count == 0`` and neutral scores, so a
    downstream model can tell "no structural evidence" from "evidence pointing
    the wrong way". Collapsing those two states is how a structural feature
    quietly becomes a proxy for node degree.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if usable is None:
        usable = np.isfinite(scores)
    else:
        usable = np.asarray(usable, dtype=bool).reshape(-1) & np.isfinite(scores)
    eligible = scores[usable]
    if eligible.size == 0:
        return {
            "dir_max": 0.0,
            "dir_mean": 0.0,
            "dir_positive_mass": 0.0,
            "dir_support_count": 0.0,
        }
    positive = eligible[eligible > 0.0]
    return {
        "dir_max": float(eligible.max()),
        "dir_mean": float(eligible.mean()),
        "dir_positive_mass": float(positive.sum() / eligible.size),
        "dir_support_count": float(eligible.size),
    }


def direction_prototype(
    unit_deltas: np.ndarray,
    scores: np.ndarray,
    *,
    tau: float,
    usable: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``delta_star = normalize(sum_i softmax(tau * c_i) delta_i)``.

    ``tau`` is a declared constant, never fitted here and never chosen by
    looking at a target metric. The softmax is computed with the usual max
    shift so a large ``tau`` cannot overflow.
    """

    unit_deltas = np.asarray(unit_deltas, dtype=np.float64)
    if unit_deltas.ndim == 1:
        unit_deltas = unit_deltas[None, :]
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.isfinite(scores)
    if usable is not None:
        mask &= np.asarray(usable, dtype=bool).reshape(-1)
    if not mask.any():
        return np.zeros(unit_deltas.shape[-1], dtype=np.float64), {
            "eligible_directions": 0,
            "degenerate_prototype": True,
        }

    eligible_deltas, eligible_scores = unit_deltas[mask], scores[mask]
    shifted = tau * eligible_scores
    weights = np.exp(shifted - shifted.max())
    weights /= weights.sum()
    combined = weights @ eligible_deltas
    norm = float(np.linalg.norm(combined))
    if norm <= EPS:
        # Directions that cancel. Real: opposed displacements with near-equal
        # weight. There is no prototype, and inventing one would fabricate a
        # direction the geometry does not contain.
        return np.zeros(unit_deltas.shape[-1], dtype=np.float64), {
            "eligible_directions": int(eligible_scores.size),
            "degenerate_prototype": True,
        }
    return combined / norm, {
        "eligible_directions": int(eligible_scores.size),
        "degenerate_prototype": False,
        "weight_max": float(weights.max()),
        "weight_entropy": float(-(weights * np.log(weights + EPS)).sum()),
    }


def mean_direction_prototype(
    unit_deltas: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
    usable: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The no-temperature control: ``normalize(mean(top-k compatible deltas))``.

    Exists so that no phase result rests on ``tau``. If the weighted and
    unweighted prototypes behave the same, the temperature was never doing
    work and should not be reported as though it were.
    """

    unit_deltas = np.asarray(unit_deltas, dtype=np.float64)
    if unit_deltas.ndim == 1:
        unit_deltas = unit_deltas[None, :]
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.isfinite(scores)
    if usable is not None:
        mask &= np.asarray(usable, dtype=bool).reshape(-1)
    if not mask.any() or top_k <= 0:
        return np.zeros(unit_deltas.shape[-1], dtype=np.float64), {
            "eligible_directions": 0,
            "degenerate_prototype": True,
        }
    eligible_deltas, eligible_scores = unit_deltas[mask], scores[mask]
    take = min(int(top_k), eligible_scores.size)
    # Descending score, ties broken by ascending original position, matching
    # candidate_expansion_v2._rank so the two modules order ties the same way.
    order = np.lexsort((np.arange(eligible_scores.size), -eligible_scores))[:take]
    combined = eligible_deltas[order].mean(axis=0)
    norm = float(np.linalg.norm(combined))
    if norm <= EPS:
        return np.zeros(unit_deltas.shape[-1], dtype=np.float64), {
            "eligible_directions": int(eligible_scores.size),
            "degenerate_prototype": True,
        }
    return combined / norm, {
        "eligible_directions": int(eligible_scores.size),
        "used_directions": int(take),
        "degenerate_prototype": False,
    }


def compatibility_gate(scores: np.ndarray, usable: np.ndarray | None = None) -> float:
    """A bounded, parameter-free ``alpha`` in ``[0, 1]`` from the same geometry.

    ``max(0, dir_max)``: nonnegative, so a candidate whose every graph-supported
    direction disagrees with the residual is left alone rather than pushed
    backwards; bounded by 1 because the inputs are cosines. No threshold, no
    scale, and nothing fitted -- the gate has to be as parameter-free as the
    offset it gates, or the "zero new parameters" claim is untrue.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.isfinite(scores)
    if usable is not None:
        mask &= np.asarray(usable, dtype=bool).reshape(-1)
    if not mask.any():
        return 0.0
    return float(max(0.0, scores[mask].max()))


def project_offset(
    node_state: np.ndarray,
    projected_direction: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """``normalize(d64 + alpha * W_d delta_star)`` -- the offset state.

    ``projected_direction`` is ``W_d @ delta_star`` computed by the caller with
    S4's **existing** frozen ``node_projection``. No new learned matrix is
    introduced, which is the whole content of the "zero new parameters" claim.

    Note what this does *not* do. S4's ``node_state`` is
    ``normalize(gelu(W_d e_v))``, so it has passed through a GELU and
    ``W_d delta_star`` has not; the sum mixes a post-activation state with a
    pre-activation displacement. That is the specified form, and it is the
    Jacobian-free first-order image of the displacement -- it ignores GELU's
    local slope. The declaration records the alternative
    ``normalize(gelu(W_d (e_v + alpha delta_star)))`` and the fact that it is
    not run as a second arm.
    """

    state = np.asarray(node_state, dtype=np.float64)
    direction = np.asarray(projected_direction, dtype=np.float64)
    if state.shape != direction.shape:
        raise ValueError(
            f"node_state {state.shape} and projected direction {direction.shape} differ"
        )
    shifted = state + float(alpha) * direction
    norm = np.linalg.norm(shifted, axis=-1, keepdims=True)
    return np.where(norm > EPS, shifted / np.where(norm > EPS, norm, 1.0), state)


# ---------------------------------------------------------------------------
# Stage-0 additions: the raw-query control, RRF fusion, and the
# error-conditioned diagnostic.
# ---------------------------------------------------------------------------

#: The project's frozen reciprocal-rank-fusion constant, declared in
#: ``configs/candidate_budget.yaml#candidate_construction.rrf_constant`` and used
#: unchanged by candidate_headroom and graph_context_pilot. Reused here rather
#: than chosen, so the fusion arm introduces no new tunable quantity.
RRF_CONSTANT = 60


def raw_query_direction(query_embedding: np.ndarray) -> Residual:
    """``normalize(q)`` -- the control that subtracts nothing.

    Present so that a directional result cannot be attributed to the residual
    idea when a bare query direction would have produced it. If this ranks as
    well as either residual, then "what the seeds do not explain" is not the
    operative construct and the projection is decoration.
    """

    query = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(query))
    if norm <= EPS:
        return Residual(
            vector=np.zeros_like(query),
            seed_count=0,
            subspace_rank=0,
            residual_norm=0.0,
            query_norm=norm,
            relative_norm=0.0,
            degenerate=True,
            fell_back_to_query=False,
        )
    return Residual(
        vector=query / norm,
        seed_count=0,
        subspace_rank=0,
        residual_norm=norm,
        query_norm=norm,
        relative_norm=1.0,
        degenerate=False,
        fell_back_to_query=False,
    )


def rank_positions(scores: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    """1-based ranks, descending by score, ties broken by ascending node id.

    The tie-break matches the frozen ``equal_rrf`` convention recorded in
    ``candidate_budget.py`` (``rrf_tie_break: ascending_global_node_id``), so
    two rankings built here order ties the same way the existing pipeline does.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    node_ids = np.asarray(node_ids).reshape(-1)
    if scores.shape != node_ids.shape:
        raise ValueError(f"scores {scores.shape} and node_ids {node_ids.shape} differ")
    order = np.lexsort((node_ids, -scores))
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def reciprocal_rank_fusion(
    rankings: Sequence[np.ndarray],
    node_ids: np.ndarray,
    *,
    constant: int = RRF_CONSTANT,
) -> np.ndarray:
    """Equal-weight RRF over rankings of one shared candidate set.

    ``sum_i 1 / (constant + rank_i)``. Equal weights and a reused constant, so
    nothing here is fitted: the arm is falsifiable exactly as it stands, which
    a tuned score-blend weight would not be.

    Returns a fused score -- higher is better -- rather than an order, so the
    caller can rank it with :func:`rank_positions` and inherit the same
    tie-break.
    """

    if not rankings:
        raise ValueError("at least one ranking is required")
    node_ids = np.asarray(node_ids).reshape(-1)
    fused = np.zeros(node_ids.size, dtype=np.float64)
    for ranking in rankings:
        ranking = np.asarray(ranking, dtype=np.float64).reshape(-1)
        if ranking.shape != fused.shape:
            raise ValueError("every ranking must cover the same candidate set")
        if np.any(ranking < 1):
            raise ValueError("ranks are 1-based")
        fused += 1.0 / (float(constant) + ranking)
    return fused


@dataclass(frozen=True)
class ErrorMargin:
    """One query's directional evidence about the mistake S4 actually made."""

    query_id: str
    relevant_index: int
    wrong_index: int
    relevant_direction: float
    wrong_direction: float
    margin: float
    first_relevant_rank: int
    stratum: str

    def as_row(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "relevant_index": self.relevant_index,
            "wrong_index": self.wrong_index,
            "dir_relevant": self.relevant_direction,
            "dir_top_wrong": self.wrong_direction,
            "directional_margin": self.margin,
            "first_relevant_rank": self.first_relevant_rank,
            "stratum": self.stratum,
        }


def _stratum(rank: int) -> str:
    if rank <= 0:
        return "absent"
    if rank == 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_5"
    if rank <= 20:
        return "rank_6_20"
    return "beyond_20"


def error_conditioned_margin(
    query_id: str,
    model_scores: np.ndarray,
    directional_scores: np.ndarray,
    node_ids: np.ndarray,
    relevant_mask: np.ndarray,
) -> ErrorMargin | None:
    """Does the direction prefer a relevant candidate over S4's top mistake?

    Restricted to the population the phase actually needs to repair: queries
    where S4's top-1 is wrong AND a relevant candidate is present in the scored
    universe. Everything else returns ``None`` and is counted separately by the
    caller -- a query with no relevant candidate scored cannot be fixed by any
    reranker, and leaving it in would dilute the margin with cases nothing could
    win.

    ``relevant_mask`` is used ONLY to select which candidates to compare, never
    to compute a direction: the directional scores are handed in already built
    from embeddings and topology.
    """

    model_scores = np.asarray(model_scores, dtype=np.float64).reshape(-1)
    directional_scores = np.asarray(directional_scores, dtype=np.float64).reshape(-1)
    relevant = np.asarray(relevant_mask, dtype=bool).reshape(-1)
    node_ids = np.asarray(node_ids).reshape(-1)
    if not (model_scores.shape == directional_scores.shape == relevant.shape == node_ids.shape):
        raise ValueError("scores, directions, mask and ids must describe the same candidates")
    if not relevant.any():
        return None

    ranks = rank_positions(model_scores, node_ids)
    top_index = int(np.argmin(ranks))
    if relevant[top_index]:
        return None  # S4 is already right here; there is no mistake to repair.

    relevant_positions = np.flatnonzero(relevant)
    first_relevant_rank = int(ranks[relevant_positions].min())

    # The best relevant candidate BY DIRECTION -- the one the mechanism would
    # have to promote. Ties fall to the lowest node id, as everywhere else.
    best = relevant_positions[
        np.lexsort(
            (node_ids[relevant_positions], -directional_scores[relevant_positions])
        )[0]
    ]
    return ErrorMargin(
        query_id=query_id,
        relevant_index=int(best),
        wrong_index=top_index,
        relevant_direction=float(directional_scores[best]),
        wrong_direction=float(directional_scores[top_index]),
        margin=float(directional_scores[best] - directional_scores[top_index]),
        first_relevant_rank=first_relevant_rank,
        stratum=_stratum(first_relevant_rank),
    )


def summarise_margins(margins: Sequence[ErrorMargin]) -> dict[str, Any]:
    """Aggregate the error-conditioned diagnostic, stratified as declared."""

    if not margins:
        return {
            "queries": 0,
            "fraction_positive": None,
            "mean_margin": None,
            "median_margin": None,
            "by_stratum": {},
        }
    values = np.array([m.margin for m in margins], dtype=np.float64)
    summary: dict[str, Any] = {
        "queries": int(values.size),
        "fraction_positive": float((values > 0).mean()),
        "mean_margin": float(values.mean()),
        "median_margin": float(np.median(values)),
        "by_stratum": {},
    }
    for stratum in ("rank_2_5", "rank_6_20", "beyond_20", "absent"):
        subset = np.array(
            [m.margin for m in margins if m.stratum == stratum], dtype=np.float64
        )
        summary["by_stratum"][stratum] = {
            "queries": int(subset.size),
            "fraction_positive": None if subset.size == 0 else float((subset > 0).mean()),
            "mean_margin": None if subset.size == 0 else float(subset.mean()),
        }
    return summary


@dataclass(frozen=True)
class PoolDirection:
    """One query's directional evidence over one scored pool, one provenance."""

    dir_max: np.ndarray
    dir_mean: np.ndarray
    support: np.ndarray
    covered: np.ndarray

    @property
    def coverage(self) -> float:
        """Fraction of scored candidates with at least one eligible displacement.

        Reported beside every margin. A signal present on a small minority of
        candidates cannot repair a ranking even where it is perfectly
        informative, and a margin quoted without this hides that.
        """

        return 0.0 if self.covered.size == 0 else float(self.covered.mean())


def seed_incident_pairs(
    pool: np.ndarray,
    seeds: np.ndarray,
    rowptr: np.ndarray,
    col: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(seed_position, pool_position)`` for every seed-to-candidate edge.

    The provenance graph decides which pairs exist, so switching G_STRUCT for
    G_KNN changes this and nothing else. Candidates unreachable from any seed
    produce no pair and are reported as uncovered rather than as zero -- "no
    structural evidence" and "evidence pointing nowhere" are different states.
    """

    pool = np.asarray(pool, dtype=np.int64).reshape(-1)
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
    if pool.size == 0 or seeds.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    order = np.argsort(pool, kind="stable")
    sorted_pool = pool[order]

    seed_positions, pool_positions = [], []
    for index, seed in enumerate(seeds):
        start, end = int(rowptr[seed]), int(rowptr[seed + 1])
        if end <= start:
            continue
        neighbours = col[start:end]
        found = np.searchsorted(sorted_pool, neighbours)
        np.clip(found, 0, sorted_pool.size - 1, out=found)
        hit = sorted_pool[found] == neighbours
        if not hit.any():
            continue
        matched = order[found[hit]]
        seed_positions.append(np.full(matched.size, index, dtype=np.int64))
        pool_positions.append(matched)

    if not pool_positions:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(seed_positions), np.concatenate(pool_positions)


def pool_directional_scores(
    residual: np.ndarray,
    pool_embeddings: np.ndarray,
    seed_embeddings: np.ndarray,
    seed_positions: np.ndarray,
    pool_positions: np.ndarray,
) -> PoolDirection:
    """``dir_max`` and ``dir_mean`` per candidate, over its incident seed edges.

    Uses the identity that keeps this cheap and exact:

        cos(r, normalize(e_v - e_s)) = (r.e_v - r.e_s) / ||e_v - e_s||
        ||e_v - e_s||^2 = ||e_v||^2 + ||e_s||^2 - 2 e_v.e_s

    so no ``pairs x dim`` displacement matrix is ever materialised. The result
    is the same quantity :func:`displacements` and :func:`compatibility` define
    pairwise; the tests hold the two forms against each other.

    Uncovered candidates get ``dir_max = dir_mean = 0`` AND ``covered = False``.
    The caller must rank on the covered mask, not on the zeros, or a candidate
    with no structural evidence outranks one the geometry actively argues
    against.
    """

    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    pool_embeddings = np.asarray(pool_embeddings, dtype=np.float64)
    seed_embeddings = np.asarray(seed_embeddings, dtype=np.float64)
    count = pool_embeddings.shape[0]

    # -inf, not 0: a candidate whose every supported direction is NEGATIVE has
    # a negative maximum, and accumulating from zero would silently report it
    # as 0.0 -- indistinguishable from a candidate with no evidence at all.
    dir_max = np.full(count, -np.inf, dtype=np.float64)
    dir_sum = np.zeros(count, dtype=np.float64)
    support = np.zeros(count, dtype=np.int64)
    if seed_positions.size == 0:
        return PoolDirection(np.zeros(count), dir_sum, support, support > 0)

    proj_pool = pool_embeddings @ residual
    proj_seed = seed_embeddings @ residual
    sq_pool = np.einsum("ij,ij->i", pool_embeddings, pool_embeddings)
    sq_seed = np.einsum("ij,ij->i", seed_embeddings, seed_embeddings)

    gram = np.einsum(
        "ij,ij->i",
        pool_embeddings[pool_positions],
        seed_embeddings[seed_positions],
    )
    distance_sq = sq_pool[pool_positions] + sq_seed[seed_positions] - 2.0 * gram
    np.maximum(distance_sq, 0.0, out=distance_sq)
    distance = np.sqrt(distance_sq)

    # A candidate that IS a seed has a zero displacement, which defines no
    # direction. Dropped rather than divided by, and it costs that candidate a
    # support count rather than producing an infinity.
    usable = distance > EPS
    numerator = proj_pool[pool_positions] - proj_seed[seed_positions]
    scores = np.where(usable, numerator / np.where(usable, distance, 1.0), 0.0)

    targets = pool_positions[usable]
    values = scores[usable]
    np.add.at(dir_sum, targets, values)
    np.add.at(support, targets, 1)
    np.maximum.at(dir_max, targets, values)

    covered = support > 0
    dir_mean = np.zeros(count, dtype=np.float64)
    np.divide(dir_sum, support, out=dir_mean, where=covered)
    dir_max[~covered] = 0.0
    if not np.all(np.isfinite(dir_max)):  # pragma: no cover - defensive
        raise RuntimeError("a covered candidate kept its sentinel maximum")
    return PoolDirection(dir_max, dir_mean, support, covered)


def rank_with_coverage(
    scores: np.ndarray, covered: np.ndarray, node_ids: np.ndarray
) -> np.ndarray:
    """Rank by directional score, with every uncovered candidate placed last.

    Ranking uncovered candidates on their zero would let "no structural
    evidence" beat a real negative score, which would make the direction-only
    arm partly a measure of graph sparsity.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    covered = np.asarray(covered, dtype=bool).reshape(-1)
    node_ids = np.asarray(node_ids).reshape(-1)
    if not (scores.shape == covered.shape == node_ids.shape):
        raise ValueError("scores, coverage and ids must describe the same candidates")
    order = np.lexsort((node_ids, -scores, ~covered))
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def expand_with_residual(
    residual: np.ndarray,
    *,
    rowptr: np.ndarray,
    col: np.ndarray,
    node_embeddings: Any,
    anchor: int,
    pool: np.ndarray,
    seeds: np.ndarray,
    budget: Any,
    num_nodes: int,
) -> Any:
    """``candidate_expansion_v2.expand`` with the residual supplied, not derived.

    The whole point of the +64 admission diagnostic is that ONE thing changes
    between arms. So this does not reimplement expansion: it calls the frozen
    module's own ``_frontier_neighbours``, ``_unit``, ``_rank`` and
    ``_matched_budget``, and returns the frozen module's own ``Expansion``. The
    only line that differs from ``expand(L1_DIRECTIONAL, ...)`` is where the
    residual comes from.

    Passing ``query_residual(e_q, e_anchor)`` here must reproduce
    ``expand(L1_DIRECTIONAL, ...)`` exactly; a test holds the two together, and
    if that equality ever breaks then this arm is no longer the mechanism M0A
    nulled and the comparison against its null is void.
    """

    from . import candidate_expansion_v2 as _cx

    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    pool = np.asarray(pool, dtype=np.int64)
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    in_pool = np.zeros(num_nodes, dtype=bool)
    in_pool[pool] = True

    per_seed, scanned, fired = _cx._frontier_neighbours(
        rowptr=rowptr, col=col, seeds=seeds, in_pool=in_pool, budget=budget
    )
    capped_seeds = sum(
        1 for _seed, neighbours in per_seed if neighbours.size > budget.per_seed_cap
    )

    degenerate = bool(np.all(residual == 0.0))
    zero_edges = 0
    nodes_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    if not degenerate:
        for seed, neighbours in per_seed:
            source = np.asarray(node_embeddings[seed], dtype=np.float64)
            targets = np.asarray(node_embeddings[neighbours], dtype=np.float64)
            directions, usable = _cx._unit(targets - source)
            zero_edges += int((~usable).sum())
            scores = directions @ residual
            scores = np.where(usable, scores, -np.inf)
            keep = _cx._rank(scores, neighbours, budget.per_seed_cap)
            nodes_parts.append(neighbours[keep])
            score_parts.append(scores[keep])

    if nodes_parts:
        nodes = np.concatenate(nodes_parts)
        scores = np.concatenate(score_parts)
        unique_nodes, inverse = np.unique(nodes, return_inverse=True)
        best = np.full(unique_nodes.size, -np.inf, dtype=np.float64)
        np.maximum.at(best, inverse, scores)
        finite = np.isfinite(best)
        unique_nodes, best = unique_nodes[finite], best[finite]
        keep = _cx._rank(best, unique_nodes, budget.graph_expansion_cap)
        admitted, admitted_scores = unique_nodes[keep], best[keep]
    else:
        admitted = np.zeros(0, dtype=np.int64)
        admitted_scores = np.zeros(0, dtype=np.float64)

    matched, used, used_scores, evicted, protected = _cx._matched_budget(
        pool=pool, seeds=seeds, anchor=int(anchor), admitted=admitted, scores=admitted_scores
    )
    additive = np.concatenate([pool, admitted]) if admitted.size else pool.copy()
    return _cx.Expansion(
        method="M2C_RESIDUAL_DIRECTIONAL",
        admitted=admitted,
        scores=admitted_scores,
        matched_pool=matched,
        additive_pool=additive,
        evicted=evicted,
        protected=protected,
        neighbours_scanned=scanned,
        scan_cap_fired=fired,
        seeds_at_the_per_seed_cap=capped_seeds,
        degenerate_residual=degenerate,
        zero_displacement_edges=zero_edges,
        diagnostics={
            "admitted_used_under_matched_budget": int(used.size),
            "admitted_scores_used": used_scores,
            "seeds_with_neighbours": len(per_seed),
        },
    )


def admission_overlap(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Jaccard and unique counts for two admitted sets.

    The decisive quantity in the +64 diagnostic. M0A's null was that the
    directional method and its blind control admitted sets whose difference
    changed no ceiling; if two arms here select the same nodes, that null
    transfers to them directly and no ceiling needs recomputing to know it.
    """

    left = np.unique(np.asarray(left, dtype=np.int64))
    right = np.unique(np.asarray(right, dtype=np.int64))
    intersection = int(np.intersect1d(left, right, assume_unique=True).size)
    union = int(left.size + right.size - intersection)
    return {
        "left_size": int(left.size),
        "right_size": int(right.size),
        "intersection": intersection,
        "union": union,
        # An empty union means neither arm admitted anything. That is perfect
        # agreement about admitting nothing, not an undefined comparison.
        "jaccard": 1.0 if union == 0 else intersection / union,
        "unique_to_left": int(left.size - intersection),
        "unique_to_right": int(right.size - intersection),
    }
