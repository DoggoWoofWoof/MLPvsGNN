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
