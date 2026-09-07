"""The M2C structural-offset primitives, checked as mathematics.

These are deterministic geometric functions, so they can be tested against
their definitions rather than against recorded output. The properties that
matter:

* the residual really is orthogonal to the seed span -- that is the whole
  claim, and a projection built the naive way silently fails it when seeds are
  near-collinear, which is the common case because seeds are retrieved by
  similarity to one query;
* every degenerate input is handled AND recorded, so a fallback cannot become
  an unannounced second method;
* the subspace residual is genuinely different from the single-anchor residual
  the repository already had, since the entire argument for running Track B
  again rests on that difference being real;
* nothing reads a label.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval import m2c_structural_offset as offset  # noqa: E402
from mp_retrieval.candidate_expansion_v2 import query_residual  # noqa: E402

DIM = 32


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260907)


# ---------------------------------------------------------------------------
# The residual is what it says it is.
# ---------------------------------------------------------------------------


def test_the_residual_is_orthogonal_to_every_seed(rng):
    """The defining property. If this fails, nothing downstream means anything."""

    query = rng.normal(size=DIM)
    seeds = rng.normal(size=(6, DIM))
    result = offset.seed_subspace_residual(query, seeds)

    assert not result.degenerate
    assert result.subspace_rank == 6
    assert np.allclose(seeds @ result.vector, 0.0, atol=1e-10)
    assert result.vector @ result.vector == pytest.approx(1.0)


def test_near_collinear_seeds_do_not_break_the_projection(rng):
    """The case a normal-equations projector gets wrong.

    ``E (E^T E)^-1 E^T`` is singular to working precision when two seeds are
    nearly parallel, and seeds here are retrieved by similarity to one query,
    so they routinely are. The SVD path has to stay orthogonal anyway.
    """

    base = rng.normal(size=DIM)
    seeds = np.stack([base, base * (1.0 + 1e-11), base + 1e-9 * rng.normal(size=DIM)])
    query = rng.normal(size=DIM)
    result = offset.seed_subspace_residual(query, seeds)

    assert result.subspace_rank == 1, "three near-parallel seeds span one direction"
    assert abs(float(base @ result.vector)) < 1e-8
    assert np.isfinite(result.vector).all()


def test_exactly_collinear_seeds_collapse_to_rank_one(rng):
    base = rng.normal(size=DIM)
    seeds = np.stack([base, 2.0 * base, -3.5 * base])
    result = offset.seed_subspace_residual(rng.normal(size=DIM), seeds)

    assert result.subspace_rank == 1
    single = offset.seed_subspace_residual(rng.normal(size=DIM), base[None, :])
    assert single.subspace_rank == 1


def test_a_rank_deficient_seed_matrix_reports_its_real_rank(rng):
    """Five seeds spanning a three-dimensional subspace are rank three."""

    basis = rng.normal(size=(3, DIM))
    weights = rng.normal(size=(5, 3))
    seeds = weights @ basis
    result = offset.seed_subspace_residual(rng.normal(size=DIM), seeds)

    assert result.seed_count == 5
    assert result.subspace_rank == 3
    assert np.allclose(seeds @ result.vector, 0.0, atol=1e-10)


def test_zero_seeds_give_the_normalized_query_and_are_not_degenerate(rng):
    """Nothing is subtracted, so the direction is well defined. Not a fallback."""

    query = rng.normal(size=DIM)
    result = offset.seed_subspace_residual(query, np.zeros((0, DIM)))

    assert result.seed_count == 0
    assert result.subspace_rank == 0
    assert result.degenerate is False
    assert result.fell_back_to_query is False
    assert np.allclose(result.vector, query / np.linalg.norm(query))


def test_a_query_inside_the_seed_span_falls_back_and_says_so(rng):
    """The degenerate case that must never pass silently.

    When the seeds already explain the query there is no "what is missing"
    direction. The fallback is the normalized query, and both flags are set so
    a row carrying it is distinguishable from a row with a real residual.
    """

    seeds = rng.normal(size=(4, DIM))
    query = seeds.T @ rng.normal(size=4)  # exactly in the span
    result = offset.seed_subspace_residual(query, seeds)

    assert result.degenerate is True
    assert result.fell_back_to_query is True
    assert result.relative_norm < offset.RESIDUAL_RELATIVE_FLOOR
    assert np.allclose(result.vector, query / np.linalg.norm(query))
    assert result.diagnostics()["degenerate_residual"] is True


def test_a_zero_query_is_degenerate_without_pretending_to_have_a_direction(rng):
    result = offset.seed_subspace_residual(np.zeros(DIM), rng.normal(size=(3, DIM)))
    assert result.degenerate is True
    assert np.allclose(result.vector, 0.0)
    assert result.fell_back_to_query is False


def test_unusable_seed_rows_are_dropped_before_they_define_a_basis(rng):
    """A zero seed row must not contribute an arbitrary basis direction."""

    seeds = rng.normal(size=(3, DIM))
    padded = np.vstack([seeds, np.zeros(DIM)])
    query = rng.normal(size=DIM)

    clean = offset.seed_subspace_residual(query, seeds)
    padded_result = offset.seed_subspace_residual(query, padded)

    assert padded_result.seed_count == 3
    assert padded_result.subspace_rank == clean.subspace_rank
    assert np.allclose(padded_result.vector, clean.vector)


def test_a_width_mismatch_raises_rather_than_broadcasting(rng):
    with pytest.raises(ValueError, match="does not match query width"):
        offset.seed_subspace_residual(rng.normal(size=DIM), rng.normal(size=(2, DIM + 1)))


# ---------------------------------------------------------------------------
# The new residual is genuinely a different object from the old one.
# ---------------------------------------------------------------------------


def test_the_subspace_residual_differs_from_the_single_anchor_residual(rng):
    """Track B's entire justification is that this difference is real.

    ``q - x_a`` subtracts the anchor vector; ``q - P q`` subtracts only the
    component of q along it. They are not nested and they do not point the same
    way, so re-testing M0A's null with the new residual is a real experiment
    rather than a re-run.
    """

    query = rng.normal(size=DIM)
    seeds = rng.normal(size=(5, DIM))
    anchor = seeds[0]

    subspace = offset.seed_subspace_residual(query, seeds)
    single = offset.single_anchor_residual(query, anchor)

    cosine = float(subspace.vector @ single.vector)
    assert abs(cosine) < 0.99, "the two residuals should not be near-identical"
    # And the one-seed case is still not the anchor case, which is the point.
    one_seed = offset.seed_subspace_residual(query, anchor[None, :])
    assert not np.allclose(one_seed.vector, single.vector)


def test_the_single_anchor_wrapper_matches_the_existing_repository_function(rng):
    """The comparison arm must be the SAME residual M0A used, not a lookalike."""

    query = rng.normal(size=DIM)
    anchor = rng.normal(size=DIM)
    assert np.allclose(
        offset.single_anchor_residual(query, anchor).vector,
        query_residual(query, anchor),
    )


def test_a_degenerate_anchor_residual_is_flagged_like_the_existing_one(rng):
    query = rng.normal(size=DIM)
    result = offset.single_anchor_residual(query, query)
    assert result.degenerate is True
    assert np.allclose(result.vector, 0.0)
    assert np.allclose(query_residual(query, query), 0.0)


# ---------------------------------------------------------------------------
# Displacements, compatibility, statistics.
# ---------------------------------------------------------------------------


def test_displacements_are_unit_vectors_and_flag_the_zero_edge(rng):
    seed = rng.normal(size=DIM)
    neighbours = np.vstack([rng.normal(size=(4, DIM)), seed])  # last is a zero edge
    deltas, usable = offset.displacements(seed, neighbours)

    assert usable[:4].all() and not usable[4]
    assert np.allclose(np.linalg.norm(deltas[:4], axis=1), 1.0)
    assert np.allclose(deltas[4], 0.0)


def test_compatibility_is_the_cosine_it_claims_to_be(rng):
    residual = offset.seed_subspace_residual(
        rng.normal(size=DIM), rng.normal(size=(3, DIM))
    ).vector
    deltas, _ = offset.displacements(rng.normal(size=DIM), rng.normal(size=(6, DIM)))
    scores = offset.compatibility(residual, deltas)

    assert scores.shape == (6,)
    assert (np.abs(scores) <= 1.0 + 1e-12).all()
    for index in range(6):
        assert scores[index] == pytest.approx(float(deltas[index] @ residual))


def test_no_structural_support_is_distinguishable_from_hostile_support():
    """The distinction a structural feature has to keep, or it becomes degree.

    A candidate with no graph-supported direction and a candidate whose every
    direction disagrees with the residual are different states. Collapsing them
    to the same number would let the model read "has neighbours" instead of
    "has neighbours pointing the right way".
    """

    none_at_all = offset.directional_statistics(np.array([]))
    all_hostile = offset.directional_statistics(np.array([-0.4, -0.9, -0.2]))

    assert none_at_all["dir_support_count"] == 0.0
    assert all_hostile["dir_support_count"] == 3.0
    assert all_hostile["dir_max"] == pytest.approx(-0.2)
    assert none_at_all["dir_max"] == 0.0
    assert all_hostile["dir_positive_mass"] == 0.0
    assert none_at_all["dir_positive_mass"] == 0.0


def test_the_statistics_are_the_four_declared_ones_and_are_computed_as_declared():
    scores = np.array([0.5, -0.25, 0.75, 0.0])
    stats = offset.directional_statistics(scores)

    assert set(stats) == {"dir_max", "dir_mean", "dir_positive_mass", "dir_support_count"}
    assert stats["dir_max"] == pytest.approx(0.75)
    assert stats["dir_mean"] == pytest.approx(0.25)
    assert stats["dir_positive_mass"] == pytest.approx((0.5 + 0.75) / 4)
    assert stats["dir_support_count"] == 4.0


def test_non_finite_scores_are_excluded_rather_than_poisoning_the_mean():
    stats = offset.directional_statistics(np.array([0.5, -np.inf, 0.1]))
    assert stats["dir_support_count"] == 2.0
    assert stats["dir_mean"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Prototypes.
# ---------------------------------------------------------------------------


def test_the_prototype_is_a_unit_vector_in_the_span_of_its_directions(rng):
    deltas, usable = offset.displacements(rng.normal(size=DIM), rng.normal(size=(5, DIM)))
    scores = offset.compatibility(rng.normal(size=DIM) / np.sqrt(DIM), deltas)
    prototype, info = offset.direction_prototype(deltas, scores, tau=8.0, usable=usable)

    assert info["degenerate_prototype"] is False
    assert np.linalg.norm(prototype) == pytest.approx(1.0)
    # A convex combination of the rows lies in their row space.
    residual_out_of_span = prototype - deltas.T @ np.linalg.lstsq(
        deltas.T, prototype, rcond=None
    )[0]
    assert np.linalg.norm(residual_out_of_span) < 1e-10


def test_a_large_temperature_does_not_overflow_and_selects_the_best_direction(rng):
    deltas, _ = offset.displacements(rng.normal(size=DIM), rng.normal(size=(4, DIM)))
    scores = np.array([0.1, 0.9, -0.3, 0.2])
    prototype, info = offset.direction_prototype(deltas, scores, tau=10_000.0)

    assert np.isfinite(prototype).all()
    assert info["weight_max"] == pytest.approx(1.0)
    assert np.allclose(prototype, deltas[1], atol=1e-8)


def test_cancelling_directions_produce_no_prototype_rather_than_a_fabricated_one():
    """Two opposed displacements with equal weight have no mean direction.

    Returning an arbitrary unit vector here would invent a direction the
    geometry does not contain, which is exactly the failure a deterministic
    mechanism is supposed to be immune to.
    """

    deltas = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    prototype, info = offset.direction_prototype(deltas, np.array([0.5, 0.5]), tau=1.0)

    assert info["degenerate_prototype"] is True
    assert np.allclose(prototype, 0.0)


def test_no_eligible_direction_produces_no_prototype():
    deltas = np.zeros((3, DIM))
    prototype, info = offset.direction_prototype(
        deltas, np.array([0.1, 0.2, 0.3]), tau=1.0, usable=np.zeros(3, dtype=bool)
    )
    assert info["eligible_directions"] == 0
    assert info["degenerate_prototype"] is True
    assert np.allclose(prototype, 0.0)


def test_the_unweighted_control_takes_the_top_k_by_score(rng):
    """It exists so that no result rests on tau. It must actually differ."""

    deltas, _ = offset.displacements(rng.normal(size=DIM), rng.normal(size=(6, DIM)))
    scores = np.array([0.9, 0.1, 0.8, -0.5, 0.2, 0.7])
    prototype, info = offset.mean_direction_prototype(deltas, scores, top_k=3)

    expected = deltas[[0, 2, 5]].mean(axis=0)
    assert info["used_directions"] == 3
    assert np.allclose(prototype, expected / np.linalg.norm(expected))


def test_the_control_ties_break_the_same_way_the_admission_code_does():
    """Descending score, then ascending position -- matching _rank in the
    existing expansion module, so the two never disagree about a tie."""

    deltas = np.eye(4)
    prototype, info = offset.mean_direction_prototype(
        deltas, np.array([0.5, 0.5, 0.5, 0.5]), top_k=2
    )
    assert info["used_directions"] == 2
    expected = (deltas[0] + deltas[1]) / 2
    assert np.allclose(prototype, expected / np.linalg.norm(expected))


# ---------------------------------------------------------------------------
# The gate and the offset.
# ---------------------------------------------------------------------------


def test_the_gate_is_bounded_nonnegative_and_parameter_free():
    assert offset.compatibility_gate(np.array([0.4, -0.2, 0.9])) == pytest.approx(0.9)
    assert offset.compatibility_gate(np.array([-0.4, -0.2, -0.9])) == 0.0
    assert offset.compatibility_gate(np.array([])) == 0.0
    # Cosines are bounded by one, so the gate is bounded by one.
    assert offset.compatibility_gate(np.array([1.0])) <= 1.0


def test_a_hostile_candidate_is_left_alone_rather_than_pushed_backwards(rng):
    """alpha = 0 must be an exact identity on the node state."""

    state = rng.normal(size=64)
    state /= np.linalg.norm(state)
    direction = rng.normal(size=64)
    shifted = offset.project_offset(state, direction, alpha=0.0)
    assert np.allclose(shifted, state)


def test_the_offset_returns_a_unit_state(rng):
    state = rng.normal(size=(5, 64))
    state /= np.linalg.norm(state, axis=1, keepdims=True)
    direction = rng.normal(size=(5, 64))
    shifted = offset.project_offset(state, direction, alpha=0.3)
    assert np.allclose(np.linalg.norm(shifted, axis=1), 1.0)


def test_an_exactly_cancelling_offset_returns_the_original_state():
    """The sum can vanish. Returning zeros would hand the scorer a dead row."""

    state = np.array([[1.0, 0.0]])
    shifted = offset.project_offset(state, -state, alpha=1.0)
    assert np.allclose(shifted, state)


def test_a_shape_mismatch_raises_rather_than_broadcasting(rng):
    with pytest.raises(ValueError, match="differ"):
        offset.project_offset(rng.normal(size=(3, 64)), rng.normal(size=(3, 32)), 0.5)


# ---------------------------------------------------------------------------
# Leakage and determinism.
# ---------------------------------------------------------------------------


def test_nothing_in_the_module_accepts_a_label(rng):
    """The firewall, checked on the signatures rather than promised in prose."""

    import inspect

    forbidden = ("gold", "label", "relevant", "supporting", "split", "dataset", "target")
    for name, function in vars(offset).items():
        if not callable(function) or name.startswith("_") or not inspect.isfunction(function):
            continue
        parameters = set(inspect.signature(function).parameters)
        for parameter in parameters:
            assert not any(word in parameter.lower() for word in forbidden), (name, parameter)


def test_the_pipeline_is_deterministic_and_a_pure_function_of_its_inputs(rng):
    query = rng.normal(size=DIM)
    seeds = rng.normal(size=(4, DIM))
    neighbours = rng.normal(size=(7, DIM))

    def run() -> np.ndarray:
        residual = offset.seed_subspace_residual(query, seeds).vector
        deltas, usable = offset.displacements(seeds[0], neighbours)
        scores = offset.compatibility(residual, deltas)
        prototype, _ = offset.direction_prototype(deltas, scores, tau=4.0, usable=usable)
        return prototype

    first, second = run(), run()
    assert np.array_equal(first, second)


def test_reordering_the_seeds_does_not_change_the_subspace_residual(rng):
    """The span is a set property. A permutation must not move the residual.

    Worth pinning: the SVD basis itself is not permutation-invariant, only the
    subspace it spans is, so this checks the projector rather than the basis.
    """

    query = rng.normal(size=DIM)
    seeds = rng.normal(size=(5, DIM))
    permuted = seeds[[3, 0, 4, 1, 2]]

    original = offset.seed_subspace_residual(query, seeds)
    reordered = offset.seed_subspace_residual(query, permuted)

    assert np.allclose(original.vector, reordered.vector, atol=1e-12)
    assert original.subspace_rank == reordered.subspace_rank
