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


#: The ONLY functions in this module permitted to see a label, and why.
#: Everything else is on the inference path and must be a pure function of
#: embeddings, topology and seeds. Adding a label-taking function without
#: adding it here fails the test below, which is the point: the allowlist has
#: to be edited deliberately, in a diff someone reviews.
LABEL_AWARE_DIAGNOSTICS = {
    # Selects WHICH candidates to compare in the dev-split error diagnostic.
    # Never used to compute a direction -- the directional scores are passed in
    # already built.
    "error_conditioned_margin",
}


def test_no_direction_building_function_accepts_a_label(rng):
    """The firewall, checked on the signatures rather than promised in prose."""

    import inspect

    forbidden = ("gold", "label", "relevant", "supporting", "split", "dataset", "target")
    checked = 0
    for name, function in vars(offset).items():
        if not callable(function) or name.startswith("_") or not inspect.isfunction(function):
            continue
        if name in LABEL_AWARE_DIAGNOSTICS:
            continue
        checked += 1
        for parameter in inspect.signature(function).parameters:
            assert not any(word in parameter.lower() for word in forbidden), (name, parameter)
    assert checked >= 8, "the sweep should cover the whole inference surface"


def test_the_label_aware_allowlist_is_accurate_and_minimal(rng):
    """An allowlist that drifts from the code is worse than no allowlist.

    Both directions are checked: every name on the list exists and really does
    take a label, and nothing on it is on the direction-building path.
    """

    import inspect

    for name in LABEL_AWARE_DIAGNOSTICS:
        function = getattr(offset, name, None)
        assert inspect.isfunction(function), f"{name} is on the allowlist but does not exist"
        parameters = list(inspect.signature(function).parameters)
        assert any("relevant" in p or "gold" in p or "label" in p for p in parameters), (
            f"{name} is on the label allowlist but takes no label; remove it"
        )
        # A diagnostic may read a label. It may not also be handed a raw query
        # or seed embedding, because then it could build a direction from one.
        assert not any(p in ("query_embedding", "seed_embeddings") for p in parameters), (
            f"{name} sees labels and embeddings; it could build a leaky direction"
        )


def test_the_diagnostic_cannot_change_a_direction_it_is_given(rng):
    """The substantive form of the allowlist claim, not just a naming check.

    Permuting the relevance mask must not alter any directional score -- it can
    only change which candidates the diagnostic compares. If a label could move
    a direction, the whole Stage-0 measurement would be circular.
    """

    node_ids = np.arange(8)
    model = rng.normal(size=8)
    directions = rng.normal(size=8)
    mask_a = np.array([0, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
    mask_b = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=bool)

    covered = np.ones(8, dtype=bool)
    first = offset.error_conditioned_margin(
        "q", model, directions, node_ids, mask_a, covered=covered
    )
    second = offset.error_conditioned_margin(
        "q", model, directions, node_ids, mask_b, covered=covered
    )

    for result in (first, second):
        if result is not None:
            assert result.wrong_direction == pytest.approx(
                float(directions[result.wrong_index])
            ), "the diagnostic must report the direction it was handed, unmodified"


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


# ---------------------------------------------------------------------------
# Stage-0 primitives: the raw-query control, RRF fusion, error margins.
# ---------------------------------------------------------------------------


def test_the_raw_query_control_subtracts_nothing(rng):
    """It has to be a genuinely different direction, or it is not a control."""

    query = rng.normal(size=DIM)
    seeds = rng.normal(size=(5, DIM))

    raw = offset.raw_query_direction(query)
    assert np.allclose(raw.vector, query / np.linalg.norm(query))
    assert raw.seed_count == 0 and raw.subspace_rank == 0
    assert not raw.degenerate

    subspace = offset.seed_subspace_residual(query, seeds)
    assert not np.allclose(raw.vector, subspace.vector)
    assert abs(float(raw.vector @ subspace.vector)) < 0.99


def test_the_raw_control_and_the_fallback_are_the_same_vector_but_not_the_same_state(rng):
    """A degenerate subspace residual falls back to normalize(q), which is
    exactly the R0 control. The vectors coincide; the records must not, or a
    fallback row would be indistinguishable from a control row."""

    seeds = rng.normal(size=(3, DIM))
    query = seeds.T @ rng.normal(size=3)

    fallback = offset.seed_subspace_residual(query, seeds)
    control = offset.raw_query_direction(query)

    assert np.allclose(fallback.vector, control.vector)
    assert fallback.degenerate and not control.degenerate
    assert fallback.fell_back_to_query and not control.fell_back_to_query


def test_a_zero_query_has_no_raw_direction_either():
    result = offset.raw_query_direction(np.zeros(DIM))
    assert result.degenerate is True
    assert np.allclose(result.vector, 0.0)


def test_ranks_are_one_based_descending_with_the_frozen_tie_break():
    scores = np.array([0.5, 0.9, 0.5, 0.1])
    node_ids = np.array([70, 11, 30, 42])
    ranks = offset.rank_positions(scores, node_ids)

    assert ranks.tolist() == [3, 1, 2, 4], "ties must fall to the lower node id"
    assert sorted(ranks.tolist()) == [1, 2, 3, 4], "ranks are a permutation"


def test_rank_positions_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="differ"):
        offset.rank_positions(np.zeros(3), np.zeros(4))


def test_the_rrf_constant_is_the_projects_frozen_one_not_a_new_choice():
    """If this drifts, the fusion arm has silently acquired a tunable knob."""

    declared = pathlib.Path(REPO_ROOT / "configs" / "candidate_budget.yaml").read_text(
        encoding="utf-8"
    )
    assert f"rrf_constant: {offset.RRF_CONSTANT}" in declared


def test_fusion_rewards_agreement_between_the_two_rankings():
    node_ids = np.arange(4)
    agreed = offset.reciprocal_rank_fusion(
        [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])], node_ids
    )
    assert offset.rank_positions(agreed, node_ids).tolist() == [1, 2, 3, 4]

    # A candidate ranked first by one source and last by the other should not
    # beat one ranked second by both.
    opposed = offset.reciprocal_rank_fusion(
        [np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])], node_ids
    )
    assert opposed[0] == pytest.approx(opposed[3]), "the fusion is symmetric"


def test_fusion_is_exactly_the_declared_formula():
    node_ids = np.arange(3)
    fused = offset.reciprocal_rank_fusion([np.array([1, 2, 3])], node_ids, constant=60)
    for index, rank in enumerate((1, 2, 3)):
        assert fused[index] == pytest.approx(1.0 / (60 + rank))


def test_fusion_refuses_malformed_rankings():
    node_ids = np.arange(3)
    with pytest.raises(ValueError, match="at least one"):
        offset.reciprocal_rank_fusion([], node_ids)
    with pytest.raises(ValueError, match="same candidate set"):
        offset.reciprocal_rank_fusion([np.array([1, 2])], node_ids)
    with pytest.raises(ValueError, match="1-based"):
        offset.reciprocal_rank_fusion([np.array([0, 1, 2])], node_ids)


def test_a_query_s4_already_gets_right_is_not_in_the_error_population():
    """The diagnostic is about mistakes. A correct top-1 has nothing to repair."""

    node_ids = np.arange(4)
    model = np.array([0.9, 0.1, 0.2, 0.3])
    relevant = np.array([True, False, False, False])
    assert (
        offset.error_conditioned_margin(
            "q", model, np.zeros(4), node_ids, relevant, covered=np.ones(4, dtype=bool)
        )
        is None
    )


def test_a_query_with_no_relevant_candidate_scored_is_excluded():
    """No reranker could win it, so counting it would dilute the margin."""

    node_ids = np.arange(4)
    model = np.array([0.9, 0.1, 0.2, 0.3])
    relevant = np.zeros(4, dtype=bool)
    assert (
        offset.error_conditioned_margin(
            "q", model, np.zeros(4), node_ids, relevant, covered=np.ones(4, dtype=bool)
        )
        is None
    )


def test_the_margin_compares_the_best_relevant_against_s4s_top_mistake():
    node_ids = np.arange(5)
    model = np.array([0.9, 0.8, 0.1, 0.2, 0.3])       # S4 ranks index 0 first
    directions = np.array([-0.5, 0.1, 0.7, 0.2, 0.0])  # index 2 is best by direction
    relevant = np.array([False, False, True, True, False])

    result = offset.error_conditioned_margin(
        "q7", model, directions, node_ids, relevant, covered=np.ones(5, dtype=bool)
    )
    assert result is not None
    assert result.wrong_index == 0, "S4's top-ranked candidate is the wrong one"
    assert result.relevant_index == 2, "the best relevant BY DIRECTION"
    assert result.margin == pytest.approx(0.7 - (-0.5))
    # Descending model order is i0, i1, i4, i3, i2, so the first relevant
    # candidate (index 3) sits at rank 4 -- note that is NOT the same candidate
    # the direction prefers, which is the situation the diagnostic is for.
    assert result.first_relevant_rank == 4
    assert result.stratum == "rank_2_5"


def test_the_strata_are_the_declared_bands():
    node_ids = np.arange(30)
    relevant = np.zeros(30, dtype=bool)
    # Strictly descending by index, so index i sits at rank i + 1 and index 0
    # is S4's wrong top-1.
    model = -np.arange(30, dtype=np.float64)
    for position, expected in ((3, "rank_2_5"), (10, "rank_6_20"), (25, "beyond_20")):
        mask = relevant.copy()
        mask[position] = True
        result = offset.error_conditioned_margin(
            "q", model, np.zeros(30), node_ids, mask, covered=np.ones(30, dtype=bool)
        )
        assert result is not None
        assert result.stratum == expected, (position, result.first_relevant_rank)


def test_summarising_an_empty_population_reports_nothing_rather_than_zero():
    """Zero and "no data" are different states, and conflating them would read
    as a measured null when nothing was measured."""

    summary = offset.summarise_margins([])
    assert summary["queries"] == 0
    assert summary["measurable"] == 0
    assert summary["fraction_of_the_error_population_measurable"] is None
    assert summary["fraction_positive"] is None
    assert summary["mean_margin"] is None
    assert summary["median_margin"] is None


def test_the_summary_reports_the_fraction_and_the_strata():
    node_ids = np.arange(4)
    margins = []
    for index, (direction, sign) in enumerate(((0.4, 1), (-0.3, -1), (0.9, 1))):
        model = np.array([1.0, 0.5, 0.4, 0.3])
        directions = np.array([0.0, direction, 0.0, 0.0])
        relevant = np.array([False, True, False, False])
        result = offset.error_conditioned_margin(
            f"q{index}", model, directions, node_ids, relevant, covered=np.ones(4, dtype=bool)
        )
        assert result is not None and np.sign(result.margin) == sign
        margins.append(result)

    summary = offset.summarise_margins(margins)
    assert summary["queries"] == 3
    assert summary["measurable"] == 3
    assert summary["fraction_of_the_error_population_measurable"] == pytest.approx(1.0)
    assert summary["fraction_positive"] == pytest.approx(2 / 3)
    assert summary["mean_margin"] == pytest.approx((0.4 - 0.3 + 0.9) / 3)
    assert summary["by_stratum"]["rank_2_5"]["queries"] == 3
    assert summary["by_stratum"]["beyond_20"]["queries"] == 0
    assert summary["by_stratum"]["beyond_20"]["mean_margin"] is None


# ---------------------------------------------------------------------------
# Coverage is a first-class outcome of the diagnostic, not a preprocessing step
# ---------------------------------------------------------------------------


def _margin(directions, relevant, covered, *, model=None):
    node_ids = np.arange(len(directions))
    model = np.array([1.0, 0.5, 0.4, 0.3]) if model is None else model
    return offset.error_conditioned_margin(
        "q", model, np.asarray(directions, dtype=np.float64), node_ids,
        np.asarray(relevant, dtype=bool), covered=np.asarray(covered, dtype=bool),
    )


def test_a_comparison_with_an_uncovered_side_has_no_margin_rather_than_a_number():
    """The defect that produced this test: the caller masked uncovered
    candidates to -inf before calling, so a comparison between two uncovered
    candidates evaluated -inf minus -inf and every reported mean was NaN.

    Coverage on the real graphs is a few percent, so this is the common case,
    not an edge case, and a sentinel would decide most of the diagnostic.
    """

    result = _margin([0.0, 0.0, 0.0, 0.0], [False, True, False, False], [False] * 4)
    assert result is not None, "the query is still in the error population"
    assert result.margin is None
    assert result.measurable is False
    assert result.relevant_covered is False and result.wrong_covered is False
    assert result.relevant_direction is None and result.wrong_direction is None


def test_the_row_says_which_side_was_missing():
    """"The relevant candidate has directional evidence and S4's mistake has
    none" and the reverse are different findings about the mechanism."""

    only_relevant = _margin(
        [0.0, 0.6, 0.0, 0.0], [False, True, False, False], [False, True, False, False]
    )
    assert only_relevant.relevant_covered is True and only_relevant.wrong_covered is False
    assert only_relevant.relevant_direction == pytest.approx(0.6)
    assert only_relevant.wrong_direction is None and only_relevant.margin is None

    only_wrong = _margin(
        [0.6, 0.0, 0.0, 0.0], [False, True, False, False], [True, False, False, False]
    )
    assert only_wrong.wrong_covered is True and only_wrong.relevant_covered is False
    assert only_wrong.wrong_direction == pytest.approx(0.6)
    assert only_wrong.margin is None


def test_a_covered_relevant_candidate_beats_an_uncovered_one_with_a_higher_score():
    """Same convention as rank_with_coverage: an uncovered candidate's stored
    score is not evidence, so it cannot win the selection on its magnitude."""

    model = np.array([1.0, 0.5, 0.4, 0.3])
    result = _margin(
        [0.0, 9.0, 0.2, 0.0],
        [False, True, True, False],
        [True, False, True, True],
        model=model,
    )
    assert result.relevant_index == 2, "index 1 scores higher but has no coverage"
    assert result.measurable is True
    assert result.margin == pytest.approx(0.2 - 0.0)


def test_the_averages_are_over_measurable_comparisons_and_the_rest_are_counted():
    """A mechanism can have a fine margin where it applies and still be unable
    to touch most of the errors. One mean would let either fact hide the other,
    so the summary reports both and never averages a missing comparison."""

    margins = [
        _margin([0.0, 0.4, 0.0, 0.0], [False, True, False, False], [True, True, True, True]),
        _margin([0.0, 0.0, 0.0, 0.0], [False, True, False, False], [False, False, False, False]),
        _margin([0.0, 0.5, 0.0, 0.0], [False, True, False, False], [False, True, False, False]),
        _margin([0.5, 0.0, 0.0, 0.0], [False, True, False, False], [True, False, False, False]),
    ]
    summary = offset.summarise_margins(margins)

    assert summary["queries"] == 4
    assert summary["measurable"] == 1
    assert summary["fraction_of_the_error_population_measurable"] == pytest.approx(0.25)
    assert summary["unmeasurable_neither_side_covered"] == 1
    assert summary["unmeasurable_only_relevant_covered"] == 1
    assert summary["unmeasurable_only_top_wrong_covered"] == 1
    assert summary["fraction_positive"] == pytest.approx(1.0)
    assert summary["mean_margin"] == pytest.approx(0.4)
    assert summary["by_stratum"]["rank_2_5"]["measurable"] == 1


def test_no_reported_average_is_ever_nan():
    """The regression itself, stated as the property that was violated."""

    margins = [
        _margin([0.0, 0.0, 0.0, 0.0], [False, True, False, False], [False, False, False, False])
        for _ in range(5)
    ]
    summary = offset.summarise_margins(margins)
    blocks = [summary, *summary["by_stratum"].values()]
    reported = [
        block[key]
        for block in blocks
        for key in ("fraction_positive", "mean_margin", "median_margin")
    ]
    assert all(value is None or np.isfinite(value) for value in reported), reported
    assert summary["measurable"] == 0
    assert summary["mean_margin"] is None


def test_coverage_and_the_scores_must_agree_about_which_candidates_have_evidence():
    """A -inf sitting under a True coverage flag is the old defect arriving by
    another route, so it is refused rather than averaged."""

    with pytest.raises(ValueError, match="non-finite direction"):
        _margin([0.0, -np.inf, 0.0, 0.0], [False, True, False, False], [True] * 4)


def test_the_coverage_mask_must_describe_the_same_candidates():
    with pytest.raises(ValueError, match="must agree in shape"):
        offset.error_conditioned_margin(
            "q",
            np.array([1.0, 0.5, 0.4, 0.3]),
            np.zeros(4),
            np.arange(4),
            np.array([False, True, False, False]),
            covered=np.ones(3, dtype=bool),
        )


# ---------------------------------------------------------------------------
# Pool-level directional scoring over a provenance graph.
# ---------------------------------------------------------------------------


def _csr(edges, num_nodes):
    """(source, target) pairs to a CSR the module can walk."""

    rowptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for source, _ in edges:
        rowptr[source + 1] += 1
    np.cumsum(rowptr, out=rowptr)
    col = np.zeros(len(edges), dtype=np.int64)
    cursor = rowptr[:-1].copy()
    for source, target in sorted(edges):
        col[cursor[source]] = target
        cursor[source] += 1
    return rowptr, col


def test_only_edges_present_in_the_provenance_graph_produce_pairs():
    """Swapping G_STRUCT for G_KNN must change this and nothing else."""

    rowptr, col = _csr([(0, 3), (0, 4), (1, 4)], num_nodes=6)
    pool = np.array([3, 4, 5])
    seeds = np.array([0, 1])

    seed_pos, pool_pos = offset.seed_incident_pairs(pool, seeds, rowptr, col)
    assert sorted(zip(seed_pos.tolist(), pool_pos.tolist())) == [(0, 0), (0, 1), (1, 1)]
    assert 2 not in pool_pos.tolist(), "node 5 has no incoming seed edge"


def test_a_pool_with_no_seed_edges_yields_no_pairs():
    rowptr, col = _csr([(0, 9)], num_nodes=10)
    seed_pos, pool_pos = offset.seed_incident_pairs(
        np.array([3, 4]), np.array([0]), rowptr, col
    )
    assert seed_pos.size == 0 and pool_pos.size == 0


def test_empty_pool_or_empty_seeds_is_not_an_error():
    rowptr, col = _csr([(0, 1)], num_nodes=3)
    for pool, seeds in ((np.array([], dtype=np.int64), np.array([0])),
                        (np.array([1]), np.array([], dtype=np.int64))):
        seed_pos, pool_pos = offset.seed_incident_pairs(pool, seeds, rowptr, col)
        assert seed_pos.size == 0 and pool_pos.size == 0


def test_the_fast_form_equals_the_pairwise_definition(rng):
    """The identity the fast path rests on, held against the slow definition.

    ``pool_directional_scores`` never materialises a displacement matrix. If it
    drifted from ``displacements``/``compatibility`` -- the definition M0A's
    admission path shares -- the ranking probe and the admission diagnostic
    would silently be measuring two different quantities.
    """

    num_nodes = 40
    embeddings = rng.normal(size=(num_nodes, DIM))
    residual = offset._unit(rng.normal(size=(1, DIM)))[0][0]
    pool = np.arange(10, 30)
    seeds = np.array([0, 1, 2, 3])
    edges = [(int(s), int(v)) for s in seeds for v in pool if rng.random() < 0.4]
    rowptr, col = _csr(edges, num_nodes)

    seed_pos, pool_pos = offset.seed_incident_pairs(pool, seeds, rowptr, col)
    fast = offset.pool_directional_scores(
        residual, embeddings[pool], embeddings[seeds], seed_pos, pool_pos
    )

    for position, node in enumerate(pool):
        incident = [int(s) for s in seeds if (int(s), int(node)) in edges]
        if not incident:
            assert not fast.covered[position]
            assert fast.dir_max[position] == 0.0 and fast.dir_mean[position] == 0.0
            continue
        deltas, _ = offset._unit(embeddings[node][None, :] - embeddings[incident])
        slow = offset.compatibility(residual, deltas)
        assert fast.support[position] == len(incident)
        assert fast.dir_max[position] == pytest.approx(slow.max(), abs=1e-10)
        assert fast.dir_mean[position] == pytest.approx(slow.mean(), abs=1e-10)


def test_coverage_is_the_fraction_with_any_supported_direction():
    rowptr, col = _csr([(0, 5), (0, 6)], num_nodes=10)
    pool = np.array([5, 6, 7, 8])
    seeds = np.array([0])
    embeddings = np.eye(10, 4)

    seed_pos, pool_pos = offset.seed_incident_pairs(pool, seeds, rowptr, col)
    result = offset.pool_directional_scores(
        np.ones(4), embeddings[pool], embeddings[seeds], seed_pos, pool_pos
    )
    assert result.coverage == pytest.approx(0.5)
    assert result.covered.tolist() == [True, True, False, False]


def test_a_candidate_that_is_its_own_seed_defines_no_direction():
    """A zero displacement is not a direction, and must not become an infinity."""

    rowptr, col = _csr([(2, 2), (2, 3)], num_nodes=5)
    pool = np.array([2, 3])
    seeds = np.array([2])
    embeddings = np.arange(20, dtype=np.float64).reshape(5, 4)

    seed_pos, pool_pos = offset.seed_incident_pairs(pool, seeds, rowptr, col)
    result = offset.pool_directional_scores(
        np.ones(4), embeddings[pool], embeddings[seeds], seed_pos, pool_pos
    )
    assert np.all(np.isfinite(result.dir_max))
    assert not result.covered[0], "the self-edge contributes no usable direction"
    assert result.covered[1]


def test_uncovered_candidates_rank_last_however_good_their_zero_looks():
    """Otherwise the direction-only arm partly measures graph sparsity."""

    node_ids = np.array([10, 11, 12, 13])
    scores = np.array([-0.9, 0.0, 0.5, 0.0])
    covered = np.array([True, False, True, False])

    ranks = offset.rank_with_coverage(scores, covered, node_ids)
    assert ranks[2] == 1, "the best covered candidate leads"
    assert ranks[0] == 2, "a negative but SUPPORTED score still beats no evidence"
    assert set(ranks[[1, 3]].tolist()) == {3, 4}
    assert ranks[1] < ranks[3], "uncovered candidates fall back to ascending node id"


def test_rank_with_coverage_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same candidates"):
        offset.rank_with_coverage(np.zeros(3), np.ones(2, dtype=bool), np.zeros(3))


def test_an_all_negative_candidate_keeps_its_negative_maximum():
    """Regression: accumulating dir_max from zero clamped negatives to 0.0.

    That made a candidate the geometry argues AGAINST look identical to one
    with no evidence at all, which is the exact collapse dir_support_count
    exists to prevent.
    """

    rowptr, col = _csr([(0, 2)], num_nodes=4)
    pool = np.array([2])
    seeds = np.array([0])
    embeddings = np.zeros((4, 3))
    embeddings[0] = [1.0, 0.0, 0.0]
    embeddings[2] = [0.0, 0.0, 0.0]  # displacement points back along -x

    seed_pos, pool_pos = offset.seed_incident_pairs(pool, seeds, rowptr, col)
    result = offset.pool_directional_scores(
        np.array([1.0, 0.0, 0.0]), embeddings[pool], embeddings[seeds], seed_pos, pool_pos
    )
    assert result.covered[0]
    assert result.dir_max[0] == pytest.approx(-1.0)
    assert result.dir_mean[0] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# The +64 admission diagnostic: one variable, held against the frozen path.
# ---------------------------------------------------------------------------


def _expansion_fixture(rng, num_nodes=60, pool_size=20):
    from mp_retrieval.candidate_expansion_v2 import ExpansionBudget

    embeddings = rng.normal(size=(num_nodes, 16))
    pool = np.arange(pool_size)
    seeds = np.array([0, 1, 2])
    edges = [
        (int(s), int(v))
        for s in seeds
        for v in range(num_nodes)
        if v not in pool.tolist() and rng.random() < 0.3
    ]
    rowptr, col = _csr(edges, num_nodes)
    budget = ExpansionBudget(
        per_seed_cap=16, graph_expansion_cap=64, neighbour_scan_cap_per_seed=4096
    )
    return embeddings, pool, seeds, rowptr, col, budget


def test_the_supplied_residual_path_reproduces_the_frozen_directional_expansion(rng):
    """The equality the whole admission comparison rests on.

    If this breaks, the legacy arm is no longer the mechanism M0A nulled and
    measuring it against that null means nothing.
    """

    from mp_retrieval import candidate_expansion_v2 as cx

    embeddings, pool, seeds, rowptr, col, budget = _expansion_fixture(rng)
    query = rng.normal(size=16)
    anchor = 3

    frozen = cx.expand(
        cx.DIRECTIONAL,
        rowptr=rowptr, col=col, node_embeddings=embeddings,
        query_embedding=query, anchor=anchor, pool=pool, seeds=seeds,
        budget=budget, num_nodes=60,
    )
    ours = offset.expand_with_residual(
        cx.query_residual(query, embeddings[anchor]),
        rowptr=rowptr, col=col, node_embeddings=embeddings,
        anchor=anchor, pool=pool, seeds=seeds, budget=budget, num_nodes=60,
    )

    assert ours.admitted.tolist() == frozen.admitted.tolist()
    assert np.allclose(ours.scores, frozen.scores)
    assert ours.matched_pool.tolist() == frozen.matched_pool.tolist()
    assert ours.additive_pool.tolist() == frozen.additive_pool.tolist()
    assert ours.evicted.tolist() == frozen.evicted.tolist()
    assert ours.neighbours_scanned == frozen.neighbours_scanned
    assert ours.seeds_at_the_per_seed_cap == frozen.seeds_at_the_per_seed_cap
    assert ours.zero_displacement_edges == frozen.zero_displacement_edges
    assert ours.method != frozen.method, "the arm must still be labelled as its own"


def test_a_different_residual_is_free_to_admit_a_different_set(rng):
    """Otherwise the seed-subspace arm could not be a distinct variable at all."""

    embeddings, pool, seeds, rowptr, col, budget = _expansion_fixture(rng)
    query = rng.normal(size=16)

    first = offset.expand_with_residual(
        offset.raw_query_direction(query).vector,
        rowptr=rowptr, col=col, node_embeddings=embeddings,
        anchor=3, pool=pool, seeds=seeds, budget=budget, num_nodes=60,
    )
    second = offset.expand_with_residual(
        offset.seed_subspace_residual(query, embeddings[seeds]).vector,
        rowptr=rowptr, col=col, node_embeddings=embeddings,
        anchor=3, pool=pool, seeds=seeds, budget=budget, num_nodes=60,
    )
    assert first.admitted.size and second.admitted.size
    assert first.scores.tolist() != second.scores.tolist()


def test_a_degenerate_residual_admits_nothing_rather_than_admitting_arbitrarily(rng):
    embeddings, pool, seeds, rowptr, col, budget = _expansion_fixture(rng)
    result = offset.expand_with_residual(
        np.zeros(16),
        rowptr=rowptr, col=col, node_embeddings=embeddings,
        anchor=3, pool=pool, seeds=seeds, budget=budget, num_nodes=60,
    )
    assert result.degenerate_residual is True
    assert result.admitted.size == 0
    assert result.matched_pool.tolist() == pool.tolist()


def test_admission_overlap_reports_both_directions():
    result = offset.admission_overlap(np.array([1, 2, 3]), np.array([2, 3, 4]))
    assert result["intersection"] == 2 and result["union"] == 4
    assert result["jaccard"] == pytest.approx(0.5)
    assert result["unique_to_left"] == 1 and result["unique_to_right"] == 1

    identical = offset.admission_overlap(np.array([5, 6]), np.array([6, 5]))
    assert identical["jaccard"] == 1.0 and identical["unique_to_left"] == 0


def test_two_arms_that_admit_nothing_agree_rather_than_being_undefined():
    """A 0/0 Jaccard reported as 0 would read as total disagreement."""

    empty = np.zeros(0, dtype=np.int64)
    assert offset.admission_overlap(empty, empty)["jaccard"] == 1.0
