"""The semantic frontier's frozen claims, pinned before any result exists.

``docs/QLS_V2_FEATURE_CATALOG.md`` commits to specific numbers and to a specific
initialization *in advance*. Several of those commitments are falsifiable
statements about the code rather than prose, and each one below is a claim the
document would be wrong about if the test failed.

The initialization claims matter most. The catalog argues that S3 "starts as S2
plus one redundant channel and can only depart from it by learning", and that
argument is what licenses reading an S3-over-S2 gain as evidence that a learned
semantic comparison helps. If S3 began anywhere else, part of the gain could be
the extra channels merely existing, and the comparison would not be clean.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mp_retrieval.qls_v2_semantic import (  # noqa: E402
    EMBEDDING_DIM,
    PARAMETER_FREE_FEATURE_NAMES,
    RUNG_FEATURES,
    RUNG_PARAMETERS,
    RUNGS,
    SEMANTIC_FEATURE_NAMES,
    V1_SEMANTIC_PARAMETERS,
    SemanticHead,
    parameter_free_scalars,
    within_query_percentile,
)


def _pair(n: int = 6, dim: int = EMBEDDING_DIM, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(dim, generator=generator, dtype=torch.float64)
    candidates = torch.randn(n, dim, generator=generator, dtype=torch.float64)
    return query, candidates


def _column(head: SemanticHead, name: str, query, candidates):
    return head(query, candidates)[:, head.feature_names.index(name)]


# --------------------------------------------------------------------------
# The frozen counts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rung", RUNGS)
def test_each_rung_has_exactly_the_parameters_the_catalog_promises(rung):
    assert SemanticHead(rung).parameter_count() == RUNG_PARAMETERS[rung]


def test_s0_through_s2_hold_no_parameters_at_all():
    """Not an optimization -- it is what those rungs are for.

    S0-S2 exist to ask whether a learned semantic comparison is needed before
    one is paid for. A single parameter anywhere in them would make the question
    unanswerable.
    """
    for rung in ("S0", "S1", "S2"):
        assert list(SemanticHead(rung).parameters()) == []


def test_the_reduction_against_v1_is_the_stated_64x():
    assert V1_SEMANTIC_PARAMETERS == 98_304
    assert RUNG_PARAMETERS["S3"] == 1_536
    assert V1_SEMANTIC_PARAMETERS / RUNG_PARAMETERS["S3"] == pytest.approx(64.0)


def test_the_rungs_are_strictly_nested():
    """A frontier whose rungs are not nested measures nothing comparable."""
    for lower, upper in zip(RUNGS, RUNGS[1:]):
        assert set(RUNG_FEATURES[lower]) < set(RUNG_FEATURES[upper])
    assert RUNG_FEATURES["S3"] == SEMANTIC_FEATURE_NAMES


# --------------------------------------------------------------------------
# The initialization, which the catalog calls "not free choice"
# --------------------------------------------------------------------------


def test_at_initialization_the_product_feature_is_the_mean_elementwise_product():
    """F4 with ``w_i = 1/768`` is exactly ``<q, d> / 768``.

    The catalog's claim is that this is a monotone function of ``<q, d>`` and so
    carries nothing S2 did not already have.
    """
    query, candidates = _pair()
    head = SemanticHead("S3").double()
    product = _column(head, "semantic_product", query, candidates)
    assert torch.allclose(product, (candidates @ query) / EMBEDDING_DIM)


def test_at_initialization_the_difference_feature_is_identically_zero():
    query, candidates = _pair()
    head = SemanticHead("S3").double()
    difference = _column(head, "semantic_difference", query, candidates)
    assert torch.equal(difference, torch.zeros_like(difference))


def test_an_inert_feature_still_trains():
    """``v = 0`` makes F5 zero but not stuck: dF5/dv_i = |q_i - d_i| != 0.

    A zero-initialized weight is usually a bug because it kills the gradient.
    Here it does not, and the catalog says so explicitly -- so the gradient is
    checked rather than assumed.
    """
    query, candidates = _pair()
    head = SemanticHead("S3").double()
    _column(head, "semantic_difference", query, candidates).sum().backward()

    gradient = head.difference_weight.grad
    assert gradient is not None
    assert torch.allclose(gradient, (candidates - query).abs().sum(dim=0))
    assert (gradient.abs() > 0).all()


def test_s3_at_initialization_is_s2_plus_one_redundant_and_one_dead_channel():
    """The claim that licenses attributing any S3 gain to learning.

    Everything S2 computes must survive unchanged into S3, and the two new
    columns must add no information at step zero: one is a rescaling of a
    quantity S2 already ranks by, the other is constant.
    """
    query, candidates = _pair(n=8, seed=3)
    shared = SemanticHead("S2").double()(query, candidates)
    full = SemanticHead("S3").double()(query, candidates)

    assert torch.allclose(full[:, :3], shared)

    dot = candidates @ query
    product, difference = full[:, 3], full[:, 4]
    # Strictly increasing in the dot product, hence rank-identical to it.
    assert torch.equal(torch.argsort(product), torch.argsort(dot))
    assert difference.unique().numel() == 1


# --------------------------------------------------------------------------
# What each parameter-free feature is, and is not
# --------------------------------------------------------------------------


def test_cosine_ignores_candidate_magnitude_but_the_product_feature_does_not():
    """F1 and F4-at-init are both monotone in similarity yet differ in kind.

    Scaling a candidate leaves the cosine alone and scales the dot product. If
    F4 were also scale-free the two channels would be the same feature twice.
    """
    query, candidates = _pair(n=4, seed=5)
    head = SemanticHead("S3").double()
    scaled = candidates.clone()
    scaled[0] *= 7.0

    before, after = head(query, candidates), head(query, scaled)
    assert torch.allclose(before[0, 0], after[0, 0])          # cosine_qd
    assert not torch.allclose(before[0, 3], after[0, 3])      # semantic_product


def test_mean_abs_diff_is_the_mean_over_the_embedding_dimension():
    query = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=torch.float64)
    candidates = torch.tensor([[1.0, 2.0, 3.0, 2.0]], dtype=torch.float64)
    scalars = parameter_free_scalars(query, candidates)
    index = PARAMETER_FREE_FEATURE_NAMES.index("mean_abs_diff")
    assert scalars[0, index].item() == pytest.approx(1.0)


def test_the_percentile_depends_on_the_candidate_set_not_the_pair_alone():
    """F2 is a *within-query* quantity; that is the whole of its content."""
    query = torch.tensor([1.0, 0.0], dtype=torch.float64)
    target = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    index = PARAMETER_FREE_FEATURE_NAMES.index("dot_qd_pct")

    weak = torch.cat([target, torch.tensor([[0.5, 0.0]], dtype=torch.float64)])
    strong = torch.cat([target, torch.tensor([[9.0, 0.0]], dtype=torch.float64)])

    assert parameter_free_scalars(query, weak)[0, index].item() == pytest.approx(1.0)
    assert parameter_free_scalars(query, strong)[0, index].item() == pytest.approx(0.0)


# --------------------------------------------------------------------------
# The tie rule, which the catalog leaves open and the code must not
# --------------------------------------------------------------------------


def test_tied_scores_share_an_averaged_rank_so_order_cannot_matter():
    """Ordinal ranking would make this feature depend on list position.

    ``argsort`` resolves ties by index, so two candidates with identical dot
    products would receive different percentiles decided by where they happened
    to sit in the candidate array. Averaging makes the value a function of the
    multiset alone.
    """
    tied = torch.tensor([5.0, 1.0, 5.0, 3.0], dtype=torch.float64)
    percentile = within_query_percentile(tied)
    assert percentile[0].item() == pytest.approx(percentile[2].item())
    assert percentile[1].item() == pytest.approx(0.0)
    assert percentile[0].item() == pytest.approx(5.0 / 6.0)


def test_permuting_the_candidates_permutes_the_percentiles_and_changes_nothing_else():
    values = torch.tensor([2.0, 2.0, 9.0, -1.0, 2.0], dtype=torch.float64)
    order = torch.tensor([3, 0, 4, 2, 1])
    direct = within_query_percentile(values)[order]
    permuted = within_query_percentile(values[order])
    assert torch.allclose(direct, permuted)


def test_a_lone_candidate_scores_the_all_tied_midpoint():
    """0.5 is the limit of the tied-group rule, not a separate convention."""
    assert within_query_percentile(torch.tensor([4.2])).item() == pytest.approx(0.5)
    allsame = within_query_percentile(torch.full((7,), 4.2, dtype=torch.float64))
    assert torch.allclose(allsame, torch.full_like(allsame, 0.5))


def test_the_percentile_survives_any_increasing_rescaling_of_the_scores():
    """A rank feature must not track the units of what it ranks."""
    values = torch.tensor([-3.0, 0.25, 7.0, 1.5], dtype=torch.float64)
    assert torch.allclose(
        within_query_percentile(values),
        within_query_percentile(values * 3.0 + 11.0),
    )


# --------------------------------------------------------------------------
# The capacity claim
# --------------------------------------------------------------------------


def test_the_difference_feature_is_not_expressible_as_any_bilinear_form():
    """New capacity, not only cheaper capacity -- checked, not asserted.

    Every bilinear form ``q^T W d`` is linear in `d`, so scaling a candidate
    scales its score by the same factor. F5 is built on ``|q - d|`` and does
    not, which is why the v1 projection could not express it at any rank.
    """
    query, candidates = _pair(n=1, seed=11)
    head = SemanticHead("S3").double()
    with torch.no_grad():
        head.difference_weight.copy_(torch.ones(EMBEDDING_DIM, dtype=torch.float64))

    single = _column(head, "semantic_difference", query, candidates)
    doubled = _column(head, "semantic_difference", query, candidates * 2.0)
    assert not torch.allclose(doubled, 2.0 * single)


def test_the_product_feature_is_bilinear_as_the_diagonal_restriction_should_be():
    """The companion check: F4 *is* linear in `d`, being ``q^T diag(w) d``."""
    query, candidates = _pair(n=1, seed=13)
    head = SemanticHead("S3").double()
    single = _column(head, "semantic_product", query, candidates)
    doubled = _column(head, "semantic_product", query, candidates * 2.0)
    assert torch.allclose(doubled, 2.0 * single)


# --------------------------------------------------------------------------
# Shapes and refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rung", RUNGS)
def test_each_rung_emits_exactly_its_own_columns(rung):
    query, candidates = _pair(n=5, seed=7)
    head = SemanticHead(rung).double()
    assert head(query, candidates).shape == (5, len(RUNG_FEATURES[rung]))


def test_a_mismatched_embedding_width_is_refused_rather_than_broadcast():
    """Silent broadcasting here would produce a plausible wrong number."""
    with pytest.raises(ValueError, match="does not match"):
        parameter_free_scalars(torch.randn(768), torch.randn(4, 512))


def test_an_unknown_rung_is_refused():
    with pytest.raises(ValueError, match="unknown rung"):
        SemanticHead("S4")
