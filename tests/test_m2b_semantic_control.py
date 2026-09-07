"""Every M2B parameter count, derived from a live model rather than read.

The failure this module exists to catch already happened once in prose.
``qls_v2_semantic.V1_SEMANTIC_PARAMETERS`` is 98,304, computed from that
module's 768-dimensional default, and the planning around this track repeated
"the historical projection is about 98K" for months while every fit ran against
a 1536-dimensional payload. The real cost at the real width is 196,608. A
declaration that transcribed the constant would have understated its own
control by a factor of two and every reduction ratio in the paper with it.

So these tests never compare one filed number against another filed number.
They instantiate ``M1AScorer`` with each candidate head and count with
``numel()``. The declaration is checked against the model; the model is never
checked against the declaration.

One test does more than count: S3 rebuilt here must reproduce the 3,585
parameters that fifteen completed M2 fits actually ran under. That is the row
with evidence behind it, and it is what licenses believing the other two.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mp_retrieval.m1a_screen import M1AScorer, build_m1a_model  # noqa: E402
from mp_retrieval.m2b_semantic_control import (  # noqa: E402
    PROJECTION_DIM,
    RUNG,
    ProjectionSemanticHead,
    feature_names,
)
from mp_retrieval.qls_v2_semantic import (  # noqa: E402
    EMBEDDING_DIM,
    RUNG_FEATURES,
    V1_SEMANTIC_PARAMETERS,
    SemanticHead,
)

M2_CONFIG = pathlib.Path("configs/m2_qls_v2_freeze.yaml")
M2B_CONFIG = pathlib.Path("configs/m2b_semantic_minimality.yaml")

#: The width every M1A/M1B/M2 fit ran at, and the only width these tests care
#: about. Deliberately not EMBEDDING_DIM, which is the module default and is
#: the source of the confusion above.
FROZEN_DIM = 1536
PRECOMPUTED_WIDTH = 9
DROPOUT = 0.2
TEMPERATURE = 0.07


def head_for(rung: str, dim: int = FROZEN_DIM) -> torch.nn.Module:
    if rung == RUNG:
        return ProjectionSemanticHead(dim=dim)
    return SemanticHead(rung=rung, dim=dim)


def model_for(rung: str, dim: int = FROZEN_DIM) -> M1AScorer:
    return M1AScorer(
        precomputed_width=PRECOMPUTED_WIDTH,
        semantic_rung=rung,
        dropout=DROPOUT,
        temperature=TEMPERATURE,
        embedding_dim=dim,
        semantic_head=head_for(rung, dim),
    )


@pytest.fixture(scope="module")
def m2b_declaration() -> dict:
    return yaml.safe_load(M2B_CONFIG.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The trap
# --------------------------------------------------------------------------


def test_the_historical_constant_is_a_768_fact_and_not_this_tracks_number():
    """The whole reason nothing in M2B is transcribed."""

    at_768 = ProjectionSemanticHead(dim=EMBEDDING_DIM).parameter_count()
    at_1536 = ProjectionSemanticHead(dim=FROZEN_DIM).parameter_count()

    assert at_768 == V1_SEMANTIC_PARAMETERS, (
        "the control is supposed to be formula-identical to v1's projection; at v1's own width "
        "it must reproduce v1's own constant, or it is not a control"
    )
    assert at_1536 == 2 * at_768
    assert at_1536 != V1_SEMANTIC_PARAMETERS, (
        "if these were equal the constant would be safe to quote and this whole discipline "
        "would be unnecessary"
    )


def test_the_declaration_files_the_measured_number_not_the_constant(m2b_declaration):
    filed = m2b_declaration["semantic_candidates"]["S4"]
    assert filed["semantic_trainable_parameters"] == ProjectionSemanticHead(
        dim=FROZEN_DIM
    ).parameter_count()
    assert filed["semantic_trainable_parameters"] != V1_SEMANTIC_PARAMETERS
    assert filed["historical_status"] == "CURRENT_DIMENSION_PROJECTION_CONTROL"


def test_s4_is_not_labelled_historically_exact(m2b_declaration):
    """The label has to say which of the two things this rung is."""

    filed = m2b_declaration["semantic_candidates"]["S4"]
    assert filed["historical_status"] != "HISTORICAL_EXACT"
    prose = filed["not_historical_exact"]
    assert "768" in prose and "1536" in prose
    assert str(V1_SEMANTIC_PARAMETERS) in prose.replace(",", "")


# --------------------------------------------------------------------------
# Counts, from live instances
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rung", ["S2", "S3", "S4"])
def test_the_declared_counts_are_what_a_live_model_holds(rung, m2b_declaration):
    model = model_for(rung)
    filed = m2b_declaration["semantic_candidates"][rung]

    assert filed["raw_embedding_dimension"] == FROZEN_DIM
    assert filed["semantic_output_width"] == len(model.semantic_head.feature_names)
    assert filed["semantic_trainable_parameters"] == model.semantic_parameter_count()
    assert filed["scorer_trainable_parameters"] == model.scorer_parameter_count()
    assert filed["total_trainable_parameters"] == model.trainable_parameter_count()
    assert filed["scorer_input_width"] == model.scorer[0].in_features


@pytest.mark.parametrize("rung", ["S2", "S3", "S4"])
def test_the_branch_emits_exactly_the_columns_it_names(rung):
    """A width taken from ``feature_names`` is only right if the forward agrees."""

    head = head_for(rung)
    with torch.no_grad():
        emitted = head(torch.randn(FROZEN_DIM), torch.randn(7, FROZEN_DIM))
    assert emitted.shape == (7, len(head.feature_names))


@pytest.mark.parametrize("rung", ["S2", "S3", "S4"])
def test_the_scorer_width_is_the_structural_block_plus_the_branch(rung):
    model = model_for(rung)
    assert model.scorer[0].in_features == PRECOMPUTED_WIDTH + len(
        model.semantic_head.feature_names
    )


@pytest.mark.parametrize("rung", ["S2", "S3", "S4"])
def test_the_scorer_obeys_the_declared_formula(rung, m2b_declaration):
    """32 * width + 65, the formula M2 verified against three real arms."""

    model = model_for(rung)
    width = model.scorer[0].in_features
    assert model.scorer_parameter_count() == 32 * width + 65
    assert m2b_declaration["semantic_candidates"]["scorer_parameters_formula"] == (
        "32 * width + 65"
    )


def test_s3_reproduces_the_count_fifteen_completed_fits_ran_under():
    """The row with evidence behind it, checked against M2's own declaration."""

    m2 = yaml.safe_load(M2_CONFIG.read_text(encoding="utf-8"))
    filed = m2["qls_universal"]["parameter_count"]
    model = model_for("S3")

    assert model.semantic_parameter_count() == filed["semantic"] == 3072
    assert model.scorer_parameter_count() == filed["scorer"] == 513
    assert model.trainable_parameter_count() == filed["total"] == 3585


def test_the_ladder_is_strictly_increasing(m2b_declaration):
    totals = [model_for(rung).trainable_parameter_count() for rung in ("S2", "S3", "S4")]
    assert totals == sorted(totals) and len(set(totals)) == 3

    ladder = m2b_declaration["semantic_candidates"]["parameter_ladder"]
    assert [ladder["S2"], ladder["S3"], ladder["S4"]] == totals
    assert ladder["S4_over_S3"] == pytest.approx(totals[2] / totals[1], abs=0.05)
    assert ladder["S3_over_S2"] == pytest.approx(totals[1] / totals[0], abs=0.05)


def test_s2_learns_no_semantic_parameter_at_all():
    """The claim that makes S2 interesting, asserted against the model."""

    model = model_for("S2")
    assert model.semantic_parameter_count() == 0
    assert not list(model.semantic_head.parameters())
    assert model.trainable_parameter_count() == model.scorer_parameter_count()


# --------------------------------------------------------------------------
# The control's shape and provenance
# --------------------------------------------------------------------------


def test_the_control_matches_the_historical_formulas():
    """Bias-free projections at v1's width -- the parts that make it a control."""

    head = ProjectionSemanticHead(dim=FROZEN_DIM)
    assert head.projection_dim == PROJECTION_DIM == 64
    for projection in (head.query_projection, head.node_projection):
        assert projection.bias is None, (
            "a bias here would be a quiet 128-parameter departure from the projection this "
            "rung exists to control for"
        )
        assert projection.weight.shape == (PROJECTION_DIM, FROZEN_DIM)
    assert len(head.feature_names) == 4 * PROJECTION_DIM + 2


def test_the_control_scales_with_the_payload_not_with_a_constant():
    for dim in (768, 1024, 1536, 3072):
        assert ProjectionSemanticHead(dim=dim).parameter_count() == 2 * dim * PROJECTION_DIM


def test_feature_names_are_unique_and_ordered():
    names = feature_names()
    assert len(names) == len(set(names))
    assert names[-2:] == ("normalized_state_dot", "raw_projection_dot_scaled")


def test_the_control_is_not_a_rung_in_the_nested_ladder():
    """RUNG_FEATURES is a strict chain; a projection block would break it."""

    assert RUNG not in RUNG_FEATURES
    assert not set(feature_names()) & set(RUNG_FEATURES["S3"])


def test_the_control_refuses_a_mismatched_query():
    head = ProjectionSemanticHead(dim=FROZEN_DIM)
    with pytest.raises(ValueError):
        head(torch.randn(768), torch.randn(4, FROZEN_DIM))
    with pytest.raises(ValueError):
        head(torch.randn(FROZEN_DIM), torch.randn(FROZEN_DIM))


def test_the_control_rejects_a_nonsense_width():
    with pytest.raises(ValueError):
        ProjectionSemanticHead(dim=0)
    with pytest.raises(ValueError):
        ProjectionSemanticHead(dim=FROZEN_DIM, projection_dim=0)


# --------------------------------------------------------------------------
# The injection point must change nothing for anyone else
# --------------------------------------------------------------------------


def test_omitting_the_head_builds_exactly_what_m2_built():
    """Every M1A/M1B/M2 caller passes nothing; that path must be untouched."""

    default = M1AScorer(
        precomputed_width=PRECOMPUTED_WIDTH,
        semantic_rung="S3",
        dropout=DROPOUT,
        temperature=TEMPERATURE,
        embedding_dim=FROZEN_DIM,
    )
    assert isinstance(default.semantic_head, SemanticHead)
    assert default.semantic_head.rung == "S3"
    assert default.trainable_parameter_count() == 3585
    assert default.scorer[0].in_features == 14


def test_the_builder_threads_the_head_through():
    model = build_m1a_model(
        precomputed_width=PRECOMPUTED_WIDTH,
        semantic_rung=RUNG,
        dropout=DROPOUT,
        temperature=TEMPERATURE,
        embedding_dim=FROZEN_DIM,
        semantic_head=ProjectionSemanticHead(dim=FROZEN_DIM),
    )
    assert isinstance(model.semantic_head, ProjectionSemanticHead)
    assert model.trainable_parameter_count() == 205217


def test_a_head_recorded_under_the_wrong_rung_is_refused():
    """A fit filed under the wrong rung name is unrecoverable after the fact."""

    with pytest.raises(ValueError, match="rung"):
        M1AScorer(
            precomputed_width=PRECOMPUTED_WIDTH,
            semantic_rung="S3",
            dropout=DROPOUT,
            temperature=TEMPERATURE,
            embedding_dim=FROZEN_DIM,
            semantic_head=ProjectionSemanticHead(dim=FROZEN_DIM),
        )


def test_the_model_runs_end_to_end_with_each_head():
    """Shape agreement all the way to a score, not only at the branch."""

    for rung in ("S2", "S3", "S4"):
        model = model_for(rung)
        nodes = torch.randn(5, FROZEN_DIM)
        queries = torch.randn(2, FROZEN_DIM)
        batch_index = torch.tensor([0, 0, 0, 1, 1])
        structural = torch.randn(5, PRECOMPUTED_WIDTH)
        model.eval()
        with torch.no_grad():
            scores = model.forward_explicit(nodes, queries, batch_index, structural)
        assert scores.shape == (5,)
        assert torch.isfinite(scores).all()
