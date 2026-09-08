"""The two Stage-1 arms, and the thing that makes transcribing a formula safe.

``m2d_stage1_arms`` re-states three tensor expressions that already exist in
``qls_v2_semantic``, for a reason its docstring gives: importing them would
either edit frozen science or bill each arm for columns it does not have. Two
copies of one formula can drift, so drift is made loud here rather than
promised away.

Two guards do that, and they fail in different ways on purpose.

**Exact equality against the live modules.** ``torch.equal``, not
``allclose``: a transcription that agreed to six decimals would be a different
formula, and this phase is about a column whose whole content is fine-grained
dimension-wise distance. Run across shapes, seeds, dtypes and the degenerate
cases -- one candidate, identical vectors, exact ties -- where a plausible
mis-transcription would still look right on random floats.

**The live source still says what was copied.** Equality only compares the two
implementations to each other; if someone edits the live formula, both this
file's transcription and the assertion that they match would happily move
together. So the expressions themselves are asserted to still be present in
the live source. That test failing does not mean the arms are wrong -- it means
the thing they were copied from changed and a human has to decide what that
means.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval import m2d_stage1_arms as arms
from mp_retrieval.m2b_semantic_control import PROJECTION_DIM, ProjectionSemanticHead
from mp_retrieval.qls_v2_semantic import (
    PARAMETER_FREE_FEATURE_NAMES,
    SemanticHead,
    parameter_free_scalars,
)

#: The width every track fit runs at. Not qls_v2_semantic.EMBEDDING_DIM, which
#: is 768 and is the default no fit has ever used.
LIVE_DIM = 1536

#: How many precomputed structural columns the scorer actually sees in the
#: fits, derived from M2B's immutable baseline table rather than typed.
#:
#: This is not decoration. An earlier version of the parameter test below built
#: at a made-up width of 40 under a docstring claiming it was the width the
#: fits run at, and every absolute count it asserted described a store that has
#: never existed. The arithmetic here is the compute record's
#: ``structural_width``: the scorer is
#: ``Linear(width + columns, HEAD_WIDTH) + Linear(HEAD_WIDTH, 1)``, so a filed
#: fit's recorded semantic and total counts pin the width exactly.
BASELINE_TABLE = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)
SEMANTIC_COLUMNS = {"S2": 3, "S3": 5, "S4": 258}


def _live_precomputed_width() -> int:
    import json

    from mp_retrieval.m1a_screen import HEAD_WIDTH

    table = json.loads(BASELINE_TABLE.read_text(encoding="utf-8"))
    widths = set()
    for row in table["rows"]:
        scorer = row["total_parameters"] - row["semantic_parameters"]
        columns, remainder = divmod(scorer - HEAD_WIDTH - HEAD_WIDTH - 1, HEAD_WIDTH)
        assert remainder == 0, row
        widths.add(columns - SEMANTIC_COLUMNS[row["rung"]])
    assert len(widths) == 1, f"the filed fits disagree about the width: {sorted(widths)}"
    return widths.pop()


LIVE_PRECOMPUTED_WIDTH = _live_precomputed_width()

SHAPES = [(1, 8), (2, 8), (5, 64), (64, 128)]


def pair(n: int, dim: int, seed: int, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(dim, generator=generator, dtype=dtype)
    candidates = torch.randn(n, dim, generator=generator, dtype=dtype)
    return query, candidates


def live_difference_column(query, candidates, weight) -> torch.Tensor:
    """The column as the live S3 module produces it, through its own forward."""

    head = SemanticHead("S3", dim=weight.numel()).to(weight.dtype)
    with torch.no_grad():
        head.difference_weight.copy_(weight)
        columns = head(query, candidates)
    return columns[:, head.feature_names.index(arms.DIFFERENCE_NAME)]


# ---------------------------------------------------------------------------
# The transcriptions are the live formulas, exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,dim", SHAPES)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_raw_scalars_are_the_live_columns_bit_for_bit(n, dim, seed) -> None:
    query, candidates = pair(n, dim, seed)
    live = parameter_free_scalars(query, candidates)
    ours = arms.raw_embedding_scalars(query, candidates)

    assert ours.shape == (n, 2)
    for index, name in enumerate(arms.RAW_SCALAR_NAMES):
        assert torch.equal(ours[:, index], live[:, PARAMETER_FREE_FEATURE_NAMES.index(name)])


@pytest.mark.parametrize("n,dim", SHAPES)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_difference_column_is_the_live_column_bit_for_bit(n, dim, seed) -> None:
    query, candidates = pair(n, dim, seed)
    weight = torch.randn(dim, generator=torch.Generator().manual_seed(seed + 100))
    ours = arms.semantic_difference_column(query, candidates, weight)

    assert ours.shape == (n,)
    assert torch.equal(ours, live_difference_column(query, candidates, weight))


def test_the_columns_agree_where_random_floats_would_hide_a_mistake() -> None:
    """Degenerate inputs, where a wrong reduction or a wrong axis still looks
    plausible: one candidate, a candidate identical to the query, exact ties,
    and a zero vector that would divide by zero without the live clamp."""

    dim = 16
    query = torch.arange(dim, dtype=torch.float32)
    candidates = torch.stack(
        [
            query.clone(),
            torch.zeros(dim),
            torch.full((dim,), 3.0),
            torch.full((dim,), 3.0),
            -query,
        ]
    )
    for subset in (candidates[:1], candidates[:2], candidates):
        live = parameter_free_scalars(query, subset)
        ours = arms.raw_embedding_scalars(query, subset)
        for index, name in enumerate(arms.RAW_SCALAR_NAMES):
            assert torch.equal(
                ours[:, index], live[:, PARAMETER_FREE_FEATURE_NAMES.index(name)]
            )

        weight = torch.linspace(-1.0, 1.0, dim)
        assert torch.equal(
            arms.semantic_difference_column(query, subset, weight),
            live_difference_column(query, subset, weight),
        )

    # The zero candidate: the live clamp is what keeps cosine finite, and a
    # transcription that dropped it would produce nan here rather than 0.
    assert torch.isfinite(arms.raw_embedding_scalars(torch.zeros(dim), candidates)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_neither_transcription_pins_a_dtype(dtype) -> None:
    query, candidates = pair(4, 32, 3, dtype=dtype)
    weight = torch.randn(32, generator=torch.Generator().manual_seed(4)).to(dtype)
    assert arms.raw_embedding_scalars(query, candidates).dtype == dtype
    assert arms.semantic_difference_column(query, candidates, weight).dtype == dtype


def test_the_live_source_still_contains_what_was_copied() -> None:
    """Equality alone cannot see a change made to both sides.

    If this fails, the arms are not necessarily wrong: the formula they were
    transcribed from moved, and someone has to decide whether the arms move
    with it. That is a human decision, which is why it is a separate failure
    from the equality tests above.
    """

    def lines(function) -> set[str]:
        return {line.strip() for line in inspect.getsource(function).splitlines()}

    # Whole lines, not substrings. Containment would accept anything appended
    # to the end of the live expression -- a scale, a clamp, a second term --
    # and an appended term that only bites on inputs these tests do not
    # generate would slip past the equality checks too.
    assert (
        "columns.append((candidates - query).abs() @ self.difference_weight)"
        in lines(SemanticHead.forward)
    )

    scalars = lines(parameter_free_scalars)
    for line in (
        "unit_query = query / query.norm().clamp_min(EPS)",
        "unit_candidates = candidates / candidates.norm(dim=1, keepdim=True).clamp_min(EPS)",
        "cosine = unit_candidates @ unit_query",
        "mean_abs_diff = (candidates - query).abs().mean(dim=1)",
    ):
        assert line in scalars, line


# ---------------------------------------------------------------------------
# The arms are S4 with columns appended, and nothing else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", [arms.CONTROL_ARM, arms.CANDIDATE_ARM])
def test_the_s4_branch_inside_an_arm_is_the_live_class(arm) -> None:
    head = arms.build_arm_head(arm, dim=64)
    assert isinstance(head.projection, ProjectionSemanticHead)
    assert head.projection.rung == "S4"
    assert head.projection.projection_dim == PROJECTION_DIM


@pytest.mark.parametrize("arm", [arms.CONTROL_ARM, arms.CANDIDATE_ARM])
def test_an_arm_does_not_perturb_the_s4_columns_it_wraps(arm) -> None:
    """The first 258 columns must be what native S4 produces, unchanged.

    Built under the same seed, so the projection consumes the same random
    numbers: if the arm reordered its own construction or touched the
    projection's parameters, these would part company.
    """

    dim, n = 64, 6
    torch.manual_seed(11)
    head = arms.build_arm_head(arm, dim=dim)
    torch.manual_seed(11)
    native = ProjectionSemanticHead(dim=dim)

    query, candidates = pair(n, dim, 12)
    with torch.no_grad():
        ours = head(query, candidates)
        theirs = native(query, candidates)

    width = len(native.feature_names)
    assert torch.equal(ours[:, :width], theirs)
    assert ours.shape == (n, width + len(head.added_names))


@pytest.mark.parametrize("arm", [arms.CONTROL_ARM, arms.CANDIDATE_ARM])
def test_the_wrapped_s4_branch_still_learns(arm) -> None:
    """Every S4 parameter must take gradient through the arm.

    A detached or frozen projection produces byte-identical forward values, so
    no equality test above can see it -- and it would silently turn both arms
    into "S4's frozen random projection plus a new column", which is a
    different experiment than the one section 8b authorised.
    """

    head = arms.build_arm_head(arm, dim=64)
    query, candidates = pair(6, 64, 13)
    head(query, candidates).sum().backward()

    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert not torch.equal(parameter.grad, torch.zeros_like(parameter.grad)), name

    # And the projection's own parameters are among them, at full count.
    projection_grads = [
        p.numel() for n, p in head.named_parameters() if n.startswith("projection.")
    ]
    assert sum(projection_grads) == head.projection.parameter_count()


def test_a1_adds_two_columns_and_zero_parameters() -> None:
    head = arms.RawSemanticSkipHead(dim=LIVE_DIM)
    assert head.rung == "A1"
    assert head.added_names == ("cosine_qd", "mean_abs_diff")
    assert head.added_parameter_count() == 0
    assert head.parameter_count() == head.projection.parameter_count()
    assert len(head.feature_names) == len(head.projection.feature_names) + 2


def test_a3_minimal_adds_one_column_and_exactly_dim_parameters() -> None:
    """1,536 -- the count section 8b measured off the live S3 module, not a
    number read from RUNG_PARAMETERS, which is stated at a width no fit uses."""

    head = arms.SemanticDifferenceHead(dim=LIVE_DIM)
    assert head.rung == "A3_MINIMAL"
    assert head.added_names == ("semantic_difference",)
    assert head.added_parameter_count() == LIVE_DIM == 1536
    assert head.difference_weight.numel() == LIVE_DIM
    assert len(head.feature_names) == len(head.projection.feature_names) + 1


def test_a3_minimals_parameter_is_initialised_exactly_as_s3s_is() -> None:
    head = arms.SemanticDifferenceHead(dim=64)
    live = SemanticHead("S3", dim=64)
    assert torch.equal(head.difference_weight, live.difference_weight)
    assert torch.equal(head.difference_weight, torch.zeros(64))
    assert head.difference_weight.requires_grad
    # No bias hides beside it: the added parameters are the one vector.
    added = [
        name for name, _ in head.named_parameters() if not name.startswith("projection.")
    ]
    assert added == ["difference_weight"]


def test_the_zero_init_is_inert_at_step_zero_and_still_trainable() -> None:
    """Section 8b's reason for keeping S3's init: the column starts at zero, so
    an eventual gain is attributable to the weights being learned rather than
    to a channel existing. It has to still have a gradient, or it never would
    be."""

    head = arms.SemanticDifferenceHead(dim=32)
    query, candidates = pair(4, 32, 5)
    columns = head(query, candidates)
    assert torch.equal(columns[:, -1], torch.zeros(4))

    columns[:, -1].sum().backward()
    gradient = head.difference_weight.grad
    assert gradient is not None
    assert not torch.equal(gradient, torch.zeros(32))
    assert torch.allclose(gradient, (candidates - query).abs().sum(dim=0))


def test_the_two_arms_are_a_control_pair_on_the_shared_column() -> None:
    """A1's mean_abs_diff IS A3-MINIMAL's column at the fixed weight 1/dim.

    This is the amendment's central causal claim. If it were false the pair
    would not isolate the learned weighting from raw access at all.
    """

    dim, n = 48, 5
    control = arms.RawSemanticSkipHead(dim=dim)
    candidate = arms.SemanticDifferenceHead(dim=dim)
    query, candidates = pair(n, dim, 9)

    with torch.no_grad():
        candidate.difference_weight.fill_(1.0 / dim)
        learned = candidate(query, candidates)[:, -1]
        uniform = control(query, candidates)[:, -1]

    assert torch.allclose(learned, uniform, atol=1e-6)
    # And the nesting is only partial: A1 carries a column A3-MINIMAL has not.
    assert "cosine_qd" in control.feature_names
    assert "cosine_qd" not in candidate.feature_names


@pytest.mark.parametrize("arm", [arms.CONTROL_ARM, arms.CANDIDATE_ARM])
def test_no_arm_computes_a_column_the_declaration_excluded(arm) -> None:
    """dot_qd_pct and semantic_product are on section 8b's do-not-add list.

    Absence from feature_names is the check that matters, and the cost side is
    covered by the arms never calling the live functions that compute them.
    """

    head = arms.build_arm_head(arm, dim=32)
    assert "dot_qd_pct" not in head.feature_names
    assert "semantic_product" not in head.feature_names


# ---------------------------------------------------------------------------
# Section 11: what the arms actually weigh, at the width the fits run at
# ---------------------------------------------------------------------------


def _built(arm: str):
    """One arm, instantiated at the width and depth the Stage-1 fits use."""

    from mp_retrieval.m1a_screen import build_m1a_model
    from scripts.run_m2b_semantic_minimality import build_semantic_head

    head = (
        build_semantic_head(arm, LIVE_DIM)
        if arm == "S4"
        else arms.build_arm_head(arm, LIVE_DIM)
    )
    model = build_m1a_model(
        precomputed_width=LIVE_PRECOMPUTED_WIDTH,
        semantic_rung=arm,
        dropout=0.1,
        temperature=1.0,
        embedding_dim=LIVE_DIM,
        semantic_head=head,
    )
    return (
        sum(p.numel() for p in model.semantic_head.parameters()),
        sum(p.numel() for p in model.scorer.parameters()),
    )


@pytest.mark.parametrize(
    "arm,semantic,scorer,total",
    [
        ("S4", 196_608, 8_609, 205_217),
        (arms.CONTROL_ARM, 196_608, 8_673, 205_281),
        (arms.CANDIDATE_ARM, 198_144, 8_641, 206_785),
    ],
)
def test_the_measured_parameter_counts_at_the_live_width(arm, semantic, scorer, total) -> None:
    """Instantiated and counted, never quoted -- and the scorer is counted too.

    An added column is not free beyond its own weights: M2B's scorer width is
    derived from ``len(feature_names)``, so each column widens the frozen
    scorer's first layer by ``head_width``. A1 declares zero added *semantic*
    parameters and still costs 64; A3-MINIMAL's 1,536-parameter diagonal costs
    1,568 in the model. Section 11 says not to call the result tiny, and the
    honest number is the second one.

    The absolute totals are the ones section 15 reports, so they are asserted
    at the width the fits actually run at -- read from the filed table above,
    not chosen here.
    """

    measured_semantic, measured_scorer = _built(arm)
    assert measured_semantic == semantic
    assert measured_scorer == scorer
    assert measured_semantic + measured_scorer == total


def test_native_s4s_count_here_is_the_one_m2b_filed() -> None:
    """The check that makes the row above a fact rather than my arithmetic.

    Native S4 is not a Stage-1 arm; it is the model M2B already fitted on these
    cells and whose checkpoint the runner reloads. If S4 built here does not
    weigh exactly what M2B recorded, then the width, the head or the scorer
    differs from the fits, and every added-parameter number reported beside it
    would be measured against the wrong incumbent.
    """

    import json

    table = json.loads(BASELINE_TABLE.read_text(encoding="utf-8"))
    filed = {
        (row["semantic_parameters"], row["total_parameters"] - row["semantic_parameters"])
        for row in table["rows"]
        if row["rung"] == "S4"
    }
    assert len(filed) == 1, f"M2B's filed S4 fits disagree with each other: {filed}"
    assert _built("S4") == filed.pop()


@pytest.mark.parametrize(
    "arm,added_semantic,added_total",
    [(arms.CONTROL_ARM, 0, 64), (arms.CANDIDATE_ARM, 1_536, 1_568)],
)
def test_what_each_arm_adds_does_not_depend_on_the_store(
    arm, added_semantic, added_total
) -> None:
    """The width-independent half of section 11, separated from the totals.

    The totals above move if the structural store ever changes width. What an
    arm ADDS does not: it is its own weights plus head_width per added column.
    Asserting it at two widths says which of the two claims is contingent.
    """

    from mp_retrieval.m1a_screen import HEAD_WIDTH, build_m1a_model
    from scripts.run_m2b_semantic_minimality import build_semantic_head

    for width in (LIVE_PRECOMPUTED_WIDTH, LIVE_PRECOMPUTED_WIDTH + 31):
        counts = {}
        for name in ("S4", arm):
            head = (
                build_semantic_head(name, LIVE_DIM)
                if name == "S4"
                else arms.build_arm_head(name, LIVE_DIM)
            )
            model = build_m1a_model(
                precomputed_width=width, semantic_rung=name, dropout=0.1,
                temperature=1.0, embedding_dim=LIVE_DIM, semantic_head=head,
            )
            counts[name] = (
                sum(p.numel() for p in model.semantic_head.parameters()),
                sum(p.numel() for p in model.scorer.parameters()),
            )
        semantic_delta = counts[arm][0] - counts["S4"][0]
        scorer_delta = counts[arm][1] - counts["S4"][1]
        assert semantic_delta == added_semantic, width
        assert semantic_delta + scorer_delta == added_total, width
        assert scorer_delta % HEAD_WIDTH == 0, "a column costs head_width in the scorer"


def test_the_frozen_scorer_takes_the_arms_without_being_changed() -> None:
    """Section 12: the scorer, the loss and the training loop stay M2's.

    The arms reach the fit through the injection point M2B already added for
    varying the semantic branch, so nothing about the scorer is redefined --
    only its input width, which was always derived from the head.
    """

    from mp_retrieval.m1a_screen import HEAD_WIDTH, build_m1a_model

    def scorer_of(arm, head):
        return build_m1a_model(
            precomputed_width=40, semantic_rung=arm, dropout=0.1, temperature=1.0,
            embedding_dim=LIVE_DIM, semantic_head=head,
        ).scorer

    from scripts.run_m2b_semantic_minimality import build_semantic_head

    native = scorer_of("S4", build_semantic_head("S4", LIVE_DIM))
    for arm in (arms.CONTROL_ARM, arms.CANDIDATE_ARM):
        head = arms.build_arm_head(arm, LIVE_DIM)
        ours = scorer_of(arm, head)
        assert type(ours) is type(native)
        assert [type(layer) for layer in ours] == [type(layer) for layer in native]
        assert ours[0].out_features == native[0].out_features == HEAD_WIDTH
        assert ours[0].in_features == native[0].in_features + len(head.added_names)


def test_a_head_whose_rung_disagrees_with_its_label_is_refused() -> None:
    """A fit recorded under the wrong arm name is unrecoverable afterwards, so
    the model refuses the pair rather than trusting the label."""

    from mp_retrieval.m1a_screen import build_m1a_model

    with pytest.raises(ValueError, match="but the model was asked for"):
        build_m1a_model(
            precomputed_width=40, semantic_rung="S4", dropout=0.1, temperature=1.0,
            embedding_dim=64, semantic_head=arms.build_arm_head(arms.CONTROL_ARM, 64),
        )


# ---------------------------------------------------------------------------
# The interface the existing runner switches on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", [arms.CONTROL_ARM, arms.CANDIDATE_ARM])
def test_an_arm_fingerprints_distinctly_from_every_rung(arm) -> None:
    """M2B's provenance hash has to tell an arm fit from an S4 fit. Same
    dataset, same regime, same store, same seed, same commit -- the semantic
    branch is the only thing that differs, so it is the only thing that can."""

    from scripts.run_m2b_semantic_minimality import (
        build_semantic_head,
        semantic_rung_fingerprint,
    )

    ours = semantic_rung_fingerprint(arms.build_arm_head(arm, dim=64))
    assert ours["rung"] == arm
    assert ours["semantic_columns"] == 258 + (1 if arm == arms.CANDIDATE_ARM else 2)

    for rung in ("S2", "S3", "S4"):
        assert ours["sha256"] != semantic_rung_fingerprint(
            build_semantic_head(rung, dim=64)
        )["sha256"]


def test_an_unknown_arm_raises_rather_than_falling_back() -> None:
    for name in ("A2", "A3", "S4", "", "a1"):
        with pytest.raises(ValueError, match="unknown Stage-1 arm"):
            arms.build_arm_head(name, dim=32)


@pytest.mark.parametrize("arm", [arms.CONTROL_ARM, arms.CANDIDATE_ARM])
def test_a_bad_shape_raises_the_live_heads_error(arm) -> None:
    head = arms.build_arm_head(arm, dim=32)
    with pytest.raises(ValueError, match="candidates must be"):
        head(torch.randn(32), torch.randn(32))
    with pytest.raises(ValueError, match="does not match candidate dim"):
        head(torch.randn(16), torch.randn(3, 32))
