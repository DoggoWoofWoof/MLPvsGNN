"""The condition-B probe: what it measures, and that it measures S3's quantities.

The gate returned STOP_PENDING_B, and this probe exists to supply the missing
term. That makes two things worth more than the usual care, and they are what
this file is mostly about.

**The admitted set is the archaeology's, not this probe's.** B is a claim about
a primitive S4 is MISSING. If this file were free to name that set, a primitive
could be admitted because it scored well, which is choosing the comparison from
the result. So the five primitives marked ``missing_from_s4`` are held equal to
``m2d_semantic_archaeology.s3_not_in_s4()`` -- the function that produced the
S3_NOT_IN_S4 finding -- and the negative control is held to be the only member
of the complement.

**The scores are S3's and S4's own columns.** A reimplemented cosine that
differed in the fourth decimal would answer B about a quantity no rung ever
computed. Every parameter-free column is checked against S3's own forward, and
the negative control against the exact column S4's head emits, on real modules
built the way M2B builds them.

The end-to-end run needs the sealed cell masters and M2B's checkpoints, which
live on the volume; this follows the convention of the Stage-0 probe's tests
and covers everything that does not.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval import m2b_semantic_control, qls_v2_semantic, run_artifacts
from scripts import m2d_semantic_archaeology as archaeology
from scripts import run_m2d_primitive_probe as probe
from scripts import run_m2d_stage0_probe as stage0

PRECOMPUTED_WIDTH = 7
EMBEDDING_DIM = 32
POOL = np.asarray([10, 20, 30, 40], dtype=np.int64)


def build(rung: str) -> torch.nn.Module:
    import scripts.run_m1a_feature_screen as _m1a
    import scripts.run_m2b_semantic_minimality as _m2b

    return _m1a.build_m1a_model(
        precomputed_width=PRECOMPUTED_WIDTH,
        semantic_rung=rung,
        dropout=0.2,
        temperature=0.07,
        embedding_dim=EMBEDDING_DIM,
        semantic_head=_m2b.build_semantic_head(rung, EMBEDDING_DIM),
    )


@pytest.fixture(scope="module")
def heads() -> dict[str, torch.nn.Module]:
    """S3 and S4 with weights that are not their initialization.

    At initialization S3's difference_weight is zero and its product_weight is
    a constant, which would make two of the five columns degenerate and let a
    wrong readout pass. The checkpoints this probe reads are trained, so the
    test perturbs both heads before comparing anything.
    """

    torch.manual_seed(0)
    models = {rung: build(rung) for rung in ("S3", "S4")}
    with torch.no_grad():
        for parameter in models["S3"].semantic_head.parameters():
            parameter.copy_(torch.randn_like(parameter))
    for model in models.values():
        model.eval()
    return models


@pytest.fixture(scope="module")
def pool() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    query = torch.randn(EMBEDDING_DIM, generator=generator)
    candidates = torch.randn(POOL.size, EMBEDDING_DIM, generator=generator)
    return query, candidates


# ---------------------------------------------------------------------------
# The admitted set is the archaeology's finding, not this file's
# ---------------------------------------------------------------------------


def test_the_primitives_marked_missing_are_exactly_the_archaeologys_finding() -> None:
    """The gate admits a primitive for B only if it is marked missing here, so
    this flag is the admission test. It has to be a citation."""

    declared = {
        spec["archaeology"] for spec in probe.PRIMITIVES.values() if spec["missing_from_s4"]
    }
    assert declared == {row["primitive"] for row in archaeology.s3_not_in_s4()}


def test_the_negative_control_is_the_only_primitive_s4_already_has() -> None:
    """A second unadmitted primitive would be a quantity measured for no stated
    reason; a control marked admitted would let S4's own geometry satisfy a
    condition about what S4 lacks."""

    present = [
        name for name, spec in probe.PRIMITIVES.items() if not spec["missing_from_s4"]
    ]
    assert present == ["normalized_state_dot"]
    assert probe.PRIMITIVES["normalized_state_dot"]["archaeology"] is None
    assert "NEGATIVE CONTROL" in probe.PRIMITIVES["normalized_state_dot"]["what_it_is"]


def test_the_control_is_a_column_s4_actually_emits() -> None:
    """Which is what makes it a control rather than a sixth candidate."""

    assert "normalized_state_dot" in m2b_semantic_control.feature_names()


def test_every_primitive_declares_a_direction_and_says_what_it_is() -> None:
    for name, spec in probe.PRIMITIVES.items():
        assert isinstance(spec["higher_is_better"], bool), name
        assert isinstance(spec["parameter_free"], bool), name
        assert spec["what_it_is"].strip(), name


def test_the_two_distances_are_the_two_ranked_ascending() -> None:
    """Direction follows the definition, not the score. Naming the ascending
    set here means a later flip has to be an edit to this test as well."""

    ascending = {
        name for name, spec in probe.PRIMITIVES.items() if not spec["higher_is_better"]
    }
    assert ascending == {"mean_abs_diff", "semantic_difference"}


def test_the_parameter_free_primitives_are_the_modules_own_three() -> None:
    free = {name for name, spec in probe.PRIMITIVES.items() if spec["parameter_free"]}
    assert free == set(qls_v2_semantic.PARAMETER_FREE_FEATURE_NAMES)


def test_the_five_admitted_primitives_are_s3s_five_columns() -> None:
    """S3 is the rung whose signal S4 lost, so B's candidate set being exactly
    S3's column set is the thing that makes this a repair question."""

    admitted = {
        name for name, spec in probe.PRIMITIVES.items() if spec["missing_from_s4"]
    }
    assert admitted == set(qls_v2_semantic.RUNG_FEATURES["S3"])


# ---------------------------------------------------------------------------
# The scores are the quantities the rungs compute
# ---------------------------------------------------------------------------


def test_every_column_is_s3s_own_column_and_not_a_reimplementation(heads, pool) -> None:
    """All five admitted primitives at once, against S3's forward.

    S3's forward is what M2B trained and what the archaeology inventoried. A
    cosine that agreed to five decimals would still answer B about a quantity
    no rung ever computed, so the comparison is exact.
    """

    query, candidates = pool
    scores = probe._primitive_scores(query, candidates, heads["S3"], heads["S4"])
    with torch.no_grad():
        columns = heads["S3"].semantic_head(query, candidates)
    for index, name in enumerate(heads["S3"].semantic_head.feature_names):
        assert np.array_equal(scores[name], columns[:, index].numpy().astype(np.float64)), name


def test_the_control_is_the_column_s4_itself_emits(heads, pool) -> None:
    """Recomputed rather than sliced out of the forward, so it is checked
    against the forward. A GELU dropped or a normalize applied in the wrong
    order would make the control a different quantity that still looks like a
    cosine."""

    query, candidates = pool
    scores = probe._primitive_scores(query, candidates, heads["S3"], heads["S4"])
    with torch.no_grad():
        emitted = heads["S4"].semantic_head(query, candidates)
    index = m2b_semantic_control.feature_names().index("normalized_state_dot")
    assert np.allclose(
        scores["normalized_state_dot"], emitted[:, index].numpy().astype(np.float64), atol=0
    )


def test_every_primitive_scores_every_candidate_exactly_once(heads, pool) -> None:
    query, candidates = pool
    scores = probe._primitive_scores(query, candidates, heads["S3"], heads["S4"])
    assert set(scores) == set(probe.PRIMITIVES)
    for name, value in scores.items():
        assert value.shape == (POOL.size,), name
        assert value.dtype == np.float64, name


def test_no_score_is_a_nan_the_writer_would_refuse(heads, pool) -> None:
    query, candidates = pool
    scores = probe._primitive_scores(query, candidates, heads["S3"], heads["S4"])
    for name, value in scores.items():
        assert np.isfinite(value).all(), name


# ---------------------------------------------------------------------------
# The direction, and the counting rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_the_declared_direction_decides_which_end_ranks_first(higher_is_better) -> None:
    score = np.asarray([0.1, 0.9, 0.5, 0.3])
    ranks = probe.ranks_by(score, POOL, higher_is_better)
    first = int(POOL[np.argmin(ranks)])
    assert first == (20 if higher_is_better else 10)
    assert sorted(ranks.tolist()) == [1, 2, 3, 4], "a ranking, not a score"


def test_the_percentile_and_the_raw_dot_are_one_sole_ranker(heads, pool) -> None:
    """The docstring's claim, and the reason dot_qd_pct is measured once.

    Within a query the percentile is a monotone transform of the raw dot, so
    as a sole ranker the two are the same ranker. Reporting both would report
    one number as two findings.
    """

    query, candidates = pool
    scores = probe._primitive_scores(query, candidates, heads["S3"], heads["S4"])
    raw_dot = (candidates @ query).numpy().astype(np.float64)
    assert np.array_equal(
        probe.ranks_by(scores["dot_qd_pct"], POOL, True),
        probe.ranks_by(raw_dot, POOL, True),
    )


def test_a_relevant_candidate_above_s4s_top_item_counts_as_a_reorder() -> None:
    relevant = np.asarray([False, True, False, False])
    ranks = probe.ranks_by(np.asarray([0.1, 0.9, 0.5, 0.3]), POOL, True)
    assert probe.reorders(ranks, relevant, 2) is True


def test_a_relevant_candidate_below_s4s_top_item_does_not() -> None:
    relevant = np.asarray([False, False, False, True])
    ranks = probe.ranks_by(np.asarray([0.1, 0.9, 0.5, 0.3]), POOL, True)
    assert probe.reorders(ranks, relevant, 1) is False


def test_being_s4s_own_top_item_is_not_a_reorder() -> None:
    """The excluded case, asserted rather than assumed: a query S4 already gets
    right cannot be evidence that anything repaired it. The run loop drops
    these, and this pins what dropping them means."""

    relevant = np.asarray([False, True, False, False])
    ranks = probe.ranks_by(np.asarray([0.1, 0.9, 0.5, 0.3]), POOL, True)
    assert probe.reorders(ranks, relevant, 1) is False


def test_the_best_relevant_candidate_is_what_counts_not_the_worst() -> None:
    """B asks whether the primitive rescues the query at all."""

    relevant = np.asarray([True, True, False, False])
    ranks = probe.ranks_by(np.asarray([0.1, 0.9, 0.5, 0.3]), POOL, True)
    assert probe.reorders(ranks, relevant, 2) is True


# ---------------------------------------------------------------------------
# The probe's own contract
# ---------------------------------------------------------------------------


def test_the_probe_cannot_be_run_without_an_identity_for_its_result() -> None:
    required = {
        action.dest for action in probe.build_parser()._actions if getattr(action, "required", False)
    }
    assert {"source_commit", "config_fingerprint", "output_root", "dataset", "regime"} <= required


def test_the_probe_has_no_flag_that_could_train_or_sweep_anything() -> None:
    flags = {action.dest for action in probe.build_parser()._actions}
    for forbidden in ("epochs", "learning_rate", "lr", "sweep", "seed", "test", "primitive"):
        assert forbidden not in flags, f"--{forbidden} would make this not a zero-training probe"


def test_the_probe_declares_the_phase_and_arm_the_artifact_path_will_carry() -> None:
    assert probe.PHASE == stage0.PHASE == "m2d"
    assert probe.ARM == "stage0_primitive_probe"
    assert probe.COMPLETE_STATUS == "M2D_PRIMITIVE_PROBE_COMPLETE"


def test_the_arm_is_not_the_one_stage_0_wrote_under() -> None:
    """Two results of one cell on one store are kept apart by the arm alone."""

    assert probe.ARM != stage0.ARM
    assert probe.COMPLETE_STATUS != stage0.COMPLETE_STATUS


def test_the_gate_reads_the_status_this_probe_writes() -> None:
    """The loader refuses anything else, so a renamed status would read as a
    corrupt artifact rather than as a measurement, and B would stay unmeasured
    while a completed run sat on disk."""

    from scripts import m2d_stage0_gate

    source = (REPO_ROOT / "scripts" / "m2d_stage0_gate.py").read_text(encoding="utf-8")
    assert probe.COMPLETE_STATUS in source
    assert m2d_stage0_gate.PRIMITIVE_ROOT.name == "stage0_primitives"


# --------------------------------------------------------------------------
# What the artifact promises about itself
# --------------------------------------------------------------------------

PROBE_TREE = ast.parse((REPO_ROOT / "scripts" / "run_m2d_primitive_probe.py").read_text("utf-8"))


def _payload_literal() -> ast.Dict:
    literal = None
    for node in ast.walk(PROBE_TREE):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "result"
            and isinstance(node.value, ast.Dict)
        ):
            literal = node.value
    assert literal is not None, "the probe's payload is no longer a dict literal named result"
    return literal


def _payload_value(key: str) -> ast.AST:
    literal = _payload_literal()
    match = [
        value
        for name, value in zip(literal.keys, literal.values, strict=True)
        if isinstance(name, ast.Constant) and name.value == key
    ]
    assert match, f"the payload has no key {key!r}"
    return match[0]


def _declared_rows_at() -> str:
    for node in ast.walk(PROBE_TREE):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "write_artifact"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "rows_at":
                assert isinstance(keyword.value, ast.Constant), "rows_at must be a literal"
                return keyword.value.value
    raise AssertionError("the probe does not call write_artifact")


def test_the_row_count_counts_something_countable() -> None:
    """The failure that cost four Stage-0 containers their last step, checked
    on the second runner before it is ever launched. write_artifact refuses a
    rows_at that resolves to anything but a list, and the payload is a literal
    in this file, so nothing about this needs a container to discover."""

    node = _payload_value(_declared_rows_at())
    assert isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("sorted", "list"), (
        f"rows_at={_declared_rows_at()!r} names a {type(node).__name__}, and "
        "write_artifact refuses anything but a list"
    )


def test_the_counted_rows_are_the_primitives_the_table_names() -> None:
    assert _declared_rows_at() == "primitives_measured"
    node = _payload_value("primitives_measured")
    assert isinstance(node.args[0], ast.Name) and node.args[0].id == "PRIMITIVES"


def test_an_empty_error_population_is_written_rather_than_divided_by() -> None:
    """A cell where S4 is right at rank 1 everywhere has no error population.
    An unguarded share would be a ZeroDivisionError, or worse a NaN, which
    write_artifact refuses on the last line after the panel has been paid for.
    """

    rows = _payload_value("primitives")
    assert isinstance(rows, ast.DictComp) and isinstance(rows.value, ast.Dict)
    share = [
        value
        for name, value in zip(rows.value.keys, rows.value.values, strict=True)
        if isinstance(name, ast.Constant) and name.value == "share_reordered"
    ]
    assert share and isinstance(share[0], ast.IfExp), (
        "share_reordered divides by the population without asking whether there is one"
    )


def test_an_empty_kernel_timing_is_written_rather_than_percentiled() -> None:
    systems = _payload_value("systems")
    assert isinstance(systems, ast.Dict)
    p50 = [
        value
        for name, value in zip(systems.keys, systems.values, strict=True)
        if isinstance(name, ast.Constant)
        and name.value == "primitive_kernel_ms_per_query_p50"
    ]
    assert p50 and isinstance(p50[0], ast.IfExp), "np.percentile([]) is not a number"


def test_a_nan_share_would_be_refused_by_the_writer() -> None:
    """Which is what makes the guard above load-bearing rather than tidy."""

    with pytest.raises(ValueError):
        run_artifacts.content_digest({"share_reordered": float("nan")})


def test_the_reported_shape_survives_its_own_digest() -> None:
    """The per-primitive block as the payload builds it, on both the measured
    and the empty case, through the real digest -- which round-trips with
    allow_nan=False and so refuses exactly what a container would refuse."""

    for population, reordered in ((500, 314), (0, 0)):
        block = {
            name: {
                "share_reordered": (reordered / population) if population else 0.0,
                "reordered": reordered,
                "population": population,
                "missing_from_s4": spec["missing_from_s4"],
                "parameter_free": spec["parameter_free"],
                "ranked": "descending" if spec["higher_is_better"] else "ascending",
                "archaeology_primitive": spec["archaeology"],
                "what_it_is": spec["what_it_is"],
            }
            for name, spec in probe.PRIMITIVES.items()
        }
        run_artifacts.content_digest(block)
