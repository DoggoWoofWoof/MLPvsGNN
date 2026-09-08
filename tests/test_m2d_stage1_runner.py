"""M2D Stage 1's runner, on the parts that can be checked without a GPU.

The fit itself needs a dataset, a sealed cell master and M2B's checkpoints, so
it is exercised where it runs. What can be checked here is everything that
decides whether the fit MEANS anything, and all of it is cheap:

* the three identities that make a delta against a filed row legitimate;
* section 8's error-conditioned analysis, in both directions, including the
  cases where a plausible implementation would silently report the wrong sign;
* section 12's list of things not to reopen -- every training argument is
  M2B's, compared against M2B's own parser rather than retyped here;
* the refusals: an unauthorised seed, an absent S4 checkpoint, an unpaired
  panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

yaml = pytest.importorskip("yaml")
pytest.importorskip("torch")

from scripts import m2d_stage1_gate as gate
from scripts import run_m2b_semantic_minimality as _m2b
from scripts import run_m2d_stage1_arms as runner

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml").read_text(encoding="utf-8")
)


def rows(values: list[float]) -> list[dict[str, float]]:
    """Per-query rows carrying just the column the analysis conditions on."""

    return [{"recall@1": value, "recall@5": 1.0} for value in values]


# ---------------------------------------------------------------------------
# Section 8: the error-conditioned analysis
# ---------------------------------------------------------------------------


def test_corrections_and_breaks_are_counted_separately() -> None:
    native = rows([0.0, 0.0, 0.0, 1.0, 1.0])
    arm = rows([1.0, 1.0, 0.0, 0.0, 1.0])
    result = runner.integration_analysis(native, arm)

    assert result["s4_top1_errors"] == 3
    assert result["s4_top1_correct"] == 2
    assert result["corrected"] == 2
    assert result["newly_broken"] == 1
    assert result["net_top1_corrections"] == 1


def test_an_arm_that_breaks_more_than_it_fixes_reports_a_negative_net() -> None:
    """The case a corrections-only report would present as a success."""

    result = runner.integration_analysis(
        rows([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        rows([1.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
    )
    assert result["corrected"] == 2
    assert result["newly_broken"] == 3
    assert result["net_top1_corrections"] == -1


def test_changing_nothing_reports_nothing() -> None:
    native = rows([0.0, 1.0, 0.0, 1.0])
    result = runner.integration_analysis(native, list(native))
    assert (result["corrected"], result["newly_broken"]) == (0, 0)
    assert result["net_top1_corrections"] == 0


def test_a_partial_gold_recovery_counts_as_a_correction() -> None:
    """On a multi-gold cell recall@1 is fractional, and 0 -> anything positive
    is a relevant node reaching rank 1, which is the event section 8 asks about."""

    result = runner.integration_analysis(rows([0.0, 0.5]), rows([0.5, 0.0]))
    assert result["corrected"] == 1
    assert result["newly_broken"] == 1


def test_the_fractions_are_reported_against_the_right_denominators() -> None:
    result = runner.integration_analysis(
        rows([0.0, 0.0, 0.0, 0.0, 1.0]), rows([1.0, 1.0, 1.0, 0.0, 0.0])
    )
    assert result["corrected_fraction_of_s4_errors"] == pytest.approx(3 / 4)
    assert result["newly_broken_fraction_of_s4_correct"] == pytest.approx(1 / 1)


def test_a_cell_s4_never_got_wrong_does_not_divide_by_zero() -> None:
    result = runner.integration_analysis(rows([1.0, 1.0]), rows([1.0, 0.0]))
    assert result["s4_top1_errors"] == 0
    assert result["corrected_fraction_of_s4_errors"] is None
    assert result["newly_broken"] == 1


def test_unpaired_row_lists_are_refused_rather_than_zipped() -> None:
    with pytest.raises(ValueError, match="nothing can be paired"):
        runner.integration_analysis(rows([0.0, 1.0]), rows([0.0]))


# ---------------------------------------------------------------------------
# The three identities
# ---------------------------------------------------------------------------


def test_the_panel_digest_is_order_sensitive() -> None:
    """Section 8 pairs rows by position, so two identical sets in different
    orders would pair every query with the wrong one and agree on every
    aggregate while doing it."""

    assert runner.panel_digest(["a", "b"]) != runner.panel_digest(["b", "a"])
    assert runner.panel_digest(["a", "b"]) == runner.panel_digest(["a", "b"])


def test_a_mismatched_identity_refuses_and_says_what_it_costs() -> None:
    with pytest.raises(ValueError, match="does not match what M2B filed"):
        runner.check_identity(
            observed="a" * 64, expected="b" * 64,
            what="the held-out panel", consequence="Every delta would be unpaired.",
        )


def test_an_unchecked_identity_is_recorded_as_unchecked() -> None:
    """A local run with nothing to match is legitimate; an artifact that
    cannot say which panel it scored is not, so the absence is written down."""

    record = runner.check_identity(
        observed="a" * 64, expected=None, what="the held-out panel", consequence="",
    )
    assert record["checked"] is False
    assert record["expected"] is None
    assert record["observed"] == "a" * 64


# ---------------------------------------------------------------------------
# Section 12: nothing about the training setup is reopened
# ---------------------------------------------------------------------------


SHARED_WITH_M2B = (
    "holdout_fraction", "epochs", "batch_size", "dropout", "temperature",
    "learning_rate", "weight_decay", "latency_queries", "latency_repeats",
    "latency_warmup", "per_seed_cap", "neighbour_scan_cap_per_seed",
    "a64_mainline_family",
)


def defaults(parser: argparse.ArgumentParser) -> dict:
    return {action.dest: action.default for action in parser._actions}


@pytest.mark.parametrize("name", SHARED_WITH_M2B)
def test_every_training_argument_is_m2bs_own(name) -> None:
    """Compared against M2B's parser, not retyped here.

    Section 12 forbids reopening the loss, the splits or the candidate
    normalisation. A number copied into this file could drift from M2B's
    silently; a comparison cannot.
    """

    ours = defaults(runner.build_parser())
    theirs = defaults(_m2b.build_parser())
    assert name in ours and name in theirs
    assert ours[name] == theirs[name], name


def launched_by_m2b() -> dict:
    """The values M2B's launcher actually passed, read as values.

    Parsed rather than string-matched: 1e-3 and 0.001 are the same learning
    rate and different text, and a test that could fail on the spelling would
    eventually be silenced by rewriting the spelling.
    """

    import ast

    tree = ast.parse(
        (REPO_ROOT / "scripts" / "modal_m2b_semantic_minimality.py").read_text(
            encoding="utf-8"
        )
    )
    passed: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if name != "Namespace":
            continue
        for keyword in node.keywords:
            try:
                passed[keyword.arg] = ast.literal_eval(keyword.value)
            except ValueError:
                pass  # a computed argument; not one of the frozen scalars
    return passed


@pytest.mark.parametrize(
    "name",
    ["holdout_fraction", "epochs", "batch_size", "dropout", "temperature",
     "learning_rate", "weight_decay", "latency_queries", "latency_repeats",
     "latency_warmup"],
)
def test_the_values_m2bs_launcher_actually_passed_are_the_same_ones(name) -> None:
    """The parser default and the launched value are two different things, and
    what M2B's fits were produced with is the second one."""

    launched = launched_by_m2b()
    assert name in launched, f"M2B's launcher does not pass {name}"
    assert defaults(runner.build_parser())[name] == launched[name], name


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_the_authorised_seeds_are_read_from_the_declaration() -> None:
    """Not typed here, and not widened by a flag.

    Stage 1 ran at seed 0. Section 15b later authorised seeds 1 and 2 for
    A3-MINIMAL alone, and this runner learned about them by reading that file.
    Had the gate been loosened by hand instead, the file's authorisation would
    have become decorative and the refusal below would guard nothing.
    """

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml").read_text(encoding="utf-8")
    )
    table = runner.authorised_seeds()

    assert sorted(table) == sorted(
        {*config["stage_1"]["seeds"], *config.get("stage_2", {}).get("seeds", [])}
    )
    assert table[0]["arms"] == runner.ARMS, "seed 0 is Stage 1's and keeps both arms"
    assert table[0]["status"] == runner.COMPLETE_STATUS
    for seed in config["stage_2"]["seeds"]:
        assert table[seed]["arms"] == ("A3_MINIMAL",)
        assert table[seed]["status"] == runner.STAGE_2_COMPLETE_STATUS


def test_an_unauthorised_seed_is_refused_before_any_data_is_touched() -> None:
    args = argparse.Namespace(arms=None, seed=7)
    with pytest.raises(ValueError, match="not authorised by the declaration"):
        runner.run(args)


def test_stage_1s_control_arm_is_refused_at_a_stage_2_seed() -> None:
    """A1 is not rerun. Its role was to make a seed-0 gain attributable, which
    a seed cannot reopen, and section 15b's four fits are four because A1 is
    not among them. Accepting it here would double the stage's cost silently."""

    args = argparse.Namespace(arms=["A1"], seed=1)
    with pytest.raises(ValueError, match="not authorised at seed 1"):
        runner.run(args)


def test_an_arm_the_declaration_never_named_is_refused() -> None:
    args = argparse.Namespace(arms=["A2"], seed=0)
    with pytest.raises(ValueError, match="not authorised at seed 0"):
        runner.run(args)


def test_an_absent_s4_checkpoint_refuses_rather_than_dropping_the_analysis(
    tmp_path,
) -> None:
    """Section 8's analysis is the mechanism-specific evidence this stage
    exists to produce. Proceeding without it would produce eight fits that
    cannot answer the question they were paid for."""

    args = argparse.Namespace(s4_checkpoint=tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError, match="section 8's error-conditioned"):
        runner.rescore_native_s4(
            args=args, precomputed_width=1, node_embeddings=None,
            query_embeddings=None, store=None, held_out=[], device=None,
        )


# ---------------------------------------------------------------------------
# The runner and the declaration agree, and so do the runner and the gate
# ---------------------------------------------------------------------------


def test_the_arms_and_the_seed_are_the_declarations() -> None:
    assert list(runner.ARMS) == DECLARATION["stage_1"]["arms"]
    assert runner.DECLARED_SEED in DECLARATION["stage_1"]["seeds"]
    assert len(DECLARATION["stage_1"]["seeds"]) == 1


def test_the_runner_and_the_gate_share_one_vocabulary() -> None:
    assert runner.ARMS == gate.ARMS
    assert runner.COMPLETE_STATUS == "M2D_STAGE1_ARM_COMPLETE"
    assert runner.NATIVE_RUNG == gate.NATIVE
    assert runner.DECLARED_SEED == gate.SEED


def test_the_runner_never_reads_the_test_split() -> None:
    """The standing rule of this track, checked in the file that could break
    it: the panel comes from the validation split and the artifact says so."""

    source = (REPO_ROOT / "scripts" / "run_m2d_stage1_arms.py").read_text(encoding="utf-8")
    assert "QuerySplit.VALIDATION" in source
    assert "QuerySplit.TEST" not in source
    assert '"test_split_read": False' in source


def test_nothing_already_fit_is_refit_to_produce_a_comparison_row() -> None:
    """Section 5's reuse rule. The runner trains the two arms and nothing else;
    native S4 is loaded from M2B's checkpoint and re-scored."""

    import ast
    import inspect
    import textwrap

    source = inspect.getsource(runner.rescore_native_s4)
    assert "_fit(" not in source
    assert "trained_here" in source

    # The load is checked as a call, not as text. An earlier version of this
    # test looked for the string "strict=True" and passed happily when the code
    # said strict=False, because the docstring above it still said strict=True.
    tree = ast.parse(textwrap.dedent(source))
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "load_state_dict"
    ]
    assert len(loads) == 1, "native S4 is loaded exactly once"
    strict = {
        keyword.arg: keyword.value
        for keyword in loads[0].keywords
    }.get("strict")
    assert isinstance(strict, ast.Constant) and strict.value is True, (
        "strict=True is load-bearing: a checkpoint whose semantic branch did not "
        "match would otherwise load with the mismatched tensors silently dropped, "
        "and every number would describe a partly randomised model while reporting "
        "it as M2B's"
    )
