"""The launcher must refuse a run that would bill a window and produce nothing.

The substrate audit was submitted against a six-hour ceiling with four graph
families of 3.31 h each. Each attempt restarted from zero, died in the same
place, and left no file behind; the run that ended it billed about two hours for
literally no output, and the only thing that stopped it was the workspace
running out of money.

The gate turns on the indivisible unit rather than the total, because a long job
of small units is fine -- completed units survive a restart. But which piece is
indivisible is a property of the *runner*, not the work: one family fits six
hours comfortably, so a per-family reading would have waved that launch straight
through. It was fatal because nothing was carried across a restart, making the
real unit the whole 13.2 h audit. A runner that does not declare where it
checkpoints is therefore costed as redoing everything.

Three failure directions are pinned below, and each is expensive in its own way:

*   admitting an oversized unit is the observed failure -- a full window billed
    for nothing, repeatedly;
*   costing a non-resumable job by its parts is how a gate looks like it is
    working while admitting exactly the run it was built to stop;
*   refusing a launch on a guessed cost is the failure that follows, because a
    gate that blocks work on invented numbers gets bypassed, and then it is not
    protecting anything at all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

spawn = pytest.importorskip("scripts.spawn_modal_jobs")

HOUR = 3600.0
FAMILIES = {
    "dataset_default": {},
    "structural_only": {},
    "knn_only": {},
    "baseline_a_simple": {},
}

_UNSET = object()


def fake_module(timeout_seconds: float, granularity: Any = _UNSET) -> SimpleNamespace:
    """A runner stub. Omitting `granularity` models one that declares nothing."""

    module = SimpleNamespace(
        MODAL_CONFIG={"timeout_seconds": timeout_seconds},
        CONFIG={"graphs": FAMILIES},
    )
    if granularity is not _UNSET:
        module.RESUME_GRANULARITY = granularity
    return module


def substrate_jobs(*datasets: str) -> list[dict]:
    return [{"dataset": name} for name in datasets]


def e2_jobs(*datasets: str) -> list[dict]:
    return [{"dataset": name, "axis": "degree_rewire", "rate": 0.1} for name in datasets]


# --------------------------------------------------------------------------
# The run that actually happened
# --------------------------------------------------------------------------


def test_the_launch_that_burned_the_workspace_is_refused():
    """Four families, six-hour ceiling, nothing carried across a restart."""
    with pytest.raises(SystemExit) as excinfo:
        spawn.gate_launch(
            "graph-substrate", fake_module(6 * HOUR), substrate_jobs("hotpotqa_clean")
        )
    message = str(excinfo.value)
    assert "REFUSED" in message
    assert "retrying cannot help" in message
    assert "13.2" in message  # the whole audit, not one 3.31 h family


def test_one_family_would_have_fitted_which_is_why_the_unit_must_follow_the_runner():
    """The near miss that makes this gate worth having.

    3.31 h against a six-hour ceiling passes even at the 1.5x safety factor, so
    a gate that costs this job by its families admits the exact launch that
    produced nothing. Only the absence of resumption makes it refusable.
    """
    admitted = spawn.gate_launch(
        "graph-substrate", fake_module(6 * HOUR, "family"), substrate_jobs("hotpotqa_clean")
    )
    assert admitted["gated"] is True
    assert admitted["largest_unit_hours"] == pytest.approx(3.31, abs=0.02)


def test_an_undeclared_runner_is_costed_as_redoing_everything():
    """Silence must not read as "checkpoints perfectly"; that is the unsafe direction."""
    units, why = spawn.measured_units(
        "graph-substrate", fake_module(24 * HOUR), substrate_jobs("hotpotqa_clean")
    )
    assert len(units) == 1
    assert "does not declare" in why


def test_the_shipped_substrate_runner_declares_family_resumption():
    """The stub above only matters if the real module agrees with it."""
    module = pytest.importorskip("scripts.modal_graph_substrate_audit")
    assert module.RESUME_GRANULARITY == "family"


def test_the_shipped_confirmation_runner_declares_seed_resumption():
    module = pytest.importorskip("scripts.modal_phase_confirmation")
    assert module.RESUME_GRANULARITY == "seed"


# --------------------------------------------------------------------------
# The fixed configuration must actually get through
# --------------------------------------------------------------------------


def test_the_current_ceiling_and_resumption_admit_the_audit():
    report = spawn.gate_launch(
        "graph-substrate", fake_module(24 * HOUR, "family"), substrate_jobs("hotpotqa_clean")
    )
    assert report["gated"] is True
    assert report["units"] == 4
    assert report["total_hours"] > 13.0


def test_a_total_above_the_ceiling_is_allowed_when_every_unit_fits():
    """13.2 h of families in a 12 h window: the case a total-based check breaks.

    Refusing this would have blocked the audit permanently even after resumption
    made it work, which is the more expensive of the two mistakes.
    """
    report = spawn.gate_launch(
        "graph-substrate", fake_module(12 * HOUR, "family"), substrate_jobs("hotpotqa_clean")
    )
    assert report["gated"] is True
    assert report["total_hours"] > 12.0
    assert "survive a restart" in report["verdict"]


# --------------------------------------------------------------------------
# E2, whose unit is a seed
# --------------------------------------------------------------------------


def test_every_e2_cell_clears_the_ceiling():
    report = spawn.gate_launch(
        "phase-confirmation", fake_module(24 * HOUR, "seed"), e2_jobs("squad_clean", "metaqa")
    )
    assert report["gated"] is True
    assert report["units"] == 10  # two cells at five seeds


def test_the_e2_unit_is_a_seed_not_a_cell():
    """metaqa's cell is 33 min; its seed is 6.6. Only the seed faces the ceiling."""
    report = spawn.gate_launch(
        "phase-confirmation", fake_module(24 * HOUR, "seed"), e2_jobs("metaqa")
    )
    assert report["largest_unit_hours"] == pytest.approx(396.0 / 3600, abs=1e-3)


def test_an_e2_launch_reports_what_it_expects_to_spend():
    """The number the operator was never shown before the bill arrived."""
    report = spawn.gate_launch(
        "phase-confirmation", fake_module(24 * HOUR, "seed"), e2_jobs("squad_clean", "metaqa")
    )
    assert report["expected_spend_usd"] > 0


def test_a_ceiling_below_one_seed_is_refused():
    with pytest.raises(SystemExit, match="REFUSED"):
        spawn.gate_launch("phase-confirmation", fake_module(60.0, "seed"), e2_jobs("metaqa"))


# --------------------------------------------------------------------------
# Never refuse on a number nobody measured
# --------------------------------------------------------------------------


def test_an_unmeasured_dataset_is_reported_ungated_rather_than_refused():
    units, why = spawn.measured_units(
        "graph-substrate", fake_module(HOUR, "family"), substrate_jobs("webqsp")
    )
    assert units is None
    assert "no measured validation query count" in why


def test_an_unmeasured_package_passes_through_with_its_reason_recorded():
    """Ungated must be visible in the launch record, not silent."""
    report = spawn.gate_launch("online-systems", fake_module(6 * HOUR), e2_jobs("webqsp"))
    assert report["gated"] is False
    assert "no measured cost model" in report["why"]
    assert report["timeout_seconds"] == 6 * HOUR


def test_an_unmeasured_dataset_does_not_drag_a_measured_one_into_a_guess():
    """A mixed launch is ungated as a whole rather than costed on partial data."""
    units, _ = spawn.measured_units(
        "graph-substrate", fake_module(HOUR, "family"), substrate_jobs("hotpotqa_clean", "webqsp")
    )
    assert units is None


# --------------------------------------------------------------------------
# Where the gate sits
# --------------------------------------------------------------------------


def test_the_gate_runs_before_anything_is_deployed_or_spawned():
    """A gate after deploy_app would refuse a run that is already billing."""
    import inspect

    source = inspect.getsource(spawn.main)
    assert source.index("gate_launch") < source.index("deploy_app(")
    assert source.index("gate_launch") < source.index(".spawn(")


def test_a_dry_run_reports_the_budget_so_it_can_be_checked_before_launching():
    import inspect

    source = inspect.getsource(spawn.main)
    dry_run_branch = source[source.index("if args.dry_run:") :]
    assert '"budget": budget' in dry_run_branch
