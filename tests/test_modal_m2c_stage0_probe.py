"""M2C's Stage-0 launcher: what it would submit, and what it must refuse.

Nothing here starts a container. What it checks is the set of things that can
only be wrong once one has already started and been billed:

- the arguments the launcher builds are exactly the arguments the runner's own
  parser accepts, name for name;
- the four cells, the container shape and the panel all come from the
  declaration and the filed compute record rather than being restated here;
- zero GPU is held, because the amendment authorises zero GPU hours and the
  refusal has to be mechanical rather than a promise in a docstring;
- the gates are checked on the path the submission actually takes, which is
  spawn_modal_jobs.py's, not only on the local entrypoint nobody uses;
- and nothing under M2's, M2B's or any other phase's prefix is written.

One test here is about the container rather than the science and is the one
most likely to be broken by a careless edit: the compute record lives under
``outputs/``, which the image does not carry, so reading it at import would
kill every remote job at startup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

modal = pytest.importorskip("modal")

from scripts import modal_m2c_stage0_probe as launcher  # noqa: E402
from scripts import run_m2b_semantic_minimality as runner_m2b  # noqa: E402
from scripts import run_m2c_stage0_probe as runner  # noqa: E402
from scripts import spawn_modal_jobs  # noqa: E402

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2c_s4_structural_conditioning.yaml").read_text(encoding="utf-8")
)
PACKAGE = "m2c-stage0-probe"

pytestmark = pytest.mark.skipif(
    not launcher.COMPUTE_RECORD_PATH.is_file(),
    reason="the Stage-0 compute record has not been generated in this checkout",
)


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(launcher.COMPUTE_RECORD_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def jobs() -> list[dict]:
    return launcher._jobs(list(launcher.CELLS))


# --------------------------------------------------------------------------
# The launcher and the runner agree on the arguments
# --------------------------------------------------------------------------


def _parser_dests() -> set[str]:
    return {
        action.dest for action in runner.build_parser()._actions if action.dest != "help"
    }


def test_the_launcher_builds_exactly_the_runners_arguments(jobs) -> None:
    """A missing or extra field is a container that starts, bills, and dies."""

    built = set(vars(launcher._runner_args(jobs[0])))
    assert built == _parser_dests(), (
        f"launcher/runner argument drift: launcher-only={sorted(built - _parser_dests())}, "
        f"runner-only={sorted(_parser_dests() - built)}"
    )


def test_no_launcher_argument_is_left_unset(jobs) -> None:
    args = launcher._runner_args(jobs[0])
    unset = sorted(
        name
        for name, value in vars(args).items()
        if value is None and name not in ("candidate_contract_compatibility", "source_commit")
    )
    assert unset == [], unset


def test_every_runner_flag_the_launcher_sets_is_one_the_runner_reads(jobs) -> None:
    """``--a64-mainline-family`` was a declared flag the runner never consulted.

    A knob whose name asserts something about the arm -- which graph A64
    expands over -- and which changes nothing is worse than no knob: it reads
    as a controlled variable and is a decoration.
    """

    source = (REPO_ROOT / "scripts" / "run_m2c_stage0_probe.py").read_text(encoding="utf-8")
    for name in vars(launcher._runner_args(jobs[0])):
        assert f"args.{name}" in source, f"the runner never reads args.{name}"


# --------------------------------------------------------------------------
# Four cells, from the declaration
# --------------------------------------------------------------------------


def test_the_cells_come_from_the_declaration_and_are_not_restated() -> None:
    cells = DECLARATION["stage_0"]["cells"]
    declared = [
        *cells["failure_cells"],
        cells["passage_r3_control"],
        cells["kb_r3_control"],
    ]
    assert [f"{d}/{r}" for d, r in launcher.CELLS.items()] == declared
    assert launcher.CELLS == {
        "squad_clean": "R1",
        "musique_clean": "R1",
        "2wiki_clean": "R3",
        "metaqa": "R3",
    }
    assert cells["do_not_run_all_fourteen"] is True


def test_one_job_per_cell_and_a_restart_redoes_exactly_one(jobs) -> None:
    assert len(jobs) == 4
    assert launcher.RESUME_GRANULARITY == "cell"
    assert [job["dataset"] for job in jobs] == list(launcher.CELLS)


def test_a_second_cell_on_one_dataset_is_refused_rather_than_overwritten() -> None:
    """Jobs are addressed by dataset, so two cells on one would collide."""

    assert len(set(launcher.CELLS)) == len(launcher.CELLS) == 4
    assert "carries two declared Stage-0 cells" in (
        (REPO_ROOT / "scripts" / "modal_m2c_stage0_probe.py").read_text(encoding="utf-8")
    )


def test_submitting_an_undeclared_cell_is_refused() -> None:
    with pytest.raises(ValueError, match="not M2C Stage-0 cells"):
        launcher._jobs(["hotpotqa_clean"])


# --------------------------------------------------------------------------
# Zero GPU, and the shape the record priced
# --------------------------------------------------------------------------


def test_no_gpu_is_held(record) -> None:
    """The amendment authorises zero GPU hours; the refusal must be mechanical."""

    assert launcher.GPU is None
    assert launcher.run_stage0.spec.gpus is None
    assert record["container"]["gpu"] is None
    assert record["container"]["gpu_hours_authorised"] == 0.0


def test_the_container_holds_the_shape_the_record_priced(record) -> None:
    assert launcher.run_stage0.spec.cpu == launcher.CPU == record["container"]["cpu"]
    assert launcher.run_stage0.spec.memory == launcher.MEMORY_MB == record["container"]["memory_mb"]
    assert launcher.TIMEOUT_SECONDS == record["container"]["timeout_seconds"] == 3600


def test_the_shape_is_not_m1as_training_shape() -> None:
    """M1A's block is a training phase's. Pricing this probe at it would report
    a threefold spend against a declaration it never breached."""

    m1a = yaml.safe_load(
        (REPO_ROOT / "configs" / "m1a_feature_screen.yaml").read_text(encoding="utf-8")
    )["modal"]
    assert m1a["gpu"] == "A10G" and launcher.MODAL_CONFIG["gpu"] is None
    assert launcher.MODAL_CONFIG["cpu"] < m1a["cpu"]
    assert launcher.MODAL_CONFIG["timeout_seconds"] < m1a["timeout_seconds"]
    # But the infra keys stay M1A's, because they name where the artifacts are.
    for key in ("result_volume", "storage_root", "execution_label", "edge_provenance_root"):
        assert launcher.MODAL_CONFIG[key] == m1a[key]


def test_a_shape_that_disagrees_with_the_filed_record_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "CPU", 32)
    with pytest.raises(SystemExit, match="container shape"):
        launcher.compute_record()


# --------------------------------------------------------------------------
# The record is read on the host and never inside the container
# --------------------------------------------------------------------------


def test_the_record_is_not_read_at_import(monkeypatch, tmp_path) -> None:
    """It lives under outputs/, which the image does not carry.

    An import-time read would kill every remote job at startup -- Modal
    imports this module in the container to find ``run_stage0`` -- and would
    also make the module unimportable on a fresh clone, where the record has
    not been generated. Both failures look like infrastructure flakiness and
    neither points at the line that caused it.
    """

    assert not hasattr(launcher, "RECORD"), "the record is held at module scope"
    assert callable(launcher.compute_record)
    monkeypatch.setattr(launcher, "COMPUTE_RECORD_PATH", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="does not exist"):
        launcher.compute_record()


def test_the_container_never_needs_the_record(jobs, monkeypatch, tmp_path) -> None:
    """Everything record-derived that the remote side uses travels in the job."""

    monkeypatch.setattr(launcher, "COMPUTE_RECORD_PATH", tmp_path / "absent.json")
    args = launcher._runner_args(jobs[0])
    assert args.admission_cap > 0


def test_the_image_carries_no_outputs_tree() -> None:
    mounted = {
        Path(entry).name
        for entry in (REPO_ROOT / "scripts" / "modal_m2c_stage0_probe.py")
        .read_text(encoding="utf-8")
        .split("add_local_dir(str(RUNTIME_REPO_ROOT / ")[1:]
        for entry in [entry.split('"')[1]]
    }
    assert mounted == {"src", "scripts", "configs"}


def test_the_admission_cap_is_the_filed_panel_divided_by_the_cells(jobs, record) -> None:
    total = record["workload"]["admission_panel_total"]
    for job in jobs:
        assert job["admission_cap"] == total // len(launcher.CELLS) == 2000


# --------------------------------------------------------------------------
# Reads M2 and M2B, writes only M2C
# --------------------------------------------------------------------------


def test_the_cell_master_and_the_checkpoint_come_from_the_sealed_trees(jobs) -> None:
    for job in jobs:
        args = launcher._runner_args(job)
        master = PurePosixPath(args.cell_features.as_posix())
        checkpoint = PurePosixPath(args.s4_checkpoint.as_posix())
        assert "m2_qls_v2_freeze" in master.parts
        assert master.parts[-2:] == (job["regime"], "cell_features")
        assert "m2b_semantic_minimality" in checkpoint.parts
        # Lowercased, because that is the directory M2B actually wrote:
        # fit_root = cell_root / rung.lower(). On Linux the difference between
        # "S4" and "s4" is a missing checkpoint, not a cosmetic one.
        assert checkpoint.parts[-3:] == (job["regime"], "s4", "checkpoint.pt")
        assert launcher.S4_RUNG in runner_m2b.ALL_RUNGS
        assert launcher.S4_FIT_DIRECTORY == launcher.S4_RUNG.lower()


def test_every_write_lands_under_m2cs_own_prefix(jobs) -> None:
    for job in jobs:
        output = PurePosixPath(launcher._runner_args(job).output.as_posix())
        assert "m2c_s4_structural_conditioning" in output.parts
        assert "stage0" in output.parts
        for foreign in ("m2_qls_v2_freeze", "m2b_semantic_minimality", "m3", "package_f"):
            assert foreign not in output.parts


def test_the_output_is_addressed_by_fingerprint_and_execution_label(jobs) -> None:
    """So a rerun under a different data build cannot land on this one's result."""

    for job in jobs:
        output = PurePosixPath(launcher._runner_args(job).output.as_posix())
        assert job["fingerprint"][:16] in output.parts
        assert launcher.MODAL_CONFIG["execution_label"] in output.parts


def test_the_panel_is_the_split_m2_scored_and_the_record_priced(jobs, record) -> None:
    """And the priced panel is what M2B's own splitter returns, not a re-derivation.

    Re-deriving it here would have been an off-by-one on musique -- 3,189 by
    truncation against the 3,190 the real split produces -- and a compute record
    that predicts a different panel size than the run reads is a record that
    cannot be checked against the run.
    """

    from scripts.run_m2b_semantic_minimality import holdout_split

    priced = {
        job["dataset"]: job["development_panel_queries"] for job in record["estimate"]["jobs"]
    }
    for job in jobs:
        args = launcher._runner_args(job)
        assert args.queries == job["validation_split_queries"]
        # The runner then takes the development portion, leaving M2B's holdout
        # -- the surface its filed numbers come from -- unexamined.
        development, _held = holdout_split(list(range(args.queries)), args.holdout_fraction)
        assert len(development) == priced[job["dataset"]], job["dataset"]


# --------------------------------------------------------------------------
# Placement, declared once
# --------------------------------------------------------------------------


def test_the_workspace_map_is_m2s_and_is_not_copied_into_m2c() -> None:
    m2 = yaml.safe_load(
        (REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").read_text(encoding="utf-8")
    )
    assert launcher.execution_placement() == m2["launch_authorization"]["execution_placement"]
    assert "execution_placement" not in DECLARATION["launch_authorization"]


def test_a_copy_of_the_map_in_m2c_is_refused_rather_than_preferred(monkeypatch) -> None:
    """Workspaces rotate; a second copy could only ever go stale."""

    monkeypatch.setitem(
        launcher.CONFIG["launch_authorization"], "execution_placement", {"squad_clean": "stale"}
    )
    with pytest.raises(SystemExit, match="delete the copy"):
        launcher.execution_placement()


def test_a_dataset_is_refused_under_a_workspace_its_artifacts_are_not_on(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_PROFILE", "some_other_workspace")
    with pytest.raises(SystemExit, match="is not where these datasets"):
        launcher.check_execution_placement(["squad_clean"])


def test_an_unreadable_profile_is_refused_rather_than_guessed(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_active_modal_profile", lambda: None)
    with pytest.raises(SystemExit, match="cannot read the active Modal profile"):
        launcher.check_execution_placement(["squad_clean"])


# --------------------------------------------------------------------------
# The gates, on the path the submission actually takes
# --------------------------------------------------------------------------


def test_the_gates_are_checked_by_the_hook_the_server_side_path_calls(monkeypatch) -> None:
    """The load-bearing one. spawn_modal_jobs.py calls check_execution_placement
    before it deploys, and calls nothing else of the launcher's; a gate only
    ``main`` consulted would be a gate the declared route walks past."""

    called: list[bool] = []
    monkeypatch.setattr(launcher, "require_authorisation", lambda: called.append(True))
    monkeypatch.setenv("MODAL_PROFILE", "extra_wNzonK")
    launcher.check_execution_placement(["squad_clean"])
    assert called == [True]


def test_the_compute_record_gate_can_fail(monkeypatch) -> None:
    monkeypatch.setitem(
        launcher.CONFIG["launch_authorization"]["gates"], "stage_0_compute_record_filed", False
    )
    with pytest.raises(SystemExit, match="has not recorded a filed"):
        launcher.require_authorisation()


def test_a_declaration_that_does_not_authorise_execution_refuses(monkeypatch) -> None:
    monkeypatch.setitem(
        launcher.CONFIG["launch_authorization"], "authorised_by_this_declaration", ["tests"]
    )
    with pytest.raises(SystemExit, match="does not authorise Stage-0 Modal execution"):
        launcher.require_authorisation()


def test_a_declaration_that_stopped_forbidding_training_refuses(monkeypatch) -> None:
    """The probe trains nothing. If the declaration ever stops saying so, this
    launcher can no longer claim the thing it spawns is the zero-training one."""

    monkeypatch.setitem(
        launcher.CONFIG["launch_authorization"], "not_authorised", ["extra seeds"]
    )
    with pytest.raises(SystemExit, match="no longer forbids learned-transform training"):
        launcher.require_authorisation()


def test_the_gates_currently_pass() -> None:
    launcher.require_authorisation()


# --------------------------------------------------------------------------
# The spawn registry and its budget gate
# --------------------------------------------------------------------------


def test_the_package_is_registered_and_its_stage_resolves() -> None:
    module_name, stages = spawn_modal_jobs.PACKAGES[PACKAGE]
    assert module_name == "scripts.modal_m2c_stage0_probe"
    assert stages == {"probe": "run_stage0"}
    assert hasattr(launcher, "run_stage0")


def test_the_budget_gate_reports_the_filed_numbers(jobs, record) -> None:
    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["gated"] is True
    assert report["units"] == 4, "one unit per cell, not one for the whole probe"
    assert report["expected_spend_usd"] == pytest.approx(
        record["estimate"]["estimated_cost_usd"], abs=0.01
    )
    assert report["container_usd_per_hour"] == pytest.approx(
        record["container"]["usd_per_hour"], abs=0.001
    )
    assert report["timeout_seconds"] == record["container"]["timeout_seconds"]
    assert report["largest_unit_hours"] * 3600 == pytest.approx(
        record["estimate"]["longest_job_seconds"], abs=2.0
    )


def test_the_expected_spend_is_inside_the_declared_ceiling(jobs) -> None:
    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    ceiling = DECLARATION["launch_authorization"]["stage_0_compute_record"]["cost_ceiling_usd"]
    assert report["expected_spend_usd"] <= ceiling
    assert report["expected_spend_usd"] * 3 <= ceiling


def test_the_utilisation_matches_the_one_the_record_was_derived_at(record) -> None:
    assert spawn_modal_jobs.UTILISATION[PACKAGE] == record["estimate"]["utilisation"]


def test_a_job_larger_than_its_window_would_be_refused(jobs, record, monkeypatch) -> None:
    """The gate exists to catch a unit that bills a full window and finishes
    nothing. It must be able to fail, or it is decoration.

    The oversized unit is injected into the record rather than by shrinking the
    timeout, because the timeout is half of the shape ``compute_record``
    cross-checks and shrinking it would trip that refusal first -- proving the
    shape check works and leaving this one unexercised.
    """

    oversized = {
        **record,
        "estimate": {
            **record["estimate"],
            "jobs": [
                {**job, "job_seconds": launcher.TIMEOUT_SECONDS * 2}
                for job in record["estimate"]["jobs"]
            ],
        },
    }
    monkeypatch.setattr(launcher, "compute_record", lambda: oversized)
    with pytest.raises(SystemExit, match="REFUSED"):
        spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
