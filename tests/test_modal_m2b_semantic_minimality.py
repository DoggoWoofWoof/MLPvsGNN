"""M2B's launcher: what it would submit, and what it must refuse to submit.

Nothing here starts a container. What it checks is everything that can only be
wrong once one has already started and been billed:

- the arguments the launcher builds are exactly the arguments the runner's own
  parser accepts, name for name -- a mismatch is a job that dies after startup;
- the smoke's cell comes from the declaration rather than being restated here;
- M2's tree is passed read-only and M2B's writes land somewhere else entirely;
- one container per dataset, with every rung inside it, because the tie-break
  orders on a latency and a latency is a property of the machine that measured
  it;
- and both stages refuse while the declaration has not authorised a fit. That
  last one is the load-bearing test today: the declaration says outright that
  the smoke is a fit and needs a further dated amendment, so a launcher that
  would run it now is the bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

modal = pytest.importorskip("modal")

from scripts import modal_m2b_semantic_minimality as launcher  # noqa: E402
from scripts import run_m2b_semantic_minimality as runner  # noqa: E402
from scripts import spawn_modal_jobs  # noqa: E402

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml").read_text(encoding="utf-8")
)
PACKAGE = "m2b-semantic-minimality"


@pytest.fixture(scope="module")
def jobs():
    return launcher._jobs(list(launcher.ALL_DATASETS))


# --------------------------------------------------------------------------
# The launcher and the runner agree on the arguments
# --------------------------------------------------------------------------


def _parser_dests() -> set[str]:
    return {
        action.dest
        for action in runner.build_parser()._actions
        if action.dest != "help"
    }


@pytest.mark.parametrize("stage", ["smoke", "headline"])
def test_the_launcher_builds_exactly_the_runners_arguments(jobs, stage) -> None:
    """A missing or extra field is a container that starts, bills, and dies."""

    job = next(item for item in jobs if item["dataset"] == launcher.SMOKE_DATASET)
    args = launcher._runner_args(job, stage=stage)
    assert set(vars(args)) == _parser_dests(), (
        f"launcher/runner argument drift for stage {stage!r}: "
        f"launcher-only={sorted(set(vars(args)) - _parser_dests())}, "
        f"runner-only={sorted(_parser_dests() - set(vars(args)))}"
    )


def test_no_launcher_argument_is_left_unset(jobs) -> None:
    args = launcher._runner_args(jobs[0], stage="headline")
    unset = sorted(
        name for name, value in vars(args).items()
        if value is None and name not in ("candidate_contract_compatibility", "rungs")
    )
    assert unset == [], unset


def test_an_unknown_stage_is_refused(jobs) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        launcher._runner_args(jobs[0], stage="build")


def test_there_is_no_build_stage_because_m2b_builds_nothing() -> None:
    assert set(launcher.STAGE_PLAN) == {"smoke", "headline"}
    assert "build" not in launcher.STAGE_FUNCTIONS


# --------------------------------------------------------------------------
# The smoke is the declared cell, at the declared panel
# --------------------------------------------------------------------------


def test_the_smoke_cell_comes_from_the_declaration() -> None:
    declared = DECLARATION["smoke_before_fanout"]["cell"]
    assert f"{launcher.SMOKE_DATASET} / {launcher.SMOKE_REGIME}" == declared
    assert launcher.SMOKE_DATASET == "2wiki_clean"
    assert launcher.SMOKE_REGIME == "R3"


def test_the_smoke_fits_only_the_two_new_rungs(jobs) -> None:
    """S3 in that cell is a completed M2 fit; re-running it buys nothing."""

    job = next(item for item in jobs if item["dataset"] == launcher.SMOKE_DATASET)
    args = launcher._runner_args(job, stage="smoke")
    assert args.rungs == ["S2", "S4"]
    assert args.regimes == ["R3"]
    assert sorted(launcher.SMOKE_RUNGS) == sorted(runner.NEW_RUNGS)
    assert runner.REUSED_RUNG not in launcher.SMOKE_RUNGS


def test_the_smoke_runs_the_full_panel_not_a_hundred_queries(jobs) -> None:
    """Its job is to measure the S4 fit time; a token panel would not."""

    job = next(item for item in jobs if item["dataset"] == launcher.SMOKE_DATASET)
    args = launcher._runner_args(job, stage="smoke")
    assert args.queries == job["validation_split_queries"]
    assert args.queries > 1000


def test_smoking_a_dataset_the_declaration_did_not_name_is_refused() -> None:
    with pytest.raises(ValueError, match="the smoke is declared for"):
        launcher._smoke_scope("metaqa")


def test_the_smoke_regime_must_exist_in_the_matrix(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "SMOKE_REGIME", "R9")
    with pytest.raises(ValueError, match="not a declared M2B cell"):
        launcher._smoke_scope(launcher.SMOKE_DATASET)


def test_the_smoke_cell_is_one_with_a_real_node_role_column() -> None:
    """R3, so a broken structural path cannot pass unnoticed on a zero column."""

    assert launcher.SMOKE_REGIME == "R3"
    assert "R3" in DECLARATION["evaluation_matrix"]["cells"][launcher.SMOKE_DATASET]


# --------------------------------------------------------------------------
# One container per dataset, every rung inside it
# --------------------------------------------------------------------------


def test_one_job_per_dataset_not_one_per_fit(jobs) -> None:
    assert len(jobs) == 6
    assert launcher.CONTAINERS_PER_DATASET == 1
    assert launcher.RESUME_GRANULARITY == "dataset"
    assert [job["dataset"] for job in jobs] == list(launcher.ALL_DATASETS)


def test_the_six_containers_carry_the_declared_forty_two_evaluations(jobs) -> None:
    cells = sum(len(job["regimes"]) for job in jobs)
    assert cells == DECLARATION["evaluation_matrix"]["cell_count"] == 14
    assert cells * len(runner.ALL_RUNGS) == DECLARATION["evaluation_matrix"]["logical_matrix"] == 42
    assert cells * len(runner.NEW_RUNGS) == DECLARATION["workload"]["new"] == 28
    assert cells == DECLARATION["workload"]["reused"] == 14


def test_the_headline_runs_every_rung_so_the_rungs_share_a_clock(jobs) -> None:
    """Passing no --rungs is how the runner is told "all three"."""

    args = launcher._runner_args(jobs[0], stage="headline")
    assert args.rungs is None
    assert list(runner.ALL_RUNGS) == ["S2", "S3", "S4"]


def test_the_headline_runs_every_declared_regime_of_its_dataset(jobs) -> None:
    for job in jobs:
        args = launcher._runner_args(job, stage="headline")
        assert args.regimes == list(
            DECLARATION["evaluation_matrix"]["cells"][job["dataset"]]
        ), job["dataset"]


def test_every_stage_asks_for_a_gpu() -> None:
    """M2B has no CPU-only half: it builds no features, so every stage trains."""

    for name in launcher.STAGE_FUNCTIONS.values():
        spec = getattr(launcher, name).spec
        assert spec.gpus == launcher.MODAL_CONFIG["gpu"], name
        assert spec.cpu == launcher.MODAL_CONFIG["cpu"], name
        assert spec.memory == launcher.MODAL_CONFIG["memory_mb"], name


# --------------------------------------------------------------------------
# M2's tree is read; M2B's is written
# --------------------------------------------------------------------------


def test_the_cell_masters_are_read_from_m2s_completed_headline_tree(jobs) -> None:
    for job in jobs:
        args = launcher._runner_args(job, stage="headline")
        remote = PurePosixPath(str(args.cell_master_root).replace("\\", "/"))
        assert launcher.M2_OUTPUT_PREFIX in remote.parts, job["dataset"]
        assert remote.parts[-2:] == ("headline", "fits"), job["dataset"]


def test_m2b_writes_nowhere_inside_m2s_tree(jobs) -> None:
    """M2's artifacts are sealed inputs; a write into them would corrupt a phase."""

    for job in jobs:
        stages = ["headline"] + (["smoke"] if job["dataset"] == launcher.SMOKE_DATASET else [])
        for stage in stages:
            args = launcher._runner_args(job, stage=stage)
            written = str(args.artifact_root).replace("\\", "/")
            output = str(args.output).replace("\\", "/")
            sealed = str(args.cell_master_root).replace("\\", "/")
            assert launcher.OUTPUT_PREFIX in written
            assert launcher.M2_OUTPUT_PREFIX not in written, (job["dataset"], stage)
            assert launcher.M2_OUTPUT_PREFIX not in output, (job["dataset"], stage)
            assert not written.startswith(sealed), (job["dataset"], stage)


def test_s3s_weights_come_from_the_same_m2_tree_as_its_cell(jobs) -> None:
    for job in jobs:
        args = launcher._runner_args(job, stage="headline")
        assert args.m2_fits_root == args.cell_master_root, job["dataset"]


def test_the_reused_checkpoint_resolves_under_that_tree(jobs) -> None:
    job = next(item for item in jobs if "R3" in item["regimes"])
    args = launcher._runner_args(job, stage="headline")
    path = runner.reused_checkpoint_path(args, "R3")
    expected = Path(job["m2_fits_remote"]) / "R3" / "qls_universal" / "checkpoint.pt"
    assert path == expected


def test_the_smoke_and_the_headline_do_not_overwrite_each_other(jobs) -> None:
    job = next(item for item in jobs if item["dataset"] == launcher.SMOKE_DATASET)
    smoke = launcher._runner_args(job, stage="smoke")
    headline = launcher._runner_args(job, stage="headline")
    assert smoke.output != headline.output
    assert smoke.artifact_root != headline.artifact_root


# --------------------------------------------------------------------------
# The panel is M2's panel
# --------------------------------------------------------------------------


def test_the_queries_are_the_validation_split_m2_scored(jobs) -> None:
    """M2B's S3 column IS M2's fit, so a different panel makes it incomparable."""

    for job in jobs:
        args = launcher._runner_args(job, stage="headline")
        assert args.queries == job["validation_split_queries"], job["dataset"]


def test_a_drifted_validation_split_stops_the_launch() -> None:
    confirmation = {"data": {"splits": {"validation": 1}}}
    with pytest.raises(ValueError, match="validation_split_queries"):
        launcher._validation_split_queries("2wiki_clean", confirmation)


def test_webqsps_panel_is_small_enough_to_disclose(jobs) -> None:
    """The n behind the point estimate that drove M2's macro gain.

    Not an assertion about the result -- an assertion that the launcher is
    honest about how few queries stand behind webqsp's numbers, so the
    diagnostic downstream has the right n to report.
    """

    job = next(item for item in jobs if item["dataset"] == "webqsp")
    args = launcher._runner_args(job, stage="headline")
    held_out = round(args.queries * args.holdout_fraction)
    assert held_out == 63


def test_the_seed_is_zero_and_the_hyperparameters_are_m2s(jobs) -> None:
    args = launcher._runner_args(jobs[0], stage="headline")
    assert args.seed == runner.DECLARED_SEED == 0
    assert (args.epochs, args.batch_size, args.dropout, args.temperature) == (3, 16, 0.2, 0.07)
    assert (args.learning_rate, args.weight_decay, args.holdout_fraction) == (1e-3, 1e-4, 0.2)
    assert (args.per_seed_cap, args.neighbour_scan_cap_per_seed) == (16, 4096)


def test_the_frozen_width_is_the_one_every_parameter_count_is_quoted_at(jobs) -> None:
    args = launcher._runner_args(jobs[0], stage="headline")
    assert args.frozen_embedding_dim == runner.FROZEN_EMBEDDING_DIM == 1536


# --------------------------------------------------------------------------
# Nothing runs until the declaration authorises a fit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["smoke", "headline"])
def test_both_stages_refuse_while_the_declaration_authorises_no_fit(stage) -> None:
    """Today's real gate. The declaration says the smoke is a fit; so is this."""

    assert DECLARATION["status"] == launcher.RECONNAISSANCE_STATUS
    with pytest.raises(RuntimeError, match="authorises no fit"):
        launcher._require_authorisation(stage)


def test_the_declaration_still_says_the_smoke_is_not_authorised() -> None:
    authorisation = DECLARATION["smoke_before_fanout"]["authorisation"]
    assert "NOT AUTHORISED BY THIS FILE" in authorisation
    assert "The smoke is a fit" in authorisation


@pytest.mark.parametrize("stage", ["smoke", "headline"])
def test_an_authorised_status_with_no_gates_is_still_refused(tmp_path, monkeypatch, stage) -> None:
    """An authorisation with nothing to check is not an authorisation."""

    path = tmp_path / "m2b.yaml"
    path.write_text(yaml.safe_dump({"status": launcher.AUTHORISED_STATUS}), encoding="utf-8")
    monkeypatch.setattr(launcher, "M2B_CONFIG_PATH", path)
    with pytest.raises(RuntimeError, match="files no launch_authorization.gates"):
        launcher._require_authorisation(stage)


def test_the_headline_refuses_while_any_gate_is_false(tmp_path, monkeypatch) -> None:
    path = tmp_path / "m2b.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "status": launcher.AUTHORISED_STATUS,
                "launch_authorization": {
                    "gates": {"amendment_filed": True, "smoke_passes": False}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "M2B_CONFIG_PATH", path)
    with pytest.raises(RuntimeError, match=r"\['smoke_passes'\] as not true"):
        launcher._require_authorisation("headline")


def test_the_smoke_is_not_gated_on_the_gates_it_exists_to_earn(tmp_path, monkeypatch) -> None:
    """Otherwise the smoke could never run first, and nothing could ever start."""

    path = tmp_path / "m2b.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "status": launcher.AUTHORISED_STATUS,
                "launch_authorization": {
                    "gates": {
                        "amendment_filed": True,
                        "selection_rule_frozen": True,
                        "feature_store_reuse_proved": True,
                        "s3_behaviour_reuse_proved": True,
                        "semantic_formulas_frozen": True,
                        "instrumentation_tests_pass": True,
                        "smoke_passes": False,
                        "compute_within_ceiling": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "M2B_CONFIG_PATH", path)
    launcher._require_authorisation("smoke")
    with pytest.raises(RuntimeError, match="as not true"):
        launcher._require_authorisation("headline")


def test_the_smoke_is_still_gated_on_the_proofs_that_precede_it(tmp_path, monkeypatch) -> None:
    path = tmp_path / "m2b.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "status": launcher.AUTHORISED_STATUS,
                "launch_authorization": {
                    "gates": {"amendment_filed": True, "feature_store_reuse_proved": False}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "M2B_CONFIG_PATH", path)
    with pytest.raises(RuntimeError, match="feature_store_reuse_proved"):
        launcher._require_authorisation("smoke")


def test_an_unrecognised_declaration_status_stops_the_module_at_import() -> None:
    """The import-time check covers a status neither known value matches."""

    source = (REPO_ROOT / "scripts" / "modal_m2b_semantic_minimality.py").read_text(
        encoding="utf-8"
    )
    assert 'if CONFIG["status"] not in (AUTHORISED_STATUS, RECONNAISSANCE_STATUS):' in source
    assert "re-check before launching" in source


# --------------------------------------------------------------------------
# Where the jobs are allowed to run
# --------------------------------------------------------------------------


def test_the_placement_is_inherited_from_m2_where_m2b_declares_none() -> None:
    """M2B reads M2's masters, which live on whichever volume M2 wrote them to."""

    placement = launcher.execution_placement()
    m2_placement = yaml.safe_load(
        (REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").read_text(encoding="utf-8")
    )["launch_authorization"]["execution_placement"]
    assert placement == m2_placement
    assert set(placement) == set(launcher.ALL_DATASETS)


def test_m2bs_own_placement_wins_when_it_declares_one(monkeypatch) -> None:
    monkeypatch.setitem(
        launcher.CONFIG, "launch_authorization", {"execution_placement": {"webqsp": "elsewhere"}}
    )
    assert launcher.execution_placement() == {"webqsp": "elsewhere"}


def test_submitting_under_the_wrong_workspace_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_PROFILE", "a-workspace-that-holds-no-masters")
    with pytest.raises(SystemExit, match="REFUSED"):
        launcher.check_execution_placement(["2wiki_clean"])


def test_the_refusal_says_why_a_misplaced_m2b_job_is_worse_than_m2s(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_PROFILE", "wrong")
    with pytest.raises(SystemExit, match="M2B cannot build a master"):
        launcher.check_execution_placement(["metaqa"])


def test_an_unplaced_dataset_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_PROFILE", "anything")
    monkeypatch.setattr(launcher, "execution_placement", lambda: {"webqsp": "anything"})
    with pytest.raises(SystemExit, match="does not say where"):
        launcher.check_execution_placement(["webqsp", "metaqa"])


def test_submitting_under_the_declared_workspace_is_allowed(monkeypatch) -> None:
    placement = launcher.execution_placement()
    monkeypatch.setenv("MODAL_PROFILE", placement["2wiki_clean"])
    checked = launcher.check_execution_placement(["2wiki_clean"])
    assert checked["profile"] == placement["2wiki_clean"]
    assert checked["placement_source"] == "m2"


def test_an_unreadable_profile_is_refused_rather_than_guessed(monkeypatch) -> None:
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    monkeypatch.setattr(launcher, "_active_modal_profile", lambda: None)
    with pytest.raises(SystemExit, match="cannot determine the active Modal profile"):
        launcher.check_execution_placement(["webqsp"])


def test_an_undeclared_dataset_never_becomes_a_job() -> None:
    with pytest.raises(ValueError, match="Undeclared M2B datasets"):
        launcher._jobs(["nq_open"])


# --------------------------------------------------------------------------
# The spawner knows about it
# --------------------------------------------------------------------------


def test_the_spawner_registers_this_package() -> None:
    assert PACKAGE in spawn_modal_jobs.PACKAGES
    module_name, stages = spawn_modal_jobs.PACKAGES[PACKAGE]
    assert module_name == "scripts.modal_m2b_semantic_minimality"
    assert stages == launcher.STAGE_FUNCTIONS


def test_every_registered_function_exists_on_the_module() -> None:
    _module, stages = spawn_modal_jobs.PACKAGES[PACKAGE]
    for stage, name in stages.items():
        assert hasattr(launcher, name), f"{stage} -> {name} does not exist"


def test_the_launcher_exposes_the_hooks_the_spawner_looks_for() -> None:
    assert callable(launcher.check_execution_placement)
    assert set(launcher.CONFIG["datasets"]) == set(launcher.ALL_DATASETS)
    assert callable(launcher._jobs)


def test_the_app_and_prefix_are_m2bs_own_not_m2s() -> None:
    assert launcher.APP_NAME.endswith("m2b-semantic-minimality")
    assert launcher.OUTPUT_PREFIX == "m2b_semantic_minimality"
    assert launcher.OUTPUT_PREFIX != launcher.M2_OUTPUT_PREFIX


def test_the_image_mounts_the_declaration_the_runner_reads() -> None:
    """run_m2b_semantic_minimality.declared_cells reads this file at run time."""

    assert launcher.M2B_CONFIG_PATH.name == "m2b_semantic_minimality.yaml"
    assert runner.DECLARATION_PATH.name == launcher.M2B_CONFIG_PATH.name


def test_the_runner_args_are_a_namespace_the_runner_can_consume(jobs) -> None:
    args = launcher._runner_args(jobs[0], stage="headline")
    assert isinstance(args, argparse.Namespace)
    assert isinstance(args.baseline, dict)
    assert "candidate_contract_sha256" in args.baseline
