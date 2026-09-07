"""The M2D Stage-0 launcher, against the declaration and the record it obeys.

Nothing here spawns anything. What it checks is the set of things that, when
wrong, produce a container that starts, bills, and either dies or -- worse --
finishes and writes a result nobody can interpret: an argument the runner does
not take, a checkpoint path that does not exist on the volume, a cell the
compute record never priced, a shape three times the one that was authorised.

The launcher/runner argument agreement test is the load-bearing one. The two
files are edited independently and the mismatch is invisible until a container
has already been billed for it.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

modal = pytest.importorskip("modal")

from scripts import m2d_stage0_compute_record as record_module
from scripts import modal_m2d_stage0_probe as launcher
from scripts import run_m2b_semantic_minimality as m2b
from scripts import run_m2d_primitive_probe as primitive_runner
from scripts import run_m2d_stage0_probe as runner
from scripts import spawn_modal_jobs

DECLARATION_PATH = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
DECLARATION = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
BASELINE_JSON = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)
PACKAGE = "m2d-stage0-probe"

#: The app hosts two runners, and every launcher/runner agreement below has to
#: hold for BOTH -- one namespace is built by one function and handed to
#: whichever module the stage names, so an argument the second runner does not
#: take is the same failure as an argument the first does not take, discovered
#: the same expensive way. Held against ``launcher.STAGES`` rather than listed
#: twice, so a third stage cannot be added without appearing here.
RUNNERS = {"stage0": runner, "primitives": primitive_runner}
STAGE_IDS = tuple(RUNNERS)

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


def _parser_dests(stage: str = "stage0") -> set[str]:
    parser = RUNNERS[stage].build_parser()
    return {action.dest for action in parser._actions if action.dest != "help"}


def test_every_stage_the_launcher_hosts_has_a_runner_this_test_covers() -> None:
    """The agreement checks below are only complete if this mapping is."""

    assert set(launcher.STAGES) == set(RUNNERS)
    for stage, module in RUNNERS.items():
        assert launcher.STAGES[stage]["module"] == module.__name__


@pytest.mark.parametrize("stage", STAGE_IDS)
def test_the_launcher_builds_exactly_the_runners_arguments(jobs, stage) -> None:
    """A missing or extra field is a container that starts, bills, and dies."""

    dests = _parser_dests(stage)
    built = set(vars(launcher._runner_args(jobs[0], stage)))
    assert built == dests, (
        f"launcher/{stage} argument drift: launcher-only={sorted(built - dests)}, "
        f"runner-only={sorted(dests - built)}"
    )


@pytest.mark.parametrize("stage", STAGE_IDS)
def test_no_launcher_argument_is_left_unset(jobs, stage) -> None:
    args = launcher._runner_args(jobs[0], stage)
    unset = sorted(
        name
        for name, value in vars(args).items()
        if value is None
        and name not in ("candidate_contract_compatibility", "source_commit", "run_id")
    )
    assert unset == [], unset


def test_the_run_id_is_left_for_the_container_to_choose(jobs) -> None:
    """It has to be, and the reason is the whole persistence contract.

    The run id is what makes two submissions of one cell address two paths. A
    host-chosen id would be the same string for every retry of a job, and the
    second write would hit an existing path -- refused if the store is honest,
    silently discarded if it is a Modal volume.
    """

    for stage, module in RUNNERS.items():
        assert launcher._runner_args(jobs[0], stage).run_id is None
        assert "current_run_id" in inspect.getsource(module)


def _runner_source(stage: str) -> str:
    return inspect.getsource(RUNNERS[stage])

#: Functions in ANOTHER module that the runner hands its WHOLE namespace to.
#: Their reads are the runner's reads, and the set is asserted below rather
#: than assumed, so a future call that passes ``args`` somewhere new fails
#: here instead of in a container.
FOREIGN_NAMESPACE_CONSUMERS = {"load_cell_under_contract": "m2b"}

#: Functions defined in the runner itself. Nothing to follow: their source is
#: already the source this test scans.
LOCAL_NAMESPACE_CONSUMERS = {"run"}


def _namespace_consumers(stage: str = "stage0") -> set[str]:
    """Every callee the runner passes the bare namespace to, by attribute name."""

    tree = ast.parse(_runner_source(stage))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        passes_namespace = any(
            isinstance(arg, ast.Name) and arg.id == "args" for arg in node.args
        ) or any(
            isinstance(kw.value, ast.Name) and kw.value.id == "args" for kw in node.keywords
        )
        if not passes_namespace:
            continue
        callee = node.func
        found.add(callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "?"))
    return found


def _attributes_read_on_the_run_path(stage: str = "stage0") -> set[str]:
    """Every ``args.X`` the run path reads, the runner's own and its callees'."""

    sources = [_runner_source(stage)]
    for name in sorted(FOREIGN_NAMESPACE_CONSUMERS):
        sources.append(inspect.getsource(getattr(m2b, name)))
    read: set[str] = set()
    for source in sources:
        read |= set(re.findall(r"\bargs\.([a-z_][a-z0-9_]*)", source))
    return read


@pytest.mark.parametrize("stage", STAGE_IDS)
def test_the_runner_hands_its_namespace_only_to_functions_this_test_follows(stage) -> None:
    """The read set below is only complete if this set is."""

    expected = set(FOREIGN_NAMESPACE_CONSUMERS) | LOCAL_NAMESPACE_CONSUMERS
    assert _namespace_consumers(stage) == expected


@pytest.mark.parametrize("stage", STAGE_IDS)
def test_every_args_attribute_the_run_path_reads_is_a_parser_dest(stage) -> None:
    """The failure this reproduces cost four containers.

    The runner hands its whole namespace to M2B's store loader, which reads
    per_seed_cap, neighbour_scan_cap_per_seed and a64_mainline_family off it to
    rebuild the cell's build key. None of the three was a flag on this parser.
    Every launcher-side check passed -- the launcher set exactly the arguments
    the parser declared -- and all four cells died on AttributeError after the
    data was loaded, which is the most expensive moment to die.

    Checking only that the launcher matches the parser cannot catch it. The
    reads are what must match the parser, and some of them are in another file.
    """

    missing = sorted(_attributes_read_on_the_run_path(stage) - _parser_dests(stage))
    assert missing == [], (
        f"the run path reads args.{{{','.join(missing)}}}, which this parser does not "
        "define. A container will raise AttributeError after loading the data."
    )


def test_the_frozen_build_key_is_m2s_and_is_not_set_by_m2d() -> None:
    """Section 1 freezes the scored universe. These three fields identify it.

    They are transcribed into the runner's defaults, so they are held equal to
    M2's declaration here -- a different value does not mislabel anything, it
    makes the sealed master refuse to load, or worse, load a different one.
    """

    frozen = yaml.safe_load(
        (REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").read_text(encoding="utf-8")
    )["qls_universal"]["hyperparameters"]
    assert launcher.BUILD_KEY is not None
    for field in ("per_seed_cap", "neighbour_scan_cap_per_seed"):
        assert int(launcher.BUILD_KEY[field]) == int(frozen[field])
    for module in RUNNERS.values():
        defaults = {
            action.dest: action.default
            for action in module.build_parser()._actions
            if action.dest != "help"
        }
        assert int(defaults["per_seed_cap"]) == int(frozen["per_seed_cap"])
        assert int(defaults["neighbour_scan_cap_per_seed"]) == int(
            frozen["neighbour_scan_cap_per_seed"]
        )


@pytest.mark.parametrize("stage", STAGE_IDS)
def test_the_launcher_passes_the_build_key_it_read_not_one_it_typed(jobs, stage) -> None:
    from scripts.run_m0b_regime_map import MAINLINE_FAMILY

    args = launcher._runner_args(jobs[0], stage)
    assert args.per_seed_cap == int(launcher.BUILD_KEY["per_seed_cap"])
    assert args.neighbour_scan_cap_per_seed == int(
        launcher.BUILD_KEY["neighbour_scan_cap_per_seed"]
    )
    assert args.a64_mainline_family == MAINLINE_FAMILY


@pytest.mark.parametrize("stage", STAGE_IDS)
def test_every_runner_flag_the_launcher_sets_is_one_the_run_path_reads(jobs, stage) -> None:
    """A knob nothing consults reads as a controlled variable and is a
    decoration. M2C shipped one; this checks M2D does not."""

    read = _attributes_read_on_the_run_path(stage)
    for name in vars(launcher._runner_args(jobs[0], stage)):
        assert name in read, f"nothing on the {stage} run path reads args.{name}"


def test_the_config_fingerprint_is_the_declarations_own_text(jobs) -> None:
    """It travels into the artifact, so it has to name the protocol that ran."""

    text = DECLARATION_PATH.read_text(encoding="utf-8")
    assert launcher.CONFIG_FINGERPRINT == hashlib.sha256(text.encode("utf-8")).hexdigest()
    for job in jobs:
        assert job["config_fingerprint"] == launcher.CONFIG_FINGERPRINT


def test_the_config_fingerprint_does_not_depend_on_line_endings(tmp_path) -> None:
    """The host checks this repo out with CRLF and the container reads LF.

    A fingerprint over the raw bytes would give one declaration two values
    depending on which side read it, and the field exists precisely to say that
    two artifacts ran under the same rules. Hashing the DECODED text makes the
    two agree, which this checks by hashing a genuinely CRLF copy through the
    same read the launcher performs.
    """

    original = DECLARATION_PATH.read_bytes()
    crlf = tmp_path / "declaration.yaml"
    crlf.write_bytes(original.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert crlf.read_bytes() != original.replace(b"\r\n", b"\n")

    same = hashlib.sha256(crlf.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    assert same == launcher.CONFIG_FINGERPRINT


# --------------------------------------------------------------------------
# Four cells, from the declaration
# --------------------------------------------------------------------------


def test_the_cells_come_from_the_declaration_and_are_not_restated() -> None:
    cells = DECLARATION["stage_0"]["cells"]
    declared = [*cells["failure_cells"], cells["passage_control"], cells["kb_control"]]
    assert sorted(f"{d}/{r}" for d, r in launcher.CELLS.items()) == sorted(declared)


def test_one_job_per_cell_and_a_restart_redoes_exactly_one(jobs) -> None:
    assert len(jobs) == len(launcher.CELLS) == 4
    assert launcher.RESUME_GRANULARITY == "cell"


def test_a_second_cell_on_one_dataset_is_refused_rather_than_overwritten() -> None:
    """One job per dataset is what lets a job be addressed by dataset. A second
    cell on an already-listed dataset has to fail at import, not at fetch."""

    source = (REPO_ROOT / "scripts" / "modal_m2d_stage0_probe.py").read_text(encoding="utf-8")
    assert "carries two declared Stage-0 cells" in source


def test_submitting_an_undeclared_cell_is_refused() -> None:
    with pytest.raises(ValueError, match="not M2D Stage-0 cells"):
        launcher._jobs(["webqsp"])


# --------------------------------------------------------------------------
# Zero GPU, and the shape the record priced
# --------------------------------------------------------------------------


def test_no_gpu_is_held(record) -> None:
    assert launcher.GPU is None
    assert launcher.run_stage0.spec.gpus is None
    assert record["container"]["gpu"] is None
    assert record["authorises_no_gpu"] is True
    assert DECLARATION["launch_authorization"]["stage_0_compute_record"]["gpu"] is None


def test_the_container_holds_the_shape_the_record_priced(record) -> None:
    assert launcher.run_stage0.spec.cpu == launcher.CPU == record["container"]["cpu_cores"]
    assert (
        launcher.run_stage0.spec.memory
        == launcher.MEMORY_MB
        == record["container"]["memory_mb"]
    )
    assert launcher.TIMEOUT_SECONDS == record["container"]["timeout_seconds"] == 3600


def test_the_shape_is_not_a_training_phases_shape() -> None:
    m1a = yaml.safe_load(
        (REPO_ROOT / "configs" / "m1a_feature_screen.yaml").read_text(encoding="utf-8")
    )["modal"]
    assert m1a["gpu"] == "A10G" and launcher.MODAL_CONFIG["gpu"] is None
    assert launcher.MODAL_CONFIG["cpu"] < m1a["cpu"]
    assert launcher.MODAL_CONFIG["timeout_seconds"] < m1a["timeout_seconds"]
    # But the infra keys stay M1A's, because they name where the artifacts are.
    for key in ("result_volume", "storage_root", "execution_label"):
        assert launcher.MODAL_CONFIG[key] == m1a[key]


def test_a_shape_that_disagrees_with_the_filed_record_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "CPU", 32)
    with pytest.raises(SystemExit, match="container shape"):
        launcher.compute_record()


def test_a_record_that_prices_a_different_set_of_cells_is_refused(monkeypatch) -> None:
    """The record is the authorisation. A cell it never priced is a job nobody
    authorised, whatever the declaration's cell block currently says."""

    monkeypatch.setitem(launcher.CELLS, "webqsp", "R1")
    with pytest.raises(SystemExit, match="prices"):
        launcher.compute_record()


# --------------------------------------------------------------------------
# The record is read on the host and never inside the container
# --------------------------------------------------------------------------


def test_the_record_is_not_read_at_import(monkeypatch, tmp_path) -> None:
    """It lives under outputs/, which the image does not carry. An import-time
    read would kill every remote job at startup."""

    assert not hasattr(launcher, "RECORD"), "the record is held at module scope"
    assert callable(launcher.compute_record)
    monkeypatch.setattr(launcher, "COMPUTE_RECORD_PATH", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="does not exist"):
        launcher.compute_record()


def test_the_container_never_needs_the_record(jobs, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launcher, "COMPUTE_RECORD_PATH", tmp_path / "absent.json")
    args = launcher._runner_args(jobs[0])
    assert args.queries > 0


def test_the_image_carries_no_outputs_tree() -> None:
    mounted = {
        Path(entry).name
        for entry in (REPO_ROOT / "scripts" / "modal_m2d_stage0_probe.py")
        .read_text(encoding="utf-8")
        .split("add_local_dir(str(RUNTIME_REPO_ROOT / ")[1:]
        for entry in [entry.split('"')[1]]
    }
    assert mounted == {"src", "scripts", "configs"}


# --------------------------------------------------------------------------
# Reads M2 and M2B, writes only M2D
# --------------------------------------------------------------------------


def test_both_sealed_rungs_are_read_and_from_m2bs_own_tree(jobs) -> None:
    """S3 as well as S4, because the question is what each finds that the other
    misses. Reading S3's weights re-opens nothing: M2B's selection is frozen."""

    for job in jobs:
        args = launcher._runner_args(job)
        for checkpoint, rung in ((args.s4_checkpoint, "s4"), (args.s3_checkpoint, "s3")):
            parts = PurePosixPath(checkpoint.as_posix()).parts
            assert "m2b_semantic_minimality" in parts
            assert parts[-3:] == (job["regime"], rung, "checkpoint.pt")
        assert args.s3_checkpoint != args.s4_checkpoint


def test_the_cell_master_comes_from_m2s_sealed_tree(jobs) -> None:
    for job in jobs:
        master = PurePosixPath(launcher._runner_args(job).cell_features.as_posix())
        assert "m2_qls_v2_freeze" in master.parts
        assert master.parts[-2:] == (job["regime"], "cell_features")


def test_every_write_lands_under_m2ds_own_prefix(jobs) -> None:
    for job in jobs:
        root = PurePosixPath(launcher._runner_args(job).output_root.as_posix())
        assert launcher.OUTPUT_PREFIX in root.parts
        assert "m2_qls_v2_freeze" not in root.parts
        assert "m2b_semantic_minimality" not in root.parts
        assert "m2c_s4_structural_conditioning" not in root.parts


def test_the_output_is_addressed_by_fingerprint_and_execution_label(jobs) -> None:
    for job in jobs:
        root = PurePosixPath(launcher._runner_args(job).output_root.as_posix())
        assert job["fingerprint"][:16] in root.parts
        assert launcher.MODAL_CONFIG["execution_label"] in root.parts


def test_the_output_root_is_a_store_and_not_a_file(jobs) -> None:
    """run_artifacts derives phase, cell, arm, commit and run id below it, which
    is what stops two submissions of one cell addressing one path."""

    for job in jobs:
        assert launcher._runner_args(job).output_root.suffix == ""


# --------------------------------------------------------------------------
# The panel is the one M2 scored and the record priced
# --------------------------------------------------------------------------


def test_the_panel_is_the_split_m2_scored_and_the_record_priced(jobs, record) -> None:
    table = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    rows = {
        (row["dataset"], row["regime"]): row
        for row in table["rows"]
        if row["seed"] == 0 and row["rung"] == "S4"
    }
    priced = {item["cell"]: item for item in record["workload"]["cells"]}
    for job in jobs:
        args = launcher._runner_args(job)
        row = rows[(job["dataset"], job["regime"])]
        assert args.queries == int(row["split_queries"]), (
            f"{job['dataset']}: the launcher asks for {args.queries} queries and M2B "
            f"scored {row['split_queries']}; the sealed master will refuse to load"
        )
        cell = priced[f"{job['dataset']}/{job['regime']}"]
        assert cell["panel_queries"] == args.queries - int(row["held_out_queries"])
        assert args.holdout_fraction == pytest.approx(
            int(row["held_out_queries"]) / args.queries, abs=0.001
        )


# --------------------------------------------------------------------------
# The workspace a job lands in
# --------------------------------------------------------------------------


def test_the_workspace_map_is_m2s_and_is_not_copied_into_m2d() -> None:
    assert "execution_placement" not in DECLARATION["launch_authorization"]
    placement = launcher.execution_placement()
    assert set(launcher.CELLS) <= set(placement)


def test_a_copy_of_the_map_in_m2d_is_refused_rather_than_preferred(monkeypatch) -> None:
    """It could only ever go stale: workspaces rotate as each hits its limit."""

    copied = dict(launcher.CONFIG["launch_authorization"])
    copied["execution_placement"] = {"metaqa": "somewhere-else"}
    monkeypatch.setitem(launcher.CONFIG, "launch_authorization", copied)
    with pytest.raises(SystemExit, match="its own execution_placement"):
        launcher.execution_placement()


def test_a_dataset_is_refused_under_a_workspace_its_artifacts_are_not_on(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_active_modal_profile", lambda: "not-the-right-workspace")
    with pytest.raises(SystemExit, match="not where these datasets"):
        launcher.check_execution_placement(["metaqa"])


def test_an_unreadable_profile_is_refused_rather_than_guessed(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_active_modal_profile", lambda: None)
    with pytest.raises(SystemExit, match="active Modal profile"):
        launcher.check_execution_placement(["metaqa"])


# --------------------------------------------------------------------------
# The gates, checked where the server-side path actually calls them
# --------------------------------------------------------------------------


def test_the_gates_are_checked_by_the_hook_the_server_side_path_calls(monkeypatch) -> None:
    """A gate only ``main`` consulted is a gate the declared route walks past."""

    called: list[str] = []
    monkeypatch.setattr(launcher, "require_authorisation", lambda: called.append("gates"))
    monkeypatch.setattr(launcher, "_active_modal_profile", lambda: None)
    with pytest.raises(SystemExit):
        launcher.check_execution_placement(["metaqa"])
    assert called == ["gates"]


def test_the_compute_record_gate_can_fail(monkeypatch) -> None:
    gates = dict(launcher.CONFIG["launch_authorization"]["gates"])
    gates["stage_0_compute_record_filed"] = False
    monkeypatch.setitem(launcher.CONFIG["launch_authorization"], "gates", gates)
    with pytest.raises(SystemExit, match="filed Stage-0 compute record"):
        launcher.require_authorisation()


def test_a_declaration_that_stops_authorising_these_diagnostics_refuses(monkeypatch) -> None:
    monkeypatch.setitem(
        launcher.CONFIG["launch_authorization"], "authorised", ["something else entirely"]
    )
    with pytest.raises(SystemExit, match="does not authorise"):
        launcher.require_authorisation()


def test_a_declaration_that_stops_restricting_the_workload_refuses(monkeypatch) -> None:
    """Four seed-0 cells is the whole authorised workload. If the declaration
    stopped forbidding the full screen and the extra seeds, this launcher could
    no longer claim the four jobs it spawns are all of it."""

    monkeypatch.setitem(launcher.CONFIG["launch_authorization"], "not_authorised", ["E2"])
    with pytest.raises(SystemExit, match="no longer restricts"):
        launcher.require_authorisation()


def test_the_gates_currently_pass() -> None:
    launcher.require_authorisation()


# --------------------------------------------------------------------------
# The spawn registry and its budget gate
# --------------------------------------------------------------------------


def test_the_package_is_registered_and_its_stages_resolve() -> None:
    """A stage the app defines but the spawner does not register can only be
    started with `modal run`, which is the thing the spawner exists to prevent:
    no placement gate, no authorisation gate, no filed budget. So the count is
    checked against the app, not just the names against a literal."""

    module_name, stages = spawn_modal_jobs.PACKAGES[PACKAGE]
    assert module_name == launcher.__name__
    assert stages == {"probe": "run_stage0", "primitives": "run_primitives"}
    assert len(stages) == len(launcher.STAGES), "an app stage nothing can launch"
    for function_name in stages.values():
        assert hasattr(launcher, function_name)


def test_the_budget_gate_reports_the_filed_numbers(jobs, record) -> None:
    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["gated"] is True
    assert report["units"] == 4, "one unit per cell, not one for the whole probe"
    assert report["expected_spend_usd"] == pytest.approx(
        record["prediction"]["expected_spend_usd"], abs=0.01
    )
    assert report["container_usd_per_hour"] == pytest.approx(
        record["container"]["usd_per_hour"], abs=0.001
    )
    assert report["timeout_seconds"] == record["container"]["timeout_seconds"]
    assert report["largest_unit_hours"] * 3600 == pytest.approx(
        record["prediction"]["largest_single_job_seconds"], abs=2.0
    )


def test_the_expected_spend_is_inside_the_declared_ceiling(jobs) -> None:
    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    ceiling = DECLARATION["launch_authorization"]["stage_0_compute_record"]["cost_ceiling_usd"]
    assert report["expected_spend_usd"] <= ceiling
    assert report["expected_spend_usd"] * 2 <= ceiling


def test_the_utilisation_matches_the_one_the_record_was_derived_at(record) -> None:
    assert spawn_modal_jobs.UTILISATION[PACKAGE] == record["prediction"]["utilisation_assumed"]
    assert spawn_modal_jobs.UTILISATION[PACKAGE] == record_module.UTILISATION


def test_a_job_larger_than_its_window_would_be_refused(jobs, record, monkeypatch) -> None:
    """The gate exists to catch a unit that bills a full window and finishes
    nothing. It must be able to fail, or it is decoration.

    The oversized unit is injected into the record rather than by shrinking the
    timeout, because the timeout is half of the shape ``compute_record``
    cross-checks and shrinking it would trip that refusal first.
    """

    oversized = {
        **record,
        "workload": {
            **record["workload"],
            "cells": [
                {**item, "seconds": 9_000.0} for item in record["workload"]["cells"]
            ],
        },
    }
    monkeypatch.setattr(launcher, "compute_record", lambda: oversized)
    with pytest.raises(SystemExit, match="REFUSED"):
        spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)


# --------------------------------------------------------------------------
# Fetching a result back
# --------------------------------------------------------------------------


def test_a_fetch_that_names_no_commit_is_refused() -> None:
    """Choosing between physical runs is a decision. A fetch that picked for
    itself would be free to report a run that never landed."""

    with pytest.raises(SystemExit, match="needs the commit"):
        launcher.fetch(expect_source_commit="")


def test_the_fetch_lists_the_store_rather_than_guessing_a_path() -> None:
    """The artifact path carries a run id the host did not choose, so there is
    no path to guess. Listing is not a convenience here; it is the only option
    that can find a result at all."""

    source = (REPO_ROOT / "scripts" / "modal_m2d_stage0_probe.py").read_text(encoding="utf-8")
    assert "listdir" in source
    assert "select_logical_result" in source
    assert "verify_artifact_file" in source


@pytest.mark.parametrize("stage", STAGE_IDS)
def test_the_fetch_prefix_is_the_logical_result_and_not_one_run(jobs, stage) -> None:
    prefix = launcher._remote_prefix(jobs[0], stage)
    assert prefix.parts[-4:] == (
        jobs[0]["dataset"],
        jobs[0]["regime"],
        "no_seed",
        RUNNERS[stage].ARM,
    )
    assert RUNNERS[stage].PHASE in prefix.parts


def test_the_two_stages_cannot_write_over_each_other(jobs) -> None:
    """What keeps two results of one cell apart on one store is the ARM, and
    the two stages ask different questions of the same panel. A shared arm
    would put a fusion result and a primitive result in one directory, where
    the fetch's own commit selection would be free to return either."""

    names = {stage: launcher._artifact_names(stage) for stage in STAGE_IDS}
    assert len({phase for phase, _, _ in names.values()}) == 1, "both stages are M2D"
    arms = {stage: arm for stage, (_, arm, _) in names.items()}
    assert len(set(arms.values())) == len(STAGE_IDS), arms
    prefixes = {str(launcher._remote_prefix(jobs[0], stage)) for stage in STAGE_IDS}
    assert len(prefixes) == len(STAGE_IDS)
    roots = {str(launcher._output_root(jobs[0], stage)) for stage in STAGE_IDS}
    assert len(roots) == len(STAGE_IDS), "and their stores are separate too"


def test_the_primitive_stage_is_priced_and_placed_like_the_one_it_follows() -> None:
    """It reuses Stage 0's filed record deliberately, so the gate has to admit
    the two failure cells under it rather than fall through to no cost model."""

    jobs = launcher._jobs(["squad_clean", "musique_clean"])
    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["gated"] is True
    assert report["units"] == 2
    assert "ceiling for its second stage" in report["basis"]


# --------------------------------------------------------------------------
# The fetch, on the path where it succeeds
# --------------------------------------------------------------------------


def _artifact_bytes(job: dict, run_id: str, commit: str, arms: int) -> tuple[str, bytes]:
    """One physical run: its remote path, and the bytes that belong there."""

    from mp_retrieval import run_artifacts

    phase, arm, filename = launcher._artifact_names()
    identity = run_artifacts.ArtifactIdentity(
        phase=phase,
        dataset=job["dataset"],
        regime=job["regime"],
        arm=arm,
        source_commit=commit,
        run_id=run_id,
    )
    payload = {
        "status": "M2D_STAGE0_COMPLETE",
        "cell": f"{job['dataset']}/{job['regime']}",
        "arms_scored": [f"arm_{index}" for index in range(arms)],
    }
    envelope = run_artifacts.build_envelope(
        identity, payload, config_fingerprint=launcher.CONFIG_FINGERPRINT, rows_at="arms_scored"
    )
    remote = str(
        launcher._output_root(job).joinpath(*identity.segments()) / filename
    )
    return remote, json.dumps(envelope).encode("utf-8")


@pytest.fixture
def staged(tmp_path, monkeypatch, jobs):
    """A volume holding two physical runs of one cell, at two commits."""

    job = jobs[0]
    wanted = "b" * 40
    stale = "a" * 40
    written = dict(
        [
            _artifact_bytes(job, "fc-STALE", stale, arms=3),
            _artifact_bytes(job, "fc-WANTED", wanted, arms=8),
        ]
    )

    monkeypatch.setattr(launcher, "HOST_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher, "_remote_artifacts", lambda prefix, stage=None: sorted(written)
    )
    # _jobs reads M2B's baselines from the repo, which the redirected root no
    # longer holds. The job itself is the real one, built above from the real
    # declaration; only the lookup is stubbed.
    monkeypatch.setattr(launcher, "_jobs", lambda requested: [job])

    def fake_download(remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(written[remote])

    monkeypatch.setattr(launcher, "_download", fake_download)
    return {"job": job, "wanted": wanted, "stale": stale, "root": tmp_path}


def test_the_fetch_selects_by_commit_and_reports_what_it_selected(staged) -> None:
    """The success path, which no test exercised until it failed on the host.

    Everything here is host-side: download, reopen, verify, select. The verify
    is the real one, so the artifact has to survive its own content digest,
    its own row recount, and the check that its recorded identity agrees with
    where it sits.
    """

    rows = launcher.fetch(staged["job"]["dataset"], expect_source_commit=staged["wanted"])
    assert len(rows) == 1
    row = rows[0]
    assert row["source_commit"] == staged["wanted"]
    assert row["run_id"] == "fc-WANTED"
    assert row["physical_runs_found"] == 2, "both runs are downloaded and verified"
    assert row["rows"] == 8, "the count is recounted from the selected run, not the other one"
    assert Path(row["local"]).name == f"{staged['job']['dataset']}_{staged['job']['regime']}.json"
    assert json.loads(Path(row["local"]).read_text(encoding="utf-8"))["identity"]["run_id"] == (
        "fc-WANTED"
    )


def test_a_commit_the_volume_does_not_hold_is_refused_rather_than_approximated(staged) -> None:
    with pytest.raises(Exception, match="c" * 8):
        launcher.fetch(staged["job"]["dataset"], expect_source_commit="c" * 40)


def test_the_staging_mirror_keeps_the_identity_and_drops_the_store_prefix(jobs) -> None:
    """What verify_artifact_file reads is the tail, and the head is where the
    file is already being put.

    Mirroring the whole remote path under the local staging root produced a
    Windows path over the 260-character limit, so the fetch died on mkdir --
    on the host, after four containers had succeeded.
    """

    job = jobs[0]
    remote, _ = _artifact_bytes(job, "fc-01M1Z11EZPD14032QSH1DV3CH6", "b" * 40, arms=8)
    suffix = launcher._staging_suffix(job, remote)

    assert suffix.parts[0] != "outputs", "the phase store prefix is not mirrored"
    assert suffix.name == launcher._artifact_names()[2]

    from mp_retrieval import run_artifacts

    segments = len(
        run_artifacts.ArtifactIdentity(
            phase="p", dataset="d", regime="R1", arm="a", source_commit="c" * 40, run_id="r"
        ).segments()
    )
    assert len(suffix.parts) == segments + 1, "identity segments plus the filename, exactly"

    local = launcher.HOST_REPO_ROOT / "outputs" / launcher.OUTPUT_PREFIX / "stage0" / "runs"
    assert len(str(local / suffix)) < 260, "a Windows path this fetch has to be able to create"
