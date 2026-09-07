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
from scripts import run_m2d_stage0_probe as runner
from scripts import spawn_modal_jobs

DECLARATION_PATH = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
DECLARATION = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
BASELINE_JSON = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)
PACKAGE = "m2d-stage0-probe"

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
    return {action.dest for action in runner.build_parser()._actions if action.dest != "help"}


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

    assert launcher._runner_args(jobs[0]).run_id is None
    assert "current_run_id" in (
        (REPO_ROOT / "scripts" / "run_m2d_stage0_probe.py").read_text(encoding="utf-8")
    )


RUNNER_SOURCE = (REPO_ROOT / "scripts" / "run_m2d_stage0_probe.py").read_text(encoding="utf-8")

#: Functions in ANOTHER module that the runner hands its WHOLE namespace to.
#: Their reads are the runner's reads, and the set is asserted below rather
#: than assumed, so a future call that passes ``args`` somewhere new fails
#: here instead of in a container.
FOREIGN_NAMESPACE_CONSUMERS = {"load_cell_under_contract": "m2b"}

#: Functions defined in the runner itself. Nothing to follow: their source is
#: already the source this test scans.
LOCAL_NAMESPACE_CONSUMERS = {"run"}


def _namespace_consumers() -> set[str]:
    """Every callee the runner passes the bare namespace to, by attribute name."""

    tree = ast.parse(RUNNER_SOURCE)
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


def _attributes_read_on_the_run_path() -> set[str]:
    """Every ``args.X`` the run path reads, the runner's own and its callees'."""

    sources = [RUNNER_SOURCE]
    for name in sorted(FOREIGN_NAMESPACE_CONSUMERS):
        sources.append(inspect.getsource(getattr(m2b, name)))
    read: set[str] = set()
    for source in sources:
        read |= set(re.findall(r"\bargs\.([a-z_][a-z0-9_]*)", source))
    return read


def test_the_runner_hands_its_namespace_only_to_functions_this_test_follows() -> None:
    """The read set below is only complete if this set is."""

    assert _namespace_consumers() == set(FOREIGN_NAMESPACE_CONSUMERS) | LOCAL_NAMESPACE_CONSUMERS


def test_every_args_attribute_the_run_path_reads_is_a_parser_dest() -> None:
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

    missing = sorted(_attributes_read_on_the_run_path() - _parser_dests())
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
    defaults = {
        action.dest: action.default
        for action in runner.build_parser()._actions
        if action.dest != "help"
    }
    assert int(defaults["per_seed_cap"]) == int(frozen["per_seed_cap"])
    assert int(defaults["neighbour_scan_cap_per_seed"]) == int(
        frozen["neighbour_scan_cap_per_seed"]
    )


def test_the_launcher_passes_the_build_key_it_read_not_one_it_typed(jobs) -> None:
    from scripts.run_m0b_regime_map import MAINLINE_FAMILY

    args = launcher._runner_args(jobs[0])
    assert args.per_seed_cap == int(launcher.BUILD_KEY["per_seed_cap"])
    assert args.neighbour_scan_cap_per_seed == int(
        launcher.BUILD_KEY["neighbour_scan_cap_per_seed"]
    )
    assert args.a64_mainline_family == MAINLINE_FAMILY


def test_every_runner_flag_the_launcher_sets_is_one_the_run_path_reads(jobs) -> None:
    """A knob nothing consults reads as a controlled variable and is a
    decoration. M2C shipped one; this checks M2D does not."""

    read = _attributes_read_on_the_run_path()
    for name in vars(launcher._runner_args(jobs[0])):
        assert name in read, f"nothing on the run path reads args.{name}"


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


def test_the_package_is_registered_and_its_stage_resolves() -> None:
    module_name, stages = spawn_modal_jobs.PACKAGES[PACKAGE]
    assert module_name == "scripts.modal_m2d_stage0_probe"
    assert stages == {"probe": "run_stage0"}
    assert hasattr(launcher, "run_stage0")


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


def test_the_fetch_prefix_is_the_logical_result_and_not_one_run(jobs) -> None:
    prefix = launcher._remote_prefix(jobs[0])
    assert prefix.parts[-4:] == (
        jobs[0]["dataset"],
        jobs[0]["regime"],
        "no_seed",
        runner.ARM,
    )
    assert runner.PHASE in prefix.parts
