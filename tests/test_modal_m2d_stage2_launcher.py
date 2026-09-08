"""M2D Stage 2's launcher, on the decisions it makes before a container exists.

Stage 1's launcher test asks which cells, which arms, which seed, which box and
what each job must reproduce. Stage 2 inherits all of that and adds one thing
that can go wrong in a way nothing downstream would catch.

**The seed has to reach the checkpoint.** M2B's headline tree is seed 0; its
resolution tree is amendment 3's seeds 1 and 2. A launcher that pointed at the
headline would re-score seed 0's S4 weights inside a container labelled seed 1,
and every artifact would be well-formed: right panel, right arm, right seed in
the payload, and a native S4 column that is a constant across the three seeds
the gate is about to average over. Nothing in the gate can see that. So the
remote path this launcher builds is compared against the `artifact_root` M2B's
own filed resolution artifacts recorded, and the expected recall@5 against that
seed's filed row.

The other addition is arithmetic. Stage 1 spawned one job per cell; Stage 2
spawns one per cell AND seed, so the job count, the compute record's per-unit
key and the spawner's cost model all have to be per (cell, seed) or the launch
is priced at half its size.

Everything else is checked the way Stage 1's is: the matrix is asserted to be
read rather than restated, and every gate is exercised against a mutated COPY
of the declaration, so a failing test never leaves the repository in a state
where a launch would be allowed.
"""

from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

yaml = pytest.importorskip("yaml")
pytest.importorskip("torch")
pytest.importorskip("modal")

from scripts import m2d_stage2_gate as gate
from scripts import modal_m2d_stage1_arms as stage_1_launcher
from scripts import modal_m2d_stage2_seeds as launcher
from scripts import run_m2d_stage1_arms as runner

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml").read_text(encoding="utf-8")
)
STAGE_2 = DECLARATION["stage_2"]
SOURCE = (REPO_ROOT / "scripts" / "modal_m2d_stage2_seeds.py").read_text(encoding="utf-8")
PACKAGE = "m2d-stage2-seeds"


@pytest.fixture
def jobs() -> list[dict]:
    return launcher._jobs(list(launcher.CELLS))


def filed_record() -> dict:
    return json.loads(launcher.COMPUTE_RECORD_PATH.read_text(encoding="utf-8"))


def mutated(monkeypatch, **changes):
    """A parsed copy of the declaration with gates or blocks changed.

    A copy rather than the file: a test that edited the declaration and then
    failed would leave a repository in which a launch is allowed.
    """

    config = copy.deepcopy(launcher.CONFIG)
    for path, value in changes.items():
        target = config
        *parents, leaf = path.split(".")
        for key in parents:
            target = target[key]
        target[leaf] = value
    monkeypatch.setattr(launcher, "CONFIG", config)
    return config


# ---------------------------------------------------------------------------
# The matrix is read, never restated
# ---------------------------------------------------------------------------


def test_the_cells_the_arm_and_the_seeds_are_the_declarations() -> None:
    assert {f"{d}/{r}" for d, r in launcher.CELLS.items()} == set(STAGE_2["cells"])
    assert list(launcher.ARMS) == STAGE_2["arms"] == ["A3_MINIMAL"]
    assert launcher.SEEDS == [int(seed) for seed in STAGE_2["seeds"]] == [1, 2]


def test_stage_1s_seed_is_reused_and_this_launcher_refuses_to_refit_it() -> None:
    """Seed 0's artifact is what Stage 1's own verdict was computed from. A
    launcher that spawned it would replace the row it is supposed to reuse."""

    assert launcher.REUSED_SEED == DECLARATION["stage_1"]["seeds"][0] == 0
    assert launcher.REUSED_SEED not in launcher.SEEDS
    assert launcher.REUSED_SEED == gate.REUSED_SEED
    assert tuple(launcher.SEEDS) == gate.NEW_SEEDS


def test_the_job_count_is_the_authorised_fit_count(jobs) -> None:
    """Section 15b authorises four fits: two cells at two seeds, one arm."""

    assert len(jobs) == len(launcher.CELLS) * len(launcher.SEEDS)
    assert len(jobs) == STAGE_2["new_fits"] == 4
    assert {(job["dataset"], job["seed"]) for job in jobs} == {
        (dataset, seed) for dataset in launcher.CELLS for seed in launcher.SEEDS
    }


def test_the_launcher_the_runner_and_the_gate_share_one_vocabulary() -> None:
    assert launcher.ARMS == (gate.ARM,)
    assert launcher.NATIVE_RUNG == runner.NATIVE_RUNG == gate.NATIVE
    assert launcher.STAGE_2_COMPLETE_STATUS == runner.STAGE_2_COMPLETE_STATUS
    assert launcher.STAGE_PREFIX == runner.STAGE_2


def test_the_stage_the_runner_stamps_is_the_one_the_gate_reads(tmp_path) -> None:
    """Three files have to agree on one string, and none of them imports it
    from the others: the runner stamps it, the launcher names it, and the gate
    refuses any artifact that does not carry it. So the agreement is checked by
    putting the runner's own status through the gate's reader.
    """

    table = runner.authorised_seeds()
    for seed in launcher.SEEDS:
        assert table[seed]["status"] == launcher.STAGE_2_COMPLETE_STATUS
        assert table[seed]["arms"] == launcher.ARMS
    assert table[launcher.REUSED_SEED]["status"] == runner.COMPLETE_STATUS

    cell = next(f"{d}/{r}" for d, r in launcher.CELLS.items())
    seed = launcher.SEEDS[0]
    fit = {
        "status": table[seed]["status"],
        "cell": cell,
        "arm": launcher.ARMS[0],
        "seed": seed,
        "test_split_read": False,
        "metrics": dict.fromkeys(gate.METRICS, 0.5),
        "parameters": {"added_semantic_parameters": gate.ADDED_SEMANTIC_PARAMETERS},
    }
    (tmp_path / "fit.json").write_text(json.dumps(fit), encoding="utf-8")
    loaded = gate.load_results(root=tmp_path, stage_1_root=None)
    assert set(loaded) == {(cell, seed)}

    (tmp_path / "fit.json").write_text(
        json.dumps(dict(fit, status="M2D_STAGE1_ARM_COMPLETE")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=launcher.STAGE_2_COMPLETE_STATUS):
        gate.load_results(root=tmp_path, stage_1_root=None)


def test_no_cell_matrix_is_typed_into_the_launcher() -> None:
    for dataset in launcher.CELLS:
        assert f'"{dataset}"' not in SOURCE, f"{dataset} is spelled in the launcher"


def test_the_launcher_declares_no_placement_of_its_own() -> None:
    assert "execution_placement" not in DECLARATION["launch_authorization"]
    assert launcher.execution_placement() == dict(
        yaml.safe_load(
            (REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").read_text(encoding="utf-8")
        )["launch_authorization"]["execution_placement"]
    )


# ---------------------------------------------------------------------------
# The seed reaches the checkpoint
# ---------------------------------------------------------------------------


def test_the_m2b_checkpoint_root_is_the_one_m2b_itself_recorded(jobs) -> None:
    """The strongest available check, and the reason this file exists.

    M2B's filed resolution artifacts carry the remote `artifact_root` its own
    containers wrote to. If this launcher's path and that string agree, the
    seed-1 and seed-2 S4 checkpoints are where it is about to look for them.
    """

    for job in jobs:
        filed = json.loads(
            (
                launcher.M2B_RESOLUTION
                / f"seed{job['seed']}"
                / f"{job['dataset']}.json"
            ).read_text(encoding="utf-8")
        )
        assert job["m2b_fits_remote"] == filed["artifact_root"]


def test_the_checkpoint_path_is_under_the_resolution_tree_not_the_headline(
    jobs,
) -> None:
    """The headline tree is seed 0. Re-scoring it under a seed-1 label would
    make the native S4 column a constant across the seeds being averaged."""

    for job in jobs:
        remote = job["m2b_fits_remote"]
        assert f"/{launcher.M2B_RESOLUTION_SUBTREE}/seed{job['seed']}/" in remote
        assert "/headline/" not in remote
        # Normalised: the args are built inside the container, where Path is
        # posix; the host builds the same string with backslashes.
        checkpoint = str(launcher._runner_args(job).s4_checkpoint).replace("\\", "/")
        assert checkpoint.endswith(
            f"{job['regime']}/{launcher.NATIVE_RUNG.lower()}/checkpoint.pt"
        )
        assert checkpoint.startswith(remote)


def test_a_seed_m2b_never_fit_has_no_checkpoint_root() -> None:
    with pytest.raises(ValueError, match="declared seeds"):
        launcher._m2b_fits_root("squad_clean", "0" * 64, seed=0)


def test_the_expected_recall_is_that_seeds_filed_row(jobs) -> None:
    for job in jobs:
        filed = json.loads(
            (
                launcher.M2B_RESOLUTION
                / f"seed{job['seed']}"
                / f"{job['dataset']}.json"
            ).read_text(encoding="utf-8")
        )
        rung = filed["cells"][job["regime"]]["rungs"][launcher.NATIVE_RUNG]
        assert int(rung["seed"]) == job["seed"]
        assert job["expects"]["s4_recall_at_5"] == pytest.approx(
            rung["metrics"]["recall@5"]
        )


def test_the_two_seeds_of_one_cell_expect_different_numbers(jobs) -> None:
    """If they did not, the seed would not have reached the checkpoint."""

    by_cell: dict[str, set[float]] = {}
    for job in jobs:
        by_cell.setdefault(job["dataset"], set()).add(job["expects"]["s4_recall_at_5"])
    for dataset, values in by_cell.items():
        assert len(values) == len(launcher.SEEDS), dataset


def test_the_expected_recall_is_also_the_baseline_tables_same_seed_row(jobs) -> None:
    """The gate pairs each Stage-2 fit against the baseline table's S3 and S4
    rows at the same seed. The container is told to reproduce M2B's artifact;
    if the table and the artifact disagreed, the gate's S4 column and the
    container's re-score would be two different numbers wearing one label."""

    table = json.loads(launcher.BASELINE_TABLE.read_text(encoding="utf-8"))
    rows = {
        (row["dataset"], row["regime"], row["rung"], int(row["seed"])): row
        for row in table["rows"]
    }
    for job in jobs:
        key = (job["dataset"], job["regime"], launcher.NATIVE_RUNG, job["seed"])
        assert rows[key]["recall@5"] == pytest.approx(job["expects"]["s4_recall_at_5"])


# ---------------------------------------------------------------------------
# The panel, which the seed must NOT reach
# ---------------------------------------------------------------------------


def test_every_seed_scores_the_panel_stage_1_scored(jobs) -> None:
    """`holdout_split` is a deterministic tail with no seed in it, so all three
    seeds score one panel. That is what makes a same-seed delta legal, and it
    is asserted rather than assumed."""

    digests = set()
    shared = set()
    for job in jobs:
        digests.add(job["expects"]["panel_sha256"])
        shared.add(job["expects"]["shared_inputs_sha256"])
    assert len(digests) == len(launcher.CELLS), "one panel digest per cell"

    for dataset, regime in launcher.CELLS.items():
        headline = json.loads(
            (launcher.M2B_HEADLINE / f"{dataset}.json").read_text(encoding="utf-8")
        )["cells"][regime]
        expected = launcher.panel_digest(list(headline["held_out_query_ids"]))
        assert expected in digests
        assert headline["shared_inputs"]["sha256"] in shared


def test_a_resolution_panel_that_had_drifted_is_refused(monkeypatch) -> None:
    """The check that makes the paragraph above enforceable rather than
    decorative. A drifted panel would turn every same-seed delta in this stage
    into a comparison across panels, and the place to find that out is here."""

    dataset, regime = next(iter(launcher.CELLS.items()))
    real = launcher._m2b_resolution_artifact(dataset, launcher.SEEDS[0])
    drifted = copy.deepcopy(real)
    drifted["cells"][regime]["held_out_query_ids"] = ["nope"]
    monkeypatch.setattr(
        launcher, "_m2b_resolution_artifact", lambda *_args, **_kw: drifted
    )
    with pytest.raises(ValueError, match="panel differs"):
        launcher.m2b_expectations(dataset, regime, launcher.SEEDS[0])


def test_drifted_structural_inputs_are_refused(monkeypatch) -> None:
    dataset, regime = next(iter(launcher.CELLS.items()))
    drifted = copy.deepcopy(launcher._m2b_resolution_artifact(dataset, launcher.SEEDS[0]))
    drifted["cells"][regime]["shared_inputs"]["sha256"] = "0" * 64
    monkeypatch.setattr(
        launcher, "_m2b_resolution_artifact", lambda *_args, **_kw: drifted
    )
    with pytest.raises(ValueError, match="structural inputs differ"):
        launcher.m2b_expectations(dataset, regime, launcher.SEEDS[0])


def test_the_panel_digest_is_the_runners_own_function() -> None:
    assert launcher.panel_digest(["a", "b"]) == runner.panel_digest(["a", "b"])


def test_the_reproduction_bound_is_stage_1s_derivation(jobs) -> None:
    """Imported, not respelled. Both Stage-2 cells carry three filed S4 seeds,
    so both take the per-cell branch -- the tighter one."""

    for job in jobs:
        expected, source = stage_1_launcher._s4_seed_spread_pp(
            job["dataset"], job["regime"]
        )
        assert job["expects"]["s4_reproduction_bound_pp"] == pytest.approx(expected)
        assert job["expects"]["s4_reproduction_bound_source"] == source
        assert "this cell's own" in source


# ---------------------------------------------------------------------------
# The gates refuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gate_name",
    [
        "stage_1_reported",
        "stage_2_amendment_filed",
        "stage_2_authorised",
        "stage_2_gate_committed",
        "stage_2_compute_record_filed",
    ],
)
def test_every_named_gate_is_a_precondition(monkeypatch, gate_name) -> None:
    mutated(monkeypatch, **{f"launch_authorization.gates.{gate_name}": False})
    with pytest.raises(SystemExit, match=gate_name):
        launcher.require_authorisation()


def test_a_fourth_seed_in_the_declaration_stops_this_launcher(monkeypatch) -> None:
    mutated(monkeypatch, **{"stage_2.seeds": [1, 2, 3]})
    with pytest.raises(SystemExit, match="this launcher spawns"):
        launcher.require_authorisation()


def test_a_second_arm_in_the_declaration_stops_this_launcher(monkeypatch) -> None:
    """A1 was Stage 1's attribution control and seeds cannot reopen what it
    answered. If the declaration ever names it here, this launcher stops
    rather than quietly spawning four fits of the arm it does know."""

    mutated(monkeypatch, **{"stage_2.arms": ["A1", "A3_MINIMAL"]})
    with pytest.raises(SystemExit, match="this launcher spawns"):
        launcher.require_authorisation()


def test_a_fit_count_that_no_longer_matches_the_matrix_stops_it(monkeypatch) -> None:
    mutated(monkeypatch, **{"stage_2.new_fits": 8})
    with pytest.raises(SystemExit, match="new fits"):
        launcher.require_authorisation()


@pytest.mark.parametrize(
    "clause",
    [
        "the full 14-cell M2D screen",
        "seeds 1 and 2 on any other arm, cell or regime",
        "a fourth seed",
    ],
)
def test_dropping_any_not_authorised_clause_stops_this_launcher(
    monkeypatch, clause
) -> None:
    remaining = [
        item
        for item in DECLARATION["launch_authorization"]["not_authorised"]
        if item != clause
    ]
    assert len(remaining) < len(DECLARATION["launch_authorization"]["not_authorised"])
    mutated(monkeypatch, **{"launch_authorization.not_authorised": remaining})
    with pytest.raises(SystemExit, match="whole authorised workload"):
        launcher.require_authorisation()


def test_a_placement_copy_appearing_in_m2d_is_refused(monkeypatch) -> None:
    mutated(monkeypatch, **{"launch_authorization.execution_placement": {"squad_clean": "x"}})
    with pytest.raises(SystemExit, match="delete the copy"):
        launcher.execution_placement()


def test_an_unknown_dataset_is_refused_before_any_job_is_built() -> None:
    with pytest.raises(ValueError, match="are not M2D Stage-2 cells"):
        launcher._jobs(["hotpotqa_clean"])


# ---------------------------------------------------------------------------
# The compute record
# ---------------------------------------------------------------------------


def with_record(monkeypatch, tmp_path, record) -> None:
    path = tmp_path / "stage2_compute_record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(launcher, "COMPUTE_RECORD_PATH", path)


def test_the_filed_record_is_read_and_accepted_as_it_stands() -> None:
    record = launcher.compute_record()
    assert record["container"]["gpu"] == launcher.GPU
    assert record["workload"]["fits"] == len(launcher.CELLS) * len(launcher.SEEDS)


def test_a_record_that_authorises_no_gpu_is_refused(monkeypatch, tmp_path) -> None:
    record = filed_record()
    record["container"]["gpu"] = None
    with_record(monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="authorises no GPU"):
        launcher.compute_record()


def test_a_record_that_prices_a_different_box_is_refused(monkeypatch, tmp_path) -> None:
    record = filed_record()
    record["container"]["cpu_cores"] = 8
    with_record(monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="disagree"):
        launcher.compute_record()


def test_a_record_that_prices_a_cell_but_not_a_seed_is_refused(
    monkeypatch, tmp_path
) -> None:
    """Per (cell, seed), not per cell. A record checked only by cell would
    admit a launch that ran a seed nobody priced."""

    record = filed_record()
    record["workload"]["cells"] = [
        dict(item, seed=launcher.SEEDS[0]) for item in record["workload"]["cells"]
    ]
    with_record(monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="nobody priced"):
        launcher.compute_record()


def test_a_record_that_is_not_a_pre_launch_record_is_refused(
    monkeypatch, tmp_path
) -> None:
    record = filed_record()
    record["filed_before_any_job_was_submitted"] = False
    with_record(monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="not a pre-launch record"):
        launcher.compute_record()


def test_a_missing_record_is_refused_at_the_moment_it_matters(
    monkeypatch, tmp_path
) -> None:
    """Absence is refused where a launch is decided, not at import: the module
    has to stay importable on a tree where the record was never generated."""

    monkeypatch.setattr(launcher, "COMPUTE_RECORD_PATH", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="does not exist"):
        launcher.require_authorisation()


def test_the_declaration_and_the_filed_record_agree() -> None:
    declared = DECLARATION["launch_authorization"]["stage_2_compute_record"]
    record = filed_record()
    container = record["container"]
    assert declared["gpu"] == container["gpu"]
    assert declared["cpu"] == container["cpu_cores"]
    assert declared["memory_mb"] == container["memory_mb"]
    assert declared["timeout_seconds"] == container["timeout_seconds"]
    assert declared["fits"] == record["workload"]["fits"]
    assert declared["cost_ceiling_usd"] == record["prediction"]["hard_ceiling_usd"]


def test_the_declared_record_points_at_files_that_exist() -> None:
    declared = DECLARATION["launch_authorization"]["stage_2_compute_record"]
    for key in ("document", "derived_by", "machine_readable"):
        assert (REPO_ROOT / declared[key]).exists(), key


# ---------------------------------------------------------------------------
# Where results land
# ---------------------------------------------------------------------------


def test_the_launcher_writes_only_under_stage_2s_own_prefix(jobs) -> None:
    for job in jobs:
        root = str(launcher._output_root(job))
        assert f"/{launcher.OUTPUT_PREFIX}/{launcher.STAGE_PREFIX}/" in root
        assert "/stage1/" not in root


def test_two_seeds_of_one_cell_do_not_address_one_result(jobs) -> None:
    """The store root carries no seed; run_artifacts' own seed segment is what
    separates them. If that segment were missing the second fit would land on
    the first and the gate would read a two-seed mean as a three-seed one."""

    prefixes = {
        str(launcher._remote_prefix(job, launcher.ARMS[0])): job["seed"] for job in jobs
    }
    assert len(prefixes) == len(jobs)
    for prefix, seed in prefixes.items():
        assert f"/seed_{seed}/" in prefix.replace("\\", "/") + "/"


def test_the_fetched_filename_carries_the_seed() -> None:
    """Two seeds of one cell are two rows of the same mean. A name that
    dropped the seed would let the second overwrite the first."""

    tree = ast.parse(SOURCE)
    fetch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "fetch"
    )
    joined = ast.dump(fetch)
    assert "_seed" in joined


def test_the_staged_path_stays_inside_windows_limit(jobs) -> None:
    job = jobs[0]
    remote = (
        f"{launcher._remote_prefix(job, launcher.ARMS[0])}/"
        f"{'a' * 32}/{launcher._artifact_names()[1]}"
    )
    suffix = launcher._staging_suffix(job, remote)
    local = REPO_ROOT / "outputs" / launcher.OUTPUT_PREFIX / "stage2" / "runs" / suffix
    assert len(str(local)) < 260


def test_a_fetch_that_picks_for_itself_is_refused() -> None:
    with pytest.raises(SystemExit, match="needs the commit"):
        launcher.fetch()


# ---------------------------------------------------------------------------
# The standing rules of this track
# ---------------------------------------------------------------------------


def test_the_launcher_is_registered_for_server_side_submission() -> None:
    from scripts.spawn_modal_jobs import PACKAGES

    module_name, stages = PACKAGES[PACKAGE]
    assert module_name == launcher.__name__
    assert stages == {"seeds": "run_stage2"}
    for function_name in stages.values():
        assert hasattr(launcher, function_name)


def test_the_launcher_names_no_test_split() -> None:
    assert "TEST" not in SOURCE.replace("LATEST", "")


def test_a_restart_redoes_one_fit() -> None:
    assert launcher.RESUME_GRANULARITY == "cell_seed"


def test_nothing_in_this_launcher_refits_a_sealed_rung() -> None:
    tree = ast.parse(SOURCE)
    names = {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg
    }
    assert "s4_checkpoint" in names
    assert "s3_checkpoint" not in names


def test_the_runner_module_is_stage_1s_because_the_architecture_is_frozen() -> None:
    """Section 3 freezes the architecture. A second runner would be a second
    implementation of a model that is not allowed to change."""

    assert launcher.RUNNER_MODULE == runner.__name__ == stage_1_launcher.RUNNER_MODULE


def test_the_training_setup_is_bit_for_bit_stage_1s(jobs) -> None:
    """A seed-to-seed difference means the seed only if nothing else moved."""

    fixed = (
        "holdout_fraction", "epochs", "batch_size", "dropout", "temperature",
        "learning_rate", "weight_decay", "latency_queries", "latency_repeats",
        "latency_warmup", "per_seed_cap", "neighbour_scan_cap_per_seed",
        "a64_mainline_family", "frozen_embedding_dim",
    )
    stage_1_jobs = {job["dataset"]: job for job in stage_1_launcher._jobs(["squad_clean"])}
    theirs = stage_1_launcher._runner_args(stage_1_jobs["squad_clean"])
    mine = launcher._runner_args(
        next(job for job in jobs if job["dataset"] == "squad_clean")
    )
    for field in fixed:
        assert getattr(mine, field) == getattr(theirs, field), field
    assert mine.cell_features == theirs.cell_features
    assert mine.queries == theirs.queries
    assert mine.seed != theirs.seed


def test_the_local_entrypoint_refuses_under_the_same_gates() -> None:
    tree = ast.parse(SOURCE)
    main = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    called = {
        node.func.id
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "check_execution_placement" in called


# ---------------------------------------------------------------------------
# The spawn registry's budget gate reads the record
# ---------------------------------------------------------------------------


def test_the_budget_gate_is_gated_at_all(jobs) -> None:
    """`gated: false` is not a refusal -- the spawner reports it and proceeds
    -- so the assertion is on the flag, not on an exception."""

    from scripts import spawn_modal_jobs

    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["gated"] is True, report.get("why")
    assert report["spend_unknown_because"] is None


def test_the_budget_gate_counts_one_unit_per_fit(jobs) -> None:
    """Stage 1 priced one unit per cell because one container held two arms.
    Here a container holds one fit, so four jobs are four units -- and a gate
    that collapsed them to two would price the stage at half its size."""

    from scripts import spawn_modal_jobs

    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["units"] == len(jobs) == 4


def test_the_budget_gate_reports_the_filed_numbers(jobs) -> None:
    from scripts import spawn_modal_jobs

    record = filed_record()
    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["container_usd_per_hour"] == pytest.approx(
        record["container"]["usd_per_hour"], abs=0.001
    )
    assert report["timeout_seconds"] == record["container"]["timeout_seconds"]
    assert report["largest_unit_hours"] * 3600 == pytest.approx(
        record["prediction"]["largest_single_job_seconds"], abs=2.0
    )


def test_the_whole_matrix_prices_to_the_records_own_compute_spend(jobs) -> None:
    """The record's headline figure also carries container overhead, which the
    spawner's gate does not model; comparing against it would fail for a
    reason that has nothing to do with this branch."""

    from scripts import spawn_modal_jobs

    record = filed_record()
    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["expected_spend_usd"] == pytest.approx(
        record["prediction"]["compute_spend_usd"], abs=0.01
    )
    overhead = record["prediction"]["container_overhead_usd"]
    assert report["expected_spend_usd"] + overhead == pytest.approx(
        record["prediction"]["expected_spend_usd"], abs=0.01
    )


def test_both_cells_live_on_one_workspace_so_one_dry_run_prices_the_stage() -> None:
    """Stage 1's cells straddled two workspaces, so neither of its dry runs
    ever displayed the whole spend. Stage 2's do not, and that is worth
    knowing before submitting rather than discovering at the prompt."""

    placement = launcher.execution_placement()
    workspaces = {placement[dataset] for dataset in launcher.CELLS}
    assert len(workspaces) == 1, f"Stage 2 straddles {sorted(workspaces)}"


def test_the_expected_spend_is_inside_the_declared_ceiling(jobs) -> None:
    from scripts import spawn_modal_jobs

    shape = DECLARATION["launch_authorization"]["stage_2_compute_record"]
    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["expected_spend_usd"] <= shape["cost_ceiling_usd"]


def test_the_utilisation_matches_the_one_the_record_was_derived_at() -> None:
    from scripts import m2d_stage2_compute_record as record_module
    from scripts import spawn_modal_jobs

    record = filed_record()
    assert spawn_modal_jobs.UTILISATION[PACKAGE] == record["prediction"]["utilisation_assumed"]
    assert spawn_modal_jobs.UTILISATION[PACKAGE] == record_module.UTILISATION


def test_nothing_has_been_submitted_yet() -> None:
    """The declaration's status is a claim about the disk. This launcher is
    committed before it is used, which is this track's standing order."""

    assert DECLARATION["status"] == "M2D_STAGE2_RECORD_FILED_NOTHING_SUBMITTED"
    assert not (REPO_ROOT / "outputs" / launcher.OUTPUT_PREFIX / "stage2").exists()
