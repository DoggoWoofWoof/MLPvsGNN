"""M2D Stage 1's launcher, on everything that decides what gets submitted.

A launcher cannot be tested by launching, so what is checked here is the set of
decisions it makes before a container exists: which cells, which arms, which
seed, which box, and -- the part that is new in Stage 1 -- what each job is told
it must reproduce.

Three groups, and they fail for different reasons.

**The matrix is read, never restated.** Cells, arms and the seed come from the
declaration. A launcher carrying its own copy can only ever disagree with the
thing that declared it, so the copies are asserted absent rather than asserted
equal.

**The gates refuse.** Stage 0's compute-record gate is inverted here: Stage 1
trains, so a record authorising no accelerator has priced a different kind of
job and is refused. Every gate is exercised by mutating a parsed copy of the
declaration rather than by editing the file, so a failing test never leaves the
repository in a state where a launch would be allowed.

**The identity expectations are real.** Section 5 reuses M2B's filed rows, and
a delta against a filed row is meaningless unless this container scored the
same queries. The expectations that make that checkable are computed host-side
here; if they were wrong, every artifact would still be produced and would
still be uncomparable. So they are compared against M2B's filed headline
directly, and the panel digest is asserted to be the runner's own function
rather than a second spelling of it.
"""

from __future__ import annotations

import ast
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

from scripts import modal_m2d_stage1_arms as launcher
from scripts import run_m2d_stage1_arms as runner

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml").read_text(encoding="utf-8")
)
SOURCE = (REPO_ROOT / "scripts" / "modal_m2d_stage1_arms.py").read_text(encoding="utf-8")


@pytest.fixture
def jobs() -> list[dict]:
    return launcher._jobs(list(launcher.CELLS))


# ---------------------------------------------------------------------------
# The matrix is read, never restated
# ---------------------------------------------------------------------------


def test_the_cells_are_the_declarations() -> None:
    cells = DECLARATION["stage_1"]["cells"]
    declared = {*cells["mandatory_blockers"], *cells["controls"]}
    assert {f"{d}/{r}" for d, r in launcher.CELLS.items()} == declared


def test_the_arms_and_the_seed_are_the_declarations() -> None:
    assert list(launcher.ARMS) == DECLARATION["stage_1"]["arms"]
    assert launcher.SEED == DECLARATION["stage_1"]["seeds"][0]
    assert len(DECLARATION["stage_1"]["seeds"]) == 1


def test_the_launcher_and_the_runner_share_one_vocabulary() -> None:
    assert launcher.ARMS == runner.ARMS
    assert launcher.SEED == runner.DECLARED_SEED
    assert launcher.NATIVE_RUNG == runner.NATIVE_RUNG


def test_the_job_count_is_the_authorised_fit_count() -> None:
    """Section 5 authorises eight fits. Four cells times two arms is eight, and
    a launcher that spawned a ninth would be spending outside the
    authorisation regardless of what it found."""

    assert len(launcher.CELLS) * len(launcher.ARMS) == DECLARATION["stage_1"]["new_fits"]


def test_no_cell_matrix_is_typed_into_the_launcher() -> None:
    """The dataset names appear as data read from the declaration, not as a
    literal list this file could quietly diverge with."""

    for dataset in launcher.CELLS:
        assert f'"{dataset}"' not in SOURCE, f"{dataset} is spelled in the launcher"


def test_the_launcher_declares_no_placement_of_its_own() -> None:
    """The workspace map moves as workspaces hit their spend limits. A second
    copy could only ever go stale, so M2D reads M2's and refuses a copy."""

    assert "execution_placement" not in DECLARATION["launch_authorization"]
    assert launcher.execution_placement() == dict(
        yaml.safe_load(
            (REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").read_text(encoding="utf-8")
        )["launch_authorization"]["execution_placement"]
    )


# ---------------------------------------------------------------------------
# The gates refuse
# ---------------------------------------------------------------------------


def mutated(monkeypatch, **changes):
    """A parsed copy of the declaration with gates or blocks changed.

    A copy rather than the file: a test that edited the declaration and then
    failed would leave a repository in which a launch is allowed.
    """

    import copy

    config = copy.deepcopy(launcher.CONFIG)
    for path, value in changes.items():
        target = config
        *parents, leaf = path.split(".")
        for key in parents:
            target = target[key]
        target[leaf] = value
    monkeypatch.setattr(launcher, "CONFIG", config)
    return config


@pytest.mark.parametrize(
    "gate",
    [
        "stage_0_advance_gate_evaluated",
        "stage_1_amendment_filed",
        "stage_1_authorised",
        "stage_1_gate_committed",
        "stage_1_compute_record_filed",
    ],
)
def test_every_named_gate_is_a_precondition(monkeypatch, gate) -> None:
    mutated(monkeypatch, **{f"launch_authorization.gates.{gate}": False})
    with pytest.raises(SystemExit, match=gate):
        launcher.require_authorisation()


def test_an_extra_seed_in_the_declaration_stops_this_launcher(monkeypatch) -> None:
    """Section 15 puts seeds 1 and 2 AFTER the gate. If the declaration ever
    grows them, this launcher must stop rather than quietly spawn the first."""

    mutated(monkeypatch, **{"stage_1.seeds": [0, 1, 2]})
    with pytest.raises(SystemExit, match="seed 0 only"):
        launcher.require_authorisation()


def test_dropping_the_not_authorised_restrictions_stops_this_launcher(
    monkeypatch,
) -> None:
    mutated(monkeypatch, **{"launch_authorization.not_authorised": ["nothing"]})
    with pytest.raises(SystemExit, match="whole"):
        launcher.require_authorisation()


def test_a_placement_copy_appearing_in_m2d_is_refused(monkeypatch) -> None:
    mutated(monkeypatch, **{"launch_authorization.execution_placement": {"squad_clean": "x"}})
    with pytest.raises(SystemExit, match="delete the copy"):
        launcher.execution_placement()


# ---------------------------------------------------------------------------
# The compute record, whose GPU rule is Stage 0's inverted
# ---------------------------------------------------------------------------


def filed_record() -> dict:
    return json.loads(launcher.COMPUTE_RECORD_PATH.read_text(encoding="utf-8"))


def with_record(monkeypatch, tmp_path, record) -> None:
    path = tmp_path / "stage1_compute_record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(launcher, "COMPUTE_RECORD_PATH", path)


def test_the_filed_record_is_read_and_accepted_as_it_stands() -> None:
    record = launcher.compute_record()
    assert record["container"]["gpu"] == launcher.GPU
    assert record["workload"]["fits"] == len(launcher.CELLS) * len(launcher.ARMS)


def test_a_record_that_authorises_no_gpu_is_refused(monkeypatch, tmp_path) -> None:
    """Stage 0's rule, inverted. Stage 0 refused to HOLD an accelerator; Stage
    1 refuses to launch without one, because a training phase priced at CPU
    rates has not been priced. Inheriting Stage 0's check unchanged would have
    made every Stage-1 launch impossible, and inheriting it deleted would have
    made the shape unchecked."""

    record = filed_record()
    record["container"]["gpu"] = None
    record["authorises_no_gpu"] = True
    with_record(monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="authorises no GPU"):
        launcher.compute_record()


def test_a_record_that_says_it_trains_nothing_is_refused(monkeypatch, tmp_path) -> None:
    record = filed_record()
    record["trains_nothing"] = True
    with_record(monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="trains nothing"):
        launcher.compute_record()


def test_a_record_that_is_not_pre_launch_is_refused(monkeypatch, tmp_path) -> None:
    record = filed_record()
    record["filed_before_any_job_was_submitted"] = False
    with_record(monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="not a pre-launch record"):
        launcher.compute_record()


@pytest.mark.parametrize(
    "field,value", [("cpu_cores", 2), ("memory_mb", 1024), ("timeout_seconds", 60)]
)
def test_a_shape_the_declaration_does_not_carry_is_refused(
    monkeypatch, tmp_path, field, value
) -> None:
    """The container is built from the declaration's copy and priced from the
    record's. If they disagree, one of the two numbers is describing a box
    nobody will run in."""

    record = filed_record()
    record["container"][field] = value
    with_record(monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="disagree"):
        launcher.compute_record()


def test_a_cell_nobody_priced_is_a_cell_nobody_authorised(monkeypatch, tmp_path) -> None:
    record = filed_record()
    record["workload"]["cells"] = record["workload"]["cells"][:2]
    with_record(monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="nobody priced"):
        launcher.compute_record()


def test_a_record_pricing_the_wrong_number_of_fits_is_refused(
    monkeypatch, tmp_path
) -> None:
    record = filed_record()
    record["workload"]["fits"] = 4
    with_record(monkeypatch, tmp_path, record)
    with pytest.raises(SystemExit, match="fits"):
        launcher.compute_record()


def test_an_absent_record_refuses_at_the_moment_it_matters(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launcher, "COMPUTE_RECORD_PATH", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="does not exist"):
        launcher.compute_record()


def test_the_container_shape_is_the_declarations_copy() -> None:
    declared = DECLARATION["launch_authorization"]["stage_1_compute_record"]
    assert (launcher.GPU, launcher.CPU, launcher.MEMORY_MB, launcher.TIMEOUT_SECONDS) == (
        declared["gpu"],
        declared["cpu"],
        declared["memory_mb"],
        declared["timeout_seconds"],
    )


def test_the_timeout_is_not_silently_inherited_from_m1a() -> None:
    """M1A allows twenty-four hours. The record deliberately does not, because
    a Stage-1 cell still running after an hour is doing something the
    prediction did not describe."""

    m1a = yaml.safe_load(
        (REPO_ROOT / "configs" / "m1a_feature_screen.yaml").read_text(encoding="utf-8")
    )["modal"]
    assert launcher.TIMEOUT_SECONDS < int(m1a["timeout_seconds"])
    assert launcher.MODAL_CONFIG["timeout_seconds"] == launcher.TIMEOUT_SECONDS


def test_the_predicted_largest_job_fits_the_timeout_it_will_run_under() -> None:
    """The two numbers come from one script and are checked against each other
    here, because a record predicting a job longer than its own timeout would
    authorise a spend that cannot produce a result."""

    record = launcher.compute_record()
    predicted = record["prediction"]["largest_single_job_seconds"]
    safety = record["prediction"]["container_safety_factor"]
    assert predicted * safety < launcher.TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# The identity expectations
# ---------------------------------------------------------------------------


def test_every_cell_carries_all_three_expectations(jobs) -> None:
    """Without these the container would still fit and still write, and every
    artifact would be uncomparable with the filed rows the gate reads."""

    for job in jobs:
        expects = job["expects"]
        assert len(expects["panel_sha256"]) == 64
        assert len(expects["shared_inputs_sha256"]) == 64
        assert 0.0 < expects["s4_recall_at_5"] <= 1.0
        assert expects["s4_reproduction_bound_pp"] > 0.0


def test_the_expectations_are_m2bs_filed_numbers_and_not_recomputed(jobs) -> None:
    for job in jobs:
        filed = json.loads(
            (launcher.M2B_HEADLINE / f"{job['dataset']}.json").read_text(encoding="utf-8")
        )["cells"][job["regime"]]
        assert job["expects"]["shared_inputs_sha256"] == filed["shared_inputs"]["sha256"]
        assert job["expects"]["s4_recall_at_5"] == pytest.approx(
            filed["rungs"]["S4"]["metrics"]["recall@5"]
        )
        assert job["expects"]["held_out_queries"] == filed["held_out_queries"]


def test_the_panel_digest_is_the_runners_own_function(jobs) -> None:
    """Two spellings of one hash is a bug that only ever surfaces as an
    identity mismatch on a paid container -- which is the failure the identity
    check exists to prevent."""

    for job in jobs:
        ids = json.loads(
            (launcher.M2B_HEADLINE / f"{job['dataset']}.json").read_text(encoding="utf-8")
        )["cells"][job["regime"]]["held_out_query_ids"]
        assert job["expects"]["panel_sha256"] == runner.panel_digest(ids)


def test_a_cell_with_three_seeds_takes_its_own_band() -> None:
    """The blockers have three filed seeds, so the bound is that cell's own
    measured spread -- the same quantity the gate calls resolvable."""

    from scripts import m2d_stage1_gate as gate

    rows = gate.baseline_rows()
    for cell in DECLARATION["stage_1"]["cells"]["mandatory_blockers"]:
        dataset, regime = cell.split("/")
        bound, why = launcher._s4_seed_spread_pp(dataset, regime)
        assert bound == pytest.approx(gate.seed_spread_pp(rows, cell))
        assert "this cell's own" in why


def test_a_cell_with_one_seed_says_its_band_is_borrowed() -> None:
    """The controls have one filed seed, so no per-cell spread exists. The
    bound is the widest measured anywhere, and the difference between the two
    kinds of number is written into the job rather than smoothed over."""

    for cell in DECLARATION["stage_1"]["cells"]["controls"]:
        dataset, regime = cell.split("/")
        bound, why = launcher._s4_seed_spread_pp(dataset, regime)
        assert "one filed seed" in why and "anywhere" in why
        assert bound > 0.0


def test_the_borrowed_band_is_no_tighter_than_any_measured_one() -> None:
    """A fallback that came out tighter than a real per-cell band would be a
    stricter test applied where there is LESS evidence, which is backwards."""

    per_cell = [
        launcher._s4_seed_spread_pp(*cell.split("/"))[0]
        for cell in DECLARATION["stage_1"]["cells"]["mandatory_blockers"]
    ]
    borrowed = launcher._s4_seed_spread_pp(
        *DECLARATION["stage_1"]["cells"]["controls"][0].split("/")
    )[0]
    assert borrowed >= max(per_cell)


def test_a_cell_with_no_filed_row_cannot_be_launched() -> None:
    with pytest.raises(ValueError, match="no filed S4 row"):
        launcher._s4_seed_spread_pp("squad_clean", "R9")


def test_a_dataset_with_no_filed_headline_refuses_rather_than_guesses(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(launcher, "M2B_HEADLINE", tmp_path)
    with pytest.raises(FileNotFoundError, match="cannot be read cannot be launched"):
        launcher.m2b_expectations("squad_clean", "R1")


# ---------------------------------------------------------------------------
# What the container is actually told
# ---------------------------------------------------------------------------


def test_the_launcher_supplies_every_argument_the_runner_requires(jobs) -> None:
    """Compared against the runner's own parser, not against a list retyped
    here. A required argument the launcher forgets is a container that starts,
    bills, and dies on an AttributeError."""

    parser = runner.build_parser()
    required = {
        action.dest
        for action in parser._actions
        if action.required and action.dest != "help"
    }
    supplied = vars(launcher._runner_args(jobs[0]))
    missing = sorted(required - set(supplied))
    assert not missing, missing


def test_every_training_argument_matches_the_runners_default(jobs) -> None:
    """Section 12's list of things not to reopen. The runner's defaults are
    already held equal to M2B's parser and to the values M2B's launcher
    actually passed, so agreeing with them is agreeing with M2B."""

    defaults = {
        action.dest: action.default for action in runner.build_parser()._actions
    }
    supplied = vars(launcher._runner_args(jobs[0]))
    for name in (
        "holdout_fraction", "epochs", "batch_size", "dropout", "temperature",
        "learning_rate", "weight_decay", "latency_queries", "latency_repeats",
        "latency_warmup",
    ):
        assert supplied[name] == defaults[name], name


def test_the_seed_and_arms_reach_the_container(jobs) -> None:
    args = launcher._runner_args(jobs[0])
    assert args.seed == launcher.SEED
    assert list(args.arms) == list(launcher.ARMS)


def test_the_checkpoint_path_uses_m2bs_lowercase_layout(jobs) -> None:
    """M2B wrote ``fit_root = cell_root / rung.lower()``. On the container's
    filesystem the difference between S4 and s4 is a missing file, and the
    runner refuses on a missing checkpoint rather than dropping section 8's
    analysis -- so the error would be correct and the container still paid."""

    args = launcher._runner_args(jobs[0])
    parts = args.s4_checkpoint.parts
    assert "s4" in parts and "S4" not in parts
    assert parts[-1] == "checkpoint.pt"


def test_the_frozen_embedding_dim_is_m1as_declared_one(jobs) -> None:
    """Every parameter count section 11 reports is a function of this number,
    and the runner refuses if the loaded embeddings disagree with it."""

    m1a = yaml.safe_load(
        (REPO_ROOT / "configs" / "m1a_feature_screen.yaml").read_text(encoding="utf-8")
    )
    declared = m1a["base"]["semantic_rung"]["frozen_embedding_dim"]
    assert launcher._runner_args(jobs[0]).frozen_embedding_dim == declared


def test_the_build_key_is_m2s_and_not_this_launchers(jobs) -> None:
    """These three decide whether the sealed cell master loads at all. A
    different value does not mislabel the arm; it makes the store refuse."""

    build_key = yaml.safe_load(
        (REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").read_text(encoding="utf-8")
    )["qls_universal"]["hyperparameters"]
    args = launcher._runner_args(jobs[0])
    assert args.per_seed_cap == build_key["per_seed_cap"]
    assert args.neighbour_scan_cap_per_seed == build_key["neighbour_scan_cap_per_seed"]


# ---------------------------------------------------------------------------
# Where the results land
# ---------------------------------------------------------------------------


def test_the_remote_prefix_uses_run_artifacts_own_seed_spelling(jobs) -> None:
    """Not "seed0". run_artifacts renders the segment as "seed_0", and a
    launcher that spelled it differently would list an empty prefix and report
    that a completed cell had produced nothing. This test exists because that
    was the first spelling written."""

    from mp_retrieval.run_artifacts import logical_segments, seed_segment

    prefix = launcher._remote_prefix(jobs[0], "A1")
    assert seed_segment(launcher.SEED) in prefix.parts
    assert prefix.parts[-1] == "A1"
    assert prefix.parts[-5:] == logical_segments(
        phase=runner.PHASE,
        dataset=jobs[0]["dataset"],
        regime=jobs[0]["regime"],
        arm="A1",
        seed=launcher.SEED,
    )


def test_the_two_arms_of_one_cell_do_not_share_a_prefix(jobs) -> None:
    """Two artifacts per cell, and run_artifacts refuses to overwrite. If both
    arms addressed one prefix the second write would fail -- or worse, the
    fetch would select between two arms as though they were two runs of one."""

    prefixes = {arm: launcher._remote_prefix(jobs[0], arm) for arm in launcher.ARMS}
    assert len(set(prefixes.values())) == len(launcher.ARMS)


def test_a_fetch_without_a_stated_commit_is_refused() -> None:
    """Selecting between physical runs is a decision. A fetch that picked for
    itself would be free to report a run that was never meant to stand."""

    with pytest.raises(SystemExit, match="needs the commit"):
        launcher.fetch(datasets="squad_clean")


def test_the_staging_suffix_stays_short_enough_for_this_host(jobs) -> None:
    """The full remote path mirrored under the local root runs past Windows'
    260-character limit, which is a fetch that fails after the science has
    already succeeded."""

    remote = str(launcher._remote_prefix(jobs[0], "A1") / "run" / "artifact.json")
    suffix = launcher._staging_suffix(jobs[0], remote)
    local = REPO_ROOT / "outputs" / launcher.OUTPUT_PREFIX / "stage1" / "runs" / suffix
    assert len(str(local)) < 260


# ---------------------------------------------------------------------------
# The standing rules of this track
# ---------------------------------------------------------------------------


def test_the_launcher_is_registered_for_server_side_submission() -> None:
    """Modal jobs are spawned through scripts/spawn_modal_jobs.py so the run
    outlives the client. A launcher missing from that registry can only be run
    the way this track forbids."""

    from scripts.spawn_modal_jobs import PACKAGES

    module, stages = PACKAGES["m2d-stage1-arms"]
    assert module == "scripts.modal_m2d_stage1_arms"
    assert set(stages.values()) <= set(dir(launcher))


def test_the_launcher_names_no_test_split() -> None:
    assert "TEST" not in SOURCE.replace("LATEST", "")


def test_a_restart_redoes_one_cell(jobs) -> None:
    """spawn_modal_jobs reads this to cost what a restart has to redo; silence
    there is costed as no resumption at all."""

    assert launcher.RESUME_GRANULARITY == "cell"


def test_the_launcher_writes_only_under_m2ds_own_prefix(jobs) -> None:
    for job in jobs:
        root = str(launcher._output_root(job))
        assert f"/{launcher.OUTPUT_PREFIX}/{launcher.STAGE_PREFIX}/" in root


def test_nothing_in_this_launcher_refits_a_sealed_rung() -> None:
    """Section 5's reuse rule, checked where it could be broken: the launcher
    passes exactly one checkpoint path, S4's, and passes no S3 path at all."""

    tree = ast.parse(SOURCE)
    names = {
        node.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg
    }
    assert "s4_checkpoint" in names
    assert "s3_checkpoint" not in names


def test_the_runner_module_the_launcher_imports_is_the_one_tested() -> None:
    """A launcher pointing at a different runner would be spawning code no
    test in this repository has read."""

    assert launcher.RUNNER_MODULE == runner.__name__


def test_the_local_entrypoint_refuses_under_the_same_gates() -> None:
    """The declared submission path is server-side, and the local fallback
    exists so the app is exercisable -- not so it can bypass a gate."""

    body = ast.parse(SOURCE)
    main = next(
        node
        for node in ast.walk(body)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    called = {
        node.func.id
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "check_execution_placement" in called


def test_an_unknown_dataset_is_refused_before_any_job_is_built() -> None:
    with pytest.raises(ValueError, match="are not M2D Stage-1 cells"):
        launcher._jobs(["2wiki_clean"])


def test_the_declared_record_points_at_files_that_exist() -> None:
    declared = DECLARATION["launch_authorization"]["stage_1_compute_record"]
    for key in ("document", "derived_by", "machine_readable"):
        assert (REPO_ROOT / declared[key]).exists(), key


def test_the_declaration_and_the_filed_record_agree() -> None:
    """They are written by one script. This is the test that makes
    'regenerate rather than reconcile them by hand' enforceable."""

    declared = DECLARATION["launch_authorization"]["stage_1_compute_record"]
    record = filed_record()
    container = record["container"]
    assert declared["gpu"] == container["gpu"]
    assert declared["cpu"] == container["cpu_cores"]
    assert declared["memory_mb"] == container["memory_mb"]
    assert declared["timeout_seconds"] == container["timeout_seconds"]
    assert declared["jobs"] == record["workload"]["jobs"]
    assert declared["fits"] == record["workload"]["fits"]
    assert declared["expected_spend_usd"] == pytest.approx(
        record["prediction"]["expected_spend_usd"], abs=0.005
    )
    assert declared["cost_ceiling_usd"] == pytest.approx(
        record["prediction"]["hard_ceiling_usd"]
    )


def test_the_filed_record_prices_a_gpu_and_says_so() -> None:
    record = filed_record()
    assert record["container"]["gpu"] is not None
    assert record["authorises_no_gpu"] is False
    assert record["trains_nothing"] is False
    assert record["new_fits"] == DECLARATION["stage_1"]["new_fits"]


def test_the_record_states_why_it_is_not_m2bs_number() -> None:
    """Section 6 forbids carrying $0.92 forward blindly. A recomputation that
    landed on a different figure without saying why the basis moved would be
    the same failure with a different number."""

    record = filed_record()
    why = record["why_this_is_not_m2bs_number"].lower()
    assert "0.92" in why
    assert record["prediction"]["expected_spend_usd"] != pytest.approx(0.92, abs=0.005)


def test_the_document_names_every_cell_it_prices() -> None:
    document = (REPO_ROOT / "docs" / "M2D_STAGE1_COMPUTE_RECORD.md").read_text(
        encoding="utf-8"
    )
    for dataset, regime in launcher.CELLS.items():
        assert f"{dataset}/{regime}" in document


def test_the_record_was_filed_before_the_commit_the_fits_ran_at() -> None:
    """The record is a prediction, and this is the durable way to say so.

    An earlier version asserted the results directory was empty. That was true
    when the record was filed and false the moment the stage it priced ran, so
    it was a test with a shelf life -- and it never checked the claim that
    matters anyway. `outputs/` is gitignored, so no artifact is in any commit
    and looking for one there would be vacuous.

    What IS checkable, and is the actual claim, is the ordering of two commits
    the artifacts and the record name themselves: the record was derived at a
    commit that is a strict ancestor of the commit the fits were produced at.
    A record filed at or after the fits would be a report of a spend that had
    already happened, and the gate it earns would be a formality.
    """

    record = filed_record()
    filed_at = record["source_commit"]
    assert record["filed_before_any_job_was_submitted"] is True

    root = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage1"
    artifacts = sorted(root.glob("*.json")) if root.exists() else []
    if not artifacts:
        pytest.skip("no Stage-1 artifact in this checkout; there is no ordering yet")

    import json as _json
    import subprocess

    def _payload(path):
        # The fetched file is an envelope carrying the identity the fetch
        # verified; the run's own record is inside it, the way the gate reads it.
        envelope = _json.loads(path.read_text(encoding="utf-8"))
        return envelope.get("payload", envelope)

    ran_at = {_payload(path)["provenance"]["source_commit"] for path in artifacts}
    assert len(ran_at) == 1, f"the fits do not agree on a commit: {sorted(ran_at)}"
    ran_at = ran_at.pop()
    assert ran_at != filed_at, (
        "the record names the same commit the fits ran at, so it was not filed "
        "before them"
    )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", filed_at, ran_at],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,  # the returncode IS the answer; a raise would lose the message
    )
    assert ancestry.returncode == 0, (
        f"the record was filed at {filed_at[:12]}, which is not an ancestor of "
        f"{ran_at[:12]}, the commit the eight fits were produced at"
    )


# ---------------------------------------------------------------------------
# The spawn registry's budget gate reads the record
# ---------------------------------------------------------------------------
#
# A filed record the launch gate does not consult is a prediction nobody is
# held to. This package was registered for spawning before it had a cost
# model, and the dry run said so in as many words -- "no measured cost model
# for m2d-stage1-arms", gated: false -- while passing every other gate and
# offering to submit. Section 6 exists to prevent exactly that: a launch whose
# spend nobody checked at submit time.


PACKAGE = "m2d-stage1-arms"


def test_the_package_is_registered_and_its_stage_resolves() -> None:
    from scripts import spawn_modal_jobs

    module_name, stages = spawn_modal_jobs.PACKAGES[PACKAGE]
    assert module_name == launcher.__name__
    assert stages == {"arms": "run_stage1"}
    for function_name in stages.values():
        assert hasattr(launcher, function_name)


def test_the_budget_gate_is_gated_at_all(jobs) -> None:
    """The regression this branch exists for. `gated: false` is not a refusal
    -- the spawner reports it and proceeds -- so the assertion is on the flag,
    not on an exception."""

    from scripts import spawn_modal_jobs

    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["gated"] is True, report.get("why")
    assert report["spend_unknown_because"] is None


def test_the_budget_gate_reports_the_filed_numbers(jobs) -> None:
    from scripts import spawn_modal_jobs

    record = filed_record()
    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["units"] == len(launcher.CELLS), "one unit per cell, not one per fit"
    assert report["container_usd_per_hour"] == pytest.approx(
        record["container"]["usd_per_hour"], abs=0.001
    )
    assert report["timeout_seconds"] == record["container"]["timeout_seconds"]
    assert report["largest_unit_hours"] * 3600 == pytest.approx(
        record["prediction"]["largest_single_job_seconds"], abs=2.0
    )


def test_the_whole_matrix_prices_to_the_records_own_compute_spend(jobs) -> None:
    """Gating every cell at once must return the record's compute figure.

    The record's headline `expected_spend_usd` also carries the container
    overhead, which the spawner's gate does not model; comparing against that
    number would fail for a reason that has nothing to do with the branch.
    """

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


def test_a_launch_is_priced_per_workspace_and_the_halves_sum_to_the_whole() -> None:
    """Stage 1's cells are split across two Modal workspaces, so the launch is
    two submissions and neither dry run ever displays the whole spend. The
    halves have to add up, or each submission looks affordable on its own
    while the stage as a whole was never priced.
    """

    from scripts import spawn_modal_jobs

    placement = launcher.execution_placement()
    by_workspace: dict[str, list[dict]] = {}
    for job in launcher._jobs(list(launcher.CELLS)):
        by_workspace.setdefault(placement[job["dataset"]], []).append(job)
    assert len(by_workspace) > 1, "one workspace: this test is checking nothing"

    halves = [
        spawn_modal_jobs.gate_launch(PACKAGE, launcher, subset)["expected_spend_usd"]
        for subset in by_workspace.values()
    ]
    whole = spawn_modal_jobs.gate_launch(
        PACKAGE, launcher, launcher._jobs(list(launcher.CELLS))
    )["expected_spend_usd"]
    assert sum(halves) == pytest.approx(whole, abs=0.01)


def test_the_expected_spend_is_inside_the_declared_ceiling(jobs) -> None:
    from scripts import spawn_modal_jobs

    shape = DECLARATION["launch_authorization"]["stage_1_compute_record"]
    report = spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
    assert report["expected_spend_usd"] <= shape["cost_ceiling_usd"]


def test_the_utilisation_matches_the_one_the_record_was_derived_at() -> None:
    """Two divisors, one number. If the spawner assumed a busier container than
    the record was derived at, the gate would report a spend the record does
    not predict and the ceiling would be checked against the wrong figure."""

    from scripts import m2d_stage1_compute_record as record_module
    from scripts import spawn_modal_jobs

    record = filed_record()
    assert spawn_modal_jobs.UTILISATION[PACKAGE] == record["prediction"]["utilisation_assumed"]
    assert spawn_modal_jobs.UTILISATION[PACKAGE] == record_module.UTILISATION


def test_a_dataset_the_record_does_not_price_is_refused_rather_than_estimated(
    jobs, monkeypatch
) -> None:
    """A cell missing from the record must stop the launch. Silently pricing
    the jobs it does know would report a spend for a smaller matrix than the
    one about to be submitted."""

    from scripts import spawn_modal_jobs

    record = filed_record()
    monkeypatch.setattr(
        launcher,
        "compute_record",
        lambda: {
            **record,
            "workload": {
                **record["workload"],
                "cells": record["workload"]["cells"][:1],
            },
        },
    )
    units, why = spawn_modal_jobs.measured_units(PACKAGE, launcher, jobs)
    assert units is None
    assert "prices no job for" in why


def test_a_job_larger_than_its_window_would_be_refused(jobs, monkeypatch) -> None:
    """The gate has to be able to fail, or it is decoration. A cell that bills
    a whole window and finishes nothing is the failure it is watching for."""

    from scripts import spawn_modal_jobs

    record = filed_record()
    monkeypatch.setattr(
        launcher,
        "compute_record",
        lambda: {
            **record,
            "workload": {
                **record["workload"],
                "cells": [
                    {**item, "seconds": 9_000.0} for item in record["workload"]["cells"]
                ],
            },
        },
    )
    with pytest.raises(SystemExit, match="REFUSED"):
        spawn_modal_jobs.gate_launch(PACKAGE, launcher, jobs)
