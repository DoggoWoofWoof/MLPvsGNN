"""What the M2 Modal launcher must get right before it is allowed to bill anything.

Three kinds of check, in the order they can go wrong:

*   Import-time correctness that only a real container discovers. M1A's launcher
    read a config it never mounted and every remote invocation crash-looped on
    FileNotFoundError before running any code; the AST walk below is the same
    guard, applied to this file.
*   Comparability. M2's whole result is a delta between fits this launcher
    submits and 19 M1A/M1B fits it reuses. If the two are trained or evaluated
    on different settings, every number in the selection report is a comparison
    between two different experiments -- so the hyperparameters and the panel
    size are held against M1A's own launcher, not restated here.
*   The gates. The declaration authorises the headline launch once every gate is
    true and pre-authorises passing them, never working around a failed one.
"""

from __future__ import annotations

import ast
import importlib
import json
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import modal_m1a_feature_screen as m1a  # noqa: E402
from scripts import modal_m2_qls_v2_freeze as launcher  # noqa: E402
from scripts.spawn_modal_jobs import PACKAGES  # noqa: E402

LAUNCHER_PATH = REPO_ROOT / "scripts" / "modal_m2_qls_v2_freeze.py"
DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
PACKAGE = "m2-qls-v2-freeze"

#: Every M1A dataset. musique_clean is deliberately absent: it is the dataset
#: M1A's own launcher excludes and M2's matrix declares, and the reason this
#: module has its own _jobs builder at all.
M1A_DATASETS = ("squad_clean", "2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp")


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(LAUNCHER_PATH.read_text(encoding="utf-8"), filename=str(LAUNCHER_PATH))


@pytest.fixture(scope="module")
def jobs() -> list[dict]:
    return launcher._jobs(list(launcher.ALL_DATASETS))


def _job(jobs: list[dict], dataset: str) -> dict:
    return next(job for job in jobs if job["dataset"] == dataset)


# --- import-time correctness the container would otherwise discover ----------


class _ModuleLevelReadTextVisitor(ast.NodeVisitor):
    """X.read_text() targets in top-level statements only, never inside a def."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        pass

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_text"
            and isinstance(node.func.value, ast.Name)
        ):
            self.names.add(node.func.value.id)
        self.generic_visit(node)


def _names_mounted_via_add_local_file(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_local_file"
            and node.args
        ):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "str":
            arg = arg.args[0] if arg.args else None
        if isinstance(arg, ast.Name):
            names.add(arg.id)
    return names


def test_every_path_read_at_module_import_time_is_mounted_into_the_image(tree) -> None:
    visitor = _ModuleLevelReadTextVisitor()
    visitor.visit(tree)
    read_at_import_time = visitor.names
    mounted = _names_mounted_via_add_local_file(tree)

    # Sanity check that the walk is finding real constants rather than passing
    # vacuously because the AST shapes stopped matching this file.
    assert "M2_CONFIG_PATH" in read_at_import_time
    assert "M1A_CONFIG_PATH" in read_at_import_time

    unmounted = read_at_import_time - mounted
    assert not unmounted, (
        f"{sorted(unmounted)} are read via .read_text() at module import time but never passed "
        "to add_local_file -- every remote container crashes with FileNotFoundError before "
        "running any real code"
    )


def test_the_reuse_manifest_the_runner_reads_at_run_time_is_mounted(tree) -> None:
    """Not read at import here, but read by the runner inside the container.

    run_m2_qls_v2_freeze.load_reuse_manifest refuses to fit anything without
    it, so an unmounted manifest is a job that dies after the container has
    started and been billed -- the same failure shape the mount test above
    exists for, one level removed.
    """
    assert "REUSE_MANIFEST_PATH" in _names_mounted_via_add_local_file(tree)


# --- scope: the six declared datasets, musique_clean included ----------------


def test_the_launcher_covers_every_declared_m2_dataset(declaration) -> None:
    assert set(launcher.ALL_DATASETS) == set(declaration["m2_selection_matrix"]["cells"])
    assert launcher.CONFIG["datasets"] is launcher.CONFIG["m2_selection_matrix"]["cells"], (
        "spawn_modal_jobs.py validates --datasets against CONFIG['datasets']; the alias must "
        "point at the matrix rather than being a second, driftable copy of it"
    )


def test_musique_clean_is_why_this_launcher_exists_at_all() -> None:
    """The one dataset M1A's launcher cannot submit. Stated as a test because
    it is the whole reason M2 does not delegate to M1A's ``_jobs``."""

    assert "musique_clean" in launcher.ALL_DATASETS
    assert "musique_clean" not in m1a.ALL_DATASETS


def test_an_undeclared_dataset_is_refused() -> None:
    with pytest.raises(ValueError, match="Undeclared M2 datasets"):
        launcher._jobs(["hotpotqa_clean", "not_a_dataset"])


# --- the validation split: derived, then cross-checked -----------------------


def test_the_validation_split_is_derived_from_the_artifact_and_agrees_with_m1a(jobs) -> None:
    for dataset in M1A_DATASETS:
        declared = launcher.M1A_CONFIG["datasets"][dataset]["validation_split_queries"]
        assert _job(jobs, dataset)["validation_split_queries"] == declared, dataset


def test_musique_clean_gets_its_split_from_the_same_code_path(jobs) -> None:
    """M1A declares no value for it, so there is nothing to cross-check --
    an absent check, not a special rule, and not a hand-entered number."""

    assert "validation_split_queries" not in launcher.M1A_CONFIG["datasets"]["musique_clean"]
    confirmation = json.loads(
        (REPO_ROOT / "outputs/sa_mlp_confirmation/musique_clean.json").read_text(encoding="utf-8")
    )
    assert _job(jobs, "musique_clean")["validation_split_queries"] == int(
        confirmation["data"]["splits"]["validation"]
    )


def test_a_drift_between_m1a_and_the_confirmation_artifact_stops_the_launch() -> None:
    """M2's comparators are M1A's own fits on that split. If the two sources
    ever stop agreeing about what "the validation split" is, the delta the
    whole phase reports is between two different panels."""

    doctored = {"data": {"splits": {"validation": 2999}}}
    with pytest.raises(ValueError, match="validation_split_queries"):
        launcher._validation_split_queries("2wiki_clean", doctored)


def test_an_incomplete_confirmation_artifact_is_refused(monkeypatch, tmp_path) -> None:
    incomplete = tmp_path / "outputs" / "sa_mlp_confirmation"
    incomplete.mkdir(parents=True)
    (incomplete / "2wiki_clean.json").write_text(
        json.dumps({"status": "SA_MLP_CONFIRMATION_DATASET_PARTIAL"}), encoding="utf-8"
    )
    monkeypatch.setattr(launcher, "HOST_REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="not a complete confirmation artifact"):
        launcher._jobs(["2wiki_clean"])


# --- comparability with the 19 reused fits -----------------------------------

#: Everything that decides what a fit IS. M2's new fits are compared against
#: M1A's, so a difference in any of these makes the delta meaningless.
SHARED_TRAINER_FIELDS = (
    "frozen_embedding_dim",
    "holdout_fraction",
    "per_seed_cap",
    "neighbour_scan_cap_per_seed",
    "edge_families",
    "a64_mainline_family",
    "semantic_rung",
    "seed",
    "epochs",
    "batch_size",
    "dropout",
    "temperature",
    "learning_rate",
    "weight_decay",
    "device",
)


def test_new_m2_fits_use_the_same_trainer_settings_as_the_m1a_fits_they_are_compared_against(
    jobs,
) -> None:
    dataset = "2wiki_clean"
    m2_args = launcher._runner_args(_job(jobs, dataset), stage="headline")
    m1a_args = m1a._runner_args(_job(m1a._jobs([dataset]), dataset), stage="headline")
    for field in SHARED_TRAINER_FIELDS:
        assert getattr(m2_args, field) == getattr(m1a_args, field), field


def test_the_headline_panel_is_the_split_the_reused_fits_were_scored_on(jobs) -> None:
    for dataset in M1A_DATASETS:
        m2_args = launcher._runner_args(_job(jobs, dataset), stage="headline")
        m1a_args = m1a._runner_args(_job(m1a._jobs([dataset]), dataset), stage="headline")
        assert m2_args.queries == m1a_args.queries, dataset


def test_the_headline_stage_runs_every_declared_regime_and_arm(jobs) -> None:
    """No narrowing at the launcher. Which arms are actually fitted is the
    reuse manifest's decision, made inside the runner."""

    args = launcher._runner_args(_job(jobs, "hotpotqa_clean"), stage="headline")
    assert args.regimes is None
    assert args.arms is None


def test_the_data_and_graph_roots_are_the_ones_m1a_used(jobs) -> None:
    for dataset in M1A_DATASETS:
        m2_job = _job(jobs, dataset)
        m1a_job = _job(m1a._jobs([dataset]), dataset)
        assert m2_job["data_remote"] == m1a_job["data_remote"], dataset
        assert m2_job["graph_root"] == m1a_job["graph_root"], dataset
        assert m2_job["fingerprint"] == m1a_job["fingerprint"], dataset


def test_the_seed_is_the_one_m2_declares(jobs, declaration) -> None:
    args = launcher._runner_args(_job(jobs, "metaqa"), stage="headline")
    assert args.seed == int(declaration["launch_authorization"]["smoke_spec"]["primary"]["seed"])
    assert args.seed == 0


# --- the smoke: read from the declaration, not restated ----------------------


def test_the_smoke_scope_comes_from_the_declaration(declaration, jobs) -> None:
    primary = declaration["launch_authorization"]["smoke_spec"]["primary"]
    assert launcher.SMOKE_DATASET == primary["dataset"]
    assert launcher.SMOKE_REGIMES == [primary["regime"]]
    assert launcher.SMOKE_ARMS == primary["arms"]
    assert launcher.SMOKE_QUERIES == primary["queries"]

    args = launcher._runner_args(_job(jobs, primary["dataset"]), stage="smoke")
    assert args.regimes == [primary["regime"]]
    assert args.arms == list(primary["arms"])
    assert args.queries == primary["queries"]


def test_the_secondary_smoke_is_the_r3_cell_that_exercises_a_real_node_role(
    declaration, jobs
) -> None:
    secondary = declaration["launch_authorization"]["smoke_spec"]["secondary_only_if_needed"]
    args = launcher._runner_args(_job(jobs, secondary["dataset"]), stage="secondary_smoke")
    assert args.regimes == ["R3"]
    assert args.arms == ["QLS-UNIVERSAL"]
    assert args.queries == secondary["queries"]
    assert secondary["verifies"] == ["node_role_nonzero_exactly_on_c3_minus_cq"]


def test_a_smoke_on_a_dataset_the_declaration_did_not_name_is_refused(jobs) -> None:
    other = next(d for d in launcher.ALL_DATASETS if d != launcher.SMOKE_DATASET)
    with pytest.raises(ValueError, match="this smoke is declared for"):
        launcher._runner_args(_job(jobs, other), stage="smoke")


def test_a_smoke_cell_that_is_not_in_the_matrix_is_refused(jobs) -> None:
    """The smoke_spec is prose in a YAML file; it can name a cell M2 never
    declared. Then the smoke would prove the pipeline works on something the
    real run never touches."""

    job = _job(jobs, launcher.SMOKE_DATASET)
    with pytest.raises(ValueError, match="not a declared M2 cell"):
        launcher._smoke_scope(job["dataset"], {**launcher.PRIMARY_SMOKE, "regime": "R9"})
    with pytest.raises(ValueError, match="smoke arms"):
        launcher._smoke_scope(job["dataset"], {**launcher.PRIMARY_SMOKE, "arms": ["BASE+GEOMETRY"]})


def test_an_unknown_stage_is_refused(jobs) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        launcher._runner_args(_job(jobs, "webqsp"), stage="rehearsal")


# --- the gates ---------------------------------------------------------------


def _gates(**overrides) -> dict:
    base = {name: True for name in launcher.CONFIG["launch_authorization"]["gates"]}
    return {"launch_authorization": {"gates": {**base, **overrides}}}


def test_unmet_gates_reads_every_gate_by_default() -> None:
    assert launcher.unmet_gates(_gates()) == []
    assert launcher.unmet_gates(_gates(engineering_smoke_passes=False)) == [
        "engineering_smoke_passes"
    ]
    assert launcher.unmet_gates(_gates(compute_within_ceiling=False, amendment_filed=False)) == [
        "amendment_filed",
        "compute_within_ceiling",
    ]


def test_a_gate_added_by_a_later_amendment_blocks_without_editing_this_file() -> None:
    """The default read is over the declaration's own gate keys, not a list
    restated in the launcher -- otherwise a new gate would be silently ignored
    by the very stage it was added to block."""

    declaration = _gates()
    declaration["launch_authorization"]["gates"]["some_future_gate"] = False
    assert launcher.unmet_gates(declaration) == ["some_future_gate"]


def test_unmet_gates_narrows_to_the_named_subset() -> None:
    declaration = _gates(engineering_smoke_passes=False)
    assert launcher.unmet_gates(declaration, only=launcher.PREREQUISITE_GATES) == []


def test_the_import_time_check_excludes_the_gates_that_are_earned_downstream() -> None:
    """Gating the smoke on engineering_smoke_passes would make the smoke unable
    to ever run first, and the other three are earned on or after it."""

    earned_later = {
        "engineering_smoke_passes",
        "feature_build_equivalence_proved",
        "compute_within_ceiling",
        "musique_clean_data_verified",
    }
    assert set(launcher.PREREQUISITE_GATES).isdisjoint(earned_later)
    assert set(launcher.PREREQUISITE_GATES) | earned_later == set(
        launcher.CONFIG["launch_authorization"]["gates"]
    ), "every gate is either an import-time prerequisite or earned downstream"


def test_the_headline_stage_rereads_the_gates_at_call_time(tree) -> None:
    """Not the import-time snapshot: the remaining gates are earned between
    this app being deployed and a headline job being spawned."""

    headline = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_m2_headline"
    )
    calls = {
        node.func.id
        for node in ast.walk(headline)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "unmet_gates" in calls
    assert any(isinstance(node, ast.Raise) for node in ast.walk(headline))
    read_paths = {
        node.func.value.id
        for node in ast.walk(headline)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
        and isinstance(node.func.value, ast.Name)
    }
    assert read_paths == {"M2_CONFIG_PATH"}


def test_the_headline_is_still_blocked_by_the_gates_steps_d_and_e_have_not_earned(
    declaration,
) -> None:
    """A live read of the committed declaration, so this test tracks the real
    state rather than a fixture: while these are false, spawning the headline
    stage raises instead of billing anything."""

    unmet = launcher.unmet_gates(declaration)
    for gate in ("feature_build_equivalence_proved", "compute_within_ceiling"):
        assert gate in unmet or declaration["launch_authorization"]["gates"][gate] is True
    if unmet:
        assert launcher.unmet_gates(declaration, only=launcher.PREREQUISITE_GATES) == [], (
            "the import-time prerequisites must already be true -- the module could not "
            "have been imported otherwise"
        )


# --- registration and isolation ----------------------------------------------


def test_the_package_is_registered_with_every_stage_this_launcher_serves() -> None:
    module_name, stages = PACKAGES[PACKAGE]
    assert module_name == "scripts.modal_m2_qls_v2_freeze"
    assert stages == launcher.STAGE_FUNCTIONS, (
        "the registry and the launcher must name the same functions; a rename in one "
        "leaves the other pointing at a name that no longer exists"
    )
    module = importlib.import_module(module_name)
    for function_name in stages.values():
        assert hasattr(module, function_name), function_name


def test_resume_granularity_is_declared_so_a_restart_can_be_sized() -> None:
    """spawn_modal_jobs._collapse_without_resumption costs an undeclared
    package as redoing everything on a restart. One dataset here is one call
    that writes its result once, at the end."""

    assert launcher.RESUME_GRANULARITY == "dataset"


def test_m2_writes_only_under_its_own_output_prefix(jobs) -> None:
    for stage in ("smoke", "headline"):
        dataset = launcher.SMOKE_DATASET if stage == "smoke" else "metaqa"
        args = launcher._runner_args(_job(jobs, dataset), stage=stage)
        for path in (args.output, args.artifact_root):
            text = str(path).replace("\\", "/")
            assert "/outputs/m2_qls_v2_freeze/" in text
            assert "m1a_feature_screen" not in text
            assert "m1b_targeted_resolution" not in text
            assert "sa_mlp_confirmation" not in text
        assert f"/{stage}/" in str(args.output).replace("\\", "/"), (
            "each stage writes to its own subtree; a smoke that overwrites the headline "
            "result would be cached back as the real one"
        )


def test_the_app_name_is_its_own(tree) -> None:
    assert launcher.APP_NAME == "message-passing-retrieval-m2-qls-v2-freeze"
    assert launcher.APP_NAME != m1a.APP_NAME


def test_the_infra_view_is_a_copy_so_it_cannot_mutate_m1as() -> None:
    """M1A's launcher hyperparameters are read live by scripts/m2_reuse_audit.py
    when it checks every one of the 19 reused fits. This module must not be able
    to change what that file says."""

    assert launcher.MODAL_CONFIG is not m1a.MODAL_CONFIG
    assert launcher.MODAL_CONFIG == m1a.MODAL_CONFIG


def test_the_source_commit_is_resolved_on_the_host(jobs) -> None:
    """The container has no .git, so the commit every fit records has to be
    resolved where the job is submitted and carried in."""

    commit = _job(jobs, "webqsp")["source_commit"]
    assert commit is not None and len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    args = launcher._runner_args(_job(jobs, "webqsp"), stage="headline")
    assert args.source_commit == commit


def test_the_declaration_names_this_launcher_and_its_registry_entry(declaration) -> None:
    prerequisite = " ".join(
        declaration["engineering_prerequisites_before_execution"]["modal_launcher"].split()
    )
    assert "scripts/modal_m2_qls_v2_freeze.py" in prerequisite
    assert "scripts/spawn_modal_jobs.py" in prerequisite
    assert PACKAGE in prerequisite
