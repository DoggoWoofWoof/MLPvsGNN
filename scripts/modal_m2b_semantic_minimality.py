"""Modal launcher for M2B: one container per dataset, every rung inside it.

The orchestration here is a scientific choice, not a convenience one.

M2B's tie-break is uncached inference p95. A latency is a property of the
machine that measured it, so comparing S4's p95 from one container against S2's
from another compares two containers as much as two rungs. Running every rung
of every regime of a dataset inside ONE container puts them on one clock, one
CPU allocation, one GPU, one thermal state -- and lets them share one loaded
structural feature store, which is also what makes "only the semantic rung
differs" true by construction rather than by care.

So: six spawned jobs, one per dataset, producing 28 new fits (14 S2 + 14 S4)
plus 14 reused S3 fits re-benchmarked for inference -- the declaration's 42
logical evaluations. Each fit is still checkpointed and result-addressable on
its own at ``<artifact_root>/<regime>/<rung>/``; sharing a container saves
startup, not identity.

M2B builds no features. Every cell master was persisted by M2's own build stage
and is read from M2's sealed output tree, which this launcher passes as
``--cell-master-root`` and never writes into. S3's weights come from the same
tree. M2B writes only under its own ``m2b_semantic_minimality`` prefix.

Two stages:
  - run_m2b_smoke:    smoke_before_fanout.cell -- 2wiki_clean / R3, S2 and S4
                      only, read from the declaration rather than restated
                      here so the two cannot drift. Its measured S4 fit time
                      is what replaces the 2.5x multiplier in the estimate;
                      until it exists, the ceiling is not final.
  - run_m2b_headline: every declared cell, every rung. Refuses while the
                      declaration has not been amended to authorise fitting,
                      and refuses while any gate the amendment files is false.

The image recipe is M2's, unchanged: run_m2b_semantic_minimality.py imports
run_m1a_feature_screen.py's feature internals and run_sa_mlp_confirmation.py's
trainer, so it needs the identical torch/torch-geometric/numba environment.

Submit through scripts/spawn_modal_jobs.py so the run is server-side and
survives client disconnection; a detached local run is not a registered
execution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import modal
import yaml

REMOTE_ROOT = "/root/message-passing-retrieval"
HOST_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REPO_ROOT = (
    HOST_REPO_ROOT
    if (HOST_REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml").is_file()
    else Path(REMOTE_ROOT)
)
M2B_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"
FORMULA_CONSTANT_SNAPSHOT_PATH = (
    RUNTIME_REPO_ROOT / "configs" / "formula_constants_at_build_commits.json"
)
M2_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
M1A_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m1a_feature_screen.yaml"
CONFIRMATION_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "sa_mlp_confirmation.yaml"

CONFIG = yaml.safe_load(M2B_CONFIG_PATH.read_text(encoding="utf-8"))
M2_CONFIG = yaml.safe_load(M2_CONFIG_PATH.read_text(encoding="utf-8"))
M1A_CONFIG = yaml.safe_load(M1A_CONFIG_PATH.read_text(encoding="utf-8"))
CONFIRMATION_CONFIG = yaml.safe_load(CONFIRMATION_CONFIG_PATH.read_text(encoding="utf-8"))

# spawn_modal_jobs.py's shared package protocol validates --datasets against
# module.CONFIG["datasets"]. M2B's schema names this evaluation_matrix.cells;
# the alias satisfies the launcher contract without changing the declaration.
CONFIG["datasets"] = CONFIG["evaluation_matrix"]["cells"]

# Copied, not aliased: M2B must not be able to change M1A's or M2's infra view.
MODAL_CONFIG = dict(M1A_CONFIG["modal"])
STORAGE_ROOT = MODAL_CONFIG["storage_root"]
APP_NAME = "message-passing-retrieval-m2b-semantic-minimality"
OUTPUT_PREFIX = "m2b_semantic_minimality"
M2_OUTPUT_PREFIX = "m2_qls_v2_freeze"
ALL_DATASETS = tuple(CONFIG["evaluation_matrix"]["cells"])

#: The status this file expects before it will fit anything. The declaration
#: currently stops for review; the amendment that authorises execution is what
#: moves it here. Checked at call time, not at import, so this module and its
#: tests exist before the amendment rather than after -- committing the
#: mechanism before the authorisation is this track's standing order.
AUTHORISED_STATUS = "DECLARED_LAUNCH_CONDITIONALLY_AUTHORISED"
RECONNAISSANCE_STATUS = "DECLARED_RECONNAISSANCE_COMPLETE_NO_FIT_AUTHORISED"
#: The phase ran, the committed rule returned SEMANTIC_PARETO_CONFLICT, and the
#: declaration stopped for review. Recognised so the module still imports --
#: the fetcher and the report read it -- but it authorises nothing: only
#: AUTHORISED_STATUS lets a stage run, so this refuses both, the same way the
#: reconnaissance status did before the amendment.
COMPLETE_STATUS = "M2B_SEMANTIC_PARETO_CONFLICT_STOPPED_FOR_REVIEW"

if CONFIG["status"] not in (AUTHORISED_STATUS, RECONNAISSANCE_STATUS, COMPLETE_STATUS):
    raise RuntimeError(
        f"Unexpected declaration status {CONFIG['status']!r} in "
        "configs/m2b_semantic_minimality.yaml -- re-check before launching"
    )

#: One dataset is one spawned call that writes its result once, at the end.
RESUME_GRANULARITY = "dataset"

#: Every rung of every regime runs in the dataset's own container. Named so the
#: cost ledger and the tests read the same number this file acts on.
CONTAINERS_PER_DATASET = 1

SMOKE_SPEC = CONFIG["smoke_before_fanout"]
SMOKE_DATASET, SMOKE_REGIME = (part.strip() for part in SMOKE_SPEC["cell"].split("/"))
#: The smoke fits the two new rungs. S3 in that cell is a completed M2 fit, so
#: re-running it would spend money to re-answer a filed question. Read from the
#: declaration rather than transcribed: amendment 2 filed this list, and a
#: second copy here is a second thing that could drift from it.
SMOKE_RUNGS = list(SMOKE_SPEC["rungs"])

app = modal.App(APP_NAME)
result_volume = modal.Volume.from_name(MODAL_CONFIG["result_volume"], create_if_missing=False)
image = (
    modal.Image.micromamba(python_version="3.11")
    .env(
        {
            "CONDA_OVERRIDE_CUDA": "12.1",
            "CUDA_HOME": "/opt/conda",
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT}/src",
        }
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "torch==2.2.1",
        "numpy<2.0",
        "scipy<1.14",
        "numba==0.60.0",
        "psutil==6.1.1",
        "pyyaml==6.0.2",
    )
    .pip_install(
        "torch-geometric==2.5.2",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html",
    )
    .add_local_dir(str(RUNTIME_REPO_ROOT / "src"), remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir(str(RUNTIME_REPO_ROOT / "scripts"), remote_path=f"{REMOTE_ROOT}/scripts")
    .add_local_file(
        str(M2B_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m2b_semantic_minimality.yaml"
    )
    # run_m2b_semantic_minimality.py imports run_m2_qls_v2_freeze.py, which reads
    # M2's declaration at import; and run_m1a_feature_screen.py resolves its own
    # REPO_ROOT by finding M1A's. Mounted so the imports succeed, not because
    # M2B reads either matrix.
    .add_local_file(str(M2_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m2_qls_v2_freeze.yaml")
    # The formula constants as they were at each store's build commit. A
    # container has no git repository, so it cannot read its own history; it
    # compares these recorded values against its own live imports instead. The
    # first M2B smoke died on `git show` here, after being billed for a GPU.
    .add_local_file(
        str(FORMULA_CONSTANT_SNAPSHOT_PATH),
        remote_path=f"{REMOTE_ROOT}/configs/formula_constants_at_build_commits.json",
    )
    .add_local_file(
        str(M1A_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m1a_feature_screen.yaml"
    )
    .add_local_file(
        str(CONFIRMATION_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/sa_mlp_confirmation.yaml"
    )
)


def _host_commit() -> str | None:
    """The commit this job is submitted from, resolved here: the container has no .git."""

    try:
        return subprocess.run(
            ["git", "-C", str(HOST_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        return None


def _validation_split_queries(dataset: str, confirmation: dict[str, Any]) -> int:
    """This dataset's validation split size, derived then cross-checked.

    The same derivation M2 used, for the same reason: M2B's fits have to score
    the identical panel M2's did or the S3 column it reuses is not comparable
    to the S2 and S4 columns beside it.
    """

    measured = int(confirmation["data"]["splits"]["validation"])
    declared = M1A_CONFIG["datasets"].get(dataset, {}).get("validation_split_queries")
    if declared is not None and int(declared) != measured:
        raise ValueError(
            f"{dataset}: configs/m1a_feature_screen.yaml declares validation_split_queries="
            f"{declared} but its confirmation artifact's validation split holds {measured} "
            "queries -- resolve this before launching; M2B's S3 column is M2's own fit on "
            "that split"
        )
    return measured


def _m2_output_root(dataset: str, fingerprint: str) -> PurePosixPath:
    """Where M2's completed headline run left this dataset's cells and fits.

    Read-only from here. The masters live at ``<root>/fits/<regime>/cell_features``
    and the S3 checkpoints at ``<root>/fits/<regime>/qls_universal/checkpoint.pt``,
    which is exactly the layout run_m2_qls_v2_freeze.py wrote.
    """

    return (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / M2_OUTPUT_PREFIX
        / dataset
        / fingerprint[:16]
        / MODAL_CONFIG["execution_label"]
        / "headline"
        / "fits"
    )


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    unknown = set(datasets) - set(ALL_DATASETS)
    if unknown:
        raise ValueError(f"Undeclared M2B datasets: {sorted(unknown)}")
    commit = _host_commit()
    jobs = []
    for dataset in datasets:
        confirmation_settings = CONFIRMATION_CONFIG["datasets"][dataset]
        confirmation_path = HOST_REPO_ROOT / f"outputs/sa_mlp_confirmation/{dataset}.json"
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
        if confirmation.get("status") != "SA_MLP_CONFIRMATION_DATASET_COMPLETE":
            raise ValueError(f"{confirmation_path} is not a complete confirmation artifact")
        fingerprint = confirmation["data_fingerprint_sha256"]
        jobs.append(
            {
                "dataset": dataset,
                "fingerprint": fingerprint,
                "data_remote": confirmation["config"]["data"],
                "baseline": confirmation["baseline"],
                "expected_queries": int(confirmation["config"]["expected_queries"]),
                "validation_split_queries": _validation_split_queries(dataset, confirmation),
                "candidate_contract_compatibility": confirmation_settings.get(
                    "candidate_contract_compatibility"
                ),
                "m2_fits_remote": str(_m2_output_root(dataset, fingerprint)),
                "regimes": list(CONFIG["evaluation_matrix"]["cells"][dataset]),
                "source_commit": commit,
            }
        )
    return jobs


def _active_modal_profile() -> str | None:
    """The profile this client will actually submit under, or None if unreadable.

    None rather than a guess, and the caller refuses on it: the wrong answer
    here sends a job to the wrong workspace, and a Modal volume lives in
    exactly one workspace.
    """

    from_env = os.environ.get("MODAL_PROFILE")
    if from_env:
        return from_env
    try:
        from modal.config import _profile
    except ImportError:  # pragma: no cover -- modal always ships this
        return None
    return _profile or None


def execution_placement() -> dict[str, str]:
    """Which workspace each dataset runs on.

    M2B's own declaration if it states one; otherwise M2's, because M2B reads
    M2's persisted cell masters and those live on whichever volume M2 wrote
    them to. Inheriting is the correct default here and not a shortcut: a
    dataset whose masters are on one workspace cannot be fit from another.
    """

    placement = CONFIG.get("launch_authorization", {}).get("execution_placement")
    if placement:
        return dict(placement)
    inherited = M2_CONFIG["launch_authorization"].get("execution_placement")
    if not inherited:
        raise SystemExit(
            "neither configs/m2b_semantic_minimality.yaml nor configs/m2_qls_v2_freeze.yaml "
            "says which workspace each dataset runs on; a dataset with no declared "
            "workspace cannot be submitted to one"
        )
    return dict(inherited)


def check_execution_placement(datasets: list[str]) -> dict[str, Any]:
    """Refuse to submit a dataset under a workspace its cell masters are not on.

    scripts/spawn_modal_jobs.py calls this by name if a launcher defines it, so
    the check runs before the app is deployed and before anything is spawned.
    M2B is stricter than M2 was in one respect: M2 could have rebuilt a missing
    feature store on the wrong workspace and merely wasted money. M2B cannot
    build one at all, so a misplaced job fails after the container has started
    and been billed, having proved nothing.
    """

    placement = execution_placement()
    profile = _active_modal_profile()
    if not profile:
        raise SystemExit(
            "cannot determine the active Modal profile -- set MODAL_PROFILE explicitly "
            "rather than letting the placement go unchecked"
        )
    undeclared = sorted(name for name in datasets if name not in placement)
    if undeclared:
        raise SystemExit(
            f"execution_placement does not say where {undeclared} run; a dataset with no "
            "declared workspace cannot be submitted to one"
        )
    misplaced = {name: placement[name] for name in datasets if placement[name] != profile}
    if misplaced:
        raise SystemExit(
            f"REFUSED: submitting under profile {profile!r}, but {misplaced} are placed "
            "elsewhere. A Modal volume lives in one workspace, so this would open a "
            "different volume than the one holding M2's persisted cell masters -- and "
            "M2B cannot build a master, so the job would fail after billing. Run these "
            f"under their own profile, or submit only the datasets placed on {profile!r}."
        )
    return {"profile": profile, "datasets": sorted(datasets), "placement_source": (
        "m2b" if CONFIG.get("launch_authorization", {}).get("execution_placement") else "m2"
    )}


def _smoke_scope(dataset: str) -> tuple[list[str], list[str]]:
    """The declared smoke's regime and rungs, checked against the matrix.

    The smoke runs the cell at its FULL declared panel, not a 100-query
    diagnostic. Its job is to produce the measured S4 fit time that replaces
    the 2.5x multiplier in the compute estimate, and a fit on a hundredth of
    the panel would not measure that. This is why it costs real money and is
    gated like the fan-out: the declaration says outright that the smoke is a
    fit.
    """

    if dataset != SMOKE_DATASET:
        raise ValueError(
            f"the smoke is declared for {SMOKE_DATASET!r}; got {dataset!r} -- see "
            "configs/m2b_semantic_minimality.yaml#smoke_before_fanout.cell"
        )
    declared = CONFIG["evaluation_matrix"]["cells"][dataset]
    if SMOKE_REGIME not in declared:
        raise ValueError(f"smoke regime {SMOKE_REGIME!r} is not a declared M2B cell for {dataset!r}")
    return [SMOKE_REGIME], list(SMOKE_RUNGS)


#: launcher stage -> (the smoke_spec it runs at or None for the declared panel,
#: the subtree it writes under). The two subtrees are separate so a smoke
#: result and a headline result are two artifacts held against each other
#: rather than one overwriting the other.
STAGE_PLAN: dict[str, tuple[bool, str]] = {
    "smoke": (True, "smoke"),
    "headline": (False, "headline"),
}


def _runner_args(job: dict[str, Any], *, stage: str) -> argparse.Namespace:
    from scripts.run_m0b_regime_map import MAINLINE_FAMILY

    dataset = job["dataset"]
    plan = STAGE_PLAN.get(stage)
    if plan is None:
        raise ValueError(f"unknown stage {stage!r}; expected one of {tuple(STAGE_PLAN)}")
    is_smoke, subtree = plan
    if is_smoke:
        regimes, rungs = _smoke_scope(dataset)
    else:
        regimes, rungs = list(job["regimes"]), None
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / OUTPUT_PREFIX
        / dataset
        / job["fingerprint"][:16]
        / MODAL_CONFIG["execution_label"]
        / subtree
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=dataset,
        data_fingerprint_sha256=job["fingerprint"],
        expected_queries=job["expected_queries"],
        frozen_embedding_dim=M1A_CONFIG["base"]["semantic_rung"]["frozen_embedding_dim"],
        baseline=job["baseline"],
        candidate_contract_compatibility=job["candidate_contract_compatibility"],
        # The same panel M2 scored. M2B's S3 column IS M2's fit, so a different
        # panel here would leave the reused column incomparable to the new ones.
        queries=int(job["validation_split_queries"]),
        holdout_fraction=0.2,
        per_seed_cap=16,
        neighbour_scan_cap_per_seed=4096,
        a64_mainline_family=MAINLINE_FAMILY,
        regimes=regimes,
        rungs=rungs,
        seed=0,
        epochs=3,
        batch_size=16,
        dropout=0.2,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device="cuda",
        latency_queries=200,
        latency_repeats=3,
        latency_warmup=8,
        source_commit=job.get("source_commit"),
        # Read M2's sealed tree; write only under M2B's own prefix.
        cell_master_root=Path(job["m2_fits_remote"]),
        m2_fits_root=Path(job["m2_fits_remote"]),
        artifact_root=Path(output_root) / "fits",
        output=Path(output_root) / "semantic_minimality.json",
    )


def _run(job: dict[str, Any], *, stage: str) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    # M2's masters were written by a different container in a different phase,
    # so this reload is load-bearing: without it the volume view is whatever it
    # was when this container started.
    result_volume.reload()
    from scripts.run_m2b_semantic_minimality import run

    args = _runner_args(job, stage=stage)
    result = run(args)
    result_volume.commit()
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "stage": stage,
        "regimes": sorted(result["cells"]),
        "rungs": result["rungs"],
        "fits_in_this_container": sum(len(cell["rungs"]) for cell in result["cells"].values()),
        "output_remote": str(args.output),
        "artifact_root_remote": str(args.artifact_root),
        "cell_master_root_remote": str(args.cell_master_root),
    }


def _require_authorisation(stage: str) -> None:
    """Re-read the declaration and refuse while it has not authorised fitting.

    Read at call time rather than trusting the import-time snapshot: the
    amendment that authorises execution is filed between this app being
    deployed and a job being spawned against it. Both stages are gated, because
    the smoke is a fit -- the declaration says so itself.
    """

    declaration = yaml.safe_load(M2B_CONFIG_PATH.read_text(encoding="utf-8"))
    status = declaration.get("status")
    if status != AUTHORISED_STATUS:
        raise RuntimeError(
            f"configs/m2b_semantic_minimality.yaml reports status {status!r}. It authorises "
            "no fit -- and smoke_before_fanout.authorisation says outright that the smoke is "
            f"a fit and needs the further dated amendment. Refusing to run stage {stage!r}."
        )
    gates = declaration.get("launch_authorization", {}).get("gates")
    if gates is None:
        raise RuntimeError(
            "the declaration reports an authorised status but files no "
            "launch_authorization.gates; an authorisation with nothing to check is not one"
        )
    if stage == "smoke":
        # The smoke's whole job is to earn the gates measured on it, so gating
        # it on those would make it unable to ever run first.
        unmet = sorted(
            name for name in ("amendment_filed", "selection_rule_frozen",
                              "feature_store_reuse_proved", "s3_behaviour_reuse_proved",
                              "semantic_formulas_frozen", "instrumentation_tests_pass")
            if name in gates and not gates.get(name)
        )
    else:
        unmet = sorted(name for name in gates if not gates.get(name))
    if unmet:
        raise RuntimeError(
            f"configs/m2b_semantic_minimality.yaml#launch_authorization.gates reports {unmet} "
            f"as not true. Stage {stage!r} is authorised once they are -- earn the gate, do "
            "not bypass it."
        )


@app.function(
    image=image,
    gpu=MODAL_CONFIG["gpu"],
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_m2b_smoke(job: dict[str, Any]) -> dict[str, Any]:
    _require_authorisation("smoke")
    return _run(job, stage="smoke")


@app.function(
    image=image,
    gpu=MODAL_CONFIG["gpu"],
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_m2b_headline(job: dict[str, Any]) -> dict[str, Any]:
    _require_authorisation("headline")
    return _run(job, stage="headline")


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative = remote_path.removeprefix(f"{STORAGE_ROOT}/")
    with local_path.open("wb") as stream:
        for chunk in result_volume.read_file(relative):
            stream.write(chunk)


#: stage name -> the Modal function that serves it. scripts/spawn_modal_jobs.py
#: registers the same mapping; the test suite holds the two to each other so a
#: renamed function cannot leave the registry pointing at a name that is gone.
STAGE_FUNCTIONS = {
    "smoke": "run_m2b_smoke",
    "headline": "run_m2b_headline",
}


@app.local_entrypoint()
def main(datasets: str = SMOKE_DATASET, stage: str = "smoke") -> None:
    # For manual debugging only. The real run goes through
    # scripts/spawn_modal_jobs.py so the call survives client disconnection.
    if stage not in STAGE_FUNCTIONS:
        raise ValueError(f"stage must be one of {tuple(STAGE_FUNCTIONS)}; got {stage!r}")
    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    jobs = _jobs(requested)
    function = globals()[STAGE_FUNCTIONS[stage]]
    results = list(function.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} M2B job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / OUTPUT_PREFIX / stage
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}.json")
    print(json.dumps(results, indent=2))
