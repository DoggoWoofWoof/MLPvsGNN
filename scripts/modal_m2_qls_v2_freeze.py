"""Modal launcher for M2's engineering smoke and, once every gate is true, the new seed-0 fits.

GPU (A10G), same image recipe as scripts/modal_m1a_feature_screen.py, because
scripts/run_m2_qls_v2_freeze.py imports run_m1a_feature_screen.py's feature-build
internals and run_sa_mlp_confirmation.py's trainer unmodified and so needs the
identical torch/torch-geometric/numba environment.

A new module rather than an edit to either existing launcher, for two reasons
that are not stylistic:

1. modal_m1a_feature_screen.py's ``_jobs`` excludes musique_clean by
   construction (its ALL_DATASETS filters it out), and M2's matrix declares it.
2. That file's hardcoded ``_runner_args`` hyperparameters are one of the fields
   scripts/m2_reuse_audit.py reads *live* when checking every one of the 19
   reused fits. It has to keep saying what it said when those fits ran.

Three Modal functions, mirroring the smoke/headline shape every launcher in this
track uses:
  - run_m2_smoke:           configs/m2_qls_v2_freeze.yaml#launch_authorization.
                            smoke_spec.primary, read from the declaration at
                            call time rather than restated here so the two
                            cannot drift. Pipeline validation only; no smoke
                            result feeds any reported metric.
  - run_m2_secondary_smoke: smoke_spec.secondary_only_if_needed -- the R3 cell
                            that exercises a genuinely nonzero NODE_ROLE column,
                            which the R2 primary cannot by construction.
  - run_m2_headline:        the declared new seed-0 fits for the requested
                            dataset(s). Refuses while ANY of the nine launch
                            gates is false, which is the declaration's own rule:
                            it pre-authorises passing the gates, never working
                            around a failed one.

``validation_split_queries`` is derived from each dataset's own completed
confirmation artifact rather than transcribed. That is where M1A's five declared
values came from -- all five still agree exactly, and _validation_split_queries
below refuses the launch if they ever stop agreeing -- and it is what lets
musique_clean, which M1A never declared a value for, be an ordinary row here
instead of a hand-entered special case.

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
    if (HOST_REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").is_file()
    else Path(REMOTE_ROOT)
)
M2_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
M1A_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m1a_feature_screen.yaml"
CONFIRMATION_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "sa_mlp_confirmation.yaml"
REUSE_MANIFEST_PATH = RUNTIME_REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "reuse_audit.json"

CONFIG = yaml.safe_load(M2_CONFIG_PATH.read_text(encoding="utf-8"))
# spawn_modal_jobs.py's shared package protocol validates --datasets against
# module.CONFIG["datasets"]. M2's declaration schema names this
# m2_selection_matrix.cells (regimes and arms are declared per dataset, not as a
# flat list), exactly as M1B's names it cells -- the alias satisfies the shared
# launcher contract without changing the declaration's own schema.
CONFIG["datasets"] = CONFIG["m2_selection_matrix"]["cells"]
M1A_CONFIG = yaml.safe_load(M1A_CONFIG_PATH.read_text(encoding="utf-8"))
CONFIRMATION_CONFIG = yaml.safe_load(CONFIRMATION_CONFIG_PATH.read_text(encoding="utf-8"))
# Copied, not aliased: nothing this module does to its infra view may leak back
# into M1A's own launcher, whose hyperparameters the reuse audit reads live.
MODAL_CONFIG = dict(M1A_CONFIG["modal"])
STORAGE_ROOT = MODAL_CONFIG["storage_root"]
APP_NAME = "message-passing-retrieval-m2-qls-v2-freeze"
OUTPUT_PREFIX = "m2_qls_v2_freeze"
ALL_DATASETS = tuple(CONFIG["m2_selection_matrix"]["cells"])

SMOKE_SPEC = CONFIG["launch_authorization"]["smoke_spec"]
PRIMARY_SMOKE = SMOKE_SPEC["primary"]
SECONDARY_SMOKE = SMOKE_SPEC["secondary_only_if_needed"]
SMOKE_DATASET = PRIMARY_SMOKE["dataset"]
SMOKE_REGIMES = [PRIMARY_SMOKE["regime"]]
SMOKE_ARMS = list(PRIMARY_SMOKE["arms"])
SMOKE_QUERIES = int(PRIMARY_SMOKE["queries"])

if CONFIG["status"] != "DECLARED_LAUNCH_CONDITIONALLY_AUTHORISED":
    raise RuntimeError(
        f"Unexpected declaration status {CONFIG['status']!r} -- re-check before launching"
    )

#: Gates that must already be true before this launcher runs EITHER stage.
#: engineering_smoke_passes is deliberately absent: the smoke's whole job is to
#: earn it, and gating it at import would make the smoke unable to ever run
#: first. feature_build_equivalence_proved, compute_within_ceiling and
#: musique_clean_data_verified are absent for their own reasons in the same
#: order -- the first two are measured on the smoke, and the third concerns a
#: dataset the smoke does not touch. All nine are checked for the headline.
PREREQUISITE_GATES = (
    "amendment_filed",
    "selection_rule_frozen",
    "reuse_audit_passes",
    "instrumentation_tests_pass",
    "engineering_tests_pass",
)


def unmet_gates(declaration: dict[str, Any], *, only: tuple[str, ...] | None = None) -> list[str]:
    """Gate names that are not true, sorted. ``only`` narrows to a named subset.

    With ``only`` omitted this reads whatever gates the declaration carries
    rather than a list restated here, so a gate added by a later amendment
    blocks the headline stage without this file needing an edit to notice it.
    """

    gates = declaration["launch_authorization"]["gates"]
    names = gates if only is None else only
    return sorted(name for name in names if not gates.get(name))


_unmet = unmet_gates(CONFIG, only=PREREQUISITE_GATES)
if _unmet:
    raise RuntimeError(
        f"configs/m2_qls_v2_freeze.yaml#launch_authorization.gates reports {_unmet} as not yet "
        "true -- these must all be true before this launcher is used for either stage"
    )

#: One dataset is one spawned call that writes its result once, at the end --
#: the same convention every launcher in this track uses.
RESUME_GRANULARITY = "dataset"

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
    .add_local_file(str(M2_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m2_qls_v2_freeze.yaml")
    # run_m1a_feature_screen.py resolves its own REPO_ROOT by finding this file,
    # and run_m2_qls_v2_freeze.py imports that module -- mounted so the import
    # succeeds, not because M2 reads M1A's cells.
    .add_local_file(
        str(M1A_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m1a_feature_screen.yaml"
    )
    .add_local_file(
        str(CONFIRMATION_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/sa_mlp_confirmation.yaml"
    )
    # run_m2_qls_v2_freeze.load_reuse_manifest reads this at run time and
    # refuses to fit anything without it -- an unmounted manifest is a job that
    # dies after the container has started and been billed.
    .add_local_file(
        str(REUSE_MANIFEST_PATH),
        remote_path=f"{REMOTE_ROOT}/outputs/{OUTPUT_PREFIX}/reuse_audit.json",
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
    """This dataset's validation split size, derived from the artifact then cross-checked.

    Derived from the confirmation artifact because that is the split M1A's own
    declared numbers were read off, and cross-checked against M1A's declaration
    wherever it declares one, so a drift between the two stops the launch rather
    than silently changing what "the validation split" means -- M2's whole
    comparison is against M1A fits scored on that split. musique_clean has no
    M1A-declared value (it was out of scope for M1A) so it has nothing to
    cross-check against; that is an absent check, not a special rule.
    """

    measured = int(confirmation["data"]["splits"]["validation"])
    declared = M1A_CONFIG["datasets"].get(dataset, {}).get("validation_split_queries")
    if declared is not None and int(declared) != measured:
        raise ValueError(
            f"{dataset}: configs/m1a_feature_screen.yaml declares validation_split_queries="
            f"{declared} but its confirmation artifact's validation split holds {measured} "
            "queries -- resolve this before launching; M2's comparators are M1A's own fits "
            "on that split"
        )
    return measured


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    unknown = set(datasets) - set(ALL_DATASETS)
    if unknown:
        raise ValueError(f"Undeclared M2 datasets: {sorted(unknown)}")
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
                "graph_root": str(
                    PurePosixPath(STORAGE_ROOT)
                    / MODAL_CONFIG["edge_provenance_root"]
                    / dataset
                    / fingerprint[:16]
                ),
                "source_commit": commit,
            }
        )
    return jobs


def _smoke_scope(dataset: str, spec: dict[str, Any]) -> tuple[list[str], list[str], int]:
    """Regimes, arms and panel size for one declared smoke, checked against the matrix."""

    if dataset != spec["dataset"]:
        raise ValueError(
            f"this smoke is declared for {spec['dataset']!r}; got {dataset!r} -- see "
            "configs/m2_qls_v2_freeze.yaml#launch_authorization.smoke_spec"
        )
    declared = CONFIG["m2_selection_matrix"]["cells"][dataset]
    regime = spec["regime"]
    if regime not in declared:
        raise ValueError(f"smoke regime {regime!r} is not a declared M2 cell for {dataset!r}")
    arms = list(spec["arms"])
    missing = [arm for arm in arms if arm not in declared[regime]]
    if missing:
        raise ValueError(f"smoke arms {missing} are not declared for {dataset}/{regime}")
    return [regime], arms, int(spec["queries"])


def _runner_args(job: dict[str, Any], *, stage: str) -> argparse.Namespace:
    from scripts.run_m0b_regime_map import MAINLINE_FAMILY

    dataset = job["dataset"]
    if stage == "smoke":
        regimes, arms, queries = _smoke_scope(dataset, PRIMARY_SMOKE)
    elif stage == "secondary_smoke":
        regimes, arms, queries = _smoke_scope(dataset, SECONDARY_SMOKE)
    elif stage == "headline":
        regimes, arms = None, None
        queries = int(job["validation_split_queries"])
    else:
        raise ValueError(f"unknown stage {stage!r}; expected one of {tuple(STAGE_FUNCTIONS)}")
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / OUTPUT_PREFIX
        / dataset
        / job["fingerprint"][:16]
        / MODAL_CONFIG["execution_label"]
        / stage
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=dataset,
        data_fingerprint_sha256=job["fingerprint"],
        expected_queries=job["expected_queries"],
        frozen_embedding_dim=M1A_CONFIG["base"]["semantic_rung"]["frozen_embedding_dim"],
        baseline=job["baseline"],
        candidate_contract_compatibility=job["candidate_contract_compatibility"],
        queries=queries,
        holdout_fraction=0.2,
        per_seed_cap=16,
        neighbour_scan_cap_per_seed=4096,
        edge_provenance_root=Path(job["graph_root"]),
        edge_families=[MAINLINE_FAMILY],
        a64_mainline_family=MAINLINE_FAMILY,
        regimes=regimes,
        arms=arms,
        semantic_rung="S3",
        seed=0,
        epochs=3,
        batch_size=16,
        dropout=0.2,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device="cuda",
        source_commit=job.get("source_commit"),
        artifact_root=Path(output_root) / "fits",
        output=Path(output_root) / "qls_v2_freeze.json",
    )


def _run(job: dict[str, Any], *, stage: str) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    result_volume.reload()
    from scripts.run_m2_qls_v2_freeze import run

    args = _runner_args(job, stage=stage)
    result = run(args)
    result_volume.commit()
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "stage": stage,
        "output_remote": str(args.output),
        "artifact_root_remote": str(args.artifact_root),
    }


@app.function(
    image=image,
    gpu=MODAL_CONFIG["gpu"],
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_m2_smoke(job: dict[str, Any]) -> dict[str, Any]:
    return _run(job, stage="smoke")


@app.function(
    image=image,
    gpu=MODAL_CONFIG["gpu"],
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_m2_secondary_smoke(job: dict[str, Any]) -> dict[str, Any]:
    return _run(job, stage="secondary_smoke")


@app.function(
    image=image,
    gpu=MODAL_CONFIG["gpu"],
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_m2_headline(job: dict[str, Any]) -> dict[str, Any]:
    # Re-read at call time rather than trusting the import-time snapshot: the
    # remaining gates are earned between this app being deployed and a headline
    # job being spawned.
    declaration = yaml.safe_load(M2_CONFIG_PATH.read_text(encoding="utf-8"))
    unmet = unmet_gates(declaration)
    if unmet:
        raise RuntimeError(
            f"configs/m2_qls_v2_freeze.yaml#launch_authorization.gates reports {unmet} as not "
            "true. The declaration authorises launching once EVERY gate is true and pre-authorises "
            "passing them, never working around a failed one -- earn the gate, do not bypass it."
        )
    return _run(job, stage="headline")


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative = remote_path.removeprefix(f"{STORAGE_ROOT}/")
    with local_path.open("wb") as stream:
        for chunk in result_volume.read_file(relative):
            stream.write(chunk)


#: stage name -> the Modal function that serves it. scripts/spawn_modal_jobs.py
#: registers the same mapping; tests/test_modal_m2_qls_v2_freeze.py holds the
#: two to each other so a renamed function cannot leave the registry pointing
#: at a name that no longer exists.
STAGE_FUNCTIONS = {
    "smoke": "run_m2_smoke",
    "secondary_smoke": "run_m2_secondary_smoke",
    "headline": "run_m2_headline",
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
        raise RuntimeError(f"{len(failures)} M2 job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / OUTPUT_PREFIX / stage
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}.json")
    print(json.dumps(results, indent=2))
