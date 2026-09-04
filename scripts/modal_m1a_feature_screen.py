"""Modal launcher for M1A step 4 (smoke) and step 5 (headline, authorised 2026-09-04).

GPU (A10G), since M1A trains -- unlike M0B/M0C's zero-training CPU launchers.
Reuses scripts/run_sa_mlp_confirmation.py's own frozen baseline artifacts
(outputs/sa_mlp_confirmation/*.json) for the data path, fingerprint, and
candidate-contract baseline, exactly as scripts/modal_m0c_bounded_r3.py does;
reuses the same edge-provenance graph root M0C already proved reachable on
the volume for R3's A64 construction. Writes to its own output prefix
(outputs/m1a_feature_screen/...) and touches no M0B/M0C/confirmation path.
Submit through scripts/spawn_modal_jobs.py so the run is server-side; a
detached local run is not a registered execution.

Two Modal functions, mirroring modal_m0b_regime_map.py / modal_m0c_bounded_
r3.py's smoke/headline shape:
  - stage="smoke"    -> run_feature_screen_smoke: the declared step-4 scope
                         only -- configs/m1a_feature_screen.yaml#compute.
                         recommended_step_4_smoke_scope, i.e. hotpotqa_clean
                         x R3 x {BASE, BASE+NODE_ROLE} -- at this track's own
                         established 100-query diagnostic panel size
                         (modal.smoke_queries in the same file). Authorised
                         by the 2026-09-04 step-4 amendment in that file.
  - stage="headline" -> run_feature_screen_headline: every declared cell/arm
                         for the requested dataset(s), at the full canonical
                         validation split (datasets.*.validation_split_
                         queries -- sampling.subsampling: none). This is
                         step 5, the real one-seed M1A screen -- authorised
                         by the 2026-09-04 step-5 amendment in that file
                         (this_file_authorises extended,
                         the_real_m1a_screen_launch_step_5 removed from
                         does_not_authorise), filed only after step 4's
                         smoke actually passed
                         (PASS_AFTER_ACCOUNTING_CORRECTION) and a real,
                         dataset-weighted cost projection was filed with a
                         $20.00 ceiling. Covers exactly the 44 predeclared
                         arms (arm_count_summary.predeclared_arms) -- the
                         reserved up-to-3 conditional interaction arms
                         (arm_count_summary.conditional_interaction_arms_
                         reserved) are a separate, later, not-yet-authorised
                         decision, not wired into run_m1a_feature_screen.py
                         at all yet.

Unlike M0C's smoke (same cells, fewer queries), M1A's smoke additionally
narrows to the declared regime/arm pair -- see recommended_step_4_smoke_
scope's own reasoning: one cell covers both the dominant feature-build cost
driver and the only genuinely new code path (R3-bounded context builder,
per-family column mask, is_structurally_admitted) at once.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import modal
import yaml

REMOTE_ROOT = "/root/message-passing-retrieval"
HOST_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REPO_ROOT = (
    HOST_REPO_ROOT
    if (HOST_REPO_ROOT / "configs" / "m1a_feature_screen.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m1a_feature_screen.yaml"
CONFIRMATION_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "sa_mlp_confirmation.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
# expected_queries and candidate_contract_compatibility are not declared in
# m1a_feature_screen.yaml (it only carries validation_split_queries, a
# different, smaller number) -- both come from the frozen confirmation
# phase's own declaration instead, the same source
# scripts/modal_sa_mlp_confirmation.py reads them from.
CONFIRMATION_CONFIG = yaml.safe_load(CONFIRMATION_CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]
STORAGE_ROOT = MODAL_CONFIG["storage_root"]
APP_NAME = MODAL_CONFIG["app"]
ALL_DATASETS = tuple(name for name in CONFIG["datasets"] if name != "musique_clean")

SMOKE_REGIMES = ["R3"]
SMOKE_ARMS = ["BASE", "BASE+NODE_ROLE"]
SMOKE_DATASET = "hotpotqa_clean"

if CONFIG["status"] != "DECLARED_NOT_LAUNCHED":
    raise RuntimeError(f"Unexpected declaration status {CONFIG['status']!r} -- re-check before launching")
# Inverted 2026-09-04 (step-5 amendment): this launcher's headline stage now
# assumes step 5 IS authorised (run_feature_screen_headline calls _run for
# real, no guard raise inside it) -- if a future edit reintroduces
# the_real_m1a_screen_launch_step_5 into does_not_authorise (a revert, a bad
# merge) without this file being updated to match, fail loudly at import
# time rather than let an actually-unauthorised launch proceed silently.
if "the_real_m1a_screen_launch_step_5" in CONFIG["does_not_authorise"]:
    raise RuntimeError(
        "configs/m1a_feature_screen.yaml lists step 5 under does_not_authorise again, but "
        "this launcher's run_feature_screen_headline is wired to actually run it -- re-sync "
        "this file with the declaration before trusting either one."
    )
if not CONFIG["compute"].get("ceiling_filed"):
    raise RuntimeError(
        "configs/m1a_feature_screen.yaml does not report a filed compute ceiling "
        "(compute.ceiling_filed) -- the step-5 amendment that authorised headline launches "
        "also files compute.ceiling_usd; re-check before launching without one."
    )

# One dataset is one spawned call that writes its result once, at the end,
# same convention as M0C's own RESUME_GRANULARITY.
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
    .add_local_file(str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m1a_feature_screen.yaml")
    .add_local_file(
        str(CONFIRMATION_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/sa_mlp_confirmation.yaml"
    )
)


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    unknown = set(datasets) - set(ALL_DATASETS)
    if unknown:
        raise ValueError(f"Undeclared or out-of-scope M1A datasets: {sorted(unknown)}")
    jobs = []
    for dataset in datasets:
        settings = CONFIG["datasets"][dataset]
        confirmation_settings = CONFIRMATION_CONFIG["datasets"][dataset]
        confirmation_path = HOST_REPO_ROOT / f"outputs/sa_mlp_confirmation/{dataset}.json"
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
        if confirmation.get("status") != "SA_MLP_CONFIRMATION_DATASET_COMPLETE":
            raise ValueError(f"{confirmation_path} is not a complete confirmation artifact")
        fingerprint = confirmation["data_fingerprint_sha256"]
        jobs.append(
            {
                "dataset": dataset,
                "settings": settings,
                "fingerprint": fingerprint,
                "data_remote": confirmation["config"]["data"],
                "baseline": confirmation["baseline"],
                "expected_queries": int(confirmation["config"]["expected_queries"]),
                "candidate_contract_compatibility": confirmation_settings.get(
                    "candidate_contract_compatibility"
                ),
                "graph_root": str(
                    PurePosixPath(STORAGE_ROOT)
                    / MODAL_CONFIG["edge_provenance_root"]
                    / dataset
                    / fingerprint[:16]
                ),
            }
        )
    return jobs


def _runner_args(job: dict[str, Any], *, stage: str) -> argparse.Namespace:
    from scripts.run_m0b_regime_map import MAINLINE_FAMILY

    if stage == "smoke":
        regimes = list(SMOKE_REGIMES)
        arms = list(SMOKE_ARMS)
        queries = int(MODAL_CONFIG["smoke_queries"])
    else:
        regimes = None
        arms = None
        queries = int(job["settings"]["validation_split_queries"])
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "m1a_feature_screen"
        / job["dataset"]
        / job["fingerprint"][:16]
        / MODAL_CONFIG["execution_label"]
        / stage
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        data_fingerprint_sha256=job["fingerprint"],
        expected_queries=job["expected_queries"],
        frozen_embedding_dim=CONFIG["base"]["semantic_rung"]["frozen_embedding_dim"],
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
        output=Path(output_root) / "feature_screen.json",
    )


def _run(job: dict[str, Any], *, stage: str) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    result_volume.reload()
    from scripts.run_m1a_feature_screen import run

    args = _runner_args(job, stage=stage)
    result = run(args)
    result_volume.commit()
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "stage": stage,
        "output_remote": str(args.output),
    }


@app.function(
    image=image,
    gpu=MODAL_CONFIG["gpu"],
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_feature_screen_smoke(job: dict[str, Any]) -> dict[str, Any]:
    if job["dataset"] != SMOKE_DATASET:
        raise ValueError(
            f"Declared step-4 smoke scope is {SMOKE_DATASET!r} only; got {job['dataset']!r} "
            "-- see configs/m1a_feature_screen.yaml#compute.recommended_step_4_smoke_scope"
        )
    return _run(job, stage="smoke")


@app.function(
    image=image,
    gpu=MODAL_CONFIG["gpu"],
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_feature_screen_headline(job: dict[str, Any]) -> dict[str, Any]:
    return _run(job, stage="headline")


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative = remote_path.removeprefix(f"{STORAGE_ROOT}/")
    with local_path.open("wb") as stream:
        for chunk in result_volume.read_file(relative):
            stream.write(chunk)


@app.local_entrypoint()
def main(datasets: str = SMOKE_DATASET, stage: str = "smoke") -> None:
    # For manual debugging only. The real run goes through
    # scripts/spawn_modal_jobs.py so the call survives client disconnection.
    if stage not in ("smoke", "headline"):
        raise ValueError(f"stage must be one of ('smoke', 'headline'); got {stage!r}")
    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    jobs = _jobs(requested)
    function = run_feature_screen_smoke if stage == "smoke" else run_feature_screen_headline
    results = list(function.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} M1A feature-screen job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "m1a_feature_screen" / stage
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}.json")
    print(json.dumps(results, indent=2))
