"""Modal launcher for M1B's engineering smoke and (once authorised) the real 19-fit run.

GPU (A10G), same image recipe as scripts/modal_m1a_feature_screen.py -- M1B
reuses run_m1a_feature_screen.py's feature-build and per-seed-fit internals
unmodified (see scripts/run_m1b_targeted_resolution.py's own docstring), so
it needs the identical torch/torch-geometric/numba environment. A new module
rather than an edit to modal_m1a_feature_screen.py for the same reason
run_m1b_targeted_resolution.py is a new script rather than an edit to
run_m1a_feature_screen.py: configs/m1b_targeted_resolution.yaml#reuse_contract's
git-source-identity check names modal_m1a_feature_screen.py's _runner_args
hardcoded hyperparameters as one of its code-identity-proven fields, so that
file has to stay byte-unchanged since the M1A step-5 launch commit, not just
until the next convenient edit.

Two Modal functions, mirroring modal_m1a_feature_screen.py's smoke/headline
shape:
  - run_targeted_resolution_smoke: the declared engineering smoke only --
                         configs/m1b_targeted_resolution.yaml#launch_
                         authorization.smoke_spec, i.e. 2wiki_clean x R3 x
                         seed=1 x {BASE, BASE+NODE_ROLE+SUPPORT}, at this
                         track's own established 100-query diagnostic panel
                         size (modal.smoke_queries, reused from M1A's own
                         filed convention since M1B has no modal: block of
                         its own). Pipeline validation only -- no scientific
                         conclusion is drawn from it, and its result feeds
                         nothing in scientific_questions or
                         promotion_rule_for_m1b.
  - run_targeted_resolution_headline: every declared cell/arm/seed for the
                         requested dataset(s) -- the real 19 new fits (8 of
                         the 27 logical fits are spliced from M1A's own
                         seed=0 headline results instead, per reuse_contract;
                         see run_m1b_targeted_resolution.run() for the splice
                         itself). Refuses to run unless launch_authorization.
                         gates.engineering_smoke_passes is true in the
                         declaration at call time -- the smoke's whole job is
                         to earn that gate, so it is checked here rather than
                         at import time (which would make the smoke unable to
                         ever run first).

Infra (GPU shape, storage root, result volume, timeout, edge-provenance root,
smoke-panel size) is unchanged from M1A and reused verbatim from that file's
own modal: block -- configs/m1b_targeted_resolution.yaml declares no modal:
section of its own, since nothing about the infra differs. Only the app name
and output prefix (outputs/m1b_targeted_resolution/...) are M1B's own.

Submit through scripts/spawn_modal_jobs.py so the run is server-side and
survives client disconnection; a detached local run is not a registered
execution, exactly the same discipline modal_m1a_feature_screen.py's own
docstring states.
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
    if (HOST_REPO_ROOT / "configs" / "m1b_targeted_resolution.yaml").is_file()
    else Path(REMOTE_ROOT)
)
M1B_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m1b_targeted_resolution.yaml"
M1A_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m1a_feature_screen.yaml"
CONFIRMATION_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "sa_mlp_confirmation.yaml"
REUSE_MANIFEST_PATH = RUNTIME_REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "reuse_audit.json"

CONFIG = yaml.safe_load(M1B_CONFIG_PATH.read_text(encoding="utf-8"))
# spawn_modal_jobs.py's shared package protocol validates --datasets against
# module.CONFIG["datasets"] for every registered package. M1B's own
# declaration schema names this "cells" (fits/regime/arms are declared per
# dataset, not a flat list) -- this alias satisfies the shared launcher
# contract without changing the declaration file's own schema.
CONFIG["datasets"] = CONFIG["cells"]
M1A_CONFIG = yaml.safe_load(M1A_CONFIG_PATH.read_text(encoding="utf-8"))
CONFIRMATION_CONFIG = yaml.safe_load(CONFIRMATION_CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = dict(M1A_CONFIG["modal"])  # copied, not aliased -- the override below must never leak into M1A's own launcher
# Amendment 5 (2026-09-05, configs/m1b_targeted_resolution.yaml) retired
# splicing: every headline fit is re-run with row capture. execution_1's
# remote outputs are the pre-amendment-5, row-less results, and
# run_m1b_targeted_resolution.run()'s own COMPLETE_STATUS cache check would
# silently hand those back instead of re-executing if the re-run wrote to the
# same path. Bumping this M1B-local copy of execution_label writes the re-run
# beside execution_1 rather than over it -- the same precedent configs/
# m0a_probe.yaml#modal.execution_label already established (see its own
# why_a_label) for a second execution that must stay distinguishable from the
# first, not a new convention invented here. M1A's own shared modal block
# (configs/m1a_feature_screen.yaml) is left untouched at execution_1; MODAL_
# CONFIG above is a copy specifically so this line can never mutate it.
MODAL_CONFIG["execution_label"] = "execution_2"
STORAGE_ROOT = MODAL_CONFIG["storage_root"]
APP_NAME = "message-passing-retrieval-m1b-targeted-resolution"
ALL_DATASETS = tuple(CONFIG["cells"])  # exactly the four declared cells -- M1B's whole scope

SMOKE_DATASET = "webqsp"  # changed from 2wiki_clean -- see configs/m1b_targeted_resolution.yaml
# amendments[2] (2026-09-05, third filing): pilgnnteam's spend limit blocked
# 2wiki_clean's workspace, and no other M1A-step-5 workspace holds its data.
# webqsp's cell has no interaction arm, so this smoke no longer exercises
# BASE+NODE_ROLE+SUPPORT on real Modal/GPU -- see that amendment's accepted-
# gap note for what covers it instead.
SMOKE_SEEDS = [1]  # not already real in M1A -- genuinely exercises the new seed loop
SMOKE_ARMS = ["BASE", "BASE+NODE_ROLE"]
SMOKE_QUERIES = int(MODAL_CONFIG["smoke_queries"])

if CONFIG["status"] != "DECLARED_NOT_LAUNCHED":
    raise RuntimeError(f"Unexpected declaration status {CONFIG['status']!r} -- re-check before launching")

# The four gates that must already be true before this launcher runs either
# stage, plus engineering_tests_pass (the cached-tensor==rebuilt-tensor test
# and the rest of tests/test_run_m1b_targeted_resolution.py must already be
# green before even the smoke runs). engineering_smoke_passes is deliberately
# NOT checked here -- it is checked inside run_targeted_resolution_headline
# only, since the smoke function's whole job is to earn that gate, and
# gating it at import time would make the smoke unable to ever run first.
_GATES = CONFIG["launch_authorization"]["gates"]
_PREREQUISITE_GATES = (
    "amendment_filed",
    "reuse_audit_passes",
    "compute_within_ceiling",
    "bootstrap_and_promotion_rule_closed",
    "engineering_tests_pass",
)
_unmet = [name for name in _PREREQUISITE_GATES if not _GATES.get(name)]
if _unmet:
    raise RuntimeError(
        f"configs/m1b_targeted_resolution.yaml#launch_authorization.gates reports {_unmet} as "
        "not yet true -- these must all be true before this launcher is used for either stage"
    )

# One dataset is one spawned call that writes its result once, at the end,
# same convention as modal_m1a_feature_screen.py's own RESUME_GRANULARITY.
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
    .add_local_file(str(M1A_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m1a_feature_screen.yaml")
    .add_local_file(
        str(CONFIRMATION_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/sa_mlp_confirmation.yaml"
    )
    .add_local_file(str(M1B_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m1b_targeted_resolution.yaml")
    .add_local_file(
        str(REUSE_MANIFEST_PATH),
        remote_path=f"{REMOTE_ROOT}/outputs/m1b_targeted_resolution/reuse_audit.json",
    )
    .add_local_dir(
        str(HOST_REPO_ROOT / "outputs" / "m1a_feature_screen" / "headline"),
        remote_path=f"{REMOTE_ROOT}/outputs/m1a_feature_screen/headline",
    )
)


def _resolve_symbolic_arm(arm: str, *, x: str | None, y: str | None) -> str:
    """Substitute the bound family for a literal X/Y token in an arm name.

    Duplicated from run_m1b_targeted_resolution.py rather than imported --
    that module does heavy top-level imports (torch, mp_retrieval.*) that
    this launcher must stay free of locally, the same reason that module
    itself duplicates _resolve_repo_root from run_m1a_feature_screen.py
    instead of importing it. This is pure string substitution with no
    ARM_FAMILIES dependency, so nothing scientific is duplicated here.
    """
    resolved = []
    for token in arm.split("+"):
        if token == "X":
            if x is None:
                raise ValueError(f"arm {arm!r} uses the X placeholder but no x_binding is declared")
            resolved.append(x)
        elif token == "Y":
            if y is None:
                raise ValueError(f"arm {arm!r} uses the Y placeholder but no y_binding is declared")
            resolved.append(y)
        else:
            resolved.append(token)
    return "+".join(resolved)


def _declared_arms(dataset: str) -> list[str]:
    entry = CONFIG["cells"][dataset]
    x = entry.get("x_binding")
    y = entry.get("y_binding")
    return [_resolve_symbolic_arm(arm, x=x, y=y) for arm in entry["arms"]]


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    unknown = set(datasets) - set(ALL_DATASETS)
    if unknown:
        raise ValueError(f"Undeclared or out-of-scope M1B datasets: {sorted(unknown)}")
    from scripts.modal_m1a_feature_screen import _jobs as _m1a_jobs

    return _m1a_jobs(list(datasets))


def _runner_args(job: dict[str, Any], *, stage: str) -> argparse.Namespace:
    from scripts.run_m0b_regime_map import MAINLINE_FAMILY

    dataset = job["dataset"]
    declared_arms = _declared_arms(dataset)
    if stage == "smoke":
        if dataset != SMOKE_DATASET:
            raise ValueError(f"smoke stage is scoped to {SMOKE_DATASET!r}; got {dataset!r}")
        arms = list(SMOKE_ARMS)
        missing = set(arms) - set(declared_arms)
        if missing:
            raise ValueError(
                f"smoke arms {sorted(missing)} are not in {dataset!r}'s declared cell arms {declared_arms}"
            )
        seeds = list(SMOKE_SEEDS)
        queries = SMOKE_QUERIES
    else:
        arms = declared_arms
        seeds = list(CONFIG["seeds"]["development_seed_values"])
        queries = int(job["settings"]["validation_split_queries"])
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "m1b_targeted_resolution"
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
        arms=arms,
        semantic_rung="S3",
        seeds=seeds,
        epochs=3,
        batch_size=16,
        dropout=0.2,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device="cuda",
        output=Path(output_root) / "targeted_resolution.json",
    )


def _run(job: dict[str, Any], *, stage: str) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    result_volume.reload()
    from scripts.run_m1b_targeted_resolution import run

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
def run_targeted_resolution_smoke(job: dict[str, Any]) -> dict[str, Any]:
    if job["dataset"] != SMOKE_DATASET:
        raise ValueError(
            f"Declared engineering-smoke scope is {SMOKE_DATASET!r} only; got {job['dataset']!r} "
            "-- see configs/m1b_targeted_resolution.yaml#launch_authorization.smoke_spec"
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
def run_targeted_resolution_headline(job: dict[str, Any]) -> dict[str, Any]:
    declaration = yaml.safe_load(M1B_CONFIG_PATH.read_text(encoding="utf-8"))
    if not declaration["launch_authorization"]["gates"]["engineering_smoke_passes"]:
        raise RuntimeError(
            "configs/m1b_targeted_resolution.yaml#launch_authorization.gates."
            "engineering_smoke_passes is not true -- the engineering smoke must actually pass "
            "and that gate must be flipped to true before the real 19-fit run is spawned"
        )
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
    function = run_targeted_resolution_smoke if stage == "smoke" else run_targeted_resolution_headline
    results = list(function.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} M1B targeted-resolution job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "m1b_targeted_resolution" / stage
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}.json")
    print(json.dumps(results, indent=2))
