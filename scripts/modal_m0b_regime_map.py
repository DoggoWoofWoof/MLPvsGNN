"""Modal launcher for M0B steps 2 and 4: the six-dataset R1/R2/R3 regime map.

CPU only, zero GPU hours authorised. Reuses the frozen candidate pools and the
Package B edge-provenance graphs; writes to its own output prefix and touches
no historical M0A/M0A.1/Safeguard-C path. Submit through
scripts/spawn_modal_jobs.py so the run is server-side; a detached local run is
not a registered execution.

Unlike scripts/modal_m0b_webqsp_probe.py (Safeguard C, webqsp only, filed and
already run), this launcher covers all six datasets and both of M0B's
sampling stages, exposed as two separate Modal functions rather than a job
field -- scripts/spawn_modal_jobs.py's own --stage flag already means "which
registered function to call" (see its PACKAGES dict; "edge-provenance" uses
the same {name: function} idiom for prepare/train), so a same-named but
differently-meaning "stage" job field would collide with it, not compose:
  - stage="smoke"    -> run_regime_map_smoke:    sampling.smoke_sample_step_2
                         (5 queries/dataset). Step 2: invariant/construction
                         correctness only.
  - stage="headline" -> run_regime_map_headline: sampling.headline_run_step_4
                         (100 queries/dataset). Step 4: only ever launched
                         after step 2's invariants and step 3's updated
                         compute projection both clear.
Both reuse the exact filed queries_per_dataset; neither is a runtime knob
here. They write to different output paths (nested by stage) so a headline
run never short-circuits on a smoke run's COMPLETE_STATUS file, and so a
single failed smoke cell can be re-run alone without re-running the other
five (see run_m0b_regime_map.py's own per-output idempotency check).
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
    if (HOST_REPO_ROOT / "configs" / "m0b_regime_map.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m0b_regime_map.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]
STORAGE_ROOT = MODAL_CONFIG["storage_root"]
# configs/m0b_regime_map.yaml's modal.app is
# "message-passing-retrieval-m0b-webqsp-probe" -- named narrowly for
# Safeguard C, which was the only thing filed under this config when that
# name was chosen. Left untouched (it is scripts/modal_m0b_webqsp_probe.py's
# already-run app identity, not a value this launcher should silently
# repurpose); this launcher gets its own app name instead, scoped to what it
# actually runs.
APP_NAME = "message-passing-retrieval-m0b-regime-map"
MAINLINE_FAMILY = CONFIG["edge_provenance_family_backing_a64"]["chosen"]
ALL_DATASETS = tuple(CONFIG["datasets"])
STAGE_SAMPLING_KEY = {"smoke": "smoke_sample_step_2", "headline": "headline_run_step_4"}

if MODAL_CONFIG.get("gpu") is not None:
    raise RuntimeError("M0B authorises zero GPU hours")

# One dataset is one spawned call that writes its result once, at the end. A
# restart therefore redoes exactly one dataset (at whichever stage's function
# was called) and no more -- matching the filed failure-handling rule that a
# single bad smoke cell is re-run alone, not the whole matrix.
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
    .add_local_file(str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m0b_regime_map.yaml")
)


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    unknown = set(datasets) - set(ALL_DATASETS)
    if unknown:
        raise ValueError(f"Undeclared M0B datasets: {sorted(unknown)}")
    jobs = []
    for dataset in datasets:
        settings = CONFIG["datasets"][dataset]
        confirmation = json.loads(
            (HOST_REPO_ROOT / settings["confirmation"]).read_text(encoding="utf-8")
        )
        fingerprint = confirmation["data_fingerprint_sha256"]
        jobs.append(
            {
                "dataset": dataset,
                "settings": settings,
                "fingerprint": fingerprint,
                "data_remote": confirmation["config"]["data"],
                "baseline": confirmation["baseline"],
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
    sampling = CONFIG["sampling"][STAGE_SAMPLING_KEY[stage]]
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "m0b_regime_map"
        / job["dataset"]
        / job["fingerprint"][:16]
        / MODAL_CONFIG["execution_label"]
        / stage
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        data_fingerprint_sha256=job["fingerprint"],
        expected_queries=int(job["settings"]["expected_queries"]),
        baseline=job["baseline"],
        candidate_contract_compatibility=job["settings"].get(
            "candidate_contract_compatibility"
        ),
        queries=int(sampling["queries_per_dataset"]),
        per_seed_cap=16,
        neighbour_scan_cap_per_seed=4096,
        edge_provenance_root=Path(job["graph_root"]),
        edge_families=[MAINLINE_FAMILY],
        a64_mainline_family=MAINLINE_FAMILY,
        output=Path(output_root) / "regime_map.json",
    )


def _run(job: dict[str, Any], *, stage: str) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    from scripts.run_m0b_regime_map import run

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
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_regime_map_smoke(job: dict[str, Any]) -> dict[str, Any]:
    return _run(job, stage="smoke")


@app.function(
    image=image,
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_regime_map_headline(job: dict[str, Any]) -> dict[str, Any]:
    return _run(job, stage="headline")


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative = remote_path.removeprefix(f"{STORAGE_ROOT}/")
    with local_path.open("wb") as stream:
        for chunk in result_volume.read_file(relative):
            stream.write(chunk)


@app.local_entrypoint()
def main(datasets: str = ",".join(ALL_DATASETS), stage: str = "smoke") -> None:
    # For manual debugging only. The real run goes through
    # scripts/spawn_modal_jobs.py so the call survives client disconnection --
    # see this module's docstring and project memory on why `modal run
    # --detach` is never used here.
    if stage not in STAGE_SAMPLING_KEY:
        raise ValueError(f"stage must be one of {sorted(STAGE_SAMPLING_KEY)}; got {stage!r}")
    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    jobs = _jobs(requested)
    function = run_regime_map_smoke if stage == "smoke" else run_regime_map_headline
    results = list(function.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} M0B regime-map job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "m0b_regime_map" / stage
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}.json")
    print(json.dumps(results, indent=2))
