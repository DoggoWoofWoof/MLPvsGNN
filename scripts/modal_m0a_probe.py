"""Modal launcher for the M0A regime probe.

CPU only, zero GPU hours authorised. The probe reads the frozen data volume and
the Package B edge-provenance graphs, writes to its own output prefix, and
touches no historical path. Submit through scripts/spawn_modal_jobs.py so the
run is server-side; a detached local run is not a registered execution.
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
    if (HOST_REPO_ROOT / "configs" / "m0a_probe.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m0a_probe.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]
STORAGE_ROOT = MODAL_CONFIG["storage_root"]

if CONFIG["status"] != "DECLARED":
    raise RuntimeError("M0A is not declared; no declaration, no launch")
if MODAL_CONFIG.get("gpu") is not None:
    raise RuntimeError("M0A authorises zero GPU hours")

# One dataset is one spawned call that writes its result once, at the end. A
# restart therefore redoes exactly one dataset and no more, and a re-spawn of a
# finished dataset returns its recorded result without recomputing it.
RESUME_GRANULARITY = "dataset"

app = modal.App(MODAL_CONFIG["app"])
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
    .add_local_file(str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m0a_probe.yaml")
)


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
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
                "graph_root": str(
                    PurePosixPath(STORAGE_ROOT)
                    / MODAL_CONFIG["edge_provenance_root"]
                    / dataset
                    / fingerprint[:16]
                ),
            }
        )
    return jobs


def _runner_args(job: dict[str, Any]) -> argparse.Namespace:
    budget = CONFIG["budget"]
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "m0a_probe"
        / job["dataset"]
        / job["fingerprint"][:16]
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        data_fingerprint_sha256=job["fingerprint"],
        queries=int(CONFIG["sampling"]["queries_per_dataset"]),
        per_seed_cap=int(budget["per_seed_cap"]),
        graph_expansion_cap=int(budget["graph_expansion_cap"]),
        neighbour_scan_cap_per_seed=int(budget["neighbour_scan_cap_per_seed"]),
        edge_provenance_root=Path(job["graph_root"]),
        edge_families=list(CONFIG["edge_provenance_control"]["families"]),
        output=Path(output_root) / "probe.json",
    )


@app.function(
    image=image,
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_probe(job: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    from scripts.run_m0a_probe import run

    args = _runner_args(job)
    result = run(args)
    result_volume.commit()
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "output_remote": str(args.output),
    }


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative = remote_path.removeprefix(f"{STORAGE_ROOT}/")
    with local_path.open("wb") as stream:
        for chunk in result_volume.read_file(relative):
            stream.write(chunk)


@app.local_entrypoint()
def main(datasets: str = "squad_clean,2wiki_clean,metaqa") -> None:
    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    unknown = set(requested) - set(CONFIG["datasets"])
    if unknown:
        raise ValueError(f"Undeclared M0A datasets: {sorted(unknown)}")
    jobs = _jobs(requested)
    results = list(run_probe.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} M0A probe job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "m0a_probe"
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}.json")
    print(json.dumps(results, indent=2))
