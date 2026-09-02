"""Modal launcher for the graph-context pilot (Stages B and C).

CPU only, read-only, no model trained. It shares the frozen data volume with the
GPU packages but writes to its own output prefix and never mutates a candidate
pool, a graph or a frozen result, so it is safe beside anything else.

Submit through ``scripts/spawn_modal_jobs.py graph-context`` so the calls are
server-side and survive client teardown. ``modal run --detach`` keeps only the
last triggered function alive once the launching process exits.
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
    if (HOST_REPO_ROOT / "configs" / "graph_context_pilot.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "graph_context_pilot.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]

# What a restart has to redo. The runner commits after each split and there is
# one split here, so the unit is the whole per-dataset job. Stage B is minutes;
# the ceiling is an hour.
RESUME_GRANULARITY = "split"
STORAGE_ROOT = MODAL_CONFIG["storage_root"]

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
    # The pilot trains nothing and builds no GNN, but it validates the frozen
    # candidate contract through the same helper every other package uses, and
    # that helper's import chain reaches operator_models. Using the shipped
    # validator rather than a copy is the point: a reimplementation could drift
    # from the check the rest of the project runs. Pinned to the substrate
    # audit's block exactly -- same shape, same CPU-only workload, already
    # proven on these graphs.
    .pip_install(
        "torch-geometric==2.5.2",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html",
    )
    .add_local_dir(str(RUNTIME_REPO_ROOT / "src"), remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir(str(RUNTIME_REPO_ROOT / "scripts"), remote_path=f"{REMOTE_ROOT}/scripts")
    .add_local_file(
        str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/graph_context_pilot.yaml"
    )
)


#: Stage and query cap for a spawned launch. The spawn path calls
#: ``module._jobs(datasets)`` with no extra arguments, so the stage is selected
#: here rather than through its argument parser -- one variable instead of a new
#: flag threaded through a launcher shared by eight packages.
STAGE = os.environ.get("GRAPH_CONTEXT_STAGE", "stage_b")
QUERY_CAP = int(os.environ.get("GRAPH_CONTEXT_QUERY_CAP", "25"))


def _jobs(
    datasets: list[str], stage: str = STAGE, query_cap: int = QUERY_CAP
) -> list[dict[str, Any]]:
    jobs = []
    for dataset in datasets:
        settings = CONFIG["datasets"][dataset]
        confirmation = json.loads(
            (HOST_REPO_ROOT / settings["confirmation"]).read_text(encoding="utf-8")
        )
        jobs.append(
            {
                "dataset": dataset,
                "settings": settings,
                "stage": stage,
                "query_cap": int(query_cap),
                "baseline": confirmation["baseline"],
                "fingerprint": confirmation["data_fingerprint_sha256"],
                "data_remote": confirmation["config"]["data"],
            }
        )
    return jobs


def _runner_args(job: dict[str, Any]) -> argparse.Namespace:
    settings = job["settings"]
    parameters = CONFIG["qls_v1_parameters"]
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "graph_context_pilot"
        / job["dataset"]
        / job["fingerprint"][:16]
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        stage=job["stage"],
        expected_queries=int(settings["expected_queries"]),
        baseline=job["baseline"],
        candidate_contract_compatibility=settings.get("candidate_contract_compatibility"),
        data_fingerprint_sha256=job["fingerprint"],
        splits=list(CONFIG["reporting"]["splits"]),
        query_cap=int(job["query_cap"]),
        damping=float(parameters["damping"]),
        ppr_iterations=int(parameters["ppr_iterations"]),
        output=Path(output_root) / f"{job['stage']}.json",
    )


@app.function(
    image=image,
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_context_pilot(job: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    from scripts.run_graph_context_pilot import run

    args = _runner_args(job)
    result = run(args, checkpoint_hook=result_volume.commit)
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "stage": job["stage"],
        "output_remote": str(args.output),
    }


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative = remote_path.removeprefix(f"{STORAGE_ROOT}/")
    with local_path.open("wb") as stream:
        for chunk in result_volume.read_file(relative):
            stream.write(chunk)


@app.local_entrypoint()
def main(
    datasets: str = "2wiki_clean,musique_clean,webqsp,hotpotqa_clean,squad_clean,metaqa",
    stage: str = "stage_b",
    query_cap: int = 25,
) -> None:
    """Blocking fallback. Prefer scripts/spawn_modal_jobs.py for a real submission."""

    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    unknown = set(requested) - set(CONFIG["datasets"])
    if unknown:
        raise ValueError(f"Unregistered graph-context datasets: {sorted(unknown)}")
    jobs = _jobs(requested, stage, query_cap)
    results = list(
        run_context_pilot.map(jobs, return_exceptions=True, wrap_returned_exceptions=False)
    )
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} graph-context job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "graph_context_pilot"
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}_{result['stage']}.json")
    print(json.dumps(results, indent=2))
