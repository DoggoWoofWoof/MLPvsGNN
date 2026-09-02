"""Modal launcher for the graph-context pilot (Stages B, C, D0 and D0b).

CPU only, read-only. It shares the frozen data volume with the GPU packages but
writes to its own output prefix and never mutates a candidate pool, a graph or a
frozen result, so it is safe beside anything else.

Stages B, C and D0 train nothing. D0b fits four linear models of at most twenty
parameters, which is still a CPU job and still costs cents -- being cheap enough
to run before the GPU stage is the whole reason it exists.

Submit through ``scripts/spawn_modal_jobs.py graph-context`` so the calls are
server-side and survive client teardown. ``modal run --detach`` keeps only the
last triggered function alive once the launching process exits.
"""

from __future__ import annotations

import argparse
import json
import os
import time
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

#: The container window, per stage. Modal fixes a timeout when the function is
#: decorated, so this is resolved at import time from the same variable that
#: selects the stage. D0b runs a whole split where B and C run a capped prefix,
#: so it gets its own window instead of the shared one being loosened for stages
#: that do not need it.
TIMEOUT_SECONDS = int(
    MODAL_CONFIG.get("stage_timeout_seconds", {}).get(STAGE, MODAL_CONFIG["timeout_seconds"])
)


def _stage_key(stage: str) -> str:
    """`stage_b` -> `B`, `stage_d0` -> `D0`. The config keys, not a character index."""
    return stage.removeprefix("stage_").upper()


def _selected_learning_rate(dataset: str, stage: str) -> float | None:
    """The rate A3's frozen protocol selected for this dataset, or None.

    Only D0b needs it, and only D0b pays the cost of a missing A3 result -- the
    other stages train nothing, so a dataset without a sealed linear control
    must not become an error for them.
    """
    if stage != "stage_d0b":
        return None
    path = HOST_REPO_ROOT / "outputs" / "p0_linear_rank_structure" / f"{dataset}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage D0b reuses A3's selected learning rate and {path} is absent"
        )
    return float(json.loads(path.read_text(encoding="utf-8"))["selected_learning_rate"])


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
                # D0b fits the frozen A3 linear control, so it needs A3's sealed
                # static features and the learning rate A3's protocol already
                # selected for this dataset. Carried, never re-selected: a stage
                # that picked its own rate per arm would be running the small
                # architecture search it was told not to run.
                "feature_remote": str(confirmation["config"]["feature_cache"]),
                "selected_learning_rate": _selected_learning_rate(dataset, stage),
                # Stage C runs Stage B's survivors and D0 runs two of them, so
                # the arm list comes from the config rather than the runner's
                # default. `stage_c` reads `stages.C`, `stage_d0` reads
                # `stages.D0` -- the suffix, not the last character, or D0 would
                # silently read `stages.0` and fall back to all six arms.
                "arms": CONFIG["stages"].get(_stage_key(stage), {}).get("arms"),
                "query_cap": int(query_cap),
                "baseline": confirmation["baseline"],
                "fingerprint": confirmation["data_fingerprint_sha256"],
                "data_remote": confirmation["config"]["data"],
            }
        )
    return jobs


def _d0b_runner_args(job: dict[str, Any]) -> argparse.Namespace:
    """Stage D0b's arguments. It fits a ranker, so it needs more than a context.

    Train and validation, both: the diagnostic exists to be fit on one and read
    on the other. `test` is absent here and refused by the runner, which is two
    places rather than one on purpose -- a launcher typo should not be able to
    open the test split.
    """
    settings = job["settings"]
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "graph_context_pilot"
        / job["dataset"]
        / job["fingerprint"][:16]
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        feature_cache=Path(job["feature_remote"]),
        dataset=job["dataset"],
        expected_queries=int(settings["expected_queries"]),
        baseline=job["baseline"],
        candidate_contract_compatibility=settings.get("candidate_contract_compatibility"),
        data_fingerprint_sha256=job["fingerprint"],
        splits=["train", "validation"],
        query_cap=int(job["query_cap"]),
        rrf_constant=int(CONFIG["stages"]["D0B"]["rrf_constant"]),
        learning_rate=float(job["selected_learning_rate"]),
        seed=int(CONFIG["stages"]["D0B"]["seed"]),
        output=Path(output_root) / "stage_d0b.json",
    )


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
        arms=job.get("arms"),
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
    timeout=TIMEOUT_SECONDS,
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_context_pilot(job: dict[str, Any]) -> dict[str, Any]:
    """Stages B, C, D0 and D0b through one function.

    They share the image, the volume, the loader and the contract check; they
    differ in what they measure. A second Modal module would have been a second
    image to keep pinned and a second registry entry to keep honest, for no
    isolation these jobs need.

    D0b fits a nineteen-parameter linear model and is still a CPU job: the whole
    point of putting it before the GPU stage is that it costs almost nothing.
    """
    os.chdir(REMOTE_ROOT)
    if job["stage"] == "stage_d0b":
        from scripts.run_graph_context_d0b import run

        args = _d0b_runner_args(job)
    else:
        if job["stage"] == "stage_d0":
            from scripts.run_graph_context_d0 import run
        else:
            from scripts.run_graph_context_pilot import run

        args = _runner_args(job)
    started = time.monotonic()
    result = run(args, checkpoint_hook=result_volume.commit)
    # The container is what is billed, and Stage D0 closed out with a wall-clock
    # bound rather than a figure because nothing recorded this. It is the
    # launcher's job, not the runner's: the runner does not know it is on Modal.
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "stage": job["stage"],
        "output_remote": str(args.output),
        "elapsed_seconds": round(time.monotonic() - started, 1),
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
