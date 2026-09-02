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

#: The accelerator, per stage, for the same reason and by the same mechanism.
#: B, C, D0 and D0b measure structure or fit twenty parameters and get ``None``;
#: D1 scores from 1536-dimensional embeddings through the frozen QLS-v1 ranker
#: and gets the shape every other trained package in this project uses. Reading
#: it here rather than in a second Modal function keeps one entrypoint, which is
#: what the spawn registry addresses.
GPU = MODAL_CONFIG["gpu"] if STAGE in set(MODAL_CONFIG.get("gpu_stages", ())) else None


def _stage_key(stage: str) -> str:
    """`stage_b` -> `B`, `stage_d0` -> `D0`. The config keys, not a character index."""
    return stage.removeprefix("stage_").upper()


def _selected_learning_rate(dataset: str, stage: str) -> float | None:
    """The rate A3's frozen protocol selected for this dataset, or None.

    Only the linear stages need it, and only they pay the cost of a missing A3
    result -- the other stages train nothing, so a dataset without a sealed
    linear control must not become an error for them.
    """
    if stage not in {"stage_d0b", "stage_d0c"}:
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
                # D1 runs the frozen QLS-v1 ranker, so its hyperparameters come
                # from the sealed confirmation that fixed them rather than from
                # this launcher. Carried whole: a subset copied here would be a
                # second place for them to drift.
                "training": confirmation["config"],
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
    _static = CONFIG["stages"]["D0B"]["static_features"]
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
        static_pagerank_damping=float(_static["pagerank_damping"]),
        static_pagerank_iterations=int(_static["pagerank_iterations"]),
        static_clustering_max_wedges=int(_static["clustering_max_wedges_per_node"]),
        output=Path(output_root) / "stage_d0b.json",
    )


def _d0c_runner_args(job: dict[str, Any]) -> argparse.Namespace:
    """Stage D0c's arguments: D0b's, plus the tail that replaces its fixed budget.

    Everything shared with D0b is read from D0B's own config block rather than
    copied into a D0C one. D0c is a follow-up to that stage under a different
    epoch budget and arm set; if the static parameters or the RRF constant ever
    moved, the two stages diverging silently would be worse than either value.

    No `query_cap`: D0c has none. The stage exists to be read against D0b, and a
    capped run would answer a different question at the same price.
    """
    settings = job["settings"]
    _static = CONFIG["stages"]["D0B"]["static_features"]
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
        holdout_fraction=float(CONFIG["stages"]["D0C"]["holdout_fraction"]),
        rrf_constant=int(CONFIG["stages"]["D0B"]["rrf_constant"]),
        learning_rate=float(job["selected_learning_rate"]),
        seed=int(CONFIG["stages"]["D0C"]["seed"]),
        static_pagerank_damping=float(_static["pagerank_damping"]),
        static_pagerank_iterations=int(_static["pagerank_iterations"]),
        static_clustering_max_wedges=int(_static["clustering_max_wedges_per_node"]),
        output=Path(output_root) / "stage_d0c.json",
    )


def _d2_runner_args(job: dict[str, Any]) -> argparse.Namespace:
    """Stage D2's arguments: D1's, with one context and one output name.

    Every training value comes from the sealed confirmation config by the same
    path D1 uses, because the whole stage rests on the two arms training exactly
    as D1's CAND arm did -- FULL_CAND is meant to reproduce it, and a
    hyperparameter typed twice is a hyperparameter that can drift once.
    """
    args = _d1_runner_args(job)
    args.holdout_fraction = float(CONFIG["stages"]["D2"]["holdout_fraction"])
    args.seed = int(CONFIG["stages"]["D2"]["seed"])
    args.output = Path(args.output).with_name("stage_d2.json")
    return args


def _d3_runner_args(job: dict[str, Any]) -> argparse.Namespace:
    """Stage D3's arguments: D2's, plus the path to the result it reuses.

    D3 decomposes D2 at D2's own operating point, so every training value is
    taken from D2's argument builder rather than restated. Two stages that must
    agree about what "identical conditions" means should not each carry their own
    copy of them; the runner then re-checks the agreement against D2's result
    file and refuses if anything drifted.
    """
    args = _d2_runner_args(job)
    args.d2_result = Path(args.output)
    args.holdout_fraction = float(CONFIG["stages"]["D3"]["holdout_fraction"])
    args.seed = int(CONFIG["stages"]["D3"]["seed"])
    args.output = Path(args.output).with_name("stage_d3.json")
    return args


def _d4_runner_args(job: dict[str, Any]) -> argparse.Namespace:
    """Stage D4's arguments: D3's, plus the result it reuses and A3's constant.

    D4 reuses D3's SEED_ID_ONLY as its control, so it must be at D3's operating
    point exactly -- taken from D3's own argument builder rather than restated.
    The RRF constant comes from D0B's block, which is where A3's frozen fusion
    constant already lives; D4 reuses A3's transform, so it must not carry a
    second copy of the constant that transform is defined by.
    """
    args = _d3_runner_args(job)
    args.d3_result = Path(args.output)
    # D4 reuses D3, and D3 reused D2. Carrying D2's path forward would suggest
    # this stage reads it, which it does not.
    del args.d2_result
    args.holdout_fraction = float(CONFIG["stages"]["D4"]["holdout_fraction"])
    args.seed = int(CONFIG["stages"]["D4"]["seed"])
    args.rrf_constant = int(CONFIG["stages"]["D0B"]["rrf_constant"])
    args.output = Path(args.output).with_name("stage_d4.json")
    return args


def _d5_runner_args(job: dict[str, Any]) -> argparse.Namespace:
    """Stage D5's arguments: D4's, plus the second result file it reuses.

    D5 assembles a five-row table from two earlier stages and trains one arm, so
    it must be at the operating point BOTH of them ran at. Taking D4's builder --
    which takes D3's, which takes D2's -- is what makes that true by construction
    rather than by three copies of the same numbers agreeing. `d3_result` is
    already set by D4's builder; only D4's own path is new here.

    The RRF constant arrives the same way, from D0B's block through D4. D5
    composes D4's prior and must not be able to disagree with it about K.
    """
    args = _d4_runner_args(job)
    args.d4_result = Path(args.output)
    args.holdout_fraction = float(CONFIG["stages"]["D5"]["holdout_fraction"])
    args.seed = int(CONFIG["stages"]["D5"]["seed"])
    args.output = Path(args.output).with_name("stage_d5.json")
    return args


def _d6_runner_args(job: dict[str, Any]) -> argparse.Namespace:
    """Stage D6's arguments: D5's, minus the files it does not read.

    D6 trains both of its own arms and reuses nothing, so it reads exactly one
    earlier result -- D5's -- and only to quote it as a descriptive reference.
    Carrying D3's and D4's paths forward would suggest this stage reads them,
    which it does not. The operating point still arrives through D5's builder,
    because the reference is only quotable if D6 ran where D5 did.
    """
    args = _d5_runner_args(job)
    args.d5_result = Path(args.output)
    del args.d3_result
    del args.d4_result
    args.holdout_fraction = float(CONFIG["stages"]["D6"]["holdout_fraction"])
    args.seed = int(CONFIG["stages"]["D6"]["seed"])
    args.output = Path(args.output).with_name("stage_d6.json")
    return args


def _d7_runner_args(job: dict[str, Any]) -> argparse.Namespace:
    """Stage D7's arguments: D6's, pointed at D6's result instead of D5's.

    D7 reuses D6's two arms as an exact matched control rather than refitting
    them, so unlike every earlier stage in this chain the operating point is not
    merely quotable -- it is load-bearing. Taking D6's builder is what makes D7
    run where D6 ran by construction; the runner then proves the reuse against
    the statistics D6 recorded and refuses if any of them disagrees.
    """
    args = _d6_runner_args(job)
    args.d6_result = Path(args.output)
    del args.d5_result
    args.holdout_fraction = float(CONFIG["stages"]["D7"]["holdout_fraction"])
    args.seed = int(CONFIG["stages"]["D7"]["seed"])
    args.output = Path(args.output).with_name("stage_d7.json")
    return args


def _d1_runner_args(job: dict[str, Any]) -> argparse.Namespace:
    """Stage D1's arguments: the frozen QLS-v1 hyperparameters, unchanged.

    Every training value here comes from the sealed confirmation config rather
    than from this file. D1 exists to move one thing -- the node space the
    feature kernel runs on -- and a hyperparameter typed twice is a
    hyperparameter that can drift once.
    """
    settings = job["settings"]
    stage = CONFIG["stages"]["D1"]
    _static = CONFIG["stages"]["D0B"]["static_features"]
    training = job["training"]
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
        holdout_fraction=0.1,
        selected_gnn=job["baseline"]["selected_gnn"]["model"],
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        hidden_dim=int(training["hidden_dim"]),
        projection_dim=int(training["projection_dim"]),
        layers=int(training["layers"]),
        dropout=float(training["dropout"]),
        temperature=float(training["temperature"]),
        ks=[1, 5, 20],
        damping=0.85,
        ppr_iterations=8,
        static_pagerank_damping=float(_static["pagerank_damping"]),
        static_pagerank_iterations=int(_static["pagerank_iterations"]),
        static_clustering_max_wedges=int(_static["clustering_max_wedges_per_node"]),
        seed=int(stage["seed"]),
        device=None,
        output=Path(output_root) / "stage_d1.json",
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
    gpu=GPU,
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
    elif job["stage"] == "stage_d0c":
        from scripts.run_graph_context_d0c import run

        args = _d0c_runner_args(job)
    elif job["stage"] == "stage_d1":
        from scripts.run_graph_context_d1 import run

        args = _d1_runner_args(job)
    elif job["stage"] == "stage_d2":
        from scripts.run_graph_context_d2 import run

        args = _d2_runner_args(job)
    elif job["stage"] == "stage_d3":
        from scripts.run_graph_context_d3 import run

        args = _d3_runner_args(job)
    elif job["stage"] == "stage_d4":
        from scripts.run_graph_context_d4 import run

        args = _d4_runner_args(job)
    elif job["stage"] == "stage_d5":
        from scripts.run_graph_context_d5 import run

        args = _d5_runner_args(job)
    elif job["stage"] == "stage_d6":
        from scripts.run_graph_context_d6 import run

        args = _d6_runner_args(job)
    elif job["stage"] == "stage_d7":
        from scripts.run_graph_context_d7 import run

        args = _d7_runner_args(job)
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
