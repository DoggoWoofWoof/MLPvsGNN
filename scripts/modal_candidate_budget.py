"""Modal launcher for the frozen Package C candidate-budget study."""

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
    if (HOST_REPO_ROOT / "configs" / "candidate_budget.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "candidate_budget.yaml"
SCREEN_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "sa_mlp_screen.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]
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
    .pip_install(
        "torch-geometric==2.5.2",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html",
    )
    .add_local_dir(str(RUNTIME_REPO_ROOT / "src"), remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir(str(RUNTIME_REPO_ROOT / "scripts"), remote_path=f"{REMOTE_ROOT}/scripts")
    .add_local_file(str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/candidate_budget.yaml")
    .add_local_file(
        str(SCREEN_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml"
    )
)


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    jobs = []
    for dataset in datasets:
        settings = CONFIG["datasets"][dataset]
        confirmation = json.loads(
            (HOST_REPO_ROOT / settings["confirmation"]).read_text(encoding="utf-8")
        )
        for budget in CONFIG["candidate_contract"]["budgets"]:
            jobs.append(
                {
                    "dataset": dataset,
                    "budget": int(budget),
                    "settings": settings,
                    "confirmation": confirmation,
                    "fingerprint": confirmation["data_fingerprint_sha256"],
                    "data_remote": confirmation["config"]["data"],
                }
            )
    return jobs


def _runner_args(job: dict[str, Any]) -> argparse.Namespace:
    settings = job["settings"]
    training = CONFIG["training"]
    parameters = CONFIG["parameter_regime"]
    contract = CONFIG["candidate_contract"]
    cache_root = (
        PurePosixPath(STORAGE_ROOT)
        / "candidate_budget_cache"
        / job["dataset"]
        / job["fingerprint"][:16]
        / f"budget_{job['budget']}"
    )
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "candidate_budget"
        / job["dataset"]
        / job["fingerprint"][:16]
        / f"budget_{job['budget']}"
    )
    screen = yaml.safe_load(
        Path(f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml").read_text(encoding="utf-8")
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        budget=int(job["budget"]),
        rrf_constant=int(contract["rrf_constant"]),
        fusion_chunk_size=4096,
        expected_queries=int(settings["expected_queries"]),
        output=Path(output_root) / "result.json",
        query_metrics_output=Path(output_root) / "query_metrics.npz",
        checkpoint_root=Path(output_root) / "checkpoints",
        topology_cache=Path(cache_root) / "packed_topology_v1",
        feature_cache=Path(cache_root) / "fixed_structural_features_v1",
        baseline=job["confirmation"]["baseline"],
        data_fingerprint_sha256=job["fingerprint"],
        selected_gnn=settings["selected_gnn"],
        candidate_contract_compatibility=settings.get("candidate_contract_compatibility"),
        required_hops=list(settings["required_hops"]),
        seeds=list(training["seeds"]),
        projection_dim=int(parameters["projection_dim"]),
        hidden_dim=int(parameters["hidden_dim"]),
        max_parameter_difference=int(parameters["maximum_sa_absolute_difference"]),
        layers=int(training["layers"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        dropout=float(training["dropout"]),
        temperature=float(training["temperature"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        ks=list(training["ks"]),
        inference_repeats=int(training["inference_repeats"]),
        device="cuda",
        feature_config={
            "retrieval_seeds": screen["retrieval_seeds"],
            "static_features": screen["static_features"],
            "query_local_features": screen["query_local_features"],
            "preprocessing": {"query_chunk_size": 8192},
        },
    )


@app.function(
    image=image,
    gpu=MODAL_CONFIG["gpu"],
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_budget(job: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    result_volume.reload()
    from scripts.run_candidate_budget import run

    args = _runner_args(job)
    result = run(args, checkpoint_hook=result_volume.commit)
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "budget": job["budget"],
        "output_remote": str(args.output),
        "query_metrics_remote": str(args.query_metrics_output),
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
) -> None:
    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    unknown = set(requested) - set(CONFIG["datasets"])
    if unknown:
        raise ValueError(f"Unregistered candidate-budget datasets: {sorted(unknown)}")
    jobs = _jobs(requested)
    results = list(run_budget.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} candidate-budget job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "candidate_budget"
    for result in results:
        budget_root = local_root / result["dataset"] / f"budget_{result['budget']}"
        _download(result["output_remote"], budget_root / "result.json")
        _download(result["query_metrics_remote"], budget_root / "query_metrics.npz")
    print(json.dumps(results, indent=2))
