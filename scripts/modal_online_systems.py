"""Modal launcher for the frozen uncached unseen-embedding systems benchmark."""

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
    if (HOST_REPO_ROOT / "configs" / "online_systems.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "online_systems.yaml"
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
    .add_local_file(str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/online_systems.yaml")
    .add_local_file(
        str(SCREEN_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml"
    )
)


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    jobs = []
    budget = int(CONFIG["input_contract"]["budget"])
    for dataset in datasets:
        settings = CONFIG["datasets"][dataset]
        confirmation = json.loads(
            (HOST_REPO_ROOT / settings["confirmation"]).read_text(encoding="utf-8")
        )
        fingerprint = confirmation["data_fingerprint_sha256"]
        budget_root = (
            PurePosixPath(STORAGE_ROOT)
            / "outputs"
            / "candidate_budget"
            / dataset
            / fingerprint[:16]
            / f"budget_{budget}"
        )
        cache_root = (
            PurePosixPath(STORAGE_ROOT)
            / "candidate_budget_cache"
            / dataset
            / fingerprint[:16]
            / f"budget_{budget}"
        )
        output_root = (
            PurePosixPath(STORAGE_ROOT)
            / "outputs"
            / "online_systems"
            / dataset
            / fingerprint[:16]
        )
        jobs.append(
            {
                "dataset": dataset,
                "settings": settings,
                "confirmation": confirmation,
                "fingerprint": fingerprint,
                "data_remote": confirmation["config"]["data"],
                "budget_result": str(budget_root / "result.json"),
                "cached_topology": str(cache_root / "packed_topology_v1"),
                "cached_features": str(cache_root / "fixed_structural_features_v1"),
                "output_remote": str(output_root / "result.json"),
            }
        )
    return jobs


def _runner_args(job: dict[str, Any]) -> argparse.Namespace:
    inputs = CONFIG["input_contract"]
    measurement = CONFIG["measurement"]
    parameters = CONFIG["parameter_regime"]
    screen = yaml.safe_load(
        Path(f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml").read_text(encoding="utf-8")
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        data_fingerprint_sha256=job["fingerprint"],
        budget_result=Path(job["budget_result"]),
        cached_topology=Path(job["cached_topology"]),
        cached_features=Path(job["cached_features"]),
        output=Path(job["output_remote"]),
        baseline=job["confirmation"]["baseline"],
        selected_gnn=job["settings"]["selected_gnn"],
        budget=int(inputs["budget"]),
        rrf_constant=int(inputs["rrf_constant"]),
        model_seed=int(inputs["model_seed"]),
        top_k=int(inputs["top_k"]),
        sample_queries=int(measurement["deterministic_evenly_spaced_queries"]),
        parity_queries=int(measurement["parity_queries"]),
        warmup_queries=int(measurement["warmup_queries"]),
        repeats=int(measurement["repeats"]),
        batch_sizes=[int(value) for value in measurement["batch_sizes"]],
        projection_dim=int(parameters["projection_dim"]),
        hidden_dim=int(parameters["hidden_dim"]),
        max_parameter_difference=int(parameters["maximum_sa_absolute_difference"]),
        layers=int(parameters["layers"]),
        dropout=float(parameters["dropout"]),
        temperature=float(parameters["temperature"]),
        device="cuda",
        feature_config={
            "query_local_features": screen["query_local_features"],
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
def run_dataset(job: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    required = [
        Path(job["budget_result"]),
        Path(job["cached_topology"]) / "metadata.json",
        Path(job["cached_features"]) / "metadata.json",
    ]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Package C budget-400 artifacts are not complete")
    from scripts.run_online_systems import run

    args = _runner_args(job)
    result = run(args)
    result_volume.commit()
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "output_remote": job["output_remote"],
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
        raise ValueError(f"Unregistered online-systems datasets: {sorted(unknown)}")
    jobs = _jobs(requested)
    results = list(run_dataset.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} online-systems job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "online_systems"
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}.json")
    print(json.dumps(results, indent=2))
