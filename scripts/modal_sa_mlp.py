"""Modal execution for the frozen one-seed Structure-Aware MLP screen."""

from __future__ import annotations

import argparse
import hashlib
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
    if (HOST_REPO_ROOT / "configs" / "sa_mlp_screen.yaml").exists()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "sa_mlp_screen.yaml"
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
    .add_local_file(str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_summary(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload["selection_validation_only"]["selected"]
    if selected != CONFIG["baseline"]["frozen_models"][payload["dataset"]]:
        raise ValueError("Local baseline selected GNN differs from the frozen SA-MLP contract")

    def seed_summary(model_name: str) -> dict[str, Any]:
        model = payload["models"][model_name]
        seed = model["seeds"]["0"]
        return {
            "model": model_name,
            "parameters": model["parameters"],
            "metrics": seed["metrics"],
            "inference": seed["inference"],
            "by_hop": seed["by_hop"],
        }

    return {
        "dataset": payload["dataset"],
        "queries": payload["data"]["queries"],
        "data_fingerprint_sha256": payload["data_fingerprint_sha256"],
        "candidate_contract_sha256": payload["data"]["candidate_contract_sha256"],
        "plain_mlp": seed_summary("plain_mlp"),
        "gnn": seed_summary(selected),
    }


def _remote_result_path(dataset: str, fingerprint: str) -> str:
    return str(
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "sa_mlp_screen"
        / dataset
        / fingerprint[:16]
        / "result.json"
    )


def _local_jobs(datasets: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_config = CONFIG["datasets"][dataset]
        path = HOST_REPO_ROOT / CONFIG["baseline"]["source_pattern"].format(dataset=dataset)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PAPER_MAIN_TABLE_DATASET_COMPLETE":
            raise ValueError(f"Frozen main-table artifact is incomplete: {path}")
        data_remote = payload["config"]["data"]
        baseline_remote = payload["config"]["output"]
        fingerprint = payload["data_fingerprint_sha256"]
        jobs.append(
            {
                "dataset": dataset,
                "expected_queries": int(dataset_config["expected_queries"]),
                "required_hops": list(dataset_config["required_hops"]),
                "data_remote": data_remote,
                "baseline_remote": baseline_remote,
                "baseline_result_sha256": _sha256(path),
                "baseline": _baseline_summary(payload),
                "fingerprint": fingerprint,
                "result_remote": _remote_result_path(dataset, fingerprint),
            }
        )
    return jobs


def _runner_namespace(job: dict[str, Any]) -> argparse.Namespace:
    training = CONFIG["training"]
    parameters = CONFIG["parameter_regime"]
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        expected_queries=int(job["expected_queries"]),
        output=Path(job["result_remote"]),
        topology_cache=Path(job["data_remote"]) / "derived" / "packed_topology_v1",
        feature_cache=Path(job["data_remote"]) / "derived" / "fixed_structural_features_v1",
        baseline=job["baseline"],
        baseline_result_sha256=job["baseline_result_sha256"],
        feature_config={
            "retrieval_seeds": CONFIG["retrieval_seeds"],
            "static_features": CONFIG["static_features"],
            "query_local_features": CONFIG["query_local_features"],
            "preprocessing": {"query_chunk_size": 8192},
        },
        required_hops=list(job["required_hops"]),
        seed=int(training["seeds"][0]),
        projection_dim=int(parameters["projection_dim"]),
        max_parameter_difference=int(parameters["allowed_absolute_parameter_difference"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        dropout=float(training["dropout"]),
        temperature=float(training["temperature"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        ks=[1, 5, 20],
        inference_repeats=int(training["inference_repeats"]),
        gap_closure_threshold=float(CONFIG["stop_gate"]["pass_threshold"]),
        device="cuda",
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
    result_volume.reload()
    baseline_path = Path(job["baseline_remote"])
    if not baseline_path.is_file() or _sha256(baseline_path) != job["baseline_result_sha256"]:
        raise ValueError("Remote frozen baseline artifact failed its SHA-256 check")
    from scripts.run_sa_mlp_screen import run

    result = run(_runner_namespace(job), checkpoint_hook=result_volume.commit)
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "fingerprint": job["fingerprint"],
        "result": job["result_remote"],
        "gap_closure": result["gap_closure"],
    }


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative = remote_path.removeprefix(f"{STORAGE_ROOT}/")
    with local_path.open("wb") as stream:
        for chunk in result_volume.read_file(relative):
            stream.write(chunk)


@app.local_entrypoint()
def main(datasets: str = "metaqa,webqsp,hotpotqa_clean") -> None:
    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    unknown = set(requested) - set(CONFIG["datasets"])
    if unknown:
        raise ValueError(f"Unregistered SA-MLP datasets: {sorted(unknown)}")
    jobs = _local_jobs(requested)
    results = list(
        run_dataset.map(jobs, return_exceptions=True, wrap_returned_exceptions=False)
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} SA-MLP job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "sa_mlp_screen"
    for result in results:
        _download(result["result"], local_root / f"{result['dataset']}.json")
    passed = sum(bool(result["gap_closure"]["dataset_pass"]) for result in results)
    decision = {
        "status": "SA_MLP_SCREEN_GATE_COMPLETE",
        "datasets_passed": passed,
        "datasets_required": int(CONFIG["stop_gate"]["datasets_required_to_pass"]),
        "gate_pass": passed >= int(CONFIG["stop_gate"]["datasets_required_to_pass"]),
        "results": results,
    }
    (local_root / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
