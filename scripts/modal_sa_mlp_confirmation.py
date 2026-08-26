"""Modal execution for the frozen six-dataset SA-MLP confirmation."""

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
    if (HOST_REPO_ROOT / "configs" / "sa_mlp_confirmation.yaml").exists()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "sa_mlp_confirmation.yaml"
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
    .add_local_file(str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/sa_mlp_confirmation.yaml")
    .add_local_file(
        str(SCREEN_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml"
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload["data"]["source_files"], sort_keys=True).encode("utf-8")
    ).hexdigest()


def _baseline_summary(payload: dict[str, Any], selected_gnn: str) -> dict[str, Any]:
    if payload.get("status") == "PAPER_MAIN_TABLE_DATASET_COMPLETE":
        observed = payload["selection_validation_only"]["selected"]
        plain_key = "plain_mlp"
        gnn_key = observed
        fingerprint = payload["data_fingerprint_sha256"]
    elif payload.get("status") == "CONFIRMATION_GATE_NOT_PAPER_FINAL":
        observed = payload["frozen_best_gnn"]
        plain_key = "plain_mlp_h64"
        gnn_key = f"{observed}_h64"
        fingerprint = _legacy_fingerprint(payload)
    else:
        raise ValueError("Unsupported or incomplete frozen baseline artifact")
    if observed != selected_gnn:
        raise ValueError("Frozen selected GNN differs from the confirmation protocol")

    def summarize(model_key: str, model_name: str) -> dict[str, Any]:
        model = payload["models"][model_key]
        return {
            "model": model_name,
            "parameters": model["parameters"],
            "seeds": {
                seed: {
                    "metrics": record["metrics"],
                    "inference": record["inference"],
                    "by_hop": record.get("by_hop", {}),
                }
                for seed, record in model["seeds"].items()
            },
        }

    return {
        "dataset": payload["dataset"],
        "queries": payload["data"]["queries"],
        "candidate_contract_sha256": payload["data"]["candidate_contract_sha256"],
        "data_fingerprint_sha256": fingerprint,
        "plain_mlp": summarize(plain_key, "plain_mlp"),
        "selected_gnn": summarize(gnn_key, selected_gnn),
    }


def _remote_paths(dataset: str, fingerprint: str) -> tuple[str, str]:
    root = (
        PurePosixPath(STORAGE_ROOT) / "outputs" / "sa_mlp_confirmation" / dataset / fingerprint[:16]
    )
    return str(root / "result.json"), str(root / "query_metrics.npz")


def _local_jobs(datasets: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_config = CONFIG["datasets"][dataset]
        baseline_path = HOST_REPO_ROOT / dataset_config["baseline"]
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline = _baseline_summary(payload, dataset_config["selected_gnn"])
        fingerprint = baseline["data_fingerprint_sha256"]
        output_remote, query_metrics_remote = _remote_paths(dataset, fingerprint)
        screen_local: Path | None = None
        screen_remote: str | None = None
        screen_sha256: str | None = None
        if "reuse_sa_seed_0" in dataset_config:
            screen_local = HOST_REPO_ROOT / dataset_config["reuse_sa_seed_0"]
            screen = json.loads(screen_local.read_text(encoding="utf-8"))
            if screen.get("status") != "SA_MLP_SCREEN_DATASET_COMPLETE":
                raise ValueError(f"Incomplete reusable screen artifact: {screen_local}")
            screen_remote = screen["config"]["output"]
            screen_sha256 = _sha256(screen_local)
        jobs.append(
            {
                "dataset": dataset,
                "expected_queries": int(dataset_config["expected_queries"]),
                "required_hops": list(dataset_config["required_hops"]),
                "selected_gnn": dataset_config["selected_gnn"],
                "data_remote": payload["config"]["data"],
                "baseline_remote": payload["config"]["output"],
                "baseline_result_sha256": _sha256(baseline_path),
                "baseline": baseline,
                "fingerprint": fingerprint,
                "screen_remote": screen_remote,
                "screen_result_sha256": screen_sha256,
                "output_remote": output_remote,
                "query_metrics_remote": query_metrics_remote,
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
        output=Path(job["output_remote"]),
        query_metrics_output=Path(job["query_metrics_remote"]),
        topology_cache=Path(job["data_remote"]) / "derived" / "packed_topology_v1",
        feature_cache=Path(job["data_remote"]) / "derived" / "fixed_structural_features_v1",
        baseline=job["baseline"],
        baseline_result_sha256=job["baseline_result_sha256"],
        data_fingerprint_sha256=job["fingerprint"],
        selected_gnn=job["selected_gnn"],
        screen_seed_0=(
            None
            if job["screen_remote"] is None
            else json.loads(Path(job["screen_remote"]).read_text(encoding="utf-8"))
        ),
        screen_result_sha256=job["screen_result_sha256"],
        feature_config={
            "retrieval_seeds": __import__("yaml").safe_load(
                Path(f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml").read_text(encoding="utf-8")
            )["retrieval_seeds"],
            "static_features": __import__("yaml").safe_load(
                Path(f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml").read_text(encoding="utf-8")
            )["static_features"],
            "query_local_features": __import__("yaml").safe_load(
                Path(f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml").read_text(encoding="utf-8")
            )["query_local_features"],
            "preprocessing": {"query_chunk_size": 8192},
        },
        required_hops=list(job["required_hops"]),
        seeds=list(training["seeds"]),
        projection_dim=int(parameters["sa_projection_dim"]),
        hidden_dim=int(parameters["seed_aware_gnn_hidden_dim"]),
        max_parameter_difference=int(parameters["maximum_sa_absolute_difference"]),
        layers=int(training["layers"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        dropout=float(training["dropout"]),
        temperature=float(training["temperature"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        ks=[1, 5, 20],
        inference_repeats=int(training["inference_repeats"]),
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
        raise ValueError("Remote frozen baseline failed its SHA-256 check")
    if job["screen_remote"] is not None:
        screen_path = Path(job["screen_remote"])
        if not screen_path.is_file() or _sha256(screen_path) != job["screen_result_sha256"]:
            raise ValueError("Remote reusable screen result failed its SHA-256 check")
    from scripts.run_sa_mlp_confirmation import run

    result = run(_runner_namespace(job), checkpoint_hook=result_volume.commit)
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "fingerprint": job["fingerprint"],
        "result": job["output_remote"],
        "query_metrics": job["query_metrics_remote"],
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
        raise ValueError(f"Unregistered confirmation datasets: {sorted(unknown)}")
    jobs = _local_jobs(requested)
    results = list(run_dataset.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} confirmation job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "sa_mlp_confirmation"
    for result in results:
        _download(result["result"], local_root / f"{result['dataset']}.json")
        _download(
            result["query_metrics"],
            local_root / f"{result['dataset']}.query_metrics.npz",
        )
    print(json.dumps(results, indent=2))
