"""Modal A10G runner for the frozen five-seed confirmation gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import modal
import yaml


REMOTE_ROOT = "/root/message-passing-retrieval"
STORAGE_ROOT = f"{REMOTE_ROOT}/storage"
HOST_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REPO_ROOT = (
    HOST_REPO_ROOT
    if (HOST_REPO_ROOT / "configs" / "confirmation.yaml").exists()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "confirmation.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]
SOURCE_FILES = (
    "nodes.npy",
    "queries_all.npy",
    "dense_top200_all.npy",
    "splade_top200_all.npy",
    "query_ids_all.json",
    "graph.pt",
)

app = modal.App(MODAL_CONFIG["app"])
volume = modal.Volume.from_name(MODAL_CONFIG["volume"], create_if_missing=True)
image = (
    modal.Image.micromamba(python_version="3.11")
    .env(
        {
            "CONDA_OVERRIDE_CUDA": "12.1",
            "CUDA_HOME": "/opt/conda",
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "PYTHONPATH": f"{REMOTE_ROOT}/src",
        }
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("torch==2.2.1", "numpy<2.0", "pyyaml==6.0.2")
    .pip_install(
        "torch-geometric==2.5.2",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html",
    )
    .add_local_dir(str(RUNTIME_REPO_ROOT / "src"), remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir(str(RUNTIME_REPO_ROOT / "scripts"), remote_path=f"{REMOTE_ROOT}/scripts")
    .add_local_file(
        str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/confirmation.yaml"
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runner_args(job: dict[str, Any]) -> list[str]:
    training = CONFIG["training"]
    selection = CONFIG["selection"]
    return [
        "--data", job["data_remote"],
        "--dataset", job["dataset"],
        "--expected-queries", str(job["expected_queries"]),
        "--best-gnn", job["best_gnn"],
        "--output", job["result_remote"],
        "--seeds", *[str(seed) for seed in CONFIG["seeds"]],
        "--hidden-widths", *[str(width) for width in CONFIG["hidden_widths"]],
        "--epochs", str(training["epochs"]),
        "--batch-size", str(training["batch_size"]),
        "--layers", str(training["layers"]),
        "--offset-directions", str(training["offset_directions"]),
        "--dropout", str(training["dropout"]),
        "--temperature", str(training["temperature"]),
        "--learning-rate", str(training["learning_rate"]),
        "--weight-decay", str(training["weight_decay"]),
        "--ks", *[str(k) for k in training["ks"]],
        "--inference-repeats", str(training["inference_repeats"]),
        "--capacity-tie-margin-percentage-points",
        str(selection["tie_margin_percentage_points"]),
        "--device", "cuda",
        "--nodes-sha256", job["file_sha256"]["nodes.npy"],
    ]


@app.function(
    image=image,
    gpu=MODAL_CONFIG["gpu"],
    volumes={STORAGE_ROOT: volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    memory=32768,
)
def run_dataset(job: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    import torch

    print(
        json.dumps(
            {
                "event": "confirmation_start",
                "dataset": job["dataset"],
                "device": torch.cuda.get_device_name(0),
                "status": CONFIG["status"],
            }
        ),
        flush=True,
    )
    Path(job["result_remote"]).parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [sys.executable, "scripts/run_confirmation.py", *_runner_args(job)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(process.stdout, flush=True)
    if process.returncode:
        raise RuntimeError(f"confirmation failed for {job['dataset']} with {process.returncode}")
    volume.commit()
    manifest = {
        "status": CONFIG["status"],
        "dataset": job["dataset"],
        "gpu": MODAL_CONFIG["gpu"],
        "data_fingerprint": job["data_fingerprint"],
        "result": job["result_remote"],
    }
    print(json.dumps({"event": "confirmation_complete", **manifest}), flush=True)
    return manifest


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative = remote_path.removeprefix(f"{STORAGE_ROOT}/")
    with local_path.open("wb") as stream:
        for chunk in volume.read_file(relative):
            stream.write(chunk)


@app.local_entrypoint()
def main(datasets: str = "2wiki_clean,musique_clean") -> None:
    requested = datasets.split(",")
    unknown = set(requested) - set(CONFIG["datasets"])
    if unknown:
        raise ValueError(f"Unregistered confirmation datasets: {sorted(unknown)}")
    source_root = Path(
        os.environ.get(
            "MP_RETRIEVAL_SOURCE_ROOT",
            str(HOST_REPO_ROOT.parent / "CRAG" / "data" / "ukb_storage"),
        )
    )
    jobs: list[dict[str, Any]] = []
    uploads: list[tuple[Path, str]] = []
    for dataset in requested:
        dataset_config = CONFIG["datasets"][dataset]
        source = source_root / dataset_config["source_subdir"]
        missing = [name for name in SOURCE_FILES if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{dataset} is missing complete source files: {missing}")
        file_sha256 = {name: _sha256(source / name) for name in SOURCE_FILES}
        fingerprint = hashlib.sha256(
            json.dumps(file_sha256, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        data_relative = f"operator_data/{dataset}/{fingerprint}"
        try:
            existing = {entry.path for entry in volume.listdir(data_relative)}
        except Exception:
            existing = set()
        for name in SOURCE_FILES:
            remote_relative = f"{data_relative}/{name}"
            if name not in existing and remote_relative not in existing:
                uploads.append((source / name, remote_relative))
        result_remote = (
            f"{STORAGE_ROOT}/outputs/confirmation/{dataset}/{fingerprint}/result.json"
        )
        jobs.append(
            {
                "dataset": dataset,
                "expected_queries": dataset_config["expected_queries"],
                "best_gnn": dataset_config["frozen_best_gnn"],
                "data_fingerprint": fingerprint,
                "file_sha256": file_sha256,
                "data_remote": f"{STORAGE_ROOT}/{data_relative}",
                "result_remote": result_remote,
            }
        )
    if uploads:
        with volume.batch_upload(force=True) as upload:
            for local_path, remote_relative in uploads:
                upload.put_file(str(local_path), remote_relative)
    results = list(
        run_dataset.map(jobs, return_exceptions=True, wrap_returned_exceptions=False)
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} confirmation job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "confirmation"
    for result in results:
        _download(result["result"], local_root / f"{result['dataset']}.json")
    print(json.dumps(results, indent=2))
