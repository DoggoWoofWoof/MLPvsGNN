"""Modal A10G runner for the restricted three-dataset pilot."""

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
    if (HOST_REPO_ROOT / "configs" / "pilot3.yaml").exists()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "pilot3.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]

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
    .add_local_file(str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/pilot3.yaml")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _comparison_args(job: dict[str, Any]) -> list[str]:
    comparison = job["comparison"]
    intervention = job["intervention"]
    args = [
        "--data",
        job["artifact_remote"],
        "--output",
        job["result_remote"],
        "--gnn",
        comparison["gnn"],
        "--match-modes",
        *comparison["match_modes"],
        "--seeds",
        *[str(seed) for seed in comparison["seeds"]],
        "--epochs",
        str(comparison["epochs"]),
        "--layers",
        str(comparison["layers"]),
        "--hidden-dim",
        str(comparison["hidden_dim"]),
        "--dropout",
        str(comparison["dropout"]),
        "--lr",
        str(comparison["learning_rate"]),
        "--ks",
        *[str(k) for k in comparison["ks"]],
        "--split-seed",
        str(comparison["split_seed"]),
        "--parameter-tolerance",
        str(comparison["parameter_tolerance"]),
        "--compute-tolerance",
        str(comparison["compute_tolerance"]),
        "--compute-calibration-queries",
        str(comparison["compute_calibration_queries"]),
        "--compute-warmups",
        str(comparison["compute_warmups"]),
        "--compute-repeats",
        str(comparison["compute_repeats"]),
        "--perturbation",
        intervention["kind"],
        "--perturbation-rate",
        str(intervention["rate"]),
        "--perturbation-seed",
        str(intervention["seed"]),
        "--allow-pilot-resplit",
        "--device",
        "cuda",
    ]
    if comparison.get("query_limit") is not None:
        args.extend(["--max-queries", str(comparison["query_limit"])])
    return args


@app.function(
    image=image,
    gpu=MODAL_CONFIG["gpu"],
    volumes={STORAGE_ROOT: volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
)
def run_dataset(job: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    import torch

    print(
        json.dumps(
            {
                "event": "dataset_start",
                "dataset": job["dataset"],
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "status": "NOT_PAPER_VALID_PILOT",
            }
        ),
        flush=True,
    )
    Path(job["result_remote"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["stats_remote"]).parent.mkdir(parents=True, exist_ok=True)
    started = subprocess.run(
        [sys.executable, "scripts/run_l2_pair.py", *_comparison_args(job)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(started.stdout, flush=True)
    if started.returncode:
        raise RuntimeError(f"paired run failed for {job['dataset']} with {started.returncode}")
    stats = subprocess.run(
        [
            sys.executable,
            "scripts/collect_l2_stats.py",
            "--data",
            job["artifact_remote"],
            "--output",
            job["stats_remote"],
            "--allow-pilot-resplit",
            "--split-seed",
            str(job["comparison"]["split_seed"]),
            "--perturbation",
            job["intervention"]["kind"],
            "--perturbation-rate",
            str(job["intervention"]["rate"]),
            "--perturbation-seed",
            str(job["intervention"]["seed"]),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(stats.stdout, flush=True)
    if stats.returncode:
        raise RuntimeError(f"statistics run failed for {job['dataset']} with {stats.returncode}")
    manifest = {
        "status": "NOT_PAPER_VALID_PILOT",
        "dataset": job["dataset"],
        "protocol_baseline": job["protocol_baseline"],
        "artifact_sha256": job["artifact_sha256"],
        "gpu": MODAL_CONFIG["gpu"],
        "result": job["result_remote"],
        "stats": job["stats_remote"],
    }
    Path(job["manifest_remote"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps({"event": "dataset_complete", **manifest}), flush=True)
    return manifest


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as stream:
        for chunk in volume.read_file(remote_path.removeprefix(f"{STORAGE_ROOT}/")):
            stream.write(chunk)


def _registered_intervention(kind: str, rate: float) -> dict[str, Any]:
    if kind == "clean":
        if rate != 0.0:
            raise ValueError("clean intervention requires rate 0")
        return dict(CONFIG["active_intervention"])
    registered = {
        item["kind"]: item.get("rates", [])
        for item in CONFIG["ordered_interventions_after_clean_gate"]
    }
    if kind not in registered or rate not in registered[kind]:
        raise ValueError(f"Intervention {kind}@{rate} is not pre-registered in pilot3.yaml")
    return {"kind": kind, "rate": rate, "seed": CONFIG["active_intervention"]["seed"]}


@app.local_entrypoint()
def main(
    datasets: str = "webqsp,2wiki_clean,musique_clean",
    intervention: str = "clean",
    rate: float = 0.0,
) -> None:
    requested = datasets.split(",")
    allowed = set(CONFIG["datasets"])
    unknown = set(requested) - allowed
    if unknown:
        raise ValueError(f"Only the registered pilot datasets are allowed: {sorted(unknown)}")
    active_intervention = _registered_intervention(intervention, rate)
    jobs = []
    try:
        existing = {entry.path for entry in volume.listdir("data/processed")}
    except Exception:
        existing = set()
    with volume.batch_upload(force=True) as upload:
        for dataset in requested:
            local_artifact = HOST_REPO_ROOT / CONFIG["datasets"][dataset]["artifact"]
            if not local_artifact.exists():
                raise FileNotFoundError(local_artifact)
            digest = _sha256(local_artifact)
            remote_relative = f"data/processed/{digest[:16]}_{local_artifact.name}"
            if remote_relative not in existing:
                upload.put_file(str(local_artifact), remote_relative)
            intervention_slug = f"{intervention}_{rate:.2f}".replace(".", "p")
            output_root = f"{STORAGE_ROOT}/outputs/pilot3_{intervention_slug}/{digest[:16]}"
            jobs.append(
                {
                    "dataset": dataset,
                    "protocol_baseline": CONFIG["protocol_baseline"],
                    "artifact_sha256": digest,
                    "artifact_remote": f"{STORAGE_ROOT}/{remote_relative}",
                    "result_remote": f"{output_root}/{dataset}_pair.json",
                    "stats_remote": f"{output_root}/{dataset}_stats.json",
                    "manifest_remote": f"{output_root}/{dataset}_manifest.json",
                    "comparison": CONFIG["comparison"],
                    "intervention": active_intervention,
                }
            )
    results = list(
        run_dataset.map(
            jobs,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        )
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} Modal dataset run(s) failed: {failures}")
    for result in results:
        dataset = result["dataset"]
        intervention_slug = f"{intervention}_{rate:.2f}".replace(".", "p")
        local_root = HOST_REPO_ROOT / "outputs" / f"pilot3_{intervention_slug}"
        _download(result["result"], local_root / f"{dataset}_pair.json")
        _download(result["stats"], local_root / f"{dataset}_stats.json")
    print(json.dumps(results, indent=2))
