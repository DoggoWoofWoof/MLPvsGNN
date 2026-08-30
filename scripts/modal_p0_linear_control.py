"""Run the frozen P0 A3 linear rank+structure control on Modal."""

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
    if (HOST_REPO_ROOT / "configs" / "p0_linear_rank_structure.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "p0_linear_rank_structure.yaml"
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
            "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT}/src",
        }
    )
    .pip_install(
        "torch==2.2.1",
        "numpy<2.0",
        "psutil==6.1.1",
        "pyyaml==6.0.2",
    )
    .add_local_dir(str(HOST_REPO_ROOT / "src"), remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir(str(HOST_REPO_ROOT / "scripts"), remote_path=f"{REMOTE_ROOT}/scripts")
    .add_local_file(
        str(CONFIG_PATH),
        remote_path=f"{REMOTE_ROOT}/configs/p0_linear_rank_structure.yaml",
    )
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _local_jobs(datasets: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for dataset in datasets:
        confirmation = _load(HOST_REPO_ROOT / "outputs" / "sa_mlp_confirmation" / f"{dataset}.json")
        rank_result = _load(HOST_REPO_ROOT / "outputs" / "p0_rank_controls" / f"{dataset}.json")
        structural_result = _load(
            HOST_REPO_ROOT / "outputs" / "p0_fixed_structural_controls" / f"{dataset}.json"
        )
        data_remote = str(confirmation["config"]["data"])
        feature_remote = str(confirmation["config"]["feature_cache"])
        if not data_remote.startswith(STORAGE_ROOT) or not feature_remote.startswith(STORAGE_ROOT):
            raise ValueError(f"{dataset} confirmation does not reference the expected Modal volume")
        fingerprint = confirmation["data_fingerprint_sha256"]
        output_root = (
            PurePosixPath(STORAGE_ROOT)
            / "outputs"
            / "p0_linear_rank_structure"
            / dataset
            / fingerprint[:16]
        )
        derived_remote = str(
            PurePosixPath(feature_remote).parent / "linear_rank_structure_inputs_v1"
        )
        jobs.append(
            {
                "dataset": dataset,
                "data_remote": data_remote,
                "feature_remote": feature_remote,
                "derived_remote": derived_remote,
                "output_remote": str(output_root / "result.json"),
                "query_metrics_remote": str(output_root / "query_metrics.npz"),
                "rank_result": rank_result,
                "structural_result": structural_result,
                "confirmation": confirmation,
            }
        )
    return jobs


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
    data = Path(job["data_remote"])
    features = Path(job["feature_remote"])
    if not data.is_dir() or not features.is_dir():
        raise FileNotFoundError(f"Sealed Modal inputs are absent for {job['dataset']}")
    from scripts.run_linear_rank_structure import run

    args = argparse.Namespace(
        config=Path(REMOTE_ROOT) / "configs" / "p0_linear_rank_structure.yaml",
        dataset=job["dataset"],
        data=data,
        feature_cache=features,
        derived_cache=Path(job["derived_remote"]),
        rank_result=None,
        structural_result=None,
        confirmation=None,
        output=Path(job["output_remote"]),
        query_metrics_output=Path(job["query_metrics_remote"]),
        device="cuda",
    )
    result = run(
        args,
        rank_result_payload=job["rank_result"],
        structural_result_payload=job["structural_result"],
        confirmation_payload=job["confirmation"],
        checkpoint_hook=result_volume.commit,
    )
    result_volume.commit()
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "selected_learning_rate": result["selected_learning_rate"],
        "test_R@5": result["aggregate"]["test_metrics"]["recall@5"],
        "output_remote": job["output_remote"],
        "query_metrics_remote": job["query_metrics_remote"],
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
    unknown = set(requested) - set(CONFIG["frozen_inputs"]["datasets"])
    if unknown:
        raise ValueError(f"Unregistered A3 datasets: {sorted(unknown)}")
    jobs = _local_jobs(requested)
    results = list(run_dataset.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} A3 job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "p0_linear_rank_structure"
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}.json")
        _download(
            result["query_metrics_remote"],
            local_root / f"{result['dataset']}.query_metrics.npz",
        )
    print(json.dumps(results, indent=2))
