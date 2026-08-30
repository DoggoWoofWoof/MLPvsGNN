"""Run P0 A2 against the already sealed QLS caches on Modal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import modal
import yaml

REMOTE_ROOT = "/root/message-passing-retrieval"
HOST_REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = HOST_REPO_ROOT / "configs" / "p0_fixed_structural_controls.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]
STORAGE_ROOT = MODAL_CONFIG["storage_root"]

app = modal.App(MODAL_CONFIG["app"])
result_volume = modal.Volume.from_name(MODAL_CONFIG["result_volume"], create_if_missing=False)
image = (
    modal.Image.micromamba(python_version="3.11")
    .env({"PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT}/src"})
    .pip_install("numpy<2.0", "pyyaml==6.0.2")
    .add_local_dir(str(HOST_REPO_ROOT / "src"), remote_path=f"{REMOTE_ROOT}/src")
    .add_local_file(
        str(HOST_REPO_ROOT / "scripts" / "run_fixed_structural_controls.py"),
        remote_path=f"{REMOTE_ROOT}/scripts/run_fixed_structural_controls.py",
    )
    .add_local_file(
        str(CONFIG_PATH),
        remote_path=f"{REMOTE_ROOT}/configs/p0_fixed_structural_controls.yaml",
    )
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _local_jobs(datasets: list[str]) -> list[dict[str, Any]]:
    jobs = []
    for dataset in datasets:
        confirmation = _load(HOST_REPO_ROOT / "outputs" / "sa_mlp_confirmation" / f"{dataset}.json")
        rank_result = _load(HOST_REPO_ROOT / "outputs" / "p0_rank_controls" / f"{dataset}.json")
        data_remote = str(confirmation["config"]["data"])
        feature_remote = str(confirmation["config"]["feature_cache"])
        if not data_remote.startswith(STORAGE_ROOT) or not feature_remote.startswith(STORAGE_ROOT):
            raise ValueError(f"{dataset} confirmation does not reference the expected Modal volume")
        fingerprint = confirmation["data_fingerprint_sha256"]
        output_root = (
            Path(STORAGE_ROOT)
            / "outputs"
            / "p0_fixed_structural_controls"
            / dataset
            / fingerprint[:16]
        )
        jobs.append(
            {
                "dataset": dataset,
                "data_remote": data_remote,
                "feature_remote": feature_remote,
                "output_remote": str(output_root / "result.json"),
                "query_metrics_remote": str(output_root / "query_metrics.npz"),
                "confirmation": confirmation,
                "rank_result": rank_result,
            }
        )
    return jobs


@app.function(
    image=image,
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
    from scripts.run_fixed_structural_controls import run

    args = argparse.Namespace(
        config=Path(REMOTE_ROOT) / "configs" / "p0_fixed_structural_controls.yaml",
        dataset=job["dataset"],
        data=data,
        feature_cache=features,
        rank_result=None,
        confirmation=None,
        output=Path(job["output_remote"]),
        query_metrics_output=Path(job["query_metrics_remote"]),
    )
    result = run(
        args,
        rank_result_payload=job["rank_result"],
        confirmation_payload=job["confirmation"],
    )
    result_volume.commit()
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "output_remote": job["output_remote"],
        "query_metrics_remote": job["query_metrics_remote"],
        "test_R@5": {method: values["recall@5"] for method, values in result["test"].items()},
        "seconds": result["timing"]["total_seconds"],
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
        raise ValueError(f"Unregistered A2 datasets: {sorted(unknown)}")
    jobs = _local_jobs(requested)
    results = list(run_dataset.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} A2 job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "p0_fixed_structural_controls"
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}.json")
        _download(
            result["query_metrics_remote"],
            local_root / f"{result['dataset']}.query_metrics.npz",
        )
    print(json.dumps(results, indent=2))
