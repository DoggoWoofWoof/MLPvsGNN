"""Modal staging and A10G execution for the frozen six-dataset paper table."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any

import modal
import yaml


REMOTE_ROOT = "/root/message-passing-retrieval"
HOST_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REPO_ROOT = (
    HOST_REPO_ROOT
    if (HOST_REPO_ROOT / "configs" / "six_dataset_study.yaml").exists()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "six_dataset_study.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]
SOURCE_MOUNT = "/root/crag-source"
SOURCE_ROOT = MODAL_CONFIG["source_root"]
STORAGE_ROOT = MODAL_CONFIG["storage_root"]
CORE_FILES = (
    "nodes.npy",
    "queries_all.npy",
    "dense_top200_all.npy",
    "splade_top200_all.npy",
    "query_ids_all.json",
    "graph.pt",
)

app = modal.App(MODAL_CONFIG["app"])
source_volume = modal.Volume.from_name(MODAL_CONFIG["source_volume"], create_if_missing=False)
result_volume = modal.Volume.from_name(MODAL_CONFIG["result_volume"], create_if_missing=True)
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
    .add_local_file(
        str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/six_dataset_study.yaml"
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    with source.open("rb") as incoming, temporary.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=16 * 1024 * 1024)
    temporary.replace(target)


def _metaqa_node_ids(source: Path, dataset_config: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    identity_source = Path(SOURCE_ROOT) / dataset_config["node_identity_source"]
    with identity_source.open("rb") as stream:
        payload = pickle.load(stream)
    mapping = payload["id_to_idx"]
    ordered = [None] * len(mapping)
    for node_id, row in mapping.items():
        row = int(row)
        if row < 0 or row >= len(ordered) or ordered[row] is not None:
            raise ValueError("MetaQA SPLADE node identity mapping is not bijective")
        ordered[row] = str(node_id)
    if any(node_id is None for node_id in ordered):
        raise ValueError("MetaQA SPLADE node identity mapping has missing rows")
    node_array = __import__("numpy").load(source / "nodes.npy", mmap_mode="r")
    if len(ordered) != int(node_array.shape[0]):
        raise ValueError("MetaQA node identity count does not match nodes.npy")
    encoded = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return encoded, {
        "source": str(identity_source.relative_to(Path(SOURCE_ROOT))),
        "source_sha256": _sha256(identity_source),
        "rows": len(ordered),
        "derivation": "row_order_from_frozen_splade_id_to_idx",
    }


@app.function(
    image=image,
    volumes={SOURCE_MOUNT: source_volume, STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def stage_dataset(job: dict[str, Any]) -> dict[str, Any]:
    dataset = job["dataset"]
    dataset_config = CONFIG["new_datasets"][dataset]
    source = Path(SOURCE_ROOT) / dataset_config["source_subdir"]
    missing = [name for name in CORE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{dataset} source volume is missing {missing}")
    file_records: dict[str, dict[str, Any]] = {}
    for name in CORE_FILES:
        path = source / name
        print(json.dumps({"event": "fingerprint", "dataset": dataset, "file": name}), flush=True)
        file_records[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    node_ids_bytes: bytes | None = None
    identity_provenance: dict[str, Any] | None = None
    if "node_identity_source" in dataset_config:
        node_ids_bytes, identity_provenance = _metaqa_node_ids(source, dataset_config)
        file_records["node_ids.json"] = {
            "bytes": len(node_ids_bytes),
            "sha256": hashlib.sha256(node_ids_bytes).hexdigest(),
        }
    fingerprint = hashlib.sha256(
        json.dumps(file_records, sort_keys=True).encode("utf-8")
    ).hexdigest()
    destination = Path(STORAGE_ROOT) / "paper_data" / dataset / fingerprint[:16]
    destination.mkdir(parents=True, exist_ok=True)
    for name in CORE_FILES:
        target = destination / name
        expected = int(file_records[name]["bytes"])
        if not target.is_file() or target.stat().st_size != expected:
            print(json.dumps({"event": "copy", "dataset": dataset, "file": name}), flush=True)
            _copy(source / name, target)
    if node_ids_bytes is not None:
        target = destination / "node_ids.json"
        if not target.is_file() or target.stat().st_size != len(node_ids_bytes):
            target.write_bytes(node_ids_bytes)
    manifest = {
        "status": "FROZEN_SOURCE_STAGED_READ_ONLY_FROM_CRAG_VOLUME",
        "dataset": dataset,
        "data_fingerprint_sha256": fingerprint,
        "files": file_records,
        "node_identity_provenance": identity_provenance,
    }
    (destination / "_frozen_source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    result_volume.commit()
    return {
        "dataset": dataset,
        "expected_queries": dataset_config["expected_queries"],
        "required_hops": dataset_config.get("required_hops", []),
        "fingerprint": fingerprint,
        "data_remote": str(destination),
        "result_remote": str(
            Path(STORAGE_ROOT)
            / "outputs"
            / "main_table"
            / dataset
            / fingerprint[:16]
            / "result.json"
        ),
    }


def _runner_namespace(job: dict[str, Any]) -> argparse.Namespace:
    training = CONFIG["training"]
    confirmation = CONFIG["confirmation"]
    selection = CONFIG["gnn_selection"]
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        expected_queries=int(job["expected_queries"]),
        output=Path(job["result_remote"]),
        topology_cache=Path(job["data_remote"]) / "derived" / "packed_topology_v1",
        required_hops=list(job["required_hops"]),
        selection_seed=int(selection["seed"]),
        seeds=list(confirmation["seeds"]),
        hidden_dim=int(training["hidden_dim"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        layers=int(training["layers"]),
        dropout=float(training["dropout"]),
        temperature=float(training["temperature"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        ks=list(training["ks"]),
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
    import torch
    from scripts.run_main_table import run

    print(
        json.dumps(
            {
                "event": "paper_main_start",
                "dataset": job["dataset"],
                "gpu": torch.cuda.get_device_name(0),
                "fingerprint": job["fingerprint"],
            }
        ),
        flush=True,
    )
    result = run(_runner_namespace(job), checkpoint_hook=result_volume.commit)
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "fingerprint": job["fingerprint"],
        "result": job["result_remote"],
        "selected_gnn": result["selection_validation_only"]["selected"],
    }


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative = remote_path.removeprefix(f"{STORAGE_ROOT}/")
    with local_path.open("wb") as stream:
        for chunk in result_volume.read_file(relative):
            stream.write(chunk)


@app.local_entrypoint()
def main(datasets: str = "webqsp,hotpotqa_clean,squad_clean,metaqa") -> None:
    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    unknown = set(requested) - set(CONFIG["new_datasets"])
    if unknown:
        raise ValueError(f"Unregistered paper datasets: {sorted(unknown)}")
    staged = list(
        stage_dataset.map(
            [{"dataset": dataset} for dataset in requested],
            return_exceptions=True,
            wrap_returned_exceptions=False,
        )
    )
    failures = [result for result in staged if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} dataset staging job(s) failed: {failures}")
    results = list(
        run_dataset.map(staged, return_exceptions=True, wrap_returned_exceptions=False)
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} paper job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "main_table"
    for result in results:
        _download(result["result"], local_root / f"{result['dataset']}.json")
    print(json.dumps(results, indent=2))
