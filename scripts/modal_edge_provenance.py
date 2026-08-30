"""Modal launcher for the frozen Package B edge-provenance comparison."""

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
    if (HOST_REPO_ROOT / "configs" / "edge_provenance.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "edge_provenance.yaml"
SCREEN_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "sa_mlp_screen.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]
STORAGE_ROOT = MODAL_CONFIG["storage_root"]
SOURCE_ROOT = MODAL_CONFIG["source_root"]

app = modal.App(MODAL_CONFIG["app"])
source_volume = modal.Volume.from_name(MODAL_CONFIG["source_volume"], create_if_missing=False)
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
    .add_local_file(str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/edge_provenance.yaml")
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


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    jobs = []
    for dataset in datasets:
        settings = CONFIG["datasets"][dataset]
        confirmation = json.loads(
            (HOST_REPO_ROOT / settings["confirmation"]).read_text(encoding="utf-8")
        )
        fingerprint = confirmation["data_fingerprint_sha256"]
        data_remote = confirmation["config"]["data"]
        graph_root = (
            PurePosixPath(STORAGE_ROOT)
            / "edge_provenance_graphs"
            / dataset
            / fingerprint[:16]
        )
        jobs.append(
            {
                "dataset": dataset,
                "settings": settings,
                "confirmation": confirmation,
                "fingerprint": fingerprint,
                "data_remote": data_remote,
                "graph_root": str(graph_root),
            }
        )
    return jobs


@app.function(
    image=image,
    volumes={SOURCE_ROOT: source_volume, STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def prepare_dataset(job: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct all edge sidecars using only read operations on CRAG."""

    os.chdir(REMOTE_ROOT)
    source_volume.reload()
    result_volume.reload()
    settings = job["settings"]
    source_data = Path(SOURCE_ROOT) / "data"
    master = source_data / "processed" / settings["master_file"]
    baseline_graph = source_data / "ukb_storage" / job["dataset"] / "gte_qwen" / "graph.pt"
    ner = source_data / "ukb_storage" / job["dataset"] / "ner_edges_w_df25.pkl"
    if _sha256(baseline_graph) != settings["source_graph_sha256"]:
        raise ValueError("CRAG source graph differs from the preregistered SHA-256")
    output_root = Path(job["graph_root"])
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = [
            output_root / family / "graph.pt"
            for family, family_config in CONFIG["graph_families"].items()
            if family_config["execution"] != "reuse_sealed_sa_mlp_confirmation"
        ]
        if (
            manifest.get("format") == "edge_provenance_families_v1"
            and manifest.get("dataset") == job["dataset"]
            and all(path.is_file() for path in required)
        ):
            return {
                "dataset": job["dataset"],
                "status": "EDGE_PROVENANCE_SIDECARS_REUSED",
                "graph_root": job["graph_root"],
                "counts": manifest["undirected_edge_counts"],
            }
    from mp_retrieval.edge_provenance import reconstruct_edge_families, save_edge_families

    identity_path = Path(job["data_remote"]) / "node_ids.json"
    families, metadata = reconstruct_edge_families(
        dataset=job["dataset"],
        master_path=master,
        baseline_graph_path=baseline_graph,
        ner_path=ner,
        expected_node_ids_path=identity_path if identity_path.is_file() else None,
    )
    save_edge_families(families, metadata, output_root)
    result_volume.commit()
    return {
        "dataset": job["dataset"],
        "status": "EDGE_PROVENANCE_SIDECARS_CREATED",
        "graph_root": job["graph_root"],
        "counts": metadata["undirected_edge_counts"],
        "alignment": metadata["node_alignment"],
    }


def _runner_args(job: dict[str, Any], family: str) -> argparse.Namespace:
    settings = job["settings"]
    training = CONFIG["training"]
    parameters = CONFIG["parameter_regime"]
    family_root = PurePosixPath(job["graph_root"]) / family
    cache_root = (
        PurePosixPath(STORAGE_ROOT)
        / "edge_provenance_cache"
        / job["dataset"]
        / job["fingerprint"][:16]
        / family
    )
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "edge_provenance"
        / job["dataset"]
        / job["fingerprint"][:16]
        / family
    )
    screen = yaml.safe_load(
        Path(f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml").read_text(encoding="utf-8")
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        family=family,
        family_graph=Path(family_root) / "graph.pt",
        family_metadata=Path(family_root) / "metadata.json",
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
def run_family(job: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    result_volume.reload()
    from scripts.run_edge_provenance import run

    args = _runner_args(job, job["family"])
    result = run(args, checkpoint_hook=result_volume.commit)
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "family": job["family"],
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
        raise ValueError(f"Unregistered edge-provenance datasets: {sorted(unknown)}")
    jobs = _jobs(requested)
    prepared = list(
        prepare_dataset.map(jobs, return_exceptions=True, wrap_returned_exceptions=False)
    )
    failures = [item for item in prepared if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} edge reconstruction job(s) failed: {failures}")
    print(json.dumps({"edge_reconstruction": prepared}, indent=2), flush=True)
    family_jobs = [
        {**job, "family": family}
        for job in jobs
        for family in CONFIG["trained_families"]
    ]
    results = list(
        run_family.map(family_jobs, return_exceptions=True, wrap_returned_exceptions=False)
    )
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} edge-family job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "edge_provenance"
    for result in results:
        family_root = local_root / result["dataset"] / result["family"]
        _download(result["output_remote"], family_root / "result.json")
        _download(result["query_metrics_remote"], family_root / "query_metrics.npz")
    print(json.dumps(results, indent=2))
