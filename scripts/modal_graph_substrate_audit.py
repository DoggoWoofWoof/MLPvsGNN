"""Modal launcher for the read-only Phase -1 graph-substrate audit.

CPU only. It shares the frozen data volume with the GPU packages but writes to
its own output prefix and never mutates a candidate pool, a graph, or a frozen
result, so it is safe to run beside an in-flight training job.

Submit it through ``scripts/spawn_modal_jobs.py graph-substrate`` so the calls
are server-side and survive client teardown.
"""

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
    if (HOST_REPO_ROOT / "configs" / "graph_substrate_audit.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "graph_substrate_audit.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]

# What a restart has to redo from the beginning, which is what the launch gate
# compares against the timeout. The runner adopts completed graph families from
# a partial audit (`resume_partial=True` below), so a killed run resumes at a
# family boundary and the unit is one family rather than the whole audit.
# Deleting that resumption without changing this would make the gate admit a
# job whose real unit is four times larger.
RESUME_GRANULARITY = "family"
STORAGE_ROOT = MODAL_CONFIG["storage_root"]
DATASET_GRAPH = "dataset_default"

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
    .add_local_file(
        str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/graph_substrate_audit.yaml"
    )
)


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    jobs = []
    for dataset in datasets:
        settings = CONFIG["datasets"][dataset]
        confirmation = json.loads(
            (HOST_REPO_ROOT / settings["confirmation"]).read_text(encoding="utf-8")
        )
        fingerprint = confirmation["data_fingerprint_sha256"]
        jobs.append(
            {
                "dataset": dataset,
                "settings": settings,
                "baseline": confirmation["baseline"],
                "fingerprint": fingerprint,
                "data_remote": confirmation["config"]["data"],
                # Package B already reconstructed the provenance families in the
                # frozen node coordinate system; the audit reads them, it does
                # not rebuild them.
                "edge_families": str(
                    PurePosixPath(STORAGE_ROOT)
                    / "edge_provenance_graphs"
                    / dataset
                    / fingerprint[:16]
                ),
            }
        )
    return jobs


def _runner_args(job: dict[str, Any]) -> argparse.Namespace:
    settings = job["settings"]
    reporting = CONFIG["reporting"]
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "graph_substrate_audit"
        / job["dataset"]
        / job["fingerprint"][:16]
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        expected_queries=int(settings["expected_queries"]),
        baseline=job["baseline"],
        candidate_contract_compatibility=settings.get("candidate_contract_compatibility"),
        data_fingerprint_sha256=job["fingerprint"],
        graphs=list(CONFIG["graphs"]),
        splits=list(reporting["splits"]),
        edge_families=job["edge_families"],
        max_hops=int(reporting["max_hops"]),
        pooled_query_cap=int(CONFIG["aggregation"]["pooled_query_cap"]),
        expansion_query_cap=int(CONFIG["aggregation"]["expansion_query_cap"]),
        # The message-flow statistics depend on which operator consumes the
        # edges, so the audit uses each dataset's frozen validation-selected
        # family rather than a single global default.
        operator_kind=str(settings["selected_gnn"]),
        output=Path(output_root) / "substrate.json",
        # The audit commits after every family/split. Without resumption a run
        # that hits the timeout throws away everything it measured and the next
        # attempt starts from the same place; only families that are themselves
        # finished are carried, and only under an identical contract.
        resume_partial=True,
    )


@app.function(
    image=image,
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_substrate(job: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    from scripts.run_graph_substrate_audit import run

    args = _runner_args(job)
    result = run(args, checkpoint_hook=result_volume.commit)
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "output_remote": str(args.output),
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
    """Blocking fallback. Prefer scripts/spawn_modal_jobs.py for a real submission."""

    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    unknown = set(requested) - set(CONFIG["datasets"])
    if unknown:
        raise ValueError(f"Unregistered graph-substrate datasets: {sorted(unknown)}")
    jobs = _jobs(requested)
    results = list(
        run_substrate.map(jobs, return_exceptions=True, wrap_returned_exceptions=False)
    )
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} graph-substrate job(s) failed: {failures}")
    local_root = HOST_REPO_ROOT / "outputs" / "graph_substrate_audit"
    for result in results:
        _download(result["output_remote"], local_root / f"{result['dataset']}.json")
    print(json.dumps(results, indent=2))
