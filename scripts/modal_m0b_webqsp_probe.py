"""Modal launcher for M0B Safeguard C: the webqsp-only systems-calibration probe.

CPU only, zero GPU hours authorised. Reuses the frozen candidate pool, the
Package B edge-provenance graphs, and the sa_mlp confirmation's baseline
contract; writes to its own output prefix. Submit through
scripts/spawn_modal_jobs.py so the run is server-side; a detached local run is
not a registered execution.

webqsp only, by Safeguard C's own mandate -- this launcher does not take a
datasets argument, unlike modal_m0a1_overlap.py, because generalising it would
misrepresent what this safeguard is: a calibration probe for the one dataset
flagged as the density risk, not a six-dataset runner.
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
    if (HOST_REPO_ROOT / "configs" / "m0b_regime_map.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m0b_regime_map.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]
STORAGE_ROOT = MODAL_CONFIG["storage_root"]
DATASET = "webqsp"
# Safeguard C's own filed range is 10-20; 15 is this launcher's fixed choice
# within it, not a runtime knob -- spawn_modal_jobs.py's dispatch passes only
# the job dict through .spawn(), so this stays a module constant rather than
# a second function argument.
PROBE_QUERIES = 15

if MODAL_CONFIG.get("gpu") is not None:
    raise RuntimeError("Safeguard C authorises zero GPU hours")

# One call writes its result once, at the end -- a restart redoes the whole
# (small, 10-20 query) probe and no more.
RESUME_GRANULARITY = "dataset"

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
    .add_local_file(str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/m0b_regime_map.yaml")
)


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    """scripts/spawn_modal_jobs.py's generic dispatch expects this shape.

    Deliberately refuses anything but exactly [\"webqsp\"] rather than quietly
    ignoring the argument -- CONFIG["datasets"] lists all six M0B datasets,
    so a bare rename of the CLI's own validation would let --datasets
    2wiki_clean pass the generic check upstream and then silently run
    webqsp instead. This is Safeguard C's probe, not a general launcher.
    """
    if list(datasets) != [DATASET]:
        raise ValueError(
            f"m0b-webqsp-probe only ever runs {DATASET!r}; got --datasets {list(datasets)}"
        )
    return [_job()]


def _job() -> dict[str, Any]:
    settings = CONFIG["datasets"][DATASET]
    confirmation = json.loads(
        (HOST_REPO_ROOT / settings["confirmation"]).read_text(encoding="utf-8")
    )
    fingerprint = confirmation["data_fingerprint_sha256"]
    return {
        "dataset": DATASET,
        "settings": settings,
        "fingerprint": fingerprint,
        "data_remote": confirmation["config"]["data"],
        "baseline": confirmation["baseline"],
        # fingerprint[:16], matching modal_m0a1_overlap.py's own convention --
        # confirmed against the real path via Volume.listdir 2026-09-03:
        # edge_provenance_graphs/webqsp/642ee12ed1125208/structural_only/graph.pt
        "graph_root": str(
            PurePosixPath(STORAGE_ROOT)
            / MODAL_CONFIG["edge_provenance_root"]
            / DATASET
            / fingerprint[:16]
        ),
    }


def _runner_args(job: dict[str, Any]) -> argparse.Namespace:
    output_root = (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / "m0b_webqsp_probe"
        / job["fingerprint"][:16]
        / MODAL_CONFIG["execution_label"]
    )
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        data_fingerprint_sha256=job["fingerprint"],
        expected_queries=int(job["settings"]["expected_queries"]),
        baseline=job["baseline"],
        candidate_contract_compatibility=job["settings"].get(
            "candidate_contract_compatibility"
        ),
        queries=PROBE_QUERIES,
        per_seed_cap=16,
        neighbour_scan_cap_per_seed=4096,
        edge_provenance_root=Path(job["graph_root"]),
        edge_families=["structural_only"],
        a64_mainline_family="structural_only",
        output=Path(output_root) / "probe.json",
    )


@app.function(
    image=image,
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_probe(job: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    from scripts.run_m0b_webqsp_probe import run

    args = _runner_args(job)
    result = run(args)
    result_volume.commit()
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
def main() -> None:
    # For manual debugging only. The real run goes through
    # scripts/spawn_modal_jobs.py so the call survives client disconnection --
    # see this module's docstring and project memory on why `modal run
    # --detach` is never used here.
    job = _job()
    result = run_probe.remote(job)
    local_path = HOST_REPO_ROOT / "outputs" / "m0b_webqsp_probe" / "webqsp.json"
    _download(result["output_remote"], local_path)
    print(json.dumps(result, indent=2))
