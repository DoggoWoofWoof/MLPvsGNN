"""Modal launcher for the cache regeneration gate.

CPU only, and it touches no frozen result. It reads the frozen clean topology
cache and the candidate pools, rebuilds a declared handful of
``phase_confirmation_cache`` cells under a scratch prefix, and compares them
against captures of the source workspace's cells held under a quarantine prefix.

The two prefixes matter. ``build_or_load_*`` loads a cell it finds instead of
building it, so a capture written at its own path would be loaded straight back
and the comparison would be a file against itself. Captures live under
``migration_reference/`` and regenerations under ``migration_regenerated/``;
neither is a path any runner reads.

Submit through ``scripts/spawn_modal_jobs.py cache-equivalence`` so the calls are
server-side and survive client teardown.
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
    if (HOST_REPO_ROOT / "configs" / "cache_equivalence.yaml").is_file()
    else Path(REMOTE_ROOT)
)
CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "cache_equivalence.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
MODAL_CONFIG = CONFIG["modal"]
STORAGE_ROOT = MODAL_CONFIG["storage_root"]
# The feature contract must be byte-for-byte the one E2 uses. A different
# dict here would change contract_sha256 and the gate would report a
# difference that has nothing to do with determinism.
FEATURE_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "sa_mlp_screen.yaml"

REFERENCE_PREFIX = CONFIG["quarantine"]["reference_prefix"]
REGENERATED_PREFIX = CONFIG["quarantine"]["regeneration_prefix"]

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
        str(CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/cache_equivalence.yaml"
    )
    .add_local_file(
        str(FEATURE_CONFIG_PATH), remote_path=f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml"
    )
    .add_local_file(
        str(RUNTIME_REPO_ROOT / "configs" / "phase_confirmation.yaml"),
        remote_path=f"{REMOTE_ROOT}/configs/phase_confirmation.yaml",
    )
)


def feature_config(screen: dict[str, Any]) -> dict[str, Any]:
    """Exactly the dict ``modal_phase_confirmation`` builds for the same call.

    Delegates to the runner so the container run and a local run cannot ask for
    different features and report the difference as non-determinism.
    """

    from scripts.run_cache_equivalence import feature_contract

    return feature_contract(screen)


def _jobs(datasets: list[str] | None = None) -> list[dict[str, Any]]:
    """One job per declared reference cell; the set comes from the config, not here.

    ``datasets`` only narrows the declared set for a partial run. It cannot
    widen it: the representative rule is the config's, not the caller's.
    """

    import sys

    sys.path.insert(0, str(HOST_REPO_ROOT))
    from scripts.migration_provenance import reference_cache_cells

    declared = reference_cache_cells()
    wanted = set(datasets) if datasets else None
    jobs = []
    for cell in declared["cells"]:
        dataset = cell["dataset"]
        if wanted is not None and dataset not in wanted:
            continue
        confirmation = json.loads(
            (HOST_REPO_ROOT / CONFIG["datasets"][dataset]["confirmation"]).read_text(
                encoding="utf-8"
            )
        )
        jobs.append(
            {
                "dataset": dataset,
                "axis": cell["axis"],
                "rate": float(cell["rate"]),
                "perturbation_seed": int(cell["perturbation_seed"]),
                "cell_prefix": cell["prefix"],
                "fingerprint": confirmation["data_fingerprint_sha256"],
                "candidate_contract_sha256": cell["candidate_contract_sha256"],
                "data_remote": confirmation["config"]["data"],
                "rule": declared["rule"],
            }
        )
    return jobs


def _runner_args(job: dict[str, Any]) -> argparse.Namespace:
    storage = PurePosixPath(STORAGE_ROOT)
    relative = job["cell_prefix"]
    data_remote = job["data_remote"]
    return argparse.Namespace(
        data=Path(data_remote),
        dataset=job["dataset"],
        axis=job["axis"],
        rate=job["rate"],
        perturbation_seed=job["perturbation_seed"],
        clean_topology_cache=Path(data_remote) / "derived" / "packed_topology_v1",
        # The capture, and a scratch root that must not exist yet.
        reference_root=Path(storage / REFERENCE_PREFIX / relative),
        regenerated_root=Path(storage / REGENERATED_PREFIX / relative),
        data_fingerprint_sha256=job["fingerprint"],
        candidate_contract_sha256=job["candidate_contract_sha256"],
        feature_config=job["feature_config"],
        output=Path(
            storage
            / "outputs"
            / "cache_equivalence"
            / job["dataset"]
            / job["fingerprint"][:16]
            / f"{job['axis']}_{job['rate']:.2f}".replace(".", "p")
            / "equivalence.json"
        ),
    )


@app.function(
    image=image,
    volumes={STORAGE_ROOT: result_volume},
    timeout=MODAL_CONFIG["timeout_seconds"],
    cpu=MODAL_CONFIG["cpu"],
    memory=MODAL_CONFIG["memory_mb"],
)
def run_equivalence(job: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    from scripts.run_cache_equivalence import run

    screen = yaml.safe_load(
        Path(f"{REMOTE_ROOT}/configs/sa_mlp_screen.yaml").read_text(encoding="utf-8")
    )
    # The expected candidate contract travels with the job, taken from the
    # frozen E1 record. The runner compares the migrated data against it, so a
    # dataset that arrived wrong stops the gate instead of quietly passing.
    job = dict(job, feature_config=feature_config(screen))
    args = _runner_args(job)
    result = run(args, checkpoint_hook=result_volume.commit)
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "axis": job["axis"],
        "rate": job["rate"],
        "output_remote": str(args.output),
    }


@app.local_entrypoint()
def main() -> None:
    """Blocking fallback. Prefer scripts/spawn_modal_jobs.py for a real submission."""

    jobs = _jobs(sorted(CONFIG["datasets"]))
    results = list(
        run_equivalence.map(jobs, return_exceptions=True, wrap_returned_exceptions=False)
    )
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} equivalence job(s) failed: {failures}")
    differing = [item for item in results if item["status"] == "CACHE_REGENERATION_DIFFERS"]
    print(json.dumps(results, indent=2))
    if differing:
        raise RuntimeError(
            f"{len(differing)} cell(s) did not regenerate to the captured contents; "
            "the cache omission is not safe as it stands"
        )
