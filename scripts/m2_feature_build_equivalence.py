#!/usr/bin/env python
"""M2 step D: is the feature build separable from the GPU fit, bit-exactly?

configs/m2_qls_v2_freeze.yaml#feature_build_compute_check asks one question
before any compute is spent: can the immutable per-(dataset, regime) feature
store be built on a CPU container, persisted, and then loaded by the GPU
trainer, with the loaded store bit-exact against what the current
single-container path builds in memory? ~83% of M2's estimated GPU-billed
seconds is that build, and an A10G sits attached and idle for all of it.

The declaration's own bar: "CPU-built store == current-path store,
element-for-element ... Not 'close', not 'same metrics after training'. A test
proving this is the gate; a reasoned argument that it must be true is not."

So this script measures rather than argues. Four stages, each on a real-shaped
fixture (the same one scripts/m2_reuse_audit.py probes with, reused rather than
reinvented so the two audits cannot disagree about what a cell looks like):

1. THE REAL CALL SITE. scripts/run_m2_qls_v2_freeze.run() is invoked for real,
   with recorders wrapped around _cell_master_local and _arm_store. Every
   argument the build receives is inspected: if no torch object and no device
   ever enters it, no CUDA state can change what comes out, and that is a
   property of the code rather than a claim about it. The persisted cell master
   and the persisted arm store are then read back and compared to what the run
   held in memory.
2. EVERY ARM, FROM A RELOADED MASTER. For each regime and each runner arm the
   M2 matrix names, the arm tensor built from the reloaded master block must be
   byte-identical to the one built from the in-memory block. This is the actual
   split: a CPU container writes the master, a GPU container slices it.
3. A DIFFERENT PROCESS, NO CUDA, ONE NUMBA THREAD. The build runs again in a
   child process with CUDA_VISIBLE_DEVICES="" and NUMBA_NUM_THREADS=1. Several
   kernels on this path are @njit(parallel=True), and a CPU container would not
   have the GPU container's core count -- so a thread-count-dependent reduction
   would silently change features across the split. The child's fixture bytes
   are hashed against the parent's, so a difference in the masters cannot be
   blamed on a different input.
4. TRAINING FROM A RELOADED STORE. Same seed, same data: the fit from the
   reloaded store must produce the identical checkpoint and the identical
   metrics, not merely similar ones.

What this cannot prove is stated in the artifact rather than glossed: no A10G
container is available to this script, so stage 3 is a CPU-vs-CPU comparison
across processes. That is the honest scope -- and stage 1's finding is what
makes it sufficient, since a build that never receives a torch object cannot
observe whether a GPU is attached.

Writes outputs/m2_qls_v2_freeze/feature_build_equivalence.json. Spends no
Modal compute and launches nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.structural_features import StructuralFeatureStore  # noqa: E402
from scripts import run_m1a_feature_screen as _m1a  # noqa: E402
from scripts import run_m2_qls_v2_freeze as _m2  # noqa: E402
from scripts.m2_reuse_audit import (  # noqa: E402
    FIXTURE_EMBEDDING_DIM,
    FIXTURE_FAMILIES,
    FIXTURE_NEIGHBOUR_SCAN_CAP,
    FIXTURE_PER_SEED_CAP,
    MAINLINE_FAMILY,
    write_probe_fixture,
)

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "feature_build_equivalence.json"
REGIMES = ("R1", "R2", "R3")
#: The fixture is loaded as this dataset, which declares all three regimes.
FIXTURE_DATASET = "hotpotqa_clean"
#: Files that define the fixture. Hashed on both sides of the process boundary
#: so a master-block difference can never be blamed on a different input.
FIXTURE_FILES = (
    "nodes.npy",
    "queries_all.npy",
    "dense_top200_all.npy",
    "splade_top200_all.npy",
    "query_ids_all.json",
    "graph.pt",
)

BUILD_KWARGS = (
    "views", "queries", "dense", "splade", "rowptr", "col",
    "num_nodes", "operators", "family_rowptr", "family_col",
    "node_embeddings", "budget",
)


# --- helpers ------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_hashes(data_root: Path) -> dict[str, str]:
    return {name: _sha256_of_file(data_root / name) for name in FIXTURE_FILES}


def declared_runner_arms() -> list[str]:
    """Every distinct runner arm the M2 matrix names, read from the declaration."""

    declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    arms: list[str] = []
    for cells in declaration["m2_selection_matrix"]["cells"].values():
        for regime_arms in cells.values():
            for declared in regime_arms:
                arm = _m2.runner_arm(declared)
                if arm not in arms:
                    arms.append(arm)
    return arms


def build_cell(fixture: dict, regime: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    scored_sets, master_blocks, _latencies = _m1a._cell_master_local(
        regime=regime, **{key: fixture[key] for key in BUILD_KWARGS}
    )
    return scored_sets, master_blocks


def arm_arrays(fixture: dict, regime: str, master_blocks: list[np.ndarray], arm: str):
    """The three arrays the trainer actually consumes, plus the width it sees."""

    store, width = _m1a._arm_store(
        arm=arm,
        regime=regime,
        master_blocks=master_blocks,
        queries=fixture["queries"],
        query_count=fixture["query_count"],
        num_nodes=fixture["num_nodes"],
    )
    return store, int(width)


def _identical(left, right) -> bool:
    """Byte-for-byte, dtype included. np.array_equal alone would call
    float16 0.5 equal to float32 0.5, which is not what "bit-exact under the
    exact dtype semantics already in force" means."""

    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _blocks_identical(left: list[np.ndarray], right: list[np.ndarray]) -> bool:
    return len(left) == len(right) and all(
        _identical(a, b) for a, b in zip(left, right, strict=True)
    )


def _store_identical(left: StructuralFeatureStore, right: StructuralFeatureStore) -> bool:
    return (
        _identical(left.local, right.local)
        and _identical(left.candidate_ptr, right.candidate_ptr)
        and _identical(left.query_position, right.query_position)
        and _identical(left.static, right.static)
    )


# --- stage 1: the real call site ----------------------------------------------


def _runner_args(fixture_root: Path, output_root: Path) -> argparse.Namespace:
    """What the Modal launcher would hand run(), pointed at the fixture.

    Hyperparameters are read off scripts/modal_m2_qls_v2_freeze._runner_args
    rather than retyped, so this probe exercises the configuration the real run
    uses and cannot drift from it.
    """

    from scripts.modal_m2_qls_v2_freeze import _runner_args as launcher_args

    template = launcher_args(
        {
            "dataset": FIXTURE_DATASET,
            "fingerprint": "0" * 64,
            "data_remote": "unused",
            "baseline": None,
            "expected_queries": 0,
            "validation_split_queries": 0,
            "candidate_contract_compatibility": None,
            "graph_root": "unused",
            "source_commit": None,
        },
        stage="headline",
    )
    data = fixture_root / "data"
    frozen = load_complete_dataset(data, dataset=FIXTURE_DATASET)
    template.data = data
    template.baseline = {"candidate_contract_sha256": frozen.metadata["candidate_contract_sha256"]}
    template.candidate_contract_compatibility = None
    template.expected_queries = len(frozen.queries)
    template.queries = 3
    template.holdout_fraction = 0.34
    template.per_seed_cap = FIXTURE_PER_SEED_CAP
    template.neighbour_scan_cap_per_seed = FIXTURE_NEIGHBOUR_SCAN_CAP
    template.frozen_embedding_dim = FIXTURE_EMBEDDING_DIM
    template.edge_provenance_root = fixture_root / "families"
    template.edge_families = list(FIXTURE_FAMILIES)
    template.a64_mainline_family = MAINLINE_FAMILY
    template.device = "cpu"
    template.epochs = 1
    template.data_fingerprint_sha256 = "0" * 64
    template.source_commit = "0" * 40
    template.artifact_root = output_root / "fits"
    template.output = output_root / "qls_v2_freeze.json"
    return template


def stage_one_real_call_site(fixture_root: Path, output_root: Path) -> dict[str, Any]:
    """Run the real runner; record what the build was handed and what it wrote."""

    recorded_build: list[dict[str, Any]] = []
    recorded_arms: list[dict[str, Any]] = []
    real_master = _m1a._cell_master_local
    real_arm_store = _m1a._arm_store

    def recording_master(**kwargs):
        result = real_master(**kwargs)
        recorded_build.append(
            {
                "regime": kwargs["regime"],
                "argument_types": {
                    name: type(value).__name__ for name, value in sorted(kwargs.items())
                },
                "scored_sets": [np.array(block, copy=True) for block in result[0]],
                "master_blocks": [np.array(block, copy=True) for block in result[1]],
            }
        )
        return result

    def recording_arm_store(**kwargs):
        store, width = real_arm_store(**kwargs)
        recorded_arms.append(
            {
                "regime": kwargs["regime"],
                "arm": kwargs["arm"],
                "argument_types": {
                    name: type(value).__name__ for name, value in sorted(kwargs.items())
                },
                "store": store,
                "precomputed_width": int(width),
            }
        )
        return store, width

    _m1a._cell_master_local = recording_master
    _m1a._arm_store = recording_arm_store
    try:
        result = _m2.run(_runner_args(fixture_root, output_root))
    finally:
        _m1a._cell_master_local = real_master
        _m1a._arm_store = real_arm_store

    torch_typenames = {"Tensor", "device", "Module", "Parameter"}
    offending = sorted(
        f"{record['regime']}.{name}={typename}"
        for record in recorded_build + recorded_arms
        for name, typename in record["argument_types"].items()
        if typename in torch_typenames
    )

    reload_checks = []
    artifact_root = Path(result["artifact_root"])
    for record in recorded_build:
        regime = record["regime"]
        scored, masters = _m2.load_cell_features(artifact_root / regime / "cell_features")
        reload_checks.append(
            {
                "regime": regime,
                "what": "cell_master",
                "identical": _blocks_identical(record["scored_sets"], scored)
                and _blocks_identical(record["master_blocks"], masters),
            }
        )
    for record in recorded_arms:
        regime = record["regime"]
        arm_cell = result["cells"][regime]
        declared = next(
            name for name in arm_cell["arms"] if _m2.runner_arm(name) == record["arm"]
        )
        root = artifact_root / regime / _m2.arm_slug(declared) / "feature_store"
        reloaded = StructuralFeatureStore.load(root)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        reload_checks.append(
            {
                "regime": regime,
                "what": f"arm_store:{record['arm']}",
                "identical": _store_identical(record["store"], reloaded),
                "precomputed_width_preserved": (
                    int(metadata["precomputed_width"]) == record["precomputed_width"]
                ),
            }
        )

    return {
        "regimes_built": [record["regime"] for record in recorded_build],
        "arms_built": [f"{r['regime']}/{r['arm']}" for r in recorded_arms],
        "build_argument_types": {
            record["regime"]: record["argument_types"] for record in recorded_build
        },
        "torch_objects_reaching_the_build": offending,
        "no_torch_object_reaches_the_build": not offending,
        "reload_checks": reload_checks,
        "every_persisted_object_reloads_identical": all(
            check["identical"] and check.get("precomputed_width_preserved", True)
            for check in reload_checks
        ),
        "recorded": {"build": recorded_build, "arms": recorded_arms},
    }


# --- stage 2: every arm, from a reloaded master -------------------------------


def stage_two_every_arm(fixture: dict, artifact_root: Path) -> dict[str, Any]:
    arms = declared_runner_arms()
    rows = []
    for regime in REGIMES:
        in_memory_scored, in_memory_masters = build_cell(fixture, regime)
        reloaded_scored, reloaded_masters = _m2.load_cell_features(
            artifact_root / regime / "cell_features"
        )
        master_identical = _blocks_identical(
            in_memory_masters, reloaded_masters
        ) and _blocks_identical(in_memory_scored, reloaded_scored)
        for arm in arms:
            try:
                memory_store, memory_width = arm_arrays(fixture, regime, in_memory_masters, arm)
            except ValueError as error:
                # NODE_ROLE outside R3 for a historical arm is refused by
                # design. A refusal on both sides is agreement, and recording
                # it keeps the row count honest rather than silently skipping.
                rows.append(
                    {
                        "regime": regime,
                        "arm": arm,
                        "refused_on_both_sides": _refuses(fixture, regime, reloaded_masters, arm),
                        "refusal": str(error),
                        "identical": None,
                    }
                )
                continue
            disk_store, disk_width = arm_arrays(fixture, regime, reloaded_masters, arm)
            rows.append(
                {
                    "regime": regime,
                    "arm": arm,
                    "master_identical": master_identical,
                    "identical": _store_identical(memory_store, disk_store),
                    "precomputed_width": memory_width,
                    "width_identical": memory_width == disk_width,
                    "nonzero_columns": int(
                        np.count_nonzero(np.asarray(memory_store.local).any(axis=0))
                    ),
                }
            )
    compared = [row for row in rows if row["identical"] is not None]
    refused = [row for row in rows if row["identical"] is None]
    return {
        "arms_checked": arms,
        "rows": rows,
        "compared": len(compared),
        "refused_on_both_sides": len(refused),
        "all_identical": all(
            row["identical"] and row["width_identical"] and row["master_identical"]
            for row in compared
        )
        and all(row["refused_on_both_sides"] for row in refused),
    }


def _refuses(fixture: dict, regime: str, masters: list[np.ndarray], arm: str) -> bool:
    try:
        arm_arrays(fixture, regime, masters, arm)
    except ValueError:
        return True
    return False


# --- stage 3: a different process, no CUDA, one numba thread ------------------


def _child_build(root: Path) -> None:
    """Rebuild the fixture and every regime's master block, and persist them."""

    fixture = write_probe_fixture(root)
    (root / "fixture_hashes.json").write_text(
        json.dumps(_fixture_hashes(root / "data"), indent=2), encoding="utf-8"
    )
    for regime in REGIMES:
        scored_sets, master_blocks = build_cell(fixture, regime)
        _m2.save_cell_features(root / "masters" / regime, scored_sets, master_blocks)


def stage_three_separate_process(fixture_root: Path, fixture: dict) -> dict[str, Any]:
    parent_hashes = _fixture_hashes(fixture_root / "data")
    with tempfile.TemporaryDirectory(
        prefix="m2_feature_build_child_", ignore_cleanup_errors=True
    ) as child_dir:
        child_root = Path(child_dir)
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "NUMBA_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")]),
        }
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child-build", str(child_root)],
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:  # pragma: no cover - surfaced, not swallowed
            raise RuntimeError(
                f"child build failed ({completed.returncode}):\n{completed.stderr[-4000:]}"
            )
        child_hashes = json.loads((child_root / "fixture_hashes.json").read_text(encoding="utf-8"))
        rows = []
        for regime in REGIMES:
            parent_scored, parent_masters = build_cell(fixture, regime)
            child_scored, child_masters = _m2.load_cell_features(child_root / "masters" / regime)
            rows.append(
                {
                    "regime": regime,
                    "scored_identical": _blocks_identical(parent_scored, child_scored),
                    "master_identical": _blocks_identical(parent_masters, child_masters),
                    "master_rows": int(sum(block.shape[0] for block in parent_masters)),
                    "master_columns": int(parent_masters[0].shape[1]) if parent_masters else 0,
                }
            )
    return {
        "child_environment": {"CUDA_VISIBLE_DEVICES": "", "NUMBA_NUM_THREADS": "1"},
        "parent_numba_threads": os.environ.get("NUMBA_NUM_THREADS", "default"),
        "fixture_bytes_identical": parent_hashes == child_hashes,
        "fixture_hashes": parent_hashes,
        "rows": rows,
        "all_identical": parent_hashes == child_hashes
        and all(row["scored_identical"] and row["master_identical"] for row in rows),
    }


# --- stage 4: training from a reloaded store ----------------------------------


def stage_four_training(
    fixture: dict, fixture_root: Path, artifact_root: Path, work_root: Path
) -> dict[str, Any]:
    """Fit the universal arm twice -- in-memory store, then reloaded store."""

    regime = "R3"
    arm = _m2.RUNNER_UNIVERSAL_ARM
    args = _runner_args(fixture_root, work_root)
    _scored, in_memory_masters = build_cell(fixture, regime)
    _reloaded_scored, reloaded_masters = _m2.load_cell_features(
        artifact_root / regime / "cell_features"
    )
    widened = [
        _m1a._widen_query(query, scored)
        for query, scored in zip(fixture["queries"], _scored, strict=True)
    ]
    train_queries, held_out = _m2.holdout_split(widened, args.holdout_fraction)
    device = torch.device("cpu")
    # Both embedding tables exactly as run() builds them: the whole node and
    # query arrays, indexed by position, not a per-query slice.
    frozen = load_complete_dataset(fixture_root / "data", dataset=FIXTURE_DATASET, require_embeddings=True)
    node_embeddings = torch.from_numpy(
        np.array(frozen.node_array, dtype=np.float32, copy=True)
    ).to(device)
    query_embeddings = torch.from_numpy(
        np.array(frozen.query_array, dtype=np.float32, copy=True)
    ).to(device)

    fits = {}
    for label, masters in (("in_memory", in_memory_masters), ("reloaded", reloaded_masters)):
        store, width = arm_arrays(fixture, regime, masters, arm)
        fits[label] = _m2.fit_one_arm(
            declared_arm=_m2.DECLARED_UNIVERSAL_ARM,
            regime=regime,
            store=store,
            precomputed_width=width,
            train_queries=train_queries,
            validation_queries=held_out,
            node_embeddings=node_embeddings,
            query_embeddings=query_embeddings,
            device=device,
            args=args,
            fit_root=work_root / "training_probe" / label,
            provenance={
                "source_commit": "0" * 40,
                "config_sha256": "0" * 64,
                "dataset_fingerprint_sha256": "0" * 64,
                "candidate_contract_sha256": "0" * 64,
                "candidate_id_order_sha256": "0" * 64,
            },
            feature_build_latency_ms={"p50": 0.0, "p95": 0.0, "p99": 0.0},
            feature_store_fingerprint=_m2._sha256_of_arrays(
                np.ascontiguousarray(store.local),
                np.ascontiguousarray(store.candidate_ptr),
                np.ascontiguousarray(store.query_position),
            ),
        )

    left = torch.load(work_root / "training_probe" / "in_memory" / "checkpoint.pt")
    right = torch.load(work_root / "training_probe" / "reloaded" / "checkpoint.pt")
    checkpoint_identical = sorted(left) == sorted(right) and all(
        torch.equal(left[key], right[key]) for key in left
    )
    metrics_identical = fits["in_memory"]["metrics"] == fits["reloaded"]["metrics"]
    return {
        "regime": regime,
        "arm": arm,
        "checkpoint_tensors": len(left),
        "checkpoint_identical": checkpoint_identical,
        "metrics_identical": metrics_identical,
        "parameters": fits["in_memory"]["parameters"],
        "metrics": fits["in_memory"]["metrics"],
        "all_identical": checkpoint_identical and metrics_identical,
    }


# --- entry point ---------------------------------------------------------------


def run_probe() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="m2_feature_build_equivalence_", ignore_cleanup_errors=True
    ) as workdir:
        root = Path(workdir)
        fixture_root = root / "fixture"
        fixture = write_probe_fixture(fixture_root)
        output_root = root / "run"
        one = stage_one_real_call_site(fixture_root, output_root)
        artifact_root = output_root / "fits"
        two = stage_two_every_arm(fixture, artifact_root)
        three = stage_three_separate_process(fixture_root, fixture)
        four = stage_four_training(fixture, fixture_root, artifact_root, root / "training")

    one.pop("recorded", None)
    equivalent = (
        one["no_torch_object_reaches_the_build"]
        and one["every_persisted_object_reloads_identical"]
        and two["all_identical"]
        and three["all_identical"]
        and four["all_identical"]
    )
    return {
        "status": "M2_FEATURE_BUILD_EQUIVALENCE_COMPLETE",
        "declaration": "configs/m2_qls_v2_freeze.yaml",
        "question": (
            "can the per-(dataset, regime) feature store be built off the GPU, persisted, and "
            "loaded by the trainer, bit-exactly?"
        ),
        "verdict": "EQUIVALENT" if equivalent else "NOT_EQUIVALENT",
        "equivalent": equivalent,
        "stage_1_real_call_site": one,
        "stage_2_every_arm_from_a_reloaded_master": two,
        "stage_3_separate_process_no_cuda_one_numba_thread": three,
        "stage_4_training_from_a_reloaded_store": four,
        "what_this_cannot_prove": (
            "No A10G container is available to this script, so stage 3 compares CPU against CPU "
            "across a process boundary rather than against a real GPU container. Stage 1 is what "
            "makes that sufficient rather than a gap: the build receives only numpy arrays, "
            "python ints and plain objects -- no torch tensor and no device -- so it cannot "
            "observe whether an accelerator is attached. In the current path the build already "
            "runs on the container's CPU while the A10G sits idle; the split changes which "
            "container that CPU belongs to, not what runs."
        ),
        "what_this_does_not_decide": (
            "Whether to adopt the split. That is feature_build_compute_check.adopt_only_if's "
            "test -- behaviour-preserving AND a small orchestration change -- and it is recorded "
            "in the declaration, not here."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--child-build",
        type=Path,
        default=None,
        help="internal: rebuild the fixture and persist every regime's master into this root",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    if args.child_build is not None:
        _child_build(args.child_build)
        return 0

    report = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "verdict", "equivalent")}, indent=2))
    return 0 if report["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
