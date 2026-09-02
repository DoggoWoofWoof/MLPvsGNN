#!/usr/bin/env python
"""Provenance and integrity for a cross-workspace migration of frozen state.

Moving an experiment to a second Modal workspace is only safe if three separate
questions are answered separately, so this emits three artifacts rather than one
summary:

``manifests``   What does each stage actually open? ``phase1_required_manifest``
                and ``e2_resume_manifest`` are derived from the code that opens
                the files, not hand-listed, so they cannot drift from the
                runners. Each records what is deliberately excluded and why.
``matrix``      What is already done? An integrity matrix over the E2 cells,
                using the resume rule ``run_phase_confirmation.py`` enforces on
                itself rather than an ordinal position in a job list.
``provenance``  Did the right bytes arrive? Repository, protocol and
                candidate-contract identity joined to per-file sizes and hashes,
                including the artifacts that cannot be regenerated at all.

The matrix vocabulary is fixed, and each state has exactly one response:

    COMPLETE   skip       status is CELL_COMPLETE and every model-seed is present
    PARTIAL    resume     a valid contract with model-seeds still missing
    MISSING    launch     no result.json at the cell path
    INVALID    diagnose   unreadable, contract mismatch, or a completion claim
                          the payload itself contradicts

INVALID never becomes a relaunch. A result that claims completion while missing
seeds is a corrupt artifact, and overwriting it would destroy the evidence of
whatever produced it.

Non-regenerable artifacts
-------------------------
E2 reuses the seed-0 validation checkpoint trained during E1 rather than
retraining it, and ``run_phase_confirmation`` verifies the checkpoint file's
SHA-256 against the value E1 recorded before loading it. Those 192 ``.pt`` files
are therefore the artifacts a migration must carry intact: if they are missing
or altered, seed 0 cannot be reproduced without retraining, which would break
the "seed-0 validation checkpoint reused without test peeking" contract.

Their expected hashes come from the frozen E1 results, not from this tool's own
copy of the bytes, so the check is independent of the transfer that is being
checked.

Usage::

    python scripts/migration_provenance.py manifests
    python scripts/migration_provenance.py matrix --staging D:/mpr_stage
    python scripts/migration_provenance.py provenance --staging D:/mpr_stage \\
        --source <workspace> --target <workspace>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import replicate_volume  # noqa: E402  (path set above)

OUTPUT_DIR = REPO_ROOT / "outputs" / "migration"
CONFIG = REPO_ROOT / "configs" / "phase_confirmation.yaml"
SCREEN_ROOT = REPO_ROOT / "outputs" / "phase_screen"
CONFIRMATIONS = REPO_ROOT / "outputs" / "sa_mlp_confirmation"

MODEL_NAMES = ("sa_mlp", "seed_aware_gnn")
SEEDS = (0, 1, 2, 3, 4)
CELL_COMPLETE = "PHASE_CONFIRMATION_CELL_COMPLETE"
SCREEN_COMPLETE = "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE"

PROTOCOL_DOCS = {
    "phase_confirmation": "docs/CONFIRMATION_PROTOCOL.md",
    "phase_screen": "docs/PHASE_SCREEN_PROTOCOL.md",
    "graph_substrate_audit": "docs/GRAPH_SUBSTRATE_AUDIT_PROTOCOL.md",
}
PROTOCOL_TAGS = {
    "phase_confirmation": "phase-confirmation-protocol-v1",
    "phase_screen": "phase-screen-protocol-v1",
}


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _git(*argv: str) -> str:
    result = subprocess.run(
        ["git", *argv], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def sha256_of_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_identity() -> dict[str, Any]:
    """What code state this migration is anchored to."""

    dirty = _git("status", "--porcelain")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_clean": not dirty,
        "uncommitted_paths": sorted(
            line[3:] for line in dirty.splitlines() if line
        ),
        "tags_at_head": sorted(filter(None, _git("tag", "--points-at", "HEAD").splitlines())),
    }


def protocol_identity() -> dict[str, Any]:
    """Protocol tag plus the SHA-256 of the document the tag names.

    The tag pins history; the hash pins the file as it stands now. Recording
    both makes an edit after the tag visible instead of implied.
    """

    record: dict[str, Any] = {}
    for name, relative in PROTOCOL_DOCS.items():
        path = REPO_ROOT / relative
        tag = PROTOCOL_TAGS.get(name)
        record[name] = {
            "document": relative,
            "document_sha256": sha256_of_text(path),
            "tag": tag,
            "tag_commit": _git("rev-list", "-n", "1", tag) if tag else None,
        }
    return record


def dataset_identity() -> dict[str, dict[str, Any]]:
    """Dataset -> data root, fingerprint and candidate contract, from the freeze."""

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    identity: dict[str, dict[str, Any]] = {}
    for dataset, spec in sorted(config["datasets"].items()):
        payload = json.loads(
            (CONFIRMATIONS / f"{dataset}.json").read_text(encoding="utf-8")
        )
        fingerprint = payload["data_fingerprint_sha256"]
        identity[dataset] = {
            "data_root": payload["config"]["data"].removeprefix(
                replicate_volume.STORAGE_PREFIX
            ),
            "data_fingerprint_sha256": fingerprint,
            "cell_prefix": fingerprint[:16],
            "selected_gnn": spec["selected_gnn"],
            "expected_queries": int(spec["expected_queries"]),
            "candidate_contract_compatibility": spec.get(
                "candidate_contract_compatibility"
            ),
        }
    return identity


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------


def rate_key(rate: float) -> str:
    """The on-volume spelling of a rate, matching ``modal_phase_confirmation``."""

    return f"{rate:.2f}".replace(".", "p")


def expected_cells() -> list[dict[str, Any]]:
    """The E2 cells the frozen config defines: rate 0.0 is the screen's clean arm."""

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    identity = dataset_identity()
    cells = []
    for dataset in sorted(config["datasets"]):
        for axis in sorted(config["axes"]):
            spec = config["axes"][axis]
            for rate in spec["rates"]:
                if float(rate) <= 0.0:
                    continue
                cells.append(
                    {
                        "dataset": dataset,
                        "axis": axis,
                        "rate": float(rate),
                        "regime": f"{axis}_{rate_key(float(rate))}",
                        "cell_prefix": identity[dataset]["cell_prefix"],
                        "data_fingerprint_sha256": identity[dataset][
                            "data_fingerprint_sha256"
                        ],
                        "perturbation_seed": int(spec["perturbation_seed"]),
                    }
                )
    return cells


def cell_relative_path(cell: dict[str, Any], kind: str) -> str:
    """Volume-relative directory for a cell, matching ``_cell_root``."""

    return f"outputs/{kind}/{cell['dataset']}/{cell['cell_prefix']}/{cell['regime']}"


# ---------------------------------------------------------------------------
# Integrity matrix
# ---------------------------------------------------------------------------


def classify_cell(cell: dict[str, Any], root: Path) -> dict[str, Any]:
    """Apply the resume rule ``run_phase_confirmation`` enforces on itself."""

    path = root / cell_relative_path(cell, "phase_confirmation") / "result.json"
    record = {
        "key": f"{cell['dataset']}/{cell['axis']}/rate_{cell['rate']:.2f}",
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "state": "MISSING",
        "action": "launch",
        "detail": "no result.json at the cell path",
        "seeds_present": 0,
        "seeds_expected": len(MODEL_NAMES) * len(SEEDS),
    }
    if not path.is_file():
        return record

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        record.update(
            state="INVALID", action="diagnose", detail=f"unreadable JSON: {error}"
        )
        return record

    # The runner refuses to continue a result whose contract disagrees with the
    # cell it was launched for, so a mismatch here is a stop, not a relaunch.
    mismatched = [
        field
        for field, expected in (
            ("dataset", cell["dataset"]),
            ("axis", cell["axis"]),
            ("data_fingerprint_sha256", cell["data_fingerprint_sha256"]),
        )
        if payload.get(field) != expected
    ]
    if float(payload.get("rate", -1.0)) != cell["rate"]:
        mismatched.append("rate")
    if mismatched:
        record.update(
            state="INVALID",
            action="diagnose",
            detail="contract mismatch on " + ", ".join(sorted(mismatched)),
        )
        return record

    progress = {
        model: sorted(int(seed) for seed in payload.get("models", {}).get(model, {}).get("seeds", {}))
        for model in MODEL_NAMES
    }
    present = sum(len(seeds) for seeds in progress.values())
    record["seeds_present"] = present
    record["model_progress"] = progress
    record["status"] = payload.get("status")

    full = all(set(progress[model]) == set(SEEDS) for model in MODEL_NAMES)
    if payload.get("status") == CELL_COMPLETE:
        if full:
            record.update(state="COMPLETE", action="skip", detail="all model-seeds present")
        else:
            # A completion claim its own payload contradicts. Never relaunch
            # over this: the file is evidence of a defect.
            record.update(
                state="INVALID",
                action="diagnose",
                detail=(
                    f"status is {CELL_COMPLETE} but only {present} of "
                    f"{record['seeds_expected']} model-seeds are recorded"
                ),
            )
        return record

    record.update(
        state="PARTIAL",
        action="resume",
        detail=f"{present} of {record['seeds_expected']} model-seeds recorded",
    )
    return record


def misrooted_hint(root: Path, rows: list[dict[str, Any]]) -> str | None:
    """Tell a mistyped root apart from a sweep that genuinely has not started.

    ``cell_relative_path`` is volume-relative and already carries the
    ``outputs/`` component, so pointing ``--results-root`` at ``<staging>/outputs``
    searches ``outputs/outputs/...`` and every cell reports MISSING. That is
    indistinguishable from an untouched sweep by the counts alone, and one of the
    two readings says to launch 96 cells over work that is already done.

    Only fires when nothing at all was found here *and* the tree is demonstrably
    one or two levels up, so it cannot mask a real empty sweep.
    """

    if any(row["state"] != "MISSING" for row in rows):
        return None
    for candidate in (root.parent, root.parent.parent):
        found = list((candidate / "outputs" / "phase_confirmation").glob("*/*/*/result.json"))
        if found:
            return (
                f"{len(found)} cell result(s) exist under {candidate}, but every cell "
                f"reports MISSING under {root}. Cell paths already carry the 'outputs/' "
                f"component, so pass --results-root {candidate}. Refusing rather than "
                "reporting an empty sweep: relaunching completed cells is the one "
                "mistake this matrix exists to prevent."
            )
    return None


def integrity_matrix(root: Path) -> dict[str, Any]:
    rows = [classify_cell(cell, root) for cell in expected_cells()]
    states = ("COMPLETE", "PARTIAL", "MISSING", "INVALID")
    counts = {state: sum(row["state"] == state for row in rows) for state in states}
    return {
        "results_root": str(root),
        "expected_conditions": len(rows),
        "expected_work_units": len(rows) * len(MODEL_NAMES) * len(SEEDS),
        "completed_work_units": sum(
            row["seeds_present"] for row in rows if row["state"] == "COMPLETE"
        ),
        "recorded_work_units": sum(row["seeds_present"] for row in rows),
        "counts": counts,
        "resume_plan": {
            "skip": sorted(row["key"] for row in rows if row["action"] == "skip"),
            "resume": sorted(row["key"] for row in rows if row["action"] == "resume"),
            "launch": sorted(row["key"] for row in rows if row["action"] == "launch"),
            "diagnose": sorted(row["key"] for row in rows if row["action"] == "diagnose"),
        },
        "cells": rows,
    }


# ---------------------------------------------------------------------------
# Reference capture for the regeneration gate
# ---------------------------------------------------------------------------

# Only these axes write a phase_confirmation_cache cell. feature_mask perturbs
# node features on device and reuses the clean caches.
TOPOLOGY_AXES = ("degree_rewire", "random_add", "hub_injection")

# Declared before any comparison is run, so the representative set cannot be
# chosen to suit an outcome. The risk being tested is per-generator rather than
# per-cell -- an unseeded RNG or an order-dependent reduction would show up in
# any cell of that axis -- so breadth over axes and rate extremes matters more
# than volume, and a second dataset guards against something specific to the
# first.
REFERENCE_RULE = (
    "every topology axis at the lowest and highest non-zero rate on the "
    "smallest dataset, plus one mid-rate cell per topology axis on the "
    "second-smallest dataset"
)
SECOND_DATASET_RATE = 0.25


def frozen_candidate_contract(dataset: str) -> str:
    """The candidate contract E1 recorded for this dataset.

    For a dataset whose pool predates the hop-metadata field this is the
    pre-hop hash, which is *not* the one a fresh ``load_complete_dataset``
    computes. Anything comparing against it must go through
    ``validate_candidate_contract`` with the compatibility mode, exactly as the
    experiment runners do; see ``candidate_contract_modes``.
    """

    for path in sorted((SCREEN_ROOT / dataset).rglob("rate_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        contract = payload.get("candidate_contract", {})
        expected = contract.get("expected_contract_sha256")
        if expected:
            return str(expected)
    raise SystemExit(f"No frozen candidate contract recorded for {dataset}")


def candidate_contract_modes() -> dict[str, str | None]:
    """Each dataset's declared compatibility mode, from the E2 config.

    Two of the six frozen pools were built before candidate hop metadata
    existed, so the hash E1 recorded for them is over a smaller field set than
    the one ``load_complete_dataset`` computes today. E1 recorded both hashes
    and a ``BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE`` proof that they describe
    the same pool; ``validate_candidate_contract`` is what re-derives the
    legacy hash and checks it.

    The mode is read from ``configs/phase_confirmation.yaml`` rather than
    listed here, so the gate asks the same question of the same datasets that
    E2 will when it resumes. A plain ``!=`` against the frozen hash rejects
    those two datasets on a field-set difference and calls it a changed pool.
    """

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return {
        name: spec.get("candidate_contract_compatibility")
        for name, spec in config["datasets"].items()
    }


def _datasets_by_size() -> list[str]:
    identity = dataset_identity()
    return sorted(identity, key=lambda name: identity[name]["expected_queries"])


def reference_cache_cells() -> dict[str, Any]:
    """The cache cells to capture from the source for the regeneration gate."""

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    identity = dataset_identity()
    modes = candidate_contract_modes()
    order = _datasets_by_size()
    smallest, second = order[0], order[1]

    cells: list[dict[str, Any]] = []
    for axis in TOPOLOGY_AXES:
        spec = config["axes"][axis]
        rates = sorted(float(rate) for rate in spec["rates"] if float(rate) > 0.0)
        chosen = [(smallest, rates[0]), (smallest, rates[-1]), (second, SECOND_DATASET_RATE)]
        for dataset, rate in chosen:
            cells.append(
                {
                    "dataset": dataset,
                    "axis": axis,
                    "rate": rate,
                    "perturbation_seed": int(spec["perturbation_seed"]),
                    "data_fingerprint_sha256": identity[dataset][
                        "data_fingerprint_sha256"
                    ],
                    # Taken from the frozen E1 result rather than from the
                    # migrated data, so the regeneration is checked against
                    # what the experiment recorded, not against itself. The
                    # mode travels with the hash because for two datasets the
                    # frozen hash covers fewer fields than a fresh load
                    # computes, and comparing them directly reads a field-set
                    # difference as a changed candidate pool.
                    "candidate_contract_sha256": frozen_candidate_contract(dataset),
                    "candidate_contract_compatibility": modes.get(dataset),
                    "prefix": (
                        f"phase_confirmation_cache/{dataset}/"
                        f"{identity[dataset]['cell_prefix']}/{axis}_{rate_key(rate)}"
                    ),
                }
            )
    return {
        "rule": REFERENCE_RULE,
        "declared_before_comparison": True,
        "smallest_dataset": smallest,
        "second_dataset": second,
        "axes_with_a_cache_cell": list(TOPOLOGY_AXES),
        "axes_without_a_cache_cell": ["feature_mask"],
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Non-regenerable artifacts
# ---------------------------------------------------------------------------


def screen_checkpoints() -> dict[str, Any]:
    """The seed-0 checkpoints E2 reuses, with the hashes E1 recorded for them.

    The expected hash is read from the frozen E1 result rather than computed
    here, so this is a check on the migration rather than a restatement of it.
    """

    required: list[dict[str, Any]] = []
    problems: list[str] = []
    for cell in expected_cells():
        local = (
            SCREEN_ROOT
            / cell["dataset"]
            / cell["axis"]
            / f"rate_{rate_key(cell['rate'])}.json"
        )
        if not local.is_file():
            problems.append(f"missing fetched screen result: {local.relative_to(REPO_ROOT)}")
            continue
        payload = json.loads(local.read_text(encoding="utf-8"))
        if payload.get("status") != SCREEN_COMPLETE:
            problems.append(f"{local.name}: status {payload.get('status')!r}")
            continue
        for model in MODEL_NAMES:
            source = payload["models"][model]
            remote = source["checkpoint_path"]
            if not remote.startswith(replicate_volume.STORAGE_PREFIX):
                problems.append(f"{local.name}/{model}: unexpected checkpoint root")
                continue
            required.append(
                {
                    "dataset": cell["dataset"],
                    "axis": cell["axis"],
                    "rate": cell["rate"],
                    "model": model,
                    "path": remote.removeprefix(replicate_volume.STORAGE_PREFIX),
                    "checkpoint_file_sha256": source["checkpoint_file_sha256"],
                    "checkpoint_state_sha256": source["checkpoint_sha256"],
                }
            )
    return {
        "why": (
            "run_phase_confirmation reuses the E1 seed-0 checkpoint and verifies "
            "this SHA-256 before loading it; a checkpoint that does not migrate "
            "intact cannot be reproduced without retraining, which the "
            "no-test-peeking contract forbids"
        ),
        "expected_files": len(expected_cells()) * len(MODEL_NAMES),
        "resolved_files": len(required),
        "problems": problems,
        "files": required,
    }


# ---------------------------------------------------------------------------
# Required manifests
# ---------------------------------------------------------------------------


def phase1_required_manifest() -> dict[str, Any]:
    identity = dataset_identity()
    return {
        "stage": "phase_minus_1_graph_substrate_audit",
        "runner": "scripts/run_graph_substrate_audit.py",
        "reads": {
            "per_dataset_root": list(replicate_volume.TOPOLOGY_ROOT_FILES),
            "trees": ["edge_provenance_graphs/"],
        },
        "opened_via": (
            "load_complete_dataset(..., require_embeddings=False); the audit then "
            "uses rowptr, col, num_nodes, queries, split() and the candidate "
            "contract metadata only"
        ),
        "excluded": {
            "nodes.npy": "embedding matrix; never read by a topological measurement",
            "queries_all.npy": "embedding matrix; never read by a topological measurement",
            "derived/packed_topology_v1": "E2 training cache; the audit builds CSR from graph.pt",
            "derived/fixed_structural_features_v1": "E2 feature cache; not a topology input",
            "derived/linear_rank_structure_inputs_v1": "Package A output; read by no runner in either slice",
            "outputs/": "results; the audit produces them and does not consume them",
            "phase_confirmation_cache/": "193.6 GB build_or_load cache, regenerated deterministically",
            "model checkpoints": "the audit trains nothing and scores nothing",
        },
        "datasets": identity,
    }


def e2_resume_required_manifest() -> dict[str, Any]:
    identity = dataset_identity()
    return {
        "stage": "e2_five_seed_phase_crossover_confirmation",
        "runner": "scripts/run_phase_confirmation.py via scripts/modal_phase_confirmation.py",
        "reads": {
            "per_dataset_root": list(replicate_volume.DATASET_ROOT_FILES),
            "per_dataset_derived": list(replicate_volume.DATASET_DERIVED_PREFIXES),
            "trees": [
                "outputs/phase_screen/",
                "outputs/phase_confirmation/",
                "outputs/sa_mlp_confirmation/",
            ],
        },
        "embeddings_required": True,
        "embeddings_reason": (
            "both models are trained and scored, so node_array and query_array "
            "are materialised on device; require_embeddings=False is not usable here"
        ),
        "excluded": {
            "phase_confirmation_cache/": (
                "193.6 GB of build_or_load_perturbed_topologies and "
                "build_or_load_structural_features output, keyed by intervention "
                "contract and regenerated when absent; migration must verify that "
                "regeneration before relying on the omission"
            ),
            "derived/linear_rank_structure_inputs_v1": "Package A output; not an E2 input",
        },
        "non_regenerable": [
            "outputs/phase_screen/**/checkpoints/*/seed_0.pt",
            "outputs/phase_screen/**/result.json",
            "outputs/phase_confirmation/**/result.json",
            "outputs/phase_confirmation/**/query_metrics.npz",
        ],
        "datasets": identity,
    }


# ---------------------------------------------------------------------------
# Input readiness
# ---------------------------------------------------------------------------

# Files that may legitimately be absent from a dataset root. ``node_ids.json``
# exists only where node identity is not the numeric suffix, and the source
# manifest was not written for every freeze; ``load_complete_dataset`` handles
# both. Treating them as required would report a healthy root as broken.
OPTIONAL_ROOT_FILES = ("node_ids.json", "_frozen_source_manifest.json")


def e2_required_paths(datasets: list[str] | None = None) -> dict[str, list[str]]:
    """Every concrete file E2 opens before it trains, grouped by why.

    Derived from ``run_phase_confirmation`` rather than from a remembered list:
    it loads the dataset with embeddings, loads the clean packed topology, loads
    the clean structural features for the ``feature_mask`` axis, reads the E1
    seed-0 screen result, and ``torch.load``s the seed-0 checkpoint whose hash
    that result records.
    """

    identity = dataset_identity()
    wanted = set(datasets) if datasets else set(identity)
    roots: list[str] = []
    derived: list[str] = []
    for dataset in sorted(wanted & set(identity)):
        root = identity[dataset]["data_root"]
        roots.extend(
            f"{root}/{name}"
            for name in replicate_volume.DATASET_ROOT_FILES
            if name not in OPTIONAL_ROOT_FILES
        )
        # metadata.json is the file each loader opens first, and the one whose
        # absence killed all nine gate jobs.
        derived.extend(
            f"{root}/{prefix}/metadata.json"
            for prefix in replicate_volume.DATASET_DERIVED_PREFIXES
        )

    screens: list[str] = []
    checkpoints: list[str] = []
    for cell in expected_cells():
        if cell["dataset"] not in wanted:
            continue
        screens.append(
            f"outputs/phase_screen/{cell['dataset']}/{cell['cell_prefix']}/"
            f"{cell['regime']}/result.json"
        )
    for record in screen_checkpoints()["files"]:
        if record["dataset"] in wanted:
            checkpoints.append(record["path"])

    return {
        "dataset_roots": sorted(set(roots)),
        "clean_derived_caches": sorted(set(derived)),
        "e1_screen_results": sorted(set(screens)),
        "e1_seed0_checkpoints": sorted(set(checkpoints)),
    }


def check_inputs(
    present: dict[str, int], datasets: list[str] | None = None
) -> dict[str, Any]:
    """Compare what E2 needs against what a root actually holds.

    ``present`` maps a volume-relative path to its size, so a zero-length file
    is a defect rather than a presence: an empty ``metadata.json`` fails at load
    time, after the container has mounted its inputs and the compute is paid
    for. Checking here is the difference between one refused launch and
    forty-eight failed ones.
    """

    required = e2_required_paths(datasets)
    groups: dict[str, Any] = {}
    missing_total = 0
    empty_total = 0
    for group, paths in required.items():
        missing = [path for path in paths if path not in present]
        empty = [path for path in paths if present.get(path) == 0]
        missing_total += len(missing)
        empty_total += len(empty)
        groups[group] = {
            "required": len(paths),
            "present": len(paths) - len(missing),
            "missing": missing[:20],
            "missing_count": len(missing),
            "empty": empty[:20],
        }
    return {
        "datasets": sorted(set(datasets)) if datasets else "all",
        "ready": missing_total == 0 and empty_total == 0,
        "missing_total": missing_total,
        "empty_total": empty_total,
        "groups": groups,
    }


def _local_listing(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _write(name: str, payload: dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    # A relative path reads better, but an output directory outside the repo is
    # legitimate -- a test redirecting OUTPUT_DIR, or a run writing to a
    # scratch volume -- and should not crash after the file is already written.
    try:
        shown: Path | str = path.relative_to(REPO_ROOT)
    except ValueError:
        shown = path
    print(f"wrote {shown}")
    return path


def cmd_manifests(args: argparse.Namespace) -> int:
    _write("phase1_required_manifest.json", phase1_required_manifest())
    _write("e2_resume_manifest.json", e2_resume_required_manifest())
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    root = Path(args.results_root) if args.results_root else Path(args.staging)
    matrix = integrity_matrix(root)
    hint = misrooted_hint(root, matrix["cells"])
    if hint:
        raise SystemExit(hint)
    _write("e2_integrity_matrix.json", matrix)
    counts = matrix["counts"]
    print(
        f"\n{matrix['expected_conditions']} conditions: "
        + "  ".join(f"{state} {counts[state]}" for state in counts)
    )
    print(
        f"{matrix['completed_work_units']}/{matrix['expected_work_units']} "
        "model-seed units in COMPLETE cells"
    )
    for action in ("diagnose", "resume", "launch"):
        keys = matrix["resume_plan"][action]
        if keys:
            print(f"\n{action} ({len(keys)}):")
            for key in keys[:12]:
                print(f"  {key}")
            if len(keys) > 12:
                print(f"  ... and {len(keys) - 12} more")
    return 1 if matrix["counts"]["INVALID"] else 0


# ---------------------------------------------------------------------------
# Whether the omitted cache may be omitted
# ---------------------------------------------------------------------------

EQUIVALENCE_DIR = REPO_ROOT / "outputs" / "cache_equivalence"

# The only two metadata keys a correct regeneration is expected to change, and
# the reason each one changes. Both are downstream of
# `local_topology_perturbations.perturb_packed_topologies`, whose contract
# digest covers a metadata dict containing `build_seconds` -- a
# `time.perf_counter()` duration. A hash over a wall-clock measurement cannot
# reproduce, so these two are not evidence of a differing cache.
#
# This set is deliberately narrow. A key outside it differing means something
# other than the timing field moved, and the summary must not report that as
# the known-benign case.
TIMING_DERIVED_KEYS = frozenset({"contract_sha256", "source_fingerprint_sha256"})


def regeneration_equivalence(root: Path | None = None) -> dict[str, Any]:
    """Summarize the gate: did the omitted cache regenerate, and what differed?

    Read from the gate's own result files rather than asserted, so the
    provenance record cannot claim an equivalence nobody ran. A cell whose
    arrays differ, or whose metadata differs on a key outside
    ``TIMING_DERIVED_KEYS``, is named individually -- a count alone would let
    one genuinely differing cell hide inside eight benign ones.
    """

    root = EQUIVALENCE_DIR if root is None else root
    results = sorted(root.rglob("equivalence.json")) if root.is_dir() else []
    if not results:
        return {
            "status": "NOT_RUN",
            "cells": 0,
            "detail": f"no equivalence.json under {root}",
        }

    cells: list[dict[str, Any]] = []
    arrays_differ: list[str] = []
    unexpected: list[str] = []
    differing_keys: set[str] = set()
    for path in results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = f"{payload.get('dataset')}/{payload.get('axis')}/rate_{float(payload.get('rate', 0)):.2f}"
        kinds = payload.get("comparison", {}).get("kinds", {})
        cell_arrays_equal = True
        cell_keys: set[str] = set()
        for kind, entry in kinds.items():
            if not entry.get("present"):
                unexpected.append(f"{name}: {kind} absent from the reference")
                cell_arrays_equal = False
                continue
            arrays = entry.get("arrays", {})
            if not arrays.get("files"):
                unexpected.append(f"{name}: {kind} compared no arrays")
                cell_arrays_equal = False
            if not arrays.get("all_equal"):
                cell_arrays_equal = False
                arrays_differ.append(
                    f"{name}: {kind} "
                    + ", ".join(
                        item["file"] for item in arrays.get("files", []) if not item.get("equal")
                    )
                )
            cell_keys.update(entry.get("metadata", {}).get("differing_keys", []) or [])
        outside = cell_keys - TIMING_DERIVED_KEYS
        if outside:
            unexpected.append(f"{name}: metadata differs on {sorted(outside)}")
        differing_keys.update(cell_keys)
        cells.append({
            "cell": name,
            "status": payload.get("status"),
            "arrays_equal": cell_arrays_equal,
            "differing_metadata_keys": sorted(cell_keys),
            "candidate_contract_proof": payload.get("regeneration", {})
            .get("candidate_contract_proof", {})
            .get("status"),
        })

    if arrays_differ or unexpected:
        status = "DIFFERS"
    elif differing_keys:
        status = "SEMANTICALLY_EQUIVALENT_TIMING_DERIVED_HASHES_DIFFER"
    else:
        status = "BIT_IDENTICAL"
    return {
        "status": status,
        "cells": len(cells),
        "datasets": sorted({cell["cell"].split("/")[0] for cell in cells}),
        "every_array_equal": not arrays_differ,
        "differing_metadata_keys": sorted(differing_keys),
        "differing_keys_are_timing_derived": bool(differing_keys)
        and not (differing_keys - TIMING_DERIVED_KEYS),
        "arrays_that_differ": sorted(arrays_differ),
        "unexpected": sorted(unexpected),
        "per_cell": cells,
    }


def cmd_provenance(args: argparse.Namespace) -> int:
    staging = Path(args.staging)
    staged: dict[str, Any] = {}
    for candidate in (
        staging / replicate_volume.manifest_name(args.slice),
        staging / replicate_volume.MANIFEST_NAME,
    ):
        if candidate.is_file():
            record = json.loads(candidate.read_text(encoding="utf-8"))
            if record.get("slice") == args.slice:
                staged = record
                break

    checkpoints = screen_checkpoints()
    payload = {
        "migration": {
            "source_workspace": args.source or staged.get("source_profile"),
            "target_workspace": args.target,
            "source_volume": staged.get("source_volume", replicate_volume.DEFAULT_VOLUME),
            "target_volume": args.volume,
            "storage_mount": replicate_volume.STORAGE_PREFIX.rstrip("/"),
            "slice": staged.get("slice"),
        },
        "repository": repository_identity(),
        "protocols": protocol_identity(),
        "datasets": dataset_identity(),
        "non_regenerable_artifacts": {
            "screen_seed0_checkpoints": checkpoints,
        },
        "regenerable_omissions": {
            "phase_confirmation_cache": {
                "migrated": False,
                "reason": "deterministic derived cache",
                "size_bytes_source": None,
                "regeneration_equivalence": regeneration_equivalence(),
            }
        },
        "staged_files": {
            # Without this flag a run made before the transfer finishes emits
            # `count: 0, total_bytes: null`, which is indistinguishable from a
            # completed transfer of nothing. The manifest has to say which of
            # the two it is.
            "staging_record_found": bool(staged),
            "requested_slice": args.slice,
            "count": len(staged.get("files", {})),
            "total_bytes": staged.get("total_bytes"),
            "failed": staged.get("failed_files", {}),
            "files": staged.get("files", {}),
        },
    }
    _write("migration_provenance.json", payload)
    print(
        f"\nscreen seed-0 checkpoints required: {checkpoints['resolved_files']}"
        f"/{checkpoints['expected_files']}"
    )
    for problem in checkpoints["problems"][:10]:
        print(f"  ! {problem}")
    return 0


def cmd_inputs(args: argparse.Namespace) -> int:
    """Refuse a launch whose inputs are not all there, before it costs anything."""

    datasets = (
        [name.strip() for name in args.datasets.split(",") if name.strip()]
        if args.datasets
        else None
    )
    if args.remote:
        volume = replicate_volume.modal.Volume.from_name(
            args.volume, create_if_missing=False
        )
        present = {
            path: size for path, size in replicate_volume._walk(volume, "/", strict=True)
        }
        where = f"{args.volume} (remote)"
    else:
        root = Path(args.results_root) if args.results_root else Path(args.staging)
        if not root.is_dir():
            raise SystemExit(f"No such root: {root}")
        present = _local_listing(root)
        where = str(root)

    report = check_inputs(present, datasets)
    report["checked"] = where
    _write("e2_input_readiness.json", report)

    print(f"\nE2 inputs at {where}")
    for group, block in report["groups"].items():
        mark = "ok " if not block["missing_count"] and not block["empty"] else "GAP"
        print(f"  {mark} {group:24s} {block['present']}/{block['required']}")
        for path in block["missing"]:
            print(f"        missing  {path}")
        for path in block["empty"]:
            print(f"        empty    {path}")
        if block["missing_count"] > len(block["missing"]):
            print(f"        ... and {block['missing_count'] - len(block['missing'])} more")
    if report["ready"]:
        print("\nevery declared E2 input is present and non-empty")
        return 0
    print(
        f"\n{report['missing_total']} missing, {report['empty_total']} empty. "
        "Refusing rather than launching: a container discovers this after its "
        "inputs are mounted and the compute is already paid for."
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=("manifests", "matrix", "provenance", "inputs")
    )
    parser.add_argument(
        "--datasets", help="inputs: restrict the check to these datasets"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="inputs: check the target volume rather than a local root",
    )
    parser.add_argument(
        "--staging", default=str(REPO_ROOT.parent / "mpr_replication_staging")
    )
    parser.add_argument(
        "--results-root",
        help="matrix: directory holding outputs/phase_confirmation (default: --staging)",
    )
    parser.add_argument("--source", help="provenance: source workspace name")
    parser.add_argument("--target", help="provenance: target workspace name")
    parser.add_argument("--volume", default=replicate_volume.DEFAULT_VOLUME)
    parser.add_argument(
        "--slice",
        default="e2_resume",
        choices=replicate_volume.SLICES,
        help="provenance: which staged slice to record",
    )
    args = parser.parse_args()

    handlers = {
        "manifests": cmd_manifests,
        "matrix": cmd_matrix,
        "provenance": cmd_provenance,
        "inputs": cmd_inputs,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
