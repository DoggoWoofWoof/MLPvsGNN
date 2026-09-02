#!/usr/bin/env python
"""Prove the omitted recompute cache regenerates to the same thing it was.

``phase_confirmation_cache/`` is 193.6 GB -- 85% of the frozen volume -- and it
is not a result. It is what ``build_or_load_perturbed_topologies`` and
``build_or_load_structural_features`` write when a cell runs, keyed by its
intervention contract, and both functions rebuild it when it is absent. A
migration that copied it would spend days moving derived bytes.

Skipping it is only defensible if regeneration is shown, not assumed. This
compares a reference cache cell captured from the source workspace against one
regenerated from the frozen clean inputs, and reports the comparison rather than
a verdict.

What "equivalent" means here
----------------------------
Not byte-identical. ``metadata.json`` records wall-clock build times, so two
correct runs of the same computation differ in those fields on every machine.
The comparison is therefore:

* every ``.npy`` array compared on dtype, shape and exact values -- no
  tolerance, because these are integer index structures and deterministic
  float reductions, and a real difference here changes what the model sees;
* every metadata field compared exactly, except a declared set of timing keys
  which are listed in the output so the exclusion is visible rather than
  implied;
* ``contract_sha256`` compared explicitly, because that is the field
  ``build_or_load_structural_features`` itself refuses to load past.

Only the three topology axes populate this cache. ``feature_mask`` perturbs node
features on device and reuses the clean caches, so it has no cell here; that is
recorded in the output rather than silently skipped.

The reference cells live under their own prefix, never under
``phase_confirmation_cache/``. If they were written there, the "regeneration"
would just load them back and the test would prove nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Timings differ between two correct runs of the same computation. Every other
# metadata field is compared exactly.
NON_SEMANTIC_METADATA_KEYS = frozenset(
    {
        "cold_build_seconds",
        "static_preprocessing_seconds",
        "local_preprocessing_seconds",
        "total_preprocessing_seconds",
        "build_seconds",
    }
)

TOPOLOGY_AXES = ("degree_rewire", "random_add", "hub_injection")
CACHE_KINDS = ("packed_topology_v1", "fixed_structural_features_v1")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_arrays(reference: Path, regenerated: Path) -> dict[str, Any]:
    """Exact comparison of every ``.npy`` in the two directories."""

    names = sorted(
        {path.name for path in reference.glob("*.npy")}
        | {path.name for path in regenerated.glob("*.npy")}
    )
    rows: list[dict[str, Any]] = []
    for name in names:
        left, right = reference / name, regenerated / name
        row: dict[str, Any] = {"file": name}
        if not left.is_file() or not right.is_file():
            row.update(
                equal=False,
                reason="absent from " + ("reference" if not left.is_file() else "regenerated"),
            )
            rows.append(row)
            continue
        a = np.load(left, mmap_mode="r")
        b = np.load(right, mmap_mode="r")
        row.update(
            dtype=str(a.dtype),
            dtype_equal=a.dtype == b.dtype,
            shape=list(a.shape),
            shape_equal=a.shape == b.shape,
        )
        if not (row["dtype_equal"] and row["shape_equal"]):
            row["equal"] = False
            rows.append(row)
            continue
        equal = bool(np.array_equal(np.asarray(a), np.asarray(b)))
        row["equal"] = equal
        if not equal:
            difference = np.asarray(a).astype(np.float64) - np.asarray(b).astype(np.float64)
            row["mismatched_elements"] = int(np.count_nonzero(difference))
            row["max_absolute_difference"] = float(np.max(np.abs(difference)))
        rows.append(row)
    return {"files": rows, "all_equal": all(row["equal"] for row in rows) and bool(rows)}


def compare_metadata(reference: Path, regenerated: Path) -> dict[str, Any]:
    left, right = reference / "metadata.json", regenerated / "metadata.json"
    if not left.is_file() or not right.is_file():
        return {
            "compared": False,
            "reason": "metadata.json absent from "
            + ("reference" if not left.is_file() else "regenerated"),
            "all_equal": False,
        }
    a, b = _load_json(left), _load_json(right)
    keys = sorted(set(a) | set(b))
    semantic = [key for key in keys if key not in NON_SEMANTIC_METADATA_KEYS]
    differing = [key for key in semantic if a.get(key) != b.get(key)]
    return {
        "compared": True,
        "excluded_timing_keys": sorted(
            key for key in keys if key in NON_SEMANTIC_METADATA_KEYS
        ),
        "compared_keys": semantic,
        "differing_keys": differing,
        "contract_sha256_equal": a.get("contract_sha256") == b.get("contract_sha256"),
        "reference_contract_sha256": a.get("contract_sha256"),
        "regenerated_contract_sha256": b.get("contract_sha256"),
        "all_equal": not differing,
        "differences": {key: {"reference": a.get(key), "regenerated": b.get(key)} for key in differing},
    }


def compare_cell(reference_root: Path, regenerated_root: Path) -> dict[str, Any]:
    """Compare one cache cell: both kinds, arrays and metadata."""

    kinds: dict[str, Any] = {}
    for kind in CACHE_KINDS:
        left, right = reference_root / kind, regenerated_root / kind
        if not left.is_dir():
            kinds[kind] = {"present": False, "reason": "absent from reference capture"}
            continue
        if not right.is_dir():
            kinds[kind] = {"present": False, "reason": "absent after regeneration"}
            continue
        arrays = compare_arrays(left, right)
        metadata = compare_metadata(left, right)
        kinds[kind] = {
            "present": True,
            "arrays": arrays,
            "metadata": metadata,
            "equivalent": arrays["all_equal"] and metadata["all_equal"],
        }
    perturbation = _compare_perturbation(reference_root, regenerated_root)
    # Every kind the reference captured must come back. Skipping absent kinds
    # here would let a regeneration that produced nothing pass by default.
    captured = [kind for kind in CACHE_KINDS if (reference_root / kind).is_dir()]
    equivalent = bool(captured) and all(
        kinds[kind].get("present") and kinds[kind].get("equivalent") for kind in captured
    )
    return {
        "kinds": kinds,
        "reference_kinds": captured,
        "perturbation_contract": perturbation,
        "equivalent": equivalent and perturbation.get("all_equal", True),
    }


def _compare_perturbation(reference_root: Path, regenerated_root: Path) -> dict[str, Any]:
    """``perturbation.json`` carries the intervention contract the runner keys on."""

    left = reference_root / "packed_topology_v1" / "perturbation.json"
    right = regenerated_root / "packed_topology_v1" / "perturbation.json"
    if not left.is_file() or not right.is_file():
        return {"compared": False, "all_equal": True, "reason": "no perturbation.json"}
    a, b = _load_json(left), _load_json(right)
    differing = sorted(
        key
        for key in set(a) | set(b)
        if key not in NON_SEMANTIC_METADATA_KEYS and a.get(key) != b.get(key)
    )
    return {
        "compared": True,
        "contract_sha256_equal": a.get("contract_sha256") == b.get("contract_sha256"),
        "reference_contract_sha256": a.get("contract_sha256"),
        "regenerated_contract_sha256": b.get("contract_sha256"),
        "differing_keys": differing,
        "all_equal": not differing,
    }


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------


def regenerate_cell(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild one cache cell from the frozen clean inputs, into a fresh root."""

    from mp_retrieval.complete_data import load_complete_dataset
    from mp_retrieval.local_topology_perturbations import (
        build_or_load_perturbed_topologies,
    )
    from mp_retrieval.structural_features import build_or_load_structural_features
    from mp_retrieval.topology_store import PackedLocalTopologies
    import hashlib

    # Neither cache builder reads an embedding value, so the regeneration runs
    # on the topology-only load and does not need the two largest files in the
    # dataset root to be present.
    dataset = load_complete_dataset(
        args.data, dataset=args.dataset, require_embeddings=False
    )
    if dataset.metadata["candidate_contract_sha256"] != args.candidate_contract_sha256:
        raise ValueError(
            "Candidate contract differs from the frozen record; refusing to regenerate"
        )
    clean = PackedLocalTopologies.load(args.clean_topology_cache)
    root = args.regenerated_root
    root.mkdir(parents=True, exist_ok=True)

    topologies, intervention = build_or_load_perturbed_topologies(
        clean,
        dataset.queries,
        root / "packed_topology_v1",
        kind=args.axis,
        rate=args.rate,
        seed=args.perturbation_seed,
    )
    feature_fingerprint = hashlib.sha256(
        (
            args.data_fingerprint_sha256
            + intervention["contract_sha256"]
            + "clean_global_static_features"
        ).encode("utf-8")
    ).hexdigest()
    build_or_load_structural_features(
        dataset,
        topologies,
        root / "fixed_structural_features_v1",
        source_fingerprint=feature_fingerprint,
        config=args.feature_config,
    )
    return {
        "intervention_contract_sha256": intervention["contract_sha256"],
        "feature_source_fingerprint_sha256": feature_fingerprint,
    }


def run(args: argparse.Namespace, checkpoint_hook=None) -> dict[str, Any]:
    if args.axis not in TOPOLOGY_AXES:
        result = {
            "status": "CACHE_EQUIVALENCE_NOT_APPLICABLE",
            "dataset": args.dataset,
            "axis": args.axis,
            "rate": args.rate,
            "reason": (
                "feature_mask perturbs node features on device and reuses the "
                "clean caches, so it writes no phase_confirmation_cache cell"
            ),
        }
        _write(args.output, result, checkpoint_hook)
        return result

    if not args.reference_root.is_dir():
        raise SystemExit(
            f"No reference capture at {args.reference_root}; upload the source "
            "cache cell to the quarantine prefix first"
        )
    # A leftover regeneration would be loaded rather than rebuilt, which would
    # make the comparison vacuous.
    if args.regenerated_root.exists() and any(args.regenerated_root.rglob("*")):
        raise SystemExit(
            f"{args.regenerated_root} already holds files; regeneration must start empty"
        )

    produced = regenerate_cell(args)
    if checkpoint_hook is not None:
        checkpoint_hook()
    comparison = compare_cell(args.reference_root, args.regenerated_root)
    result = {
        "status": (
            "CACHE_REGENERATION_EQUIVALENT"
            if comparison["equivalent"]
            else "CACHE_REGENERATION_DIFFERS"
        ),
        "dataset": args.dataset,
        "axis": args.axis,
        "rate": args.rate,
        "perturbation_seed": args.perturbation_seed,
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract_sha256": args.candidate_contract_sha256,
        "reference_root": str(args.reference_root),
        "regenerated_root": str(args.regenerated_root),
        "regeneration": produced,
        "comparison": comparison,
        "equivalence_definition": {
            "arrays": "exact: dtype, shape and every element",
            "metadata": "exact, excluding declared timing keys",
            "excluded_timing_keys": sorted(NON_SEMANTIC_METADATA_KEYS),
            "why_not_byte_identical": (
                "metadata.json records wall-clock build seconds, so two correct "
                "runs of the same computation differ in those fields"
            ),
        },
    }
    _write(args.output, result, checkpoint_hook)
    return result


def _write(output: Path, payload: dict[str, Any], checkpoint_hook=None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    if checkpoint_hook is not None:
        checkpoint_hook()


def feature_contract(screen: dict[str, Any]) -> dict[str, Any]:
    """The dict E2 hands ``build_or_load_structural_features``, from the screen config.

    It is hashed into ``contract_sha256``, and the builder refuses to load a
    cache whose contract disagrees. Asking for anything else -- the whole screen
    config, say -- makes the regeneration build a different cache and the gate
    report a difference that says nothing about determinism.

    Defined here rather than in the launcher so the local run and the container
    run cannot drift apart; ``modal_cache_equivalence`` delegates to this.
    """

    return {
        "retrieval_seeds": screen["retrieval_seeds"],
        "static_features": screen["static_features"],
        "query_local_features": screen["query_local_features"],
        "preprocessing": {"query_chunk_size": 8192},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--axis", required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--perturbation-seed", type=int, required=True)
    parser.add_argument("--clean-topology-cache", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--regenerated-root", type=Path, required=True)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--candidate-contract-sha256", required=True)
    parser.add_argument("--feature-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import yaml

    # --feature-config points at configs/sa_mlp_screen.yaml, the frozen screen
    # config. Only the four keys E2 forwards become the contract; passing the
    # whole file would hash a different dict.
    screen = yaml.safe_load(args.feature_config.read_text(encoding="utf-8"))
    args.feature_config = feature_contract(screen)
    return args


def main() -> int:
    result = run(parse_args())
    print(json.dumps({key: result[key] for key in ("status", "dataset", "axis", "rate")}))
    return 0 if result["status"] != "CACHE_REGENERATION_DIFFERS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
