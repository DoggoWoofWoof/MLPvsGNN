#!/usr/bin/env python
"""Did the M2 smoke verify what the declaration said it would verify?

configs/m2_qls_v2_freeze.yaml#launch_authorization.gates.engineering_smoke_passes
is earned by launch_authorization.smoke_spec, which lists ten named items across
a primary R2 cell and a secondary R3 one. Reading those off a run log and
asserting them in conversation is exactly the failure
instrumentation_requirement.also_applies_to_analysis_products exists to
prevent, so this script checks each declared item mechanically and writes the
answer to disk.

Five of the ten are settled by the result JSON the run already writes. Five are
claims about the FEATURE COLUMNS, and no result JSON can settle those -- they
need the persisted store. So this script reads the stores the smoke wrote and
binds them to the run by fingerprint: an arm store is accepted only when its
metadata's fingerprint_sha256 equals the fingerprint the result recorded, and a
cell master only when it equals the cell's own. Where the files sit locally is
therefore irrelevant to the evidence; a store from any other run is refused.

Column indices are derived from run_m1a_feature_screen.MASTER_COLUMNS and
ARM_FAMILIES -- the same two objects the composer uses -- and never from the
store's local_feature_names, which carries ten names for a nine-column array
(structural_features.py's own 10-column vocabulary, kept for the raw block).

Writes outputs/m2_qls_v2_freeze/smoke_verification.json. Spends no compute.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.run_m1a_feature_screen import (  # noqa: E402
    ARM_FAMILIES,
    MASTER_COLUMNS,
    UNIVERSAL_ARM,
)

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "m2_qls_v2_freeze"
OUTPUT_PATH = OUTPUT_ROOT / "smoke_verification.json"

ARM = "QLS-UNIVERSAL"
SEMANTIC_WIDTH_S3 = 5
HEX64 = re.compile(r"^[0-9a-f]{64}$")

#: Where each declared verifies item is settled. "result" items need only the
#: run's own JSON; "store" items are claims about the columns and need the
#: persisted arrays. A declared item absent from here stops the check rather
#: than passing silently -- see verify().
SETTLED_BY: dict[str, str] = {
    "total_parameters_equal_3585": "result",
    "checkpoint_written": "result",
    "per_query_rows_written": "result",
    "aggregate_reconstructed_from_rows": "result",
    "feature_store_fingerprint_recorded": "result",
    "fourteen_column_schema": "store",
    "node_role_column_present": "store",
    "node_role_identically_zero": "store",
    "support_and_path_columns_active": "store",
    "node_role_nonzero_exactly_on_c3_minus_cq": "store",
}


def arm_column_index() -> dict[str, int | slice]:
    """Where each family lands in the composed 9-column arm block.

    _compose stacks BASE first, then ARM_FAMILIES[arm] in its declared order,
    so the offsets follow from those two objects and cannot drift from them.
    """

    offsets: dict[str, int | slice] = {}
    cursor = 0
    for family in ("BASE", *ARM_FAMILIES[UNIVERSAL_ARM]):
        span = MASTER_COLUMNS[family]
        width = span.stop - span.start
        offsets[family] = cursor if width == 1 else slice(cursor, cursor + width)
        cursor += width
    offsets["_width"] = cursor
    return offsets


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _store(root: Path, expected_fingerprint: str, kind: str) -> tuple[dict[str, Any], Path]:
    """A store is evidence only if it is provably the one the run wrote."""

    if not root.is_dir():
        raise SystemExit(f"{root} is missing; pull the {kind} store the smoke wrote")
    metadata = _load_json(root / "metadata.json")
    actual = metadata.get("fingerprint_sha256")
    if actual != expected_fingerprint:
        raise SystemExit(
            f"{root} is not the {kind} store this run wrote: its fingerprint is {actual}, "
            f"the result recorded {expected_fingerprint}"
        )
    return metadata, root


def _ragged(root: Path) -> list[np.ndarray]:
    flat = np.load(root / "scored_flat.npy")
    offsets = np.load(root / "scored_offsets.npy")
    return [flat[offsets[i]:offsets[i + 1]] for i in range(offsets.size - 1)]


def check_result_items(result: dict[str, Any], regime: str) -> dict[str, dict[str, Any]]:
    cell = result["cells"][regime]
    arm = cell["arms"][ARM]
    params = arm["parameters"]
    instrumentation = arm["instrumentation"]
    fingerprint = instrumentation.get("feature_store_fingerprint_sha256", "")
    return {
        "total_parameters_equal_3585": {
            "passed": params == {"total": 3585, "semantic": 3072, "scorer": 513},
            "observed": params,
        },
        "checkpoint_written": {
            "passed": bool(instrumentation.get("checkpoint")),
            "observed": instrumentation.get("checkpoint"),
        },
        "per_query_rows_written": {
            "passed": bool(instrumentation.get("per_query_rows"))
            and int(instrumentation.get("query_ids", 0)) > 0,
            "observed": {
                "path": instrumentation.get("per_query_rows"),
                "query_ids": instrumentation.get("query_ids"),
                "candidate_ids_sha256": instrumentation.get("candidate_ids_sha256"),
            },
        },
        "aggregate_reconstructed_from_rows": {
            "passed": instrumentation.get("aggregate_metrics_reconstructed_from_rows") is True,
            "observed": instrumentation.get("aggregate_metrics_reconstructed_from_rows"),
        },
        "feature_store_fingerprint_recorded": {
            "passed": bool(HEX64.match(fingerprint or "")),
            "observed": fingerprint,
        },
    }


def check_store_items(primary_arm: Path, primary_cell: Path, secondary_arm: Path,
                      secondary_cell: Path, arm_result: dict[str, Any]) -> dict[str, Any]:
    index = arm_column_index()
    node_role = index["NODE_ROLE"]
    support = index["SUPPORT"]
    path_span = index["PATH"]

    local = np.load(primary_arm / "local.npy")
    scorer = arm_result["parameters"]["scorer"]
    # scorer = 32*width + 65 -- the one place the SCORER's own input width is
    # observable, so the 14 is read back rather than asserted.
    implied_width = (scorer - 65) / 32
    role_column = local[:, node_role]
    support_column = local[:, support]
    path_block = local[:, path_span]

    cq_sets = _ragged(primary_cell)      # R2 scored == Cq
    c3_sets = _ragged(secondary_cell)    # R3 scored == C3
    r3_master = np.load(secondary_cell / "master_flat.npy")
    r3_offsets = np.load(secondary_cell / "scored_offsets.npy")
    master_role = MASTER_COLUMNS["NODE_ROLE"].start

    exact = 0
    admitted = 0
    observed_nonzero = 0
    mismatched: list[int] = []
    for i, (cq, c3) in enumerate(zip(cq_sets, c3_sets)):
        role = r3_master[r3_offsets[i]:r3_offsets[i + 1], master_role]
        expected = ~np.isin(c3, cq)
        seen = role > 0.5
        admitted += int(expected.sum())
        observed_nonzero += int(seen.sum())
        if np.array_equal(expected, seen):
            exact += 1
        else:
            mismatched.append(i)

    r3_local = np.load(secondary_arm / "local.npy")
    r3_arm_role = r3_local[:, node_role].astype(np.float32)
    r3_master_role = r3_master[:, master_role].astype(np.float32)

    return {
        "fourteen_column_schema": {
            "passed": local.shape[1] == 9
            and implied_width == 14
            and 9 + SEMANTIC_WIDTH_S3 == 14,
            "observed": {
                "persisted_precomputed_columns": int(local.shape[1]),
                "persisted_dtype": str(local.dtype),
                "semantic_columns_at_s3": SEMANTIC_WIDTH_S3,
                "scorer_parameters": scorer,
                "width_implied_by_scorer_32w_plus_65": implied_width,
            },
        },
        "node_role_column_present": {
            "passed": isinstance(node_role, int) and node_role < local.shape[1],
            "observed": {
                "column_index": node_role,
                "derived_from": "run_m1a_feature_screen.ARM_FAMILIES[UNIVERSAL_ARM] order",
                "arm_in_store_metadata": _load_json(primary_arm / "metadata.json")["arm"],
            },
        },
        "node_role_identically_zero": {
            "passed": bool(np.count_nonzero(role_column) == 0),
            "observed": {
                "regime": "R2",
                "rows": int(local.shape[0]),
                "nonzero_rows": int(np.count_nonzero(role_column)),
            },
        },
        "support_and_path_columns_active": {
            "passed": bool(np.count_nonzero(support_column) > 0)
            and bool(np.count_nonzero(path_block) > 0),
            "observed": {
                "support_nonzero_rows": int(np.count_nonzero(support_column)),
                "support_of_rows": int(local.shape[0]),
                "path_nonzero_entries": int(np.count_nonzero(path_block)),
                "path_of_entries": int(path_block.size),
            },
        },
        "node_role_nonzero_exactly_on_c3_minus_cq": {
            "passed": exact == len(c3_sets) and len(c3_sets) > 0 and admitted > 0,
            "observed": {
                "regime": "R3",
                "queries": len(c3_sets),
                "queries_exact": exact,
                "queries_mismatched": mismatched[:10],
                "rows_in_c3_minus_cq": admitted,
                "rows_with_node_role_nonzero": observed_nonzero,
                "why_the_two_scored_sets_are_cq_and_c3": (
                    "run_m1a_feature_screen._cell_master_local persists scored = unique(pool) "
                    "under R2, which IS Cq, and scored = unique(C3) under R3. The claim is "
                    "therefore a set difference over persisted arrays, with nothing inferred."
                ),
            },
        },
        "_additional_findings": {
            "arm_node_role_column_equals_the_masters": bool(
                np.array_equal(r3_arm_role, r3_master_role)
            ),
            "r3_node_role_is_binary": bool(
                np.isin(np.unique(r3_master_role), (0.0, 1.0)).all()
            ),
            "r2_master_node_role_nonzero_rows": int(
                np.count_nonzero(np.load(primary_cell / "master_flat.npy")[:, master_role])
            ),
            "note": (
                "Not declared verifies items. Recorded because the slice, not just the master, "
                "is what a fit reads: a correct master sliced at the wrong offset would pass "
                "every declared item above and still train on the wrong column."
            ),
        },
    }


def verify(store_root: Path, dataset: str) -> dict[str, Any]:
    declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    spec = declaration["launch_authorization"]["smoke_spec"]
    primary_spec = spec["primary"]
    secondary_spec = spec["secondary_only_if_needed"]
    declared = list(primary_spec["verifies"]) + list(secondary_spec["verifies"])

    unimplemented = [item for item in declared if item not in SETTLED_BY]
    if unimplemented:
        raise SystemExit(
            "the declaration asks the smoke to verify "
            f"{unimplemented}, which this script does not check"
        )

    primary = _load_json(OUTPUT_ROOT / "smoke" / f"{dataset}.json")
    secondary = _load_json(OUTPUT_ROOT / "secondary_smoke" / f"{dataset}.json")
    primary_regime = primary_spec["regime"]
    secondary_regime = secondary_spec["regime"]
    for artifact, regime, label in ((primary, primary_regime, "primary"),
                                    (secondary, secondary_regime, "secondary")):
        if regime not in artifact["cells"]:
            raise SystemExit(f"the {label} smoke did not run {regime}")
        if artifact.get("queries") != primary_spec["queries"]:
            raise SystemExit(
                f"the {label} smoke ran {artifact.get('queries')} queries, "
                f"the declaration says {primary_spec['queries']}"
            )

    primary_arm_result = primary["cells"][primary_regime]["arms"][ARM]
    secondary_arm_result = secondary["cells"][secondary_regime]["arms"][ARM]
    stores = {
        "primary_arm": _store(
            store_root / "primary" / "arm",
            primary_arm_result["instrumentation"]["feature_store_fingerprint_sha256"],
            "primary arm",
        ),
        "primary_cell": _store(
            store_root / "primary" / "cell",
            primary["cells"][primary_regime]["cell_features_fingerprint_sha256"],
            "primary cell",
        ),
        "secondary_arm": _store(
            store_root / "secondary" / "arm",
            secondary_arm_result["instrumentation"]["feature_store_fingerprint_sha256"],
            "secondary arm",
        ),
        "secondary_cell": _store(
            store_root / "secondary" / "cell",
            secondary["cells"][secondary_regime]["cell_features_fingerprint_sha256"],
            "secondary cell",
        ),
    }

    checks: dict[str, Any] = {}
    checks.update(check_result_items(primary, primary_regime))
    store_checks = check_store_items(
        stores["primary_arm"][1], stores["primary_cell"][1],
        stores["secondary_arm"][1], stores["secondary_cell"][1],
        primary_arm_result,
    )
    additional = store_checks.pop("_additional_findings")
    checks.update(store_checks)

    for name, check in checks.items():
        check["declared_by"] = (
            "primary" if name in primary_spec["verifies"] else "secondary_only_if_needed"
        )
        check["settled_by"] = SETTLED_BY[name]

    failed = sorted(name for name, check in checks.items() if not check["passed"])
    passed = not failed

    return {
        "status": "M2_SMOKE_VERIFICATION_COMPLETE",
        "declaration": "configs/m2_qls_v2_freeze.yaml",
        "question": "did the smoke verify every item launch_authorization.smoke_spec declares?",
        "verdict": "PASSED" if passed else "FAILED",
        "passed": passed,
        "failed_items": failed,
        "dataset": dataset,
        "cells": {
            "primary": {"regime": primary_regime, "arms": [ARM],
                        "queries": primary["queries"]},
            "secondary": {"regime": secondary_regime, "arms": [ARM],
                          "queries": secondary["queries"]},
        },
        "declared_items": declared,
        "items_checked": len(checks),
        "checks": checks,
        "additional_findings": additional,
        "stores_bound_by_fingerprint": {
            key: {
                "fingerprint_sha256": metadata["fingerprint_sha256"],
                "format": metadata["format"],
                "shape": metadata.get("local_shape")
                or [metadata.get("queries"), metadata.get("master_columns")],
            }
            for key, (metadata, _) in stores.items()
        },
        "why_fingerprints_and_not_paths": (
            "The store files are pulled off the Modal volume to be read, so a local path proves "
            "nothing about which run produced them. Each store's own recorded fingerprint is "
            "checked against the fingerprint the result JSON recorded for that cell and arm, and "
            "a mismatch stops this script. That binding is what makes these column claims "
            "evidence about the smoke rather than about some array on this machine."
        ),
        "scientific_scope": (
            "None. smoke_spec.purpose is PIPELINE_VALIDATION_ONLY_NO_SCIENTIFIC_CONCLUSION and "
            "no number here feeds scientific_questions, qls_cell or universal_selection_rule."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-root", type=Path, required=True,
        help="directory holding primary/{arm,cell} and secondary/{arm,cell}, pulled from Modal",
    )
    parser.add_argument("--dataset", default=None, help="defaults to the declared smoke dataset")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    dataset = args.dataset
    if dataset is None:
        declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
        dataset = declaration["launch_authorization"]["smoke_spec"]["primary"]["dataset"]

    report = verify(args.store_root, dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "verdict": report["verdict"],
        "items_checked": report["items_checked"],
        "failed_items": report["failed_items"],
        "checks": {name: check["passed"] for name, check in report["checks"].items()},
        "additional_findings": report["additional_findings"],
    }, indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
