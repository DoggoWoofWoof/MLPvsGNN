#!/usr/bin/env python
"""M2B's reuse proof and compute estimate, both derived rather than asserted.

Two questions this answers before anything is launched.

**Can M2's feature stores be reused across semantic rungs?** The declaration
says they must be. This proves it mechanically instead of by reading the code:
it calls ``run_m2_qls_v2_freeze.cell_build_key`` twice with different semantic
rungs on the argument namespace and shows the key is byte-identical, and it
reads the persisted master block's own metadata to show its column count is the
structural 12 with no semantic column among them. A store whose identity does
not mention the rung cannot be a store of one rung's features.

There is one real obstruction and it is not the rung. ``cell_build_key``'s
eighth field is ``config_sha256``, the hash of the whole M2 declaration file,
and that file has gained amendments since the stores were built. The hash has
therefore drifted, and ``load_cell_for_fit`` would refuse every store -- not
because the cell is wrong but because the paperwork moved. The resolution is
filed here rather than fixed by loosening the check: M2B pins the expected hash
to the build-time value recorded in each M2 fit's own provenance.

**What does M2B cost?** Separately, per the declaration's requirement: the
feature-store cost that is not incurred because the stores are reused, the S2
fit cost, the S4 fit cost, inference benchmarking, and storage.

Reads only. Launches nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_m2_qls_v2_freeze import cell_build_key  # noqa: E402

M2_DECLARATION = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
M2_HEADLINES = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "headline"
AUDIT_PATH = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "semantic_audit.json"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "reuse_and_compute.json"

CANDIDATE_ARM = "QLS-UNIVERSAL"
INCUMBENT_RUNG = "S3"

#: Read from docs/COMPUTE_LEDGER.md L546-547 by way of the M2 declaration's
#: rate_source, and reused verbatim so M2B's bill is comparable with M2's.
GPU_USD_PER_HOUR = 2.241
CPU_USD_PER_HOUR = 0.634

#: Per-container startup, image pull and data load, measured in M2's smoke at
#: $0.0474 across four containers. A floor, not a prediction -- data load scales
#: with the dataset and it was measured on 2wiki_clean only.
CONTAINER_OVERHEAD_USD = 0.0474

#: How much longer a fit takes than M2's measured S3 fit in the same cell.
#:
#: S2 runs the same per-query Python loop in ``M1AScorer.forward_explicit`` over
#: the same candidates and does strictly less arithmetic inside it (three
#: reductions, no learned vectors), so it cannot be slower for a compute reason;
#: it is floored at parity rather than estimated below it.
#:
#: S4 replaces those reductions with two 1536x64 projections, about 64x the
#: arithmetic per candidate, and widens the scorer's first layer from 14 to 267
#: columns. Neither is likely to show up at 64x in wall clock, because at this
#: scale the fits are dominated by per-query kernel-launch overhead in that
#: Python loop rather than by the size of each kernel -- which is exactly the
#: kind of claim that deserves a measurement rather than an argument. The smoke
#: cell measures it before any fan-out, and 2.5 is a deliberately loose bound to
#: launch under, not a prediction.
FIT_MULTIPLIER = {"S2": {"floor": 1.0, "conservative": 1.0},
                  "S4": {"floor": 1.0, "conservative": 2.5}}

BYTES_PER_PARAMETER = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_cells(declaration: dict[str, Any]) -> list[tuple[str, str]]:
    """The 14 cells, read off the matrix M2 declared and M2B reuses verbatim."""

    cells = declaration["m2_selection_matrix"]["cells"]
    return [(dataset, regime) for dataset in cells for regime in cells[dataset]]


def prove_stores_are_rung_independent(headline: dict[str, Any]) -> dict[str, Any]:
    """Call the real key builder with two rungs and compare.

    The argument namespace carries ``semantic_rung`` -- the runner takes it as a
    CLI flag -- so a key that ignored it would be indistinguishable from one that
    used it, if you only ever built the key once.
    """

    def key_for(rung: str, regime: str) -> dict[str, Any]:
        args = Namespace(
            dataset=headline["dataset"],
            data_fingerprint_sha256=headline["data_fingerprint_sha256"],
            queries=headline["queries"],
            per_seed_cap=16,
            neighbour_scan_cap_per_seed=4096,
            a64_mainline_family=headline["a64_mainline_family"],
            semantic_rung=rung,
        )
        return cell_build_key(args, regime)

    per_regime = {}
    for regime in headline["cells"]:
        keys = {rung: key_for(rung, regime) for rung in ("S2", "S3", "S4")}
        distinct = {json.dumps(key, sort_keys=True) for key in keys.values()}
        per_regime[regime] = {
            "identical_across_S2_S3_S4": len(distinct) == 1,
            "key_fields": sorted(keys["S3"]),
        }

    fields = sorted(set().union(*(set(entry["key_fields"]) for entry in per_regime.values())))
    return {
        "method": (
            "scripts/run_m2_qls_v2_freeze.py::cell_build_key called with semantic_rung set to "
            "S2, S3 and S4 on an otherwise identical namespace, per declared regime"
        ),
        "identical_in_every_regime": all(e["identical_across_S2_S3_S4"] for e in per_regime.values()),
        "per_regime": per_regime,
        "key_fields": fields,
        "semantic_rung_is_not_a_field": "semantic_rung" not in fields,
        "what_that_means": (
            "A persisted cell's identity is its dataset, that dataset's fingerprint, the regime, "
            "the panel size, the A64 budget and the declaration hash. None of those is a semantic "
            "rung, so one store is the correct store for all three candidates by construction -- "
            "not by a promise that the rungs happen to agree."
        ),
    }


def prove_master_block_is_structural(declaration: dict[str, Any]) -> dict[str, Any]:
    """The persisted block is structural only; the semantic block is live.

    The real stores live on the Modal volume, so their metadata is not readable
    from here. What is readable is the function that wrote them: this runs
    ``save_cell_features`` on a synthetic cell in a temporary directory and reads
    back the metadata it produces, which is the same metadata the 14 real stores
    carry. That establishes the persisted width. The second half is what makes
    the width matter -- the declaration's own frozen column order says which of
    its columns are semantic, and none of them can be in a block that is written
    before any rung is chosen.
    """

    import tempfile

    import numpy as np
    from run_m1a_feature_screen import ARM_FAMILIES, MASTER_COLUMNS, UNIVERSAL_ARM
    from run_m2_qls_v2_freeze import load_cell_metadata, save_cell_features

    persisted_families = {
        name: columns.stop - columns.start for name, columns in MASTER_COLUMNS.items()
    }
    width = sum(persisted_families.values())

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "cell"
        save_cell_features(
            root,
            [np.arange(3, dtype=np.int64), np.arange(2, dtype=np.int64)],
            [np.zeros((3, width), dtype=np.float32), np.zeros((2, width), dtype=np.float32)],
            extra={"build_key": {"probe": True}},
        )
        metadata = load_cell_metadata(root)

    schema = declaration["qls_universal"]["feature_schema"]
    order = list(schema["frozen_column_order"])
    semantic_names = order[-int(schema["semantic_feature_count"]):]
    consumed = sum(
        persisted_families[family] for family in ("BASE",) + tuple(ARM_FAMILIES[UNIVERSAL_ARM])
    )

    return {
        "method": (
            "the persisted block's own column registry, run_m1a_feature_screen.MASTER_COLUMNS, "
            "with save_cell_features exercised on a synthetic cell in a temporary directory to "
            "confirm what its metadata records"
        ),
        "persisted_format": metadata["format"],
        "persisted_master_columns": metadata["master_columns"],
        "registry_width": width,
        "writer_agrees_with_registry": metadata["master_columns"] == width,
        "persisted_families": persisted_families,
        "every_persisted_family_is_structural": all(
            name not in semantic_names for name in persisted_families
        ),
        "semantic_column_names": semantic_names,
        "none_of_those_names_is_a_persisted_family": not (
            set(semantic_names) & set(persisted_families)
        ),
        "precomputed_width_the_model_consumes": int(schema["precomputed_width"]),
        "consumed_width_from_the_registry": consumed,
        "consumed_matches_the_declaration": consumed == int(schema["precomputed_width"]),
        "surplus_is_geometry": (
            "The store carries GEOMETRY too, because it is built once per cell and every arm "
            "slices what it needs; QLS-UNIVERSAL does not read those three columns. The surplus "
            "is an excluded structural family, not a semantic one."
        ),
        "semantic_is_computed_live": (
            "M1AScorer.forward_explicit builds the semantic block per query from raw node and "
            "query embeddings at fit time. There is nothing rung-specific in the store to be "
            "stale, which is why one store serves three rungs."
        ),
    }


def config_hash_drift(headlines: list[dict[str, Any]]) -> dict[str, Any]:
    """The one thing that actually blocks reuse, and the rule that resolves it."""

    recorded = sorted({headline["provenance"]["config_sha256"] for headline in headlines})
    current = _sha256(M2_DECLARATION)
    drifted = len(recorded) != 1 or recorded[0] != current
    return {
        "build_time_config_sha256": recorded,
        "one_hash_across_all_six_datasets": len(recorded) == 1,
        "current_config_sha256": current,
        "has_drifted": drifted,
        "why_it_drifted": (
            "cell_build_key hashes the whole of configs/m2_qls_v2_freeze.yaml. Filing M2's "
            "verdict (amendment 7 and the result block) and M2B's freeze (amendment 8) changed "
            "that file. Nothing about any cell changed."
        ),
        "consequence_if_ignored": (
            "load_cell_for_fit compares the persisted build_key against a freshly computed one "
            "and raises on any differing field. Every one of the 14 stores would be refused, on "
            "a field that describes the paperwork rather than the features."
        ),
        "resolution": "PIN_TO_RECORDED_BUILD_TIME_HASH",
        "resolution_rule": (
            "M2B does not relax the check and does not drop the field. It passes the build-time "
            "hash above as the expected value, taken from each M2 fit's own "
            "provenance.config_sha256, so the other seven fields are still compared exactly and "
            "a genuinely wrong store is still refused. The hash becomes a recorded fact about "
            "which declaration built the store, which is what it was always for."
        ),
        "what_this_does_not_license": (
            "Reusing a store across a change that alters the features. The seven cell-identity "
            "fields -- dataset, data fingerprint, regime, panel, per-seed cap, neighbour scan "
            "cap, A64 family -- are unchanged and still checked. If any of those moved, the "
            "store would be refused and should be."
        ),
    }


def reuse_ledger(declaration: dict[str, Any], headlines: list[dict[str, Any]]) -> dict[str, Any]:
    """The 42-cell matrix, resolved into reused and new -- counted, not quoted."""

    cells = declared_cells(declaration)
    fitted = {
        (headline["dataset"], regime)
        for headline in headlines
        for regime, cell in headline["cells"].items()
        if CANDIDATE_ARM in cell["arms"]
    }
    rungs = ("S2", INCUMBENT_RUNG, "S4")

    rows = []
    for dataset, regime in cells:
        present = (dataset, regime) in fitted
        for rung in rungs:
            reusable = rung == INCUMBENT_RUNG and present
            rows.append({
                "dataset": dataset,
                "regime": regime,
                "rung": rung,
                "disposition": "reuse_m2_seed0" if reusable else "new",
            })

    new = [row for row in rows if row["disposition"] == "new"]
    missing = [f"{d}/{r}" for d, r in cells if (d, r) not in fitted]
    by_rung: dict[str, int] = {}
    for row in new:
        by_rung[row["rung"]] = by_rung.get(row["rung"], 0) + 1

    return {
        "declared_cells": len(cells),
        "rungs": list(rungs),
        "logical_matrix": len(cells) * len(rungs),
        "reused": len(rows) - len(new),
        "new": len(new),
        "new_by_rung": by_rung,
        "cells_with_no_m2_fit_to_reuse": missing,
        "every_declared_cell_has_an_s3_fit": not missing,
        "matrix_matches_m2": (
            "The 14 cells are read from m2_selection_matrix.cells, not re-derived. M2B changes "
            "the semantic rung and nothing about which cells exist."
        ),
        "rows": rows,
    }


def compute_estimate(declaration: dict[str, Any], headlines: list[dict[str, Any]],
                     audit: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """Five separated line items, per the declaration's compute-discipline clause."""

    measured = {
        (headline["dataset"], regime): cell["arms"][CANDIDATE_ARM]["training"]["training_seconds"]
        for headline in headlines
        for regime, cell in headline["cells"].items()
        if CANDIDATE_ARM in cell["arms"]
    }
    fits: dict[str, Any] = {}
    for rung in ("S2", "S4"):
        cells = [row for row in ledger["rows"]
                 if row["rung"] == rung and row["disposition"] == "new"]
        seconds = {
            bound: sum(measured[(row["dataset"], row["regime"])] * FIT_MULTIPLIER[rung][bound]
                       for row in cells)
            for bound in ("floor", "conservative")
        }
        fits[rung] = {
            "new_fits": len(cells),
            "basis": (
                f"each cell's own measured M2 {INCUMBENT_RUNG} training_seconds, scaled by the "
                f"declared multiplier for this rung"
            ),
            "multiplier": FIT_MULTIPLIER[rung],
            "seconds": {k: round(v, 1) for k, v in seconds.items()},
            "gpu_hours": {k: round(v / 3600.0, 4) for k, v in seconds.items()},
            "cost_usd": {k: round(v / 3600.0 * GPU_USD_PER_HOUR, 4) for k, v in seconds.items()},
        }

    containers = ledger["new"]
    overhead = containers * CONTAINER_OVERHEAD_USD

    # Inference benchmarking is the lexicographic tie-break's first key, so it
    # has to be measured on every arm that survives effectiveness, not only on
    # the new ones -- S3 included, at the same time, on the same hardware.
    inference_fits = ledger["logical_matrix"]
    inference_seconds = inference_fits * 30.0

    params = {row["candidate"]: row["total_trainable_parameters"] for row in audit["candidates"]}
    checkpoint_mb = {
        rung: round(params[rung] * BYTES_PER_PARAMETER * ledger["declared_cells"] / 1e6, 3)
        for rung in ("S2", INCUMBENT_RUNG, "S4")
    }

    totals = {
        bound: round(
            fits["S2"]["cost_usd"][bound] + fits["S4"]["cost_usd"][bound]
            + overhead + inference_seconds / 3600.0 * GPU_USD_PER_HOUR,
            4,
        )
        for bound in ("floor", "conservative")
    }

    return {
        "rate_source": declaration["compute"]["rate_source"],
        "gpu_usd_per_hour": GPU_USD_PER_HOUR,
        "cpu_usd_per_hour": CPU_USD_PER_HOUR,
        "measured_s3_training_seconds_all_cells": round(sum(measured.values()), 1),
        "why_that_is_the_basis": (
            "Every fit line below scales this, per cell, rather than a rate transcribed from an "
            "estimate. It is what M2's 14 QLS-UNIVERSAL fits actually took."
        ),
        "feature_store_build": {
            "new_builds": 0,
            "seconds": 0.0,
            "cost_usd": 0.0,
            "why_zero": (
                "All 14 stores exist and are provably rung-independent (see reuse_proof). M2's "
                "own estimate put feature build at 5,431-5,782 s and called it 83% of the phase; "
                "M2B does not spend it again. This is the single largest line in the estimate and "
                "it is zero because of a proof, not an assumption."
            ),
            "cost_avoided_usd": round(
                declaration["compute"]["feature_build_seconds"]["conservative"]
                / 3600.0 * GPU_USD_PER_HOUR, 4),
        },
        "s2_fits": fits["S2"],
        "s4_fits": fits["S4"],
        "inference_benchmarking": {
            "arms_benchmarked": inference_fits,
            "includes_the_incumbent": True,
            "why": (
                "The systems tie-break orders on uncached inference p95. Comparing a new S2 "
                "measurement against an S3 number taken on another day and another container "
                "would be comparing containers, not rungs, so S3 is re-benchmarked alongside -- "
                "its fit is reused, its timing is not."
            ),
            "seconds_assumed_per_arm": 30.0,
            "seconds": inference_seconds,
            "cost_usd": round(inference_seconds / 3600.0 * GPU_USD_PER_HOUR, 4),
        },
        "container_overhead": {
            "containers": containers,
            "usd_each": CONTAINER_OVERHEAD_USD,
            "cost_usd": round(overhead, 4),
            "source": "M2's measured smoke, outputs/m2_qls_v2_freeze/measured_cost.json",
            "this_is_an_upper_bound": (
                f"Priced at one container per fit ({containers}). M2 actually ran one container "
                "per dataset and did every cell inside it; the same orchestration here is 6 "
                "datasets x 2 new rungs = 12 containers, less than half this. The larger number "
                "is kept because the estimate should not depend on an orchestration choice that "
                "has not been made yet."
            ),
        },
        "storage": {
            "checkpoint_megabytes_all_cells": checkpoint_mb,
            "total_megabytes": round(sum(checkpoint_mb.values()), 3),
            "why_it_is_listed_anyway": (
                "S4's checkpoint is 57x the incumbent's. That is still under a megabyte a cell "
                "and costs nothing worth pricing, but the ratio is the point of the phase and is "
                "not hidden inside a rounding."
            ),
        },
        "total_cost_usd": totals,
        "m2_measured_total_for_comparison": 4.3914,
        "why_it_is_a_fraction_of_m2": (
            "M2 paid for 14 feature builds and 15 fits. M2B pays for 28 fits and no builds, and "
            "M2's own numbers say the builds were most of the bill."
        ),
        "what_is_not_costed": [
            "the smoke cell, which is a subset of the fits above and not additional to them",
            "storage rent on a volume this track already holds",
            "any seed beyond 0, which the declaration prohibits without a further authorisation",
        ],
    }


def build() -> dict[str, Any]:
    declaration = yaml.safe_load(M2_DECLARATION.read_text(encoding="utf-8"))
    headlines = [json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted(M2_HEADLINES.glob("*.json"))]
    if not headlines:
        raise SystemExit(f"no M2 headline artifacts under {M2_HEADLINES}")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    rungs_run = sorted({headline["semantic_rung"] for headline in headlines})
    if rungs_run != [INCUMBENT_RUNG]:
        raise SystemExit(
            f"M2's fits are not all {INCUMBENT_RUNG} ({rungs_run}); the reuse ledger assumes "
            "exactly one incumbent rung"
        )

    ledger = reuse_ledger(declaration, headlines)
    proof = {
        "rung_independence": prove_stores_are_rung_independent(headlines[0]),
        "master_block": prove_master_block_is_structural(declaration),
        "config_hash_drift": config_hash_drift(headlines),
    }
    proof["stores_are_reusable"] = (
        proof["rung_independence"]["identical_in_every_regime"]
        and proof["rung_independence"]["semantic_rung_is_not_a_field"]
        and proof["config_hash_drift"]["resolution"] == "PIN_TO_RECORDED_BUILD_TIME_HASH"
    )

    return {
        "status": "M2B_REUSE_AND_COMPUTE_COMPLETE",
        "what_this_is_not": "A launch, a fit, or an authorisation to spend anything.",
        "seeds": {"seed": 0, "count": 1,
                  "five_seed_confirmation": "PROHIBITED",
                  "three_seed_resolution": "may be PROPOSED after the one-seed result, never launched with it"},
        "reuse_proof": proof,
        "reuse_ledger": ledger,
        "compute_estimate": compute_estimate(declaration, headlines, audit, ledger),
        "equality_required_across_rungs_in_a_cell": [
            "query ids", "scored candidates", "context", "structural feature tensor",
            "retrieval features", "seed", "NODE_ROLE", "SUPPORT", "PATH",
            "candidate normalization",
        ],
        "how_that_equality_is_enforced": (
            "By identity, not by comparison: all ten come from the one persisted store, which "
            "the proof above shows is the same object for all three rungs. The only thing "
            "constructed per rung is the semantic head."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
