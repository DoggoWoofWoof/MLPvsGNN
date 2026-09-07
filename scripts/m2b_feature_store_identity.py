"""Can M2B reuse M2's fourteen persisted cell masters, and on what authority?

M2's ``load_cell_for_fit`` compares a store's recorded ``build_key`` against
the key the current process would build, and one of that key's eight fields is
the SHA-256 of the whole declaration YAML. M2B amended that YAML -- to record
which model M2 selected -- so the hash moved and all fourteen stores would now
be refused. Nothing about those tensors changed.

The cheap fix is to pin the old hash. This script implements the other one:
``scripts/feature_build_contract.py`` defines a ``feature_build_contract_sha256``
over only the inputs that can change a cell tensor, and this script proves, per
store, that

* the contract reconstructed BACKWARD from what that build actually recorded
* equals the contract built FORWARD from the launcher's real arguments and the
  live formula constants,

which is the condition for treating the reconstruction as that store's identity
rather than as a convenient re-labelling. It also checks that the formula
constants substituted into the reconstruction were the same at each store's own
build commit, and reports the paperwork hash drift the whole exercise exists to
stop mattering.

Nothing here writes to the volume. Each store's original ``config_sha256`` is
read, carried forward under ``original_full_config_sha256``, and left alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import feature_build_contract as fbc  # noqa: E402
from scripts import modal_m2_qls_v2_freeze as launcher  # noqa: E402

M2_DECLARATION = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
BUILD_ARTIFACT_DIR = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "build"
STORE_METADATA_DIR = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "reference" / "m2_cell_metadata"
)
OUTPUT_PATH = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "feature_store_identity.json"
)

#: The declaration hash every M2 store was built under. Recorded here so the
#: drift this script exists to neutralise is stated rather than implied; it is
#: never compared against anything.
M2_BUILD_TIME_CONFIG_SHA256 = (
    "d9518792075054612f70cd77d710c4af71f70407c287e8b883b87e68b84747a1"
)

VERDICT_PERMITTED = "FEATURE_STORE_REUSE_PERMITTED"
VERDICT_FORBIDDEN = "FEATURE_STORE_REUSE_FORBIDDEN"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _current_config_sha256() -> str:
    return hashlib.sha256(M2_DECLARATION.read_bytes()).hexdigest()


def store_metadata_paths() -> list[tuple[str, str, Path]]:
    """The (dataset, regime, path) of every downloaded store metadata file."""

    rows = []
    for path in sorted(STORE_METADATA_DIR.glob("*__*.json")):
        dataset, regime = path.stem.split("__", 1)
        rows.append((dataset, regime, path))
    return rows


def declared_cells() -> list[tuple[str, str]]:
    """Which (dataset, regime) cells M2 actually built, from the build artifacts."""

    cells = []
    for path in sorted(BUILD_ARTIFACT_DIR.glob("*.json")):
        artifact = _load_json(path)
        for regime in artifact["cells"]:
            cells.append((artifact["dataset"], regime))
    return sorted(cells)


def _field_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    )


def audit_store(dataset: str, regime: str, metadata_path: Path) -> dict[str, Any]:
    """One store: both directions, the constants check, and the drift record."""

    store = _load_json(metadata_path)
    artifact = _load_json(BUILD_ARTIFACT_DIR / f"{dataset}.json")

    backward = fbc.reconstruct_m2_contract(store_metadata=store, build_artifact=artifact)

    # The forward direction runs the launcher's own argument construction, so
    # "what current code would build" is the real code path rather than a
    # transcription of it. The candidate hashes come from what this build
    # measured, because measuring them again needs the corpus, which is on the
    # volume -- see candidate_hashes_are_from_the_recorded_build below.
    job = launcher._jobs([dataset])[0]
    args = launcher._runner_args(job, stage="headline")
    provenance = artifact["provenance"]
    forward = fbc.contract_from_runner_args(
        args,
        regime,
        candidate_contract_sha256=provenance["candidate_contract_sha256"],
        candidate_id_order_sha256=provenance["candidate_id_order_sha256"],
        store_format=store["format"],
        master_columns=store["master_columns"],
        master_dtype=store["master_dtype"],
    )

    differences = _field_differences(backward, forward)
    constants = fbc.formula_constants_unchanged_since(store["source_commit"])
    identity = fbc.m2_store_identity(store_metadata=store, build_artifact=artifact)

    return {
        "dataset": dataset,
        "regime": regime,
        "feature_build_contract_sha256": fbc.contract_sha256(backward),
        "forward_contract_sha256": fbc.contract_sha256(forward),
        "directions_agree": fbc.contract_sha256(backward) == fbc.contract_sha256(forward),
        "differing_fields": differences,
        "original_full_config_sha256": identity["original_full_config_sha256"],
        "store_fingerprint_sha256": store["fingerprint_sha256"],
        "source_commit": store["source_commit"],
        "formula_constants_unchanged_since_build": constants["all_unchanged"],
        "formula_constants": constants["per_constant"],
        "contract": backward,
    }


def build_report() -> dict[str, Any]:
    declared = declared_cells()
    downloaded = [(dataset, regime) for dataset, regime, _path in store_metadata_paths()]
    missing = sorted(set(declared) - set(downloaded))
    extra = sorted(set(downloaded) - set(declared))

    stores = [
        audit_store(dataset, regime, path)
        for dataset, regime, path in store_metadata_paths()
    ]

    failures: list[str] = []
    if missing:
        failures.append(f"no store metadata for {missing}")
    if extra:
        failures.append(f"store metadata for undeclared cells {extra}")
    for entry in stores:
        cell = f"{entry['dataset']}/{entry['regime']}"
        if not entry["directions_agree"]:
            failures.append(f"{cell}: forward and backward contracts differ on {entry['differing_fields']}")
        if not entry["formula_constants_unchanged_since_build"]:
            failures.append(f"{cell}: a formula constant changed since {entry['source_commit']}")

    # Every store must also agree with every other on the parts of the contract
    # that are cell-independent. If two stores disagreed about the formula
    # version or the column layout they were not built by the same code, and a
    # per-store check alone would not notice.
    shared_fields = (
        "contract_version", "feature_formula_version", "feature_builder",
        "context_arm", "feature_damping", "feature_ppr_iterations",
        "feature_normalisation", "master_column_layout", "master_column_count",
        "store_format", "master_columns", "master_dtype",
        "query_split", "query_selection",
    )
    shared = {
        field: sorted({json.dumps(entry["contract"][field], sort_keys=True) for entry in stores})
        for field in shared_fields
    }
    for field, values in shared.items():
        if len(values) != 1:
            failures.append(f"stores disagree about {field}: {values}")

    current = _current_config_sha256()
    return {
        "status": "M2B_FEATURE_STORE_IDENTITY_COMPLETE",
        "question": (
            "May M2B reuse M2's fourteen persisted cell masters, given that the "
            "declaration hash their build_key records no longer matches the declaration?"
        ),
        "cells_declared": len(declared),
        "cells_audited": len(stores),
        "the_problem": {
            "m2_build_key_field": "config_sha256",
            "what_it_hashes": "the entire configs/m2_qls_v2_freeze.yaml file",
            "recorded_in_every_store": M2_BUILD_TIME_CONFIG_SHA256,
            "the_same_file_now": current,
            "drifted": current != M2_BUILD_TIME_CONFIG_SHA256,
            "why_it_drifted": (
                "M2B added amendment 8 and a selected: block recording which model M2 "
                "chose. That is a statement about authorship and statistics; it cannot "
                "move a float in a structural feature tensor."
            ),
            "why_pinning_the_old_hash_was_rejected": (
                "It works once. M3, M4 and the canonical migration each amend a "
                "declaration too, and each would hit the same refusal and be tempted "
                "to pin again, until the identity check certifies nothing."
            ),
        },
        "the_replacement": {
            "field": "feature_build_contract_sha256",
            "defined_in": "scripts/feature_build_contract.py",
            "hashes_only": sorted(
                set(stores[0]["contract"]) if stores else set()
            ),
            "excludes": [
                "config_sha256", "source_commit", "seed", "epochs", "learning_rate",
                "batch_size", "dropout", "temperature", "semantic_rung",
                "compute ceilings", "authorisation", "selection wording",
            ],
            "why_semantic_rung_is_excluded": (
                "The master block is structural. cell_build_key never carried the rung "
                "either, which is what makes one store serve S2, S3 and S4 alike -- "
                "proved separately by scripts/m2b_reuse_and_compute.py."
            ),
            "both_hashes_kept": (
                "original_full_config_sha256 travels beside it as provenance. Nothing "
                "in this repository rewrites a store's recorded config_sha256, and this "
                "script never writes to the volume at all."
            ),
        },
        "formula_version_is_defended": {
            "version": fbc.FEATURE_FORMULA_VERSION,
            "pinned_behaviour_sha256": fbc.pinned_formula_behaviour(),
            "how": (
                "tests/test_feature_build_contract.py runs the real _cell_master_local "
                "over a deterministic toy cell in R1/R2/R3 and requires the digest to "
                "match the pin, so a changed formula cannot keep an unchanged version "
                "string."
            ),
        },
        "candidate_hashes_are_from_the_recorded_build": (
            "validate_candidate_contract measures the corpus, which lives on the Modal "
            "volume, so this local audit passes the hashes that build recorded into "
            "both directions. That closes the declaration-side gap, not the data-side "
            "one: the M2B runner recomputes the contract in the container from the "
            "candidate contract it measures there, and refuses any store whose "
            "reconstructed hash differs."
        ),
        "shared_contract_fields": {
            field: json.loads(values[0]) if len(values) == 1 else values
            for field, values in shared.items()
        },
        "stores": stores,
        "failed_checks": failures,
        "verdict": VERDICT_PERMITTED if not failures else VERDICT_FORBIDDEN,
        "what_the_verdict_means": (
            "M2B may load the fourteen persisted cell masters under their reconstructed "
            "contract hashes; the feature build stays at $0."
            if not failures else
            "M2B may not reuse the persisted masters. The feature build must be repaid "
            "and the compute estimate recomputed before anything launches."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {
        key: report[key]
        for key in ("status", "cells_declared", "cells_audited", "failed_checks", "verdict")
    }
    summary["contract_hashes"] = {
        f"{entry['dataset']}/{entry['regime']}": entry["feature_build_contract_sha256"][:16]
        for entry in report["stores"]
    }
    summary["declaration_hash_drifted"] = report["the_problem"]["drifted"]
    print(json.dumps(summary, indent=2))
    return 0 if report["verdict"] == VERDICT_PERMITTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
