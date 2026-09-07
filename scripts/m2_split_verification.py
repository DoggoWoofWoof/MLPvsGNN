#!/usr/bin/env python
"""Did the CPU-build / GPU-fit split reproduce the one-container run on real hardware?

configs/m2_qls_v2_freeze.yaml#launch_authorization.smoke_spec.split_verification
is the last piece of the step-D evidence chain, and the only piece
scripts/m2_feature_build_equivalence.py cannot supply. That probe runs where
torch is a CPU build, so its cross-process stage compares CPU against CPU; it
says so in its own what_this_cannot_prove. Here the two sides are two Modal
containers on the declared smoke cell -- one with an A10G, one with no
accelerator at all -- which is the comparison the deployment actually makes.

Reads three artifacts already downloaded from the volume:

  outputs/m2_qls_v2_freeze/smoke/<dataset>.json        one container, --stage full
  outputs/m2_qls_v2_freeze/smoke_build/<dataset>.json  CPU container, --stage build
  outputs/m2_qls_v2_freeze/smoke_fit/<dataset>.json    GPU container, --stage fit

and writes outputs/m2_qls_v2_freeze/split_verification.json.

The bar is the declaration's, not a softer one: the split fit's feature-store
fingerprint and its metrics must EQUAL the one-container run's. A fingerprint is
a sha256 over the exact arrays the trainer consumes, so equal fingerprints is
bit-exactness, and equal metrics is what bit-exactness has to produce. Timings
and memory are excluded on purpose -- they are properties of the hardware, not
of the features, and requiring them to match would fail a run that is correct.

Spends no compute and launches nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "m2_qls_v2_freeze"
OUTPUT_PATH = OUTPUT_ROOT / "split_verification.json"

#: Per-cell fields that must match. Everything here is a property of the
#: features or the panel; nothing here is a property of the container.
CELL_FIELDS = (
    "regime",
    "regime_headroom",
    "scored_node_count",
    "train_queries",
    "held_out_queries",
    "arms_run",
    "cell_features_fingerprint_sha256",
)

#: Per-arm fields that must match, and where each lives.
ARM_FIELDS = (
    ("arm", ()),
    ("runner_arm", ()),
    ("regime", ()),
    ("seed", ()),
    ("semantic_rung", ()),
    ("precomputed_width", ()),
    ("parameters", ()),
    ("metrics", ()),
    ("ceiling_attainment_at_5", ()),
    ("matrix_status", ()),
    ("feature_store_fingerprint_sha256", ("instrumentation",)),
    ("candidate_ids_sha256", ("instrumentation",)),
    ("candidate_contract_sha256", ("instrumentation",)),
    ("dataset_fingerprint_sha256", ("instrumentation",)),
    ("config_sha256", ("instrumentation",)),
    ("aggregate_metrics_reconstructed_from_rows", ("instrumentation",)),
    ("query_ids", ("instrumentation",)),
    ("parameter_counts_total_semantic_scorer", ("instrumentation",)),
)


def _at(node: dict[str, Any], path: tuple[str, ...], key: str) -> Any:
    for step in path:
        node = node[step]
    return node[key]


def _load(dataset: str, stage: str) -> dict[str, Any]:
    path = OUTPUT_ROOT / stage / f"{dataset}.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing -- run the {stage} stage and download it before verifying"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def compare(whole: dict[str, Any], build: dict[str, Any], fit: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []

    def check(name: str, left: Any, right: Any) -> None:
        same = left == right
        rows.append({"field": name, "equal": same, "one_container": left, "split": right})
        if not same:
            mismatches.append(name)

    # The two halves must have run the same cell, or the comparison is empty.
    check("dataset", whole["dataset"], fit["dataset"])
    check("data_fingerprint_sha256", whole["data_fingerprint_sha256"], fit["data_fingerprint_sha256"])
    check("queries", whole["queries"], fit["queries"])
    check("seed", whole["seed"], fit["seed"])
    check("semantic_rung", whole["semantic_rung"], fit["semantic_rung"])
    check("holdout_fraction", whole["holdout_fraction"], fit["holdout_fraction"])
    check("cells_run", sorted(whole["cells"]), sorted(fit["cells"]))

    for regime in sorted(fit["cells"]):
        left_cell, right_cell = whole["cells"][regime], fit["cells"][regime]
        for field in CELL_FIELDS:
            check(f"cells.{regime}.{field}", left_cell[field], right_cell[field])
        for arm in sorted(right_cell["arms"]):
            left_arm, right_arm = left_cell["arms"][arm], right_cell["arms"][arm]
            for key, path in ARM_FIELDS:
                label = ".".join(("cells", regime, "arms", arm, *path, key))
                check(label, _at(left_arm, path, key), _at(right_arm, path, key))

    # The fit stage must be reporting the BUILD container's measurement, not one
    # it took itself -- otherwise a load timing is filed under a build's name.
    latency_carried = {
        regime: fit["cells"][regime]["uncached_feature_build_latency_ms"]
        == build["cells"][regime]["uncached_feature_build_latency_ms"]
        for regime in sorted(fit["cells"])
    }

    return {
        "rows": rows,
        "mismatches": mismatches,
        "fields_compared": len(rows),
        "build_latency_carried_from_the_building_container": latency_carried,
        "every_field_equal": not mismatches,
    }


def verify(dataset: str) -> dict[str, Any]:
    declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    spec = declaration["launch_authorization"]["smoke_spec"]
    split_spec = spec["split_verification"]
    whole = _load(dataset, "smoke")
    build = _load(dataset, "smoke_build")
    fit = _load(dataset, "smoke_fit")

    if whole["feature_build_stage"] != "full":
        raise SystemExit(f"the one-container artifact reports stage {whole['feature_build_stage']!r}")
    if build["feature_build_stage"] != "build" or fit["feature_build_stage"] != "fit":
        raise SystemExit("the split artifacts are not a build and a fit")
    if dataset != spec["primary"]["dataset"]:
        raise SystemExit(f"split_verification.cell is SAME_AS_PRIMARY, which is {spec['primary']['dataset']!r}")

    comparison = compare(whole, build, fit)
    built_nothing = all(cell["arms"] == {} for cell in build["cells"].values())
    latency_carried = all(comparison["build_latency_carried_from_the_building_container"].values())

    checks = {
        "cpu_container_writes_the_master_and_trains_nothing": (
            built_nothing and build["status"] == "M2_QLS_V2_FREEZE_FEATURE_BUILD_COMPLETE"
        ),
        "gpu_container_trains_from_the_persisted_master": (
            fit["status"] == "M2_QLS_V2_FREEZE_DATASET_COMPLETE"
            and any(cell["arms"] for cell in fit["cells"].values())
        ),
        "split_store_fingerprint_equals_the_one_container_run_s": not [
            name for name in comparison["mismatches"] if name.endswith("fingerprint_sha256")
        ],
        "split_metrics_equal_the_one_container_run_s": not [
            name for name in comparison["mismatches"] if ".metrics" in name
        ],
    }
    declared = list(split_spec["verifies"])
    missing = [name for name in declared if name not in checks]
    if missing:
        raise SystemExit(f"the declaration asks for checks this script does not run: {missing}")

    passed = comparison["every_field_equal"] and all(checks[name] for name in declared)
    return {
        "status": "M2_SPLIT_VERIFICATION_COMPLETE",
        "declaration": "configs/m2_qls_v2_freeze.yaml",
        "question": (
            "does the CPU-build / GPU-fit split reproduce the one-container run on real "
            "Modal hardware, bit-exactly?"
        ),
        "verdict": "REPRODUCED" if passed else "NOT_REPRODUCED",
        "reproduced": passed,
        "dataset": dataset,
        "cell": {"regime": spec["primary"]["regime"], "arms": list(spec["primary"]["arms"]),
                 "queries": spec["primary"]["queries"]},
        "containers": {
            "one_container": {"stage": whole["feature_build_stage"], "accelerator": "A10G"},
            "build": {"stage": build["feature_build_stage"], "accelerator": None},
            "fit": {"stage": fit["feature_build_stage"], "accelerator": "A10G"},
        },
        "declared_checks": {name: checks[name] for name in declared},
        "fields_compared": comparison["fields_compared"],
        "mismatches": comparison["mismatches"],
        "build_latency_carried_from_the_building_container": (
            comparison["build_latency_carried_from_the_building_container"]
        ),
        "build_latency_carried": latency_carried,
        "excluded_from_the_comparison": (
            "Wall-clock timings, GPU and CPU memory, and the artifact paths. These are "
            "properties of the container, not of the features -- the split's whole point is "
            "that they differ. What must not differ is anything the features or the metrics "
            "depend on, and that is what is compared above."
        ),
        "if_this_fails": (
            "feature_build_compute_check.result and launch_authorization.smoke_spec."
            "split_verification.passes_only_if both say the same thing: the split is retired "
            "and the headline runs --stage full, which remains the runner's default and stays "
            "tested. A mismatch retires the split, not the phase."
        ),
        "comparison": comparison["rows"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="defaults to the declared smoke dataset")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    dataset = args.dataset
    if dataset is None:
        declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
        dataset = declaration["launch_authorization"]["smoke_spec"]["primary"]["dataset"]

    report = verify(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        key: report[key]
        for key in ("status", "verdict", "reproduced", "dataset", "fields_compared",
                    "mismatches", "declared_checks", "build_latency_carried")
    }, indent=2))
    return 0 if report["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
