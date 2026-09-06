#!/usr/bin/env python
"""Re-estimate for amendment 5: all 27 logical M1B fits re-run with row capture.

A new, separate estimate -- scripts/m1b_compute_estimate.py is not edited,
since it is the filed record of what justified the original
compute_within_ceiling gate for the 19-new-fit plan. This computes the same
way (real measured rates only, feature-build billed once per dataset at the
GPU rate, same $2.241/h GPU rate from docs/COMPUTE_LEDGER.md L546-547), but
against the real M1B headline results now on disk -- which, unlike the
original estimate, has a genuine measured rate for all 9 (dataset, arm)
cells including the 2wiki_clean BASE+NODE_ROLE+SUPPORT interaction arm, so
nothing here is a conservative guess.

Feature-build repeats once per dataset per invocation (unchanged behaviour,
not a new cost): the runner's own feature_store_reuse rebuilds nothing
within one dataset's arm/seed loop, but a second invocation of the whole
runner naturally re-pays that one-time-per-dataset cost again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.m1b_compute_estimate import (  # noqa: E402
    GPU_RATE_USD_PER_H,
    R3_FEATURE_MS,
)

RESULTS_DIR = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "headline"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "rerun_with_rows_estimate.json"
DATASETS = ["2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp"]
ALREADY_SPENT_USD = 2.28  # scripts/m1b_compute_estimate.py's own filed figure
FILED_CEILING_USD = 5.0


def main() -> dict:
    results = {ds: json.loads((RESULTS_DIR / f"{ds}.json").read_text(encoding="utf-8")) for ds in DATASETS}
    val_queries = {ds: results[ds]["queries"] for ds in DATASETS}

    feature_build_seconds = {ds: R3_FEATURE_MS[ds] / 1000.0 * val_queries[ds] for ds in DATASETS}
    feature_build_total_s = sum(feature_build_seconds.values())

    arm_breakdown = []
    fit_seconds_total = 0.0
    total_fits = 0
    for ds in DATASETS:
        arms = results[ds]["cells"]["R3"]["arms"]
        for arm, armdata in arms.items():
            n_seeds = len(armdata["seeds"])
            rate = armdata["seeds"]["0"]["training"]["training_seconds"]
            cost = rate * n_seeds
            fit_seconds_total += cost
            total_fits += n_seeds
            arm_breakdown.append({
                "dataset": ds, "arm": arm, "rate_seconds_per_fit": round(rate, 3),
                "seeds_refit": n_seeds, "cost_seconds": round(cost, 2),
                "rate_source": "real M1B headline training_seconds (seed=0) -- no guessing needed, every arm now has a real run",
            })

    pure_compute_gpu_h = (feature_build_total_s + fit_seconds_total) / 3600.0
    cost_all_gpu = pure_compute_gpu_h * GPU_RATE_USD_PER_H
    cumulative_usd = ALREADY_SPENT_USD + cost_all_gpu

    manifest = {
        "purpose": "amendment_5_rerun_with_row_capture -- re-fit all 27 logical fits so per-query rows can be persisted",
        "logical_fits_to_rerun": total_fits,
        "feature_build": {
            "basis": "repeats once per dataset per invocation, unchanged rate from the original estimate",
            "per_dataset_seconds": {ds: round(s, 1) for ds, s in feature_build_seconds.items()},
            "total_seconds": round(feature_build_total_s, 1),
        },
        "arm_breakdown": arm_breakdown,
        "fit_seconds_total": round(fit_seconds_total, 1),
        "pure_compute_gpu_h": round(pure_compute_gpu_h, 4),
        "rerun_cost_usd_all_gpu_billed": round(cost_all_gpu, 3),
        "already_spent_usd": ALREADY_SPENT_USD,
        "cumulative_usd": round(cumulative_usd, 3),
        "filed_ceiling_usd": FILED_CEILING_USD,
        "within_ceiling": cumulative_usd <= FILED_CEILING_USD,
        "gpu_rate_usd_per_h": GPU_RATE_USD_PER_H,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_PATH, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")
    print(
        f"\nVERDICT: rerun ${cost_all_gpu:.3f} + already-spent ${ALREADY_SPENT_USD:.2f} = "
        f"${cumulative_usd:.3f} cumulative vs ${FILED_CEILING_USD:.2f} ceiling -- "
        f"within_ceiling={cumulative_usd <= FILED_CEILING_USD}"
    )
    return manifest


if __name__ == "__main__":
    main()
