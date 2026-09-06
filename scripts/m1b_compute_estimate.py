#!/usr/bin/env python
"""M1B compute estimate, derived entirely from M1A's own real measurements.

Read-only. No guessed rates: feature-build seconds come from the real
steady-state per-query latency M1A step 3 measured (filed in
configs/m1a_feature_screen.yaml#compute.inputs), and per-arm training
seconds come from the real M1A seed=0 headline results. The one genuinely
new arm (2wiki_clean BASE+NODE_ROLE+SUPPORT) has no direct measurement, so
its rate is the conservative (not averaged) max of its two constituent
single-family arms on the same dataset -- the same "use the slower real
bracket, never a guess" convention M1A step 3 used.

Ceiling proposal follows the same convention used throughout this track:
2x the worst-case real bracket, rounded up to a stated round figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_edge_provenance import _atomic_json  # noqa: E402

RESULTS_DIR = REPO_ROOT / "outputs" / "m1a_feature_screen" / "headline"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "compute_estimate.json"
DATASETS = ["2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp"]

# docs/COMPUTE_LEDGER.md L546-547, reused verbatim throughout this track.
GPU_RATE_USD_PER_H = 2.241
CPU_RATE_USD_PER_H = 0.634

# Real per-query R3 feature-build latency (steady-state mean, ms), from
# configs/m1a_feature_screen.yaml#compute.inputs.real_per_query_feature_build_ms_steady_state_mean
R3_FEATURE_MS = {
    "2wiki_clean": 6.329,
    "hotpotqa_clean": 81.079,
    "metaqa": 14.553,
    "webqsp": 45.792,
}

# (dataset, arm) whose real M1A seed=0 R3 training_seconds is the per-seed
# rate for that (dataset, arm) -- seed-independent, since only model/training
# RNG depends on seed (confirmed: _cell_master_local and holdout_split are
# both seed-free).
RATE_ARMS = [
    ("2wiki_clean", "BASE"), ("2wiki_clean", "BASE+NODE_ROLE"), ("2wiki_clean", "BASE+SUPPORT"),
    ("hotpotqa_clean", "BASE"), ("hotpotqa_clean", "BASE+SUPPORT"),
    ("metaqa", "BASE"), ("metaqa", "BASE+PATH"),
    ("webqsp", "BASE"), ("webqsp", "BASE+NODE_ROLE"),
]

# (dataset, arm, rate_key, new_seed_count). rate_key=None means the new
# 2wiki_clean BASE+NODE_ROLE+SUPPORT arm, whose rate is derived below rather
# than read directly (no M1A measurement exists for it).
NEW_FITS = [
    ("2wiki_clean", "BASE", ("2wiki_clean", "BASE"), 2),                          # seeds 1,2 (seed 0 reused)
    ("2wiki_clean", "BASE+NODE_ROLE", ("2wiki_clean", "BASE+NODE_ROLE"), 2),       # seeds 1,2
    ("2wiki_clean", "BASE+NODE_ROLE+SUPPORT", None, 3),                           # all 3 seeds, brand new arm
    ("hotpotqa_clean", "BASE", ("hotpotqa_clean", "BASE"), 2),
    ("hotpotqa_clean", "BASE+SUPPORT", ("hotpotqa_clean", "BASE+SUPPORT"), 2),
    ("metaqa", "BASE", ("metaqa", "BASE"), 2),
    ("metaqa", "BASE+PATH", ("metaqa", "BASE+PATH"), 2),
    ("webqsp", "BASE", ("webqsp", "BASE"), 2),
    ("webqsp", "BASE+NODE_ROLE", ("webqsp", "BASE+NODE_ROLE"), 2),
]


def main() -> dict:
    results = {ds: json.loads((RESULTS_DIR / f"{ds}.json").read_text(encoding="utf-8")) for ds in DATASETS}
    val_queries = {ds: results[ds]["queries"] for ds in DATASETS}

    feature_build_seconds = {
        ds: R3_FEATURE_MS[ds] / 1000.0 * val_queries[ds] for ds in DATASETS
    }
    feature_build_total_s = sum(feature_build_seconds.values())

    arm_seconds = {
        (ds, arm): results[ds]["cells"]["R3"]["arms"][arm]["training"]["training_seconds"]
        for ds, arm in RATE_ARMS
    }

    node_role_s = arm_seconds[("2wiki_clean", "BASE+NODE_ROLE")]
    support_s = arm_seconds[("2wiki_clean", "BASE+SUPPORT")]
    interaction_estimate_s = max(node_role_s, support_s)

    new_fit_breakdown = []
    fit_seconds_total = 0.0
    total_new_fits = 0
    for ds, arm, rate_key, n in NEW_FITS:
        rate = interaction_estimate_s if rate_key is None else arm_seconds[rate_key]
        cost = rate * n
        fit_seconds_total += cost
        total_new_fits += n
        new_fit_breakdown.append({
            "dataset": ds, "arm": arm, "rate_seconds_per_fit": round(rate, 2),
            "new_fit_count": n, "cost_seconds": round(cost, 2),
            "rate_source": (
                "real M1A seed=0 training_seconds"
                if rate_key is not None
                else f"conservative max(2wiki_clean BASE+NODE_ROLE={node_role_s:.2f}s, "
                     f"2wiki_clean BASE+SUPPORT={support_s:.2f}s) -- no direct measurement exists "
                     f"for this new interaction arm"
            ),
        })

    pure_compute_gpu_h = (feature_build_total_s + fit_seconds_total) / 3600.0
    cost_all_gpu = pure_compute_gpu_h * GPU_RATE_USD_PER_H
    cost_blended = (
        (feature_build_total_s / 3600.0) * CPU_RATE_USD_PER_H
        + (fit_seconds_total / 3600.0) * GPU_RATE_USD_PER_H
    )
    ceiling = max(round(2 * cost_all_gpu, 2), 5.0)

    manifest = {
        "logical_fits": 27,
        "reused_fits": 8,
        "new_fits": total_new_fits,
        "feature_build": {
            "basis": "one build per (dataset, R3) cell, 4 cells total, reused across every seed/arm in that cell",
            "per_dataset_seconds": {ds: round(s, 1) for ds, s in feature_build_seconds.items()},
            "total_seconds": round(feature_build_total_s, 1),
            "total_gpu_h": round(feature_build_total_s / 3600.0, 4),
        },
        "new_fit_breakdown": new_fit_breakdown,
        "new_fits_total_seconds": round(fit_seconds_total, 1),
        "new_fits_total_gpu_h": round(fit_seconds_total / 3600.0, 4),
        "pure_compute_gpu_h": round(pure_compute_gpu_h, 4),
        "cost_usd_all_gpu_billed_conservative": round(cost_all_gpu, 2),
        "cost_usd_blended_feature_build_cpu_fit_gpu_optimistic": round(cost_blended, 2),
        "bootstrap_analysis_cost": (
            "pure-CPU, post-hoc, over already-scored held-out queries -- negligible next to "
            f"the ${cost_all_gpu:.2f} GPU-billed total above; not separately costed"
        ),
        "gpu_rate_usd_per_h": GPU_RATE_USD_PER_H,
        "cpu_rate_usd_per_h": CPU_RATE_USD_PER_H,
        "rate_source": "docs/COMPUTE_LEDGER.md L546-547",
        "proposed_ceiling_usd": ceiling,
        "ceiling_basis": "2x the worst-case real bracket (all-GPU-billed), rounded up to a stated round figure",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_PATH, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")
    print(
        f"\nVERDICT: {total_new_fits} new fits, {pure_compute_gpu_h:.4f} GPU-h pure-compute, "
        f"${cost_all_gpu:.2f} all-GPU-billed, proposed ceiling ${ceiling:.2f}"
    )
    return manifest


if __name__ == "__main__":
    main()
