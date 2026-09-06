#!/usr/bin/env python
"""M2 compute estimate, derived entirely from real M1A / M0C measurements.

Read-only. Launches nothing, fits nothing, authorises nothing -- it is the
arithmetic configs/m2_qls_v2_freeze.yaml amendment 1 asks for ahead of any
launch amendment. Every rate is a real measurement from this track:

  feature build   per (dataset, regime) cell, M1A's own full-split
                  uncached_feature_build_latency_ms (p50 = steady-state
                  floor; mean = conservative, it includes the first-query
                  JIT compile outlier) x that cell's validation split size.
                  musique_clean has no M1A-scale measurement, so its R1
                  figure is M0C's 100-query steady-state mean, scaled by the
                  largest M0C->M1A drift observed on its twin squad_clean/R1
                  (a measured calibration, not a guess).
  fit             per new fit, the slowest real seed-0 training_seconds M1A
                  recorded on the SAME cell, whatever the arm -- the widest
                  measured arm (7 precomputed columns) is 2 short of the
                  universal arm's 9, and M1A's own data shows fit time is
                  width-insensitive (e.g. hotpotqa_clean/R3: BASE 121.8s vs
                  BASE+PATH 119.8s), so the cell's slowest arm is the
                  conservative bracket. musique_clean, never fit by M1A,
                  uses M1A's filed conservative per-train-query rate
                  (0.0154934 s/query, configs/m1a_feature_screen.yaml
                  #compute.inputs.real_m1a_scorer_training_seconds_per_query);
                  the max full-scale per-query rate M1A actually observed is
                  reported as the floor, and the historical SA_MLP 5-seed
                  mean on the same split as an independent sanity anchor.

The workload itself is read from the declaration's m2_selection_matrix, not
retyped here, and the counts are asserted against its workload block.

Ceiling proposal follows the track's convention: 2x the worst-case real
bracket (all-GPU-billed), rounded up to a stated round figure, never below
the $5 floor M1B used.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_edge_provenance import _atomic_json  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
M1A_HEADLINE_DIR = REPO_ROOT / "outputs" / "m1a_feature_screen" / "headline"
M0C_HEADLINE_DIR = REPO_ROOT / "outputs" / "m0c_bounded_r3" / "headline"
SA_MLP_DIR = REPO_ROOT / "outputs" / "sa_mlp_confirmation"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "compute_estimate.json"

# docs/COMPUTE_LEDGER.md L546-547, reused verbatim throughout this track.
GPU_RATE_USD_PER_H = 2.241
CPU_RATE_USD_PER_H = 0.634

# configs/m1a_feature_screen.yaml#compute.inputs.real_m1a_scorer_training_seconds_per_query
M1A_FILED_FIT_SECONDS_PER_TRAIN_QUERY = 0.0154934
HOLDOUT_FRACTION = 0.2  # modal_m1a_feature_screen.py _runner_args, unchanged for M2

M1B_R3_CELLS_WITH_ROWS = {("2wiki_clean", "R3"), ("hotpotqa_clean", "R3"), ("metaqa", "R3"), ("webqsp", "R3")}
UNIVERSAL_MATRIX_NAME = "QLS-UNIVERSAL"
MUSIQUE = "musique_clean"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _holdout(n: int) -> tuple[int, int]:
    # scripts/run_graph_context_d1.py::holdout_split, the runner's own rule
    held = max(1, int(round(n * HOLDOUT_FRACTION)))
    return n - held, held


def _runner_arm(matrix_arm: str, schema: dict) -> str:
    return schema["arm_name"] if matrix_arm == UNIVERSAL_MATRIX_NAME else matrix_arm


def main() -> dict:
    declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    matrix = declaration["m2_selection_matrix"]
    schema = declaration["qls_universal"]["feature_schema"]
    m1a_datasets = [d for d in matrix["cells"] if d != MUSIQUE]
    headline = {d: _load(M1A_HEADLINE_DIR / f"{d}.json") for d in m1a_datasets}

    # --- musique_clean's only inputs: M0C (zero-training) and the historical SA_MLP split ---
    m0c_musique = _load(M0C_HEADLINE_DIR / f"{MUSIQUE}.json")["systems"]["r1_feature_latency_ms"]["steady_state"]
    m0c_squad = _load(M0C_HEADLINE_DIR / "squad_clean.json")["systems"]["r1_feature_latency_ms"]["steady_state"]
    m1a_squad_r1 = headline["squad_clean"]["cells"]["R1"]["uncached_feature_build_latency_ms"]
    # how far M1A's full-split measurement drifted above M0C's 100-query panel on the twin cell
    m0c_to_m1a_calibration = max(m1a_squad_r1["mean"] / m0c_squad["mean"], m1a_squad_r1["p50"] / m0c_squad["p50"])
    musique_val_n = _load(SA_MLP_DIR / f"{MUSIQUE}.json")["data"]["splits"]["validation"]
    musique_sa_mlp_train_s = _load(SA_MLP_DIR / f"{MUSIQUE}.json")["models"]["sa_mlp"]["aggregate"]["training_seconds"]["mean"]
    musique_train_n, musique_held_n = _holdout(musique_val_n)
    observed_fit_rates = {
        (d, r, a): res["training"]["training_seconds"] / cell["train_queries"]
        for d in m1a_datasets
        for r, cell in headline[d]["cells"].items()
        for a, res in cell["arms"].items()
    }
    max_observed_rate_key = max(observed_fit_rates, key=observed_fit_rates.get)
    max_observed_rate = observed_fit_rates[max_observed_rate_key]

    # --- per-cell inputs ---
    cells: list[dict] = []
    workload: list[dict] = []
    for dataset, regimes in matrix["cells"].items():
        for regime, arms in regimes.items():
            if dataset == MUSIQUE:
                val_n, train_n, held_n = musique_val_n, musique_train_n, musique_held_n
                feature_ms_floor = m0c_musique["p50"]
                feature_ms_conservative = m0c_musique["mean"] * m0c_to_m1a_calibration
                feature_source = (
                    f"M0C 100-query steady-state (p50 {m0c_musique['p50']:.3f}ms, mean {m0c_musique['mean']:.3f}ms); "
                    f"conservative = mean x {m0c_to_m1a_calibration:.3f}, the largest M0C->M1A drift measured on squad_clean/R1"
                )
                fit_rate_conservative = train_n * M1A_FILED_FIT_SECONDS_PER_TRAIN_QUERY
                fit_rate_floor = train_n * max_observed_rate
                fit_source = (
                    f"no M1A fit exists: conservative = {train_n} train queries x M1A's filed {M1A_FILED_FIT_SECONDS_PER_TRAIN_QUERY} s/query; "
                    f"floor = x {max_observed_rate:.5f} s/query, the max full-scale rate M1A observed ({'/'.join(max_observed_rate_key)}); "
                    f"sanity anchor: historical SA_MLP (~213K params) 5-seed mean on this split = {musique_sa_mlp_train_s:.1f}s"
                )
            else:
                cell = headline[dataset]["cells"][regime]
                val_n = headline[dataset]["queries"]
                train_n, held_n = cell["train_queries"], cell["held_out_queries"]
                latency = cell["uncached_feature_build_latency_ms"]
                feature_ms_floor, feature_ms_conservative = latency["p50"], latency["mean"]
                feature_source = "M1A headline uncached_feature_build_latency_ms, full validation split (p50 floor; mean conservative, includes the JIT cold start)"
                slowest_arm = max(cell["arms"], key=lambda a: cell["arms"][a]["training"]["training_seconds"])
                fit_rate_conservative = cell["arms"][slowest_arm]["training"]["training_seconds"]
                fit_rate_floor = min(res["training"]["training_seconds"] for res in cell["arms"].values())
                fit_source = f"slowest real M1A seed-0 arm on this cell ({slowest_arm}); floor = fastest arm"
            new_arms = [a for a, status in arms.items() if status == "new"]
            entry = {
                "dataset": dataset,
                "regime": regime,
                "validation_queries": val_n,
                "train_queries": train_n,
                "held_out_queries": held_n,
                "feature_build": {
                    "per_query_ms_floor": round(feature_ms_floor, 3),
                    "per_query_ms_conservative": round(feature_ms_conservative, 3),
                    "seconds_floor": round(feature_ms_floor / 1000.0 * val_n, 1),
                    "seconds_conservative": round(feature_ms_conservative / 1000.0 * val_n, 1),
                    "source": feature_source,
                },
                "new_fits": new_arms,
                "fit_seconds_per_fit_floor": round(fit_rate_floor, 2),
                "fit_seconds_per_fit_conservative": round(fit_rate_conservative, 2),
                "fit_seconds_floor": round(fit_rate_floor * len(new_arms), 1),
                "fit_seconds_conservative": round(fit_rate_conservative * len(new_arms), 1),
                "fit_rate_source": fit_source,
            }
            cells.append(entry)
            for arm, status in arms.items():
                if status == "new":
                    source = "M2 fit, seed 0"
                elif (dataset, regime) in M1B_R3_CELLS_WITH_ROWS:
                    source = "reuse: M1A headline seed 0; M1B execution_2 seed-0 re-fit (per-query rows) preferred at launch"
                else:
                    source = "reuse: M1A headline seed 0"
                workload.append({
                    "dataset": dataset, "regime": regime, "arm": arm, "runner_arm": _runner_arm(arm, schema),
                    "seed": 0, "status": "new" if status == "new" else "reused", "matrix_label": status, "source": source,
                })

    new_fits = sum(1 for w in workload if w["status"] == "new")
    reused_fits = sum(1 for w in workload if w["status"] == "reused")
    declared = matrix["workload"]
    if (new_fits, reused_fits, len(workload), len(cells)) != (
        declared["new_fits"], declared["reused_fits"], declared["logical_fits"], declared["cells"]
    ):
        raise SystemExit(
            f"matrix enumerates new={new_fits} reused={reused_fits} logical={len(workload)} cells={len(cells)} "
            f"but the declaration's workload block says {declared}"
        )

    def total(key: str) -> float:
        return sum(c[key] if key in c else c["feature_build"][key] for c in cells)

    feature_floor = total("seconds_floor")
    feature_conservative = total("seconds_conservative")
    fit_floor = total("fit_seconds_floor")
    fit_conservative = total("fit_seconds_conservative")
    gpu_h_floor = (feature_floor + fit_floor) / 3600.0
    gpu_h_conservative = (feature_conservative + fit_conservative) / 3600.0
    cost_all_gpu_conservative = gpu_h_conservative * GPU_RATE_USD_PER_H
    cost_all_gpu_floor = gpu_h_floor * GPU_RATE_USD_PER_H
    cost_blended_conservative = (
        feature_conservative / 3600.0 * CPU_RATE_USD_PER_H + fit_conservative / 3600.0 * GPU_RATE_USD_PER_H
    )
    ceiling = max(float(math.ceil(2.0 * cost_all_gpu_conservative)), 5.0)

    per_dataset_chain_s = {}
    for c in cells:
        per_dataset_chain_s[c["dataset"]] = per_dataset_chain_s.get(c["dataset"], 0.0) + (
            c["feature_build"]["seconds_conservative"] + c["fit_seconds_conservative"]
        )
    longest_dataset = max(per_dataset_chain_s, key=per_dataset_chain_s.get)

    manifest = {
        "status": "M2_COMPUTE_ESTIMATE_NOT_A_LAUNCH_AUTHORISATION",
        "declaration": "configs/m2_qls_v2_freeze.yaml",
        "cells": declared["cells"],
        "logical_fits": len(workload),
        "new_fits": new_fits,
        "reused_fits": reused_fits,
        "seeds_per_fit": 1,
        "workload": workload,
        "per_cell": cells,
        "feature_build": {
            "basis": "one master-block build per (dataset, regime) cell, 14 cells; M1A never persisted its stores, so every cell with a new fit rebuilds once",
            "seconds_floor": round(feature_floor, 1),
            "seconds_conservative": round(feature_conservative, 1),
            "gpu_h_conservative": round(feature_conservative / 3600.0, 4),
            "musique_m0c_to_m1a_calibration_factor": round(m0c_to_m1a_calibration, 4),
        },
        "new_fits_seconds_floor": round(fit_floor, 1),
        "new_fits_seconds_conservative": round(fit_conservative, 1),
        "new_fits_gpu_h_conservative": round(fit_conservative / 3600.0, 4),
        "pure_compute_gpu_h_floor": round(gpu_h_floor, 4),
        "pure_compute_gpu_h_conservative": round(gpu_h_conservative, 4),
        "cost_usd_all_gpu_billed_floor": round(cost_all_gpu_floor, 2),
        "cost_usd_all_gpu_billed_conservative": round(cost_all_gpu_conservative, 2),
        "cost_usd_blended_feature_build_cpu_fit_gpu_conservative": round(cost_blended_conservative, 2),
        "wall_clock_with_per_dataset_parallelism": {
            "containers": len(per_dataset_chain_s),
            "per_dataset_serial_chain_minutes_conservative": {d: round(s / 60.0, 1) for d, s in per_dataset_chain_s.items()},
            "longest_chain": longest_dataset,
            "hours_conservative": round(per_dataset_chain_s[longest_dataset] / 3600.0, 2),
            "excludes": "per-container Modal startup and data load, unmeasured (same caveat as M1A step 5)",
        },
        "gpu_rate_usd_per_h": GPU_RATE_USD_PER_H,
        "cpu_rate_usd_per_h": CPU_RATE_USD_PER_H,
        "rate_source": "docs/COMPUTE_LEDGER.md L546-547",
        "proposed_ceiling_usd": ceiling,
        "ceiling_basis": "2x the worst-case real bracket (all-GPU-billed, conservative), rounded up to a whole dollar, never below M1B's $5 floor",
        "not_costed": [
            "gray-zone 3-seed expansions (only by further amendment, only where a verdict lands between 0.25pp and 0.50pp)",
            "bootstrap analysis (pure CPU, post hoc, negligible -- M1B precedent)",
            "the reuse audit (read-only, no compute)",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_PATH, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("workload", "per_cell")}, indent=2))
    print(f"\n{'cell':<22}{'val_n':>7}{'train':>7}{'feat_s(p50/mean)':>20}{'new':>5}{'fit_s/fit(min/max)':>22}")
    for c in cells:
        fb = c["feature_build"]
        print(
            f"{c['dataset'] + '/' + c['regime']:<22}{c['validation_queries']:>7}{c['train_queries']:>7}"
            f"{fb['seconds_floor']:>10.1f}/{fb['seconds_conservative']:<9.1f}{len(c['new_fits']):>5}"
            f"{c['fit_seconds_per_fit_floor']:>11.1f}/{c['fit_seconds_per_fit_conservative']:<10.1f}"
        )
    print(f"\nWrote {OUTPUT_PATH}")
    print(
        f"\nVERDICT: {new_fits} new fits + {reused_fits} reused = {len(workload)} logical, "
        f"{gpu_h_floor:.3f}-{gpu_h_conservative:.3f} GPU-h pure compute, "
        f"${cost_all_gpu_floor:.2f}-${cost_all_gpu_conservative:.2f} all-GPU-billed, "
        f"proposed ceiling ${ceiling:.2f}. Nothing launched."
    )
    return manifest


if __name__ == "__main__":
    main()
