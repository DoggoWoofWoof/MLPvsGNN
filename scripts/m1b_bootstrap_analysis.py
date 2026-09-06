#!/usr/bin/env python
"""M1B's preregistered paired query-level bootstrap and promotion-rule verdict.

Implements configs/m1b_targeted_resolution.yaml#uncertainty_procedure and
#promotion_rule_for_m1b exactly as filed on 2026-09-05, before any M1B result
existed: for each declared cell's designated comparison (challenger arm vs
its base), draw 10,000 bootstrap replicates, each a whole-query resample
(with replacement) of the held-out set shared across both arms and all 3
development seeds, compute bootstrap_statistic = mean_s(delta_seed) per
replicate, and take the [2.5th, 97.5th] percentile as the 95% CI. Reported
alongside, never merged into it, is the real (non-resampled) 3-seed spread.

Needs amendment 5's per_query_recall_at_5_by_seed (scripts/
run_m1b_targeted_resolution.py's _run_arm_with_rows) -- the whole reason that
amendment exists is that this analysis was not computable without it: no
per-query outcome was persisted anywhere before, for any of the 27 logical
fits, spliced or freshly trained.

RNG scope: the declaration fixes one bootstrap_rng_seed value, not an
explicit per-cell-vs-single-global-stream choice. This implementation
re-seeds a fresh numpy Generator with that same filed value once per dataset
cell, rather than advancing one stream across all 4 -- the more conservative
reading, since it makes re-running any single cell's analysis in isolation
reproduce exactly, independent of what order the other cells were processed
in or whether they ran at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_m1b_targeted_resolution import _declared_cell, _load_declaration  # noqa: E402

HEADLINE_DIR = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "headline"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "bootstrap_analysis.json"
DATASETS = ["2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp"]
MATERIAL_THRESHOLD_PP = 0.50  # promotion_rule_for_m1b's own filed threshold, reused not reinvented


def _comparison_for(declaration: dict[str, Any], dataset: str) -> tuple[str, str]:
    """(challenger_arm, base_arm) for a dataset's promotion-rule comparison.

    Uses the cell's own explicit primary_comparison when declared --
    2wiki_clean only, "BASE+NODE_ROLE+SUPPORT vs BASE+NODE_ROLE", not vs
    BASE, since that pairing is the actual scientific question
    (scientific_questions.2wiki_r3: does structure add anything *after*
    NODE_ROLE, not after nothing). Every other cell declares exactly 2 arms
    with no override, where the non-BASE arm vs BASE is unambiguous.
    """
    raw = declaration["cells"][dataset]
    if "primary_comparison" in raw:
        challenger, base = (part.strip() for part in raw["primary_comparison"].split(" vs "))
        return challenger, base
    resolved = _declared_cell(declaration, dataset)
    arms = resolved["arms"]
    if len(arms) != 2 or "BASE" not in arms:
        raise ValueError(
            f"{dataset}: arms {arms} has no explicit primary_comparison and is not a "
            "simple 2-arm {BASE, X} cell -- which pair to test is ambiguous"
        )
    challenger = arms[0] if arms[1] == "BASE" else arms[1]
    return challenger, "BASE"


def _per_query_arrays(
    arm_data: dict[str, Any], seeds: list[str]
) -> tuple[list[str], dict[str, np.ndarray]]:
    """A fixed query_id order and each seed's recall@5 array aligned to it.

    Sorted rather than insertion order: reproducible across a JSON round-trip
    without relying on dict-key ordering being preserved by whatever wrote or
    re-serialised the file.
    """
    per_query_by_seed = arm_data["per_query_recall_at_5_by_seed"]
    query_order = sorted(per_query_by_seed[seeds[0]])
    for seed in seeds:
        if set(per_query_by_seed[seed]) != set(query_order):
            raise ValueError(
                "held-out query_id set differs between seeds -- holdout_split is "
                "declared seed-independent (uncertainty_procedure.design.query_level); "
                "this breaks the paired-resampling premise and must be resolved, not ignored"
            )
    arrays = {
        seed: np.array([per_query_by_seed[seed][qid] for qid in query_order], dtype=np.float64)
        for seed in seeds
    }
    return query_order, arrays


def analyse_cell(
    *,
    dataset: str,
    result: dict[str, Any],
    challenger: str,
    base: str,
    bootstrap_resample_count: int,
    bootstrap_rng_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    cell = result["cells"]["R3"]
    seeds = sorted(cell["arms"][base]["seeds"])  # ["0", "1", "2"]
    challenger_order, challenger_arrays = _per_query_arrays(cell["arms"][challenger], seeds)
    base_order, base_arrays = _per_query_arrays(cell["arms"][base], seeds)
    if challenger_order != base_order:
        raise ValueError(
            f"{dataset}: {challenger!r} and {base!r} do not share an identical held-out "
            "query_id set/order -- holdout_split is declared arm-independent; this "
            "breaks the paired-resampling premise"
        )
    query_order = challenger_order
    n = len(query_order)

    # --- real (non-resampled) 3-seed spread -- a separate quantity, never combined with the bootstrap CI ---
    real_delta_by_seed = {
        seed: float(challenger_arrays[seed].mean() - base_arrays[seed].mean()) for seed in seeds
    }
    for seed in seeds:
        recorded = (
            cell["arms"][challenger]["seeds"][seed]["metrics"]["recall@5"]
            - cell["arms"][base]["seeds"][seed]["metrics"]["recall@5"]
        )
        if abs(real_delta_by_seed[seed] - recorded) > 1e-6:
            raise ValueError(
                f"{dataset}/seed={seed}: delta from per-query rows ({real_delta_by_seed[seed]!r}) "
                f"!= delta from recorded aggregate metrics ({recorded!r}) -- the captured rows do "
                "not reproduce the metric they were supposedly built from"
            )
    deltas = np.array([real_delta_by_seed[s] for s in seeds])
    real_mean_pp = float(deltas.mean() * 100)
    real_sd_pp = float(deltas.std(ddof=1) * 100) if len(deltas) > 1 else 0.0
    sign_pattern = "".join("+" if d > 0 else ("-" if d < 0 else "0") for d in deltas)

    # --- paired query-level bootstrap: one shared resample per replicate, across both arms and all 3 seeds ---
    rng = np.random.default_rng(bootstrap_rng_seed)
    idx = rng.integers(0, n, size=(bootstrap_resample_count, n))
    per_seed_replicate_delta = np.stack(
        [challenger_arrays[seed][idx].mean(axis=1) - base_arrays[seed][idx].mean(axis=1) for seed in seeds],
        axis=0,
    )  # shape (3, bootstrap_resample_count)
    bootstrap_statistic = per_seed_replicate_delta.mean(axis=0)  # mean_s(delta_seed) per replicate
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(bootstrap_statistic, 100 * alpha))
    ci_upper = float(np.percentile(bootstrap_statistic, 100 * (1 - alpha)))

    promoted = (
        real_mean_pp > MATERIAL_THRESHOLD_PP
        and ci_lower > 0
        and all(d * 100 > -MATERIAL_THRESHOLD_PP for d in deltas)
    )
    harmful = (
        real_mean_pp < -MATERIAL_THRESHOLD_PP
        and ci_upper < 0
        and all(d * 100 < MATERIAL_THRESHOLD_PP for d in deltas)
    )
    verdict = "PROMOTED" if promoted else ("HARMFUL" if harmful else "not confirmed by M1B")

    return {
        "dataset": dataset,
        "comparison": f"{challenger} vs {base}",
        "held_out_queries": n,
        "seed_level": {
            "delta_r5_pp_by_seed": {s: real_delta_by_seed[s] * 100 for s in seeds},
            "mean_pp": real_mean_pp,
            "sample_sd_pp": real_sd_pp,
            "min_pp": float(deltas.min() * 100),
            "max_pp": float(deltas.max() * 100),
            "sign_pattern": sign_pattern,
        },
        "bootstrap": {
            "resample_count": bootstrap_resample_count,
            "rng_seed": bootstrap_rng_seed,
            "confidence_level": confidence_level,
            "ci_lower_pp": ci_lower * 100,
            "ci_upper_pp": ci_upper * 100,
        },
        "promotion_rule_verdict": verdict,
    }


def main() -> dict[str, Any]:
    declaration = _load_declaration()
    procedure = declaration["uncertainty_procedure"]
    resample_count = int(procedure["bootstrap_resample_count"])
    rng_seed = int(procedure["bootstrap_rng_seed"])
    confidence = float(procedure["confidence_level"])

    cells = {}
    for dataset in DATASETS:
        result = json.loads((HEADLINE_DIR / f"{dataset}.json").read_text(encoding="utf-8"))
        challenger, base = _comparison_for(declaration, dataset)
        cells[dataset] = analyse_cell(
            dataset=dataset,
            result=result,
            challenger=challenger,
            base=base,
            bootstrap_resample_count=resample_count,
            bootstrap_rng_seed=rng_seed,
            confidence_level=confidence,
        )

    manifest = {
        "status": "M1B_BOOTSTRAP_ANALYSIS_COMPLETE",
        "procedure": "configs/m1b_targeted_resolution.yaml#uncertainty_procedure",
        "rng_seed_scope": (
            "one fresh RNG, re-seeded with the same filed bootstrap_rng_seed, per "
            "dataset cell -- see this module's own docstring for why."
        ),
        "cells": cells,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_PATH, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")
    return manifest


if __name__ == "__main__":
    main()
