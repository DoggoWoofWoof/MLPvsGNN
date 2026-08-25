#!/usr/bin/env python
"""Evaluate the preregistered set-coverage variant against frozen confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_confirmation import (
    METRICS,
    _aggregate_grouped_seeds,
    _paired_group_gap,
    mean_sd_ci,
)


def _paired_metrics(
    left: dict[str, Any],
    right: dict[str, Any],
    seeds: list[str],
) -> dict[str, Any]:
    return {
        metric: mean_sd_ci(
            [
                float(left["seeds"][seed]["metrics"][metric])
                - float(right["seeds"][seed]["metrics"][metric])
                for seed in seeds
            ]
        )
        for metric in METRICS
    }


def analyze(
    variant: dict[str, Any],
    confirmation: dict[str, Any],
    *,
    required_r20_gain: float,
    required_gap_fraction: float,
    required_r1_retention: float,
) -> dict[str, Any]:
    if variant["dataset"] != confirmation["dataset"]:
        raise ValueError("Variant and confirmation datasets differ")
    if (
        variant["data"]["candidate_contract_sha256"]
        != confirmation["data"]["candidate_contract_sha256"]
    ):
        raise ValueError("Variant and confirmation candidate contracts differ")
    variant_model = variant["model"]
    seeds = sorted(variant_model["seeds"], key=int)
    max_width = max(int(width) for width in confirmation["config"]["hidden_widths"])
    original_key = f"offset_mlp_k4_h{max_width}"
    gnn_key = f"{confirmation['frozen_best_gnn']}_h{max_width}"
    original = confirmation["models"][original_key]
    gnn = confirmation["models"][gnn_key]
    if seeds != sorted(original["seeds"], key=int) or seeds != sorted(gnn["seeds"], key=int):
        raise ValueError("Variant and confirmation seeds differ")
    versus_original = _paired_metrics(variant_model, original, seeds)
    versus_gnn = _paired_metrics(variant_model, gnn, seeds)
    original_r20 = float(original["aggregate"]["test_metrics"]["recall@20"]["mean"])
    variant_r20 = float(variant_model["aggregate"]["test_metrics"]["recall@20"]["mean"])
    gnn_r20 = float(gnn["aggregate"]["test_metrics"]["recall@20"]["mean"])
    denominator = gnn_r20 - original_r20
    gap_fraction = (variant_r20 - original_r20) / denominator if denominator else float("nan")
    original_r1_advantage = (
        float(original["aggregate"]["test_metrics"]["recall@1"]["mean"])
        - float(gnn["aggregate"]["test_metrics"]["recall@1"]["mean"])
    )
    variant_r1_advantage = (
        float(variant_model["aggregate"]["test_metrics"]["recall@1"]["mean"])
        - float(gnn["aggregate"]["test_metrics"]["recall@1"]["mean"])
    )
    r1_retention = (
        variant_r1_advantage / original_r1_advantage
        if original_r1_advantage
        else float("nan")
    )
    first_seed = variant_model["seeds"][seeds[0]]
    answer_count = {
        query_id: int(row["gold_count"])
        for query_id, row in first_seed["per_query"].items()
    }
    by_answer_count = {
        "variant": _aggregate_grouped_seeds(variant_model, answer_count),
        "original_k4": _aggregate_grouped_seeds(original, answer_count),
        "gnn": _aggregate_grouped_seeds(gnn, answer_count),
        "variant_minus_original_k4": _paired_group_gap(
            variant_model,
            original,
            answer_count,
        ),
        "variant_minus_gnn": _paired_group_gap(variant_model, gnn, answer_count),
    }
    primary = (
        float(versus_original["recall@20"]["mean"]) >= required_r20_gain
        and float(versus_original["recall@20"]["ci95_low"]) > 0.0
    )
    return {
        "status": variant["status"],
        "dataset": variant["dataset"],
        "contract_match": True,
        "variant_key": variant_model["key"],
        "original_key": original_key,
        "gnn_key": gnn_key,
        "paired_variant_minus_original_k4": versus_original,
        "paired_variant_minus_gnn": versus_gnn,
        "success_criteria": {
            "required_r20_gain": required_r20_gain,
            "required_gap_fraction": required_gap_fraction,
            "required_r1_retention": required_r1_retention,
            "observed_r20_gap_fraction_closed": gap_fraction,
            "observed_r1_advantage_retained": r1_retention,
            "primary_dataset_success": primary,
            "gap_fraction_target_met": gap_fraction >= required_gap_fraction,
            "r1_retention_target_met": r1_retention >= required_r1_retention,
        },
        "answer_count": by_answer_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-r20-gain", type=float, default=0.02)
    parser.add_argument("--required-gap-fraction", type=float, default=0.25)
    parser.add_argument("--required-r1-retention", type=float, default=0.75)
    args = parser.parse_args()
    variant = json.loads(args.variant.read_text(encoding="utf-8"))
    confirmation = json.loads(args.confirmation.read_text(encoding="utf-8"))
    result = analyze(
        variant,
        confirmation,
        required_r20_gain=args.required_r20_gain,
        required_gap_fraction=args.required_gap_fraction,
        required_r1_retention=args.required_r1_retention,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": result["dataset"],
                "success": result["success_criteria"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
