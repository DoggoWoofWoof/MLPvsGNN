"""M1B's preregistered paired query-level bootstrap: promotion-rule verdicts.

Every construction below uses a per-query-constant offset between challenger
and base wherever the test's intent is a specific seed-level/sign condition
in isolation: a resample of a population whose per-query difference never
varies has an invariant mean, so the bootstrap CI collapses to a point at
that constant and cannot itself introduce or hide ambiguity. Only
test_not_confirmed_when_bootstrap_ci_straddles_zero deliberately varies the
per-query delta, since exercising real bootstrap spread is its whole point.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.m1b_bootstrap_analysis import _comparison_for, analyse_cell
from scripts.run_m1b_targeted_resolution import _load_declaration

SEEDS = ["0", "1", "2"]


def _arm_block(values_by_seed: dict[str, list[float]], query_ids: list[str]) -> dict[str, Any]:
    return {
        "seeds": {
            seed: {"metrics": {"recall@5": sum(values) / len(values)}}
            for seed, values in values_by_seed.items()
        },
        "per_query_recall_at_5_by_seed": {
            seed: dict(zip(query_ids, values, strict=True)) for seed, values in values_by_seed.items()
        },
    }


def _fake_result(
    *, base: dict[str, list[float]], challenger: dict[str, list[float]], query_ids: list[str]
) -> dict[str, Any]:
    return {
        "cells": {
            "R3": {
                "arms": {
                    "BASE": _arm_block(base, query_ids),
                    "CHALLENGER": _arm_block(challenger, query_ids),
                }
            }
        }
    }


def _analyse(base_values: list[float], challenger_values: list[float], n: int = 24) -> dict[str, Any]:
    query_ids = [f"q{i}" for i in range(n)]
    base = {seed: base_values for seed in SEEDS}
    challenger = {seed: challenger_values for seed in SEEDS}
    result = _fake_result(base=base, challenger=challenger, query_ids=query_ids)
    return analyse_cell(
        dataset="toy",
        result=result,
        challenger="CHALLENGER",
        base="BASE",
        bootstrap_resample_count=2000,
        bootstrap_rng_seed=1,
        confidence_level=0.95,
    )


def test_promoted_when_a_clear_positive_effect_with_no_seed_contradiction():
    n = 24
    base_values = [0.0 if i % 2 == 0 else 0.4 for i in range(n)]
    challenger_values = [v + 0.15 for v in base_values]  # constant +15pp offset, every query, every seed
    report = _analyse(base_values, challenger_values, n=n)
    assert report["seed_level"]["mean_pp"] == pytest.approx(15.0)
    assert report["seed_level"]["sign_pattern"] == "+++"
    assert report["bootstrap"]["ci_lower_pp"] > 0
    assert report["promotion_rule_verdict"] == "PROMOTED"


def test_harmful_when_a_clear_negative_effect():
    n = 24
    base_values = [0.2 if i % 2 == 0 else 0.6 for i in range(n)]
    challenger_values = [v - 0.15 for v in base_values]  # constant -15pp offset
    report = _analyse(base_values, challenger_values, n=n)
    assert report["seed_level"]["mean_pp"] == pytest.approx(-15.0)
    assert report["seed_level"]["sign_pattern"] == "---"
    assert report["bootstrap"]["ci_upper_pp"] < 0
    assert report["promotion_rule_verdict"] == "HARMFUL"


def test_not_confirmed_when_bootstrap_ci_straddles_zero():
    """A mean improvement past the 0.50pp magnitude threshold, but built from
    per-query deltas that swing widely in sign -- real bootstrap resampling
    (unlike the constant-offset constructions above) should find this mean
    unstable enough that the 95% CI still includes zero."""
    n = 40
    base_values = [0.3] * n
    # Per-query delta alternates +0.5 / -0.488: mean = (20*0.5 - 20*0.488)/40 = 0.006 = +0.6pp.
    deltas = [0.5 if i % 2 == 0 else -0.488 for i in range(n)]
    challenger_values = [b + d for b, d in zip(base_values, deltas, strict=True)]
    report = _analyse(base_values, challenger_values, n=n)
    assert report["seed_level"]["mean_pp"] == pytest.approx(0.6, abs=1e-6)
    assert report["seed_level"]["mean_pp"] > 0.50  # clears the magnitude threshold alone
    assert report["bootstrap"]["ci_lower_pp"] < 0 < report["bootstrap"]["ci_upper_pp"]
    assert report["promotion_rule_verdict"] == "not confirmed by M1B"


def test_not_confirmed_when_one_seed_contradicts_sign():
    """Mean across seeds clears +0.50pp and the (degenerate, zero-variance)
    bootstrap CI would read entirely positive, but one individual seed's own
    delta is below -0.50pp -- promotion_rule_for_m1b's no-seed-contradiction
    condition must still block PROMOTED."""
    n = 24
    query_ids = [f"q{i}" for i in range(n)]
    base_values = [0.3] * n
    per_seed_offset = {"0": 0.02, "1": 0.02, "2": -0.006}  # +2pp, +2pp, -0.6pp -> mean = +1.13pp
    base = {seed: base_values for seed in SEEDS}
    challenger = {seed: [v + per_seed_offset[seed] for v in base_values] for seed in SEEDS}
    result = _fake_result(base=base, challenger=challenger, query_ids=query_ids)
    report = analyse_cell(
        dataset="toy", result=result, challenger="CHALLENGER", base="BASE",
        bootstrap_resample_count=2000, bootstrap_rng_seed=1, confidence_level=0.95,
    )
    assert report["seed_level"]["mean_pp"] > 0.50
    assert report["seed_level"]["sign_pattern"] == "++-"
    assert report["bootstrap"]["ci_lower_pp"] > 0  # the degenerate CI alone would read PROMOTED
    assert report["promotion_rule_verdict"] == "not confirmed by M1B"


def test_mismatched_query_id_set_between_seeds_raises():
    query_ids = [f"q{i}" for i in range(10)]
    base = {seed: [0.5] * 10 for seed in SEEDS}
    challenger = {seed: [0.6] * 10 for seed in SEEDS}
    result = _fake_result(base=base, challenger=challenger, query_ids=query_ids)
    # Break seed "1"'s BASE key set only.
    broken = dict(result["cells"]["R3"]["arms"]["BASE"]["per_query_recall_at_5_by_seed"]["1"])
    del broken["q9"]
    broken["q_other"] = 0.5
    result["cells"]["R3"]["arms"]["BASE"]["per_query_recall_at_5_by_seed"]["1"] = broken
    with pytest.raises(ValueError, match="differs between seeds"):
        analyse_cell(
            dataset="toy", result=result, challenger="CHALLENGER", base="BASE",
            bootstrap_resample_count=100, bootstrap_rng_seed=1, confidence_level=0.95,
        )


def test_mismatched_query_id_set_between_arms_raises():
    query_ids = [f"q{i}" for i in range(10)]
    base = {seed: [0.5] * 10 for seed in SEEDS}
    challenger = {seed: [0.6] * 10 for seed in SEEDS}
    result = _fake_result(base=base, challenger=challenger, query_ids=query_ids)
    challenger_ids = [f"r{i}" for i in range(10)]  # entirely disjoint from BASE's q0..q9
    result["cells"]["R3"]["arms"]["CHALLENGER"]["per_query_recall_at_5_by_seed"] = {
        seed: dict(zip(challenger_ids, [0.6] * 10, strict=True)) for seed in SEEDS
    }
    with pytest.raises(ValueError, match="do not share an identical held-out"):
        analyse_cell(
            dataset="toy", result=result, challenger="CHALLENGER", base="BASE",
            bootstrap_resample_count=100, bootstrap_rng_seed=1, confidence_level=0.95,
        )


def test_cross_check_against_recorded_aggregate_raises_on_mismatch():
    query_ids = [f"q{i}" for i in range(10)]
    base = {seed: [0.5] * 10 for seed in SEEDS}
    challenger = {seed: [0.6] * 10 for seed in SEEDS}
    result = _fake_result(base=base, challenger=challenger, query_ids=query_ids)
    result["cells"]["R3"]["arms"]["CHALLENGER"]["seeds"]["1"]["metrics"]["recall@5"] = 0.999
    with pytest.raises(ValueError, match="do not reproduce the metric"):
        analyse_cell(
            dataset="toy", result=result, challenger="CHALLENGER", base="BASE",
            bootstrap_resample_count=100, bootstrap_rng_seed=1, confidence_level=0.95,
        )


def test_comparison_for_reads_the_real_filed_declaration():
    declaration = _load_declaration()
    assert _comparison_for(declaration, "2wiki_clean") == ("BASE+NODE_ROLE+SUPPORT", "BASE+NODE_ROLE")
    assert _comparison_for(declaration, "hotpotqa_clean") == ("BASE+SUPPORT", "BASE")
    assert _comparison_for(declaration, "metaqa") == ("BASE+PATH", "BASE")
    assert _comparison_for(declaration, "webqsp") == ("BASE+NODE_ROLE", "BASE")


def test_bootstrap_rng_is_deterministic_given_the_same_seed():
    n = 24
    base_values = [0.1 * (i % 5) for i in range(n)]
    challenger_values = [v + 0.03 for v in base_values]
    first = _analyse(base_values, challenger_values, n=n)
    second = _analyse(base_values, challenger_values, n=n)
    assert first["bootstrap"] == second["bootstrap"]
