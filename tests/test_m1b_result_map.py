"""M1B result map + Pareto survivor set, built from a bootstrap analysis."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.m1b_result_map import build_result_map
from scripts.run_m1b_targeted_resolution import _load_declaration

REAL_DATASETS = ["2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp"]


def _cell(verdict: str, mean_pp: float, ci: tuple[float, float]) -> dict[str, Any]:
    return {
        "comparison": "BASE+X vs BASE",
        "held_out_queries": 100,
        "seed_level": {"mean_pp": mean_pp, "sample_sd_pp": 0.1, "min_pp": mean_pp - 0.1, "max_pp": mean_pp + 0.1, "sign_pattern": "+++"},
        "bootstrap": {"resample_count": 10000, "rng_seed": 20260905, "confidence_level": 0.95, "ci_lower_pp": ci[0], "ci_upper_pp": ci[1]},
        "promotion_rule_verdict": verdict,
    }


def _declaration_stub(datasets: list[str]) -> dict[str, Any]:
    return {"cells": {d: {} for d in datasets}}


def test_promoted_cells_enter_the_survivor_set_others_do_not():
    bootstrap_analysis = {
        "cells": {
            "promoted_one": _cell("PROMOTED", 1.5, (0.2, 2.8)),
            "harmful_one": _cell("HARMFUL", -1.2, (-2.1, -0.3)),
            "unconfirmed_one": _cell("not confirmed by M1B", 0.8, (-0.1, 1.7)),
        }
    }
    declaration = _declaration_stub(["promoted_one", "harmful_one", "unconfirmed_one"])
    built = build_result_map(bootstrap_analysis, declaration)

    assert set(built["result_map"]) == {"promoted_one", "harmful_one", "unconfirmed_one"}
    assert built["result_map"]["harmful_one"]["verdict"] == "HARMFUL"
    assert built["result_map"]["unconfirmed_one"]["verdict"] == "not confirmed by M1B"

    survivor_datasets = {s["dataset"] for s in built["pareto_survivor_set"]}
    assert survivor_datasets == {"promoted_one"}
    survivor = built["pareto_survivor_set"][0]
    assert survivor["mean_r5_improvement_pp"] == pytest.approx(1.5)
    assert survivor["bootstrap_ci_pp"] == [0.2, 2.8]
    assert survivor["status"] == "FROZEN_PARETO_CANDIDATE_RESERVED_FOR_5_SEED_CONFIRMATION"


def test_no_survivors_when_nothing_is_promoted():
    bootstrap_analysis = {
        "cells": {
            "a": _cell("HARMFUL", -0.9, (-1.5, -0.2)),
            "b": _cell("not confirmed by M1B", 0.3, (-0.4, 1.0)),
        }
    }
    declaration = _declaration_stub(["a", "b"])
    built = build_result_map(bootstrap_analysis, declaration)
    assert built["pareto_survivor_set"] == []
    assert len(built["result_map"]) == 2


def test_all_four_promoted_all_four_survive():
    bootstrap_analysis = {"cells": {d: _cell("PROMOTED", 2.0, (0.5, 3.5)) for d in REAL_DATASETS}}
    declaration = _declaration_stub(REAL_DATASETS)
    built = build_result_map(bootstrap_analysis, declaration)
    assert {s["dataset"] for s in built["pareto_survivor_set"]} == set(REAL_DATASETS)


def test_mismatched_cell_sets_between_analysis_and_declaration_raises():
    bootstrap_analysis = {"cells": {"only_this_one": _cell("PROMOTED", 1.0, (0.1, 1.9))}}
    declaration = _declaration_stub(["only_this_one", "missing_from_analysis"])
    with pytest.raises(ValueError, match="!= declared cells"):
        build_result_map(bootstrap_analysis, declaration)


def test_against_the_real_filed_declaration_cell_set():
    declaration = _load_declaration()
    bootstrap_analysis = {"cells": {d: _cell("not confirmed by M1B", 0.1, (-0.2, 0.4)) for d in REAL_DATASETS}}
    built = build_result_map(bootstrap_analysis, declaration)
    assert set(built["result_map"]) == set(REAL_DATASETS)
