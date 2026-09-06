"""Tests for the M2 QLS-CELL derivation (scripts/m2_qls_cell_derivation.py).

Two classes of test, kept distinct: pure-logic tests over synthetic cells, and
cross-checks against the real filed M1A/M1B artifacts -- the derivation's whole
purpose is to be faithful to those, so a drift between them must fail loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.m2_qls_cell_derivation import (
    DATASETS,
    HEADLINE_DIR,
    M1B_RESULT_MAP_PATH,
    _m1a_arm_label,
    assemble_qls_cell,
    build_m1a_classification,
    classify_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
M1A_DECLARATION = REPO_ROOT / "configs" / "m1a_feature_screen.yaml"


def _cell(base_r5: float, arms: dict[str, float]) -> dict:
    def arm(name: str, r5: float) -> dict:
        return {
            "arm": name,
            "metrics": {"recall@5": r5},
            "parameters": {"total": 3457},
        }

    return {
        "train_queries": 800,
        "held_out_queries": 200,
        "arms": {"BASE": arm("BASE", base_r5), **{n: arm(n, v) for n, v in arms.items()}},
    }


# --------------------------------------------------------------------------
# M1A's real step-6 labelling rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize("delta_pp", [0.0, 0.1, 0.25, -0.25, -0.2])
def test_inside_tolerance_is_no_effect(delta_pp: float) -> None:
    assert _m1a_arm_label(delta_pp) == "NO_EFFECT"


@pytest.mark.parametrize("delta_pp", [0.26, 0.49, -0.3])
def test_gray_zone_is_unresolved_not_a_new_label(delta_pp: float) -> None:
    assert _m1a_arm_label(delta_pp) == "UNRESOLVED_BY_FILED_RULE"


@pytest.mark.parametrize("delta_pp", [0.51, 2.6, -0.6, -3.0])
def test_material_magnitude_still_unresolved_without_a_ci(delta_pp: float) -> None:
    """M1A issued no PROMOTED/HARMFUL label at all: both require "uncertainty
    not overlapping zero", which a one-seed screen cannot supply. Reproducing
    the idealised four-label rule instead would silently promote effects M1B
    later failed to confirm (webqsp NODE_ROLE, +2.358pp one seed, CI crossing
    zero on three)."""
    assert _m1a_arm_label(delta_pp) == "UNRESOLVED_BY_FILED_RULE"


def test_classify_cell_computes_deltas_against_base() -> None:
    out = classify_cell(_cell(0.50, {"BASE+SUPPORT": 0.515, "BASE+PATH": 0.5005}))
    assert out["base_recall_at_5"] == 0.50
    assert out["arms"]["BASE+SUPPORT"]["delta_r5_pp_vs_base"] == 1.5
    assert out["arms"]["BASE+SUPPORT"]["m1a_label"] == "UNRESOLVED_BY_FILED_RULE"
    assert out["arms"]["BASE+PATH"]["delta_r5_pp_vs_base"] == 0.05
    assert out["arms"]["BASE+PATH"]["m1a_label"] == "NO_EFFECT"
    assert "BASE" not in out["arms"]


# --------------------------------------------------------------------------
# Cross-checks against the real filed artifacts
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_qls_cell() -> dict:
    missing = [p for p in (M1B_RESULT_MAP_PATH, *(HEADLINE_DIR / f"{d}.json" for d in DATASETS)) if not p.exists()]
    if missing:
        pytest.skip(f"real M1A/M1B evidence not present locally: {missing[0]}")
    m1b = json.loads(M1B_RESULT_MAP_PATH.read_text(encoding="utf-8"))
    return assemble_qls_cell(build_m1a_classification(), m1b)


def test_promoted_cells_match_m1bs_own_survivor_set(real_qls_cell: dict) -> None:
    survivors = json.loads(M1B_RESULT_MAP_PATH.read_text(encoding="utf-8"))["pareto_survivor_set"]
    confirmed = {
        (dataset, regime): cell["features"]
        for dataset, regimes in real_qls_cell.items()
        for regime, cell in regimes.items()
        if cell["confirmation_level"] == "M1B_BOOTSTRAP_CONFIRMED"
    }
    assert confirmed == {
        ("hotpotqa_clean", "R3"): ["SUPPORT"],
        ("metaqa", "R3"): ["PATH"],
    }
    assert {s["dataset"] for s in survivors} == {d for d, _ in confirmed}


def test_not_confirmed_features_are_excluded(real_qls_cell: dict) -> None:
    assert real_qls_cell["webqsp"]["R3"]["features"] == []
    assert real_qls_cell["webqsp"]["R3"]["confirmation_level"] == "M1B_BOOTSTRAP_NOT_CONFIRMED"
    assert real_qls_cell["2wiki_clean"]["R3"]["per_feature"]["SUPPORT"]["status"] == (
        "M1B_BOOTSTRAP_NOT_CONFIRMED"
    )
    assert "SUPPORT" not in real_qls_cell["2wiki_clean"]["R3"]["features"]


def test_2wiki_node_role_is_provisional_not_confirmed(real_qls_cell: dict) -> None:
    """The single most load-bearing honesty check in this file: 2wiki's
    NODE_ROLE was never bootstrap-tested against plain BASE, and webqsp's
    comparable one-seed effect failed exactly that test."""
    r3 = real_qls_cell["2wiki_clean"]["R3"]
    assert r3["features"] == ["NODE_ROLE"]
    assert r3["per_feature"]["NODE_ROLE"]["status"] == "UNRESOLVED_PROVISIONAL_M1A_ONE_SEED_ONLY"


def test_r1_r2_feature_carryover_is_never_labelled_confirmed(real_qls_cell: dict) -> None:
    """M1B tested R3 only. Any R1/R2 cell that carries a feature must say so."""
    for dataset, regimes in real_qls_cell.items():
        for regime, cell in regimes.items():
            if regime != "R3" and cell["features"]:
                assert cell["confirmation_level"] == "UNRESOLVED_PROVISIONAL_M1A_ONE_SEED_ONLY", (
                    f"{dataset}/{regime} claims {cell['confirmation_level']} for a "
                    "regime M1B never tested"
                )


def test_every_declared_m1a_cell_is_present(real_qls_cell: dict) -> None:
    declared = yaml.safe_load(M1A_DECLARATION.read_text(encoding="utf-8"))["datasets"]
    expected = {
        (dataset, regime)
        for dataset, spec in declared.items()
        if "cells" in spec
        for regime in spec["cells"]
    }
    derived = {(d, r) for d, regimes in real_qls_cell.items() for r in regimes}
    assert derived == expected


def test_derived_cells_match_the_real_headline_results(real_qls_cell: dict) -> None:
    for dataset, regimes in real_qls_cell.items():
        data = json.loads((HEADLINE_DIR / f"{dataset}.json").read_text(encoding="utf-8"))
        assert set(regimes) == set(data["cells"]), dataset
