"""UMLP-D0.3 checks: the gap quantities, the feasibility cuts, the learned-movement categories, the query FC requirement,
the D0.1 groups, and the frozen configs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
spec = importlib.util.spec_from_file_location("umlp_d03", REPO_ROOT / "scripts" / "umlp_d03_score_gap.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

# kept rows of one toy query, twin ranks 1..8 (s_hat descending); golds at ranks 1, 4 and 7
SHAT = np.array([3.0, 2.0, 1.5, 1.2, 1.0, 0.4, -0.5, -1.0])
RANK = np.arange(1, 9)


def golds(*positions):
    g = np.zeros(SHAT.size, dtype=bool)
    g[list(positions)] = True
    return g


def test_gap_quantities_and_feasibility_cuts():
    u = D.query_units(SHAT, RANK, golds(0, 3, 6))
    assert list(u["miss"]) == [6]
    assert u["kth"] == 1.0 and not u["rank5_is_gold"]
    assert u["required"][0] == pytest.approx(1.0 - (-0.5) + D.EPS)
    assert list(u["disp"]) == [1, 2, 4]                      # non-golds at ranks 2, 3, 5 (rank 4 is a gold), hardest first
    assert (u["hardest"], u["nearest"]) == (1, 4)
    assert D.pair_required(SHAT, 6, 1) == pytest.approx(2.5 + D.EPS)
    assert D.pair_required(SHAT, 6, 4) == pytest.approx(u["required"][0])   # the nearest displacer is the rank-5 candidate here
    req = u["required"][0]
    assert not req < D.TAU and req < 2 * D.TAU                # beyond one-sided, within two-sided
    assert D.pair_required(SHAT, 0, 1) == 0.0                 # a gold already above the displacer needs nothing


def test_rank5_gold_and_no_units():
    u = D.query_units(SHAT, RANK, golds(0, 3, 4, 6))
    assert u["rank5_is_gold"] and list(u["disp"]) == [1, 2] and u["nearest"] == 2
    assert D.query_units(SHAT, RANK, golds(1, 6)) is None     # the twin's top-1 is not a gold
    assert D.query_units(SHAT, RANK, golds(0, 3)) is None     # no gold below rank 5
    u = D.query_units(SHAT, RANK, golds(0, 1, 2, 3, 4, 6))    # ranks 2-5 all golds: a unit with no displacer
    assert u["disp"].size == 0 and u["hardest"] == u["nearest"] == -1


def test_learned_movement_categories():
    delta = np.array([0.5, -0.2, 0.1, 0.1])
    rank_off = np.array([3, 4, 9, 2])
    c = D.movement(delta, rank_off, 0, 1, r=1.0)              # m = 0.7, served order reversed
    assert c["correct"] and c["reversed"] and not c["far_too_small"] and not c["bad"]
    c = D.movement(delta, rank_off, 2, 3, r=1.0)              # m = 0
    assert c["none"] and not c["correct"] and not c["wrong"] and c["bad"]
    c = D.movement(delta, rank_off, 1, 0, r=0.5)              # m = -0.7
    assert c["wrong"] and c["bad"] and not c["reversed"]
    c = D.movement(delta, rank_off, 2, 1, r=1.0)              # m = 0.3 < r / 2, not reversed
    assert c["correct"] and c["far_too_small"] and c["bad"]
    c = D.movement(delta, rank_off, 2, 1, r=0.5)              # m = 0.3 >= r / 2
    assert c["correct"] and not c["far_too_small"] and not c["bad"]
    assert D.movement(delta, rank_off, 2, 1, r=1.0)["m"] == pytest.approx(0.3)


def test_query_full_coverage_requirement():
    g = golds(0, 6)
    req, ok = D.query_fc_requirement(SHAT, g, gold_total=2)  # both golds must clear the 4th highest non-gold (1.0)
    assert ok and req == pytest.approx(1.0 + 0.5 + D.EPS)
    req, ok = D.query_fc_requirement(SHAT, g, gold_total=3)  # a gold outside the pool: no reranker covers it
    assert not ok and np.isnan(req)
    six = golds(0, 1, 2, 3, 4, 6)
    req, ok = D.query_fc_requirement(SHAT, six, gold_total=6)
    assert not ok                                             # six golds cannot all fit in a top 5
    req, ok = D.query_fc_requirement(SHAT, golds(0, 1, 2, 3, 6), gold_total=5)
    assert ok and req == pytest.approx(1.0 + 0.5 + D.EPS)    # five golds must clear the highest non-gold


def test_bootstrap_share_is_a_ratio_over_resampled_queries():
    flags = np.array([True, False, True, True])
    q = np.array([0, 0, 1, 2])
    s = D.share_ci(flags, q)
    assert s["share"] == 0.75 and s["units"] == 4 and s["queries"] == 3
    assert 0 <= s["low"] <= s["share"] <= s["high"] <= 1
    assert D.share_ci([], [])["share"] is None


@pytest.mark.skipif(not (D.SIDECAR.exists() and D.D01.SIDECAR.exists()), reason="needs the D0.1 and D0.3 sidecars (outputs/ is not tracked)")
def test_groups_equal_d01_groups_for_seed():
    z = D.load_sidecar()
    with np.load(D.D01.SIDECAR) as f:
        z1 = {k: f[k] for k in f.files}
    gnn = D.D01.load_declaration()["arms_scored"]["mp_reference"]
    assert D.check_groups_against_d01(z, z1, gnn) == {}


def test_frozen_configs_unchanged():
    D.frozen_unchanged()
    decl = D.load_declaration()
    assert decl["units"]["epsilon"] == D.EPS and float(str(decl["units"]["tau"]).split()[0]) == D.TAU
    assert decl["population"]["unit"].startswith("the missed secondary gold")
