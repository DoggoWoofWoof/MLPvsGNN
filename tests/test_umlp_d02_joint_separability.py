"""UMLP-D0.2 protocol checks: grouped folds, oriented pairs, query weights, inputs, out-of-fold scoring."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("umlp_d02", REPO_ROOT / "scripts" / "umlp_d02_joint_separability.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def test_fold_is_a_function_of_the_query_id():
    assert D.fold_of("abc") == int(hashlib.sha256(b"abc").hexdigest(), 16) % 5
    assert {D.fold_of(f"q{i}") for i in range(200)} == set(range(5))


def test_pairs_and_query_weights():
    qi = np.array([0, 0, 0, 1, 1, 1, 1])
    pos = np.array([1, 0, 0, 1, 1, 0, 0], dtype=bool)
    neg = np.array([0, 1, 1, 0, 0, 1, 1], dtype=bool)
    G, N, Q = D.build_pairs(qi, pos, neg, np.array([True, True]))
    assert len(G) == 2 + 4 and set(Q.tolist()) == {0, 1}
    assert all(qi[g] == qi[n] for g, n in zip(G, N))
    w = D.query_weights(Q)
    for q in (0, 1):
        assert 2 * w[Q == q].sum() == pytest.approx(1.0)   # both orientations carry w: each query totals 1
    G2, _, _ = D.build_pairs(qi, pos, neg, np.array([True, False]))
    assert len(G2) == 2   # an unanalysed query contributes no pair


def test_out_of_fold_and_grouped():
    rng = np.random.default_rng(0)
    nq = 60
    qi = np.repeat(np.arange(nq), 4)
    pos = np.tile([True, False, False, False], nq)
    neg = ~pos
    V = rng.normal(size=(qi.size, 3))
    V[pos, 0] += 2.0
    G, N, Q = D.build_pairs(qi, pos, neg, np.ones(nq, dtype=bool))
    fold_q = np.arange(nq) % 5
    oof, info = D.cross_fit(V, G, N, Q, fold_q, "linear")
    assert not np.isnan(oof[np.concatenate([G, N])]).any()
    assert all(f["train_pairs"] + f["test_pairs"] == len(G) for f in info["folds"])
    ev = D.evaluate(oof, G, N, Q)
    assert ev["pair_auc"] > 0.8 and ev["queries"] == nq


def test_linear_probe_is_logistic_on_d_without_intercept():
    rng = np.random.default_rng(1)
    Xg, Xn = rng.normal(size=(50, 4)) + 0.5, rng.normal(size=(50, 4))
    score, params, _ = D.fit_linear(Xg, Xn, np.full(50, 0.01))
    assert params == 4
    assert score(np.zeros((1, 4)))[0] == 0.0   # no intercept: g(0) = 0


def test_feature_sets_have_no_forbidden_inputs():
    side = REPO_ROOT / "outputs" / "umlp_d01" / "candidates_2wiki_gate.npz"
    if not side.exists():
        pytest.skip("D0.1 sidecar not present")
    with np.load(side) as f:
        z = {k: f[k] for k in ("column_names", "read_columns")}
    screen = json.loads((REPO_ROOT / "outputs" / "universal_v2" / "feature_screen.json").read_text(encoding="utf-8"))
    sets = D.feature_sets(z, screen)
    assert sets["A"] == sets["B"] and len(sets["A"]) == 130 and len(sets["C"]) == 165
    for cols in sets.values():
        assert not any(("gnn" in c or "twin" in c or "rank" in c.replace("seed_rank", "")) for c in cols)


def test_frozen_configs_unchanged():
    rec = REPO_ROOT / "outputs" / "umlp_d02" / "record.json"
    if not rec.exists():
        pytest.skip("record not present")
    pinned = json.loads(rec.read_text(encoding="utf-8"))["frozen_config_sha256"]
    for p in D.FROZEN_CONFIGS:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == pinned[p.name]
