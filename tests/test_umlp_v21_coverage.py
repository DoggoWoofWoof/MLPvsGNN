"""UMLP-v2.1 objective checks: the coverage term, its detached top-5, the patched loss, the gate's population."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
spec = importlib.util.spec_from_file_location("umlp_v21", REPO_ROOT / "scripts" / "umlp_v21_coverage.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def toy_batch(sizes, golds):
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    gold = torch.zeros(int(ptr[-1]), dtype=torch.bool)
    for q, g in enumerate(golds):
        for i in g:
            gold[ptr[q] + i] = True
    nq = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    return SimpleNamespace(qptr=torch.tensor(ptr), gold=gold, node_query=nq, n_queries=len(sizes))


def brute(s, gold, sizes, top_h=5):
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    tot, has = 0.0, 0
    for q in range(len(sizes)):
        ss, gg = s[ptr[q]:ptr[q + 1]], gold[ptr[q]:ptr[q + 1]]
        if gg.sum() == 0:
            continue
        has += 1
        if gg.sum() < 2:
            continue
        neg = np.sort(ss[~gg])[::-1][:top_h]
        tot += np.mean([np.log1p(np.exp(n - g)) for g in ss[gg] for n in neg])
    return tot / has


def test_coverage_matches_brute_force_and_single_gold_is_zero():
    sizes, golds = [12, 9, 7, 5], [[0, 3, 5], [2], [], [0, 1]]
    b = toy_batch(sizes, golds)
    s = torch.randn(sum(sizes), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    got = float(V.coverage_loss(s, b))
    assert got == pytest.approx(brute(s.numpy(), b.gold.numpy(), sizes), rel=1e-10)
    single = toy_batch([6, 6], [[1], [4]])
    assert float(V.coverage_loss(torch.randn(12), single)) == 0.0


def test_top_h_selection_carries_no_gradient():
    b = toy_batch([10], [[0, 1]])
    s = torch.arange(10, dtype=torch.float64).flip(0).clone().requires_grad_(True)   # golds 9, 8; non-golds 7..0
    V.coverage_loss(s, b).backward()
    g = s.grad.numpy()
    assert (g[2:7] > 0).all() and (g[7:] == 0).all() and (g[:2] < 0).all()   # only the top-5 non-golds and the golds move


def test_patched_loss_is_listwise_plus_coverage_and_restored():
    from mp_retrieval import m3b_train
    from mp_retrieval.m3b_models import listwise_loss
    orig = m3b_train.listwise_loss
    b = toy_batch([8, 6], [[0, 2], [1]])
    s = torch.randn(14, dtype=torch.float64)
    with V.patched_loss() as obj:
        assert m3b_train.listwise_loss is obj
        assert float(obj(s, b)) == pytest.approx(float(listwise_loss(s, b) + V.LAMBDA * V.coverage_loss(s, b)))
    assert m3b_train.listwise_loss is orig


def test_declared_constants():
    decl = V.yaml.safe_load(V.CONFIG.read_text(encoding="utf-8"))
    assert V.LAMBDA == float(decl["intervention"]["lambda"].split()[0]) == 1.0
    assert V.TOP_H == 5 and V.HIT1_TOL == -0.005 and V.ARM == "u_mlp_v2_mix"


def test_gate_reads_v2_gate_rows_only():
    p = REPO_ROOT / "outputs" / "universal_v2" / "eval" / "2wiki.npz"
    if not p.exists():
        pytest.skip("v2 eval not present")
    half, stored = V.stored_control("2wiki", [V.key_of(V.CONTROL, 0)])
    assert half.sum() == 6290 and all(v.size == 6290 for v in stored.values())


def test_frozen_configs_unchanged():
    for p in V.FROZEN_CONFIGS:
        assert p.exists()
    rec = REPO_ROOT / "outputs" / "umlp_d02" / "record.json"
    if rec.exists():
        import json
        pinned = json.loads(rec.read_text(encoding="utf-8"))["frozen_config_sha256"]
        for name, digest in pinned.items():
            assert hashlib.sha256((REPO_ROOT / "configs" / name).read_bytes()).hexdigest() == digest
