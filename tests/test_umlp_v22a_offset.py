"""UMLP-v2.2A checks: identity at w = 0, frozen base, the pair/protect terms, the patched loss."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
spec = importlib.util.spec_from_file_location("umlp_v22a", REPO_ROOT / "scripts" / "umlp_v22a_offset.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def toy_batch(sizes, golds, n_cols=4, seed=0):
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    gold = torch.zeros(int(ptr[-1]), dtype=torch.bool)
    for q, g in enumerate(golds):
        for i in g:
            gold[ptr[q] + i] = True
    nq = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    x = torch.randn(int(ptr[-1]), n_cols, generator=torch.Generator().manual_seed(seed))
    return SimpleNamespace(qptr=torch.tensor(ptr), gold=gold, node_query=nq, n_queries=len(sizes), x=x)


class ToyBase(nn.Module):
    def __init__(self, n_cols):
        super().__init__()
        self.lin = nn.Linear(n_cols, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, batch):
        return self.drop(self.lin(batch.x)).squeeze(1)


def test_identity_at_zero_and_frozen_base():
    b = toy_batch([7, 5], [[1, 3], [0]])
    base = ToyBase(4)
    m = V.make_offset_model(base, 4)
    m.train()
    assert not base.training                          # the base stays in eval mode under train()
    assert [n for n, p in m.named_parameters() if p.requires_grad] == ["w"]
    s = m(b)
    s0 = base(b).detach()
    for a, e in ((0, 7), (7, 12)):
        assert np.array_equal(np.argsort(-s[a:e].detach().numpy(), kind="stable"), np.argsort(-s0[a:e].numpy(), kind="stable"))
    assert float(m.last[1].abs().max()) == 0.0


def test_secondary_pairs_and_protect():
    # one query: golds at 0 (top) and 3; non-golds 1, 2, 4..9 with s0 descending by index
    b = toy_batch([10], [[0, 3]])
    s0 = torch.tensor([10.0, 9, 8, 1, 7, 6, 5, 4, 3, 2], dtype=torch.float64)
    delta = torch.zeros(10, dtype=torch.float64)
    # L_secondary = mean over extra {3} x top-5 non-golds {1,2,4,5,6} of softplus(s_n - s_g); protect 0 (margin unchanged)
    want = np.mean([np.log1p(np.exp(s0[n].item() - s0[3].item())) for n in (1, 2, 4, 5, 6)])
    got = float(V.offset_loss(s0.clone(), b, s0, delta))
    assert got == pytest.approx(want)
    # shrinking the first gold's margin by 0.5 costs lambda_p * 0.5
    s = s0.clone()
    s[0] -= 0.5
    got2 = float(V.offset_loss(s, b, s0, delta))
    want2 = np.mean([np.log1p(np.exp(s[n].item() - s[3].item())) for n in (1, 2, 4, 5, 6)]) + V.LAMBDA_P * 0.5
    assert got2 == pytest.approx(want2)


def test_protect_skips_when_base_top1_is_not_gold_and_size_term():
    b = toy_batch([4], [[2]])
    s0 = torch.tensor([3.0, 2, 1, 0], dtype=torch.float64)
    delta = torch.full((4,), 0.5, dtype=torch.float64)
    assert float(V.offset_loss(s0.clone(), b, s0, delta)) == pytest.approx(V.LAMBDA_D * 0.25)


def test_patched_loss_restored():
    from mp_retrieval import m3b_train
    orig = m3b_train.listwise_loss
    m = V.make_offset_model(ToyBase(4), 4)
    with V.patched_loss(m) as obj:
        assert m3b_train.listwise_loss is obj
    assert m3b_train.listwise_loss is orig


def test_declared_constants():
    decl = V.yaml.safe_load(V.CONFIG.read_text(encoding="utf-8"))
    o = decl["objective"]
    assert (V.TAU, V.LAMBDA_P, V.LAMBDA_D, V.TOP_N) == (float(str(decl["model"]["tau"]).split()[0]), o["lambda_p"], o["lambda_delta"], 5)


def test_frozen_configs_unchanged():
    for p in V.FROZEN_CONFIGS:
        assert p.exists()
    assert "status: RUN" in V21_text()


def V21_text():
    return (REPO_ROOT / "configs" / "umlp_v21_coverage_objective.yaml").read_text(encoding="utf-8")
