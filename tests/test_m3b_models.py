"""The three arms and the training loop on a toy substrate: the step-0
identity (every arm returns the fixed base z-score exactly), the control's
independence from the edges, the metrics against brute force, and one short
fit that runs end to end through the cache."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mp_retrieval import m3b_features as F  # noqa: E402
from mp_retrieval import m3b_models as M  # noqa: E402
from mp_retrieval import m3b_pools as P  # noqa: E402
from mp_retrieval import m3b_train as T  # noqa: E402


class _Nodes:
    def __init__(self, matrix):
        self.matrix = matrix

    def read(self, rows):
        return self.matrix[np.asarray(rows)].astype(np.float32)


def _graph(rng, n, m, weighted, typed):
    src = rng.integers(0, n, size=m).astype(np.int32)
    dst = rng.integers(0, n, size=m).astype(np.int32)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    return SimpleNamespace(src=src, dst=dst, weight=rng.random(src.size).astype(np.float32) if weighted else None,
                           rel=rng.integers(0, 4, size=src.size).astype(np.int16) if typed else None, n_nodes=n, directed=not weighted)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = np.random.default_rng(1)
    n = 400
    stores = {"structural": P.FamilyStore.from_graph(_graph(rng, n, 1200, False, True), "structural"),
              "ner": P.FamilyStore.from_graph(_graph(rng, n, 900, True, False), "ner"),
              "knn": P.FamilyStore.from_graph(_graph(rng, n, 1600, True, False), "knn")}
    emb = rng.normal(size=(n, F.DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    rel_emb = rng.normal(size=(4, F.DIM)).astype(np.float32)
    rel_emb /= np.linalg.norm(rel_emb, axis=1, keepdims=True)
    rel = F.RelationTable.from_arrays(rel_emb, stores["structural"].rel_count, n)
    ctx = T.DatasetContext("toy", stores, _Nodes(emb), rel)
    base = tmp_path_factory.mktemp("cache")
    for carve, count in (("fit", 40), ("select", 12)):
        writer = T.CacheWriter(base / carve, {"dataset": "toy", "carve": carve})
        for i in range(count):
            q = rng.normal(size=F.DIM).astype(np.float32)
            q /= np.linalg.norm(q)
            # a planted signal: gold nodes are the nearest to q among a random subset
            dense_ids = np.argsort(-(emb @ q))[:100].astype(np.int64)
            splade_ids = rng.permutation(n)[:100].astype(np.int64)
            inp = F.QueryInputs(q, dense_ids, np.sort(emb[dense_ids] @ q)[::-1].astype(np.float32), splade_ids, np.sort(rng.random(100) * 10)[::-1].astype(np.float32))
            seeds = P.seeds_of(dense_ids, splade_ids)
            pool = np.union1d(np.union1d(dense_ids[:40], splade_ids[:40]), P.expand_hops(seeds, list(stores.values()), {"hops": 1, "per_seed_cap": 8}))
            pool = np.union1d(pool, seeds)
            compiled = F.compile_query(inp, pool, seeds, stores, ctx.nodes, rel)
            gold = np.sort(np.searchsorted(pool, dense_ids[[0, 2, 5]]))
            writer.add(f"q{i}", i, q, compiled, gold, 3)
        writer.write()
    fit = T.CarveData(base / "fit", ctx)
    select = T.CarveData(base / "select", ctx)
    return SimpleNamespace(fit=fit, select=select, ctx=ctx)


def test_every_arm_returns_the_base_z_score_at_step_zero(world):
    batch = world.fit.pack(np.arange(4))
    base_index = F.IDX["rrf"]
    z = M.segment_zscore(batch.x, batch.node_query, batch.n_queries)[:, base_index]
    for model in (M.QLSU(F.N_COLUMNS, 64, base_index), M.UniversalGAT(F.N_COLUMNS, 64, base_index, layers=2),
                  M.UniversalGAT(F.N_COLUMNS, 64, base_index, layers=2, message_passing=False)):
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(batch), z, atol=1e-6, rtol=0)


def test_the_control_passes_no_message_edges_and_the_gat_passes_them_all(world):
    """The input block reads the edges for the fixed C prototypes in every arm
    (they are part of F(q, v)); what distinguishes the control is that its
    GATv2 layers receive no message edges."""
    batch = world.fit.pack(np.arange(3))
    assert batch.edge_index.shape[1] > 0
    for mp in (False, True):
        model = M.UniversalGAT(F.N_COLUMNS, 64, F.IDX["rrf"], layers=2, message_passing=mp)
        seen = []
        original = model.convs[0].forward

        def spy(x, edge_index, edge_attr=None, _orig=original):
            seen.append(int(edge_index.shape[1]))
            return _orig(x, edge_index, edge_attr)

        model.convs[0].forward = spy
        model.eval()
        with torch.no_grad():
            model(batch)
        assert seen == [batch.edge_index.shape[1] if mp else 0]


def test_zscore_and_log_softmax_match_per_query_torch(world):
    batch = world.fit.pack(np.arange(5))
    z = M.segment_zscore(batch.x, batch.node_query, batch.n_queries)
    ptr = batch.qptr.tolist()
    for i in range(batch.n_queries):
        x = batch.x[ptr[i]:ptr[i + 1]]
        std = x.std(0, unbiased=False)
        expected = torch.where(std < 1e-6, torch.zeros_like(x), (x - x.mean(0)) / std.clamp_min(1e-6))
        torch.testing.assert_close(z[ptr[i]:ptr[i + 1]], expected, atol=1e-4, rtol=1e-4)
    scores = torch.randn(batch.x.shape[0])
    lp = M.segment_log_softmax(scores, batch.node_query, batch.n_queries)
    for i in range(batch.n_queries):
        torch.testing.assert_close(lp[ptr[i]:ptr[i + 1]], torch.log_softmax(scores[ptr[i]:ptr[i + 1]], 0), atol=1e-5, rtol=1e-5)


def test_rank_metrics_against_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(5, 60))
        scores = rng.normal(size=n)
        gold = np.unique(rng.integers(0, n, size=int(rng.integers(0, 4))))
        total = int(gold.size + rng.integers(0, 3))
        m = T.rank_metrics(scores, gold, total)
        order = list(np.argsort(-scores, kind="stable"))
        ranks = sorted(order.index(g) + 1 for g in gold)
        for k in T.KS:
            assert m[f"recall@{k}"] == pytest.approx(sum(r <= k for r in ranks) / max(total, 1))
        assert m["mrr"] == pytest.approx(1 / ranks[0] if ranks else 0.0)
        assert m["hit@1"] == float(bool(ranks) and ranks[0] == 1)
        assert m["hit@1"] <= m["mrr"] <= 1.0
        assert T.mrr_audit(np.asarray([m["first_gold_rank"]]), np.asarray([m["mrr"]]))["max_abs_diff"] < 1e-12


def test_a_short_fit_runs_through_the_cache_and_improves_on_the_planted_signal(world):
    torch.set_num_threads(2)
    base_index = F.IDX["rrf"]
    model = M.QLSU(F.N_COLUMNS, 32, base_index)
    before = T.evaluate_carve(model, world.select)["recall@5"].mean()
    model, record = T.fit_model(model, {"toy": world.fit}, {"toy": world.select}, seed=0, arm="qls_u_sota_v1", config={"H": 32},
                                max_epochs=3, batches_per_epoch=25, batch_size=4, patience=3, log=lambda *_: None)
    after = T.evaluate_carve(model, world.select)["recall@5"].mean()
    assert record.epochs_run == 3 and record.parameters > 0 and record.steps > 0
    assert after >= before
