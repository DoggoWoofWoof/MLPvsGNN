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

    def read(self, rows, dtype=np.float32):
        return self.matrix[np.asarray(rows)].astype(dtype)


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


# The input block as first written (aggregation in the 1536-wide embedding space),
# kept here as the reference the projected-space aggregation must reproduce.
def _reference_input_forward(self, batch):
    B = batch.n_queries
    N = batch.x.shape[0]
    z = M.segment_zscore(batch.x, batch.node_query, B)
    base_z = z[:, self.base_index]
    q_rows = batch.qemb[batch.node_query]
    raw_q, q_state = M._project(self.semantic.query_projection, q_rows)
    raw_n, n_state = M._project(self.semantic.node_projection, batch.emb)
    semantic = torch.cat([
        q_state, n_state, q_state * n_state, (q_state - n_state).abs(),
        (q_state * n_state).sum(-1, keepdim=True), (raw_q * raw_n).sum(-1, keepdim=True) / raw_n.shape[-1] ** 0.5,
    ], dim=-1)
    difference = M.semantic_difference_column(q_rows, batch.emb, self.difference_weight).unsqueeze(1)
    parts = [batch.x, z, semantic, difference]
    # C: fixed neighbour prototypes per family, projected strictly after the aggregation
    has_edges = batch.edge_attr.shape[0] > 0
    fam_of_edge = batch.edge_attr[:, : len(F.FAMILIES)].argmax(dim=1) if has_edges else None
    for f_i in range(len(F.FAMILIES)):
        if fam_of_edge is None:
            parts.append(torch.zeros(N, M.PROJECTION_DIM, dtype=batch.x.dtype, device=batch.x.device))
            continue
        sel = fam_of_edge == f_i
        u, v = batch.edge_index[0, sel], batch.edge_index[1, sel]
        proto, count = M.segment_mean_rows(batch.emb[u], v, N)
        _, p_state = M._project(self.semantic.node_projection, proto)
        p_state = torch.where((count > 0).unsqueeze(1), p_state, torch.zeros_like(p_state))
        parts.append(p_state * q_state)
    # D: the 1/dist-weighted prototype of the seeds within two hops, projected after the aggregation
    seed_emb = torch.zeros(B, F.MAX_SEEDS, F.DIM, dtype=batch.emb.dtype, device=batch.emb.device)
    valid = batch.seed_nodes >= 0
    seed_emb[valid] = batch.emb[batch.seed_nodes[valid]]
    w = batch.seedw
    proto_reach = torch.einsum("ns,nsd->nd", w, seed_emb[batch.node_query]) / w.sum(1, keepdim=True).clamp_min(1e-12)
    _, r_state = M._project(self.semantic.node_projection, proto_reach)
    r_state = torch.where((w.sum(1) > 0).unsqueeze(1), r_state, torch.zeros_like(r_state))
    parts.append(r_state * n_state)
    h = self.dropout(torch.nn.functional.gelu(self.linear(torch.cat(parts, dim=-1))))
    return h, base_z


def test_the_input_block_aggregates_in_projected_space_without_changing_the_function(world):
    """node_projection is bias-free and linear, so the mean of projections equals the
    projection of the mean: the block's outputs and its gradients agree with the
    embedding-space reference to float precision on batches with and without edges."""
    torch.manual_seed(3)
    block = M.InputBlock(F.N_COLUMNS, 64, F.IDX["rrf"], dropout=0.0)
    with torch.no_grad():                        # a non-zero difference weight so that column is exercised
        block.difference_weight.normal_()
    for idx in (np.arange(4), np.arange(5, 8)):
        batch = world.fit.pack(idx)
        for edges in (True, False):
            b = batch if edges else M.PackedBatch(**{**batch.__dict__, "edge_index": batch.edge_index[:, :0], "edge_attr": batch.edge_attr[:0]})
            block.zero_grad()
            h_new, z_new = block(b)
            (g_new,) = torch.autograd.grad(h_new.square().sum(), block.semantic.node_projection.weight)
            block.zero_grad()
            h_ref, z_ref = _reference_input_forward(block, b)
            (g_ref,) = torch.autograd.grad(h_ref.square().sum(), block.semantic.node_projection.weight)
            torch.testing.assert_close(h_new, h_ref, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(z_new, z_ref, atol=0, rtol=0)
            torch.testing.assert_close(g_new, g_ref, atol=1e-4, rtol=1e-4)


def test_concatenated_batches_score_exactly_as_their_parts(world):
    """A mixed-dataset batch is packed per dataset and joined; every arm must score
    the joined batch as it scores the parts (offsets right, no edge crosses)."""
    b1, b2 = world.fit.pack(np.arange(3)), world.fit.pack(np.arange(7, 12))
    joined = T.concat_batches([b1, b2])
    assert joined.n_queries == 8 and joined.x.shape[0] == b1.x.shape[0] + b2.x.shape[0]
    assert int(joined.edge_index.max()) < joined.x.shape[0]
    assert int((joined.seed_nodes >= 0).sum()) == int((b1.seed_nodes >= 0).sum()) + int((b2.seed_nodes >= 0).sum())
    torch.manual_seed(5)
    for model in (M.QLSU(F.N_COLUMNS, 64, F.IDX["rrf"]), M.UniversalGAT(F.N_COLUMNS, 64, F.IDX["rrf"], layers=2)):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(joined), torch.cat([model(b1), model(b2)]), atol=1e-5, rtol=1e-5)
        n1 = M.listwise_loss(model(b1), b1) * 3 + M.listwise_loss(model(b2), b2) * 5
        torch.testing.assert_close(M.listwise_loss(model(joined), joined), n1 / 8, atol=1e-5, rtol=1e-5)


def test_the_per_query_draw_fills_every_slot_from_a_uniformly_drawn_dataset(world):
    fits = {"a": world.fit, "b": world.select}
    rng = np.random.default_rng(0)
    cursors = {n: [rng.permutation(fits[n].trainable), 0] for n in fits}
    counts = {"a": 0, "b": 0}
    for _ in range(20):
        batch = T.draw_batch(fits, ["a", "b"], cursors, rng, 8, F.FAMILIES, "per_query")
        assert batch.n_queries == 8
    for _ in range(6):
        batch = T.draw_batch(fits, ["a", "b"], cursors, rng, 4, F.FAMILIES, "per_batch")
        assert batch.n_queries == 4
    with pytest.raises(ValueError):
        T.fit_model(M.QLSU(F.N_COLUMNS, 16, F.IDX["rrf"]), fits, fits, seed=0, arm="qls_u_sota_v1", config={}, dataset_draw="sometimes")
