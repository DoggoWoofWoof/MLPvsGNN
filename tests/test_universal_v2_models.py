"""The v2 arms on a toy substrate (configs/universal_v2.yaml
#authorization_2026_09_19.step_2_authorised.tests_required_beyond_shapes and
#training.step_0_identity, #learned_vs_fixed_propagation.guard): the step-0
identity of every arm, the twin guard (body receives only (h, base_z); the
score is unchanged when every edge is removed after the input block), the
U-GNN reading the edges, message passing disabled reducing the learned
propagation contribution exactly to zero, the fixed-score residual, the shared
cell, the evidence-flow initialisation, no dataset identity, mixed batches,
one state dict over typed and structural edges, the parameter budgets, the
gated block, the v2 pack against the pinned pack, the sampler and loss
identity."""

from __future__ import annotations

import inspect
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
from mp_retrieval import m3b_models as M3  # noqa: E402
from mp_retrieval import m3b_pools as P  # noqa: E402
from mp_retrieval import m3b_train as T  # noqa: E402
from mp_retrieval import universal_v2_features as V  # noqa: E402
from mp_retrieval import universal_v2_models as M  # noqa: E402


class _Nodes:
    def __init__(self, matrix):
        self.matrix = matrix

    def read(self, rows, dtype=np.float32):
        return self.matrix[np.asarray(rows)].astype(dtype)


def _graph(rng, n, m, weighted, typed, n_rel=4):
    src = rng.integers(0, n, size=m).astype(np.int32)
    dst = rng.integers(0, n, size=m).astype(np.int32)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    return SimpleNamespace(src=src, dst=dst, weight=rng.random(src.size).astype(np.float32) if weighted else None,
                           rel=rng.integers(0, n_rel, size=src.size).astype(np.int16) if typed else None, n_nodes=n, directed=not weighted)


def _unit(rng, shape):
    x = rng.normal(size=shape).astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _dataset(rng, name, n, typed, base, count=24):
    """A toy dataset: typed (metaqa-style, relation table) or untyped (2wiki-style hyperlinks: one relation, no table)."""
    stores = {"structural": P.FamilyStore.from_graph(_graph(rng, n, 3 * n, False, typed, n_rel=5 if typed else 1), "structural"),
              "ner": P.FamilyStore.from_graph(_graph(rng, n, n, True, False), "ner"),
              "knn": P.FamilyStore.from_graph(_graph(rng, n, 2 * n, True, False), "knn")}
    emb = _unit(rng, (n, F.DIM))
    rel = F.RelationTable.from_arrays(_unit(rng, (5, F.DIM)), stores["structural"].rel_count, n) if typed else None
    ctx = T.DatasetContext(name, stores, _Nodes(emb), rel)
    for carve, k in (("fit", count), ("select", 8)):
        writer = V.CacheWriterV2(base / name / carve, {"dataset": name, "kind": carve, "feature_contract": V.CONTRACT_NAME, "n_columns": V.N_COLUMNS})
        for i in range(k):
            q = _unit(rng, F.DIM)
            dense_ids = np.argsort(-(emb @ q))[:60].astype(np.int64)
            splade_ids = rng.permutation(n)[:60].astype(np.int64)
            inp = F.QueryInputs(q, dense_ids, np.sort(emb[dense_ids] @ q)[::-1].astype(np.float32), splade_ids, np.sort(rng.random(60) * 10)[::-1].astype(np.float32))
            seeds = P.seeds_of(dense_ids, splade_ids)
            pool = np.union1d(np.union1d(dense_ids[:30], splade_ids[:30]), P.expand_hops(seeds, list(stores.values()), {"hops": 1, "per_seed_cap": 6}))
            pool = np.union1d(pool, seeds)
            compiled = V.compile_query_v2(inp, pool, seeds, stores, ctx.nodes, rel)
            gold = np.sort(np.searchsorted(pool, dense_ids[[0, 2, 5]]))
            writer.add(f"{name}-q{i}", i, q, compiled, gold, 3)
        writer.write()
    return ctx


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = np.random.default_rng(1)
    base = tmp_path_factory.mktemp("v2cache")
    contexts = {"typed": _dataset(rng, "typed", 300, True, base), "untyped": _dataset(rng, "untyped", 260, False, base)}
    bank, ctx2 = M.build_relation_bank(contexts)
    # a synthetic core: the M3B retrieval / topology columns plus every v2 column (the screen is not run here)
    # (splade_score_norm left out as the M3B screen dropped it, so the evidence substitution is exercised)
    core78 = [c for c in F.COLUMNS if F.GROUP_OF[c] in ("retrieval", "topology", "typed_relations_B") and c != "splade_score_norm"]
    core = core78 + list(V.V2_COLUMNS)
    idx = np.asarray([V.IDX[c] for c in core], dtype=np.int64)
    idx78 = np.asarray([V.IDX[c] for c in core78], dtype=np.int64)
    pairs = [{"earlier": "splade_rr", "later": "splade_score_norm", "abs_spearman": 0.9985}]
    names, subs = M.resolve_evidence_columns(core, pairs)
    inputs = {"columns": core, "base_local": core.index("rrf"), "evidence_local": [core.index(c) for c in names],
              "core78_columns": core78, "core78_base_local": core78.index("rrf"), "core78_evidence_local": [core78.index(c) for c in names]}
    fits = {n: M.CarveDataV2(base / n / "fit", ctx2[n], columns=idx) for n in contexts}
    selects = {n: M.CarveDataV2(base / n / "select", ctx2[n], columns=idx) for n in contexts}
    fits78 = {n: T.CarveData(base / n / "fit", ctx2[n], columns=idx78) for n in contexts}
    return SimpleNamespace(base=base, contexts=contexts, ctx2=ctx2, bank=bank, core=core, core78=core78, idx=idx, idx78=idx78,
                           inputs=inputs, fits=fits, selects=selects, fits78=fits78, substitutions=subs)


def _arm(world, arm, **kw):
    model = M.build_arm(arm, world.inputs, hidden=64, selected_gnn="u_gnn_v2_ef", **kw)
    if isinstance(model, M.UGNNv2):
        model.set_relation_bank(world.bank)
    model.eval()
    return model


def _randomise_readout(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.readout.out.weight.copy_(torch.randn(model.readout.out.weight.shape, generator=g) * 0.3)
        model.readout.out.bias.fill_(0.1)


def _strip_edges_after_input(model):
    """The input block runs on the full batch; every edge is removed before the
    rest of the model sees it. A score that changes reads edges after the block."""
    original = model.input.forward

    def wrapped(batch, _orig=original):
        out = _orig(batch)
        batch.edge_index = torch.empty(2, 0, dtype=torch.long)
        batch.edge_attr = torch.empty(0, batch.edge_attr.shape[1], dtype=batch.edge_attr.dtype)
        return out

    model.input.forward = wrapped


def _batch(world, name, k=3, model=None):
    """The batch as the given arm reads it: the control is packed 8 wide by the
    pinned reader over the M3B core; the core78 ablation reads a column view of
    the v2 batch; every other arm reads the v2 batch."""
    if isinstance(model, M3.UniversalGAT):
        return world.fits78[name].pack(np.arange(k))
    full = world.fits[name].pack(np.arange(k))
    if getattr(model, "arm", None) == "u_gnn_v2_core78":
        return M.batch_view(full, [world.core.index(c) for c in world.core78], M.N_EDGE_FEATURES_V2)
    return full


# ── step-0 identity and the residual ─────────────────────────────────────────


def test_every_arm_returns_the_base_z_score_at_step_zero(world):
    for name in world.contexts:
        for arm in M.ARMS:
            model = _arm(world, arm)
            batch = _batch(world, name, model=model)
            z = M3.segment_zscore(batch.x, batch.node_query, batch.n_queries)[:, model.input.base_index]
            with torch.no_grad():
                torch.testing.assert_close(model(batch), z, atol=1e-6, rtol=0)


def test_the_fixed_score_residual_reproduces_s_fixed_exactly_when_the_correction_is_zero(world):
    batch = _batch(world, "typed")
    model = _arm(world, "u_gnn_v2_ef")
    with torch.no_grad():
        score = model(batch)
    assert model.last_correction is not None and float(model.last_correction.abs().max()) == 0.0
    torch.testing.assert_close(score, model.readout.base_weight * model.last_base_z, atol=0, rtol=0)
    readout = M3.ResidualReadout(16)
    h = torch.randn(10, 16)
    base_z = torch.randn(10)
    torch.testing.assert_close(readout(h, base_z), base_z, atol=0, rtol=0)


# ── the twin guard and the propagation boundary ──────────────────────────────


def test_the_twin_body_receives_only_h_and_base_z_and_never_the_batch(world):
    for arm in M.TWIN_CANDIDATES:
        model = _arm(world, arm)
        seen = []
        for module in (model.body, model.readout):
            orig = module.forward

            def spy(*args, _orig=orig, **kwargs):
                seen.append(tuple(type(a) for a in args) + tuple(type(v) for v in kwargs.values()))
                return _orig(*args, **kwargs)

            module.forward = spy
        with torch.no_grad():
            model(_batch(world, "typed"))
        assert seen and all(all(t is torch.Tensor for t in call) for call in seen)
        assert not any(M3.PackedBatch in call for call in seen)


def test_the_twin_score_is_unchanged_when_every_edge_is_removed_after_the_input_block(world):
    for arm in M.TWIN_CANDIDATES:
        batch = _batch(world, "typed")                   # fresh per arm: the strip mutates the batch
        assert batch.edge_index.shape[1] > 0
        model = _arm(world, arm)
        _randomise_readout(model)
        with torch.no_grad():
            before = model(batch)
        _strip_edges_after_input(model)
        with torch.no_grad():
            after = model(batch)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        assert float((before - M3.segment_zscore(batch.x, batch.node_query, batch.n_queries)[:, model.input.base_index]).abs().max()) > 1e-3


def test_the_u_gnn_reads_the_edges_after_the_input_block(world):
    for arm in M.GNN_CANDIDATES:
        batch = _batch(world, "typed")
        model = _arm(world, arm)
        _randomise_readout(model)
        with torch.no_grad():
            before = model(batch)
        _strip_edges_after_input(model)
        with torch.no_grad():
            after = model(batch)
        assert float((before - after).abs().max()) > 1e-5, arm


def test_disabling_message_passing_reduces_the_learned_propagation_contribution_exactly_to_zero(world):
    batch = _batch(world, "typed")
    on = M.UGNNv2(len(world.core), 64, world.inputs["base_local"], evidence_index=world.inputs["evidence_local"], message_passing=True)
    off = M.UGNNv2(len(world.core), 64, world.inputs["base_local"], evidence_index=world.inputs["evidence_local"], message_passing=False)
    off.load_state_dict(on.state_dict())
    for model in (on, off):
        model.set_relation_bank(world.bank)
        model.eval()
    _randomise_readout(on)
    _randomise_readout(off)
    with torch.no_grad():
        s_on, s_off = on(batch), off(batch)
    assert float((s_on - s_off).abs().max()) > 1e-5              # with edges, learned propagation contributes
    _strip_edges_after_input(off)
    with torch.no_grad():
        s_off_stripped = off(batch)
    torch.testing.assert_close(s_off, s_off_stripped, atol=0, rtol=0)   # message passing off: no edge reaches the cell
    ptr = batch.qptr.tolist()
    for i in range(batch.n_queries):                                   # and a node depends on nothing beyond itself
        single = world.fits["typed"].pack(np.asarray([i]))
        with torch.no_grad():
            torch.testing.assert_close(off(single), s_off[ptr[i]:ptr[i + 1]], atol=1e-5, rtol=1e-5)


# ── the shared cell, the evidence flow, the relation bank ────────────────────


def test_the_shared_recurrent_cell_shares_its_parameters_across_steps(world):
    model = _arm(world, "u_gnn_v2")
    cells = [m for m in model.modules() if isinstance(m, M.QueryRelationCell)]
    assert len(cells) == 1 and model.steps == 3
    calls = []
    orig = model.cell.forward

    def spy(*args, _orig=orig, **kwargs):
        calls.append(id(model.cell))
        return _orig(*args, **kwargs)

    model.cell.forward = spy
    batch = _batch(world, "typed")
    with torch.no_grad():
        model(batch)
    assert calls == [id(model.cell)] * 3
    assert model.last_gates.shape == (3, batch.x.shape[0]) and bool((model.last_gates > 0).all()) and bool((model.last_gates < 1).all())
    assert 0 < sum(p.numel() for p in model.cell.parameters()) < M.parameter_count(model)


def test_the_evidence_flow_initialisation_comes_only_from_inference_safe_retrieval_columns(world):
    for name in M.EVIDENCE_COLUMNS:
        assert F.GROUP_OF[name] == "retrieval", name
    assert world.substitutions == {"splade_score_norm": "splade_rr"}
    assert [world.core[i] for i in world.inputs["evidence_local"]] == ["rrf", "dense_cos", "splade_rr", "is_seed"]
    batch = _batch(world, "typed")
    ev = M.evidence_features(batch.x, batch.node_query, batch.n_queries, world.inputs["evidence_local"])
    assert ev.shape == (batch.x.shape[0], 4)
    rrf = batch.x[:, world.inputs["base_local"]]
    for i in range(batch.n_queries):
        sl = slice(int(batch.qptr[i]), int(batch.qptr[i + 1]))
        torch.testing.assert_close(ev[sl, 0], rrf[sl] / rrf[sl].max(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ev[:, 1:], batch.x[:, world.inputs["evidence_local"][1:]], atol=0, rtol=0)
    other = torch.ones(batch.x.shape[1], dtype=torch.bool)          # every other column may change without touching p^0
    other[world.inputs["evidence_local"]] = False
    x2 = batch.x.clone()
    x2[:, other] = torch.randn(int(other.sum()))
    torch.testing.assert_close(M.evidence_features(x2, batch.node_query, batch.n_queries, world.inputs["evidence_local"]), ev, atol=0, rtol=0)
    model = _arm(world, "u_gnn_v2_ef")
    with torch.no_grad():
        model(batch)
    assert model.last_gates2.shape == (3, batch.x.shape[0])
    with pytest.raises(ValueError):
        M.resolve_evidence_columns([c for c in world.core if c not in ("splade_score_norm", "splade_rr")], [])


def test_the_relation_bank_is_served_data_not_a_weight(world):
    model = _arm(world, "u_gnn_v2")
    assert "relation_bank" not in model.state_dict()
    assert model.relation_bank.shape == (world.bank.n_rows, F.DIM) and world.bank.n_rows == 5 and world.bank.offsets == {"typed": 0}
    _randomise_readout(model)
    batch = _batch(world, "typed")
    assert int((batch.edge_attr[:, M.SLOT0:] >= 0).sum()) > 0
    with torch.no_grad():
        before = model(batch)
    model.set_relation_bank(torch.zeros_like(model.relation_bank))
    with torch.no_grad():
        after = model(batch)
    assert float((before - after).abs().max()) > 1e-6


# ── no dataset identity, mixed batches, one state dict ───────────────────────


def test_no_dataset_id_enters_either_model(world):
    assert "dataset" not in inspect.signature(M.build_arm).parameters
    assert set(f for f in M3.PackedBatch.__dataclass_fields__) == {"x", "qptr", "node_query", "emb", "qemb", "seedw", "seed_nodes", "edge_index", "edge_attr", "gold"}
    for arm in M.ARMS:
        model = _arm(world, arm)
        assert not any("dataset" in n for n, _ in model.named_parameters()) and not any("dataset" in n for n, _ in model.named_buffers())
    # the same queries under another dataset name are the same batch and the same scores
    ctx = world.ctx2["typed"]
    renamed = M.DatasetContextV2("something_else", ctx.stores, ctx.nodes, ctx.rel_table, ctx.rel_offset)
    a = M.CarveDataV2(world.base / "typed" / "fit", ctx, columns=world.idx).pack(np.arange(3))
    b = M.CarveDataV2(world.base / "typed" / "fit", renamed, columns=world.idx).pack(np.arange(3))
    for field in M3.PackedBatch.__dataclass_fields__:
        torch.testing.assert_close(getattr(a, field), getattr(b, field), atol=0, rtol=0)


def _centred(model, batch):
    with torch.no_grad():
        s = model(batch)
    mean = torch.zeros(batch.n_queries).index_add_(0, batch.node_query, s) / torch.bincount(batch.node_query, minlength=batch.n_queries)
    return s - mean[batch.node_query]


def test_one_batch_may_contain_queries_of_several_datasets(world):
    typed, untyped = world.fits["typed"].pack(np.arange(2)), world.fits["untyped"].pack(np.arange(3))
    mixed = T.concat_batches([typed, untyped])
    assert mixed.n_queries == 5 and mixed.edge_attr.shape[1] == M.N_EDGE_FEATURES_V2
    for arm in ("u_mlp_v2", "u_mlp_v2_mix", "u_gnn_v2", "u_gnn_v2_ef"):
        model = _arm(world, arm)
        _randomise_readout(model)
        joint = _centred(model, mixed)
        parts = torch.cat([_centred(model, typed), _centred(model, untyped)])
        torch.testing.assert_close(joint, parts, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        loss = M3.listwise_loss(model(mixed), mixed)
    assert torch.isfinite(loss)


def test_one_state_dict_handles_typed_and_structural_edges(world):
    typed, untyped = world.fits["typed"].pack(np.arange(3)), world.fits["untyped"].pack(np.arange(3))
    rel_mask = M.ATTR["rel_mask"]
    struct_t = typed.edge_attr[:, M.STRUCT_FAMILY] > 0.5
    struct_u = untyped.edge_attr[:, M.STRUCT_FAMILY] > 0.5
    assert bool((typed.edge_attr[struct_t, rel_mask] == 1).all()) and bool((typed.edge_attr[struct_t, M.SLOT0] >= 0).all())
    assert bool((untyped.edge_attr[struct_u, rel_mask] == 0).all()) and bool((untyped.edge_attr[:, M.SLOT0:] == -1).all())
    first = _arm(world, "u_gnn_v2_ef")
    _randomise_readout(first)
    second = M.UGNNv2(len(world.core), 64, world.inputs["base_local"], evidence_index=world.inputs["evidence_local"])
    second.load_state_dict(first.state_dict())
    second.set_relation_bank(world.bank)
    second.eval()
    for batch in (typed, untyped):
        with torch.no_grad():
            torch.testing.assert_close(first(batch), second(batch), atol=0, rtol=0)
    with torch.no_grad():
        s_u = first(untyped)
    assert float((s_u - M3.segment_zscore(untyped.x, untyped.node_query, untyped.n_queries)[:, first.input.base_index]).abs().max()) > 1e-4


# ── budgets, the gated block, the v2 pack, the sampler and loss identity ─────


def test_parameter_budgets_hold_at_the_164_column_screen_input(world):
    """The declared budgets (GNN arms <= 450,000, twins <= 350,000) at the widest
    possible core: 78 + 86 = 164 columns (amendment 2); the control is the
    M3B GAT count."""
    core = [f"c{i}" for i in range(78)] + list(V.V2_COLUMNS)
    assert len(core) == 164
    inputs = {"columns": core, "base_local": 0, "evidence_local": [0, 1, 2, 3], "core78_columns": core[:78], "core78_base_local": 0,
              "core78_evidence_local": [0, 1, 2, 3]}
    counts = {}
    for arm in M.ARMS:
        model = M.build_arm(arm, inputs, hidden=128, selected_gnn="u_gnn_v2_ef")
        counts[arm] = M.check_parameter_budget(arm, model)
    assert counts["gat_universal_v1_trio"] == 353_410
    assert max(counts[a] for a in M.GNN_CANDIDATES) <= 450_000 and max(counts[a] for a in M.TWIN_CANDIDATES) <= 350_000
    assert counts["u_gnn_v2_ef"] > counts["u_gnn_v2"] and counts["u_mlp_v2_mix"] > counts["u_mlp_v2"]
    with pytest.raises(ValueError):
        M.check_parameter_budget("u_mlp_v2", M.build_arm("u_mlp_v2", inputs, hidden=512))


def test_the_gated_input_block_with_open_gates_is_the_m3b_input_block(world):
    batch = _batch(world, "typed")
    block_index = M.block_index_of(world.core)
    assert int((block_index >= 0).sum()) == 86 and block_index.max() == 8 and (block_index[: len(world.core78)] == -1).all()
    plain = M3.InputBlock(len(world.core), 64, world.inputs["base_local"], dropout=0.0)
    gated = M.GatedInputBlock(len(world.core), 64, world.inputs["base_local"], block_index, dropout=0.0)
    gated.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        gated.gate_bias.fill_(30.0)
    plain.eval()
    gated.eval()
    with torch.no_grad():
        h_p, z_p = plain(batch)
        h_g, z_g = gated(batch)
    torch.testing.assert_close(h_g, h_p, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(z_g, z_p, atol=0, rtol=0)
    torch.testing.assert_close(gated.last_block_gates, torch.ones(batch.n_queries, 9), atol=1e-6, rtol=0)
    with torch.no_grad():
        gated.gate_bias.zero_()
        h_half, _ = gated(batch)
    torch.testing.assert_close(gated.last_block_gates, torch.full((batch.n_queries, 9), 0.5), atol=1e-6, rtol=0)
    assert float((h_half - h_p).abs().max()) > 1e-4
    model = _arm(world, "u_mlp_v2_mix")
    with torch.no_grad():
        model(batch)
    assert model.last_block_gates.shape == (batch.n_queries, 9)


def test_pack_queries_v2_equals_the_pinned_pack_on_every_shared_field_and_carries_the_relation_rows(world):
    for name in world.contexts:
        ctx = world.ctx2[name]
        data = world.fits[name]
        queries = [data.query(i) for i in range(4)]
        a = M.pack_queries_v2(queries, ctx)
        b = T.pack_queries(queries, ctx)
        for field in M3.PackedBatch.__dataclass_fields__:
            if field == "edge_attr":
                torch.testing.assert_close(a.edge_attr[:, : M3.N_EDGE_FEATURES], b.edge_attr, atol=0, rtol=0)
            else:
                torch.testing.assert_close(getattr(a, field), getattr(b, field), atol=0, rtol=0)
        assert a.edge_attr.shape[1] == M.N_EDGE_FEATURES_V2
        view = M.batch_view(a, np.arange(a.x.shape[1]), M3.N_EDGE_FEATURES)
        torch.testing.assert_close(view.edge_attr, b.edge_attr, atol=0, rtol=0)
        torch.testing.assert_close(view.x, b.x, atol=0, rtol=0)
        slots = a.edge_attr[:, M.SLOT0:]
        struct = a.edge_attr[:, M.STRUCT_FAMILY] > 0.5
        assert bool((slots[~struct] == -1).all())
        if ctx.rel_table is None:
            assert bool((slots == -1).all())
            continue
        # the slots of one query against the typed entries read directly
        qd = queries[0]
        (ev, eu, erel, _), pair_id, (u, v) = F.typed_pool_edges(ctx.stores["structural"], qd["pool"], F.IN_POOL_CAP)
        n0 = int(a.qptr[1])
        first = (a.edge_index[1] < n0) & struct
        rows = slots[first]
        assert rows.shape[0] == u.size
        for p in range(u.size):
            rels = erel[pair_id == p][: M.K_REL].astype(np.float32) + ctx.rel_offset
            expect = np.full(M.K_REL, -1.0, dtype=np.float32)
            expect[: rels.size] = rels
            np.testing.assert_array_equal(rows[p].numpy(), expect)
        stats = M.relation_slot_stats(pair_id)
        assert stats["pairs"] == u.size and stats["entries"] == ev.size and 0 <= stats["truncated"] <= stats["pairs"]


def test_the_sampler_and_the_loss_are_the_pinned_m3b_functions(world, monkeypatch):
    """The v2 fit runs mp_retrieval.m3b_train.fit_model itself over CarveDataV2:
    the batches it draws are the batches the pinned sampler draws for the same
    seed, and the loss is m3b_models.listwise_loss. One short fit of the twin
    proves the loop runs end to end over the v2 cache."""
    torch.set_num_threads(2)
    drawn = []
    original = T.draw_indices

    def record(*args, **kwargs):
        out = original(*args, **kwargs)
        drawn.append([(n, idx.copy()) for n, idx in out])
        return out

    monkeypatch.setattr(T, "draw_indices", record)
    losses = []
    original_loss = T.listwise_loss

    def loss_spy(scores, batch):
        value = original_loss(scores, batch)
        losses.append(float(value))
        return value

    monkeypatch.setattr(T, "listwise_loss", loss_spy)
    model = M.build_arm("u_mlp_v2", world.inputs, hidden=32)
    model, record_ = T.fit_model(model, world.fits, world.selects, seed=0, arm="u_mlp_v2", config={"H": 32}, max_epochs=1,
                                 batches_per_epoch=6, batch_size=4, patience=2, log=lambda *_: None)
    assert record_.epochs_run == 1 and record_.steps == 6 and len(losses) == 6 and all(np.isfinite(losses))
    # the same draws again, from the pinned sampler alone
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    names = sorted(world.fits)
    cursors = {n: [rng.permutation(world.fits[n].trainable), 0] for n in names}
    expect = [original(world.fits, names, cursors, rng, 4, "per_query") for _ in range(6)]
    assert len(drawn) == 6
    for got, exp in zip(drawn, expect):
        assert [n for n, _ in got] == [n for n, _ in exp]
        for (_, g), (_, e) in zip(got, exp):
            np.testing.assert_array_equal(g, e)
    assert T.listwise_loss is loss_spy and original_loss is M3.listwise_loss
