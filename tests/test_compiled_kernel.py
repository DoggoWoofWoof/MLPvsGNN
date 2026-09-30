"""The compiled-kernel arm (src/mp_retrieval/compiled_kernel.py, configs/deploy_ck_2wiki.yaml#tests): level 1's JL
matrix; at W_msg = 0 the arm is u_mlp_v2_mix exactly, and at the same seed it starts from u_mlp_v2_mix's weights; the
query-free neighbour side; the weights; the compiled form; entry order; the arm reads edges after the input block
(an MP arm); the self-entry control reads no edge; the parameter count against the GNN budget; gradients reach the
kernel."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mp_retrieval import compiled_kernel as CK  # noqa: E402
from mp_retrieval import m3b_features as F  # noqa: E402
from mp_retrieval import m3b_models as M3  # noqa: E402
from mp_retrieval import m3b_pools as P  # noqa: E402
from mp_retrieval import m3b_train as T  # noqa: E402
from mp_retrieval import universal_v2_features as V  # noqa: E402
from mp_retrieval import universal_v2_models as M  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _dataset(rng, name, n, typed, base, count=6):
    """A toy dataset as tests/test_universal_v2_models.py builds one (typed: a relation table; untyped: 2wiki-style)."""
    stores = {"structural": P.FamilyStore.from_graph(_graph(rng, n, 3 * n, False, typed, n_rel=5 if typed else 1), "structural"),
              "ner": P.FamilyStore.from_graph(_graph(rng, n, n, True, False), "ner"),
              "knn": P.FamilyStore.from_graph(_graph(rng, n, 2 * n, True, False), "knn")}
    emb = _unit(rng, (n, F.DIM))
    rel = F.RelationTable.from_arrays(_unit(rng, (5, F.DIM)), stores["structural"].rel_count, n) if typed else None
    ctx = T.DatasetContext(name, stores, _Nodes(emb), rel)
    writer = V.CacheWriterV2(base / name / "fit", {"dataset": name, "kind": "fit", "feature_contract": V.CONTRACT_NAME, "n_columns": V.N_COLUMNS})
    for i in range(count):
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
    rng = np.random.default_rng(3)
    base = tmp_path_factory.mktemp("ckcache")
    contexts = {"typed": _dataset(rng, "typed", 300, True, base), "untyped": _dataset(rng, "untyped", 260, False, base)}
    _, ctx2 = M.build_relation_bank(contexts)
    core = [c for c in F.COLUMNS if F.GROUP_OF[c] in ("retrieval", "topology")] + list(V.V2_COLUMNS)[:40]
    idx = np.asarray([V.IDX[c] for c in core], dtype=np.int64)
    inputs = {"columns": core, "base_local": core.index("rrf")}
    fits = {n: M.CarveDataV2(base / n / "fit", ctx2[n], columns=idx) for n in contexts}
    return SimpleNamespace(inputs=inputs, fits=fits)


def _batch(world, name="typed", k=4):
    return world.fits[name].pack(np.arange(k))


def _ck(world, side="qi", form="KERN", seed=0, hidden=64, msg=True):
    torch.manual_seed(seed)
    model = CK.build_ck(world.inputs, hidden=hidden, side=side, form=form)
    if msg:   # a non-zero message map, so the kernel reaches the score
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            model.kernel.w_msg.weight.copy_(torch.randn(model.kernel.w_msg.weight.shape, generator=g) * 0.2)
            model.readout.out.weight.copy_(torch.randn(model.readout.out.weight.shape, generator=g) * 0.3)
    model.eval()
    return model


def _hq(model, batch):
    with torch.no_grad():
        h, _ = model.input(batch)
        return h, model.q_state(batch)


# ── level 1's projection; the twin at W_msg = 0 ──────────────────────────────


def test_the_jl_matrix_is_level_ones():
    L1 = _load("mp_approx_l1_for_ck", "scripts/mp_approx_l1.py")
    assert np.array_equal(CK.jl_matrix(), L1.jl_matrix())
    assert (CK.JL_SEED, CK.JL_DIM) == (L1.JL_SEED, L1.JL_DIM)


def test_at_zero_message_map_the_arm_is_the_twin_and_starts_from_its_weights(world):
    for side, form in CK.COMBOS:
        torch.manual_seed(7)
        twin = M.build_arm("u_mlp_v2_mix", world.inputs, hidden=64)
        ck = _ck(world, side=side, form=form, seed=7, msg=False)
        twin_state = twin.state_dict()
        ck_state = ck.state_dict()
        assert set(twin_state) <= set(ck_state)
        assert all(torch.equal(twin_state[k], ck_state[k]) for k in twin_state)      # same seed: the same starting weights
        g = torch.Generator().manual_seed(11)
        with torch.no_grad():                                                       # a readout that reads h
            w = torch.randn(twin.readout.out.weight.shape, generator=g)
            twin.readout.out.weight.copy_(w)
            ck.readout.out.weight.copy_(w)
        twin.eval()
        batch = _batch(world)
        with torch.no_grad():
            assert torch.equal(ck(batch), twin(batch))


# ── the neighbour side ───────────────────────────────────────────────────────


def _perturbed(batch, cols=(), emb=False):
    b = M3.PackedBatch(**{f: getattr(batch, f).clone() for f in batch.__dataclass_fields__})
    g = torch.Generator().manual_seed(5)
    for c in cols:
        b.edge_attr[:, c] = torch.rand(b.edge_attr.shape[0], generator=g)
    if emb:
        b.emb = (b.emb.to(torch.float32) + 0.1 * torch.randn(b.emb.shape, generator=g)).to(b.emb.dtype)
    return b


def test_the_query_free_side_reads_nothing_that_depends_on_the_query_or_the_pool(world):
    batch = _batch(world)
    ck = _ck(world, "qi")
    h, q = _hq(ck, batch)
    g = torch.Generator().manual_seed(9)
    h2, q2 = torch.randn(h.shape, generator=g), torch.randn(q.shape, generator=g)
    with torch.no_grad():
        _, _, psi, val = ck.kernel.parts(batch, h, q)
        _, _, psi_b, val_b = ck.kernel.parts(batch, h2, q2)                         # the query-dependent row and the query
        wb = _perturbed(batch, cols=(M.ATTR["weight"], M.ATTR["rel_compat"]))       # the pool-relative weight and rel_compat
        _, _, psi_c, val_c = ck.kernel.parts(wb, h, q)
        assert torch.equal(psi, psi_b) and torch.equal(val, val_b)
        assert torch.equal(psi, psi_c) and torch.equal(val, val_c)
        _, _, psi_d, val_d = ck.kernel.parts(_perturbed(batch, emb=True), h, q)     # X_u R
        _, _, psi_e, val_e = ck.kernel.parts(_perturbed(batch, cols=(M.ATTR["dir_fwd"],)), h, q)
        assert not torch.equal(psi, psi_d) and not torch.equal(val, val_d)
        assert not torch.equal(psi, psi_e) and not torch.equal(val, val_e)
        C, c, _ = ck.kernel.moments(batch, h, q)
        C2, c2, _ = ck.kernel.moments(wb, h2, q2)
        assert torch.equal(C, C2) and torch.equal(c, c2)


def test_the_weight_side_reads_the_weight_and_the_full_side_reads_h(world):
    batch = _batch(world)
    qw, full = _ck(world, "qw"), _ck(world, "full")
    h, q = _hq(qw, batch)
    g = torch.Generator().manual_seed(9)
    h2 = torch.randn(h.shape, generator=g)
    wb = _perturbed(batch, cols=(M.ATTR["weight"],))
    with torch.no_grad():
        assert torch.equal(qw.kernel.parts(batch, h, q)[2], qw.kernel.parts(batch, h2, q)[2])
        assert not torch.equal(qw.kernel.parts(batch, h, q)[2], qw.kernel.parts(wb, h, q)[2])
        assert not torch.equal(full.kernel.parts(batch, h, q)[2], full.kernel.parts(batch, h2, q)[2])
    assert CK.entry_width("qi", 64) == CK.JL_DIM + 4 + 3 and CK.entry_width("qw", 64) == CK.JL_DIM + 4 + 4
    assert CK.entry_width("full", 64) == 64 + 4 + 5


# ── weights, the compiled form, entry order ──────────────────────────────────


def test_the_weights_are_positive_and_sum_to_one_and_the_mean_is_uniform(world):
    batch = _batch(world)
    n = batch.x.shape[0]
    for form in CK.FORMS:
        ck = _ck(world, "qi", form)
        h, q = _hq(ck, batch)
        with torch.no_grad():
            _, alpha = ck.kernel(batch, h, q, return_alpha=True)
            _, v, _ = ck.kernel.entries(batch, n)
            sums = torch.zeros(n, CK.HEADS, dtype=alpha.dtype).index_add_(0, v, alpha)
        assert bool((alpha > 0).all())
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-6, rtol=0)
        if form == "MEAN":
            indeg = torch.zeros(n).index_add_(0, batch.edge_index[1], torch.ones(batch.edge_index.shape[1]))
            torch.testing.assert_close(alpha[:, 0], (1.0 / (indeg + 1.0))[v], atol=1e-7, rtol=0)
        if form == "SELF":
            assert torch.equal(v, torch.arange(n)) and bool((alpha == 1.0).all())


def test_the_compiled_form_equals_the_forward_output(world):
    for name in ("typed", "untyped"):
        batch = _batch(world, name)
        for side, form in CK.COMBOS:
            ck = _ck(world, side, form)
            h, q = _hq(ck, batch)
            with torch.no_grad():
                torch.testing.assert_close(ck.kernel.compiled(batch, h, q), ck.kernel(batch, h, q), atol=2e-5, rtol=1e-4)
                torch.testing.assert_close(ck.forward_compiled(batch), ck(batch), atol=2e-5, rtol=1e-4)


def test_the_output_does_not_depend_on_entry_order(world):
    batch = _batch(world)
    ck = _ck(world, "qi")
    g = torch.Generator().manual_seed(4)
    perm = torch.randperm(batch.edge_index.shape[1], generator=g)
    shuffled = M3.PackedBatch(**{f: getattr(batch, f) for f in batch.__dataclass_fields__})
    shuffled.edge_index, shuffled.edge_attr = batch.edge_index[:, perm], batch.edge_attr[perm]
    with torch.no_grad():
        torch.testing.assert_close(ck(shuffled), ck(batch), atol=2e-5, rtol=1e-4)


# ── an MP arm: it reads the edges after the input block ──────────────────────


def test_the_arm_reads_edges_after_the_input_block(world):
    ck = _ck(world, "qi")
    batch = _batch(world)
    with torch.no_grad():
        full = ck(batch)
    original = ck.input.forward

    def stripped(b, _orig=original):
        out = _orig(b)
        b.edge_index = torch.empty(2, 0, dtype=torch.long)
        b.edge_attr = torch.empty(0, b.edge_attr.shape[1], dtype=b.edge_attr.dtype)
        return out

    ck.input.forward = stripped
    with torch.no_grad():
        assert not torch.allclose(ck(_batch(world)), full)


def test_the_self_control_reads_no_edge_and_is_the_value_map_of_the_nodes_own_row(world):
    batch = _batch(world)
    ck = _ck(world, "qi", "SELF")
    assert ck.arm == "ck_self" and not hasattr(ck.kernel, "w_phi") and not hasattr(ck.kernel, "w_psi")
    with pytest.raises(ValueError):
        CK.OneHopKernel(64, "full", "SELF")
    h, q = _hq(ck, batch)
    n = h.shape[0]
    bare = M3.PackedBatch(**{f: getattr(batch, f) for f in batch.__dataclass_fields__})
    bare.edge_index = torch.empty(2, 0, dtype=torch.long)
    bare.edge_attr = torch.empty(0, batch.edge_attr.shape[1], dtype=batch.edge_attr.dtype)
    with torch.no_grad():
        m = ck.kernel(batch, h, q)
        assert torch.equal(m, ck.kernel(bare, h, q))                                                  # no edge is read
        assert torch.equal(m, ck.kernel(_perturbed(batch, cols=tuple(range(batch.edge_attr.shape[1]))), h, q))
        side = torch.zeros(n, len(CK.ENTRY_FAMILIES) + 3)
        side[:, len(CK.FAMILIES)] = 1.0
        rows = batch.emb.to(torch.float32) @ ck.kernel.jl
        want = torch.nn.functional.linear(torch.cat([rows, side], dim=1), ck.kernel.w_val.weight, ck.kernel.w_val.bias)
        torch.testing.assert_close(m, want, atol=2e-6, rtol=1e-5)
        assert not torch.equal(m, ck.kernel(_perturbed(batch, emb=True), h, q))


# ── budget and gradients ─────────────────────────────────────────────────────


def test_the_parameter_count_is_within_the_gnn_budget():
    UV2 = _load("universal_v2_run_for_ck", "scripts/universal_v2_run.py")
    cfg, cfg_m3b, _ = UV2.load_configs()
    inputs = UV2.model_inputs(cfg, cfg_m3b)
    twin = M3.parameter_count(M.build_arm("u_mlp_v2_mix", inputs))
    counts = {(s, f): M3.parameter_count(CK.build_ck(inputs, side=s, form=f)) for s, f in CK.COMBOS}
    assert twin == 330_955
    kern_qi = (128 + 64) * 128 + 128 + (64 + 7) * 128 + 128 + (64 + 7) * 64 + 64 + 64 * 128 + 128
    assert counts[("qi", "KERN")] == twin + kern_qi == 377_803
    assert counts[("qi", "SELF")] == twin + (64 + 7) * 64 + 64 + 64 * 128 + 128 == 343_883
    assert all(c <= M.PARAMETER_BUDGET["gnn"] for c in counts.values())


def test_gradients_reach_the_kernel_after_the_message_map_moves(world):
    """The readout starts at zero (the step-0 identity), so the body and W_msg move from step 1 and the kernel from step 2."""
    torch.manual_seed(0)
    ck = CK.build_ck(world.inputs, hidden=64, side="qi")
    ck.train()
    opt = torch.optim.AdamW(ck.parameters(), lr=1e-2)
    batch = _batch(world)

    def grad(p):
        return 0.0 if p.grad is None else float(p.grad.abs().sum())

    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss = M3.listwise_loss(ck(batch), batch)
        loss.backward()
        assert torch.isfinite(loss)
        kernel = [grad(ck.kernel.w_psi.weight), grad(ck.kernel.w_val.weight), grad(ck.kernel.w_phi.weight)]
        if step == 0:
            assert grad(ck.kernel.w_msg.weight) == 0.0 and kernel == [0.0, 0.0, 0.0]
        elif step == 1:
            assert grad(ck.kernel.w_msg.weight) > 0 and kernel == [0.0, 0.0, 0.0]
        else:
            assert all(k > 0 for k in kernel)
        opt.step()
