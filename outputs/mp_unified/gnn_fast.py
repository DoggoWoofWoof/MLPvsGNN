"""Exact serving form of the six GNN's forward (UGNNv2, u_gnn_v2_ef, eval mode): the same function on the same
weights, with the query-independent and step-invariant terms computed once. A serving form, not a new model.

What moves (profiled at batch 1 on a 2wiki-median pool, 107 nodes and 463 edges, one thread: 20.9 ms forward):
  - the relation-bank projection, all 7,067 relation-text rows of the six (1536 -> 32), is a model constant; the
    original computes it inside every forward (10.7 ms of the 20.9), even on a graph none of whose edges reads it;
  - the cell's step-invariant edge terms: query(q_state), qh[v] * rel1(r_all), rel2(r_all) and msg_r(r_all), with
    the self-loop indices and rows, once per query instead of once per step (the cell is one instance, shared);
  - the evidence flow's attention beta and in-degree, once per query; with no typed STRUCT edge every step returns
    p unchanged (torch.where on an all-false mask), so the state is p^0 and the steps are skipped;
  - row gathers use index_select (the same copies as advanced indexing, a faster kernel); at batch 1 the semantic
    difference column broadcasts the one query row instead of gathering it to every node (the same elementwise
    values, the same matrix-vector product).
Every floating-point operation keeps the original's operands and order, so the scores are the original's bit for
bit: the self-test checks torch.equal on random batches (batch 1 and 3, untyped and typed edges, empty bank and
non-empty), and lean_time5 checks every timed query.

Optional index-time store (FastGNN(..., node_store=...)): node_projection(e_v) for every node of the graph,
computed once at index time like the lean MLP's store, so a query gathers 64 floats per node instead of projecting
1536. A GEMM need not round a row the same at every batch height, so this variant is checked to tolerance
(max |diff|) and top-5 identity, not bitwise.

    python outputs/mp_unified/gnn_fast.py --selftest
"""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from mp_retrieval import universal_v2_models as UM  # noqa: E402
from mp_retrieval.m3b_features import FAMILIES, MAX_SEEDS  # noqa: E402
from mp_retrieval.m3b_models import segment_zscore  # noqa: E402

NF = len(FAMILIES)


def segment_softmax(score, index, n):
    """universal_v2_models.segment_softmax with index_select gathers: the same operands in the same order."""
    H = score.shape[1]
    smax = torch.full((n, H), -float("inf"), dtype=score.dtype, device=score.device)
    smax = smax.scatter_reduce(0, index.unsqueeze(1).expand(-1, H), score, reduce="amax", include_self=True)
    ex = (score - smax.index_select(0, index)).exp()
    denom = torch.zeros(n, H, dtype=score.dtype, device=score.device).index_add_(0, index, ex)
    return ex / denom.index_select(0, index).clamp_min(1e-30)


def segment_mean_rows(values, index, n):
    """m3b_models.segment_mean_rows, unchanged in arithmetic."""
    out = torch.zeros(n, values.shape[1], dtype=values.dtype, device=values.device).index_add_(0, index, values)
    ones = torch.ones(index.numel(), dtype=values.dtype, device=values.device)
    count = torch.zeros(n, dtype=values.dtype, device=values.device).index_add_(0, index, ones)
    return out / count.clamp_min(1.0).unsqueeze(1), count


class FastGNN:
    """Serve a loaded UGNNv2 in eval mode. ``node_store`` (optional): (n_nodes, 64) float32 node projections, and
    then every call passes ``pool`` (the global node ids of the batch rows)."""

    def __init__(self, model, node_store=None):
        if not isinstance(model, UM.UGNNv2):
            raise TypeError("FastGNN serves a UGNNv2")
        if model.training:
            raise ValueError("FastGNN serves a model in eval mode (dropout is the identity there)")
        self.m = model
        self.all_families = bool(model.family_mask.all())
        with torch.inference_mode():
            self.bank_proj = model.relations.project_bank(model.relation_bank.to(model.readout.out.weight.dtype))
        self.node_store = node_store

    # ── the input block: InputBlock.forward copied, q_state handed on ──
    def _input(self, batch, pool=None):
        inp = self.m.input
        B = batch.n_queries
        N = batch.x.shape[0]
        P = inp.semantic.projection_dim
        nq = batch.node_query
        z = segment_zscore(batch.x, nq, B)
        base_z = z[:, inp.base_index]
        pre_q = inp.semantic.query_projection(batch.qemb).index_select(0, nq)
        raw_q = Fn.gelu(pre_q)
        q_state = Fn.normalize(raw_q, dim=-1)
        emb = batch.emb if batch.emb.dtype == torch.float32 else batch.emb.to(torch.float32)
        if self.node_store is not None:
            pre_n = self.node_store.index_select(0, pool)
        else:
            pre_n = inp.semantic.node_projection(emb)
        raw_n = Fn.gelu(pre_n)
        n_state = Fn.normalize(raw_n, dim=-1)
        semantic = torch.cat([
            q_state, n_state, q_state * n_state, (q_state - n_state).abs(),
            (q_state * n_state).sum(-1, keepdim=True), (raw_q * raw_n).sum(-1, keepdim=True) / raw_n.shape[-1] ** 0.5,
        ], dim=-1)
        q_rows = batch.qemb if B == 1 else batch.qemb.index_select(0, nq)
        difference = ((emb - q_rows).abs() @ inp.difference_weight).unsqueeze(1)
        del emb
        parts = [batch.x, z, semantic, difference]
        has_edges = batch.edge_attr.shape[0] > 0
        fam_of_edge = batch.edge_attr[:, :NF].argmax(dim=1) if has_edges else None
        for f_i in range(NF):
            if fam_of_edge is None:
                parts.append(torch.zeros(N, P, dtype=batch.x.dtype, device=batch.x.device))
                continue
            sel = fam_of_edge == f_i
            u, v = batch.edge_index[0, sel], batch.edge_index[1, sel]
            proto_pre, count = segment_mean_rows(pre_n.index_select(0, u), v, N)
            p_state = Fn.normalize(Fn.gelu(proto_pre), dim=-1)
            p_state = torch.where((count > 0).unsqueeze(1), p_state, torch.zeros_like(p_state))
            parts.append(p_state * q_state)
        seed_pre = torch.zeros(B, MAX_SEEDS, P, dtype=pre_n.dtype, device=pre_n.device)
        valid = batch.seed_nodes >= 0
        seed_pre[valid] = pre_n[batch.seed_nodes[valid]]
        w = batch.seedw
        reach_pre = torch.einsum("ns,nsp->np", w, seed_pre.index_select(0, nq)) / w.sum(1, keepdim=True).clamp_min(1e-12)
        r_state = Fn.normalize(Fn.gelu(reach_pre), dim=-1)
        r_state = torch.where((w.sum(1) > 0).unsqueeze(1), r_state, torch.zeros_like(r_state))
        parts.append(r_state * n_state)
        h = Fn.gelu(inp.linear(torch.cat(parts, dim=-1)))
        return h, base_z

    def __call__(self, batch, pool=None):
        with torch.inference_mode():
            return self._forward(batch, pool)

    def _forward(self, batch, pool=None):
        m = self.m
        h, base_z = self._input(batch, pool)
        B, N = batch.n_queries, h.shape[0]
        nq = batch.node_query
        q_state = Fn.normalize(Fn.gelu(m.input.semantic.query_projection(batch.qemb)), dim=-1).index_select(0, nq)
        if m.message_passing and batch.edge_attr.shape[0]:
            if self.all_families:
                edge_index, edge_attr = batch.edge_index, batch.edge_attr
            else:
                keep = m.family_mask[batch.edge_attr[:, :NF].argmax(dim=1)]
                edge_index, edge_attr = batch.edge_index[:, keep], batch.edge_attr[keep]
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, UM.N_EDGE_FEATURES_V2, dtype=h.dtype)
        r = m.relations(edge_attr, self.bank_proj)
        # ── the shared cell, its step-invariant terms once ──
        c = m.cell
        H, K = c.hidden, c.heads
        loops = torch.arange(N, dtype=torch.long)
        u = torch.cat([edge_index[0], loops])
        v = torch.cat([edge_index[1], loops])
        r_all = torch.cat([r, m.relations.self_loop.unsqueeze(0).expand(N, -1)])
        qh = c.query(q_state)
        qr = qh.index_select(0, v) * c.rel1(r_all)
        r2 = c.rel2(r_all)
        mr = c.msg_r(r_all)
        for _ in range(m.steps):
            e = c.src(h).index_select(0, u) + c.dst(h).index_select(0, v)
            e += qr
            e += r2
            e = Fn.leaky_relu(e, 0.2)
            score = (e.view(-1, K, H // K) * c.att).sum(-1)
            alpha = segment_softmax(score, v, N)
            msg = (c.msg_h(h).index_select(0, u) + mr).view(-1, K, H // K) * alpha.unsqueeze(-1)
            mm = torch.zeros(N, K, H // K, dtype=h.dtype).index_add_(0, v, msg).view(N, H)
            g = torch.sigmoid(c.gate(torch.cat([h, mm, q_state], dim=-1)))
            h = (1.0 - g) * h + g * c.norm(h + mm)
        out = h
        if m.evidence is not None:
            ev = m.evidence
            p = ev.initial(UM.evidence_features(batch.x, nq, B, m.evidence_index))
            if edge_attr.shape[0]:
                typed = (edge_attr[:, UM.STRUCT_FAMILY] > 0.5) & (edge_attr[:, UM.ATTR["rel_mask"]] > 0.5)
                t_index, t_r = edge_index[:, typed], r[typed]
            else:
                t_index, t_r = edge_index, r
            if t_index.shape[1]:
                tu, tv = t_index[0], t_index[1]
                score2 = ((ev.query(q_state).index_select(0, tv) * ev.rel(t_r)) @ ev.att).unsqueeze(1)
                beta = segment_softmax(score2, tv, N)
                ones = torch.ones(tv.numel(), dtype=p.dtype)
                has_in = (torch.zeros(N, dtype=p.dtype).index_add_(0, tv, ones) > 0).unsqueeze(1)
                for _ in range(m.steps):
                    m2 = torch.zeros(N, ev.d_p, dtype=p.dtype).index_add_(0, tv, beta * ev.prop(p).index_select(0, tu))
                    g2 = torch.sigmoid(ev.gate(torch.cat([p, m2, q_state], dim=-1)))
                    p = torch.where(has_in, (1.0 - g2) * p + g2 * m2, p)
            out = torch.cat([h, p], dim=-1)
        return m.readout(out, base_z)


def node_store_of(model, read_rows, n_nodes, chunk=4096):
    """node_projection(e) for every node, at index time: read_rows(ids) -> (len, 1536) float16/32 rows."""
    w = model.input.semantic.node_projection
    out = torch.zeros(n_nodes, w.out_features, dtype=torch.float32)
    with torch.inference_mode():
        for s in range(0, n_nodes, chunk):
            ids = np.arange(s, min(s + chunk, n_nodes))
            e = torch.as_tensor(np.asarray(read_rows(ids)))
            out[s:s + ids.size] = w(e.to(torch.float32))
    return out


# ── self-test ──


def _batch(rng, sizes, typed, slots, bank_rows, n_scalars=129):
    from mp_retrieval.m3b_models import PackedBatch
    xs, nqs, embs, ei, ea, seeds, seedw, qptr = [], [], [], [], [], [], [], [0]
    off = 0
    for q, n in enumerate(sizes):
        E = int(rng.integers(0, 5 * n))
        xs.append(rng.random((n, n_scalars), dtype=np.float32))
        nqs.append(np.full(n, q))
        embs.append(rng.standard_normal((n, 1536)).astype(np.float16))
        ei.append(rng.integers(0, n, (2, E)) + off)
        fam = rng.choice(3, E, p=[0.567, 0.307, 0.126])
        a = np.zeros((E, UM.N_EDGE_FEATURES_V2), np.float32)
        a[np.arange(E), fam] = 1.0
        a[:, UM.ATTR["weight"]] = rng.random(E)
        fw = (fam == 0) & (rng.random(E) < 0.5)
        a[:, UM.ATTR["dir_fwd"]] = fw
        a[:, UM.ATTR["dir_bwd"]] = (fam == 0) & ~fw
        a[:, UM.SLOT0:] = -1
        if typed:
            tm = (fam == 0) & (rng.random(E) < 0.7)
            a[:, UM.ATTR["rel_mask"]] = tm
            a[:, UM.ATTR["rel_compat"]] = rng.random(E) * tm
            if slots:
                k = rng.integers(1, UM.K_REL + 1, E)
                for j in range(UM.K_REL):
                    a[:, UM.SLOT0 + j] = np.where(tm & (j < k), rng.integers(0, bank_rows, E), -1)
        ea.append(a)
        s = np.full(MAX_SEEDS, -1)
        ns = int(rng.integers(0, min(MAX_SEEDS, n) + 1))
        s[:ns] = rng.choice(n, ns, replace=False) + off
        seeds.append(s)
        w = rng.random((n, MAX_SEEDS)).astype(np.float32)
        w[:, ns:] = 0
        seedw.append(w)
        off += n
        qptr.append(off)
    return PackedBatch(x=torch.from_numpy(np.concatenate(xs)), qptr=torch.tensor(qptr), node_query=torch.from_numpy(np.concatenate(nqs)),
                       emb=torch.from_numpy(np.concatenate(embs)), qemb=torch.from_numpy(rng.standard_normal((len(sizes), 1536)).astype(np.float32)),
                       seedw=torch.from_numpy(np.concatenate(seedw)), seed_nodes=torch.from_numpy(np.stack(seeds)),
                       edge_index=torch.from_numpy(np.concatenate(ei, 1)), edge_attr=torch.from_numpy(np.concatenate(ea)),
                       gold=torch.zeros(off, dtype=torch.bool))


def _model(seed, bank_rows, families=FAMILIES):
    torch.manual_seed(seed)
    m = UM.UGNNv2(129, 128, 3, 0.2, 3, 4, 32, [0, 1, 2, 3], 8, families=families)
    with torch.no_grad():
        for p in m.parameters():   # the readout is zero-initialised; give every weight a trained-like scale
            p.add_(torch.randn_like(p) * 0.05)
    m.set_relation_bank(torch.randn(bank_rows, 1536) / 1536 ** 0.5)
    return m.eval()


def selftest():
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261003)
    n_eq = 0
    for case, (sizes, typed, slots, bank_rows, families) in enumerate([
            ((107,), False, False, 0, FAMILIES), ((107,), False, False, 50, FAMILIES), ((60,), True, True, 50, FAMILIES),
            ((40, 1, 75), True, True, 50, FAMILIES), ((33, 20), True, False, 0, FAMILIES), ((50,), True, True, 50, ("structural", "ner")),
            ((1,), False, False, 50, FAMILIES), ((90,), True, True, 50, FAMILIES)]):
        for rep in range(3):
            m = _model(100 * case + rep, bank_rows, families)
            b = _batch(rng, sizes, typed, slots, max(bank_rows, 1))
            if case == 7 and rep == 2:   # no edges at all
                b = b.__class__(**{**b.__dict__, "edge_index": b.edge_index[:, :0], "edge_attr": b.edge_attr[:0]})
            with torch.no_grad():
                ref = m(b)
            got = FastGNN(m)(b)
            assert torch.equal(ref, got), (case, rep, float((ref - got).abs().max()))
            n_eq += 1
            # the node store: the same projections gathered by pool id
            pool = torch.from_numpy(rng.permutation(5000)[:b.x.shape[0]])
            store = torch.zeros(5000, 64)
            with torch.no_grad():
                store[pool] = m.input.semantic.node_projection(b.emb.to(torch.float32))
            got_s = FastGNN(m, node_store=store)(b, pool)
            assert float((ref - got_s).abs().max()) <= 1e-4, (case, rep, float((ref - got_s).abs().max()))
    # the bank is hoisted: the fast form never projects it per call
    m = _model(7, 50)
    f = FastGNN(m)
    calls = []
    orig = m.relations.project_bank
    m.relations.project_bank = lambda bank: calls.append(1) or orig(bank)
    f(_batch(rng, (30,), True, True, 50))
    assert not calls
    m.relations.project_bank = orig
    for bad in (lambda: FastGNN(_model(1, 0).train()), lambda: FastGNN(torch.nn.Linear(2, 2))):
        try:
            bad()
            raise AssertionError("FastGNN accepted a model it does not serve")
        except (ValueError, TypeError):
            pass
    print(f"selftest: {n_eq} random batches (batch 1 and 3; untyped and typed edges; slots into a bank and an empty bank; "
          "a family subset; a one-node pool; no edges) score bit for bit as UGNNv2.forward; the node-store variant is "
          "within 1e-4; the bank is projected once at load. all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        raise SystemExit("usage: gnn_fast.py --selftest (FastGNN is imported by lean_time5.py)")
