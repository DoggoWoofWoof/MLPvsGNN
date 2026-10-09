"""C3 (docs/C3_FASTEST_FORMS.md): the fastest exact serving form of every model, the GNNs included.

The six GNN (u_gnn_v2_ef) is served by gnn_fast.FastGNN, which already hoists the model constants and the cell's
step-invariant terms. Profiled cold at batch 1 on musique (about 2,150 nodes and 55,000 edges per pool), three
quarters of its forward is per-edge elementwise work on (edges x 128) tensors: gathers, adds, the attention sum, the
segment softmax and the scatter of the messages, three steps. FusedGNN keeps every module's own GEMMs in torch and
moves only that per-edge work into one numba pass per destination node:

  relation rows   the relation encoder and the cell's rel1, rel2 and msg_r read the edge attributes, whose rows repeat
                  (about 900 distinct rows of 55,000 on musique; one self-loop row): they run on the distinct rows
                  (a hash of each row's bits) and the pass gathers them per edge;
  attention       e = (src[u] + dst[v] + qh[v] * rel1[r]) + rel2[r], leaky_relu, the per-head dot with att, the
                  segment softmax over each node's in-edges (self-loop last) and the message sum, in the original
                  edge order per node;
  input block     each family's segment mean of the node projections over its edges, in edge order per node (the
                  same additions in the same order as index_add_).

The attention's per-head sum and exp are not torch's vectorised kernels, so FusedGNN is checked to a tolerance (the
scores within 1e-4 of FastGNN's, the same top 5), on random batches here and on every timed question in the harness.

    python outputs/mp_unified/c3_fast.py --selftest
"""
import os
import sys
from pathlib import Path

THREADS = 1
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
           "LEAN_TIME_THREADS"):
    os.environ[_v] = str(THREADS)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import gnn_fast as GF  # noqa: E402
import lean_mlp2 as L2  # noqa: E402
import lean_screen3 as S3  # noqa: E402
import zlink as ZL  # noqa: E402
import zprop as ZP  # noqa: E402

UM, NF, MAX_SEEDS = GF.UM, GF.NF, GF.MAX_SEEDS
TOL = 1e-4
FM = {"reassoc", "contract", "nsz", "arcp"}     # no nnan/ninf: -inf and NaN keep their meaning


# ── the per-edge kernels ─────────────────────────────────────────────────────


@numba.njit(cache=True, fastmath=False)
def csr_by_dst(v, n):
    """A stable sort of the edges by destination: order (edge ids, original order within a node) and ptr."""
    cnt = np.zeros(n + 1, np.int64)
    for j in range(v.size):
        cnt[v[j] + 1] += 1
    for t in range(n):
        cnt[t + 1] += cnt[t]
    pos = cnt[:-1].copy()
    order = np.empty(v.size, np.int64)
    for j in range(v.size):
        t = v[j]
        order[pos[t]] = j
        pos[t] += 1
    return order, cnt


@numba.njit(cache=True)
def unique_rows(W):
    """Distinct rows of an int32 view (bitwise equality), first-seen order: (first row of each, inverse)."""
    E, C = W.shape
    cap = 1
    while cap < 2 * E + 2:
        cap *= 2
    slot = np.full(cap, -1, np.int64)
    inv = np.empty(E, np.int64)
    first = np.empty(E, np.int64)
    nu = 0
    for j in range(E):
        hsh = np.uint64(1469598103934665603)
        for c in range(C):
            hsh = (hsh ^ np.uint64(np.uint32(W[j, c]))) * np.uint64(1099511628211)
        k = np.int64(hsh & np.uint64(cap - 1))
        while True:
            g = slot[k]
            if g < 0:
                slot[k] = nu
                first[nu] = j
                inv[j] = nu
                nu += 1
                break
            r = first[g]
            same = True
            for c in range(C):
                if W[r, c] != W[j, c]:
                    same = False
                    break
            if same:
                inv[j] = g
                break
            k = (k + 1) & (cap - 1)
    return first[:nu].copy(), inv


@numba.njit(cache=True, fastmath=FM)
def attend(S, D, QH, R1, R2, MH, MR, att, u, v, rid, order, ptr, K):
    """One cell step's messages: mm[t] = sum over t's in-edges j (original order) of
    (MH[u_j] + MR[r_j]) * softmax_t(score_j), score_j[k] = sum_h leaky((S[u_j] + D[t] + QH[t] * R1[r_j]) + R2[r_j]) * att."""
    N, H = S.shape
    Hk = H // K
    out = np.zeros((N, H), np.float32)
    maxdeg = 0
    for t in range(N):
        d = ptr[t + 1] - ptr[t]
        if d > maxdeg:
            maxdeg = d
    sc = np.empty((maxdeg, K), np.float32)
    e = np.empty(H, np.float32)
    for t in range(N):
        a, b = ptr[t], ptr[t + 1]
        if a == b:
            continue
        smax = np.full(K, -np.inf, np.float32)
        for jj in range(a, b):
            j = order[jj]
            uj, rj = u[j], rid[j]
            for h in range(H):
                x = S[uj, h] + D[t, h]
                x = x + QH[t, h] * R1[rj, h]
                x = x + R2[rj, h]
                e[h] = x if x > 0 else np.float32(0.2) * x
            for k in range(K):
                s = np.float32(0.0)
                for h in range(k * Hk, (k + 1) * Hk):
                    s += e[h] * att[k, h - k * Hk]
                sc[jj - a, k] = s
                if s > smax[k]:
                    smax[k] = s
        den = np.zeros(K, np.float32)
        for jj in range(a, b):
            for k in range(K):
                x = np.float32(np.exp(sc[jj - a, k] - smax[k]))
                sc[jj - a, k] = x
                den[k] += x
        for k in range(K):
            if den[k] < np.float32(1e-30):
                den[k] = np.float32(1e-30)
        for jj in range(a, b):
            j = order[jj]
            uj, rj = u[j], rid[j]
            for k in range(K):
                al = sc[jj - a, k] / den[k]
                for h in range(k * Hk, (k + 1) * Hk):
                    out[t, h] += (MH[uj, h] + MR[rj, h]) * al
    return out


@numba.njit(cache=True, fastmath=False)
def seg_mean_rows(P, u, v, order, ptr):
    """m3b_models.segment_mean_rows(P[u], v, N): sums in edge order per node, then the count (clamped at 1)."""
    N, D = P.shape
    out = np.zeros((N, D), np.float32)
    count = np.zeros(N, np.float32)
    for t in range(N):
        for jj in range(ptr[t], ptr[t + 1]):
            j = order[jj]
            for d in range(D):
                out[t, d] += P[u[j], d]
            count[t] += np.float32(1.0)
        c = count[t] if count[t] > np.float32(1.0) else np.float32(1.0)
        for d in range(D):
            out[t, d] = out[t, d] / c
    return out, count


# ── the served model ─────────────────────────────────────────────────────────


class FusedGNN(GF.FastGNN):
    """FastGNN with the per-edge work fused (numba); a serving form of the same UGNNv2."""

    def _input(self, batch, pool=None):
        inp = self.m.input
        B = batch.n_queries
        N = batch.x.shape[0]
        P = inp.semantic.projection_dim
        nq = batch.node_query
        z = GF.segment_zscore(batch.x, nq, B)
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
        fam_of_edge = batch.edge_attr[:, :NF].argmax(dim=1).numpy() if has_edges else None
        pre_np = pre_n.numpy()
        eu, ev = batch.edge_index[0].numpy(), batch.edge_index[1].numpy()
        for f_i in range(NF):
            if fam_of_edge is None:
                parts.append(torch.zeros(N, P, dtype=batch.x.dtype, device=batch.x.device))
                continue
            sel = fam_of_edge == f_i
            u, v = eu[sel], ev[sel]
            order, ptr = csr_by_dst(v, N)
            pm, count = seg_mean_rows(pre_np, u, v, order, ptr)
            p_state = Fn.normalize(Fn.gelu(torch.from_numpy(pm)), dim=-1)
            p_state = torch.where(torch.from_numpy(count > 0).unsqueeze(1), p_state, torch.zeros_like(p_state))
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
        c = m.cell
        K = c.heads
        # the relation rows: distinct edge attributes only, then the self-loop row last
        ea = np.ascontiguousarray(edge_attr.numpy())
        first, inv_np = unique_rows(ea.view(np.int32))
        inv = torch.from_numpy(inv_np)
        r_u = m.relations(edge_attr.index_select(0, torch.from_numpy(first)), self.bank_proj)
        r_rows = torch.cat([r_u, m.relations.self_loop.unsqueeze(0)])
        R1 = c.rel1(r_rows).contiguous().numpy()
        R2 = c.rel2(r_rows).contiguous().numpy()
        MR = c.msg_r(r_rows).contiguous().numpy()
        loops = np.arange(N, dtype=np.int64)
        u = np.concatenate([edge_index[0].numpy(), loops])
        v = np.concatenate([edge_index[1].numpy(), loops])
        rid = np.concatenate([inv_np, np.full(N, r_u.shape[0], np.int64)])
        order, ptr = csr_by_dst(v, N)
        QH = c.query(q_state).contiguous().numpy()
        att = c.att.detach().contiguous().numpy()
        for _ in range(m.steps):
            S = c.src(h).contiguous().numpy()
            D = c.dst(h).contiguous().numpy()
            MH = c.msg_h(h).contiguous().numpy()
            mm = torch.from_numpy(attend(S, D, QH, R1, R2, MH, MR, att, u, v, rid, order, ptr, K))
            g = torch.sigmoid(c.gate(torch.cat([h, mm, q_state], dim=-1)))
            h = (1.0 - g) * h + g * c.norm(h + mm)
        out = h
        if m.evidence is not None:
            ev = m.evidence
            p = ev.initial(UM.evidence_features(batch.x, nq, B, m.evidence_index))
            if edge_attr.shape[0]:
                typed = (edge_attr[:, UM.STRUCT_FAMILY] > 0.5) & (edge_attr[:, UM.ATTR["rel_mask"]] > 0.5)
                if bool(typed.any()):
                    r = r_u.index_select(0, inv)
                    t_index, t_r = edge_index[:, typed], r[typed]
                else:
                    t_index, t_r = edge_index[:, typed], None
            else:
                t_index, t_r = edge_index, None
            if t_index.shape[1]:
                tu, tv = t_index[0], t_index[1]
                score2 = ((ev.query(q_state).index_select(0, tv) * ev.rel(t_r)) @ ev.att).unsqueeze(1)
                beta = GF.segment_softmax(score2, tv, N)
                ones = torch.ones(tv.numel(), dtype=p.dtype)
                has_in = (torch.zeros(N, dtype=p.dtype).index_add_(0, tv, ones) > 0).unsqueeze(1)
                for _ in range(m.steps):
                    m2 = torch.zeros(N, ev.d_p, dtype=p.dtype).index_add_(0, tv, beta * ev.prop(p).index_select(0, tu))
                    g2 = torch.sigmoid(ev.gate(torch.cat([p, m2, q_state], dim=-1)))
                    p = torch.where(has_in, (1.0 - g2) * p + g2 * m2, p)
            out = torch.cat([h, p], dim=-1)
        return m.readout(out, base_z)


# ── the MLP's and zsp's inputs: exact (integer and identical-operand) savings ──


@numba.njit(cache=True)
def bucket_unique(n, a, k2):
    """np.unique(a * M + k2) split back into (a, k2), for k2 < M: rows bucketed by a, each bucket sorted, repeats
    collapsed; the same sorted keys."""
    cnt = np.zeros(n + 1, np.int64)
    for j in range(a.size):
        cnt[a[j] + 1] += 1
    for t in range(n):
        cnt[t + 1] += cnt[t]
    pos = cnt[:-1].copy()
    kb = np.empty(a.size, np.int64)
    for j in range(a.size):
        kb[pos[a[j]]] = k2[j]
        pos[a[j]] += 1
    ao = np.empty(a.size, np.int64)
    ko = np.empty(a.size, np.int64)
    m = 0
    for t in range(n):
        seg = np.sort(kb[cnt[t]:cnt[t + 1]])
        for i in range(seg.size):
            if i == 0 or seg[i] != seg[i - 1]:
                ao[m] = t
                ko[m] = seg[i]
                m += 1
    return ao[:m].copy(), ko[:m].copy()


def pairs_fast(n, u, v):
    """lean_mlp2.pairs, by bucket_unique: the same (u, v) arrays."""
    u = u.astype(np.int64)
    v = v.astype(np.int64)
    ok = u != v
    return bucket_unique(n, np.concatenate([u[ok], v[ok]]), np.concatenate([v[ok], u[ok]]))


def links_fast(n, eu, ev, ef):
    """c1_cold.links_of (zlink.chunk_edges for one question), by bucket_unique: the same arrays and dtypes."""
    u, v, f = eu.astype(np.int64), ev.astype(np.int64), ef.astype(np.int64)
    if n >= ZL.NMAX:
        raise SystemExit(f"a pool of {n} rows, above int16")
    loop = u == v
    a, b, ff = np.r_[u[~loop], v[~loop]], np.r_[v[~loop], u[~loop]], np.r_[f[~loop], f[~loop]]
    ao, ko = bucket_unique(n, a, b * ZL.FAMS + ff)
    return int(ao.size), ao.astype(np.int16), (ko // ZL.FAMS).astype(np.int16), (ko % ZL.FAMS).astype(np.int8)


def seed_dists_fast(n, codes, store, seeds, buckets, x_rrf, eu, ev, efam):
    """lean_cache.store_seed_dists: the seed products P @ P[S].T once (dist_fast computed the same product of the same
    operands again), the pairs by pairs_fast, the distance kernel unchanged. The same arrays."""
    P = store.decode(codes)
    valid = seeds >= 0
    S, Bk = seeds[valid].astype(np.int64), buckets[valid]
    seed = np.zeros((n, 3), np.float32)
    G = None
    if S.size:
        G = P @ P[S].T
        seed[:, 0] = G.mean(1)
        seed[:, 1] = G.max(1)
        if bool((Bk == 0).any()):
            seed[:, 2] = G[:, Bk == 0].max(1)
    eu64, ev64 = eu.astype(np.int64), ev.astype(np.int64)
    st = efam == 0
    rrf = x_rrf.astype(np.float64)
    s_seed = rrf[S] / max(float(rrf.max()), 1e-12) if S.size else np.zeros(0)
    u, v = pairs_fast(n, eu64[st], ev64[st])
    out = np.zeros((n, 14), np.float32)
    G64 = G.astype(np.float64) if S.size else np.zeros((n, 0))
    L2._dist_nb(n, L2.csr_of(n, u), v, S, s_seed.astype(np.float64), G64, np.ascontiguousarray(G64[S]), out)
    return seed, out


# ── the MLP's and zsp's forward at batch 1 ───────────────────────────────────


def zcols(x, ref, eps=1e-6):
    """lean_screen3.seg_zscore_ref for one segment, every column at once (the whole pool's z-score where fewer than
    two reference rows or an sd under eps)."""
    n = x.shape[0]
    mean = x.sum(0) / max(n, 1)
    c = x - mean
    var = (c * c).sum(0) / max(n, 1)
    sd = torch.where(var == 0, torch.zeros_like(var), var.sqrt())
    full = torch.where(sd < eps, torch.zeros_like(c), c / sd.clamp_min(eps))
    r = ref.to(x.dtype).unsqueeze(1)
    cnt = float(ref.sum())
    den = max(cnt, 1.0)
    mr = (x * r).sum(0) / den
    cr = x - mr
    vr = (cr * cr * r).sum(0) / den
    sdr = torch.where(vr == 0, torch.zeros_like(vr), vr.sqrt())
    ok = (sdr >= eps) & (cnt >= 2)
    return torch.where(ok, cr / sdr.clamp_min(eps), full)


@numba.njit(cache=True)
def prop_numba(zc, u, v, f, N, F):
    deg = np.zeros((N, F), np.float32)
    sm = np.zeros((N, F), np.float32)
    ex = np.zeros((N, F), np.float32)
    for j in range(u.size):
        x = zc[u[j]]
        deg[v[j], f[j]] += np.float32(1.0)
        sm[v[j], f[j]] += x
        ex[v[j], f[j]] += np.float32(np.exp(x))
    return deg, sm, ex


class FastZ:
    """zrc's (zrm.ZRM: ZRet's forward; no chains on an untyped graph) or zsp's (zprop.ZProp) forward for one question,
    from its float16 rows, store codes and links. The per-block [raw, z, mask] concatenation is folded into the first
    layer (the mask columns are 1 at serving, so they become a bias), and every block's z-scores come in one pass. The
    same function on the same weights; checked to TOL and the top 5."""

    def __init__(self, model, blocks, span, store):
        m = model
        if list(m.blocks) != list(blocks):
            raise SystemExit("the model's blocks are not the carve's")
        if getattr(m, "ctx_mode", "none") != "none":
            raise SystemExit("FastZ serves ctx none")
        self.m, self.zsp = m, isinstance(m, ZP.ZProp)
        W1 = m.l1.weight.detach()
        fixed, raw_at, z_at, m_at = [], [], [], []
        off = 0
        semb_raw = semb_z = None
        for b in m.blocks:
            if b == "SEMB":
                w = int(m.U.shape[1])
                semb_raw, semb_z = list(range(off, off + w)), list(range(off + w, off + 2 * w))
            else:
                a, e = span[b]
                w = e - a
                fixed += list(range(a, e))
                raw_at += list(range(off, off + w))
                z_at += list(range(off + w, off + 2 * w))
            m_at.append(off + 2 * w)
            off += 2 * w + 1
        if off != W1.shape[1]:
            raise SystemExit(f"the first layer reads {W1.shape[1]} inputs, the blocks give {off}")
        if semb_raw is not None:
            raw_at += semb_raw
            z_at += semb_z
        self.fixed = torch.tensor(fixed, dtype=torch.long)
        self.Wr = W1[:, raw_at].T.contiguous()
        self.Wz = W1[:, z_at].T.contiguous()
        self.b1 = (m.l1.bias.detach() + W1[:, m_at].sum(1)).contiguous()
        self.semb = semb_raw is not None
        ra, _re = span["rank"]
        self.i_rrf = fixed.index(ra + S3.I_RRF)
        self.s = torch.from_numpy(store.s)
        self.c = torch.from_numpy(np.ascontiguousarray(store.c, np.float32))
        self.kappa = float(store.kappa)

    @torch.inference_mode()
    def __call__(self, r):
        m = self.m
        X = torch.from_numpy(r["X"]).to(torch.float32)
        R = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).index_select(1, self.fixed)
        if self.semb:
            qe = torch.from_numpy(r["q_emb16"]).to(torch.float32)
            P = torch.from_numpy(r["codes"]).to(torch.float32) * self.s + self.c
            P = torch.cat([P, torch.full((P.shape[0], 1), self.kappa)], 1)
            R = torch.cat([R, (qe @ m.U) * (P @ m.V)], 1)
        ref = R[:, self.i_rrf] > 0
        Z = zcols(R, ref)
        h = Fn.gelu(torch.addmm(self.b1, R, self.Wr).addmm_(Z, self.Wz))
        h = Fn.gelu(m.l2(h))
        s = m.base_w * Z[:, self.i_rrf] + m.out(h).squeeze(-1)
        if not self.zsp:
            return s
        _ne, a, b, f = r["links"]
        n = s.shape[0]
        zs = zcols(s.unsqueeze(1), torch.ones(n, dtype=torch.bool)).squeeze(1)
        zc = zs.clamp(-ZP.ZCLIP, ZP.ZCLIP).numpy()
        deg, sm, ex = prop_numba(zc, a.astype(np.int64), b.astype(np.int64), f.astype(np.int64), n, ZP.FAMS)
        deg, sm, ex = torch.from_numpy(deg), torch.from_numpy(sm), torch.from_numpy(ex)
        has = deg > 0
        zero = torch.zeros_like(deg)
        x = torch.cat([torch.where(has, sm / deg.clamp_min(1.0), zero),
                       torch.where(has, torch.log(ex.clamp_min(1e-30)), zero), torch.log1p(deg), zs.unsqueeze(1)], 1)
        return s + m.prop_head(x)


# ── selftest ─────────────────────────────────────────────────────────────────


def compare(a, b):
    a, b = a.numpy(), b.numpy()
    return float(np.abs(a - b).max()) if a.size else 0.0, bool(np.array_equal(np.sort(np.argsort(-a, kind="stable")[:5]),
                                                                              np.sort(np.argsort(-b, kind="stable")[:5])))


def selftest():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    worst = 0.0
    for seed in range(3):
        m = GF._model(seed, 7)
        m.eval()
        ref, fus = GF.FastGNN(m), FusedGNN(m)
        for sizes in ([40], [17, 33, 5], [120]):
            for typed in (False, True):
                b = GF._batch(rng, sizes, typed, 3, 7)
                x, y = ref(b), fus(b)
                d, top = compare(x, y)
                worst = max(worst, d)
                assert d < TOL, (seed, sizes, typed, d)
    # seg_mean_rows equals index_add_'s sums bit for bit
    P = rng.standard_normal((30, 8)).astype(np.float32)
    u = rng.integers(0, 30, 200)
    v = rng.integers(0, 30, 200)
    o, p = csr_by_dst(v, 30)
    a, cnt = seg_mean_rows(P, u, v, o, p)
    b, c2 = GF.segment_mean_rows(torch.from_numpy(P)[torch.from_numpy(u)], torch.from_numpy(v), 30)
    assert np.array_equal(a, b.numpy()) and np.array_equal(cnt, c2.numpy())
    print(f"c3_fast selftest: FusedGNN within {worst:.2e} of FastGNN (tolerance {TOL:g}); segment means bit for bit: ok")
    return 0



# ── the MLP's batched form (warm, batch 16) ──────────────────────────────────


@torch.inference_mode()
def fastz_batch(fz, rs):
    """FastZ's folded forward over several questions' rows at once: per-question z-scores by lean_screen3's
    seg_zscore_ref, zsp's step by zprop.prop_inputs over the questions' links (offset per question)."""
    m = fz.m
    n_np = np.asarray([r["n"] for r in rs], np.int64)
    B = len(rs)
    nq = torch.from_numpy(np.repeat(np.arange(B), n_np))
    X = torch.from_numpy(np.concatenate([r["X"] for r in rs])).to(torch.float32)
    R = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).index_select(1, fz.fixed)
    if fz.semb:
        qe = torch.from_numpy(np.stack([r["q_emb16"] for r in rs])).to(torch.float32)
        P = torch.from_numpy(np.concatenate([r["codes"] for r in rs])).to(torch.float32) * fz.s + fz.c
        P = torch.cat([P, torch.full((P.shape[0], 1), fz.kappa)], 1)
        R = torch.cat([R, (qe @ m.U).index_select(0, nq) * (P @ m.V)], 1)
    ref = (R[:, fz.i_rrf] > 0).to(R.dtype)
    Z = S3.seg_zscore_ref(R, nq, B, ref)
    h = Fn.gelu(torch.addmm(fz.b1, R, fz.Wr).addmm_(Z, fz.Wz))
    h = Fn.gelu(m.l2(h))
    s = m.base_w * Z[:, fz.i_rrf] + m.out(h).squeeze(-1)
    if not fz.zsp:
        return s, n_np
    off = np.r_[0, np.cumsum(n_np)[:-1]]
    lk = {"u": torch.from_numpy(np.concatenate([r["links"][1].astype(np.int64) + o for r, o in zip(rs, off)])),
          "v": torch.from_numpy(np.concatenate([r["links"][2].astype(np.int64) + o for r, o in zip(rs, off)])),
          "f": torch.from_numpy(np.concatenate([r["links"][3].astype(np.int64) for r in rs]))}
    return s + m.prop_head(ZP.prop_inputs(s, lk, nq, B)), n_np


# ── the timed harness (docs/C3_FASTEST_FORMS.md) ─────────────────────────────

ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "c3"
PATHS = ("zrc3", "zsp3", "gnn3", "zrc2", "gnn6")
FAST = ("zrc3", "zsp3", "gnn3")


def _c2():
    import c2_fast as C2
    return C2


def lean_rows3(su, sh, T):
    """c2_fast.lean_rows_mlp with SEED and DISTS from seed_dists_fast (the same arrays)."""
    C2 = _c2()
    C1 = C2.C1
    c0 = clock()
    comp = sh["comp"]
    n = int(comp.pool.size)
    eu, ev, ef, efw, ebw = C1.edges_of(comp)
    c1 = clock()
    sl = np.full(C2.ML8.MAX_SEEDS, -1, np.int64)
    sb = np.full(C2.ML8.MAX_SEEDS, -1, np.int64)
    s_loc = np.asarray(comp.seeds_local, np.int64)
    sl[:s_loc.size], sb[:s_loc.size] = s_loc, sh["bucket"]
    x16 = np.asarray(comp.scalars[:, su.columns], dtype=np.float16)
    walk = C2.walk_block(n, sl, sb, eu, ev, ef, efw, ebw)
    walkf = C2.walkf_block(n, sl, sb, eu, ev, ef, efw, ebw)
    codes = su.codes[sh["pool"]]
    seed, dists = seed_dists_fast(n, codes, su.store, sl, sb, x16[:, su.c_rrf_x], eu, ev, ef)
    X = np.empty((n, su.W), np.float16)
    X[:, :su.span[C2.LC.XC_BLOCKS[-1]][1]] = x16[:, su.xcols]
    for b, arr in (("WALK", walk), ("WALKF", walkf), ("SEED", seed), ("DISTS", dists)):
        a, e = su.span[b]
        X[:, a:e] = arr.astype(np.float16)
    c2 = clock()
    T["edges"], T["lean"] = c1 - c0, c2 - c1
    return {"n": n, "X": X, "codes": codes, "q_emb16": np.asarray(sh["qemb"], np.float32).astype(np.float16),
            "e": (eu, ev, ef, efw, ebw), "sl": sl, "sb": sb}


def clock():
    import time
    return time.perf_counter()


class Fast:
    """The fast forms on a C1 Setup: FastZ for zrc and zsp, FusedGNN for the six GNN."""

    def __init__(self, su):
        self.z = {k: FastZ(su.models[k][0].eval(), su.models[k][1], su.span, su.store) for k in ("zrc", "zsp")}
        self.g = FusedGNN(su.gnn)


def run_path3(su, fa, k, i, T, keep=None):
    """One path for question i, cold, every stage of its own timed into T; returns (scores, shared, rows or batch).
    The fast paths also time a warm repeat of their forward on the inputs they built (T['warm_forward'])."""
    C2 = _c2()
    C1 = C2.C1
    if k in ("zrc2", "gnn6"):
        return C2.run_path(su, k, i, T)
    if k == "gnn3":
        sh = C1.shared(su, i, T)
        c0 = clock()
        b = C1.LT.fast_pack(sh["comp"], sh["E"], sh["qemb"], su.columns, True)
        c1 = clock()
        with torch.inference_mode():
            sc = fa.g(b)
        torch.topk(sc, min(5, int(sh["pool"].size)))
        c2 = clock()
        with torch.inference_mode():
            fa.g(b)
        c3 = clock()
        T["pack"], T["forward"], T["warm_forward"] = c1 - c0, c2 - c1, c3 - c2
        return sc, sh, b
    sh = C2.shared_mlp(su, i, T)
    r = lean_rows3(su, sh, T)
    c0 = clock()
    if k == "zsp3":
        eu, ev, ef = r["e"][:3]
        r["links"] = links_fast(r["n"], eu, ev, ef)
    c1 = clock()
    fz = fa.z["zsp" if k == "zsp3" else "zrc"]
    sc = fz(r)
    torch.topk(sc, min(5, r["n"]))
    c2 = clock()
    fz(r)
    c3 = clock()
    if k == "zsp3":
        T["links"] = c1 - c0
    T["forward"], T["warm_forward"] = c2 - c1, c3 - c2
    return sc, sh, r


def check_c3(su, fa, i, out):
    """Untimed. The fast forms against their references on the same question: pools and seeds against the look's; zrc3's
    rows and store codes equal zrc2's bit for bit and zsp3's links equal c1_cold.links_of's; zrc3 and zsp3 (FastZ) within
    TOL of zrc's and zsp's forward on zrc2's rows with the same top 5; gnn3 (FusedGNN) within TOL of gnn6 (FastGNN) with
    the same top 5; gnn6 against the look's stored scores as C1 checks it."""
    C2 = _c2()
    C1 = C2.C1
    ref = su.ref[i]
    sc = {k: v[0] for k, v in out.items()}
    sh = {k: v[1] for k, v in out.items()}
    r = {k: v[2] for k, v in out.items()}
    c = {"pool": all(bool(np.array_equal(sh[k]["pool"], ref["pool"])) for k in PATHS)}
    c["seeds"] = all(bool(np.array_equal(r[k]["sl"], ref["sl"]) and np.array_equal(r[k]["sb"], ref["sb"]))
                     for k in ("zrc3", "zsp3", "zrc2"))
    c["rows_bits"] = all(C2.same(r[k]["X"], r["zrc2"]["X"]) and np.array_equal(r[k]["codes"], r["zrc2"]["codes"])
                         for k in ("zrc3", "zsp3"))
    eu, ev, ef = r["zrc2"]["e"][:3]
    L1 = C1.links_of(r["zrc2"]["n"], eu, ev, ef)
    c["links_bits"] = bool(L1[0] == r["zsp3"]["links"][0] and all(
        a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(L1[1:], r["zsp3"]["links"][1:])))
    ra = dict(r["zrc2"])
    ra["links"] = L1
    with torch.inference_mode():
        zsp_ref = C1.forward(su, "zsp", C1.Rows([ra], su.span, su.W, su.c_rrf, su.store), 1, links=True)
    for k, want in (("zrc3", sc["zrc2"]), ("zsp3", zsp_ref), ("gnn3", sc["gnn6"])):
        d, top = compare(want, sc[k])
        c[f"{k}_diff"], c[f"{k}_top5_same"] = d, top
        c[f"{k}_within_tol"] = d <= TOL
    g = sc["gnn6"].numpy()
    c["gnn6_vs_look"] = float(np.abs(g - ref["gnn0"].astype(np.float32)).max()) if g.size else 0.0
    c["gnn6_top5_same"] = bool(np.array_equal(C1.top5(g), C1.top5(ref["gnn0"].astype(np.float32))))
    return c


def batch_group3(su, fa, rows_z, rows_s, batches, singles):
    """Warm batch 16, one forward per fast path over a group's questions (inputs built at batch 1): FastZ's batched
    form for zrc and zsp, FusedGNN over the group's packed graphs (C1's pack, untimed here as in C1)."""
    C2 = _c2()
    C1 = C2.C1
    out = {"size": len(rows_z)}
    for k, rs in (("zrc3", rows_z), ("zsp3", rows_s)):
        c0 = clock()
        s, n_np = fastz_batch(fa.z["zsp" if k == "zsp3" else "zrc"], rs)
        C1.topk_per_q(s, n_np)
        out[k] = {"forward": clock() - c0}
        d, _t = compare(torch.cat(singles[k]), s)
        out[k]["vs_b1"] = d
    qds = [{"pool": sh["comp"].pool, "x": sh["comp"].scalars[:, su.columns], "seedw": sh["comp"].seedw,
            "qemb": sh["qemb"], "seeds": sh["comp"].seeds_local, "gold": np.zeros(0, np.int64), "gold_total": 0,
            "emb": sh["E"]} for sh in batches]
    b = C1.V2.pack_queries_v2(qds, su.context)
    c0 = clock()
    with torch.inference_mode():
        sc = fa.g(b)
    C1.topk_per_q(sc, np.asarray([int(sh["pool"].size) for sh in batches]))
    out["gnn3"] = {"forward": clock() - c0}
    d, _t = compare(torch.cat(singles["gnn3"]), sc)
    out["gnn3"]["vs_b1"] = d
    return out


def run(a):
    import gc
    import json
    import time
    C2 = _c2()
    C1 = C2.C1
    t_start = time.time()
    pinned = C1.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    torch.set_num_threads(1)
    C1.log(f"{a.dataset}: pin {pinned}")
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    if a.dataset in C1.TYPED:
        raise SystemExit(f"{a.dataset}: typed graphs follow the four untyped datasets (docs/C3_FASTEST_FORMS.md)")
    su = C1.Setup(a.dataset, a.queries)
    fa = Fast(su)
    nq = len(su.prep.pools) - 1
    t = time.time()
    for k in PATHS:                       # numba's compile: the carve's next question, once, index time
        run_path3(su, fa, k, nq, {})
    su.index["jit_question_s"] = time.time() - t
    C1.log(f"{a.dataset}: {nq} measured questions, index {su.index}")
    recs, checks, b16 = [], [], []
    gz, gs, gb, sg = [], [], [], {k: [] for k in FAST}
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            order = PATHS[i % len(PATHS):] + PATHS[:i % len(PATHS)]
            T, out = {"order": list(order)}, {}
            for k in order:
                T[k] = {}
                out[k] = run_path3(su, fa, k, i, T[k])
            T["n"] = int(out["gnn6"][1]["pool"].size)
            recs.append(T)
            checks.append(check_c3(su, fa, i, out))
            gz.append(out["zrc3"][2])
            gs.append(out["zsp3"][2])
            gb.append(out["gnn3"][1])
            for k in FAST:
                sg[k].append(out[k][0])
            if len(gz) == C1.B16 or i == nq - 1:
                b16.append(batch_group3(su, fa, gz, gs, gb, sg))
                gz, gs, gb, sg = [], [], [], {k: [] for k in FAST}
            del out
            if (i + 1) % 25 == 0:
                gc.collect()
                C1.log(f"  {i + 1}/{nq}")
    finally:
        gc.enable()
    rec = {"dataset": a.dataset, "carve": C1.CARVE, "queries": nq, "declared_in": "docs/C3_FASTEST_FORMS.md",
           "threads": 1, "tol": TOL, "pin": pinned, "index": su.index, "paths": list(PATHS), "per_question": recs,
           "batch16": b16, "checks": checks, "check_summary": summarize(checks),
           "fits": {k: {"dir": str(C1.FITS[k].relative_to(ROOT)).replace("\\", "/"),
                        "models_sha256": C1.sha_file(C1.FITS[k] / "models.pt"), "candidate": C1.CAND} for k in C1.FITS},
           "basis": su.basis, "basis_sha256": su.basis_sha256, "look_records": su.look_head["records"],
           "peak_rss_bytes": C1.LC.peak_rss(), "script_sha256": C1.sha_file(__file__),
           "c2_script_sha256": C1.sha_file(C2.__file__), "c1_script_sha256": C1.sha_file(C1.__file__),
           "numpy": np.__version__, "torch": torch.__version__, "numba": numba.__version__,
           "seconds": round(time.time() - t_start, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out or OUT / f"{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    C1.LC.write_json(out, rec)
    C1.log(f"{a.dataset}: done in {rec['seconds']}s; checks {rec['check_summary']}; -> {out}")
    return 0


GOOD_KEYS = ("pool", "seeds", "rows_bits", "links_bits", "zrc3_within_tol", "zrc3_top5_same", "zsp3_within_tol",
             "zsp3_top5_same", "gnn3_within_tol", "gnn3_top5_same", "gnn6_top5_same")


def summarize(checks):
    s = {"questions": len(checks)}
    for key in GOOD_KEYS:
        v = [c[key] for c in checks if key in c]
        if v:
            s[f"{key}_ok"] = int(sum(v))
    for key in ("zrc3_diff", "zsp3_diff", "gnn3_diff", "gnn6_vs_look"):
        v = [c[key] for c in checks if key in c]
        if v:
            s[f"{key}_max"] = max(v)
    return s


def cold(T, k):
    """A path's cold total (s): every stage but the warm repeat."""
    return sum(v for s_, v in T[k].items() if s_ != "warm_forward")


def report():
    import json
    C2 = _c2()
    C1 = C2.C1
    out = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name.startswith("report"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        ds = r["dataset"]
        P = r["per_question"]
        good = [j for j, c in enumerate(r["checks"]) if all(c.get(k, True) for k in GOOD_KEYS)]
        bad_share = 1 - len(good) / max(len(P), 1)
        d = {"questions": len(P), "kept": len(good), "failing_share": bad_share, "stopped": bad_share > 0.02}
        rng = np.random.default_rng(0)
        col = lambda k: np.asarray([cold(P[j], k) for j in good])  # noqa: E731
        for k in PATHS:
            x = col(k) * 1e3
            d[f"{k}_cold_ms"] = {p: float(np.percentile(x, p)) for p in (50, 95, 99)}
            d[f"{k}_cold_ms"]["p50_ci"] = C1.boot_ci(x, lambda v: np.percentile(v, 50), rng)
        for num, den in (("gnn3", "zrc3"), ("gnn3", "zsp3"), ("zsp3", "zrc3"), ("gnn6", "gnn3"), ("zrc2", "zrc3"),
                         ("gnn6", "zrc2")):
            d[f"cold_{num}_over_{den}"] = C2.ratio_ci(col(num), col(den), rng)
        wcol = lambda k: np.asarray([P[j][k]["warm_forward"] for j in good])  # noqa: E731
        for k in FAST:
            d[f"{k}_warm_b1_ms_p50"] = float(np.percentile(wcol(k), 50)) * 1e3
        d["warm_b1_gnn3_over_zrc3"] = C2.ratio_ci(wcol("gnn3"), wcol("zrc3"), rng)
        d["warm_b1_gnn3_over_zsp3"] = C2.ratio_ci(wcol("gnn3"), wcol("zsp3"), rng)
        b16 = {}
        for g in r["batch16"]:
            for k in FAST:
                b16.setdefault(k, []).append(g[k]["forward"] / g["size"])
        d["b16_vs_b1_max"] = {k: max(g[k]["vs_b1"] for g in r["batch16"]) for k in FAST}
        d["warm_b16_per_q_ms_p50"] = {k: float(np.percentile(v, 50)) * 1e3 for k, v in b16.items()}
        st = {}
        for k in PATHS:
            parts = {}
            for j in good:
                for s_, v in P[j][k].items():
                    parts.setdefault(s_, []).append(v * 1e3)
            st[k] = {s_: float(np.percentile(v, 50)) for s_, v in parts.items()}
        d["stage_p50_ms"] = st
        d["index"] = r["index"]
        d["checks"] = r["check_summary"]
        d["pin"] = r["pin"]
        d["peak_rss_gb"] = r["peak_rss_bytes"] / 1e9 if r.get("peak_rss_bytes") else None
        out[ds] = d
    C1.LC.write_json(OUT / "report.json", out)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
    lines = ["Cold, batch 1, total per question, p50 ms:", "",
             "| dataset | zrc3 (MLP) | zsp3 (GNN track) | gnn3 (six GNN) | gnn3 / zrc3 | gnn3 / zsp3 | zrc2 | gnn6 | "
             "zrc2 / zrc3 | gnn6 / gnn3 |", "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | --- |"]
    for ds, d in out.items():
        m = lambda k: f"{d[f'{k}_cold_ms'][50]:.1f}"  # noqa: E731
        lines.append(f"| {ds} | {m('zrc3')} | {m('zsp3')} | {m('gnn3')} | {f(d['cold_gnn3_over_zrc3'])} | "
                     f"{f(d['cold_gnn3_over_zsp3'])} | {m('zrc2')} | {m('gnn6')} | {f(d['cold_zrc2_over_zrc3'])} | "
                     f"{f(d['cold_gnn6_over_gnn3'])} |")
    lines += ["", "Warm forward per question, p50 ms (inputs built; batch 1 and batch 16):", "",
              "| dataset | zrc3 b1 | zsp3 b1 | gnn3 b1 | gnn3 / zrc3 b1 | zrc3 b16 | zsp3 b16 | gnn3 b16 | "
              "gnn3 / zrc3 b16 |", "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for ds, d in out.items():
        w = d["warm_b16_per_q_ms_p50"]
        lines.append(f"| {ds} | {d['zrc3_warm_b1_ms_p50']:.2f} | {d['zsp3_warm_b1_ms_p50']:.2f} | "
                     f"{d['gnn3_warm_b1_ms_p50']:.2f} | {f(d['warm_b1_gnn3_over_zrc3'])} | {w['zrc3']:.2f} | "
                     f"{w['zsp3']:.2f} | {w['gnn3']:.2f} | {w['gnn3'] / w['zrc3']:.1f}x |")
    lines += ["", "Checks (questions passing / measured):", ""]
    for ds, d in out.items():
        lines.append(f"- {ds}: kept {d['kept']}/{d['questions']}; {d['checks']}")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def check(a):
    """Development check (untimed numbers): every path on the first questions, the C3 checks."""
    C2 = _c2()
    C1 = C2.C1
    C1.pin(2)
    torch.set_num_threads(1)
    su = C1.Setup(a.dataset, a.queries)
    fa = Fast(su)
    nq = len(su.prep.pools) - 1
    bad = []
    rs = {k: [] for k in FAST}
    gz, gs, gb, sg = [], [], [], {k: [] for k in FAST}
    for i in [nq] + list(range(nq)):
        out = {}
        T = {}
        for k in PATHS:
            T[k] = {}
            out[k] = run_path3(su, fa, k, i, T[k])
        c = check_c3(su, fa, i, out)
        if not all(c.get(k, True) for k in GOOD_KEYS):
            bad.append((i, {k: c[k] for k in c if not c[k] or k.endswith("diff")}))
        if i != nq:
            for k in FAST:
                rs[k].append((cold(T, k), T[k]["warm_forward"]))
            if len(gz) < C1.B16:
                gz.append(out["zrc3"][2])
                gs.append(out["zsp3"][2])
                gb.append(out["gnn3"][1])
                for k in FAST:
                    sg[k].append(out[k][0])
    g = batch_group3(su, fa, gz, gs, gb, sg)
    C1.log(f"  batch {g['size']}: " + ", ".join(f"{k} vs batch 1 {g[k]['vs_b1']:.2e}, "
                                                  f"{g[k]['forward'] / g['size'] * 1e3:.2f} ms/q" for k in FAST))
    if max(g[k]["vs_b1"] for k in FAST) > TOL:
        bad.append(("batch16", {k: g[k]["vs_b1"] for k in FAST}))
    C1.log(f"c3 check {a.dataset}: {nq} questions, {len(bad)} failing {bad[:3]}")
    for k, v in rs.items():
        v = np.asarray(v)
        C1.log(f"  {k}: cold p50 {np.median(v[:, 0]) * 1e3:.2f} ms, warm forward p50 {np.median(v[:, 1]) * 1e3:.2f} ms")
    return 0 if not bad else 1


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--dataset")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run(a)
    if a.cmd == "check":
        return check(a)
    if a.cmd == "report":
        return report()
    raise SystemExit("c3_fast: check|run|report --dataset D, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
