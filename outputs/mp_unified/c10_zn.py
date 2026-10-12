"""C10 (docs/C10_ZN_COLD.md): zn, the lean MLP that reads no edges (docs/ZN_LEAN_NO_MP.md), against zsp, the lean
GNN, cold, batch 1, one thread, each at its fastest exact form.

Four paths per question, in rotated order:

  zn10    FastPool (m3b's frozen pool construction in one numba pass), FastRead (the pool's node rows through
          FastCompiler's exact float16 table), the retrieval columns (FastCompiler's own kernels), the seed products
          from the store codes decoded once (Decoder), the float16 rows, and ZFast.
  zsp10   FastPool; c2_fast's partial compile reading the node rows through FastCompiler's own float16 table, with C4's
          galloping typed edges; c3_fast's lean rows with the codes decoded once; C6's chain entries on the typed
          graphs; c3_fast's links; ZFast.
  zn4     zn by the C9-era forms (the frozen pool code, the node read, the store's decode, c3_fast.FastZ).
  zsp4 / zsp6   zsp's fastest filed path, unchanged: C4's on the untyped graphs, C6's on the typed graphs.

ZFast is c3_fast.FastZ with the first layer's two halves in one GEMM over [raw | z], the raw columns read through the
float16 table, every column's z-scores in one numba pass over the rows (lean_screen3.seg_zscore_ref's rule, sums in
float64), zsp's step (zprop.prop_inputs) in another, and C6's FastChain for the chain term on typed graphs.

Every reused buffer (FastCompiler's, FastRead's, the decoders', ZFast's) is sized and touched once at setup to the
largest measured pool, for both models alike. Checks run untimed on every question (see check() and GOOD).

    python outputs/mp_unified/c10_zn.py --selftest
    python outputs/mp_unified/c10_zn.py check --dataset 2wiki --queries 30
    python outputs/mp_unified/c10_zn.py run --dataset 2wiki
    python outputs/mp_unified/c10_zn.py report
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

import json  # noqa: E402
import time  # noqa: E402

import c4_fast as C4  # noqa: E402
import c6_chains as C6  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

C5, C3, C2, C1 = C6.C5, C6.C3, C6.C2, C6.C1
CF, S3 = C6.CF, C3.S3
LC, LM, LG, LT, FF, T2, ZP, L3 = C1.LC, C1.LM, C1.LG, C1.LT, C1.FF, C1.T2, C1.ZP, C1.L3
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "c10"
ZN_FIT = ROOT / "outputs" / "screen" / "fits" / "scr-zn"
NMP = ["rank", "dense_cos", "SEMB", "SEED"]
ZN_SPAN = {"rank": (0, 5), "dense_cos": (5, 6), "SEED": (6, 9)}
ZN_W = 9
ZN_C_RRF = ZN_SPAN["rank"][0] + LM.SPLIT["rank"].index("rrf")
RCOLS = ("dense_cos", "dense_rr", "dense_in", "splade_rr", "splade_in", "splade_score_norm", "rrf", "agreement",
         "is_seed", "seed_rank")
RANK_AT = [RCOLS.index(c) for c in LM.SPLIT["rank"]]
COS_AT = RCOLS.index("dense_cos")
MAX_SEEDS = C2.ML8.MAX_SEEDS
TOL = C3.TOL
EPS = 1e-6
GOOD = ("pool", "seeds", "read_bits", "P_bits", "zn_rows_bits", "zsp10_rows_bits", "zsp10_compile_bits",
        "zsp10_links_bits", "zn10_within_tol", "zn10_top5_same", "zn4_within_tol", "zn4_top5_same", "zsp10_within_tol",
        "zsp10_top5_same")
TYPED_GOOD = ("zsp10_entries_bits", "zsp10_entries_filed", "zsp10_e_rel")
clock = time.perf_counter


# ── the pool: m3b's frozen construction in one numba pass ────────────────────


@numba.njit(cache=True)
def _uniq_first(x):
    """m3b_pools._first_occurrence_unique for a short array (the seed lists hold at most ten ids)."""
    out = np.empty(x.size, np.int64)
    m = 0
    for i in range(x.size):
        v = x[i]
        new = True
        for j in range(m):
            if out[j] == v:
                new = False
                break
        if new:
            out[m] = v
            m += 1
    return out[:m].copy()


@numba.njit(cache=True)
def _rrf_base(d, s, budget, constant):
    """m3a.rrf_fused_rows' row (rank_fusion.rrf_rankings at dense weight 0.5 over the two top-200 lists, cut at
    |dense U splade|), then its first budget ids. err 1: unequal or empty widths; 2: an id repeated within a list."""
    w = min(200, d.size)
    if min(200, s.size) != w or w == 0:
        return np.empty(0, np.int64), 1
    ids = np.empty(2 * w, np.int64)
    for i in range(w):
        ids[i] = d[i]
        ids[w + i] = s[i]
    by = np.argsort(ids, kind="mergesort")             # stable: of a pair, the dense entry sorts first
    for j in range(2 * w - 1):
        if ids[by[j]] == ids[by[j + 1]] and (by[j] < w) == (by[j + 1] < w):
            return np.empty(0, np.int64), 2
    vid = np.empty(2 * w, np.int64)
    vsc = np.empty(2 * w, np.float64)
    m = 0
    j = 0
    while j < 2 * w:
        p = by[j]
        sd = 0.0
        ss = 0.0
        if p < w:
            sd = 1.0 / (np.float64(constant) + np.float64(p + 1))
        else:
            ss = 1.0 / (np.float64(constant) + np.float64(p - w + 1))
        if j + 1 < 2 * w and ids[by[j + 1]] == ids[p]:
            q = by[j + 1]
            sd = sd + 0.0
            ss = 0.0 + 1.0 / (np.float64(constant) + np.float64(q - w + 1))
            j += 2
        else:
            j += 1
        vid[m] = ids[p]
        vsc[m] = -(0.5 * sd + 0.5 * ss)
        m += 1
    order = np.argsort(vsc[:m], kind="mergesort")      # ties keep ascending id order
    k = min(budget, m)
    out = np.empty(k, np.int64)
    for t in range(k):
        out[t] = vid[order[t]]
    return out, 0


@numba.njit(cache=True)
def fast_pool(d, s, kind, budget, constant, nf, ips, cols, hops, cap1, cap2, limit, mvis, mpool, g):
    """lean_time.PoolBuilder's (pool, seeds) for one question, and each seed's bucket (0: the first id of either list).
    kind 0: equal_rrf_budget_<budget>; kind 1: dense_top<budget>. nf families (0: RETRIEVAL); m3b_pools.expand_hops
    with hops, per_seed_cap cap1, per_frontier_cap cap2, visited_cap limit. mvis and mpool are stamp arrays over the
    graph's nodes (stamp g marks this question). err 0 when served; 3: an id outside the graph."""
    N = mvis.size
    empty = np.empty(0, np.int64)
    nd = min(5, d.size)
    ns = min(5, s.size)
    merged = np.empty(nd + ns, np.int64)
    for i in range(nd):
        merged[i] = d[i]
    for i in range(ns):
        merged[nd + i] = s[i]
    seeds = _uniq_first(merged)
    for i in range(seeds.size):
        if seeds[i] < 0 or seeds[i] >= N:
            return empty, empty, empty, 3
    if kind == 0:
        base, err = _rrf_base(d, s, budget, constant)
        if err != 0:
            return empty, empty, empty, err
    else:
        base = d[:min(budget, d.size)].copy()
    cap = 1
    if nf > 0:
        cap = seeds.size * nf * cap1 + max(limit, 0) + 1
    exp = np.empty(cap, np.int64)
    ne = 0
    if nf > 0:
        for si in range(seeds.size):                    # hop 1: the first cap1 neighbours, node, family, rank order
            x = seeds[si]
            for f in range(nf):
                ip = ips[f]
                cl = cols[f]
                a = ip[x]
                b = ip[x + 1]
                if b - a > cap1:
                    b = a + cap1
                for t in range(a, b):
                    y = np.int64(cl[t])
                    if mvis[y] != g:
                        mvis[y] = g
                        exp[ne] = y
                        ne += 1
        if hops > 1:
            nvis = ne
            for si in range(seeds.size):
                x = seeds[si]
                if mvis[x] != g:
                    mvis[x] = g
                    nvis += 1
            start = 0
            end = ne
            for _h in range(2, hops + 1):
                bud = limit - nvis
                if bud <= 0 or end == start:
                    break
                fresh = 0
                for ii in range(start, end):
                    x = exp[ii]
                    for f in range(nf):
                        ip = ips[f]
                        cl = cols[f]
                        a = ip[x]
                        b = ip[x + 1]
                        if b - a > cap2:
                            b = a + cap2
                        for t in range(a, b):
                            y = np.int64(cl[t])
                            if mvis[y] != g:
                                mvis[y] = g
                                exp[ne] = y
                                ne += 1
                                fresh += 1
                                if fresh == bud:
                                    break
                        if fresh == bud:
                            break
                    if fresh == bud:
                        break
                if fresh == 0:
                    break
                nvis += fresh
                start = end
                end = ne
    buf = np.empty(base.size + ne + seeds.size, np.int64)
    m = 0
    for i in range(base.size):
        x = base[i]
        if x < 0 or x >= N:
            return empty, empty, empty, 3
        if mpool[x] != g:
            mpool[x] = g
            buf[m] = x
            m += 1
    for i in range(ne):
        x = exp[i]
        if mpool[x] != g:
            mpool[x] = g
            buf[m] = x
            m += 1
    for i in range(seeds.size):
        x = seeds[i]
        if mpool[x] != g:
            mpool[x] = g
            buf[m] = x
            m += 1
    pool = np.sort(buf[:m])
    bucket = np.ones(seeds.size, np.int64)
    for i in range(seeds.size):
        if (d.size > 0 and seeds[i] == d[0]) or (s.size > 0 and seeds[i] == s[0]):
            bucket[i] = 0
    return pool, seeds, bucket, 0


class FastPool:
    """lean_time.PoolBuilder (m3b's frozen construction) in one numba pass: the same pool and seeds (the selftest
    compares them on random graphs for every base, regime and hop setting the six datasets use), and the buckets."""

    WHY = ("", "unequal or empty list widths", "an id repeated within a list", "an id outside the graph")

    def __init__(self, builder, n_nodes):
        name = builder.construction["base_pool"]
        if name.startswith("equal_rrf_budget_"):
            self.kind, self.budget = 0, int(name[len("equal_rrf_budget_"):])
        elif name.startswith("dense_top"):
            self.kind, self.budget = 1, int(name[len("dense_top"):])
        else:
            raise SystemExit(f"FastPool serves the equal_rrf_budget_* and dense_top* bases, not {name}")
        self.constant = int(builder.constant)
        fams = list(builder.fams) if builder.fams is not None else []
        self.nf = len(fams)
        if fams:
            ips = [np.ascontiguousarray(f.indptr, dtype=np.int64) for f in fams]
            dt = np.int32 if all(f.col.dtype == np.int32 for f in fams) else np.int64
            cols = [np.ascontiguousarray(f.col, dtype=dt) for f in fams]
            st = builder.setting
            self.hops, self.cap1 = int(st["hops"]), int(st["per_seed_cap"])
            if self.hops > 1:
                self.cap2, self.limit = int(st["per_frontier_cap"]), int(st["visited_cap"])
            else:
                self.cap2 = self.limit = 0
        else:
            ips, cols = [np.zeros(1, np.int64)], [np.zeros(0, np.int32)]
            self.hops = self.cap1 = self.cap2 = self.limit = 0
        while len(ips) < 3:
            ips.append(ips[0])
            cols.append(cols[0])
        self.ips, self.cols = tuple(ips), tuple(cols)
        self.mvis = np.zeros(int(n_nodes), np.int32)
        self.mpool = np.zeros(int(n_nodes), np.int32)
        self.g = 0

    def __call__(self, d_ids, s_ids):
        self.g += 1
        if self.g >= 2 ** 31 - 1:
            self.mvis[:] = 0
            self.mpool[:] = 0
            self.g = 1
        pool, seeds, bucket, err = fast_pool(
            np.ascontiguousarray(d_ids, dtype=np.int64), np.ascontiguousarray(s_ids, dtype=np.int64), self.kind,
            self.budget, self.constant, self.nf, self.ips, self.cols, self.hops, self.cap1, self.cap2, self.limit,
            self.mvis, self.mpool, self.g)
        if err:
            raise ValueError(f"FastPool: {self.WHY[err]}")
        return pool, seeds, bucket


# ── the node rows and the store codes ────────────────────────────────────────


@numba.njit(cache=True)
def glut_rows(src, rows, lut, out):
    d = out.shape[1]
    for i in range(rows.shape[0]):
        s = src[rows[i]]
        o = out[i]
        for j in range(d):
            o[j] = lut[s[j]]


@numba.njit(cache=True)
def glut_shards(shards, rows, shard_size, lut, out):
    d = out.shape[1]
    for i in range(rows.shape[0]):
        r = rows[i]
        s = shards[r // shard_size][r % shard_size]
        o = out[i]
        for j in range(d):
            o[j] = lut[s[j]]


class FastRead:
    """The pool's node rows as nodes.read gives them (float16 to float32), through FastCompiler's exact table (built by
    numpy's own cast of every float16 pattern): the same bits, checked on every question."""

    def __init__(self, fc, nodes):
        self.fc, self.nodes = fc, nodes
        self.mode = "ram" if fc._h16 is not None else ("shards" if fc._shards is not None else "read")
        self.buf = np.empty((0, FF.DIM), np.float32)

    def reserve(self, n):
        if self.buf.shape[0] < n:
            self.buf = np.empty((n, FF.DIM), np.float32)
            self.buf.fill(0.0)

    def __call__(self, pool):
        if self.mode == "read":
            return np.asarray(self.nodes.read(pool), np.float32)
        n = int(pool.size)
        self.reserve(n)
        E = self.buf[:n]
        if self.mode == "ram":
            glut_rows(self.fc._h16, pool, self.fc._lut16, E)
        else:
            glut_shards(self.fc._shards, pool, self.fc._shard_size, self.fc._lut16, E)
        return E


@numba.njit(cache=True)
def decode_rows(codes, rows, s, c, kappa, P):
    """lean_mlp3.Store.decode(codes[rows]): codes * s + c in float32 (no FMA), then kappa."""
    k = s.size
    for i in range(rows.size):
        src = codes[rows[i]]
        o = P[i]
        for j in range(k):
            o[j] = np.float32(src[j]) * s[j] + c[j]
        o[k] = kappa


class Decoder:
    """The pool's store codes decoded once, straight from the index's code table: Store.decode's bits."""

    def __init__(self, su):
        st = su.store
        self.codes = su.codes
        self.s = np.ascontiguousarray(st.s, np.float32)
        self.c = np.ascontiguousarray(st.c, np.float32)
        self.kappa = np.float32(st.kappa)
        self.buf = np.empty((0, L3.STORE_DIM), np.float32)

    def reserve(self, n):
        if self.buf.shape[0] < n:
            self.buf = np.empty((n, L3.STORE_DIM), np.float32)
            self.buf.fill(0.0)

    def __call__(self, pool):
        n = int(pool.size)
        self.reserve(n)
        P = self.buf[:n]
        decode_rows(self.codes, pool, self.s, self.c, self.kappa, P)
        return P


def seed_of(n, P, sl, sb):
    """lean_cache.store_seed_dists' SEED block (c3_fast.seed_dists_fast's lines on the same P)."""
    valid = sl >= 0
    S, Bk = sl[valid].astype(np.int64), sb[valid]
    seed = np.zeros((n, 3), np.float32)
    if S.size:
        G = P @ P[S].T
        seed[:, 0] = G.mean(1)
        seed[:, 1] = G.max(1)
        if bool((Bk == 0).any()):
            seed[:, 2] = G[:, Bk == 0].max(1)
    return seed


def seed_dists10(n, P, seeds, buckets, x_rrf, eu, ev, efam):
    """c3_fast.seed_dists_fast with P given (decoded once): the same lines after its decode."""
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
    u, v = C3.pairs_fast(n, eu64[st], ev64[st])
    out = np.zeros((n, 14), np.float32)
    G64 = G.astype(np.float64) if S.size else np.zeros((n, 0))
    C3.L2._dist_nb(n, C3.L2.csr_of(n, u), v, S, s_seed.astype(np.float64), G64, np.ascontiguousarray(G64[S]), out)
    return seed, out


# ── the forward ──────────────────────────────────────────────────────────────


@numba.njit(cache=True)
def rz_fill(X16, fixed, lut, PV, qU, i_rrf, eps, RZ, zr):
    """RZ[:, :C] = the raw columns (X's fixed columns through the float16 table, non-finite to 0, then SEMB =
    qU * PV); RZ[:, C:] = their z-scores by lean_screen3.seg_zscore_ref's rule (against the rows with rrf > 0 when
    at least two and an sd of at least eps, else the whole pool's, 0 under eps), sums in float64; zr = rrf's z."""
    n = X16.shape[0]
    F = fixed.size
    D = qU.size
    C = F + D
    for i in range(n):
        row = X16[i]
        o = RZ[i]
        for k in range(F):
            v = lut[row[fixed[k]]]
            if not np.isfinite(v):
                v = np.float32(0.0)
            o[k] = v
        for k in range(D):
            o[F + k] = qU[k] * PV[i, k]
    s1 = np.zeros(C)
    r1 = np.zeros(C)
    cnt = 0
    for i in range(n):
        o = RZ[i]
        ref = o[i_rrf] > 0
        if ref:
            cnt += 1
        for k in range(C):
            x = np.float64(o[k])
            s1[k] += x
            if ref:
                r1[k] += x
    den = np.float64(max(n, 1))
    denr = np.float64(max(cnt, 1))
    mean = s1 / den
    meanr = r1 / denr
    s2 = np.zeros(C)
    r2 = np.zeros(C)
    for i in range(n):
        o = RZ[i]
        ref = o[i_rrf] > 0
        for k in range(C):
            x = np.float64(o[k])
            c = x - mean[k]
            s2[k] += c * c
            if ref:
                cr = x - meanr[k]
                r2[k] += cr * cr
    sd = np.sqrt(s2 / den)
    sdr = np.sqrt(r2 / denr)
    for i in range(n):
        o = RZ[i]
        for k in range(C):
            x = np.float64(o[k])
            if cnt >= 2 and sdr[k] >= eps:
                z = (x - meanr[k]) / sdr[k]
            elif sd[k] < eps:
                z = 0.0
            else:
                z = (x - mean[k]) / sd[k]
            o[C + k] = np.float32(z)
        zr[i] = o[C + i_rrf]


@numba.njit(cache=True)
def prop_x(s, a, b, f, F, zclip, eps, x):
    """zprop.prop_inputs for one question: the score's pool z-score (float64 sums; 0 under eps), clamped at zclip;
    per family the neighbours' mean, soft maximum (log of the sum of exp) and log1p of the degree, in edge order with
    float32 sums as c3_fast.prop_numba; then the z-score."""
    n = s.size
    t = 0.0
    for i in range(n):
        t += np.float64(s[i])
    mean = t / max(n, 1)
    v = 0.0
    for i in range(n):
        c = np.float64(s[i]) - mean
        v += c * c
    sd = np.sqrt(v / max(n, 1))
    zc = np.empty(n, np.float32)
    for i in range(n):
        z = np.float32(0.0)
        if sd >= eps:
            z = np.float32((np.float64(s[i]) - mean) / sd)
        x[i, 3 * F] = z
        if z > zclip:
            z = np.float32(zclip)
        elif z < -zclip:
            z = np.float32(-zclip)
        zc[i] = z
    deg = np.zeros((n, F), np.float32)
    sm = np.zeros((n, F), np.float32)
    ex = np.zeros((n, F), np.float32)
    for j in range(a.size):
        u = np.int64(a[j])
        t_ = np.int64(b[j])
        ff = np.int64(f[j])
        xz = zc[u]
        deg[t_, ff] += np.float32(1.0)
        sm[t_, ff] += xz
        ex[t_, ff] += np.float32(np.exp(xz))
    for i in range(n):
        for k in range(F):
            dg = deg[i, k]
            if dg > 0:
                x[i, k] = sm[i, k] / dg
                e = ex[i, k]
                if e < np.float32(1e-30):
                    e = np.float32(1e-30)
                x[i, F + k] = np.log(e)
            else:
                x[i, k] = np.float32(0.0)
                x[i, F + k] = np.float32(0.0)
            x[i, 2 * F + k] = np.log1p(dg)


class ZFast:
    """c3_fast.FastZ's folded forward (zn: lean_screen3.ZRet; zsp: zprop.ZProp, chain term by chains_fast.FastChain on
    typed graphs) with [raw | z] in one GEMM, the raw columns and z-scores by rz_fill, zsp's step by prop_x. The same
    function on the same weights; checked against the module within TOL with the same top 5."""

    def __init__(self, model, blocks, span, store, lut, rel_unit=None, rel_centered=None):
        fz = C3.FastZ(model, blocks, span, store)
        self.m, self.zsp, self.semb = model, fz.zsp, fz.semb
        self.fixed = fz.fixed.numpy().astype(np.int64)
        self.i_rrf = int(fz.i_rrf)
        self.C = int(self.fixed.size) + (int(model.U.shape[1]) if self.semb else 0)
        self.W1 = torch.cat([fz.Wr, fz.Wz], 0).contiguous()
        self.b1 = fz.b1
        self.lut = lut
        self.fch = CF.FastChain(model, rel_unit, rel_centered) if rel_unit is not None and hasattr(model, "cm_b") else None
        self.q0 = np.zeros(0, np.float32)
        self.RZ = np.empty((0, 2 * self.C), np.float32)
        self.zr = np.empty(0, np.float32)
        self.xp = np.empty((0, ZP.N_IN), np.float32)

    def reserve(self, n):
        if self.RZ.shape[0] < n:
            self.RZ = np.empty((n, 2 * self.C), np.float32)
            self.zr = np.empty(n, np.float32)
            self.xp = np.empty((n, ZP.N_IN), np.float32)
            for a in (self.RZ, self.zr, self.xp):
                a.fill(0.0)

    @torch.inference_mode()
    def __call__(self, r):
        m = self.m
        n = int(r["n"])
        self.reserve(n)
        RZ, zr = self.RZ[:n], self.zr[:n]
        qe = torch.from_numpy(r["q_emb16"]).to(torch.float32)
        if self.semb:
            qU = (qe @ m.U).numpy()
            PV = (torch.from_numpy(r["P"]) @ m.V).numpy()
        else:
            qU, PV = self.q0, np.zeros((n, 0), np.float32)
        rz_fill(r["X"].view(np.uint16), self.fixed, self.lut, PV, qU, self.i_rrf, EPS, RZ, zr)
        h = Fn.gelu(torch.addmm(self.b1, torch.from_numpy(RZ), self.W1))
        h = Fn.gelu(m.l2(h))
        s = m.base_w * torch.from_numpy(zr) + m.out(h).squeeze(-1)
        if self.fch is not None and r.get("chains") is not None:
            fm, fr = self.fch(qe, r["chains"], n)
            s = s + (m.cm_gate[0] * fm + m.cm_gate[1] * fr)
        if not self.zsp:
            return s
        _ne, a, b, f = r["links"]
        xp = self.xp[:n]
        prop_x(s.numpy(), a, b, f, ZP.FAMS, float(ZP.ZCLIP), EPS, xp)
        return s + m.prop_head(torch.from_numpy(xp))


# ── the paths ────────────────────────────────────────────────────────────────


def qinputs(su, i):
    p = su.prep
    return T2.QueryInputs(p.qemb[i], p.dense_ids[i], p.dense_scores[i], p.splade_ids[i], p.splade_scores[i])


def seed_rows(s_loc, bucket):
    sl = np.full(MAX_SEEDS, -1, np.int64)
    sb = np.full(MAX_SEEDS, -1, np.int64)
    sl[:s_loc.size], sb[:s_loc.size] = s_loc, bucket
    return sl, sb


def zn_retrieval(fc, K, inp, pool, seeds, E):
    """The retrieval columns of c2_fast.compile_mlp (FastCompiler's lines and kernels), into a 10-column table."""
    n = int(pool.size)
    lookup = fc._lookup
    lookup[pool] = np.arange(n, dtype=np.int32)
    try:
        seeds_local = lookup[seeds].astype(np.int64)
        q = np.asarray(inp.q, dtype=np.float32)
        dense_cos = (np.asarray(E, dtype=np.float32) @ q).astype(np.float32)
        d_rank = np.zeros(n, np.float32)
        d_score = np.zeros(n, np.float32)
        d_in = np.zeros(n, np.bool_)
        s_rank = np.zeros(n, np.float32)
        s_score = np.zeros(n, np.float32)
        s_in = np.zeros(n, np.bool_)
        K.list_ranks(np.asarray(inp.dense_ids, dtype=np.int64), np.asarray(inp.dense_scores).astype(np.float32), lookup,
                     fc.n_nodes, d_rank, d_score, d_in)
        K.list_ranks(np.asarray(inp.splade_ids, dtype=np.int64), np.asarray(inp.splade_scores).astype(np.float32),
                     lookup, fc.n_nodes, s_rank, s_score, s_in)
        top_splade = float(inp.splade_scores[0]) if inp.splade_scores.size else 0.0
        rrf = np.empty(n, np.float32)
        X = np.zeros((n, len(RCOLS)), np.float32)
        K.fill_retrieval(X, dense_cos, d_in, d_rank, s_in, s_rank, s_score, top_splade > 0,
                         np.float32(max(top_splade, 1e-12)), seeds_local, np.arange(len(RCOLS), dtype=np.int64), rrf)
    finally:
        lookup[pool] = -1
    return X, seeds_local


def zn_rows(n, Xr, seed):
    """zn's float16 rows: rank, dense_cos and SEED (the cache's columns of those blocks)."""
    X = np.empty((n, ZN_W), np.float16)
    X[:, 0:5] = Xr[:, RANK_AT]
    X[:, 5] = Xr[:, COS_AT]
    X[:, 6:9] = seed
    return X


def run_zn10(bn, i, T):
    su = bn.su
    p = su.prep
    d_ids, s_ids = np.asarray(p.dense_ids[i]), np.asarray(p.splade_ids[i])
    c0 = clock()
    pool, seeds, bucket = bn.fpool(d_ids, s_ids)
    c1 = clock()
    E = bn.read_zn(pool)
    c2 = clock()
    Xr, s_loc = zn_retrieval(su.fc, bn.K0, qinputs(su, i), pool, seeds, E)
    c3 = clock()
    n = int(pool.size)
    sl, sb = seed_rows(s_loc, bucket)
    P = bn.dec_zn(pool)
    X = zn_rows(n, Xr, seed_of(n, P, sl, sb))
    c4 = clock()
    r = {"n": n, "X": X, "P": P, "q_emb16": np.asarray(p.qemb[i], np.float32).astype(np.float16), "sl": sl, "sb": sb}
    sc = bn.zf_zn(r)
    torch.topk(sc, min(5, n))
    c5 = clock()
    bn.zf_zn(r)
    c6 = clock()
    T["pool"], T["read"], T["retrieval"], T["seed"] = c1 - c0, c2 - c1, c3 - c2, c4 - c3
    T["forward"], T["warm_forward"] = c5 - c4, c6 - c5
    return sc, {"pool": pool, "seeds": seeds, "bucket": bucket, "E": E}, r


def run_zn4(bn, i, T):
    """zn by the C9-era forms, as profiled before this stage: the frozen pool code, the node read, the store's decode
    of the pool's codes, c3_fast.FastZ."""
    su = bn.su
    p = su.prep
    d_ids, s_ids = np.asarray(p.dense_ids[i]), np.asarray(p.splade_ids[i])
    c0 = clock()
    pool, seeds = su.builder(d_ids, s_ids)
    pool, seeds = np.asarray(pool, np.int64), np.asarray(seeds, np.int64)
    c1 = clock()
    E = su.context.nodes.read(pool)
    c2 = clock()
    fc = su.fc
    K = fc._K_small if pool.size < FF.SERIAL_BELOW else fc.K
    Xr, s_loc = zn_retrieval(fc, K, qinputs(su, i), pool, seeds, E)
    c3 = clock()
    n = int(pool.size)
    top = {int(d_ids[0]), int(s_ids[0])}
    sl, sb = seed_rows(s_loc, np.asarray([0 if int(s) in top else 1 for s in seeds], dtype=np.int64))
    codes = su.codes[pool]
    X = zn_rows(n, Xr, seed_of(n, su.store.decode(codes), sl, sb))
    c4 = clock()
    r = {"n": n, "X": X, "codes": codes, "q_emb16": np.asarray(p.qemb[i], np.float32).astype(np.float16), "sl": sl,
         "sb": sb}
    sc = bn.fz_zn4(r)
    torch.topk(sc, min(5, n))
    c5 = clock()
    bn.fz_zn4(r)
    c6 = clock()
    T["pool"], T["read"], T["retrieval"], T["seed"] = c1 - c0, c2 - c1, c3 - c2, c4 - c3
    T["forward"], T["warm_forward"] = c5 - c4, c6 - c5
    return sc, {"pool": pool, "seeds": seeds, "E": E}, r


def lean_rows10(su, sh, T, dec):
    """c3_fast.lean_rows3 with the codes decoded once (Decoder) and seed_dists10 on that P."""
    c0 = clock()
    comp = sh["comp"]
    n = int(comp.pool.size)
    eu, ev, ef, efw, ebw = C1.edges_of(comp)
    c1 = clock()
    sl, sb = seed_rows(np.asarray(comp.seeds_local, np.int64), sh["bucket"])
    x16 = np.asarray(comp.scalars[:, su.columns], dtype=np.float16)
    walk = C2.walk_block(n, sl, sb, eu, ev, ef, efw, ebw)
    walkf = C2.walkf_block(n, sl, sb, eu, ev, ef, efw, ebw)
    P = dec(sh["pool"])
    seed, dists = seed_dists10(n, P, sl, sb, x16[:, su.c_rrf_x], eu, ev, ef)
    X = np.empty((n, su.W), np.float16)
    X[:, :su.span[LC.XC_BLOCKS[-1]][1]] = x16[:, su.xcols]
    for b, arr in (("WALK", walk), ("WALKF", walkf), ("SEED", seed), ("DISTS", dists)):
        a, e = su.span[b]
        X[:, a:e] = arr.astype(np.float16)
    c2 = clock()
    T["edges"], T["lean"] = c1 - c0, c2 - c1
    return {"n": n, "X": X, "P": P, "q_emb16": np.asarray(sh["qemb"], np.float32).astype(np.float16),
            "e": (eu, ev, ef, efw, ebw), "sl": sl, "sb": sb}


def run_zsp10(bn, i, T):
    su, ks = bn.su, bn.ks
    p = su.prep
    d_ids, s_ids = np.asarray(p.dense_ids[i]), np.asarray(p.splade_ids[i])
    fc = su.fc
    c0 = clock()
    pool, seeds, bucket = bn.fpool(d_ids, s_ids)
    c1 = clock()
    inp = qinputs(su, i)
    saved = (fc.K, fc._K_small, fc.parallel)
    fc.K, fc._K_small = bn.k_ser                       # the serial twins at every size, C4's galloping typed edges
    fc.parallel = False
    try:
        comp = C2.compile_mlp(fc, inp, pool, seeds, embeddings=None)
    finally:
        fc.K, fc._K_small, fc.parallel = saved
    c2 = clock()
    T["pool"], T["compile"] = c1 - c0, c2 - c1
    typed = ks.take() if bn.typed else None
    sh = {"pool": pool, "seeds": seeds, "bucket": bucket, "comp": comp, "qemb": np.asarray(p.qemb[i])}
    r = lean_rows10(su, sh, T, bn.dec_zsp)
    if bn.typed:
        c0 = clock()
        e_rel = C5.e_rel_of(r, typed)
        r["chains"] = C5.one_question_chains(su, C6.chains_of6(su, "zsp", r, e_rel, sh["qemb"]))
        T["chains"] = clock() - c0
        r["e_rel"] = e_rel
    c0 = clock()
    eu, ev, ef = r["e"][:3]
    r["links"] = C3.links_fast(r["n"], eu, ev, ef)
    c1 = clock()
    sc = bn.zf_zsp(r)
    torch.topk(sc, min(5, r["n"]))
    c2 = clock()
    bn.zf_zsp(r)
    c3 = clock()
    T["links"], T["forward"], T["warm_forward"] = c1 - c0, c2 - c1, c3 - c2
    return sc, sh, r


# ── the bench ────────────────────────────────────────────────────────────────


def load_zn(su):
    blob = torch.load(ZN_FIT / "models.pt", weights_only=False)
    if blob["basis"] != su.basis:
        raise SystemExit(f"zn's fit is on basis {blob['basis']}, the setup's on {su.basis}")
    ms = {nm: (m, bl) for nm, m, bl in LG.load_models(blob, "cpu", cls=S3.ZRet)}
    m, bl = ms[C1.CAND]
    if list(bl) != NMP:
        raise SystemExit(f"zn's fit reads {bl}, not {NMP}")
    return m.eval(), {"dir": str(ZN_FIT.relative_to(ROOT)).replace("\\", "/"),
                      "models_sha256": C1.sha_file(ZN_FIT / "models.pt"), "candidate": C1.CAND, "blocks": NMP}


class Bench:
    """C4's (untyped graphs) or C5/C6's (typed graphs) setup, with zn's fit and the fast forms of this stage."""

    def __init__(self, a):
        self.typed = a.dataset in C1.TYPED
        if self.typed:
            su, _fa5, ks, nq, self.filed, self.frec, self.part0 = C5.prepare(a)
            self.fa = C6.Fast6(su)
            self.ref_k = "zsp6"
        else:
            su = C1.Setup(a.dataset, a.queries)
            self.fa = C3.Fast(su)
            ks = C4.Kernels(su.fc)
            nq = len(su.prep.pools) - 1
            self.ref_k = "zsp4"
        self.su, self.ks, self.nq = su, ks, nq
        self.paths = ("zn10", "zsp10", "zn4", self.ref_k)
        self.good = GOOD + (f"{self.ref_k}_within_tol", f"{self.ref_k}_top5_same") + (TYPED_GOOD if self.typed else ())
        t = time.time()
        self.zn, self.zn_rec = load_zn(su)
        lut = su.fc._lut16
        self.fz_zn4 = C3.FastZ(self.zn, NMP, ZN_SPAN, su.store)
        self.zf_zn = ZFast(self.zn, NMP, ZN_SPAN, su.store, lut)
        mz, bz = su.models["zsp"]
        if self.typed:
            self.zf_zsp = ZFast(mz, bz, su.span, su.store, lut, su.rel_unit, su.rel_centered)
        else:
            self.zf_zsp = ZFast(mz, bz, su.span, su.store, lut)
        self.fpool = FastPool(su.builder, su.fc.n_nodes)
        self.read_zn = FastRead(su.fc, su.context.nodes)
        self.dec_zn, self.dec_zsp = Decoder(su), Decoder(su)
        self.K0 = FF._kernels(False)
        # zsp10's compile: the serial table at every pool size (one thread: a prange kernel only adds its scheduling),
        # with C4's galloping typed edges (on typed graphs Kernels5's stash over them, which the chains read)
        self.k_ser = (su.fc._K_small,) * 2 if self.typed else (ks.gallop[1],) * 2
        # every reused buffer sized and touched once, to the largest measured pool, for both models alike
        self.nmax = max(int(np.asarray(x).size) for x in su.prep.pools)
        su.fc._buffers(self.nmax)
        for b in (su.fc._E, su.fc._P, su.fc._P1, su.fc._P2):
            b.fill(0.0)
        for x in (self.read_zn, self.dec_zn, self.dec_zsp, self.zf_zn, self.zf_zsp):
            x.reserve(self.nmax)
        su.index["c10_setup_s"] = time.time() - t
        su.index["presized_rows"] = self.nmax
        su.index["read_mode"] = self.read_zn.mode

    def path(self, k, i, T):
        if k == "zn10":
            return run_zn10(self, i, T)
        if k == "zsp10":
            return run_zsp10(self, i, T)
        if k == "zn4":
            return run_zn4(self, i, T)
        if self.typed:
            return C6.run_path6(self.su, self.fa, self.ks, k, i, T)
        return C4.run_path4(self.su, self.fa, self.ks, k, i, T)

    @torch.inference_mode()
    def zn_module(self, X, codes, q16):
        rows = C1.Rows([{"n": X.shape[0], "X": X, "codes": codes, "q_emb16": q16}], ZN_SPAN, ZN_W, ZN_C_RRF,
                       self.su.store)
        feats, nq, base_z, _g = rows.batch(np.arange(1), NMP)
        return self.zn(feats, torch.ones((1, len(NMP)), dtype=torch.float32), nq, 1, base_z)

    @torch.inference_mode()
    def zsp_module(self, ra):
        su = self.su
        if self.typed:
            return C5.ref_forward(su, "zsp", ra, ra["chains"])
        return C1.forward(su, "zsp", C1.Rows([ra], su.span, su.W, su.c_rrf, su.store), 1, links=True)

    def check(self, i, out):
        """Untimed. Pools and seeds against the look's; zn10's node rows against nodes.read's and both paths' decoded
        codes against Store.decode's, bits; zn10's and zn4's rows against the reference path's rank, dense_cos and SEED
        columns, bits; zsp10's compile (scalars and edges), rows and links against the reference path's, bits (typed
        graphs: its chain entries against zsp6's and the filed build's, its e_rel against the look's); every path's
        scores against its module's forward on its own rows (within TOL, the same top 5)."""
        su, rk = self.su, self.ref_k
        ref = su.ref[i]
        sc = {k: v[0] for k, v in out.items()}
        sh = {k: v[1] for k, v in out.items()}
        r = {k: v[2] for k, v in out.items()}
        pool = sh[rk]["pool"]
        n = int(pool.size)
        c = {"pool": all(bool(np.array_equal(sh[k]["pool"], ref["pool"])) for k in self.paths)}
        c["seeds"] = all(bool(np.array_equal(r[k]["sl"], ref["sl"]) and np.array_equal(r[k]["sb"], ref["sb"]))
                         for k in self.paths)
        c["read_bits"] = C2.same(sh["zn10"]["E"], sh["zn4"]["E"])
        codes = su.codes[pool]
        P_ref = su.store.decode(codes)
        c["P_bits"] = C2.same(r["zn10"]["P"], P_ref) and C2.same(r["zsp10"]["P"], P_ref)
        Xr = r[rk]["X"]
        want = np.concatenate([Xr[:, a:e] for a, e in (su.span["rank"], su.span["dense_cos"], su.span["SEED"])], 1)
        c["zn_rows_bits"] = C2.same(r["zn10"]["X"], want) and C2.same(r["zn4"]["X"], want)
        c["zsp10_rows_bits"] = C2.same(r["zsp10"]["X"], Xr)
        a_, b_ = sh["zsp10"]["comp"], sh[rk]["comp"]
        ok = np.array_equal(a_.scalars, b_.scalars, equal_nan=True) and a_.edges.keys() == b_.edges.keys()
        ok = ok and all(x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True)
                        for f in a_.edges for x, y in zip(a_.edges[f], b_.edges[f]))
        c["zsp10_compile_bits"] = bool(ok)
        eu, ev, ef = r[rk]["e"][:3]
        L1 = C1.links_of(n, eu, ev, ef)
        Lz = r["zsp10"]["links"]
        c["zsp10_links_bits"] = bool(L1[0] == Lz[0] and all(x.dtype == y.dtype and np.array_equal(x, y)
                                                            for x, y in zip(L1[1:], Lz[1:])))
        if self.typed:
            a, b = r["zsp10"]["chains"], r[rk]["chains"]
            c["zsp10_entries_bits"] = bool(all(
                a[x].dtype == b[x].dtype and a[x].shape == b[x].shape and np.array_equal(C1.bits(a[x]), C1.bits(b[x]))
                for x in ("ent_row", "ent_z", "ent_b", "ent_m", "q_rel")))
            got = {"row": a["ent_row"], "z": a["ent_z"], "b": a["ent_b"], "m": a["ent_m"], "q_rel": a["q_rel"]}
            c["zsp10_entries_filed"] = bool(i < len(self.filed["zsp"]) and C5.same_entries(got, self.filed["zsp"][i]))
            c["zsp10_e_rel"] = bool(np.array_equal(r["zsp10"]["e_rel"], ref["e_rel"]))
        for k in ("zn10", "zn4"):
            want_s = self.zn_module(r[k]["X"], codes, r[k]["q_emb16"])
            d, top = C3.compare(want_s, sc[k])
            c[f"{k}_diff"], c[f"{k}_top5_same"], c[f"{k}_within_tol"] = d, top, d <= TOL
        for k in ("zsp10", rk):
            ra = dict(r[k])
            ra["codes"], ra["links"] = codes, L1
            d, top = C3.compare(self.zsp_module(ra), sc[k])
            c[f"{k}_diff"], c[f"{k}_top5_same"], c[f"{k}_within_tol"] = d, top, d <= TOL
        return c

    def summarize(self, checks):
        s = {"questions": len(checks), "kept": int(sum(all(c[k] for k in self.good) for c in checks))}
        for k in self.good:
            s[k] = int(sum(bool(c[k]) for c in checks))
        for k in sorted({k for c in checks for k in c if k.endswith("_diff")}):
            v = [c[k] for c in checks if k in c]
            s[f"{k}_max"] = float(max(v)) if v else 0.0
        return s


# ── run, check ───────────────────────────────────────────────────────────────


def cold(T, k):
    return sum(v for s_, v in T[k].items() if s_ != "warm_forward")


def run(a):
    import gc
    t_start = time.time()
    pinned = C1.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    torch.set_num_threads(THREADS)
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    bn = Bench(a)
    su, nq, paths = bn.su, bn.nq, bn.paths
    t = time.time()
    for k in paths:                                    # numba's compile and loads: the carve's next question, once
        bn.path(k, nq, {})
    su.index["jit_question_s"] = time.time() - t
    C1.log(f"{a.dataset}: {nq} measured questions, index {su.index}")
    recs, checks = [], []
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            order = paths[i % len(paths):] + paths[:i % len(paths)]
            T, out = {"order": list(order)}, {}
            for k in order:
                T[k] = {}
                out[k] = bn.path(k, i, T[k])
            T["n"] = int(out["zn10"][1]["pool"].size)
            recs.append(T)
            checks.append(bn.check(i, out))
            del out
            if (i + 1) % 25 == 0:
                gc.collect()
                C1.log(f"  {i + 1}/{nq}")
    finally:
        gc.enable()
    summ = bn.summarize(checks)
    rec = {"dataset": a.dataset, "carve": C1.CARVE, "queries": nq, "declared_in": "docs/C10_ZN_COLD.md",
           "threads": THREADS, "pin": pinned, "index": su.index, "paths": list(paths), "ref_path": bn.ref_k,
           "good": list(bn.good), "per_question": recs, "checks": checks, "check_summary": summ, "tol": TOL,
           "fits": {"zn": bn.zn_rec,
                    "zsp": {"dir": str(C1.FITS["zsp"].relative_to(ROOT)).replace("\\", "/"),
                            "models_sha256": C1.sha_file(C1.FITS["zsp"] / "models.pt"), "candidate": C1.CAND}},
           "basis": su.basis, "basis_sha256": su.basis_sha256, "look_records": su.look_head["records"],
           "peak_rss_bytes": LC.peak_rss(), "script_sha256": C1.sha_file(__file__),
           "c3_sha256": C1.sha_file(C3.__file__), "c4_sha256": C1.sha_file(C4.__file__),
           "c5_sha256": C1.sha_file(C5.__file__), "c6_sha256": C1.sha_file(C6.__file__),
           "chains_fast_sha256": C1.sha_file(CF.__file__), "numpy": np.__version__, "torch": torch.__version__,
           "numba": numba.__version__, "seconds": round(time.time() - t_start, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if bn.typed:
        rec["chain_builds"] = {k: {"root": str(C5.CH_ROOT[k].relative_to(ROOT)).replace("\\", "/"), **bn.frec[k]}
                               for k in bn.frec}
        rec["step1_part0_record_sha256"] = bn.part0.record_sha256
        rec["relations_sha256"] = su.rel_sha256
    out = Path(a.out or OUT / f"{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out, rec)
    C1.log(f"{a.dataset}: done in {rec['seconds']}s; checks {summ}; -> {out}")
    return 0


def check(a):
    """The development check: every path on every question, every check; prints the failing questions and each path's
    cold p50 (development numbers, not the stage's)."""
    C1.pin(2)
    torch.set_num_threads(THREADS)
    bn = Bench(a)
    bad, checks, tm = [], [], {k: [] for k in bn.paths}
    for i in [bn.nq] + list(range(bn.nq)):
        T = {k: {} for k in bn.paths}
        out = {k: bn.path(k, i, T[k]) for k in bn.paths}
        if i == bn.nq:
            continue
        c = bn.check(i, out)
        checks.append(c)
        for k in bn.paths:
            tm[k].append(cold(T, k))
        if not all(c[k] for k in bn.good):
            bad.append((i, {k: c[k] for k in bn.good if not c[k]},
                        {k: round(v, 7) for k, v in c.items() if k.endswith("_diff")}))
    s = bn.summarize(checks)
    C1.log(f"c10 check {a.dataset}: {bn.nq} questions, {len(bad)} failing {bad[:3]}")
    C1.log(f"  {s}")
    C1.log("  development cold p50 (fixed order, not the stage's timing): "
           + ", ".join(f"{k} {np.median(v) * 1e3:.2f} ms" for k, v in tm.items()))
    return 0 if not bad else 1


# ── the report ───────────────────────────────────────────────────────────────


def report():
    res = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name.startswith("report"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        P, paths, rk = r["per_question"], r["paths"], r["ref_path"]
        good = [j for j, c in enumerate(r["checks"]) if all(c[k] for k in r["good"])]
        d = {"questions": len(P), "kept": len(good), "stopped": 1 - len(good) / max(len(P), 1) > 0.02, "ref_path": rk}
        rng = np.random.default_rng(0)
        col = lambda k: np.asarray([cold(P[j], k) for j in good])  # noqa: E731
        wf = lambda k: np.asarray([P[j][k]["warm_forward"] for j in good])  # noqa: E731
        for k in paths:
            d[f"{k}_cold_ms"] = {str(p): float(np.percentile(col(k) * 1e3, p)) for p in (50, 95, 99)}
            d[f"{k}_stage_p50_ms"] = {s_: float(np.percentile([P[j][k][s_] for j in good], 50)) * 1e3
                                      for s_ in P[good[0]][k] if s_ != "warm_forward"}
            d[f"{k}_warm_forward_p50_ms"] = float(np.percentile(wf(k), 50)) * 1e3
        g = "zsp10" if d["zsp10_cold_ms"]["50"] <= d[f"{rk}_cold_ms"]["50"] else rk
        d["gnn_fastest"] = g
        d["headline"] = C2.ratio_ci(col(g), col("zn10"), rng)
        d["at_least_2x"] = d["headline"]["ratio"] >= 2.0
        d["ci_above_2x"] = d["headline"]["ci"][0] >= 2.0
        for num, den in (("zsp10", "zn10"), (rk, "zn10"), ("zn4", "zn10"), (rk, "zsp10"), (rk, "zn4")):
            d[f"cold_{num}_over_{den}"] = C2.ratio_ci(col(num), col(den), rng)
        d["warm_forward_zsp10_over_zn10"] = C2.ratio_ci(wf("zsp10"), wf("zn10"), rng)
        d["checks"] = r["check_summary"]
        res[r["dataset"]] = d
    LC.write_json(OUT / "report.json", res)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
    lines = ["Cold, batch 1, total per question, p50 ms; ratios with bootstrap 95% CIs. The headline is the faster GNN form "
             "(zsp10 or zsp's filed path) over zn10.", "",
             "| dataset | zn10 | zn4 | zsp10 | zsp filed (path) | **fastest GNN / zn10** | at least 2x | zsp10 / zn10 | "
             "zsp filed / zn10 | zn4 / zn10 | zsp filed / zsp10 | zsp filed / zn4 | warm forward zsp10 / zn10 |",
             "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ds, d in res.items():
        rk = d["ref_path"]
        m = lambda k: f"{d[f'{k}_cold_ms']['50']:.2f}"  # noqa: E731
        two = "yes (CI above 2)" if d["ci_above_2x"] else ("yes" if d["at_least_2x"] else "no")
        lines.append(f"| {ds} | {m('zn10')} | {m('zn4')} | {m('zsp10')} | {m(rk)} ({rk}) | **{f(d['headline'])}** "
                     f"({d['gnn_fastest']}) | {two} | {f(d['cold_zsp10_over_zn10'])} | {f(d[f'cold_{rk}_over_zn10'])} | "
                     f"{f(d['cold_zn4_over_zn10'])} | {f(d[f'cold_{rk}_over_zsp10'])} | {f(d[f'cold_{rk}_over_zn4'])} | "
                     f"{f(d['warm_forward_zsp10_over_zn10'])} |")
    lines += ["", "Cold p95 / p99 ms:", ""]
    for ds, d in res.items():
        lines.append(f"- {ds}: " + "; ".join(f"{k} {d[f'{k}_cold_ms']['95']:.2f} / {d[f'{k}_cold_ms']['99']:.2f}"
                                             for k in ("zn10", "zsp10", "zn4", d["ref_path"])))
    lines += ["", "Stage p50 ms (cold), and the warm forward p50 beside:", ""]
    for ds, d in res.items():
        for k in ("zn10", "zsp10", "zn4", d["ref_path"]):
            lines.append(f"- {ds} {k}: " + ", ".join(f"{s_} {v:.3f}" for s_, v in d[f"{k}_stage_p50_ms"].items())
                         + f"; warm forward {d[f'{k}_warm_forward_p50_ms']:.3f}")
    lines += ["", "Checks:", ""] + [f"- {ds}: kept {d['kept']}/{d['questions']}{' (STOPPED: more than 2% failing)' if d['stopped'] else ''}; "
                                    f"{d['checks']}" for ds, d in res.items()]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def _store(rng):
    V = np.linalg.qr(rng.standard_normal((FF.DIM, L3.STORE_K)))[0].astype(np.float32)
    return L3.Store({"m": (rng.standard_normal(FF.DIM) * 0.05).astype(np.float32), "V": V,
                     "w": (rng.random(L3.STORE_K) + 0.01).astype(np.float32)})


def _selftest_pool(rng):
    from types import SimpleNamespace
    from mp_retrieval import m3b_pools as MP
    m3b_compile = C1.V2.M3B_RUN.load_script("m3b_compile")
    m3b_contract = C1.V2.M3B_RUN.load_script("m3b_contract")
    m3a = m3b_compile.load_script("m3a_headroom")
    cfg_h = {"retrieval_pools": {"equal_rrf": {"constant": 60}},
             "graph_regimes": {"STRUCT": ["structural"], "FULL": ["structural", "ner", "knn"]}}
    settings = [{"hops": 1, "per_seed_cap": 25}, {"hops": 1, "per_seed_cap": 3},
                {"hops": 2, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 2000},
                {"hops": 3, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 2000},
                {"hops": 3, "per_seed_cap": 4, "per_frontier_cap": 3, "visited_cap": 60},
                {"hops": 3, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 5}]
    cases = 0
    for trial in range(3):
        N = int(rng.integers(300, 3000))
        stores = {}
        for fam in ("structural", "ner", "knn"):
            E = int(rng.integers(N // 2, 8 * N))
            g = SimpleNamespace(n_nodes=N, src=rng.integers(0, N, E).astype(np.int32),
                                dst=rng.integers(0, N, E).astype(np.int32),
                                weight=None if fam == "structural" else rng.random(E).astype(np.float32),
                                rel=rng.integers(0, 6, E).astype(np.int16) if fam == "structural" else None)
            stores[fam] = MP.FamilyStore.from_graph(g, fam)
        for base in ("equal_rrf_budget_50", "equal_rrf_budget_200", "dense_top200", "dense_top20"):
            for regime in ("RETRIEVAL", "STRUCT", "FULL"):
                for st in (settings if regime != "RETRIEVAL" else [None]):
                    cons = {"base_pool": base, "regime": regime, "setting": st}
                    ref = LT.PoolBuilder(cons, cfg_h, stores, m3a, m3b_contract, m3b_compile)
                    fp = FastPool(ref, N)
                    for L in (30, 200, 260):
                        span = min(N, int(L * rng.uniform(1.2, 3.0)))
                        d = rng.permutation(span)[:L].astype(np.int64)
                        s = rng.permutation(span)[:L].astype(np.int64)
                        want_pool, want_seeds = ref(d, s)
                        pool, seeds, bucket = fp(d, s)
                        top = {int(d[0]), int(s[0])}
                        want_b = np.asarray([0 if int(x) in top else 1 for x in want_seeds], np.int64)
                        assert C2.same(pool, np.asarray(want_pool, np.int64)), (trial, cons, L)
                        assert C2.same(seeds, np.asarray(want_seeds, np.int64)), (trial, cons, L)
                        assert np.array_equal(bucket, want_b), (trial, cons, L)
                        cases += 1
                    if base.startswith("equal_rrf"):        # unequal list widths: both refuse
                        d, s = rng.permutation(N)[:30].astype(np.int64), rng.permutation(N)[:200].astype(np.int64)
                        for fn in (ref, fp):
                            try:
                                fn(d, s)
                                raise AssertionError("unequal widths were served")
                            except ValueError:
                                pass
    return cases


def selftest():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    LG.bind_device_ops()                               # the store-width SEMB (V: STORE_DIM rows), as the fits use
    cases = _selftest_pool(rng)
    # the float16 table, rows and shards, against numpy's cast (inf, nan, subnormal, -0 included)
    lut = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    H = rng.integers(0, 65536, (500, 64)).astype(np.uint16)
    H[0, :5] = [0x7C00, 0xFC00, 0x7E00, 0x0001, 0x8000]
    rows = np.r_[0, rng.integers(0, 500, 76)].astype(np.int64)
    want = H[rows].view(np.float16).astype(np.float32)
    out = np.empty((rows.size, 64), np.float32)
    glut_rows(H, rows, lut, out)
    assert C2.same(out, want)
    shards = numba.typed.List()
    for k in range(0, 500, 128):
        shards.append(np.ascontiguousarray(H[k:k + 128]))
    out2 = np.empty_like(out)
    glut_shards(shards, rows, 128, lut, out2)
    assert C2.same(out2, want)
    # the decode, against Store.decode
    from types import SimpleNamespace
    store = _store(rng)
    codes = rng.integers(-127, 128, (400, L3.STORE_K)).astype(np.int8)
    dec = Decoder(SimpleNamespace(store=store, codes=codes))
    rr = rng.integers(0, 400, 90).astype(np.int64)
    assert C2.same(np.array(dec(rr)), store.decode(codes[rr]))
    # ZFast against c3_fast.FastZ: zn (ZRet) and a zsp-like ZProp, NaNs in the rows, random links
    worst = 0.0
    widths = {"rank": len(LM.SPLIT["rank"]), "dense_cos": 1, "SEMB": LM.SEMB_DIM, "SEED": 3}
    bsp = ["rank", "dense_cos", "WALK", "SEMB", "SEED"]
    span_sp = {"rank": (0, 5), "dense_cos": (5, 6), "WALK": (6, 10), "SEED": (10, 13)}
    for seed in range(3):
        mz = S3.ZRet(NMP, widths, 64, seed=seed).eval()
        ms = ZP.ZProp(bsp, {**widths, "WALK": 4}, 64, seed=seed).eval()
        with torch.no_grad():
            for m in (mz, ms):
                for p_ in m.out.parameters():
                    p_.normal_()
            ms.sp_w2.normal_()
            ms.sp_b2.normal_()
        for m, blocks, span, W in ((mz, NMP, ZN_SPAN, ZN_W), (ms, bsp, span_sp, 13)):
            fz = C3.FastZ(m, blocks, span, store)
            zf = ZFast(m, blocks, span, store, lut)
            for n in (1, 2, 5, 40, 300):
                X = rng.standard_normal((n, W)).astype(np.float32)
                X[:, ZN_C_RRF] = np.where(rng.random(n) < 0.5, rng.random(n) * 0.03, 0.0)
                X[rng.random((n, W)) < 0.02] = np.nan
                cz = rng.integers(-127, 128, (n, L3.STORE_K)).astype(np.int8)
                ne = int(rng.integers(0, 4 * n + 1))
                r = {"n": n, "X": X.astype(np.float16), "codes": cz, "P": store.decode(cz),
                     "q_emb16": rng.standard_normal(FF.DIM).astype(np.float16),
                     "links": C3.links_fast(n, rng.integers(0, n, ne), rng.integers(0, n, ne), rng.integers(0, 3, ne))}
                d, top = C3.compare(fz(r), zf(r))
                worst = max(worst, d)
                assert d < TOL and top, (seed, type(m).__name__, n, d, top)
    print(f"c10 selftest: FastPool equals PoolBuilder on {cases} random cases (and refuses unequal widths as it does); "
          f"the float16 table and the decode bit for bit; ZFast within {worst:.2e} of FastZ (tolerance {TOL:g}), the "
          f"same top 5: ok")
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run", "check", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "report":
        return report()
    if a.cmd == "run":
        return run(a)
    if a.cmd == "check":
        return check(a)
    raise SystemExit("c10_zn: run|check --dataset D, report, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
