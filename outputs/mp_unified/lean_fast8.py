"""Design look (untracked; not a result and not filed): lean_fast7's fused features with NBR inside one kernel.

lean_time7 (time7-2w, pinned) left NBR as the largest fused stage: NBR_DLIST_round took 0.80 ms of lean2s+f7's 2.66 ms
resident features, almost all of it lean_mlp3.nbr_of. That function gathers P[us] (one 257-float row per pool pair,
about 1 MB a query), sorts, and makes about twenty numpy calls. _nbr_core computes the same groups, sums and norms in
one pass over the cached P, in numpy's own summation orders:

  order   np.argsort(v, kind="stable") is a stable counting sort by destination;
  acc     np.add.reduceat(P[us], starts, axis=0): each group's first row, plus the float32 pairwise sum (numpy's
          FLOAT_pairwise_sum: -0.0 start below 8 terms, eight accumulators up to 128, halves split at n2 = n//2 - n2%8
          above) of the group's other rows, column by column;
  mean    acc / d, d = the group's size as float32 (np.bincount(v)[rows]);
  norm    np.linalg.norm(mean, axis=1) = sqrt(add.reduce(mean * mean, axis=1)): +0.0 plus the pairwise sum of the row;
  cos     np.add.reduceat(cos[us], starts) likewise (first plus the pairwise sum of the rest) / d, and
          np.maximum.reduceat(cos[us], starts) with numpy's scalar maximum (a >= b or isnan(a) ? a : b).

The one product, mean @ Q, stays numpy's sgemv on a (k, 257) C-contiguous numpy-allocated operand, because OpenBLAS's
sgemv order is not replicated (reference_exact_fast_compile_rules). Everything else is lean_fast7.FusedLean, unchanged.

    python outputs/mp_unified/lean_fast8.py --selftest
"""
import os
import sys

os.environ.setdefault("LEAN_TIME_THREADS", "1")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ["LEAN_TIME_THREADS"])
sys.dont_write_bytecode = True

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lean_fast7 as F7  # noqa: E402

L2, L3, LFU = F7.L2, F7.L3, F7.LFU
_F0, _F1 = F7._F0, F7._F1
NBR_EPS = np.float32(1e-12)                     # np.maximum(mn, 1e-12): a weak Python float, so float32 (NEP 50)
STACK = 64                                       # pairwise split depth: 2 * log2(n / 128) + 2 entries suffice far past 1e9


# ── kernels ──────────────────────────────────────────────────────────────────


@njit(nogil=True)
def _pw_rows_leaf(P, us, lo, n, out, acc8):
    """out[:] = numpy's float32 pairwise sum (n <= 128 terms) of the rows P[us[lo:lo + n]], column by column."""
    D = P.shape[1]
    if n < 8:
        for j in range(D):
            out[j] = np.float32(-0.0)
        for t in range(n):
            r = us[lo + t]
            for j in range(D):
                out[j] += P[r, j]
        return
    for k in range(8):
        r = us[lo + k]
        for j in range(D):
            acc8[k, j] = P[r, j]
    i = 8
    lim = n - (n % 8)
    while i < lim:
        for k in range(8):
            r = us[lo + i + k]
            for j in range(D):
                acc8[k, j] += P[r, j]
        i += 8
    for j in range(D):
        out[j] = ((acc8[0, j] + acc8[1, j]) + (acc8[2, j] + acc8[3, j])) + ((acc8[4, j] + acc8[5, j]) + (acc8[6, j] + acc8[7, j]))
    while i < n:
        r = us[lo + i]
        for j in range(D):
            out[j] += P[r, j]
        i += 1


@njit(nogil=True)
def _pw_rows(P, us, lo, n, out, acc8, vals, s_lo, s_n, s_join):
    """out[:] = numpy's float32 pairwise sum of the rows P[us[lo:lo + n]] for any n (lean_fast7._pairwise's explicit
    stack, vector-valued)."""
    if n <= 128:
        _pw_rows_leaf(P, us, lo, n, out, acc8)
        return
    D = P.shape[1]
    nv = 0
    s_lo[0] = lo
    s_n[0] = n
    s_join[0] = False
    top = 1
    while top > 0:
        top -= 1
        l = s_lo[top]
        m = s_n[top]
        if s_join[top]:
            for j in range(D):
                vals[nv - 2, j] = vals[nv - 2, j] + vals[nv - 1, j]       # left + right
            nv -= 1
        elif m <= 128:
            _pw_rows_leaf(P, us, l, m, vals[nv], acc8)
            nv += 1
        else:
            m2 = m // 2
            m2 -= m2 % 8
            s_lo[top] = l
            s_n[top] = m
            s_join[top] = True
            s_lo[top + 1] = l + m2
            s_n[top + 1] = m - m2
            s_join[top + 1] = False
            s_lo[top + 2] = l
            s_n[top + 2] = m2
            s_join[top + 2] = False
            top += 3
    for j in range(D):
        out[j] = vals[0, j]


@njit(nogil=True)
def _pw_gather(x, us, lo, n, buf):
    """numpy's float32 pairwise sum of x[us[lo:lo + n]] (cos[us] gathered into buf, then lean_fast7._pairwise)."""
    for t in range(n):
        buf[t] = x[us[lo + t]]
    return F7._pairwise(buf, 0, n)


@njit(nogil=True, cache=True)
def _nbr_core(n, P, cos, u, v, mean):
    """lean_mlp3.nbr_of's groups: the destinations with an edge (rows, ascending), each group's mean row into
    mean[:k] (a numpy-allocated (n, D) buffer), the mean cosine, the max cosine and the mean row's norm. Returns
    (k, rows, c0, c2, mn)."""
    m = u.shape[0]
    D = P.shape[1]
    start = np.zeros(n + 1, np.int64)
    for e in range(m):
        start[v[e] + 1] += 1
    for x in range(n):
        start[x + 1] += start[x]
    fill = start[:n].copy()
    us = np.empty(m, np.int64)
    for e in range(m):                                   # stable: np.argsort(v, kind="stable")
        x = v[e]
        us[fill[x]] = u[e]
        fill[x] += 1
    rows = np.empty(n, np.int64)
    c0 = np.empty(n, np.float32)
    c2 = np.empty(n, np.float32)
    mn = np.empty(n, np.float32)
    tmp = np.empty(D, np.float32)
    sq = np.empty(D, np.float32)
    acc8 = np.empty((8, D), np.float32)
    vals = np.empty((STACK, D), np.float32)
    s_lo = np.empty(3 * STACK, np.int64)
    s_n = np.empty(3 * STACK, np.int64)
    s_join = np.empty(3 * STACK, np.bool_)
    cbuf = np.empty(max(m, 1), np.float32)
    k = 0
    for x in range(n):
        s = start[x]
        e = start[x + 1]
        if e == s:
            continue
        c = e - s
        d = np.float32(c)
        r0 = us[s]
        rows[k] = x
        if c > 1:
            _pw_rows(P, us, s + 1, c - 1, tmp, acc8, vals, s_lo, s_n, s_join)
            for j in range(D):
                mean[k, j] = (P[r0, j] + tmp[j]) / d
        else:
            for j in range(D):
                mean[k, j] = P[r0, j] / d
        for j in range(D):
            sq[j] = mean[k, j] * mean[k, j]
        mn[k] = np.sqrt(_F0 + F7._pairwise(sq, 0, D))
        a = cos[r0]
        if c > 1:
            a = a + _pw_gather(cos, us, s + 1, c - 1, cbuf)
        c0[k] = a / d
        mx = cos[r0]
        for t in range(s + 1, e):
            y = cos[us[t]]
            if not (mx >= y or np.isnan(mx)):
                mx = y
        c2[k] = mx
        k += 1
    return k, rows[:k], c0[:k], c2[:k], mn[:k]


@njit(nogil=True, cache=True)
def _nbr_write(rows, c0, gq, mn, c2, out):
    """nbr_of's five columns into out (n, 5), already zero: mean cosine, gq / max(mn, 1e-12), max cosine, norm, 1."""
    for i in range(rows.shape[0]):
        x = rows[i]
        den = mn[i] if (mn[i] >= NBR_EPS or np.isnan(mn[i])) else NBR_EPS
        out[x, 0] = c0[i]
        out[x, 1] = gq[i] / den
        out[x, 2] = c2[i]
        out[x, 3] = mn[i]
        out[x, 4] = _F1


def nbr_fused(n, P, q, cos, u, v, out):
    """L3.nbr_of(n, P, q, cos, u, v) into out (an (n, 5) zero float32 view), bit for bit."""
    if u.size == 0:
        return
    mean = np.empty((n, P.shape[1]), np.float32)
    k, rows, c0, c2, mn = _nbr_core(n, P, cos, u, v, mean)
    gq = mean[:k] @ q
    _nbr_write(rows, c0, gq, mn, c2, out)


# ── the fused path ───────────────────────────────────────────────────────────


class FusedLean8(F7.FusedLean):
    """lean_fast7.FusedLean with NBR from nbr_fused; every other stage is lean_fast7's."""

    def __init__(self, fc, codes, pstore, sv):
        super().__init__(fc, codes, pstore, sv)
        self.name = sv.name + "+f8"

    def __call__(self, inp, pool, seeds, buckets, qemb16, T=None):
        fc, need, fu = self.fc, self.need, self.fu
        clock = time.perf_counter
        t0 = clock()
        n = int(pool.size)
        lookup = fc._lookup
        lookup[pool] = np.arange(n, dtype=np.int32)
        try:
            sl = lookup[seeds].astype(np.int64)
            R5 = np.empty((n, 5), np.float32)
            rrf = np.empty(n, np.float32)
            dscore = np.empty(n, np.float32)
            din = np.empty(n, np.bool_)
            F7._rank5(np.asarray(inp.dense_ids, dtype=np.int64), np.asarray(inp.dense_scores).astype(np.float32),
                      np.asarray(inp.splade_ids, dtype=np.int64), np.asarray(inp.splade_scores).astype(np.float32),
                      lookup, fc.n_nodes, sl, R5, rrf, dscore, din)
            t1 = clock()
            if self.edges:
                s = fc._s["structural"]
                u, v, fw, bw = F7._struct_pairs(s["tindptr"], s["tcol"], s["tdir"], pool, lookup, fc.cap, self.fw_dir, self.bw_dir)
            else:
                u = v = np.empty(0, np.int64)
                fw = bw = np.empty(0, np.bool_)
            t2 = clock()
        finally:
            lookup[pool] = -1
        rrf16 = rrf.astype(np.float16).astype(np.float32)
        if self.store:
            P = np.empty((n, L3.STORE_DIM), np.float32)
            F7._decode(self.codes, pool, self.s, self.c, self.kappa, P)
            q = self.ps.query(np.asarray(inp.q))
            cos = P @ q
        else:
            P, q, cos = np.zeros((n, L3.STORE_DIM), np.float32), np.zeros(L3.STORE_DIM, np.float32), np.zeros(n, np.float32)
        t3 = clock()
        valid = sl >= 0
        S, Bk = sl[valid], buckets[valid]
        F = np.zeros((n, self.Wf), np.float32)
        if "rank" in need:
            a, _b = self.f_slot["rank"]
            F[:, a:a + 5] = R5
        G = None
        if S.size and need & {"SEED", "DISTS"}:
            G = P @ P[S].T
        if "SEED" in need and S.size:
            a, _b = self.f_slot["SEED"]
            F[:, a] = G.mean(1)
            F[:, a + 1] = G.max(1)
            b0 = Bk == 0
            if bool(b0.any()):
                F[:, a + 2] = G[:, b0].max(1)
        t4 = clock()
        if "WALK" in need:
            a, b = self.f_slot["WALK"]
            C = np.empty((n, 11))
            deg = np.empty(n, np.int64)
            F7._walk_counts(n, S, Bk, u, v, fw, bw, C, deg)
            Wv = F[:, a:b]
            Wv[:, :11] = np.log1p(C)
            Wv[:, 11] = np.log1p(deg.astype(np.float32))
            F7._first_hit(Wv, S)
        t5 = clock()
        if need & {"DISTS", "NBR2S"}:
            pu_, pv_, ptr = F7._sym_pairs(n, u, v)
            if "DISTS" in need:
                a, b = self.f_slot["DISTS"]
                r64 = rrf16.astype(np.float64)
                s_seed = r64[S] / max(float(r64.max()), 1e-12) if S.size else np.zeros(0)
                G64 = G.astype(np.float64) if S.size else np.zeros((n, 0))
                L2._dist_nb(n, ptr, pv_, S, s_seed, G64, np.ascontiguousarray(G64[S]), F[:, a:b])
            if "NBR2S" in need:
                a, b = self.f_slot["NBR2S"]
                L2._nbr2_nb(n, ptr, pv_, cos.astype(np.float64), F[:, a:b])
        t6 = clock()
        if "NBR" in need:
            a, b = self.f_slot["NBR"]
            nbr_fused(n, P, q, cos, u, v, F[:, a:b])
        if "DLIST" in need:
            a, b = self.f_slot["DLIST"]
            F7._dlist(R5[:, 0], dscore, cos, F[:, a:b])
        F16 = F.astype(np.float16).astype(np.float32)
        t7 = clock()
        XZ = np.empty((n, 2 * self.W), np.float32)
        if "SEMB" in need:
            a, b = self.x_slot["SEMB"]
            np.multiply((qemb16.astype(np.float32) @ fu.U)[None, :], P @ fu.V, out=XZ[:, a:b])
        if "SEM" in need:
            a, b = self.x_slot["SEM"]
            prod = P * q[None, :]
            XZ[:, a:b - 1] = prod
            XZ[:, b - 1] = prod.sum(1)
        t8 = clock()
        F7._fold(F16, self.src, XZ, self.W)
        base = np.empty(n, np.float32)
        F7._base(rrf16, self.base_w, self.bo, base)
        t9 = clock()
        if T is not None:
            T.update({"rank": t1 - t0, "edges": t2 - t1, "store": t3 - t2, "SEED": t4 - t3, "WALK": t5 - t4, "DISTS": t6 - t5,
                      "NBR_DLIST_round": t7 - t6, "SEMB_SEM": t8 - t7, "fold": t9 - t8})
        return XZ, base


def warm():
    """JIT every kernel (lean_fast7's and these) on toy inputs before anything is timed."""
    F7.warm()
    selftest(quick=True)


# ── self-test ────────────────────────────────────────────────────────────────


def _check(n, P, q, cos, u, v, tag):
    ref = L3.nbr_of(n, P, q, cos, u, v)
    out = np.zeros((n, 5), np.float32)
    nbr_fused(n, P, q, cos, u, v, out)
    assert np.array_equal(out.view(np.uint32), ref.view(np.uint32)), (tag, np.argwhere(out.view(np.uint32) != ref.view(np.uint32))[:5])


def selftest(quick=False):
    rng = np.random.default_rng(20261003)
    D = L3.STORE_DIM
    cases = 4 if quick else 80
    sizes = {"lt8": 0, "8to128": 0, "gt128": 0}
    for case in range(cases):
        n = int(rng.integers(1, 200))
        P = (rng.standard_normal((n, D)) * rng.choice([1e-3, 1.0, 30.0])).astype(np.float32)
        if case % 9 == 4:
            P[:, rng.random(D) < 0.2] = 0.0                     # exact zeros and signed zeros in columns
            P[rng.random(P.shape) < 0.05] = -0.0
        q = rng.standard_normal(D).astype(np.float32)
        cos = P @ q
        kind = case % 4
        if kind == 0:                                       # the fused path's own pairs, as _struct_pairs gives them
            m = int(rng.integers(0, n * min(n, 40) + 1))
            vv = np.sort(rng.integers(0, n, size=m))
            uu = rng.integers(0, n, size=m)
        elif kind == 1:                                     # unsorted destinations, multi-edges
            m = int(rng.integers(0, 4 * n + 1))
            vv = rng.integers(0, n, size=m)
            uu = rng.integers(0, n, size=m)
        elif kind == 2:                                     # a hub: one destination with hundreds of edges
            m = int(rng.integers(130, 1200))
            vv = np.where(rng.random(m) < 0.8, rng.integers(0, n), rng.integers(0, n, size=m))
            uu = rng.integers(0, n, size=m)
        else:                                               # every group size from 1 to 300 once
            gs = rng.permutation(np.arange(1, min(n, 300) + 1))
            vv = np.repeat(rng.permutation(n)[:gs.size], gs)
            vv = vv[rng.permutation(vv.size)]
            uu = rng.integers(0, n, size=vv.size)
        uu, vv = uu.astype(np.int64), vv.astype(np.int64)
        if vv.size:
            cnt = np.bincount(vv, minlength=n)
            cnt = cnt[cnt > 0]
            sizes["lt8"] += int((cnt - 1 < 8).sum())
            sizes["8to128"] += int(((cnt - 1 >= 8) & (cnt - 1 <= 128)).sum())
            sizes["gt128"] += int((cnt - 1 > 128).sum())
        _check(n, P, q, cos, uu, vv, case)
    # the store's own rows (int8 codes decoded with kappa) on _struct_pairs' pairs
    for case in range(2 if quick else 20):
        n_nodes = int(rng.integers(30, 400))
        n = int(rng.integers(1, min(n_nodes, 160)))
        pool = np.sort(rng.choice(n_nodes, size=n, replace=False)).astype(np.int64)
        lookup = np.full(n_nodes, -1, np.int32)
        lookup[pool] = np.arange(n, dtype=np.int32)
        tindptr, tcol, _trel, tdir = F7._rand_csr(rng, n_nodes, int(rng.integers(1, 12)))
        u, v, _fw, _bw = F7._struct_pairs(tindptr, tcol, tdir, pool, lookup, int(rng.choice([1, 8, 64, 1000])), 2, 1)
        basis = {"m": rng.standard_normal(1536).astype(np.float32) * 0.01,
                 "V": np.linalg.qr(rng.standard_normal((1536, L3.STORE_K)))[0].astype(np.float32),
                 "w": np.sort(rng.random(L3.STORE_K).astype(np.float32))[::-1].copy()}
        st = L3.Store(basis)
        codes = rng.integers(-127, 128, size=(n_nodes, L3.STORE_K)).astype(np.int8)
        P = st.decode(codes[pool])
        q = st.query(rng.standard_normal(1536).astype(np.float32))
        _check(n, P, q, P @ q, u, v, ("store", case))
    if not quick:
        print(f"selftest: nbr_fused = lean_mlp3.nbr_of bit for bit on {cases} random cases (sorted and unsorted "
              f"destinations, multi-edges, hubs, every group size 1..300, zero and signed-zero columns; group sizes "
              f"{sizes}) and on 20 int8 PCA store pools with _struct_pairs' pairs. all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        t = time.time()
        selftest()
        print(f"({time.time() - t:.0f}s)")
