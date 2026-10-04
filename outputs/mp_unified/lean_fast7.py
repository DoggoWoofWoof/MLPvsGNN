"""Design look (untracked; not a result and not filed): the lean MLP's per-query features in a few fused kernels.

lean_time6 (time6-2w-t2: 2wiki select, laptop, one pinned core) put p5's warm features at 2.2 ms against the GNN's
4.65 (0.47 of it) and lean's at 2.7 (0.60). At about 106 pool nodes a lean path spends most of that in about 150 small
numpy calls (two rank lists, three walk directions of three hops each, the symmetric pair set, a float16 round trip per
block, the fold's concatenate, nan_to_num and two z-scores), not in arithmetic.

FusedLean computes the XZ and base that lean_time2.Lean.__call__ followed by lean_time6.fold_rows compute, bit for bit,
in about a third of the calls:
  rank    one kernel: both cached lists' first-occurrence ranks (fast_features' list_ranks) and the five rank columns of
          its fill_retrieval (dense_rr, splade_rr, rrf, agreement, is_seed), plus the raw rrf, without the 197-column X;
  edges   one kernel: the structural pool pairs (u, v) and their direction flags, as edges_fast.typed_edges_gallop
          (every row merged with the pool, as lean_time3.fast_typed runs it) followed by fast_features' typed_attr
          give them, without the per-entry arrays and in one pass;
  store   one kernel gathers and decodes the pool's int8 PCA codes (lean_mlp3.Store.decode); the query side and the
          inner products stay numpy (Store.query, P @ q);
  SEED    numpy, as lean_time2.seed_blk (G = P @ P[S].T, its row means and maxima). G cast to float64 is DISTS's gram,
          which lean_mlp2.dist_fast computes again with the same call;
  WALK    one kernel for the eleven float64 walk counts (added in edge order, as np.bincount adds them) and the
          degrees, one numpy log1p over the counts, one for the degrees, one kernel for the first-hit one-hot;
  DISTS, NBR2S  one kernel for the symmetric pair set and its row pointers (sorted and unique: lean_mlp2.pairs and
          csr_of), then lean_mlp2's own numba kernels on it;
  NBR     lean_mlp3.nbr_of, unchanged;
  DLIST   one kernel;
  round   one float16 round trip over every rounded block at once (elementwise, so equal to block by block);
  fold    SEMB and SEM in numpy as fold_rows computes them, written into the slots of one preallocated XZ; one kernel
          copies the rounded blocks into their slots, applies nan_to_num and writes the z-scores (column sums added in
          row order from +0.0: numpy's order for an axis-0 reduction of a C-contiguous matrix); one kernel for the base
          (the rrf column's z-score with numpy's pairwise sums from +0.0: the order of an (n, 1) reduction).
The forward is lean_time6.fold_forward, unchanged. The summation orders were measured on this numpy (2.3.2) against
candidates (scratchpad red_order.py, 400 random cases each, and signed-zero cases); --selftest checks every kernel
against the code it replaces.

Supported blocks: rank, SEM, SEMB, SEED, WALK, NBR, DLIST, DISTS and NBR2S on the int8 PCA store (lean_mlp3 saves, and
lean_mlp4 saves of sets without AW). A set with a compiled block (dense_cos, seed_e, topo_*, depth_*, gcs, ...) or an
NER/kNN block (DISTF, WALKF, NBRF, NBR2F) is refused.

    python outputs/mp_unified/lean_fast7.py --selftest
"""
import os
import sys

os.environ.setdefault("LEAN_TIME_THREADS", "1")       # lean_time2 reads it at import and sets the thread variables
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ["LEAN_TIME_THREADS"])
sys.dont_write_bytecode = True

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import edges_fast as EF  # noqa: E402
import lean_fuse as LFU  # noqa: E402
import lean_time2 as T2  # noqa: E402

LM, L2, L3, LT, FF, V2 = T2.LM, T2.L2, T2.L3, T2.LT, T2.FF, T2.V2
FWD, BWD = LT.FWD, LT.BWD
SUPPORTED = ("rank", "SEM", "SEMB", "SEED", "WALK", "NBR", "DLIST", "DISTS", "NBR2S")
WIDTH = {"rank": 5, "SEED": 3, "WALK": 16, "NBR": 5, "DLIST": 3, "DISTS": 14, "NBR2S": 3, "SEM": L3.STORE_DIM + 1}
RANK_COLS = ["dense_rr", "splade_rr", "rrf", "agreement", "is_seed"]
assert LM.SPLIT["rank"] == RANK_COLS, LM.SPLIT["rank"]
assert {FWD, BWD} == {3, 4}, (FWD, BWD)          # typed_attr: column 3 marks dir 2 entries, column 4 dir 1 entries
DIR_OF_COL = {3: 2, 4: 1}
EDGE_USERS = T2.EDGE_USERS
STORE_USERS = T2.STORE_USERS
EPS = np.float32(LFU.EPS)                          # numpy compares sd < 1e-6 in float32 (a Python float is weak)
_F0 = FF._F0
_F1 = FF._F1
_RRF = FF._RRF


# ── kernels ──────────────────────────────────────────────────────────────────


@njit(nogil=True, cache=True)
def _rank5(d_ids, d_sc, s_ids, s_sc, lookup, n_nodes, sl, R5, rrf, dscore, din):
    """list_ranks on both lists and fill_retrieval's dense_rr, splade_rr, rrf, agreement and is_seed into R5 (n, 5);
    the raw rrf, the listed dense score (0 elsewhere) and the dense hit flags."""
    n = R5.shape[0]
    d_rank = np.zeros(n, np.float32)
    d_score = np.zeros(n, np.float32)
    s_rank = np.zeros(n, np.float32)
    d_in = np.zeros(n, np.bool_)
    s_in = np.zeros(n, np.bool_)
    for i in range(d_ids.shape[0] - 1, -1, -1):
        g = d_ids[i]
        if g < 0 or g >= n_nodes:
            continue
        loc = lookup[g]
        if loc >= 0:
            d_rank[loc] = np.float32(i + 1)
            d_score[loc] = d_sc[i]
            d_in[loc] = True
    for i in range(s_ids.shape[0] - 1, -1, -1):
        g = s_ids[i]
        if g < 0 or g >= n_nodes:
            continue
        loc = lookup[g]
        if loc >= 0:
            s_rank[loc] = np.float32(i + 1)
            s_in[loc] = True
    for v in range(n):
        a = _F0
        b = _F0
        R5[v, 0] = _F0
        R5[v, 1] = _F0
        R5[v, 3] = _F0
        R5[v, 4] = _F0
        if d_in[v]:
            R5[v, 0] = _F1 / max(d_rank[v], _F1)
            a = _F1 / (_RRF + d_rank[v])
        if s_in[v]:
            R5[v, 1] = _F1 / max(s_rank[v], _F1)
            b = _F1 / (_RRF + s_rank[v])
        r = a + b
        rrf[v] = r
        R5[v, 2] = r
        if d_in[v] and s_in[v]:
            R5[v, 3] = _F1
        dscore[v] = d_score[v] if d_in[v] else _F0
        din[v] = d_in[v]
    for k in range(sl.shape[0]):
        loc = sl[k]
        if loc < 0:                                    # fill_retrieval's X[loc] wraps a missing seed (-1) to the last row
            loc += n
        R5[loc, 4] = _F1


@njit(nogil=True)
def _lower_bound(a, lo, hi, x):
    """edges_fast._lower_bound."""
    if lo >= hi or a[lo] >= x:
        return lo
    prev = lo
    step = 1
    cur = lo + 1
    while cur < hi and a[cur] < x:
        prev = cur
        step *= 2
        cur = prev + step
    if cur > hi:
        cur = hi
    left = prev + 1
    right = cur
    while left < right:
        mid = (left + right) >> 1
        if a[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left


@njit(nogil=True, cache=True)
def _struct_pairs(tindptr, tcol, tdir, pool, lookup, cap, fw_dir, bw_dir):
    """typed_edges_gallop(..., scan_max=-1)'s pairs (pu = neighbour, pv = owner; int64) and typed_attr's direction
    columns as flags: fw[p] is True when one of the pair's entries has tdir == fw_dir (bw likewise)."""
    n = pool.shape[0]
    m = n * min(n, cap) if cap > 0 else 0
    pu = np.empty(m, np.int64)
    pv = np.empty(m, np.int64)
    fw = np.zeros(m, np.bool_)
    bw = np.zeros(m, np.bool_)
    p = 0
    for v in range(n):
        g = pool[v]
        hi = tindptr[g + 1]
        pos = tindptr[g]
        pairs_v = 0
        for j in range(n):
            t = pool[j]
            pos = _lower_bound(tcol, pos, hi, t)
            if pos >= hi:
                break
            if tcol[pos] != t:
                continue
            pairs_v += 1
            if pairs_v > cap:
                break
            pu[p] = j
            pv[p] = v
            f = False
            b = False
            while pos < hi and tcol[pos] == t:
                d = tdir[pos]
                if d == fw_dir:
                    f = True
                if d == bw_dir:
                    b = True
                pos += 1
            fw[p] = f
            bw[p] = b
            p += 1
    return pu[:p], pv[:p], fw[:p], bw[:p]


@njit(nogil=True, cache=True)
def _decode(codes, pool, s, c, kappa, P):
    """lean_mlp3.Store.decode(codes[pool]) into P (n, STORE_K + 1): codes * s + c, then kappa."""
    n = pool.shape[0]
    k = s.shape[0]
    for v in range(n):
        g = pool[v]
        for j in range(k):
            P[v, j] = np.float32(codes[g, j]) * s[j] + c[j]
        P[v, k] = kappa


@njit(nogil=True, cache=True)
def _walk_counts(n, S, Bk, u, v, fw, bw, C, deg):
    """walk_blk's eleven float64 hop counts into C (n, 11) and the in-degrees (int64) into deg."""
    m = u.shape[0]
    s0 = np.zeros(n)
    sb = np.zeros(n)
    for j in range(S.shape[0]):
        s0[S[j]] = 1.0
        if Bk[j] == 0:
            sb[S[j]] = 1.0
    col = 0
    for mode in range(3):
        c = s0
        for _h in range(3):
            nc = np.zeros(n)
            for e in range(m):
                if mode == 0 or (mode == 1 and fw[e]) or (mode == 2 and bw[e]):
                    nc[v[e]] += c[u[e]]
            for x in range(n):
                C[x, col] = nc[x]
            c = nc
            col += 1
    c = sb
    for _h in range(2):
        nc = np.zeros(n)
        for e in range(m):
            nc[v[e]] += c[u[e]]
        for x in range(n):
            C[x, col] = nc[x]
        c = nc
        col += 1
    for x in range(n):
        deg[x] = 0
    for e in range(m):
        deg[v[e]] += 1


@njit(nogil=True, cache=True)
def _first_hit(W, S):
    """walk_blk's first-hit one-hot (columns 12..15) from its unmasked hop columns 0..2 (after log1p)."""
    n = W.shape[0]
    first = np.full(n, 3, np.int64)
    for x in range(n):
        for h in (2, 1, 0):
            if W[x, h] > 0:
                first[x] = h
    for j in range(S.shape[0]):
        first[S[j]] = -1
    for x in range(n):
        for h in range(3):
            W[x, 12 + h] = _F1 if first[x] == h else _F0
        W[x, 15] = _F1 if first[x] == -1 else _F0


@njit(nogil=True, cache=True)
def _sym_pairs(n, u, v):
    """lean_mlp2.pairs(n, u, v) and csr_of(n, its u): both directions, diagonal dropped, duplicates collapsed, sorted by
    key u * n + v; the row pointers of the sorted pairs."""
    m = u.shape[0]
    keys = np.empty(2 * m, np.int64)
    k = 0
    for e in range(m):
        a = u[e]
        b = v[e]
        if a != b:
            keys[k] = a * n + b
            keys[k + 1] = b * n + a
            k += 2
    keys = np.sort(keys[:k])
    nu = 0
    for i in range(k):
        if i == 0 or keys[i] != keys[i - 1]:
            keys[nu] = keys[i]
            nu += 1
    pu = np.empty(nu, np.int64)
    pv = np.empty(nu, np.int64)
    ptr = np.zeros(n + 1, np.int64)
    for i in range(nu):
        pu[i] = keys[i] // n
        pv[i] = keys[i] % n
        ptr[pu[i] + 1] += 1
    for x in range(n):
        ptr[x + 1] += ptr[x]
    return pu, pv, ptr


@njit(nogil=True, cache=True)
def _dlist(dense_rr, dscore, cos, out):
    """DLIST: the listed dense score, the listed flag, the store cosine of unlisted nodes."""
    for x in range(out.shape[0]):
        if dense_rr[x] > 0:
            out[x, 0] = dscore[x]
            out[x, 1] = _F1
            out[x, 2] = _F0
        else:
            out[x, 0] = _F0
            out[x, 1] = _F0
            out[x, 2] = cos[x]


@njit(nogil=True, cache=True)
def _fold(F16, src, XZ, W):
    """XZ[:, :W]: each column j with src[j] >= 0 copied from F16[:, src[j]] (the others already hold SEMB / SEM);
    nan_to_num on all W; XZ[:, W:]: lean_fuse.zscore of them (column sums in row order from +0.0)."""
    n = XZ.shape[0]
    for x in range(n):
        for j in range(W):
            if src[j] >= 0:
                y = F16[x, src[j]]
            else:
                y = XZ[x, j]
            if not np.isfinite(y):
                y = _F0
            XZ[x, j] = y
    mu = np.zeros(W, np.float32)
    for x in range(n):
        for j in range(W):
            mu[j] += XZ[x, j]
    nf = np.float32(n)
    for j in range(W):
        mu[j] = mu[j] / nf
    ss = np.zeros(W, np.float32)
    for x in range(n):
        for j in range(W):
            d = XZ[x, j] - mu[j]
            ss[j] += d * d
    for j in range(W):
        sd = np.sqrt(ss[j] / nf)
        den = max(sd, EPS)
        for x in range(n):
            if sd < EPS:
                XZ[x, W + j] = _F0
            else:
                XZ[x, W + j] = (XZ[x, j] - mu[j]) / den


@njit(nogil=True)
def _pw_leaf(a, lo, n):
    """numpy's float32 pairwise sum of a[lo:lo + n] for n <= 128 (FLOAT_pairwise_sum: -0.0 start below 8 terms,
    eight accumulators from 8 to 128)."""
    if n < 8:
        res = np.float32(-0.0)
        for i in range(n):
            res += a[lo + i]
        return res
    r0 = a[lo]
    r1 = a[lo + 1]
    r2 = a[lo + 2]
    r3 = a[lo + 3]
    r4 = a[lo + 4]
    r5 = a[lo + 5]
    r6 = a[lo + 6]
    r7 = a[lo + 7]
    i = 8
    while i < n - (n % 8):
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@njit(nogil=True)
def _pairwise(a, lo, n):
    """numpy's float32 pairwise sum of a[lo:lo + n]: above 128 terms, the sum of the two halves split at
    n2 = n // 2 - (n // 2) % 8, each summed the same way. Iterative (an explicit stack), because numba cannot
    reliably load a cached function that calls a recursive one (it segfaults)."""
    if n <= 128:
        return _pw_leaf(a, lo, n)
    s_lo = np.empty(96, np.int64)
    s_n = np.empty(96, np.int64)
    s_join = np.empty(96, np.bool_)
    vals = np.empty(96, np.float32)
    top = 0
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
            vals[nv - 2] = vals[nv - 2] + vals[nv - 1]          # left + right
            nv -= 1
        elif m <= 128:
            vals[nv] = _pw_leaf(a, l, m)
            nv += 1
        else:
            m2 = m // 2
            m2 -= m2 % 8
            s_lo[top] = l                                       # the join, after both halves
            s_n[top] = m
            s_join[top] = True
            s_lo[top + 1] = l + m2                              # the right half, summed second
            s_n[top + 1] = m - m2
            s_join[top + 1] = False
            s_lo[top + 2] = l                                   # the left half, summed first
            s_n[top + 2] = m2
            s_join[top + 2] = False
            top += 3
    return vals[0]


@njit(nogil=True, cache=True)
def _base(rrf16, base_w, bo, out):
    """lean_fuse.zscore(rrf16[:, None])[:, 0] * base_w + bo: an (n, 1) reduction, so numpy's pairwise sums from +0.0."""
    n = rrf16.shape[0]
    nf = np.float32(n)
    mu = (_F0 + _pairwise(rrf16, 0, n)) / nf
    d = np.empty(n, np.float32)
    for x in range(n):
        d[x] = rrf16[x] - mu
    sq = np.empty(n, np.float32)
    for x in range(n):
        sq[x] = d[x] * d[x]
    sd = np.sqrt((_F0 + _pairwise(sq, 0, n)) / nf)
    den = max(sd, EPS)
    for x in range(n):
        z = _F0 if sd < EPS else d[x] / den
        out[x] = z * base_w + bo


# ── the fused path ───────────────────────────────────────────────────────────


class FusedLean:
    """One served lean set (a lean_time2.Served with .folded, a lean_time3.Folded) on the int8 PCA store."""

    def __init__(self, fc, codes, pstore, sv):
        if sv.store != "pca256":
            raise SystemExit(f"{sv.name}: FusedLean serves the int8 PCA store only (store {sv.store})")
        fu = sv.folded.fu
        bad = [b for b in fu.active if b not in SUPPORTED]
        if bad:
            raise SystemExit(f"{sv.name}: blocks {bad} are not fused")
        self.fc, self.codes, self.ps, self.sv, self.fu = fc, codes, pstore, sv, fu
        self.need = set(fu.active)
        assert self.need == set(sv.active), (self.need, sv.active)
        self.f_slot, self.x_slot = {}, {}
        fo = xo = 0
        for b in fu.active:
            w = int(fu.widths[b])
            if b in WIDTH:
                assert w == WIDTH[b], (b, w)
            elif b != "SEMB":
                raise SystemExit(f"{sv.name}: no width for {b}")
            self.x_slot[b] = (xo, xo + w)
            xo += w
            if b not in ("SEM", "SEMB"):
                self.f_slot[b] = (fo, fo + w)
                fo += w
        self.W, self.Wf = xo, fo
        if 2 * self.W != fu.in_w:
            raise SystemExit(f"{sv.name}: the fold reads {fu.in_w} inputs, the blocks give {2 * self.W}")
        self.src = np.full(self.W, -1, np.int64)
        for b, (fa, fb) in self.f_slot.items():
            xa, _xb = self.x_slot[b]
            self.src[xa:xa + fb - fa] = np.arange(fa, fb)
        self.edges = bool(self.need & EDGE_USERS)
        self.store = bool(self.need & STORE_USERS)
        self.fw_dir, self.bw_dir = DIR_OF_COL[FWD], DIR_OF_COL[BWD]
        self.s = np.ascontiguousarray(pstore.s, dtype=np.float32)
        self.c = np.ascontiguousarray(pstore.c, dtype=np.float32)
        self.kappa = np.float32(pstore.kappa)
        self.base_w, self.bo = np.float32(fu.base_w), np.float32(fu.bo)
        self.name = sv.name + "+f7"

    def __call__(self, inp, pool, seeds, buckets, qemb16, T=None):
        """(XZ, base) for one query, equal bit for bit to lean_time6.fold_rows(sv.folded, *lean_time2.Lean(...)); T, if
        given, receives the seconds per stage."""
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
            _rank5(np.asarray(inp.dense_ids, dtype=np.int64), np.asarray(inp.dense_scores).astype(np.float32),
                   np.asarray(inp.splade_ids, dtype=np.int64), np.asarray(inp.splade_scores).astype(np.float32),
                   lookup, fc.n_nodes, sl, R5, rrf, dscore, din)
            t1 = clock()
            if self.edges:
                s = fc._s["structural"]
                u, v, fw, bw = _struct_pairs(s["tindptr"], s["tcol"], s["tdir"], pool, lookup, fc.cap, self.fw_dir, self.bw_dir)
            else:
                u = v = np.empty(0, np.int64)
                fw = bw = np.empty(0, np.bool_)
            t2 = clock()
        finally:
            lookup[pool] = -1
        rrf16 = rrf.astype(np.float16).astype(np.float32)
        if self.store:
            P = np.empty((n, L3.STORE_DIM), np.float32)
            _decode(self.codes, pool, self.s, self.c, self.kappa, P)
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
            _walk_counts(n, S, Bk, u, v, fw, bw, C, deg)
            Wv = F[:, a:b]
            Wv[:, :11] = np.log1p(C)
            Wv[:, 11] = np.log1p(deg.astype(np.float32))
            _first_hit(Wv, S)
        t5 = clock()
        if need & {"DISTS", "NBR2S"}:
            pu_, pv_, ptr = _sym_pairs(n, u, v)
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
            F[:, a:b] = L3.nbr_of(n, P, q, cos, u, v)
        if "DLIST" in need:
            a, b = self.f_slot["DLIST"]
            _dlist(R5[:, 0], dscore, cos, F[:, a:b])
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
        _fold(F16, self.src, XZ, self.W)
        base = np.empty(n, np.float32)
        _base(rrf16, self.base_w, self.bo, base)
        t9 = clock()
        if T is not None:
            T.update({"rank": t1 - t0, "edges": t2 - t1, "store": t3 - t2, "SEED": t4 - t3, "WALK": t5 - t4, "DISTS": t6 - t5,
                      "NBR_DLIST_round": t7 - t6, "SEMB_SEM": t8 - t7, "fold": t9 - t8})
        return XZ, base


def warm():
    """JIT every kernel (cached on disk where numba can) on a toy query before anything is timed."""
    selftest(quick=True)


# ── self-test ────────────────────────────────────────────────────────────────


def _rand_csr(rng, n_nodes, avg):
    """A structural-like typed store: rows sorted by (tail, rel), a few multi-entries per neighbour, dirs 1/2."""
    indptr = [0]
    col, rel, dr = [], [], []
    for _g in range(n_nodes):
        k = int(rng.poisson(avg)) if rng.random() > 0.05 else int(rng.integers(avg * 5, avg * 12))
        nb = np.sort(rng.choice(n_nodes, size=min(k, n_nodes), replace=False))
        for t in nb:
            reps = 1 + int(rng.random() < 0.3)
            for r in range(reps):
                col.append(int(t))
                rel.append(r)
                dr.append(int(rng.integers(0, 3)))
        indptr.append(len(col))
    return (np.asarray(indptr, np.int64), np.asarray(col, np.int32), np.asarray(rel, np.int32), np.asarray(dr, np.int32))


def selftest(quick=False):
    rng = np.random.default_rng(20261003)
    K = FF._kernels(False)
    cases = 3 if quick else 60
    for case in range(cases):
        n_nodes = int(rng.integers(30, 400))
        n = int(rng.integers(1, min(n_nodes, 160)))
        pool = np.sort(rng.choice(n_nodes, size=n, replace=False)).astype(np.int64)
        lookup = np.full(n_nodes, -1, np.int32)
        lookup[pool] = np.arange(n, dtype=np.int32)
        # rank lists: duplicates, out-of-range ids, nodes outside the pool
        d_ids = rng.integers(-3, n_nodes + 5, size=int(rng.integers(0, 60))).astype(np.int64)
        s_ids = rng.integers(-3, n_nodes + 5, size=int(rng.integers(0, 60))).astype(np.int64)
        d_sc = rng.standard_normal(d_ids.size).astype(np.float32)
        s_sc = np.sort(rng.random(s_ids.size).astype(np.float32))[::-1].copy()
        seeds = rng.choice(pool, size=min(n, int(rng.integers(0, 11))), replace=False)
        if case % 7 == 3 and seeds.size:
            seeds = np.r_[seeds, np.int64(pool.max() + 1 if pool.max() + 1 < n_nodes else 0)]   # a seed outside the pool
        sl = lookup[seeds].astype(np.int64)
        X = np.zeros((n, 10), np.float32)
        d_rank, d_score, d_in = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.bool_)
        s_rank, s_score, s_in = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.bool_)
        K.list_ranks(d_ids, d_sc, lookup, n_nodes, d_rank, d_score, d_in)
        K.list_ranks(s_ids, s_sc, lookup, n_nodes, s_rank, s_score, s_in)
        top = float(s_sc[0]) if s_sc.size else 0.0
        rrf_ref = np.empty(n, np.float32)
        K.fill_retrieval(X, np.zeros(n, np.float32), d_in, d_rank, s_in, s_rank, s_score, top > 0, np.float32(max(top, 1e-12)), sl,
                         np.arange(10, dtype=np.int64), rrf_ref)
        R5 = np.empty((n, 5), np.float32)
        rrf, dscore, din = np.empty(n, np.float32), np.empty(n, np.float32), np.empty(n, np.bool_)
        _rank5(d_ids, d_sc, s_ids, s_sc, lookup, n_nodes, sl, R5, rrf, dscore, din)
        idx = [LT.RET_COLS.index(c) for c in RANK_COLS]
        assert np.array_equal(R5.view(np.uint32), np.ascontiguousarray(X[:, idx]).view(np.uint32)), case
        assert np.array_equal(rrf.view(np.uint32), rrf_ref.view(np.uint32)), case
        assert np.array_equal(dscore.view(np.uint32), np.where(d_in, d_score, 0.0).astype(np.float32).view(np.uint32)), case
        assert np.array_equal(din, d_in), case
        # structural pairs and flags
        tindptr, tcol, trel, tdir = _rand_csr(rng, n_nodes, int(rng.integers(1, 12)))
        cap = int(rng.choice([1, 3, 8, 64, 1000]))
        ev, eu, erel, edir, pair_id, pu_r, pv_r = EF.typed_edges_gallop(tindptr, tcol, trel, tdir, pool, lookup, cap, -1)
        attr = K.typed_attr(pair_id, erel, edir, np.zeros(1, np.float32), False, int(pu_r.size))
        u, v, fw, bw = _struct_pairs(tindptr, tcol, tdir, pool, lookup, cap, DIR_OF_COL[FWD], DIR_OF_COL[BWD])
        assert np.array_equal(u, pu_r.astype(np.int64)) and np.array_equal(v, pv_r.astype(np.int64)), case
        assert np.array_equal(fw, attr[:, FWD] > 0.5) and np.array_equal(bw, attr[:, BWD] > 0.5), case
        # WALK
        valid = sl >= 0
        S = sl[valid]
        Bk = rng.integers(0, 2, size=S.size).astype(np.int64)
        ref = T2.walk_blk(n, S, Bk, u, v, fw, bw)
        C, deg = np.empty((n, 11)), np.empty(n, np.int64)
        _walk_counts(n, S, Bk, u, v, fw, bw, C, deg)
        Wm = np.zeros((n, 16), np.float32)
        Wm[:, :11] = np.log1p(C)
        Wm[:, 11] = np.log1p(deg.astype(np.float32))
        _first_hit(Wm, S)
        assert np.array_equal(Wm.view(np.uint32), ref.view(np.uint32)), case
        # pairs and their csr
        a_, b_ = L2.pairs(n, u, v)
        pu_, pv_, ptr = _sym_pairs(n, u, v)
        assert np.array_equal(pu_, a_) and np.array_equal(pv_, b_) and np.array_equal(ptr, L2.csr_of(n, a_)), case
        # store decode, SEED's G reused by DISTS, NBR2S
        basis = {"m": rng.standard_normal(1536).astype(np.float32) * 0.01,
                 "V": np.linalg.qr(rng.standard_normal((1536, L3.STORE_K)))[0].astype(np.float32),
                 "w": np.sort(rng.random(L3.STORE_K).astype(np.float32))[::-1].copy()}
        st = L3.Store(basis)
        codes = rng.integers(-127, 128, size=(n_nodes, L3.STORE_K)).astype(np.int8)
        P = np.empty((n, L3.STORE_DIM), np.float32)
        _decode(codes, pool, np.ascontiguousarray(st.s, dtype=np.float32), np.ascontiguousarray(st.c, dtype=np.float32),
                np.float32(st.kappa), P)
        Pr = st.decode(codes[pool])
        assert np.array_equal(P.view(np.uint32), Pr.view(np.uint32)), case
        if S.size:
            G = P @ P[S].T
            r16 = rrf.astype(np.float16).astype(np.float64)
            s_seed = r16[S] / max(float(r16.max()), 1e-12)
            G64 = G.astype(np.float64)
            out = np.zeros((n, 20), np.float32)
            L2._dist_nb(n, ptr, pv_, S, s_seed, G64, np.ascontiguousarray(G64[S]), out[:, 3:17])
            dref = L2.dist_fast(n, P, S, s_seed, a_, b_)
            assert np.array_equal(out[:, 3:17].view(np.uint32), dref.view(np.uint32)), case
        cos = P @ st.query(rng.standard_normal(1536).astype(np.float32))
        out = np.zeros((n, 7), np.float32)
        L2._nbr2_nb(n, ptr, pv_, cos.astype(np.float64), out[:, 2:5])
        assert np.array_equal(out[:, 2:5].view(np.uint32), L2.nbr2_fast(n, cos, a_, b_).view(np.uint32)), case
        # DLIST
        dl = np.zeros((n, 3), np.float32)
        _dlist(R5[:, 0], dscore, cos, dl)
        listed = X[:, LT.RET_COLS.index("dense_rr")] > 0
        dref = np.stack([np.where(listed, dscore, 0.0), listed, np.where(listed, 0.0, cos)], 1).astype(np.float32)
        assert np.array_equal(dl.view(np.uint32), dref.view(np.uint32)), case
        # the fold: rounded columns copied, nan_to_num, z-scores; and the base
        W = int(rng.integers(2, 400))                               # a real set has W >= 5; numpy reduces (n, 1) pairwise
        F16 = (rng.standard_normal((n, W + 3)) * rng.choice([1e-4, 1.0, 50.0], size=W + 3)).astype(np.float16).astype(np.float32)
        F16[rng.random(F16.shape) < 0.03] = np.nan
        F16[rng.random(F16.shape) < 0.02] = np.inf
        F16[rng.random(F16.shape) < 0.02] = -np.inf
        F16[rng.random(F16.shape) < 0.03] = -0.0
        F16[:, rng.random(W + 3) < 0.1] = 2.5                       # constant columns: sd 0
        F16[:, rng.random(W + 3) < 0.05] = -0.0                     # signed-zero columns
        src = rng.permutation(W + 3)[:W].astype(np.int64)
        src[rng.random(W) < 0.25] = -1                              # columns already in XZ (SEMB, SEM)
        XZ = np.empty((n, 2 * W), np.float32)
        own = rng.standard_normal((n, W)).astype(np.float32)
        XZ[:, :W] = own
        Xr = np.where(src[None, :] >= 0, F16[:, np.maximum(src, 0)], own).astype(np.float32)
        Xr = np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0)
        XZr = np.concatenate([Xr, LFU.zscore(Xr)], 1)
        _fold(F16, src, XZ, W)
        assert np.array_equal(XZ.view(np.uint32), XZr.view(np.uint32)), case
        r16 = rrf.astype(np.float16).astype(np.float32)
        if case % 5 == 0:
            r16 = (rng.standard_normal(n) * 1e-7).astype(np.float32)   # near the eps edge
        bw_, bo_ = np.float32(rng.standard_normal()), np.float32(rng.standard_normal())
        base = np.empty(n, np.float32)
        _base(r16, bw_, bo_, base)
        bref = LFU.zscore(r16[:, None].astype(np.float32))[:, 0] * bw_ + bo_
        assert np.array_equal(base.view(np.uint32), bref.view(np.uint32)), case
    # pairwise sums against numpy on the lengths that switch branches
    for m in (0, 1, 7, 8, 9, 15, 16, 17, 127, 128, 129, 255, 256, 257, 300, 1000):
        x = (rng.standard_normal(m) * 3).astype(np.float32)
        assert np.float32(_F0 + _pairwise(x, 0, m)).view(np.uint32) == np.add.reduce(x[:, None], axis=0)[0].view(np.uint32) if m else True, m
    if not quick:
        print(f"selftest: on {cases} random pools (1 to 159 nodes, lists with duplicates, out-of-range ids and a seed outside "
              "the pool, typed rows with hubs and multi-entry neighbours, caps 1 to 1000) every kernel equals the code it "
              "replaces bit for bit: _rank5 = list_ranks x2 + fill_retrieval's five rank columns and rrf; _struct_pairs = "
              "typed_edges_gallop(-1) + typed_attr's direction flags; _walk_counts + log1p + _first_hit = walk_blk; "
              "_sym_pairs = lean_mlp2.pairs + csr_of; _decode = Store.decode; SEED's G reused in DISTS = dist_fast; NBR2S = "
              "nbr2_fast; _dlist = DLIST; _fold = nan_to_num + lean_fuse.zscore (with NaN, +-inf, constant and "
              "pre-filled columns); _base = lean_fuse.zscore of the rrf column * base_w + bo (near-eps sd included); "
              "_pairwise = numpy's (n, 1) reduction on lengths 0 to 1000. all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
