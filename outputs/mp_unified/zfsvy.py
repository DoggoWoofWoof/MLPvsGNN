"""FSV's second set of faster exact forms of the cold path (candidate amendment A6 of docs/FSV_FAST_VERSIONS.md). Development
only until A6 is declared: no timed number is made here, and nothing of zfsvc.py, zfsvx.py, zpko.py, ltwdiag.py or the C-stage
modules is edited (they are imported). It runs on top of zfsvx's forms (A4), which it installs first.

    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy.py profile --dataset musique --questions 40 --blas 2 --lp 8
"""
import sys

import zfsvx as X                       # first: it imports zfsvc, which sets the thread variables before numpy loads

import argparse
import contextlib
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

C = X.C
np = C.np
clock = C.clock


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zfsvy: {msg}", flush=True)


def p50(v):
    return float(np.median(np.asarray(v, np.float64))) * 1000.0


def mean(v):
    return float(np.mean(np.asarray(v, np.float64))) * 1000.0


# ── the forms ────────────────────────────────────────────────────────────────

STATS = {"zref_x": 0, "zref_orig": 0, "next_x": 0, "unit_x": 0, "links_x": 0, "links_orig": 0}
ON = {"zref": True, "nbr": True, "unit": True, "links": True}              # the forms in force (devy switches them one at a time; a run has all)
_ORIG = {}
_Y = {}


def ykernels():
    """The numba kernels of A6's forms, compiled on first use (cache=True beside the module, as zfsvx's). Serial, float32 scalars
    only (a Python float constant would promote an operation to float64), no fast-math (no reassociation, no fused multiply-add),
    error_model numpy (no zero-division branch: every divisor is at least eps or 1)."""
    if _Y:
        return _Y
    import numba

    @numba.njit(cache=True, error_model="numpy")
    def zref(x, w, lo, hi, eps, out):
        """lean_screen3.seg_zscore_ref for the rows lo..hi of one segment (one question's), every column at once, in the original's
        float32 operations in their order: torch's index_add_ takes the rows one after the other, so each column's sum is the
        sequential float32 sum in row order; the mean is that sum over the count, the centred values are float32 differences, the
        variance is the sequential sum of their squares (times w, for the reference rows) over the count, sd = 0 where the variance
        is 0, else its square root; a column takes the reference rows' z-score when there are at least two of them and sd >= eps,
        else the whole segment's (0 under eps). Another segment's rows never enter a segment's sums, as in the original."""
        m = x.shape[1]
        n = hi - lo
        zero = np.float32(0.0)
        one = np.float32(1.0)
        den_all = np.float32(max(n, 1))
        s_all = np.zeros(m, np.float32)
        s_ref = np.zeros(m, np.float32)
        cnt = zero
        for i in range(lo, hi):
            wi = w[i]
            cnt += wi
            for k in range(m):
                xv = x[i, k]
                s_all[k] += xv
                s_ref[k] += xv * wi
        den_ref = cnt if cnt > one else one
        mean_all = np.empty(m, np.float32)
        mean_ref = np.empty(m, np.float32)
        for k in range(m):
            mean_all[k] = s_all[k] / den_all
            mean_ref[k] = s_ref[k] / den_ref
        v_all = np.zeros(m, np.float32)
        v_ref = np.zeros(m, np.float32)
        for i in range(lo, hi):
            wi = w[i]
            for k in range(m):
                xv = x[i, k]
                c1 = xv - mean_all[k]
                v_all[k] += c1 * c1
                c2 = xv - mean_ref[k]
                v_ref[k] += (c2 * c2) * wi
        mu = np.empty(m, np.float32)
        dv = np.empty(m, np.float32)
        zr = np.zeros(m, np.bool_)
        for k in range(m):
            va = v_all[k] / den_all
            sda = zero if va == 0 else np.sqrt(va)
            vr = v_ref[k] / den_ref
            sdr = zero if vr == 0 else np.sqrt(vr)
            if cnt >= 2 and sdr >= eps:
                mu[k] = mean_ref[k]
                dv[k] = sdr
            else:
                mu[k] = mean_all[k]
                dv[k] = eps if sda < eps else sda       # clamp_min: a NaN stays NaN
                zr[k] = sda < eps
        for i in range(lo, hi):
            for k in range(m):
                if zr[k]:
                    out[i, k] = zero
                else:
                    out[i, k] = (x[i, k] - mu[k]) / dv[k]

    @numba.njit(cache=True, error_model="numpy")
    def zref_segs(x, w, bounds, eps, out):
        """zref on each segment of a batch: the question s's rows are bounds[s]..bounds[s + 1] (contiguous, in question order)."""
        for s in range(bounds.size - 1):
            if bounds[s + 1] > bounds[s]:
                zref(x, w, bounds[s], bounds[s + 1], eps, out)

    @numba.njit(cache=True, error_model="numpy")
    def nbr_lists(u, v, f, top, in_top):
        """LTW.fam_bits and the candidate test of zfsvc.next_cold for every anchor at once, from the links' sorted symmetric
        closure (links_fast's output: the entries (a, b, family) of both directions, sorted by (a, b * FAMS + family), no entry
        twice): the entries of anchor a are the run with u == a (found by binary search), the neighbours b come in ascending
        order with their families together, the bits are (1 << family) or-ed over a neighbour's entries, and a neighbour that is
        one of the five anchors (the anchor itself included) is not a candidate. The original's two directions give the same
        neighbours and bits because the closure holds each entry's mirror."""
        n_e = u.size
        k5 = top.size
        s = np.empty(k5, np.int64)
        e = np.empty(k5, np.int64)
        tot = 0
        for k in range(k5):
            a = top[k]
            lo = 0
            hi = n_e
            while lo < hi:
                mid = (lo + hi) >> 1
                if u[mid] < a:
                    lo = mid + 1
                else:
                    hi = mid
            s[k] = lo
            hi = n_e
            while lo < hi:
                mid = (lo + hi) >> 1
                if u[mid] <= a:
                    lo = mid + 1
                else:
                    hi = mid
            e[k] = lo
            tot += e[k] - s[k]
        c = np.empty(tot, np.int64)
        bt = np.empty(tot, np.int64)
        off = np.zeros(k5 + 1, np.int64)
        pos = 0
        for k in range(k5):
            i = s[k]
            while i < e[k]:
                b = v[i]
                bits = np.int64(0)
                while i < e[k] and v[i] == b:
                    bits |= np.int64(1) << np.int64(f[i])
                    i += 1
                if not in_top[b]:
                    c[pos] = b
                    bt[pos] = bits
                    pos += 1
            off[k + 1] = pos
        return c[:pos], bt[:pos], off

    @numba.njit(cache=True, error_model="numpy")
    def unit32(x, kind, lo, cnt):
        """ltwdiag.unit in place on float32 rows: a / max(norm, 1e-12), the norm numpy's float32 one: the squares are float32
        products, their sum is numpy's pairwise sum (the postfix program kind/lo/cnt: a leaf of up to 128 elements is eight
        interleaved accumulators over blocks of eight, combined as ((0+1)+(2+3))+((4+5)+(6+7)), then the remainder one by
        one; larger runs split at n/2 less its remainder mod 8, left plus right), then float32's square root."""
        m, d = x.shape
        tiny = np.float32(1e-12)
        sq = np.empty(d, np.float32)
        stack = np.empty(96, np.float32)
        r = np.empty(8, np.float32)
        for i in range(m):
            for j in range(d):
                v = x[i, j]
                sq[j] = v * v
            sp = 0
            for p in range(kind.size):
                if kind[p] == 0:
                    a0 = lo[p]
                    nn = cnt[p]
                    if nn < 8:
                        res = np.float32(-0.0)
                        for t in range(nn):
                            res += sq[a0 + t]
                    else:
                        for t in range(8):
                            r[t] = sq[a0 + t]
                        t = 8
                        lim = nn - (nn % 8)
                        while t < lim:
                            for u in range(8):
                                r[u] += sq[a0 + t + u]
                            t += 8
                        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
                        while t < nn:
                            res += sq[a0 + t]
                            t += 1
                    stack[sp] = res
                    sp += 1
                else:
                    sp -= 1
                    stack[sp - 1] = stack[sp - 1] + stack[sp]
            nrm = np.sqrt(stack[0])
            den = tiny if nrm < tiny else nrm                   # np.maximum: a NaN stays NaN
            for j in range(d):
                x[i, j] = x[i, j] / den

    @numba.njit(cache=True, error_model="numpy")
    def links_k(n, eu, ev, ef, fams):
        """c3_fast.links_fast's four values for int16 edges (u, v) and int8 families: each non-loop edge in both directions as the
        entry (a, b * fams + family); the entries sorted by (that key, then a stably) in two counting-sort passes, which is the
        sorted order of (a, key); repeats collapsed; b and family split back. Integer work only, so the same arrays. An edge outside
        its pool or its families returns status 1 (the caller takes the original)."""
        m = eu.size
        nl = 0
        for i in range(m):
            u = np.int64(eu[i])
            v = np.int64(ev[i])
            f = np.int64(ef[i])
            if u < 0 or v < 0 or u >= n or v >= n or f < 0 or f >= fams:
                return 1, 0, np.empty(0, np.int16), np.empty(0, np.int16), np.empty(0, np.int8)
            if u != v:
                nl += 1
        nn = 2 * nl
        fm = np.int32(fams)
        a = np.empty(nn, np.int32)
        k = np.empty(nn, np.int32)
        p = 0
        for i in range(m):
            u = np.int32(eu[i])
            v = np.int32(ev[i])
            if u != v:
                f = np.int32(ef[i])
                a[p] = u
                k[p] = v * fm + f
                a[p + nl] = v
                k[p + nl] = u * fm + f
                p += 1
        kk = n * fams
        ck = np.zeros(kk + 1, np.int64)
        for q in range(nn):
            ck[k[q] + 1] += 1
        for t in range(kk):
            ck[t + 1] += ck[t]
        a1 = np.empty(nn, np.int32)
        k1 = np.empty(nn, np.int32)
        for q in range(nn):
            t = k[q]
            pos = ck[t]
            ck[t] = pos + 1
            a1[pos] = a[q]
            k1[pos] = t
        ca = np.zeros(n + 1, np.int64)
        for q in range(nn):
            ca[a1[q] + 1] += 1
        for t in range(n):
            ca[t + 1] += ca[t]
        a2 = np.empty(nn, np.int32)
        k2 = np.empty(nn, np.int32)
        for q in range(nn):
            t = a1[q]
            pos = ca[t]
            ca[t] = pos + 1
            a2[pos] = t
            k2[pos] = k1[q]
        ao = np.empty(nn, np.int16)
        bo = np.empty(nn, np.int16)
        fo = np.empty(nn, np.int8)
        cnt = 0
        for q in range(nn):
            if q == 0 or a2[q] != a2[q - 1] or k2[q] != k2[q - 1]:
                ao[cnt] = a2[q]
                bo[cnt] = k2[q] // fm
                fo[cnt] = k2[q] % fm
                cnt += 1
        return 0, cnt, ao[:cnt].copy(), bo[:cnt].copy(), fo[:cnt].copy()

    _Y.update(zref=zref_segs, nbr=nbr_lists, unit32=unit32, links=links_k)
    return _Y


def seg_zscore_ref_x(x, nq, B, ref, eps=1e-6):
    """lean_screen3.seg_zscore_ref for a batch of B questions (one at every cold forward, 16 in the warm batches) of a float32 CPU
    matrix with no gradient: one numba pass over the questions' segments (their rows contiguous and in question order, as
    zfsvc.build lays them out) whose float32 operations are the original's, in its order (see the kernel). Any other input, a
    batch whose question ids are not sorted or not in 0..B-1 included, takes the original (counted)."""
    torch = C.torch
    n = x.shape[0]
    if (x.dtype != torch.float32 or x.dim() != 2 or x.device.type != "cpu" or n == 0 or n >= 1 << 24
            or torch.is_grad_enabled() or ref.dtype != torch.float32 or ref.shape != (n,)):
        bounds = None
    elif B == 1:
        bounds = np.array([0, n], np.int64)
    elif nq.dtype not in (torch.int32, torch.int64) or nq.shape != (n,) or nq.device.type != "cpu":   # np.arange's int is 32-bit on Windows with numpy 1.x
        bounds = None
    else:
        q = nq.detach().numpy()
        bounds = (np.searchsorted(q, np.arange(B + 1)).astype(np.int64)
                  if 0 <= q[0] and q[-1] < B and bool((q[1:] >= q[:-1]).all()) else None)
    if bounds is None:
        STATS["zref_orig"] += 1
        return _ORIG["seg_zscore_ref"](x, nq, B, ref, eps)
    xn = np.ascontiguousarray(x.detach().numpy())
    out = np.empty_like(xn)
    ykernels()["zref"](xn, np.ascontiguousarray(ref.detach().numpy()), bounds, np.float32(eps), out)
    STATS["zref_x"] += 1
    return torch.from_numpy(out)


def patch_zref(on):
    """lean_screen3.seg_zscore_ref as the forms give it (on) or as the module has it. Both models' encode read it by that name."""
    with X.thread_env_kept():
        import lean_screen3 as S3
    _ORIG.setdefault("seg_zscore_ref", S3.seg_zscore_ref)
    S3.seg_zscore_ref = seg_zscore_ref_x if on else _ORIG["seg_zscore_ref"]


def patch_next(on):
    """zfsvc.next_cold as the forms give it (on) or as the module has it: stage (b) of the MLP calls it by that name."""
    _ORIG.setdefault("next_cold", C.next_cold)
    C.next_cold = next_cold_y if on else _ORIG["next_cold"]


def links_x(n, eu, ev, ef):
    """c3_fast.links_fast for the edges stage (b) hands it (int16 rows, int8 families, one dimension, contiguous): one numba call
    that returns the same four values; any other input, or an edge outside its pool, takes the original (counted)."""
    if (n >= C.C3.ZL.NMAX or eu.dtype != np.int16 or ev.dtype != np.int16 or ef.dtype != np.int8 or eu.ndim != 1
            or eu.size != ev.size or eu.size != ef.size
            or not (eu.flags.c_contiguous and ev.flags.c_contiguous and ef.flags.c_contiguous)):
        STATS["links_orig"] += 1
        return _ORIG["links_fast"](n, eu, ev, ef)
    st, cnt, ao, bo, fo = ykernels()["links"](np.int64(n), eu, ev, ef, np.int64(C.C3.ZL.FAMS))
    if st:
        STATS["links_orig"] += 1
        return _ORIG["links_fast"](n, eu, ev, ef)
    STATS["links_x"] += 1
    return int(cnt), ao, bo, fo


def patch_links(on):
    """c3_fast.links_fast as the forms give it (on) or as the module has it: stage (b) of both models calls it by that name."""
    _ORIG.setdefault("links_fast", C.C3.links_fast)
    C.C3.links_fast = links_x if on else _ORIG["links_fast"]


def patch_forms(on):
    patch_zref(on and ON["zref"])
    patch_next(on and (ON["nbr"] or ON["unit"]))
    patch_links(on and ON["links"])


@contextlib.contextmanager
def variant(name):
    """Development: the forms of A6 in force as named: 'orig' (none of them: the modules' own functions), 'all', or one form."""
    saved = dict(ON)
    try:
        for k in ON:
            ON[k] = name in ("all", k)
        patch_forms(name != "orig")
        yield
    finally:
        ON.update(saved)
        patch_forms(True)


def put_back_ok():
    with X.thread_env_kept():
        import lean_screen3 as S3
    return (S3.seg_zscore_ref is _ORIG["seg_zscore_ref"] and C.next_cold is _ORIG["next_cold"]
            and C.C3.links_fast is _ORIG["links_fast"])


def lists_of_x(top, in_top, link):
    """zfsvc.next_cold's candidate lists of the five anchors from the links' sorted closure (see the kernel): one numba call."""
    _ne, la, lb, lf = link
    c, bt, off = ykernels()["nbr"](la, lb, lf, top, in_top)
    return [(k, int(top[k]), c[off[k]:off[k + 1]], bt[off[k]:off[k + 1]]) for k in range(top.size) if off[k + 1] > off[k]]


_PW = {}


def pw_program(d):
    """numpy's float32 pairwise sum over d contiguous elements as a postfix program (leaf runs and adds), for unit32."""
    if d not in _PW:
        kind, lo, cnt = [], [], []

        def rec(a, n):
            if n <= 128:
                kind.append(0)
                lo.append(a)
                cnt.append(n)
            else:
                n2 = n // 2
                n2 -= n2 % 8
                rec(a, n2)
                rec(a + n2, n - n2)
                kind.append(1)
                lo.append(0)
                cnt.append(0)

        rec(0, d)
        _PW[d] = tuple(np.asarray(v, np.int64) for v in (kind, lo, cnt))
    return _PW[d]


def unit_x(nodes, ids):
    """ltwdiag.unit(nodes.read(ids)): the float16 rows converted by torch (exact, as numpy's), made unit length in one pass whose
    float32 operations are numpy's norm's and division's (see unit32)."""
    x = X.f32_rows(nodes, ids)
    if x.shape[0]:
        ykernels()["unit32"](x, *pw_program(x.shape[1]))
    STATS["unit_x"] += 1
    return x


def read_chunk_x(z, pre, nd, sel, idx):
    """ltwdiag.read_chunk for the lists zfsvc.next_cold reads: the unit rows by unit_x, and nothing next_cold does not read (the
    `free` columns and the identity check, whose result it drops, are not made); the other values are the original's lines."""
    LTW = C.LTW
    sel = np.asarray(sel, np.int64)
    flat = np.concatenate(idx).astype(np.int64) if len(idx) else np.zeros(0, np.int64)
    anode, cnode = z[f"{pre}_anode"][sel], z[f"{pre}_cnode"][flat]
    uq = np.unique(np.r_[anode, cnode])
    V = unit_x(nd, uq)
    xa, Xc = V[np.searchsorted(uq, anode)], V[np.searchsorted(uq, cnode)]
    qv = LTW.unit(z["qe"][z[f"{pre}_q"][sel]].astype(np.float32))
    off = np.r_[0, np.cumsum([x.size for x in idx])].astype(np.int64)
    cxc, crank, cfam = z[f"{pre}_cxc"][flat], z[f"{pre}_crank"][flat], z[f"{pre}_cfam"][flat]
    F = np.empty((flat.size, len(LTW.FEATS)), np.float32)
    rv = np.empty_like(qv)
    for li in range(sel.size):
        s_, e_ = off[li], off[li + 1]
        cv, s, ca, res, pair = LTW.pair_feats(qv[li], xa[li], Xc[s_:e_])
        F[s_:e_] = LTW.feature_rows(cv, s, ca, res, pair, cxc[s_:e_], crank[s_:e_], cfam[s_:e_])
        r = qv[li] - ca * xa[li]
        rv[li] = r / max(float(np.linalg.norm(r)), 1e-12)
    X16 = Xc.astype(np.float16)
    return {"sel": sel, "flat": flat, "off": off, "qv": qv, "xa": xa, "rv": rv, "X16": X16, "F": F,
            "free": [None] * sel.size}, (int(flat.size + sel.size), 0, 0.0)


def next_cold_y(S, j, r, A, link, T_):
    """zfsvc.next_cold with its candidate lists from lists_of_x and its chunk from read_chunk_x; every other line is the original's,
    in its order."""
    LTW, LTMC = C.LTW, C.LTMC
    n, Xr = r["n"], r["X"]
    x6 = Xr[:, S.take].astype(np.float32)
    o = LTW.CH.order_of(x6[:, LTW.J_RRF])
    rk = np.empty(n, np.int64)
    rk[o] = np.arange(1, n + 1)
    top = o[:LTW.TOP]
    in_top = np.zeros(n, bool)
    in_top[top] = True
    lists = lists_of_x(top, in_top, link) if ON["nbr"] else lists_of_original(n, top, in_top, link)
    X5 = np.full((n, LTMC.TOP), LTMC.FILL, np.float32)
    cnt = np.zeros(n, np.int64)
    if lists:
        c_ = np.concatenate([x[2] for x in lists])
        z = {"qe": r["q_emb16"][None, :].astype(np.float16), "T_q": np.zeros(len(lists), np.int64),
             "T_anode": np.asarray([A.pool[x[1]] for x in lists], np.int64), "T_cnode": A.pool[c_],
             "T_c": c_, "T_ct": np.zeros(c_.size, bool), "T_cfam": np.concatenate([x[3] for x in lists]).astype(np.int8),
             "T_crank": rk[c_], "T_cxc": x6[c_], "T_axc": x6[[x[1] for x in lists]],
             "T_off": np.r_[0, np.cumsum([x[2].size for x in lists])].astype(np.int64)}
        sel = np.arange(len(lists))
        idx = LTW.lists_of(z, "T", sel)
        b, _chk = (read_chunk_x if ON["unit"] else LTW.read_chunk)(z, "T", S.cx.nodes, sel, idx)
        its = LTW.items_of(b, z, "T", LTMC.DS_K[S.name])
        m_, mu, sd = S.folds[int(S.hfold[j])]
        for it, sc in zip(its, LTW.score_items(m_, its, mu, sd)):
            rows = it["loc"]
            X5[rows, lists[it["list"]][0]] = LTMC.log_ratio(sc)
            cnt[rows] += 1
    Cn = np.empty((n, LTMC.WIDTH), np.float16)
    Cn[:, :LTMC.TOP] = X5
    Cn[:, LTMC.TOP] = X5.max(1)
    Cn[:, LTMC.TOP + 1] = cnt
    STATS["next_x"] += 1
    return Cn


@contextlib.contextmanager
def installed(out_dir=None):
    """zfsvx's forms (A4) and A6's on top of them in this process, put back on exit (also after an error). The callers assert from
    STATS (both) that each ran."""
    for k in STATS:
        STATS[k] = 0
    with X.installed(out_dir) as P:
        patch_forms(True)
        try:
            yield P
        finally:
            patch_forms(False)
            if not put_back_ok():
                raise SystemExit("the forms were not put back")


# ── the checks (no data) ─────────────────────────────────────────────────────


def selftest_zref(rng, trials):
    """seg_zscore_ref_x against lean_screen3.seg_zscore_ref, byte for byte, on random matrices: constant columns, a segment with no,
    one, two and many reference rows, one row, ties, large and small scales, a column slice of a wider matrix."""
    torch = C.torch
    with X.thread_env_kept():
        import lean_screen3 as S3
    orig = _ORIG.get("seg_zscore_ref", S3.seg_zscore_ref)
    bad = 0
    n_ok = 0
    shapes = [(1, 3), (2, 4), (3, 1), (7, 5), (50, 8), (500, 13), (2000, 46), (4845, 128), (671, 64)]
    with torch.no_grad():
        for _t in range(trials):
            for (n, m) in shapes:
                x = torch.from_numpy(rng.standard_normal((n, m)).astype(np.float32) * np.float32(10.0 ** rng.integers(-3, 4)))
                kind = int(rng.integers(0, 6))
                if kind == 1:
                    x[:, int(rng.integers(0, m))] = float(rng.standard_normal())                 # a constant column
                elif kind == 2:
                    x = torch.round(x * 4) / 4                                                  # many ties
                elif kind == 3 and m > 1:
                    x[:, 0] = 0.0
                    x[:, m - 1] = torch.from_numpy((rng.standard_normal(n) * 1e-7).astype(np.float32))   # sd under eps
                elif kind == 4:
                    x = x.abs()
                ref_n = [0, 1, 2, max(n // 3, 0), n][int(rng.integers(0, 5))]
                ref = torch.zeros(n)
                ref[torch.from_numpy(rng.permutation(n)[:min(ref_n, n)])] = 1.0
                nq = torch.zeros(n, dtype=torch.long)
                want = orig(x, nq, 1, ref)
                for src in (x, torch.cat([x, x[:, :1]], 1)[:, :m]):                             # a contiguous matrix, a slice
                    got = seg_zscore_ref_x(src, nq, 1, ref)
                    n_ok += 1
                    if got.dtype != want.dtype or got.shape != want.shape or got.numpy().tobytes() != want.numpy().tobytes():
                        bad += 1
                        if bad <= 3:
                            d = (got.numpy() != want.numpy()).sum()
                            log(f"zref DIFFERS: shape {(n, m)} kind {kind} ref rows {int(ref.sum())}: {d} values")
        # batches of several questions (the warm batches of 16): each question's rows contiguous and in question order, some
        # questions short or empty, the ids 32- or 64-bit; the form must take them itself, not hand them to the original
        o0 = STATS["zref_orig"]
        n_batch = 0
        for _t in range(trials):
            for B in (2, 3, 16, 16):
                m = int(rng.choice([1, 5, 46, 128]))
                ns = [int(rng.choice([0, 1, 2, 3, 40, 300, 700])) for _ in range(B)]
                n = sum(ns)
                if n == 0:
                    continue
                x = torch.from_numpy(rng.standard_normal((n, m)).astype(np.float32) * np.float32(10.0 ** rng.integers(-3, 4)))
                if int(rng.integers(0, 3)) == 1:
                    x = torch.round(x * 4) / 4
                ref = torch.from_numpy((rng.random(n) < float(rng.choice([0.0, 0.01, 0.3, 1.0]))).astype(np.float32))
                nq = torch.from_numpy(np.repeat(np.arange(B), ns).astype(np.int32 if rng.integers(0, 2) else np.int64))
                want = orig(x, nq, B, ref)
                got = seg_zscore_ref_x(x, nq, B, ref)
                n_ok += 1
                n_batch += 1
                if got.dtype != want.dtype or got.shape != want.shape or got.numpy().tobytes() != want.numpy().tobytes():
                    bad += 1
                    if bad <= 3:
                        log(f"zref DIFFERS: batch of {B} questions {ns}, {m} columns: {(got.numpy() != want.numpy()).sum()} values")
        if STATS["zref_orig"] != o0:
            bad += 1
            log(f"zref: {STATS['zref_orig'] - o0} of {n_batch} batches went to the original")
        # the inputs the form hands to the original: question ids not sorted, a float64 matrix, an input with a gradient
        STATS["zref_orig"] = 0
        x = torch.randn(9, 3)
        ref = torch.ones(9)
        seg_zscore_ref_x(x, torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0]), 2, ref)
        seg_zscore_ref_x(x.double(), torch.zeros(9, dtype=torch.long), 1, ref)
        with torch.enable_grad():
            seg_zscore_ref_x(x, torch.zeros(9, dtype=torch.long), 1, ref)
    if STATS["zref_orig"] != 3:
        bad += 1
        log(f"zref: the original was taken {STATS['zref_orig']} times of 3")
    log(f"zref: {n_ok} comparisons, {bad} differ")
    return bad


def lists_of_original(n, top, in_top, link):
    """zfsvc.next_cold's candidate lists of the five anchors, as it builds them."""
    _ne, la, lb, lf = link
    u, v, f = (np.asarray(x, np.int64) for x in (la, lb, lf))
    out = []
    for k, a in enumerate(top):
        a = int(a)
        bits = C.LTW.fam_bits(n, a, u, v, f)
        cand = np.flatnonzero((bits > 0) & ~in_top)
        if cand.size:
            out.append((k, a, cand, bits[cand]))
    return out


def same_lists(a, b):
    return len(a) == len(b) and all(x[0] == y[0] and x[1] == y[1] and X.same_bytes(x[2], y[2]) and X.same_bytes(x[3], y[3])
                                    for x, y in zip(a, b))


def selftest_nbr(rng, trials):
    """lists_of_x against the original's per-anchor fam_bits scans, on links_fast's own output for random multigraphs (loops,
    repeated entries, several families on one pair, isolated anchors, anchors adjacent to each other)."""
    bad = n_ok = 0
    for _t in range(trials):
        for n in (60, 400, 3000, 5800):
            m = int(rng.integers(0, 12 * n))
            hub = rng.integers(0, n, 3)
            eu = np.where(rng.random(m) < 0.3, rng.choice(hub, m), rng.integers(0, n, m)).astype(np.int64)
            ev = rng.integers(0, n, m).astype(np.int64)
            ef = rng.integers(0, 3, m).astype(np.int64)
            link = _ORIG.get("links_fast", C.C3.links_fast)(n, eu, ev, ef)
            top = rng.permutation(n)[:5].astype(np.int64)
            if rng.random() < 0.5 and m:
                top[:2] = eu[:2]                                                    # anchors that have links, maybe equal
                top = np.unique(top)[:5]
                top = np.r_[top, rng.permutation(n)[:5 - top.size]].astype(np.int64)
                if np.unique(top).size < top.size:
                    continue
            in_top = np.zeros(n, bool)
            in_top[top] = True
            n_ok += 1
            if not same_lists(lists_of_original(n, top, in_top, link), lists_of_x(top, in_top, link)):
                bad += 1
                log(f"nbr DIFFERS: n {n}, {m} edges")
    log(f"nbr: {n_ok} comparisons, {bad} differ")
    return bad


def selftest_links(rng, trials):
    """links_x against c3_fast.links_fast, byte for byte (the count, the three arrays and their dtypes), on random multigraphs of the
    dtypes stage (b) hands it: loops, repeated and mirrored edges, hubs, one family or three, no edge, one row; and the inputs that
    take the original (another dtype, an edge outside its pool, a pool above int16) answer as it does."""
    orig = _ORIG.get("links_fast", C.C3.links_fast)
    bad = n_ok = 0
    for _t in range(trials):
        for n in (1, 2, 7, 60, 400, 3000, 5800, 32767):
            for m in (0, 1, 5, int(rng.integers(0, 4 * n + 1)), int(rng.integers(0, 14 * n + 1))):
                hub = rng.integers(0, n, 3)
                eu = np.where(rng.random(m) < 0.3, rng.choice(hub, m), rng.integers(0, n, m)).astype(np.int16)
                ev = rng.integers(0, n, m).astype(np.int16)
                ef = np.sort(rng.integers(0, 3 if rng.random() < 0.7 else 1, m)).astype(np.int8)       # family by family
                if rng.random() < 0.3 and m:
                    ev[: m // 3] = eu[: m // 3]                                                        # loops
                    eu[m // 2:], ev[m // 2:] = ev[: m - m // 2].copy(), eu[: m - m // 2].copy()          # mirrored and repeated
                want, got = orig(n, eu, ev, ef), links_x(n, eu, ev, ef)
                n_ok += 1
                if (got[0] != want[0] or any(g.dtype != w.dtype or g.shape != w.shape or g.tobytes() != w.tobytes()
                                             for g, w in zip(got[1:], want[1:]))):
                    bad += 1
                    log(f"links DIFFERS: n {n}, {m} edges")
    STATS["links_orig"] = 0
    eu, ev, ef = (np.asarray([0, 1, 2, 2], np.int16), np.asarray([1, 2, 0, 2], np.int16), np.asarray([0, 1, 2, 0], np.int8))
    for args in ((5, eu.astype(np.int64), ev, ef), (5, eu, ev, ef.astype(np.int64)), (2, eu, ev, ef), (5, eu[::2], ev[::2], ef[::2]),
                 (5, eu, ev, np.asarray([0, 1, 3, 0], np.int8)), (1 << 15, eu, ev, ef)):
        try:
            want = orig(*args)
        except SystemExit:
            want = "exit"
        try:
            got = links_x(*args)
        except SystemExit:
            got = "exit"
        n_ok += 1
        same = (got == "exit") if want == "exit" else (got != "exit" and got[0] == want[0] and all(
            g.dtype == w.dtype and g.tobytes() == w.tobytes() for g, w in zip(got[1:], want[1:])))
        if not same:
            bad += 1
            log("links DIFFERS: an input for the original")
    if STATS["links_orig"] < 5:
        bad += 1
        log(f"links: the original was taken {STATS['links_orig']} times of at least 5")
    log(f"links: {n_ok} comparisons, {bad} differ")
    return bad


def selftest_unit(rng, trials):
    """unit_x against ltwdiag.unit(nodes.read(ids)), byte for byte, on random float16 stores: widths that exercise numpy's pairwise
    sum (under 8, one block, the leaf size 128 and one over it, several splits, 1,536), zero rows, rows of several scales, unsorted
    and repeated ids, no ids."""
    LTW = C.LTW
    bad = n_ok = 0
    for _t in range(trials):
        for d in (1, 3, 7, 8, 9, 100, 127, 128, 129, 255, 256, 257, 1000, 1536, 2049):
            m = (rng.standard_normal((300, d)) * 10.0 ** rng.integers(-3, 4, (300, 1))).astype(np.float16)
            m[rng.integers(0, 300, 5)] = 0
            m[rng.integers(0, 300, 5), :d // 2] = 0
            nd = X.FakeNodes(m)
            for ids in (np.arange(300), rng.permutation(300)[:57], rng.integers(0, 300, 40), np.zeros(0, np.int64), np.asarray([4])):
                got, want = unit_x(nd, ids), LTW.unit(nd.read(ids))
                n_ok += 1
                if got.dtype != want.dtype or got.shape != want.shape or got.tobytes() != want.tobytes():
                    bad += 1
                    if bad <= 3:
                        log(f"unit DIFFERS: width {d}, {len(ids)} rows: {int((got != want).sum())} values")
    log(f"unit: {n_ok} comparisons, {bad} differ")
    return bad


def selftest_chunk(rng, trials):
    """read_chunk_x against ltwdiag.read_chunk on random lists over a random float16 store: every array it hands on byte for byte
    and the count of rows it read (lists of one to five anchors, candidates that repeat or are the anchor's own node)."""
    LTW = C.LTW
    bad = n_ok = 0
    nst, d = 2500, 1536
    for _t in range(trials):
        m = (rng.standard_normal((nst, d)) * 10.0 ** rng.integers(-2, 3, (nst, 1))).astype(np.float16)
        m[rng.integers(0, nst, 4)] = 0
        nd = X.FakeNodes(m)
        for n_lists in (1, 2, 5):
            sizes = rng.integers(1, 300, n_lists)
            tot = int(sizes.sum())
            z = {"qe": rng.standard_normal((3, d)).astype(np.float16), "T_q": rng.integers(0, 3, n_lists),
                 "T_anode": rng.integers(0, nst, n_lists), "T_cnode": rng.integers(0, nst, tot),
                 "T_c": rng.integers(0, 1000, tot), "T_ct": np.zeros(tot, bool), "T_cfam": rng.integers(1, 8, tot).astype(np.int8),
                 "T_crank": rng.integers(1, 5000, tot), "T_cxc": rng.random((tot, 6)).astype(np.float32),
                 "T_axc": rng.random((n_lists, 6)).astype(np.float32), "T_off": np.r_[0, np.cumsum(sizes)].astype(np.int64)}
            sel = np.arange(n_lists)
            idx = LTW.lists_of(z, "T", sel)
            want, wchk = LTW.read_chunk(z, "T", nd, sel, idx)
            got, gchk = read_chunk_x(z, "T", nd, sel, idx)
            n_ok += 1
            keys = ("sel", "flat", "off", "qv", "xa", "rv", "X16", "F")
            if gchk[0] != wchk[0] or any(got[k].dtype != want[k].dtype or got[k].shape != want[k].shape
                                         or got[k].tobytes() != want[k].tobytes() for k in keys):
                bad += 1
                log(f"chunk DIFFERS: {n_lists} lists, {tot} candidates: " + ", ".join(k for k in keys if got[k].tobytes() != want[k].tobytes()))
    log(f"chunk: {n_ok} comparisons, {bad} differ")
    return bad


def selftest(a):
    C._load()
    log(f"threads {C.set_threads(C.BLAS)}")
    rng = np.random.default_rng(11)
    bad = {}
    with installed():
        bad["zref"] = selftest_zref(rng, int(a.trials))
        bad["nbr"] = selftest_nbr(rng, int(a.trials))
        bad["links"] = selftest_links(rng, max(int(a.trials) // 2, 1))
        bad["unit"] = selftest_unit(rng, max(int(a.trials) // 4, 1))
        bad["chunk"] = selftest_chunk(rng, max(int(a.trials) // 2, 1))
    if not put_back_ok() or X._zg6r().g6r_step is not X._ORIG["g6r_step"]:
        raise SystemExit("the forms were not put back")
    log("SELFTEST " + ("PASS" if not any(bad.values()) else f"FAIL {bad}"))
    return 0 if not any(bad.values()) else 1


def zbench(a):
    """Development, no data: seg_zscore_ref on random float32 matrices of a pool's shape (one thread): the original's and the form's
    median ms, and equal bytes."""
    C._load()
    torch = C.torch
    log(f"threads {C.set_threads(C.BLAS)}")
    with X.thread_env_kept():
        import lean_screen3 as S3
    orig = S3.seg_zscore_ref
    rng = np.random.default_rng(3)
    with torch.no_grad():
        for (n, m) in ((4845, 128), (4845, 16), (671, 128), (671, 16), (1123, 128)):
            x = torch.from_numpy(rng.standard_normal((n, m)).astype(np.float32))
            ref = (torch.from_numpy(rng.random(n).astype(np.float32)) > 0.8).float()
            nq = torch.zeros(n, dtype=torch.long)
            _ORIG.setdefault("seg_zscore_ref", orig)
            seg_zscore_ref_x(x, nq, 1, ref)
            ts = {"o": [], "x": []}
            outs = {}
            for rep in range(15):
                for f in (("o", "x") if rep % 2 == 0 else ("x", "o")):
                    c0 = clock()
                    outs[f] = orig(x, nq, 1, ref) if f == "o" else seg_zscore_ref_x(x, nq, 1, ref)
                    ts[f].append(clock() - c0)
            log(f"zref {n}x{m}: original {p50(ts['o']):.2f} ms, form {p50(ts['x']):.2f} ms, x{p50(ts['o']) / p50(ts['x']):.1f}; "
                f"equal {outs['x'].numpy().tobytes() == outs['o'].numpy().tobytes()}")
    LTW = C.LTW
    nst, d = 6000, 1536
    m16 = (rng.standard_normal((nst, d)) * 3).astype(np.float16)
    nd = X.FakeNodes(m16)
    for k in (30, 120, 400, 1500):
        ids = np.unique(rng.integers(0, nst, k))
        ts = {"o": [], "x": []}
        for rep in range(15):
            for f in (("o", "x") if rep % 2 == 0 else ("x", "o")):
                c0 = clock()
                LTW.unit(nd.read(ids)) if f == "o" else unit_x(nd, ids)
                ts[f].append(clock() - c0)
        log(f"unit {ids.size} rows: original {p50(ts['o']):.3f} ms, form {p50(ts['x']):.3f} ms, x{p50(ts['o']) / p50(ts['x']):.1f}")
    for sz in (20, 100, 300):
        sizes = np.full(5, sz)
        tot = int(sizes.sum())
        z = {"qe": rng.standard_normal((1, d)).astype(np.float16), "T_q": np.zeros(5, np.int64),
             "T_anode": rng.integers(0, nst, 5), "T_cnode": rng.integers(0, nst, tot), "T_c": rng.integers(0, 1000, tot),
             "T_ct": np.zeros(tot, bool), "T_cfam": rng.integers(1, 8, tot).astype(np.int8), "T_crank": rng.integers(1, 5000, tot),
             "T_cxc": rng.random((tot, 6)).astype(np.float32), "T_axc": rng.random((5, 6)).astype(np.float32),
             "T_off": np.r_[0, np.cumsum(sizes)].astype(np.int64)}
        sel = np.arange(5)
        idx = LTW.lists_of(z, "T", sel)
        ts = {"o": [], "x": []}
        for rep in range(15):
            for f in (("o", "x") if rep % 2 == 0 else ("x", "o")):
                c0 = clock()
                (LTW.read_chunk if f == "o" else read_chunk_x)(z, "T", nd, sel, idx)
                ts[f].append(clock() - c0)
        log(f"read_chunk 5 lists x {sz}: original {p50(ts['o']):.3f} ms, form {p50(ts['x']):.3f} ms, x{p50(ts['o']) / p50(ts['x']):.1f}")
    orig_links = _ORIG.get("links_fast", C.C3.links_fast)
    for n, m in ((4845, 70000), (671, 2000), (1123, 3200)):
        eu, ev = (rng.integers(0, n, m).astype(np.int16) for _ in range(2))
        ef = np.sort(rng.integers(0, 3, m)).astype(np.int8)
        links_x(n, eu, ev, ef)
        ts = {"o": [], "x": []}
        outs = {}
        for rep in range(15):
            for f in (("o", "x") if rep % 2 == 0 else ("x", "o")):
                c0 = clock()
                outs[f] = orig_links(n, eu, ev, ef) if f == "o" else links_x(n, eu, ev, ef)
                ts[f].append(clock() - c0)
        log(f"links n={n}, {m} edges: original {p50(ts['o']):.3f} ms, form {p50(ts['x']):.3f} ms, x{p50(ts['o']) / p50(ts['x']):.1f}; "
            f"equal {all(a.tobytes() == b.tobytes() for a, b in zip(outs['o'][1:], outs['x'][1:])) and outs['o'][0] == outs['x'][0]}")
    return 0


# ── the profile: where stage (b), NEXT and the forward spend their time (development) ───────────────────────────────


def compile_laps(S, j, A):
    """C11.compile_serial's body with c2_fast.compile_mlp's own laps (retrieval = the row gather, the dense cosines and the
    list ranks; edges; topology; D = the seed products; depth_basis). Returns (comp, laps, seconds)."""
    fc = S.bn.su.fc
    saved = (fc.K, fc._K_small, fc.parallel)
    fc.K, fc._K_small = S.bn.k_ser
    fc.parallel = False
    laps = {}
    c0 = clock()
    try:
        comp = C.C11.C2.compile_mlp(fc, C.C10.qinputs(S.su, j), A.pool, A.seeds, embeddings=None, timings=laps)
    finally:
        fc.K, fc._K_small, fc.parallel = saved
    return comp, laps, clock() - c0


def rows_laps(S, sh, T):
    """C10.lean_rows10 line for line, timed in its parts: edges_of; the seed rows and the float16 scalars; the WALK block; the
    WALKF block; the decoder; seed_dists10; the assembly of X. Returns what lean_rows10 returns."""
    C10, su, dec = C.C10, S.su, S.dec
    c0 = clock()
    comp = sh["comp"]
    n = int(comp.pool.size)
    eu, ev, ef, efw, ebw = C10.C1.edges_of(comp)
    c1 = clock()
    sl, sb = C10.seed_rows(np.asarray(comp.seeds_local, np.int64), sh["bucket"])
    x16 = np.asarray(comp.scalars[:, su.columns], dtype=np.float16)
    c2 = clock()
    walk = C10.C2.walk_block(n, sl, sb, eu, ev, ef, efw, ebw)
    c3 = clock()
    walkf = C10.C2.walkf_block(n, sl, sb, eu, ev, ef, efw, ebw)
    c4 = clock()
    P = dec(sh["pool"])
    c5 = clock()
    seed, dists = C10.seed_dists10(n, P, sl, sb, x16[:, su.c_rrf_x], eu, ev, ef)
    c6 = clock()
    Xm = np.empty((n, su.W), np.float16)
    Xm[:, :su.span[C10.LC.XC_BLOCKS[-1]][1]] = x16[:, su.xcols]
    for b, arr in (("WALK", walk), ("WALKF", walkf), ("SEED", seed), ("DISTS", dists)):
        a, e = su.span[b]
        Xm[:, a:e] = arr.astype(np.float16)
    c7 = clock()
    T.update(r_edges=c1 - c0, r_scalars=c2 - c1, r_walk=c3 - c2, r_walkf=c4 - c3, r_decode=c5 - c4, r_seed=c6 - c5,
             r_assemble=c7 - c6)
    return {"n": n, "X": Xm, "P": P, "q_emb16": np.asarray(sh["qemb"], np.float32).astype(np.float16),
            "e": (eu, ev, ef, efw, ebw), "sl": sl, "sb": sb}


def links_laps(r, T):
    """c3_fast.links_fast in its parts: the int64 copies and the doubled arrays; bucket_unique; the narrowed outputs."""
    C3 = C.C3
    eu, ev, ef = r["e"][0], r["e"][1], r["e"][2]
    n = r["n"]
    c0 = clock()
    u, v, f = eu.astype(np.int64), ev.astype(np.int64), ef.astype(np.int64)
    loop = u == v
    a, b, ff = np.r_[u[~loop], v[~loop]], np.r_[v[~loop], u[~loop]], np.r_[f[~loop], f[~loop]]
    k2 = b * C3.ZL.FAMS + ff
    c1 = clock()
    ao, ko = C3.bucket_unique(n, a, k2)
    c2 = clock()
    out = (int(ao.size), ao.astype(np.int16), (ko // C3.ZL.FAMS).astype(np.int16), (ko % C3.ZL.FAMS).astype(np.int8))
    c3 = clock()
    T.update(l_prep=c1 - c0, l_unique=c2 - c1, l_narrow=c3 - c2)
    return out


def read_chunk_laps(z, pre, nd, sel, idx, T):
    """ltwdiag.read_chunk line for line, timed in its parts: the list indexes; nd.read; unit; the split; the per-list pair
    features and the identity check; the float16 rows. Returns what read_chunk returns."""
    LTW, CH = C.LTW, C.LTW.CH
    c0 = clock()
    sel = np.asarray(sel, np.int64)
    flat = np.concatenate(idx).astype(np.int64) if len(idx) else np.zeros(0, np.int64)
    anode, cnode = z[f"{pre}_anode"][sel], z[f"{pre}_cnode"][flat]
    uq = np.unique(np.r_[anode, cnode])
    c1 = clock()
    raw = nd.read(uq)
    c2 = clock()
    V = LTW.unit(raw)
    c3 = clock()
    xa, Xc = V[np.searchsorted(uq, anode)], V[np.searchsorted(uq, cnode)]
    qv = LTW.unit(z["qe"][z[f"{pre}_q"][sel]].astype(np.float32))
    off = np.r_[0, np.cumsum([x.size for x in idx])].astype(np.int64)
    cxc, crank, cfam, axc = z[f"{pre}_cxc"][flat], z[f"{pre}_crank"][flat], z[f"{pre}_cfam"][flat], z[f"{pre}_axc"][sel]
    F = np.empty((flat.size, len(LTW.FEATS)), np.float32)
    rv = np.empty_like(qv)
    free = []
    bad, worst = 0, 0.0
    c4 = clock()
    for li in range(sel.size):
        s_, e_ = off[li], off[li + 1]
        cv, s, ca, res, pair = LTW.pair_feats(qv[li], xa[li], Xc[s_:e_])
        F[s_:e_] = LTW.feature_rows(cv, s, ca, res, pair, cxc[s_:e_], crank[s_:e_], cfam[s_:e_])
        free.append({"rrf": cxc[s_:e_, LTW.J_RRF].astype(np.float64), "cos": cv, "anc": s, "res": res, "pair": pair})
        r = qv[li] - ca * xa[li]
        rv[li] = r / max(float(np.linalg.norm(r)), 1e-12)
        dif = np.abs(np.r_[cv - cxc[s_:e_, LTW.J_DC], ca - axc[li, LTW.J_DC]])
        bad += int((dif > CH.TOL_COS).sum())
        worst = max(worst, float(dif.max()))
    c5 = clock()
    X16 = Xc.astype(np.float16)
    c6 = clock()
    T.update(rc_index=c1 - c0, rc_read=c2 - c1, rc_unit=c3 - c2, rc_split=c4 - c3, rc_lists=c5 - c4, rc_x16=c6 - c5,
             rc_rows=int(uq.size), rc_cands=int(flat.size))
    return {"sel": sel, "flat": flat, "off": off, "qv": qv, "xa": xa, "rv": rv, "X16": X16, "F": F, "free": free}, \
        (int(flat.size + sel.size), bad, worst)


def next_laps(S, j, r, A, link, T):
    """zfsvc.next_cold line for line, timed in its parts. Returns the NEXT columns (the caller compares them with next_cold's)."""
    LTW, LTMC, torch = C.LTW, C.LTMC, C.torch
    c0 = clock()
    n, Xm = r["n"], r["X"]
    _ne, la, lb, lf = link
    u, v, f = (np.asarray(x, np.int64) for x in (la, lb, lf))
    x6 = Xm[:, S.take].astype(np.float32)
    c1 = clock()
    o = LTW.CH.order_of(x6[:, LTW.J_RRF])
    rk = np.empty(n, np.int64)
    rk[o] = np.arange(1, n + 1)
    top = o[:LTW.TOP]
    in_top = np.zeros(n, bool)
    in_top[top] = True
    c2 = clock()
    lists = []
    t_bits = t_cand = 0.0
    for k, a in enumerate(top):
        a = int(a)
        s0 = clock()
        bits = LTW.fam_bits(n, a, u, v, f)
        s1 = clock()
        cand = np.flatnonzero((bits > 0) & ~in_top)
        if cand.size:
            lists.append((k, a, cand, bits[cand]))
        t_bits += s1 - s0
        t_cand += clock() - s1
    X5 = np.full((n, LTMC.TOP), LTMC.FILL, np.float32)
    cnt = np.zeros(n, np.int64)
    t_read = t_items = t_collate = t_model = t_fill = 0.0
    if lists:
        c_ = np.concatenate([x[2] for x in lists])
        z = {"qe": r["q_emb16"][None, :].astype(np.float16), "T_q": np.zeros(len(lists), np.int64),
             "T_anode": np.asarray([A.pool[x[1]] for x in lists], np.int64), "T_cnode": A.pool[c_],
             "T_c": c_, "T_ct": np.zeros(c_.size, bool), "T_cfam": np.concatenate([x[3] for x in lists]).astype(np.int8),
             "T_crank": rk[c_], "T_cxc": x6[c_], "T_axc": x6[[x[1] for x in lists]],
             "T_off": np.r_[0, np.cumsum([x[2].size for x in lists])].astype(np.int64)}
        sel = np.arange(len(lists))
        idx = LTW.lists_of(z, "T", sel)
        d0 = clock()
        b, _chk = read_chunk_laps(z, "T", S.cx.nodes, sel, idx, T)
        d1 = clock()
        its = LTW.items_of(b, z, "T", LTMC.DS_K[S.name])
        d2 = clock()
        m_, mu, sd = S.folds[int(S.hfold[j])]
        scores = []
        with torch.no_grad():
            for bb in range(0, len(its), LTW.BATCH):
                part = its[bb:bb + LTW.BATCH]
                e0 = clock()
                args, _mask, _pos = LTW.collate(part, mu, sd)
                e1 = clock()
                s = m_(*args).numpy().astype(np.float64)
                e2 = clock()
                scores += [s[i, :it["F"].shape[0]] for i, it in enumerate(part)]
                t_collate += e1 - e0
                t_model += e2 - e1
        f0 = clock()
        for it, sc in zip(its, scores):
            rows = it["loc"]
            X5[rows, lists[it["list"]][0]] = LTMC.log_ratio(sc)
            cnt[rows] += 1
        t_fill = clock() - f0
        t_read, t_items = d1 - d0, d2 - d1
        T["n_lists"] = len(lists)
        T["n_args"] = len(its)
    c4 = clock()
    Cn = np.empty((n, LTMC.WIDTH), np.float16)
    Cn[:, :LTMC.TOP] = X5
    Cn[:, LTMC.TOP] = X5.max(1)
    Cn[:, LTMC.TOP + 1] = cnt
    c5 = clock()
    T.update(n_prep=c1 - c0, n_order=c2 - c1, n_bits=t_bits, n_cand=t_cand, n_read_chunk=t_read, n_items=t_items,
             n_collate=t_collate, n_model=t_model, n_fill=t_fill, n_pack=c5 - c4, n_total=c5 - c0)
    return Cn


class SegZ:
    """Times lean_screen3.seg_zscore_ref (every block's z-scores, in both models' encode) while installed."""

    def __init__(self):
        self.t, self.n, self.orig, self.S3 = 0.0, 0, None, None

    def __enter__(self):
        with X.thread_env_kept():
            import lean_screen3 as S3
        self.S3 = S3
        self.orig = self.S3.seg_zscore_ref

        def timed(*a, **k):
            t0 = clock()
            out = self.orig(*a, **k)
            self.t += clock() - t0
            self.n += 1
            return out

        self.S3.seg_zscore_ref = timed
        return self

    def __exit__(self, *a):
        self.S3.seg_zscore_ref = self.orig

    def reset(self):
        self.t, self.n = 0.0, 0


def profile(a):
    C._load()
    log(f"pin: {X.pin_dev(a.lp)}")
    nq = int(a.questions)
    torch, KEYS = C.torch, C.KEYS
    seg = {}

    def add(T, key=None):
        for k, v in T.items():
            if isinstance(v, (int, float)):
                seg.setdefault(k, []).append(float(v))

    with X.installed():
        S = C.setup(a.dataset, nq)
        C.prepass(S, range(nq + 1))
        S.threads = C.set_threads(C.BLAS)
        log(f"threads {S.threads}")
        for key in KEYS:                                                    # the JIT question, both models
            A = C.stage_a(S, nq, {})
            C.fill_codes(S, nq, A.pool)
            C.run_model(S, key, nq, A, {})
        for key in KEYS:
            log(X.model_line(S, key))
        gc.collect()
        eq_next = 0
        sizes = []
        with SegZ() as sz:
            for j in range(nq):
                A = C.stage_a(S, j, {})
                C.fill_codes(S, j, A.pool)
                T = {}
                comp, laps, tc = compile_laps(S, j, A)
                T.update({f"c_{k}": v for k, v in laps.items()})
                T["compile"] = tc
                sh = {"pool": A.pool, "seeds": A.seeds, "bucket": A.bucket, "comp": comp,
                      "qemb": np.asarray(S.cx.prep.qemb[j])}
                c0 = clock()
                r = rows_laps(S, sh, T)
                T["rows"] = clock() - c0
                c0 = clock()
                link = links_laps(r, T)
                T["links"] = clock() - c0
                want = C.next_cold(S, j, r, A, link, {})
                Cn = want
                try:
                    c0 = clock()
                    Cn = next_laps(S, j, r, A, link, T)
                    T["next"] = clock() - c0
                    eq_next += X.same_bytes(Cn, want)
                except Exception as e:  # noqa: BLE001
                    log(f"q{j} next_laps failed (the profile goes on): {type(e).__name__}: {str(e)[:300]}")
                    Cn = want
                pack = {"n": r["n"], "X": r["X"], "P": r["P"], "q16": r["q_emb16"], "kt": A.kt, "ne": link[0], "lu": link[1],
                        "lv": link[2], "lf": link[3]}
                feats = {}
                for key in KEYS:
                    pk = dict(pack, nxt=Cn) if key == "mlp" else pack
                    c0 = clock()
                    feats[key] = C.build(S, key, [pk])
                    T[f"assemble_{key}"] = clock() - c0
                sizes.append((int(A.pool.size), int(link[0])))
                got = {}
                for key in KEYS:
                    ft, nqt, base_z = feats[key]
                    sz.reset()
                    c0 = clock()
                    sc = C.forward(S, key, ft, nqt, base_z, 1)
                    torch.topk(sc, min(C.TOP, int(sc.numel())))
                    T[f"forward_{key}"] = clock() - c0
                    T[f"segz_{key}"] = sz.t
                    T[f"segz_calls_{key}"] = sz.n
                    got[key] = sc.numpy().copy()
                    if j < int(a.prof):
                        X.forward_profile(S, key, j, int(A.pool.size), ft, nqt, base_z, got[key])
                add(T)
                if j < 3 or j % 10 == 0:
                    log(f"q{j} n={A.pool.size} edges={link[0]} compile {T['compile'] * 1000:.1f} rows {T['rows'] * 1000:.1f} "
                        f"links {T['links'] * 1000:.1f} next {T.get('next', 0) * 1000:.1f} mlp fwd {T['forward_mlp'] * 1000:.1f} "
                        f"(z {T['segz_mlp'] * 1000:.1f}) gnn fwd {T['forward_gnn'] * 1000:.1f} (z {T['segz_gnn'] * 1000:.1f})")
        if X.STATS["setup_fast"] != 1 or X.STATS["lean_rule_x"] < nq or X.STATS["g6r_step_x"] < nq * X._zg6r().LAYERS:
            raise SystemExit(f"the forms did not run: {X.STATS}")
    sz_ = np.array(sizes)
    log(f"pool rows p50 {int(np.median(sz_[:, 0]))} (max {int(sz_[:, 0].max())}), link entries p50 {int(np.median(sz_[:, 1]))} "
        f"(max {int(sz_[:, 1].max())}); NEXT equals next_cold's on {eq_next} of {nq}")
    log("segment p50 / mean ms (the profile's own timers; development, not measurements)")
    for k in sorted(seg):
        if k in ("n_lists", "n_args", "rc_rows", "rc_cands") or k.startswith("segz_calls"):
            log(f"  {k:<18} p50 {float(np.median(seg[k])):9.1f}  mean {float(np.mean(seg[k])):9.1f}  (a count)")
        else:
            log(f"  {k:<18} p50 {p50(seg[k]):9.2f}  mean {mean(seg[k]):9.2f}")
    log("PROFILE_JSON " + json.dumps({"dataset": a.dataset, "questions": nq, "blas": C.BLAS, "eq_next": int(eq_next),
                                      "sizes": sizes, "seg": {k: [float(np.median(v)), float(np.mean(v))] for k, v in seg.items()}}))
    log("PROFILE " + ("PASS" if eq_next == nq else "FAIL"))
    return 0 if eq_next == nq else 1


# ── the development checks ───────────────────────────────────────────────────

VARIANTS = ("orig", "all", "zref", "nbr", "unit", "links")
SEGMENTS = ("compile", "edges", "lean", "links", "next", "assemble", "forward", "b_total", "total")


def same_tree(a, b):
    """Byte-for-byte equality of two feature values (tensors, arrays, tuples, dicts, None)."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, dict):
        return set(a) == set(b) and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    if C.torch.is_tensor(a):
        a, b = a.detach().numpy(), b.detach().numpy()
    return X.same_bytes(a, b)


def same_run(o, y):
    """Two zfsvc.run_model results (scores, pack, (features, question index, base z)) byte for byte: the names of what differs."""
    bad = []
    if not X.same_bytes(o[0], y[0]):
        bad.append("scores")
    for k in o[1]:
        if k not in y[1] or not (o[1][k] == y[1][k] if np.isscalar(o[1][k]) else X.same_bytes(o[1][k], y[1][k])):
            bad.append(f"pack.{k}")
    if not same_tree(o[2], y[2]):
        bad.append("features")
    return bad


def forward_ops(S, key, j, n, feats, nqt, base_z):
    """torch's profiler on one model's forward with A6's forms in force (the ops by self time). A development print: an error in it
    is logged and the checks go on."""
    torch = C.torch
    from torch.profiler import ProfilerActivity, profile
    try:
        with variant("all"), torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as pr:
            C.forward(S, key, feats, nqt, base_z, 1)
        X.log_prof(pr, f"{key} forward with A6's forms, q{j} (n={n})")
    except Exception as e:  # noqa: BLE001
        log(f"q{j} {key} forward profile failed (the checks go on): {type(e).__name__}: {str(e)[:300]}")


def forms_ran(nq):
    """True when every form ran (both modules' counters): the callers stop if not."""
    xs, zl = X.STATS, X._zg6r().LAYERS
    return (xs["setup_fast"] == 1 and xs["lean_rule_x"] >= nq and xs["g6r_step_x"] >= nq * zl and xs["g6r_step_orig"] == 0
            and STATS["zref_x"] >= nq and STATS["next_x"] >= nq and STATS["unit_x"] > 0 and STATS["zref_orig"] == 0
            and STATS["links_x"] >= nq and STATS["links_orig"] == 0)


def devy(a):
    """A2's development checks (zfsvc.dev_checks: the pool, the KT columns and I against PKO's rule, the features against the carve's,
    the scores against the filed reads within 1e-4 with the same top five, the float16 NEXT flips counted) with A4's and A6's forms
    installed, and beside them, per question and model, stage (b) and the forward run without A6's forms, with all of them and with
    each alone, in a rotating order: every output (scores, pack, features) byte for byte equal to the run without them, and each
    segment's median. Development numbers: no timed pass."""
    C._load()
    log(f"pin: {X.pin_dev(a.lp)}")
    nq = int(a.questions)
    KEYS = C.KEYS
    with installed():
        S = C.setup(a.dataset, nq)
        C.prepass(S, range(nq + 1))
        S.threads = C.set_threads(C.BLAS)
        log(f"threads {S.threads}")
        for key in KEYS:                                                    # the JIT question: every variant, both models
            A = C.stage_a(S, nq, {})
            C.fill_codes(S, nq, A.pool)
            for v in VARIANTS:
                with variant(v):
                    C.run_model(S, key, nq, A, {})
        for key in KEYS:
            log(X.model_line(S, key))
        gc.collect()
        seg = {key: {v: {} for v in VARIANTS} for key in KEYS}
        sizes, failing, differ = [], [], {}
        md, top5 = {k: 0.0 for k in KEYS}, {k: 0 for k in KEYS}
        flips = vals = fq = 0
        for j in range(nq):
            A = C.stage_a(S, j, {})
            C.fill_codes(S, j, A.pool)
            outs, line = {}, ""
            for ki, key in enumerate(KEYS):
                got = {}
                k0 = (j + ki) % len(VARIANTS)
                for v in VARIANTS[k0:] + VARIANTS[:k0]:
                    T = {}
                    with variant(v):
                        got[v] = C.run_model(S, key, j, A, T)
                    for k in SEGMENTS:
                        if k in T:
                            seg[key][v].setdefault(k, []).append(float(T[k]))
                for v in VARIANTS[1:]:
                    d = same_run(got["orig"], got[v])
                    if d:
                        differ.setdefault(f"{key}:{v}", []).append(j)
                        log(f"q{j} {key} variant {v} DIFFERS from the run without A6's forms: {d}")
                outs[key] = got["all"]
                line += (f" {key} (b)+(c) {seg[key]['orig']['total'][-1] * 1000:.1f} -> {seg[key]['all']['total'][-1] * 1000:.1f}"
                         f" (next {seg[key]['orig'].get('next', [0])[-1] * 1000:.1f} -> {seg[key]['all'].get('next', [0])[-1] * 1000:.1f}"
                         f", forward {seg[key]['orig']['forward'][-1] * 1000:.1f} -> {seg[key]['all']['forward'][-1] * 1000:.1f})")
                del got
            res = C.dev_checks(S, j, A, outs)
            bad = {k: v for k, v in res.items()
                   if v is False or (isinstance(v, dict) and (v.get("ok") is False or v.get("identical") is False))}
            if set(bad) == {"mlp_feats"} and set(bad["mlp_feats"]["per_key"]) == {"NEXT"}:
                bad = {}                                                    # float16 NEXT values a step off are counted
            if bad:
                failing.append(j)
                log(f"q{j} FAILED: {json.dumps(bad, default=str)[:1200]}")
            flips += res["next_flips"]
            vals += res["next_values"]
            fq += res["next_flips"] > 0
            for key in KEYS:
                md[key] = max(md[key], res[f"{key}_scores"]["max_diff"] or 0.0)
                top5[key] += bool(res[f"{key}_scores"]["top5_same"])
            sizes.append((int(A.pool.size), int(outs["gnn"][1]["ne"])))
            log(f"q{j} n={A.pool.size} edges={sizes[-1][1]}{line}; scores max diff mlp {res['mlp_scores']['max_diff']} gnn "
                f"{res['gnn_scores']['max_diff']}, NEXT flips {res['next_flips']}/{res['next_values']}")
            if j < int(a.prof):
                for key in KEYS:
                    forward_ops(S, key, j, int(A.pool.size), *outs[key][2])
            del outs
        ran = forms_ran(nq)
        stats, xstats = dict(STATS), dict(X.STATS)
        fast_index = dict(S.fast_index)
    sz = np.array(sizes)
    log(f"pool rows p50 {int(np.median(sz[:, 0]))} (max {int(sz[:, 0].max())}), link edges p50 {int(np.median(sz[:, 1]))} (max "
        f"{int(sz[:, 1].max())})")
    out = {}
    for key in KEYS:
        out[key] = {v: {k: [p50(x), mean(x)] for k, x in seg[key][v].items()} for v in VARIANTS}
        for k in SEGMENTS:
            if k in seg[key]["orig"]:
                log(f"  {key:<4} {k:<9} p50 ms " + "  ".join(f"{v} {out[key][v][k][0]:7.2f}" for v in VARIANTS if k in seg[key][v])
                    + "   (mean orig/all " + f"{out[key]['orig'][k][1]:.2f}/{out[key]['all'][k][1]:.2f})")
        o, y = out[key]["orig"]["total"][0], out[key]["all"]["total"][0]
        log(f"{key}: (b)+(c) p50 {o:.2f} -> {y:.2f} ms, x{o / y:.2f}; scores vs the filed read: max diff {md[key]:.3g}, same top "
            f"five on {top5[key]} of {nq}")
    log(f"NEXT: {flips} float16 values of {vals} differ from the filed read, in {fq} questions; forms ran: {ran} "
        f"{json.dumps(stats)} {json.dumps(xstats)}")
    ok = not failing and not differ and ran
    log("DEVY_JSON " + json.dumps({"dataset": a.dataset, "questions": nq, "blas": C.BLAS, "fast_index": fast_index, "seg": out,
                                   "max_diff": md, "top5_same": top5, "next_flips": flips, "next_values": vals, "sizes": sizes,
                                   "stats": stats, "stats_x": xstats, "failing": failing, "differ": differ}))
    log("DEVY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ── the timed run (Track A's M5) ─────────────────────────────────────────────


def run(a):
    """zfsvc.run (the timed passes, the checks, the raw record) with A4's forms and A6's installed; the raw files go to
    outputs/fsv/timing/a6 (zfsvc's and A4's raw files are not touched). Track A's re-time (M5) runs this; the forms' author makes
    no timed number."""
    C._load()
    ns = SimpleNamespace(dataset=a.dataset, nq=a.nq, lps=a.lps, require_pin=a.require_pin, overwrite=a.overwrite, full_rule=False,
                         questions=a.questions)
    with installed(C.OUT / "a6") as P:
        rc = C.run(ns)
        nq = a.nq or C.T.NQ
        if not forms_ran(nq):
            raise SystemExit(f"the forms did not run: {X.STATS} {STATS}")
        out_file = (C.RAW if nq == C.T.NQ else C.OUT / "raw_smoke") / f"{a.dataset}__b{C.BLAS}.json"
        d = json.load(open(out_file))
        d["head"]["forms"] = {"zfsvx_sha256": C.T.sha_src(X.__file__), "zfsvy_sha256": C.T.sha_src(__file__),
                              "fast_index": P.S.fast_index, "stats": dict(X.STATS), "stats_y": dict(STATS)}
        C.T.write_json(out_file, d)
    return rc


M5_TEXT = (("# FSV timing, Family 1, A2 forms", "# FSV timing, Family 1, A6 forms (M5)"),
           ("each at the stage forms of amendment A2 (`docs/FSV_FAST_VERSIONS.md` section 10).",
            "each at the stage forms of amendment A2 (`docs/FSV_FAST_VERSIONS.md` section 10) with amendment A4's forms of stage "
            "(a) and of the GNN's layers (section 12) and amendment A6's forms of stage (b) and of the encoder (section 13). This "
            "file is M5, the timing of those forms; the passes of A4's forms (M3) and of A2's own forms (M2) stand as the record "
            "of them."),
           ("- Stage (a) is PKO's own rule on one question at batch 1 on one thread; no fused forward exists for either model "
            "(eager torch, one thread);",
            "- Stage (a) is PKO's rule's own calls on one question at batch 1 on one thread, with the walks on P4b's records and "
            "only the orders the pool and the KT columns read (amendment A4); the GNN's layer step is A4's; stage (b)'s links of "
            "both models, the MLP's NEXT columns and the encoder's segment z-scores of both models are A6's (eager torch, one "
            "thread);"))


def report(a):
    """zfsvc.report on the M5 raw files (outputs/fsv/timing/a6/raw by default), labelled A6: the reduction is zfsvc's, and the label,
    the amendment and the three phrases of its prose that name A2's forms are replaced (each must be found, or the report stops);
    the forms' own record (both modules' sha256, the records' index time and size, what ran) joins each dataset."""
    raw_dir = Path(a.raw_dir) if a.raw_dir else C.OUT / "a6" / "raw"
    out_dir = Path(a.out_dir) if a.out_dir else C.OUT / "a6"
    rc = C.report(SimpleNamespace(raw_dir=str(raw_dir), out_dir=str(out_dir)))
    f = out_dir / "report.json"
    rep = json.loads(f.read_text(encoding="utf-8"))
    rep["label"] = "A6 forms (M5)"
    rep["amendment"] = "docs/FSV_FAST_VERSIONS.md section 13 (A6)"
    for tag, Pz in rep["passes"].items():
        for ds, d in Pz["datasets"].items():
            head = json.loads((raw_dir / f"{ds}__{tag}.json").read_text(encoding="utf-8"))["head"]
            if "forms" not in head or "stats_y" not in head["forms"]:
                raise SystemExit(f"{ds}__{tag}: the raw file does not record A6's forms (it is not a zfsvy run)")
            d["forms"] = head["forms"]
    C.T.write_json(f, rep)
    md = C._md(rep)
    for old, new in M5_TEXT:
        if old not in md:
            raise SystemExit(f"zfsvc's report text changed: not found: {old[:60]}")
        md = md.replace(old, new)
    (out_dir / "report.md").write_text(md, encoding="utf-8", newline="\n")
    log(f"M5 report: {out_dir / 'report.json'}, report.md")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "zbench", "profile", "devy", "run", "report"))
    ap.add_argument("--raw-dir", dest="raw_dir", default="", help="report: the raw files (default outputs/fsv/timing/a6/raw)")
    ap.add_argument("--out-dir", dest="out_dir", default="", help="report: where report.json and report.md go (a6)")
    ap.add_argument("--nq", type=int, default=0, help="run: a smoke run of this many questions, 0 = the 200")
    ap.add_argument("--lps", default="2,4", help="run: the logical processors the process is pinned to")
    ap.add_argument("--require-pin", dest="require_pin", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--questions", type=int, default=40)
    ap.add_argument("--blas", default=C.BLAS, help="OpenBLAS threads; zfsvc reads it from argv too (default there: 1)")
    ap.add_argument("--lp", type=int, default=8)
    ap.add_argument("--prof", type=int, default=2, help="torch's profiler on this many questions' forwards")
    ap.add_argument("--trials", type=int, default=6)
    a = ap.parse_args(argv)
    return {"selftest": selftest, "zbench": zbench, "profile": profile, "devy": devy, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
