"""FSV's faster exact forms of the cold path (candidate amendment A4 of docs/FSV_FAST_VERSIONS.md). Development only: no timed
number is made here, and nothing of zfsvc.py, zpko.py or the C-stage modules is edited (they are imported).

  stage (a)  zfsvc.lean_rule's calls with
             1. the walks on P4b's declared form (scripts/p4b_fast_walks.py: breadth-first numbering, 16-byte hot and cold
                records, merged neighbour lists, generation-stamped keys), built once from the unions at index time; the
                relational walks on the same records (frwalk: zu2m_a's relational push, statement for statement, the steer a
                bitmap, the weights summed first). Every walk's order is the original's, byte for byte.
             2. only the walks the pool and the KT columns read (A1st, A1all, Kst, KTst, KTall, and the five relational walks);
                Kall, A1Tst and A1Tall feed orders nothing downstream reads, so they are not walked
             3. only the orders those columns read (KIND_T, A1st, A1all, Kst, KTst, KTall), from p1e_orders' and rule_orders'
                own lines
             4. the expansion's orders: kt_cos only (the pick's order), its cosines from the float16 rows as the store holds
                them, converted and normalised in one numba pass whose float32 norm is certified to be the reference's
                (einsum's float64 sum differs from an eight-lane sum by 1.7e-13 relative at most; a norm whose float32 rounding
                the band could change takes the reference's own form for the whole matrix); the rows only N2 brings stay zero
                rows in the product (a float32 row of a matrix-vector product depends on its place, not on the other rows)
  stage (c)  zg6r's layer step (g6r_step_x) for a pool with no typed edge, one question, float32, no gradient: the attention logit
             from each node's own dot product (the gather of a product summed over the 64 values is the sum gathered), and the
             messages with their two scatter-adds in one numba pass over the edges, in the edges' order; every other op is the
             module's own, in its order. Any other input takes the module's step (counted). The MLP's forward is the module's.
  installed  the forms in this process (zfsvc.lean_rule, zfsvc.setup, zg6r.g6r_step replaced, the P4b records built at index time),
             put back on exit and asserted; nothing of zfsvc.py, zg6r.py or the C-stage modules is edited
  checks     every question: the pool, the expansion and the KT columns equal zfsvc.lean_rule's byte for byte, and with them the
             first KIND_T and the six orders the columns read; each model's forward equals the module's byte for byte, and A2's own
             development checks (zfsvc.dev_checks) hold with the forms installed; selftest: the walks, the orders, the cosines and
             the layers on random inputs

    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvx.py selftest --blas 2
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvx.py profile --dataset musique --questions 40 --blas 2 --lp 8
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvx.py devx --dataset musique --questions 40 --blas 2 --lp 8
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvx.py run --dataset musique --blas 2 --lps 2,4 --require-pin
                                                      (the timed passes with the forms: Track A's re-time; raw files in outputs/fsv/timing/a4)
    python outputs/mp_unified/zfsvx.py report         (zfsvc's reduction of those raw files, labelled A4: outputs/fsv/timing/a4/report.json, .md)
"""
import sys

import zfsvc as C                       # first: it sets the thread variables before numpy loads

import argparse
import contextlib
import gc
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

np = C.np
clock = C.clock


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zfsvx: {msg}", flush=True)


class Seg:
    """Adds the time of its block to T[name]."""
    __slots__ = ("T", "n", "t0")

    def __init__(self, T, n):
        self.T, self.n = T, n

    def __enter__(self):
        self.t0 = clock()
        return self

    def __exit__(self, *a):
        self.T[self.n] = self.T.get(self.n, 0.0) + clock() - self.t0


# ── the walks on P4b's records ───────────────────────────────────────────────


_F = {}
EMPTY_I = np.empty(0, np.int64)
EMPTY_F = np.empty(0, np.float64)


def fkernels():
    """P4b's records and helpers (scripts/p4b_fast_walks.py, imported) and the relational push on them: zu2m_a's _rpush's
    statements in its order (family 0's edges weigh wq[class], the other families' 1, each times floor + ex when steered; the
    weights summed first, then each neighbour's share), a walk's key a generation, the top by original id."""
    if _F:
        return _F
    import p4b_fast_walks as P4B
    from numba import njit
    _seeds, _pop, _settle, _families, _top, _bit, _prefetch = (P4B._seeds, P4B._pop, P4B._settle, P4B._families, P4B._top,
                                                               P4B._bit, P4B._prefetch)

    @njit
    def merge_cls(ip0, ip1, ip2, orig, cls, mcls):
        """The class of every entry of the merged list: family 0's entries carry their class, the others 0 (not read)."""
        pos = 0
        for x in range(ip0.size - 1):
            u = orig[x]
            for k in range(ip0[u], ip0[u + 1]):
                mcls[pos] = cls[k]
                pos += 1
            for _k in range((ip1[u + 1] - ip1[u]) + (ip2[u + 1] - ip2[u])):
                mcls[pos] = 0
                pos += 1
        return pos

    @njit(inline="always")
    def _rspread(cols, mcls, a, b, rel, m, nf, wq, floor, steered, bm, ex, hot, gen4, eps, queue, nt, tail, cnt, pf):
        """zu2m_a's _rspread on the records: the weights of [a, b) summed in order, then m / (nf * sum) times each weight to
        its neighbour. A neighbour off the steer set has ex 0, so floor + ex is floor."""
        tot = 0.0
        for k in range(a, b):
            w = 1.0
            if rel:
                w = wq[mcls[k]]
            if steered:
                v = cols[k]
                if _bit(bm, v):
                    w = w * (floor + ex[v])
                else:
                    w = w * floor
            tot += w
        base = m / (nf * tot)
        for k in range(a, b):
            if pf > 0 and k + pf < b:
                _prefetch(hot, cols[k + pf])
            v = cols[k]
            w = 1.0
            if rel:
                w = wq[mcls[k]]
            if steered:
                if _bit(bm, v):
                    w = w * (floor + ex[v])
                else:
                    w = w * floor
            add = base * w
            key = hot[v].key
            if key < gen4:
                key = gen4
                hot[v].key = key
                rv = 0.0 + add
                nt += 1
            else:
                rv = hot[v].r + add
            hot[v].r = rv
            if key & 1 == 0 and rv >= eps * max(hot[v].deg, 1):
                hot[v].key = key + 1
                queue[tail] = v
                tail += 1
                if tail == queue.size:
                    tail = 0
                cnt += 1
        return nt, tail, cnt

    @njit
    def frwalk(seeds, wts, st_nodes, st_ex, floor, steered, wq, hot, cold, cols, mcls, gen4, alpha, eps, queue, pl, pv, bm, ex,
               bmax, out, orig, pf, pq, pc, ph, hc):
        """zu2m_a's relational walk (mode 1) then step 4's top. Returns (written, touched)."""
        if steered:
            for k in range(st_nodes.size):
                v = st_nodes[k]
                ex[v] = st_ex[k]
                bm[v >> 6] |= np.int64(1) << (v & 63)
        nt, tail, cnt = _seeds(seeds, wts, hot, gen4, queue)
        head = 0
        npl = 0
        while cnt > 0:
            u, head, cnt = _pop(hot, cold, cols, queue, head, cnt, pq, pc, ph, hc)
            du = hot[u].deg
            ru = hot[u].r
            if ru < eps * max(du, 1):
                continue
            hot[u].r = 0.0
            npl = _settle(hot, cold, u, ru, alpha, pl, pv, npl)
            if du == 0:
                continue
            m = (1.0 - alpha) * ru
            a, l0, l1, l2, nf = _families(cold, u, du)
            if l0 > 0:
                nt, tail, cnt = _rspread(cols, mcls, a, a + l0, True, m, nf, wq, floor, steered, bm, ex, hot, gen4, eps, queue,
                                         nt, tail, cnt, pf)
            if l1 > 0:
                nt, tail, cnt = _rspread(cols, mcls, a + l0, a + l0 + l1, False, m, nf, wq, floor, steered, bm, ex, hot, gen4,
                                         eps, queue, nt, tail, cnt, pf)
            if l2 > 0:
                nt, tail, cnt = _rspread(cols, mcls, a + l0 + l1, a + du, False, m, nf, wq, floor, steered, bm, ex, hot, gen4,
                                         eps, queue, nt, tail, cnt, pf)
        _avail, w = _top(pl, pv, npl, bmax, out, orig)
        if steered:
            for k in range(st_nodes.size):
                v = st_nodes[k]
                ex[v] = 0.0
                bm[v >> 6] = 0
        return w, nt

    _F.update(P4B=P4B, merge_cls=merge_cls, frwalk=frwalk)
    return _F


class Fast:
    """Index time: P4b's declared form (breadth-first numbering, hot and cold records, merged lists) of the unions, and each
    union's merged list's relational classes. The unweighted and the steered walks are P4b's own (walk, twalk); the relational
    walk is frwalk on the same records. Every call returns the order in the original node ids."""

    def __init__(self, unions, cls):
        F = fkernels()
        t0 = clock()
        self.wk, self.orig = F["P4B"].form_walker(unions)
        t1 = clock()
        self.mcls = {}
        cls = np.ascontiguousarray(cls)
        for u, f in self.wk.fu.items():
            m = np.zeros(f.cols.size, cls.dtype)
            if F["merge_cls"](*(unions[u].args[i] for i in (0, 2, 4)), self.orig, cls, m) != m.size:
                raise SystemExit("the classes' merged list is not the union's")
            self.mcls[u] = m
        self.build = (t1 - t0, clock() - t1)
        self.outs = {}

    def nbytes(self):
        return self.wk.nbytes() + sum(int(m.nbytes) for m in self.mcls.values())

    def walk(self, u, restart):
        return self.wk.walk(u, restart)[0]

    def twalk(self, u, restart, steer):
        return self.wk.twalk(u, restart, steer)[0]

    def rwalk(self, u, restart, steer, wq, alpha, eps, bmax):
        """zu2m_a.rwalk's order for one restart (mode 1) on union u; steer = (nodes, ex, floor) or None."""
        nodes, w = restart
        if nodes.size == 0:
            return EMPTY_I
        wk = self.wk
        f = wk.fu[u]
        out = self.outs.get(int(bmax))
        if out is None:
            out = self.outs[int(bmax)] = np.zeros(int(bmax), np.int64)
        if steer is None:
            st_nodes, st_ex, floor = EMPTY_I, EMPTY_F, 1.0
        else:
            st_nodes, st_ex, floor = wk._in(steer[0]), np.ascontiguousarray(steer[1], np.float64), float(steer[2])
        cnt, _nt = fkernels()["frwalk"](wk._in(nodes), np.ascontiguousarray(w, np.float64), st_nodes, st_ex, floor,
                                        steer is not None, np.ascontiguousarray(wq, np.float64), f.hot, f.cold, f.cols,
                                        self.mcls[u], f.next_gen(), float(alpha), float(eps), wk.queue, wk.pl, wk.pv, wk.bm,
                                        wk.ex, int(bmax), out, wk.orig, wk.pf, wk.pq, wk.pc, wk.ph, wk.hc)
        return out[:cnt].copy()


# ── the lean orders ─────────────────────────────────────────────────────────


def kt_orders_x(Kc, Dc, a1st, a1all, kst, r0, ktst, ktall, route, knd, constant):
    """The six orders zu2d_pools.kt_orders reads, from p1e_orders' and rule_orders' own lines: the orders of A1st, A1all, Kst
    and R0 cut at NMAX; KTst and KTall cut at NMAX; KIND_T = ROUTEst for an entity question, RRF5T otherwise."""
    nm = Dc.NMAX
    o_a1st, o_a1all, o_kst, o_r0 = (np.asarray(x, np.int64)[:nm] for x in (a1st, a1all, kst, r0))
    o_kt, o_kta = np.asarray(ktst, np.int64)[:Kc.NMAX], np.asarray(ktall, np.int64)[:Kc.NMAX]
    if knd == "entity":
        kind_t = Dc.backfill(o_kst, o_a1st) if route else o_a1st
    else:
        kind_t = Dc.rrf([o_a1st, o_a1all, o_kt, o_kta, o_r0], constant)
    return {"KIND_T": kind_t, "A1st": o_a1st, "A1all": o_a1all, "Kst": o_kst, "KTst": o_kt, "KTall": o_kta}


# ── the cosines of the expansion's orders ───────────────────────────────────

NORM_DELTA = 1e-12          # the relative band a float64 sum of 1536 squares can differ by between two summation orders: 1.7e-13
_O = {}


def okernels():
    if _O:
        return _O
    from numba import njit

    @njit(inline="always")
    def _sumsq(src, i, dim):
        """The float64 sum of row i's squares (each product exact in float64), in eight lanes."""
        a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = 0.0
        j = 0
        while j + 8 <= dim:
            v0 = np.float64(src[i, j])
            v1 = np.float64(src[i, j + 1])
            v2 = np.float64(src[i, j + 2])
            v3 = np.float64(src[i, j + 3])
            v4 = np.float64(src[i, j + 4])
            v5 = np.float64(src[i, j + 5])
            v6 = np.float64(src[i, j + 6])
            v7 = np.float64(src[i, j + 7])
            a0 += v0 * v0
            a1 += v1 * v1
            a2 += v2 * v2
            a3 += v3 * v3
            a4 += v4 * v4
            a5 += v5 * v5
            a6 += v6 * v6
            a7 += v7 * v7
            j += 8
        tl = 0.0
        while j < dim:
            v0 = np.float64(src[i, j])
            tl += v0 * v0
            j += 1
        return (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) + tl

    @njit
    def sumsq_rows(src, out):
        for i in range(src.shape[0]):
            out[i] = _sumsq(src, i, src.shape[1])

    @njit
    def unit_rows(src, dst, pos, delta):
        """dst[pos[i]] = d5b_neighbour_orders.unit's row i of src (float32, C order): each row's norm is the float64 sum of its
        squares (exact products; the sum in eight lanes, not einsum's order), its square root cut to float32, the row divided by
        it (by 1 when it is not above 0), in float32. The sum can differ from einsum's by 1.7e-13 relative at most (any order of
        1536 non-negative float64 terms is that close to the exact sum): the float32 norm is the one einsum's would give when the
        band [y (1 - delta), y (1 + delta)] holds no float32 rounding midpoint. Returns -1, or the first row where it does (or
        the sum is not a number): nothing of such a row or any after it is to be used."""
        n, dim = src.shape
        for i in range(n):
            y = np.sqrt(_sumsq(src, i, dim))
            lo = np.float32(y * (1.0 - delta))
            hi = np.float32(y * (1.0 + delta))
            if not lo == hi:
                return i
            d = lo
            if not d > np.float32(0.0):
                d = np.float32(1.0)
            p = pos[i]
            for j in range(dim):
                dst[p, j] = src[i, j] / d
        return -1

    _O.update(unit_rows=unit_rows, sumsq_rows=sumsq_rows)
    return _O


def f32_rows(nodes, ids):
    """The rows of ids as float32 from the float16 the store holds: the gather unconverted, then torch's conversion (exact, as
    numpy's astype, and several times faster)."""
    return C.torch.from_numpy(nodes.read(ids, np.float16)).to(C.torch.float32).numpy()


def unit_of(nodes, ids, delta=NORM_DELTA):
    """d5b_neighbour_orders.unit(Rows.read(ids)) (float32, C order), the norms by unit_rows; the reference's own form for the
    whole matrix where a norm is not certified."""
    x = f32_rows(nodes, ids)
    if x.shape[0] and okernels()["unit_rows"](x, x, np.arange(x.shape[0]), delta) >= 0:
        STATS["unit_fallback"] += 1
        x = C.Z.D5B.unit(f32_rows(nodes, ids))
    return x


def unit_into(nodes, ids, dst, pos, delta=NORM_DELTA):
    """dst[pos] = d5b_neighbour_orders.unit's rows of ids' rows (the other rows of dst are left as they are)."""
    x = f32_rows(nodes, ids)
    if okernels()["unit_rows"](x, dst, pos, delta) >= 0:
        STATS["unit_fallback"] += 1
        dst[pos] = C.Z.D5B.unit(f32_rows(nodes, ids))


def kt_cos_x(nodes, q, full, n1, cos1, r0, iout, n2, c, rrf):
    """zu2e.question_orders' 'kt_cos' (its M = 0 column, the only one PICK reads), from its own lines for that column: the cosine
    of every node of U = full, N1, R0, I outside B and N2, its first order by cosine over those of full, N1, R0 and I, fused with
    full's by RRF. The rows N2 alone brings (not in full, R0 or I) are not read: their cosines reach no order, and a row of the
    cosine product depends on its place in the matrix, not on the other rows' values (sgemv's kernels take the rows by position),
    so the matrix keeps N2's places as zero rows and every other row's product is the original's."""
    full = np.asarray(full, np.int64)
    n1 = np.asarray(n1, np.int64)
    r0, iout, n2 = (np.asarray(x, np.int64) for x in (r0, iout, n2))
    U = np.unique(np.concatenate([full, n1, r0, iout, n2]))
    rest = np.setdiff1d(U, n1, assume_unique=True)
    cu = np.empty(U.size, np.float32)
    if n1.size:
        cu[np.searchsorted(U, n1)] = cos1
    if rest.size:
        need = np.isin(rest, full) | np.isin(rest, r0) | np.isin(rest, iout)
        part = []
        for a in range(0, rest.size, C.Z.U2E.CHUNK):
            nd = need[a:a + C.Z.U2E.CHUNK]
            m = np.zeros((nd.size, q.size), np.float32)
            sel = np.flatnonzero(nd)
            if sel.size:
                unit_into(nodes, rest[a:a + nd.size][sel], m, sel)
            part.append(m @ q)
        cu[np.searchsorted(U, rest)] = np.concatenate(part)
    base = np.isin(U, full) | np.isin(U, n1) | np.isin(U, r0) | np.isin(U, iout)
    ids = U[base]
    cos_0 = ids[np.lexsort((ids, -cu[base].astype(np.float64)))]
    return rrf([full, cos_0], c, nmax=C.Z.U2E.BIG)


# ── stage (a)'s rule: the profile clone (zfsvc.lean_rule, line for line, each call timed) and the workspace form ────────


def prof_rule(S, j, linked, I, Bq, T, X=None):
    """zfsvc.lean_rule with a timer around each call; the outputs are lean_rule's. X (a dict) takes the first KIND_T and the
    six orders the columns read."""
    Z = C.Z
    cx, ctx = S.cx, S.ctx
    Kc, Dc, prep, constant, threads = cx.K, cx.D, cx.prep, cx.constant, cx.threads
    js, lk = [j], {j: linked}
    with Seg(T, "sources"):
        srcs = Dc.sources_batch(prep, js, lk, cx.unions, constant, threads)
    with Seg(T, "steered"):
        steered = Kc.steered_batch(prep, js, lk, cx.unions, srcs, constant, threads)
    with Seg(T, "restarts"):
        k0, a1 = Kc.restarts_batch(prep, js, lk, cx.unions, srcs, constant)
    with Seg(T, "steer"):
        stz = [Kc.steer(prep, j, constant)]
    with Seg(T, "omega"):
        q = Z.D5B.unit(np.asarray(prep.qemb[j], np.float64))
        wa = [Z.A.omega(q, ctx.mu, Z.Q.A_BETA, True)[0]]
        wqs = [Z.P.pair_weights(x, ctx.ws) for x in wa] if ctx.passage else wa
    wal = {}
    for w in Z.Q.WALKS:
        with Seg(T, "rwalk_" + w):
            wal[w] = Z.A.rwalk(cx.unions[Z.Q.WALK_UNION[w]], a1 if w.startswith("A1") else k0,
                               stz if w.startswith("KT") else None, wqs, 1, ctx.ty, ctx.alpha, ctx.eps, ctx.bmax, threads)
    src, _nn, (L, Ls_) = srcs[0]
    with Seg(T, "route_kind"):
        route = Dc.routes(L, Ls_)
        knd = Kc.kind(Ls_, prep.seeds[j], cx.nonly)
    with Seg(T, "scaled"):
        sc = (Z.A.scaled(wal["A1st"][0]), Z.A.scaled(wal["A1all"][0]), Z.A.scaled(wal["Kst"][0]), src[3], src[4])
        stc = dict(steered[0])
        stc["KTst"], stc["KTall"] = wal["KTst"][0][0], wal["KTall"][0][0]
    with Seg(T, "p1e_orders1"):
        o1, _p1 = Kc.p1e_orders(sc, stc, route, knd, cx.n_st, constant)
    kt = np.asarray(o1["KIND_T"], np.int64)
    with Seg(T, "n1_table"):
        n1 = Z.D5B.n1_table(cx.adj, Bq, Z.D5.weights_of(stz[0], Bq))[0]
    with Seg(T, "cos1"):
        rows = Z.M.Rows(cx.nodes, verify=False)
        rows.add(n1)
        cos1 = (Z.D5B.unit(rows.read(n1)) @ q) if n1.size else np.zeros(0, np.float32)
    with Seg(T, "ranks"):
        cr = Z.D5B.ranks_desc(cos1, n1) if n1.size else Z.S.EMPTY
    with Seg(T, "second_link"):
        n2, first, _taken, _hubs = Z.U2E.second_link(cx.adj, n1, cr, Bq)
    r0src = np.asarray(src[4][0], np.int64)
    with Seg(T, "rows_add"):
        rows.add(np.concatenate([kt, r0src, I, np.asarray(n2, np.int64)]))
    with Seg(T, "outside"):
        full = Z.F.outside(kt, Bq)
        fill = int(Z.M.fill_of(full, I.size, Bq.size))
        r0 = Z.F.outside(r0src, Bq)
        iout = Z.F.outside(I, Bq)
    with Seg(T, "question_orders"):
        ords, _sizes, _U, _mem = Z.U2E.question_orders(full, n1, cr, cos1, r0, iout, n2, first, rows.read, q, constant, Dc.rrf)
    with Seg(T, "kept"):
        kt_cos = np.asarray(ords[Z.M.NAMES[ctx.plan.o]], np.int64)
        _kq, _k, keep = (int(x[0]) for x in Z.kept_counts([kt_cos.size], [fill], [Bq.size], ctx.plan))
        exp = kt_cos[:keep]
        pool = np.union1d(Bq, exp)
    with Seg(T, "p1e_orders2"):
        oe, pe = Kc.p1e_orders(src, steered[0], bool(route), knd, cx.n_st, constant)
    with Seg(T, "kt_cols"):
        ko = Z.ZP.kt_orders(oe, pe)
        kt16 = Z.KT3.columns(ko, pool, exp, knd == "entity").astype(Z.ZP.KT_DT)
    if X is not None:
        X["kt"], X["oe"] = kt, ko
    return pool, kt16, exp


def lean_rule_x(S, j, linked, I, Bq, T, X=None):
    """zfsvc.lean_rule with the walks on P4b's records (S.fast, built at index time), only the walks and orders the pool and the
    columns read, and the rest of its calls as they are. X (a dict) takes the first KIND_T and the six orders the columns read."""
    Z = C.Z
    cx, ctx, fast = S.cx, S.ctx, S.fast
    Kc, Dc, prep, constant = cx.K, cx.D, cx.prep, cx.constant
    if Z.M.NAMES[ctx.plan.o] != "kt_cos":
        raise SystemExit(f"the pick's order is {Z.M.NAMES[ctx.plan.o]}: the lean orders hold kt_cos's column only")
    STATS["lean_rule_x"] += 1
    with Seg(T, "x_unw"):                                       # the unweighted walks the columns read
        L, Ls_ = Dc.anchors(prep, j, linked)
        a1 = Dc.a1_restart(prep, j, linked, cx.unions["st"].deg, constant)
        k0 = Dc.B.uniform(Ls_)
        stz = Kc.steer(prep, j, constant)
        u_a1st = fast.walk("st", a1)
        u_a1all = fast.walk("all", a1)
        u_kst = fast.walk("st", k0)
        u_ktst = fast.twalk("st", k0, stz)
        u_ktall = fast.twalk("all", k0, stz)
        r0 = Dc.r0_source(prep, j, constant)[0]
    with Seg(T, "x_rel"):                                       # the relational walks of the pool's KIND_T
        q = Z.D5B.unit(np.asarray(prep.qemb[j], np.float64))
        wa = Z.A.omega(q, ctx.mu, Z.Q.A_BETA, True)[0]
        wq = Z.P.pair_weights(wa, ctx.ws) if ctx.passage else wa
        wal = {}
        for w in Z.Q.WALKS:
            wal[w] = fast.rwalk(Z.Q.WALK_UNION[w], a1 if w.startswith("A1") else k0, stz if w.startswith("KT") else None, wq,
                                ctx.alpha, ctx.eps, ctx.bmax)
    with Seg(T, "x_ord1"):
        route = Dc.routes(L, Ls_)
        knd = Kc.kind(Ls_, prep.seeds[j], cx.nonly)
        o1 = kt_orders_x(Kc, Dc, wal["A1st"], wal["A1all"], wal["Kst"], r0, wal["KTst"], wal["KTall"], route, knd, constant)
        kt = np.asarray(o1["KIND_T"], np.int64)
    with Seg(T, "n1_table"):
        n1 = Z.D5B.n1_table(cx.adj, Bq, Z.D5.weights_of(stz, Bq))[0]
    with Seg(T, "cos1"):
        cos1 = (unit_of(cx.nodes, n1) @ q) if n1.size else np.zeros(0, np.float32)
    with Seg(T, "ranks"):
        cr = Z.D5B.ranks_desc(cos1, n1) if n1.size else Z.S.EMPTY
    with Seg(T, "second_link"):
        n2, _first, _taken, _hubs = Z.U2E.second_link(cx.adj, n1, cr, Bq)
    r0src = np.asarray(r0, np.int64)
    with Seg(T, "outside"):
        full = Z.F.outside(kt, Bq)
        fill = int(Z.M.fill_of(full, I.size, Bq.size))
        r0o = Z.F.outside(r0src, Bq)
        iout = Z.F.outside(I, Bq)
    with Seg(T, "question_orders"):
        kt_cos = kt_cos_x(cx.nodes, q, full, n1, cos1, r0o, iout, n2, constant, Dc.rrf)
    with Seg(T, "kept"):
        kt_cos = np.asarray(kt_cos, np.int64)
        _kq, _k, keep = (int(x[0]) for x in Z.kept_counts([kt_cos.size], [fill], [Bq.size], ctx.plan))
        exp = kt_cos[:keep]
        pool = np.union1d(Bq, exp)
    with Seg(T, "x_ord2"):
        oe = kt_orders_x(Kc, Dc, u_a1st, u_a1all, u_kst, r0, u_ktst, u_ktall, bool(route), knd, constant)
    with Seg(T, "kt_cols"):
        kt16 = Z.KT3.columns(oe, pool, exp, knd == "entity").astype(Z.ZP.KT_DT)
    if X is not None:
        X["kt"], X["oe"] = kt, oe
    return pool, kt16, exp


# ── stage (c): the GNN's layers, and the installation of the forms ───────────

_G = {}
_ORIG = {}
STATS = {"lean_rule_x": 0, "setup_fast": 0, "g6r_step_x": 0, "g6r_step_orig": 0, "unit_fallback": 0}


def gkernels():
    if _G:
        return _G
    from numba import njit

    @njit
    def edge_acc(hm, u, f, v, at, s, ex, mean, num):
        """The two scatter-adds of zg6r.g6r_step in one pass over the edges, in their order: mean[at[e]] += s[e] * hm[u[e], f[e]]
        and num[v[e]] += ex[e] * hm[u[e], f[e]], each product a rounded float32 and each sum a float32 sum, as torch's index_add_
        takes the edges one after the other and adds each row to its target (alpha 1: fmadd(b, 1, a) is a + b)."""
        d_ = num.shape[1]
        for e in range(u.size):
            base = f[e] * d_
            ue = u[e]
            a = at[e]
            ve = v[e]
            se = s[e]
            xe = ex[e]
            for d in range(d_):
                x = hm[ue, base + d]
                mean[a, d] += se * x
                num[ve, d] += xe * x

    _G["edge_acc"] = edge_acc
    return _G


def _zg6r():
    import zg6r
    return zg6r


def g6r_step_x(m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, te, codes, cte):
    """zg6r.g6r_step for a pool with no typed edge, one question, float32, no gradient. The attention logit from a per-node dot
    product (the gather of the linear map's rows, times the question's context, summed over the 64 values, is each node's sum
    gathered: the same float32 values), the messages and their two scatter-adds in one numba pass (edge_acc). Every other op is
    the original's, in its order. Any other input takes the original's step."""
    R, torch = _zg6r(), C.torch
    if te.numel() or B != 1 or h.dtype != torch.float32 or torch.is_grad_enabled():
        STATS["g6r_step_orig"] += 1
        return _ORIG["g6r_step"](m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, te, codes, cte)
    Fn, D, FAMS, LCLIP = torch.nn.functional, R.D, R.FAMS, R.LCLIP
    wm, wa, fb, ws, wmean, wagg, wx, b = (getattr(m, n) for n in R.G6.G6_LAYER[k])
    N = h.shape[0]
    c = torch.zeros(B, D, dtype=h.dtype).index_add_(0, nq, w * h)
    hm = Fn.linear(h, wm)
    ln = (Fn.linear(h, wa) * c[0]).sum(1) / math.sqrt(D)
    lg = (ln[u] + fb[f]).clamp(-LCLIP, LCLIP)
    s, ex = torch.sigmoid(lg), torch.exp(lg)
    mean_a = torch.zeros(N * FAMS, D, dtype=h.dtype)
    num = torch.zeros(N, D, dtype=h.dtype)
    gkernels()["edge_acc"](hm.numpy(), u.numpy(), f.numpy(), v.numpy(), at.numpy(), s.numpy(), ex.numpy(), mean_a.numpy(),
                           num.numpy())
    den = torch.zeros(N, dtype=h.dtype).index_add_(0, v, ex)
    mean = mean_a.view(N, FAMS, D) / deg
    att = num / den.clamp_min(1e-30).unsqueeze(1)
    pre = Fn.linear(h, ws) + Fn.linear(mean.reshape(N, FAMS * D), wmean) + Fn.linear(att, wagg) + Fn.linear(xb, wx) + b
    STATS["g6r_step_x"] += 1
    return h + Fn.gelu(Fn.layer_norm(pre, (D,)))


def patch_layers(on):
    """zg6r's layer step as the forms give it (on) or as the module has it: the module's own function is kept the first time."""
    R = _zg6r()
    _ORIG.setdefault("g6r_step", R.g6r_step)
    R.g6r_step = g6r_step_x if on else _ORIG["g6r_step"]


@contextlib.contextmanager
def installed(out_dir=None):
    """The forms in this process, put back on exit (also after an error): zfsvc's stage (a) rule (lean_rule_x, the keys of
    zfsvc.lean_rule's T), its setup (the P4b records built at index time, after the original setup) and zg6r's layer step. The
    callers assert from STATS that each ran. out_dir: where a run's raw files go (zfsvc's OUT and RAW are replaced)."""
    if _ORIG.get("installed"):
        raise SystemExit("the forms are already installed")
    _ORIG.update(installed=True, lean_rule=C.lean_rule, setup=C.setup, out=C.OUT, raw=C.RAW)
    for k in STATS:
        STATS[k] = 0
    box = SimpleNamespace(S=None, orig_rule=C.lean_rule)

    def setup_x(ds_name, nq, full_rule=False):
        if full_rule:
            raise SystemExit("the forms replace the lean rule: not with --full-rule")
        S = _ORIG["setup"](ds_name, nq, full_rule)
        t0 = clock()
        S.fast = Fast({k: S.cx.unions[k] for k in ("st", "all")}, S.ctx.ty[0])
        S.fast_index = {"seconds": clock() - t0, "numbering_s": S.fast.build[0], "records_s": S.fast.build[1],
                        "bytes": S.fast.nbytes()}
        STATS["setup_fast"] += 1
        box.S = S
        log(f"{ds_name}: P4b records built at index time: {S.fast_index['seconds']:.1f} s, {S.fast_index['bytes'] / 2 ** 20:.0f} MiB")
        return S

    def lean_rule_p(S, j, linked, I, Bq, T_):
        seg = {}
        out = lean_rule_x(S, j, linked, I, Bq, seg)
        T_["a_walk"] = seg["x_unw"] + seg["x_rel"]
        T_["a_order"] = sum(seg[k] for k in ("x_ord1", "n1_table", "cos1", "ranks", "second_link", "outside", "question_orders",
                                             "kept"))
        T_["a_kt"] = seg["x_ord2"] + seg["kt_cols"]
        return out

    C.lean_rule, C.setup = lean_rule_p, setup_x
    patch_layers(True)
    if out_dir is not None:
        C.OUT = out_dir
        C.RAW = out_dir / "raw"
    try:
        yield box
    finally:
        C.lean_rule, C.setup, C.OUT, C.RAW = _ORIG["lean_rule"], _ORIG["setup"], _ORIG["out"], _ORIG["raw"]
        patch_layers(False)
        _ORIG["installed"] = False
        if _zg6r().g6r_step is not _ORIG["g6r_step"] or C.lean_rule is not box.orig_rule:
            raise SystemExit("the forms were not put back")


# ── the checks ──────────────────────────────────────────────────────────────


def same_bytes(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def same_out(a, b):
    """The three outputs (pool, KT columns, expansion) byte for byte."""
    return len(a) == len(b) and all(same_bytes(x, y) for x, y in zip(a, b))


def same_x(a, b):
    """The first KIND_T and the six orders the columns read, byte for byte."""
    return (same_bytes(a["kt"], b["kt"]) and set(a["oe"]) == set(b["oe"])
            and all(same_bytes(a["oe"][k], b["oe"][k]) for k in b["oe"]))


def rand_graph(n, rng, degs=(8.0, 3.0, 5.0)):
    """A random three-family graph as step 4's Unions ('st': structural only; 'all': the three) and the relational types
    (component of each structural entry among 4)."""
    S4 = C.Kx.S4

    def csr(deg):
        m = int(n * deg)
        u = rng.integers(0, n, m)
        v = rng.integers(0, n, m)
        o = np.lexsort((v, u))
        ip = np.zeros(n + 1, np.int64)
        np.add.at(ip, u + 1, 1)
        return np.cumsum(ip), v[o].astype(np.int32)

    fam = {}
    for k, deg in zip(S4.FAMILIES, degs):
        ip, col = csr(deg)
        fam[k] = SimpleNamespace(n_nodes=n, indptr=ip, col=col)
    empty = SimpleNamespace(n_nodes=n, indptr=np.zeros(n + 1, np.int64), col=np.zeros(0, np.int32))
    unions = {"st": S4.Union({"structural": fam["structural"], "ner": empty, "knn": empty}), "all": S4.Union(fam)}
    ne = int(fam["structural"].indptr[-1])
    ty = (rng.integers(0, 4, ne).astype(np.int8), np.zeros(ne, np.int64), np.ones(ne, np.int32), np.zeros(1, np.int16))
    return unions, ty


def selftest(a):
    """The walks on a small random graph against the modules' own batch kernels (every walk's order byte for byte, steered and
    unsteered, empty and large restarts, the top cut and not), the lean orders against p1e_orders' on random sources, and the
    cosines and kt_cos against question_orders' (selftest_cos). Needs no data."""
    C._load()
    rng = np.random.default_rng(7)
    n = int(a.nodes)
    S4, B, E, D, Kc, A = C.Kx.S4, C.Kx.B, C.E, C.Kx.D, C.Kx, C.Z.A

    def rand_order(size):
        x = np.unique(rng.integers(0, n, size)).astype(np.int64)
        return x[rng.permutation(x.size)]

    unions, ty = rand_graph(n, rng)
    fast = Fast(unions, ty[0])
    log(f"P4b records: {fast.nbytes() / 2 ** 20:.1f} MiB, numbering {fast.build[0]:.2f} s, records {fast.build[1]:.2f} s")
    alpha, eps = float(E.ALPHA), float(S4.EPS)
    bmax = int(Kc.NMAX)
    bad = {"fw": 0, "ft": 0, "fr": 0, "o": 0}
    cuts = 0
    trials = int(a.trials)
    for t in range(trials):
        k = int(rng.choice([0, 1, 3, 12, 40]))
        seeds = np.unique(rng.integers(0, n, k)).astype(np.int64)
        w = rng.random(seeds.size) + 0.1
        restart = (seeds, w / w.sum()) if seeds.size else (np.empty(0, np.int64), np.empty(0))
        stn = np.unique(rng.integers(0, n, int(rng.integers(5, 300)))).astype(np.int64)
        steer = (stn, rng.random(stn.size) * 0.05, float(rng.random() * 0.02 + 0.001))
        wq = rng.random(5) + 0.2
        for un in ("st", "all"):
            u = unions[un]
            want = B.walk_mass(u, [restart], 1)[0][0]
            bad["fw"] += not same_bytes(want, fast.walk(un, restart))
            cuts += int(want.size >= bmax)
            want = Kc.twalk(u, [restart], [steer], 1)[0]
            bad["ft"] += not same_bytes(want, fast.twalk(un, restart, steer))
            for st_ in (None, steer):
                want = A.rwalk(u, [restart], None if st_ is None else [st_], [wq], 1, ty, alpha, eps, bmax, 1)[0][0]
                bad["fr"] += not same_bytes(want, fast.rwalk(un, restart, st_, wq, alpha, eps, bmax))
    log(f"walks: {trials} trials x 2 unions: P4b records unweighted {bad['fw']}, steered {bad['ft']}, relational {bad['fr']} "
        f"differ; {cuts} unweighted walks cut at {bmax}")
    nsrc = 0
    for t in range(trials):
        sd = np.unique(rng.integers(0, n, int(rng.choice([1, 3, 12])))).astype(np.int64)
        r1 = (sd, np.full(sd.size, 1.0 / sd.size))
        sk = sd[:max(1, sd.size // 2)]
        r2 = (sk, np.full(sk.size, 1.0 / sk.size))
        src = (B.walk_mass(unions["st"], [r1], 1)[0], B.walk_mass(unions["all"], [r1], 1)[0],
               B.walk_mass(unions["st"], [r2], 1)[0], B.walk_mass(unions["all"], [r2], 1)[0])
        r0n = rand_order(400)
        src = src + ((r0n, np.full(r0n.size, 1.0 / r0n.size)),)
        st = {kk: rand_order(3000) for kk in ("KTst", "KTall", "A1Tst", "A1Tall")}
        for route in (False, True):
            for knd in ("entity", "passage"):
                out, p1d = Kc.p1e_orders(src, st, route, knd, n, 60)
                want = C.Z.ZP.kt_orders(out, p1d)
                got = kt_orders_x(Kc, D, src[0][0], src[1][0], src[2][0], src[4][0], st["KTst"], st["KTall"], route, knd, 60)
                bad["o"] += not (set(want) == set(got) and all(same_bytes(want[kk], got[kk]) for kk in want))
                nsrc += 1
    log(f"orders: {nsrc} combinations against p1e_orders: {bad['o']} differ")
    bad.update(selftest_cos(rng, int(a.trials)))
    bad.update(selftest_layers(rng, int(a.trials)))
    bad.update(selftest_installed(rng))
    ok = not any(bad.values())
    log("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


class FakeNodes:
    """DenseNodes' read over a float16 matrix held in RAM (m3b_features.DenseNodes with its matrix, line for line)."""

    def __init__(self, m16):
        self.m = m16

    def read(self, rows, dtype=np.float32):
        gathered = self.m[np.asarray(rows, dtype=np.int64)]
        return gathered if gathered.dtype == dtype else gathered.astype(dtype)


def selftest_cos(rng, trials, nn=12000, dim=1536):
    """The cosines and the kt_cos order against zu2e.question_orders' (through Rows and d5b_neighbour_orders.unit), on a random
    float16 store with zero, tiny and large rows: unit_of's matrix and the cosines byte for byte, the sums' largest relative
    distance from einsum's, the not-certified path (a band wide enough to hold midpoints), kt_cos for random sources with N2
    overlapping the others, N1 or N2 empty, and (to show the test can fail) the compacted matrix's cosines, which differ."""
    Z, D5B = C.Z, C.Z.D5B
    rrf = C.Kx.D.rrf
    m16 = np.empty((nn, dim), np.float16)
    for a in range(0, nn, 1000):
        m16[a:a + 1000] = rng.standard_normal((min(1000, nn - a), dim)) * rng.choice([0.01, 1.0, 30.0], (min(1000, nn - a), 1))
    m16[rng.choice(nn, 20, replace=False)] = 0
    m16[rng.choice(nn, 20, replace=False), 3:] = 0                  # a single nonzero value
    nodes = FakeNodes(m16)
    ok = {"cos_unit": 0, "cos_kt": 0, "cos_fallback": 0}
    ids = np.sort(rng.choice(nn, 6000, replace=False)).astype(np.int64)
    x32 = nodes.read(ids)
    ref = D5B.unit(x32)
    got = unit_of(nodes, ids)
    ok["cos_unit"] += not same_bytes(ref, got)
    s_ein = np.einsum("...i,...i->...", x32, x32, dtype=np.float64)
    s_mine = np.empty(ids.size)
    okernels()["sumsq_rows"](x32, s_mine)
    rel = float(np.max(np.abs(s_ein - s_mine) / np.maximum(s_ein, 1e-300)))
    log(f"cosines: {ids.size} rows: unit_of equal {not ok['cos_unit']}; largest relative distance of the eight-lane sum from "
        f"einsum's {rel:.2e} (the band is {NORM_DELTA:.0e}, the bound 1.7e-13)")
    if rel > 1e-13:
        ok["cos_unit"] += 1
    wide = unit_of(nodes, ids, delta=1e-8)                           # a band that holds midpoints: the reference's own form
    ok["cos_fallback"] += not same_bytes(ref, wide)
    dst = np.zeros((ids.size + 7, dim), np.float32)
    pos = rng.permutation(ids.size + 7)[:ids.size]
    for dl in (NORM_DELTA, 1e-8):
        dst[:] = 0
        unit_into(nodes, ids, dst, pos, dl)
        ok["cos_fallback"] += not same_bytes(ref, dst[pos])
    diff_compact = 0
    for t in range(trials):
        sizes = [int(x) for x in rng.choice([0, 1, 3, 200, 1500, 2500], 1)] + [int(rng.choice([0, 400, 1200]))]
        B = np.sort(rng.choice(nn, 60, replace=False)).astype(np.int64)
        kt = rng.choice(nn, 1953, replace=False).astype(np.int64)
        r0 = rng.choice(nn, 400, replace=False).astype(np.int64)
        I = rng.choice(nn, 600, replace=False).astype(np.int64)
        n1 = np.setdiff1d(rng.choice(nn, sizes[0], replace=False), B).astype(np.int64)
        n2 = np.setdiff1d(np.unique(np.concatenate([rng.choice(nn, sizes[1], replace=False), kt[:300], I[:100]])), np.r_[B, n1])
        if not sizes[1]:
            n2 = n2[:0]
        first = rng.integers(0, 100, n2.size).astype(np.int64)
        q = D5B.unit(rng.standard_normal(dim))
        rows = Z.M.Rows(nodes, verify=True)
        rows.add(n1)
        cos1 = (D5B.unit(rows.read(n1)) @ q) if n1.size else np.zeros(0, np.float32)
        cr = D5B.ranks_desc(cos1, n1) if n1.size else Z.S.EMPTY
        rows.add(np.concatenate([kt, r0, I, n2]))
        full, r0o, iout = (Z.F.outside(x, B) for x in (kt, r0, I))
        want = Z.U2E.question_orders(full, n1, cr, cos1, r0o, iout, n2, first, rows.read, q, 60, rrf)[0]["kt_cos"]
        cos1x = (unit_of(nodes, n1) @ q) if n1.size else np.zeros(0, np.float32)
        ok["cos_unit"] += not same_bytes(cos1, cos1x)
        got = kt_cos_x(nodes, q, full, n1, cos1x, r0o, iout, n2, 60, rrf)
        ok["cos_kt"] += not same_bytes(want, got)
        rest = np.setdiff1d(np.unique(np.concatenate([full, n1, r0o, iout, n2])), n1)
        need = rest[np.isin(rest, full) | np.isin(rest, r0o) | np.isin(rest, iout)]
        if need.size and rest.size != need.size:
            full_c = D5B.unit(rows.read(rest)) @ q
            comp_c = D5B.unit(rows.read(need)) @ q
            diff_compact += int(not np.array_equal(full_c[np.searchsorted(rest, need)], comp_c))
    log(f"kt_cos: {trials} trials against question_orders: cos1 and the order differ {ok['cos_unit']} and {ok['cos_kt']} times; "
        f"the not-certified path differs {ok['cos_fallback']} times; compacting the matrix would have changed a cosine in "
        f"{diff_compact} of them")
    return ok


def rand_gnn(rng, n, ne, nb=30):
    """A random zg6r layer stack's inputs (the weights as a namespace with the model's parameter names; h, xb, the context
    weights, one question's edges in a random order): what g6r_layers takes."""
    R, torch = _zg6r(), C.torch
    D, FAMS, K_R = R.D, R.FAMS, R.K_R
    m = SimpleNamespace(training=False)
    for k in range(R.LAYERS):
        shapes = [(FAMS * D, D), (D, D), (FAMS,), (D, D), (D, FAMS * D), (D, D), (D, nb), (D,)]
        for nme, sh in zip(R.G6.G6_LAYER[k], shapes):
            fan = sh[-1] if len(sh) > 1 else 8
            setattr(m, nme, torch.from_numpy(((rng.random(sh) * 2 - 1) / fan ** 0.5).astype(np.float32)))
        for nme, sh in zip(R.G6R_LAYER[k], [(D, K_R), (K_R, D), (K_R,), (K_R, D)]):
            setattr(m, nme, torch.from_numpy(((rng.random(sh) * 2 - 1) / 4).astype(np.float32)))
    u, v, f = (torch.from_numpy(x.astype(np.int64)) for x in (rng.integers(0, n, ne), rng.integers(0, n, ne),
                                                              rng.integers(0, FAMS, ne)))
    h = torch.from_numpy(rng.standard_normal((n, D)).astype(np.float32))
    xb = torch.from_numpy(rng.standard_normal((n, nb)).astype(np.float32))
    w = torch.softmax(torch.from_numpy(rng.standard_normal(n).astype(np.float32)), 0).unsqueeze(1)
    return m, h, xb, w, u, v, f, torch.zeros(n, dtype=torch.long)


def layers_both(args, B=1, te=None, codes=None, cte=None, grad=False):
    """g6r_layers on the same inputs with the module's step and with the forms' step: (original, forms) outputs."""
    R, torch = _zg6r(), C.torch
    m, h, xb, w, u, v, f, nq = args
    te = torch.zeros(0, dtype=torch.long) if te is None else te
    outs = []
    for on in (False, True):
        patch_layers(on)
        with torch.set_grad_enabled(grad):
            outs.append(R.g6r_layers(m, h, xb, w, u, v, f, nq, B, te, codes, cte).detach().clone())
    patch_layers(False)
    return outs


def selftest_layers(rng, trials):
    """The forms' layer step against the module's on random inputs: float32 outputs byte for byte for graphs of 1 to 1500 nodes
    with no edge, a few, repeated targets and sources, and 40 per node; the typed-edge, two-question and gradient inputs take the
    module's own step (counted), and the forms' step was the one used otherwise (counted)."""
    torch = C.torch
    bad = {"layers": 0, "layers_fallback": 0}
    shapes = [(1, 0), (1, 6), (2, 3), (3, 40), (7, 40), (50, 400), (300, 6000), (1500, 60000)]
    STATS.update(g6r_step_x=0, g6r_step_orig=0)
    for t in range(trials):
        n, ne = shapes[t % len(shapes)]
        args = rand_gnn(rng, n, ne)
        o, x = layers_both(args)
        bad["layers"] += not torch.equal(o, x)
    used = STATS["g6r_step_x"]
    if used != trials * _zg6r().LAYERS:
        bad["layers"] += 1
    args = rand_gnn(rng, 40, 300)
    te = torch.tensor([0, 2, 5], dtype=torch.long)
    codes = torch.from_numpy(rng.standard_normal((2, _zg6r().K_R)).astype(np.float32))
    STATS.update(g6r_step_x=0, g6r_step_orig=0)
    o, x = layers_both(args, te=te, codes=codes, cte=torch.tensor([0, 1, 0], dtype=torch.long))
    bad["layers_fallback"] += not torch.equal(o, x)
    m, h, xb, w, u, v, f, nq = args
    nq2 = torch.from_numpy((np.arange(40) % 2).astype(np.int64))
    o, x = layers_both((m, h, xb, torch.softmax(torch.randn(40), 0).unsqueeze(1), u, v, f, nq2), B=2)
    bad["layers_fallback"] += not torch.equal(o, x)
    o, x = layers_both(args, grad=True)
    bad["layers_fallback"] += not torch.equal(o, x)
    bad["layers_fallback"] += (STATS["g6r_step_x"] != 0) or (STATS["g6r_step_orig"] != 3 * _zg6r().LAYERS)
    log(f"layers: {trials} graphs: the forms' step differs from the module's in {bad['layers']}; typed-edge, two-question and "
        f"gradient inputs took the module's step ({STATS['g6r_step_orig']} of {3 * _zg6r().LAYERS} layers) and differ in "
        f"{bad['layers_fallback']}")
    return bad


def selftest_installed(rng, n=2000):
    """installed() on a stand-in setup (a random graph's unions): the patched setup builds the records after the original setup and
    refuses --full-rule, the stage's rule and the layer step are the forms' inside, and every name is put back on exit, also after
    an error inside the block."""
    R = _zg6r()
    unions, ty = rand_graph(n, rng)
    S0 = SimpleNamespace(cx=SimpleNamespace(unions=unions), ctx=SimpleNamespace(ty=ty))
    asked = []

    def stand_in(ds_name, nq, full_rule=False):
        asked.append((ds_name, nq, full_rule))
        return S0

    keep = (C.setup, C.lean_rule, C.OUT, C.RAW, R.g6r_step)
    bad = {"installed": 0}
    C.setup = stand_in
    try:
        for fail in (False, True):
            inside = None
            try:
                with installed(C.OUT / "stand_in") as P:
                    S = C.setup("d", 3)
                    refused = False
                    try:
                        C.setup("d", 3, True)
                    except SystemExit:
                        refused = True
                    inside = (S is S0 and hasattr(S0, "fast") and S0.fast_index["bytes"] > 0 and P.S is S0 and refused
                              and STATS["setup_fast"] == 1 and C.setup is not stand_in and C.lean_rule is not keep[1]
                              and R.g6r_step is g6r_step_x and str(C.RAW).endswith("stand_in/raw".replace("/", os.sep)))
                    if fail:
                        raise RuntimeError("an error inside the block")
            except RuntimeError:
                pass
            back = (C.setup is stand_in and C.lean_rule is keep[1] and C.OUT == keep[2] and C.RAW == keep[3]
                    and R.g6r_step is keep[4] and not _ORIG.get("installed"))
            bad["installed"] += not (inside and back)
            if hasattr(S0, "fast"):
                del S0.fast, S0.fast_index
    finally:
        C.setup = keep[0]
    log(f"installed: the forms in the process and put back on exit, with and without an error inside: {bad['installed']} wrong")
    return bad


# ── the micro-benchmark (development: which form of a walk is faster, on a random graph) ─────────────────────────────────


def best(fn, reps):
    ts = []
    r = None
    for _ in range(reps):
        c0 = clock()
        r = fn()
        ts.append(clock() - c0)
    return min(ts) * 1000.0, float(np.median(ts)) * 1000.0, r


def micro(a):
    """Time per walk on a random graph: the modules' batch kernels (arrays made per call) against the walks on P4b's records
    (unweighted, steered, relational, relational steered). Prints min/median ms per form and whether the output is equal."""
    C._load()
    rng = np.random.default_rng(11)
    n = int(a.nodes)
    S4, B, E, Kc, A = C.Kx.S4, C.Kx.B, C.E, C.Kx, C.Z.A
    unions, ty = rand_graph(n, rng, (float(a.sdeg), 3.0, 5.0))
    alpha, eps, bmax = float(E.ALPHA), float(S4.EPS), int(Kc.NMAX)
    wq = rng.random(5) + 0.2
    reps = int(a.trials)
    wk = Fast(unions, ty[0])
    log(f"p4b form built, {wk.nbytes() / 2 ** 20:.0f} MiB")
    for ns in (3, 40):
        seeds = np.unique(rng.integers(0, n, ns)).astype(np.int64)
        w = rng.random(seeds.size) + 0.1
        restart = (seeds, w / w.sum())
        stn = np.unique(rng.integers(0, n, 300)).astype(np.int64)
        steer = (stn, rng.random(stn.size) * 0.05, 0.004)
        for un in ("st", "all"):
            u = unions[un]
            cases = {
                "unw": (lambda: B.walk_mass(u, [restart], 1)[0][0], lambda: wk.walk(un, restart)),
                "stee": (lambda: Kc.twalk(u, [restart], [steer], 1)[0], lambda: wk.twalk(un, restart, steer)),
                "rel": (lambda: A.rwalk(u, [restart], None, [wq], 1, ty, alpha, eps, bmax, 1)[0][0],
                        lambda: wk.rwalk(un, restart, None, wq, alpha, eps, bmax)),
                "rel+s": (lambda: A.rwalk(u, [restart], [steer], [wq], 1, ty, alpha, eps, bmax, 1)[0][0],
                          lambda: wk.rwalk(un, restart, steer, wq, alpha, eps, bmax)),
            }
            for name, (orig_fn, p4b_fn) in cases.items():
                orig_fn()
                p4b_fn()                                        # compile and pages
                o_min, o_med, ro = best(orig_fn, reps)
                p_min, p_med, rp = best(p4b_fn, reps)
                log(f"{name:<6} seeds {seeds.size:>3} {un:<3} out {np.asarray(ro).size:>5}  orig {o_min:7.2f}/{o_med:7.2f}  "
                    f"p4b {p_min:7.2f}/{p_med:7.2f}  x{o_med / max(p_med, 1e-9):4.2f}  equal {same_bytes(ro, rp)}")
    return 0


# ── the profile ─────────────────────────────────────────────────────────────


def pin_dev(lp):
    """Development stays off logical processors 2 and 4, where the timed passes run (and off their siblings 3 and 5): one
    other processor, EcoQoS opted out, priority unchanged. Returns what was done."""
    if os.name != "nt" or lp < 0:
        return {"pinned": False, "why": "not asked or not Windows"}
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    HANDLE = ctypes.c_void_p
    k32.GetCurrentProcess.restype = HANDLE
    k32.GetCurrentProcess.argtypes = []
    k32.GetProcessAffinityMask.argtypes = [HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k32.GetProcessAffinityMask.restype = ctypes.c_int
    k32.SetProcessAffinityMask.argtypes = [HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = ctypes.c_int
    k32.SetProcessInformation.argtypes = [HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    k32.SetProcessInformation.restype = ctypes.c_int
    h = k32.GetCurrentProcess()
    pm, sm = ctypes.c_size_t(), ctypes.c_size_t()
    if not k32.GetProcessAffinityMask(h, ctypes.byref(pm), ctypes.byref(sm)):
        return {"pinned": False, "why": "GetProcessAffinityMask failed"}
    allowed = [i for i in range(64) if (pm.value >> i) & 1 and i not in (2, 3, 4, 5)]
    if not allowed:
        return {"pinned": False, "why": "no processor outside 2 to 5", "mask": int(pm.value)}
    pick = lp if lp in allowed else allowed[len(allowed) // 2]
    out = {"allowed": allowed, "lp": pick, "affinity_ok": bool(k32.SetProcessAffinityMask(h, 1 << pick))}
    st = C.C1._PTS(1, 0x1, 0)
    out["ecoqos_off_ok"] = bool(k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))
    out["pinned"] = out["affinity_ok"]
    return out


def inputs_of(S, j):
    """Stage (a)'s inputs to the rule, as zfsvc.stage_a builds them: today's pool, base and seeds, the text links."""
    d, s = S.d[j], S.s[j]
    I, seeds, _bucket = S.fpool(d, s)
    Bq = np.union1d(C.base_of(S, d, s), seeds)
    linked = C.E.link(S.texts[j], S.index) if S.index is not None else C.EMPTY
    return np.array(I, np.int64), Bq, np.asarray(linked, np.int64)


PROF_ORDER = ("sources", "steered", "restarts", "steer", "omega", "rwalk_A1st", "rwalk_A1all", "rwalk_Kst", "rwalk_KTst",
              "rwalk_KTall", "route_kind", "scaled", "p1e_orders1", "x_unw", "x_rel", "x_ord1", "n1_table", "cos1", "ranks",
              "second_link", "rows_add", "outside", "question_orders", "kept", "p1e_orders2", "x_ord2", "kt_cols")


def stat(v):
    v = np.asarray(v, np.float64) * 1000.0
    return float(np.median(v)), float(v.mean()), float(np.percentile(v, 90))


def profile(a):
    C._load()
    log(f"pin: {pin_dev(a.lp)}")
    S = C.open_a(a.dataset, a.questions)
    log(f"threads {C.set_threads(C.BLAS)}")
    nq = a.questions
    ins = [inputs_of(S, j) for j in range(nq + 1)]
    n = int(S.cx.unions["st"].deg.size)
    if int(S.cx.unions["all"].deg.size) != n:
        raise SystemExit("the two unions hold different node counts")
    t0 = clock()
    S.fast = Fast({k: S.cx.unions[k] for k in ("st", "all")}, S.ctx.ty[0])
    log(f"P4b records: {n} nodes, {S.fast.nbytes() / 2 ** 20:.0f} MiB; index time {clock() - t0:.1f} s (numbering "
        f"{S.fast.build[0]:.1f} s, records and classes {S.fast.build[1]:.1f} s)")
    t0 = clock()
    for j in range(nq + 1):                                     # index time, untimed: the compile and the pages, as zfsvc.prepass
        C.lean_rule(S, j, ins[j][2], ins[j][0], ins[j][1], {})
        lean_rule_x(S, j, ins[j][2], ins[j][0], ins[j][1], {})
    log(f"warm pass on {nq + 1} questions ({clock() - t0:.0f} s)")
    gc.collect()
    forms = ("orig", "prof", "x")
    tot = {f: [] for f in forms}
    seg = {f: {} for f in forms}
    allok = True
    for j in range(nq):
        I, Bq, linked = ins[j]
        T, X, out = {f: {} for f in forms}, {f: {} for f in forms}, {}
        for f in (forms if j % 2 == 0 else forms[::-1]):
            c0 = clock()
            if f == "orig":
                out[f] = C.lean_rule(S, j, linked, I, Bq, T[f])
            elif f == "prof":
                out[f] = prof_rule(S, j, linked, I, Bq, T[f], X[f])
            else:
                out[f] = lean_rule_x(S, j, linked, I, Bq, T[f], X[f])
            tot[f].append(clock() - c0)
            for k, v in T[f].items():
                seg[f].setdefault(k, []).append(v)
        ok = {"prof": same_out(out["orig"], out["prof"]), "x": same_out(out["orig"], out["x"]),
              "x_inner": same_x(X["prof"], X["x"])}
        allok &= all(ok.values())
        log(f"q{j} n={out['orig'][0].size} orig {tot['orig'][-1] * 1000:.1f} (walk {T['orig']['a_walk'] * 1000:.1f} order "
            f"{T['orig']['a_order'] * 1000:.1f} kt {T['orig']['a_kt'] * 1000:.1f}) prof {tot['prof'][-1] * 1000:.1f} "
            f"x {tot['x'][-1] * 1000:.1f} (unw {T['x']['x_unw'] * 1000:.1f} rel {T['x']['x_rel'] * 1000:.1f} "
            f"ord {(T['x']['x_ord1'] + T['x']['x_ord2']) * 1000:.1f}) equal {ok}")
    rows = []
    for k in PROF_ORDER:
        r = {"seg": k}
        for f in ("prof", "x"):
            if k in seg[f]:
                r[f] = stat(seg[f][k])
        rows.append(r)
    log("segment: p50 / mean / p90 (ms)")
    for r in rows:
        log(f"  {r['seg']:<16} prof " + (" ".join(f"{x:7.1f}" for x in r["prof"]) if "prof" in r else "      -") + "   x " +
            (" ".join(f"{x:7.1f}" for x in r["x"]) if "x" in r else "      -"))
    for k in ("a_walk", "a_order", "a_kt"):
        log(f"  orig {k:<10} " + " ".join(f"{x:7.1f}" for x in stat(seg["orig"][k])))
    s_o, s_p, s_x = stat(tot["orig"]), stat(tot["prof"]), stat(tot["x"])
    log(f"TOTAL p50/mean/p90 ms: orig {s_o[0]:.1f}/{s_o[1]:.1f}/{s_o[2]:.1f}  prof {s_p[0]:.1f}/{s_p[1]:.1f}/{s_p[2]:.1f}  "
        f"x {s_x[0]:.1f}/{s_x[1]:.1f}/{s_x[2]:.1f}; speed-up p50 {s_o[0] / s_x[0]:.2f}x mean {s_o[1] / s_x[1]:.2f}x")
    log("PROFILE_JSON " + json.dumps({"dataset": a.dataset, "questions": nq, "blas": C.BLAS, "orig": s_o, "prof": s_p, "x": s_x,
                                      "seg": {r["seg"]: {f: r[f] for f in ("prof", "x") if f in r} for r in rows}}))
    log("PROFILE " + ("PASS" if allok else "FAIL"))
    return 0 if allok else 1


def log_prof(pr, title, n=12):
    """torch's profiler: the ops by self CPU time."""
    ev = list(pr.key_averages())
    tot = sum(e.self_cpu_time_total for e in ev) or 1.0
    log(f"profile {title}: {tot / 1000:.1f} ms of self time in {sum(e.count for e in ev)} calls; the ops by self time:")
    for e in sorted(ev, key=lambda e: -e.self_cpu_time_total)[:n]:
        log(f"   {e.key:<36} {e.count:>5} calls {e.self_cpu_time_total / 1000:8.2f} ms {100 * e.self_cpu_time_total / tot:5.1f}%")


def model_line(S, key):
    """One model's shape, for the development log: its class, blocks and widths, context mode, hidden size and parameters."""
    try:
        m = S.models[key]
        w = getattr(m, "widths", None)
        h = getattr(getattr(m, "l1", None), "out_features", None)
        return (f"{key}: {type(m).__name__}, blocks {S.blocks[key]}, widths {w if not isinstance(w, dict) else dict(w)}, ctx "
                f"{getattr(m, 'ctx_mode', None)}, hidden {h}, {sum(int(p.numel()) for p in m.parameters())} parameters")[:600]
    except Exception as e:  # noqa: BLE001
        return f"{key}: model line failed: {type(e).__name__}: {str(e)[:200]}"


def gnn_split(S, feats, nqt):
    """The GNN's forward in its parts (wall ms) with the module's layer step and with the forms': the encoder, the chain features,
    the input map and the context weights, the layers, the readout. zg6r.ZG6R.forward and g6r_stack, line for line (no typed edge,
    one question, evaluation); returns {forms?: (parts, scores)}."""
    R, torch = _zg6r(), C.torch
    m = S.models["gnn"]
    keep = torch.ones((1, len(S.blocks["gnn"])), dtype=torch.float32)
    res = {}
    with torch.no_grad():
        for on in (False, True):
            patch_layers(on)
            t = [clock()]
            H = m.encode(feats, keep, nqt, 1)
            t.append(clock())
            xb = torch.cat([feats[R.KT_KEY].to(H.dtype), m.chain(feats, keep, nqt).to(H.dtype)], 1)
            t.append(clock())
            h0 = R.Fn.gelu(R.Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b))
            z0 = R.LG.seg_zscore8D(R.Fn.linear(h0, m.g6_ctx_w).squeeze(-1).unsqueeze(1), nqt, 1).squeeze(1).clamp(-R.ZCLIP, R.ZCLIP)
            w = R.G2.seg_softmax_rows(z0, nqt, 1).unsqueeze(1)
            lk, gd = feats[R.LK_KEY], feats[R.G6R_KEY]
            te = torch.nonzero(gd["ci"] >= 0).view(-1)
            t.append(clock())
            h = R.g6r_layers(m, h0, xb, w, lk["u"], lk["v"], lk["f"], nqt, 1, te, None, None)
            t.append(clock())
            s = (R.Fn.linear(h, m.g6_out_w) + R.Fn.linear(h0, m.g6_skip_w) + m.g6_out_b).squeeze(-1)
            t.append(clock())
            res[on] = ([(t[i + 1] - t[i]) * 1000 for i in range(5)], s.numpy().copy(), int(lk["u"].numel()))
    patch_layers(True)
    return res


def forward_profile(S, key, j, n, feats, nqt, base_z, scores):
    """devx: torch's profiler on one model's forward with the module's layers and, for the GNN, with the forms' too, then the GNN's
    forward in its parts. A development print: an error in it is logged and the checks go on."""
    torch = C.torch
    from torch.profiler import ProfilerActivity, profile
    try:
        patch_layers(False)
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as pr:
            C.forward(S, key, feats, nqt, base_z, 1)
        log_prof(pr, f"{key} forward, the module's, q{j} (n={n})")
        if key == "gnn":
            patch_layers(True)
            with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as pr:
                C.forward(S, key, feats, nqt, base_z, 1)
            log_prof(pr, f"{key} forward, the forms', q{j} (n={n})")
            sp = gnn_split(S, feats, nqt)
            parts = "encode, chain, input map and context, layers, readout"
            log(f"q{j} gnn forward parts ms ({parts}): module's layers {' '.join(f'{x:.1f}' for x in sp[False][0])}; "
                f"forms' {' '.join(f'{x:.1f}' for x in sp[True][0])}; edges {sp[True][2]}; scores equal to the module's forward: "
                f"{same_bytes(sp[False][1], scores) and same_bytes(sp[True][1], scores)}")
    except Exception as e:  # noqa: BLE001
        log(f"q{j} {key} forward profile failed (the checks go on): {type(e).__name__}: {str(e)[:300]}")
    finally:
        patch_layers(True)


def fwdmicro(a):
    """Development, no data: torch's profiler (one thread) on the module's layers for a random graph of --nodes nodes and --edges
    edges (the ops by self time), then the module's layers against the forms': median ms and equal."""
    C._load()
    log(f"threads {C.set_threads(C.BLAS)}")
    torch = C.torch
    from torch.profiler import ProfilerActivity, profile
    R = _zg6r()
    args = rand_gnn(np.random.default_rng(3), int(a.nodes), int(a.edges))
    m, h, xb, w, u, v, f, nq = args
    te = torch.zeros(0, dtype=torch.long)
    with torch.no_grad():
        for on in (False, True):
            patch_layers(on)
            R.g6r_layers(m, h, xb, w, u, v, f, nq, 1, te, None, None)
            with profile(activities=[ProfilerActivity.CPU]) as pr:
                R.g6r_layers(m, h, xb, w, u, v, f, nq, 1, te, None, None)
            log_prof(pr, ("the forms'" if on else "the module's") + f" layers, {int(a.nodes)} nodes, {int(a.edges)} edges")
        ts = {False: [], True: []}
        outs = {}
        for rep in range(9):
            for on in ((False, True) if rep % 2 == 0 else (True, False)):
                patch_layers(on)
                c0 = clock()
                outs[on] = R.g6r_layers(m, h, xb, w, u, v, f, nq, 1, te, None, None)
                ts[on].append(clock() - c0)
        patch_layers(False)
    o, x = (float(np.median(ts[k])) * 1000 for k in (False, True))
    log(f"layers p50: module's {o:.1f} ms, forms' {x:.1f} ms, x{o / x:.2f}; equal {bool(torch.equal(outs[False], outs[True]))}")
    return 0


def devx(a):
    """A2's development checks (zfsvc.dev_checks: the pool, the KT columns and I against PKO's rule, the features against the
    carve's, the scores against the filed reads within 1e-4 with the same top five, the float16 NEXT flips counted) with the
    forms installed, and beside them, per question: stage (a) as zfsvc has it and as the forms give it (byte for byte equal; the
    medians), each model's forward with the module's layers and with the forms' (byte for byte equal; the medians), and torch's
    profiler on the module's forward of the first --prof questions. Development numbers: no timed pass."""
    C._load()
    log(f"pin: {pin_dev(a.lp)}")
    nq = int(a.questions)
    torch, KEYS = C.torch, C.KEYS
    with installed() as P:
        S = C.setup(a.dataset, nq)
        C.prepass(S, range(nq + 1))
        S.threads = C.set_threads(C.BLAS)
        log(f"threads {S.threads}")
        for key in KEYS:                                                    # the JIT question, both models
            A = C.stage_a(S, nq, {})
            C.fill_codes(S, nq, A.pool)
            C.run_model(S, key, nq, A, {})
        for key in KEYS:
            log(model_line(S, key))
        ins = [inputs_of(S, j) for j in range(nq + 1)]
        for j in range(nq + 1):                                             # the module's own rule: its compile and pages
            P.orig_rule(S, j, ins[j][2], ins[j][0], ins[j][1], {})
        gc.collect()
        sizes = []
        tot = {"orig": [], "x": []}
        seg, fw = {}, {k: {"orig": [], "x": []} for k in KEYS}
        a_eq = fw_eq = flips = vals = fq = 0
        failing, md, top5 = [], {k: 0.0 for k in KEYS}, {k: 0 for k in KEYS}
        for j in range(nq):
            I, Bq, linked = ins[j]
            out, T = {}, {}
            for f in (("orig", "x") if j % 2 == 0 else ("x", "orig")):
                c0 = clock()
                out[f] = P.orig_rule(S, j, linked, I, Bq, {}) if f == "orig" else lean_rule_x(S, j, linked, I, Bq, T)
                tot[f].append(clock() - c0)
            a_eq += same_out(out["orig"], out["x"])
            for k, v in T.items():
                seg.setdefault(k, []).append(v)
            A = C.stage_a(S, j, {})
            C.fill_codes(S, j, A.pool)
            outs = {key: C.run_model(S, key, j, A, {}) for key in KEYS}
            res = C.dev_checks(S, j, A, outs)
            bad = {k: v for k, v in res.items()
                   if v is False or (isinstance(v, dict) and (v.get("ok") is False or v.get("identical") is False))}
            if set(bad) == {"mlp_feats"} and set(bad["mlp_feats"]["per_key"]) == {"NEXT"}:
                bad = {}                                                    # float16 NEXT values a step off are counted
            if bad or not same_out(out["orig"], out["x"]):
                failing.append(j)
                log(f"q{j} FAILED: {json.dumps(bad, default=str)[:1200]}")
            flips += res["next_flips"]
            vals += res["next_values"]
            fq += res["next_flips"] > 0
            ne = int(outs["gnn"][1]["ne"])
            sizes.append((int(A.pool.size), ne))
            line = f"q{j} n={A.pool.size} edges={ne} (a) orig {tot['orig'][-1] * 1000:.1f} x {tot['x'][-1] * 1000:.1f} ms; forward"
            for key in KEYS:
                s, _pack, (feats, nqt, base_z) = outs[key]
                got = {}
                for f in (("orig", "x") if (j + KEYS.index(key)) % 2 == 0 else ("x", "orig")):
                    patch_layers(f == "x")
                    c0 = clock()
                    sc = C.forward(S, key, feats, nqt, base_z, 1)
                    torch.topk(sc, min(C.TOP, int(sc.numel())))
                    fw[key][f].append(clock() - c0)
                    got[f] = sc.numpy().copy()
                patch_layers(True)
                fw_eq += same_bytes(got["orig"], got["x"])
                md[key] = max(md[key], res[f"{key}_scores"]["max_diff"] or 0.0)
                top5[key] += bool(res[f"{key}_scores"]["top5_same"])
                line += f" {key} {fw[key]['orig'][-1] * 1000:.1f}/{fw[key]['x'][-1] * 1000:.1f}"
                if j < int(a.prof):
                    forward_profile(S, key, j, int(A.pool.size), feats, nqt, base_z, got["orig"])
            log(line + f"; scores max diff mlp {res['mlp_scores']['max_diff']} gnn {res['gnn_scores']['max_diff']}, NEXT flips "
                f"{res['next_flips']}/{res['next_values']}")
        if STATS["setup_fast"] != 1 or STATS["lean_rule_x"] < 3 * nq or STATS["g6r_step_x"] < nq * _zg6r().LAYERS:
            raise SystemExit(f"the forms did not run: {STATS}")
        fast_index = dict(S.fast_index)
    p50 = lambda v: float(np.median(v)) * 1000
    s_o, s_x = p50(tot["orig"]), p50(tot["x"])
    log(f"stage (a) p50 ms: zfsvc's {s_o:.1f}, the forms' {s_x:.1f}, x{s_o / s_x:.2f} (mean {np.mean(tot['orig']) * 1000:.1f} / "
        f"{np.mean(tot['x']) * 1000:.1f}, x{np.mean(tot['orig']) / np.mean(tot['x']):.2f}); equal on {a_eq} of {nq}")
    log("the forms' segments p50 ms: " + ", ".join(f"{k} {p50(v):.1f}" for k, v in seg.items()))
    sz = np.array(sizes)
    log(f"pool rows p50 {int(np.median(sz[:, 0]))} (max {int(sz[:, 0].max())}), link edges p50 {int(np.median(sz[:, 1]))} (max "
        f"{int(sz[:, 1].max())}), edges per row p50 {float(np.median(sz[:, 1] / sz[:, 0])):.1f}")
    for key in KEYS:
        o, x = p50(fw[key]["orig"]), p50(fw[key]["x"])
        log(f"{key} forward + top five p50 ms: module's layers {o:.1f}, forms' {x:.1f}, x{o / x:.2f}; scores vs the filed read: "
            f"max diff {md[key]:.3g}, same top five on {top5[key]} of {nq}")
    log(f"forwards equal byte for byte on {fw_eq} of {len(KEYS) * nq}; NEXT: {flips} float16 values of {vals} differ, in {fq} "
        f"questions; layer steps: {STATS['g6r_step_x']} by the forms, {STATS['g6r_step_orig']} by the module; norms the band could "
        f"not certify (the reference's form for that matrix): {STATS['unit_fallback']}")
    ok = not failing and a_eq == nq and fw_eq == len(KEYS) * nq and STATS["g6r_step_orig"] == 0
    log("DEVX_JSON " + json.dumps({"dataset": a.dataset, "questions": nq, "blas": C.BLAS, "fast_index": fast_index,
                                   "a": {"orig": p50(tot["orig"]), "x": s_x}, "seg": {k: p50(v) for k, v in seg.items()},
                                   "fw": {k: {"orig": p50(v["orig"]), "x": p50(v["x"])} for k, v in fw.items()},
                                   "max_diff": md, "top5_same": top5, "next_flips": flips, "next_values": vals,
                                   "sizes": sizes, "stats": dict(STATS), "failing": failing}))
    log("DEVX " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def run(a):
    """zfsvc.run (the timed passes, the checks, the raw record) with the forms installed; the raw files go to outputs/fsv/timing/a4
    (zfsvc's raw files are not touched). Track A's re-time (M3) runs this; the forms' author makes no timed number."""
    C._load()
    ns = SimpleNamespace(dataset=a.dataset, nq=a.nq, lps=a.lps, require_pin=a.require_pin, overwrite=a.overwrite, full_rule=False,
                         questions=a.questions)
    with installed(C.OUT / "a4") as P:
        rc = C.run(ns)
        nq = a.nq or C.T.NQ
        if STATS["setup_fast"] != 1 or STATS["lean_rule_x"] < nq or STATS["g6r_step_x"] < nq * _zg6r().LAYERS:
            raise SystemExit(f"the forms did not run: {STATS}")
        out_file = (C.RAW if nq == C.T.NQ else C.OUT / "raw_smoke") / f"{a.dataset}__b{C.BLAS}.json"
        d = json.load(open(out_file))
        d["head"]["forms"] = {"zfsvx_sha256": C.T.sha_src(__file__), "fast_index": P.S.fast_index, "stats": dict(STATS)}
        C.T.write_json(out_file, d)
    return rc


M3_TEXT = (("# FSV timing, Family 1, A2 forms", "# FSV timing, Family 1, A4 forms (M3)"),
           ("each at the stage forms of amendment A2 (`docs/FSV_FAST_VERSIONS.md` section 10).",
            "each at the stage forms of amendment A2 (`docs/FSV_FAST_VERSIONS.md` section 10) with amendment A4's forms of stage "
            "(a) and of the GNN's layers (section 12). This file is M3, the timing of those forms; the passes of A2's own forms "
            "(M2) stand as the record of them."),
           ("- Stage (a) is PKO's own rule on one question at batch 1 on one thread; no fused forward exists for either model "
            "(eager torch, one thread);",
            "- Stage (a) is PKO's rule's own calls on one question at batch 1 on one thread, with the walks on P4b's records and "
            "only the orders the pool and the KT columns read (amendment A4); the GNN's layer step is A4's and the MLP's forward "
            "the module's (eager torch, one thread);"))


def report(a):
    """zfsvc.report on the M3 raw files (outputs/fsv/timing/a4/raw by default), labelled A4: the reduction is zfsvc's, and the
    label, the amendment and the three phrases of its prose that name A2's forms are replaced (each must be found, or the report
    stops); the forms' own record (zfsvx's sha256, the records' index time and size, what ran) joins each dataset."""
    raw_dir = Path(a.raw_dir) if a.raw_dir else C.OUT / "a4" / "raw"
    out_dir = Path(a.out_dir) if a.out_dir else C.OUT / "a4"
    rc = C.report(SimpleNamespace(raw_dir=str(raw_dir), out_dir=str(out_dir)))
    f = out_dir / "report.json"
    rep = json.loads(f.read_text(encoding="utf-8"))
    rep["label"] = "A4 forms (M3)"
    rep["amendment"] = "docs/FSV_FAST_VERSIONS.md section 12 (A4)"
    for tag, P in rep["passes"].items():
        for ds, d in P["datasets"].items():
            head = json.loads((raw_dir / f"{ds}__{tag}.json").read_text(encoding="utf-8"))["head"]
            if "forms" not in head:
                raise SystemExit(f"{ds}__{tag}: the raw file does not record the forms (it is not a zfsvx run)")
            d["forms"] = head["forms"]
    C.T.write_json(f, rep)
    md = C._md(rep)
    for old, new in M3_TEXT:
        if old not in md:
            raise SystemExit(f"zfsvc's report text changed: not found: {old[:60]}")
        md = md.replace(old, new)
    (out_dir / "report.md").write_text(md, encoding="utf-8", newline="\n")
    log(f"M3 report: {out_dir / 'report.json'}, report.md")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "profile", "micro", "cos", "fwdmicro", "devx", "run", "report"))
    ap.add_argument("--raw-dir", dest="raw_dir", default="", help="report: the raw files (default outputs/fsv/timing/a4/raw)")
    ap.add_argument("--out-dir", dest="out_dir", default="", help="report: where report.json and report.md go (a4)")
    ap.add_argument("--edges", type=int, default=40000)
    ap.add_argument("--prof", type=int, default=2, help="devx: torch's profiler on this many questions' forwards")
    ap.add_argument("--nq", type=int, default=0, help="run: a smoke run of this many questions, 0 = the 200")
    ap.add_argument("--lps", default="2,4", help="run: the logical processors the process is pinned to")
    ap.add_argument("--require-pin", dest="require_pin", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dataset", default="musique")
    ap.add_argument("--questions", type=int, default=40)
    ap.add_argument("--blas", default=C.BLAS, help="OpenBLAS threads; zfsvc reads it from argv too (default there: 1)")
    ap.add_argument("--lp", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=30000)
    ap.add_argument("--sdeg", type=float, default=8.0)
    ap.add_argument("--trials", type=int, default=25)
    a = ap.parse_args(argv)
    if a.cmd == "cos":
        C._load()
        bad = selftest_cos(np.random.default_rng(5), int(a.trials))
        log("COS " + ("PASS" if not any(bad.values()) else "FAIL"))
        return 0 if not any(bad.values()) else 1
    return {"selftest": selftest, "profile": profile, "micro": micro, "fwdmicro": fwdmicro, "devx": devx, "run": run,
            "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
