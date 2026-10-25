"""U2q (docs/U2Q_ARRIVAL_SPLIT_CARRIED.md): U2m (d)'s mass split by arrival type, on the carried walk. An input for
U2n's models, never a pool rule: nothing trains, nothing is fit, and only the beside table reads golds.

    python outputs/mp_unified/zu2q.py stats DATASET [--threads 2] [--limit Q [--dev-a]] [--shard I/K]
    python outputs/mp_unified/zu2q.py merge DATASET --parts K
    python outputs/mp_unified/zu2q.py beside
    python outputs/mp_unified/zu2q.py --selftest

stats    One dataset's carve (s1sel; webqsp's s1eval, beside), after U2p's verdict is filed. The carried walk is the
         one U2p's verdict carries (section 2): U2p's pair weight at its beta on the passage graphs (beta = 0, (a)'s
         walk, when NO_GAIN) and (a)'s RWe at beta 0.5 on the KBs. Each question walks the five walks kind_t reads
         with (a)'s kernel, the carried pick's rows are rebuilt (B and order[:keep]), and the five walks run again
         with the split kernel, which keeps each arrival type's mass for those rows. Checked: the split walks' orders
         and raw masses equal (a)'s kernel's bit for bit; the parts sum to p (on the KB, the (relation, direction)
         parts to the structural part); the rows and held golds equal the carried pick's filed ones. --limit Q: the
         first Q questions, every check, nothing written and no number logged. --dev-a (with --limit, before U2p's
         verdict only): (a)'s pick, U2p's beta = 0. --shard I/K: the I-th of K contiguous question ranges.
merge    Join a dataset's K parts into the file one run over every question writes.
beside   The beside table over the datasets written, with (d)'s filed shares beside, deciding nothing:
         outputs/u2q/beside.json and .md.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zu2m as M  # noqa: E402  (U2m's walk, cells, columns, rows and arrays, unchanged)
import zu2m_b as MB  # noqa: E402  (section 11's carried frontier and plan, unchanged)
import zu2m_d as MD  # noqa: E402  (section 13's split bookkeeping, arrays and beside, unchanged)
import zu2m_a as A  # noqa: E402  (section 14's relational kernel, omega and filed arrays, unchanged)
import zu2m_a_types as AT  # noqa: E402  (section 14's filed EM types, unchanged)
import zu2p as P  # noqa: E402  (U2p's score types, pair weight, filed arrays and verdict, unchanged)

U2E, U2C, F, P1H, D5, D5B = M.U2E, M.U2C, M.F, M.P1H, M.D5, M.D5B
log, utc, write_json, sha_file = M.log, M.utc, M.write_json, M.sha_file
DECLARED = "docs/U2Q_ARRIVAL_SPLIT_CARRIED.md"
OUT = ROOT / "outputs" / "u2q"
SIX, NAMES, BASES = M.SIX, M.NAMES, M.BASES
EMPTY = M.EMPTY
WALKS, WALK_UNION, TYPES = MD.WALKS, MD.WALK_UNION, MD.TYPES
REL_TOL, ABS_TOL = MD.REL_TOL, MD.ABS_TOL
SLOT_BUDGET = MD.SLOT_BUDGET
BATCH = A.BATCH_A
A_BETA = P.A_FORM[1]                                  # (a)'s pick: RWe at beta 0.5


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the split kernel ((a)'s relational push, with (d)'s arrivals at pool rows kept beside r) ──────────────────────


_K = {}


def kernels():
    """The weighted split kernel, compiled on first use."""
    if _K:
        return _K
    from numba import njit, prange
    import step4_pool_reach as S4
    _top = S4._top

    @njit(inline="always")
    def _w(e, v, mode, cls, bst, bln, trel, wq, steered, floor, ex):
        """(a)'s edge weight, statement for statement: omega (mode 1: its type's; mode 2: the mean over its facts;
        mode 0: 1) times the walk's own s(v)."""
        w = 1.0
        if mode == 1:
            w = wq[cls[e]]
        elif mode == 2:
            s = 0.0
            a = bst[e]
            for t in range(a, a + bln[e]):
                s += wq[trel[t]]
            w = s / bln[e]
        if steered:
            w = w * (floor + ex[v])
        return w

    @njit
    def _arrive(v, u, amt, fam, ridx, pend, typed, tip, tcol, toff, rpend, nxt, dhead, inl):
        """(d)'s arrival, statement for statement."""
        k = ridx[v]
        if k < 0:
            return
        pend[k, fam + 1] += amt
        if typed and fam == 0:
            a = tip[v]
            b = tip[v + 1]
            lo, hi = a, b
            while lo < hi:
                mid = (lo + hi) // 2
                if tcol[mid] < u:
                    lo = mid + 1
                else:
                    hi = mid
            e0 = lo
            hi = b
            while lo < hi:
                mid = (lo + hi) // 2
                if tcol[mid] <= u:
                    lo = mid + 1
                else:
                    hi = mid
            c = lo - e0
            if c > 0:
                each = amt / c
                for e in range(e0, lo):
                    s = toff[k] + (e - a)
                    if inl[s] == 0:
                        inl[s] = 1
                        nxt[s] = dhead[k]
                        dhead[k] = s
                    rpend[s] += each

    @njit
    def _settle(u, alpha, ridx, pend, parts, row0, typed, rpend, rsett, slot0, nxt, dhead, inl):
        """(d)'s settle, statement for statement."""
        k = ridx[u]
        if k < 0:
            return
        for t in range(4):
            parts[row0 + k, t] += alpha * pend[k, t]
            pend[k, t] = 0.0
        if typed:
            s = dhead[k]
            while s >= 0:
                rsett[slot0 + s] += alpha * rpend[s]
                rpend[s] = 0.0
                inl[s] = 0
                s2 = nxt[s]
                nxt[s] = -1
                s = s2
            dhead[k] = -1

    @njit
    def _wspread(ip, cc, fam, u, m, nf, mode, cls, bst, bln, trel, wq, steered, floor, ex, r, flag, touched, queue,
                 deg, eps, n, nt, tail, ridx, pend, typed, tip, tcol, toff, rpend, nxt, dhead, inl):
        """(a)'s _rspread, the same statements on r in the same order, with the arrivals at pool rows kept."""
        a = ip[u]
        b = ip[u + 1]
        if b <= a:
            return nt, tail
        tot = 0.0
        for e in range(a, b):
            tot += _w(e, cc[e], mode, cls, bst, bln, trel, wq, steered, floor, ex)
        base = m / (nf * tot)
        for e in range(a, b):
            v = cc[e]
            if flag[v] & 2 == 0:
                flag[v] |= 2
                touched[nt] = v
                nt += 1
            amt = base * _w(e, v, mode, cls, bst, bln, trel, wq, steered, floor, ex)
            r[v] += amt
            if flag[v] & 1 == 0 and r[v] >= eps * max(deg[v], 1):
                flag[v] |= 1
                queue[tail % n] = v
                tail += 1
            if ridx[v] >= 0:
                _arrive(v, u, amt, fam, ridx, pend, typed, tip, tcol, toff, rpend, nxt, dhead, inl)
        return nt, tail

    @njit
    def _wpush(seeds, wts, ip0, c0, ip1, c1, ip2, c2, deg, mode, cls, bst, bln, trel, wq, steered, floor, ex, alpha,
               eps, p, r, flag, touched, queue, ridx, pend, parts, row0, typed, tip, tcol, toff, rpend, rsett, slot0,
               nxt, dhead, inl):
        """(a)'s _rpush, the same statements on p and r in the same order, with (d)'s settle at each push."""
        n = deg.size
        nt = 0
        tail = 0
        for k in range(seeds.size):
            s = seeds[k]
            if flag[s] & 2 == 0:
                flag[s] |= 2
                touched[nt] = s
                nt += 1
            r[s] += wts[k]
            if ridx[s] >= 0:
                pend[ridx[s], 0] += wts[k]
        for k in range(seeds.size):
            s = seeds[k]
            if flag[s] & 1 == 0:
                flag[s] |= 1
                queue[tail % n] = s
                tail += 1
        head = 0
        while head < tail:
            u = queue[head % n]
            head += 1
            flag[u] &= 254
            du = deg[u]
            ru = r[u]
            if ru < eps * max(du, 1):
                continue
            r[u] = 0.0
            p[u] += alpha * ru
            _settle(u, alpha, ridx, pend, parts, row0, typed, rpend, rsett, slot0, nxt, dhead, inl)
            if du == 0:
                continue
            m = (1.0 - alpha) * ru
            nf = 0
            if ip0[u + 1] > ip0[u]:
                nf += 1
            if ip1[u + 1] > ip1[u]:
                nf += 1
            if ip2[u + 1] > ip2[u]:
                nf += 1
            nt, tail = _wspread(ip0, c0, 0, u, m, nf, mode, cls, bst, bln, trel, wq, steered, floor, ex, r, flag,
                                touched, queue, deg, eps, n, nt, tail, ridx, pend, typed, tip, tcol, toff, rpend, nxt,
                                dhead, inl)
            nt, tail = _wspread(ip1, c1, 1, u, m, nf, 0, cls, bst, bln, trel, wq, steered, floor, ex, r, flag,
                                touched, queue, deg, eps, n, nt, tail, ridx, pend, typed, tip, tcol, toff, rpend, nxt,
                                dhead, inl)
            nt, tail = _wspread(ip2, c2, 2, u, m, nf, 0, cls, bst, bln, trel, wq, steered, floor, ex, r, flag,
                                touched, queue, deg, eps, n, nt, tail, ridx, pend, typed, tip, tcol, toff, rpend, nxt,
                                dhead, inl)
        return nt

    @njit(parallel=True)
    def wsplit_batch(seeds_ptr, seeds_flat, w_flat, st_ptr, st_nodes, st_ex, floors, steered, wq_ptr, wq_flat, mode,
                     cls, bst, bln, trel, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, bmax, n_chunks, rows_ptr,
                     rows_flat, typed, tip, tcol, slot_ptr, out_nodes, out_mass, out_count, p_rows, parts, rsett):
        nq = seeds_ptr.size - 1
        n = deg.size
        for t in prange(n_chunks):
            lo = (nq * t) // n_chunks
            hi = (nq * (t + 1)) // n_chunks
            if lo < hi:
                mr = 1
                ms = 1
                for q in range(lo, hi):
                    mr = max(mr, rows_ptr[q + 1] - rows_ptr[q])
                    ms = max(ms, slot_ptr[q + 1] - slot_ptr[q])
                p = np.zeros(n)
                r = np.zeros(n)
                flag = np.zeros(n, dtype=np.uint8)
                touched = np.empty(n, dtype=np.int32)
                queue = np.empty(n, dtype=np.int32)
                ex = np.zeros(n)
                ridx = np.full(n, -1, dtype=np.int64)
                pend = np.zeros((mr, 4))
                toff = np.zeros(mr + 1, dtype=np.int64)
                dhead = np.full(mr, -1, dtype=np.int64)
                rpend = np.zeros(ms)
                nxt = np.full(ms, -1, dtype=np.int64)
                inl = np.zeros(ms, dtype=np.uint8)
                for q in range(lo, hi):
                    r0 = rows_ptr[q]
                    nr = rows_ptr[q + 1] - r0
                    toff[0] = 0
                    for k in range(nr):
                        v = rows_flat[r0 + k]
                        ridx[v] = k
                        if typed:
                            toff[k + 1] = toff[k] + (tip[v + 1] - tip[v])
                        else:
                            toff[k + 1] = 0
                    seeds = seeds_flat[seeds_ptr[q]:seeds_ptr[q + 1]]
                    if seeds.size == 0:
                        out_count[q] = 0
                    else:
                        if steered:
                            for k in range(st_ptr[q], st_ptr[q + 1]):
                                ex[st_nodes[k]] = st_ex[k]
                        nt = _wpush(seeds, w_flat[seeds_ptr[q]:seeds_ptr[q + 1]], ip0, c0, ip1, c1, ip2, c2, deg,
                                    mode, cls, bst, bln, trel, wq_flat[wq_ptr[q]:wq_ptr[q + 1]], steered, floors[q],
                                    ex, alpha, eps, p, r, flag, touched, queue, ridx, pend, parts, r0, typed, tip,
                                    tcol, toff, rpend, rsett, slot_ptr[q], nxt, dhead, inl)
                        _avail, cnt = _top(p, flag, touched, nt, bmax, out_nodes[q])
                        out_count[q] = cnt
                        for k in range(cnt):
                            out_mass[q, k] = p[out_nodes[q, k]]
                        for k in range(nr):
                            p_rows[r0 + k] = p[rows_flat[r0 + k]]
                        for k in range(nt):
                            v = touched[k]
                            p[v] = 0.0
                            r[v] = 0.0
                            flag[v] = 0
                        if steered:
                            for k in range(st_ptr[q], st_ptr[q + 1]):
                                ex[st_nodes[k]] = 0.0
                    for k in range(nr):
                        ridx[rows_flat[r0 + k]] = -1
                        for tt in range(4):
                            pend[k, tt] = 0.0
                        dhead[k] = -1
                    for s in range(toff[nr]):
                        rpend[s] = 0.0
                        nxt[s] = -1
                        inl[s] = 0

    _K.update(wsplit_batch=wsplit_batch)
    return _K


def split_walk(union, restarts, steers, wqs, mode, ty, rows, typed, alpha, eps, bmax, threads):
    """Per question: (order, its raw masses, p on the rows, the rows' four parts, the settled (relation, direction)
    slots or None). As zu2m_a.rwalk for steers, wqs, mode and ty (component per entry, block start, block length,
    typed relation); rows: each question's pool rows; typed: (tip, tcol, trel, tdir) or None."""
    import step4_pool_reach as S4
    kn = kernels()
    nq = len(restarts)
    out = [None] * nq
    if nq == 0:
        return out
    cls, bst, bln, trel = ty
    tip = typed[0] if typed is not None else np.zeros(1, np.int64)
    tcol = np.ascontiguousarray(typed[1]) if typed is not None else np.zeros(0, np.int32)
    slots = np.asarray([int((tip[np.asarray(rw, np.int64) + 1] - tip[np.asarray(rw, np.int64)]).sum())
                        if typed is not None else 0 for rw in rows], np.int64)
    groups, cur, tot = [], [], 0
    for i in range(nq):
        if cur and tot + slots[i] > SLOT_BUDGET:
            groups.append(cur)
            cur, tot = [], 0
        cur.append(i)
        tot += int(slots[i])
    groups.append(cur)
    for g in groups:
        s_ptr, s_flat = S4.flat([np.asarray(restarts[i][0], np.int64) for i in g])
        w_flat = np.concatenate([np.asarray(restarts[i][1], np.float64) for i in g])
        if steers is None:
            t_ptr, t_nodes = np.zeros(len(g) + 1, np.int64), np.zeros(0, np.int64)
            t_ex, floors = np.zeros(0), np.ones(len(g))
        else:
            t_ptr, t_nodes = S4.flat([np.asarray(steers[i][0], np.int64) for i in g])
            t_ex = np.concatenate([np.asarray(steers[i][1], np.float64) for i in g])
            floors = np.asarray([steers[i][2] for i in g], np.float64)
        q_ptr = np.concatenate([[0], np.cumsum([np.asarray(wqs[i]).size for i in g])]).astype(np.int64)
        wq_flat = np.concatenate([np.asarray(wqs[i], np.float64) for i in g]) if q_ptr[-1] else np.ones(1)
        r_ptr, r_flat = S4.flat([np.asarray(rows[i], np.int64) for i in g])
        sl_ptr = np.concatenate([[0], np.cumsum(slots[g])]).astype(np.int64)
        on = np.full((len(g), bmax), -1, dtype=np.int64)
        om = np.zeros((len(g), bmax))
        oc = np.zeros(len(g), dtype=np.int64)
        p_rows = np.zeros(r_flat.size)
        parts = np.zeros((r_flat.size, 4))
        rsett = np.zeros(max(int(sl_ptr[-1]), 1))
        kn["wsplit_batch"](s_ptr, s_flat, w_flat, t_ptr, t_nodes, t_ex, floors, steers is not None, q_ptr, wq_flat,
                           int(mode), cls, bst, bln, trel, *union.args, float(alpha), float(eps), int(bmax),
                           max(1, min(len(g), 4 * threads)), r_ptr, r_flat, typed is not None,
                           np.asarray(tip, np.int64), tcol, sl_ptr, on, om, oc, p_rows, parts, rsett)
        for gi, i in enumerate(g):
            c = int(oc[gi])
            a, b = int(r_ptr[gi]), int(r_ptr[gi + 1])
            sl = rsett[int(sl_ptr[gi]):int(sl_ptr[gi + 1])].copy() if typed is not None else None
            out[i] = (on[gi, :c].copy(), om[gi, :c].copy(), p_rows[a:b].copy(), parts[a:b].copy(), sl)
    return out


# ── the carried walk (section 2) ─────────────────────────────────────────────


def carried_walk(dev_a=False):
    """The walk section 2 fixes: U2p's filed verdict's pick (its point and beta), or with dev_a (development runs
    before U2p's verdict only) (a)'s pick at U2p's beta = 0."""
    _vrec, names, G_, _vb, va, ca = P.carried()
    a_cfg = A.config_name(ca)
    vp_path = P.OUT / "verdict.json"
    if not vp_path.exists():
        if not dev_a:
            raise SystemExit("U2p's verdict is not filed (outputs/u2p/verdict.json): U2q runs after it (section 2); "
                             "a development run may pass --dev-a")
        point = P.CARRIED[0].rsplit("|", 2)[0]
        return {"verdict_p": None, "names": names, "G": G_, "va": va, "ca": ca, "pi": names.index(point), "ci": 0,
                "beta": 0.0, "pick": P.CARRIED[0], "a_cfg": a_cfg}
    if dev_a:
        raise SystemExit("U2p's verdict is filed: U2q reads the walk it carries, not --dev-a")
    vp = json.loads(vp_path.read_text(encoding="utf-8"))
    if vp.get("script_sha256") != P.script_sha256() or vp.get("verdict_a_sha256") != sha_file(A.OUT / "verdict_a.json"):
        raise SystemExit("outputs/u2p/verdict.json is not the committed zu2p.py's verdict on (a)'s")
    if vp["points"] != names or int(vp["G"]) != int(G_):
        raise SystemExit("U2p's verdict is not on the carried family's points and G")
    if vp["verdict"] == "NO_GAIN":
        if vp["pick"] != P.CARRIED[0]:
            raise SystemExit("U2p's NO_GAIN does not carry (a)'s pick")
        point, ci = P.CARRIED[0].rsplit("|", 2)[0], 0
    elif vp["verdict"] in ("LINKWEIGHT_RAISED", "LINKWEIGHT_CHEAPER"):
        point, gs, form, cfg = vp["pick"].rsplit("|", 3)
        if gs != f"G{G_}" or form != a_cfg:
            raise SystemExit(f"U2p's pick {vp['pick']} is not on (a)'s walk at G = {G_}")
        ci = [P.config_name(c) for c in range(P.NC)].index(cfg)
        if ci == 0 or abs(float(vp["beta"]) - float(P.CONFIGS[ci])) > 1e-12:
            raise SystemExit(f"U2p's pick {vp['pick']} does not carry its beta")
    else:
        raise SystemExit(f"U2p's verdict {vp['verdict']} is not one of its three")
    return {"verdict_p": vp, "names": names, "G": G_, "va": va, "ca": ca, "pi": names.index(point), "ci": ci,
            "beta": float(P.CONFIGS[ci]), "pick": vp["pick"], "a_cfg": a_cfg}


def filed_rows(name, cw):
    """(rows per question, held per gold, ptr, source, (a)'s record) of the carried pick's filed arrays: (a)'s at its
    pick's configuration when beta = 0 or on a KB (U2p checked its beta = 0 to be (a)'s), else U2p's at its pick."""
    rows_a, held_a, ptr_a, ra = P.load_a_pick(name, cw["names"], cw["G"], cw["va"], cw["ca"])
    if cw["ci"] == 0 or name not in P.PASSAGE:
        return rows_a[:, cw["pi"]], held_a[:, cw["pi"]], ptr_a, "U2m (a)", ra
    x = P.load_p(name, cw["names"], cw["G"])
    if x is None:
        raise SystemExit(f"{name}: U2p's filed arrays are not under outputs/u2p/stats")
    ap, rp = x
    if rp["file_sha256"] != cw["verdict_p"]["stats"][name]["file_sha256"]:
        raise SystemExit(f"{name}: U2p's filed arrays are not the ones its verdict read")
    return (np.asarray(ap["rows"][:, cw["ci"], cw["pi"]], np.int64), np.asarray(ap["held"][:, cw["ci"], cw["pi"]], bool),
            np.asarray(ap["ptr"], np.int64), "U2p", ra)


# ── stats (the laptop, or elsewhere in shards) ───────────────────────────────


def stats_stage(name, threads=2, limit=0, shard=None, dev_a=False):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if D.SOURCES != ("A1st", "A1all", "Kst", "Kall", "R0") or F.CONSTRUCTION != "KIND_T":
        raise SystemExit("P1d's sources are not A1st, A1all, Kst, Kall and R0, or P1f's construction is not KIND_T")
    if tuple(A.WALKS) != tuple(WALKS) or dict(A.WALK_UNION) != dict(WALK_UNION):
        raise SystemExit("(a)'s walks are not (d)'s")
    if dev_a and not limit:
        raise SystemExit("--dev-a is for development runs (--limit) only")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    cw = carried_walk(dev_a)
    names, G_, pi, beta = cw["names"], cw["G"], cw["pi"], cw["beta"]
    if G_ != 0:
        raise SystemExit("U2q rebuilds the rows at G = 0 (section 4)")
    x2m = M.load_u2m(name)
    if x2m is None:
        raise SystemExit(f"{name}: U2m's filed arrays are not under outputs/u2m/stats")
    a2m, r2m = x2m
    rows_f, held_f, ptr_f, src_f, ra = filed_rows(name, cw)
    (pl,) = MB.point_plan(name, a2m, r2m, [names[pi]])
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    if list(op.cfg_h["graph_regimes"]["FULL"]) != list(S4.FAMILIES) or \
            tuple(S4.FAMILIES) != ("structural", "ner", "knn"):
        raise SystemExit("FULL is not the three families in their order")
    cells, own_k = M.cells_of(construction)
    if cells != r2m["cells"] or own_k != r2m["own_cell"] or pl["k"] != own_k:
        raise SystemExit(f"{name}: the cells are not U2m's, or the pick is not on the own construction")
    n = int(ds.n_nodes)
    g = U2C.graph_csr(name, "chosen", n)
    passage = g is not None
    if passage != (name in P.PASSAGE):
        raise SystemExit(f"{name}: a chosen graph exists where U2p does not walk, or not where it does")
    stores, union, desc = stores0, union0, {"graph": "cached (the KB)"}
    if passage:
        ip, col, desc = g
        swap = U2C.csr_store(n, ip, col)
        del ip, col
        stores = dict(stores0)
        if "structural" in families:
            stores["structural"] = swap
        union = S4B.RegimeUnion(stores, families, n) if families else union0
        load0 = S4.m3b_pools.load_or_build_store

        def load(ds_, family, cache_dir):
            if family == "structural":
                if ds_.name != name:
                    raise SystemExit(f"{ds_.name}: U2m swaps {name}'s graph only")
                return swap
            return load0(ds_, family, cache_dir)

        S4.m3b_pools.load_or_build_store = load
    if desc != r2m["graph_detail"]:
        raise SystemExit(f"{name}: the graph is not the one U2m's filed run read")
    tarr, arec = AT.load_types(name, desc)
    if arec["file_sha256"] != ra["types_sha256"]:
        raise SystemExit(f"{name}: the EM types are not the ones (a)'s stats read")
    pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c == carve]
    (_carve, pop), = pops
    index = texts = None
    if name != "squad":
        index = E.build_index(E.node_names(ds), E.stopwords())
        texts = E.query_texts(ds)
        if len(texts) != int(ds.n_queries):
            raise SystemExit(f"{name}: {len(texts)} query records, not {ds.n_queries}")
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    prep0 = op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores0, op.m3a, op.m3b_contract)[0]
    prep = prep0 if stores is stores0 else \
        op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores, op.m3a, op.m3b_contract)[0]
    if any(not np.array_equal(np.asarray(x, np.int64), np.asarray(y, np.int64)) for x, y in zip(prep.seeds, prep0.seeds)):
        raise SystemExit(f"{name}/{carve}: the seeds on the graph are not G0's")
    own_b = [bp for _bs, bp in BASES].index(construction["base_pool"])
    base_rows = op.m3b_compile.base_rows({"base_pool": BASES[own_b][1]}, prep0.dense_ids, prep0.splade_ids, op.m3a,
                                         op.m3b_contract, constant)
    inc_own = [np.asarray(p, np.int64) for p in prep.pools]
    linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [EMPTY] * len(prep.seeds)
    cache = {}
    stores_all, unions = G.open_unions(cache, op, name, union)
    adj = [(np.ascontiguousarray(stores_all[f].indptr, np.int64), np.ascontiguousarray(stores_all[f].col))
           for f in S4.FAMILIES]
    for u_, _fams in D.UNIONS:
        if not (np.array_equal(unions[u_].args[0], adj[0][0]) and np.array_equal(unions[u_].args[1], adj[0][1])):
            raise SystemExit(f"{name}: union {u_}'s structural family is not the store's")
    st = stores_all["structural"]
    typed = None
    if not passage and getattr(st, "typed_indptr", None) is not None:
        typed = (np.asarray(st.typed_indptr, np.int64), np.asarray(st.typed_col), np.asarray(st.typed_rel),
                 np.asarray(st.typed_dir))
    if bool(arec["typed"]) != (typed is not None) or int(arec["entries"]) != int(adj[0][0][-1]) or \
            tarr["cls"].size != int(adj[0][0][-1]):
        raise SystemExit(f"{name}: the EM types are not this structural family's")
    mu = np.asarray(tarr["mu"], np.float64)
    k_a = int(arec["null"])
    if mu.shape[0] != k_a:
        raise SystemExit(f"{name}: (a)'s EM centres are not its K types before the null one")
    if passage:
        cls_s, zbar, srec = P.load_types(name, desc)
        if cls_s.size != tarr["cls"].size:
            raise SystemExit(f"{name}: the score types are not this structural family's")
        if cw["verdict_p"] is not None:
            xp = P.load_p(name, names, G_)
            if xp is None or xp[1]["types_sha256"] != srec["file_sha256"]:
                raise SystemExit(f"{name}: the score types are not the ones U2p's stats read")
            del xp
        ty = (np.ascontiguousarray(P.pair_types(tarr["cls"], cls_s, k_a), np.int16),) + A.no_types()[1:]
        ws = P.weights(zbar, beta)
        types_sha = srec["file_sha256"]
        del cls_s
    else:
        ty = (np.ascontiguousarray(tarr["cls"], np.int8), np.ascontiguousarray(tarr["bst"], np.int64),
              np.ascontiguousarray(tarr["bln"], np.int32), np.ascontiguousarray(typed[2], np.int16)) \
            if typed is not None else (np.ascontiguousarray(tarr["cls"], np.int8),) + A.no_types()[1:]
        ws = None
        types_sha = None
    del tarr
    walk_name = f"{cw['a_cfg']} x LW@{beta:g}" if passage else cw["a_cfg"]
    log(f"{name}/{carve}: prepared; pick {names[pi]} at G = {G_}; walk {walk_name}; typed {typed is not None}; rows "
        f"from {src_f} ({time.time() - t0:.0f}s)")
    n_st = int(unions["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    nq = min(limit, len(prep.seeds)) if limit else len(prep.seeds)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(a2m["ptr"][:nq + 1], ptr) or not np.array_equal(ptr_f[:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: U2m's or the filed pick's golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    alpha, eps, bmax = float(E.ALPHA), float(S4.EPS), int(D.NMAX)
    bud = U2E.budgets()
    acc = {"rows": [], "gold_row_beside": [], "p": [], "parts": [], "top_total": [], "entity": [], "rel": []}
    t = time.time()
    done = 0
    for lo in range(q0, q1, BATCH):
        js = list(range(lo, min(q1, lo + BATCH)))
        srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
        steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
        k0, a1 = K.restarts_batch(prep, js, linked, unions, srcs, constant)
        stz = [K.steer(prep, j, constant) for j in js]
        qs = [D5B.unit(np.asarray(prep.qemb[j], np.float64)) for j in js]
        wa = [A.omega(q, mu, A_BETA, True)[0] for q in qs]
        wqs = [P.pair_weights(x, ws) for x in wa] if passage else wa
        wal = {w: A.rwalk(unions[WALK_UNION[w]], a1 if w.startswith("A1") else k0,
                          stz if w.startswith("KT") else None, wqs, 1, ty, alpha, eps, bmax, threads) for w in WALKS}
        rows_b, ent_b = [], []
        for i, j in enumerate(js):
            src, _nn, (L, Ls_) = srcs[i]
            route = D.routes(L, Ls_)
            knd = K.kind(Ls_, prep.seeds[j], nonly)
            seeds = np.asarray(prep.seeds[j], np.int64)
            q = qs[i]
            w_all = K.steer(prep, j, constant)
            rows = M.Rows(nodes, verify=bool(limit))
            gj = golds[j]
            g0, g1 = int(ptr[j]), int(ptr[j + 1])
            B = np.unique(np.union1d(np.asarray(base_rows[j], np.int64), seeds))
            n1, _cnt, _ws, _up, _inv = D5B.n1_table(adj, B, D5.weights_of(w_all, B))
            rows.add(n1)
            cos1 = D5B.unit(rows.read(n1)) @ q if n1.size else np.empty(0, np.float32)
            cr = D5B.ranks_desc(cos1, n1) if n1.size else EMPTY
            n2, first, _taken, _hubs = U2E.second_link(adj, n1, cr, B)
            I = inc_own[j]
            if not np.isin(seeds, B).all() or not np.isin(B, I).all():
                raise SystemExit(f"{name}/{carve} question {j}: the own cell's B or I is malformed")
            in_b = np.isin(gj, B)
            sc = (A.scaled(wal["A1st"][i]), A.scaled(wal["A1all"][i]), A.scaled(wal["Kst"][i]), src[3], src[4])
            stc = dict(steered[i])
            stc["KTst"], stc["KTall"] = wal["KTst"][i][0], wal["KTall"][i][0]
            out, p1d = K.p1e_orders(sc, stc, route, knd, n_st, constant)
            kt = np.asarray(out[F.CONSTRUCTION], np.int64)
            want_kt = p1d["ROUTEst"] if knd == "entity" else out["RRF5T"]
            if not np.array_equal(kt, np.asarray(want_kt, np.int64)):
                raise SystemExit(f"{name}/{carve} question {j}: KIND_T is not ROUTEst or RRF5T by the kind")
            rec = [kt, np.asarray(out["RRF5T"], np.int64),
                   np.asarray(D.rrf([p1d["ROUTEst"], out["RRF5T"]], constant), np.int64)]
            rows.add(np.concatenate(rec + [src[4][0], I, n2]))
            full = F.outside(rec[pl["r"]], B)
            fill = M.fill_of(full, I.size, B.size)
            ords, _sizes, U, mem = U2E.question_orders(full, n1, cr, cos1, F.outside(src[4][0], B), F.outside(I, B),
                                                       n2, first, rows.read, q, constant, D.rrf)
            if not U2E.orders_exact(ords, B, full, n1, U, mem):
                raise SystemExit(f"{name}/{carve} question {j}: an order is not exactly its sources' nodes")
            order = np.asarray(ords[NAMES[pl["o"]]], np.int64)
            _bn, kind, prm = bud[pl["b"]]
            kq = int(U2E.budget_k(kind, prm, np.asarray([fill]))[0])
            kk = min(int(order.size), kq)
            keep = int(P1H.keep_counts(np.asarray([B.size]), np.asarray([kk]), pl["t"])[0])
            pos = U2E.places(gj, in_b, order)
            held = (pos == 0) | ((pos >= 1) & (pos <= keep))
            prow = np.concatenate([B, order[:keep]])
            if np.unique(prow).size != prow.size or prow.size != int(rows_f[j]):
                raise SystemExit(f"{name}/{carve} question {j}: the rows are not the carried pick's filed rows")
            if not np.array_equal(held, held_f[g0:g1]) or not np.array_equal(np.isin(gj, prow), held):
                raise SystemExit(f"{name}/{carve} question {j}: the held golds are not the carried pick's")
            rows_b.append(prow)
            ent_b.append(knd == "entity")
            acc["gold_row_beside"].append(np.isin(prow, gj))
        res = {w: split_walk(unions[WALK_UNION[w]], a1 if w.startswith("A1") else k0,
                             stz if w.startswith("KT") else None, wqs, 1, ty, rows_b, typed, alpha, eps, bmax, threads)
               for w in WALKS}
        for i, j in enumerate(js):
            prow = rows_b[i]
            pv = np.zeros((prow.size, len(WALKS)))
            pa = np.zeros((prow.size, len(WALKS), 4))
            tt = np.zeros(len(WALKS))
            rel_q = []
            for wi, w in enumerate(WALKS):
                o_nodes, o_mass, p_rows, parts, sl = res[w][i]
                r_nodes, r_mass = wal[w][i]
                if not np.array_equal(o_nodes, r_nodes) or not np.array_equal(o_mass, r_mass):
                    raise SystemExit(f"{name}/{carve} question {j}: {w}'s split walk is not (a)'s kernel's")
                if not MD.parts_ok(p_rows, parts):
                    raise SystemExit(f"{name}/{carve} question {j}: {w}'s parts do not sum to p")
                if WALK_UNION[w] == "st" and (parts[:, 2:] != 0).any():
                    raise SystemExit(f"{name}/{carve} question {j}: {w} (st) has ner or knn mass")
                if typed is not None:
                    rk, rr, rd, rm = MD.rel_parts(prow, sl, typed)
                    srel = np.bincount(rk, weights=rm, minlength=prow.size) if rk.size else np.zeros(prow.size)
                    if not np.all(np.abs(srel - parts[:, 1]) <= REL_TOL * np.abs(parts[:, 1]) + ABS_TOL):
                        raise SystemExit(f"{name}/{carve} question {j}: {w}'s (relation, direction) parts do not "
                                         "sum to its structural part")
                    rel_q.append((rk, np.full(rk.size, wi, np.int64), rr, rd, rm))
                pv[:, wi] = p_rows
                pa[:, wi, :] = parts
                tt[wi] = o_mass.sum()
            acc["rows"].append(prow)
            acc["p"].append(pv.astype(np.float32))
            acc["parts"].append(pa.astype(np.float32))
            acc["top_total"].append(tt)
            acc["entity"].append(ent_b[i])
            if typed is not None:
                acc["rel"].append(tuple(np.concatenate([x[c] for x in rel_q]) for c in range(5)))
            done += 1
            if done % 100 == 0:
                log(f"{name}/{carve}: {done}/{q1 - q0} ({time.time() - t:.0f}s)")
    head = {"declared_in": DECLARED, "stage": "U2q", "dataset": name, "carve": carve, "graph_detail": desc,
            "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(), "zu2m_b_sha256": MB.script_sha256(),
            "zu2m_d_sha256": MD.script_sha256(), "zu2m_a_sha256": A.script_sha256(),
            "zu2m_a_types_sha256": AT.script_sha256(), "zu2p_sha256": P.script_sha256(),
            "zu2e_sha256": U2E.script_sha256(), "u2m_stats_sha256": r2m["file_sha256"],
            "verdict_a_sha256": sha_file(A.OUT / "verdict_a.json"),
            "verdict_p_sha256": sha_file(P.OUT / "verdict.json") if cw["verdict_p"] is not None else None,
            "pick": cw["pick"], "point": names[pi], "G": G_, "beta": beta, "walk": walk_name,
            "rows_from": src_f, "a_stats_sha256": ra["file_sha256"], "em_types_sha256": arec["file_sha256"],
            "score_types_sha256": types_sha, "walks": list(WALKS), "types": list(TYPES), "typed": typed is not None,
            "threads": threads, "alpha": alpha, "eps": eps, "nmax": bmax}
    checks = {"graph_equal_u2m": True, "seeds_equal_g0": True, "em_types_equal_a": True, "kind_t_by_kind": True,
              "orders_exact": True, "rows_equal_carried": True, "held_equal_carried": True,
              "orders_masses_equal_a_kernel": True, "parts_sum_to_p": True, "relation_parts_sum": typed is not None,
              "rows_served_verified": bool(limit)}
    if limit:
        log(f"{name}/{carve}: development run on questions {q0}-{q1 - 1} of {nq} ({time.time() - t:.0f}s): every "
            "check passed, every served row verified; nothing written")
        return 0
    arr = MD.arrays_of(acc, ptr[q0:q1 + 1], typed is not None)
    if shard:
        od = OUT / "stats" / "parts"
        od.mkdir(parents=True, exist_ok=True)
        stem = f"{name}__{carve}__{shard[0]}of{shard[1]}"
        p = od / f"{stem}.npz"
        tmp = od / f"{stem}.tmp.npz"
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, p)
        write_json(od / f"{stem}.json", {**head, "part": shard[0], "parts": shard[1], "q0": q0, "q1": q1,
                                         "questions_all": nq, "file_sha256": sha_file(p), "checks": checks,
                                         "seconds": round(time.time() - t0, 1), "utc": utc()})
        log(f"{name}/{carve} part {shard[0]}/{shard[1]} (questions {q0}-{q1 - 1}, {time.time() - t0:.0f}s): written")
        return 0
    return write_stats(name, carve, arr, head, checks, round(time.time() - t0, 1))


def write_stats(name, carve, arr, head, checks, seconds, extra=None):
    od = OUT / "stats"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{name}__{carve}.npz"
    tmp = od / f"{name}__{carve}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, p)
    nq = int(arr["rows_ptr"].size - 1)
    rec = {**head, "questions": nq, "rows": int(arr["rows"].size), "golds": int(arr["ptr"][-1] - arr["ptr"][0]),
           "rel_entries": int(arr["rel_row"].size) if "rel_row" in arr else None, "file_sha256": sha_file(p),
           "beside": MD.beside_of(arr, head["typed"]), "checks": checks, "seconds": seconds, **(extra or {}),
           "utc": utc()}
    write_json(od / f"{name}__{carve}.json", rec)
    log(f"{name}/{carve} ({nq} q, {seconds:.0f}s): written")
    return 0


def _ok_checks(r):
    return all(v for c, v in r["checks"].items() if c not in ("rows_served_verified", "relation_parts_sum"))


def merge_stage(name, k):
    carve = U2C.carve_of(name)
    od = OUT / "stats" / "parts"
    recs, parts = [], []
    for i in range(k):
        stem = f"{name}__{carve}__{i}of{k}"
        rp = od / f"{stem}.json"
        if not rp.exists():
            raise SystemExit(f"{rp}: missing")
        r = json.loads(rp.read_text(encoding="utf-8"))
        p = od / f"{stem}.npz"
        if r.get("script_sha256") != script_sha256() or sha_file(p) != r["file_sha256"] or not _ok_checks(r):
            raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
        if r["part"] != i or r["parts"] != k or (recs and r["q0"] != recs[-1]["q1"]):
            raise SystemExit(f"part {i}: not the {i}-th of {k} question ranges")
        if recs and (r["pick"], r["beta"]) != (recs[0]["pick"], recs[0]["beta"]):
            raise SystemExit(f"part {i}: another walk than part 0's")
        with np.load(p) as z:
            parts.append({kx: z[kx] for kx in z.files})
        recs.append(r)
    if recs[0]["q0"] != 0 or recs[-1]["q1"] != recs[0]["questions_all"]:
        raise SystemExit(f"{name}/{carve}: the parts do not cover every question")
    gptr, rptr = [np.asarray([0], np.int64)], [np.asarray([0], np.int64)]
    for i, p in enumerate(parts):
        q = np.asarray(p["ptr"], np.int64)
        if int(q[0]) != int(gptr[-1][-1]):
            raise SystemExit(f"part {i}: its ptr does not continue the previous part's")
        gptr.append(q[1:])
        rptr.append(np.asarray(p["rows_ptr"][1:], np.int64) + int(rptr[-1][-1]))
    arr = {"rows_ptr": np.concatenate(rptr), "ptr": np.concatenate(gptr)}
    for kx in ("rows", "gold_row_beside", "p", "parts", "top_total", "entity"):
        arr[kx] = np.concatenate([p[kx] for p in parts])
    if "rel_row" in parts[0]:
        off = np.cumsum([0] + [int(p["rows"].size) for p in parts[:-1]])
        arr["rel_row"] = np.concatenate([p["rel_row"] + off[i] for i, p in enumerate(parts)])
        for kx in ("rel_walk", "rel_rel", "rel_dir", "rel_mass"):
            arr[kx] = np.concatenate([p[kx] for p in parts])
    head = {kx: recs[0][kx] for kx in recs[0] if kx not in (
        "part", "parts", "q0", "q1", "questions_all", "file_sha256", "checks", "seconds", "utc")}
    return write_stats(name, carve, arr, head, recs[0]["checks"], round(sum(r["seconds"] for r in recs), 1),
                       {"shards": k, "part_sha256": [r["file_sha256"] for r in recs],
                        "seconds_longest_part": max(r["seconds"] for r in recs)})


# ── beside (the laptop) ──────────────────────────────────────────────────────


def beside_stage():
    recs = {}
    for d in SIX:
        carve = U2C.carve_of(d)
        rp = OUT / "stats" / f"{d}__{carve}.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text(encoding="utf-8"))
        p = OUT / "stats" / f"{d}__{carve}.npz"
        if r.get("script_sha256") != script_sha256() or sha_file(p) != r["file_sha256"] or not _ok_checks(r):
            raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
        names = MD.rel_names(d) if r["typed"] else None
        if names is not None:
            for w in WALKS:
                for x in r["beside"][w].get("top_relations_gold", []):
                    x["name"] = names[x["relation"]] if x["relation"] < len(names) else None
        dp = MD.OUT / "stats" / f"{d}__{carve}.json"
        filed_d = None
        if dp.exists():
            rd = json.loads(dp.read_text(encoding="utf-8"))
            if rd.get("script_sha256") == MD.script_sha256() and \
                    sha_file(MD.OUT / "stats" / f"{d}__{carve}.npz") == rd["file_sha256"]:
                filed_d = {w: {lab: rd["beside"][w][lab]["share"] for lab in ("gold", "other")} for w in WALKS}
        recs[d] = {"carve": carve, "pick": r["pick"], "walk": r["walk"], "beta": r["beta"],
                   "questions": r["questions"], "rows": r["rows"], "file_sha256": r["file_sha256"],
                   "beside": r["beside"], "d_filed_shares": filed_d}
    if not recs:
        raise SystemExit("no U2q stats are written")
    picks = {(x["pick"], x["beta"]) for x in recs.values()}
    if len(picks) != 1:
        raise SystemExit(f"the datasets' stats read different walks: {sorted(picks)}")
    rec = {"declared_in": DECLARED, "stage": "U2q, beside", "script_sha256": script_sha256(),
           "decides": "nothing (an input for U2n's models)", "datasets": recs,
           "missing": [d for d in SIX if d not in recs], "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "beside.json", rec)
    (OUT / "beside.md").write_text(beside_md(rec), encoding="utf-8", newline="\n")
    log(f"U2q beside: {sorted(recs)} written; missing {rec['missing']}")
    return 0


def beside_md(rec):
    L = ["# U2q: the mass split by arrival type on the carried walk, beside", "",
         f"Declared in {rec['declared_in']}. It decides nothing: the parts are an input for U2n's models. Each cell is "
         "the mean share of the type in a row's mass on the carried walk, with (d)'s filed share (U2e's walk at "
         "beta = 0) in brackets.", ""]
    for d, x in rec["datasets"].items():
        L += [f"## {d} ({x['carve']}, {x['questions']} questions, {x['rows']:,} rows; `{x['pick']}`, walk {x['walk']})",
              "", "| walk | rows | " + " | ".join(TYPES) + " |", "|---|---|" + "---:|" * len(TYPES)]
        for w in WALKS:
            for lab in ("gold", "other"):
                s = x["beside"][w][lab]["share"]
                dd = (x["d_filed_shares"] or {}).get(w, {}).get(lab, {})
                L.append(f"| {w} | {lab} | " + " | ".join(
                    ("-" if s[t] is None else f"{s[t]:.4f}") + (f" ({dd[t]:.4f})" if dd.get(t) is not None else "")
                    for t in TYPES) + " |")
        tops = {w: x["beside"][w].get("top_relations_gold") for w in WALKS}
        if any(tops.values()):
            L += ["", "Top (relation, direction) shares of the gold rows' structural mass (direction 1: the row is the "
                  "stored source; 2: the destination):", ""]
            for w in WALKS:
                if tops[w]:
                    L.append(f"- {w}: " + "; ".join(f"{t.get('name') or t['relation']} ({t['direction']}) "
                                                     f"{t['share']:.3f}" for t in tops[w]))
        L.append("")
    if rec["missing"]:
        L += [f"Not written: {', '.join(rec['missing'])}.", ""]
    return "\n".join(L)


# ── self-test ────────────────────────────────────────────────────────────────


def _brute_w(seeds, wts, fams, typed, mode, cls, wq, steer, alpha, eps, rows):
    """The weighted push transcribed in plain Python with a per-arrival log, settled push by push: (p, parts by row,
    {(row, relation, direction): mass}). (d)'s _brute with (a)'s weight w = omega(entry) x s(v)."""
    n = len(fams[0][0]) - 1
    deg = [sum(int(f[0][v + 1] - f[0][v]) for f in fams) for v in range(n)]
    p = [0.0] * n
    r = [0.0] * n
    queued = [False] * n
    pend = {v: [0.0] * 4 for v in rows}
    rpend = {v: {} for v in rows}
    parts = {v: [0.0] * 4 for v in rows}
    rel = {}
    queue = []
    for s, w in zip(seeds, wts):
        r[s] += w
        if s in pend:
            pend[s][0] += w
    for s in seeds:
        if not queued[s]:
            queued[s] = True
            queue.append(s)
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        queued[u] = False
        ru = r[u]
        if ru < eps * max(deg[u], 1):
            continue
        r[u] = 0.0
        p[u] += alpha * ru
        if u in pend:
            for t in range(4):
                parts[u][t] += alpha * pend[u][t]
                pend[u][t] = 0.0
            for key, a in rpend[u].items():
                rel[(u,) + key] = rel.get((u,) + key, 0.0) + alpha * a
            rpend[u] = {}
        if deg[u] == 0:
            continue
        m = (1.0 - alpha) * ru
        nf = sum(1 for f in fams if f[0][u + 1] > f[0][u])
        for fi, (ip, cc) in enumerate(fams):
            a, b = int(ip[u]), int(ip[u + 1])
            if b <= a:
                continue
            ws = []
            for e in range(a, b):
                v = int(cc[e])
                w = wq[int(cls[e])] if (fi == 0 and mode == 1) else 1.0
                if steer is not None:
                    w = w * (steer[0] + steer[1][v])
                ws.append(w)
            tot = sum(ws)
            for e, w in zip(range(a, b), ws):
                v = int(cc[e])
                amt = m / (nf * tot) * w
                r[v] += amt
                if not queued[v] and r[v] >= eps * max(deg[v], 1):
                    queued[v] = True
                    queue.append(v)
                if v in pend:
                    pend[v][fi + 1] += amt
                    if typed is not None and fi == 0:
                        tip, tcol, trel, tdir = typed
                        es = [x for x in range(int(tip[v]), int(tip[v + 1])) if int(tcol[x]) == u]
                        for x in es:
                            key = (int(trel[x]), int(tdir[x]))
                            rpend[v][key] = rpend[v].get(key, 0.0) + amt / len(es)
    return np.asarray(p), parts, rel


def _close(a, b, tol=1e-12):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= tol * np.maximum(np.abs(a), np.abs(b)) + 1e-300))


def _selftest():
    import p1e_anchor_kind as K
    E = K.E
    global SLOT_BUDGET
    rng = np.random.default_rng(37)
    alpha, eps = float(E.ALPHA), 1e-4
    n_bit = n_brute = n_d = n_rel = n_grp = 0
    for case in range(48):
        n = int(rng.integers(8, 60))
        typed_on = case % 2 == 1
        fams, typed = MD._toy(rng, n, typed_on)
        un = MD._Union(fams)
        e0 = max(int(fams[0][0][-1]), 1)
        nt_ = int(rng.integers(2, 9))
        cls = rng.integers(0, nt_, e0).astype(np.int8 if case % 3 else np.int16)
        ty = (cls,) + A.no_types()[1:]
        steered = (case // 2) % 2 == 0
        nqs = int(rng.integers(1, 4))
        restarts, steers, wqs, rows = [], [], [], []
        for _q in range(nqs):
            sd = np.unique(rng.integers(0, n, int(rng.integers(1, 4)))).astype(np.int64)
            w = rng.random(sd.size) + 0.1
            restarts.append((sd, w / w.sum()))
            sn = np.unique(rng.integers(0, n, int(rng.integers(0, n)))).astype(np.int64)
            steers.append((sn, rng.random(sn.size), float(rng.random() * 0.1 + 0.01)))
            wqs.append(np.exp(rng.normal(size=nt_)))
            rows.append(np.unique(rng.integers(0, n, int(rng.integers(1, n)))).astype(np.int64))
        st = steers if steered else None
        if case % 8 == 7:
            SLOT_BUDGET = 3                                   # several kernel calls per walk
            n_grp += 1
        try:
            for mode in (1, 0):
                wq_m = wqs if mode == 1 else [np.ones(nt_) for _ in wqs]
                ref = A.rwalk(un, restarts, st, wq_m, mode, ty, alpha, eps, n, 2)
                got = split_walk(un, restarts, st, wq_m, mode, ty, rows, typed, alpha, eps, n, 2)
                for qi in range(nqs):
                    o_nodes, o_mass, p_rows, parts, sl = got[qi]
                    if not (np.array_equal(o_nodes, ref[qi][0]) and np.array_equal(o_mass, ref[qi][1])):
                        raise SystemExit(f"self-test case {case} mode {mode}: the split walk is not (a)'s kernel's")
                    n_bit += 1
                    stq = None
                    if steered:
                        ex = np.zeros(n)
                        ex[steers[qi][0]] = steers[qi][1]
                        stq = (steers[qi][2], ex)
                    bp, bparts, brel = _brute_w([int(x) for x in restarts[qi][0]],
                                                [float(x) for x in restarts[qi][1]], fams, typed, mode, cls,
                                                [float(x) for x in wq_m[qi]], stq, alpha, eps,
                                                [int(x) for x in rows[qi]])
                    if not _close(p_rows, bp[rows[qi]]) or \
                            not _close(parts, np.asarray([bparts[int(v)] for v in rows[qi]])):
                        raise SystemExit(f"self-test case {case} mode {mode}: the parts are not the per-arrival log")
                    if not MD.parts_ok(p_rows, parts):
                        raise SystemExit(f"self-test case {case} mode {mode}: the parts do not sum to p")
                    n_brute += 1
                    if typed is not None:
                        rk, rr, rd, rm = MD.rel_parts(rows[qi], sl, typed)
                        want = {}
                        for (v, rl, dr), ms in brel.items():
                            if ms != 0.0:
                                k = int(np.flatnonzero(rows[qi] == v)[0])
                                want[(k, rl, dr)] = want.get((k, rl, dr), 0.0) + ms
                        have = {(int(a), int(b), int(c)): float(m) for a, b, c, m in zip(rk, rr, rd, rm)}
                        if set(have) != set(want) or not _close([have[k] for k in sorted(want)],
                                                                [want[k] for k in sorted(want)]):
                            raise SystemExit(f"self-test case {case} mode {mode}: the relation parts are not the log")
                        n_rel += 1
                if mode == 0:
                    dref = MD.split_walk(un, restarts, st, rows, typed, alpha, eps, n, 2)
                    for qi in range(nqs):
                        for a_, b_ in zip(got[qi], dref[qi]):
                            if (a_ is None) != (b_ is None) or (a_ is not None and not np.array_equal(a_, b_)):
                                raise SystemExit(f"self-test case {case}: with every omega 1 the split kernel is not "
                                                 "(d)'s bit for bit")
                        n_d += 1
        finally:
            SLOT_BUDGET = MD.SLOT_BUDGET
    if min(n_bit, n_brute, n_d, n_rel, n_grp) == 0:
        raise SystemExit("self-test: a kind of case never ran")
    log(f"zu2q self-test: the split kernel's orders and masses equal (a)'s kernel bit for bit ({n_bit} walks, random "
        f"omega and every omega 1, typed and untyped, steered and unsteered, {n_grp} cases split across kernel calls); "
        f"its p and parts equal the per-arrival log within 1e-12 ({n_brute} walks; {n_rel} with relation parts); "
        f"with every omega 1 its p, parts and relation slots equal (d)'s split kernel bit for bit ({n_d} walks)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("stats", "merge", "beside"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--dev-a", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.stage == "stats":
        shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
        return stats_stage(a.dataset, a.threads, a.limit, shard, a.dev_a)
    if a.stage == "merge":
        return merge_stage(a.dataset, a.parts)
    if a.stage == "beside":
        return beside_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
