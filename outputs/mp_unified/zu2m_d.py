"""U2m (d): the mass split by arrival type (docs/U2M_POOLS_THREE_FAMILIES.md, section 13). An input for U2n's models,
never a pool rule: nothing trains, nothing is fit, and only the beside table reads golds.

    python outputs/mp_unified/zu2m_d.py stats DATASET [--threads 2] [--limit Q] [--shard I/K] [--pick NAME --g G]
    python outputs/mp_unified/zu2m_d.py merge DATASET --parts K
    python outputs/mp_unified/zu2m_d.py beside
    python outputs/mp_unified/zu2m_d.py --selftest

stats    One dataset's carve (s1sel; webqsp's s1eval, beside). Each question walks as zu2m_b.py's stats walk, and the
         rows of the pick section 11 carries are rebuilt (B, order[:keep] and the restored rows at its G). Then the
         five walks kind_t reads (A1st, A1all, Kst, KTst, KTall) run again with the split kernels, which keep, for the
         pool's rows only, each arrival type's pending and settled mass beside r. Checked: every walk's order (and
         the unsteered walks' masses) equals the original kernel's; the parts sum to p; on the KB the (relation,
         direction) parts sum to the structural part; the rows and held golds equal the carried pick's filed ones.
         --limit Q: the first Q questions, every check, nothing written and no number logged. --pick/--g: development
         runs only, before (b)'s verdict is filed. --shard I/K: the I-th of K contiguous question ranges.
merge    Join a dataset's K parts into the file one run over every question writes.
beside   The beside table over the datasets written, deciding nothing: outputs/u2m/d/beside_d.json and .md.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zu2m as M  # noqa: E402  (U2m's walk, cells, columns, rows and arrays, unchanged)
import zu2m_b as MB  # noqa: E402  (section 11's carried pick, plan, lists and restore, unchanged)

U2E, U2C, F, D5, D5B = M.U2E, M.U2C, M.F, M.D5, M.D5B
log, utc, write_json, sha_file = M.log, M.utc, M.write_json, M.sha_file
DECLARED = "docs/U2M_POOLS_THREE_FAMILIES.md, section 13"
OUT = M.OUT / "d"
SIX, TRAIN5, NAMES, RECIPES, GRID, BASES = M.SIX, M.TRAIN5, M.NAMES, M.RECIPES, M.GRID, M.BASES
EMPTY = M.EMPTY
WALKS = ("A1st", "A1all", "Kst", "KTst", "KTall")
WALK_UNION = {"A1st": "st", "A1all": "all", "Kst": "st", "KTst": "st", "KTall": "all"}
UNSTEERED = ("A1st", "A1all", "Kst")        # P1d's sources: walk_mass returns their masses
SRC_INDEX = {"A1st": 0, "A1all": 1, "Kst": 2}  # their places in P1d's SOURCES
TYPES = ("restart", "structural", "ner", "knn")
REL_TOL, ABS_TOL = 1e-9, 1e-15
SLOT_BUDGET = 20_000_000                    # typed slots per kernel call (the KB's relation parts)
REL_KEY = 65536                             # relation ids are int16


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the split kernels (step 4e's and P1e's push, with the arrivals at pool rows kept beside r) ────────────────────


_K = {}


def kernels():
    """The split kernels, compiled on first use (numba is imported here, as zu2m.py imports it in its stats)."""
    if _K:
        return _K
    from numba import njit, prange
    import step4_pool_reach as S4
    _top = S4._top

    @njit
    def _arrive(v, u, amt, fam, ridx, pend, typed, tip, tcol, toff, rpend, nxt, dhead, inl):
        """An arrival of amt at v from u over family fam: v's type part, and on the KB's structural family its
        (relation, direction) entries for u in v's typed row, split equally."""
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
        """u's push: alpha times each pending part settles, and the pending parts clear."""
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
    def _sspread(ip, cc, fam, u, m, nf, steered, floor, ex, r, flag, touched, queue, deg, eps, n, nt, tail, ridx, pend,
                 typed, tip, tcol, toff, rpend, nxt, dhead, inl):
        """step 4's _spread (uniform within the family) or P1e's _tspread (w = floor + ex), with the arrivals kept."""
        a = ip[u]
        b = ip[u + 1]
        if b <= a:
            return nt, tail
        share = 0.0
        base = 0.0
        if steered:
            tot = 0.0
            for k in range(a, b):
                tot += floor + ex[cc[k]]
            base = m / (nf * tot)
        else:
            share = m / (nf * (b - a))
        for k in range(a, b):
            v = cc[k]
            if flag[v] & 2 == 0:
                flag[v] |= 2
                touched[nt] = v
                nt += 1
            if steered:
                amt = base * (floor + ex[v])
            else:
                amt = share
            r[v] += amt
            if flag[v] & 1 == 0 and r[v] >= eps * max(deg[v], 1):
                flag[v] |= 1
                queue[tail % n] = v
                tail += 1
            if ridx[v] >= 0:
                _arrive(v, u, amt, fam, ridx, pend, typed, tip, tcol, toff, rpend, nxt, dhead, inl)
        return nt, tail

    @njit
    def _spush(seeds, wts, ip0, c0, ip1, c1, ip2, c2, deg, steered, floor, ex, alpha, eps, p, r, flag, touched, queue,
               ridx, pend, parts, row0, typed, tip, tcol, toff, rpend, rsett, slot0, nxt, dhead, inl):
        """step 4e's _wpush (nothing excluded) or P1e's _tpush, the same statements on p and r in the same order."""
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
            nt, tail = _sspread(ip0, c0, 0, u, m, nf, steered, floor, ex, r, flag, touched, queue, deg, eps, n, nt,
                                tail, ridx, pend, typed, tip, tcol, toff, rpend, nxt, dhead, inl)
            nt, tail = _sspread(ip1, c1, 1, u, m, nf, steered, floor, ex, r, flag, touched, queue, deg, eps, n, nt,
                                tail, ridx, pend, typed, tip, tcol, toff, rpend, nxt, dhead, inl)
            nt, tail = _sspread(ip2, c2, 2, u, m, nf, steered, floor, ex, r, flag, touched, queue, deg, eps, n, nt,
                                tail, ridx, pend, typed, tip, tcol, toff, rpend, nxt, dhead, inl)
        return nt

    @njit(parallel=True)
    def split_batch(seeds_ptr, seeds_flat, w_flat, st_ptr, st_nodes, st_ex, floors, steered, ip0, c0, ip1, c1, ip2,
                    c2, deg, alpha, eps, bmax, n_chunks, rows_ptr, rows_flat, typed, tip, tcol, slot_ptr, out_nodes,
                    out_mass, out_count, p_rows, parts, rsett):
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
                        nt = _spush(seeds, w_flat[seeds_ptr[q]:seeds_ptr[q + 1]], ip0, c0, ip1, c1, ip2, c2, deg,
                                    steered, floors[q], ex, alpha, eps, p, r, flag, touched, queue, ridx, pend, parts,
                                    r0, typed, tip, tcol, toff, rpend, rsett, slot_ptr[q], nxt, dhead, inl)
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

    _K.update(split_batch=split_batch, spush=_spush)
    return _K


def split_walk(union, restarts, steers, rows, typed, alpha, eps, bmax, threads):
    """Per question: (order, its raw masses, p on the rows, the rows' four parts, the settled (relation, direction)
    slots or None). steers None: step 4e's walk; else P1e's steered walk. typed: (tip, tcol, trel, tdir) or None."""
    import step4_pool_reach as S4
    kn = kernels()
    nq = len(restarts)
    out = [None] * nq
    if nq == 0:
        return out
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
        r_ptr, r_flat = S4.flat([np.asarray(rows[i], np.int64) for i in g])
        sl_ptr = np.concatenate([[0], np.cumsum(slots[g])]).astype(np.int64)
        on = np.full((len(g), bmax), -1, dtype=np.int64)
        om = np.zeros((len(g), bmax))
        oc = np.zeros(len(g), dtype=np.int64)
        p_rows = np.zeros(r_flat.size)
        parts = np.zeros((r_flat.size, 4))
        rsett = np.zeros(max(int(sl_ptr[-1]), 1))
        kn["split_batch"](s_ptr, s_flat, w_flat, t_ptr, t_nodes, t_ex, floors, steers is not None, *union.args,
                          float(alpha), float(eps), int(bmax), max(1, min(len(g), 4 * threads)), r_ptr, r_flat,
                          typed is not None, np.asarray(tip, np.int64), tcol, sl_ptr, on, om, oc, p_rows, parts, rsett)
        for gi, i in enumerate(g):
            c = int(oc[gi])
            a, b = int(r_ptr[gi]), int(r_ptr[gi + 1])
            sl = rsett[int(sl_ptr[gi]):int(sl_ptr[gi + 1])].copy() if typed is not None else None
            out[i] = (on[gi, :c].copy(), om[gi, :c].copy(), p_rows[a:b].copy(), parts[a:b].copy(), sl)
    return out


def rel_parts(rows, sl, typed):
    """The settled slots of one question's rows as (row index, relation, direction, mass), summed per row, relation
    and direction; rows with no entry give none."""
    tip, _tcol, trel, tdir = typed
    rows = np.asarray(rows, np.int64)
    deg = (tip[rows + 1] - tip[rows]).astype(np.int64)
    toff = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
    s = np.flatnonzero(sl)
    if not s.size:
        return EMPTY, EMPTY, EMPTY, np.empty(0, np.float64)
    k = np.searchsorted(toff, s, side="right") - 1
    e = tip[rows[k]] + (s - toff[k])
    key = (k * REL_KEY + np.asarray(trel[e], np.int64)) * 4 + np.asarray(tdir[e], np.int64)
    u, inv = np.unique(key, return_inverse=True)
    mass = np.bincount(inv, weights=sl[s], minlength=u.size)
    return u // (REL_KEY * 4), (u // 4) % REL_KEY, u % 4, mass


def parts_ok(total, parts):
    return bool(np.all(np.abs(parts.sum(axis=-1) - total) <= REL_TOL * np.abs(total) + ABS_TOL))


# ── the carried pick ─────────────────────────────────────────────────────────


def carried_pick(pick=None, g=None, dev=False):
    """(name, G, (b)'s verdict or None) of the pick section 11 carries."""
    p = MB.OUT / "verdict_b.json"
    if p.exists():
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("script_sha256") != MB.script_sha256() or rec.get("zu2m_sha256") != M.script_sha256():
            raise SystemExit("outputs/u2m/b/verdict_b.json is not the committed zu2m_b.py's verdict")
        nm, gs = rec["pick"].rsplit("|G", 1)          # (b)'s pick is "<point>|G<g>"; the point is the name
        if int(gs) != int(rec["G"]):
            raise SystemExit("outputs/u2m/b/verdict_b.json: its pick's G is not its G")
        if pick is not None and (pick != nm or int(g) != int(rec["G"])):
            raise SystemExit("--pick/--g are not the pick (b)'s verdict carries")
        return nm, int(rec["G"]), rec
    if not dev or pick is None or g is None:
        raise SystemExit("(b)'s verdict is not filed (outputs/u2m/b/verdict_b.json); a development run may name "
                         "--pick and --g")
    if int(g) not in MB.GS:
        raise SystemExit(f"--g {g}: not one of section 11's G")
    return pick, int(g), None


# ── stats (the laptop, or Modal in shards) ───────────────────────────────────


def stats_stage(name, threads=2, limit=0, shard=None, pick=None, g_arg=None):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if D.SOURCES[:3] != ("A1st", "A1all", "Kst") or F.CONSTRUCTION != "KIND_T":
        raise SystemExit("P1d's first three sources are not A1st, A1all and Kst, or P1f's construction is not KIND_T")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    pk, G_, vb = carried_pick(pick, g_arg, dev=bool(limit))
    x2m = M.load_u2m(name)
    if x2m is None:
        raise SystemExit(f"{name}: U2m's filed arrays are not under outputs/u2m/stats")
    a2m, r2m = x2m
    (pl,) = MB.point_plan(name, a2m, r2m, [pk])
    gi = MB.GS.index(G_)
    ab = None
    if G_ > 0:
        _vrec, _car, _pts, names_b = MB.carried()
        xb = MB.load_b(name, names_b)
        if xb is None:
            raise SystemExit(f"{name}: (b)'s filed arrays are not under outputs/u2m/b/stats")
        ab = (xb[0], names_b.index(pk))
        if vb is not None and vb["stats"][name]["file_sha256"] != xb[1]["file_sha256"]:
            raise SystemExit(f"{name}: (b)'s filed arrays are not the ones its verdict read")
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    full_fams = list(op.cfg_h["graph_regimes"]["FULL"])
    if full_fams != list(S4.FAMILIES) or tuple(S4.FAMILIES) != ("structural", "ner", "knn"):
        raise SystemExit("FULL is not the three families in their order")
    cells, own_k = M.cells_of(construction)
    if cells != r2m["cells"] or own_k != r2m["own_cell"]:
        raise SystemExit(f"{name}: the cells are not U2m's")
    if pl["k"] != own_k:
        raise SystemExit(f"{pk}: (d) reads the carried pick, here a point on the own construction")
    n = int(ds.n_nodes)
    g = U2C.graph_csr(name, "chosen", n)
    stores, union, desc = stores0, union0, {"graph": "cached (the KB)"}
    if g is not None:
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
    st = stores_all["structural"]
    typed = None
    if g is None and getattr(st, "typed_indptr", None) is not None:
        typed = (np.asarray(st.typed_indptr, np.int64), np.asarray(st.typed_col), np.asarray(st.typed_rel),
                 np.asarray(st.typed_dir))
        for u_, _fams in D.UNIONS:
            if not np.array_equal(unions[u_].args[0], np.asarray(st.indptr, np.int64)):
                raise SystemExit(f"{name}: union {u_}'s structural rows are not the typed store's")
    untyped = [adj[i] for i, f in enumerate(S4.FAMILIES) if not (typed is not None and f == "structural")]
    log(f"{name}/{carve}: prepared; pick {pk} at G = {G_}; typed {typed is not None} ({time.time() - t0:.0f}s)")
    n_st = int(unions["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    nq = min(limit, len(prep.seeds)) if limit else len(prep.seeds)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(a2m["ptr"][:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: U2m's golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    # the filed rows and held golds of the carried pick, question for question
    if G_ == 0:
        sub = {kx: a2m[kx] for kx in ("bs", "fill", "lens", "pos", "ptr")}
        rows_f, held_f = M.held_of(sub, pl["k"], pl["r"], pl["o"], pl["b"], pl["t"])
    else:
        rows_f = np.asarray(ab[0]["rows"][:, ab[1], gi], np.int64)
        held_f = np.asarray(ab[0]["held"][:, ab[1], gi], bool)
    alpha, eps, bmax = float(E.ALPHA), float(S4.EPS), int(D.NMAX)
    acc = {"rows": [], "gold_row_beside": [], "p": [], "parts": [], "top_total": [], "entity": [], "rel": []}
    t = time.time()
    done = 0
    max_slots = 0
    for lo in range(q0, q1, S4.BATCH):
        js = list(range(lo, min(q1, lo + S4.BATCH)))
        srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
        steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
        k0, a1 = K.restarts_batch(prep, js, linked, unions, srcs, constant)
        stz = [K.steer(prep, j, constant) for j in js]
        rows_b, ent_b = [], []
        for i, j in enumerate(js):
            src, _nn, (L, Ls_) = srcs[i]
            route = D.routes(L, Ls_)
            knd = K.kind(Ls_, prep.seeds[j], nonly)
            out, p1d = K.p1e_orders(src, steered[i], route, knd, n_st, constant)
            kt = np.asarray(out[F.CONSTRUCTION], np.int64)
            want_kt = p1d["ROUTEst"] if knd == "entity" else out["RRF5T"]
            if not np.array_equal(kt, np.asarray(want_kt, np.int64)):
                raise SystemExit(f"{name}/{carve} question {j}: KIND_T is not ROUTEst or RRF5T by the kind")
            rec = [kt, np.asarray(out["RRF5T"], np.int64),
                   np.asarray(D.rrf([p1d["ROUTEst"], out["RRF5T"]], constant), np.int64)]
            seeds = np.asarray(prep.seeds[j], np.int64)
            q = D5B.unit(np.asarray(prep.qemb[j], np.float64))
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
            rows.add(np.concatenate(rec + [src[4][0], I, n2]))
            in_b = np.isin(gj, B)
            full = F.outside(rec[pl["r"]], B)
            ords, _sizes, U, mem = U2E.question_orders(full, n1, cr, cos1, F.outside(src[4][0], B), F.outside(I, B),
                                                       n2, first, rows.read, q, constant, D.rrf)
            if not U2E.orders_exact(ords, B, full, n1, U, mem):
                raise SystemExit(f"{name}/{carve} question {j}: an order is not exactly its sources' nodes")
            order = np.asarray(ords[NAMES[pl["o"]]], np.int64)
            pos = U2E.places(gj, in_b, order)
            if order.size != int(a2m["lens"][j, pl["c"]]) or not np.array_equal(pos, a2m["pos"][g0:g1, pl["c"]]):
                raise SystemExit(f"{name}/{carve} question {j}: the pick's column is not U2m's")
            kk, keep = int(pl["kk"][j]), int(pl["keep"][j])
            rest = MB.restore(MB.build_lists(B, order, kk, keep, untyped, typed), kk, keep, G_) if G_ > 0 else EMPTY
            if rest.size and not np.isin(rest, order[keep:kk]).all():
                raise SystemExit(f"{name}/{carve} question {j}: a restored row is outside order[keep:kk]")
            prow = np.concatenate([B, order[:keep], np.asarray(rest, np.int64)])
            if np.unique(prow).size != prow.size or prow.size != int(rows_f[j]):
                raise SystemExit(f"{name}/{carve} question {j}: the rows are not the carried pick's filed rows")
            if not np.array_equal(np.isin(gj, prow), held_f[g0:g1]):
                raise SystemExit(f"{name}/{carve} question {j}: the held golds are not the carried pick's")
            rows_b.append(prow)
            ent_b.append(knd == "entity")
            acc["gold_row_beside"].append(np.isin(prow, gj))
        res = {}
        for w in WALKS:
            rs = a1 if w.startswith("A1") else k0
            res[w] = split_walk(unions[WALK_UNION[w]], rs, stz if w.startswith("KT") else None, rows_b, typed, alpha,
                                eps, bmax, threads)
        for i, j in enumerate(js):
            src = srcs[i][0]
            prow = rows_b[i]
            pv = np.zeros((prow.size, len(WALKS)))
            pa = np.zeros((prow.size, len(WALKS), 4))
            tt = np.zeros(len(WALKS))
            rel_q = []
            for wi, w in enumerate(WALKS):
                o_nodes, o_mass, p_rows, parts, sl = res[w][i]
                if w in UNSTEERED:
                    s_nodes, s_mass = src[SRC_INDEX[w]]
                    tot = o_mass.sum()
                    sc = o_mass / tot if tot > 0 else o_mass
                    if not np.array_equal(o_nodes, s_nodes) or not np.array_equal(sc, s_mass):
                        raise SystemExit(f"{name}/{carve} question {j}: {w}'s split walk is not walk_mass's")
                elif not np.array_equal(o_nodes, np.asarray(steered[i][w], np.int64)):
                    raise SystemExit(f"{name}/{carve} question {j}: {w}'s split walk is not P1e's twalk")
                if not parts_ok(p_rows, parts):
                    raise SystemExit(f"{name}/{carve} question {j}: {w}'s parts do not sum to p")
                if WALK_UNION[w] == "st" and (parts[:, 2:] != 0).any():
                    raise SystemExit(f"{name}/{carve} question {j}: {w} (st) has ner or knn mass")
                if typed is not None:
                    max_slots = max(max_slots, int(sl.size))
                    rk, rr, rd, rm = rel_parts(prow, sl, typed)
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
    head = {"declared_in": DECLARED, "stage": "U2m (d)", "dataset": name, "carve": carve, "graph_detail": desc,
            "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(), "zu2m_b_sha256": MB.script_sha256(),
            "zu2e_sha256": U2E.script_sha256(), "u2m_stats_sha256": r2m["file_sha256"], "pick": pk, "G": G_,
            "verdict_b_sha256": sha_file(MB.OUT / "verdict_b.json") if vb is not None else None,
            "walks": list(WALKS), "types": list(TYPES), "typed": typed is not None, "threads": threads,
            "alpha": alpha, "eps": eps, "nmax": bmax}
    checks = {"graph_equal_u2m": True, "seeds_equal_g0": True, "kind_t_by_kind": True, "column_equal_u2m": True,
              "rows_equal_carried": True, "held_equal_carried": True, "orders_equal_original": True,
              "masses_equal_walk_mass": True, "parts_sum_to_p": True, "relation_parts_sum": typed is not None,
              "rows_served_verified": bool(limit)}
    if limit:
        log(f"{name}/{carve}: development run on questions {q0}-{q1 - 1} of {nq} ({time.time() - t:.0f}s): "
            f"every check passed, every served row verified (largest typed slots per question: {max_slots}); "
            "nothing written")
        return 0
    arr = arrays_of(acc, ptr[q0:q1 + 1], typed is not None)
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


def arrays_of(acc, gptr, typed):
    nr = np.asarray([x.size for x in acc["rows"]], np.int64)
    arr = {"rows_ptr": np.concatenate([[0], np.cumsum(nr)]).astype(np.int64),
           "rows": np.concatenate(acc["rows"]).astype(np.int64) if nr.size else np.empty(0, np.int64),
           "gold_row_beside": np.concatenate(acc["gold_row_beside"]) if nr.size else np.empty(0, bool),
           "p": np.concatenate(acc["p"]) if nr.size else np.empty((0, len(WALKS)), np.float32),
           "parts": np.concatenate(acc["parts"]) if nr.size else np.empty((0, len(WALKS), 4), np.float32),
           "top_total": np.asarray(acc["top_total"], np.float64).reshape(-1, len(WALKS)),
           "entity": np.asarray(acc["entity"], bool), "ptr": np.asarray(gptr, np.int64)}
    if typed:
        off = arr["rows_ptr"][:-1]
        cols = [np.concatenate([x[c] + (off[i] if c == 0 else 0) for i, x in enumerate(acc["rel"])])
                for c in range(5)]
        arr.update(rel_row=cols[0].astype(np.int64), rel_walk=cols[1].astype(np.int8),
                   rel_rel=cols[2].astype(np.int16), rel_dir=cols[3].astype(np.int8),
                   rel_mass=cols[4].astype(np.float32))
    return arr


def beside_of(arr, typed):
    """Per walk: the mean share of each type in a row's mass over gold rows and over the other rows (rows with mass);
    on the KB, the five (relation, direction) pairs with the largest share of the gold rows' structural mass."""
    gold = arr["gold_row_beside"]
    out = {}
    for wi, w in enumerate(WALKS):
        p = arr["p"][:, wi].astype(np.float64)
        parts = arr["parts"][:, wi, :].astype(np.float64)
        x = {}
        for lab, sel in (("gold", gold), ("other", ~gold)):
            m = sel & (p > 0)
            x[lab] = {"rows": int(sel.sum()), "with_mass": int(m.sum()),
                      "share": {t: (round(float((parts[m, ti] / p[m]).mean()), 4) if m.any() else None)
                                for ti, t in enumerate(TYPES)}}
        if typed:
            sel = (arr["rel_walk"] == wi) & gold[arr["rel_row"]]
            den = float(parts[gold, 1].sum())
            if sel.any() and den > 0:
                key = arr["rel_rel"][sel].astype(np.int64) * 4 + arr["rel_dir"][sel].astype(np.int64)
                u, inv = np.unique(key, return_inverse=True)
                ms = np.bincount(inv, weights=arr["rel_mass"][sel].astype(np.float64), minlength=u.size)
                top = np.argsort(-ms, kind="stable")[:5]
                x["top_relations_gold"] = [{"relation": int(u[k] // 4), "direction": int(u[k] % 4),
                                            "share": round(float(ms[k] / den), 4)} for k in top]
            else:
                x["top_relations_gold"] = []
        out[w] = x
    return out


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
           "beside": beside_of(arr, head["typed"]), "checks": checks, "seconds": seconds, **(extra or {}),
           "utc": utc()}
    write_json(od / f"{name}__{carve}.json", rec)
    log(f"{name}/{carve} ({nq} q, {seconds:.0f}s): written")
    return 0


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
        if r.get("script_sha256") != script_sha256() or sha_file(p) != r["file_sha256"] or not all(
                v for c, v in r["checks"].items() if c not in ("rows_served_verified", "relation_parts_sum")):
            raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
        if r["part"] != i or r["parts"] != k or (recs and r["q0"] != recs[-1]["q1"]):
            raise SystemExit(f"part {i}: not the {i}-th of {k} question ranges")
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


def rel_names(name):
    p = M.ROOT / "outputs" / "m3b" / "relations" / f"{name}_rel_vocab.json"
    return json.loads(p.read_text(encoding="utf-8"))["vocab"] if p.exists() else None


def beside_stage():
    recs = {}
    for d in SIX:
        carve = U2C.carve_of(d)
        rp = OUT / "stats" / f"{d}__{carve}.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text(encoding="utf-8"))
        p = OUT / "stats" / f"{d}__{carve}.npz"
        if r.get("script_sha256") != script_sha256() or sha_file(p) != r["file_sha256"] or not all(
                v for c, v in r["checks"].items() if c not in ("rows_served_verified", "relation_parts_sum")):
            raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
        names = rel_names(d) if r["typed"] else None
        if names is not None:
            for w in WALKS:
                for x in r["beside"][w].get("top_relations_gold", []):
                    x["name"] = names[x["relation"]] if x["relation"] < len(names) else None
        recs[d] = {"carve": carve, "pick": r["pick"], "G": r["G"], "questions": r["questions"], "rows": r["rows"],
                   "file_sha256": r["file_sha256"], "beside": r["beside"]}
    if not recs:
        raise SystemExit("no (d) stats are written")
    picks = {(x["pick"], x["G"]) for x in recs.values()}
    if len(picks) != 1:
        raise SystemExit(f"the datasets' stats read different picks: {sorted(picks)}")
    rec = {"declared_in": DECLARED, "stage": "U2m (d), beside", "script_sha256": script_sha256(),
           "decides": "nothing (an input for U2n's models)", "datasets": recs, "missing": [d for d in SIX if d not in recs],
           "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "beside_d.json", rec)
    (OUT / "beside_d.md").write_text(beside_md(rec), encoding="utf-8", newline="\n")
    log(f"U2m (d) beside: {sorted(recs)} written; missing {rec['missing']}")
    return 0


def beside_md(rec):
    L = ["# U2m (d): the mass split by arrival type, beside", "",
         f"Declared in {rec['declared_in']}. It decides nothing: the parts are an input for U2n's models.", ""]
    for d, x in rec["datasets"].items():
        L += [f"## {d} ({x['carve']}, {x['questions']} questions, {x['rows']:,} rows; pick `{x['pick']}`, G = "
              f"{x['G']})", "",
              "| walk | rows | " + " | ".join(TYPES) + " |", "|---|---|" + "---:|" * len(TYPES)]
        for w in WALKS:
            for lab in ("gold", "other"):
                s = x["beside"][w][lab]["share"]
                L.append(f"| {w} | {lab} | " + " | ".join("-" if s[t] is None else f"{s[t]:.4f}" for t in TYPES) + " |")
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


def _brute(seeds, wts, fams, typed, steer, alpha, eps, rows):
    """The push transcribed in plain Python with a per-arrival log, settled push by push: (p, parts by row,
    {(row, relation, direction): mass})."""
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
            nb = [int(x) for x in cc[a:b]]
            if steer is None:
                amts = [m / (nf * (b - a))] * len(nb)
            else:
                floor, ex = steer
                tot = 0.0
                for v in nb:
                    tot += floor + ex[v]
                base = m / (nf * tot)
                amts = [base * (floor + ex[v]) for v in nb]
            for v, amt in zip(nb, amts):
                r[v] += amt
                if not queued[v] and r[v] >= eps * max(deg[v], 1):
                    queued[v] = True
                    queue.append(v)
                if v in pend:
                    pend[v][fi + 1] += amt
                    if typed is not None and fi == 0:
                        tip, tcol, trel, tdir = typed
                        es = [e for e in range(int(tip[v]), int(tip[v + 1])) if int(tcol[e]) == u]
                        for e in es:
                            key = (int(trel[e]), int(tdir[e]))
                            rpend[v][key] = rpend[v].get(key, 0.0) + amt / len(es)
    return p, parts, rel


def _toy(rng, n, typed_on):
    """Three symmetric families as collapsed CSRs; the first typed (relation, direction) when typed_on."""
    fams = []
    typed = None
    for fi in range(3):
        m = int(rng.integers(0, 3 * n))
        src = rng.integers(0, n, m)
        dst = rng.integers(0, n, m)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        if fi == 0 and typed_on:
            rel = rng.integers(0, 3, src.size)
            heads = np.concatenate([src, dst])
            tails = np.concatenate([dst, src])
            rel2 = np.concatenate([rel, rel])
            dir2 = np.concatenate([np.ones(src.size, np.int8), np.full(src.size, 2, np.int8)])
            o = np.lexsort((rel2, tails, heads))
            heads, tails, rel2, dir2 = heads[o], tails[o], rel2[o], dir2[o]
            tip = np.concatenate([[0], np.cumsum(np.bincount(heads, minlength=n))]).astype(np.int64)
            typed = (tip, tails.astype(np.int32), rel2.astype(np.int16), dir2)
            k = np.ones(heads.size, bool)
            if heads.size:
                k[1:] = (heads[1:] != heads[:-1]) | (tails[1:] != tails[:-1])
            ch, ct = heads[k], tails[k]
        else:
            pairs = np.unique(np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], 1), axis=0) \
                if src.size else np.zeros((0, 2), np.int64)
            ch, ct = pairs[:, 0], pairs[:, 1]
        ip = np.concatenate([[0], np.cumsum(np.bincount(ch, minlength=n))]).astype(np.int64)
        fams.append((ip, np.asarray(ct, np.int32)))
    return fams, typed


class _Union:
    def __init__(self, fams):
        n = fams[0][0].size - 1
        deg = np.zeros(n, np.int64)
        for ip, _c in fams:
            deg += np.diff(ip)
        self.args = tuple(x for ip, c in fams for x in (ip, c)) + (deg,)
        self.deg = deg


def _selftest():
    import p1e_anchor_kind as K
    E, S4 = K.E, K.S4
    rng = np.random.default_rng(13)
    alpha, eps = float(E.ALPHA), 1e-4
    cases = n_rel = n_steer = n_mass = 0
    for case in range(48):
        n = int(rng.integers(8, 60))
        typed_on = case % 2 == 0
        fams, typed = _toy(rng, n, typed_on)
        un = _Union(fams)
        steered = case % 3 == 0
        nq = int(rng.integers(1, 4))
        restarts, steers, rows = [], [], []
        for _q in range(nq):
            ns = int(rng.integers(0, 4)) if case % 5 else int(rng.integers(1, 4))
            sd = np.unique(rng.integers(0, n, ns)).astype(np.int64)
            w = rng.random(sd.size) + 0.1
            restarts.append((sd, w / w.sum() if w.size else w))
            sn = np.unique(rng.integers(0, n, int(rng.integers(0, n)))).astype(np.int64)
            steers.append((sn, rng.random(sn.size), float(rng.random() * 0.1 + 0.01)))
            rows.append(rng.permutation(n)[:int(rng.integers(0, n + 1))].astype(np.int64))
        res = split_walk(un, restarts, steers if steered else None, rows, typed, alpha, eps, n, 2)
        for q in range(nq):
            sd, w = restarts[q]
            p0 = np.zeros(n)
            r0 = np.zeros(n)
            fl = np.zeros(n, np.uint8)
            tch = np.empty(n, np.int32)
            qu = np.empty(n, np.int32)
            if sd.size:
                if steered:
                    ex = np.zeros(n)
                    ex[steers[q][0]] = steers[q][1]
                    nt = K._tpush(sd, w, *un.args, steers[q][2], ex, alpha, eps, p0, r0, fl, tch, qu)
                else:
                    nt, _w = E._wpush(sd, w, np.empty(0, np.int64), *un.args, alpha, eps, p0, r0, fl, tch, qu)
                want = np.full(n, -1, np.int64)
                _a, cnt = S4._top(p0, fl, tch, nt, n, want)
                want = want[:cnt]
            else:
                want = np.empty(0, np.int64)
            o_nodes, o_mass, p_rows, parts, sl = res[q]
            if not np.array_equal(o_nodes, want) or not np.array_equal(o_mass, p0[want]) or \
                    not np.array_equal(p_rows, p0[rows[q]]):
                raise SystemExit(f"self-test case {case}: the split kernel's p is not the original's")
            stq = (steers[q][2], np.bincount(steers[q][0], weights=steers[q][1], minlength=n)) if steered else None
            bp, bparts, brel = _brute([int(x) for x in sd], [float(x) for x in w], fams, typed, stq, alpha, eps,
                                      set(int(x) for x in rows[q]))
            if not np.allclose(np.asarray(bp)[rows[q]], p_rows, rtol=1e-12, atol=0):
                raise SystemExit(f"self-test case {case}: the brute force's p is not the kernel's")
            want_parts = np.asarray([bparts[int(v)] for v in rows[q]]).reshape(-1, 4)
            if not np.allclose(want_parts, parts, rtol=1e-12, atol=1e-300) or not parts_ok(p_rows, parts):
                raise SystemExit(f"self-test case {case}: the parts are not the per-arrival log's")
            if typed is not None:
                rk, rr, rd, rm = rel_parts(rows[q], sl, typed)
                got = {(int(rows[q][k]), int(a), int(b)): float(c) for k, a, b, c in zip(rk, rr, rd, rm)}
                if set(got) != set(brel) or any(abs(got[k] - brel[k]) > 1e-12 * abs(brel[k]) for k in brel):
                    raise SystemExit(f"self-test case {case}: the relation parts are not the per-arrival log's")
                srel = np.bincount(rk, weights=rm, minlength=rows[q].size) if rk.size else np.zeros(rows[q].size)
                if not np.all(np.abs(srel - parts[:, 1]) <= REL_TOL * np.abs(parts[:, 1]) + ABS_TOL):
                    raise SystemExit(f"self-test case {case}: the relation parts do not sum to the structural part")
            cases += 1
            n_mass += int((p_rows > 0).any())
            n_steer += int(steered and (p_rows > 0).any())
            n_rel += int(typed is not None and bool(brel))
    if min(n_rel, n_steer, n_mass) == 0:
        raise SystemExit("self-test: a kind of case never carried mass")
    log(f"zu2m_d self-test: {cases} toy walks ({n_mass} with mass on their rows, {n_steer} of them steered, {n_rel} "
        "with relation parts): p and every order equal the original kernels' bit for bit; the parts and the "
        "(relation, direction) parts equal a per-arrival log settled push by push, and sum to p and to the "
        "structural part")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("stats", "merge", "beside"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--pick", default=None)
    ap.add_argument("--g", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.stage == "stats":
        if a.dataset not in SIX:
            raise SystemExit(f"stats: a dataset of {SIX}")
        if (a.pick is not None or a.g is not None) and not a.limit:
            raise SystemExit("--pick/--g are for development runs (--limit) only")
        shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
        return stats_stage(a.dataset, a.threads, a.limit, shard, a.pick, a.g)
    if a.stage == "merge":
        return merge_stage(a.dataset, a.parts)
    if a.stage == "beside":
        return beside_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
