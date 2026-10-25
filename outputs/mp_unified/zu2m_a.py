"""U2m (a): the relational walk on (b)'s carried frontier (docs/U2M_POOLS_THREE_FAMILIES.md, section 14). Pools only:
nothing trains, and no weight, type or fit reads a question's gold or label.

    python outputs/mp_unified/zu2m_a.py stats DATASET [--threads 2] [--limit Q] [--shard I/K] [--g G]
    python outputs/mp_unified/zu2m_a.py merge DATASET --parts K
    python outputs/mp_unified/zu2m_a.py verdict
    python outputs/mp_unified/zu2m_a.py --selftest

stats    One dataset's carve (s1sel; webqsp's s1eval, beside), after zu2m_a_types.py DATASET. Each question walks as
         zu2m_b.py's stats walk (beta = 0, the carried kernels). Then the five walks kind_t reads (A1st, A1all, Kst,
         KTst, KTall) run again with the relational kernel at each grid point: within the structural family u's mass
         goes to v in proportion to omega_q(t(u -> v)) x s(v), omega = exp(beta z), z the question's cosine to the
         type's vector standardized over the graph's types (RWr: the KB's relations, the mean over the pair's facts;
         RWe: the EM components, the null type at z = 0). KIND_T's order and the carried frontier's columns are
         rebuilt on each, and for each point the rows and every gold's held are recorded (at G > 0 with (b)'s
         restore). Checked: beta = 0 equals (b)'s filed rows and held golds, question for question; the relational
         kernel with every omega 1 equals the carried kernels (self-test). --limit Q: the first Q questions, every
         check, nothing written and no number logged. --g: development runs before (b)'s verdict is filed.
         --shard I/K: the I-th of K contiguous question ranges.
merge    Join a dataset's K parts into the file one run over every question writes.
verdict  Section 7's rule over the configurations x points, the verdict (a), and beside. Writes
         outputs/u2m/a/verdict_a.json and verdict_a.md.
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
import zu2m_b as MB  # noqa: E402  (section 11's carried frontier, plan, lists and restore, unchanged)
import zu2m_a_types as AT  # noqa: E402  (section 14's types: the EM, the entries' components, RWr's blocks)

U2E, U2C, F, P1H, D5, D5B = M.U2E, M.U2C, M.F, M.P1H, M.D5, M.D5B
log, utc, write_json, sha_file = M.log, M.utc, M.write_json, M.sha_file
DECLARED = "docs/U2M_POOLS_THREE_FAMILIES.md, section 14"
OUT = M.OUT / "a"
SIX, TRAIN5, NAMES, BASES = M.SIX, M.TRAIN5, M.NAMES, M.BASES
R = M.R
EMPTY = M.EMPTY
FORMS = ("RWr", "RWe")
BETAS = (0.5, 1.0, 2.0)
CONFIGS = (("carried", 0.0),) + tuple((f, b) for f in FORMS for b in BETAS)
NC = len(CONFIGS)
WALKS = ("A1st", "A1all", "Kst", "KTst", "KTall")
WALK_UNION = {"A1st": "st", "A1all": "all", "Kst": "st", "KTst": "st", "KTall": "all"}
BATCH_A = 100
VERDICTS_A = ("RELATIONAL_RAISED", "RELATIONAL_CHEAPER", "NO_GAIN")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def config_name(ci):
    form, beta = CONFIGS[ci]
    return "beta0" if ci == 0 else f"{form}@{beta:g}"


def eff_key(ci, typed):
    """The walk a configuration runs: RWr is RWe where the structural family has no relations."""
    form, beta = CONFIGS[ci]
    return ("RWe" if form == "RWr" and not typed else form), beta


def omega(q, vecs, beta, null):
    """(omega over the types, the top type's share of it): exp(beta z), z the cosine to each type's vector
    standardized over the types (unweighted); the null type, when there is one, last at omega 1."""
    c = np.asarray(vecs, np.float64) @ np.asarray(q, np.float64)
    sd = float(c.std())
    z = (c - c.mean()) / sd if sd > 0 else np.zeros_like(c)
    w = np.exp(float(beta) * z)
    top = float(w.max() / w.sum()) if w.size else 0.0
    return (np.concatenate([w, [1.0]]) if null else w), top


# ── the relational kernel (P1e's push; within the structural family each edge weighs omega x s(v)) ────────────


_K = {}


def kernels():
    if _K:
        return _K
    from numba import njit, prange
    import step4_pool_reach as S4
    _top = S4._top

    @njit(inline="always")
    def _w(e, v, mode, cls, bst, bln, trel, wq, steered, floor, ex):
        """An edge's weight: omega (mode 1: its component's; mode 2: the mean over its facts; mode 0: 1) times the
        walk's own s(v) (1, or the steer floor + ex[v])."""
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

    @njit(inline="always")
    def _rspread(ip, cc, u, m, nf, mode, cls, bst, bln, trel, wq, steered, floor, ex, r, flag, touched, queue, deg,
                 eps, n, nt, tail):
        """u's mass m over its neighbours in one family: 1/nf of it, in proportion to w within the family. With
        every w = 1 (or w = the steer) the statements are step 4's _spread's (P1e's _tspread's)."""
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
            r[v] += base * _w(e, v, mode, cls, bst, bln, trel, wq, steered, floor, ex)
            if flag[v] & 1 == 0 and r[v] >= eps * max(deg[v], 1):
                flag[v] |= 1
                queue[tail % n] = v
                tail += 1
        return nt, tail

    @njit
    def _rpush(seeds, wts, ip0, c0, ip1, c1, ip2, c2, deg, mode, cls, bst, bln, trel, wq, steered, floor, ex, alpha,
               eps, p, r, flag, touched, queue):
        """P1e's _tpush (step 4e's _wpush when unsteered), the structural family weighed by mode."""
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
            nt, tail = _rspread(ip0, c0, u, m, nf, mode, cls, bst, bln, trel, wq, steered, floor, ex, r, flag, touched,
                                queue, deg, eps, n, nt, tail)
            nt, tail = _rspread(ip1, c1, u, m, nf, 0, cls, bst, bln, trel, wq, steered, floor, ex, r, flag, touched,
                                queue, deg, eps, n, nt, tail)
            nt, tail = _rspread(ip2, c2, u, m, nf, 0, cls, bst, bln, trel, wq, steered, floor, ex, r, flag, touched,
                                queue, deg, eps, n, nt, tail)
        return nt

    @njit(parallel=True)
    def rppr_batch(seeds_ptr, seeds_flat, w_flat, st_ptr, st_nodes, st_ex, floors, steered, wq_ptr, wq_flat, mode,
                   cls, bst, bln, trel, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, bmax, n_chunks, out_nodes, out_mass,
                   out_count):
        nq = seeds_ptr.size - 1
        n = deg.size
        for t in prange(n_chunks):
            lo = (nq * t) // n_chunks
            hi = (nq * (t + 1)) // n_chunks
            if lo < hi:
                p = np.zeros(n)
                r = np.zeros(n)
                flag = np.zeros(n, dtype=np.uint8)
                touched = np.empty(n, dtype=np.int32)
                queue = np.empty(n, dtype=np.int32)
                ex = np.zeros(n)
                for q in range(lo, hi):
                    seeds = seeds_flat[seeds_ptr[q]:seeds_ptr[q + 1]]
                    if seeds.size == 0:
                        out_count[q] = 0
                        continue
                    if steered:
                        for k in range(st_ptr[q], st_ptr[q + 1]):
                            ex[st_nodes[k]] = st_ex[k]
                    nt = _rpush(seeds, w_flat[seeds_ptr[q]:seeds_ptr[q + 1]], ip0, c0, ip1, c1, ip2, c2, deg, mode, cls,
                                bst, bln, trel, wq_flat[wq_ptr[q]:wq_ptr[q + 1]], steered, floors[q], ex, alpha, eps,
                                p, r, flag, touched, queue)
                    _avail, cnt = _top(p, flag, touched, nt, bmax, out_nodes[q])
                    out_count[q] = cnt
                    for k in range(cnt):
                        out_mass[q, k] = p[out_nodes[q, k]]
                    for k in range(nt):
                        v = touched[k]
                        p[v] = 0.0
                        r[v] = 0.0
                        flag[v] = 0
                    if steered:
                        for k in range(st_ptr[q], st_ptr[q + 1]):
                            ex[st_nodes[k]] = 0.0

    _K.update(rppr_batch=rppr_batch, rpush=_rpush)
    return _K


def no_types():
    return (np.zeros(1, np.int8), np.zeros(1, np.int64), np.ones(1, np.int32), np.zeros(1, np.int16))


def rwalk(union, restarts, steers, wqs, mode, ty, alpha, eps, bmax, threads):
    """Per restart: (nodes by mass descending then position, their raw masses). steers None: unsteered; wqs: each
    question's omega over the types; ty: (component per entry, block start, block length, typed relation)."""
    import step4_pool_reach as S4
    nq = len(restarts)
    if nq == 0:
        return []
    s_ptr, s_flat = S4.flat([np.asarray(r[0], np.int64) for r in restarts])
    w_flat = np.concatenate([np.asarray(r[1], np.float64) for r in restarts])
    if steers is None:
        t_ptr, t_nodes = np.zeros(nq + 1, np.int64), np.zeros(0, np.int64)
        t_ex, floors = np.zeros(0), np.ones(nq)
    else:
        t_ptr, t_nodes = S4.flat([np.asarray(s[0], np.int64) for s in steers])
        t_ex = np.concatenate([np.asarray(s[1], np.float64) for s in steers])
        floors = np.asarray([s[2] for s in steers], np.float64)
    q_ptr = np.concatenate([[0], np.cumsum([np.asarray(w).size for w in wqs])]).astype(np.int64)
    wq_flat = np.concatenate([np.asarray(w, np.float64) for w in wqs]) if q_ptr[-1] else np.ones(1)
    cls, bst, bln, trel = ty
    out = np.full((nq, bmax), -1, dtype=np.int64)
    om = np.zeros((nq, bmax))
    oc = np.zeros(nq, dtype=np.int64)
    kernels()["rppr_batch"](s_ptr, s_flat, w_flat, t_ptr, t_nodes, t_ex, floors, steers is not None, q_ptr, wq_flat,
                            int(mode), cls, bst, bln, trel, *union.args, float(alpha), float(eps), int(bmax),
                            max(1, min(nq, 4 * threads)), out, om, oc)
    return [(out[q, :oc[q]].copy(), om[q, :oc[q]].copy()) for q in range(nq)]


def scaled(x):
    """walk_mass's form of a walk: its nodes and its masses scaled to sum to 1."""
    nodes, mass = x
    tot = mass.sum()
    return nodes, (mass / tot if tot > 0 else mass)


def decide_a(level, cost, beta, car_level, car_cost):
    """Section 14's verdict (a) from the pick's level, cost and beta, and the carried pick's level and cost."""
    if level is not None and level > car_level + 1e-9:
        return "RELATIONAL_RAISED"
    if level is not None and abs(level - car_level) < 1e-9 and beta > 0 and cost < car_cost - 1e-9:
        return "RELATIONAL_CHEAPER"
    return "NO_GAIN"


# ── the carried family ───────────────────────────────────────────────────────


def carried_family(g=None, dev=False):
    """((c) and (e)'s record, the carried frontier's points, the G of (b)'s pick, (b)'s verdict or None)."""
    vrec, _car, _pts, names = MB.carried()
    p = MB.OUT / "verdict_b.json"
    if p.exists():
        vb = json.loads(p.read_text(encoding="utf-8"))
        if vb.get("script_sha256") != MB.script_sha256() or vb.get("zu2m_sha256") != M.script_sha256():
            raise SystemExit("outputs/u2m/b/verdict_b.json is not the committed zu2m_b.py's verdict")
        if vb["points"] != names:
            raise SystemExit("(b)'s verdict is not over the carried frontier's points")
        if g is not None and int(g) != int(vb["G"]):
            raise SystemExit(f"--g {g}: (b)'s verdict carries G = {vb['G']}")
        return vrec, names, int(vb["G"]), vb
    if not dev or g is None:
        raise SystemExit("(b)'s verdict is not filed (outputs/u2m/b/verdict_b.json); a development run may name --g")
    if int(g) not in MB.GS:
        raise SystemExit(f"--g {g}: not one of section 11's G")
    return vrec, names, int(g), None


# ── stats (the laptop, or Modal in shards) ───────────────────────────────────


def stats_stage(name, threads=2, limit=0, shard=None, g_arg=None):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if D.SOURCES != ("A1st", "A1all", "Kst", "Kall", "R0") or F.CONSTRUCTION != "KIND_T":
        raise SystemExit("P1d's sources are not A1st, A1all, Kst, Kall and R0, or P1f's construction is not KIND_T")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    _vrec, names, G_, vb = carried_family(g_arg, dev=bool(limit))
    x2m = M.load_u2m(name)
    if x2m is None:
        raise SystemExit(f"{name}: U2m's filed arrays are not under outputs/u2m/stats")
    a2m, r2m = x2m
    xb = MB.load_b(name, names)
    if xb is None:
        raise SystemExit(f"{name}: (b)'s filed arrays are not under outputs/u2m/b/stats")
    ab, rb = xb
    if vb is not None and vb["stats"][name]["file_sha256"] != rb["file_sha256"]:
        raise SystemExit(f"{name}: (b)'s filed arrays are not the ones its verdict read")
    gi = MB.GS.index(G_)
    plan = MB.point_plan(name, a2m, r2m, names)
    NP = len(plan)
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    if list(op.cfg_h["graph_regimes"]["FULL"]) != list(S4.FAMILIES) or \
            tuple(S4.FAMILIES) != ("structural", "ner", "knn"):
        raise SystemExit("FULL is not the three families in their order")
    cells, own_k = M.cells_of(construction)
    if cells != r2m["cells"] or own_k != r2m["own_cell"]:
        raise SystemExit(f"{name}: the cells are not U2m's")
    if any(pl["k"] != own_k for pl in plan):
        raise SystemExit("(a) reads the carried frontier on the own construction; a point is on another")
    recipes = sorted({pl["r"] for pl in plan})
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
    tarr, trec = AT.load_types(name, desc)
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
    for u_, _fams in D.UNIONS:
        if not (np.array_equal(unions[u_].args[0], adj[0][0]) and np.array_equal(unions[u_].args[1], adj[0][1])):
            raise SystemExit(f"{name}: union {u_}'s structural family is not the store's")
    typed = None
    if g is None and getattr(st, "typed_indptr", None) is not None:
        typed = (np.asarray(st.typed_indptr, np.int64), np.asarray(st.typed_col), np.asarray(st.typed_rel),
                 np.asarray(st.typed_dir))
    if bool(trec["typed"]) != (typed is not None) or int(trec["entries"]) != int(adj[0][0][-1]) or \
            tarr["cls"].size != int(adj[0][0][-1]):
        raise SystemExit(f"{name}: the types are not this structural family's")
    mu = np.asarray(tarr["mu"], np.float64)
    if typed is not None:
        rvec, _vocab = AT.relation_vectors(name)
        ty = (np.ascontiguousarray(tarr["cls"], np.int8), np.ascontiguousarray(tarr["bst"], np.int64),
              np.ascontiguousarray(tarr["bln"], np.int32), np.ascontiguousarray(typed[2], np.int16))
    else:
        rvec = None
        ty = (np.ascontiguousarray(tarr["cls"], np.int8),) + no_types()[1:]
    untyped = [adj[i] for i, f in enumerate(S4.FAMILIES) if not (typed is not None and f == "structural")]
    keys = sorted({eff_key(ci, typed is not None) for ci in range(1, NC)}, key=lambda k: (FORMS.index(k[0]), k[1]))
    log(f"{name}/{carve}: prepared; {NP} points at G = {G_}; typed {typed is not None}; walks {keys} "
        f"({time.time() - t0:.0f}s)")
    n_st = int(unions["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    nq = min(limit, len(prep.seeds)) if limit else len(prep.seeds)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(a2m["ptr"][:nq + 1], ptr) or not np.array_equal(ab["ptr"][:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: U2m's or (b)'s golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    rows_f = np.asarray(ab["rows"][:, :, gi], np.int64)
    held_f = np.asarray(ab["held"][:, :, gi], bool)
    alpha, eps, bmax = float(E.ALPHA), float(S4.EPS), int(D.NMAX)
    bud = U2E.budgets()
    acc = {"rows": [], "held": [], "top": [], "changed": [], "entity": []}
    t = time.time()
    done = 0
    for lo in range(q0, q1, BATCH_A):
        js = list(range(lo, min(q1, lo + BATCH_A)))
        srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
        steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
        k0, a1 = K.restarts_batch(prep, js, linked, unions, srcs, constant)
        stz = [K.steer(prep, j, constant) for j in js]
        qs = [D5B.unit(np.asarray(prep.qemb[j], np.float64)) for j in js]
        wal = {}
        for key in keys:
            form, beta = key
            vecs, null, mode = (rvec, False, 2) if form == "RWr" else (mu, True, 1)
            om = [omega(q, vecs, beta, null) for q in qs]
            res = {w: rwalk(unions[WALK_UNION[w]], a1 if w.startswith("A1") else k0,
                            stz if w.startswith("KT") else None, [o[0] for o in om], mode, ty, alpha, eps, bmax,
                            threads) for w in WALKS}
            wal[key] = (res, [o[1] for o in om])
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
            r0o, iout = F.outside(src[4][0], B), F.outside(I, B)
            rq = np.zeros((NC, NP), np.int64)
            hq = np.zeros((gj.size, NC, NP), bool)
            tq = np.zeros(NC)
            cq = np.zeros(NC, bool)
            kt0 = None
            for ci in range(NC):
                if ci == 0:
                    sc, stc = src, steered[i]
                else:
                    res, tops = wal[eff_key(ci, typed is not None)]
                    sc = (scaled(res["A1st"][i]), scaled(res["A1all"][i]), scaled(res["Kst"][i]), src[3], src[4])
                    stc = dict(steered[i])
                    stc["KTst"], stc["KTall"] = res["KTst"][i][0], res["KTall"][i][0]
                    tq[ci] = tops[i]
                out, p1d = K.p1e_orders(sc, stc, route, knd, n_st, constant)
                kt = np.asarray(out[F.CONSTRUCTION], np.int64)
                want_kt = p1d["ROUTEst"] if knd == "entity" else out["RRF5T"]
                if not np.array_equal(kt, np.asarray(want_kt, np.int64)):
                    raise SystemExit(f"{name}/{carve} question {j}: KIND_T is not ROUTEst or RRF5T by the kind")
                if ci == 0:
                    kt0 = kt
                cq[ci] = not np.array_equal(kt, kt0)
                rec = [kt, np.asarray(out["RRF5T"], np.int64),
                       np.asarray(D.rrf([p1d["ROUTEst"], out["RRF5T"]], constant), np.int64)]
                rows.add(np.concatenate(rec + [src[4][0], I, n2]))
                orr = {}
                for r in recipes:
                    full = F.outside(rec[r], B)
                    fill = M.fill_of(full, I.size, B.size)
                    ords, _sizes, U, mem = U2E.question_orders(full, n1, cr, cos1, r0o, iout, n2, first, rows.read, q,
                                                               constant, D.rrf)
                    if not U2E.orders_exact(ords, B, full, n1, U, mem):
                        raise SystemExit(f"{name}/{carve} question {j}: an order is not exactly its sources' nodes")
                    if ci == 0 and fill != int(a2m["fill"][j, own_k * R + r]):
                        raise SystemExit(f"{name}/{carve} question {j}: today's fill is not U2m's")
                    orr[r] = (ords, fill)
                for pi, pl in enumerate(plan):
                    ords, fill = orr[pl["r"]]
                    order = np.asarray(ords[NAMES[pl["o"]]], np.int64)
                    _bn, kind, prm = bud[pl["b"]]
                    kq = int(U2E.budget_k(kind, prm, np.asarray([fill]))[0])
                    kk = min(int(order.size), kq)
                    keep = int(P1H.keep_counts(np.asarray([B.size]), np.asarray([kk]), pl["t"])[0])
                    rest = MB.restore(MB.build_lists(B, order, kk, keep, untyped, typed), kk, keep, G_) \
                        if G_ > 0 else EMPTY
                    if rest.size and not np.isin(rest, order[keep:kk]).all():
                        raise SystemExit(f"{name}/{carve} question {j}: a restored row is outside order[keep:kk]")
                    pos = U2E.places(gj, in_b, order)
                    held = (pos == 0) | ((pos >= 1) & (pos <= keep)) | np.isin(gj, rest)
                    nrow = int(B.size + keep + rest.size)
                    if ci == 0 and (kk != int(pl["kk"][j]) or keep != int(pl["keep"][j]) or nrow != int(rows_f[j, pi])
                                    or not np.array_equal(held, held_f[g0:g1, pi])):
                        raise SystemExit(f"{name}/{carve} question {j}: beta = 0 of {pl['name']} is not (b)'s filed "
                                         f"rows and held golds at G = {G_}")
                    rq[ci, pi] = nrow
                    hq[:, ci, pi] = held
            acc["rows"].append(rq)
            acc["held"].append(hq)
            acc["top"].append(tq)
            acc["changed"].append(cq)
            acc["entity"].append(knd == "entity")
            done += 1
            if done % 100 == 0:
                log(f"{name}/{carve}: {done}/{q1 - q0} ({time.time() - t:.0f}s)")
    head = {"declared_in": DECLARED, "stage": "U2m (a)", "dataset": name, "carve": carve, "graph_detail": desc,
            "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(), "zu2m_b_sha256": MB.script_sha256(),
            "zu2m_a_types_sha256": AT.script_sha256(), "zu2e_sha256": U2E.script_sha256(),
            "u2m_stats_sha256": r2m["file_sha256"], "b_stats_sha256": rb["file_sha256"],
            "types_sha256": trec["file_sha256"],
            "verdict_b_sha256": sha_file(MB.OUT / "verdict_b.json") if vb is not None else None,
            "points": names, "G": G_, "configs": [config_name(ci) for ci in range(NC)],
            "walks_run": [f"{f}@{b:g}" for f, b in keys], "typed": typed is not None, "threads": threads,
            "alpha": alpha, "eps": eps, "nmax": bmax}
    checks = {"graph_equal_u2m": True, "types_of_this_graph": True, "seeds_equal_g0": True, "kind_t_by_kind": True,
              "orders_exact": True, "fill_equal_u2m": True, "beta0_equal_b": True, "restored_inside_cut": True,
              "rows_served_verified": bool(limit)}
    if limit:
        log(f"{name}/{carve}: development run on questions {q0}-{q1 - 1} of {nq} ({time.time() - t:.0f}s): every "
            "check passed, every served row verified; nothing written")
        return 0
    arr = arrays_of(acc, ptr[q0:q1 + 1], NP)
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


def arrays_of(acc, gptr, np_):
    nq = len(acc["rows"])
    return {"rows": np.asarray(acc["rows"], np.int64).reshape(nq, NC, np_),
            "held": np.concatenate(acc["held"]).reshape(-1, NC, np_) if nq else np.zeros((0, NC, np_), bool),
            "top": np.asarray(acc["top"], np.float64).reshape(nq, NC),
            "changed": np.asarray(acc["changed"], bool).reshape(nq, NC),
            "entity": np.asarray(acc["entity"], bool), "ptr": np.asarray(gptr, np.int64)}


def summary_of(arr):
    return {"rows_sum": int(arr["rows"].sum()), "held_sum": int(arr["held"].sum()),
            "changed_sum": int(arr["changed"].sum()), "questions": int(arr["rows"].shape[0])}


def write_stats(name, carve, arr, head, checks, seconds, extra=None):
    od = OUT / "stats"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{name}__{carve}.npz"
    tmp = od / f"{name}__{carve}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, p)
    rec = {**head, "questions": int(arr["rows"].shape[0]), "golds": int(arr["ptr"][-1] - arr["ptr"][0]),
           "file_sha256": sha_file(p), "summary": summary_of(arr), "checks": checks, "seconds": seconds,
           **(extra or {}), "utc": utc()}
    write_json(od / f"{name}__{carve}.json", rec)
    log(f"{name}/{carve} ({rec['questions']} q, {seconds:.0f}s): written")
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
                v for c, v in r["checks"].items() if c != "rows_served_verified"):
            raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
        if r["part"] != i or r["parts"] != k or (recs and r["q0"] != recs[-1]["q1"]):
            raise SystemExit(f"part {i}: not the {i}-th of {k} question ranges")
        with np.load(p) as z:
            parts.append({kx: z[kx] for kx in z.files})
        recs.append(r)
    if recs[0]["q0"] != 0 or recs[-1]["q1"] != recs[0]["questions_all"]:
        raise SystemExit(f"{name}/{carve}: the parts do not cover every question")
    gptr = [np.asarray([0], np.int64)]
    for i, p in enumerate(parts):
        q = np.asarray(p["ptr"], np.int64)
        if int(q[0]) != int(gptr[-1][-1]):
            raise SystemExit(f"part {i}: its ptr does not continue the previous part's")
        gptr.append(q[1:])
    arr = {"ptr": np.concatenate(gptr)}
    for kx in ("rows", "held", "top", "changed", "entity"):
        arr[kx] = np.concatenate([p[kx] for p in parts])
    head = {kx: recs[0][kx] for kx in recs[0] if kx not in (
        "part", "parts", "q0", "q1", "questions_all", "file_sha256", "checks", "seconds", "utc")}
    return write_stats(name, carve, arr, head, recs[0]["checks"], round(sum(r["seconds"] for r in recs), 1),
                       {"shards": k, "part_sha256": [r["file_sha256"] for r in recs],
                        "seconds_longest_part": max(r["seconds"] for r in recs)})


def load_a(name, names, g):
    carve = U2C.carve_of(name)
    rp = OUT / "stats" / f"{name}__{carve}.json"
    if not rp.exists():
        return None
    rec = json.loads(rp.read_text(encoding="utf-8"))
    p = OUT / "stats" / f"{name}__{carve}.npz"
    if rec.get("script_sha256") != script_sha256() or sha_file(p) != rec["file_sha256"] or not all(
            v for c, v in rec["checks"].items() if c != "rows_served_verified"):
        raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
    if rec["points"] != names or int(rec["G"]) != int(g) or rec["configs"] != [config_name(c) for c in range(NC)]:
        raise SystemExit(f"{rp}: not the carried family's points, its G, or section 14's grid")
    with np.load(p) as z:
        a = {k: z[k] for k in z.files}
    if summary_of(a) != rec["summary"]:
        raise SystemExit(f"{p}: the arrays do not give its summary")
    return a, rec


# ── the verdict (the laptop) ─────────────────────────────────────────────────


def verdict_stage():
    t0 = time.time()
    vrec, names, G_, vb = carried_family()
    aa, ra, missing = {}, {}, []
    for d in SIX:
        x = load_a(d, names, G_)
        if x is None:
            missing.append(d)
        else:
            aa[d], ra[d] = x
            if ra[d]["verdict_b_sha256"] != sha_file(MB.OUT / "verdict_b.json"):
                raise SystemExit(f"{d}: (a)'s stats were not run on the filed (b) verdict")
    if missing:
        raise SystemExit(f"INCOMPLETE: no (a) stats for {missing}")
    cap = float(vrec["cap"])
    car_pick, car_level, car_cost = vb["pick"], float(vb["level"]), float(vb["cost"])
    w = F.weights()
    if w != vrec["weights"]:
        raise SystemExit("P1f's weights are not the ones (c) and (e)'s verdict used")
    ceil = np.asarray([vrec["ceilings"][d] for d in TRAIN5], np.float64)
    NP = len(names)
    cands = [(ci, pi) for ci in range(NC) for pi in range(NP)]   # beta = 0 first, then the forms, beta, the points

    def cname(ci, pi):
        return f"{names[pi]}|G{G_}|{config_name(ci)}"

    ptrs = {d: aa[d]["ptr"] - aa[d]["ptr"][0] for d in SIX}
    tab = {d: {"ALL": np.asarray([U2E.all_of(aa[d]["held"][:, ci, pi], ptrs[d]) for ci, pi in cands]),
               "rows": np.asarray([round(float(aa[d]["rows"][:, ci, pi].astype(np.float64).mean()), 1)
                                   for ci, pi in cands])} for d in SIX}
    for pi, nm in enumerate(names):
        i = cands.index((0, pi))
        filed = vb["every_candidate"][f"{nm}|G{G_}"]
        for d in SIX:
            if tab[d]["ALL"][i] != filed[d]["ALL"] or tab[d]["rows"][i] != filed[d]["rows"]:
                raise SystemExit(f"{d}: beta = 0 of {nm} is not (b)'s filed candidate at G = {G_}")
    A = np.stack([tab[d]["ALL"] for d in TRAIN5], axis=1)
    RW = np.stack([tab[d]["rows"] for d in TRAIN5], axis=1)
    costs = np.asarray([F.cost(dict(zip(TRAIN5, r.tolist())), w) for r in RW], np.float64)
    cost_r = np.asarray([round(float(c), 1) for c in costs])
    fr = U2E.frontier(A, costs, ceil)
    lv, kk = M.pick_of(fr, cost_r, cap)
    if lv is None:
        raise SystemExit("no point within the cap: the carried pick at beta = 0 should be one")
    ci_, pi_ = cands[kk]
    va = decide_a(lv, float(cost_r[kk]), CONFIGS[ci_][1], car_level, car_cost)
    pick = cname(ci_, pi_) if va != "NO_GAIN" else f"{car_pick}|beta0"
    if va == "NO_GAIN" and (ci_ != 0 or f"{names[pi_]}|G{G_}" != car_pick):
        raise SystemExit(f"NO_GAIN, but the frontier's pick is {cname(ci_, pi_)}, not the carried {car_pick}")

    def point(t, tgt, k_):
        if k_ is None:
            return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": None}
        return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": cname(*cands[k_]),
                "cost": float(cost_r[k_]), "within_cap": bool(cost_r[k_] <= cap + 1e-9),
                "s1sel": {d: {"ALL": float(tab[d]["ALL"][k_]), "rows": float(tab[d]["rows"][k_])} for d in TRAIN5},
                "webqsp_s1eval": {"ALL": float(tab["webqsp"]["ALL"][k_]), "rows": float(tab["webqsp"]["rows"][k_])}}

    every = {cname(ci, pi): {"cost": float(cost_r[i]),
                             **{d: {"ALL": float(tab[d]["ALL"][i]), "rows": float(tab[d]["rows"][i])} for d in SIX}}
             for i, (ci, pi) in enumerate(cands)}
    pc = names.index(car_pick.split("|")[0])
    at_carried = {config_name(ci): every[cname(ci, pc)] for ci in range(NC)}
    # the pick's misses against the carried pick's, by section 7's causes on the carried walk
    misses = {}
    cell, r, o, b, _t = MB.parse(names[pc])
    for d in SIX:
        a2m, r2m = M.load_u2m(d)
        k = r2m["own_cell"] if cell == "own" else r2m["cells"].index(cell)
        _rw, held1 = M.held_of(a2m, k, r, o, b, 1.0)
        reach = (a2m["pos"] >= 0).any(axis=1)
        cause = np.where(held1, "TAU_CUT", np.where(reach, "BUDGET", "UNREACHED"))
        h0 = aa[d]["held"][:, 0, pc]
        h1 = aa[d]["held"][:, ci_, pi_]
        misses[d] = {"carried_missed": int((~h0).sum()), "pick_missed": int((~h1).sum()),
                     "recovered": {c: int((h1 & ~h0 & (cause == c)).sum()) for c in U2E.CAUSES},
                     "lost": {c: int((h0 & ~h1 & (cause == c)).sum()) for c in U2E.CAUSES}}
    moved = {d: {config_name(ci): round(float(aa[d]["changed"][:, ci].mean()), 4) for ci in range(1, NC)}
             for d in SIX}
    tops = {d: {config_name(ci): round(float(aa[d]["top"][:, ci].mean()), 4) for ci in range(1, NC)} for d in SIX}
    types = {}
    for d in SIX:
        _arr, tr = AT.load_types(d)
        if tr["file_sha256"] != ra[d]["types_sha256"]:
            raise SystemExit(f"{d}: the types are not the ones (a)'s stats read")
        types[d] = {kx: tr.get(kx) for kx in ("steps", "converged", "fit_pairs", "fit_pairs_with_direction",
                                               "fit_mean_cos", "entries", "entries_null", "relations",
                                               "entries_with_several_facts", "nmi_em_type_relation")}
        types[d]["fit_size_min_max"] = [int(min(tr["fit_sizes"])), int(max(tr["fit_sizes"]))]
    rec = {"declared_in": DECLARED, "stage": "U2m (a)", "script_sha256": script_sha256(),
           "zu2m_sha256": M.script_sha256(), "zu2m_b_sha256": MB.script_sha256(),
           "zu2m_a_types_sha256": AT.script_sha256(), "verdict_b_sha256": sha_file(MB.OUT / "verdict_b.json"),
           "verdict_a": va, "pick": pick, "level": lv, "cost": float(cost_r[kk]),
           "form": CONFIGS[ci_][0] if va != "NO_GAIN" else "carried",
           "beta": CONFIGS[ci_][1] if va != "NO_GAIN" else 0.0,
           "carried_b": {"pick": car_pick, "level": car_level, "cost": car_cost}, "cap": cap, "G": G_,
           "points": names, "configs": [config_name(c) for c in range(NC)], "candidates": len(cands), "weights": w,
           "ceilings": dict(zip(TRAIN5, ceil.tolist())), "frontier": [point(t, tg, k_) for t, tg, k_ in fr],
           "every_candidate": every, "at_carried_point": at_carried, "misses": misses, "kind_t_moved": moved,
           "top_type_share": tops, "types": types,
           "stats": {d: {"file_sha256": ra[d]["file_sha256"], "seconds": ra[d]["seconds"]} for d in SIX},
           "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "verdict_a.json", rec)
    (OUT / "verdict_a.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"U2m (a): {va}; carried {pick} at level {lv}, cost {cost_r[kk]} ({time.time() - t0:.0f}s)")
    return 0


def report_md(rec):
    c = rec["carried_b"]
    L = [f"# U2m (a): {rec['verdict_a']}", "",
         f"Declared in {rec['declared_in']}. Carried forward: `{rec['pick']}` at level {rec['level']}, cost "
         f"{rec['cost']} (cap {rec['cap']}).", "",
         f"(b) carried `{c['pick']}` at level {c['level']}, cost {c['cost']}. {rec['candidates']} candidates: "
         f"{len(rec['points'])} points of the carried frontier at G = {rec['G']} x {rec['configs']}.", "",
         "## The frontier", "", "| level | pick | cost | within the cap |", "|---:|---|---:|---|"]
    for p in rec["frontier"]:
        if p["pick"] is None:
            L.append(f"| {p['level']} | none | | |")
        else:
            L.append(f"| {p['level']} | `{p['pick']}` | {p['cost']} | {p['within_cap']} |")
    L += ["", "## At the carried pick's point (ALL / rows; webqsp is s1eval)", "",
          "| configuration | cost | " + " | ".join(SIX) + " |", "|---|---:|" + "---|" * len(SIX)]
    for cn, x in rec["at_carried_point"].items():
        L.append(f"| {cn} | {x['cost']} | " + " | ".join(f"{x[d]['ALL']} / {x[d]['rows']}" for d in SIX) + " |")
    L += ["", "## The pick's misses against the carried pick's", "",
          "| dataset | carried missed | pick missed | recovered (TAU_CUT/BUDGET/UNREACHED) | lost |",
          "|---|---:|---:|---|---|"]
    for d, x in rec["misses"].items():
        L.append(f"| {d} | {x['carried_missed']} | {x['pick_missed']} | "
                 + "/".join(str(v) for v in x["recovered"].values()) + " | "
                 + "/".join(str(v) for v in x["lost"].values()) + " |")
    L += ["", "## The types (beside)", "",
          "| dataset | EM steps | converged | mean cos | component sizes | null entries | NMI with relation |",
          "|---|---:|---|---:|---|---:|---:|"]
    for d, x in rec["types"].items():
        L.append(f"| {d} | {x['steps']} | {x['converged']} | {x['fit_mean_cos']} | {x['fit_size_min_max']} | "
                 f"{x['entries_null']} | {x['nmi_em_type_relation']} |")
    L += ["", "webqsp's s1eval is beside; it never moves the choice.", ""]
    return "\n".join(L)


# ── self-test ────────────────────────────────────────────────────────────────


def _brute_push(seeds, wts, fams, deg, mode, ty, wq, steer, alpha, eps):
    """The relational push in plain Python, FIFO in seed order, as the kernel orders its pushes."""
    cls, bst, bln, trel = ty
    n = deg.size
    p = [0.0] * n
    r = [0.0] * n
    queued = [False] * n
    queue = []
    for s, w in zip(seeds, wts):
        r[s] += w
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
        if ru < eps * max(int(deg[u]), 1):
            continue
        r[u] = 0.0
        p[u] += alpha * ru
        if deg[u] == 0:
            continue
        m = (1.0 - alpha) * ru
        nf = sum(1 for ip, _c in fams if ip[u + 1] > ip[u])
        for fi, (ip, cc) in enumerate(fams):
            a, b = int(ip[u]), int(ip[u + 1])
            if b <= a:
                continue
            ws = []
            for e in range(a, b):
                v = int(cc[e])
                w = 1.0
                if fi == 0 and mode == 1:
                    w = wq[int(cls[e])]
                elif fi == 0 and mode == 2:
                    w = sum(wq[int(trel[t])] for t in range(int(bst[e]), int(bst[e]) + int(bln[e]))) / int(bln[e])
                if steer is not None:
                    w = w * (steer[0] + steer[1][v])
                ws.append(w)
            tot = sum(ws)
            for e, w in zip(range(a, b), ws):
                v = int(cc[e])
                r[v] += m / (nf * tot) * w
                if not queued[v] and r[v] >= eps * max(int(deg[v]), 1):
                    queued[v] = True
                    queue.append(v)
    return np.asarray(p)


def _selftest():
    import p1e_anchor_kind as K
    import zu2m_d as MD
    E, S4 = K.E, K.S4
    rng = np.random.default_rng(29)
    alpha, eps = float(E.ALPHA), 1e-4
    n_bit = n_brute = n_w = 0
    for case in range(60):
        n = int(rng.integers(8, 60))
        typed_on = case % 2 == 0
        fams, typed = MD._toy(rng, n, typed_on)
        un = MD._Union(fams)
        ip0, c0 = fams[0]
        e0 = int(ip0[-1])
        cls = rng.integers(0, 5, max(e0, 1)).astype(np.int8)
        if typed is not None:
            bst, bln = AT.blocks(ip0, c0, typed)
            if not bst.size:
                bst, bln = np.zeros(1, np.int64), np.ones(1, np.int32)
            ty = (cls, bst, bln, np.ascontiguousarray(typed[2], np.int16))
            nt_ = int(typed[2].max()) + 1 if typed[2].size else 1
        else:
            ty = (cls,) + no_types()[1:]
            nt_ = 1
        steered = case % 3 == 0
        nq = int(rng.integers(1, 4))
        restarts, steers = [], []
        for _q in range(nq):
            ns = int(rng.integers(0, 4)) if case % 5 else int(rng.integers(1, 4))
            sd = np.unique(rng.integers(0, n, ns)).astype(np.int64)
            w = rng.random(sd.size) + 0.1
            restarts.append((sd, w / w.sum() if w.size else w))
            sn = np.unique(rng.integers(0, n, int(rng.integers(0, n)))).astype(np.int64)
            steers.append((sn, rng.random(sn.size), float(rng.random() * 0.1 + 0.01)))
        stz = steers if steered else None
        modes = (0, 1) + ((2,) if typed is not None else ())
        for mode in modes:
            nw = 6 if mode == 1 else (nt_ if mode == 2 else 1)
            # every omega 1: the carried kernels, bit for bit
            ones = [np.ones(nw) for _q in range(nq)]
            res = rwalk(un, restarts, stz, ones, mode, ty, alpha, eps, n, 2)
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
                o_nodes, o_mass = res[q]
                if not np.array_equal(o_nodes, want) or not np.array_equal(o_mass, p0[want]):
                    raise SystemExit(f"self-test case {case} mode {mode}: with every omega 1 the relational kernel "
                                     "is not the carried kernel")
                n_bit += 1
            # omega != 1: a plain Python push
            wqs = [rng.random(nw) * 3 + 0.05 for _q in range(nq)]
            res = rwalk(un, restarts, stz, wqs, mode, ty, alpha, eps, n, 2)
            for q in range(nq):
                sd, w = restarts[q]
                if not sd.size:
                    continue
                stq = None
                if steered:
                    ex = np.zeros(n)
                    ex[steers[q][0]] = steers[q][1]
                    stq = (steers[q][2], ex)
                bp = _brute_push([int(x) for x in sd], [float(x) for x in w], fams, un.deg, mode, ty,
                                 [float(x) for x in wqs[q]], stq, alpha, eps)
                o_nodes, o_mass = res[q]
                if not np.allclose(bp[o_nodes], o_mass, rtol=1e-12, atol=0) or \
                        set(np.flatnonzero(bp > 0).tolist()) != set(o_nodes.tolist()):
                    raise SystemExit(f"self-test case {case} mode {mode}: the relational kernel is not the plain push")
                n_brute += 1
                n_w += int(mode > 0 and o_nodes.size > 1)
    if min(n_bit, n_brute, n_w) == 0:
        raise SystemExit("self-test: a kind of case never ran")
    # omega: standardized, the null type at 1, beta = 0 gives ones
    vecs = AT.unit_rows(rng.normal(size=(9, 7)))
    q = AT.unit_rows(rng.normal(size=(1, 7)))[0]
    w, top = omega(q, vecs, 1.0, True)
    z = np.log(w[:-1])
    if w[-1] != 1.0 or abs(z.mean()) > 1e-12 or abs(z.std() - 1.0) > 1e-12 or abs(top - w[:-1].max() / w[:-1].sum()) > 0:
        raise SystemExit("self-test: omega is not exp(beta z) with z standardized and the null type at 1")
    if not np.array_equal(omega(q, vecs, 0.0, False)[0], np.ones(9)):
        raise SystemExit("self-test: beta = 0 does not give omega 1")
    # the verdict rule
    if decide_a(0.96, 1800, 1.0, 0.955, 1812.6) != "RELATIONAL_RAISED" or \
            decide_a(0.955, 1700, 1.0, 0.955, 1812.6) != "RELATIONAL_CHEAPER" or \
            decide_a(0.955, 1700, 0.0, 0.955, 1812.6) != "NO_GAIN" or \
            decide_a(0.955, 1812.6, 2.0, 0.955, 1812.6) != "NO_GAIN" or \
            decide_a(0.950, 900, 2.0, 0.955, 1812.6) != "NO_GAIN":
        raise SystemExit("self-test: the verdict rule is not section 14's")
    if [config_name(c) for c in range(NC)] != ["beta0", "RWr@0.5", "RWr@1", "RWr@2", "RWe@0.5", "RWe@1", "RWe@2"]:
        raise SystemExit("self-test: the grid is not section 14's order")
    if eff_key(1, False) != ("RWe", 0.5) or eff_key(1, True) != ("RWr", 0.5):
        raise SystemExit("self-test: RWr is not RWe on an untyped structural family")
    log(f"zu2m_a self-test: with every omega 1 the relational kernel equals step 4e's and P1e's kernels bit for bit "
        f"({n_bit} toy walks, typed and untyped, steered and unsteered, modes 0-2); with omega != 1 it equals a "
        f"plain Python push within 1e-12 ({n_brute} walks, {n_w} weighted with mass beyond one node); omega, the "
        "grid's order and the verdict rule check")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("stats", "merge", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--g", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.stage == "stats":
        if a.dataset not in SIX:
            raise SystemExit(f"stats: a dataset of {SIX}")
        if a.g is not None and not a.limit:
            raise SystemExit("--g is for development runs (--limit) only")
        shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
        return stats_stage(a.dataset, a.threads, a.limit, shard, a.g)
    if a.stage == "merge":
        return merge_stage(a.dataset, a.parts)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
