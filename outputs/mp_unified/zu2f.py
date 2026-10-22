"""U2f: where each question's pool stops, without the golds (docs/U2F_POOL_STOP.md). Pools only: nothing trains,
nothing is fit, and no rule reads a gold, today's fill or the dataset's name.

    python outputs/mp_unified/zu2f.py stats DATASET [--threads 2] [--limit Q] [--shard I/K]
    python outputs/mp_unified/zu2f.py merge DATASET --parts K
    python outputs/mp_unified/zu2f.py verdict
    python outputs/mp_unified/zu2f.py --selftest

stats    One dataset's selection carve (s1sel; webqsp's s1eval, beside) on U2e's graph. Each question is rebuilt by
         U2e's own steps and checked against U2e's filed arrays (section 6). Per question: |B|, the lengths of kt_cos
         and kt, the 43 bands' thresholds and the rows table (how many nodes outside B each SET rule keeps). Per gold:
         its place in kt_cos and in kt (0 in B, p >= 1 the p-th node, -1 outside), its entry level e and its cosine c.
         --limit Q: the first Q questions, every check, nothing written and no number logged. --shard I/K: the I-th of
         K contiguous question ranges, into outputs/u2f/stats/parts/ (outputs/mp_unified/zu2f_modal.py runs them on
         Modal).
merge    Join a dataset's K parts into the file one run over every question writes.
verdict  The 21,070 rules (section 4) on the five training datasets' s1sel, webqsp's s1eval beside, the frontier, the
         pick at U2e's level and the verdict (section 5). Writes outputs/u2f/verdict.json and verdict.md.
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
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zu2e as U2E  # noqa: E402  (U2e's orders, places, checks, filed arrays, frontier; through it U2c, P1g, P1f, P1h)

U2C, PG, F, P1H, D5, D5B = U2E.U2C, U2E.PG, U2E.F, U2E.P1H, U2E.D5, U2E.D5B
log, utc, write_json, sha_file = U2E.log, U2E.utc, U2E.write_json, U2E.sha_file
DECLARED = "docs/U2F_POOL_STOP.md"
OUT = ROOT / "outputs" / "u2f"
SIX, TRAIN5 = U2E.SIX, U2E.TRAIN5
NAMES = U2E.NAMES
KTC, KT = NAMES.index("kt_cos"), NAMES.index("kt")     # U2e's pick's order (M = 0) and KIND_T's
K0 = (0, 25, 50, 100, 200)
TAUS_F = (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0)
BETAS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0)
DELTAS = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2)
BANDS = ("none",) + tuple(f"b{b:g}d{d:g}" for b in BETAS for d in DELTAS)
XS = (0.5, 1.0, 2.0)
DEPTH_ORDERS = ("kt_cos", "kt")
SHAPE = (len(K0), len(TAUS_F), len(BANDS))
N_SET = SHAPE[0] * SHAPE[1] * SHAPE[2]
E_CAP = 1.0 - 1e-12            # a listed node's entry level is at most this, so tau = 1 keeps every listed node
NO_R = 1 << 40                 # r outside kt_cos
NO_C = -2.0                    # c outside U_0, and a non-finite cosine
MASS_TOL = 1e-9
MIN_CUT = P1H.MIN_CUT
VERDICTS = ("FEWER_ROWS", "SAME_ROWS", "NO_CHANGE")
EMPTY = U2E.EMPTY
CH = 301                       # SET rules per block when the DEPTH rules are read
assert len(BANDS) == 43 and N_SET == 3010 and MIN_CUT == 0.10


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the steered walk's masses ────────────────────────────────────────────────

_KERNEL = []


def mass_kernel():
    """P1e's tppr_batch, also writing each listed node's mass p, as P1b's wppr_mass_batch writes its walk's. The push
    (P1e's _tpush) and the top-k (step 4's _top) are called unchanged."""
    if _KERNEL:
        return _KERNEL[0]
    import p1e_anchor_kind as K
    from numba import njit, prange
    tpush, top = K._tpush, K.S4._top

    @njit(parallel=True)
    def tppr_mass_batch(seeds_ptr, seeds_flat, w_flat, st_ptr, st_nodes, st_ex, floors, ip0, c0, ip1, c1, ip2, c2,
                        deg, alpha, eps, bmax, n_chunks, out_nodes, out_mass, out_count):
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
                    for k in range(st_ptr[q], st_ptr[q + 1]):
                        ex[st_nodes[k]] = st_ex[k]
                    nt = tpush(seeds, w_flat[seeds_ptr[q]:seeds_ptr[q + 1]], ip0, c0, ip1, c1, ip2, c2, deg,
                               floors[q], ex, alpha, eps, p, r, flag, touched, queue)
                    _avail, cnt = top(p, flag, touched, nt, bmax, out_nodes[q])
                    out_count[q] = cnt
                    for k in range(cnt):
                        out_mass[q, k] = p[out_nodes[q, k]]
                    for k in range(nt):
                        v = touched[k]
                        p[v] = 0.0
                        r[v] = 0.0
                        flag[v] = 0
                    for k in range(st_ptr[q], st_ptr[q + 1]):
                        ex[st_nodes[k]] = 0.0

    _KERNEL.append(tppr_mass_batch)
    return tppr_mass_batch


def twalk_mass(union, restarts, steers, threads):
    """P1e's twalk with masses: [(nodes by mass descending then position, masses scaled to sum 1)] per restart, as
    P1b's walk_mass scales its walk's; an empty restart walks to ()."""
    import p1e_anchor_kind as K
    S4 = K.S4
    nq = len(restarts)
    if nq == 0:
        return []
    s_ptr, s_flat = S4.flat([r[0] for r in restarts])
    w_flat = np.concatenate([r[1] for r in restarts]).astype(np.float64)
    t_ptr, t_nodes = S4.flat([st[0] for st in steers])
    t_ex = np.concatenate([st[1] for st in steers]).astype(np.float64)
    floors = np.asarray([st[2] for st in steers], np.float64)
    out = np.full((nq, K.NMAX), -1, dtype=np.int64)
    mass = np.zeros((nq, K.NMAX), dtype=np.float64)
    cnt = np.zeros(nq, dtype=np.int64)
    mass_kernel()(s_ptr, s_flat, w_flat, t_ptr, t_nodes, t_ex, floors, *union.args, float(K.E.ALPHA), float(S4.EPS),
                  int(K.NMAX), max(1, min(nq, 4 * threads)), out, mass, cnt)
    res = []
    for q in range(nq):
        c = int(cnt[q])
        m = mass[q, :c].copy()
        tot = m.sum()
        res.append((out[q, :c].copy(), m / tot if tot > 0 else m))
    return res


def kt_mass_batch(prep, js, linked, unions, srcs, constant, threads):
    """{KTst, KTall}: per question of js, P1e's steered K walk on each union with its masses."""
    import p1e_anchor_kind as K
    k0, _a1 = K.restarts_batch(prep, js, linked, unions, srcs, constant)
    stz = [K.steer(prep, j, constant) for j in js]
    return {f"KT{u}": twalk_mass(unions[u], k0, stz, threads) for u, _fams in K.UNIONS}


# ── one question ─────────────────────────────────────────────────────────────


def question_orders(full, n1, cr, cos1, r0, iout, n2, first, read, q, c, rrf):
    """U2e's question_orders, unchanged, that also returns U's cosines cu (section 3's c on U_0)."""
    full = np.asarray(full, np.int64)
    n1 = np.asarray(n1, np.int64)
    U = np.unique(np.concatenate([full, n1, np.asarray(r0, np.int64), np.asarray(iout, np.int64),
                                  np.asarray(n2, np.int64)]))
    rest = np.setdiff1d(U, n1, assume_unique=True)
    cu = np.empty(U.size, np.float32)
    if n1.size:
        cu[np.searchsorted(U, n1)] = cos1
    if rest.size:
        cu[np.searchsorted(U, rest)] = np.concatenate(
            [D5B.unit(read(rest[a:a + U2E.CHUNK])) @ q for a in range(0, rest.size, U2E.CHUNK)])
    base = np.isin(U, full) | np.isin(U, n1) | np.isin(U, r0) | np.isin(U, iout)
    fu = np.full(U.size, U2E.BIG, np.int64)
    if np.asarray(n2).size:
        k = np.minimum(np.searchsorted(n2, U), n2.size - 1)
        hit = n2[k] == U
        fu[hit] = first[k[hit]]
    n1c = n1[np.argsort(np.asarray(cr, np.int64), kind="stable")]
    out = {"kt": full, "kt_n1": rrf([full, n1c], c, nmax=U2E.BIG)}
    sizes, mem = {}, {}
    for m in U2E.MS:
        mk = base | (fu < m)
        ids = U[mk]
        cos_m = ids[np.lexsort((ids, -cu[mk].astype(np.float64)))]
        out[U2E.col_name("cos", m)] = cos_m
        out[U2E.col_name("kt_cos", m)] = rrf([full, cos_m], c, nmax=U2E.BIG)
        out[U2E.col_name("kt_cos_r0", m)] = rrf([full, cos_m, r0], c, nmax=U2E.BIG)
        sizes[m], mem[m] = int(mk.sum()), mk
    return out, sizes, U, mem, cu


def kind_t_sources(src, kt, route, knd):
    """KIND_T's own sources for one question, each (nodes, masses) in its order (section 3): ROUTEst's two on an
    entity question on ROUTE's condition, A1st on another entity question, RRF5T's five on a passage question."""
    a1st, a1all, kst, _kall, r0 = src
    if knd == "entity":
        return [kst, a1st] if route else [a1st]
    return [a1st, a1all, kt["KTst"], kt["KTall"], r0]


def masses_ok(nodes, mass):
    """Finite, non-negative, non-increasing along the order and summing to 1 (an empty source has none)."""
    nodes, mass = np.asarray(nodes), np.asarray(mass, np.float64)
    if nodes.size != mass.size or np.unique(nodes).size != nodes.size:
        return False
    if mass.size == 0:
        return True
    return bool(np.isfinite(mass).all() and (mass >= 0).all() and (np.diff(mass) <= 0).all()
                and abs(float(mass.sum()) - 1.0) <= MASS_TOL)


def entry_levels(V, sources):
    """e(v) for each node of V (sorted): the least, over the sources that list v, of the share of the source's mass
    on the nodes before v in its order, at most E_CAP; inf where no source lists v."""
    e = np.full(V.size, np.inf)
    for nodes, mass in sources:
        nodes = np.asarray(nodes, np.int64)
        if nodes.size == 0 or V.size == 0:
            continue
        before = np.minimum(np.concatenate([[0.0], np.cumsum(np.asarray(mass, np.float64))[:-1]]), E_CAP)
        k = np.minimum(np.searchsorted(V, nodes), V.size - 1)
        hit = V[k] == nodes
        np.minimum.at(e, k[hit], before[hit])
    return e


def node_values(B, ktc, U0, c0, sources):
    """Section 3 for one question: V (the nodes outside B that U_0 holds or a source lists, sorted), r, e, c, and how
    many cosines on U_0 are not finite (counted, then outside every band)."""
    B = np.asarray(B, np.int64)
    if np.isin(np.asarray(U0, np.int64), B).any():
        raise SystemExit("U_0 holds a node of B")
    lst = [np.asarray(s[0], np.int64) for s in sources if np.asarray(s[0]).size]
    listed = np.setdiff1d(np.concatenate(lst), B) if lst else EMPTY
    V = np.union1d(np.asarray(U0, np.int64), listed)
    r = np.full(V.size, NO_R, np.int64)
    ktc = np.asarray(ktc, np.int64)
    if ktc.size:
        k = np.searchsorted(V, ktc)
        if not (k < V.size).all() or not np.array_equal(V[np.minimum(k, V.size - 1)], ktc):
            raise SystemExit("kt_cos holds a node outside V")
        r[k] = np.arange(1, ktc.size + 1)
    c = np.full(V.size, NO_C, np.float32)
    U0 = np.asarray(U0, np.int64)
    if U0.size:
        c[np.searchsorted(V, U0)] = np.asarray(c0, np.float32)
    bad = ~np.isfinite(c)
    c[bad] = NO_C
    return V, r, entry_levels(V, sources), c, int(bad.sum())


def thresholds(cB):
    """The 43 bands' thresholds for one question (section 3): inf (no band), then Q_beta(c(B)) - delta for beta in
    BETAS and, within each, delta in DELTAS; numpy's linear quantile in float64 over B's finite cosines. With no finite
    cosine in B, every band is empty."""
    c = np.asarray(cB, np.float64)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return np.full(len(BANDS), np.inf)
    q = np.quantile(c, np.asarray(BETAS, np.float64))
    return np.concatenate([[np.inf], (q[:, None] - np.asarray(DELTAS, np.float64)[None, :]).ravel()])


def rows_table(r, e, c, thr):
    """keep[a, b, t]: how many nodes of V the SET rule (K0[a], TAUS_F[b], band t) keeps (rows outside B)."""
    c = np.asarray(c, np.float64)
    o = np.argsort(-c, kind="stable")
    cs = c[o]
    nb = np.searchsorted(-cs, -np.asarray(thr, np.float64), side="right")      # how many c >= theta
    T = np.empty(SHAPE, np.int32)
    for a, k0 in enumerate(K0):
        mr = r <= k0
        for b, tau in enumerate(TAUS_F):
            m = mr | (e < tau)
            cn = np.concatenate([[0], np.cumsum(~m[o])])
            T[a, b] = int(m.sum()) + cn[nb]
    return T


def gold_values(gj, V, r, e, c):
    """Each gold's r, e and c: its node's, or (NO_R, inf, NO_C) when V does not hold it."""
    gj = np.asarray(gj, np.int64)
    if V.size == 0:
        return (np.full(gj.size, NO_R, np.int64), np.full(gj.size, np.inf), np.full(gj.size, NO_C, np.float32))
    k = np.minimum(np.searchsorted(V, gj), V.size - 1)
    hit = V[k] == gj
    return (np.where(hit, r[k], NO_R), np.where(hit, e[k], np.inf), np.where(hit, c[k], np.float32(NO_C)))


def gold_entry(g, sources):
    """A gold's entry level by a per-node loop over the sources (the check's second count)."""
    best = math.inf
    for nodes, mass in sources:
        nodes = np.asarray(nodes, np.int64)
        idx = np.flatnonzero(nodes == g)
        if idx.size:
            acc = 0.0
            for x in np.asarray(mass, np.float64)[:int(idx[0])]:
                acc += float(x)
            best = min(best, min(acc, E_CAP))
    return best


# ── stats (Modal, in shards; or the laptop) ──────────────────────────────────

PER_QUESTION = ("bs", "len_ktc", "len_kt", "n_v", "entity", "route", "nonfinite", "thr", "keep")
PER_GOLD = ("pos_ktc", "pos_kt", "e", "c")
ARRAYS = PER_QUESTION + PER_GOLD + ("ptr",)
HEAD_KEYS = ("declared_in", "stage", "dataset", "carve", "graph_detail", "script_sha256", "zu2e_sha256",
             "u2e_stats_sha256", "zu2c_sha256", "p1e_script_sha256", "freeze_RECORD_SHA256", "threads", "k0", "taus",
             "bands", "nmax", "chunk")


def load_u2e(name):
    x = U2E.load_u2e(name)
    if x is None:
        raise SystemExit(f"{name}: U2e's filed arrays are not under outputs/u2e/stats")
    return x


def stats_stage(name, threads=2, limit=0, shard=None):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if tuple(D.SOURCES) != ("A1st", "A1all", "Kst", "Kall", "R0") or F.CONSTRUCTION != "KIND_T":
        raise SystemExit("P1d's sources or P1f's construction are not the ones section 3 names")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    e2, r2 = load_u2e(name)
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
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
                    raise SystemExit(f"{ds_.name}: U2f swaps {name}'s graph only")
                return swap
            return load0(ds_, family, cache_dir)

        S4.m3b_pools.load_or_build_store = load
    if desc != r2["graph_detail"]:
        raise SystemExit(f"{name}: the graph is not the one U2e's filed run read")
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
    if prep.dense_ids.shape[1] < 1000 or prep.splade_ids.shape[1] < 1000:
        raise SystemExit(f"{name}/{carve}: retrieval lists are not 1000 deep")
    bases = op.m3b_compile.base_rows(construction, prep0.dense_ids, prep0.splade_ids, op.m3a, op.m3b_contract,
                                     constant)
    _U0, _BQ, bs, _inc0, _f = E.union_pools(name, carve, pop, prep0, bases)     # base, seeds: step 4d's
    del bases, _U0
    if any(not np.array_equal(np.asarray(a, np.int64), np.asarray(b, np.int64)) for a, b in zip(prep.seeds, prep0.seeds)):
        raise SystemExit(f"{name}/{carve}: the seeds on the graph are not G0's")
    inc = [np.asarray(p, np.int64) for p in prep.pools]
    linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [EMPTY] * len(bs)
    log(f"{name}/{carve}: prepared ({time.time() - t0:.0f}s)")

    cache = {}
    stores_all, unions = G.open_unions(cache, op, name, union)
    adj = [(np.ascontiguousarray(stores_all[f].indptr, np.int64), np.ascontiguousarray(stores_all[f].col))
           for f in S4.FAMILIES]
    n_st = int(unions["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    nq = min(limit, len(bs)) if limit else len(bs)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if e2["bs"].size != len(bs) or not np.array_equal(e2["ptr"][:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: U2e's questions and golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    pq = {k: [] for k in PER_QUESTION}
    pg = {k: [] for k in PER_GOLD}
    k0_arr = np.asarray(K0, np.int64)
    t = time.time()
    for lo in range(q0, q1, S4.BATCH):
        js = list(range(lo, min(q1, lo + S4.BATCH)))
        srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
        steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
        ktm = kt_mass_batch(prep, js, linked, unions, srcs, constant, threads)
        for i, j in enumerate(js):
            src, _nn, (L, Ls) = srcs[i]
            route = D.routes(L, Ls)
            knd = K.kind(Ls, prep.seeds[j], nonly)
            kt = {u: ktm[u][i] for u in ("KTst", "KTall")}
            if not all(np.array_equal(kt[u][0], steered[i][u]) for u in kt):
                raise SystemExit(f"{name}/{carve} question {j}: the copied kernel's KT orders are not P1e's")
            o = np.asarray(K.p1e_orders(src, steered[i], route, knd, n_st, constant)[0][F.CONSTRUCTION], np.int64)
            B = np.unique(np.asarray(bs[j], np.int64))
            take = max(0, int(inc[j].size) - B.size)
            full = F.outside(o, B)
            fill = full[:take]
            w_B = D5.weights_of(K.steer(prep, j, constant), B)
            q = D5B.unit(np.asarray(prep.qemb[j], np.float64))
            n1, _cnt, _ws, _up, _inv = D5B.n1_table(adj, B, w_B)
            cos1 = D5B.unit(nodes.read(n1)) @ q if n1.size else np.empty(0, np.float32)
            cr = D5B.ranks_desc(cos1, n1) if n1.size else EMPTY
            gj = golds[j]
            g0, g1 = int(ptr[j]), int(ptr[j + 1])
            listed = np.isin(gj, prep.dense_ids[j]) | np.isin(gj, prep.splade_ids[j])
            one = np.isin(gj, n1)
            kp = PG.positions_in(gj, full)
            ktp = np.where(kp < full.size, kp, -1).astype(np.int64)
            in_i = np.isin(gj, inc[j])
            r0 = F.outside(src[4][0], B)
            iout = F.outside(inc[j], B)
            n2, first, taken, hubs = U2E.second_link(adj, n1, cr, B)
            ords, sizes, U, mem, cu = question_orders(full, n1, cr, cos1, r0, iout, n2, first, nodes.read, q,
                                                      constant, D.rrf)
            if not U2E.orders_exact(ords, B, full, n1, U, mem):
                raise SystemExit(f"{name}/{carve} question {j}: an order is not exactly its sources' nodes")
            in_b = np.isin(gj, B)
            pos = np.stack([U2E.places(gj, in_b, ords[nm]) for nm in NAMES], axis=1)
            nf = np.full(gj.size, -1, np.int32)
            if n2.size:
                k = np.minimum(np.searchsorted(n2, gj), n2.size - 1)
                hit = n2[k] == gj
                nf[hit] = first[k[hit]]
            mine_q = (B.size, fill.size, inc[j].size, [ords[nm].size for nm in NAMES], n1.size, r0.size, iout.size,
                      [int((first < m).sum()) for m in U2E.M_GRID], [sizes[m] for m in U2E.MS], taken, hubs)
            mine_g = (pos, nf, listed, one, ktp, in_i)
            if not (all(np.array_equal(np.asarray(v), e2[k][j]) for k, v in zip(U2E.PER_QUESTION, mine_q))
                    and all(np.array_equal(v, e2[k][g0:g1]) for k, v in zip(U2E.PER_GOLD, mine_g))):
                raise SystemExit(f"{name}/{carve} question {j}: a value is not U2e's filed one")
            # section 3: the nodes' r, e and c, B's thresholds and the rows table
            sources = kind_t_sources(src, kt, route, knd)
            if not all(masses_ok(*s) for s in sources):
                raise SystemExit(f"{name}/{carve} question {j}: a source's masses are not a distribution in its order")
            ktc = ords["kt_cos"]
            U0, c0 = U[mem[0]], cu[mem[0]]
            V, r, e, c, bad = node_values(B, ktc, U0, c0, sources)
            cB = D5B.unit(nodes.read(B)) @ q if B.size else np.empty(0, np.float32)
            thr = thresholds(cB)
            keep = rows_table(r, e, c, thr)
            if not np.array_equal(keep[:, 0, 0], np.minimum(k0_arr, ktc.size)):
                raise SystemExit(f"{name}/{carve} question {j}: SET(k0, 0, none) is not B and kt_cos's first k0")
            rg, eg, cg = gold_values(gj, V, r, e, c)
            want_r = np.where(in_b, 0, np.where(rg < NO_R, rg, -1))
            ku = np.minimum(np.searchsorted(U0, gj), max(U0.size - 1, 0))
            in_u0 = (U0[ku] == gj) & ~in_b if U0.size else np.zeros(gj.size, bool)
            cu0 = np.where(np.isfinite(c0[ku]), c0[ku], np.float32(NO_C)) if U0.size else cg
            if not (np.array_equal(want_r, pos[:, KTC])
                    and np.array_equal(cg[in_u0], cu0[in_u0]) and (cg[~in_u0] == np.float32(NO_C)).all()
                    and all(float(eg[x]) == gold_entry(int(gj[x]), sources) for x in np.flatnonzero(~in_b))):
                raise SystemExit(f"{name}/{carve} question {j}: a gold's r, e or c is not its node's")
            pq["bs"].append(B.size)
            pq["len_ktc"].append(ktc.size)
            pq["len_kt"].append(ords["kt"].size)
            pq["n_v"].append(V.size)
            pq["entity"].append(knd == "entity")
            pq["route"].append(bool(route))
            pq["nonfinite"].append(bad + int((~np.isfinite(cB)).sum()))
            pq["thr"].append(thr)
            pq["keep"].append(keep)
            pg["pos_ktc"].append(pos[:, KTC])
            pg["pos_kt"].append(pos[:, KT])
            pg["e"].append(eg)
            pg["c"].append(cg.astype(np.float32))
        if (lo // S4.BATCH) % 10 == 9:
            log(f"{name}/{carve}: {js[-1] + 1}/{nq} ({time.time() - t:.0f}s)")
    arr = arrays_of(pq, pg, ptr[q0:q1 + 1])
    head = {"declared_in": DECLARED, "stage": "U2f", "dataset": name, "carve": carve, "graph_detail": desc,
            "script_sha256": script_sha256(), "zu2e_sha256": U2E.script_sha256(),
            "u2e_stats_sha256": r2["file_sha256"], "zu2c_sha256": U2C.script_sha256(),
            "p1e_script_sha256": K.script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
            "threads": threads, "k0": list(K0), "taus": list(TAUS_F), "bands": list(BANDS), "nmax": int(K.NMAX),
            "chunk": U2E.CHUNK}
    checks = {"graph_equal_u2e": True, "seeds_equal_g0": True, "u2e_values_reproduced": True,
              "kt_kernel_equals_p1e": True, "masses_are_distributions": True, "orders_exact": True,
              "set_k0_rows": True, "gold_values_are_their_nodes": True}
    if limit:
        log(f"{name}/{carve}: development run on questions {q0}-{q1 - 1} of {nq} ({time.time() - t:.0f}s): "
            "every check passed; nothing written")
        return 0
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


def arrays_of(pq, pg, ptr):
    a = {"bs": np.asarray(pq["bs"], np.int64), "len_ktc": np.asarray(pq["len_ktc"], np.int64),
         "len_kt": np.asarray(pq["len_kt"], np.int64), "n_v": np.asarray(pq["n_v"], np.int64),
         "entity": np.asarray(pq["entity"], bool), "route": np.asarray(pq["route"], bool),
         "nonfinite": np.asarray(pq["nonfinite"], np.int64),
         "thr": np.asarray(pq["thr"], np.float64).reshape(-1, len(BANDS)),
         "keep": np.asarray(pq["keep"], np.int32).reshape((-1,) + SHAPE),
         "pos_ktc": np.concatenate(pg["pos_ktc"]).astype(np.int32) if pg["pos_ktc"] else np.zeros(0, np.int32),
         "pos_kt": np.concatenate(pg["pos_kt"]).astype(np.int32) if pg["pos_kt"] else np.zeros(0, np.int32),
         "e": np.concatenate(pg["e"]).astype(np.float64) if pg["e"] else np.zeros(0),
         "c": np.concatenate(pg["c"]).astype(np.float32) if pg["c"] else np.zeros(0, np.float32),
         "ptr": np.asarray(ptr, np.int64)}
    return {k: a[k] for k in ARRAYS}


def all_of(held, ptr):
    return U2E.all_of(np.asarray(held, bool), ptr)


def summary_of(a):
    """The sizes, the kinds, the non-finite cosines, and ALL of B, of kt_cos whole and of the listed nodes."""
    ptr = a["ptr"]
    inb = a["pos_ktc"] == 0
    return {"B": round(float(a["bs"].mean()), 1), "V": round(float(a["n_v"].mean()), 1),
            "kt_cos": round(float(a["len_ktc"].mean()), 1), "kt": round(float(a["len_kt"].mean()), 1),
            "entity_questions": int(a["entity"].sum()), "route_questions": int((a["entity"] & a["route"]).sum()),
            "nonfinite_cosines": int(a["nonfinite"].sum()), "ALL_B": all_of(inb, ptr),
            "ALL_kt_cos": all_of(a["pos_ktc"] >= 0, ptr), "ALL_listed": all_of(inb | np.isfinite(a["e"]), ptr)}


def write_stats(name, carve, arr, head, checks, seconds, extra=None):
    od = OUT / "stats"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{name}__{carve}.npz"
    tmp = od / f"{name}__{carve}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, p)
    nq = int(arr["bs"].size)
    rec = {**head, "questions": nq, "golds": int(arr["ptr"][-1]), "file_sha256": sha_file(p),
           "summary": summary_of(arr), "checks": checks, "seconds": seconds, **(extra or {}), "utc": utc()}
    write_json(od / f"{name}__{carve}.json", rec)
    log(f"{name}/{carve} ({nq} q, {seconds:.0f}s): written")
    return 0


def merge_arrays(parts):
    """The arrays of one run over every question, from its shards' arrays in order."""
    out = {k: np.concatenate([p[k] for p in parts]) for k in PER_QUESTION + PER_GOLD}
    ptr = [np.asarray([0], np.int64)]
    for a, p in enumerate(parts):
        q = np.asarray(p["ptr"], np.int64)
        if q.size != p["bs"].shape[0] + 1 or int(q[0]) != int(ptr[-1][-1]):
            raise SystemExit(f"part {a}: its ptr does not continue the previous part's")
        ptr.append(q[1:])
    out["ptr"] = np.concatenate(ptr)
    return {k: out[k] for k in ARRAYS}


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
        if r.get("script_sha256") != script_sha256() or sha_file(p) != r["file_sha256"] or not all(r["checks"].values()):
            raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
        with np.load(p) as z:
            parts.append({a: z[a] for a in z.files})
        recs.append(r)
    head = {a: recs[0][a] for a in HEAD_KEYS}
    nq = recs[0]["questions_all"]
    for i, r in enumerate(recs):
        if {a: r[a] for a in HEAD_KEYS} != head or r["checks"] != recs[0]["checks"] or r["questions_all"] != nq:
            raise SystemExit(f"part {i}: another script, freeze, graph, thread count, check set or carve")
        if (r["part"], r["parts"]) != (i, k) or (r["q0"], r["q1"]) != U2C.shard_range(nq, i, k):
            raise SystemExit(f"part {i}: not the {i}-th of {k} question ranges")
    arr = merge_arrays(parts)
    e2, _r2 = load_u2e(name)
    if arr["bs"].size != nq or not np.array_equal(e2["ptr"], arr["ptr"]) or not np.array_equal(e2["bs"], arr["bs"]):
        raise SystemExit(f"{name}/{carve}: the merged parts do not cover U2e's questions and golds")
    return write_stats(name, carve, arr, head, recs[0]["checks"], round(sum(r["seconds"] for r in recs), 1),
                       {"shards": k, "part_sha256": [r["file_sha256"] for r in recs],
                        "seconds_longest_part": max(r["seconds"] for r in recs)})


# ── the rules and the verdict (the laptop) ───────────────────────────────────


def load_u2f(name):
    carve = U2C.carve_of(name)
    rp = OUT / "stats" / f"{name}__{carve}.json"
    if not rp.exists():
        return None
    rec = json.loads(rp.read_text(encoding="utf-8"))
    p = OUT / "stats" / f"{name}__{carve}.npz"
    if rec.get("script_sha256") != script_sha256() or sha_file(p) != rec["file_sha256"] or not all(rec["checks"].values()):
        raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
    with np.load(p) as z:
        a = {k: z[k] for k in z.files}
    if summary_of(a) != rec["summary"] or tuple(rec["bands"]) != BANDS or tuple(rec["taus"]) != TAUS_F \
            or tuple(rec["k0"]) != K0:
        raise SystemExit(f"{p}: not this script's grid, or the arrays do not give the record's summary")
    return a, rec


def set_name(ia, ib, it):
    return f"k{K0[ia]}:t{TAUS_F[ib]:g}:{BANDS[it]}"


def candidates():
    """[(name, spec)] in section 4's order: SET (k0, tau, band), then DEPTH (kt_cos then kt, x, the SET rule)."""
    sets = [(ia, ib, it) for ia in range(SHAPE[0]) for ib in range(SHAPE[1]) for it in range(SHAPE[2])]
    out = [(f"set:{set_name(*s)}", ("SET",) + s) for s in sets]
    for o in DEPTH_ORDERS:
        for x in XS:
            out += [(f"depth:{o}:x{x:g}:{set_name(*s)}", ("DEPTH", o, x, s)) for s in sets]
    return out


def _complete(h, ptr, has):
    """Per column of h [golds x m]: the questions with a gold whose golds are all held."""
    cs = np.zeros((h.shape[0] + 1, h.shape[1]), np.int64)
    np.cumsum(~h, axis=0, out=cs[1:])
    return ((cs[ptr[1:]] - cs[ptr[:-1]] == 0) & has[:, None]).sum(axis=0)


def table(a):
    """(complete questions, rows summed) of every candidate on one carve's arrays, columnwise, in candidates()'s
    order; ALL = complete / questions and mean rows = rows / questions, as P1g's summarize computes them."""
    ptr = np.asarray(a["ptr"], np.int64)
    ng = np.diff(ptr)
    nq = ng.size
    has = ng > 0
    qi = np.repeat(np.arange(nq), ng)
    inb = a["pos_ktc"] == 0
    pk = np.asarray(a["pos_ktc"], np.int64)                  # int64 first: NO_R does not fit the filed int32
    rg = np.where(pk >= 1, pk, NO_R)
    eg = np.asarray(a["e"], np.float64)
    cb = np.asarray(a["c"], np.float64)[:, None] >= a["thr"][qi]
    keep = np.asarray(a["keep"], np.int64).reshape(nq, N_SET)
    bsum = int(np.asarray(a["bs"], np.int64).sum())
    n = N_SET * (1 + len(DEPTH_ORDERS) * len(XS))
    comp, rows = np.empty(n, np.int64), np.empty(n, np.int64)
    i = 0
    for k0 in K0:
        base = inb | (rg <= k0)
        for tau in TAUS_F:
            comp[i:i + len(BANDS)] = _complete((base | (eg < tau))[:, None] | cb, ptr, has)
            i += len(BANDS)
    rows[:N_SET] = bsum + keep.sum(axis=0)
    for o in DEPTH_ORDERS:
        po = np.asarray(a["pos_ktc"] if o == "kt_cos" else a["pos_kt"], np.int64)[:, None]
        ln = np.asarray(a["len_ktc"] if o == "kt_cos" else a["len_kt"], np.int64)[:, None]
        for x in XS:
            for s0 in range(0, N_SET, CH):
                kq = np.rint(x * keep[:, s0:s0 + CH]).astype(np.int64)
                rows[i + s0:i + s0 + kq.shape[1]] = bsum + np.minimum(kq, ln).sum(axis=0)
                comp[i + s0:i + s0 + kq.shape[1]] = _complete(inb[:, None] | ((po >= 1) & (po <= kq[qi])), ptr, has)
            i += N_SET
    return comp, rows


def rounded(comp, rows, nq):
    """P1g's ALL (4 places) and mean rows (1 place), rounded as Python rounds."""
    return (np.asarray([round(float(c / nq), 4) for c in comp]), np.asarray([round(float(r / nq), 1) for r in rows]))


def rows_held(a, spec, qi=None):
    """One rule's per-question rows and per-gold held on one carve's arrays (section 4, directly)."""
    ptr = a["ptr"]
    if qi is None:
        qi = np.repeat(np.arange(ptr.size - 1), np.diff(ptr))
    bs = np.asarray(a["bs"], np.int64)
    inb = a["pos_ktc"] == 0
    ia, ib, it = spec[3] if spec[0] == "DEPTH" else spec[1:]
    n_q = np.asarray(a["keep"][:, ia, ib, it], np.int64)
    if spec[0] == "SET":
        pk = np.asarray(a["pos_ktc"], np.int64)
        rg = np.where(pk >= 1, pk, NO_R)
        held = inb | (rg <= K0[ia]) | (a["e"] < TAUS_F[ib]) | (a["c"].astype(np.float64) >= a["thr"][qi, it])
        return bs + n_q, held
    _k, o, x, _s = spec
    po = a["pos_ktc"] if o == "kt_cos" else a["pos_kt"]
    ln = np.asarray(a["len_ktc"] if o == "kt_cos" else a["len_kt"], np.int64)
    kq = np.rint(x * n_q).astype(np.int64)
    return bs + np.minimum(kq, ln), inb | ((po >= 1) & (po <= kq[qi]))


def summary(a, spec, nm):
    rows, held = rows_held(a, spec)
    return PG.summarize({"rows": rows[:, None], "held": held[:, None], "ptr": a["ptr"]}, [(nm, None, None)])[nm]


def target_m(nq, tgt):
    """The fewest complete questions whose ALL (P1g's rounding) reaches tgt."""
    for m in range(max(0, int(math.floor(tgt * nq)) - 2), nq + 1):
        if round(m / nq, 4) >= tgt - 1e-9:
            return m
    return None


def perfect_stop(a, po, tgt):
    """The room (reads golds; never a rule): the fewest mean rows any per-question stop down an order can have at
    ALL >= tgt, each question kept down to its deepest gold, the cheapest questions first, the others B alone."""
    ptr = np.asarray(a["ptr"], np.int64)
    nq = ptr.size - 1
    need = np.full(nq, np.inf)
    po = np.asarray(po, np.int64)
    for q in range(nq):
        p = po[ptr[q]:ptr[q + 1]]
        if p.size and (p >= 0).all():
            need[q] = float(p.max())
    m = target_m(nq, tgt)
    s = np.sort(need)
    if m is None or (m > 0 and not np.isfinite(s[m - 1])):
        return None
    return round(float((int(np.asarray(a["bs"], np.int64).sum()) + s[:m].sum()) / nq), 1)


def decide(k, cost_r, cap):
    """Section 5: the verdict of the frontier's point k at the incumbent's level."""
    if k is None or cost_r[k] > cap + 1e-9:
        return "NO_CHANGE"
    if cost_r[k] <= (1.0 - MIN_CUT) * cap + 1e-9:
        return "FEWER_ROWS"
    return "SAME_ROWS"


def families(cands):
    """{family: [candidate indices]} for the cheapest point of each (filed beside)."""
    out = {"set: band alone (k0 = 0, tau = 0)": [], "set: nuclei alone (k0 = 0, no band)": [],
           "set: k0 = 0 (band, nuclei or both)": [], "set: k0 > 0": []}
    for o in DEPTH_ORDERS:
        for x in XS:
            out[f"depth: {o}, x = {x:g}"] = []
    for i, (_nm, sp) in enumerate(cands):
        if sp[0] == "SET":
            _s, ia, ib, it = sp
            if ia == 0 and ib == 0:
                out["set: band alone (k0 = 0, tau = 0)"].append(i)
            if ia == 0 and it == 0:
                out["set: nuclei alone (k0 = 0, no band)"].append(i)
            out["set: k0 = 0 (band, nuclei or both)" if ia == 0 else "set: k0 > 0"].append(i)
        else:
            out[f"depth: {sp[1]}, x = {sp[2]:g}"].append(i)
    return out


def verdict_stage():
    t0 = time.time()
    arrs, recs, missing = {}, {}, []
    for d in SIX:
        x = load_u2f(d)
        if x is None:
            missing.append(d)
        else:
            arrs[d], recs[d] = x
    if missing:
        raise SystemExit(f"INCOMPLETE: no U2f stats for {missing}")
    v2e = json.loads((U2E.OUT / "verdict.json").read_text(encoding="utf-8"))
    if v2e.get("script_sha256") != U2E.script_sha256() or v2e.get("verdict") != "RAISED":
        raise SystemExit("outputs/u2e/verdict.json: not the committed zu2e.py's RAISED record")
    for d in SIX:
        r2 = json.loads((U2E.OUT / "stats" / f"{d}__{U2C.carve_of(d)}.json").read_text(encoding="utf-8"))
        if recs[d]["u2e_stats_sha256"] != r2["file_sha256"] or recs[d]["zu2e_sha256"] != U2E.script_sha256():
            raise SystemExit(f"{d}: U2f's stats were not checked against U2e's filed arrays here")
    cap, level, incumbent = float(v2e["pick_cost"]), float(v2e["level"]), v2e["pick"]
    ceil = np.asarray([v2e["ceilings"][d] for d in TRAIN5], np.float64)
    ceil_all = {**v2e["ceilings"], "webqsp": v2e["per_dataset"]["webqsp"]["ceiling"]}
    cands = candidates()
    names = [c[0] for c in cands]
    ALL, ROWS = {}, {}
    for d in SIX:
        comp, rows = table(arrs[d])
        ALL[d], ROWS[d] = rounded(comp, rows, int(arrs[d]["bs"].size))
        log(f"U2f verdict: {d} read ({time.time() - t0:.0f}s)")
    w = F.weights()
    A = np.stack([ALL[d] for d in TRAIN5], axis=1)
    costs = np.asarray([F.cost({d: float(ROWS[d][i]) for d in TRAIN5}, w) for i in range(len(cands))], np.float64)
    cost_r = np.asarray([round(float(c), 1) for c in costs])
    pts = U2E.frontier(A, costs, ceil)
    lv = [k for t, _tg, k in pts if abs(t - level) < 1e-9]
    if len(lv) != 1:
        raise SystemExit(f"U2e's level {level} is not one of U2e's levels")
    k_pick = lv[0]
    verdict = decide(k_pick, cost_r, cap)
    checked = {}

    def summ(i, d):
        """P1g's summary of candidate i on d, checked equal to the columnwise table's ALL and rows."""
        if (i, d) not in checked:
            s = summary(arrs[d], cands[i][1], names[i])
            if s["ALL"] != float(ALL[d][i]) or s["rows"] != float(ROWS[d][i]):
                raise SystemExit(f"{names[i]} on {d}: the columnwise table is not P1g's summary")
            checked[(i, d)] = s
        return checked[(i, d)]

    def point(t, tgt, k):
        if k is None:
            return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": None}
        return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": names[k], "cost": float(cost_r[k]),
                "within_cap": bool(cost_r[k] <= cap + 1e-9), "s1sel": {d: summ(k, d) for d in TRAIN5},
                "webqsp_s1eval": summ(k, "webqsp")}

    frontier = [point(t, tg, k) for t, tg, k in pts]
    within = [p["level"] for p in frontier if p["pick"] is not None and p["within_cap"]]
    by_family = {}
    for f, idx in families(cands).items():
        idx = np.asarray(idx)
        by_family[f] = [None if k is None else {"pick": names[idx[k]], "cost": float(cost_r[idx[k]])}
                        for _t, _tg, k in U2E.frontier(A[idx], costs[idx], ceil)]
    per_dataset = {}
    for d in SIX:
        rows_at = {}
        for t in U2E.LEVELS:
            tg = round(min(t, float(ceil_all[d]) - PG.TOL), 4)
            i = np.flatnonzero(ALL[d] >= tg - 1e-9)
            if not i.size:
                rows_at[str(t)] = None
                continue
            k = int(i[np.argmin(ROWS[d][i])])
            rows_at[str(t)] = {"target": tg, "rows": float(ROWS[d][k]), "ALL": float(ALL[d][k]), "candidate": names[k]}
        per_dataset[d] = {"ceiling": float(ceil_all[d]), "fewest_rows": rows_at}
    tg_pick = dict(zip(TRAIN5, np.round(np.minimum(level, ceil - PG.TOL), 4).tolist()))
    room = {}
    for o in DEPTH_ORDERS:
        rws = {d: perfect_stop(arrs[d], arrs[d]["pos_ktc" if o == "kt_cos" else "pos_kt"], tg_pick[d]) for d in TRAIN5}
        room[o] = {"rows": rws, "cost": None if any(v is None for v in rws.values()) else round(F.cost(rws, w), 1)}
    pick_cost = None if k_pick is None else float(cost_r[k_pick])
    share = None
    if pick_cost is not None and room["kt_cos"]["cost"] is not None and cap > room["kt_cos"]["cost"]:
        share = round((cap - pick_cost) / (cap - room["kt_cos"]["cost"]), 4)
    rec = {"declared_in": DECLARED, "stage": "U2f", "script_sha256": script_sha256(), "zu2e_sha256": U2E.script_sha256(),
           "verdict": verdict, "level": level, "cap": cap, "incumbent": incumbent,
           "pick": None if k_pick is None else names[k_pick], "pick_cost": pick_cost,
           "pick_s1sel": None if k_pick is None else {d: summ(k_pick, d) for d in TRAIN5},
           "pick_webqsp_s1eval": None if k_pick is None else summ(k_pick, "webqsp"),
           "incumbent_s1sel": v2e["pick_s1sel"], "incumbent_webqsp_s1eval": v2e["pick_webqsp_s1eval"],
           "candidates": {"set": N_SET, "depth": len(cands) - N_SET}, "weights": w,
           "ceilings": dict(zip(TRAIN5, ceil.tolist())), "levels": list(U2E.LEVELS), "frontier": frontier,
           "highest_level_within_cap": max(within) if within else None, "by_family": by_family,
           "per_dataset": per_dataset, "room": {**room, "targets": tg_pick, "share_captured": share},
           "stats": {d: {"file_sha256": recs[d]["file_sha256"], "summary": recs[d]["summary"]} for d in SIX},
           "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "verdict.json", rec)
    (OUT / "verdict.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"U2f: {verdict}: {rec['pick']} at level {level} (cost {pick_cost}; U2e's pick {cap}) "
        f"({time.time() - t0:.0f}s)")
    return 0


def report_md(rec):
    L = [f"# U2f: {rec['verdict']}", "",
         f"Pick at level {rec['level']}: `{rec['pick']}`, cost {rec['pick_cost']} (U2e's pick `{rec['incumbent']}`: "
         f"{rec['cap']}). Declared in {rec['declared_in']}.", ""]
    if rec["pick"] is not None:
        L += ["| dataset | pick ALL | rows | rows p90 | U2e's pick ALL | rows |", "|---|---:|---:|---:|---:|---:|"]
        for d in TRAIN5:
            p, c = rec["pick_s1sel"][d], rec["incumbent_s1sel"][d]
            L.append(f"| {d} | {p['ALL']} | {p['rows']} | {p['rows_p90']} | {c['ALL']} | {c['rows']} |")
        p, c = rec["pick_webqsp_s1eval"], rec["incumbent_webqsp_s1eval"]
        L.append(f"| webqsp (s1eval, beside) | {p['ALL']} | {p['rows']} | {p['rows_p90']} | {c['ALL']} | {c['rows']} |")
    rm = rec["room"]
    L += ["", f"The room (reads golds): a perfect stop at these targets costs {rm['kt_cos']['cost']} in kt_cos and "
          f"{rm['kt']['cost']} in kt; the pick captures {rm['share_captured']} of the room in kt_cos. The highest "
          f"level within {rec['cap']}: {rec['highest_level_within_cap']}.", "",
          "## The frontier (every training dataset at min(level, its ceiling - 0.005))", "",
          "| level | cost | within cap | pick | " + " | ".join(f"{d} ALL (rows)" for d in TRAIN5) + " | webqsp ALL (rows) |",
          "|---|---:|---|---|" + "---|" * (len(TRAIN5) + 1)]
    for e in rec["frontier"]:
        if e["pick"] is None:
            L.append(f"| {e['level']} | - | - | none meets | " + " | ".join("-" for _ in TRAIN5) + " | - |")
            continue
        L.append(f"| {e['level']} | {e['cost']} | {e['within_cap']} | `{e['pick']}` | "
                 + " | ".join(f"{e['s1sel'][d]['ALL']} ({e['s1sel'][d]['rows']})" for d in TRAIN5)
                 + f" | {e['webqsp_s1eval']['ALL']} ({e['webqsp_s1eval']['rows']}) |")
    fams = list(rec["by_family"])
    L += ["", "## The cheapest point of each family (cost; - when none meets)", "",
          "| level | " + " | ".join(fams) + " |", "|---|" + "---:|" * len(fams)]
    for i, t in enumerate(rec["levels"]):
        L.append(f"| {t} | " + " | ".join("-" if rec["by_family"][f][i] is None else str(rec["by_family"][f][i]["cost"])
                                          for f in fams) + " |")
    L += ["", "## Per dataset: the fewest mean rows reaching each level (an upper bound only)", "",
          "| dataset | ceiling | " + " | ".join(str(t) for t in rec["levels"]) + " |",
          "|---|---:|" + "---:|" * len(rec["levels"])]
    for d, e in rec["per_dataset"].items():
        L.append(f"| {d} | {e['ceiling']} | " + " | ".join(
            "-" if e["fewest_rows"][str(t)] is None else str(e["fewest_rows"][str(t)]["rows"]) for t in rec["levels"])
            + " |")
    L += ["", "## The perfect stop's rows per dataset (the room; reads golds)", "",
          "| order | " + " | ".join(TRAIN5) + " | cost |", "|---|" + "---:|" * (len(TRAIN5) + 1)]
    for o in DEPTH_ORDERS:
        L.append(f"| {o} | " + " | ".join(str(rm[o]["rows"][d]) for d in TRAIN5) + f" | {rm[o]['cost']} |")
    L += ["", "## The stats", "", "| dataset | B | V | kt_cos | kt | entity q | route q | non-finite cosines | ALL B | "
          "ALL kt_cos | ALL listed |", "|---|" + "---:|" * 10]
    for d, s in rec["stats"].items():
        s = s["summary"]
        L.append(f"| {d} | {s['B']} | {s['V']} | {s['kt_cos']} | {s['kt']} | {s['entity_questions']} | "
                 f"{s['route_questions']} | {s['nonfinite_cosines']} | {s['ALL_B']} | {s['ALL_kt_cos']} | "
                 f"{s['ALL_listed']} |")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _toy_sources(rng, n, B, k):
    """k sources over n nodes, each (nodes, masses by mass descending then node, summing to 1)."""
    out = []
    for _ in range(k):
        if rng.random() < 0.15:
            out.append((EMPTY, np.empty(0)))
            continue
        nodes = rng.choice(n, int(rng.integers(1, min(n, 60))), replace=False).astype(np.int64)
        m = rng.random(nodes.size) ** 3
        m[rng.random(nodes.size) < 0.2] = m[0]                      # ties in mass
        o = np.lexsort((nodes, -m))
        out.append((nodes[o], m[o] / m.sum()))
    return out


def _toy_question(rng, n=400):
    """A toy question's node values, B, kt_cos and kt, golds and its U2f arrays (as stats_stage derives them)."""
    B = np.unique(rng.choice(n, int(rng.integers(0, 12)), replace=False)).astype(np.int64)
    rest = np.setdiff1d(np.arange(n), B)
    U0 = np.unique(rest[rng.random(rest.size) < rng.uniform(0.05, 0.6)]).astype(np.int64)
    c0 = rng.uniform(-0.3, 0.9, U0.size).astype(np.float32)
    if U0.size > 3:
        c0[1] = c0[2]                                               # ties in cosine
        if rng.random() < 0.3:
            c0[0] = np.nan
    ktc = rng.permutation(U0)
    sources = _toy_sources(rng, n, B, int(rng.choice([1, 2, 5])))
    V, r, e, c, bad = node_values(B, ktc, U0, c0, sources)
    cB = rng.uniform(-0.2, 0.95, B.size).astype(np.float32)
    thr = thresholds(cB)
    keep = rows_table(r, e, c, thr)
    kt = rng.permutation(V)[:int(rng.integers(0, V.size + 1))] if V.size else EMPTY
    gj = np.unique(rng.choice(n, int(rng.integers(0, 6)), replace=False)).astype(np.int64)
    in_b = np.isin(gj, B)
    rg, eg, cg = gold_values(gj, V, r, e, c)
    q = {"B": B, "V": V, "r": r, "e": e, "c": c, "ktc": ktc, "kt": kt, "thr": thr, "gj": gj, "bad": bad,
         "sources": sources, "U0": U0, "c0": c0}
    arr = {"bs": B.size, "len_ktc": ktc.size, "len_kt": kt.size, "n_v": V.size, "entity": False, "route": False,
           "nonfinite": bad, "thr": thr, "keep": keep, "pos_ktc": U2E.places(gj, in_b, ktc),
           "pos_kt": U2E.places(gj, in_b, kt), "e": eg, "c": cg.astype(np.float32)}
    return q, arr


def _toy_arrays(rng, nq):
    qs, pq, pg = [], {k: [] for k in PER_QUESTION}, {k: [] for k in PER_GOLD}
    for _ in range(nq):
        q, a = _toy_question(rng)
        qs.append(q)
        for k in PER_QUESTION:
            pq[k].append(a[k])
        for k in PER_GOLD:
            pg[k].append(a[k])
    ptr = np.concatenate([[0], np.cumsum([q["gj"].size for q in qs])]).astype(np.int64)
    return qs, arrays_of(pq, pg, ptr)


def _direct_pool(q, spec):
    """A rule's pool for one toy question, built from its nodes (section 4, by sets)."""
    B, V, r, e, c, thr = q["B"], q["V"], q["r"], q["e"], q["c"], q["thr"]
    ia, ib, it = spec[3] if spec[0] == "DEPTH" else spec[1:]
    kept = {int(v) for v, rv, ev, cv in zip(V, r, e, c)
            if rv <= K0[ia] or ev < TAUS_F[ib] or float(cv) >= float(thr[it])}
    if spec[0] == "SET":
        return set(B.tolist()) | kept
    order = q["ktc"] if spec[1] == "kt_cos" else q["kt"]
    k = int(np.rint(spec[2] * len(kept)))
    return set(B.tolist()) | set(order[:k].tolist())


def _selftest():
    import p1d_free_pool as P1D
    import p1e_anchor_kind as K
    rng = np.random.default_rng(0)
    cands = candidates()
    assert len(cands) == 21070 and len({c[0] for c in cands}) == len(cands)
    assert [c[1][0] for c in cands[:N_SET]] == ["SET"] * N_SET and cands[N_SET][0].startswith("depth:kt_cos:x0.5:")
    assert cands[0][0] == "set:k0:t0:none" and cands[1][0] == "set:k0:t0:b1d0"
    assert KTC == 6 and KT == 0 and NAMES[KTC] == "kt_cos"
    # 1. the entry levels against a per-node loop; tau's nucleus is the fewest nodes whose mass reaches tau
    for trial in range(80):
        n = int(rng.integers(5, 300))
        B = np.unique(rng.choice(n, int(rng.integers(0, 6)), replace=False)).astype(np.int64)
        srcs = _toy_sources(rng, n, B, int(rng.integers(1, 6)))
        assert all(masses_ok(*s) for s in srcs)
        V = np.setdiff1d(np.unique(np.concatenate([s[0] for s in srcs] + [rng.choice(n, 5)])), B)
        e = entry_levels(V, srcs)
        for k, v in enumerate(V):
            assert e[k] == gold_entry(int(v), srcs), (trial, v)
        for nodes, mass in srcs:
            for tau in (0.3, 0.5, 0.9, 1.0):
                if not nodes.size:
                    continue
                cum = np.cumsum(mass)
                kk = int(np.searchsorted(cum >= tau, True)) + 1 if (cum >= tau).any() else nodes.size
                nuc = set(nodes[:kk].tolist())
                ex = np.concatenate([[0.0], cum[:-1]])
                assert {int(v) for v, x in zip(nodes, ex) if min(x, E_CAP) < tau} == nuc or tau == 1.0, (trial, tau)
                if tau == 1.0:
                    assert {int(v) for v, x in zip(nodes, ex) if min(x, E_CAP) < tau} == set(nodes.tolist())
    assert not masses_ok(np.asarray([1, 2]), np.asarray([0.4, 0.6])) and not masses_ok(np.asarray([1]), np.asarray([0.9]))
    assert masses_ok(EMPTY, np.empty(0)) and not masses_ok(np.asarray([1, 1]), np.asarray([0.5, 0.5]))
    # 2. the rows table against a direct count; 3. the thresholds against a direct quantile
    for trial in range(60):
        q, a = _toy_question(rng)
        V, r, e, c, thr = q["V"], q["r"], q["e"], q["c"], q["thr"]
        for ia, k0 in enumerate(K0):
            for ib, tau in enumerate(TAUS_F):
                for it in range(len(BANDS)):
                    want = sum(1 for rv, ev, cv in zip(r, e, c) if rv <= k0 or ev < tau or float(cv) >= float(thr[it]))
                    assert a["keep"][ia, ib, it] == want, (trial, ia, ib, it)
        assert np.array_equal(a["keep"][:, 0, 0], np.minimum(np.asarray(K0), q["ktc"].size))
        assert (c[np.isin(V, q["U0"])] >= -1).sum() == int(np.isfinite(q["c0"]).sum()) and \
            q["bad"] == int((~np.isfinite(q["c0"])).sum())
        cB = rng.uniform(-1, 1, int(rng.integers(1, 30)))
        th = thresholds(cB)
        s = np.sort(cB)
        assert th[0] == np.inf and th.size == 43
        for ib_, beta in enumerate(BETAS):
            hpos = (s.size - 1) * beta
            lo_ = int(math.floor(hpos))
            want = s[lo_] + (hpos - lo_) * (s[min(lo_ + 1, s.size - 1)] - s[lo_])
            for idl, dl in enumerate(DELTAS):
                assert abs(th[1 + ib_ * len(DELTAS) + idl] - (want - dl)) < 1e-12, (beta, dl)
        assert th[1] == s[-1] and th[1 + 6 * len(DELTAS)] == s[0]
    assert (thresholds(np.empty(0)) == np.inf).all() and (thresholds(np.asarray([np.nan])) == np.inf).all()
    # 4. SET and DEPTH against direct pools; 5. the columnwise table against P1g's summarize
    qs, arr = _toy_arrays(rng, 50)
    comp, rows = table(arr)
    nq = len(qs)
    ALL_, ROWS_ = rounded(comp, rows, nq)
    pick = [0, 1, 42, N_SET - 1, N_SET, len(cands) - 1] + rng.choice(len(cands), 120, replace=False).tolist()
    for i in pick:
        nm, sp = cands[i]
        allc, rws = 0, []
        for q in qs:
            pool = _direct_pool(q, sp)
            rws.append(len(pool))
            allc += int(q["gj"].size > 0 and all(int(g) in pool for g in q["gj"]))
        s = summary(arr, sp, nm)
        assert s["ALL"] == round(allc / nq, 4) == ALL_[i], (nm, s, allc)
        assert s["rows"] == round(float(np.mean(rws)), 1) == ROWS_[i], (nm, s, np.mean(rws))
    # 6. the frontier, the pick and the verdict on toy tables (U2e's frontier, section 5's verdict)
    A = np.asarray([[0.95, 0.96], [0.97, 0.97], [0.96, 0.99], [0.99, 0.99], [0.97, 0.97]])
    costs = np.asarray([100.0, 150.0, 140.0, 400.0, 150.0])
    pts = U2E.frontier(A, costs, A.max(axis=0))
    assert [k for _t, _g, k in pts] == [0, 2, 2, 1, 1, 3, 3, 3, 3, 3, 3]
    cr_ = np.round(costs, 1)
    assert decide(2, cr_, 160.0) == "FEWER_ROWS" and decide(2, cr_, 150.0) == "SAME_ROWS"
    assert decide(2, cr_, 155.6) == "FEWER_ROWS" and decide(2, cr_, 155.5) == "SAME_ROWS"
    assert decide(2, cr_, 139.9) == "NO_CHANGE" and decide(None, cr_, 1e9) == "NO_CHANGE"
    assert decide(2, cr_, 140.0) == "SAME_ROWS"
    assert target_m(1000, 0.955) == 955 and target_m(1497, 0.955) == 1430 and round(1430 / 1497, 4) >= 0.955
    assert round(1429 / 1497, 4) < 0.955
    # the perfect stop on a toy: every question kept to its deepest gold, the cheapest first
    toy = {"ptr": np.asarray([0, 2, 3, 3, 5, 6]), "bs": np.asarray([2, 2, 2, 2, 2])}
    po = np.asarray([3, 0, 7, 1, -1, 0])                 # deepest golds 3, 7, none, unreachable, 0 (all in B)
    assert perfect_stop(toy, po, 0.4) == round((10 + 0 + 3) / 5, 1)
    assert perfect_stop(toy, po, 0.6) == round((10 + 0 + 3 + 7) / 5, 1) and perfect_stop(toy, po, 0.8) is None
    # 7. the copied kernel against P1e's on toy graphs: the same orders, masses a distribution; every weight 1 gives
    #    P1b's walk's masses
    n = 30
    edges = {"structural": [(i, i + 1) for i in range(12)] + [(0, 20), (20, 21), (0, 25), (0, 26)],
             "ner": [(1, 13), (13, 14), (5, 15), (2, 22)],
             "knn": [(15, 16), (3, 17), (17, 18), (25, 26), (22, 23)]}
    Bm = K.B
    stores = {f: Bm._Store(n, ed) for f, ed in edges.items()}
    unions = {u: K.S4B.RegimeUnion(stores, list(fams), n) for u, fams in K.UNIONS}
    rs = [Bm.uniform([0]), Bm.uniform([3, 17]), (np.asarray([0, 5, 17]), np.asarray([0.5, 0.3, 0.2])), Bm.EMPTY]
    dense = np.stack([rng.permutation(n - 1)[:12] for _ in range(4)]).astype(np.int64)
    splade = np.stack([rng.permutation(n - 1)[:12] for _ in range(4)]).astype(np.int64)
    from types import SimpleNamespace
    tq = SimpleNamespace(seeds=[np.union1d(dense[q_, :5], splade[q_, :5]) for q_ in range(4)], dense_ids=dense,
                         splade_ids=splade)
    stz = [K.steer(tq, j, 60) for j in range(4)]
    for u, _f in K.UNIONS:
        for st in (stz, K.unit_steers(None, range(4))):
            got = twalk_mass(unions[u], rs, st, 2)
            want = K.twalk(unions[u], rs, st, 2)
            assert all(np.array_equal(g_[0], w_) for g_, w_ in zip(got, want)) and got[3][0].size == 0, u
            assert all(masses_ok(*g_) for g_ in got)
        plain = twalk_mass(unions[u], rs, K.unit_steers(None, range(4)), 2)
        wm = Bm.walk_mass(unions[u], rs, 2)
        assert all(np.array_equal(a_[0], b_[0]) and np.allclose(a_[1], b_[1], rtol=1e-9, atol=1e-15)
                   for a_, b_ in zip(plain, wm)), u
    linked = [dense[0, :2].copy(), np.empty(0, np.int64), np.asarray([dense[2, 0], 29]), np.asarray([29])]
    srcs = P1D.sources_batch(tq, range(4), linked, unions, 60, 2)
    sw = K.steered_batch(tq, list(range(4)), linked, unions, srcs, 60, 2)
    km = kt_mass_batch(tq, list(range(4)), linked, unions, srcs, 60, 2)
    assert all(np.array_equal(km[u][i][0], sw[i][u]) for u in ("KTst", "KTall") for i in range(4))
    src = srcs[0][0]
    ktd = {u: km[u][0] for u in ("KTst", "KTall")}
    assert [s[0].tolist() for s in kind_t_sources(src, ktd, True, "entity")] == [src[2][0].tolist(), src[0][0].tolist()]
    assert [s[0].tolist() for s in kind_t_sources(src, ktd, False, "entity")] == [src[0][0].tolist()]
    assert [s[0].tolist() for s in kind_t_sources(src, ktd, True, "passage")] == \
        [src[0][0].tolist(), src[1][0].tolist(), ktd["KTst"][0].tolist(), ktd["KTall"][0].tolist(), src[4][0].tolist()]
    assert all(masses_ok(*s) for i in range(4) for s in srcs[i][0])
    # 8. the copied orders function against U2e's, with ties, in one chunk or many
    chunk0 = U2E.CHUNK
    dim = 8
    Vv = rng.standard_normal((500, dim)).astype(np.float32)
    Vv[7] = Vv[3]
    read = lambda rr: Vv[np.asarray(rr, np.int64)]  # noqa: E731
    try:
        for trial in range(30):
            U2E.CHUNK = 7 if trial % 2 else chunk0
            pool = rng.permutation(500)
            B = np.sort(pool[:int(rng.integers(0, 10))]).astype(np.int64)
            rest = pool[B.size:]
            full = rest[rng.random(rest.size) < 0.3][:int(rng.integers(0, 120))].astype(np.int64)
            n1 = np.unique(rest[rng.random(rest.size) < 0.2]).astype(np.int64)
            n2 = np.setdiff1d(np.unique(rest[rng.random(rest.size) < 0.2]), n1).astype(np.int64)
            first = rng.integers(0, 100, n2.size).astype(np.int64)
            r0 = rng.permutation(rest[rng.random(rest.size) < 0.1]).astype(np.int64)
            iout = np.unique(rest[rng.random(rest.size) < 0.1]).astype(np.int64)
            qv = D5B.unit(rng.standard_normal(dim))
            cos1 = D5B.unit(read(n1)) @ qv if n1.size else np.empty(0, np.float32)
            cr = D5B.ranks_desc(cos1, n1) if n1.size else EMPTY
            a1 = U2E.question_orders(full, n1, cr, cos1, r0, iout, n2, first, read, qv, 60, P1D.rrf)
            a2 = question_orders(full, n1, cr, cos1, r0, iout, n2, first, read, qv, 60, P1D.rrf)
            assert set(a1[0]) == set(a2[0]) and all(np.array_equal(a1[0][k], a2[0][k]) for k in a1[0])
            assert a1[1] == a2[1] and np.array_equal(a1[2], a2[2])
            assert all(np.array_equal(a1[3][m], a2[3][m]) for m in U2E.MS)
            cosv = {int(v): float(x) for v, x in zip(a2[2], a2[4])}
            o0 = a2[0]["cos"]
            assert [int(v) for v in o0] == sorted((int(v) for v in o0), key=lambda v: (-cosv[v], v))
    finally:
        U2E.CHUNK = chunk0
    # 9. the shards: any split merges back to the whole; parts out of order are refused
    ptr = arr["ptr"]
    for k in (1, 2, 3, 7, nq):
        parts = []
        for i in range(k):
            q0, q1 = U2C.shard_range(nq, i, k)
            g0, g1 = int(ptr[q0]), int(ptr[q1])
            parts.append({**{x: arr[x][q0:q1] for x in PER_QUESTION}, **{x: arr[x][g0:g1] for x in PER_GOLD},
                          "ptr": ptr[q0:q1 + 1]})
        m = merge_arrays(parts)
        assert list(m) == list(ARRAYS) and all(np.array_equal(m[x], arr[x]) for x in ARRAYS), k
    try:
        merge_arrays([parts[1], parts[0]] + parts[2:])
    except SystemExit:
        pass
    else:
        raise AssertionError("parts out of order merged")
    assert summary_of(m) == summary_of(arr)
    print("zu2f selftest: OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("stats", "merge", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", help="I/K: the I-th (from 0) of K contiguous question ranges")
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return 0
    if a.stage == "stats":
        if not a.dataset:
            raise SystemExit("stats needs DATASET")
        shard = None
        if a.shard:
            i, k = (int(x) for x in a.shard.split("/"))
            shard = (i, k)
        return stats_stage(a.dataset, a.threads, a.limit, shard)
    if a.stage == "merge":
        if not a.dataset or a.parts < 1:
            raise SystemExit("merge needs DATASET and --parts K")
        return merge_stage(a.dataset, a.parts)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
