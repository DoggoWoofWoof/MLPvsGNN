"""U2e: the pools toward the ceiling, with a second link and fused orders (docs/U2E_POOLS_TO_THE_CEILING.md). Pools
only: nothing trains, nothing is fit, and no order, budget or cut reads a gold.

    python outputs/mp_unified/zu2e.py stats DATASET [--threads 2] [--limit Q] [--shard I/K]
    python outputs/mp_unified/zu2e.py merge DATASET --parts K
    python outputs/mp_unified/zu2e.py verdict
    python outputs/mp_unified/zu2e.py --selftest

stats    One dataset's selection carve (s1sel; webqsp's s1eval, beside) on U2c's chosen graph: W2_S swapped in on the
         passage datasets as U2c swaps it, the KB on metaqa and webqsp. Per question: the sources (KIND_T's order, B's
         one-link neighbours N1, the first stage's lists R0, the frozen I, the second link N2(M)), the 14 orders and
         their lengths. Per gold: its place in each order (0 in B, p >= 1 the p-th node, -1 outside). Every question is
         checked against U2c's filed arrays on the same graph (section 7). --limit Q: the first Q questions, every
         check, nothing written and no number logged. --shard I/K: the I-th of K contiguous question ranges, into
         outputs/u2e/stats/parts/ (outputs/mp_unified/zu2e_modal.py runs them on Modal).
merge    Join a dataset's K parts into the file one run over every question writes.
verdict  The 13,860 U2e candidates (14 orders x 99 budgets x P1h's 10 taus) and U2c's 6,706, the frontier, the pick
         under the cap, the ledger. Writes outputs/u2e/verdict.json and verdict.md.
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

import zu2c as U2C  # noqa: E402  (U2c's graph swap, build-order places, filed arrays, grid, cut and ledger)

PG, F, P1H = U2C.PG, U2C.F, U2C.P1H
D5, D5B = U2C.D5, U2C.D5B
log, utc, write_json, sha_file = U2C.log, U2C.utc, U2C.write_json, U2C.sha_file
DECLARED = "docs/U2E_POOLS_TO_THE_CEILING.md"
OUT = ROOT / "outputs" / "u2e"
SIX, TRAIN5, PASSAGE = U2C.SIX, U2C.TRAIN5, U2C.PASSAGE
NMAX = F.NMAX                     # P1f's cap on every order; the second link's two guards
M_GRID = (10, 30, 100)            # the second link from N1's first M nodes by the question's cosine
MS = (0,) + M_GRID                # M = 0: no second link
CHUNK = 65536                     # rows read per cosine product outside N1
BIG = 1 << 40                     # P1d's rrf uncut
COLUMNS = (("kt", 0), ("kt_n1", 0)) + tuple((o, m) for o in ("cos", "kt_cos", "kt_cos_r0") for m in MS)
FIX_K = (25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000, 15000, 20000)
ADA_F = (1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)
ADA_K0 = (0, 25, 50, 100, 200, 300, 500, 1000)
TAUS = P1H.TAUS
LEVELS = tuple(round(0.95 + 0.005 * i, 3) for i in range(11))
MIN_CUT = P1H.MIN_CUT
CHECK_CANDS = (("f3+cut_cos_7000", 3.0, "cut_cos_7000"), ("f3+none", 3.0, "none"))
VERDICTS = ("RAISED", "FEWER_ROWS", "NO_CHANGE")
CAUSES = ("TAU_CUT", "BUDGET", "UNREACHED")
EMPTY = PG.EMPTY


def col_name(o, m):
    return o if m == 0 else f"{o}+n2_{m}"


NAMES = tuple(col_name(o, m) for o, m in COLUMNS)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── one question ─────────────────────────────────────────────────────────────


def second_link(adj, n1, cr, B, mmax=M_GRID[-1], cap=NMAX):
    """N2: the neighbours over adj's families of N1's first mmax nodes by the question's cosine (cr: N1's 1-based
    cosine ranks, D5b's), outside B and N1. A parent with more than cap links is passed over and the next takes its
    place; the parents stop at the first after which N2 holds at least cap nodes, which is kept whole.
    Returns (N2 sorted, each node's first parent's index among the parents taken, parents taken, hubs passed over
    ahead of the last parent taken)."""
    n1 = np.asarray(n1, np.int64)
    if n1.size == 0 or mmax == 0:
        return EMPTY, EMPTY, 0, 0
    order = n1[np.argsort(np.asarray(cr, np.int64), kind="stable")]
    deg = np.zeros(order.size, np.int64)
    for ip, _col in adj:
        deg += np.asarray(ip[order + 1], np.int64) - np.asarray(ip[order], np.int64)
    ok = deg <= cap
    idx = np.flatnonzero(ok)[:mmax]
    if idx.size == 0:
        return EMPTY, EMPTY, 0, 0
    ids, par, _f = D5.neighbours(adj, order[idx])
    keep = ~np.isin(ids, B) & ~np.isin(ids, n1)
    ids, par = ids[keep], par[keep]
    if ids.size == 0:
        return EMPTY, EMPTY, int(idx.size), int((~ok[:idx[-1] + 1]).sum())
    n2, inv = np.unique(ids, return_inverse=True)
    first = np.full(n2.size, idx.size, np.int64)
    np.minimum.at(first, inv, par)
    after = np.cumsum(np.bincount(first, minlength=idx.size))        # |N2| after parents 0..p
    stop = np.flatnonzero(after >= cap)
    taken = int(stop[0]) + 1 if stop.size else int(idx.size)
    kept = first < taken
    return n2[kept], first[kept], taken, int((~ok[:idx[taken - 1] + 1]).sum())


def question_orders(full, n1, cr, cos1, r0, iout, n2, first, read, q, c, rrf):
    """The 14 orders of COLUMNS for one question, each node once, ties by id. full: KIND_T's order outside B; n1, cr,
    cos1: N1 (sorted), its cosine ranks and values (D5b's); r0: R0 outside B in its order; iout: the frozen I outside
    B; n2, first: second_link's. Returns ({name: order}, {M: |U_M|}, U, {M: U_M's mask over U})."""
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
            [D5B.unit(read(rest[a:a + CHUNK])) @ q for a in range(0, rest.size, CHUNK)])
    base = np.isin(U, full) | np.isin(U, n1) | np.isin(U, r0) | np.isin(U, iout)
    fu = np.full(U.size, BIG, np.int64)
    if np.asarray(n2).size:
        k = np.minimum(np.searchsorted(n2, U), n2.size - 1)
        hit = n2[k] == U
        fu[hit] = first[k[hit]]
    n1c = n1[np.argsort(np.asarray(cr, np.int64), kind="stable")]
    out = {"kt": full, "kt_n1": rrf([full, n1c], c, nmax=BIG)}
    sizes, mem = {}, {}
    for m in MS:
        mk = base | (fu < m)
        ids = U[mk]
        cos_m = ids[np.lexsort((ids, -cu[mk].astype(np.float64)))]
        out[col_name("cos", m)] = cos_m
        out[col_name("kt_cos", m)] = rrf([full, cos_m], c, nmax=BIG)
        out[col_name("kt_cos_r0", m)] = rrf([full, cos_m, r0], c, nmax=BIG)
        sizes[m], mem[m] = int(mk.sum()), mk
    return out, sizes, U, mem


def orders_exact(ords, B, full, n1, U, mem):
    """Every order holds each node once, none of B, and exactly its sources' nodes."""
    if np.isin(U, B).any():
        return False
    want = {"kt": np.unique(full), "kt_n1": np.union1d(full, n1)}
    for m in MS:
        for o in ("cos", "kt_cos", "kt_cos_r0"):
            want[col_name(o, m)] = U[mem[m]]
    for nm in NAMES:
        o = ords[nm]
        if o.size != want[nm].size or not np.array_equal(np.sort(o), want[nm]):
            return False
    return True


def places(golds, inB, order):
    """Each gold's place in an order: 0 in B, p >= 1 the order's p-th node, -1 outside."""
    i = PG.positions_in(golds, order)
    p = np.where(i < order.size, i + 1, -1)
    p[inB] = 0
    return p.astype(np.int32)


# ── stats (Modal, in shards; or the laptop) ──────────────────────────────────

PER_QUESTION = ("bs", "fill", "inc", "lens", "n_n1", "n_r0", "n_i", "n_n2", "n_u", "taken", "hubs")
PER_GOLD = ("pos", "n2_first", "listed", "one_link", "kt_pos", "in_I")
ARRAYS = PER_QUESTION + PER_GOLD + ("ptr",)
HEAD_KEYS = ("declared_in", "stage", "dataset", "carve", "graph_detail", "script_sha256", "zu2c_sha256",
             "u2c_stats_sha256", "p1g_script_sha256", "zp1h_sha256", "p1e_script_sha256", "freeze_RECORD_SHA256",
             "threads", "columns", "m_grid", "nmax", "chunk")


def load_u2c(name):
    x = U2C.load_stats(name, "chosen")
    if x is None:
        raise SystemExit(f"{name}: U2c's filed arrays are not under outputs/u2c/stats")
    return x


def stats_stage(name, threads=2, limit=0, shard=None):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if D.SOURCES[4] != "R0":
        raise SystemExit("P1d's fifth source is not R0")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    ref, rref = load_u2c(name)
    names805 = [c[0] for c in PG.candidates()]
    cc = [names805.index(nm) for nm, _f, _e in CHECK_CANDS]
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
                    raise SystemExit(f"{ds_.name}: U2e swaps {name}'s graph only")
                return swap
            return load0(ds_, family, cache_dir)

        S4.m3b_pools.load_or_build_store = load
    if desc != rref["graph_detail"]:
        raise SystemExit(f"{name}: the graph is not the one U2c's filed run read")
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

    fams = {nm: f for nm, f, _p in PG.extras()}
    cut_o, cut_k = {nm: p for nm, _f, p in PG.extras()}["cut_cos_7000"]
    if cut_o != "cos" or fams["cut_cos_7000"] != "CUT":
        raise SystemExit("P1g's cut_cos_7000 is not a cut of N1 by the question's cosine")
    cache = {}
    stores_all, unions = G.open_unions(cache, op, name, union)
    adj = [(np.ascontiguousarray(stores_all[f].indptr, np.int64), np.ascontiguousarray(stores_all[f].col))
           for f in S4.FAMILIES]
    n_st = int(unions["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    st = F.load_stats(name, carve)
    nq = min(limit, len(bs)) if limit else len(bs)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(st["ptr"][:nq + 1], ptr) or not np.array_equal(ref["ptr"][:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: P1f's or U2c's golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    pq = {k: [] for k in PER_QUESTION}
    pg = {k: [] for k in PER_GOLD}
    t = time.time()
    for lo in range(q0, q1, S4.BATCH):
        js = list(range(lo, min(q1, lo + S4.BATCH)))
        srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
        steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
        for i, j in enumerate(js):
            src, _nn, (L, Ls) = srcs[i]
            route = D.routes(L, Ls)
            knd = K.kind(Ls, prep.seeds[j], nonly)
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
            if not (B.size == int(ref["bs"][j]) and fill.size == int(ref["fill"][j])
                    and inc[j].size == int(ref["inc"][j]) and np.array_equal(ktp, ref["kt_pos"][g0:g1])
                    and np.array_equal(listed, ref["listed"][g0:g1]) and np.array_equal(one, ref["one_link"][g0:g1])
                    and np.array_equal(in_i, ref["in_I"][g0:g1])):
                raise SystemExit(f"{name}/{carve} question {j}: B, today's fill, I or a gold's flags are not U2c's")
            cut = cr <= cut_k
            sets = {"none": EMPTY, "cut_cos_7000": n1[cut]}
            ranks = {"cut_cos_7000": np.asarray(cr[cut], np.int64)}
            r2, _h2, p2 = U2C.place_question(B, full, fill.size, gj, sets, ranks, CHECK_CANDS, fams)
            if not (np.array_equal(r2, ref["rows"][j, cc]) and np.array_equal(p2, ref["pos"][g0:g1][:, cc])):
                raise SystemExit(f"{name}/{carve} question {j}: N1's cosine cut does not give U2c's rows and places")
            r0 = F.outside(src[4][0], B)
            iout = F.outside(inc[j], B)
            n2, first, taken, hubs = second_link(adj, n1, cr, B)
            ords, sizes, U, mem = question_orders(full, n1, cr, cos1, r0, iout, n2, first, nodes.read, q, constant,
                                                  D.rrf)
            if not orders_exact(ords, B, full, n1, U, mem):
                raise SystemExit(f"{name}/{carve} question {j}: an order is not exactly its sources' nodes")
            in_b = np.isin(gj, B)
            pg["pos"].append(np.stack([places(gj, in_b, ords[nm]) for nm in NAMES], axis=1))
            nf = np.full(gj.size, -1, np.int32)
            if n2.size:
                k = np.minimum(np.searchsorted(n2, gj), n2.size - 1)
                hit = n2[k] == gj
                nf[hit] = first[k[hit]]
            pg["n2_first"].append(nf)
            pg["listed"].append(listed)
            pg["one_link"].append(one)
            pg["kt_pos"].append(ktp)
            pg["in_I"].append(in_i)
            pq["bs"].append(B.size)
            pq["fill"].append(fill.size)
            pq["inc"].append(inc[j].size)
            pq["lens"].append([ords[nm].size for nm in NAMES])
            pq["n_n1"].append(n1.size)
            pq["n_r0"].append(r0.size)
            pq["n_i"].append(iout.size)
            pq["n_n2"].append([int((first < m).sum()) for m in M_GRID])
            pq["n_u"].append([sizes[m] for m in MS])
            pq["taken"].append(taken)
            pq["hubs"].append(hubs)
        if (lo // S4.BATCH) % 10 == 9:
            log(f"{name}/{carve}: {js[-1] + 1}/{nq} ({time.time() - t:.0f}s)")
    arr = arrays_of(pq, pg, ptr[q0:q1 + 1])
    head = {"declared_in": DECLARED, "stage": "U2e", "dataset": name, "carve": carve, "graph_detail": desc,
            "script_sha256": script_sha256(), "zu2c_sha256": U2C.script_sha256(),
            "u2c_stats_sha256": rref["file_sha256"], "p1g_script_sha256": PG.script_sha256(),
            "zp1h_sha256": P1H.script_sha256(), "p1e_script_sha256": K.script_sha256(),
            "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "threads": threads, "columns": list(NAMES),
            "m_grid": list(M_GRID), "nmax": NMAX, "chunk": CHUNK}
    checks = {"graph_equal_u2c": True, "seeds_equal_g0": True, "b_fill_i_gold_flags_equal_u2c": True,
              "u2c_rows_places_reproduced": True, "orders_exact": True}
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
    nc, nm = len(NAMES), len(M_GRID)
    a = {"bs": np.asarray(pq["bs"], np.int64), "fill": np.asarray(pq["fill"], np.int64),
         "inc": np.asarray(pq["inc"], np.int64),
         "lens": np.asarray(pq["lens"], np.int32).reshape(-1, nc),
         "n_n1": np.asarray(pq["n_n1"], np.int64), "n_r0": np.asarray(pq["n_r0"], np.int64),
         "n_i": np.asarray(pq["n_i"], np.int64), "n_n2": np.asarray(pq["n_n2"], np.int64).reshape(-1, nm),
         "n_u": np.asarray(pq["n_u"], np.int64).reshape(-1, len(MS)), "taken": np.asarray(pq["taken"], np.int64),
         "hubs": np.asarray(pq["hubs"], np.int64),
         "pos": np.concatenate(pg["pos"]) if pg["pos"] else np.zeros((0, nc), np.int32),
         "n2_first": np.concatenate(pg["n2_first"]) if pg["n2_first"] else np.zeros(0, np.int32),
         "listed": np.concatenate(pg["listed"]) if pg["listed"] else np.zeros(0, bool),
         "one_link": np.concatenate(pg["one_link"]) if pg["one_link"] else np.zeros(0, bool),
         "kt_pos": np.concatenate(pg["kt_pos"]) if pg["kt_pos"] else np.zeros(0, np.int64),
         "in_I": np.concatenate(pg["in_I"]) if pg["in_I"] else np.zeros(0, bool),
         "ptr": np.asarray(ptr, np.int64)}
    return {k: a[k] for k in ARRAYS}


def summary_of(a):
    """Per order its whole reach (ALL, recall at full length, tau = 1) and mean length; the sources' mean sizes."""
    ptr = a["ptr"]
    ng = np.diff(ptr)
    qi = np.repeat(np.arange(ng.size), ng)
    out = {"orders": {}}
    for ci, nm in enumerate(NAMES):
        held = a["pos"][:, ci] >= 0
        hits = np.bincount(qi, weights=held.astype(np.float64), minlength=ng.size)
        has = ng > 0
        out["orders"][nm] = {"ALL": round(float((has & (hits == ng)).mean()), 4),
                             "recall": round(float(np.where(has, hits / np.maximum(ng, 1), 0.0).mean()), 4),
                             "length": round(float(a["lens"][:, ci].mean()), 1)}
    out["sizes"] = {"B": round(float(a["bs"].mean()), 1), "fill": round(float(a["fill"].mean()), 1),
                    "N1": round(float(a["n_n1"].mean()), 1), "R0": round(float(a["n_r0"].mean()), 1),
                    "I": round(float(a["n_i"].mean()), 1),
                    **{f"N2_{m}": round(float(a["n_n2"][:, i].mean()), 1) for i, m in enumerate(M_GRID)},
                    **{f"U_{m}": round(float(a["n_u"][:, i].mean()), 1) for i, m in enumerate(MS)},
                    "parents_taken": round(float(a["taken"].mean()), 2), "hubs": round(float(a["hubs"].mean()), 3),
                    "questions_with_a_hub": int((a["hubs"] > 0).sum())}
    return out


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
    st = F.load_stats(name, carve)
    if arr["bs"].size != nq or not np.array_equal(st["ptr"][:nq + 1], arr["ptr"]):
        raise SystemExit(f"{name}/{carve}: the merged parts do not cover P1f's questions and golds")
    return write_stats(name, carve, arr, head, recs[0]["checks"], round(sum(r["seconds"] for r in recs), 1),
                       {"shards": k, "part_sha256": [r["file_sha256"] for r in recs],
                        "seconds_longest_part": max(r["seconds"] for r in recs)})


# ── the verdict (the laptop) ─────────────────────────────────────────────────


def load_u2e(name):
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
    if rec["columns"] != list(NAMES) or summary_of(a) != rec["summary"]:
        raise SystemExit(f"{p}: not this script's columns, or the arrays do not give the record's summary")
    return a, rec


def budgets():
    """[(name, kind, params)] in section 5's order."""
    return [(f"fix_{k}", "FIX", (k,)) for k in FIX_K] + \
        [(f"ada_f{f:g}_k{k0}", "ADA", (f, k0)) for f in ADA_F for k0 in ADA_K0]


def budget_k(kind, prm, fill):
    fill = np.asarray(fill, np.int64)
    if kind == "FIX":
        return np.full(fill.size, int(prm[0]), np.int64)
    f, k0 = prm
    return np.maximum(int(k0), np.rint(float(f) * fill).astype(np.int64))


def u2e_candidates():
    """[(name, column index, budget name, kind, params, tau)]: order, then budget, then tau from 1 down."""
    out = []
    for ci, nm in enumerate(NAMES):
        for bn, kind, prm in budgets():
            for t in sorted(TAUS, reverse=True):
                out.append((f"{nm}:{bn}@{P1H.key(t)}", ci, bn, kind, prm, t))
    return out


def evaluate(a, ci, kq, tau, qi=None):
    """(rows, held) of one U2e candidate on one dataset's arrays: B, the order's first min(k_q, length) nodes, then
    P1h's cut at tau."""
    b = np.asarray(a["bs"], np.int64)
    k = np.minimum(np.asarray(a["lens"][:, ci], np.int64), kq)
    keep = P1H.keep_counts(b, k, tau)
    if qi is None:
        qi = np.repeat(np.arange(b.size), np.diff(a["ptr"]))
    p = a["pos"][:, ci]
    return b + keep, (p == 0) | ((p >= 1) & (p <= keep[qi]))


def summ(rows, held, ptr, nm):
    return PG.summarize({"rows": rows[:, None], "held": held[:, None], "ptr": ptr}, [(nm, None, None)])[nm]


def u2e_table(arrays, cands):
    """{candidate: {dataset: P1g's summary}} over U2e's candidates."""
    out = {c[0]: {} for c in cands}
    for d, a in arrays.items():
        qi = np.repeat(np.arange(a["bs"].size), np.diff(a["ptr"]))
        kq = {bn: budget_k(kind, prm, a["fill"]) for bn, kind, prm in budgets()}
        for nm, ci, bn, _kind, _prm, t in cands:
            rows, held = evaluate(a, ci, kq[bn], t, qi)
            out[nm][d] = summ(rows, held, a["ptr"], nm)
    return out


def frontier(A, costs, ceil, levels=LEVELS, tol=PG.TOL):
    """A: [candidates x datasets] ALL; costs: [candidates]; ceil: [datasets]. Per level t: (t, the targets
    min(t, ceil - tol), the index of the cheapest candidate meeting every target or None); ties go to the earlier."""
    out = []
    for t in levels:
        tgt = np.round(np.minimum(t, np.asarray(ceil, np.float64) - tol), 4)
        ok = (A >= tgt[None, :] - 1e-9).all(axis=1)
        k = None
        if ok.any():
            i = np.flatnonzero(ok)
            k = int(i[np.argmin(costs[i])])
        out.append((t, tgt, k))
    return out


def decide(points, cost_r, cap, incumbent):
    """Section 6: (verdict, index of what goes to U2d, its level)."""
    within = [(t, k) for t, _tgt, k in points if k is not None and cost_r[k] <= cap + 1e-9]
    if not within:
        raise SystemExit("no level's point is within the cap; U2c's chosen should meet 0.95 within it")
    t_star, k_star = max(within, key=lambda x: x[0])
    if t_star > LEVELS[0] + 1e-9:
        return "RAISED", k_star, t_star
    if cost_r[k_star] <= (1.0 - MIN_CUT) * cap + 1e-9:
        return "FEWER_ROWS", k_star, t_star
    return "NO_CHANGE", incumbent, LEVELS[0]


def ledger_of(held, held_tau1, reach, ptr, hyper=None):
    """The pick's misses by cause: TAU_CUT (its rule at tau = 1 holds it), BUDGET (some source or U2c candidate holds
    it), UNREACHED (none does); the unreached that today's graph (the hyperlinks) reaches."""
    ng = np.diff(ptr)
    qi = np.repeat(np.arange(ng.size), ng)
    miss = ~held
    cause = np.where(held_tau1, "TAU_CUT", np.where(reach, "BUDGET", "UNREACHED"))
    out = {"golds": int(held.size), "missed": int(miss.sum()), "questions": int(ng.size),
           "questions_missing_a_gold": int(np.unique(qi[miss]).size),
           "by_cause": {c: int((miss & (cause == c)).sum()) for c in CAUSES},
           "questions_by_cause": {c: int(np.unique(qi[miss & (cause == c)]).size) for c in CAUSES}}
    if hyper is not None:
        out["unreached_the_hyperlinks_reach"] = int((miss & (cause == "UNREACHED") & hyper).sum())
    return out


def all_of(held, ptr):
    ng = np.diff(ptr)
    qi = np.repeat(np.arange(ng.size), ng)
    return round(float(((np.bincount(qi[~held], minlength=ng.size) == 0) & (ng > 0)).mean()), 4)


def verdict_stage():
    t0 = time.time()
    a2e, r2e, a2c, missing = {}, {}, {}, []
    for d in SIX:
        x = load_u2e(d)
        if x is None:
            missing.append(d)
        else:
            a2e[d], r2e[d] = x
            a2c[d] = load_u2c(d)[0]
    if missing:
        raise SystemExit(f"INCOMPLETE: no U2e stats for {missing}")
    gname, _csha = U2C.chosen_graph()
    v2c = json.loads((U2C.OUT / "verdict.json").read_text(encoding="utf-8"))
    if v2c.get("script_sha256") != U2C.script_sha256() or v2c["verdict"] != "TARGET_MET" or v2c["graph"] != gname:
        raise SystemExit("outputs/u2c/verdict.json: not the committed zu2c.py's TARGET_MET on the chosen graph")
    cap, inc_name = float(v2c["graphs"]["chosen"]["cost"]), "u2c:" + v2c["chosen"]
    for d in SIX:
        if not np.array_equal(a2e[d]["ptr"], a2c[d]["ptr"]) or not np.array_equal(a2e[d]["bs"], a2c[d]["bs"]):
            raise SystemExit(f"{d}: U2e's questions and golds are not U2c's")
    grid2c = U2C.grid_candidates()
    cands2e = u2e_candidates()
    tab = {f"u2c:{nm}": e for nm, e in U2C.table_of(a2c, grid2c).items()}
    log(f"U2e verdict: U2c's {len(grid2c)} candidates read ({time.time() - t0:.0f}s)")
    tab.update(u2e_table(a2e, cands2e))
    order = [f"u2c:{g[0]}" for g in grid2c] + [c[0] for c in cands2e]
    log(f"U2e verdict: {len(order)} candidates on six datasets ({time.time() - t0:.0f}s)")
    w = F.weights()
    A = np.asarray([[tab[c][d]["ALL"] for d in TRAIN5] for c in order], np.float64)
    costs = np.asarray([F.cost({d: tab[c][d]["rows"] for d in TRAIN5}, w) for c in order], np.float64)
    cost_r = np.asarray([round(float(c), 1) for c in costs])      # P1g's choose rounds as Python does
    ceil = A.max(axis=0)
    pts = frontier(A, costs, ceil)
    inc_i = order.index(inc_name)
    verdict, k_pick, level = decide(pts, cost_r, cap, inc_i)
    pick = order[k_pick]

    def point(t, tgt, k):
        if k is None:
            return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": None}
        c = order[k]
        return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": c, "cost": float(cost_r[k]),
                "within_cap": bool(cost_r[k] <= cap + 1e-9), "s1sel": {d: tab[c][d] for d in TRAIN5},
                "webqsp_s1eval": tab[c]["webqsp"]}

    fam_of = {f"u2c:{g[0]}": "u2c" for g in grid2c}
    m_of = {f"u2c:{g[0]}": None for g in grid2c}
    for nm, ci, *_r in cands2e:
        fam_of[nm], m_of[nm] = COLUMNS[ci][0], COLUMNS[ci][1]
    fams = ("u2c", "kt", "kt_n1", "cos", "kt_cos", "kt_cos_r0")
    by_family = {}
    for f in fams:
        idx = np.asarray([i for i, c in enumerate(order) if fam_of[c] == f])
        by_family[f] = [None if k is None else {"pick": order[idx[k]], "cost": float(cost_r[idx[k]])}
                        for _t, _tg, k in frontier(A[idx], costs[idx], ceil)]
    by_m = {}
    for m in MS:
        idx = np.asarray([i for i, c in enumerate(order) if m_of[c] == m])
        by_m[str(m)] = [None if k is None else {"pick": order[idx[k]], "cost": float(cost_r[idx[k]])}
                        for _t, _tg, k in frontier(A[idx], costs[idx], ceil)]
    per_dataset = {}
    for d in SIX:
        al = np.asarray([tab[c][d]["ALL"] for c in order])
        rw = np.asarray([tab[c][d]["rows"] for c in order])
        cd = float(al.max())
        rows_at = {}
        for t in LEVELS:
            tg = round(min(t, cd - PG.TOL), 4)
            i = np.flatnonzero(al >= tg - 1e-9)
            k = int(i[np.argmin(rw[i])])
            rows_at[str(t)] = {"target": tg, "rows": float(rw[k]), "ALL": float(al[k]), "candidate": order[k]}
        per_dataset[d] = {"ceiling": cd, "fewest_rows": rows_at}

    # the pick's arrays, its rule at tau = 1, every source's reach, and the ledger
    by2e = {c[0]: c for c in cands2e}
    by2c = {f"u2c:{g[0]}": g for g in grid2c}
    col805 = {nm: i for i, nm in enumerate(c[0] for c in PG.candidates())}

    def rows_held(c, d, tau1=False):
        if c in by2c:
            _nm, src, gated, tau = by2c[c]
            r, p = U2C.column(a2c[d], col805, src, gated)
            return U2C.cut_column(a2c[d], r, p, 1.0 if tau1 else tau)
        _nm, ci, _bn, kind, prm, tau = by2e[c]
        return evaluate(a2e[d], ci, budget_k(kind, prm, a2e[d]["fill"]), 1.0 if tau1 else tau)

    sources, ledger = {}, {}
    for d in SIX:
        e, ptr = a2e[d], a2e[d]["ptr"]
        pos = e["pos"]
        in_b = pos[:, 0] == 0
        u2c_any = (a2c[d]["pos"] >= 0).any(axis=1)
        reach = (pos >= 0).any(axis=1) | u2c_any
        ci = {nm: i for i, nm in enumerate(NAMES)}
        sources[d] = {"B": all_of(in_b, ptr), "KT": all_of(pos[:, ci["kt"]] >= 0, ptr),
                      "KT+N1": all_of(pos[:, ci["kt_n1"]] >= 0, ptr),
                      **{f"U_{m}": all_of(pos[:, ci[col_name('cos', m)]] >= 0, ptr) for m in MS},
                      "every U2c candidate": all_of(u2c_any, ptr), "all": all_of(reach, ptr),
                      "golds_outside_all": int((~reach).sum()),
                      "questions_outside_all": int(np.unique(np.repeat(np.arange(ptr.size - 1),
                                                                       np.diff(ptr))[~reach]).size)}
        _r, held = rows_held(pick, d)
        _r1, held1 = rows_held(pick, d, tau1=True)
        hyper = None
        if d in PASSAGE:
            at = U2C.load_stats(d, "today")[0]
            if not np.array_equal(at["ptr"], ptr):
                raise SystemExit(f"{d}: U2c's today arrays are not these questions")
            hyper = (at["pos"] >= 0).any(axis=1)
        ledger[d] = ledger_of(held, held1 | held, reach | held1 | held, ptr, hyper)
    rec = {"declared_in": DECLARED, "stage": "U2e", "script_sha256": script_sha256(), "zu2c_sha256": U2C.script_sha256(),
           "graph": gname, "verdict": verdict, "pick": pick, "level": level, "cap": cap, "u2c_chosen": inc_name,
           "pick_cost": float(cost_r[k_pick]), "u2c_chosen_cost": float(cost_r[inc_i]), "weights": w,
           "candidates": {"u2c": len(grid2c), "u2e": len(cands2e)},
           "ceilings": dict(zip(TRAIN5, ceil.tolist())),
           "pick_s1sel": {d: tab[pick][d] for d in TRAIN5}, "pick_webqsp_s1eval": tab[pick]["webqsp"],
           "u2c_chosen_s1sel": {d: tab[inc_name][d] for d in TRAIN5}, "u2c_chosen_webqsp_s1eval": tab[inc_name]["webqsp"],
           "frontier": [point(t, tg, k) for t, tg, k in pts],
           "by_family": by_family, "by_m": by_m, "levels": list(LEVELS), "per_dataset": per_dataset,
           "sources": sources, "ledger": ledger,
           "stats": {d: {"file_sha256": r2e[d]["file_sha256"], "summary": r2e[d]["summary"]} for d in SIX},
           "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "verdict.json", rec)
    (OUT / "verdict.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"U2e: {verdict}: {pick} at level {level} (cost {rec['pick_cost']}; cap {cap}) ({time.time() - t0:.0f}s)")
    return 0


def report_md(rec):
    L = [f"# U2e: {rec['verdict']}", "",
         f"Pick: `{rec['pick']}` at level {rec['level']}, cost {rec['pick_cost']} (cap {rec['cap']}, U2c's chosen "
         f"`{rec['u2c_chosen']}` {rec['u2c_chosen_cost']}). Declared in {rec['declared_in']}. Graph {rec['graph']}.", "",
         "| dataset | pick ALL | rows | rows p90 | U2c's chosen ALL | rows | ceiling (one candidate) | all sources |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in TRAIN5:
        p, c = rec["pick_s1sel"][d], rec["u2c_chosen_s1sel"][d]
        L.append(f"| {d} | {p['ALL']} | {p['rows']} | {p['rows_p90']} | {c['ALL']} | {c['rows']} | "
                 f"{rec['per_dataset'][d]['ceiling']} | {rec['sources'][d]['all']} |")
    p, c = rec["pick_webqsp_s1eval"], rec["u2c_chosen_webqsp_s1eval"]
    L += [f"| webqsp (s1eval, beside) | {p['ALL']} | {p['rows']} | {p['rows_p90']} | {c['ALL']} | {c['rows']} | "
          f"{rec['per_dataset']['webqsp']['ceiling']} | {rec['sources']['webqsp']['all']} |", "",
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
    L += ["", "## The cheapest point of each order family (cost; - when none meets)", "",
          "| level | " + " | ".join(rec["by_family"]) + " | " + " | ".join(f"M={m}" for m in rec["by_m"]) + " |",
          "|---|" + "---:|" * (len(rec["by_family"]) + len(rec["by_m"]))]
    for i, t in enumerate(rec["levels"]):
        cells = [("-" if v[i] is None else str(v[i]["cost"])) for v in rec["by_family"].values()]
        cells += [("-" if v[i] is None else str(v[i]["cost"])) for v in rec["by_m"].values()]
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "## Per dataset: the fewest mean rows reaching each level (an upper bound only)", "",
          "| dataset | ceiling | " + " | ".join(str(t) for t in rec["levels"]) + " |",
          "|---|---:|" + "---:|" * len(rec["levels"])]
    for d, e in rec["per_dataset"].items():
        L.append(f"| {d} | {e['ceiling']} | " + " | ".join(str(e["fewest_rows"][str(t)]["rows"]) for t in rec["levels"])
                 + " |")
    L += ["", "## Sources: ALL of each alone and together", "",
          "| dataset | B | KT | KT+N1 | " + " | ".join(f"U_{m}" for m in MS) + " | every U2c candidate | all | golds "
          "outside all | questions |", "|---|" + "---:|" * (8 + len(MS) - 1)]
    for d, s in rec["sources"].items():
        L.append(f"| {d} | {s['B']} | {s['KT']} | {s['KT+N1']} | " + " | ".join(str(s[f'U_{m}']) for m in MS)
                 + f" | {s['every U2c candidate']} | {s['all']} | {s['golds_outside_all']} | {s['questions_outside_all']} |")
    L += ["", "## Ledger: the pick's misses", "",
          "| dataset | golds | missed | TAU_CUT | BUDGET | UNREACHED | questions missing a gold | unreached the "
          "hyperlinks reach |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d, e in rec["ledger"].items():
        L.append(f"| {d} | {e['golds']} | {e['missed']} | {e['by_cause']['TAU_CUT']} | {e['by_cause']['BUDGET']} | "
                 f"{e['by_cause']['UNREACHED']} | {e['questions_missing_a_gold']} | "
                 f"{e.get('unreached_the_hyperlinks_reach', '-')} |")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _toy_adj(rng, n, ms):
    out = []
    for m in ms:
        s, t = rng.integers(0, n, m), rng.integers(0, n, m)
        k = np.unique(np.concatenate([s * n + t, t * n + s]).astype(np.int64))
        ip = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(k // n, minlength=n), out=ip[1:])
        out.append((ip, (k % n).astype(np.int32)))
    return out


def _direct_second_link(adj, n1, cr, B, mmax, cap):
    order = [int(v) for v in np.asarray(n1)[np.argsort(cr, kind="stable")]]
    seen = set(int(x) for x in B) | set(order)
    first, taken, hubs, pending = {}, 0, 0, 0
    for v in order:
        if taken == mmax:
            break
        if sum(int(ip[v + 1] - ip[v]) for ip, _c in adj) > cap:
            pending += 1
            continue
        for ip, col in adj:
            for u in col[ip[v]:ip[v + 1]]:
                u = int(u)
                if u not in seen and u not in first:
                    first[u] = taken
        taken += 1
        hubs += pending
        pending = 0
        if len(first) >= cap:
            break
    ks = sorted(first)
    return np.asarray(ks, np.int64), np.asarray([first[k] for k in ks], np.int64), taken, hubs


def _direct_rrf(orders, c):
    sc = {}
    for o in orders:
        for r, v in enumerate(o, 1):
            sc[int(v)] = sc.get(int(v), 0.0) + 1.0 / (c + r)
    return np.asarray(sorted(sc, key=lambda v: (-sc[v], v)), np.int64)


def _selftest():
    import p1d_free_pool as P1D
    rng = np.random.default_rng(0)
    # 1. the second link, its two guards and its nesting, against a direct build
    for trial in range(60):
        n = int(rng.integers(30, 400))
        adj = _toy_adj(rng, n, (int(rng.integers(0, 6 * n)), int(rng.integers(0, 3 * n)), int(rng.integers(0, n))))
        B = np.unique(rng.choice(n, int(rng.integers(1, 12)), replace=False)).astype(np.int64)
        n1 = D5B.n1_table(adj, B, np.ones(B.size))[0]
        cr = rng.permutation(n1.size).astype(np.int64) + 1
        for mmax, cap in ((100, NMAX), (3, 10), (5, 25), (int(rng.integers(1, 20)), int(rng.integers(5, 60)))):
            got = second_link(adj, n1, cr, B, mmax, cap)
            want = _direct_second_link(adj, n1, cr, B, mmax, cap)
            assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1]), (trial, mmax, cap)
            assert got[2:] == want[2:], (trial, mmax, cap, got[2:], want[2:])
            assert not np.isin(got[0], B).any() and not np.isin(got[0], n1).any()
            assert got[0].size == 0 or got[1].max() < got[2]
    # 2. the orders against a direct sort or RRF, with ties, in one chunk or many; 3. the places
    global CHUNK
    chunk0 = CHUNK
    dim = 8
    V = rng.standard_normal((500, dim)).astype(np.float32)
    V[7] = V[3]                                         # two nodes of equal cosine
    read = lambda r: V[np.asarray(r, np.int64)]  # noqa: E731
    for trial in range(40):
        CHUNK = 7 if trial % 2 else chunk0
        pool = rng.permutation(500)
        B = np.sort(pool[:int(rng.integers(0, 10))]).astype(np.int64)
        rest = pool[B.size:]
        full = rest[rng.random(rest.size) < 0.3][:int(rng.integers(0, 120))].astype(np.int64)
        n1 = np.unique(rest[rng.random(rest.size) < 0.2]).astype(np.int64)
        n2 = np.setdiff1d(np.unique(rest[rng.random(rest.size) < 0.2]), n1).astype(np.int64)
        first = rng.integers(0, 100, n2.size).astype(np.int64)
        r0 = rng.permutation(rest[rng.random(rest.size) < 0.1]).astype(np.int64)
        iout = np.unique(rest[rng.random(rest.size) < 0.1]).astype(np.int64)
        q = D5B.unit(rng.standard_normal(dim))
        cos1 = D5B.unit(read(n1)) @ q if n1.size else np.empty(0, np.float32)
        cr = D5B.ranks_desc(cos1, n1) if n1.size else EMPTY
        ords, sizes, U, mem = question_orders(full, n1, cr, cos1, r0, iout, n2, first, read, q, 60, P1D.rrf)
        assert orders_exact(ords, B, full, n1, U, mem)
        out1 = np.setdiff1d(U, n1)                      # the values: section 3's expressions; the orders: direct
        vals = [D5B.unit(read(out1[a:a + CHUNK])) @ q for a in range(0, out1.size, CHUNK)]
        cosv = dict(zip(out1.tolist(), np.concatenate(vals).tolist() if vals else []))
        cosv.update(zip(n1.tolist(), cos1.tolist()))
        n1c = [int(v) for v in sorted(n1, key=lambda v: (-cosv[int(v)], int(v)))]
        assert np.array_equal(ords["kt"], full)
        assert np.array_equal(ords["kt_n1"], _direct_rrf([full, n1c], 60))
        for m in MS:
            um = set(full.tolist()) | set(n1.tolist()) | set(r0.tolist()) | set(iout.tolist()) | \
                set(n2[first < m].tolist())
            cm = np.asarray(sorted(um, key=lambda v: (-cosv[v], v)), np.int64)
            assert sizes[m] == len(um) and np.array_equal(ords[col_name("cos", m)], cm), (trial, m)
            assert np.array_equal(ords[col_name("kt_cos", m)], _direct_rrf([full, cm], 60))
            assert np.array_equal(ords[col_name("kt_cos_r0", m)], _direct_rrf([full, cm, r0], 60))
        golds = np.unique(rng.choice(500, int(rng.integers(1, 8)), replace=False)).astype(np.int64)
        in_b = np.isin(golds, B)
        for nm in NAMES:
            o = list(ords[nm])
            want = [0 if g in B else (o.index(g) + 1 if g in o else -1) for g in golds]
            assert np.array_equal(places(golds, in_b, ords[nm]), want), (trial, nm)
    CHUNK = chunk0
    # 4. the budgets, the cut and the summary against a per-question count
    cands = u2e_candidates()
    assert len(cands) == len(NAMES) * 99 * len(TAUS) == 13860 and len({c[0] for c in cands}) == len(cands)
    assert len(budgets()) == 99 and len(NAMES) == 14 and LEVELS[0] == 0.95 and LEVELS[-1] == 1.0
    nq = 40
    ptr = np.concatenate([[0], np.cumsum(rng.integers(1, 4, nq))]).astype(np.int64)
    ng = int(ptr[-1])
    qi = np.repeat(np.arange(nq), np.diff(ptr))
    a = {"bs": rng.integers(0, 6, nq).astype(np.int64), "fill": rng.integers(0, 300, nq).astype(np.int64),
         "lens": rng.integers(0, 3000, (nq, len(NAMES))).astype(np.int32), "ptr": ptr}
    pos = rng.integers(-1, 3000, (ng, len(NAMES)))
    pos = np.where(pos > a["lens"][qi], -1, pos)
    pos[(pos == 0) & (a["bs"][qi] == 0)[:, None]] = -1
    a["pos"] = pos.astype(np.int32)
    pick = [cands[i] for i in rng.choice(len(cands), 60, replace=False)]
    tab = u2e_table({"x": a}, pick)
    for nm, ci, _bn, kind, prm, tau in pick:
        allc, rec_, rws = 0, 0.0, []
        for j in range(nq):
            if kind == "FIX":
                kq = prm[0]
            else:
                kq = max(prm[1], int(np.rint(prm[0] * int(a["fill"][j]))))
            b, kk = int(a["bs"][j]), min(int(a["lens"][j, ci]), kq)
            keep = int(P1H.keep_counts(np.asarray([b]), np.asarray([kk]), tau)[0])
            gp = a["pos"][ptr[j]:ptr[j + 1], ci]
            h = (gp == 0) | ((gp >= 1) & (gp <= keep))
            allc += int(h.all())
            rec_ += float(h.mean())
            rws.append(b + keep)
        assert tab[nm]["x"]["ALL"] == round(allc / nq, 4), nm
        assert tab[nm]["x"]["recall"] == round(rec_ / nq, 4), nm
        assert tab[nm]["x"]["rows"] == round(float(np.mean(rws)), 1), nm
    # 5. the frontier, the pick and the verdict on toy tables
    A = np.asarray([[0.95, 0.96], [0.97, 0.97], [0.96, 0.99], [0.99, 0.99], [0.97, 0.97]])
    costs = np.asarray([100.0, 150.0, 140.0, 400.0, 150.0])
    pts = frontier(A, costs, A.max(axis=0))
    assert [k for _t, _g, k in pts] == [0, 2, 2, 1, 1, 3, 3, 3, 3, 3, 3]     # rows 1 and 4 tie: the earlier
    assert pts[-1][2] == 3 and np.allclose(pts[-1][1], [0.985, 0.985])
    cr_ = np.round(costs, 1)
    assert decide(pts, cr_, 160.0, 0) == ("RAISED", 1, 0.97)
    assert decide(pts, cr_, 100.0, 4) == ("NO_CHANGE", 4, 0.95)
    c95 = np.asarray([95.0, 150, 140, 400, 150])
    assert decide(frontier(A, c95, A.max(axis=0)), c95, 100.0, 4) == ("NO_CHANGE", 4, 0.95)
    assert decide(frontier(A, c95, A.max(axis=0)), c95, 130.0, 4) == ("FEWER_ROWS", 0, 0.95)
    assert decide(frontier(A, np.asarray([80.0, 150, 140, 400, 150]), A.max(axis=0)), np.asarray([80.0, 150, 140, 400, 150]),
                  100.0, 4) == ("FEWER_ROWS", 0, 0.95)
    tie = frontier(np.asarray([[0.96], [0.96]]), np.asarray([5.0, 5.0]), np.asarray([0.96]))
    assert tie[0][2] == 0
    # 6. the ledger's causes add up to the misses
    held = rng.random(ng) < 0.6
    h1 = held | (rng.random(ng) < 0.5)
    reach = h1 | (rng.random(ng) < 0.5)
    led = ledger_of(held, h1, reach, ptr, reach)
    assert sum(led["by_cause"].values()) == led["missed"] == int((~held).sum())
    assert led["by_cause"]["UNREACHED"] == int((~reach).sum())
    # 7. the shards: any split merges back to the whole; parts out of order are refused
    whole = {"bs": a["bs"], "fill": a["fill"], "inc": rng.integers(0, 90, nq), "lens": a["lens"],
             "n_n1": rng.integers(0, 50, nq), "n_r0": rng.integers(0, 50, nq), "n_i": rng.integers(0, 50, nq),
             "n_n2": rng.integers(0, 50, (nq, len(M_GRID))), "n_u": rng.integers(0, 90, (nq, len(MS))),
             "taken": rng.integers(0, 100, nq), "hubs": rng.integers(0, 3, nq), "pos": a["pos"],
             "n2_first": rng.integers(-1, 100, ng).astype(np.int32), "listed": rng.random(ng) < 0.5,
             "one_link": rng.random(ng) < 0.3, "kt_pos": rng.integers(-1, 40, ng), "in_I": rng.random(ng) < 0.6,
             "ptr": ptr}
    for k in (1, 2, 3, 7, nq):
        parts = []
        for i in range(k):
            q0, q1 = U2C.shard_range(nq, i, k)
            g0, g1 = int(ptr[q0]), int(ptr[q1])
            parts.append({**{x: whole[x][q0:q1] for x in PER_QUESTION}, **{x: whole[x][g0:g1] for x in PER_GOLD},
                          "ptr": ptr[q0:q1 + 1]})
        m = merge_arrays(parts)
        assert list(m) == list(ARRAYS) and all(np.array_equal(m[x], whole[x]) for x in ARRAYS), k
    try:
        merge_arrays([parts[1], parts[0]] + parts[2:])
    except SystemExit:
        pass
    else:
        raise AssertionError("parts out of order merged")
    assert summary_of(m) == summary_of(whole)
    print("zu2e selftest: OK")


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
