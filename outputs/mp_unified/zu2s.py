"""U2s (docs/U2S_WALK_MASS_BUDGET.md): one construction for all six, with each question's budget read from its own walk
mass. Pools only: nothing trains, nothing is fit, and no construction, recipe, order, budget or cut reads a gold.

    python outputs/mp_unified/zu2s.py stats DATASET [--threads 2] [--limit Q] [--shard I/K]
    python outputs/mp_unified/zu2s.py merge DATASET --parts K
    python outputs/mp_unified/zu2s.py verdict
    python outputs/mp_unified/zu2s.py --selftest

stats    One dataset's carve (s1sel; webqsp's s1eval, beside) on the carried walk, as U2q reads it: U2p's pair weight
         at its beta on the passage graphs, (a)'s RWe at beta 0.5 on the KBs. Each question walks the five walks kind_t
         reads with (a)'s kernel; for each of U2m's 12 constructions and the own one, and each of the 3 recipes, U2e's
         14 orders are built with their lengths and every gold's place. Per base and MASS walk (A1all, KTall): m_q, the
         nodes the walk needs, outside B, to hold tau_w of its mass outside B (section 4). Checked (section 7): U2m's
         walk-free arrays question for question; the carried frontier's 8 points on the own construction with kind_t
         against their filed rows and held golds; KIND_T by the kind; every order exact; m_q's bounds. --limit Q: the
         first Q questions, every check, every served row verified, nothing written and no number logged. --shard
         I/K: the I-th of K contiguous question ranges, into outputs/u2s/stats/parts/.
merge    Join a dataset's K parts into the file one run over every question writes.
verdict  U2p's filed table reproduced at the carried frontier's 8 points; frontiers C (universal kind_t), E (universal
         rrf5t and both) and O (the own construction's kind_t) over 219 budgets (U2e's 99, then 120 MASS); the
         verdicts (c), (e) and (o) and what is carried (section 6). Writes outputs/u2s/verdict.json and verdict.md.
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

import zu2m as M  # noqa: E402  (U2m's cells, columns, rows, arrays, frontier pick, unchanged)
import zu2m_b as MB  # noqa: E402  (section 11's point names, unchanged)
import zu2m_a as A  # noqa: E402  (section 14's relational kernel and omega, unchanged)
import zu2m_a_types as AT  # noqa: E402  (section 14's filed EM types, unchanged)
import zu2p as P  # noqa: E402  (U2p's score types, pair weight, filed arrays and verdict, unchanged)
import zu2q as Q  # noqa: E402  (U2q's carried walk, unchanged)

U2E, U2C, F, P1H, PG, D5, D5B = M.U2E, M.U2C, M.F, M.P1H, M.PG, M.D5, M.D5B
log, utc, write_json, sha_file = M.log, M.utc, M.write_json, M.sha_file
DECLARED = "docs/U2S_WALK_MASS_BUDGET.md"
OUT = ROOT / "outputs" / "u2s"
SIX, TRAIN5, NAMES, BASES, GRID, RECIPES = M.SIX, M.TRAIN5, M.NAMES, M.BASES, M.GRID, M.RECIPES
EXPANSIONS, M_GRID, MS, LEVELS = M.EXPANSIONS, M.M_GRID, M.MS, M.LEVELS
R, NO, EMPTY = M.R, M.NO, M.EMPTY
TAUS_DESC = tuple(sorted(M.TAUS, reverse=True))        # P1h's cut, from 1 down (U2e's candidate order)
NT = len(TAUS_DESC)
MASS_WALKS = ("A1all", "KTall")
MASS_TAUS = (0.5, 0.8, 0.9, 0.95, 0.99)
MASS_F = (1, 2, 4, 8)
MASS_K0 = (0, 50, 300)
NW, NTW = len(MASS_WALKS), len(MASS_TAUS)
NMQ = len(BASES) * NW * NTW                            # m_q per question: base, walk, tau_w
FORMS = ("FIX", "ADA", "MASS")
VERDICTS_C, VERDICTS_E = M.VERDICTS_C, M.VERDICTS_E
VERDICTS_O = ("OWN_RAISED", "OWN_CHEAPER", "NO_GAIN")
KIND_FREE = M.KIND_FREE


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def budgets():
    """[(name, kind, params)] in section 4's order: U2e's 99, then the 120 MASS budgets (walk, tau_w, f, k0)."""
    return U2E.budgets() + [(f"mass_{w}_t{tw:g}_f{f}_k{k0}", "MASS", (wi, ti, f, k0))
                            for wi, w in enumerate(MASS_WALKS) for ti, tw in enumerate(MASS_TAUS)
                            for f in MASS_F for k0 in MASS_K0]


NB = len(budgets())


def mq_col(bi, wi, ti):
    return (bi * NW + wi) * NTW + ti


def mq_of(a, bi):
    """The m_q columns of base bi (questions x walks * tau_w)."""
    return np.asarray(a["mq"][:, mq_col(bi, 0, 0):mq_col(bi + 1, 0, 0)], np.int64)


def budget_k(kind, prm, fill, mq):
    """k_q per question: U2e's for FIX and ADA (today's fill); MASS max(k0, round(f x m_q)) on the cell's base's m_q."""
    if kind != "MASS":
        return U2E.budget_k(kind, prm, fill)
    wi, ti, f, k0 = prm
    return np.maximum(int(k0), np.rint(float(f) * np.asarray(mq[:, wi * NTW + ti], np.int64)).astype(np.int64))


def mass_counts(nodes, mass, B, taus=MASS_TAUS):
    """(m_q per tau_w, the walk's node count outside B): the smallest m whose first m masses outside B, in the kernel's
    order, sum (float64) to at least tau_w of their total; 0 when that total is 0."""
    out = ~np.isin(np.asarray(nodes, np.int64), B)
    cs = np.cumsum(np.asarray(mass, np.float64)[out])
    tot = float(cs[-1]) if cs.size else 0.0
    if tot <= 0:
        return [0] * len(taus), int(cs.size)
    return [int(np.searchsorted(cs, float(t) * tot, side="left")) + 1 for t in taus], int(cs.size)


def mq_ok(mq, nw):
    """Section 7: per base and walk, m_q non-decreasing in tau_w, at most the count outside B, 0 exactly when it is 0."""
    m = np.asarray(mq, np.int64).reshape(len(BASES) * NW, NTW)
    c = np.asarray(nw, np.int64).reshape(len(BASES) * NW)
    return bool((np.diff(m, axis=1) >= 0).all() and (m <= c[:, None]).all() and
                ((m[:, 0] == 0) == (c == 0)).all() and ((m == 0).all(axis=1) == (c == 0)).all())


def filed_points(name, cw):
    """(rows (questions x points), held (golds x points), ptr, source, (a)'s record) of the carried frontier's filed
    arrays on the carried walk: (a)'s on the KBs (and at beta = 0), U2p's at its beta on the passage graphs. Their
    pick column is checked to be zu2q.filed_rows'."""
    rows_a, held_a, ptr_a, ra = P.load_a_pick(name, cw["names"], cw["G"], cw["va"], cw["ca"])
    if cw["ci"] == 0 or name not in P.PASSAGE:
        rows, held, ptr, src = rows_a, held_a, ptr_a, "U2m (a)"
    else:
        x = P.load_p(name, cw["names"], cw["G"])
        if x is None:
            raise SystemExit(f"{name}: U2p's filed arrays are not under outputs/u2p/stats")
        ap, rp = x
        if rp["file_sha256"] != cw["verdict_p"]["stats"][name]["file_sha256"]:
            raise SystemExit(f"{name}: U2p's filed arrays are not the ones its verdict read")
        rows, held, ptr, src = ap["rows"][:, cw["ci"], :], ap["held"][:, cw["ci"], :], ap["ptr"], "U2p"
    rows, held, ptr = np.asarray(rows, np.int64), np.asarray(held, bool), np.asarray(ptr, np.int64)
    r1, h1, p1, s1, _ra = Q.filed_rows(name, cw)
    if not (np.array_equal(rows[:, cw["pi"]], r1) and np.array_equal(held[:, cw["pi"]], h1)
            and np.array_equal(ptr, p1) and src == s1):
        raise SystemExit(f"{name}: the carried frontier's filed arrays do not give U2q's pick column")
    return rows, held, ptr, src, ra


# ── stats (the laptop, or elsewhere in shards) ───────────────────────────────

PER_QUESTION = M.PER_QUESTION + ("mq", "nw")
PER_GOLD = M.PER_GOLD
ARRAYS = PER_QUESTION + PER_GOLD + ("ptr",)
HEAD_KEYS = ("declared_in", "stage", "dataset", "carve", "graph_detail", "script_sha256", "zu2m_sha256",
             "zu2m_b_sha256", "zu2m_a_sha256", "zu2m_a_types_sha256", "zu2p_sha256", "zu2q_sha256", "zu2e_sha256",
             "u2m_stats_sha256", "verdict_a_sha256", "verdict_p_sha256", "pick", "points", "G", "beta", "walk",
             "rows_from", "a_stats_sha256", "em_types_sha256", "score_types_sha256", "freeze_RECORD_SHA256",
             "threads", "alpha", "eps", "nmax", "cells", "own_cell", "cell_base", "recipes", "orders", "bases",
             "expansions", "mass_walks", "mass_taus")


def stats_stage(name, threads=2, limit=0, shard=None):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if D.SOURCES != ("A1st", "A1all", "Kst", "Kall", "R0") or F.CONSTRUCTION != "KIND_T":
        raise SystemExit("P1d's sources are not A1st, A1all, Kst, Kall and R0, or P1f's construction is not KIND_T")
    if tuple(A.WALKS) != tuple(Q.WALKS) or dict(A.WALK_UNION) != dict(Q.WALK_UNION) or \
            not set(MASS_WALKS) <= set(Q.WALKS):
        raise SystemExit("(a)'s walks are not (d)'s, or a MASS walk is not one of them")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    cw = Q.carried_walk()
    names, G_, pi, beta = cw["names"], cw["G"], cw["pi"], cw["beta"]
    if G_ != 0:
        raise SystemExit("U2s builds the pools at G = 0 (section 2)")
    x2m = M.load_u2m(name)
    if x2m is None:
        raise SystemExit(f"{name}: U2m's filed arrays are not under outputs/u2m/stats")
    a2m, r2m = x2m
    rows_f, held_f, ptr_f, src_f, ra = filed_points(name, cw)
    plans = [MB.parse(nm) for nm in names]
    if any(p[0] != "own" for p in plans) or rows_f.shape[1] != len(names):
        raise SystemExit("a carried point is not on the own construction, or the filed arrays are not its 8 points")
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    cfg_set = {s["name"]: s for s in op.cfg_h["graph_regimes"]["expansion_settings"]}
    h3s = [c["setting"] for c in op.construction.values() if c.get("setting") and c["setting"]["name"] == M.H3["name"]]
    if cfg_set.get("h1_c25") != M.H1 or cfg_set.get("h2_c25") != M.H2 or not h3s or any(s != M.H3 for s in h3s):
        raise SystemExit("U2m section 4's hop settings are not the filed ones")
    full_fams = list(op.cfg_h["graph_regimes"]["FULL"])
    if full_fams != list(S4.FAMILIES) or tuple(S4.FAMILIES) != ("structural", "ner", "knn"):
        raise SystemExit("FULL is not the three families in their order")
    cells, own_k = M.cells_of(construction)
    if cells != r2m["cells"] or own_k != r2m["own_cell"]:
        raise SystemExit(f"{name}: the cells are not U2m's")
    own_b = [bp for _bs, bp in BASES].index(construction["base_pool"])
    cell_base = [GRID[k][1] if k < len(GRID) else own_b for k in range(len(cells))]
    if cell_base[own_k] != own_b:
        raise SystemExit(f"{name}: the own cell's base is not the construction's")
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
    if prep.dense_ids.shape[1] < 1000 or prep.splade_ids.shape[1] < 1000:
        raise SystemExit(f"{name}/{carve}: retrieval lists are not 1000 deep")
    if any(not np.array_equal(np.asarray(x, np.int64), np.asarray(y, np.int64)) for x, y in zip(prep.seeds, prep0.seeds)):
        raise SystemExit(f"{name}/{carve}: the seeds on the graph are not G0's")
    base_rows = [op.m3b_compile.base_rows({"base_pool": bp}, prep0.dense_ids, prep0.splade_ids, op.m3a,
                                          op.m3b_contract, constant) for _bs, bp in BASES]
    inc_own = [np.asarray(p, np.int64) for p in prep.pools]
    linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [EMPTY] * len(prep.seeds)
    cache = {}
    stores_all, unions = G.open_unions(cache, op, name, union)
    adj = [(np.ascontiguousarray(stores_all[f].indptr, np.int64), np.ascontiguousarray(stores_all[f].col))
           for f in S4.FAMILIES]
    for u_, _fams in D.UNIONS:
        if not (np.array_equal(unions[u_].args[0], adj[0][0]) and np.array_equal(unions[u_].args[1], adj[0][1])):
            raise SystemExit(f"{name}: union {u_}'s structural family is not the store's")
    fams_full = [stores_all[f] for f in full_fams]
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
    log(f"{name}/{carve}: prepared; walk {walk_name}; cells {cells} (own {cells[own_k]}); points from {src_f} "
        f"({time.time() - t0:.0f}s)")
    n_st = int(unions["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    nq = min(limit, len(prep.seeds)) if limit else len(prep.seeds)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(a2m["ptr"][:nq + 1], ptr) or not np.array_equal(ptr_f[:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: U2m's or the filed points' golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    alpha, eps, bmax = float(E.ALPHA), float(S4.EPS), int(D.NMAX)
    bud = budgets()
    pq = {k: [] for k in PER_QUESTION}
    pg = {k: [] for k in PER_GOLD}
    t = time.time()
    done = 0
    for lo in range(q0, q1, Q.BATCH):
        js = list(range(lo, min(q1, lo + Q.BATCH)))
        srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
        steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
        k0, a1 = K.restarts_batch(prep, js, linked, unions, srcs, constant)
        stz = [K.steer(prep, j, constant) for j in js]
        qs = [D5B.unit(np.asarray(prep.qemb[j], np.float64)) for j in js]
        wa = [A.omega(q, mu, Q.A_BETA, True)[0] for q in qs]
        wqs = [P.pair_weights(x, ws) for x in wa] if passage else wa
        wal = {w: A.rwalk(unions[Q.WALK_UNION[w]], a1 if w.startswith("A1") else k0,
                          stz if w.startswith("KT") else None, wqs, 1, ty, alpha, eps, bmax, threads) for w in Q.WALKS}
        for i, j in enumerate(js):
            src, _nn, (L, Ls_) = srcs[i]
            route = D.routes(L, Ls_)
            knd = K.kind(Ls_, prep.seeds[j], nonly)
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
            seeds = np.asarray(prep.seeds[j], np.int64)
            q = qs[i]
            w_all = stz[i]
            rows = M.Rows(nodes, verify=bool(limit))
            gj = golds[j]
            g0, g1 = int(ptr[j]), int(ptr[j + 1])
            per_b = []
            for bi in range(len(BASES)):
                B = np.unique(np.union1d(np.asarray(base_rows[bi][j], np.int64), seeds))
                n1, _cnt, _ws, _up, _inv = D5B.n1_table(adj, B, D5.weights_of(w_all, B))
                per_b.append({"B": B, "n1": n1, "base": np.asarray(base_rows[bi][j], np.int64),
                              "r0": F.outside(src[4][0], B)})
            rows.add(np.concatenate([x["n1"] for x in per_b]))
            for x in per_b:
                n1 = x["n1"]
                x["cos1"] = D5B.unit(rows.read(n1)) @ q if n1.size else np.empty(0, np.float32)
                x["cr"] = D5B.ranks_desc(x["cos1"], n1) if n1.size else EMPTY
                x["n2"], x["first"], x["taken"], x["hubs"] = U2E.second_link(adj, n1, x["cr"], x["B"])
            exp = [None] + [S4.m3b_pools.expand_hops(seeds, fams_full, st_) for _en, st_ in EXPANSIONS[1:]]
            inc = []
            for k in range(len(cells)):
                if k < len(GRID):
                    _nm, bi, ei = GRID[k]
                    pool, _added = op.m3b_compile.build_pool(per_b[bi]["base"], seeds, exp[ei])
                    inc.append((bi, np.asarray(pool, np.int64)))
                else:
                    inc.append((own_b, inc_own[j]))
            if not np.array_equal(inc[own_k][1], inc_own[j]):
                raise SystemExit(f"{name}/{carve} question {j}: the own construction's I is not the filed one")
            for k, (bi, I) in enumerate(inc):
                if not np.isin(seeds, per_b[bi]["B"]).all() or not np.isin(per_b[bi]["B"], I).all() or \
                        (k < len(GRID) and GRID[k][2] == 0 and not np.array_equal(I, per_b[bi]["B"])):
                    raise SystemExit(f"{name}/{carve} question {j}: cell {cells[k]}'s B or I is malformed")
            rows.add(np.concatenate(rec + [src[4][0]] + [I for _bi, I in inc] + [x["n2"] for x in per_b]))
            lens = np.zeros(len(cells) * R * NO, np.int32)
            n_u = np.zeros(len(cells) * R * len(MS), np.int64)
            fill = np.zeros(len(cells) * R, np.int64)
            pos = np.zeros((gj.size, len(cells) * R * NO), np.int32)
            for k, (bi, I) in enumerate(inc):
                x = per_b[bi]
                B = x["B"]
                iout = F.outside(I, B)
                in_b = np.isin(gj, B)
                for r in range(R):
                    full = F.outside(rec[r], B)
                    fill[k * R + r] = M.fill_of(full, I.size, B.size)
                    ords, sizes, U, mem = U2E.question_orders(full, x["n1"], x["cr"], x["cos1"], x["r0"], iout,
                                                              x["n2"], x["first"], rows.read, q, constant, D.rrf)
                    if not U2E.orders_exact(ords, B, full, x["n1"], U, mem):
                        raise SystemExit(f"{name}/{carve} question {j}: an order of {cells[k]}/{RECIPES[r]} is not "
                                         "exactly its sources' nodes")
                    for o, nm in enumerate(NAMES):
                        c = M.column(k, r, o)
                        lens[c] = ords[nm].size
                        pos[:, c] = U2E.places(gj, in_b, ords[nm])
                    n_u[(k * R + r) * len(MS):(k * R + r + 1) * len(MS)] = [sizes[m] for m in MS]
            in_I = np.stack([np.isin(gj, I) for _bi, I in inc], axis=1)
            one = np.stack([np.isin(gj, x["n1"]) for x in per_b], axis=1)
            bs_q = [per_b[bi]["B"].size for bi, _I in inc]
            inc_q = [I.size for _bi, I in inc]
            n_i_q = [F.outside(I, per_b[bi]["B"]).size for bi, I in inc]
            n_n2_q = [int((x["first"] < m).sum()) for x in per_b for m in M_GRID]
            if not (np.array_equal(bs_q, a2m["bs"][j]) and np.array_equal(inc_q, a2m["inc"][j])
                    and np.array_equal(n_i_q, a2m["n_i"][j])
                    and np.array_equal([x["n1"].size for x in per_b], a2m["n_n1"][j])
                    and np.array_equal([x["r0"].size for x in per_b], a2m["n_r0"][j])
                    and np.array_equal(n_n2_q, a2m["n_n2"][j])
                    and np.array_equal([x["taken"] for x in per_b], a2m["taken"][j])
                    and np.array_equal([x["hubs"] for x in per_b], a2m["hubs"][j])
                    and np.array_equal(in_I, a2m["in_I"][g0:g1]) and np.array_equal(one, a2m["one_link"][g0:g1])):
                raise SystemExit(f"{name}/{carve} question {j}: a walk-free count is not U2m's")
            mq = np.zeros(NMQ, np.int64)
            nw = np.zeros(len(BASES) * NW, np.int64)
            for bi, x in enumerate(per_b):
                for wi, w in enumerate(MASS_WALKS):
                    ms, cnt = mass_counts(wal[w][i][0], wal[w][i][1], x["B"])
                    mq[mq_col(bi, wi, 0):mq_col(bi, wi, 0) + NTW] = ms
                    nw[bi * NW + wi] = cnt
            if not mq_ok(mq, nw):
                raise SystemExit(f"{name}/{carve} question {j}: m_q breaks section 7's bounds")
            bo = per_b[own_b]["B"].size
            for p_, (_cell, r, o, b, tau) in enumerate(plans):
                c = M.column(own_k, r, o)
                _bn, kind, prm = bud[b]
                kq = int(U2E.budget_k(kind, prm, np.asarray([fill[own_k * R + r]]))[0])
                keep = int(P1H.keep_counts(np.asarray([bo]), np.asarray([min(int(lens[c]), kq)]), tau)[0])
                held = (pos[:, c] == 0) | ((pos[:, c] >= 1) & (pos[:, c] <= keep))
                if bo + keep != int(rows_f[j, p_]) or not np.array_equal(held, held_f[g0:g1, p_]):
                    raise SystemExit(f"{name}/{carve} question {j}: {names[p_]} is not its filed rows and held golds")
            pq["bs"].append(bs_q)
            pq["inc"].append(inc_q)
            pq["n_i"].append(n_i_q)
            pq["fill"].append(fill)
            pq["lens"].append(lens)
            pq["n_u"].append(n_u)
            pq["n_n1"].append([x["n1"].size for x in per_b])
            pq["n_r0"].append([x["r0"].size for x in per_b])
            pq["n_n2"].append(n_n2_q)
            pq["taken"].append([x["taken"] for x in per_b])
            pq["hubs"].append([x["hubs"] for x in per_b])
            pq["mq"].append(mq)
            pq["nw"].append(nw)
            pg["pos"].append(pos)
            pg["in_I"].append(in_I)
            pg["one_link"].append(one)
            done += 1
            if done % 100 == 0:
                log(f"{name}/{carve}: {done}/{q1 - q0} ({time.time() - t:.0f}s)")
    head = {"declared_in": DECLARED, "stage": "U2s", "dataset": name, "carve": carve, "graph_detail": desc,
            "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(), "zu2m_b_sha256": MB.script_sha256(),
            "zu2m_a_sha256": A.script_sha256(), "zu2m_a_types_sha256": AT.script_sha256(),
            "zu2p_sha256": P.script_sha256(), "zu2q_sha256": Q.script_sha256(), "zu2e_sha256": U2E.script_sha256(),
            "u2m_stats_sha256": r2m["file_sha256"], "verdict_a_sha256": sha_file(A.OUT / "verdict_a.json"),
            "verdict_p_sha256": sha_file(P.OUT / "verdict.json"), "pick": cw["pick"], "points": list(names),
            "G": G_, "beta": beta, "walk": walk_name, "rows_from": src_f, "a_stats_sha256": ra["file_sha256"],
            "em_types_sha256": arec["file_sha256"], "score_types_sha256": types_sha,
            "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "threads": threads, "alpha": alpha, "eps": eps,
            "nmax": bmax, "cells": cells, "own_cell": own_k, "cell_base": cell_base, "recipes": list(RECIPES),
            "orders": list(NAMES), "bases": [list(b) for b in BASES], "expansions": [[en, st_] for en, st_ in EXPANSIONS],
            "mass_walks": list(MASS_WALKS), "mass_taus": list(MASS_TAUS)}
    checks = {"graph_equal_u2m": True, "seeds_equal_g0": True, "em_types_equal_a": True, "kind_t_by_kind": True,
              "own_I_equal_filed": True, "cells_well_formed": True, "orders_exact": True,
              "walk_free_equal_u2m": True, "mq_bounds": True, "points_equal_filed": True,
              "rows_served_verified": bool(limit)}
    if limit:
        log(f"{name}/{carve}: development run on questions {q0}-{q1 - 1} of {nq} ({time.time() - t:.0f}s): every "
            "check passed, every served row verified; nothing written")
        return 0
    arr = arrays_of(pq, pg, ptr[q0:q1 + 1], len(cells))
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


def arrays_of(pq, pg, ptr, nk):
    a = M.arrays_of({k: pq[k] for k in M.PER_QUESTION}, pg, ptr, nk)
    a["mq"] = np.asarray(pq["mq"], np.int64).reshape(-1, NMQ)
    a["nw"] = np.asarray(pq["nw"], np.int64).reshape(-1, len(BASES) * NW)
    return {k: a[k] for k in ARRAYS}


def summary_of(a, cells):
    """U2m's summary of the arrays, and per base, MASS walk and tau_w the mean m_q (with the walk's mean count
    outside B)."""
    out = M.summary_of(a, cells)
    out["mq"] = {f"{bs}/{w}": {"outside_B": round(float(a["nw"][:, bi * NW + wi].mean()), 1),
                               **{f"t{tw:g}": round(float(a["mq"][:, mq_col(bi, wi, ti)].mean()), 1)
                                  for ti, tw in enumerate(MASS_TAUS)}}
                 for bi, (bs, _bp) in enumerate(BASES) for wi, w in enumerate(MASS_WALKS)}
    return out


def write_stats(name, carve, arr, head, checks, seconds, extra=None):
    od = OUT / "stats"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{name}__{carve}.npz"
    tmp = od / f"{name}__{carve}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, p)
    nq = int(arr["bs"].shape[0])
    rec = {**head, "questions": nq, "golds": int(arr["ptr"][-1]), "file_sha256": sha_file(p),
           "summary": summary_of(arr, head["cells"]), "checks": checks, "seconds": seconds, **(extra or {}),
           "utc": utc()}
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


def _ok_checks(r):
    return all(v for c, v in r["checks"].items() if c != "rows_served_verified")


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
        with np.load(p) as z:
            parts.append({a: z[a] for a in z.files})
        recs.append(r)
    head = {a: recs[0][a] for a in HEAD_KEYS}
    nq = recs[0]["questions_all"]
    for i, r in enumerate(recs):
        if {a: r[a] for a in HEAD_KEYS} != head or r["checks"] != recs[0]["checks"] or r["questions_all"] != nq:
            raise SystemExit(f"part {i}: another script, freeze, walk, graph, thread count, check set or carve")
        if (r["part"], r["parts"]) != (i, k) or (r["q0"], r["q1"]) != U2C.shard_range(nq, i, k):
            raise SystemExit(f"part {i}: not the {i}-th of {k} question ranges")
    arr = merge_arrays(parts)
    a2m = M.load_u2m(name)[0]
    if arr["bs"].shape[0] != nq or not np.array_equal(a2m["ptr"][:nq + 1], arr["ptr"]):
        raise SystemExit(f"{name}/{carve}: the merged parts do not cover U2m's questions and golds")
    return write_stats(name, carve, arr, head, recs[0]["checks"], round(sum(r["seconds"] for r in recs), 1),
                       {"shards": k, "part_sha256": [r["file_sha256"] for r in recs],
                        "seconds_longest_part": max(r["seconds"] for r in recs)})


def load_u2s(name):
    carve = U2C.carve_of(name)
    rp = OUT / "stats" / f"{name}__{carve}.json"
    if not rp.exists():
        return None
    rec = json.loads(rp.read_text(encoding="utf-8"))
    p = OUT / "stats" / f"{name}__{carve}.npz"
    if rec.get("script_sha256") != script_sha256() or sha_file(p) != rec["file_sha256"] or not _ok_checks(rec):
        raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
    with np.load(p) as z:
        a = {k: z[k] for k in z.files}
    if rec["cells"][:len(GRID)] != [g[0] for g in GRID] or rec["recipes"] != list(RECIPES) or \
            rec["orders"] != list(NAMES) or rec["mass_walks"] != list(MASS_WALKS) or \
            rec["mass_taus"] != list(MASS_TAUS) or summary_of(a, rec["cells"]) != rec["summary"]:
        raise SystemExit(f"{p}: not this script's cells, recipes, orders or MASS walks, or the arrays do not give "
                         "its summary")
    return a, rec


# ── the verdict (the laptop) ─────────────────────────────────────────────────


def column_table(b, ln, p, KQ, ptr, taus=TAUS_DESC):
    """(ALL, rows), budgets x taus, of one order's column at every budget (KQ: budgets x questions): U2e's evaluate
    vectorized over the budgets, each value rounded as P1g's summarize rounds it."""
    ng = np.diff(ptr)
    nq = ng.size
    qi = np.repeat(np.arange(nq), ng)
    has = ng > 0
    nb = KQ.shape[0]
    kk = np.minimum(np.asarray(ln, np.int64)[None, :], KQ)
    bt = np.ascontiguousarray(np.broadcast_to(b, (nb, nq))).ravel()
    p = np.asarray(p, np.int64)[None, :]
    AL, RW = np.empty((nb, len(taus))), np.empty((nb, len(taus)))
    for ti, t in enumerate(taus):
        keep = P1H.keep_counts(bt, kk.ravel(), t).reshape(nb, nq)
        held = (p == 0) | ((p >= 1) & (p <= keep[:, qi]))
        cs = np.concatenate([np.zeros((nb, 1), np.int64), np.cumsum(held, axis=1, dtype=np.int64)], axis=1)
        hits = cs[:, ptr[1:]] - cs[:, ptr[:-1]]
        n_all = (has[None, :] & (hits == ng[None, :])).sum(axis=1)
        n_rows = (b[None, :] + keep).sum(axis=1)
        AL[:, ti] = [round(float(x) / nq, 4) for x in n_all.tolist()]
        RW[:, ti] = [round(float(x) / nq, 1) for x in n_rows.tolist()]
    return AL, RW


def table_arrays(a, ks, rs, cell_base):
    """(ALL, rows) of every candidate of cells ks x recipes rs on one dataset's arrays, flat in (cell, recipe, order,
    budget, tau from 1 down) order (section 5)."""
    ptr = np.asarray(a["ptr"], np.int64)
    bud = budgets()
    AL = np.empty((len(ks), len(rs), NO, NB, NT))
    RW = np.empty_like(AL)
    for ki, k in enumerate(ks):
        b = np.asarray(a["bs"][:, k], np.int64)
        mq = mq_of(a, cell_base[k])
        for ri, r in enumerate(rs):
            fl = a["fill"][:, k * R + r]
            KQ = np.stack([budget_k(kind, prm, fl, mq) for _bn, kind, prm in bud])
            for o in range(NO):
                c = M.column(k, r, o)
                AL[ki, ri, o], RW[ki, ri, o] = column_table(b, a["lens"][:, c], a["pos"][:, c], KQ, ptr)
    return AL.ravel(), RW.ravel()


def decode(i, nk, nr):
    """(cell slot, recipe slot, order, budget, tau slot) of a flat table index over nk cells x nr recipes."""
    return tuple(int(x) for x in np.unravel_index(int(i), (nk, nr, NO, NB, NT)))


def held_of(a, k, r, o, b, t, cell_base):
    """(rows, held) of one candidate on one dataset's arrays: U2e's evaluate with section 4's budget."""
    _bn, kind, prm = budgets()[b]
    c = M.column(k, r, o)
    sub = {"bs": a["bs"][:, k], "lens": a["lens"][:, [c]], "pos": a["pos"][:, [c]], "ptr": a["ptr"]}
    return U2E.evaluate(sub, 0, budget_k(kind, prm, a["fill"][:, k * R + r], mq_of(a, cell_base[k])), t)


def three_frontiers(Au, cu, cru, gru, Ao, co, cro, oro, ceil, cap):
    """Section 6's frontiers over the universal table (ALL, cost, rounded cost, recipe per candidate) and the own one:
    {key: (points [(t, targets, index or None)], level, index)} for C (universal kind_t), E (universal rrf5t and both)
    and O (the own construction's kind_t); indices are into the key's own table."""
    out = {}
    for key, idx, Am, cm, crm in (("C", np.flatnonzero(gru == 0), Au, cu, cru),
                                  ("E", np.flatnonzero(np.isin(gru, KIND_FREE)), Au, cu, cru),
                                  ("O", np.flatnonzero(oro == 0), Ao, co, cro)):
        pts = [(t, tg, None if k_ is None else int(idx[k_])) for t, tg, k_ in U2E.frontier(Am[idx], cm[idx], ceil)]
        lv, kk = M.pick_of(pts, crm, cap)
        out[key] = (pts, lv, kk)
    return out


def level_of(al, ceil):
    """The highest level whose targets min(t, ceil - tol) one candidate's ALL (TRAIN5) meets, or None."""
    best = None
    for t in LEVELS:
        tgt = np.round(np.minimum(t, np.asarray(ceil, np.float64) - PG.TOL), 4)
        if (np.asarray(al, np.float64) >= tgt - 1e-9).all():
            best = t
    return best


def decide(l_c, l_e, l_o, cost_o, inc_level, inc_cost):
    """Section 6: (verdict (c), verdict (e), verdict (o), what is carried: 'E', 'C', 'O' or 'U2p')."""
    floor = inc_level - PG.TOL
    if l_c is None or l_c < floor - 1e-9:
        vc = "NOT_UNIFIED"
    elif l_c > inc_level + 1e-9:
        vc = "UNIFIED_RAISED"
    elif abs(l_c - inc_level) < 1e-9:
        vc = "UNIFIED_HOLDS"
    else:
        vc = "UNIFIED_AT_A_PRICE"
    if l_e is not None and l_e >= floor - 1e-9 and (l_c is None or l_e >= l_c - 1e-9):
        ve = "KIND_FREE_NO_PRICE"
    elif l_e is not None and l_c is not None and l_e >= floor - 1e-9 and abs(l_e - (l_c - PG.TOL)) < 1e-9:
        ve = "KIND_FREE_AT_A_PRICE"
    else:
        ve = "KIND_FREE_TOO_DEAR"
    if l_o is not None and l_o > inc_level + 1e-9:
        vo = "OWN_RAISED"
    elif l_o is not None and abs(l_o - inc_level) < 1e-9 and cost_o is not None and cost_o < inc_cost - 1e-9:
        vo = "OWN_CHEAPER"
    else:
        vo = "NO_GAIN"
    if ve != "KIND_FREE_TOO_DEAR":
        carried = "E"
    elif vc != "NOT_UNIFIED":
        carried = "C"
    elif vo != "NO_GAIN":
        carried = "O"
    else:
        carried = "U2p"
    return vc, ve, vo, carried


def avg_ranks(x):
    x = np.asarray(x, np.float64)
    o = np.argsort(x, kind="stable")
    _u, start, cnt = np.unique(x[o], return_index=True, return_counts=True)
    r = np.empty(x.size)
    r[o] = np.repeat(start + (cnt - 1) / 2.0 + 1.0, cnt)
    return r


def spearman(x, y):
    """Spearman's rank correlation (average ranks for ties); None when either side is constant."""
    rx, ry = avg_ranks(x), avg_ranks(y)
    if rx.size < 2 or rx.std() == 0 or ry.std() == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 4)


def verdict_stage():
    t0 = time.time()
    cw = Q.carried_walk()
    vp = cw["verdict_p"]
    if vp["verdict"] == "NO_GAIN":
        inc_level, inc_cost = float(P.CARRIED[1]), float(P.CARRIED[2])
    else:
        inc_level, inc_cost = float(vp["level"]), float(vp["cost"])
    inc_name, cap = cw["pick"], float(vp["cap"])
    inc_point = cw["names"][cw["pi"]]
    am, rm, missing = {}, {}, []
    for d in SIX:
        x = load_u2s(d)
        if x is None:
            missing.append(d)
        else:
            am[d], rm[d] = x
    if missing:
        raise SystemExit(f"INCOMPLETE: no U2s stats for {missing}")
    for d in SIX:
        r = rm[d]
        if (r["pick"], r["points"], r["G"], r["beta"]) != (cw["pick"], list(cw["names"]), cw["G"], cw["beta"]) or \
                r["verdict_p_sha256"] != sha_file(P.OUT / "verdict.json"):
            raise SystemExit(f"{d}: U2s's stats are not on the walk U2p's filed verdict carries")
        if not np.array_equal(am[d]["ptr"], M.load_u2m(d)[0]["ptr"]):
            raise SystemExit(f"{d}: U2s's questions and golds are not U2m's")
    cb = {d: rm[d]["cell_base"] for d in SIX}
    ok_ = {d: rm[d]["own_cell"] for d in SIX}
    tabs = {d: table_arrays(am[d], range(len(GRID)), range(R), cb[d]) for d in SIX}
    own = {d: table_arrays(am[d], [ok_[d]], range(R), cb[d]) for d in SIX}
    log(f"U2s verdict: tables of {tabs['metaqa'][0].size} universal and {own['metaqa'][0].size} own candidates on six "
        f"datasets ({time.time() - t0:.0f}s)")
    bud = budgets()
    w = F.weights()

    def own_index(point):
        _cell, r, o, b, t = MB.parse(point)
        return int(np.ravel_multi_index((0, r, o, b, TAUS_DESC.index(t)), (1, R, NO, NB, NT)))

    # the check: U2p's filed table at the carried frontier's 8 points, on all six, from U2s's arrays
    cfg = P.config_name(cw["ci"])
    for pt in cw["names"]:
        i = own_index(pt)
        e = vp["every_candidate"][f"{pt}|G{cw['G']}|{cw['a_cfg']}|{cfg}"]
        for d in SIX:
            if (float(own[d][0][i]), float(own[d][1][i])) != (float(e[d]["ALL"]), float(e[d]["rows"])):
                raise SystemExit(f"{d}: U2s's table is not U2p's filed table at {pt}")
        c_ = round(F.cost({d: float(own[d][1][i]) for d in TRAIN5}, w), 1)
        if c_ != float(e["cost"]):
            raise SystemExit(f"{pt}: U2s's cost {c_} is not U2p's filed {e['cost']}")
    log(f"U2s verdict: U2p's filed table reproduced at the carried frontier's {len(cw['names'])} points "
        f"({time.time() - t0:.0f}s)")

    nu = len(GRID) * R * NO * NB * NT
    A_ = np.stack([tabs[d][0] for d in TRAIN5], axis=1)
    RW = np.stack([tabs[d][1] for d in TRAIN5], axis=1)
    costs = np.asarray([F.cost(dict(zip(TRAIN5, r)), w) for r in RW.tolist()], np.float64)
    cost_r = np.asarray([round(float(c), 1) for c in costs])
    ceil = A_.max(axis=0)
    gk, gr, go, gb, _gt = np.unravel_index(np.arange(nu), (len(GRID), R, NO, NB, NT))
    form_of_b = np.asarray([FORMS.index(x[1]) for x in bud])
    gf = form_of_b[gb]
    no_ = R * NO * NB * NT
    A_o = np.stack([own[d][0] for d in TRAIN5], axis=1)
    RW_o = np.stack([own[d][1] for d in TRAIN5], axis=1)
    c_o = np.asarray([F.cost(dict(zip(TRAIN5, r)), w) for r in RW_o.tolist()], np.float64)
    cr_o = np.asarray([round(float(c), 1) for c in c_o])
    _ok, orr, oo, ob, _ot = np.unravel_index(np.arange(no_), (1, R, NO, NB, NT))
    of = form_of_b[ob]
    i_inc = own_index(inc_point)
    l_inc = level_of(A_o[i_inc], ceil)
    if l_inc is None or abs(l_inc - inc_level) > 1e-9 or abs(cr_o[i_inc] - inc_cost) > 1e-9:
        raise SystemExit(f"the incumbent gives level {l_inc} at {cr_o[i_inc]}, not U2p's {inc_level} at {inc_cost}")
    log(f"U2s verdict: {nu} universal candidates, ceilings {dict(zip(TRAIN5, ceil.tolist()))} "
        f"({time.time() - t0:.0f}s)")

    def name_u(i):
        k, r, o, b, t = decode(i, len(GRID), R)
        return M.candidate_name(GRID[k][0], r, o, bud[b][0], TAUS_DESC[t])

    def name_o(i):
        _k, r, o, b, t = decode(i, 1, R)
        return M.candidate_name("own", r, o, bud[b][0], TAUS_DESC[t])

    tab_u = {"name": name_u, "A": {d: tabs[d][0] for d in SIX}, "R": {d: tabs[d][1] for d in SIX}, "cr": cost_r}
    tab_o = {"name": name_o, "A": {d: own[d][0] for d in SIX}, "R": {d: own[d][1] for d in SIX}, "cr": cr_o}

    def point(t, tgt, i, tb):
        if i is None:
            return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": None}
        return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": tb["name"](i),
                "cost": float(tb["cr"][i]), "within_cap": bool(tb["cr"][i] <= cap + 1e-9),
                "s1sel": {d: {"ALL": float(tb["A"][d][i]), "rows": float(tb["R"][d][i])} for d in TRAIN5},
                "webqsp_s1eval": {"ALL": float(tb["A"]["webqsp"][i]), "rows": float(tb["R"]["webqsp"][i])}}

    tf = three_frontiers(A_, costs, cost_r, gr, A_o, c_o, cr_o, orr, ceil, cap)
    fr = {key: {"points": [point(t, tg, k_, tab_o if key == "O" else tab_u) for t, tg, k_ in pts], "level": lv,
                "index": kk} for key, (pts, lv, kk) in tf.items()}
    log(f"U2s verdict: frontiers C {fr['C']['level']}, E {fr['E']['level']}, O {fr['O']['level']} "
        f"({time.time() - t0:.0f}s)")
    cost_o = None if fr["O"]["index"] is None else float(cr_o[fr["O"]["index"]])
    vc, ve, vo, carried = decide(fr["C"]["level"], fr["E"]["level"], fr["O"]["level"], cost_o, inc_level, inc_cost)

    def pick_rec(key):
        i = fr[key]["index"]
        if i is None:
            return {"level": None, "name": None, "cost": None, "form": None}
        if key == "O":
            return {"level": fr[key]["level"], "name": name_o(i), "cost": float(cr_o[i]), "form": FORMS[of[i]]}
        return {"level": fr[key]["level"], "name": name_u(i), "cost": float(cost_r[i]), "form": FORMS[gf[i]]}

    picks = {key: pick_rec(key) for key in ("C", "E", "O")}
    carried_name = inc_name.rsplit("|", 3)[0] if carried == "U2p" else picks[carried]["name"]

    # beside: the cheapest point of each budget form, construction and recipe at every level, and the own frontiers
    def cheapest(groups, Am, cm, crm, namef):
        out = {}
        for g_, idx in groups.items():
            out[g_] = [None if k_ is None else {"pick": namef(int(idx[k_])), "cost": float(crm[idx[k_]])}
                       for _t, _tg, k_ in U2E.frontier(Am[idx], cm[idx], ceil)]
        return out
    by_form = cheapest({f: np.flatnonzero(gf == fi) for fi, f in enumerate(FORMS)}, A_, costs, cost_r, name_u)
    by_form_c = cheapest({f: np.flatnonzero((gf == fi) & (gr == 0)) for fi, f in enumerate(FORMS)}, A_, costs, cost_r,
                         name_u)
    by_form_own = cheapest({f: np.flatnonzero((of == fi) & (orr == 0)) for fi, f in enumerate(FORMS)}, A_o, c_o, cr_o,
                           name_o)
    by_cell = cheapest({GRID[k][0]: np.flatnonzero(gk == k) for k in range(len(GRID))}, A_, costs, cost_r, name_u)
    by_recipe = cheapest({rn: np.flatnonzero(gr == r) for r, rn in enumerate(RECIPES)}, A_, costs, cost_r, name_u)
    own_fr = cheapest({rn: np.flatnonzero(orr == r) for r, rn in enumerate(RECIPES)}, A_o, c_o, cr_o, name_o)
    per_dataset, union_ceiling, mq_beside = {}, {}, {}
    for d in SIX:
        al, rw = tabs[d]
        cd = float(al.max())
        rows_at = {}
        for t in LEVELS:
            tg = round(min(t, cd - PG.TOL), 4)
            i = np.flatnonzero(al >= tg - 1e-9)
            k_ = int(i[np.argmin(rw[i])])
            rows_at[str(t)] = {"target": tg, "rows": float(rw[k_]), "ALL": float(al[k_]), "candidate": name_u(k_)}
        per_dataset[d] = {"ceiling": cd, "fewest_rows": rows_at}
        reach = (am[d]["pos"] >= 0).any(axis=1)
        union_ceiling[d] = {"ALL": U2E.all_of(reach, am[d]["ptr"]), "golds_outside": int((~reach).sum())}
        bo = cb[d][ok_[d]]
        fl = am[d]["fill"][:, ok_[d] * R]
        mq_beside[d] = {}
        for wi, wn in enumerate(MASS_WALKS):
            e = {"outside_B_median": float(np.median(am[d]["nw"][:, bo * NW + wi]))}
            for ti, tw in enumerate(MASS_TAUS):
                m = am[d]["mq"][:, mq_col(bo, wi, ti)]
                e[f"t{tw:g}"] = {"median": float(np.median(m)), "p90": float(np.percentile(m, 90)),
                                 "mean": round(float(m.mean()), 1), "spearman_fill": spearman(m, fl)}
            mq_beside[d][wn] = e
        mq_beside[d]["today_fill"] = {"median": float(np.median(fl)), "p90": float(np.percentile(fl, 90))}

    # the picks and the carried one: their arrays, their rule at tau = 1, the ledger and the price against U2p's pick
    def args_of(key, d):
        if key == "U2p":
            _k, r, o, b, t = decode(i_inc, 1, R)
            return ok_[d], r, o, b, TAUS_DESC[t]
        i = fr[key]["index"]
        if key == "O":
            _k, r, o, b, t = decode(i, 1, R)
            return ok_[d], r, o, b, TAUS_DESC[t]
        k, r, o, b, t = decode(i, len(GRID), R)
        return k, r, o, b, TAUS_DESC[t]

    ledgers, price = {}, {}
    for key in [x for x in ("C", "E", "O") if fr[x]["index"] is not None] + (["U2p"] if carried == "U2p" else []):
        ledgers[key] = {}
        for d in SIX:
            a = am[d]
            reach = (a["pos"] >= 0).any(axis=1)
            k, r, o, b, t = args_of(key, d)
            rows, held = held_of(a, k, r, o, b, t, cb[d])
            _r1, held1 = held_of(a, k, r, o, b, 1.0, cb[d])
            ledgers[key][d] = U2E.ledger_of(held, held1 | held, reach | held1 | held, a["ptr"])
            if key == carried:
                price[d] = {"carried": {"ALL": U2E.all_of(held, a["ptr"]), "rows": round(float(rows.mean()), 1)},
                            "u2p_pick": {"ALL": float(own[d][0][i_inc]), "rows": float(own[d][1][i_inc])}}
    if carried == "U2p" or not price:
        for d in SIX:
            price[d] = {"carried": {"ALL": float(own[d][0][i_inc]), "rows": float(own[d][1][i_inc])},
                        "u2p_pick": {"ALL": float(own[d][0][i_inc]), "rows": float(own[d][1][i_inc])}}
    nxt = {"E": "one rule on all six (kind-free): U2n builds on these pools; (b)'s restore is re-read and U2q's arrays "
                "recomputed on them, each in its own file",
           "C": "one rule on all six (kind_t): U2n builds on these pools; (b)'s restore is re-read and U2q's arrays "
                "recomputed on them, each in its own file",
           "O": "the own construction with pick O's budget; the unified form's next forms: a budget from RRF5T's "
                "agreement across its five orders, and P1h's cut over the walk's mass",
           "U2p": "U2p's pick stays; (c)'s next forms: a budget from RRF5T's agreement across its five orders, and "
                  "P1h's cut over the walk's mass"}[carried]
    rec = {"declared_in": DECLARED, "stage": "U2s", "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(),
           "zu2q_sha256": Q.script_sha256(), "zu2e_sha256": U2E.script_sha256(),
           "verdict_p_sha256": sha_file(P.OUT / "verdict.json"), "verdict_c": vc, "verdict_e": ve, "verdict_o": vo,
           "carried": carried, "carried_pick": carried_name, "next": nxt, "cap": cap,
           "u2p_pick": {"name": inc_name, "level": inc_level, "cost": inc_cost},
           "pick_C": picks["C"], "pick_E": picks["E"], "pick_O": picks["O"], "weights": w,
           "candidates": {"universal": nu, "own": no_, "budgets": NB, "mass_budgets": NB - len(U2E.budgets())},
           "ceilings": dict(zip(TRAIN5, ceil.tolist())), "levels": list(LEVELS),
           "u2p_points_reproduced": list(cw["names"]),
           "frontier_C": fr["C"]["points"], "frontier_E": fr["E"]["points"], "frontier_O": fr["O"]["points"],
           "own_frontiers": own_fr, "by_form": by_form, "by_form_kind_t": by_form_c, "by_form_own": by_form_own,
           "by_cell": by_cell, "by_recipe": by_recipe, "per_dataset": per_dataset, "union_ceiling": union_ceiling,
           "ledgers": ledgers, "price": price, "mq": mq_beside,
           "own_cells": {d: rm[d]["cells"][ok_[d]] for d in SIX},
           "stats": {d: {"file_sha256": rm[d]["file_sha256"], "seconds": rm[d]["seconds"]} for d in SIX},
           "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "verdict.json", rec)
    (OUT / "verdict.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"U2s: {vc}, {ve}, {vo}; carried {carried}: {carried_name} ({time.time() - t0:.0f}s)")
    return 0


def report_md(rec):
    def pk(p):
        if p["name"] is None:
            return "none within the cap"
        return f"`{p['name']}` ({p['form']}) at level {p['level']}, cost {p['cost']}"
    L = [f"# U2s: {rec['verdict_c']}, {rec['verdict_e']}, {rec['verdict_o']}", "",
         f"Declared in {rec['declared_in']}. Carried forward: **{rec['carried']}**, `{rec['carried_pick']}`.", "",
         f"- Pick C (kind_t, universal): {pk(rec['pick_C'])}.",
         f"- Pick E (kind-free, universal): {pk(rec['pick_E'])}.",
         f"- Pick O (the own construction, kind_t, 219 budgets): {pk(rec['pick_O'])}.",
         f"- U2p's pick: `{rec['u2p_pick']['name']}` at level {rec['u2p_pick']['level']}, cost "
         f"{rec['u2p_pick']['cost']}. Cap {rec['cap']}.",
         f"- U2p's filed table reproduced at its {len(rec['u2p_points_reproduced'])} points on all six.",
         f"- Next: {rec['next']}.", "",
         "## The carried pick against U2p's pick", "",
         "| dataset | carried ALL | rows | U2p's pick ALL | rows | union ceiling |", "|---|---:|---:|---:|---:|---:|"]
    for d, p in rec["price"].items():
        L.append(f"| {d} | {p['carried']['ALL']} | {p['carried']['rows']} | {p['u2p_pick']['ALL']} | "
                 f"{p['u2p_pick']['rows']} | {rec['union_ceiling'][d]['ALL']} |")
    for key, title in (("frontier_C", "Frontier C (universal kind_t)"), ("frontier_E", "Frontier E (universal "
                       "kind-free)"), ("frontier_O", "Frontier O (the own construction, kind_t)")):
        L += ["", f"## {title}: every training dataset at min(level, its ceiling - 0.005)", "",
              "| level | cost | within cap | pick | " + " | ".join(f"{d} ALL (rows)" for d in TRAIN5)
              + " | webqsp ALL (rows) |", "|---|---:|---|---|" + "---|" * (len(TRAIN5) + 1)]
        for e in rec[key]:
            if e["pick"] is None:
                L.append(f"| {e['level']} | - | - | none meets | " + " | ".join("-" for _ in TRAIN5) + " | - |")
                continue
            L.append(f"| {e['level']} | {e['cost']} | {e['within_cap']} | `{e['pick']}` | "
                     + " | ".join(f"{e['s1sel'][d]['ALL']} ({e['s1sel'][d]['rows']})" for d in TRAIN5)
                     + f" | {e['webqsp_s1eval']['ALL']} ({e['webqsp_s1eval']['rows']}) |")
    for key, title in (("by_form", "budget form (universal, every recipe)"), ("by_form_kind_t", "budget form "
                       "(universal kind_t)"), ("by_form_own", "budget form (the own construction, kind_t)"),
                       ("by_cell", "construction"), ("by_recipe", "recipe")):
        L += ["", f"## The cheapest point of each {title} (cost; - when none meets)", "",
              "| level | " + " | ".join(rec[key]) + " |", "|---|" + "---:|" * len(rec[key])]
        for i, t in enumerate(rec["levels"]):
            L.append(f"| {t} | " + " | ".join("-" if v[i] is None else str(v[i]["cost"]) for v in rec[key].values())
                     + " |")
    L += ["", "## The own construction's frontier per recipe (cost; beside)", "",
          "| level | " + " | ".join(rec["own_frontiers"]) + " |", "|---|" + "---:|" * len(rec["own_frontiers"])]
    for i, t in enumerate(rec["levels"]):
        L.append(f"| {t} | " + " | ".join("-" if v[i] is None else str(v[i]["cost"])
                                          for v in rec["own_frontiers"].values()) + " |")
    L += ["", "## Per dataset: the fewest mean rows reaching each level, universal candidates (an upper bound only)", "",
          "| dataset | ceiling | " + " | ".join(str(t) for t in rec["levels"]) + " |",
          "|---|---:|" + "---:|" * len(rec["levels"])]
    for d, e in rec["per_dataset"].items():
        L.append(f"| {d} | {e['ceiling']} | " + " | ".join(str(e["fewest_rows"][str(t)]["rows"]) for t in rec["levels"])
                 + " |")
    L += ["", "## m_q on the own construction's base: median / 90th percentile (Spearman with today's fill)", "",
          "| dataset | today's fill | " + " | ".join(f"{w} t{tw:g}" for w in MASS_WALKS for tw in MASS_TAUS) + " |",
          "|---|---:|" + "---:|" * (NW * NTW)]
    for d, e in rec["mq"].items():
        L.append(f"| {d} | {e['today_fill']['median']} / {e['today_fill']['p90']} | "
                 + " | ".join(f"{e[w][f't{tw:g}']['median']} / {e[w][f't{tw:g}']['p90']} "
                              f"({e[w][f't{tw:g}']['spearman_fill']})" for w in MASS_WALKS for tw in MASS_TAUS) + " |")
    for key, led in rec["ledgers"].items():
        L += ["", f"## Ledger: pick {key}'s misses", "",
              "| dataset | golds | missed | TAU_CUT | BUDGET | UNREACHED | questions missing a gold |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for d, e in led.items():
            L.append(f"| {d} | {e['golds']} | {e['missed']} | {e['by_cause']['TAU_CUT']} | {e['by_cause']['BUDGET']} | "
                     f"{e['by_cause']['UNREACHED']} | {e['questions_missing_a_gold']} |")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _brute_mq(nodes, mass, B, taus):
    keep = [(n, m) for n, m in zip(np.asarray(nodes).tolist(), np.asarray(mass, np.float64).tolist()) if n not in set(B)]
    tot = 0.0
    for _n, m in keep:
        tot += m
    out = []
    for t in taus:
        if tot <= 0:
            out.append(0)
            continue
        s, m_ = 0.0, 0
        while s < t * tot:
            s += keep[m_][1]
            m_ += 1
        out.append(m_)
    return out, len(keep)


def _selftest():
    rng = np.random.default_rng(0)
    # 1. the budgets: U2e's 99 first, then the 120 MASS budgets in section 4's order, names unique
    bud = budgets()
    assert NB == 219 and bud[:99] == U2E.budgets() and len({b[0] for b in bud}) == 219
    mass = bud[99:]
    assert len(mass) == 120 and all(b[1] == "MASS" for b in mass)
    assert mass[0][0] == "mass_A1all_t0.5_f1_k0" and mass[-1][0] == "mass_KTall_t0.99_f8_k300"
    nm = [b[0] for b in bud]
    assert bud[nm.index("mass_KTall_t0.9_f4_k50")][2] == (1, 2, 4, 50)
    assert {x for _n, _k, (w, t, f, k0) in mass for x in [(w, t, f, k0)]} == \
        {(w, t, f, k0) for w in range(NW) for t in range(NTW) for f in MASS_F for k0 in MASS_K0}
    # 2. m_q against a brute count: ties, B inside the list, an empty walk, a walk all inside B
    for trial in range(300):
        n = int(rng.integers(0, 60))
        nodes = rng.permutation(200)[:n]
        m = np.sort(rng.integers(1, 6, n).astype(np.float64) / rng.choice([1.0, 7.0, 3.0]))[::-1]
        if trial % 7 == 0:
            m = np.full(n, 0.25)
        B = np.unique(np.concatenate([rng.choice(200, int(rng.integers(0, 30)), replace=False),
                                      nodes[:int(rng.integers(0, n + 1))] if trial % 5 == 0 else nodes[:0]]))
        got, cnt = mass_counts(nodes, m, B)
        want, wc = _brute_mq(nodes, m, B, MASS_TAUS)
        assert got == want and cnt == wc, (trial, got, want)
        assert all(a <= b for a, b in zip(got, got[1:])) and all(x <= cnt for x in got) and ((got[0] == 0) == (cnt == 0))
    assert mass_counts(EMPTY, np.zeros(0), EMPTY) == ([0] * NTW, 0)
    assert mass_counts(np.arange(5), np.ones(5), np.arange(10)) == ([0] * NTW, 0)
    assert mass_counts(np.arange(4), np.asarray([0.5, 0.3, 0.1, 0.1]), EMPTY)[0] == [1, 2, 3, 4, 4]
    assert mass_counts(np.arange(4), np.asarray([0.5, 0.3, 0.1, 0.1]), np.asarray([0]))[0] == [1, 2, 3, 3, 3]
    # 3. the m_q layout and its bounds
    assert {mq_col(b, w, t) for b in range(3) for w in range(NW) for t in range(NTW)} == set(range(NMQ))
    good = np.asarray([[1, 2, 2, 3, 5], [0, 0, 0, 0, 0]] * 3).ravel()
    assert mq_ok(good, [6, 0] * 3) and not mq_ok(good, [4, 0] * 3) and not mq_ok(good, [6, 1] * 3)
    assert not mq_ok(np.asarray([[2, 1, 2, 3, 5], [0, 0, 0, 0, 0]] * 3).ravel(), [6, 0] * 3)
    # 4. the vectorized table against U2e's evaluate and summ, on toy arrays of three cells over two bases
    nq, nk = 40, 3
    cell_base = [0, 2, 2]
    ptr = np.concatenate([[0], np.cumsum(rng.integers(0, 4, nq))]).astype(np.int64)
    ng = int(ptr[-1])
    qi = np.repeat(np.arange(nq), np.diff(ptr))
    a = {"bs": rng.integers(0, 6, (nq, nk)).astype(np.int64), "fill": rng.integers(0, 300, (nq, nk * R)),
         "lens": rng.integers(0, 3000, (nq, nk * R * NO)).astype(np.int32), "ptr": ptr,
         "mq": rng.integers(0, 900, (nq, NMQ)).astype(np.int64)}
    a["mq"][rng.random(nq) < 0.2] = 0
    pos = rng.integers(-1, 3000, (ng, nk * R * NO))
    pos = np.where(pos > a["lens"][qi], -1, pos)
    for k in range(nk):
        cs_ = [M.column(k, r, o) for r in range(R) for o in range(NO)]
        sub = pos[:, cs_]
        sub[(sub == 0) & (a["bs"][qi, k] == 0)[:, None]] = -1
        pos[:, cs_] = sub
    a["pos"] = pos.astype(np.int32)
    AL, RW = table_arrays(a, range(nk), range(R), cell_base)
    assert AL.size == nk * R * NO * NB * NT
    for i in list(rng.choice(AL.size, 700, replace=False)) + [0, AL.size - 1]:
        k, r, o, b, t = decode(i, nk, R)
        rows_, held = held_of(a, k, r, o, b, TAUS_DESC[t], cell_base)
        s = U2E.summ(rows_, held, ptr, "x")
        assert (AL[i], RW[i]) == (s["ALL"], s["rows"]), (k, r, o, b, t)
    for i in np.flatnonzero(np.isin(np.unravel_index(np.arange(AL.size), (nk, R, NO, NB, NT))[3],
                                    np.arange(99, NB)))[::997]:
        k, r, o, b, t = decode(i, nk, R)
        _bn, kind, (wi, ti, f, k0) = bud[b]
        want_k = np.maximum(k0, np.rint(f * a["mq"][:, mq_col(cell_base[k], wi, ti)]).astype(np.int64))
        assert np.array_equal(budget_k(kind, (wi, ti, f, k0), a["fill"][:, k * R + r], mq_of(a, cell_base[k])), want_k)
    e = {"bs": a["bs"][:, 1], "fill": a["fill"][:, R + 2], "lens": a["lens"][:, [M.column(1, 2, o) for o in range(NO)]],
         "pos": a["pos"][:, [M.column(1, 2, o) for o in range(NO)]], "ptr": ptr}
    t2 = U2E.u2e_table({"x": e}, U2E.u2e_candidates())
    A1, R1 = table_arrays(a, [1], [2], cell_base)
    A1, R1 = A1.reshape(NO, NB, NT)[:, :99].ravel(), R1.reshape(NO, NB, NT)[:, :99].ravel()
    assert all((A1[i], R1[i]) == (t2[c[0]]["x"]["ALL"], t2[c[0]]["x"]["rows"]) for i, c in enumerate(U2E.u2e_candidates()))
    assert M.candidate_name(GRID[0][0], 0, 0, bud[0][0], TAUS_DESC[0]) == "rrf50+none/kind_t/kt:fix_25@1"
    assert decode(0, 12, R) == (0, 0, 0, 0, 0) and decode(12 * R * NO * NB * NT - 1, 12, R) == (11, 2, NO - 1, NB - 1, NT - 1)
    i = int(np.ravel_multi_index((0, 0, NAMES.index("kt_cos"), nm.index("ada_f16_k300"), TAUS_DESC.index(0.85)),
                                 (1, R, NO, NB, NT)))
    _c, r_, o_, b_, t_ = MB.parse("own/kind_t/kt_cos:ada_f16_k300@0.85")
    assert (r_, o_, b_, t_) == (0, NAMES.index("kt_cos"), nm.index("ada_f16_k300"), 0.85)
    assert decode(i, 1, R) == (0, r_, o_, b_, TAUS_DESC.index(t_))
    # 5. the verdicts and what is carried (the incumbent at 0.96 for 1,953.4)
    inc = (0.96, 1953.4)
    assert decide(0.965, None, 0.96, 1953.4, *inc) == ("UNIFIED_RAISED", "KIND_FREE_TOO_DEAR", "NO_GAIN", "C")
    assert decide(0.96, 0.96, 0.96, 1953.4, *inc) == ("UNIFIED_HOLDS", "KIND_FREE_NO_PRICE", "NO_GAIN", "E")
    assert decide(0.96, 0.955, 0.96, 1953.4, *inc) == ("UNIFIED_HOLDS", "KIND_FREE_AT_A_PRICE", "NO_GAIN", "E")
    assert decide(0.955, 0.955, 0.96, 1953.4, *inc) == ("UNIFIED_AT_A_PRICE", "KIND_FREE_NO_PRICE", "NO_GAIN", "E")
    assert decide(0.955, 0.95, 0.96, 1953.4, *inc) == ("UNIFIED_AT_A_PRICE", "KIND_FREE_TOO_DEAR", "NO_GAIN", "C")
    assert decide(0.95, 0.95, 0.96, 1953.4, *inc) == ("NOT_UNIFIED", "KIND_FREE_TOO_DEAR", "NO_GAIN", "U2p")
    assert decide(None, 0.96, 0.96, 1953.4, *inc) == ("NOT_UNIFIED", "KIND_FREE_NO_PRICE", "NO_GAIN", "E")
    assert decide(None, None, 0.965, 1999.0, *inc) == ("NOT_UNIFIED", "KIND_FREE_TOO_DEAR", "OWN_RAISED", "O")
    assert decide(None, None, 0.96, 1900.0, *inc) == ("NOT_UNIFIED", "KIND_FREE_TOO_DEAR", "OWN_CHEAPER", "O")
    assert decide(0.95, 0.955, 0.96, 1953.4, *inc)[1:] == ("KIND_FREE_NO_PRICE", "NO_GAIN", "E")
    assert decide(0.96, 0.95, 0.96, 1953.4, *inc)[1] == "KIND_FREE_TOO_DEAR"
    ceil = np.asarray([0.99, 0.999, 0.98, 0.988, 0.996])
    assert level_of([0.9653, 0.9933, 0.9668, 0.9602, 0.9699], ceil) == 0.96
    assert level_of([0.9653, 0.9933, 0.9668, 0.9499, 0.9699], ceil) is None
    ceil2 = np.asarray([0.99, 0.999, 0.973, 0.988, 0.996])
    assert level_of([0.97] * 5, ceil2) == 0.97
    assert level_of([0.985, 0.994, 0.968, 0.983, 0.991], ceil2) == 1.0
    assert level_of([0.985, 0.994, 0.9679, 0.983, 0.991], ceil2) == 0.965
    # 6. the three frontiers on toy tables, against a brute search (ties to the earlier candidate; the cap on the
    # rounded cost; each frontier only over its recipes)
    for trial in range(40):
        nu_, no2 = int(rng.integers(30, 400)), int(rng.integers(10, 120))
        Au = np.round(rng.uniform(0.93, 1.0, (nu_, 5)), 4)
        Ao = np.round(rng.uniform(0.93, 1.0, (no2, 5)), 4)
        cu = rng.choice(rng.uniform(1200, 2600, 25), nu_)
        co = rng.choice(rng.uniform(1200, 2600, 25), no2)
        cru, cro = np.round(cu, 1), np.round(co, 1)
        gru, oro = rng.integers(0, R, nu_), rng.integers(0, R, no2)
        ce = np.maximum(Au.max(axis=0), rng.uniform(0.95, 1.0, 5)) if trial % 2 else Au.max(axis=0)
        cap_ = float(rng.choice([1953.5, 1500.0, 3000.0]))
        tf = three_frontiers(Au, cu, cru, gru, Ao, co, cro, oro, ce, cap_)
        for key, Am, cm, crm, rows_ in (("C", Au, cu, cru, np.flatnonzero(gru == 0)),
                                        ("E", Au, cu, cru, np.flatnonzero(gru > 0)),
                                        ("O", Ao, co, cro, np.flatnonzero(oro == 0))):
            want_pts, want = [], (None, None)
            for t in LEVELS:
                tg = [round(min(t, float(c) - 0.005), 4) for c in ce]
                k_ = None
                for i in rows_.tolist():
                    if all(Am[i, d] >= tg[d] - 1e-9 for d in range(5)) and (k_ is None or cm[i] < cm[k_]):
                        k_ = i
                want_pts.append(k_)
                if k_ is not None and crm[k_] <= cap_ + 1e-9:
                    want = (t, k_)
            pts, lv, kk = tf[key]
            assert [p[2] for p in pts] == want_pts and (lv, kk) == want, (trial, key)
    # 7. Spearman against scipy's
    from scipy.stats import spearmanr
    for _ in range(50):
        x = rng.integers(0, 12, 60)
        y = x + rng.integers(-5, 6, 60)
        assert abs(spearman(x, y) - round(float(spearmanr(x, y).statistic), 4)) < 1e-12
    assert spearman(np.ones(9), np.arange(9)) is None
    # 8. the shards: any split merges back to the whole; parts out of order are refused
    whole = {"bs": a["bs"], "inc": rng.integers(0, 90, (nq, nk)), "n_i": rng.integers(0, 90, (nq, nk)),
             "fill": a["fill"], "lens": a["lens"], "n_u": rng.integers(0, 90, (nq, nk * R * len(MS))),
             "n_n1": rng.integers(0, 50, (nq, 3)), "n_r0": rng.integers(0, 50, (nq, 3)),
             "n_n2": rng.integers(0, 50, (nq, 9)), "taken": rng.integers(0, 100, (nq, 3)),
             "hubs": rng.integers(0, 3, (nq, 3)), "mq": a["mq"], "nw": rng.integers(0, 900, (nq, 6)),
             "pos": a["pos"], "in_I": rng.random((ng, nk)) < 0.6, "one_link": rng.random((ng, 3)) < 0.3, "ptr": ptr}
    for k in (1, 2, 3, 7, nq):
        parts = []
        for i in range(k):
            q0, q1 = U2C.shard_range(nq, i, k)
            g0, g1 = int(ptr[q0]), int(ptr[q1])
            parts.append({**{x: whole[x][q0:q1] for x in PER_QUESTION}, **{x: whole[x][g0:g1] for x in PER_GOLD},
                          "ptr": ptr[q0:q1 + 1]})
        mg = merge_arrays(parts)
        assert list(mg) == list(ARRAYS) and all(np.array_equal(mg[x], whole[x]) for x in ARRAYS), k
    try:
        merge_arrays([parts[1], parts[0]] + parts[2:])
    except SystemExit:
        pass
    else:
        raise AssertionError("parts out of order merged")
    cells3 = ["rrf50+none", "dense200+none", "dense200+FULL:h1_c25"]
    assert summary_of(mg, cells3) == summary_of(whole, cells3)
    assert set(summary_of(whole, cells3)["mq"]) == {f"{bs}/{w}" for bs, _bp in BASES for w in MASS_WALKS}
    # 9. the per-question lists give the arrays (shapes and types); one dataset's record is written and read back
    pq_ = {x: [whole[x][j] for j in range(nq)] for x in PER_QUESTION}
    pg_ = {x: [whole[x][ptr[j]:ptr[j + 1]] for j in range(nq)] for x in PER_GOLD}
    ar = arrays_of(pq_, pg_, ptr, nk)
    assert list(ar) == list(ARRAYS) and all(np.array_equal(ar[x], whole[x]) and ar[x].dtype == whole[x].dtype
                                            for x in ARRAYS)
    import shutil
    import tempfile
    global OUT
    out0, tmp = OUT, Path(tempfile.mkdtemp(prefix="zu2s_selftest_"))
    try:
        OUT = tmp
        n12 = len(GRID)
        a12 = {"bs": rng.integers(0, 60, (nq, n12)), "inc": rng.integers(0, 90, (nq, n12)),
               "n_i": rng.integers(0, 90, (nq, n12)), "fill": rng.integers(0, 300, (nq, n12 * R)),
               "lens": rng.integers(0, 3000, (nq, n12 * R * NO)).astype(np.int32),
               "n_u": rng.integers(0, 90, (nq, n12 * R * len(MS))), "n_n1": whole["n_n1"], "n_r0": whole["n_r0"],
               "n_n2": whole["n_n2"], "taken": whole["taken"], "hubs": whole["hubs"], "mq": whole["mq"],
               "nw": whole["nw"], "pos": rng.integers(-1, 3000, (ng, n12 * R * NO)).astype(np.int32),
               "in_I": rng.random((ng, n12)) < 0.6, "one_link": whole["one_link"], "ptr": ptr}
        head = {"script_sha256": script_sha256(), "cells": [g[0] for g in GRID], "recipes": list(RECIPES),
                "orders": list(NAMES), "mass_walks": list(MASS_WALKS), "mass_taus": list(MASS_TAUS)}
        write_stats("metaqa", U2C.carve_of("metaqa"), {x: a12[x] for x in ARRAYS}, head,
                    {"mq_bounds": True, "rows_served_verified": False}, 1.0)
        back, rec = load_u2s("metaqa")
        assert all(np.array_equal(back[x], a12[x]) for x in ARRAYS) and rec["questions"] == nq
        assert rec["summary"] == summary_of(a12, head["cells"]) and load_u2s("squad") is None
    finally:
        OUT = out0
        shutil.rmtree(tmp, ignore_errors=True)
    _verdict_e2e(np.random.default_rng(1))
    print("zu2s selftest: OK")


POINTS8 = ("own/kind_t/kt_cos:fix_15000@0.975", "own/kind_t/kt_cos:ada_f12_k0@0.85",
           "own/kind_t/kt_cos:ada_f16_k300@0.85", "own/kind_t/kt_cos:ada_f24_k0@0.95",
           "own/kind_t/kt_cos:ada_f24_k1000@1", "own/kind_t/kt_cos+n2_30:fix_15000@0.95",
           "own/kind_t/kt_cos+n2_100:ada_f3_k1000@0.975", "own/kind_t/kt_cos+n2_100:ada_f12_k0@0.8")


def _verdict_e2e(rng):
    """10. The verdict end to end on synthetic stats for all six, in a temporary OUT: the carried walk, U2m's ptr and
    U2p's filed table are stood in for, the table built from the same arrays, so every check must pass and every
    output must be written; the verdicts must be decide's on the picks' levels."""
    import shutil
    import tempfile
    global OUT
    plans = [MB.parse(p) for p in POINTS8]
    assert all(p[0] == "own" and p[1] == 0 for p in plans)
    n12 = len(GRID)
    out0, tmp = OUT, Path(tempfile.mkdtemp(prefix="zu2s_verdict_"))
    load0, cw0 = M.load_u2m, Q.carried_walk
    try:
        OUT = tmp
        vp_sha = sha_file(P.OUT / "verdict.json")
        pick = f"{POINTS8[2]}|G0|RWe@0.5|LW@0.5"
        arrs, heads = {}, {}
        for d in SIX:
            nq = int(rng.integers(25, 45))
            ng_q = rng.integers(0, 4, nq)
            ptr = np.concatenate([[0], np.cumsum(ng_q)]).astype(np.int64)
            ng = int(ptr[-1])
            qi = np.repeat(np.arange(nq), ng_q)
            own_k = n12 if d in ("metaqa", "hotpotqa") else int(rng.integers(0, n12))
            nk = n12 + int(own_k == n12)
            cells = [g[0] for g in GRID] + (["rrf50+STRUCT:h3_c25_v2000"] if own_k == n12 else [])
            cell_base = [GRID[k][1] if k < n12 else 0 for k in range(nk)]
            nc = nk * R * NO
            lens = rng.integers(0, 3000, (nq, nc)).astype(np.int32)
            pos = np.where(rng.random((ng, nc)) < 0.93, 0, rng.integers(1, 120, (ng, nc)))
            pos = np.where(pos > lens[qi], -1, pos).astype(np.int32)
            mq3 = np.sort(rng.integers(0, 900, (nq, len(BASES) * NW, NTW)), axis=2)
            a = {"bs": rng.integers(1, 60, (nq, nk)), "inc": rng.integers(60, 150, (nq, nk)),
                 "n_i": rng.integers(0, 90, (nq, nk)), "fill": rng.integers(0, 300, (nq, nk * R)), "lens": lens,
                 "n_u": rng.integers(0, 90, (nq, nk * R * len(MS))), "n_n1": rng.integers(0, 50, (nq, len(BASES))),
                 "n_r0": rng.integers(0, 50, (nq, len(BASES))),
                 "n_n2": rng.integers(0, 50, (nq, len(BASES) * len(M_GRID))),
                 "taken": rng.integers(0, 100, (nq, len(BASES))), "hubs": rng.integers(0, 3, (nq, len(BASES))),
                 "mq": mq3.reshape(nq, NMQ), "nw": mq3.max(axis=2) + rng.integers(0, 50, (nq, len(BASES) * NW)),
                 "pos": pos, "in_I": rng.random((ng, nk)) < 0.6, "one_link": rng.random((ng, len(BASES))) < 0.3,
                 "ptr": ptr}
            head = {"script_sha256": script_sha256(), "cells": cells, "own_cell": own_k, "cell_base": cell_base,
                    "recipes": list(RECIPES), "orders": list(NAMES), "mass_walks": list(MASS_WALKS),
                    "mass_taus": list(MASS_TAUS), "pick": pick, "points": list(POINTS8), "G": 0, "beta": 0.5,
                    "verdict_p_sha256": vp_sha}
            write_stats(d, U2C.carve_of(d), {x: a[x] for x in ARRAYS}, head, {"mq_bounds": True}, 1.0)
            arrs[d], heads[d] = a, head
        w = F.weights()
        own = {d: table_arrays(arrs[d], [heads[d]["own_cell"]], range(R), heads[d]["cell_base"]) for d in SIX}

        def idx(p):
            _c, r, o, b, t = MB.parse(p)
            return int(np.ravel_multi_index((0, r, o, b, TAUS_DESC.index(t)), (1, R, NO, NB, NT)))

        every = {f"{p}|G0|RWe@0.5|LW@0.5": {"cost": round(F.cost({d: float(own[d][1][idx(p)]) for d in TRAIN5}, w), 1),
                                            **{d: {"ALL": float(own[d][0][idx(p)]), "rows": float(own[d][1][idx(p)])}
                                               for d in SIX}} for p in POINTS8}
        ceil = np.asarray([table_arrays(arrs[d], range(n12), range(R), heads[d]["cell_base"])[0].max() for d in TRAIN5])
        lv = level_of([own[d][0][idx(POINTS8[2])] for d in TRAIN5], ceil)
        assert lv is not None, "the synthetic incumbent reaches no level"
        vp = {"verdict": "LINKWEIGHT_RAISED", "level": lv, "cost": every[pick]["cost"], "cap": 1953.5,
              "every_candidate": every}
        cw = {"verdict_p": vp, "names": list(POINTS8), "G": 0, "pi": 2, "ci": 1, "beta": 0.5, "pick": pick,
              "a_cfg": "RWe@0.5"}
        M.load_u2m = lambda d_: ({"ptr": arrs[d_]["ptr"]}, {})
        Q.carried_walk = lambda: cw
        assert verdict_stage() == 0
        v = json.loads((OUT / "verdict.json").read_text(encoding="utf-8"))
        assert (OUT / "verdict.md").read_text(encoding="utf-8").startswith(f"# U2s: {v['verdict_c']}")
        pc, pe, po = v["pick_C"], v["pick_E"], v["pick_O"]
        assert po["level"] is not None and po["level"] >= lv - 1e-9 and po["cost"] <= 1953.5
        assert (v["verdict_c"], v["verdict_e"], v["verdict_o"], v["carried"]) == \
            decide(pc["level"], pe["level"], po["level"], po["cost"], lv, every[pick]["cost"])
        assert all(len(v[k]) == len(LEVELS) for k in ("frontier_C", "frontier_E", "frontier_O"))
        assert set(v["price"]) == set(SIX) and set(v["mq"]) == set(SIX) and v["candidates"]["budgets"] == NB
        assert v["u2p_points_reproduced"] == list(POINTS8) and set(v["by_form"]) == set(FORMS)
        for key, p_ in (("C", pc), ("E", pe)):
            if p_["name"] is not None:
                cell, rn = p_["name"].split("/")[:2]
                assert cell in [g[0] for g in GRID] and (rn == "kind_t") == (key == "C")
    finally:
        OUT, M.load_u2m, Q.carried_walk = out0, load0, cw0
        shutil.rmtree(tmp, ignore_errors=True)


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
