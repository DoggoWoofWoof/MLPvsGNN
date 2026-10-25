"""U2m (b): the list-whole cut on (c) and (e)'s carried frontier (docs/U2M_POOLS_THREE_FAMILIES.md, section 11).
Pools only: nothing trains, and no list, restore or size reads a gold.

    python outputs/mp_unified/zu2m_b.py stats DATASET [--threads 2] [--limit Q] [--shard I/K]
    python outputs/mp_unified/zu2m_b.py merge DATASET --parts K
    python outputs/mp_unified/zu2m_b.py verdict
    python outputs/mp_unified/zu2m_b.py --selftest

stats    One dataset's carve (s1sel; webqsp's s1eval, beside). Each question walks once, as zu2m.py's stats walk, and
         the columns that the carried frontier's points read are rebuilt by section 5's path. Each rebuilt column's
         length and every gold's place are checked against U2m's filed arrays. Then, for each point and each G, the
         lists the cut splits are restored (section 11), and the rows and every gold's held are recorded.
         --limit Q: the first Q questions, every check, nothing written and no number logged. --shard I/K: the I-th
         of K contiguous question ranges, into outputs/u2m/b/stats/parts/ (outputs/mp_unified/zu2m_b_modal.py runs
         them on Modal).
merge    Join a dataset's K parts into the file one run over every question writes.
verdict  Section 7's rule over the points x G, the verdict (b) and what is carried forward. Writes
         outputs/u2m/b/verdict_b.json and verdict_b.md.
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

import zu2m as M  # noqa: E402  (U2m's walk, cells, columns, rows, arrays and verdict helpers, unchanged)

U2E, U2C, PG, F, P1H, D5, D5B = M.U2E, M.U2C, M.PG, M.F, M.P1H, M.D5, M.D5B
log, utc, write_json, sha_file = M.log, M.utc, M.write_json, M.sha_file
DECLARED = "docs/U2M_POOLS_THREE_FAMILIES.md, section 11"
OUT = M.OUT / "b"
SIX, TRAIN5, NAMES, TAUS, RECIPES, GRID, BASES, EXPANSIONS = (M.SIX, M.TRAIN5, M.NAMES, M.TAUS, M.RECIPES, M.GRID,
                                                              M.BASES, M.EXPANSIONS)
R = M.R
EMPTY = M.EMPTY
GS = (0, 10, 25, 50, 100, 200, 500)
GMAX = GS[-1]
BIG = np.int64(1) << 40                            # the place of a node outside B and order[:kk]
VERDICTS_B = ("LIST_WHOLE_RAISED", "LIST_WHOLE_CHEAPER", "NO_GAIN")
ARRAYS = ("rows", "held", "ptr")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the carried frontier's points ────────────────────────────────────────────


def parse(name):
    """(cell, recipe index, order index, budget index, tau) of a candidate's name (zu2m.candidate_name)."""
    cell, rn, rest = name.split("/")
    ob, tk = rest.rsplit("@", 1)
    on, bn = ob.split(":")
    taus = {P1H.key(t): t for t in TAUS}
    return cell, RECIPES.index(rn), NAMES.index(on), [b[0] for b in U2E.budgets()].index(bn), taus[tk]


def index_key(p):
    """A parsed point's place in (c) and (e)'s candidate index: cell, recipe, order, budget, tau from 1 down."""
    cell, r, o, b, t = p
    k = -1 if cell == "own" else [g[0] for g in GRID].index(cell)
    return k, r, o, b, sorted(TAUS, reverse=True).index(t)


def carried():
    """(c) and (e)'s filed verdict: (its record, what it carried, the carried frontier's points by level, the distinct
    point names in candidate-index order)."""
    p = M.OUT / "verdict_ce.json"
    if not p.exists():
        raise SystemExit("(c) and (e)'s verdict is not filed: outputs/u2m/verdict_ce.json")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("script_sha256") != M.script_sha256() or rec.get("zu2e_sha256") != U2E.script_sha256():
        raise SystemExit("outputs/u2m/verdict_ce.json is not the committed zu2m.py's verdict")
    car = rec["carried"]
    pts = rec["own_frontiers"]["kind_t"] if car == "U2e" else rec[f"frontier_{car}"]
    names = {x["pick"] for x in pts if x["pick"] is not None}
    if car == "U2e":
        names.add(rec["u2e_pick"]["name"])        # U2e's pick stays: it is a candidate even off its own frontier
    if not names:
        raise SystemExit("the carried frontier has no point")
    return rec, car, pts, sorted(names, key=lambda nm: index_key(parse(nm)))


def filed_values(rec, pts, nm):
    """{dataset: (ALL, rows)} of a carried-frontier point as (c) and (e)'s verdict filed it."""
    for x in pts:
        if x["pick"] == nm:
            return {**{d: (x["s1sel"][d]["ALL"], x["s1sel"][d]["rows"]) for d in TRAIN5},
                    "webqsp": (x["webqsp_s1eval"]["ALL"], x["webqsp_s1eval"]["rows"])}
    if nm == rec["u2e_pick"]["name"]:
        return {d: (p["u2e_pick"]["ALL"], p["u2e_pick"]["rows"]) for d, p in rec["price"].items()}
    raise SystemExit(f"{nm}: not a filed point")


def carried_level_cost(rec):
    car = rec["carried"]
    x = rec["u2e_pick"] if car == "U2e" else rec[f"pick_{car}"]
    return float(x["level"]), float(x["cost"]), (x["name"] if car != "U2e" else rec["u2e_pick"]["name"])


# ── the lists and the restore (no gold) ──────────────────────────────────────


def gather(ip, nodes):
    """(entry indices, each entry's owner index in nodes) of every row entry of nodes under a CSR indptr."""
    nodes = np.asarray(nodes, np.int64)
    if not nodes.size:
        return EMPTY, EMPTY
    a = np.asarray(ip[nodes], np.int64)
    cnt = (np.asarray(ip[nodes + 1], np.int64) - a)
    tot = int(cnt.sum())
    if tot == 0:
        return EMPTY, EMPTY
    own = np.repeat(np.arange(nodes.size, dtype=np.int64), cnt)
    idx = np.repeat(a - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(tot, dtype=np.int64)
    return idx, own


def build_lists(B, order, kkmax, keepmax, untyped, typed, gmax=GMAX):
    """Every list that a point with kk <= kkmax and keep <= keepmax could restore (section 11), as arrays. Places:
    0 in B, p for the order's p-th node, BIG outside B and order[:kkmax]. Per list: A = max(its owner's place, its
    members' smallest place), MX = its members' largest place, S = its size. Per entry (a member placed in the order,
    not in B): its list, place and node. untyped: [(indptr, col)], one list per owner and family; typed: (indptr,
    col, rel, dir) or None, one list per owner, relation and direction. An owner is never a member of its own list."""
    B = np.asarray(B, np.int64)
    order = np.asarray(order, np.int64)
    kkmax, keepmax = min(int(kkmax), order.size), min(int(keepmax), int(kkmax), order.size)
    H = np.concatenate([B, order[:kkmax]])
    pl = np.concatenate([np.zeros(B.size, np.int64), np.arange(1, kkmax + 1, dtype=np.int64)])
    s = np.argsort(H, kind="stable")
    Hs, Ps = H[s], pl[s]

    def place(x):
        if not Hs.size or not x.size:
            return np.full(x.size, BIG, np.int64)
        k = np.minimum(np.searchsorted(Hs, x), Hs.size - 1)
        return np.where(Hs[k] == x, Ps[k], BIG)

    U = np.concatenate([B, order[:keepmax]])
    pu = np.concatenate([np.zeros(B.size, np.int64), np.arange(1, keepmax + 1, dtype=np.int64)])
    el, en, owner = [], [], []
    nl = 0
    for ip, col in untyped:
        cnt = np.asarray(ip[U + 1], np.int64) - np.asarray(ip[U], np.int64)
        sel = np.flatnonzero((cnt > 0) & (cnt <= gmax + 1))          # + 1: a self-link is dropped below
        idx, own = gather(ip, U[sel])
        if idx.size:
            mem = np.asarray(col[idx], np.int64)
            ow = sel[own]
            k = mem != U[ow]
            el.append(nl + ow[k])
            en.append(mem[k])
        owner.append(np.arange(U.size, dtype=np.int64))
        nl += U.size
    if typed is not None:
        tip, tcol, trel, tdir = typed
        idx, own = gather(tip, U)
        if idx.size:
            mem = np.asarray(tcol[idx], np.int64)
            k = mem != U[own]
            idx, own, mem = idx[k], own[k], mem[k]
            key = (own * 65536 + np.asarray(trel[idx], np.int64)) * 4 + np.asarray(tdir[idx], np.int64)
            uk, inv = np.unique(key, return_inverse=True)
            el.append(nl + inv.astype(np.int64))
            en.append(mem)
            owner.append(uk // (65536 * 4))
            nl += uk.size
    out = {"A": EMPTY, "MX": EMPTY, "S": EMPTY, "e_list": EMPTY, "e_place": EMPTY, "e_node": EMPTY}
    if not el or nl == 0:
        return out
    el, en = np.concatenate(el), np.concatenate(en)
    if not el.size:
        return out
    owner = np.concatenate(owner)
    nm = int(en.max()) + 1
    uk = np.unique(el * nm + en)                                    # each member once per list
    el, en = uk // nm, uk % nm
    S = np.bincount(el, minlength=nl).astype(np.int64)
    ep = place(en)
    starts = np.flatnonzero(np.concatenate([[True], el[1:] != el[:-1]]))
    lids = el[starts]
    MN = np.full(nl, BIG, np.int64)
    MX = np.full(nl, BIG, np.int64)
    MN[lids] = np.minimum.reduceat(ep, starts)
    MX[lids] = np.maximum.reduceat(ep, starts)
    A = np.maximum(pu[owner], MN)
    ok = (S > 0) & (S <= gmax) & (MX < BIG) & (MX > 0) & (A <= keepmax)
    new = np.cumsum(ok) - 1
    e = ok[el] & (ep > 0)
    return {"A": A[ok], "MX": MX[ok], "S": S[ok], "e_list": new[el[e]], "e_place": ep[e], "e_node": en[e]}


def restore(Ls, kk, keep, G):
    """The rows restored at (kk, keep, G), sorted: the members in order[keep:kk] of every list of at most G nodes that
    B and order[:kk] hold, with a member in B or order[:keep] and its owner there too."""
    if G <= 0 or not Ls["A"].size:
        return EMPTY
    q = (Ls["A"] <= keep) & (Ls["MX"] <= kk) & (Ls["MX"] > keep) & (Ls["S"] <= G)
    e = q[Ls["e_list"]] & (Ls["e_place"] > keep)
    return np.unique(Ls["e_node"][e])


def restore_brute(B, order, kk, keep, G, untyped, typed):
    """restore, list by list and edge by edge (the self-test's reference)."""
    place = {int(b): 0 for b in B}
    for i, v in enumerate(np.asarray(order)[:kk].tolist()):
        place.setdefault(int(v), i + 1)
    P = [int(b) for b in B] + [int(v) for v in np.asarray(order)[:keep].tolist()]
    out = set()
    for u in P:
        lists = [set(int(x) for x in col[ip[u]:ip[u + 1]]) - {u} for ip, col in untyped]
        if typed is not None:
            tip, tcol, trel, tdir = typed
            groups = {}
            for e in range(int(tip[u]), int(tip[u + 1])):
                if int(tcol[e]) != u:
                    groups.setdefault((int(trel[e]), int(tdir[e])), set()).add(int(tcol[e]))
            lists += list(groups.values())
        for L in lists:
            if not L or len(L) > G or not all(x in place for x in L):
                continue
            if not any(place[x] <= keep for x in L):
                continue
            out |= {x for x in L if place[x] > keep}
    return np.asarray(sorted(out), np.int64)


# ── stats (the laptop, or Modal in shards) ───────────────────────────────────


def point_plan(name, a, rec_u2m, names):
    """Per point on this dataset: (cell index, recipe, order, budget, tau, column, kk per question, keep per
    question)."""
    cells, own_k = rec_u2m["cells"], rec_u2m["own_cell"]
    bud = U2E.budgets()
    plan = []
    for nm in names:
        cell, r, o, b, t = parse(nm)
        k = own_k if cell == "own" else cells.index(cell)
        c = M.column(k, r, o)
        _bn, kind, prm = bud[b]
        kq = U2E.budget_k(kind, prm, a["fill"][:, k * R + r])
        kk = np.minimum(np.asarray(a["lens"][:, c], np.int64), kq)
        keep = P1H.keep_counts(np.asarray(a["bs"][:, k], np.int64), kk, t)
        plan.append({"name": nm, "k": k, "r": r, "o": o, "b": b, "t": t, "c": c, "kk": kk, "keep": keep})
    return plan


def stats_stage(name, threads=2, limit=0, shard=None):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if D.SOURCES[4] != "R0" or F.CONSTRUCTION != "KIND_T":
        raise SystemExit("P1d's fifth source is not R0, or P1f's construction is not KIND_T")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    vrec, car, _pts, names = carried()
    x2m = M.load_u2m(name)
    if x2m is None:
        raise SystemExit(f"{name}: U2m's filed arrays are not under outputs/u2m/stats")
    a2m, r2m = x2m
    if vrec["stats"][name]["file_sha256"] != r2m["file_sha256"]:
        raise SystemExit(f"{name}: U2m's filed arrays are not the ones (c) and (e)'s verdict read")
    plan = point_plan(name, a2m, r2m, names)
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    full_fams = list(op.cfg_h["graph_regimes"]["FULL"])
    if full_fams != list(S4.FAMILIES):
        raise SystemExit("FULL is not the three families in their order")
    cells, own_k = M.cells_of(construction)
    if cells != r2m["cells"] or own_k != r2m["own_cell"]:
        raise SystemExit(f"{name}: the cells are not U2m's")
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
    base_rows = [op.m3b_compile.base_rows({"base_pool": bp}, prep0.dense_ids, prep0.splade_ids, op.m3a,
                                          op.m3b_contract, constant) for _bs, bp in BASES]
    own_b = [bp for _bs, bp in BASES].index(construction["base_pool"])
    inc_own = [np.asarray(p, np.int64) for p in prep.pools]
    linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [EMPTY] * len(prep.seeds)

    def base_of(k):
        return GRID[k][1] if k < len(GRID) else own_b

    need_k = sorted({p["k"] for p in plan})
    need_b = sorted({base_of(k) for k in need_k})
    need_e = sorted({GRID[k][2] for k in need_k if k < len(GRID) and GRID[k][2] > 0})
    need_kr = sorted({(p["k"], p["r"]) for p in plan})
    cache = {}
    stores_all, unions = G.open_unions(cache, op, name, union)
    adj = [(np.ascontiguousarray(stores_all[f].indptr, np.int64), np.ascontiguousarray(stores_all[f].col))
           for f in S4.FAMILIES]
    fams_full = [stores_all[f] for f in full_fams]
    st = stores_all["structural"]
    typed = None
    if g is None and getattr(st, "typed_indptr", None) is not None:
        typed = (np.asarray(st.typed_indptr, np.int64), np.asarray(st.typed_col), np.asarray(st.typed_rel),
                 np.asarray(st.typed_dir))
    untyped = [adj[i] for i, f in enumerate(S4.FAMILIES) if not (typed is not None and f == "structural")]
    list_fams = [f for f in S4.FAMILIES if not (typed is not None and f == "structural")] + \
        (["structural (relation, direction)"] if typed is not None else [])
    log(f"{name}/{carve}: prepared; {len(plan)} points on cells {[cells[k] for k in need_k]}; lists {list_fams} "
        f"({time.time() - t0:.0f}s)")
    n_st = int(unions["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    nq = min(limit, len(prep.seeds)) if limit else len(prep.seeds)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(a2m["ptr"][:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: U2m's golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    NP, NG = len(plan), len(GS)
    rows_out = np.zeros((q1 - q0, NP, NG), np.int64)
    held_out = np.zeros((int(ptr[q1] - ptr[q0]), NP, NG), bool)
    t = time.time()
    done = 0
    for lo in range(q0, q1, S4.BATCH):
        js = list(range(lo, min(q1, lo + S4.BATCH)))
        srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
        steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
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
            per_b = {}
            for bi in need_b:
                B = np.unique(np.union1d(np.asarray(base_rows[bi][j], np.int64), seeds))
                n1, _cnt, _ws, _up, _inv = D5B.n1_table(adj, B, D5.weights_of(w_all, B))
                per_b[bi] = {"B": B, "n1": n1, "base": np.asarray(base_rows[bi][j], np.int64),
                             "r0": F.outside(src[4][0], B)}
            rows.add(np.concatenate([x["n1"] for x in per_b.values()]))
            for x in per_b.values():
                n1 = x["n1"]
                x["cos1"] = D5B.unit(rows.read(n1)) @ q if n1.size else np.empty(0, np.float32)
                x["cr"] = D5B.ranks_desc(x["cos1"], n1) if n1.size else EMPTY
                x["n2"], x["first"], x["taken"], x["hubs"] = U2E.second_link(adj, n1, x["cr"], x["B"])
            exp = {ei: S4.m3b_pools.expand_hops(seeds, fams_full, EXPANSIONS[ei][1]) for ei in need_e}
            inc = {}
            for k in need_k:
                if k < len(GRID):
                    _nm, bi, ei = GRID[k]
                    pool, _added = op.m3b_compile.build_pool(per_b[bi]["base"], seeds, exp.get(ei))
                    inc[k] = np.asarray(pool, np.int64)
                else:
                    inc[k] = inc_own[j]
                B = per_b[base_of(k)]["B"]
                if not np.isin(seeds, B).all() or not np.isin(B, inc[k]).all():
                    raise SystemExit(f"{name}/{carve} question {j}: cell {cells[k]}'s B or I is malformed")
            rows.add(np.concatenate(rec + [src[4][0]] + list(inc.values()) + [x["n2"] for x in per_b.values()]))
            for k, r in need_kr:
                x = per_b[base_of(k)]
                B = x["B"]
                in_b = np.isin(gj, B)
                full = F.outside(rec[r], B)
                ords, _sizes, U, mem = U2E.question_orders(full, x["n1"], x["cr"], x["cos1"], x["r0"],
                                                           F.outside(inc[k], B), x["n2"], x["first"], rows.read, q,
                                                           constant, D.rrf)
                if not U2E.orders_exact(ords, B, full, x["n1"], U, mem):
                    raise SystemExit(f"{name}/{carve} question {j}: an order of {cells[k]}/{RECIPES[r]} is not "
                                     "exactly its sources' nodes")
                for o in sorted({p["o"] for p in plan if (p["k"], p["r"]) == (k, r)}):
                    c = M.column(k, r, o)
                    order = np.asarray(ords[NAMES[o]], np.int64)
                    pos = U2E.places(gj, in_b, order)
                    if order.size != int(a2m["lens"][j, c]) or not np.array_equal(pos, a2m["pos"][g0:g1, c]):
                        raise SystemExit(f"{name}/{carve} question {j}: column {cells[k]}/{RECIPES[r]}/{NAMES[o]} "
                                         "is not U2m's")
                    pis = [pi for pi, p in enumerate(plan) if p["c"] == c]
                    kkmax = max(int(plan[pi]["kk"][j]) for pi in pis)
                    keepmax = max(int(plan[pi]["keep"][j]) for pi in pis)
                    Lt = build_lists(B, order, kkmax, keepmax, untyped, typed)
                    for pi in pis:
                        kk, keep = int(plan[pi]["kk"][j]), int(plan[pi]["keep"][j])
                        base_held = (pos == 0) | ((pos >= 1) & (pos <= keep))
                        prev = None
                        for gi, G_ in enumerate(GS):
                            rest = restore(Lt, kk, keep, G_)
                            if rest.size and not np.isin(rest, order[keep:kk]).all():
                                raise SystemExit(f"{name}/{carve} question {j}: a restored row is outside "
                                                 "order[keep:kk]")
                            if prev is not None and not np.isin(prev, rest).all():
                                raise SystemExit(f"{name}/{carve} question {j}: a larger G restores less")
                            prev = rest
                            rows_out[j - q0, pi, gi] = B.size + keep + rest.size
                            held_out[g0 - int(ptr[q0]):g1 - int(ptr[q0]), pi, gi] = base_held | np.isin(gj, rest)
            done += 1
            if done % 100 == 0:
                log(f"{name}/{carve}: {done}/{q1 - q0} ({time.time() - t:.0f}s)")
    # G = 0 is (c) and (e)'s candidate, question for question
    qr = np.arange(q0, q1)
    sub = {kx: a2m[kx][q0:q1] for kx in ("bs", "fill", "lens")}
    sub["pos"] = a2m["pos"][int(ptr[q0]):int(ptr[q1])]
    sub["ptr"] = ptr[q0:q1 + 1] - ptr[q0]
    for pi, p in enumerate(plan):
        r0, h0 = M.held_of(sub, p["k"], p["r"], p["o"], p["b"], p["t"])
        if not np.array_equal(rows_out[:, pi, 0], r0) or not np.array_equal(held_out[:, pi, 0], h0):
            raise SystemExit(f"{name}/{carve}: G = 0 is not (c) and (e)'s {p['name']}")
        if (rows_out[:, pi, 1:] < rows_out[:, pi, :-1]).any() or (held_out[:, pi, :-1] & ~held_out[:, pi, 1:]).any():
            raise SystemExit(f"{name}/{carve}: {p['name']}: a larger G holds less")
        if (rows_out[:, pi, -1] > np.asarray(sub["bs"][:, p["k"]], np.int64) + p["kk"][qr]).any():
            raise SystemExit(f"{name}/{carve}: {p['name']}: more rows than B and the budget")
    arr = {"rows": rows_out.astype(np.int32), "held": held_out, "ptr": ptr[q0:q1 + 1]}
    head = {"declared_in": DECLARED, "stage": "U2m (b)", "dataset": name, "carve": carve, "graph_detail": desc,
            "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(), "zu2e_sha256": U2E.script_sha256(),
            "u2m_stats_sha256": r2m["file_sha256"], "verdict_ce_sha256": sha_file(M.OUT / "verdict_ce.json"),
            "carried": car, "points": names, "gs": list(GS), "lists": list_fams, "threads": threads,
            "point_cells": {p["name"]: cells[p["k"]] for p in plan}}
    checks = {"graph_equal_u2m": True, "cells_equal_u2m": True, "seeds_equal_g0": True, "kind_t_by_kind": True,
              "cells_well_formed": True, "orders_exact": True, "columns_equal_u2m": True,
              "restored_in_cut_part": True, "monotone_in_G": True, "G0_equal_ce": True,
              "rows_served_verified": bool(limit)}
    if limit:
        log(f"{name}/{carve}: development run on questions {q0}-{q1 - 1} of {nq} ({time.time() - t:.0f}s): "
            "every check passed, every served row verified; nothing written")
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


def summary_of(arr, names):
    """Per point and G: ALL and mean rows, rounded as zu2m's table rounds them."""
    ptr = arr["ptr"] - arr["ptr"][0]
    out = {}
    for pi, nm in enumerate(names):
        out[nm] = {str(G_): {"ALL": U2E.all_of(arr["held"][:, pi, gi], ptr),
                             "rows": round(float(arr["rows"][:, pi, gi].astype(np.float64).mean()), 1)}
                   for gi, G_ in enumerate(GS)}
    return out


def write_stats(name, carve, arr, head, checks, seconds, extra=None):
    od = OUT / "stats"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{name}__{carve}.npz"
    tmp = od / f"{name}__{carve}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, p)
    nq = int(arr["rows"].shape[0])
    rec = {**head, "questions": nq, "golds": int(arr["ptr"][-1]), "file_sha256": sha_file(p),
           "summary": summary_of(arr, head["points"]), "checks": checks, "seconds": seconds, **(extra or {}),
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
                v for c, v in r["checks"].items() if c != "rows_served_verified"):
            raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
        if r["part"] != i or r["parts"] != k or (recs and r["q0"] != recs[-1]["q1"]):
            raise SystemExit(f"part {i}: not the {i}-th of {k} question ranges")
        with np.load(p) as z:
            parts.append({kx: z[kx] for kx in z.files})
        recs.append(r)
    if recs[0]["q0"] != 0 or recs[-1]["q1"] != recs[0]["questions_all"]:
        raise SystemExit(f"{name}/{carve}: the parts do not cover every question")
    ptr = [np.asarray([0], np.int64)]
    for i, p in enumerate(parts):
        q = np.asarray(p["ptr"], np.int64)
        if q.size != p["rows"].shape[0] + 1 or int(q[0]) != int(ptr[-1][-1]):
            raise SystemExit(f"part {i}: its ptr does not continue the previous part's")
        ptr.append(q[1:])
    arr = {"rows": np.concatenate([p["rows"] for p in parts]), "held": np.concatenate([p["held"] for p in parts]),
           "ptr": np.concatenate(ptr)}
    head = {kx: recs[0][kx] for kx in recs[0] if kx not in (
        "part", "parts", "q0", "q1", "questions_all", "file_sha256", "checks", "seconds", "utc")}
    return write_stats(name, carve, arr, head, recs[0]["checks"], round(sum(r["seconds"] for r in recs), 1),
                       {"shards": k, "part_sha256": [r["file_sha256"] for r in recs],
                        "seconds_longest_part": max(r["seconds"] for r in recs)})


# ── the verdict (the laptop) ─────────────────────────────────────────────────


def load_b(name, names):
    carve = U2C.carve_of(name)
    rp = OUT / "stats" / f"{name}__{carve}.json"
    if not rp.exists():
        return None
    rec = json.loads(rp.read_text(encoding="utf-8"))
    p = OUT / "stats" / f"{name}__{carve}.npz"
    if rec.get("script_sha256") != script_sha256() or sha_file(p) != rec["file_sha256"] or not all(
            v for c, v in rec["checks"].items() if c != "rows_served_verified"):
        raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
    if rec["points"] != names or rec["gs"] != list(GS):
        raise SystemExit(f"{rp}: not the carried frontier's points or not section 11's G")
    with np.load(p) as z:
        a = {k: z[k] for k in z.files}
    if summary_of(a, names) != rec["summary"]:
        raise SystemExit(f"{p}: the arrays do not give its summary")
    return a, rec


def decide_b(level, cost, g, car_level, car_cost):
    """Section 11's verdict (b) from the pick's level, cost and G, and the carried pick's level and cost."""
    if level is not None and level > car_level + 1e-9:
        return "LIST_WHOLE_RAISED"
    if level is not None and abs(level - car_level) < 1e-9 and g > 0 and cost < car_cost - 1e-9:
        return "LIST_WHOLE_CHEAPER"
    return "NO_GAIN"


def verdict_stage():
    t0 = time.time()
    vrec, car, pts, names = carried()
    ab, rb, missing = {}, {}, []
    for d in SIX:
        x = load_b(d, names)
        if x is None:
            missing.append(d)
        else:
            ab[d], rb[d] = x
    if missing:
        raise SystemExit(f"INCOMPLETE: no (b) stats for {missing}")
    cap = float(vrec["cap"])
    car_level, car_cost, car_name = carried_level_cost(vrec)
    w = F.weights()
    if w != vrec["weights"]:
        raise SystemExit("P1f's weights are not the ones (c) and (e)'s verdict used")
    ceil = np.asarray([vrec["ceilings"][d] for d in TRAIN5], np.float64)
    NP, NG = len(names), len(GS)
    cands = [(gi, pi) for gi in range(NG) for pi in range(NP)]        # G ascending, then the candidate index

    def cname(gi, pi):
        return f"{names[pi]}|G{GS[gi]}"

    ptrs = {d: ab[d]["ptr"] - ab[d]["ptr"][0] for d in SIX}
    tab = {d: {"ALL": np.asarray([U2E.all_of(ab[d]["held"][:, pi, gi], ptrs[d]) for gi, pi in cands]),
               "rows": np.asarray([round(float(ab[d]["rows"][:, pi, gi].astype(np.float64).mean()), 1)
                                   for gi, pi in cands])} for d in SIX}
    # the check: at G = 0 every point is (c) and (e)'s, as filed
    for pi, nm in enumerate(names):
        i = cands.index((0, pi))
        for d, (al, rw) in filed_values(vrec, pts, nm).items():
            if tab[d]["ALL"][i] != al or tab[d]["rows"][i] != rw:
                raise SystemExit(f"{d}: G = 0 of {nm} is not (c) and (e)'s filed point")
    A = np.stack([tab[d]["ALL"] for d in TRAIN5], axis=1)
    RW = np.stack([tab[d]["rows"] for d in TRAIN5], axis=1)
    costs = np.asarray([F.cost(dict(zip(TRAIN5, r.tolist())), w) for r in RW], np.float64)
    cost_r = np.asarray([round(float(c), 1) for c in costs])
    g0 = np.asarray([gi == 0 for gi, _pi in cands])
    i0 = np.flatnonzero(g0)
    back = U2E.frontier(A[i0], costs[i0], ceil)
    for (t, _tg, k_), x in zip(back, pts):
        want = x["pick"]
        got = None if k_ is None else names[cands[int(i0[k_])][1]]
        if abs(t - x["level"]) > 1e-9 or got != want:
            raise SystemExit(f"level {t}: the frontier over G = 0 is {got}, not the carried frontier's {want}")
    log(f"U2m (b) verdict: G = 0 gives the carried frontier back ({time.time() - t0:.0f}s)")
    fr = U2E.frontier(A, costs, ceil)
    lv, kk = M.pick_of(fr, cost_r, cap)
    if lv is None:
        raise SystemExit("no point within the cap: the carried pick at G = 0 should be one")
    gi_, pi_ = cands[kk]
    vb = decide_b(lv, float(cost_r[kk]), GS[gi_], car_level, car_cost)
    pick = cname(gi_, pi_) if vb != "NO_GAIN" else f"{car_name}|G0"
    if vb == "NO_GAIN" and (gi_ != 0 or names[pi_] != car_name):
        raise SystemExit(f"NO_GAIN, but the frontier's pick is {cname(gi_, pi_)}, not the carried {car_name} at G = 0")

    def point(t, tgt, k_):
        if k_ is None:
            return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": None}
        return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": cname(*cands[k_]),
                "cost": float(cost_r[k_]), "within_cap": bool(cost_r[k_] <= cap + 1e-9),
                "s1sel": {d: {"ALL": float(tab[d]["ALL"][k_]), "rows": float(tab[d]["rows"][k_])} for d in TRAIN5},
                "webqsp_s1eval": {"ALL": float(tab["webqsp"]["ALL"][k_]), "rows": float(tab["webqsp"]["rows"][k_])}}

    # beside: every candidate; the restored rows; the carried pick's misses each G recovers, by section 7's causes
    every = {cname(gi, pi): {"cost": float(cost_r[i]),
                             **{d: {"ALL": float(tab[d]["ALL"][i]), "rows": float(tab[d]["rows"][i])} for d in SIX}}
             for i, (gi, pi) in enumerate(cands)}
    restored = {d: {cname(gi, pi): round(float((ab[d]["rows"][:, pi, gi] - ab[d]["rows"][:, pi, 0]).mean()), 1)
                    for gi, pi in cands if gi > 0} for d in SIX}
    pc = names.index(car_name)
    recovered = {}
    for d in SIX:
        a2m, r2m = M.load_u2m(d)
        cell, r, o, b, t = parse(car_name)
        k = r2m["own_cell"] if cell == "own" else r2m["cells"].index(cell)
        _rw, held1 = M.held_of(a2m, k, r, o, b, 1.0)
        reach = (a2m["pos"] >= 0).any(axis=1)
        h0 = ab[d]["held"][:, pc, 0]
        cause = np.where(held1, "TAU_CUT", np.where(reach, "BUDGET", "UNREACHED"))
        recovered[d] = {"missed": int((~h0).sum()),
                        "by_cause": {c: int((~h0 & (cause == c)).sum()) for c in U2E.CAUSES},
                        "recovered": {str(GS[gi]): {c: int((ab[d]["held"][:, pc, gi] & ~h0 & (cause == c)).sum())
                                                    for c in U2E.CAUSES} for gi in range(1, NG)}}
    rec = {"declared_in": DECLARED, "stage": "U2m (b)", "script_sha256": script_sha256(),
           "zu2m_sha256": M.script_sha256(), "verdict_ce_sha256": sha_file(M.OUT / "verdict_ce.json"),
           "verdict_b": vb, "pick": pick, "level": lv, "cost": float(cost_r[kk]),
           "G": GS[gi_] if vb != "NO_GAIN" else 0,
           "carried_ce": {"what": car, "pick": car_name, "level": car_level, "cost": car_cost}, "cap": cap,
           "points": names, "gs": list(GS), "candidates": len(cands), "weights": w,
           "ceilings": dict(zip(TRAIN5, ceil.tolist())), "frontier": [point(t, tg, k_) for t, tg, k_ in fr],
           "every_candidate": every, "restored_rows_mean": restored, "carried_pick_recovered": recovered,
           "stats": {d: {"file_sha256": rb[d]["file_sha256"], "seconds": rb[d]["seconds"]} for d in SIX},
           "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "verdict_b.json", rec)
    (OUT / "verdict_b.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"U2m (b): {vb}; carried {pick} at level {lv}, cost {cost_r[kk]} ({time.time() - t0:.0f}s)")
    return 0


def report_md(rec):
    c = rec["carried_ce"]
    L = [f"# U2m (b): {rec['verdict_b']}", "",
         f"Declared in {rec['declared_in']}. Carried forward: `{rec['pick']}` at level {rec['level']}, cost "
         f"{rec['cost']} (cap {rec['cap']}).", "",
         f"(c) and (e) carried {c['what']}: `{c['pick']}` at level {c['level']}, cost {c['cost']}. "
         f"{rec['candidates']} candidates: {len(rec['points'])} points of its frontier x G in {rec['gs']}.", "",
         "## The frontier", "", "| level | pick | cost | within the cap |", "|---:|---|---:|---|"]
    for p in rec["frontier"]:
        if p["pick"] is None:
            L.append(f"| {p['level']} | none | | |")
        else:
            L.append(f"| {p['level']} | `{p['pick']}` | {p['cost']} | {p['within_cap']} |")
    L += ["", "## The carried pick's misses that each G recovers", "",
          "| dataset | missed | " + " | ".join(f"G={g}" for g in rec["gs"][1:]) + " |",
          "|---|---:|" + "---:|" * (len(rec["gs"]) - 1)]
    for d, x in rec["carried_pick_recovered"].items():
        L.append(f"| {d} | {x['missed']} | " + " | ".join(str(sum(v.values())) for v in x["recovered"].values()) + " |")
    L += ["", "webqsp's s1eval is beside; it never moves the choice.", ""]
    return "\n".join(L)


# ── self-test ────────────────────────────────────────────────────────────────


def _selftest():
    rng = np.random.default_rng(11)
    for trial in range(60):
        n = int(rng.integers(20, 90))

        def sym(m):
            s = rng.integers(0, n, m)
            d = rng.integers(0, n, m)
            s, d = np.concatenate([s, d]), np.concatenate([d, s])
            o = np.lexsort((d, s))
            s, d = s[o], d[o]
            k = np.ones(s.size, bool)
            k[1:] = (s[1:] != s[:-1]) | (d[1:] != d[:-1])
            s, d = s[k], d[k]
            ip = np.zeros(n + 1, np.int64)
            np.cumsum(np.bincount(s, minlength=n), out=ip[1:])
            return ip, d.astype(np.int32)
        untyped = [sym(int(rng.integers(n, 4 * n))) for _ in range(int(rng.integers(1, 3)))]
        typed = None
        if trial % 2:
            m = int(rng.integers(n, 5 * n))
            s, d = rng.integers(0, n, m), rng.integers(0, n, m)
            rel = rng.integers(0, 4, m)
            heads, tails = np.concatenate([s, d]), np.concatenate([d, s])
            rel2 = np.concatenate([rel, rel])
            dir2 = np.concatenate([np.ones(m, np.int8), np.full(m, 2, np.int8)])
            o = np.lexsort((rel2, tails, heads))
            heads, tails, rel2, dir2 = heads[o], tails[o], rel2[o], dir2[o]
            tip = np.zeros(n + 1, np.int64)
            np.cumsum(np.bincount(heads, minlength=n), out=tip[1:])
            typed = (tip, tails.astype(np.int32), rel2.astype(np.int16), dir2)
        perm = rng.permutation(n)
        nb = int(rng.integers(1, 6))
        B = np.sort(perm[:nb]).astype(np.int64)
        order = perm[nb:nb + int(rng.integers(0, n - nb + 1))].astype(np.int64)
        pts = []
        for _ in range(4):
            kk = int(rng.integers(0, order.size + 1))
            pts.append((kk, int(rng.integers(0, kk + 1))))
        Lt = build_lists(B, order, max(p[0] for p in pts), max(p[1] for p in pts), untyped, typed, gmax=12)
        for kk, keep in pts:
            prev = EMPTY
            for G_ in (0, 1, 3, 6, 12):
                got = restore(Lt, kk, keep, G_)
                want = restore_brute(B, order, kk, keep, G_, untyped, typed)
                assert np.array_equal(got, want), (trial, kk, keep, G_, got, want)
                assert np.isin(prev, got).all() and np.isin(got, order[keep:kk]).all()
                prev = got
    nm = "rrf50+FULL:h2_c25/both/kt_cos+n2_10:ada_f12_k0@0.85"
    cell, r, o, b, t = parse(nm)
    assert M.candidate_name(cell, r, o, U2E.budgets()[b][0], t) == nm
    assert parse("own/kind_t/kt_cos:ada_f12_k0@0.85")[0] == "own"
    assert decide_b(0.96, 1900.0, 25, 0.955, 1800.0) == "LIST_WHOLE_RAISED"
    assert decide_b(0.955, 1700.0, 25, 0.955, 1800.0) == "LIST_WHOLE_CHEAPER"
    assert decide_b(0.955, 1800.0, 0, 0.955, 1800.0) == "NO_GAIN"
    assert decide_b(0.955, 1700.0, 0, 0.955, 1800.0) == "NO_GAIN"
    log("zu2m_b self-test: the restore against a per-list loop (60 toy graphs, typed and untyped), the names, "
        "the verdicts: passed")


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
