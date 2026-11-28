"""BRG1 (docs/BRG1_BRIDGE_TRACE.md): why the budgeted walk does not bridge, and what Amazon's first stage lacks. Read-only diagnostics; no pool, no cell and no rule is changed.

The pool code is PL2 to PL6's own (zpl2g.py to zpl6g.py, zdrp1.py, zdrp2.py, zstk3.py are imported and edited nowhere). The walk is the dense-only, no-K, e3v arm of DRP1 (the SPLADE slot is the
dense list, the K store empty, no name seeds). `trace` re-runs the budget-following walk (zpl5g.expand5) hop by hop with every hop's candidate list kept, asserts that the re-run
equals the engine's expansion and grade on every question (G-B1), and gives each gold node of each question a reason for being in or out of the pool.

    python -E -P -B outputs/mp_unified/zbrg1.py selftest
    python -E -P -B outputs/mp_unified/zbrg1.py trace --ds D | --kb K [--slice N] [--smoke M] [--host]
        # N chunks of the seeded order (3 by default): outputs/brg1/<name>/trace_c<chunk>.npz and gate_trace.json; --smoke M grades M questions of the first chunk, prints the gate only, writes nothing
    python -E -P -B outputs/mp_unified/zbrg1.py read --ds D | --kb K
        # aggregates the chunk files: outputs/brg1/read_<name>.json
    python -E -P -B outputs/mp_unified/zbrg1.py links --ds D | --kb K [--host]
        # gold-to-gold links per family over the whole split: outputs/brg1/links_<name>.json
    python -E -P -B outputs/mp_unified/zbrg1.py lists --kb K [--text] [--host]
        # first-stage lists of one STaRK set (dense, SPLADE, fused) at depth 2,500: outputs/brg1/lists_<kb>.json

Run from the workspace root.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zdrp1 as D1  # noqa: E402
import zdrp2 as D2  # noqa: E402
import zleg1p as L  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl5g as Z5  # noqa: E402
import zpl6g as Z6  # noqa: E402
import zstk3 as T3  # noqa: E402

DOC = "docs/BRG1_BRIDGE_TRACE.md"
OUT = ROOT / "outputs" / "brg1"
SEVEN = D2.DS
KBS = T3.KB_ORDER
# the declared cell of each traced set: the dense-only (`both`) arm's best cell within 2,500 nodes in DRP1's reads (section 16 and STK3's STaRK rows); G-B0 re-reads it
CELL = {"musique": "rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr", "webqsp": "dense_top200+STRUCT:h3_bf_v2000~cs-fu",
        "amazon": "rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr", "prime": "dense_top200+STRUCT:h3_bf_v2000~cs-fu", "mag": "dense_top200+STRUCT:h3_bf_v2000~cs-fu"}
SLICE = 3                                   # chunks of the seeded order (128 questions each)
BFS_CAP = 60_000_000                        # entries gathered at one hop of the uncapped search; above it the search stops and the question is flagged
DEPTHS = (5, 10, 50, 100, 200, 500, 1000, 2500)
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc

R_HEAD, R_EXP, R_NOIN, R_BUD, R_WIN, R_UP, R_NOP = range(7)
REASONS = ("IN_HEAD", "IN_EXP", "NO_IN_EDGE", "BUDGET_CUT", "WINDOW_CUT", "UPSTREAM_CUT", "NO_PATH_IN_H")
COLS = ("q", "ng", "g", "dpos", "reason", "hop", "crank", "cn", "budget", "n_in", "gstate", "pstate", "gdpos", "pdpos", "gany", "dist")
CI = {c: i for i, c in enumerate(COLS)}
QCOLS = ("q", "ng", "present", "size", "seed_gold", "head_gold", "exp_gold", "bfs_cut", "flat", "flat_depth") + tuple(f"{k}{h}" for h in (1, 2, 3) for k in ("fr", "nc", "bu", "ke"))
QI = {c: i for i, c in enumerate(QCOLS)}


# ── pure helpers (self-tested) ───────────────────────────────────────────────

def trace_walk(seeds, fams1, fams2, setting, mods, scorer):
    """zpl5g.expand5 for a budget-following walk, written out so that every hop's candidate list is kept. Returns (expansion, hop records)."""
    key = next(m for m in Z5.KEYS5 if m in mods)
    rr = "rr" in mods
    hops, limit, win = int(setting["hops"]), int(setting["visited_cap"]), int(setting["window"])
    seeds = np.asarray(seeds, dtype=np.int64)
    visited, frontier = np.unique(seeds), seeds
    out, recs = [], []
    for h in range(1, hops + 1):
        rem = limit - int(visited.size)
        if rem <= 0 or frontier.size == 0:
            break
        budget = -(-rem // (hops - h + 1))
        cols, grp, fam, rank = Z5.capped_parts5(frontier, fams1 if h == 1 else fams2, win, scorer)
        if cols.size == 0:
            break
        k = np.isin(cols, visited, invert=True)
        fa = Z5.order_fresh5(cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))
        fresh = fa[:budget]
        recs.append({"h": h, "frontier": frontier, "budget": int(budget), "fa": fa})
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return (np.concatenate(out) if out else np.empty(0, dtype=np.int64)), recs


def in_edges(fam, golds, n):
    """For each node of `golds`: the nodes whose row of `fam` lists it and its place in that row. {gold: (rows, places)}; the family's own storage order."""
    ip, col = np.asarray(fam.indptr), np.asarray(fam.col)
    if col.size == 0:
        return {}
    mark = np.zeros(n, dtype=np.bool_)
    mark[np.asarray(golds, dtype=np.int64)] = True
    hit = np.flatnonzero(mark[col])
    if hit.size == 0:
        return {}
    row = np.searchsorted(ip, hit, side="right") - 1
    pos = hit - ip[row]
    g = col[hit].astype(np.int64)
    o = np.argsort(g, kind="stable")
    g, row, pos = g[o], row[o], pos[o]
    ug, st = np.unique(g, return_index=True)
    en = np.append(st[1:], g.size)
    return {int(a): (row[s:e].astype(np.int64), pos[s:e].astype(np.int64)) for a, s, e in zip(ug, st, en)}


def bfs_dist(seeds, fams1, fams2, hops, buf):
    """Uncapped breadth-first distances (0 = seed) over the hop-wise families, no window and no budget. buf: a uint8 array over the nodes, all 255 on entry.
    Returns (reached nodes, True when a hop gathered more than BFS_CAP entries and the search stopped there). The caller resets buf[reached] to 255."""
    fr = np.unique(np.asarray(seeds, dtype=np.int64))
    buf[fr] = 0
    reached = [fr]
    for h in range(1, hops + 1):
        nxt = []
        for fam in (fams1 if h == 1 else fams2):
            ip, col = fam.indptr, fam.col
            st = ip[fr].astype(np.int64)
            ln = ip[fr + 1].astype(np.int64) - st
            tot = int(ln.sum())
            if tot == 0:
                continue
            if tot > BFS_CAP:
                return np.concatenate(reached), True
            off = np.cumsum(ln) - ln
            idx = np.arange(tot, dtype=np.int64) - np.repeat(off, ln) + np.repeat(st, ln)
            nxt.append(np.asarray(col)[idx].astype(np.int64))
        if not nxt:
            break
        c = np.unique(np.concatenate(nxt))
        c = c[buf[c] == 255]
        if c.size == 0:
            break
        buf[c] = h
        reached.append(c)
        fr = c
    return np.concatenate(reached), False


def pool_state_of(parent_nodes, expanded, head, golds_pool):
    """The best state of the parents that are in the pool: 1 = one of them was expanded (its row was read), 2 = it is in the head only, 3 = it was kept at the last hop (in the
    expansion, not expanded), 0 = no parent in the pool. `golds_pool` restricts the parents (None = any)."""
    p = parent_nodes if golds_pool is None else parent_nodes[np.isin(parent_nodes, golds_pool)]
    if p.size == 0:
        return 0, p
    if np.isin(p, expanded).any():
        return 1, p
    if np.isin(p, head).any():
        return 2, p
    return 3, p


def question_rows(q, golds, lst, b, exp, seeds, recs, ie, names1, names2, win, hops, dist, bfs_cut, flat_list):
    """The gold rows (COLS) and the question row (QCOLS) of one question. lst: the cell's ranked list (P2.Lst); b: its head depth; exp: the walk's expansion; ie: {family: in_edges};
    flat_list: the e3v dense list, whose prefix of the pool's size is the flat pool at the same node count (`flat`: its golds; `flat_depth`: the list's depth)."""
    golds = np.asarray(golds, dtype=np.int64)
    head = lst.L[:b].astype(np.int64)
    pool = np.union1d(head, exp)
    expanded = np.unique(np.concatenate([r["frontier"] for r in recs])) if recs else np.unique(np.asarray(seeds, dtype=np.int64))
    ghead = np.isin(golds, head)
    gexp = np.isin(golds, exp) & ~ghead
    gpool = golds[np.isin(golds, pool)]
    dpos_all = lst.pos(golds)
    rows = []
    allfam = list(dict.fromkeys(list(names1) + list(names2)))
    for i, g in enumerate(golds):
        g = int(g)
        row = {c: -1 for c in COLS}
        row.update(q=q, ng=int(golds.size), g=g, dpos=int(dpos_all[i]) if dpos_all[i] < P2.INF else -1, dist=int(dist[g]) if dist[g] < 255 else 255)
        if ghead[i]:
            row["reason"] = R_HEAD
        elif gexp[i]:
            row["reason"] = R_EXP
        else:
            parents = [ie[f][g] for f in allfam if f in ie and g in ie[f]]
            row["n_in"] = int(sum(p[0].size for p in parents))
            if not parents:
                row["reason"] = R_NOIN
            else:
                cut = None
                for r in recs:
                    w = np.flatnonzero(r["fa"] == g)
                    if w.size and (cut is None or w[0] / r["budget"] < cut[0]):
                        cut = (w[0] / r["budget"], r["h"], int(w[0]), int(r["fa"].size), int(r["budget"]))
                if cut is not None:
                    row.update(reason=R_BUD, hop=cut[1], crank=cut[2], cn=cut[3], budget=cut[4])
                else:
                    wc = False
                    for r in recs:
                        for f in (names1 if r["h"] == 1 else names2):
                            if f in ie and g in ie[f]:
                                srcs, pos = ie[f][g]
                                m = pos >= win
                                if m.any() and np.isin(srcs[m], r["frontier"]).any():
                                    wc = True
                    row["reason"] = R_WIN if wc else (R_UP if dist[g] <= hops else R_NOP)
                psrc = np.unique(np.concatenate([p[0] for p in parents]))
                gs = np.setdiff1d(golds, [g])
                row["gany"] = int(np.isin(psrc, gs).any())
                row["gstate"], gp = pool_state_of(psrc, expanded, head, np.intersect1d(gpool, gs))
                row["pstate"], pp = pool_state_of(psrc, expanded, head, pool)
                if gp.size:
                    d_ = lst.pos(gp)
                    d_ = d_[d_ < P2.INF]
                    row["gdpos"] = int(d_.min()) if d_.size else -1
                d_ = lst.pos(psrc)
                d_ = d_[d_ < P2.INF]
                row["pdpos"] = int(d_.min()) if d_.size else -1
        rows.append([row[c] for c in COLS])
    qr = {c: 0 for c in QCOLS}
    qr.update(q=q, ng=int(golds.size), present=int(np.isin(golds, pool).sum()), size=int(pool.size), seed_gold=int(np.isin(golds, seeds).sum()), head_gold=int(ghead.sum()),
              exp_gold=int(gexp.sum()), bfs_cut=int(bfs_cut), flat=int(np.isin(golds, np.asarray(flat_list)[:pool.size]).sum()), flat_depth=int(np.asarray(flat_list).size))
    for r in recs[:3]:
        h = r["h"]
        qr[f"fr{h}"], qr[f"nc{h}"], qr[f"bu{h}"] = int(r["frontier"].size), int(r["fa"].size), int(r["budget"])
        qr[f"ke{h}"] = int(min(r["budget"], r["fa"].size))
    return np.asarray(rows, dtype=np.int64).reshape(-1, len(COLS)), np.asarray([qr[c] for c in QCOLS], dtype=np.int64)


# ── the walk's inputs ────────────────────────────────────────────────────────

def cell_parts(name):
    kind, ln, b, wk = Z5.parse5(name)
    if kind != "G":
        stop(f"{name}: BRG1 traces union cells (G) only")
    regime, stg, mods = wk
    st = Z5.SET5[stg]
    if not st.get("bf"):
        stop(f"{name}: BRG1 traces budget-following walks only")
    h1, h2 = Z5.REG3[regime]
    return ln, int(b), regime, st, mods, list(h1), list(h2)


def load(a):
    """(name, op, src, chunk order, chunk range, number of chunks, cell). STaRK sets by --kb, the seven by --ds."""
    if a.kb:
        T3.HOSTMODE = bool(a.host)
        op, src = T3.load_stark_source(a.kb)
        nch = (src.nq + T3.CHUNK - 1) // T3.CHUNK
        return a.kb, op, src, T3.chunk_order(nch, KBS.index(a.kb)), T3.chunk_range, nch
    Z6.prepare(bool(a.host))
    cells = D2.cells_of(a.ds)
    op = P2.opened()
    src = P2.load_source(a.ds, "eval", "e3v")
    bad = D2.g_e0(a.ds, src, cells)
    if bad:
        stop(f"{a.ds}: G-E0 failed: {bad}")
    return a.ds, op, src, cells["chunk_order"], L.chunk_range, len(cells["chunk_order"])


def declared_cell_gate(name, kb):
    """G-B0: the declared cell is the dense-only arm's best within 2,500 nodes in the filed read."""
    if kb:
        got = rjson(ROOT / "outputs" / "drp1" / f"read_{name}.json")["arms"]["both"]["best_arm"]["2500"]["name"]
    else:
        got = rjson(ROOT / "outputs" / "drp1e" / f"read_{name}.json")["best_within"]["2500"]["both"]["name"]
    return [] if got == CELL[name] else [f"{name}: the read's best cell is {got}, the declared one {CELL[name]}"]


def states(op, d, plan, lo, hi, limit=None, chunk=64):
    """(q, State5, seeds, query vector) for rows lo..hi: zpl5g.iter_states5's own loop, with the seeds and the question vector kept."""
    M3A = op.m3a
    dvec = d.dvec
    for c0 in range(lo, hi, chunk):
        c1 = min(c0 + chunk, hi)
        dd = np.asarray(d.dense[c0:c1], dtype=np.int64)
        ss = np.asarray(d.splade[c0:c1], dtype=np.int64)
        seeds = M3A.stable_union_rows(dd[:, :5], ss[:, :5])
        none = [None] * (c1 - c0)
        f2 = M3A.rrf_fused_rows(dd[:, :200], ss[:, :200], P2.RRF_CONST) if "F2" in plan.lists else none
        f1 = M3A.rrf_fused_rows(dd, ss, P2.RRF_CONST) if plan.need_f1 else none
        qv = d.qvec(c0, c1)
        for j in range(c1 - c0):
            if limit is not None and c0 + j >= lo + limit:
                return
            yield c0 + j, Z5.pool_state5(plan, dd[j], ss[j], f2[j], f1[j], seeds[j], d.fams, qv[j], dvec, d.n), seeds[j], qv[j]


# ── trace ────────────────────────────────────────────────────────────────────

def trace_range(op, d, plan, cell, lo, hi, limit=None, flat_arr=None):
    ln, b, regime, setting, mods, n1, n2 = cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    win, hops = int(setting["window"]), int(setting["hops"])
    walk = Z5.parse5(cell)[3]
    allfam = list(dict.fromkeys(n1 + n2))
    qs = list(range(lo, hi if limit is None else min(hi, lo + limit)))
    golds_all = np.unique(np.concatenate([d.golds[q] for q in qs])) if qs else np.empty(0, np.int64)
    ie = {f: in_edges(d.fams[f], golds_all, d.n) for f in allfam}
    buf = np.full(d.n, 255, dtype=np.uint8)
    G, Q, bad = [], [], []
    for q, st, seeds, qv in states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        try:
            exp, recs = trace_walk(seeds, fams1, fams2, setting, mods, scorer)
        finally:
            scorer.release()
        eng = st.W[walk].exp
        if not np.array_equal(exp, eng):
            bad.append([q, "expansion", int(exp.size), int(eng.size)])
        P = np.zeros(1, dtype=np.int16)
        S = np.zeros(1, dtype=np.int32)
        Z3.grade3(plan, st, d.golds[q], P, S)
        reached, cut = bfs_dist(seeds, fams1, fams2, hops, buf)
        try:
            g_rows, q_row = question_rows(q, d.golds[q], st.L[ln], b, exp, seeds, recs, ie, n1, n2, win, hops, buf, cut, flat_arr[q])
        finally:
            buf[reached] = 255
        if int(q_row[QI["present"]]) != int(P[0]) or int(q_row[QI["size"]]) != int(S[0]):
            bad.append([q, "grade", [int(q_row[QI["present"]]), int(q_row[QI["size"]])], [int(P[0]), int(S[0])]])
        G.append(g_rows)
        Q.append(q_row)
    return (np.concatenate(G) if G else np.zeros((0, len(COLS)), np.int64)), (np.asarray(Q) if Q else np.zeros((0, len(QCOLS)), np.int64)), bad


def cmd_trace(a):
    name, op, src, order, crange, nch = load(a)
    cell = CELL.get(name)
    if cell is None:
        stop(f"{name}: no declared cell")
    bad0 = declared_cell_gate(name, bool(a.kb))
    if bad0:
        stop(f"G-B0 failed: {bad0}")
    d = D1.variant(src, "both")
    for x in D2.arm_fails(src, d, "both"):
        stop(f"G-D1 failed: {x}")
    plan = Z5.Plan5([cell])
    if a.kb:
        with np.load(T3.STARK / a.kb / "first.npz") as z:
            flat_arr = z["top_dense_e3v"]
        if flat_arr.shape[0] != src.nq or not np.array_equal(flat_arr[:, :src.dense.shape[1]], src.dense):
            stop(f"{name}: first.npz's e3v list is not the source's dense list")
    else:
        flat_arr = np.asarray(d.dense)
    outd = OUT / name
    if a.smoke:
        lo, hi = crange(order[0], src.nq)
        t0 = time.time()
        _G, _Q, bad = trace_range(op, d, plan, cell, lo, hi, limit=int(a.smoke), flat_arr=flat_arr)
        print(f"smoke {name}: {int(a.smoke)} questions, G-B1 mismatches {len(bad)} {bad[:3]}, {time.time() - t0:.1f} s")
        return 0 if not bad else 3
    top = min(int(a.slice or SLICE), nch)
    outd.mkdir(parents=True, exist_ok=True)
    gate_p = outd / "gate_trace.json"
    gates = rjson(gate_p) if gate_p.is_file() else {"stage": "BRG1 gates", "declared_in": DOC, "set": name, "cell": cell, "chunks": {}}
    for pos in range(top):
        c = order[pos]
        path = outd / f"trace_c{c:04d}.npz"
        if path.is_file():
            continue
        lo, hi = crange(c, src.nq)
        t0 = time.time()
        G, Q, bad = trace_range(op, d, plan, cell, lo, hi, flat_arr=flat_arr)
        gates["chunks"][str(c)] = {"questions": hi - lo, "G-B1": {"mismatches": len(bad), "first": bad[:5]}, "seconds": round(time.time() - t0, 1)}
        gates["utc"], gates["verdict"] = utc(), ("PASS" if all(v["G-B1"]["mismatches"] == 0 for v in gates["chunks"].values()) else "FAIL")
        wjson(gate_p, gates)
        if bad:
            stop(f"{name}: chunk {c}: G-B1 failed ({len(bad)} mismatches, first {bad[:3]})")
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as f:
            np.savez(f, gold=G, quest=Q, cols=np.asarray(COLS), qcols=np.asarray(QCOLS), cell=np.asarray(cell), lo=lo, hi=hi)
        os.replace(tmp, path)
        log(f"{name}: chunk {c} (questions {lo}-{hi}): G-B1 PASS, {G.shape[0]} golds, {time.time() - t0:.0f} s ({P2.peak_gb()} GB)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────────

def share(k, n):
    return {"n": int(k), "of": int(n), "share": (k / n) if n else None, "wilson": L.wilson(int(k), int(n)) if n else None}


def quant(x, qs=(0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, dtype=np.float64)
    return {str(q): float(np.quantile(x, q)) for q in qs} if x.size else {}


def aggregate(G, Q, win):
    """Every number of the read from the gold rows and the question rows."""
    g = {c: G[:, i] for c, i in CI.items()}
    out = {"questions": int(Q.shape[0]), "golds": int(G.shape[0])}
    nq = Q.shape[0]
    out["ALL"] = share(int((Q[:, QI["present"]] == Q[:, QI["ng"]]).sum()), nq)
    out["mean_nodes"] = float(Q[:, QI["size"]].mean())
    out["by_gold_count"] = {}
    for lab, m in (("1", Q[:, QI["ng"]] == 1), ("2", Q[:, QI["ng"]] == 2), ("3", Q[:, QI["ng"]] == 3), ("4+", Q[:, QI["ng"]] >= 4)):
        if m.any():
            out["by_gold_count"][lab] = {"questions": int(m.sum()), "ALL": share(int((Q[m, QI["present"]] == Q[m, QI["ng"]]).sum()), int(m.sum())),
                                         "mean_nodes": float(Q[m, QI["size"]].mean())}
    rs = g["reason"]
    out["golds_by_reason"] = {REASONS[r]: share(int((rs == r).sum()), len(rs)) for r in range(7)}
    miss = G[rs >= R_NOIN]
    gm = {c: miss[:, i] for c, i in CI.items()}
    nm = len(miss)
    out["missed"] = nm
    out["missed_by_reason"] = {REASONS[r]: share(int((gm["reason"] == r).sum()), nm) for r in range(R_NOIN, 7)}
    out["missed_dpos_in_list"] = share(int((gm["dpos"] >= 0).sum()), nm)
    out["missed_dpos_quantiles"] = quant(gm["dpos"][gm["dpos"] >= 0])
    hasq = {int(q) for q in Q[Q[:, QI["present"]] > 0, QI["q"]]}
    in_bridge = np.asarray([int(q) in hasq for q in gm["q"]], dtype=bool)
    out["missed_in_questions_with_a_gold_in_pool"] = share(int(in_bridge.sum()), nm)
    # the bridge: a missed gold with an edge from another gold of its question
    out["missed_with_gold_parent"] = share(int((gm["gany"] == 1).sum()), nm)
    out["gold_parent_state_of_missed"] = {s: share(int((gm["gstate"] == k).sum()), nm) for k, s in ((1, "parent expanded"), (2, "parent in head only"), (3, "parent kept at last hop"), (0, "none in pool"))}
    out["any_parent_state_of_missed"] = {s: share(int((gm["pstate"] == k).sum()), nm) for k, s in ((1, "parent expanded"), (2, "parent in head only"), (3, "parent kept at last hop"), (0, "none in pool"))}
    hp = gm["gstate"] == 2
    out["head_only_gold_parent_depth"] = {"quantiles": quant(gm["gdpos"][hp & (gm["gdpos"] >= 0)]),
                                          **{f"within_{s}": share(int((hp & (gm["gdpos"] >= 0) & (gm["gdpos"] < s)).sum()), int(hp.sum())) for s in (10, 20, 50, 100)}}
    withp = gm["n_in"] > 0
    out["missed_with_any_in_edge"] = share(int(withp.sum()), nm)
    out["parent_in_dense_top_s"] = {f"s{s}": share(int((withp & (gm["pdpos"] >= 0) & (gm["pdpos"] < s)).sum()), int(withp.sum())) for s in (5, 10, 20, 50, 100, 200)}
    bm = gm["reason"] == R_BUD
    ratio = gm["crank"][bm] / np.maximum(gm["budget"][bm], 1)
    out["budget_cut"] = {"n": int(bm.sum()), "rank_over_budget_quantiles": quant(ratio), "hop_counts": {str(h): int((gm["hop"][bm] == h).sum()) for h in (1, 2, 3)},
                         **{f"within_{x}x": share(int((ratio <= x).sum()), int(bm.sum())) for x in (2, 4, 10)}, "candidates_quantiles": quant(gm["cn"][bm]), "budget_quantiles": quant(gm["budget"][bm])}
    out["window_cut"] = {"n": int((gm["reason"] == R_WIN).sum()), "window": int(win)}
    out["dist_of_missed"] = {("none" if k == 255 else str(k)): int((gm["dist"] == k).sum()) for k in (1, 2, 3, 255)}
    # sole blockers: incomplete questions whose missed golds all have one reason
    sole = {r: 0 for r in REASONS[R_NOIN:]}
    inc = 0
    for q in np.unique(miss[:, CI["q"]]):
        rr = np.unique(miss[miss[:, CI["q"]] == q, CI["reason"]])
        inc += 1
        if rr.size == 1:
            sole[REASONS[int(rr[0])]] += 1
    out["incomplete_questions"] = inc
    out["sole_blocker"] = {k: share(v, inc) for k, v in sole.items()}
    out["seed_has_a_gold"] = share(int((Q[:, QI["seed_gold"]] > 0).sum()), nq)
    out["hops"] = {str(h): {k: float(Q[:, QI[f"{k}{h}"]].mean()) for k in ("fr", "nc", "bu", "ke")} for h in (1, 2, 3)}
    out["bfs_cut_questions"] = int(Q[:, QI["bfs_cut"]].sum())
    uncapped = Q[:, QI["flat_depth"]] >= Q[:, QI["size"]]
    if uncapped.all():
        pa = (Q[:, QI["present"]] == Q[:, QI["ng"]]).astype(np.float64)
        fa_ = (Q[:, QI["flat"]] == Q[:, QI["ng"]]).astype(np.float64)
        rng = np.random.default_rng(20261128)
        bs = np.asarray([(pa - fa_)[rng.integers(0, nq, nq)].mean() for _ in range(2000)])
        out["walk_vs_flat"] = {"pool_ALL": float(pa.mean()), "flat_ALL": float(fa_.mean()), "pool_minus_flat": float((pa - fa_).mean()),
                               "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                               "golds_pool": int(Q[:, QI["present"]].sum()), "golds_flat": int(Q[:, QI["flat"]].sum()),
                               "by_gold_count": {lab: {"questions": int(m.sum()), "pool_ALL": float(pa[m].mean()), "flat_ALL": float(fa_[m].mean())}
                                                 for lab, m in (("1", Q[:, QI["ng"]] == 1), ("2", Q[:, QI["ng"]] == 2), ("3", Q[:, QI["ng"]] == 3), ("4+", Q[:, QI["ng"]] >= 4)) if m.any()}}
    else:
        out["walk_vs_flat"] = {"capped": int((~uncapped).sum()), "note": "the dense list is shallower than the pool for some questions: no flat comparison"}
    return out


def cmd_read(a):
    name = a.kb or a.ds
    folder = OUT / name
    gate = rjson(folder / "gate_trace.json") if (folder / "gate_trace.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        stop(f"{name}: no PASS gate_trace.json: no number is read")
    files = sorted(folder.glob("trace_c*.npz"))
    if not files:
        stop(f"{name}: no chunk file")
    Gs, Qs = [], []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["cols"]] != list(COLS) or [str(x) for x in z["qcols"]] != list(QCOLS) or str(z["cell"]) != CELL[name]:
                stop(f"{p.name}: not this module's columns or cell")
            Gs.append(z["gold"])
            Qs.append(z["quest"])
        finally:
            z.close()
    G, Q = np.concatenate(Gs), np.concatenate(Qs)
    win = int(cell_parts(CELL[name])[3]["window"])
    rec = {"stage": "BRG1 read", "declared_in": DOC, "set": name, "cell": CELL[name], "utc": utc(), "chunks": [p.name for p in files], **aggregate(G, Q, win)}
    wjson(OUT / f"read_{name}.json", rec)
    log(f"{name}: read written ({rec['questions']} questions, {rec['golds']} golds, ALL {rec['ALL']['share']:.3f} at {rec['mean_nodes']:.0f} nodes)")
    return 0


# ── gold-to-gold links ───────────────────────────────────────────────────────

def components(n, edges):
    """Connected components of n items under undirected edges [(i, j)]: the number of components."""
    par = list(range(n))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for i, j in edges:
        par[f(i)] = f(j)
    return len({f(i) for i in range(n)})


def link_stats(golds_by_q, ie_by_family, win):
    """Per family set: the share of golds (of questions with two or more) with a linked partner gold, the share of questions whose golds form one component, the places of the
    linking edges. ie_by_family: {family: in_edges over all golds}."""
    sets = {"S": ["structural"], "R": ["ner"], "K": ["knn"], "S+R": ["structural", "ner"], "S+R+K": ["structural", "ner", "knn"]}
    out = {}
    for lab, fl in sets.items():
        if any(f not in ie_by_family for f in fl):
            continue
        tot_g = lin_g = tot_q = con_q = 0
        pl = []
        by = {}
        for q, gs in golds_by_q:
            if gs.size < 2:
                continue
            idx = {int(x): i for i, x in enumerate(gs)}
            edges, linked = set(), set()
            for f in fl:
                for b in gs:
                    e = ie_by_family[f].get(int(b))
                    if e is None:
                        continue
                    srcs, pos = e
                    for s_, p_ in zip(srcs, pos):
                        if int(s_) in idx and int(s_) != int(b):
                            edges.add((idx[int(s_)], idx[int(b)]))
                            linked |= {idx[int(s_)], idx[int(b)]}
                            pl.append(int(p_))
            tot_g += gs.size
            lin_g += len(linked)
            tot_q += 1
            conn = components(gs.size, edges) == 1
            con_q += int(conn)
            k = "2" if gs.size == 2 else ("3" if gs.size == 3 else "4+")
            b_ = by.setdefault(k, [0, 0])
            b_[0] += 1
            b_[1] += int(conn)
        out[lab] = {"golds_with_a_linked_gold": share(lin_g, tot_g), "questions_one_component": share(con_q, tot_q),
                    "by_gold_count": {k: share(v[1], v[0]) for k, v in sorted(by.items())}, "edges": len(pl),
                    "edge_place_ge_window": share(int((np.asarray(pl) >= win).sum()), len(pl)) if pl else None}
    return out


def cmd_links(a):
    name, op, src, _o, _r, _n = load(a)
    ds_golds = [(q, np.asarray(src.golds[q], dtype=np.int64)) for q in range(src.nq)]
    multi = [(q, g) for q, g in ds_golds if g.size >= 2]
    allg = np.unique(np.concatenate([g for _q, g in multi])) if multi else np.empty(0, np.int64)
    ie = {}
    for f in ("structural", "ner", "knn"):
        fam = src.fams[f]
        ie[f] = in_edges(fam, allg, src.n) if np.asarray(fam.col).size else {}
    out = {"stage": "BRG1 links", "declared_in": DOC, "set": name, "utc": utc(), "questions": src.nq, "questions_with_2_or_more_golds": len(multi), "window": int(Z5.WINDOW),
           "gold_count": {str(int(k)): int(v) for k, v in zip(*np.unique(np.minimum([g.size for _q, g in ds_golds], 4), return_counts=True))},
           "links": link_stats(multi, ie, int(Z5.WINDOW))}
    OUT.mkdir(parents=True, exist_ok=True)
    wjson(OUT / f"links_{name}.json", out)
    log(f"{name}: links written ({len(multi)} questions with two or more golds)")
    return 0


# ── Amazon's first stage ─────────────────────────────────────────────────────

TOK = re.compile(r"[a-z0-9]+")


def recall_table(pos, gold_q, ng):
    """pos: (golds,) place of each gold in one list (INF when absent); gold_q: its question index; ng: golds per question. Gold-level recall and question-level ALL at the depths."""
    t = {}
    nq = ng.size
    for k in DEPTHS:
        hit = pos < k
        per_q = np.bincount(gold_q, weights=hit.astype(np.float64), minlength=nq)
        t[str(k)] = {"gold_recall": float(hit.mean()), "ALL": float((per_q == ng).mean()),
                     "ALL_by_gold_count": {lab: float((per_q[m] == ng[m]).mean()) for lab, m in (("1", ng == 1), ("2", ng == 2), ("3", ng == 3), ("4+", ng >= 4)) if m.any()}}
    return t


def cmd_lists(a):
    import zstk2 as K2
    import zworld as W
    kb = a.kb
    T3.HOSTMODE = bool(a.host)
    op, src = T3.load_stark_source(kb)
    with np.load(T3.STARK / kb / "first.npz") as z:
        dense = z["top_dense_e3v"].astype(np.int64)
        splade = z["top_splade"].astype(np.int64)
    nq = src.nq
    M3A = op.m3a
    fused = []
    for c0 in range(0, nq, 64):
        for r in M3A.rrf_fused_rows(dense[c0:c0 + 64], splade[c0:c0 + 64], P2.RRF_CONST):
            fused.append(np.asarray(r, dtype=np.int64)[:dense.shape[1]])
    ng = np.asarray([g.size for g in src.golds], dtype=np.int64)
    gq = np.repeat(np.arange(nq), ng)
    gn = np.concatenate([np.asarray(g, dtype=np.int64) for g in src.golds])
    pos = {}
    for lab, lists in (("dense", [dense[i] for i in range(nq)]), ("splade", [splade[i] for i in range(nq)]), ("fused", fused)):
        pos[lab] = np.concatenate([P2.Lst(lists[i]).pos(src.golds[i]) for i in range(nq)])
    out = {"stage": "BRG1 lists", "declared_in": DOC, "kb": kb, "utc": utc(), "questions": nq, "golds": int(gn.size), "list_depth": int(dense.shape[1]),
           "gold_count": {str(int(k)): int(v) for k, v in zip(*np.unique(np.minimum(ng, 4), return_counts=True))},
           "tables": {lab: recall_table(p, gq, ng) for lab, p in pos.items()}}
    k = 1000
    pd, ps = pos["dense"], pos["splade"]
    out["overlap_at_1000"] = {"both": int(((pd < k) & (ps < k)).sum()), "dense_only": int(((pd < k) & (ps >= k)).sum()), "splade_only": int(((pd >= k) & (ps < k)).sum()),
                              "neither": int(((pd >= k) & (ps >= k)).sum())}
    k = int(dense.shape[1])
    out["overlap_at_depth"] = {"depth": k, "both": int(((pd < k) & (ps < k)).sum()), "dense_only": int(((pd < k) & (ps >= k)).sum()), "splade_only": int(((pd >= k) & (ps < k)).sum()),
                               "neither": int(((pd >= k) & (ps >= k)).sum())}
    out["seed_quality"] = {"dense_top5_has_a_gold": float(np.mean([np.isin(src.golds[i], dense[i, :5]).any() for i in range(nq)])),
                           "dense_top10_has_a_gold": float(np.mean([np.isin(src.golds[i], dense[i, :10]).any() for i in range(nq)])),
                           "splade_top5_has_a_gold": float(np.mean([np.isin(src.golds[i], splade[i, :5]).any() for i in range(nq)])),
                           "fused_top5_has_a_gold": float(np.mean([np.isin(src.golds[i], fused[i][:5]).any() for i in range(nq)]))}
    out["dense_place_quantiles_of_golds_found"] = quant(pd[pd < dense.shape[1]])
    if a.text:
        w = W.load(K2.KBS[kb])
        qtxt = {int(q["row"]): q["question"] for q in w.iter_queries(splits=["test"])}
        rows = [int(r) for r in src.idx]
        need = {int(x) for x in gn}
        tl, toks = {}, {}
        for i, (nid, text) in enumerate(w.iter_nodes()):
            if i in need:
                tl[i] = len(text)
                toks[i] = set(TOK.findall(text.lower()))
        grp = {"dense_top100": (pd < 100), "dense_found_100_to_1000": ((pd >= 100) & (pd < 1000)), "splade_only_at_1000": ((pd >= 1000) & (ps < 1000)), "neither_at_1000": ((pd >= 1000) & (ps >= 1000))}
        feats = {}
        for lab, m in grp.items():
            ii = np.flatnonzero(m)
            lens = np.asarray([tl[int(gn[i])] for i in ii], dtype=np.float64)
            ov = []
            for i in ii:
                qt = {t for t in TOK.findall(qtxt[rows[int(gq[i])]].lower()) if len(t) >= 4}
                ov.append(len(qt & toks[int(gn[i])]) / len(qt) if qt else 0.0)
            feats[lab] = {"golds": int(ii.size), "text_chars_quantiles": quant(lens), "question_terms_in_text_mean": float(np.mean(ov)) if ov else None,
                          "question_terms_in_text_quantiles": quant(ov)}
        out["text_features"] = feats
    OUT.mkdir(parents=True, exist_ok=True)
    wjson(OUT / f"lists_{kb}.json", out)
    log(f"{kb}: lists written ({nq} questions)")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

def _toy(rng, n, fams=2, deg=6):
    out = []
    for _ in range(fams):
        ip, col = [0], []
        for _v in range(n):
            k = int(rng.integers(0, deg + 1))
            col += [int(x) for x in rng.integers(0, n, size=k)]
            ip.append(len(col))
        out.append(P2.CSR(np.asarray(ip, dtype=np.int64), np.asarray(col, dtype=np.int32)))
    return out


def cmd_selftest(_a):
    Z6.prepare(False)
    rng = np.random.default_rng(7)
    n, dim = 400, 8
    V = rng.normal(size=(n, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    fams = _toy(rng, n)
    ok = 0
    for trial in range(30):
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        seeds = rng.choice(n, size=5, replace=False)
        for stg, mods in (("h2_bf_v1000", ("cs", "vr")), ("h3_bf_v1000", ("cs", "fu")), ("h2_bf_v1000", ("cs", "vr", "rr")), ("h3_bf_v2000", ("cs", "vr"))):
            st = dict(Z5.SET5[stg])
            st["visited_cap"] = 60 if trial % 2 else int(st["visited_cap"])        # a small budget makes the cut bite
            sc1 = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
            e1 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc1)
            sc1.release()
            sc2 = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
            e2, recs = trace_walk(seeds, [fams[0]], fams, st, mods, sc2)
            sc2.release()
            assert np.array_equal(e1, e2), (trial, stg, mods)
            ok += 1
    # in_edges and bfs_dist against brute force
    G = np.asarray([3, 17, 99, 250])
    ie = in_edges(fams[0], G, n)
    for g in G:
        want = [(r, p) for r in range(n) for p, c in enumerate(fams[0].col[fams[0].indptr[r]:fams[0].indptr[r + 1]]) if c == g]
        got = list(zip(ie[int(g)][0], ie[int(g)][1])) if int(g) in ie else []
        assert sorted(want) == sorted((int(a), int(b)) for a, b in got), g
    buf = np.full(n, 255, dtype=np.uint8)
    seeds = np.asarray([1, 2, 3])
    reached, cut = bfs_dist(seeds, [fams[0]], fams, 3, buf)
    ref = {int(s): 0 for s in seeds}
    fr = set(int(s) for s in seeds)
    for h in (1, 2, 3):
        nxt = set()
        for v in fr:
            for f in ([fams[0]] if h == 1 else fams):
                nxt |= {int(c) for c in f.col[f.indptr[v]:f.indptr[v + 1]]}
        nxt = {v for v in nxt if v not in ref}
        ref.update({v: h for v in nxt})
        fr = nxt
    assert not cut and {int(v): int(buf[v]) for v in reached} == ref
    buf[reached] = 255
    assert (buf == 255).all()
    # question_rows: reasons are consistent with the trace on random questions, in three toy graphs (dense, sparse, long rows with a narrow window)
    seen = set()
    for deg, win_, nfam_seed in ((6, 512, 11), (2, 512, 12), (14, 3, 13)):
        r2 = np.random.default_rng(nfam_seed)
        fm = _toy(r2, n, deg=deg)
        for trial in range(60):
            q = r2.normal(size=dim)
            q /= np.linalg.norm(q)
            order = np.argsort(-(V @ q), kind="stable")
            seeds = order[:5]
            lst = P2.Lst(order.astype(np.int64))
            golds = np.sort(r2.choice(n, size=int(r2.integers(1, 5)), replace=False))
            st = dict(Z5.SET5["h2_bf_v1000"])
            st["visited_cap"], st["window"] = 40, win_
            sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
            exp, recs = trace_walk(seeds, [fm[0]], fm, st, ("cs", "vr"), sc)
            sc.release()
            ie = {"f0": in_edges(fm[0], golds, n), "f1": in_edges(fm[1], golds, n)}
            buf = np.full(n, 255, dtype=np.uint8)
            reached, cut = bfs_dist(seeds, [fm[0]], fm, 2, buf)
            gr, qr = question_rows(trial, golds, lst, 20, exp, seeds, recs, {"structural": ie["f0"], "ner": ie["f1"]}, ["structural"], ["structural", "ner"], win_, 2, buf, cut, order)
            buf[reached] = 255
            pool = np.union1d(order[:20], exp)
            assert int(qr[QI["flat"]]) == int(np.isin(golds, order[:pool.size]).sum()) and int(qr[QI["flat_depth"]]) == n
            assert int(qr[QI["present"]]) == int(np.isin(golds, pool).sum()) and int(qr[QI["size"]]) == pool.size
            for row in gr:
                r = {c: int(row[i]) for c, i in CI.items()}
                g = r["g"]
                seen.add(r["reason"])
                assert (r["reason"] <= R_EXP) == bool(np.isin(g, pool)), r
                if r["reason"] == R_NOIN:
                    assert g not in ie["f0"] and g not in ie["f1"], r
                if r["reason"] == R_BUD:
                    h = recs[r["hop"] - 1]
                    assert h["fa"][r["crank"]] == g and r["crank"] >= h["budget"], r
                if r["reason"] == R_WIN:
                    ok_ = False
                    for rc in recs:
                        for nm_, f_ in (("structural", ie["f0"]), ("ner", ie["f1"])):
                            if (rc["h"] == 1 and nm_ != "structural") or g not in f_:
                                continue
                            srcs, pos = f_[g]
                            ok_ |= bool(((pos >= win_) & np.isin(srcs, rc["frontier"])).any())
                    assert ok_, r
                    assert not any((rc["fa"] == g).any() for rc in recs), r
                if r["reason"] == R_NOP:
                    assert r["dist"] == 255, r
                if r["reason"] == R_UP:
                    assert r["dist"] <= 2, r
    assert {R_NOIN, R_BUD, R_WIN, R_UP, R_NOP} <= seen, f"the toy graphs did not exercise every reason: {sorted(seen)}"
    # components and recall_table
    assert components(4, [(0, 1), (2, 3)]) == 2 and components(3, [(0, 1), (1, 2)]) == 1
    t = recall_table(np.asarray([0, 6, 1, P2.INF]), np.asarray([0, 0, 1, 2]), np.asarray([2, 1, 1]))
    assert abs(t["5"]["gold_recall"] - 0.5) < 1e-9 and abs(t["10"]["ALL"] - 2 / 3) < 1e-9
    print(f"zbrg1 selftest: expansions equal on {ok} walks, in_edges and bfs_dist match brute force, reasons seen {sorted(seen)}: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("selftest", "trace", "read", "links", "lists"):
        p = sub.add_parser(nm)
        if nm != "selftest":
            p.add_argument("--ds", choices=SEVEN)
            p.add_argument("--kb", choices=KBS)
            p.add_argument("--host", action="store_true")
        if nm == "trace":
            p.add_argument("--slice", type=int)
            p.add_argument("--smoke", type=int)
        if nm == "lists":
            p.add_argument("--text", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd != "selftest" and bool(getattr(a, "ds", None)) == bool(getattr(a, "kb", None)):
        stop("give exactly one of --ds and --kb")
    return {"selftest": cmd_selftest, "trace": cmd_trace, "read": cmd_read, "links": cmd_links, "lists": cmd_lists}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
