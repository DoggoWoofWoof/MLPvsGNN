"""SEM2 R8, Part A: the room (docs/SEM2_EDGE_OPERATORS.md section 16, declared 26 Nov 2026 before any R8 number).

    python outputs/mp_unified/zsem2r8.py selftest
    python outputs/mp_unified/zsem2r8.py reach --ds D [--host] [--threads 8] [--smoke]     (the host: the first stage reads the e3v vector shards)
    python outputs/mp_unified/zsem2r8.py report                                              (laptop: the calls, from the records)

A wrapper over zsem2 (R1: the graph, the plan), zsem2t (R2a: the questions and the first stage) and zsem1 (the RRF); it edits none.
Part A reads no scorer and fits nothing. For each question of the selection carve:
- F-list = the first F = 1,000 nodes of the base list RRF(v, SPLADE) (SEM1's, C = 60); seeds = its first 25.
- the walk follows the stored links R1 gave an operator (R both ways, S in its stored direction): hop 1 A = N+(seeds) minus the seeds; hop 2 D =
  N+(the hop-1 nodes that are in the pool). The pool is the F-list plus the nodes a program adds; cost = the number of added nodes.
- a missed gold (not in the F-list) is class 1 (in A), class 2 (in D only, reached through a hop-1 node that is a parent) or beyond (neither).
- Pi_oracle: minimise the added nodes subject to every reachable missed gold being added and the closure (a hop-2 node needs a parent in the
  pool); exact by HiGHS branch and bound (scipy.optimize.milp, relative gap 0, time limit 60 s); the LP relaxation beside. A question with a
  gold beyond the walk is infeasible. H(M') = 100 x (completable at M') / questions.
Outputs under outputs/sem2/r8/<ds>/ (and outputs/sem2/r8/smoke/<ds>/ for --smoke); the report under outputs/sem2/r8/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2 as Z  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

E = Z.E
log, stop = Z.log, Z.stop
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 16"
OUT = Z.ROOT / "outputs" / "sem2" / "r8"
SEVEN = Z.SEVEN
F_DEPTH = 1000                 # 16.3: the base list's depth, the pool's first-stage part
B_SEEDS = 25                   # 16.3: the walk's seeds
LADDER = (25, 100, 500, 1500)  # 16.3: added-node budgets; the main read is the last
MAIN = 1500
ROOM_POINTS = 1.0              # 16.5
TIME_LIMIT = 60.0              # 16.4
OPT_SHARE = 0.995              # G-P: proven-optimal share
BOOT, SEED = 2000, 0
STATUS = {"none": 0, "closed": 1, "optimal": 2, "not_optimal": 3, "infeasible": 4}


def script_sha():
    return Z.sha_file(Path(__file__))


# -- the closure-knapsack, exact (16.4) --------------------------------------


def solve_closure(w, free, parents, budget=None, forced=None, mode="max", time_limit=TIME_LIMIT):
    """The 0-1 program of 16.4 over the variables 0..n-1. A variable u with free[u] False needs a parent in the pool:
    x_u <= sum of x_a over parents[u] (a list of variable indices; a parent that is already in the F-list is expressed by free[u] True).
    mode 'max': maximise sum w_u x_u with sum x <= budget. mode 'min': minimise sum x_u (w ignored) with x_u = 1 for u in forced.
    Returns (x, status, objective, lp_objective): status 'optimal' only if proven; the LP relaxation is the same rows with x in [0, 1]."""
    from scipy.optimize import LinearConstraint, milp
    from scipy.sparse import coo_matrix
    n = len(free)
    rows, cols, vals, ub = [], [], [], []
    r = 0
    ubd = np.ones(n)
    for u in range(n):
        if free[u]:
            continue
        ps = parents[u]
        if len(ps) == 0:
            ubd[u] = 0.0                                   # no parent in the pool can ever exist: u cannot be added
            continue
        rows.append(r); cols.append(u); vals.append(1.0)
        for a in ps:
            rows.append(r); cols.append(a); vals.append(-1.0)
        ub.append(0.0)
        r += 1
    if mode == "max":
        rows += [r] * n; cols += list(range(n)); vals += [1.0] * n
        ub.append(float(budget))
        r += 1
        c = -np.asarray(w, float)
    else:
        c = np.ones(n)
    lb = np.zeros(n)
    if forced is not None and len(forced):
        lb[np.asarray(forced, np.int64)] = 1.0
    if np.any(lb > ubd):
        return None, "infeasible", None, None
    cons = []
    if r:
        A = coo_matrix((vals, (rows, cols)), shape=(r, n)).tocsr()
        cons = [LinearConstraint(A, -np.inf, np.asarray(ub))]
    from scipy.optimize import Bounds
    bnd = Bounds(lb, ubd)
    res = milp(c, constraints=cons, bounds=bnd, integrality=np.ones(n), options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status == 2:
        return None, "infeasible", None, None
    st = "optimal" if res.status == 0 else "not_optimal"
    if res.x is None:
        return None, st, None, None
    x = np.rint(res.x).astype(np.int8)
    lp = milp(c, constraints=cons, bounds=bnd, integrality=np.zeros(n), options={"time_limit": time_limit})
    obj = float(c @ x)
    return x, st, obj, (float(lp.fun) if lp.status == 0 else None)


def brute_closure(w, free, parents, budget=None, forced=None, mode="max"):
    """Exhaustive optimum of the same program (n <= 14): the objective value, or None if infeasible."""
    n = len(free)
    best = None
    fm = 0
    for u in (forced if forced is not None else []):
        fm |= 1 << int(u)
    for m in range(1 << n):
        if (m & fm) != fm:
            continue
        sel = [u for u in range(n) if m >> u & 1]
        if mode == "max" and len(sel) > budget:
            continue
        ok = True
        for u in sel:
            if not free[u] and not any(m >> a & 1 for a in parents[u]):
                ok = False
                break
        if not ok:
            continue
        v = float(sum(w[u] for u in sel)) if mode == "max" else float(len(sel))
        if best is None or (v > best if mode == "max" else v < best):
            best = v
    return best


# -- the graph, the walk, the oracle -----------------------------------------


def edge_rows(G, op_of, mark, backward):
    """(a, b) sorted unique pairs with a marked: forward = b a stored out-neighbour of a (R both ways, S src to dst); backward = a an
    end that b has a stored edge TO (R both ways, S: b = src, a = dst). Only labels with an operator."""
    n = G.n
    pa, pb = [], []
    for L in range(G.nR):
        if op_of[L] < 0:
            continue
        keys = G.r_keys[L]
        for i in range(0, keys.size, 1 << 24):
            a, b = Z.ends(keys[i:i + (1 << 24)], n)
            m = mark[a]
            pa.append(a[m]); pb.append(b[m])
            m = mark[b]
            pa.append(b[m]); pb.append(a[m])
    if G.s_key.size:
        ok = op_of[G.nR + G.s_rel] >= 0
        if backward:
            m = ok & mark[G.s_dst]
            pa.append(G.s_dst[m]); pb.append(G.s_src[m])
        else:
            m = ok & mark[G.s_src]
            pa.append(G.s_src[m]); pb.append(G.s_dst[m])
    if not pa:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    k = np.unique(np.concatenate(pa).astype(np.int64) * n + np.concatenate(pb).astype(np.int64))
    return k // n, k % n


def direct_rows(G, op_of, nodes, backward):
    """The same rows by another code path (np.isin over the keys, no boolean mark): G-A's independent read."""
    n = G.n
    nodes = np.unique(np.asarray(nodes, np.int64))
    pa, pb = [], []
    for L in range(G.nR):
        if op_of[L] < 0:
            continue
        keys = G.r_keys[L]
        for i in range(0, keys.size, 1 << 24):
            a, b = Z.ends(keys[i:i + (1 << 24)], n)
            m = np.isin(a, nodes)
            pa.append(a[m]); pb.append(b[m])
            m = np.isin(b, nodes)
            pa.append(b[m]); pb.append(a[m])
    if G.s_key.size:
        for L in range(G.RS):
            if op_of[G.nR + L] < 0:
                continue
            sel = G.s_rel == L
            s, d = G.s_src[sel], G.s_dst[sel]
            if backward:
                m = np.isin(d, nodes)
                pa.append(d[m]); pb.append(s[m])
            else:
                m = np.isin(s, nodes)
                pa.append(s[m]); pb.append(d[m])
    if not pa:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    k = np.unique(np.concatenate(pa).astype(np.int64) * n + np.concatenate(pb).astype(np.int64))
    return k // n, k % n


def out_degree(G, op_of):
    """The number of stored out-edges per node over the followed labels (an undirected pair counts at both ends; a pair under two labels twice): an upper bound."""
    n = G.n
    deg = np.zeros(n, np.int64)
    for L in range(G.nR):
        if op_of[L] < 0:
            continue
        keys = G.r_keys[L]
        for i in range(0, keys.size, 1 << 24):
            a, b = Z.ends(keys[i:i + (1 << 24)], n)
            deg += np.bincount(a, minlength=n)
            deg += np.bincount(b, minlength=n)
    if G.s_key.size:
        ok = op_of[G.nR + G.s_rel] >= 0
        deg += np.bincount(G.s_src[ok], minlength=n)
    return deg


def oracle_one(seeds, Fset, gold, fa, fb, ba, bb, time_limit=TIME_LIMIT):
    """Pi_oracle for one question (16.4). Returns a dict: classes, the optimum, its status, the LP bound."""
    out = {"n_gold": int(gold.size), "n_missed": 0, "c1": 0, "c2": 0, "beyond": 0, "cost": 0, "status": "none", "lp_eq": 1, "A": 0, "A_new": 0, "D_ub": 0, "needy": 0}
    lo, hi = np.searchsorted(fa, seeds, "left"), np.searchsorted(fa, seeds, "right")
    A = np.setdiff1d(np.unique(fb[Z.expand(lo, hi)]), seeds)
    out["A"] = int(A.size)
    out["A_new"] = int(np.setdiff1d(A, Fset).size)
    missed = np.setdiff1d(gold, Fset)
    out["n_missed"] = int(missed.size)
    if not missed.size:
        return out
    in_a = np.isin(missed, A)
    c1 = missed[in_a]
    rest = missed[~in_a]
    c2, par, beyond = [], {}, 0
    for g in rest:
        l, h = np.searchsorted(ba, g, "left"), np.searchsorted(ba, g, "right")
        P = np.intersect1d(bb[l:h], A)
        if P.size:
            c2.append(int(g)); par[int(g)] = P
        else:
            beyond += 1
    out.update(c1=int(c1.size), c2=len(c2), beyond=int(beyond))
    if beyond:
        out["status"] = "infeasible"; out["cost"] = -1
        return out
    free_set = np.union1d(Fset, c1)
    needy = [g for g in c2 if not np.isin(par[g], free_set).any()]
    out["needy"] = len(needy)
    if not needy:
        out["status"] = "closed"; out["cost"] = int(c1.size + len(c2))
        return out
    # variables: the needy golds, then the paid parents of those golds; the program as declared, solved exactly
    pids = np.unique(np.concatenate([par[g] for g in needy]))
    idx = {int(a): len(needy) + j for j, a in enumerate(pids)}
    n = len(needy) + pids.size
    free = [False] * len(needy) + [True] * pids.size
    parents = [[idx[int(a)] for a in par[g]] for g in needy] + [[] for _ in range(pids.size)]
    x, st, obj, lp = solve_closure(np.zeros(n), free, parents, forced=list(range(len(needy))), mode="min", time_limit=time_limit)
    if x is None or st != "optimal":
        out["status"] = "not_optimal"; out["cost"] = -2
        return out
    out["status"] = "optimal"
    out["cost"] = int(c1.size + (len(c2) - len(needy)) + obj)
    out["lp_eq"] = int(lp is not None and abs(lp - obj) < 1e-6)
    return out


def reach_engine(G, op_of, base, flat, off, b_seeds=B_SEEDS, F=F_DEPTH, time_limit=TIME_LIMIT, log_every=250):
    n, nq = G.n, base.shape[0]
    Fl, Sd, Ms = [], [], []
    mark_s = np.zeros(n, bool)
    mark_g = np.zeros(n, bool)
    for q in range(nq):
        r = base[q]
        r = r[r >= 0]
        Fl.append(np.unique(r[:F]))
        Sd.append(r[:b_seeds].astype(np.int64))
        mark_s[Sd[-1]] = True
        ms = np.setdiff1d(flat[off[q]:off[q + 1]], Fl[-1])
        Ms.append(ms)
        mark_g[ms] = True
    t0 = time.time()
    fa, fb = edge_rows(G, op_of, mark_s, False)
    log(f"forward rows from {int(mark_s.sum()):,} seeds: {fa.size:,} ({time.time() - t0:.0f} s)")
    ba, bb = edge_rows(G, op_of, mark_g, True)
    log(f"backward rows into {int(mark_g.sum()):,} missed golds: {ba.size:,} ({time.time() - t0:.0f} s)")
    deg = out_degree(G, op_of)
    cols = ("n_gold", "n_missed", "c1", "c2", "beyond", "cost", "A", "A_new", "needy", "lp_eq")
    res = {k: np.zeros(nq, np.int64) for k in cols}
    res["status"] = np.zeros(nq, np.int8)
    res["D_ub"] = np.zeros(nq, np.int64)
    for q in range(nq):
        gold = flat[off[q]:off[q + 1]]
        o = oracle_one(Sd[q], Fl[q], gold, fa, fb, ba, bb, time_limit)
        for k in cols:
            res[k][q] = o[k]
        res["status"][q] = STATUS[o["status"]]
        lo, hi = np.searchsorted(fa, Sd[q], "left"), np.searchsorted(fa, Sd[q], "right")
        A = np.setdiff1d(np.unique(fb[Z.expand(lo, hi)]), Sd[q])
        res["D_ub"][q] = int(deg[A].sum())
        if log_every and (q + 1) % log_every == 0:
            log(f"question {q + 1:,} of {nq:,} ({time.time() - t0:.0f} s)")
    return res, {"forward_rows": int(fa.size), "backward_rows": int(ba.size), "seeds": int(mark_s.sum()), "missed_golds": int(mark_g.sum())}


# -- the data (the host) ------------------------------------------------------


def graph_only(ds, host):
    """zsem2.load_six's graph part (the same checks, gate G-E) without the vectors; MultiHop-RAG through zsem2.load_mhr (609 rows)."""
    if ds == "multihoprag":
        G, V, inputs, ge = Z.load_mhr("e3v")
        return G, inputs, ge
    import u1b
    import zul1 as UL
    served, freeze = u1b.package_root(host)
    man = json.loads((served / ds / "DATASET.json").read_text(encoding="utf-8"))
    n, nsha = int(man["nodes"]["n"]), man["nodes"]["sha256"]
    F, sh, slab = UL.families(served, ds, n, nsha, freeze)
    fp = json.loads((UL.OUT / ds / "grade.json").read_text(encoding="utf-8"))["family_pairs"]
    ge = {}
    for fam in ("W2_S", "CN_all", "F0T") + (("S",) if ds in UL.HAS_S else ()):
        ge[fam] = {"pairs": int(F[fam].size), "record": int(fp[fam])}
        if F[fam].size != fp[fam]:
            stop(f"gate G-E: {ds}: {fam} has {F[fam].size} pairs, UL1's grade {fp[fam]}")
    for k in ("N_sp", "K", "S"):
        F.pop(k, None)
    R = [("R.mention", F.pop("W2_S")), ("R.name_comention", F["CN_all"]),
         ("R.title_comention", np.setdiff1d(F.pop("F0T"), F.pop("CN_all"), assume_unique=True))]
    S, vocab = None, ()
    if ds in UL.HAS_S:
        f = json.loads((served / ds / "graph" / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))["families"]["structural"]
        p = served / ds / "graph" / Path(f["file"]).name
        vocab = [str(v) for v in (f.get("relation_vocabulary") or [])]
        RS = max(len(vocab), 1)
        with np.load(p) as z:
            a, b, r = z["src"].astype(np.int64), z["dst"].astype(np.int64), z["rel"].astype(np.int64)
        m = a != b
        S = np.unique((a[m] * n + b[m]) * RS + r[m])
        del a, b, r, m
        k = S // RS
        a, b = Z.ends(k, n)
        und = np.unique((np.minimum(a, b) * n + np.maximum(a, b)) * RS + (S - k * RS))
        del a, b, k
        if not np.array_equal(und, slab[0] * RS + slab[1]):
            stop(f"gate G-E: {ds}: the directed S read is not zul1.s_labels' (pair, relation) rows")
        ge["S directed"] = {"edges": int(S.size), "pair_relation_rows": int(und.size), "verdict": "PASS"}
        del und
    G = Z.Graph(ds, n, R, S, vocab)
    return G, {"freeze_RECORD_SHA256": freeze, "nodes_sha256": nsha, **sh}, ge


def questions8(ds, host):
    """The selection carve (16.3): s1sel; MultiHop-RAG's dev split; WebQSP's PL dev (outputs/pl2/webqsp/pop_dev.npz)."""
    import zsem2t as T
    if ds != "webqsp":
        return T.questions(ds, "e3v", "sel", host)
    import zenc3v  # noqa: F401  (registers e3v)
    import zenc1 as E1
    import zsem1 as S1
    import zul1 as UL
    served, _freeze = E1.package(host)
    p = Z.ROOT / "outputs" / "pl2" / "webqsp" / "pop_dev.npz"
    if not p.is_file():
        stop(f"{p} is not here")
    with np.load(p) as z:
        ids, flat0, off0, dg = [str(x) for x in z["ids"]], z["flat"].astype(np.int64), z["off"], str(z["digest"])
    golds = [np.unique(flat0[off0[i]:off0[i + 1]]) for i in range(len(ids))]
    eval_ids, _g, _s = UL.carve_questions("webqsp", "s1eval")
    if set(ids) & set(eval_ids):
        stop("gate G-Q: PL's dev carve and s1eval share questions")
    flat, off = S1.flatten(golds)
    rows, Q = S1.query_rows(served, "e3v", ds, ids)
    spl = S1.splade(served, ds, rows)
    paths = E.vec_shards(served, "e3v", ds, "docs")
    info = {"carve": "PL dev (pop_dev.npz)", "digest": dg, "questions": len(ids), "with_gold": int((np.diff(off) > 0).sum()), "disjoint_from_s1eval": True}
    return {"ids": ids, "rows": rows, "Q": Q, "flat": flat, "off": off, "spl": spl, "paths": paths, "served": served, "info": info}


# -- the read -----------------------------------------------------------------


def cmd_reach(a):
    import zsem1 as S1
    import zsem2t as T
    t0 = time.time()
    ds = a.ds
    od = (OUT / "smoke" / ds) if a.smoke else (OUT / ds)
    G, inputs, ge = graph_only(ds, a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}}
    log(f"{ds}: graph {G.n:,} nodes ({time.time() - t0:.0f} s)")
    split = Z.node_split(G.name, G.n)
    labs, ops, op_of, _groups = Z.plan(G, split, dict(Z.CFG))
    r1 = json.loads((Z.OUT / ds / "e3v.json").read_text(encoding="utf-8"))
    if [o["name"] for o in ops] != list(r1["operators"]):
        stop("gate G-E: the labels followed are not R1's operator labels")
    gates["G-E"]["labels"] = [o["name"] for o in ops]
    qd = questions8(ds, a.host)
    ids, flat, off = qd["ids"], qd["flat"], qd["off"]
    keep = np.flatnonzero(np.diff(off) > 0)
    if a.smoke:
        keep = keep[:60]
    sel = np.zeros(len(ids), bool)
    sel[keep] = True
    v_top = T.vlist(qd, G.n)
    base = S1.rrf((v_top, qd["spl"]), F_DEPTH)
    rec5 = S1.mean(T.recall(base[:, :5], flat, off))
    p5 = Path(__file__).resolve().parents[1] / "sem2" / "r2" / "sel" / "e3v" / f"{ds}.json"
    if p5.is_file():
        want = json.loads(p5.read_text(encoding="utf-8"))["base5"]
        ok = abs(rec5 - want) <= 5e-4
        gates["G-Q"] = {"base5": round(rec5, 6), "r2a_selection_base5": want, "verdict": "PASS" if ok else "FAIL"}
        if not ok:
            stop(f"gate G-Q: {gates['G-Q']}")
    else:
        gates["G-Q"] = {"base5": round(rec5, 6), "verdict": "PASS", "note": "no R2a selection record to equal" if not a.smoke else "smoke"}
    gold_in_range = bool(flat.size == 0 or (flat.min() >= 0 and flat.max() < G.n))
    if not gold_in_range:
        stop("gate G-Q: a gold position is out of range")
    # the questions read: the ones with a gold (and, in a smoke run, the first 60)
    q_ix = np.flatnonzero(sel)
    sub_flat, sub_off = [], [0]
    for q in q_ix:
        sub_flat.append(flat[off[q]:off[q + 1]])
        sub_off.append(sub_off[-1] + int(off[q + 1] - off[q]))
    sflat = np.concatenate(sub_flat) if sub_flat else np.zeros(0, np.int64)
    soff = np.asarray(sub_off, np.int64)
    sbase = base[q_ix]
    res, meta = reach_engine(G, op_of, sbase, sflat, soff)
    # G-A: the adjacency by another code path, on 40 questions' seeds and 40 questions' missed golds
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(q_ix), min(40, len(q_ix)), replace=False)
    seeds_chk = np.unique(np.concatenate([sbase[i][sbase[i] >= 0][:B_SEEDS] for i in pick]))
    miss_chk = np.unique(np.concatenate([np.setdiff1d(sflat[soff[i]:soff[i + 1]], sbase[i][sbase[i] >= 0][:F_DEPTH]) for i in pick] + [np.zeros(0, np.int64)]))
    mk = np.zeros(G.n, bool)
    mk[seeds_chk] = True
    f1 = edge_rows(G, op_of, mk, False)
    f2 = direct_rows(G, op_of, seeds_chk, False)
    mk = np.zeros(G.n, bool)
    mk[miss_chk] = True
    b1 = edge_rows(G, op_of, mk, True)
    b2 = direct_rows(G, op_of, miss_chk, True)
    ga = bool(np.array_equal(f1[0], f2[0]) and np.array_equal(f1[1], f2[1]) and np.array_equal(b1[0], b2[0]) and np.array_equal(b1[1], b2[1]))
    gates["G-A"] = {"seeds": int(seeds_chk.size), "missed_golds": int(miss_chk.size), "forward_rows": int(f1[0].size), "backward_rows": int(b1[0].size),
                    "verdict": "PASS" if ga else "FAIL"}
    if not ga:
        stop(f"gate G-A: {gates['G-A']}")
    st = res["status"]
    inst = int(np.isin(st, [STATUS["closed"], STATUS["optimal"], STATUS["not_optimal"]]).sum())
    opt = int(np.isin(st, [STATUS["closed"], STATUS["optimal"]]).sum())
    feas = res["cost"] >= 0
    lo_b = res["c1"] + res["c2"]
    hi_b = res["c1"] + 2 * res["c2"]
    bounds_ok = bool(np.all((res["cost"][feas & (res["n_missed"] > 0)] >= lo_b[feas & (res["n_missed"] > 0)]) &
                            (res["cost"][feas & (res["n_missed"] > 0)] <= hi_b[feas & (res["n_missed"] > 0)])))
    share = opt / inst if inst else 1.0                      # no instance (every gold already in the F-list): vacuous
    gates["G-P"] = {"instances": inst, "proven_optimal": opt, "share": round(share, 6), "cost_inside_bounds": bounds_ok,
                    "lp_equals_ip_share": round(float(res["lp_eq"][st == STATUS["optimal"]].mean()), 6) if (st == STATUS["optimal"]).any() else None,
                    "verdict": "PASS" if (share >= OPT_SHARE and bounds_ok) else "FAIL"}
    if gates["G-P"]["verdict"] != "PASS":
        log(f"gate G-P FAILED: {gates['G-P']}")
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / "reach.npz", ids=np.asarray([ids[q] for q in q_ix]), **res)
    rec = {"stage": "SEM2 R8 part A", "declared_in": DECLARED, "dataset": ds, "carve": qd["info"], "questions": int(q_ix.size), "F": F_DEPTH, "seeds": B_SEEDS,
           "ladder": list(LADDER), "gates": gates, "engine": meta, "labels": [o["name"] for o in ops], "inputs": inputs, "threads": E.THREADS,
           "smoke": bool(a.smoke), "npz_sha256": Z.sha_file(od / "reach.npz"), "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": script_sha()}
    Z.wjson(od / "reach.json", rec)
    log(f"{ds}: part A written ({rec['seconds']} s); gates {json.dumps({k: (v['verdict'] if isinstance(v, dict) else v) for k, v in gates.items()})}")
    return 0 if gates["G-P"]["verdict"] == "PASS" else 1


# -- the report (16.5, 16.8) --------------------------------------------------


def boot_share(ind, rng_seed=SEED):
    ind = np.asarray(ind, float)
    n = ind.size
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(BOOT, n))
    d = ind[idx].mean(1) * 100
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def part_a_row(res, ds):
    """The measures of 16.5 from the per-question arrays."""
    n = res["n_gold"].size
    nm = res["n_missed"]
    miss_q = nm > 0
    cov0 = 100 * float((~miss_q).mean())
    tot = float(nm.sum())
    sh = {k: (100 * float(res[k].sum()) / tot if tot else 0.0) for k in ("c1", "c2", "beyond")}
    out = {"questions": int(n), "cov0": round(cov0, 2), "missed_questions": int(miss_q.sum()), "missed_golds": int(tot),
           "missed_gold_share": {k: round(v, 2) for k, v in sh.items()}, "H": {}}
    feas = (res["cost"] >= 0) & miss_q
    for M in LADDER:
        comp = feas & (res["cost"] <= M)
        lo, hi = boot_share(comp)
        out["H"][str(M)] = {"completable": int(comp.sum()), "H_points": round(100 * float(comp.mean()), 2), "lo": round(lo, 2), "hi": round(hi, 2)}
    out["room"] = bool(out["H"][str(MAIN)]["H_points"] >= ROOM_POINTS)
    cmp_ = feas & (res["cost"] <= MAIN)
    c = res["cost"][cmp_]
    out["cost_star"] = {"median": float(np.median(c)) if c.size else None, "mean": round(float(c.mean()), 2) if c.size else None,
                        "p90": float(np.percentile(c, 90)) if c.size else None}
    fld = miss_q
    unord = (res["A_new"] + res["D_ub"])[fld]
    out["field"] = {"A_median": float(np.median(res["A"][fld])) if fld.any() else None, "A_new_median": float(np.median(res["A_new"][fld])) if fld.any() else None,
                    "unordered_cost_median": float(np.median(unord)) if fld.any() else None,
                    "share_unordered_over_1500": round(100 * float((unord > MAIN).mean()), 2) if fld.any() else None}
    out["status"] = {k: int((res["status"] == v).sum()) for k, v in STATUS.items()}
    return out


def cmd_report(_a):
    sha = script_sha()
    rows, miss = {}, []
    for ds in SEVEN:
        p = OUT / ds / "reach.json"
        if not p.is_file():
            miss.append(ds)
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["script_sha256"] != sha:
            stop(f"{ds}: the record was made by other code")
        z = np.load(OUT / ds / "reach.npz")
        if Z.sha_file(OUT / ds / "reach.npz") != r["npz_sha256"]:
            stop(f"{ds}: reach.npz is not its record's")
        row = part_a_row({k: z[k] for k in z.files if k != "ids"}, ds)
        row["gates"] = {k: (v["verdict"] if isinstance(v, dict) and "verdict" in v else v) for k, v in r["gates"].items()}
        row["seconds"] = r["seconds"]
        rows[ds] = row
    room = [d for d, r in rows.items() if r["room"]]
    rep = {"declared_in": DECLARED, "script_sha256": sha, "missing": miss, "datasets": rows, "room": room,
           "part_b_runs_on": room if (len(room) >= 1) else [], "verdict_if_complete": ("PART_A_ONLY_NO_ROOM" if not room else ("INCONCLUSIVE_B_NEEDS_TWO" if len(room) < 2 else "PART_B_ON_ROOM"))}
    Z.wjson(OUT / "report.json", rep)
    L = ["# SEM2 R8 part A: the room", "", f"Declared in {DECLARED}. Questions: the selection carve; F = {F_DEPTH}, seeds {B_SEEDS}, two hops. H(M') in points: the first stage's ALL-gold coverage gain if a perfect router spent M' added nodes.", "",
         "| dataset | questions | cov0 | missed golds: class 1 / class 2 / beyond (%) | H(25) | H(100) | H(500) | H(1500) [95%] | ROOM | median cost* | median unordered cost |", "|---|---:|---:|---|---:|---:|---:|---|---|---:|---:|"]
    for ds, r in rows.items():
        h = r["H"]
        sh = r["missed_gold_share"]
        L.append(f"| {ds} | {r['questions']} | {r['cov0']:.2f} | {sh['c1']:.1f} / {sh['c2']:.1f} / {sh['beyond']:.1f} | {h['25']['H_points']:.2f} | {h['100']['H_points']:.2f} | {h['500']['H_points']:.2f} | "
                 f"{h['1500']['H_points']:.2f} [{h['1500']['lo']:.2f}, {h['1500']['hi']:.2f}] | {'ROOM' if r['room'] else 'NO_ROOM'} | {r['cost_star']['median']} | {r['field']['unordered_cost_median']} |")
    L += ["", f"ROOM on: {', '.join(room) if room else 'none'}. Missing rows: {', '.join(miss) if miss else 'none'}."]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    log(f"report written: ROOM on {room}; missing {miss}")
    return 0


# -- the self-test -------------------------------------------------------------


def toy_graph(rng, n=48):
    pairs = set()
    while len(pairs) < 70:
        a, b = rng.integers(0, n, 2)
        if a != b:
            pairs.add((int(min(a, b)), int(max(a, b))))
    keys = np.asarray(sorted(a * n + b for a, b in pairs), np.int64)
    pairs2 = set()
    while len(pairs2) < 25:
        a, b = rng.integers(0, n, 2)
        if a != b:
            pairs2.add((int(min(a, b)), int(max(a, b))))
    keys2 = np.asarray(sorted(a * n + b for a, b in pairs2), np.int64)
    S = set()
    while len(S) < 40:
        a, b, r = rng.integers(0, n, 2).tolist() + [int(rng.integers(0, 3))]
        if a != b:
            S.add((a * n + b) * 3 + r)
    S = np.asarray(sorted(S), np.int64)
    G = Z.Graph("toy", n, [("R.a", keys), ("R.b", keys2)], S, ["r0", "r1", "r2"])
    op_of = np.asarray([0, 1, 2, 3, -1], np.int64)               # R.a, R.b, S.r0, S.r1 followed; S.r2 not
    return G, op_of


def py_out(G, op_of):
    n = G.n
    out = {x: set() for x in range(n)}
    for L in range(G.nR):
        if op_of[L] < 0:
            continue
        for k in G.r_keys[L].tolist():
            a, b = divmod(k, n)
            out[a].add(b); out[b].add(a)
    for s, d, r in zip(G.s_src.tolist(), G.s_dst.tolist(), G.s_rel.tolist()):
        if op_of[G.nR + r] >= 0:
            out[s].add(d)
    return out


def py_oracle(out, seeds, Fset, gold):
    """The same program by plain python: classes and the exact minimum by exhaustive search over the relevant nodes (golds and their hop-1 parents)."""
    A = set()
    for s in seeds:
        A |= out[s]
    A -= set(seeds)
    missed = sorted(set(gold) - set(Fset))
    c1 = [g for g in missed if g in A]
    c2, beyond = [], 0
    for g in missed:
        if g in A:
            continue
        parents = [a for a in A if g in out[a]]
        if parents:
            c2.append(g)
        else:
            beyond += 1
    if not missed:
        return {"c1": 0, "c2": 0, "beyond": 0, "cost": 0}
    if beyond:
        return {"c1": len(c1), "c2": len(c2), "beyond": beyond, "cost": -1}
    rel = sorted(set(c1) | set(c2) | {a for g in c2 for a in A if g in out[a]})
    best = None
    must = set(c1) | set(c2)
    others = [x for x in rel if x not in must and x not in Fset]
    for k in range(len(others) + 1):
        for comb in itertools.combinations(others, k):
            pool = set(Fset) | must | set(comb)
            if all(any(a in pool for a in A if g in out[a]) for g in c2):
                best = len(must) + k
                break
        if best is not None:
            break
    return {"c1": len(c1), "c2": len(c2), "beyond": 0, "cost": best}


def selftest(_a):
    rng = np.random.default_rng(8)
    # 1. the exact solver against exhaustive search, on random small instances of both programs
    n_ok = 0
    for t in range(200):
        n = int(rng.integers(3, 11))
        free = [bool(rng.random() < 0.35) for _ in range(n)]
        parents = [[int(a) for a in rng.choice(n, int(rng.integers(0, 4)), replace=False) if a != u] for u in range(n)]
        w = [float(x) for x in rng.random(n)]
        bud = int(rng.integers(1, n + 1))
        x, st, obj, lp = solve_closure(w, free, parents, budget=bud, mode="max")
        ref = brute_closure(w, free, parents, budget=bud, mode="max")
        assert st == "optimal" and ref is not None and abs(-obj - ref) < 1e-7, (t, st, obj, ref)
        assert lp is not None and lp <= obj + 1e-7
        n_ok += 1
        forced = [int(u) for u in rng.choice(n, int(rng.integers(0, 3)), replace=False)]
        x, st, obj, lp = solve_closure(w, free, parents, forced=forced, mode="min")
        ref = brute_closure(w, free, parents, forced=forced, mode="min")
        if ref is None:
            assert x is None and st == "infeasible", (t, st)
        else:
            assert st == "optimal" and abs(obj - ref) < 1e-7, (t, st, obj, ref)
            assert lp is not None and lp <= obj + 1e-7
    # 2. the adjacency by two code paths, and the walk, the classes and the cost against plain python
    nq_tot = 0
    n_feas = n_inf = n_needy = 0
    for g_i in range(12):
        G, op_of = toy_graph(np.random.default_rng(100 + g_i))
        n = G.n
        out = py_out(G, op_of)
        nodes = rng.choice(n, 9, replace=False)
        mk = np.zeros(n, bool)
        mk[nodes] = True
        for backward in (False, True):
            a1 = edge_rows(G, op_of, mk, backward)
            a2 = direct_rows(G, op_of, nodes, backward)
            assert np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1])
        a1 = edge_rows(G, op_of, mk, False)
        want = sorted((x, y) for x in nodes.tolist() for y in out[x])
        assert list(zip(a1[0].tolist(), a1[1].tolist())) == want, "forward rows"
        nq = 40
        base = np.stack([rng.permutation(n)[:20] for _ in range(nq)]).astype(np.int64)
        base[:, 17:] = np.where(rng.random((nq, 3)) < 0.3, -1, base[:, 17:])
        flat, off = [], [0]
        for q in range(nq):
            k = int(rng.integers(1, 5))
            flat += sorted(set(rng.choice(n, k, replace=False).tolist()))
            off.append(len(flat))
        flat, off = np.asarray(flat, np.int64), np.asarray(off, np.int64)
        res, _meta = reach_engine(G, op_of, base, flat, off, b_seeds=4, F=12, log_every=0)
        for q in range(nq):
            r = base[q][base[q] >= 0]
            ref = py_oracle(out, [int(x) for x in r[:4]], [int(x) for x in r[:12]], [int(x) for x in flat[off[q]:off[q + 1]]])
            assert (res["c1"][q], res["c2"][q], res["beyond"][q]) == (ref["c1"], ref["c2"], ref["beyond"]), (g_i, q, res["c1"][q], res["c2"][q], res["beyond"][q], ref)
            if ref["beyond"] == 0 and res["n_missed"][q] > 0:
                assert res["cost"][q] == ref["cost"], (g_i, q, res["cost"][q], ref)
                n_feas += 1
                n_needy += int(res["needy"][q] > 0)
            elif ref["beyond"]:
                assert res["cost"][q] == -1 and res["status"][q] == STATUS["infeasible"]
                n_inf += 1
            nq_tot += 1
    assert n_feas > 30 and n_inf > 30 and n_needy > 3, (n_feas, n_inf, n_needy)
    # 3. the report's rule over stand-in arrays
    nq = 200
    res = {"n_gold": np.full(nq, 2), "n_missed": np.r_[np.ones(50, int), np.zeros(150, int)], "c1": np.r_[np.ones(50, int), np.zeros(150, int)],
           "c2": np.zeros(nq, int), "beyond": np.zeros(nq, int), "cost": np.r_[np.full(50, 3), np.zeros(150, int)], "A": np.full(nq, 100), "A_new": np.full(nq, 90),
           "D_ub": np.r_[np.full(25, 4000), np.full(nq - 25, 10)], "status": np.r_[np.full(50, STATUS["closed"]), np.zeros(150, int)].astype(np.int8), "lp_eq": np.ones(nq, int), "needy": np.zeros(nq, int)}
    row = part_a_row(res, "toy")
    assert abs(row["H"]["1500"]["H_points"] - 25.0) < 1e-9 and row["room"] and row["cov0"] == 75.0 and row["field"]["share_unordered_over_1500"] == 50.0
    log(f"self-test PASS: HiGHS equals exhaustive search on {n_ok} random instances of both programs (LP <= IP); the adjacency by two code paths equals plain python; "
        f"{nq_tot} toy questions: classes and cost equal an exhaustive oracle ({n_feas} feasible, {n_needy} with a paid parent, {n_inf} infeasible); the report's rule runs")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "reach", "report"))
    p.add_argument("--ds", choices=SEVEN)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="60 questions; written under outputs/sem2/r8/smoke/, read by nothing")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "report":
        return cmd_report(a)
    if not a.ds:
        stop("reach needs --ds")
    return cmd_reach(a)


if __name__ == "__main__":
    sys.exit(main())
