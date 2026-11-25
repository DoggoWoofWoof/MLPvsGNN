"""UL5 (docs/UL5_DEPTH_SIGNAL.md), the solver half: a label-free depth signal as a constrained program, laptop only (numpy and scipy).

    python outputs/mp_unified/zul5cp.py selftest
    python outputs/mp_unified/zul5cp.py gates                      # G5-1 on the dev extraction (data identity with the filed grids)
    python outputs/mp_unified/zul5cp.py solve                      # dev: P1 exhaustive over the declared policy class, P2 beside, the null, the oracle and the uniform
    python outputs/mp_unified/zul5cp.py grade                      # eval, once, for the dev-chosen policy (only after the dev call is filed)
    python outputs/mp_unified/zul5cp.py report

The decision variables are one level per bin (y[b, j] in {0, 1}) of a policy that is the SAME on all seven datasets: a statistic s, a sign, k - 1 breakpoints and one level of the
universal ladder (M walk nodes kept; the last level is the uncut pool) per bin, the levels non-decreasing. P1 minimises the worst dataset's mean nodes subject to ALL >= 0.95 on every
dataset (a mixed-integer program, solved exactly by branch and bound; the breakpoints are enumerated exhaustively over the declared grid). Reads outputs/ul5/<d>/ext_<carve>*.npz
and the filed dev grids; writes outputs/ul5/ only.
"""
import argparse
import hashlib
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
OUT = ROOT / "outputs" / "ul5"
DECLARED = "docs/UL5_DEPTH_SIGNAL.md"
SEVEN = ("multihoprag", "squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp")
LADDER_M = (0, 100, 200, 400, 800, 1200, 1600, 2000, 2500, 3500, 5000, 7500, 10000, 15000, 20000, 30000, None)    # the universal ladder (None = the uncut pool); same as zul5st's
J = len(LADDER_M)
STATS = ("log10_pool", "growth", "cos1", "gap15", "agree")
SIGNS = (1, -1)
KS = (2, 3, 4)
RHO = 0.95
QUANTS = tuple(round(0.1 * i, 1) for i in range(1, 10))
CAPS = (2500, 3500, 5000)
NULLS = 20
NULL_SEED = 20261126
SAVE_MIN = 0.10                          # the call: the saving over the uniform level must be at least this
BOUND = 2500
EPS = 1e-7
TOL = 1e-9
PARSIMONY = 1.03                         # the smallest k whose worst-dataset nodes are within this factor of the best over k


def stop(msg):
    raise SystemExit(f"[zul5cp] STOP: {msg}")


def log(msg):
    print(f"[zul5cp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── data ─────────────────────────────────────────────────────────────────────

def load_ext(d, carve):
    """The extraction of one dataset and carve: parts stacked on questions (they must tile), cells in the declared order."""
    fs = sorted((OUT / d).glob(f"ext_{carve}*.npz"))
    fs = [f for f in fs if ".tmp" not in f.name]
    if not fs:
        stop(f"{d}: no {carve} extraction in outputs/ul5/{d}")
    zs = [np.load(f, allow_pickle=True) for f in fs]
    zs.sort(key=lambda z: int(z["lo"]))
    if int(zs[0]["lo"]) != 0 or any(int(a["hi"]) != int(b["lo"]) for a, b in zip(zs, zs[1:])):
        stop(f"{d}: the {carve} parts do not tile the questions")
    names = [str(x) for x in zs[0]["names"]]
    if any([str(x) for x in z["names"]] != names for z in zs):
        stop(f"{d}: the {carve} parts hold different cells")
    return {"names": names, "present": np.concatenate([z["present"] for z in zs], axis=0).astype(np.int64), "size": np.concatenate([z["size"] for z in zs], axis=0).astype(np.int64),
            "gold": np.concatenate([z["gold_counts"] for z in zs]).astype(np.int64), "qids": np.concatenate([z["qids"] for z in zs]),
            "stats": np.concatenate([z["stats"] for z in zs], axis=0), "files": [f.name for f in fs]}


def ladder_arrays(e):
    """(A, Z) on the universal ladder: A[q, j] = 1 iff every gold is in the pool of level j, Z[q, j] = its size. A family with one cell (SQuAD) repeats it on every level."""
    n = len(e["names"])
    A = ((e["present"] == e["gold"][:, None]) & (e["gold"][:, None] > 0)).astype(np.int64)
    Z = e["size"].astype(np.float64)
    if n == 1:
        return np.repeat(A, J, axis=1), np.repeat(Z, J, axis=1)
    if n != J:
        stop(f"{n} cells, expected {J} or 1")
    return A, Z


def statistics(e):
    """The five declared statistics, per question. The first two are read off the pool sizes (the base and the uncut pool); the last three are the extraction's."""
    Z = e["size"].astype(np.float64)
    base, pool = Z[:, 0], Z[:, -1]
    out = {"log10_pool": np.log10(np.maximum(pool, 1.0)), "growth": pool / np.maximum(base, 1.0)}
    for i, nm in enumerate(("cos1", "gap15", "agree")):
        out[nm] = e["stats"][:, i]
    return out


def wquantile(v, w, p):
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return float(v[o][min(int(np.searchsorted(cw, p - 1e-12, side="left")), v.size - 1)])


def deciles(vals):
    """The declared breakpoint grid of a statistic: the nine pooled deciles with every dataset weighted equally (a number, the same for all seven)."""
    v = np.concatenate([vals[d] for d in SEVEN])
    w = np.concatenate([np.full(vals[d].size, 1.0 / vals[d].size) for d in SEVEN])
    return sorted({wquantile(v, w, p) for p in QUANTS})


# ── the programs ─────────────────────────────────────────────────────────────

def bin_tensors(U, tau, A, Z):
    """Per dataset and bin the covered count and the node sum at every level: shapes D x k x J. U[d]: the signed statistic of d's questions; tau: the sorted breakpoints."""
    k = len(tau) + 1
    As, Zs, Ns, Bs = [], [], [], []
    for d in SEVEN:
        b = (U[d][:, None] > np.asarray(tau, dtype=np.float64)[None, :]).sum(1) if tau else np.zeros(U[d].size, dtype=np.int64)
        B = np.zeros((U[d].size, k))
        B[np.arange(U[d].size), b] = 1.0
        As.append(B.T @ A[d].astype(np.float64))
        Zs.append(B.T @ Z[d])
        Ns.append(U[d].size)
        Bs.append(B.sum(0))
    return np.stack(As), np.stack(Zs), np.asarray(Ns, dtype=np.float64), np.stack(Bs)


def _mono(k):
    rows = []
    for b in range(k - 1):
        r = np.zeros(k * J + 1)
        r[b * J:(b + 1) * J] = np.arange(J)
        r[(b + 1) * J:(b + 2) * J] = -np.arange(J)
        rows.append(r)
    return rows


def solve_p1(As, Zs, Ns, rho=RHO, integer=True, monotone=True, wrong=False):
    """P1: minimise t (the worst dataset's mean nodes) with ALL_d >= rho for every dataset, one level per bin, the levels non-decreasing. wrong = True maximises the total nodes (a deliberately wrong
    objective, for the self-test). Returns (t, levels) or None when infeasible."""
    D, k, _ = As.shape
    n = k * J + 1
    c = np.zeros(n)
    c[-1] = 1.0
    c[:-1] = EPS * (Zs / Ns[:, None, None]).sum(0).reshape(-1)
    if wrong:                                                       # maximise the total nodes instead: a deliberately wrong objective
        c[-1] = 0.0
        c[:-1] = -(Zs / Ns[:, None, None]).sum(0).reshape(-1)
    rows, lo, hi = [], [], []
    for b in range(k):
        r = np.zeros(n)
        r[b * J:(b + 1) * J] = 1.0
        rows.append(r)
        lo.append(1.0)
        hi.append(1.0)
    for d in range(D):
        r = np.zeros(n)
        r[:-1] = As[d].reshape(-1)
        rows.append(r)
        lo.append(rho * Ns[d] - TOL)
        hi.append(np.inf)
    for d in range(D):
        r = np.zeros(n)
        r[:-1] = Zs[d].reshape(-1)
        r[-1] = -Ns[d]
        rows.append(r)
        lo.append(-np.inf)
        hi.append(0.0)
    if monotone:
        for r in _mono(k):
            rows.append(r)
            lo.append(-np.inf)
            hi.append(0.0)
    integ = np.zeros(n)
    integ[:-1] = 1.0 if integer else 0.0
    ub = np.full(n, np.inf)
    ub[:-1] = 1.0
    res = milp(c, constraints=LinearConstraint(np.asarray(rows), lo, hi), integrality=integ, bounds=Bounds(np.zeros(n), ub))
    if res.status != 0 or res.x is None:
        return None
    y = res.x[:-1].reshape(k, J)
    lev = y.argmax(1) if integer else None
    t = float(res.x[-1])
    if wrong:
        t = float(max(sum(Zs[d, b, lev[b]] for b in range(k)) / Ns[d] for d in range(D)))
    return {"t": t, "levels": [int(x) for x in lev] if lev is not None else None, "y": y}


def solve_p2(As, Zs, Ns, cap, integer=True, monotone=True):
    """P2: maximise r (the worst dataset's ALL) with mean nodes <= cap on every dataset. Returns (r, levels) or None when infeasible."""
    D, k, _ = As.shape
    n = k * J + 1
    c = np.zeros(n)
    c[-1] = -1.0
    c[:-1] = -EPS * (As / Ns[:, None, None]).sum(0).reshape(-1)
    rows, lo, hi = [], [], []
    for b in range(k):
        r = np.zeros(n)
        r[b * J:(b + 1) * J] = 1.0
        rows.append(r)
        lo.append(1.0)
        hi.append(1.0)
    for d in range(D):
        r = np.zeros(n)
        r[:-1] = As[d].reshape(-1)
        r[-1] = -Ns[d]
        rows.append(r)
        lo.append(0.0)
        hi.append(np.inf)
    for d in range(D):
        r = np.zeros(n)
        r[:-1] = Zs[d].reshape(-1)
        rows.append(r)
        lo.append(-np.inf)
        hi.append(cap * Ns[d] + TOL)
    if monotone:
        for r in _mono(k):
            rows.append(r)
            lo.append(-np.inf)
            hi.append(0.0)
    integ = np.zeros(n)
    integ[:-1] = 1.0 if integer else 0.0
    ub = np.full(n, np.inf)
    ub[:-1] = 1.0
    ub[-1] = 1.0
    res = milp(c, constraints=LinearConstraint(np.asarray(rows), lo, hi), integrality=integ, bounds=Bounds(np.zeros(n), ub))
    if res.status != 0 or res.x is None:
        return None
    y = res.x[:-1].reshape(k, J)
    return {"r": float(res.x[-1]), "levels": [int(x) for x in y.argmax(1)] if integer else None, "y": y}


def measure(As, Zs, Ns, levels):
    """Per dataset ALL and mean nodes of a level assignment (one level per bin)."""
    k = As.shape[1]
    alls = np.asarray([sum(As[d, b, levels[b]] for b in range(k)) / Ns[d] for d in range(As.shape[0])])
    nodes = np.asarray([sum(Zs[d, b, levels[b]] for b in range(k)) / Ns[d] for d in range(As.shape[0])])
    return alls, nodes


def candidates(vals_by_stat, ks=KS):
    """Every declared policy: (statistic, sign, k, breakpoints). Breakpoints are subsets of the pooled deciles of the statistic (on the signed scale)."""
    for s in STATS:
        grid = deciles(vals_by_stat[s])
        for sg in SIGNS:
            g = sorted({sg * x for x in grid})
            for k in ks:
                for tau in itertools.combinations(g, k - 1):
                    yield s, sg, k, tau


def search_p1(vals, A, Z, shuffle=None, ks=KS, rho=RHO):
    """Exhaustive over the declared class. The LP relaxation prunes a candidate that cannot beat the best so far (valid for a minimum). Returns the best per (statistic, sign, k) and overall."""
    best_by = {}
    overall = None
    n_solved = n_pruned = 0
    U_cache = {}
    for s, sg, k, tau in candidates(vals, ks):
        key = (s, sg)
        if key not in U_cache:
            U_cache[key] = {d: sg * (vals[s][d] if shuffle is None else vals[s][d][shuffle[d]]) for d in SEVEN}
        As, Zs, Ns, Bs = bin_tensors(U_cache[key], tau, A, Z)
        cur = best_by.get((s, sg, k))
        lb = solve_p1(As, Zs, Ns, rho, integer=False)
        if lb is None:
            continue
        if cur is not None and lb["t"] >= cur["t"] - 1e-9:
            n_pruned += 1
            continue
        r = solve_p1(As, Zs, Ns, rho, integer=True)
        n_solved += 1
        if r is None:
            continue
        rec = {"stat": s, "sign": sg, "k": k, "tau": [float(x) for x in tau], "levels": r["levels"], "t": r["t"], "lp_t": lb["t"]}
        if cur is None or r["t"] < cur["t"] - 1e-9:
            best_by[(s, sg, k)] = rec
        if overall is None or r["t"] < overall["t"] - 1e-9:
            overall = rec
    return best_by, overall, n_solved, n_pruned


def choose(best_by):
    """The declared choice: over every (statistic, sign) the smallest k whose t is within PARSIMONY of the best t over k <= 4; then the least t."""
    t_best = min(r["t"] for r in best_by.values())
    ok = [r for r in best_by.values() if r["t"] <= PARSIMONY * t_best]
    kmin = min(r["k"] for r in ok)
    return min((r for r in ok if r["k"] == kmin), key=lambda r: (r["t"], r["stat"], -r["sign"]))


def uniform(A, Z, rho=RHO):
    """k = 1 by enumeration: the one level of the ladder every dataset takes. (worst dataset's nodes, level, per-dataset nodes, per-dataset ALL), or None."""
    best = None
    for j in range(J):
        alls = np.asarray([A[d][:, j].mean() for d in SEVEN])
        nodes = np.asarray([Z[d][:, j].mean() for d in SEVEN])
        if (alls >= rho - TOL).all():
            cand = (float(nodes.max()), j, nodes, alls)
            if best is None or cand[0] < best[0]:
                best = cand
    return best


# ── the oracle on the same ladder ────────────────────────────────────────────

def oracle(A, Z, rho=RHO):
    import zul4lp as L4
    out = {}
    for d in SEVEN:
        n0, a0, nstar, delta = L4.depth_core(A[d].astype(bool), Z[d])
        m, got = L4.q_min(n0, a0, delta, rho)
        out[d] = None if m is None else float(m)
    return out


# ── self-test ────────────────────────────────────────────────────────────────

def selftest():
    rng = np.random.default_rng(7)
    ok = {}
    D = len(SEVEN)
    # a toy instance: D datasets, 40 questions, J levels with nested pools; a signal that is informative in half of them
    A, Z, V = {}, {}, {}
    for i, d in enumerate(SEVEN):
        q = 40
        need = rng.integers(0, J, size=q)
        sizes = np.cumsum(rng.integers(5, 30, size=(q, J)), axis=1).astype(np.float64)
        A[d] = (np.arange(J)[None, :] >= need[:, None]).astype(np.int64)
        Z[d] = sizes
        V[d] = need + rng.normal(0, 0.5 if i % 2 == 0 else 6.0, size=q)
    # P1 against brute force over every monotone assignment, k = 2 and 3, a few breakpoints
    bad = caught = 0
    for tau in [(float(np.median(np.concatenate(list(V.values())))),), (3.0, 8.0), (-1.0, 5.0)]:
        As, Zs, Ns, Bs = bin_tensors(V, tau, A, Z)
        k = As.shape[1]
        for rho in (0.6, 0.8, 0.95):
            r = solve_p1(As, Zs, Ns, rho, integer=True)
            best = None
            for lev in itertools.product(range(J), repeat=k):
                if any(lev[b] > lev[b + 1] for b in range(k - 1)):
                    continue
                alls, nodes = measure(As, Zs, Ns, lev)
                if (alls >= rho - TOL).all():
                    t = float(nodes.max())
                    if best is None or t < best - 1e-9:
                        best = t
            if (r is None) != (best is None) or (r is not None and abs(r["t"] - best) > 1e-3):
                bad += 1
            # the relaxation is a lower bound and a wrong objective is caught
            if r is not None:
                lb = solve_p1(As, Zs, Ns, rho, integer=False)
                if lb["t"] > r["t"] + 1e-6:
                    bad += 1
                wrong = solve_p1(As, Zs, Ns, rho, integer=True, wrong=True)
                if wrong is not None and wrong["t"] > r["t"] + 1e-6:
                    caught += 1
    ok["P1_equals_brute_force"] = bad == 0
    ok["wrong_objective_caught"] = caught > 0                  # maximising t instead of minimising it gives a different, larger t on at least one instance
    # monotone constraint binds: dropping it can only lower t, and a toy where it matters lowers it strictly or ties
    As, Zs, Ns, Bs = bin_tensors(V, (3.0, 8.0), A, Z)
    a = solve_p1(As, Zs, Ns, 0.9, integer=True)
    b = solve_p1(As, Zs, Ns, 0.9, integer=True, monotone=False)
    ok["monotone_cannot_lower"] = bool(a is None or b is None or b["t"] <= a["t"] + 1e-9)
    ok["levels_monotone"] = bool(a is None or all(a["levels"][i] <= a["levels"][i + 1] for i in range(len(a["levels"]) - 1)))
    # P2 against brute force
    bad = 0
    for cap in (60.0, 120.0, 250.0):
        r = solve_p2(As, Zs, Ns, cap, integer=True)
        k = As.shape[1]
        best = None
        for lev in itertools.product(range(J), repeat=k):
            if any(lev[b] > lev[b + 1] for b in range(k - 1)):
                continue
            alls, nodes = measure(As, Zs, Ns, lev)
            if (nodes <= cap + 1e-9).all():
                v = float(alls.min())
                if best is None or v > best + 1e-9:
                    best = v
        if (r is None) != (best is None) or (r is not None and abs(r["r"] - best) > 1e-3):
            bad += 1
    ok["P2_equals_brute_force"] = bad == 0
    # k = 1 by enumeration equals P1 with one bin
    As1, Zs1, Ns1, _ = bin_tensors(V, (), A, Z)
    u = uniform(A, Z, 0.8)
    p1 = solve_p1(As1, Zs1, Ns1, 0.8, integer=True)
    ok["uniform_equals_P1_k1"] = bool((u is None and p1 is None) or (u is not None and p1 is not None and abs(u[0] - p1["t"]) < 1e-3))
    # the pooled deciles are a number: a within-dataset shuffle leaves them unchanged
    vals = {"x": {d: V[d] for d in SEVEN}}
    sh = {d: rng.permutation(V[d].size) for d in SEVEN}
    ok["deciles_invariant_to_shuffle"] = deciles({d: V[d][sh[d]] for d in SEVEN}) == deciles(V)
    # the search finds the planted signal: with an informative statistic the best t is below the uniform's
    vals = {s: {d: V[d] for d in SEVEN} for s in STATS}
    best_by, overall, ns, npn = search_p1(vals, A, Z, ks=(2, 3), rho=0.8)
    u = uniform(A, Z, 0.8)
    ok["search_beats_uniform_on_a_planted_signal"] = bool(overall is not None and (u is None or overall["t"] <= u[0] + 1e-9))
    ok["choose_is_parsimonious"] = bool(overall is None or choose(best_by)["k"] <= overall["k"])
    ok["verdict"] = "PASS" if all(v for k, v in ok.items() if k != "verdict") else "FAIL"
    ok.update({"script_sha256": script_sha256(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selftest.json").write_text(json.dumps(ok, indent=1), encoding="utf-8")
    log(f"selftest {ok['verdict']}: {ok}")
    return 0 if ok["verdict"] == "PASS" else 1


# ── gates (data identity with the filed grids) ───────────────────────────────

def cmd_gates(a=None):
    import zul4lp as L4
    out = {"stage": "UL5 gates", "declared_in": DECLARED, "datasets": {}}
    ok_all = True
    for d in SEVEN:
        e = load_ext(d, "dev")
        names, present, size, gold, qids = L4.load_dev_grids(d)
        same_q = bool(np.array_equal(qids, e["qids"]) and np.array_equal(gold, e["gold"]))
        shared = [n for n in e["names"] if n in names]
        eq = []
        for n in shared:
            i, j = e["names"].index(n), names.index(n)
            eq.append(bool(np.array_equal(present[:, j], e["present"][:, i]) and np.array_equal(size[:, j], e["size"][:, i])))
        nest = bool(len(e["names"]) < 2 or ((np.diff(e["present"], axis=1) >= 0).all() and (np.diff(e["size"], axis=1) >= 0).all()))
        heads = []
        for f in sorted((OUT / d).glob("ext_dev*.json")):
            heads.append(json.loads(f.read_text(encoding="utf-8")))
        shas = {h["code_sha256"] for h in heads}
        res = {"questions": int(e["gold"].size), "same_questions_and_golds": same_q, "shared_cells": shared, "shared_equal": eq, "nested": nest, "one_code_sha": len(shas) == 1,
               "cells_sha": sorted({h["cells_sha256"] for h in heads})}
        res["pass"] = bool(same_q and shared and all(eq) and nest and len(shas) == 1)
        ok_all &= res["pass"]
        out["datasets"][d] = res
        log(f"G5-1 {d}: {'PASS' if res['pass'] else 'FAIL'} ({len(shared)} shared cells equal: {sum(eq)}; questions {res['questions']})")
    out["verdict"] = "PASS" if ok_all else "FAIL"
    (OUT / "gates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0 if ok_all else 1


# ── dev solve ────────────────────────────────────────────────────────────────

def load_all(carve):
    E, A, Z, VALS = {}, {}, {}, {s: {} for s in STATS}
    for d in SEVEN:
        E[d] = load_ext(d, carve)
        A[d], Z[d] = ladder_arrays(E[d])
        for s, v in statistics(E[d]).items():
            VALS[s][d] = v
    return E, A, Z, VALS


def cmd_solve(a=None):
    t0 = time.time()
    g = json.loads((OUT / "gates.json").read_text(encoding="utf-8")) if (OUT / "gates.json").is_file() else {}
    st = json.loads((OUT / "selftest.json").read_text(encoding="utf-8")) if (OUT / "selftest.json").is_file() else {}
    if g.get("verdict") != "PASS" or st.get("verdict") != "PASS" or st.get("script_sha256") != script_sha256():
        stop("gates.json and selftest.json must both be PASS for this script before the solve (run `selftest` and `gates`)")
    E, A, Z, VALS = load_all("dev")
    res = {"stage": "UL5 dev solve", "declared_in": DECLARED, "script_sha256": script_sha256(), "ladder_m": [m for m in LADDER_M], "stats": list(STATS),
           "quantiles": list(QUANTS), "rho": RHO, "datasets": {d: int(E[d]["gold"].size) for d in SEVEN}}
    uni = uniform(A, Z)
    res["uniform"] = None if uni is None else {"t": uni[0], "level": uni[1], "m": LADDER_M[uni[1]], "nodes": dict(zip(SEVEN, map(float, uni[2]))), "all": dict(zip(SEVEN, map(float, uni[3])))}
    res["oracle_min_nodes"] = oracle(A, Z)
    res["oracle_worst"] = max(v for v in res["oracle_min_nodes"].values() if v is not None) if all(v is not None for v in res["oracle_min_nodes"].values()) else None
    best_by, overall, ns, npn = search_p1(VALS, A, Z)
    res["p1"] = {"best_by": [r for r in best_by.values()], "overall": overall, "milp_solved": ns, "pruned_by_lp": npn}
    if not best_by:
        res["call"] = "INFEASIBLE"
        (OUT / "dev_solve.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 1
    pick = choose(best_by)
    res["chosen"] = pick
    # the chosen policy's dev measures and the programs beside it
    U = {d: pick["sign"] * VALS[pick["stat"]][d] for d in SEVEN}
    As, Zs, Ns, Bs = bin_tensors(U, tuple(pick["tau"]), A, Z)
    alls, nodes = measure(As, Zs, Ns, pick["levels"])
    res["chosen_dev"] = {"all": dict(zip(SEVEN, map(float, alls))), "nodes": dict(zip(SEVEN, map(float, nodes))), "t": float(nodes.max()),
                         "bin_share": {d: [float(x) for x in (Bs[i] / Ns[i])] for i, d in enumerate(SEVEN)}, "levels_m": [LADDER_M[j] for j in pick["levels"]]}
    if uni is not None:
        phi = {}
        for i, d in enumerate(SEVEN):
            o = res["oracle_min_nodes"][d]
            den = uni[2][i] - o if o is not None else 0.0
            phi[d] = None if den <= 1.0 else float((uni[2][i] - nodes[i]) / den)
        res["phi"] = phi
        res["saving_vs_uniform"] = float(1.0 - pick["t"] / uni[0])
        res["best_saving_vs_uniform"] = float(1.0 - overall["t"] / uni[0])
    # P1 without WebQSP beside (six datasets), P2 under the caps, LP relaxation of the chosen
    lbp = solve_p1(As, Zs, Ns, RHO, integer=False)
    res["chosen_lp_t"] = None if lbp is None else lbp["t"]
    res["p2"] = {}
    for cap in CAPS:
        best = None
        for s, sg, k, tau in candidates(VALS):
            U2 = {d: sg * VALS[s][d] for d in SEVEN}
            As2, Zs2, Ns2, _ = bin_tensors(U2, tau, A, Z)
            r = solve_p2(As2, Zs2, Ns2, cap, integer=True)
            if r is not None and (best is None or r["r"] > best["r"] + 1e-9):
                best = {"stat": s, "sign": sg, "k": k, "tau": [float(x) for x in tau], "levels": r["levels"], "r": r["r"]}
        res["p2"][str(cap)] = best
    ures = None
    for cap in CAPS:
        rbest = None
        for j in range(J):
            nodes_u = np.asarray([Z[d][:, j].mean() for d in SEVEN])
            alls_u = np.asarray([A[d][:, j].mean() for d in SEVEN])
            if (nodes_u <= cap + 1e-9).all() and (rbest is None or alls_u.min() > rbest["r"]):
                rbest = {"level": j, "m": LADDER_M[j], "r": float(alls_u.min()), "all": dict(zip(SEVEN, map(float, alls_u)))}
        res["p2"][f"uniform_{cap}"] = rbest
    # the null: the statistic shuffled within every dataset, the whole search repeated
    rng = np.random.default_rng(NULL_SEED)
    nulls = []
    for i in range(NULLS):
        sh = {d: rng.permutation(E[d]["gold"].size) for d in SEVEN}
        _bb, ov, _a, _b = search_p1(VALS, A, Z, shuffle=sh)
        nulls.append(None if ov is None else ov["t"])
        log(f"null {i + 1}/{NULLS}: best t {nulls[-1]}")
    res["null_best_t"] = nulls
    ok_n = [x for x in nulls if x is not None]
    res["beats_every_null"] = bool(uni is not None and overall is not None and ok_n and overall["t"] < min(ok_n) - 1e-9)
    res["null_saving"] = None if (uni is None or not ok_n) else {"min": float(1 - max(ok_n) / uni[0]), "mean": float(1 - np.mean(ok_n) / uni[0]), "max": float(1 - min(ok_n) / uni[0])}
    # the call
    sv = res.get("saving_vs_uniform")
    if uni is None or sv is None or sv < SAVE_MIN or not res["beats_every_null"]:
        res["call"] = "NO_SIGNAL"
    elif pick["t"] <= BOUND:
        res["call"] = "SIGNAL_AT_BOUND"
    else:
        res["call"] = "SIGNAL_PARTIAL"
    res["seconds"] = round(time.time() - t0, 1)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "dev_solve.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"dev call {res['call']}: uniform {None if uni is None else round(uni[0])}, chosen {round(pick['t'])} ({pick['stat']}, sign {pick['sign']}, k {pick['k']}), best {round(overall['t'])}, nulls {ok_n[:3]}...")
    return 0


# ── eval (once) ──────────────────────────────────────────────────────────────

def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def cmd_grade(a=None):
    ds = json.loads((OUT / "dev_solve.json").read_text(encoding="utf-8"))
    if (OUT / "eval_grade.json").is_file():
        stop("eval_grade.json exists: the eval carve is graded once")
    if ds["call"] == "NO_SIGNAL":
        stop("the dev call is NO_SIGNAL: the eval carve is not graded (docs section 8)")
    pick = ds["chosen"]
    E, A, Z, VALS = load_all("eval")
    U = {d: pick["sign"] * VALS[pick["stat"]][d] for d in SEVEN}
    As, Zs, Ns, Bs = bin_tensors(U, tuple(pick["tau"]), A, Z)
    alls, nodes = measure(As, Zs, Ns, pick["levels"])
    uni = ds["uniform"]
    out = {"stage": "UL5 eval", "declared_in": DECLARED, "script_sha256": script_sha256(), "policy": pick, "datasets": {}}
    for i, d in enumerate(SEVEN):
        n = int(Ns[i])
        k = int(round(alls[i] * n))
        lo, hi = wilson(k, n)
        uj = uni["level"]
        out["datasets"][d] = {"questions": n, "all": float(alls[i]), "wilson": [lo, hi], "nodes": float(nodes[i]), "uniform_all": float(A[d][:, uj].mean()), "uniform_nodes": float(Z[d][:, uj].mean()),
                              "dev_all": ds["chosen_dev"]["all"][d], "dev_nodes": ds["chosen_dev"]["nodes"][d]}
    out["worst_nodes"] = float(nodes.max())
    out["min_all"] = float(alls.min())
    ok_all = bool((alls >= RHO - 0.02 - 1e-12).all())
    ok_nodes = bool(nodes.max() <= 1.10 * ds["chosen_dev"]["t"] + 1e-9)
    out["eval_holds"] = bool(ok_all and ok_nodes)
    out["at_bound"] = bool(nodes.max() <= BOUND)
    out["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "eval_grade.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"eval: min ALL {out['min_all']:.4f}, worst nodes {out['worst_nodes']:.0f}, holds {out['eval_holds']}, at bound {out['at_bound']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("selftest", "gates", "solve", "grade"))
    a = ap.parse_args(argv)
    return {"selftest": lambda x: selftest(), "gates": cmd_gates, "solve": cmd_solve, "grade": cmd_grade}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
