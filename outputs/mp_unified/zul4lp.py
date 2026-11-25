"""UL4 (docs/UL4_UNIVERSAL_LP.md): one universal rule as a linear program. Two layers, solved apart, laptop only, numpy and scipy only.

  L  (links): one keep-fraction per SHARED cell key (UL3's C and PC cells, the same keys in every dataset), minimising the worst judged cell's price at
     a reach target; S never removed (A.1) is the call, S free beside. Selected on the dev columns, the integer rule graded once on eval's columns.
  Q  (depth): per question, one level of a nested ladder of pools (PL6's U95 pick's base+walk and its cuts), minimising mean nodes at ALL >= rho or
     maximising ALL at a mean-node cap. Dev only. An oracle bound (it reads the golds), never a method.

    python outputs/mp_unified/zul4lp.py selftest
    python outputs/mp_unified/zul4lp.py lp-links
    python outputs/mp_unified/zul4lp.py lp-depth [--ds D]
    python outputs/mp_unified/zul4lp.py report

Reads (never edits): outputs/ul3/<d>/cells.json, outputs/pl<2..7>/<d>/grid_dev_frozen*.npz and outputs/pl6/<d>/select.json. Writes outputs/ul4/ only.
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
OUT = ROOT / "outputs" / "ul4"
UL3 = ROOT / "outputs" / "ul3"
DECLARED = "docs/UL4_UNIVERSAL_LP.md"

DS6 = ("multihoprag", "musique", "metaqa", "webqsp", "hotpotqa", "2wiki")
DS7 = ("multihoprag", "squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp")
JUDGED = (("multihoprag", "RK", "GP"), ("musique", "RK", "GP"), ("musique", "RK", "CH"), ("metaqa", "SRK", "GP"), ("webqsp", "SRK", "GP"),
          ("hotpotqa", "RK", "GP"), ("hotpotqa", "SRK", "GP"), ("2wiki", "RK", "GP"), ("2wiki", "SRK", "GP"))
SPACES = {"K1": "C", "K2": "PC"}
RHOS = (0.80, 0.90, 0.95, 0.99)
CALL_RHO = 0.95
R_MIN, TAU_MAX = 0.92, 0.50                         # UL3's thresholds, fixed before
TAUS = (0.30, 0.40, 0.50, 0.60, 0.80)
CAPS = (2500, 3500, 5000, 7000, 10000)
ALL_RHOS = (0.95, 0.98)
FAMILY = {                                          # PL6's U95 picks, base+walk without the cut (docs/UL4_UNIVERSAL_LP.md section 4)
    "multihoprag": "rrf1000_b25+FULL:h1_c100",
    "squad": "rrf1000_b15",
    "hotpotqa": "rrf1000_b25+STRUCT:h1_c100~cs",
    "2wiki": "rrf1000_b400+FULL:h1_c100~s20",
    "musique": "rrf1000_b400+FULL:h3_bf_v20000~cs-fu",
    "metaqa": "rrf1000_b25+STRUCT:h3_c100_v5000",
    "webqsp": "rrf1000_b400+FULL:h3_bf_v40000~cs-vr",
}


def stop(msg):
    raise SystemExit(f"[zul4lp] STOP: {msg}")


def log(msg):
    print(f"[zul4lp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── layer L: the shared-key link programs ────────────────────────────────────


def key_of(cell):
    return f"{cell['class']}|{cell['bin']}"


def is_s(key):
    return "S" in key.split("|")[0].split("+")


class Space:
    """The judged cells' columns over one shared key space: cov/cand on 'sel' and 'eval' (J x m), the any-link totals (J,), the S-key mask."""

    def __init__(self, recs, variant):
        self.js = list(recs)
        keys = sorted({key_of(c) for kd in recs.values() for c in kd[variant]["cells"].values()})
        self.keys, kidx = keys, {k: i for i, k in enumerate(keys)}
        J, m = len(self.js), len(keys)
        self.cov = {s: np.zeros((J, m)) for s in ("sel", "eval")}
        self.cand = {s: np.zeros((J, m)) for s in ("sel", "eval")}
        self.tcov = {s: np.zeros(J) for s in ("sel", "eval")}
        self.tcand = {s: np.zeros(J) for s in ("sel", "eval")}
        for j, jk in enumerate(self.js):
            kd = recs[jk]
            for c in kd[variant]["cells"].values():
                i = kidx[key_of(c)]
                for s in ("sel", "eval"):
                    col = c.get(s)
                    if col:
                        self.cov[s][j, i] += col["covered"]
                        self.cand[s][j, i] += col["candidates"]
            for s in ("sel", "eval"):
                self.tcov[s][j], self.tcand[s][j] = kd["any"][s]["covered"], kd["any"][s]["candidates"]
                if int(self.cov[s][j].sum()) != int(self.tcov[s][j]) or int(self.cand[s][j].sum()) != int(self.tcand[s][j]):
                    stop(f"G4-1: {jk} {variant} {s}: the keys do not sum to the any-link totals")   # G4-1
        self.smask = np.array([is_s(k) for k in keys])

    def coef(self, s):
        """R[j, i] (the reach share one unit of key i buys cell j) and P[j, i] (its price share), on carve s."""
        return self.cov[s] / self.tcov[s][:, None], self.cand[s] / self.tcand[s][:, None]


def _bounds(m, protect_mask, extra=1):
    lb, ub = np.zeros(m + extra), np.ones(m + extra)
    if protect_mask is not None:
        lb[:m][protect_mask] = 1.0
    return lb, ub


def solve_u(R, P, rho, protect_mask=None, integer=False):
    """min tau s.t. R x >= rho, P x <= tau, 0 <= x <= 1 (x in {0,1} when integer). Returns tau, x, the reach duals (<= 0 where binding)."""
    J, m = R.shape
    c = np.zeros(m + 1)
    c[-1] = 1.0
    A = np.zeros((2 * J, m + 1))
    b = np.zeros(2 * J)
    A[:J, :m], b[:J] = -R, -rho
    A[J:, :m], A[J:, -1] = P, -1.0
    lb, ub = _bounds(m, protect_mask)
    if integer:
        res = milp(c, constraints=LinearConstraint(A, -np.inf, b), bounds=Bounds(lb, ub), integrality=np.r_[np.ones(m), 0.0])
        if res.x is None:
            stop(f"the integer program has no solution at rho {rho}")
        return float(res.x[-1]), np.rint(res.x[:m]), None
    res = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lb, ub)), method="highs")
    if res.status != 0:
        stop(f"the linear program failed at rho {rho}: {res.message}")
    return float(res.x[-1]), res.x[:m].copy(), np.asarray(res.ineqlin.marginals)[:J].copy()


def solve_maxmin(R, P, tau, protect_mask=None):
    """max r s.t. R x >= r, P x <= tau (the cost in reach of a price)."""
    J, m = R.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0
    A = np.zeros((2 * J, m + 1))
    b = np.zeros(2 * J)
    A[:J, :m], A[:J, -1] = -R, 1.0
    A[J:, :m], b[J:] = P, tau
    lb, ub = _bounds(m, protect_mask)
    res = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lb, ub)), method="highs")
    if res.status != 0:
        return None
    return float(res.x[-1]), res.x[:m].copy()


def grade_x(sp, x, carve):
    R, P = sp.coef(carve)
    return R @ x, P @ x


def load_links():
    recs = {}
    for d in DS6:
        r = json.loads((UL3 / d / "cells.json").read_text(encoding="utf-8"))
        for (dd, ver, kind) in JUDGED:
            if dd == d:
                recs[(dd, ver, kind)] = r["versions"][ver]["kinds"][kind]
        recs_sha = r.get("zul3a_sha256")
    return recs, recs_sha


def lp_links():
    recs, _sha = load_links()
    out = {"stage": "UL4 links", "declared_in": DECLARED, "judged": ["/".join(j) for j in recs], "spaces": {}}
    for sname, variant in SPACES.items():
        sp = Space(recs, variant)
        so = {"keys": sp.keys, "s_keys": [k for k, f in zip(sp.keys, sp.smask) if f], "prot": {}}
        for pname, pmask in (("A1", sp.smask), ("S_free", None)):
            R, P = sp.coef("sel")
            po = {"rho": {}}
            for rho in sorted(set(RHOS) | {0.94, 0.96}):
                tau, x, dual = solve_u(R, P, rho, pmask)
                tj = []
                for j in range(len(sp.js)):
                    tj.append(solve_u(R[j:j + 1], P[j:j + 1], rho, pmask)[0])
                rr = R @ x
                pp = P @ x
                po["rho"][str(rho)] = {"tau": tau, "own_tau": tj, "cost_of_universality": tau - max(tj), "binding_reach": [bool(abs(d_) > 1e-9) for d_ in dual],
                                       "dev_reach": rr.tolist(), "dev_price": pp.tolist(), "x": x.tolist()}
            R2, P2 = sp.coef("sel")
            po["frontier"] = {}
            for tau in TAUS:
                f = solve_maxmin(R2, P2, tau, pmask)
                po["frontier"][str(tau)] = None if f is None else {"max_min_reach": f[0], "x": f[1].tolist()}
            tau_i, xi, _ = solve_u(R, P, CALL_RHO, pmask, integer=True)
            er, ep = grade_x(sp, xi, "eval")
            dr, dp = grade_x(sp, xi, "sel")
            ok = [(er[j] >= R_MIN and ep[j] <= TAU_MAX) for j in range(len(sp.js))]
            po["call"] = {"tau_dev": tau_i, "x": xi.tolist(), "kept_keys": [k for k, v in zip(sp.keys, xi) if v > 0.5],
                          "dev_reach": dr.tolist(), "dev_price": dp.tolist(), "eval_reach": er.tolist(), "eval_price": ep.tolist(),
                          "screen_in": [bool(o) for o in ok], "verdict": "UNIVERSAL_EXISTS" if all(ok) else "NOT_EXISTS"}
            so["prot"][pname] = po
        Rs, Ps = sp.coef("sel")
        so["s_floor_price_dev"] = (Ps @ sp.smask.astype(float)).tolist()
        so["s_floor_reach_dev"] = (Rs @ sp.smask.astype(float)).tolist()
        Re, Pe = sp.coef("eval")
        so["s_floor_price_eval"] = (Pe @ sp.smask.astype(float)).tolist()
        # G4-2: the fractional own optimum is at most UL3's filed integer keep-set price on dev (S free, C and PC, rho .95)
        g42 = []
        for j, jk in enumerate(sp.js):
            filed = recs[jk][variant]["kept"]["0.95"]["sel"]["price_retained"]["pt"]
            own = so["prot"]["S_free"]["rho"]["0.95"]["own_tau"][j]
            if own > filed + 1e-6:
                stop(f"G4-2: {jk} {variant}: the fractional optimum {own} exceeds UL3's integer keep-set price {filed}")
            g42.append({"cell": "/".join(jk), "lp_own": own, "filed_integer": filed})
        so["g4_2"] = g42
        out["spaces"][sname] = so
        log(f"links {sname}: {len(sp.keys)} keys; A1 tau(.95) {so['prot']['A1']['rho']['0.95']['tau']:.3f}, S free {so['prot']['S_free']['rho']['0.95']['tau']:.3f}; "
            f"call A1 {so['prot']['A1']['call']['verdict']}")
    out["script_sha256"] = script_sha256()
    return out


# ── layer Q: depth allocation over a nested ladder ───────────────────────────


def load_dev_grids(d, want=None):
    """The union of PL2 to PL7's dev grids on the laptop: names, present (Q x K int16), size (Q x K int32), gold counts, qids. Part files of one grid are
    stacked on questions; a cell held by two grids must be equal element for element (G4-3)."""
    cells = {}
    gold = qids = None
    for rd in range(2, 8):
        files = [f for f in sorted((ROOT / "outputs" / f"pl{rd}" / d).glob("grid_dev_frozen*.npz")) if "smoke" not in f.name]
        groups = {}
        for f in files:
            groups.setdefault(re.sub(r"\.part\d+of\d+", "", f.name), []).append(f)
        for stem, fs in groups.items():
            zs = [np.load(f, allow_pickle=True) for f in fs]
            zs.sort(key=lambda z: int(z["lo"]))
            if int(zs[0]["lo"]) != 0 or any(int(a["hi"]) != int(b["lo"]) for a, b in zip(zs, zs[1:])):
                stop(f"G4-3: {d} pl{rd} {stem}: the parts do not tile the dev questions")
            names = [str(x) for x in zs[0]["names"]]
            if any([str(x) for x in z["names"]] != names for z in zs):
                stop(f"G4-3: {d} pl{rd} {stem}: the parts hold different cells")
            q = np.concatenate([z["qids"] for z in zs])
            g = np.concatenate([z["gold_counts"] for z in zs]).astype(np.int64)
            if qids is None:
                qids, gold = q, g
            elif not (np.array_equal(qids, q) and np.array_equal(gold, g)):
                stop(f"G4-3: {d} pl{rd} {stem}: question order or gold counts differ")
            pres = np.concatenate([z["present"] for z in zs], axis=0)
            size = np.concatenate([z["size"] for z in zs], axis=0)
            for k, nm in enumerate(names):
                if want is not None and nm not in want:
                    continue
                if nm in cells:
                    if not (np.array_equal(cells[nm][0], pres[:, k]) and np.array_equal(cells[nm][1], size[:, k])):
                        stop(f"G4-3: {d}: cell {nm} differs between two grids")
                else:
                    cells[nm] = (pres[:, k].copy(), size[:, k].copy())
    names = sorted(cells)
    present = np.stack([cells[n][0] for n in names], axis=1)
    size = np.stack([cells[n][1] for n in names], axis=1)
    return names, present, size, gold, qids


def select_table(d):
    s = json.loads((ROOT / "outputs" / "pl6" / d / "select.json").read_text(encoding="utf-8"))
    return {r["name"]: r for r in s["table"]}, s


def family_names(d, names):
    """The declared family (docs/UL4_UNIVERSAL_LP.md section 4): every grid cell named P or P+top{M}, P PL6's pick (a coarser first run took only the cells
    PL6's select table lists; the grids hold a denser ladder, so that run could not see below the table's smallest cut)."""
    rx = re.compile(re.escape(FAMILY[d]) + r"(\+top\d+)?$")
    fam = [n for n in names if rx.match(n)]
    if not fam:
        stop(f"{d}: no grid cell of the family {FAMILY[d]}")
    return fam


def depth_core(A, N):
    """Per question (rows of A: covering, N: nodes; both Q x K): the cheapest pool overall (n0, and whether it covers) and the cheapest covering pool
    (nstar, inf when none). The programs are fractional knapsacks over the upgrade costs delta = nstar - n0."""
    n0 = N.min(1).astype(np.float64)
    a0 = (A & (N == n0[:, None])).any(1)
    big = np.inf
    nstar = np.where(A, N, big).min(1).astype(np.float64)
    delta = np.where(a0, 0.0, nstar - n0)
    return n0, a0, nstar, delta


def q_min(n0, a0, delta, rho):
    """Least mean nodes at ALL >= rho: take the cheapest upgrades first, the last one fractional (the LP optimum). None when the family cannot reach rho."""
    Q = n0.size
    need = rho * Q - a0.sum()
    base = n0.sum()
    if need <= 1e-12:
        return base / Q, float(a0.mean())
    ups = np.sort(delta[(~a0) & np.isfinite(delta)])
    if need > ups.size + 1e-9:
        return None, None
    k = int(np.floor(need + 1e-12))
    frac = need - k
    extra = ups[:k].sum() + (ups[k] * frac if frac > 1e-12 else 0.0)
    return (base + extra) / Q, rho


def q_cap(n0, a0, delta, cap):
    """Most ALL at mean nodes <= cap (the LP optimum). None when the cheapest pools already exceed the cap."""
    Q = n0.size
    budget = cap * Q - n0.sum()
    if budget < -1e-9:
        return None
    ups = np.sort(delta[(~a0) & np.isfinite(delta)])
    cs = np.cumsum(ups)
    k = int(np.searchsorted(cs, budget, side="right"))
    got = float(k)
    if k < ups.size:
        got += (budget - (cs[k - 1] if k else 0.0)) / ups[k]
    return float((a0.sum() + min(got, ups.size)) / Q)


def hull_min_nodes(alls, nodes, rho):
    """Least mean nodes of a randomised uniform rule (a mix of two cells) at ALL >= rho: the lower-right hull of the cells' (ALL, nodes) points."""
    pts = sorted(set(zip(alls.tolist(), nodes.tolist())))
    best = None
    for (a1, n1) in pts:
        if a1 >= rho and (best is None or n1 < best):
            best = n1
    for (a1, n1), (a2, n2) in itertools.combinations(pts, 2):
        lo, hi = ((a1, n1), (a2, n2)) if a1 < a2 else ((a2, n2), (a1, n1))
        if lo[0] < rho <= hi[0] and hi[0] > lo[0]:
            t = (rho - lo[0]) / (hi[0] - lo[0])
            v = lo[1] + t * (hi[1] - lo[1])
            if best is None or v < best:
                best = v
    return best


def hull_cap_all(alls, nodes, cap):
    pts = sorted(set(zip(alls.tolist(), nodes.tolist())))
    best = max([a for a, n in pts if n <= cap], default=None)
    for (a1, n1), (a2, n2) in itertools.combinations(pts, 2):
        if n1 == n2:
            continue
        lo, hi = ((a1, n1), (a2, n2)) if n1 < n2 else ((a2, n2), (a1, n1))
        if lo[1] <= cap < hi[1]:
            t = (cap - lo[1]) / (hi[1] - lo[1])
            v = lo[0] + t * (hi[0] - lo[0])
            if best is None or v > best:
                best = v
    return best


def programs(A, N, label):
    n0, a0, nstar, delta = depth_core(A, N)
    Q = n0.size
    o = {"label": label, "questions": int(Q), "cells": int(A.shape[1]), "cheapest_pool_mean": float(n0.mean()), "all_at_cheapest": float(a0.mean()),
         "ceiling_all": float(np.isfinite(nstar).mean())}
    for rho in sorted(set(ALL_RHOS) | {0.94, 0.96}):
        nodes, got = q_min(n0, a0, delta, rho)
        o[f"min_nodes@{rho}"] = nodes
    for cap in CAPS:
        o[f"all@cap{cap}"] = q_cap(n0, a0, delta, cap)
    o["share_cover_within"] = {str(t): float((nstar <= t).mean()) for t in CAPS}
    o["unreachable_share"] = float((~np.isfinite(nstar)).mean())
    m94, m96 = o["min_nodes@0.94"], o["min_nodes@0.96"]
    o["nodes_per_point_at_0.95"] = None if (m94 is None or m96 is None) else (m96 - m94) / 2.0
    return o


def lp_depth(ds):
    out = {"stage": "UL4 depth", "declared_in": DECLARED, "datasets": {}}
    for d in ds:
        t0 = time.time()
        table, sel = select_table(d)
        names, pres, size, gold, qids = load_dev_grids(d)
        fam = family_names(d, names)
        if int(sel["questions"]) != int(qids.size):
            stop(f"G4-3: {d}: {qids.size} dev questions in the grids, {sel['questions']} in PL6's select")
        valid = gold > 0
        idx = {n: i for i, n in enumerate(names)}
        miss = [n for n in table if (n == FAMILY[d] or n.startswith(FAMILY[d] + "+top")) and n not in idx]
        if miss:
            stop(f"G4-3: {d}: cells of PL6's table not in the grids: {miss[:3]}")
        A_all = pres[valid] == gold[valid][:, None]
        N_all = size[valid].astype(np.int64)
        fi = [idx[n] for n in fam]
        Af, Nf = A_all[:, fi], N_all[:, fi]
        alls, nodes = Af.mean(0), Nf.mean(0)
        g44, n44 = 0.0, 0
        for k, n in enumerate(fam):                                                      # G4-4: every family cell PL6's table lists
            if n in table:
                n44 += 1
                g44 = max(g44, abs(alls[k] - table[n]["all"]), abs(nodes[k] - table[n]["nodes"]))
        if g44 > 1e-9 or n44 == 0:
            stop(f"G4-4: {d}: the family's dev ALL or nodes differ from PL6's select by {g44} over {n44} cells")
        # the ladder's nesting (the +top cuts are prefixes of one list): count the questions where a larger cut holds fewer golds or fewer nodes
        order = sorted((k for k in range(len(fam))), key=lambda k: nodes[k])
        nest_bad = int(sum(((pres[valid][:, idx[fam[b]]] < pres[valid][:, idx[fam[a]]]) | (size[valid][:, idx[fam[b]]] < size[valid][:, idx[fam[a]]])).sum()
                           for a, b in zip(order, order[1:])))
        fam_o = programs(Af, Nf, "family")
        fam_o["uniform"] = [{"name": n, "all": float(alls[k]), "nodes": float(nodes[k])} for k, n in sorted(enumerate(fam), key=lambda t: nodes[t[0]])]
        fam_o["uniform_min_nodes@0.95"] = min([float(nodes[k]) for k in range(len(fam)) if alls[k] >= 0.95] or [None]) if (alls >= 0.95).any() else None
        fam_o["hull_min_nodes@0.95"] = hull_min_nodes(alls, nodes, 0.95)
        fam_o["levels"] = len(fam)
        fam_o["smallest_level_nodes"] = float(nodes.min())
        fam_o["uniform_all@cap"] = {str(c): (float(max([alls[k] for k in range(len(fam)) if nodes[k] <= c]))
                                             if any(nodes[k] <= c for k in range(len(fam))) else None) for c in CAPS}
        fam_o["hull_all@cap"] = {str(c): hull_cap_all(alls, nodes, c) for c in CAPS}
        full_o = programs(A_all, N_all, "all cells (optimistic: a question may pick a lucky small cell)")
        out["datasets"][d] = {"family_name": FAMILY[d], "family_cells": len(fam), "g4_4_max_abs_diff": g44, "g4_4_cells_checked": n44,
                              "nesting_violations": nest_bad, "family": fam_o, "all_cells": full_o, "seconds": round(time.time() - t0, 1)}
        log(f"{d}: Q {int(valid.sum())}, family {len(fam)} cells, all {len(names)}; Q-min(.95) family {fam_o['min_nodes@0.95']}, uniform {fam_o['uniform_min_nodes@0.95']}, "
            f"all-cells {full_o['min_nodes@0.95']}")
    out["script_sha256"] = script_sha256()
    return out


# ── the self-test ────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(7)
    # L: LP, ILP against brute force
    for trial in range(6):
        J, m = 3, 8
        cov = rng.integers(0, 40, (J, m)).astype(float)
        cand = rng.integers(1, 200, (J, m)).astype(float)
        R, P = cov / cov.sum(1, keepdims=True), cand / cand.sum(1, keepdims=True)
        prot = np.zeros(m, bool)
        prot[0] = True
        for pm in (None, prot):
            rho = 0.8
            tau_i, xi, _ = solve_u(R, P, rho, pm, integer=True)
            best = None
            for bits in itertools.product((0, 1), repeat=m):
                x = np.array(bits, float)
                if pm is not None and not x[pm].all():
                    continue
                if (R @ x >= rho - 1e-12).all():
                    t = (P @ x).max()
                    best = t if best is None or t < best else best
            assert best is not None and abs(best - tau_i) < 1e-7, (trial, best, tau_i)
            tau_l, xl, dual = solve_u(R, P, rho, pm)
            assert tau_l <= tau_i + 1e-9 and (R @ xl >= rho - 1e-9).all() and (P @ xl <= tau_l + 1e-9).all()
            # a deliberately wrong objective (maximise the price) must not equal the optimum
            wrong = max((P @ np.array(b, float)).max() for b in itertools.product((0, 1), repeat=m) if (R @ np.array(b, float) >= rho).all())
            assert wrong >= tau_i - 1e-12
    # the frontier is the inverse of LP-U: the price of the frontier's reach, solved the other way, is the cap it was given (the Lagrangian sweep, G4-5)
    for trial in range(6):
        J, m = 3, 8
        cov = rng.integers(1, 40, (J, m)).astype(float)
        cand = rng.integers(1, 200, (J, m)).astype(float)
        R, P = cov / cov.sum(1, keepdims=True), cand / cand.sum(1, keepdims=True)
        for tau in (0.4, 0.6, 0.8):
            f = solve_maxmin(R, P, tau)
            assert f is not None
            r = f[0]
            if r >= 1 - 1e-9:
                continue
            t_at, _, _ = solve_u(R, P, r - 1e-9)
            t_above, _, _ = solve_u(R, P, min(1.0, r + 1e-3))
            assert t_at <= tau + 1e-7 and t_above >= tau - 1e-7, (trial, tau, r, t_at, t_above)
    # Q: the knapsack against linprog on the assignment variables, and brute force on the integers
    for trial in range(8):
        Q, K = 6, 3
        N = rng.integers(5, 60, (Q, K))
        A = rng.random((Q, K)) < 0.5
        n0, a0, nstar, delta = depth_core(A, N)
        for rho in (0.5, 0.67, 0.83):
            got, _ = q_min(n0, a0, delta, rho)
            c = N.reshape(-1).astype(float) / Q
            Aeq = np.zeros((Q, Q * K))
            for q in range(Q):
                Aeq[q, q * K:(q + 1) * K] = 1
            res = linprog(c, A_ub=-(A.reshape(1, -1).astype(float) / Q), b_ub=[-rho], A_eq=Aeq, b_eq=np.ones(Q), bounds=(0, 1), method="highs")
            if res.status == 0:
                assert got is not None and abs(got - res.fun) < 1e-7, (trial, rho, got, res.fun)
            else:
                assert got is None, (trial, rho, got)
            best = None
            for pick in itertools.product(range(K), repeat=Q):
                if np.mean([A[q, p] for q, p in enumerate(pick)]) >= rho - 1e-12:
                    v = np.mean([N[q, p] for q, p in enumerate(pick)])
                    best = v if best is None or v < best else best
            if got is not None:
                assert best is not None and got <= best + 1e-9
        for cap in (20, 30, 45):
            ga = q_cap(n0, a0, delta, cap)
            c = -(A.reshape(-1).astype(float) / Q)
            Aeq = np.zeros((Q, Q * K))
            for q in range(Q):
                Aeq[q, q * K:(q + 1) * K] = 1
            res = linprog(c, A_ub=N.reshape(1, -1).astype(float) / Q, b_ub=[cap], A_eq=Aeq, b_eq=np.ones(Q), bounds=(0, 1), method="highs")
            if res.status == 0:
                assert ga is not None and abs(ga + res.fun) < 1e-7, (trial, cap, ga, res.fun)
            else:
                assert ga is None
    # the report's hull on a stand-in: a mix of two cells beats the better single cell
    alls, nodes = np.array([0.9, 0.99]), np.array([100.0, 1000.0])
    assert abs(hull_min_nodes(alls, nodes, 0.95) - (100 + (0.05 / 0.09) * 900)) < 1e-9
    log("selftest PASS: LP/ILP equal brute force on toy links (S free and S protected), the depth knapsack equals linprog and is below the integer optimum, the hull is right")


# ── report ───────────────────────────────────────────────────────────────────


def pct(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def num(x, nd=0):
    return "-" if x is None else f"{x:,.{nd}f}"


def render_links(L):
    o = ["## Layer L: the shared link keep-rule (dev solves, eval grades the integer rule; point values)", ""]
    js = L["judged"]
    for sname, so in L["spaces"].items():
        o += [f"### Key space {sname} ({len(so['keys'])} keys)", ""]
        o += ["| program | rho | worst price tau* | largest own optimum | cost of universality | binding reach cells |", "|---|---|---|---|---|---|"]
        for pname in ("A1", "S_free"):
            for rho in RHOS:
                r = so["prot"][pname]["rho"][str(rho)]
                bind = ", ".join(j for j, b in zip(js, r["binding_reach"]) if b) or "none"
                o.append(f"| {'S protected (A.1, the call)' if pname == 'A1' else 'S free'} | {rho:.2f} | {pct(r['tau'])} | {pct(max(r['own_tau']))} | {pct(r['cost_of_universality'])} | {bind} |")
        o += ["", f"The price of one point of reach near the call (tau*(0.96) - tau*(0.94)) / 2, in points of price: "
              f"S protected {pct((so['prot']['A1']['rho']['0.96']['tau'] - so['prot']['A1']['rho']['0.94']['tau']) / 2)}; "
              f"S free {pct((so['prot']['S_free']['rho']['0.96']['tau'] - so['prot']['S_free']['rho']['0.94']['tau']) / 2)}.", ""]
        o += ["**Each judged cell's own optimum at rho 0.95 (dev price; the least a rule made for that cell alone could pay), and the S floor (the price of S alone):**", "",
              "| judged cell | own optimum, S free | own optimum, S protected | S floor (dev) | S floor (eval) | S-alone reach (dev) |", "|---|---|---|---|---|---|"]
        for j, jn in enumerate(js):
            o.append(f"| {jn} | {pct(so['prot']['S_free']['rho']['0.95']['own_tau'][j])} | {pct(so['prot']['A1']['rho']['0.95']['own_tau'][j])} | "
                     f"{pct(so['s_floor_price_dev'][j])} | {pct(so['s_floor_price_eval'][j])} | {pct(so['s_floor_reach_dev'][j])} |")
        o += ["", "**The cost in reach of a price (max over x of the worst judged cell's dev reach, at price <= tau for every cell):**", "",
              "| price cap tau | S protected | S free |", "|---|---|---|"]
        for tau in TAUS:
            a = so["prot"]["A1"]["frontier"][str(tau)]
            b = so["prot"]["S_free"]["frontier"][str(tau)]
            o.append(f"| {tau:.2f} | {'infeasible' if a is None else pct(a['max_min_reach'])} | {'infeasible' if b is None else pct(b['max_min_reach'])} |")
        for pname, title in (("A1", "S protected (the call)"), ("S_free", "S free (beside)")):
            c = so["prot"][pname]["call"]
            dropped = [k for k in so["keys"] if k not in set(c["kept_keys"])]
            o += ["", f"**The integer rule, {title}, solved on dev at rho 0.95: {c['verdict']}** (dev worst price {pct(c['tau_dev'])}; kept {len(c['kept_keys'])} of {len(so['keys'])} keys; "
                  f"dropped: {', '.join(dropped) if dropped else 'none'}).", "",
                  "| judged cell | dev reach / price | eval reach / price | SCREEN_IN |", "|---|---|---|---|"]
            for j, jn in enumerate(js):
                o.append(f"| {jn} | {pct(c['dev_reach'][j])} / {pct(c['dev_price'][j])} | {pct(c['eval_reach'][j])} / {pct(c['eval_price'][j])} | {'IN' if c['screen_in'][j] else 'OUT'} |")
        o.append("")
    return o


def render_depth(Dp):
    o = ["## Layer Q: the depth allocation (dev; an oracle bound, never a method)", "",
         "Mean nodes per question at ALL >= 0.95 on the dev carve, by rule. **Uniform** = one level for every question (the family's own table); **hull** = a randomised uniform rule; "
         "**oracle depth** = each question gets the cheapest level that holds all its golds (the LP optimum; it reads the golds).", "",
         "| dataset | family cells | uniform (least nodes at 0.95) | hull | oracle depth (family) | oracle ALL at 2,500 | uniform ALL at 2,500 | family ceiling on ALL | verdict at 2,500 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for d in DS7:
        r = Dp["datasets"].get(d)
        if not r:
            continue
        f = r["family"]
        mn = f["min_nodes@0.95"]
        verdict = "NOT_READ" if mn is None else ("FEASIBLE_AT_2500" if mn <= 2500 else "INFEASIBLE_AT_2500")
        o.append(f"| {d} | {r['family_cells']} | {num(f['uniform_min_nodes@0.95'])} | {num(f['hull_min_nodes@0.95'])} | {num(mn)} | {pct(f['all@cap2500'])} | "
                 f"{pct(f['uniform_all@cap']['2500'])} | {pct(f['ceiling_all'])} | {verdict} |")
    o += ["", "**The menu each program had (the family's levels, its smallest level's mean nodes, and whether the nesting held):**", "",
          "| dataset | levels | smallest level (mean nodes) | questions where a larger cut held less | G4-4 cells checked |", "|---|---|---|---|---|"]
    for d in DS7:
        r = Dp["datasets"].get(d)
        if r:
            o.append(f"| {d} | {r['family_cells']} | {num(r['family']['smallest_level_nodes'])} | {r['nesting_violations']} | {r['g4_4_cells_checked']} |")
    o += ["", "**ALL (percent) at a mean-node cap, oracle depth in the family (uniform in brackets):**", "", "| dataset | " + " | ".join(f"{c:,}" for c in CAPS) + " |", "|---|" + "---|" * len(CAPS)]
    for d in DS7:
        r = Dp["datasets"].get(d)
        if r:
            f = r["family"]
            o.append(f"| {d} | " + " | ".join(f"{pct(f['all@cap' + str(c)])} ({pct(f['uniform_all@cap'][str(c)])})" for c in CAPS) + " |")
    o += ["", "**Share of questions whose cheapest covering pool in the family is at most N nodes (percent):**", "", "| dataset | " + " | ".join(f"{c:,}" for c in CAPS) + " | covered by no level |", "|---|" + "---|" * (len(CAPS) + 1)]
    for d in DS7:
        r = Dp["datasets"].get(d)
        if r:
            f = r["family"]
            o.append(f"| {d} | " + " | ".join(pct(f["share_cover_within"][str(c)]) for c in CAPS) + f" | {pct(f['unreachable_share'])} |")
    o += ["", "**The price of ALL, and the optimistic bound beside.** Nodes per point of ALL at 0.95 (oracle depth, family) and the optimistic all-cell bound (a question may pick a lucky small cell of ~6,000):", "",
          "| dataset | nodes per point of ALL | oracle depth at 0.98 (family) | all-cell oracle at 0.95 (optimistic) | all-cell ALL at 2,500 (optimistic) |", "|---|---|---|---|---|"]
    for d in DS7:
        r = Dp["datasets"].get(d)
        if r:
            f, a = r["family"], r["all_cells"]
            o.append(f"| {d} | {num(f['nodes_per_point_at_0.95'])} | {num(f['min_nodes@0.98'])} | {num(a['min_nodes@0.95'])} | {pct(a['all@cap2500'])} |")
    o.append("")
    return o


def cmd_report():
    parts = ["# UL4 report (docs/UL4_UNIVERSAL_LP.md)", "", ""]
    for nm, rend in (("links", render_links), ("depth", render_depth)):
        p = OUT / f"{nm}.json"
        if p.is_file():
            parts += rend(json.loads(p.read_text(encoding="utf-8")))
        else:
            parts += [f"({nm}.json is not here)", ""]
    (OUT / "report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    log(f"report written: {(OUT / 'report.md').relative_to(ROOT)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "lp-links", "lp-depth", "report"))
    ap.add_argument("--ds", default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "lp-links":
        out = lp_links()
        (OUT / "links.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    elif a.cmd == "lp-depth":
        ds = DS7 if a.ds is None else tuple(a.ds.split(","))
        out = lp_depth(ds)
        p = OUT / "depth.json"
        if a.ds is not None and p.is_file():
            prev = json.loads(p.read_text(encoding="utf-8"))
            prev["datasets"].update(out["datasets"])
            out = {**prev, "script_sha256": out["script_sha256"]}
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    else:
        cmd_report()


if __name__ == "__main__":
    main()
