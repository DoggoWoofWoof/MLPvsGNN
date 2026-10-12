"""P1b: one restart for all six datasets, its source weights set per question and fit by EM
(docs/P1B_RESTART_MIXTURE.md). A diagnostic of pools: no model trains or reads, and nothing is adopted.

    python scripts/p1b_restart_mixture.py gather DATASET [--threads 5] [--host]   a training dataset's fit carve
    python scripts/p1b_restart_mixture.py fit                                     EM on the five gathers
    python scripts/p1b_restart_mixture.py apply DATASET [--threads 5] [--host]    s1sel (training datasets) and s1eval
    python scripts/p1b_restart_mixture.py verdict
    python scripts/p1b_restart_mixture.py --selftest

Six restart sources per question, each a distribution over nodes, on the dataset's regime union graph with step 4e's
kernel (its alpha and eps) and nothing excluded:
    S0  today's seeds, uniform                      S  the walk from S0  (P1a's A0)
    L0  our exact-match linked nodes, uniform       L  the walk from L0  (P1a's LINKED)
    R0  the RRF-weighted dense and SPLADE lists     R  the walk from R0
A walk's masses are its top NMAX nodes', scaled to sum to 1. squad has no graph: there S, L and R are S0, L0 and R0.
The mixture is m_q = sum_k w_k(q) m_k with w(q) = softmax(W x(q)), x(q) fifteen label-free features of the question's
own lists, links and walks (FEATURES). W is fit by EM on the training datasets' fit carves, every dataset weighted
equally and webqsp never; the likelihood is that of the golds under the mixture. The pool at N is the mixture's top N
(ties by node position).
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import p1a_pool_scoreboard as P  # noqa: E402

E, F = P.E, P.F
S4 = E.S4
_wpush = E._wpush
_top = S4._top

OUT = ROOT / "outputs" / "p1b"
GATE = OUT / "gate.json"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
NS, NMAX, TIE, KB, SIX, M_KBQA = P.NS, P.NMAX, P.TIE, P.KB, P.SIX, P.M_KBQA
TRAIN5 = E.TRAIN5
SOURCES = ("S0", "L0", "R0", "S", "L", "R")
FEATURES = ("log_seeds", "log_linked", "no_link", "dense_top", "dense_gap", "splade_top", "splade_gap",
            "linked_in_lists", "seeds_linked", "S_top", "L_top", "R_top", "S_top10", "L_top10", "R_top10")
LAMBDA = 1e-3          # the gate's L2 penalty, on an objective that averages to one gold
EM_MAX = 200
EM_TOL = 1e-9          # stop when the penalized log-likelihood gains less than this, relatively
FLOOR = 1e-9           # the log floor of a walk's top mass
log, utc, write_json, sha_file = P.log, P.utc, P.write_json, P.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fit_carve(name):
    return "s1fit" if name == "musique" else "fit"


# ── the walk with its masses ─────────────────────────────────────────────────


@njit(parallel=True)
def wppr_mass_batch(seeds_ptr, seeds_flat, w_flat, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, bmax, n_chunks,
                    out_nodes, out_mass, out_count):
    """step 4e's wppr_batch with nothing excluded, writing each question's top masses beside its order."""
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
            excl = np.empty(0, dtype=np.int64)
            for q in range(lo, hi):
                seeds = seeds_flat[seeds_ptr[q]:seeds_ptr[q + 1]]
                wts = w_flat[seeds_ptr[q]:seeds_ptr[q + 1]]
                nt, _work = _wpush(seeds, wts, excl, ip0, c0, ip1, c1, ip2, c2, deg, alpha, eps, p, r, flag, touched,
                                   queue)
                _avail, cnt = _top(p, flag, touched, nt, bmax, out_nodes[q])
                out_count[q] = cnt
                for k in range(cnt):
                    out_mass[q, k] = p[out_nodes[q, k]]
                for k in range(nt):
                    v = touched[k]
                    p[v] = 0.0
                    r[v] = 0.0
                    flag[v] = 0


def walk_mass(union, restarts, threads):
    """[(nodes by mass descending then position, masses scaled to sum 1)] per restart; an empty restart walks to ()."""
    nq = len(restarts)
    s_ptr, s_flat = S4.flat([r[0] for r in restarts])
    w_flat = np.concatenate([r[1] for r in restarts]).astype(np.float64) if nq else np.empty(0)
    out = np.full((nq, NMAX), -1, dtype=np.int64)
    mass = np.zeros((nq, NMAX), dtype=np.float64)
    cnt = np.zeros(nq, dtype=np.int64)
    wppr_mass_batch(s_ptr, s_flat, w_flat, *union.args, float(E.ALPHA), float(S4.EPS), int(NMAX),
                    max(1, min(nq, 4 * threads)), out, mass, cnt)
    res = []
    for q in range(nq):
        c = int(cnt[q])
        m = mass[q, :c].copy()
        tot = m.sum()
        res.append((out[q, :c].copy(), m / tot if tot > 0 else m))
    return res


# ── sources, features and the mixture ────────────────────────────────────────


EMPTY = (np.empty(0, np.int64), np.empty(0, np.float64))


def uniform(nodes):
    u = np.unique(np.asarray(nodes, np.int64))
    return (u, np.full(u.size, 1.0 / u.size)) if u.size else EMPTY


def by_mass(nodes, mass):
    """Nodes by mass descending then position ascending, as the walk orders them."""
    o = np.lexsort((nodes, -mass))
    return nodes[o], mass[o]


def zero_step(prep, j, linked_j, constant):
    rn, rw = E.rrf_weights(prep.dense_ids[j], prep.splade_ids[j], constant)
    return uniform(prep.seeds[j]), uniform(linked_j), by_mass(rn, rw / rw.sum())


def sources_batch(name, prep, js, linked, union, constant, threads):
    """Per question of js: the six sources (S0, L0, R0, S, L, R), each (nodes, masses) by mass."""
    zs = [zero_step(prep, j, linked[j], constant) for j in js]
    if name == "squad":
        return [z + z for z in zs]
    walked = [walk_mass(union, [z[k] for z in zs], threads) for k in range(3)]
    return [zs[i] + tuple(walked[k][i] for k in range(3)) for i in range(len(zs))]


def features(prep, j, src):
    """FEATURES, raw. Lists, links and walks only: never a gold."""
    s0, l0, r0, s, lw, r = src
    d = np.sort(np.asarray(prep.dense_scores[j], np.float64))[::-1]
    sp = np.sort(np.asarray(prep.splade_scores[j], np.float64))[::-1]
    heads = np.union1d(np.asarray(prep.dense_ids[j][:100], np.int64), np.asarray(prep.splade_ids[j][:100], np.int64))
    nl, ns = l0[0].size, s0[0].size
    x = [np.log1p(ns), np.log1p(nl), float(nl == 0), d[0], d[0] - d[9], np.log1p(max(sp[0], 0.0)),
         np.log1p(max(sp[0] - sp[9], 0.0)), float(np.isin(l0[0], heads).sum()) / max(1, nl),
         float(np.isin(s0[0], l0[0]).sum()) / max(1, ns)]
    x += [float(np.log(max(w[1][0], FLOOR))) if w[1].size else float(np.log(FLOOR)) for w in (s, lw, r)]
    x += [float(w[1][:10].sum()) for w in (s, lw, r)]
    return np.asarray(x, np.float64)


def gold_masses(src, golds):
    """(|golds|, 6): each gold's mass under each source, 0 where the source does not hold it."""
    out = np.zeros((golds.size, len(SOURCES)))
    for k, (nodes, mass) in enumerate(src):
        pos = P.positions(nodes, golds)
        out[:, k] = np.where(pos >= 0, mass[np.maximum(pos, 0)] if mass.size else 0.0, 0.0)
    return out


def softmax(a):
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)


def gate_weights(gate, X):
    W = np.asarray(gate["W"], np.float64)
    Z = (X - np.asarray(gate["mu"])) / np.asarray(gate["sd"])
    return softmax(np.hstack([Z, np.ones((Z.shape[0], 1))]) @ W.T)


def mix_order(src, w, nmax=NMAX):
    nodes = np.concatenate([s[0] for s in src])
    if nodes.size == 0:
        return np.empty(0, np.int64)
    mass = np.concatenate([w[k] * s[1] for k, s in enumerate(src)])
    u, inv = np.unique(nodes, return_inverse=True)
    m = np.bincount(inv, weights=mass, minlength=u.size)
    keep = m > 0
    u, m = u[keep], m[keep]
    return u[np.lexsort((u, -m))][:nmax]


# ── gather (fit carves) ──────────────────────────────────────────────────────


def gather_path(name):
    return OUT / f"gather_{name}.npz"


def gather_stage(name, threads):
    if name not in TRAIN5:
        raise SystemExit(f"{name} never fits")
    op = S4.Opened()
    carve = fit_carve(name)

    def sink(name, carve_, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        nq = len(U)
        X, GM, gptr, reached = [], [], [0], np.zeros(len(SOURCES))
        golds_n = 0
        for lo in range(0, nq, S4.BATCH):
            js = range(lo, min(nq, lo + S4.BATCH))
            srcs = sources_batch(name, prep, js, linked, union, constant, threads)
            for j, src in zip(js, srcs):
                g = np.unique(np.asarray(pop.golds[j], np.int64))
                X.append(features(prep, j, src))
                gm = gold_masses(src, g)
                GM.append(gm)
                gptr.append(gptr[-1] + g.size)
                reached += (gm > 0).sum(axis=0)
                golds_n += g.size
        X = np.vstack(X)
        GM = np.vstack(GM)
        tmp = OUT / f"gather_{name}.tmp.npz"
        np.savez(tmp, X=X, GM=GM, gptr=np.asarray(gptr, np.int64), ids=np.asarray(list(pop.ids)))
        os.replace(tmp, gather_path(name))
        rec = {"declared_in": "docs/P1B_RESTART_MIXTURE.md", "dataset": name, "carve": carve_, "questions": nq,
               "carve_sha256": pop.digest, "golds": int(golds_n), "features": list(FEATURES), "sources": list(SOURCES),
               "golds_reached_by_source": {s: round(float(c) / max(1, golds_n), 4) for s, c in zip(SOURCES, reached)},
               "golds_reached_by_any": round(float((GM.sum(axis=1) > 0).mean()), 4),
               "npz_sha256": sha_file(gather_path(name)), "script_sha256": script_sha256(),
               "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "seconds": round(time.time() - t, 1), "utc": utc()}
        write_json(OUT / f"gather_{name}.json", rec)
        log(f"{name}/{carve_} ({nq} q, {rec['seconds']:.0f}s): golds reached " + ", ".join(
            f"{s} {v}" for s, v in rec["golds_reached_by_source"].items()) + f"; any {rec['golds_reached_by_any']}")

    OUT.mkdir(parents=True, exist_ok=True)
    E.run_carves(op, name, {carve}, ("A3",), threads, sink)
    return 0


# ── the EM fit ───────────────────────────────────────────────────────────────


def em(X, GM, gptr, qw, lam=LAMBDA, iters=EM_MAX, tol=EM_TOL):
    """EM for the gate. X: (Q, F) standardized; GM: (G, K) golds' masses; gptr: (Q + 1) into GM; qw: (Q) question
    weights. Each gold weighs qw[q] / |G_q|; golds no source reaches carry no likelihood. Returns W (K, F + 1) and
    the trace of the penalized log-likelihood."""
    from scipy.optimize import minimize
    Q, K = X.shape[0], GM.shape[1]
    Xb = np.hstack([X, np.ones((Q, 1))])
    ng = np.diff(gptr)
    qof = np.repeat(np.arange(Q), ng)
    gw = (qw / np.maximum(ng, 1))[qof]
    keep = GM.sum(axis=1) > 0
    qk, gk, Mk = qof[keep], gw[keep], GM[keep]
    W = np.zeros((K, Xb.shape[1]))

    def penalized_ll(W):
        w = softmax(Xb @ W.T)
        return float((gk * np.log((w[qk] * Mk).sum(axis=1))).sum() - lam * (W ** 2).sum())

    trace = [penalized_ll(W)]
    for _ in range(iters):
        w = softmax(Xb @ W.T)
        num = w[qk] * Mk
        R = num / num.sum(axis=1, keepdims=True)
        T = np.zeros((Q, K))
        np.add.at(T, qk, gk[:, None] * R)
        c = T.sum(axis=1)

        def neg(wflat):
            Wm = wflat.reshape(K, -1)
            A = Xb @ Wm.T
            A = A - A.max(axis=1, keepdims=True)
            lse = np.log(np.exp(A).sum(axis=1))
            obj = (T * (A - lse[:, None])).sum() - lam * (Wm ** 2).sum()
            sm = np.exp(A - lse[:, None])
            grad = (T - c[:, None] * sm).T @ Xb - 2 * lam * Wm
            return -obj, -grad.ravel()

        res = minimize(neg, W.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-10})
        W = res.x.reshape(K, -1)
        trace.append(penalized_ll(W))
        if trace[-1] - trace[-2] < tol * abs(trace[-2]):
            break
    return W, trace


def fit_stage():
    parts = {}
    for name in TRAIN5:
        rec = json.loads((OUT / f"gather_{name}.json").read_text(encoding="utf-8"))
        if sha_file(gather_path(name)) != rec["npz_sha256"]:
            raise SystemExit(f"{gather_path(name)}: not its record's file")
        with np.load(gather_path(name)) as z:
            parts[name] = ({k: z[k] for k in ("X", "GM", "gptr")}, rec)
    shas = {(r["script_sha256"], r["freeze_RECORD_SHA256"]) for _d, r in parts.values()}
    if len(shas) != 1:
        raise SystemExit(f"the gathers ran on {len(shas)} scripts or freezes")
    X = np.vstack([d["X"] for d, _r in parts.values()])
    GM = np.vstack([d["GM"] for d, _r in parts.values()])
    gptr = [0]
    for d, _r in parts.values():
        gptr += (gptr[-1] + d["gptr"][1:]).tolist()
    gptr = np.asarray(gptr, np.int64)
    qw = np.concatenate([np.full(d["X"].shape[0], 1.0 / (len(parts) * d["X"].shape[0])) for d, _r in parts.values()])
    mu = (qw[:, None] * X).sum(axis=0) / qw.sum()
    sd = np.sqrt((qw[:, None] * (X - mu) ** 2).sum(axis=0) / qw.sum())
    sd = np.where(sd > 1e-6, sd, 1.0)
    t = time.time()
    W, trace = em((X - mu) / sd, GM, gptr, qw)
    gate = {"declared_in": "docs/P1B_RESTART_MIXTURE.md", "sources": list(SOURCES), "features": list(FEATURES),
            "W": W.round(10).tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "lambda": LAMBDA,
            "em_iterations": len(trace) - 1, "trace": [round(v, 8) for v in trace],
            "gathers": {n: r["npz_sha256"] for n, (_d, r) in parts.items()},
            "gather_script_sha256": next(iter(shas))[0], "freeze_RECORD_SHA256": next(iter(shas))[1],
            "script_sha256": script_sha256(), "seconds": round(time.time() - t, 1), "utc": utc()}
    mean_w = {}
    for name, (d, _r) in parts.items():
        w = gate_weights(gate, d["X"])
        mean_w[name] = {s: round(float(v), 4) for s, v in zip(SOURCES, w.mean(axis=0))}
    gate["fit_mean_weights"] = mean_w
    write_json(GATE, gate)
    log(f"gate: {gate['em_iterations']} EM iterations, penalized log-likelihood {trace[0]:.4f} -> {trace[-1]:.4f}")
    for name, mw in mean_w.items():
        log(f"  {name}: " + ", ".join(f"{s} {v}" for s, v in mw.items()))
    return 0


# ── apply (s1sel, s1eval) ────────────────────────────────────────────────────


def part_path(name):
    return OUT / f"mx_{name}.json"


DIAG = ("MX", "S", "L", "A1MIX") + SOURCES


def apply_stage(name, threads):
    op = S4.Opened()
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    rec = {"declared_in": "docs/P1B_RESTART_MIXTURE.md", "dataset": name, "ns": list(NS), "alpha": E.ALPHA,
           "eps": S4.EPS, "gate_sha256": sha_file(GATE), "script_sha256": script_sha256(),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "carves": {}}
    if gate["script_sha256"] != script_sha256():
        raise SystemExit("the gate was fit by another version of this script")
    if gate["freeze_RECORD_SHA256"] != op.freeze["RECORD_SHA256"]:
        raise SystemExit("the gate was fit on another freeze")

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        nq = len(U)
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
        p4e = P.p4e_pools(name, carve, pop)
        pos = {o: [] for o in DIAG}
        lens = {o: [] for o in DIAG}
        wsum = np.zeros(len(SOURCES))
        for lo in range(0, nq, S4.BATCH):
            js = range(lo, min(nq, lo + S4.BATCH))
            srcs = sources_batch(name, prep, js, linked, union, constant, threads)
            X = np.vstack([features(prep, j, s) for j, s in zip(js, srcs)])
            Wq = gate_weights(gate, X)
            wsum += Wq.sum(axis=0)
            half = np.zeros(len(SOURCES))
            half[SOURCES.index("S")] = half[SOURCES.index("R")] = 0.5
            for i, (j, src) in enumerate(zip(js, srcs)):
                orders = {"MX": mix_order(src, Wq[i]), "S": src[3][0], "L": src[4][0], "A1MIX": mix_order(src, half)}
                for k, s in enumerate(SOURCES):
                    orders[s] = src[k][0][:NMAX]
                for o in DIAG:
                    pos[o].append(P.positions(orders[o], golds[j]))
                    lens[o].append(int(orders[o].size))
        sizes = {"I_q": np.asarray([np.asarray(inc[j]).size for j in range(nq)]),
                 "P4e": np.asarray([p4e[j].size for j in range(nq)])}
        ent = {"questions": nq, "carve_sha256": pop.digest,
               "mean_weights": {s: round(float(v) / max(1, nq), 4) for s, v in zip(SOURCES, wsum)},
               "orders": {o: {"curve": P.curve(pos[o], lens[o]), "len_mean": round(float(np.mean(lens[o])), 1),
                              "at_I_q": P.at_sizes(pos[o], sizes["I_q"]), "at_P4e": P.at_sizes(pos[o], sizes["P4e"])}
                          for o in DIAG},
               "sets": {"I_q": P.set_cov([np.asarray(inc[j], np.int64) for j in range(nq)], golds),
                        "P4e": P.set_cov(p4e, golds)},
               "seconds": round(time.time() - t, 1)}
        rec["carves"][f"{name}__{carve}"] = ent
        rec["utc"] = utc()
        write_json(part_path(name), rec)
        log(f"{name}/{carve} ({nq} q, {ent['seconds']:.0f}s): weights " + ", ".join(
            f"{s} {v}" for s, v in ent["mean_weights"].items()) + " | " + " | ".join(
            f"{o} " + " ".join(f"{n}:{ent['orders'][o]['curve'][str(n)]['ALL']}" for n in (100, 1000, 7000))
            for o in ("MX", "A1MIX", "L")))

    OUT.mkdir(parents=True, exist_ok=True)
    wanted = {"s1eval"} | ({"s1sel"} if name in TRAIN5 else set())
    E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


# ── the verdict (section 4 of the declared file) ─────────────────────────────


def consistency(mx, p1a, name):
    """P1b's S and L walks against P1a's A0 and LINKED: equal at every N on both measures (squad: L only, since P1a's
    walk orders there are its RRF order)."""
    bad = []
    pairs = (("L", "LINKED"),) if name == "squad" else (("S", "A0"), ("L", "LINKED"))
    for ours, theirs in pairs:
        for n in NS:
            for m in ("ALL", "recall"):
                a = mx["orders"][ours]["curve"][str(n)][m]
                b = p1a["orders"][theirs]["curve"][str(n)][m]
                if a != b:
                    bad.append({"order": ours, "p1a": theirs, "N": n, "measure": m, "p1b": a, "p1a_value": b})
    return bad


def call(mx, p1a, name):
    below, above = [], []
    for n in NS:
        for m in ("ALL", "recall"):
            ours = mx["orders"]["MX"]["curve"][str(n)][m]
            best_o = max(P.SOTA, key=lambda o: p1a["orders"][o]["curve"][str(n)][m])
            best = p1a["orders"][best_o]["curve"][str(n)][m]
            row = {"N": n, "measure": m, "MX": ours, "sota": best, "order": best_o, "gap": round(ours - best, 4),
                   "A3": p1a["orders"]["A3"]["curve"][str(n)][m]}
            if ours < best - TIE:
                below.append(row)
            elif ours > best + TIE:
                above.append(row)
    out = {"call": "BELOW" if below else "AT_OR_ABOVE", "below": below, "above": above,
           "largest_gap": min((b["gap"] for b in below), default=0.0),
           "largest_lead": max((a["gap"] for a in above), default=0.0)}
    for size in ("at_I_q", "at_P4e"):
        best_o = max(P.SOTA, key=lambda o: p1a["orders"][o][size]["ALL"])
        out[size] = {"MX": mx["orders"]["MX"][size], "A3": p1a["orders"]["A3"][size], "best_sota": best_o,
                     "sota": p1a["orders"][best_o][size], "set": mx["sets"]["I_q" if size == "at_I_q" else "P4e"]}
    if name in KB:
        m = M_KBQA[name]
        ora = p1a["orders"]["ORACLE"]["curve"][str(m)]
        reach = next((n for n in NS if mx["orders"]["MX"]["curve"][str(n)]["ALL"] >= ora["ALL"]), None)
        out["oracle"] = {"m": m, "ORACLE_at_m": ora, "MX_at_m": mx["orders"]["MX"]["curve"][str(m)],
                         "MX_reaches_oracle_ALL_at_N": reach}
    return out


def verdict_stage():
    if not P.VERDICT.exists():
        raise SystemExit("P1a's verdict is not filed")
    parts, p1a = {}, {}
    for name in SIX:
        for d, path in ((parts, part_path(name)), (p1a, P.part_path(name))):
            if not path.exists():
                raise SystemExit(f"{path} is missing")
            d[name] = json.loads(path.read_text(encoding="utf-8"))
    if len({(r["script_sha256"], r["gate_sha256"], r["freeze_RECORD_SHA256"]) for r in parts.values()}) != 1:
        raise SystemExit("the parts ran on different scripts, gates or freezes")
    calls, checks = {}, {}
    for name in SIX:
        for c, ent in parts[name]["carves"].items():
            if c not in p1a[name]["carves"]:
                raise SystemExit(f"P1a has no {c}")
            if ent["questions"] != p1a[name]["carves"][c]["questions"]:
                raise SystemExit(f"{c}: {ent['questions']} questions here, {p1a[name]['carves'][c]['questions']} in P1a")
            bad = consistency(ent, p1a[name]["carves"][c], name)
            if bad:
                raise SystemExit(f"{c}: P1b's walks differ from P1a's at {len(bad)} points, first {bad[0]}")
            checks[c] = "S and L equal P1a's A0 and LINKED at every point" if name != "squad" else \
                "L equals P1a's LINKED at every point"
            calls[c] = call(ent, p1a[name]["carves"][c], name)
    head = {n: calls[f"{n}__s1eval"]["call"] for n in SIX}
    rec = {"declared_in": "docs/P1B_RESTART_MIXTURE.md", "calls": calls, "checks": checks, "s1eval": head,
           "all_six_at_or_above": all(v == "AT_OR_ABOVE" for v in head.values()),
           "p1a_parts": {n: sha_file(P.part_path(n)) for n in SIX}, "gate_sha256": sha_file(GATE), "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(parts, p1a, calls), encoding="utf-8", newline="\n")
    log("calls on s1eval: " + ", ".join(f"{n} {v}" for n, v in head.items()))
    return 0


def report_md(parts, p1a, calls):
    L = ["# P1b: the restart mixture against the SOTA pools on the full corpora (s1eval)", "",
         "ALL (every gold in the pool) at N rows; recall in brackets. MX and its sources from outputs/p1b/mx_<dataset>.json;"
         " A3 and the SOTA orders from P1a's records.", ""]
    show = (25, 100, 500, 1000, 2000, 7000)
    for name in SIX:
        c = f"{name}__s1eval"
        ent, pa, cl = parts[name]["carves"][c], p1a[name]["carves"][c], calls[c]
        L += [f"## {name}: {cl['call']}", "", "| order | " + " | ".join(f"N = {n:,}" for n in show) + " |",
              "| --- |" + " ---: |" * len(show)]
        rows = [("MX", ent["orders"]["MX"]["curve"])] + [(o, pa["orders"][o]["curve"]) for o in ("A3",) + P.SOTA]
        if name in KB:
            rows.append(("ORACLE", pa["orders"]["ORACLE"]["curve"]))
        for o, cv in rows:
            L.append(f"| {o} | " + " | ".join(f"{cv[str(n)]['ALL']:.3f} ({cv[str(n)]['recall']:.3f})" for n in show)
                     + " |")
        L.append("")
        L.append("- Mean source weights: " + ", ".join(f"{s} {v}" for s, v in ent["mean_weights"].items()) + ".")
        for size, label in (("at_I_q", "today's pools (I_q)"), ("at_P4e", "step 4e's pools")):
            s = cl[size]
            L.append(f"- At each question's size of {label} ({s['set']['size']:,.0f} rows mean): the set "
                     f"{s['set']['ALL']:.3f}, MX {s['MX']['ALL']:.3f}, A3 {s['A3']['ALL']:.3f}, best SOTA order "
                     f"{s['best_sota']} {s['sota']['ALL']:.3f}.")
        if cl["below"]:
            L.append(f"- Below at {len(cl['below'])} points; the largest gap {cl['largest_gap']:+.4f}.")
        if cl["above"]:
            L.append(f"- Above at {len(cl['above'])} points; the largest lead {cl['largest_lead']:+.4f}.")
        if name in KB:
            o = cl["oracle"]
            L.append(f"- Assigned topic entities (diagnostic), m = {o['m']:,}: ORACLE {o['ORACLE_at_m']['ALL']:.3f}, "
                     f"MX {o['MX_at_m']['ALL']:.3f}; MX reaches ORACLE's ALL at "
                     f"N = {o['MX_reaches_oracle_ALL_at_N'] or 'beyond 7,000'}.")
        L.append("")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


class _Store:
    def __init__(self, n, edges):
        self.n_nodes = n
        src = np.asarray([a for a, _b in edges] + [b for _a, b in edges], np.int64)
        dst = np.asarray([b for _a, b in edges] + [a for a, _b in edges], np.int64)
        o = np.lexsort((dst, src))
        src, dst = src[o], dst[o]
        self.indptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(src, minlength=n), out=self.indptr[1:])
        self.col = dst.astype(np.int32)


def selftest():
    # the walk with masses orders as step 4e's walk does
    n = 12
    stores = {S4.FAMILIES[0]: _Store(n, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7)]),
              S4.FAMILIES[1]: _Store(n, [(1, 8), (8, 9), (4, 10)]),
              S4.FAMILIES[2]: _Store(n, [(10, 11), (2, 9)])}
    union = S4.Union(stores)
    rs = [(np.asarray([0, 3]), np.asarray([0.75, 0.25])), (np.asarray([9]), np.asarray([1.0])), EMPTY]
    got = walk_mass(union, rs, 2)
    ref = E.walk(union, [rs[0], rs[1]], [np.empty(0, np.int64)] * 2, NMAX, 2)
    assert got[0][0].tolist() == ref[0].tolist() and got[1][0].tolist() == ref[1].tolist()
    assert abs(got[0][1].sum() - 1) < 1e-12 and np.all(np.diff(got[0][1]) <= 0) and got[2][0].size == 0
    # the mixture: a one-hot weight reproduces that source's order
    src = (uniform([4, 2]), EMPTY, by_mass(np.asarray([7, 2, 5]), np.asarray([0.2, 0.5, 0.3])),
           got[0], got[1], EMPTY)
    for k in (0, 2, 3, 4):
        w = np.zeros(6)
        w[k] = 1.0
        assert mix_order(src, w).tolist() == src[k][0].tolist(), k
    w = np.full(6, 1 / 6)
    mo = mix_order(src, w)
    assert np.unique(mo).size == mo.size
    gm = gold_masses(src, np.asarray([2, 9, 99]))
    assert gm.shape == (3, 6) and gm[0, 0] == 0.5 and gm[0, 2] == 0.5 and gm[2].sum() == 0
    # EM finds the source that holds the golds, question by question
    rng = np.random.default_rng(0)
    Q = 400
    X = rng.normal(size=(Q, 2))
    good = (X[:, 0] > 0).astype(int)          # questions with x0 > 0 have their gold under source 1, else source 0
    GM = np.full((Q, 6), 1e-4)
    GM[np.arange(Q), good] = 0.5
    W, trace = em(X, GM, np.arange(Q + 1), np.full(Q, 1.0 / Q))
    assert all(b >= a - 1e-9 for a, b in zip(trace, trace[1:])), trace[:5]
    w = softmax(np.hstack([X, np.ones((Q, 1))]) @ W.T)
    assert (w.argmax(axis=1) == good).mean() > 0.95
    print("selftest: ok (walk masses order as step 4e's walk, one-hot mixtures, gold masses, EM monotone and "
          "picks the right source)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("gather", "fit", "apply", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage in ("gather", "apply"):
        if a.dataset not in SIX:
            raise SystemExit(f"dataset must be one of {SIX}")
        if a.host:
            import importlib.util
            spec = importlib.util.spec_from_file_location("s4f_run_one", ROOT / "outputs" / "step4f" / "run_one.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.on_the_mirror(F)
        return gather_stage(a.dataset, a.threads) if a.stage == "gather" else apply_stage(a.dataset, a.threads)
    if a.stage == "fit":
        return fit_stage()
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
