"""P2b: one link rule for all six datasets, the families' walks weighted per question (docs/P2B_FAMILY_GATE.md).
A diagnostic of pools: no model trains or reads, and nothing is adopted.

    python scripts/p2b_family_gate.py gather DATASET [--threads 5] [--host] [--limit Q]   a training dataset's fit carve
    python scripts/p2b_family_gate.py fit                                                 EM on the five gathers
    python scripts/p2b_family_gate.py apply DATASET [--threads 5] [--host] [--limit Q]    s1sel (training) and s1eval
    python scripts/p2b_family_gate.py verdict
    python scripts/p2b_family_gate.py --selftest

Nine sources per question, on every dataset: P1b's zero-step S0, L0 and R0, and each of them walked with step 4e's
kernel (its alpha and eps, nothing excluded) on the structural family alone (S_st, L_st, R_st) and on all three
families (S_all, L_all, R_all), over M3B's stores. MXU never reads the dataset's regime: squad's families are walked as
every other dataset's are.
MXU is P1b's mixture over the nine: m_q = sum_k w_k(q) m_k, w(q) = softmax(W x(q)), x(q) the twenty-four label-free
FEATURES, W fit by P1b's EM on the five training datasets' fit carves (every dataset weighted equally, webqsp never).
The pool at N is the mixture's top N, ties by node position.
MX, P1b's filed gate on P1b's six sources (the zero-step three and the regime's column of walks; on squad the zero-step
three twice), is recomputed from the same walks beside it. The verdict refuses unless it equals P1b's filed records.
--limit Q (gather, apply): a development run on the first Q questions of each carve. It writes nothing and logs no
number, only that every step ran, the features are finite, and P1b's six sources rebuilt from the nine equal P1b's own
sources on the regime's union, array for array. apply reads a uniform gate when none is fit.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import p1b_restart_mixture as B  # noqa: E402
import p2a_reach as R2  # noqa: E402

P, E, F, S4, S4B = B.P, B.E, B.F, B.S4, R2.S4B
OUT = ROOT / "outputs" / "p2b"
GATE = OUT / "gate.json"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
NS, NMAX, TIE, KB, SIX, TRAIN5 = B.NS, B.NMAX, B.TIE, B.KB, B.SIX, B.TRAIN5
UNIONS = (("st", ("structural",)), ("all", tuple(S4.FAMILIES)))
WALKS = ("S_st", "L_st", "R_st", "S_all", "L_all", "R_all")
SOURCES = ("S0", "L0", "R0") + WALKS
FEATURES = (B.FEATURES[:9] + tuple(f"{w}_top" for w in WALKS) + tuple(f"{w}_top10" for w in WALKS)
            + tuple(f"deg_{f}" for f in S4.FAMILIES))
COLUMN = {"S": 3, "SNK": 6, "none": 0}       # where P1b's three walked sources sit among the nine, by regime
MEASURES = ("ALL", "recall")
DIAG = ("MXU", "MX") + SOURCES
log, utc, write_json, sha_file = B.log, B.utc, B.write_json, B.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the nine sources and their features ──────────────────────────────────────


def regime_of(families, construction):
    """The regime's family code (S, SNK or none), checked against the families step 4e opened."""
    reg = R2.REGIME[construction["regime"]]
    if R2.canon("".join(c for c in "SNK" if R2.FAM[c] in families)) != reg:
        raise SystemExit(f"the regime's families {families} are not {reg}")
    return reg


def open_unions(cache, op, name, union):
    """All three families' stores (P2a's) and the two unions MXU walks, opened once per dataset."""
    if "unions" not in cache:
        stores, n = R2.family_stores(op, name, union)
        cache["stores"] = stores
        cache["unions"] = {u: S4B.RegimeUnion(stores, list(fams), n) for u, fams in UNIONS}
    return cache["stores"], cache["unions"]


def sources_batch(prep, js, linked, unions, constant, threads):
    """Per question of js: the nine SOURCES, each (nodes, masses) by mass."""
    zs = [B.zero_step(prep, j, linked[j], constant) for j in js]
    out = [list(z) for z in zs]
    for u, _fams in UNIONS:
        walked = [B.walk_mass(unions[u], [z[k] for z in zs], threads) for k in range(3)]
        for i in range(len(zs)):
            out[i] += [walked[k][i] for k in range(3)]
    return [tuple(s) for s in out]


def p1b_six(src, regime):
    """P1b's six sources (S0, L0, R0, S, L, R) from the nine: the regime's column of walks (squad: the zero-step
    three again, as P1b has them)."""
    c = COLUMN[regime]
    return tuple(src[:3]) + tuple(src[c:c + 3])


def same_sources(a, b):
    return len(a) == len(b) and all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def seed_degrees(stores, seeds):
    """For each family, the mean over the seeds of log(1 + the seed's degree in that family)."""
    s = np.asarray(seeds, np.int64)
    if s.size == 0:
        return [0.0] * len(S4.FAMILIES)
    return [float(np.log1p((stores[f].indptr[s + 1] - stores[f].indptr[s]).astype(np.float64)).mean())
            for f in S4.FAMILIES]


def features(prep, j, src, stores):
    """FEATURES, raw: P1b's nine of the lists and links, each walk's top mass (log) and top-10 mass, and the families'
    degrees at the seeds. Never a gold."""
    x = list(B.features(prep, j, p1b_six(src, "S"))[:9])
    x += [float(np.log(max(w[1][0], B.FLOOR))) if w[1].size else float(np.log(B.FLOOR)) for w in src[3:]]
    x += [float(w[1][:10].sum()) for w in src[3:]]
    x += seed_degrees(stores, src[0][0])
    return np.asarray(x, np.float64)


def gold_masses(src, golds):
    """(|golds|, len(src)): each gold's mass under each source, 0 where the source does not hold it."""
    out = np.zeros((golds.size, len(src)))
    for k, (nodes, mass) in enumerate(src):
        pos = P.positions(nodes, golds)
        out[:, k] = np.where(pos >= 0, mass[np.maximum(pos, 0)] if mass.size else 0.0, 0.0)
    return out


def dev_check(name, prep, js, linked, union, constant, threads, srcs, regime):
    """A development run's check: P1b's six sources rebuilt from the nine equal P1b's own on the regime's union."""
    own = B.sources_batch(name, prep, js, linked, union, constant, threads)
    bad = sum(not same_sources(p1b_six(s, regime), o) for s, o in zip(srcs, own))
    if bad:
        raise SystemExit(f"{name}: P1b's six sources rebuilt from the nine differ from P1b's own on {bad} questions")


def on_the_mirror():
    import importlib.util
    spec = importlib.util.spec_from_file_location("s4f_run_one", ROOT / "outputs" / "step4f" / "run_one.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.on_the_mirror(F)


# ── gather (fit carves) ──────────────────────────────────────────────────────


def gather_path(name):
    return OUT / f"gather_{name}.npz"


def gather_stage(name, threads, limit=0):
    if name not in TRAIN5:
        raise SystemExit(f"{name} never fits")
    op = S4.Opened()
    carve = B.fit_carve(name)
    cache = {}

    def sink(name, carve_, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        regime = regime_of(families, construction)
        stores, unions = open_unions(cache, op, name, union)
        nq = min(limit, len(U)) if limit else len(U)
        X, GM, gptr, reached, golds_n = [], [], [0], np.zeros(len(SOURCES)), 0
        for lo in range(0, nq, S4.BATCH):
            js = range(lo, min(nq, lo + S4.BATCH))
            srcs = sources_batch(prep, js, linked, unions, constant, threads)
            if limit:
                dev_check(name, prep, js, linked, union, constant, threads, srcs, regime)
            for j, src in zip(js, srcs):
                g = np.unique(np.asarray(pop.golds[j], np.int64))
                X.append(features(prep, j, src, stores))
                gm = gold_masses(src, g)
                GM.append(gm)
                gptr.append(gptr[-1] + g.size)
                reached += (gm > 0).sum(axis=0)
                golds_n += g.size
        X, GM = np.vstack(X), np.vstack(GM)
        if not np.isfinite(X).all():
            raise SystemExit(f"{name}/{carve_}: a feature is not finite")
        if limit:
            log(f"{name}/{carve_}: development run on {nq} questions ({time.time() - t:.0f}s): every step ran, the "
                "features are finite, and P1b's six sources rebuilt from the nine equal P1b's own; nothing written")
            return
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = OUT / f"gather_{name}.tmp.npz"
        np.savez(tmp, X=X, GM=GM, gptr=np.asarray(gptr, np.int64), ids=np.asarray(list(pop.ids)))
        os.replace(tmp, gather_path(name))
        rec = {"declared_in": "docs/P2B_FAMILY_GATE.md", "dataset": name, "carve": carve_, "questions": nq,
               "carve_sha256": pop.digest, "regime": regime, "golds": int(golds_n), "features": list(FEATURES),
               "sources": list(SOURCES),
               "golds_reached_by_source": {s: round(float(c) / max(1, golds_n), 4) for s, c in zip(SOURCES, reached)},
               "golds_reached_by_any": round(float((GM.sum(axis=1) > 0).mean()), 4),
               "npz_sha256": sha_file(gather_path(name)), "script_sha256": script_sha256(),
               "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "seconds": round(time.time() - t, 1), "utc": utc()}
        write_json(OUT / f"gather_{name}.json", rec)
        log(f"{name}/{carve_} ({nq} q, {rec['seconds']:.0f}s): golds reached " + ", ".join(
            f"{s} {v}" for s, v in rec["golds_reached_by_source"].items()) + f"; any {rec['golds_reached_by_any']}")

    E.run_carves(op, name, {carve}, ("A3",), threads, sink)
    return 0


# ── the EM fit (P1b's) ───────────────────────────────────────────────────────


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
    W, trace = B.em((X - mu) / sd, GM, gptr, qw)
    gate = {"declared_in": "docs/P2B_FAMILY_GATE.md", "sources": list(SOURCES), "features": list(FEATURES),
            "W": W.round(10).tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "lambda": B.LAMBDA,
            "em_iterations": len(trace) - 1, "trace": [round(v, 8) for v in trace],
            "gathers": {n: r["npz_sha256"] for n, (_d, r) in parts.items()},
            "gather_script_sha256": next(iter(shas))[0], "freeze_RECORD_SHA256": next(iter(shas))[1],
            "script_sha256": script_sha256(), "seconds": round(time.time() - t, 1), "utc": utc()}
    gate["fit_mean_weights"] = {name: {s: round(float(v), 4) for s, v in zip(SOURCES, B.gate_weights(gate, d["X"]).mean(axis=0))}
                                for name, (d, _r) in parts.items()}
    write_json(GATE, gate)
    log(f"gate: {gate['em_iterations']} EM iterations, penalized log-likelihood {trace[0]:.4f} -> {trace[-1]:.4f}")
    for name, mw in gate["fit_mean_weights"].items():
        log(f"  {name}: " + ", ".join(f"{s} {v}" for s, v in mw.items()))
    return 0


# ── apply (s1sel, s1eval) ────────────────────────────────────────────────────


def part_path(name):
    return OUT / f"mxu_{name}.json"


def uniform_gate():
    return {"W": np.zeros((len(SOURCES), len(FEATURES) + 1)).tolist(), "mu": [0.0] * len(FEATURES),
            "sd": [1.0] * len(FEATURES)}


def apply_stage(name, threads, limit=0):
    op = S4.Opened()
    if GATE.exists():
        gate = json.loads(GATE.read_text(encoding="utf-8"))
        if not limit and gate["script_sha256"] != script_sha256():
            raise SystemExit("the gate was fit by another version of this script")
        if not limit and gate["freeze_RECORD_SHA256"] != op.freeze["RECORD_SHA256"]:
            raise SystemExit("the gate was fit on another freeze")
    elif limit:
        gate = uniform_gate()
    else:
        raise SystemExit(f"{GATE} is missing")
    pg = json.loads(B.GATE.read_text(encoding="utf-8"))
    if pg["freeze_RECORD_SHA256"] != op.freeze["RECORD_SHA256"]:
        raise SystemExit("P1b's gate was fit on another freeze")
    rec = {"declared_in": "docs/P2B_FAMILY_GATE.md", "dataset": name, "ns": list(NS), "alpha": E.ALPHA, "eps": S4.EPS,
           "gate_sha256": sha_file(GATE) if GATE.exists() else None, "p1b_gate_sha256": sha_file(B.GATE),
           "script_sha256": script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "limit": limit,
           "carves": {}}
    cache = {}

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        regime = regime_of(families, construction)
        stores, unions = open_unions(cache, op, name, union)
        nq = min(limit, len(U)) if limit else len(U)
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds[:nq]]
        p4e = P.p4e_pools(name, carve, pop)
        pos = {o: [] for o in DIAG}
        lens = {o: [] for o in DIAG}
        wsum, wsum_mx = np.zeros(len(SOURCES)), np.zeros(len(B.SOURCES))
        for lo in range(0, nq, S4.BATCH):
            js = range(lo, min(nq, lo + S4.BATCH))
            srcs = sources_batch(prep, js, linked, unions, constant, threads)
            if limit:
                dev_check(name, prep, js, linked, union, constant, threads, srcs, regime)
            six = [p1b_six(s, regime) for s in srcs]
            XU = np.vstack([features(prep, j, s, stores) for j, s in zip(js, srcs)])
            if not np.isfinite(XU).all():
                raise SystemExit(f"{name}/{carve}: a feature is not finite")
            Wu = B.gate_weights(gate, XU)
            Wx = B.gate_weights(pg, np.vstack([B.features(prep, j, s) for j, s in zip(js, six)]))
            wsum += Wu.sum(axis=0)
            wsum_mx += Wx.sum(axis=0)
            for i, (j, src) in enumerate(zip(js, srcs)):
                orders = {"MXU": B.mix_order(src, Wu[i]), "MX": B.mix_order(six[i], Wx[i])}
                for k, s in enumerate(SOURCES):
                    orders[s] = src[k][0][:NMAX]
                for o in DIAG:
                    pos[o].append(P.positions(orders[o], golds[j]))
                    lens[o].append(int(orders[o].size))
        sizes = {"I_q": np.asarray([np.asarray(inc[j]).size for j in range(nq)]),
                 "P4e": np.asarray([p4e[j].size for j in range(nq)])}
        ent = {"questions": nq, "carve_sha256": pop.digest, "regime": regime,
               "mean_weights": {s: round(float(v) / max(1, nq), 4) for s, v in zip(SOURCES, wsum)},
               "mean_weights_mx": {s: round(float(v) / max(1, nq), 4) for s, v in zip(B.SOURCES, wsum_mx)},
               "orders": {o: {"curve": P.curve(pos[o], lens[o]), "len_mean": round(float(np.mean(lens[o])), 1),
                              "at_I_q": P.at_sizes(pos[o], sizes["I_q"]), "at_P4e": P.at_sizes(pos[o], sizes["P4e"])}
                          for o in DIAG},
               "sets": {"I_q": P.set_cov([np.asarray(inc[j], np.int64) for j in range(nq)], golds),
                        "P4e": P.set_cov(p4e[:nq], golds)},
               "seconds": round(time.time() - t, 1)}
        if limit:
            log(f"{name}/{carve}: development run on {nq} questions ({time.time() - t:.0f}s): every step ran, the "
                "features are finite, and P1b's six sources rebuilt from the nine equal P1b's own; nothing written")
            return
        OUT.mkdir(parents=True, exist_ok=True)
        ppath = OUT / f"pos_{name}__{carve}.npz"
        tmp = OUT / f"pos_{name}__{carve}.tmp.npz"
        np.savez(tmp, ptr=np.concatenate([[0], np.cumsum([g.size for g in golds])]).astype(np.int64),
                 MXU=np.concatenate(pos["MXU"]).astype(np.int64), MX=np.concatenate(pos["MX"]).astype(np.int64))
        os.replace(tmp, ppath)
        ent["positions_npz_sha256"] = sha_file(ppath)
        rec["carves"][f"{name}__{carve}"] = ent
        rec["utc"] = utc()
        write_json(part_path(name), rec)
        log(f"{name}/{carve} ({nq} q, {ent['seconds']:.0f}s): MXU weights " + ", ".join(
            f"{s} {v}" for s, v in ent["mean_weights"].items()) + " | " + " | ".join(
            f"{o} " + " ".join(f"{n}:{ent['orders'][o]['curve'][str(n)]['ALL']}" for n in (100, 1000, 7000))
            + f" @I_q {ent['orders'][o]['at_I_q']['ALL']}" for o in ("MXU", "MX")))

    wanted = {"s1eval"} | ({"s1sel"} if name in TRAIN5 else set())
    E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


# ── the verdict (section 4 of the declared file) ─────────────────────────────


# Bug fix, 13 Oct, before the verdict (docs/P2B_FAMILY_GATE.md, section 4): P1b's apply listed S and L twice
# (DIAG = ("MX", "S", "L", "A1MIX") + SOURCES, and SOURCES holds S and L), so each question's positions went into
# pos["S"] and pos["L"] twice. P1b's curves are means over those lists and are unaffected. Its at-size values are not
# the walks' values: at_sizes zips the doubled list with the nq sizes and divides by its length, about half. Rebuilt
# from P1b's own walks on metaqa's s1sel, the doubled reading equals P1b's filed values exactly and the single reading
# equals P2b's. P1b's consistency check established its S and L as P1a's A0 and LINKED at every N, and P1a filed those
# walks' values at the same two sizes once per question, so the at-size points of S and L are checked against them.
AT_SIZE_FROM_P1A = {"S": "A0", "L": "LINKED"}


def check_against_p1b(ent, p1b_ent, name, p1a_ent):
    """MX and the zero-step sources against P1b's filed records at every point; on the graph datasets, the regime's
    column of walks against P1b's S, L and R (S and L at the two sizes against P1a's A0 and LINKED, see above)."""
    pairs = [("MX", "MX"), ("S0", "S0"), ("L0", "L0"), ("R0", "R0")]
    if name != "squad":
        col = "st" if ent["regime"] == "S" else "all"
        pairs += [(f"S_{col}", "S"), (f"L_{col}", "L"), (f"R_{col}", "R")]
    bad = []
    for ours, theirs in pairs:
        a, b = ent["orders"][ours], p1b_ent["orders"][theirs]
        for n in NS:
            for m in MEASURES:
                if a["curve"][str(n)][m] != b["curve"][str(n)][m]:
                    bad.append({"order": ours, "p1b": theirs, "N": n, "measure": m, "p2b": a["curve"][str(n)][m],
                                "p1b_value": b["curve"][str(n)][m]})
        src, ref = ("p1a", p1a_ent["orders"][AT_SIZE_FROM_P1A[theirs]]) if theirs in AT_SIZE_FROM_P1A else ("p1b", b)
        for s in ("at_I_q", "at_P4e"):
            if a[s] != ref[s]:
                bad.append({"order": ours, src: AT_SIZE_FROM_P1A.get(theirs, theirs), "size": s, "p2b": a[s],
                            f"{src}_value": ref[s]})
    return bad


def p1b_doubled(p1b_ent, p1a_ent):
    """P1b's filed S and L at-size values beside P1a's A0 and LINKED (the same walks, read once per question)."""
    return {o: {s: {"p1b_filed": p1b_ent["orders"][o][s], "p1a": p1a_ent["orders"][r][s]} for s in ("at_I_q", "at_P4e")}
            for o, r in AT_SIZE_FROM_P1A.items() if o in p1b_ent["orders"] and r in p1a_ent["orders"]}


def compare(ent, a="MXU", b="MX"):
    """a minus b at every N and both measures, its label, and the differences at the matched sizes."""
    pts = [{"N": n, "measure": m,
            "d": round(ent["orders"][a]["curve"][str(n)][m] - ent["orders"][b]["curve"][str(n)][m], 4)}
           for n in NS for m in MEASURES]
    up = [p for p in pts if p["d"] > TIE]
    down = [p for p in pts if p["d"] < -TIE]
    out = {"label": "MIXED" if up and down else "ABOVE" if up else "BELOW" if down else "TIE", "above": up,
           "below": down, "largest": max(p["d"] for p in pts), "smallest": min(p["d"] for p in pts)}
    for s in ("at_I_q", "at_P4e"):
        out[s] = {m: round(ent["orders"][a][s][m] - ent["orders"][b][s][m], 4) for m in MEASURES}
    return out


def the_call(parts):
    """Section 4: ONE_RULE when MXU is nowhere below MX at |I_q| on s1eval, above it somewhere, and not below it on
    the five s1sel carves' mean ALL at |I_q|; REGIME_STAYS otherwise."""
    ev = {n: compare(parts[n]["carves"][f"{n}__s1eval"]) for n in SIX}
    short = [{"dataset": n, "measure": m, "d": ev[n]["at_I_q"][m]} for n in SIX for m in MEASURES
             if ev[n]["at_I_q"][m] < -TIE]
    gain = [{"dataset": n, "measure": m, "d": ev[n]["at_I_q"][m]} for n in SIX for m in MEASURES
            if ev[n]["at_I_q"][m] > TIE]
    sel = {o: round(float(np.mean([parts[n]["carves"][f"{n}__s1sel"]["orders"][o]["at_I_q"]["ALL"] for n in TRAIN5])),
                    4) for o in ("MXU", "MX")}
    sel["d"] = round(sel["MXU"] - sel["MX"], 4)
    call = "ONE_RULE" if not short and gain and sel["d"] >= -TIE else "REGIME_STAYS"
    return {"call": call, "s1eval": ev, "short": short, "gain": gain, "s1sel_mean_ALL_at_I_q": sel}


def verdict_stage():
    for path in (GATE, B.GATE, P.VERDICT):
        if not path.exists():
            raise SystemExit(f"{path} is missing")
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    parts, p1b, p1a = {}, {}, {}
    for name in SIX:
        for d, path in ((parts, part_path(name)), (p1b, B.part_path(name)), (p1a, P.part_path(name))):
            if not path.exists():
                raise SystemExit(f"{path} is missing")
            d[name] = json.loads(path.read_text(encoding="utf-8"))
    keys = {(r["script_sha256"], r["gate_sha256"], r["p1b_gate_sha256"], r["freeze_RECORD_SHA256"])
            for r in parts.values()}
    if len(keys) != 1:
        raise SystemExit("the parts ran on different scripts, gates or freezes")
    script, gate_sha, p1b_gate_sha, freeze = next(iter(keys))
    if gate_sha != sha_file(GATE) or gate["script_sha256"] != script or gate["gather_script_sha256"] != script:
        raise SystemExit("the gathers, the gate and the reads did not run on one script and one gate")
    if p1b_gate_sha != sha_file(B.GATE):
        raise SystemExit("P1b's gate is not the one the parts read")
    if any(r.get("limit") for r in parts.values()):
        raise SystemExit("a part ran with --limit")
    checks, doubled = {}, {}
    for name in SIX:
        want = {f"{name}__s1eval"} | ({f"{name}__s1sel"} if name in TRAIN5 else set())
        if set(parts[name]["carves"]) != want:
            raise SystemExit(f"{name}: carves {sorted(parts[name]['carves'])}, not {sorted(want)}")
        for c, ent in parts[name]["carves"].items():
            pb = p1b[name]["carves"].get(c)
            if pb is None:
                raise SystemExit(f"P1b has no {c}")
            if ent["questions"] != pb["questions"] or ent["carve_sha256"] != pb["carve_sha256"]:
                raise SystemExit(f"{c}: not P1b's questions")
            pa = p1a[name]["carves"].get(c)
            if pa is None or pa["questions"] != ent["questions"]:
                raise SystemExit(f"P1a has no {c} on these questions")
            bad = check_against_p1b(ent, pb, name, pa)
            if bad:
                write_json(OUT / "check_failed.json", {"carve": c, "points": bad})
                raise SystemExit(f"{c}: MXU's records differ from P1b's at {len(bad)} points (outputs/p2b/"
                                 f"check_failed.json), first {bad[0]}")
            checks[c] = "MX, S0, L0 and R0 equal P1b's at every point" + (
                "" if name == "squad" else "; the regime's walks equal P1b's S, L and R at every N, R at both sizes, "
                                           "and S and L at both sizes equal P1a's A0 and LINKED")
            if name != "squad":
                doubled[c] = p1b_doubled(pb, pa)
    res = the_call(parts)
    sota = {}
    for name in SIX:
        ent = parts[name]["carves"][f"{name}__s1eval"]
        sota[name] = B.call({"orders": {"MX": ent["orders"]["MXU"]}, "sets": ent["sets"]},
                            p1a[name]["carves"][f"{name}__s1eval"], name)
    rec = {"declared_in": "docs/P2B_FAMILY_GATE.md", **res, "beside_vs_sota": sota, "checks": checks,
           "p1b_doubled_at_size": doubled, "verdict_script_sha256": script_sha256(),
           "fit_mean_weights": gate["fit_mean_weights"], "script_sha256": script, "gate_sha256": gate_sha,
           "p1b_gate_sha256": p1b_gate_sha, "freeze_RECORD_SHA256": freeze,
           "parts": {n: sha_file(part_path(n)) for n in SIX}, "p1b_parts": {n: sha_file(B.part_path(n)) for n in SIX},
           "p1a_parts": {n: sha_file(P.part_path(n)) for n in SIX}, "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(parts, p1a, res, sota, gate), encoding="utf-8", newline="\n")
    log(f"P2b: {res['call']}; at I_q on s1eval (ALL): " + ", ".join(
        f"{n} {res['s1eval'][n]['at_I_q']['ALL']:+.4f} ({res['s1eval'][n]['label']})" for n in SIX)
        + f"; s1sel mean ALL at I_q MXU {res['s1sel_mean_ALL_at_I_q']['MXU']} vs MX {res['s1sel_mean_ALL_at_I_q']['MX']}")
    return 0


def report_md(parts, p1a, res, sota, gate):
    show = (25, 100, 500, 1000, 2000, 7000)
    L = [f"# P2b: the families' walks weighted per question, against P1b's MX (s1eval): {res['call']}", "",
         "ALL (every gold in the pool) at N, gold recall in brackets. MXU, MX and the nine sources from "
         "outputs/p2b/mxu_<dataset>.json; the SOTA orders from P1a's records.", ""]
    for name in SIX:
        c = f"{name}__s1eval"
        ent, pa, cl, so = parts[name]["carves"][c], p1a[name]["carves"][c], res["s1eval"][name], sota[name]
        L += [f"## {name} (regime {ent['regime']}, {ent['questions']} questions): MXU against MX {cl['label']}", "",
              "| order | " + " | ".join(f"N = {n:,}" for n in show) + " | at I_q |", "| --- |" + " ---: |" * (len(show) + 1)]
        rows = [(o, ent["orders"][o]) for o in DIAG] + [(f"{o} (P1a)", pa["orders"][o]) for o in P.SOTA]
        for o, od in rows:
            L.append(f"| {o} | " + " | ".join(f"{od['curve'][str(n)]['ALL']:.3f} ({od['curve'][str(n)]['recall']:.3f})"
                                              for n in show) + f" | {od['at_I_q']['ALL']:.3f} |")
        L += ["", f"- At |I_q| ({ent['sets']['I_q']['size']:,.0f} rows mean), MXU minus MX: ALL "
              f"{cl['at_I_q']['ALL']:+.4f}, recall {cl['at_I_q']['recall']:+.4f}. The set itself: "
              f"{ent['sets']['I_q']['ALL']:.3f}.",
              f"- Over the grid: largest {cl['largest']:+.4f}, smallest {cl['smallest']:+.4f}.",
              "- MXU's mean weights: " + ", ".join(f"{s} {v}" for s, v in ent["mean_weights"].items()) + ".",
              "- MX's mean weights: " + ", ".join(f"{s} {v}" for s, v in ent["mean_weights_mx"].items()) + ".",
              f"- MXU against the best SOTA order (P1b's rule, beside): {so['call']}"
              + (f", below at {len(so['below'])} points, the largest gap {so['largest_gap']:+.4f}" if so["below"] else "")
              + (f"; above at {len(so['above'])} points, the largest lead {so['largest_lead']:+.4f}" if so["above"] else "")
              + ".", ""]
    sel = res["s1sel_mean_ALL_at_I_q"]
    L += ["## The call", "",
          f"- Below MX by more than {TIE} at |I_q| on s1eval: "
          + (", ".join(f"{s['dataset']} {s['measure']} {s['d']:+.4f}" for s in res["short"]) or "nowhere") + ".",
          f"- Above MX by more than {TIE} at |I_q| on s1eval: "
          + (", ".join(f"{s['dataset']} {s['measure']} {s['d']:+.4f}" for s in res["gain"]) or "nowhere") + ".",
          f"- The five s1sel carves' mean ALL at |I_q|: MXU {sel['MXU']:.4f}, MX {sel['MX']:.4f} ({sel['d']:+.4f}).",
          f"- **{res['call']}**.", "",
          "The gate's mean weights on the fit carves:", ""]
    for name, mw in gate["fit_mean_weights"].items():
        L.append(f"- {name}: " + ", ".join(f"{s} {v}" for s, v in mw.items()))
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _fake_order(v, size=100.0):
    return {"curve": {str(n): {"ALL": v, "recall": v, "pool": float(n)} for n in NS},
            "at_I_q": {"ALL": v, "recall": v, "size": size}, "at_P4e": {"ALL": v, "recall": v, "size": size}}


def verdict_selftest():
    """The verdict's checks, labels and call on made-up records, in a temporary folder."""
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="p2b_selftest_"))
    saved = (globals()["OUT"], globals()["GATE"], globals()["VERDICT"], globals()["VERDICT_MD"], B.OUT, B.GATE,
             P.OUT, P.VERDICT)
    try:
        globals().update(OUT=tmp / "p2b", GATE=tmp / "p2b" / "gate.json", VERDICT=tmp / "p2b" / "verdict.json",
                         VERDICT_MD=tmp / "p2b" / "verdict.md")
        B.OUT, B.GATE, P.OUT, P.VERDICT = tmp / "p1b", tmp / "p1b" / "gate.json", tmp / "p1a", tmp / "p1a" / "verdict.json"
        for d in ("p2b", "p1b", "p1a"):
            (tmp / d).mkdir(parents=True)
        write_json(P.VERDICT, {})
        write_json(B.GATE, {"W": []})
        write_json(GATE, {"script_sha256": "s", "gather_script_sha256": "s",
                          "fit_mean_weights": {n: {s: round(1 / 9, 4) for s in SOURCES} for n in TRAIN5}})
        regimes = {"metaqa": "S", "squad": "none", "musique": "SNK", "hotpotqa": "S", "2wiki": "S", "webqsp": "SNK"}

        def write_all(mxu_on=None, mxu_d=0.0, mx_off=None, a0_off=None):
            for n in SIX:
                carves = [f"{n}__s1eval"] + ([f"{n}__s1sel"] if n in TRAIN5 else [])
                ours = {"script_sha256": "s", "gate_sha256": sha_file(GATE), "p1b_gate_sha256": sha_file(B.GATE),
                        "freeze_RECORD_SHA256": "f", "limit": 0, "carves": {}}
                theirs = {"carves": {}}
                pa = {"carves": {}}
                for c in carves:
                    ent = {"questions": 10, "carve_sha256": "c", "regime": regimes[n],
                           "mean_weights": {s: 0.1 for s in SOURCES}, "mean_weights_mx": {s: 0.1 for s in B.SOURCES},
                           "orders": {o: _fake_order(0.5) for o in DIAG},
                           "sets": {"I_q": {"ALL": 0.9, "recall": 0.9, "size": 100.0},
                                    "P4e": {"ALL": 0.9, "recall": 0.9, "size": 200.0}}}
                    if n == mxu_on and c.endswith("s1eval"):
                        ent["orders"]["MXU"] = _fake_order(round(0.5 + mxu_d, 4))
                    ours["carves"][c] = ent
                    tb = {"questions": 10, "carve_sha256": "c",
                          "orders": {o: _fake_order(0.5) for o in ("MX", "S0", "L0", "R0", "S", "L", "R")}}
                    if n == mx_off and c.endswith("s1eval"):
                        tb["orders"]["MX"]["curve"]["100"]["ALL"] = 0.51
                    for o in ("S", "L"):      # P1b's doubled lists: its filed at-size values are about half
                        for s in ("at_I_q", "at_P4e"):
                            tb["orders"][o][s] = {"ALL": 0.25, "recall": 0.25, "size": 100.0}
                    theirs["carves"][c] = tb
                    po = {o: _fake_order(0.49) for o in ("A3", "A0", "ORACLE") + P.SOTA}
                    for o in ("A0", "LINKED"):    # the same walks as P1b's S and L, filed once per question
                        for s in ("at_I_q", "at_P4e"):
                            po[o][s] = {"ALL": 0.5, "recall": 0.5, "size": 100.0}
                    if n == a0_off and c.endswith("s1sel"):
                        po["A0"]["at_P4e"]["recall"] = 0.51
                    pa["carves"][c] = {"questions": 10, "orders": po}
                write_json(part_path(n), ours)
                write_json(B.part_path(n), theirs)
                write_json(P.part_path(n), pa)

        write_all("webqsp", 0.01)
        assert verdict_stage() == 0
        v = json.loads(VERDICT.read_text(encoding="utf-8"))
        assert v["call"] == "ONE_RULE" and v["s1eval"]["webqsp"]["label"] == "ABOVE", v["call"]
        assert all(v["s1eval"][n]["label"] == "TIE" for n in SIX if n != "webqsp")
        assert v["beside_vs_sota"]["webqsp"]["call"] == "AT_OR_ABOVE"
        assert v["p1b_doubled_at_size"]["metaqa__s1sel"]["S"]["at_I_q"] == {
            "p1b_filed": {"ALL": 0.25, "recall": 0.25, "size": 100.0}, "p1a": {"ALL": 0.5, "recall": 0.5, "size": 100.0}}
        assert "squad__s1eval" not in v["p1b_doubled_at_size"] and len(v["p1b_doubled_at_size"]) == 9
        assert VERDICT_MD.read_text(encoding="utf-8").startswith("# P2b: the families' walks weighted per question")
        write_all("squad", -0.01)
        verdict_stage()
        v = json.loads(VERDICT.read_text(encoding="utf-8"))
        assert v["call"] == "REGIME_STAYS" and v["short"][0]["dataset"] == "squad" and v["s1eval"]["squad"]["label"] == "BELOW"
        write_all()
        verdict_stage()
        assert json.loads(VERDICT.read_text(encoding="utf-8"))["call"] == "REGIME_STAYS"      # no gain anywhere
        write_all("webqsp", 0.01, mx_off="musique")
        try:
            verdict_stage()
            raise AssertionError("a recomputed MX that differs from P1b's was not refused")
        except SystemExit as e:
            assert "differ from P1b's" in str(e), e
        write_all("webqsp", 0.01, a0_off="hotpotqa")
        try:
            verdict_stage()
            raise AssertionError("a walk whose at-size values differ from P1a's A0 was not refused")
        except SystemExit as e:
            assert "hotpotqa__s1sel" in str(e) and "'p1a': 'A0'" in str(e), e
    finally:
        (globals()["OUT"], globals()["GATE"], globals()["VERDICT"], globals()["VERDICT_MD"], B.OUT, B.GATE, P.OUT,
         P.VERDICT) = saved
        shutil.rmtree(tmp, ignore_errors=True)


def selftest():
    from types import SimpleNamespace
    n = 30
    edges = {"structural": [(i, i + 1) for i in range(12)] + [(0, 20), (20, 21)],
             "ner": [(1, 13), (13, 14), (5, 15), (2, 22)],
             "knn": [(15, 16), (3, 17), (17, 18), (25, 26), (22, 23)]}
    stores = {f: B._Store(n, e) for f, e in edges.items()}
    unions = {u: S4B.RegimeUnion(stores, list(fams), n) for u, fams in UNIONS}
    # each union's walks equal step 4e's walk on a union of those families alone
    rs = [(np.asarray([0, 3]), np.asarray([0.75, 0.25])), (np.asarray([22]), np.asarray([1.0]))]
    for u, fams in UNIONS:
        ref = S4.Union({f: stores[f] if f in fams else S4B.Empty(n) for f in S4.FAMILIES})
        got = B.walk_mass(unions[u], rs, 2)
        want = E.walk(ref, rs, [np.empty(0, np.int64)] * 2, NMAX, 2)
        assert all(g[0].tolist() == w.tolist() for g, w in zip(got, want)), u
    # the nine sources on toy questions; P1b's six rebuilt from them equal P1b's own on each regime's union
    rng = np.random.default_rng(0)
    nq = 3
    dense = np.stack([rng.permutation(n)[:12] for _ in range(nq)]).astype(np.int64)
    splade = np.stack([rng.permutation(n)[:12] for _ in range(nq)]).astype(np.int64)
    prep = SimpleNamespace(seeds=[np.union1d(dense[q, :5], splade[q, :5]) for q in range(nq)], dense_ids=dense,
                           splade_ids=splade, dense_scores=np.sort(rng.random((nq, 12)))[:, ::-1],
                           splade_scores=np.sort(rng.random((nq, 12)) * 20)[:, ::-1])
    linked = [np.asarray([2, 21]), np.empty(0, np.int64), np.asarray([25])]
    js = range(nq)
    srcs = sources_batch(prep, js, linked, unions, 60, 2)
    assert all(len(s) == len(SOURCES) for s in srcs) and srcs[1][1][0].size == 0 and srcs[1][4][0].size == 0
    for regime, fams in (("S", ["structural"]), ("SNK", list(S4.FAMILIES))):
        own = B.sources_batch("x", prep, js, linked, S4B.RegimeUnion(stores, fams, n), 60, 2)
        assert all(same_sources(p1b_six(s, regime), o) for s, o in zip(srcs, own)), regime
    own = B.sources_batch("squad", prep, js, linked, S4B.RegimeUnion(stores, [], n), 60, 2)
    assert all(same_sources(p1b_six(s, "none"), o) for s, o in zip(srcs, own))
    assert not same_sources(p1b_six(srcs[0], "S"), p1b_six(srcs[0], "SNK"))
    # features: P1b's nine first, then each walk's top mass and top-10 mass, then the families' degrees at the seeds
    for j, s in zip(js, srcs):
        x = features(prep, j, s, stores)
        assert x.size == len(FEATURES) == 24 and np.isfinite(x).all()
        assert np.array_equal(x[:9], B.features(prep, j, p1b_six(s, "SNK"))[:9])
        for k, w in enumerate(s[3:]):
            assert x[9 + k] == (float(np.log(max(w[1][0], B.FLOOR))) if w[1].size else float(np.log(B.FLOOR)))
            assert x[15 + k] == float(w[1][:10].sum())
        for k, f in enumerate(S4.FAMILIES):
            want = np.mean([np.log1p(sum((a == v) + (b == v) for a, b in edges[f])) for v in s[0][0]])
            assert abs(x[21 + k] - want) < 1e-12, (f, x[21 + k], want)
    # a one-hot mixture reproduces each of the nine sources; gold masses as P1b's on its six
    for k in range(len(SOURCES)):
        w = np.zeros(len(SOURCES))
        w[k] = 1.0
        assert B.mix_order(srcs[0], w).tolist() == srcs[0][k][0].tolist(), k
    g = np.asarray(sorted({int(srcs[0][3][0][0]), int(srcs[0][8][0][-1]), int(srcs[0][2][0][0])}))
    assert np.array_equal(gold_masses(p1b_six(srcs[0], "S"), g), B.gold_masses(p1b_six(srcs[0], "S"), g))
    assert gold_masses(srcs[0], g).shape == (g.size, len(SOURCES))
    # EM over nine sources: monotone, and it picks each question's source
    Q = 400
    X = rng.normal(size=(Q, 2))
    good = np.where(X[:, 0] > 0, 8, 4)
    GM = np.full((Q, len(SOURCES)), 1e-4)
    GM[np.arange(Q), good] = 0.5
    W, trace = B.em(X, GM, np.arange(Q + 1), np.full(Q, 1.0 / Q))
    assert all(b >= a - 1e-9 for a, b in zip(trace, trace[1:])), trace[:5]
    assert (B.softmax(np.hstack([X, np.ones((Q, 1))]) @ W.T).argmax(axis=1) == good).mean() > 0.95
    verdict_selftest()
    print("selftest: ok (unions walk as step 4e on their families alone, P1b's six rebuilt from the nine equal P1b's "
          "own, features, one-hot mixtures, gold masses, EM over nine sources, the verdict's checks, labels and call)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("gather", "fit", "apply", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="a development run on the first Q questions; writes nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage in ("gather", "apply"):
        if a.dataset not in SIX:
            raise SystemExit(f"dataset must be one of {SIX}")
        if a.host:
            on_the_mirror()
        return (gather_stage if a.stage == "gather" else apply_stage)(a.dataset, a.threads, a.limit)
    if a.stage == "fit":
        return fit_stage()
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
