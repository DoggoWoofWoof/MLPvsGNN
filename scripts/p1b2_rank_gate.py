"""P1b2: P1b's restart sources plus HippoRAG 2's own walk, the gate fit by EM to the golds' ranks over fixed blends
(docs/P1B2_RANK_GATE.md). A diagnostic of pools: no model trains or reads, and nothing is adopted.

    python scripts/p1b2_rank_gate.py gather DATASET [--threads 5] [--host]   a training dataset's fit carve
    python scripts/p1b2_rank_gate.py fit                                     EM on the five gathers
    python scripts/p1b2_rank_gate.py apply DATASET [--threads 5] [--host]    s1sel (training datasets) and s1eval
    --limit N (gather, apply): a development run on the first N questions of one carve; it writes nothing and logs
    no number (apply reads a uniform gate), only that every step ran and the features equal P1b's
    python scripts/p1b2_rank_gate.py verdict
    python scripts/p1b2_rank_gate.py --selftest

Seven sources per question: P1b's six (S0, L0, R0, S, L, R) and A1, the walk from HippoRAG 2's restart (half the seeds,
half the RRF lists), on the dataset's regime union with step 4e's kernel. squad has no graph: there A1 is the RRF order
(P1a's A1 there) with its weights scaled to sum 1.
Eight classes, each a fixed blend of the sources' masses: S, L, R and A1 (one source each), LR (half L, half R), SL
(half S, half L), SLR (a third each) and MX (P1b's filed gate). A class's order is its blend's top NMAX nodes, ties by
node position.
A class's evidence for a question is exp(-area / TAU), where area sums (1 - ALL) + (1 - recall) over P1a's N grid with
that question's golds: the call's own measure, read on one question.
The gate pi(x) = softmax(V x) over the classes, x P1b's fifteen label-free features, is fit by EM on the five training
datasets' fit carves (every dataset weighted equally, webqsp never):
    E-step: each question's responsibility over the classes, pi_c(x) e_c normalized;
    M-step: the softmax regression of V on those responsibilities (P1b's em, one row of evidence per question).
The pool at N is the top N of the class the gate rates highest among those with a non-empty order (RC). Reported
beside: the soft read (RCS, the blend sum_c pi_c blend_c), and the oracle (ORC, per question the class with the least
area; it reads the golds).
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

P, E, F, S4 = B.P, B.E, B.F, B.S4
OUT = ROOT / "outputs" / "p1b2"
GATE = OUT / "gate.json"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
NS, NMAX, TIE, KB, SIX, M_KBQA, TRAIN5 = B.NS, B.NMAX, B.TIE, B.KB, B.SIX, B.M_KBQA, B.TRAIN5
SOURCES = B.SOURCES + ("A1",)
CLASSES = ("S", "L", "R", "A1", "LR", "SL", "SLR", "MX")
TAU = 1.0              # the evidence's temperature, in grid points of area
DIAG = ("RC", "RCS", "ORC") + CLASSES
log, utc, write_json, sha_file = B.log, B.utc, B.write_json, B.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── sources, classes and the area ────────────────────────────────────────────


def blend(c, w_mx):
    """Class c's weights over the seven sources; w_mx is P1b's gate's weights over its six."""
    w = np.zeros(len(SOURCES))
    ix = SOURCES.index
    if c in ("S", "L", "R", "A1"):
        w[ix(c)] = 1.0
    elif c == "LR":
        w[ix("L")] = w[ix("R")] = 0.5
    elif c == "SL":
        w[ix("S")] = w[ix("L")] = 0.5
    elif c == "SLR":
        w[ix("S")] = w[ix("L")] = w[ix("R")] = 1.0 / 3.0
    elif c == "MX":
        w[:len(B.SOURCES)] = w_mx
    else:
        raise ValueError(c)
    return w


def rrf_source(d_ids, s_ids, constant):
    """squad's A1: the RRF order of the two lists (ties by node position, P1a's A1 there), its weights scaled to 1."""
    u, w = E.rrf_weights(d_ids, s_ids, constant)
    if u.size == 0:
        return B.EMPTY
    o = np.lexsort((u, -w))
    return u[o], w[o] / w.sum()


def sources7(name, prep, js, linked, union, constant, threads):
    """Per question of js: P1b's six sources and A1, each (nodes, masses) by mass."""
    six = B.sources_batch(name, prep, js, linked, union, constant, threads)
    if name == "squad":
        a1 = [rrf_source(prep.dense_ids[j], prep.splade_ids[j], constant) for j in js]
    else:
        rs = [E.restart("A1", prep.seeds[j], prep.dense_ids[j], prep.splade_ids[j], linked[j], union.deg, constant)
              for j in js]
        a1 = B.walk_mass(union, rs, threads)
    return [s + (a,) for s, a in zip(six, a1)]


def area(order, golds):
    """Sum over the N grid of (1 - ALL) + (1 - recall) for one question (a question without golds misses everywhere,
    as in P1a's curves)."""
    if golds.size == 0:
        return 2.0 * len(NS)
    pos = P.positions(order, golds)
    hit = (pos[None, :] >= 0) & (pos[None, :] < np.asarray(NS)[:, None])
    return float((1.0 - hit.all(axis=1)).sum() + (1.0 - hit.mean(axis=1)).sum())


def class_orders(src, w_mx):
    """A one-source class keeps its source's own order (as P1b read S and L); the others are mixtures."""
    return {c: src[SOURCES.index(c)][0][:NMAX] if c in SOURCES else B.mix_order(src, blend(c, w_mx))
            for c in CLASSES}


def hard_choice(pi, orders):
    """The class the gate rates highest among those with a non-empty order (ties to the first in CLASSES)."""
    avail = np.asarray([orders[c].size > 0 for c in CLASSES])
    if not avail.any():
        return 0
    return int(np.argmax(np.where(avail, pi, -1.0)))


def soft_weights(pi, orders, w_mx):
    avail = np.asarray([orders[c].size > 0 for c in CLASSES], dtype=np.float64)
    p = pi * avail
    p = p / p.sum() if p.sum() > 0 else np.full(len(CLASSES), 1.0 / len(CLASSES))
    return sum(p[k] * blend(c, w_mx) for k, c in enumerate(CLASSES))


def p1b_gate():
    gate = json.loads(B.GATE.read_text(encoding="utf-8"))
    if gate["script_sha256"] != B.script_sha256():
        raise SystemExit("P1b's gate is not its filed script's")
    return gate


# ── gather (fit carves) ──────────────────────────────────────────────────────


def gather_path(name):
    return OUT / f"gather_{name}.npz"


def gather_stage(name, threads, limit=None):
    if name not in TRAIN5:
        raise SystemExit(f"{name} never fits")
    op = S4.Opened()
    gate1 = p1b_gate()
    carve = B.fit_carve(name)
    with np.load(B.gather_path(name)) as z:
        x_p1b, ids_p1b = z["X"], z["ids"]

    def sink(name, carve_, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        nq = len(U) if not limit else min(len(U), limit)
        X, A = [], []
        for lo in range(0, nq, S4.BATCH):
            js = range(lo, min(nq, lo + S4.BATCH))
            srcs = sources7(name, prep, js, linked, union, constant, threads)
            xb = np.vstack([B.features(prep, j, s[:len(B.SOURCES)]) for j, s in zip(js, srcs)])
            wmx = B.gate_weights(gate1, xb)
            for i, (j, src) in enumerate(zip(js, srcs)):
                g = np.unique(np.asarray(pop.golds[j], np.int64))
                orders = class_orders(src, wmx[i])
                A.append([area(orders[c], g) for c in CLASSES])
            X.append(xb)
        X, A = np.vstack(X), np.asarray(A, np.float64)
        ref = (ids_p1b, x_p1b) if not limit else (ids_p1b[:nq], x_p1b[:nq])
        if list(pop.ids)[:nq] != ref[0].tolist() or X.shape != ref[1].shape or not np.array_equal(X, ref[1]):
            raise SystemExit(f"{name}/{carve_}: the features differ from P1b's gather")
        if limit:
            log(f"{name}/{carve_} development run: {nq} questions, features equal P1b's, {A.shape[1]} class areas "
                f"each, {time.time() - t:.0f}s; nothing written")
            return
        tmp = OUT / f"gather_{name}.tmp.npz"
        np.savez(tmp, X=X, A=A, ids=np.asarray(list(pop.ids)))
        os.replace(tmp, gather_path(name))
        best = A.argmin(axis=1)
        tied = (A == A.min(axis=1, keepdims=True)).all(axis=1)
        rec = {"declared_in": "docs/P1B2_RANK_GATE.md", "dataset": name, "carve": carve_, "questions": nq,
               "carve_sha256": pop.digest, "classes": list(CLASSES), "tau": TAU,
               "mean_area": {c: round(float(v), 4) for c, v in zip(CLASSES, A.mean(axis=0))},
               "least_area_share": {c: round(float(((best == k) & ~tied).mean()), 4) for k, c in enumerate(CLASSES)},
               "all_classes_tied": round(float(tied.mean()), 4), "features_equal_p1b": True,
               "npz_sha256": sha_file(gather_path(name)), "p1b_gate_sha256": sha_file(B.GATE),
               "script_sha256": script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
               "seconds": round(time.time() - t, 1), "utc": utc()}
        write_json(OUT / f"gather_{name}.json", rec)
        log(f"{name}/{carve_} ({nq} q, {rec['seconds']:.0f}s): least area " + ", ".join(
            f"{c} {v}" for c, v in rec["least_area_share"].items()) + f"; all tied {rec['all_classes_tied']}")

    OUT.mkdir(parents=True, exist_ok=True)
    if p1b_gate()["freeze_RECORD_SHA256"] != op.freeze["RECORD_SHA256"]:
        raise SystemExit("P1b's gate was fit on another freeze")
    E.run_carves(op, name, {carve}, ("A3",), threads, sink)
    return 0


# ── the EM fit ───────────────────────────────────────────────────────────────


def evidence(A):
    """exp(-area / TAU), shifted per question so that its best class has 1 (the responsibilities do not change)."""
    return np.exp(-(A - A.min(axis=1, keepdims=True)) / TAU)


def fit_stage():
    parts = {}
    for name in TRAIN5:
        rec = json.loads((OUT / f"gather_{name}.json").read_text(encoding="utf-8"))
        if sha_file(gather_path(name)) != rec["npz_sha256"]:
            raise SystemExit(f"{gather_path(name)}: not its record's file")
        with np.load(gather_path(name)) as z:
            parts[name] = ({k: z[k] for k in ("X", "A")}, rec)
    shas = {(r["script_sha256"], r["freeze_RECORD_SHA256"], r["p1b_gate_sha256"]) for _d, r in parts.values()}
    if len(shas) != 1:
        raise SystemExit(f"the gathers ran on {len(shas)} scripts, freezes or P1b gates")
    X = np.vstack([d["X"] for d, _r in parts.values()])
    A = np.vstack([d["A"] for d, _r in parts.values()])
    qw = np.concatenate([np.full(d["X"].shape[0], 1.0 / (len(parts) * d["X"].shape[0])) for d, _r in parts.values()])
    mu = (qw[:, None] * X).sum(axis=0) / qw.sum()
    sd = np.sqrt((qw[:, None] * (X - mu) ** 2).sum(axis=0) / qw.sum())
    sd = np.where(sd > 1e-6, sd, 1.0)
    t = time.time()
    V, trace = B.em((X - mu) / sd, evidence(A), np.arange(X.shape[0] + 1, dtype=np.int64), qw)
    gate = {"declared_in": "docs/P1B2_RANK_GATE.md", "classes": list(CLASSES), "features": list(B.FEATURES),
            "W": V.round(10).tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "lambda": B.LAMBDA, "tau": TAU,
            "em_iterations": len(trace) - 1, "trace": [round(v, 8) for v in trace],
            "gathers": {n: r["npz_sha256"] for n, (_d, r) in parts.items()},
            "gather_script_sha256": next(iter(shas))[0], "freeze_RECORD_SHA256": next(iter(shas))[1],
            "p1b_gate_sha256": next(iter(shas))[2], "script_sha256": script_sha256(),
            "seconds": round(time.time() - t, 1), "utc": utc()}
    fit = {}
    for name, (d, _r) in parts.items():
        pi = B.gate_weights(gate, d["X"])
        fit[name] = {"mean_pi": {c: round(float(v), 4) for c, v in zip(CLASSES, pi.mean(axis=0))},
                     "argmax_share": {c: round(float((pi.argmax(axis=1) == k).mean()), 4) for k, c in enumerate(CLASSES)}}
    gate["fit"] = fit
    write_json(GATE, gate)
    log(f"gate: {gate['em_iterations']} EM iterations, penalized log-likelihood {trace[0]:.4f} -> {trace[-1]:.4f}")
    for name, f in fit.items():
        log(f"  {name}: argmax " + ", ".join(f"{c} {v}" for c, v in f["argmax_share"].items()))
    return 0


# ── apply (s1sel, s1eval) ────────────────────────────────────────────────────


def part_path(name):
    return OUT / f"rc_{name}.json"


def apply_stage(name, threads, limit=None):
    op = S4.Opened()
    gate1 = p1b_gate()
    if limit:
        gate = {"W": np.zeros((len(CLASSES), len(B.FEATURES) + 1)), "mu": np.zeros(len(B.FEATURES)),
                "sd": np.ones(len(B.FEATURES)), "script_sha256": script_sha256(),
                "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "p1b_gate_sha256": sha_file(B.GATE)}
    else:
        gate = json.loads(GATE.read_text(encoding="utf-8"))
    if gate["script_sha256"] != script_sha256():
        raise SystemExit("the gate was fit by another version of this script")
    if gate["freeze_RECORD_SHA256"] != op.freeze["RECORD_SHA256"] or gate["p1b_gate_sha256"] != sha_file(B.GATE):
        raise SystemExit("the gate was fit on another freeze or P1b gate")
    rec = {"declared_in": "docs/P1B2_RANK_GATE.md", "dataset": name, "ns": list(NS), "alpha": E.ALPHA, "eps": S4.EPS,
           "gate_sha256": None if limit else sha_file(GATE), "p1b_gate_sha256": sha_file(B.GATE), "script_sha256": script_sha256(),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "carves": {}}

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        nq = len(U) if not limit else min(len(U), limit)
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
        p4e = P.p4e_pools(name, carve, pop)
        pos = {o: [] for o in DIAG}
        lens = {o: [] for o in DIAG}
        pisum = np.zeros(len(CLASSES))
        rc_n = np.zeros(len(CLASSES))
        orc_n = np.zeros(len(CLASSES))
        for lo in range(0, nq, S4.BATCH):
            js = range(lo, min(nq, lo + S4.BATCH))
            srcs = sources7(name, prep, js, linked, union, constant, threads)
            xb = np.vstack([B.features(prep, j, s[:len(B.SOURCES)]) for j, s in zip(js, srcs)])
            wmx = B.gate_weights(gate1, xb)
            pis = B.gate_weights(gate, xb)
            pisum += pis.sum(axis=0)
            for i, (j, src) in enumerate(zip(js, srcs)):
                orders = class_orders(src, wmx[i])
                k = hard_choice(pis[i], orders)
                areas = [area(orders[c], golds[j]) for c in CLASSES]
                kb = int(np.argmin(areas))
                rc_n[k] += 1
                orc_n[kb] += 1
                orders["RC"] = orders[CLASSES[k]]
                orders["RCS"] = B.mix_order(src, soft_weights(pis[i], orders, wmx[i]))
                orders["ORC"] = orders[CLASSES[kb]]
                for o in DIAG:
                    pos[o].append(P.positions(orders[o], golds[j]))
                    lens[o].append(int(orders[o].size))
        sizes = {"I_q": np.asarray([np.asarray(inc[j]).size for j in range(nq)]),
                 "P4e": np.asarray([p4e[j].size for j in range(nq)])}
        ent = {"questions": nq, "carve_sha256": pop.digest,
               "mean_pi": {c: round(float(v) / max(1, nq), 4) for c, v in zip(CLASSES, pisum)},
               "rc_share": {c: round(float(v) / max(1, nq), 4) for c, v in zip(CLASSES, rc_n)},
               "orc_share": {c: round(float(v) / max(1, nq), 4) for c, v in zip(CLASSES, orc_n)},
               "orders": {o: {"curve": P.curve(pos[o], lens[o]), "len_mean": round(float(np.mean(lens[o])), 1),
                              "at_I_q": P.at_sizes(pos[o], sizes["I_q"]), "at_P4e": P.at_sizes(pos[o], sizes["P4e"])}
                          for o in DIAG},
               "sets": {"I_q": P.set_cov([np.asarray(inc[j], np.int64) for j in range(nq)], golds),
                        "P4e": P.set_cov(p4e, golds)},
               "seconds": round(time.time() - t, 1)}
        if limit:
            log(f"{name}/{carve} development run: {nq} questions, {len(DIAG)} orders and their curves, "
                f"{ent['seconds']:.0f}s; nothing written")
            return
        rec["carves"][f"{name}__{carve}"] = ent
        rec["utc"] = utc()
        write_json(part_path(name), rec)
        log(f"{name}/{carve} ({nq} q, {ent['seconds']:.0f}s): RC " + ", ".join(
            f"{c} {v}" for c, v in ent["rc_share"].items() if v) + " | " + " | ".join(
            f"{o} " + " ".join(f"{n}:{ent['orders'][o]['curve'][str(n)]['ALL']}" for n in (25, 100, 1000, 7000))
            for o in ("RC", "RCS", "ORC", "MX", "A1")))

    OUT.mkdir(parents=True, exist_ok=True)
    wanted = {"s1eval"} | ({"s1sel"} if name in TRAIN5 and not limit else set())
    E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


# ── the verdict (section 4 of the declared file) ─────────────────────────────


def consistency(ent, p1a, p1b, name):
    """The one-source classes against P1a's orders, and class MX against P1b's MX: equal at every N on both measures
    (squad: L and A1 only, since P1a's walk orders there are its RRF order)."""
    pairs = [("L", p1a, "LINKED"), ("A1", p1a, "A1"), ("MX", p1b, "MX")]
    if name != "squad":
        pairs.insert(0, ("S", p1a, "A0"))
    bad = []
    for ours, rec, theirs in pairs:
        for n in NS:
            for m in ("ALL", "recall"):
                a = ent["orders"][ours]["curve"][str(n)][m]
                b = rec["orders"][theirs]["curve"][str(n)][m]
                if a != b:
                    bad.append({"order": ours, "against": theirs, "N": n, "measure": m, "here": a, "there": b})
    return bad


def call(ent, p1a, name, o="RC"):
    below, above = [], []
    for n in NS:
        for m in ("ALL", "recall"):
            ours = ent["orders"][o]["curve"][str(n)][m]
            best_o = max(P.SOTA, key=lambda s: p1a["orders"][s]["curve"][str(n)][m])
            best = p1a["orders"][best_o]["curve"][str(n)][m]
            row = {"N": n, "measure": m, o: ours, "sota": best, "order": best_o, "gap": round(ours - best, 4),
                   "A3": p1a["orders"]["A3"]["curve"][str(n)][m], "MX": ent["orders"]["MX"]["curve"][str(n)][m]}
            if ours < best - TIE:
                below.append(row)
            elif ours > best + TIE:
                above.append(row)
    out = {"call": "BELOW" if below else "AT_OR_ABOVE", "below": below, "above": above,
           "largest_gap": min((b["gap"] for b in below), default=0.0),
           "largest_lead": max((a["gap"] for a in above), default=0.0)}
    for size in ("at_I_q", "at_P4e"):
        best_o = max(P.SOTA, key=lambda s: p1a["orders"][s][size]["ALL"])
        out[size] = {o: ent["orders"][o][size], "MX": ent["orders"]["MX"][size], "A3": p1a["orders"]["A3"][size],
                     "best_sota": best_o, "sota": p1a["orders"][best_o][size],
                     "set": ent["sets"]["I_q" if size == "at_I_q" else "P4e"]}
    if name in KB:
        m = M_KBQA[name]
        ora = p1a["orders"]["ORACLE"]["curve"][str(m)]
        reach = next((n for n in NS if ent["orders"][o]["curve"][str(n)]["ALL"] >= ora["ALL"]), None)
        out["oracle"] = {"m": m, "ORACLE_at_m": ora, "at_m": ent["orders"][o]["curve"][str(m)],
                         "reaches_oracle_ALL_at_N": reach}
    return out


def p1c_pick(parts, head):
    """Section 5: RC if all six are AT_OR_ABOVE; otherwise the higher of RC and MX in mean ALL at |I_q| over the five
    training datasets' s1sel carves (ties to MX)."""
    if all(v == "AT_OR_ABOVE" for v in head.values()):
        return {"construction": "RC", "rule": "all six AT_OR_ABOVE"}
    mean = {o: round(float(np.mean([parts[n]["carves"][f"{n}__s1sel"]["orders"][o]["at_I_q"]["ALL"] for n in TRAIN5])), 4)
            for o in ("RC", "MX")}
    return {"construction": "RC" if mean["RC"] > mean["MX"] else "MX",
            "rule": "the higher mean ALL at |I_q| over the five s1sel carves (ties to MX)", "s1sel_mean_at_I_q": mean}


def verdict_stage():
    if not B.VERDICT.exists():
        raise SystemExit("P1b's verdict is not filed")
    parts, p1a, p1b = {}, {}, {}
    for name in SIX:
        for d, path in ((parts, part_path(name)), (p1a, P.part_path(name)), (p1b, B.part_path(name))):
            if not path.exists():
                raise SystemExit(f"{path} is missing")
            d[name] = json.loads(path.read_text(encoding="utf-8"))
    if len({(r["script_sha256"], r["gate_sha256"], r["freeze_RECORD_SHA256"]) for r in parts.values()}) != 1:
        raise SystemExit("the parts ran on different scripts, gates or freezes")
    calls, beside, checks = {}, {}, {}
    for name in SIX:
        for c, ent in parts[name]["carves"].items():
            if ent["questions"] != p1a[name]["carves"][c]["questions"]:
                raise SystemExit(f"{c}: {ent['questions']} questions here, {p1a[name]['carves'][c]['questions']} in P1a")
            bad = consistency(ent, p1a[name]["carves"][c], p1b[name]["carves"][c], name)
            if bad:
                raise SystemExit(f"{c}: the classes differ from P1a's or P1b's orders at {len(bad)} points, first {bad[0]}")
            checks[c] = "S, L, A1 equal P1a's A0, LINKED, A1 and MX equals P1b's MX at every point" if name != "squad" \
                else "L, A1 equal P1a's LINKED, A1 and MX equals P1b's MX at every point"
            calls[c] = call(ent, p1a[name]["carves"][c], name)
            beside[c] = {o: call(ent, p1a[name]["carves"][c], name, o) for o in ("RCS", "ORC")}
    head = {n: calls[f"{n}__s1eval"]["call"] for n in SIX}
    rec = {"declared_in": "docs/P1B2_RANK_GATE.md", "calls": calls, "beside": beside, "checks": checks, "s1eval": head,
           "all_six_at_or_above": all(v == "AT_OR_ABOVE" for v in head.values()), "p1c": p1c_pick(parts, head),
           "p1a_parts": {n: sha_file(P.part_path(n)) for n in SIX}, "p1b_parts": {n: sha_file(B.part_path(n)) for n in SIX},
           "gate_sha256": sha_file(GATE), "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(parts, p1a, calls, beside, rec["p1c"]), encoding="utf-8", newline="\n")
    log("calls on s1eval: " + ", ".join(f"{n} {v}" for n, v in head.items()) + f"; P1c reads {rec['p1c']['construction']}")
    return 0


def report_md(parts, p1a, calls, beside, p1c):
    L = ["# P1b2: the rank-fit gate against the SOTA pools on the full corpora (s1eval)", "",
         "ALL at N rows; recall in brackets. RC, RCS, ORC and the classes from outputs/p1b2/rc_<dataset>.json; A3 and the "
         "SOTA orders from P1a's records. ORC reads the golds (an upper bound for the gate).", ""]
    show = (25, 100, 500, 1000, 2000, 7000)
    for name in SIX:
        c = f"{name}__s1eval"
        ent, pa, cl = parts[name]["carves"][c], p1a[name]["carves"][c], calls[c]
        L += [f"## {name}: {cl['call']}", "", "| order | " + " | ".join(f"N = {n:,}" for n in show) + " |",
              "| --- |" + " ---: |" * len(show)]
        rows = [(o, ent["orders"][o]["curve"]) for o in ("RC", "RCS", "ORC", "MX")]
        rows += [(o, pa["orders"][o]["curve"]) for o in ("A3",) + P.SOTA]
        for o, cv in rows:
            L.append(f"| {o} | " + " | ".join(f"{cv[str(n)]['ALL']:.3f} ({cv[str(n)]['recall']:.3f})" for n in show)
                     + " |")
        L += ["", "- RC's class shares: " + ", ".join(f"{k} {v}" for k, v in ent["rc_share"].items() if v) + ".",
              "- The oracle's: " + ", ".join(f"{k} {v}" for k, v in ent["orc_share"].items() if v) + "."]
        s = cl["at_I_q"]
        L.append(f"- At each question's size of today's pools ({s['set']['size']:,.0f} rows mean): the set "
                 f"{s['set']['ALL']:.3f}, RC {s['RC']['ALL']:.3f}, MX {s['MX']['ALL']:.3f}, A3 {s['A3']['ALL']:.3f}, "
                 f"best SOTA order {s['best_sota']} {s['sota']['ALL']:.3f}.")
        if cl["below"]:
            L.append(f"- Below at {len(cl['below'])} points (N " + ", ".join(sorted({str(b['N']) for b in cl['below']},
                     key=int)) + f"); the largest gap {cl['largest_gap']:+.4f}.")
        if cl["above"]:
            L.append(f"- Above at {len(cl['above'])} points; the largest lead {cl['largest_lead']:+.4f}.")
        L.append(f"- Beside: RCS {beside[c]['RCS']['call']}, ORC {beside[c]['ORC']['call']}.")
        L.append("")
    L.append(f"P1c reads {p1c['construction']} ({p1c['rule']}" + (f"; s1sel means {p1c['s1sel_mean_at_I_q']}"
             if "s1sel_mean_at_I_q" in p1c else "") + ").")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    w_mx = np.asarray([0.1, 0.1, 0.1, 0.3, 0.2, 0.2])
    for c in CLASSES:
        w = blend(c, w_mx)
        assert w.shape == (len(SOURCES),) and abs(w.sum() - 1) < 1e-12 and (w >= 0).all(), c
    assert blend("MX", w_mx)[-1] == 0 and blend("A1", w_mx)[-1] == 1
    # the area: every gold first is 0; no gold anywhere is the whole grid
    g = np.asarray([3, 7])
    assert area(np.asarray([3, 7, 1]), g) == 0.0
    assert area(np.asarray([1, 2]), g) == 2.0 * len(NS) and area(np.asarray([1]), np.empty(0, np.int64)) == 2.0 * len(NS)
    o = np.concatenate([[3], np.arange(100, 130), [7]])            # the second gold at position 31
    a = area(o, g)
    k = sum(n <= 31 for n in NS)                                    # grid points where the second gold is outside
    assert abs(a - (k * 1.0 + k * 0.5)) < 1e-12, (a, k)
    # one-source classes reproduce their source; MX is P1b's mixture
    src = (B.uniform([4, 2]), B.EMPTY, B.by_mass(np.asarray([7, 2, 5]), np.asarray([0.2, 0.5, 0.3])),
           B.by_mass(np.asarray([4, 9, 2]), np.asarray([0.5, 0.3, 0.2])), B.EMPTY,
           B.by_mass(np.asarray([5, 8]), np.asarray([0.6, 0.4])), B.by_mass(np.asarray([8, 4]), np.asarray([0.7, 0.3])))
    orders = class_orders(src, w_mx)
    for c in ("S", "R", "A1"):
        assert orders[c].tolist() == src[SOURCES.index(c)][0].tolist(), c
    assert orders["L"].size == 0 and orders["MX"].tolist() == B.mix_order(src[:6], w_mx).tolist()
    # squad's A1 is P1a's RRF order there, its weights summing to one
    d, s_ = np.asarray([5, 3, 9, 4]), np.asarray([3, 7, 5])
    n_, m_ = rrf_source(d, s_, 60)
    assert n_.tolist() == E.rrf_order(d, s_, np.empty(0, np.int64), 60, NMAX).tolist() and abs(m_.sum() - 1) < 1e-12
    assert rrf_source(np.empty(0, np.int64), np.empty(0, np.int64), 60)[0].size == 0
    # the hard read skips empty classes; the soft read is a blend over the available ones
    pi = np.zeros(len(CLASSES))
    pi[CLASSES.index("L")] = 0.9
    pi[CLASSES.index("A1")] = 0.1
    assert CLASSES[hard_choice(pi, orders)] == "A1"
    sw = soft_weights(pi, orders, w_mx)
    assert abs(sw.sum() - 1) < 1e-12 and sw[SOURCES.index("A1")] == 1.0
    # EM over the classes picks the class with the least area, question by question
    rng = np.random.default_rng(0)
    Q = 400
    X = rng.normal(size=(Q, 2))
    good = np.where(X[:, 0] > 0, CLASSES.index("A1"), CLASSES.index("L"))
    A = np.full((Q, len(CLASSES)), 6.0)
    A[np.arange(Q), good] = 1.0
    A[:40] = 3.0                                                    # tied questions carry nothing
    V, trace = B.em(X, evidence(A), np.arange(Q + 1, dtype=np.int64), np.full(Q, 1.0 / Q))
    assert all(b >= a_ - 1e-9 for a_, b in zip(trace, trace[1:])), trace[:5]
    pi = B.softmax(np.hstack([X, np.ones((Q, 1))]) @ V.T)
    assert (pi[40:].argmax(axis=1) == good[40:]).mean() > 0.95
    print("selftest: ok (blends, the area on the N grid, one-source classes and MX, squad's A1 as P1a's RRF order, hard and soft reads over available "
          "classes, EM monotone and picks the least-area class)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("gather", "fit", "apply", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int)
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
        return (gather_stage if a.stage == "gather" else apply_stage)(a.dataset, a.threads, a.limit)
    if a.stage == "fit":
        return fit_stage()
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
