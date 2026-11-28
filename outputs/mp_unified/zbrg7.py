"""BRG7 (docs/BRG7_LEARNED_ORDER_CAP_WALK.md): the learned candidate order of BRG5 on MetaQA's per-node-cap walk, at caps 25, 50 and 100.

Imports zbrg4.py (the cap walk and its cell), zbrg5.py (the 12 features, the collector, the estimators), zbrg5b.py (the per-hop fit) and zbrg6.py (the chunk split); none is edited. Six variants: c25, c50, c100 (the cell's own
arrival order at that cap) and l25, l50, l100 (the distinct fresh candidates of hops 2 and 3 reordered by a logistic regression on the 12 features, fitted per cap and hop on the chunks at positions 3 to 8 of the chunk order).
Screen only; nothing is adopted.

    python -E -P -B outputs/mp_unified/zbrg7.py selftest
    python -E -P -B outputs/mp_unified/zbrg7.py screen --ds metaqa [--smoke M] [--host]
        # outputs/brg7/metaqa/: train_c<chunk>.npz (positions 3 to 8), fit.json, scr_c<chunk>.npz (positions 0 to 2 and 9 and up), gate_m.json
    python -E -P -B outputs/mp_unified/zbrg7.py read --ds metaqa
        # outputs/brg7/read_metaqa.json

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
import time  # noqa: E402

import numpy as np  # noqa: E402
import zbrg1 as B  # noqa: E402
import zbrg4 as G4  # noqa: E402
import zbrg5 as Z  # noqa: E402
import zbrg5b as A  # noqa: E402
import zbrg6 as Z6  # noqa: E402
import zdrp1 as D1  # noqa: E402
import zdrp2 as D2  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl5g as Z5  # noqa: E402

Z4 = Z5.Z4
DOC = "docs/BRG7_LEARNED_ORDER_CAP_WALK.md"
OUT = ROOT / "outputs" / "brg7"
OUT4 = ROOT / "outputs" / "brg4"
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc
CAPS = (25, 50, 100)
WALKS = tuple((kind, cap) for cap in CAPS for kind in (None, "lr"))          # c25, l25, c50, l50, c100, l100
VN = tuple(("c" if kind is None else "l") + str(cap) for kind, cap in WALKS)
HOPS = Z.HOPS
QCOLS = ("q", "ng") + tuple(f"{k}{v}" for v in range(len(VN)) for k in ("p", "s", "u")) + tuple(f"{k}{v}_{h}" for v in range(len(VN)) for h in HOPS for k in ("c", "k"))
QI = {c: i for i, c in enumerate(QCOLS)}
SCORED = (2, 3)                                                               # hop 1 (every neighbour of the seeds) has no budget


def cap_walk_screen(seeds, fams1, fams2, setting, mods, scorer, pick=None, collect=None):
    """zbrg4.trace_cap_walk (kind control) with the fresh candidates of hops 2 and 3 reordered by the score of `pick(h, X)` (None keeps the arrival order of that hop) and `collect(h, fa, X)` called with each hop's
    candidates in the arrival order and their 12 features. Returns (expansion, hop records {h, budget, fa})."""
    first = Z5.m3b_pools._first_occurrence_unique
    key = "vr" if "vr" in mods else ("sp" if "sp" in mods else ("rk" if "rk" in mods else ""))
    rr, eq = "rr" in mods, "eq" in mods
    cap, cap2 = int(setting["per_seed_cap"]), int(setting["per_frontier_cap"]) if "per_frontier_cap" in setting else 0
    seeds = np.asarray(seeds, dtype=np.int64)
    hop1 = first(Z4.capped_parts(seeds, fams1, cap, scorer, False)[0])
    hops = int(setting["hops"])
    recs = [{"h": 1, "budget": int(hop1.size), "fa": hop1}]
    if hops == 1:
        return hop1, recs
    limit = int(setting["visited_cap"])
    visited = np.union1d(seeds, hop1)
    out = [hop1]
    frontier = hop1
    for h in range(2, hops + 1):
        rem = limit - int(visited.size)
        if rem <= 0 or frontier.size == 0:
            break
        budget = -(-rem // (hops - h + 1)) if eq else rem
        cols, grp, fam, rank = Z4.capped_parts(frontier, fams2, cap2, scorer, False)
        if cols.size == 0:
            break
        k = np.isin(cols, visited, invert=True)
        fa = G4.order_cap("control", cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))
        if fa.size and (pick is not None or collect is not None):
            X = Z.hop_features(fa, cols[k], grp[k], fam[k], rank[k], frontier, len(fams2), scorer, budget).astype(np.float32)
            if collect is not None:
                collect(h, fa, X)
            if pick is not None:
                sc = pick(h, X)
                if sc is not None:
                    fa = fa[np.lexsort((np.arange(fa.size), -np.asarray(sc, dtype=np.float64)))]
        recs.append({"h": h, "budget": int(budget), "fa": fa})
        fresh = fa[:budget]
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return np.concatenate(out), recs


def row(q, golds, head, exps, recs_all):
    golds = np.asarray(golds, dtype=np.int64)
    r = {c: 0 for c in QCOLS}
    r["q"], r["ng"] = q, int(golds.size)
    for v, (exp, recs) in enumerate(zip(exps, recs_all)):
        pool = np.union1d(head, exp)
        r[f"p{v}"], r[f"s{v}"] = int(np.isin(golds, pool).sum()), int(pool.size)
        cand = np.concatenate([head] + [rec["fa"] for rec in recs]) if recs else head
        r[f"u{v}"] = int(np.isin(golds, cand).sum())
        for rec in recs[:3]:
            h = rec["h"]
            r[f"c{v}_{h}"] = int(np.isin(golds, rec["fa"]).sum())
            r[f"k{v}_{h}"] = int(np.isin(golds, rec["fa"][:rec["budget"]]).sum())
    return np.asarray([r[c] for c in QCOLS], dtype=np.int64)


def with_cap(setting, cap):
    sv = dict(setting)
    sv["per_seed_cap"] = sv["per_frontier_cap"] = int(cap)
    return sv


def collect_chunk(op, d, plan, cell, lo, hi, chunk, limit=None):
    ln, b, regime, setting, mods, n1, n2 = G4.cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    cols = {cap: Z.Collector() for cap in CAPS}
    nq = 0
    for q, st, seeds, qv in B.states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        try:
            for cap in CAPS:
                cols[cap].golds = d.golds[q]
                cap_walk_screen(seeds, fams1, fams2, with_cap(setting, cap), mods, scorer, pick=None, collect=cols[cap])
        finally:
            scorer.release()
        nq += 1
    arrs, bad, nonfin = {}, [], 0
    for cap in CAPS:
        col = cols[cap]
        nonfin += col.nonfinite
        for h in HOPS:
            X = np.concatenate(col.X[h]) if col.X[h] else np.zeros((0, Z.NF), np.float32)
            y = np.concatenate(col.y[h]) if col.y[h] else np.zeros(0, np.uint8)
            if int(y.sum()) != col.recount[h]:
                bad.append([cap, h, int(y.sum()), col.recount[h]])
            X, y = Z.subsample(X, y, Z.NEG_CAP // len(Z.TRAIN_POS), Z.SEED + 100 * chunk + 10 * cap + h)
            arrs[f"X{cap}_{h}"], arrs[f"y{cap}_{h}"], arrs[f"n{cap}_{h}"] = X, y, np.int64(col.recount[h])
    return arrs, bad, nonfin, nq


def load_train(files, cap):
    train = {}
    for h in SCORED:
        Xs, ys = [], []
        for p in files:
            z = np.load(p, allow_pickle=False)
            try:
                Xs.append(z[f"X{cap}_{h}"])
                ys.append(z[f"y{cap}_{h}"])
            finally:
                z.close()
        train[h] = (np.concatenate(Xs), np.concatenate(ys))
    return train


def eval_range(op, d, plan, cell, models, lo, hi, limit=None):
    """models: {cap: {h: model or None}}."""
    ln, b, regime, setting, mods, n1, n2 = G4.cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    walk = Z5.parse5(cell)[3]
    Q, bad = [], []
    for q, st, seeds, qv in B.states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        try:
            exps, recs_all = [], []
            for kind, cap in WALKS:
                pick = None if kind is None else (lambda h, X, m=models[cap]: Z.score_of(m.get(h), X))
                exp, recs = cap_walk_screen(seeds, fams1, fams2, with_cap(setting, cap), mods, scorer, pick=pick)
                exps.append(exp)
                recs_all.append(recs)
        finally:
            scorer.release()
        if not np.array_equal(exps[0], st.W[walk].exp):
            bad.append([q, "expansion", int(exps[0].size), int(st.W[walk].exp.size)])
        P = np.zeros(1, dtype=np.int16)
        S = np.zeros(1, dtype=np.int32)
        Z3.grade3(plan, st, d.golds[q], P, S)
        r = row(q, d.golds[q], st.L[ln].L[:b].astype(np.int64), exps, recs_all)
        if int(r[QI["p0"]]) != int(P[0]) or int(r[QI["s0"]]) != int(S[0]):
            bad.append([q, "grade", [int(r[QI["p0"]]), int(r[QI["s0"]])], [int(P[0]), int(S[0])]])
        Q.append(r)
    return (np.asarray(Q) if Q else np.zeros((0, len(QCOLS)), np.int64)), bad


def g_m2(name, c, Q):
    """G-M2: c25, c50 and c100 reproduce BRG4's cap25_control, cap50_control and cap100_control rows of every question of a screen chunk."""
    p = OUT4 / name / f"ord_c{c:04d}.npz"
    if not p.is_file():
        return [[c, "no BRG4 order-sweep chunk file"]]
    z = np.load(p, allow_pickle=False)
    try:
        old, cols, var = z["quest"], [str(x) for x in z["qcols"]], [str(x) for x in z["variants"]]
    finally:
        z.close()
    oi = {k: i for i, k in enumerate(cols)}
    n = Q.shape[0]
    if old.shape[0] < n or not np.array_equal(old[:n, oi["q"]], Q[:, QI["q"]]):
        return [[c, "questions differ"]]
    bad = []
    for v, cap in ((0, 25), (2, 50), (4, 100)):
        ov = var.index(f"cap{cap}_control")
        for new, old_col in ((f"p{v}", f"p{ov}"), (f"s{v}", f"s{ov}")):
            k = int((old[:n, oi[old_col]] != Q[:, QI[new]]).sum())
            if k:
                bad.append([c, new, k])
    return bad


def cmd_screen(a):
    name, op, src, order, crange, nch = B.load(a)
    cell = G4.CELL.get(name)
    if cell is None:
        stop(f"{name}: no declared cell")
    bad0 = G4.declared_cell_gate(name)
    if bad0:
        stop(f"G-C0 failed: {bad0}")
    d = D1.variant(src, "both")
    for x in D2.arm_fails(src, d, "both"):
        stop(f"G-D1 failed: {x}")
    plan = Z5.Plan5([cell])
    fit, read, screen, unseen = Z6.split_chunks(order, nch)
    if len(fit) != 6 or set(fit) & set(read) or screen != [order[p] for p in range(3)]:
        stop(f"G-M1 failed: fit {fit}, read {read[:4]}")
    outd = OUT / name
    outd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.smoke:
        m = int(a.smoke)
        lo, hi = crange(fit[0], src.nq)
        arrs, bad, nonfin, nq = collect_chunk(op, d, plan, cell, lo, hi, fit[0], limit=m)
        models = {}
        for cap in CAPS:
            tr = {h: (arrs[f"X{cap}_{h}"], arrs[f"y{cap}_{h}"]) for h in SCORED}
            models[cap] = A.fit_kind(tr, "lr")[0]
        lo, hi = crange(read[0], src.nq)
        Q, bad2 = eval_range(op, d, plan, cell, models, lo, hi, limit=m)
        b3 = g_m2(name, read[0], Q)
        print(f"smoke {name}: fit chunks {len(fit)}, read chunks {len(read)}; train {nq} questions, G-M1 recount mismatches {len(bad)}, nonfinite {nonfin}; eval {m} questions, G-M0 mismatches {len(bad2)} {bad2[:3]}, G-M2 {b3}; "
              f"ALL {dict(zip(VN, [round(float((Q[:, QI[f'p{v}']] == Q[:, QI['ng']]).mean()), 3) for v in range(len(VN))]))}; {time.time() - t0:.1f} s")
        return 0 if not (bad or bad2 or b3 or nonfin) else 3
    gate_p = outd / "gate_m.json"
    gates = rjson(gate_p) if gate_p.is_file() else {"stage": "BRG7 gates", "declared_in": DOC, "set": name, "cell": cell, "fit_chunks": fit, "screen_chunks": screen, "read_chunks": read, "variants": list(VN),
                                                    "questions_expected": int(src.nq - sum(crange(c, src.nq)[1] - crange(c, src.nq)[0] for c in fit)), "train": {}, "chunks": {}}
    files = []
    for c in fit:
        path = outd / f"train_c{c:04d}.npz"
        if not path.is_file():
            lo, hi = crange(c, src.nq)
            t1 = time.time()
            arrs, bad, nonfin, nq = collect_chunk(op, d, plan, cell, lo, hi, c)
            gates["train"][str(c)] = {"questions": nq, "G-M1_recount_mismatches": len(bad), "G-M1_nonfinite": nonfin, "seconds": round(time.time() - t1, 1),
                                      "positives": {f"{cap}_{h}": int(arrs[f"y{cap}_{h}"].sum()) for cap in CAPS for h in SCORED}}
            wjson(gate_p, gates)
            if bad or nonfin:
                stop(f"{name}: training chunk {c}: recount {bad[:3]} or nonfinite {nonfin}")
            Z.atomic_savez(path, **arrs)
            log(f"{name}: training chunk {c}: {nq} questions, positives {gates['train'][str(c)]['positives']}, {time.time() - t1:.0f} s ({P2.peak_gb()} GB)")
        files.append(path)
    models, rec = {}, {}
    for cap in CAPS:
        models[cap], rec[str(cap)] = A.fit_kind(load_train(files, cap), "lr")
    wjson(outd / "fit.json", {"stage": "BRG7 fit", "declared_in": DOC, "set": name, "utc": utc(), "fit_chunks": fit, "fits": rec})
    log(f"{name}: fitted on {len(fit)} chunks: " + "; ".join(f"cap {c} positives {[v[str(h)]['positives'] for h in SCORED]}" for c, v in rec.items()))
    for c in read:
        path = outd / f"scr_c{c:04d}.npz"
        if path.is_file():
            continue
        lo, hi = crange(c, src.nq)
        t1 = time.time()
        Q, bad = eval_range(op, d, plan, cell, models, lo, hi)
        b3 = g_m2(name, c, Q) if c in screen else None
        gates["chunks"][str(c)] = {"questions": int(Q.shape[0]), "seen": bool(c in screen), "G-M0": {"mismatches": len(bad), "first": bad[:5]},
                                   "G-M2": None if b3 is None else {"mismatches": len(b3), "first": b3[:5]}, "seconds": round(time.time() - t1, 1)}
        done = len(gates["chunks"]) == len(read)
        ok = all(v["G-M0"]["mismatches"] == 0 and (v["G-M2"] is None or v["G-M2"]["mismatches"] == 0) for v in gates["chunks"].values()) and all(v["G-M1_recount_mismatches"] == 0 and v["G-M1_nonfinite"] == 0 for v in gates["train"].values())
        gates["questions_read"] = int(sum(v["questions"] for v in gates["chunks"].values()))
        gates["G-M3"] = {"complete": bool(done and gates["questions_read"] == gates["questions_expected"])}
        gates["verdict"] = "PASS" if (done and ok and gates["G-M3"]["complete"]) else ("FAIL" if not ok else "RUNNING")
        gates["utc"] = utc()
        wjson(gate_p, gates)
        if bad or b3:
            stop(f"{name}: chunk {c}: G-M0 {bad[:3]} or G-M2 {b3[:3] if b3 else b3} failed")
        Z.atomic_savez(path, quest=Q, qcols=np.asarray(QCOLS), variants=np.asarray(VN), cell=np.asarray(cell), lo=lo, hi=hi, seen=np.asarray(c in screen))
        log(f"{name}: chunk {c} ({'screen' if c in screen else 'unseen'}; questions {lo}-{hi}): G-M0{' and G-M2' if b3 is not None else ''} PASS, {Q.shape[0]} questions, {time.time() - t1:.0f} s ({P2.peak_gb()} GB); {len(gates['chunks'])}/{len(read)} chunks")
    return 0


def aggregate(Q):
    ng = Q[:, QI["ng"]]
    nq = Q.shape[0]
    if nq == 0:
        return {"questions": 0}
    full = [(Q[:, QI[f"p{v}"]] == ng).astype(np.float64) for v in range(len(VN))]
    rng = np.random.default_rng(Z.BOOT_SEED)
    boot = rng.integers(0, nq, size=(Z.BOOT_N, nq))

    def paired(i, j):
        diff = full[i] - full[j]
        bs = diff[boot].mean(axis=1)
        ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        return {"diff": float(diff.mean()), "ci95": ci, "call": D2.gain_call(ci)}

    out = {"questions": int(nq), "golds": int(ng.sum()), "variants": {}}
    for v, nm in enumerate(VN):
        rec = {"ALL": float(full[v].mean()), "golds_in_pool": float(Q[:, QI[f"p{v}"]].sum() / ng.sum()), "mean_nodes": float(Q[:, QI[f"s{v}"]].mean()),
               "ceiling_ALL": float((Q[:, QI[f"u{v}"]] == ng).mean()), "nodes_over_c25": float(Q[:, QI[f"s{v}"]].mean() / Q[:, QI["s0"]].mean())}
        if v:
            rec["vs_c25"] = paired(v, 0)
            rec["reached"] = bool(rec["vs_c25"]["call"] == "ABOVE" and rec["nodes_over_c25"] <= 1.05)
        if WALKS[v][0] == "lr":
            rec["vs_same_cap_control"] = paired(v, v - 1)
            rec["nodes_over_same_cap_control"] = float(Q[:, QI[f"s{v}"]].mean() / Q[:, QI[f"s{v - 1}"]].mean())
        rec["hops"] = {str(h): {"candidate_golds": int(Q[:, QI[f"c{v}_{h}"]].sum()), "kept_golds": int(Q[:, QI[f"k{v}_{h}"]].sum())} for h in HOPS}
        out["variants"][nm] = rec
    return out


def cmd_read(a):
    name = a.ds
    folder = OUT / name
    gate = rjson(folder / "gate_m.json") if (folder / "gate_m.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        stop(f"{name}: no PASS gate_m.json: no number is read")
    files = sorted(folder.glob("scr_c*.npz"))
    if len(files) != len(gate["chunks"]) or len(files) != len(gate["read_chunks"]):
        stop(f"{name}: {len(files)} chunk files, {len(gate['chunks'])} gate entries, {len(gate['read_chunks'])} chunks to read")
    Qs, seen = [], []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["qcols"]] != list(QCOLS) or [str(x) for x in z["variants"]] != list(VN) or str(z["cell"]) != G4.CELL[name]:
                stop(f"{p.name}: not this module's columns, variants or cell")
            Qs.append(z["quest"])
            seen.append(np.full(z["quest"].shape[0], bool(z["seen"])))
        finally:
            z.close()
    Q, S = np.concatenate(Qs), np.concatenate(seen)
    rec = {"stage": "BRG7 read", "declared_in": DOC, "set": name, "cell": G4.CELL[name], "utc": utc(), "fit_chunks": gate["fit_chunks"], "screen_chunks": gate["screen_chunks"], "read_chunks": gate["read_chunks"],
           "fit": rjson(folder / "fit.json")["fits"], "unseen": aggregate(Q[~S]), "remainder": aggregate(Q), "screen": aggregate(Q[S])}
    wjson(OUT / f"read_{name}.json", rec)
    for part in ("unseen", "remainder", "screen"):
        x = rec[part]
        log(f"{name}: {part} ({x['questions']} questions): " + "; ".join(f"{k} {100 * v['ALL']:.1f} @{v['mean_nodes']:.0f}" + (f" ({100 * v['vs_c25']['diff']:+.2f} {v['vs_c25']['call']})" if "vs_c25" in v else "") for k, v in x["variants"].items()))
    return 0


def cmd_selftest(_a):
    B.Z6.prepare(False)
    rng = np.random.default_rng(17)
    n, dim = 500, 8
    V = rng.normal(size=(n, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    fams = B._toy(rng, n, deg=30)
    ok = nfeat = 0
    for trial in range(24):
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        seeds = rng.choice(n, size=4, replace=False)
        for cap in (3, 8, 25):
            for vcap in (50, 2000):
                st = dict(Z5.SET5["h3_c25_v2000"])
                st["per_seed_cap"] = st["per_frontier_cap"] = cap
                st["visited_cap"] = vcap if trial % 2 == 0 else 40
                for mods in ((), ("rr",)):
                    sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
                    e1, r1 = G4.trace_cap_walk(seeds, [fams[0]], fams, st, mods, sc, "control")
                    sc.release()
                    sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
                    e2, r2 = cap_walk_screen(seeds, [fams[0]], fams, st, mods, sc)
                    assert np.array_equal(e1, e2) and [x["h"] for x in r1] == [x["h"] for x in r2], ("control differs from zbrg4's walk", trial, cap, vcap, mods)
                    sc.release()
                    seen = []
                    sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
                    e3, _ = cap_walk_screen(seeds, [fams[0]], fams, st, mods, sc, collect=lambda h, fa, X: seen.append((h, fa, X)))
                    assert np.array_equal(e1, e3), "collecting changes the walk"
                    for h, fa, X in seen:
                        assert h in SCORED and X.shape == (fa.size, Z.NF) and np.isfinite(X).all()
                        nfeat += int(fa.size)
                    e4, _ = cap_walk_screen(seeds, [fams[0]], fams, st, mods, sc, pick=lambda h, X: -np.arange(X.shape[0], dtype=np.float64))
                    assert np.array_equal(e1, e4), "a score equal to the arrival place changes the walk"
                    e5, r5 = cap_walk_screen(seeds, [fams[0]], fams, st, mods, sc, pick=lambda h, X: np.arange(X.shape[0], dtype=np.float64))
                    sc.release()
                    h1 = r5[0]["fa"]
                    room = max(0, int(st["visited_cap"]) - int(np.union1d(seeds, h1).size))                 # hop 1 (every neighbour of the seeds) is not budgeted; later hops fill what is left
                    assert np.unique(e5).size == e5.size and e5.size - h1.size <= room, "picked walk repeats a node or breaks the visited cap"
                    ok += 1
    assert VN == ("c25", "l25", "c50", "l50", "c100", "l100") and len(QCOLS) == 2 + 3 * 6 + 6 * 6
    nq = 600
    rs = np.random.default_rng(3)
    base = rs.random(nq) < 0.80
    better = base | (rs.random(nq) < 0.5)
    Q = np.zeros((nq, len(QCOLS)), np.int64)
    Q[:, QI["ng"]] = 2
    for v in range(6):
        Q[:, QI[f"s{v}"]] = 2000 + 10 * v
        Q[:, QI[f"u{v}"]] = 2
        Q[:, QI[f"p{v}"]] = np.where(better if v == 3 else base, 2, 1)
    g = aggregate(Q)["variants"]
    assert g["l50"]["vs_c25"]["call"] == "ABOVE" and g["l50"]["reached"] and g["l50"]["vs_same_cap_control"]["call"] == "ABOVE"
    assert g["c50"]["vs_c25"]["call"] == "LEVEL" and "vs_same_cap_control" not in g["c50"] and not g["l25"]["reached"]
    assert abs(g["l50"]["nodes_over_c25"] - 2030 / 2000) < 1e-9 and abs(g["l50"]["nodes_over_same_cap_control"] - 2030 / 2020) < 1e-9
    assert aggregate(Q[:0])["questions"] == 0
    print(f"zbrg7 selftest: the cap walk with the arrival order equals zbrg4's trace_cap_walk on {ok} walks (caps 3, 8, 25; visited caps 40, 50, 2000; with and without rr), collecting and a score equal to the arrival place leave the walk "
          f"unchanged, a reversed score keeps the visited cap and repeats no node ({nfeat} finite feature rows), the paired calls and the pool ratios on a synthetic case: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for c in ("screen", "read"):
        p = sub.add_parser(c)
        p.add_argument("--ds", choices=sorted(G4.CELL), required=True)
        p.add_argument("--host", action="store_true")
        if c == "screen":
            p.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    a.kb = None
    if a.cmd == "selftest":
        return cmd_selftest(a)
    return cmd_screen(a) if a.cmd == "screen" else cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
