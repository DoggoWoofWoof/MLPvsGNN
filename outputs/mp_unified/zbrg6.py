"""BRG6 (docs/BRG6_LEARNED_ORDER_REMAINDER.md): the learned candidate order of BRG5 amendment 1 (`lr6`) read on every chunk of the split outside its fit.

Imports zbrg5.py and zbrg5b.py (neither edited). Two variants, `control` (the cell's order) and `lr6` (logistic regression on 12 walk features, per hop, fitted on the chunks at positions 3 to 8 of the set's chunk order), on the chunks at
positions 0 to 2 (the BRG5 screen chunks, seen) and 9 and up (unseen). Screen only; nothing is adopted.

    python -E -P -B outputs/mp_unified/zbrg6.py selftest
    python -E -P -B outputs/mp_unified/zbrg6.py screen --ds D | --kb K [--smoke M] [--host]
        # outputs/brg6/<name>/: scr_c<chunk>.npz, gate_a6.json
    python -E -P -B outputs/mp_unified/zbrg6.py read --ds D | --kb K
        # outputs/brg6/read_<name>.json

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
import zbrg5 as Z  # noqa: E402
import zbrg5b as A  # noqa: E402
import zdrp1 as D1  # noqa: E402
import zdrp2 as D2  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl5g as Z5  # noqa: E402

DOC = "docs/BRG6_LEARNED_ORDER_REMAINDER.md"
OUT = ROOT / "outputs" / "brg6"
OUT5 = ROOT / "outputs" / "brg5"
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc
VAR = ("control", "lr6")
HOPS = Z.HOPS
NSCREEN, FIT_POS = 3, A.POS6                                 # positions 0-2 are BRG5's screen chunks, 3-8 the fit's
QCOLS = ("q", "ng") + tuple(f"{k}{v}" for v in range(len(VAR)) for k in ("p", "s", "u")) + tuple(f"{k}{v}_{h}" for v in range(len(VAR)) for h in HOPS for k in ("c", "k"))
QI = {c: i for i, c in enumerate(QCOLS)}
IDX = [A.QI[c] for c in QCOLS]                               # the same column names inside amendment 1's wider layout (variants 0 and 1 of its five)
A1_LR6 = A.VAR.index("lr6")


def split_chunks(order, nch):
    """(fit chunks, chunks read in the order they run, screen chunks, unseen chunks) from the set's chunk order."""
    fit = [order[p] for p in FIT_POS if p < nch]
    read = [order[p] for p in range(nch) if p not in FIT_POS]
    return fit, read, read[:NSCREEN], read[NSCREEN:]


def eval_range(op, d, plan, cell, models, lo, hi, limit=None):
    ln, b, regime, setting, mods, n1, n2 = B.cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    walk = Z5.parse5(cell)[3]
    Q, bad = [], []
    for q, st, seeds, qv in B.states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        try:
            exps, recs_all = [], []
            for kind in VAR:
                pick = None if kind == "control" else (lambda h, X, m=models: Z.score_of(m.get(h), X))
                exp, recs = Z.walk_screen(seeds, fams1, fams2, setting, mods, scorer, pick=pick)
                exps.append(exp)
                recs_all.append(recs)
        finally:
            scorer.release()
        if not np.array_equal(exps[0], st.W[walk].exp):
            bad.append([q, "expansion", int(exps[0].size), int(st.W[walk].exp.size)])
        P = np.zeros(1, dtype=np.int16)
        S = np.zeros(1, dtype=np.int32)
        Z3.grade3(plan, st, d.golds[q], P, S)
        full = A.row(q, d.golds[q], st.L[ln].L[:b].astype(np.int64), exps, recs_all)
        r = full[IDX]
        if int(r[QI["p0"]]) != int(P[0]) or int(r[QI["s0"]]) != int(S[0]):
            bad.append([q, "grade", [int(r[QI["p0"]]), int(r[QI["s0"]])], [int(P[0]), int(S[0])]])
        Q.append(r)
    return (np.asarray(Q) if Q else np.zeros((0, len(QCOLS)), np.int64)), bad


def g_r2(name, c, Q, limit=None):
    """G-R2: on a screen chunk the `lr6` row of every question equals amendment 1's `lr6` row."""
    p = OUT5 / name / "a1" / f"scr_c{c:04d}.npz"
    if not p.is_file():
        return [[c, "no amendment 1 chunk file"]]
    z = np.load(p, allow_pickle=False)
    try:
        old, cols = z["quest"], [str(x) for x in z["qcols"]]
    finally:
        z.close()
    oi = {k: i for i, k in enumerate(cols)}
    n = Q.shape[0]
    if old.shape[0] < n or not np.array_equal(old[:n, oi["q"]], Q[:, QI["q"]]):
        return [[c, "questions differ"]]
    bad = []
    for new, old_col in (("p1", f"p{A1_LR6}"), ("s1", f"s{A1_LR6}"), ("p0", "p0"), ("s0", "s0")):
        k = int((old[:n, oi[old_col]] != Q[:, QI[new]]).sum())
        if k:
            bad.append([c, new, k])
    return bad


def cmd_screen(a):
    name, op, src, order, crange, nch = B.load(a)
    cell = B.CELL.get(name)
    if cell is None:
        stop(f"{name}: no declared cell")
    bad0 = B.declared_cell_gate(name, bool(a.kb))
    if bad0:
        stop(f"G-B0 failed: {bad0}")
    d = D1.variant(src, "both")
    for x in D2.arm_fails(src, d, "both"):
        stop(f"G-D1 failed: {x}")
    plan = Z5.Plan5([cell])
    fit, read, screen, unseen = split_chunks(order, nch)
    a1gate = rjson(OUT5 / name / "a1" / "gate_a1.json")
    if fit != a1gate["train6"] or len(fit) != 6 or set(fit) & set(read):
        stop(f"G-R1 failed: the fit chunks {fit} are not amendment 1's {a1gate['train6']} or overlap the read")
    files6 = [OUT5 / name / f"train_c{c:04d}.npz" for c in fit[:3]] + [OUT5 / name / "a1" / f"train_c{c:04d}.npz" for c in fit[3:]]
    for p in files6:
        if not p.is_file():
            stop(f"{name}: training file {p.name} is missing")
    models, rec = A.fit_kind(Z.load_train(files6), "lr")
    want = rjson(OUT5 / name / "a1" / "fit.json")["fits"]["lr6"]
    for h in HOPS:
        if rec[str(h)]["positives"] != want[str(h)]["positives"] or rec[str(h)]["rows"] != want[str(h)]["rows"] or rec[str(h)]["fitted"] != want[str(h)]["fitted"]:
            stop(f"G-R1 failed: hop {h} fit {rec[str(h)]} differs from amendment 1's {want[str(h)]}")
    t0 = time.time()
    if a.smoke:
        m = int(a.smoke)
        lo, hi = crange(read[0], src.nq)
        Q, bad = eval_range(op, d, plan, cell, models, lo, hi, limit=m)
        b2 = g_r2(name, read[0], Q)
        print(f"smoke {name}: fit chunks {len(fit)}, read chunks {len(read)} (screen {len(screen)}, unseen {len(unseen)}); {m} questions, G-R0 mismatches {len(bad)} {bad[:3]}, G-R2 {b2}; "
              f"ALL control {float((Q[:, QI['p0']] == Q[:, QI['ng']]).mean()):.3f} lr6 {float((Q[:, QI['p1']] == Q[:, QI['ng']]).mean()):.3f}; {time.time() - t0:.1f} s")
        return 0 if not (bad or b2) else 3
    outd = OUT / name
    outd.mkdir(parents=True, exist_ok=True)
    gate_p = outd / "gate_a6.json"
    gates = rjson(gate_p) if gate_p.is_file() else {"stage": "BRG6 gates", "declared_in": DOC, "set": name, "cell": cell, "fit_chunks": fit, "read_chunks": read, "screen_chunks": screen,
                                                    "questions_expected": int(src.nq - sum(crange(c, src.nq)[1] - crange(c, src.nq)[0] for c in fit)), "fit": rec, "chunks": {}}
    gates["G-R1"] = {"fit_equals_amendment_1": True, "disjoint": True}
    for c in read:
        path = outd / f"scr_c{c:04d}.npz"
        if path.is_file():
            continue
        lo, hi = crange(c, src.nq)
        t1 = time.time()
        Q, bad = eval_range(op, d, plan, cell, models, lo, hi)
        b2 = g_r2(name, c, Q) if c in screen else None
        gates["chunks"][str(c)] = {"questions": int(Q.shape[0]), "seen": bool(c in screen), "G-R0": {"mismatches": len(bad), "first": bad[:5]},
                                   "G-R2": None if b2 is None else {"mismatches": len(b2), "first": b2[:5]}, "seconds": round(time.time() - t1, 1)}
        done = len(gates["chunks"]) == len(read)
        ok = all(v["G-R0"]["mismatches"] == 0 and (v["G-R2"] is None or v["G-R2"]["mismatches"] == 0) for v in gates["chunks"].values())
        gates["questions_read"] = int(sum(v["questions"] for v in gates["chunks"].values()))
        gates["G-R3"] = {"complete": bool(done and gates["questions_read"] == gates["questions_expected"])}
        gates["verdict"] = "PASS" if (done and ok and gates["G-R3"]["complete"]) else ("FAIL" if not ok else "RUNNING")
        gates["utc"] = utc()
        wjson(gate_p, gates)
        if bad or b2:
            stop(f"{name}: chunk {c}: G-R0 {bad[:3]} or G-R2 {b2[:3] if b2 else b2} failed")
        Z.atomic_savez(path, quest=Q, qcols=np.asarray(QCOLS), variants=np.asarray(VAR), cell=np.asarray(cell), lo=lo, hi=hi, seen=np.asarray(c in screen))
        log(f"{name}: chunk {c} ({'screen' if c in screen else 'unseen'}; questions {lo}-{hi}): G-R0{' and G-R2' if b2 is not None else ''} PASS, {Q.shape[0]} questions, {time.time() - t1:.0f} s ({P2.peak_gb()} GB); {len(gates['chunks'])}/{len(read)} chunks")
    return 0


def aggregate(Q):
    ng = Q[:, QI["ng"]]
    nq = Q.shape[0]
    if nq == 0:
        return {"questions": 0}
    full = [(Q[:, QI[f"p{v}"]] == ng).astype(np.float64) for v in range(len(VAR))]
    rng = np.random.default_rng(Z.BOOT_SEED)
    boot = rng.integers(0, nq, size=(Z.BOOT_N, nq))
    out = {"questions": int(nq), "golds": int(ng.sum()), "variants": {}}
    for v, nm in enumerate(VAR):
        rec = {"ALL": float(full[v].mean()), "golds_in_pool": float(Q[:, QI[f"p{v}"]].sum() / ng.sum()), "mean_nodes": float(Q[:, QI[f"s{v}"]].mean()),
               "ceiling_ALL": float((Q[:, QI[f"u{v}"]] == ng).mean())}
        if v:
            diff = full[v] - full[0]
            bs = diff[boot].mean(axis=1)
            ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
            rec["vs_control"] = {"diff": float(diff.mean()), "ci95": ci, "call": D2.gain_call(ci)}
            rec["nodes_over_control"] = float(Q[:, QI[f"s{v}"]].mean() / Q[:, QI["s0"]].mean())
            rec["reached"] = bool(rec["vs_control"]["call"] == "ABOVE" and rec["nodes_over_control"] <= 1.02)
        rec["hops"] = {str(h): {"candidate_golds": int(Q[:, QI[f"c{v}_{h}"]].sum()), "kept_golds": int(Q[:, QI[f"k{v}_{h}"]].sum())} for h in HOPS}
        out["variants"][nm] = rec
    return out


def cmd_read(a):
    name = a.kb or a.ds
    folder = OUT / name
    gate = rjson(folder / "gate_a6.json") if (folder / "gate_a6.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        stop(f"{name}: no PASS gate_a6.json: no number is read")
    files = sorted(folder.glob("scr_c*.npz"))
    if len(files) != len(gate["chunks"]) or len(files) != len(gate["read_chunks"]):
        stop(f"{name}: {len(files)} chunk files, {len(gate['chunks'])} gate entries, {len(gate['read_chunks'])} chunks to read")
    Qs, seen = [], []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["qcols"]] != list(QCOLS) or [str(x) for x in z["variants"]] != list(VAR) or str(z["cell"]) != B.CELL[name]:
                stop(f"{p.name}: not this module's columns, variants or cell")
            Qs.append(z["quest"])
            seen.append(np.full(z["quest"].shape[0], bool(z["seen"])))
        finally:
            z.close()
    Q, S = np.concatenate(Qs), np.concatenate(seen)
    rec = {"stage": "BRG6 read", "declared_in": DOC, "set": name, "cell": B.CELL[name], "utc": utc(), "fit_chunks": gate["fit_chunks"], "screen_chunks": gate["screen_chunks"], "read_chunks": gate["read_chunks"],
           "fit": gate["fit"], "unseen": aggregate(Q[~S]), "remainder": aggregate(Q), "screen": aggregate(Q[S])}
    wjson(OUT / f"read_{name}.json", rec)
    for part in ("unseen", "remainder", "screen"):
        x = rec[part]
        log(f"{name}: {part} ({x['questions']} questions): " + "; ".join(f"{k} {100 * v['ALL']:.1f} @{v['mean_nodes']:.0f}" + (f" ({100 * v['vs_control']['diff']:+.2f} {v['vs_control']['call']})" if "vs_control" in v else "") for k, v in x["variants"].items()))
    return 0


def cmd_selftest(_a):
    order = [10, 18, 21, 16, 2, 0, 17, 11, 5, 1, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 19, 20]
    fit, read, screen, unseen = split_chunks(order, 22)
    assert fit == [16, 2, 0, 17, 11, 5] and screen == [10, 18, 21] and unseen == [1, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 19, 20] and read == screen + unseen
    assert not set(fit) & set(read) and sorted(fit + read) == list(range(22)) and len(read) == 16
    fit, read, screen, unseen = split_chunks(order[:12], 12)             # a short set keeps the fit chunks that exist
    assert len(fit) == 6 and len(read) == 6 and not set(fit) & set(read)
    assert [A.QCOLS[i] for i in IDX] == list(QCOLS) and A1_LR6 == 2
    nq = 600
    rs = np.random.default_rng(3)
    base = rs.random(nq) < 0.80
    better = base | (rs.random(nq) < 0.5)                                # `better` contains every complete question of `base`
    for lr, ctl, call in ((better, base, "ABOVE"), (base, better, "BELOW"), (base, base, "LEVEL")):
        Q = np.zeros((nq, len(QCOLS)), np.int64)
        Q[:, QI["ng"]] = 2
        Q[:, QI["p0"]] = np.where(ctl, 2, 1)
        Q[:, QI["p1"]] = np.where(lr, 2, 1)
        Q[:, QI["s0"]], Q[:, QI["s1"]] = 2000, 2010
        Q[:, QI["u0"]] = Q[:, QI["u1"]] = 2
        g = aggregate(Q)["variants"]["lr6"]
        assert g["vs_control"]["call"] == call, (call, g["vs_control"])
        assert abs(g["nodes_over_control"] - 1.005) < 1e-9 and g["reached"] == (call == "ABOVE")
        assert abs(g["vs_control"]["diff"] - (lr.mean() - ctl.mean())) < 1e-12
    S = np.zeros(nq, bool)
    S[:100] = True
    assert aggregate(Q[S])["questions"] == 100 and aggregate(Q[~S])["questions"] == 500 and aggregate(Q[:0])["questions"] == 0
    print("zbrg6 selftest: the chunk split (fit 6, read the rest, screen first), the column map into amendment 1's layout, the paired calls on three synthetic cases (ABOVE, BELOW, LEVEL) and the subset masks: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for c in ("screen", "read"):
        p = sub.add_parser(c)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--ds", choices=sorted(set(B.CELL) & set(B.SEVEN)))
        g.add_argument("--kb", choices=sorted(set(B.CELL) & set(B.KBS)))
        p.add_argument("--host", action="store_true")
        if c == "screen":
            p.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return cmd_selftest(a)
    return cmd_screen(a) if a.cmd == "screen" else cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
