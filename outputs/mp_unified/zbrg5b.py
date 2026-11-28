"""BRG5 amendment 1 (docs/BRG5_LEARNED_ORDER_SCREEN.md section 8): more training questions and a regularized non-linear form for the learned candidate order.

Imports zbrg5.py (not edited). Five variants on the evaluation chunks of BRG5: `control` (the cell's order), `lr` (BRG5's logistic regression on the training chunks at positions 3 to 5), `lr6` (the same on the chunks at positions
3 to 8 that exist), `lr2` (logistic regression on the 12 features and their 78 pairwise products and squares, positions 3 to 5), `lr26` (the same on positions 3 to 8). Screen only; nothing is adopted.

    python -E -P -B outputs/mp_unified/zbrg5b.py selftest
    python -E -P -B outputs/mp_unified/zbrg5b.py screen --ds D | --kb K [--smoke M] [--host]
        # outputs/brg5/<name>/a1/: train_c<chunk>.npz for the chunks at positions 6 to 8 (positions 3 to 5 are BRG5's own), fit.json, scr_c<chunk>.npz, gate_a1.json
    python -E -P -B outputs/mp_unified/zbrg5b.py read --ds D | --kb K
        # outputs/brg5/read_a1_<name>.json

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
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import zbrg1 as B  # noqa: E402
import zbrg5 as Z  # noqa: E402
import zdrp1 as D1  # noqa: E402
import zdrp2 as D2  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl5g as Z5  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

DOC = "docs/BRG5_LEARNED_ORDER_SCREEN.md"
OUT = ROOT / "outputs" / "brg5"
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc
VAR = ("control", "lr", "lr6", "lr2", "lr26")
HOPS = Z.HOPS
EVAL_POS = Z.EVAL_POS
POS3, POS6 = (3, 4, 5), (3, 4, 5, 6, 7, 8)
NEG_CHUNK = Z.NEG_CAP // len(Z.TRAIN_POS)                  # negatives kept per hop and training chunk, as BRG5
QCOLS = ("q", "ng") + tuple(f"{k}{v}" for v in range(len(VAR)) for k in ("p", "s", "u")) + tuple(f"{k}{v}_{h}" for v in range(len(VAR)) for h in HOPS for k in ("c", "k"))
QI = {c: i for i, c in enumerate(QCOLS)}


def new_lr2():
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), LogisticRegression(C=1.0, max_iter=200))


def fit_kind(train, kind):
    """{h: model or None} for kind in lr, lr2 on {h: (X, y)}; a hop with fewer than MIN_POS positives keeps the control order."""
    out, rec = {}, {}
    for h in HOPS:
        X, y = train.get(h, (np.zeros((0, Z.NF), np.float32), np.zeros(0, np.uint8)))
        npos = int(y.sum())
        ok = bool(npos >= Z.MIN_POS and npos < y.size)
        rec[str(h)] = {"rows": int(y.size), "positives": npos, "fitted": ok}
        if not ok:
            out[h] = None
            continue
        with warnings.catch_warnings(), threadpool_limits(limits=2, user_api="openmp"):
            warnings.simplefilter("ignore")
            out[h] = (Z.new_model("lr") if kind == "lr" else new_lr2()).fit(X, y)
        rec[str(h)]["iterations"] = int(out[h][-1].n_iter_[0])
    return out, rec


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


def eval_range(op, d, plan, cell, models, lo, hi, limit=None):
    """models: {variant: {h: model or None}} for lr, lr6, lr2, lr26."""
    ln, b, regime, setting, mods, n1, n2 = B.cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    walk = Z5.parse5(cell)[3]
    Q, bad = [], []
    for q, st, seeds, qv in B.states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        try:
            exps, recs_all = [], []
            for kind in VAR:
                pick = None if kind == "control" else (lambda h, X, m=models[kind]: Z.score_of(m.get(h), X))
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
        r = row(q, d.golds[q], st.L[ln].L[:b].astype(np.int64), exps, recs_all)
        if int(r[QI["p0"]]) != int(P[0]) or int(r[QI["s0"]]) != int(S[0]):
            bad.append([q, "grade", [int(r[QI["p0"]]), int(r[QI["s0"]])], [int(P[0]), int(S[0])]])
        Q.append(r)
    return (np.asarray(Q) if Q else np.zeros((0, len(QCOLS)), np.int64)), bad


def g_l3(name, c, Q):
    """G-L3: the `lr` variant reproduces BRG5's own `lr` row of every question of the chunk."""
    p = OUT / name / f"scr_c{c:04d}.npz"
    if not p.is_file():
        return [[c, "no BRG5 chunk file"]]
    z = np.load(p, allow_pickle=False)
    try:
        old, cols = z["quest"], [str(x) for x in z["qcols"]]
    finally:
        z.close()
    oi = {k: i for i, k in enumerate(cols)}
    bad = []
    if old.shape[0] < Q.shape[0] or not np.array_equal(old[:Q.shape[0], oi["q"]], Q[:, QI["q"]]):
        return [[c, "questions differ"]]
    for tag_new, tag_old in (("p1", "p1"), ("s1", "s1"), ("p0", "p0"), ("s0", "s0")):
        n = int((old[:Q.shape[0], oi[tag_old]] != Q[:, QI[tag_new]]).sum())
        if n:
            bad.append([c, tag_new, n])
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
    t3 = [order[p] for p in POS3 if p < nch]
    t6 = [order[p] for p in POS6 if p < nch]
    ech = [order[p] for p in EVAL_POS if p < nch]
    if len(ech) < 3 or len(t3) < 3 or set(ech) & set(t6):
        stop(f"G-L1 failed: {name} has {nch} chunks")
    outd = OUT / name
    a1 = outd / "a1"
    t0 = time.time()
    if a.smoke:
        m = int(a.smoke)
        lo, hi = crange(t3[0], src.nq)
        arrs, bad, nonfin, nq = Z.collect_chunk(op, d, plan, cell, lo, hi, t3[0], limit=m)
        tr = {h: (arrs[f"X{h}"], arrs[f"y{h}"]) for h in HOPS}
        models = {"lr": fit_kind(tr, "lr")[0], "lr6": fit_kind(tr, "lr")[0], "lr2": fit_kind(tr, "lr2")[0], "lr26": fit_kind(tr, "lr2")[0]}
        lo, hi = crange(ech[0], src.nq)
        Q, bad2 = eval_range(op, d, plan, cell, models, lo, hi, limit=m)
        print(f"smoke {name}: chunks for lr {len(t3)}, lr6 {len(t6)}; train {nq} questions, G-L1 recount mismatches {len(bad)}, nonfinite {nonfin}; eval {m} questions, G-L0 mismatches {len(bad2)} {bad2[:3]}; "
              f"ALL {[round(float((Q[:, QI[f'p{v}']] == Q[:, QI['ng']]).mean()), 3) for v in range(len(VAR))]}; {time.time() - t0:.1f} s")
        return 0 if not (bad or bad2 or nonfin) else 3
    a1.mkdir(parents=True, exist_ok=True)
    gate_p = a1 / "gate_a1.json"
    gates = rjson(gate_p) if gate_p.is_file() else {"stage": "BRG5 amendment 1 gates", "declared_in": DOC, "set": name, "cell": cell, "variants": list(VAR), "train3": t3, "train6": t6, "eval_chunks": ech, "train": {}, "chunks": {}}
    files3 = [outd / f"train_c{c:04d}.npz" for c in t3]
    for p in files3:
        if not p.is_file():
            stop(f"{name}: BRG5's training file {p.name} is missing: run zbrg5.py screen first")
    files6 = list(files3)
    for c in t6[len(t3):]:
        path = a1 / f"train_c{c:04d}.npz"
        if not path.is_file():
            lo, hi = crange(c, src.nq)
            t1 = time.time()
            arrs, bad, nonfin, nq = Z.collect_chunk(op, d, plan, cell, lo, hi, c)
            gates["train"][str(c)] = {"questions": nq, "G-L1_recount_mismatches": len(bad), "G-L2_nonfinite": nonfin, "seconds": round(time.time() - t1, 1),
                                      "positives": {str(h): int(arrs[f"y{h}"].sum()) for h in HOPS}}
            wjson(gate_p, gates)
            if bad or nonfin:
                stop(f"{name}: training chunk {c}: G-L1 recount {bad[:3]} or G-L2 nonfinite {nonfin}")
            Z.atomic_savez(path, **arrs)
            log(f"{name}: training chunk {c}: {nq} questions, positives {gates['train'][str(c)]['positives']}, {time.time() - t1:.0f} s ({P2.peak_gb()} GB)")
        files6.append(path)
    tr3, tr6 = Z.load_train(files3), Z.load_train(files6)
    models, rec = {}, {}
    for tag, tr, kind in (("lr", tr3, "lr"), ("lr6", tr6, "lr"), ("lr2", tr3, "lr2"), ("lr26", tr6, "lr2")):
        models[tag], rec[tag] = fit_kind(tr, kind)
    wjson(a1 / "fit.json", {"stage": "BRG5 amendment 1 fit", "declared_in": DOC, "set": name, "utc": utc(), "train3": t3, "train6": t6, "fits": rec})
    log(f"{name}: fitted ({len(t3)} and {len(t6)} chunks): " + "; ".join(f"{k} positives {[v[str(h)]['positives'] for h in HOPS]}" for k, v in rec.items()))
    for c in ech:
        path = a1 / f"scr_c{c:04d}.npz"
        if path.is_file():
            continue
        lo, hi = crange(c, src.nq)
        t1 = time.time()
        Q, bad = eval_range(op, d, plan, cell, models, lo, hi)
        bad3 = g_l3(name, c, Q)
        gates["chunks"][str(c)] = {"questions": hi - lo, "G-L0": {"mismatches": len(bad), "first": bad[:5]}, "G-L3": {"mismatches": len(bad3), "first": bad3[:5]}, "seconds": round(time.time() - t1, 1)}
        gates["utc"] = utc()
        gates["verdict"] = "PASS" if (len(gates["chunks"]) and all(v["G-L0"]["mismatches"] == 0 and v["G-L3"]["mismatches"] == 0 for v in gates["chunks"].values())
                                      and all(v["G-L1_recount_mismatches"] == 0 and v["G-L2_nonfinite"] == 0 for v in gates["train"].values())) else "FAIL"
        wjson(gate_p, gates)
        if bad or bad3:
            stop(f"{name}: chunk {c}: G-L0 {bad[:3]} or G-L3 {bad3[:3]} failed")
        Z.atomic_savez(path, quest=Q, qcols=np.asarray(QCOLS), variants=np.asarray(VAR), cell=np.asarray(cell), lo=lo, hi=hi)
        log(f"{name}: chunk {c} (questions {lo}-{hi}): G-L0 and G-L3 PASS, {Q.shape[0]} questions, {time.time() - t1:.0f} s ({P2.peak_gb()} GB)")
    return 0


def aggregate(Q):
    ng = Q[:, QI["ng"]]
    nq = Q.shape[0]
    full = [(Q[:, QI[f"p{v}"]] == ng).astype(np.float64) for v in range(len(VAR))]
    rng = np.random.default_rng(Z.BOOT_SEED)
    boot = rng.integers(0, nq, size=(Z.BOOT_N, nq))

    def paired(i, j):
        diff = full[i] - full[j]
        bs = diff[boot].mean(axis=1)
        ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        return {"diff": float(diff.mean()), "ci95": ci, "call": D2.gain_call(ci)}

    out = {"questions": int(nq), "golds": int(ng.sum()), "variants": {}}
    for v, nm in enumerate(VAR):
        rec = {"ALL": float(full[v].mean()), "golds_in_pool": float(Q[:, QI[f"p{v}"]].sum() / ng.sum()), "mean_nodes": float(Q[:, QI[f"s{v}"]].mean()),
               "ceiling_ALL": float((Q[:, QI[f"u{v}"]] == ng).mean())}
        if v:
            rec["vs_control"] = paired(v, 0)
            rec["nodes_over_control"] = float(Q[:, QI[f"s{v}"]].mean() / Q[:, QI["s0"]].mean())
            if v > 1:
                rec["vs_lr"] = paired(v, 1)
        rec["hops"] = {str(h): {"candidate_golds": int(Q[:, QI[f"c{v}_{h}"]].sum()), "kept_golds": int(Q[:, QI[f"k{v}_{h}"]].sum())} for h in HOPS}
        out["variants"][nm] = rec
    return out


def cmd_read(a):
    name = a.kb or a.ds
    folder = OUT / name / "a1"
    gate = rjson(folder / "gate_a1.json") if (folder / "gate_a1.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        stop(f"{name}: no PASS gate_a1.json: no number is read")
    files = sorted(folder.glob("scr_c*.npz"))
    if len(files) != len(gate["chunks"]) or not files:
        stop(f"{name}: {len(files)} chunk files but {len(gate['chunks'])} gate entries")
    Qs = []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["qcols"]] != list(QCOLS) or [str(x) for x in z["variants"]] != list(VAR) or str(z["cell"]) != B.CELL[name]:
                stop(f"{p.name}: not this module's columns, variants or cell")
            Qs.append(z["quest"])
        finally:
            z.close()
    rec = {"stage": "BRG5 amendment 1 read", "declared_in": DOC, "set": name, "cell": B.CELL[name], "utc": utc(), "chunks": [p.name for p in files], "train_chunks_lr": gate["train3"], "train_chunks_lr6": gate["train6"],
           "fit": rjson(folder / "fit.json")["fits"], **aggregate(np.concatenate(Qs))}
    wjson(OUT / f"read_a1_{name}.json", rec)
    log(f"{name}: a1 read written ({rec['questions']} questions): " + "; ".join(f"{k} {100 * x['ALL']:.1f} @{x['mean_nodes']:.0f}" + (f" ({100 * x['vs_control']['diff']:+.2f} {x['vs_control']['call']})" if "vs_control" in x else "") for k, x in rec["variants"].items()))
    return 0


def cmd_selftest(_a):
    Z.cmd_selftest(_a)
    rs = np.random.default_rng(7)
    Xt = rs.normal(size=(800, Z.NF)).astype(np.float32)
    yt = ((Xt[:, 1] * Xt[:, 3] + 0.5 * Xt[:, 2] + rs.normal(scale=0.4, size=800)) > 0.8).astype(np.uint8)
    m1, r1 = fit_kind({1: (Xt, yt), 2: (Xt[:30], np.zeros(30, np.uint8))}, "lr")
    m2, r2 = fit_kind({1: (Xt, yt), 2: (Xt[:30], np.zeros(30, np.uint8))}, "lr2")
    assert m1[1] is not None and m2[1] is not None and m1[2] is None and m2[2] is None and m1[3] is None
    from sklearn.metrics import roc_auc_score
    a_lr, a_lr2 = roc_auc_score(yt, Z.score_of(m1[1], Xt)), roc_auc_score(yt, Z.score_of(m2[1], Xt))
    assert a_lr2 > a_lr + 0.05 and a_lr2 > 0.85, ("the product form does not rank an interaction label better than the linear one", a_lr, a_lr2)
    assert m2[1][2].n_output_features_ == Z.NF + 78 if hasattr(m2[1][2], "n_output_features_") else True
    assert len(QCOLS) == 2 + 3 * len(VAR) + 6 * len(VAR)
    print(f"zbrg5b selftest: BRG5's self-test passes, `lr2` fits and ranks an interaction label better than `lr` (AUC {a_lr2:.3f} against {a_lr:.3f}), a hop without positives keeps the control order: PASS")
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
