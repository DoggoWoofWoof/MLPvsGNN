"""BRG3 (docs/BRG3_DEPTH_AND_BEAM_SCREEN.md): do more hops, with or without a narrow frontier (a beam), reach the golds that lie beyond the cell's hops?

Read-only screen on the subsets of BRG1/BRG2: no pool, cell or rule of the filed work is changed and nothing is adopted. The walk is the budget-following walk of PL5 (zpl5g.expand5) re-run hop by hop as in
zbrg1.py / zbrg2.py with the cell's own candidate order and window (512), and with the number of hops and the frontier of the later hops replaced by one of six variants. zbrg1.py, zbrg2.py and the pool code
(zpl2g.py to zpl6g.py, zdrp1.py, zdrp2.py, zstk3.py) are imported and edited nowhere.

    python -E -P -B outputs/mp_unified/zbrg3.py selftest
    python -E -P -B outputs/mp_unified/zbrg3.py depth --ds D | --kb K [--pos 0,1,2] [--smoke M] [--host]
        # chunk positions of the seeded order (BRG1's three by default): outputs/brg3/<name>/dep_c<chunk>.npz and gate_depth.json
    python -E -P -B outputs/mp_unified/zbrg3.py read --ds D | --kb K
        # aggregates the chunk files: outputs/brg3/read_<name>.json

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
import zbrg2 as B2  # noqa: E402
import zdrp1 as D1  # noqa: E402
import zdrp2 as D2  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl5g as Z5  # noqa: E402

DOC = "docs/BRG3_DEPTH_AND_BEAM_SCREEN.md"
OUT = ROOT / "outputs" / "brg3"
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc
BEAM = 100
# (name, hops, beam); hops 0 = the cell's own hop count; beam 0 = the whole fresh set is the next frontier (the walk of the cell)
VARIANTS = (("control", 0, 0), ("h3", 3, 0), ("h4", 4, 0), ("h3b100", 3, BEAM), ("h4b100", 4, BEAM), ("h5b100", 5, BEAM))
VNAME = tuple(v[0] for v in VARIANTS)
QCOLS3 = ("q", "ng") + tuple(f"{k}{v}" for v in range(len(VARIANTS)) for k in ("p", "s", "u"))
QI3 = {c: i for i, c in enumerate(QCOLS3)}
BOOT_N, BOOT_SEED = 2000, 20261128


def walk_beam(seeds, fams1, fams2, setting, mods, scorer, hops, beam):
    """zbrg1.trace_walk with the hop count replaced and, from hop 2 on, the frontier cut to the first `beam` fresh nodes in the cell's order (beam 0: the whole fresh set, as expand5).
    The budget of a hop is the cell's rule over the new hop count. Returns (expansion, hop records {h, budget, fa})."""
    key = next(m for m in Z5.KEYS5 if m in mods)
    rr = "rr" in mods
    limit, win = int(setting["visited_cap"]), int(setting["window"])
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
        fa = B2.order_alt("control", cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))
        fresh = fa[:budget]
        recs.append({"h": h, "budget": int(budget), "fa": fa})
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh[:beam] if beam else fresh
    return (np.concatenate(out) if out else np.empty(0, dtype=np.int64)), recs


def question_row(q, golds, head, exps, recs_all):
    golds = np.asarray(golds, dtype=np.int64)
    r = {c: 0 for c in QCOLS3}
    r["q"], r["ng"] = q, int(golds.size)
    for v, (exp, recs) in enumerate(zip(exps, recs_all)):
        pool = np.union1d(head, exp)
        r[f"p{v}"], r[f"s{v}"] = int(np.isin(golds, pool).sum()), int(pool.size)
        cand = np.concatenate([head] + [rec["fa"] for rec in recs]) if recs else head
        r[f"u{v}"] = int(np.isin(golds, cand).sum())
    return np.asarray([r[c] for c in QCOLS3], dtype=np.int64)


def depth_range(op, d, plan, cell, lo, hi, limit=None):
    ln, b, regime, setting, mods, n1, n2 = B.cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    walk = Z5.parse5(cell)[3]
    h0 = int(setting["hops"])
    Q, bad = [], []
    for q, st, seeds, qv in B.states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        try:
            exps, recs_all = [], []
            for _nm, hh, bm in VARIANTS:
                exp, recs = walk_beam(seeds, fams1, fams2, setting, mods, scorer, hh or h0, bm)
                exps.append(exp)
                recs_all.append(recs)
        finally:
            scorer.release()
        eng = st.W[walk].exp
        if not np.array_equal(exps[0], eng):
            bad.append([q, "expansion", int(exps[0].size), int(eng.size)])
        P = np.zeros(1, dtype=np.int16)
        S = np.zeros(1, dtype=np.int32)
        Z3.grade3(plan, st, d.golds[q], P, S)
        row = question_row(q, d.golds[q], st.L[ln].L[:b].astype(np.int64), exps, recs_all)
        if int(row[QI3["p0"]]) != int(P[0]) or int(row[QI3["s0"]]) != int(S[0]):
            bad.append([q, "grade", [int(row[QI3["p0"]]), int(row[QI3["s0"]])], [int(P[0]), int(S[0])]])
        Q.append(row)
    return (np.asarray(Q) if Q else np.zeros((0, len(QCOLS3)), np.int64)), bad


def cmd_depth(a):
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
    outd = OUT / name
    if a.smoke:
        lo, hi = crange(order[0], src.nq)
        t0 = time.time()
        Q, bad = depth_range(op, d, plan, cell, lo, hi, limit=int(a.smoke))
        print(f"smoke {name}: {int(a.smoke)} questions, G-H1 mismatches {len(bad)} {bad[:3]}, {time.time() - t0:.1f} s")
        return 0 if not bad else 3
    poss = [int(x) for x in str(a.pos).split(",")] if a.pos else list(range(min(B.SLICE, nch)))
    outd.mkdir(parents=True, exist_ok=True)
    gate_p = outd / "gate_depth.json"
    for pos in poss:
        c = order[pos]
        path = outd / f"dep_c{c:04d}.npz"
        if path.is_file():
            continue
        lo, hi = crange(c, src.nq)
        t0 = time.time()
        Q, bad = depth_range(op, d, plan, cell, lo, hi)
        gates = rjson(gate_p) if gate_p.is_file() else {"stage": "BRG3 gates", "declared_in": DOC, "set": name, "cell": cell, "variants": list(VNAME), "chunks": {}}
        gates["chunks"][str(c)] = {"questions": hi - lo, "G-H1": {"mismatches": len(bad), "first": bad[:5]}, "seconds": round(time.time() - t0, 1)}
        gates["utc"], gates["verdict"] = utc(), ("PASS" if all(v["G-H1"]["mismatches"] == 0 for v in gates["chunks"].values()) else "FAIL")
        wjson(gate_p, gates)
        if bad:
            stop(f"{name}: chunk {c}: G-H1 failed ({len(bad)} mismatches, first {bad[:3]})")
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as f:
            np.savez(f, quest=Q, qcols=np.asarray(QCOLS3), variants=np.asarray(VNAME), cell=np.asarray(cell), lo=lo, hi=hi)
        os.replace(tmp, path)
        log(f"{name}: chunk {c} (questions {lo}-{hi}): G-H1 PASS, {Q.shape[0]} questions, {time.time() - t0:.0f} s ({P2.peak_gb()} GB)")
    return 0


def aggregate(Q):
    ng = Q[:, QI3["ng"]]
    nq = Q.shape[0]
    full = [(Q[:, QI3[f"p{v}"]] == ng).astype(np.float64) for v in range(len(VARIANTS))]
    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.integers(0, nq, size=(BOOT_N, nq))
    out = {"questions": int(nq), "golds": int(ng.sum()), "variants": {}}
    for v, nm in enumerate(VNAME):
        rec = {"ALL": float(full[v].mean()), "golds_in_pool": float(Q[:, QI3[f"p{v}"]].sum() / ng.sum()), "mean_nodes": float(Q[:, QI3[f"s{v}"]].mean()),
               "ceiling_ALL": float((Q[:, QI3[f"u{v}"]] == ng).mean()), "ceiling_golds": float(Q[:, QI3[f"u{v}"]].sum() / ng.sum())}
        if v:
            diff = full[v] - full[0]
            bs = diff[boot].mean(axis=1)
            ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
            rec.update({"minus_control": float(diff.mean()), "ci95": ci, "call": D2.gain_call(ci), "nodes_over_control": float(Q[:, QI3[f"s{v}"]].mean() / Q[:, QI3["s0"]].mean())})
        for lab, m in (("1", ng == 1), ("2", ng == 2), ("3", ng == 3), ("4+", ng >= 4)):
            if m.any():
                rec.setdefault("by_gold_count", {})[lab] = {"questions": int(m.sum()), "ALL": float(full[v][m].mean())}
        out["variants"][nm] = rec
    return out


def cmd_read(a):
    name = a.kb or a.ds
    folder = OUT / name
    gate = rjson(folder / "gate_depth.json") if (folder / "gate_depth.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        stop(f"{name}: no PASS gate_depth.json: no number is read")
    files = sorted(folder.glob("dep_c*.npz"))
    if not files:
        stop(f"{name}: no chunk file")
    if len(files) != len(gate["chunks"]):
        stop(f"{name}: {len(files)} chunk files but {len(gate['chunks'])} gate entries: a gate entry is missing")
    Qs = []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["qcols"]] != list(QCOLS3) or [str(x) for x in z["variants"]] != list(VNAME) or str(z["cell"]) != B.CELL[name]:
                stop(f"{p.name}: not this module's columns, variants or cell")
            Qs.append(z["quest"])
        finally:
            z.close()
    rec = {"stage": "BRG3 read", "declared_in": DOC, "set": name, "cell": B.CELL[name], "utc": utc(), "chunks": [p.name for p in files], **aggregate(np.concatenate(Qs))}
    wjson(OUT / f"read_{name}.json", rec)
    log(f"{name}: read written ({rec['questions']} questions): " + "; ".join(f"{k} {100 * x['ALL']:.1f}" + (f" ({100 * x['minus_control']:+.2f} {x['call']})" if "call" in x else "") for k, x in rec["variants"].items()))
    return 0


def cmd_selftest(_a):
    B.Z6.prepare(False)
    rng = np.random.default_rng(13)
    n, dim = 400, 8
    V = rng.normal(size=(n, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    fams = B._toy(rng, n, deg=10)
    ok = 0
    for trial in range(20):
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        seeds = rng.choice(n, size=5, replace=False)
        for stg, mods in (("h2_bf_v1000", ("cs", "vr")), ("h3_bf_v1000", ("cs", "fu")), ("h2_bf_v1000", ("cs", "vr", "rr")), ("h3_bf_v2000", ("cs", "vr"))):
            st = dict(Z5.SET5[stg])
            st["visited_cap"] = 60 if trial % 2 else int(st["visited_cap"])
            sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
            e1 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc)
            sc.release()
            sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
            e2, _r = walk_beam(seeds, [fams[0]], fams, st, mods, sc, int(st["hops"]), 0)
            assert np.array_equal(e1, e2), ("beam 0 differs from expand5", trial, stg, mods)
            for hh, bm in ((3, 0), (4, 0), (3, 5), (4, 5), (5, 5)):
                e3, recs = walk_beam(seeds, [fams[0]], fams, st, mods, sc, hh, bm)
                assert np.unique(e3).size == e3.size and not np.isin(e3, seeds).any(), (hh, bm, trial)
                assert e3.size <= int(st["visited_cap"]), (hh, bm, trial)
                assert len(recs) <= hh
            e4, _r = walk_beam(seeds, [fams[0]], fams, st, mods, sc, 3, 1)         # a beam of one node: the third hop's candidates are the rows of one node
            sc.release()
            ok += 1
    print(f"zbrg3 selftest: beam 0 equals expand5 on {ok} walks (four settings), deeper and beamed walks keep distinct fresh nodes inside the cap: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for c in ("depth", "read"):
        p = sub.add_parser(c)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--ds", choices=B.SEVEN)
        g.add_argument("--kb", choices=B.KBS)
        p.add_argument("--host", action="store_true")
        if c == "depth":
            p.add_argument("--pos", default="")
            p.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return cmd_selftest(a)
    if a.cmd == "depth":
        return cmd_depth(a)
    return cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
