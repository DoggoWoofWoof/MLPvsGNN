"""BRG8 (docs/BRG8_ROW_LENGTH_CAP.md): a per-row cap that depends on the length of the row, on MetaQA's cap walk.

Imports zbrg4.py (the walk and its cell) and zbrg7.py (the row layout); neither is edited. Every frontier node's row in a family is kept whole when it has at most `L` entries and cut at `H` entries when it is longer;
`base` (25, 25) is the cell itself. Nothing is fitted. Screen only; nothing is adopted.

    python -E -P -B outputs/mp_unified/zbrg8.py selftest
    python -E -P -B outputs/mp_unified/zbrg8.py screen --ds metaqa [--smoke M] [--host]
        # outputs/brg8/metaqa/: scr_c<chunk>.npz for every chunk, gate_n.json
    python -E -P -B outputs/mp_unified/zbrg8.py read --ds metaqa
        # outputs/brg8/read_metaqa.json

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
import zbrg7 as Z7  # noqa: E402
import zdrp1 as D1  # noqa: E402
import zdrp2 as D2  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl5g as Z5  # noqa: E402

Z4 = Z5.Z4
DOC = "docs/BRG8_ROW_LENGTH_CAP.md"
OUT = ROOT / "outputs" / "brg8"
OUT4 = ROOT / "outputs" / "brg4"
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc
VARS = (("base", 25, 25), ("L50", 50, 25), ("L100", 100, 25), ("L200", 200, 25), ("L100h10", 100, 10), ("L200h10", 200, 10))
VN = tuple(v[0] for v in VARS)
QCOLS, QI, row = Z7.QCOLS, Z7.QI, Z7.row                      # six variants: the same column layout as BRG7's six
HOPS = Z.HOPS
NSCREEN = 3


def len_capped_parts(nodes, families, L, H):
    """zpl4g.capped_parts (no cosine) with the per-row rule: a row of at most L entries is kept whole, a longer one is cut at H. Same layout: (cols, parent index, family index, rank), node-major, then family, then row order."""
    e64, e8 = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)
    if nodes.size == 0:
        return e64, e64, e8, e64
    cols, grps, fams_, ranks = [], [], [], []
    for f_i, fam in enumerate(families):
        starts = fam.indptr[nodes]
        rowlen = fam.indptr[nodes + 1] - starts
        lens = np.where(rowlen <= L, rowlen, np.minimum(rowlen, H))
        total = int(lens.sum())
        if total == 0:
            continue
        off = np.cumsum(lens) - lens
        within = np.arange(total, dtype=np.int64) - np.repeat(off, lens)
        cols.append(fam.col[np.repeat(starts, lens) + within].astype(np.int64, copy=False))
        grps.append(np.repeat(np.arange(nodes.size, dtype=np.int64), lens))
        fams_.append(np.full(total, f_i, dtype=np.int8))
        ranks.append(within)
    if not cols:
        return e64, e64, e8, e64
    if len(cols) == 1:
        return cols[0], grps[0], fams_[0], ranks[0]
    rank, fam, grp = np.concatenate(ranks), np.concatenate(fams_), np.concatenate(grps)
    order = np.lexsort((rank, fam, grp))
    return np.concatenate(cols)[order], grp[order], fam[order], rank[order]


def len_cap_walk(seeds, fams1, fams2, setting, mods, scorer, L, H):
    """zbrg4.trace_cap_walk (kind control) with len_capped_parts in place of the per-node cap on every hop. Returns (expansion, hop records {h, budget, fa})."""
    first = Z5.m3b_pools._first_occurrence_unique
    key = "vr" if "vr" in mods else ("sp" if "sp" in mods else ("rk" if "rk" in mods else ""))
    rr, eq = "rr" in mods, "eq" in mods
    seeds = np.asarray(seeds, dtype=np.int64)
    hop1 = first(len_capped_parts(seeds, fams1, L, H)[0])
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
        cols, grp, fam, rank = len_capped_parts(frontier, fams2, L, H)
        if cols.size == 0:
            break
        k = np.isin(cols, visited, invert=True)
        fa = G4.order_cap("control", cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))
        recs.append({"h": h, "budget": int(budget), "fa": fa})
        fresh = fa[:budget]
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return np.concatenate(out), recs


def eval_range(op, d, plan, cell, lo, hi, limit=None):
    ln, b, regime, setting, mods, n1, n2 = G4.cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    walk = Z5.parse5(cell)[3]
    Q, bad = [], []
    for q, st, seeds, qv in B.states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        try:
            exps, recs_all = [], []
            for _nm, L, H in VARS:
                exp, recs = len_cap_walk(seeds, fams1, fams2, setting, mods, scorer, L, H)
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


def g_n1(name, c, Q):
    """G-N1: `base` reproduces BRG4's cap25_control row of every question of a screen chunk."""
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
    ov = var.index("cap25_control")
    bad = []
    for new, old_col in (("p0", f"p{ov}"), ("s0", f"s{ov}")):
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
    t0 = time.time()
    if a.smoke:
        m = int(a.smoke)
        lo, hi = crange(order[0], src.nq)
        Q, bad = eval_range(op, d, plan, cell, lo, hi, limit=m)
        b1 = g_n1(name, order[0], Q)
        print(f"smoke {name}: {m} questions, G-N0 mismatches {len(bad)} {bad[:3]}, G-N1 {b1}; ALL {dict(zip(VN, [round(float((Q[:, QI[f'p{v}']] == Q[:, QI['ng']]).mean()), 3) for v in range(len(VN))]))}; "
              f"nodes {dict(zip(VN, [round(float(Q[:, QI[f's{v}']].mean())) for v in range(len(VN))]))}; {time.time() - t0:.1f} s")
        return 0 if not (bad or b1) else 3
    outd = OUT / name
    outd.mkdir(parents=True, exist_ok=True)
    gate_p = outd / "gate_n.json"
    gates = rjson(gate_p) if gate_p.is_file() else {"stage": "BRG8 gates", "declared_in": DOC, "set": name, "cell": cell, "variants": [list(v) for v in VARS], "order": [int(c) for c in order[:nch]], "questions_expected": int(src.nq), "chunks": {}}
    for pos in range(nch):
        c = order[pos]
        path = outd / f"scr_c{c:04d}.npz"
        if path.is_file():
            continue
        lo, hi = crange(c, src.nq)
        t1 = time.time()
        Q, bad = eval_range(op, d, plan, cell, lo, hi)
        b1 = g_n1(name, c, Q) if pos < NSCREEN else None
        gates["chunks"][str(c)] = {"position": pos, "questions": int(Q.shape[0]), "G-N0": {"mismatches": len(bad), "first": bad[:5]}, "G-N1": None if b1 is None else {"mismatches": len(b1), "first": b1[:5]}, "seconds": round(time.time() - t1, 1)}
        done = len(gates["chunks"]) == nch
        ok = all(v["G-N0"]["mismatches"] == 0 and (v["G-N1"] is None or v["G-N1"]["mismatches"] == 0) for v in gates["chunks"].values())
        gates["questions_read"] = int(sum(v["questions"] for v in gates["chunks"].values()))
        gates["G-N3"] = {"complete": bool(done and gates["questions_read"] == gates["questions_expected"])}
        gates["verdict"] = "PASS" if (done and ok and gates["G-N3"]["complete"]) else ("FAIL" if not ok else "RUNNING")
        gates["utc"] = utc()
        wjson(gate_p, gates)
        if bad or b1:
            stop(f"{name}: chunk {c}: G-N0 {bad[:3]} or G-N1 {b1[:3] if b1 else b1} failed")
        Z.atomic_savez(path, quest=Q, qcols=np.asarray(QCOLS), variants=np.asarray(VN), cell=np.asarray(cell), lo=lo, hi=hi, pos=np.asarray(pos))
        log(f"{name}: chunk {c} (position {pos}{', screen' if pos < NSCREEN else ''}; questions {lo}-{hi}): G-N0{' and G-N1' if b1 is not None else ''} PASS, {Q.shape[0]} questions, {time.time() - t1:.0f} s ({P2.peak_gb()} GB); {len(gates['chunks'])}/{nch} chunks")
    return 0


def aggregate(Q):
    ng = Q[:, QI["ng"]]
    nq = Q.shape[0]
    if nq == 0:
        return {"questions": 0}
    full = [(Q[:, QI[f"p{v}"]] == ng).astype(np.float64) for v in range(len(VN))]
    rng = np.random.default_rng(Z.BOOT_SEED)
    boot = rng.integers(0, nq, size=(Z.BOOT_N, nq))
    out = {"questions": int(nq), "golds": int(ng.sum()), "variants": {}}
    for v, nm in enumerate(VN):
        rec = {"ALL": float(full[v].mean()), "golds_in_pool": float(Q[:, QI[f"p{v}"]].sum() / ng.sum()), "mean_nodes": float(Q[:, QI[f"s{v}"]].mean()),
               "ceiling_ALL": float((Q[:, QI[f"u{v}"]] == ng).mean()), "nodes_over_base": float(Q[:, QI[f"s{v}"]].mean() / Q[:, QI["s0"]].mean())}
        if v:
            diff = full[v] - full[0]
            bs = diff[boot].mean(axis=1)
            ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
            rec["vs_base"] = {"diff": float(diff.mean()), "ci95": ci, "call": D2.gain_call(ci)}
        rec["hops"] = {str(h): {"candidate_golds": int(Q[:, QI[f"c{v}_{h}"]].sum()), "kept_golds": int(Q[:, QI[f"k{v}_{h}"]].sum())} for h in HOPS}
        out["variants"][nm] = rec
    return out


def cmd_read(a):
    name = a.ds
    folder = OUT / name
    gate = rjson(folder / "gate_n.json") if (folder / "gate_n.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        stop(f"{name}: no PASS gate_n.json: no number is read")
    files = sorted(folder.glob("scr_c*.npz"))
    if len(files) != len(gate["chunks"]) or len(files) != len(gate["order"]):
        stop(f"{name}: {len(files)} chunk files, {len(gate['chunks'])} gate entries, {len(gate['order'])} chunks")
    Qs, pos = [], []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["qcols"]] != list(QCOLS) or [str(x) for x in z["variants"]] != list(VN) or str(z["cell"]) != G4.CELL[name]:
                stop(f"{p.name}: not this module's columns, variants or cell")
            Qs.append(z["quest"])
            pos.append(np.full(z["quest"].shape[0], int(z["pos"])))
        finally:
            z.close()
    Q, P = np.concatenate(Qs), np.concatenate(pos)
    unseen = P >= NSCREEN
    rec = {"stage": "BRG8 read", "declared_in": DOC, "set": name, "cell": G4.CELL[name], "utc": utc(), "variants": [list(v) for v in VARS], "whole": aggregate(Q), "unseen": aggregate(Q[unseen]),
           "half_even": aggregate(Q[unseen & (P % 2 == 0)]), "half_odd": aggregate(Q[unseen & (P % 2 == 1)]), "screen": aggregate(Q[~unseen])}
    for nm in VN[1:]:
        calls = [rec[k]["variants"][nm]["vs_base"]["call"] for k in ("unseen", "half_even", "half_odd")]
        rec["whole"]["variants"][nm]["reached"] = bool(all(c == "ABOVE" for c in calls) and rec["unseen"]["variants"][nm]["nodes_over_base"] <= 1.05)
    wjson(OUT / f"read_{name}.json", rec)
    for part in ("unseen", "half_even", "half_odd", "whole", "screen"):
        x = rec[part]
        log(f"{name}: {part} ({x['questions']} questions): " + "; ".join(f"{k} {100 * v['ALL']:.1f} @{v['mean_nodes']:.0f}" + (f" ({100 * v['vs_base']['diff']:+.2f} {v['vs_base']['call']})" if "vs_base" in v else "") for k, v in x["variants"].items()))
    return 0


def cmd_selftest(_a):
    B.Z6.prepare(False)
    rng = np.random.default_rng(23)
    n, dim = 500, 8
    V = rng.normal(size=(n, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    fams = B._toy(rng, n, deg=40)
    maxlen = max(int((f.indptr[1:] - f.indptr[:-1]).max()) for f in fams)
    assert maxlen > 30, maxlen
    ok = 0
    for trial in range(30):
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        sc0 = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
        nodes = rng.choice(n, size=int(rng.integers(1, 60)), replace=False).astype(np.int64)
        for cap in (3, 8, 25):
            a1 = len_capped_parts(nodes, fams, cap, cap)
            a2 = Z4.capped_parts(nodes, fams, cap, sc0, False)
            assert all(np.array_equal(x, y) for x, y in zip(a1, a2)), ("L = H = cap differs from capped_parts", trial, cap)
            a3 = len_capped_parts(nodes, [fams[0]], cap, cap)
            a4 = Z4.capped_parts(nodes, [fams[0]], cap, sc0, False)
            assert all(np.array_equal(x, y) for x, y in zip(a3, a4)), "one family differs"
        whole = len_capped_parts(nodes, fams, maxlen, 3)                       # L at least every row: nothing is cut
        big = Z4.capped_parts(nodes, fams, maxlen, sc0, False)
        assert all(np.array_equal(x, y) for x, y in zip(whole, big)), "L above every row keeps every row whole"
        mid = len_capped_parts(nodes, fams, 20, 5)
        lens = np.bincount(mid[1], minlength=nodes.size)
        for f_i, f in enumerate(fams):                                          # short rows whole, long rows cut at H, per node and family
            rl = (f.indptr[nodes + 1] - f.indptr[nodes])
            got = np.bincount(mid[1][mid[2] == f_i], minlength=nodes.size)
            assert np.array_equal(got, np.where(rl <= 20, rl, np.minimum(rl, 5))), "row rule"
        assert lens.sum() == mid[0].size
        sc0.release()
        seeds = rng.choice(n, size=4, replace=False)
        st = dict(Z5.SET5["h3_c25_v2000"])
        st["per_seed_cap"] = st["per_frontier_cap"] = 8
        st["visited_cap"] = 60 if trial % 2 else 2000
        for mods in ((), ("rr",)):
            sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
            e1, r1 = G4.trace_cap_walk(seeds, [fams[0]], fams, st, mods, sc, "control")
            e2, r2 = Z7.cap_walk_screen(seeds, [fams[0]], fams, st, mods, sc)
            e3, r3 = len_cap_walk(seeds, [fams[0]], fams, st, mods, sc, 8, 8)
            sc.release()
            assert np.array_equal(e1, e3) and np.array_equal(e2, e3) and [x["h"] for x in r1] == [x["h"] for x in r3], "L = H = cap walk differs from zbrg4's and zbrg7's"
            sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
            e4, r4 = len_cap_walk(seeds, [fams[0]], fams, st, mods, sc, 20, 5)
            sc.release()
            assert np.unique(e4).size == e4.size, "a node repeats"
            ok += 1
    assert VN == ("base", "L50", "L100", "L200", "L100h10", "L200h10") and len(VARS) == 6
    nq = 600
    rs = np.random.default_rng(3)
    base = rs.random(nq) < 0.80
    better = base | (rs.random(nq) < 0.5)
    Qs = np.zeros((nq, len(QCOLS)), np.int64)
    Qs[:, QI["ng"]] = 2
    for v in range(6):
        Qs[:, QI[f"s{v}"]] = 2000 + 10 * v
        Qs[:, QI[f"u{v}"]] = 2
        Qs[:, QI[f"p{v}"]] = np.where(better if v == 2 else base, 2, 1)
    g = aggregate(Qs)["variants"]
    assert g["L100"]["vs_base"]["call"] == "ABOVE" and g["L50"]["vs_base"]["call"] == "LEVEL" and abs(g["L100"]["nodes_over_base"] - 1.01) < 1e-9
    assert aggregate(Qs[:0])["questions"] == 0
    print(f"zbrg8 selftest: the row rule with L = H = cap equals zpl4g.capped_parts (three caps, one and several families), a row at most L is kept whole and a longer one cut at H, the walk with L = H = cap equals zbrg4's "
          f"trace_cap_walk and zbrg7's cap_walk_screen on {ok} toy walks, the paired calls on a synthetic case: PASS")
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
