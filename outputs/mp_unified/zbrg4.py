"""BRG4 (docs/BRG4_CAP_WALK_TRACE.md): BRG1's trace for a per-node-cap walk (MetaQA's cell), whose hop is not budget-following.

Read-only diagnostics; no pool, cell or rule changes and nothing is adopted. zbrg1.py (its gold rows, aggregate, links of the walk's inputs), zpl4g.py's walk and the pool code are imported and edited nowhere.
`trace` re-runs the bounded walk of zpl4g.expand4 (a per-node cap of the first N neighbours of every frontier node, the visited cap keeping the first fresh nodes of a hop, no cosine inside the cap)
hop by hop with every hop's candidate list kept, asserts that the re-run equals the engine's expansion and grade on every question (G-C1), and gives each gold a reason as BRG1 does
(BUDGET_CUT = the gold was a candidate past the visited cap; WINDOW_CUT = the gold sits beyond the per-node cap in a frontier node's row).

    python -E -P -B outputs/mp_unified/zbrg4.py selftest
    python -E -P -B outputs/mp_unified/zbrg4.py trace --ds D [--slice N] [--smoke M] [--host]
        # N chunks of the seeded order (3 by default): outputs/brg4/<name>/trace_c<chunk>.npz and gate_trace.json; --smoke M grades M questions of the first chunk, prints the gate only
    python -E -P -B outputs/mp_unified/zbrg4.py read --ds D
        # aggregates the chunk files: outputs/brg4/read_<name>.json

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
import zdrp1 as D1  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl5g as Z5  # noqa: E402

Z4 = Z5.Z4
DOC = "docs/BRG4_CAP_WALK_TRACE.md"
OUT = ROOT / "outputs" / "brg4"
# the declared cell of each traced set: the dense-only (`both`) arm's best cell within 2,500 nodes in DRP1's whole-split read (section 17); G-C0 re-reads it
CELL = {"metaqa": "rrf1000_b800+STRUCT:h3_c25_v2000"}
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc


def trace_cap_walk(seeds, fams1, fams2, setting, mods, scorer):
    """zpl4g.expand4 for a walk without cosine inside the cap (no `cs`), written out so that every hop's candidate list is kept. Returns (expansion, hop records {h, frontier, budget, fa})."""
    first = Z5.m3b_pools._first_occurrence_unique
    key = "vr" if "vr" in mods else ("sp" if "sp" in mods else ("rk" if "rk" in mods else ""))
    rr, eq = "rr" in mods, "eq" in mods
    cap, cap2 = int(setting["per_seed_cap"]), int(setting["per_frontier_cap"]) if "per_frontier_cap" in setting else 0
    seeds = np.asarray(seeds, dtype=np.int64)
    hop1 = first(Z4.capped_parts(seeds, fams1, cap, scorer, False)[0])
    hops = int(setting["hops"])
    recs = [{"h": 1, "frontier": seeds, "budget": int(hop1.size), "fa": hop1}]
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
        fa = Z4.order_fresh(cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))
        recs.append({"h": h, "frontier": frontier, "budget": int(budget), "fa": fa})
        fresh = fa[:budget]
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return np.concatenate(out), recs


def cell_parts(name):
    kind, ln, b, wk = Z5.parse5(name)
    if kind != "G":
        stop(f"{name}: BRG4 traces union cells (G) only")
    regime, stg, mods = wk
    st = Z5.SET5[stg]
    if st.get("bf"):
        stop(f"{name}: BRG4 traces per-node-cap walks only (budget-following walks are BRG1's)")
    if "cs" in mods:
        stop(f"{name}: a cosine-ordered cap (cs) has no row position to read a cap cut from")
    if int(st["hops"]) > 1 and int(st["per_frontier_cap"]) != int(st["per_seed_cap"]):
        stop(f"{name}: the per-seed and per-frontier caps differ")
    h1, h2 = Z5.REG3[regime]
    return ln, int(b), regime, st, mods, list(h1), list(h2)


def declared_cell_gate(name):
    """G-C0: the declared cell is the dense-only arm's best within 2,500 nodes in the filed whole-split read."""
    got = rjson(ROOT / "outputs" / "drp1e" / f"read_{name}.json")["best_within"]["2500"]["both"]["name"]
    return [] if got == CELL[name] else [f"{name}: the read's best cell is {got}, the declared one {CELL[name]}"]


def trace_range(op, d, plan, cell, lo, hi, limit=None, flat_arr=None):
    ln, b, regime, setting, mods, n1, n2 = cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    win, hops = int(setting["per_seed_cap"]), int(setting["hops"])
    walk = Z5.parse5(cell)[3]
    allfam = list(dict.fromkeys(n1 + n2))
    qs = list(range(lo, hi if limit is None else min(hi, lo + limit)))
    golds_all = np.unique(np.concatenate([d.golds[q] for q in qs])) if qs else np.empty(0, np.int64)
    ie = {f: B.in_edges(d.fams[f], golds_all, d.n) for f in allfam}
    buf = np.full(d.n, 255, dtype=np.uint8)
    G, Q, bad = [], [], []
    for q, st, seeds, qv in B.states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        try:
            exp, recs = trace_cap_walk(seeds, fams1, fams2, setting, mods, scorer)
        finally:
            scorer.release()
        eng = st.W[walk].exp
        if not np.array_equal(exp, eng):
            bad.append([q, "expansion", int(exp.size), int(eng.size)])
        P = np.zeros(1, dtype=np.int16)
        S = np.zeros(1, dtype=np.int32)
        Z3.grade3(plan, st, d.golds[q], P, S)
        reached, cut = B.bfs_dist(seeds, fams1, fams2, hops, buf)
        try:
            g_rows, q_row = B.question_rows(q, d.golds[q], st.L[ln], b, exp, seeds, recs, ie, n1, n2, win, hops, buf, cut, flat_arr[q])
        finally:
            buf[reached] = 255
        if int(q_row[B.QI["present"]]) != int(P[0]) or int(q_row[B.QI["size"]]) != int(S[0]):
            bad.append([q, "grade", [int(q_row[B.QI["present"]]), int(q_row[B.QI["size"]])], [int(P[0]), int(S[0])]])
        G.append(g_rows)
        Q.append(q_row)
    return (np.concatenate(G) if G else np.zeros((0, len(B.COLS)), np.int64)), (np.asarray(Q) if Q else np.zeros((0, len(B.QCOLS)), np.int64)), bad


def cmd_trace(a):
    name, op, src, order, crange, nch = B.load(a)
    cell = CELL.get(name)
    if cell is None:
        stop(f"{name}: no declared cell")
    bad0 = declared_cell_gate(name)
    if bad0:
        stop(f"G-C0 failed: {bad0}")
    d = D1.variant(src, "both")
    for x in B.D2.arm_fails(src, d, "both"):
        stop(f"G-D1 failed: {x}")
    plan = Z5.Plan5([cell])
    flat_arr = np.asarray(d.dense)
    outd = OUT / name
    if a.smoke:
        lo, hi = crange(order[0], src.nq)
        t0 = time.time()
        _G, _Q, bad = trace_range(op, d, plan, cell, lo, hi, limit=int(a.smoke), flat_arr=flat_arr)
        print(f"smoke {name}: {int(a.smoke)} questions, G-C1 mismatches {len(bad)} {bad[:3]}, {time.time() - t0:.1f} s")
        return 0 if not bad else 3
    top = min(int(a.slice or B.SLICE), nch)
    outd.mkdir(parents=True, exist_ok=True)
    gate_p = outd / "gate_trace.json"
    gates = rjson(gate_p) if gate_p.is_file() else {"stage": "BRG4 gates", "declared_in": DOC, "set": name, "cell": cell, "chunks": {}}
    for pos in range(top):
        c = order[pos]
        path = outd / f"trace_c{c:04d}.npz"
        if path.is_file():
            continue
        lo, hi = crange(c, src.nq)
        t0 = time.time()
        G, Q, bad = trace_range(op, d, plan, cell, lo, hi, flat_arr=flat_arr)
        gates["chunks"][str(c)] = {"questions": hi - lo, "G-C1": {"mismatches": len(bad), "first": bad[:5]}, "seconds": round(time.time() - t0, 1)}
        gates["utc"], gates["verdict"] = utc(), ("PASS" if all(v["G-C1"]["mismatches"] == 0 for v in gates["chunks"].values()) else "FAIL")
        wjson(gate_p, gates)
        if bad:
            stop(f"{name}: chunk {c}: G-C1 failed ({len(bad)} mismatches, first {bad[:3]})")
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as f:
            np.savez(f, gold=G, quest=Q, cols=np.asarray(B.COLS), qcols=np.asarray(B.QCOLS), cell=np.asarray(cell), lo=lo, hi=hi)
        os.replace(tmp, path)
        log(f"{name}: chunk {c} (questions {lo}-{hi}): G-C1 PASS, {G.shape[0]} golds, {time.time() - t0:.0f} s ({P2.peak_gb()} GB)")
    return 0


def cmd_read(a):
    name = a.ds
    folder = OUT / name
    gate = rjson(folder / "gate_trace.json") if (folder / "gate_trace.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        stop(f"{name}: no PASS gate_trace.json: no number is read")
    files = sorted(folder.glob("trace_c*.npz"))
    if not files:
        stop(f"{name}: no chunk file")
    if len(files) != len(gate["chunks"]):
        stop(f"{name}: {len(files)} chunk files but {len(gate['chunks'])} gate entries")
    Gs, Qs = [], []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["cols"]] != list(B.COLS) or [str(x) for x in z["qcols"]] != list(B.QCOLS) or str(z["cell"]) != CELL[name]:
                stop(f"{p.name}: not this module's columns or cell")
            Gs.append(z["gold"])
            Qs.append(z["quest"])
        finally:
            z.close()
    G, Q = np.concatenate(Gs), np.concatenate(Qs)
    st = cell_parts(CELL[name])[3]
    rec = {"stage": "BRG4 read", "declared_in": DOC, "set": name, "cell": CELL[name], "utc": utc(), "chunks": [p.name for p in files], **B.aggregate(G, Q, int(st["per_seed_cap"]))}
    wjson(OUT / f"read_{name}.json", rec)
    log(f"{name}: read written ({rec['questions']} questions, {rec['golds']} golds, ALL {rec['ALL']['share']:.3f} at {rec['mean_nodes']:.0f} nodes)")
    return 0


def cmd_selftest(_a):
    B.Z6.prepare(False)
    rng = np.random.default_rng(17)
    n, dim = 400, 8
    V = rng.normal(size=(n, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    fams = B._toy(rng, n, deg=12)
    ok = 0
    for trial in range(20):
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        seeds = rng.choice(n, size=5, replace=False)
        for hops, cap, lim, mods in ((2, 5, 60, ()), (3, 5, 60, ()), (3, 4, 40, ("vr",)), (3, 6, 80, ("eq",)), (3, 5, 50, ("vr", "rr")), (1, 5, 60, ())):
            st = {"name": "t", "hops": hops, "per_seed_cap": cap, "per_frontier_cap": cap, "visited_cap": lim}
            sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
            e1 = Z4.expand4(seeds, [fams[0]], fams, st, mods, sc)
            e2, recs = trace_cap_walk(seeds, [fams[0]], fams, st, mods, sc)
            sc.release()
            assert np.array_equal(e1, e2), ("trace differs from expand4", trial, hops, cap, mods)
            assert len(recs) <= hops and recs[0]["h"] == 1
            assert all(np.unique(r["fa"]).size == r["fa"].size for r in recs), "a candidate list repeats a node"
            for r in recs[1:]:
                assert not np.isin(r["fa"], seeds).any()
            ok += 1
    print(f"zbrg4 selftest: the traced cap walk equals zpl4g.expand4 on {ok} walks (six settings, with vr, rr and eq), candidate lists are distinct and skip the seeds: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for c in ("trace", "read"):
        p = sub.add_parser(c)
        p.add_argument("--ds", choices=sorted(CELL), required=True)
        p.add_argument("--host", action="store_true")
        if c == "trace":
            p.add_argument("--slice", type=int, default=0)
            p.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    a.kb = None
    if a.cmd == "selftest":
        return cmd_selftest(a)
    return cmd_trace(a) if a.cmd == "trace" else cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
