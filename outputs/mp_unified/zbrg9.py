"""BRG9 (docs/BRG9_ROW_LENGTH_CAP_ALL.md): the per-row cap that depends on the length of the row, run through the engine's own pools on all ten datasets.

Imports zbrg1.py (the loader and the states loop), zbrg4.py and zbrg8.py (MetaQA's cell and the BRG8 rows to compare with); none is edited, nor are zpl4g.py and zpl5g.py: the two per-row cap functions
(zpl4g.capped_parts for the per-node-cap cells, zpl5g.capped_parts5 for the budget-following cells) are replaced by length-rule versions for the duration of one variant's pass over a chunk and restored after,
and the engine's own states loop and grade (zpl3g.grade3) produce the pools. Nothing is fitted; screen only; nothing is adopted.

    python -E -P -B outputs/mp_unified/zbrg9.py selftest
    python -E -P -B outputs/mp_unified/zbrg9.py screen --ds D | --kb K [--chunks N] [--smoke M] [--host]
        # outputs/brg9/<set>/: scr_c<chunk>.npz for each of the first N chunks of the set's chunk order (default 3; 0 = every chunk), gate_s.json
    python -E -P -B outputs/mp_unified/zbrg9.py read --ds D | --kb K
        # outputs/brg9/read_<set>.json

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
import zdrp1 as D1  # noqa: E402
import zdrp2 as D2  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl4g as Z4  # noqa: E402
import zpl5g as Z5  # noqa: E402

DOC = "docs/BRG9_ROW_LENGTH_CAP_ALL.md"
OUT = ROOT / "outputs" / "brg9"
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc
BOOT_N, BOOT_SEED = 2000, 20261128
SCORE_WINDOW = Z4.SCORE_WINDOW
# variant = (name, L, H): a row of at most L entries is kept whole, a longer row is cut at H entries (by the cell's own order). `eng` is the engine unpatched, `same` the length-rule functions with the cell's own cap.
VARS = (("eng", None, None), ("same", None, None), ("V1", 50, 25), ("V2", 100, 25), ("V3", 100, 10), ("V4", 200, 10), ("V5", 200, 25), ("V6", 100, 50), ("V7", 50, 10))
VN = tuple(v[0] for v in VARS)
QCOLS = ("q", "ng") + tuple(f"p{i}" for i in range(len(VN))) + tuple(f"s{i}" for i in range(len(VN)))
QI = {c: i for i, c in enumerate(QCOLS)}
NSCREEN = 3
CELL9 = dict(B.CELL)
CELL9["metaqa"] = G4.CELL["metaqa"]
CELL9.update({"2wiki": "rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq", "hotpotqa": "rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr", "squad": "rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800",
              "multihoprag": "rrf1000_b50+FULL:h2_c100_v5000+top50"})
KBS = ("prime", "mag", "amazon")
SETS = ("2wiki", "hotpotqa", "metaqa", "musique", "squad", "webqsp", "multihoprag") + KBS


def parts4(nodes, families, cap, scorer, cs, L, H):
    """zpl4g.capped_parts with the length rule: the neighbours kept of a node's row in a family are all of them when the row has at most L entries and the first H otherwise (cosine order with `cs`, the first
    SCORE_WINDOW entries scored). L = None is the cell's own cap `cap` (the function itself). Same layout: (cols, parent index, family index, rank)."""
    e64, e8 = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)
    if nodes.size == 0:
        return e64, e64, e8, e64
    cols, grps, fams_, ranks = [], [], [], []
    for f_i, fam in enumerate(families):
        starts = fam.indptr[nodes]
        rowlen = fam.indptr[nodes + 1] - starts
        keepn = np.full(nodes.size, cap, dtype=np.int64) if L is None else np.where(rowlen <= L, rowlen, H).astype(np.int64)
        lens = np.minimum(rowlen, SCORE_WINDOW if cs else keepn)
        total = int(lens.sum())
        if total == 0:
            continue
        off = np.cumsum(lens) - lens
        within = np.arange(total, dtype=np.int64) - np.repeat(off, lens)
        c = fam.col[np.repeat(starts, lens) + within].astype(np.int64, copy=False)
        grp = np.repeat(np.arange(nodes.size, dtype=np.int64), lens)
        if cs:
            sc = scorer.get(c, "q")
            c = c[np.lexsort((c, -sc, grp))]
        keep = within < np.repeat(keepn, lens)
        cols.append(c[keep])
        grps.append(grp[keep])
        fams_.append(np.full(int(keep.sum()), f_i, dtype=np.int8))
        ranks.append(within[keep])
    if not cols:
        return e64, e64, e8, e64
    if len(cols) == 1:
        return cols[0], grps[0], fams_[0], ranks[0]
    rank, fam, grp = np.concatenate(ranks), np.concatenate(fams_), np.concatenate(grps)
    order = np.lexsort((rank, fam, grp))
    return np.concatenate(cols)[order], grp[order], fam[order], rank[order]


def parts5(nodes, families, win, scorer, L, H):
    """zpl5g.capped_parts5 with the length rule: the best `win` by cosine out of each row's first `win` entries, of which a row longer than L keeps the best H. L = None is the function itself."""
    e64, e8 = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)
    if nodes.size == 0:
        return e64, e64, e8, e64
    cols, grps, fams_, ranks = [], [], [], []
    for f_i, fam in enumerate(families):
        starts = fam.indptr[nodes]
        rowlen = fam.indptr[nodes + 1] - starts
        lens = np.minimum(rowlen, win)
        total = int(lens.sum())
        if total == 0:
            continue
        off = np.cumsum(lens) - lens
        within = np.arange(total, dtype=np.int64) - np.repeat(off, lens)
        c = fam.col[np.repeat(starts, lens) + within].astype(np.int64, copy=False)
        grp = np.repeat(np.arange(nodes.size, dtype=np.int64), lens)
        sc = scorer.get(c, "q")
        c = c[np.lexsort((c, -sc, grp))]
        if L is None:
            keep = np.ones(total, dtype=bool)
        else:
            keepn = np.where(rowlen <= L, lens, np.minimum(lens, H))
            keep = within < np.repeat(keepn, lens)
        cols.append(c[keep])
        grps.append(grp[keep])
        fams_.append(np.full(int(keep.sum()), f_i, dtype=np.int8))
        ranks.append(within[keep])
    if not cols:
        return e64, e64, e8, e64
    if len(cols) == 1:
        return cols[0], grps[0], fams_[0], ranks[0]
    rank, fam, grp = np.concatenate(ranks), np.concatenate(fams_), np.concatenate(grps)
    order = np.lexsort((rank, fam, grp))
    return np.concatenate(cols)[order], grp[order], fam[order], rank[order]


class Rule:
    """Replace zpl4g.capped_parts and zpl5g.capped_parts5 by the length-rule versions for the length of a with-block, and put the engine's own back after (also on an error)."""

    def __init__(self, L, H):
        self.L, self.H = L, H

    def __enter__(self):
        self.o4, self.o5 = Z4.capped_parts, Z5.capped_parts5
        L, H = self.L, self.H
        Z4.capped_parts = lambda nodes, families, cap, scorer, cs: parts4(nodes, families, cap, scorer, cs, L, H)
        Z5.capped_parts5 = lambda nodes, families, win, scorer: parts5(nodes, families, win, scorer, L, H)
        return self

    def __exit__(self, *exc):
        Z4.capped_parts, Z5.capped_parts5 = self.o4, self.o5
        return False


def eval_range(op, d, plan, lo, hi, limit=None):
    """Per question: golds, and for every variant the golds in the pool and the pool size, graded by the engine (zpl3g.grade3) on the states the engine builds under the variant's cap functions."""
    res = {}
    for v, (nm, L, H) in enumerate(VARS):
        rows = []
        ctx = Rule(L, H) if nm != "eng" else None
        if ctx is not None:
            ctx.__enter__()
        try:
            for q, st, _seeds, _qv in B.states(op, d, plan, lo, hi, limit):
                P = np.zeros(1, dtype=np.int16)
                S = np.zeros(1, dtype=np.int32)
                Z3.grade3(plan, st, d.golds[q], P, S)
                rows.append((q, int(np.unique(d.golds[q]).size), int(P[0]), int(S[0])))
        finally:
            if ctx is not None:
                ctx.__exit__()
        res[v] = rows
    nq = len(res[0])
    Q = np.zeros((nq, len(QCOLS)), dtype=np.int64)
    for v in range(len(VN)):
        if len(res[v]) != nq or any(res[v][i][0] != res[0][i][0] for i in range(nq)):
            raise RuntimeError(f"variant {VN[v]} did not give the engine's questions")
        for i, r in enumerate(res[v]):
            if v == 0:
                Q[i, QI["q"]], Q[i, QI["ng"]] = r[0], r[1]
            Q[i, QI[f"p{v}"]], Q[i, QI[f"s{v}"]] = r[2], r[3]
    return Q


def g_s0(Q):
    """G-S0: the length-rule functions with the cell's own cap (`same`) give the engine's pool and grade on every question."""
    k = np.flatnonzero((Q[:, QI["p0"]] != Q[:, QI["p1"]]) | (Q[:, QI["s0"]] != Q[:, QI["s1"]]))
    return [[int(Q[i, QI["q"]]), [int(Q[i, QI["p0"]]), int(Q[i, QI["s0"]])], [int(Q[i, QI["p1"]]), int(Q[i, QI["s1"]])]] for i in k[:5]], int(k.size)


def old_rows(name, c):
    """The earlier filed rows of this set's chunk c to compare with: BRG5's control (the five budget-following sets) or BRG8's variants (MetaQA). (None, None, None) when there are none."""
    if name == "metaqa":
        p = ROOT / "outputs" / "brg8" / "metaqa" / f"scr_c{c:04d}.npz"
        want = {"eng": "base", "V1": "L50", "V2": "L100", "V3": "L100h10"}
    elif name in B.CELL:
        p = ROOT / "outputs" / "brg5" / name / f"scr_c{c:04d}.npz"
        want = {"eng": "control"}
    else:
        return None, None, None
    if not p.is_file():
        return None, None, None
    z = np.load(p, allow_pickle=False)
    try:
        quest, cols, var = z["quest"], [str(x) for x in z["qcols"]], [str(x) for x in z["variants"]]
    finally:
        z.close()
    return quest, {k: i for i, k in enumerate(cols)}, (var, want)


def g_s1(name, c, Q):
    """G-S1: on the screen chunks `eng` (and, for MetaQA, V1, V2, V3) equal the earlier filed rows of the same questions (golds in the pool and pool size): 0 differences. [] and None when nothing is filed."""
    old, oi, vw = old_rows(name, c)
    if old is None:
        return None
    var, want = vw
    if old.shape[0] < Q.shape[0] or not np.array_equal(old[:Q.shape[0], oi["q"]], Q[:, QI["q"]]):
        return [[c, "questions differ"]]
    bad = []
    for ours, theirs in want.items():
        j, v = var.index(theirs), VN.index(ours)
        pc = f"p{j}"
        sc = f"s{j}"
        a1 = int((old[:Q.shape[0], oi[pc]] != Q[:, QI[f"p{v}"]]).sum())
        a2 = int((old[:Q.shape[0], oi[sc]] != Q[:, QI[f"s{v}"]]).sum())
        if a1 or a2:
            bad.append([c, ours, theirs, a1, a2])
    return bad


def atomic_savez(path, **arrs):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, **arrs)
    os.replace(tmp, path)


def cmd_screen(a):
    name, op, src, order, crange, nch = B.load(a)
    cell = CELL9.get(name)
    if cell is None:
        stop(f"{name}: no declared cell")
    if name in KBS:
        got = rjson(ROOT / "outputs" / "drp1" / f"read_{name}.json")["arms"]["both"]["best_arm"]["2500"]["name"]
    else:
        got = rjson(ROOT / "outputs" / "drp1e" / f"read_{name}.json")["best_within"]["2500"]["both"]["name"]
    if got != cell:
        stop(f"G-C0 failed: {name}: the filed read's best cell is {got}, the declared one {cell}")
    d = D1.variant(src, "both")
    for x in D2.arm_fails(src, d, "both"):
        stop(f"G-D1 failed: {x}")
    plan = Z5.Plan5([cell])
    npos = nch if int(a.chunks) == 0 else min(int(a.chunks), nch)
    t0 = time.time()
    if a.smoke:
        m = int(a.smoke)
        lo, hi = crange(order[0], src.nq)
        Q = eval_range(op, d, plan, lo, hi, limit=m)
        bad, nb = g_s0(Q)
        b1 = g_s1(name, order[0], Q) if m >= 128 else None
        print(f"smoke {name} ({cell}): {m} questions, G-S0 mismatches {nb} {bad[:2]}; "
              f"ALL {dict(zip(VN, [round(float((Q[:, QI[f'p{v}']] == Q[:, QI['ng']]).mean()), 3) for v in range(len(VN))]))}; nodes {dict(zip(VN, [round(float(Q[:, QI[f's{v}']].mean())) for v in range(len(VN))]))}; {time.time() - t0:.1f} s")
        return 0 if not nb else 3
    outd = OUT / name
    outd.mkdir(parents=True, exist_ok=True)
    gate_p = outd / "gate_s.json"
    gates = rjson(gate_p) if gate_p.is_file() else {"stage": "BRG9 gates", "declared_in": DOC, "set": name, "cell": cell, "variants": [list(v) for v in VARS], "chunks": {}}
    gates["order"] = [int(c) for c in order[:npos]]
    gates["questions_expected"] = int(sum(crange(c, src.nq)[1] - crange(c, src.nq)[0] for c in order[:npos]))
    for pos in range(npos):
        c = order[pos]
        path = outd / f"scr_c{c:04d}.npz"
        if path.is_file():
            continue
        lo, hi = crange(c, src.nq)
        t1 = time.time()
        Q = eval_range(op, d, plan, lo, hi)
        bad, nb = g_s0(Q)
        b1 = g_s1(name, c, Q) if pos < NSCREEN else None
        gates["chunks"][str(c)] = {"position": pos, "questions": int(Q.shape[0]), "G-S0": {"mismatches": nb, "first": bad}, "G-S1": None if b1 is None else {"mismatches": len(b1), "first": b1[:5]}, "seconds": round(time.time() - t1, 1)}
        gates["questions_read"] = int(sum(v["questions"] for v in gates["chunks"].values()))
        done = len(gates["chunks"]) == npos
        ok = all(v["G-S0"]["mismatches"] == 0 and (v["G-S1"] is None or v["G-S1"]["mismatches"] == 0) for v in gates["chunks"].values())
        gates["G-S3"] = {"complete": bool(done and gates["questions_read"] == gates["questions_expected"])}
        gates["verdict"] = "PASS" if (done and ok and gates["G-S3"]["complete"]) else ("FAIL" if not ok else "RUNNING")
        gates["utc"] = utc()
        wjson(gate_p, gates)
        if nb or b1:
            stop(f"{name}: chunk {c}: G-S0 {bad[:3]} or G-S1 {b1[:3] if b1 else b1} failed")
        atomic_savez(path, quest=Q, qcols=np.asarray(QCOLS), variants=np.asarray(VN), cell=np.asarray(cell), lo=lo, hi=hi, pos=np.asarray(pos))
        log(f"{name}: chunk {c} (position {pos}; questions {lo}-{hi}): G-S0{' and G-S1' if b1 is not None else ''} PASS, {Q.shape[0]} questions, {time.time() - t1:.0f} s ({P2.peak_gb()} GB); {len(gates['chunks'])}/{npos} chunks")
    return 0


def aggregate(Q):
    ng = Q[:, QI["ng"]]
    nq = Q.shape[0]
    if nq == 0:
        return {"questions": 0}
    full = [(Q[:, QI[f"p{v}"]] == ng).astype(np.float64) for v in range(len(VN))]
    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.integers(0, nq, size=(BOOT_N, nq))
    out = {"questions": int(nq), "golds": int(ng.sum()), "variants": {}}
    for v, nm in enumerate(VN):
        rec = {"ALL": float(full[v].mean()), "golds_in_pool": float(Q[:, QI[f"p{v}"]].sum() / ng.sum()), "mean_nodes": float(Q[:, QI[f"s{v}"]].mean()), "nodes_over_eng": float(Q[:, QI[f"s{v}"]].mean() / Q[:, QI["s0"]].mean())}
        if v:
            diff = full[v] - full[0]
            bs = diff[boot].mean(axis=1)
            ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
            rec["vs_eng"] = {"diff": float(diff.mean()), "ci95": ci, "call": D2.gain_call(ci)}
        out["variants"][nm] = rec
    return out


def cmd_read(a):
    name = a.ds or a.kb
    folder = OUT / name
    gate = rjson(folder / "gate_s.json") if (folder / "gate_s.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        stop(f"{name}: no PASS gate_s.json: no number is read")
    files = sorted(folder.glob("scr_c*.npz"))
    if len(files) != len(gate["chunks"]) or len(files) != len(gate["order"]):
        stop(f"{name}: {len(files)} chunk files, {len(gate['chunks'])} gate entries, {len(gate['order'])} chunks")
    Qs, pos = [], []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["qcols"]] != list(QCOLS) or [str(x) for x in z["variants"]] != list(VN) or str(z["cell"]) != CELL9[name]:
                stop(f"{p.name}: not this module's columns, variants or cell")
            Qs.append(z["quest"])
            pos.append(np.full(z["quest"].shape[0], int(z["pos"])))
        finally:
            z.close()
    Q, P = np.concatenate(Qs), np.concatenate(pos)
    rec = {"stage": "BRG9 read", "declared_in": DOC, "set": name, "cell": CELL9[name], "utc": utc(), "variants": [list(v) for v in VARS], "chunks_read": len(files), "all": aggregate(Q), "screen": aggregate(Q[P < NSCREEN])}
    if len(files) > NSCREEN:
        rec["unseen"] = aggregate(Q[P >= NSCREEN])
        rec["half_even"] = aggregate(Q[(P >= NSCREEN) & (P % 2 == 0)])
        rec["half_odd"] = aggregate(Q[(P >= NSCREEN) & (P % 2 == 1)])
    wjson(OUT / f"read_{name}.json", rec)
    for part in [k for k in ("all", "unseen", "half_even", "half_odd") if k in rec]:
        x = rec[part]
        log(f"{name}: {part} ({x['questions']} questions): " + "; ".join(f"{k} {100 * v['ALL']:.1f} @{v['mean_nodes']:.0f}" + (f" ({100 * v['vs_eng']['diff']:+.2f} {v['vs_eng']['call']})" if "vs_eng" in v else "") for k, v in x["variants"].items()))
    return 0


def cmd_selftest(_a):
    rng = np.random.default_rng(31)
    n, dim = 600, 8
    V = rng.normal(size=(n, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    fams = B._toy(rng, n, deg=40) if hasattr(B, "_toy") else None
    assert fams is not None
    ok = 0
    for trial in range(40):
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        nodes = rng.choice(n, size=int(rng.integers(1, 70)), replace=False).astype(np.int64)
        for fs in (fams, [fams[0]]):
            sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
            for cap in (3, 8, 25):
                for cs in (False, True):
                    a1 = parts4(nodes, fs, cap, sc, cs, None, None)
                    a2 = Z4.capped_parts(nodes, fs, cap, sc, cs)
                    assert all(np.array_equal(x, y) for x, y in zip(a1, a2)), ("parts4 with the cell's own cap differs", cap, cs)
                    a3 = parts4(nodes, fs, cap, sc, cs, cap, cap)
                    assert all(np.array_equal(x, y) for x, y in zip(a3, a2)), ("L = H = cap differs", cap, cs)
            for win in (4, 16, 512):
                b1 = parts5(nodes, fs, win, sc, None, None)
                b2 = Z5.capped_parts5(nodes, fs, win, sc)
                assert all(np.array_equal(x, y) for x, y in zip(b1, b2)), ("parts5 with the cell's own window differs", win)
                b3 = parts5(nodes, fs, win, sc, win, win)
                assert all(np.array_equal(x, y) for x, y in zip(b3, b2)), "parts5 L = H = window differs"
            for cs in (False, True):
                m = parts4(nodes, fs, 25, sc, cs, 20, 5)
                for f_i, f in enumerate(fs):
                    rl = (f.indptr[nodes + 1] - f.indptr[nodes])
                    got = np.bincount(m[1][m[2] == f_i], minlength=nodes.size)
                    exp = np.where(rl <= 20, rl, np.minimum(rl, 5)) if not cs else np.where(rl <= 20, np.minimum(rl, 512), np.minimum(np.minimum(rl, 512), 5))
                    assert np.array_equal(got, exp), ("row rule parts4", cs)
            m5 = parts5(nodes, fs, 512, sc, 20, 5)
            for f_i, f in enumerate(fs):
                rl = (f.indptr[nodes + 1] - f.indptr[nodes])
                got = np.bincount(m5[1][m5[2] == f_i], minlength=nodes.size)
                assert np.array_equal(got, np.where(rl <= 20, np.minimum(rl, 512), np.minimum(np.minimum(rl, 512), 5))), "row rule parts5"
            sc.release()
        ok += 1
    o4, o5 = Z4.capped_parts, Z5.capped_parts5
    with Rule(50, 25):
        assert Z4.capped_parts is not o4 and Z5.capped_parts5 is not o5
    assert Z4.capped_parts is o4 and Z5.capped_parts5 is o5, "the engine's functions are not restored"
    try:
        with Rule(50, 25):
            raise ValueError("x")
    except ValueError:
        pass
    assert Z4.capped_parts is o4 and Z5.capped_parts5 is o5, "not restored after an error"
    # the expansion of a toy walk under Rule(None, None) equals the engine's, and under a rule it differs from the engine only where a row is longer than L
    seeds = rng.choice(n, size=4, replace=False)
    st = dict(Z5.SET5["h3_c25_v2000"])
    st["per_seed_cap"] = st["per_frontier_cap"] = 8
    st["visited_cap"] = 80
    for mods in ((), ("cs", "vr"), ("rr",)):
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
        e0 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc)
        sc.release()
        sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
        with Rule(None, None):
            e1 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc)
        with Rule(40, 5):
            e2 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc)
        sc.release()
        assert np.array_equal(e0, e1), "Rule(None, None) changes the engine's expansion"
        assert np.unique(e2).size == e2.size
    assert VN[0] == "eng" and VN[1] == "same" and len(VARS) == 9 and len(QCOLS) == 2 + 2 * len(VN)
    nq = 500
    rs = np.random.default_rng(5)
    base = rs.random(nq) < 0.8
    Qs = np.zeros((nq, len(QCOLS)), np.int64)
    Qs[:, QI["ng"]] = 2
    for v in range(len(VN)):
        Qs[:, QI[f"s{v}"]] = 2000
        Qs[:, QI[f"p{v}"]] = np.where(base | (rs.random(nq) < 0.5) if v == 3 else base, 2, 1)
    g = aggregate(Qs)["variants"]
    assert g["V2"]["vs_eng"]["call"] == "ABOVE" and g["same"]["vs_eng"]["call"] == "LEVEL"
    assert aggregate(Qs[:0])["questions"] == 0
    print(f"zbrg9 selftest: parts4 and parts5 with the cell's own cap or window equal zpl4g.capped_parts and zpl5g.capped_parts5 (with and without cs, one and several families, {ok} trials), L = H = cap equal them too, "
          f"the row rule keeps a row of at most L whole and cuts a longer one at H, the functions are restored after a with-block and after an error, the toy walk under Rule(None, None) equals the engine's expansion, "
          f"the paired calls on a synthetic case: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for c in ("screen", "read"):
        p = sub.add_parser(c)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--ds", choices=[s for s in SETS if s not in KBS])
        g.add_argument("--kb", choices=KBS)
        p.add_argument("--host", action="store_true")
        if c == "screen":
            p.add_argument("--smoke", type=int, default=0)
            p.add_argument("--chunks", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return cmd_selftest(a)
    return cmd_screen(a) if a.cmd == "screen" else cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
