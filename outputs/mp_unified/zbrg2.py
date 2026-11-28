"""BRG2 (docs/BRG2_ORDER_AND_WINDOW_SCREEN.md): which candidate order and which scoring window recover the golds that BRG1 found cut at a hop's budget or beyond the window.

Read-only screen on the subsets of BRG1: no pool, cell or rule of the filed work is changed and nothing is adopted. The walk is the budget-following walk of PL5 (zpl5g.expand5) re-run hop by hop as in zbrg1.py with
the hop's candidate order and scoring window replaced by one of six variants (control and four orders at the window 512, control and the primary order at the wide window 4,096). zbrg1.py and the pool code
(zpl2g.py to zpl6g.py, zdrp1.py, zdrp2.py, zstk3.py) are imported and edited nowhere.

    python -E -P -B outputs/mp_unified/zbrg2.py selftest
    python -E -P -B outputs/mp_unified/zbrg2.py order --ds D | --kb K [--pos 0,1,2] [--smoke M] [--host]
        # chunk positions of the seeded order (BRG1's three by default): outputs/brg2/<name>/ord_c<chunk>.npz and gate_order.json; --smoke M grades M questions of the first chunk, prints the gate only
    python -E -P -B outputs/mp_unified/zbrg2.py read --ds D | --kb K
        # aggregates the chunk files: outputs/brg2/read_<name>.json

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
import zdrp2 as D2  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl5g as Z5  # noqa: E402

DOC = "docs/BRG2_ORDER_AND_WINDOW_SCREEN.md"
OUT = ROOT / "outputs" / "brg2"
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc
# (candidate order, scoring window): control at 512 is the filed walk; `fus` is the primary order of the page, `sup` and `ppr` its two parts
VARIANTS = (("control", 512), ("sup", 512), ("ppr", 512), ("fus", 512), ("control", Z5.WIDE), ("fus", Z5.WIDE))
VNAME = tuple(f"{k}@{w}" for k, w in VARIANTS)
HOPS = (1, 2, 3)
QCOLS2 = ("q", "ng") + tuple(f"{k}{v}" for v in range(len(VARIANTS)) for k in ("p", "s", "u")) + tuple(f"{k}{v}_{h}" for v in range(len(VARIANTS)) for h in HOPS for k in ("c", "k"))
QI2 = {c: i for i, c in enumerate(QCOLS2)}
BOOT_N, BOOT_SEED = 2000, 20261128


# ── the orders (pure, self-tested) ───────────────────────────────────────────

def _group_order(kind, cols, grp, rank, key, scorer, n_par):
    """The distinct nodes of one candidate list in the variant's order. control: PL5's own. sup: the number of distinct frontier nodes whose row lists the node, most first. ppr: the sum over those
    frontier nodes of 1 / (RRF_C + place of the frontier node + 1) (a frontier node's place is its place in the frontier, the order the previous hop kept it in). fus: the reciprocal-rank fusion of the
    control order and the ppr order. Ties go to the control order."""
    ctrl = Z5.order_group5(cols, grp, rank, key, scorer, n_par)
    if kind == "control":
        return ctrl
    u = np.unique(cols)
    m = int(grp.max()) + 1
    pr = np.unique(cols.astype(np.int64) * m + grp.astype(np.int64))
    inv = np.searchsorted(u, pr // m)
    pp = pr % m
    s = np.bincount(inv, minlength=u.size).astype(np.float64) if kind == "sup" else np.bincount(inv, weights=1.0 / (Z5.RRF_C + pp + 1.0), minlength=u.size)
    pos = np.empty(u.size, dtype=np.int64)
    pos[np.searchsorted(u, ctrl)] = np.arange(ctrl.size, dtype=np.int64)
    o1 = u[np.lexsort((pos, -s))]
    if kind in ("sup", "ppr"):
        return o1
    p1 = np.empty(u.size, dtype=np.int64)
    p1[np.searchsorted(u, o1)] = np.arange(u.size, dtype=np.int64)
    f = 1.0 / (Z5.RRF_C + pos + 1.0) + 1.0 / (Z5.RRF_C + p1 + 1.0)
    return u[np.lexsort((pos, -f))]


def order_alt(kind, cols, grp, fam, rank, key, rr, scorer, n_par):
    """zpl5g.order_fresh5 with the group order of the variant; the families' lists are merged by place exactly as there."""
    if cols.size == 0:
        return cols
    if not rr:
        return _group_order(kind, cols, grp, rank, key, scorer, n_par)
    nodes, pos, fm = [], [], []
    for f in np.unique(fam):
        m = fam == f
        o = _group_order(kind, cols[m], grp[m], rank[m], key, scorer, n_par)
        nodes.append(o)
        pos.append(np.arange(o.size, dtype=np.int64))
        fm.append(np.full(o.size, int(f), dtype=np.int64))
    nodes, pos, fm = np.concatenate(nodes), np.concatenate(pos), np.concatenate(fm)
    nodes = nodes[np.lexsort((fm, pos))]
    return Z5.m3b_pools._first_occurrence_unique(nodes)


def walk_alt(seeds, fams1, fams2, setting, mods, scorer, kind, win):
    """zbrg1.trace_walk with the variant's order and scoring window. Returns (expansion, hop records {h, budget, fa})."""
    key = next(m for m in Z5.KEYS5 if m in mods)
    rr = "rr" in mods
    hops, limit = int(setting["hops"]), int(setting["visited_cap"])
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
        fa = order_alt(kind, cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))
        fresh = fa[:budget]
        recs.append({"h": h, "budget": int(budget), "fa": fa})
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return (np.concatenate(out) if out else np.empty(0, dtype=np.int64)), recs


def question_row(q, golds, head, exps, recs_all):
    """One row of QCOLS2: per variant the golds in the pool (head and expansion) and its size, per hop the golds among the candidates and among the kept."""
    golds = np.asarray(golds, dtype=np.int64)
    r = {c: 0 for c in QCOLS2}
    r["q"], r["ng"] = q, int(golds.size)
    for v, (exp, recs) in enumerate(zip(exps, recs_all)):
        pool = np.union1d(head, exp)
        r[f"p{v}"], r[f"s{v}"] = int(np.isin(golds, pool).sum()), int(pool.size)
        cand = np.concatenate([head] + [rec["fa"] for rec in recs]) if recs else head
        r[f"u{v}"] = int(np.isin(golds, cand).sum())          # golds in the head or among any hop's candidates: the pool of a perfect order at this variant's frontier
        for rec in recs[:3]:
            h = rec["h"]
            r[f"c{v}_{h}"] = int(np.isin(golds, rec["fa"]).sum())
            r[f"k{v}_{h}"] = int(np.isin(golds, rec["fa"][:rec["budget"]]).sum())
    return np.asarray([r[c] for c in QCOLS2], dtype=np.int64)


# ── order ────────────────────────────────────────────────────────────────────

def order_range(op, d, plan, cell, lo, hi, limit=None):
    ln, b, regime, setting, mods, n1, n2 = B.cell_parts(cell)
    fams1, fams2 = [d.fams[f] for f in n1], [d.fams[f] for f in n2]
    walk = Z5.parse5(cell)[3]
    Q, bad = [], []
    for q, st, seeds, qv in B.states(op, d, plan, lo, hi, limit):
        scorer = Z3.Scorer(d.dvec, {"q": np.asarray(qv, dtype=np.float64)}, d.n)
        try:
            exps, recs_all = [], []
            for kind, win in VARIANTS:
                exp, recs = walk_alt(seeds, fams1, fams2, dict(setting, window=int(win)), mods, scorer, kind, int(win))
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
        if int(row[QI2["p0"]]) != int(P[0]) or int(row[QI2["s0"]]) != int(S[0]):
            bad.append([q, "grade", [int(row[QI2["p0"]]), int(row[QI2["s0"]])], [int(P[0]), int(S[0])]])
        Q.append(row)
    return (np.asarray(Q) if Q else np.zeros((0, len(QCOLS2)), np.int64)), bad


def cmd_order(a):
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
        Q, bad = order_range(op, d, plan, cell, lo, hi, limit=int(a.smoke))
        print(f"smoke {name}: {int(a.smoke)} questions, G-O1 mismatches {len(bad)} {bad[:3]}, {time.time() - t0:.1f} s")
        return 0 if not bad else 3
    poss = [int(x) for x in str(a.pos).split(",")] if a.pos else list(range(min(B.SLICE, nch)))
    outd.mkdir(parents=True, exist_ok=True)
    gate_p = outd / "gate_order.json"
    for pos in poss:
        c = order[pos]
        path = outd / f"ord_c{c:04d}.npz"
        if path.is_file():
            continue
        lo, hi = crange(c, src.nq)
        t0 = time.time()
        Q, bad = order_range(op, d, plan, cell, lo, hi)
        gates = rjson(gate_p) if gate_p.is_file() else {"stage": "BRG2 gates", "declared_in": DOC, "set": name, "cell": cell, "variants": list(VNAME), "chunks": {}}
        gates["chunks"][str(c)] = {"questions": hi - lo, "G-O1": {"mismatches": len(bad), "first": bad[:5]}, "seconds": round(time.time() - t0, 1)}
        gates["utc"], gates["verdict"] = utc(), ("PASS" if all(v["G-O1"]["mismatches"] == 0 for v in gates["chunks"].values()) else "FAIL")
        wjson(gate_p, gates)
        if bad:
            stop(f"{name}: chunk {c}: G-O1 failed ({len(bad)} mismatches, first {bad[:3]})")
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as f:
            np.savez(f, quest=Q, qcols=np.asarray(QCOLS2), variants=np.asarray(VNAME), cell=np.asarray(cell), lo=lo, hi=hi)
        os.replace(tmp, path)
        log(f"{name}: chunk {c} (questions {lo}-{hi}): G-O1 PASS, {Q.shape[0]} questions, {time.time() - t0:.0f} s ({P2.peak_gb()} GB)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────────

def aggregate(Q):
    ng = Q[:, QI2["ng"]]
    nq = Q.shape[0]
    full = [(Q[:, QI2[f"p{v}"]] == ng).astype(np.float64) for v in range(len(VARIANTS))]
    rng = np.random.default_rng(BOOT_SEED)
    boot = rng.integers(0, nq, size=(BOOT_N, nq))
    out = {"questions": int(nq), "golds": int(ng.sum()), "variants": {}}
    for v, nm in enumerate(VNAME):
        rec = {"ALL": float(full[v].mean()), "golds_in_pool": float(Q[:, QI2[f"p{v}"]].sum() / ng.sum()), "mean_nodes": float(Q[:, QI2[f"s{v}"]].mean()),
               "ceiling_ALL": float((Q[:, QI2[f"u{v}"]] == ng).mean()), "ceiling_golds": float(Q[:, QI2[f"u{v}"]].sum() / ng.sum())}
        if v:
            diff = full[v] - full[0]
            bs = diff[boot].mean(axis=1)
            ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
            rec.update({"minus_control": float(diff.mean()), "ci95": ci, "call": D2.gain_call(ci), "nodes_over_control": float(Q[:, QI2[f"s{v}"]].mean() / Q[:, QI2["s0"]].mean())})
        rec["hops"] = {str(h): {"candidate_golds": int(Q[:, QI2[f"c{v}_{h}"]].sum()), "kept_golds": int(Q[:, QI2[f"k{v}_{h}"]].sum()),
                                "kept_share": (float(Q[:, QI2[f"k{v}_{h}"]].sum() / Q[:, QI2[f"c{v}_{h}"]].sum()) if Q[:, QI2[f"c{v}_{h}"]].sum() else None)} for h in HOPS}
        for lab, m in (("1", ng == 1), ("2", ng == 2), ("3", ng == 3), ("4+", ng >= 4)):
            if m.any():
                rec.setdefault("by_gold_count", {})[lab] = {"questions": int(m.sum()), "ALL": float(full[v][m].mean())}
        out["variants"][nm] = rec
    return out


def cmd_read(a):
    name = a.kb or a.ds
    folder = OUT / name
    gate = rjson(folder / "gate_order.json") if (folder / "gate_order.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        stop(f"{name}: no PASS gate_order.json: no number is read")
    files = sorted(folder.glob("ord_c*.npz"))
    if not files:
        stop(f"{name}: no chunk file")
    Qs = []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["qcols"]] != list(QCOLS2) or [str(x) for x in z["variants"]] != list(VNAME) or str(z["cell"]) != B.CELL[name]:
                stop(f"{p.name}: not this module's columns, variants or cell")
            Qs.append(z["quest"])
        finally:
            z.close()
    rec = {"stage": "BRG2 read", "declared_in": DOC, "set": name, "cell": B.CELL[name], "utc": utc(), "chunks": [p.name for p in files], **aggregate(np.concatenate(Qs))}
    wjson(OUT / f"read_{name}.json", rec)
    v = rec["variants"]
    log(f"{name}: read written ({rec['questions']} questions): " + "; ".join(f"{k} {100 * x['ALL']:.1f}" + (f" ({100 * x['minus_control']:+.2f} {x['call']})" if "call" in x else "") for k, x in v.items()))
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def cmd_selftest(_a):
    B.Z6.prepare(False)
    rng = np.random.default_rng(11)
    n, dim = 400, 8
    V = rng.normal(size=(n, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    fams = B._toy(rng, n, deg=12)
    ok = 0
    for trial in range(24):
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        seeds = rng.choice(n, size=5, replace=False)
        for stg, mods in (("h2_bf_v1000", ("cs", "vr")), ("h3_bf_v1000", ("cs", "fu")), ("h2_bf_v1000", ("cs", "vr", "rr")), ("h3_bf_v2000", ("cs", "vr"))):
            st = dict(Z5.SET5[stg])
            st["visited_cap"] = 60 if trial % 2 else int(st["visited_cap"])
            for win in (512, 5):                                       # a window of 5 makes the window bite on the toy rows
                s_ = dict(st, window=win)
                sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
                e1 = Z5.expand5(seeds, [fams[0]], fams, s_, mods, sc)
                sc.release()
                sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
                e2, _r = walk_alt(seeds, [fams[0]], fams, s_, mods, sc, "control", win)
                assert np.array_equal(e1, e2), ("control differs from expand5", trial, stg, mods, win)
                for kind in ("sup", "ppr", "fus"):
                    e3, recs = walk_alt(seeds, [fams[0]], fams, s_, mods, sc, kind, win)
                    assert e3.size <= e1.size + int(s_["visited_cap"]) and np.unique(e3).size == e3.size, (kind, trial)
                    for rec in recs:
                        assert np.unique(rec["fa"]).size == rec["fa"].size
                sc.release()
                ok += 1
    # the orders on a hand-made candidate list: node 7 listed by three frontier nodes (0, 1, 2), node 9 by one (0), node 5 by two (1, 2), node 4 by one (2)
    cols = np.asarray([9, 7, 5, 7, 5, 7, 4], dtype=np.int64)
    grp = np.asarray([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    rank = np.asarray([0, 1, 0, 1, 0, 1, 2], dtype=np.int64)
    sc = Z3.Scorer(lambda x: np.zeros((len(x), 1)), {"q": np.zeros(1)}, 20)
    ctrl = Z5.order_group5(cols, grp, rank, "rk", sc, 3)
    sup = _group_order("sup", cols, grp, rank, "rk", sc, 3)
    ppr = _group_order("ppr", cols, grp, rank, "rk", sc, 3)
    fus = _group_order("fus", cols, grp, rank, "rk", sc, 3)
    sc.release()
    assert sorted(ctrl.tolist()) == sorted(sup.tolist()) == sorted(ppr.tolist()) == sorted(fus.tolist()) == [4, 5, 7, 9]
    assert sup.tolist()[:2] == [7, 5], sup.tolist()                  # three parents, then two
    assert ppr.tolist()[0] == 7, ppr.tolist()                        # 7 has the most and the best-placed parents
    print(f"zbrg2 selftest: control equals expand5 on {ok} walks (windows 512 and 5, four settings), the orders keep the candidate set, support and ppr counts match the hand-made list: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for c in ("order", "read"):
        p = sub.add_parser(c)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--ds", choices=B.SEVEN)
        g.add_argument("--kb", choices=B.KBS)
        p.add_argument("--host", action="store_true")
        if c == "order":
            p.add_argument("--pos", default="")
            p.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return cmd_selftest(a)
    if a.cmd == "order":
        return cmd_order(a)
    return cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
