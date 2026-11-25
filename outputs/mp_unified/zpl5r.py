"""PL5 report and diagnostics (docs/PL5_COVERAGE_ROUND4.md): the diagnostics of section 1 and, with the results, the report tables.

    python outputs/mp_unified/zpl5r.py place --ds D [D ...] [--n 300] [--host]     # D3: where the cap-cut golds of PL4's chosen cell sit (labels read; post-hoc; decides nothing)

`place` (D3) takes PL4's chosen cell (R95 of the reading `all`, outputs/pl4/<d>/select.json) on the SHORT datasets, re-walks it on n dev questions evenly spaced over
the carve (the same questions as PL4's D1/D2, `zpl4r.py miss`), and for every gold missing from the pool that D2 calls a cap cut (a neighbour of the hop's frontier that the
per-seed or per-frontier cap dropped) asks four questions of the hop's candidate list read WITHOUT any per-node cap:
  - is the gold inside the scoring window (the first SCORE_WINDOW = 512 entries of a parent's row, in row order)? A gold outside every parent's window is out of reach of any rule
    that reads the window, whatever the budget;
  - if inside: its place in the best parent's cosine-ordered list (0 = the parent's nearest neighbour to the question);
  - its position among the hop's fresh candidates (not visited before the hop) under the order `vr` (cosine to the question, highest first), `rk` (the smallest place any parent
    gives it, then cosine) and `sp` (the number of parents that name it, then cosine): the number of fresh nodes a hop must keep to keep the gold.
Nothing here chooses a cell or a rule; it only counts. Writes outputs/pl5/<d>/place.json.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PL5 = ROOT / "outputs" / "pl5"
PL4 = ROOT / "outputs" / "pl4"
CUTS = (250, 500, 1000, 1500, 2000, 3000, 5000)


def row_position(np, nodes, families, gold):
    """The smallest position (0-based, in the stored row order) at which `gold` stands in any frontier node's row of any family; None if it stands in none."""
    best = None
    for fam in families:
        starts = fam.indptr[nodes]
        lens = fam.indptr[nodes + 1] - starts
        total = int(lens.sum())
        if total == 0:
            continue
        off = np.cumsum(lens) - lens
        within = np.arange(total, dtype=np.int64) - np.repeat(off, lens)
        c = fam.col[np.repeat(starts, lens) + within].astype(np.int64, copy=False)
        hit = np.flatnonzero(c == gold)
        if hit.size:
            v = int(within[hit].min())
            best = v if best is None else min(best, v)
    return best


WINDOWS = (512, 1024, 2048, 4096, 8192, 16384, 65536)


def cmd_place(a):
    for _p in (ROOT / "scripts", ROOT / "src", HERE):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import numpy as np
    import zpl4g as Z4
    import zpl4r as R4
    Z3, P2 = Z4.Z3, Z4.P2
    P2.HOST = bool(a.host)
    op = P2.opened()
    M3A = op.m3a
    W = Z4.SCORE_WINDOW
    for ds in a.ds:
        sel = Z4.rjson(PL4 / ds / "select.json")
        name = sel["picks"]["R95_all"]["name"]
        p = Z4.parse4(name)
        if p[0] not in ("G", "K"):
            Z4.log(f"{ds}: {name} has no walk")
            continue
        src = P2.load_source(ds, "dev", "frozen")
        plan = Z4.Plan4([name])
        n = min(a.n, src.nq)
        regime, stg, mods = p[3]
        h1, h2 = Z4.REG3[regime]
        idx = np.unique(np.linspace(0, src.nq - 1, n).astype(np.int64))
        rows, rows_b = [], []
        n_missing = n_cap = n_bud = 0
        t0 = Z4.time.time()
        for q0 in idx:
            for q, st in Z4.iter_states4(op, src, plan, int(q0), int(q0) + 1):
                wk = st.W[p[3]]
                exp = wk.exp
                lst = st.L[p[1]]
                base = lst.L[:p[2]]
                if p[0] == "G":
                    pool = np.union1d(base, exp)
                else:
                    ordered = exp[wk.ro[p[5]]] if exp.size else exp
                    cand = ordered[lst.pos(ordered) >= p[2]] if exp.size else exp
                    pool = np.union1d(base, cand[:p[4]])
                g = src.golds[q]
                miss = g[~np.isin(g, pool)]
                if miss.size == 0:
                    continue
                n_missing += int(miss.size)
                d_ = src.dense[q].astype(np.int64)
                s_ = src.splade[q].astype(np.int64)
                seeds5 = M3A.stable_union_rows(d_[None, :5], s_[None, :5])[0]
                f1_ = M3A.rrf_fused_rows(d_[None, :], s_[None, :], P2.RRF_CONST)[0]
                sd = np.asarray(f1_[:10] if "s10" in mods else (f1_[:20] if "s20" in mods else seeds5), dtype=np.int64)
                scorer = Z3.Scorer(src.dvec, {"q": np.asarray(src.qvec(q, q + 1)[0], dtype=np.float64)}, src.n)
                trace = R4.trace_walk(np, Z4, sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], P2.SETTINGS[stg], mods, scorer)
                cs_cell = "cs" in mods
                cap2 = int(P2.SETTINGS[stg].get("per_frontier_cap") or 0)
                limit = int(P2.SETTINGS[stg].get("visited_cap") or 0)
                for gold in miss:
                    gold = int(gold)
                    bh = next((t["h"] for t in trace if t["h"] > 1 and gold in t["capped"] and gold not in t["kept"]), None)
                    if bh is not None:                             # a budget cut (D2): where would each order have put it among the hop's own capped candidates?
                        n_bud += 1
                        t = trace[bh - 1]
                        vis = np.union1d(sd, np.concatenate([x["kept"] for x in trace[:bh - 1]]))
                        cols, grp, fam, rank = Z4.capped_parts(t["frontier"], t["fams"], cap2, scorer, cs_cell)
                        k = np.isin(cols, vis, invert=True)
                        rb = {"q": int(q), "hop": int(bh), "left": int(limit - vis.size), "pos": {}}
                        for key in ("vr", "rk", "sp"):
                            f = Z4.order_fresh(cols[k], grp[k], fam[k], rank[k], key, False, scorer, int(t["frontier"].size))
                            rb["pos"][key] = int(np.flatnonzero(f == gold)[0])
                            rb["n_fresh"] = int(f.size)
                        rows_b.append(rb)
                        continue
                    cap_h = next((t["h"] for t in trace if gold not in t["capped"] and gold in R4.uncapped_neighbours(np, t["frontier"], t["fams"])), None)
                    if cap_h is None:
                        continue                                   # upstream or beyond the reach (D2)
                    n_cap += 1
                    t = trace[cap_h - 1]
                    vis = np.union1d(sd, np.concatenate([x["kept"] for x in trace[:cap_h - 1]])) if cap_h > 1 else np.asarray(sd, dtype=np.int64)
                    cols, grp, fam, rank = Z4.capped_parts(t["frontier"], t["fams"], W, scorer, True)
                    k = np.isin(cols, vis, invert=True)
                    row = {"q": int(q), "hop": int(cap_h), "inside": False, "place": None, "n_fresh": None, "pos": {}, "row_pos": None}
                    if not (k.any() and bool(np.isin(gold, cols[k]).any())):
                        row["row_pos"] = row_position(np, t["frontier"], t["fams"], gold)
                    if k.any() and bool(np.isin(gold, cols[k]).any()):
                        row["inside"] = True
                        row["place"] = int(rank[k][cols[k] == gold].min())
                        for key in ("vr", "rk", "sp"):
                            f = Z4.order_fresh(cols[k], grp[k], fam[k], rank[k], key, False, scorer, int(t["frontier"].size))
                            row["pos"][key] = int(np.flatnonzero(f == gold)[0])
                            row["n_fresh"] = int(f.size)
                    rows.append(row)
                scorer.release()
        summ = {"golds_missing": n_missing, "cap_cut": n_cap, "by_hop": {}}
        for h in sorted({r["hop"] for r in rows}):
            rh = [r for r in rows if r["hop"] == h]
            ins = [r for r in rh if r["inside"]]
            summ["by_hop"][str(h)] = {"cap_cut": len(rh), "inside_window": len(ins), "outside_window": len(rh) - len(ins),
                                      "outside_row_position_within": {str(w_): int(sum(r["row_pos"] is not None and r["row_pos"] < w_ for r in rh if not r["inside"])) for w_ in WINDOWS},
                                      "place_median": (float(np.median([r["place"] for r in ins])) if ins else None),
                                      "n_fresh_median": (float(np.median([r["n_fresh"] for r in ins])) if ins else None),
                                      "keeps_at": {key: {str(c): int(sum(r["pos"][key] < c for r in ins)) for c in CUTS} for key in ("vr", "rk", "sp")}}
        summ["budget_cut"] = n_bud
        summ["budget_by_hop"] = {}
        for h in sorted({r["hop"] for r in rows_b}):
            rh = [r for r in rows_b if r["hop"] == h]
            summ["budget_by_hop"][str(h)] = {"budget_cut": len(rh), "n_fresh_median": float(np.median([r["n_fresh"] for r in rh])), "left_median": float(np.median([r["left"] for r in rh])),
                                             "keeps_at": {key: {str(c): int(sum(r["pos"][key] < c for r in rh)) for c in CUTS} for key in ("vr", "rk", "sp")},
                                             "fits_in_what_was_left": {key: int(sum(r["pos"][key] < r["left"] for r in rh)) for key in ("vr", "rk", "sp")}}
        out = {"stage": "PL5 D3 (post-hoc diagnostic on PL4's chosen cell, labels read, decides nothing)", "declared_in": "docs/PL5_COVERAGE_ROUND4.md", "dataset": ds,
               "pl4_cell": name, "questions": int(idx.size), "carve": "dev", "first": "frozen", "window": W, "cuts": list(CUTS), "summary": summ, "rows": rows, "rows_budget": rows_b,
               "seconds": round(Z4.time.time() - t0, 1), "utc": Z4.utc()}
        (PL5 / ds).mkdir(parents=True, exist_ok=True)
        Z4.wjson(PL5 / ds / "place.json", out)
        Z4.log(f"D3 {ds} {name}: {idx.size} q, {n_missing} golds missing, {n_cap} cap-cut, {n_bud} budget-cut; cap " + json.dumps(summ["by_hop"]) + "; budget " + json.dumps(summ["budget_by_hop"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("place")
    p.add_argument("--ds", nargs="+", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    return {"place": cmd_place}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
