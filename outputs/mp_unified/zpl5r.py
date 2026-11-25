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


def trace_walk5(np, Z5, seeds, fams1, fams2, setting, mods, scorer):
    """expand5's loop, recording at every hop its frontier, its family list, the fresh candidate entries (as a set) and the nodes it kept. A walk that is not
    budget-following is zpl4r.trace_walk (PL4's, PL3's, PL2's and the one-hop window walks). `miss` checks the trace against expand5's walk."""
    import zpl4r as R4
    if not setting.get("bf"):
        return R4.trace_walk(np, Z5.Z4, seeds, fams1, fams2, setting, mods, scorer)
    key = next(m for m in Z5.KEYS5 if m in mods)
    rr = "rr" in mods
    hops, limit, win = int(setting["hops"]), int(setting["visited_cap"]), int(setting["window"])
    seeds = np.asarray(seeds, dtype=np.int64)
    visited = np.unique(seeds)
    frontier = seeds
    out = []
    for h in range(1, hops + 1):
        rem = limit - int(visited.size)
        if rem <= 0 or frontier.size == 0:
            break
        budget = -(-rem // (hops - h + 1))
        fams = fams1 if h == 1 else fams2
        cols, grp, fam, rank = Z5.capped_parts5(frontier, fams, win, scorer)
        if cols.size == 0:
            break
        k = np.isin(cols, visited, invert=True)
        fresh = Z5.order_fresh5(cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))[:budget]
        out.append({"h": h, "frontier": frontier, "fams": fams, "capped": np.unique(cols[k]), "kept": fresh})
        if fresh.size == 0:
            break
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return out


def uncapped5(np, nodes, families):
    import zpl4r as R4
    return R4.uncapped_neighbours(np, nodes, families)


def cmd_miss(a):
    """D1 and D2 (PL4 section 10, docs/PL5_COVERAGE_ROUND4.md section 10) at the chosen cell of the reading `all` of a SHORT dataset, from PL5's select.json."""
    for _p in (ROOT / "scripts", ROOT / "src", HERE):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import numpy as np
    import zpl5g as Z5
    Z3, P2 = Z5.Z3, Z5.P2
    P2.HOST = bool(a.host)
    out_root, doc = PL5, Z5.DOC
    if getattr(a, "pl6", False):
        # PL6 (docs/PL6_COVERAGE_ROUND5.md section 7): the same diagnostic at the cell rule U names in PL6's select.json; the ladder's settings are added by zpl6g.prepare()
        import zpl6g as Z6
        Z6.prepare(bool(a.host))
        out_root, doc = ROOT / "outputs" / "pl6", Z6.DOC
    pl7 = bool(getattr(a, "pl7", False))
    if pl7:
        # PL7 (docs/PL7_COVERAGE_RESTRICTION.md section 7): the same diagnostic at the cell R7(2,500) names in PL7's select.json (picks["2500"]["R95_all"])
        import zpl7g as Z7
        Z7.prepare7(bool(a.host))
        out_root, doc = ROOT / "outputs" / "pl7", Z7.DOC
    op = P2.opened()
    M3A = op.m3a
    unc = {"hops": 3, "per_seed_cap": 10 ** 9, "per_frontier_cap": 10 ** 9, "visited_cap": 10 ** 12}
    for ds in a.ds:
        sel = Z5.rjson(out_root / ds / "select.json")
        pk = sel["picks"]["2500"]["R95_all"] if pl7 else sel["picks"]["R95_all"]
        name = pk["name"]
        if pk["reached"] and not a.all:
            Z5.log(f"{ds}: R95 reaches 0.95 ({name}); D1 is for the SHORT datasets (use --all to run anyway)")
            continue
        p = Z5.parse5(name)
        if p[0] not in ("G", "K"):
            Z5.log(f"{ds}: the chosen cell {name} has no walk; nothing to classify")
            continue
        src = P2.load_source(ds, "dev", "frozen")
        plan = Z5.Plan5([name])
        n = min(a.n, src.nq)
        regime, stg, mods = p[3]
        h1, h2 = Z5.REG3[regime]
        setting = Z5.SET5[stg]
        fam_all = [src.fams[f] for f in P2.REGIMES["FULL"]]
        tot = {k: 0 for k in ("q_missing", "golds_missing", "R", "C", "U", "U_in_lists", "q_R", "q_C", "q_U")}
        stage = {}
        mismatch = 0
        nodes = []
        t0 = Z5.time.time()
        idx = np.unique(np.linspace(0, src.nq - 1, n).astype(np.int64))      # evenly spaced over the carve: the questions of PL3's D1 and PL4's D1/D2

        def sample():
            for q0 in idx:
                yield from Z5.iter_states5(op, src, plan, int(q0), int(q0) + 1)
        for q, st in sample():
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
            nodes.append(int(pool.size))
            g = src.golds[q]
            miss = g[~np.isin(g, pool)]
            if miss.size == 0:
                continue
            d = src.dense[q].astype(np.int64)
            s_ = src.splade[q].astype(np.int64)
            seeds5 = M3A.stable_union_rows(d[None, :5], s_[None, :5])[0]
            f1_ = M3A.rrf_fused_rows(d[None, :], s_[None, :], P2.RRF_CONST)[0]
            sd = np.asarray(f1_[:10] if "s10" in mods else (f1_[:20] if "s20" in mods else seeds5), dtype=np.int64)
            reach_ = np.union1d(sd, Z3.expand3(sd, fam_all, fam_all, unc, False, False, None))
            in_exp = np.isin(miss, exp) if exp.size else np.zeros(miss.size, bool)
            in_reach = np.isin(miss, reach_)
            r_, c_, u_ = in_exp, (~in_exp) & in_reach, (~in_exp) & (~in_reach)
            lists = np.union1d(d[:1000], s_[:1000])
            tot["q_missing"] += 1
            tot["golds_missing"] += int(miss.size)
            tot["R"] += int(r_.sum())
            tot["C"] += int(c_.sum())
            tot["U"] += int(u_.sum())
            tot["U_in_lists"] += int(np.isin(miss[u_], lists).sum())
            tot["q_R"] += int(r_.any())
            tot["q_C"] += int(c_.any())
            tot["q_U"] += int(u_.any())
            if c_.any():
                scorer = Z3.Scorer(src.dvec, {"q": np.asarray(src.qvec(q, q + 1)[0], dtype=np.float64)}, src.n)
                trace = trace_walk5(np, Z5, sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], setting, mods, scorer)
                scorer.release()
                if not np.array_equal(np.concatenate([t["kept"] for t in trace]) if trace else np.empty(0, dtype=np.int64), exp):
                    mismatch += 1
                first_budget_hop = 1 if setting.get("bf") else 2             # hop 1 has a visited budget only in a budget-following walk
                for gold in miss[c_]:
                    gold = int(gold)
                    budget_h = next((t["h"] for t in trace if t["h"] >= first_budget_hop and gold in t["capped"] and gold not in t["kept"]), None)
                    cap_h = None
                    if budget_h is None:
                        for t in trace:
                            if gold not in t["capped"] and gold in uncapped5(np, t["frontier"], t["fams"]):
                                cap_h = t["h"]
                                break
                    key = f"budget{budget_h}" if budget_h else (f"cap{cap_h}" if cap_h else "upstream")
                    stage[key] = stage.get(key, 0) + 1
        out = {"stage": ("PL7" if pl7 else ("PL6" if out_root != PL5 else "PL5")) + " D1 + D2 (post-hoc diagnostic, labels read, decides nothing)", "declared_in": doc, "dataset": ds, "cell": name, "questions": int(idx.size),
               "carve": "dev", "first": "frozen", "nodes_mean": float(np.mean(nodes)), "bound": Z5.BOUND, "totals": tot,
               "D2": {"by_stage_hop": stage, "c_golds": tot["C"], "trace_matches_engine": mismatch == 0, "trace_mismatch_questions": mismatch},
               "uncapped_setting": "3 hops, no per-seed, per-frontier or visited cap, FULL families, the cell's own seeds", "seconds": round(Z5.time.time() - t0, 1),
               "code_path_sha256": Z5.code_path_sha256(), "utc": Z5.utc()}
        Z5.wjson(out_root / ds / "miss.json", out)
        Z5.log(f"D1 {ds} {name}: {idx.size} q, {tot['q_missing']} miss a gold ({tot['golds_missing']} golds): R {tot['R']}, C {tot['C']}, U {tot['U']} (in lists {tot['U_in_lists']}) "
               f"at {np.mean(nodes):.0f} nodes; D2 {stage} (trace = engine: {mismatch == 0})")
    return 0


def cmd_confirm_par(a):
    """zpl5g.py confirm, with the eval questions cut into K slices run in K processes (the same pools, the same arrays, the same records: the slices are concatenated before
    anything is graded into a record). Used where the single-process confirm would take hours (SQuAD's 11,873 eval questions). --verify compares with an existing confirm grid."""
    for _p in (ROOT / "scripts", ROOT / "src", HERE):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import subprocess
    import numpy as np
    import zpl5g as Z5
    P2 = Z5.P2
    P2.HOST = bool(a.host)
    Z5.gates_passed5(a.ds)
    sel = Z5.rjson(PL5 / a.ds / "select.json")
    names = Z5.confirm_names5(sel)
    plan = Z5.Plan5(names)
    src = P2.load_source(a.ds, "eval", a.first)
    nq, k = src.nq, int(a.slices)
    tmp = PL5 / a.ds / f"_confirm_par_{a.first}"
    if a.worker is not None:
        i = int(a.worker)
        op = P2.opened()
        if plan.need_vec:
            src.dvec(np.asarray([0]))
        lo, hi = i * nq // k, (i + 1) * nq // k
        Z5.log(f"{a.ds}/eval/{a.first}: slice {i} of {k}, rows {lo}..{hi}")
        P, S = Z5.evaluate5(op, src, plan, lo, hi)
        tmp.mkdir(parents=True, exist_ok=True)
        np.savez(tmp / f"slice{i}of{k}.npz", P=P, S=S, lo=lo, hi=hi)
        return 0
    t0 = Z5.time.time()
    tmp.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(k):
        cmd = [sys.executable, str(Path(__file__).resolve()), "confirm-par", "--ds", a.ds, "--first", a.first, "--slices", str(k), "--worker", str(i)]
        if a.host:
            cmd.append("--host")
        procs.append(subprocess.Popen(cmd, stdout=open(tmp / f"slice{i}.log", "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    if any(rcs):
        Z5.stop(f"{a.ds}/{a.first}: a slice failed ({rcs}); see {tmp}")
    P = np.concatenate([np.load(tmp / f"slice{i}of{k}.npz")["P"] for i in range(k)], axis=0)
    S = np.concatenate([np.load(tmp / f"slice{i}of{k}.npz")["S"] for i in range(k)], axis=0)
    base = Z5.grid_path5(a.ds, "eval", a.first, "confirm")
    if a.verify:
        g0 = Z5.load_grid5(base)
        same = bool(np.array_equal(g0["present"], P) and np.array_equal(g0["size"], S) and list(g0["names"]) == list(plan.names))
        Z5.log(f"{a.ds}/{a.first}: sliced run equals the filed confirm grid: {same}")
        return 0 if same else 3
    Z5.save_grid5(base, src, plan, P, S, 0, src.nq, Z5.time.time() - t0, stage="PL5 confirm grid")
    g = {"names": plan.names, "present": P, "size": S, "gold_counts": src.gold_counts}
    rows = {r["pool"]: r for r in Z5.rows_of_grid5(g, a.ds, "eval")}
    res = {"stage": "PL5 confirm", "declared_in": Z5.DOC, "dataset": a.ds, "first": a.first, "questions": src.nq, "utc": Z5.utc(), "slices": k, "cells": {}}
    for n in plan.names:
        r = rows[n]
        kk = int(round(r["all_gold_at_pool"] * src.nq))
        res["cells"][n] = {"all": r["all_gold_at_pool"], "all_ci": P2.wilson(kk, src.nq), "any": r["any_gold_at_pool"], "coverage": r["coverage_micro"],
                           "nodes": r["candidates_mean"], "p95": r["candidates_p95"], "max": r["candidates_max"], "over_bound": bool(r["candidates_mean"] > Z5.BOUND)}
    res["named"] = {}
    for key, c in sel["picks"].items():
        if key.split("_", 1)[1] in Z5.READINGS and c["name"] in res["cells"]:
            e = res["cells"][c["name"]]
            res["named"][key] = {"name": c["name"], "dev_all": c["all"], "dev_nodes": c["nodes"], "eval_all": e["all"], "eval_nodes": e["nodes"], "floor": c["floor"],
                                 "confirmed": bool(e["all"] >= c["floor"]), "short_by": max(0.0, c["floor"] - e["all"]), "dev_to_eval": e["all"] - c["all"],
                                 "over_bound": e["over_bound"]}
    e = res["cells"][sel["old"]]
    res["old"] = {"name": sel["old"], "eval_all": e["all"], "eval_nodes": e["nodes"]}
    Z5.wjson(PL5 / a.ds / f"confirm_{a.first}.json", res)
    for key, v in res["named"].items():
        Z5.log(f"{a.ds}/{a.first} {key}: {v['name']} eval ALL {v['eval_all']:.4f} at {v['eval_nodes']:.1f} nodes "
               f"({'CONFIRMED' if v['confirmed'] else 'SHORT by %.2f pts' % (100 * v['short_by'])}; dev {v['dev_all']:.4f})" + ("  OVER_BOUND" if v["over_bound"] else ""))
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    return 0


TAGS5 = ("bf", "vr", "fu", "rr", "seeds", "u1", "wide")
TAGNAME5 = {"bf": "budget-following cap (any key)", "vr": "cosine order (bf)", "fu": "fused order (bf)", "rr": "families take turns (bf)", "seeds": "seeds 10/20",
            "u1": "one-hop whole-window walks", "wide": "4,096-entry window"}
ANY5 = (3000, 4000, 5000, 6000, 7000)


def cmd_report(a):
    """outputs/pl5/report.md and report.json from the filed records only (zpl4r.cmd_report's layout, PL5's readings: all / prev / new)."""
    import zpl4r as R4
    rj, f4, f1, pts, cell, short_cell = R4.rj, R4.f4, R4.f1, R4.pts, R4.cell, R4.short_cell
    ORDER, BOUND, BUDGETS = R4.ORDER, R4.BOUND, R4.BUDGETS
    rep, md = {}, []
    sel, conf = {}, {}
    for ds in ORDER:
        sp = PL5 / ds / "select.json"
        if sp.is_file():
            sel[ds] = rj(sp)
            conf[ds] = {fs: rj(PL5 / ds / f"confirm_{fs}.json") for fs in ("frozen", "e3v") if (PL5 / ds / f"confirm_{fs}.json").is_file()}
    md.append("### Gates and cost\n")
    md.append("| dataset | G-R (PL2's, PL3's and PL4's cells rebuilt: first 400 dev questions, PL4's 150) | dev grid: parts, seconds (sum), peak GB (max) |")
    md.append("|---|---|---|")
    for ds in ORDER:
        gp, hp = PL5 / f"gr_{ds}.json", PL5 / f"gr_{ds}_host.json"
        g = rj(gp) if gp.is_file() else (rj(hp) if hp.is_file() else None)
        gtxt = (f"{g['verdict']} ({g['questions']} q, {g['elements']:,} elements, mismatches {g['present_mismatches']}/{g['size_mismatches']})" if g else "not run")
        if ds in sel:
            hs = sel[ds]["grid_heads"]
            ctxt = f"{len(hs)}, {sum(h['seconds'] for h in hs):,.0f} s, {max(h['peak_gb'] for h in hs)} GB"
        else:
            ctxt = "NOT READ"
        md.append(f"| {ds} | {gtxt} | {ctxt} |")
    sp = PL5 / "selftest.json"
    if sp.is_file():
        s = rj(sp)
        md.append(f"\nSelf-test: {s['verdict']} (cells {s['cells']['verdict']}, G-F {s['G-F']['verdict']}, G-D {s['G-D']['verdict']}; engine "
                  f"{', '.join(e['dataset'] + ' ' + e['verdict'] + ' ' + str(e['checks']) for e in s['engine'])}; G-X5 {', '.join(x['verdict'] for x in s['G-X5'])}); "
                  f"code-path sha256 `{s['code_path_sha256'][:8]}...`.\n")
    md.append("### Dev reading (the rule's own numbers, dev carve, frozen lists): PL4's chosen cell (the incumbent, reading `prev`) against the grid of all 6,078 cells\n")
    md.append("| dataset | dev q | PL4 R95 (prev) | R95 (all cells) | PL4 R98 | R98 (all cells) | R95 of the PL5 cells only | prev pick re-read = PL4's filed pick |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            md.append(f"| {ds} | NOT READ | | | | | | |")
            continue
        s, p = sel[ds], sel[ds]["picks"]
        md.append(f"| {ds} | {s['questions']} | {cell(p['R95_prev'])} | {cell(p['R95_all'])} | {cell(p['R98_prev'])} | {cell(p['R98_all'])} | {cell(p['R95_new'])} | "
                  f"{s['prev_pick_equals_pl4_filed']} |")
        rep[ds] = {"dev_questions": s["questions"], "picks": {k: {x: v[x] for x in ("name", "all", "any", "nodes", "reached")} for k, v in p.items() if ":" not in k}}
    md.append("")
    for floor in ("95", "98"):
        md.append(f"### The chosen cell at 0.{floor} (R{floor}) and its confirmation on eval: PL4's against the grid of all cells (and the PL5 cells alone)\n")
        md.append("| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |")
        md.append("|---|---|---|---|---|---|")
        for ds in ORDER:
            if ds not in sel:
                continue
            for rd, lab in (("prev", "PL4"), ("all", "all"), ("new", "PL5 only")):
                c = sel[ds]["picks"].get(f"R{floor}_{rd}")
                if not c:
                    continue
                row = [f"| {ds} {lab} | `{c['name']}`{'' if c['reached'] else ' (SHORT)'} | {f4(c['all'])} | {f1(c['nodes'])}"]
                for fs in ("frozen", "e3v"):
                    cf = conf[ds].get(fs)
                    n = cf["named"].get(f"R{floor}_{rd}") if cf else None
                    row.append(f"{f4(n['eval_all'])} ({f1(n['eval_nodes'])}){'' if n['confirmed'] else ' SHORT'}{' OVER_BOUND' if n.get('over_bound') else ''}" if n else "not read")
                md.append(" | ".join(row) + " |")
        md.append("")
    md.append("### The filed own construction on eval (PL2's OLD cell, re-read by this engine)\n")
    md.append("| dataset | OLD | eval ALL frozen (nodes) | eval ALL e3v (nodes) |")
    md.append("|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        row = [f"| {ds} | `{sel[ds]['old']}`"]
        for fs in ("frozen", "e3v"):
            cf = conf[ds].get(fs)
            row.append(f"{f4(cf['old']['eval_all'])} ({f1(cf['old']['eval_nodes'])})" if cf else "not read")
        md.append(" | ".join(row) + " |")
    md.append("")
    md.append("### The price of the floor at any size, dev (the bound is not applied): the smallest cell with ALL >= 0.95 and >= 0.98, PL4's grid / all cells\n")
    md.append("| dataset | 0.95: PL4's grid | 0.95: all cells | 0.98: PL4's grid | 0.98: all cells |")
    md.append("|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        p = sel[ds]["picks"]
        cs = []
        for fl in ("95", "98"):
            for rd in ("prev", "all"):
                c = p[f"R{fl}any_{rd}"]
                cs.append(f"`{c['name']}` {f4(c['all'])} at {f1(c['nodes'])}" if c["reached"] else f"none (best `{c['name']}` {f4(c['all'])} at {f1(c['nodes'])})")
        md.append(f"| {ds} | {cs[0]} | {cs[1]} | {cs[2]} | {cs[3]} |")
        rep.setdefault(ds, {})["price_any_size"] = {"prev_95": cs[0], "all_95": cs[1], "prev_98": cs[2], "all_98": cs[3]}
    md.append("")
    md.append("### The highest ALL on dev (the bound is 2,500 mean nodes): PL4's grid / all cells, within the bound and at any size\n")
    md.append("| dataset | PL4's grid, within the bound | all cells, within the bound | PL4's grid, any size | all cells, any size |")
    md.append("|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        tb = sel[ds]["table"]
        old = [x for x in tb if x["family"] != "pl5"]
        cs = []
        for bd in (BOUND, None):
            for rows in (old, tb):
                b = R4.best_all(rows, bd)
                cs.append(f"`{b['name']}` {f4(b['all'])} at {f1(b['nodes'])}")
        md.append(f"| {ds} | {cs[0]} | {cs[1]} | {cs[2]} | {cs[3]} |")
    md.append("")
    md.append("### What the PL5 cells bought (dev frontier of all cells minus the frontier of PL2-PL4's cells, at equal mean nodes; ALL)\n")
    grid_n = (25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500) + ANY5
    md.append("| dataset | " + " | ".join(str(n) for n in grid_n) + " |")
    md.append("|---|" + "---|" * len(grid_n))
    for ds in ORDER:
        if ds not in sel:
            continue
        b = {x["nodes"]: x for x in sel[ds]["bought"]}
        cells_ = []
        for n in grid_n:
            x = b.get(n)
            cells_.append("-" if not x or x["all"] is None else f"{f4(x['prev'])} / {f4(x['all'])} ({pts(x['gain'])})")
        md.append(f"| {ds} | " + " | ".join(cells_) + " |")
        rep.setdefault(ds, {})["bought"] = sel[ds]["bought"]
    md.append("\n(each entry: the frontier of PL2-PL4's cells / the frontier of all cells (points bought); `-` = no cell is that small; 3,000 and above are beside the bound)\n")
    md.append("### What each option bought: R95 of the option's own cells (dev, inside the bound) beside PL4's R95, and its best ALL inside the bound\n")
    md.append("| dataset | PL4 R95 | " + " | ".join(TAGNAME5[t] for t in TAGS5) + " |")
    md.append("|---|---|" + "---|" * len(TAGS5))
    for ds in ORDER:
        if ds not in sel:
            continue
        p = sel[ds]["picks"]
        row = [f"| {ds} | {short_cell(p['R95_prev'])}"]
        for t in TAGS5:
            c = p.get(f"R95_only:{t}")
            row.append(short_cell(c) if c else "-")
        md.append(" | ".join(row) + " |")
    md.append("")
    md.append("### The ceiling the filed reach blocks set (M3A's populations: the share of questions missing a gold from the union pool, and where the missing golds sit)\n")
    md.append("| dataset | questions | with a gold missing | STRUCT within 1/2/3 hops | NER within 1/2/3 | KNN within 1/2/3 | FULL within 1/2/3 | questions whose missing golds all lie within 3 (FULL) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ds in ORDER:
        r = R4.reach(ds)
        if not r:
            continue
        g = lambda k: f"{r[k]['w1']:.3f} / {r[k]['w2']:.3f} / {r[k]['w3']:.3f}"  # noqa: E731
        md.append(f"| {ds} | {r['FULL']['questions']} | {r['FULL']['with_missing']} | {g('STRUCT')} | {g('NER')} | {g('KNN')} | {g('FULL')} | {r['FULL']['all3']} of {r['FULL']['with_missing']} |")
        rep.setdefault(ds, {})["reach"] = r
    md.append("")
    cp = PL5 / "call.json"
    if cp.is_file():
        c = rj(cp)
        md.append(f"### The call, the old rule as PL3 and PL4 state it (R95 of all cells against R95 of PL2-PL4's cells): {c['call']} (read {len(c['datasets_read'])} of 7: "
                  f"{', '.join(c['datasets_read'])}; BELOW {c['counts']['old']['BELOW']}, ABOVE {c['counts']['old']['ABOVE']}, LEVEL {c['counts']['old']['LEVEL']})\n")
        md.append("| dataset | status | NEW (R95, all cells) | NEW ALL (nodes) | OLD (PL4 R95) | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |")
        md.append("|---|---|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                md.append(f"| {ds} | NOT READ | | | | | | |")
                continue
            md.append(f"| {ds} | {v['status']}{' (same cell)' if v['same_cell'] else ''} | `{v['new']}` | {f4(v['new_all'])} ({f1(v['new_nodes'])}) | `{v['old']}` | "
                      f"{f4(v['old_all'])} ({f1(v['old_nodes'])}) | {pts(v['all_diff'])} [{pts(v['all_diff_ci'][0])}, {pts(v['all_diff_ci'][1])}] | "
                      f"{v['nodes_ratio']:.3f} [{v['nodes_ratio_ci'][0]:.3f}, {v['nodes_ratio_ci'][1]:.3f}] |")
        md.append("\nBeside the call (decide nothing): R98 and the PL5-cells-only pick against PL4's R95 / R98.\n")
        md.append("| dataset | R98 status | R98 NEW ALL (nodes) vs OLD | PL5-only status | PL5-only cell | ALL (nodes) |")
        md.append("|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                continue
            r98, no = v["r98"], v["new_only"]
            md.append(f"| {ds} | {r98['status']} | {f4(r98['new_all'])} ({f1(r98['new_nodes'])}) vs {f4(r98['old_all'])} ({f1(r98['old_nodes'])}) | {no['status']} | "
                      f"`{no['new']}` | {f4(no['new_all'])} ({f1(no['new_nodes'])}) |")
        md.append("")
        md.append(f"### The call, the frontier reading (matched node counts inside the bound, declared before any number): {c['call_frontier']} "
                  f"(BELOW {c['counts']['frontier']['BELOW']}, ABOVE {c['counts']['frontier']['ABOVE']}, LEVEL {c['counts']['frontier']['LEVEL']})\n")
        md.append("Per dataset and budget N: the cell of all cells with the highest dev ALL at most N dev nodes against the same for PL2-PL4's cells; eval, frozen lists; paired bootstrap of the ALL difference.\n")
        md.append("| dataset | frontier status | N | NEW cell | OLD cell | NEW ALL (nodes) | OLD ALL (nodes) | ALL diff, 95% CI | status |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                md.append(f"| {ds} | NOT READ | | | | | | | |")
                continue
            for n in BUDGETS:
                fr = v["frontier"].get(str(n))
                if not fr or fr["status"] == "NONE":
                    md.append(f"| {ds} | {v['frontier_status']} | {n} | none | | | | | |")
                    continue
                md.append(f"| {ds} | {v['frontier_status']} | {n} | `{fr['new']}`{' (same cell)' if fr['same_cell'] else ''} | `{fr['old']}` | {f4(fr['new_all'])} ({f1(fr['new_nodes'])}) | "
                          f"{f4(fr['old_all'])} ({f1(fr['old_nodes'])}) | {pts(fr['all_diff'])} [{pts(fr['all_diff_ci'][0])}, {pts(fr['all_diff_ci'][1])}] | {fr['status']} |")
        md.append("")
        md.append(f"### Beside the calls: the price of the floor at any size (eval; the bound does not apply; decides nothing). Counts for R95any: {c['counts']['price_R95any']}; "
                  f"the any-size frontier (3,000-7,000): {c['counts']['frontier_any']}\n")
        md.append("| dataset | floor | status | NEW cell | NEW ALL (nodes) | OLD cell | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                continue
            for tag, lab in (("R95any", "0.95"), ("R98any", "0.98")):
                x = v["price"][tag]
                md.append(f"| {ds} | {lab} | {x['status']}{' (same cell)' if x['same_cell'] else ''} | `{x['new']}` | {f4(x['new_all'])} ({f1(x['new_nodes'])}) | `{x['old']}` | "
                          f"{f4(x['old_all'])} ({f1(x['old_nodes'])}) | {pts(x['all_diff'])} [{pts(x['all_diff_ci'][0])}, {pts(x['all_diff_ci'][1])}] | "
                          f"{x['nodes_ratio']:.3f} [{x['nodes_ratio_ci'][0]:.3f}, {x['nodes_ratio_ci'][1]:.3f}] |")
        md.append("")
        md.append("| dataset | any-size frontier status | N | NEW cell | OLD cell | NEW ALL (nodes) | OLD ALL (nodes) | ALL diff, 95% CI | status |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                continue
            for n in ANY5:
                fr = v["frontier_any"].get(str(n))
                if not fr or fr["status"] == "NONE":
                    md.append(f"| {ds} | {v['frontier_any_status']} | {n} | none | | | | | |")
                    continue
                md.append(f"| {ds} | {v['frontier_any_status']} | {n} | `{fr['new']}`{' (same cell)' if fr['same_cell'] else ''} | `{fr['old']}` | {f4(fr['new_all'])} ({f1(fr['new_nodes'])}) | "
                          f"{f4(fr['old_all'])} ({f1(fr['old_nodes'])}) | {pts(fr['all_diff'])} [{pts(fr['all_diff_ci'][0])}, {pts(fr['all_diff_ci'][1])}] | {fr['status']} |")
        md.append("")
        rep["call"] = {"call": c["call"], "call_frontier": c["call_frontier"], "datasets_read": c["datasets_read"], "counts": c["counts"],
                       "statuses": {k: v["status"] for k, v in c["datasets"].items()}, "frontier_statuses": {k: v.get("frontier_status") for k, v in c["datasets"].items()}}
    md.append("### The 2,500 bound on e3v lists (selection reads the frozen dev lists; this is the same cell re-run on the eval carve's e3v lists)\n")
    md.append("| dataset | cell (R95, all cells) | nodes: dev frozen | eval frozen | eval e3v | e3v over the bound |")
    md.append("|---|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        c = sel[ds]["picks"]["R95_all"]
        nf = conf[ds].get("frozen", {}).get("named", {}).get("R95_all")
        ne = conf[ds].get("e3v", {}).get("named", {}).get("R95_all")
        md.append(f"| {ds} | `{c['name']}` | {f1(c['nodes'])} | {f1(nf['eval_nodes']) if nf else 'not read'} | {f1(ne['eval_nodes']) if ne else 'not read'} | "
                  f"{('OVER_BOUND' if ne['over_bound'] else 'within') if ne else '-'} |")
    md.append("")
    mrows = [(ds, rj(PL5 / ds / "miss.json")) for ds in ORDER if (PL5 / ds / "miss.json").is_file()]
    if mrows:
        md.append("### D1: what remains unreached at the chosen cell of a SHORT dataset (post-hoc; labels read; n dev questions evenly spaced over the carve)\n")
        md.append("| dataset | chosen cell | n | questions missing a gold | missing golds | per question | R (ranked out) | C (cut by the caps or the budget) | U (unreached) | of U: in the top-1000 lists | nodes |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ds, m in mrows:
            t = m["totals"]
            md.append(f"| {ds} | `{m['cell']}` | {m['questions']} | {t['q_missing']} ({t['q_missing'] / m['questions']:.3f}) | {t['golds_missing']} | {t['golds_missing'] / max(1, t['q_missing']):.2f} | "
                      f"{t['R']} (q {t['q_R']}) | {t['C']} (q {t['q_C']}) | {t['U']} (q {t['q_U']}) | {t['U_in_lists']} | {f1(m['nodes_mean'])} |")
        md.append("")
        md.append("### D2: which stage cut the C golds (post-hoc; labels read; the cell's walk traced hop by hop)\n")
        md.append("| dataset | cell | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 1 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ds, m in mrows:
            d2 = m.get("D2")
            if not d2:
                continue
            h = d2["by_stage_hop"]
            md.append(f"| {ds} | `{m['cell']}` | {m['totals']['C']} | {h.get('cap1', 0)} | {h.get('cap2', 0)} | {h.get('cap3', 0)} | {h.get('budget1', 0)} | {h.get('budget2', 0)} | "
                      f"{h.get('budget3', 0)} | {h.get('upstream', 0)} | {d2['trace_matches_engine']} |")
        md.append("")
        rep["D1"] = {ds: m["totals"] for ds, m in mrows}
        rep["D2"] = {ds: m.get("D2") for ds, m in mrows}
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    (PL5 / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    print("\n".join(md))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("place")
    p.add_argument("--ds", nargs="+", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--host", action="store_true")
    p = sub.add_parser("miss")
    p.add_argument("--ds", nargs="+", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--all", action="store_true")
    p.add_argument("--host", action="store_true")
    p.add_argument("--pl6", action="store_true", help="diagnose the cell rule U names in outputs/pl6/<d>/select.json (R95_all) and write outputs/pl6/<d>/miss.json")
    p.add_argument("--pl7", action="store_true", help="diagnose the cell R7(2,500) names in outputs/pl7/<d>/select.json (picks['2500']['R95_all']) and write outputs/pl7/<d>/miss.json")
    p = sub.add_parser("confirm-par")
    p.add_argument("--ds", required=True)
    p.add_argument("--first", default="frozen", choices=("frozen", "e3v"))
    p.add_argument("--slices", type=int, default=6)
    p.add_argument("--worker", default=None)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--host", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(PL5 / "report.md"))
    a = ap.parse_args(argv)
    return {"place": cmd_place, "miss": cmd_miss, "confirm-par": cmd_confirm_par, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
