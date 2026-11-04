"""U2z (docs/U2Z_NEIGHBOURHOOD_BUDGET.md): pool depth from the question's own neighbourhood size, on U2s's filed s1sel
arrays of musique, hotpotqa and 2wiki. Pools only: nothing trains, nothing is fit, no walk runs; the new candidates are
budgets read from the filed n_n1 counts (the size of the base's one-link neighbourhood), no gold and no per-dataset
constant.

    python outputs/mp_unified/zu2z.py dev
    python outputs/mp_unified/zu2z.py choose [--cache DIR]
    python outputs/mp_unified/zu2z.py --selftest

dev      the new tables on musique's arrays for one construction, each checked against U2e's evaluate; prints only OK
         and the seconds (no coverage value).
choose   U2w's candidates (U2s's universal ones and the own construction's) plus the 120,960 new ones, on U2s's filed
         s1sel stats of the three datasets, with U2w's checks (pick O's filed point; U2s's filed frontier C) and the
         new tables' check; frontiers C, E and O over P1f's weights restricted to the three; U2w's section 4 verdicts
         and what is carried. Writes outputs/u2z/choice.json and choice.md. --cache DIR keeps U2s's tables between runs
         (keyed by the stats file's sha256).
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zu2s as S  # noqa: E402  (U2s's arrays, tables, frontiers and verdict pieces, unchanged)
import zu2w as W  # noqa: E402  (U2w's choice pieces and verdicts, unchanged)

M, Q, P, F, PG, U2E, P1H = S.M, S.Q, S.P, S.F, S.PG, S.U2E, S.P1H
utc, write_json, sha_file = S.utc, S.write_json, S.sha_file

DECLARED = "docs/U2Z_NEIGHBOURHOOD_BUDGET.md"
OUT = ROOT / "outputs" / "u2z"
THREE, PICK_O, CAP, ROW_PRICE = W.THREE, W.PICK_O, W.CAP, W.ROW_PRICE
OTHERS = ("metaqa", "squad", "webqsp")
LAMS = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)      # steps of about sqrt(2)
K0S = (0, 50, 300)
FORMS_Z = ("NBRQ", "NBRC")
NCELL = len(S.GRID)
LEVELS_LOW = tuple(round(0.80 + 0.005 * i, 3) for i in range(41))      # beside: 0.80 to 1.00


def log(msg):
    print(f"[u2z {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha256():
    return sha_file(Path(__file__))


def nbr_budgets():
    """[(name, kind, (lambda, k0))]: NBRQ(l, k0) for every l and k0, then NBRC(l) for every l (44 in all)."""
    return ([(f"nbq_l{lam:g}_k{k0}", "NBRQ", (lam, k0)) for lam in LAMS for k0 in K0S] +
            [(f"nbc_l{lam:g}", "NBRC", (lam, 0)) for lam in LAMS])


NBZ = len(nbr_budgets())


def nbr_k(kind, prm, s):
    """k_q per question from s, the base's one-link neighbourhood size per question: NBRQ max(k0, round(l x s_q)); NBRC
    round(l x the corpus's mean s) for every question."""
    s = np.asarray(s, np.int64)
    lam, k0 = prm
    if kind == "NBRQ":
        return np.maximum(int(k0), np.rint(float(lam) * s).astype(np.int64))
    return np.full(s.size, int(np.rint(float(lam) * float(s.mean()))), np.int64)


def table_nbr(a, ks, cell_base):
    """(ALL, rows) of every new candidate of cells ks x recipes x orders on one dataset's arrays, flat in (cell, recipe,
    order, budget, tau from 1 down) order, with U2s's column_table."""
    ptr = np.asarray(a["ptr"], np.int64)
    bud = nbr_budgets()
    AL = np.empty((len(ks), S.R, S.NO, NBZ, S.NT))
    RW = np.empty_like(AL)
    for ki, k in enumerate(ks):
        b = np.asarray(a["bs"][:, k], np.int64)
        s = np.asarray(a["n_n1"][:, cell_base[k]], np.int64)
        KQ = np.stack([nbr_k(kind, prm, s) for _n, kind, prm in bud])
        for r in range(S.R):
            for o in range(S.NO):
                c = M.column(k, r, o)
                AL[ki, r, o], RW[ki, r, o] = S.column_table(b, a["lens"][:, c], a["pos"][:, c], KQ, ptr)
    return AL.ravel(), RW.ravel()


def decode_z(i, nk=NCELL):
    return tuple(int(x) for x in np.unravel_index(int(i), (nk, S.R, S.NO, NBZ, S.NT)))


def name_z(i):
    k, r, o, b, t = decode_z(i)
    return M.candidate_name(S.GRID[k][0], r, o, nbr_budgets()[b][0], S.TAUS_DESC[t])


def eval_one(a, k, r, o, kq, tau):
    """(ALL, mean rows) of one candidate by U2e's evaluate (the path S.held_of uses), rounded as column_table rounds."""
    c = M.column(k, r, o)
    sub = {"bs": a["bs"][:, k], "lens": a["lens"][:, [c]], "pos": a["pos"][:, [c]], "ptr": a["ptr"]}
    rows, held = U2E.evaluate(sub, 0, kq, tau)
    ptr = np.asarray(a["ptr"], np.int64)
    ng, nq = np.diff(ptr), ptr.size - 1
    cs = np.concatenate([[0], np.cumsum(held, dtype=np.int64)])
    hits = cs[ptr[1:]] - cs[ptr[:-1]]
    return round(float(((ng > 0) & (hits == ng)).sum()) / nq, 4), round(float(rows.sum()) / nq, 1)


def check_tables(a, cell_base, ks, AL, RW, rng, n=8):
    """n random new candidates of cells ks (the table's cell order), each recomputed by eval_one: equal to the table."""
    bud = nbr_budgets()
    for _ in range(n):
        ki, r, o, bi, ti = (int(rng.integers(x)) for x in (len(ks), S.R, S.NO, NBZ, S.NT))
        k = ks[ki]
        _bn, kind, prm = bud[bi]
        kq = nbr_k(kind, prm, a["n_n1"][:, cell_base[k]])
        got = eval_one(a, k, r, o, kq, S.TAUS_DESC[ti])
        i = np.ravel_multi_index((ki, r, o, bi, ti), (len(ks), S.R, S.NO, NBZ, S.NT))
        if got != (float(AL[i]), float(RW[i])):
            raise SystemExit(f"new table at cell {k}, recipe {r}, order {o}, budget {bud[bi][0]}, tau {ti}: "
                             f"{(float(AL[i]), float(RW[i]))} is not evaluate's {got}")


def dev():
    t0 = time.time()
    a, rec = S.load_u2s("musique")
    cb = rec["cell_base"]
    ks = [2]
    AL, RW = table_nbr(a, ks, cb)
    check_tables(a, cb, ks, AL, RW, np.random.default_rng(0), n=12)
    print(f"dev: OK, {AL.size} candidates of cell {rec['cells'][2]} in {time.time() - t0:.1f}s", flush=True)


# ── the choice (the laptop) ──────────────────────────────────────────────────


def _cached(cache, tag, sha, make):
    if cache is None:
        return make()
    p = Path(cache) / f"{tag}.npz"
    if p.exists():
        with np.load(p) as z:
            if str(z["sha"]) == sha:
                return z["AL"], z["RW"]
    AL, RW = make()
    Path(cache).mkdir(parents=True, exist_ok=True)
    np.savez(p, sha=sha, AL=AL, RW=RW)
    return AL, RW


def frontier_curve(A, costs, cost_r, ceil, idx, levels, point):
    """[(level, point record)] of the cheapest candidate in idx meeting every target at each level (past the cap too)."""
    out = []
    for t, tg, k in U2E.frontier(A[idx], costs[idx], ceil, levels=levels):
        out.append(point(t, tg, None if k is None else int(idx[k])))
    return out


def choose(cache=None):
    t0 = time.time()
    cw = Q.carried_walk()
    am, rm = {}, {}
    for d in THREE:
        x = S.load_u2s(d)
        if x is None:
            raise SystemExit(f"INCOMPLETE: no U2s stats for {d}")
        am[d], rm[d] = x
        r = rm[d]
        if (r["pick"], r["points"], r["G"], r["beta"]) != (cw["pick"], list(cw["names"]), cw["G"], cw["beta"]) or \
                r["verdict_p_sha256"] != sha_file(P.OUT / "verdict.json"):
            raise SystemExit(f"{d}: U2s's stats are not on the walk U2p's filed verdict carries")
        if not np.array_equal(am[d]["ptr"], M.load_u2m(d)[0]["ptr"]):
            raise SystemExit(f"{d}: U2s's questions and golds are not U2m's")
    vs_path = S.OUT / "verdict.json"
    vs = json.loads(vs_path.read_text(encoding="utf-8"))
    w = F.weights()
    if {d: w[d] for d in THREE} != {d: vs["weights"][d] for d in THREE}:
        raise SystemExit("P1f's weights are not U2s's filed weights on the three")
    bud, budz = S.budgets(), nbr_budgets()
    cb = {d: rm[d]["cell_base"] for d in THREE}
    tabs, own = {}, {}
    for d in THREE:
        sha = rm[d]["file_sha256"]
        tabs[d] = _cached(cache, f"u2s_universal_{d}", sha,
                          lambda d=d: S.table_arrays(am[d], range(NCELL), range(S.R), cb[d]))
        own[d] = _cached(cache, f"u2s_own_{d}", sha,
                         lambda d=d: S.table_arrays(am[d], [rm[d]["own_cell"]], range(S.R), cb[d]))
    log(f"U2s's tables of {tabs[THREE[0]][0].size} universal and {own[THREE[0]][0].size} own candidates "
        f"({time.time() - t0:.0f}s)")

    # U2w's checks: pick O's filed point and U2s's filed frontier C, on each of the three
    i_inc = W.own_index(PICK_O)
    if W.name_o(i_inc, bud) != PICK_O:
        raise SystemExit(f"own index {i_inc} is {W.name_o(i_inc, bud)}, not {PICK_O}")
    fo = [p for p in vs["frontier_O"] if p.get("pick") == PICK_O and abs(p["level"] - 0.96) < 1e-9]
    if len(fo) != 1:
        raise SystemExit("U2s's frontier O has no single 0.96 point at pick O")
    for d in THREE:
        got = (float(own[d][0][i_inc]), float(own[d][1][i_inc]))
        if got != (float(fo[0]["s1sel"][d]["ALL"]), float(fo[0]["s1sel"][d]["rows"])):
            raise SystemExit(f"{d}: pick O gives {got}, not U2s's filed {fo[0]['s1sel'][d]}")
    n_c = 0
    for p in vs["frontier_C"]:
        if not p.get("pick"):
            continue
        i = W.index_u(p["pick"], bud)
        for d in THREE:
            got = (float(tabs[d][0][i]), float(tabs[d][1][i]))
            if got != (float(p["s1sel"][d]["ALL"]), float(p["s1sel"][d]["rows"])):
                raise SystemExit(f"{d}: U2s's frontier C point at {p['level']} gives {got}, not {p['s1sel'][d]}")
        n_c += 1
    log(f"checks: pick O's filed point and U2s's {n_c} filed frontier C points reproduced ({time.time() - t0:.0f}s)")

    # the new tables, with their own check
    tabz = {d: table_nbr(am[d], range(NCELL), cb[d]) for d in THREE}
    rng = np.random.default_rng(0)
    for d in THREE:
        check_tables(am[d], cb[d], list(range(NCELL)), tabz[d][0], tabz[d][1], rng, n=24)
    log(f"new tables of {tabz[THREE[0]][0].size} candidates, each dataset checked on 24 random ones against "
        f"U2e's evaluate ({time.time() - t0:.0f}s)")

    nu, nz = tabs[THREE[0]][0].size, tabz[THREE[0]][0].size
    Au = np.stack([tabs[d][0] for d in THREE], axis=1)
    Ru = np.stack([tabs[d][1] for d in THREE], axis=1)
    Az = np.stack([tabz[d][0] for d in THREE], axis=1)
    Rz = np.stack([tabz[d][1] for d in THREE], axis=1)
    A_, RW = np.concatenate([Au, Az]), np.concatenate([Ru, Rz])
    costs, cost_r = W.costs_of(RW, w)
    cw_ = json.loads((W.OUT / "choice.json").read_text(encoding="utf-8"))
    ceil = np.asarray([cw_["ceilings"][d] for d in THREE], np.float64)             # U2w's filed ceilings: the targets
    if not np.array_equal(Au.max(axis=0), ceil):
        raise SystemExit(f"U2s's ceilings {Au.max(axis=0).tolist()} are not U2w's filed {ceil.tolist()}")
    ceil_all = A_.max(axis=0)                  # beside: whether a new candidate reaches past them (the targets keep U2w's)
    _gk, gr_u, _go, gb_u, _gt = np.unravel_index(np.arange(nu), (NCELL, S.R, S.NO, S.NB, S.NT))
    _zk, gr_z, _zo, gb_z, _zt = np.unravel_index(np.arange(nz), (NCELL, S.R, S.NO, NBZ, S.NT))
    gr = np.concatenate([gr_u, gr_z])
    form_u = np.asarray([S.FORMS.index(x[1]) for x in bud])[gb_u]
    form_z = np.asarray([FORMS_Z.index(x[1]) for x in budz])[gb_z]
    A_o = np.stack([own[d][0] for d in THREE], axis=1)
    RW_o = np.stack([own[d][1] for d in THREE], axis=1)
    c_o, cr_o = W.costs_of(RW_o, w)
    no_ = S.R * S.NO * S.NB * S.NT
    _ok, orr, _oo, ob, _ot = np.unravel_index(np.arange(no_), (1, S.R, S.NO, S.NB, S.NT))
    l_inc = S.level_of(A_o[i_inc], ceil)
    if l_inc is None:
        raise SystemExit("pick O meets no level on the three")
    cost_inc = float(cr_o[i_inc])
    if (l_inc, cost_inc) != (cw_["incumbent"]["level"], cw_["incumbent"]["cost"]):
        raise SystemExit(f"pick O gives level {l_inc}, cost {cost_inc}, not U2w's filed {cw_['incumbent']}")
    form_o = np.asarray([S.FORMS.index(x[1]) for x in bud])[ob]
    log(f"ceilings {dict(zip(THREE, ceil.tolist()))}; pick O on the three: level {l_inc}, cost {cost_inc} "
        f"({time.time() - t0:.0f}s)")

    def name_j(j):
        return W.name_u(j, bud) if j < nu else name_z(j - nu)

    def form_j(j):
        return S.FORMS[int(form_u[j])] if j < nu else FORMS_Z[int(form_z[j - nu])]

    def point(t, tgt, j, own_=False):
        rec = {"level": t, "targets": dict(zip(THREE, np.asarray(tgt).tolist())), "pick": None}
        if j is None:
            return rec
        if own_:
            rec.update({"pick": W.name_o(j, bud), "cost": float(cr_o[j]), "within_cap": bool(cr_o[j] <= CAP + 1e-9),
                        "price": round(float(cr_o[j]) / cost_inc, 3), "form": S.FORMS[int(form_o[j])],
                        "s1sel": {d: {"ALL": float(A_o[j, i]), "rows": float(RW_o[j, i])} for i, d in enumerate(THREE)}})
            return rec
        rec.update({"pick": name_j(j), "cost": float(cost_r[j]), "within_cap": bool(cost_r[j] <= CAP + 1e-9),
                    "price": round(float(cost_r[j]) / cost_inc, 3), "form": form_j(j),
                    "s1sel": {d: {"ALL": float(A_[j, i]), "rows": float(RW[j, i])} for i, d in enumerate(THREE)}})
        return rec

    def frontiers(sel):
        """S.three_frontiers over the candidates in the index set sel (joint indices; the own table as always)."""
        s = np.asarray(sel)
        out = S.three_frontiers(A_[s], costs[s], cost_r[s], gr[s], A_o, c_o, cr_o, orr, ceil, CAP)
        return {k: ([(t, tg, None if i is None else int(s[i])) for t, tg, i in pts], lv,
                    None if kk is None else int(s[kk])) if k != "O" else (pts, lv, kk)
                for k, (pts, lv, kk) in out.items()}

    def fr_rec(tfx):
        return {key: {"level": lv, "points": [point(t, tg, k_, key == "O") for t, tg, k_ in pts],
                      "pick": None if kk is None else point(lv, W.point_at(pts, lv)[0], kk, key == "O")}
                for key, (pts, lv, kk) in tfx.items()}

    # the re-run check: U2s's universal candidates alone give U2w's filed frontiers C and E (every pick, level and cost)
    fr_u = fr_rec(frontiers(np.arange(nu)))
    for key in ("C", "E"):
        want = [(p.get("pick"), p.get("cost")) for p in cw_[f"frontier_{key}"]["points"]]
        got = [(p.get("pick"), p.get("cost")) for p in fr_u[key]["points"]]
        if got != want or fr_u[key]["level"] != cw_[f"frontier_{key}"]["level"]:
            raise SystemExit(f"U2s's candidates alone do not give U2w's filed frontier {key}")
    log(f"checks: U2s's candidates alone give U2w's filed frontiers C and E ({time.time() - t0:.0f}s)")

    tf = frontiers(np.arange(nu + nz))
    tz = frontiers(np.arange(nu, nu + nz))
    _tg, c_k = W.point_at(tf["C"][0], l_inc)
    _tg, e_k = W.point_at(tf["E"][0], l_inc)
    c_at = None if c_k is None else float(cost_r[c_k])
    e_at = None if e_k is None else float(cost_r[e_k])
    vc, ve, carried = W.decide(tf["C"][1], tf["E"][1], l_inc, cost_inc, c_at, e_at)
    log(f"C {tf['C'][1]}, E {tf['E'][1]} within the cap; at {l_inc}: C {c_at}, E {e_at}; {vc}, {ve}, carried {carried} "
        f"({time.time() - t0:.0f}s)")

    fr, frz = fr_rec(tf), fr_rec(tz)
    tgt_inc = np.round(np.minimum(l_inc, ceil - PG.TOL), 4)
    if carried in ("E", "C"):
        carried_rec = fr[carried]["pick"]
    elif carried in ("C_ROW", "E_ROW"):
        ck = "C" if carried == "C_ROW" else "E"
        carried_rec = point(l_inc, W.point_at(tf[ck][0], l_inc)[0], c_k if ck == "C" else e_k)
    else:
        carried_rec = None

    # beside, deciding nothing
    meets = (A_ >= tgt_inc[None, :] - 1e-9).all(axis=1)
    idx_c = np.flatnonzero(gr == 0)
    idx_e = np.flatnonzero(np.isin(gr, S.KIND_FREE))
    idx_cz, idx_ez = idx_c[idx_c >= nu], idx_e[idx_e >= nu]

    def cheapest(mask):
        i = np.flatnonzero(meets & mask)
        return None if i.size == 0 else point(l_inc, tgt_inc, int(i[np.argmin(costs[i])]))

    by_budget = {}
    for b_i, (bn, kind, _prm) in enumerate(budz):
        m = np.zeros(nu + nz, bool)
        m[nu:] = gb_z == b_i
        by_budget[bn] = cheapest(m)
    curves = {nm: [p for p in frontier_curve(A_, costs, cost_r, ceil, ix, LEVELS_LOW, point)]
              for nm, ix in (("joint_C", idx_c), ("joint_E", idx_e), ("new_C", idx_cz), ("new_E", idx_ez))}
    per_dataset = {}
    for j, d in enumerate(THREE):
        per_dataset[d] = {}
        for t in LEVELS_LOW[::2]:
            tg = round(min(t, float(ceil[j]) - PG.TOL), 4)
            ok = Az[:, j] >= tg - 1e-9
            per_dataset[d][str(t)] = None if not ok.any() else round(float(Rz[ok, j].min()), 1)
    mean_s = {d: {S.BASES[b][0]: round(float(am[d]["n_n1"][:, b].mean()), 1) for b in range(len(S.BASES))}
              for d in THREE}
    mean_i = {d: {rm[d]["cells"][k]: round(float(am[d]["inc"][:, k].mean()), 1) for k in range(len(rm[d]["cells"]))}
              for d in THREE}
    rec = {"declared_in": DECLARED, "stage": "U2z", "part": "A", "script_sha256": script_sha256(),
           "zu2s_sha256": S.script_sha256(), "zu2w_sha256": W.script_sha256(),
           "u2s_verdict_sha256": sha_file(vs_path), "stats_sha256": {d: rm[d]["file_sha256"] for d in THREE},
           "datasets": list(THREE), "weights": {d: w[d] for d in THREE}, "cap": CAP, "row_price": ROW_PRICE,
           "lambdas": list(LAMS), "k0s": list(K0S), "budgets": [b[0] for b in budz],
           "candidates": nu, "new_candidates": nz, "own_candidates": no_, "ceilings": dict(zip(THREE, ceil.tolist())),
           "ceilings_all_candidates": dict(zip(THREE, ceil_all.tolist())),
           "incumbent": {"name": PICK_O, "level": l_inc, "cost": cost_inc,
                         "s1sel": {d: {"ALL": float(A_o[i_inc, j]), "rows": float(RW_o[i_inc, j])}
                                   for j, d in enumerate(THREE)}},
           "verdict_c": vc, "verdict_e": ve, "carried": carried, "carried_pick": carried_rec,
           "at_incumbent_level": {"C": None if c_k is None else point(l_inc, tgt_inc, c_k),
                                  "E": None if e_k is None else point(l_inc, tgt_inc, e_k)},
           "frontier_C": fr["C"], "frontier_E": fr["E"], "frontier_O": fr["O"],
           "new_frontier_C": frz["C"], "new_frontier_E": frz["E"], "by_new_budget": by_budget,
           "curves_0.80_to_1.00": curves, "per_dataset_fewest_rows_new": per_dataset,
           "mean_n_n1": mean_s, "mean_I": mean_i,
           "checks": {"pick_o_filed_point": True, "u2s_frontier_c_points": n_c, "u2w_frontiers_c_and_e": True,
                      "walk_and_questions": True, "weights": True, "new_tables_vs_evaluate": 24 * len(THREE),
                      "ceilings_are_u2w": True, "incumbent_is_u2w": True},
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "choice.json", rec)
    (OUT / "choice.md").write_text(report_md(rec), encoding="utf-8")
    log(f"wrote {OUT / 'choice.json'} and choice.md ({time.time() - t0:.0f}s)")
    try:
        other = beside_other(rec, am, rm)
    except (SystemExit, Exception) as e:      # beside, deciding nothing: its failure never loses the choice
        other = {"error": f"{type(e).__name__}: {e}"}
    if other is not None:
        rec["beside_other_corpora"] = other
        write_json(OUT / "choice.json", rec)
        (OUT / "choice.md").write_text(report_md(rec), encoding="utf-8")
        log(f"added the other corpora ({time.time() - t0:.0f}s)")
    return rec


def beside_other(rec, am, rm):
    """The carried pick, or else the cheapest new point at the incumbent's level, read on metaqa, squad and webqsp's
    filed arrays beside pick O (deciding nothing)."""
    p = rec["carried_pick"] or rec["at_incumbent_level"]["C"]
    if not p or not p.get("pick") or "@" not in p["pick"]:
        return None
    cell, rname, rest = p["pick"].split("/", 2)
    order, tail = rest.split(":", 1)
    bn, tau = tail.rsplit("@", 1)
    budz = {b[0]: b for b in nbr_budgets()}
    if bn not in budz:
        return {"pick": p["pick"], "note": "the pick is one of U2s's own candidates, not a new one"}
    _n, kind, prm = budz[bn]
    r, o = list(S.RECIPES).index(rname), list(S.NAMES).index(order)
    t = [x for x in S.TAUS_DESC if P1H.key(x) == tau][0]
    bud = {b[0]: b for b in S.budgets()}
    out = {"pick": p["pick"]}
    for d in OTHERS:
        x = S.load_u2s(d)
        if x is None:
            out[d] = None
            continue
        a, rr = x
        cbd = rr["cell_base"]
        k = [g[0] for g in S.GRID].index(cell)
        got = eval_one(a, k, r, o, nbr_k(kind, prm, a["n_n1"][:, cbd[k]]), t)
        ko = rr["own_cell"]
        _bn, bk, bp = bud["ada_f16_k0"]
        mq = S.mq_of(a, cbd[ko])
        ref = eval_one(a, ko, 0, list(S.NAMES).index("kt_cos"),
                       S.budget_k(bk, bp, a["fill"][:, ko * S.R], mq), 0.85)
        out[d] = {"pick": {"ALL": got[0], "rows": got[1]}, "pick_O": {"ALL": ref[0], "rows": ref[1]},
                  "mean_n_n1": round(float(a["n_n1"][:, cbd[k]].mean()), 1)}
    return out


def report_md(rec):
    D = rec["datasets"]
    inc = rec["incumbent"]
    L = ["# U2z stage A: pool depth from the question's own neighbourhood size", "",
         f"Declared in {rec['declared_in']}. Verdicts: **{rec['verdict_c']}**, **{rec['verdict_e']}**; carried: "
         f"**{rec['carried']}**.", "",
         f"Incumbent (pick O, per-dataset construction): `{inc['name']}`, level {inc['level']}, cost {inc['cost']} "
         f"rows on the three (P1f's weights {rec['weights']}); cap {rec['cap']}; row price bound {rec['row_price']} x "
         f"the incumbent's cost = {round(rec['row_price'] * inc['cost'], 1)}. Candidates: U2s's {rec['candidates']} "
         f"universal ones and {rec['new_candidates']} new ones.", ""]
    cp = rec["carried_pick"]
    if cp:
        L += [f"Carried: `{cp['pick']}` ({cp['form']}), level {cp['level']}, cost {cp['cost']} rows "
              f"({cp['price']} x pick O), within the cap: {cp['within_cap']}.", ""]

    def row(label, p):
        if not p or not p.get("pick"):
            return f"| {label} | - | - | " + " | ".join("-" for _ in D) + " |"
        return (f"| {label} | {p['level']} | {p['cost']} | "
                + " | ".join(f"{p['s1sel'][d]['ALL']} ({p['s1sel'][d]['rows']})" for d in D) + " |")

    L += ["## ALL (rows) on s1sel", "", "| pools | level | cost | " + " | ".join(D) + " |", "|---|---:|---:|" + "---|" * len(D)]
    L.append(f"| pick O `{inc['name']}` | {inc['level']} | {inc['cost']} | "
             + " | ".join(f"{inc['s1sel'][d]['ALL']} ({inc['s1sel'][d]['rows']})" for d in D) + " |")
    L.append(row(f"carried `{cp['pick']}`" if cp else "carried", cp))
    for key in ("C", "E"):
        p = rec["at_incumbent_level"][key]
        L.append(row(f"joint {key}'s cheapest at {inc['level']}" + (f" `{p['pick']}`" if p else ""), p))
    for key in ("C", "E"):
        p = rec[f"new_frontier_{key}"]["points"]
        at = [q for q in p if abs(q["level"] - inc["level"]) < 1e-9]
        L.append(row(f"new {key}'s cheapest at {inc['level']}" + (f" `{at[0]['pick']}`" if at and at[0].get("pick") else ""),
                     at[0] if at else None))
    for key, ttl in (("frontier_C", "Frontier C (joint)"), ("frontier_E", "Frontier E (joint)"),
                     ("new_frontier_C", "Beside: frontier C over the new candidates alone"),
                     ("new_frontier_E", "Beside: frontier E over the new candidates alone"),
                     ("frontier_O", "Beside: frontier O (pick O's own construction)")):
        L += ["", f"## {ttl} (every level, past the cap too)", "",
              "| level | cost | within cap | price | form | pick | " + " | ".join(D) + " |",
              "|---:|---:|---|---:|---|---|" + "---|" * len(D)]
        for p in rec[key]["points"]:
            if not p.get("pick"):
                L.append(f"| {p['level']} | - | - | - | - | none meets | " + " | ".join("-" for _ in D) + " |")
            else:
                L.append(f"| {p['level']} | {p['cost']} | {p['within_cap']} | {p['price']} | {p['form']} | `{p['pick']}` | "
                         + " | ".join(f"{p['s1sel'][d]['ALL']} ({p['s1sel'][d]['rows']})" for d in D) + " |")
    L += ["", "## Beside: the level reached within the cap and below (joint C and the new candidates' C)", "",
          "| level | joint C cost | " + " | ".join(f"joint {d}" for d in D) + " | new C cost | "
          + " | ".join(f"new {d}" for d in D) + " |", "|---:|---:|" + "---:|" * len(D) + "---:|" + "---:|" * len(D)]
    for pj, pn in zip(rec["curves_0.80_to_1.00"]["joint_C"], rec["curves_0.80_to_1.00"]["new_C"]):
        def cells(p):
            if not p.get("pick"):
                return ["-"] + ["-"] * len(D)
            return [str(p["cost"])] + [str(p["s1sel"][d]["rows"]) for d in D]
        L.append(f"| {pj['level']} | " + " | ".join(cells(pj) + cells(pn)) + " |")
    L += ["", f"## Beside: the cheapest point at level {inc['level']} by new budget", "", "| budget | cost | pick |",
          "|---|---:|---|"]
    for bn, p in rec["by_new_budget"].items():
        L.append(f"| {bn} | {p['cost'] if p else '-'} | {('`' + p['pick'] + '`') if p else 'none meets'} |")
    L += ["", "## Beside: mean one-link neighbourhood size n_n1 by base, and mean |I| by construction", ""]
    for d in D:
        L.append(f"- {d}: n_n1 {rec['mean_n_n1'][d]}; |I| {rec['mean_I'][d]}")
    L += ["", "## Beside: per dataset, the fewest mean rows any new candidate needs (upper bound only)", "",
          "| level | " + " | ".join(D) + " |", "|---:|" + "---:|" * len(D)]
    for t in rec["per_dataset_fewest_rows_new"][D[0]]:
        L.append(f"| {t} | " + " | ".join(str(rec["per_dataset_fewest_rows_new"][d][t]) for d in D) + " |")
    ob = rec.get("beside_other_corpora")
    if ob and "error" in ob:
        L += ["", f"## Beside: the other corpora's filed arrays: not read ({ob['error']})"]
    elif ob and "note" in ob:
        L += ["", f"## Beside: the other corpora: `{ob['pick']}`: {ob['note']}"]
    elif ob:
        L += ["", f"## Beside: `{ob['pick']}` on the other corpora's filed arrays, with pick O", ""]
        for d in OTHERS:
            x = ob.get(d)
            L.append(f"- {d}: " + ("no filed stats" if not x else
                                    f"pick ALL {x['pick']['ALL']} ({x['pick']['rows']} rows), pick O ALL "
                                    f"{x['pick_O']['ALL']} ({x['pick_O']['rows']} rows), mean n_n1 {x['mean_n_n1']}"))
    ca = rec["ceilings_all_candidates"]
    L += ["", f"Ceilings (U2w's, the targets' bound): {rec['ceilings']}; over every candidate here: {ca}."]
    return "\n".join(L) + "\n"


def _selftest():
    # the budgets: 33 per-question and 11 per-corpus, unique names
    bud = nbr_budgets()
    nq_b = len(LAMS) * len(K0S)
    assert NBZ == nq_b + len(LAMS) == 44 and len({b[0] for b in bud}) == 44
    assert bud[0][0] == "nbq_l0.25_k0" and bud[-1][0] == "nbc_l8" and bud[nq_b - 1][1] == "NBRQ" and bud[nq_b][1] == "NBRC"
    assert [b[0] for b in bud[:5]] == ["nbq_l0.25_k0", "nbq_l0.25_k50", "nbq_l0.25_k300", "nbq_l0.35_k0",
                                       "nbq_l0.35_k50"], [b[0] for b in bud[:5]]
    # k_q: rounding, the k0 floor, the corpus mean
    s = np.asarray([0, 3, 10, 2610, 1366])
    assert nbr_k("NBRQ", (0.5, 0), s).tolist() == [0, 2, 5, 1305, 683]        # np.rint: 1.5 -> 2, 0.5-multiples to even
    assert nbr_k("NBRQ", (0.5, 50), s).tolist() == [50, 50, 50, 1305, 683]
    kc = nbr_k("NBRC", (2.0, 0), s)                                           # 2 x 797.8 = 1595.6 -> 1596
    assert kc.tolist() == [1596] * 5, kc
    assert nbr_k("NBRC", (0.25, 0), np.asarray([5, 7])).tolist() == [2, 2]
    # the tables against U2e's evaluate on a toy: 2 cells, 3 recipes, 14 orders, 6 questions, 9 golds
    rng = np.random.default_rng(1)
    nq, nk = 6, 2
    ncol = nk * S.R * S.NO
    ptr = np.asarray([0, 2, 3, 3, 6, 8, 9])
    ng = int(ptr[-1])
    a = {"bs": rng.integers(40, 60, size=(nq, nk)), "n_n1": rng.integers(0, 3000, size=(nq, 3)),
         "lens": rng.integers(0, 400, size=(nq, ncol)).astype(np.int32),
         "pos": rng.integers(-1, 450, size=(ng, ncol)).astype(np.int32), "ptr": ptr}
    a["pos"][rng.random(a["pos"].shape) < 0.2] = 0
    cb = [0, 2]
    AL, RW = table_nbr(a, [0, 1], cb)
    assert AL.size == nk * S.R * S.NO * NBZ * S.NT
    check_tables(a, cb, [0, 1], AL, RW, rng, n=60)
    # a budget at or above every order's length keeps the whole order: the value at tau 1 is the ceiling's
    big = nbr_k("NBRQ", (8.0, 300), np.full(nq, 10 ** 6))
    assert (big >= 10 ** 6).all()
    # the joint index mapping
    assert name_z(0).endswith(":nbq_l0.25_k0@1") or ":nbq_l0.25_k0@" in name_z(0), name_z(0)
    assert decode_z(S.NT) == (0, 0, 0, 1, 0) and decode_z(NBZ * S.NT) == (0, 0, 1, 0, 0)
    # the frontier curve: the cheapest candidate meeting every target at each level
    A = np.asarray([[0.90, 0.92], [0.95, 0.96], [0.99, 0.99]])
    c = np.asarray([10.0, 20.0, 90.0])
    got = frontier_curve(A, c, c, np.asarray([0.99, 0.99]), np.arange(3), (0.90, 0.95, 0.98),
                         lambda t, tg, k: (round(t, 3), k))
    assert got == [(0.9, 0), (0.95, 1), (0.98, 2)], got
    print("selftest: PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("dev", "choose"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
    elif a.cmd == "dev":
        dev()
    elif a.cmd == "choose":
        choose(a.cache)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
