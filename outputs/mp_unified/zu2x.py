"""U2x: a pool budget read from RRF5T's agreement across its five orders (docs/U2X_RRF5T_AGREEMENT_BUDGET.md).

    python outputs/mp_unified/zu2x.py --selftest
    python outputs/mp_unified/zu2x.py dev D [--limit 5] [--threads 2]      (U2s's development run; nothing written)
    python outputs/mp_unified/zu2x.py stats D [--shard I K] [--threads 2]  (host CPU)
    python outputs/mp_unified/zu2x.py merge D K
    python outputs/mp_unified/zu2x.py choose                               (laptop) -> outputs/u2x/choice.json, .md

`stats` runs zu2s.stats_stage unchanged, with three in-process patches that are restored afterwards:
- p1e_anchor_kind.p1e_orders: after the original, RRF5T's fused scores are rebuilt from its five orders (P1d's rrf,
  scores kept) and their order must be RRF5T's, question for question;
- zu2s.mass_counts: for each base's B, RRF5T's fused-score mass outside B is also counted at U2s's five tau_w. This is
  U2s's MASS rule with the recipe's fused score in place of one walk's mass (U2s section 10, first form);
- zu2s.OUT: outputs/u2x, so nothing under outputs/u2s is written.
`choose` adds the 60 AGREE budgets (MASS's grid of tau_w, f and k0) to U2s's 219 on every universal construction and
decides with U2w's rule (zu2w.decide), unchanged. Imports zu2s and zu2w unchanged; writes only under outputs/u2x.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zu2s as S  # noqa: E402  (U2s's stats pass, budgets, tables and frontiers, unchanged)
import zu2w as W  # noqa: E402  (U2w's cost, levels, verdict, incumbent and checks, unchanged)

M, F, P, Q, PG, U2C, U2E = S.M, S.F, S.P, S.Q, S.PG, S.U2C, S.U2E
log, utc, write_json, sha_file = S.log, S.utc, S.write_json, S.sha_file
DECLARED = "docs/U2X_RRF5T_AGREEMENT_BUDGET.md"
OUT = ROOT / "outputs" / "u2x"
THREE = W.THREE
FORM = "AGREE"
FIVE = ("A1st", "A1all", "KTst", "KTall", "R0")       # RRF5T's five orders (p1e_anchor_kind.p1e_orders)


def script_sha256():
    return sha_file(Path(__file__))


def budgets_agree():
    """[(name, form, (tau_w slot, f, k0))]: MASS's grid (U2s section 4) with RRF5T's fused score as the mass."""
    return [(f"agree_t{tw:g}_f{f}_k{k0}", FORM, (ti, f, k0)) for ti, tw in enumerate(S.MASS_TAUS)
            for f in S.MASS_F for k0 in S.MASS_K0]


NBA = len(budgets_agree())


def budget_k_agree(prm, aq):
    """k_q per question: max(k0, round(f x a_q)), MASS's rule on the base's fused-score count a_q (questions x tau_w)."""
    ti, f, k0 = prm
    return np.maximum(int(k0), np.rint(float(f) * np.asarray(aq[:, ti], np.int64)).astype(np.int64))


def fused(orders, c, nmax):
    """P1d's rrf (scripts/p1d_free_pool.py) with its scores kept: (nodes in RRF's order, each one's fused score)."""
    keep = [np.asarray(o, np.int64) for o in orders if np.asarray(o).size]
    if not keep:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    ids = np.concatenate(keep)
    sc = np.concatenate([1.0 / (c + np.arange(1, o.size + 1, dtype=np.float64)) for o in keep])
    u, inv = np.unique(ids, return_inverse=True)
    m = np.bincount(inv, weights=sc, minlength=u.size)
    o = np.lexsort((u, -m))[:nmax]
    return u[o], m[o]


def agree_ok(aq, na):
    """U2s section 7's bounds on one walk: a_q non-decreasing in tau_w, at most the count outside B, 0 exactly when
    that count is 0 (questions x bases x tau_w, questions x bases)."""
    aq, na = np.asarray(aq, np.int64), np.asarray(na, np.int64)
    return bool(aq.ndim == 3 and na.shape == aq.shape[:2] and aq.shape[2] == len(S.MASS_TAUS) and
                (np.diff(aq, axis=2) >= 0).all() and (aq <= na[:, :, None]).all() and
                ((aq[:, :, 0] == 0) == (na == 0)).all() and ((aq == 0).all(axis=2) == (na == 0)).all())


# ── stats: U2s's pass with RRF5T's fused scores captured ─────────────────────


class Capture:
    """Per question, in the pass's order: each base's (a_q at U2s's five tau_w, fused nodes outside B)."""

    def __init__(self, nbases):
        self.nbases, self.cur, self.last_b, self.q, self.rows, self.questions = nbases, None, None, [], [], 0

    def start(self, ids, sc):
        self.close()
        self.cur, self.last_b, self.q = (ids, sc), None, []
        self.questions += 1

    def close(self):
        if self.cur is None:
            return
        if len(self.q) != self.nbases:
            raise SystemExit(f"question {len(self.rows)}: {len(self.q)} bases counted, not {self.nbases}")
        self.rows.append(self.q)
        self.cur, self.q = None, []

    def arrays(self):
        self.close()
        aq = np.asarray([[ms for ms, _c in q] for q in self.rows], np.int64).reshape(len(self.rows), self.nbases,
                                                                                      len(S.MASS_TAUS))
        na = np.asarray([[c for _ms, c in q] for q in self.rows], np.int64).reshape(len(self.rows), self.nbases)
        return aq, na


@contextlib.contextmanager
def patched(cap):
    import p1e_anchor_kind as K           # P1e's orders (imports numba)
    orig_p1e, orig_mc, out0 = K.p1e_orders, S.mass_counts, S.OUT

    def p1e(src, st, route, knd, n_st, constant):
        out, p1d = orig_p1e(src, st, route, knd, n_st, constant)
        o = K.D.rule_orders(src, route, constant)
        five = [o["A1st"], o["A1all"], np.asarray(st["KTst"], np.int64)[:K.NMAX],
                np.asarray(st["KTall"], np.int64)[:K.NMAX], o["R0"]]
        ids, sc = fused(five, constant, K.D.NMAX)
        if not np.array_equal(ids, np.asarray(out["RRF5T"], np.int64)):
            raise SystemExit(f"question {cap.questions}: the fused scores' order is not RRF5T's")
        cap.start(ids, sc)
        return out, p1d

    def mc(nodes, mass, B, taus=S.MASS_TAUS):
        r = orig_mc(nodes, mass, B, taus)
        if cap.cur is None:
            raise SystemExit("a mass count before any question's orders")
        if cap.last_b is not B:            # the first walk of a new base: count the fused score outside its B once
            cap.last_b = B
            cap.q.append(orig_mc(cap.cur[0], cap.cur[1], B, S.MASS_TAUS))
        return r

    K.p1e_orders, S.mass_counts, S.OUT = p1e, mc, OUT
    try:
        yield
    finally:
        K.p1e_orders, S.mass_counts, S.OUT = orig_p1e, orig_mc, out0


def agree_paths(name, shard=None):
    carve = U2C.carve_of(name)
    if shard:
        stem = f"{name}__{carve}__{shard[0]}of{shard[1]}"
        d = OUT / "stats" / "parts"
    else:
        stem = f"{name}__{carve}"
        d = OUT / "stats"
    return d / f"{stem}.agree.npz", d / f"{stem}.agree.json"


def write_agree(name, aq, na, extra, shard=None):
    p, rp = agree_paths(name, shard)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name.replace(".agree.npz", ".agree.tmp.npz"))
    np.savez_compressed(tmp, aq=aq, na=na)
    tmp.replace(p)
    write_json(rp, {"declared_in": DECLARED, "stage": "U2x", "dataset": name, "carve": U2C.carve_of(name),
                    "script_sha256": script_sha256(), "zu2s_sha256": S.script_sha256(), "five": list(FIVE),
                    "mass_taus": list(S.MASS_TAUS), "bases": [list(b) for b in S.BASES],
                    "questions": int(aq.shape[0]), "file_sha256": sha_file(p), "checks": {"bounds": True,
                                                                                         "rrf5t_order": True},
                    **extra, "utc": utc()})
    return p


def run_pass(name, threads, limit=0, shard=None):
    cap = Capture(len(S.BASES))
    t0 = time.time()
    with patched(cap):
        rc = S.stats_stage(name, threads, limit, shard)
    aq, na = cap.arrays()
    if rc != 0 or not agree_ok(aq, na):
        raise SystemExit(f"{name}: the pass failed, or the fused-score counts break section 7's bounds")
    return aq, na, round(time.time() - t0, 1)


def dev(name, threads, limit):
    aq, na, sec = run_pass(name, threads, limit)
    if aq.shape[0] != limit:
        raise SystemExit(f"{name}: {aq.shape[0]} questions captured, not {limit}")
    log(f"{name}: development run on {limit} questions ({sec:.0f}s): RRF5T's order and the bounds hold; a_q at "
        f"tau_w {S.MASS_TAUS} per base, first question {aq[0].tolist()}; nothing written")
    return 0


def stats(name, threads, shard):
    carve = U2C.carve_of(name)
    aq, na, sec = run_pass(name, threads, 0, shard)
    if shard:
        r = json.loads((OUT / "stats" / "parts" / f"{name}__{carve}__{shard[0]}of{shard[1]}.json").read_text(
            encoding="utf-8"))
        if aq.shape[0] != r["q1"] - r["q0"]:
            raise SystemExit(f"{name}: {aq.shape[0]} questions captured, not the part's {r['q1'] - r['q0']}")
        extra = {"part": shard[0], "parts": shard[1], "q0": r["q0"], "q1": r["q1"], "seconds": sec,
                 "u2s_part_sha256": r["file_sha256"]}
    else:
        r = json.loads((OUT / "stats" / f"{name}__{carve}.json").read_text(encoding="utf-8"))
        if aq.shape[0] != r["questions"]:
            raise SystemExit(f"{name}: {aq.shape[0]} questions captured, not {r['questions']}")
        extra = {"seconds": sec, "u2s_stats_sha256": r["file_sha256"]}
    p = write_agree(name, aq, na, extra, shard)
    log(f"{name}: fused-score counts written to {p.name} ({sec:.0f}s)")
    return 0


def merge(name, k):
    out0 = S.OUT
    S.OUT = OUT
    try:
        S.merge_stage(name, k)
    finally:
        S.OUT = out0
    aqs, nas, shas = [], [], []
    for i in range(k):
        p, rp = agree_paths(name, (i, k))
        r = json.loads(rp.read_text(encoding="utf-8"))
        if r["script_sha256"] != script_sha256() or sha_file(p) != r["file_sha256"] or (r["part"], r["parts"]) != (i, k):
            raise SystemExit(f"{rp}: not this script's, not its file, or not part {i} of {k}")
        with np.load(p) as z:
            aqs.append(z["aq"])
            nas.append(z["na"])
        shas.append(r["file_sha256"])
    aq, na = np.concatenate(aqs), np.concatenate(nas)
    carve = U2C.carve_of(name)
    r = json.loads((OUT / "stats" / f"{name}__{carve}.json").read_text(encoding="utf-8"))
    if aq.shape[0] != r["questions"] or not agree_ok(aq, na):
        raise SystemExit(f"{name}: the merged counts do not cover the merged stats' questions, or break the bounds")
    write_agree(name, aq, na, {"shards": k, "part_sha256": shas, "u2s_stats_sha256": r["file_sha256"]})
    log(f"{name}: merged {k} parts of fused-score counts")
    return 0


def load_agree(name):
    p, rp = agree_paths(name)
    r = json.loads(rp.read_text(encoding="utf-8"))
    if r["script_sha256"] != script_sha256() or sha_file(p) != r["file_sha256"]:
        raise SystemExit(f"{rp}: not this script's, or not its file")
    with np.load(p) as z:
        aq, na = z["aq"], z["na"]
    if not agree_ok(aq, na):
        raise SystemExit(f"{p}: the counts break section 7's bounds")
    return aq, na, r


# ── choose (the laptop) ──────────────────────────────────────────────────────


def agree_tables(a, aq, ks, rs, cell_base):
    """(ALL, rows) of every AGREE candidate of cells ks x recipes rs on one dataset's arrays, flat in (cell, recipe,
    order, AGREE budget, tau from 1 down) order: zu2s.table_arrays with the AGREE budgets."""
    ptr = np.asarray(a["ptr"], np.int64)
    bud = budgets_agree()
    AL = np.empty((len(ks), len(rs), S.NO, len(bud), S.NT))
    RW = np.empty_like(AL)
    for ki, k in enumerate(ks):
        b = np.asarray(a["bs"][:, k], np.int64)
        KQ = np.stack([budget_k_agree(prm, aq[:, cell_base[k], :]) for _n, _f, prm in bud])
        for ri, r in enumerate(rs):
            for o in range(S.NO):
                c = M.column(k, r, o)
                AL[ki, ri, o], RW[ki, ri, o] = S.column_table(b, a["lens"][:, c], a["pos"][:, c], KQ, ptr)
    return AL.ravel(), RW.ravel()


def name_a(i):
    k, r, o, b, t = (int(x) for x in np.unravel_index(int(i), (len(S.GRID), S.R, S.NO, NBA, S.NT)))
    return M.candidate_name(S.GRID[k][0], r, o, budgets_agree()[b][0], S.TAUS_DESC[t])


def load_rerun(name, filed):
    """U2x's rerun of U2s's pass (outputs/u2x/stats), checked array for array against U2s's filed stats."""
    out0 = S.OUT
    S.OUT = OUT
    try:
        x = S.load_u2s(name)
    finally:
        S.OUT = out0
    if x is None:
        raise SystemExit(f"INCOMPLETE: no U2x rerun of U2s's stats for {name}")
    a, r = x
    if set(a) != set(filed) or any(not np.array_equal(a[k], filed[k]) for k in filed):
        raise SystemExit(f"{name}: the rerun's arrays are not U2s's filed arrays")
    return r


def choose():
    t0 = time.time()
    u2w_path = W.OUT / "choice.json"
    u2w = json.loads(u2w_path.read_text(encoding="utf-8"))
    if u2w["script_sha256"] != W.script_sha256() or u2w["carried"] is not None:
        raise SystemExit("U2w's filed choice is not zu2w.py's, or it carried a pick (U2x follows NOT_UNIFIED only)")
    am, rm, ag, rerun = {}, {}, {}, {}
    for d in THREE:
        x = S.load_u2s(d)
        if x is None:
            raise SystemExit(f"INCOMPLETE: no U2s stats for {d}")
        am[d], rm[d] = x
        if rm[d]["file_sha256"] != u2w["stats_sha256"][d]:
            raise SystemExit(f"{d}: U2s's stats are not the ones U2w read")
        rerun[d] = load_rerun(d, am[d])
        aq, na, ra = load_agree(d)
        if ra["u2s_stats_sha256"] != rerun[d]["file_sha256"] or aq.shape[:2] != (am[d]["bs"].shape[0], len(S.BASES)):
            raise SystemExit(f"{d}: the fused-score counts are not the rerun's questions and bases")
        ag[d] = (aq, na, ra)
    w = F.weights()
    if {d: w[d] for d in THREE} != u2w["weights"]:
        raise SystemExit("P1f's weights are not U2w's")
    bud = S.budgets()
    cb = {d: rm[d]["cell_base"] for d in THREE}
    tabs = {d: S.table_arrays(am[d], range(len(S.GRID)), range(S.R), cb[d]) for d in THREE}
    tag = {d: agree_tables(am[d], ag[d][0], range(len(S.GRID)), range(S.R), cb[d]) for d in THREE}
    own = {d: S.table_arrays(am[d], [rm[d]["own_cell"]], range(S.R), cb[d]) for d in THREE}
    nu = len(S.GRID) * S.R * S.NO * S.NB * S.NT
    na_ = len(S.GRID) * S.R * S.NO * NBA * S.NT
    log(f"tables of {nu} universal, {na_} AGREE and {own[THREE[0]][0].size} own candidates ({time.time() - t0:.0f}s)")

    # U2w's checks: pick O's filed point and U2w's filed frontier C points, from the same tables
    i_inc = W.own_index(W.PICK_O)
    A_o = np.stack([own[d][0] for d in THREE], axis=1)
    RW_o = np.stack([own[d][1] for d in THREE], axis=1)
    for j, d in enumerate(THREE):
        if (float(A_o[i_inc, j]), float(RW_o[i_inc, j])) != (u2w["incumbent"]["s1sel"][d]["ALL"],
                                                             u2w["incumbent"]["s1sel"][d]["rows"]):
            raise SystemExit(f"{d}: pick O is not U2w's filed incumbent")
    n_c = 0
    for p in u2w["frontier_C"]["points"]:
        if not p.get("pick"):
            continue
        i = W.index_u(p["pick"], bud)
        for d in THREE:
            if (float(tabs[d][0][i]), float(tabs[d][1][i])) != (p["s1sel"][d]["ALL"], p["s1sel"][d]["rows"]):
                raise SystemExit(f"{d}: U2w's frontier C point at {p['level']} is not reproduced")
        n_c += 1

    A_ = np.concatenate([np.stack([tabs[d][0] for d in THREE], axis=1), np.stack([tag[d][0] for d in THREE], axis=1)])
    RW = np.concatenate([np.stack([tabs[d][1] for d in THREE], axis=1), np.stack([tag[d][1] for d in THREE], axis=1)])
    costs, cost_r = W.costs_of(RW, w)
    ceil = A_.max(axis=0)
    ceil_u = np.stack([tabs[d][0] for d in THREE], axis=1).max(axis=0)
    gr = np.concatenate([np.unravel_index(np.arange(nu), (len(S.GRID), S.R, S.NO, S.NB, S.NT))[1],
                         np.unravel_index(np.arange(na_), (len(S.GRID), S.R, S.NO, NBA, S.NT))[1]])
    is_ag = np.arange(nu + na_) >= nu
    c_o, cr_o = W.costs_of(RW_o, w)
    no_ = S.R * S.NO * S.NB * S.NT
    orr = np.unravel_index(np.arange(no_), (1, S.R, S.NO, S.NB, S.NT))[1]
    l_inc = S.level_of(A_o[i_inc], ceil)
    cost_inc = float(cr_o[i_inc])
    if l_inc is None:
        raise SystemExit("pick O meets no level on the three")
    tf = S.three_frontiers(A_, costs, cost_r, gr, A_o, c_o, cr_o, orr, ceil, W.CAP)
    _tg, c_k = W.point_at(tf["C"][0], l_inc)
    _tg, e_k = W.point_at(tf["E"][0], l_inc)
    c_at = None if c_k is None else float(cost_r[c_k])
    e_at = None if e_k is None else float(cost_r[e_k])
    vc, ve, carried = W.decide(tf["C"][1], tf["E"][1], l_inc, cost_inc, c_at, e_at)
    log(f"ceilings {dict(zip(THREE, ceil.tolist()))}; pick O level {l_inc}, cost {cost_inc}; C {tf['C'][1]}, "
        f"E {tf['E'][1]} within the cap; at {l_inc}: C {c_at}, E {e_at}; {vc}, {ve}, carried {carried} "
        f"({time.time() - t0:.0f}s)")
    # beside: the frontiers over the AGREE candidates alone
    tfa = S.three_frontiers(A_[is_ag], costs[is_ag], cost_r[is_ag], gr[is_ag], A_o, c_o, cr_o, orr, ceil, W.CAP)

    def point(t, tgt, i, agree_only=False):
        rec = {"level": t, "targets": dict(zip(THREE, np.asarray(tgt).tolist())), "pick": None}
        if i is None:
            return rec
        g = int(i) + (nu if agree_only else 0)
        rec.update({"pick": name_a(g - nu) if g >= nu else W.name_u(g, bud), "cost": float(cost_r[g]),
                    "within_cap": bool(cost_r[g] <= W.CAP + 1e-9), "price": round(float(cost_r[g]) / cost_inc, 3),
                    "form": FORM if g >= nu else bud[int(np.unravel_index(g, (len(S.GRID), S.R, S.NO, S.NB,
                                                                                  S.NT))[3])][1],
                    "s1sel": {d: {"ALL": float(A_[g, j]), "rows": float(RW[g, j])} for j, d in enumerate(THREE)}})
        return rec

    def frontier(tfx, key, agree_only=False):
        pts, lv, kk = tfx[key]
        return {"level": lv, "points": [point(t, tg, k_, agree_only) for t, tg, k_ in pts],
                "pick": None if kk is None else point(lv, W.point_at(pts, lv)[0], kk, agree_only)}

    carried_rec = None
    if carried in ("E", "C"):
        carried_rec = frontier(tf, carried)["pick"]
    elif carried in ("C_ROW", "E_ROW"):
        key = carried[0]
        carried_rec = point(l_inc, W.point_at(tf[key][0], l_inc)[0], c_k if key == "C" else e_k)
    tgt_inc = np.round(np.minimum(l_inc, ceil - PG.TOL), 4)
    meets = (A_ >= tgt_inc[None, :] - 1e-9).all(axis=1)
    i_ag = np.flatnonzero(meets & is_ag)
    agree_at = None if i_ag.size == 0 else point(l_inc, tgt_inc, int(i_ag[np.argmin(costs[i_ag])]))
    rec = {"declared_in": DECLARED, "stage": "U2x", "part": "A", "script_sha256": script_sha256(),
           "zu2s_sha256": S.script_sha256(), "zu2w_sha256": W.script_sha256(), "u2w_choice_sha256": sha_file(u2w_path),
           "stats_sha256": {d: rm[d]["file_sha256"] for d in THREE},
           "rerun_stats_sha256": {d: rerun[d]["file_sha256"] for d in THREE},
           "agree_sha256": {d: ag[d][2]["file_sha256"] for d in THREE}, "datasets": list(THREE),
           "weights": {d: w[d] for d in THREE}, "cap": W.CAP, "row_price": W.ROW_PRICE, "candidates": nu,
           "agree_candidates": na_, "ceilings": dict(zip(THREE, ceil.tolist())),
           "ceilings_without_agree": dict(zip(THREE, ceil_u.tolist())),
           "incumbent": {"name": W.PICK_O, "level": l_inc, "cost": cost_inc},
           "verdict_c": vc, "verdict_e": ve, "carried": carried, "carried_pick": carried_rec,
           "at_incumbent_level": {"C": None if c_k is None else point(l_inc, tgt_inc, c_k),
                                  "E": None if e_k is None else point(l_inc, tgt_inc, e_k),
                                  "AGREE": agree_at},
           "frontier_C": frontier(tf, "C"), "frontier_E": frontier(tf, "E"),
           "agree_only": {"C": frontier(tfa, "C", True), "E": frontier(tfa, "E", True)},
           "checks": {"rerun_equals_u2s": True, "rrf5t_order": True, "bounds": True, "pick_o_filed_point": True,
                      "u2w_frontier_c_points": n_c, "weights": True},
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "choice.json", rec)
    (OUT / "choice.md").write_text(report_md(rec), encoding="utf-8", newline="\r\n")
    log(f"wrote {OUT / 'choice.json'} and choice.md ({time.time() - t0:.0f}s)")
    return rec


def report_md(rec):
    def cells(p):
        return " | ".join(f"{p['s1sel'][d]['ALL']} ({p['s1sel'][d]['rows']})" for d in THREE) if p and p.get("pick") \
            else " | ".join("-" for _ in THREE)

    L = [f"# U2x stage A: a budget from RRF5T's agreement, on {', '.join(THREE)}", "",
         f"Declared in {DECLARED}. Verdicts: **{rec['verdict_c']}**, **{rec['verdict_e']}**; carried: "
         f"**{rec['carried']}**.", "",
         f"Incumbent pick O `{rec['incumbent']['name']}`: level {rec['incumbent']['level']}, cost "
         f"{rec['incumbent']['cost']} rows; cap {rec['cap']}; row price {rec['row_price']} x the incumbent's cost. "
         f"Candidates: U2s's {rec['candidates']} universal ones and {rec['agree_candidates']} AGREE ones.", "",
         "## ALL (rows) on s1sel at the incumbent's level", "", f"| pools | cost | {' | '.join(THREE)} |",
         "|---|---:|" + "---|" * len(THREE)]
    for key, p in (("C", rec["at_incumbent_level"]["C"]), ("E", rec["at_incumbent_level"]["E"]),
                   ("AGREE's cheapest", rec["at_incumbent_level"]["AGREE"])):
        L.append(f"| {key}: `{p['pick'] if p else '-'}` | {p['cost'] if p and p.get('pick') else '-'} | {cells(p)} |")
    for key in ("frontier_C", "frontier_E"):
        L += ["", f"## {key.replace('_', ' ')} (U2s's and AGREE's candidates)", "",
              f"| level | cost | within cap | form | pick | {' | '.join(THREE)} |", "|---:|---:|---|---|---|" +
              "---|" * len(THREE)]
        for p in rec[key]["points"]:
            if p.get("pick"):
                L.append(f"| {p['level']} | {p['cost']} | {p['within_cap']} | {p['form']} | `{p['pick']}` | {cells(p)} |")
    for key in ("C", "E"):
        L += ["", f"## Beside: frontier {key} over the AGREE candidates alone", "",
              f"| level | cost | within cap | pick | {' | '.join(THREE)} |", "|---:|---:|---|---|" + "---|" * len(THREE)]
        for p in rec["agree_only"][key]["points"]:
            if p.get("pick"):
                L.append(f"| {p['level']} | {p['cost']} | {p['within_cap']} | `{p['pick']}` | {cells(p)} |")
    L += ["", f"Checks: {rec['checks']}.", ""]
    return "\n".join(L)


# ── self-test ────────────────────────────────────────────────────────────────


def _selftest():
    import p1d_free_pool as D
    rng = np.random.default_rng(0)
    # 1. fused() is P1d's rrf with its scores, on random orders with overlaps, empties and ties
    for _ in range(200):
        orders = [rng.permutation(60)[:rng.integers(0, 40)] for _ in range(5)]
        ids, sc = fused(orders, 60, 30)
        assert np.array_equal(ids, D.rrf(orders, 60, 30)), "fused order is not rrf's"
        assert (np.diff(sc) <= 0).all()
    # 2. the AGREE budgets are MASS's grid and rule on one walk
    ab = budgets_agree()
    assert len(ab) == len(S.MASS_TAUS) * len(S.MASS_F) * len(S.MASS_K0) == NBA == 60
    mass = [b for b in S.budgets() if b[1] == "MASS" and b[2][0] == 0]
    aq = rng.integers(0, 900, size=(50, len(S.MASS_TAUS))).cumsum(axis=1)
    mq = np.zeros((50, S.NW * S.NTW), np.int64)
    mq[:, :S.NTW] = aq
    for (_n, _f, prm), (_nm, kind, mprm) in zip(ab, mass):
        assert prm == mprm[1:] and np.array_equal(budget_k_agree(prm, aq), S.budget_k(kind, mprm, None, mq))
    # 3. the capture: one entry per base per question, the fused count once per B, in the pass's order
    cap = Capture(3)
    calls = 0
    for qn in range(4):
        ids = rng.permutation(100)[:80]
        sc = np.sort(rng.random(80))[::-1]
        cap.start(ids, sc)
        for _b in range(3):
            B = rng.permutation(100)[:10]
            for _w in range(2):
                if cap.last_b is not B:
                    cap.last_b = B
                    cap.q.append(S.mass_counts(ids, sc, B))
                calls += 1
    a2, n2 = cap.arrays()
    assert a2.shape == (4, 3, len(S.MASS_TAUS)) and n2.shape == (4, 3) and agree_ok(a2, n2) and calls == 24
    bad = Capture(3)
    bad.start(np.arange(3), np.ones(3))
    bad.q.append(([1] * len(S.MASS_TAUS), 3))
    try:
        bad.close()
        raise AssertionError("a question with one base of three passed")
    except SystemExit:
        pass
    # 4. agree_tables equals U2e's evaluate, candidate by candidate, on made-up arrays
    nq, gper = 30, 2
    ncol = len(S.GRID) * S.R * S.NO
    ptr = np.arange(0, nq * gper + 1, gper, dtype=np.int64)
    a = {"bs": rng.integers(5, 40, size=(nq, len(S.GRID))), "lens": rng.integers(0, 300, size=(nq, ncol)),
         "pos": rng.integers(-1, 300, size=(nq * gper, ncol)), "ptr": ptr}
    a["pos"] = np.where(a["pos"] > np.repeat(a["lens"], gper, axis=0), -1, a["pos"])
    aqx = rng.integers(0, 200, size=(nq, len(S.BASES), len(S.MASS_TAUS))).cumsum(axis=2)
    cbx = [S.GRID[k][1] for k in range(len(S.GRID))]
    AL, RW = agree_tables(a, aqx, [0, 3], [0, 2], cbx)
    for ki, k in enumerate((0, 3)):
        for ri, r in enumerate((0, 2)):
            for o in (0, S.NO - 1):
                for bi in (0, 17, NBA - 1):
                    for ti in (0, S.NT - 1):
                        c = M.column(k, r, o)
                        sub = {"bs": a["bs"][:, k], "lens": a["lens"][:, [c]], "pos": a["pos"][:, [c]], "ptr": ptr}
                        kq = budget_k_agree(ab[bi][2], aqx[:, cbx[k], :])
                        rows, held = U2E.evaluate(sub, 0, kq, S.TAUS_DESC[ti])
                        i = np.ravel_multi_index((ki, ri, o, bi, ti), (2, 2, S.NO, NBA, S.NT))
                        ng = np.diff(ptr)
                        want_all = round(float(sum(bool(held[ptr[q]:ptr[q + 1]].all()) and ng[q] > 0
                                                   for q in range(nq))) / nq, 4)
                        assert AL[i] == want_all and RW[i] == round(float(np.sum(rows)) / nq, 1), (k, r, o, bi, ti)
    # 5. names decode back to the grid, and the verdict is U2w's own function
    assert name_a(0).startswith(S.GRID[0][0] + "/") and "agree_t0.5_f1_k0" in name_a(0)
    assert W.decide(0.96, None, 0.96, 1000.0, 900.0, None)[0] == "UNIFIED_HOLDS"
    log("selftest: PASS (fused = rrf; AGREE = MASS's rule on the fused score; capture per base; tables = U2e's "
        "evaluate; names; U2w's verdict)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("dev", "stats", "merge", "choose"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("k", nargs="?", type=int)
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "K"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.cmd in ("dev", "stats", "merge") and a.name not in THREE:
        raise SystemExit(f"{a.cmd}: a dataset of {THREE}")
    if a.cmd == "dev":
        return dev(a.name, a.threads, a.limit)
    if a.cmd == "stats":
        return stats(a.name, a.threads, tuple(a.shard) if a.shard else None)
    if a.cmd == "merge":
        return merge(a.name, a.k)
    if a.cmd == "choose":
        choose()
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
