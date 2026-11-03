"""U2w (docs/U2W_ONE_POOL_RULE_THREE_DATASETS.md): one pool rule on 2wiki, hotpotqa and musique, chosen from U2s's
filed s1sel arrays. Stage A: pools only; nothing trains, nothing is fit, no walk runs and no candidate is new.

    python outputs/mp_unified/zu2w.py choose
    python outputs/mp_unified/zu2w.py --selftest

choose   U2s's universal candidates (12 constructions x 3 recipes x 14 orders x 219 budgets x 10 taus) and the own
         construction's, on U2s's filed s1sel stats of the three datasets, with U2s's verdict checks; pick O's filed
         frontier-O point and U2s's filed frontier C reproduced on each of the three; frontiers C, E and O over P1f's
         weights restricted to the three; section 4's verdicts and what is carried. Writes outputs/u2w/choice.json and
         choice.md.
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

M, MB, Q, P, F, PG = S.M, S.MB, S.Q, S.P, S.F, S.PG
utc, write_json, sha_file = S.utc, S.write_json, S.sha_file

DECLARED = "docs/U2W_ONE_POOL_RULE_THREE_DATASETS.md"
OUT = ROOT / "outputs" / "u2w"
THREE = tuple(d for d in S.TRAIN5 if d in ("musique", "hotpotqa", "2wiki"))    # U2s's order
PICK_O = "own/kind_t/kt_cos:ada_f16_k0@0.85"
CAP = 1953.5
ROW_PRICE = 2.0
VERDICTS_C = ("UNIFIED_RAISED", "UNIFIED_HOLDS", "UNIFIED_AT_A_PRICE", "UNIFIED_AT_A_ROW_PRICE", "NOT_UNIFIED")
VERDICTS_E = ("KIND_FREE_NO_PRICE", "KIND_FREE_AT_A_PRICE", "KIND_FREE_AT_A_ROW_PRICE", "KIND_FREE_TOO_DEAR")


def log(msg):
    print(f"[u2w {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha256():
    return sha_file(Path(__file__))


def costs_of(RW, w, names=THREE):
    """(costs, rounded costs) of [candidates x datasets] mean rows: P1f's cost over the given datasets' keys."""
    c = np.asarray([F.cost(dict(zip(names, r)), w) for r in RW.tolist()], np.float64)
    return c, np.asarray([round(float(x), 1) for x in c])


def point_at(pts, level):
    """(targets, index) of a frontier's point at one level, past the cap too; (None, None) when the level is absent."""
    for t, tg, k in pts:
        if abs(t - level) < 1e-9:
            return tg, k
    return None, None


def decide(l_c, l_e, l_inc, cost_inc, c_at, e_at, row_price=ROW_PRICE):
    """Section 4: (verdict (c), verdict (e), what is carried: 'E', 'C', 'C_ROW', 'E_ROW' or None). l_c and l_e are the
    frontiers' levels within the cap (None: no point within it); c_at and e_at are the rounded costs of C's and E's
    points at the incumbent's level, past the cap too (None: no candidate meets it)."""
    tol = PG.TOL
    bound = row_price * cost_inc + 1e-9
    floor = l_inc - tol
    if l_c is not None and l_c > l_inc + 1e-9:
        vc = "UNIFIED_RAISED"
    elif l_c is not None and abs(l_c - l_inc) < 1e-9:
        vc = "UNIFIED_HOLDS"
    elif l_c is not None and abs(l_c - floor) < 1e-9:
        vc = "UNIFIED_AT_A_PRICE"
    elif c_at is not None and c_at <= bound:
        vc = "UNIFIED_AT_A_ROW_PRICE"
    else:
        vc = "NOT_UNIFIED"
    if l_e is not None and l_e >= floor - 1e-9 and (l_c is None or l_e >= l_c - 1e-9):
        ve = "KIND_FREE_NO_PRICE"
    elif l_e is not None and l_c is not None and l_e >= floor - 1e-9 and abs(l_e - (l_c - tol)) < 1e-9:
        ve = "KIND_FREE_AT_A_PRICE"
    elif e_at is not None and e_at <= bound and (c_at is None or e_at <= c_at + 1e-9):
        ve = "KIND_FREE_AT_A_ROW_PRICE"
    else:
        ve = "KIND_FREE_TOO_DEAR"
    if ve in VERDICTS_E[:2]:
        carried = "E"
    elif vc in VERDICTS_C[:3]:
        carried = "C"
    elif vc == "UNIFIED_AT_A_ROW_PRICE" and (e_at is None or c_at <= e_at + 1e-9):
        carried = "C_ROW"
    elif ve == "KIND_FREE_AT_A_ROW_PRICE":
        carried = "E_ROW"
    else:
        carried = None
    return vc, ve, carried


def own_index(point):
    _cell, r, o, b, t = MB.parse(point)
    return int(np.ravel_multi_index((0, r, o, b, S.TAUS_DESC.index(t)), (1, S.R, S.NO, S.NB, S.NT)))


def name_u(i, bud):
    k, r, o, b, t = S.decode(i, len(S.GRID), S.R)
    return M.candidate_name(S.GRID[k][0], r, o, bud[b][0], S.TAUS_DESC[t])


def name_o(i, bud):
    _k, r, o, b, t = S.decode(i, 1, S.R)
    return M.candidate_name("own", r, o, bud[b][0], S.TAUS_DESC[t])


def index_u(name, bud):
    """The universal table's flat index of a candidate name (section 5's order)."""
    cell = name.split("/", 1)[0]
    k = [g[0] for g in S.GRID].index(cell)
    per = S.R * S.NO * S.NB * S.NT
    for j in range(per):
        i = k * per + j
        if name_u(i, bud) == name:
            return i
    raise SystemExit(f"{name}: not a universal candidate")


def choose():
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
    bud = S.budgets()
    cb = {d: rm[d]["cell_base"] for d in THREE}
    tabs = {d: S.table_arrays(am[d], range(len(S.GRID)), range(S.R), cb[d]) for d in THREE}
    own = {d: S.table_arrays(am[d], [rm[d]["own_cell"]], range(S.R), cb[d]) for d in THREE}
    log(f"tables of {tabs[THREE[0]][0].size} universal and {own[THREE[0]][0].size} own candidates on {THREE} "
        f"({time.time() - t0:.0f}s)")

    # the checks (section 2): pick O's filed point, and U2s's filed frontier C, on each of the three
    i_inc = own_index(PICK_O)
    if name_o(i_inc, bud) != PICK_O:
        raise SystemExit(f"own index {i_inc} is {name_o(i_inc, bud)}, not {PICK_O}")
    fo = [p for p in vs["frontier_O"] if p.get("pick") == PICK_O and abs(p["level"] - 0.96) < 1e-9]
    if len(fo) != 1:
        raise SystemExit("U2s's frontier O has no single 0.96 point at pick O")
    for d in THREE:
        got = (float(own[d][0][i_inc]), float(own[d][1][i_inc]))
        if got != (float(fo[0]["s1sel"][d]["ALL"]), float(fo[0]["s1sel"][d]["rows"])):
            raise SystemExit(f"{d}: pick O gives {got}, not U2s's filed {fo[0]['s1sel'][d]}")
    c_checked = []
    for p in vs["frontier_C"]:
        if not p.get("pick"):
            continue
        i = index_u(p["pick"], bud)
        for d in THREE:
            got = (float(tabs[d][0][i]), float(tabs[d][1][i]))
            if got != (float(p["s1sel"][d]["ALL"]), float(p["s1sel"][d]["rows"])):
                raise SystemExit(f"{d}: U2s's frontier C point at {p['level']} gives {got}, not {p['s1sel'][d]}")
        c_checked.append((p["level"], p["pick"], i))
    log(f"checks: pick O's filed point and U2s's {len(c_checked)} filed frontier C points reproduced on the three "
        f"({time.time() - t0:.0f}s)")

    nu = len(S.GRID) * S.R * S.NO * S.NB * S.NT
    A_ = np.stack([tabs[d][0] for d in THREE], axis=1)
    RW = np.stack([tabs[d][1] for d in THREE], axis=1)
    costs, cost_r = costs_of(RW, w)
    ceil = A_.max(axis=0)
    gk, gr, _go, gb, _gt = np.unravel_index(np.arange(nu), (len(S.GRID), S.R, S.NO, S.NB, S.NT))
    form_of_b = np.asarray([S.FORMS.index(x[1]) for x in bud])
    gf = form_of_b[gb]
    no_ = S.R * S.NO * S.NB * S.NT
    A_o = np.stack([own[d][0] for d in THREE], axis=1)
    RW_o = np.stack([own[d][1] for d in THREE], axis=1)
    c_o, cr_o = costs_of(RW_o, w)
    _ok, orr, _oo, ob, _ot = np.unravel_index(np.arange(no_), (1, S.R, S.NO, S.NB, S.NT))
    l_inc = S.level_of(A_o[i_inc], ceil)
    if l_inc is None:
        raise SystemExit("pick O meets no level on the three")
    cost_inc = float(cr_o[i_inc])
    log(f"ceilings {dict(zip(THREE, ceil.tolist()))}; pick O on the three: level {l_inc}, cost {cost_inc} "
        f"({time.time() - t0:.0f}s)")

    tf = S.three_frontiers(A_, costs, cost_r, gr, A_o, c_o, cr_o, orr, ceil, CAP)
    _tg, c_k = point_at(tf["C"][0], l_inc)
    _tg, e_k = point_at(tf["E"][0], l_inc)
    c_at = None if c_k is None else float(cost_r[c_k])
    e_at = None if e_k is None else float(cost_r[e_k])
    vc, ve, carried = decide(tf["C"][1], tf["E"][1], l_inc, cost_inc, c_at, e_at)
    log(f"C {tf['C'][1]}, E {tf['E'][1]} within the cap; at {l_inc}: C {c_at}, E {e_at}; {vc}, {ve}, carried "
        f"{carried} ({time.time() - t0:.0f}s)")

    def point(t, tgt, i, own_=False):
        rec = {"level": t, "targets": dict(zip(THREE, tgt.tolist())), "pick": None}
        if i is None:
            return rec
        A_m, R_m, cr = (A_o, RW_o, cr_o) if own_ else (A_, RW, cost_r)
        rec.update({"pick": name_o(i, bud) if own_ else name_u(i, bud), "cost": float(cr[i]),
                    "within_cap": bool(cr[i] <= CAP + 1e-9),
                    "price": round(float(cr[i]) / cost_inc, 3),
                    "form": S.FORMS[int((form_of_b[ob] if own_ else gf)[i])],
                    "s1sel": {d: {"ALL": float(A_m[i, j]), "rows": float(R_m[i, j])} for j, d in enumerate(THREE)}})
        return rec

    fr = {key: {"level": lv, "points": [point(t, tg, k_, key == "O") for t, tg, k_ in pts],
                "pick": None if kk is None else point(lv, point_at(pts, lv)[0], kk, key == "O")}
          for key, (pts, lv, kk) in tf.items()}
    which = {"E": e_k, "C": c_k, "C_ROW": c_k, "E_ROW": e_k}
    if carried == "E":
        carried_rec = fr["E"]["pick"]
    elif carried == "C":
        carried_rec = fr["C"]["pick"]
    elif carried in ("C_ROW", "E_ROW"):
        carried_rec = point(l_inc, point_at(tf["C" if carried == "C_ROW" else "E"][0], l_inc)[0], which[carried])
    else:
        carried_rec = None

    # beside, deciding nothing
    tgt_inc = np.round(np.minimum(l_inc, ceil - PG.TOL), 4)
    meets = (A_ >= tgt_inc[None, :] - 1e-9).all(axis=1)

    def cheapest(mask):
        i = np.flatnonzero(meets & mask)
        return None if i.size == 0 else point(l_inc, tgt_inc, int(i[np.argmin(costs[i])]))

    by_form = {f: cheapest(gf == fi) for fi, f in enumerate(S.FORMS)}
    by_cell = {S.GRID[k][0]: cheapest(gk == k) for k in range(len(S.GRID))}
    per_dataset = {}
    for j, d in enumerate(THREE):
        per_dataset[d] = {}
        for t in S.LEVELS:
            tg = round(min(t, float(ceil[j]) - PG.TOL), 4)
            ok = A_[:, j] >= tg - 1e-9
            per_dataset[d][str(t)] = None if not ok.any() else round(float(RW[ok, j].min()), 1)
    six_c = next((p for p in vs["frontier_C"] if p.get("pick") and abs(p["level"] - 0.95) < 1e-9), None)
    six_c_rec = None
    if six_c is not None:
        i6 = index_u(six_c["pick"], bud)
        six_c_rec = {"pick": six_c["pick"], "six_cost": six_c["cost"], "cost_on_three": float(cost_r[i6]),
                     "s1sel": {d: {"ALL": float(tabs[d][0][i6]), "rows": float(tabs[d][1][i6])} for d in THREE}}
    rec = {"declared_in": DECLARED, "stage": "U2w", "part": "A", "script_sha256": script_sha256(),
           "zu2s_sha256": S.script_sha256(), "u2s_verdict_sha256": sha_file(vs_path),
           "stats_sha256": {d: rm[d]["file_sha256"] for d in THREE}, "datasets": list(THREE),
           "weights": {d: w[d] for d in THREE}, "cap": CAP, "row_price": ROW_PRICE, "candidates": nu,
           "own_candidates": no_, "ceilings": dict(zip(THREE, ceil.tolist())),
           "incumbent": {"name": PICK_O, "level": l_inc, "cost": cost_inc,
                         "s1sel": {d: {"ALL": float(A_o[i_inc, j]), "rows": float(RW_o[i_inc, j])}
                                   for j, d in enumerate(THREE)}},
           "verdict_c": vc, "verdict_e": ve, "carried": carried, "carried_pick": carried_rec,
           "at_incumbent_level": {"C": None if c_k is None else point(l_inc, tgt_inc, c_k),
                                  "E": None if e_k is None else point(l_inc, tgt_inc, e_k)},
           "frontier_C": fr["C"], "frontier_E": fr["E"], "frontier_O": fr["O"],
           "by_form": by_form, "by_cell": by_cell, "per_dataset_fewest_rows": per_dataset,
           "u2s_six_pick_C_0.95": six_c_rec,
           "checks": {"pick_o_filed_point": True, "u2s_frontier_c_points": len(c_checked),
                      "walk_and_questions": True, "weights": True},
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "choice.json", rec)
    (OUT / "choice.md").write_text(report_md(rec), encoding="utf-8")
    log(f"wrote {OUT / 'choice.json'} and choice.md ({time.time() - t0:.0f}s)")
    return rec


def report_md(rec):
    D = rec["datasets"]
    inc = rec["incumbent"]
    L = ["# U2w stage A: one pool rule on 2wiki, hotpotqa and musique", "",
         f"Declared in {rec['declared_in']}. Verdicts: **{rec['verdict_c']}**, **{rec['verdict_e']}**; carried: "
         f"**{rec['carried']}**.", "",
         f"Incumbent (pick O, per-dataset construction): `{inc['name']}`, level {inc['level']}, cost {inc['cost']} "
         f"rows on the three (P1f's weights {rec['weights']}); cap {rec['cap']}; row price bound "
         f"{rec['row_price']} x the incumbent's cost.", ""]
    cp = rec["carried_pick"]
    if cp:
        L += [f"Carried: `{cp['pick']}` ({cp['form']}), level {cp['level']}, cost {cp['cost']} rows "
              f"({cp['price']} x pick O), within the cap: {cp['within_cap']}.", ""]
    L += ["## ALL (rows) on s1sel", "", "| pools | level | cost | " + " | ".join(D) + " |",
          "|---|---:|---:|" + "---|" * len(D)]

    def row(label, p):
        if not p or not p.get("pick"):
            return f"| {label} | - | - | " + " | ".join("-" for _ in D) + " |"
        return (f"| {label} | {p['level']} | {p['cost']} | "
                + " | ".join(f"{p['s1sel'][d]['ALL']} ({p['s1sel'][d]['rows']})" for d in D) + " |")

    L.append(f"| pick O `{inc['name']}` | {inc['level']} | {inc['cost']} | "
             + " | ".join(f"{inc['s1sel'][d]['ALL']} ({inc['s1sel'][d]['rows']})" for d in D) + " |")
    L.append(row(f"carried `{cp['pick']}`" if cp else "carried", cp))
    for key in ("C", "E"):
        p = rec["at_incumbent_level"][key]
        L.append(row(f"{key}'s cheapest at {inc['level']}" + (f" `{p['pick']}`" if p else ""), p))
    s6 = rec["u2s_six_pick_C_0.95"]
    if s6:
        L.append(f"| U2s's six-dataset C at 0.950 `{s6['pick']}` | - | {s6['cost_on_three']} | "
                 + " | ".join(f"{s6['s1sel'][d]['ALL']} ({s6['s1sel'][d]['rows']})" for d in D) + " |")
    for key in ("C", "E", "O"):
        L += ["", f"## Frontier {key} (every level, past the cap too)", "",
              "| level | cost | within cap | price | pick | " + " | ".join(D) + " |",
              "|---:|---:|---|---:|---|" + "---|" * len(D)]
        for p in rec[f"frontier_{key}"]["points"]:
            if not p.get("pick"):
                L.append(f"| {p['level']} | - | - | - | none meets | " + " | ".join("-" for _ in D) + " |")
            else:
                L.append(f"| {p['level']} | {p['cost']} | {p['within_cap']} | {p['price']} | `{p['pick']}` | "
                         + " | ".join(f"{p['s1sel'][d]['ALL']} ({p['s1sel'][d]['rows']})" for d in D) + " |")
    L += ["", f"## Beside: the cheapest point at level {inc['level']} by budget form and by construction", "",
          "| group | cost | pick |", "|---|---:|---|"]
    for grp in ("by_form", "by_cell"):
        for g, p in rec[grp].items():
            L.append(f"| {g} | {p['cost'] if p else '-'} | {('`' + p['pick'] + '`') if p else 'none meets'} |")
    L += ["", "## Beside: per dataset, the fewest mean rows any universal candidate needs (upper bound only)", "",
          "| level | " + " | ".join(D) + " |", "|---:|" + "---:|" * len(D)]
    for t in rec["per_dataset_fewest_rows"][D[0]]:
        L.append(f"| {t} | " + " | ".join(str(rec["per_dataset_fewest_rows"][d][t]) for d in D) + " |")
    return "\n".join(L) + "\n"


def _selftest():
    w = {"musique": 7018, "hotpotqa": 13335, "2wiki": 18504}
    rows = np.asarray([[5522.2, 713.1, 1070.5]])
    c, cr = costs_of(rows, w)
    hand = (7018 * 5522.2 + 13335 * 713.1 + 18504 * 1070.5) / (7018 + 13335 + 18504)
    assert abs(c[0] - hand) < 1e-9 and cr[0] == round(hand, 1), (c, hand)
    # frontier and the pick within the cap, hand-built: three candidates, three datasets
    A = np.asarray([[0.950, 0.970, 0.960], [0.960, 0.965, 0.970], [0.990, 0.990, 0.990]])
    cs = np.asarray([100.0, 200.0, 5000.0])
    ceil = A.max(axis=0)
    pts = S.U2E.frontier(A, cs, ceil)
    got = {round(t, 3): k for t, _tg, k in pts}
    assert got[0.95] == 0 and got[0.955] == 1 and got[0.96] == 1 and got[0.965] == 2 and got[0.985] == 2, got
    lv, kk = M.pick_of(pts, cs, 1953.5)
    assert (round(lv, 3), kk) == (0.96, 1), (lv, kk)
    # the verdicts: each outcome on inputs built for it
    assert decide(0.965, None, 0.96, 1000.0, 900.0, None)[0] == "UNIFIED_RAISED"
    assert decide(0.96, None, 0.96, 1000.0, 900.0, None)[0] == "UNIFIED_HOLDS"
    assert decide(0.955, None, 0.96, 1000.0, 2500.0, None)[0] == "UNIFIED_AT_A_PRICE"
    assert decide(0.95, None, 0.96, 1000.0, 1999.0, None)[0] == "UNIFIED_AT_A_ROW_PRICE"
    assert decide(None, None, 0.96, 1000.0, 2001.0, None)[0] == "NOT_UNIFIED"
    assert decide(None, None, 0.96, 1000.0, None, None)[0] == "NOT_UNIFIED"
    assert decide(0.96, 0.96, 0.96, 1000.0, 900.0, 900.0)[1] == "KIND_FREE_NO_PRICE"
    assert decide(None, 0.955, 0.96, 1000.0, 3000.0, 1500.0)[1] == "KIND_FREE_NO_PRICE"
    assert decide(0.965, 0.96, 0.96, 1000.0, 900.0, 900.0)[1] == "KIND_FREE_AT_A_PRICE"
    assert decide(None, None, 0.96, 1000.0, 1900.0, 1800.0)[1] == "KIND_FREE_AT_A_ROW_PRICE"
    assert decide(None, None, 0.96, 1000.0, 1700.0, 1800.0)[1] == "KIND_FREE_TOO_DEAR"
    assert decide(None, None, 0.96, 1000.0, None, 2100.0)[1] == "KIND_FREE_TOO_DEAR"
    # the carry order: E, then C, then the cheaper row-priced point (ties to C), then nothing
    assert decide(0.96, 0.96, 0.96, 1000.0, 900.0, 900.0)[2] == "E"
    assert decide(0.96, 0.955, 0.96, 1000.0, 900.0, 950.0)[2] == "E"      # kind-free at one step's price (U2s)
    assert decide(0.96, 0.95, 0.96, 1000.0, 900.0, 950.0)[2] == "C"
    assert decide(None, None, 0.96, 1000.0, 1800.0, 1800.0)[2] == "C_ROW"
    assert decide(None, None, 0.96, 1000.0, 1900.0, 1800.0)[2] == "E_ROW"
    assert decide(None, None, 0.96, 1000.0, 2500.0, 1800.0)[2] == "E_ROW"
    assert decide(None, None, 0.96, 1000.0, 2500.0, 2600.0)[2] is None
    assert THREE == ("musique", "hotpotqa", "2wiki"), THREE
    print("selftest: PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("choose",))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
    elif a.cmd == "choose":
        choose()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
