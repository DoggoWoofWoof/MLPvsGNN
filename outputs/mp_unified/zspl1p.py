"""SPL1 amendment A8 (docs/SPL1_SPARSE_LEG_SCREEN.md section 8.8): the pool coverage of the filed pool cells with C3's SPLADE list in place of the record's,
and the decision rule D1. Imports the pinned pool modules (zpl2g, zpl5g, zpl6g) and edits none of them; in its own process it replaces only the function
that reads retrieval_cache/splade_top1000.npz. Everything else is the filed engine: the frozen dense list, graph, vectors, eval carve, cell definitions.

  selftest                       toy checks (padding, the row mapping, the paired bootstrap and calls, the decision rule)
  pool --ds D                    G-P1 on the old cell, then C3's arm on every cell; C0's arm is the filed eval confirmation grids -> outputs/spl1/pool/D.json
  call                           D1 over the three datasets and the card speed ratio -> outputs/spl1/pool/call.json and a markdown table on stdout

Usage (laptop): python -E -P -B outputs/mp_unified/zspl1p.py pool --ds squad
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zpl2g as P2  # noqa: E402
import zpl5g as Z5  # noqa: E402
import zpl6g as Z6  # noqa: E402

ROOT = P2.ROOT
DOC = "docs/SPL1_SPARSE_LEG_SCREEN.md section 8.8 (A8)"
OUT = ROOT / "outputs" / "spl1" / "pool"
SPL = ROOT / "outputs" / "spl1"
DATASETS = ("squad", "multihoprag", "musique")
BUDGETS = tuple(Z5.BUDGETS)
BOOT_N, BOOT_SEED = 1000, 0
LEVEL_PT, PRICE_PT, NODES_MAX, SPEED_MIN = -1.0, -1.5, 1.05, 3.0
LEG_FILE = "splade_top1000.npz"


# ── the replaced reader ───────────────────────────────────────────────────────────────────────────────────────

class Leg:
    """A list array addressed by the record's cache row: leg[idx] gives the rows of the questions idx (any order)."""

    def __init__(self, rows, ids):
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        self.sorted, self.order, self.ids = rows[order], order, ids
        if np.any(np.diff(self.sorted) == 0):
            P2.stop("the list's rows repeat a question")

    def __getitem__(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        pos = np.searchsorted(self.sorted, idx)
        pos[pos >= self.sorted.size] = 0
        if not np.array_equal(self.sorted[pos], idx):
            P2.stop("a question of the carve is not in the list's rows")
        return self.ids[self.order[pos]]


def fill_pad(ids, n):
    """Where a node has score zero the list holds -1: the missing node ids are appended in ascending order (section 8.8)."""
    out = np.array(ids, dtype=np.int32, copy=True)
    for q in np.flatnonzero((out < 0).any(axis=1)):
        row = out[q]
        pos = row >= 0
        have = np.zeros(n, dtype=bool)
        have[row[pos]] = True
        miss = np.flatnonzero(~have)
        k = int((~pos).sum())
        if miss.size < k:
            P2.stop("a padded row has fewer missing nodes than pads")
        row[~pos] = miss[:k]
    if (out < 0).any():
        P2.stop("padding left a -1")
    return out


_ORIG = P2._read_ids
_ACTIVE = {}


def _patched(path):
    if Path(path).name == LEG_FILE and "leg" in _ACTIVE:
        return _ACTIVE["leg"]
    return _ORIG(path)


P2._read_ids = _patched


# ── reads ────────────────────────────────────────────────────────────────────────────────────────────────────

def boot(a3, a0, rng):
    """Paired bootstrap of the mean difference (points) over questions; 1,000 resamples."""
    d = a3.astype(np.float64) - a0.astype(np.float64)
    idx = rng.integers(0, d.size, size=(BOOT_N, d.size))
    m = d[idx].mean(axis=1) * 100.0
    return float(d.mean() * 100.0), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def call_of(lo, nodes_ratio):
    if nodes_ratio > NODES_MAX:
        return "MORE_NODES"
    return "BETTER" if lo > 0 else ("LEVEL" if lo >= LEVEL_PT else "WORSE")


def cells_of(ds):
    sel5 = P2.rjson(Z5.PL5 / ds / "select.json")
    p6 = Z6.PL6 / ds / "select.json"
    if not p6.is_file():
        P2.stop(f"{ds}: PL6's select.json is not here")
    sel6 = P2.rjson(p6)
    if sel5["old"] != sel6["old"]:
        P2.stop(f"{ds}: PL5 and PL6 name different old cells")
    roles = {}
    for b in BUDGETS:
        c = sel5["frontier_picks"]["all"][str(b)]
        if c:
            roles[f"frontier_{b}"] = c["name"]
    roles["R95_bounded"] = sel6["picks"]["R95_bounded"]["name"]
    roles["old"] = sel5["old"]
    names = []
    for n in roles.values():
        if n not in names:
            names.append(n)
    return names, roles, {"pl5_select_sha256": P2.sha256_file(Z5.PL5 / ds / "select.json"), "pl6_select_sha256": P2.sha256_file(p6)}


def filed_arrays(ds, names):
    """The filed eval confirmation grids' per-question present counts and sizes (PL5's, then PL6's) for the named cells."""
    got, gc, qids = {}, None, None
    grids = [Z6.pl5_confirm_grid(ds, "frozen")]
    p6 = Z6.PL6 / ds / "grid_eval_frozen_confirm"
    grids.append(Z5.load_grid5(p6) if Path(str(p6) + ".npz").is_file() else None)
    for g in grids:
        if g is None:
            continue
        if gc is None:
            gc, qids = g["gold_counts"], g["qids"]
        elif not np.array_equal(gc, g["gold_counts"]) or qids != g["qids"]:
            P2.stop(f"{ds}: PL5's and PL6's eval grids are not on the same questions")
        for i, n in enumerate(g["names"]):
            got.setdefault(n, (g["present"][:, i].astype(np.int64), g["size"][:, i].astype(np.int64)))
    missing = [n for n in names if n not in got]
    if missing:
        P2.stop(f"{ds}: cells not in the filed eval grids: {missing}")
    return got, gc, qids


def run_arm(ds, leg, names):
    """The filed engine on the eval carve with `leg` as the SPLADE list; returns present, size (questions x cells in plan order), plan names, source."""
    _ACTIVE["leg"] = leg
    try:
        op = P2.opened()
        src = P2.load_source(ds, "eval", "frozen")
        plan = Z5.Plan5(names)
        P, S = Z5.evaluate5(op, src, plan, 0, src.nq)
    finally:
        _ACTIVE.pop("leg", None)
    return P, S, plan.names, src


def cmd_pool(a):
    ds = a.ds
    t0 = time.time()
    c3p = SPL / "c3" / ds / "top1000.npz"
    if not c3p.is_file():
        P2.stop(f"{ds}: C3's list is not here ({c3p}); fetch outputs/spl1/c3/{ds}/* first")
    with np.load(c3p) as z:
        rows, ids3 = z["rows"].astype(np.int64), z["ids"].astype(np.int32)
    names, roles, shas = cells_of(ds)
    filed, gc, qids = filed_arrays(ds, names)
    # G-P1: C0's own list through the replaced reader and the row mapping reproduces the filed old cell on every question
    import zworld
    w = zworld.load(ds)
    rec = _ORIG(Path(w.root) / "retrieval_cache" / LEG_FILE)
    P0, S0, pn0, src0 = run_arm(ds, Leg(rows, np.asarray(rec[rows], dtype=np.int32)), [roles["old"]])
    if list(src0.qids) != list(qids):
        P2.stop(f"{ds}: the carve's questions are not the filed grids'")
    fp, fs = filed[roles["old"]]
    g_p1 = {"cell": roles["old"], "questions": int(src0.nq), "present_equal": bool(np.array_equal(P0[:, 0].astype(np.int64), fp)),
            "size_equal": bool(np.array_equal(S0[:, 0].astype(np.int64), fs))}
    g_p1["pass"] = g_p1["present_equal"] and g_p1["size_equal"]
    P2.log(f"{ds}: G-P1 {'PASS' if g_p1['pass'] else 'FAIL'} on {roles['old']} ({src0.nq} questions)")
    if not g_p1["pass"]:
        P2.wjson(OUT / f"{ds}.json", {"stage": "SPL1 A8 pool read", "declared_in": DOC, "dataset": ds, "g_p1": g_p1, "verdict": "G-P1 FAILED: nothing read"})
        return 1
    ids3 = fill_pad(ids3, int(w.n)) if (ids3 < 0).any() else ids3
    P3, S3, pn3, src3 = run_arm(ds, Leg(rows, ids3), names)
    if list(src3.qids) != list(qids):
        P2.stop(f"{ds}: C3's carve differs from the filed grids'")
    rng = np.random.default_rng([BOOT_SEED, 1])
    cells = {}
    for j, n in enumerate(pn3):
        f_p, f_s = filed[n]
        a0, a3 = f_p >= gc, P3[:, j].astype(np.int64) >= gc
        d, lo, hi = boot(a3, a0, rng)
        n0, n3 = float(f_s.mean()), float(S3[:, j].mean())
        cells[n] = {"c0_all": float(a0.mean()), "c3_all": float(a3.mean()), "diff_pt": d, "lo_pt": lo, "hi_pt": hi,
                    "c0_nodes": n0, "c3_nodes": n3, "nodes_ratio": n3 / n0 if n0 else 1.0, "call": call_of(lo, n3 / n0 if n0 else 1.0)}
    np.savez(OUT / f"{ds}_c3_grid.npz", names=np.asarray(pn3), present=P3, size=S3, gold_counts=gc, qids=np.asarray(qids))
    res = {"stage": "SPL1 A8 pool read", "declared_in": DOC, "dataset": ds, "questions": int(src3.nq), "roles": roles, "g_p1": g_p1,
           "code_sha256": P2.sha256_file(Path(__file__)), "select_shas": shas, "boot": {"n": BOOT_N, "seed": BOOT_SEED},
           "cells": cells, "seconds": round(time.time() - t0, 1), "utc": P2.utc()}
    P2.wjson(OUT / f"{ds}.json", res)
    for role, n in roles.items():
        c = cells[n]
        P2.log(f"{ds} {role}: C0 {c['c0_all']:.4f} C3 {c['c3_all']:.4f} diff {c['diff_pt']:+.2f} [{c['lo_pt']:+.2f}, {c['hi_pt']:+.2f}] {c['call']} nodes {c['c0_nodes']:.1f} -> {c['c3_nodes']:.1f}")
    return 0


# ── the decision ─────────────────────────────────────────────────────────────────────────────────────────────

def decide(deciding, ratio):
    """D1 of section 8.8. deciding: list of (dataset, role, call, point difference in points)."""
    if ratio is None or ratio < SPEED_MIN:
        return "C0_STAYS", "the card speed ratio is below 3.0"
    calls = [c for _, _, c, _ in deciding]
    if all(c in ("BETTER", "LEVEL") for c in calls):
        return "ADOPT_C3", "all deciding cells BETTER or LEVEL"
    if all(c in ("BETTER", "LEVEL", "WORSE") for c in calls) and any(c == "WORSE" for c in calls) and all(d >= PRICE_PT for _, _, _, d in deciding):
        return "ADOPT_C3_AT_PRICE", "some cells WORSE, every point difference -1.5 points or better"
    return "C0_STAYS", "a deciding cell is MORE_NODES or its point difference is below -1.5 points"


def speed_ratio():
    def rate(key, sub):
        p = SPL / key / sub / "encode.json"
        return json.loads(p.read_text(encoding="utf-8")).get("block_rate_median") if p.is_file() else None
    c3, c0 = rate("c3", "squad"), rate("c0", "squad")
    if c3 and c0:
        return {"basis": "card, SQuAD block median", "c3": c3, "c0": c0, "ratio": c3 / c0}
    c3, c0 = rate("c3", "squad_smoke_cpu"), rate("c0", "squad_smoke_cpu")
    if c3 and c0:
        return {"basis": "CPU, first 512 SQuAD nodes (the card row of C0 is not here)", "c3": c3, "c0": c0, "ratio": c3 / c0}
    return {"basis": "none", "c3": c3, "c0": c0, "ratio": None}


def cmd_call(a):
    recs = {}
    for ds in DATASETS:
        p = OUT / f"{ds}.json"
        if not p.is_file():
            P2.stop(f"{ds}: no pool read yet ({p.name}); nothing is decided on fewer than three datasets")
        r = json.loads(p.read_text(encoding="utf-8"))
        if "cells" not in r:
            P2.stop(f"{ds}: the pool read did not pass G-P1")
        recs[ds] = r
    sp = speed_ratio()
    deciding = []
    lines = ["| dataset | cell | role | C0 all-gold | C3 all-gold | diff pt [95% interval] | call | nodes C0 -> C3 |", "|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        r = recs[ds]
        for role, n in r["roles"].items():
            c = r["cells"][n]
            lines.append(f"| {ds} | {n} | {role} | {c['c0_all']:.4f} | {c['c3_all']:.4f} | {c['diff_pt']:+.2f} [{c['lo_pt']:+.2f}, {c['hi_pt']:+.2f}] | {c['call']} | {c['c0_nodes']:.1f} -> {c['c3_nodes']:.1f} |")
            if role in ("frontier_2500", "R95_bounded"):
                deciding.append((ds, role, c["call"], c["diff_pt"]))
    verdict, why = decide(deciding, sp["ratio"])
    out = {"stage": "SPL1 A8 call", "declared_in": DOC, "speed": sp, "deciding": [list(x) for x in deciding], "decision": verdict, "because": why, "utc": P2.utc()}
    P2.wjson(OUT / "call.json", out)
    print("\n".join(lines))
    print(f"\nspeed ({sp['basis']}): C3 {sp['c3']} vs C0 {sp['c0']} -> ratio {sp['ratio']}")
    print(f"D1: {verdict} ({why})")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────────────────────────────────────

def cmd_selftest(a):
    ok = []
    ids = np.array([[3, 1, -1, -1, -1], [0, 1, 2, 3, 4]], dtype=np.int32)
    f = fill_pad(ids, 5)
    ok.append(("padding appends the missing ids ascending, no repeats", f[0].tolist() == [3, 1, 0, 2, 4] and sorted(f[1].tolist()) == [0, 1, 2, 3, 4]))
    leg = Leg(np.array([9, 2, 5]), np.array([[90], [20], [50]]))
    ok.append(("the row mapping answers in the asked order", leg[np.array([5, 9, 2])].ravel().tolist() == [50, 90, 20]))
    try:
        leg[np.array([4])]
        miss = False
    except SystemExit:
        miss = True
    ok.append(("a question that is not in the list's rows stops", miss))
    rng = np.random.default_rng([0, 1])
    same = boot(np.ones(500, bool), np.ones(500, bool), rng)
    worse = boot(np.zeros(500, bool), np.ones(500, bool), rng)
    better = boot(np.ones(500, bool), np.zeros(500, bool), rng)
    ok.append(("bootstrap: equal arrays give 0 [0, 0]; all-worse -100; all-better +100", same == (0.0, 0.0, 0.0) and worse[0] == -100.0 and better[0] == 100.0))
    ok.append(("calls: BETTER, LEVEL, WORSE, MORE_NODES", [call_of(0.5, 1.0), call_of(-0.9, 1.0), call_of(-1.1, 1.0), call_of(2.0, 1.2)] == ["BETTER", "LEVEL", "WORSE", "MORE_NODES"]))
    d = [("a", "x", "LEVEL", -0.4)] * 6
    w = d[:5] + [("a", "x", "WORSE", -1.2)]
    ww = d[:5] + [("a", "x", "WORSE", -1.8)]
    mn = d[:5] + [("a", "x", "MORE_NODES", -0.2)]
    ok.append(("D1: ADOPT_C3, AT_PRICE, C0_STAYS (-1.8), C0_STAYS (MORE_NODES), C0_STAYS (speed 2.0), C0_STAYS (no speed)",
               [decide(d, 6.0)[0], decide(w, 6.0)[0], decide(ww, 6.0)[0], decide(mn, 6.0)[0], decide(d, 2.0)[0], decide(d, None)[0]]
               == ["ADOPT_C3", "ADOPT_C3_AT_PRICE", "C0_STAYS", "C0_STAYS", "C0_STAYS", "C0_STAYS"]))
    for n, v in ok:
        print(("PASS " if v else "FAIL ") + n)
    print(f"{sum(v for _, v in ok)}/{len(ok)} checks pass")
    return 0 if all(v for _, v in ok) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    p = sub.add_parser("pool")
    p.add_argument("--ds", required=True, choices=DATASETS)
    p.set_defaults(fn=cmd_pool)
    sub.add_parser("call").set_defaults(fn=cmd_call)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
