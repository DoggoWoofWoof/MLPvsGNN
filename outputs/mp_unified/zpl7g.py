"""PL7 (docs/PL7_COVERAGE_RESTRICTION.md): round 6 of the coverage target. The same rule on all seven, a node ceiling K walked back DOWN from PL6's unbounded pick to 2,500.

    python outputs/mp_unified/zpl7g.py cells [--write]                     # the declared cells per dataset and their sha256
    python outputs/mp_unified/zpl7g.py selftest [--ds D ...] [--n 1]       # cells, parse, pinned hashes, the engine against a per-node reference, the hard cuts' sizes
    python outputs/mp_unified/zpl7g.py grid --ds D [--part I K] [--host]   # dev grid (frozen lists) of the dataset's PL7 cells; datasets with no PL7 cells have no grid
    python outputs/mp_unified/zpl7g.py select --ds D                        # the rule at every ceiling on dev, over PL6's table plus the PL7 cells; G-C (controls) first
    python outputs/mp_unified/zpl7g.py confirm --ds D [--part I K] [--host] # eval: only the cells the picks need that PL5's and PL6's confirmation grids do not hold
    python outputs/mp_unified/zpl7g.py call                                 # eval, per dataset and ceiling: the pick, its price, its status against the bounded pick and against PL6's pick
    python outputs/mp_unified/zpl7g.py rows [--confirm] [--ds D ...] [--write]   # the host rows

This module imports zpl6g.py (PL6's engine), which imports zpl5g.py to zpl2g.py; it edits none of them. It adds ONE thing to the engine: the visited budget V = 2,500 (the h2 and h3 c100
walks and the h2 and h3 budget-following walks) to the settings table, after the earlier hashes are cached. Every other cell is a name the earlier engine already parses (a walk at a V
it already holds, with a `+top{M}` size that no earlier round used). Every pool function takes and reads no labels; everything is deterministic.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zpl6g as Z6  # noqa: E402

Z5, P2, Z3, Z4 = Z6.Z5, Z6.P2, Z6.Z3, Z6.Z4
DOC = "docs/PL7_COVERAGE_RESTRICTION.md"
PL5, PL6 = Z6.PL5, Z6.PL6
PL7 = ROOT / "outputs" / "pl7"
SEVEN = Z6.SEVEN
log, stop, rel, rjson, wjson, utc = P2.log, P2.stop, P2.rel, P2.rjson, P2.wjson, P2.utc
walk_name = Z6.walk_name

# ── the declared restriction ladder (docs/PL7_COVERAGE_RESTRICTION.md sections 3 to 5) ───────

TARGET = 0.95
BESIDE_FLOOR = 0.98
END = 2500                                                  # the user's 25 Nov ruling: the ladder walks down to here, and the carry is read here only
LADDER = (2500, 3500, 5000, 7000, 10000, 14000, 20000, 28000, 40000)   # 2,500 x 2^(j/2) rounded to 500
VSET = (2500, 5000, 10000, 20000, 40000)                     # visited budgets: 2,500 x 2^j
BASES7 = Z6.BASES6
BASE_B = {"rrf1000_b25": 25, "rrf1000_b400": 400, "rrf1000_b1600": 1600}
# What PL6's filed dev selects said (read before this page was written): per dataset that stayed above 2,500 under rule U, the visited budget of the walk of PL6's U95 pick and its dev mean nodes.
READ6 = {"musique": (20000, 15400.0), "webqsp": (40000, 39571.6), "metaqa": (5000, 4983.6)}
ROWS_PARTS = 3


def settings7():
    out = {}
    for h in (2, 3):
        out[f"h{h}_c100_v{END}"] = {"name": f"h{h}_c100_v{END}", "hops": h, "per_seed_cap": 100, "per_frontier_cap": 100, "visited_cap": END}
        out[Z5.bf_name(h, END)] = {"name": Z5.bf_name(h, END), "hops": h, "per_seed_cap": Z6.WINDOW, "per_frontier_cap": Z6.WINDOW, "visited_cap": END, "bf": True, "window": Z6.WINDOW}
    return out


SET7 = settings7()


def walks7(v):
    """PL6's 13 walks (docs PL6 section 4) at any visited budget V."""
    c2, c3, b2, b3 = f"h2_c100_v{v}", f"h3_c100_v{v}", Z5.bf_name(2, v), Z5.bf_name(3, v)
    W = walk_name
    return [W("STRUCT", c3), W("STRUCT", c3, ("cs", "sp", "eq")), W("FULL", c3, ("cs", "sp", "eq")), W("FULL", c3, ("cs", "sp")), W("FULL", c3, ("cs", "vr", "eq")),
            W("FULL", c2, ("cs", "vr")), W("FULL", c2, ("s10", "cs", "vr")), W("FULL", c2, ("s20", "cs", "vr")), W("FULL", c2, ("s20", "cs", "rk")),
            W("FULL", b3, ("cs", "vr")), W("FULL", b3, ("cs", "fu")), W("STRUCT", b3, ("cs", "fu")), W("FULL", b2, ("s10", "cs", "vr"))]


def plan7(ds):
    """(visited budgets, ceilings above 2,500 that this dataset's ladder has) for a dataset, from PL6's U95 pick: V up to the pick's walk's budget, K below the pick's mean nodes."""
    if ds not in READ6:
        return [], []
    v_max, n_pick = READ6[ds]
    return [v for v in VSET if v <= v_max], [k for k in LADDER if k < n_pick]


def ceilings(ds):
    """Every ceiling the dataset is read at: 2,500 always (the end of the ladder), and each ladder step below PL6's pick."""
    return sorted({END} | set(plan7(ds)[1]))


def cell_list7(ds):
    """Per dataset: every walk x base as a union (at V = 2,500 new; at V >= 5,000 a control that an earlier round already holds where it does) and as the HARD cut `+top{K - b}`
    (the first K - b nodes of the walk's cosine order, b the base's budget, so the union has at most K nodes) for every ceiling K < V."""
    vs, ks = plan7(ds)
    out = []
    for v in vs:
        for w in walks7(v):
            for b in BASES7:
                out.append(f"{b}+{w}")
                for k in ks:
                    m = k - BASE_B[b]
                    if k < v and m >= 1:
                        out.append(f"{b}+{w}+top{m}")
    return out


EXPECT7 = {"musique": (624, "373ff76a51f4479b57b653b28606ffd8c451286dac091028d73ee659e949b19c"),
           "webqsp": (975, "ebe7442dd61ec5260f351da9b38c7fb6e4a228755ea9052127f9ea0e62fcfc7e"),
           "metaqa": (156, "0e9ff2e94427d323aaa5810aa024889bf57e42fd9674f949fa6ae5a5d7d0f9c4")}    # fixed at the declaration commit: dataset -> (cells, cells_sha256)


def constants7():
    return {"end": END, "ladder": list(LADDER), "vset": list(VSET), "bases": list(BASES7), "base_b": BASE_B, "read6": {k: list(v) for k, v in READ6.items()},
            "settings": {k: SET7[k] for k in sorted(SET7)}, "walks": {str(v): walks7(v) for v in VSET}, "target": TARGET, "beside_floor": BESIDE_FLOOR, "boot": [P2.BOOT_N, P2.BOOT_SEED],
            "parts": ROWS_PARTS}


# ── the one change to the engine ─────────────────────────────────────────────

_PREP7 = {}


def prepare7(host=False):
    """PL6's prepare (every earlier hash cached, then its settings and parser), PL6's own hash cached, THEN V = 2,500 added to the same tables. Idempotent."""
    prev = Z6.prepare(host)
    if _PREP7:
        return _PREP7
    _PREP7.update(prev)
    _PREP7["pl6"] = Z6.code_path_sha256()
    _PREP7["pl6_script"] = Z6.script_sha256()
    for k, v in SET7.items():
        Z6.SET6[k] = v                                       # the parser PL6 installed reads this table
        Z5.SET5[k] = v
    return _PREP7


_SHA = {}


def code_path_sha256():
    if "code" not in _SHA:
        h = hashlib.sha256()
        for fn in (settings7, walks7, plan7, ceilings, cell_list7, pick7, restricted_picks):
            h.update(inspect.getsource(fn).replace("\r\n", "\n").encode("utf-8"))
        h.update(json.dumps(constants7(), sort_keys=True).encode("utf-8"))
        h.update(_PREP7["pl6"].encode("utf-8"))
        _SHA["code"] = h.hexdigest()
    return _SHA["code"]


def script_sha256():
    if "script" not in _SHA:
        _SHA["script"] = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return _SHA["script"]


# ── the rule (docs section 4) ────────────────────────────────────────────────

def pick7(rows, floor, k):
    """R7(K): over the cells whose dev mean nodes are at most K, the smallest mean nodes with dev ALL >= floor (ties: higher ALL, then the name); if none reaches it, the highest ALL
    (ties: fewer nodes, then the name). Returns (cell, reached) or (None, False) when no cell is that small."""
    within = [r for r in rows if r["nodes"] <= k]
    if not within:
        return None, False
    return Z6.pick_rule(within, floor)


def restricted_picks(table_all, table_old, ks):
    """Per ceiling: R95 and R98 over every cell (`all`) and over PL2 to PL6's cells alone (`old`: what the PL7 cells add)."""
    out = {}
    for k in ks:
        rec = {}
        for fl, tag in ((TARGET, "R95"), (BESIDE_FLOOR, "R98")):
            for nm, tb in (("all", table_all), ("old", table_old)):
                c, reached = pick7(tb, fl, k)
                rec[f"{tag}_{nm}"] = None if c is None else {**c, "reached": reached, "floor": fl}
        out[str(k)] = rec
    return out


# ── commands: cells, gates ───────────────────────────────────────────────────

def earlier_names():
    return set(Z5.cell_list_union5()) | set(Z6.cell_list6_all())


def cmd_cells(a):
    prior = earlier_names()
    out, allnames = {}, []
    for ds in SEVEN:
        names = cell_list7(ds)
        allnames += names
        ctrl = [n for n in names if n in prior]
        out[ds] = {"visited": plan7(ds)[0], "ceilings": ceilings(ds), "cells": len(names), "distinct": len(set(names)), "sha256": P2.cells_sha256(names) if names else None,
                   "controls_in_pl2_pl6": len(ctrl), "new": len(names) - len(ctrl), "walks": len({n.split("+")[1] for n in names})}
        log(f"{ds}: visited {out[ds]['visited']}, ceilings {out[ds]['ceilings']}, {len(names)} cells ({len(set(names))} distinct), {out[ds]['controls_in_pl2_pl6']} already held by PL2 to PL6 (controls), "
            f"{out[ds]['new']} new, {out[ds]['walks']} walks, sha256 {out[ds]['sha256']}")
    parsed = 0
    for n in sorted(set(allnames)):
        Z5.parse5(n)
        parsed += 1
    log(f"all {parsed} distinct cell names parse")
    if a.write:
        PL7.mkdir(parents=True, exist_ok=True)
        wjson(PL7 / "cells.json", {"stage": "PL7 cells", "declared_in": DOC, "datasets": out, "cells": {ds: cell_list7(ds) for ds in SEVEN if cell_list7(ds)}, "constants": constants7(), "utc": utc()})
        log("wrote outputs/pl7/cells.json")
    return 0


def require_cells7(ds):
    names = cell_list7(ds)
    if not names:
        stop(f"{ds}: PL7 declares no cells for this dataset (its pick is inside 2,500)")
    if ds in EXPECT7:
        n, sha = EXPECT7[ds]
        if len(names) != n or len(set(names)) != n or P2.cells_sha256(names) != sha:
            stop(f"the PL7 {ds} cell list is not the declared one")
    return names


def gates_passed7():
    st = PL7 / "selftest.json"
    if not st.is_file() or rjson(st)["verdict"] != "PASS":
        stop("outputs/pl7/selftest.json is not PASS: run `zpl7g.py selftest` first")


def grid_path7(d, carve, first, tag=""):
    return PL7 / d / f"grid_{carve}_{first}{('_' + tag) if tag else ''}"


def save_grid7(base, src, plan, P, S, lo, hi, seconds, part=None, stage="PL7 grid"):
    p = Path(str(base) + (f".part{part[0]}of{part[1]}" if part else "") + ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez(tmp, names=np.asarray(plan.names, dtype=str), present=P, size=S, qids=np.asarray(src.qids[lo:hi], dtype=str),
             gold_counts=src.gold_counts[lo:hi].astype(np.int32), lo=np.asarray(lo), hi=np.asarray(hi))
    os.replace(tmp, p)
    head = {"stage": stage, "declared_in": DOC, "dataset": src.name, "carve": src.carve, "first": src.first, "questions": int(hi - lo), "rows": [int(lo), int(hi)],
            "part": list(part) if part else None, "cells": plan.C, "cells_sha256": P2.cells_sha256(plan.names), "n_nodes": int(src.n), "population_digest": src.pop_digest,
            "code_path_sha256": code_path_sha256(), "pinned": dict(_PREP7), "script_sha256": script_sha256(),
            "constants_sha256": hashlib.sha256(json.dumps(constants7(), sort_keys=True).encode()).hexdigest(), "npz": rel(p), "npz_sha256": P2.sha256_file(p),
            "seconds": round(seconds, 1), "peak_gb": P2.peak_gb(), "host": P2.HOST, "utc": utc()}
    wjson(Path(str(p)[:-4] + ".json"), head)
    return p, head


def cmd_grid(a):
    t0 = time.time()
    gates_passed7()
    names = require_cells7(a.ds)
    plan = Z5.Plan5(names)
    op = P2.opened()
    src = P2.load_source(a.ds, "dev", "frozen")
    lo, hi, part = 0, src.nq, None
    if a.part:
        i, k = int(a.part[0]), int(a.part[1])
        lo, hi = i * src.nq // k, (i + 1) * src.nq // k
        part = (i, k)
    log(f"{a.ds}/dev/frozen: {src.nq} questions, rows {lo}..{hi}, {plan.C} cells, {len(plan.walks)} walks, graph {src.graph_kind} ({P2.peak_gb()} GB)")
    P, S = Z5.evaluate5(op, src, plan, lo, hi)
    p, head = save_grid7(grid_path7(a.ds, "dev", "frozen"), src, plan, P, S, lo, hi, time.time() - t0, part)
    log(f"{a.ds}/dev/frozen: filed {rel(p)} in {head['seconds']} s")
    return 0


# ── select: G-C (controls) and the rule at every ceiling ─────────────────────

def controls_arrays(ds):
    """Every cell PL2 to PL6 filed an array for (dev, frozen lists): name -> (present, size) columns, for G-C."""
    g, _ = Z5.union_dev5(ds)
    out = {"qids": g["qids"], "gold_counts": g["gold_counts"], "cols": {n: i for i, n in enumerate(g["names"])}, "present": g["present"], "size": g["size"]}
    extra = []
    for r in Z6.RUNGS:
        base = Z6.grid_path6(ds, r, "dev", "frozen")
        if Z5.heads_of(base):
            extra.append(Z5.load_grid5(base))
    return out, extra


def gc_check(ds, g7):
    """G-C: every PL7 cell whose name an earlier round filed equals the filed arrays element for element, on every dev question."""
    base, extra = controls_arrays(ds)
    if g7["qids"] != base["qids"]:
        stop(f"{ds}: PL7's dev grid is not on PL5's dev questions")
    idx7 = {n: i for i, n in enumerate(g7["names"])}
    compared, bad, first = 0, 0, None
    srcs = [("pl2-pl5", base["cols"], base["present"], base["size"])] + [(f"pl6-r{r}", {n: i for i, n in enumerate(g["names"])}, g["present"], g["size"]) for r, g in zip(Z6.RUNGS, extra)]
    seen = set()
    for tag, cols, pres, siz in srcs:
        for n, i in idx7.items():
            if n in cols and (tag, n) not in seen:
                seen.add((tag, n))
                compared += 1
                if not (np.array_equal(g7["present"][:, i], pres[:, cols[n]]) and np.array_equal(g7["size"][:, i], siz[:, cols[n]])):
                    bad += 1
                    first = first or n
    return {"cells_compared": compared, "questions": len(g7["qids"]), "elements": compared * len(g7["qids"]) * 2, "differing_cells": bad, "first_differing": first,
            "verdict": "PASS" if bad == 0 and compared > 0 else "FAIL"}


def cmd_select(a):
    P2.HOST = bool(getattr(a, "host", False))
    ds = a.ds
    gates_passed7()
    p6 = PL6 / ds / "select.json"
    if not p6.is_file():
        stop(f"{ds}: PL6's select.json is not here")
    s6 = rjson(p6)
    table6 = [dict(x) for x in s6["table"]]
    have = {x["name"] for x in table6}
    new_rows, heads, gc = [], [], None
    names = cell_list7(ds)
    if names:
        base = grid_path7(ds, "dev", "frozen")
        hs = Z5.heads_of(base)
        if not hs:
            stop(f"{ds}: no PL7 dev grid heads")
        for hp in hs:
            hd = rjson(hp)
            if hd["code_path_sha256"] != code_path_sha256() or hd["cells_sha256"] != P2.cells_sha256(names):
                stop(f"G-U: {hp.name} was written by other PL7 code or another cell list than the one that is running")
        g7 = Z5.load_grid5(base)
        if g7["names"] != names:
            stop(f"{ds}: the PL7 dev grid does not name the declared cells in the declared order")
        gc = gc_check(ds, g7)
        log(f"G-C {ds}: {gc['verdict']} ({gc['cells_compared']} control cells x {gc['questions']} questions, {gc['differing_cells']} differ)")
        if gc["verdict"] != "PASS":
            wjson(PL7 / ds / "gc_FAIL.json", gc)
            stop(f"{ds}: G-C failed (first differing cell {gc['first_differing']}): the PL7 engine does not reproduce a cell an earlier round filed; nothing is selected")
        rows = Z5.rows_of_grid5(g7, ds, "dev")
        for n, rw in zip(g7["names"], rows):
            if n in have:
                continue
            new_rows.append({"name": n, "family": "pl7", "tags": ["pl7"], "nodes": rw["candidates_mean"], "all": rw["all_gold_at_pool"], "any": rw["any_gold_at_pool"],
                             "coverage": rw["coverage_micro"], "p95": rw["candidates_p95"], "max": rw["candidates_max"]})
        heads = [{"file": rel(hp), "code_path_sha256": rjson(hp)["code_path_sha256"], "npz_sha256": rjson(hp)["npz_sha256"], "seconds": rjson(hp)["seconds"], "peak_gb": rjson(hp)["peak_gb"],
                  "host": rjson(hp)["host"]} for hp in hs]
    table = table6 + new_rows
    ks = ceilings(ds)
    out = {"stage": "PL7 select", "declared_in": DOC, "dataset": ds, "carve": "dev", "first": "frozen", "questions": s6["questions"], "ceilings": ks, "rule": "R7(K): R95 over the cells within K dev nodes",
           "pl6_select_sha256": P2.sha256_file(p6), "code_path_sha256": code_path_sha256(), "pinned": dict(_PREP7), "script_sha256": script_sha256(), "utc": utc(), "grid_heads": heads,
           "n_nodes": s6.get("n_nodes") or (rjson(Z5.heads_of(grid_path7(ds, "dev", "frozen"))[0])["n_nodes"] if heads else None), "g_c": gc,
           "new_cells": len(new_rows), "pl7_new_cells_dev_best_all": max([r["all"] for r in new_rows], default=None)}
    out["picks"] = restricted_picks(table, [t for t in table if t["family"] != "pl7"], ks)
    out["u_pick"] = s6["picks"]["R95_all"]                                   # PL6's rule U (no bound), the reference every ceiling is read against
    out["u_pick98"] = s6["picks"]["R98_all"]
    out["incumbent"] = s6["incumbent"]                                        # PL5's bounded R95 (what PL2 to PL5 carried)
    out["old"] = s6["old"]
    out["frontier"] = {"all": P2.frontier(table), "old": P2.frontier([t for t in table if t["family"] != "pl7"])}
    out["table_pl7"] = new_rows
    wjson(PL7 / ds / "select.json", out)
    for k in ks:
        c = out["picks"][str(k)]["R95_all"]
        o = out["picks"][str(k)]["R95_old"]
        log(f"{ds} K={k}: R95_all {c['name']}  ALL {c['all']:.4f} at {c['nodes']:.1f}{'' if c['reached'] else '  (SHORT)'};  without the PL7 cells {o['all']:.4f} at {o['nodes']:.1f}")
    return 0


# ── confirm and call ─────────────────────────────────────────────────────────

def needed_names(sel):
    names = []

    def add(n):
        if n and n not in names:
            names.append(n)
    for k in sel["ceilings"]:
        for key in ("R95_all", "R95_old", "R98_all"):
            c = sel["picks"][str(k)][key]
            add(c["name"] if c else None)
    add(sel["incumbent"]["name"])
    add(sel["u_pick"]["name"])
    add(sel["u_pick98"]["name"])
    add(sel["old"])
    return names


def eval_grids(ds, first="frozen"):
    out = []
    for base in (Z5.grid_path5(ds, "eval", first, "confirm"), PL6 / ds / f"grid_eval_{first}_confirm", PL7 / ds / f"grid_eval_{first}_confirm"):
        if Z5.heads_of(base) or Path(str(base) + ".npz").is_file():
            out.append(Z5.load_grid5(base))
    return out


def eval_arrays(ds, first="frozen"):
    out, gc = {}, None
    for g in eval_grids(ds, first):
        if gc is None:
            gc = g["gold_counts"]
        elif not np.array_equal(gc, g["gold_counts"]):
            stop(f"{ds}/{first}: the confirmation grids of PL5, PL6 and PL7 are not on the same questions")
        for i, n in enumerate(g["names"]):
            out.setdefault(n, ((g["present"][:, i].astype(np.int64) == g["gold_counts"]), g["size"][:, i].astype(np.float64)))
    return out


def cmd_confirm(a):
    P2.HOST = bool(a.host)
    gates_passed7()
    ds = a.ds
    sel = rjson(PL7 / ds / "select.json")
    if sel["code_path_sha256"] != code_path_sha256():
        stop("G-U: select.json was written by other PL7 code than the one that is running")
    names = needed_names(sel)
    have = set()
    for g in eval_grids(ds, "frozen"):
        have |= set(g["names"])
    missing = [n for n in names if n not in have]
    log(f"{ds}/eval/frozen: {len(names)} cells needed, {len(names) - len(missing)} already in PL5's or PL6's confirmation grids, {len(missing)} to build")
    if not missing:
        return 0
    t0 = time.time()
    op = P2.opened()
    plan = Z5.Plan5(missing)
    src = P2.load_source(ds, "eval", "frozen")
    lo, hi, part = 0, src.nq, None
    if a.part:
        i, k = int(a.part[0]), int(a.part[1])
        lo, hi = i * src.nq // k, (i + 1) * src.nq // k
        part = (i, k)
    log(f"{ds}/eval/frozen: {src.nq} questions, rows {lo}..{hi}, {plan.C} cells, {len(plan.walks)} walks")
    P, S = Z5.evaluate5(op, src, plan, lo, hi)
    p, head = save_grid7(PL7 / ds / "grid_eval_frozen_confirm", src, plan, P, S, lo, hi, time.time() - t0, part, stage="PL7 confirm grid")
    log(f"{ds}/eval/frozen: filed {rel(p)} in {head['seconds']} s")
    return 0


def cmd_call(a):
    out = {"stage": "PL7 call", "declared_in": DOC, "utc": utc(), "datasets": {}}
    codes = {}
    for di, ds in enumerate(SEVEN):
        sp = PL7 / ds / "select.json"
        if not sp.is_file():
            out["datasets"][ds] = {"status": "NOT_READ", "why": "no PL7 select file here"}
            continue
        sel = rjson(sp)
        arr = eval_arrays(ds)
        need = needed_names(sel)
        miss = [n for n in need if n not in arr]
        if miss:
            out["datasets"][ds] = {"status": "NOT_READ", "why": f"{len(miss)} eval cells missing, e.g. {miss[0]}"}
            continue
        codes[ds] = sel["code_path_sha256"]
        inc, upick = sel["incumbent"]["name"], sel["u_pick"]["name"]
        nq = len(arr[inc][0])
        n_nodes = sel.get("n_nodes")

        def cmp(nn, on, floor, kk, tag):
            rng = np.random.default_rng([P2.BOOT_SEED, di, int(kk), tag])
            new, old = arr[nn], arr[on]
            pr = P2.paired(new, old, rng)
            nr, orr = bool(new[0].mean() >= floor), bool(old[0].mean() >= floor)
            all_status = "BELOW" if pr["all_diff_ci"][1] < 0 else ("ABOVE" if pr["all_diff_ci"][0] > 0 else "LEVEL")
            return {"status": Z3.status_of(pr, nr, orr), "all_status": all_status, "new": nn, "old": on, "same_cell": nn == on, "new_all": float(new[0].mean()), "old_all": float(old[0].mean()),
                    "new_nodes": float(new[1].mean()), "old_nodes": float(old[1].mean()), "new_reaches": nr, "old_reaches": orr, **pr}
        per_k = {}
        for k in sel["ceilings"]:
            pk = sel["picks"][str(k)]
            nn = pk["R95_all"]["name"]
            e = arr[nn]
            kk = int(e[0].sum())
            lo_w, hi_w = P2.wilson(kk, nq)
            rec = {"ceiling": k, "pick": nn, "dev_all": pk["R95_all"]["all"], "dev_nodes": pk["R95_all"]["nodes"], "dev_reached": bool(pk["R95_all"]["reached"]),
                   "eval_all": float(e[0].mean()), "eval_all_wilson": [lo_w, hi_w], "eval_reached": bool(e[0].mean() >= TARGET), "eval_nodes": float(e[1].mean()),
                   "eval_p95": float(np.percentile(e[1], 95)), "eval_max": float(e[1].max()), "share_of_graph": (float(e[1].mean() / n_nodes) if n_nodes else None),
                   "nodes_over_2500": float(e[1].mean() / END),
                   "at_floor": bool(e[0].mean() >= TARGET and lo_w < TARGET),
                   "vs_bounded": cmp(nn, inc, TARGET, k, 1), "vs_u": cmp(nn, upick, TARGET, k, 2),
                   "vs_without_pl7_cells": cmp(nn, pk["R95_old"]["name"], TARGET, k, 3)}
            c98 = pk["R98_all"]
            if c98:
                rec["r98"] = {"pick": c98["name"], "dev_all": c98["all"], "dev_nodes": c98["nodes"], "eval_all": float(arr[c98["name"]][0].mean()), "eval_nodes": float(arr[c98["name"]][1].mean())}
            per_k[str(k)] = rec
        knee = next((k for k in sel["ceilings"] if per_k[str(k)]["eval_reached"]), None)
        end = per_k[str(END)]
        u_e = arr[upick]
        out["datasets"][ds] = {"status": end["vs_bounded"]["status"], "ceilings": sel["ceilings"], "per_ceiling": per_k, "knee_ceiling": knee,
                               "knee_eval_nodes": per_k[str(knee)]["eval_nodes"] if knee else None, "knee_over_2500": (per_k[str(knee)]["eval_nodes"] / END if knee else None),
                               "u_pick": upick, "u_eval_all": float(u_e[0].mean()), "u_eval_nodes": float(u_e[1].mean()), "u_dev_nodes": sel["u_pick"]["nodes"],
                               "end_all_cost_vs_u": end["vs_u"]["all_diff"], "end_all_cost_ci": end["vs_u"]["all_diff_ci"], "end_nodes_ratio_vs_u": end["vs_u"]["nodes_ratio"],
                               "end_reaches_eval": end["eval_reached"], "universal_same_as_bounded": (bool(end["pick"] == inc) if sel["incumbent"]["reached"] else None),
                               "bounded_pick_reached_dev": bool(sel["incumbent"]["reached"]), "n_nodes": n_nodes, "eval_questions": nq}
    read = {k: v for k, v in out["datasets"].items() if v["status"] != "NOT_READ"}
    out["datasets_read"] = sorted(read)
    if len(set(codes.values())) > 1:
        stop(f"G-U: the code-path hash differs between datasets: {codes}")
    n_below = sum(v["status"] == "BELOW" for v in read.values())
    n_above = sum(v["status"] == "ABOVE" for v in read.values())
    universal = all(v["universal_same_as_bounded"] for v in read.values() if v["bounded_pick_reached_dev"])
    complete = len(read) == len(SEVEN)
    out["call"] = ("CARRY_PL7" if (complete and n_below == 0 and n_above >= 1 and universal) else ("INCOMPLETE" if not complete else "NOT_CARRIED"))
    out["counts"] = {"BELOW": n_below, "ABOVE": n_above, "LEVEL": len(read) - n_below - n_above}
    out["universality_check"] = {"datasets_that_met_the_target_inside_the_bound": sorted(k for k, v in read.items() if v["bounded_pick_reached_dev"]), "unchanged_on_all_of_them": bool(universal)}
    out["code_path_equal_everywhere"] = len(set(codes.values())) <= 1
    PL7.mkdir(parents=True, exist_ok=True)
    wjson(PL7 / "call.json", out)
    for ds, v in out["datasets"].items():
        if v["status"] == "NOT_READ":
            log(f"call {ds}: NOT_READ ({v['why']})")
            continue
        e = v["per_ceiling"][str(END)]
        log(f"call {ds} at K={END}: {v['status']}  NEW {e['pick']} eval ALL {e['eval_all']:.4f} @ {e['eval_nodes']:.0f}; bounded pick ALL {e['vs_bounded']['old_all']:.4f} @ {e['vs_bounded']['old_nodes']:.0f}; "
            f"knee K {v['knee_ceiling']}")
    log(f"call: {out['call']} (BELOW {n_below}, ABOVE {n_above}, read {len(read)}/7)")
    return 0


# ── host rows ────────────────────────────────────────────────────────────────

ROW_CODE = ("docs/PL7_COVERAGE_RESTRICTION.md", "outputs/mp_unified/zpl7g.py", "outputs/mp_unified/zpl6g.py", "outputs/mp_unified/zpl5g.py", "outputs/mp_unified/zpl4g.py",
            "outputs/mp_unified/zpl3g.py", "outputs/mp_unified/zpl2g.py", "outputs/mp_unified/zpkoh.py", "outputs/mp_unified/zpko.py", "outputs/host_ops/retry_cmd.py",
            "outputs/host_ops/pylib_run.py", "outputs/host_ops/pylib/numba067_llvmlite049_cp313_win_amd64.zip")
ROW_MEM = {"musique": 3, "webqsp": 10, "metaqa": 3}
ROW_RUN = "python outputs/host_ops/retry_cmd.py --tries 2 --wait 90 -- python outputs/host_ops/pylib_run.py outputs/mp_unified/zpl7g.py"


def row(name, deps, cpus, mem, inputs, cmd):
    return "|".join(["line", "x", ",".join(deps), name, str(cpus), str(mem), ";".join(inputs), f"{ROW_RUN} {cmd}"])


def cmd_rows(a):
    rows = []
    ds_list = a.ds or [d for d in SEVEN if cell_list7(d)]
    for ds in ds_list:
        if a.confirm:
            ins = list(ROW_CODE) + ["outputs/pl7/selftest.json", f"outputs/pl7/{ds}/select.json"]
            for i in range(ROWS_PARTS):
                rows.append(row(f"pl7-{ds}cf{i}", [], 2, ROW_MEM[ds], ins, f"confirm --ds {ds} --part {i} {ROWS_PARTS} --host"))
        else:
            ins = list(ROW_CODE) + ["outputs/pl7/selftest.json"]
            for i in range(ROWS_PARTS):
                rows.append(row(f"pl7-{ds}g{i}", [], 2, ROW_MEM[ds], ins, f"grid --ds {ds} --part {i} {ROWS_PARTS} --host"))
    nl = chr(10)
    text = nl.join(rows)
    print(text)
    if a.write:
        PL7.mkdir(parents=True, exist_ok=True)
        with open(PL7 / ("feed_rows_confirm.txt" if a.confirm else "feed_rows.txt"), "a", encoding="utf-8", newline=nl) as f:
            f.write(text + nl)
        log(f"appended {len(rows)} rows")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def cmd_selftest(a):
    t0 = time.time()
    res = {"stage": "PL7 selftest", "declared_in": DOC, "utc": utc(), "script_sha256": script_sha256(), "code_path_sha256": code_path_sha256()}
    cells = {"datasets": {}, "problems": []}
    prior = earlier_names()
    for ds in SEVEN:
        names = cell_list7(ds)
        cells["datasets"][ds] = {"n": len(names), "distinct": len(set(names)), "sha256": P2.cells_sha256(names) if names else None, "controls": sum(n in prior for n in names)}
        if len(set(names)) != len(names):
            cells["problems"].append(f"{ds}: repeated names")
        if ds in EXPECT7 and (len(names), P2.cells_sha256(names)) != tuple(EXPECT7[ds]):
            cells["problems"].append(f"{ds}: differs from the declared list")
        for n in names:
            try:
                Z5.parse5(n)
            except Exception as e:
                cells["problems"].append(f"{n}: {e}")
        # every cut is hard: the union of the base and the first K - b nodes has at most K nodes, so M + b is a ceiling on the pool
        for n in names:
            if "+top" in n:
                m = int(n.rsplit("+top", 1)[1])
                b = BASE_B[n.split("+", 1)[0]]
                if m + b not in ceilings(ds):
                    cells["problems"].append(f"{n}: M + b = {m + b} is not a ceiling of {ds}")
    for v in (10000, 20000, 40000):                          # PL7's walk list at a PL6 budget is PL6's
        r = {Z6.vee(x): x for x in Z6.RUNGS}[v]
        if walks7(v) != Z6.walks6(r):
            cells["problems"].append(f"walks7({v}) differs from PL6's rung {r} walks")
    cells["verdict"] = "PASS" if not cells["problems"] else "FAIL"
    res["cells"] = cells
    pin = {"problems": []}
    prev = rjson(PL5 / "selftest.json")
    if _PREP7["pl5"] != prev["code_path_sha256"]:
        pin["problems"].append("PL5's code-path hash differs from the one filed at its selftest")
    if _PREP7["pl5_script"] != prev["script_sha256"]:
        pin["problems"].append("zpl5g.py differs from the file that was self-tested in PL5")
    p6 = rjson(PL6 / "selftest.json")
    if _PREP7["pl6"] != p6["code_path_sha256"]:
        pin["problems"].append("PL6's code-path hash differs from the one filed at its selftest")
    if _PREP7["pl6_script"] != p6["script_sha256"]:
        pin["problems"].append("zpl6g.py differs from the file that was self-tested in PL6")
    pin["hashes"] = dict(_PREP7)
    pin["verdict"] = "PASS" if not pin["problems"] else "FAIL"
    res["G-P"] = pin
    op = P2.opened()
    eng = []
    for ds in a.ds:
        src = P2.load_source(ds, "dev", "frozen")
        M3A = op.m3a
        bad, checks = [], 0
        for q in range(min(a.n, src.nq)):
            d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
            seeds5 = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
            f1 = M3A.rrf_fused_rows(d[None, :], s[None, :], P2.RRF_CONST)[0]
            qv = src.qvec(q, q + 1)[0]
            scorer = Z3.Scorer(src.dvec, {"q": np.asarray(qv, dtype=np.float64)}, src.n)
            for w in walks7(END):
                regime, stg, mods = Z5.parse_walk5(w)
                sd = np.asarray(f1[:10] if "s10" in mods else (f1[:20] if "s20" in mods else seeds5), dtype=np.int64)
                h1, h2 = Z5.REG3[regime]
                a_ = Z5.expand5(sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], Z5.SET5[stg], mods, scorer)
                b_ = Z5.ref_walk5(sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], Z5.SET5[stg], mods, qv, src.dvec)
                checks += 1
                if not np.array_equal(a_, b_):
                    bad.append([q, w])
            scorer.release()
        # the hard cuts on a real question: every cell's pool is within its ceiling
        probe = [f"{b}+{w}+top{k - BASE_B[b]}" for w in walks7(5000)[:3] + walks7(5000)[9:11] for b in BASES7 for k in (2500, 3500)]
        P, S = Z5.evaluate5(op, src, Z5.Plan5(probe), 0, min(a.n, src.nq))
        over = []
        for j, n in enumerate(probe):
            k = int(n.rsplit("+top", 1)[1]) + BASE_B[n.split("+", 1)[0]]
            if (S[:, j] > k).any():
                over.append([n, int(S[:, j].max()), k])
        eng.append({"dataset": ds, "visited": END, "questions": min(a.n, src.nq), "checks": checks, "mismatches": bad[:5], "n_mismatches": len(bad), "hard_cut_cells_probed": len(probe),
                    "hard_cut_over_ceiling": over[:5], "n_nodes": int(src.n), "verdict": "PASS" if not bad and not over else "FAIL"})
        log(f"engine vs per-node reference, {ds}, V {END}: {checks} checks, {len(bad)} mismatches; hard cuts: {len(probe)} cells probed, {len(over)} over their ceiling")
    res["engine"] = eng
    res["verdict"] = "PASS" if cells["verdict"] == "PASS" and pin["verdict"] == "PASS" and all(e["verdict"] == "PASS" for e in eng) else "FAIL"
    res["seconds"] = round(time.time() - t0, 1)
    PL7.mkdir(parents=True, exist_ok=True)
    wjson(PL7 / "selftest.json", res)
    log(f"selftest: {res['verdict']} (cells {cells['verdict']}, G-P {pin['verdict']}, engine {[e['verdict'] for e in eng]}) in {res['seconds']} s")
    for pr in cells["problems"][:8] + pin["problems"][:8]:
        log("   problem: " + pr)
    return 0 if res["verdict"] == "PASS" else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cells")
    p.add_argument("--write", action="store_true")
    p = sub.add_parser("selftest")
    p.add_argument("--ds", nargs="+", default=["metaqa"], choices=SEVEN)
    p.add_argument("--n", type=int, default=1)
    p = sub.add_parser("grid")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--part", nargs=2, metavar=("I", "K"))
    p = sub.add_parser("select")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p = sub.add_parser("confirm")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--part", nargs=2, metavar=("I", "K"))
    p = sub.add_parser("rows")
    p.add_argument("--ds", nargs="*", choices=SEVEN)
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--write", action="store_true")
    sub.add_parser("call")
    for name in ("selftest", "grid", "select", "confirm"):
        sub.choices[name].add_argument("--host", action="store_true", help="open the package as the lab host's mirror")
    a = ap.parse_args(argv)
    prepare7(bool(getattr(a, "host", False)))
    code_path_sha256()
    script_sha256()
    return {"cells": cmd_cells, "selftest": cmd_selftest, "grid": cmd_grid, "select": cmd_select, "confirm": cmd_confirm, "call": cmd_call, "rows": cmd_rows}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
