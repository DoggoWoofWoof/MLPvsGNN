"""PL6 (docs/PL6_COVERAGE_ROUND5.md): round 5 of the coverage target. One selection rule, no node bound, a budget ladder that grows until the target is met.

    python outputs/mp_unified/zpl6g.py cells [--write]                         # the declared ladder cells per rung and their sha256
    python outputs/mp_unified/zpl6g.py time --ds D --rung R [--n 3] [--walks SUBSTR]   # cost of a rung's walks, no labels read
    python outputs/mp_unified/zpl6g.py gr --ds D [--n 100] [--n4 40] [--host]  # G-R: PL5's 1,872 cells and PL4's 1,932 re-built by this engine == their filed dev arrays
    python outputs/mp_unified/zpl6g.py ladder [--ds D ...]                     # where each dataset stands: rung read, pick, and the rung the rule asks for next
    python outputs/mp_unified/zpl6g.py grid --ds D --rung R --carve dev --first frozen [--part I K] [--host]
    python outputs/mp_unified/zpl6g.py select --ds D --rung R                  # the rule on dev over rung 0 (PL2-PL5's 6,078 cells) and rungs 1..R
    python outputs/mp_unified/zpl6g.py confirm --ds D [--first frozen e3v] [--host]   # eval: only the cells the picks need that PL5's confirmation grid does not hold
    python outputs/mp_unified/zpl6g.py call                                    # the universal rule's pick against the bounded pick PL2-PL5 carried, over the seven
    python outputs/mp_unified/zpl6g.py selftest [--ds D ...]                   # cells, parse, pinned hashes, and the engine against a per-node reference

This module imports zpl5g.py (PL5's engine), which imports zpl4g.py, zpl3g.py and zpl2g.py; it edits none of them. It adds ONE thing to the engine: four larger visited
budgets (V = 10,000 x 2^(rung-1): 10k, 20k, 40k, 80k) for the walks that held the dev frontier above 2,500 nodes in PL2-PL5 (the c100 walks of PL3 and PL4, and PL5's
budget-following walks), by adding their settings to PL5's settings table and teaching PL5's walk parser their names, AFTER the earlier rounds' code-path hashes have
been computed (they are cached, so every earlier grid head stays valid); gate G-P checks that the pinned files' hashes equal the filed ones. The pool code itself,
every walk's expansion, every order, every grade is the earlier rounds' code unchanged; gate G-R re-builds PL5's and PL4's cells with this process and compares them to
their filed arrays element for element. Every pool function takes and reads no labels. Everything is deterministic.
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
import zpl5g as Z5  # noqa: E402

P2, Z3, Z4 = Z5.P2, Z5.Z3, Z5.Z4
DOC = "docs/PL6_COVERAGE_ROUND5.md"
PL5 = Z5.PL5
PL6 = ROOT / "outputs" / "pl6"
SEVEN = Z5.SEVEN
LAPTOP6 = Z5.LAPTOP5
log, stop, rel, rjson, wjson, utc = P2.log, P2.stop, P2.rel, P2.rjson, P2.wjson, P2.utc
walk_name = Z3.walk_name

# ── the declared ladder (docs/PL6_COVERAGE_ROUND5.md sections 4 and 5) ───────

TARGET = 0.95                                               # the rule's floor: ALL-gold coverage of the pool
BESIDE_FLOOR = 0.98                                         # read beside, decides nothing
RISE = 0.005                                                # the ladder goes on only while the best dev ALL rose by at least this much over the rung before
V_BASE = 10000
RUNGS = (1, 2, 3, 4)                                        # rung 0 = PL2's, PL3's, PL4's and PL5's 6,078 cells, already read
BASES6 = ("rrf1000_b25", "rrf1000_b400", "rrf1000_b1600")
FRACS = (1, 2, 3)                                           # a ranked cell keeps M = V x f / 4 nodes of the walk, f in FRACS; the union cell keeps all of it
ANY6 = (3000, 4000, 5000, 6000, 7000, 10000, 15000, 20000, 30000, 40000, 60000, 80000)   # the frontier read beside: matched node counts, no bound
WINDOW = Z5.WINDOW


def vee(r):
    return V_BASE * 2 ** (r - 1)


def settings6():
    out = {}
    for r in RUNGS:
        v = vee(r)
        for h in (2, 3):
            out[f"h{h}_c100_v{v}"] = {"name": f"h{h}_c100_v{v}", "hops": h, "per_seed_cap": 100, "per_frontier_cap": 100, "visited_cap": v}
            out[Z5.bf_name(h, v)] = {"name": Z5.bf_name(h, v), "hops": h, "per_seed_cap": WINDOW, "per_frontier_cap": WINDOW, "visited_cap": v, "bf": True, "window": WINDOW}
    return out


SET6 = settings6()


def walks6(r):
    """The 13 walks of a rung: the nine c100 walks and the four budget-following walks that held the dev frontier above 2,500 nodes in PL2-PL5 (docs section 4), at the rung's V."""
    v = vee(r)
    c2, c3, b2, b3 = f"h2_c100_v{v}", f"h3_c100_v{v}", Z5.bf_name(2, v), Z5.bf_name(3, v)
    W = walk_name
    return [W("STRUCT", c3), W("STRUCT", c3, ("cs", "sp", "eq")), W("FULL", c3, ("cs", "sp", "eq")), W("FULL", c3, ("cs", "sp")), W("FULL", c3, ("cs", "vr", "eq")),
            W("FULL", c2, ("cs", "vr")), W("FULL", c2, ("s10", "cs", "vr")), W("FULL", c2, ("s20", "cs", "vr")), W("FULL", c2, ("s20", "cs", "rk")),
            W("FULL", b3, ("cs", "vr")), W("FULL", b3, ("cs", "fu")), W("STRUCT", b3, ("cs", "fu")), W("FULL", b2, ("s10", "cs", "vr"))]


def sizes6(r):
    return [vee(r) * f // 4 for f in FRACS]


def cell_list6(r):
    """A rung's cells: every walk x 3 bases as unions, and x 3 sizes as ranked cells (the first M of the walk's own cosine order)."""
    out = []
    for w in walks6(r):
        for b in BASES6:
            out.append(f"{b}+{w}")
        for b in BASES6:
            for m in sizes6(r):
                out.append(f"{b}+{w}+top{m}")
    return out


def cell_list6_all():
    return [n for r in RUNGS for n in cell_list6(r)]


EXPECT6 = {1: (156, "1dd177c06a0d5c8895dd0cddb4e1001d45652fd747d9b4e401241b471d817373"), 2: (156, "0b296eb8a030caef3ee13452a9ae0b1a7e679095b1068fbd3550f27fe66b9bbf"),
           3: (156, "f3afe89b1f3e1da581cb8fae83b657bedfc61e4cc4fe0544cb44dc3eb5499644"), 4: (156, "1d0d4c56aa6aa194b4d2899f7f3b76004ed9737340fc7a633331fffafe8d8900")}    # fixed at the declaration commit (docs section 5)
EXPECT6_ALL_SHA256 = "6cc6d826ff78b74b7fd94796c62b0d31105f939ddfb71106413c9232a8d4607d"


def constants6():
    return {"settings": {k: SET6[k] for k in sorted(SET6)}, "v_base": V_BASE, "rungs": list(RUNGS), "bases": list(BASES6), "fracs": list(FRACS), "target": TARGET,
            "beside_floor": BESIDE_FLOOR, "rise": RISE, "any_budgets": list(ANY6), "walks": {str(r): walks6(r) for r in RUNGS}, "boot": [P2.BOOT_N, P2.BOOT_SEED]}


def require_cells6(r):
    names = cell_list6(r)
    n, sha = EXPECT6[r]
    if len(names) != n or len(set(names)) != n:
        stop(f"the PL6 rung {r} cell list has {len(names)} cells ({len(set(names))} distinct), expected {n}")
    if P2.cells_sha256(names) != sha:
        stop(f"the PL6 rung {r} cell list is not the declared one (cells_sha256 differs from EXPECT6)")


def tags6(name):
    """Which rung and family a ladder cell belongs to (descriptive); earlier cells carry PL5's tags."""
    if name in LADDER_SET:
        regime, setting, mods = Z5.parse5(name)[3]
        r = RUNGS[[vee(x) for x in RUNGS].index(SET6[setting]["visited_cap"])]
        return (f"r{r}", "bf" if SET6[setting].get("bf") else "c100")
    return Z5.tags_of5(name)


def family6(name):
    return "pl6" if name in LADDER_SET else Z5.family5(name)


LADDER_SET = set(cell_list6_all())

# ── the one change to the engine: the ladder's settings, after the earlier hashes are cached ─────

_PREPARED = {}
_ORIG_PARSE = Z5.parse_walk5


def parse_walk6(w):
    regime, rest = w.split(":")
    setting, _, ms = rest.partition("~")
    if setting not in SET6:
        return _ORIG_PARSE(w)
    mods = tuple(ms.split("-")) if ms else ()
    if regime not in Z5.REG3:
        raise ValueError(f"unknown walk {w}")
    if any(m not in Z5.MODS5 for m in mods) or list(mods) != sorted(mods, key=Z5.MODS5.index) or ("s10" in mods and "s20" in mods):
        raise ValueError(f"unknown walk {w}")
    st = SET6[setting]
    if st.get("bf"):
        if "cs" not in mods or sum(m in Z5.KEYS5 for m in mods) != 1 or "eq" in mods or any(m in ("sp", "rk") for m in mods):
            raise ValueError(f"a budget-following walk needs cs and exactly one of vr, fu, and takes no eq, sp or rk: {w}")
        return (regime, setting, mods)
    keys4 = Z4.KEYS4
    if sum(m in keys4 for m in mods) > 1 or "fu" in mods:
        raise ValueError(f"one order key at most (vr, sp, rk): {w}")
    if "eq" in mods and st["hops"] != 3:
        raise ValueError(f"eq needs three hops: {w}")
    if "rr" in mods and regime != "FULL":
        raise ValueError(f"rr needs regime FULL: {w}")
    return (regime, setting, mods)


def prepare(host=False):
    """Compute (and so cache) every earlier round's hashes, THEN add the ladder's settings and parser. Idempotent."""
    P2.HOST = bool(host)
    if _PREPARED:
        return _PREPARED
    _PREPARED["pl2"] = P2.code_path_sha256()
    _PREPARED["pl2_script"] = P2.script_sha256()
    _PREPARED["pl3"] = Z3.code_path_sha256()
    _PREPARED["pl4"] = Z4.code_path_sha256()
    _PREPARED["pl5"] = Z5.code_path_sha256()
    _PREPARED["pl5_script"] = Z5.script_sha256()
    for k, v in SET6.items():
        Z5.SET5[k] = v
    Z5.parse_walk5 = parse_walk6
    return _PREPARED


_SHA = {}


def code_path_sha256():
    """PL6's one hash: its ladder code and constants and PL5's code-path hash (which holds PL4's, PL3's and PL2's): the same on every dataset (G-U)."""
    if "code" not in _SHA:
        h = hashlib.sha256()
        for fn in (settings6, walks6, sizes6, cell_list6, parse_walk6, vee, select_rule6):
            h.update(inspect.getsource(fn).replace("\r\n", "\n").encode("utf-8"))
        h.update(json.dumps(constants6(), sort_keys=True).encode("utf-8"))
        h.update(_PREPARED["pl5"].encode("utf-8"))
        _SHA["code"] = h.hexdigest()
    return _SHA["code"]


def script_sha256():
    if "script" not in _SHA:
        _SHA["script"] = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return _SHA["script"]


# ── the rule (docs section 6) ────────────────────────────────────────────────

def pick_rule(rows, floor):
    """The universal rule: the smallest mean nodes with dev ALL >= floor (ties: higher ALL, then the name); if no cell reaches it, the highest ALL (ties: fewer nodes, then the name).
    No bound is applied. Returns (cell, reached)."""
    hit = [r for r in rows if r["all"] >= floor]
    if hit:
        return min(hit, key=lambda r: (r["nodes"], -r["all"], r["name"])), True
    return min(rows, key=lambda r: (-r["all"], r["nodes"], r["name"])), False


def decide_ladder(best, n_nodes=None):
    """The ladder's stop rule. best[r] = the best dev ALL over rung 0..r. Returns (state, next rung or None):
    REACHED (the pick reaches the target), STOPPED_RISING (the last rung raised the best ALL by less than RISE), GRAPH_END (a rung's budget has met the graph's size),
    LADDER_END (rung 4 read; a further rung is another page), RUN_RUNG (the next rung is asked for)."""
    k = len(best) - 1
    if best[k] >= TARGET:
        return "REACHED", None
    if k >= 1 and best[k] - best[k - 1] < RISE:
        return "STOPPED_RISING", None
    if k >= 1 and n_nodes is not None and vee(k) >= n_nodes:
        return "GRAPH_END", None
    if k >= len(RUNGS):
        return "LADDER_END", None
    return "RUN_RUNG", k + 1


def select_rule6(rows, floor=TARGET):
    return pick_rule(rows, floor)


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_cells(a):
    tot = 0
    out = {}
    for r in RUNGS:
        names = cell_list6(r)
        tot += len(names)
        out[str(r)] = {"v": vee(r), "cells": len(names), "distinct": len(set(names)), "sha256": P2.cells_sha256(names), "walks": walks6(r), "sizes": sizes6(r)}
        log(f"rung {r}: V {vee(r)}, {len(names)} cells ({len(set(names))} distinct), {len(walks6(r))} walks, sha256 {out[str(r)]['sha256']}")
    allset = set(Z5.cell_list_union5())
    clash = [n for n in cell_list6_all() if n in allset]
    log(f"ladder cells {tot}; overlap with PL2-PL5's 6,078: {len(clash)}")
    parsed = 0
    for n in cell_list6_all():
        Z5.parse5(n)
        parsed += 1
    log(f"all {parsed} ladder cell names parse")
    if a.write:
        PL6.mkdir(parents=True, exist_ok=True)
        wjson(PL6 / "cells.json", {"stage": "PL6 cells", "declared_in": DOC, "rungs": out, "cells": {str(r): cell_list6(r) for r in RUNGS}, "overlap_with_pl2_pl5": len(clash),
                                   "all_sha256": P2.cells_sha256(cell_list6_all()), "utc": utc()})
        log("wrote outputs/pl6/cells.json")
    return 0


def cmd_time(a):
    op = P2.opened()
    names = cell_list6(a.rung)
    if a.walks:
        names = [n for n in names if a.walks in n]
    plan = Z5.Plan5(names)
    src = P2.load_source(a.ds, "dev", "frozen")
    n = min(a.n, src.nq)
    Z5.TIMES = {}
    t0 = time.time()
    for q, st in Z5.iter_states5(op, src, plan, 0, n):
        pass
    tot = time.time() - t0
    sc = Z5.TIMES.pop("_scored", 0)
    res = {"dataset": a.ds, "rung": a.rung, "v": vee(a.rung), "questions": n, "cells": plan.C, "walks": len(plan.walks), "seconds_per_question": tot / n, "nodes_scored_per_question": sc / n,
           "peak_gb": P2.peak_gb(), "n_nodes": int(src.n), "per_walk_ms": {walk_name(*w): 1000 * t / n for w, t in Z5.TIMES.items()}, "labels_read": False, "host": P2.HOST, "utc": utc()}
    log(f"{a.ds} rung {a.rung} (V {vee(a.rung)}): {n} questions, {plan.C} cells, {len(plan.walks)} walks, {tot:.1f} s = {tot / n:.2f} s/question; nodes scored {sc / n:,.0f}/question; "
        f"peak {P2.peak_gb()} GB; graph {src.n:,} nodes")
    for w, t in sorted(Z5.TIMES.items(), key=lambda kv: -kv[1])[:int(a.top)]:
        log(f"   {walk_name(*w):44s} {1000 * t / n:9.1f} ms/question")
    if a.write:
        PL6.mkdir(parents=True, exist_ok=True)
        wjson(PL6 / f"time_{a.ds}_r{a.rung}{'_host' if P2.HOST else ''}.json", res)
    return 0


def gates_passed6(ds):
    """Laptop: this dataset's own G-R. Host: the host's G-R on MuSiQue (as PL5's), whatever the dataset. And the self-test, filed on the laptop."""
    p = PL6 / ("gr_musique_host.json" if P2.HOST else f"gr_{ds}.json")
    if not p.is_file() or rjson(p)["verdict"] != "PASS":
        stop(f"{p.name} is not PASS: run `zpl6g.py gr --ds {'musique --host' if P2.HOST else ds}` first")
    st = PL6 / "selftest.json"
    if not st.is_file() or rjson(st)["verdict"] != "PASS":
        stop("outputs/pl6/selftest.json is not PASS: run `zpl6g.py selftest` first")


def cmd_gr(a):
    """G-R: PL5's 1,872 cells on the first a.n dev questions and PL4's 1,932 on the first a.n4, built by THIS process (the ladder's settings added), equal their filed dev arrays."""
    t0 = time.time()
    op = P2.opened()
    names5, names4 = Z5.cell_list5(), Z4.cell_list4()
    src = P2.load_source(a.ds, "dev", "frozen")
    r5 = Z5.load_grid5(Z5.grid_path5(a.ds, "dev", "frozen"))
    r4 = Z5.load_grid5(Z4.grid_path4(a.ds, "dev", "frozen"))
    n, n4 = min(a.n, src.nq), min(a.n4, src.nq)
    if r5["names"] != names5 or r4["names"] != names4 or any(r["qids"][:n] != list(src.qids[:n]) for r in (r5, r4)):
        stop(f"{a.ds}: PL5's and PL4's dev grids do not name the cells or questions this engine expects")
    P, S = Z5.evaluate5(op, src, Z5.Plan5(names5), 0, n)
    bad_p, bad_s = np.argwhere(P != r5["present"][:n]), np.argwhere(S != r5["size"][:n])
    P4_, S4_ = Z5.evaluate5(op, src, Z5.Plan5(names4), 0, n4)
    bad_p4, bad_s4 = np.argwhere(P4_ != r4["present"][:n4]), np.argwhere(S4_ != r4["size"][:n4])
    ok = not (len(bad_p) or len(bad_s) or len(bad_p4) or len(bad_s4))
    first = None
    for nm, bp, bs in ((names5, bad_p, bad_s), (names4, bad_p4, bad_s4)):
        if first is None and (len(bp) or len(bs)):
            first = [int((bp if len(bp) else bs)[0][0]), nm[int((bp if len(bp) else bs)[0][1])]]
    out = {"stage": "PL6 G-R", "declared_in": DOC, "dataset": a.ds, "host": P2.HOST, "questions_pl5": int(n), "questions_pl4": int(n4), "pl5_cells": len(names5), "pl4_cells": len(names4),
           "elements": int(P.size + P4_.size), "present_mismatches": int(len(bad_p) + len(bad_p4)), "size_mismatches": int(len(bad_s) + len(bad_s4)), "first_mismatch": first,
           "pinned": dict(_PREPARED), "code_path_sha256": code_path_sha256(), "script_sha256": script_sha256(), "verdict": "PASS" if ok else "FAIL",
           "seconds": round(time.time() - t0, 1), "peak_gb": P2.peak_gb(), "utc": utc()}
    PL6.mkdir(parents=True, exist_ok=True)
    wjson(PL6 / f"gr_{a.ds}{'_host' if P2.HOST else ''}.json", out)
    log(f"G-R {a.ds}: {out['verdict']}, {n} q x {len(names5)} cells + {n4} x {len(names4)} = {out['elements']} elements, mismatches {out['present_mismatches']}/{out['size_mismatches']} ({out['seconds']} s)")
    return 0 if ok else 3


def grid_path6(d, r, carve, first, tag=""):
    return PL6 / d / f"grid_{carve}_{first}_r{r}{('_' + tag) if tag else ''}"


def save_grid6(base, src, plan, P, S, lo, hi, seconds, r, part=None, stage="PL6 grid"):
    p = Path(str(base) + (f".part{part[0]}of{part[1]}" if part else "") + ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez(tmp, names=np.asarray(plan.names, dtype=str), present=P, size=S, qids=np.asarray(src.qids[lo:hi], dtype=str),
             gold_counts=src.gold_counts[lo:hi].astype(np.int32), lo=np.asarray(lo), hi=np.asarray(hi))
    os.replace(tmp, p)
    head = {"stage": stage, "declared_in": DOC, "dataset": src.name, "carve": src.carve, "first": src.first, "rung": r, "v": vee(r), "questions": int(hi - lo), "rows": [int(lo), int(hi)],
            "part": list(part) if part else None, "cells": plan.C, "cells_sha256": P2.cells_sha256(plan.names), "n_nodes": int(src.n), "population_digest": src.pop_digest,
            "code_path_sha256": code_path_sha256(), "pinned": dict(_PREPARED), "script_sha256": script_sha256(),
            "constants_sha256": hashlib.sha256(json.dumps(constants6(), sort_keys=True).encode()).hexdigest(), "npz": rel(p), "npz_sha256": P2.sha256_file(p),
            "seconds": round(seconds, 1), "peak_gb": P2.peak_gb(), "host": P2.HOST, "utc": utc()}
    wjson(Path(str(p)[:-4] + ".json"), head)
    return p, head


def ladder_status(ds):
    """What the rule says about a dataset, from the filed selects: ('REACHED'|..., next rung), and the highest rung read."""
    p5 = PL5 / ds / "select.json"
    if not p5.is_file():
        return {"state": "RUNG0_NOT_READ", "next": None, "read": None}
    best = [max(x["all"] for x in rjson(p5)["table"])]
    k = 0
    n_nodes = None
    while (PL6 / ds / f"select_r{k + 1}.json").is_file():
        k += 1
        s = rjson(PL6 / ds / f"select_r{k}.json")
        best = s["best_by_rung"]
        n_nodes = s.get("n_nodes")
    state, nxt = decide_ladder(best, n_nodes)
    return {"state": state, "next": nxt, "read": k, "best_by_rung": best}


def cmd_ladder(a):
    for ds in (a.ds or SEVEN):
        st = ladder_status(ds)
        log(f"{ds}: {st['state']}" + (f" -> rung {st['next']} (V {vee(st['next'])})" if st["next"] else "") + f"  read to rung {st['read']}  best dev ALL by rung {st.get('best_by_rung')}")
    return 0


def cmd_grid(a):
    t0 = time.time()
    gates_passed6(a.ds)
    st = ladder_status(a.ds)
    if st["state"] != "RUN_RUNG" or st["next"] != a.rung:
        stop(f"{a.ds}: the rule does not ask for rung {a.rung} (state {st['state']}, next {st['next']}); `zpl6g.py ladder --ds {a.ds}`")
    require_cells6(a.rung)
    plan = Z5.Plan5(cell_list6(a.rung))
    op = P2.opened()
    src = P2.load_source(a.ds, a.carve, a.first)
    lo, hi, part = 0, src.nq, None
    if a.part:
        i, k = int(a.part[0]), int(a.part[1])
        lo, hi = i * src.nq // k, (i + 1) * src.nq // k
        part = (i, k)
    log(f"{a.ds}/{a.carve}/{a.first} rung {a.rung} (V {vee(a.rung)}): {src.nq} questions, rows {lo}..{hi}, {plan.C} cells, {len(plan.walks)} walks, graph {src.graph_kind} ({P2.peak_gb()} GB)")
    P, S = Z5.evaluate5(op, src, plan, lo, hi)
    p, head = save_grid6(grid_path6(a.ds, a.rung, a.carve, a.first), src, plan, P, S, lo, hi, time.time() - t0, a.rung, part)
    log(f"{a.ds}/{a.carve}/{a.first} rung {a.rung}: filed {rel(p)} in {head['seconds']} s")
    return 0


def load_rung_dev(ds, r):
    base = grid_path6(ds, r, "dev", "frozen")
    g = Z5.load_grid5(base)
    heads = Z5.heads_of(base)
    if not heads:
        stop(f"{ds}: no PL6 rung {r} grid heads")
    for hp in heads:
        hd = rjson(hp)
        if hd["code_path_sha256"] != code_path_sha256() or hd["cells_sha256"] != EXPECT6[r][1]:
            stop(f"G-U: {hp.name} was written by other PL6 code or another cell list than the one that is running")
    if P2.cells_sha256(g["names"]) != EXPECT6[r][1]:
        stop(f"the PL6 rung {r} dev grid is not the declared cell list")
    return g, heads


def cmd_select(a):
    P2.HOST = bool(a.host)
    ds, k = a.ds, a.rung
    p5 = PL5 / ds / "select.json"
    if not p5.is_file():
        stop(f"{ds}: PL5's select.json (rung 0) is not here")
    s5 = rjson(p5)
    table = [dict(x) for x in s5["table"]]
    heads = []
    best = [max(x["all"] for x in table)]
    n_nodes = None
    qids = None
    for r in range(1, k + 1):
        g, hs = load_rung_dev(ds, r)
        if qids is not None and g["qids"] != qids:
            stop(f"{ds}: the rungs' dev grids are not on the same questions")
        qids = g["qids"]
        if len(qids) != s5["questions"]:
            stop(f"{ds}: rung {r}'s dev grid has {len(qids)} questions, PL5's select {s5['questions']}")
        rows = Z5.rows_of_grid5(g, ds, "dev")
        for n, rw in zip(g["names"], rows):
            table.append({"name": n, "family": "pl6", "tags": list(tags6(n)), "nodes": rw["candidates_mean"], "all": rw["all_gold_at_pool"], "any": rw["any_gold_at_pool"],
                          "coverage": rw["coverage_micro"], "p95": rw["candidates_p95"], "max": rw["candidates_max"]})
        best.append(max(x["all"] for x in table))
        heads += [{"file": rel(hp), "rung": r, "code_path_sha256": rjson(hp)["code_path_sha256"], "npz_sha256": rjson(hp)["npz_sha256"], "seconds": rjson(hp)["seconds"],
                   "peak_gb": rjson(hp)["peak_gb"], "host": rjson(hp)["host"]} for hp in hs]
        n_nodes = rjson(hs[0])["n_nodes"]
    out = {"stage": "PL6 select", "declared_in": DOC, "dataset": ds, "rung": k, "carve": "dev", "first": "frozen", "questions": s5["questions"], "rule": "R95any over rungs 0..rung, no bound",
           "pl5_select_sha256": P2.sha256_file(p5), "code_path_sha256": code_path_sha256(), "pinned": dict(_PREPARED), "script_sha256": script_sha256(), "utc": utc(), "grid_heads": heads,
           "n_nodes": n_nodes}
    picks = {}
    readings = {"all": table, "rung0": [t for t in table if t["family"] != "pl6"], "bounded": [t for t in table if t["nodes"] <= P2.BOUND and t["family"] != "pl6"]}
    for fl, tag in ((TARGET, "R95"), (BESIDE_FLOOR, "R98")):
        for rd in ("all", "rung0", "bounded"):
            c, reached = pick_rule(readings[rd], fl)
            picks[f"{tag}_{rd}"] = {**c, "reached": reached, "floor": fl}
    out["picks"] = picks
    out["best_by_rung"] = best
    state, nxt = decide_ladder(best, n_nodes)
    out["ladder"] = {"state": state, "next_rung": nxt, "best_by_rung": best, "rise_required": RISE, "target": TARGET}
    out["frontier_picks"] = {rd: {str(n): Z5.frontier_pick(readings[rd], n) for n in ANY6} for rd in ("all", "rung0")}
    out["frontier"] = {rd: P2.frontier(readings[rd]) for rd in ("all", "rung0")}
    inc = s5["picks"]["R95_all"]
    out["incumbent"] = {"name": inc["name"], "all": inc["all"], "nodes": inc["nodes"], "reached": inc["reached"], "role": "PL5's bounded R95 of all its cells (what PL2-PL5 carried)"}
    out["universal_same_as_bounded"] = bool(picks["R95_all"]["name"] == inc["name"]) if inc["reached"] else None
    out["old"] = s5["old"]
    out["table"] = table
    wjson(PL6 / ds / f"select_r{k}.json", out)
    wjson(PL6 / ds / "select.json", out)
    for key in ("R95_all", "R95_rung0", "R95_bounded", "R98_all", "R98_rung0"):
        c = picks[key]
        log(f"{ds} rung {k} {key}: {c['name']}  ALL {c['all']:.4f} at {c['nodes']:.1f} nodes{'' if c['reached'] else '  (SHORT)'}")
    log(f"{ds}: best dev ALL by rung {['%.4f' % b for b in best]}; ladder {state}" + (f" -> rung {nxt}" if nxt else "")
        + f"; same cell as PL5's bounded pick: {out['universal_same_as_bounded']}")
    return 0


# ── confirm and call ─────────────────────────────────────────────────────────

def needed_names(sel):
    names = []

    def add(n):
        if n and n not in names:
            names.append(n)
    for key in ("R95_all", "R95_rung0", "R95_bounded", "R98_all", "R98_rung0"):
        add(sel["picks"][key]["name"])
    add(sel["incumbent"]["name"])
    for rd in ("all", "rung0"):
        for n in ANY6:
            c = sel["frontier_picks"][rd][str(n)]
            add(c["name"] if c else None)
    add(sel["old"])
    return names


def pl5_confirm_grid(ds, first):
    base = Z5.grid_path5(ds, "eval", first, "confirm")
    return Z5.load_grid5(base) if Path(str(base) + ".npz").is_file() else None


def cmd_confirm(a):
    P2.HOST = bool(a.host)
    gates_passed6(a.ds)
    sel = rjson(PL6 / a.ds / "select.json")
    names = needed_names(sel)
    op = P2.opened()
    for first in a.first:
        t0 = time.time()
        g5 = pl5_confirm_grid(a.ds, first)
        have = set(g5["names"]) if g5 else set()
        missing = [n for n in names if n not in have]
        log(f"{a.ds}/eval/{first}: {len(names)} cells needed, {len(names) - len(missing)} already in PL5's confirmation grid, {len(missing)} to build")
        if missing:
            plan = Z5.Plan5(missing)
            try:
                src = P2.load_source(a.ds, "eval", first)
            except SystemExit as e:
                log(f"{a.ds}/eval/{first}: not read here ({e})")
                continue
            if plan.need_vec:
                try:
                    src.dvec(np.asarray([0]))
                except Exception as e:
                    log(f"{a.ds}/eval/{first}: the ranking vectors are not here ({e}); left for the host")
                    continue
            P, S = Z5.evaluate5(op, src, plan, 0, src.nq)
            save_grid6(PL6 / a.ds / f"grid_eval_{first}_confirm", src, plan, P, S, 0, src.nq, time.time() - t0, 0, stage="PL6 confirm grid")
        wjson(PL6 / a.ds / f"confirm_{first}.json", {"stage": "PL6 confirm", "declared_in": DOC, "dataset": a.ds, "first": first, "needed": names, "built_here": missing,
                                                     "in_pl5_confirm_grid": [n for n in names if n in have], "utc": utc()})
    return 0


def eval_arrays(ds, first):
    """name -> (all-gold bool per question, size per question) over PL5's confirmation grid and PL6's, one question order."""
    out, gc = {}, None
    for g in (pl5_confirm_grid(ds, first), (Z5.load_grid5(PL6 / ds / f"grid_eval_{first}_confirm") if Path(str(PL6 / ds / f"grid_eval_{first}_confirm") + ".npz").is_file() else None)):
        if g is None:
            continue
        if gc is None:
            gc = g["gold_counts"]
        elif not np.array_equal(gc, g["gold_counts"]):
            stop(f"{ds}/{first}: PL5's and PL6's eval grids are not on the same questions")
        for i, n in enumerate(g["names"]):
            out.setdefault(n, ((g["present"][:, i].astype(np.int64) == g["gold_counts"]), g["size"][:, i].astype(np.float64)))
    return out


def frontier_status(prs):
    below = any(p["all_diff_ci"][1] < 0 for p in prs)
    above = sum(p["all_diff_ci"][0] > 0 for p in prs) >= 2
    return "BELOW" if below else ("ABOVE" if above else "LEVEL")


def cmd_call(a):
    out = {"stage": "PL6 call", "declared_in": DOC, "utc": utc(), "datasets": {}}
    codes = {}
    for k, ds in enumerate(SEVEN):
        rng = np.random.default_rng([P2.BOOT_SEED, k])
        sp = PL6 / ds / "select.json"
        if not sp.is_file():
            sp5 = PL5 / ds / "select.json"
            if not sp5.is_file():
                out["datasets"][ds] = {"status": "NOT_READ", "why": "no select file here"}
                continue
            stop(f"{ds}: PL6 select (rung 0 reading) is not filed: run `zpl6g.py select --ds {ds} --rung 0`")
        sel = rjson(sp)
        st = ladder_status(ds)
        if st["state"] == "RUN_RUNG":
            out["datasets"][ds] = {"status": "NOT_READ", "why": f"the ladder asks for rung {st['next']}"}
            continue
        arr = eval_arrays(ds, "frozen")
        need = needed_names(sel)
        miss = [n for n in need if n not in arr]
        if miss:
            out["datasets"][ds] = {"status": "NOT_READ", "why": f"{len(miss)} eval cells missing, e.g. {miss[0]}"}
            continue
        codes[ds] = sel["code_path_sha256"]
        picks = sel["picks"]

        def cmp(nn, on, floor):
            new, old = arr[nn], arr[on]
            pr = P2.paired(new, old, rng)
            nr, orr = bool(new[0].mean() >= floor), bool(old[0].mean() >= floor)
            all_status = "BELOW" if pr["all_diff_ci"][1] < 0 else ("ABOVE" if pr["all_diff_ci"][0] > 0 else "LEVEL")
            return {"status": Z3.status_of(pr, nr, orr), "all_status": all_status, "new": nn, "old": on, "same_cell": nn == on, "new_all": float(new[0].mean()), "old_all": float(old[0].mean()),
                    "new_nodes": float(new[1].mean()), "old_nodes": float(old[1].mean()), "new_p95": float(np.percentile(new[1], 95)), "new_max": float(new[1].max()),
                    "new_reaches": nr, "old_reaches": orr, **pr}
        rec = {}
        inc = sel["incumbent"]["name"]
        rec["main"] = cmp(picks["R95_all"]["name"], inc, TARGET)                       # decides: the rule's pick against the bounded pick PL2-PL5 carried
        rec["ladder"] = cmp(picks["R95_all"]["name"], picks["R95_rung0"]["name"], TARGET)    # beside: what the ladder bought over rung 0 under the same rule
        rec["r98"] = cmp(picks["R98_all"]["name"], picks["R98_rung0"]["name"], BESIDE_FLOOR)  # beside: the 0.98 floor
        fr = {}
        for n in ANY6:
            cn, co = sel["frontier_picks"]["all"][str(n)], sel["frontier_picks"]["rung0"][str(n)]
            if cn is None or co is None:
                fr[str(n)] = {"status": "NONE", "why": "no cell within this many dev nodes"}
                continue
            c = cmp(cn["name"], co["name"], TARGET)
            c["status"] = "BELOW" if c["all_diff_ci"][1] < 0 else ("ABOVE" if c["all_diff_ci"][0] > 0 else "LEVEL")
            c["new_dev_all"], c["old_dev_all"], c["new_dev_nodes"], c["old_dev_nodes"] = cn["all"], co["all"], cn["nodes"], co["nodes"]
            fr[str(n)] = c
        used = [v for v in fr.values() if v["status"] != "NONE"]
        e = arr[picks["R95_all"]["name"]]
        out["datasets"][ds] = {**rec["main"], "status": rec["main"]["status"], "ladder_cmp": rec["ladder"], "r98": rec["r98"], "frontier": fr, "frontier_status": frontier_status(used),
                               "dev_reached": bool(picks["R95_all"]["reached"]), "eval_reached": bool(e[0].mean() >= TARGET), "dev_all": picks["R95_all"]["all"],
                               "dev_nodes": picks["R95_all"]["nodes"], "rung": sel["rung"], "ladder_state": st["state"], "bounded_pick_reached_dev": bool(sel["incumbent"]["reached"]),
                               "universal_same_as_bounded": sel["universal_same_as_bounded"], "n_nodes": sel.get("n_nodes")}
    read = {k: v for k, v in out["datasets"].items() if v["status"] != "NOT_READ"}
    out["datasets_read"] = sorted(read)
    if len(set(codes.values())) > 1:
        stop(f"G-U: the code-path hash differs between datasets: {codes}")
    n_below = sum(v["status"] == "BELOW" for v in read.values())
    n_above = sum(v["status"] == "ABOVE" for v in read.values())
    f_below = sum(v["frontier_status"] == "BELOW" for v in read.values())
    f_above = sum(v["frontier_status"] == "ABOVE" for v in read.values())
    universal = all(v["universal_same_as_bounded"] for v in read.values() if v["bounded_pick_reached_dev"])
    complete = len(read) == len(SEVEN)
    out["universality_check"] = {"datasets_that_met_the_target_inside_the_bound": sorted(k for k, v in read.items() if v["bounded_pick_reached_dev"]),
                                 "unchanged_on_all_of_them": bool(universal)}
    out["call"] = ("CARRY_PL6" if (complete and n_below == 0 and n_above >= 1 and universal) else ("INCOMPLETE" if not complete else "NOT_CARRIED"))
    out["counts"] = {"old": {"BELOW": n_below, "ABOVE": n_above, "LEVEL": len(read) - n_below - n_above},
                     "frontier": {"BELOW": f_below, "ABOVE": f_above, "LEVEL": len(read) - f_below - f_above}}
    out["code_path_equal_everywhere"] = len(set(codes.values())) <= 1
    PL6.mkdir(parents=True, exist_ok=True)
    wjson(PL6 / "call.json", out)
    for ds, v in out["datasets"].items():
        log(f"call {ds}: {v['status']}" + ("" if v["status"] == "NOT_READ" else f"  NEW {v['new']} ALL {v['new_all']:.4f} @ {v['new_nodes']:.0f}; bounded pick {v['old']} ALL {v['old_all']:.4f} @ {v['old_nodes']:.0f}"))
    log(f"call: {out['call']} (BELOW {n_below}, ABOVE {n_above}, read {len(read)}/7; universality unchanged on {out['universality_check']['datasets_that_met_the_target_inside_the_bound']}: {universal})")
    return 0


# ── host rows (docs section 11) ──────────────────────────────────────────────

ROW_CODE = ("docs/PL6_COVERAGE_ROUND5.md", "outputs/mp_unified/zpl6g.py", "outputs/mp_unified/zpl5g.py", "outputs/mp_unified/zpl4g.py", "outputs/mp_unified/zpl3g.py",
            "outputs/mp_unified/zpl2g.py", "outputs/mp_unified/zpkoh.py", "outputs/mp_unified/zpko.py", "outputs/host_ops/retry_cmd.py", "outputs/host_ops/pylib_run.py",
            "outputs/host_ops/pylib/numba067_llvmlite049_cp313_win_amd64.zip")
ROW_MEM = {"musique": 3, "webqsp": 8, "metaqa": 3, "squad": 3, "multihoprag": 3, "hotpotqa": 20, "2wiki": 22}
ROW_RUN = "python outputs/host_ops/retry_cmd.py --tries 2 --wait 90 -- python outputs/host_ops/pylib_run.py outputs/mp_unified/zpl6g.py"


def row(name, deps, cpus, mem, inputs, cmd):
    return "|".join(["line", "x", ",".join(deps), name, str(cpus), str(mem), ";".join(inputs), f"{ROW_RUN} {cmd}"])


def cmd_rows(a):
    """The host rows of one rung for the datasets the rule asks it for (default: the ones `ladder` says RUN_RUNG for that rung): the host's G-R on MuSiQue, the dev grid in
    three parts, the select. Printed; --write appends them to outputs/pl6/feed_rows.txt. Row inputs are files only (the feeder's rule); outputs/pl5/<d>/select.json and the earlier rungs'
    select_r<k>.json are the rule's own state."""
    rows = []
    ds_list = a.ds or [d for d in SEVEN if ladder_status(d)["next"] == a.rung]
    gr = "pl6-musiquegr"
    if a.gr:
        rows.append(row(gr, [], 2, 3, list(ROW_CODE) + [f"outputs/pl5/musique/grid_dev_frozen.part{i}of3.{e}" for i in range(3) for e in ("json", "npz")]
                        + [f"outputs/pl4/musique/grid_dev_frozen.part{i}of3.{e}" for i in range(3) for e in ("json", "npz")]
                        + [f"outputs/pl{k}/musique/grid_dev_frozen.{e}" for k in (2, 3) for e in ("json", "npz")] + ["outputs/m3a/headroom/musique.json"],
                        "gr --ds musique --host"))
    for ds in ds_list:
        tag = ds.replace("2wiki", "2wiki")
        ins = list(ROW_CODE) + ["outputs/pl6/selftest.json", f"outputs/pl5/{ds}/select.json"] + [f"outputs/pl6/{ds}/select_r{k}.json" for k in range(1, a.rung)]
        gnames = []
        for i in range(3):
            n = f"pl6-{tag}r{a.rung}g{i}"
            gnames.append(n)
            rows.append(row(n, [gr] if a.rung == 1 else [f"pl6-{tag}r{a.rung - 1}sel"], 2, ROW_MEM[ds], ins, f"grid --ds {ds} --rung {a.rung} --carve dev --first frozen --part {i} 3 --host"))
        rows.append(row(f"pl6-{tag}r{a.rung}sel", gnames, 1, 3, ins, f"select --ds {ds} --rung {a.rung} --host"))
    nl = chr(10)
    text = nl.join(rows)
    print(text)
    if a.write:
        PL6.mkdir(parents=True, exist_ok=True)
        with open(PL6 / "feed_rows.txt", "a", encoding="utf-8", newline=nl) as f:
            f.write(text + nl)
        log(f"appended {len(rows)} rows to outputs/pl6/feed_rows.txt")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def cmd_selftest(a):
    t0 = time.time()
    res = {"stage": "PL6 selftest", "declared_in": DOC, "utc": utc(), "script_sha256": script_sha256(), "code_path_sha256": code_path_sha256()}
    cells = {"rungs": {}, "problems": []}
    for r in RUNGS:
        names = cell_list6(r)
        sha = P2.cells_sha256(names)
        cells["rungs"][str(r)] = {"n": len(names), "distinct": len(set(names)), "sha256": sha, "expected": list(EXPECT6[r])}
        if len(names) != EXPECT6[r][0] or len(set(names)) != len(names) or sha != EXPECT6[r][1]:
            cells["problems"].append(f"rung {r} differs from the declared list")
    clash = [n for n in cell_list6_all() if n in set(Z5.cell_list_union5())]
    if clash or len(set(cell_list6_all())) != len(cell_list6_all()):
        cells["problems"].append(f"{len(clash)} ladder cells overlap earlier rounds or each other")
    for n in cell_list6_all():
        try:
            Z5.parse5(n)
        except Exception as e:
            cells["problems"].append(f"{n}: {e}")
    cells["verdict"] = "PASS" if not cells["problems"] else "FAIL"
    res["cells"] = cells
    pin = {"problems": []}
    prev = rjson(PL5 / "selftest.json")
    if _PREPARED["pl5"] != prev["code_path_sha256"]:
        pin["problems"].append("PL5's code-path hash differs from the one filed at its selftest")
    if _PREPARED["pl5_script"] != prev["script_sha256"]:
        pin["problems"].append("zpl5g.py differs from the file that was self-tested in PL5")
    pin["hashes"] = dict(_PREPARED)
    pin["verdict"] = "PASS" if not pin["problems"] else "FAIL"
    res["G-P"] = pin
    # the engine against a per-node reference on the ladder's walks: PL5's reference walkers, given the ladder's settings
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
            for w in walks6(a.rung):
                regime, stg, mods = Z5.parse_walk5(w)
                sd = np.asarray(f1[:10] if "s10" in mods else (f1[:20] if "s20" in mods else seeds5), dtype=np.int64)
                h1, h2 = Z5.REG3[regime]
                a_ = Z5.expand5(sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], Z5.SET5[stg], mods, scorer)
                b_ = Z5.ref_walk5(sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], Z5.SET5[stg], mods, qv, src.dvec)
                checks += 1
                if not np.array_equal(a_, b_):
                    bad.append([q, w])
            scorer.release()
        eng.append({"dataset": ds, "rung": a.rung, "questions": min(a.n, src.nq), "checks": checks, "mismatches": bad[:5], "n_mismatches": len(bad), "n_nodes": int(src.n),
                    "verdict": "PASS" if not bad else "FAIL"})
        log(f"engine vs per-node reference, {ds}, rung {a.rung}: {checks} checks, {len(bad)} mismatches")
    res["engine"] = eng
    res["verdict"] = "PASS" if cells["verdict"] == "PASS" and pin["verdict"] == "PASS" and all(e["verdict"] == "PASS" for e in eng) else "FAIL"
    res["seconds"] = round(time.time() - t0, 1)
    PL6.mkdir(parents=True, exist_ok=True)
    wjson(PL6 / "selftest.json", res)
    log(f"selftest: {res['verdict']} (cells {cells['verdict']}, G-P {pin['verdict']}, engine {[e['verdict'] for e in eng]}) in {res['seconds']} s")
    return 0 if res["verdict"] == "PASS" else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cells")
    p.add_argument("--write", action="store_true")
    p = sub.add_parser("time")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--rung", type=int, required=True, choices=RUNGS)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--walks", default="")
    p.add_argument("--top", type=int, default=13)
    p.add_argument("--write", action="store_true")
    p = sub.add_parser("gr")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--n4", type=int, default=40)
    p = sub.add_parser("ladder")
    p.add_argument("--ds", nargs="*", choices=SEVEN)
    p = sub.add_parser("grid")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--rung", type=int, required=True, choices=RUNGS)
    p.add_argument("--carve", default="dev", choices=("dev",))
    p.add_argument("--first", default="frozen", choices=("frozen",))
    p.add_argument("--part", nargs=2, metavar=("I", "K"))
    p = sub.add_parser("select")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--rung", type=int, required=True, choices=(0,) + RUNGS)
    p = sub.add_parser("confirm")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--first", nargs="+", default=["frozen"], choices=("frozen", "e3v"))
    p = sub.add_parser("rows")
    p.add_argument("--rung", type=int, required=True, choices=RUNGS)
    p.add_argument("--ds", nargs="*", choices=SEVEN)
    p.add_argument("--gr", action="store_true", help="include the host's G-R on MuSiQue (the first rung's rows)")
    p.add_argument("--write", action="store_true")
    sub.add_parser("call")
    p = sub.add_parser("selftest")
    p.add_argument("--ds", nargs="+", default=["metaqa"], choices=SEVEN)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--rung", type=int, default=1, choices=RUNGS)
    for name in ("time", "gr", "grid", "select", "confirm", "selftest"):
        sub.choices[name].add_argument("--host", action="store_true", help="open the package as the lab host's mirror")
    a = ap.parse_args(argv)
    prepare(bool(getattr(a, "host", False)))
    code_path_sha256()
    script_sha256()
    return {"cells": cmd_cells, "time": cmd_time, "gr": cmd_gr, "ladder": cmd_ladder, "grid": cmd_grid, "select": cmd_select, "confirm": cmd_confirm, "call": cmd_call,
            "selftest": cmd_selftest, "rows": cmd_rows}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
