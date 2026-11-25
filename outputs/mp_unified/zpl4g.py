"""PL4 (docs/PL4_COVERAGE_ROUND3.md): round 3 of the coverage target. Four new ways to spend a walk's node budget, on top of PL3's grid.

    python outputs/mp_unified/zpl4g.py cells [--write]                         # the declared cell list and its sha256
    python outputs/mp_unified/zpl4g.py time --ds D [--n 40] [--walks SUBSTR]   # cost of the walks, no labels read
    python outputs/mp_unified/zpl4g.py gr --ds D [--n 400] [--host]            # G-R: PL2's 804 and PL3's 1,470 cells re-built by this engine == their dev arrays
    python outputs/mp_unified/zpl4g.py grid --ds D --carve dev --first frozen [--part I K] [--host]
    python outputs/mp_unified/zpl4g.py select --ds D [--host]                  # the rule on dev over PL2's, PL3's and PL4's cells
    python outputs/mp_unified/zpl4g.py confirm --ds D [--first frozen e3v] [--host]
    python outputs/mp_unified/zpl4g.py call                                    # PL4's chosen cells against PL3's, both readings, over the seven
    python outputs/mp_unified/zpl4g.py selftest [--ds D ...]                   # G-X4, G-F, G-D, and the engine against a per-node reference

This module imports zpl3g.py (PL3's engine) and zpl2g.py (PL2's) and edits neither. It adds, around the same pool code, four walk options at
hops 2 and 3: the visited cap keeps the fresh nodes of a hop by support (`sp`: the number of distinct frontier nodes that name them) or by
row rank (`rk`: the best place any frontier node gives them), instead of PL3's cosine (`vr`) or arrival order; each hop may spend only an
equal share of what is left (`eq`); and the families take turns (`rr`). With every new option off the engine gives PL3's and PL2's cells
element for element (gate G-R). Every pool function takes and reads no labels; labels enter in zpl3g.grade3 only. Everything is deterministic:
integer walks, float64 inner products in numpy's pairwise order (no BLAS), ties to the lower id.
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
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402

DOC = "docs/PL4_COVERAGE_ROUND3.md"
PL4 = ROOT / "outputs" / "pl4"
SEVEN, MHR = P2.SEVEN, P2.MHR
LAPTOP5 = Z3.LAPTOP5
FLOORS, BOUND = P2.FLOORS, P2.BOUND
SCORE_WINDOW = Z3.SCORE_WINDOW
REG3 = Z3.REG3
S2, S3, S3S = Z3.S2, Z3.S3, Z3.S3S
BASES4 = Z3.BASES3
M4 = Z3.M3
BUDGETS = (500, 1000, 1500, 2000, 2500)                     # the matched node counts of the frontier reading (section 6 of the page)

# ── the declared cells (docs/PL4_COVERAGE_ROUND3.md section 5) ───────────────

MODS4 = ("s10", "s20", "cs", "vr", "sp", "rk", "rr", "eq")
KEYS4 = ("vr", "sp", "rk")
NEW_MODS = ("sp", "rk", "rr", "eq")
walk_name = Z3.walk_name


def walks4():
    w = []

    def add(reg, st, mods):
        w.append(walk_name(reg, st, mods))
    for reg, sts in (("STRUCT", (S3S, S3)), ("FULL", (S2, S3S, S3))):      # G1: the order key at the visited cap, cosine-ordered caps
        for st in sts:
            for key in ("sp", "rk"):
                add(reg, st, ("cs", key))
    for st in (S3S, S3):                                                   # G1: and with the caps in row order (rk reads the stored weights' order)
        for key in ("sp", "rk"):
            add("FULL", st, (key,))
    for reg in ("STRUCT", "FULL"):                                         # G2: each hop spends an equal share of what is left
        for st in (S3S, S3):
            for key in ("vr", "sp", "rk"):
                add(reg, st, ("cs", key, "eq"))
    for st in (S3S, S3):
        add("FS", st, ("cs", "vr", "eq"))
    for st in (S2, S3S):                                                   # G3: the families take turns
        for key in ("vr", "sp", "rk"):
            add("FULL", st, ("cs", key, "rr"))
    for st in (S3S, S3):
        for key in ("vr", "sp", "rk"):
            add("FULL", st, ("cs", key, "rr", "eq"))
    for sd in ("s10", "s20"):                                              # G4: seeds from the fused list
        for key in ("sp", "rk"):
            add("FULL", S2, (sd, "cs", key))
        add("FULL", S3S, (sd, "cs", "vr", "eq"))
    return w


P2_SET = set(P2.cell_list())
Z3_SET = set(Z3.cell_list3())


def cell_list4():
    """PL4's own cells: every walk x (6 bases as unions, 6 bases x 6 sizes as ranked cells), minus PL2's and PL3's."""
    seen = P2_SET | Z3_SET
    out = []

    def add(n):
        if n not in seen:
            seen.add(n)
            out.append(n)
    for w in walks4():
        for b in BASES4:
            add(f"{b}+{w}")
        for b in BASES4:
            for m in M4:
                add(f"{b}+{w}+top{m}")
    return out


def cell_list_union4():
    return P2.cell_list() + Z3.cell_list3() + cell_list4()


EXPECT4_CELLS = 1932                                         # the declared cell list (docs section 5)
EXPECT4_SHA256 = "32502a00db12e3afffd325170183f0508a7f108afecd505c73a516a4908eed96"


def constants4():
    return {"settings": {k: P2.SETTINGS[k] for k in (Z3.S1, S2, S3, S3S)}, "regimes": {k: [list(v[0]), list(v[1])] for k, v in REG3.items()},
            "mods_order": list(MODS4), "keys": list(KEYS4), "bases": list(BASES4), "m": list(M4), "score_window": SCORE_WINDOW, "walks": walks4(),
            "budgets": list(BUDGETS), "floors": list(FLOORS), "bound": BOUND, "boot": [P2.BOOT_N, P2.BOOT_SEED],
            "rank": "float64 inner product, ties to the lower node id"}


def require_cells4():
    names = cell_list4()
    if len(names) != EXPECT4_CELLS or len(set(names)) != EXPECT4_CELLS:
        P2.stop(f"the PL4 cell list has {len(names)} cells ({len(set(names))} distinct), expected {EXPECT4_CELLS}")
    if P2.cells_sha256(names) != EXPECT4_SHA256:
        P2.stop("the PL4 cell list is not the declared one (cells_sha256 differs from EXPECT4_SHA256)")


# ── cell names -> a plan ─────────────────────────────────────────────────────

def parse_walk4(w):
    regime, rest = w.split(":")
    setting, _, ms = rest.partition("~")
    mods = tuple(ms.split("-")) if ms else ()
    if regime not in REG3 or setting not in P2.SETTINGS:
        raise ValueError(f"unknown walk {w}")
    if any(m not in MODS4 for m in mods) or list(mods) != sorted(mods, key=MODS4.index) or ("s10" in mods and "s20" in mods):
        raise ValueError(f"unknown walk {w}")
    hops = P2.SETTINGS[setting]["hops"]
    if sum(m in KEYS4 for m in mods) > 1:
        raise ValueError(f"one order key at most: {w}")
    if hops == 1 and any(m in KEYS4 + ("rr", "eq") for m in mods):
        raise ValueError(f"nothing to order or split in a one-hop walk: {w}")
    if "eq" in mods and hops != 3:
        raise ValueError(f"eq splits what is left over the remaining hops; it needs three hops: {w}")
    if "rr" in mods and regime != "FULL":
        raise ValueError(f"rr needs two or more families at hops 2 and 3 (regime FULL): {w}")
    return (regime, setting, mods)


def parse4(name):
    """('R', list, b) | ('FU',) | ('G', list, b, walk) | ('K', list, b, walk, M, rv); a walk = (regime, setting, mods); rv = 'q' or 'b'."""
    if "+" not in name:
        return P2.parse(name)
    parts = name.split("+")
    ln, b = P2.base_of(parts[0])
    wk = parse_walk4(parts[1])
    if len(parts) == 3:
        if parts[2].startswith("top"):
            return ("K", ln, b, wk, int(parts[2][3:]), "q")
        if parts[2].startswith("brg"):
            return ("K", ln, b, wk, int(parts[2][3:]), "b")
        raise ValueError(f"unknown cell {name}")
    if len(parts) != 2:
        raise ValueError(f"unknown cell {name}")
    return ("G", ln, b, wk)


def tags_of4(name):
    """Which PL4 option a cell belongs to (descriptive, for what each option bought). PL2's and PL3's own cells are tagged pl2 / pl3."""
    if name in P2_SET:
        return ("pl2",)
    if name in Z3_SET:
        return ("pl3",)
    p = parse4(name)
    mods = p[3][2]
    tags = [m for m in NEW_MODS if m in mods]
    if "s10" in mods or "s20" in mods:
        tags.append("seeds")
    return tuple(tags)


def family4(name):
    return "pl2" if name in P2_SET else ("pl3" if name in Z3_SET else "pl4")


class Plan4:
    def __init__(self, names):
        self.names = list(names)
        if len(set(self.names)) != len(self.names):
            P2.stop("a cell name repeats")
        self.C = len(self.names)
        ret, fu, walks = {}, [], {}
        for i, nm in enumerate(self.names):
            p = parse4(nm)
            if p[0] == "R":
                ret.setdefault(p[1], ([], []))
                ret[p[1]][0].append(i)
                ret[p[1]][1].append(p[2])
            elif p[0] == "FU":
                fu.append(i)
            else:
                w = walks.setdefault(p[3], {"G": {}, "K": {}})
                if p[0] == "G":
                    g_ = w["G"].setdefault(p[1], ([], []))
                    g_[0].append(i)
                    g_[1].append(p[2])
                else:
                    k = w["K"].setdefault((p[1], p[2], p[5]), ([], []))
                    k[0].append(i)
                    k[1].append(p[4])
        self.ret = {ln: (np.asarray(c, np.int64), np.asarray(b, np.int64)) for ln, (c, b) in ret.items()}
        self.fu = fu
        self.walks = {}
        for w, v in walks.items():
            self.walks[w] = {"G": {ln: (np.asarray(c, np.int64), np.asarray(b, np.int64)) for ln, (c, b) in v["G"].items()},
                             "K": {k: (np.asarray(c, np.int64), np.asarray(m, np.int64)) for k, (c, m) in v["K"].items()}}
        lists = set(self.ret)
        for v in self.walks.values():
            lists |= set(v["G"]) | {k[0] for k in v["K"]}
        if fu:
            lists |= {"D", "S"}
        self.lists = lists
        self.walk_rvs = {w: sorted({k[2] for k in v["K"]}) for w, v in self.walks.items()}
        self.need_bridge = any("b" in r for r in self.walk_rvs.values())
        self.need_vec = any(v["K"] for v in self.walks.values()) or any(any(m in w[2] for m in ("cs", "vr", "sp", "rk")) for w in self.walks)
        self.need_f1 = "F1" in lists or any(("s10" in w[2] or "s20" in w[2]) for w in self.walks) or self.need_bridge
        self.families = sorted({f for w in self.walks for part in REG3[w[0]] for f in part})


# ── the pool code (no labels anywhere in this block) ─────────────────────────

def capped_parts(nodes, families, cap, scorer, cs):
    """The kept neighbours of every node in every family, in capped_multi's / capped_sim's order (node-major, then family, then the row's
    own order, or the best by cosine out of the first SCORE_WINDOW with `cs`), with the index of the node that names each, its family and
    its place in that node's list: (cols, parent index, family index, rank)."""
    e64, e8 = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)
    if nodes.size == 0:
        return e64, e64, e8, e64
    cols, grps, fams_, ranks = [], [], [], []
    for f_i, fam in enumerate(families):
        starts = fam.indptr[nodes]
        lens = np.minimum(fam.indptr[nodes + 1] - starts, SCORE_WINDOW if cs else cap)
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
        keep = within < cap
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


def order_group(cols, grp, rank, key, scorer, n_par):
    """The distinct nodes of one candidate list in the cell's order: arrival (no key); `vr` by -cosine; `sp` by -support (the number of distinct
    parents that name the node), then -cosine; `rk` by the smallest rank any parent gives it, then -cosine. Ties to the lower id."""
    u, first_idx, inv = np.unique(cols, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    if not key:
        return u[np.argsort(first_idx, kind="stable")]
    cos = scorer.get(u, "q")
    if key == "vr":
        return u[np.lexsort((u, -cos))]
    if key == "sp":
        pair = np.unique(inv.astype(np.int64) * n_par + grp)
        sup = np.bincount(pair // n_par, minlength=u.size)
        return u[np.lexsort((u, -cos, -sup))]
    o2 = np.lexsort((rank, inv))
    invs = inv[o2]
    starts = np.flatnonzero(np.concatenate(([True], invs[1:] != invs[:-1])))
    best = rank[o2][starts]
    return u[np.lexsort((u, -cos, best))]


def order_fresh(cols, grp, fam, rank, key, rr, scorer, n_par):
    """The fresh candidates of a hop in the cell's order. rr: every family's list is ordered by the key on its own, and the lists are merged
    by place (every family's first, then every family's second, ...), the first place a node takes being kept."""
    if cols.size == 0:
        return cols
    if not rr:
        return order_group(cols, grp, rank, key, scorer, n_par)
    nodes, pos, fm = [], [], []
    for f in np.unique(fam):
        m = fam == f
        o = order_group(cols[m], grp[m], rank[m], key, scorer, n_par)
        nodes.append(o)
        pos.append(np.arange(o.size, dtype=np.int64))
        fm.append(np.full(o.size, int(f), dtype=np.int64))
    nodes, pos, fm = np.concatenate(nodes), np.concatenate(pos), np.concatenate(fm)
    nodes = nodes[np.lexsort((fm, pos))]
    return m3b_pools._first_occurrence_unique(nodes)


def expand4(seeds, fams1, fams2, setting, mods, scorer):
    """The bounded walk of zpl3g.expand3 with the four PL4 options. mods = the walk's option names. With none of sp, rk, rr, eq on it is
    expand3 exactly (cs and vr as there); fams1 = the families walked from the seeds, fams2 = the families walked at hops 2 and 3."""
    first = m3b_pools._first_occurrence_unique
    cs = "cs" in mods
    key = "vr" if "vr" in mods else ("sp" if "sp" in mods else ("rk" if "rk" in mods else ""))
    rr, eq = "rr" in mods, "eq" in mods
    cap = int(setting["per_seed_cap"])
    hop1 = first(capped_parts(seeds, fams1, cap, scorer, cs)[0])
    hops = int(setting["hops"])
    if hops == 1:
        return hop1
    cap2 = int(setting["per_frontier_cap"])
    limit = int(setting["visited_cap"])
    visited = np.union1d(seeds, hop1)
    out = [hop1]
    frontier = hop1
    for h in range(2, hops + 1):
        rem = limit - int(visited.size)
        if rem <= 0 or frontier.size == 0:
            break
        budget = -(-rem // (hops - h + 1)) if eq else rem
        cols, grp, fam, rank = capped_parts(frontier, fams2, cap2, scorer, cs)
        if cols.size == 0:
            break
        k = np.isin(cols, visited, invert=True)
        fresh = order_fresh(cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))[:budget]
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return np.concatenate(out)


TIMES = None                                                 # a dict while `time` runs: seconds per walk


def pool_state4(plan, d, s, f2, f1, seeds5, fams, qv, dvec, n_nodes):
    """Everything a question's pools are made of: the ranked lists and each walk's expansion with its orders (zpl3g.pool_state3 with expand4).
    It takes and reads no labels."""
    st = Z3.State3()
    st.L = {}
    if "D" in plan.lists:
        st.L["D"] = P2.Lst(d)
    if "S" in plan.lists:
        st.L["S"] = P2.Lst(s)
    if "F2" in plan.lists:
        st.L["F2"] = P2.Lst(f2)
    if "F1" in plan.lists:
        st.L["F1"] = P2.Lst(f1)
    st.d200, st.s200 = (d[:200], s[:200]) if plan.fu else (None, None)
    scorer = None
    if plan.need_vec:
        rvs = {"q": np.asarray(qv, dtype=np.float64)}
        if plan.need_bridge:
            top = dvec(np.asarray(f1[:Z3.BRIDGE_DOCS], dtype=np.int64)).astype(np.float64).mean(axis=0)
            rvs["b"] = rvs["q"] + top / np.sqrt((top * top).sum())
        scorer = Z3.Scorer(dvec, rvs, n_nodes)
    seeds = {"": seeds5}
    if plan.need_f1:
        seeds["s10"] = np.asarray(f1[:10], dtype=np.int64)
        seeds["s20"] = np.asarray(f1[:20], dtype=np.int64)
    st.W = {}
    for w, grp in plan.walks.items():
        t0 = time.perf_counter() if TIMES is not None else 0.0
        regime, stg, mods = w
        sd = seeds["s10" if "s10" in mods else ("s20" if "s20" in mods else "")]
        h1, h2 = REG3[regime]
        exp = expand4(sd, [fams[f] for f in h1], [fams[f] for f in h2], P2.SETTINGS[stg], mods, scorer)
        wk = Z3.Walk3()
        wk.exp = exp
        wk.ro, wk.inv, wk.se, wk.sx = {}, {}, None, None
        if grp["K"]:
            for rv in plan.walk_rvs[w]:
                sc = scorer.get(exp, rv) if exp.size else np.empty(0, np.float64)
                ro = np.lexsort((exp, -sc))
                inv = np.empty(exp.size, np.int64)
                inv[ro] = np.arange(exp.size)
                wk.ro[rv], wk.inv[rv] = ro, inv
            wk.se = np.argsort(exp, kind="stable")
            wk.sx = exp[wk.se]
        st.W[w] = wk
        if TIMES is not None:
            TIMES[w] = TIMES.get(w, 0.0) + time.perf_counter() - t0
    if scorer is not None:
        if TIMES is not None:
            TIMES["_scored"] = TIMES.get("_scored", 0) + scorer.n_scored
        scorer.release()
    return st


# ── the engine ───────────────────────────────────────────────────────────────

def iter_states4(op, src, plan, lo, hi, chunk=64):
    """Yield (question row, State3) for rows lo..hi: no labels are read here (a source whose label arrays are None runs the same)."""
    M3A = op.m3a
    fams = src.fams
    dvec = src.dvec if plan.need_vec else None
    for c0 in range(lo, hi, chunk):
        c1 = min(c0 + chunk, hi)
        d = np.asarray(src.dense[c0:c1], dtype=np.int64)
        s = np.asarray(src.splade[c0:c1], dtype=np.int64)
        seeds = M3A.stable_union_rows(d[:, :5], s[:, :5])
        none = [None] * (c1 - c0)
        f2 = M3A.rrf_fused_rows(d[:, :200], s[:, :200], P2.RRF_CONST) if "F2" in plan.lists else none
        f1 = M3A.rrf_fused_rows(d, s, P2.RRF_CONST) if plan.need_f1 else none
        qv = src.qvec(c0, c1) if plan.need_vec else none
        for j in range(c1 - c0):
            yield c0 + j, pool_state4(plan, d[j], s[j], f2[j], f1[j], seeds[j], fams, qv[j], dvec, src.n)


def evaluate4(op, src, plan, lo, hi, every=300):
    Q = hi - lo
    P = np.zeros((Q, plan.C), dtype=np.int16)
    S = np.zeros((Q, plan.C), dtype=np.int32)
    t0 = last = time.time()
    for q, st in iter_states4(op, src, plan, lo, hi):
        Z3.grade3(plan, st, src.golds[q], P[q - lo], S[q - lo])
        if (q - lo + 1) % every == 0 and time.time() - last > 20:
            last = time.time()
            log(f"{src.name}/{src.carve}/{src.first}: {q - lo + 1}/{Q} questions, {last - t0:.0f} s ({P2.peak_gb()} GB)")
    return P, S


_SHA = {}


def code_path_sha256():
    """One hash of PL4's pool code, its constants and PL3's code-path hash (which holds PL2's) (G-U): the same on every dataset."""
    if "code" not in _SHA:
        h = hashlib.sha256()
        for fn in (parse_walk4, parse4, Plan4, capped_parts, order_group, order_fresh, expand4, pool_state4, iter_states4, evaluate4):
            h.update(inspect.getsource(fn).replace("\r\n", "\n").encode("utf-8"))
        h.update(json.dumps(constants4(), sort_keys=True).encode("utf-8"))
        h.update(Z3.code_path_sha256().encode("utf-8"))
        _SHA["code"] = h.hexdigest()
    return _SHA["code"]


def script_sha256():
    if "script" not in _SHA:
        _SHA["script"] = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return _SHA["script"]


# ── small helpers ────────────────────────────────────────────────────────────

log, stop, rel, rjson, wjson, utc = P2.log, P2.stop, P2.rel, P2.rjson, P2.wjson, P2.utc
load_grid4, heads_of = Z3.load_grid3, Z3.heads_of
rows_of_grid4 = Z3.rows_of_grid3


def grid_path4(d, carve, first, tag=""):
    return PL4 / d / f"grid_{carve}_{first}{('_' + tag) if tag else ''}"


def save_grid4(base, src, plan, P, S, lo, hi, seconds, part=None, stage="PL4 grid"):
    p = Path(str(base) + (f".part{part[0]}of{part[1]}" if part else "") + ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez(tmp, names=np.asarray(plan.names, dtype=str), present=P, size=S, qids=np.asarray(src.qids[lo:hi], dtype=str),
             gold_counts=src.gold_counts[lo:hi].astype(np.int32), lo=np.asarray(lo), hi=np.asarray(hi))
    os.replace(tmp, p)
    head = {"stage": stage, "declared_in": DOC, "dataset": src.name, "carve": src.carve, "first": src.first, "questions": int(hi - lo),
            "rows": [int(lo), int(hi)], "part": list(part) if part else None, "cells": plan.C, "cells_sha256": P2.cells_sha256(plan.names),
            "graph": getattr(src, "graph_kind", None), "population_digest": src.pop_digest, "code_path_sha256": code_path_sha256(),
            "pl3_code_path_sha256": Z3.code_path_sha256(), "pl2_code_path_sha256": P2.code_path_sha256(), "script_sha256": script_sha256(),
            "constants_sha256": hashlib.sha256(json.dumps(constants4(), sort_keys=True).encode()).hexdigest(), "npz": rel(p),
            "npz_sha256": P2.sha256_file(p), "seconds": round(seconds, 1), "peak_gb": P2.peak_gb(), "host": P2.HOST, "utc": utc()}
    wjson(Path(str(p)[:-4] + ".json"), head)
    return p, head


# ── commands: cells, time ────────────────────────────────────────────────────

def cmd_cells(a):
    new, uni = cell_list4(), cell_list_union4()
    fam = {}
    for n in new:
        fam[tags_of4(n)] = fam.get(tags_of4(n), 0) + 1
    log(f"PL4 cells {len(new)} ({len(set(new))} distinct), sha256 {P2.cells_sha256(new)}; union with PL2's and PL3's {len(uni)} ({len(set(uni))} distinct), "
        f"sha256 {P2.cells_sha256(uni)}")
    log(f"walks {len(walks4())}; tags {fam}")
    log(f"code path sha256 {code_path_sha256()}")
    if a.write:
        wjson(PL4 / "cells.json", {"cells": new, "sha256": P2.cells_sha256(new), "union_sha256": P2.cells_sha256(uni), "n": len(new)})
    return 0


def cmd_time(a):
    global TIMES
    op = P2.opened()
    names = cell_list4()
    if a.walks:
        names = [n for n in names if a.walks in n]
    plan = Plan4(names)
    src = P2.load_source(a.ds, "dev", "frozen")
    n = min(a.n, src.nq)
    TIMES = {}
    t0 = time.time()
    for q, st in iter_states4(op, src, plan, 0, n):
        pass
    tot = time.time() - t0
    sc = TIMES.pop("_scored", 0)
    log(f"{a.ds}: {n} questions, {plan.C} cells, {len(plan.walks)} walks, {tot:.1f} s = {tot / n:.3f} s/question; nodes scored {sc / n:.0f}/question; peak {P2.peak_gb()} GB")
    for w, t in sorted(TIMES.items(), key=lambda kv: -kv[1])[:int(a.top)]:
        log(f"   {walk_name(*w):44s} {1000 * t / n:8.1f} ms/question")
    return 0


# ── G-R: PL2's and PL3's cells, re-built by this engine ──────────────────────

def cmd_gr(a):
    t0 = time.time()
    op = P2.opened()
    names = cell_list_union4()[:len(P2.cell_list()) + len(Z3.cell_list3())]
    plan = Plan4(names)
    src = P2.load_source(a.ds, "dev", "frozen")
    r2 = P2.load_grid(a.ds, "dev", "frozen")
    r3 = load_grid4(Z3.grid_path3(a.ds, "dev", "frozen"))
    n = min(a.n, src.nq)
    if r2["names"] + r3["names"] != names or r2["qids"][:n] != list(src.qids[:n]) or r3["qids"][:n] != list(src.qids[:n]):
        stop(f"{a.ds}: PL2's and PL3's dev grids do not name the same cells or questions")
    P, S = evaluate4(op, src, plan, 0, n)
    ref_p = np.concatenate([r2["present"][:n], r3["present"][:n]], axis=1)
    ref_s = np.concatenate([r2["size"][:n], r3["size"][:n]], axis=1)
    bad_p = np.argwhere(P != ref_p)
    bad_s = np.argwhere(S != ref_s)
    out = {"stage": "PL4 G-R", "declared_in": DOC, "dataset": a.ds, "host": P2.HOST, "questions": int(n), "cells": len(names),
           "pl2_cells": len(P2.cell_list()), "pl3_cells": len(Z3.cell_list3()), "elements": int(P.size), "present_mismatches": int(len(bad_p)),
           "size_mismatches": int(len(bad_s)),
           "first_mismatch": ([int(bad_p[0][0]), names[int(bad_p[0][1])]] if len(bad_p) else (
               [int(bad_s[0][0]), names[int(bad_s[0][1])]] if len(bad_s) else None)),
           "pl3_code_path_sha256": Z3.code_path_sha256(), "pl2_code_path_sha256": P2.code_path_sha256(), "code_path_sha256": code_path_sha256(),
           "script_sha256": script_sha256(), "verdict": "PASS" if not len(bad_p) and not len(bad_s) else "FAIL",
           "seconds": round(time.time() - t0, 1), "peak_gb": P2.peak_gb(), "utc": utc()}
    wjson(PL4 / f"gr_{a.ds}{'_host' if P2.HOST else ''}.json", out)
    log(f"G-R {a.ds}: {out['verdict']}, {n} questions x {len(names)} cells = {out['elements']} elements, mismatches {out['present_mismatches']}/{out['size_mismatches']} "
        f"({out['seconds']} s)")
    return 0 if out["verdict"] == "PASS" else 3


def gates_passed4(ds):
    ref = ds if (ds in LAPTOP5 and not P2.HOST) else "musique"
    p = PL4 / f"gr_{ref}.json"
    if not p.is_file() or rjson(p)["verdict"] != "PASS":
        stop(f"{p.name} is not PASS: run `zpl4g.py gr --ds {ref}` first")
    if P2.HOST:
        h = PL4 / "gr_musique_host.json"
        if not h.is_file() or rjson(h)["verdict"] != "PASS":
            stop("gr_musique_host.json is not PASS: run `zpl4g.py gr --ds musique --host` first")
    st = PL4 / "selftest.json"
    if not st.is_file() or rjson(st)["verdict"] != "PASS":
        stop("selftest.json is not PASS: run `zpl4g.py selftest` first")


def cmd_grid(a):
    t0 = time.time()
    P2.HOST = bool(a.host)
    gates_passed4(a.ds)
    names = cell_list4() if a.cells == "new" else [str(x) for x in rjson(a.cells)]
    if a.cells == "new":
        require_cells4()
    plan = Plan4(names)
    op = P2.opened()
    src = P2.load_source(a.ds, a.carve, a.first)
    lo, hi, part = 0, src.nq, None
    if a.part:
        i, k = int(a.part[0]), int(a.part[1])
        lo, hi = i * src.nq // k, (i + 1) * src.nq // k
        part = (i, k)
    log(f"{a.ds}/{a.carve}/{a.first}: {src.nq} questions, rows {lo}..{hi}, {plan.C} cells, {len(plan.walks)} walks, graph {src.graph_kind} ({P2.peak_gb()} GB)")
    P, S = evaluate4(op, src, plan, lo, hi)
    tag = a.tag or ("" if a.cells == "new" else "sub")
    p, head = save_grid4(grid_path4(a.ds, a.carve, a.first, tag), src, plan, P, S, lo, hi, time.time() - t0, part)
    log(f"{a.ds}/{a.carve}/{a.first}: filed {rel(p)} in {head['seconds']} s")
    return 0


# ── select ───────────────────────────────────────────────────────────────────

def union_dev4(ds):
    """PL2's, PL3's and PL4's dev grids side by side (same questions, disjoint cells), their heads checked against the code that is running."""
    g2 = P2.load_grid(ds, "dev", "frozen")
    for hp in heads_of(P2.grid_path(ds, "dev", "frozen")):
        hd = rjson(hp)
        if hd["code_path_sha256"] != P2.code_path_sha256() or hd["cells_sha256"] != P2.EXPECT_CELLS_SHA256:
            stop(f"G-U: {hp.name} (PL2) was written by other pool code or another cell list than the one that is running")
    g3 = load_grid4(Z3.grid_path3(ds, "dev", "frozen"))
    for hp in heads_of(Z3.grid_path3(ds, "dev", "frozen")):
        hd = rjson(hp)
        if hd["code_path_sha256"] != Z3.code_path_sha256() or hd["cells_sha256"] != Z3.EXPECT3_SHA256:
            stop(f"G-U: {hp.name} (PL3) was written by other pool code or another cell list than the one that is running")
    g4 = load_grid4(grid_path4(ds, "dev", "frozen"))
    h4 = heads_of(grid_path4(ds, "dev", "frozen"))
    if not h4:
        stop(f"{ds}: no PL4 grid heads")
    for hp in h4:
        hd = rjson(hp)
        if hd["code_path_sha256"] != code_path_sha256() or hd["cells_sha256"] != EXPECT4_SHA256:
            stop(f"G-U: {hp.name} was written by other PL4 code or another cell list than the one that is running")
    if g2["qids"] != g3["qids"] or g2["qids"] != g4["qids"] or not np.array_equal(g2["gold_counts"], g4["gold_counts"]):
        stop(f"{ds}: the three dev grids are not on the same questions")
    if len(set(g2["names"]) | set(g3["names"]) | set(g4["names"])) != len(g2["names"]) + len(g3["names"]) + len(g4["names"]):
        stop("the cells of the three grids overlap")
    if P2.cells_sha256(g4["names"]) != EXPECT4_SHA256:
        stop("the PL4 dev grid is not the declared cell list")
    return {"names": g2["names"] + g3["names"] + g4["names"], "present": np.concatenate([g2["present"], g3["present"], g4["present"]], axis=1),
            "size": np.concatenate([g2["size"], g3["size"], g4["size"]], axis=1), "qids": g2["qids"], "gold_counts": g2["gold_counts"]}, h4


READINGS = ("all", "pl3", "new")
TAGS4 = ("sp", "rk", "eq", "rr", "seeds")


def frontier_pick(rows, n):
    """The matched-node-count pick: the highest ALL among cells with at most n mean nodes (ties to fewer nodes, then the name)."""
    within = [r for r in rows if r["nodes"] <= n]
    if not within:
        return None
    return min(within, key=lambda r: (-r["all"], r["nodes"], r["name"]))


def cmd_select(a):
    P2.HOST = bool(a.host)
    gates_passed4(a.ds)
    g, heads = union_dev4(a.ds)
    names = g["names"]
    rows = rows_of_grid4(g, a.ds, "dev")
    table = [{"name": n, "family": family4(n), "tags": list(tags_of4(n)), "nodes": r["candidates_mean"], "all": r["all_gold_at_pool"], "any": r["any_gold_at_pool"],
              "coverage": r["coverage_micro"], "p95": r["candidates_p95"], "max": r["candidates_max"]} for n, r in zip(names, rows)]
    readings = {"all": table, "pl3": [t for t in table if t["family"] in ("pl2", "pl3")], "new": [t for t in table if t["family"] == "pl4"]}
    for tg in TAGS4:
        readings[f"pl3+{tg}"] = [t for t in table if t["family"] in ("pl2", "pl3") or tg in t["tags"]]
        readings[f"only:{tg}"] = [t for t in table if t["family"] == "pl4" and tg in t["tags"]]
    out = {"stage": "PL4 select", "declared_in": DOC, "dataset": a.ds, "carve": "dev", "first": "frozen", "questions": len(g["qids"]), "bound": BOUND,
           "cells_sha256_pl4": EXPECT4_SHA256, "cells_sha256_pl3": Z3.EXPECT3_SHA256, "cells_sha256_pl2": P2.EXPECT_CELLS_SHA256,
           "code_path_sha256": code_path_sha256(), "pl3_code_path_sha256": Z3.code_path_sha256(), "pl2_code_path_sha256": P2.code_path_sha256(),
           "script_sha256": script_sha256(), "utc": utc(),
           "grid_heads": [{"file": rel(hp), "code_path_sha256": rjson(hp)["code_path_sha256"], "npz_sha256": rjson(hp)["npz_sha256"],
                           "seconds": rjson(hp)["seconds"], "peak_gb": rjson(hp)["peak_gb"]} for hp in heads]}
    out["picks"] = {}
    for floor in FLOORS:
        for rd, rs in readings.items():
            if not rs:
                continue
            c, reached = P2.pick(rs, floor)
            out["picks"][f"R{int(round(floor * 100))}_{rd}"] = {**c, "reached": reached, "floor": floor}
    out["short"] = {f"{int(round(f * 100))}": not out["picks"][f"R{int(round(f * 100))}_all"]["reached"] for f in FLOORS}
    out["frontier_picks"] = {}
    for rd in ("all", "pl3", "new"):
        out["frontier_picks"][rd] = {str(n): frontier_pick(readings[rd], n) for n in BUDGETS}
    out["frontier"] = {rd: P2.frontier(readings[rd]) for rd in ("all", "pl3", "new")}
    grid_n = (25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500)
    out["bought"] = [{"nodes": n, "pl3": P2.front_at(out["frontier"]["pl3"], n), "new": P2.front_at(out["frontier"]["new"], n), "all": P2.front_at(out["frontier"]["all"], n),
                      "gain": (None if P2.front_at(out["frontier"]["all"], n) is None or P2.front_at(out["frontier"]["pl3"], n) is None
                               else P2.front_at(out["frontier"]["all"], n) - P2.front_at(out["frontier"]["pl3"], n))} for n in grid_n]
    pl3_sel = ROOT / "outputs" / "pl3" / a.ds / "select.json"
    same = None
    if pl3_sel.is_file():
        f3 = rjson(pl3_sel)["picks"]
        same = f3["R95_all"]["name"] == out["picks"]["R95_pl3"]["name"] and f3["R98_all"]["name"] == out["picks"]["R98_pl3"]["name"]
    out["pl3_pick_equals_filed"] = same
    out["old"] = P2.old_cell(None, a.ds)
    out["table"] = table
    wjson(PL4 / a.ds / "select.json", out)
    for k in ("R95_all", "R95_pl3", "R95_new", "R98_all", "R98_pl3", "R98_new"):
        c = out["picks"][k]
        log(f"{a.ds} {k}: {c['name']}  ALL {c['all']:.4f} at {c['nodes']:.1f} nodes{'' if c['reached'] else '  (SHORT)'}")
    log(f"{a.ds}: PL3's pick re-read equals its filed select.json: {same}")
    return 0


# ── confirm and call ─────────────────────────────────────────────────────────

def confirm_names4(sel):
    names = []

    def add(n):
        if n and n not in names:
            names.append(n)
    for k in ("R95_all", "R98_all", "R95_pl3", "R98_pl3", "R95_new", "R98_new"):
        add(sel["picks"][k]["name"])
    for rd in ("all", "pl3"):
        for n in BUDGETS:
            c = sel["frontier_picks"][rd][str(n)]
            add(c["name"] if c else None)
    add(sel["old"])
    return names


def cmd_confirm(a):
    P2.HOST = bool(a.host)
    gates_passed4(a.ds)
    op = P2.opened()
    sel = rjson(PL4 / a.ds / "select.json")
    names = confirm_names4(sel)
    plan = Plan4(names)
    for first in a.first:
        t0 = time.time()
        try:
            src = P2.load_source(a.ds, "eval", first)
        except SystemExit as e:
            log(f"{a.ds}/eval/{first}: not read here ({e})")
            continue
        if plan.need_vec:
            try:
                src.dvec(np.asarray([0]))
            except Exception as e:
                log(f"{a.ds}/eval/{first}: the ranking vectors are not here ({e}); the vector-using cells are left for the host")
                continue
        P, S = evaluate4(op, src, plan, 0, src.nq)
        save_grid4(grid_path4(a.ds, "eval", first, "confirm"), src, plan, P, S, 0, src.nq, time.time() - t0, stage="PL4 confirm grid")
        g = {"names": plan.names, "present": P, "size": S, "gold_counts": src.gold_counts}
        rows = {r["pool"]: r for r in rows_of_grid4(g, a.ds, "eval")}
        res = {"stage": "PL4 confirm", "declared_in": DOC, "dataset": a.ds, "first": first, "questions": src.nq, "utc": utc(), "cells": {}}
        for n in plan.names:
            r = rows[n]
            k = int(round(r["all_gold_at_pool"] * src.nq))
            res["cells"][n] = {"all": r["all_gold_at_pool"], "all_ci": P2.wilson(k, src.nq), "any": r["any_gold_at_pool"], "coverage": r["coverage_micro"],
                               "nodes": r["candidates_mean"], "p95": r["candidates_p95"], "max": r["candidates_max"],
                               "over_bound": bool(r["candidates_mean"] > BOUND)}
        res["named"] = {}
        for k, c in sel["picks"].items():
            if k.split("_", 1)[1] in READINGS and c["name"] in res["cells"]:
                e = res["cells"][c["name"]]
                res["named"][k] = {"name": c["name"], "dev_all": c["all"], "dev_nodes": c["nodes"], "eval_all": e["all"], "eval_nodes": e["nodes"],
                                   "floor": c["floor"], "confirmed": bool(e["all"] >= c["floor"]), "short_by": max(0.0, c["floor"] - e["all"]),
                                   "dev_to_eval": e["all"] - c["all"], "over_bound": e["over_bound"]}
        e = res["cells"][sel["old"]]
        res["old"] = {"name": sel["old"], "eval_all": e["all"], "eval_nodes": e["nodes"]}
        wjson(PL4 / a.ds / f"confirm_{first}.json", res)
        for k, v in res["named"].items():
            log(f"{a.ds}/{first} {k}: {v['name']} eval ALL {v['eval_all']:.4f} at {v['eval_nodes']:.1f} nodes "
                f"({'CONFIRMED' if v['confirmed'] else 'SHORT by %.2f pts' % (100 * v['short_by'])}; dev {v['dev_all']:.4f})" + ("  OVER_BOUND" if v["over_bound"] else ""))
    return 0


def status_of(pr, new_reaches, old_reaches):
    return Z3.status_of(pr, new_reaches, old_reaches)


def frontier_status(prs):
    """The frontier reading's status of one dataset from its per-budget paired results: BELOW if any budget's ALL difference is below 0 (upper end < 0),
    ABOVE if at least two budgets are above 0 (lower end > 0) and none is below, LEVEL otherwise."""
    below = any(p["all_diff_ci"][1] < 0 for p in prs)
    above = sum(p["all_diff_ci"][0] > 0 for p in prs) >= 2
    return "BELOW" if below else ("ABOVE" if above else "LEVEL")


def cmd_call(a):
    out = {"stage": "PL4 call", "declared_in": DOC, "utc": utc(), "datasets": {}}
    codes = {}
    for k, ds in enumerate(SEVEN):
        rng = np.random.default_rng([P2.BOOT_SEED, k])
        sp, cp = PL4 / ds / "select.json", grid_path4(ds, "eval", "frozen", "confirm")
        if not sp.is_file() or not Path(str(cp) + ".npz").is_file():
            out["datasets"][ds] = {"status": "NOT_READ", "why": "no PL4 select or confirm file here"}
            continue
        sel = rjson(sp)
        codes[ds] = sel["code_path_sha256"]
        g = load_grid4(cp)
        at = {n: i for i, n in enumerate(g["names"])}
        gc_ = g["gold_counts"]

        def arrs(n):
            i = at[n]
            return (g["present"][:, i].astype(np.int64) == gc_), g["size"][:, i].astype(np.float64)
        rec = {}
        for tag, nk, ok in (("R95", "R95_all", "R95_pl3"), ("R98", "R98_all", "R98_pl3"), ("R95_new_only", "R95_new", "R95_pl3")):
            new_name, old_name = sel["picks"][nk]["name"], sel["picks"][ok]["name"]
            new, old = arrs(new_name), arrs(old_name)
            pr = P2.paired(new, old, rng)
            floor = FLOORS[1] if tag == "R98" else FLOORS[0]
            nr, orr = bool(new[0].mean() >= floor), bool(old[0].mean() >= floor)
            rec[tag] = {"status": status_of(pr, nr, orr), "new": new_name, "old": old_name, "same_cell": new_name == old_name, "new_all": float(new[0].mean()),
                        "old_all": float(old[0].mean()), "new_nodes": float(new[1].mean()), "old_nodes": float(old[1].mean()), "new_reaches": nr, "old_reaches": orr,
                        "within_bound": bool(sel["picks"][nk]["nodes"] <= BOUND), **pr}
        fr = {}
        for n in BUDGETS:
            cn, co = sel["frontier_picks"]["all"][str(n)], sel["frontier_picks"]["pl3"][str(n)]
            if cn is None or co is None:
                fr[str(n)] = {"status": "NONE", "why": "no cell within this many dev nodes"}
                continue
            new, old = arrs(cn["name"]), arrs(co["name"])
            pr = P2.paired(new, old, rng)
            pst = "BELOW" if pr["all_diff_ci"][1] < 0 else ("ABOVE" if pr["all_diff_ci"][0] > 0 else "LEVEL")
            fr[str(n)] = {"status": pst, "new": cn["name"], "old": co["name"], "same_cell": cn["name"] == co["name"], "new_dev_all": cn["all"], "old_dev_all": co["all"],
                          "new_dev_nodes": cn["nodes"], "old_dev_nodes": co["nodes"], "new_all": float(new[0].mean()), "old_all": float(old[0].mean()),
                          "new_nodes": float(new[1].mean()), "old_nodes": float(old[1].mean()), **pr}
        used = [v for v in fr.values() if v["status"] != "NONE"]
        out["datasets"][ds] = {**rec["R95"], "status": rec["R95"]["status"], "r98": rec["R98"], "new_only": rec["R95_new_only"], "frontier": fr,
                               "frontier_status": frontier_status(used)}
    read = {k: v for k, v in out["datasets"].items() if v["status"] != "NOT_READ"}
    out["datasets_read"] = sorted(read)
    out["code_path_equal_everywhere"] = len(set(codes.values())) <= 1
    if len(set(codes.values())) > 1:
        stop(f"G-U: the code-path hash differs between datasets: {codes}")
    n_below = sum(v["status"] == "BELOW" for v in read.values())
    n_above = sum(v["status"] == "ABOVE" for v in read.values())
    f_below = sum(v["frontier_status"] == "BELOW" for v in read.values())
    f_above = sum(v["frontier_status"] == "ABOVE" for v in read.values())
    inside = all(v["within_bound"] for v in read.values())
    complete = len(read) == len(SEVEN)
    out["call"] = ("CARRY_PL4" if (complete and n_below == 0 and n_above >= 1 and inside) else ("INCOMPLETE" if not complete else "NOT_CARRIED"))
    out["call_frontier"] = ("CARRY_PL4_F" if (complete and n_below == 0 and f_below == 0 and f_above >= 1 and inside)
                            else ("INCOMPLETE" if not complete else "NOT_CARRIED"))
    out["counts"] = {"old": {"BELOW": n_below, "ABOVE": n_above, "LEVEL": len(read) - n_below - n_above},
                     "frontier": {"BELOW": f_below, "ABOVE": f_above, "LEVEL": len(read) - f_below - f_above}}
    wjson(PL4 / "call.json", out)
    for ds, v in out["datasets"].items():
        log(f"call {ds}: {v['status']} / frontier {v.get('frontier_status')}" + ("" if v["status"] == "NOT_READ" else
            f"  NEW {v['new']} ALL {v['new_all']:.4f} @ {v['new_nodes']:.0f}; PL3 {v['old']} ALL {v['old_all']:.4f} @ {v['old_nodes']:.0f}"))
    log(f"call (old rule): {out['call']} (BELOW {n_below}, ABOVE {n_above}, read {len(read)}/7); call (frontier reading): {out['call_frontier']} "
        f"(BELOW {f_below}, ABOVE {f_above})")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def ref_walk4(seeds, fams1, fams2, setting, mods, qv, dvec):
    """The PL4 walk as a per-node loop, written from the declaration (docs section 4), for G-X4 and the engine test."""
    cs, rr, eq = "cs" in mods, "rr" in mods, "eq" in mods
    key = "vr" if "vr" in mods else ("sp" if "sp" in mods else ("rk" if "rk" in mods else ""))
    cap, cap2, limit, hops = setting["per_seed_cap"], setting.get("per_frontier_cap"), setting.get("visited_cap"), setting["hops"]
    memo = {}

    def one(nodes):
        nodes = [int(x) for x in nodes]
        miss = sorted(set(x for x in nodes if x not in memo))
        if miss:
            for x, v in zip(miss, P2.dot64(dvec(np.asarray(miss, dtype=np.int64)), qv)):
                memo[x] = float(v)
        return [memo[x] for x in nodes]

    def lists(v, fams, c):
        out = []
        for fi, f in enumerate(fams):
            row = [int(x) for x in f.col[f.indptr[v]:f.indptr[v + 1]]]
            if cs and row:
                row = row[:SCORE_WINDOW]
                sc = one(row)
                row = [row[i] for i in sorted(range(len(row)), key=lambda i: (-sc[i], row[i]))]
            out += [(x, fi, r) for r, x in enumerate(row[:c])]
        return out

    def distinct(ents):
        seen, out = set(), []
        for x, _p, _f, _r in ents:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def order(ents):
        nodes = distinct(ents)
        if not key or not nodes:
            return nodes
        cos = dict(zip(nodes, one(nodes)))
        if key == "vr":
            return sorted(nodes, key=lambda x: (-cos[x], x))
        if key == "sp":
            parents = {}
            for (y, p_, _f, _r) in ents:
                parents.setdefault(y, set()).add(p_)
            return sorted(nodes, key=lambda x: (-len(parents[x]), -cos[x], x))
        best = {}
        for (y, _p, _f, r) in ents:
            if y not in best or r < best[y]:
                best[y] = r
        return sorted(nodes, key=lambda x: (best[x], -cos[x], x))
    hop1 = distinct([(x, 0, fi, r) for v in seeds for (x, fi, r) in lists(int(v), fams1, cap)])
    out = list(hop1)
    if hops == 1:
        return np.asarray(out, dtype=np.int64)
    visited = set(int(x) for x in seeds) | set(hop1)
    frontier = hop1
    for h in range(2, hops + 1):
        rem = limit - len(visited)
        if rem <= 0 or not frontier:
            break
        budget = -(-rem // (hops - h + 1)) if eq else rem
        ents = [(x, pi, fi, r) for pi, v in enumerate(frontier) for (x, fi, r) in lists(int(v), fams2, cap2) if x not in visited]
        if not ents:
            break
        if rr:
            per = {fi: order([e for e in ents if e[2] == fi]) for fi in sorted(set(e[2] for e in ents))}
            merged = sorted(((pos, fi, x) for fi, lst in per.items() for pos, x in enumerate(lst)), key=lambda t: (t[0], t[1]))
            seen, fresh = set(), []
            for _pos, _fi, x in merged:
                if x not in seen:
                    seen.add(x)
                    fresh.append(x)
        else:
            fresh = order(ents)
        fresh = fresh[:budget]
        if not fresh:
            break
        out += fresh
        visited |= set(fresh)
        frontier = fresh
    return np.asarray(out, dtype=np.int64)


def brute_cell4(op, name, d, s, g, fams, seeds5, qv, dvec, exps):
    """One cell of one question by explicit sets (exps caches the per-node walks of this question), for the engine to be held to."""
    M3A = op.m3a
    p = parse4(name)
    f2 = M3A.rrf_fused_rows(d[None, :200], s[None, :200], P2.RRF_CONST)[0]
    f1 = M3A.rrf_fused_rows(d[None, :], s[None, :], P2.RRF_CONST)[0]
    lists = {"D": d, "S": s, "F2": f2, "F1": f1}
    if p[0] == "FU":
        pool = np.union1d(d[:200], s[:200])
    elif p[0] == "R":
        pool = lists[p[1]][:p[2]]
    else:
        base = lists[p[1]][:p[2]]
        regime, stg, mods = p[3]
        if p[3] not in exps:
            sd = np.asarray(f1[:10] if "s10" in mods else (f1[:20] if "s20" in mods else seeds5), dtype=np.int64)
            h1, h2 = REG3[regime]
            exps[p[3]] = ref_walk4(sd, [fams[f] for f in h1], [fams[f] for f in h2], P2.SETTINGS[stg], mods, qv, dvec)
        exp = exps[p[3]]
        if p[0] == "G":
            pool = np.union1d(base, exp)
        else:
            cand = exp[np.isin(exp, base, invert=True)]
            r = np.asarray(qv, dtype=np.float64)
            if p[5] == "b":
                top = dvec(np.asarray(f1[:Z3.BRIDGE_DOCS], dtype=np.int64)).astype(np.float64).mean(axis=0)
                r = r + top / np.sqrt((top * top).sum())
            sc = P2.dot64(dvec(cand), r) if cand.size else np.empty(0)
            order = sorted(range(cand.size), key=lambda i: (-sc[i], int(cand[i])))
            pool = np.union1d(base, cand[order[:p[4]]])
    return int(np.isin(g, pool).sum()), int(np.unique(pool).size)


def selftest_engine4(op, ds, n_q, per_walk):
    """The engine against brute_cell4 on a sample of questions: every walk, a few of its cells."""
    src = P2.load_source(ds, "dev", "frozen")
    names = cell_list4()
    plan = Plan4(names)
    hi = min(n_q, src.nq)
    P, S = evaluate4(op, src, plan, 0, hi)
    M3A = op.m3a
    bad, checks = [], 0
    rng = np.random.default_rng(11)
    by_walk = {}
    for i, nm in enumerate(names):
        p = parse4(nm)
        by_walk.setdefault(p[3], []).append(i)
    for q in range(0, hi, max(1, hi // 4)):
        d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
        seeds5 = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
        qv = src.qvec(q, q + 1)[0]
        exps = {}
        for w, idx in by_walk.items():
            for i in rng.choice(idx, size=min(per_walk, len(idx)), replace=False):
                nm = names[int(i)]
                got = brute_cell4(op, nm, d, s, src.golds[q], src.fams, seeds5, qv, src.dvec, exps)
                checks += 1
                if got != (int(P[q, i]), int(S[q, i])):
                    bad.append((q, nm, got, (int(P[q, i]), int(S[q, i]))))
    return {"dataset": ds, "questions": hi, "checks": checks, "mismatches": bad[:5], "n_mismatches": len(bad), "verdict": "PASS" if not bad else "FAIL"}


def gate_gx4(op, ds, n_q=60, n_ref=3):
    """With the PL4 options off expand4 == zpl3g.expand3 on n_q questions (every regime, every setting, cs and vr in every combination);
    every PL4 walk and every PL3 walk against the per-node reference on n_ref questions."""
    src = P2.load_source(ds, "dev", "frozen")
    M3A = op.m3a
    nq = min(n_q, src.nq)
    nbad = 0
    for q in range(nq):
        d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
        seeds5 = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
        qv = src.qvec(q, q + 1)[0]
        scorer = Z3.Scorer(src.dvec, {"q": np.asarray(qv, dtype=np.float64)}, src.n)
        for rg, (h1, h2) in REG3.items():
            for stg in P2.SETTINGS:
                for cs in (False, True):
                    for vr in (False, True):
                        if vr and P2.SETTINGS[stg]["hops"] == 1:
                            continue
                        mods = tuple(m for m, on in (("cs", cs), ("vr", vr)) if on)
                        a_ = expand4(seeds5, [src.fams[f] for f in h1], [src.fams[f] for f in h2], P2.SETTINGS[stg], mods, scorer)
                        b_ = Z3.expand3(seeds5, [src.fams[f] for f in h1], [src.fams[f] for f in h2], P2.SETTINGS[stg], cs, vr, scorer)
                        if not np.array_equal(a_, b_):
                            nbad += 1
        scorer.release()
    bad_ref, n_checked = 0, 0
    for q in range(min(n_ref, src.nq)):
        d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
        seeds5 = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
        f1 = M3A.rrf_fused_rows(d[None, :], s[None, :], P2.RRF_CONST)[0]
        qv = src.qvec(q, q + 1)[0]
        scorer = Z3.Scorer(src.dvec, {"q": np.asarray(qv, dtype=np.float64)}, src.n)
        for w in Z3.walks_all() + walks4():
            regime, stg, mods = parse_walk4(w)
            sd = np.asarray(f1[:10] if "s10" in mods else (f1[:20] if "s20" in mods else seeds5), dtype=np.int64)
            h1, h2 = REG3[regime]
            a_ = expand4(sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], P2.SETTINGS[stg], mods, scorer)
            b_ = ref_walk4(sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], P2.SETTINGS[stg], mods, qv, src.dvec)
            n_checked += 1
            if not np.array_equal(a_, b_):
                bad_ref += 1
        scorer.release()
    return {"dataset": ds, "off_mode_vs_expand3_mismatches": nbad, "per_node_reference_checks": n_checked, "per_node_reference_mismatches": bad_ref,
            "verdict": "PASS" if nbad == 0 and bad_ref == 0 else "FAIL"}


def gate_gf4():
    bad = []
    fns = (parse_walk4, parse4, Plan4, capped_parts, order_group, order_fresh, expand4, pool_state4, iter_states4)
    for fn in fns:
        if any("gold" in p.lower() or p in ("g", "labels") for p in inspect.signature(fn).parameters):
            bad.append(f"{fn.__name__}: a label parameter")
        if "gold" in inspect.getsource(fn).lower():
            bad.append(f"{fn.__name__}: the source names a label")
    return {"checked": [f.__qualname__ for f in fns], "problems": bad, "verdict": "PASS" if not bad else "FAIL"}


def cmd_gd(a):
    """Internal: print the hash of the states of a sample (run under different thread counts by selftest)."""
    op = P2.opened()
    src = P2.load_source(a.ds, "dev", "frozen")
    src.golds = None                                           # the states are built with the labels withheld
    plan = Plan4(cell_list4())
    h = hashlib.sha256()
    for q, st in iter_states4(op, src, plan, 0, min(a.n, src.nq), chunk=a.chunk):
        h.update(Z3.state_hash3(plan, st).encode())
    print("GDHASH " + h.hexdigest(), flush=True)
    return 0


def gate_gd4(ds, n=24):
    out = {}
    for th, chunk in ((1, 64), (4, 7)):
        env = dict(os.environ, OPENBLAS_NUM_THREADS=str(th), OMP_NUM_THREADS=str(th), MKL_NUM_THREADS=str(th), NUMBA_NUM_THREADS=str(th))
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "gd", "--ds", ds, "--n", str(n), "--chunk", str(chunk)],
                           capture_output=True, text=True, env=env, cwd=str(ROOT))
        line = [x for x in r.stdout.splitlines() if x.startswith("GDHASH")]
        if r.returncode or not line:
            return {"verdict": "FAIL", "why": (r.stderr or r.stdout)[-400:]}
        out[f"threads{th}_chunk{chunk}"] = line[0].split()[1]
    return {"hashes": out, "verdict": "PASS" if len(set(out.values())) == 1 else "FAIL"}


def cmd_selftest(a):
    t0 = time.time()
    op = P2.opened()
    res = {"stage": "PL4 selftest", "declared_in": DOC, "utc": utc(), "script_sha256": script_sha256(), "code_path_sha256": code_path_sha256()}
    names = cell_list4()
    res["cells"] = {"n": len(names), "distinct": len(set(names)), "sha256": P2.cells_sha256(names), "expected": EXPECT4_SHA256,
                    "union_distinct": len(set(cell_list_union4())),
                    "verdict": "PASS" if len(names) == EXPECT4_CELLS == len(set(names)) and P2.cells_sha256(names) == EXPECT4_SHA256
                    and len(set(cell_list_union4())) == len(P2.cell_list()) + len(Z3.cell_list3()) + len(names) else "FAIL"}
    res["G-F"] = gate_gf4()
    res["engine"], res["G-X4"] = [], []
    for ds in a.ds:
        res["engine"].append(selftest_engine4(op, ds, a.n, a.per_walk))
        res["G-X4"].append(gate_gx4(op, ds))
        log(f"{ds}: engine {res['engine'][-1]['verdict']} ({res['engine'][-1]['checks']} checks), G-X4 {res['G-X4'][-1]['verdict']}")
    res["G-D"] = gate_gd4(a.ds[0])
    res["verdict"] = "PASS" if all(x["verdict"] == "PASS" for x in [res["cells"], res["G-F"], res["G-D"]] + res["engine"] + res["G-X4"]) else "FAIL"
    res["seconds"] = round(time.time() - t0, 1)
    wjson(PL4 / "selftest.json", res)
    log(f"selftest: {res['verdict']} (cells {res['cells']['verdict']}, G-F {res['G-F']['verdict']}, G-D {res['G-D']['verdict']}) in {res['seconds']} s")
    return 0 if res["verdict"] == "PASS" else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cells")
    p.add_argument("--write", action="store_true")
    p = sub.add_parser("time")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--walks", default="")
    p.add_argument("--top", type=int, default=12)
    p = sub.add_parser("gr")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--n", type=int, default=400)
    p = sub.add_parser("grid")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--carve", required=True, choices=("dev", "eval", "filed"))
    p.add_argument("--first", default="frozen", choices=("frozen", "e3v"))
    p.add_argument("--part", nargs=2, metavar=("I", "K"))
    p.add_argument("--cells", default="new")
    p.add_argument("--tag", default="")
    p = sub.add_parser("select")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p = sub.add_parser("confirm")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--first", nargs="+", default=["frozen", "e3v"], choices=("frozen", "e3v"))
    sub.add_parser("call")
    p = sub.add_parser("selftest")
    p.add_argument("--ds", nargs="+", default=["multihoprag", "squad"], choices=SEVEN)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--per-walk", dest="per_walk", type=int, default=3)
    p = sub.add_parser("gd")
    p.add_argument("--ds", required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--chunk", type=int, default=64)
    for name in ("time", "gr", "grid", "select", "confirm", "selftest", "gd"):
        sub.choices[name].add_argument("--host", action="store_true", help="open the package as the lab host's mirror")
    a = ap.parse_args(argv)
    P2.HOST = bool(getattr(a, "host", False))
    P2.code_path_sha256()
    P2.script_sha256()
    Z3.code_path_sha256()
    code_path_sha256()
    script_sha256()
    return {"cells": cmd_cells, "time": cmd_time, "gr": cmd_gr, "grid": cmd_grid, "select": cmd_select, "confirm": cmd_confirm, "call": cmd_call,
            "selftest": cmd_selftest, "gd": cmd_gd}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
