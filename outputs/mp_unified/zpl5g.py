"""PL5 (docs/PL5_COVERAGE_ROUND4.md): round 4 of the coverage target. The caps follow the budget.

    python outputs/mp_unified/zpl5g.py cells [--write]                         # the declared cell list and its sha256
    python outputs/mp_unified/zpl5g.py time --ds D [--n 40] [--walks SUBSTR]   # cost of the walks, no labels read
    python outputs/mp_unified/zpl5g.py gr --ds D [--n 400] [--n4 150] [--host] # G-R: PL2's 804, PL3's 1,470 and PL4's 1,932 cells re-built by this engine == their dev arrays
    python outputs/mp_unified/zpl5g.py grid --ds D --carve dev --first frozen [--part I K] [--host]
    python outputs/mp_unified/zpl5g.py select --ds D [--host]                  # the rule on dev over PL2's, PL3's, PL4's and PL5's cells
    python outputs/mp_unified/zpl5g.py confirm --ds D [--first frozen e3v] [--host]
    python outputs/mp_unified/zpl5g.py call                                    # PL5's chosen cells against the earlier rounds' best, both readings, over the seven
    python outputs/mp_unified/zpl5g.py selftest [--ds D ...]                   # G-X5, G-F, G-D, and the engine against a per-node reference

This module imports zpl4g.py (PL4's engine), zpl3g.py (PL3's) and zpl2g.py (PL2's) and edits none of them. It adds the budget-following walk: the per-seed
and per-frontier caps are the scoring window (the first 512 entries of a row, or 4,096 on the `_w4096` walks, best by cosine first) and the visited budget is the walk's one size parameter;
every hop, the first included, keeps at most an equal share of what is left of it, by one order key (`vr` cosine, or `fu`, the reciprocal-rank fusion of
cosine, support and row rank), and the families may take turns (`rr`). It also adds the one-hop window walks. With every new walk off the engine gives PL4's,
PL3's and PL2's cells element for element (gate G-R). Every pool function takes and reads no labels; labels enter in zpl3g.grade3 only. Everything is
deterministic: integer walks, float64 inner products in numpy's pairwise order (no BLAS), ties to the lower id.
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
import zpl4g as Z4  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402

DOC = "docs/PL5_COVERAGE_ROUND4.md"
PL4 = ROOT / "outputs" / "pl4"
PL5 = ROOT / "outputs" / "pl5"
SEVEN, MHR = P2.SEVEN, P2.MHR
LAPTOP5 = Z3.LAPTOP5
FLOORS, BOUND = P2.FLOORS, P2.BOUND
SCORE_WINDOW = Z3.SCORE_WINDOW
REG3 = Z3.REG3
S2, S3, S3S = Z3.S2, Z3.S3, Z3.S3S
BASES5 = Z3.BASES3
M5 = Z3.M3
BUDGETS = Z4.BUDGETS                                         # the matched node counts of the frontier reading (PL4 section 6): inside the bound
ANY_BUDGETS = (3000, 4000, 5000, 6000, 7000)                 # read beside, above the bound: they decide nothing
RRF_C = P2.RRF_CONST                                         # `fu` fuses with the constant the first stage's fusion uses

# ── the declared cells (docs/PL5_COVERAGE_ROUND4.md section 5) ───────────────

WINDOW = SCORE_WINDOW                                        # the cap of a budget-following walk: the whole scoring window (PL2-PL4's, 512)
WIDE = 4096                                                  # the wide scoring window of the `_w4096` walks: a node's first 4096 neighbours are scored
BF_V = (1000, 2000, 5000)                                    # the walk's visited budgets
WIDE_V = (2000, 5000)                                        # the budgets the wide window is declared for
SET5 = dict(P2.SETTINGS)
for _h in (2, 3):
    for _v in BF_V:
        SET5[f"h{_h}_bf_v{_v}"] = {"name": f"h{_h}_bf_v{_v}", "hops": _h, "per_seed_cap": WINDOW, "per_frontier_cap": WINDOW, "visited_cap": _v, "bf": True,
                                   "window": WINDOW}
    for _v in WIDE_V:
        SET5[f"h{_h}_bf_v{_v}_w{WIDE}"] = {"name": f"h{_h}_bf_v{_v}_w{WIDE}", "hops": _h, "per_seed_cap": WIDE, "per_frontier_cap": WIDE, "visited_cap": _v,
                                           "bf": True, "window": WIDE}
SET5["h1_cW"] = {"name": "h1_cW", "hops": 1, "per_seed_cap": WINDOW}
MODS5 = ("s10", "s20", "cs", "vr", "sp", "rk", "fu", "rr", "eq")
KEYS5 = ("vr", "sp", "rk", "fu")
TAGS5 = ("bf", "vr", "fu", "rr", "seeds", "u1", "wide")
walk_name = Z3.walk_name


def bf_name(h, v, wide=False):
    return f"h{h}_bf_v{v}" + (f"_w{WIDE}" if wide else "")


def walks5():
    w = []

    def add(reg, st, mods):
        w.append(walk_name(reg, st, mods))
    for reg in ("STRUCT", "FULL"):                                         # the budget-following walk, two order keys
        for h in (2, 3):
            for v in BF_V:
                for key in ("vr", "fu"):
                    add(reg, bf_name(h, v), ("cs", key))
    for h in (2, 3):                                                       # the families take turns
        for v in BF_V:
            add("FULL", bf_name(h, v), ("cs", "vr", "rr"))
    for h in (2, 3):                                                       # seeds from the fused list
        for v in (2000, 5000):
            add("FULL", bf_name(h, v), ("s10", "cs", "vr"))
    for reg in ("STRUCT", "FULL"):                                         # the budget-following walk with the wide scoring window
        for h in (2, 3):
            for v in WIDE_V:
                add(reg, bf_name(h, v, True), ("cs", "vr"))
    for reg in ("STRUCT", "FULL"):                                         # one hop, the whole scoring window
        add(reg, "h1_cW", ("cs",))
        add(reg, "h1_cW", ("s10", "cs"))
    return w


P2_SET = set(P2.cell_list())
Z3_SET = set(Z3.cell_list3())
P4_SET = set(Z4.cell_list4())


def sizes_of(setting):
    """The ranked cells of a walk cut it to M nodes: a budget-following walk has at most its visited budget, so M above that repeats the union cell."""
    st = SET5[setting]
    return [m for m in M5 if m <= st["visited_cap"]] if st.get("bf") else list(M5)


def cell_list5():
    """PL5's own cells: every walk x (6 bases as unions, 6 bases x its sizes as ranked cells), minus PL2's, PL3's and PL4's."""
    seen = P2_SET | Z3_SET | P4_SET
    out = []

    def add(n):
        if n not in seen:
            seen.add(n)
            out.append(n)
    for w in walks5():
        sizes = sizes_of(w.split(":")[1].partition("~")[0])
        for b in BASES5:
            add(f"{b}+{w}")
        for b in BASES5:
            for m in sizes:
                add(f"{b}+{w}+top{m}")
    return out


def cell_list_union5():
    return P2.cell_list() + Z3.cell_list3() + Z4.cell_list4() + cell_list5()


EXPECT5_CELLS = 1872                                            # the declared cell list (docs section 5), fixed at the declaration commit
EXPECT5_SHA256 = "a5a759803992cc0524a4ebd4de21d21396ea599dd9f0597329be9842ab9baaeb"


def constants5():
    return {"settings": {k: SET5[k] for k in sorted(SET5) if k in (Z3.S1, S2, S3, S3S) or SET5[k].get("bf") or k == "h1_cW"},
            "regimes": {k: [list(v[0]), list(v[1])] for k, v in REG3.items()}, "mods_order": list(MODS5), "keys": list(KEYS5), "bases": list(BASES5), "m": list(M5),
            "score_window": SCORE_WINDOW, "walks": walks5(), "budgets": list(BUDGETS), "any_budgets": list(ANY_BUDGETS), "floors": list(FLOORS), "bound": BOUND,
            "boot": [P2.BOOT_N, P2.BOOT_SEED], "rrf_c": RRF_C, "bf_v": list(BF_V), "wide": WIDE, "wide_v": list(WIDE_V),
            "rank": "float64 inner product, ties to the lower node id; fu = sum of 1/(RRF_C + place + 1) over the places under vr, sp, rk"}


def require_cells5():
    names = cell_list5()
    if len(names) != EXPECT5_CELLS or len(set(names)) != EXPECT5_CELLS:
        P2.stop(f"the PL5 cell list has {len(names)} cells ({len(set(names))} distinct), expected {EXPECT5_CELLS}")
    if P2.cells_sha256(names) != EXPECT5_SHA256:
        P2.stop("the PL5 cell list is not the declared one (cells_sha256 differs from EXPECT5_SHA256)")


# ── cell names -> a plan ─────────────────────────────────────────────────────

def parse_walk5(w):
    regime, rest = w.split(":")
    setting, _, ms = rest.partition("~")
    mods = tuple(ms.split("-")) if ms else ()
    if regime not in REG3 or setting not in SET5:
        raise ValueError(f"unknown walk {w}")
    if any(m not in MODS5 for m in mods) or list(mods) != sorted(mods, key=MODS5.index) or ("s10" in mods and "s20" in mods):
        raise ValueError(f"unknown walk {w}")
    st = SET5[setting]
    if st.get("bf"):
        if "cs" not in mods or sum(m in KEYS5 for m in mods) != 1 or "eq" in mods or any(m in ("sp", "rk") for m in mods):
            raise ValueError(f"a budget-following walk needs cs and exactly one of vr, fu, and takes no eq, sp or rk: {w}")
        if "rr" in mods and regime != "FULL":
            raise ValueError(f"rr needs two or more families at hops 2 and 3 (regime FULL): {w}")
        return (regime, setting, mods)
    if setting == "h1_cW":
        if "cs" not in mods or any(m not in ("s10", "s20", "cs") for m in mods):
            raise ValueError(f"the one-hop window walk takes cs and a seed option only: {w}")
        return (regime, setting, mods)
    if "fu" in mods:
        raise ValueError(f"fu is read by the budget-following walks only: {w}")
    return Z4.parse_walk4(w)


def parse5(name):
    """('R', list, b) | ('FU',) | ('G', list, b, walk) | ('K', list, b, walk, M, rv); a walk = (regime, setting, mods); rv = 'q' or 'b'."""
    if "+" not in name:
        return P2.parse(name)
    parts = name.split("+")
    ln, b = P2.base_of(parts[0])
    wk = parse_walk5(parts[1])
    if len(parts) == 3:
        if parts[2].startswith("top"):
            return ("K", ln, b, wk, int(parts[2][3:]), "q")
        if parts[2].startswith("brg"):
            return ("K", ln, b, wk, int(parts[2][3:]), "b")
        raise ValueError(f"unknown cell {name}")
    if len(parts) != 2:
        raise ValueError(f"unknown cell {name}")
    return ("G", ln, b, wk)


def family5(name):
    return "pl2" if name in P2_SET else ("pl3" if name in Z3_SET else ("pl4" if name in P4_SET else "pl5"))


def tags_of5(name):
    """Which PL5 option a cell belongs to (descriptive, for what each option bought). Earlier rounds' cells are tagged pl2 / pl3 / pl4."""
    f = family5(name)
    if f != "pl5":
        return (f,)
    regime, setting, mods = parse5(name)[3]
    tags = []
    if SET5[setting].get("bf"):
        tags.append("bf")
        tags += [k for k in ("vr", "fu") if k in mods]
        if "rr" in mods:
            tags.append("rr")
        if SET5[setting]["window"] > WINDOW:
            tags.append("wide")
    else:
        tags.append("u1")
    if "s10" in mods or "s20" in mods:
        tags.append("seeds")
    return tuple(tags)


class Plan5:
    def __init__(self, names):
        self.names = list(names)
        if len(set(self.names)) != len(self.names):
            P2.stop("a cell name repeats")
        self.C = len(self.names)
        ret, fu, walks = {}, [], {}
        for i, nm in enumerate(self.names):
            p = parse5(nm)
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
        self.need_vec = any(v["K"] for v in self.walks.values()) or any(any(m in w[2] for m in ("cs", "vr", "sp", "rk", "fu")) for w in self.walks)
        self.need_f1 = "F1" in lists or any(("s10" in w[2] or "s20" in w[2]) for w in self.walks) or self.need_bridge
        self.families = sorted({f for w in self.walks for part in REG3[w[0]] for f in part})


# ── the pool code (no labels anywhere in this block) ─────────────────────────

def order_group5(cols, grp, rank, key, scorer, n_par):
    """The distinct nodes of one candidate list in the cell's order. vr, sp, rk are PL4's (zpl4g.order_group). `fu` is their reciprocal-rank fusion: a node's
    score is the sum over vr, sp, rk, in that order, of 1 / (RRF_C + place + 1), place its 0-based place in that key's order of the same list; the nodes are
    ordered by -score and then by node id."""
    if key != "fu":
        return Z4.order_group(cols, grp, rank, key, scorer, n_par)
    u = np.unique(cols)
    score = np.zeros(u.size, dtype=np.float64)
    for k_ in ("vr", "sp", "rk"):
        o = Z4.order_group(cols, grp, rank, k_, scorer, n_par)
        pos = np.empty(u.size, dtype=np.int64)
        pos[np.searchsorted(u, o)] = np.arange(o.size, dtype=np.int64)
        score += 1.0 / (RRF_C + pos + 1.0)
    return u[np.lexsort((u, -score))]


def order_fresh5(cols, grp, fam, rank, key, rr, scorer, n_par):
    """The fresh candidates of a hop in the cell's order. rr: every family's list is ordered by the key on its own, and the lists are merged
    by place (every family's first, then every family's second, ...), the first place a node takes being kept."""
    if cols.size == 0:
        return cols
    if not rr:
        return order_group5(cols, grp, rank, key, scorer, n_par)
    nodes, pos, fm = [], [], []
    for f in np.unique(fam):
        m = fam == f
        o = order_group5(cols[m], grp[m], rank[m], key, scorer, n_par)
        nodes.append(o)
        pos.append(np.arange(o.size, dtype=np.int64))
        fm.append(np.full(o.size, int(f), dtype=np.int64))
    nodes, pos, fm = np.concatenate(nodes), np.concatenate(pos), np.concatenate(fm)
    nodes = nodes[np.lexsort((fm, pos))]
    return m3b_pools._first_occurrence_unique(nodes)


def capped_parts5(nodes, families, win, scorer):
    """zpl4g.capped_parts with cs on and the scoring window `win` (PL2-PL4's is SCORE_WINDOW = 512): the kept neighbours of every node in every family,
    the best `win` by cosine out of each row's first `win` entries (ties to the lower id), node-major, then family, then best first, with the index of
    the node that names each, its family and its place in that node's list: (cols, parent index, family index, rank)."""
    e64, e8 = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)
    if nodes.size == 0:
        return e64, e64, e8, e64
    cols, grps, fams_, ranks = [], [], [], []
    for f_i, fam in enumerate(families):
        starts = fam.indptr[nodes]
        lens = np.minimum(fam.indptr[nodes + 1] - starts, win)
        total = int(lens.sum())
        if total == 0:
            continue
        off = np.cumsum(lens) - lens
        within = np.arange(total, dtype=np.int64) - np.repeat(off, lens)
        c = fam.col[np.repeat(starts, lens) + within].astype(np.int64, copy=False)
        grp = np.repeat(np.arange(nodes.size, dtype=np.int64), lens)
        sc = scorer.get(c, "q")
        c = c[np.lexsort((c, -sc, grp))]
        cols.append(c)
        grps.append(grp)
        fams_.append(np.full(c.size, f_i, dtype=np.int8))
        ranks.append(within)
    if not cols:
        return e64, e64, e8, e64
    if len(cols) == 1:
        return cols[0], grps[0], fams_[0], ranks[0]
    rank, fam, grp = np.concatenate(ranks), np.concatenate(fams_), np.concatenate(grps)
    order = np.lexsort((rank, fam, grp))
    return np.concatenate(cols)[order], grp[order], fam[order], rank[order]


def expand5(seeds, fams1, fams2, setting, mods, scorer):
    """The PL5 walk. A setting that is not budget-following is zpl4g.expand4 (PL4's, PL3's and PL2's walks, and the one-hop window walks). A budget-following
    walk (setting['bf']) has no per-node cap but the scoring window: at every hop h = 1..H it lists, for every frontier node (the seeds at hop 1, the nodes the
    previous hop kept, in the order it kept them) and every family of the hop (fams1 at hop 1, fams2 after), the node's first `window` neighbours (512, or 4096 for a `_w4096` walk) ordered by
    cosine to the question, highest first (ties to the lower id); removes what is already visited (the seeds and every node kept before); orders the distinct
    nodes that remain by the walk's key (`vr` cosine, or `fu` the fusion of vr, sp and rk; with `rr` the families' lists are ordered on their own and merged by
    place); and keeps the first ceil(rem / (H - h + 1)) of them, rem = the visited budget minus the nodes visited so far (so what a hop cannot fill passes to the
    next, and the last hop keeps what is left)."""
    if not setting.get("bf"):
        return Z4.expand4(seeds, fams1, fams2, setting, mods, scorer)
    key = next(m for m in KEYS5 if m in mods)
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
        cols, grp, fam, rank = capped_parts5(frontier, fams1 if h == 1 else fams2, win, scorer)
        if cols.size == 0:
            break
        k = np.isin(cols, visited, invert=True)
        fresh = order_fresh5(cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))[:budget]
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


TIMES = None                                                 # a dict while `time` runs: seconds per walk


def pool_state5(plan, d, s, f2, f1, seeds5, fams, qv, dvec, n_nodes):
    """Everything a question's pools are made of: the ranked lists and each walk's expansion with its orders (zpl3g.pool_state3 with expand5).
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
        exp = expand5(sd, [fams[f] for f in h1], [fams[f] for f in h2], SET5[stg], mods, scorer)
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

def iter_states5(op, src, plan, lo, hi, chunk=64):
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
            yield c0 + j, pool_state5(plan, d[j], s[j], f2[j], f1[j], seeds[j], fams, qv[j], dvec, src.n)


def evaluate5(op, src, plan, lo, hi, every=300):
    Q = hi - lo
    P = np.zeros((Q, plan.C), dtype=np.int16)
    S = np.zeros((Q, plan.C), dtype=np.int32)
    t0 = last = time.time()
    for q, st in iter_states5(op, src, plan, lo, hi):
        Z3.grade3(plan, st, src.golds[q], P[q - lo], S[q - lo])
        if (q - lo + 1) % every == 0 and time.time() - last > 20:
            last = time.time()
            log(f"{src.name}/{src.carve}/{src.first}: {q - lo + 1}/{Q} questions, {last - t0:.0f} s ({P2.peak_gb()} GB)")
    return P, S


_SHA = {}


def code_path_sha256():
    """One hash of PL5's pool code, its constants and PL4's code-path hash (which holds PL3's, which holds PL2's) (G-U): the same on every dataset."""
    if "code" not in _SHA:
        h = hashlib.sha256()
        for fn in (parse_walk5, parse5, Plan5, capped_parts5, order_group5, order_fresh5, expand5, pool_state5, iter_states5, evaluate5, walks5, cell_list5, sizes_of):
            h.update(inspect.getsource(fn).replace("\r\n", "\n").encode("utf-8"))
        h.update(json.dumps(constants5(), sort_keys=True).encode("utf-8"))
        h.update(Z4.code_path_sha256().encode("utf-8"))
        _SHA["code"] = h.hexdigest()
    return _SHA["code"]


def script_sha256():
    if "script" not in _SHA:
        _SHA["script"] = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return _SHA["script"]


# ── small helpers ────────────────────────────────────────────────────────────

log, stop, rel, rjson, wjson, utc = P2.log, P2.stop, P2.rel, P2.rjson, P2.wjson, P2.utc
load_grid5, heads_of = Z3.load_grid3, Z3.heads_of
rows_of_grid5 = Z3.rows_of_grid3


def grid_path5(d, carve, first, tag=""):
    return PL5 / d / f"grid_{carve}_{first}{('_' + tag) if tag else ''}"


def save_grid5(base, src, plan, P, S, lo, hi, seconds, part=None, stage="PL5 grid"):
    p = Path(str(base) + (f".part{part[0]}of{part[1]}" if part else "") + ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez(tmp, names=np.asarray(plan.names, dtype=str), present=P, size=S, qids=np.asarray(src.qids[lo:hi], dtype=str),
             gold_counts=src.gold_counts[lo:hi].astype(np.int32), lo=np.asarray(lo), hi=np.asarray(hi))
    os.replace(tmp, p)
    head = {"stage": stage, "declared_in": DOC, "dataset": src.name, "carve": src.carve, "first": src.first, "questions": int(hi - lo),
            "rows": [int(lo), int(hi)], "part": list(part) if part else None, "cells": plan.C, "cells_sha256": P2.cells_sha256(plan.names),
            "graph": getattr(src, "graph_kind", None), "population_digest": src.pop_digest, "code_path_sha256": code_path_sha256(),
            "pl4_code_path_sha256": Z4.code_path_sha256(), "pl3_code_path_sha256": Z3.code_path_sha256(), "pl2_code_path_sha256": P2.code_path_sha256(),
            "script_sha256": script_sha256(),
            "constants_sha256": hashlib.sha256(json.dumps(constants5(), sort_keys=True).encode()).hexdigest(), "npz": rel(p),
            "npz_sha256": P2.sha256_file(p), "seconds": round(seconds, 1), "peak_gb": P2.peak_gb(), "host": P2.HOST, "utc": utc()}
    wjson(Path(str(p)[:-4] + ".json"), head)
    return p, head


# ── commands: cells, time ────────────────────────────────────────────────────

def cmd_cells(a):
    new, uni = cell_list5(), cell_list_union5()
    fam = {}
    for n in new:
        fam[tags_of5(n)] = fam.get(tags_of5(n), 0) + 1
    log(f"PL5 cells {len(new)} ({len(set(new))} distinct), sha256 {P2.cells_sha256(new)}; union with PL2's, PL3's and PL4's {len(uni)} ({len(set(uni))} distinct), "
        f"sha256 {P2.cells_sha256(uni)}")
    log(f"walks {len(walks5())}; tags {fam}")
    log(f"code path sha256 {code_path_sha256()}")
    if a.write:
        wjson(PL5 / "cells.json", {"cells": new, "sha256": P2.cells_sha256(new), "union_sha256": P2.cells_sha256(uni), "n": len(new)})
    return 0


def cmd_time(a):
    global TIMES
    op = P2.opened()
    names = cell_list5()
    if a.walks:
        names = [n for n in names if a.walks in n]
    plan = Plan5(names)
    src = P2.load_source(a.ds, "dev", "frozen")
    n = min(a.n, src.nq)
    TIMES = {}
    t0 = time.time()
    for q, st in iter_states5(op, src, plan, 0, n):
        pass
    tot = time.time() - t0
    sc = TIMES.pop("_scored", 0)
    log(f"{a.ds}: {n} questions, {plan.C} cells, {len(plan.walks)} walks, {tot:.1f} s = {tot / n:.3f} s/question; nodes scored {sc / n:.0f}/question; peak {P2.peak_gb()} GB")
    for w, t in sorted(TIMES.items(), key=lambda kv: -kv[1])[:int(a.top)]:
        log(f"   {walk_name(*w):44s} {1000 * t / n:8.1f} ms/question")
    return 0


# ── G-R: PL2's and PL3's cells, re-built by this engine ──────────────────────

def cmd_gr(a):
    """G-R: PL2's and PL3's cells on the first a.n dev questions and PL4's on the first a.n4, built by THIS engine, equal the filed dev arrays element for element."""
    t0 = time.time()
    op = P2.opened()
    names_a = cell_list_union5()[:len(P2.cell_list()) + len(Z3.cell_list3())]
    names_b = Z4.cell_list4()
    src = P2.load_source(a.ds, "dev", "frozen")
    r2 = P2.load_grid(a.ds, "dev", "frozen")
    r3 = load_grid5(Z3.grid_path3(a.ds, "dev", "frozen"))
    r4 = load_grid5(Z4.grid_path4(a.ds, "dev", "frozen"))
    n, n4 = min(a.n, src.nq), min(a.n4, src.nq)
    if (r2["names"] + r3["names"] != names_a or r4["names"] != names_b
            or any(r["qids"][:n] != list(src.qids[:n]) for r in (r2, r3, r4))):
        stop(f"{a.ds}: PL2's, PL3's and PL4's dev grids do not name the cells or questions this engine expects")
    P, S = evaluate5(op, src, Plan5(names_a), 0, n)
    ref_p = np.concatenate([r2["present"][:n], r3["present"][:n]], axis=1)
    ref_s = np.concatenate([r2["size"][:n], r3["size"][:n]], axis=1)
    bad_p, bad_s = np.argwhere(P != ref_p), np.argwhere(S != ref_s)
    P4_, S4_ = evaluate5(op, src, Plan5(names_b), 0, n4)
    bad_p4, bad_s4 = np.argwhere(P4_ != r4["present"][:n4]), np.argwhere(S4_ != r4["size"][:n4])
    first = None
    for nm, bp, bs in ((names_a, bad_p, bad_s), (names_b, bad_p4, bad_s4)):
        if first is None and (len(bp) or len(bs)):
            first = [int((bp if len(bp) else bs)[0][0]), nm[int((bp if len(bp) else bs)[0][1])]]
    ok = not (len(bad_p) or len(bad_s) or len(bad_p4) or len(bad_s4))
    out = {"stage": "PL5 G-R", "declared_in": DOC, "dataset": a.ds, "host": P2.HOST, "questions": int(n), "questions_pl4": int(n4),
           "cells": len(names_a) + len(names_b), "pl2_cells": len(P2.cell_list()), "pl3_cells": len(Z3.cell_list3()), "pl4_cells": len(names_b),
           "elements": int(P.size + P4_.size), "present_mismatches": int(len(bad_p) + len(bad_p4)), "size_mismatches": int(len(bad_s) + len(bad_s4)),
           "first_mismatch": first, "pl4_code_path_sha256": Z4.code_path_sha256(), "pl3_code_path_sha256": Z3.code_path_sha256(),
           "pl2_code_path_sha256": P2.code_path_sha256(), "code_path_sha256": code_path_sha256(), "script_sha256": script_sha256(),
           "verdict": "PASS" if ok else "FAIL", "seconds": round(time.time() - t0, 1), "peak_gb": P2.peak_gb(), "utc": utc()}
    wjson(PL5 / f"gr_{a.ds}{'_host' if P2.HOST else ''}.json", out)
    log(f"G-R {a.ds}: {out['verdict']}, {n} questions x {len(names_a)} cells + {n4} x {len(names_b)} = {out['elements']} elements, mismatches "
        f"{out['present_mismatches']}/{out['size_mismatches']} ({out['seconds']} s)")
    return 0 if ok else 3


def gates_passed5(ds):
    ref = ds if (ds in LAPTOP5 and not P2.HOST) else "musique"
    p = PL5 / f"gr_{ref}.json"
    if not p.is_file() or rjson(p)["verdict"] != "PASS":
        stop(f"{p.name} is not PASS: run `zpl5g.py gr --ds {ref}` first")
    if P2.HOST:
        h = PL5 / "gr_musique_host.json"
        if not h.is_file() or rjson(h)["verdict"] != "PASS":
            stop("gr_musique_host.json is not PASS: run `zpl5g.py gr --ds musique --host` first")
    st = PL5 / "selftest.json"
    if not st.is_file() or rjson(st)["verdict"] != "PASS":
        stop("selftest.json is not PASS: run `zpl5g.py selftest` first")


def cmd_grid(a):
    t0 = time.time()
    P2.HOST = bool(a.host)
    gates_passed5(a.ds)
    names = cell_list5() if a.cells == "new" else [str(x) for x in rjson(a.cells)]
    if a.cells == "new":
        require_cells5()
    plan = Plan5(names)
    op = P2.opened()
    src = P2.load_source(a.ds, a.carve, a.first)
    lo, hi, part = 0, src.nq, None
    if a.part:
        i, k = int(a.part[0]), int(a.part[1])
        lo, hi = i * src.nq // k, (i + 1) * src.nq // k
        part = (i, k)
    log(f"{a.ds}/{a.carve}/{a.first}: {src.nq} questions, rows {lo}..{hi}, {plan.C} cells, {len(plan.walks)} walks, graph {src.graph_kind} ({P2.peak_gb()} GB)")
    P, S = evaluate5(op, src, plan, lo, hi)
    tag = a.tag or ("" if a.cells == "new" else "sub")
    p, head = save_grid5(grid_path5(a.ds, a.carve, a.first, tag), src, plan, P, S, lo, hi, time.time() - t0, part)
    log(f"{a.ds}/{a.carve}/{a.first}: filed {rel(p)} in {head['seconds']} s")
    return 0


# ── select ───────────────────────────────────────────────────────────────────

def union_dev5(ds):
    """PL2's, PL3's, PL4's and PL5's dev grids side by side (same questions, disjoint cells), their heads checked against the code that is running."""
    g2 = P2.load_grid(ds, "dev", "frozen")
    for hp in heads_of(P2.grid_path(ds, "dev", "frozen")):
        hd = rjson(hp)
        if hd["code_path_sha256"] != P2.code_path_sha256() or hd["cells_sha256"] != P2.EXPECT_CELLS_SHA256:
            stop(f"G-U: {hp.name} (PL2) was written by other pool code or another cell list than the one that is running")
    g3 = load_grid5(Z3.grid_path3(ds, "dev", "frozen"))
    for hp in heads_of(Z3.grid_path3(ds, "dev", "frozen")):
        hd = rjson(hp)
        if hd["code_path_sha256"] != Z3.code_path_sha256() or hd["cells_sha256"] != Z3.EXPECT3_SHA256:
            stop(f"G-U: {hp.name} (PL3) was written by other pool code or another cell list than the one that is running")
    g4 = load_grid5(Z4.grid_path4(ds, "dev", "frozen"))
    for hp in heads_of(Z4.grid_path4(ds, "dev", "frozen")):
        hd = rjson(hp)
        if hd["code_path_sha256"] != Z4.code_path_sha256() or hd["cells_sha256"] != Z4.EXPECT4_SHA256:
            stop(f"G-U: {hp.name} (PL4) was written by other pool code or another cell list than the one that is running")
    g5 = load_grid5(grid_path5(ds, "dev", "frozen"))
    h5 = heads_of(grid_path5(ds, "dev", "frozen"))
    if not h5:
        stop(f"{ds}: no PL5 grid heads")
    for hp in h5:
        hd = rjson(hp)
        if hd["code_path_sha256"] != code_path_sha256() or hd["cells_sha256"] != EXPECT5_SHA256:
            stop(f"G-U: {hp.name} was written by other PL5 code or another cell list than the one that is running")
    gs = (g2, g3, g4, g5)
    if any(g["qids"] != g2["qids"] for g in gs) or not np.array_equal(g2["gold_counts"], g5["gold_counts"]):
        stop(f"{ds}: the four dev grids are not on the same questions")
    if len(set().union(*[set(g["names"]) for g in gs])) != sum(len(g["names"]) for g in gs):
        stop("the cells of the four grids overlap")
    if P2.cells_sha256(g5["names"]) != EXPECT5_SHA256:
        stop("the PL5 dev grid is not the declared cell list")
    return {"names": sum((g["names"] for g in gs), []), "present": np.concatenate([g["present"] for g in gs], axis=1),
            "size": np.concatenate([g["size"] for g in gs], axis=1), "qids": g2["qids"], "gold_counts": g2["gold_counts"]}, h5


READINGS = ("all", "prev", "new")


def frontier_pick(rows, n):
    """The matched-node-count pick: the highest ALL among cells with at most n mean nodes (ties to fewer nodes, then the name)."""
    within = [r for r in rows if r["nodes"] <= n]
    if not within:
        return None
    return min(within, key=lambda r: (-r["all"], r["nodes"], r["name"]))


def pick_any(rows, floor):
    """R95 / R98 at any size (no bound): the smallest mean nodes with ALL >= floor; if none, the highest ALL (ties to fewer nodes)."""
    reach = [r for r in rows if r["all"] >= floor]
    if reach:
        return min(reach, key=lambda r: (r["nodes"], -r["all"], r["name"])), True
    return min(rows, key=lambda r: (-r["all"], r["nodes"], r["name"])), False


def cmd_select(a):
    P2.HOST = bool(a.host)
    gates_passed5(a.ds)
    g, heads = union_dev5(a.ds)
    names = g["names"]
    rows = rows_of_grid5(g, a.ds, "dev")
    table = [{"name": n, "family": family5(n), "tags": list(tags_of5(n)), "nodes": r["candidates_mean"], "all": r["all_gold_at_pool"], "any": r["any_gold_at_pool"],
              "coverage": r["coverage_micro"], "p95": r["candidates_p95"], "max": r["candidates_max"]} for n, r in zip(names, rows)]
    readings = {"all": table, "prev": [t for t in table if t["family"] != "pl5"], "pl3": [t for t in table if t["family"] in ("pl2", "pl3")],
                "new": [t for t in table if t["family"] == "pl5"]}
    for tg in TAGS5:
        readings[f"only:{tg}"] = [t for t in table if t["family"] == "pl5" and tg in t["tags"]]
    out = {"stage": "PL5 select", "declared_in": DOC, "dataset": a.ds, "carve": "dev", "first": "frozen", "questions": len(g["qids"]), "bound": BOUND,
           "cells_sha256_pl5": EXPECT5_SHA256, "cells_sha256_pl4": Z4.EXPECT4_SHA256, "cells_sha256_pl3": Z3.EXPECT3_SHA256, "cells_sha256_pl2": P2.EXPECT_CELLS_SHA256,
           "code_path_sha256": code_path_sha256(), "pl4_code_path_sha256": Z4.code_path_sha256(), "pl3_code_path_sha256": Z3.code_path_sha256(),
           "pl2_code_path_sha256": P2.code_path_sha256(), "script_sha256": script_sha256(), "utc": utc(),
           "grid_heads": [{"file": rel(hp), "code_path_sha256": rjson(hp)["code_path_sha256"], "npz_sha256": rjson(hp)["npz_sha256"],
                           "seconds": rjson(hp)["seconds"], "peak_gb": rjson(hp)["peak_gb"]} for hp in heads]}
    out["picks"] = {}
    for floor in FLOORS:
        pc = int(round(floor * 100))
        for rd, rs in readings.items():
            if not rs:
                continue
            c, reached = P2.pick(rs, floor)
            out["picks"][f"R{pc}_{rd}"] = {**c, "reached": reached, "floor": floor}
        for rd in ("all", "prev", "new"):                                  # at any size: the price of the floor, read beside (the bound does not apply)
            c, reached = pick_any(readings[rd], floor)
            out["picks"][f"R{pc}any_{rd}"] = {**c, "reached": reached, "floor": floor}
    out["short"] = {f"{int(round(f * 100))}": not out["picks"][f"R{int(round(f * 100))}_all"]["reached"] for f in FLOORS}
    out["frontier_picks"] = {}
    for rd in ("all", "prev"):
        out["frontier_picks"][rd] = {str(n): frontier_pick(readings[rd], n) for n in BUDGETS + ANY_BUDGETS}
    out["frontier"] = {rd: P2.frontier(readings[rd]) for rd in ("all", "prev", "new", "pl3")}
    grid_n = (25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500) + ANY_BUDGETS
    out["bought"] = [{"nodes": n, "prev": P2.front_at(out["frontier"]["prev"], n), "new": P2.front_at(out["frontier"]["new"], n), "all": P2.front_at(out["frontier"]["all"], n),
                      "gain": (None if P2.front_at(out["frontier"]["all"], n) is None or P2.front_at(out["frontier"]["prev"], n) is None
                               else P2.front_at(out["frontier"]["all"], n) - P2.front_at(out["frontier"]["prev"], n))} for n in grid_n]
    pl4_sel = PL4 / a.ds / "select.json"
    same = None
    if pl4_sel.is_file():
        f4 = rjson(pl4_sel)["picks"]
        same = f4["R95_all"]["name"] == out["picks"]["R95_prev"]["name"] and f4["R98_all"]["name"] == out["picks"]["R98_prev"]["name"]
    out["prev_pick_equals_pl4_filed"] = same
    out["old"] = P2.old_cell(None, a.ds)
    out["table"] = table
    wjson(PL5 / a.ds / "select.json", out)
    for k in ("R95_all", "R95_prev", "R95_new", "R98_all", "R98_prev", "R98_new", "R95any_all", "R95any_prev", "R98any_all", "R98any_prev"):
        c = out["picks"][k]
        log(f"{a.ds} {k}: {c['name']}  ALL {c['all']:.4f} at {c['nodes']:.1f} nodes{'' if c['reached'] else '  (SHORT)'}")
    log(f"{a.ds}: PL4's pick re-read (reading `prev`) equals its filed select.json: {same}")
    return 0


# ── confirm and call ─────────────────────────────────────────────────────────


def confirm_names5(sel):
    names = []

    def add(n):
        if n and n not in names:
            names.append(n)
    for k in ("R95_all", "R98_all", "R95_prev", "R98_prev", "R95_new", "R98_new", "R95any_all", "R98any_all", "R95any_prev", "R98any_prev"):
        add(sel["picks"][k]["name"])
    for rd in ("all", "prev"):
        for n in BUDGETS + ANY_BUDGETS:
            c = sel["frontier_picks"][rd][str(n)]
            add(c["name"] if c else None)
    add(sel["old"])
    return names


def cmd_confirm(a):
    P2.HOST = bool(a.host)
    gates_passed5(a.ds)
    op = P2.opened()
    sel = rjson(PL5 / a.ds / "select.json")
    names = confirm_names5(sel)
    plan = Plan5(names)
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
        P, S = evaluate5(op, src, plan, 0, src.nq)
        save_grid5(grid_path5(a.ds, "eval", first, "confirm"), src, plan, P, S, 0, src.nq, time.time() - t0, stage="PL5 confirm grid")
        g = {"names": plan.names, "present": P, "size": S, "gold_counts": src.gold_counts}
        rows = {r["pool"]: r for r in rows_of_grid5(g, a.ds, "eval")}
        res = {"stage": "PL5 confirm", "declared_in": DOC, "dataset": a.ds, "first": first, "questions": src.nq, "utc": utc(), "cells": {}}
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
        wjson(PL5 / a.ds / f"confirm_{first}.json", res)
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
    out = {"stage": "PL5 call", "declared_in": DOC, "utc": utc(), "datasets": {}}
    codes = {}
    for k, ds in enumerate(SEVEN):
        rng = np.random.default_rng([P2.BOOT_SEED, k])
        sp, cp = PL5 / ds / "select.json", grid_path5(ds, "eval", "frozen", "confirm")
        if not sp.is_file() or not Path(str(cp) + ".npz").is_file():
            out["datasets"][ds] = {"status": "NOT_READ", "why": "no PL5 select or confirm file here"}
            continue
        sel = rjson(sp)
        codes[ds] = sel["code_path_sha256"]
        g = load_grid5(cp)
        at = {n: i for i, n in enumerate(g["names"])}
        gc_ = g["gold_counts"]

        def arrs(n):
            i = at[n]
            return (g["present"][:, i].astype(np.int64) == gc_), g["size"][:, i].astype(np.float64)
        rec = {}
        for tag, nk, ok in (("R95", "R95_all", "R95_prev"), ("R98", "R98_all", "R98_prev"), ("R95_new_only", "R95_new", "R95_prev")):
            new_name, old_name = sel["picks"][nk]["name"], sel["picks"][ok]["name"]
            new, old = arrs(new_name), arrs(old_name)
            pr = P2.paired(new, old, rng)
            floor = FLOORS[1] if tag == "R98" else FLOORS[0]
            nr, orr = bool(new[0].mean() >= floor), bool(old[0].mean() >= floor)
            rec[tag] = {"status": status_of(pr, nr, orr), "new": new_name, "old": old_name, "same_cell": new_name == old_name, "new_all": float(new[0].mean()),
                        "old_all": float(old[0].mean()), "new_nodes": float(new[1].mean()), "old_nodes": float(old[1].mean()), "new_reaches": nr, "old_reaches": orr,
                        "within_bound": bool(sel["picks"][nk]["nodes"] <= BOUND), **pr}

        def budget_row(cn, co):
            new, old = arrs(cn["name"]), arrs(co["name"])
            pr = P2.paired(new, old, rng)
            pst = "BELOW" if pr["all_diff_ci"][1] < 0 else ("ABOVE" if pr["all_diff_ci"][0] > 0 else "LEVEL")
            return {"status": pst, "new": cn["name"], "old": co["name"], "same_cell": cn["name"] == co["name"], "new_dev_all": cn["all"], "old_dev_all": co["all"],
                    "new_dev_nodes": cn["nodes"], "old_dev_nodes": co["nodes"], "new_all": float(new[0].mean()), "old_all": float(old[0].mean()),
                    "new_nodes": float(new[1].mean()), "old_nodes": float(old[1].mean()), **pr}
        fr = {}
        for n in BUDGETS:
            cn, co = sel["frontier_picks"]["all"][str(n)], sel["frontier_picks"]["prev"][str(n)]
            fr[str(n)] = budget_row(cn, co) if (cn is not None and co is not None) else {"status": "NONE", "why": "no cell within this many dev nodes"}
        price = {}                                                         # beside the calls: the price of the floor at any size (the bound does not apply)
        for tag, floor in (("R95any", FLOORS[0]), ("R98any", FLOORS[1])):
            new_name, old_name = sel["picks"][f"{tag}_all"]["name"], sel["picks"][f"{tag}_prev"]["name"]
            new, old = arrs(new_name), arrs(old_name)
            pr = P2.paired(new, old, rng)
            nr, orr = bool(new[0].mean() >= floor), bool(old[0].mean() >= floor)
            price[tag] = {"status": status_of(pr, nr, orr), "new": new_name, "old": old_name, "same_cell": new_name == old_name, "new_all": float(new[0].mean()),
                          "old_all": float(old[0].mean()), "new_nodes": float(new[1].mean()), "old_nodes": float(old[1].mean()), "new_reaches": nr, "old_reaches": orr,
                          "new_dev_reaches": bool(sel["picks"][f"{tag}_all"]["reached"]), "old_dev_reaches": bool(sel["picks"][f"{tag}_prev"]["reached"]), **pr}
        fr_any = {}
        for n in ANY_BUDGETS:
            cn, co = sel["frontier_picks"]["all"][str(n)], sel["frontier_picks"]["prev"][str(n)]
            fr_any[str(n)] = budget_row(cn, co) if (cn is not None and co is not None) else {"status": "NONE", "why": "no cell within this many dev nodes"}
        used = [v for v in fr.values() if v["status"] != "NONE"]
        used_any = [v for v in fr_any.values() if v["status"] != "NONE"]
        out["datasets"][ds] = {**rec["R95"], "status": rec["R95"]["status"], "r98": rec["R98"], "new_only": rec["R95_new_only"], "frontier": fr,
                               "frontier_status": frontier_status(used), "price": price, "frontier_any": fr_any, "frontier_any_status": frontier_status(used_any)}
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
    out["call"] = ("CARRY_PL5" if (complete and n_below == 0 and n_above >= 1 and inside) else ("INCOMPLETE" if not complete else "NOT_CARRIED"))
    out["call_frontier"] = ("CARRY_PL5_F" if (complete and n_below == 0 and f_below == 0 and f_above >= 1 and inside)
                            else ("INCOMPLETE" if not complete else "NOT_CARRIED"))
    out["counts"] = {"old": {"BELOW": n_below, "ABOVE": n_above, "LEVEL": len(read) - n_below - n_above},
                     "frontier": {"BELOW": f_below, "ABOVE": f_above, "LEVEL": len(read) - f_below - f_above},
                     "price_R95any": {s: sum(v["price"]["R95any"]["status"] == s for v in read.values()) for s in ("BELOW", "ABOVE", "LEVEL")},
                     "frontier_any": {s: sum(v["frontier_any_status"] == s for v in read.values()) for s in ("BELOW", "ABOVE", "LEVEL")}}
    wjson(PL5 / "call.json", out)
    for ds, v in out["datasets"].items():
        log(f"call {ds}: {v['status']} / frontier {v.get('frontier_status')}" + ("" if v["status"] == "NOT_READ" else
            f"  NEW {v['new']} ALL {v['new_all']:.4f} @ {v['new_nodes']:.0f}; PREV {v['old']} ALL {v['old_all']:.4f} @ {v['old_nodes']:.0f}"))
    log(f"call (old rule): {out['call']} (BELOW {n_below}, ABOVE {n_above}, read {len(read)}/7); call (frontier reading): {out['call_frontier']} "
        f"(BELOW {f_below}, ABOVE {f_above})")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def ref_walk5(seeds, fams1, fams2, setting, mods, qv, dvec):
    """The PL5 walk as a per-node loop, written from the declaration (docs section 4), for G-X5 and the engine test. A walk that is not budget-following is
    PL4's reference."""
    if not setting.get("bf"):
        return Z4.ref_walk4(seeds, fams1, fams2, setting, mods, qv, dvec)
    key = next(m for m in KEYS5 if m in mods)
    rr = "rr" in mods
    hops, limit, win = setting["hops"], setting["visited_cap"], setting["window"]
    memo = {}

    def one(nodes):
        nodes = [int(x) for x in nodes]
        miss = sorted(set(x for x in nodes if x not in memo))
        if miss:
            for x, v in zip(miss, P2.dot64(dvec(np.asarray(miss, dtype=np.int64)), qv)):
                memo[x] = float(v)
        return [memo[x] for x in nodes]

    def lists(v, fams):
        out = []
        for fi, f in enumerate(fams):
            row = [int(x) for x in f.col[f.indptr[v]:f.indptr[v + 1]]][:win]
            if row:
                sc = one(row)
                row = [row[i] for i in sorted(range(len(row)), key=lambda i: (-sc[i], row[i]))]
            out += [(x, fi, r) for r, x in enumerate(row[:win])]
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
        if not nodes:
            return nodes
        cos = dict(zip(nodes, one(nodes)))
        by_vr = sorted(nodes, key=lambda x: (-cos[x], x))
        if key == "vr":
            return by_vr
        parents, best = {}, {}
        for (y, p_, _f, r) in ents:
            parents.setdefault(y, set()).add(p_)
            if y not in best or r < best[y]:
                best[y] = r
        by_sp = sorted(nodes, key=lambda x: (-len(parents[x]), -cos[x], x))
        by_rk = sorted(nodes, key=lambda x: (best[x], -cos[x], x))
        score = {x: 0.0 for x in nodes}
        for lst in (by_vr, by_sp, by_rk):
            for pos, x in enumerate(lst):
                score[x] += 1.0 / (RRF_C + pos + 1.0)
        return sorted(nodes, key=lambda x: (-score[x], x))
    seeds = [int(x) for x in seeds]
    visited = set(seeds)
    frontier = list(seeds)
    out = []
    for h in range(1, hops + 1):
        rem = limit - len(visited)
        if rem <= 0 or not frontier:
            break
        budget = -(-rem // (hops - h + 1))
        fams = fams1 if h == 1 else fams2
        ents = [(x, pi, fi, r) for pi, v in enumerate(frontier) for (x, fi, r) in lists(int(v), fams) if x not in visited]
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


def brute_cell5(op, name, d, s, g, fams, seeds5, qv, dvec, exps):
    """One cell of one question by explicit sets (exps caches the per-node walks of this question), for the engine to be held to."""
    M3A = op.m3a
    p = parse5(name)
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
            exps[p[3]] = ref_walk5(sd, [fams[f] for f in h1], [fams[f] for f in h2], SET5[stg], mods, qv, dvec)
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


def selftest_engine5(op, ds, n_q, per_walk):
    """The engine against brute_cell5 on a sample of questions: every walk, a few of its cells."""
    src = P2.load_source(ds, "dev", "frozen")
    names = cell_list5()
    plan = Plan5(names)
    hi = min(n_q, src.nq)
    P, S = evaluate5(op, src, plan, 0, hi)
    M3A = op.m3a
    bad, checks = [], 0
    rng = np.random.default_rng(11)
    by_walk = {}
    for i, nm in enumerate(names):
        p = parse5(nm)
        by_walk.setdefault(p[3], []).append(i)
    for q in range(0, hi, max(1, hi // 4)):
        d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
        seeds5 = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
        qv = src.qvec(q, q + 1)[0]
        exps = {}
        for w, idx in by_walk.items():
            for i in rng.choice(idx, size=min(per_walk, len(idx)), replace=False):
                nm = names[int(i)]
                got = brute_cell5(op, nm, d, s, src.golds[q], src.fams, seeds5, qv, src.dvec, exps)
                checks += 1
                if got != (int(P[q, i]), int(S[q, i])):
                    bad.append((q, nm, got, (int(P[q, i]), int(S[q, i]))))
    return {"dataset": ds, "questions": hi, "checks": checks, "mismatches": bad[:5], "n_mismatches": len(bad), "verdict": "PASS" if not bad else "FAIL"}


def gate_gx5(op, ds, n_q=60, n_ref=3):
    """With the PL5 options off expand5 == zpl3g.expand3 on n_q questions (every regime, every setting, cs and vr in every combination);
    every PL5 walk and every PL3 walk against the per-node reference on n_ref questions."""
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
                        if vr and SET5[stg]["hops"] == 1:
                            continue
                        mods = tuple(m for m, on in (("cs", cs), ("vr", vr)) if on)
                        a_ = expand5(seeds5, [src.fams[f] for f in h1], [src.fams[f] for f in h2], SET5[stg], mods, scorer)
                        b_ = Z3.expand3(seeds5, [src.fams[f] for f in h1], [src.fams[f] for f in h2], SET5[stg], cs, vr, scorer)
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
        for w in Z3.walks_all() + Z4.walks4() + walks5():
            regime, stg, mods = parse_walk5(w)
            sd = np.asarray(f1[:10] if "s10" in mods else (f1[:20] if "s20" in mods else seeds5), dtype=np.int64)
            h1, h2 = REG3[regime]
            a_ = expand5(sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], SET5[stg], mods, scorer)
            b_ = ref_walk5(sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], SET5[stg], mods, qv, src.dvec)
            n_checked += 1
            if not np.array_equal(a_, b_):
                bad_ref += 1
        scorer.release()
    return {"dataset": ds, "off_mode_vs_expand3_mismatches": nbad, "per_node_reference_checks": n_checked, "per_node_reference_mismatches": bad_ref,
            "verdict": "PASS" if nbad == 0 and bad_ref == 0 else "FAIL"}


def gate_gf5():
    bad = []
    fns = (parse_walk5, parse5, Plan5, capped_parts5, order_group5, order_fresh5, expand5, pool_state5, iter_states5)
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
    plan = Plan5(cell_list5())
    h = hashlib.sha256()
    for q, st in iter_states5(op, src, plan, 0, min(a.n, src.nq), chunk=a.chunk):
        h.update(Z3.state_hash3(plan, st).encode())
    print("GDHASH " + h.hexdigest(), flush=True)
    return 0


def gate_gd5(ds, n=24):
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
    res = {"stage": "PL5 selftest", "declared_in": DOC, "utc": utc(), "script_sha256": script_sha256(), "code_path_sha256": code_path_sha256()}
    names = cell_list5()
    res["cells"] = {"n": len(names), "distinct": len(set(names)), "sha256": P2.cells_sha256(names), "expected": EXPECT5_SHA256,
                    "union_distinct": len(set(cell_list_union5())),
                    "verdict": "PASS" if len(names) == EXPECT5_CELLS == len(set(names)) and P2.cells_sha256(names) == EXPECT5_SHA256
                    and len(set(cell_list_union5())) == len(P2.cell_list()) + len(Z3.cell_list3()) + len(Z4.cell_list4()) + len(names) else "FAIL"}
    res["G-F"] = gate_gf5()
    res["engine"], res["G-X5"] = [], []
    for ds in a.ds:
        res["engine"].append(selftest_engine5(op, ds, a.n, a.per_walk))
        res["G-X5"].append(gate_gx5(op, ds))
        log(f"{ds}: engine {res['engine'][-1]['verdict']} ({res['engine'][-1]['checks']} checks), G-X5 {res['G-X5'][-1]['verdict']}")
    res["G-D"] = gate_gd5(a.ds[0])
    res["verdict"] = "PASS" if all(x["verdict"] == "PASS" for x in [res["cells"], res["G-F"], res["G-D"]] + res["engine"] + res["G-X5"]) else "FAIL"
    res["seconds"] = round(time.time() - t0, 1)
    wjson(PL5 / "selftest.json", res)
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
    p.add_argument("--n4", type=int, default=150)
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
