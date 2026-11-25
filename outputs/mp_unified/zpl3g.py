"""PL3 (docs/PL3_COVERAGE_ROUND2.md): round 2 of the coverage target. New pool-construction families on top of PL2's grid.

    python outputs/mp_unified/zpl3g.py cells [--write]                         # the declared cell lists and their sha256
    python outputs/mp_unified/zpl3g.py time --ds D [--n 40] [--walks SUBSTR]   # cost of the walks, no labels read
    python outputs/mp_unified/zpl3g.py gr --ds D [--n 400] [--host]            # G-R: PL2's 804 cells re-built by this engine == PL2's dev arrays
    python outputs/mp_unified/zpl3g.py grid --ds D --carve dev --first frozen [--part I K] [--host]
    python outputs/mp_unified/zpl3g.py select --ds D [--host]                  # the rule on dev over PL2's cells + PL3's
    python outputs/mp_unified/zpl3g.py confirm --ds D [--first frozen e3v] [--host]
    python outputs/mp_unified/zpl3g.py call                                    # PL3's chosen cell against PL2's, one call over the seven
    python outputs/mp_unified/zpl3g.py selftest [--ds D ...]                   # G-X3, G-F, G-D, and the engine against a per-node reference

This module imports zpl2g.py (PL2's engine: the sources, the lists, the grading helpers, the walk of m3b_pools) and edits none of it. It
adds, around the same pool code, five options a walk can switch on: the neighbours kept by the caps are the best by cosine to the
question instead of the first in the CSR row (`cs`); the visited cap keeps the best fresh nodes by cosine instead of the first to arrive
(`vr`); the walk may use one family set at hop 1 and another at hops 2 and 3 (regime `FS`); the seeds may be the top 10 or 20 of the fused
list (`s10`, `s20`); a ranked cell may order the expansion by cosine to the question plus the best first-stage documents (`brg`). With every
option off the engine gives PL2's cells element for element (gate G-R). Every pool function takes and reads no labels; labels enter in
grade3 only. Everything is deterministic: integer walks, float64 inner products in numpy's pairwise order (no BLAS), ties to the lower id.
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
from mp_retrieval import m3b_pools  # noqa: E402

DOC = "docs/PL3_COVERAGE_ROUND2.md"
PL3 = ROOT / "outputs" / "pl3"
SEVEN, SIX, MHR = P2.SEVEN, P2.SIX, P2.MHR
LAPTOP5 = ("multihoprag", "squad", "musique", "metaqa", "webqsp")
FLOORS, BOUND = P2.FLOORS, P2.BOUND
SCORE_WINDOW = 512                                          # `cs`: the neighbours scored per node and family are its first 512 in row order
INF = P2.INF

# ── the declared cells (docs/PL3_COVERAGE_ROUND2.md section 5) ───────────────

S1, S2, S3, S3S = "h1_c100", "h2_c100_v5000", "h3_c100_v5000", "h3_c25_v2000"
REG3 = {**{k: (v, v) for k, v in P2.REGIMES.items()}, "FS": (P2.REGIMES["FULL"], P2.REGIMES["STRUCT"])}
MODS_ORDER = ("s10", "s20", "cs", "vr")
BASES3 = ("rrf1000_b25", "rrf1000_b100", "rrf1000_b400", "rrf1000_b800", "rrf1000_b1600", "dense_top200")
M3 = (50, 100, 200, 400, 800, 1600)
BRG_BASES, BRG_M = ("rrf1000_b100", "rrf1000_b400"), (100, 400)
BRIDGE_DOCS = 3                                              # `brg`: the question vector plus the unit centroid of the fused list's first 3 documents


def walk_name(regime, setting, mods=()):
    return f"{regime}:{setting}" + (("~" + "-".join(mods)) if mods else "")


def walks_all():
    w = []
    for reg in ("STRUCT", "FULL"):                           # candidates 1 and 6: similarity caps (cs), visited cap by rank (vr)
        for mods in ((), ("cs",), ("vr",), ("cs", "vr")):    # the whole 2 x 2 at two hops
            w.append(walk_name(reg, S2, mods))
        for st in (S3, S3S):                                 # at three hops: arrival order, the rank cap, both (cs alone is read at S2)
            for mods in ((), ("vr",), ("cs", "vr")):
                w.append(walk_name(reg, st, mods))
    for st in (S2, S3, S3S):                                 # candidate 2: FULL at hop 1, STRUCT at hops 2 and 3
        for mods in ((), ("cs", "vr")):
            w.append(walk_name("FS", st, mods))
    for reg in ("STRUCT", "FULL"):                           # the one-hop walks
        for mods in ((), ("cs",)):
            w.append(walk_name(reg, S1, mods))
    for s_ in ("s10", "s20"):                                # candidate 4: seeds from the fused list
        w.append(walk_name("FULL", S1, (s_,)))
        w.append(walk_name("FULL", S2, (s_, "cs", "vr")))
        w.append(walk_name("STRUCT", S3S, (s_, "cs", "vr")))
    return w


def brg_walks():
    return [walk_name(r, S1) for r in ("STRUCT", "FULL")] + [walk_name(r, st, m) for r in ("STRUCT", "FULL") for st in (S2, S3) for m in ((), ("cs", "vr"))]


def cell_list3():
    """PL3's own cells: every walk x (5 bases as unions, 5 bases x 6 sizes as ranked cells) and the bridge-ranked cells, minus PL2's."""
    seen = set(P2.cell_list())
    out = []

    def add(n):
        if n not in seen:
            seen.add(n)
            out.append(n)
    for w in walks_all():
        for b in BASES3:
            add(f"{b}+{w}")
        for b in BASES3:
            for m in M3:
                add(f"{b}+{w}+top{m}")
    for w in brg_walks():
        for b in BRG_BASES:
            for m in BRG_M:
                add(f"{b}+{w}+brg{m}")
    return out


def cell_list_union():
    return P2.cell_list() + cell_list3()


EXPECT3_CELLS = 1470                                         # the declared cell list (docs section 5)
EXPECT3_SHA256 = "ff879881090993bca1e063b715724c0d89411b8cacc77f5df7bcccc2b29c642e"


def constants3():
    return {"settings": {k: P2.SETTINGS[k] for k in (S1, S2, S3, S3S)}, "regimes": {k: [list(v[0]), list(v[1])] for k, v in REG3.items()},
            "mods_order": list(MODS_ORDER), "bases": list(BASES3), "m": list(M3), "brg": [list(BRG_BASES), list(BRG_M), BRIDGE_DOCS],
            "score_window": SCORE_WINDOW, "walks": walks_all(), "brg_walks": brg_walks(), "floors": list(FLOORS), "bound": BOUND,
            "boot": [P2.BOOT_N, P2.BOOT_SEED], "seeds": {"s5": "dense top 5 U splade top 5", "s10": "fused list top 10", "s20": "fused list top 20"},
            "rank": "float64 inner product, ties to the lower node id"}


def require_cells():
    names = cell_list3()
    if len(names) != EXPECT3_CELLS or len(set(names)) != EXPECT3_CELLS:
        P2.stop(f"the PL3 cell list has {len(names)} cells ({len(set(names))} distinct), expected {EXPECT3_CELLS}")
    if P2.cells_sha256(names) != EXPECT3_SHA256:
        P2.stop("the PL3 cell list is not the declared one (cells_sha256 differs from EXPECT3_SHA256)")


# ── cell names -> a plan ─────────────────────────────────────────────────────

def parse_walk(w):
    regime, rest = w.split(":")
    setting, _, ms = rest.partition("~")
    mods = tuple(ms.split("-")) if ms else ()
    if regime not in REG3 or setting not in P2.SETTINGS:
        raise ValueError(f"unknown walk {w}")
    if any(m not in MODS_ORDER for m in mods) or list(mods) != sorted(mods, key=MODS_ORDER.index) or ("s10" in mods and "s20" in mods):
        raise ValueError(f"unknown walk {w}")
    if "vr" in mods and P2.SETTINGS[setting]["hops"] == 1:
        raise ValueError(f"vr has nothing to rank in a one-hop walk: {w}")
    return (regime, setting, mods)


def parse3(name):
    """('R', list, b) | ('FU',) | ('G', list, b, walk) | ('K', list, b, walk, M, rv); a walk = (regime, setting, mods); rv = 'q' or 'b'."""
    if "+" not in name:
        return P2.parse(name)
    parts = name.split("+")
    ln, b = P2.base_of(parts[0])
    wk = parse_walk(parts[1])
    if len(parts) == 3:
        if parts[2].startswith("top"):
            return ("K", ln, b, wk, int(parts[2][3:]), "q")
        if parts[2].startswith("brg"):
            return ("K", ln, b, wk, int(parts[2][3:]), "b")
        raise ValueError(f"unknown cell {name}")
    if len(parts) != 2:
        raise ValueError(f"unknown cell {name}")
    return ("G", ln, b, wk)


def tags_of(name):
    """Which PL3 option a cell belongs to (descriptive, for what each family bought). PL2's own cells are tagged pl2."""
    if name in PL2_SET:
        return ("pl2",)
    p = parse3(name)
    tags = []
    if p[0] in ("G", "K"):
        regime, _st, mods = p[3]
        has_seed = any(m in mods for m in ("s10", "s20"))
        if has_seed:
            tags.append("seeds")
        elif regime == "FS":
            tags.append("fs")
        elif "cs" in mods and "vr" in mods:
            tags.append("csvr")
        elif "cs" in mods:
            tags.append("cs")
        elif "vr" in mods:
            tags.append("vr")
        else:
            tags.append("off")
        if p[0] == "K" and p[5] == "b":
            tags.append("brg")
    if p[0] in ("G", "K") and p[2] == 1600 and p[1] == "F1":
        tags.append("deep")
    return tuple(tags) or ("off",)


PL2_SET = set(P2.cell_list())


class Plan3:
    def __init__(self, names):
        self.names = list(names)
        if len(set(self.names)) != len(self.names):
            P2.stop("a cell name repeats")
        self.C = len(self.names)
        ret, fu, walks = {}, [], {}
        for i, nm in enumerate(self.names):
            p = parse3(nm)
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
        self.need_vec = any(v["K"] for v in self.walks.values()) or any(("cs" in w[2] or "vr" in w[2]) for w in self.walks)
        self.need_f1 = "F1" in lists or any(("s10" in w[2] or "s20" in w[2]) for w in self.walks) or self.need_bridge
        self.families = sorted({f for w in self.walks for part in REG3[w[0]] for f in part})


# ── the pool code (no labels anywhere in this block) ─────────────────────────

_BUF = {}                                                    # one score buffer per (node count, vector name), NaN where a node has not been scored this question


class Scorer:
    """Float64 inner products of node vectors with the question's ranking vectors, remembered for the question: a node is scored once. The
    memory is a dense array over the node ids, reset at the end of the question."""

    def __init__(self, dvec, rvs, n_nodes):
        self.dvec, self.rvs = dvec, rvs
        self.buf = {}
        for k in rvs:
            if (n_nodes, k) not in _BUF:
                _BUF[(n_nodes, k)] = np.full(n_nodes, np.nan, dtype=np.float64)
            self.buf[k] = _BUF[(n_nodes, k)]
        self.touched = {k: [] for k in rvs}
        self.n_scored = 0

    def get(self, nodes, rv="q"):
        nodes = np.asarray(nodes, dtype=np.int64)
        if nodes.size == 0:
            return np.empty(0, np.float64)
        buf = self.buf[rv]
        out = buf[nodes]
        miss = np.isnan(out)
        if miss.any():
            new = np.unique(nodes[miss])
            buf[new] = P2.dot64(self.dvec(new), self.rvs[rv])
            self.touched[rv].append(new)
            self.n_scored += int(new.size)
            out = buf[nodes]
        return out

    def release(self):
        for k, lst in self.touched.items():
            for a_ in lst:
                self.buf[k][a_] = np.nan
            lst.clear()


def capped_sim(nodes, families, cap, scorer):
    """The best `cap` neighbours by cosine of every node in every family (ties to the lower id), out of each row's first SCORE_WINDOW entries;
    ordered as capped_multi orders: node-major, then family, then best first."""
    if nodes.size == 0:
        return np.empty(0, dtype=np.int64)
    cols, node_idx, fam_idx, rank = [], [], [], []
    for f_i, fam in enumerate(families):
        starts = fam.indptr[nodes]
        lens = np.minimum(fam.indptr[nodes + 1] - starts, SCORE_WINDOW)
        total = int(lens.sum())
        if total == 0:
            continue
        off = np.cumsum(lens) - lens
        within = np.arange(total, dtype=np.int64) - np.repeat(off, lens)
        c = fam.col[np.repeat(starts, lens) + within].astype(np.int64, copy=False)
        grp = np.repeat(np.arange(nodes.size, dtype=np.int64), lens)
        sc = scorer.get(c, "q")
        o = np.lexsort((c, -sc, grp))
        c = c[o]
        keep = within < cap
        cols.append(c[keep])
        node_idx.append(grp[keep])
        fam_idx.append(np.full(int(keep.sum()), f_i, dtype=np.int8))
        rank.append(within[keep])
    if not cols:
        return np.empty(0, dtype=np.int64)
    if len(cols) == 1:
        return cols[0]
    order = np.lexsort((np.concatenate(rank), np.concatenate(fam_idx), np.concatenate(node_idx)))
    return np.concatenate(cols)[order]


def expand3(seeds, fams1, fams2, setting, cs, vr, scorer):
    """The bounded walk of m3b_pools.expand_hops with three options. fams1 = the families walked from the seeds, fams2 = the families walked
    at hops 2 and 3. cs: caps keep the best neighbours by cosine instead of the first in the row. vr: the visited cap keeps the best fresh
    nodes of a hop by cosine (ties to the lower id) instead of the first to arrive, and the next hop expands them best first. All three off
    and fams1 == fams2: m3b_pools.expand_hops exactly."""
    first = m3b_pools._first_occurrence_unique
    cap = int(setting["per_seed_cap"])
    hop1 = first(capped_sim(seeds, fams1, cap, scorer) if cs else m3b_pools.capped_multi(seeds, fams1, cap))
    hops = int(setting["hops"])
    if hops == 1:
        return hop1
    cap2 = int(setting["per_frontier_cap"])
    limit = int(setting["visited_cap"])
    visited = np.union1d(seeds, hop1)
    out = [hop1]
    frontier = hop1
    for _ in range(2, hops + 1):
        budget = limit - int(visited.size)
        if budget <= 0 or frontier.size == 0:
            break
        cand = capped_sim(frontier, fams2, cap2, scorer) if cs else m3b_pools.capped_multi(frontier, fams2, cap2)
        if cand.size == 0:
            break
        cand = cand[np.isin(cand, visited, invert=True)]
        fresh = first(cand)
        if vr and fresh.size:
            fresh = fresh[np.lexsort((fresh, -scorer.get(fresh, "q")))]
        fresh = fresh[:budget]
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return np.concatenate(out)


class Walk3:
    __slots__ = ("exp", "ro", "inv", "se", "sx")


class State3:
    __slots__ = ("L", "W", "d200", "s200")


TIMES = None                                                 # a dict while `time` runs: seconds per walk


def pool_state3(plan, d, s, f2, f1, seeds5, fams, qv, dvec, n_nodes):
    """Everything a question's pools are made of: the ranked lists and each walk's expansion with its orders. d and s are the dense and
    splade rows, f2 and f1 the fused rows, seeds5 the dense-5 U splade-5 seeds, fams the family CSRs, qv the question vector and dvec(nodes)
    the node vectors. It takes and reads no labels."""
    st = State3()
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
            top = dvec(np.asarray(f1[:BRIDGE_DOCS], dtype=np.int64)).astype(np.float64).mean(axis=0)
            rvs["b"] = rvs["q"] + top / np.sqrt((top * top).sum())
        scorer = Scorer(dvec, rvs, n_nodes)
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
        exp = expand3(sd, [fams[f] for f in h1], [fams[f] for f in h2], P2.SETTINGS[stg], "cs" in mods, "vr" in mods, scorer)
        wk = Walk3()
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


def state_hash3(plan, st):
    h = hashlib.sha256()
    for k in sorted(st.L):
        h.update(k.encode())
        h.update(np.ascontiguousarray(st.L[k].L).tobytes())
    for w in sorted(st.W):
        h.update(repr(w).encode())
        h.update(np.ascontiguousarray(st.W[w].exp).tobytes())
        for rv in sorted(st.W[w].ro):
            h.update(rv.encode())
            h.update(np.ascontiguousarray(st.W[w].ro[rv]).tobytes())
    return h.hexdigest()


# ── grading: the only place the labels enter ─────────────────────────────────

def grade3(plan, st, g, outp, outs):
    """Fill one question's present counts and pool sizes for every cell. g = the question's sorted unique labelled nodes."""
    gp = {}

    def gpos(ln):
        if ln not in gp:
            gp[ln] = st.L[ln].pos(g)
        return gp[ln]

    for ln, (ci, bs) in plan.ret.items():
        lst = st.L[ln]
        outp[ci] = np.searchsorted(np.sort(gpos(ln)), bs, side="left")
        outs[ci] = np.minimum(bs, lst.n)
    for ci in plan.fu:
        u = np.union1d(st.d200, st.s200)
        outp[ci] = int(np.isin(g, u).sum())
        outs[ci] = u.size
    for w, grp in plan.walks.items():
        wk = st.W[w]
        exp = wk.exp
        ing = np.isin(g, exp) if exp.size else np.zeros(g.size, bool)
        for ln, (ci, bs) in grp["G"].items():
            lst = st.L[ln]
            pg = gpos(ln)
            pe = np.sort(lst.pos(exp))
            hit = ing[:, None] | (pg[:, None] < bs[None, :])
            outp[ci] = hit.sum(axis=0)
            outs[ci] = np.minimum(bs, lst.n) + exp.size - np.searchsorted(pe, bs, side="left")
        for (ln, b, rv), (ci, ms) in grp["K"].items():
            lst = st.L[ln]
            pg = gpos(ln)
            base_hit = pg < b
            if exp.size == 0:
                outp[ci] = int(base_hit.sum())
                outs[ci] = min(b, lst.n)
                continue
            notin = lst.pos(exp)[wk.ro[rv]] >= b
            cum = np.cumsum(notin)
            tot = int(cum[-1])
            eidx = wk.se[np.minimum(np.searchsorted(wk.sx, g), exp.size - 1)]
            r = wk.inv[rv][eidx]
            via = (ing & notin[r])[:, None] & (cum[r][:, None] <= ms[None, :])
            outp[ci] = (base_hit[:, None] | via).sum(axis=0)
            outs[ci] = min(b, lst.n) + np.minimum(ms, tot)


# ── the engine ───────────────────────────────────────────────────────────────

def iter_states3(op, src, plan, lo, hi, chunk=64):
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
            yield c0 + j, pool_state3(plan, d[j], s[j], f2[j], f1[j], seeds[j], fams, qv[j], dvec, src.n)


def evaluate3(op, src, plan, lo, hi, every=300):
    Q = hi - lo
    P = np.zeros((Q, plan.C), dtype=np.int16)
    S = np.zeros((Q, plan.C), dtype=np.int32)
    t0 = last = time.time()
    for q, st in iter_states3(op, src, plan, lo, hi):
        grade3(plan, st, src.golds[q], P[q - lo], S[q - lo])
        if (q - lo + 1) % every == 0 and time.time() - last > 20:
            last = time.time()
            log(f"{src.name}/{src.carve}/{src.first}: {q - lo + 1}/{Q} questions, {last - t0:.0f} s ({P2.peak_gb()} GB)")
    return P, S


_SHA = {}


def code_path_sha256():
    """One hash of PL3's pool code, its constants and PL2's code-path hash (G-U): the same on every dataset, taken once at the start of a run."""
    if "code" not in _SHA:
        h = hashlib.sha256()
        for fn in (parse_walk, parse3, Plan3, Scorer, capped_sim, expand3, pool_state3, grade3, iter_states3, evaluate3):
            h.update(inspect.getsource(fn).replace("\r\n", "\n").encode("utf-8"))
        h.update(json.dumps(constants3(), sort_keys=True).encode("utf-8"))
        h.update(P2.code_path_sha256().encode("utf-8"))
        _SHA["code"] = h.hexdigest()
    return _SHA["code"]


def script_sha256():
    if "script" not in _SHA:
        _SHA["script"] = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return _SHA["script"]


# ── small helpers ────────────────────────────────────────────────────────────

log, stop, rel, rjson, wjson, utc = P2.log, P2.stop, P2.rel, P2.rjson, P2.wjson, P2.utc


def rows_of_grid3(g, dataset, population):
    return [P2.cell_row(g["present"][:, i], g["gold_counts"], g["size"][:, i], dataset=dataset, population=population, pool=nm)
            for i, nm in enumerate(g["names"])]


def grid_path3(d, carve, first, tag=""):
    return PL3 / d / f"grid_{carve}_{first}{('_' + tag) if tag else ''}"


def save_grid3(base, src, plan, P, S, lo, hi, seconds, part=None, stage="PL3 grid"):
    p = Path(str(base) + (f".part{part[0]}of{part[1]}" if part else "") + ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez(tmp, names=np.asarray(plan.names, dtype=str), present=P, size=S, qids=np.asarray(src.qids[lo:hi], dtype=str),
             gold_counts=src.gold_counts[lo:hi].astype(np.int32), lo=np.asarray(lo), hi=np.asarray(hi))
    os.replace(tmp, p)
    head = {"stage": stage, "declared_in": DOC, "dataset": src.name, "carve": src.carve, "first": src.first, "questions": int(hi - lo),
            "rows": [int(lo), int(hi)], "part": list(part) if part else None, "cells": plan.C, "cells_sha256": P2.cells_sha256(plan.names),
            "graph": getattr(src, "graph_kind", None), "population_digest": src.pop_digest, "code_path_sha256": code_path_sha256(),
            "pl2_code_path_sha256": P2.code_path_sha256(), "pl2_script_sha256": P2.script_sha256(), "script_sha256": script_sha256(),
            "constants_sha256": hashlib.sha256(json.dumps(constants3(), sort_keys=True).encode()).hexdigest(), "npz": rel(p),
            "npz_sha256": P2.sha256_file(p), "seconds": round(seconds, 1), "peak_gb": P2.peak_gb(), "host": P2.HOST, "utc": utc()}
    wjson(Path(str(p)[:-4] + ".json"), head)
    return p, head


def load_grid3(base):
    base = Path(base)
    whole = Path(str(base) + ".npz")
    files = [whole] if whole.is_file() else sorted(base.parent.glob(base.name + ".part*of*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    if not files:
        stop(f"no grid at {rel(base)}")
    parts = []
    for f in files:
        with np.load(f) as z:
            parts.append({k: z[k] for k in z.files})
    parts.sort(key=lambda p: int(p["lo"]))
    pos = 0
    for p in parts:
        if int(p["lo"]) != pos:
            stop(f"the parts of {rel(base)} do not tile the questions (gap at {pos})")
        pos = int(p["hi"])
    names = [str(x) for x in parts[0]["names"]]
    for p in parts:
        if [str(x) for x in p["names"]] != names:
            stop(f"the parts of {rel(base)} name different cells")
    return {"names": names, "present": np.concatenate([p["present"] for p in parts]), "size": np.concatenate([p["size"] for p in parts]),
            "qids": [str(x) for p in parts for x in p["qids"]], "gold_counts": np.concatenate([p["gold_counts"] for p in parts]).astype(np.int64)}


def heads_of(base):
    base = Path(base)
    whole = Path(str(base) + ".json")
    hs = [whole] if whole.is_file() else sorted(base.parent.glob(base.name + ".part*of*.json"))
    return [h for h in hs if not h.name.endswith(".tmp.json")]


# ── commands: cells, time ────────────────────────────────────────────────────

def cmd_cells(a):
    new, uni = cell_list3(), cell_list_union()
    fam = {}
    for n in new:
        fam[tags_of(n)] = fam.get(tags_of(n), 0) + 1
    log(f"PL3 cells {len(new)} ({len(set(new))} distinct), sha256 {P2.cells_sha256(new)}; union with PL2's {len(uni)} ({len(set(uni))} distinct), "
        f"sha256 {P2.cells_sha256(uni)}")
    log(f"walks {len(walks_all())} (+{len(brg_walks())} bridge walks); tags {fam}")
    log(f"code path sha256 {code_path_sha256()}")
    if a.write:
        wjson(PL3 / "cells.json", {"cells": new, "sha256": P2.cells_sha256(new), "union_sha256": P2.cells_sha256(uni), "n": len(new)})
    return 0


def cmd_time(a):
    global TIMES
    op = P2.opened()
    names = cell_list3()
    if a.walks:
        names = [n for n in names if a.walks in n]
    plan = Plan3(names)
    src = P2.load_source(a.ds, "dev", "frozen")
    n = min(a.n, src.nq)
    TIMES = {}
    t0 = time.time()
    for q, st in iter_states3(op, src, plan, 0, n):
        pass
    tot = time.time() - t0
    sc = TIMES.pop("_scored", 0)
    log(f"{a.ds}: {n} questions, {plan.C} cells, {len(plan.walks)} walks, {tot:.1f} s = {tot / n:.3f} s/question; nodes scored {sc / n:.0f}/question; peak {P2.peak_gb()} GB")
    for w, t in sorted(TIMES.items(), key=lambda kv: -kv[1])[:int(a.top)]:
        log(f"   {walk_name(*w):44s} {1000 * t / n:8.1f} ms/question")
    return 0


# ── G-R: PL2's cells, re-built by this engine ────────────────────────────────

def cmd_gr(a):
    t0 = time.time()
    op = P2.opened()
    names = P2.cell_list()
    plan = Plan3(names)
    src = P2.load_source(a.ds, "dev", "frozen")
    ref = P2.load_grid(a.ds, "dev", "frozen")
    n = min(a.n, src.nq)
    if ref["names"] != names or ref["qids"][:n] != list(src.qids[:n]):
        stop(f"{a.ds}: PL2's dev grid does not name the same cells or questions")
    P, S = evaluate3(op, src, plan, 0, n)
    bad_p = np.argwhere(P != ref["present"][:n])
    bad_s = np.argwhere(S != ref["size"][:n])
    out = {"stage": "PL3 G-R", "declared_in": DOC, "dataset": a.ds, "host": P2.HOST, "questions": int(n), "cells": len(names),
           "elements": int(P.size), "present_mismatches": int(len(bad_p)), "size_mismatches": int(len(bad_s)),
           "first_mismatch": ([int(bad_p[0][0]), names[int(bad_p[0][1])]] if len(bad_p) else (
               [int(bad_s[0][0]), names[int(bad_s[0][1])]] if len(bad_s) else None)),
           "pl2_code_path_sha256": P2.code_path_sha256(), "code_path_sha256": code_path_sha256(), "script_sha256": script_sha256(),
           "verdict": "PASS" if not len(bad_p) and not len(bad_s) else "FAIL", "seconds": round(time.time() - t0, 1), "peak_gb": P2.peak_gb(), "utc": utc()}
    wjson(PL3 / f"gr_{a.ds}{'_host' if P2.HOST else ''}.json", out)
    log(f"G-R {a.ds}: {out['verdict']}, {n} questions x {len(names)} cells = {out['elements']} elements, mismatches {out['present_mismatches']}/{out['size_mismatches']} "
        f"({out['seconds']} s)")
    return 0 if out["verdict"] == "PASS" else 3


def gates_passed3(ds):
    ref = ds if (ds in LAPTOP5 and not P2.HOST) else "musique"
    p = PL3 / f"gr_{ref}.json"
    if not p.is_file() or rjson(p)["verdict"] != "PASS":
        stop(f"{p.name} is not PASS: run `zpl3g.py gr --ds {ref}` first")
    if P2.HOST:
        h = PL3 / "gr_musique_host.json"
        if not h.is_file() or rjson(h)["verdict"] != "PASS":
            stop("gr_musique_host.json is not PASS: run `zpl3g.py gr --ds musique --host` first")
    st = PL3 / "selftest.json"
    if not st.is_file() or rjson(st)["verdict"] != "PASS":
        stop("selftest.json is not PASS: run `zpl3g.py selftest` first")


def cmd_grid(a):
    t0 = time.time()
    P2.HOST = bool(a.host)
    gates_passed3(a.ds)
    names = cell_list3() if a.cells == "new" else [str(x) for x in rjson(a.cells)]
    if a.cells == "new":
        require_cells()
    plan = Plan3(names)
    op = P2.opened()
    src = P2.load_source(a.ds, a.carve, a.first)
    lo, hi, part = 0, src.nq, None
    if a.part:
        i, k = int(a.part[0]), int(a.part[1])
        lo, hi = i * src.nq // k, (i + 1) * src.nq // k
        part = (i, k)
    log(f"{a.ds}/{a.carve}/{a.first}: {src.nq} questions, rows {lo}..{hi}, {plan.C} cells, {len(plan.walks)} walks, graph {src.graph_kind} ({P2.peak_gb()} GB)")
    P, S = evaluate3(op, src, plan, lo, hi)
    tag = a.tag or ("" if a.cells == "new" else "sub")
    p, head = save_grid3(grid_path3(a.ds, a.carve, a.first, tag), src, plan, P, S, lo, hi, time.time() - t0, part)
    log(f"{a.ds}/{a.carve}/{a.first}: filed {rel(p)} in {head['seconds']} s")
    return 0


# ── select ───────────────────────────────────────────────────────────────────

def union_dev(ds):
    """PL2's dev grid and PL3's side by side (same questions, disjoint cells), their heads checked against the code that is running."""
    g2 = P2.load_grid(ds, "dev", "frozen")
    for hp in heads_of(P2.grid_path(ds, "dev", "frozen")):
        hd = rjson(hp)
        if hd["code_path_sha256"] != P2.code_path_sha256() or hd["cells_sha256"] != P2.EXPECT_CELLS_SHA256:
            stop(f"G-U: {hp.name} (PL2) was written by other pool code or another cell list than the one that is running")
    g3 = load_grid3(grid_path3(ds, "dev", "frozen"))
    h3 = heads_of(grid_path3(ds, "dev", "frozen"))
    if not h3:
        stop(f"{ds}: no PL3 grid heads")
    for hp in h3:
        hd = rjson(hp)
        if hd["code_path_sha256"] != code_path_sha256() or hd["cells_sha256"] != EXPECT3_SHA256:
            stop(f"G-U: {hp.name} was written by other PL3 code or another cell list than the one that is running")
    if g2["qids"] != g3["qids"] or not np.array_equal(g2["gold_counts"], g3["gold_counts"]):
        stop(f"{ds}: PL2's and PL3's dev grids are not on the same questions")
    if set(g2["names"]) & set(g3["names"]):
        stop("PL2's and PL3's cells overlap")
    if P2.cells_sha256(g3["names"]) != EXPECT3_SHA256:
        stop("the PL3 dev grid is not the declared cell list")
    return {"names": g2["names"] + g3["names"], "present": np.concatenate([g2["present"], g3["present"]], axis=1),
            "size": np.concatenate([g2["size"], g3["size"]], axis=1), "qids": g2["qids"], "gold_counts": g2["gold_counts"]}, h3


READINGS = ("all", "pl2", "new")
TAGS = ("off", "cs", "vr", "csvr", "fs", "seeds", "deep", "brg")


def family3(name):
    return "pl2" if name in PL2_SET else "new"


def cmd_select(a):
    P2.HOST = bool(a.host)
    gates_passed3(a.ds)
    g, heads = union_dev(a.ds)
    names = g["names"]
    rows = rows_of_grid3(g, a.ds, "dev")
    table = [{"name": n, "family": family3(n), "tags": list(tags_of(n)), "nodes": r["candidates_mean"], "all": r["all_gold_at_pool"], "any": r["any_gold_at_pool"],
              "coverage": r["coverage_micro"], "p95": r["candidates_p95"], "max": r["candidates_max"]} for n, r in zip(names, rows)]
    readings = {"all": table, "pl2": [t for t in table if t["family"] == "pl2"], "new": [t for t in table if t["family"] == "new"]}
    for tg in TAGS:
        readings[f"pl2+{tg}"] = [t for t in table if t["family"] == "pl2" or tg in t["tags"]]
        readings[f"only:{tg}"] = [t for t in table if t["family"] == "new" and tg in t["tags"]]
    out = {"stage": "PL3 select", "declared_in": DOC, "dataset": a.ds, "carve": "dev", "first": "frozen", "questions": len(g["qids"]), "bound": BOUND,
           "cells_sha256_pl3": EXPECT3_SHA256, "cells_sha256_pl2": P2.EXPECT_CELLS_SHA256, "code_path_sha256": code_path_sha256(),
           "pl2_code_path_sha256": P2.code_path_sha256(), "script_sha256": script_sha256(), "utc": utc(),
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
    out["frontier"] = {rd: P2.frontier(readings[rd]) for rd in ("all", "pl2", "new")}
    grid_n = (25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500)
    out["bought"] = [{"nodes": n, "pl2": P2.front_at(out["frontier"]["pl2"], n), "new": P2.front_at(out["frontier"]["new"], n), "all": P2.front_at(out["frontier"]["all"], n),
                      "gain": (None if P2.front_at(out["frontier"]["all"], n) is None or P2.front_at(out["frontier"]["pl2"], n) is None
                               else P2.front_at(out["frontier"]["all"], n) - P2.front_at(out["frontier"]["pl2"], n))} for n in grid_n]
    pl2_sel = PL3.parent / "pl2" / a.ds / "select.json"
    same = None
    if pl2_sel.is_file():
        same = rjson(pl2_sel)["picks"]["R95_all"]["name"] == out["picks"]["R95_pl2"]["name"] and rjson(pl2_sel)["picks"]["R98_all"]["name"] == out["picks"]["R98_pl2"]["name"]
    out["pl2_pick_equals_filed"] = same
    out["old"] = P2.old_cell(None, a.ds)
    out["table"] = table
    wjson(PL3 / a.ds / "select.json", out)
    for k in ("R95_all", "R95_pl2", "R95_new", "R98_all", "R98_pl2", "R98_new"):
        c = out["picks"][k]
        log(f"{a.ds} {k}: {c['name']}  ALL {c['all']:.4f} at {c['nodes']:.1f} nodes{'' if c['reached'] else '  (SHORT)'}")
    log(f"{a.ds}: PL2's pick re-read equals its filed select.json: {same}")
    return 0


# ── confirm and call ─────────────────────────────────────────────────────────

def confirm_names3(sel):
    names = []
    for k in ("R95_all", "R98_all", "R95_pl2", "R98_pl2", "R95_new", "R98_new"):
        c = sel["picks"][k]["name"]
        if c not in names:
            names.append(c)
    if sel["old"] not in names:
        names.append(sel["old"])
    return names


def cmd_confirm(a):
    P2.HOST = bool(a.host)
    gates_passed3(a.ds)
    op = P2.opened()
    sel = rjson(PL3 / a.ds / "select.json")
    names = confirm_names3(sel)
    plan = Plan3(names)
    for first in a.first:
        t0 = time.time()
        try:
            src = P2.load_source(a.ds, "eval", first)
        except SystemExit as e:
            log(f"{a.ds}/eval/{first}: not read here ({e})")
            continue
        plan_use = plan
        if plan.need_vec:
            try:
                src.dvec(np.asarray([0]))
            except Exception as e:
                log(f"{a.ds}/eval/{first}: the ranking vectors are not here ({e}); the vector-using cells are left for the host")
                continue
        P, S = evaluate3(op, src, plan_use, 0, src.nq)
        save_grid3(grid_path3(a.ds, "eval", first, "confirm"), src, plan_use, P, S, 0, src.nq, time.time() - t0, stage="PL3 confirm grid")
        g = {"names": plan_use.names, "present": P, "size": S, "gold_counts": src.gold_counts}
        rows = {r["pool"]: r for r in rows_of_grid3(g, a.ds, "eval")}
        res = {"stage": "PL3 confirm", "declared_in": DOC, "dataset": a.ds, "first": first, "questions": src.nq, "utc": utc(), "cells": {}}
        for n in plan_use.names:
            r = rows[n]
            k = int(round(r["all_gold_at_pool"] * src.nq))
            res["cells"][n] = {"all": r["all_gold_at_pool"], "all_ci": P2.wilson(k, src.nq), "any": r["any_gold_at_pool"], "coverage": r["coverage_micro"],
                               "nodes": r["candidates_mean"], "p95": r["candidates_p95"], "max": r["candidates_max"]}
        res["named"] = {}
        for k, c in sel["picks"].items():
            if k.split("_", 1)[1] in READINGS and c["name"] in res["cells"]:
                e = res["cells"][c["name"]]
                res["named"][k] = {"name": c["name"], "dev_all": c["all"], "dev_nodes": c["nodes"], "eval_all": e["all"], "eval_nodes": e["nodes"],
                                   "floor": c["floor"], "confirmed": bool(e["all"] >= c["floor"]), "short_by": max(0.0, c["floor"] - e["all"]),
                                   "dev_to_eval": e["all"] - c["all"]}
        e = res["cells"][sel["old"]]
        res["old"] = {"name": sel["old"], "eval_all": e["all"], "eval_nodes": e["nodes"]}
        wjson(PL3 / a.ds / f"confirm_{first}.json", res)
        for k, v in res["named"].items():
            log(f"{a.ds}/{first} {k}: {v['name']} eval ALL {v['eval_all']:.4f} at {v['eval_nodes']:.1f} nodes "
                f"({'CONFIRMED' if v['confirmed'] else 'SHORT by %.2f pts' % (100 * v['short_by'])}; dev {v['dev_all']:.4f})")
    return 0


def status_of(pr, new_reaches, old_reaches):
    below = (not new_reaches and old_reaches) or (pr["all_diff_ci"][1] < 0 and pr["nodes_ratio_ci"][0] >= 1)
    above = (new_reaches and not old_reaches) or (pr["nodes_ratio_ci"][1] < 1) or (pr["all_diff_ci"][0] > 0 and pr["nodes_ratio_ci"][1] <= 1)
    return "BELOW" if below else ("ABOVE" if above else "LEVEL")


def cmd_call(a):
    out = {"stage": "PL3 call", "declared_in": DOC, "utc": utc(), "datasets": {}}
    codes = {}
    rng = np.random.default_rng(P2.BOOT_SEED)
    for ds in SEVEN:
        sp, cp = PL3 / ds / "select.json", grid_path3(ds, "eval", "frozen", "confirm")
        if not sp.is_file() or not Path(str(cp) + ".npz").is_file():
            out["datasets"][ds] = {"status": "NOT_READ", "why": "no PL3 select or confirm file here"}
            continue
        sel = rjson(sp)
        codes[ds] = sel["code_path_sha256"]
        g = load_grid3(cp)
        at = {n: i for i, n in enumerate(g["names"])}
        gc_ = g["gold_counts"]

        def arrs(n):
            i = at[n]
            return (g["present"][:, i].astype(np.int64) == gc_), g["size"][:, i].astype(np.float64)
        rec = {}
        for tag, nk, ok in (("R95", "R95_all", "R95_pl2"), ("R98", "R98_all", "R98_pl2"), ("R95_new_only", "R95_new", "R95_pl2")):
            new_name, old_name = sel["picks"][nk]["name"], sel["picks"][ok]["name"]
            new, old = arrs(new_name), arrs(old_name)
            pr = P2.paired(new, old, rng)
            floor = FLOORS[1] if tag == "R98" else FLOORS[0]
            nr, orr = bool(new[0].mean() >= floor), bool(old[0].mean() >= floor)
            rec[tag] = {"status": status_of(pr, nr, orr), "new": new_name, "old": old_name, "same_cell": new_name == old_name, "new_all": float(new[0].mean()),
                        "old_all": float(old[0].mean()), "new_nodes": float(new[1].mean()), "old_nodes": float(old[1].mean()), "new_reaches": nr, "old_reaches": orr,
                        "within_bound": bool(new[1].mean() <= BOUND), **pr}
        out["datasets"][ds] = {**rec["R95"], "status": rec["R95"]["status"], "r98": rec["R98"], "new_only": rec["R95_new_only"]}
    read = {k: v for k, v in out["datasets"].items() if v["status"] != "NOT_READ"}
    out["datasets_read"] = sorted(read)
    out["code_path_equal_everywhere"] = len(set(codes.values())) <= 1
    if len(set(codes.values())) > 1:
        stop(f"G-U: the code-path hash differs between datasets: {codes}")
    n_below = sum(v["status"] == "BELOW" for v in read.values())
    n_above = sum(v["status"] == "ABOVE" for v in read.values())
    out["call"] = ("CARRY_PL3" if (len(read) == len(SEVEN) and n_below == 0 and n_above >= 1 and all(v["within_bound"] for v in read.values()))
                   else ("INCOMPLETE" if len(read) < len(SEVEN) else "NOT_CARRIED"))
    wjson(PL3 / "call.json", out)
    for ds, v in out["datasets"].items():
        log(f"call {ds}: {v['status']}" + ("" if v["status"] == "NOT_READ" else f"  NEW {v['new']} ALL {v['new_all']:.4f} @ {v['new_nodes']:.0f}; PL2 {v['old']} ALL {v['old_all']:.4f} @ {v['old_nodes']:.0f}"))
    log(f"call: {out['call']} (BELOW {n_below}, ABOVE {n_above}, read {len(read)}/7)")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def ref_walk3(seeds, fams1, fams2, setting, cs, vr, qv, dvec):
    """The PL3 walk as a per-node loop, written from the declaration (docs section 4), for G-X3 and the engine test."""
    cap, cap2, limit, hops = setting["per_seed_cap"], setting.get("per_frontier_cap"), setting.get("visited_cap"), setting["hops"]

    memo = {}

    def one(nodes):
        nodes = [int(x) for x in nodes]
        miss = sorted(set(x for x in nodes if x not in memo))
        if miss:
            for x, v in zip(miss, P2.dot64(dvec(np.asarray(miss, dtype=np.int64)), qv)):
                memo[x] = float(v)
        return [memo[x] for x in nodes]

    def nbrs(v, fams, c):
        out = []
        for f in fams:
            row = [int(x) for x in f.col[f.indptr[v]:f.indptr[v + 1]]]
            if cs and row:
                row = row[:SCORE_WINDOW]
                sc = one(row)
                row = [row[i] for i in sorted(range(len(row)), key=lambda i: (-sc[i], row[i]))]
            out += row[:c]
        return out
    seen, hop1 = set(), []
    for v in seeds:
        for x in nbrs(int(v), fams1, cap):
            if x not in seen:
                seen.add(x)
                hop1.append(x)
    out = list(hop1)
    if hops == 1:
        return np.asarray(out, dtype=np.int64)
    visited = set(int(x) for x in seeds) | set(hop1)
    frontier = hop1
    for _ in range(2, hops + 1):
        budget = limit - len(visited)
        if budget <= 0 or not frontier:
            break
        fresh, seen2 = [], set()
        for v in frontier:
            for x in nbrs(int(v), fams2, cap2):
                if x not in visited and x not in seen2:
                    seen2.add(x)
                    fresh.append(x)
        if vr and fresh:
            sc = one(fresh)
            fresh = [fresh[i] for i in sorted(range(len(fresh)), key=lambda i: (-sc[i], fresh[i]))]
        fresh = fresh[:budget]
        if not fresh:
            break
        out += fresh
        visited |= set(fresh)
        frontier = fresh
    return np.asarray(out, dtype=np.int64)


def brute_cell3(op, name, d, s, g, fams, seeds5, qv, dvec, exps):
    """One cell of one question by explicit sets (exps caches the per-node walks of this question), for the engine to be held to."""
    M3A = op.m3a
    p = parse3(name)
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
            exps[p[3]] = ref_walk3(sd, [fams[f] for f in h1], [fams[f] for f in h2], P2.SETTINGS[stg], "cs" in mods, "vr" in mods, qv, dvec)
        exp = exps[p[3]]
        if p[0] == "G":
            pool = np.union1d(base, exp)
        else:
            cand = exp[np.isin(exp, base, invert=True)]
            r = np.asarray(qv, dtype=np.float64)
            if p[5] == "b":
                top = dvec(np.asarray(f1[:BRIDGE_DOCS], dtype=np.int64)).astype(np.float64).mean(axis=0)
                r = r + top / np.sqrt((top * top).sum())
            sc = P2.dot64(dvec(cand), r) if cand.size else np.empty(0)
            order = sorted(range(cand.size), key=lambda i: (-sc[i], int(cand[i])))
            pool = np.union1d(base, cand[order[:p[4]]])
    return int(np.isin(g, pool).sum()), int(np.unique(pool).size)


def selftest_engine3(op, ds, n_q, per_walk):
    """The engine against brute_cell3 on a sample of questions: every walk, a few of its cells, and the retrieval cells."""
    src = P2.load_source(ds, "dev", "frozen")
    names = cell_list3()
    plan = Plan3(names)
    hi = min(n_q, src.nq)
    P, S = evaluate3(op, src, plan, 0, hi)
    M3A = op.m3a
    bad, checks = [], 0
    rng = np.random.default_rng(11)
    by_walk = {}
    for i, nm in enumerate(names):
        p = parse3(nm)
        by_walk.setdefault(p[3], []).append(i)
    for q in range(0, hi, max(1, hi // 4)):
        d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
        seeds5 = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
        qv = src.qvec(q, q + 1)[0]
        exps = {}
        for w, idx in by_walk.items():
            for i in rng.choice(idx, size=min(per_walk, len(idx)), replace=False):
                nm = names[int(i)]
                got = brute_cell3(op, nm, d, s, src.golds[q], src.fams, seeds5, qv, src.dvec, exps)
                checks += 1
                if got != (int(P[q, i]), int(S[q, i])):
                    bad.append((q, nm, got, (int(P[q, i]), int(S[q, i]))))
    return {"dataset": ds, "questions": hi, "checks": checks, "mismatches": bad[:5], "n_mismatches": len(bad), "verdict": "PASS" if not bad else "FAIL"}


def gate_gx3(op, ds, n_q=60, n_ref=3):
    """Off-mode expand3 == expand_hops on n_q questions (every regime, every setting); every option against the per-node reference on n_ref."""
    src = P2.load_source(ds, "dev", "frozen")
    M3A = op.m3a
    nq = min(n_q, src.nq)
    nbad = 0
    for rg, fl in P2.REGIMES.items():
        fam = [src.fams[f] for f in fl]
        for stg in P2.SETTINGS:
            for q in range(nq):
                d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
                seeds = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
                if not np.array_equal(expand3(seeds, fam, fam, P2.SETTINGS[stg], False, False, None), m3b_pools.expand_hops(seeds, fam, P2.SETTINGS[stg])):
                    nbad += 1
    bad_ref, n_checked = 0, 0
    for q in range(min(n_ref, src.nq)):
        d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
        seeds5 = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
        f1 = M3A.rrf_fused_rows(d[None, :], s[None, :], P2.RRF_CONST)[0]
        qv = src.qvec(q, q + 1)[0]
        scorer = Scorer(src.dvec, {"q": np.asarray(qv, dtype=np.float64)}, src.n)
        for w in walks_all():
            regime, stg, mods = parse_walk(w)
            sd = np.asarray(f1[:10] if "s10" in mods else (f1[:20] if "s20" in mods else seeds5), dtype=np.int64)
            h1, h2 = REG3[regime]
            a_ = expand3(sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], P2.SETTINGS[stg], "cs" in mods, "vr" in mods, scorer)
            b_ = ref_walk3(sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], P2.SETTINGS[stg], "cs" in mods, "vr" in mods, qv, src.dvec)
            n_checked += 1
            if not np.array_equal(a_, b_):
                bad_ref += 1
        scorer.release()
    return {"dataset": ds, "off_mode_vs_expand_hops_mismatches": nbad, "per_node_reference_checks": n_checked, "per_node_reference_mismatches": bad_ref,
            "verdict": "PASS" if nbad == 0 and bad_ref == 0 else "FAIL"}


def gate_gf3():
    bad = []
    fns = (parse_walk, parse3, Plan3, Scorer, Scorer.get, capped_sim, expand3, pool_state3, state_hash3, iter_states3)
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
    plan = Plan3(cell_list3())
    h = hashlib.sha256()
    for q, st in iter_states3(op, src, plan, 0, min(a.n, src.nq), chunk=a.chunk):
        h.update(state_hash3(plan, st).encode())
    print("GDHASH " + h.hexdigest(), flush=True)
    return 0


def gate_gd3(ds, n=24):
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
    res = {"stage": "PL3 selftest", "declared_in": DOC, "utc": utc(), "script_sha256": script_sha256(), "code_path_sha256": code_path_sha256()}
    names = cell_list3()
    res["cells"] = {"n": len(names), "distinct": len(set(names)), "sha256": P2.cells_sha256(names), "expected": EXPECT3_SHA256,
                    "union_distinct": len(set(cell_list_union())),
                    "verdict": "PASS" if len(names) == EXPECT3_CELLS == len(set(names)) and P2.cells_sha256(names) == EXPECT3_SHA256
                    and len(set(cell_list_union())) == len(P2.cell_list()) + len(names) else "FAIL"}
    res["G-F"] = gate_gf3()
    res["engine"], res["G-X3"] = [], []
    for ds in a.ds:
        res["engine"].append(selftest_engine3(op, ds, a.n, a.per_walk))
        res["G-X3"].append(gate_gx3(op, ds))
        log(f"{ds}: engine {res['engine'][-1]['verdict']} ({res['engine'][-1]['checks']} checks), G-X3 {res['G-X3'][-1]['verdict']}")
    res["G-D"] = gate_gd3(a.ds[0])
    res["verdict"] = "PASS" if all(x["verdict"] == "PASS" for x in [res["cells"], res["G-F"], res["G-D"]] + res["engine"] + res["G-X3"]) else "FAIL"
    res["seconds"] = round(time.time() - t0, 1)
    wjson(PL3 / "selftest.json", res)
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
    code_path_sha256()
    script_sha256()
    return {"cells": cmd_cells, "time": cmd_time, "gr": cmd_gr, "grid": cmd_grid, "select": cmd_select, "confirm": cmd_confirm, "call": cmd_call,
            "selftest": cmd_selftest, "gd": cmd_gd}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
