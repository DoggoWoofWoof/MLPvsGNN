"""PL2 (docs/PL2_COVERAGE_TARGET.md): the coverage target, the smallest pools that hold every gold.

    python outputs/mp_unified/zpl2g.py cells                                   # the declared cell list and its sha256
    python outputs/mp_unified/zpl2g.py gk --ds musique                         # G-K (and G-K2 with --ds multihoprag)
    python outputs/mp_unified/zpl2g.py grid --ds D --carve dev --first frozen [--part I K] [--cells all|filed|FILE.json]
    python outputs/mp_unified/zpl2g.py select --ds D                           # the rule on dev: R95, R98, the curves
    python outputs/mp_unified/zpl2g.py confirm --ds D [--first frozen e3v]     # the named cells on eval
    python outputs/mp_unified/zpl2g.py call                                    # NEW against OLD, one call over the seven
    python outputs/mp_unified/zpl2g.py selftest [--ds D ...]                   # G-X, G-F, G-D, and the engine against a brute-force pool builder

The engine grades every cell of a (question, regime, setting) from positions: a base is a prefix of a ranked list, so a gold is in
base U expansion iff its position in the list is below the prefix length or it is in the expansion, and the union's size is
|prefix| + |expansion| - |expansion within the prefix|. No pool is materialised twice. The functions that build pools (pool_state and
what it calls) take no labels and read none; labels enter only in grade(). This module imports scripts/m3a_headroom.py and
mp_retrieval.m3b_pools (the walk, byte-identical to m3a's for hops <= 2) and edits none of them. Everything is deterministic: the
walk is integer, the ranking vector is a float64 sum in numpy's pairwise order (no BLAS), ties go to the lower node id.
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
import gc  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.candidate_headroom import headroom_metrics  # noqa: E402
from mp_retrieval.headroom_v2 import full_coverage_ceiling  # noqa: E402

DOC = "docs/PL2_COVERAGE_TARGET.md"
PL2 = ROOT / "outputs" / "pl2"
CSR_DIR = PL2 / "csr"
M3B_CSR = ROOT / "outputs" / "m3b" / "csr"
E0_DIR = ROOT / "outputs" / "enc" / "e0" / "e3v"
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
SEVEN = SIX + ("multihoprag",)
MHR = "multihoprag"
FLOORS = (0.95, 0.98)
BOUND = 2500.0
RRF_CONST = 60
BOOT_N, BOOT_SEED = 1000, 20261125
INF = 1 << 40
KS = (1, 5, 20)                                            # configs/m3a_headroom.yaml retrieval_pools.ks (checked at open)

# ── the declared grid (docs/PL2_COVERAGE_TARGET.md section 5) ────────────────

SETTINGS = {
    "h1_c25": {"name": "h1_c25", "hops": 1, "per_seed_cap": 25},
    "h1_c100": {"name": "h1_c100", "hops": 1, "per_seed_cap": 100},
    "h2_c25": {"name": "h2_c25", "hops": 2, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 2000},
    "h3_c25_v2000": {"name": "h3_c25_v2000", "hops": 3, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 2000},
    "h3_c25_v4000": {"name": "h3_c25_v4000", "hops": 3, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 4000},
    "h2_c25_v5000": {"name": "h2_c25_v5000", "hops": 2, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 5000},
    "h2_c100_v2000": {"name": "h2_c100_v2000", "hops": 2, "per_seed_cap": 100, "per_frontier_cap": 100, "visited_cap": 2000},
    "h2_c100_v5000": {"name": "h2_c100_v5000", "hops": 2, "per_seed_cap": 100, "per_frontier_cap": 100, "visited_cap": 5000},
    "h3_c100_v5000": {"name": "h3_c100_v5000", "hops": 3, "per_seed_cap": 100, "per_frontier_cap": 100, "visited_cap": 5000},
}
FILED_SETTINGS = ("h1_c25", "h1_c100", "h2_c25")
REGIMES = {"STRUCT": ("structural",), "NER": ("ner",), "KNN": ("knn",), "FULL": ("structural", "ner", "knn")}
R0 = ("dense_top200", "splade_top200", "frozen_union", "equal_rrf_budget_50", "equal_rrf_budget_100", "equal_rrf_budget_200",
      "equal_rrf_budget_400")
RRF_B = (10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000)
DENSE_K = (5, 10, 20, 25, 30, 50, 75, 100, 150, 300, 400, 600, 800, 1000)
G_BASES = ("equal_rrf_budget_50", "equal_rrf_budget_200", "dense_top200")
FILED_BASES = ("equal_rrf_budget_50", "equal_rrf_budget_200")
G2_BASE, G2_SETTINGS = "rrf1000_b800", FILED_SETTINGS
K_BASES = ("rrf1000_b25", "rrf1000_b50", "rrf1000_b100", "rrf1000_b200", "dense_top25", "dense_top50", "dense_top100", "dense_top200")
K_M = (25, 50, 100, 200, 400)
K_SETTINGS = ("h1_c100", "h2_c25", "h2_c100_v5000", "h3_c100_v5000")
EXPECT_CELLS = 804
EXPECT_CELLS_SHA256 = "76bdf3a822e6bb8caf21121f853c9a8383e7844f99425e004d18c75f2aededc1"     # the declared 804 cells (zpl2g.py cells), fixed at the declaration commit


def cell_list():
    out = list(R0)
    out += [f"rrf1000_b{b}" for b in RRF_B] + [f"dense_top{k}" for k in DENSE_K]
    out += [f"{base}+{rg}:{st}" for base in G_BASES for rg in REGIMES for st in SETTINGS]
    out += [f"{G2_BASE}+{rg}:{st}" for rg in REGIMES for st in G2_SETTINGS]
    out += [f"{base}+{rg}:{st}+top{m}" for base in K_BASES for rg in REGIMES for st in K_SETTINGS for m in K_M]
    return out


def filed_cells():
    """The 31 cells of the filed grid (R0 and the filed G0 cells), in the filed order."""
    return list(R0) + [f"{base}+{rg}:{st}" for rg in ("STRUCT", "NER", "KNN", "FULL") for st in FILED_SETTINGS for base in FILED_BASES]


def cells_sha256(names):
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def constants():
    return {"settings": SETTINGS, "regimes": {k: list(v) for k, v in REGIMES.items()}, "r0": list(R0), "rrf_b": list(RRF_B),
            "dense_k": list(DENSE_K), "g_bases": list(G_BASES), "g2": [G2_BASE, list(G2_SETTINGS)], "k": [list(K_BASES), list(K_M), list(K_SETTINGS)],
            "floors": list(FLOORS), "bound": BOUND, "rrf_constant": RRF_CONST, "ks": list(KS), "boot": [BOOT_N, BOOT_SEED],
            "seeds": "dense top 5 U splade top 5, first occurrence", "rank": "float64 inner product, ties to the lower node id"}


# ── small helpers ────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zpl2g: {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZPL2G: {msg}")


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def wjson(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(json.dumps(obj, indent=1, default=_js).encode("utf-8"))
    os.replace(tmp, p)


def rjson(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def peak_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2 ** 30, 2)
    except Exception:
        return None


def digest_ids(ids):
    return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


# ── cell names -> a plan ─────────────────────────────────────────────────────

def base_of(base):
    """A base pool name -> (list name, prefix length). F2 = the filed fusion of the two top-200 lists, F1 = the fusion of the full
    first-stage lists (at most 1,000 deep), D = the dense list, S = the splade list."""
    if base.startswith("equal_rrf_budget_"):
        return "F2", int(base[len("equal_rrf_budget_"):])
    if base.startswith("rrf1000_b"):
        return "F1", int(base[len("rrf1000_b"):])
    if base.startswith("dense_top"):
        return "D", int(base[len("dense_top"):])
    if base.startswith("splade_top"):
        return "S", int(base[len("splade_top"):])
    raise ValueError(f"unknown base {base}")


def parse(name):
    """('R', list, b) | ('FU',) | ('G', list, b, regime, setting) | ('K', list, b, regime, setting, M)."""
    if "+" not in name:
        if name == "frozen_union":
            return ("FU",)
        ln, b = base_of(name)
        return ("R", ln, b)
    parts = name.split("+")
    ln, b = base_of(parts[0])
    regime, setting = parts[1].split(":")
    if regime not in REGIMES or setting not in SETTINGS:
        raise ValueError(f"unknown cell {name}")
    if len(parts) == 3:
        if not parts[2].startswith("top"):
            raise ValueError(f"unknown cell {name}")
        return ("K", ln, b, regime, setting, int(parts[2][3:]))
    return ("G", ln, b, regime, setting)


def family_of(name):
    if name in R0:
        return "R0"
    p = parse(name)
    if p[0] == "R":
        return "R1"
    if p[0] == "G":
        return "G2" if name.startswith(G2_BASE + "+") else "G"
    return "K"


def labels_of(name, dataset, population):
    p = parse(name)
    if p[0] in ("R", "FU"):
        return {"dataset": dataset, "population": population, "regime": "RETRIEVAL", "setting": "-", "base_pool": "-", "pool": name}
    base = name.split("+")[0]
    return {"dataset": dataset, "population": population, "regime": p[3], "setting": p[4], "base_pool": base, "pool": name}


class Plan:
    def __init__(self, names):
        self.names = list(names)
        if len(set(self.names)) != len(self.names):
            stop("a cell name repeats")
        self.C = len(self.names)
        ret, fu, walks = {}, [], {}
        for i, nm in enumerate(self.names):
            p = parse(nm)
            if p[0] == "R":
                ret.setdefault(p[1], ([], []))
                ret[p[1]][0].append(i)
                ret[p[1]][1].append(p[2])
            elif p[0] == "FU":
                fu.append(i)
            else:
                w = walks.setdefault((p[3], p[4]), {"G": {}, "K": {}})
                if p[0] == "G":
                    g = w["G"].setdefault(p[1], ([], []))
                    g[0].append(i)
                    g[1].append(p[2])
                else:
                    k = w["K"].setdefault((p[1], p[2]), ([], []))
                    k[0].append(i)
                    k[1].append(p[5])
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
        self.need_vec = any(v["K"] for v in self.walks.values())
        self.families = sorted({f for (rg, _st) in self.walks for f in REGIMES[rg]})


# ── the pool code (no labels anywhere in this block) ─────────────────────────

class Lst:
    """A ranked list of unique node ids with a position lookup (INF for a node that is not in the list)."""
    __slots__ = ("L", "n", "order", "Ls")

    def __init__(self, L):
        self.L = L
        self.n = int(L.size)
        self.order = np.argsort(L, kind="stable")
        self.Ls = L[self.order]

    def pos(self, x):
        x = np.asarray(x, dtype=np.int64)
        if self.n == 0 or x.size == 0:
            return np.full(x.shape, INF, dtype=np.int64)
        i = np.minimum(np.searchsorted(self.Ls, x), self.n - 1)
        return np.where(self.Ls[i] == x, self.order[i], INF)


class Walk:
    __slots__ = ("exp", "ro", "inv", "se", "sx")


class State:
    __slots__ = ("L", "W", "d200", "s200")


def dot64(V, q):
    """The float64 inner product of fp16 rows with a vector, summed in numpy's pairwise order, no BLAS; chunked."""
    q64 = np.asarray(q, dtype=np.float64)
    out = np.empty(V.shape[0], dtype=np.float64)
    for a in range(0, V.shape[0], 1024):
        out[a:a + 1024] = (V[a:a + 1024].astype(np.float64) * q64).sum(axis=1)
    return out


def pool_state(plan, d, s, f2, f1, seeds, fams, qv, dvec):
    """Everything a question's pools are made of: the ranked lists, each walk's expansion and (for ranked expansion) its order. d and s
    are the dense and splade rows, f2 and f1 the two fused rows, seeds the seed list, fams the family CSRs, qv the question vector and
    dvec(nodes) the node vectors. It takes and reads no labels."""
    st = State()
    st.L = {}
    if "D" in plan.lists:
        st.L["D"] = Lst(d)
    if "S" in plan.lists:
        st.L["S"] = Lst(s)
    if "F2" in plan.lists:
        st.L["F2"] = Lst(f2)
    if "F1" in plan.lists:
        st.L["F1"] = Lst(f1)
    st.d200, st.s200 = (d[:200], s[:200]) if plan.fu else (None, None)
    exps = {}
    for w in plan.walks:
        rg, stg = w
        exps[w] = m3b_pools.expand_hops(seeds, [fams[f] for f in REGIMES[rg]], SETTINGS[stg])
    kw = [w for w, v in plan.walks.items() if v["K"]]
    sc_all, allnodes = None, None
    if kw:
        allnodes = np.unique(np.concatenate([exps[w] for w in kw]))
        sc_all = dot64(dvec(allnodes), qv) if allnodes.size else np.empty(0, np.float64)
    st.W = {}
    for w, exp in exps.items():
        wk = Walk()
        wk.exp = exp
        wk.ro = wk.inv = wk.se = wk.sx = None
        if w in kw:
            sc = sc_all[np.searchsorted(allnodes, exp)] if exp.size else np.empty(0, np.float64)
            wk.ro = np.lexsort((exp, -sc))
            wk.inv = np.empty(exp.size, np.int64)
            wk.inv[wk.ro] = np.arange(exp.size)
            wk.se = np.argsort(exp, kind="stable")
            wk.sx = exp[wk.se]
        st.W[w] = wk
    return st


def state_hash(plan, st):
    h = hashlib.sha256()
    for k in sorted(st.L):
        h.update(k.encode())
        h.update(np.ascontiguousarray(st.L[k].L).tobytes())
    for w in sorted(st.W):
        h.update(repr(w).encode())
        h.update(np.ascontiguousarray(st.W[w].exp).tobytes())
        if st.W[w].ro is not None:
            h.update(np.ascontiguousarray(st.W[w].ro).tobytes())
    return h.hexdigest()


# ── grading: the only place the labels enter ─────────────────────────────────

def grade(plan, st, g, outp, outs):
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
        for (ln, b), (ci, ms) in grp["K"].items():
            lst = st.L[ln]
            pg = gpos(ln)
            base_hit = pg < b
            if exp.size == 0:
                outp[ci] = int(base_hit.sum())
                outs[ci] = min(b, lst.n)
                continue
            notin = lst.pos(exp)[wk.ro] >= b
            cum = np.cumsum(notin)
            tot = int(cum[-1])
            eidx = wk.se[np.minimum(np.searchsorted(wk.sx, g), exp.size - 1)]
            r = wk.inv[eidx]
            via = (ing & notin[r])[:, None] & (cum[r][:, None] <= ms[None, :])
            outp[ci] = (base_hit[:, None] | via).sum(axis=0)
            outs[ci] = min(b, lst.n) + np.minimum(ms, tot)


# ── sources: a dataset, a carve, a first stage ───────────────────────────────

class CSR:
    """The two arrays the walk reads (the FamilyCSR interface)."""

    def __init__(self, indptr, col):
        self.indptr, self.col = indptr, col
        self.n_nodes = int(indptr.size - 1)


class Fams:
    def __init__(self, loader):
        self._l, self._c = loader, {}

    def __getitem__(self, f):
        if f not in self._c:
            self._c[f] = self._l(f)
        return self._c[f]


class VecStore:
    """fp16 rows of one embeddings directory (manifest.json, shard_%05d.npy), memory-mapped."""

    def __init__(self, d):
        self.dir = Path(d)
        mf = rjson(self.dir / "manifest.json")
        self.n_rows, self.shard_size, self.n_shards = int(mf["n_rows"]), int(mf["shard_size"]), int(mf["n_shards"])
        self._mm = {}

    def _shard(self, k):
        if k not in self._mm:
            self._mm[k] = np.load(self.dir / f"shard_{k:05d}.npy", mmap_mode="r")
        return self._mm[k]

    def rows(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_rows):
            raise IndexError(f"row out of range for {self.dir}")
        sh, off = idx // self.shard_size, idx % self.shard_size
        out = None
        for k in np.unique(sh):
            m = np.flatnonzero(sh == k)
            part = np.asarray(self._shard(int(k))[off[m]])
            if out is None:
                out = np.empty((idx.size, part.shape[1]), dtype=np.float16)
            out[m] = part
        return out if out is not None else np.empty((0, 0), dtype=np.float16)


class Source:
    pass


_OP = {}
HOST = False                                               # --host: the lab host's package mirror (zpkoh.apply_mirror), as every host job opens it


def opened():
    if "op" not in _OP:
        if HOST:
            import zpkoh
            zpkoh.apply_mirror(log)                        # S6.host_mode's checks and the in-memory package_root; no pool code reads it
        import step4_pool_reach as S4
        op = S4.Opened()
        op.S4 = S4
        if tuple(int(k) for k in op.cfg_h["retrieval_pools"]["ks"]) != KS:
            stop("the headroom config's ks differ from the declared (1, 5, 20)")
        got = [dict(s) for s in op.cfg_h["graph_regimes"]["expansion_settings"]]
        if got != [SETTINGS[k] for k in FILED_SETTINGS]:
            stop("the filed expansion settings differ from the declared ones")
        if int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"]) != RRF_CONST:
            stop("the filed RRF constant differs from the declared one")
        _OP["op"] = op
    return _OP["op"]


def load_family(op, ds, name, fam):
    own = CSR_DIR / f"{name}_{fam}.npz"
    for p in (own, M3B_CSR / f"{name}_{fam}.npz"):
        if p.is_file():
            with np.load(p) as z:
                return CSR(z["indptr"], z["col"])
    c = op.m3a.FamilyCSR(ds.graph(fam), fam)
    CSR_DIR.mkdir(parents=True, exist_ok=True)
    tmp = own.with_name(own.stem + ".tmp.npz")
    np.savez(tmp, indptr=c.indptr, col=c.col)
    os.replace(tmp, own)
    return CSR(c.indptr, c.col)


def _read_ids(path):
    with np.load(path) as z:
        return z["ids"]


def load_pop(op, ds, name, carve):
    """Questions of a carve: ids, cache rows, resolved labelled nodes. filed = M3A's eval population (zero-label questions excluded),
    eval = step 1's s1eval inside it, dev = step 1's s1sel (WebQSP: the non-holdout half of train)."""
    f = PL2 / name / f"pop_{carve}.npz"
    if f.is_file():
        with np.load(f) as z:
            ids = [str(x) for x in z["ids"]]
            idx, flat, off, dg = z["idx"], z["flat"], z["off"], str(z["digest"])
        return {"ids": ids, "idx": idx, "golds": [flat[off[i]:off[i + 1]] for i in range(len(ids))], "digest": dg}
    M3A, S4, cfg_h = op.m3a, op.S4, op.cfg_h
    positions = M3A.node_position_map(ds)
    row_of = {q: j for j, q in enumerate(ds.query_ids)}
    if carve in ("filed", "eval"):
        split = cfg_h["populations"]["eval_splits"][name]
        idx, rows = M3A.population_rows(ds, split, cfg_h)
        golds = M3A.resolve_gold(rows, positions, name)
        ids = [r["query_id"] for r in rows]
        keep = [i for i, g in enumerate(golds) if g.size]
        ids, idx, golds = [ids[i] for i in keep], idx[keep], [golds[i] for i in keep]
        if carve == "eval":
            ent = op.carves["per_dataset"][name]["eval"]
            if S4.digest(ids) != ent["population_sha256"]:
                stop(f"{name}: the filed population is not the one carves.json read")
            at = {q: i for i, q in enumerate(ids)}
            sel = [at[q] for q in ent["ids"]]
            if S4.digest(list(ent["ids"])) != ent["sha256"]:
                stop(f"{name}: the s1eval ids do not hash to carves.json's sha256")
            ids, idx, golds = list(ent["ids"]), idx[sel], [golds[i] for i in sel]
    elif carve == "dev":
        if name == "webqsp":
            ids = sorted(r["query_id"] for r in ds.queries("train"))[1::2]
        else:
            ent = op.carves["per_dataset"][name]["dselect"]
            ids = list(ent["ids"])
            if S4.digest(ids) != ent["sha256"]:
                stop(f"{name}: the s1sel ids do not hash to carves.json's sha256")
        want = set(ids)
        by = {r["query_id"]: r for r in ds.queries("train") if r["query_id"] in want}
        golds = M3A.resolve_gold([by[q] for q in ids], positions, name)
        idx = np.asarray([row_of[q] for q in ids], dtype=np.int64)
        keep = [i for i, g in enumerate(golds) if g.size]
        if name != "webqsp" and len(keep) != len(ids):
            stop(f"{name}: a zero-label question in s1sel")
        ids, idx, golds = [ids[i] for i in keep], idx[keep], [golds[i] for i in keep]
    else:
        stop(f"unknown carve {carve}")
    del positions
    gc.collect()
    flat = np.concatenate(golds).astype(np.int64)
    off = np.r_[0, np.cumsum([g.size for g in golds])].astype(np.int64)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")                  # parts of one grid may build this at the same time
    np.savez(tmp, ids=np.asarray(ids, dtype=str), idx=idx, flat=flat, off=off, digest=np.asarray(digest_ids(ids)))
    os.replace(tmp, f)
    return {"ids": ids, "idx": idx, "golds": golds, "digest": digest_ids(ids)}


def finish_source(src, pop, dense, splade, fams, qdir, ddir):
    src.qids, src.idx, src.golds, src.pop_digest = pop["ids"], pop["idx"], pop["golds"], pop["digest"]
    src.gold_counts = np.asarray([g.size for g in src.golds], dtype=np.int64)
    src.dense, src.splade, src.fams = dense, splade, fams
    src._qs, src._ds, src._qdir, src._ddir = None, None, qdir, ddir
    src.nq = len(src.qids)
    return src


def _qvec(self, c0, c1):
    if self._qs is None:
        self._qs = VecStore(self._qdir)
    return self._qs.rows(self.idx[c0:c1])


def _dvec(self, nodes):
    if self._ds is None:
        self._ds = VecStore(self._ddir)
    return self._ds.rows(nodes)


Source.qvec = _qvec
Source.dvec = _dvec


def load_source(name, carve, first):
    op = opened()
    src = Source()
    src.name, src.carve, src.first = name, carve, first
    if name == MHR:
        return load_mhr(op, src, carve, first)
    ds = op.canonical.Dataset(name, root=str(op.served))
    src.n = int(ds.n_nodes)
    pop = load_pop(op, ds, name, carve)
    rc = Path(ds.dir) / "retrieval_cache"
    splade = _read_ids(rc / "splade_top1000.npz")[pop["idx"]].astype(np.int32)
    if first == "frozen":
        dense = _read_ids(rc / "dense_top1000.npz")[pop["idx"]].astype(np.int32)
    elif first == "e3v":
        p = E0_DIR / f"{name}.npz"
        if carve != "eval" or not p.is_file():
            stop(f"e3v lists exist for the eval carve of the six only ({rel(p)})")
        with np.load(p) as z:
            if not np.array_equal(z["rows"].astype(np.int64), pop["idx"]) or [str(x) for x in z["ids"]] != pop["ids"]:
                stop(f"{name}: E0's e3v rows are not the eval carve's")
            dense = z["top_ids"].astype(np.int32)
    else:
        stop(f"unknown first stage {first}")
    fams = Fams(lambda f: load_family(op, ds, name, f))
    vroot = Path(op.served) / name / "embeddings" / "dense"
    e3 = ROOT / "outputs" / "enc" / "e3v" / name
    qdir, ddir = (vroot / "queries", vroot / "docs") if first == "frozen" else (e3 / "queries", e3 / "docs")
    src.graph_kind = "package"
    return finish_source(src, pop, dense, splade, fams, qdir, ddir)


def load_mhr(op, src, carve, first):
    import zpl1h as H
    P = H.package()
    src.n = int(P["n"])
    pops = dict(H.populations(op, P))
    pp = pops["s1sel" if carve in ("dev", "filed") else "s1eval"]
    pop = {"ids": list(pp.ids), "idx": np.asarray(pp.idx, dtype=np.int64), "golds": [np.asarray(g, dtype=np.int64) for g in pp.golds],
           "digest": pp.digest}
    rc = P["root"] / MHR / "retrieval_cache"
    splade = _read_ids(rc / "splade_top1000.npz")[pop["idx"]].astype(np.int32)
    if first == "frozen":
        dense = _read_ids(rc / "dense_top1000.npz")[pop["idx"]].astype(np.int32)
    else:
        _rec, (e3v_ids, _sc), _g = H.read_first()
        dense = e3v_ids[pop["idx"]].astype(np.int32)
    tab, lay = H.read_layout()
    fam = H.layout_families(op, tab, src.n)
    fams = {f: CSR(fam[f].indptr, fam[f].col) for f in fam}
    vroot = P["root"] / MHR / "embeddings" / "dense"
    e3 = ROOT / "outputs" / "enc" / "e3v" / MHR
    qdir, ddir = (vroot / "queries", vroot / "docs") if first == "frozen" else (e3 / "queries", e3 / "docs")
    src.graph_kind = "layout:B0"
    src.layout_sha256 = lay["layout_npz_sha256"]
    return finish_source(src, pop, dense, splade, fams, qdir, ddir)


# ── the engine ───────────────────────────────────────────────────────────────

def iter_states(op, src, plan, lo, hi, chunk=64):
    """Yield (question row, State) for rows lo..hi: no labels are read here (a source whose label arrays are None runs the same)."""
    M3A = op.m3a
    fams = src.fams
    dvec = src.dvec if plan.need_vec else None
    for c0 in range(lo, hi, chunk):
        c1 = min(c0 + chunk, hi)
        d = np.asarray(src.dense[c0:c1], dtype=np.int64)
        s = np.asarray(src.splade[c0:c1], dtype=np.int64)
        seeds = M3A.stable_union_rows(d[:, :5], s[:, :5])
        none = [None] * (c1 - c0)
        f2 = M3A.rrf_fused_rows(d[:, :200], s[:, :200], RRF_CONST) if "F2" in plan.lists else none
        f1 = M3A.rrf_fused_rows(d, s, RRF_CONST) if "F1" in plan.lists else none
        qv = src.qvec(c0, c1) if plan.need_vec else none
        for j in range(c1 - c0):
            yield c0 + j, pool_state(plan, d[j], s[j], f2[j], f1[j], seeds[j], fams, qv[j], dvec)


def evaluate(op, src, plan, lo, hi, every=300):
    Q = hi - lo
    P = np.zeros((Q, plan.C), dtype=np.int16)
    S = np.zeros((Q, plan.C), dtype=np.int32)
    t0 = last = time.time()
    for q, st in iter_states(op, src, plan, lo, hi):
        grade(plan, st, src.golds[q], P[q - lo], S[q - lo])
        if (q - lo + 1) % every == 0 and time.time() - last > 20:
            last = time.time()
            log(f"{src.name}/{src.carve}/{src.first}: {q - lo + 1}/{Q} questions, {last - t0:.0f} s ({peak_gb()} GB)")
    return P, S


def size_stats_arr(sizes):
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.size == 0:
        return {"candidates_mean": 0.0, "candidates_p50": 0.0, "candidates_p95": 0.0, "candidates_max": 0}
    return {"candidates_mean": float(sizes.mean()), "candidates_p50": float(np.percentile(sizes, 50)),
            "candidates_p95": float(np.percentile(sizes, 95)), "candidates_max": int(sizes.max())}


def cell_row(present, gold_counts, sizes, **labels):
    """m3a_headroom.cell's row, from per-question counts."""
    present = np.asarray(present, dtype=np.int64)
    gold_counts = np.asarray(gold_counts, dtype=np.int64)
    metrics = dict(headroom_metrics(present, gold_counts, ks=KS))
    metrics.update(full_coverage_ceiling(present, gold_counts, ks=KS))
    perfect = metrics.get("recall_ceiling_perfect_retrieval@5")
    if perfect:
        metrics["fraction_of_attainable@5"] = float(metrics["recall_ceiling@5"] / perfect)
    metrics.update(size_stats_arr(sizes))
    return {**labels, **metrics}


def code_path_sha256():
    """One hash of the pool code and the constants (G-U): the same on every dataset or the report refuses to file."""
    h = hashlib.sha256()
    for fn in (parse, base_of, Plan, Lst, dot64, pool_state, grade, iter_states, evaluate, cell_row, size_stats_arr):
        h.update(inspect.getsource(fn).replace("\r\n", "\n").encode("utf-8"))
    h.update(json.dumps(constants(), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def cell_names_of(spec):
    if spec in (None, "all"):
        return cell_list()
    if spec == "filed":
        return filed_cells()
    return [str(x) for x in rjson(spec)]


def require_cells():
    names = cell_list()
    if len(names) != EXPECT_CELLS or len(set(names)) != EXPECT_CELLS:
        stop(f"the declared cell list has {len(names)} cells ({len(set(names))} distinct), expected {EXPECT_CELLS}")
    if cells_sha256(names) != EXPECT_CELLS_SHA256:
        stop("the cell list is not the declared one (cells_sha256 differs from EXPECT_CELLS_SHA256)")


# ── grid files ───────────────────────────────────────────────────────────────

def grid_path(d, carve, first, tag=""):
    return PL2 / d / f"grid_{carve}_{first}{('_' + tag) if tag else ''}"


def save_grid(base, src, plan, P, S, lo, hi, seconds, part=None):
    p = Path(str(base) + (f".part{part[0]}of{part[1]}" if part else "") + ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez(tmp, names=np.asarray(plan.names, dtype=str), present=P, size=S, qids=np.asarray(src.qids[lo:hi], dtype=str),
             gold_counts=src.gold_counts[lo:hi].astype(np.int32), lo=np.asarray(lo), hi=np.asarray(hi))
    os.replace(tmp, p)
    head = {"stage": "PL2 grid", "declared_in": DOC, "dataset": src.name, "carve": src.carve, "first": src.first, "questions": int(hi - lo),
            "rows": [int(lo), int(hi)], "part": list(part) if part else None, "cells": plan.C, "cells_sha256": cells_sha256(plan.names),
            "graph": getattr(src, "graph_kind", None), "population_digest": src.pop_digest, "code_path_sha256": code_path_sha256(),
            "constants_sha256": hashlib.sha256(json.dumps(constants(), sort_keys=True).encode()).hexdigest(),
            "script_sha256": script_sha256(), "npz": rel(p), "npz_sha256": sha256_file(p), "seconds": round(seconds, 1), "peak_gb": peak_gb(),
            "utc": utc()}
    wjson(Path(str(p)[:-4] + ".json"), head)
    return p, head


def load_grid(d, carve, first, tag=""):
    base = grid_path(d, carve, first, tag)
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


def rows_of_grid(g, dataset, population):
    return [cell_row(g["present"][:, i], g["gold_counts"], g["size"][:, i], **labels_of(nm, dataset, population)) for i, nm in enumerate(g["names"])]


# ── commands: cells, grid ────────────────────────────────────────────────────

def cmd_cells(a):
    names = cell_list()
    fam = {}
    for n in names:
        fam[family_of(n)] = fam.get(family_of(n), 0) + 1
    log(f"{len(names)} cells ({len(set(names))} distinct), sha256 {cells_sha256(names)}, families {fam}")
    log(f"walks per question: {len({(parse(n)[3], parse(n)[4]) for n in names if parse(n)[0] in ('G', 'K')})}")
    log(f"code path sha256 {code_path_sha256()}")
    if a.write:
        wjson(PL2 / "cells.json", {"cells": names, "sha256": cells_sha256(names), "families": fam})
    return 0


def cmd_grid(a):
    t0 = time.time()
    names = cell_names_of(a.cells)
    if a.cells in (None, "all"):
        require_cells()
    plan = Plan(names)
    op = opened()
    src = load_source(a.ds, a.carve, a.first)
    lo, hi = 0, src.nq
    part = None
    if a.part:
        i, k = int(a.part[0]), int(a.part[1])
        lo, hi = i * src.nq // k, (i + 1) * src.nq // k
        part = (i, k)
    log(f"{a.ds}/{a.carve}/{a.first}: {src.nq} questions, rows {lo}..{hi}, {plan.C} cells, graph {src.graph_kind} ({peak_gb()} GB)")
    P, S = evaluate(op, src, plan, lo, hi)
    tag = a.tag or ("" if a.cells in (None, "all") else "sub")
    p, head = save_grid(grid_path(a.ds, a.carve, a.first, tag), src, plan, P, S, lo, hi, time.time() - t0, part)
    log(f"{a.ds}/{a.carve}/{a.first}: filed {rel(p)} in {head['seconds']} s")
    return 0


# ── G-K ──────────────────────────────────────────────────────────────────────

def compare_cells(mine, filed, tol=1e-9):
    key = lambda c: (c["pool"], c["regime"], c["setting"], c["base_pool"])  # noqa: E731
    fm = {key(c): c for c in filed}
    mm = {key(c): c for c in mine}
    miss = [list(k) for k in fm if k not in mm] + [list(k) for k in mm if k not in fm]
    worst, per = 0.0, []
    for k, c in fm.items():
        if k in mm:
            d = max((abs(float(mm[k][x]) - float(c[x])) for x in c if x not in ("pool", "regime", "setting", "base_pool", "source")), default=0.0)
            worst = max(worst, d)
            per.append({"pool": k[0], "candidates_mean": [mm[k]["candidates_mean"], c["candidates_mean"]],
                        "all_gold_at_pool": [mm[k]["all_gold_at_pool"], c["all_gold_at_pool"]], "max_abs_diff": d})
    return {"cells_filed": len(fm), "cells_here": len(mm), "missing_or_extra": miss, "max_abs_diff": worst, "tolerance": tol,
            "verdict": "PASS" if (not miss and len(fm) == len(mm) and worst <= tol) else "FAIL", "per_cell": per}


def cmd_gk(a):
    t0 = time.time()
    op = opened()
    MC = op.m3b_contract
    names = filed_cells()
    plan = Plan(names)
    out = {"stage": "PL2 G-K", "declared_in": DOC, "dataset": a.ds, "script_sha256": script_sha256(), "utc": utc(), "host": HOST}
    if a.ds == MHR:
        arms = {}
        for arm in ("e3v", "frozen"):
            src = load_source(MHR, "dev", arm)
            P, S = evaluate(op, src, plan, 0, src.nq)
            g = {"names": names, "present": P, "size": S, "gold_counts": src.gold_counts}
            mine = [{"pool": r["pool"], "regime": r["regime"], "setting": r["setting"], "base_pool": r["base_pool"], "source": "pl2",
                     **{k: r[k] for k in MC.KEEP if k in r}} for r in rows_of_grid(g, MHR, "eval") if "fraction_of_attainable@5" in r]
            filed = rjson(ROOT / "outputs" / "pl1" / "multihoprag" / "RK" / "s0" / f"knee_{arm}.json")["cells"]
            arms[arm] = {**compare_cells(mine, filed), "questions": src.nq}
            log(f"G-K2 {MHR}/{arm}: {arms[arm]['verdict']}, {arms[arm]['cells_here']}/{arms[arm]['cells_filed']} cells, max |diff| {arms[arm]['max_abs_diff']:.3g}")
        out.update({"gate": "G-K2", "arms": arms, "verdict": "PASS" if all(v["verdict"] == "PASS" for v in arms.values()) else "FAIL"})
        wjson(PL2 / ("gk2_multihoprag_host.json" if HOST else "gk2_multihoprag.json"), out)
        return 0 if out["verdict"] == "PASS" else 3
    src = load_source(a.ds, "filed", "frozen")
    P, S = evaluate(op, src, plan, 0, src.nq)
    g = {"names": names, "present": P, "size": S, "gold_counts": src.gold_counts}
    mine = [{"pool": r["pool"], "regime": r["regime"], "setting": r["setting"], "base_pool": r["base_pool"], "source": "pl2",
             **{k: r[k] for k in MC.KEEP if k in r}} for r in rows_of_grid(g, a.ds, "eval") if "fraction_of_attainable@5" in r]
    filed = MC.committed_cells(a.ds)
    res = compare_cells(mine, filed)
    csr = {}
    ds = op.canonical.Dataset(a.ds, root=str(op.served))
    for f in ("structural", "ner", "knn"):
        c0 = op.m3a.FamilyCSR(ds.graph(f), f)
        c1 = src.fams[f]
        csr[f] = bool(np.array_equal(c0.indptr, c1.indptr) and np.array_equal(c0.col, c1.col))
    res["csr_equals_family_csr"] = csr
    if not all(csr.values()):
        res["verdict"] = "FAIL"
    out.update({"gate": "G-K", "questions": src.nq, "population_digest": src.pop_digest, **res, "seconds": round(time.time() - t0, 1), "peak_gb": peak_gb()})
    wjson(PL2 / f"gk_{a.ds}{'_host' if HOST else ''}.json", out)
    log(f"G-K {a.ds}: {res['verdict']}, {res['cells_here']}/{res['cells_filed']} cells, max |diff| {res['max_abs_diff']:.3g}, csr {csr} ({out['seconds']} s)")
    return 0 if res["verdict"] == "PASS" else 3


def gates_passed(ds):
    p = PL2 / ("gk2_multihoprag.json" if ds == MHR else "gk_musique.json")
    if not p.is_file() or rjson(p)["verdict"] != "PASS":
        stop(f"{p.name} is not PASS: run `zpl2g.py gk --ds {'multihoprag' if ds == MHR else 'musique'}` first")
    if HOST:                                               # a host run also needs the host's own G-K (the filed record is the laptop's)
        h = p.with_name(p.stem + "_host.json")
        if not h.is_file() or rjson(h)["verdict"] != "PASS":
            stop(f"{h.name} is not PASS: run `zpl2g.py gk --ds {'multihoprag' if ds == MHR else 'musique'} --host` first")
    return rjson(p)


# ── select ───────────────────────────────────────────────────────────────────

def frontier(rows):
    best, out = -1.0, []
    for r in sorted(rows, key=lambda r: (r["nodes"], -r["all"], r["name"])):
        if r["all"] > best + 1e-12:
            out.append(r)
            best = r["all"]
    return out


def front_at(fr, n):
    v = [r["all"] for r in fr if r["nodes"] <= n]
    return max(v) if v else None


def pick(rows, floor):
    within = [r for r in rows if r["nodes"] <= BOUND]
    reach = [r for r in within if r["all"] >= floor]
    if reach:
        return min(reach, key=lambda r: (r["nodes"], -r["all"], r["name"])), True
    return min(within, key=lambda r: (-r["all"], r["nodes"], r["name"])), False


def old_cell(op, ds):
    if ds == MHR:
        return "equal_rrf_budget_50"
    c = rjson(ROOT / "outputs" / "m3b" / "contract" / f"{ds}.json")["construction"]
    if c["regime"] == "RETRIEVAL":
        return c["base_pool"]
    return f"{c['base_pool']}+{c['regime']}:{c['setting']['name']}"


def cmd_select(a):
    gates_passed(a.ds)
    g = load_grid(a.ds, "dev", "frozen")
    names = g["names"]
    if cells_sha256(names) != EXPECT_CELLS_SHA256:
        stop("the dev grid is not the declared cell list")
    rows = rows_of_grid(g, a.ds, "dev")
    table = [{"name": n, "family": family_of(n), "nodes": r["candidates_mean"], "all": r["all_gold_at_pool"], "any": r["any_gold_at_pool"],
              "coverage": r["coverage_micro"], "p95": r["candidates_p95"], "max": r["candidates_max"]} for n, r in zip(names, rows)]
    readings = {"all": table, "retrieval": [t for t in table if t["family"] in ("R0", "R1")], "graph": [t for t in table if t["family"] not in ("R0", "R1")]}
    out = {"stage": "PL2 select", "declared_in": DOC, "dataset": a.ds, "carve": "dev", "first": "frozen", "questions": len(g["qids"]), "bound": BOUND,
           "cells_sha256": cells_sha256(names), "code_path_sha256": code_path_sha256(), "utc": utc(), "script_sha256": script_sha256()}
    out["picks"] = {}
    for floor in FLOORS:
        for rd, rs in readings.items():
            c, reached = pick(rs, floor)
            out["picks"][f"R{int(round(floor * 100))}_{rd}"] = {**c, "reached": reached, "floor": floor}
    out["short"] = {f"{int(round(f * 100))}": not out["picks"][f"R{int(round(f * 100))}_all"]["reached"] for f in FLOORS}
    fr = {rd: frontier(rs) for rd, rs in readings.items()}
    out["frontier"] = fr
    grid_n = (25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500)
    out["bought"] = [{"nodes": n, "retrieval": front_at(fr["retrieval"], n), "graph": front_at(fr["graph"], n), "all": front_at(fr["all"], n),
                      "links_bought": (None if front_at(fr["graph"], n) is None or front_at(fr["retrieval"], n) is None
                                       else front_at(fr["graph"], n) - front_at(fr["retrieval"], n))} for n in grid_n]
    old = old_cell(None, a.ds)
    oc = next(t for t in table if t["name"] == old)
    out["old"] = {"name": old, **oc}
    out["table"] = table
    wjson(PL2 / a.ds / "select.json", out)
    for k in ("R95_all", "R95_retrieval", "R95_graph", "R98_all", "R98_retrieval", "R98_graph"):
        c = out["picks"][k]
        log(f"{a.ds} {k}: {c['name']}  ALL {c['all']:.4f} at {c['nodes']:.1f} nodes{'' if c['reached'] else '  (SHORT)'}")
    log(f"{a.ds} OLD {old}: ALL {oc['all']:.4f} at {oc['nodes']:.1f} nodes")
    return 0


# ── confirm and call ─────────────────────────────────────────────────────────

def confirm_names(sel):
    names = []
    for k, c in sel["picks"].items():
        if c["name"] not in names:
            names.append(c["name"])
    if sel["old"]["name"] not in names:
        names.append(sel["old"]["name"])
    return names


def cmd_confirm(a):
    gates_passed(a.ds)
    op = opened()
    sel = rjson(PL2 / a.ds / "select.json")
    names = confirm_names(sel)
    plan = Plan(names)
    for first in a.first:
        t0 = time.time()
        try:
            src = load_source(a.ds, "eval", first)
        except SystemExit as e:
            log(f"{a.ds}/eval/{first}: not read here ({e})")
            continue
        if plan.need_vec:
            try:
                src.dvec(np.asarray([0]))
            except Exception as e:                         # the vectors are not on this machine: say so, read the rest
                log(f"{a.ds}/eval/{first}: the ranking vectors are not here ({e}); the ranked cells are left for the host")
                keep = [n for n in names if parse(n)[0] != "K"]
                plan_use = Plan(keep)
            else:
                plan_use = plan
        else:
            plan_use = plan
        P, S = evaluate(op, src, plan_use, 0, src.nq)
        save_grid(grid_path(a.ds, "eval", first, "confirm"), src, plan_use, P, S, 0, src.nq, time.time() - t0)
        g = {"names": plan_use.names, "present": P, "size": S, "gold_counts": src.gold_counts}
        rows = {r["pool"]: r for r in rows_of_grid(g, a.ds, "eval")}
        res = {"stage": "PL2 confirm", "declared_in": DOC, "dataset": a.ds, "first": first, "questions": src.nq, "utc": utc(), "cells": {}}
        for n in plan_use.names:
            r = rows[n]
            k = int(round(r["all_gold_at_pool"] * src.nq))
            res["cells"][n] = {"all": r["all_gold_at_pool"], "all_ci": wilson(k, src.nq), "any": r["any_gold_at_pool"], "coverage": r["coverage_micro"],
                               "nodes": r["candidates_mean"], "p95": r["candidates_p95"], "max": r["candidates_max"]}
        res["named"] = {}
        for k, c in sel["picks"].items():
            if c["name"] in res["cells"]:
                e = res["cells"][c["name"]]
                res["named"][k] = {"name": c["name"], "dev_all": c["all"], "dev_nodes": c["nodes"], "eval_all": e["all"], "eval_nodes": e["nodes"],
                                   "floor": c["floor"], "confirmed": bool(e["all"] >= c["floor"]), "short_by": max(0.0, c["floor"] - e["all"]),
                                   "dev_to_eval": e["all"] - c["all"]}
        if sel["old"]["name"] in res["cells"]:
            e = res["cells"][sel["old"]["name"]]
            res["old"] = {"name": sel["old"]["name"], "dev_all": sel["old"]["all"], "eval_all": e["all"], "eval_nodes": e["nodes"]}
        wjson(PL2 / a.ds / f"confirm_{first}.json", res)
        for k, v in res["named"].items():
            log(f"{a.ds}/{first} {k}: {v['name']} eval ALL {v['eval_all']:.4f} at {v['eval_nodes']:.1f} nodes "
                f"({'CONFIRMED' if v['confirmed'] else 'SHORT by %.2f pts' % (100 * v['short_by'])}; dev {v['dev_all']:.4f})")
    return 0


def paired(new, old, rng):
    """new, old: (all_gold bool array, sizes array) on the same questions. Paired bootstrap of the ALL difference and the nodes ratio."""
    an, sn = new
    ao, so = old
    q = an.size
    d, r = [], []
    for _ in range(BOOT_N // 100):
        idx = rng.integers(0, q, size=(100, q))
        d.append(an[idx].mean(axis=1) - ao[idx].mean(axis=1))
        r.append(sn[idx].mean(axis=1) / so[idx].mean(axis=1))
    d, r = np.concatenate(d), np.concatenate(r)
    return {"all_diff": float(an.mean() - ao.mean()), "all_diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "nodes_ratio": float(sn.mean() / so.mean()), "nodes_ratio_ci": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}


def cmd_call(a):
    out = {"stage": "PL2 call", "declared_in": DOC, "utc": utc(), "datasets": {}}
    codes = {}
    rng = np.random.default_rng(BOOT_SEED)
    for ds in SEVEN:
        sp, cp = PL2 / ds / "select.json", PL2 / ds / "grid_eval_frozen_confirm.npz"
        if not sp.is_file() or not cp.is_file():
            out["datasets"][ds] = {"status": "NOT_READ", "why": "no select or confirm file here"}
            continue
        sel = rjson(sp)
        codes[ds] = sel["code_path_sha256"]
        g = load_grid(ds, "eval", "frozen", "confirm")
        at = {n: i for i, n in enumerate(g["names"])}
        new_name, old_name = sel["picks"]["R95_all"]["name"], sel["old"]["name"]
        if new_name not in at or old_name not in at:
            out["datasets"][ds] = {"status": "NOT_READ", "why": "a named cell is not in the confirm grid"}
            continue
        gc_ = g["gold_counts"]

        def arrs(n):
            i = at[n]
            return (g["present"][:, i].astype(np.int64) == gc_), g["size"][:, i].astype(np.float64)
        new, old = arrs(new_name), arrs(old_name)
        pr = paired(new, old, rng)
        floor = FLOORS[0]
        nr, orr = bool(new[0].mean() >= floor), bool(old[0].mean() >= floor)
        below = (not nr and orr) or (pr["all_diff_ci"][1] < 0 and pr["nodes_ratio_ci"][0] >= 1)
        above = (nr and not orr) or (pr["nodes_ratio_ci"][1] < 1) or (pr["all_diff_ci"][0] > 0 and pr["nodes_ratio_ci"][1] <= 1)
        status = "BELOW" if below else ("ABOVE" if above else "LEVEL")
        out["datasets"][ds] = {"status": status, "new": new_name, "old": old_name, "new_all": float(new[0].mean()), "old_all": float(old[0].mean()),
                               "new_nodes": float(new[1].mean()), "old_nodes": float(old[1].mean()), "new_reaches": nr, "old_reaches": orr,
                               "within_bound": bool(new[1].mean() <= BOUND), **pr}
    read = {k: v for k, v in out["datasets"].items() if v["status"] != "NOT_READ"}
    out["datasets_read"] = sorted(read)
    out["code_path_equal_everywhere"] = len(set(codes.values())) <= 1
    if len(set(codes.values())) > 1:
        stop(f"G-U: the code-path hash differs between datasets: {codes}")
    n_below = sum(v["status"] == "BELOW" for v in read.values())
    n_above = sum(v["status"] == "ABOVE" for v in read.values())
    out["call"] = ("CARRY_PL2" if (len(read) == len(SEVEN) and n_below == 0 and n_above >= 1 and all(v["within_bound"] for v in read.values()))
                   else ("INCOMPLETE" if len(read) < len(SEVEN) else "NOT_CARRIED"))
    wjson(PL2 / "call.json", out)
    for ds, v in out["datasets"].items():
        log(f"call {ds}: {v['status']}" + ("" if v["status"] == "NOT_READ" else f"  NEW {v['new']} ALL {v['new_all']:.4f} @ {v['new_nodes']:.0f}; OLD {v['old']} ALL {v['old_all']:.4f} @ {v['old_nodes']:.0f}"))
    log(f"call: {out['call']} (BELOW {n_below}, ABOVE {n_above}, read {len(read)}/7)")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def brute_cell(op, name, d, s, g, fams, seeds, qv, dvec):
    """One cell of one question by explicit sets (np.union1d, np.isin, sorted()), for the engine to be held to."""
    M3A = op.m3a
    p = parse(name)
    f2 = M3A.rrf_fused_rows(d[None, :200], s[None, :200], RRF_CONST)[0]
    f1 = M3A.rrf_fused_rows(d[None, :], s[None, :], RRF_CONST)[0]
    lists = {"D": d, "S": s, "F2": f2, "F1": f1}
    if p[0] == "FU":
        pool = np.union1d(d[:200], s[:200])
    elif p[0] == "R":
        pool = lists[p[1]][:p[2]]
    else:
        base = lists[p[1]][:p[2]]
        exp = m3b_pools.expand_hops(seeds, [fams[f] for f in REGIMES[p[3]]], SETTINGS[p[4]])
        if p[0] == "G":
            pool = np.union1d(base, exp)
        else:
            cand = exp[np.isin(exp, base, invert=True)]
            sc = dot64(dvec(cand), qv) if cand.size else np.empty(0)
            order = sorted(range(cand.size), key=lambda i: (-sc[i], int(cand[i])))
            pool = np.union1d(base, cand[order[:p[5]]])
    return int(np.isin(g, pool).sum()), int(np.unique(pool).size)


def selftest_engine(op, ds, n_q, every):
    src = load_source(ds, "dev", "frozen")
    names = cell_list()
    plan = Plan(names)
    hi = min(n_q, src.nq)
    P, S = evaluate(op, src, plan, 0, hi)
    M3A = op.m3a
    bad = []
    rng = np.random.default_rng(7)
    checks = 0
    for q in range(0, hi, max(1, hi // 12)):
        d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
        seeds = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
        qv = src.qvec(q, q + 1)[0] if plan.need_vec else None
        for i in range(0, len(names), every):
            nm = names[i]
            p_, s_ = brute_cell(op, nm, d, s, src.golds[q], src.fams, seeds, qv, src.dvec)
            checks += 1
            if (p_, s_) != (int(P[q, i]), int(S[q, i])):
                bad.append((q, nm, (p_, s_), (int(P[q, i]), int(S[q, i]))))
        for i in rng.choice(len(names), size=20, replace=False):
            nm = names[int(i)]
            p_, s_ = brute_cell(op, nm, d, s, src.golds[q], src.fams, seeds, qv, src.dvec)
            checks += 1
            if (p_, s_) != (int(P[q, i]), int(S[q, i])):
                bad.append((q, nm, (p_, s_), (int(P[q, i]), int(S[q, i]))))
    return {"dataset": ds, "questions": hi, "checks": checks, "mismatches": bad[:5], "n_mismatches": len(bad), "verdict": "PASS" if not bad else "FAIL"}


def gate_gx(op, ds, n_q=200):
    src = load_source(ds, "dev", "frozen")
    M3A = op.m3a
    nq = min(n_q, src.nq)
    nbad = 0
    for rg, fl in REGIMES.items():
        fam_m3a = [src.fams[f] for f in fl]
        for stg in FILED_SETTINGS:
            for q in range(nq):
                d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
                seeds = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
                a_ = m3b_pools.expand_hops(seeds, fam_m3a, SETTINGS[stg])
                b_ = M3A.expand(seeds, fam_m3a, SETTINGS[stg])
                if not np.array_equal(a_, b_):
                    nbad += 1
    # hop 3 against an independent per-node loop
    bad3 = 0
    for q in range(min(25, src.nq)):
        d, s = src.dense[q].astype(np.int64), src.splade[q].astype(np.int64)
        seeds = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
        for rg in ("STRUCT", "FULL"):
            fl = [src.fams[f] for f in REGIMES[rg]]
            st = SETTINGS["h3_c25_v2000"]
            got = m3b_pools.expand_hops(seeds, fl, st)
            ref = ref_walk(seeds, fl, st)
            if not np.array_equal(got, ref):
                bad3 += 1
    return {"dataset": ds, "hops_le2_mismatches": nbad, "hop3_mismatches": bad3, "verdict": "PASS" if nbad == 0 and bad3 == 0 else "FAIL"}


def ref_walk(seeds, fams, setting):
    """The walk as a per-node loop, written from the declaration (docs section 5), for G-X."""
    cap, cap2, limit, hops = setting["per_seed_cap"], setting.get("per_frontier_cap"), setting.get("visited_cap"), setting["hops"]

    def nbrs(v, c):
        out = []
        for f in fams:
            out += [int(x) for x in f.col[f.indptr[v]:f.indptr[v] + min(c, int(f.indptr[v + 1] - f.indptr[v]))]]
        return out
    seen, hop1 = set(), []
    for v in seeds:
        for x in nbrs(int(v), cap):
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
            for x in nbrs(int(v), cap2):
                if x not in visited and x not in seen2:
                    seen2.add(x)
                    fresh.append(x)
        fresh = fresh[:budget]
        if not fresh:
            break
        out += fresh
        visited |= set(fresh)
        frontier = fresh
    return np.asarray(out, dtype=np.int64)


def gate_gf():
    bad = []
    for fn in (parse, base_of, Plan, Lst, Lst.pos, dot64, pool_state, state_hash, iter_states):
        if any("gold" in p.lower() or p in ("g", "labels") for p in inspect.signature(fn).parameters):
            bad.append(f"{fn.__name__}: a label parameter")
        if "gold" in inspect.getsource(fn).lower():
            bad.append(f"{fn.__name__}: the source names a label")
    return {"checked": ["parse", "base_of", "Plan", "Lst", "Lst.pos", "dot64", "pool_state", "state_hash", "iter_states"], "problems": bad,
            "verdict": "PASS" if not bad else "FAIL"}


def cmd_gd(a):
    """Internal: print the hash of the states of a sample (run under different thread counts by selftest)."""
    op = opened()
    src = load_source(a.ds, "dev", "frozen")
    src.golds = None                                            # the states are built with the labels withheld
    plan = Plan(cell_list())
    lo, hi = 0, min(a.n, src.nq)
    h = hashlib.sha256()
    for q, st in iter_states(op, src, plan, lo, hi, chunk=a.chunk):
        h.update(state_hash(plan, st).encode())
    print("GDHASH " + h.hexdigest(), flush=True)
    return 0


def gate_gd(ds, n=40):
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
    op = opened()
    res = {"stage": "PL2 selftest", "declared_in": DOC, "utc": utc(), "script_sha256": script_sha256()}
    names = cell_list()
    res["cells"] = {"n": len(names), "distinct": len(set(names)), "sha256": cells_sha256(names), "expected": EXPECT_CELLS_SHA256,
                    "verdict": "PASS" if len(names) == EXPECT_CELLS == len(set(names)) and cells_sha256(names) == EXPECT_CELLS_SHA256 else "FAIL"}
    res["G-F"] = gate_gf()
    res["engine"], res["G-X"] = [], []
    for ds in a.ds:
        res["engine"].append(selftest_engine(op, ds, a.n, a.every))
        res["G-X"].append(gate_gx(op, ds))
        log(f"{ds}: engine {res['engine'][-1]['verdict']} ({res['engine'][-1]['checks']} checks), G-X {res['G-X'][-1]['verdict']}")
    res["G-D"] = gate_gd(a.ds[0])
    res["verdict"] = "PASS" if all(x["verdict"] == "PASS" for x in [res["cells"], res["G-F"], res["G-D"]] + res["engine"] + res["G-X"]) else "FAIL"
    res["seconds"] = round(time.time() - t0, 1)
    wjson(PL2 / "selftest.json", res)
    log(f"selftest: {res['verdict']} (cells {res['cells']['verdict']}, G-F {res['G-F']['verdict']}, G-D {res['G-D']['verdict']}) in {res['seconds']} s")
    return 0 if res["verdict"] == "PASS" else 3


def main(argv=None):
    global HOST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cells")
    p.add_argument("--write", action="store_true")
    p = sub.add_parser("gk")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p = sub.add_parser("grid")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--carve", required=True, choices=("dev", "eval", "filed"))
    p.add_argument("--first", default="frozen", choices=("frozen", "e3v"))
    p.add_argument("--part", nargs=2, metavar=("I", "K"))
    p.add_argument("--cells", default="all")
    p.add_argument("--tag", default="")
    p = sub.add_parser("select")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p = sub.add_parser("confirm")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--first", nargs="+", default=["frozen", "e3v"], choices=("frozen", "e3v"))
    sub.add_parser("call")
    p = sub.add_parser("selftest")
    p.add_argument("--ds", nargs="+", default=["multihoprag", "squad"], choices=SEVEN)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--every", type=int, default=37)
    p = sub.add_parser("gd")
    p.add_argument("--ds", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--chunk", type=int, default=64)
    for name in ("gk", "grid", "select", "confirm", "selftest", "gd"):
        sub.choices[name].add_argument("--host", action="store_true", help="open the package as the lab host's mirror")
    a = ap.parse_args(argv)
    HOST = bool(getattr(a, "host", False))
    return {"cells": cmd_cells, "gk": cmd_gk, "grid": cmd_grid, "select": cmd_select, "confirm": cmd_confirm, "call": cmd_call,
            "selftest": cmd_selftest, "gd": cmd_gd}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
