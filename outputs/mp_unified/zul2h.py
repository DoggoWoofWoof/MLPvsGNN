"""UL2H (docs/UL2H_HOP_COVERAGE_LEDGER.md): the hop-coverage ledger on the e3v layout, all seven datasets. The seeds of a question are the
dense top five U the SPLADE top five (first occurrence kept, dense first). The graph is a unit's whole undirected graph (a label, B0, B0 plus one
form, B0 plus every form), never a pool and never capped; the walk is a breadth-first search from the seeds to depth 3 (levels 1 and 2 from the seeds,
"within 3" read from the gold's side). Measures: ALL / ANY / recall of the golds within h hops, the missed golds' distance and rescue classes,
the uncapped pool sizes, the graph floors. Nothing trains and no model reads anything; no filed graph, pool, fit or record changes. Gold positions
are the grading only. zul2s, zul2p, zul2c, zul2u, zul2, zul1, zul2m, zap4 and zenc are imported, never edited. Outputs only under
outputs/ul2p/hops/. Arrays (npz) are never written; per-unit checkpoints (ckpt/, json) are working files. UL2H decides nothing: it measures and names.

    python outputs/mp_unified/zul2h.py selftest
    python outputs/mp_unified/zul2h.py ledger --ds D [--host] [--threads 1]
    python outputs/mp_unified/zul2h.py report                                  # H1-H5, the graph floors, the lists the declaration names
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zul2s as SS  # noqa: E402  (imports zul2p -> zul2c -> zul2u -> zul2 -> zmhb -> zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402

P, AP, E, UL, C, L2 = SS.P, SS.AP, SS.E, SS.P.UL, SS.C, SS.L2
DECLARED = "docs/UL2H_HOP_COVERAGE_LEDGER.md"
OUT = ROOT / "outputs" / "ul2p" / "hops"
DS7, FORMS, DECIDE, BESIDE = P.DS7, P.FORMS, P.DECIDE, P.BESIDE
TOP = SS.TOP                                      # 5: the dense list's and SPLADE's top rows
BUDGET3 = 1 << 24                                 # hop-3 size is exact where the expansion gathers at most this many entries
BIG_N = 100_000                                   # above this many nodes, hop-3 size is read on every SAMPLE_EVERY-th question only
SAMPLE_EVERY = 10
GATHER_LIMIT = 1 << 24                            # one gather's memory bound (entries); never changes a result
CSR_CHUNK = 1 << 24
FLOORS = (("G95", 0.95), ("G98", 0.98))
HOPS = (1, 2, 3)
EMPTY = np.zeros(0, np.int64)
CONST = {"seeds": "dense5 U splade5", "budget3": BUDGET3, "big_n": BIG_N, "sample_every": SAMPLE_EVERY, "hops": list(HOPS)}
log = P.log


def stop(msg):
    raise SystemExit(f"[zul2h] STOP: {msg}")


def rel(p):
    return P.rel(p)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the graph: per component, forward and reverse adjacency in CSR (int32 indices) ───────────────────


class Comp:
    """One pair array's undirected adjacency: halves = [(indptr, indices) forward, (indptr, indices) reverse]."""

    def __init__(self, halves, pairs, n):
        self.halves, self.pairs, self.n = halves, int(pairs), n

    def deg(self, nodes):
        nodes = np.asarray(nodes, np.int64)
        return sum((ip[nodes + 1] - ip[nodes]) for ip, _ix in self.halves)


def build_comp(keys, n, chunk=CSR_CHUNK):
    """Keys: sorted unique int64 a*n+b with a < b. Forward CSR by key order; reverse CSR by a chunked counting sort."""
    keys = np.asarray(keys, np.int64)
    m = keys.size
    cnt_a = np.zeros(n, np.int64)
    cnt_b = np.zeros(n, np.int64)
    fx = np.empty(m, np.int32)
    for lo in range(0, m, chunk):
        k = keys[lo:lo + chunk]
        a = k // n
        b = k - a * n
        if not bool((a < b).all()) or not bool((b < n).all()):
            stop("a pair key is not a < b < n")
        cnt_a += np.bincount(a, minlength=n)
        cnt_b += np.bincount(b, minlength=n)
        fx[lo:lo + k.size] = b
    ipf = np.zeros(n + 1, np.int64)
    np.cumsum(cnt_a, out=ipf[1:])
    ipr = np.zeros(n + 1, np.int64)
    np.cumsum(cnt_b, out=ipr[1:])
    rx = np.empty(m, np.int32)
    fill = ipr[:-1].copy()
    for lo in range(0, m, chunk):
        k = keys[lo:lo + chunk]
        a = k // n
        b = k - a * n
        o = np.argsort(b, kind="stable")
        bs, as_ = b[o], a[o]
        first = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
        lens = np.diff(np.r_[first, bs.size])
        rank = np.arange(bs.size) - np.repeat(first, lens)
        pos = np.repeat(fill[bs[first]], lens) + rank
        rx[pos] = as_
        fill[bs[first]] += lens
    return Comp([(ipf, fx), (ipr, rx)], m, n)


def gather_chunks(ip, ix, nodes, limit):
    """The neighbours (with repeats) of nodes in one CSR half, in chunks of at most limit entries (a single node may exceed it)."""
    st = ip[nodes]
    ln = ip[nodes + 1] - st
    cum = np.cumsum(ln)
    if cum.size == 0 or int(cum[-1]) == 0:
        return
    nn = nodes.size
    i = 0
    while i < nn:
        base = int(cum[i - 1]) if i else 0
        j = int(np.searchsorted(cum, base + limit, side="right"))
        j = min(max(j, i + 1), nn)
        tot = int(cum[j - 1]) - base
        if tot:
            loc = cum[i:j] - base - ln[i:j]
            off = np.repeat(st[i:j] - loc, ln[i:j]) + np.arange(tot)
            yield ix[off]
        i = j


class Walker:
    """Breadth-first search to depth 3 over a unit (a list of Comp), with per-node visited flags reset by the touched nodes."""

    def __init__(self, n):
        self.n = n
        self.vis = np.zeros(n, bool)
        self.mark = np.zeros(n, np.int64)

    def halves(self, comps):
        return [h for c in comps for h in c.halves]

    def expand(self, halves, front, limit=GATHER_LIMIT):
        front = np.asarray(front, np.int64)
        if front.size == 0:
            return EMPTY
        out = []
        for ip, ix in halves:
            for nb in gather_chunks(ip, ix, front, limit):
                nb = nb[~self.vis[nb]]
                if nb.size == 0:
                    continue
                ar = np.arange(nb.size)
                self.mark[nb] = ar
                nb = nb[self.mark[nb] == ar]
                self.vis[nb] = True
                out.append(nb.astype(np.int64))
        return np.concatenate(out) if out else EMPTY

    def gold_side(self, halves, golds, r2):
        """Within 3 hops: a gold not within 2 with a neighbour within 2."""
        r3 = r2.copy()
        for i in np.flatnonzero(~r2):
            g = int(golds[i])
            for ip, ix in halves:
                a, b = int(ip[g]), int(ip[g + 1])
                if b > a and bool(self.vis[ix[a:b]].any()):
                    r3[i] = True
                    break
        return r3

    def walk(self, comps, seeds, golds, want3=True, budget=BUDGET3, limit=GATHER_LIMIT, maxh=3):
        """reach (4, len(golds)): gold within 0, 1, 2, 3 hops of the seeds; sizes (4,): nodes within 0, 1, 2 (exact) and 3 (nan when not read)."""
        vis = self.vis
        hv = self.halves(comps)
        seeds = np.asarray(seeds, np.int64)
        golds = np.asarray(golds, np.int64)
        vis[seeds] = True
        touched = [seeds]
        reach = np.zeros((4, golds.size), bool)
        nn = np.full(4, np.nan)
        reach[0] = vis[golds]
        nn[0] = seeds.size
        L1 = self.expand(hv, seeds, limit)
        touched.append(L1)
        reach[1] = vis[golds]
        nn[1] = seeds.size + L1.size
        if maxh >= 2:
            L2_ = self.expand(hv, L1, limit)
            touched.append(L2_)
            reach[2] = vis[golds]
            nn[2] = nn[1] + L2_.size
            reach[3] = self.gold_side(hv, golds, reach[2])
            if want3:
                cost = sum(int((ip[L2_ + 1] - ip[L2_]).sum()) for ip, _ix in hv) if L2_.size else 0
                if cost <= budget:
                    L3 = self.expand(hv, L2_, limit)
                    touched.append(L3)
                    nn[3] = nn[2] + L3.size
        vis[np.concatenate(touched)] = False
        return reach, nn


class WalkerForwardOnly(Walker):                   # a wrong engine: pairs are read in one direction only
    def halves(self, comps):
        return [c.halves[0] for c in comps]


class WalkerNoGoldSide(Walker):                    # a wrong engine: "within 3" is taken as "within 2"
    def gold_side(self, halves, golds, r2):
        return r2.copy()


# ── the first stage: seeds U, gates ──────────────────────────────────────────


def union_seeds(dense5, spl5):
    """Per question: the dense five, then the SPLADE five not already in, in order (5 to 10 distinct rows)."""
    out = []
    for d_, s_ in zip(np.asarray(dense5), np.asarray(spl5)):
        u = [int(x) for x in d_]
        seen = set(u)
        for x in s_:
            x = int(x)
            if x not in seen:
                seen.add(x)
                u.append(x)
        out.append(np.array(u, np.int64))
    return out


def check_union(useeds, dense5, spl5, n, what):
    """G-H1: the SPLADE five are five distinct rows in [0, n); every union is 5 to 10 distinct rows, the dense five first."""
    s = np.asarray(spl5)
    if s.ndim != 2 or s.shape[1] != TOP or not bool((s >= 0).all()) or not bool((s < n).all()):
        stop(f"G-H1: {what}: the SPLADE top {TOP} is not {TOP} rows in [0, {n})")
    ss = np.sort(s, axis=1)
    if not bool((ss[:, 1:] != ss[:, :-1]).all()):
        stop(f"G-H1: {what}: a question's SPLADE five are not distinct")
    d5 = np.asarray(dense5)
    for q, u in enumerate(useeds):
        if not (TOP <= u.size <= 2 * TOP) or np.unique(u).size != u.size or not np.array_equal(u[:TOP], d5[q]) or u.min() < 0 or u.max() >= n:
            stop(f"G-H1: {what}: question {q}'s union is not 5 to 10 distinct rows in [0, {n}) with the dense five first")
    return {"questions": len(useeds), "mean_union": round(float(np.mean([u.size for u in useeds])), 4),
            "min_union": int(min(u.size for u in useeds)), "max_union": int(max(u.size for u in useeds))}


class Pop:
    """One cell's questions with a gold: seeds U, golds flat (gq question, gn node, go offsets), found (a gold in the seeds), counts."""

    def __init__(self, cv, ids, golds, dense5, spl5, n, zero_gold=0, hop=None):
        keep = [q for q in range(len(golds)) if np.asarray(golds[q]).size]
        self.cv, self.n, self.zero_gold = cv, n, int(zero_gold) + len(golds) - len(keep)
        self.ids = [ids[q] for q in keep]
        self.golds = [np.unique(np.asarray(golds[q], np.int64)) for q in keep]
        self.dense5 = np.asarray(dense5, np.int64)[keep]
        self.spl5 = np.asarray(spl5, np.int64)[keep]
        self.useeds = union_seeds(self.dense5, self.spl5)
        self.nq = len(self.golds)
        self.gcnt = np.array([g.size for g in self.golds], float)
        self.go = np.r_[0, np.cumsum(self.gcnt.astype(np.int64))]
        self.gq = np.repeat(np.arange(self.nq), self.gcnt.astype(np.int64))
        self.gn = np.concatenate(self.golds) if self.nq else EMPTY
        self.found = np.concatenate([np.isin(self.golds[q], self.useeds[q]) for q in range(self.nq)]) if self.nq else np.zeros(0, bool)
        self.mcnt = np.bincount(self.gq[~self.found], minlength=self.nq).astype(float)
        self.seeds_mean = float(np.mean([u.size for u in self.useeds])) if self.nq else 0.0
        self.hop = np.zeros(self.nq, np.int64) if hop is None else np.asarray(hop, np.int64)[keep]
        h = hashlib.sha256()
        h.update(np.concatenate(self.useeds).tobytes() + self.gn.tobytes() + str(n).encode())
        self.sha = h.hexdigest()


def make_pops(d, D):
    """The cells' populations: UL2S's own loaders (their gates G-S1 to G-S5 stop the run), then the SPLADE top five re-read from the list they
    open (its argument is taken as the rrf is made), then G-H1."""
    cap = {}
    orig = SS.seeds_rrf

    def grab(dense, splade):
        cap["spl"] = np.asarray(splade)
        return orig(dense, splade)

    SS.seeds_rrf = grab
    try:
        pops0, gs, src = SS.first_mhr(D) if d == "multihoprag" else SS.first_host(d, D)
    finally:
        SS.seeds_rrf = orig
    spl = cap.get("spl")
    if spl is None:
        stop("the SPLADE list was not read")
    out, g1 = [], {}
    if d == "multihoprag":
        with np.load(ROOT / "outputs" / "pl1" / "multihoprag" / "RK" / "s0" / "gold.npz") as z:
            gids = [str(x) for x in z["ids"]]
        at = {x: j for j, x in enumerate(gids)}
        for p in pops0:
            rows = np.array([at[x] for x in p.ids], np.int64)
            out.append(Pop(p.cv, p.ids, p.golds, p.seeds["dense"], spl[rows][:, :TOP], D.n, p.zero_gold, p.hop))
    else:
        p = pops0[0]
        if spl.shape[0] != len(p.ids):
            stop(f"{d}: the SPLADE rows ({spl.shape[0]}) are not the carve's questions ({len(p.ids)})")
        out.append(Pop(p.cv, p.ids, p.golds, p.seeds["dense"], spl[:, :TOP], D.n, p.zero_gold, p.hop))
    for q in out:
        g1[q.cv] = check_union(q.useeds, q.dense5, q.spl5, D.n, f"{d}/{q.cv}")
        if q.nq and (int(q.gn.min()) < 0 or int(q.gn.max()) >= D.n):
            stop(f"G-H1: {d}/{q.cv}: a gold is not a row in [0, {D.n})")
    gs = {**gs, "G-H1": g1}
    return out, gs, src


# ── the units ────────────────────────────────────────────────────────────────


def unit_plan(v):
    """(unit, component names) in processing order: the labels, any, B0 plus each form, all."""
    srk = v == "SRK"
    base = (["S"] if srk else []) + ["W", "T0", "K"]
    plan = []
    if srk:
        plan.append(("label:S", ["S"]))
    plan += [("label:R.mention", ["W"]), ("label:R.name_comention", ["CN"]), ("label:R.title_comention", ["TI"]), ("label:K.knn", ["K"]),
             ("any", base)]
    plan += [(f"plus:{f}", base + [f"F:{f}"]) for f in FORMS]
    plan.append(("all", base + [f"F:{f}" for f in FORMS]))
    return plan


def label_units(v):
    return [u for u, _c in unit_plan(v) if u.startswith("label:")]


class RealProv:
    """The components' sorted key arrays from one loaded dataset (zul2p's loaders): S (SRK), W = R.mention, CN = R.name_comention, T0 = F0T,
    TI = R.title_comention (F0T minus CN), K = K.knn, F:f = form f beyond B0 (beyond S too in SRK), as zul2p.form_units selects it."""

    def __init__(self, D):
        self.D = D

    def keys(self, v, name):
        F, z = self.D.F, self.D.z
        if name == "S":
            return F["S"]
        if name == "W":
            return F["W2_S"]
        if name == "CN":
            return F["CN_all"]
        if name == "T0":
            return F["F0T"]
        if name == "K":
            return F["Ke"]
        if name == "TI":
            return np.setdiff1d(F["F0T"], F["CN_all"], assume_unique=True)
        r = z[f"rank__{name[2:]}"]
        sel = r >= 0
        if v == "SRK" and z["inS"].size:
            sel &= ~z["inS"]
        return z["allF"][sel]

    def check(self, v):
        """G-H4: R.name_comention is inside F0T, so that every label's graph is inside the any-link graph."""
        F = self.D.F
        bad = int((~UL.member(F["CN_all"], F["F0T"])).sum()) if F["CN_all"].size else 0
        return {"name_comention_outside_F0T": bad}


class CompReg:
    def __init__(self, prov, v, n):
        self.prov, self.v, self.n, self.c = prov, v, n, {}

    def get(self, name):
        if name not in self.c:
            t0 = time.time()
            self.c[name] = build_comp(self.prov.keys(self.v, name), self.n)
            log(f"graph {self.v}/{name}: {self.c[name].pairs:,} pairs ({time.time() - t0:.0f} s)")
        return self.c[name]

    def drop(self, *names):
        for x in names:
            self.c.pop(x, None)


# ── one unit over one cell, and its checkpoint ───────────────────────────────


def run_unit(pop, comps, W, big, with_deg=False):
    reach = np.zeros((4, pop.gn.size), bool)
    nodes = np.full((4, pop.nq), np.nan)
    t0, t1 = time.time(), time.time()
    for q in range(pop.nq):
        g0, g1 = pop.go[q], pop.go[q + 1]
        want3 = (not big) or q % SAMPLE_EVERY == 0
        r, nn = W.walk(comps, pop.useeds[q], pop.gn[g0:g1], want3)
        reach[:, g0:g1] = r
        nodes[:, q] = nn
        if time.time() - t1 > 120:
            t1 = time.time()
            log(f"  {pop.cv}: {q + 1}/{pop.nq} questions ({t1 - t0:.0f} s)")
    out = {"reach": reach, "nodes": nodes, "pairs": int(sum(c.pairs for c in comps))}
    if with_deg:
        out["deg0"] = sum(c.deg(pop.gn) for c in comps).astype(np.int64)
    return out


def ckpt_path(d, v, unit, cv, out=None):
    return (out or OUT) / d / "ckpt" / f"{v}__{unit.replace(':', '_')}__{cv}.json"


def save_ckpt(path, pop, n, res):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"hdr": {"script": script_sha256(), "pop": pop.sha, "n": n, "const": CONST}, "pairs": int(res["pairs"]),
           "reach": res["reach"].astype(np.int8).tolist(),
           "nodes": [[None if np.isnan(x) else float(x) for x in row] for row in res["nodes"]]}
    if "deg0" in res:
        obj["deg0"] = res["deg0"].tolist()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def load_ckpt(path, pop, n, with_deg):
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if obj.get("hdr") != {"script": script_sha256(), "pop": pop.sha, "n": n, "const": CONST} or (with_deg and "deg0" not in obj):
        return None
    res = {"pairs": int(obj["pairs"]), "reach": np.array(obj["reach"], bool).reshape(4, pop.gn.size),
           "nodes": np.array([[np.nan if x is None else x for x in row] for row in obj["nodes"]], float).reshape(4, pop.nq)}
    if "deg0" in obj:
        res["deg0"] = np.array(obj["deg0"], np.int64)
    return res


# ── the measures ─────────────────────────────────────────────────────────────


def boot_ratio(num, den):
    return AP.boot(lambda s: AP.ratio(s["a"], s["b"]), {"a": num, "b": den})


def hop_measures(pop, r, x, x_prev):
    """One unit at one hop: ALL, ANY, recall (golds within h hops over golds), missed reach, and the size (nodes: mean over the questions
    read; nodes_lb: unread questions filled by the previous hop's size; bound: some question unread)."""
    one = np.ones(pop.nq)
    cr = np.bincount(pop.gq, weights=r, minlength=pop.nq)
    m = ~pop.found
    mr = np.bincount(pop.gq[m], weights=r[m], minlength=pop.nq)
    meas = ~np.isnan(x)
    return {"ALL": boot_ratio((cr == pop.gcnt).astype(float), one), "ANY": boot_ratio((cr > 0).astype(float), one),
            "recall": boot_ratio(cr, pop.gcnt), "missed_reach": boot_ratio(mr, pop.mcnt) if pop.mcnt.sum() else None,
            "reached_missed": int(mr.sum()), "measured_share": round(float(meas.mean()), 5),
            "nodes": round(float(x[meas].mean()), 3) if meas.any() else None,
            "nodes_lb": round(float(np.where(meas, x, x_prev).mean()), 3), "bound": bool(not meas.all())}


def unit_record(pop, res):
    """One unit's record; pairs = the components' pair counts added (a pair in two components is counted twice)."""
    nodes = res["nodes"]
    rec = {"pairs": int(res["pairs"])}
    for h in HOPS:
        rec[f"h{h}"] = hop_measures(pop, res["reach"][h], nodes[h], nodes[h - 1])
    s = float(nodes[0].mean())
    rec["price_per_seed"] = round((float(nodes[1].mean()) - s) / s, 4) if s else None
    return rec


def seeds_record(pop):
    r = pop.found
    one = np.ones(pop.nq)
    cr = np.bincount(pop.gq, weights=r, minlength=pop.nq)
    return {"questions": pop.nq, "golds": int(pop.gn.size), "found_golds": int(pop.found.sum()), "missed_golds": int((~pop.found).sum()),
            "questions_with_missed": int((pop.mcnt > 0).sum()), "seeds_mean": round(pop.seeds_mean, 4),
            "ALL": boot_ratio((cr == pop.gcnt).astype(float), one), "ANY": boot_ratio((cr > 0).astype(float), one),
            "recall": boot_ratio(cr, pop.gcnt)}


def share_of(pop, mask):
    """A set of missed golds: its count and its share of the missed golds (ratio of sums over the questions that have a missed gold)."""
    qa = pop.mcnt > 0
    col = np.bincount(pop.gq[mask], minlength=pop.nq).astype(float)
    return {"count": int(mask.sum()), "share": boot_ratio(col[qa], pop.mcnt[qa]) if qa.any() else None}


def classes_record(pop, R, labels, deg0):
    """The missed golds' distance in the any-link graph, and per hop the label attribution (reached) and the rescue classes (not reached)."""
    M = ~pop.found
    ra = R["any"]
    out = {"missed": int(M.sum()), "questions_with_missed": int((pop.mcnt > 0).sum()),
           "dist_any": {"1": share_of(pop, M & ra[1]), "2": share_of(pop, M & ra[2] & ~ra[1]), "3": share_of(pop, M & ra[3] & ~ra[2]),
                        "4": share_of(pop, M & ~ra[3])},
           "isolated_in_all": share_of(pop, M & (deg0 == 0))}
    for h in HOPS:
        c = {}
        A = M & ra[h]
        alone = sum(R[u][h].astype(int) for u in labels)
        c["any"] = share_of(pop, A)
        for u in labels:
            c[u] = share_of(pop, M & R[u][h])
            c[f"ONLY:{u}"] = share_of(pop, A & R[u][h] & (alone == 1))
        c["MULTI_ALONE"] = share_of(pop, A & (alone >= 2))
        c["MIXED"] = share_of(pop, A & (alone == 0))
        Uu = M & ~ra[h]
        pl = sum(R[f"plus:{f}"][h].astype(int) for f in FORMS)
        c["unreached"] = share_of(pop, Uu)
        for f in FORMS:
            c[f"rescue:{f}"] = share_of(pop, Uu & R[f"plus:{f}"][h])
            c[f"ONLY_F:{f}"] = share_of(pop, Uu & R[f"plus:{f}"][h] & (pl == 1))
        c["rescue_all"] = share_of(pop, Uu & R["all"][h])
        c["ALL_ONLY"] = share_of(pop, Uu & R["all"][h] & (pl == 0))
        none = Uu & ~R["all"][h]
        c["NONE"] = share_of(pop, none)
        c["ISOLATED"] = share_of(pop, none & (deg0 == 0))
        c["FAR"] = share_of(pop, none & (deg0 > 0))
        out[f"h{h}"] = c
    return out


def floors_record(units):
    """G95 / G98: the cell (unit, hop 1-3) with the smallest nodes (nodes_lb at hop 3) among those whose ALL point reaches the floor, else
    SHORT_GRAPH; the best-ALL cell is beside."""
    cells = []
    for u, rec in units.items():
        for h in HOPS:
            m = rec[f"h{h}"]
            if m["ALL"] is not None:
                cells.append((u, h, m["ALL"]["pt"], m["nodes_lb"], m["bound"]))
    out = {}
    for tag, thr in FLOORS:
        ok = [c for c in cells if c[2] >= thr]
        if ok:
            c = min(ok, key=lambda x: (x[3], -x[2], x[0], x[1]))
            out[tag] = {"unit": c[0], "hop": c[1], "ALL": c[2], "nodes": c[3], "bound": c[4]}
        else:
            out[tag] = "SHORT_GRAPH"
    if cells:
        c = max(cells, key=lambda x: (x[2], -x[3], x[0], -x[1]))
        out["best_ALL"] = {"unit": c[0], "hop": c[1], "ALL": c[2], "nodes": c[3], "bound": c[4]}
    return out


def nodes_vs_any(pop, res, any_res):
    """Mean difference in nodes against the any-link unit per hop, over the questions both read."""
    out = {}
    for h in HOPS:
        d = res["nodes"][h] - any_res["nodes"][h]
        d = d[~np.isnan(d)]
        out[f"h{h}"] = {"mean_diff": round(float(d.mean()), 3) if d.size else None, "questions": int(d.size)}
    return out


def check_monotone(pop, R, N, v):
    """G-H3: along the subset order (a label inside any, any inside each plus, each plus inside all): no gold within h hops of a subset's graph
    and not of its superset's, and no exact size larger."""
    labels = label_units(v)
    pairs = [(u, "any") for u in labels] + [("any", f"plus:{f}") for f in FORMS] + [(f"plus:{f}", "all") for f in FORMS]
    bad = []
    for a, b in pairs:
        if bool((R[a] & ~R[b]).any()):
            bad.append(f"reach {a}>{b}")
        both = ~np.isnan(N[a]) & ~np.isnan(N[b])
        if bool((N[a][both] > N[b][both]).any()):
            bad.append(f"nodes {a}>{b}")
    return {"pairs_checked": len(pairs), "violations": bad}


# ── the dense-only one-hop check against UL2S (G-H2) ─────────────────────────


def one_hop_counts(pop, comps, W):
    """Missed golds and those within one hop of the dense top five alone in one unit (UL2S's missed and reached)."""
    miss = reached = 0
    for q in range(pop.nq):
        g = pop.golds[q]
        s = pop.dense5[q]
        r, _nn = W.walk(comps, s, g, False, maxh=1)
        miss += int((~r[0]).sum())
        reached += int((r[1] & ~r[0]).sum())
    return miss, reached


def ul2s_dense_ref(d, v, cv):
    p = SS.OUT / d / "ledger.json"
    if not p.is_file():
        return None
    try:
        u = json.loads(p.read_text(encoding="utf-8"))["versions"][v]["cells"][cv]["lists"]["dense"]["kinds"]["seed"]["units"]["any"]
        return int(u["missed"]), int(u["reached"])
    except (KeyError, ValueError):
        return None


# ── the ledger ───────────────────────────────────────────────────────────────


def process(d, n, pops, prov, vers, out=None, gh2_ref=ul2s_dense_ref, ckpt=True):
    """Every version, unit and cell: the walk, the records, the gates G-H2 to G-H4. Returns {version: {"b0_checks", "cells"}}."""
    big = n > BIG_N
    W = Walker(n)
    res_all = {}
    for v in vers:
        chk = prov.check(v)
        if any(chk.values()):
            stop(f"G-H4: {d}/{v}: {chk}")
        reg = CompReg(prov, v, n)
        plan = unit_plan(v)
        labels = label_units(v)
        R = {p.cv: {} for p in pops}
        N = {p.cv: {} for p in pops}
        deg0 = {}
        recs = {p.cv: {} for p in pops}
        gh2 = {p.cv: None for p in pops}
        t0 = time.time()
        for unit, names in plan:
            wd = unit == "all"
            got, need = {}, []
            for p in pops:
                r = load_ckpt(ckpt_path(d, v, unit, p.cv, out), p, n, wd) if ckpt else None
                if r is None:
                    need.append(p)
                else:
                    got[p.cv] = r
            if need:
                comps = [reg.get(c) for c in names]
                for p in need:
                    got[p.cv] = run_unit(p, comps, W, big, wd)
                    if ckpt:
                        save_ckpt(ckpt_path(d, v, unit, p.cv, out), p, n, got[p.cv])
                    log(f"{d}/{v}/{unit}/{p.cv}: walked ({time.time() - t0:.0f} s)")
            if unit == "any":
                ref_cache = {}
                for p in pops:
                    ref = gh2_ref(d, v, p.cv)
                    if ref is None:
                        gh2[p.cv] = {"call": "NOT_RUN", "why": "UL2S's ledger for this dataset and cell is not on this host"}
                        continue
                    cp = (ckpt_path(d, v, "gh2", p.cv, out)) if ckpt else None
                    mine = None
                    if cp is not None and cp.is_file():
                        try:
                            o = json.loads(cp.read_text(encoding="utf-8"))
                            if o.get("hdr") == {"script": script_sha256(), "pop": p.sha}:
                                mine = tuple(o["mine"])
                        except (OSError, ValueError):
                            mine = None
                    if mine is None:
                        mine = one_hop_counts(p, [reg.get(c) for c in names], W)
                        if cp is not None:
                            cp.parent.mkdir(parents=True, exist_ok=True)
                            cp.write_text(json.dumps({"hdr": {"script": script_sha256(), "pop": p.sha}, "mine": list(mine)}), encoding="utf-8")
                    gh2[p.cv] = {"call": "PASS" if mine == ref else "FAIL", "mine": {"missed": mine[0], "reached": mine[1]},
                                 "ul2s": {"missed": ref[0], "reached": ref[1]}}
                    if mine != ref:
                        stop(f"G-H2: {d}/{v}/{p.cv}: dense top five, one hop, any-link: missed/reached {mine}, UL2S's {ref}")
            for p in pops:
                R[p.cv][unit], N[p.cv][unit] = got[p.cv]["reach"], got[p.cv]["nodes"]
                if wd:
                    deg0[p.cv] = got[p.cv]["deg0"]
                recs[p.cv][unit] = got[p.cv]
            if unit == "label:K.knn":
                reg.drop("CN", "TI")
        reg.c.clear()
        cells = {}
        for p in pops:
            units = {u: unit_record(p, recs[p.cv][u]) for u, _c in plan}
            for u, _c in plan:
                if u != "any":
                    units[u]["nodes_vs_any"] = nodes_vs_any(p, recs[p.cv][u], recs[p.cv]["any"])
            mono = check_monotone(p, R[p.cv], N[p.cv], v)
            if mono["violations"]:
                stop(f"G-H3: {d}/{v}/{p.cv}: {mono['violations']}")
            cells[p.cv] = {"questions": p.nq, "zero_gold_excluded": p.zero_gold, "seeds": seeds_record(p), "units": units,
                           "classes": classes_record(p, R[p.cv], labels, deg0[p.cv]), "floors": floors_record(units),
                           "gates": {"G-H2": gh2[p.cv], "G-H3": mono}}
        res_all[v] = {"cells": cells}
        log(f"{d}/{v}: {len(plan)} units over {len(pops)} cell(s) ({time.time() - t0:.0f} s)")
    return res_all


def cmd_ledger(a):
    d, t0 = a.ds, time.time()
    D = P.load_mhr() if d == "multihoprag" else P.load_host(d, a.host)
    gt = P.gates(D, D.g)
    pops, gs, src = make_pops(d, D)
    log(f"{d}: gates G-P1..G-P3, G-S1..G-S5 and G-H1 pass; {sum(p.nq for p in pops)} questions, versions {D.vers} ({time.time() - t0:.0f} s)")
    rec = {"dataset": d, "declared_in": DECLARED, "status": "read", "nodes": D.n, "nodes_sha256": D.nsha, "sources": {**D.src, **src},
           "gates": {**gt, **gs}, "constants": CONST,
           "multi_gold": any(g.size > 1 for p in pops for g in p.golds),
           "population": {p.cv: {"questions": p.nq, "zero_gold_excluded": p.zero_gold, "golds": int(p.gn.size)} for p in pops}}
    rec["versions"] = process(d, D.n, pops, RealProv(D), D.vers)
    rec["memory"] = C.peak_gb()
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zul2h_sha256=script_sha256(),
               zul2s_sha256=SS.script_sha256(), zul2p_sha256=P.script_sha256(),
               **{k: v for k, v in C.codes().items() if k in ("zul2c_sha256", "zul2u_sha256", "zul2_sha256")})
    L2.wjson(OUT / d / "ledger.json", rec)
    log(f"{d}: ledger {rel(OUT / d / 'ledger.json')} ({rec['seconds']} s)")
    return 0


# ── the report: H1-H5, the floors, the lists the declaration names ───────────


def load_ledgers():
    out = {}
    for d in DS7:
        p = OUT / d / "ledger.json"
        if p.is_file():
            out[d] = json.loads(p.read_text(encoding="utf-8"))
    return out


def cell_of(led, carve=DECIDE):
    """(version, cell) of a ledger's deciding cell (the first version)."""
    for v, vr in led["versions"].items():
        c = vr["cells"].get(carve)
        if c:
            return v, c
    return None, None


def pt(x):
    return None if not x else x["pt"]


def lo(x):
    return None if not x else x["ci"][0]


def verdict(held, reads, detail):
    return {"call": "NOT_READ" if not reads else ("HELD" if held == reads else "NOT_HELD"), "held": held, "reads": reads, "detail": detail}


def expectations(L):
    E_ = {}
    rows = {d: cell_of(led) for d, led in L.items()}
    rows = {d: vc for d, vc in rows.items() if vc[1]}
    held = reads = 0
    det = []
    for d in ("multihoprag", "squad"):
        if d in rows:
            a = pt(rows[d][1]["units"]["any"]["h3"]["ALL"])
            reads += 1
            held += int(a is not None and a >= 0.95)
            det.append(f"{d}: ALL in any at 3 hops {a}")
    E_["H1"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d in ("multihoprag", "squad"):
        if d in rows:
            g = rows[d][1]["floors"]["G95"]
            reads += 1
            ok = isinstance(g, dict) and g["unit"] not in ("any", "all")
            held += int(ok)
            det.append(f"{d}: G95 {g if not isinstance(g, dict) else (g['unit'], g['hop'], g['nodes'])}")
    E_["H2"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, (v, c) in rows.items():
        lab = {u: r for u, r in c["units"].items() if u.startswith("label:") and r["pairs"] > 0}
        reads += 1
        if lab:
            best = min(lab, key=lambda u: (lab[u]["price_per_seed"], u))
            held += int(best == "label:K.knn")
            det.append(f"{d}: cheapest label at hop 1 {best} ({lab[best]['price_per_seed']})")
        else:
            det.append(f"{d}: no label has pairs")
    E_["H3"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, (v, c) in rows.items():
        k = c["classes"]
        if k["dist_any"]["4"]["count"] > 0:
            reads += 1
            r = k["h3"]["rescue_all"]["count"]
            held += int(r >= 1)
            det.append(f"{d}: {k['dist_any']['4']['count']} beyond 3 hops in any; all reaches {r}")
    E_["H4"] = verdict(held, reads, det)
    if not reads:
        E_["H4"]["call"] = "NOT_APPLICABLE" if rows else "NOT_READ"
    held = reads = 0
    det = []
    if "multihoprag" in rows:
        a = pt(rows["multihoprag"][1]["units"]["any"]["h1"]["ALL"])
        reads, held = 1, int(a is not None and a >= 0.95)
        det.append(f"multihoprag: ALL in any at 1 hop {a}")
    E_["H5"] = verdict(held, reads, det)
    return E_


def follows(L):
    out = {}
    order = ["musique", "metaqa", "webqsp", "2wiki", "hotpotqa", "squad", "multihoprag"]
    for d in order:
        if d not in L:
            continue
        v, c = cell_of(L[d])
        if not c:
            continue
        f = c["floors"]
        k = c["classes"]
        w = {"version": v, "floors": f, "dist_any": {x: (r["count"], pt(r["share"])) for x, r in k["dist_any"].items()}, "hops": {}}
        for h in HOPS:
            ch = k[f"h{h}"]
            resc = []
            for fm in FORMS:
                r = ch[f"rescue:{fm}"]
                nd = c["units"][f"plus:{fm}"]["nodes_vs_any"][f"h{h}"]["mean_diff"]
                resc.append({"form": fm, "count": r["count"], "share": r["share"], "nodes_vs_any": nd, "low_above_0": bool(r["share"] and r["share"]["ci"][0] > 0)})
            lab = [(u, ch[f"ONLY:{u}"]["count"]) for u in label_units(v)]
            w["hops"][f"h{h}"] = {"rescue": resc, "only_label": lab, "top_only_label": max(lab, key=lambda x: (x[1], x[0]))[0] if any(x[1] for x in lab) else None,
                                  "rescue_all": ch["rescue_all"]["count"], "ALL_ONLY": ch["ALL_ONLY"]["count"], "NONE": ch["NONE"]["count"],
                                  "ISOLATED": ch["ISOLATED"]["count"], "FAR": ch["FAR"]["count"], "unreached": ch["unreached"]["count"],
                                  "MULTI_ALONE": ch["MULTI_ALONE"]["count"], "MIXED": ch["MIXED"]["count"]}
        cands = []
        for h in HOPS:
            for r in w["hops"][f"h{h}"]["rescue"]:
                if r["low_above_0"]:
                    cands.append(f"ADD form {r['form']} at {h} hop(s): rescues {r['count']} missed golds ({100 * pt(r['share']):.2f}%) for {r['nodes_vs_any']} more nodes per question")
            t = w["hops"][f"h{h}"]["top_only_label"]
            if t:
                cands.append(f"PRIORITISE label {t} at {h} hop(s): it alone links the most golds the other labels alone do not")
        if f["G95"] == "SHORT_GRAPH":
            cands.append("SHORT_GRAPH: no pool of these units at 3 hops from these seeds reaches ALL 0.95; the seeds and these rules, not the cap, limit coverage; "
                         "a new link family or a deeper seed set is needed (none is named here)")
        w["candidates"] = cands
        out[d] = w
    return out


def ci(x, scale=100.0, nd=2):
    return "-" if not x else f"{x['pt'] * scale:.{nd}f} [{x['ci'][0] * scale:.{nd}f}, {x['ci'][1] * scale:.{nd}f}]"


def render(L, E_, W):
    o = ["# UL2H: the hop-coverage ledger, report", "", f"Declared in `{DECLARED}`. Read: {', '.join(sorted(L))}. Points with 95% intervals over questions, in percent; "
         "nodes are the mean uncapped pool size (hop 3: `lb` = unread questions filled by the 2-hop size; `*` = some question unread).", ""]
    o += ["## Expectations", "", "| id | call | detail |", "|---|---|---|"]
    for k, e in E_.items():
        o.append(f"| {k} | {e['call']} | {'; '.join(e['detail']) if e['detail'] else ''} |")
    for d in W:
        v, c = cell_of(L[d])
        s = c["seeds"]
        o += ["", f"## {d} ({v}, {c['questions']} questions, {s['golds']} golds, {s['missed_golds']} missed by the seeds, mean seeds {s['seeds_mean']})", "",
              f"Seeds only: ALL {ci(s['ALL'])}, ANY {ci(s['ANY'])}, recall {ci(s['recall'])}.", "",
              f"Graph floors: G95 {W[d]['floors']['G95']}; G98 {W[d]['floors']['G98']}; best ALL {W[d]['floors'].get('best_ALL')}.", "",
              "| unit | pairs | hop | ALL | ANY | recall | nodes | price/seed |", "|---|---:|---:|---|---|---|---:|---:|"]
        for u, r in c["units"].items():
            for h in HOPS:
                m = r[f"h{h}"]
                nd = m["nodes_lb"] if h == 3 else m["nodes"]
                o.append(f"| {u} | {r['pairs']:,} | {h} | {ci(m['ALL'])} | {ci(m['ANY'])} | {ci(m['recall'])} | {nd}{'*' if m['bound'] else ''} | "
                         f"{r['price_per_seed'] if h == 1 else ''} |")
        k = c["classes"]
        o += ["", "Missed golds by distance in the any-link graph: " + "; ".join(f"{x}: {r['count']} ({ci(r['share'])})" for x, r in k["dist_any"].items()) +
              f"; isolated in all: {k['isolated_in_all']['count']}.", ""]
        o += ["| hop | reached by any | unreached | rescue all | ALL_ONLY | NONE | ISOLATED | FAR | MULTI_ALONE | MIXED |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for h in HOPS:
            ch = k[f"h{h}"]
            o.append(f"| {h} | {ch['any']['count']} | {ch['unreached']['count']} | {ch['rescue_all']['count']} | {ch['ALL_ONLY']['count']} | {ch['NONE']['count']} | "
                     f"{ch['ISOLATED']['count']} | {ch['FAR']['count']} | {ch['MULTI_ALONE']['count']} | {ch['MIXED']['count']} |")
        o += ["", "| hop | form | rescued golds | share | nodes vs any | low end above 0 |", "|---:|---|---:|---|---:|---|"]
        for h in HOPS:
            for r in W[d]["hops"][f"h{h}"]["rescue"]:
                o.append(f"| {h} | {r['form']} | {r['count']} | {ci(r['share'])} | {r['nodes_vs_any']} | {r['low_above_0']} |")
        o += ["", "Candidates named (not adopted): " + ("; ".join(W[d]["candidates"]) if W[d]["candidates"] else "none")]
    return "\n".join(o) + "\n"


def cmd_report(_a):
    L = load_ledgers()
    if not L:
        stop("no ledger is written")
    E_ = expectations(L)
    W = follows(L)
    rec = {"declared_in": DECLARED, "datasets": sorted(L), "expectations": E_, "follows": W,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "zul2h_sha256": script_sha256()}
    OUT.mkdir(parents=True, exist_ok=True)
    L2.wjson(OUT / "report.json", rec)
    (OUT / "report.md").write_bytes(render(L, E_, W).encode("utf-8"))
    log("report written: " + ", ".join(f"{k} {e['call']}" for k, e in E_.items()))
    return 0


# ── the self-test: set loops on random graphs, wrong engines, hand-made measures, then MultiHop-RAG's real graph ──


class ToyProv:
    """Components for the self-test: {name: sorted unique keys}."""

    def __init__(self, n, comps, forms):
        self.n, self.comps, self.forms = n, comps, forms

    def keys(self, v, name):
        return self.forms[name[2:]] if name.startswith("F:") else self.comps[name] if name != "TI" else np.setdiff1d(self.comps["T0"], self.comps["CN"])

    def check(self, v):
        return {"name_comention_outside_F0T": int((~np.isin(self.comps["CN"], self.comps["T0"])).sum())}


def rand_keys(rng, n, m):
    a = rng.integers(0, n, size=m)
    b = rng.integers(0, n, size=m)
    ok = a != b
    return np.unique(np.minimum(a, b)[ok] * n + np.maximum(a, b)[ok])


def brute(n, keys_list, seeds, golds, budget=BUDGET3, want3=True):
    adj = [set() for _ in range(n)]
    degc = [np.zeros(n, np.int64) for _ in keys_list]
    for i, keys in enumerate(keys_list):
        for k in keys:
            a, b = divmod(int(k), n)
            adj[a].add(b)
            adj[b].add(a)
            degc[i][a] += 1
            degc[i][b] += 1
    dist = {int(s): 0 for s in seeds}
    lv = [set(dist)]
    for h in (1, 2, 3):
        nw = set()
        for x in lv[-1]:
            for y in adj[x]:
                if y not in dist:
                    dist[y] = h
                    nw.add(y)
        lv.append(nw)
    reach = np.array([[int(g) in dist and dist[int(g)] <= h for g in golds] for h in range(4)])
    nodes = np.array([len(set().union(*lv[:h + 1])) for h in range(4)], float)
    cost = sum(int(dc[list(lv[2])].sum()) for dc in degc) if lv[2] else 0
    if not want3 or cost > budget:
        nodes[3] = np.nan
    return reach, nodes


def toy_pop(rng, n, nq, cv="s1eval"):
    golds = [np.sort(rng.choice(n, int(rng.integers(1, 5)), replace=False)) for _ in range(nq)]
    dense, spl = [], []
    for q in range(nq):
        d_ = [int(x) for x in rng.choice(n, TOP, replace=False)]
        s_ = [int(x) for x in rng.choice(n, TOP, replace=False)]
        if q % 3 == 0:                                # a gold among the dense five, and a repeated row between the two lists
            d_ = [int(golds[q][0])] + [x for x in d_ if x != golds[q][0]][:TOP - 1]
            s_ = [d_[2]] + [x for x in s_ if x != d_[2]][:TOP - 1]
        dense.append(d_)
        spl.append(s_)
    dense, spl = np.array(dense), np.array(spl)
    return Pop(cv, [str(i) for i in range(nq)], golds, dense, spl, n)


def selftest():
    rng = np.random.default_rng(7)
    # 1. CSR: forward and reverse adjacency reproduce the pairs, chunk size one or tiny, isolated nodes and an empty component
    for n, m, ch in ((40, 90, 7), (40, 90, 1), (12, 0, 5), (300, 2000, 1 << 24)):
        keys = rand_keys(rng, n, m) if m else EMPTY
        cp = build_comp(keys, n, ch)
        adj = [set() for _ in range(n)]
        for k in keys:
            adj[int(k) // n].add(int(k) % n)
            adj[int(k) % n].add(int(k) // n)
        for x in range(n):
            got = set()
            for ip, ix in cp.halves:
                got |= set(int(y) for y in ix[ip[x]:ip[x + 1]])
            assert got == adj[x], (n, m, ch, x)
        assert cp.pairs == keys.size and int(cp.deg(np.arange(n)).sum()) == 2 * keys.size
    # 2. the walk equals set loops (exact sizes, hop 3 by the gold's side), with tiny gather chunks, a budget that skips some hop-3 sizes, and unions of components
    for t in range(60):
        n = int(rng.integers(20, 90))
        parts = [rand_keys(rng, n, int(rng.integers(0, 3 * n))) for _ in range(int(rng.integers(1, 4)))]
        if t % 3 == 0 and parts[0].size:                        # a hub
            hub = int(rng.integers(0, n))
            parts[0] = np.unique(np.concatenate([parts[0], np.array([min(hub, x) * n + max(hub, x) for x in range(n) if x != hub])]))
        comps = [build_comp(k, n) for k in parts]
        W = Walker(n)
        for q in range(8):
            seeds = rng.choice(n, int(rng.integers(5, 11)), replace=False)
            golds = np.sort(rng.choice(n, int(rng.integers(1, 5)), replace=False))
            budget = BUDGET3 if t % 2 else int(rng.integers(0, 60))
            r, nn = W.walk(comps, seeds, golds, True, budget=budget, limit=int(rng.integers(1, 40)))
            rb, nb = brute(n, parts, seeds, golds, budget=budget)
            assert np.array_equal(r, rb), (t, q)
            assert np.array_equal(np.isnan(nn), np.isnan(nb)) and np.allclose(nn[~np.isnan(nn)], nb[~np.isnan(nb)]), (t, q, nn, nb)
            assert not W.vis.any(), "visited flags are not reset"
            r1, _ = W.walk(comps, seeds, golds, False, maxh=1)
            assert np.array_equal(r1[:2], rb[:2])
    # 3. two wrong engines are caught
    caught = {"forward_only": 0, "no_gold_side": 0}
    n = 60
    parts = [rand_keys(rng, n, 150), rand_keys(rng, n, 70)]
    comps = [build_comp(k, n) for k in parts]
    for q in range(40):
        seeds = rng.choice(n, 6, replace=False)
        golds = np.sort(rng.choice(n, 4, replace=False))
        rb, _ = brute(n, parts, seeds, golds)
        for name, Wx in (("forward_only", WalkerForwardOnly(n)), ("no_gold_side", WalkerNoGoldSide(n))):
            r, _nn = Wx.walk(comps, seeds, golds, False)
            caught[name] += int(not np.array_equal(r, rb))
    assert all(v > 0 for v in caught.values()), caught
    # 4. the measures on a hand-made cell: a path 0-1-2-3-4-5 (label A) and 6-7 (label B); seeds 0 and 6; golds {2}, {3, 7}, {5}, {0, 6}
    n = 10
    ka = np.array([i * n + i + 1 for i in range(5)], np.int64)
    kb = np.array([6 * n + 7], np.int64)
    pop = Pop("s1eval", ["a", "b", "c", "d"], [np.array([2]), np.array([3, 7]), np.array([5]), np.array([0, 6])], np.array([[0, 6, 8, 9, 4]] * 4),
              np.array([[0, 6, 8, 9, 4]] * 4), n)
    pop.useeds = [np.array([0, 6])] * 4                          # the declared seeds of this hand-made case
    pop.dense5 = np.array([[0, 6, 8, 9, 4]] * 4)
    pop.found = np.concatenate([np.isin(pop.golds[q], pop.useeds[q]) for q in range(4)])
    pop.mcnt = np.bincount(pop.gq[~pop.found], minlength=4).astype(float)
    pop.seeds_mean = 2.0
    cmp = [build_comp(ka, n), build_comp(kb, n)]
    r, nn = Walker(n).walk(cmp, pop.useeds[0], np.array([2, 3, 5, 7, 0]), True)
    assert r.tolist() == [[False, False, False, False, True], [False, False, False, True, True], [True, False, False, True, True],
                          [True, True, False, True, True]], r.tolist()
    assert nn.tolist() == [2, 4, 5, 6], nn.tolist()              # seeds 0,6; then 1,7; then 2; then 3
    res = {"reach": np.zeros((4, pop.gn.size), bool), "nodes": np.full((4, 4), np.nan)}
    W_ = Walker(n)
    for q in range(4):
        g0, g1 = pop.go[q], pop.go[q + 1]
        rr_, nq_ = W_.walk(cmp, pop.useeds[q], pop.gn[g0:g1], True)
        res["reach"][:, g0:g1] = rr_
        res["nodes"][:, q] = nq_
    m = hop_measures(pop, res["reach"][2], res["nodes"][2], res["nodes"][1])
    # golds: q0 {2}: dist 2; q1 {3, 7}: dist 3 and 1; q2 {5}: dist 5; q3 {0, 6}: both seeds; hop 2: reached = 1 + 1 + 0 + 2 of 6 golds
    assert m["ALL"]["pt"] == 0.5 and m["ANY"]["pt"] == 0.75 and abs(m["recall"]["pt"] - 4 / 6) < 1e-5 and m["reached_missed"] == 2, m
    m3 = hop_measures(pop, res["reach"][3], res["nodes"][3], res["nodes"][2])
    assert m3["ALL"]["pt"] == 0.75 and m3["bound"] is False and m3["nodes"] == 6.0, m3
    s_ = seeds_record(pop)
    assert s_["found_golds"] == 2 and s_["missed_golds"] == 4 and s_["ALL"]["pt"] == 0.25 and s_["questions_with_missed"] == 3, s_
    # 5. the union keeps the dense five first and the SPLADE rows not already in
    assert [u.tolist() for u in union_seeds([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[5, 6, 1, 7, 8], [9, 10, 11, 12, 13]])] == [[1, 2, 3, 4, 5, 6, 7, 8],
                                                                                                                              [1, 2, 3, 4, 5, 9, 10, 11, 12, 13]]
    # 6. the classes and floors equal their loops on a random cell with the whole plan (toy components, one version RK)
    n = 80
    comps_t = {"W": rand_keys(rng, n, 60), "T0": rand_keys(rng, n, 40), "K": rand_keys(rng, n, 30)}
    comps_t["CN"] = comps_t["T0"][::2].copy()
    forms_t = {f: rand_keys(rng, n, 25 + 5 * i) for i, f in enumerate(FORMS)}
    prov = ToyProv(n, comps_t, forms_t)
    pop = toy_pop(rng, n, 40)
    out = ROOT / "outputs" / "ul2p" / "hops_selftest_tmp"
    try:
        for rep in range(2):                                      # the second pass reads the checkpoints
            ra = process("toy", n, [pop], prov, ["RK"], out=out, gh2_ref=lambda d, v, cv: None, ckpt=True)
            if rep == 0:
                first = json.dumps(ra, sort_keys=True)
        assert first == json.dumps(ra, sort_keys=True), "the checkpoint read differs from the run"
        cell = ra["RK"]["cells"]["s1eval"]
        # recompute every unit with set loops and compare the reach-derived counts
        plan = unit_plan("RK")
        keys_of = {nm: prov.keys("RK", nm) for nm in ("W", "T0", "K", "CN", "TI") + tuple(f"F:{f}" for f in FORMS)}
        Rb = {}
        for u, names in plan:
            Rb[u] = np.zeros((4, pop.gn.size), bool)
            for q in range(pop.nq):
                g0, g1 = pop.go[q], pop.go[q + 1]
                Rb[u][:, g0:g1], _ = brute(n, [keys_of[x] for x in names], pop.useeds[q], pop.gn[g0:g1])
        degb = np.zeros(pop.gn.size, np.int64)
        for nm in ("W", "T0", "K") + tuple(f"F:{f}" for f in FORMS):
            for k in keys_of[nm]:
                degb[pop.gn == int(k) // n] += 1
                degb[pop.gn == int(k) % n] += 1
        cb = classes_record(pop, Rb, label_units("RK"), degb)
        assert json.dumps(cb, sort_keys=True) == json.dumps(cell["classes"], sort_keys=True)
        M = ~pop.found
        for h in HOPS:
            c = cb[f"h{h}"]
            assert c["any"]["count"] + c["unreached"]["count"] == int(M.sum())
            assert c["rescue_all"]["count"] + c["NONE"]["count"] == c["unreached"]["count"]
            assert c["ISOLATED"]["count"] + c["FAR"]["count"] == c["NONE"]["count"]
            assert (sum(c[f"ONLY:{u}"]["count"] for u in label_units("RK")) + c["MULTI_ALONE"]["count"] + c["MIXED"]["count"]) == c["any"]["count"]
        d4 = sum(cb["dist_any"][x]["count"] for x in "1234")
        assert d4 == int(M.sum())
        fl = cell["floors"]
        assert set(fl) >= {"G95", "G98", "best_ALL"}
        # floors: a hand-made units table
        fake = {"u1": {f"h{h}": {"ALL": {"pt": a}, "nodes_lb": s, "bound": False} for h, a, s in ((1, 0.2, 10), (2, 0.96, 50), (3, 0.99, 400))},
                "u2": {f"h{h}": {"ALL": {"pt": a}, "nodes_lb": s, "bound": False} for h, a, s in ((1, 0.1, 5), (2, 0.5, 20), (3, 0.97, 100))}}
        ff = floors_record(fake)
        assert ff["G95"]["unit"] == "u1" and ff["G95"]["hop"] == 2 and ff["G98"]["unit"] == "u1" and ff["G98"]["hop"] == 3 and ff["best_ALL"]["ALL"] == 0.99, ff
        assert floors_record({"u": {f"h{h}": {"ALL": {"pt": 0.5}, "nodes_lb": 1, "bound": False} for h in HOPS}})["G95"] == "SHORT_GRAPH"
        # a wrong subset order is caught by G-H3
        Rw = {u: r.copy() for u, r in Rb.items()}
        Rw["any"][2] = False
        Nn = {u: np.full((4, pop.nq), np.nan) for u in Rb}
        if not check_monotone(pop, Rw, Nn, "RK")["violations"] and Rb["label:R.mention"][2].any():
            raise AssertionError("G-H3 did not catch a label reaching a gold its any-link graph does not")
        assert check_monotone(pop, Rb, Nn, "RK")["violations"] == []
        # the gate G-H2 stops on a mismatch
        try:
            process("toy", n, [pop], prov, ["RK"], out=out, gh2_ref=lambda d, v, cv: (1, 1), ckpt=False)
            raise AssertionError("G-H2 did not stop")
        except SystemExit as e:
            assert "G-H2" in str(e)
        ok = process("toy", n, [pop], prov, ["RK"], out=out, gh2_ref=lambda d, v, cv: one_hop_counts(pop, [build_comp(k, n) for k in (
            keys_of["W"], keys_of["T0"], keys_of["K"])], Walker(n)), ckpt=False)
        assert ok["RK"]["cells"]["s1eval"]["gates"]["G-H2"]["call"] == "PASS"
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    # 7. the real graph: MultiHop-RAG's any-link graph and all-forms graph, every test question, against set loops; the gates pass
    D = P.load_mhr()
    P.gates(D, D.g)
    pops, gs, _src = make_pops("multihoprag", D)
    prov = RealProv(D)
    v = D.vers[0]
    assert prov.check(v) == {"name_comention_outside_F0T": 0}
    W = Walker(D.n)
    p = [x for x in pops if x.cv == DECIDE][0]
    for unit, names in (("any", ["W", "T0", "K"]), ("all", ["W", "T0", "K"] + [f"F:{f}" for f in FORMS])):
        kl = [prov.keys(v, x) for x in names]
        cl = [build_comp(k, D.n) for k in kl]
        adj = [set() for _ in range(D.n)]
        for keys in kl:
            for k in keys:
                a, b = divmod(int(k), D.n)
                adj[a].add(b)
                adj[b].add(a)
        for q in range(0, p.nq, 7):
            g0, g1 = p.go[q], p.go[q + 1]
            r, nn = W.walk(cl, p.useeds[q], p.gn[g0:g1], True)
            dist = {int(s): 0 for s in p.useeds[q]}
            lv = [set(dist)]
            for h in (1, 2, 3):
                nw = set()
                for x in lv[-1]:
                    for y in adj[x]:
                        if y not in dist:
                            dist[y] = h
                            nw.add(y)
                lv.append(nw)
            rb = np.array([[int(g) in dist and dist[int(g)] <= h for g in p.gn[g0:g1]] for h in range(4)])
            assert np.array_equal(r, rb), (unit, q)
            assert nn[:3].tolist() == [len(set().union(*lv[:h + 1])) for h in range(3)]
    sq = p.nq
    assert sq == 1130, sq
    cl = [build_comp(prov.keys(v, x), D.n) for x in ("W", "T0", "K")]
    g2 = []
    for pp in pops:                                   # G-H2 itself: the dense top five alone, one hop, any-link, against UL2S's filed counts
        ref = ul2s_dense_ref("multihoprag", v, pp.cv)
        if ref is not None:
            mine = one_hop_counts(pp, cl, W)
            assert mine == ref, (pp.cv, mine, ref)
            g2.append(pp.cv)
    assert g2, "UL2S's MultiHop-RAG ledger is not here to check G-H2 against"
    log("selftest: the CSR equals the pairs; the walk equals set loops on random graphs (tiny gather chunks, budgets, hubs, unions of components) and on "
        "MultiHop-RAG's real graph (any-link and all-forms); two wrong engines are caught; the measures equal hand-made values; classes, floors and G-H2/G-H3 "
        "equal their loops; checkpoints reproduce the run; gates G-S*, G-H1 and G-H4 pass on MultiHop-RAG")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zul2h.py")
    ap.add_argument("cmd", choices=["selftest", "ledger", "report"])
    ap.add_argument("--ds", choices=DS7)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "ledger" and not a.ds:
        stop("--ds is needed")
    return {"selftest": lambda _a: selftest(), "ledger": cmd_ledger, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
