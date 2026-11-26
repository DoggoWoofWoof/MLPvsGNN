"""GP3 (docs/GP3_NODE_COUNT_RULE.md): the node-count rule. The walk of the base layout's any-link graph (GP1's graph, levels and shortest-path links) in three hops, in which
every hop keeps (a) every S-tail of a kept parent (S whole) and (b) the first N_h other candidates in the order of one of GP2's scores (K1..K4, or a random order as the
null; A1: PP = the PageRank mass of the walk's own layers), N_1 >= N_2 >= N_3 from one ladder; the pool is the first-stage base, the seeds and the kept nodes, counted as DISTINCT NODES. The walker pass stores, per question, per
score and per ladder step, the sizes and the places of the golds, so every triple of the ladder is graded exactly offline.

    python outputs/mp_unified/zgp3.py selftest                                   # the walker equals a brute-force walk on toy graphs (no real data)
    python outputs/mp_unified/zgp3.py time --ds D [--n 20] [--host]              # cost of the walk on the first questions of a part, NO label read
    python outputs/mp_unified/zgp3.py extract --ds D --part q|r [--host] [--threads 1] [--shard I/N] [--limit N]
The program (P1 / P2 per score by exact enumeration, the random-order null, per-cell optima, the final rule, the rest-of-dev grade) is outputs/mp_unified/zgp3s.py, which reads
the chunks this file writes; this file is frozen when the first row is queued.
It imports zgp1, zgp1p, zgp2 (read-only, pinned by sha256 in zgp2) and the UL2H/UL3A engines they import; nothing earlier changes. The walk reads no label: golds enter only
as probes whose place in an already fixed order is read (self-test G3-5 runs the walk with and without them and compares every size).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zgp1 as G1  # noqa: E402  (first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zgp1p as GP  # noqa: E402
import zgp2 as Z  # noqa: E402

H, T3, UL, P, L2 = G1.H, G1.T3, G1.UL, G1.P, G1.L2
DECLARED = "docs/GP3_NODE_COUNT_RULE.md"
OUT = ROOT / "outputs" / "gp3"
EMPTY = np.zeros(0, np.int64)
LADDER = (0, 10, 30, 100, 300, 1000, 3000, 10000, 30000)  # N_h: other candidates kept at hop h (S-tails come on top of it); 9 steps, 165 triples N_1 >= N_2 >= N_3
SCORES = ("K1", "K2", "K3", "K4", "PP", "R0", "R1")       # GP2's scores (text-free ones), A1's PageRank mass PP, and two random orders (the null)
PPR_BITS = 56                                              # A1: the seed mass sums to 2^56 (integers: sums are exact and independent of the order of addition)
PPR_W = 30                                                 # seed weights: 2^30 // (60 + rank) per first-stage list
BASE_B = {"multihoprag": 25, "squad": 15, "hotpotqa": 25, "2wiki": 400, "musique": 400, "metaqa": 25, "webqsp": 400}   # the first-stage base of UL5's family per dataset: rrf1000_b<B>
RRF_CONST = 60
RKCAP = (1 << 19) - 1                                      # source rank clip in the composite key
VTCAP = 255                                                # kept-parent count clip
BIGR = 1 << 30
GATHER_CAP = 12_000_000                                    # raw neighbour entries gathered per hop table; over it the parents are thinned in hash order (CAPPED)
CHUNK_Q = 20                                               # questions per work chunk (a checkpoint)
QUARTER_M = 4
CELLS = GP.CELLS
log = G1.log


def stop(msg):
    raise SystemExit(f"[zgp3] STOP: {msg}")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(p):
    return G1.rel(p)


# ── keys ─────────────────────────────────────────────────────────────────────


def sortable_f32(x):
    """float32 -> int64 in [0, 2^32), monotone in the value (-0.0 read as 0.0)."""
    u = (np.asarray(x, np.float32) + np.float32(0.0)).view(np.uint32).astype(np.int64)
    return np.where(u >> 31, (~u) & 0xFFFFFFFF, u | 0x80000000)


def node_keys(score, u, rk, vt, dtq, cu, qseed):
    """The sort key (larger = earlier) of each candidate node. u = the first-stage rank bin (0 worst .. 6 best, GP2's ur), rk = the smallest source rank over the node's kept
    incoming links, vt = how many kept parents name it, dtq = sortable_f32 of the largest dot over those links. K1 = (u, dot), K2 = (-rank, dot), K3 = (u, -rank, votes, dot), K4 = (dot)."""
    if score == "K4":
        return dtq.astype(np.int64)
    rkq = (RKCAP - np.minimum(rk, RKCAP)).astype(np.int64)
    if score == "K1":
        return (u.astype(np.int64) << 32) | dtq
    if score == "K2":
        return (rkq << 32) | dtq
    if score == "K3":
        return ((((u.astype(np.int64) << 19) | rkq) << 8 | np.minimum(vt, VTCAP).astype(np.int64)) << 32) | dtq
    if score in ("R0", "R1"):
        x = (np.int64(2 if score == "R1" else 1) << 58) | (np.int64(qseed) << 36) | cu.astype(np.int64)
        return (G1.mix64(x) >> np.uint64(2)).astype(np.int64)
    stop(f"unknown score {score}")


def src_rank(a, dot):
    """1 + the number of links of the same source with a strictly larger dot (GP1's rank), per link."""
    if a.size == 0:
        return np.zeros(0, np.int64)
    o = np.lexsort((-dot, a))
    a_s, d_s = a[o], dot[o]
    new_a = np.r_[True, a_s[1:] != a_s[:-1]]
    start = np.flatnonzero(new_a)
    lens = np.diff(np.r_[start, a_s.size])
    first = np.repeat(start, lens)
    newrun = new_a | np.r_[True, d_s[1:] != d_s[:-1]]
    run_start = np.maximum.accumulate(np.where(newrun, np.arange(a_s.size), 0))
    r = np.empty(a.size, np.int64)
    r[o] = run_start - first + 1
    return r


class Table:
    """The links (a -> c) of one hop sorted by the tail c: dot key, source rank, S flag, the source's degree (stored neighbour entries, A1); groups = distinct tails."""

    def __init__(self, a, c, dotq, rank, isS, dg=None):
        o = np.argsort(c, kind="stable")
        self.a, self.c, self.dotq, self.rank, self.S = a[o], c[o], dotq[o], rank[o], isS[o]
        self.dg = (dg if dg is not None else np.ones(a.size, np.int64))[o]
        self._groups()

    def _groups(self):
        self.gs = np.flatnonzero(np.r_[True, self.c[1:] != self.c[:-1]]) if self.c.size else EMPTY
        self.cu = self.c[self.gs]

    def subset(self, rows):
        t = object.__new__(Table)
        t.a, t.c, t.dotq, t.rank, t.S, t.dg = self.a[rows], self.c[rows], self.dotq[rows], self.rank[rows], self.S[rows], self.dg[rows]
        t._groups()
        return t


class Ctx:
    """What the walk needs: the graph and its levels, the pair dot, the S test, the first-stage bins; per question set_question()."""

    def __init__(self, G, W, lev, n, dotf, isS, ladder=LADDER, scores=SCORES, ubin=None, seedw=None):
        self.G, self.W, self.lev, self.n, self.dotf, self.isS = G, W, lev, n, dotf, isS
        self.ladder, self.scores = tuple(ladder), tuple(scores)
        self.posbuf = np.full(n, BIGR, np.int32)
        self.mbuf = np.zeros(n, np.int64)
        self._ubin, self._seedw = ubin, seedw
        self.ld = self.ls = self.base = None

    def seed_mass(self, seeds):
        """A1: the restart mass of each seed (integers summing to about 2^PPR_BITS): proportional to the seed's first-stage weight 2^PPR_W // (60 + rank) in the dense list plus the same in
        the SPLADE list (0 where the seed is not in a list). `seedw` replaces the weights in the self-test."""
        seeds = np.asarray(seeds, np.int64)
        if self._seedw is not None:
            w = np.asarray(self._seedw(seeds), np.int64)
        else:
            dr, sr = Z.rank_in(self.ld, seeds).astype(np.int64), Z.rank_in(self.ls, seeds).astype(np.int64)
            w = np.where(dr > 0, (1 << PPR_W) // (RRF_CONST + np.maximum(dr, 1)), 0) + np.where(sr > 0, (1 << PPR_W) // (RRF_CONST + np.maximum(sr, 1)), 0)
        tot = int(w.sum())
        return ((w << 30) // max(tot, 1)) << (PPR_BITS - 30)

    def set_question(self, ld, ls, base):
        self.ld, self.ls, self.base = ld, ls, np.asarray(base, np.int64)

    def ubin(self, cu):
        if self._ubin is not None:
            return self._ubin(cu)
        dr, sr = Z.rank_in(self.ld, cu), Z.rank_in(self.ls, cu)
        ur = np.where((dr > 0) & (sr > 0), np.minimum(dr, sr), np.where(dr > 0, dr, sr))
        return Z.rank_bin(ur)


def hop_table(ctx, parents, level):
    """The distinct links (a, c) from the parents to the nodes of BFS level `level`, with their dot, source rank and S flag. Over GATHER_CAP raw entries the parents are thinned in hash order."""
    n = ctx.n
    parents = np.unique(np.asarray(parents, np.int64))
    if parents.size == 0:
        return Table(EMPTY, EMPTY, EMPTY, EMPTY, np.zeros(0, bool)), False
    s, x, thin = ctx.G.gather(parents, limit=GATHER_CAP)
    if s.size == 0:
        return Table(EMPTY, EMPTY, EMPTY, EMPTY, np.zeros(0, bool)), bool(thin)
    m = ctx.lev[x] == level
    k = np.unique(s[m] * n + x[m])
    a, c = k // n, k % n
    dot = ctx.dotf(a * n + c) if a.size else np.zeros(0, np.float32)
    dg = sum((ip[a + 1] - ip[a]).astype(np.int64) for ip, _ix in ctx.G.hv) if a.size else np.zeros(0, np.int64)   # A1: the source's stored neighbour entries
    return Table(a, c, sortable_f32(dot), src_rank(a, dot), ctx.isS(a, c) if a.size else np.zeros(0, bool), dg), bool(thin)


def link_weights(ctx, tab, nodes, mass):
    """A1: per link of `tab` the mass its source passes along it: mass(a) // (2 * degree(a)) (damping 0.5, the transition 1/degree of PageRank); 0 for a source that is not in `nodes`."""
    if tab.c.size == 0:
        return np.zeros(0, np.int64)
    ctx.mbuf[nodes] = mass
    w = ctx.mbuf[tab.a] // (2 * np.maximum(tab.dg, 1))
    ctx.mbuf[nodes] = 0
    return w


def posmap(ctx, tab_a, nodes, pos):
    """Per link, the place of its source in the kept order (0 = an S-tail, 1.. = the other kept nodes in order, BIGR = not kept)."""
    ctx.posbuf[nodes] = pos
    p = ctx.posbuf[tab_a]
    ctx.posbuf[nodes] = BIGR
    return p


def layer(ctx, tab, m, score, nmax, qseed, probes, free, lw=None):
    """One hop's kept set from the links of `tab` whose mask m is true. Returns the S-tail nodes, the first nmax other nodes in order, the count of new (not in the free set) nodes
    among the S-tails and cumulatively along the order, and the place of each probe node (-1 not a candidate, 0 an S-tail, k the k-th other node). For the PageRank score PP (A1)
    `lw` holds the mass each link carries and a node's key is the sum over its masked links; the keys of the kept nodes come back as their mass for the next hop."""
    ngold = probes.size
    out = {"s_nodes": EMPTY, "tops": EMPTY, "s_nb": 0, "cum_nb": np.zeros(1, np.int64), "gpos": np.full(ngold, -1, np.int64), "ncand": 0, "s_mass": EMPTY, "tops_mass": EMPTY}
    if tab.c.size == 0 or not m.any():
        return out
    cnt = np.add.reduceat(m.astype(np.int32), tab.gs)
    ok = cnt > 0
    cu = tab.cu[ok]
    mrk = np.minimum.reduceat(np.where(m, tab.rank, BIGR), tab.gs)[ok]
    mdt = np.maximum.reduceat(np.where(m, tab.dotq, -1), tab.gs)[ok]
    anyS = np.maximum.reduceat((m & tab.S).astype(np.int8), tab.gs)[ok] > 0
    if score == "PP":
        key = np.add.reduceat(np.where(m, lw, 0).astype(np.int64), tab.gs)[ok]
    else:
        key = node_keys(score, ctx.ubin(cu), mrk, cnt[ok], mdt, cu, qseed)
    nonS = ~anyS
    kk, cc = key[nonS], cu[nonS]
    out["ncand"] = int(cu.size)
    if nmax > 0 and kk.size:
        if kk.size <= nmax:
            sel = np.lexsort((cc, -kk))
            tops, tm = cc[sel], kk[sel]
        else:
            thr = np.partition(kk, kk.size - nmax)[kk.size - nmax]
            cand = kk >= thr
            sel = np.lexsort((cc[cand], -kk[cand]))[:nmax]
            tops, tm = cc[cand][sel], kk[cand][sel]
    else:
        tops, tm = EMPTY, EMPTY
    s_nodes = cu[anyS]
    out["s_nodes"], out["tops"] = s_nodes, tops
    if score == "PP":
        out["s_mass"], out["tops_mass"] = key[anyS], tm
    nf_s = np.searchsorted(free, s_nodes)
    nf_s = ~((nf_s < free.size) & (free[np.minimum(nf_s, max(free.size - 1, 0))] == s_nodes)) if free.size else np.ones(s_nodes.size, bool)
    out["s_nb"] = int(nf_s.sum())
    if tops.size:
        pt = np.searchsorted(free, tops)
        nf_t = ~((pt < free.size) & (free[np.minimum(pt, max(free.size - 1, 0))] == tops)) if free.size else np.ones(tops.size, bool)
        out["cum_nb"] = np.r_[0, np.cumsum(nf_t)].astype(np.int64)
    for j in range(ngold):
        g = int(probes[j])
        i = int(np.searchsorted(cu, g))
        if i >= cu.size or int(cu[i]) != g:
            continue
        if anyS[i]:
            out["gpos"][j] = 0
        else:
            kg = key[i]
            out["gpos"][j] = 1 + int((kk > kg).sum()) + int(((kk == kg) & (cc < g)).sum())
    return out


def size_at(r, N):
    return r["s_nb"] + int(r["cum_nb"][min(int(N), r["cum_nb"].size - 1)])


def bins_of(pos, ladder):
    """The smallest ladder index whose N reaches place `pos` (0: an S-tail); NL when the node is not a candidate or lies beyond the ladder."""
    NL = len(ladder)
    b = np.searchsorted(np.asarray(ladder, np.int64), pos, side="left")
    return np.where(pos < 0, NL, b).astype(np.int8)


def run_question(ctx, seeds, golds, qseed):
    """The three-hop walk of one question for every score and every ladder step. No label is read; the golds are probes read after the orders are fixed.
    Returns the arrays described in the module's section 'what is stored'."""
    ladder, scores, n, G, W, lev = ctx.ladder, ctx.scores, ctx.n, ctx.G, ctx.W, ctx.lev
    NL, NS = len(ladder), len(scores)
    seeds = np.asarray(seeds, np.int64)
    golds = np.asarray(golds, np.int64)
    free = np.union1d(ctx.base, seeds).astype(np.int64)
    L1, L2_ = G1.levels(W, G, lev, seeds)
    try:
        ng = golds.size
        glev = np.zeros(ng, np.int8)
        for j in range(ng):
            g = int(golds[j])
            glev[j] = int(lev[g]) if lev[g] < G1.DEFAULT_LEV else G1.gold_distance(G, lev, g)
        gfree = np.isin(golds, free)
        sz1 = np.full((NS, NL), -1, np.int32)
        sz2 = np.full((NS, NL, NL), -1, np.int32)
        sz3 = np.full((NS, NL, NL, NL), -1, np.int32)
        gb1 = np.full((NS, ng), NL, np.int8)
        gb2 = np.full((NS, NL, ng), NL, np.int8)
        gb3 = np.full((NS, NL, NL, ng), NL, np.int8)
        diag = {"seeds": int(seeds.size), "free": int(free.size), "L1": int(L1.size), "L2": int(L2_.size), "links": [0, 0, 0], "capped": [False, False, False], "ncand": [0, 0, 0]}
        # hop 1
        tab1, cap = hop_table(ctx, seeds, 1)
        diag["links"][0], diag["capped"][0] = int(tab1.c.size), cap
        m1 = np.ones(tab1.c.size, bool)
        kept1 = []
        pp = "PP" in scores
        lw1 = link_weights(ctx, tab1, np.unique(seeds), ctx.seed_mass(np.unique(seeds))) if pp else None
        for k, K in enumerate(scores):
            r = layer(ctx, tab1, m1, K, ladder[-1], qseed, golds, free, lw=lw1 if K == "PP" else None)
            if k == 0:
                diag["ncand"][0] = r["ncand"]
            for i1 in range(NL):
                sz1[k, i1] = size_at(r, ladder[i1])
            gb1[k] = bins_of(r["gpos"], ladder)
            kept1.append((np.concatenate([r["s_nodes"], r["tops"]]), np.concatenate([np.zeros(r["s_nodes"].size, np.int32), np.arange(1, r["tops"].size + 1, dtype=np.int32)]),
                          np.concatenate([r["s_mass"], r["tops_mass"]])))
        # hop 2
        tab2, cap = hop_table(ctx, np.concatenate([x[0] for x in kept1]), 2)
        diag["links"][1], diag["capped"][1] = int(tab2.c.size), cap
        kept2 = {}
        for k, K in enumerate(scores):
            p2 = posmap(ctx, tab2.a, kept1[k][0], kept1[k][1])
            lw2 = link_weights(ctx, tab2, kept1[k][0], kept1[k][2]) if K == "PP" else None
            for i1 in range(NL):
                r = layer(ctx, tab2, p2 <= ladder[i1], K, ladder[i1], qseed, golds, free, lw=lw2)
                diag["ncand"][1] = max(diag["ncand"][1], r["ncand"])
                for i2 in range(i1 + 1):
                    sz2[k, i1, i2] = size_at(r, ladder[i2])
                gb2[k, i1] = bins_of(r["gpos"], ladder)
                kept2[(k, i1)] = (np.concatenate([r["s_nodes"], r["tops"]]), np.concatenate([np.zeros(r["s_nodes"].size, np.int32), np.arange(1, r["tops"].size + 1, dtype=np.int32)]),
                                  np.concatenate([r["s_mass"], r["tops_mass"]]))
        # hop 3
        tab3, cap = hop_table(ctx, np.concatenate([v_[0] for v_ in kept2.values()]), 3)
        diag["links"][2], diag["capped"][2] = int(tab3.c.size), cap
        for (k, i1), (nodes, pos, mass) in kept2.items():
            p3 = posmap(ctx, tab3.a, nodes, pos)
            keep3 = p3 <= ladder[i1]
            sub = tab3.subset(keep3)
            ps = p3[keep3]
            lw3 = link_weights(ctx, tab3, nodes, mass)[keep3] if scores[k] == "PP" else None
            for i2 in range(i1 + 1):
                r = layer(ctx, sub, ps <= ladder[i2], scores[k], ladder[i2], qseed, golds, free, lw=lw3)
                diag["ncand"][2] = max(diag["ncand"][2], r["ncand"])
                for i3 in range(i2 + 1):
                    sz3[k, i1, i2, i3] = size_at(r, ladder[i3])
                gb3[k, i1, i2] = bins_of(r["gpos"], ladder)
    finally:
        G1.reset(W, lev, seeds, L1, L2_)
    return {"nfree": int(free.size), "glev": glev, "gfree": gfree, "sz1": sz1, "sz2": sz2, "sz3": sz3, "gb1": gb1, "gb2": gb2, "gb3": gb3, "diag": diag}


# ── self-test: the walker equals a brute-force walk ──────────────────────────


def naive_walk(adj, lev, seeds, dot, isS, ubin, score, Ns, qseed, n, m0=None):
    """The walk by definition, with Python sets and loops: returns the kept set of each hop. For PP, m0 = {seed: mass}; a candidate's key is the sum over its links from kept parents
    of mass(parent) // (2 * degree(parent)); the kept nodes (S-tails too) carry that key on as their mass."""
    prev = set(int(s) for s in seeds)
    mass = dict(m0) if m0 is not None else {}
    kept = []
    for h in (1, 2, 3):
        fr = {a: [c for c in adj[a] if lev[c] == h] for a in prev}
        cand = {}
        for a in prev:
            for c in fr[a]:
                rk = 1 + sum(1 for c2 in fr[a] if dot(a, c2) > dot(a, c))
                cand.setdefault(c, []).append((a, rk, dot(a, c), isS(a, c)))
        s_tails = {c for c, ls in cand.items() if any(x[3] for x in ls)}
        recs = []
        pkey = {}
        if score == "PP":
            for c, ls in cand.items():
                pkey[c] = sum(mass[x[0]] // (2 * len(adj[x[0]])) for x in ls)
        for c, ls in cand.items():
            if c in s_tails:
                continue
            u = int(ubin(np.array([c]))[0])
            rk, vt, dt = min(x[1] for x in ls), len(ls), max(x[2] for x in ls)
            if score == "K1":
                kt = (u, dt)
            elif score == "K2":
                kt = (-rk, dt)
            elif score == "K3":
                kt = (u, -rk, min(vt, VTCAP), dt)
            elif score == "K4":
                kt = (dt,)
            elif score == "PP":
                kt = (pkey[c],)
            else:
                kt = (int(node_keys(score, np.array([u]), np.array([rk]), np.array([vt]), np.array([0]), np.array([c]), qseed)[0]),)
            recs.append((tuple(-x for x in kt), c))
        recs.sort()
        k_h = s_tails | {c for _kt, c in recs[:Ns[h - 1]]}
        kept.append(k_h)
        prev = k_h
        if score == "PP":
            mass = {c: pkey[c] for c in k_h}
    return kept


def toy_runs():
    """The toy graphs of the self-test: runs the walker on each, checks it against the brute-force walk, returns what the grading self-test of zgp3s.py needs."""
    rng = np.random.default_rng(3)
    # keys: the composite orders like the tuples
    for _ in range(200):
        m = 40
        u = rng.integers(0, 7, m)
        rk = rng.integers(1, 2 * RKCAP, m)
        vt = rng.integers(1, 400, m)
        dt = np.round(rng.random(m), 3).astype(np.float32) * np.where(rng.random(m) < 0.3, 0, 1).astype(np.float32)
        dq = sortable_f32(dt)
        cu = np.arange(m)
        for sc, tup in (("K1", lambda i: (u[i], dt[i])), ("K2", lambda i: (-min(rk[i], RKCAP), dt[i])), ("K3", lambda i: (u[i], -min(rk[i], RKCAP), min(vt[i], VTCAP), dt[i])), ("K4", lambda i: (dt[i],))):
            key = node_keys(sc, u, rk, vt, dq, cu, 7)
            a = np.lexsort((cu, -key))
            b = sorted(range(m), key=lambda i: (tuple(-x for x in tup(i)), i))
            assert a.tolist() == b, sc
    assert sortable_f32(np.array([-1.5, -0.0, 0.0, 1e-9, 2.0], np.float32)).tolist() == sorted(sortable_f32(np.array([-1.5, -0.0, 0.0, 1e-9, 2.0], np.float32)).tolist())
    # src_rank: strictly larger dots
    a = np.array([0, 0, 0, 1, 1, 2], np.int64)
    d = np.array([0.5, 0.5, 0.9, 0.1, 0.2, 0.3], np.float32)
    assert src_rank(a, d).tolist() == [2, 2, 1, 2, 1, 1]
    # the walk on toy graphs against the brute force
    ladder = (0, 1, 3, 6)
    NL = len(ladder)
    trials = checked = covered_checks = 0
    packed, naive_rows = [], []
    toy_triples = [(i, j, k_) for i in range(NL) for j in range(i + 1) for k_ in range(j + 1)]
    for trial in range(70):
        n = int(rng.integers(14, 46))
        keys = G1.toy_graph(rng, n, int(rng.integers(2 * n, 5 * n)))
        if keys.size < 8:
            continue
        comp = H.build_comp(keys, n)
        G = G1.Graph([comp], n)
        adj = {i: set() for i in range(n)}
        for kk in keys.tolist():
            adj[kk // n].add(kk % n)
            adj[kk % n].add(kk // n)
        W = H.Walker(n)
        lev = np.full(n, G1.DEFAULT_LEV, np.int8)
        seeds = np.unique(rng.integers(0, n, int(rng.integers(1, 4))))
        golds = np.unique(rng.integers(0, n, int(rng.integers(1, 6))))
        mod = int(rng.choice([4, 16, 997]))
        dotf = lambda kk_, mod=mod: (((np.minimum(kk_ // n, kk_ % n) * 7919 + np.maximum(kk_ // n, kk_ % n) * 104729) % mod).astype(np.float32) / np.float32(mod))  # noqa: E731
        s_mod = int(rng.choice([1000003, 5, 2]))
        isS = lambda a_, c_, s_mod=s_mod: ((np.minimum(a_, c_) * 31 + np.maximum(a_, c_) * 17) % s_mod == 0)  # noqa: E731
        ubin = lambda c_: ((np.asarray(c_) * 5 + 3) % 7).astype(np.int8)  # noqa: E731
        ctx = Ctx(G, W, lev, n, dotf, isS, ladder=ladder, ubin=ubin, seedw=lambda s_: (np.asarray(s_) * 13 + 5) % 9 + 1)
        su = np.unique(seeds)
        m0 = dict(zip(su.tolist(), ctx.seed_mass(su).tolist()))
        assert sum(m0.values()) <= 1 << PPR_BITS and sum(m0.values()) > (1 << PPR_BITS) - len(su) * (1 << (PPR_BITS - 30)) - 1
        base = np.unique(rng.integers(0, n, int(rng.integers(1, 6))))
        ctx.set_question(None, None, base)
        qseed = int(rng.integers(0, 1000))
        res = run_question(ctx, seeds, golds, qseed)
        res0 = run_question(ctx, seeds, EMPTY, qseed)                          # G3-5: the sizes do not depend on the probes
        for key_ in ("sz1", "sz2", "sz3"):
            assert np.array_equal(res[key_], res0[key_]), key_
        assert bool((lev == G1.DEFAULT_LEV).all()) and not bool(W.vis.any())
        # levels for the naive walk
        lev_n = np.full(n, 9, np.int64)
        for s in seeds.tolist():
            lev_n[s] = 0
        fr = list(seeds.tolist())
        for h in (1, 2):
            nf = []
            for u_ in fr:
                for x in adj[u_]:
                    if lev_n[x] == 9:
                        lev_n[x] = h
                        nf.append(x)
            fr = nf
        lev_n = np.where(lev_n == 9, 3, lev_n)
        dot1 = lambda a_, c_: float(dotf(np.array([a_ * n + c_]))[0])  # noqa: E731
        isS1 = lambda a_, c_: bool(isS(np.array([a_]), np.array([c_]))[0])  # noqa: E731
        free = set(np.union1d(base, seeds).tolist())
        assert res["nfree"] == len(free)
        nv_all = np.zeros((len(SCORES), len(toy_triples)), bool)
        nv_size = np.zeros((len(SCORES), len(toy_triples)), np.int64)
        for k, K in enumerate(SCORES):
            for i1 in range(NL):
                for i2 in range(i1 + 1):
                    for i3 in range(i2 + 1):
                        kept = naive_walk(adj, lev_n, seeds, dot1, isS1, ubin, K, (ladder[i1], ladder[i2], ladder[i3]), qseed, n, m0)
                        pool = free | kept[0] | kept[1] | kept[2]
                        size = res["nfree"] + res["sz1"][k, i1] + res["sz2"][k, i1, i2] + res["sz3"][k, i1, i2, i3]
                        assert size == len(pool), (trial, K, i1, i2, i3, size, len(pool))
                        ti = toy_triples.index((i1, i2, i3))
                        nv_size[k, ti] = len(pool)
                        nv_all[k, ti] = all(int(g) in pool for g in golds.tolist())
                        for j, g in enumerate(golds.tolist()):
                            lv = int(res["glev"][j])
                            if res["gfree"][j] or lv == 0:
                                ok = True
                            elif lv == 1:
                                ok = res["gb1"][k, j] <= i1
                            elif lv == 2:
                                ok = res["gb2"][k, i1, j] <= i2
                            elif lv == 3:
                                ok = res["gb3"][k, i1, i2, j] <= i3
                            else:
                                ok = False
                            assert bool(ok) == (g in pool), (trial, K, i1, i2, i3, g, lv, bool(ok), g in pool)
                            covered_checks += 1
                        checked += 1
        packed.append(res)
        naive_rows.append((nv_all, nv_size))
        trials += 1
    assert trials > 40 and checked > 2000, (trials, checked)
    return trials, checked, covered_checks, packed, naive_rows, toy_triples


def selftest():
    trials, checked, covered_checks, _packed, _naive, _tt = toy_runs()
    log(f"selftest PASS: composite keys equal the tuple orders, src_rank, {trials} toy graphs x {len(SCORES)} scores x 20 triples = {checked} pools and {covered_checks} gold checks equal the brute-force walk; the sizes do not depend on the probes")
    return 0


# ── the population and the real context ──────────────────────────────────────


def part_indices(nq, part):
    return G1.part_indices(nq, part)


def real_context(d, v, D, host):
    """The graph of the cell, the walker, the dot, the S test, and the dev lists."""
    prov = H.RealProv(D)
    reg = H.CompReg(prov, v, D.n)
    srk = v == "SRK"
    names = (["S"] if srk else []) + ["W", "T0", "K"]
    G = G1.Graph([reg.get(x) for x in names], D.n)
    W = H.Walker(D.n)
    lev = np.full(D.n, G1.DEFAULT_LEV, np.int8)
    rows = T3.rows_of(d, D, host)
    dotf = lambda keys: T3.pair_dot(rows, keys, D.n, block=G1.DOT_BLOCK)  # noqa: E731
    sk = D.F["S"] if srk else None
    isS = (lambda a, c: UL.member(G1.link_key(a, c, D.n), sk)) if srk else (lambda a, c: np.zeros(np.asarray(a).size, bool))
    return Ctx(G, W, lev, D.n, dotf, isS), reg


def base_lists(ld, ls, b):
    import m3a_headroom as M3A
    return np.asarray(M3A.rrf_fused_rows(np.asarray(ld, np.int64)[None, :], np.asarray(ls, np.int64)[None, :], RRF_CONST)[0][:b], np.int64)


def setup(a):
    Z.gate_pins()
    d = a.ds
    D = P.load_mhr() if d == "multihoprag" else P.load_host(d, a.host)
    gt = P.gates(D, D.g)
    sl = G1.dev_slice(d, D, a.host)
    L = Z.dev_lists(d)
    if "not_read" in sl:
        stop(sl["not_read"])
    nq = len(sl["ids"])
    if L["nq"] != nq:
        stop(f"G3-1: {d}: {L['nq']} dev questions rebuilt, {nq} in the slice")
    for qi in range(min(nq, 50)):
        sd = set(sl["seeds"][qi].tolist())
        if sd != (set(L["dense5"][qi].tolist()) | set(L["spl5"][qi].tolist())):
            stop(f"G3-1: {d}: question {qi}: the seeds are not the first-stage top fives")
    return D, gt, sl, L, nq


def cmd_time(a):
    t0 = time.time()
    D, gt, sl, L, nq = setup(a)
    sel = part_indices(nq, "q")[:a.n]
    rec = []
    for _dd, v in [(dd, vv) for dd, vv in CELLS if dd == a.ds]:
        ctx, _reg = real_context(a.ds, v, D, a.host)
        for qi in sel.tolist():
            ld, ls = L["dense"](qi), L["spl"](qi)
            ctx.set_question(ld, ls, base_lists(ld, ls, BASE_B[a.ds]))
            t1 = time.time()
            r = run_question(ctx, sl["seeds"][qi], EMPTY, qi)
            dg = r["diag"]
            rec.append((round(time.time() - t1, 2), dg))
            log(f"{a.ds}/{v} q{qi}: {rec[-1][0]} s, links {dg['links']}, candidates {dg['ncand']}, L1 {dg['L1']}, L2 {dg['L2']}, capped {dg['capped']} ({G1.C.peak_gb()} GB)")
    secs = [x for x, _d in rec]
    log(f"{a.ds}: {len(rec)} question-walks, mean {np.mean(secs):.2f} s, max {max(secs):.2f} s, total {time.time() - t0:.0f} s (no label read)")
    return 0


def pack_chunk(res, qs):
    """The arrays of one chunk of questions: per question sizes; per gold level, free flag, bins."""
    Q = len(res)
    gq = np.concatenate([np.full(r["glev"].size, i, np.int32) for i, r in enumerate(res)]) if Q else np.zeros(0, np.int32)
    cat = lambda k: np.concatenate([np.moveaxis(r[k], -1, 0) for r in res]) if Q else np.zeros(0)  # noqa: E731  (gold axis first)
    out = {"qidx": np.asarray(qs, np.int32), "nfree": np.asarray([r["nfree"] for r in res], np.int32),
           "sz1": np.stack([r["sz1"] for r in res]), "sz2": np.stack([r["sz2"] for r in res]), "sz3": np.stack([r["sz3"] for r in res]),
           "gq": gq, "glev": np.concatenate([r["glev"] for r in res]), "gfree": np.concatenate([r["gfree"] for r in res]),
           "gb1": cat("gb1").astype(np.int8), "gb2": cat("gb2").astype(np.int8), "gb3": cat("gb3").astype(np.int8),
           "links": np.asarray([r["diag"]["links"] for r in res], np.int64), "ncand": np.asarray([r["diag"]["ncand"] for r in res], np.int64),
           "capped": np.asarray([r["diag"]["capped"] for r in res], bool), "lv12": np.asarray([[r["diag"]["L1"], r["diag"]["L2"]] for r in res], np.int64)}
    return out


def cmd_extract(a):
    d, t0 = a.ds, time.time()
    if a.part not in ("q", "r"):
        stop("--part is q or r")
    D, gt, sl, L, nq = setup(a)
    sel = part_indices(nq, a.part)
    if a.limit:
        sel = sel[:a.limit]
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    if not (shard_n >= 1 and 0 <= shard_i < shard_n):
        stop(f"--shard {a.shard}")
    mine = [c0 for c0 in range(0, sel.size, CHUNK_Q) if (c0 // CHUNK_Q) % shard_n == shard_i]
    tag = "" if shard_n == 1 else f"_s{shard_i}of{shard_n}"
    OUT_D = OUT / d
    wd = OUT_D / ("work_limit" if a.limit else "work")
    wd.mkdir(parents=True, exist_ok=True)
    zsha = script_sha256()
    log(f"{d}: {nq} dev questions with a gold, part {a.part}: {sel.size}; ladder {LADDER}; scores {SCORES} ({time.time() - t0:.0f} s)")
    rec = {"dataset": d, "declared_in": DECLARED, "part": a.part, "status": "read", "nodes": D.n, "nodes_sha256": D.nsha, "shard": [shard_i, shard_n], "nq": nq,
           "gates": {"UL2P": gt, "G3-1": "PASS"}, "versions": {}, "constants": {"LADDER": list(LADDER), "SCORES": list(SCORES), "BASE_B": BASE_B[d], "GATHER_CAP": GATHER_CAP, "RKCAP": RKCAP,
                                                                                "VTCAP": VTCAP, "CHUNK_Q": CHUNK_Q, "QUARTER_M": QUARTER_M}}
    for _dd, v in [(dd, vv) for dd, vv in CELLS if dd == d]:
        srk = v == "SRK"
        ctx, _reg = real_context(d, v, D, a.host)
        stats = {"questions": 0, "golds": 0, "free_golds": 0, "levels": [0, 0, 0, 0, 0], "capped_questions": 0}
        chunks = []
        for c0 in mine:
            qs = sel[c0:c0 + CHUNK_Q]
            cp = wd / f"{v}_{a.part}_{c0:06d}.npz"
            if cp.is_file():
                with np.load(cp) as z:
                    same = str(z["zsha"]) == zsha
                    if same:
                        cs = json.loads(str(z["cstats"]))
                if same:
                    chunks.append(cp)
                    for k_, x in cs.items():
                        stats[k_] = [p + q for p, q in zip(stats[k_], x)] if k_ == "levels" else stats[k_] + x
                    continue
                log(f"{rel(cp)} was written by another version of zgp3: recomputed")
            res = []
            for qi in qs.tolist():
                ld, ls = L["dense"](qi), L["spl"](qi)
                ctx.set_question(ld, ls, base_lists(ld, ls, BASE_B[d]))
                r = run_question(ctx, sl["seeds"][qi], sl["golds"][qi], int(qi))
                if not srk and not bool((r["sz1"][:, 0] == 0).all() and (r["sz2"][:, 0, 0] == 0).all() and (r["sz3"][:, 0, 0, 0] == 0).all()):
                    stop(f"G3-4: {d}/{v}: question {qi}: an RK cell keeps nodes at N = 0")
                if len(res) < 2:                                                    # G3-5: the sizes do not depend on the probes
                    r0 = run_question(ctx, sl["seeds"][qi], EMPTY, int(qi))
                    for k_ in ("sz1", "sz2", "sz3"):
                        if not np.array_equal(r[k_], r0[k_]):
                            stop(f"G3-5: {d}/{v}: question {qi}: {k_} depends on the probes")
                res.append(r)
            pk = pack_chunk(res, qs)
            cs = {"questions": int(len(res)), "golds": int(pk["glev"].size), "free_golds": int(pk["gfree"].sum()),
                  "levels": [int((pk["glev"] == j).sum()) for j in range(5)], "capped_questions": int(pk["capped"].any(axis=1).sum())}
            tmp = cp.with_name(cp.name + ".tmp.npz")
            np.savez_compressed(tmp, nq=np.asarray(nq), zsha=np.array(zsha), cstats=np.array(json.dumps(cs)), **pk)
            tmp.replace(cp)
            for k_, x in cs.items():
                stats[k_] = [p + q for p, q in zip(stats[k_], x)] if k_ == "levels" else stats[k_] + x
            chunks.append(cp)
            log(f"{d}/{v}/{a.part}: {min(c0 + CHUNK_Q, sel.size)}/{sel.size} questions ({time.time() - t0:.0f} s, {G1.C.peak_gb()} GB)")
        rec["versions"][v] = {"stats": stats, "chunks": [rel(x) for x in chunks]}
    rec["memory"] = G1.C.peak_gb()
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zgp3_sha256=script_sha256(), zgp2_sha256=Z.script_sha256(),
               zgp1_sha256=G1.script_sha256())
    fn = OUT_D / f"extract_{a.part}{tag}{'_limit' if a.limit else ''}.json"
    L2.wjson(fn, rec)
    log(f"{d}: {rel(fn)} ({rec['seconds']} s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    t = sub.add_parser("time")
    t.add_argument("--ds", required=True)
    t.add_argument("--n", type=int, default=20)
    t.add_argument("--host", action="store_true")
    t.add_argument("--threads", type=int, default=1)
    x = sub.add_parser("extract")
    x.add_argument("--ds", required=True)
    x.add_argument("--part", required=True)
    x.add_argument("--host", action="store_true")
    x.add_argument("--threads", type=int, default=1)
    x.add_argument("--limit", type=int, default=0)
    x.add_argument("--shard", default="0/1")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    return {"time": cmd_time, "extract": cmd_extract}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
