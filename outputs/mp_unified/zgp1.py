"""GP1 (docs/GP1_GOLD_LINK_PATTERNS.md): which links carry the gold paths? The engine: for the golds the first stage misses, the shortest link paths
from the seeds (dense top 5 U SPLADE top 5) over the any-link graph B0 (UL2H's graph and walker) are the NEEDED links; the other frontier links that leave
the same sources at the same step are the CONTROLS (sampled, weighted). Per link the declared attributes are read (label, dot, rank, votes, who names
whom, exactness, token count, capitalisation, lead, offset, count, question overlap, relation). Nothing trains, no model reads anything, no LLM or spaCy;
no filed graph, pool, fit or record changes; the gold is the grading only. zul2h, zul2p, zul3a (and what they import) are imported, never edited.
Outputs only under outputs/gp1/ (aggregated records are committed; the link-level work files are not).

    python outputs/mp_unified/zgp1.py selftest
    python outputs/mp_unified/zgp1.py extract --ds D --part q|r [--host] [--threads 1] [--limit N]
The program (screen, solve, grade, report) is zgp1p.py.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zul2h as H  # noqa: E402  (imports zul2s -> zul2p -> zul2c -> zul2u -> zul2 -> zmhb -> zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import mmap  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import zul3a as T3  # noqa: E402

P, C, L2, UL, E, AP, SS = H.P, H.C, H.L2, H.UL, H.E, H.AP, H.SS
DECLARED = "docs/GP1_GOLD_LINK_PATTERNS.md"
OUT = ROOT / "outputs" / "gp1"
EMPTY = np.zeros(0, np.int64)
CAP_SRC = 200                                       # controls kept per source (section 4)
CAP_Q = 20_000                                      # controls kept per question
CAP_GOLD = 5_000                                    # needed units kept per gold
ENUM_LIMIT = 30_000_000                             # gathered entries per enumeration before the nodes are thinned by hash order
QUARTER_M = 4                                       # the quarter Q: dev questions with index mod 4 = 0; R: the rest
DOT_BLOCK = 1 << 14
G5_PER_CHUNK = 1000                                 # R.mention links recounted per chunk by an independent regular expression (G-G5: 2,000 per dataset)
DEFAULT_LEV = 3                                     # lev: 0 seeds, 1, 2; 3 = not within two hops
CHUNK_Q = 50                                        # questions per work chunk (a checkpoint)
STOP_WORDS = frozenset("a an the of in on at to for and or but by with from as is are was were be been being this that these those it its his her their "
                       "which who whom whose what when where why how did do does had has have not no yes than then there here into over under after before "
                       "between during about against also one two".split())
ATTRS = ("grp", "dot", "rank", "votes", "dir", "exact", "ntok", "caps", "lead", "off", "cnt", "qov", "rel")
log = P.log


def stop(msg):
    raise SystemExit(f"[zgp1] STOP: {msg}")


def rel(p):
    return P.rel(p)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── hashing, the graph ───────────────────────────────────────────────────────


def mix64(x):
    """splitmix64 of an int array: the fixed order of every cap (no random state)."""
    x = np.asarray(x, np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def link_key(a, c, n):
    a, c = np.asarray(a, np.int64), np.asarray(c, np.int64)
    return np.minimum(a, c) * n + np.maximum(a, c)


class Graph:
    """The any-link graph as UL2H's components (undirected CSR halves): distinct neighbours of a node, pairs of a node list."""

    def __init__(self, comps, n):
        self.hv = [h for c in comps for h in c.halves]
        self.n = n

    def nb(self, x):
        x = int(x)
        parts = [ix[ip[x]:ip[x + 1]] for ip, ix in self.hv]
        a = np.concatenate(parts) if parts else EMPTY
        return np.unique(a.astype(np.int64))

    def gather(self, nodes, limit=ENUM_LIMIT):
        """(src, nbr, thinned): every (node, neighbour) entry of the nodes over all halves (repeats across components are not removed here).
        Over `limit` entries the nodes are thinned in hash order and thinned = True."""
        nodes = np.unique(np.asarray(nodes, np.int64))
        thinned = False
        tot = sum(int((ip[nodes + 1] - ip[nodes]).sum()) for ip, _ix in self.hv) if nodes.size else 0
        if tot > limit:
            thinned = True
            nodes = nodes[np.argsort(mix64(nodes), kind="stable")]
            deg = sum((ip[nodes + 1] - ip[nodes]) for ip, _ix in self.hv)
            keep = max(1, int(np.searchsorted(np.cumsum(deg), limit, side="right")))
            nodes = nodes[:keep]
        src, nbr = [], []
        for ip, ix in self.hv:
            st = ip[nodes]
            ln = ip[nodes + 1] - st
            t = int(ln.sum())
            if t == 0:
                continue
            loc = np.cumsum(ln) - ln
            off = np.repeat(st - loc, ln) + np.arange(t)
            nbr.append(ix[off].astype(np.int64))
            src.append(np.repeat(nodes, ln))
        return (np.concatenate(src), np.concatenate(nbr), thinned) if src else (EMPTY, EMPTY, thinned)


# ── levels, distances, the shortest-path links ───────────────────────────────


def levels(W, G, lev, seeds):
    """lev[seeds] = 0, L1 = 1, L2 = 2 (default 3 elsewhere); returns (L1, L2) and leaves W.vis marked (reset by `reset`)."""
    seeds = np.asarray(seeds, np.int64)
    W.vis[seeds] = True
    L1 = W.expand(G.hv, seeds)
    L2_ = W.expand(G.hv, L1)
    lev[seeds] = 0
    lev[L1] = 1
    lev[L2_] = 2
    return L1, L2_


def reset(W, lev, seeds, L1, L2_):
    for x in (seeds, L1, L2_):
        W.vis[x] = False
        lev[x] = DEFAULT_LEV


def gold_distance(G, lev, g):
    """0, 1, 2 from the levels; 3 when g (not within two hops) has a neighbour at level 2; else 4 (unreached)."""
    if lev[g] < DEFAULT_LEV:
        return int(lev[g])
    ng = G.nb(g)
    return 3 if ng.size and bool((lev[ng] == 2).any()) else 4


def up(G, lev, nodes, level):
    """The (node, x) pairs, x a neighbour of the node at level `level`: distinct, plus whether the gather was thinned."""
    s, x, thin = G.gather(nodes)
    m = lev[x] == level
    k = np.unique(s[m] * G.n + x[m])
    return k // G.n, k % G.n, thin


def gold_units(G, lev, g, d):
    """(a, c, step, capped): the links on some shortest path of length d from a seed to g, oriented from the lower level to the higher; capped
    when the enumeration was thinned or more than CAP_GOLD distinct links were found (the first CAP_GOLD in hash order of the link key are kept)."""
    n = G.n
    ng = G.nb(g)
    capped = False
    if d == 1:
        a = ng[lev[ng] == 0]
        A, Cc, S = a, np.full(a.size, g, np.int64), np.ones(a.size, np.int8)
    elif d == 2:
        m = ng[lev[ng] == 1]
        mm, ss, thin = up(G, lev, m, 0)
        capped |= thin
        A = np.concatenate([ss, m])
        Cc = np.concatenate([mm, np.full(m.size, g, np.int64)])
        S = np.concatenate([np.ones(ss.size, np.int8), np.full(m.size, 2, np.int8)])
    elif d == 3:
        m2 = ng[lev[ng] == 2]
        p2, p1, thin1 = up(G, lev, m2, 1)
        m1 = np.unique(p1)
        q1, s0, thin2 = up(G, lev, m1, 0)
        capped |= thin1 | thin2
        A = np.concatenate([s0, p1, m2])
        Cc = np.concatenate([q1, p2, np.full(m2.size, g, np.int64)])
        S = np.concatenate([np.ones(s0.size, np.int8), np.full(p1.size, 2, np.int8), np.full(m2.size, 3, np.int8)])
    else:
        return EMPTY, EMPTY, np.zeros(0, np.int8), False
    key = A * n + Cc
    _, ui = np.unique(key, return_index=True)
    A, Cc, S = A[ui], Cc[ui], S[ui]
    if A.size > CAP_GOLD:
        keep = np.sort(np.argsort(mix64(link_key(A, Cc, n)), kind="stable")[:CAP_GOLD])
        A, Cc, S = A[keep], Cc[keep], S[keep]
        capped = True
    return A, Cc, S, capped


def analyse_levels(G, W, lev, seeds, golds):
    """Per gold: distance and units. Returns the lists (dist, units) and the nodes to reset."""
    L1, L2_ = levels(W, G, lev, seeds)
    out = []
    for g in golds.tolist():
        d = gold_distance(G, lev, g)
        out.append((d, gold_units(G, lev, g, d) if 1 <= d <= 3 else (EMPTY, EMPTY, np.zeros(0, np.int8), False)))
    return out, (L1, L2_)


def sample_controls(G, lev, src, step, needed_tails, cap, dotf):
    """One source's frontier links (neighbours at level `step`, default 3 for step 3) minus its needed tails: (c, dot, rank, weight) of the sampled
    controls, the needed links' (dot, rank), and the exact control count. rank = 1 + the number of frontier links with a strictly larger dot."""
    nbr = G.nb(src)
    fr = nbr[lev[nbr] == step]
    dots = dotf(src * G.n + fr) if fr.size else np.zeros(0, np.float32)
    srt = np.sort(dots)

    def rank_of(v):
        return (fr.size - np.searchsorted(srt, v, side="right") + 1).astype(np.int64)
    isn = np.isin(fr, needed_tails)
    cc, cd = fr[~isn], dots[~isn]
    ncs = int(cc.size)
    if ncs > cap:
        sel = np.sort(np.argpartition(mix64(link_key(np.full(cc.size, src), cc, G.n)), cap - 1)[:cap]) if cap > 0 else np.zeros(0, np.int64)
        w = ncs / cap
    else:
        sel = np.arange(ncs)
        w = 1.0
    nd = dots[isn]
    return {"c": cc[sel], "dot": cd[sel], "rank": rank_of(cd[sel]), "w": np.full(sel.size, w), "n_ctrl": ncs,
            "needed_c": fr[isn], "needed_dot": nd, "needed_rank": rank_of(nd)}


def question_records(G, W, lev, seeds, golds, dotf, cap_src=CAP_SRC, cap_q=CAP_Q):
    """Everything one question gives: the golds' distances and units, the link table (needed units, then sampled controls), the incidences.
    Link table columns: a, c, step, kind (1 needed, 0 control), w (needed: sum of 1/|N(g)| over the golds it serves; control: the sampling
    weight), dot, rank, votes, tdeg. Raises through stop() on a gate (G-G3, G-G4) failure."""
    n = G.n
    seeds = np.asarray(seeds, np.int64)
    golds = np.asarray(golds, np.int64)
    per_gold, rs = analyse_levels(G, W, lev, seeds, golds)
    try:
        dist = np.array([d for d, _u in per_gold], np.int64)
        capped = np.array([u[3] for _d, u in per_gold], bool)
        keys, wsum, first = {}, {}, {}
        inc_g, inc_k = [], []
        for gi, (d, (A, Cc, S, _cp)) in enumerate(per_gold):
            if not A.size:
                continue
            lk = link_key(A, Cc, n) * 4 + S                                  # a link is one unit per step
            wg = 1.0 / A.size
            for j in range(A.size):
                k = int(lk[j])
                if k not in keys:
                    keys[k] = len(keys)
                    first[k] = (int(A[j]), int(Cc[j]), int(S[j]))
                    wsum[k] = 0.0
                wsum[k] += wg
                inc_g.append(gi)
                inc_k.append(keys[k])
        nu = len(keys)
        ua = np.array([first[k][0] for k in keys], np.int64) if nu else EMPTY
        uc = np.array([first[k][1] for k in keys], np.int64) if nu else EMPTY
        us = np.array([first[k][2] for k in keys], np.int8) if nu else np.zeros(0, np.int8)
        uw = np.array([wsum[k] for k in keys], np.float64) if nu else np.zeros(0)
        reach_golds = int(((dist >= 1) & (dist <= 3)).sum())
        if nu and abs(float(uw.sum()) - float(sum(1 for gi in range(len(per_gold)) if per_gold[gi][1][0].size))) > 1e-6:
            stop("G-G3: the unit weights do not sum to the number of golds with units")
        # sources: one per (a, step)
        srcs = np.unique(ua * 4 + us.astype(np.int64)) if nu else EMPTY
        ncap = min(cap_src, max(1, cap_q // max(1, srcs.size)))
        rows = {"dot": np.zeros(nu, np.float32), "rank": np.zeros(nu, np.int64)}
        ctrl = {"a": [], "c": [], "step": [], "dot": [], "rank": [], "w": []}
        ctrl_stats = {"sources": int(srcs.size), "ctrl_exact": 0, "ctrl_sampled": 0}
        for sk in srcs.tolist():
            a, st = sk // 4, sk % 4
            m = (ua == a) & (us == st)
            nt = np.unique(uc[m])
            r = sample_controls(G, lev, a, st, nt, ncap, dotf)
            # the needed links' own dot and rank, in the order of the unit table
            if m.any():
                idx = np.flatnonzero(m)
                pos = np.searchsorted(r["needed_c"], uc[idx]) if r["needed_c"].size else np.zeros(idx.size, np.int64)
                order = np.argsort(r["needed_c"], kind="stable")
                srt_c = r["needed_c"][order]
                pos = np.searchsorted(srt_c, uc[idx])
                if not bool((pos < srt_c.size).all()) or not bool((srt_c[np.minimum(pos, srt_c.size - 1)] == uc[idx]).all()):
                    stop("G-G3: a needed link is not a frontier link of its source")
                rows["dot"][idx] = r["needed_dot"][order][pos]
                rows["rank"][idx] = r["needed_rank"][order][pos]
            nfr = int((lev[G.nb(a)] == st).sum())
            if abs(float(r["w"].sum()) - r["n_ctrl"]) > 1e-6 * max(1, r["n_ctrl"]) or (r["n_ctrl"] + nt.size != nfr):
                stop(f"G-G4: source {a}: control weights {r['w'].sum()} for {r['n_ctrl']} controls, {nfr} frontier links, {nt.size} needed")
            ctrl_stats["ctrl_exact"] += r["n_ctrl"]
            ctrl_stats["ctrl_sampled"] += int(r["c"].size)
            for k, v in (("c", r["c"]), ("dot", r["dot"]), ("rank", r["rank"]), ("w", r["w"])):
                ctrl[k].append(v)
            ctrl["a"].append(np.full(r["c"].size, a, np.int64))
            ctrl["step"].append(np.full(r["c"].size, st, np.int8))
        cat = lambda k, dt: np.concatenate(ctrl[k]).astype(dt) if ctrl[k] else np.zeros(0, dt)  # noqa: E731
        ca, cc_, cs = cat("a", np.int64), cat("c", np.int64), cat("step", np.int8)
        a_all = np.concatenate([ua, ca])
        c_all = np.concatenate([uc, cc_])
        s_all = np.concatenate([us, cs])
        kind = np.concatenate([np.ones(nu, np.int8), np.zeros(ca.size, np.int8)])
        w = np.concatenate([uw, cat("w", np.float64)])
        dot = np.concatenate([rows["dot"], cat("dot", np.float32)]).astype(np.float32)
        rank = np.concatenate([rows["rank"], cat("rank", np.int64)])
        votes = np.zeros(a_all.size, np.int64)
        tdeg = np.zeros(a_all.size, np.int64)
        cache = {}
        for i in range(a_all.size):
            c, st = int(c_all[i]), int(s_all[i])
            if c not in cache:
                nc = G.nb(c)
                cache[c] = (nc, int(nc.size))
            nc, dg = cache[c]
            votes[i] = int((lev[nc] == st - 1).sum())
            tdeg[i] = dg
        return {"dist": dist, "capped": capped, "n_units": nu, "reach_golds": reach_golds, "a": a_all, "c": c_all, "step": s_all, "kind": kind, "w": w,
                "dot": dot, "rank": rank, "votes": votes, "tdeg": tdeg, "inc_g": np.array(inc_g, np.int64), "inc_k": np.array(inc_k, np.int64),
                "stats": ctrl_stats}
    finally:
        reset(W, lev, seeds, *rs)


# ── text attributes ──────────────────────────────────────────────────────────

_SENT = re.compile(r"[.!?][\"')\]]*(?:\s|$)")
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def _word_find_all(title, text):
    out, start, L = [], 0, len(title)
    if not L:
        return out
    while True:
        i = text.find(title, start)
        if i < 0:
            break
        j = i + L
        ok = (i == 0 or not text[i - 1].isalnum()) and (j >= len(text) or not text[j].isalnum())
        if ok:
            out.append(i)
        start = j if ok else i + 1
    return out


def mention_stats(title, text):
    """(level, count, first offset, lead) of the title in the text: level 3 = exactly as written (case-sensitive, word-bounded), 2 = only case-folded,
    1 = only by an alias form (the title without its trailing parenthetical, the text before its first comma, its last token of four characters or
    more), 0 = not at all. count and first offset are those of the matching form; lead = the first match lies in the first sentence."""
    if not title or not text:
        return 0, 0, -1, False
    pos = _word_find_all(title, text)
    level = 3 if pos else 0
    if not pos:
        tl = title.lower()
        pos = _word_find_all(tl, text.lower())
        level = 2 if pos else 0
    if not pos:
        forms = []
        f1 = _PAREN.sub("", title).strip()
        if f1 and f1 != title:
            forms.append(f1)
        if "," in title:
            f2 = title.split(",")[0].strip()
            if f2 and f2 != title:
                forms.append(f2)
        toks = title.split()
        if len(toks) >= 2 and len(toks[-1]) >= 4:
            forms.append(toks[-1])
        best = None
        for f in dict.fromkeys(forms):
            if len(f) < 4:
                continue
            p = _word_find_all(f, text)
            if p and (best is None or p[0] < best[0]):
                best = p
        if best:
            pos, level = best, 1
    if not pos:
        return 0, 0, -1, False
    m = _SENT.search(text, 1)
    end = m.end() if m else len(text)
    return level, len(pos), pos[0], pos[0] < end


def title_shape(title):
    """(token count capped at 3, capitalisation code 0 lowercase / 1 mixed / 2 every word capitalised)."""
    toks = title.split()
    words = [t for t in toks if t[:1].isalpha()]
    if not words:
        return min(max(len(toks), 1), 3), 0
    up_ = sum(1 for t in words if t[:1].isupper())
    return min(max(len(toks), 1), 3), (2 if up_ == len(words) else (0 if up_ == 0 else 1))


def content_tokens(title):
    return {t for t in re.findall(r"\w+", title.lower()) if t not in STOP_WORDS}


class NodeStore:
    """A node's (title, text) by position: nodes.jsonl through a memory map and an offset table (a KB node's title is its text), or a record list."""

    def __init__(self, path=None, n=None, records=None):
        self.cache = {}
        self.records = records
        if records is not None:
            self.n = len(records)
            return
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        off = [0]
        while True:
            line = self.mm.readline()
            if not line:
                break
            off.append(self.mm.tell())
        self.off = np.asarray(off, np.int64)
        self.n = self.off.size - 1
        if n is not None and self.n != n:
            stop(f"{path}: {self.n} lines for {n} nodes")

    def get(self, i):
        i = int(i)
        v = self.cache.get(i)
        if v is None:
            rec = self.records[i] if self.records is not None else json.loads(self.mm[self.off[i]:self.off[i + 1]])
            t = rec.get("title")
            v = (t if t is not None else (rec.get("text") or ""), rec.get("text") or "")
            self.cache[i] = v
            if len(self.cache) > 400_000:
                self.cache.clear()
        return v


def recount(title, text):
    """G-G5's independent count: (count, first offset) of the title as written, word-bounded, by a regular expression."""
    pat = re.compile(r"(?<![^\W_])" + re.escape(title) + r"(?![^\W_])")
    ms = list(pat.finditer(text))
    return (len(ms), ms[0].start()) if ms else (0, -1)


def g5_check(store, a, c, ta, bits, mention_bit, limit):
    """Up to `limit` R.mention links whose title occurs exactly as written: the stored count and offset equal the regular-expression recount."""
    idx = np.flatnonzero(((bits >> mention_bit) & 1).astype(bool) & (ta["exact"] == 3))
    idx = idx[np.argsort(mix64(link_key(a[idx], c[idx], 1 << 31)), kind="stable")[:limit]]
    bad = []
    for i in idx.tolist():
        title, text = store.get(c[i])[0], store.get(a[i])[1]
        n_, off_ = recount(title, text)
        if min(n_, 255) != int(ta["cnt"][i]) or abs(off_ / max(1, len(text)) - float(ta["off"][i])) > 1e-6:
            bad.append((int(a[i]), int(c[i]), n_, int(ta["cnt"][i])))
    return int(idx.size), bad


def text_attrs(store, a, c, qtok):
    """Per link (a -> c): dir (0 none / 1 out: a's text names c / 2 in: c's text names a / 3 mutual), exact (the out-mention's level), ntok, caps, lead
    (0 no / 1 yes / 2 no out-mention), off (nan when none), cnt (0 when none), qov (content tokens of c's title in the question's tokens)."""
    m = a.size
    dr, ex = np.zeros(m, np.uint8), np.zeros(m, np.uint8)
    ntok, caps, lead = np.zeros(m, np.uint8), np.zeros(m, np.uint8), np.full(m, 2, np.uint8)
    off, cnt, qov = np.full(m, np.nan, np.float32), np.zeros(m, np.uint8), np.zeros(m, np.uint8)
    for i in range(m):
        ta, xa = store.get(a[i])
        tc, xc = store.get(c[i])
        lo, co, fo, ld = mention_stats(tc, xa)
        li = mention_stats(ta, xc)[0]
        dr[i] = (1 if lo else 0) | (2 if li else 0)
        ex[i] = lo
        ntok[i], caps[i] = title_shape(tc)
        if lo:
            lead[i] = 1 if ld else 0
            off[i] = fo / max(1, len(xa))
            cnt[i] = min(co, 255)
        qov[i] = min(len(content_tokens(tc) & qtok[i]), 255) if qtok is not None else 0
    return {"dir": dr, "exact": ex, "ntok": ntok, "caps": caps, "lead": lead, "off": off, "cnt": cnt, "qov": qov}


# ── the population: the dev carve, its seeds ─────────────────────────────────


def dev_slice(d, D, host):
    """The dev carve's questions with a gold, their seed unions, and the question texts. The first stage is the package's frozen lists (the lists PL's dev
    grids read): the dense top 5 U SPLADE top 5. WebQSP: PL's dev (pop_dev.npz). MultiHop-RAG: its dev split (PL1 S0a's lists, zul2h's own pops)."""
    if d == "multihoprag":
        pops, _gs, _src = H.make_pops(d, D)
        p = [x for x in pops if x.cv == P.BESIDE][0]
        import zul2m as M2
        root = M2.M.mhr_root()
        by_id = {str(x["query_id"]): x["question"] for x in M2.M.queries_of(root)}
        miss = [i for i in p.ids if str(i) not in by_id]
        if miss:
            stop(f"G-G1: multihoprag: {len(miss)} dev ids are not query ids (first {miss[0]!r})")
        return {"ids": p.ids, "golds": p.golds, "seeds": p.useeds, "qtext": [str(by_id[str(i)]) for i in p.ids], "dense5": p.dense5,
                "spl5": p.spl5, "carve": "dev split"}
    root = E.served_root()
    if d == "webqsp":
        pz = ROOT / "outputs" / "pl2" / "webqsp" / "pop_dev.npz"
        with np.load(pz) as z:
            ids, rows, flat, off = [str(x) for x in z["ids"]], z["idx"].astype(np.int64), z["flat"].astype(np.int64), z["off"]
        golds = [np.unique(flat[off[i]:off[i + 1]]) for i in range(len(ids))]
        eval_ids, _g, _s = UL.carve_questions("webqsp", C.DECIDE)
        if set(ids) & set(eval_ids):
            stop("G-G1: WebQSP's dev carve shares a question with s1eval")
        carve = "PL dev (pop_dev.npz)"
    else:
        try:
            ids, golds, _qsha = UL.carve_questions(d, C.BESIDE)
        except (OSError, KeyError, SystemExit) as err:
            return {"not_read": f"{d}: no dev carve (s1sel) to read: {err}"}
        qids = json.loads((root / d / "queries" / "query_ids.json").read_text(encoding="utf-8"))
        qmap = {q: j for j, q in enumerate(qids)}
        rows = np.asarray([qmap[q] for q in ids], np.int64)
        pz = ROOT / "outputs" / "pl2" / d / "pop_dev.npz"
        if pz.is_file():
            with np.load(pz) as z:
                if [str(x) for x in z["ids"]] != [str(x) for x in ids]:
                    stop(f"G-G1: {d}: the carve's ids are not PL's dev ids")
        carve = "s1sel"
    rc = root / d / "retrieval_cache"
    dense = np.asarray(np.load(rc / "dense_top1000.npz", mmap_mode="r")["ids"][rows][:, :H.TOP], np.int64)
    spl = np.asarray(np.load(rc / "splade_top1000.npz", mmap_mode="r")["ids"][rows][:, :H.TOP], np.int64)
    keep = [q for q in range(len(golds)) if np.asarray(golds[q]).size]
    import zenc
    qt = zenc.query_texts(root, d, rows[keep].tolist())
    pop = H.Pop(carve, [ids[q] for q in keep], [golds[q] for q in keep], dense[keep], spl[keep], D.n, len(golds) - len(keep))
    H.check_union(pop.useeds, pop.dense5, pop.spl5, D.n, f"{d}/{carve}")
    return {"ids": pop.ids, "golds": pop.golds, "seeds": pop.useeds, "qtext": [str(qt[int(r)]) for r in rows[keep]], "dense5": pop.dense5,
            "spl5": pop.spl5, "carve": carve}


def part_indices(nq, part):
    idx = np.arange(nq)
    return idx[idx % QUARTER_M == 0] if part == "q" else idx[idx % QUARTER_M != 0]


# ── the extraction ───────────────────────────────────────────────────────────


def label_columns(F, srk, n, a, c):
    """The link's label bits (S if SRK, R.mention, R.name_comention, R.title_comention, K.knn: zul2p.base_units' bit order), its S > R > K group
    (0 S, 1 R, 2 K) and nothing else: memberships in the sorted key arrays."""
    k = link_key(a, c, n)
    inW, inCN, inT0, inK = (UL.member(k, F[x]) for x in ("W2_S", "CN_all", "F0T", "Ke"))
    inTI = inT0 & ~inCN
    parts = ([UL.member(k, F["S"])] if srk else []) + [inW, inCN, inTI, inK]
    bits = np.zeros(k.size, np.int64)
    for i, p in enumerate(parts):
        bits |= p.astype(np.int64) << i
    grp = np.full(k.size, 2, np.uint8)
    grp[inW | inCN | inTI] = 1
    if srk:
        grp[parts[0]] = 0
    return bits.astype(np.uint8), grp


def add_stats(stats, cs):
    for k, v in cs.items():
        stats[k] = [x + y for x, y in zip(stats[k], v)] if k == "dist" else stats[k] + v


def cmd_extract(a):
    d, t0 = a.ds, time.time()
    if a.part not in ("q", "r"):
        stop("--part is q or r")
    D = P.load_mhr() if d == "multihoprag" else P.load_host(d, a.host)
    gt = P.gates(D, D.g)
    sl = dev_slice(d, D, a.host)
    if "not_read" in sl:
        (OUT / d).mkdir(parents=True, exist_ok=True)
        L2.wjson(OUT / d / f"extract_{a.part}.json", {"dataset": d, "declared_in": DECLARED, "part": a.part, "status": "NOT_READ", "why": sl["not_read"]})
        log(sl["not_read"])
        return 0
    nq = len(sl["ids"])
    sel = part_indices(nq, a.part)
    if a.limit:
        sel = sel[:a.limit]
    log(f"{d}: {nq} dev questions with a gold ({sl['carve']}), part {a.part}: {sel.size}; gates G-P1..G-P3, G-H1 pass ({time.time() - t0:.0f} s)")
    if d == "multihoprag":
        import zul2m as M2
        _root, nodes, _n, _s = M2.head()
        store = NodeStore(records=nodes)
    else:
        store = NodeStore(E.served_root() / d / "nodes.jsonl", D.n)
    rows = T3.rows_of(d, D, a.host)
    dotf = lambda keys: T3.pair_dot(rows, keys, D.n, block=DOT_BLOCK)  # noqa: E731  (UL3's dot, blocks of 16,384 pairs; each row is the same float32 sum)
    prov = H.RealProv(D)
    W = H.Walker(D.n)
    lev = np.full(D.n, DEFAULT_LEV, np.int8)
    OUT_D = OUT / d
    (OUT_D / "work").mkdir(parents=True, exist_ok=True)
    rec = {"dataset": d, "declared_in": DECLARED, "part": a.part, "status": "read", "carve": sl["carve"], "nodes": D.n, "nodes_sha256": D.nsha,
           "sources": D.src, "gates": {"UL2P": gt, "G-G1": {"questions": nq, "seeds_ok": True}}, "versions": {},
           "constants": {"CAP_SRC": CAP_SRC, "CAP_Q": CAP_Q, "CAP_GOLD": CAP_GOLD, "QUARTER_M": QUARTER_M, "ENUM_LIMIT": ENUM_LIMIT, "seeds": "dense5 U splade5, frozen package lists"}}
    qtoks = [content_tokens(t) for t in sl["qtext"]]
    zsha = script_sha256()
    for v in D.vers:
        srk = v == "SRK"
        reg = H.CompReg(prov, v, D.n)
        names = (["S"] if srk else []) + ["W", "T0", "K"]
        G = Graph([reg.get(x) for x in names], D.n)
        slab = None
        if srk and D.slab is not None:
            ks, rs, vocab = D.slab
            o = np.argsort(ks, kind="stable")
            slab = (ks[o], rs[o], list(vocab))
        stats = {"questions": int(sel.size), "golds": 0, "found": 0, "dist": [0, 0, 0, 0, 0], "capped_golds": 0, "units": 0, "ctrl_sampled": 0,
                 "ctrl_exact": 0, "sources": 0, "g_g2_checked": 0, "g5_checked": 0, "g5_bad": 0, "mention_links": 0, "mention_exact_none": 0}
        chunks = []
        for c0 in range(0, sel.size, CHUNK_Q):
            qs = sel[c0:c0 + CHUNK_Q]
            cp = OUT_D / ("work_limit" if a.limit else "work") / f"{v}_{a.part}_{c0:06d}.npz"
            if cp.is_file():
                with np.load(cp) as z:
                    same = str(z["zsha"]) == zsha
                    if same:
                        add_stats(stats, json.loads(str(z["cstats"])))
                if same:
                    chunks.append(cp)
                    continue
                log(f"{rel(cp)} was written by another version of zgp1: recomputed")
            cp.parent.mkdir(parents=True, exist_ok=True)
            cs = {k: ([0] * 5 if k == "dist" else 0) for k in stats if k not in ("questions", "found")}
            col = {k: [] for k in ("q", "a", "c", "step", "kind", "w", "dot", "rank", "votes", "tdeg")}
            gcol = {k: [] for k in ("q", "gnode", "dist", "capped", "nunits")}
            inc_g, inc_row, row_base, gold_base = [], [], 0, 0
            for qi in qs.tolist():
                seeds, golds = sl["seeds"][qi], sl["golds"][qi]
                r = question_records(G, W, lev, seeds, golds, dotf)
                reach, _nn = W.walk([reg.get(x) for x in names], seeds, golds, False)   # G-G2: my distances equal UL2H's walker's reach
                walk_d = np.where(reach[0], 0, np.where(reach[1], 1, np.where(reach[2], 2, np.where(reach[3], 3, 4))))
                if not np.array_equal(walk_d, r["dist"]):
                    stop(f"G-G2: {d}/{v}: question {qi}: distances {r['dist'].tolist()} against the walker's {walk_d.tolist()}")
                cs["g_g2_checked"] += 1
                k = r["a"].size
                for x in ("a", "c", "step", "kind", "w", "dot", "rank", "votes", "tdeg"):
                    col[x].append(r[x])
                col["q"].append(np.full(k, qi, np.int32))
                gcol["q"].append(np.full(golds.size, qi, np.int32))
                gcol["gnode"].append(golds)
                gcol["dist"].append(r["dist"])
                gcol["capped"].append(r["capped"])
                nun = np.bincount(r["inc_g"], minlength=golds.size) if r["inc_g"].size else np.zeros(golds.size, np.int64)
                gcol["nunits"].append(nun)
                inc_g.append(r["inc_g"] + gold_base)
                inc_row.append(r["inc_k"] + row_base)
                row_base += k
                gold_base += golds.size
                cs["golds"] += int(golds.size)
                cs["dist"] = [x + int((r["dist"] == j).sum()) for j, x in enumerate(cs["dist"])]
                cs["capped_golds"] += int(r["capped"].sum())
                cs["units"] += r["n_units"]
                cs["ctrl_sampled"] += r["stats"]["ctrl_sampled"]
                cs["ctrl_exact"] += r["stats"]["ctrl_exact"]
                cs["sources"] += r["stats"]["sources"]
            cat = {k: (np.concatenate(v_) if v_ else np.zeros(0)) for k, v_ in col.items()}
            qq = np.concatenate([np.full(x.size, i, np.int64) for i, x in zip(qs.tolist(), col["a"])]) if col["a"] else EMPTY
            qtok = [qtoks[i] for i in qq.tolist()]
            bits, grp = label_columns(D.F, srk, D.n, cat["a"], cat["c"])
            ta = text_attrs(store, cat["a"], cat["c"], qtok)
            mb = 1 if srk else 0
            ck, bad = g5_check(store, cat["a"], cat["c"], ta, bits.astype(np.int64), mb, G5_PER_CHUNK)
            cs["g5_checked"] += ck
            cs["g5_bad"] += len(bad)
            if bad:
                stop(f"G-G5: {d}/{v}: the mention count or offset differs from the regular-expression recount: {bad[:3]}")
            ism = ((bits >> mb) & 1).astype(bool)
            cs["mention_links"] += int(ism.sum())
            cs["mention_exact_none"] += int((ism & (ta["exact"] == 0)).sum())
            relid = np.full(cat["a"].size, -1, np.int16)
            if slab is not None and cat["a"].size:
                kk = link_key(cat["a"], cat["c"], D.n)
                pos = np.minimum(np.searchsorted(slab[0], kk), slab[0].size - 1)
                hit = slab[0][pos] == kk
                relid[hit] = slab[1][pos[hit]]
            np.savez_compressed(cp, q=cat["q"].astype(np.int32), a=cat["a"].astype(np.int32), c=cat["c"].astype(np.int32), step=cat["step"].astype(np.uint8),
                                kind=cat["kind"].astype(np.uint8), w=cat["w"].astype(np.float32), dot=cat["dot"].astype(np.float32),
                                rank=cat["rank"].astype(np.int32), votes=cat["votes"].astype(np.int32), tdeg=cat["tdeg"].astype(np.int32), prof=bits, grp=grp,
                                rel=relid, **ta, gq=np.concatenate(gcol["q"]), gnode=np.concatenate(gcol["gnode"]).astype(np.int32),
                                gdist=np.concatenate(gcol["dist"]).astype(np.uint8), gcapped=np.concatenate(gcol["capped"]).astype(np.uint8),
                                gunits=np.concatenate(gcol["nunits"]).astype(np.int32), inc_g=np.concatenate(inc_g) if inc_g else EMPTY,
                                inc_row=np.concatenate(inc_row) if inc_row else EMPTY, cstats=np.array(json.dumps(cs)), zsha=np.array(zsha))
            add_stats(stats, cs)
            chunks.append(cp)
            log(f"{d}/{v}/{a.part}: {min(c0 + CHUNK_Q, sel.size)}/{sel.size} questions ({time.time() - t0:.0f} s)")
        stats["found"] = stats["dist"][0]
        stats["reachable_missed"] = int(sum(stats["dist"][1:4]))
        stats["unreached"] = stats["dist"][4]
        rec["versions"][v] = {"stats": stats, "chunks": [rel(x) for x in chunks]}
        if stats["reachable_missed"] and stats["capped_golds"] / stats["reachable_missed"] > 0.05:
            rec["versions"][v]["flag"] = "CAPPED"
        reg.c.clear()
    rec["memory"] = C.peak_gb()
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zgp1_sha256=script_sha256(),
               zul2h_sha256=H.script_sha256(), zul3a_sha256=T3.script_sha256())
    fn = OUT_D / (f"extract_{a.part}_limit.json" if a.limit else f"extract_{a.part}.json")
    L2.wjson(fn, rec)
    log(f"{d}: {rel(fn)} ({rec['seconds']} s)")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def toy_graph(rng, n, m):
    a = rng.integers(0, n, m)
    b = rng.integers(0, n, m)
    k = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    k = k[(k // n) < (k % n)]
    return k


def brute_dag(adj, seeds, g, d):
    """All shortest paths of length d from a seed to g by explicit enumeration: the set of (lower-level node, higher-level node, step)."""
    out = set()

    def rec(path):
        if len(path) == d + 1:
            if path[-1] == g:
                for i in range(d):
                    out.add((path[i], path[i + 1], i + 1))
            return
        for x in adj[path[-1]]:
            if x not in path:
                rec(path + [x])
    for s in seeds:
        rec([s])
    return out


def brute_dist(adj, seeds, g, n):
    dist = {s: 0 for s in seeds}
    fr = list(seeds)
    h = 0
    while fr and h < 6:
        h += 1
        nf = []
        for u in fr:
            for x in adj[u]:
                if x not in dist:
                    dist[x] = h
                    nf.append(x)
        fr = nf
    return dist.get(g, 99)


def selftest():
    rng = np.random.default_rng(0)
    # mention_stats and title_shape
    assert mention_stats("Paris", "He lives in Paris. Paris is large.") == (3, 2, 12, False) or mention_stats("Paris", "He lives in Paris. Paris is large.")[0] == 3
    lv, cn, fo, ld = mention_stats("Paris", "Paris is the capital. It is big.")
    assert (lv, cn, fo, ld) == (3, 1, 0, True), (lv, cn, fo, ld)
    assert mention_stats("paris", "Visit Paris.")[0] == 2
    assert mention_stats("John Smith (actor)", "John Smith was born. John Smith")[0] == 1
    assert mention_stats("Parisian", "Visit Paris.")[0] == 0 and mention_stats("Paris", "Parisian food")[0] == 0
    assert mention_stats("Alice Cooper", "later, Cooper wrote")[0] == 1
    assert title_shape("New York City") == (3, 2) and title_shape("the old man") == (3, 0) and title_shape("Mont Blanc de") == (3, 1)
    # the shortest-path links and the levels equal brute force; a wrong DAG is caught
    bad_total = wrong_caught = 0
    for trial in range(120):
        n = int(rng.integers(8, 28))
        keys = toy_graph(rng, n, int(rng.integers(n, 4 * n)))
        if keys.size < 4:
            continue
        comp = H.build_comp(keys, n)
        G = Graph([comp], n)
        adj = {i: set() for i in range(n)}
        for k in keys.tolist():
            adj[k // n].add(k % n)
            adj[k % n].add(k // n)
        W = H.Walker(n)
        lev = np.full(n, DEFAULT_LEV, np.int8)
        seeds = np.unique(rng.integers(0, n, int(rng.integers(1, 4))))
        golds = np.unique(rng.integers(0, n, int(rng.integers(1, 5))))
        L1, L2_ = levels(W, G, lev, seeds)
        for g in golds.tolist():
            d = gold_distance(G, lev, g)
            bd = brute_dist(adj, seeds.tolist(), g, n)
            assert d == (bd if bd <= 3 else 4), (trial, g, d, bd)
            if 1 <= d <= 3:
                A, Cc, S, cp = gold_units(G, lev, g, d)
                got = {(int(x), int(y), int(s)) for x, y, s in zip(A.tolist(), Cc.tolist(), S.tolist())}
                want = brute_dag(adj, seeds.tolist(), g, d)
                assert got == want, (trial, g, d, got ^ want)
                longer = brute_dag(adj, seeds.tolist(), g, d + 1) if d < 3 else set()
                if longer - want:
                    wrong_caught += 1
                bad_total += 1
        reset(W, lev, seeds, L1, L2_)
        assert not lev.astype(bool).sum() == 0 or bool((lev == DEFAULT_LEV).all())
        assert not bool(W.vis.any())
    assert bad_total > 100 and wrong_caught > 5, (bad_total, wrong_caught)
    # question_records: weights, controls, the gates; the walker's reach equals my distances
    for trial in range(60):
        n = int(rng.integers(10, 40))
        keys = toy_graph(rng, n, int(rng.integers(2 * n, 5 * n)))
        if keys.size < 6:
            continue
        comp = H.build_comp(keys, n)
        G = Graph([comp], n)
        W = H.Walker(n)
        lev = np.full(n, DEFAULT_LEV, np.int8)
        seeds = np.unique(rng.integers(0, n, int(rng.integers(1, 4))))
        golds = np.unique(rng.integers(0, n, int(rng.integers(1, 5))))
        dotf = lambda kk: ((kk * 2654435761) % 1000).astype(np.float32) / 1000.0  # noqa: E731
        r = question_records(G, W, lev, seeds, golds, dotf, cap_src=3, cap_q=12)
        reach, _ = W.walk([comp], seeds, golds, False)
        wd = np.where(reach[0], 0, np.where(reach[1], 1, np.where(reach[2], 2, np.where(reach[3], 3, 4))))
        assert np.array_equal(wd, r["dist"]), (wd, r["dist"])
        nd = r["kind"] == 1
        if nd.any():
            ng = len({gi for gi in r["inc_g"].tolist()})
            assert abs(float(r["w"][nd].sum()) - ng) < 1e-9
            assert (r["rank"][nd] >= 1).all() and (r["votes"][nd] >= 1).all()
        assert bool((lev == DEFAULT_LEV).all()) and not bool(W.vis.any())
    log("selftest PASS: mention_stats, the shortest-path links equal brute force (a longer-path DAG is caught), levels equal the walker, weights, control gates")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    x = sub.add_parser("extract")
    x.add_argument("--ds", required=True)
    x.add_argument("--part", required=True)
    x.add_argument("--host", action="store_true")
    x.add_argument("--threads", type=int, default=1)
    x.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    return cmd_extract(a)


if __name__ == "__main__":
    sys.exit(main())
