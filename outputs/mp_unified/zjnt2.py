"""JNT1 amendment 2 (docs/JNT1_LEARNED_WALK.md section 11; zjnt1.py stays frozen and its old-seed chain is a labelled screen): a learned link score (per-hop one-hidden-layer MLPs, no message passing) orders a budgeted three-hop walk of the any-link graph
(S, W, T0; K dropped) from the e3v dense top-5 seeds (computed here from the stored e3v rows, never the older encoder's cached lists), against the question-cosine order (B1) and the flat e3v list (B0), at equal widths.

    python outputs/mp_unified/zjnt2.py selftest                                   # the walker, the labels, the features and the MLP on toy graphs (no real data)
    python outputs/mp_unified/zjnt2.py probe  --ds D [--host] [--n 20] [--v V]     # prices the walk on the first read questions (B1 arm at the five triples), no label read
    python outputs/mp_unified/zjnt2.py labels --ds D --round R [--host] [--v V] [--read-n 300] [--fit-n 2000]   # candidate tables of the fit questions (round 1 needs round 0's model)
    python outputs/mp_unified/zjnt2.py fit    --ds D --round R [--shuffle]         # fits the MLP on the tables
    python outputs/mp_unified/zjnt2.py read   --ds D [--host] [--v V] [--read-n 300] [--shuffle]   # B0, B1, L0, L1 at the five triples on the read questions, the calls
    python outputs/mp_unified/zjnt2.py read   --ds D [--host] [--v V] --b-only [--seeds dense|dense+splade|dense+qn|dense+qg|dense+splade+qg|all]      # B0 and B1 only (no model): the pruned parameter-free pool of a dataset

The walk reads no label: golds enter the labels of the FIT questions only (to fit) and the read's grading (after the pools are fixed). Records: outputs/jnt1/<dataset>/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zgp3 as G3  # noqa: E402  (first: imports zgp1, which sets the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import re  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

G1, H, T3, P, E = G3.G1, G3.H, G3.T3, G3.P, G3.G1.E
DECLARED = "docs/JNT1_LEARNED_WALK.md"
OUT = ROOT / "outputs" / "jnt2"
EMPTY = np.zeros(0, np.int64)
WIDTHS = ((10, 20, 30), (20, 60, 120), (40, 150, 400), (80, 400, 1500), (150, 900, 3000))   # section 2: fixed
FIT_TRIPLE = 2                                         # A1.2: the triple whose walk generates the training candidates
CELL_V = {"multihoprag": "RK", "musique": "RK", "squad": "RK", "metaqa": "SRK", "webqsp": "SRK", "hotpotqa": "SRK", "2wiki": "SRK"}   # GP3's cells; hotpotqa/2wiki also have RK (--v)
BIT = {"S": 1, "W": 2, "T0": 4}
GATHER_CAP = G3.GATHER_CAP
NEG_PER, POS_CAP, CHUNK_Q = 200, 1000, 20              # A1.2
HID, EPOCHS, LR, WD, BATCH, SEED = 64, 40, 3e-3, 1e-5, 4096, 20261127
NFEAT = 14
BLK = 20000
MIN_HEAD_ROWS = 2000
log = G1.log


def stop(msg):
    raise SystemExit(f"[zjnt2] STOP: {msg}")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def in_sorted(x, s):
    if s.size == 0 or x.size == 0:
        return np.zeros(x.size, bool)
    p = np.minimum(np.searchsorted(s, x), s.size - 1)
    return s[p] == x


# ── the walker (graph-agnostic: a context with n, graphs{name: gather}, rows.take, rows.dim) ───────────────────


class Ctx:
    def __init__(self, n, graphs, rows):
        self.n, self.graphs, self.rows = n, graphs, rows
        self.thin = False


def links_of(ctx, parents):
    """The distinct links s -> x of the parents over the components, with the OR of the component bits of each link."""
    n = ctx.n
    ks, fl = [], []
    for nm, g in ctx.graphs.items():
        s, x, th = g.gather(parents, limit=GATHER_CAP)
        ctx.thin = ctx.thin or bool(th)
        if s.size:
            ks.append(s * n + x)
            fl.append(np.full(s.size, BIT[nm], np.int8))
    if not ks:
        return EMPTY, EMPTY, np.zeros(0, np.int8)
    k = np.concatenate(ks)
    f = np.concatenate(fl)
    o = np.argsort(k, kind="stable")
    k, f = k[o], f[o]
    st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    return k[st] // n, k[st] % n, np.bitwise_or.reduceat(f, st)


def top_w(cand, sc, w):
    """Indices of the w best candidates by (-score, node id)."""
    if w >= cand.size:
        return np.lexsort((cand, -sc))
    thr = np.partition(sc, sc.size - w)[sc.size - w]
    ix = np.flatnonzero(sc >= thr)
    return ix[np.lexsort((cand[ix], -sc[ix]))][:w]


def cosq(ctx, cand, qv):
    out = np.empty(cand.size, np.float32)
    for b in range(0, cand.size, 100_000):
        out[b:b + 100_000] = ctx.rows.take(cand[b:b + 100_000]) @ qv
    return out


def groups(inv, nc):
    """order and starts of the candidates' groups of links (every candidate has a link)."""
    order = np.argsort(inv, kind="stable")
    si = inv[order]
    st = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
    assert st.size == nc
    return order, st


def feats(ctx, qv, h, s, x, f, cand, inv, ps, use_ps):
    """The 14 label-free features of the candidates of hop h (section 2, A1.3, A2.3). ps = (sorted kept nodes of hop h-1, their scores) or None."""
    nc = cand.size
    qc = cosq(ctx, cand, qv)
    parents, pinv = np.unique(s, return_inverse=True)
    Xp = ctx.rows.take(parents)
    qcp = Xp @ qv
    dl = np.empty(s.size, np.float32)
    for b in range(0, s.size, BLK):
        dl[b:b + BLK] = np.einsum("ij,ij->i", Xp[pinv[b:b + BLK]], ctx.rows.take(x[b:b + BLK]))
    order, st = groups(inv, nc)
    gmax = lambda v: np.maximum.reduceat(v[order], st)   # noqa: E731
    gmin = lambda v: np.minimum.reduceat(v[order], st)   # noqa: E731
    pc_max = gmax(dl)
    qp_max = gmax(qcp[pinv])
    r_all = np.empty(nc, np.int64)
    r_all[np.lexsort((cand, -qc))] = np.arange(1, nc + 1)
    ql = qc[inv]
    o = np.lexsort((x, -ql, s))
    ss = s[o]
    sst = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    first = np.repeat(sst, np.diff(np.r_[sst, ss.size]))
    rk = np.empty(s.size, np.int64)
    rk[o] = np.arange(s.size) - first + 1
    r_src = gmin(rk)
    od = np.lexsort((x, -dl, s))                        # A2.3: the source-relative rank by the pair dot (GP1's rank), beside the question-cosine one
    sd_ = s[od]
    sdt = np.flatnonzero(np.r_[True, sd_[1:] != sd_[:-1]])
    fd = np.repeat(sdt, np.diff(np.r_[sdt, sd_.size]))
    rkd = np.empty(s.size, np.int64)
    rkd[od] = np.arange(s.size) - fd + 1
    r_dot = gmin(rkd)
    fl = np.bitwise_or.reduceat(f[order], st)           # A2.3: the component bits are independent (a link held by S and W has both), no priority
    npar = np.diff(np.r_[st, inv.size]).astype(np.float32)
    if use_ps and ps is not None and ps[0].size:
        p = np.minimum(np.searchsorted(ps[0], s), ps[0].size - 1)
        pv = np.where(ps[0][p] == s, ps[1][p], 0.0).astype(np.float32)
        psb = gmax(pv)
    else:
        psb = np.zeros(nc, np.float32)
    F = np.zeros((nc, NFEAT), np.float32)
    F[:, 0], F[:, 1], F[:, 2] = qc, pc_max, qp_max
    F[:, 3], F[:, 4] = np.log1p(r_all), np.log1p(r_src)
    F[:, 5], F[:, 6], F[:, 7] = (fl & 1) > 0, (fl & 2) > 0, (fl & 4) > 0
    F[:, 7 + h] = 1.0                                   # columns 8, 9, 10 = hop 1, 2, 3
    F[:, 11] = np.log1p(npar)
    F[:, 12] = psb
    F[:, 13] = np.log1p(r_dot)
    return F, qc


def walk(ctx, qv, seeds, widths, model=None, use_ps=False, collect=None):
    """The three-hop walk. model None = B1 (score = the question cosine), else the MLP's logit. collect(h, F, cand) is called per hop (the tables).
    Returns the pool (sorted distinct nodes) and the kept nodes per hop."""
    seeds = np.unique(np.asarray(seeds, np.int64))
    pool, parents, ps, kept_h = seeds, seeds, None, []
    for h, w in enumerate(widths, 1):
        s, x, f = links_of(ctx, parents) if parents.size else (EMPTY, EMPTY, np.zeros(0, np.int8))
        m = ~in_sorted(x, pool)
        s, x, f = s[m], x[m], f[m]
        if x.size == 0:
            kept_h.append(EMPTY)
            parents = EMPTY
            continue
        cand, inv = np.unique(x, return_inverse=True)
        if model is None and collect is None:
            sc = cosq(ctx, cand, qv)
        else:
            F, qc = feats(ctx, qv, h, s, x, f, cand, inv, ps, use_ps)
            sc = qc if model is None else model.logit(F)
            if collect is not None:
                collect(h, F, cand)
        o = top_w(cand, sc, w)
        kept = cand[o]
        kept_h.append(kept)
        so = np.argsort(kept, kind="stable")
        ps = (kept[so], sc[o][so].astype(np.float32))
        pool = np.union1d(pool, kept)
        parents = kept
    return pool, kept_h


def gold_sets(ctx, golds):
    """B0 = the golds, B1 = B0 and their neighbours, B2 = B1 and theirs (any-link graph): a candidate of hop h is positive if it lies in B(3-h)."""
    b0 = np.unique(np.asarray(golds, np.int64))
    out = [b0]
    cur = b0
    for _ in range(2):
        s, x, _f = links_of(ctx, cur)
        cur = np.union1d(cur, x)
        out.append(cur)
    return out


# ── the MLP (numpy; one hidden layer; logistic loss; Adam) ─────────────────────────────────────────────────────────


class MLP:
    def __init__(self, W1, b1, W2, b2, mu, sd):
        self.W1, self.b1, self.W2, self.b2, self.mu, self.sd = W1, b1, W2, b2, mu, sd

    def logit(self, F):
        z = (F - self.mu) / self.sd
        return (np.maximum(z @ self.W1 + self.b1, 0.0) @ self.W2 + self.b2).astype(np.float32)

    def save(self, p):
        np.savez(p, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, p):
        with np.load(p) as z:
            return cls(*(z[k] for k in ("W1", "b1", "W2", "b2", "mu", "sd")))


def fit_mlp(X, y, seed=SEED, epochs=EPOCHS, lr=LR, wd=WD, batch=BATCH, hid=HID):
    rng = np.random.default_rng(seed)
    mu = X.mean(0).astype(np.float32)
    sd = (X.std(0) + 1e-6).astype(np.float32)
    Z = ((X - mu) / sd).astype(np.float32)
    y = y.astype(np.float32)
    W1 = (rng.standard_normal((X.shape[1], hid)) * np.sqrt(2.0 / X.shape[1])).astype(np.float32)
    b1 = np.zeros(hid, np.float32)
    W2 = (rng.standard_normal(hid) * np.sqrt(1.0 / hid)).astype(np.float32)
    b2 = np.float32(0.0)
    params = [W1, b1, W2, np.array(b2, np.float32)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    n = Z.shape[0]
    for ep in range(epochs):
        perm = rng.permutation(n)
        loss = 0.0
        for b in range(0, n, batch):
            ix = perm[b:b + batch]
            z, yy = Z[ix], y[ix]
            a = np.maximum(z @ params[0] + params[1], 0.0)
            lg = a @ params[2] + params[3]
            p = 1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30)))
            loss += float(-(yy * np.log(p + 1e-9) + (1 - yy) * np.log(1 - p + 1e-9)).sum())
            dl = (p - yy) / ix.size
            g2 = a.T @ dl
            gb2 = np.array(dl.sum(), np.float32)
            da = np.outer(dl, params[2]) * (a > 0)
            g1 = z.T @ da
            gb1 = da.sum(0)
            grads = [g1 + wd * params[0], gb1, g2 + wd * params[2], gb2]
            t += 1
            for i, (pp, g) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                pp -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        if ep in (0, epochs - 1):
            log(f"  fit epoch {ep + 1}/{epochs}: mean logistic loss {loss / n:.4f} ({n:,} rows)")
    return MLP(params[0], params[1], params[2], np.float32(params[3]), mu, sd)


class HopModel:
    """A2.4: one MLP per hop (the candidates of hop h are scored by head h, which sees only the rows of its hop); the pooled model is the fallback of a hop with too few rows."""

    def __init__(self, heads, pooled):
        self.heads, self.pooled = heads, pooled

    def logit(self, F):
        h = int(np.argmax(F[0, 8:11])) + 1 if F.shape[0] else 1
        return (self.heads.get(h) or self.pooled).logit(F)

    def save(self, p):
        d = {}
        for k, m in (("p", self.pooled), ("h1", self.heads.get(1)), ("h2", self.heads.get(2)), ("h3", self.heads.get(3))):
            if m is not None:
                for nm in ("W1", "b1", "W2", "b2", "mu", "sd"):
                    d[f"{k}_{nm}"] = getattr(m, nm)
        np.savez(p, **d)

    @classmethod
    def load(cls, p):
        with np.load(p) as z:
            def get(k):
                return MLP(*(z[f"{k}_{nm}"] for nm in ("W1", "b1", "W2", "b2", "mu", "sd"))) if f"{k}_W1" in z.files else None
            heads = {h: get(f"h{h}") for h in (1, 2, 3) if get(f"h{h}") is not None}
            return cls(heads, get("p"))


def auc_of(lg, y, seed=SEED):
    ix = np.random.default_rng(seed).choice(lg.size, min(200_000, lg.size), replace=False)
    yy, pp = y[ix], lg[ix]
    npos = int(yy.sum())
    if npos == 0 or npos == yy.size:
        return None
    ranks = np.empty(pp.size)
    ranks[np.argsort(pp, kind="stable")] = np.arange(1, pp.size + 1)
    return round(float((ranks[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * (pp.size - npos))), 4)


# ── toy graphs for the self-test ──────────────────────────────────────────────────────────────────────────────────


class ToyRows:
    def __init__(self, X):
        self.X, self.dim = X, X.shape[1]

    def take(self, idx):
        return self.X[np.asarray(idx, np.int64)]


class ToyGraph:
    def __init__(self, n, edges):
        self.adj = [[] for _ in range(n)]
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)

    def gather(self, nodes, limit=None):
        s, x = [], []
        for a in np.unique(np.asarray(nodes, np.int64)).tolist():
            for b in self.adj[a]:
                s.append(a)
                x.append(b)
        return np.asarray(s, np.int64), np.asarray(x, np.int64), False


def toy_ctx(rng, n=70, dim=8):
    X = rng.standard_normal((n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    graphs = {}
    for nm, k in (("S", 60), ("W", 80), ("T0", 50)):
        ed = [(int(a), int(b)) for a, b in rng.integers(0, n, size=(k, 2)) if a != b]
        graphs[nm] = ToyGraph(n, ed)
    return Ctx(n, graphs, ToyRows(X))


def brute_walk(ctx, qv, seeds, widths):
    n = ctx.n
    pool = set(int(a) for a in seeds)
    parents = set(pool)
    for w in widths:
        cand = set()
        for nm, g in ctx.graphs.items():
            for a in parents:
                for b in g.adj[a]:
                    if b not in pool:
                        cand.add(b)
        sc = {c: float(ctx.rows.X[c] @ qv) for c in cand}
        kept = sorted(cand, key=lambda c: (-sc[c], c))[:w]
        pool |= set(kept)
        parents = set(kept)
    return sorted(pool)


def brute_ball(ctx, seeds, k):
    ball = set(int(a) for a in seeds)
    fr = set(ball)
    for _ in range(k):
        nf = set()
        for nm, g in ctx.graphs.items():
            for a in fr:
                nf.update(b for b in g.adj[a] if b not in ball)
        ball |= nf
        fr = nf
    return ball


def selftest():
    rng = np.random.default_rng(1)
    trials = 0
    for _ in range(40):
        ctx = toy_ctx(rng)
        qv = rng.standard_normal(8).astype(np.float32)
        qv /= np.linalg.norm(qv)
        seeds = rng.choice(ctx.n, size=int(rng.integers(1, 5)), replace=False)
        big = walk(ctx, qv, seeds, (10 ** 6,) * 3)[0]
        assert sorted(brute_ball(ctx, seeds, 3)) == big.tolist(), "G-J0: the walker with huge widths is not the 3-hop BFS ball"
        for wd in ((1, 1, 1), (2, 3, 4), (5, 2, 7), (3, 3, 3), (10, 20, 30)):
            got = walk(ctx, qv, seeds, wd)[0].tolist()
            assert got == brute_walk(ctx, qv, seeds, wd), f"G-J0: B1 pool differs from the brute-force top-w walk at {wd}"
            got2 = walk(ctx, qv, seeds, wd, collect=lambda h, F, c: None)[0].tolist()   # the feature path scores by the same cosine
            assert got2 == got, "G-J0: the feature path changes the B1 pool"
        trials += 1
    # top_w ties
    c = np.arange(10, dtype=np.int64)
    sc = np.array([1, 1, 0.5, 1, 0.2, 0.5, 1, 0.9, 0.9, 0.5], np.float32)
    for w in (1, 3, 5, 10, 12):
        exp = sorted(range(10), key=lambda i: (-sc[i], i))[:w]
        assert c[top_w(c, sc, w)].tolist() == exp, "top_w ties"
    # labels: B sets are the distance balls of the golds
    for _ in range(20):
        ctx = toy_ctx(rng)
        g = rng.choice(ctx.n, size=int(rng.integers(1, 4)), replace=False)
        b = gold_sets(ctx, g)
        for k in range(3):
            assert b[k].tolist() == sorted(brute_ball(ctx, g, k)), "gold ball"
    # features: shapes, finite, label-free (the same features with different golds), hop columns, S priority, round 0 ps column
    ctx = toy_ctx(rng)
    qv = rng.standard_normal(8).astype(np.float32)
    qv /= np.linalg.norm(qv)
    seeds = np.array([0, 1, 2])
    store = {}
    walk(ctx, qv, seeds, (6, 8, 10), collect=lambda h, F, cand: store.__setitem__(h, (F.copy(), cand.copy())))
    for h, (F, cand) in store.items():
        assert F.shape == (cand.size, NFEAT) and np.isfinite(F).all(), "features finite"
        assert (F[:, 7 + h] == 1).all() and F[:, 8:11].sum() == cand.size, "hop one-hot"
        assert ((F[:, 5] + F[:, 6] + F[:, 7]) >= 1).all(), "every candidate has a component bit"
        assert (F[:, 13] >= np.log(2) - 1e-6).all(), "pair-dot source rank is at least 1"
        assert (F[:, 12] == 0).all(), "round 0 ps column is 0"
        assert np.allclose(F[:, 0], ctx.rows.X[cand] @ qv, atol=1e-6), "question cosine column"
        assert (F[:, 3] >= 0).all() and F[:, 11].min() >= np.log(2) - 1e-6, "ranks and parent counts"
    if 1 in store:                                      # hop 1: the component bits equal a brute-force check against the seeds, independently
        F1, c1 = store[1]
        for nm, col in (("S", 5), ("W", 6), ("T0", 7)):
            exp = np.array([any(int(c) in ctx.graphs[nm].adj[int(s0)] for s0 in seeds) for c in c1])
            assert (F1[:, col] == exp).all(), f"component bit {nm}"
    # the MLP learns a separable toy and logit equals the manual forward
    X = rng.standard_normal((4000, NFEAT)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 3] > 0.3).astype(np.float32)
    mdl = fit_mlp(X, y, epochs=25, lr=2e-2, batch=512)
    lg = mdl.logit(X)
    acc = float(((lg > 0) == (y > 0)).mean())
    assert acc > 0.9, f"MLP toy accuracy {acc}"
    z = (X[:5] - mdl.mu) / mdl.sd
    man = np.maximum(z @ mdl.W1 + mdl.b1, 0) @ mdl.W2 + mdl.b2
    assert np.allclose(man, mdl.logit(X[:5]), atol=1e-5), "logit forward"
    # a model that is the question cosine reproduces B1
    class Cos(MLP):
        def __init__(self):
            pass

        def logit(self, F):
            return F[:, 0]

    for wd in ((2, 3, 4), (10, 20, 30)):
        a = walk(ctx, qv, seeds, wd)[0]
        b = walk(ctx, qv, seeds, wd, model=Cos(), use_ps=True)[0]
        assert a.tolist() == b.tolist(), "a model equal to the cosine reproduces B1"
    import tempfile
    # A2.5 the question-name rule on a toy: unique titles are found longest first, an ambiguous title and a stop-like token are not
    global node_titles
    saved = node_titles
    node_titles = lambda d_: iter(["The Godfather", "Godfather (film)", "Paris", "Paris (film)", "Mount Everest", "Film", "Kismet (1944 film)", "Kismet (2020 film)", "Obama"])
    slq = {"ids": ["a", "b", "c", "d"], "qtext": ["Who directed the godfather in film?", "Where is mount everest and paris located in film?", "Which film did obama like?", "Which film was filed in Kismet?"]}
    with tempfile.TemporaryDirectory() as td:
        got, grp, _inf = qn_extra("toy", slq, 9, Path(td))
    node_titles = saved
    assert got[0].tolist() == [0], got[0]                                    # 'the godfather' is node 0 whole; 'Godfather (film)' does not make it ambiguous
    assert got[1].tolist() == [2, 4], got[1]                                 # 'mount everest' (node 4) and 'paris' (node 2: the whole title wins over a bracketed one)
    assert got[2].tolist() == [8], got[2]                                    # 'film' is in three questions (a stop-like token), 'obama' is a name
    assert got[3].size == 0, got[3]                                          # two nodes have the stripped title 'kismet': not a seed
    assert grp[3].tolist() == [6, 7], grp[3]                                 # ... they are a group, kept apart for the bounded `qg` variant
    assert all(g.size == 0 for g in grp[:3]), grp
    heads = {h: fit_mlp(X[h * 1000:(h + 1) * 1000 + 500], y[h * 1000:(h + 1) * 1000 + 500], epochs=3, batch=256, seed=SEED + h) for h in (1, 2)}
    hm = HopModel(heads, mdl)
    with tempfile.TemporaryDirectory() as td:
        hm.save(Path(td) / "m.npz")
        hm2 = HopModel.load(Path(td) / "m.npz")
    for h in (1, 2, 3):
        Fh = X[:50].copy()
        Fh[:, 8:11] = 0
        Fh[:, 7 + h] = 1
        assert np.allclose(hm.logit(Fh), hm2.logit(Fh), atol=1e-6), "HopModel round trip"
        assert np.allclose(hm.logit(Fh), (heads.get(h) or mdl).logit(Fh), atol=1e-6), "HopModel dispatch"
    log(f"selftest PASS: {trials} toy graphs: huge widths = the 3-hop BFS ball, 5 width triples = the brute-force top-w walk, the feature path = the cosine path; top_w ties; gold balls = BFS; features finite, independent component bits (= brute force at hop 1), label-free; per-hop model round trip and dispatch; the question-name rule on a toy; MLP forward = manual (toy accuracy {acc:.3f})")
    return 0


# ── the real context ──────────────────────────────────────────────────────────────────────────────────────────────


def real_ctx(d, host, v):
    D = P.load_mhr() if d == "multihoprag" else P.load_host(d, host)
    reg = H.CompReg(H.RealProv(D), v, D.n)
    names = (["S"] if v == "SRK" else []) + ["W", "T0"]
    graphs = {nm: G1.Graph([reg.get(nm)], D.n) for nm in names}
    rows = T3.rows_of(d, D, host)
    ctx = Ctx(D.n, graphs, rows)
    allg = G1.Graph([reg.get(nm) for nm in names], D.n)
    return D, ctx, allg


def question_vectors(d, sl):
    """The e3v question vectors of the slice's questions (unit norm), by the package's query order."""
    if d == "multihoprag":
        import zul2m as M2
        root, _nodes, _n, _sha = M2.head()
        queries = M2.M.queries_of(M2.M.mhr_root())
        pos = {str(x["query_id"]): j for j, x in enumerate(queries)}
    else:
        root = E.served_root()
        qids = json.loads((root / d / "queries" / "query_ids.json").read_text(encoding="utf-8"))
        pos = {str(q): j for j, q in enumerate(qids)}
    maps = [np.load(p, mmap_mode="r") for p in E.vec_shards(root, "e3v", d, "queries")]
    starts = np.cumsum([0] + [int(m.shape[0]) for m in maps])
    out = np.empty((len(sl["ids"]), int(maps[0].shape[1])), np.float32)
    for i, q in enumerate(sl["ids"]):
        j = pos[str(q)]
        k = int(np.searchsorted(starts, j, side="right") - 1)
        out[i] = np.asarray(maps[k][j - starts[k]], np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def flat_topk(rows, Q, k, block=1 << 18, qb=256):
    """Per question the k nodes of the highest question cosine, ordered by (-cosine, id): one pass over the stored e3v rows per group of qb questions."""
    if Q.shape[0] > qb:
        return np.concatenate([flat_topk(rows, Q[i:i + qb], k, block, qb) for i in range(0, Q.shape[0], qb)])
    nq = Q.shape[0]
    bs = np.full((nq, k), -np.inf, np.float32)
    bi = np.zeros((nq, k), np.int64)
    for si, mp in enumerate(rows.maps):
        base = int(rows.starts[si])
        for b0 in range(0, int(mp.shape[0]), block):
            blk = np.asarray(mp[b0:b0 + block], np.float32)
            S = Q @ blk.T
            ids = np.arange(base + b0, base + b0 + blk.shape[0], dtype=np.int64)
            kk = min(k, blk.shape[0])
            part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            cs = np.take_along_axis(S, part, 1)
            ci = ids[part]
            ms = np.concatenate([bs, cs], 1)
            mi = np.concatenate([bi, ci], 1)
            top = np.argpartition(-ms, k - 1, axis=1)[:, :k]
            bs = np.take_along_axis(ms, top, 1)
            bi = np.take_along_axis(mi, top, 1)
    out = np.empty((nq, k), np.int64)
    for q in range(nq):
        out[q] = bi[q][np.lexsort((bi[q], -bs[q]))]
    return out


def seeds_e3v(ctx, sl, Q, out_d):
    """A2.1: the seeds of every arm are the e3v dense top 5 of the question (the defined encoder), by one exact pass over the stored e3v rows for all the slice's questions;
    cached with the question ids' sha. The older encoder's cached lists (retrieval_cache) are never read for a seed."""
    qs = hashlib.sha256("\n".join(str(i) for i in sl["ids"]).encode("utf-8")).hexdigest()
    p = out_d / "seeds5.npz"
    if p.is_file():
        with np.load(p) as z:
            if str(z["qsha"]) == qs and int(z["rows_n"]) == int(ctx.n):
                return z["seeds"]
    t0 = time.time()
    S5 = flat_topk(ctx.rows, Q, 5)
    out_d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, seeds=S5, qsha=np.array(qs), rows_n=np.asarray(int(ctx.n)))
    log(f"e3v dense top-5 seeds of {Q.shape[0]} questions over {ctx.n:,} nodes: {time.time() - t0:.0f} s")
    return S5


QN_GROUP, QN_GROUP_MAX = 3, 5000
QN_NGRAM, QN_CAP, QN_MINLEN, QN_DFQ = 8, 5, 4, 0.02       # A2.5: longest span 8 tokens, at most 5 name seeds, a key of at least 4 characters, a span needs a token in at most 2 percent of the slice's questions (floor 3)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def qn_key(toks):
    ws = list(toks)
    if ws and ws[0] == "the":
        ws = ws[1:]
    return " ".join(ws)


def node_titles(d):
    """The titles of the dataset's nodes in node order (a KB node's title is its text), streamed from the served package."""
    if d == "multihoprag":
        import zul2m as M2
        _root, nodes, _n, _sha = M2.head()
        for rec in nodes:
            t_ = rec.get("title")
            yield t_ if t_ is not None else (rec.get("text") or "")
        return
    with open(E.served_root() / d / "nodes.jsonl", "rb") as f:
        for raw in f:
            rec = json.loads(raw)
            t_ = rec.get("title")
            yield t_ if t_ is not None else (rec.get("text") or "")


def qn_extra(d, sl, n, out_d):
    """A2.5, the question-name rule (the lexical leg SPLADE gave, without it): a node is a name seed of a question if its title, as a whole, is a span of the question
    (lower-cased word tokens, a leading 'the' dropped) and no other node has that title; a title with a bracket also names the node by its text without the bracket when no node has it whole.
    Spans are taken longest first, left to right, without overlap; a span needs one token that at most QN_DFQ of the slice's questions contain (so 'who', 'film' are not names);
    at most QN_CAP seeds per question, longer spans first. A title that several nodes share (a group: an article's paragraphs, a common name) is not a name seed; it is kept apart as a group
    (at most QN_GROUP_MAX nodes) for the `qg` variant, which takes the QN_GROUP nodes of a question's matched groups with the highest question cosine (a bounded, separately counted expansion).
    No model, no encoder, no label."""
    qs = hashlib.sha256("\n".join(str(i) for i in sl["ids"]).encode("utf-8")).hexdigest()
    p = out_d / "qn_seeds.npz"
    if p.is_file():
        with np.load(p) as z:
            if str(z["qsha"]) == qs and int(z["rows_n"]) == int(n):
                return ([z["ids"][z["ptr"][i]:z["ptr"][i + 1]] for i in range(len(sl["ids"]))],
                        [z["gids"][z["gptr"][i]:z["gptr"][i + 1]] for i in range(len(sl["ids"]))], json.loads(str(z["info"])))
    t0 = time.time()
    toks = [WORD_RE.findall(str(x).lower()) for x in sl["qtext"]]
    dfq = {}
    for tk in toks:
        for w in set(tk):
            dfq[w] = dfq.get(w, 0) + 1
    thr = max(3, int(np.ceil(QN_DFQ * len(toks))))
    want = set()
    for tk in toks:
        for i in range(len(tk)):
            for m in range(1, min(QN_NGRAM, len(tk) - i) + 1):
                want.add(qn_key(tk[i:i + m]))
    ex, st = {}, {}
    for i, t_ in enumerate(node_titles(d)):
        full = qn_key(WORD_RE.findall(t_.lower()))
        if full in want:
            ex.setdefault(full, []).append(i)
        if "(" in t_:
            sk = qn_key(WORD_RE.findall(re.sub(r"\([^)]*\)", " ", t_).lower()))
            if sk in want:
                st.setdefault(sk, []).append(i)
    index, gindex = {}, {}
    for k in set(ex) | set(st):
        if len(k) < QN_MINLEN:
            continue
        if len(ex.get(k, ())) == 1:
            index[k] = ex[k][0]
        elif not ex.get(k) and len(st.get(k, ())) == 1:
            index[k] = st[k][0]
        else:
            g = ex.get(k) or st.get(k) or []
            if 2 <= len(g) <= QN_GROUP_MAX:
                gindex[k] = g
    out, gout, ptr, gptr = [], [], [0], [0]
    for tk in toks:
        found, gfound, i = [], [], 0
        while i < len(tk):
            hit = None
            for m in range(min(QN_NGRAM, len(tk) - i), 0, -1):
                k = qn_key(tk[i:i + m])
                if (k in index or k in gindex) and any(dfq[w] <= thr for w in tk[i:i + m]):
                    hit = (m, k)
                    break
            if hit:
                if hit[1] in index:
                    found.append((-hit[0], i, index[hit[1]]))
                else:
                    gfound.extend(gindex[hit[1]])
                i += hit[0]
            else:
                i += 1
        found.sort()
        ids_ = list(dict.fromkeys(f[2] for f in found))[:QN_CAP]
        out.append(np.asarray(sorted(ids_), np.int64))
        ptr.append(ptr[-1] + len(ids_))
        gl = np.unique(np.asarray(gfound, np.int64))
        gout.append(gl)
        gptr.append(gptr[-1] + gl.size)
    info = {"titles_with_a_unique_node": len(index), "titles_shared_by_nodes": len(gindex), "questions": len(toks), "with_a_name_seed": int(sum(1 for o in out if o.size)),
            "mean_name_seeds": round(float(np.mean([o.size for o in out])), 3), "with_a_group": int(sum(1 for o in gout if o.size)),
            "mean_group_nodes": round(float(np.mean([o.size for o in gout])), 2), "dfq_threshold": thr, "seconds": round(time.time() - t0, 1)}
    np.savez_compressed(p, ids=np.concatenate(out) if out else np.zeros(0, np.int64), ptr=np.asarray(ptr, np.int64), gids=np.concatenate(gout) if gout else np.zeros(0, np.int64),
                        gptr=np.asarray(gptr, np.int64), qsha=np.array(qs), rows_n=np.asarray(int(n)), info=np.array(json.dumps(info)))
    log(f"question-name seeds: {info}")
    return out, gout, info


SEED_VARIANTS = ("dense", "dense+splade", "dense+qn", "dense+qg", "dense+splade+qg")


def seed_sets(a, sl, S5, ctx, out_d, Q):
    """A2.5: the seed variants of a parameter-free read: dense (the e3v top 5, the default), dense+splade (adds the filed SPLADE top 5), dense+qn (adds the question-name seeds),
    dense+qg (adds the name seeds and the QN_GROUP highest-cosine nodes of the groups the question names) and dense+splade+qg. Returns the seeds per question, the rule's info and the
    group seeds per question (for the precision record)."""
    if a.seeds not in SEED_VARIANTS:
        stop(f"--seeds {a.seeds}: choose from {', '.join(SEED_VARIANTS)}")
    parts = set(a.seeds.split("+"))
    info = {}
    extra = [[] for _ in range(len(sl["ids"]))]
    gsel = [np.zeros(0, np.int64) for _ in range(len(sl["ids"]))]
    if "splade" in parts:
        for i in range(len(extra)):
            extra[i].append(np.asarray(sl["spl5"][i], np.int64).ravel())
    if "qn" in parts or "qg" in parts:
        qn, gr, info = qn_extra(a.ds, sl, ctx.n, out_d)
        for i in range(len(extra)):
            extra[i].append(qn[i])
            if "qg" in parts and gr[i].size:
                sc = cosq(ctx, gr[i], Q[i])
                gsel[i] = np.sort(gr[i][top_w(gr[i], sc, QN_GROUP)])
                extra[i].append(gsel[i])
    SV = [np.unique(np.concatenate([np.asarray(S5[i], np.int64)] + extra[i])) for i in range(len(extra))]
    return SV, info, gsel


_MEMO = {}


def setup(a):
    """Memoised per process: a read with several seed variants builds the graphs once."""
    k = (a.ds, bool(a.host), a.v, a.read_n, getattr(a, "fit_n", 0))
    if k not in _MEMO:
        _MEMO[k] = _setup(a)
    return _MEMO[k]


def _setup(a):
    d = a.ds
    v = a.v or CELL_V[d]
    D, ctx, allg = real_ctx(d, a.host, v)
    sl = G1.dev_slice(d, D, a.host)
    if "not_read" in sl:
        stop(sl["not_read"])
    nq = len(sl["ids"])
    read_n = min(a.read_n, nq // 2) if d == "multihoprag" and a.read_n >= nq else a.read_n
    read = np.arange(min(read_n, nq))
    rest = np.arange(read.size, nq)
    fit_n = getattr(a, "fit_n", 0)
    fit = np.sort(np.random.default_rng(SEED).permutation(rest)[:fit_n]) if fit_n else rest[:0]
    return D, ctx, allg, sl, v, read, fit


# ── probe ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def cmd_probe(a):
    t0 = time.time()
    D, ctx, allg, sl, v, read, _fit = setup(a)
    Q = question_vectors(a.ds, sl)
    out_d = OUT / a.ds
    out_d.mkdir(parents=True, exist_ok=True)
    S5 = seeds_e3v(ctx, sl, Q, out_d)
    rec = []
    for qi in read[:a.n].tolist():
        for ti, wd in enumerate(WIDTHS):
            t1 = time.time()
            pool, kept = walk(ctx, Q[qi], S5[qi], wd)
            rec.append({"q": qi, "triple": ti, "secs": round(time.time() - t1, 2), "nodes": int(pool.size), "kept": [int(k.size) for k in kept]})
        t1 = time.time()
        wd = WIDTHS[-1]
        sink = {}
        walk(ctx, Q[qi], S5[qi], wd, collect=lambda h, F, c: sink.__setitem__(h, int(c.size)))
        rec.append({"q": qi, "triple": "feat", "secs": round(time.time() - t1, 2), "cand": [sink.get(h, 0) for h in (1, 2, 3)]})
        log(f"{a.ds} q{qi}: B1 five triples {sum(r['secs'] for r in rec if r['q'] == qi and r['triple'] != 'feat'):.1f} s, feature walk (widest) {rec[-1]['secs']} s, candidates {rec[-1]['cand']}, thin {ctx.thin} ({G1.C.peak_gb()} GB)")
    secs = [r["secs"] for r in rec if r["triple"] != "feat"]
    fsec = [r["secs"] for r in rec if r["triple"] == "feat"]
    summ = {"dataset": a.ds, "v": v, "questions": int(min(a.n, read.size)), "b1_five_triples_secs_per_question": round(sum(secs) / max(1, min(a.n, read.size)), 2),
            "feature_walk_widest_secs_per_question": round(float(np.mean(fsec)), 2) if fsec else None, "thinned": bool(ctx.thin), "peak_gb": G1.C.peak_gb(), "nodes": D.n,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "zjnt2_sha256": script_sha256(), "rows": rec}
    (out_d / "probe.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    log(f"{a.ds}: probe {summ['b1_five_triples_secs_per_question']} s a question for B1 at five triples, {summ['feature_walk_widest_secs_per_question']} s for one feature walk at the widest; thinned {ctx.thin}; {time.time() - t0:.0f} s")
    return 0


# ── labels (the candidate tables of the fit questions) ────────────────────────────────────────────────────────────


def cmd_labels(a):
    t0 = time.time()
    D, ctx, allg, sl, v, read, fit = setup(a)
    Q = question_vectors(a.ds, sl)
    out_d = OUT / a.ds
    S5 = seeds_e3v(ctx, sl, Q, out_d)
    tdir = out_d / f"tables_r{a.round}"
    tdir.mkdir(parents=True, exist_ok=True)
    model = None
    if a.round == 1:
        mp = out_d / "model_r0.npz"
        if not mp.is_file():
            stop(f"round 1 needs {mp}")
        model = HopModel.load(mp)
    wd = WIDTHS[FIT_TRIPLE]
    gctx = Ctx(D.n, {"all": allg}, ctx.rows)
    BIT["all"] = 7
    zsha = script_sha256()
    nck = (fit.size + CHUNK_Q - 1) // CHUNK_Q
    for ci in range(nck):
        cp = tdir / f"chunk_{ci:04d}.npz"
        if cp.is_file():
            with np.load(cp) as z:
                if str(z["zsha"]) == zsha and int(z["nq"]) == fit.size:
                    continue
        Fs, ys, qs, hs, dgs, pops = [], [], [], [], [], []
        for qi in fit[ci * CHUNK_Q:(ci + 1) * CHUNK_Q].tolist():
            B = gold_sets(gctx, sl["golds"][qi])
            rng = np.random.default_rng(SEED + qi)
            tab = {}

            def grab(h, F, cand, tab=tab):
                tab[h] = (F, cand)
            walk(ctx, Q[qi], S5[qi], wd, model=model, use_ps=a.round == 1, collect=grab)
            for h, (F, cand) in tab.items():
                dg = np.full(cand.size, 9, np.int8)             # A2.2: the distance of the candidate to its nearest gold (0, 1, 2; 9 = farther), stored so a label variant needs no new walk
                for k in (2, 1, 0):
                    dg[in_sorted(cand, B[k])] = k
                pos = dg <= 3 - h                                # the declared label: within the hops that remain
                pops.append([qi, h, cand.size] + [int((dg <= k).sum()) for k in (0, 1, 2)])   # A2.2: the population counts before any cap (candidates; those within 0, 1, 2 hops of a gold)
                pi, ni = np.flatnonzero(pos), np.flatnonzero(~pos)
                if pi.size > POS_CAP:
                    pi = np.sort(rng.choice(pi, POS_CAP, replace=False))
                if ni.size > NEG_PER:
                    ni = np.sort(rng.choice(ni, NEG_PER, replace=False))
                sel = np.concatenate([pi, ni])
                Fs.append(F[sel])
                ys.append(np.r_[np.ones(pi.size, np.int8), np.zeros(ni.size, np.int8)])
                qs.append(np.full(sel.size, qi, np.int32))
                hs.append(np.full(sel.size, h, np.int8))
                dgs.append(dg[sel])
        tmp = cp.with_name(cp.name + ".tmp.npz")
        np.savez_compressed(tmp, F=np.concatenate(Fs) if Fs else np.zeros((0, NFEAT), np.float32), y=np.concatenate(ys) if ys else np.zeros(0, np.int8),
                            q=np.concatenate(qs) if qs else np.zeros(0, np.int32), h=np.concatenate(hs) if hs else np.zeros(0, np.int8), dg=np.concatenate(dgs) if dgs else np.zeros(0, np.int8), pop=np.asarray(pops, np.int64).reshape(-1, 6),
                            zsha=np.array(zsha), nq=np.asarray(fit.size), thin=np.asarray(bool(ctx.thin or gctx.thin)))
        tmp.replace(cp)
        log(f"{a.ds}/r{a.round}: chunk {ci + 1}/{nck} ({min((ci + 1) * CHUNK_Q, fit.size)}/{fit.size} questions, {time.time() - t0:.0f} s, {G1.C.peak_gb()} GB, thinned {ctx.thin or gctx.thin})")
    log(f"{a.ds}: round {a.round} tables for {fit.size} fit questions in {time.time() - t0:.0f} s")
    return 0


def cmd_fit(a):
    t0 = time.time()
    d_ = OUT / a.ds / f"tables_r{a.round}"
    parts = sorted(d_.glob("chunk_*.npz"))
    if not parts:
        stop(f"no tables in {d_}")
    F, dg, q, h, pop, thin = [], [], [], [], [], False
    for p in parts:
        with np.load(p) as z:
            F.append(z["F"])
            dg.append(z["dg"])
            q.append(z["q"])
            h.append(z["h"])
            pop.append(z["pop"])
            thin = thin or bool(z["thin"])
    F, dg, q, h, pop = np.concatenate(F), np.concatenate(dg), np.concatenate(q), np.concatenate(h), np.concatenate(pop)
    # A2.2: the label is the declared one (a candidate of hop h is positive within 3 - h hops of a gold) unless more than half of the hop's POPULATION of candidates would be positive,
    # in which case it is tightened by one hop at a time (a label that most candidates carry separates nothing); the population rates are counted before the sampling caps.
    ymax, popinfo = {}, {}
    for hh in (1, 2, 3):
        pr = pop[pop[:, 1] == hh]
        nc = int(pr[:, 2].sum())
        cum = [int(pr[:, 3 + k].sum()) for k in (0, 1, 2)]
        yh = 3 - hh
        while yh > 0 and nc and cum[yh] / nc > 0.5:
            yh -= 1
        ymax[hh] = yh
        popinfo[hh] = {"candidates": nc, "within0": cum[0], "within1": cum[1], "within2": cum[2], "declared_rate": round(cum[3 - hh] / nc, 4) if nc else None, "label_hops": yh,
                       "label_rate": round(cum[yh] / nc, 4) if nc else None}
    y = np.zeros(dg.size, np.int8)
    for hh in (1, 2, 3):
        y[h == hh] = (dg[h == hh] <= ymax[hh])
    if a.shuffle:                                                # G-J2: the labels permuted within each (question, hop)
        rng = np.random.default_rng(SEED + 1)
        key = q.astype(np.int64) * 8 + h
        for kk in np.unique(key):
            ix = np.flatnonzero(key == kk)
            y[ix] = y[ix][rng.permutation(ix.size)]
    log(f"{a.ds}/r{a.round}: {F.shape[0]:,} sampled rows, {int(y.sum()):,} positives ({100 * y.mean():.1f} percent of the sample), {np.unique(q).size} questions{' (labels shuffled within each question and hop)' if a.shuffle else ''}; population label rates by hop {[popinfo[k]['label_rate'] for k in (1, 2, 3)]} (declared {[popinfo[k]['declared_rate'] for k in (1, 2, 3)]}, label hops {[ymax[k] for k in (1, 2, 3)]})")
    pooled = fit_mlp(F, y)
    heads, per = {}, {}
    for hh in (1, 2, 3):
        ix = np.flatnonzero(h == hh)
        npos = int(y[ix].sum())
        ok = ix.size >= MIN_HEAD_ROWS and 10 <= npos <= ix.size - 10
        if ok:
            heads[hh] = fit_mlp(F[ix], y[ix], seed=SEED + hh)
        m = heads.get(hh) or pooled
        per[str(hh)] = {"rows": int(ix.size), "positives": npos, "sample_pos_rate": round(npos / max(1, ix.size), 4), "population": popinfo[hh], "own_head": bool(ok),
                        "train_auc_sample": auc_of(m.logit(F[ix]), y[ix]) if ix.size else None}
    mdl = HopModel(heads, pooled)
    tag = "_shuf" if a.shuffle else ""
    mdl.save(OUT / a.ds / f"model_r{a.round}{tag}.npz")
    rec = {"dataset": a.ds, "round": a.round, "shuffle": bool(a.shuffle), "rows": int(F.shape[0]), "positives": int(y.sum()), "questions": int(np.unique(q).size), "label_hops": {str(k): v for k, v in ymax.items()}, "per_hop": per,
           "pooled_train_auc_sample": auc_of(pooled.logit(F), y), "thinned_in_tables": bool(thin), "seconds": round(time.time() - t0, 1), "zjnt2_sha256": script_sha256()}
    (OUT / a.ds / f"fit_r{a.round}{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log(f"{a.ds}: fitted round {a.round}{tag}: per-hop population label rates {[per[k]['population']['label_rate'] for k in sorted(per)]}, per-hop train AUC {[per[k]['train_auc_sample'] for k in sorted(per)]}; {time.time() - t0:.0f} s")
    return 0


# ── read ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def arm_stats(pool, golds):
    inn = in_sorted(np.asarray(golds, np.int64), pool)
    return [float(inn.all()), float(inn.any()), float(inn.mean()), float(pool.size)]


def boot(dif, nb=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = dif.size
    ix = rng.integers(0, n, size=(nb, n))
    m = dif[ix].mean(1)
    return float(dif.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def call(lo, hi):
    return "ABOVE" if lo > 1.0 else "BELOW" if hi < -1.0 else "LEVEL"


ARMS = ("B0", "B1", "L0", "L1")


def cmd_read(a):
    t0 = time.time()
    D, ctx, allg, sl, v, read, _fit = setup(a)
    Q = question_vectors(a.ds, sl)
    out_d = OUT / a.ds
    S5 = seeds_e3v(ctx, sl, Q, out_d)
    if a.seeds != "dense" and not a.b_only:
        stop("--seeds other than dense needs --b-only (the models were fitted on the dense seeds)")
    SV, sinfo, gsel = seed_sets(a, sl, S5, ctx, out_d, Q)
    tag = ("_shuf" if a.shuffle else "") + ("_b" if a.b_only else "") + ("" if a.seeds == "dense" else "_s-" + a.seeds)
    models = {}
    if not a.b_only:
        for r in (0, 1):
            mp = out_d / f"model_r{r}{'_shuf' if a.shuffle else ''}.npz"
            if mp.is_file():
                models[r] = HopModel.load(mp)
        if 0 not in models:
            stop("no round 0 model: run labels --round 0 and fit first")
    zsha = script_sha256()
    ck = out_d / f"read_work{tag}"
    ck.mkdir(parents=True, exist_ok=True)
    nT = len(WIDTHS)
    nck = (read.size + CHUNK_Q - 1) // CHUNK_Q
    big = min(max(sum(w) for w in WIDTHS) + 100, int(ctx.n))
    flat_p = out_d / f"flat_topk_{read.size}_{big}.npz"                   # shared by every seed variant and arm: the flat e3v list does not depend on the seeds
    if flat_p.is_file():
        with np.load(flat_p) as z:
            flat = z["flat"]
    else:
        log(f"{a.ds}: flat e3v question-cosine lists (k = {big}) for {read.size} questions")
        flat = flat_topk(ctx.rows, Q[read], big)
        np.savez_compressed(flat_p, flat=flat)
    n50 = min(50, read.size)
    chk = flat_topk(ctx.rows, Q[read[:n50]], 5, block=1 << 16)           # G-J1 (STOP): the e3v top 5 is the same by a second pass with other blocks and equals the flat list's first 5
    bad = sum(int(set(chk[i].tolist()) != set(S5[qi].tolist())) for i, qi in enumerate(read[:n50].tolist()))
    badf = sum(int(set(flat[i, :5].tolist()) != set(S5[qi].tolist())) for i, qi in enumerate(read[:n50].tolist()))
    older = sum(int(set(S5[qi].tolist()) != set(np.asarray(sl["dense5"][qi]).tolist())) for qi in read[:n50].tolist())      # information only: the older encoder's filed lists
    seed_any = float(np.mean([bool(in_sorted(np.asarray(sl["golds"][qi], np.int64), SV[qi]).any()) for qi in read.tolist()]))
    seed_cov = float(np.mean([float(in_sorted(np.asarray(sl["golds"][qi], np.int64), SV[qi]).mean()) for qi in read.tolist()]))
    seed_n = float(np.mean([SV[qi].size for qi in read.tolist()]))
    qn_gold = None
    if "qn" in a.seeds or "qg" in a.seeds:
        qn_, gr_, _i = qn_extra(a.ds, sl, ctx.n, out_d)
        gs = lambda x: np.sort(np.asarray(x, np.int64))  # noqa: E731
        tot = sum(int(qn_[qi].size) for qi in read.tolist())
        hit = sum(int(in_sorted(qn_[qi], gs(sl["golds"][qi])).sum()) for qi in read.tolist())
        gtot = sum(int(gsel[qi].size) for qi in read.tolist())
        ghit = sum(int(in_sorted(gsel[qi], gs(sl["golds"][qi])).sum()) for qi in read.tolist())
        qn_gold = {"name_seeds": tot, "gold": hit, "precision": round(hit / max(1, tot), 4), "questions_with_one": int(sum(1 for qi in read.tolist() if qn_[qi].size)),
                   "group_seeds": gtot, "group_gold": ghit, "group_precision": round(ghit / max(1, gtot), 4), "questions_with_a_group": int(sum(1 for qi in read.tolist() if gsel[qi].size))}
    log(f"G-J1: e3v top-5 second pass differs on {bad} of {n50}, flat list's first 5 on {badf} of {n50} (stop above 2); the older encoder's filed top 5 differ from the e3v seeds on {older} of {n50} (information only); a gold is among the seeds on {100 * seed_any:.1f} percent of the read questions")
    if bad > 2 or badf > 2:
        stop(f"G-J1: the e3v seeds are not stable ({bad}, {badf} of {n50} differ)")
    res = np.full((read.size, nT, len(ARMS), 4), np.nan, np.float32)
    for ci in range(nck):
        cp = ck / f"chunk_{ci:04d}.npz"
        sub = read[ci * CHUNK_Q:(ci + 1) * CHUNK_Q]
        if cp.is_file():
            with np.load(cp) as z:
                if str(z["zsha"]) == zsha:
                    res[ci * CHUNK_Q:ci * CHUNK_Q + sub.size] = z["res"]
                    continue
        part = np.full((sub.size, nT, len(ARMS), 4), np.nan, np.float32)
        for j, qi in enumerate(sub.tolist()):
            g = sl["golds"][qi]
            for ti, wd in enumerate(WIDTHS):
                p1, _k = walk(ctx, Q[qi], SV[qi], wd)
                part[j, ti, 1] = arm_stats(p1, g)
                if a.b_only:
                    continue
                p0, _k = walk(ctx, Q[qi], SV[qi], wd, model=models[0], use_ps=False)
                part[j, ti, 2] = arm_stats(p0, g)
                if 1 in models:
                    pl, _k = walk(ctx, Q[qi], SV[qi], wd, model=models[1], use_ps=True)
                    part[j, ti, 3] = arm_stats(pl, g)
        tmp = cp.with_name(cp.name + ".tmp.npz")
        np.savez_compressed(tmp, res=part, zsha=np.array(zsha))
        tmp.replace(cp)
        res[ci * CHUNK_Q:ci * CHUNK_Q + sub.size] = part
        log(f"{a.ds}: read chunk {ci + 1}/{nck} ({min((ci + 1) * CHUNK_Q, read.size)}/{read.size} questions, {time.time() - t0:.0f} s, {G1.C.peak_gb()} GB, thinned {ctx.thin})")
    done = ~np.isnan(res[:, 0, 1, 0])
    summary = {"dataset": a.ds, "v": v, "declared_in": DECLARED + " section 11", "status": "COMPLETE" if bool(done.all()) else "PARTIAL", "questions": int(done.sum()), "of": int(read.size),
               "shuffle": bool(a.shuffle), "b_only": bool(a.b_only), "seeds": a.seeds, "seeds_note": "dense = the e3v top 5 (seeds5.npz); splade = the filed SPLADE top 5; qn = the question-name seeds (A2.5)", "seed_mean_nodes": round(seed_n, 2), "seed_coverage_pct": round(100 * seed_cov, 2), "qn": sinfo, "qn_vs_gold": qn_gold,
               "gj1": {"second_pass_differs": bad, "flat_first5_differs": badf, "of": n50, "older_encoder_filed_top5_differs_info_only": older}, "gold_among_seeds_pct": round(100 * seed_any, 2),
               "thinned": bool(ctx.thin), "triples": []}
    for ti, wd in enumerate(WIDTHS):
        L1ok = (not a.b_only) and bool(np.isfinite(res[done, ti, 3, 0]).all()) and 1 in models
        nm = float(np.nanmean(res[done, ti, 1 if a.b_only else 2, 3]))
        nflat = int(round(nm))
        for j, qi in enumerate(read.tolist()):
            if done[j]:
                res[j, ti, 0] = arm_stats(np.unique(flat[j, :nflat]), sl["golds"][qi])
        row = {"widths": list(wd), "n_flat": nflat, "arms": {}, "calls": {}, "b1_all": [int(x) for x in res[done, ti, 1, 0]], "b0_all": [int(x) for x in res[done, ti, 0, 0]]}
        for k, nmarm in enumerate(ARMS):
            if (nmarm in ("L0", "L1") and a.b_only) or (nmarm == "L1" and not L1ok):
                continue
            x = res[done, ti, k]
            row["arms"][nmarm] = {"ALL": round(100 * float(x[:, 0].mean()), 2), "ANY": round(100 * float(x[:, 1].mean()), 2), "cov": round(100 * float(x[:, 2].mean()), 2), "nodes": round(float(x[:, 3].mean()), 1)}
        for A, Bn in (("L0", "B1"), ("L0", "B0"), ("L1", "B1"), ("L1", "B0"), ("B1", "B0")):
            if A not in row["arms"] or Bn not in row["arms"]:
                continue
            dif = 100 * (res[done, ti, ARMS.index(A), 0] - res[done, ti, ARMS.index(Bn), 0])
            m_, lo, hi = boot(dif)
            row["calls"][f"{A}-{Bn}"] = {"diff": round(m_, 2), "lo": round(lo, 2), "hi": round(hi, 2), "call": call(lo, hi)}
        summary["triples"].append(row)
    summary.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zjnt2_sha256=script_sha256())
    (out_d / f"read{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for row in summary["triples"]:
        log(f"{a.ds} {row['widths']} flat n={row['n_flat']}: " + "; ".join(f"{k} ALL {v_['ALL']:.1f} ({v_['nodes']:.0f} nodes)" for k, v_ in row["arms"].items()) + " | " + "; ".join(f"{k} {c['diff']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}] {c['call']}" for k, c in row["calls"].items()))
    log(f"{a.ds}: read {summary['status']} ({summary['questions']}/{summary['of']}), {time.time() - t0:.0f} s")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for nm in ("probe", "labels", "fit", "read"):
        p = sub.add_parser(nm)
        p.add_argument("--ds", required=True)
        p.add_argument("--host", action="store_true")
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--v", default="")
        p.add_argument("--read-n", dest="read_n", type=int, default=300)
        if nm == "probe":
            p.add_argument("--n", type=int, default=20)
        if nm in ("labels", "fit"):
            p.add_argument("--round", type=int, default=0, choices=(0, 1))
        if nm == "labels":
            p.add_argument("--fit-n", dest="fit_n", type=int, default=2000)
        if nm in ("fit", "read"):
            p.add_argument("--shuffle", action="store_true")
        if nm == "read":
            p.add_argument("--b-only", dest="b_only", action="store_true")
            p.add_argument("--seeds", default="dense", help="dense | dense+splade | dense+qn | dense+qg | dense+splade+qg | all (all five, b-only)")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "read" and a.seeds == "all":
        if not a.b_only:
            stop("--seeds all needs --b-only")
        for sv in SEED_VARIANTS:
            a.seeds = sv
            cmd_read(a)
        return 0
    return {"probe": cmd_probe, "labels": cmd_labels, "fit": cmd_fit, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
