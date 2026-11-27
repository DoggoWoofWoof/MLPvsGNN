"""JNT1 (docs/JNT1_LEARNED_WALK.md and its amendments): a learned link score (a one-hidden-layer MLP, no message passing) orders a budgeted three-hop walk of the any-link graph
(S, W, T0; K dropped) from the dense top-5 seeds, against the question-cosine order (B1) and the flat dense list (B0), at equal widths.

    python outputs/mp_unified/zjnt1.py selftest                                   # the walker, the labels, the features and the MLP on toy graphs (no real data)
    python outputs/mp_unified/zjnt1.py probe  --ds D [--host] [--n 20] [--v V]     # prices the walk on the first read questions (B1 arm at the five triples), no label read
    python outputs/mp_unified/zjnt1.py labels --ds D --round R [--host] [--v V] [--read-n 300] [--fit-n 2000]   # candidate tables of the fit questions (round 1 needs round 0's model)
    python outputs/mp_unified/zjnt1.py fit    --ds D --round R [--shuffle]         # fits the MLP on the tables
    python outputs/mp_unified/zjnt1.py read   --ds D [--host] [--v V] [--read-n 300] [--shuffle]   # B0, B1, L0, L1 at the five triples on the read questions, the calls

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
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

G1, H, T3, P, E = G3.G1, G3.H, G3.T3, G3.P, G3.G1.E
DECLARED = "docs/JNT1_LEARNED_WALK.md"
OUT = ROOT / "outputs" / "jnt1"
EMPTY = np.zeros(0, np.int64)
WIDTHS = ((10, 20, 30), (20, 60, 120), (40, 150, 400), (80, 400, 1500), (150, 900, 3000))   # section 2: fixed
FIT_TRIPLE = 2                                         # A1.2: the triple whose walk generates the training candidates
CELL_V = {"multihoprag": "RK", "musique": "RK", "squad": "RK", "metaqa": "SRK", "webqsp": "SRK", "hotpotqa": "SRK", "2wiki": "SRK"}   # GP3's cells; hotpotqa/2wiki also have RK (--v)
BIT = {"S": 1, "W": 2, "T0": 4}
GATHER_CAP = G3.GATHER_CAP
NEG_PER, POS_CAP, CHUNK_Q = 200, 1000, 20              # A1.2
HID, EPOCHS, LR, WD, BATCH, SEED = 64, 40, 3e-3, 1e-5, 4096, 20261127
NFEAT = 13
BLK = 20000
log = G1.log


def stop(msg):
    raise SystemExit(f"[zjnt1] STOP: {msg}")


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
    """The 13 label-free features of the candidates of hop h (section 2, A1.4). ps = (sorted kept nodes of hop h-1, their scores) or None."""
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
    fl = np.bitwise_or.reduceat(f[order], st)
    isS = (fl & 1) > 0
    isW = (~isS) & ((fl & 2) > 0)
    isT = (~isS) & (~isW)
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
    F[:, 5], F[:, 6], F[:, 7] = isS, isW, isT
    F[:, 7 + h] = 1.0                                   # columns 8, 9, 10 = hop 1, 2, 3
    F[:, 11] = np.log1p(npar)
    F[:, 12] = psb
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
        assert ((F[:, 5] + F[:, 6] + F[:, 7]) == 1).all(), "component one-hot"
        assert (F[:, 12] == 0).all(), "round 0 ps column is 0"
        assert np.allclose(F[:, 0], ctx.rows.X[cand] @ qv, atol=1e-6), "question cosine column"
        assert (F[:, 3] >= 0).all() and F[:, 11].min() >= np.log(2) - 1e-6, "ranks and parent counts"
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
    log(f"selftest PASS: {trials} toy graphs: huge widths = the 3-hop BFS ball, 5 width triples = the brute-force top-w walk, the feature path = the cosine path; top_w ties; gold balls = BFS; features finite, one-hot, label-free; MLP forward = manual (toy accuracy {acc:.3f})")
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


def flat_topk(rows, Q, k, block=1 << 18):
    """Per question the k nodes of the highest question cosine, ordered by (-cosine, id): one pass over the stored e3v rows."""
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


def setup(a):
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
    rec = []
    for qi in read[:a.n].tolist():
        for ti, wd in enumerate(WIDTHS):
            t1 = time.time()
            pool, kept = walk(ctx, Q[qi], sl["dense5"][qi], wd)
            rec.append({"q": qi, "triple": ti, "secs": round(time.time() - t1, 2), "nodes": int(pool.size), "kept": [int(k.size) for k in kept]})
        t1 = time.time()
        wd = WIDTHS[-1]
        sink = {}
        walk(ctx, Q[qi], sl["dense5"][qi], wd, collect=lambda h, F, c: sink.__setitem__(h, int(c.size)))
        rec.append({"q": qi, "triple": "feat", "secs": round(time.time() - t1, 2), "cand": [sink.get(h, 0) for h in (1, 2, 3)]})
        log(f"{a.ds} q{qi}: B1 five triples {sum(r['secs'] for r in rec if r['q'] == qi and r['triple'] != 'feat'):.1f} s, feature walk (widest) {rec[-1]['secs']} s, candidates {rec[-1]['cand']}, thin {ctx.thin} ({G1.C.peak_gb()} GB)")
    secs = [r["secs"] for r in rec if r["triple"] != "feat"]
    fsec = [r["secs"] for r in rec if r["triple"] == "feat"]
    summ = {"dataset": a.ds, "v": v, "questions": int(min(a.n, read.size)), "b1_five_triples_secs_per_question": round(sum(secs) / max(1, min(a.n, read.size)), 2),
            "feature_walk_widest_secs_per_question": round(float(np.mean(fsec)), 2) if fsec else None, "thinned": bool(ctx.thin), "peak_gb": G1.C.peak_gb(), "nodes": D.n,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "zjnt1_sha256": script_sha256(), "rows": rec}
    (out_d / "probe.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    log(f"{a.ds}: probe {summ['b1_five_triples_secs_per_question']} s a question for B1 at five triples, {summ['feature_walk_widest_secs_per_question']} s for one feature walk at the widest; thinned {ctx.thin}; {time.time() - t0:.0f} s")
    return 0


# ── labels (the candidate tables of the fit questions) ────────────────────────────────────────────────────────────


def cmd_labels(a):
    t0 = time.time()
    D, ctx, allg, sl, v, read, fit = setup(a)
    Q = question_vectors(a.ds, sl)
    out_d = OUT / a.ds / f"tables_r{a.round}"
    out_d.mkdir(parents=True, exist_ok=True)
    model = None
    if a.round == 1:
        mp = OUT / a.ds / "model_r0.npz"
        if not mp.is_file():
            stop(f"round 1 needs {mp}")
        model = MLP.load(mp)
    wd = WIDTHS[FIT_TRIPLE]
    gctx = Ctx(D.n, {"all": allg}, ctx.rows)
    BIT["all"] = 7
    zsha = script_sha256()
    nck = (fit.size + CHUNK_Q - 1) // CHUNK_Q
    for ci in range(nck):
        cp = out_d / f"chunk_{ci:04d}.npz"
        if cp.is_file():
            with np.load(cp) as z:
                if str(z["zsha"]) == zsha and int(z["nq"]) == fit.size:
                    continue
        Fs, ys, qs, hs = [], [], [], []
        for qi in fit[ci * CHUNK_Q:(ci + 1) * CHUNK_Q].tolist():
            B = gold_sets(gctx, sl["golds"][qi])
            rng = np.random.default_rng(SEED + qi)
            tab = {}

            def grab(h, F, cand, tab=tab):
                tab[h] = (F, cand)
            walk(ctx, Q[qi], sl["dense5"][qi], wd, model=model, use_ps=a.round == 1, collect=grab)
            for h, (F, cand) in tab.items():
                pos = in_sorted(cand, B[3 - h])
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
        tmp = cp.with_name(cp.name + ".tmp.npz")
        np.savez_compressed(tmp, F=np.concatenate(Fs) if Fs else np.zeros((0, NFEAT), np.float32), y=np.concatenate(ys) if ys else np.zeros(0, np.int8),
                            q=np.concatenate(qs) if qs else np.zeros(0, np.int32), h=np.concatenate(hs) if hs else np.zeros(0, np.int8), zsha=np.array(zsha), nq=np.asarray(fit.size))
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
    F, y, q = [], [], []
    for p in parts:
        with np.load(p) as z:
            F.append(z["F"])
            y.append(z["y"])
            q.append(z["q"])
    F, y, q = np.concatenate(F), np.concatenate(y), np.concatenate(q)
    if a.shuffle:
        rng = np.random.default_rng(SEED + 1)
        for qq in np.unique(q):
            ix = np.flatnonzero(q == qq)
            y[ix] = y[ix][rng.permutation(ix.size)]
    log(f"{a.ds}/r{a.round}: {F.shape[0]:,} rows, {int(y.sum()):,} positives ({100 * y.mean():.1f} percent), {np.unique(q).size} questions{' (labels shuffled within each question)' if a.shuffle else ''}")
    mdl = fit_mlp(F, y)
    tag = "_shuf" if a.shuffle else ""
    mdl.save(OUT / a.ds / f"model_r{a.round}{tag}.npz")
    pr = 1.0 / (1.0 + np.exp(-mdl.logit(F)))
    auc_sample = np.random.default_rng(SEED).choice(F.shape[0], min(200_000, F.shape[0]), replace=False)
    yy, pp = y[auc_sample], pr[auc_sample]
    ranks = np.empty(pp.size)
    ranks[np.argsort(pp, kind="stable")] = np.arange(1, pp.size + 1)
    npos = int(yy.sum())
    auc = (ranks[yy == 1].sum() - npos * (npos + 1) / 2) / max(1, npos * (pp.size - npos))
    rec = {"dataset": a.ds, "round": a.round, "shuffle": bool(a.shuffle), "rows": int(F.shape[0]), "positives": int(y.sum()), "questions": int(np.unique(q).size),
           "train_auc_sample": round(float(auc), 4), "seconds": round(time.time() - t0, 1), "zjnt1_sha256": script_sha256()}
    (OUT / a.ds / f"fit_r{a.round}{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log(f"{a.ds}: fitted round {a.round}{tag}: train AUC (200k-row sample) {auc:.3f}; {time.time() - t0:.0f} s")
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
    tag = "_shuf" if a.shuffle else ""
    models = {}
    for r in (0, 1):
        mp = out_d / f"model_r{r}{tag}.npz"
        if mp.is_file():
            models[r] = MLP.load(mp)
    if 0 not in models:
        stop("no round 0 model: run labels --round 0 and fit first")
    zsha = script_sha256()
    ck = out_d / f"read_work{tag}"
    ck.mkdir(parents=True, exist_ok=True)
    nT = len(WIDTHS)
    nck = (read.size + CHUNK_Q - 1) // CHUNK_Q
    big = min(max(sum(w) for w in WIDTHS) + 100, int(ctx.n))
    flat_p = ck / "flat_topk.npz"
    if flat_p.is_file():
        with np.load(flat_p) as z:
            flat = z["flat"]
    else:
        log(f"{a.ds}: flat e3v question-cosine lists (k = {big}) for {read.size} questions")
        flat = flat_topk(ctx.rows, Q[read], big)
        np.savez_compressed(flat_p, flat=flat)
    mism = sum(int(set(flat[i, :5].tolist()) != set(np.asarray(sl["dense5"][qi]).tolist())) for i, qi in enumerate(read[:50].tolist()))
    log(f"G-J1: the flat list's first 5 differ from the filed dense top 5 on {mism} of {min(50, read.size)} questions")
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
                p1, _k = walk(ctx, Q[qi], sl["dense5"][qi], wd)
                part[j, ti, 1] = arm_stats(p1, g)
                p0, _k = walk(ctx, Q[qi], sl["dense5"][qi], wd, model=models[0], use_ps=False)
                part[j, ti, 2] = arm_stats(p0, g)
                if 1 in models:
                    pl, _k = walk(ctx, Q[qi], sl["dense5"][qi], wd, model=models[1], use_ps=True)
                    part[j, ti, 3] = arm_stats(pl, g)
        tmp = cp.with_name(cp.name + ".tmp.npz")
        np.savez_compressed(tmp, res=part, zsha=np.array(zsha))
        tmp.replace(cp)
        res[ci * CHUNK_Q:ci * CHUNK_Q + sub.size] = part
        log(f"{a.ds}: read chunk {ci + 1}/{nck} ({min((ci + 1) * CHUNK_Q, read.size)}/{read.size} questions, {time.time() - t0:.0f} s, {G1.C.peak_gb()} GB, thinned {ctx.thin})")
    done = ~np.isnan(res[:, 0, 1, 0])
    arm = {"L1": 3, "L0": 2}
    summary = {"dataset": a.ds, "v": v, "declared_in": DECLARED, "status": "COMPLETE" if bool(done.all()) else "PARTIAL", "questions": int(done.sum()), "of": int(read.size),
               "shuffle": bool(a.shuffle), "gj1_flat_vs_filed_dense5_mismatches": mism, "thinned": bool(ctx.thin), "triples": []}
    for ti, wd in enumerate(WIDTHS):
        L1ok = bool(np.isfinite(res[done, ti, 3, 0]).all()) and 1 in models
        nm = float(np.nanmean(res[done, ti, 2, 3]))
        nflat = int(round(nm))
        for j, qi in enumerate(read.tolist()):
            if done[j]:
                res[j, ti, 0] = arm_stats(np.unique(flat[j, :nflat]), sl["golds"][qi])
        row = {"widths": list(wd), "n_flat": nflat, "arms": {}, "calls": {}}
        for k, nmarm in enumerate(ARMS):
            if nmarm == "L1" and not L1ok:
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
    summary.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zjnt1_sha256=script_sha256())
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
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    return {"probe": cmd_probe, "labels": cmd_labels, "fit": cmd_fit, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
