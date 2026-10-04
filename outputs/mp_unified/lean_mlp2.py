"""Design look (untracked; not a result and not filed): lean_mlp's cost-aware MLP with seed-centric, non-MP stand-ins
for the three compiled blocks ln-2w's greedy kept at a cost.

ln-2w (3 Oct, 2wiki x4 -> select) picked 9 blocks at 9.01 ms of approximate compile (the full set 12.75). Three of the
nine need the 1536-wide embedding gather or the all-pairs depth basis, and dropping each cost select points (R@5/FC@5):
seed_e 0.49/0.67, depth_FULL 0.42/1.34, dense_cos 0.56/0.94. The stand-ins here need neither:

    DLIST  the dense list's own score where the node is on the cached top-1000 list, the listed flag, and the
           projected cosine where it is not. The list ships its scores, so the rank lap reads them with the ranks; the
           look's dense_cos is that score on listed nodes only if the list was scored with the served embeddings
           (lean_time.py checks it on the live inputs)
    DISTS  per seed, a breadth-first pass over the pool's structural pairs (S <= 10 seeds, 3 hops). The compile's views
           are binary, symmetric and deduplicated with the diagonal dropped (fast_features._sym_csr), so the seeds at
           exactly t hops from v are the seeds whose pass reaches v at hop t: the depth basis's seed columns come out of
           S passes instead of the all-pairs reach bitsets. Per hop t in 1..3: log1p(seeds at t), their seed mass
           (s_k = rrf_k / max rrf over the pool), the projected cosine to their mean; and the basis's walk columns over
           the same pairs, support_t (p <- the mean over v's neighbours of p, from s) for t = 1..3 and log1p(branch_t)
           (v's neighbours with a walk of length t-1 from a seed) for t = 2, 3. 14 columns.
    DISTF  DISTS over all three families (the FULL view)
    WALKF  lean_mlp's WALK over all three families (NER and kNN pairs walk both ways), plus log1p of the hop-1 seed
           walks over NER and over kNN edges
    NBRF   lean_mlp's NBR over all three families, plus has-neighbour over NER and over kNN
    NBR2S  structural two-step walks over the view's pairs: the walk-weighted mean (A A c)/(A A 1) and the two-step max
           of the projected query cosine c, and log1p(A A 1)
    NBR2F  NBR2S over all three families
What the basis has and these do not are its ring columns (ring_qmean/qmax/ring_n over the nodes at exactly t hops from
v), the only all-pairs part; NBR (one step) and NBR2* (two steps, walk-weighted, not exact-distance) stand in for them.
Every new block is a fixed count or a fixed mean of fixed projections. DISTS/DISTF and NBR2* propagate over two to three
hops with no parameter (parameter-free propagation, flagged as such, as SubgraphRAG's DDE is); none learns anything
over edges, so a model on them is non-MP in the project's sense.

Everything else is lean_mlp's, imported and called unchanged (the model, the fit, the greedy, quality, read_one); the
new blocks join its cost model by the edges they need (the structural share for DISTS/NBR2S, all families for the rest)
plus their own numpy timings. Fixed refits: full, lean (ln-2w's), lean2 (rank, the lean and the new blocks: no
embedding gather, no all-pairs pass), lean2s (lean2 without the NER/kNN edge build) and the greedy's pick.

    python outputs/mp_unified/lean_mlp2.py --train 2wiki=x4 --select 2wiki=select --read 2wiki=x1 --threads 2 \
        --save-models outputs/mp_unified/lean/l2-2w_models.pt --out outputs/mp_unified/lean/l2-2w.json
    python outputs/mp_unified/lean_mlp2.py --check 2wiki=select     # DISTS/DISTF against the look's compiled columns
    python outputs/mp_unified/lean_mlp2.py --selftest
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lean_mlp as LM  # noqa: E402  (sets the BLAS thread defaults before numpy loads)
import numpy as np  # noqa: E402
import torch  # noqa: E402

H = 3
NEW = ("DLIST", "DISTS", "DISTF", "WALKF", "NBRF", "NBR2S", "NBR2F")
NEW_W = {"DLIST": 3, "DISTS": 14, "DISTF": 14, "WALKF": 18, "NBRF": 7, "NBR2S": 3, "NBR2F": 3}
STRUCT_NEW = {"DISTS", "NBR2S"}
FULL_NEW = {"DISTF", "WALKF", "NBRF", "NBR2F"}
LM.STRUCT_USERS |= STRUCT_NEW | FULL_NEW
LM.NK_USERS |= FULL_NEW
DIST_COLS = ([f"{k}_h{t}" for t in range(1, H + 1) for k in ("log_seeds_at", "seedmass", "seedproto")]
             + [f"support_h{t}" for t in range(1, H + 1)] + ["log_branch_h2", "log_branch_h3"])
log = LM.log


# ── new blocks, per query ────────────────────────────────────────────────────


def pairs(n, u, v):
    """The view's binary symmetric pairs as fast_features._sym_csr builds them: both directions, diagonal dropped,
    duplicates collapsed (sorted by (u, v))."""
    u = u.astype(np.int64)
    v = v.astype(np.int64)
    ok = u != v
    key = np.unique(np.concatenate([u[ok] * n + v[ok], v[ok] * n + u[ok]]))
    return key // n, key % n


def seed_dist(n, u, v, S):
    """D[k, w]: the hop distance from seed k to node w over the pairs u -> v, to H (H + 1 beyond it or unreached)."""
    k = S.size
    D = np.full((k, n), H + 1, np.int8)
    if k == 0:
        return D
    rows = np.arange(k)
    F = np.zeros((k, n), bool)
    F[rows, S] = True
    D[rows, S] = 0
    seen = F.copy()
    flat = rows[:, None] * n + v[None, :]
    for h in range(1, H + 1):
        hit = F[:, u]
        if not hit.any():
            break
        nxt = np.zeros(k * n, bool)
        nxt[flat[hit]] = True
        nxt = nxt.reshape(k, n) & ~seen
        D[nxt] = h
        seen |= nxt
        F = nxt
    return D


def dist_block(n, Pn, S, s_seed, u, v):
    """DISTS/DISTF's 14 columns over one view's pairs (u, v), as DIST_COLS names them."""
    out = np.zeros((n, 14), np.float32)
    D = seed_dist(n, u, v, S)
    if S.size:
        G = (Pn @ Pn[S].T).astype(np.float64)
        gram = G[S]
        for t in range(1, H + 1):
            M = (D == t).T.astype(np.float64)
            cnt = M.sum(1)
            pn = np.sqrt(np.maximum(((M @ gram) * M).sum(1), 0.0))
            c = 3 * (t - 1)
            out[:, c] = np.log1p(cnt)
            out[:, c + 1] = M @ s_seed
            out[:, c + 2] = np.where(cnt > 0, (G * M).sum(1) / (pn + 1e-12), 0.0)
    deg = np.bincount(v, minlength=n).astype(np.float64)
    a = np.zeros(n)
    a[S] = 1.0
    p = np.zeros(n)
    p[S] = s_seed
    for t in range(H):
        br = np.bincount(v, weights=(a[u] > 0).astype(np.float64), minlength=n)
        a = np.bincount(v, weights=a[u], minlength=n)
        p = np.bincount(v, weights=p[u], minlength=n) / np.maximum(deg, 1.0)
        out[:, 9 + t] = p
        if t >= 1:
            out[:, 11 + t] = np.log1p(br)
    return out


def nbr2_block(n, cos, u, v):
    """Two-step walks over the pairs u -> v: their mean and max of the projected query cosine, log1p of their count."""
    out = np.zeros((n, 3), np.float32)
    if u.size == 0:
        return out
    c = cos.astype(np.float64)
    deg = np.bincount(v, minlength=n).astype(np.float64)
    s1 = np.bincount(v, weights=c[u], minlength=n)
    s2 = np.bincount(v, weights=s1[u], minlength=n)
    w2 = np.bincount(v, weights=deg[u], minlength=n)
    m1 = np.full(n, -np.inf)
    np.maximum.at(m1, v, c[u])
    m2 = np.full(n, -np.inf)
    np.maximum.at(m2, v, m1[u])
    has = w2 > 0
    out[:, 0] = np.where(has, s2 / np.maximum(w2, 1.0), 0.0)
    out[:, 1] = np.where(has & np.isfinite(m2), m2, 0.0)
    out[:, 2] = np.log1p(w2)
    return out


try:
    from numba import njit
except ImportError:                                   # the numpy reference above serves
    njit = None

if njit is not None:
    @njit(cache=False)
    def _dist_nb(n, ptr, idx, S, s_seed, G, gram, out):
        """dist_block over the view's csr (rows by u, the pairs' key order): one breadth-first pass per seed column."""
        k = S.shape[0]
        D = np.full((k, n), H + 1, np.int8)
        queue = np.empty(n, np.int64)
        for j in range(k):
            s = S[j]
            D[j, s] = 0
            head = 0
            tail = 1
            queue[0] = s
            while head < tail:
                x = queue[head]
                head += 1
                dx = D[j, x]
                if dx >= H:
                    continue
                for jj in range(ptr[x], ptr[x + 1]):
                    y = idx[jj]
                    if D[j, y] == H + 1:
                        D[j, y] = dx + 1
                        queue[tail] = y
                        tail += 1
        for w in range(n):
            for t in range(1, H + 1):
                cnt = 0.0
                mass = 0.0
                dot = 0.0
                nrm = 0.0
                for a in range(k):
                    if D[a, w] == t:
                        cnt += 1.0
                        mass += s_seed[a]
                        dot += G[w, a]
                        for b in range(k):
                            if D[b, w] == t:
                                nrm += gram[a, b]
                c = 3 * (t - 1)
                out[w, c] = np.log1p(cnt)
                out[w, c + 1] = mass
                if cnt > 0:
                    out[w, c + 2] = dot / (np.sqrt(max(nrm, 0.0)) + 1e-12)
        a_ = np.zeros(n)
        p = np.zeros(n)
        for j in range(k):
            a_[S[j]] = 1.0
            p[S[j]] = s_seed[j]
        a2 = np.empty(n)
        p2 = np.empty(n)
        for t in range(H):
            for x in range(n):
                acc = 0.0
                br = 0.0
                pc = 0.0
                for jj in range(ptr[x], ptr[x + 1]):
                    y = idx[jj]
                    acc += a_[y]
                    if a_[y] > 0:
                        br += 1.0
                    pc += p[y]
                deg = ptr[x + 1] - ptr[x]
                a2[x] = acc
                p2[x] = pc / deg if deg > 0 else 0.0
                out[x, 9 + t] = p2[x]
                if t >= 1:
                    out[x, 11 + t] = np.log1p(br)
            for x in range(n):
                a_[x] = a2[x]
                p[x] = p2[x]

    @njit(cache=False)
    def _nbr2_nb(n, ptr, idx, cos, out):
        s1 = np.zeros(n)
        m1 = np.full(n, -np.inf)
        deg = np.zeros(n)
        for x in range(n):
            for jj in range(ptr[x], ptr[x + 1]):
                c = cos[idx[jj]]
                s1[x] += c
                if c > m1[x]:
                    m1[x] = c
            deg[x] = ptr[x + 1] - ptr[x]
        for x in range(n):
            s2 = 0.0
            w2 = 0.0
            m2 = -np.inf
            for jj in range(ptr[x], ptr[x + 1]):
                y = idx[jj]
                s2 += s1[y]
                w2 += deg[y]
                if m1[y] > m2:
                    m2 = m1[y]
            if w2 > 0:
                out[x, 0] = s2 / w2
                if np.isfinite(m2):
                    out[x, 1] = m2
            out[x, 2] = np.log1p(w2)


def csr_of(n, u):
    """Row pointers of key-sorted pairs (rows by u; the column index array is v itself)."""
    return np.concatenate([[0], np.cumsum(np.bincount(u, minlength=n))]).astype(np.int64)


def dist_fast(n, Pn, S, s_seed, u, v):
    if njit is None:
        return dist_block(n, Pn, S, s_seed, u, v)
    out = np.zeros((n, 14), np.float32)
    G = (Pn @ Pn[S].T).astype(np.float64) if S.size else np.zeros((n, 0))
    _dist_nb(n, csr_of(n, u), v, S, s_seed.astype(np.float64), G, np.ascontiguousarray(G[S]), out)
    return out


def nbr2_fast(n, cos, u, v):
    if njit is None:
        return nbr2_block(n, cos, u, v)
    out = np.zeros((n, 3), np.float32)
    _nbr2_nb(n, csr_of(n, u), v, cos.astype(np.float64), out)
    return out


def warm():
    """JIT both kernels on a two-node graph before anything is timed."""
    u, v = pairs(3, np.asarray([0, 1]), np.asarray([1, 2]))
    Pn = np.eye(3, 4, dtype=np.float32)
    dist_fast(3, Pn, np.asarray([0], np.int64), np.asarray([1.0]), u, v)
    nbr2_fast(3, np.ones(3, np.float32), u, v)


def new_query(n, proj, q_emb, R, seeds, buckets, x_dense_cos, x_dense_rr, x_rrf, eu, ev, efam, efwd, ebwd, timer):
    """The new blocks of one query (float32), each timed (s) into timer. proj, q_emb, R, seeds and the edges as
    lean_mlp.lean_query takes them; x_* are the look's (float16) dense_cos, dense_rr and rrf columns. The normalised
    projections and the projected query cosine are lean_mlp's (its SEM timer carries them) and are not timed here."""
    P = proj.astype(np.float32)
    Pn = P / np.maximum(np.linalg.norm(P, axis=1), 1e-12)[:, None]
    q = q_emb.astype(np.float32) @ R
    q /= max(float(np.linalg.norm(q)), 1e-12)
    cos = Pn @ q
    out = {}
    t = time.perf_counter()
    listed = x_dense_rr.astype(np.float32) > 0
    dc = x_dense_cos.astype(np.float32)
    out["DLIST"] = np.stack([np.where(listed, dc, 0.0), listed, np.where(listed, 0.0, cos)], 1).astype(np.float32)
    timer["DLIST"] += time.perf_counter() - t
    valid = seeds >= 0
    S = seeds[valid].astype(np.int64)
    rrf = x_rrf.astype(np.float64)
    s_seed = rrf[S] / max(float(rrf.max()), 1e-12) if S.size else np.zeros(0)
    eu64, ev64 = eu.astype(np.int64), ev.astype(np.int64)
    st = efam == 0
    for name, sel, d_name, n2_name in (("S", st, "DISTS", "NBR2S"), ("F", np.ones_like(st), "DISTF", "NBR2F")):
        t = time.perf_counter()
        u, v = pairs(n, eu64[sel], ev64[sel])
        tp = time.perf_counter() - t
        t = time.perf_counter()
        out[d_name] = dist_fast(n, Pn, S, s_seed, u, v)
        timer[d_name] += time.perf_counter() - t + tp
        t = time.perf_counter()
        out[n2_name] = nbr2_fast(n, cos, u, v)
        timer[n2_name] += time.perf_counter() - t + tp
    # WALKF / NBRF: lean_mlp's own WALK and NBR, called on every family as one structural multigraph
    scratch = {b: 0.0 for b in ("SEM", "SEED", "WALK", "NBR")}
    fam0 = np.zeros_like(efam)
    sym = efam > 0
    fw = np.where(sym, 1, efwd).astype(efwd.dtype)
    bw = np.where(sym, 1, ebwd).astype(ebwd.dtype)
    L, _q, _Pn = LM.lean_query(n, proj, q_emb, R, seeds, buckets, eu, ev, fam0, fw, bw, scratch)
    t = time.perf_counter()
    s0 = np.zeros(n)
    s0[S] = 1.0
    wx = np.zeros((n, 2), np.float32)
    nx = np.zeros((n, 2), np.float32)
    for j, f in enumerate((1, 2)):
        m = efam == f
        wx[:, j] = np.log1p(np.bincount(ev64[m], weights=s0[eu64[m]], minlength=n))
        nx[:, j] = np.bincount(ev64[m], minlength=n) > 0
    te = time.perf_counter() - t
    out["WALKF"] = np.concatenate([L["WALK"], wx], 1)
    out["NBRF"] = np.concatenate([L["NBR"], nx], 1)
    timer["WALKF"] += scratch["WALK"] + te / 2
    timer["NBRF"] += scratch["NBR"] + te / 2
    return out


# ── loading ──────────────────────────────────────────────────────────────────


def chunk_rows(d, limit):
    """Yield (chunk arrays, query index) in LM.Carve's order: listed chunks only, sorted, up to limit rows."""
    recs = sorted(d.glob("record*.json"))
    listed = set()
    for r_ in recs:
        listed |= set(json.loads(r_.read_text(encoding="utf-8"))["chunks"])
    rows = 0
    for ch in sorted((d / "chunks").glob("c*.npz")):
        if int(ch.stem[1:]) not in listed:
            continue
        z = np.load(ch)
        arrays = {k: z[k] for k in ("q_pool_size", "q_edges", "proj", "q_emb", "x", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd",
                                    "q_seed_local", "q_seed_bucket")}
        for i in range(arrays["q_pool_size"].size):
            if limit is not None and rows >= limit:
                return
            yield arrays, i
            rows += 1


class Carve2(LM.Carve):
    """LM.Carve's rows, blocks and timings unchanged, plus the new blocks from a second pass over the same chunks."""

    def __init__(self, ds, carve, limit=None, root=LM.LOOK):
        super().__init__(ds, carve, limit, root)
        warm()
        R = LM.projection()
        ci = {c: i for i, c in enumerate(self.columns)}
        c_dc, c_dr, c_rrf = ci["dense_cos"], ci["dense_rr"], ci["rrf"]
        self.timer2 = {b: 0.0 for b in NEW}
        per_q = {b: [] for b in NEW}
        new = {b: [] for b in NEW}
        k = 0
        cache = None
        for arrays, i in chunk_rows(Path(root) / ds / carve, limit):
            if cache is None or cache[0] is not arrays:
                no = np.concatenate([[0], np.cumsum(arrays["q_pool_size"])])
                eo = np.concatenate([[0], np.cumsum(arrays["q_edges"])])
                cache = (arrays, no, eo)
            _, no, eo = cache
            a, b, ea, eb = no[i], no[i + 1], eo[i], eo[i + 1]
            n = int(arrays["q_pool_size"][i])
            assert n == int(self.n[k]), (ds, carve, k)
            X = arrays["x"][a:b]
            before = dict(self.timer2)
            out = new_query(n, arrays["proj"][a:b], arrays["q_emb"][i], R, arrays["q_seed_local"][i], arrays["q_seed_bucket"][i],
                            X[:, c_dc], X[:, c_dr], X[:, c_rrf], arrays["e_u"][ea:eb], arrays["e_v"][ea:eb],
                            arrays["e_fam"][ea:eb], arrays["e_fwd"][ea:eb], arrays["e_bwd"][ea:eb], self.timer2)
            for bk in NEW:
                new[bk].append(out[bk].astype(np.float16))
                per_q[bk].append(1e3 * (self.timer2[bk] - before[bk]))
            k += 1
        assert k == self.rows, (ds, carve, k, self.rows)
        self.new = {bk: np.concatenate(v) for bk, v in new.items()}
        self.lean_ms.update({bk: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                                  "mean": float(np.mean(v))} for bk, v in per_q.items()})
        self.widths.update(NEW_W)

    def block(self, b, idx):
        if b in NEW:
            return self.new[b][idx]
        return super().block(b, idx)


# ── main ─────────────────────────────────────────────────────────────────────


def costs_of(carves):
    out = []
    for c in carves:
        lm = dict(c.lean_ms)
        lm["SEMB"] = {"p50": 0.0}
        out.append((LM.profile(c.ds), c.struct_share, lm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="2wiki=x4")
    ap.add_argument("--select", default="2wiki=select")
    ap.add_argument("--read", default="2wiki=x1")
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--final-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.3, help="points of mean(R@5, FC@5) on select")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--fixed", default="", help="extra refits: name=blockA+blockB;name2=...")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--save-models")
    ap.add_argument("--load-models", help="read only: load the models from this .pt file and read the --read carves")
    ap.add_argument("--check", help="DS=CARVE: DISTS/DISTF against the look's compiled depth columns, then exit")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(a.threads)
    if a.check:
        return check(a)
    t0 = time.time()
    if a.load_models:
        return read_only(a, t0)
    tr = [Carve2(ds, cv, a.limit) for ds, cv in LM.parse_sets(a.train)]
    se = [Carve2(ds, cv, a.limit) for ds, cv in LM.parse_sets(a.select)]
    log(f"loaded train {[(c.ds, c.carve, c.rows) for c in tr]} select {[(c.ds, c.carve, c.rows) for c in se]} ({time.time() - t0:.0f}s)")
    log(f"new-block numpy ms p50 {({b: round(tr[0].lean_ms[b]['p50'], 3) for b in NEW})}")
    present = list(LM.COMPILED) + list(LM.LEAN) + list(NEW) if a.blocks == "all" else a.blocks.split("+")
    dead = []
    for b in list(present):
        if b in LM.LEAN:
            continue
        v = np.concatenate([c.block(b, np.arange(int(c.off[-1]))).astype(np.float32) for c in tr])
        if float(np.nanstd(v)) == 0.0:
            dead.append(b)
            present.remove(b)
    log(f"blocks {present}; constant on the training rows (dropped) {dead}")
    costs = costs_of(tr)
    out = {"look": "lean_mlp2", "args": vars(a), "blocks": present, "dead": dead,
           "lean_ms": {f"{c.ds}={c.carve}": c.lean_ms for c in tr + se},
           "profiles": {c.ds: LM.profile(c.ds) for c in tr}, "struct_share": {f"{c.ds}={c.carve}": c.struct_share for c in tr},
           "assumptions": ["lean_mlp's cost assumptions", "the new blocks timed in numpy here, each with the shared projection "
                           "and pair build counted in full (pessimistic when several are kept)",
                           "DLIST reads the dense list's scores at no gather cost (checked on live inputs by lean_time.py)"],
           "lean_mlp_sha256": hashlib.sha256((HERE / "lean_mlp.py").read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    models = {}
    pick = None
    if not a.no_greedy:
        log("fit with block dropout")
        model, curve = LM.fit(tr, se, present, "drop", a.epochs, a.lr, a.seed, a.hidden)
        out["drop_curve"] = curve
        path = LM.greedy(model, se, present, costs)
        out["path"] = path
        full_q = 0.5 * (path[0]["select"][0] + path[0]["select"][1])
        ok = [p for p in path if full_q - 0.5 * (p["select"][0] + p["select"][1]) <= a.tol / 100.0]
        pick = min(ok, key=lambda p: p["cost_ms"])
        out["pick"] = {k: pick[k] for k in ("blocks", "cost_ms", "select")}
        log(f"pick {pick['blocks']} at {pick['cost_ms']:.2f} ms (full {path[0]['cost_ms']:.2f} ms)")
        models["drop/full"] = (model, present, {b: 1.0 for b in present})
        models["drop/pick"] = (model, present, {b: (1.0 if b in pick["blocks"] else 0.0) for b in present})
    lean = [b for b in present if b in LM.NEVER or b in LM.LEAN]
    fixed = {"full": present, "lean": lean, "lean2": lean + [b for b in present if b in NEW],
             "lean2s": lean + [b for b in present if b in NEW and b not in FULL_NEW]}
    if pick is not None:
        fixed["pick"] = pick["blocks"]
    for part in [p for p in a.fixed.split(";") if p]:
        name, bl = part.split("=")
        fixed[name] = [b for b in bl.split("+") if b in present]
    out["fixed_sets"] = fixed
    out["fixed_costs_ms"] = {name: LM.cost_multi(bl, costs) for name, bl in fixed.items()}
    out["fixed_curves"] = {}
    for name, bl in fixed.items():
        log(f"refit {name} ({len(bl)} blocks, {LM.cost_multi(bl, costs):.2f} ms): {bl}")
        m2, cur = LM.fit(tr, se, bl, "fixed", a.final_epochs, a.lr, a.seed, a.hidden)
        out["fixed_curves"][name] = cur
        models[name] = (m2, bl, {b: 1.0 for b in bl})
    if a.save_models:
        Path(a.save_models).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"models": {name: {"state": m.state_dict(), "blocks": bl, "keep": keep, "widths": m.widths, "hidden": a.hidden}
                               for name, (m, bl, keep) in models.items()},
                    "pick": out.get("pick"), "present": present, "dead": dead, "args": vars(a)}, a.save_models)
        log(f"saved {len(models)} models to {a.save_models}")
    del tr, se
    read_all(a, models, pick, out)
    out["seconds"] = time.time() - t0
    write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def write(a, out):
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")


def read_only(a, t0):
    blob = torch.load(a.load_models, weights_only=False)
    models = {}
    for name, d in blob["models"].items():
        m = LM.LeanMLP(d["blocks"], d["widths"], d["hidden"])
        m.load_state_dict(d["state"])
        models[name] = (m, d["blocks"], d["keep"])
    pick = blob.get("pick")
    out = {"look": "lean_mlp2", "mode": "read_only", "args": vars(a), "loaded": a.load_models,
           "loaded_sha256": hashlib.sha256(Path(a.load_models).read_bytes()).hexdigest(), "fit_args": blob.get("args"),
           "pick": pick, "blocks": blob.get("present"), "dead": blob.get("dead"),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    log(f"loaded {list(models)} from {a.load_models}")
    read_all(a, models, pick, out)
    out["seconds"] = time.time() - t0
    write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def read_all(a, models, pick, out):
    """LM.read_all on Carve2 carves."""
    rng = np.random.default_rng(20261003)
    reads, read_costs, read_lean_ms, read_rows = {}, {}, {}, {}
    for ds, cv in LM.parse_sets(a.read):
        key = f"{ds}={cv}"
        c = Carve2(ds, cv, a.limit)
        read_rows[key] = {"rows": c.rows, "chunks": c.chunks_read, "n_chunks": c.n_chunks, "carve_queries": c.carve_queries}
        (prof, share, lm), = costs_of([c])
        read_lean_ms[key] = c.lean_ms
        read_costs[key] = {name: LM.cost_ms(bl, prof, share, lm) for name, (_m, bl, _k) in models.items()
                           if not name.startswith("drop/")}
        if pick is not None:
            read_costs[key]["drop/pick"] = LM.cost_ms(pick["blocks"], prof, share, lm)
        log(f"read {key} ({c.rows} rows); approximate compile ms {({k: round(v, 2) for k, v in read_costs[key].items()})}")
        base_rows = {}
        for name, (m, bl, keep) in models.items():
            ref = base_rows.get("drop/full" if name == "drop/pick" else "full")
            r, rows = LM.read_one(m, c, bl, keep, rng, ref)
            if name in ("full", "drop/full"):
                base_rows[name] = rows
            reads[f"{name}@{key}"] = r
            log(f"  {name}@{key}: {LM.fmt(r)}")
        del c
    out["read_costs_ms"] = read_costs
    out["read_lean_ms"] = read_lean_ms
    out["read_rows"] = read_rows
    out["reads"] = reads


# ── checks ───────────────────────────────────────────────────────────────────


COMPILED_PAIRS = {"seeds_at_h{t}": ("log_seeds_at_h{t}", (2, 3)), "seedmass_h{t}": ("seedmass_h{t}", (3,)),
                  "has_h{t}": (None, (2, 3)), "support_h{t}": ("support_h{t}", (2, 3)), "branch_h{t}": ("log_branch_h{t}", (3,))}


def check(a):
    """DISTS/DISTF against the look's own compiled depth columns (float16), where the look has them."""
    ds, cv = a.check.split("=")
    c = Carve2(ds, cv, a.limit)
    ci = {col: i for i, col in enumerate(c.columns)}
    out = {}
    for view, blk in (("STRUCT", "DISTS"), ("FULL", "DISTF")):
        mine = c.new[blk].astype(np.float32)
        for comp, (name, ts) in COMPILED_PAIRS.items():
            for t in ts:
                col = comp.format(t=t) + f"_{view}"
                if col not in ci:
                    continue
                ref = c.x[:, ci[col]].astype(np.float32)
                if name is None:
                    got = (mine[:, DIST_COLS.index(f"log_seeds_at_h{t}")] > 0).astype(np.float32)
                else:
                    got = mine[:, DIST_COLS.index(name.format(t=t))].astype(np.float16).astype(np.float32)
                d = np.abs(got - ref)
                tol = 2e-3 * np.maximum(np.abs(ref), 1.0)
                out[col] = {"rows": int(ref.size), "match": float((d <= tol).mean()), "max_abs": float(d.max()),
                            "ref_nonzero": float((ref != 0).mean())}
                log(f"  {col:22s} vs {blk}.{name or 'has'}: match {out[col]['match']:.5f}, max |d| {out[col]['max_abs']:.4g}, "
                    f"nonzero {out[col]['ref_nonzero']:.3f}")
    for col in ("paths_h3_STRUCT", "paths_h3_FULL"):
        if col in ci:
            blk = "WALK" if col.endswith("STRUCT") else "WALKF"
            got = (c.lean["WALK"] if blk == "WALK" else c.new["WALKF"])[:, 2].astype(np.float32)
            ref = c.x[:, ci[col]].astype(np.float32)
            d = np.abs(got - ref)
            out[col] = {"match": float((d <= 2e-3 * np.maximum(np.abs(ref), 1.0)).mean()), "max_abs": float(d.max()),
                        "note": f"{blk} col 2 counts multi-edges; the compile's view is binary"}
            log(f"  {col:22s} vs {blk}[:, 2] (multi-edge walks): match {out[col]['match']:.5f}, max |d| {out[col]['max_abs']:.4g}")
    log(f"new-block numpy ms {({b: round(c.lean_ms[b]['p50'], 3) for b in NEW})}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"check": a.check, "rows": c.rows, "columns": out, "lean_ms": c.lean_ms}, indent=1),
                               encoding="utf-8")
    return 0


def selftest():
    rng = np.random.default_rng(1)
    n = 12
    E = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 6), (6, 7), (8, 9), (2, 6), (5, 10), (10, 11)]
    eu = np.asarray([x for a, b in E for x in (a, b)] + [2, 2], np.int16)
    ev = np.asarray([x for a, b in E for x in (b, a)] + [3, 2], np.int16)        # a duplicate and a self-loop
    u, v = pairs(n, eu, ev)
    A = np.zeros((n, n))
    for a_, b_ in E:
        A[a_, b_] = A[b_, a_] = 1
    assert np.array_equal(np.bincount(v, minlength=n), A.sum(1)) and len(u) == 2 * len(E), "pairs: binary, symmetric, no diagonal"
    S = np.asarray([0, 4, 8, 0])                                                   # a repeated seed column, as a list may give
    s_seed = np.asarray([1.0, 0.5, 0.25, 1.0])
    proj = rng.standard_normal((n, LM.PROJ_DIM)).astype(np.float32)
    Pn = proj / np.linalg.norm(proj, axis=1, keepdims=True)
    out = dist_block(n, Pn, S, s_seed, u, v)
    Dist = np.full((n, n), 99)                                                     # all-pairs BFS, brute force
    for src in range(n):
        Dist[src, src] = 0
        fr = [src]
        for h in range(1, 10):
            nx = [w for x in fr for w in np.flatnonzero(A[x]) if Dist[src, w] == 99]
            for w in nx:
                Dist[src, w] = h
            fr = list(set(nx))
    for t in range(1, H + 1):
        for w in range(n):
            ks = [k for k in range(S.size) if Dist[w, S[k]] == t]
            c = 3 * (t - 1)
            assert np.isclose(out[w, c], np.log1p(len(ks)), atol=1e-6), (t, w)
            assert np.isclose(out[w, c + 1], s_seed[ks].sum(), atol=1e-6), (t, w)
            if ks:
                m = Pn[S[ks]].sum(0)
                assert np.isclose(out[w, c + 2], Pn[w] @ m / np.linalg.norm(m), atol=1e-5), (t, w)
            else:
                assert out[w, c + 2] == 0
    a_ = np.zeros(n)
    a_[S] = 1.0
    p = np.zeros(n)
    for k in range(S.size):
        p[S[k]] = s_seed[k]                                                       # s[seeds_local] = ... (last write wins)
    deg = A.sum(1)
    for t in range(H):
        br = A @ (a_ > 0)
        a_ = A @ a_
        p = (A @ p) / np.maximum(deg, 1)
        assert np.allclose(out[:, 9 + t], p, atol=1e-6), ("support", t)
        if t >= 1:
            assert np.allclose(out[:, 11 + t], np.log1p(br), atol=1e-6), ("branch", t)
    print("DIST block equals brute force: seeds at exactly t (all-pairs BFS) with mass and projected proto, a repeated "
          "seed column, support and branch over the binary symmetric view (duplicates and the self-loop dropped)")
    cos = Pn @ (rng.standard_normal(LM.PROJ_DIM) / 10)
    o2 = nbr2_block(n, cos, u, v)
    A2 = A @ A
    for w in range(n):
        wt = A2[w].sum()
        if wt > 0:
            assert np.isclose(o2[w, 0], A2[w] @ cos / wt, atol=1e-6)
            mx = max(cos[z] for y in np.flatnonzero(A[w]) for z in np.flatnonzero(A[y]))
            assert np.isclose(o2[w, 1], mx, atol=1e-6)
        assert np.isclose(o2[w, 2], np.log1p(wt), atol=1e-6)
    print("NBR2 equals brute force: two-step walk-weighted mean and max of the query cosine, log1p of the walk count")
    if njit is not None:
        warm()
        for trial in range(20):
            m_ = int(rng.integers(5, 40))
            eu_r = rng.integers(0, m_, size=3 * m_)
            ev_r = rng.integers(0, m_, size=3 * m_)
            ur, vr = pairs(m_, eu_r, ev_r)
            Sr = rng.choice(m_, size=int(rng.integers(0, 6)), replace=True).astype(np.int64)
            sr = rng.uniform(0.1, 1.0, Sr.size)
            Pr = rng.standard_normal((m_, LM.PROJ_DIM)).astype(np.float32)
            Pr /= np.linalg.norm(Pr, axis=1, keepdims=True)
            cr = rng.standard_normal(m_).astype(np.float32)
            assert np.allclose(dist_fast(m_, Pr, Sr, sr, ur, vr), dist_block(m_, Pr, Sr, sr, ur, vr), atol=1e-6), trial
            assert np.allclose(nbr2_fast(m_, cr, ur, vr), nbr2_block(m_, cr, ur, vr), atol=1e-6), trial
        print("numba DIST and NBR2 kernels equal the numpy reference on 20 random multigraphs (repeated seeds, no seed)")
    # new_query end to end on a small query: struct + NER + kNN entries stored both ways, as the look stores them
    efam = np.asarray([0] * 12 + [1] * 4 + [2] * 2, np.int8)
    eu2 = np.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 1, 6, 0, 7, 7, 0, 9, 8], np.int16)
    ev2 = np.asarray([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 1, 7, 0, 0, 7, 8, 9], np.int16)
    efwd = np.asarray([1, 0] * 6 + [0] * 6, np.int8)
    ebwd = np.asarray([0, 1] * 6 + [0] * 6, np.int8)
    seeds = np.asarray([0, 4, -1, -1])
    buckets = np.asarray([0, 1, -1, -1])
    R = LM.projection()
    proj16 = rng.standard_normal((n, LM.PROJ_DIM)).astype(np.float16)
    q_emb = rng.standard_normal(1536).astype(np.float32)
    x_dc = rng.uniform(0.1, 0.6, n).astype(np.float16)
    x_dr = np.where(np.arange(n) % 3 == 0, 0.0, 1.0 / (1 + np.arange(n))).astype(np.float16)
    x_rrf = rng.uniform(0.0, 0.03, n).astype(np.float16)
    timer = {b: 0.0 for b in NEW}
    o = new_query(n, proj16, q_emb, R, seeds, buckets, x_dc, x_dr, x_rrf, eu2, ev2, efam, efwd, ebwd, timer)
    assert all(o[b].shape == (n, NEW_W[b]) and o[b].dtype == np.float32 for b in NEW), {b: o[b].shape for b in NEW}
    listed = x_dr > 0
    assert np.allclose(o["DLIST"][:, 0], np.where(listed, x_dc.astype(np.float32), 0)) and np.array_equal(o["DLIST"][:, 1], listed)
    assert np.all(o["DLIST"][listed, 2] == 0) and np.all(o["DLIST"][~listed, 2] != 0)
    tm = {b: 0.0 for b in ("SEM", "SEED", "WALK", "NBR")}
    L_s, _, _ = LM.lean_query(n, proj16, q_emb, R, seeds, buckets, eu2, ev2, efam, efwd, ebwd, tm)
    # node 7 is reached only over NER (0-7), node 8/9 only over kNN between themselves
    assert L_s["WALK"][7, 0] == 0 and o["WALKF"][7, 0] > 0 and o["WALKF"][7, 16] > 0 and o["WALKF"][7, 17] == 0
    assert o["NBRF"][8, 6] == 1 and o["NBRF"][8, 5] == 0 and o["NBRF"][7, 5] == 1
    assert o["DISTS"][7, 0] == 0 and o["DISTF"][7, 0] > 0, "seed 0 reaches node 7 at hop 1 over the FULL view only"
    assert o["WALKF"][7, 3] > 0, "NER pairs walk forward as well (both directions)"
    assert all(timer[b] > 0 for b in NEW)
    print("new_query: DLIST's listed score and fallback; WALKF/NBRF/DISTF see NER and kNN pairs that the structural "
          "blocks do not; every block timed")
    for b in NEW:
        assert b in LM.STRUCT_USERS and ((b in FULL_NEW) == (b in LM.NK_USERS)) or b == "DLIST"
    prof = {"retrieval": 2.0, "edges": 4.0, "topology": 2.0, "C": 1.5, "gcs": 0.1, "D": 2.0, "B": 0.0, "depth_basis": 1.0,
            "typed_basis": 0.0}
    lm = {b: {"p50": 0.1} for b in LM.LEAN + NEW}
    assert abs(LM.cost_ms(["rank", "DLIST"], prof, 0.5, lm) - (0.5 + 0.1)) < 1e-9, "DLIST: no gather, no edge"
    assert abs(LM.cost_ms(["rank", "DISTS"], prof, 0.5, lm) - (0.5 + 2.0 + 0.1)) < 1e-9, "DISTS: structural edges only"
    assert abs(LM.cost_ms(["rank", "DISTF"], prof, 0.5, lm) - (0.5 + 4.0 + 0.1)) < 1e-9, "DISTF: every family's edges"
    print("cost model: DLIST needs neither gather nor edges, DISTS the structural edges, DISTF all of them")
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
