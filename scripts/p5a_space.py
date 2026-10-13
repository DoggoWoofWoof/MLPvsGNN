"""P5a: the node space read as a retriever on its own (docs/P5A_NODE_SPACE.md). A diagnostic: no ranking model trains
or reads, and nothing is adopted.

    python scripts/p5a_space.py base [--host] [--limit R]
    python scripts/p5a_space.py train DATASET [--host] [--device cuda] [--threads 4] [--limit Q --steps S]
    python scripts/p5a_space.py map
    python scripts/p5a_space.py read1 DATASET --part K/N [--host] [--limit Q]
    python scripts/p5a_space.py read DATASET [--host] [--device cuda] [--threads 4] [--limit Q --steps S]
    python scripts/p5a_space.py verdict
    python scripts/p5a_space.py --selftest

base: the frame (the mean and the top 64 principal directions of up to FRAME_N frozen node vectors from each corpus)
and every corpus's start u0 = P (x - mu), kept at 16 bits.
train: the space u = u0 + delta and its offsets, fit by a translation objective with online hard EM over the latent
offsets, kept at 16 bits; V0's offsets (every node held at its start); on the five training datasets, the chains'
evidence on up to SAMPLE questions of the fit carve, for both spaces, read on the kept tables.
map: one chain map per space (V's and V0's), fit by EM on the five training datasets' evidence.
read1: zsp's top 5 on the s1eval sample, through P2a's reader (its compile and read, unmasked), one of P2a's parts.
read: on the s1eval sample, every order (V, V0, VO, H2, H1) and both reads (miss, low).
--limit (base, train, read1, read): a development run. It writes nothing and logs no measure, only that every step
ran, the checks and the fit's loss. Where a stage's inputs are not on disk, a development train or read builds them in
memory: a frame from this corpus, a short space (--steps) and a zero map.
"""
from __future__ import annotations

import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "read1":
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
               "LEAN_TIME_THREADS"):
        os.environ[_v] = "1"                  # P2a's reader reads with one thread, as P2a part 2 did
else:
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "4")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs" / "p5a"
SPACE = OUT / "space"
FRAME = SPACE / "frame.npz"
BASE = SPACE / "base.json"
MAP = OUT / "map.npz"
MAP_JSON = OUT / "map.json"
READ1 = OUT / "read1"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
CSR = ROOT / "outputs" / "m3b" / "csr"
P2A = ROOT / "outputs" / "p2a"
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
TRAIN5 = SIX[:5]
KB = ("metaqa", "webqsp")
FAMS = ("structural", "ner", "knn")
READ1_PARTS = {"metaqa": 8, "squad": 9, "musique": 2, "hotpotqa": 6, "2wiki": 10, "webqsp": 2}   # P2a part 2's

DIM = 64
FRAME_N = 50_000          # frozen node vectors per corpus in the frame's sample (all of a smaller corpus)
K_LATENT = 8              # latent offsets per family without named relations
KMEANS_N = 200_000        # start displacements in a latent family's k-means
KMEANS_IT = 20
STEPS = 3000              # steps of the space fit
BATCH = 65536             # edges per step, shared equally by the families
NEG = 16                  # uniform random tails per edge
LR = 1e-3                 # SparseAdam (the nodes) and Adam (the offsets)
LAMBDA = 1.0              # the prior pulling each step's touched nodes back to their start
GAMMA_Q = 0.05            # the margin: this quantile of the start's distance between random node pairs
GAMMA_PAIRS = 100_000
SAMPLE = 2000             # questions per carve, seed 0
TOP1, TOP2, TOPK12 = 8, 32, 8
PRUNE = 32                # chains per fit question whose golds' ranks are counted
VO_FIRST = 64             # first hops of a two-hop search for the chains nearest a gold
EM_MAX, EM_TOL, L2 = 200, 1e-9, 1e-3
TOL = 1e-5                # a node is nearer than a gold only beyond TOL (|t|^2 + |u_g|^2): float32 rounding
NMAX = 1000
GRID = {"miss": (10, 25, 50, 100, 200, 500, 1000), "low": (5, 10, 25, 50, 100)}
ORDERS = ("V", "V0", "VO", "H2", "H1")
RING_CLASSES = 3          # a gold's ring from the seeds: 1, 2, or beyond 2
TIE = 0.002
BOOT = 2000
QBATCH = 2000             # questions per pass over the frozen vectors (H2's cosines): the whole sample
CHUNK = 262_144           # nodes per distance block
H1_SETTING = {"name": "h3_c25_v2000", "hops": 3, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 2000}
R1_SHARE = 0.01           # read1's zsp R@5 may differ from P2a's filed read on at most this share of a dataset


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def ids_sha(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def save_npz(path, **arrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def save_npy(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def peak_rss():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return int(getattr(mi, "peak_wset", 0) or getattr(mi, "rss", 0))
    except Exception:
        return 0


def full_fp32(torch):
    """Distances and ranks need full float32 matmuls: no TF32 on the card."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def gpu_peak(torch, dev):
    return int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0


def sample_of(q):
    """The questions read: up to SAMPLE positions of a carve of q questions, drawn with seed 0, in carve order."""
    return np.sort(np.random.default_rng(0).permutation(q)[:SAMPLE])


def fit_carve(name):
    return "s1fit" if name == "musique" else "fit"


# ── the frame and the start ──────────────────────────────────────────────────


def frame_of(sample):
    """mu and the top DIM principal directions (rows) of the sample, from its covariance in float64; each direction's
    sign makes its largest coordinate positive."""
    x = np.asarray(sample)
    n, d = x.shape
    mu = np.zeros(d, np.float64)
    for lo in range(0, n, 65536):
        mu += x[lo:lo + 65536].astype(np.float64).sum(0)
    mu /= n
    cov = np.zeros((d, d), np.float64)
    for lo in range(0, n, 65536):
        c = x[lo:lo + 65536].astype(np.float64) - mu
        cov += c.T @ c
    cov /= max(1, n - 1)
    w, v = np.linalg.eigh(cov)
    o = np.argsort(-w, kind="stable")[:DIM]
    P = v[:, o].T
    s = np.sign(P[np.arange(P.shape[0]), np.abs(P).argmax(1)])
    s[s == 0] = 1.0
    return mu.astype(np.float32), (P * s[:, None]).astype(np.float32), w[o]


def project(x, mu, P):
    return ((np.asarray(x, np.float32) - mu) @ P.T).astype(np.float32)


def qframe(q, mu, P):
    """A question's input: its frozen vector in the frame, with a trailing 1."""
    return np.r_[project(np.asarray(q, np.float32)[None], mu, P)[0], np.float32(1.0)].astype(np.float32)


def margin(u0, rng, pairs=GAMMA_PAIRS):
    n = u0.shape[0]
    a = rng.integers(0, n, pairs)
    b = rng.integers(0, n, pairs)
    keep = a != b
    d = np.linalg.norm(np.asarray(u0[a[keep]], np.float64) - np.asarray(u0[b[keep]], np.float64), axis=1)
    return float(np.quantile(d, GAMMA_Q))


# ── the space ────────────────────────────────────────────────────────────────


class Family:
    """One link family's edges h -> t. Named: rel holds each edge's relation (offset +r). Latent: the E-step picks each
    edge's offset among n_rel and, when signed (ner, knn: one edge per unordered pair), its sign."""

    def __init__(self, name, h, t, rel=None, n_rel=K_LATENT, signed=True):
        self.name = name
        self.h = np.asarray(h, np.int64)
        self.t = np.asarray(t, np.int64)
        self.rel = None if rel is None else np.asarray(rel, np.int64)
        self.n_rel = int(n_rel)
        self.signed = bool(signed) and rel is None

    @property
    def named(self):
        return self.rel is not None


def assign(disp, r, signed):
    """The hard E-step (torch): each displacement's offset k and sign s minimising |disp - s r_k|."""
    import torch
    dot = disp @ r.T
    if signed:
        k = ((r * r).sum(1)[None, :] - 2.0 * dot.abs()).argmin(1)
        s = torch.sign(dot.gather(1, k[:, None]).squeeze(1))
        s[s == 0] = 1.0
    else:
        k = ((r * r).sum(1)[None, :] - 2.0 * dot).argmin(1)
        s = torch.ones(k.shape, dtype=disp.dtype, device=disp.device)
    return k, s


def assign_np(disp, r, signed):
    dot = disp @ r.T
    if signed:
        k = ((r * r).sum(1)[None, :] - 2.0 * np.abs(dot)).argmin(1)
        s = np.sign(dot[np.arange(k.size), k])
        s[s == 0] = 1.0
    else:
        k = ((r * r).sum(1)[None, :] - 2.0 * dot).argmin(1)
        s = np.ones(k.size)
    return k, s


def kmeans(disp, K, rng, signed, iters=KMEANS_IT):
    """k-means over displacements (float64, CPU); signed folds each displacement onto its offset's side."""
    m = disp.shape[0]
    r = disp[rng.choice(m, K, replace=m < K)].copy()
    for _ in range(iters):
        k, s = assign_np(disp, r, signed)
        sd = disp * s[:, None]
        new = np.stack([np.bincount(k, weights=sd[:, j], minlength=K) for j in range(disp.shape[1])], 1)
        cnt = np.bincount(k, minlength=K)
        empty = cnt == 0
        new = new / np.maximum(cnt, 1)[:, None]
        new[empty] = r[empty]
        r = new
    return r


def init_offsets(u0, f, rng):
    """Named: each relation's mean start displacement (0 for a relation without edges). Latent: k-means over up to
    KMEANS_N sampled start displacements."""
    if not f.h.size:
        return np.zeros((f.n_rel, DIM), np.float64)
    if f.named:
        acc = np.zeros((f.n_rel, DIM), np.float64)
        for lo in range(0, f.h.size, 1 << 20):
            d = np.asarray(u0[f.t[lo:lo + (1 << 20)]], np.float64) - np.asarray(u0[f.h[lo:lo + (1 << 20)]], np.float64)
            rel = f.rel[lo:lo + (1 << 20)]
            for j in range(DIM):
                acc[:, j] += np.bincount(rel, weights=d[:, j], minlength=f.n_rel)
        cnt = np.bincount(f.rel, minlength=f.n_rel)
        return acc / np.maximum(cnt, 1)[:, None]
    m = min(f.h.size, KMEANS_N)
    sel = np.sort(rng.choice(f.h.size, m, replace=False))
    d = np.asarray(u0[f.t[sel]], np.float64) - np.asarray(u0[f.h[sel]], np.float64)
    return kmeans(d, f.n_rel, rng, f.signed)


def fit_space(u0, fams, device, seed=0, steps=STEPS, logf=log, lam=LAMBDA, fit_nodes=True, batch=BATCH):
    """u = u0 + delta and the families' offsets, fit to the edges by softplus(d - gamma) plus the mean over NEG uniform
    tails of softplus(gamma - d), d = |u_h + r - u_t|, with a prior of weight lam on each step's touched nodes' squared
    delta. fit_nodes=False holds every node at its start (only the offsets fit). Returns u (device float32), the
    offsets (CPU float32 tensors) and the fit's diagnostics."""
    import torch
    F = torch.nn.functional
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = u0.shape[0]
    U0 = torch.from_numpy(np.ascontiguousarray(u0)).to(device)      # the start as kept (16 bits)
    gamma = margin(u0, rng)
    emb = None
    if fit_nodes:
        emb = torch.nn.Embedding(n, DIM, sparse=True).to(device)
        torch.nn.init.zeros_(emb.weight)

    def vec(i):
        return U0[i].float() + emb(i) if fit_nodes else U0[i].float()

    offs, H, T, R = [], [], [], []
    for f in fams:
        offs.append(torch.nn.Parameter(torch.from_numpy(init_offsets(u0, f, rng).astype(np.float32)).to(device)))
        H.append(torch.from_numpy(f.h.astype(np.int32)).to(device))
        T.append(torch.from_numpy(f.t.astype(np.int32)).to(device))
        R.append(torch.from_numpy(f.rel.astype(np.int32)).to(device) if f.named else None)
    live = [i for i, f in enumerate(fams) if f.h.size]
    per = max(1, batch // max(1, len(live)))
    opt_o = torch.optim.Adam(offs, lr=LR)
    opt_u = torch.optim.SparseAdam(list(emb.parameters()), lr=LR) if fit_nodes else None
    t0 = time.time()
    hist = []
    for step in range(steps if live else 0):
        loss = 0.0
        touched = []
        for i in live:
            f = fams[i]
            j = torch.randint(0, H[i].numel(), (per,), generator=gen, device=device)
            h, t = H[i][j].long(), T[i][j].long()
            uh = vec(h)
            ut = vec(t)
            if f.named:
                r = offs[i][R[i][j].long()]
            else:
                with torch.no_grad():
                    k, s = assign(ut - uh, offs[i], f.signed)
                r = offs[i][k] * s[:, None]
            tn = torch.randint(0, n, (per, NEG), generator=gen, device=device)
            un = vec(tn)
            tgt = uh + r
            dp = (tgt - ut).norm(dim=1)
            dn = (tgt[:, None, :] - un).norm(dim=2)
            loss = loss + (F.softplus(dp - gamma).mean() + F.softplus(gamma - dn).mean()) / len(live)
            touched += [h, t, tn.reshape(-1)]
        if fit_nodes:
            loss = loss + lam * (emb(torch.cat(touched)) ** 2).sum(1).mean()
        opt_o.zero_grad()
        if opt_u is not None:
            opt_u.zero_grad()
        loss.backward()
        opt_o.step()
        if opt_u is not None:
            opt_u.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            lv = float(loss.detach())
            hist.append((step, round(lv, 5)))
            logf(f"  step {step}/{steps} loss {lv:.4f} ({time.time() - t0:.0f}s)")
    if opt_u is not None:
        opt_u.state.clear()                          # the moments go before u is formed: the card's peak
        emb.weight.grad = None
    with torch.no_grad():
        u = U0.float() + emb.weight if fit_nodes else U0.float()
        info = {"gamma": gamma, "loss_hist": hist, "seconds": round(time.time() - t0, 1), "fit_nodes": fit_nodes,
                "delta_rms": float(emb.weight.pow(2).sum(1).mean().sqrt()) if fit_nodes else 0.0, "families": {}}
        for i, f in enumerate(fams):
            if not f.h.size:
                info["families"][f.name] = {"edges": 0}
                continue
            m = min(H[i].numel(), 1_000_000)
            j = torch.randperm(H[i].numel(), generator=gen, device=device)[:m]
            d = u[T[i][j].long()] - u[H[i][j].long()]
            if f.named:
                rr, share = offs[i][R[i][j].long()], None
            else:
                k, s = assign(d, offs[i], f.signed)
                rr = offs[i][k] * s[:, None]
                share = [round(x, 4) for x in (torch.bincount(k, minlength=f.n_rel).float() / k.numel()).tolist()]
            r2 = ((d - rr) ** 2).sum(1)
            info["families"][f.name] = {"edges": int(f.h.size), "offsets": f.n_rel, "named": f.named,
                                        "signed": f.signed, "residual2_median": float(r2.median()),
                                        "displacement2_median": float((d ** 2).sum(1).median()),
                                        "latent_share": share}
        offsets = [o.detach().float().cpu() for o in offs]
    del emb, opt_u, opt_o, H, T, R
    return u, offsets, info


def csr_dir(op):
    return Path(op.m3b_compile.CSR_CACHE)


def edge_families(name, csr=CSR):
    """The three link families' edges, self-loops dropped. structural: directed (its typed entries whose row node is
    the source), named on metaqa and webqsp, one relation written down elsewhere (latent, sign fixed); ner and knn:
    one edge per unordered pair (latent, signed)."""
    out = []
    for f in FAMS:
        p = csr / f"{name}_{f}.npz"
        if not p.exists():
            raise SystemExit(f"{p} is missing: the stage reads M3B's stores, it does not build them")
        with np.load(p) as z:
            if f == "structural":
                ip, col, rel, dr = z["typed_indptr"], z["typed_col"], z["typed_rel"], z["typed_dir"]
                n_rel = int(z["rel_count"].size)
                row = np.repeat(np.arange(ip.size - 1, dtype=np.int64), np.diff(ip))
                keep = (dr == 1) & (row != col)
                h, t, r = row[keep], col[keep].astype(np.int64), rel[keep].astype(np.int64)
                if name in KB:
                    out.append(Family(f, h, t, rel=r, n_rel=n_rel))
                else:
                    if n_rel != 1:
                        raise SystemExit(f"{p}: {n_rel} relations on a passage corpus")
                    out.append(Family(f, h, t, signed=False))
            else:
                ip, col = z["indptr"], z["col"]
                row = np.repeat(np.arange(ip.size - 1, dtype=np.int64), np.diff(ip))
                keep = row < col
                out.append(Family(f, row[keep], col[keep].astype(np.int64)))
    return out


def family_stores(S4, ds, name, csr):
    """M3B's three stores of the corpus (never built here) and their adjacency as the ring kernel reads it."""
    stores = []
    for f in FAMS:
        if not (csr / f"{name}_{f}.npz").exists():
            raise SystemExit(f"{csr / f'{name}_{f}.npz'} is missing: the stage reads M3B's stores, it does not build them")
        stores.append(S4.m3b_pools.load_or_build_store(ds, f, csr))
    adj = [(np.ascontiguousarray(s.indptr, np.int64), np.ascontiguousarray(s.col, np.int32)) for s in stores]
    return stores, adj


# ── chains and the map ───────────────────────────────────────────────────────


def chain_table(offsets):
    """Every signed offset of a corpus, families in FAMS order, each offset forward then backward: (C, DIM)."""
    import torch
    rows = []
    for o in offsets:
        for k in range(o.shape[0]):
            rows += [o[k], -o[k]]
    return torch.stack(rows) if rows else torch.zeros(0, DIM)


def chain_labels(fam_names, sizes):
    return [f"{f}:{k}:{s}" for f, n in zip(fam_names, sizes) for k in range(n) for s in ("+", "-")]


def decode(c, C):
    """A chain id: c < C is the one-hop chain c; C + c1 C + c2 is the two-hop chain (c1, c2)."""
    return (int(c), -1) if c < C else divmod(int(c) - C, C)


def chain_steps(R, chains):
    """Each chain's displacement: r_c1, plus r_c2 for a two-hop chain. chains: [(c1, c2 or -1)]."""
    import torch
    a = torch.as_tensor([c for c, _ in chains], dtype=torch.long, device=R.device)
    b = torch.as_tensor([max(c, 0) for _, c in chains], dtype=torch.long, device=R.device)
    two = torch.as_tensor([c >= 0 for _, c in chains], dtype=R.dtype, device=R.device)
    return R[a] + R[b] * two[:, None]


def map_logprobs(Wm, x, r):
    """log pi over [one-hop chains (C), two-hop chains (C * C)] per question; x (Q, DIM + 1), r (C, DIM)."""
    import torch
    W1, W2, g = Wm
    gl = x @ g
    lp1 = torch.log_softmax((x @ W1.T) @ r.T, 1)
    lp2 = torch.log_softmax((x @ W2.T) @ r.T, 1)
    one = torch.nn.functional.logsigmoid(gl)[:, None] + lp1
    two = torch.nn.functional.logsigmoid(-gl)[:, None, None] + lp1[:, :, None] + lp2[:, None, :]
    return one, two.reshape(two.shape[0], -1)


def zero_map():
    import torch
    return (torch.zeros(DIM, DIM + 1), torch.zeros(DIM, DIM + 1), torch.zeros(DIM + 1))


def resp_stats(resp, C):
    """A question's responsibilities over [one-hop (C), two-hop (C * C)] as what the M-step reads: each first hop's
    mass (one-hop and two-hop together), each second hop's, and the one-hop and two-hop totals."""
    r1, r2 = resp[:, :C], resp[:, C:].reshape(-1, C, C)
    return r1 + r2.sum(2), r2.sum(1), r1.sum(1), r2.sum((1, 2))


def mstep_terms(Wm, x, r, stats):
    """sum_c resp(q, c) log pi(c | q) per question, from resp_stats: log pi of a two-hop chain is
    logsigmoid(-g x) + log softmax_1(c1) + log softmax_2(c2), so the sum needs only the marginals (exactly)."""
    import torch
    W1, W2, g = Wm
    m1, m2, a1, a2 = stats
    gl = x @ g
    lp1 = torch.log_softmax((x @ W1.T) @ r.T, 1)
    lp2 = torch.log_softmax((x @ W2.T) @ r.T, 1)
    F = torch.nn.functional
    return (m1 * lp1).sum(1) + (m2 * lp2).sum(1) + a1 * F.logsigmoid(gl) + a2 * F.logsigmoid(-gl)


def em_map(groups, logf=log):
    """groups: per dataset (x (Q, DIM+1), r (C, DIM), ev1 (Q, C), ev2 (Q, C*C)), each chain's evidence (the corpus's
    floor 1/n where uncounted). E-step: responsibilities pi(c|q) e(q,c) normalised; M-step: the map's weighted
    log-likelihood with an L2 penalty (L-BFGS). Every dataset weighs the same, and every question within it."""
    import torch
    W1 = torch.zeros(DIM, DIM + 1, dtype=torch.float64, requires_grad=True)
    W2 = torch.zeros(DIM, DIM + 1, dtype=torch.float64, requires_grad=True)
    g = torch.zeros(DIM + 1, dtype=torch.float64, requires_grad=True)
    Wm = (W1, W2, g)
    gs = [(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(r, dtype=torch.float64),
           torch.log(torch.as_tensor(e1, dtype=torch.float64)), torch.log(torch.as_tensor(e2, dtype=torch.float64)))
          for x, r, e1, e2 in groups]

    def penalized():
        tot = 0.0
        for x, r, le1, le2 in gs:
            o, t = map_logprobs(Wm, x, r)
            tot = tot + torch.logsumexp(torch.cat([o + le1, t + le2], 1), 1).mean()
        return tot / len(gs) - L2 * sum((w ** 2).sum() for w in Wm)

    trace = []
    with torch.no_grad():
        trace.append(float(penalized()))
    for _ in range(EM_MAX):
        with torch.no_grad():
            stats = []
            for x, r, le1, le2 in gs:
                o, t = map_logprobs(Wm, x, r)
                stats.append(resp_stats(torch.softmax(torch.cat([o + le1, t + le2], 1), 1), r.shape[0]))
        opt = torch.optim.LBFGS(list(Wm), lr=1.0, max_iter=100, tolerance_grad=1e-10, tolerance_change=1e-12,
                                history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            obj = 0.0
            for (x, r, _a, _b), st in zip(gs, stats):
                obj = obj - mstep_terms(Wm, x, r, st).mean()
            obj = obj / len(gs) + L2 * sum((w ** 2).sum() for w in Wm)
            obj.backward()
            return obj

        opt.step(closure)
        with torch.no_grad():
            trace.append(float(penalized()))
        if abs(trace[-1] - trace[-2]) <= EM_TOL * abs(trace[-2]):
            break
    logf(f"  EM: {len(trace) - 1} iterations")
    return tuple(w.detach().float() for w in Wm), trace


def chains_for(Wm, x, R):
    """The map's chains for one question: the TOP1 most likely one-hop chains, and the TOP2 most likely two-hop chains
    among the TOPK12 best first and TOPK12 best second hops; ties by chain id. Returns [(c1, c2 or -1)], pi, the gate."""
    import torch
    W1, W2, g = Wm
    C = R.shape[0]
    h = float(torch.sigmoid(x @ g))
    p1 = torch.softmax((W1 @ x) @ R.T, 0).double().cpu().numpy()
    p2 = torch.softmax((W2 @ x) @ R.T, 0).double().cpu().numpy()
    ids = np.arange(C)
    one = h * p1
    o1 = np.lexsort((ids, -one))[:TOP1]
    f = np.lexsort((ids, -p1))[:TOPK12]
    s = np.lexsort((ids, -p2))[:TOPK12]
    pid = (f[:, None] * C + s[None, :]).ravel()
    pi2 = ((1.0 - h) * p1[f][:, None] * p2[s][None, :]).ravel()
    o2 = np.lexsort((pid, -pi2))[:TOP2]
    chains = [(int(c), -1) for c in o1] + [(int(pid[i] // C), int(pid[i] % C)) for i in o2]
    return chains, np.r_[one[o1], pi2[o2]], h


# ── distances on the card ────────────────────────────────────────────────────


def masked_knn(U, UN, targets, k, mask, chunk=CHUNK):
    """Each target's k nearest nodes outside `mask` (a sorted int64 array): ids (L, k) and squared distances, nearest
    first, ties by node id. Slots past the nodes outside the mask hold an infinite distance."""
    import torch
    n = U.shape[0]
    tn = (targets ** 2).sum(1)
    mk = torch.from_numpy(np.ascontiguousarray(mask, np.int64)).to(U.device) if mask.size else None
    keep = k + 16
    best_d = best_i = None
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        d = tn[:, None] + UN[None, lo:hi] - 2.0 * (targets @ U[lo:hi].T)
        if mk is not None:
            a, b = np.searchsorted(mask, [lo, hi])
            if b > a:
                d[:, mk[a:b] - lo] = float("inf")
        dv, di = torch.topk(d, min(keep, hi - lo), dim=1, largest=False, sorted=False)
        di = di + lo
        if best_d is not None:
            dv, di = torch.cat([best_d, dv], 1), torch.cat([best_i, di], 1)
        o = torch.argsort(di, dim=1, stable=True)
        dv, di = dv.gather(1, o), di.gather(1, o)
        o = torch.argsort(dv, dim=1, stable=True)[:, :keep]
        best_d, best_i = dv.gather(1, o), di.gather(1, o)
    return best_i[:, :k], best_d[:, :k]


def fuse_entries(node, score, d2, nmax):
    """Each node's best entry (the largest score, then the smallest squared distance), the nodes ordered by it, ties
    by node id; the first nmax."""
    if not node.size:
        return np.empty(0, np.int64)
    o = np.lexsort((d2, -score, node))
    node, score, d2 = node[o], score[o], d2[o]
    first = np.r_[True, node[1:] != node[:-1]]
    node, score, d2 = node[first], score[first], d2[first]
    o = np.lexsort((node, d2, -score))
    return node[o][:nmax].astype(np.int64)


def depth_of(w, nmax=NMAX):
    """How deep each list (weight w) can still reach the fused top nmax: the top list alone gives nmax nodes of score
    at least max w / nmax, so an entry at position p counts only while w / p >= max w / nmax."""
    w = np.asarray(w, np.float64)
    return np.minimum(nmax, np.floor(nmax * w / w.max() * (1.0 + 1e-9)).astype(np.int64) + 1)


def fused_miss(U, UN, targets, w, mask):
    """The fused order of the nodes outside mask: each target's list by distance, fused by the largest w / position,
    ties by squared distance, then node id; NMAX nodes."""
    w = np.asarray(w, np.float64)
    depth = depth_of(w)
    ids, d2 = masked_knn(U, UN, targets, int(depth.max()), mask)
    ids, d2 = ids.cpu().numpy(), d2.double().cpu().numpy()
    pos = np.broadcast_to(np.arange(1, ids.shape[1] + 1)[None, :], ids.shape)
    keep = (pos <= depth[:, None]) & np.isfinite(d2)
    return fuse_entries(ids[keep], (w[:, None] / pos)[keep], d2[keep], NMAX)


def fused_low(U, targets, w, cand, nmax):
    """The fused order of the candidates (sorted int64): each target ranks every candidate by distance (ties by node
    id), fused by the largest w / position, ties by squared distance, then node id."""
    import torch
    if not cand.size:
        return np.empty(0, np.int64)
    uc = U[torch.from_numpy(cand).to(U.device)]
    d = ((targets ** 2).sum(1)[:, None] + (uc ** 2).sum(1)[None, :] - 2.0 * targets @ uc.T).double().cpu().numpy()
    L, m = d.shape
    o = np.argsort(d, axis=1, kind="stable")
    pos = np.empty_like(o)
    np.put_along_axis(pos, o, np.broadcast_to(np.arange(1, m + 1)[None, :], (L, m)), axis=1)
    score = np.asarray(w, np.float64)[:, None] / pos
    s = score.max(0)
    db = np.where(score == s[None, :], d, np.inf).min(0)
    return cand[np.lexsort((cand, db, -s))][:nmax].astype(np.int64)


def nearest_chains(us, ug, R, keep, first_cap=VO_FIRST):
    """The `keep` chains whose targets come nearest a gold (least over seeds s and golds g of |u_s + r - u_g|^2), as
    (chain ids, squared distances). One-hop chains are ids 0..C-1; a two-hop chain (c1, c2) is C + c1 C + c2, its
    first hops the first_cap one-hop chains nearest a gold (all of them when C <= first_cap). Ties by chain id."""
    import torch
    C = R.shape[0]
    gn = (ug ** 2).sum(1)

    def mind(t):
        flat = t.reshape(-1, t.shape[-1])
        d = (flat ** 2).sum(1)[:, None] + gn[None, :] - 2.0 * flat @ ug.T
        return d.min(1).values.reshape(t.shape[:-1])

    d1 = mind(us[:, None, :] + R[None]).min(0).values.double().cpu().numpy()
    first = np.lexsort((np.arange(C), d1))[:min(first_cap, C)]
    ids, ds = [np.arange(C)], [d1]
    for a in range(0, first.size, 8):
        f = first[a:a + 8]
        t = us[:, None, None, :] + R[torch.from_numpy(f).to(R.device)][None, :, None, :] + R[None, None, :, :]
        ids.append((C + f[:, None] * C + np.arange(C)[None, :]).ravel())
        ds.append(mind(t).min(0).values.double().cpu().numpy().ravel())
    ids, ds = np.concatenate(ids), np.concatenate(ds)
    o = np.lexsort((ids, ds))[:keep]
    return ids[o], ds[o]


def chain_evidence(U, UN, seeds, golds, R, chains, chunk=CHUNK):
    """e(q, c) per chain: the mean over the golds of the best over the seeds of 1/rank, the rank of the gold among all
    nodes but the seeds by squared distance to u_s + r_c (1 = nearest). A node counts as nearer only when its squared
    distance is below the gold's by more than TOL (|t|^2 + |u_g|^2), a float32 rounding margin."""
    import torch
    S, G, K = len(seeds), len(golds), len(chains)
    si = torch.from_numpy(np.asarray(seeds, np.int64)).to(U.device)
    gi = torch.from_numpy(np.asarray(golds, np.int64)).to(U.device)
    t = (U[si][None, :, :] + chain_steps(R, chains)[:, None, :]).reshape(K * S, -1)
    tn = (t ** 2).sum(1)
    gd = ((t[:, None, :] - U[gi][None]) ** 2).sum(2)
    thr = gd - TOL * (tn[:, None] + (U[gi] ** 2).sum(1)[None, :])
    closer = torch.zeros(K * S, G, dtype=torch.long, device=U.device)
    for lo in range(0, U.shape[0], chunk):
        hi = min(U.shape[0], lo + chunk)
        d = tn[:, None] + UN[None, lo:hi] - 2.0 * (t @ U[lo:hi].T)
        for g in range(G):
            closer[:, g] += (d < thr[:, g:g + 1]).sum(1)
    ds = tn[:, None] + UN[si][None, :] - 2.0 * (t @ U[si].T)
    for g in range(G):
        closer[:, g] -= (ds < thr[:, g:g + 1]).sum(1)
    rank = (closer + 1).clamp(min=1).reshape(K, S, G)
    return (1.0 / rank.double()).max(1).values.mean(1).cpu().numpy()


# ── the graph's hops ─────────────────────────────────────────────────────────

_RINGS = []


def ring_kernel():
    """rings(ip0, c0, ip1, c1, ip2, c2, seeds, mark, out) -> (n1, n2): ring 1 (the seeds' neighbours over the three
    families, seeds excluded) in out[:n1] and ring 2 in out[n1:n1 + n2], in the order found; mark (int8, zero) is
    returned zero."""
    if not _RINGS:
        from numba import njit

        @njit(cache=False)
        def visit(ip, col, v, mark, out, k):
            for e in range(ip[v], ip[v + 1]):
                w = col[e]
                if mark[w] == 0:
                    mark[w] = 1
                    out[k] = w
                    k += 1
            return k

        @njit(cache=False)
        def rings(ip0, c0, ip1, c1, ip2, c2, seeds, mark, out):
            for s in seeds:
                mark[s] = 1
            k = 0
            for s in seeds:
                k = visit(ip0, c0, s, mark, out, k)
                k = visit(ip1, c1, s, mark, out, k)
                k = visit(ip2, c2, s, mark, out, k)
            n1 = k
            for i in range(n1):
                v = out[i]
                k = visit(ip0, c0, v, mark, out, k)
                k = visit(ip1, c1, v, mark, out, k)
                k = visit(ip2, c2, v, mark, out, k)
            for s in seeds:
                mark[s] = 0
            for i in range(k):
                mark[out[i]] = 0
            return n1, k - n1

        _RINGS.append(rings)
    return _RINGS[0]


def rings_of(adj, seeds, mark, out):
    (ip0, c0), (ip1, c1), (ip2, c2) = adj
    n1, n2 = ring_kernel()(ip0, c0, ip1, c1, ip2, c2, np.asarray(seeds, np.int64), mark, out)
    return np.sort(out[:n1]).astype(np.int32), np.sort(out[n1:n1 + n2]).astype(np.int32)


def first_unique(a):
    _, i = np.unique(a, return_index=True)
    return a[np.sort(i)]


def h2_orders(r1, r2, need, cos, pool, seeds, dense, nlow):
    """H2. miss: ring 1's then ring 2's nodes outside the pool, each by the frozen cosine (ties by node id), then the
    dense list's remaining nodes; NMAX nodes. low: the pool's nodes but the seeds, ring 1's, then ring 2's, then the
    rest, each by cosine; nlow nodes."""
    c1, c2 = cos[np.searchsorted(need, r1)], cos[np.searchsorted(need, r2)]
    m1, m2 = ~np.isin(r1, pool), ~np.isin(r2, pool)
    a, ca, b, cb = r1[m1], c1[m1], r2[m2], c2[m2]
    miss = [a[np.lexsort((a, -ca))], b[np.lexsort((b, -cb))]]
    if a.size + b.size < NMAX:
        dl = first_unique(np.asarray(dense, np.int64))
        miss.append(dl[~np.isin(dl, np.concatenate([pool, r1, r2]))])
    miss = np.concatenate(miss)[:NMAX]
    cand = np.setdiff1d(pool, seeds)
    cc = cos[np.searchsorted(need, cand)]
    ring = np.full(cand.size, 3, np.int64)
    ring[np.isin(cand, r2)] = 2
    ring[np.isin(cand, r1)] = 1
    low = cand[np.lexsort((cand, -cc, ring))][:nlow]
    return miss.astype(np.int64), low.astype(np.int64)


def cosines(emb, qv, need, device):
    """For each question b, the frozen cosine of each node of need[b] (sorted) to its question: one pass over the
    corpus's frozen vectors against the question vectors, both scaled to unit length, on the card."""
    import torch
    B = len(need)
    lens = np.asarray([x.size for x in need], np.int64)
    off = np.r_[0, np.cumsum(lens)]
    flat = np.concatenate(need).astype(np.int32) if B else np.empty(0, np.int32)
    vals = np.full(flat.size, np.nan, np.float32)
    edges = np.minimum(np.arange(emb.n_shards + 1, dtype=np.int64) * emb.shard_size, emb.n_rows)
    cut = np.stack([np.searchsorted(x, edges) for x in need]) if B else np.zeros((0, edges.size), np.int64)
    q = np.asarray(qv, np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    Q = torch.from_numpy(q).to(device)
    for k in range(emb.n_shards):
        st, en = cut[:, k], cut[:, k + 1]
        ln = en - st
        tot = int(ln.sum())
        if not tot:
            continue
        idx = np.repeat(off[:-1] + st - (np.cumsum(ln) - ln), ln) + np.arange(tot, dtype=np.int64)
        rows = torch.from_numpy(flat[idx].astype(np.int64) - int(edges[k])).to(device)
        cols = torch.from_numpy(np.repeat(np.arange(B, dtype=np.int64), ln)).to(device)
        xs = torch.from_numpy(np.array(emb.shard(k), dtype=np.float16)).to(device).float()
        xs = xs / xs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        vals[idx] = (xs @ Q.T)[rows, cols].cpu().numpy()
        del xs
    if np.isnan(vals).any():
        raise SystemExit("a needed node's cosine was not read")
    return [vals[off[b]:off[b + 1]] for b in range(B)]


def found_counts(order, T, grid):
    """|order[:N] ∩ T| for each N of the grid."""
    c = np.cumsum(np.isin(order, T)) if order.size else np.zeros(0, np.int64)
    return [int(c[min(N, c.size) - 1]) if c.size else 0 for N in grid]


# ── the data ─────────────────────────────────────────────────────────────────


def modules():
    import p1a_pool_scoreboard as P
    return P.E, P.F, P.E.S4


def opened(host):
    E, F, S4 = modules()
    if host:
        import importlib.util
        spec = importlib.util.spec_from_file_location("s4f_run_one", ROOT / "outputs" / "step4f" / "run_one.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.on_the_mirror(F)
    return S4.Opened()


def dense_docs(op, name):
    ds = op.canonical.Dataset(name, root=str(op.served))
    return ds, ds.embeddings("dense", "docs")


def load_frame():
    if not FRAME.exists():
        return None
    with np.load(FRAME) as z:
        return z["mu"], z["P"]


def dev_frame_u0(emb, logf=log):
    """A development run's frame and start, in memory, from this corpus alone (never written)."""
    n = emb.n_rows
    pick = np.sort(np.random.default_rng(0).permutation(n)[:min(n, 5000)])
    mu, P, _ = frame_of(emb.read(pick))
    u0 = np.empty((n, DIM), np.float16)
    for k in range(emb.n_shards):
        lo = k * emb.shard_size
        u0[lo:min(n, lo + emb.shard_size)] = project(np.asarray(emb.shard(k)), mu, P).astype(np.float16)
    logf("  development frame and start built in memory from this corpus")
    return mu, P, u0


def offsets_record(fams, offs, offs0):
    out = {}
    for f, o, o0 in zip(fams, offs, offs0):
        out[f"V|{f.name}"] = o.numpy()
        out[f"V0|{f.name}"] = o0.numpy()
    return out


def load_space(name):
    p = SPACE / f"space_{name}.npz"
    with np.load(p) as z:
        offs = {sp: [torch_tensor(z[f"{sp}|{f}"]) for f in FAMS] for sp in ("V", "V0")}
    return offs


def torch_tensor(a):
    import torch
    return torch.from_numpy(np.asarray(a, np.float32))


# ── base ─────────────────────────────────────────────────────────────────────


def base_stage(host, limit=None):
    op = opened(host)
    t0 = time.time()
    rows, parts = [], {}
    embs = {}
    for name in SIX:
        _ds, emb = dense_docs(op, name)
        embs[name] = emb
        n = emb.n_rows
        m = min(n, limit or FRAME_N)
        pick = np.sort(np.random.default_rng(0).permutation(n)[:m])
        rows.append(emb.read(pick))
        parts[name] = {"rows": int(m), "corpus": int(n),
                       "pick_sha256": hashlib.sha256(pick.astype(np.int64).tobytes()).hexdigest()}
        log(f"{name}: {m} of {n} frozen vectors drawn")
    mu, P, ev = frame_of(np.concatenate(rows))
    del rows
    orth = float(np.abs(P.astype(np.float64) @ P.T.astype(np.float64) - np.eye(DIM)).max())
    if not (np.isfinite(mu).all() and np.isfinite(P).all()) or orth > 1e-4:
        raise SystemExit(f"the frame is not orthonormal ({orth})")
    if limit:
        for name in SIX:
            x = project(np.asarray(embs[name].shard(0)), mu, P)
            if x.shape[1] != DIM or not np.isfinite(x).all():
                raise SystemExit(f"{name}: the start's first shard is not finite")
        log(f"development run: the frame from {sum(p['rows'] for p in parts.values())} vectors is orthonormal; every "
            f"corpus's first shard projects ({time.time() - t0:.0f}s); nothing written")
        return 0
    save_npz(FRAME, mu=mu, P=P, eigenvalues=ev)
    rec = {"declared_in": "docs/P5A_NODE_SPACE.md", "stage": "base", "dim": DIM, "frame_n": FRAME_N, "parts": parts,
           "orthonormal_max_error": orth, "frame_sha256": sha_file(FRAME), "u0": {}}
    for name in SIX:
        emb = embs[name]
        n = emb.n_rows
        u0 = np.empty((n, DIM), np.float16)
        for k in range(emb.n_shards):
            lo = k * emb.shard_size
            u0[lo:min(n, lo + emb.shard_size)] = project(np.asarray(emb.shard(k)), mu, P).astype(np.float16)
        if not np.isfinite(u0).all():
            raise SystemExit(f"{name}: the start is not finite")
        p = SPACE / f"u0_{name}.npy"
        save_npy(p, u0)
        rec["u0"][name] = {"rows": int(n), "sha256": sha_file(p), "bytes": p.stat().st_size}
        log(f"{name}: start written ({time.time() - t0:.0f}s)")
        del u0
    rec.update({"script_sha256": script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
                "seconds": round(time.time() - t0, 1), "peak_rss_bytes": peak_rss(), "utc": utc()})
    write_json(BASE, rec)
    log(f"base: {BASE.relative_to(ROOT)}")
    return 0


# ── train ────────────────────────────────────────────────────────────────────


def train_stage(name, device, threads, host, limit=None, steps=None):
    import torch
    torch.set_num_threads(threads)
    full_fp32(torch)
    E, _F, S4 = modules()
    op = opened(host)
    dev = torch.device(device)
    t0 = time.time()
    _ds, emb = dense_docs(op, name)
    fr = load_frame()
    if fr is not None and (SPACE / f"u0_{name}.npy").exists():
        mu, P = fr
        u0 = np.load(SPACE / f"u0_{name}.npy")
        made = "base"
    elif limit:
        mu, P, u0 = dev_frame_u0(emb)
        made = "development"
    else:
        raise SystemExit("the base stage's frame and start are not on disk")
    if u0.shape != (emb.n_rows, DIM):
        raise SystemExit(f"{name}: the start has {u0.shape}, the corpus {emb.n_rows} rows")
    fams = edge_families(name, csr_dir(op))
    if any(int(f.h.max(initial=-1)) >= u0.shape[0] or int(f.t.max(initial=-1)) >= u0.shape[0] for f in fams):
        raise SystemExit(f"{name}: an edge names a node outside the corpus")
    log(f"{name}: {u0.shape[0]} nodes, edges " + ", ".join(f"{f.name} {f.h.size}" for f in fams)
        + f" ({made} start, {device})")
    n_steps = steps if (limit and steps) else STEPS
    u, offs, info = fit_space(u0, fams, dev, steps=n_steps)
    u16 = u.half().cpu().numpy()
    del u
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    _u0, offs0, info0 = fit_space(u0, fams, dev, steps=n_steps, fit_nodes=False)
    del _u0
    for f in fams:                                    # the edges are on the card no more; keep names and flags
        f.h = f.t = np.empty(0, np.int64)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    finite = (all(np.isfinite(v) for _s, v in info["loss_hist"] + info0["loss_hist"])
              and bool(np.isfinite(u16).all()) and all(bool(torch.isfinite(o).all()) for o in offs + offs0))
    if not finite:
        raise SystemExit(f"{name}: a fit is not finite")
    log(f"{name}: space fit ({info['seconds']:.0f}s) and start's offsets ({info0['seconds']:.0f}s); finite")
    spaces = {"V": (torch.from_numpy(u16).float().to(dev), chain_table(offs).to(dev)),
              "V0": (torch.from_numpy(np.asarray(u0, np.float16)).float().to(dev), chain_table(offs0).to(dev))}
    norms = {sp: (Us ** 2).sum(1) for sp, (Us, _R) in spaces.items()}
    ev_rec = None
    evid = None
    if name in TRAIN5:
        carve = fit_carve(name)
        holder = {}

        def sink(name_, carve_, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
            Q = len(pop.ids)
            samp = sample_of(Q)
            if limit:
                samp = samp[:limit]
            X, keep_ids, ng = [], [], []
            ch = {sp: [] for sp in spaces}
            ev = {sp: [] for sp in spaces}
            skipped = 0
            t = time.time()
            for c, j in enumerate(samp):
                seeds = np.asarray(prep.seeds[j], np.int64)
                golds = np.setdiff1d(np.unique(np.asarray(pop.golds[j], np.int64)), seeds)
                if not golds.size:
                    skipped += 1
                    continue
                X.append(qframe(prep.qemb[j], mu, P))
                keep_ids.append(pop.ids[j])
                ng.append(golds.size)
                for sp, (Us, Rs) in spaces.items():
                    UN = norms[sp]
                    si = torch.from_numpy(seeds).to(dev)
                    gi = torch.from_numpy(golds).to(dev)
                    cid, _d = nearest_chains(Us[si], Us[gi], Rs, PRUNE)
                    C = Rs.shape[0]
                    e = chain_evidence(Us, UN, seeds, golds, Rs, [decode(x, C) for x in cid])
                    ch[sp].append(cid)
                    ev[sp].append(e)
                if (c + 1) % 200 == 0:
                    log(f"{name}/{carve_}: evidence {c + 1}/{samp.size} ({time.time() - t:.0f}s)")
            holder["out"] = {"x": np.asarray(X, np.float32), "ids": np.asarray(keep_ids),
                             "n_gold": np.asarray(ng, np.int64),
                             **{f"chains_{sp}": np.asarray(ch[sp], np.int64).reshape(-1, PRUNE) for sp in spaces},
                             **{f"e_{sp}": np.asarray(ev[sp], np.float64).reshape(-1, PRUNE) for sp in spaces}}
            holder["rec"] = {"carve": carve_, "carve_questions": Q, "sampled": int(samp.size),
                             "without_a_non_seed_gold": skipped, "kept": len(keep_ids),
                             "carve_sha256": pop.digest, "sample_ids_sha256": ids_sha([pop.ids[j] for j in samp]),
                             "seconds": round(time.time() - t, 1)}

        E.run_carves(op, name, {carve}, (), threads, sink)
        evid, ev_rec = holder["out"], holder["rec"]
        if not all(np.isfinite(evid[f"e_{sp}"]).all() for sp in spaces):
            raise SystemExit(f"{name}: an evidence is not finite")
    if limit:
        if evid is not None:
            C = spaces["V"][1].shape[0]
            groups = [(evid["x"], spaces["V"][1].cpu().numpy(), *dense_evidence(evid["chains_V"], evid["e_V"], C,
                                                                                    1.0 / u0.shape[0]))]
            em_map(groups, logf=log)
        log(f"{name} development run: both fits ran and are finite; evidence "
            + (f"on {ev_rec['kept']} of {ev_rec['sampled']} questions" if ev_rec else "not read (webqsp never fits)")
            + f"; {time.time() - t0:.0f}s; nothing written")
        return 0
    up = SPACE / f"u_{name}.npy"
    save_npy(up, u16)
    sp_path = SPACE / f"space_{name}.npz"
    save_npz(sp_path, **offsets_record(fams, offs, offs0),
             named=np.asarray([f.named for f in fams]), signed=np.asarray([f.signed for f in fams]),
             n_rel=np.asarray([f.n_rel for f in fams], np.int64))
    rec = {"declared_in": "docs/P5A_NODE_SPACE.md", "stage": "train", "dataset": name, "nodes": int(u0.shape[0]),
           "fit": info, "start_fit": info0, "steps": n_steps, "batch": BATCH, "neg": NEG, "lr": LR, "lambda": LAMBDA,
           "u_sha256": sha_file(up), "space_sha256": sha_file(sp_path),
           "frame_sha256": sha_file(FRAME), "u0_sha256": sha_file(SPACE / f"u0_{name}.npy")}
    if evid is not None:
        ep = OUT / f"evidence_{name}.npz"
        save_npz(ep, **evid)
        rec["evidence"] = {**ev_rec, "npz_sha256": sha_file(ep)}
    rec.update({"script_sha256": script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
                "device": str(dev), "gpu_peak_bytes": gpu_peak(torch, dev), "peak_rss_bytes": peak_rss(),
                "seconds": round(time.time() - t0, 1), "utc": utc()})
    write_json(SPACE / f"space_{name}.json", rec)
    log(f"{name}: outputs/p5a/space/space_{name}.json ({rec['seconds']:.0f}s)")
    return 0


def dense_evidence(chains, e, C, floor):
    """The evidence as (Q, C) and (Q, C * C): the counted chains' values, the corpus's floor 1/n elsewhere."""
    Q = chains.shape[0]
    ev1 = np.full((Q, C), floor)
    ev2 = np.full((Q, C * C), floor)
    for q in range(Q):
        for c, v in zip(chains[q], e[q]):
            if c < C:
                ev1[q, c] = max(v, floor)
            else:
                ev2[q, c - C] = max(v, floor)
    return ev1, ev2


# ── the map ──────────────────────────────────────────────────────────────────


def map_stage(threads=2):
    import torch
    torch.set_num_threads(threads)
    t0 = time.time()
    out, rec = {}, {"declared_in": "docs/P5A_NODE_SPACE.md", "stage": "map", "datasets": list(TRAIN5), "spaces": {}}
    srecs = {}
    for name in TRAIN5:
        srecs[name] = json.loads((SPACE / f"space_{name}.json").read_text(encoding="utf-8"))
        if sha_file(OUT / f"evidence_{name}.npz") != srecs[name]["evidence"]["npz_sha256"]:
            raise SystemExit(f"{name}: the evidence is not its record's")
        if sha_file(SPACE / f"space_{name}.npz") != srecs[name]["space_sha256"]:
            raise SystemExit(f"{name}: the space is not its record's")
    if len({r["script_sha256"] for r in srecs.values()}) != 1 or len({r["frame_sha256"] for r in srecs.values()}) != 1:
        raise SystemExit("the spaces ran on different scripts or frames")
    offs = {name: load_space(name) for name in TRAIN5}
    for sp in ("V", "V0"):
        groups, sizes = [], {}
        for name in TRAIN5:
            with np.load(OUT / f"evidence_{name}.npz") as z:
                x, ch, e = z["x"], z[f"chains_{sp}"], z[f"e_{sp}"]
            r = chain_table(offs[name][sp]).numpy()
            C = r.shape[0]
            if C > 128:
                raise SystemExit(f"{name}: {C} chains is too many to fit the map over every two-hop chain")
            groups.append((x, r, *dense_evidence(ch, e, C, 1.0 / srecs[name]["nodes"])))
            sizes[name] = {"questions": int(x.shape[0]), "chains": C}
        Wm, trace = em_map(groups)
        if not all(np.isfinite(t) for t in trace) or not all(bool(torch.isfinite(w).all()) for w in Wm):
            raise SystemExit(f"{sp}: the map is not finite")
        out.update({f"{sp}|W1": Wm[0].numpy(), f"{sp}|W2": Wm[1].numpy(), f"{sp}|g": Wm[2].numpy()})
        gate = {}
        for name, (x, r, _e1, _e2) in zip(TRAIN5, groups):
            gate[name] = round(float(torch.sigmoid(torch.as_tensor(x) @ Wm[2]).mean()), 4)
        rec["spaces"][sp] = {"trace": [round(t, 8) for t in trace], "iterations": len(trace) - 1, "data": sizes,
                             "gate_mean_fit": gate}
    save_npz(MAP, **out)
    rec.update({"map_sha256": sha_file(MAP), "space_records": {n: sha_file(SPACE / f"space_{n}.json") for n in TRAIN5},
                "script_sha256": script_sha256(), "seconds": round(time.time() - t0, 1), "utc": utc()})
    write_json(MAP_JSON, rec)
    log(f"map: {MAP_JSON.relative_to(ROOT)}")
    return 0


def load_map():
    import torch
    with np.load(MAP) as z:
        return {sp: tuple(torch.from_numpy(z[f"{sp}|{k}"]) for k in ("W1", "W2", "g")) for sp in ("V", "V0")}


# ── read1: zsp's top 5 through P2a's reader ──────────────────────────────────


def zsp_read(R2, rd, i):
    """P2a's Reader.question, its unmasked arm only: the pool, the seeds and both fits' scores (zrc first, as there)."""
    su = rd.su
    p = su.prep
    pool, seeds, bucket = rd.fpool(np.asarray(p.dense_ids[i]), np.asarray(p.splade_ids[i]))
    qemb = np.asarray(p.qemb[i])
    comp, typed = rd.compile(i, pool, seeds, ())
    sh = {"pool": pool, "seeds": seeds, "bucket": bucket, "comp": comp, "qemb": qemb}
    sc = {}
    for k in R2.MODELS:
        r = rd.C10.lean_rows10(su, sh, {}, rd.dec[k])
        if rd.typed:
            e_rel = rd.C5.e_rel_of(r, typed)
            r["chains"] = rd.C5.one_question_chains(su, rd.C6.chains_of6(su, k, r, e_rel, qemb))
            r["e_rel"] = e_rel
        if k == "zsp":
            eu, ev, ef = r["e"][:3]
            r["links"] = rd.C3.links_fast(r["n"], eu, ev, ef)
        sc[k] = np.array(rd.zf[k](r).numpy(), np.float32, copy=True)
    return np.asarray(pool, np.int64), np.asarray(seeds, np.int64), sc


def read1_stage(name, part, host, limit=None):
    import p2a_rank as R2
    t0 = time.time()
    where = R2.on_the_mirror() if host else "laptop"
    C11 = R2.frozen_modules()
    C1 = C11.C1
    import torch
    basis = torch.load(C1.FITS["zrc"] / "models.pt", weights_only=False, map_location="cpu")["basis"]
    cache = R2.Cache(C1.LC, name, basis)
    look_ids = json.loads((C1.LM.LOOK / name / C1.CARVE / "ids.json").read_text(encoding="utf-8"))
    if cache.ids != look_ids:
        raise SystemExit(f"{name}: step 1's ids are not the look's carve order")
    Q = len(cache.ids)
    k, n = part
    if n != READ1_PARTS[name]:
        raise SystemExit(f"{name}: P2a read it in {READ1_PARTS[name]} parts")
    a, b = R2.part_range(Q, k, n)
    samp = sample_of(Q)
    sel = samp[(samp >= a) & (samp < b)]
    if limit:
        sel = sel[:limit]
    p2a_npz = P2A / f"rank_{name}_{k}of{n}.npz"
    with np.load(p2a_npz) as z:
        M, p2a_ids = z["M"], z["ids"]
    if p2a_ids.tolist() != cache.ids[a:b]:
        raise SystemExit(f"{p2a_npz}: its ids are not the part's")
    pools, seeds_l, top5, r5, r5p = [], [], [], [], []
    if sel.size:
        nq = int(sel[-1]) - a + 1
        rd = R2.Reader(name, a, nq)
        su = rd.su
        if su.basis != basis or list(su.ids) != cache.ids[a:a + nq]:
            raise SystemExit(f"{name}: the setup's basis or questions are not the cache's")
        log(f"{name} part {k}/{n}: {sel.size} sampled questions in {a} to {b}; setup {time.time() - t0:.0f}s ({where})")
        t = time.time()
        for c, g in enumerate(sel):
            i = int(g) - a
            pool, seeds, sc = zsp_read(R2, rd, i)
            if not np.array_equal(pool, np.asarray(cache.rows(int(g), "node"), np.int64)):
                raise SystemExit(f"{name}: question {g} ({cache.ids[g]}): the pool is not step 1's")
            s = sc["zsp"]
            pools.append(pool)
            seeds_l.append(seeds)
            top5.append(pool[R2.top5(s)])
            r5.append(float(rd.measures(s, np.asarray(cache.rows(int(g), "gold")), cache.gold_total(int(g)))[0]))
            r5p.append(float(M[0, 1, i, 0]))
            if (c + 1) % 100 == 0:
                log(f"{name} part {k}/{n}: {c + 1}/{sel.size} ({time.time() - t:.0f}s)")
    r5, r5p = np.asarray(r5, np.float64), np.asarray(r5p, np.float64)
    differ = int((np.abs(r5 - r5p) > 1e-6).sum())
    chk = {"questions": int(sel.size), "pool_equal_step1": int(sel.size), "zsp_r5_differs_from_p2a": differ}
    if limit:
        log(f"{name} part {k}/{n} development run: {sel.size} questions ({time.time() - t0:.0f}s); checks {chk}; "
            "nothing written")
        return 0
    stem = f"{name}_{k}of{n}"
    p = READ1 / f"{stem}.npz"
    save_npz(p, pos=np.asarray(sel, np.int64), ids=np.asarray([cache.ids[g] for g in sel]),
             pool_ptr=np.r_[0, np.cumsum([x.size for x in pools])].astype(np.int64),
             pool=np.concatenate(pools) if pools else np.empty(0, np.int64),
             seed_ptr=np.r_[0, np.cumsum([x.size for x in seeds_l])].astype(np.int64),
             seeds=np.concatenate(seeds_l) if seeds_l else np.empty(0, np.int64),
             top5=np.asarray(top5, np.int64).reshape(-1, 5), r5=r5, r5_p2a=r5p)
    rec = {"declared_in": "docs/P5A_NODE_SPACE.md", "stage": "read1", "dataset": name, "part": [k, n],
           "questions": [a, b], "carve_questions": Q, "checks": chk, "npz_sha256": sha_file(p),
           "sample_sha256": ids_sha([cache.ids[g] for g in sample_of(Q)]),
           "fits": {m: sha_file(C1.FITS[m] / "models.pt") for m in R2.MODELS}, "p2a_npz_sha256": sha_file(p2a_npz),
           "frozen_module_sha256": {f: sha_file(R2.MPU / f) for f in R2.FROZEN}, "script_sha256": script_sha256(),
           "where": where, "seconds": round(time.time() - t0, 1), "peak_rss_bytes": peak_rss(), "utc": utc()}
    write_json(READ1 / f"{stem}.json", rec)
    log(f"{name} part {k}/{n}: outputs/p5a/read1/{stem}.json")
    return 0


def read1_load(name):
    """zsp's top 5, pools and seeds on the whole sample, joined from the parts in carve order."""
    n = READ1_PARTS[name]
    out = {"pos": [], "ids": [], "pools": [], "seeds": [], "top5": [], "r5": [], "r5_p2a": [], "records": {}}
    shas = set()
    for k in range(n):
        rp = READ1 / f"{name}_{k}of{n}.json"
        if not rp.exists():
            return None
        rec = json.loads(rp.read_text(encoding="utf-8"))
        p = READ1 / f"{name}_{k}of{n}.npz"
        if sha_file(p) != rec["npz_sha256"]:
            raise SystemExit(f"{p}: not its record's file")
        out["records"][rp.name] = sha_file(rp)
        shas.add((rec["script_sha256"], rec["sample_sha256"], json.dumps(rec["fits"], sort_keys=True)))
        with np.load(p) as z:
            out["pos"] += z["pos"].tolist()
            out["ids"] += z["ids"].tolist()
            pp, sp = z["pool_ptr"], z["seed_ptr"]
            out["pools"] += [z["pool"][pp[i]:pp[i + 1]] for i in range(pp.size - 1)]
            out["seeds"] += [z["seeds"][sp[i]:sp[i + 1]] for i in range(sp.size - 1)]
            out["top5"] += list(z["top5"])
            out["r5"] += z["r5"].tolist()
            out["r5_p2a"] += z["r5_p2a"].tolist()
    if len(shas) != 1:
        raise SystemExit(f"{name}: read1's parts ran on different scripts, samples or fits")
    out["sample_sha256"] = next(iter(shas))[1]
    return out


# ── read ─────────────────────────────────────────────────────────────────────


def read_stage(name, device, threads, host, limit=None, steps=None):
    import torch
    torch.set_num_threads(threads)
    full_fp32(torch)
    E, _F, S4 = modules()
    op = opened(host)
    dev = torch.device(device)
    t0 = time.time()
    ds, emb = dense_docs(op, name)
    n = emb.n_rows
    fr = load_frame()
    have = fr is not None and (SPACE / f"u_{name}.npy").exists() and MAP.exists()
    if have:
        mu, P = fr
        u0 = np.load(SPACE / f"u0_{name}.npy")
        u = np.load(SPACE / f"u_{name}.npy")
        offs = load_space(name)
        maps = load_map()
        made = "filed"
    elif limit:
        mu, P, u0 = dev_frame_u0(emb)
        fams = edge_families(name, csr_dir(op))
        uu, o1, _i = fit_space(u0, fams, dev, steps=steps or 20, logf=lambda *_: None)
        u = uu.half().cpu().numpy()
        _x, o0, _i0 = fit_space(u0, fams, dev, steps=steps or 20, logf=lambda *_: None,
                                fit_nodes=False)
        offs = {"V": o1, "V0": o0}
        maps = {"V": zero_map(), "V0": zero_map()}
        made = "development (short space, zero map)"
        del uu, _x
    else:
        raise SystemExit(f"{name}: the frame, the space or the map is not on disk")
    r1 = read1_load(name)
    if r1 is None and not limit:
        raise SystemExit(f"{name}: read1's parts are not all on disk")
    stores, adj = family_stores(S4, ds, name, csr_dir(op))
    if any(ip.size - 1 != n for ip, _c in adj):
        raise SystemExit(f"{name}: a family's store is not the corpus's size")
    U = {"V": torch.from_numpy(u).float().to(dev), "V0": torch.from_numpy(u0).float().to(dev)}
    UN = {k: (v ** 2).sum(1) for k, v in U.items()}
    R = {sp: chain_table(offs[sp]).to(dev) for sp in ("V", "V0")}
    W = {sp: tuple(w.float().to(dev) for w in maps[sp]) for sp in ("V", "V0")}
    del u, u0
    log(f"{name}: {n} nodes, {R['V'].shape[0]} one-hop chains ({made}, {device}); setup {time.time() - t0:.0f}s")
    mark = np.zeros(n, np.int8)
    buf = np.empty(n, np.int64)
    nlow = GRID["low"][-1]
    holder = {}

    def sink(name_, carve, pop, prep, Uu, BQ, bs, inc, linked, union, constant, families, construction):
        Q = len(pop.ids)
        samp = sample_of(Q)
        sample_sha = ids_sha([pop.ids[j] for j in samp])
        if limit:
            samp = samp[:limit]
        ids = [pop.ids[j] for j in samp]
        if r1 is not None:
            if r1["sample_sha256"] != sample_sha or r1["ids"][:len(ids)] != ids or r1["pos"][:len(ids)] != samp.tolist():
                raise SystemExit(f"{name}: the sample is not read1's")
        nQ = samp.size
        res = {rd_: {"T": np.zeros(nQ, np.int64), "T_ring": np.zeros((nQ, RING_CLASSES), np.int64),
                     "found": np.zeros((nQ, len(ORDERS), len(GRID[rd_])), np.int32),
                     "found_ring": np.zeros((nQ, len(ORDERS), len(GRID[rd_]), RING_CLASSES), np.int32),
                     "v_not_h2": np.zeros((nQ, len(GRID[rd_])), np.int32),
                     "olen": np.zeros((nQ, len(ORDERS)), np.int32)} for rd_ in GRID}
        use = {f"{k}_{sp}": np.zeros(nQ, np.float64 if k == "gate" else np.int64)
               for sp in ("V", "V0") for k in ("gate", "top1", "top2")}
        vo_chain = {rd_: np.full(nQ, -1, np.int64) for rd_ in GRID}
        sizes = np.zeros((nQ, 3), np.int64)
        checked = 0
        t = time.time()
        for lo in range(0, nQ, QBATCH):
            js = samp[lo:lo + QBATCH]
            rings, needs = [], []
            for j in js:
                r_1, r_2 = rings_of(adj, prep.seeds[j], mark, buf)
                rings.append((r_1, r_2))
                needs.append(np.union1d(np.union1d(r_1, r_2), np.asarray(inc[j], np.int64)))
            cos = cosines(emb, np.stack([prep.qemb[j] for j in js]), needs, dev)
            for b, j in enumerate(js):
                qi = lo + b
                seeds = np.asarray(prep.seeds[j], np.int64)
                pool = np.unique(np.asarray(inc[j], np.int64))
                if r1 is not None:
                    if not (np.array_equal(pool, np.unique(r1["pools"][qi]))
                            and np.array_equal(np.unique(seeds), np.unique(r1["seeds"][qi]))):
                        raise SystemExit(f"{name}: {pop.ids[j]}: the pool or seeds are not P2a's reader's")
                    checked += 1
                    top = np.asarray(r1["top5"][qi], np.int64)
                else:
                    top = np.empty(0, np.int64)
                golds = np.unique(np.asarray(pop.golds[j], np.int64))
                inpool = np.isin(golds, pool)
                T = {"miss": golds[~inpool],
                     "low": golds[inpool & ~np.isin(golds, seeds) & ~np.isin(golds, top)]}
                r_1, r_2 = rings[b]
                sizes[qi] = (r_1.size, r_2.size, pool.size)
                cand = np.setdiff1d(pool, seeds)
                od = {rd_: {} for rd_ in GRID}
                od["miss"]["H2"], od["low"]["H2"] = h2_orders(r_1, r_2, needs[b], cos[b], pool, seeds,
                                                              prep.dense_ids[j], nlow)
                exp = np.asarray(S4.m3b_pools.expand_hops(seeds, stores, H1_SETTING), np.int64)
                od["miss"]["H1"] = exp[~np.isin(exp, pool)][:NMAX]
                od["low"]["H1"] = exp[np.isin(exp, cand)][:nlow]
                x = torch.from_numpy(qframe(prep.qemb[j], mu, P)).to(dev)
                si = torch.from_numpy(seeds).to(dev)
                for sp in ("V", "V0"):
                    chains, w, h = chains_for(W[sp], x, R[sp])
                    use[f"gate_{sp}"][qi] = h
                    C = R[sp].shape[0]
                    use[f"top1_{sp}"][qi] = chains[0][0]
                    c1, c2 = next(c for c in chains if c[1] >= 0)
                    use[f"top2_{sp}"][qi] = C + c1 * C + c2
                    tg = (U[sp][si][None, :, :] + chain_steps(R[sp], chains)[:, None, :]).reshape(-1, DIM)
                    wt = np.repeat(w, seeds.size)
                    od["miss"][sp] = fused_miss(U[sp], UN[sp], tg, wt, pool)
                    od["low"][sp] = fused_low(U[sp], tg, wt, cand, nlow)
                for rd_ in GRID:
                    if T[rd_].size:
                        cid, _d = nearest_chains(U["V"][si], U["V"][torch.from_numpy(T[rd_]).to(dev)], R["V"], 1)
                        vo_chain[rd_][qi] = cid[0]
                        tg = U["V"][si] + chain_steps(R["V"], [decode(cid[0], R["V"].shape[0])])
                        wt = np.ones(seeds.size)
                        od[rd_]["VO"] = (fused_miss(U["V"], UN["V"], tg, wt, pool) if rd_ == "miss"
                                         else fused_low(U["V"], tg, wt, cand, nlow))
                    else:
                        od[rd_]["VO"] = np.empty(0, np.int64)
                for rd_ in GRID:
                    Tr = T[rd_]
                    rg = np.full(Tr.size, 3, np.int64)
                    rg[np.isin(Tr, r_2)] = 2
                    rg[np.isin(Tr, r_1)] = 1
                    rr = res[rd_]
                    rr["T"][qi] = Tr.size
                    for c in range(RING_CLASSES):
                        rr["T_ring"][qi, c] = int((rg == c + 1).sum())
                    for oi, o in enumerate(ORDERS):
                        rr["olen"][qi, oi] = od[rd_][o].size
                        rr["found"][qi, oi] = found_counts(od[rd_][o], Tr, GRID[rd_])
                        for c in range(RING_CLASSES):
                            rr["found_ring"][qi, oi, :, c] = found_counts(od[rd_][o], Tr[rg == c + 1], GRID[rd_])
                    v, h2 = od[rd_]["V"], od[rd_]["H2"]
                    rr["v_not_h2"][qi] = [int(np.isin(np.setdiff1d(v[:N], h2[:N]), Tr).sum()) for N in GRID[rd_]]
                if (qi + 1) % 100 == 0:
                    log(f"{name}: {qi + 1}/{nQ} questions ({time.time() - t:.0f}s)")
        holder.update(res=res, use=use, vo=vo_chain, sizes=sizes, ids=ids, samp=samp, Q=Q, checked=checked,
                      sample_sha=sample_sha, carve_sha=pop.digest, seconds=time.time() - t)

    E.run_carves(op, name, {"s1eval"}, (), threads, sink)
    h = holder
    if limit:
        log(f"{name} development run: {len(h['ids'])} questions, every order and both reads ran; pools and seeds "
            f"checked against P2a's reader on {h['checked']}; {h['seconds']:.0f}s "
            f"({h['seconds'] / max(1, len(h['ids'])):.2f}s each), gpu peak "
            f"{gpu_peak(torch, dev) / 2 ** 30:.1f} GB, peak {peak_rss() / 2 ** 30:.1f} GB; nothing written")
        return 0
    arrs = {f"{rd_}|{k}": v for rd_, d in h["res"].items() for k, v in d.items()}
    arrs.update({f"use|{k}": v for k, v in h["use"].items()})
    arrs.update({f"vo|{rd_}": v for rd_, v in h["vo"].items()})
    arrs.update(sizes=h["sizes"], pos=np.asarray(h["samp"], np.int64), ids=np.asarray(h["ids"]))
    p = OUT / f"read_{name}.npz"
    save_npz(p, **arrs)
    r5, r5p = np.asarray(r1["r5"]), np.asarray(r1["r5_p2a"])
    srec = json.loads((SPACE / f"space_{name}.json").read_text(encoding="utf-8"))
    rec = {"declared_in": "docs/P5A_NODE_SPACE.md", "stage": "read", "dataset": name, "carve": "s1eval",
           "carve_questions": h["Q"], "questions": len(h["ids"]), "carve_sha256": h["carve_sha"],
           "sample_sha256": h["sample_sha"],
           "checks": {"pools_and_seeds_equal_p2a_reader": h["checked"],
                      "zsp_r5_differs_from_p2a": int((np.abs(r5 - r5p) > 1e-6).sum()),
                      "zsp_r5_questions": int(r5.size)},
           "orders": list(ORDERS), "grid": {k: list(v) for k, v in GRID.items()}, "h1_setting": H1_SETTING,
           "npz_sha256": sha_file(p), "read1_records": r1["records"], "frame_sha256": sha_file(FRAME),
           "space_record_sha256": sha_file(SPACE / f"space_{name}.json"), "space_sha256": srec["space_sha256"],
           "u_sha256": srec["u_sha256"], "map_sha256": sha_file(MAP), "script_sha256": script_sha256(),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "device": str(dev),
           "gpu_peak_bytes": gpu_peak(torch, dev), "peak_rss_bytes": peak_rss(),
           "read_s": round(h["seconds"], 1), "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(OUT / f"read_{name}.json", rec)
    log(f"{name}: outputs/p5a/read_{name}.json ({rec['seconds']:.0f}s)")
    return 0


# ── the verdict (section 5 of the declared file) ─────────────────────────────


def ratio_boot(fa, fb, t, rng_seed=0, boot=BOOT):
    """(sum fa - sum fb) / sum t over questions, with a question bootstrap's 95% interval."""
    fa, fb, t = (np.asarray(v, np.float64) for v in (fa, fb, t))
    q = t.size
    rng = np.random.default_rng(rng_seed)
    ds = np.empty(boot)
    for a in range(0, boot, 100):
        b = min(boot, a + 100)
        ix = rng.integers(0, q, size=(b - a, q))
        ds[a:b] = (fa[ix].sum(1) - fb[ix].sum(1)) / np.maximum(t[ix].sum(1), 1e-12)
    return float((fa.sum() - fb.sum()) / t.sum()), float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975))


def counted(d, lo, hi):
    if d > TIE and lo > 0:
        return 1
    if d < -TIE and hi < 0:
        return -1
    return 0


def label(signs):
    up, down = any(s > 0 for s in signs), any(s < 0 for s in signs)
    return "MIXED" if up and down else "MORE" if up else "FEWER" if down else "TIE"


def verdict_stage():
    recs, data = {}, {}
    for name in SIX:
        rp = OUT / f"read_{name}.json"
        recs[name] = json.loads(rp.read_text(encoding="utf-8"))
        p = OUT / f"read_{name}.npz"
        if sha_file(p) != recs[name]["npz_sha256"]:
            raise SystemExit(f"{p}: not its record's file")
        with np.load(p) as z:
            data[name] = {k: z[k] for k in z.files}
    for key in ("script_sha256", "frame_sha256", "map_sha256"):
        if len({r[key] for r in recs.values()}) != 1:
            raise SystemExit(f"the reads ran on different {key}")
    checks = {}
    for name, r in recs.items():
        c = r["checks"]
        share = c["zsp_r5_differs_from_p2a"] / max(1, c["zsp_r5_questions"])
        checks[name] = {**c, "share": round(share, 4)}
        if c["pools_and_seeds_equal_p2a_reader"] != r["questions"] or share > R1_SHARE:
            raise SystemExit(f"{name}: the checks fail ({checks[name]})")
    spaces = {}
    for name in SIX:
        s = json.loads((SPACE / f"space_{name}.json").read_text(encoding="utf-8"))
        if sha_file(SPACE / f"space_{name}.json") != recs[name]["space_record_sha256"]:
            raise SystemExit(f"{name}: the space record is not the read's")
        for fit in ("fit", "start_fit"):
            if not all(np.isfinite(v) for _s, v in s[fit]["loss_hist"]):
                raise SystemExit(f"{name}: the {fit}'s loss is not finite")
        spaces[name] = {"delta_rms": round(s["fit"]["delta_rms"], 4), "gamma": round(s["fit"]["gamma"], 4),
                        "loss_first_last": [s["fit"]["loss_hist"][0][1], s["fit"]["loss_hist"][-1][1]],
                        "start_loss_first_last": [s["start_fit"]["loss_hist"][0][1],
                                                  s["start_fit"]["loss_hist"][-1][1]],
                        "families": {f: {k: v.get(k) for k in ("edges", "offsets", "residual2_median",
                                                                "displacement2_median", "latent_share")}
                                     for f, v in s["fit"]["families"].items()}}
    oi = {o: k for k, o in enumerate(ORDERS)}
    calls, found, beside = {}, {}, {}
    for rd_ in GRID:
        calls[rd_], found[rd_], beside[rd_] = {}, {}, {}
        for name in SIX:
            D = data[name]
            T = D[f"{rd_}|T"]
            keep = T > 0
            Fd = D[f"{rd_}|found"][keep].astype(np.float64)
            t = T[keep]
            found[rd_][name] = {"questions": int(keep.sum()), "golds": int(t.sum()),
                                **{o: [round(float(Fd[:, oi[o], k].sum() / max(1, t.sum())), 4)
                                       for k in range(len(GRID[rd_]))] for o in ORDERS}}
            if not keep.any():
                calls[rd_][name] = {"label": "NO_GOLDS"}
                continue
            pts, signs = [], []
            for k, N in enumerate(GRID[rd_]):
                d, lo, hi = ratio_boot(Fd[:, oi["V"], k], Fd[:, oi["H2"], k], t)
                pts.append({"N": N, "delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                            "counted": counted(d, lo, hi)})
                signs.append(pts[-1]["counted"])
            vo_signs = []
            for k in range(len(GRID[rd_])):
                d, lo, hi = ratio_boot(Fd[:, oi["VO"], k], Fd[:, oi["H2"], k], t)
                vo_signs.append(counted(d, lo, hi))
            calls[rd_][name] = {"label": label(signs), "points": pts, "VO_against_H2": label(vo_signs)}
            Tr = D[f"{rd_}|T_ring"][keep]
            FR = D[f"{rd_}|found_ring"][keep]
            beside[rd_][name] = {
                "V_not_H2": [round(float(D[f"{rd_}|v_not_h2"][keep][:, k].sum() / t.sum()), 4)
                             for k in range(len(GRID[rd_]))],
                "by_ring": {f"ring_{c + 1 if c < 2 else '3+'}": {
                    "golds": int(Tr[:, c].sum()),
                    **{o: [round(float(FR[:, oi[o], k, c].sum() / max(1, Tr[:, c].sum())), 4)
                           for k in range(len(GRID[rd_]))] for o in ("V", "H2")}} for c in range(RING_CLASSES)},
                "order_length_mean": {o: round(float(D[f"{rd_}|olen"][keep][:, oi[o]].mean()), 1) for o in ORDERS}}
    finds = {}
    for rd_ in GRID:
        labs = [calls[rd_][n]["label"] for n in SIX if calls[rd_][n]["label"] != "NO_GOLDS"]
        finds[rd_] = "FINDS_MORE" if ("MORE" in labs and "FEWER" not in labs) else "DOES_NOT"
    call = "GOES_FURTHER" if "FINDS_MORE" in finds.values() else "STOPS"
    where = {}
    for rd_ in GRID:
        where[rd_] = {}
        for n in SIX:
            c = calls[rd_][n]
            if c["label"] == "NO_GOLDS":
                continue
            where[rd_][n] = ("the map" if c["label"] != "MORE" and c["VO_against_H2"] == "MORE"
                             else "the space" if c["label"] != "MORE" else "-")
    use = {}
    for name in SIX:
        D = data[name]
        use[name] = {}
        for sp in ("V", "V0"):
            t1 = np.unique(D[f"use|top1_{sp}"], return_counts=True)
            t2 = np.unique(D[f"use|top2_{sp}"], return_counts=True)
            use[name][sp] = {"gate_mean": round(float(D[f"use|gate_{sp}"].mean()), 4),
                             "top1_most_used": [[int(a), int(b)] for a, b in
                                                sorted(zip(*t1), key=lambda x: (-x[1], x[0]))[:5]],
                             "top2_most_used": [[int(a), int(b)] for a, b in
                                                sorted(zip(*t2), key=lambda x: (-x[1], x[0]))[:5]]}
    rec = {"declared_in": "docs/P5A_NODE_SPACE.md", "call": call, "finds": finds, "calls": calls, "found": found,
           "beside": beside, "where_it_fell_short": where if call == "STOPS" else None, "fits": spaces,
           "chain_use": use, "checks": checks, "grid": {k: list(v) for k, v in GRID.items()}, "boot": BOOT,
           "tie": TIE, "reads": {n: sha_file(OUT / f"read_{n}.json") for n in SIX}, "map_json": sha_file(MAP_JSON),
           "script_sha256": script_sha256(), "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"P5a: {call}; " + "; ".join(f"{rd_} {finds[rd_]} (" + ", ".join(
        f"{n} {calls[rd_][n]['label']}" for n in SIX) + ")" for rd_ in GRID))
    return 0


def report_md(rec):
    L = [f"# P5a verdict: {rec['call']}", ""]
    for rd_, grid in rec["grid"].items():
        L += [f"## The {rd_} read: {rec['finds'][rd_]}", "",
              "| dataset | call | VO against H2 | " + " | ".join(f"N={N}" for N in grid) + " |",
              "| --- | --- | --- | " + " | ".join("---:" for _ in grid) + " |"]
        for n, c in rec["calls"][rd_].items():
            if c["label"] == "NO_GOLDS":
                L.append(f"| {n} | NO_GOLDS | | " + " | ".join("" for _ in grid) + " |")
                continue
            L.append(f"| {n} | {c['label']} | {c['VO_against_H2']} | "
                     + " | ".join(f"{p['delta']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]" for p in c["points"]) + " |")
        L += ["", "found(N) per order:", "", "| dataset | order | " + " | ".join(f"N={N}" for N in grid) + " |",
              "| --- | --- | " + " | ".join("---:" for _ in grid) + " |"]
        for n, f in rec["found"][rd_].items():
            for o in ORDERS:
                L.append(f"| {n} | {o} | " + " | ".join(f"{v:.4f}" for v in f[o]) + " |")
        L.append("")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    import torch
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = torch.device("cpu")
    # the E-step recovers each displacement's offset and sign
    rels = rng.normal(0, 1.0, (3, DIM)).astype(np.float32) * 0.5
    d = torch.from_numpy(rels[[0, 0, 1]] * np.asarray([[1.0], [-1.0], [1.0]], np.float32))
    k, s = assign(d, torch.from_numpy(rels), True)
    assert k.tolist() == [0, 0, 1] and s.tolist() == [1.0, -1.0, 1.0], (k, s)
    kn, sn = assign_np(d.numpy().astype(np.float64), rels.astype(np.float64), True)
    assert kn.tolist() == [0, 0, 1] and sn.tolist() == [1.0, -1.0, 1.0]
    k, s = assign(d[[0, 2]], torch.from_numpy(rels), False)
    assert k.tolist() == [0, 1] and s.tolist() == [1.0, 1.0]
    # the frame: orthonormal, and the dominant direction first
    X = rng.normal(0, 1, (4000, 80))
    X[:, 3] *= 10.0
    mu, P, ev = frame_of(X.astype(np.float16))
    assert P.shape == (DIM, 80) and np.allclose(P @ P.T, np.eye(DIM), atol=1e-4) and abs(P[0, 3]) > 0.99, P[0, 3]
    assert ev[0] >= ev[1] >= ev[-1]
    # the fit's loss falls on a toy graph; the start's fit keeps every node at its start
    n, K = 3000, 3
    true = rng.normal(0, 1.0, (n, DIM)).astype(np.float32) * 0.3
    rr = rng.normal(0, 1.0, (K, DIM)).astype(np.float32) * 0.5
    h = rng.integers(0, n, 2000)
    kk = rng.integers(0, K, 2000)
    t = np.asarray([int(np.argmin(((true - x) ** 2).sum(1))) for x in true[h] + rr[kk]])
    keep = t != h
    u0 = true + rng.normal(0, 0.05, true.shape).astype(np.float32)
    fam = Family("structural", h[keep], t[keep], n_rel=K, signed=False)
    u, offs, info = fit_space(u0, [fam, Family("knn", h[keep][:500], t[keep][:500], n_rel=K)], dev, steps=60,
                              logf=lambda *_: None, batch=4096)
    assert u.shape == (n, DIM) and offs[0].shape == (K, DIM) and info["loss_hist"][-1][1] < info["loss_hist"][0][1]
    us, _o, _i = fit_space(u0, [fam], dev, steps=5, logf=lambda *_: None, fit_nodes=False, batch=4096)
    assert torch.equal(us, torch.from_numpy(u0))
    named = Family("structural", h[keep], t[keep], rel=kk[keep], n_rel=K)
    r0 = init_offsets(u0, named, rng)
    dd = u0[t[keep]].astype(np.float64) - u0[h[keep]].astype(np.float64)
    assert np.allclose(r0[1], dd[kk[keep] == 1].mean(0))
    # the chunked nearest-node search with a mask equals brute force, ties by node id
    U = torch.from_numpy(u0)
    U[7] = U[5]                                                   # an exact tie: 5 before 7
    UN = (U ** 2).sum(1)
    tg = U[[5, 100, 200]] + 0.001
    mask = np.asarray([100, 300, 2999], np.int64)
    ids, d2 = masked_knn(U, UN, tg, 20, mask, chunk=700)
    full = (tg ** 2).sum(1)[:, None] + UN[None] - 2.0 * tg @ U.T
    full[:, torch.from_numpy(mask)] = float("inf")
    for i in range(3):
        o = np.lexsort((np.arange(n), full[i].numpy()))[:20]
        assert ids[i].tolist() == o.tolist(), (i, ids[i][:5], o[:5])
    assert ids[0].tolist().index(5) < ids[0].tolist().index(7)
    # the gold ranks equal brute force, seeds excluded
    seeds = np.asarray([0, 1], np.int64)
    R = torch.from_numpy(rr)
    chains = [(0, -1), (1, 2)]
    golds = np.asarray([10, 20, 30], np.int64)
    e = chain_evidence(U, UN, seeds, golds, R, chains, chunk=700)
    ref = []
    for c1, c2 in chains:
        stp = R[c1] + (R[c2] if c2 >= 0 else 0)
        best = np.zeros(golds.size)
        for s_ in seeds:
            tt = U[s_] + stp
            dist = ((U - tt) ** 2).sum(1).numpy().astype(np.float64)
            for gi, g in enumerate(golds):
                tol = TOL * (float((tt.double() ** 2).sum()) + float((U[g].double() ** 2).sum()))
                closer = dist < dist[g] - tol
                closer[seeds] = False
                best[gi] = max(best[gi], 1.0 / (1 + closer.sum()))
        ref.append(best.mean())
    assert np.allclose(e, ref, rtol=1e-6), (e, ref)
    # the chains nearest a gold equal brute force over every one-hop and two-hop chain
    Rt = chain_table([torch.from_numpy(rr)])
    C = Rt.shape[0]
    us_, ug_ = U[torch.from_numpy(seeds)], U[torch.from_numpy(golds)]
    cid, cd = nearest_chains(us_, ug_, Rt, 5)
    allc = [(c, -1) for c in range(C)] + [(a, b) for a in range(C) for b in range(C)]
    bd = [float(min(((us_[si] + chain_steps(Rt, [c])[0] - ug_) ** 2).sum(1).min() for si in range(2))) for c in allc]
    bo = np.lexsort((np.arange(len(allc)), np.asarray(bd)))[:5]
    assert cid.tolist() == bo.tolist() and np.allclose(cd, np.asarray(bd)[bo], rtol=1e-4), (cid, bo)
    cid2, _ = nearest_chains(us_, ug_, Rt, 3, first_cap=2)
    first = np.lexsort((np.arange(C), np.asarray(bd[:C])))[:2]
    sub = list(range(C)) + [C + a * C + b for a in first for b in range(C)]
    so = sorted(sub, key=lambda c: (bd[c], c))[:3]
    assert cid2.tolist() == so, (cid2, so)
    # fusion: each node's best entry; the depth cut leaves the fused top unchanged
    node = np.asarray([1, 2, 3, 3, 4])
    o = fuse_entries(node, np.asarray([1.0, 0.5, 0.5, 0.9, 0.5]), np.asarray([0.1, 0.2, 0.3, 0.3, 0.1]), 10)
    assert o.tolist() == [1, 3, 4, 2], o
    w = np.asarray([0.6, 0.6, 0.05, 0.01])
    L = 4
    tgts = U[[11, 12, 13, 14]] + 0.01
    big = 40
    ii, dq = masked_knn(U, UN, tgts, big, mask, chunk=700)
    ii, dq = ii.numpy(), dq.double().numpy()
    posf = np.broadcast_to(np.arange(1, big + 1)[None, :], (L, big))
    full_order = fuse_entries(ii.ravel(), (w[:, None] / posf).ravel(), dq.ravel(), 10)
    dep = depth_of(w, 10)
    kp = posf <= dep[:, None]
    cut_order = fuse_entries(ii[kp], (w[:, None] / posf)[kp], dq[kp], 10)
    assert full_order.tolist() == cut_order.tolist(), (full_order, cut_order)
    # the low read's fusion equals a brute-force max(w / position)
    cand = np.asarray([3, 9, 40, 41, 77, 500], np.int64)
    lo_ = fused_low(U, tgts, w, cand, 10)
    sc = {}
    for li in range(L):
        dist = ((U[torch.from_numpy(cand)] - tgts[li]) ** 2).sum(1).numpy().astype(np.float64)
        for p_, ci in enumerate(np.lexsort((cand, dist))):
            v = (w[li] / (p_ + 1), -dist[ci])
            sc[int(cand[ci])] = max(sc.get(int(cand[ci]), (-1, 0)), v)
    bo = sorted(sc, key=lambda v: (-sc[v][0], -sc[v][1], v))
    assert lo_.tolist() == bo, (lo_, bo)
    # the rings equal a breadth-first search, and the marks come back clear
    ip = np.asarray([0, 2, 3, 4, 5, 5, 6], np.int64)
    col = np.asarray([1, 2, 3, 4, 0, 5], np.int32)
    ipe = np.zeros(7, np.int64)
    cole = np.zeros(0, np.int32)
    adj = [(ip, col), (ipe, cole), (ipe, cole)]
    mk = np.zeros(6, np.int8)
    bf = np.empty(6, np.int64)
    a1, a2 = rings_of(adj, np.asarray([0], np.int64), mk, bf)
    assert a1.tolist() == [1, 2] and a2.tolist() == [3, 4] and not mk.any(), (a1, a2)
    # H2: ring 1 then ring 2, each by cosine (ties by id), then the dense list; the low read by ring, then the rest
    r1_, r2_ = np.asarray([1, 2, 6]), np.asarray([3, 4, 8])
    pool = np.asarray([0, 6, 8, 9], np.int64)
    need = np.union1d(np.union1d(r1_, r2_), pool)
    cv = {0: .1, 1: .2, 2: .9, 3: .5, 4: .5, 6: .3, 8: .7, 9: .95}
    cosv = np.asarray([cv[v] for v in need], np.float32)
    miss, low = h2_orders(r1_, r2_, need, cosv, pool, np.asarray([0]), np.asarray([9, 11, 2, 12]), 10)
    assert miss.tolist() == [2, 1, 3, 4, 11, 12], miss
    assert low.tolist() == [6, 8, 9], low
    # EM: the log-likelihood never falls, and the map picks each question's chain
    Cc, Qn = 3, 200
    r = torch.eye(Cc, DIM)
    lab = torch.arange(Qn) % 2
    x = torch.zeros(Qn, DIM + 1)
    x[:, -1] = 1
    x[torch.arange(Qn), lab] = 1.0
    e1 = torch.full((Qn, Cc), 1e-3)
    e1[torch.arange(Qn), lab] = 1.0
    Wm, trace = em_map([(x, r, e1, torch.full((Qn, Cc * Cc), 1e-3))], logf=lambda *_: None)
    assert all(b >= a - 1e-9 for a, b in zip(trace, trace[1:])), trace
    one, _two = map_logprobs(Wm, x, r)
    assert (one.argmax(1) == lab).all()
    ch, wv, gate = chains_for(Wm, x[1], r)
    assert ch[0] == (1, -1) and gate > 0.5, (ch, gate)
    assert len(ch) == min(TOP1, Cc) + min(TOP2, min(TOPK12, Cc) ** 2) and np.all(wv > 0)
    # the M-step's marginal form equals the full sum over every one-hop and two-hop chain
    g64 = torch.Generator().manual_seed(1)
    Wr = tuple(torch.randn(s, generator=g64, dtype=torch.float64) for s in ((DIM, DIM + 1), (DIM, DIM + 1), (DIM + 1,)))
    xr = torch.randn(7, DIM + 1, generator=g64, dtype=torch.float64)
    rr4 = torch.randn(4, DIM, generator=g64, dtype=torch.float64) * 0.1
    resp = torch.softmax(torch.randn(7, 4 + 16, generator=g64, dtype=torch.float64), 1)
    o4, t4 = map_logprobs(Wr, xr, rr4)
    full_sum = (resp * torch.cat([o4, t4], 1)).sum(1)
    assert torch.allclose(mstep_terms(Wr, xr, rr4, resp_stats(resp, 4)), full_sum, rtol=1e-10, atol=1e-10)
    # the sample: sorted, unique, the same every time
    s1, s2 = sample_of(5000), sample_of(5000)
    assert np.array_equal(s1, s2) and s1.size == SAMPLE and np.all(np.diff(s1) > 0) and sample_of(1503).size == 1503
    # dense evidence: counted chains carry their value, the floor elsewhere
    e1d, e2d = dense_evidence(np.asarray([[0, 3 + 1 * 3 + 2]]), np.asarray([[0.5, 0.25]]), 3, 1e-4)
    assert e1d[0, 0] == 0.5 and e2d[0, 1 * 3 + 2] == 0.25 and e1d[0, 1] == 1e-4
    verdict_selftest()
    print("selftest passed")
    return 0


def verdict_selftest():
    """The verdict's plumbing on made-up reads in a temporary folder: V finds one more miss gold than H2 on every
    musique question (MORE there, TIE elsewhere) and one fewer low gold on every squad question (FEWER)."""
    import shutil
    import tempfile
    g = globals()
    keep = {k: g[k] for k in ("OUT", "SPACE", "MAP_JSON", "VERDICT", "VERDICT_MD")}
    tmp = Path(tempfile.mkdtemp(prefix="p5a_selftest_"))
    try:
        g["OUT"], g["SPACE"] = tmp, tmp / "space"
        g["MAP_JSON"], g["VERDICT"], g["VERDICT_MD"] = tmp / "map.json", tmp / "verdict.json", tmp / "verdict.md"
        write_json(g["MAP_JSON"], {"stage": "map"})
        rng = np.random.default_rng(0)
        nq = 40
        for name in SIX:
            arrs = {}
            for rd_, grid in GRID.items():
                L = len(grid)
                T = rng.integers(1, 4, nq)
                Tr = np.stack([T - T // 2, T // 2, np.zeros(nq, np.int64)], 1)
                fnd = np.zeros((nq, len(ORDERS), L), np.int32)
                fnd[:, :, :] = np.minimum(T, 1)[:, None, None]
                if name == "musique" and rd_ == "miss":
                    fnd[:, ORDERS.index("V"), :] += 1
                if name == "squad" and rd_ == "low":
                    fnd[:, ORDERS.index("V"), :] = 0
                fr = np.zeros((nq, len(ORDERS), L, RING_CLASSES), np.int32)
                fr[..., 0] = fnd
                arrs.update({f"{rd_}|T": T, f"{rd_}|T_ring": Tr, f"{rd_}|found": fnd, f"{rd_}|found_ring": fr,
                             f"{rd_}|v_not_h2": np.zeros((nq, L), np.int32),
                             f"{rd_}|olen": np.full((nq, len(ORDERS)), 10, np.int32)})
                arrs[f"vo|{rd_}"] = np.zeros(nq, np.int64)
            for sp in ("V", "V0"):
                arrs[f"use|gate_{sp}"] = np.full(nq, 0.5)
                arrs[f"use|top1_{sp}"] = rng.integers(0, 4, nq)
                arrs[f"use|top2_{sp}"] = rng.integers(4, 20, nq)
            save_npz(tmp / f"read_{name}.npz", **arrs)
            srec = {"fit": {"delta_rms": 0.1, "gamma": 1.0, "loss_hist": [[0, 1.3], [9, 1.2]],
                            "families": {f: {"edges": 5, "offsets": 8, "residual2_median": 0.1,
                                             "displacement2_median": 0.2, "latent_share": None} for f in FAMS}},
                    "start_fit": {"loss_hist": [[0, 1.3], [9, 1.25]]}}
            write_json(g["SPACE"] / f"space_{name}.json", srec)
            write_json(tmp / f"read_{name}.json", {
                "questions": nq, "npz_sha256": sha_file(tmp / f"read_{name}.npz"), "script_sha256": "s",
                "frame_sha256": "f", "map_sha256": "m", "space_record_sha256": sha_file(g["SPACE"] / f"space_{name}.json"),
                "checks": {"pools_and_seeds_equal_p2a_reader": nq, "zsp_r5_differs_from_p2a": 0,
                           "zsp_r5_questions": nq}})
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            verdict_stage()
        v = json.loads(g["VERDICT"].read_text(encoding="utf-8"))
        assert v["calls"]["miss"]["musique"]["label"] == "MORE" and v["finds"]["miss"] == "FINDS_MORE", v["finds"]
        assert all(v["calls"]["miss"][n]["label"] == "TIE" for n in SIX if n != "musique")
        assert v["calls"]["low"]["squad"]["label"] == "FEWER" and v["finds"]["low"] == "DOES_NOT"
        assert v["call"] == "GOES_FURTHER" and v["where_it_fell_short"] is None
        assert g["VERDICT_MD"].read_text(encoding="utf-8").startswith("# P5a verdict: GOES_FURTHER")
    finally:
        g.update(keep)
        shutil.rmtree(tmp, ignore_errors=True)


# ── main ─────────────────────────────────────────────────────────────────────


def parse_part(s):
    try:
        k, n = (int(v) for v in s.split("/"))
    except Exception:
        raise SystemExit(f"--part {s}: need K/N")
    if not 0 <= k < n:
        raise SystemExit(f"--part {s}: need 0 <= K < N")
    return k, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("base", "train", "map", "read1", "read", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--part")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.steps and not a.limit:
        raise SystemExit("--steps is for development runs (with --limit) only")
    if a.stage in ("train", "read1", "read") and a.dataset not in SIX:
        raise SystemExit(f"dataset must be one of {SIX}")
    device = a.device
    if a.stage in ("train", "read") and device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.stage == "base":
        return base_stage(a.host, a.limit)
    if a.stage == "train":
        return train_stage(a.dataset, device, a.threads, a.host, a.limit, a.steps)
    if a.stage == "map":
        return map_stage(a.threads)
    if a.stage == "read1":
        if not a.part:
            raise SystemExit("read1 needs --part K/N (P2a's parts)")
        return read1_stage(a.dataset, parse_part(a.part), a.host, a.limit)
    if a.stage == "read":
        return read_stage(a.dataset, device, a.threads, a.host, a.limit, a.steps)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
