"""SEM2 R1: edge labels as learned operators on frozen vectors, read as link recovery (docs/SEM2_EDGE_OPERATORS.md
section 7, declared 23 Nov 2026 before any number).

    python outputs/mp_unified/zsem2.py selftest [--threads 8]
    python outputs/mp_unified/zsem2.py r1 --ds D --arm e3v|frozen [--host] [--threads 8] [--smoke --out DIR]
    python outputs/mp_unified/zsem2.py report

Links only: no question and no gold is read. Each label gets an operator on x = P v / |P v| (P: D -> D + 256, starting
as [I; N(0, 1e-4^2)], amendment 7.11 (a)); a directed label (S) a block rotation, an undirected one (R) a free weight per
plane starting at 1 (amendment 7.11 (b)). At the start every score is the cosine (gate G-0). The fit reads train-node edges only (node-inductive split); the read
ranks each held-out tail among all nodes, filtered, against the cosine. zul1, zul2, zmhr and zenc are imported and
never edited. Outputs only under outputs/sem2/r1/ (json and md committed; npz not).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zenc as E  # noqa: E402  (reads --threads from sys.argv before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 7"
OUT = ROOT / "outputs" / "sem2" / "r1"
SEVEN = ("multihoprag", "squad", "musique", "metaqa", "webqsp", "hotpotqa", "2wiki")   # section 7.10's row order
SMALL = ("multihoprag", "squad", "musique", "metaqa")                                  # the beside fits D' = D and map only
ARMS = ("e3v", "frozen")
TEST, DEV, TRAIN = 0, 1, 2
CFG = {"extra": 256, "tau": 0.05, "batch": 1024, "neg": 4096, "lr": 1e-3, "epoch": 200_000, "max_epochs": 10,
       "patience": 2, "min_op": 500, "top_s": 12, "n_test": 1000, "n_dev": 500}
SMOKE = {"batch": 256, "neg": 512, "epoch": 4096, "max_epochs": 1, "n_test": 60, "n_dev": 40}
BOOT, SEED = 2000, 0
EPS = 1e-6                  # a score above the true tail's by more than this ranks above it; within it, a tie (half)
NORM_TOL = 1e-2             # section 7.1
G0_SCORE, G0_MRR = 1e-4, 1e-3
EXTRA_INIT = 1e-4          # amendment 7.11 (a): the extra rows' start, N(0, EXTRA_INIT^2) per entry
SMALL_POP = 4_000_000       # a label's oriented edges up to this are enumerated; above, sampled by rejection


def log(msg):
    print(f"[zsem2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zsem2: {msg}")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def torch_():
    import torch
    torch.set_num_threads(E.THREADS)
    return torch


# ── the split (section 7.3) ─────────────────────────────────────────────────


def node_split(name, n):
    out = np.empty(n, np.int8)
    pre = f"sem2|{name}|".encode()
    for i in range(n):
        u = int.from_bytes(hashlib.sha256(pre + str(i).encode()).digest()[:8], "big") / 2 ** 64
        out[i] = TEST if u < 0.10 else (DEV if u < 0.20 else TRAIN)
    return out


def ends(keys, n):
    a = keys // n
    return a, keys - a * n


def expand(lo, hi):
    """All positions of the ranges [lo, hi), in order."""
    cnt = (hi - lo).astype(np.int64)
    tot = int(cnt.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    return np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo.astype(np.int64), cnt)


def member(sk, keys):
    sk = np.asarray(sk, np.int64)
    if keys.size == 0 or sk.size == 0:
        return np.zeros(sk.size, bool)
    i = np.searchsorted(keys, sk)
    i[i == keys.size] = keys.size - 1
    return keys[i] == sk


# ── the graph: labels, operators, read groups ───────────────────────────────


class Graph:
    """R labels: undirected pair keys (min * n + max), sorted unique. S: directed edges as one sorted key array
    ((src * n + dst) * RS + rel), with src, dst and rel beside it. Labels: R first (ids 0..nR-1), then S relation r as
    nR + r."""

    def __init__(self, name, n, R, S=None, s_vocab=()):
        self.name, self.n = name, n
        self.r_names = [k for k, _ in R]
        self.r_keys = [v for _, v in R]
        self.nR = len(R)
        self.s_vocab = list(s_vocab)
        self.RS = max(len(self.s_vocab), 1)
        if S is None:
            self.s_key = np.zeros(0, np.int64)
        else:
            self.s_key = S
        k = self.s_key // self.RS
        self.s_rel = (self.s_key - k * self.RS).astype(np.int64)
        self.s_src, self.s_dst = ends(k, n)
        del k

    def lab_name(self, L):
        return self.r_names[L] if L < self.nR else "S." + self.s_vocab[L - self.nR]


def count_r(keys, n, split, pred, block=1 << 24):
    c = 0
    for i in range(0, keys.size, block):
        a, b = ends(keys[i:i + block], n)
        c += int(pred(split[a], split[b]).sum()) + int(pred(split[b], split[a]).sum())
    return c


def is_train(sa, sb):
    return (sa == TRAIN) & (sb == TRAIN)


def is_dev(sa, sb):
    return (sa == DEV) & (sb != TEST)


def is_test(sa, sb):
    return sa == TEST


class Group:
    """A read label: one R label (both orientations of each pair), or a set of S relations (each edge with its own
    relation's operator)."""

    def __init__(self, G, name, lab=None, gi=None):
        self.G, self.name, self.lab, self.gi = G, name, lab, gi
        self.directed = gi is not None
        self.pop = int(gi.size) if self.directed else 2 * int(G.r_keys[lab].size)

    def edges(self, idx):
        G = self.G
        idx = np.asarray(idx, np.int64)
        if self.directed:
            j = self.gi[idx]
            return G.s_src[j], G.s_dst[j], G.nR + G.s_rel[j]
        a, b = ends(G.r_keys[self.lab][idx // 2], G.n)
        o = (idx % 2) == 0
        return np.where(o, a, b), np.where(o, b, a), np.full(idx.size, self.lab, np.int64)

    def sample(self, split, pred, k, rng):
        if self.pop == 0 or k <= 0:
            z = np.zeros(0, np.int64)
            return z, z, z
        if self.pop <= SMALL_POP:
            idx = np.arange(self.pop, dtype=np.int64)
            s, d, lab = self.edges(idx)
            idx = idx[pred(split[s], split[d])]
            if idx.size > k:
                idx = np.sort(rng.choice(idx, k, replace=False))
            return self.edges(idx)
        m = min(self.pop, 15 * k + 10_000)
        while True:
            idx = rng.choice(self.pop, m, replace=False)
            s, d, lab = self.edges(idx)
            ok = np.flatnonzero(pred(split[s], split[d]))
            if ok.size >= k or m == self.pop:
                ok = ok[:k]
                return s[ok], d[ok], lab[ok]
            m = min(self.pop, 4 * m)


def label_summary(labs):
    """Every R label and every S label with its own operator; the S relations pooled into S.rare in one line."""
    out, rare = {}, {"relations": 0, "edges": 0, "train_edges": 0}
    for v in labs.values():
        if v.get("op") == "S.rare":
            rare["relations"] += 1
            rare["edges"] += v["edges"]
            rare["train_edges"] += v["train_edges"]
        else:
            out[v["name"]] = {k: x for k, x in v.items() if k != "name"}
    if rare["relations"]:
        out["S.rare (pooled)"] = rare
    return out


def plan(G, split, cfg):
    """Train-edge counts per label, the operators (section 7.2) and the read groups."""
    n = G.n
    labs = {}
    for L, keys in enumerate(G.r_keys):
        labs[L] = {"name": G.r_names[L], "directed": False, "pairs": int(keys.size), "train_edges": count_r(keys, n, split, is_train)}
    if G.s_key.size:
        tr = (split[G.s_src] == TRAIN) & (split[G.s_dst] == TRAIN)
        ct = np.bincount(G.s_rel[tr], minlength=G.RS)
        ca = np.bincount(G.s_rel, minlength=G.RS)
        for r in np.flatnonzero(ca):
            labs[G.nR + int(r)] = {"name": "S." + G.s_vocab[r], "directed": True, "edges": int(ca[r]), "train_edges": int(ct[r])}
    ops, op_of = [], np.full(G.nR + G.RS, -1, np.int64)
    for L in range(G.nR):
        if labs[L]["train_edges"] >= cfg["min_op"]:
            op_of[L] = len(ops)
            ops.append({"name": labs[L]["name"], "directed": False})
    rare = None
    for L in sorted(x for x in labs if x >= G.nR):
        if labs[L]["train_edges"] >= cfg["min_op"]:
            op_of[L] = len(ops)
            ops.append({"name": labs[L]["name"], "directed": True})
        else:
            if rare is None:
                rare = len(ops)
                ops.append({"name": "S.rare", "directed": True})
            op_of[L] = rare
    for L, v in labs.items():
        v["op"] = ops[op_of[L]]["name"] if op_of[L] >= 0 else None
    groups = [Group(G, G.r_names[L], lab=L) for L in range(G.nR) if op_of[L] >= 0]
    own_s = sorted((x for x in labs if x >= G.nR and labs[x]["op"] == labs[x]["name"]), key=lambda x: (-labs[x]["train_edges"], x))
    top = own_s[:cfg["top_s"]]
    for L in top:
        groups.append(Group(G, labs[L]["name"], gi=np.flatnonzero(G.s_rel == L - G.nR)))
    if G.s_key.size:
        rest = np.flatnonzero(~np.isin(G.s_rel, np.asarray([L - G.nR for L in top], np.int64)))
        if rest.size:
            groups.append(Group(G, "S.other", gi=rest))
    return labs, ops, op_of, groups


# ── vectors ─────────────────────────────────────────────────────────────────


class Vecs:
    def __init__(self, V, block=1 << 18):
        self.V, self.n, self.D = V, V.shape[0], V.shape[1]
        inv = np.empty(self.n, np.float32)
        worst = 0.0
        for i in range(0, self.n, block):
            x = V[i:i + block].astype(np.float32)
            nr = np.sqrt((x * x).sum(1))
            worst = max(worst, float(np.abs(nr - 1).max()) if nr.size else 0.0)
            if (nr <= 0).any():
                stop("a zero vector row")
            inv[i:i + block] = 1.0 / nr
        if worst > NORM_TOL:
            stop(f"a row's norm is off 1 by {worst:.4g} > {NORM_TOL} (section 7.1)")
        self.inv, self.worst = inv, worst

    def rows(self, idx):
        if isinstance(idx, slice):
            x = self.V[idx].astype(np.float32)
            x *= self.inv[idx][:, None]
        else:
            x = self.V[idx].astype(np.float32)
            x *= self.inv[idx][:, None]
        return x


# ── the model (section 7.4) ─────────────────────────────────────────────────


def make_model(torch, D, Dp, n_ops, form):
    nn = torch.nn

    class Op(nn.Module):
        def __init__(self):
            super().__init__()
            P = torch.zeros(Dp, D)
            P[:D, :D] = torch.eye(D)
            if Dp > D:          # amendment 7.11 (a): rows at exactly 0 get no gradient in any form
                g = torch.Generator().manual_seed(SEED)
                P[D:] = EXTRA_INIT * torch.randn(Dp - D, D, generator=g)
            self.P = nn.Parameter(P)
            self.form = form
            if form == "rotation" or form == "rotation_D":
                self.theta = nn.Parameter(torch.zeros(n_ops, Dp // 2))
                self.w = nn.Parameter(torch.ones(n_ops, Dp // 2))    # amendment 7.11 (b): undirected plane weights
            elif form == "translation":
                self.b = nn.Parameter(torch.zeros(n_ops, Dp))

        def embed(self, v):
            x = v @ self.P.T
            return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

        def query(self, x, op, directed):
            if self.form == "map_only":
                return x
            m = x.shape[0]
            if self.form == "translation":
                b = self.b[op]
                return x + torch.where(directed[:, None], b, 0.5 * b)
            th = self.theta[op]
            c, s = torch.cos(th), torch.sin(th)
            x2 = x.view(m, Dp // 2, 2)
            x0, x1 = x2[..., 0], x2[..., 1]
            rot = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).view(m, Dp)
            sym = (x2 * self.w[op].unsqueeze(-1)).view(m, Dp)
            return torch.where(directed[:, None], rot, sym)

    if Dp % 2:
        stop(f"D' = {Dp} is odd")
    return Op()


class Cosine:
    form = "cosine"

    def embed(self, v):
        return v

    def query(self, x, op, directed):
        return x


# ── the read (section 7.6) ──────────────────────────────────────────────────


def exclusions(G, smp):
    """(node, query) pairs each query leaves out of its rank: its source, and its source's true tails under the
    edge's own label (any split; the true tail itself among them). Sorted by node."""
    s, lab = smp["s"], smp["lab"]
    Q, n = s.size, G.n
    qs, ns = [np.arange(Q, dtype=np.int64)], [s.astype(np.int64)]
    for L in np.unique(lab[lab < G.nR]):
        J = np.flatnonzero(lab == L)
        mark = np.zeros(n, bool)
        mark[s[J]] = True
        pa, pb = [], []
        keys = G.r_keys[int(L)]
        for i in range(0, keys.size, 1 << 24):
            a, b = ends(keys[i:i + (1 << 24)], n)
            m = mark[a]
            pa.append(a[m])
            pb.append(b[m])
            m = mark[b]
            pa.append(b[m])
            pb.append(a[m])
        src, nbr = np.concatenate(pa), np.concatenate(pb)
        o = np.argsort(src, kind="stable")
        src, nbr = src[o], nbr[o]
        lo, hi = np.searchsorted(src, s[J], "left"), np.searchsorted(src, s[J], "right")
        qs.append(np.repeat(J, hi - lo))
        ns.append(nbr[expand(lo, hi)])
    J = np.flatnonzero(lab >= G.nR)
    if J.size:
        lo, hi = np.searchsorted(G.s_src, s[J], "left"), np.searchsorted(G.s_src, s[J], "right")
        pos = expand(lo, hi)
        qq = np.repeat(J, hi - lo)
        keep = G.s_rel[pos] == (lab[qq] - G.nR)
        qs.append(qq[keep])
        ns.append(G.s_dst[pos[keep]])
    q, nd = np.concatenate(qs), np.concatenate(ns)
    k = np.unique(nd * Q + q)
    nd, q = k // Q, k % Q
    tails = member(smp["d"] * Q + np.arange(Q), k)                 # every true tail is in its own list
    if not tails.all():
        stop("a query's true tail is not among its source's tails")
    return nd, q


def rank_read(torch, sc, X, smp, excl, op_of):
    s, d, lab = smp["s"], smp["d"], smp["lab"]
    Q = s.size
    if Q == 0:
        return np.zeros(0), np.zeros(0)
    op = torch.from_numpy(np.maximum(op_of[lab], 0))
    dirn = torch.from_numpy(lab >= smp["nR"])
    en, eq = excl
    with torch.no_grad():
        q = sc.query(sc.embed(torch.from_numpy(X.rows(s))), op, dirn)
        t = (q * sc.embed(torch.from_numpy(X.rows(d)))).sum(1)
        NB = max(1024, min(65536, int(1e8 // Q)))
        greater = torch.zeros(Q, dtype=torch.float64)
        ties = torch.zeros(Q, dtype=torch.float64)
        hi_t, lo_t = (t + EPS)[:, None], (t - EPS)[:, None]
        for b0 in range(0, X.n, NB):
            b1 = min(X.n, b0 + NB)
            S = q @ sc.embed(torch.from_numpy(X.rows(slice(b0, b1)))).T
            lo, hi = np.searchsorted(en, [b0, b1])
            if hi > lo:
                S[torch.from_numpy(eq[lo:hi]), torch.from_numpy(en[lo:hi] - b0)] = float("-inf")
            g = S > hi_t
            greater += g.sum(1)
            ties += ((S >= lo_t) & ~g).sum(1)
    rank = 1.0 + greater.numpy() + 0.5 * ties.numpy()
    return rank, t.numpy()


def measures(rank):
    if rank.size == 0:
        return {"n": 0}
    return {"n": int(rank.size), "mrr": round(100 * float((1.0 / rank).mean()), 2),
            "h10": round(100 * float((rank <= 10).mean()), 2), "h100": round(100 * float((rank <= 100).mean()), 2)}


def per_group(rank, smp, names):
    out = {"pooled": measures(rank)}
    for g, nm in enumerate(names):
        out[nm] = measures(rank[smp["grp"] == g])
    return out


def paired(x, y):
    """Mean of x - y (points) over the pooled sample, its paired 95% interval (2,000 resamples, seed 0) and the call."""
    import zap1 as Z
    n = x.size
    dlt, lo, hi = Z.boot_diff(x, y, np.arange(n), n, BOOT, SEED)
    return {"diff": Z.pct(dlt), "lo": Z.pct(lo), "hi": Z.pct(hi), "call": Z.call(lo, hi)}


# ── the fit (section 7.5) ───────────────────────────────────────────────────


def true_mask(G, s_rows, lab_rows, cand):
    """[B, C]: candidate c is a true tail of row i's source under row i's label, or is the source itself."""
    B, C, n = s_rows.size, cand.size, G.n
    M = np.zeros((B, C), bool)
    ss, cc = s_rows[:, None], cand[None, :]
    for L in np.unique(lab_rows[lab_rows < G.nR]):
        I = np.flatnonzero(lab_rows == L)
        k = np.minimum(ss[I], cc) * n + np.maximum(ss[I], cc)
        M[I] = member(k.ravel(), G.r_keys[int(L)]).reshape(I.size, C)
    I = np.flatnonzero(lab_rows >= G.nR)
    if I.size:
        k = (ss[I] * n + cc) * G.RS + (lab_rows[I] - G.nR)[:, None]
        M[I] = member(k.ravel(), G.s_key).reshape(I.size, C)
    M |= cc == ss
    return M


def take(groups, split, pred, k, rng):
    s, d, lab, grp = [], [], [], []
    for g, G_ in enumerate(groups):
        a, b, c = G_.sample(split, pred, k, rng)
        s.append(a)
        d.append(b)
        lab.append(c)
        grp.append(np.full(a.size, g, np.int64))
    cat = lambda x: np.concatenate(x).astype(np.int64)  # noqa: E731
    return {"s": cat(s), "d": cat(d), "lab": cat(lab), "grp": cat(grp)}


def fit(torch, form, G, X, split, groups, op_of, ops, dev, dev_ex, cos_dev, cfg, gates):
    D = X.D
    Dp = D if form == "rotation_D" else D + cfg["extra"]
    model = make_model(torch, D, Dp, len(ops), form)
    train_nodes = np.flatnonzero(split == TRAIN)
    hist, t0 = [], time.time()
    r0, t0s = rank_read(torch, model, X, dev, dev_ex, op_of)
    g0 = {"max_score_diff": float(np.abs(t0s - cos_dev[1]).max()) if t0s.size else 0.0,
          "mrr_diff": float(abs((1.0 / r0).mean() - (1.0 / cos_dev[0]).mean())) if r0.size else 0.0}
    g0["verdict"] = "PASS" if g0["max_score_diff"] <= G0_SCORE and g0["mrr_diff"] <= G0_MRR else "FAIL"
    gates[f"G-0 {form}"] = g0
    if g0["verdict"] != "PASS":
        stop(f"gate G-0 ({form}): {g0}")
    best = {"epoch": 0, "dev_mrr": measures(r0).get("mrr", 0.0),
            "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    hist.append({"epoch": 0, "dev_mrr": best["dev_mrr"], "seconds": round(time.time() - t0, 1)})
    log(f"{form}: D' {Dp}, {len(ops)} operators; epoch 0 dev MRR {best['dev_mrr']} (= cosine)")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    share = cfg["epoch"] // max(len(groups), 1)
    gs_ok, since = True, 0
    for ep in range(1, cfg["max_epochs"] + 1):
        te = time.time()
        rng = np.random.default_rng(ep)
        tr = take(groups, split, is_train, share, rng)
        m = tr["s"].size
        gs_ok &= bool(((split[tr["s"]] == TRAIN) & (split[tr["d"]] == TRAIN)).all())
        perm = rng.permutation(m)
        tot, nb = 0.0, 0
        model.train()
        for b0 in range(0, m, cfg["batch"]):
            I = perm[b0:b0 + cfg["batch"]]
            s, d, lab = tr["s"][I], tr["d"][I], tr["lab"][I]
            neg = train_nodes[rng.integers(0, train_nodes.size, cfg["neg"])]
            cand = np.concatenate([d, neg])
            mask = true_mask(G, s, lab, cand)
            mask[np.arange(I.size), np.arange(I.size)] = False
            xs = model.embed(torch.from_numpy(X.rows(s)))
            xc = model.embed(torch.from_numpy(X.rows(cand)))
            q = model.query(xs, torch.from_numpy(np.maximum(op_of[lab], 0)), torch.from_numpy(lab >= G.nR))
            logits = (q @ xc.T) / cfg["tau"]
            logits = logits.masked_fill(torch.from_numpy(mask), float("-inf"))
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(I.size))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        model.eval()
        r, _ = rank_read(torch, model, X, dev, dev_ex, op_of)
        mrr = measures(r).get("mrr", 0.0)
        hist.append({"epoch": ep, "edges": int(m), "loss": round(tot / max(nb, 1), 4), "dev_mrr": mrr,
                     "seconds": round(time.time() - te, 1)})
        log(f"{form}: epoch {ep}: {m:,} edges, loss {tot / max(nb, 1):.4f}, dev MRR {mrr} ({time.time() - te:.0f} s)")
        if mrr > best["dev_mrr"]:
            best = {"epoch": ep, "dev_mrr": mrr, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            since = 0
        else:
            since += 1
            if since >= cfg["patience"]:
                break
    gates[f"G-S {form} training"] = "PASS" if gs_ok else "FAIL"
    if not gs_ok:
        stop(f"gate G-S ({form}): a training edge touches a test or dev node")
    model.load_state_dict(best["state"])
    model.eval()
    rec = {"D_prime": Dp, "history": hist, "best_epoch": best["epoch"], "best_dev_mrr": best["dev_mrr"],
           "seconds": round(time.time() - t0, 1)}
    return model, rec, {k: v.numpy() for k, v in best["state"].items()}


# ── the datasets ────────────────────────────────────────────────────────────


def load_six(d, arm, host):
    import u1b
    import zul1 as UL
    import zul2 as U2
    served, freeze = u1b.package_root(host)
    man = json.loads((served / d / "DATASET.json").read_text(encoding="utf-8"))
    n, nsha = int(man["nodes"]["n"]), man["nodes"]["sha256"]
    F, sh, slab = UL.families(served, d, n, nsha, freeze)
    fp = json.loads((UL.OUT / d / "grade.json").read_text(encoding="utf-8"))["family_pairs"]
    ge = {}
    for fam in ("W2_S", "CN_all", "F0T") + (("S",) if d in UL.HAS_S else ()):
        ge[fam] = {"pairs": int(F[fam].size), "record": int(fp[fam])}
        if F[fam].size != fp[fam]:
            stop(f"gate G-E: {d}: {fam} has {F[fam].size} pairs, UL1's grade {fp[fam]}")
    for k in ("N_sp", "K", "S"):
        F.pop(k, None)
    R = [("R.mention", F.pop("W2_S")), ("R.name_comention", F["CN_all"]),
         ("R.title_comention", np.setdiff1d(F.pop("F0T"), F.pop("CN_all"), assume_unique=True))]
    S, vocab = None, ()
    if d in UL.HAS_S:
        f = json.loads((served / d / "graph" / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))["families"]["structural"]
        p = served / d / "graph" / Path(f["file"]).name                     # zul1.s_labels checked its sha256 above
        vocab = [str(v) for v in (f.get("relation_vocabulary") or [])]
        RS = max(len(vocab), 1)
        with np.load(p) as z:
            a, b, r = z["src"].astype(np.int64), z["dst"].astype(np.int64), z["rel"].astype(np.int64)
        m = a != b
        S = np.unique((a[m] * n + b[m]) * RS + r[m])
        del a, b, r, m
        k = S // RS
        a, b = ends(k, n)
        und = np.unique((np.minimum(a, b) * n + np.maximum(a, b)) * RS + (S - k * RS))
        del a, b, k
        if not np.array_equal(und, slab[0] * RS + slab[1]):
            stop(f"gate G-E: {d}: the directed S read is not zul1.s_labels' (pair, relation) rows")
        ge["S directed"] = {"edges": int(S.size), "pair_relation_rows": int(und.size), "verdict": "PASS"}
        del und
    G = Graph(d, n, R, S, vocab)
    V, vrec = U2.load_rows(served, arm, d, n)
    inputs = {"freeze_RECORD_SHA256": freeze, "nodes_sha256": nsha, **sh, "vectors": vrec}
    return G, V, inputs, ge


def load_mhr(arm):
    import zmhr as M
    root = M.mhr_root()
    M.point_zenc()
    n = len(M.nodes_of(root))
    rec = json.loads((M.OUT / "links.json").read_text(encoding="utf-8"))
    if rec["nodes"] != n:
        stop(f"gate G-E: links.json has {rec['nodes']} nodes, the package {n}")
    K, ge = {}, {}
    for k in ("F0", "F0T", "W2_S"):
        p = M.OUT / "links" / "rule" / f"{k}.npz"
        if sha_file(p) != rec["files"][k]:
            stop(f"gate G-E: {p.name} is not links.json's")
        K[k] = np.load(p)["keys"].astype(np.int64)
        a, b = ends(K[k], n)
        if not ((a < b).all() and (np.diff(K[k]) > 0).all()):
            stop(f"gate G-E: {k}'s keys are not sorted unique pairs (min, max)")
    want = {"F0": rec["F0"], "F0T": rec["F0T"], "W2_S": rec["W2_S"]["pairs"]}
    for k, v in want.items():
        ge[k] = {"pairs": int(K[k].size), "record": int(v)}
        if K[k].size != v:
            stop(f"gate G-E: {k} has {K[k].size} pairs, links.json {v}")
    R = [("R.mention", K["W2_S"]), ("R.name_comention", K["F0"]),
         ("R.title_comention", np.setdiff1d(K["F0T"], K["F0"], assume_unique=True))]
    G = Graph(M.DS, n, R)
    paths = E.vec_shards(root, arm, M.DS, "docs")
    V = np.concatenate([np.load(p) for p in paths])
    if V.shape[0] != n:
        stop(f"{arm}: {V.shape[0]} vector rows for {n} nodes")
    vrec = {"shards": len(paths), "dim": int(V.shape[1]), "dtype": str(V.dtype)}
    if arm != "frozen":
        vrec["manifest_sha256"] = E.sha256(E.vec_dir(root, arm, M.DS, "docs") / "manifest.json")
    inputs = {"links_json_sha256": sha_file(M.OUT / "links.json"), **{f"{k}_npz_sha256": rec["files"][k] for k in K},
              "vectors": vrec}
    return G, V, inputs, ge


# ── r1 ──────────────────────────────────────────────────────────────────────


def run(torch, G, X, cfg, forms, gates):
    t0 = time.time()
    split = node_split(G.name, G.n)
    log(f"{G.name}: split test {int((split == TEST).sum()):,}, dev {int((split == DEV).sum()):,}, "
        f"train {int((split == TRAIN).sum()):,} ({time.time() - t0:.0f} s)")
    labs, ops, op_of, groups = plan(G, split, cfg)
    names = [g.name for g in groups]
    log(f"{G.name}: {len(labs)} labels, {len(ops)} operators, read groups {names}")
    if not groups:
        stop("no read label")
    rng = np.random.default_rng(SEED)
    test = take(groups, split, is_test, cfg["n_test"], rng)
    dev = take(groups, split, is_dev, cfg["n_dev"], rng)
    for smp in (test, dev):
        smp["nR"] = G.nR
    gs = {"test sources are test nodes": bool((split[test["s"]] == TEST).all()),
          "dev sources are dev nodes, targets not test": bool(((split[dev["s"]] == DEV) & (split[dev["d"]] != TEST)).all())}
    gates["G-S samples"] = "PASS" if all(gs.values()) else "FAIL"
    if not all(gs.values()):
        stop(f"gate G-S: {gs}")
    test_ex, dev_ex = exclusions(G, test), exclusions(G, dev)
    log(f"{G.name}: test {test['s'].size:,} edges ({test_ex[0].size:,} left out), dev {dev['s'].size:,} "
        f"({dev_ex[0].size:,}) ({time.time() - t0:.0f} s)")
    cos = Cosine()
    cos_dev = rank_read(torch, cos, X, dev, dev_ex, op_of)
    cos_test, _ = rank_read(torch, cos, X, test, test_ex, op_of)
    log(f"{G.name}: cosine test {measures(cos_test)}, dev {measures(cos_dev[0])} ({time.time() - t0:.0f} s)")
    read = {"cosine": per_group(cos_test, test, names)}
    ranks = {"cosine": cos_test}
    fits, params = {}, {}
    for form in forms:
        model, rec, st = fit(torch, form, G, X, split, groups, op_of, ops, dev, dev_ex, cos_dev, cfg, gates)
        r, _ = rank_read(torch, model, X, test, test_ex, op_of)
        read[form], ranks[form], fits[form] = per_group(r, test, names), r, rec
        params.update({f"{form}.{k}": v for k, v in st.items()})
        log(f"{G.name}: {form} test {measures(r)} ({time.time() - t0:.0f} s)")
    rr = {k: 1.0 / v for k, v in ranks.items()}
    call = paired(rr["rotation"], rr["cosine"])
    call["call"] = {"ABOVE": "LEARNS", "LEVEL": "LEVEL", "BELOW": "LOSES"}[call["call"]]
    beside = {f"{f} - cosine": paired(rr[f], rr["cosine"]) for f in forms if f != "rotation"}
    if "rotation_D" in rr:
        beside["rotation - rotation_D (X3)"] = paired(rr["rotation"], rr["rotation_D"])
    per_label = {nm: paired(rr["rotation"][test["grp"] == g], rr["cosine"][test["grp"] == g])
                 for g, nm in enumerate(names) if (test["grp"] == g).sum() > 1}
    out = {"nodes": int(G.n), "split": {k: int((split == v).sum()) for k, v in (("test", TEST), ("dev", DEV), ("train", TRAIN))},
           "labels": label_summary(labs),
           "operators": [o["name"] for o in ops], "read_groups": {nm: {"test": int((test["grp"] == g).sum()),
                                                                       "dev": int((dev["grp"] == g).sum())} for g, nm in enumerate(names)},
           "excluded_pairs": {"test": int(test_ex[0].size), "dev": int(dev_ex[0].size)},
           "call": call, "per_label": per_label, "beside": beside, "read": read, "fits": fits}
    arrays = {"test_s": test["s"], "test_d": test["d"], "test_lab": test["lab"], "test_grp": test["grp"],
              **{f"rank_{k}": v for k, v in ranks.items()}, **params}
    return out, arrays


def cmd_r1(a):
    torch = torch_()
    t0 = time.time()
    cfg = dict(CFG)
    if a.smoke:
        cfg.update(SMOKE)
        if not a.out:
            stop("--smoke writes only under --out")
    if a.ds == "multihoprag":
        G, V, inputs, ge = load_mhr(a.arm)
    else:
        G, V, inputs, ge = load_six(a.ds, a.arm, a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}}
    log(f"{a.ds} {a.arm}: {G.n:,} nodes, R {[k.size for k in G.r_keys]}, S {G.s_key.size:,}, rows {V.shape} {V.dtype} "
        f"({time.time() - t0:.0f} s)")
    X = Vecs(V)
    forms = ("rotation", "translation") + (("rotation_D", "map_only") if a.ds in SMALL else ())
    out, arrays = run(torch, G, X, cfg, forms, gates)
    rec = {"stage": "SEM2 R1", "declared_in": DECLARED, "dataset": a.ds, "arm": a.arm, "dim": int(X.D),
           "worst_norm_offset": round(X.worst, 6), "config": cfg, "smoke": bool(a.smoke), "gates": gates, **out,
           "inputs": inputs, "threads": E.THREADS, "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": sha_file(Path(__file__))}
    od = Path(a.out) if a.out else OUT / a.ds
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"{a.arm}.npz", **arrays)
    rec["npz_sha256"] = sha_file(od / f"{a.arm}.npz")
    wjson(od / f"{a.arm}.json", rec)
    c = rec["call"]
    log(f"{a.ds} {a.arm}: rotation - cosine MRR {c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}; "
        f"written {od / (a.arm + '.json')} ({rec['seconds']} s)")
    return 0


# ── report (section 7.7) ────────────────────────────────────────────────────


def cmd_report(a):
    rows = {}
    for d in SEVEN:
        for arm in ARMS:
            p = OUT / d / f"{arm}.json"
            if p.is_file():
                r = json.loads(p.read_text(encoding="utf-8"))
                if r.get("smoke"):
                    stop(f"{p} is a smoke run")
                rows[(d, arm)] = r
    per_arm = {}
    for arm in ARMS:
        calls = {d: rows[(d, arm)]["call"]["call"] for d in SEVEN if (d, arm) in rows}
        missing = [d for d in SEVEN if d not in calls]
        n_learn = sum(v == "LEARNS" for v in calls.values())
        if missing:
            verdict = "INCOMPLETE"
        else:
            verdict = "HOLDS" if n_learn == len(SEVEN) else ("PARTIAL" if n_learn else "FAILS")
        per_arm[arm] = {"verdict": verdict, "learns": n_learn, "calls": calls, "missing": missing}
    c = lambda d, arm: rows[(d, arm)]["call"] if (d, arm) in rows else None  # noqa: E731
    x = {}
    got = [c(d, "e3v") for d in ("metaqa", "webqsp")]
    x["X1"] = None if None in got else all(g["call"] == "LEARNS" for g in got)
    x["X2"] = None if c("musique", "e3v") is None else c("musique", "e3v")["call"] == "LEARNS"
    x3 = [rows[k]["beside"].get("rotation - rotation_D (X3)") for k in rows if k[0] in SMALL]
    x["X3"] = None if not x3 or None in x3 else all(v["call"] != "BELOW" for v in x3)
    both = [d for d in SEVEN if (d, "e3v") in rows and (d, "frozen") in rows]
    below = sum(c(d, "frozen")["diff"] < c(d, "e3v")["diff"] for d in both)
    x["X4"] = None if len(both) < len(SEVEN) else below > len(both) / 2
    rep = {"stage": "SEM2 R1 report", "declared_in": DECLARED, "per_arm": per_arm, "expectations": x,
           "x4_frozen_below_e3v": f"{below} of {len(both)}",
           "rows": {f"{d} {arm}": {"call": r["call"], "cosine": r["read"]["cosine"]["pooled"], "rotation": r["read"]["rotation"]["pooled"],
                                   "beside": r["beside"], "per_label": r["per_label"], "sha256": sha_file(OUT / d / f"{arm}.json")}
                    for (d, arm), r in rows.items()},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": sha_file(Path(__file__))}
    wjson(OUT / "report.json", rep)
    L = ["# SEM2 R1: link recovery under learned operators", "",
         f"Declared in {DECLARED}. Written by zsem2.py report at {rep['utc']}. MRR in points; intervals are paired 95% "
         "bootstrap intervals over the pooled test edges.", "",
         "| dataset | arm | cosine MRR | rotation MRR | rotation - cosine | call | translation - cosine |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for d in SEVEN:
        for arm in ARMS:
            if (d, arm) not in rows:
                L.append(f"| {d} | {arm} | (not run) | | | | |")
                continue
            r = rows[(d, arm)]
            cc, t = r["call"], r["beside"].get("translation - cosine", {})
            L.append(f"| {d} | {arm} | {r['read']['cosine']['pooled'].get('mrr')} | {r['read']['rotation']['pooled'].get('mrr')} | "
                     f"{cc['diff']:+.2f} [{cc['lo']:+.2f}, {cc['hi']:+.2f}] | {cc['call']} | "
                     f"{t.get('diff', 0):+.2f} [{t.get('lo', 0):+.2f}, {t.get('hi', 0):+.2f}] {t.get('call', '')} |")
    L += ["", "**R1 per arm:** " + "; ".join(f"{arm} {v['verdict']} ({v['learns']} of {len(SEVEN)} LEARN)" for arm, v in per_arm.items()), "",
          "**Expectations:** " + "; ".join(f"{k} {'held' if v else ('failed' if v is False else 'not readable yet')}" for k, v in x.items()), ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    log(f"report: {per_arm}; {x}")
    return 0


# ── self-test (section 7.8) ─────────────────────────────────────────────────


def rotate_np(V, th):
    m, D = V.shape
    x = V.reshape(m, D // 2, 2)
    c, s = np.cos(th), np.sin(th)
    return np.stack([x[..., 0] * c - x[..., 1] * s, x[..., 0] * s + x[..., 1] * c], -1).reshape(m, D)


def brute_rank(sc, torch, X, smp, i, G, op_of):
    """One query's filtered rank by a full sort (the read's check)."""
    s, d, L = int(smp["s"][i]), int(smp["d"][i]), int(smp["lab"][i])
    with torch.no_grad():
        allx = sc.embed(torch.from_numpy(X.rows(slice(0, X.n))))
        q = sc.query(sc.embed(torch.from_numpy(X.rows(np.asarray([s])))), torch.tensor([max(op_of[L], 0)]),
                     torch.tensor([L >= G.nR]))
        sc_all = (allx @ q[0]).numpy().astype(np.float64)
        t = float((q[0] * sc.embed(torch.from_numpy(X.rows(np.asarray([d]))))[0]).sum())
    drop = {s}
    if L < G.nR:
        a, b = ends(G.r_keys[L], G.n)
        drop |= set(b[a == s].tolist()) | set(a[b == s].tolist())
    else:
        m = (G.s_src == s) & (G.s_rel == L - G.nR)
        drop |= set(G.s_dst[m].tolist())
    keep = np.ones(X.n, bool)
    keep[list(drop)] = False
    v = sc_all[keep]
    return 1 + float((v > t + EPS).sum()) + 0.5 * float(((v >= t - EPS) & (v <= t + EPS)).sum())


def selftest(a):
    torch = torch_()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, D = 4000, 32
    V = rng.standard_normal((n, D)).astype(np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    th = rng.uniform(1.0, 2.0, D // 2) * rng.choice([-1.0, 1.0], D // 2)        # cosine near chance on the directed label
    src, dst = np.arange(0, 1000), np.arange(1000, 2000)
    V[dst] = unit(rotate_np(V[src], th) + 0.2 * rng.standard_normal((1000, D)) / np.sqrt(D))
    w = np.repeat(rng.uniform(-1, 1, D // 2), 2)                                 # plane weights on the undirected one
    aa, bb = np.arange(2000, 2600), np.arange(2600, 3200)
    V[bb] = unit(V[aa] * w + 0.2 * rng.standard_normal((600, D)) / np.sqrt(D))
    V = V.astype(np.float32)
    rk = np.unique(np.minimum(aa, bb).astype(np.int64) * n + np.maximum(aa, bb))
    S = np.unique(src.astype(np.int64) * n + dst)
    G = Graph("toy", n, [("R.toy", rk)], S, ["toy_rel"])
    X = Vecs(V.astype(np.float16))
    cfg = dict(CFG, batch=256, neg=512, epoch=40_000, max_epochs=25, patience=4, lr=0.02, n_test=300, n_dev=200, extra=8, min_op=100)
    gates = {}
    out, arrays = run(torch, G, X, cfg, ("rotation", "translation", "rotation_D", "map_only"), gates)
    # the read's ranks against a full sort, on the trained rotation and on the cosine
    split = node_split("toy", n)
    labs, ops, op_of, groups = plan(G, split, cfg)
    test = take(groups, split, is_test, cfg["n_test"], np.random.default_rng(SEED))
    test["nR"] = G.nR
    if not np.array_equal(test["s"], arrays["test_s"]):
        stop("self-test: the test sample is not reproducible")
    cos = Cosine()
    ex = exclusions(G, test)
    r, _ = rank_read(torch, cos, X, test, ex, op_of)
    chk = [abs(brute_rank(cos, torch, X, test, i, G, op_of) - r[i]) for i in range(0, test["s"].size, 25)]
    if max(chk) > 0:
        stop(f"self-test: the read's ranks differ from a full sort by up to {max(chk)}")
    c = out["call"]
    rot, cs = out["read"]["rotation"], out["read"]["cosine"]
    log(f"self-test: cosine {cs['pooled']}, rotation {rot['pooled']}; per group: "
        f"{ {k: (cs[k].get('mrr'), rot[k].get('mrr')) for k in out['read_groups']} }; call {c}; gates {gates}")
    ok = (c["call"] == "LEARNS" and c["diff"] > 30 and rot["S.toy_rel"]["mrr"] > cs["S.toy_rel"]["mrr"] + 40
          and rot["R.toy"]["mrr"] > cs["R.toy"]["mrr"] + 20 and all(v == "PASS" or (isinstance(v, dict) and v.get("verdict") == "PASS")
                                                                     for v in gates.values()))
    if not ok:
        stop("self-test FAILED")
    log(f"self-test PASS: {len(chk)} ranks equal a full sort; rotation lifts MRR by {c['diff']:+.2f} points on planted operators")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "r1", "report"))
    p.add_argument("--ds", choices=SEVEN)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "report":
        return cmd_report(a)
    if not (a.ds and a.arm):
        stop("r1 needs --ds and --arm")
    return cmd_r1(a)


if __name__ == "__main__":
    sys.exit(main())
