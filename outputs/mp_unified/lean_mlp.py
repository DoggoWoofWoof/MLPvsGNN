"""Design look (untracked; not a result and not filed): a lean, cost-aware MLP on the look's compiled inputs.

The user (3 Oct): the MLP must be faster than the GNN in every setting, so find which feature families it needs, drop
the rest, and make the extraction cheap at zero-shot inference. The twin and the GNN share one query-local compile
(129 columns in 13 families, plus the 1536-wide embeddings), so an MLP that reads all of it can only win by the
forward pass. This look fits MLPs from scratch on a carve's look rows and asks which families pay for their cost.

Blocks. The compiled ones are look_x_six's column_blocks, with two split by what they depend on:
    rank       dense_rr, splade_rr, rrf, agreement, is_seed       (the retrievers' lists; always kept)
    dense_cos  dense_cos                                          (needs the 1536-wide embedding gather)
    seed_e     cos_v_seedproto, max_cos_v_seed                    (embeddings only, no edge)
    seed_r     cos_v_reachproto, has_reach_seed, cos_v_seedproto_h2, has_seed_h2   (needs the FULL view's reach)
    and topo_STRUCT/NER/KNN/FULL, nbr_agg, gcs, typed_rel, depth_STRUCT/FULL, typed_v2, ordered as they are.
The lean ones are computed here from what a zero-shot deployment can index cheaply, the node's fixed 128-wide random
projection of its embedding (the look's proj: 256 bytes a node in float16 against 6 KB for the 1536-wide float32 row):
    SEM    the normalised projected query times the normalised node projection (128) and their sum, the cosine
    SEMB   learned: (q_emb U) * (n_proj V), 64 wide, U 1536x64 and V 128x64; the node side is a fixed function of the
           node, so a deployment precomputes n_proj V into the index and the query pays one 1536x64 product
    SEED   projected cosine to the seeds' mean, max over the seeds, max over the rank-1 (bucket-0) seeds
    WALK   seed walk counts over STRUCTURAL edges only (hops 1-3: undirected, forward, backward; the rank-1 seeds'
           hops 1-2), log degree, the first hop that reaches the node, is-seed: fixed counts, no PPR
    NBR    over structural edges: mean neighbour cosine, the cosine of the neighbour mean, max neighbour cosine, the
           neighbour mean's norm (cohesion), has-neighbour
No lean block learns anything over edges: WALK and NBR are fixed counts and fixed means of fixed projections, so a
model on them is non-MP in the project's sense (no learned h_v update); SEMB is learned per node, never over edges.

Model: per node [raw | within-query z-score | presence flag] per block -> Linear(H) GELU Linear(H) GELU -> score,
added to the z-scored rrf column (the twin's own base), the output layer zero-initialised. Training under --mode drop
draws a keep rate r ~ U(0.15, 1) per row and keeps each block with probability r, so one fit is read with any block
switched off. Loss: the listwise loss (log-softmax over the pool, mean over the row's in-pool golds).

Selection: on the SELECT carve, greedy backward elimination, each step dropping the block with the smallest loss in
mean(R@5, FC@5) per approximate compile millisecond saved (a block whose removal does not lower it goes first). The
cost model maps blocks to the fast compile's per-group p50 (crag_profile/<ds>_fast.json) with their dependencies;
splits inside a group (the edge build by family edge share, the retrieval lap 25/75 between the rank lists and the
embedding gather, D, depth_basis and typed_basis halves) are assumptions and flagged as such; the lean blocks are
timed here in numpy (pessimistic against numba). The cheapest set within --tol points of the full one on select is
refitted without dropout and read once on each read carve against the six twin and GNN scores stored in the look.

    python outputs/mp_unified/lean_mlp.py --train 2wiki=x4 --select 2wiki=select --read 2wiki=x1,hotpotqa=x1 \
        --out outputs/mp_unified/lean/l1-2w.json [--epochs 12] [--limit 400] [--threads 2]
    python outputs/mp_unified/lean_mlp.py --selftest
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

HERE = Path(__file__).resolve().parent
LOOK = HERE / "look"
PROFILE = Path(os.environ.get("LEAN_PROFILE_DIR", str(HERE / "lean" / "crag_profile")))
PROJ_DIM, PROJ_SEED = 128, 20261001
SEMB_DIM = 64
METRICS = ("recall@5", "full_coverage@5", "hit@1")
SPLIT = {"rank": ["dense_rr", "splade_rr", "rrf", "agreement", "is_seed"], "dense_cos": ["dense_cos"],
         "seed_e": ["cos_v_seedproto", "max_cos_v_seed"],
         "seed_r": ["cos_v_reachproto", "has_reach_seed", "cos_v_seedproto_h2", "has_seed_h2"]}
WHOLE = ("topo_STRUCT", "topo_NER", "topo_KNN", "topo_FULL", "nbr_agg", "gcs", "typed_rel", "depth_STRUCT", "depth_FULL",
         "typed_v2", "ordered")
COMPILED = ("rank", "dense_cos", "topo_STRUCT", "topo_NER", "topo_KNN", "topo_FULL", "nbr_agg", "gcs", "seed_e", "seed_r",
            "typed_rel", "depth_STRUCT", "depth_FULL", "typed_v2", "ordered")
LEAN = ("SEM", "SEMB", "SEED", "WALK", "NBR")
STORED_LEAN = ("SEED", "WALK", "NBR")
LEAN_W = {"SEM": PROJ_DIM + 1, "SEMB": SEMB_DIM, "SEED": 3, "WALK": 16, "NBR": 5}
NEVER = ("rank",)
SCORE_COL = {"twin0": 0, "gnn0": 3}
BOOT = 1000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def projection():
    return (np.random.default_rng(PROJ_SEED).standard_normal((1536, PROJ_DIM)) / math.sqrt(PROJ_DIM)).astype(np.float32)


# ── lean blocks, per query ───────────────────────────────────────────────────


def lean_query(n, proj, q_emb, R, seeds, buckets, eu, ev, efam, efwd, ebwd, timer):
    """The stored lean blocks of one query (float32) and the query's normalised projection; each block timed (s)."""
    t = time.perf_counter()
    P = proj.astype(np.float32)
    pn = np.linalg.norm(P, axis=1)
    Pn = P / np.maximum(pn, 1e-12)[:, None]
    q = q_emb.astype(np.float32) @ R
    q /= max(float(np.linalg.norm(q)), 1e-12)
    cos = Pn @ q
    _sem = np.concatenate([Pn * q[None, :], cos[:, None]], axis=1)   # timed here, built per batch in training
    t1 = time.perf_counter()
    timer["SEM"] += t1 - t
    valid = seeds >= 0
    S, Bk = seeds[valid].astype(np.int64), buckets[valid]
    seed = np.zeros((n, 3), np.float32)
    if S.size:
        Ps = Pn[S]
        m = Ps.mean(0)
        m /= max(float(np.linalg.norm(m)), 1e-12)
        G = Pn @ Ps.T
        seed[:, 0] = Pn @ m
        seed[:, 1] = G.max(1)
        if bool((Bk == 0).any()):
            seed[:, 2] = G[:, Bk == 0].max(1)
    t2 = time.perf_counter()
    timer["SEED"] += t2 - t1
    st = efam == 0
    u, v = eu[st].astype(np.int64), ev[st].astype(np.int64)
    fw, bw = efwd[st].astype(bool), ebwd[st].astype(bool)
    walk = np.zeros((n, 16), np.float32)
    s0 = np.zeros(n, np.float64)
    s0[S] = 1.0
    sb = np.zeros(n, np.float64)
    sb[S[Bk == 0]] = 1.0
    col = 0
    for mask in (None, fw, bw):
        uu, vv = (u, v) if mask is None else (u[mask], v[mask])
        c = s0
        for _h in range(3):
            c = np.bincount(vv, weights=c[uu], minlength=n)
            walk[:, col] = np.log1p(c)
            col += 1
    c = sb
    for _h in range(2):
        c = np.bincount(v, weights=c[u], minlength=n)
        walk[:, col] = np.log1p(c)
        col += 1
    deg = np.bincount(v, minlength=n).astype(np.float32)
    walk[:, col] = np.log1p(deg)
    col += 1
    first = np.full(n, 3, np.int64)
    for h in (2, 1, 0):
        first[walk[:, h] > 0] = h
    first[S] = -1
    for h in range(3):
        walk[:, col + h] = first == h
    walk[:, col + 3] = first == -1
    t3 = time.perf_counter()
    timer["WALK"] += t3 - t2
    nbr = np.zeros((n, 5), np.float32)
    if u.size:
        order = np.argsort(v, kind="stable")
        vs, us = v[order], u[order]
        starts = np.flatnonzero(np.r_[True, vs[1:] != vs[:-1]])
        rows = vs[starts]
        acc = np.add.reduceat(Pn[us], starts, axis=0)
        cu = cos[us]
        d = deg[rows]
        mean = acc / d[:, None]
        mn = np.linalg.norm(mean, axis=1)
        nbr[rows, 0] = np.add.reduceat(cu, starts) / d
        nbr[rows, 1] = (mean @ q) / np.maximum(mn, 1e-12)
        nbr[rows, 2] = np.maximum.reduceat(cu, starts)
        nbr[rows, 3] = mn
        nbr[rows, 4] = 1.0
    timer["NBR"] += time.perf_counter() - t3
    return {"SEED": seed, "WALK": walk, "NBR": nbr}, q, Pn


# ── loading ──────────────────────────────────────────────────────────────────


class Carve:
    """One carve's rows: node-major float16 blocks, query offsets, gold flags, the stored six-pair scores."""

    def __init__(self, ds, carve, limit=None, root=LOOK):
        d = Path(root) / ds / carve
        recs = sorted(d.glob("record*.json"))
        if not recs:
            raise SystemExit(f"{d}: no look record")
        rec = json.loads(recs[0].read_text(encoding="utf-8"))
        self.ds, self.carve = ds, carve
        self.columns = rec["columns"]
        ci = {c: i for i, c in enumerate(self.columns)}
        cb = rec["column_blocks"]
        self.blocks_idx = {b: np.asarray([ci[c] for c in cols], np.int64) for b, cols in SPLIT.items()}
        self.blocks_idx.update({b: np.asarray(cb[b], np.int64) for b in WHOLE if b in cb})
        R = projection()
        xs, proj, qp, qe, gold, score, qn, gt = [], [], [], [], [], [], [], []
        lean = {b: [] for b in STORED_LEAN}
        self.timer = {b: 0.0 for b in LEAN if b != "SEMB"}
        per_q_ms = {b: [] for b in self.timer}
        fam = np.zeros(3, np.int64)
        rows = 0
        self.chunks_read = 0
        self.n_chunks = int(rec["n_chunks"])
        self.carve_queries = int(rec["carve_queries"])
        listed = set()
        for r_ in recs:
            listed |= set(json.loads(r_.read_text(encoding="utf-8"))["chunks"])
        for ch in sorted((d / "chunks").glob("c*.npz")):
            if int(ch.stem[1:]) not in listed:
                continue                              # a chunk file whose shard has not filed its record yet
            self.chunks_read += 1
            z = np.load(ch)
            qps, qed = z["q_pool_size"], z["q_edges"]
            no = np.concatenate([[0], np.cumsum(qps)])
            eo = np.concatenate([[0], np.cumsum(qed)])
            X, PR, QE, G, SC = z["x"], z["proj"], z["q_emb"], z["is_gold"], z["score"]
            EU, EV, EF, EFW, EBW = z["e_u"], z["e_v"], z["e_fam"], z["e_fwd"], z["e_bwd"]
            SL, SB, GT = z["q_seed_local"], z["q_seed_bucket"], z["q_gold_total"]
            for i in range(qps.size):
                if limit is not None and rows >= limit:
                    break
                a, b = no[i], no[i + 1]
                ea, eb = eo[i], eo[i + 1]
                before = dict(self.timer)
                L, q, Pn = lean_query(int(qps[i]), PR[a:b], QE[i], R, SL[i], SB[i], EU[ea:eb], EV[ea:eb], EF[ea:eb], EFW[ea:eb],
                                      EBW[ea:eb], self.timer)
                for k in STORED_LEAN:
                    lean[k].append(L[k].astype(np.float16))
                for k in self.timer:
                    per_q_ms[k].append(1e3 * (self.timer[k] - before[k]))
                fam += np.bincount(EF[ea:eb], minlength=3)[:3]
                xs.append(X[a:b])
                proj.append(Pn.astype(np.float16))
                qp.append(q)
                qe.append(QE[i].astype(np.float16))
                gold.append(G[a:b])
                score.append(SC[a:b])
                qn.append(int(qps[i]))
                gt.append(int(GT[i]))
                rows += 1
            if limit is not None and rows >= limit:
                break
        self.x = np.concatenate(xs)
        self.proj = np.concatenate(proj)
        self.qp = np.stack(qp).astype(np.float32)
        self.q_emb = np.stack(qe)
        self.lean = {k: np.concatenate(v) for k, v in lean.items()}
        self.gold = np.concatenate(gold)
        self.score = np.concatenate(score)
        self.n = np.asarray(qn, np.int64)
        self.gold_total = np.asarray(gt, np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.n)])
        self.lean_ms = {k: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(np.mean(v))}
                        for k, v in per_q_ms.items()}
        self.fam_entries = [int(v) for v in fam]
        self.struct_share = float(fam[0] / max(fam.sum(), 1))
        self.rrf = self.columns.index("rrf")
        self.widths = {b: int(v.size) for b, v in self.blocks_idx.items()}
        self.widths.update(LEAN_W)
        self.rows = int(self.n.size)

    def block(self, b, idx):
        if b in STORED_LEAN:
            return self.lean[b][idx]
        return self.x[idx][:, self.blocks_idx[b]]


# ── model ────────────────────────────────────────────────────────────────────


def seg_zscore(x, nq, B, eps=1e-6):
    ones = torch.ones(nq.numel(), dtype=x.dtype)
    cnt = torch.zeros(B, dtype=x.dtype).index_add_(0, nq, ones).clamp_min(1.0).unsqueeze(1)
    mean = torch.zeros(B, x.shape[1], dtype=x.dtype).index_add_(0, nq, x) / cnt
    c = x - mean[nq]
    var = torch.zeros(B, x.shape[1], dtype=x.dtype).index_add_(0, nq, c * c) / cnt
    sd = var.sqrt()[nq]
    z = c / sd.clamp_min(eps)
    return torch.where(sd < eps, torch.zeros_like(z), z)


def seg_log_softmax(s, nq, B):
    mx = torch.full((B,), -float("inf"), dtype=s.dtype).scatter_reduce(0, nq, s, reduce="amax", include_self=True)
    sh = s - mx[nq]
    den = torch.zeros(B, dtype=s.dtype).index_add_(0, nq, sh.exp())
    return sh - den.clamp_min(1e-30).log()[nq]


class LeanMLP(nn.Module):
    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0):
        super().__init__()
        self.blocks = list(blocks)
        self.widths = {b: int(widths[b]) for b in self.blocks}
        self.in_w = sum(2 * w + 1 for w in self.widths.values())
        self.l1 = nn.Linear(self.in_w, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.base_w = nn.Parameter(torch.ones(1))
        self.drop = nn.Dropout(dropout)
        if "SEMB" in self.blocks:
            g = torch.Generator().manual_seed(seed + 7)
            self.U = nn.Parameter(torch.randn(1536, SEMB_DIM, generator=g) / math.sqrt(1536))
            self.V = nn.Parameter(torch.randn(PROJ_DIM, SEMB_DIM, generator=g) / math.sqrt(PROJ_DIM))

    def forward(self, feats, keep, nq, B, base_z):
        """feats[b]: (N, w) float32 raw ('SEMB' instead gives (q_emb (B, 1536), proj (N, 128))); keep: (B, n_blocks)."""
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            if b == "SEMB":
                qe, pr = feats[b]
                raw = (qe @ self.U)[nq] * (pr @ self.V)
            else:
                raw = feats[b]
            parts += [raw * m, seg_zscore(raw, nq, B) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * base_z + self.out(h).squeeze(-1)


def clean(a):
    return torch.from_numpy(np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))


def batch_of(carve, qs, blocks):
    idx = np.concatenate([np.arange(carve.off[q], carve.off[q + 1]) for q in qs])
    nq_np = np.repeat(np.arange(len(qs)), carve.n[qs])
    nq = torch.from_numpy(nq_np)
    feats = {}
    for b in blocks:
        if b == "SEM":
            pr = carve.proj[idx].astype(np.float32)
            prod = pr * carve.qp[qs][nq_np]
            feats[b] = torch.from_numpy(np.concatenate([prod, prod.sum(1, keepdims=True)], axis=1))
        elif b == "SEMB":
            feats[b] = (torch.from_numpy(carve.q_emb[qs].astype(np.float32)), torch.from_numpy(carve.proj[idx].astype(np.float32)))
        else:
            feats[b] = clean(carve.block(b, idx))
    rrf = torch.from_numpy(carve.x[idx, carve.rrf].astype(np.float32))
    base_z = seg_zscore(rrf.unsqueeze(1), nq, len(qs)).squeeze(1)
    gold = torch.from_numpy(carve.gold[idx])
    return feats, nq, base_z, gold, idx


def listwise(scores, gold, nq, B):
    lp = seg_log_softmax(scores, nq, B)
    g = gold.to(scores.dtype)
    ng = torch.zeros(B).index_add_(0, nq, g)
    pq = -torch.zeros(B).index_add_(0, nq, lp * g)
    has = ng > 0
    return (pq[has] / ng[has]).mean() if bool(has.any()) else scores.sum() * 0.0


def row_metrics(scores, gold, off, gold_total):
    """Per row R@5, FC@5, hit@1 over the row's gold_total, ties broken by pool position (the look's own metrics)."""
    out = np.zeros((off.size - 1, 3))
    for i in range(off.size - 1):
        s, g = scores[off[i]:off[i + 1]], gold[off[i]:off[i + 1]]
        gt = int(gold_total[i])
        if gt == 0:
            continue
        order = np.lexsort((np.arange(s.size), -s))
        top = int(g[order[:5]].sum())
        out[i] = (top / gt, float(top == gt), float(g[order[0]]))
    return out


def fit(train, select, blocks, mode, epochs, lr, seed, hidden):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: train[0].widths[b] for b in blocks}
    model = LeanMLP(blocks, widths, hidden, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    units = [(ci, q) for ci, c in enumerate(train) for q in range(c.rows)]
    best, best_state, curve = -1.0, None, []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(units))
        t0 = time.time()
        tot, nb = 0.0, 0
        for k in range(0, len(order), 32):
            chunk = [units[j] for j in order[k:k + 32]]
            for ci in sorted({c for c, _ in chunk}):
                qs = np.asarray([q for c, q in chunk if c == ci])
                feats, nq, base_z, gold, _ = batch_of(train[ci], qs, blocks)
                B = qs.size
                if mode == "drop":
                    r = rng.uniform(0.15, 1.0, size=(B, 1))
                    keep = (rng.uniform(size=(B, len(blocks))) < r).astype(np.float32)
                    keep[:, [blocks.index(b) for b in NEVER if b in blocks]] = 1.0
                else:
                    keep = np.ones((B, len(blocks)), np.float32)
                s = model(feats, torch.from_numpy(keep), nq, B, base_z)
                loss = listwise(s, gold, nq, B)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
        q = quality(model, select, blocks, {b: 1.0 for b in blocks})
        curve.append({"epoch": ep, "loss": tot / max(nb, 1), "select": q, "seconds": time.time() - t0})
        log(f"  ep {ep}: loss {tot / max(nb, 1):.4f} select R@5/FC@5/hit@1 {[round(v, 4) for v in q]} ({time.time() - t0:.0f}s)")
        score = 0.5 * (q[0] + q[1])
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, curve


@torch.no_grad()
def scores_of(model, c, blocks, keep_map):
    model.eval()
    s_all = np.zeros(int(c.off[-1]), np.float32)
    for k in range(0, c.rows, 64):
        qs = np.arange(k, min(k + 64, c.rows))
        feats, nq, base_z, _gold, idx = batch_of(c, qs, blocks)
        keep = torch.tensor([[keep_map.get(b, 1.0) for b in blocks]] * qs.size, dtype=torch.float32)
        s_all[idx] = model(feats, keep, nq, qs.size, base_z).numpy()
    return s_all


def quality(model, carves, blocks, keep_map):
    m = np.concatenate([row_metrics(scores_of(model, c, blocks, keep_map), c.gold, c.off, c.gold_total) for c in carves])
    return [float(v) for v in m.mean(0)]


def boot_diff(a, b, rng, n=BOOT):
    d = a - b
    idx = rng.integers(0, d.shape[0], size=(n, d.shape[0]))
    bs = d[idx].mean(1)
    return [[float(d.mean(0)[j]), [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]] for j in range(3)]


# ── cost model ───────────────────────────────────────────────────────────────


def profile(ds):
    p = PROFILE / f"{ds}_fast.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text(encoding="utf-8"))["summary"]["groups_p50"]
    return {k: float(v["fast"]) for k, v in g.items()}


STRUCT_USERS = {"topo_STRUCT", "topo_FULL", "nbr_agg", "gcs", "seed_r", "depth_STRUCT", "depth_FULL", "typed_rel", "typed_v2",
                "ordered", "WALK", "NBR"}
NK_USERS = {"topo_NER", "topo_KNN", "topo_FULL", "nbr_agg", "gcs", "seed_r", "depth_FULL"}
EMB_USERS = {"dense_cos", "nbr_agg", "seed_e", "seed_r", "depth_STRUCT", "depth_FULL"}
VIEW_USERS = {"STRUCT": {"topo_STRUCT", "gcs", "depth_STRUCT"}, "NER": {"topo_NER"}, "KNN": {"topo_KNN"},
              "FULL": {"topo_FULL", "gcs", "seed_r", "depth_FULL"}}


def cost_ms(S, prof, share_struct, lean_ms):
    """Approximate per-query compile ms of a block set (the fast compile's p50 groups, split as the docstring says)."""
    S = set(S)
    if prof is None:
        return float("nan")
    c = prof.get("retrieval", 0.0) * (0.25 + 0.75 * bool(S & EMB_USERS))
    c += prof.get("edges", 0.0) * (share_struct * bool(S & STRUCT_USERS) + (1 - share_struct) * bool(S & NK_USERS))
    c += prof.get("topology", 0.0) * sum(bool(S & users) for users in VIEW_USERS.values()) / 4.0
    c += prof.get("C", 0.0) * ("nbr_agg" in S) + prof.get("gcs", 0.0) * ("gcs" in S)
    c += prof.get("D", 0.0) * (0.5 * ("seed_e" in S) + 0.5 * ("seed_r" in S))
    c += prof.get("B", 0.0) * ("typed_rel" in S)
    c += prof.get("depth_basis", 0.0) * (0.5 * ("depth_STRUCT" in S) + 0.5 * ("depth_FULL" in S))
    c += prof.get("typed_basis", 0.0) * (0.5 * ("typed_v2" in S) + 0.5 * ("ordered" in S))
    c += sum(lean_ms[b]["p50"] for b in S if b in lean_ms)
    return float(c)


def cost_multi(S, costs):
    """The mean of cost_ms over the training datasets, each with its own profile, structural share and lean timings."""
    return float(np.mean([cost_ms(S, prof, share, lean_ms) for prof, share, lean_ms in costs]))


def greedy(model, select, blocks, costs):
    S = list(blocks)
    q_full = quality(model, select, blocks, {b: 1.0 for b in blocks})
    path = [{"blocks": list(S), "cost_ms": cost_multi(S, costs), "select": q_full, "dropped": None}]
    while [b for b in S if b not in NEVER]:
        cands = []
        here = 0.5 * (path[-1]["select"][0] + path[-1]["select"][1])
        for b in S:
            if b in NEVER:
                continue
            T = [x for x in S if x != b]
            q = quality(model, select, blocks, {x: (1.0 if x in T else 0.0) for x in blocks})
            dq = here - 0.5 * (q[0] + q[1])
            dc = path[-1]["cost_ms"] - cost_multi(T, costs)
            cands.append((dq / max(dc, 1e-3) if dq > 0 else dq - 1e3 * dc, b, q, T, dq, dc))
        cands.sort(key=lambda t: t[0])
        _, b, q, T, dq, dc = cands[0]
        S = T
        path.append({"blocks": list(S), "cost_ms": cost_multi(S, costs), "select": q, "dropped": b,
                     "loss_pts": 100 * dq, "saved_ms": dc,
                     "candidates": {c[1]: {"loss_pts": 100 * c[4], "saved_ms": c[5]} for c in cands}})
        log(f"  drop {b:13s} -> {len(S):2d} blocks, cost {path[-1]['cost_ms']:6.2f} ms, select {[round(v, 4) for v in q]}")
    return path


# ── main ─────────────────────────────────────────────────────────────────────


def parse_sets(s):
    return [tuple(p.split("=")) for p in (s or "").split(",") if p]


def read_one(model, c, blocks, keep_map, rng, full_rows=None):
    m = row_metrics(scores_of(model, c, blocks, keep_map), c.gold, c.off, c.gold_total)
    ok = c.gold_total > 0
    m = m[ok]
    rec = {"mean": [float(v) for v in m.mean(0)], "rows": int(ok.sum())}
    ref = {name: row_metrics(c.score[:, col], c.gold, c.off, c.gold_total)[ok] for name, col in SCORE_COL.items()}
    for name, r in ref.items():
        rec[name] = [float(v) for v in r.mean(0)]
        rec[f"minus_{name}"] = boot_diff(m, r, rng)
    t, g, x = ref["twin0"].mean(0), ref["gnn0"].mean(0), m.mean(0)
    rec["rho"] = [float((x[j] - t[j]) / (g[j] - t[j])) if abs(g[j] - t[j]) > 1e-9 else None for j in range(3)]
    rec["abs"] = [float(x[j] / g[j]) if g[j] > 0 else None for j in range(3)]
    if full_rows is not None:
        rec["minus_full"] = boot_diff(m, full_rows, rng)
    return rec, m


def fmt(r):
    pts = r["minus_twin0"]
    s = " ".join(f"{100 * p[0]:+.2f}[{100 * p[1][0]:+.2f},{100 * p[1][1]:+.2f}]" for p in pts)
    out = f"mean {[round(v, 4) for v in r['mean']]} vs twin0 {s} rho {[None if v is None else round(v, 3) for v in r['rho']]}"
    if "minus_full" in r:
        out += " vs full " + " ".join(f"{100 * p[0]:+.2f}[{100 * p[1][0]:+.2f},{100 * p[1][1]:+.2f}]" for p in r["minus_full"])
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
    ap.add_argument("--save-models", help="write every fitted model (state, blocks, keep map) to this .pt file")
    ap.add_argument("--load-models", help="read only: load the models from this .pt file and read the --read carves")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    if a.load_models:
        return read_only(a, t0)
    tr = [Carve(ds, cv, a.limit) for ds, cv in parse_sets(a.train)]
    se = [Carve(ds, cv, a.limit) for ds, cv in parse_sets(a.select)]
    log(f"loaded train {[(c.ds, c.carve, c.rows) for c in tr]} select {[(c.ds, c.carve, c.rows) for c in se]} ({time.time() - t0:.0f}s)")
    present = list(COMPILED) + list(LEAN) if a.blocks == "all" else a.blocks.split("+")
    dead = []
    for b in list(present):
        if b in LEAN:
            continue
        v = np.concatenate([c.block(b, np.arange(int(c.off[-1]))).astype(np.float32) for c in tr])
        if float(np.nanstd(v)) == 0.0:
            dead.append(b)
            present.remove(b)
    log(f"blocks {present}; constant on the training rows (dropped) {dead}")
    costs = []
    for c in tr:
        lm = dict(c.lean_ms)
        lm["SEMB"] = {"p50": 0.0}
        costs.append((profile(c.ds), c.struct_share, lm))
    out = {"look": "lean_mlp", "args": vars(a), "blocks": present, "dead": dead,
           "lean_ms": {f"{c.ds}={c.carve}": c.lean_ms for c in tr + se},
           "profiles": {c.ds: profile(c.ds) for c in tr}, "struct_share": {f"{c.ds}={c.carve}": c.struct_share for c in tr},
           "fam_entries": {f"{c.ds}={c.carve}": c.fam_entries for c in tr},
           "assumptions": ["retrieval lap split 25/75 between the rank lists and the embedding gather",
                           "edge build split by the structural share of pool edge entries",
                           "topology lap split equally over its four views", "D, depth_basis, typed_basis split in halves",
                           "lean blocks timed in numpy here, not in the numba compile", "SEMB's node side precomputed at index time",
                           "selection cost is the mean over the training datasets' profiles"],
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    models = {}
    pick = None
    if not a.no_greedy:
        log("fit with block dropout")
        model, curve = fit(tr, se, present, "drop", a.epochs, a.lr, a.seed, a.hidden)
        out["drop_curve"] = curve
        path = greedy(model, se, present, costs)
        out["path"] = path
        full_q = 0.5 * (path[0]["select"][0] + path[0]["select"][1])
        ok = [p for p in path if full_q - 0.5 * (p["select"][0] + p["select"][1]) <= a.tol / 100.0]
        pick = min(ok, key=lambda p: p["cost_ms"])
        out["pick"] = {k: pick[k] for k in ("blocks", "cost_ms", "select")}
        log(f"pick {pick['blocks']} at {pick['cost_ms']:.2f} ms (full {path[0]['cost_ms']:.2f} ms)")
        models["drop/full"] = (model, present, {b: 1.0 for b in present})
        models["drop/pick"] = (model, present, {b: (1.0 if b in pick["blocks"] else 0.0) for b in present})
    fixed = {"full": present, "compiled": [b for b in present if b in COMPILED],
             "lean": [b for b in present if b in NEVER or b in LEAN]}
    if pick is not None:
        fixed["pick"] = pick["blocks"]
    for part in [p for p in a.fixed.split(";") if p]:
        name, bl = part.split("=")
        fixed[name] = [b for b in bl.split("+") if b in present]
    out["fixed_sets"] = fixed
    out["fixed_curves"] = {}
    for name, bl in fixed.items():
        log(f"refit {name} ({len(bl)} blocks, {cost_multi(bl, costs):.2f} ms): {bl}")
        m2, cur = fit(tr, se, bl, "fixed", a.final_epochs, a.lr, a.seed, a.hidden)
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
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def read_only(a, t0):
    blob = torch.load(a.load_models, weights_only=False)
    models = {}
    for name, d in blob["models"].items():
        m = LeanMLP(d["blocks"], d["widths"], d["hidden"])
        m.load_state_dict(d["state"])
        models[name] = (m, d["blocks"], d["keep"])
    pick = blob.get("pick")
    out = {"look": "lean_mlp", "mode": "read_only", "args": vars(a), "loaded": a.load_models,
           "loaded_sha256": hashlib.sha256(Path(a.load_models).read_bytes()).hexdigest(), "fit_args": blob.get("args"),
           "pick": pick, "blocks": blob.get("present"), "dead": blob.get("dead"),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    log(f"loaded {list(models)} from {a.load_models}")
    read_all(a, models, pick, out)
    out["seconds"] = time.time() - t0
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def read_all(a, models, pick, out):
    rng = np.random.default_rng(20261003)
    reads, read_costs, read_lean_ms, read_rows = {}, {}, {}, {}
    for ds, cv in parse_sets(a.read):
        key = f"{ds}={cv}"
        c = Carve(ds, cv, a.limit)
        read_rows[key] = {"rows": c.rows, "chunks": c.chunks_read, "n_chunks": c.n_chunks, "carve_queries": c.carve_queries}
        lm = dict(c.lean_ms)
        lm["SEMB"] = {"p50": 0.0}
        read_lean_ms[key] = c.lean_ms
        read_costs[key] = {name: cost_ms(bl, profile(ds), c.struct_share, lm) for name, (_m, bl, _k) in models.items()
                           if not name.startswith("drop/")}
        if pick is not None:
            read_costs[key]["drop/pick"] = cost_ms(pick["blocks"], profile(ds), c.struct_share, lm)
        log(f"read {key} ({c.rows} rows); approximate compile ms {({k: round(v, 2) for k, v in read_costs[key].items()})}")
        base_rows = {}
        for name, (m, bl, keep) in models.items():
            ref = base_rows.get("drop/full" if name == "drop/pick" else "full")
            r, rows = read_one(m, c, bl, keep, rng, ref)
            if name in ("full", "drop/full"):
                base_rows[name] = rows
            reads[f"{name}@{key}"] = r
            log(f"  {name}@{key}: {fmt(r)}")
        del c
    out["read_costs_ms"] = read_costs
    out["read_lean_ms"] = read_lean_ms
    out["read_rows"] = read_rows
    out["reads"] = reads


def selftest():
    rng = np.random.default_rng(0)
    R = projection()
    n = 7
    proj = rng.standard_normal((n, PROJ_DIM)).astype(np.float16)
    q_emb = rng.standard_normal(1536).astype(np.float32)
    seeds = np.asarray([0, 3, 6, -1])
    buckets = np.asarray([0, 1, 0, -1])
    eu = np.asarray([0, 1, 1, 2, 3, 4, 2, 5, 1, 0], np.int16)
    ev = np.asarray([1, 0, 2, 1, 4, 3, 5, 2, 0, 1], np.int16)
    efam = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 0, 0], np.int8)
    efwd = np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 0, 1], np.int8)
    ebwd = 1 - efwd
    timer = {b: 0.0 for b in ("SEM", "SEED", "WALK", "NBR")}
    L, q, Pn = lean_query(n, proj, q_emb, R, seeds, buckets, eu, ev, efam, efwd, ebwd, timer)
    st = efam == 0
    A, Af, Ab = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for u, v, fw in zip(eu[st], ev[st], efwd[st]):
        A[v, u] += 1
        (Af if fw else Ab)[v, u] += 1
    s0 = np.zeros(n)
    s0[[0, 3, 6]] = 1
    sb = np.zeros(n)
    sb[[0, 6]] = 1
    col = 0
    for M in (A, Af, Ab):
        c = s0
        for _h in range(3):
            c = M @ c
            assert np.allclose(L["WALK"][:, col], np.log1p(c), atol=1e-5), col
            col += 1
    c = sb
    for _h in range(2):
        c = A @ c
        assert np.allclose(L["WALK"][:, col], np.log1p(c), atol=1e-5), col
        col += 1
    assert np.allclose(L["WALK"][:, 11], np.log1p(A.sum(1))), "log degree"
    # node 5 is reached only through a NER edge; node 6 is an isolated seed
    assert L["WALK"][5, :12].sum() == 0 and L["WALK"][5, 12:].sum() == 0
    assert L["WALK"][1, 12] == 1 and L["WALK"][2, 13] == 1 and L["WALK"][4, 12] == 1, "first-hop one-hot"
    assert L["WALK"][[0, 3, 6], 15].tolist() == [1, 1, 1] and L["WALK"][[1, 2, 4, 5], 15].sum() == 0, "is-seed"
    P = proj.astype(np.float32)
    Pn_ = P / np.linalg.norm(P, axis=1, keepdims=True)
    qp = (q_emb @ R) / np.linalg.norm(q_emb @ R)
    assert np.allclose(Pn, Pn_, atol=1e-6) and np.allclose(q, qp, atol=1e-6)
    m = Pn_[[0, 3, 6]].mean(0)
    G = Pn_ @ Pn_[[0, 3, 6]].T
    assert np.allclose(L["SEED"][:, 0], Pn_ @ (m / np.linalg.norm(m)), atol=1e-5)
    assert np.allclose(L["SEED"][:, 1], G.max(1), atol=1e-5) and np.allclose(L["SEED"][:, 2], G[:, [0, 2]].max(1), atol=1e-5)
    for i in range(n):
        nb = [int(u) for u, v, f in zip(eu, ev, efam) if f == 0 and v == i]
        if nb:
            mean = Pn_[nb].mean(0)
            cu = Pn_[nb] @ qp
            want = [cu.mean(), mean @ qp / np.linalg.norm(mean), cu.max(), np.linalg.norm(mean), 1.0]
            assert np.allclose(L["NBR"][i], want, atol=1e-4), (i, L["NBR"][i], want)
        else:
            assert L["NBR"][i].sum() == 0
    print("lean blocks equal brute force (walks by direction and from rank-1 seeds, first hop, is-seed, projected seed "
          "cosines, neighbour mean, mean-vector cosine, max and cohesion, with a multi-edge)")
    widths = {"rank": 3, "WALK": 16, "SEMB": SEMB_DIM}
    model = LeanMLP(["rank", "WALK", "SEMB"], widths)
    model.eval()
    N, B = 10, 2
    nq = torch.tensor([0] * 6 + [1] * 4)
    feats = {"rank": torch.randn(N, 3), "WALK": torch.randn(N, 16), "SEMB": (torch.randn(B, 1536), torch.randn(N, PROJ_DIM))}
    base = torch.randn(N)
    assert torch.allclose(model(feats, torch.ones(B, 3), nq, B, base), base), "zero-initialised output scores the base"
    with torch.no_grad():
        model.out.weight.normal_()
    keep = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    s1 = model(feats, keep, nq, B, base)
    feats2 = {"rank": feats["rank"], "WALK": torch.randn(N, 16) * 50, "SEMB": (torch.randn(B, 1536), torch.randn(N, PROJ_DIM))}
    assert torch.allclose(s1, model(feats2, keep, nq, B, base)), "a masked block does not reach the score"
    keep2 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    s3, s4 = model(feats, keep2, nq, B, base), model(feats2, keep2, nq, B, base)
    assert torch.allclose(s3[6:], s4[6:]) and not torch.allclose(s3[:6], s4[:6]), "masks are per row"
    print("unfitted model scores the base; a masked block cannot reach the score; masks are per row")
    prof = {"retrieval": 2.0, "edges": 4.0, "topology": 2.0, "edge_weights": 0.1, "C": 1.5, "gcs": 0.1, "D": 2.0, "B": 0.0,
            "depth_basis": 1.0, "typed_basis": 0.0}
    lm = {b: {"p50": 0.1} for b in LEAN}
    assert abs(cost_ms(["rank"], prof, 0.5, lm) - 0.5) < 1e-9
    assert abs(cost_ms(["rank", "WALK"], prof, 0.5, lm) - (0.5 + 2.0 + 0.1)) < 1e-9
    assert abs(cost_ms(["rank", "seed_e"], prof, 0.5, lm) - (2.0 + 1.0)) < 1e-9, "seed_e needs embeddings, no edge"
    assert abs(cost_ms(["rank", "seed_r"], prof, 0.5, lm) - (2.0 + 4.0 + 0.5 + 1.0)) < 1e-9, "seed_r needs the FULL view"
    full = cost_ms(list(COMPILED), prof, 0.5, lm)
    assert abs(full - (2 + 4 + 2 + 1.5 + 0.1 + 2 + 0 + 1 + 0)) < 1e-9, full
    print("cost model: rank alone, structural walks, seed_e without edges, seed_r's FULL view, the full compile (no edge_weights)")
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
