"""Design look (untracked; not a result and not filed): a lean, cost-aware MLP on the look's compiled inputs.

The user (3 Oct): the MLP must be faster than the GNN in every setting, so find which feature families it needs, drop
the rest, and make the extraction cheap at zero-shot inference. The twin and the GNN share one query-local compile
(129 columns in 13 families, plus the 1536-wide embeddings), so an MLP that reads all of it can only win by the
forward pass. This look fits MLPs from scratch on a carve's look rows and asks which families pay for their cost.

Families (blocks):
    the 13 compiled blocks of look_x_six.column_blocks (retrieval, topo_STRUCT/NER/KNN/FULL, nbr_agg, gcs, seedcond,
    typed_rel, depth_STRUCT/FULL, typed_v2, ordered), read from the look's x;
    four lean blocks computed here from what a zero-shot deployment can index cheaply:
      SEM    the query against the node's fixed 128-wide random projection (the look's proj): q_p * n_p and their cosine
      SEED   the node's projected cosine to the seeds' projected mean, its max over seeds, and to the rank-1 seed
      WALK   seed walk counts over STRUCTURAL edges only (hops 1-3, undirected / forward / backward), the rank-1 seed's
             hops 1-2, log degree and the first hop that reaches the node: fixed counts, no learned update, no PPR
      NBR    over structural edges: the projected neighbour mean's cosine to the query, the max projected neighbour
             cosine, and has-neighbour
Every lean block is a per-node function of the node, its seed walks and its structural neighbours' FIXED projections;
nothing is learned over edges, so a model on them is non-MP in the project's sense (no learned h_v update).

Model: per node [raw | within-query z-score | presence flag] per block -> Linear(H) GELU Linear(H) GELU -> score,
added to the z-scored rrf column (the twin's own base), with the output layer zero-initialised. Training (--mode drop)
draws a keep rate r ~ U(0.15, 1) per row and keeps each block with probability r, so one fit can be read with any
block switched off (a frozen read per subset, which a plain fit cannot give). Loss: the historical listwise loss.

Selection: on the SELECT carve, greedy backward elimination over the blocks, each step dropping the block with the
smallest quality loss (mean of R@5 and FC@5) per approximate compile millisecond saved; the cost model maps each
block to the fast compile's per-group p50 (scratchpad crag_profile/<ds>_fast.json) with the groups' dependencies,
splits the edge build by family edge share (approximate, flagged), and times the lean blocks here. The cheapest
subset within --tol points of the full set on select is refitted without dropout (--mode fixed) and read once on
the read carve, against the six twin and GNN scores stored in the look (twin0, gnn0).

    python outputs/mp_unified/lean_mlp.py --train 2wiki=x4 --select 2wiki=select --read 2wiki=x1,hotpotqa=x1 \
        --out outputs/mp_unified/lean/l1-2w.json [--epochs 8] [--limit 400] [--threads 2]
    python outputs/mp_unified/lean_mlp.py --selftest
"""
import argparse
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
ROOT = HERE.parents[1]
LOOK = HERE / "look"
PROFILE = Path(os.environ.get("LEAN_PROFILE_DIR", str(HERE / "lean" / "crag_profile")))
PROJ_DIM, PROJ_SEED = 128, 20261001
METRICS = ("recall@5", "full_coverage@5", "hit@1")
COMPILED = ("retrieval", "topo_STRUCT", "topo_NER", "topo_KNN", "topo_FULL", "nbr_agg", "gcs", "seedcond", "typed_rel",
            "depth_STRUCT", "depth_FULL", "typed_v2", "ordered")
LEAN = ("SEM", "SEED", "WALK", "NBR")
LEAN_W = {"SEM": PROJ_DIM + 1, "SEED": 3, "WALK": 16, "NBR": 3}
SCORE_COL = {"twin0": 0, "gnn0": 3}
BOOT = 1000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def projection():
    return (np.random.default_rng(PROJ_SEED).standard_normal((1536, PROJ_DIM)) / math.sqrt(PROJ_DIM)).astype(np.float32)


# ── lean blocks, per query ───────────────────────────────────────────────────


def lean_query(n, proj, q_emb, R, seeds, buckets, eu, ev, efam, efwd, ebwd, timer):
    """The four lean blocks of one query (float32), each timed into timer[block] (seconds)."""
    t = time.perf_counter()
    P = proj.astype(np.float32)
    q = q_emb.astype(np.float32) @ R
    q /= max(float(np.linalg.norm(q)), 1e-12)
    pn = np.linalg.norm(P, axis=1)
    prod = P * q[None, :]
    cos = prod.sum(1) / np.maximum(pn, 1e-12)
    sem = np.concatenate([prod, cos[:, None]], axis=1)
    t1 = time.perf_counter()
    timer["SEM"] += t1 - t
    S = seeds[seeds >= 0]
    Pn = P / np.maximum(pn, 1e-12)[:, None]
    seed = np.zeros((n, 3), np.float32)
    if S.size:
        Ps = Pn[S]
        m = Ps.mean(0)
        m /= max(float(np.linalg.norm(m)), 1e-12)
        G = Pn @ Ps.T
        seed[:, 0] = Pn @ m
        seed[:, 1] = G.max(1)
        top = S[buckets[: S.size] == 0]
        seed[:, 2] = G[:, int(np.flatnonzero(buckets[: S.size] == 0)[0])] if top.size else 0.0
    t2 = time.perf_counter()
    timer["SEED"] += t2 - t1
    st = efam == 0
    u, v = eu[st].astype(np.int64), ev[st].astype(np.int64)
    fw, bw = efwd[st].astype(bool), ebwd[st].astype(bool)
    walk = np.zeros((n, 16), np.float32)
    s0 = np.zeros(n, np.float32)
    s0[S] = 1.0
    sb = np.zeros(n, np.float32)
    if S.size:
        sb[S[buckets[: S.size] == 0]] = 1.0
    col = 0
    for mask in (np.ones(u.size, bool), fw, bw):
        uu, vv = u[mask], v[mask]
        c = s0
        for _h in range(3):
            c = np.bincount(vv, weights=c[uu], minlength=n).astype(np.float32)
            walk[:, col] = np.log1p(c)
            col += 1
    c = sb
    for _h in range(2):
        c = np.bincount(v, weights=c[u], minlength=n).astype(np.float32)
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
        walk[:, col + h] = (first == h)
    t3 = time.perf_counter()
    timer["WALK"] += t3 - t2
    nbr = np.zeros((n, 3), np.float32)
    if u.size:
        acc = np.zeros((n, PROJ_DIM), np.float32)
        np.add.at(acc, v, Pn[u])
        has = deg > 0
        mean = acc / np.maximum(deg, 1.0)[:, None]
        nbr[:, 0] = np.where(has, (mean @ q) / np.maximum(np.linalg.norm(mean, axis=1), 1e-12), 0.0)
        cu = Pn[u] @ q
        mx = np.full(n, -np.inf, np.float32)
        np.maximum.at(mx, v, cu)
        nbr[:, 1] = np.where(has, mx, 0.0)
        nbr[:, 2] = has
    timer["NBR"] += time.perf_counter() - t3
    return {"SEM": sem, "SEED": seed, "WALK": walk, "NBR": nbr}


# ── loading ──────────────────────────────────────────────────────────────────


class Carve:
    """One carve's rows: node-major float16 blocks, query offsets, gold flags, the stored six-pair scores and metrics."""

    def __init__(self, ds, carve, limit=None, root=LOOK):
        d = Path(root) / ds / carve
        recs = sorted(d.glob("record*.json"))
        if not recs:
            raise SystemExit(f"{d}: no look record")
        rec = json.loads(recs[0].read_text(encoding="utf-8"))
        self.ds, self.carve = ds, carve
        self.columns = rec["columns"]
        self.blocks_idx = {b: np.asarray(v, np.int64) for b, v in rec["column_blocks"].items() if b in COMPILED}
        R = projection()
        xs, lean, gold, score, qn, qmet = [], {b: [] for b in LEAN}, [], [], [], []
        self.timer = {b: 0.0 for b in LEAN}
        per_q_ms = {b: [] for b in LEAN}
        rows = 0
        for ch in sorted((d / "chunks").glob("c*.npz")):
            z = np.load(ch)
            qp, qe = z["q_pool_size"], z["q_edges"]
            no = np.concatenate([[0], np.cumsum(qp)])
            eo = np.concatenate([[0], np.cumsum(qe)])
            for i in range(qp.size):
                if limit is not None and rows >= limit:
                    break
                a, b = no[i], no[i + 1]
                ea, eb = eo[i], eo[i + 1]
                before = dict(self.timer)
                L = lean_query(int(qp[i]), z["proj"][a:b], z["q_emb"][i], R, z["q_seed_local"][i], z["q_seed_bucket"][i],
                               z["e_u"][ea:eb], z["e_v"][ea:eb], z["e_fam"][ea:eb], z["e_fwd"][ea:eb], z["e_bwd"][ea:eb],
                               self.timer)
                for k in LEAN:
                    lean[k].append(L[k].astype(np.float16))
                    per_q_ms[k].append(1e3 * (self.timer[k] - before[k]))
                xs.append(z["x"][a:b])
                gold.append(z["is_gold"][a:b])
                score.append(z["score"][a:b])
                qn.append(int(qp[i]))
                qmet.append(z["q_metrics"][i])
                rows += 1
            if limit is not None and rows >= limit:
                break
        self.x = np.concatenate(xs)
        self.lean = {k: np.concatenate(v) for k, v in lean.items()}
        self.gold = np.concatenate(gold)
        self.score = np.concatenate(score)
        self.n = np.asarray(qn, np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.n)])
        self.q_metrics = np.stack(qmet)
        self.lean_ms = {k: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(np.mean(v))}
                        for k, v in per_q_ms.items()}
        self.rrf = self.columns.index("rrf")
        self.widths = {b: int(self.blocks_idx[b].size) for b in COMPILED if b in self.blocks_idx}
        self.widths.update(LEAN_W)
        has_gold = np.add.reduceat(self.gold.astype(np.int64), self.off[:-1]) > 0
        self.rows = int(self.n.size)
        self.rows_with_gold = int(has_gold.sum())

    def block(self, b, sl):
        if b in LEAN:
            return self.lean[b][sl]
        return self.x[sl][:, self.blocks_idx[b]]


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
    def __init__(self, blocks, widths, hidden=128, dropout=0.1):
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

    def forward(self, feats, keep, nq, B, base_z):
        """feats[b]: (N, w) float32 raw; keep: (B, n_blocks) float32 0/1 per row and block."""
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = feats[b]
            parts += [raw * m, seg_zscore(raw, nq, B) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * base_z + self.out(h).squeeze(-1)


def batch_of(carve, qs, blocks):
    sls = [slice(int(carve.off[q]), int(carve.off[q + 1])) for q in qs]
    idx = np.concatenate([np.arange(s.start, s.stop) for s in sls])
    nq = torch.from_numpy(np.repeat(np.arange(len(qs)), carve.n[qs]))
    feats = {}
    for b in blocks:
        if b in LEAN:
            a = carve.lean[b][idx]
        else:
            a = carve.x[idx][:, carve.blocks_idx[b]]
        feats[b] = torch.from_numpy(np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
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


def row_metrics(scores, gold, off):
    """Per row R@5, FC@5, hit@1 (rows without an in-pool gold are skipped by the caller's mask)."""
    out = np.zeros((off.size - 1, 3))
    has = np.zeros(off.size - 1, bool)
    for i in range(off.size - 1):
        s, g = scores[off[i]:off[i + 1]], gold[off[i]:off[i + 1]]
        ng = int(g.sum())
        if ng == 0:
            continue
        has[i] = True
        order = np.lexsort((np.arange(s.size), -s))   # ties broken by pool position, as the rank metrics do
        top = g[order[:5]].sum()
        out[i] = (top / ng, float(top == ng), float(g[order[0]]))
    return out, has


def gold_total(carve):
    """Recall is over in-pool golds here; the look's own rank metrics are the reference for the stored pair."""
    return None


def fit(train, select, blocks, mode, epochs, lr, seed, hidden, threads, log_every=1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: train[0].widths[b] for b in blocks}
    model = LeanMLP(blocks, widths, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    units = [(ci, q) for ci, c in enumerate(train) for q in range(c.rows)]
    best, best_state, curve = -1.0, None, []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(units))
        t0 = time.time()
        tot = 0.0
        nb = 0
        for k in range(0, len(order), 32):
            chunk = [units[j] for j in order[k:k + 32]]
            for ci in sorted({c for c, _ in chunk}):
                qs = np.asarray([q for c, q in chunk if c == ci])
                feats, nq, base_z, gold, _ = batch_of(train[ci], qs, blocks)
                B = qs.size
                if mode == "drop":
                    r = rng.uniform(0.15, 1.0, size=(B, 1))
                    keep = (rng.uniform(size=(B, len(blocks))) < r).astype(np.float32)
                else:
                    keep = np.ones((B, len(blocks)), np.float32)
                s = model(feats, torch.from_numpy(keep), nq, B, base_z)
                loss = listwise(s, gold, nq, B)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
                nb += 1
        q = quality(model, select, blocks, {b: 1.0 for b in blocks})
        curve.append({"epoch": ep, "loss": tot / max(nb, 1), "select": q, "seconds": time.time() - t0})
        if ep % log_every == 0:
            log(f"  ep {ep}: loss {tot / max(nb, 1):.4f} select R@5/FC@5/hit@1 {[round(v, 4) for v in q]} ({time.time() - t0:.0f}s)")
        score = 0.5 * (q[0] + q[1])
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, curve


@torch.no_grad()
def scores_of(model, carves, blocks, keep_map):
    model.eval()
    out = []
    for c in carves:
        s_all = np.zeros(int(c.off[-1]), np.float32)
        for k in range(0, c.rows, 64):
            qs = np.arange(k, min(k + 64, c.rows))
            feats, nq, base_z, _gold, idx = batch_of(c, qs, blocks)
            keep = torch.tensor([[keep_map.get(b, 1.0) for b in blocks]] * qs.size, dtype=torch.float32)
            s_all[idx] = model(feats, keep, nq, qs.size, base_z).numpy()
        out.append(s_all)
    return out


def quality(model, carves, blocks, keep_map):
    ss = scores_of(model, carves, blocks, keep_map)
    ms = []
    for c, s in zip(carves, ss):
        m, has = row_metrics(s, c.gold, c.off)
        ms.append(m[has])
    m = np.concatenate(ms)
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


def cost_ms(S, prof, share_struct, lean_ms):
    """Approximate per-query compile ms of a block set (p50 groups; the edge build split by family edge share)."""
    S = set(S)
    if prof is None:
        return float("nan")
    struct_users = {"topo_STRUCT", "topo_FULL", "nbr_agg", "gcs", "seedcond", "depth_STRUCT", "depth_FULL", "typed_rel",
                    "typed_v2", "ordered", "WALK", "NBR"}
    nk_users = {"topo_NER", "topo_KNN", "topo_FULL", "nbr_agg", "gcs", "seedcond", "depth_FULL"}
    views = set()
    if S & {"topo_STRUCT", "gcs", "depth_STRUCT"}:
        views.add("STRUCT")
    if "topo_NER" in S:
        views.add("NER")
    if "topo_KNN" in S:
        views.add("KNN")
    if S & {"topo_FULL", "gcs", "seedcond", "depth_FULL"}:
        views.add("FULL")
    c = prof.get("retrieval", 0.0)
    c += prof.get("edges", 0.0) * (share_struct * bool(S & struct_users) + (1 - share_struct) * bool(S & nk_users))
    c += prof.get("topology", 0.0) * len(views) / 4.0
    c += prof.get("edge_weights", 0.0) * bool(S & nk_users)
    c += prof.get("C", 0.0) * ("nbr_agg" in S) + prof.get("gcs", 0.0) * ("gcs" in S) + prof.get("D", 0.0) * ("seedcond" in S)
    c += prof.get("B", 0.0) * ("typed_rel" in S)
    c += prof.get("depth_basis", 0.0) * (0.5 * ("depth_STRUCT" in S) + 0.5 * ("depth_FULL" in S))
    c += prof.get("typed_basis", 0.0) * (0.5 * ("typed_v2" in S) + 0.5 * ("ordered" in S))
    c += sum(lean_ms[b]["p50"] for b in S if b in lean_ms)
    return float(c)


def greedy(model, select, blocks, prof, share, lean_ms, never=("retrieval",)):
    S = list(blocks)
    path = []
    q_full = quality(model, select, blocks, {b: 1.0 for b in blocks})
    path.append({"blocks": list(S), "cost_ms": cost_ms(S, prof, share, lean_ms), "select": q_full, "dropped": None})
    while len([b for b in S if b not in never]) > 0:
        cands = []
        for b in S:
            if b in never:
                continue
            T = [x for x in S if x != b]
            q = quality(model, select, blocks, {x: (1.0 if x in T else 0.0) for x in blocks})
            dq = 0.5 * (path[-1]["select"][0] + path[-1]["select"][1]) - 0.5 * (q[0] + q[1])
            dc = path[-1]["cost_ms"] - cost_ms(T, prof, share, lean_ms)
            cands.append((dq / max(dc, 1e-3) if dq > 0 else dq - 1e3 * dc, b, q, T))
        cands.sort(key=lambda t: t[0])
        _, b, q, T = cands[0]
        S = T
        path.append({"blocks": list(S), "cost_ms": cost_ms(S, prof, share, lean_ms), "select": q, "dropped": b})
        log(f"  drop {b:13s} -> {len(S)} blocks, cost {path[-1]['cost_ms']:.2f} ms, select {[round(v, 4) for v in q]}")
    return path


# ── main ─────────────────────────────────────────────────────────────────────


def parse_sets(s):
    out = []
    for part in [p for p in (s or "").split(",") if p]:
        ds, cv = part.split("=")
        out.append((ds, cv))
    return out


def read_block(model, c, blocks, keep_map, rng):
    s = scores_of(model, [c], blocks, keep_map)[0]
    m, has = row_metrics(s, c.gold, c.off)
    ref = {}
    for name, col in SCORE_COL.items():
        r, _ = row_metrics(c.score[:, col], c.gold, c.off)
        ref[name] = r[has]
    m = m[has]
    rec = {"fit": [float(v) for v in m.mean(0)], "rows": int(has.sum())}
    for name in SCORE_COL:
        rec[name] = [float(v) for v in ref[name].mean(0)]
        rec[f"minus_{name}"] = boot_diff(m, ref[name], rng)
    gap = ref["gnn0"].mean(0) - ref["twin0"].mean(0)
    rec["rho"] = [float((m.mean(0)[j] - ref["twin0"].mean(0)[j]) / gap[j]) if abs(gap[j]) > 1e-9 else None for j in range(3)]
    rec["abs"] = [float(m.mean(0)[j] / ref["gnn0"].mean(0)[j]) if ref["gnn0"].mean(0)[j] > 0 else None for j in range(3)]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="2wiki=x4")
    ap.add_argument("--select", default="2wiki=select")
    ap.add_argument("--read", default="2wiki=x1")
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--final-epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.3, help="points of mean(R@5, FC@5) on select")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--fixed", default="", help="extra fixed-mode refits: name=blockA+blockB;name2=...")
    ap.add_argument("--out", required=False)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    tr = [Carve(ds, cv, a.limit) for ds, cv in parse_sets(a.train)]
    se = [Carve(ds, cv, a.limit) for ds, cv in parse_sets(a.select)]
    rd = {f"{ds}={cv}": (ds, cv) for ds, cv in parse_sets(a.read)}
    log(f"loaded train {[(c.ds, c.carve, c.rows) for c in tr]} select {[(c.ds, c.carve, c.rows) for c in se]} ({time.time() - t0:.0f}s)")
    present = [b for b in COMPILED if all(b in c.widths and c.widths[b] > 0 for c in tr)] + list(LEAN)
    if a.blocks != "all":
        present = [b for b in a.blocks.split("+")]
    # a block whose columns never vary is dead on these graphs (typed blocks on passage graphs): dropped, recorded
    dead = []
    for b in list(present):
        if b in LEAN:
            continue
        v = np.concatenate([c.block(b, slice(0, int(c.off[-1]))).astype(np.float32) for c in tr])
        if float(np.nanstd(v)) == 0.0:
            dead.append(b)
            present.remove(b)
    log(f"blocks {present}; dead here {dead}")
    ds0 = tr[0].ds
    prof = profile(ds0)
    fam_share = None
    share = 0.57
    out = {"look": "lean_mlp", "train": a.train, "select": a.select, "read": a.read, "blocks": present, "dead": dead,
           "epochs": a.epochs, "lr": a.lr, "hidden": a.hidden, "seed": a.seed, "tol": a.tol, "limit": a.limit,
           "lean_ms": {f"{c.ds}={c.carve}": c.lean_ms for c in tr + se}, "profile": prof, "edge_share_struct": share,
           "script_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()}
    lean_ms = tr[0].lean_ms
    log("fit (block dropout)")
    model, curve = fit(tr, se, present, "drop", a.epochs, a.lr, a.seed, a.hidden, a.threads)
    out["drop_curve"] = curve
    path = greedy(model, se, present, prof, share, lean_ms)
    out["path"] = path
    full_q = 0.5 * (path[0]["select"][0] + path[0]["select"][1])
    ok = [p for p in path if full_q - 0.5 * (p["select"][0] + p["select"][1]) <= a.tol / 100.0]
    pick = min(ok, key=lambda p: p["cost_ms"])
    out["pick"] = pick
    log(f"pick {pick['blocks']} at {pick['cost_ms']:.2f} ms (full {path[0]['cost_ms']:.2f} ms)")
    rng = np.random.default_rng(20261003)
    reads = {}
    carves = {}
    for key, (ds, cv) in rd.items():
        carves[key] = Carve(ds, cv, a.limit)
    # the dropout model read with the full set and with the picked set (frozen subset reads)
    for key, c in carves.items():
        reads[f"drop/full@{key}"] = read_block(model, c, present, {b: 1.0 for b in present}, rng)
        reads[f"drop/pick@{key}"] = read_block(model, c, present, {b: (1.0 if b in pick["blocks"] else 0.0) for b in present}, rng)
    # fixed refits: the full set, the pick, the compiled-only set (the twin's families), the lean-only set
    fixed = {"full": present, "pick": pick["blocks"], "compiled": [b for b in present if b in COMPILED],
             "lean": ["retrieval"] + [b for b in present if b in LEAN]}
    for part in [p for p in a.fixed.split(";") if p]:
        name, bl = part.split("=")
        fixed[name] = [b for b in bl.split("+") if b in present]
    out["fixed_sets"] = fixed
    out["fixed_cost_ms"] = {k: cost_ms(v, prof, share, lean_ms) for k, v in fixed.items()}
    out["fixed_curves"] = {}
    for name, bl in fixed.items():
        log(f"refit {name}: {bl}")
        m2, cur = fit(tr, se, bl, "fixed", a.final_epochs, a.lr, a.seed, a.hidden, a.threads)
        out["fixed_curves"][name] = cur
        for key, c in carves.items():
            reads[f"{name}@{key}"] = read_block(m2, c, bl, {b: 1.0 for b in bl}, rng)
            r = reads[f"{name}@{key}"]
            log(f"  {name}@{key}: fit {[round(v, 4) for v in r['fit']]} rho {[None if v is None else round(v, 3) for v in r['rho']]} "
                f"abs {[None if v is None else round(v, 4) for v in r['abs']]} vs twin0 R@5 {100 * r['minus_twin0'][0][0]:+.2f} "
                f"[{100 * r['minus_twin0'][0][1][0]:+.2f}, {100 * r['minus_twin0'][0][1][1]:+.2f}]")
    out["reads"] = reads
    out["seconds"] = time.time() - t0
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def selftest():
    rng = np.random.default_rng(0)
    R = projection()
    n = 7
    proj = rng.standard_normal((n, PROJ_DIM)).astype(np.float16)
    q = rng.standard_normal(1536).astype(np.float32)
    seeds = np.asarray([0, 3, -1])
    buckets = np.asarray([0, 1, -1])
    eu = np.asarray([0, 1, 1, 2, 3, 4, 2, 5], np.int16)
    ev = np.asarray([1, 0, 2, 1, 4, 3, 5, 2], np.int16)
    efam = np.asarray([0, 0, 0, 0, 0, 0, 1, 1], np.int8)
    efwd = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], np.int8)
    ebwd = 1 - efwd
    timer = {b: 0.0 for b in LEAN}
    L = lean_query(n, proj, q, R, seeds, buckets, eu, ev, efam, efwd, ebwd, timer)
    # brute-force walks over the structural edges only
    A = np.zeros((n, n))
    for u, v, f in zip(eu, ev, efam):
        if f == 0:
            A[v, u] += 1
    s0 = np.zeros(n)
    s0[[0, 3]] = 1
    c1 = A @ s0
    c2 = A @ c1
    c3 = A @ c2
    assert np.allclose(L["WALK"][:, 0], np.log1p(c1)) and np.allclose(L["WALK"][:, 1], np.log1p(c2)) and np.allclose(L["WALK"][:, 2], np.log1p(c3))
    Af = np.zeros((n, n))
    for u, v, f, w in zip(eu, ev, efam, efwd):
        if f == 0 and w:
            Af[v, u] += 1
    assert np.allclose(L["WALK"][:, 3], np.log1p(Af @ s0)), "forward walks"
    # node 5 is reached only through a NER edge: no structural walk, degree 0, first hop 'none'
    assert L["WALK"][5, :9].sum() == 0 and L["WALK"][5, 11] == 0 and L["WALK"][5, 12:15].sum() == 0
    assert L["WALK"][1, 12] == 1 and L["WALK"][2, 13] == 1, "first-hop one-hot"
    Pn = proj.astype(np.float32) / np.linalg.norm(proj.astype(np.float32), axis=1, keepdims=True)
    qp = (q @ R) / np.linalg.norm(q @ R)
    assert np.allclose(L["SEM"][:, -1], Pn @ qp, atol=1e-4), "projected cosine"
    m = Pn[[0, 3]].mean(0)
    assert np.allclose(L["SEED"][:, 0], Pn @ (m / np.linalg.norm(m)), atol=1e-5) and np.allclose(L["SEED"][:, 2], Pn @ Pn[0], atol=1e-5)
    nb = [[u for u, v, f in zip(eu, ev, efam) if f == 0 and v == i] for i in range(n)]
    for i in range(n):
        if nb[i]:
            mean = Pn[nb[i]].mean(0)
            assert abs(L["NBR"][i, 0] - mean @ qp / np.linalg.norm(mean)) < 1e-4 and abs(L["NBR"][i, 1] - max(Pn[nb[i]] @ qp)) < 1e-4
        else:
            assert L["NBR"][i].sum() == 0
    print("lean blocks equal brute force (walks by direction, first hop, projected cosines, neighbour mean and max)")
    # the model: an unfitted model scores the base exactly, and a masked block cannot move the score
    widths = {"retrieval": 3, "WALK": 16}
    model = LeanMLP(["retrieval", "WALK"], widths)
    N, B = 10, 2
    nq = torch.tensor([0] * 6 + [1] * 4)
    feats = {"retrieval": torch.randn(N, 3), "WALK": torch.randn(N, 16)}
    base = torch.randn(N)
    s = model(feats, torch.ones(B, 2), nq, B, base)
    assert torch.allclose(s, base), "zero-initialised output scores the base"
    with torch.no_grad():
        model.out.weight.normal_()
    keep = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    s1 = model(feats, keep, nq, B, base)
    feats2 = dict(feats)
    feats2["WALK"] = torch.randn(N, 16) * 50
    s2 = model(feats2, keep, nq, B, base)
    assert torch.allclose(s1, s2), "a masked block does not reach the score"
    print("unfitted model scores the base; a masked block cannot reach the score")
    # the cost model: dependencies
    prof = {"retrieval": 2.0, "edges": 4.0, "topology": 2.0, "edge_weights": 0.1, "C": 1.5, "gcs": 0.1, "D": 2.0, "B": 0.0,
            "depth_basis": 1.0, "typed_basis": 0.0}
    lm = {b: {"p50": 0.1} for b in LEAN}
    assert abs(cost_ms(["retrieval"], prof, 0.5, lm) - 2.0) < 1e-9
    assert abs(cost_ms(["retrieval", "WALK"], prof, 0.5, lm) - (2.0 + 2.0 + 0.1)) < 1e-9
    full = cost_ms(list(COMPILED), prof, 0.5, lm)
    assert abs(full - (2 + 4 + 2 + 0.1 + 1.5 + 0.1 + 2 + 0 + 1 + 0)) < 1e-9, full
    assert abs(cost_ms(["retrieval", "seedcond"], prof, 0.5, lm) - (2 + 4 + 0.5 + 0.1 + 2)) < 1e-9
    print("cost model: retrieval alone, structural walks, the full compile and seedcond's FULL-view dependency")
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
