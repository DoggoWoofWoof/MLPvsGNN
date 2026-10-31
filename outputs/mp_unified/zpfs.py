"""PFS, the PKO feature screen (docs/PKO_FEATURE_SCREEN.md): what the cached inputs and the pool's edges carry, read on the CPU.

Two reads, no model, labels used only to describe the data (never as an input), on each dataset's PKO fit carve (musique: s1fit), the
first --nq questions in the carve's order:

  columns   for each cached column of the lean pick (rank, dense_cos, topo_STRUCT, depth_STRUCT, WALK, WALKF, SEED, DISTS): the share
            of rows at its modal value, the gold rate there and elsewhere, and the AUC of its within-question z-score for gold, over all
            rows and over the rows beyond the first stage's top five (rrf rank >= 5), where the golds a ranker misses sit.
  edges     one hop from the first stage's top K (K = 5 and 20 by rrf): the share of the golds outside the top K that sit one edge from
            it (per edge family); then, among those receivers, how well node-only scalars, each edge scalar, and a small logistic fit on
            node-only and on node + edge scalars (cross-fitted on question parity) put a gold receiver in the top 5 and top 20.

The edge scalars are made from what the cache holds: the cosine of the two rows' stored 128-wide projections, the number of top-K senders,
their rank-weighted count, the receiver's degree and the families on the pair. The links hold only (u, v, family).

    python outputs/mp_unified/zpfs.py columns --dataset hotpotqa
    python outputs/mp_unified/zpfs.py edges --dataset hotpotqa --k 5,20
    python outputs/mp_unified/zpfs.py prf --dataset hotpotqa              (amendment A2)
    python outputs/mp_unified/zpfs.py --selftest
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

DECLARED = "docs/PKO_FEATURE_SCREEN.md"
CACHE, LINKS, OUT = "outputs/pko/cache", "outputs/pko/links", "outputs/pfs"
CARVE = {"2wiki": "fit", "hotpotqa": "fit", "squad": "fit", "webqsp": "fit", "musique": "s1fit"}
NQ = {"2wiki": 1000, "hotpotqa": 1000, "squad": 1000, "webqsp": 600, "musique": 400}
BASIS = "hotpotqa"
AUC_ROWS = 1_000_000
FAMS = 3
EDGE_NAMES = ["base_z", "dense_cos", "seed0", "seed1", "seed2", "cos_max", "log_senders", "soft_senders", "log_deg", "fam0", "fam1", "fam2"]
NODE_ONLY = ["base_z", "dense_cos", "seed0", "seed1", "seed2"]
SCALARS = ["base_z", "dense_cos", "seed0", "cos_max", "soft_senders", "neg_log_deg"]
EDGE_ONLY = ["cos_max", "log_senders", "soft_senders", "log_deg", "fam0", "fam1", "fam2"]
FULL_BLOCKS = ["rank", "dense_cos", "topo_STRUCT", "depth_STRUCT", "WALK", "WALKF", "SEED", "DISTS"]    # the lean pick's fixed blocks


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def auc(s, y):
    """Tie-aware AUC (the Mann-Whitney U), None when one class is empty."""
    s = np.asarray(s, np.float64)
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return None
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    _u, start, cnt = np.unique(ss, return_index=True, return_counts=True)
    r = np.empty(s.size, np.float64)
    r[o] = np.repeat(start + (cnt + 1) / 2.0, cnt)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def hit_at(score, y, q, ms=(5, 20)):
    """Among questions with a gold receiver: the share whose top-m receivers by score hold a gold, and the share a random order would
    give (exact, from the pool size and the golds in it). Ties are broken by position."""
    score = np.asarray(score, np.float64)
    y = np.asarray(y, bool)
    q = np.asarray(q, np.int64)
    o = np.lexsort((np.arange(q.size), -score, q))
    qs, ys = q[o], y[o]
    first = np.r_[True, qs[1:] != qs[:-1]]
    grp = np.cumsum(first) - 1
    start = np.flatnonzero(first)
    pos = np.arange(qs.size) - start[grp]
    n = np.diff(np.r_[start, qs.size])
    g = np.bincount(grp, weights=ys.astype(np.float64), minlength=start.size)
    has = g > 0
    out = {"questions": int(has.sum())}
    for m in ms:
        got = np.zeros(start.size, bool)
        got[grp[ys & (pos < m)]] = True
        chance = np.ones(start.size)
        for i in range(m):                                   # P(no gold in a random top m) = prod (n - g - i) / (n - i)
            num = np.clip(n - g - i, 0, None)
            den = np.clip(n - i, 1, None)
            chance *= np.where(n - i > 0, num / den, 0.0)
        out[f"hit@{m}"] = float(got[has].mean()) if has.any() else None
        out[f"chance@{m}"] = float((1.0 - chance)[has].mean()) if has.any() else None
    return out


def lr_crossfit(X, y, q, l2=1e-3, iters=80):
    """Logistic regression on standardised columns, fitted on one question parity and scored on the other. Returns the scores and the
    standardised weights (mean over the two folds)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    q = np.asarray(q, np.int64)
    score = np.zeros(y.size)
    ws = []
    for fold in (0, 1):
        tr, te = (q % 2) != fold, (q % 2) == fold
        if tr.sum() == 0 or te.sum() == 0 or y[tr].sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float64)
        yt = torch.tensor(y[tr], dtype=torch.float64)
        w = torch.zeros(X.shape[1] + 1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = Xt @ w[:-1] + w[-1]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + l2 * (w[:-1] ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        wn = w.detach().numpy()
        score[te] = ((X[te] - mu) / sd) @ wn[:-1] + wn[-1]
        ws.append(wn[:-1])
    return score, (np.mean(ws, 0) if ws else np.zeros(X.shape[1]))


# ── the cache's carve, with edges, on the CPU ───────────────────────────────


def open_carve(ds, carve=None, basis=BASIS):
    import lean_cache as LC
    import lean_gpu as LG
    import zlink as ZL
    LG.bind_device_ops()
    cls = ZL.link_carve(LG.CacheCarve, LINKS)
    return cls(ds, carve or CARVE[ds], basis, "cpu", Path(CACHE), True)


def col_names(c):
    import lean_mlp as LM
    names = []
    for b, (a, e) in c.span.items():
        names += list(LM.SPLIT["rank"]) if b == "rank" else ([b] if e - a == 1 else [f"{b}[{i}]" for i in range(e - a)])
    return names


def columns_cmd(a):
    import lean_gpu as LG
    import zlink as ZL
    t0 = time.time()
    c = open_carve(a.dataset)
    nq = min(a.nq or NQ[a.dataset], c.rows)
    R = int(c.off_np[nq])
    X = torch.nan_to_num(c.X[:R].to(torch.float32), nan=0.0)
    nan_share = float(torch.isnan(c.X[:R].to(torch.float32)).float().mean())
    gold = c.gold[:R].numpy()
    nqv = torch.from_numpy(np.repeat(np.arange(nq), c.n_np[:nq]))
    rank = ZL.pool_rank(X[:, c.c_rrf].contiguous(), nqv, nq).numpy()
    Z = LG.seg_zscore8D(X, nqv, nq).numpy()
    names = col_names(c)
    assert len(names) == X.shape[1], (len(names), X.shape)
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(R, min(R, AUC_ROWS), replace=False))
    beyond = sub[rank[sub] >= 5]
    out = {"declared_in": DECLARED, "dataset": a.dataset, "carve": c.carve, "questions": nq, "rows": R, "gold_rows": int(gold.sum()),
           "golds_beyond_top5": int(gold[rank >= 5].sum()), "nan_share": nan_share, "part_shas": c.part_shas, "columns": {}}
    Xn = X.numpy()
    for j, nm in enumerate(names):
        v = Xn[:, j]
        vals, cnt = np.unique(v[sub], return_counts=True)
        mode = float(vals[cnt.argmax()])
        at = v == mode
        out["columns"][nm] = {"mode": mode, "share_at_mode": float(at.mean()),
                              "gold_rate_at_mode": float(gold[at].mean()) if at.any() else None,
                              "gold_rate_elsewhere": float(gold[~at].mean()) if (~at).any() else None,
                              "share_of_golds_at_mode": float(gold[at].sum() / max(gold.sum(), 1)),
                              "auc_z": auc(Z[sub, j], gold[sub]), "auc_z_beyond_top5": auc(Z[beyond, j], gold[beyond]),
                              "distinct_in_sample": int(vals.size)}
    out["seconds"] = time.time() - t0
    p = Path(OUT) / f"columns_{a.dataset}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"columns {a.dataset}: {nq} questions, {R} rows, {len(names)} columns ({out['seconds']:.0f}s) -> {p}")
    for nm in names:
        r = out["columns"][nm]
        log(f"  {nm:22s} mode {r['mode']:9.3f} at {r['share_at_mode']:.3f}  auc {r['auc_z'] if r['auc_z'] is None else round(r['auc_z'], 3)}  "
            f"beyond5 {r['auc_z_beyond_top5'] if r['auc_z_beyond_top5'] is None else round(r['auc_z_beyond_top5'], 3)}")
    return 0


def receivers(c, qs, K, full=False):
    """For questions qs: one row per receiver (a row outside the top K by rrf, one edge from a top-K row), its labels and scalars.
    Also the golds outside the top K, and how many of them have an edge from the top K, per family."""
    import zlink as ZL
    feats, nq, base_z, gold = c.batch(qs, FULL_BLOCKS if full else ["dense_cos", "SEED"])
    lk = feats[ZL.LK_KEY]
    B, N = qs.size, nq.numel()
    cnt = c.n_np[qs]
    seg = np.cumsum(cnt) - cnt
    idx = np.repeat(c.off_np[qs] - seg, cnt) + np.arange(N)
    P = c.decode(c.row[torch.from_numpy(idx)])[:, :128]
    P = P / P.norm(dim=1, keepdim=True).clamp_min(1e-9)
    rank = ZL.pool_rank(base_z, nq, B)
    top = rank < K
    u, v, f = lk["u"], lk["v"], lk["f"]
    elig = top[u] & ~top[v]
    gold = gold.bool()
    miss = gold & ~top
    nmiss = torch.zeros(B, dtype=torch.int64).index_add_(0, nq, miss.to(torch.int64))
    fam_reach = []
    for fam in range(FAMS):
        hit = torch.zeros(N, dtype=torch.bool)
        hit[v[elig & (f == fam)]] = True
        fam_reach.append(torch.zeros(B, dtype=torch.int64).index_add_(0, nq, (hit & miss).to(torch.int64)))
    anyhit = torch.zeros(N, dtype=torch.bool)
    anyhit[v[elig]] = True
    reach_any = torch.zeros(B, dtype=torch.int64).index_add_(0, nq, (anyhit & miss).to(torch.int64))
    deg = torch.bincount(v, minlength=N).to(torch.float32)
    ue, ve, fe = u[elig], v[elig], f[elig]
    key = ue * N + ve
    uk, inv = torch.unique(key, return_inverse=True)
    pu, pv = uk // N, uk % N
    famflag = torch.zeros(uk.numel(), FAMS).index_add_(0, inv, torch.nn.functional.one_hot(fe, FAMS).to(torch.float32)).clamp_max(1.0)
    cosv = (P[pu] * P[pv]).sum(1)
    soft = 1.0 / (1.0 + rank[pu].to(torch.float32))
    rows = torch.unique(pv)
    pos = torch.full((N,), -1, dtype=torch.int64)
    pos[rows] = torch.arange(rows.numel())
    pr = pos[pv]
    n_r = rows.numel()
    cmax = torch.full((n_r,), -2.0).scatter_reduce(0, pr, cosv, reduce="amax", include_self=True)
    nsend = torch.zeros(n_r).index_add_(0, pr, torch.ones(pr.numel()))
    softs = torch.zeros(n_r).index_add_(0, pr, soft)
    fam_any = torch.zeros(n_r, FAMS).index_add_(0, pr, famflag).clamp_max(1.0)
    seed = feats["SEED"]
    cols = [base_z[rows], feats["dense_cos"][rows, 0], seed[rows, 0], seed[rows, 1], seed[rows, 2], cmax, torch.log1p(nsend), softs,
            torch.log1p(deg[rows]), fam_any[:, 0], fam_any[:, 1], fam_any[:, 2]]
    M = torch.stack([x.to(torch.float32) for x in cols], 1)
    out = {"X": M.numpy(), "y": gold[rows].numpy(), "q": qs[nq[rows].numpy()], "n_miss": nmiss.numpy(), "reach_any": reach_any.numpy(),
           "reach_fam": np.stack([x.numpy() for x in fam_reach], 1)}
    if full:                                                  # the lean pick's columns, raw and as the model sees them (z-score over the whole pool)
        import lean_gpu as LG
        raw = torch.cat([feats[b] for b in FULL_BLOCKS], 1)
        z = LG.seg_zscore8D(raw, nq, B)
        out["XN"] = torch.cat([raw[rows], z[rows]], 1).numpy()
    return out


def edge_tables(c, nq, K, step=16, full=False):
    parts = [receivers(c, np.arange(k, min(k + step, nq)), K, full) for k in range(0, nq, step)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def edge_report(t, K):
    X, y, q = t["X"], t["y"], t["q"]
    tot, any_, fam = int(t["n_miss"].sum()), int(t["reach_any"].sum()), t["reach_fam"].sum(0)
    out = {"K": K, "missing_golds": tot, "one_hop_reach": any_ / max(tot, 1), "one_hop_reach_by_family": [float(x) / max(tot, 1) for x in fam],
           "receivers": int(y.size), "gold_receivers": int(y.sum()), "questions_with_gold_receiver": int(len(np.unique(q[y])))}
    if y.sum() == 0:
        return out
    name = {n: i for i, n in enumerate(EDGE_NAMES)}
    sc = {"base_z": X[:, name["base_z"]], "dense_cos": X[:, name["dense_cos"]], "seed0": X[:, name["seed0"]],
          "cos_max": X[:, name["cos_max"]], "soft_senders": X[:, name["soft_senders"]], "neg_log_deg": -X[:, name["log_deg"]]}
    out["scalars"] = {k: {"auc": auc(v, y), **hit_at(v, y, q)} for k, v in sc.items()}
    node = [name[n] for n in NODE_ONLY]
    s_node, w_node = lr_crossfit(X[:, node], y, q)
    s_all, w_all = lr_crossfit(X, y, q)
    out["lr_node_only"] = {"auc": auc(s_node, y), **hit_at(s_node, y, q), "weights": dict(zip(NODE_ONLY, [round(float(w), 3) for w in w_node]))}
    out["lr_node_plus_edge"] = {"auc": auc(s_all, y), **hit_at(s_all, y, q), "weights": dict(zip(EDGE_NAMES, [round(float(w), 3) for w in w_all]))}
    if "XN" in t:                                    # amendment A1: against every column of the lean pick, raw and z-scored
        XN = t["XN"]
        ed = [name[n] for n in EDGE_ONLY]
        s_f, _w = lr_crossfit(XN, y, q)
        s_fe, w_fe = lr_crossfit(np.concatenate([XN, X[:, ed]], 1), y, q)
        out["lr_full"] = {"auc": auc(s_f, y), **hit_at(s_f, y, q), "columns": int(XN.shape[1])}
        out["lr_full_plus_edge"] = {"auc": auc(s_fe, y), **hit_at(s_fe, y, q),
                                    "edge_weights": dict(zip(EDGE_ONLY, [round(float(w), 3) for w in w_fe[-len(ed):]]))}
    return out


# ── amendment A2: similarity to the first stage's top rows, over every pool row ───────────────────────────


def recall_at(score, y, q, ms=(5, 20)):
    """Per question with a gold: the share of its golds among its top m rows by score, averaged over those questions. Ties by position."""
    score = np.asarray(score, np.float64)
    y = np.asarray(y, bool)
    q = np.asarray(q, np.int64)
    o = np.lexsort((np.arange(q.size), -score, q))
    qs, ys = q[o], y[o]
    first = np.r_[True, qs[1:] != qs[:-1]]
    grp = np.cumsum(first) - 1
    start = np.flatnonzero(first)
    pos = np.arange(qs.size) - start[grp]
    g = np.bincount(grp, weights=ys.astype(np.float64), minlength=start.size)
    has = g > 0
    out = {"questions": int(has.sum())}
    for m in ms:
        got = np.bincount(grp, weights=(ys & (pos < m)).astype(np.float64), minlength=start.size)
        out[f"recall@{m}"] = float((got[has] / g[has]).mean()) if has.any() else None
    return out


PRF_NAMES = ["prf5", "prf20", "maxcos5"]


def prf_batch(c, qs):
    """For questions qs, every pool row: the lean pick's columns (raw, then z-scored over the pool), the three first-stage-relative columns
    (cosine of the row's stored 128-wide projection to the centroid of the top 5 and of the top 20 rows by rrf, and the largest cosine to
    another of the top 5), raw then z-scored, the labels, and the rrf rank."""
    import lean_gpu as LG
    import zlink as ZL
    feats, nq, base_z, gold = c.batch(qs, FULL_BLOCKS)
    B, N = qs.size, nq.numel()
    cnt = c.n_np[qs]
    seg = np.cumsum(cnt) - cnt
    idx = np.repeat(c.off_np[qs] - seg, cnt) + np.arange(N)
    P = c.decode(c.row[torch.from_numpy(idx)])[:, :128]
    P = P / P.norm(dim=1, keepdim=True).clamp_min(1e-9)
    rank = ZL.pool_rank(base_z, nq, B)
    sel = rank < 20
    top = torch.full((B, 20), -1, dtype=torch.int64)
    top[nq[sel], rank[sel]] = torch.nonzero(sel).squeeze(1)
    assert bool((top >= 0).all()), "a pool has fewer than 20 rows, or rrf's rank has ties"
    c5 = torch.nn.functional.normalize(P[top[:, :5]].mean(1), dim=1)
    c20 = torch.nn.functional.normalize(P[top].mean(1), dim=1)
    prf5, prf20 = (P * c5[nq]).sum(1), (P * c20[nq]).sum(1)
    mx = torch.full((N,), -2.0)
    for k in range(5):
        s = (P * P[top[:, k]][nq]).sum(1)
        s = torch.where(rank == k, torch.full_like(s, -2.0), s)          # a top row is not compared with itself
        mx = torch.maximum(mx, s)
    D = torch.stack([prf5, prf20, mx], 1)
    raw = torch.cat([feats[b] for b in FULL_BLOCKS], 1)
    XN = torch.cat([raw, LG.seg_zscore8D(raw, nq, B)], 1)
    DN = torch.cat([D, LG.seg_zscore8D(D, nq, B)], 1)
    return {"XN": XN.numpy(), "D": DN.numpy(), "y": gold.bool().numpy(), "q": qs[nq.numpy()], "rank": rank.numpy()}


def prf_cmd(a):
    t0 = time.time()
    c = open_carve(a.dataset)
    nq = min(a.nq or NQ[a.dataset], c.rows)
    parts = [prf_batch(c, np.arange(k, min(k + 16, nq))) for k in range(0, nq, 16)]
    t = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    y, q, rank = t["y"], t["q"], t["rank"]
    far = rank >= 5
    out = {"declared_in": DECLARED, "amendment": "A2", "dataset": a.dataset, "carve": c.carve, "questions": nq, "rows": int(y.size),
           "gold_rows": int(y.sum()), "golds_beyond_top5": int(y[far].sum()), "part_shas": c.part_shas, "columns": {}}
    for j, nm in enumerate(PRF_NAMES):
        v = t["D"][:, 3 + j]                                            # within-question z-score, as the columns' AUCs of section 2
        out["columns"][nm] = {"auc_z": auc(v, y), "auc_z_beyond_top5": auc(v[far], y[far])}
    s0, _w = lr_crossfit(t["XN"], y, q)
    s1, w1 = lr_crossfit(np.concatenate([t["XN"], t["D"]], 1), y, q)
    for nm, s in (("lr_lean", s0), ("lr_lean_plus_prf", s1)):
        out[nm] = {"auc": auc(s, y), **hit_at(s, y, q), **{k: v for k, v in recall_at(s, y, q).items() if k != "questions"}}
    out["prf_weights"] = dict(zip(PRF_NAMES + [n + "_z" for n in PRF_NAMES], [round(float(w), 3) for w in w1[-6:]]))
    out["seconds"] = time.time() - t0
    p = Path(OUT) / f"prf_{a.dataset}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"prf {a.dataset}: {nq} questions, {y.size} rows, golds beyond top 5 {out['golds_beyond_top5']} ({out['seconds']:.0f}s) -> {p}")
    for nm in PRF_NAMES:
        log(f"   {nm}: auc {out['columns'][nm]['auc_z']:.3f} beyond top 5 {out['columns'][nm]['auc_z_beyond_top5']:.3f}")
    for nm in ("lr_lean", "lr_lean_plus_prf"):
        r = out[nm]
        log(f"   {nm}: auc {r['auc']:.3f} recall@5 {r['recall@5']:.3f} recall@20 {r['recall@20']:.3f} hit@5 {r['hit@5']:.3f} (chance {r['chance@5']:.3f})")
    return 0


def edges_cmd(a):
    t0 = time.time()
    c = open_carve(a.dataset)
    nq = min(a.nq or NQ[a.dataset], c.rows)
    ks = [int(x) for x in a.k.split(",")]
    out = {"declared_in": DECLARED, "dataset": a.dataset, "carve": c.carve, "questions": nq, "link_edges": int(c.links["e_u"].size),
           "full": bool(a.full), "part_shas": c.part_shas, "by_k": {}}
    for K in ks:
        t = edge_tables(c, nq, K, full=a.full)
        r = edge_report(t, K)
        out["by_k"][str(K)] = r
        log(f"edges {a.dataset} K={K}: missing golds {r['missing_golds']}, one-hop reach {r['one_hop_reach']:.3f} "
            f"(by family {[round(x, 3) for x in r['one_hop_reach_by_family']]}), receivers {r['receivers']}, gold receivers {r['gold_receivers']}")
        for k in ("lr_node_only", "lr_node_plus_edge", "lr_full", "lr_full_plus_edge"):
            if k in r:
                log(f"   {k}: auc {r[k]['auc']:.3f} hit@5 {r[k]['hit@5']:.3f} (chance {r[k]['chance@5']:.3f}) hit@20 {r[k]['hit@20']:.3f}")
    out["seconds"] = time.time() - t0
    p = Path(OUT) / f"edges{'_full' if a.full else ''}_{a.dataset}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"edges {a.dataset}: {nq} questions ({out['seconds']:.0f}s) -> {p}")
    return 0


# ── the self-test ───────────────────────────────────────────────────────────


class FakeCarve:
    """Three questions of 60 rows. Gold row 1 sits next to a top row by a family-0 edge and shares its direction; decoys do not."""

    def __init__(self):
        import zlink as ZL
        self.n_np = np.array([60, 60, 60])
        self.off_np = np.r_[0, np.cumsum(self.n_np)]
        rng = np.random.default_rng(3)
        self.E = torch.tensor(rng.standard_normal((180, 128)), dtype=torch.float32)
        self.gold_t = torch.zeros(180, dtype=torch.bool)
        e_u, e_v, e_f, q_ne = [], [], [], []
        for q in range(3):
            o = q * 60
            self.gold_t[o + 0] = True
            self.gold_t[o + 30] = True                                     # the missing gold, outside the top K
            self.E[o + 30] = self.E[o + 0] * 0.9 + 0.1 * self.E[o + 30]  # near its sender
            ee = [(0, 30, 0), (30, 0, 0)] + [(0, 31 + i, 1) for i in range(10)] + [(31 + i, 0, 1) for i in range(10)]
            e_u += [a for a, _b, _c in ee]
            e_v += [b for _a, b, _c in ee]
            e_f += [cc for _a, _b, cc in ee]
            q_ne.append(len(ee))
        self.links = {"e_u": np.array(e_u, np.int16), "e_v": np.array(e_v, np.int16), "e_fam": np.array(e_f, np.int8),
                      "q_ne": np.array(q_ne, np.int64), "e_off": np.r_[0, np.cumsum(q_ne)]}
        self.row = torch.arange(180)
        self.rows = 3
        self.ZL = ZL

    def decode(self, rows):
        return torch.cat([self.E[rows], torch.ones(rows.numel(), 1)], 1)

    def batch(self, qs, blocks):
        qs = np.asarray(qs, np.int64)
        cnt = self.n_np[qs]
        N = int(cnt.sum())
        idx = np.concatenate([np.arange(self.off_np[q], self.off_np[q + 1]) for q in qs])
        nq = torch.from_numpy(np.repeat(np.arange(qs.size), cnt))
        bz = torch.zeros(N)
        for i, q in enumerate(qs):
            bz[i * 60:(i + 1) * 60] = -torch.arange(60, dtype=torch.float32)       # row 0 is the first stage's top, then 1, 2, ...
        g = torch.Generator().manual_seed(5)
        feats = {"dense_cos": torch.rand(N, 1, generator=g), "SEED": torch.rand(N, 3, generator=g), "rank": torch.rand(N, 5, generator=g),
                 "topo_STRUCT": torch.rand(N, 4, generator=g), "depth_STRUCT": torch.rand(N, 2, generator=g), "WALK": torch.rand(N, 6, generator=g),
                 "WALKF": torch.rand(N, 3, generator=g), "DISTS": torch.rand(N, 2, generator=g)}
        ne = self.links["q_ne"][qs]
        eix = np.concatenate([np.arange(self.links["e_off"][q], self.links["e_off"][q + 1]) for q in qs])
        eq = np.repeat(np.arange(qs.size), ne)
        feats[self.ZL.LK_KEY] = {"u": torch.from_numpy(eq * 60 + self.links["e_u"][eix].astype(np.int64)),
                                 "v": torch.from_numpy(eq * 60 + self.links["e_v"][eix].astype(np.int64)),
                                 "f": torch.from_numpy(self.links["e_fam"][eix].astype(np.int64))}
        return feats, nq, bz, self.gold_t[torch.from_numpy(idx)]


def selftest():
    assert abs(auc([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-12 and abs(auc([1, 1, 1, 1], [0, 0, 1, 1]) - 0.5) < 1e-12
    assert abs(auc([3, 2, 1, 0], [0, 0, 1, 1]) - 0.0) < 1e-12 and auc([1, 2], [0, 0]) is None
    h = hit_at([0.9, 0.1, 0.5, 0.8, 0.2, 0.1], [1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1], ms=(1, 2))
    assert h["questions"] == 2 and abs(h["hit@1"] - 0.5) < 1e-12 and abs(h["hit@2"] - 0.5) < 1e-12, h
    assert abs(h["chance@1"] - (1 / 3 + 1 / 3) / 2) < 1e-12 and abs(h["chance@2"] - (2 / 3 + 2 / 3) / 2) < 1e-12, h
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 3))
    y = (X[:, 0] + 0.3 * rng.standard_normal(4000)) > 1.0
    q = np.arange(4000) // 20
    s, w = lr_crossfit(X, y, q)
    assert auc(s, y) > 0.9 and w[0] > abs(w[1]) * 3 and w[0] > abs(w[2]) * 3, (auc(s, y), w)
    c = FakeCarve()
    t = edge_tables(c, 3, K=1, step=2)
    assert t["n_miss"].tolist() == [1, 1, 1] and t["reach_any"].tolist() == [1, 1, 1], t
    assert t["reach_fam"].tolist() == [[1, 0, 0]] * 3, t["reach_fam"]
    assert t["y"].sum() == 3 and t["X"].shape[1] == len(EDGE_NAMES) and set(t["q"].tolist()) == {0, 1, 2}
    j = EDGE_NAMES.index("cos_max")
    assert float(t["X"][t["y"], j].min()) > 0.7 > float(t["X"][~t["y"], j].max()), "the gold receiver's cosine to its sender"
    assert EDGE_NAMES.index("fam0") == 9 and t["X"][t["y"], 9].min() == 1.0 and t["X"][t["y"], 10].max() == 0.0
    tf = edge_tables(c, 3, K=1, step=2, full=True)
    assert tf["XN"].shape == (t["X"].shape[0], 2 * 26) and np.array_equal(tf["X"], t["X"]), tf["XN"].shape
    rf = edge_report(tf, 1)
    assert set(rf) >= {"lr_full", "lr_full_plus_edge"} and rf["lr_full"]["columns"] == 52 and "cos_max" in rf["lr_full_plus_edge"]["edge_weights"], rf.keys()
    r = edge_report(t, 1)
    assert r["one_hop_reach"] == 1.0 and r["scalars"]["cos_max"]["auc"] > 0.99 and r["scalars"]["cos_max"]["hit@5"] == 1.0, r
    ra = recall_at([0.9, 0.1, 0.5, 0.8, 0.2, 0.1], [1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1], ms=(1, 2))
    assert ra["questions"] == 2 and abs(ra["recall@1"] - 0.5) < 1e-12 and abs(ra["recall@2"] - 0.5) < 1e-12, ra
    pb = prf_batch(c, np.array([0, 1, 2]))
    assert pb["XN"].shape == (180, 2 * 26) and pb["D"].shape == (180, 6) and pb["y"].sum() == 6, (pb["XN"].shape, pb["D"].shape)
    mxc = pb["D"][:, 2]
    gold30 = np.flatnonzero(pb["y"] & (pb["rank"] >= 5))
    assert gold30.size == 3 and mxc[gold30].min() > 0.7 > mxc[(~pb["y"]) & (pb["rank"] >= 5)].max(), (mxc[gold30], mxc[~pb["y"]].max())
    assert (pb["D"][pb["rank"] == 0, 2] < 0.7).all(), "a top row is not compared with itself"
    log("selftest: auc, hit_at and recall_at (with their chance), the logistic cross-fit, the receivers on a three-question carve (reach, "
        "families, the sender cosine), the edge report and the first-stage-relative columns all pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for nm in ("columns", "edges", "prf"):
        p = sub.add_parser(nm)
        p.add_argument("--dataset", required=True, choices=sorted(CARVE))
        p.add_argument("--nq", type=int, default=0)
        p.add_argument("--threads", type=int, default=6)
        if nm == "edges":
            p.add_argument("--k", default="5,20")
            p.add_argument("--full", action="store_true", help="amendment A1: also the lean pick's columns as the node baseline")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.error("a command or --selftest")
    torch.set_num_threads(a.threads)
    return {"columns": columns_cmd, "edges": edges_cmd, "prf": prf_cmd}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
