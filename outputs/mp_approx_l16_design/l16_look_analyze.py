"""Design look for a 2wiki typed-walk level (untracked; not a result and not filed), on what l16_look_score.py kept for
carve x1 (2wiki train-split rows the twin and the GNN never trained on). Descriptive only: it compares relation-token
schemes for a non-message-passing typed-walk model on 2wiki, which has no relation slots (its structural family is
one 'hyperlink' relation), so a relation type must be discovered from what the GNN also reads and then frozen.

Schemes (each walk of 1 or 2 edges from the bucket-0 or the bucket-1 seeds, non-backtracking, a type = (bucket,
token sequence), its reach set uniform):
  T0     family x direction: structural fwd-only, bwd-only, both; ner; knn (5 tokens)
  N{k}   T0 x the target node's cluster (spherical k-means, k = 4, 8, 16, on the fixed 128-d projection of the pool
         nodes of half A)
  D{k}   T0 x the cluster of the edge's endpoint difference (k-means on half A's structural edges, k = 8, 16;
         ner and knn keep T0's token)
  W2     T0 with ner and knn split at half A's median weight (7 tokens)

Per scheme it reports reach statistics, a gold-chosen single-type ceiling (optimistic), and a quick fit: a
query-weighted mixture over the row's types and a null type, s(v) = z_twin(v) + kappa * sum_tau p(tau | q)
1[v in R_tau] / |R_tau|^beta, with p a softmax of <A q, e_tau>, e_tau = bucket + position-1 token + position-2 token +
length (no free per-type vector), fitted on half A (even rows) by a listwise softmax cross-entropy and read on half B
(odd rows) against twin seed 0 and GNN seed 0. A diagnostic reports how many of the golds the GNN brings into its top 5
lie one or two edges from a seed, by token.

    python outputs/mp_approx_l16_design/l16_look_analyze.py [--look DIR] [--schemes T0,N8,...]
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
FUNCS = ("twin0", "twin1", "twin2", "gnn0", "gnn1", "gnn2")
METRICS = ("recall@1", "recall@5", "recall@10", "recall@20", "hit@1", "mrr", "ndcg@5", "ndcg@20", "full_coverage@5",
           "full_coverage@20", "first_gold_rank", "gold_in_pool", "gold_total", "pool_size")
RI = [METRICS.index(m) for m in ("recall@5", "full_coverage@5", "hit@1")]
FAMDIR = ("S-fwd", "S-bwd", "S-both", "NER", "KNN")
TYPES2W = ("compositional", "inference", "comparison", "bridge_comparison")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(look: Path):
    ids = json.loads((look / "ids.json").read_text(encoding="utf-8"))
    info = json.loads((look / "info.json").read_text(encoding="utf-8"))
    Q = [None] * len(ids)
    paths = sorted((look / "chunks").glob("c*.npz"))
    all_rows = np.sort(np.concatenate([np.load(p)["chunk_rows"] for p in paths]))   # the scored rows, ascending = ids order
    if all_rows.size != len(ids) or np.unique(all_rows).size != all_rows.size:
        raise SystemExit("the chunks do not hold each scored row once")
    for p in paths:
        with np.load(p) as zf:
            z = {key: zf[key] for key in zf.files}   # each array read once; every slice below is a copy, so no row keeps a chunk alive
        npool, ned = z["q_pool_size"], z["q_edges"]
        po, eo = np.r_[0, np.cumsum(npool)], np.r_[0, np.cumsum(ned)]
        for i, row in enumerate(z["chunk_rows"]):
            a, b = po[i], po[i + 1]
            c, d = eo[i], eo[i + 1]
            Q[int(np.searchsorted(all_rows, row))] = {
                "n": int(npool[i]), "gt": int(z["q_gold_total"][i]), "metrics": z["q_metrics"][i].copy(), "qemb": z["q_emb"][i].copy(),
                "seeds": z["q_seed_local"][i].copy(), "bucket": z["q_seed_bucket"][i].copy(), "pool": z["pool"][a:b].copy(),
                "score": z["score"][a:b].copy(), "gold": z["is_gold"][a:b].copy(), "proj": z["proj"][a:b].astype(np.float32),
                "u": z["e_u"][c:d].astype(np.int64), "v": z["e_v"][c:d].astype(np.int64), "fam": z["e_fam"][c:d].astype(np.int64),
                "fwd": z["e_fwd"][c:d].astype(bool), "bwd": z["e_bwd"][c:d].astype(bool), "w": z["e_w"][c:d].copy()}
        del z
    missing = [i for i, q in enumerate(Q) if q is None]
    if missing:
        raise SystemExit(f"{len(missing)} rows missing from the chunks")
    for q, inf in zip(Q, info):
        q["type"] = inf["type"]
    return ids, Q


def metrics_of(score, gold, gt):
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    rank = np.empty(score.size, dtype=np.int64)
    rank[order] = np.arange(1, score.size + 1)
    gr = np.sort(rank[np.flatnonzero(gold)])
    total = max(gt, 1)
    first = int(gr[0]) if gr.size else 0
    return np.asarray([float((gr <= 5).sum()) / total, float((gr <= 5).sum() == total), float(first == 1)])


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


# ── tokens ───────────────────────────────────────────────────────────────────


def famdir(q):
    f = q["fam"]
    t = np.where(f == 1, 3, np.where(f == 2, 4, np.where(q["fwd"] & q["bwd"], 2, np.where(q["fwd"], 0, 1))))
    return t.astype(np.int64)


def kmeans_sph(X, k, seed=0, iters=30):
    rng = np.random.default_rng(seed)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    C = [X[rng.integers(X.shape[0])]]
    d2 = 1 - X @ C[0]
    for _ in range(1, k):
        p = np.maximum(d2, 0) / max(np.maximum(d2, 0).sum(), 1e-12)
        C.append(X[rng.choice(X.shape[0], p=p)])
        d2 = np.minimum(d2, 1 - X @ C[-1])
    C = np.stack(C)
    for _ in range(iters):
        lab = np.argmax(X @ C.T, 1)
        for j in range(k):
            m = lab == j
            if m.any():
                c = X[m].sum(0)
                C[j] = c / max(np.linalg.norm(c), 1e-12)
    return C


def kmeans_l2(X, k, seed=0, iters=30):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(X.shape[0])]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        C.append(X[rng.choice(X.shape[0], p=d2 / max(d2.sum(), 1e-12))])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = np.stack(C)
    for _ in range(iters):
        lab = nearest_l2(X, C)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return C


def nearest_l2(X, C):
    """argmin_j ||x - c_j||^2 without an (n, k, d) array"""
    return np.argmin((C * C).sum(1)[None, :] - 2.0 * (X @ C.T), 1)


def schemes_tokens(Q, A_rows, names):
    """{scheme: (n_tokens, [per-query token per edge])}, the clusters fitted on half A only."""
    out = {}
    base = [famdir(q) for q in Q]
    if "T0" in names:
        out["T0"] = (5, base)
    for name in names:
        if name.startswith("N"):
            k = int(name[1:])
            seen, X = set(), []
            for i in A_rows:
                q = Q[i]
                for j, g in enumerate(q["pool"]):
                    if int(g) not in seen:
                        seen.add(int(g))
                        X.append(q["proj"][j])
            X = np.asarray(X, dtype=np.float32)
            if X.shape[0] > 120000:
                X = X[np.random.default_rng(1).choice(X.shape[0], 120000, replace=False)]
            C = kmeans_sph(X, k)
            toks = []
            for q, b in zip(Q, base):
                P = q["proj"] / np.maximum(np.linalg.norm(q["proj"], axis=1, keepdims=True), 1e-12)
                lab = np.argmax(P @ C.T, 1)
                toks.append(b * k + lab[q["v"]])
            out[name] = (5 * k, toks)
        elif name.startswith("D"):
            k = int(name[1:])
            X = []
            for i in A_rows:
                q = Q[i]
                m = q["fam"] == 0
                X.append(q["proj"][q["v"][m]] - q["proj"][q["u"][m]])
            X = np.concatenate(X).astype(np.float32)
            if X.shape[0] > 60000:
                X = X[np.random.default_rng(2).choice(X.shape[0], 60000, replace=False)]
            C = kmeans_l2(X, k)
            toks = []
            for q, b in zip(Q, base):
                t = b.copy()
                m = q["fam"] == 0
                if m.any():
                    Dv = q["proj"][q["v"][m]] - q["proj"][q["u"][m]]
                    lab = nearest_l2(Dv, C)
                    t[m] = 2 + b[m] * k + lab   # 0, 1 = ner, knn; structural tokens follow
                t[q["fam"] == 1] = 0
                t[q["fam"] == 2] = 1
                toks.append(t)
            out[name] = (2 + 3 * k, toks)
        elif name == "W2":
            wn = np.median(np.concatenate([Q[i]["w"][Q[i]["fam"] == 1] for i in A_rows]))
            wk = np.median(np.concatenate([Q[i]["w"][Q[i]["fam"] == 2] for i in A_rows]))
            toks = []
            for q, b in zip(Q, base):
                t = b.copy()
                t[(q["fam"] == 1) & (q["w"] > wn)] = 5
                t[(q["fam"] == 2) & (q["w"] > wk)] = 6
                toks.append(t)
            out["W2"] = (7, toks)
    return out


def walk_types(q, tok, nt, max_len=2):
    """Non-backtracking walks of 1 and 2 edges from each bucket's seeds; returns {code: reach set (sorted nodes)} with
    code = (b * (nt + 1) + t1 + 1) * (nt + 1) + (t2 + 1, 0 for a 1-edge walk)."""
    u, v = q["u"], q["v"]
    order = np.argsort(u, kind="stable")
    u, v, tok = u[order], v[order], tok[order]
    n = q["n"]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(u, minlength=n))
    types = {}
    tb = nt + 1
    for b in (0, 1):
        S = np.unique(q["seeds"][(q["bucket"] == b) & (q["seeds"] >= 0)])
        if S.size == 0:
            continue
        deg = indptr[S + 1] - indptr[S]
        src = np.repeat(S, deg)
        e1 = np.concatenate([np.arange(indptr[s], indptr[s + 1]) for s in S]) if deg.sum() else np.zeros(0, np.int64)
        t1, v1 = tok[e1], v[e1]
        c1 = (b * tb + t1 + 1) * tb
        for c in np.unique(c1):
            types[int(c)] = np.unique(v1[c1 == c])
        deg2 = indptr[v1 + 1] - indptr[v1]
        if max_len < 2 or deg2.sum() == 0:
            continue
        rep = np.repeat(np.arange(v1.size), deg2)
        e2 = np.repeat(indptr[v1] - (np.cumsum(deg2) - deg2), deg2) + np.arange(int(deg2.sum()))
        keep = v[e2] != src[rep]
        rep, e2 = rep[keep], e2[keep]
        c2 = c1[rep] + tok[e2] + 1
        w2 = v[e2]
        o = np.argsort(c2, kind="stable")
        c2, w2 = c2[o], w2[o]
        if c2.size == 0:   # every 2-edge walk went back to its seed
            continue
        starts = np.flatnonzero(np.r_[True, c2[1:] != c2[:-1]])
        ends = np.r_[starts[1:], c2.size]
        for s0, s1 in zip(starts, ends):
            types[int(c2[s0])] = np.unique(w2[s0:s1])
    return types


# ── the quick fit ────────────────────────────────────────────────────────────


class Mix(torch.nn.Module):
    def __init__(self, nt, d=64, qdim=1536):
        super().__init__()
        self.A = torch.nn.Linear(qdim, d, bias=False)
        self.bk = torch.nn.Parameter(torch.zeros(2, d))
        self.t1 = torch.nn.Parameter(torch.randn(nt, d) * 0.1)
        self.t2 = torch.nn.Parameter(torch.randn(nt + 1, d) * 0.1)
        self.ln = torch.nn.Parameter(torch.zeros(2, d))
        self.c = torch.nn.Parameter(torch.zeros(2, 2))
        self.nu = torch.nn.Parameter(torch.zeros(d))
        self.c_null = torch.nn.Parameter(torch.zeros(()))
        self.log_kappa = torch.nn.Parameter(torch.tensor(0.0))
        self.beta_raw = torch.nn.Parameter(torch.tensor(0.0))
        self.nt = nt

    def forward(self, qemb, tb, t1, t2, tmask, ent_q, ent_t, ent_node, z, nmask):
        aq = self.A(qemb)                                         # (B, d)
        L = (t2 > 0).long()
        e = self.bk[tb] + self.t1[t1] + self.t2[t2] + self.ln[L]   # (B, T, d)
        w = (aq[:, None, :] * e).sum(-1) + self.c[tb, L]
        w = w.masked_fill(~tmask, float("-inf"))
        null = aq @ self.nu + self.c_null
        logits = torch.cat([w, null[:, None]], 1)
        p = torch.softmax(logits, 1)[:, :-1]                       # (B, T)
        return p


def pack(rows, Q, TY, z_of, nt):
    B = len(rows)
    T = max(1, max(len(TY[i]) for i in rows))
    N = max(Q[i]["n"] for i in rows)
    tb = torch.zeros(B, T, dtype=torch.long)
    t1 = torch.zeros(B, T, dtype=torch.long)
    t2 = torch.zeros(B, T, dtype=torch.long)
    tmask = torch.zeros(B, T, dtype=torch.bool)
    eq, et, en, ew = [], [], [], []
    z = torch.full((B, N), float("-inf"))
    gold = torch.zeros(B, N)
    tb1 = nt + 1
    for bi, i in enumerate(rows):
        codes = sorted(TY[i])
        for ti, c in enumerate(codes):
            pre, last = divmod(c, tb1)
            b, first = divmod(pre, tb1)
            tb[bi, ti], t1[bi, ti], t2[bi, ti], tmask[bi, ti] = b, first - 1, last, True
            R = TY[i][c]
            eq.append(np.full(R.size, bi))
            et.append(np.full(R.size, ti))
            en.append(R)
            ew.append(np.full(R.size, R.size, dtype=np.float32))
        n = Q[i]["n"]
        z[bi, :n] = torch.as_tensor(z_of[i], dtype=torch.float32)
        gold[bi, :n] = torch.as_tensor(Q[i]["gold"], dtype=torch.float32)
    cat = (lambda xs, dt: torch.as_tensor(np.concatenate(xs) if xs else np.zeros(0), dtype=dt))
    return (torch.as_tensor(np.stack([Q[i]["qemb"] for i in rows])), tb, t1, t2, tmask, cat(eq, torch.long), cat(et, torch.long),
            cat(en, torch.long), cat(ew, torch.float32), z, gold)


def scores_of(model, P):
    qemb, tb, t1, t2, tmask, eq, et, en, ew, z, gold = P
    p = model(qemb, tb, t1, t2, tmask, None, None, None, z, None)
    kappa = torch.exp(model.log_kappa)
    beta = torch.nn.functional.softplus(model.beta_raw)
    contrib = p[eq, et] * ew.pow(-beta)
    boost = torch.zeros_like(z).index_put((eq, en), contrib, accumulate=True)
    s = torch.where(torch.isfinite(z), z + kappa * boost, z)
    return s, gold


def fit_read(Q, TY, nt, A_rows, B_rows, z_of, epochs=12, seed=0, log_every=False):
    torch.manual_seed(seed)
    model = Mix(nt)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    val = A_rows[::8]
    vs = set(val)
    tr = [i for i in A_rows if i not in vs]
    best, best_state = -1.0, None
    for ep in range(epochs):
        perm = rng.permutation(len(tr))
        model.train()
        for s0 in range(0, len(tr), 128):
            rows = [tr[j] for j in perm[s0:s0 + 128]]
            P = pack(rows, Q, TY, z_of, nt)
            s, gold = scores_of(model, P)
            ls = torch.log_softmax(s, 1)
            g = gold / gold.sum(1, keepdim=True).clamp(min=1)
            loss = -(torch.where(gold > 0, ls, torch.zeros_like(ls)) * g).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = read_rows(model, Q, TY, nt, val, z_of)
        score = float(m[:, :2].mean())
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, read_rows(model, Q, TY, nt, B_rows, z_of)


def read_rows(model, Q, TY, nt, rows, z_of):
    model.eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(rows), 256):
            rr = rows[s0:s0 + 256]
            P = pack(rr, Q, TY, z_of, nt)
            s, _gold = scores_of(model, P)
            for bi, i in enumerate(rr):
                out.append(metrics_of(s[bi, :Q[i]["n"]].numpy(), Q[i]["gold"], Q[i]["gt"]))
    return np.asarray(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default=str(HERE / "look" / "x1"))
    ap.add_argument("--schemes", default="T0-1,T0,W2,N4,N8,N8-1,N16,D8,D16")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default=str(HERE / "l16_look_analyze.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    t0 = time.time()
    ids, Q = load(Path(a.look))
    n = len(Q)
    A_rows, B_rows = list(range(0, n, 2)), list(range(1, n, 2))
    log(f"{n} rows loaded ({len(A_rows)} A, {len(B_rows)} B) in {time.time() - t0:.0f}s")
    res = {"rows": n, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # baselines and the stored-metric check
    M = np.stack([q["metrics"] for q in Q])[:, :, RI]           # (n, 6, 3)
    for i in range(0, n, 97):
        q = Q[i]
        for f in (0, 3):
            if not np.array_equal(metrics_of(q["score"][:, f], q["gold"], q["gt"]), M[i, f]):
                raise SystemExit(f"row {i}: metrics_of differs from the stored metrics of {FUNCS[f]}")
    ty = np.asarray([q["type"] for q in Q])
    base = {}
    for name, sel in [("all", np.ones(n, bool))] + [(t, ty == t) for t in TYPES2W]:
        base[name] = {"rows": int(sel.sum()), **{f: [round(float(x), 4) for x in M[sel, fi].mean(0)] for fi, f in enumerate(FUNCS)}}
    res["baselines (R@5, FC@5, hit@1)"] = base
    log("baselines: " + json.dumps(base["all"]))
    Bsel = np.asarray(B_rows)
    tB, gB = M[Bsel, 0], M[Bsel, 3]
    res["B twin0 / gnn0"] = {"twin0": tB.mean(0).round(4).tolist(), "gnn0": gB.mean(0).round(4).tolist()}
    z_of = [zscore(q["score"][:, 0]) for q in Q]
    # the GNN's lifted golds: golds in the GNN's top 5 and not in the twin's, by distance from a seed and token (T0)
    lift = {"golds_lifted": 0, "one_edge_from_seed": {}, "two_edges_only": 0, "farther": 0}
    for q in Q:
        rt = np.argsort(-q["score"][:, 0].astype(np.float64), kind="stable")[:5]
        rg = np.argsort(-q["score"][:, 3].astype(np.float64), kind="stable")[:5]
        new = [g for g in np.flatnonzero(q["gold"]) if g in set(rg.tolist()) and g not in set(rt.tolist())]
        if not new:
            continue
        S = set(q["seeds"][q["seeds"] >= 0].tolist())
        fd = famdir(q)
        for g in new:
            lift["golds_lifted"] += 1
            into = [FAMDIR[t] for uu, vv, t in zip(q["u"], q["v"], fd) if vv == g and uu in S]
            if g in S:
                lift["one_edge_from_seed"]["is_seed"] = lift["one_edge_from_seed"].get("is_seed", 0) + 1
            elif into:
                for t in sorted(set(into)):
                    lift["one_edge_from_seed"][t] = lift["one_edge_from_seed"].get(t, 0) + 1
            else:
                nb1 = {vv for uu, vv in zip(q["u"], q["v"]) if uu in S}
                if any(vv == g and uu in nb1 for uu, vv in zip(q["u"], q["v"])):
                    lift["two_edges_only"] += 1
                else:
                    lift["farther"] += 1
    res["gnn_lift"] = lift
    log("gnn lift: " + json.dumps(lift))
    names = a.schemes.split(",")
    toks = schemes_tokens(Q, A_rows, sorted({nm.split("-")[0] for nm in names}))
    res["schemes"] = {}
    for name in names:
        nt, tk = toks[name.split("-")[0]]
        max_len = 1 if name.endswith("-1") else 2
        t1 = time.time()
        TY = [walk_types(q, t, nt, max_len) for q, t in zip(Q, tk)]
        nty = np.asarray([len(t) for t in TY])
        cover = np.asarray([float(np.isin(np.flatnonzero(q["gold"]), np.concatenate(list(t.values())) if t else np.zeros(0)).mean())
                            if q["gold"].any() else 0.0 for q, t in zip(Q, TY)])
        vocab = len(set().union(*[set(t) for t in (TY[i] for i in A_rows)]))
        # gold-chosen single-type ceiling on B (optimistic)
        ceil = []
        for i in B_rows:
            q = Q[i]
            best = metrics_of(z_of[i], q["gold"], q["gt"])
            for c, R in TY[i].items():
                s = z_of[i].copy()
                s[R] += 1e3
                m = metrics_of(s, q["gold"], q["gt"])
                best = np.maximum(best, m)
            ceil.append(best)
        ceil = np.asarray(ceil)
        model, mB = fit_read(Q, TY, nt, A_rows, B_rows, z_of, epochs=a.epochs)
        gain_g = gB.mean(0) - tB.mean(0)
        rho = [(float(mB[:, j].mean() - tB[:, j].mean()) / float(gain_g[j])) if abs(gain_g[j]) > 1e-9 else None for j in range(3)]
        by_type = {}
        for t in TYPES2W:
            sel = ty[Bsel] == t
            if sel.any():
                by_type[t] = {"rows": int(sel.sum()), "fit": mB[sel].mean(0).round(4).tolist(), "twin0": tB[sel].mean(0).round(4).tolist(),
                              "gnn0": gB[sel].mean(0).round(4).tolist()}
        res["schemes"][name] = {
            "tokens": nt, "types_per_row": {"mean": float(nty.mean()), "p95": float(np.percentile(nty, 95)), "max": int(nty.max())},
            "vocab_A": vocab, "gold_reached_share": float(cover.mean()),
            "ceiling_B": ceil.mean(0).round(4).tolist(), "fit_B": mB.mean(0).round(4).tolist(),
            "rho_fit (R@5, FC@5, hit@1)": [None if r is None else round(r, 3) for r in rho],
            "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
            "by_type": by_type, "seconds": round(time.time() - t1, 1)}
        log(f"{name}: {json.dumps({k: v for k, v in res['schemes'][name].items() if k != 'by_type'})}")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
