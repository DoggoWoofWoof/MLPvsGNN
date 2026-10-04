"""Design look (untracked; not a result and not filed): lean_mlp3 with one more block, AW. AW is the walks from the
seeds, each edge weighted by the query and the text of the edge's label.

Every other block, the carves, the stores (--store rand128 | pca256), the greedy, the refits and the reads are
lean_mlp3's, called unchanged. This look adds:

AW (6 columns per pool node)
  * Walks are kept separately for one hop (seed s -> v, over every family's pool edges) and two hops (s -> u -> v,
    v != s).
  * A walk's edge is the token (direction, label):
      direction: anchor_walk's famdir, i.e. 0 forward, 1 backward, 2 both (structural), 3 ner, 4 knn.
      label: a structural edge's anchor phrase, the last two tokens its source article puts before the link
        (outputs/mp_approx_2wiki_anchor/anchor_extract.py; a fixed corpus attribute, read through anchors_compact.npz).
          - The phrase is that of the stored edge u -> v, or of v -> u (the inverse) when only that one is stored.
          - Its rank by count below K_LAB = 4096, else OTHER.
          - ner and knn edges have no label (OTHER).
  * The edge's logit is <W_q q, h(label) + D[direction]> / sqrt(64) + b[direction], with
    h(label) = P z(label) (+ a free row for the top --aw-res labels).
      - z(label) is the whitened PCA of the label's gte-Qwen2 text vector, r = --aw-r axes, as in anchor_gen3. Its
        basis is fitted once on the first training graph's 4096-phrase table and is saved, so a graph the model never
        saw is coded through its own phrases' text.
      - OTHER has one free vector.
  * The 6 columns:
      - per one-hop walk: log1p of the sum of sigmoid(logit), the max logit, and log1p of the sigmoid sum over
        bucket-0 seeds only;
      - per two-hop walk, the same three on sigmoid(l1) * sigmoid(l2) and l1 + l2.
  * Training applies relation dropout: each walk edge's label becomes OTHER with probability --aw-drop. A read with
    every label OTHER (NR) is the no-relation read, which shows whether the block degrades gracefully.
  * Non-MP: learned weights of the query and an edge's own label, applied to fixed seed-walk counts. No node state is
    propagated and no neighbour's embedding or score is read. Two-hop walks are factorised products, as in the
    type-factorised track.
  * Cost: the walk entries are timed per query in the carve pass. The label lookup goes to index time, where a typed
    CSR holds the anchor rank in place of the relation.

The epoch rule is --rule:
  - rf: lean_mlp's mean of R@5 and FC@5;
  - rfh: the mean of all three, so hit@1 is protected.

    python outputs/mp_unified/lean_mlp4.py --store pca256 --train 2wiki=x4 --select 2wiki=select --read 2wiki=x1 \
        --save-models outputs/mp_unified/lean/l4-2w_models.pt --out outputs/mp_unified/lean/l4-2w.json
"""
import os
import sys

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import lean_mlp as LM  # noqa: E402
import lean_mlp2 as L2  # noqa: E402
import lean_mlp3 as L3  # noqa: E402

AW_W = 6
N_DIR = 5
K_LAB = 4096
ANCHORS = {"2wiki": ROOT / "outputs" / "mp_approx_2wiki_anchor" / "host",
           "hotpotqa": ROOT / "outputs" / "mp_approx_hotpot_anchor" / "host",
           "musique": ROOT / "outputs" / "mp_approx_musique_anchor" / "host"}
LM.STRUCT_USERS |= {"AW"}
LM.NK_USERS |= {"AW"}
log = LM.log
_BATCH_OF = LM.batch_of


# ── the labels ───────────────────────────────────────────────────────────────


def served_root():
    """The served package root (the laptop's package, or the host mirror through the substitution) and its canonical module."""
    sys.dont_write_bytecode = True
    import look_x_six as LX
    V2 = LX.V2
    _cfg, cfg_m3b, _cfg_h = V2.load_configs()
    m3b_compile = V2.M3B_RUN.load_script("m3b_compile")
    _m3a, canonical, served, freeze = m3b_compile.open_package(cfg_m3b)
    return Path(served), canonical, freeze["RECORD_SHA256"]


class Labels:
    """A passage graph's anchor-phrase rank (capped at K_LAB = OTHER) per stored structural edge, by global (src, dst),
    and its 4096-phrase text table. A graph without an anchor table has no labels: every edge reads OTHER."""

    def __init__(self, ds, served, canonical):
        self.ds = ds
        d = ANCHORS.get(ds)
        self.ok = d is not None and (d / "anchors_compact.npz").exists() and (d / "phrase_emb_w2_K4096.npy").exists()
        self.seconds = 0.0
        if not self.ok:
            log(f"{ds}: no anchor table; every AW label reads OTHER")
            return
        t = time.time()
        self.n_nodes = int(canonical.Dataset(ds, root=str(served)).n_nodes)
        gm = json.loads((served / ds / "graph" / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))
        self.manifest_entry = {k: v for k, v in gm.items() if "struct" in k.lower()} or None
        with np.load(served / ds / "graph" / "structural.npz") as z:
            src, dst = z["src"].astype(np.int64), z["dst"].astype(np.int64)
        with np.load(d / "anchors_compact.npz") as z:
            w2 = z["w2r"]
        if w2.size != src.size:
            raise SystemExit(f"{ds}: {w2.size} anchors for {src.size} structural edges")
        key = src * self.n_nodes + dst
        del src, dst
        order = np.argsort(key, kind="stable")
        self.skey = key[order]
        del key
        self.rank = np.minimum(w2[order].astype(np.int32), K_LAB).astype(np.int16)
        del order, w2
        self.phrase = np.load(d / "phrase_emb_w2_K4096.npy").astype(np.float32)
        self.phrase_sha256 = hashlib.sha256((d / "phrase_emb_w2_K4096.npy").read_bytes()).hexdigest()
        self.seconds = time.time() - t
        log(f"{ds}: anchor table over {self.skey.size} structural edges ({self.seconds:.0f}s)")

    def lookup(self, gu, gv):
        k = gu * self.n_nodes + gv
        j = np.searchsorted(self.skey, k)
        jc = np.minimum(j, self.skey.size - 1)
        hit = (j < self.skey.size) & (self.skey[jc] == k)
        return np.where(hit, self.rank[jc].astype(np.int32), -1)


def code_basis(phrase, r):
    """anchor_gen3's whitened PCA of a phrase table: mu, the top r axes and their variances."""
    mu = phrase.mean(0)
    X = (phrase - mu).astype(np.float64)
    w, V = np.linalg.eigh(X.T @ X / X.shape[0])
    idx = np.argsort(w)[::-1][:r]
    return {"mu": mu.astype(np.float32), "V": V[:, idx].astype(np.float32), "w": w[idx].astype(np.float32), "r": int(r)}


def codes_of(labels, basis):
    """z(label) for every label of a graph, row K_LAB (OTHER) zero; all zero without labels."""
    Z = np.zeros((K_LAB + 1, basis["r"]), np.float32)
    if labels is not None and labels.ok:
        Z[:K_LAB] = (labels.phrase[:K_LAB] - basis["mu"]) @ basis["V"] / np.sqrt(np.maximum(basis["w"], 1e-12))
    return Z


# ── the walk entries ─────────────────────────────────────────────────────────


def famdir(fam, fwd, bwd):
    return np.where(fam == 1, 3, np.where(fam == 2, 4, np.where(fwd & bwd, 2, np.where(fwd, 0, 1)))).astype(np.int8)


def walk_entries(n, pool, eu, ev, efam, efwd, efwd_b, seeds, buckets, labels, checks):
    """One query's AW walks. Returns (one-hop: node, dir, label, bucket), (two-hop: node, dir1, label1, dir2, label2,
    bucket) and the seconds spent in the label lookup (index-time work at serving)."""
    u, v = eu.astype(np.int64), ev.astype(np.int64)
    fwd, bwd = efwd.astype(bool), efwd_b.astype(bool)
    d = famdir(efam, fwd, bwd)
    lab = np.full(u.size, K_LAB, np.int32)
    t_lab = 0.0
    st = efam == 0
    if labels is not None and labels.ok and st.any():
        t = time.perf_counter()
        gu, gv = pool[u[st]].astype(np.int64), pool[v[st]].astype(np.int64)
        lf, lb = labels.lookup(gu, gv), labels.lookup(gv, gu)
        t_lab = time.perf_counter() - t
        f_s, b_s = fwd[st], bwd[st]
        checks["fwd_flag"] += int(f_s.sum())
        checks["fwd_found"] += int((f_s & (lf >= 0)).sum())
        checks["bwd_flag"] += int(b_s.sum())
        checks["bwd_found"] += int((b_s & (lb >= 0)).sum())
        a = np.where(d[st] == 1, lb, lf)
        checks["missing"] += int((a < 0).sum())
        lab[st] = np.where(a >= 0, a, K_LAB)
    valid = seeds >= 0
    S, Bk = seeds[valid].astype(np.int64), buckets[valid].astype(np.int64)
    sb = np.full(n, -1, np.int64)
    sb[S[::-1]] = Bk[::-1]                     # a node listed twice keeps its first bucket
    first = np.flatnonzero(sb[u] >= 0)
    one = (v[first], d[first], lab[first], sb[u[first]])
    # two hops: s -> u (an edge in `first`), u -> v (any edge out of u), v != s
    order = np.argsort(u, kind="stable")
    us = u[order]
    nodes = np.arange(n)
    start, end = np.searchsorted(us, nodes), np.searchsorted(us, nodes, side="right")
    mid = v[first]
    cnt = end[mid] - start[mid]
    rep = np.repeat(first, cnt)
    pos = np.repeat(start[mid] - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(int(cnt.sum()))
    second = order[pos]
    keep = v[second] != u[rep]
    rep, second = rep[keep], second[keep]
    two = (v[second], d[rep], lab[rep], d[second], lab[second], sb[u[rep]])
    return one, two, t_lab


def chunk_rows4(d, limit):
    """lean_mlp2.chunk_rows' row order with the arrays the walks read."""
    recs = sorted(d.glob("record*.json"))
    listed = set()
    for r_ in recs:
        listed |= set(json.loads(r_.read_text(encoding="utf-8"))["chunks"])
    rows = 0
    for ch in sorted((d / "chunks").glob("c*.npz")):
        if int(ch.stem[1:]) not in listed:
            continue
        z = np.load(ch)
        arrays = {k: z[k] for k in ("q_pool_size", "q_edges", "pool", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "q_seed_local",
                                    "q_seed_bucket")}
        for i in range(arrays["q_pool_size"].size):
            if limit is not None and rows >= limit:
                return
            yield arrays, i
            rows += 1


class Carve4(L3.Carve3):
    """lean_mlp3's carve plus the AW walks of every row and the graph's label codes."""

    def __init__(self, ds, carve, store=None, nodes=None, limit=None, root=LM.LOOK, labels=None, basis=None):
        super().__init__(ds, carve, store, nodes, limit, root)
        one = {k: [] for k in ("node", "dir", "lab", "b")}
        two = {k: [] for k in ("node", "d1", "l1", "d2", "l2", "b")}
        n1, n2, per_q, lab_s = [], [], [], 0.0
        self.label_checks = {"fwd_flag": 0, "fwd_found": 0, "bwd_flag": 0, "bwd_found": 0, "missing": 0}
        k, cache = 0, None
        for arrays, i in chunk_rows4(Path(root) / ds / carve, limit):
            if cache is None or cache[0] is not arrays:
                cache = (arrays, np.concatenate([[0], np.cumsum(arrays["q_pool_size"])]), np.concatenate([[0], np.cumsum(arrays["q_edges"])]))
            _, no, eo = cache
            a, b, ea, eb = no[i], no[i + 1], eo[i], eo[i + 1]
            n = int(arrays["q_pool_size"][i])
            assert n == int(self.n[k]), (ds, carve, k)
            t = time.perf_counter()
            o, w, t_lab = walk_entries(n, arrays["pool"][a:b], arrays["e_u"][ea:eb], arrays["e_v"][ea:eb], arrays["e_fam"][ea:eb],
                                       arrays["e_fwd"][ea:eb], arrays["e_bwd"][ea:eb], arrays["q_seed_local"][i],
                                       arrays["q_seed_bucket"][i], labels, self.label_checks)
            per_q.append(1e3 * (time.perf_counter() - t - t_lab))
            lab_s += t_lab
            for key, arr in zip(one, o):
                one[key].append(arr)
            for key, arr in zip(two, w):
                two[key].append(arr)
            n1.append(o[0].size)
            n2.append(w[0].size)
            k += 1
        assert k == self.rows, (ds, carve, k, self.rows)
        self.aw1 = {"node": np.concatenate(one["node"]).astype(np.int64), "dir": np.concatenate(one["dir"]).astype(np.int64),
                    "lab": np.concatenate(one["lab"]).astype(np.int64), "b": np.concatenate(one["b"]).astype(np.int64)}
        self.aw2 = {"node": np.concatenate(two["node"]).astype(np.int64), "d1": np.concatenate(two["d1"]).astype(np.int64),
                    "l1": np.concatenate(two["l1"]).astype(np.int64), "d2": np.concatenate(two["d2"]).astype(np.int64),
                    "l2": np.concatenate(two["l2"]).astype(np.int64), "b": np.concatenate(two["b"]).astype(np.int64)}
        self.aw_off1 = np.concatenate([[0], np.cumsum(n1)]).astype(np.int64)
        self.aw_off2 = np.concatenate([[0], np.cumsum(n2)]).astype(np.int64)
        self.aw_Z = codes_of(labels, basis)
        self.aw_nr = False
        self.widths["AW"] = AW_W
        self.lean_ms["AW"] = {"p50": float(np.percentile(per_q, 50)), "p95": float(np.percentile(per_q, 95)), "mean": float(np.mean(per_q))}
        self.aw_stats = {"one_hop_mean": float(np.mean(n1)), "two_hop_mean": float(np.mean(n2)), "two_hop_max": int(np.max(n2)),
                         "label_lookup_s": lab_s, "labelled": bool(labels is not None and labels.ok),
                         "other_share_one_hop": float((self.aw1["lab"] == K_LAB).mean()) if self.aw1["lab"].size else None,
                         "checks": self.label_checks}
        log(f"{ds}={carve}: AW walks one-hop {np.mean(n1):.1f}, two-hop {np.mean(n2):.1f} (max {np.max(n2)}) a row; "
            f"entry ms p50 {self.lean_ms['AW']['p50']:.3f}; label lookup {lab_s:.1f}s; checks {self.label_checks}")
        c = self.label_checks
        if labels is not None and labels.ok and (c["fwd_found"] != c["fwd_flag"] or c["bwd_found"] != c["bwd_flag"]):
            raise SystemExit(f"{ds}={carve}: the pool's direction flags do not match the stored edges: {c}")


def aw_batch(carve, qs):
    """The AW inputs of a batch: walk entries with batch-local node and query indices, the queries and the label codes."""
    base = np.concatenate([[0], np.cumsum(carve.n[qs])])[:-1]
    s1 = [np.arange(carve.aw_off1[q], carve.aw_off1[q + 1]) for q in qs]
    s2 = [np.arange(carve.aw_off2[q], carve.aw_off2[q + 1]) for q in qs]
    i1, i2 = np.concatenate(s1), np.concatenate(s2)
    q1 = np.repeat(np.arange(len(qs)), [s.size for s in s1])
    q2 = np.repeat(np.arange(len(qs)), [s.size for s in s2])
    t = torch.from_numpy
    out = {"qe": t(carve.q_emb[qs].astype(np.float32)), "Z": t(carve.aw_Z), "N": int(carve.n[qs].sum()),
           "n1": t(carve.aw1["node"][i1] + base[q1]), "q1": t(q1), "d1": t(carve.aw1["dir"][i1]), "l1": t(carve.aw1["lab"][i1]),
           "b1": t(carve.aw1["b"][i1]),
           "n2": t(carve.aw2["node"][i2] + base[q2]), "q2": t(q2), "d2a": t(carve.aw2["d1"][i2]), "l2a": t(carve.aw2["l1"][i2]),
           "d2b": t(carve.aw2["d2"][i2]), "l2b": t(carve.aw2["l2"][i2]), "b2": t(carve.aw2["b"][i2])}
    if carve.aw_nr:
        out["l1"] = torch.full_like(out["l1"], K_LAB)
        out["l2a"] = torch.full_like(out["l2a"], K_LAB)
        out["l2b"] = torch.full_like(out["l2b"], K_LAB)
    return out


def batch_of4(carve, qs, blocks):
    feats, nq, base_z, gold, idx = _BATCH_OF(carve, qs, [b for b in blocks if b != "AW"])
    if "AW" in blocks:
        feats["AW"] = aw_batch(carve, qs)
    return feats, nq, base_z, gold, idx


# ── the model ────────────────────────────────────────────────────────────────


class AWNet(nn.Module):
    def __init__(self, r, D=64, k_res=0, p_drop=0.25, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 11)
        self.D, self.k_res, self.p_drop = D, k_res, p_drop
        self.P = nn.Parameter(torch.randn(r, D, generator=g) / math.sqrt(r))
        self.dirv = nn.Parameter(torch.randn(N_DIR, D, generator=g) / math.sqrt(D))
        self.other = nn.Parameter(torch.zeros(D))
        self.Wq = nn.Parameter(torch.randn(1536, D, generator=g) / math.sqrt(1536))
        self.bias = nn.Parameter(torch.zeros(N_DIR))
        self.res = nn.Parameter(torch.zeros(k_res, D)) if k_res else None

    def logit(self, phi, qi, d, lab, Z):
        is_o = (lab >= K_LAB).unsqueeze(1)
        h = Z[lab.clamp(max=K_LAB)] @ self.P
        h = torch.where(is_o, self.other.unsqueeze(0).expand_as(h), h)
        if self.res is not None:
            m = lab < self.k_res
            h = h + torch.where(m.unsqueeze(1), self.res[lab.clamp(max=self.k_res - 1)], torch.zeros_like(h))
        return (phi[qi] * (h + self.dirv[d])).sum(1) / math.sqrt(self.D) + self.bias[d]

    def forward(self, A):
        phi = A["qe"] @ self.Wq
        Z, N = A["Z"], A["N"]
        l1, l2a, l2b = A["l1"], A["l2a"], A["l2b"]
        if self.training and self.p_drop > 0:
            l1 = torch.where(torch.rand(l1.shape) < self.p_drop, torch.full_like(l1, K_LAB), l1)
            l2a = torch.where(torch.rand(l2a.shape) < self.p_drop, torch.full_like(l2a, K_LAB), l2a)
            l2b = torch.where(torch.rand(l2b.shape) < self.p_drop, torch.full_like(l2b, K_LAB), l2b)
        cols = []
        for walks in ((A["n1"], self.logit(phi, A["q1"], A["d1"], l1, Z), None, A["b1"]),
                      (A["n2"], self.logit(phi, A["q2"], A["d2a"], l2a, Z), self.logit(phi, A["q2"], A["d2b"], l2b, Z), A["b2"])):
            node, la, lb, bk = walks
            g = torch.sigmoid(la) if lb is None else torch.sigmoid(la) * torch.sigmoid(lb)
            sc = la if lb is None else la + lb
            s = torch.zeros(N).index_add(0, node, g)
            mx = torch.full((N,), float("-inf")).scatter_reduce(0, node, sc, reduce="amax", include_self=True)
            m0 = bk == 0
            s0 = torch.zeros(N).index_add(0, node[m0], g[m0])
            cols += [torch.log1p(s), torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx)), torch.log1p(s0)]
        return torch.stack(cols, 1)


class LeanMLP4(L3.LeanMLP3):
    """lean_mlp3's model; with AW in its blocks, an AWNet turns the batch's walks into AW's six raw columns."""

    cfg = {"r": 32, "D": 64, "k_res": 0, "p_drop": 0.25}

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0):
        super().__init__(blocks, widths, hidden, dropout, seed)
        if "AW" in self.blocks:
            c = LeanMLP4.cfg
            self.awn = AWNet(c["r"], c["D"], c["k_res"], c["p_drop"], seed)

    def forward(self, feats, keep, nq, B, base_z):
        if "AW" in self.blocks and isinstance(feats.get("AW"), dict):
            feats = dict(feats)
            feats["AW"] = self.awn(feats["AW"])
        return super().forward(feats, keep, nq, B, base_z)


def fit4(train, select, blocks, mode, epochs, lr, seed, hidden, rule="rf"):
    """lean_mlp.fit with the epoch rule as a choice: rf (its mean of R@5 and FC@5) or rfh (the mean of all three)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: train[0].widths[b] for b in blocks}
    model = LeanMLP4(blocks, widths, hidden, seed=seed)
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
                feats, nq, base_z, gold, _ = batch_of4(train[ci], qs, blocks)
                B = qs.size
                if mode == "drop":
                    r = rng.uniform(0.15, 1.0, size=(B, 1))
                    keep = (rng.uniform(size=(B, len(blocks))) < r).astype(np.float32)
                    keep[:, [blocks.index(b) for b in LM.NEVER if b in blocks]] = 1.0
                else:
                    keep = np.ones((B, len(blocks)), np.float32)
                s = model(feats, torch.from_numpy(keep), nq, B, base_z)
                loss = LM.listwise(s, gold, nq, B)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
        q = LM.quality(model, select, blocks, {b: 1.0 for b in blocks})
        curve.append({"epoch": ep, "loss": tot / max(nb, 1), "select": q, "seconds": time.time() - t0})
        log(f"  ep {ep}: loss {tot / max(nb, 1):.4f} select R@5/FC@5/hit@1 {[round(v, 4) for v in q]} ({time.time() - t0:.0f}s)")
        score = 0.5 * (q[0] + q[1]) if rule == "rf" else (q[0] + q[1] + q[2]) / 3.0
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, curve


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="pca256", choices=("rand128", "pca256"))
    ap.add_argument("--basis-from", default="")
    ap.add_argument("--fit-nodes", type=int, default=60000)
    ap.add_argument("--train", default="2wiki=x4")
    ap.add_argument("--select", default="2wiki=select")
    ap.add_argument("--read", default="2wiki=x1")
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--final-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--fixed", default="")
    ap.add_argument("--sets", default="full,lean,leanA,lean2A")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--rule", default="rf", choices=("rf", "rfh"))
    ap.add_argument("--aw-r", type=int, default=32)
    ap.add_argument("--aw-dim", type=int, default=64)
    ap.add_argument("--aw-res", type=int, default=0, help="free label rows for the top N labels (0: text codes only)")
    ap.add_argument("--aw-drop", type=float, default=0.25)
    ap.add_argument("--save-models")
    ap.add_argument("--load-models")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    LM.LeanMLP = LeanMLP4
    LM.batch_of = batch_of4                   # lean_mlp's quality, greedy and read paths batch through this name
    blob = torch.load(a.load_models, weights_only=False) if a.load_models else None
    store_name = blob["store"] if blob is not None else a.store
    L3.LeanMLP3.dim = L3.STORE_DIM if store_name == "pca256" else LM.PROJ_DIM
    if blob is not None:
        LeanMLP4.cfg = dict(blob["aw_cfg"])
    else:
        LeanMLP4.cfg = {"r": a.aw_r, "D": a.aw_dim, "k_res": a.aw_res, "p_drop": a.aw_drop}
    names = sorted({ds for spec in (a.read,) + (() if blob is not None else (a.train, a.select)) for ds, _cv in LM.parse_sets(spec)})
    served, canonical, freeze = served_root()
    labels = {n: Labels(n, served, canonical) for n in names}
    if blob is not None:
        cbasis = blob["code_basis"]
    else:
        src = LM.parse_sets(a.train)[0][0]
        if not labels[src].ok:
            raise SystemExit(f"{src}: the first training graph has no anchor table to fit the code basis on")
        cbasis = code_basis(labels[src].phrase[:K_LAB], a.aw_r)
        cbasis["from"] = src
    store, nodes, basis_info = None, {}, None
    if store_name == "pca256":
        if blob is not None:
            basis = blob["basis"]
            nodes, _fz = L3.open_nodes(names)
        else:
            src = a.basis_from or LM.parse_sets(a.train)[0][0]
            nodes, _fz = L3.open_nodes(sorted(set(names) | {src}))
            basis = L3.fit_basis(nodes[src], a.fit_nodes)
            basis["from"] = src
            log(f"basis from {src}: {basis['rows']} nodes, {basis['explained']:.3f} of the variance on {L3.STORE_K} axes")
        store = L3.Store(basis)
        basis_info = {k: v for k, v in basis.items() if k not in ("m", "V", "w")}

    def carve(ds, cv):
        return Carve4(ds, cv, store, nodes.get(ds), a.limit, labels=labels[ds], basis=cbasis)

    if blob is not None:
        return read_only(a, t0, blob, carve, basis_info, freeze)
    tr = [carve(ds, cv) for ds, cv in LM.parse_sets(a.train)]
    se = [carve(ds, cv) for ds, cv in LM.parse_sets(a.select)]
    log(f"loaded train {[(c.ds, c.carve, c.rows) for c in tr]} select {[(c.ds, c.carve, c.rows) for c in se]} ({time.time() - t0:.0f}s)")
    present = list(LM.COMPILED) + list(LM.LEAN) + list(L2.NEW) + ["AW"] if a.blocks == "all" else a.blocks.split("+")
    dead = []
    for b in list(present):
        if b in LM.LEAN or b == "AW":
            continue
        v = np.concatenate([c.block(b, np.arange(int(c.off[-1]))).astype(np.float32) for c in tr])
        if float(np.nanstd(v)) == 0.0:
            dead.append(b)
            present.remove(b)
    log(f"blocks {present}; constant on the training rows (dropped) {dead}")
    costs = L2.costs_of(tr)
    out = {"look": "lean_mlp4", "store": store_name, "basis": basis_info, "freeze": freeze, "args": vars(a), "blocks": present,
           "dead": dead, "aw_cfg": LeanMLP4.cfg, "code_basis": {k: v for k, v in cbasis.items() if k not in ("mu", "V", "w")},
           "aw_stats": {f"{c.ds}={c.carve}": c.aw_stats for c in tr + se},
           "lean_ms": {f"{c.ds}={c.carve}": c.lean_ms for c in tr + se},
           "profiles": {c.ds: LM.profile(c.ds) for c in tr}, "struct_share": {f"{c.ds}={c.carve}": c.struct_share for c in tr},
           "labels": {n: {"ok": l.ok, "seconds": l.seconds, "phrase_sha256": getattr(l, "phrase_sha256", None),
                          "manifest": getattr(l, "manifest_entry", None)} for n, l in labels.items()},
           "assumptions": ["lean_mlp3's cost assumptions", "AW's lap is the walk-entry build; the label lookup is index-time "
                           "(a typed CSR carrying the anchor rank) and is reported apart"],
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "lean_mlp3_sha256": hashlib.sha256((HERE / "lean_mlp3.py").read_bytes()).hexdigest()}
    models = {}
    pick = None
    if not a.no_greedy:
        log("fit with block dropout")
        model, curve = fit4(tr, se, present, "drop", a.epochs, a.lr, a.seed, a.hidden, a.rule)
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
    aw = ["AW"] if "AW" in present else []
    every = {"full": present, "lean": lean, "leanA": lean + aw, "lean2A": lean + [b for b in present if b in L2.NEW] + aw,
             "lean2": lean + [b for b in present if b in L2.NEW]}
    fixed = {k: v for k, v in every.items() if k in a.sets.split(",")}
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
        m2, cur = fit4(tr, se, bl, "fixed", a.final_epochs, a.lr, a.seed, a.hidden, a.rule)
        out["fixed_curves"][name] = cur
        models[name] = (m2, bl, {b: 1.0 for b in bl})
    if a.save_models:
        Path(a.save_models).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"models": {name: {"state": m.state_dict(), "blocks": bl, "keep": keep, "widths": m.widths, "hidden": a.hidden}
                               for name, (m, bl, keep) in models.items()},
                    "pick": out.get("pick"), "present": present, "dead": dead, "args": vars(a), "store": store_name,
                    "basis": None if store is None else store.basis, "aw_cfg": LeanMLP4.cfg, "code_basis": cbasis}, a.save_models)
        log(f"saved {len(models)} models to {a.save_models}")
    del tr, se
    read_all(a, models, pick, out, carve)
    out["seconds"] = time.time() - t0
    L2.write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def read_only(a, t0, blob, carve, basis_info, freeze):
    models = {}
    for name, d in blob["models"].items():
        m = LeanMLP4(d["blocks"], d["widths"], d["hidden"])
        m.load_state_dict(d["state"])
        models[name] = (m, d["blocks"], d["keep"])
    pick = blob.get("pick")
    out = {"look": "lean_mlp4", "mode": "read_only", "store": blob["store"], "basis": basis_info, "freeze": freeze, "args": vars(a),
           "loaded": a.load_models, "loaded_sha256": hashlib.sha256(Path(a.load_models).read_bytes()).hexdigest(),
           "fit_args": blob.get("args"), "aw_cfg": blob["aw_cfg"], "pick": pick, "blocks": blob.get("present"),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    log(f"loaded {list(models)} from {a.load_models} (store {blob['store']}, AW {blob['aw_cfg']})")
    read_all(a, models, pick, out, carve)
    out["seconds"] = time.time() - t0
    L2.write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


def read_all(a, models, pick, out, carve):
    """lean_mlp3.read_all, plus each AW model's no-relation read (NR: every AW label OTHER) against its own ID read."""
    rng = np.random.default_rng(20261003)
    reads, read_costs, read_lean_ms, read_rows, aw_stats = {}, {}, {}, {}, {}
    for ds, cv in LM.parse_sets(a.read):
        key = f"{ds}={cv}"
        c = carve(ds, cv)
        read_rows[key] = {"rows": c.rows, "chunks": c.chunks_read, "n_chunks": c.n_chunks, "carve_queries": c.carve_queries}
        aw_stats[key] = c.aw_stats
        (prof, share, lm), = L2.costs_of([c])
        read_lean_ms[key] = c.lean_ms
        read_costs[key] = {name: LM.cost_ms(bl, prof, share, lm) for name, (_m, bl, _k) in models.items() if not name.startswith("drop/")}
        if pick is not None:
            read_costs[key]["drop/pick"] = LM.cost_ms(pick["blocks"], prof, share, lm)
        log(f"read {key} ({c.rows} rows); approximate compile ms {({k: round(v, 2) for k, v in read_costs[key].items()})}")
        base_rows, id_rows = {}, {}
        for name, (m, bl, keep) in models.items():
            ref = base_rows.get("drop/full" if name == "drop/pick" else "full")
            r, rows = LM.read_one(m, c, bl, keep, rng, ref)
            if name in ("full", "drop/full"):
                base_rows[name] = rows
            id_rows[name] = rows
            reads[f"{name}@{key}"] = r
            log(f"  {name}@{key}: {LM.fmt(r)}")
        c.aw_nr = True
        for name, (m, bl, keep) in models.items():
            if "AW" not in bl or keep.get("AW", 1.0) == 0.0:
                continue
            r, _rows = LM.read_one(m, c, bl, keep, rng, id_rows[name])
            reads[f"{name}@{key}/NR"] = r
            log(f"  {name}@{key}/NR: {LM.fmt(r)} (vs full = vs its own ID read)")
        c.aw_nr = False
        del c
    out["read_costs_ms"] = read_costs
    out["read_lean_ms"] = read_lean_ms
    out["read_rows"] = read_rows
    out["read_aw_stats"] = aw_stats
    out["reads"] = reads


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(5)
    # walk_entries against brute force on a random multigraph
    for trial in range(30):
        n = int(rng.integers(3, 30))
        m = int(rng.integers(0, 80))
        eu = rng.integers(0, n, m)
        ev = rng.integers(0, n, m)
        ok = eu != ev
        eu, ev = eu[ok], ev[ok]
        fam = rng.integers(0, 3, eu.size).astype(np.int8)
        fwd = (rng.uniform(size=eu.size) < 0.6) & (fam == 0)
        bwd = ((rng.uniform(size=eu.size) < 0.6) | ~fwd) & (fam == 0)
        S = rng.choice(n, size=min(n, int(rng.integers(1, 5))), replace=False)
        seeds = np.r_[S, -np.ones(10 - S.size, np.int64)]
        buckets = np.r_[rng.integers(0, 2, S.size), -np.ones(10 - S.size, np.int64)]
        checks = {"fwd_flag": 0, "fwd_found": 0, "bwd_flag": 0, "bwd_found": 0, "missing": 0}
        one, two, _t = walk_entries(n, np.arange(n), eu, ev, fam, fwd, bwd, seeds, buckets, None, checks)
        d = famdir(fam, fwd, bwd)
        bucket_of = {}
        for s_, b_ in zip(S.tolist(), buckets[:S.size].tolist()):
            bucket_of.setdefault(s_, b_)
        want1 = sorted((int(ev[j]), int(d[j]), K_LAB, bucket_of[int(eu[j])]) for j in range(eu.size) if int(eu[j]) in bucket_of)
        got1 = sorted(zip(*(x.tolist() for x in one)))
        assert want1 == got1, ("one-hop", trial)
        want2 = sorted((int(ev[k]), int(d[j]), K_LAB, int(d[k]), K_LAB, bucket_of[int(eu[j])])
                       for j in range(eu.size) if int(eu[j]) in bucket_of
                       for k in range(eu.size) if int(eu[k]) == int(ev[j]) and int(ev[k]) != int(eu[j]))
        got2 = sorted(zip(*(x.tolist() for x in two)))
        assert want2 == got2, ("two-hop", trial)
    # AWNet: shapes, the NR read, and a model without AW equals lean_mlp3's
    Z = torch.from_numpy(rng.standard_normal((K_LAB + 1, 8)).astype(np.float32))
    Z[K_LAB] = 0
    net = AWNet(8, 16, k_res=4, p_drop=0.0)
    A = {"qe": torch.randn(2, 1536), "Z": Z, "N": 7,
         "n1": torch.tensor([0, 1, 1, 5]), "q1": torch.tensor([0, 0, 0, 1]), "d1": torch.tensor([0, 1, 3, 2]),
         "l1": torch.tensor([3, K_LAB, K_LAB, 4000]), "b1": torch.tensor([0, 1, 0, 0]),
         "n2": torch.tensor([2, 6]), "q2": torch.tensor([0, 1]), "d2a": torch.tensor([0, 4]), "l2a": torch.tensor([1, K_LAB]),
         "d2b": torch.tensor([1, 0]), "l2b": torch.tensor([2, 7]), "b2": torch.tensor([1, 0])}
    f = net(A)
    assert tuple(f.shape) == (7, AW_W) and torch.isfinite(f).all()
    assert float(f[3].abs().sum()) == 0.0 and float(f[4].abs().sum()) == 0.0, "a node without walks reads zero"
    assert float(f[1, 2]) > 0 and float(f[1, 0]) > float(f[1, 2]) and float(f[0, 2]) > 0
    L3.LeanMLP3.dim = LM.PROJ_DIM
    m3 = L3.LeanMLP3(["rank", "SEED"], {"rank": 3, "SEED": 3})
    m4 = LeanMLP4(["rank", "SEED"], {"rank": 3, "SEED": 3})
    m4.load_state_dict(m3.state_dict())
    m3.eval(), m4.eval()
    fe = {"rank": torch.randn(5, 3), "SEED": torch.randn(5, 3)}
    nq = torch.tensor([0, 0, 0, 1, 1])
    bz = torch.randn(5)
    assert torch.equal(m3(fe, torch.ones(2, 2), nq, 2, bz), m4(fe, torch.ones(2, 2), nq, 2, bz))
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
