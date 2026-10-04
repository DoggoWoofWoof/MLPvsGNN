"""Design look (untracked; not a result and not filed): S3 of the transfer plan. Can a KB's relations be typed without
labels, and does the unlabeled query distribution say which type a query asks for?

Part A, typing. The stored triples h -> t of a KB are collected from the look chunks of the carves read: the union of
the rows' pools, which is index-time structure (no gold, no label, no model). Each triple gets label-free descriptors:
    TT    its endpoints' text, [n(p_h), n(p_t)], p the look's 128-d projection of a node's unit embedding
    CARD  its cardinality pattern: log1p of h's and t's out- and in-degree in the union graph, z-scored, scaled by
          1/sqrt(2) so its expected squared norm (4 x 1/2) is TT's (2 unit vectors)
    CTX   its neighbourhood, what one line-graph layer sees: [n(mean n(p) of t's other heads), n(mean n(p) of h's
          other tails)] (zero where there is no other)
k-means (scikit-learn, seeded; fitted on at most --sample triples, then every triple assigned) types the triples under
each descriptor set, and schema<k> types a triple by the k-means clusters (k of them) of its two endpoints' text. Each
typing is graded by NMI (geometric, as qd_gnn8.nmi), purity and each relation's best F1 against the true relation,
which is read only to grade (a triple's relation: its smallest relation id; 'single' rows: triples with one relation).

Part B, alignment (Swastik's 'matching the query distribution'). A row's candidate types T_i are the typed incidences
z = (type, direction) of the stored triples in the ball of radius --hops around its rank-1 seeds (bucket 0) in its
pool: the edges whose near endpoint is k < hops structural edges from the seed set and whose far endpoint is k + 1;
direction 0 when the near endpoint is the head (at radius 1: the seed is the head), 1 when it is the tail. On metaqa
the rank-1 seed is mostly a node with one relation, so radius 1 is near trivial there and radius 2 asks which of the
next hop's relations the query wants. EM over the fit carve's queries, no gold:
    p(z | i) = pi_z exp(kappa cos(q_i, mu_z)) / sum over z' in T_i of the same
    E: r_iz = p(z | i);   M: mu_z = n(sum_i r_iz q_i),  pi_z = sum_i r_iz / rows
q is the query's unit embedding under the look's projection, so cos(q, p) is the dense cosine up to the projection.
Graded on the read carve: a row's asked types A_i are the types of its ball's edges that lie on a shortest path from
the seed set to gold of D <= 3 structural edges in its pool (gold is read only here), and each rule's score over T_i
is read as the mean per-row AUC (A_i against T_i - A_i; ties count half) and the top-1 hit (ties shared), overall and
by D. Rules:
    prior-T  how often z is a candidate over the fit rows (query-blind)
    count    log1p of how many of the row's seed edges have the type (query-blind)
    desc     cos(q, d_z), d_z the unit mean text of the type's far endpoints (label-free, no fit: zero-shot)
    em<k>    log pi_z + kappa cos(q, mu_z), EM started from d_z; em<k>-pi its pi alone; em<k>-r started at random;
             em<k>-tx fitted on the read carve's own queries (transductive, still no gold)
    sup<k>   log pi_z + kappa cos(q, mu_z) with mu_z, pi_z the fit rows' queries that ask z (uses gold: a bound)
    name     cos(q, the relation name's text) under oracle typing only (the label as text: awc's signal)
Typings: oracle (the true relation as the type id: alignment without the typing question), schema<k>, and k-means.
When --fit and --read name one carve (webqsp has one full carve), every fit is transductive and sup is in-sample.

    python outputs/mp_unified/reltype11.py --ds metaqa --fit fit --read x1f --out outputs/mp_unified/lean/rt11-mq.json
    python outputs/mp_unified/reltype11.py --ds webqsp --fit selectf --read selectf --ks 32,256 --k-nodes 16,64 \
        --b-typings oracle,schema64,TT@256,TT+CARD+CTX@256 --out outputs/mp_unified/lean/rt11-wq.json
    python outputs/mp_unified/reltype11.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOOK = HERE / "look"
REL = ROOT / "outputs" / "m3b" / "relations"
PROJ_DIM, PROJ_SEED = 128, 20261001          # look_x_six's and look_score_kb's projection
SEED = 20261004
SHIFT = 32
MAXHOP = 3
EM_ITERS = 30
BOOT = 1000
CHUNK_RE = re.compile(r"c(\d{5})\.npz")
KEYS = ("q_pool_size", "q_edges", "q_emb", "q_seed_local", "q_seed_bucket", "pool", "proj", "is_gold",
        "e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "e_rel")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def proj_matrix():
    return (np.random.default_rng(PROJ_SEED).standard_normal((1536, PROJ_DIM)) / math.sqrt(PROJ_DIM)).astype(np.float32)


# ── reading the looks ────────────────────────────────────────────────────────


def chunk_paths(ds, cv, root=LOOK, limit=None):
    d = Path(root) / ds / cv
    recs = sorted(d.glob("record*.json"))
    if not recs:
        raise SystemExit(f"{d}: no look record")
    listed = set()
    rec0 = json.loads(recs[0].read_text(encoding="utf-8"))
    for r_ in recs:
        rr = json.loads(r_.read_text(encoding="utf-8"))
        if not rr.get("full"):
            raise SystemExit(f"{r_}: not a --full look (it has no edges)")
        listed |= set(rr["chunks"])
    rel = rec0.get("relations") or {}
    if not rel.get("typed"):
        raise SystemExit(f"{d}: not a typed KB look")
    if rec0.get("proj") != {"dim": PROJ_DIM, "seed": PROJ_SEED}:
        raise SystemExit(f"{d}: projection {rec0.get('proj')}, not dim {PROJ_DIM} seed {PROJ_SEED}")
    paths = []
    for p in sorted((d / "chunks").glob("c*.npz")):
        m = CHUNK_RE.fullmatch(p.name)
        if m and int(m.group(1)) in listed:
            paths.append(p)
    if not paths:
        raise SystemExit(f"{d}: no listed chunk is here")
    if limit:
        paths = paths[:limit]
    return paths, {"n_chunks": int(rec0["n_chunks"]), "listed": len(listed), "read": len(paths),
                   "n_relations": int(rel["n_relations"]), "offset": int(rel["offset"])}


def dedupe(keys, rows, gap=False):
    """Sorted unique keys, the first row of each, and how many later rows differ from their key's first (gap: the
    largest absolute difference instead)."""
    order = np.argsort(keys, kind="stable")
    k, r = keys[order], rows[order]
    first = np.r_[True, k[1:] != k[:-1]]
    grp = np.cumsum(first) - 1
    if gap:
        d = np.abs(r.astype(np.float32) - r[first][grp].astype(np.float32))
        return k[first], r[first], float(d.max()) if d.size else 0.0
    diff = r != r[first][grp]
    if diff.ndim > 1:
        diff = diff.any(axis=tuple(range(1, diff.ndim)))
    return k[first], r[first], int(diff.sum())


def gold_dist(n, u, v, gold):
    """Edges from every gold node (undirected structural adjacency), at most MAXHOP; MAXHOP + 1 beyond."""
    dist = np.full(n, MAXHOP + 1, dtype=np.int64)
    if gold.size == 0:
        return dist
    A = sp.csr_matrix((np.ones(u.size, dtype=np.float32), (u, v)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32)
    dist[gold] = 0
    front = np.zeros(n, dtype=np.float32)
    front[gold] = 1.0
    for d in range(1, MAXHOP + 1):
        nxt = (A @ front > 0) & (dist > MAXHOP)
        if not nxt.any():
            break
        dist[nxt] = d
        front = nxt.astype(np.float32)
    return dist


def ball_incidences(pool, hl, tl, d_seed, d_gold, D, hops):
    """The stored triples hl -> tl (pool-local, from either message of a pair) inside the seeds' ball of radius hops:
    an edge whose near endpoint is k < hops edges from the seed set and its far endpoint k + 1 (edges between two nodes
    at one distance are on no shortest path and are left out). Direction 0 when the near endpoint is the head (at the
    seed: the seed is the head), 1 when it is the tail; on when the edge lies on a shortest path from the seed set to
    gold (near distance + 1 + far's gold distance = D, the seed-gold distance, D <= MAXHOP). One entry per (triple key,
    direction), on if any of its copies is."""
    dh, dt = d_seed[hl], d_seed[tl]
    away = (dt == dh + 1) & (dh < hops)
    tow = (dh == dt + 1) & (dt < hops)
    keep = away | tow
    if not keep.any():
        return None
    hl, tl, away = hl[keep], tl[keep], away[keep]
    x = np.where(away, hl, tl)
    y = np.where(away, tl, hl)
    on = (D <= MAXHOP) & (d_seed[x] + 1 + d_gold[y] == D)
    key = (pool[hl] << SHIFT) | pool[tl]
    dr = (~away).astype(np.int8)
    order = np.lexsort((dr, key))
    key, dr, on = key[order], dr[order], on[order]
    first = np.r_[True, (key[1:] != key[:-1]) | (dr[1:] != dr[:-1])]
    grp = np.cumsum(first) - 1
    onu = np.bincount(grp, weights=on.astype(np.float64)) > 0
    return key[first], dr[first], onu


class Collector:
    """Stored triples (deduplicated on global (h, t)), the node table, and per row of each carve its projected query,
    its seed-gold distance D and, per ball radius, its rank-1 seeds' incidences (triple key, direction, on a shortest
    seed-gold path)."""

    def __init__(self, P, hops=(1,)):
        self.P = P
        self.hops = tuple(hops)
        self.tk, self.ts, self.nk, self.np_ = [], [], [], []
        self.slot_mismatch, self.proj_gap = 0, 0.0
        self.both = self.one_way = self.struct_msgs = 0
        self.carves = {}

    def consolidate(self):
        if len(self.tk) > 1:
            k, s, m = dedupe(np.concatenate(self.tk), np.concatenate(self.ts))
            self.slot_mismatch += m
            self.tk, self.ts = [k], [s]
        if len(self.nk) > 1:
            k, p, g = dedupe(np.concatenate(self.nk), np.concatenate(self.np_), gap=True)
            self.proj_gap = max(self.proj_gap, g)
            self.nk, self.np_ = [k], [p]

    def add(self, name, paths):
        C = self.carves.setdefault(name, {"q": [], "D": [], "rows": 0, "rows_gold": 0, "rows_seed0": 0,
                                          "inc": {h: {"row": [], "key": [], "dir": [], "on": []} for h in self.hops},
                                          "rows_asked": {h: 0 for h in self.hops}})
        for ci, p in enumerate(paths):
            with np.load(p) as z:
                c = {k: z[k] for k in KEYS}
            n_off = np.r_[0, np.cumsum(c["q_pool_size"])]
            e_off = np.r_[0, np.cumsum(c["q_edges"])]
            tk, ts = [], []
            pool_all = c["pool"].astype(np.int64)
            if pool_all.size and int(pool_all.max()) >= (1 << SHIFT - 1):
                raise SystemExit(f"{p}: a node id does not fit {SHIFT - 1} bits")
            k_n, p_n, g = dedupe(pool_all, c["proj"], gap=True)
            self.proj_gap = max(self.proj_gap, g)
            self.nk.append(k_n)
            self.np_.append(p_n)
            qp = unit(unit(c["q_emb"]) @ self.P)
            for i in range(c["q_pool_size"].size):
                a, b = int(n_off[i]), int(n_off[i + 1])
                ea, eb = int(e_off[i]), int(e_off[i + 1])
                pool = pool_all[a:b]
                n = b - a
                fam = c["e_fam"][ea:eb]
                s = fam == 0
                u = c["e_u"][ea:eb][s].astype(np.int64)
                v = c["e_v"][ea:eb][s].astype(np.int64)
                fw = c["e_fwd"][ea:eb][s] == 1
                bw = c["e_bwd"][ea:eb][s] == 1
                rel = c["e_rel"][ea:eb][s]
                self.struct_msgs += int(u.size)
                self.both += int((fw & bw).sum())
                lk = u * n + v
                rk = v * n + u
                self.one_way += int((~np.isin(rk, lk)).sum())
                hl = np.r_[u[fw], v[bw]]
                tl = np.r_[v[fw], u[bw]]
                tk.append((pool[hl] << SHIFT) | pool[tl])
                ts.append(np.r_[rel[fw], rel[bw]])
                row = C["rows"]
                C["rows"] += 1
                C["q"].append(qp[i])
                gold = np.flatnonzero(c["is_gold"][a:b])
                C["rows_gold"] += int(gold.size > 0)
                sl = c["q_seed_local"][i]
                bk = c["q_seed_bucket"][i]
                s0 = np.unique(sl[(sl >= 0) & (bk == 0)])
                if s0.size == 0:
                    C["D"].append(MAXHOP + 1)
                    continue
                C["rows_seed0"] += 1
                d_gold = gold_dist(n, u, v, gold)
                d_seed = gold_dist(n, u, v, s0)
                D = int(d_gold[s0].min())
                C["D"].append(D)
                for hops in self.hops:
                    inc = ball_incidences(pool, hl, tl, d_seed, d_gold, D, hops)
                    if inc is None:
                        continue
                    key, dr, onr = inc
                    I = C["inc"][hops]
                    I["row"].append(np.full(key.size, row, dtype=np.int64))
                    I["key"].append(key)
                    I["dir"].append(dr)
                    I["on"].append(onr)
                    C["rows_asked"][hops] += int(onr.any())
            if tk:
                k, s_, m = dedupe(np.concatenate(tk), np.concatenate(ts))
                self.slot_mismatch += m
                self.tk.append(k)
                self.ts.append(s_)
            if (ci + 1) % 64 == 0:
                self.consolidate()
                log(f"  {name}: chunk {ci + 1}/{len(paths)}, {C['rows']} rows, {self.tk[0].size if self.tk else 0} triples")
        self.consolidate()

    def finish(self):
        self.consolidate()
        keys, slots = self.tk[0], self.ts[0]
        nodes, proj = self.nk[0], self.np_[0]
        H, T = keys >> SHIFT, keys & ((1 << SHIFT) - 1)
        Hi, Ti = np.searchsorted(nodes, H), np.searchsorted(nodes, T)
        if not (np.array_equal(nodes[Hi], H) and np.array_equal(nodes[Ti], T)):
            raise SystemExit("a triple's endpoint is not in the node table")
        out = {}
        for name, C in self.carves.items():
            Dr = np.asarray(C["D"], dtype=np.int64)
            inc = {}
            for hops, I in C["inc"].items():
                key = np.concatenate(I["key"]) if I["key"] else np.zeros(0, np.int64)
                tri = np.searchsorted(keys, key)
                if key.size and not np.array_equal(keys[np.minimum(tri, keys.size - 1)], key):
                    raise SystemExit(f"{name}: a seed incidence is not a collected triple")
                inc[hops] = {"row": np.concatenate(I["row"]) if I["row"] else np.zeros(0, np.int64),
                             "tri": tri.astype(np.int64),
                             "dir": np.concatenate(I["dir"]) if I["dir"] else np.zeros(0, np.int8),
                             "on": np.concatenate(I["on"]) if I["on"] else np.zeros(0, bool)}
            out[name] = {"Q": np.stack(C["q"]).astype(np.float32), "D": Dr, "inc": inc,
                         "stats": {"rows": C["rows"], "rows_gold": C["rows_gold"], "rows_seed0": C["rows_seed0"],
                                   "seed_gold_D": {str(d): int((Dr == d).sum()) for d in range(MAXHOP + 2)},
                                   "rows_asked": {f"h{h}": int(x) for h, x in C["rows_asked"].items()}}}
        info = {"triples": int(keys.size), "nodes": int(nodes.size), "struct_messages": self.struct_msgs,
                "both_ways": self.both, "messages_without_reverse": self.one_way,
                "slot_mismatches": self.slot_mismatch, "proj_max_gap": round(self.proj_gap, 6)}
        return {"H": Hi, "T": Ti, "slots": slots, "proj": proj, "nodes": nodes}, out, info


# ── part A: typing ───────────────────────────────────────────────────────────


def main_rel(slots):
    v = np.where(slots >= 0, slots.astype(np.int64), np.iinfo(np.int64).max)
    r = v.min(1)
    return np.where(r == np.iinfo(np.int64).max, -1, r), (slots >= 0).sum(1)


class Desc:
    """The triples' label-free descriptors, built per batch of triples (a large union graph is never held at 512
    columns): TT, CARD and CTX as the docstring defines them."""

    def __init__(self, G):
        self.Pn = unit(G["proj"].astype(np.float32))
        self.H, self.T = G["H"], G["T"]
        nn = self.Pn.shape[0]
        self.outd = np.bincount(self.H, minlength=nn).astype(np.float32)
        self.ind = np.bincount(self.T, minlength=nn).astype(np.float32)
        card = np.stack([np.log1p(self.outd[self.H]), np.log1p(self.ind[self.H]), np.log1p(self.outd[self.T]),
                         np.log1p(self.ind[self.T])], 1).astype(np.float64)
        self.card = ((card - card.mean(0)) / np.maximum(card.std(0), 1e-9) / math.sqrt(2.0)).astype(np.float32)
        one = np.ones(self.H.size, dtype=np.float32)
        self.Sin = sp.csr_matrix((one, (self.T, self.H)), shape=(nn, nn)) @ self.Pn       # sum of each node's heads
        self.Sout = sp.csr_matrix((one, (self.H, self.T)), shape=(nn, nn)) @ self.Pn      # sum of each node's tails
        self.n = self.H.size

    def rows(self, idx, blocks):
        H, T, Pn = self.H[idx], self.T[idx], self.Pn
        out = []
        for b in blocks:
            if b == "TT":
                out += [Pn[H], Pn[T]]
            elif b == "CARD":
                out.append(self.card[idx])
            elif b == "CTX":
                ct = (self.Sin[T] - Pn[H]) / np.maximum(self.ind[T] - 1, 1)[:, None]
                ch = (self.Sout[H] - Pn[T]) / np.maximum(self.outd[H] - 1, 1)[:, None]
                ct[self.ind[T] <= 1] = 0.0
                ch[self.outd[H] <= 1] = 0.0
                out += [unit(ct), unit(ch)]
            else:
                raise SystemExit(f"descriptor block {b}: TT, CARD or CTX")
        return np.ascontiguousarray(np.hstack(out), dtype=np.float32)


def kmeans(rows_of, n, k, seed, sample, batch=200000):
    """k-means fitted on at most `sample` items (a seeded draw), then every item assigned; rows_of(idx) gives rows."""
    from sklearn.cluster import KMeans
    if n <= k:
        return np.arange(n)
    idx = np.arange(n) if n <= sample else np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
    km = KMeans(n_clusters=k, n_init=2 if k <= 64 else 1, max_iter=100, random_state=seed).fit(rows_of(idx))
    out = np.empty(n, dtype=np.int64)
    for a in range(0, n, batch):
        out[a:a + batch] = km.predict(rows_of(np.arange(a, min(a + batch, n))))
    return out


def nmi_geo(a, b):
    """qd_gnn8.nmi's form, I / sqrt(H(a) H(b)), on two label vectors."""
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    C = sp.coo_matrix((np.ones(ia.size), (ia, ib))).tocsr()
    C.sum_duplicates()
    P = C.data / ia.size
    rr, cc = C.nonzero()
    pa = np.asarray(C.sum(1)).ravel() / ia.size
    pb = np.asarray(C.sum(0)).ravel() / ia.size
    I = float((P * np.log(P / (pa[rr] * pb[cc]))).sum())
    Ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    Hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return I / math.sqrt(Ha * Hb) if Ha > 0 and Hb > 0 else 0.0


def grade_types(lab, rel, multi):
    m = rel >= 0
    lab, rel1, mult = lab[m], rel[m], multi[m]
    out = {"triples": int(m.sum()), "types_used": int(np.unique(lab).size), "nmi": round(nmi_geo(rel1, lab), 4)}
    one = mult == 1
    out["nmi_single"] = round(nmi_geo(rel1[one], lab[one]), 4) if one.any() else None
    _, il = np.unique(lab, return_inverse=True)
    ur, ir = np.unique(rel1, return_inverse=True)
    C = sp.coo_matrix((np.ones(il.size), (il, ir))).tocsr()
    C.sum_duplicates()
    out["purity"] = round(float(np.asarray(C.max(1).todense()).sum()) / il.size, 4)
    nl = np.asarray(C.sum(1)).ravel()
    nr = np.asarray(C.sum(0)).ravel()
    Cc = C.tocoo()
    f1 = 2 * Cc.data / (nl[Cc.row] + nr[Cc.col])
    best = np.zeros(ur.size)
    np.maximum.at(best, Cc.col, f1)
    out["f1_macro"] = round(float(best.mean()), 4)
    out["f1_weighted"] = round(float((best * nr).sum() / nr.sum()), 4)
    if ur.size <= 32:
        out["f1_per_relation"] = {int(r): round(float(f), 4) for r, f in zip(ur, best)}
    return out


def part_a(G, sets, ks, k_nodes, sample, seed=SEED):
    rel, multi = main_rel(G["slots"])
    D = Desc(G)
    Pn = D.Pn
    res = {"relations_present": int(np.unique(rel[rel >= 0]).size), "multi_relation_share": round(float((multi > 1).mean()), 4),
           "no_relation_triples": int((rel < 0).sum()), "typings": {}}
    typings = {"oracle": None}
    res["typings"]["oracle"] = grade_types(np.where(rel >= 0, rel, -1), rel, multi)
    for kn in k_nodes:
        t0 = time.time()
        c = kmeans(lambda i: Pn[i], Pn.shape[0], kn, seed, sample)
        _, lab = np.unique(c[G["H"]] * kn + c[G["T"]], return_inverse=True)
        typings[f"schema{kn}"] = lab
        res["typings"][f"schema{kn}"] = {**grade_types(lab, rel, multi), "seconds": round(time.time() - t0, 1)}
        log(f"  A schema{kn}: {res['typings'][f'schema{kn}']}")
    for s in sets:
        blocks = s.split("+")
        for k in ks:
            t0 = time.time()
            lab = kmeans(lambda i: D.rows(i, blocks), D.n, k, seed, sample)
            typings[f"{s}@{k}"] = lab
            res["typings"][f"{s}@{k}"] = {**grade_types(lab, rel, multi), "seconds": round(time.time() - t0, 1)}
            log(f"  A {s}@{k}: {res['typings'][f'{s}@{k}']}")
    return res, typings, Pn


# ── part B: alignment ────────────────────────────────────────────────────────


def typed(inc, G, lab):
    """Incidences as (row, z, on); oracle (lab None) gives one entry per relation slot of the triple."""
    if lab is None:
        S = G["slots"][inc["tri"]]
        ok = S >= 0
        rep = ok.sum(1)
        return (np.repeat(inc["row"], rep), 2 * S[ok].astype(np.int64) + np.repeat(inc["dir"].astype(np.int64), rep),
                np.repeat(inc["on"], rep))
    return inc["row"], 2 * lab[inc["tri"]] + inc["dir"].astype(np.int64), inc["on"]


def pairs(row, z, on, Kz):
    key = row.astype(np.int64) * Kz + z
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    onp = np.bincount(inv, weights=on.astype(np.float64), minlength=uk.size) > 0
    r = uk // Kz
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]]) if uk.size else np.zeros(0, np.int64)
    return {"row": r, "z": uk % Kz, "n": cnt, "on": onp, "starts": starts,
            "grp": np.repeat(np.arange(starts.size), np.diff(np.r_[starts, uk.size]))}


def descs(G, lab, Pn, K):
    """d_z: the unit mean text of the far endpoints of type z (z = 2 type + direction)."""
    if lab is None:
        S = G["slots"]
        ok = S >= 0
        tri = np.repeat(np.arange(S.shape[0]), ok.sum(1))
        ty = S[ok].astype(np.int64)
    else:
        tri, ty = np.arange(lab.size), lab
    D = np.zeros((2 * K, Pn.shape[1]), dtype=np.float32)
    one = np.ones(tri.size, dtype=np.float32)
    nn = Pn.shape[0]
    D[0::2] = sp.csr_matrix((one, (ty, G["T"][tri])), shape=(K, nn)) @ Pn
    D[1::2] = sp.csr_matrix((one, (ty, G["H"][tri])), shape=(K, nn)) @ Pn
    return unit(D)


def em(Q, pr, Kz, kappa, init, seed=SEED, iters=EM_ITERS):
    rng = np.random.default_rng(seed)
    mu = np.array(init, dtype=np.float32, copy=True)
    dead = np.linalg.norm(mu, axis=1) < 1e-6
    mu[dead] = unit(rng.standard_normal((int(dead.sum()), Q.shape[1])))
    rows_n = pr["starts"].size
    pi = np.full(Kz, 1.0 / Kz)
    Qr = Q[pr["row"]]
    trace = []
    for _ in range(iters):
        logit = np.log(pi[pr["z"]] + 1e-12) + kappa * np.einsum("kd,kd->k", Qr, mu[pr["z"]])
        mx = np.maximum.reduceat(logit, pr["starts"])
        e = np.exp(logit - mx[pr["grp"]])
        s = np.add.reduceat(e, pr["starts"])
        r = e / s[pr["grp"]]
        trace.append(round(float((np.log(s) + mx).mean()), 5))
        W = sp.csr_matrix((r.astype(np.float32), (pr["z"], pr["grp"])), shape=(Kz, rows_n))
        M = W @ Q[pr["row"][pr["starts"]]]
        mass = np.bincount(pr["z"], weights=r, minlength=Kz)
        live = mass > 1e-9
        mu[live] = unit(M[live])
        pi = mass / rows_n
    p_ = pi[pi > 0]
    return mu, pi, {"loglik": trace[::5] + [trace[-1]], "pi_perplexity": round(float(np.exp(-(p_ * np.log(p_)).sum())), 2)}


def grade_rule(pr, score, rng, Drow=None):
    """Per asked row: the AUC of its asked types against its other candidates (ties half; rows with no other
    candidate have none) and the top-1 hit (ties shared); means with a bootstrap interval, and by the row's seed-gold
    distance D (means and row counts)."""
    aucs, hits, da, dh = [], [], [], []
    st = np.r_[pr["starts"], pr["z"].size]
    for g in range(pr["starts"].size):
        a, b = st[g], st[g + 1]
        on, s = pr["on"][a:b], score[a:b]
        if not on.any():
            continue
        d = int(Drow[pr["row"][a]]) if Drow is not None else -1
        top = s == s.max()
        hits.append(float(on[top].mean()))
        dh.append(d)
        if (~on).any():
            sp_, sn = s[on], s[~on]
            aucs.append(float((sp_[:, None] > sn[None, :]).mean() + 0.5 * (sp_[:, None] == sn[None, :]).mean()))
            da.append(d)
    out = {}
    for nm, x, dd in (("auc", aucs, da), ("top1", hits, dh)):
        x = np.asarray(x)
        if x.size == 0:
            out[nm] = None
            continue
        bs = x[rng.integers(0, x.size, size=(BOOT, x.size))].mean(1)
        out[nm] = [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        out[f"{nm}_rows"] = int(x.size)
        if Drow is not None:
            dd = np.asarray(dd)
            out[f"{nm}_by_D"] = {str(d): [round(float(x[dd == d].mean()), 4), int((dd == d).sum())] for d in np.unique(dd)}
    return out


def part_b(G, F, R, lab, K, Pn, kappas, name_vec=None, tx=False, seed=SEED):
    Kz = 2 * K
    pf = pairs(*typed(F, G, lab), Kz)
    pr = pairs(*typed(R, G, lab), Kz)
    D = descs(G, lab, Pn, K)
    Qf, Qr = F["Q"], R["Q"]
    cos = lambda mu: np.einsum("kd,kd->k", Qr[pr["row"]], mu[pr["z"]])  # noqa: E731
    rules, fits = {}, {}
    pres = np.bincount(pf["z"], minlength=Kz) / max(pf["starts"].size, 1)
    rules["prior-T"] = np.log(pres[pr["z"]] + 1e-9)
    rules["count"] = np.log1p(pr["n"]).astype(np.float64)
    rules["desc"] = cos(D)
    sup_n = np.bincount(pf["z"], weights=pf["on"].astype(np.float64), minlength=Kz)
    W = sp.csr_matrix((pf["on"].astype(np.float32), (pf["z"], pf["grp"])), shape=(Kz, max(pf["starts"].size, 1)))
    mu_s = unit(W @ Qf[pf["row"][pf["starts"]]])
    pi_s = sup_n / max(int((np.bincount(pf["grp"], weights=pf["on"].astype(np.float64)) > 0).sum()), 1)
    for kappa in kappas:
        kk = f"{kappa:g}"
        mu, pi, tr = em(Qf, pf, Kz, kappa, D, seed)
        rules[f"em{kk}"] = np.log(pi[pr["z"]] + 1e-12) + kappa * cos(mu)
        rules[f"em{kk}-pi"] = np.log(pi[pr["z"]] + 1e-12)
        fits[f"em{kk}"] = tr
        mu_r, pi_r, tr_r = em(Qf, pf, Kz, kappa, np.zeros_like(D), seed)
        rules[f"em{kk}-r"] = np.log(pi_r[pr["z"]] + 1e-12) + kappa * cos(mu_r)
        fits[f"em{kk}-r"] = tr_r
        if tx:
            mu_t, pi_t, tr_t = em(Qr, pr, Kz, kappa, D, seed)
            rules[f"em{kk}-tx"] = np.log(pi_t[pr["z"]] + 1e-12) + kappa * cos(mu_t)
            fits[f"em{kk}-tx"] = tr_t
        rules[f"sup{kk}"] = np.log(pi_s[pr["z"]] + 1e-9) + kappa * cos(mu_s)
    if name_vec is not None:
        rules["name"] = cos(np.repeat(name_vec, 2, axis=0))
    rng = np.random.default_rng(seed)
    graded = {nm: grade_rule(pr, s, rng, R.get("D")) for nm, s in rules.items()}
    st = np.r_[pr["starts"], pr["z"].size]
    sizes = np.diff(st)
    asked = np.bincount(pr["grp"], weights=pr["on"].astype(np.float64)) if pr["z"].size else np.zeros(0)
    cover = {"fit_rows_with_candidates": int(pf["starts"].size), "read_rows_with_candidates": int(pr["starts"].size),
             "read_rows_asked": int((asked > 0).sum()),
             "candidates_p50": float(np.median(sizes)) if sizes.size else None,
             "asked_p50": float(np.median(asked[asked > 0])) if (asked > 0).any() else None,
             "chance_top1": round(float(np.mean((asked / np.maximum(sizes, 1))[asked > 0])), 4) if (asked > 0).any() else None}
    return {"types": int(K), "cover": cover, "rules": graded, "em": fits}


# ── run ──────────────────────────────────────────────────────────────────────


def run(a, root=LOOK, rel_dir=REL):
    t0 = time.time()
    P = proj_matrix()
    hops_list = [int(h) for h in str(a.hops).split(",") if h]
    if not hops_list or min(hops_list) < 1 or max(hops_list) > MAXHOP:
        raise SystemExit(f"--hops {a.hops}: radii from 1 to {MAXHOP}")
    col = Collector(P, hops_list)
    carves = {"fit": a.fit, "read": a.read}
    info = {}
    for role in ("fit", "read") if a.fit != a.read else ("fit",):
        paths, inf = chunk_paths(a.ds, carves[role], root, a.limit)
        info[carves[role]] = inf
        log(f"{a.ds}={carves[role]} ({role}): {inf['read']} of {inf['n_chunks']} chunks")
        col.add(role, paths)
    G, rows, ginfo = col.finish()
    if a.fit == a.read:
        rows["read"] = rows["fit"]
    log(f"union graph: {ginfo}; collected in {time.time() - t0:.0f}s")
    res = {"look": "reltype11", "args": vars(a), "script_sha256": sha(__file__), "carves": info, "graph": ginfo,
           "rows": {r: rows[r]["stats"] for r in ("fit", "read")}, "transductive": a.fit == a.read}
    sets = [s for s in a.sets.split(",") if s]
    ks = [int(k) for k in a.ks.split(",") if k]
    k_nodes = [int(k) for k in a.k_nodes.split(",") if k]
    A, typings, Pn = part_a(G, sets, ks, k_nodes, a.sample)
    res["A"] = A
    name_vec = None
    nv = Path(rel_dir) / f"{a.ds}_rel_embeddings.npy"
    if nv.exists():
        E = np.load(nv).astype(np.float32)
        name_vec = unit(unit(E) @ P)
        res["name_embeddings_sha256"] = sha(nv)
    res["B"] = {}
    kappas = [float(x) for x in a.kappa.split(",") if x]
    n_rel = max(info[c]["n_relations"] for c in info)
    for tname in [t for t in a.b_typings.split(",") if t]:
        if tname not in typings:
            raise SystemExit(f"--b-typings {tname}: not a part-A typing ({sorted(typings)})")
        lab = typings[tname]
        K = n_rel if lab is None else int(lab.max()) + 1
        res["B"][tname] = {}
        for hops in hops_list:
            t1 = time.time()
            F = {"Q": rows["fit"]["Q"], "D": rows["fit"]["D"], **rows["fit"]["inc"][hops]}
            Rr = {"Q": rows["read"]["Q"], "D": rows["read"]["D"], **rows["read"]["inc"][hops]}
            b = part_b(G, F, Rr, lab, K, Pn, kappas, name_vec if lab is None else None, tx=(a.fit != a.read))
            b["seconds"] = round(time.time() - t1, 1)
            res["B"][tname][f"h{hops}"] = b
            log(f"  B {tname} h{hops} (K {K}): cover {b['cover']}")
            for nm, g in b["rules"].items():
                log(f"     {nm:12s} auc {g['auc']} top1 {g['top1']} auc_by_D {g.get('auc_by_D')}")
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="metaqa")
    ap.add_argument("--fit", default="fit")
    ap.add_argument("--read", default="x1f")
    ap.add_argument("--sets", default="TT,CARD,CTX,TT+CARD,TT+CTX,TT+CARD+CTX")
    ap.add_argument("--ks", default="9,32")
    ap.add_argument("--k-nodes", default="8,32")
    ap.add_argument("--b-typings", default="oracle,schema32,TT@9,TT+CARD+CTX@9,TT@32,TT+CARD+CTX@32")
    ap.add_argument("--kappa", default="5,20")
    ap.add_argument("--hops", default="1,2")
    ap.add_argument("--sample", type=int, default=200000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    res = run(a)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def toy_kb(rng, n_movies=60, n_people=40, n_years=6, n_genres=5):
    """Movies -> people (directed_by 0, one each from 15 directors; starred 1, three each), -> year (2), -> genre (3, on
    two thirds of the movies). A node's text is its kind's vector plus noise."""
    dim = 1536
    kinds = {"m": rng.standard_normal(dim), "p": rng.standard_normal(dim), "y": rng.standard_normal(dim),
             "g": rng.standard_normal(dim)}
    ids, emb = [], []

    def node(kind, n):
        out = []
        for _ in range(n):
            ids.append(1000 + len(ids) * 7)
            emb.append(kinds[kind] * 1.5 + rng.standard_normal(dim))
            out.append(len(ids) - 1)
        return out
    M, Pp, Y, Gn = node("m", n_movies), node("p", n_people), node("y", n_years), node("g", n_genres)
    trip = set()
    for m in M:
        trip.add((m, int(rng.choice(Pp[:15])), 0))
        for a_ in rng.choice(Pp, 3, replace=False):
            trip.add((m, int(a_), 1))
        trip.add((m, int(rng.choice(Y)), 2))
        if rng.random() < 2 / 3:
            trip.add((m, int(rng.choice(Gn)), 3))
    return np.asarray(ids, dtype=np.int64), unit(np.asarray(emb)), sorted(trip), M, kinds


def toy_looks(root, rng, P, carves=("fit", "read"), rows_per=64, chunk=8):
    """Two look carves over the toy KB in look_score_kb's chunk layout. A row asks one relation of a movie (the seed is
    the head) or, one row in four, which movies a director directed (the seed is the tail). Its query is the asked
    (relation, direction)'s template plus a little of the seed's text and noise; the templates of year and genre lean
    toward their far endpoints' kind, as a KB's question words do. Each stored pair's two messages are each kept with
    probability 0.8 (at least one), as the look's packed batches need not hold both."""
    ids, E, trip, M, kinds = toy_kb(rng)
    dim = E.shape[1]
    far = {0: "p", 1: "p", 2: "y", 3: "g"}
    tmpl = {(r, 0): rng.standard_normal(dim) + (0.8 * kinds[far[r]] if r >= 2 else 0) for r in range(4)}
    tmpl[(0, 1)] = rng.standard_normal(dim)
    by_pair = {}
    for h, t, r in trip:
        by_pair.setdefault((h, t), []).append(r)
    out_adj, in_dir = {}, {}
    for (h, t), rs in by_pair.items():
        out_adj.setdefault(h, []).append((t, rs))
        if 0 in rs:
            in_dir.setdefault(t, []).append(h)
    directors = sorted(in_dir)
    truth = []
    for cv in carves:
        d = Path(root) / "toy" / cv
        (d / "chunks").mkdir(parents=True)
        rows = []
        for _ in range(rows_per):
            if rng.random() < 0.25:
                s = int(rng.choice(directors))
                r, dr, gold = 0, 1, sorted(in_dir[s])
            else:
                s = int(rng.choice(M))
                have = sorted({r_ for _, rs in out_adj[s] for r_ in rs})
                r, dr = int(rng.choice(have)), 0
                gold = [t for t, rs in out_adj[s] if r in rs]
            q = unit(tmpl[(r, dr)] + 0.25 * E[s] * math.sqrt(dim) + 0.3 * rng.standard_normal(dim))
            rows.append((s, r, dr, gold, q))
        n_ch = math.ceil(rows_per / chunk)
        for ci in range(n_ch):
            A = {k: [] for k in KEYS}
            for (s, r, dr, gold, q) in rows[ci * chunk:(ci + 1) * chunk]:
                nb = sorted({t for t, _ in out_adj.get(s, [])} | {h for (h, t) in by_pair if t == s})
                two = sorted({x for (h, t) in by_pair for x in (h, t) if (h in nb or t in nb) and x != s})[:25]
                extra = [int(x) for x in rng.choice(len(ids), 6, replace=False)]
                pool = list(dict.fromkeys([s] + nb + two + extra))
                loc = {g_: i for i, g_ in enumerate(pool)}
                msgs = []                                            # (u, v, fam, fwd, bwd, slots)
                for (h, t), rs in by_pair.items():
                    if h in loc and t in loc:
                        sl = (rs + [-1, -1, -1, -1])[:4]
                        keep_f = rng.random() < 0.8
                        keep_b = rng.random() < 0.8 or not keep_f
                        if keep_f:
                            msgs.append((loc[h], loc[t], 0, 1, 0, sl))
                        if keep_b:
                            msgs.append((loc[t], loc[h], 0, 0, 1, sl))
                msgs.append((0, len(pool) - 1, 1, 0, 0, [-1] * 4))  # an NER edge: never a triple
                seeds, bucket = np.full(10, -1), np.full(10, -1)
                seeds[0], bucket[0] = 0, 0
                seeds[1], bucket[1] = len(pool) - 1, 1
                isg = np.zeros(len(pool), bool)
                isg[[loc[g_] for g_ in gold]] = True
                A["q_pool_size"].append(len(pool))
                A["q_edges"].append(len(msgs))
                A["q_emb"].append(q * 3.0)
                A["q_seed_local"].append(seeds)
                A["q_seed_bucket"].append(bucket)
                A["pool"].append(ids[pool].astype(np.int32))
                A["proj"].append((E[pool] @ P).astype(np.float16))
                A["is_gold"].append(isg)
                for k, j, dt in (("e_u", 0, np.int16), ("e_v", 1, np.int16), ("e_fam", 2, np.int8), ("e_fwd", 3, np.int8),
                                 ("e_bwd", 4, np.int8)):
                    A[k].append(np.asarray([m_[j] for m_ in msgs], dtype=dt))
                A["e_rel"].append(np.asarray([m_[5] for m_ in msgs], dtype=np.int16).reshape(-1, 4))
                truth.append((cv, s, r, dr, pool, gold))
            arr = {k: (np.asarray(v) if k.startswith("q_") else np.concatenate(v)) for k, v in A.items()}
            np.savez(d / "chunks" / f"c{ci:05d}.npz", **arr)
        rec = {"full": True, "chunks": list(range(n_ch)), "n_chunks": n_ch, "proj": {"dim": PROJ_DIM, "seed": PROJ_SEED},
               "relations": {"typed": True, "offset": 0, "n_relations": 4, "k_rel": 4}}
        (d / "record_0of1.json").write_text(json.dumps(rec), encoding="utf-8")
        np.savez(d / "chunks" / "c99999.tmp.npz", x=np.zeros(1))      # not a listed chunk: never read
    return ids, E, trip, truth, tmpl


def brute_dist(n, u, v, gold):
    from collections import deque
    adj = [[] for _ in range(n)]
    for a_, b_ in zip(u, v):
        adj[a_].append(b_)
        adj[b_].append(a_)
    dist = [MAXHOP + 1] * n
    dq = deque()
    for g_ in gold:
        dist[g_] = 0
        dq.append(g_)
    while dq:
        x = dq.popleft()
        if dist[x] >= MAXHOP:
            continue
        for y in adj[x]:
            if dist[y] > dist[x] + 1:
                dist[y] = dist[x] + 1
                dq.append(y)
    return np.asarray(dist)


def selftest():
    import tempfile
    rng = np.random.default_rng(7)
    P = proj_matrix()
    for _ in range(20):
        n = int(rng.integers(2, 30))
        m = int(rng.integers(0, 60))
        u, v = rng.integers(0, n, m), rng.integers(0, n, m)
        gold = np.unique(rng.integers(0, n, int(rng.integers(0, 4))))
        assert np.array_equal(gold_dist(n, u, v, gold), brute_dist(n, u, v, gold)), "gold_dist is not BFS"
    a_ = rng.integers(0, 5, 400)
    assert abs(nmi_geo(a_, (a_ * 7 + 3) % 11) - 1.0) < 1e-9, "a relabelled copy has NMI 1"
    b_ = rng.integers(0, 3, 400)
    C = np.zeros((5, 3))
    np.add.at(C, (a_, b_), 1)
    Pj = C / C.sum()
    pa, pb = Pj.sum(1), Pj.sum(0)
    nz = Pj > 0
    I = (Pj[nz] * np.log(Pj[nz] / np.outer(pa, pb)[nz])).sum()
    assert abs(nmi_geo(a_, b_) - I / math.sqrt((pa * np.log(pa)).sum() * (pb * np.log(pb)).sum())) < 1e-9
    row = np.repeat(np.arange(30), 6)
    z = np.tile(np.arange(6), 30)
    on = rng.random(row.size) < 0.3
    pr = pairs(row, z, on, 6)
    s = rng.integers(0, 3, pr["z"].size).astype(np.float64)
    g = grade_rule(pr, s, np.random.default_rng(1))
    au, hi = [], []
    for r in range(30):
        mm = pr["row"] == r
        o, ss = pr["on"][mm], s[mm]
        if not o.any():
            continue
        hi.append(o[ss == ss.max()].mean())
        if (~o).any():
            au.append(np.mean([1.0 if x > y else 0.5 if x == y else 0.0 for x in ss[o] for y in ss[~o]]))
    assert g["auc"][0] == round(float(np.mean(au)), 4) and g["top1"][0] == round(float(np.mean(hi)), 4), "AUC/top-1"
    with tempfile.TemporaryDirectory() as td:
        ids, E, trip, truth, tmpl = toy_looks(td, rng, P)
        a = argparse.Namespace(ds="toy", fit="fit", read="read", sets="TT,CARD,TT+CARD+CTX", ks="4,8", k_nodes="4",
                               b_typings="oracle,schema4,TT@4,TT+CARD+CTX@8", kappa="5,20", hops="1,2", sample=200000,
                               limit=None, out=None)
        rel_dir = Path(td) / "rel"
        rel_dir.mkdir()
        names = np.stack([unit(tmpl[(r, 0)]) for r in range(4)]).astype(np.float16)
        np.save(rel_dir / "toy_rel_embeddings.npy", names)
        res = run(a, root=td, rel_dir=rel_dir)
        col = Collector(P, (1, 2))
        col.add("fit", chunk_paths("toy", "fit", td)[0])
        read_paths = chunk_paths("toy", "read", td)[0]
        col.add("read", read_paths)
        G, rows, info = col.finish()
        got = {(int(G["nodes"][h]), int(G["nodes"][t])): sorted(int(x) for x in s_ if x >= 0)
               for h, t, s_ in zip(G["H"], G["T"], G["slots"])}
        want = {}
        for h, t, r in trip:
            want.setdefault((int(ids[h]), int(ids[t])), []).append(r)
        inpool = set()
        for (_, _, _, _, pool, _) in truth:
            pids = {int(ids[x]) for x in pool}
            inpool |= {k for k in want if k[0] in pids and k[1] in pids}
        assert set(got) == inpool and all(got[k] == sorted(want[k]) for k in got), "collected triples differ from the toy"
        assert info["slot_mismatches"] == 0 and info["proj_max_gap"] < 1e-3 and info["messages_without_reverse"] > 0, info
        nodes_g = {int(x): i for i, x in enumerate(G["nodes"])}
        R = rows["read"]["inc"][1]
        for i, (cv, s_, r, dr, pool, gold) in enumerate([t_ for t_ in truth if t_[0] == "read"]):
            mm = R["row"] == i
            hs, ts_ = G["H"][R["tri"][mm]], G["T"][R["tri"][mm]]
            sg = nodes_g[int(ids[s_])]
            assert ((R["dir"][mm] == 0) == (hs == sg)).all() and ((R["dir"][mm] == 1) == (ts_ == sg)).all(), "direction"
            far = np.where(R["dir"][mm] == 0, ts_, hs)
            gold_g = {nodes_g[int(ids[x])] for x in gold}
            on_want = np.asarray([int(f) in gold_g for f in far])
            assert np.array_equal(R["on"][mm], on_want), (i, "on these 1-hop rows, on-path = a seed edge to a gold node")
            pids, sid = {int(ids[x]) for x in pool}, int(ids[s_])
            n_inc = sum(1 for (h, t) in want if sid in (h, t) and h in pids and t in pids)
            assert int(mm.sum()) == n_inc, (i, int(mm.sum()), n_inc, "every stored pair at the seed, once")
        # the radius-2 ball, by brute force from the chunks' messages: BFS from the seeds and from gold, then every
        # stored pair whose near endpoint is at 0 or 1 and its far one a step further, on when it lies on a shortest
        # seed-gold path
        R2 = rows["read"]["inc"][2]
        got2 = {}
        for rw, tr_, d_, o_ in zip(R2["row"], R2["tri"], R2["dir"], R2["on"]):
            got2.setdefault(int(rw), {})[(int(G["nodes"][G["H"][tr_]]), int(G["nodes"][G["T"][tr_]]), int(d_))] = bool(o_)
        row = 0
        n_far = 0
        for p in read_paths:
            with np.load(p) as z:
                c = {k: z[k] for k in KEYS}
            n_off = np.r_[0, np.cumsum(c["q_pool_size"])]
            e_off = np.r_[0, np.cumsum(c["q_edges"])]
            for i in range(c["q_pool_size"].size):
                a_, b_ = int(n_off[i]), int(n_off[i + 1])
                ea, eb = int(e_off[i]), int(e_off[i + 1])
                pool, n = c["pool"][a_:b_].astype(np.int64), b_ - a_
                s = c["e_fam"][ea:eb] == 0
                u, v = c["e_u"][ea:eb][s].astype(np.int64), c["e_v"][ea:eb][s].astype(np.int64)
                fw, bw = c["e_fwd"][ea:eb][s] == 1, c["e_bwd"][ea:eb][s] == 1
                sl, bk = c["q_seed_local"][i], c["q_seed_bucket"][i]
                s0 = np.unique(sl[(sl >= 0) & (bk == 0)])
                ds_ = brute_dist(n, u, v, s0)
                dg = brute_dist(n, u, v, np.flatnonzero(c["is_gold"][a_:b_]))
                D = min(int(dg[x]) for x in s0)
                want2 = {}
                for uu, vv, f_, b2 in zip(u, v, fw, bw):
                    for ok, (h, t) in ((f_, (uu, vv)), (b2, (vv, uu))):
                        if not ok:
                            continue
                        if ds_[t] == ds_[h] + 1 and ds_[h] < 2:
                            d_, near, far = 0, h, t
                        elif ds_[h] == ds_[t] + 1 and ds_[t] < 2:
                            d_, near, far = 1, t, h
                        else:
                            continue
                        k_ = (int(pool[h]), int(pool[t]), d_)
                        want2[k_] = want2.get(k_, False) or (D <= MAXHOP and ds_[near] + 1 + dg[far] == D)
                        n_far += int(ds_[near] == 1)
                assert got2.get(row, {}) == want2, (row, "the radius-2 ball differs from brute force")
                row += 1
        assert row == rows["read"]["stats"]["rows"] and n_far > 0, (row, n_far, "the ball reaches past the seed")
        A = res["A"]["typings"]
        assert A["oracle"]["nmi"] == 1.0 and A["oracle"]["purity"] == 1.0
        B = res["B"]["oracle"]["h1"]["rules"]
        # the name vectors are the forward templates: the reverse rows (a director's movies) have no name to match
        assert B["sup20"]["auc"][0] > 0.95 and B["name"]["auc"][0] > B["prior-T"]["auc"][0] + 0.2, B
        for tn, bh in res["B"].items():
            for hk, b in bh.items():
                for nm, tr in b["em"].items():
                    ll = tr["loglik"]
                    assert all(y >= x - 1e-4 for x, y in zip(ll, ll[1:])), (tn, hk, nm, ll, "EM lowered its likelihood")
        res2 = run(a, root=td, rel_dir=rel_dir)

        def strip(x):
            if isinstance(x, dict):
                return {k: strip(v) for k, v in x.items() if k not in ("seconds", "script_sha256")}
            return x
        assert json.dumps(strip(res), sort_keys=True) == json.dumps(strip(res2), sort_keys=True), "not deterministic"
        print("toy A:", {k: (v["nmi"], v["purity"]) for k, v in A.items()})
        for tn in res["B"]:
            for hk in res["B"][tn]:
                b = res["B"][tn][hk]
                print(f"toy B {tn} {hk} (candidates p50 {b['cover']['candidates_p50']}):",
                      {k: v["auc"][0] for k, v in b["rules"].items()})
    print("selftest: gold_dist is BFS; nmi is qd_gnn8's geometric form; AUC and top-1 match brute counting with ties; the "
          "collected triples and slots are the toy's, read from either message of a pair; unlisted chunks are skipped; "
          "each seed's stored pairs are its radius-1 incidences once, with their direction and on-path flag; the radius-2 "
          "ball matches brute force; oracle typing grades NMI 1; sup and name align queries to types; EM never lowers "
          "its likelihood; deterministic. all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
