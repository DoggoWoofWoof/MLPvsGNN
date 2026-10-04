"""Design look (untracked; not a result and not filed): S6 part 3 of the transfer plan, one model for nodes and
chains. Part 2a (chainscore18, read 11:50 on 4 Oct) learned a chain scorer g that reads 0.95 of the twin's lead
in-domain on metaqa but only AT the untyped walk zero-shot on webqsp (its W1): it trusts fine chains it cannot
choose. By seed-gold distance it loses webqsp's one-step questions (22.2 R@5 against the untyped walk's 45.2) and wins
the two-step ones (24.5 against 2.3); a per-row choice between the two would read 28.9. Many of webqsp's one-step
chains end at CVT mediators (1.59M of its 2.59M nodes), whose fact text holds their neighbours' names but which are
never gold. Swastik's direction (4 Oct): the node MLP and the chain scorer are parts of one model, coupled by an EM
refinement and trained end to end, so that what the nodes look like can correct which chain the scorer trusts.

The model. Per row, two parts and one loss:
    f   an MLP (two hidden layers of 64, ReLU) over 13 graph-agnostic node features (below): a score per pool node
    g   chainscore18's pair MLP over its 33 features plus cos_s (the cosine of the chain's end text to the seeds'
        pooled text): a logit per (row, candidate chain) at every level (none, dir, tt<k>, exact), and the prior
        log p0 = log_softmax(g - logC) over the row's candidates, as chainscore18
The EM coupling (m_c is a chain's end distribution over the row's non-seed nodes, chainpop17's):
    s_0 = f, and for t = 1..T, with pi_{t-1} = softmax(s_{t-1}) over the row's non-seed nodes:
        E   ev_c = sum_v m_c(v) pi_{t-1}(v);   log p_t = log p0 + beta log(ev_c + 1e-8), normalised over the row
        M   agg_t(v) = sum_c p_t(c) m_c(v);    s_t = f + gamma log(eps + agg_t)
    T = 0 is one product:  s = f + gamma log(eps + sum_c p0(c) m_c(v))
beta and gamma (softplus, started at 1) and eps (exp, started at 1e-3) are learned. The E step reweights each chain
by the node-score mass its ends hold (the chain posterior when the answer is drawn from pi); the M step mixes the
chains' end distributions back into the node scores. The loss, per (row, regime), is -log of softmax(s_T) summed over
the row's in-pool non-seed gold (rows with such gold), plus chainscore18's chain loss on p_T: -log of p_T on the
chains that alone reach the row's best single-chain R@5 (rows with one; every level allowed). Ranking: s_T over the
non-seed nodes, ties by rrf rank, seeds last (chainpop17's rule).

Node features (graph-agnostic; computed on the row's graph after its transform, so a mediator gets them too):
    cq      cos(q, the node's text)                 cs      cos(the rank-1 seeds' pooled text, the node's text)
    rr      -log(1 + rrf rank) / log(pool size)
    d1 d2 d3 dinf   one-hot undirected hop distance from the rank-1 seeds (1, 2, 3, more)
    deg     log(1 + distinct neighbours)            nseed1  a seed of a lower bucket (not rank 1)
    w1 w2 w3        log(1 + n_ns m_h(v)): m_h the untyped walk's end distribution after h steps (the none level's
                    chain of length h, always to 3 steps), n_ns the row's non-seed nodes, so 1 is uniform mass
    qnb     the largest cq among the node's neighbours

Granularity regimes (option A; augmentation of metaqa's fit carve, training only). A transform rewrites the graph's
relation ids or its structure; the levels (endpoint-text descriptors over the carve's union graph, Ward cut, exact),
the walks, the population regimes' fits and every feature are then rebuilt on the transformed graph:
    id    none
    al4   r -> 4r + a hash of (global head, global tail, r) mod 4: each relation split into four aliases at random,
          as a KB with many relation ids for one meaning (webqsp's graph has 4,347). An alias keeps r's name.
    sp4   r -> 4r + the tail's text cluster (spherical k-means, K 4, over the union graph's nodes): relations split
          by what they point at. r's name.
    mg3   r -> perm(r) // 3: relations merged three at a time, coarser than the KB's own. The members' mean name.
    rf    a random half of the present relations reified: (h, r, t) -> (h, r, m) and (m, n_rel + r, t), one
          mediator m per distinct (h, r, t), its text n(p_h + p_t), its rrf max(rrf_h, rrf_t), never gold, as
          webqsp's CVTs. Both halves keep r's name.
al4 and sp4 use hops 2, ks 2,4,16 and the regimes 1, 0.05, 0 on the even (al4) and odd (sp4) rows; rf hops 3, ks 2,4
and the same three regimes on the even rows; mg3 hops 3, ks 2 and chainscore18's six regimes on every row. A relabel
(al4, sp4, mg3) leaves the graph's structure, the none level and the node features as id's (--selftest checks it).
A row's index (even, odd) counts the carve's rows in chunk order; the union graph is always the whole carve's.

Arms (one seed; the epoch chosen by R@5 on metaqa's select carve, own population, whole-carve regime):
    g18     chainscore18's full arm (its train_arm, unchanged) on this file's id builds: the repro (R0)
    g       f = 0 and T = 0: the chain scorer alone, trained with both losses
    f       no g: the node MLP alone (node loss)
    fg      T = 0: one product of f and g's mixture
    em      T = 2: the EM coupling
    g-gr, f-gr, em-gr   g, f and em trained on the id pool and the four transforms
Training: Adam 1e-3, weight decay 1e-4, 12 epochs of 5,960 examples in batches of 64. An example is a (transform, row,
regime): the transform uniform among the arm's, the row uniform among its rows with in-pool gold or an on chain, the
regime uniform among the build's. Inputs standardised on the id fit build (pairs over every regime). torch
deterministic on one thread.

Reads: metaqa x1f (in-domain: other rows of the graph trained on); webqsp selectf (zero-shot, 305 rows, its own
population; the twin valid); webqsp sf (zero-shot: selectf and webqsp's fit carve, 1,544 rows read as one
population; the six pair trained on the fit rows, so on sf an arm is set against the walks only). The walks
(none/rd and the other level/fit posterior walks) are chainscore18's reference arms, recomputed from the same builds.

Verdicts, fixed at 14:28 on 4 Oct before any number from this file. Paired row bootstrap (BOOT 1000) on R@5:
    J1  webqsp sf: em-gr minus none/rd (part 1's best label-free arm on webqsp): ABOVE / AT / BELOW. em the same,
        reported beside it
    J2  em minus fg, on each read: HELPS / HURTS / SAME (does iterating add to one product?)
    J3  em minus g, em minus f, em-gr minus g-gr, em-gr minus f-gr, on each read (does the joint model beat its parts?)
    J4  em-gr minus em, g-gr minus g, f-gr minus f, on each read (do the granularity regimes help?)
    J5  webqsp sf by seed-gold distance D (1 and 2): em-gr minus none/rd: ABOVE / AT / BELOW
    M1  metaqa: em and em-gr against g18 and against the twin (ABOVE / AT / BELOW), and their share of the twin's lead
        over rrf-s (HIGH if its interval lies at or above 0.5, LOW if below 0.25, else MID)
    Z1  webqsp selectf: em-gr and em against l7g-j3a's band (0.20 to 0.25 R@5): ABOVE / WITHIN / BELOW, and their
        share of the twin's lead
    R0  g18's per-row reads equal chainscore18's full arm's (|diff| < 1e-6) on metaqa and on webqsp selectf:
        REPRODUCES / DIFFERS
If J1 is ABOVE for em-gr, the GNN takes f's place in the same coupling and option B (passage graphs typed by
endpoint-text clusters) follows. If not, the result goes to Swastik first. Train-split rows throughout: a look, not a
result.

    python outputs/mp_unified/chainscore19.py build --ds metaqa --carve fit --hops 3 --ks 2,4 \
        --regimes 1,0.25,0.05,0.01,0.002,0 --out outputs/mp_unified/lean/cs19-mq-fit-id.npz
    python outputs/mp_unified/chainscore19.py build --ds metaqa --carve fit --transform al4 --rows even --hops 2 \
        --ks 2,4,16 --regimes 1,0.05,0 --out outputs/mp_unified/lean/cs19-mq-fit-al4.npz
    (sp4: --rows odd, as al4; mg3: --rows all --hops 3 --ks 2 and the six regimes; rf: --rows even --hops 3 --ks 2,4
     --regimes 1,0.05,0)
    python outputs/mp_unified/chainscore19.py build --ds metaqa --carve select --hops 3 --ks 2,4 \
        --out outputs/mp_unified/lean/cs19-mq-select.npz
    python outputs/mp_unified/chainscore19.py build --ds metaqa --carve x1f --hops 3 --ks 2,4 \
        --out outputs/mp_unified/lean/cs19-mq-x1f.npz
    python outputs/mp_unified/chainscore19.py build --ds webqsp --carve selectf --hops 2 --ks 4,16,64,256 \
        --out outputs/mp_unified/lean/cs19-wq.npz
    python outputs/mp_unified/chainscore19.py build --ds webqsp --carve selectf+fitf --hops 2 --ks 4,16,64,256 \
        --out outputs/mp_unified/lean/cs19-wq-sf.npz
    python outputs/mp_unified/chainscore19.py train --arm em-gr --fit outputs/mp_unified/lean/cs19-mq-fit-id.npz \
        --aug al4=outputs/mp_unified/lean/cs19-mq-fit-al4.npz (and sp4, mg3, rf) \
        --select outputs/mp_unified/lean/cs19-mq-select.npz --read metaqa=outputs/mp_unified/lean/cs19-mq-x1f.npz \
        --read webqsp=outputs/mp_unified/lean/cs19-wq.npz --read webqsp_sf=outputs/mp_unified/lean/cs19-wq-sf.npz \
        --out outputs/mp_unified/lean/cs19-em-gr.json --rows-out outputs/mp_unified/lean/cs19-em-gr.rows.npz
    python outputs/mp_unified/chainscore19.py grade --arm em-gr=outputs/mp_unified/lean/cs19-em-gr.json (each arm) \
        --read metaqa=... --read webqsp=... --read webqsp_sf=... --cs18-rows outputs/mp_unified/lean/cs18.rows.npz \
        --out outputs/mp_unified/lean/cs19.json
    python outputs/mp_unified/chainscore19.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainpop17's and chainscore18's count, so the id builds reproduce theirs

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainpop17 as CP  # noqa: E402
import chainscore18 as CS  # noqa: E402
import reltype11 as RT  # noqa: E402

LOOK, REL = RT.LOOK, RT.REL
SEED = CS.SEED
BOOT = 1000
EPOCHS = 12
PER_EPOCH = 5960
BATCH = 64
LR = 1e-3
WD = 1e-4
HID = 64
T_EM = 2
EPS0 = 1e-3
NEG = -1e9
NODE_HOPS = 3
J3A = (0.20, 0.25)
NODEF = ("cq", "cs", "rr", "d1", "d2", "d3", "dinf", "deg", "nseed1", "w1", "w2", "w3", "qnb")
PAIRF = CS.ALLF + ("cos_s",)
TRANSFORMS = ("id", "al4", "sp4", "mg3", "rf")
ARMS = ("g18", "g", "f", "fg", "em", "g-gr", "f-gr", "em-gr")
NONE = CP.Level("none")
U64 = np.uint64
log, unit, sha = RT.log, RT.unit, CS.sha


def arm_spec(arm):
    base, gr = (arm[:-3], True) if arm.endswith("-gr") else (arm, False)
    spec = {"g": {"f": False, "g": True, "T": 0}, "f": {"f": True, "g": False, "T": 0},
            "fg": {"f": True, "g": True, "T": 0}, "em": {"f": True, "g": True, "T": T_EM}}[base]
    return dict(spec, gr=gr)


# ── transforms ───────────────────────────────────────────────────────────────


def mix64(x):
    """splitmix64's finaliser, wrapping on uint64."""
    x = np.array(x, dtype=np.uint64, copy=True)
    with np.errstate(over="ignore"):
        x ^= x >> U64(30)
        x *= U64(0xBF58476D1CE4E5B9)
        x ^= x >> U64(27)
        x *= U64(0x94D049BB133111EB)
        x ^= x >> U64(31)
    return x


def alias(hg, tg, r, k):
    """A hash of (global head, global tail, relation) mod k: the same triple gets the same alias in every pool."""
    key = (np.asarray(hg, np.int64).astype(np.uint64) << U64(32)) | np.asarray(tg, np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = mix64(mix64(key) ^ ((np.asarray(r, np.int64).astype(np.uint64) + U64(1)) * U64(0x9E3779B97F4A7C15)))
    return (h % U64(k)).astype(np.int64)


def skmeans(X, K, seed, iters=25):
    """Spherical k-means with a k-means++ start on cosine distance; deterministic for a seed."""
    rng = np.random.default_rng(seed)
    X = unit(X)
    n = X.shape[0]
    C = np.zeros((K, X.shape[1]), np.float32)
    C[0] = X[rng.integers(n)]
    dmin = np.maximum(1.0 - X @ C[0], 0.0).astype(np.float64)
    for k in range(1, K):
        p = dmin / dmin.sum() if dmin.sum() > 0 else np.full(n, 1.0 / n)
        C[k] = X[rng.choice(n, p=p)]
        dmin = np.minimum(dmin, np.maximum(1.0 - X @ C[k], 0.0))
    for _ in range(iters):
        lab = np.argmax(X @ C.T, 1)
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = unit(X[m].sum(0))
    return np.argmax(X @ C.T, 1).astype(np.int64)


class Transform:
    """A rewrite of the carve's relation ids (al4, sp4, mg3) or of its structure (rf), applied alike to the union
    graph (for the levels) and to every row (for the walks and features)."""

    def __init__(self, name, n_rel, G, Pn, present):
        if name not in TRANSFORMS:
            raise SystemExit(f"unknown transform {name}")
        self.name, self.n_rel0, self.nodes = name, n_rel, G["nodes"]
        self.info = {"name": name}
        if name == "id":
            self.n_rel = n_rel
        elif name in ("al4", "sp4"):
            self.n_rel = 4 * n_rel
            if name == "sp4":
                self.clus = skmeans(Pn, 4, SEED + 4)
                self.info["cluster_sizes"] = np.bincount(self.clus, minlength=4).tolist()
        elif name == "mg3":
            self.perm = np.random.default_rng(SEED + 3).permutation(n_rel)
            self.n_rel = int(self.perm.max()) // 3 + 1
            self.info["groups"] = [np.flatnonzero(self.perm // 3 == j).tolist() for j in range(self.n_rel)]
        else:
            idx = np.flatnonzero(present)
            pick = np.random.default_rng(SEED + 5).choice(idx, size=idx.size // 2, replace=False)
            self.reif = np.zeros(n_rel, bool)
            self.reif[pick] = True
            self.n_rel = 2 * n_rel
            self.info["reified"] = sorted(int(x) for x in pick)
        self.info["n_rel"] = self.n_rel

    def relabel(self, S, hg, tg):
        """Relation slots S (triples x slots, -1 empty) of triples with global endpoints hg, tg."""
        if self.name in ("id", "rf"):
            return S
        ok = S >= 0
        out = np.full(S.shape, -1, np.int64)
        r = S[ok].astype(np.int64)
        if self.name == "mg3":
            out[ok] = self.perm[r] // 3
            return out
        tri = np.nonzero(ok)[0]
        if self.name == "al4":
            out[ok] = 4 * r + alias(hg[tri], tg[tri], r, 4)
        else:
            out[ok] = 4 * r + self.clus[np.searchsorted(self.nodes, tg[tri])]
        return out

    def names(self, name_vec):
        if name_vec is None or self.name == "id":
            return name_vec
        if self.name in ("al4", "sp4"):
            return np.repeat(name_vec, 4, axis=0)
        if self.name == "mg3":
            out = np.zeros((self.n_rel, name_vec.shape[1]))
            np.add.at(out, self.perm // 3, name_vec.astype(np.float64))
            return unit(out)
        return np.r_[name_vec, name_vec]

    def descriptors(self, G, Pn):
        """chainpop17's endpoint descriptors on the transformed union graph: tt (n_rel x 2d) and present."""
        S = G["slots"]
        ok = S >= 0
        tri = np.repeat(np.arange(S.shape[0]), ok.sum(1))
        nn = Pn.shape[0]
        if self.name != "rf":
            ty = self.relabel(S, self.nodes[G["H"]], self.nodes[G["T"]])[ok].astype(np.int64)
            one = np.ones(tri.size)
            Dt = sp.csr_matrix((one, (ty, G["T"][tri])), shape=(self.n_rel, nn)) @ Pn
            Dh = sp.csr_matrix((one, (ty, G["H"][tri])), shape=(self.n_rel, nn)) @ Pn
            present = np.bincount(ty, minlength=self.n_rel) > 0
        else:
            ty = S[ok].astype(np.int64)
            m = self.reif[ty]
            h, t = G["H"][tri], G["T"][tri]
            one = np.ones(int((~m).sum()))
            Dt = np.asarray(sp.csr_matrix((one, (ty[~m], t[~m])), shape=(self.n_rel, nn)) @ Pn, dtype=np.float64)
            Dh = np.asarray(sp.csr_matrix((one, (ty[~m], h[~m])), shape=(self.n_rel, nn)) @ Pn, dtype=np.float64)
            hm, tm, rm = h[m], t[m], ty[m]
            med = unit(Pn[hm] + Pn[tm]).astype(np.float64)
            np.add.at(Dt, rm, med)
            np.add.at(Dh, rm, Pn[hm].astype(np.float64))
            np.add.at(Dt, self.n_rel0 + rm, Pn[tm].astype(np.float64))
            np.add.at(Dh, self.n_rel0 + rm, med)
            present = np.bincount(np.r_[ty, self.n_rel0 + rm], minlength=self.n_rel) > 0
        tt = np.c_[unit(Dt), unit(Dh)] / math.sqrt(2)
        return tt, present

    def reify_row(self, n, hl, tl, sl, pn, rrf):
        """rf on one row: the distinct (h, t, r) of the row's triples, the reified ones through a mediator each
        (appended after the pool's nodes). Returns the new pool size, triples (one slot each), texts and rrf."""
        ok = sl >= 0
        k = ok.sum(1)
        key = np.unique((np.repeat(hl, k) * n + np.repeat(tl, k)) * self.n_rel0 + sl[ok].astype(np.int64))
        r = key % self.n_rel0
        ht = key // self.n_rel0
        h, t = ht // n, ht % n
        m = self.reif[r]
        M = int(m.sum())
        med = n + np.arange(M, dtype=np.int64)
        hl2 = np.r_[h[~m], h[m], med]
        tl2 = np.r_[t[~m], med, t[m]]
        sl2 = np.r_[r[~m], r[m], self.n_rel0 + r[m]][:, None]
        pn2 = np.r_[pn, unit(pn[h[m]] + pn[t[m]])].astype(np.float32)
        rrf2 = np.r_[rrf, np.maximum(rrf[h[m]], rrf[t[m]])]
        return n + M, hl2, tl2, sl2, pn2, rrf2


# ── build: walks, regimes, node and pair features ────────────────────────────


def node_features(n, hl, tl, sl, pn, q, seeds, seeds1, rank, sbar):
    F = np.zeros((n, len(NODEF)), np.float32)
    cq = pn @ q
    F[:, 0] = cq
    if seeds.size:
        F[:, 1] = pn @ sbar
    F[:, 2] = -np.log1p(rank) / math.log(max(n, 2))
    g = CP.typed_graph(n, hl, tl, sl, NONE)
    src, dst = g[0], g[2]
    nb = src != dst
    F[:, 7] = np.log1p(np.bincount(src[nb], minlength=n))
    F[seeds1, 8] = 1.0
    qn = np.full(n, -np.inf, np.float32)
    np.maximum.at(qn, dst[nb], cq[src[nb]])
    F[:, 12] = np.where(np.isfinite(qn), qn, 0.0)
    if seeds.size:
        dist = RT.gold_dist(n, src, dst, seeds)
        F[:, 3], F[:, 4], F[:, 5], F[:, 6] = dist == 1, dist == 2, dist == 3, dist > 3
        ch, _ = CP.walk(n, g, seeds, NODE_HOPS, CP.MAX_CHAINS)
        nns = n - seeds.size
        for types, nodes, mass in CP.candidates(ch, seeds):
            F[nodes, 8 + len(types)] = np.log1p(nns * mass)
    else:
        F[:, 6] = 1.0
    return F


def collect19(paths, pcarve, levels, P, hops, cap, rrf_col, m_idx, tf, keep_row):
    """chainscore18.collect's pass (its helpers, unchanged) over the kept rows after the transform, keeping every
    row's chain node lists (int32), and adding each row's node features and each pair's cos_s."""
    L = len(levels)
    R = {k: [] for k in ("Q", "n", "gold", "gt", "rank", "order_ns", "seeds_rr", "D", "twin0", "rrf", "rrf_s", "XN",
                         "seed0", "carve", "row")}
    V = [{"key": {}, "types": []} for _ in range(L)]
    RD = [CP.Grow(RT.PROJ_DIM) for _ in range(L)]
    C = [{"cid": [], "ptr": [], "node": [], "mass": [], "met": [], "met32": [], "e": [], "ws": [], "cs": []}
         for _ in range(L)]
    st = {"rows": 0, "rows_seed0": 0, "capped": [0] * L, "no_chain": [0] * L, "nodes": 0, "mediators": 0,
          "incidence": 0}
    seen = {}
    for ci, (p, cv) in enumerate(zip(paths, pcarve)):
        with np.load(p) as zf:
            c = {k: zf[k] for k in CP.KEYS}
        n_off = np.r_[0, np.cumsum(c["q_pool_size"])]
        e_off = np.r_[0, np.cumsum(c["q_edges"])]
        Qc = unit(unit(c["q_emb"]) @ P)
        for i in range(c["q_pool_size"].size):
            gi = seen.get(cv, 0)
            seen[cv] = gi + 1
            if not keep_row(gi):
                continue
            a, b = int(n_off[i]), int(n_off[i + 1])
            ea, eb = int(e_off[i]), int(e_off[i + 1])
            n = b - a
            hl, tl, sl, u, v = CP.row_triples(c, a, b, ea, eb)
            gold = c["is_gold"][a:b].astype(bool)
            gt = int(c["q_gold_total"][i])
            rrf = c["x"][a:b, rrf_col].astype(np.float64)
            pn = unit(c["proj"][a:b].astype(np.float32))
            if tf.name == "rf":
                n0 = n
                n, hl, tl, sl, pn, rrf = tf.reify_row(n, hl, tl, sl, pn, rrf)
                gold = np.r_[gold, np.zeros(n - n0, bool)]
                u, v = hl, tl
                st["mediators"] += n - n0
            elif tf.name != "id":
                pool = c["pool"][a:b].astype(np.int64)
                sl = tf.relabel(sl, pool[hl], pool[tl])
            order, rank = CP.rr_order(rrf)
            sl0 = c["q_seed_local"][i]
            bk = c["q_seed_bucket"][i]
            seeds = np.unique(sl0[(sl0 >= 0) & (bk == 0)]).astype(np.int64)
            seeds1 = np.setdiff1d(np.unique(sl0[(sl0 >= 0) & (bk >= 1)]).astype(np.int64), seeds)
            is_seed = np.zeros(n, bool)
            is_seed[seeds] = True
            order_ns = order[~is_seed[order]]
            seeds_rr = order[is_seed[order]]
            sbar = unit(pn[seeds].sum(0)) if seeds.size else np.zeros(RT.PROJ_DIM, np.float32)
            R["Q"].append(Qc[i])
            R["n"].append(n)
            R["gold"].append(np.flatnonzero(gold).astype(np.int32))
            R["gt"].append(gt)
            R["rank"].append(rank.astype(np.int32))
            R["order_ns"].append(order_ns.astype(np.int32))
            R["seeds_rr"].append(seeds_rr.astype(np.int32))
            R["twin0"].append(c["q_metrics"][i, 0, m_idx])
            R["rrf"].append(CP.metrics(order[:5], gold, gt))
            R["rrf_s"].append(CP.metrics(CP.top5(np.zeros(0, np.int64), np.zeros(0), rank, order_ns, seeds_rr), gold,
                                         gt))
            R["XN"].append(node_features(n, hl, tl, sl, pn, Qc[i], seeds, seeds1, rank, sbar).astype(np.float16))
            R["seed0"].append(is_seed)
            R["carve"].append(cv)
            R["row"].append(gi)
            st["nodes"] += n
            if seeds.size:
                st["rows_seed0"] += 1
                dg = RT.gold_dist(n, u, v, np.flatnonzero(gold))
                R["D"].append(int(dg[seeds].min()))
            else:
                R["D"].append(RT.MAXHOP + 1)
            for li, lev in enumerate(levels):
                g = CP.typed_graph(n, hl, tl, sl, lev)
                ch, capped = CP.walk(n, g, seeds, hops, cap)
                st["capped"][li] += int(capped)
                cand = CP.candidates(ch, seeds)
                st["no_chain"][li] += int(not cand)
                Vl = V[li]
                cid = np.empty(len(cand), np.int64)
                for j, (types, _, _) in enumerate(cand):
                    k = lev.key(types)
                    x = Vl["key"].get(k)
                    if x is None:
                        x = len(Vl["types"])
                        Vl["key"][k] = x
                        Vl["types"].append(types)
                    cid[j] = x
                lens = np.asarray([len(nd) for _, nd, _ in cand], np.int64)
                nodes = np.concatenate([nd for _, nd, _ in cand]) if cand else np.zeros(0, np.int64)
                mass = np.concatenate([m for _, _, m in cand]) if cand else np.zeros(0)
                met = np.zeros((len(cand), 3), np.float32)
                met32 = np.zeros((len(cand), 3), np.float32)
                ws = np.zeros((len(cand), 6), np.float32)
                for j, (types, nd, m) in enumerate(cand):
                    met[j] = CP.metrics(CP.top5(nd, m, rank, order_ns, seeds_rr), gold, gt)
                    met32[j] = CP.metrics(CP.top5(nd, m.astype(np.float32).astype(np.float64), rank, order_ns,
                                                  seeds_rr), gold, gt)
                    mp = m[m > 0]
                    ent = float(-(mp * np.log(mp)).sum())
                    ws[j] = (len(types), CS.fwd_share(types, lev), math.log1p(nd.size),
                             ent / math.log(nd.size) if nd.size > 1 else 0.0, float(m.max()), float(rank[nd].min()))
                if cand:
                    W = sp.csr_matrix((mass, (np.repeat(np.arange(len(cand)), lens), nodes)), shape=(len(cand), n))
                    E_ = np.asarray(W @ pn)
                    RD[li].ensure(len(Vl["types"]))
                    RD[li].a[cid] += E_
                    csv = (unit(E_) @ sbar).astype(np.float32)
                else:
                    E_ = np.zeros((0, RT.PROJ_DIM))
                    csv = np.zeros(0, np.float32)
                C[li]["cid"].append(cid)
                C[li]["met"].append(met)
                C[li]["met32"].append(met32)
                C[li]["e"].append(E_.astype(np.float16))
                C[li]["ws"].append(ws)
                C[li]["cs"].append(csv)
                C[li]["ptr"].append(np.r_[0, np.cumsum(lens)])
                C[li]["node"].append(nodes.astype(np.int32))
                C[li]["mass"].append(mass.astype(np.float32))
                st["incidence"] += int(nodes.size)
            st["rows"] += 1
        if (ci + 1) % 25 == 0 or ci + 1 == len(paths):
            log(f"  chunk {ci + 1}/{len(paths)}: {st['rows']} rows, {st['nodes']} nodes, {st['incidence']} "
                f"chain-node entries; chains " + ", ".join(f"{lev.name} {len(V[li]['types'])}"
                                                           for li, lev in enumerate(levels)))
    if not st["rows"]:
        raise SystemExit("no row was kept")
    R["Q"] = np.stack(R["Q"]).astype(np.float32)
    for k in ("twin0", "rrf", "rrf_s"):
        R[k] = np.asarray(R[k], np.float64)
    R["D"] = np.asarray(R["D"], np.int64)
    rd = [unit(RD[li].a[:len(V[li]["types"])]) for li in range(L)]
    return R, V, C, rd, st


def build19(a, root=LOOK, rel_dir=REL):
    t0 = time.time()
    P = RT.proj_matrix()
    carves = [x for x in str(a.carve).split("+") if x]
    paths, pcarve, infos, rec = [], [], [], None
    for k, cv in enumerate(carves):
        ps, info = RT.chunk_paths(a.ds, cv, root, a.limit)
        r_ = CP.read_record(a.ds, cv, root)
        if rec is None:
            rec = r_
        elif r_ != rec:
            raise SystemExit(f"{a.ds}={cv}: its record's rrf column or metrics differ from {carves[0]}'s")
        if infos and info["n_relations"] != infos[0]["n_relations"]:
            raise SystemExit(f"{a.ds}={cv}: {info['n_relations']} relations, not {infos[0]['n_relations']}")
        paths += ps
        pcarve += [k] * len(ps)
        infos.append(info)
    rrf_col, m_idx = rec
    n_rel = infos[0]["n_relations"]
    log(f"{a.ds}={a.carve}: " + ", ".join(f"{cv} {inf['read']} of {inf['n_chunks']} chunks"
                                          for cv, inf in zip(carves, infos))
        + f"; {n_rel} relations; transform {a.transform}, rows {a.rows}")
    col = RT.Collector(P, (1,))
    col.add("read", paths)
    G, _rows, ginfo = col.finish()
    Pn = unit(G["proj"].astype(np.float32))
    ty0 = G["slots"][G["slots"] >= 0].astype(np.int64)
    if ty0.size and int(ty0.max()) >= n_rel:
        raise SystemExit(f"a relation id {int(ty0.max())} is past the record's {n_rel}")
    tf = Transform(a.transform, n_rel, G, Pn, np.bincount(ty0, minlength=n_rel) > 0)
    tt, present = tf.descriptors(G, Pn)
    ks = [int(k) for k in str(a.ks).split(",") if k]
    levels = CP.hierarchy(tt, present, ks, tf.n_rel)
    log(f"union graph {ginfo['triples']} triples; after {tf.name} {int(present.sum())} of {tf.n_rel} relations "
        "present; levels " + ", ".join(f"{lev.name} (Kz {lev.Kz})" for lev in levels))
    name_vec, name_sha = None, None
    nv = Path(rel_dir) / f"{a.ds}_rel_embeddings.npy"
    if nv.exists():
        E = np.load(nv).astype(np.float32)
        if E.shape[0] != n_rel:
            raise SystemExit(f"{nv}: {E.shape[0]} names for {n_rel} relations")
        name_vec = tf.names(unit(unit(E) @ P))
        name_sha = sha(nv)
    keep_row = {"all": lambda i: True, "even": lambda i: i % 2 == 0, "odd": lambda i: i % 2 == 1}[a.rows]
    t1 = time.time()
    R, V, C, rd_all, st = collect19(paths, pcarve, levels, P, a.hops, a.max_chains, rrf_col, m_idx, tf, keep_row)
    N, L = st["rows"], len(levels)
    A = CP.anchors(levels, V, rd_all, name_vec, present)
    log(f"walks in {time.time() - t1:.0f}s")
    rowoff, levoff, p_lev, cnt = CS.layout(C, N, L)
    FIa = CS.fixed_features(R, C, levels, A, rowoff, levoff, cnt)
    regs = [float(x) for x in str(a.regimes).split(",") if x]
    if regs[0] != 1.0:
        raise SystemExit("the first regime must be 1 (the whole carve)")
    t2 = time.time()
    FRa, gids, post1, summ = CS.regime_features(R, C, V, levels, A, rd_all, regs, rowoff, levoff,
                                                np.random.default_rng(CS.SEED), True)
    log(f"regimes in {time.time() - t2:.0f}s")
    pick = [(li, i) for i in range(N) for li in range(L)]
    met = np.concatenate([C[li]["met"][i] for li, i in pick])
    met32 = np.concatenate([C[li]["met32"][i] for li, i in pick])
    p_cid = np.concatenate([C[li]["cid"][i] for li, i in pick])
    FS = np.concatenate([C[li]["cs"][i] for li, i in pick]).astype(np.float32)
    gold_off = np.r_[0, np.cumsum([x.size for x in R["gold"]])].astype(np.int64)
    meta = {"look": "chainscore19", "args": dict(vars(a)), "script_sha256": sha(__file__),
            "chainscore18_sha256": sha(CS.__file__), "chainpop17_sha256": sha(CP.__file__),
            "reltype11_sha256": sha(RT.__file__), "carves": carves, "carve": infos, "graph": ginfo,
            "transform": tf.info, "relations_present": int(present.sum()), "name_embeddings_sha256": name_sha,
            "levels": [{"name": lev.name, "Kz": lev.Kz, "vocab": len(V[li]["types"]),
                        "pairs": int(cnt[:, li].sum()), "parent_missing": A[li]["parent_missing"]}
                       for li, lev in enumerate(levels)],
            "rows": N, "rows_seed0": st["rows_seed0"], "capped": st["capped"], "no_chain": st["no_chain"],
            "pairs": int(rowoff[-1]), "nodes": st["nodes"], "mediators": st["mediators"],
            "incidence": st["incidence"], "regimes": regs, "fits": summ}
    arr = {"FI": FIa, "FR": FRa, "gid": gids, "rowoff": rowoff, "p_lev": p_lev, "p_cid": p_cid, "met": met,
           "met32": met32, "D": R["D"], "n": np.asarray(R["n"], np.int32), "gt": np.asarray(R["gt"], np.int32),
           "gold": np.concatenate(R["gold"]).astype(np.int32), "gold_off": gold_off, "twin0": R["twin0"],
           "rrf": R["rrf"], "rrf_s": R["rrf_s"], "kinds": np.asarray([CS.kind(lev.name) for lev in levels]),
           "level_names": np.asarray([lev.name for lev in levels])}
    parts_n, parts_m, ends = [], [], []
    base = 0
    for li, i in pick:
        pt = C[li]["ptr"][i]
        parts_n.append(C[li]["node"][i])
        parts_m.append(C[li]["mass"][i])
        ends.append(base + pt[1:])
        base += int(pt[-1])
    arr["node"] = np.concatenate(parts_n).astype(np.int32)
    arr["mass"] = np.concatenate(parts_m).astype(np.float32)
    arr["node_off"] = np.r_[0, np.concatenate(ends)].astype(np.int64)
    for k, ko in (("rank", "rank_off"), ("order_ns", "order_off"), ("seeds_rr", "seeds_off")):
        arr[k] = np.concatenate(R[k]).astype(np.int32)
        arr[ko] = np.r_[0, np.cumsum([x.size for x in R[k]])].astype(np.int64)
    arr["post1"] = post1
    arr["FS"] = FS
    arr["XN"] = np.concatenate(R["XN"]).astype(np.float16)
    arr["xn_off"] = np.r_[0, np.cumsum(R["n"])].astype(np.int64)
    arr["seed0"] = np.concatenate(R["seed0"])
    arr["carve_ix"] = np.asarray(R["carve"], np.int8)
    arr["row_ix"] = np.asarray(R["row"], np.int64)
    meta["seconds"] = round(time.time() - t0, 1)
    arr["meta"] = np.asarray(json.dumps(meta))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **arr)
    os.replace(tmp, out)
    log(f"built {N} rows, {st['nodes']} nodes ({st['mediators']} mediators), {int(rowoff[-1])} pairs, "
        f"{st['incidence']} chain-node entries, {len(regs)} regime(s) in {time.time() - t0:.0f}s -> {out}")
    return meta


# ── the joint model ──────────────────────────────────────────────────────────


def prep(d):
    """Training labels: the chain labels over every level (chainscore18's rule) and each node's gold flag (in the
    pool, not a seed); the rows with either."""
    on, has_on = CS.labels(d, np.ones(int(d["rowoff"][-1]), bool))
    xo = d["xn_off"]
    N = d["n"].size
    gf = np.zeros(int(xo[-1]), bool)
    gf[xo[np.repeat(np.arange(N), np.diff(d["gold_off"]))] + d["gold"].astype(np.int64)] = True
    gf &= ~d["seed0"]
    has_g = np.bincount(np.repeat(np.arange(N), d["n"].astype(np.int64)), weights=gf, minlength=N) > 0
    d["on"], d["gflag"] = on, gf
    d["train_rows"] = np.flatnonzero(has_g | has_on)
    d["n_regimes"] = int(d["FR"].shape[0])
    return d


def pair_raw(d, ri, p0, p1):
    X = np.concatenate([d["FI"][p0:p1], d["FR"][ri, p0:p1], d["FS"][p0:p1, None]], 1)
    X[:, CS.LRC] = np.maximum(X[:, CS.LRC], CS.LR_FLOOR)
    return X


def moments(blocks, width):
    s1, s2, n = np.zeros(width), np.zeros(width), 0
    for B in blocks:
        B = B.astype(np.float64)
        s1 += B.sum(0)
        s2 += (B * B).sum(0)
        n += B.shape[0]
    m = s1 / max(n, 1)
    s = np.sqrt(np.maximum(s2 / max(n, 1) - m * m, 0.0))
    s[s < 1e-6] = 1.0
    return m.astype(np.float32), s.astype(np.float32)


def input_stats(d):
    """Node features over every node of the id fit build; pair features over its pairs in every regime."""
    X = d["XN"]
    mn, sn = moments((X[a:a + (1 << 20)] for a in range(0, X.shape[0], 1 << 20)), len(NODEF))
    Pp = int(d["rowoff"][-1])
    mp, spp = moments((pair_raw(d, ri, a, min(Pp, a + (1 << 18))) for ri in range(d["FR"].shape[0])
                       for a in range(0, Pp, 1 << 18)), len(PAIRF))
    return {"mn": mn, "sn": sn, "mp": mp, "sp": spp}


def make_batch(items, builds, st, pairs=True, train=True):
    """Rows (build, row, regime) as flat node and pair inputs, padded (rows x nodes) and (rows x pairs) masks, and
    the chain-node incidence as flat indices into those."""
    import torch
    items = np.asarray(items, np.int64).reshape(-1, 3)
    B = len(items)
    ns = np.asarray([int(builds[bi]["n"][i]) for bi, i, _ in items])
    ks = np.asarray([int(builds[bi]["rowoff"][i + 1] - builds[bi]["rowoff"][i]) for bi, i, _ in items])
    Ln, Lk = int(ns.max()), max(int(ks.max()) if pairs else 1, 1)
    nm = np.zeros((B, Ln), bool)
    gm = np.zeros((B, Ln), bool)
    pm = np.zeros((B, Lk), bool)
    on = np.zeros((B, Lk), bool)
    lc = np.zeros((B, Lk), np.float32)
    xn, npos, xp, ppos, ip, inn, im = [], [], [], [], [], [], []
    for j, (bi, i, ri) in enumerate(items):
        d = builds[bi]
        a, n = int(d["xn_off"][i]), int(ns[j])
        xn.append(d["XN"][a:a + n])
        npos.append(j * Ln + np.arange(n))
        nm[j, :n] = ~d["seed0"][a:a + n]
        if train:
            gm[j, :n] = d["gflag"][a:a + n]
        k = int(ks[j])
        if not (pairs and k):
            continue
        p0 = int(d["rowoff"][i])
        xp.append(pair_raw(d, ri, p0, p0 + k))
        ppos.append(j * Lk + np.arange(k))
        pm[j, :k] = True
        lc[j, :k] = d["FI"][p0:p0 + k, CS.LOGC]
        if train:
            on[j, :k] = d["on"][p0:p0 + k]
        no = d["node_off"][p0:p0 + k + 1]
        ip.append(j * Lk + np.repeat(np.arange(k), np.diff(no)))
        inn.append(j * Ln + d["node"][no[0]:no[-1]].astype(np.int64))
        im.append(d["mass"][no[0]:no[-1]])
    t = torch.from_numpy
    bt = {"B": B, "Ln": Ln, "Lk": Lk, "nm": t(nm), "gm": t(gm), "pm": t(pm), "on": t(on), "lc": t(lc),
          "Xn": t(((np.concatenate(xn).astype(np.float32) - st["mn"]) / st["sn"]).astype(np.float32)),
          "npos": t(np.concatenate(npos))}
    if xp:
        bt["Xp"] = t(((np.concatenate(xp) - st["mp"]) / st["sp"]).astype(np.float32))
        bt["ppos"] = t(np.concatenate(ppos))
        bt["ip"], bt["in"], bt["im"] = t(np.concatenate(ip)), t(np.concatenate(inn)), t(np.concatenate(im))
    else:
        bt["Xp"] = None
    return bt


def make_joint(spec, seed):
    import torch

    def mlp(k):
        return torch.nn.Sequential(torch.nn.Linear(k, HID), torch.nn.ReLU(), torch.nn.Linear(HID, HID),
                                   torch.nn.ReLU(), torch.nn.Linear(HID, 1))

    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = mlp(len(NODEF)) if spec["f"] else None
            self.g = mlp(len(PAIRF)) if spec["g"] else None
            inv1 = math.log(math.e - 1.0)
            self.beta_r = torch.nn.Parameter(torch.tensor(inv1))
            self.gamma_r = torch.nn.Parameter(torch.tensor(inv1))
            self.leps = torch.nn.Parameter(torch.tensor(math.log(EPS0)))
            self.T = spec["T"]

    torch.manual_seed(seed)
    return Joint()


def masked_logsoftmax(z, m):
    import torch
    return torch.log_softmax(z.masked_fill(~m, NEG), 1).masked_fill(~m, NEG)


def forward(model, bt):
    """s_T over each row's nodes (rows x nodes) and log p_T over its chains (rows x pairs; None without g)."""
    import torch
    F_ = torch.nn.functional
    B, Ln, Lk = bt["B"], bt["Ln"], bt["Lk"]
    f = torch.zeros(B * Ln)
    if model.f is not None:
        f = f.index_copy(0, bt["npos"], model.f(bt["Xn"]).squeeze(-1))
    f = f.reshape(B, Ln)
    if model.g is None:
        return f, None
    z = torch.full((B * Lk,), NEG)
    if bt["Xp"] is not None:
        z = z.index_copy(0, bt["ppos"], model.g(bt["Xp"]).squeeze(-1))
    lp0 = masked_logsoftmax(z.reshape(B, Lk) - bt["lc"], bt["pm"])
    if bt["Xp"] is None:
        return f + F_.softplus(model.gamma_r) * model.leps, lp0
    gamma, eps = F_.softplus(model.gamma_r), torch.exp(model.leps)
    ip, inn, im = bt["ip"], bt["in"], bt["im"]

    def mstep(lp):
        agg = torch.zeros(B * Ln).index_add(0, inn, torch.exp(lp).reshape(-1)[ip] * im)
        return f + gamma * torch.log(eps + agg.reshape(B, Ln))

    if model.T == 0:
        return mstep(lp0), lp0
    beta = F_.softplus(model.beta_r)
    s, lp = f, lp0
    for _ in range(model.T):
        pi = torch.softmax(s.masked_fill(~bt["nm"], NEG), 1)
        ev = torch.zeros(B * Lk).index_add(0, ip, pi.reshape(-1)[inn] * im).reshape(B, Lk)
        lp = masked_logsoftmax(lp0 + beta * torch.log(ev + 1e-8), bt["pm"])
        s = mstep(lp)
    return s, lp


def objective(s, lp, bt):
    import torch
    zero = s.sum() * 0.0
    lpi = masked_logsoftmax(s, bt["nm"])
    hg = bt["gm"].any(1)
    ln = (-torch.logsumexp(lpi.masked_fill(~bt["gm"], NEG), 1))[hg].mean() if bool(hg.any()) else zero
    lc = zero
    if lp is not None:
        ho = bt["on"].any(1)
        if bool(ho.any()):
            lc = (-torch.logsumexp(lp.masked_fill(~bt["on"], NEG), 1))[ho].mean()
    return ln + lc, float(ln.detach()), float(lc.detach())


def coupling(model):
    import torch
    F_ = torch.nn.functional
    with torch.no_grad():
        return {"beta": round(float(F_.softplus(model.beta_r)), 5), "gamma": round(float(F_.softplus(model.gamma_r)), 5),
                "eps": float(torch.exp(model.leps)), "T": model.T}


def rank_row(d, i, s):
    a, n = int(d["xn_off"][i]), int(d["n"][i])
    ns = np.flatnonzero(~d["seed0"][a:a + n])
    rank = d["rank"][d["rank_off"][i]:d["rank_off"][i + 1]].astype(np.int64)
    top = ns[np.lexsort((rank[ns], -s[ns]))[:5]]
    if top.size < 5:
        srr = d["seeds_rr"][d["seeds_off"][i]:d["seeds_off"][i + 1]].astype(np.int64)
        top = np.r_[top, srr[:5 - top.size]]
    gold = np.zeros(n, bool)
    gold[d["gold"][d["gold_off"][i]:d["gold_off"][i + 1]]] = True
    return CP.metrics(top, gold, int(d["gt"][i]))


def read_joint(model, d, st, ri=0, batch=BATCH):
    """Per row (R@5, FC@5, hit@1) of s_T's ranking, and p_T's mean mass on each level."""
    import torch
    N = d["n"].size
    out = np.zeros((N, 3))
    L = d["level_names"].size
    ml = np.zeros(L)
    plev = d["p_lev"].astype(np.int64)
    with torch.no_grad():
        for b0 in range(0, N, batch):
            rows = np.arange(b0, min(N, b0 + batch))
            items = np.c_[np.zeros(rows.size, np.int64), rows, np.full(rows.size, ri)]
            bt = make_batch(items, [d], st, pairs=model.g is not None, train=False)
            s, lp = forward(model, bt)
            s = s.double().numpy()
            Pm = None if lp is None else np.exp(lp.double().numpy())
            for j, i in enumerate(rows):
                out[i] = rank_row(d, i, s[j, :int(d["n"][i])])
                if Pm is not None:
                    p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
                    if p1 > p0:
                        ml += np.bincount(plev[p0:p1], weights=Pm[j, :p1 - p0], minlength=L)
    return out, ml / max(N, 1)


def draw(rng, builds, use, count):
    """count examples (build, row, regime): the build uniform among use, the row uniform among its training rows,
    the regime uniform among its regimes."""
    t = rng.integers(len(use), size=count)
    out = []
    for k, bi in enumerate(use):
        m = int((t == k).sum())
        if not m:
            continue
        tr = builds[bi]["train_rows"]
        out.append(np.c_[np.full(m, bi), tr[rng.integers(tr.size, size=m)],
                         rng.integers(builds[bi]["n_regimes"], size=m)])
    items = np.concatenate(out)
    return items[rng.permutation(len(items))]


def train_joint(arm, builds, Sd, epochs, per_epoch, batch, seed, threads=1):
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    spec = arm_spec(arm)
    use = list(range(len(builds))) if spec["gr"] else [0]
    st = input_stats(builds[0])
    model = make_joint(spec, seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)
    hist, best, best_state = [], None, None
    for ep in range(epochs):
        t1 = time.time()
        model.train()
        items = draw(rng, builds, use, per_epoch)
        tot, tn, tc, nb = 0.0, 0.0, 0.0, 0
        for b0 in range(0, len(items), batch):
            bt = make_batch(items[b0:b0 + batch], builds, st, pairs=spec["g"])
            s, lp = forward(model, bt)
            loss, ln, lc = objective(s, lp, bt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            tn += ln
            tc += lc
            nb += 1
        model.eval()
        x, _ = read_joint(model, Sd, st)
        sr5 = float(x[:, 0].mean())
        hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 5), "node_loss": round(tn / max(nb, 1), 5),
                     "chain_loss": round(tc / max(nb, 1), 5), "select_r5": round(sr5, 5),
                     "by_build": np.bincount(items[:, 0], minlength=len(builds)).tolist(), **coupling(model),
                     "seconds": round(time.time() - t1, 1)})
        log(f"    {arm} ep {ep}: loss {tot / max(nb, 1):.4f} (node {tn / max(nb, 1):.4f}, chain "
            f"{tc / max(nb, 1):.4f}) select R@5 {sr5:.4f} {coupling(model)} {time.time() - t1:.0f}s")
        if best is None or sr5 > best[1]:
            best = (ep, sr5)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_epoch": best[0], "best_select_r5": round(best[1], 5), "stats": st,
                   "coupling": coupling(model), "spec": spec}


# ── train (one arm) and grade ────────────────────────────────────────────────


def summarise(x, D, ml, names):
    return {"mean": [round(float(v), 5) for v in x.mean(0)], "rows": int(x.shape[0]),
            "mass_by_level": None if ml is None else {str(n_): round(float(v), 4) for n_, v in zip(names, ml)},
            "by_D": {str(int(dd)): [round(float(x[D == dd, 0].mean()), 5), int((D == dd).sum())] for dd in np.unique(D)}}


def train_cmd(a):
    t0 = time.time()
    res = {"look": "chainscore19", "arm": a.arm, "args": dict(vars(a)), "script_sha256": sha(__file__),
           "chainscore18_sha256": sha(CS.__file__), "chainpop17_sha256": sha(CP.__file__), "inputs": {},
           "inputs_sha256": {}, "reads": {}, "rows_out": a.rows_out}
    for p_ in [a.fit, a.select] + [x.split("=", 1)[1] for x in a.aug + a.read]:
        res["inputs_sha256"][p_] = sha(p_)
    fit = prep(CS.load(a.fit))
    Sd = CS.load(a.select)
    res["inputs"] = {"fit": fit["meta"], "select": Sd["meta"]}
    if a.arm == "g18":
        model, info = CS.train_arm("full", fit, Sd, a.epochs, CS.SEED)
        res.update({"history": info["history"], "best_epoch": info["best_epoch"],
                    "best_select_r5": info["best_select_r5"], "kinds": info["kinds_final"]})
    else:
        builds = [fit]
        if arm_spec(a.arm)["gr"]:
            if sorted(x.split("=", 1)[0] for x in a.aug) != sorted(TRANSFORMS[1:]):
                raise SystemExit(f"a -gr arm takes --aug for each of {TRANSFORMS[1:]}")
            for x in a.aug:
                nm_, p_ = x.split("=", 1)
                d = prep(CS.load(p_))
                if d["meta"]["transform"]["name"] != nm_:
                    raise SystemExit(f"{p_} holds transform {d['meta']['transform']['name']}, not {nm_}")
                res["inputs"][nm_] = d["meta"]
                builds.append(d)
        res["train_rows"] = {("id" if k == 0 else builds[k]["meta"]["transform"]["name"]): int(b["train_rows"].size)
                             for k, b in enumerate(builds)}
        model, info = train_joint(a.arm, builds, Sd, a.epochs, a.per_epoch, a.batch, SEED)
        del builds
        res.update({"history": info["history"], "best_epoch": info["best_epoch"],
                    "best_select_r5": info["best_select_r5"], "coupling": info["coupling"], "spec": info["spec"]})
    fit_meta = fit["meta"]
    del fit, Sd
    rows_out = {}
    for x in a.read:
        nm_, p_ = x.split("=", 1)
        t1 = time.time()
        d = CS.load(p_)
        if a.arm == "g18":
            X = (CS.features(d, 0, info["col_idx"]) - info["mean"]) / info["std"]
            allow = CS.pair_mask(d, info["kinds_final"])
            p = CS.softmax_rows(d, CS.pair_logits(model, X, d), allow)
            xr = CS.score_rows(d, p, allow)
            ml = np.bincount(d["p_lev"].astype(np.int64), weights=p, minlength=d["level_names"].size) / d["n"].size
        else:
            xr, ml = read_joint(model, d, info["stats"])
        rows_out[nm_] = xr.astype(np.float32)
        rows_out[f"{nm_}__D"] = d["D"]
        res["reads"][nm_] = summarise(xr, d["D"], ml, d["level_names"])
        res["inputs"][nm_] = d["meta"]
        log(f"  {a.arm} on {nm_}: {res['reads'][nm_]['mean']} mass {res['reads'][nm_]['mass_by_level']} "
            f"({time.time() - t1:.0f}s)")
        del d
    p = Path(a.rows_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows_out)
    os.replace(tmp, p)
    if a.state_out and a.arm != "g18":
        import torch
        sp_ = Path(a.state_out)
        torch.save({"state": model.state_dict(), "stats": info["stats"], "spec": info["spec"],
                    "fit_meta_sha": fit_meta.get("script_sha256")}, sp_.with_name(sp_.name + ".tmp"))
        os.replace(sp_.with_name(sp_.name + ".tmp"), sp_)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def ci_label(ci, labels_):
    return {"above": labels_[0], "below": labels_[1], "spans": labels_[2]}[CP.side(ci)]


def grade_cmd(a):
    t0 = time.time()
    arms, rows = {}, {}
    for x in a.arm:
        nm_, p_ = x.split("=", 1)
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        if js["arm"] != nm_:
            raise SystemExit(f"{p_} is arm {js['arm']}, not {nm_}")
        with np.load(js["rows_out"]) as z:
            rows[nm_] = {k: z[k] for k in z.files}
        arms[nm_] = js
    refs, Ds = {}, {}
    for x in a.read:
        nm_, p_ = x.split("=", 1)
        d = CS.load(p_)
        refs[nm_] = CS.reference_arms(d)
        Ds[nm_] = d["D"]
        del d
    res = {"look": "chainscore19", "args": dict(vars(a)), "script_sha256": sha(__file__),
           "arms": {k: {"best_epoch": v["best_epoch"], "best_select_r5": v["best_select_r5"],
                        "coupling": v.get("coupling"), "reads": v["reads"]} for k, v in arms.items()},
           "refs": {nm_: {k: [round(float(u), 5) for u in x.mean(0)] for k, x in r.items()} for nm_, r in refs.items()},
           "verdicts": {}}
    hs = ("HELPS", "HURTS", "SAME")
    abv = ("ABOVE", "BELOW", "AT")
    V = {"J2": {}, "J3": {}, "J4": {}}

    def r5(arm, nm_):
        x = rows.get(arm, {}).get(nm_)
        return None if x is None else x[:, 0].astype(np.float64)

    for nm_ in refs:
        N = refs[nm_]["rrf_s"].shape[0]
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        for tag, prs in (("J2", (("em", "fg"),)), ("J3", (("em", "g"), ("em", "f"), ("em-gr", "g-gr"), ("em-gr", "f-gr"))),
                         ("J4", (("em-gr", "em"), ("g-gr", "g"), ("f-gr", "f")))):
            for x_, y_ in prs:
                X_, Y_ = r5(x_, nm_), r5(y_, nm_)
                if X_ is None or Y_ is None:
                    continue
                ci = CP.boot_mean(X_ - Y_, idx)
                V[tag].setdefault(nm_, {})[f"{x_} - {y_}"] = {"diff": ci, "verdict": ci_label(ci, hs)}
        tw, fl = refs[nm_]["twin0"][:, 0], refs[nm_]["rrf_s"][:, 0]
        if nm_ == "webqsp_sf":
            j1, j5 = {}, {}
            for arm in ("em-gr", "em"):
                X_ = r5(arm, nm_)
                if X_ is None:
                    continue
                ci = CP.boot_mean(X_ - refs[nm_]["none/rd"][:, 0], idx)
                j1[arm] = {"diff": ci, "verdict": ci_label(ci, abv)}
            X_ = r5("em-gr", nm_)
            if X_ is not None:
                for dd in (1, 2):
                    m = Ds[nm_] == dd
                    if not m.any():
                        continue
                    idd = np.random.default_rng(SEED).integers(0, int(m.sum()), size=(BOOT, int(m.sum())))
                    ci = CP.boot_mean((X_ - refs[nm_]["none/rd"][:, 0])[m], idd)
                    j5[f"D{dd}"] = {"diff": ci, "verdict": ci_label(ci, abv), "rows": int(m.sum())}
            V["J1"], V["J5"] = j1, j5
        if nm_ == "metaqa":
            m1 = {}
            for arm in ("em", "em-gr"):
                X_ = r5(arm, nm_)
                if X_ is None:
                    continue
                e = {}
                if r5("g18", nm_) is not None:
                    ci = CP.boot_mean(X_ - r5("g18", nm_), idx)
                    e["vs_g18"] = {"diff": ci, "verdict": ci_label(ci, abv)}
                ci = CP.boot_mean(X_ - tw, idx)
                e["vs_twin"] = {"diff": ci, "verdict": ci_label(ci, abv)}
                sh = CP.share(X_, fl, tw, idx)
                e["share"], e["share_verdict"] = sh, "HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID"
                m1[arm] = e
            V["M1"] = m1
        if nm_ == "webqsp":
            z1 = {}
            for arm in ("em-gr", "em"):
                X_ = r5(arm, nm_)
                if X_ is None:
                    continue
                m = CP.boot_mean(X_, idx)
                sh = CP.share(X_, fl, tw, idx)
                z1[arm] = {"mean": m, "band": list(J3A),
                           "verdict": "ABOVE" if m[1] > J3A[1] else "BELOW" if m[2] < J3A[0] else "WITHIN",
                           "share": sh, "share_verdict": "HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID",
                           "minus_none_rd": CP.boot_mean(X_ - refs[nm_]["none/rd"][:, 0], idx)}
            V["Z1"] = z1
    if a.cs18_rows and "g18" in rows:
        r0 = {}
        with np.load(a.cs18_rows) as z:
            for nm_ in ("metaqa", "webqsp"):
                k = f"{nm_}__arm__full"
                if k in z.files and nm_ in rows["g18"]:
                    dif = float(np.abs(rows["g18"][nm_].astype(np.float64) - z[k].astype(np.float64)).max())
                    r0[nm_] = {"max_abs_diff": dif, "verdict": "REPRODUCES" if dif < 1e-6 else "DIFFERS"}
        V["R0"] = r0
    res["verdicts"] = V
    res["seconds"] = round(time.time() - t0, 1)
    log(f"verdicts {json.dumps(V)}")
    return res


# ── selftest ─────────────────────────────────────────────────────────────────


def forward_np(model, d, i, st, ri=0):
    """A float64 numpy reference of forward() on one row, from the model's weights."""
    def mlp_np(seq, X):
        W = [(m.weight.detach().double().numpy(), m.bias.detach().double().numpy()) for m in seq
             if hasattr(m, "weight")]
        h = X
        for k, (w, b) in enumerate(W):
            h = h @ w.T + b
            if k < len(W) - 1:
                h = np.maximum(h, 0.0)
        return h[:, 0]

    a, n = int(d["xn_off"][i]), int(d["n"][i])
    Xn = (d["XN"][a:a + n].astype(np.float32) - st["mn"]) / st["sn"]
    f = mlp_np(model.f, Xn.astype(np.float64)) if model.f is not None else np.zeros(n)
    if model.g is None:
        return f, None
    p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
    Xp = ((pair_raw(d, ri, p0, p1) - st["mp"]) / st["sp"]).astype(np.float32).astype(np.float64)
    z = mlp_np(model.g, Xp) - d["FI"][p0:p1, CS.LOGC].astype(np.float64)
    lp0 = z - np.logaddexp.reduce(z)
    no = d["node_off"][p0:p1 + 1]
    M = sp.csr_matrix((d["mass"][no[0]:no[-1]].astype(np.float64),
                       (np.repeat(np.arange(p1 - p0), np.diff(no)), d["node"][no[0]:no[-1]])), shape=(p1 - p0, n))
    sp1 = lambda x: math.log1p(math.exp(float(x)))  # noqa: E731
    gamma, eps, beta = sp1(model.gamma_r), math.exp(float(model.leps)), sp1(model.beta_r)
    mstep = lambda lp: f + gamma * np.log(eps + M.T @ np.exp(lp))  # noqa: E731
    if model.T == 0:
        return mstep(lp0), lp0
    ns = ~d["seed0"][a:a + n]
    s, lp = f, lp0
    for _ in range(model.T):
        e = np.where(ns, np.exp(s - s[ns].max()), 0.0)
        pi = e / e.sum()
        lp = lp0 + beta * np.log(M @ pi + 1e-8)
        lp = lp - np.logaddexp.reduce(lp)
        s = mstep(lp)
    return s, lp


def selftest():
    import torch
    P = RT.proj_matrix()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        roots = {}
        for nm_, sd, rows in (("train", 11, 96), ("select", 12, 48), ("read", 13, 64)):
            root = td / nm_
            reld = td / f"{nm_}_rel"
            reld.mkdir(parents=True)
            _truth, names = CP.toy_world(root, np.random.default_rng(sd), P, rows=rows)
            np.save(reld / "toy_rel_embeddings.npy", names)
            roots[nm_] = (root, reld)

        def args(**kw):
            base = dict(ds="toy", carve="read", hops=2, ks="2", limit=None, max_chains=CP.MAX_CHAINS, regimes="1",
                        transform="id", rows="all", no_nodes=False)
            base.update(kw)
            return SimpleNamespace(**base)

        # 1. the id build is chainscore18's, array by array
        built = {}
        for nm_, regs in (("train", "1,0.25,0"), ("select", "1"), ("read", "1")):
            root, reld = roots[nm_]
            CS.build(args(regimes=regs, out=str(td / f"{nm_}18.npz")), root=root, rel_dir=reld)
            build19(args(regimes=regs, out=str(td / f"{nm_}.npz")), root=root, rel_dir=reld)
            d18, d19 = CS.load(td / f"{nm_}18.npz"), CS.load(td / f"{nm_}.npz")
            for k, x in d18.items():
                if k == "meta":
                    continue
                y = d19[k]
                if x.dtype.kind in "iu":
                    assert np.array_equal(x.astype(np.int64), y.astype(np.int64)), (nm_, k)
                elif x.dtype.kind == "f":
                    assert x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True), (nm_, k)
                else:
                    assert np.array_equal(x, y), (nm_, k)
            assert np.isfinite(d19["XN"].astype(np.float32)).all() and np.isfinite(d19["FS"]).all()
            built[nm_] = d19
        print("toy: the id builds equal chainscore18's on every array")
        # 2. a relabel keeps the structure, the none level and the node features
        root, reld = roots["train"]
        did = built["train"]
        for tfn, n_rel in (("al4", 16), ("sp4", 16), ("mg3", 2)):
            meta = build19(args(regimes="1,0", transform=tfn, out=str(td / f"train-{tfn}.npz")), root=root,
                           rel_dir=reld)
            dt = CS.load(td / f"train-{tfn}.npz")
            assert meta["transform"]["n_rel"] == n_rel and meta["levels"][-1]["Kz"] == 2 * n_rel, (tfn, meta["levels"])
            for k in ("XN", "seed0", "n", "gold", "gold_off", "D", "rank", "gt", "rrf_s", "xn_off"):
                assert np.array_equal(dt[k], did[k]), (tfn, k)
            for i in range(dt["n"].size):
                lv = []
                for d_ in (did, dt):
                    ro = d_["rowoff"]
                    ix = np.flatnonzero(d_["p_lev"][ro[i]:ro[i + 1]] == 0) + ro[i]
                    no = d_["node_off"]
                    lv.append([(d_["node"][no[j]:no[j + 1]].tolist(), d_["mass"][no[j]:no[j + 1]].tolist()) for j in ix])
                assert lv[0] == lv[1], (tfn, i)
            print(f"toy {tfn}: {meta['relations_present']} of {n_rel} relations present, levels "
                  f"{[(x['name'], x['vocab']) for x in meta['levels']]}; structure, none level and node features as id")
        # 3. rf: mediators added after the pool, two neighbours each, never gold; distances only grow
        meta = build19(args(regimes="1,0", transform="rf", out=str(td / "train-rf.npz")), root=root, rel_dir=reld)
        dr = CS.load(td / "train-rf.npz")
        nid, nrf = did["n"].astype(np.int64), dr["n"].astype(np.int64)
        assert (nrf >= nid).all() and meta["mediators"] == int((nrf - nid).sum()) > 0
        assert (dr["D"] >= did["D"]).all() and (dr["D"] > did["D"]).any()
        for i in range(nrf.size):
            assert (dr["gold"][dr["gold_off"][i]:dr["gold_off"][i + 1]] < nid[i]).all()
            a = int(dr["xn_off"][i])
            Xm = dr["XN"][a + nid[i]:a + nrf[i]].astype(np.float32)
            assert np.allclose(Xm[:, NODEF.index("deg")], math.log(3), atol=2e-3), i
        print(f"toy rf: {meta['mediators']} mediators over {nrf.size} rows (reified {meta['transform']['reified']}); "
              "never gold, two neighbours each, seed-gold distances only grow")
        # 4. two carves read as one population
        root, reld = roots["read"]
        shutil.copytree(root / "toy" / "read", root / "toy" / "read2")
        build19(args(carve="read+read2", out=str(td / "read2.npz")), root=root, rel_dir=reld)
        d2, d1 = CS.load(td / "read2.npz"), built["read"]
        N = d1["n"].size
        h, hx = int(d2["rowoff"][N]), int(d2["xn_off"][N])
        assert d2["n"].size == 2 * N and (d2["carve_ix"][:N] == 0).all() and (d2["carve_ix"][N:] == 1).all()
        assert np.array_equal(d2["row_ix"][N:], np.arange(N)) and np.array_equal(d2["FI"][:h], d1["FI"])
        assert np.array_equal(d2["FI"][h:], d1["FI"]) and np.array_equal(d2["XN"][hx:], d1["XN"])
        print("toy: two carves read as one population; per-row arrays repeat, the population features change")
        # 5. training: deterministic, the loss falls, the forward matches a float64 numpy reference
        tr = prep(CS.load(td / "train.npz"))
        aug = [prep(CS.load(td / f"train-{t}.npz")) for t in TRANSFORMS[1:]]
        runs = {}
        for arm in ("g", "f", "fg", "em", "em-gr"):
            builds = [tr] + (aug if arm_spec(arm)["gr"] else [])
            model, info = train_joint(arm, builds, built["select"], 4, 192, 16, SEED)
            x, ml = read_joint(model, built["read"], info["stats"])
            runs[arm] = (model, info, x)
            h_ = info["history"]
            print(f"toy {arm}: loss {h_[0]['loss']} -> {h_[-1]['loss']}, best epoch {info['best_epoch']}, read "
                  f"{np.round(x.mean(0), 4).tolist()}, coupling {info['coupling']}")
            assert h_[-1]["loss"] < h_[0]["loss"], (arm, h_)
        model2, info2 = train_joint("em-gr", [tr] + aug, built["select"], 4, 192, 16, SEED)
        x2, _ = read_joint(model2, built["read"], info2["stats"])
        assert info2["history"] == runs["em-gr"][1]["history"] and np.array_equal(x2, runs["em-gr"][2])
        worst = 0.0
        for arm in ("g", "f", "fg", "em"):
            model, info, _ = runs[arm]
            rows = np.arange(min(12, built["read"]["n"].size))
            items = np.c_[np.zeros(rows.size, np.int64), rows, np.zeros(rows.size, np.int64)]
            with torch.no_grad():
                bt = make_batch(items, [built["read"]], info["stats"], pairs=model.g is not None, train=False)
                s, lp = forward(model, bt)
            for j, i in enumerate(rows):
                n = int(built["read"]["n"][i])
                k = int(built["read"]["rowoff"][i + 1] - built["read"]["rowoff"][i])
                s_np, lp_np = forward_np(model, built["read"], i, info["stats"])
                ds_ = float(np.abs(s[j, :n].double().numpy() - s_np).max() / max(1.0, np.abs(s_np).max()))
                dp_ = 0.0 if lp_np is None else float(np.abs(np.exp(lp[j, :k].double().numpy()) - np.exp(lp_np)).max())
                worst = max(worst, ds_, dp_)
        assert worst < 1e-4, worst
        print(f"toy: training deterministic, the loss falls in every arm; forward vs the numpy reference {worst:.2e}")
        # 6. g18: chainscore18's full arm reads the same on this file's builds as on chainscore18's
        outs = []
        for suf in ("18", ""):
            T_, S_, R_ = (CS.load(td / f"{x}{suf}.npz") for x in ("train", "select", "read"))
            model, info = CS.train_arm("full", T_, S_, 3, CS.SEED)
            X = (CS.features(R_, 0, info["col_idx"]) - info["mean"]) / info["std"]
            allow = CS.pair_mask(R_, info["kinds_final"])
            outs.append(CS.score_rows(R_, CS.softmax_rows(R_, CS.pair_logits(model, X, R_), allow), allow))
        assert np.array_equal(outs[0], outs[1])
        print(f"toy g18: {np.round(outs[0].mean(0), 4).tolist()} on both builds")
    print("selftest: the id builds equal chainscore18's; relabels keep structure, the none level and node features; "
          "rf mediators are never gold and lengthen paths; two carves form one population; training is "
          "deterministic and its loss falls; the batched forward matches a float64 reference; g18 reproduces. "
          "all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--ds", required=True)
    b.add_argument("--carve", required=True, help="a carve, or carves joined by + (read as one population)")
    b.add_argument("--hops", type=int, default=3)
    b.add_argument("--ks", default="2,4")
    b.add_argument("--regimes", default="1")
    b.add_argument("--transform", default="id", choices=TRANSFORMS)
    b.add_argument("--rows", default="all", choices=("all", "even", "odd"))
    b.add_argument("--max-chains", type=int, default=CP.MAX_CHAINS)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=ARMS)
    t.add_argument("--fit", required=True)
    t.add_argument("--aug", action="append", default=[])
    t.add_argument("--select", required=True)
    t.add_argument("--read", action="append", default=[])
    t.add_argument("--epochs", type=int, default=EPOCHS)
    t.add_argument("--per-epoch", type=int, default=PER_EPOCH)
    t.add_argument("--batch", type=int, default=BATCH)
    t.add_argument("--out", required=True)
    t.add_argument("--rows-out", required=True)
    t.add_argument("--state-out")
    g = sub.add_parser("grade")
    g.add_argument("--arm", action="append", required=True)
    g.add_argument("--read", action="append", required=True)
    g.add_argument("--cs18-rows")
    g.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "build":
        if not 1 <= a.hops <= RT.MAXHOP:
            raise SystemExit(f"--hops from 1 to {RT.MAXHOP}")
        build19(a)
        return 0
    if a.cmd in ("train", "grade"):
        res = train_cmd(a) if a.cmd == "train" else grade_cmd(a)
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                       encoding="utf-8")
        os.replace(tmp, p)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
