"""Design look (untracked; not a result and not filed): S6 part 1 of the transfer plan, "transfer the estimator, not
the weights". Can a KB's relation chains be estimated, coarse to fine, from the graph's own unlabeled queries, so that
a walk along the estimated chains retrieves answers on a graph no model was trained on? Model-free: nothing is
trained, gold enters nothing before grading, and no relation name is read except by the arms marked nm.

Types, coarse to fine (one nested hierarchy per graph; index-time, no names, no gold):
  none    every structural edge is one type, read both ways (an untyped walk)
  dir     z = d: 0 when the walk steps from a triple's head to its tail, 1 from its tail to its head
  tt<k>   z = 2 c + d, c the relation's cluster among k: Ward clusters of the relations' endpoint text,
          TT_r = [n(sum of n(p_t) over r's tails), n(sum of n(p_h) over its heads)] over the distinct triples of the
          read rows' pools (reltype11's collector), cut from one dendrogram, so the levels nest
  exact   z = 2 r + d, r the KB's relation id (index-time structure; its name is not read)
A chain is a sequence of 1 to --hops types. A row's candidate chains C_i are the chains whose walk from its rank-1
seeds (bucket 0) reaches a node outside them. The walk puts mass 1/|S0| on each seed; a step of type z sends each
node's mass equally over its type-z out-edges in the pool (a node with none drops its mass). The chain's end
distribution m_c is the mass that arrives, renormalised over the non-seed nodes.

Population fit (per graph and level, on the read rows themselves: transductive, no gold). EM for a mixture over each
row's own candidates,
    p(c | i) = pi_c exp(kappa cos(q_i, mu_c)) / sum over c' in C_i of the same,   kappa 20,
maximising the conditional likelihood sum_i log [sum_{c in C_i} pi_c f_c(q_i) / sum_{c in C_i} pi_c], plus a
symmetric Dirichlet prior on pi (alpha 1 pseudo-count per chain) and a von Mises-Fisher prior on mu_c around an
anchor a_c with weight tau:
    E   r_ic = p(c | i)
    M   mu_c = n(sum_i r_ic q_i + tau a_c)
        pi_c = (sum_i r_ic + alpha) / (sum over rows i with c in C_i of 1 / S_i + alpha K), then normalised;
        S_i = sum over c in C_i of the old pi, K the level's chain count
This is an MM step, so the objective never falls (--selftest checks it). q_i = n(n(qemb_i) P), P the look's
128-wide projection; p_v a node's stored projection. Fits (anchor, weight, start):
    pop    no anchor, started at rd_c       rd_c = n(sum over rows of sum_v m_c(v) n(p_v)), the chain's pooled
    pop1   tau 1 toward rd_c                       end-node text (label-free, nothing fitted); tau 1 is altau16's
    rd     mu_c = rd_c held, pi fitted             cross-validated choice on both KBs
    hpop   tau 4 toward the parent chain's hpop mu (each type mapped one level up), started at rd_c: hierarchical
           shrinkage, coarse to fine (none has no parent: its hpop is its pop). Started at the parent's mu instead,
           sibling chains would start equal and only their availability could split them
    nm1    tau 1 toward nm_c, started there   nm_c = n(sum over the chain's types of n(sum of n(name_r P) over
    nm     mu_c = nm_c held, pi fitted              the type's relations)): the relation names' text (tt, exact)
Score: s_i(v) = sum over c in C_i of r_ic m_c(v). Bucket-0 seeds rank last; ties go to rrf, then pool position. A row
with no candidate chain is ranked as rrf-s (rrf, seeds last).

Label-free choice. Every arm (level, fit) is fitted on one half of the read rows and scored on the other (2 folds,
rng SEED) by the held-out log-likelihood of the queries,
    ll_i = log sum_{c in C_i} pi_c exp(kappa cos(q_i, mu_c)) - log sum_{c in C_i} pi_c
(the vMF constant is common at one kappa). The likelihood cannot see which chain a cluster of queries is attached to:
swapping the (mu, pi) of two chains that every row has as candidates leaves it unchanged, so a free fit can win on
likelihood with its clusters on the wrong chains. The selftest's first toy did exactly that (random templates: CV took
exact/pop, chain AUC 0.54, while nm, the name-anchored fit, read twice its R@5). What attaches a cluster to a chain is
the text a fit starts from or is anchored to, so the choice is made within each attachment source: cv-nn is the arm
with the highest mean ll among pop, pop1, rd and hpop (end text), cv the same among nm1 and nm (names). Ties go to the
coarser level, then to the fit listed first. cv-all, the highest over every arm, is reported without a verdict.

Grading (gold is read only here). Per row R@5, FC@5 and hit@1 over the row's gold total (lean_mlp.row_metrics' rule),
paired against rrf alone (floor10's floor), rrf-s, and twin0 (the six pair's trained twin, from the look's q_metrics),
with a row bootstrap (BOOT 1000). The share of the twin's lead is (arm - rrf-s) / (twin0 - rrf-s) in R@5, taken as a
ratio of bootstrap means. Chain identification: in each row, a chain is on when it alone reaches the row's best R@5
(above 0). The posterior's AUC and top-1 against on are read by seed-gold distance D (reltype11.grade_rule). The chain
ceiling is each row's best single chain (by R@5, then FC@5, then hit@1), chosen with gold.
Verdicts, fixed at 09:57 on 4 Oct before any number on a real graph. They replace the first set (written with a
forward-dated 09:58 stamp), which chose cv over every arm; the toy showed that rule throws away what the names attach.
Each is read for cv-nn and for cv:
    V1  against rrf-s, R@5: ABOVE_FLOOR if the interval lies above 0, BELOW_FLOOR if below it, else AT_FLOOR
    V2  against the passage-trained zero-shot MLP (l3-2w-zk's block-dropout fit, all blocks: metaqa 9.55, webqsp
        14.03 R@5), unpaired: ABOVE_ZS if the interval of the arm's mean lies above it, BELOW_ZS if below, else AT_ZS
    V3  share of the twin's lead: HIGH if its interval lies at or above 0.5, LOW if it lies below 0.25, else MID
    V4  against the best arm of its own attachment source in hindsight (by R@5, read with gold): MATCHES if within
        1 point, else MISSES
    V7  minus none/pop (the untyped walk, weighted only by length), R@5: HELPS / HURTS / SAME by the interval: do
        the estimated relation types beat no types?
    V6  (webqsp) against l7g-j3a's KB-trained zero-shot MLP (0.20 to 0.25 R@5): ABOVE if its interval lies above
        0.25, BELOW if it lies below 0.20, else WITHIN
and once:
    V5  (webqsp) cv-nn minus exact/pop (the finest level, no shrinkage), R@5: HELPS / HURTS / SAME by the interval
    V5b (webqsp) hpop minus pop at cv-nn's level (unless that is none): HELPS / HURTS / SAME
If V1 is ABOVE_FLOOR on metaqa for neither cv-nn nor cv, part 2 (multi-regime training on these walk scores) is not
run. Part 1 reads train-split rows (metaqa x1f, webqsp selectf): a look, not a result.

    python outputs/mp_unified/chainpop17.py --ds metaqa --read x1f --hops 3 --ks 2,4 \
        --out outputs/mp_unified/lean/cp17-mq.json
    python outputs/mp_unified/chainpop17.py --ds webqsp --read selectf --hops 2 --ks 4,16,64,256 \
        --out outputs/mp_unified/lean/cp17-wq.json
    python outputs/mp_unified/chainpop17.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reltype11 as RT  # noqa: E402

LOOK, REL = RT.LOOK, RT.REL
SEED = 20261004
KAPPA = 20.0
ALPHA = 1.0
ITERS = 30
TAU1 = 1.0
N0 = 4.0
BOOT = 1000
FOLDS = 2
MAX_CHAINS = 20000
ZS_MLP = {"metaqa": 0.0955, "webqsp": 0.1403}      # l3-2w-zk: the block-dropout fit, all blocks
J3A = (0.20, 0.25)                                 # l7g-j3a: webqsp zero-shot of the KB-trained lean MLP
FITS = ("pop", "pop1", "rd", "hpop", "nm1", "nm")
NAMED = ("nm1", "nm")
KEYS = ("q_pool_size", "q_edges", "q_emb", "q_seed_local", "q_seed_bucket", "q_gold_total", "q_metrics", "pool",
        "proj", "is_gold", "x", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "e_rel")
log = RT.log
unit = RT.unit


# ── the type hierarchy ───────────────────────────────────────────────────────


class Level:
    """One level: name, type count Kz, z of (relation, direction), and the parent type of each z."""

    def __init__(self, name, lab=None, parent=None):
        self.name, self.lab, self.parent = name, lab, parent
        if name == "none":
            self.Kz = 1
        elif name == "dir":
            self.Kz = 2
        else:
            self.Kz = 2 * (int(lab.max()) + 1)
        self.B = self.Kz + 1

    def z(self, r, d):
        if self.name == "none":
            return np.zeros(r.size, np.int64)
        if self.name == "dir":
            return np.full(r.size, d, np.int64)
        return 2 * self.lab[r] + d

    def up(self, z):
        """The parent level's type of each z."""
        p = self.parent
        if p is None:
            raise ValueError("none has no parent")
        if p.name == "none":
            return np.zeros_like(z)
        if p.name == "dir":
            return z % 2
        # each cluster's relations share one parent cluster (checked in hierarchy())
        return 2 * self.cpar[z // 2] + z % 2

    def key(self, types):
        k = 0
        for j, t in enumerate(types):
            k += (int(t) + 1) * self.B ** j
        return k


def hierarchy(tt, present, ks, n_rel):
    """none < dir < tt<k> (k ascending, below the number of present relations) < exact. tt: the relations' endpoint
    descriptors (n_rel x 2d), present: which relations occur in the read pools. Ward on the present relations; an
    absent relation takes the cluster of its nearest present one (it never occurs in a walk)."""
    none = Level("none")
    dr = Level("dir", parent=none)
    levels = [none, dr]
    idx = np.flatnonzero(present)
    Z = linkage(tt[idx].astype(np.float64), method="ward") if idx.size > 1 else None
    prev = dr
    nn_ = None
    if idx.size < n_rel:
        sims = tt @ tt[idx].T
        nn_ = idx[np.argmax(sims, axis=1)]
    for k in sorted(set(ks)):
        if k >= idx.size or Z is None:
            continue
        labp = fcluster(Z, t=k, criterion="maxclust") - 1
        lab = np.zeros(n_rel, np.int64)
        lab[idx] = labp
        if nn_ is not None:
            lab = lab[np.where(present, np.arange(n_rel), nn_)]
        lev = Level(f"tt{k}", lab=lab, parent=prev)
        levels.append(lev)
        prev = lev
    ex = Level("exact", lab=np.arange(n_rel, dtype=np.int64), parent=prev)
    levels.append(ex)
    for lev in levels[2:]:
        p = lev.parent
        K = lev.Kz // 2
        if p.name == "dir":
            lev.cpar = None
            continue
        cpar = np.full(K, -1, np.int64)
        for r in range(n_rel):
            c, pc = lev.lab[r], p.lab[r]
            if cpar[c] == -1:
                cpar[c] = pc
            elif cpar[c] != pc:
                raise SystemExit(f"{lev.name} does not nest in {p.name}")
        lev.cpar = cpar
    return levels


# ── walks ────────────────────────────────────────────────────────────────────


def row_triples(c, a, b, ea, eb):
    """The row's structural stored triples (pool-local head, tail, relation slots), from either message of a pair,
    as reltype11's collector reads them; and its structural message endpoints (for distances)."""
    s = c["e_fam"][ea:eb] == 0
    u = c["e_u"][ea:eb][s].astype(np.int64)
    v = c["e_v"][ea:eb][s].astype(np.int64)
    fw = c["e_fwd"][ea:eb][s] == 1
    bw = c["e_bwd"][ea:eb][s] == 1
    rel = c["e_rel"][ea:eb][s].astype(np.int64)
    return np.r_[u[fw], v[bw]], np.r_[v[fw], u[bw]], np.r_[rel[fw], rel[bw]], u, v


def typed_graph(n, hl, tl, sl, lev):
    """Typed out-edges (src, z, dst), one per distinct (src, z, dst), sorted by (src, z, dst); with each edge's
    (src, z) out-degree and the per-src offsets."""
    ok = sl >= 0
    k = ok.sum(1)
    h, t, r = np.repeat(hl, k), np.repeat(tl, k), sl[ok]
    src = np.r_[h, t]
    dst = np.r_[t, h]
    z = np.r_[lev.z(r, 0), lev.z(r, 1)]
    key = np.unique((src * lev.Kz + z) * n + dst)
    dst = key % n
    sz = key // n
    z = sz % lev.Kz
    src = sz // lev.Kz
    if src.size:
        first = np.r_[True, sz[1:] != sz[:-1]]
        starts = np.flatnonzero(first)
        cnt = np.diff(np.r_[starts, src.size])
        deg = np.repeat(cnt, cnt).astype(np.float64)
    else:
        deg = np.zeros(0)
    indptr = np.searchsorted(src, np.arange(n + 1))
    return src, z, dst, deg, indptr


def walk(n, g, seeds, hops, cap):
    """Every chain of 1..hops types from the seeds, in prefix order: (types, nodes, mass) for each (z sequence) with
    mass reaching some node. Stops extending once the chain count passes cap (returns capped True)."""
    src, z, dst, deg, indptr = g
    out = []
    if seeds.size == 0 or src.size == 0:
        return out, False
    front = [((), seeds, np.full(seeds.size, 1.0 / seeds.size))]
    for _ in range(hops):
        nxt = []
        for pre, idx, w in front:
            lens = indptr[idx + 1] - indptr[idx]
            tot = int(lens.sum())
            if tot == 0:
                continue
            sel = np.repeat(indptr[idx] - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(tot)
            ww = np.repeat(w, lens) / deg[sel]
            key = z[sel] * n + dst[sel]
            uq, inv = np.unique(key, return_inverse=True)
            m = np.bincount(inv, weights=ww)
            zz, vv = uq // n, uq % n
            st = np.flatnonzero(np.r_[True, zz[1:] != zz[:-1]])
            en = np.r_[st[1:], zz.size]
            for a_, b_ in zip(st, en):
                nxt.append((pre + (int(zz[a_]),), vv[a_:b_], m[a_:b_]))
        out.extend(nxt)
        front = nxt
        if len(out) > cap:
            return out, True
    return out, False


def candidates(chains, seeds):
    """The chains reaching a non-seed node, each renormalised over its non-seed nodes."""
    out = []
    for types, nodes, mass in chains:
        keep = ~np.isin(nodes, seeds)
        if keep.any():
            m = mass[keep]
            out.append((types, nodes[keep], m / m.sum()))
    return out


# ── ranking and metrics ──────────────────────────────────────────────────────


def rr_order(rrf):
    """Pool positions in rrf order, ties by position, and each node's rank in it."""
    order = np.lexsort((np.arange(rrf.size), -rrf))
    rank = np.empty(rrf.size, np.int64)
    rank[order] = np.arange(rrf.size)
    return order, rank


def top5(nodes, vals, rank, order_ns, seeds_rr):
    """The top 5 of a score that is vals on nodes (vals > 0), 0 on the other non-seed nodes and below 0 on the seeds,
    ties by rank: the fast form of np.lexsort((rank, -s))[:5] (--selftest checks they agree)."""
    if nodes.size:
        o = np.lexsort((rank[nodes], -vals))
        top = nodes[o[:5]]
    else:
        top = nodes
    if top.size < 5:
        need = 5 - top.size
        cand = order_ns[:need + nodes.size]
        cand = cand[~np.isin(cand, nodes)][:need]
        top = np.r_[top, cand]
        if top.size < 5:
            top = np.r_[top, seeds_rr[:5 - top.size]]
    return top


def metrics(top, gold, gt):
    if gt == 0 or top.size == 0:
        return (0.0, 0.0, 0.0)
    k = int(gold[top].sum())
    return (k / gt, float(k == gt), float(gold[top[0]]))


# ── collection ───────────────────────────────────────────────────────────────


class Grow:
    """A float64 matrix that grows by rows."""

    def __init__(self, d):
        self.a = np.zeros((1024, d))
        self.n = 0

    def ensure(self, n):
        if n > self.a.shape[0]:
            b = np.zeros((max(n, 2 * self.a.shape[0]), self.a.shape[1]))
            b[:self.a.shape[0]] = self.a
            self.a = b
        self.n = max(self.n, n)


def collect(paths, levels, P, hops, cap, rrf_col, m_idx):
    """One pass over the read chunks: per row its query, rank data, gold (for grading), floors and, per level, its
    candidate chains (vocabulary ids, end distributions, single-chain metrics); per level the chains' pooled end
    text."""
    L = len(levels)
    R = {"Q": [], "n": [], "gold": [], "gt": [], "seeds": [], "rank": [], "order_ns": [], "seeds_rr": [], "D": [],
         "twin0": [], "rrf": [], "rrf_s": []}
    V = [{"key": {}, "types": []} for _ in range(L)]
    RD = [Grow(RT.PROJ_DIM) for _ in range(L)]
    C = [{"cid": [], "ptr": [], "node": [], "mass": [], "met": []} for _ in range(L)]
    st = {"rows": 0, "rows_seed0": 0, "capped": [0] * L, "no_chain": [0] * L}
    for ci, p in enumerate(paths):
        with np.load(p) as zf:
            c = {k: zf[k] for k in KEYS}
        n_off = np.r_[0, np.cumsum(c["q_pool_size"])]
        e_off = np.r_[0, np.cumsum(c["q_edges"])]
        Qc = unit(unit(c["q_emb"]) @ P)
        for i in range(c["q_pool_size"].size):
            a, b = int(n_off[i]), int(n_off[i + 1])
            ea, eb = int(e_off[i]), int(e_off[i + 1])
            n = b - a
            hl, tl, sl, u, v = row_triples(c, a, b, ea, eb)
            gold = c["is_gold"][a:b].astype(bool)
            gt = int(c["q_gold_total"][i])
            rrf = c["x"][a:b, rrf_col].astype(np.float64)
            order, rank = rr_order(rrf)
            sl0 = c["q_seed_local"][i]
            bk = c["q_seed_bucket"][i]
            seeds = np.unique(sl0[(sl0 >= 0) & (bk == 0)]).astype(np.int64)
            is_seed = np.zeros(n, bool)
            is_seed[seeds] = True
            order_ns = order[~is_seed[order]]
            seeds_rr = order[is_seed[order]]
            pn = unit(c["proj"][a:b].astype(np.float32))
            R["Q"].append(Qc[i])
            R["n"].append(n)
            R["gold"].append(np.flatnonzero(gold).astype(np.int32))
            R["gt"].append(gt)
            R["seeds"].append(seeds)
            R["rank"].append(rank.astype(np.int32))
            R["order_ns"].append(order_ns.astype(np.int32))
            R["seeds_rr"].append(seeds_rr.astype(np.int32))
            R["twin0"].append(c["q_metrics"][i, 0, m_idx])
            R["rrf"].append(metrics(order[:5], gold, gt))
            R["rrf_s"].append(metrics(top5(np.zeros(0, np.int64), np.zeros(0), rank, order_ns, seeds_rr), gold, gt))
            if seeds.size:
                st["rows_seed0"] += 1
                dg = RT.gold_dist(n, u, v, np.flatnonzero(gold))
                R["D"].append(int(dg[seeds].min()))
            else:
                R["D"].append(RT.MAXHOP + 1)
            for li, lev in enumerate(levels):
                g = typed_graph(n, hl, tl, sl, lev)
                ch, capped = walk(n, g, seeds, hops, cap)
                st["capped"][li] += int(capped)
                cand = candidates(ch, seeds)
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
                for j, (_, nd, m) in enumerate(cand):
                    met[j] = metrics(top5(nd, m, rank, order_ns, seeds_rr), gold, gt)
                if cand:
                    W = sp.csr_matrix((mass, (np.repeat(np.arange(len(cand)), lens), nodes)), shape=(len(cand), n))
                    RD[li].ensure(len(Vl["types"]))
                    RD[li].a[cid] += W @ pn
                C[li]["cid"].append(cid)
                C[li]["ptr"].append(np.r_[0, np.cumsum(lens)])
                C[li]["node"].append(nodes.astype(np.int32))
                C[li]["mass"].append(mass.astype(np.float32))
                C[li]["met"].append(met)
            st["rows"] += 1
        if (ci + 1) % 25 == 0 or ci + 1 == len(paths):
            log(f"  chunk {ci + 1}/{len(paths)}: {st['rows']} rows; chains "
                + ", ".join(f"{lev.name} {len(V[li]['types'])}" for li, lev in enumerate(levels)))
    R["Q"] = np.stack(R["Q"]).astype(np.float32)
    for k in ("twin0", "rrf", "rrf_s"):
        R[k] = np.asarray(R[k], np.float64)
    R["D"] = np.asarray(R["D"], np.int64)
    rd = [unit(RD[li].a[:len(V[li]["types"])]) for li in range(len(levels))]
    return R, V, C, rd, st


# ── population fit ───────────────────────────────────────────────────────────


def pairs_of(Cl, rows):
    """(row, chain) pairs of the given rows (ascending), grouped by row, rows without chains left out."""
    row, cid = [], []
    for i in rows:
        x = Cl["cid"][i]
        if x.size:
            row.append(np.full(x.size, i, np.int64))
            cid.append(x)
    if not row:
        return None
    row = np.concatenate(row)
    cid = np.concatenate(cid)
    starts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
    grp = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, row.size]))
    return {"row": row, "c": cid, "starts": starts, "grp": grp, "rows": row[starts]}


def pair_cos(Q, pr, mu):
    out = np.empty(pr["row"].size, np.float64)
    B = 1 << 16
    for a in range(0, out.size, B):
        out[a:a + B] = np.einsum("kd,kd->k", Q[pr["row"][a:a + B]], mu[pr["c"][a:a + B]])
    return out


def em_fit(Q, pr, K, init, anchor=None, tau=0.0, fixed=False, kappa=KAPPA, iters=ITERS, alpha=ALPHA):
    """MAP-EM with an MM step for pi (see the docstring). Returns mu, pi and the objective at each round."""
    mu = np.array(init, dtype=np.float32, copy=True)
    pi = np.full(K, 1.0 / K)
    q = Q[pr["rows"]]
    nr = pr["starts"].size
    trace = []
    for it in range(iters + 1):
        logit = np.log(pi[pr["c"]]) + kappa * pair_cos(Q, pr, mu)
        mx = np.maximum.reduceat(logit, pr["starts"])
        e = np.exp(logit - mx[pr["grp"]])
        s = np.add.reduceat(e, pr["starts"])
        S = np.add.reduceat(pi[pr["c"]], pr["starts"])
        J = float((np.log(s) + mx - np.log(S)).sum() + alpha * np.log(pi).sum() - alpha * K * np.log(pi.sum()))
        if tau and anchor is not None:
            J += kappa * tau * float(np.einsum("kd,kd->", anchor.astype(np.float64), mu.astype(np.float64)))
        trace.append(J)
        if it == iters:
            break
        r = e / s[pr["grp"]]
        Rm = np.bincount(pr["c"], weights=r, minlength=K)
        Wm = np.bincount(pr["c"], weights=(1.0 / S)[pr["grp"]], minlength=K)
        pi = (Rm + alpha) / (Wm + alpha * K)
        pi = pi / pi.sum()
        if not fixed:
            M = sp.csr_matrix((r, (pr["c"], pr["grp"])), shape=(K, nr)) @ q.astype(np.float64)
            if tau and anchor is not None:
                M = M + tau * anchor
            live = np.linalg.norm(M, axis=1) > 1e-9
            mu[live] = unit(M[live])
    return mu, pi, trace


def row_ll(Q, pr, mu, pi, kappa=KAPPA):
    logit = np.log(pi[pr["c"]]) + kappa * pair_cos(Q, pr, mu)
    mx = np.maximum.reduceat(logit, pr["starts"])
    s = np.add.reduceat(np.exp(logit - mx[pr["grp"]]), pr["starts"])
    S = np.add.reduceat(pi[pr["c"]], pr["starts"])
    return np.log(s) + mx - np.log(S)


def posterior(Q, pr, mu, pi, kappa=KAPPA):
    logit = np.log(pi[pr["c"]]) + kappa * pair_cos(Q, pr, mu)
    mx = np.maximum.reduceat(logit, pr["starts"])
    e = np.exp(logit - mx[pr["grp"]])
    return e / np.add.reduceat(e, pr["starts"])[pr["grp"]]


def anchors(levels, V, rd, name_vec, present):
    """Per level: rd (pooled end text), nm (names, over the relations present in the read pools; None on none and
    dir) and each chain's parent chain id."""
    out = []
    for li, lev in enumerate(levels):
        T = V[li]["types"]
        nm = None
        if name_vec is not None and lev.name not in ("none", "dir"):
            K = lev.Kz // 2
            cn = np.zeros((K, name_vec.shape[1]))
            np.add.at(cn, lev.lab[present], name_vec[present])
            cn = unit(cn)
            nm = np.zeros((len(T), name_vec.shape[1]))
            for j, types in enumerate(T):
                for t in types:
                    nm[j] += cn[t // 2]
            nm = unit(nm)
        par, miss = None, 0
        if lev.parent is not None:
            pk = V[li - 1]["key"]
            par = np.full(len(T), -1, np.int64)
            for j, types in enumerate(T):
                up = lev.up(np.asarray(types, np.int64))
                x = pk.get(levels[li - 1].key(up))
                if x is None:
                    miss += 1
                else:
                    par[j] = x
        out.append({"rd": rd[li], "nm": nm, "par": par, "parent_missing": miss})
    return out


def fits_of(lev, A):
    return [f for f in FITS if not (f in NAMED and A["nm"] is None) and not (f == "hpop" and lev.parent is None)]


def fit_level(Q, pr, lev, A, K, parent_hmu):
    """All fits of one level on one set of rows. Returns {fit: (mu, pi, trace)} and this level's hpop mu (pop's at
    none) for the next level."""
    res = {}
    rd = A["rd"]
    for f in fits_of(lev, A):
        if f == "pop":
            res[f] = em_fit(Q, pr, K, rd)
        elif f == "pop1":
            res[f] = em_fit(Q, pr, K, rd, rd, TAU1)
        elif f == "rd":
            res[f] = em_fit(Q, pr, K, rd, fixed=True)
        elif f == "hpop":
            pm = parent_hmu[np.maximum(A["par"], 0)]
            pm[A["par"] < 0] = rd[A["par"] < 0]
            pm = unit(pm)
            res[f] = em_fit(Q, pr, K, rd, pm, N0)
        elif f == "nm1":
            res[f] = em_fit(Q, pr, K, A["nm"], A["nm"], TAU1)
        elif f == "nm":
            res[f] = em_fit(Q, pr, K, A["nm"], fixed=True)
    hmu = res["hpop"][0] if "hpop" in res else res["pop"][0]
    return res, hmu


# ── grading ──────────────────────────────────────────────────────────────────


def boot_mean(x, idx):
    bs = x[idx].mean(1)
    return [round(float(x.mean()), 5), round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def side(ci, lo_t=0.0, hi_t=None):
    hi_t = lo_t if hi_t is None else hi_t
    if ci[1] > hi_t:
        return "above"
    if ci[2] < lo_t:
        return "below"
    return "spans"


def share(arm, floor, twin, idx):
    num, den = (arm - floor), (twin - floor)
    pt = float(num.mean() / den.mean()) if den.mean() != 0 else float("nan")
    bs = num[idx].mean(1) / np.where(den[idx].mean(1) == 0, np.nan, den[idx].mean(1))
    return [round(pt, 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]


def family(k):
    """An arm's attachment source: names (nm1, nm) or end text (every other fit)."""
    return "cv" if k.split("/")[1] in NAMED else "cv-nn"


def verdicts(ds, arms, picks, order, R, idx):
    out = {}
    floor, twin = R["rrf_s"][:, 0], R["twin0"][:, 0]
    hs = {"above": "HELPS", "below": "HURTS", "spans": "SAME"}
    for tag in ("cv-nn", "cv"):
        nm = picks[tag]
        if nm is None:
            out[tag] = None
            continue
        x = arms[nm][:, 0]
        d1 = boot_mean(x - floor, idx)
        v1 = {"above": "ABOVE_FLOOR", "below": "BELOW_FLOOR", "spans": "AT_FLOOR"}[side(d1)]
        m = boot_mean(x, idx)
        zs = ZS_MLP.get(ds)
        v2 = None
        if zs is not None:
            v2 = "ABOVE_ZS" if m[1] > zs else "BELOW_ZS" if m[2] < zs else "AT_ZS"
        sh = share(x, floor, twin, idx)
        v3 = "HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID"
        fam = [k for k in order if k in arms and family(k) == tag]
        best = max(fam, key=lambda k: (arms[k][:, 0].mean(), -order.index(k)))
        gap = float(arms[best][:, 0].mean() - x.mean())
        d7 = boot_mean(x - arms["none/pop"][:, 0], idx)
        e = {"arm": nm, "V1": v1, "minus_rrf_s": d1, "V2": v2, "mean": m, "zs_mlp": zs, "V3": v3, "share": sh,
             "V4": {"best": best, "gap": round(gap, 5), "verdict": "MATCHES" if gap <= 0.01 else "MISSES"},
             "V7": {"minus_none_pop": d7, "verdict": hs[side(d7)]}}
        if ds == "webqsp":
            e["V6"] = {"band": list(J3A), "verdict": "ABOVE" if m[1] > J3A[1] else "BELOW" if m[2] < J3A[0] else "WITHIN"}
        out[tag] = e
    if ds == "webqsp" and picks["cv-nn"] is not None:
        nm = picks["cv-nn"]
        d5 = boot_mean(arms[nm][:, 0] - arms["exact/pop"][:, 0], idx)
        out["V5"] = {"diff": d5, "verdict": hs[side(d5)]}
        lv = nm.split("/")[0]
        if lv != "none" and f"{lv}/hpop" in arms:
            d5b = boot_mean(arms[f"{lv}/hpop"][:, 0] - arms[f"{lv}/pop"][:, 0], idx)
            out["V5b"] = {"level": lv, "diff": d5b, "verdict": hs[side(d5b)]}
        else:
            out["V5b"] = {"level": lv, "verdict": None}
    return out


# ── run ──────────────────────────────────────────────────────────────────────


def read_record(ds, cv, root):
    recs = sorted((Path(root) / ds / cv).glob("record*.json"))
    rec = json.loads(recs[0].read_text(encoding="utf-8"))
    cols = rec["columns"]
    names = rec["metric_names"]
    if rec["functions"][0] != "twin0":
        raise SystemExit("the look's first function is not twin0")
    return cols.index("rrf"), [names.index(m) for m in ("recall@5", "full_coverage@5", "hit@1")]


def run(a, root=LOOK, rel_dir=REL):
    t0 = time.time()
    P = RT.proj_matrix()
    paths, info = RT.chunk_paths(a.ds, a.read, root, a.limit)
    rrf_col, m_idx = read_record(a.ds, a.read, root)
    n_rel = info["n_relations"]
    log(f"{a.ds}={a.read}: {info['read']} of {info['n_chunks']} chunks, {n_rel} relations")
    # the relations' endpoint text, over the distinct triples of the read pools
    col = RT.Collector(P, (1,))
    col.add("read", paths)
    G, _rows, ginfo = col.finish()
    Pn = unit(G["proj"].astype(np.float32))
    S = G["slots"]
    ok = S >= 0
    tri = np.repeat(np.arange(S.shape[0]), ok.sum(1))
    ty = S[ok].astype(np.int64)
    if ty.size and int(ty.max()) >= n_rel:
        raise SystemExit(f"a relation id {int(ty.max())} is past the record's {n_rel}")
    one = np.ones(tri.size)
    nn = Pn.shape[0]
    Dt = sp.csr_matrix((one, (ty, G["T"][tri])), shape=(n_rel, nn)) @ Pn
    Dh = sp.csr_matrix((one, (ty, G["H"][tri])), shape=(n_rel, nn)) @ Pn
    present = np.bincount(ty, minlength=n_rel) > 0
    tt = np.c_[unit(Dt), unit(Dh)] / math.sqrt(2)
    ks = [int(k) for k in str(a.ks).split(",") if k]
    levels = hierarchy(tt, present, ks, n_rel)
    log(f"union graph {ginfo['triples']} triples, {int(present.sum())} relations present; levels "
        + ", ".join(f"{lev.name} (Kz {lev.Kz})" for lev in levels) + f"; {time.time() - t0:.0f}s")
    name_vec = None
    res = {"look": "chainpop17", "args": vars(a), "script_sha256": RT.sha(__file__), "reltype11_sha256": RT.sha(RT.__file__),
           "carve": info, "graph": ginfo, "relations_present": int(present.sum()),
           "levels": [{"name": lev.name, "Kz": lev.Kz} for lev in levels]}
    nv = Path(rel_dir) / f"{a.ds}_rel_embeddings.npy"
    if nv.exists():
        E = np.load(nv).astype(np.float32)
        if E.shape[0] != n_rel:
            raise SystemExit(f"{nv}: {E.shape[0]} names for {n_rel} relations")
        name_vec = unit(unit(E) @ P)
        res["name_embeddings_sha256"] = RT.sha(nv)
    # walks
    t1 = time.time()
    R, V, C, rd, st = collect(paths, levels, P, a.hops, a.max_chains, rrf_col, m_idx)
    N = st["rows"]
    A = anchors(levels, V, rd, name_vec, present)
    res["chains"] = {}
    for li, lev in enumerate(levels):
        cnt = np.asarray([x.size for x in C[li]["cid"]])
        hops_n = np.bincount([len(t) for t in V[li]["types"]], minlength=a.hops + 1)[1:].tolist()
        res["chains"][lev.name] = {"vocab": len(V[li]["types"]), "vocab_by_hops": hops_n,
                                   "per_row_p50": float(np.median(cnt)), "per_row_p90": float(np.percentile(cnt, 90)),
                                   "per_row_max": int(cnt.max()), "rows_capped": st["capped"][li],
                                   "rows_without_chain": st["no_chain"][li], "parent_missing": A[li]["parent_missing"]}
    res["rows"] = {"rows": N, "rows_seed0": st["rows_seed0"],
                   "seed_gold_D": {str(d): int((R["D"] == d).sum()) for d in range(RT.MAXHOP + 2)}}
    log(f"walks in {time.time() - t1:.0f}s: {res['chains']}")
    # label-free choice: 2-fold held-out likelihood
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    fold = np.zeros(N, np.int64)
    fold[perm[N // 2:]] = 1
    t2 = time.time()
    ll = {}
    for f_ in range(FOLDS):
        tr = np.flatnonzero(fold != f_)
        ho = np.flatnonzero(fold == f_)
        hmu = None
        for li, lev in enumerate(levels):
            K = len(V[li]["types"])
            prt = pairs_of(C[li], tr)
            prh = pairs_of(C[li], ho)
            if prt is None:
                break            # a finer level has a chain only where a coarser one does
            fits, hmu = fit_level(R["Q"], prt, lev, A[li], K, hmu)
            if prh is None:
                continue
            for f, (mu, pi, _tr) in fits.items():
                x = np.zeros(N)
                x[prh["rows"]] = row_ll(R["Q"], prh, mu, pi)
                ll.setdefault(f"{lev.name}/{f}", np.zeros(N))
                ll[f"{lev.name}/{f}"][ho] = x[ho]
    order = [f"{lev.name}/{f}" for lev in levels for f in fits_of(lev, A[levels.index(lev)])]
    cv_ll = {k: round(float(ll[k].mean()), 5) for k in order if k in ll}

    def pick(keys):
        keys = [k for k in keys if k in cv_ll]
        return max(keys, key=lambda k: (cv_ll[k], -order.index(k))) if keys else None
    picks = {"cv-nn": pick([k for k in order if family(k) == "cv-nn"]),
             "cv": pick([k for k in order if family(k) == "cv"]), "cv-all": pick(order)}
    log(f"cv in {time.time() - t2:.0f}s: {picks}; {cv_ll}")
    # final fits on every read row, posteriors, scores
    t3 = time.time()
    arms, post, traces = {}, {}, {}
    hmu = None
    allrows = np.arange(N)
    for li, lev in enumerate(levels):
        K = len(V[li]["types"])
        pr = pairs_of(C[li], allrows)
        if pr is None:
            break
        fits, hmu = fit_level(R["Q"], pr, lev, A[li], K, hmu)
        for f, (mu, pi, tr_) in fits.items():
            nm = f"{lev.name}/{f}"
            post[nm] = (li, pr, posterior(R["Q"], pr, mu, pi))
            traces[nm] = {"first": round(tr_[0], 3), "last": round(tr_[-1], 3),
                          "monotone": bool(all(y >= x - 1e-6 * max(1.0, abs(x)) for x, y in zip(tr_, tr_[1:]))),
                          "pi_perplexity": round(float(np.exp(-(pi * np.log(pi)).sum())), 2)}
    for nm, (li, pr, r) in post.items():
        out = np.zeros((N, 3))
        st_ = np.r_[pr["starts"], pr["row"].size]
        has = np.zeros(N, bool)
        for g in range(pr["starts"].size):
            i = int(pr["rows"][g])
            has[i] = True
            rr = r[st_[g]:st_[g + 1]]
            ptr = C[li]["ptr"][i]
            node = C[li]["node"][i].astype(np.int64)
            w = np.repeat(rr, np.diff(ptr)) * C[li]["mass"][i]
            s = np.bincount(node, weights=w, minlength=R["n"][i])
            sup = np.flatnonzero(s > 0)
            gold = np.zeros(R["n"][i], bool)
            gold[R["gold"][i]] = True
            out[i] = metrics(top5(sup, s[sup], R["rank"][i], R["order_ns"][i], R["seeds_rr"][i]), gold, R["gt"][i])
        out[~has] = R["rrf_s"][~has]
        arms[nm] = out
    # chain ceiling and identification
    ceil, ident = {}, {}
    for li, lev in enumerate(levels):
        best = R["rrf_s"].copy()
        on_all, rows_l, cid_l, starts = [], [], [], []
        pos = 0
        for i in range(N):
            met = C[li]["met"][i]
            if met.shape[0] == 0:
                continue
            j = np.lexsort((-met[:, 2], -met[:, 1], -met[:, 0]))[0]
            best[i] = met[j]
            mx = met[:, 0].max()
            on_all.append((met[:, 0] == mx) & (mx > 0))
            rows_l.append(np.full(met.shape[0], i))
            cid_l.append(C[li]["cid"][i])
            starts.append(pos)
            pos += met.shape[0]
        ceil[lev.name] = best
        if starts:
            prl = {"starts": np.asarray(starts), "z": np.concatenate(cid_l), "on": np.concatenate(on_all),
                   "row": np.concatenate(rows_l)}
            for nm, (lj, pr, r) in post.items():
                if lj == li:
                    ident[nm] = RT.grade_rule(prl, np.log(r + 1e-300), np.random.default_rng(SEED), R["D"])
    log(f"scored in {time.time() - t3:.0f}s")
    # grading
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, N, size=(BOOT, N))
    ref = {"rrf": R["rrf"], "rrf_s": R["rrf_s"], "twin0": R["twin0"]}
    M3 = ("recall@5", "full_coverage@5", "hit@1")
    table = {}
    for nm, x in list(arms.items()) + [(f"ceiling/{k}", v) for k, v in ceil.items()]:
        e = {"mean": [round(float(v), 5) for v in x.mean(0)], "cv_ll": cv_ll.get(nm)}
        for rn, rv in ref.items():
            e[f"minus_{rn}"] = {m: boot_mean(x[:, j] - rv[:, j], idx) for j, m in enumerate(M3)}
        if nm in arms:
            e["share_of_twin_lead"] = share(x[:, 0], R["rrf_s"][:, 0], R["twin0"][:, 0], idx)
            e["fit"] = traces[nm]
            if nm in ident:
                e["chain_id"] = ident[nm]
        table[nm] = e
    res["floors"] = {k: [round(float(v), 5) for v in ref[k].mean(0)] for k in ref}
    res["arms"] = table
    best = max(arms, key=lambda k: (arms[k][:, 0].mean(), -order.index(k)))
    res["cv"] = {"ll": cv_ll, "cv": picks["cv"], "cv_nn": picks["cv-nn"], "cv_all": picks["cv-all"], "folds": FOLDS,
                 "best_in_hindsight": best}
    res["verdicts"] = verdicts(a.ds, arms, picks, order, R, idx)
    res["seconds"] = round(time.time() - t0, 1)
    if a.rows_out:
        np.savez_compressed(a.rows_out, arms=np.stack([arms[k] for k in order if k in arms]),
                            arm_names=np.asarray([k for k in order if k in arms]), rrf=R["rrf"], rrf_s=R["rrf_s"],
                            twin0=R["twin0"], D=R["D"],
                            ceilings=np.stack([ceil[lev.name] for lev in levels]),
                            ceiling_names=np.asarray([lev.name for lev in levels]))
    log(f"floors {res['floors']}")
    for nm in order:
        if nm in table:
            e = table[nm]
            log(f"  {nm:14s} R@5/FC@5/hit@1 {e['mean']} ll {e['cv_ll']} vs rrf-s {e['minus_rrf_s']['recall@5']} "
                f"share {e['share_of_twin_lead']}")
    for lev in levels:
        log(f"  ceiling/{lev.name:8s} {table['ceiling/' + lev.name]['mean']}")
    log(f"verdicts {json.dumps(res['verdicts'])}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="metaqa")
    ap.add_argument("--read", default="x1f")
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--ks", default="2,4")
    ap.add_argument("--max-chains", type=int, default=MAX_CHAINS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out")
    ap.add_argument("--rows-out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not 1 <= a.hops <= RT.MAXHOP:
        raise SystemExit(f"--hops from 1 to {RT.MAXHOP}")
    res = run(a)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

METRIC_NAMES = ["recall@1", "recall@5", "recall@10", "recall@20", "hit@1", "mrr", "ndcg@5", "ndcg@20", "full_coverage@5",
                "full_coverage@20", "first_gold_rank", "gold_in_pool", "gold_total", "pool_size"]


def lean_metrics(s, gold, gt):
    """lean_mlp.row_metrics for one row (ties by pool position)."""
    if gt == 0:
        return (0.0, 0.0, 0.0)
    order = np.lexsort((np.arange(s.size), -s))
    top = int(gold[order[:5]].sum())
    return (top / gt, float(top == gt), float(gold[order[0]]))


def toy_world(root, rng, P, rows=120, chunk=8):
    """A look carve 'read' over reltype11's toy KB with one- and two-hop questions. A question asks a chain from its
    seed: a movie's director, actors, year or genre (one hop), a director's movies (one hop, backwards), the years of a
    director's movies, the genres of an actor's movies, or the other movies of a movie's director (two hops). Its query
    is the chain's template plus a little of the seed's text and noise. A template leans toward its end nodes' kind, as
    a KB's question words do ("who", "what year", "which genre"), so end text can attach a cluster of queries to its
    chain; director and actor questions both end at people, so only their availability can tell them apart. Gold is
    the chain's end set less the seed, and one row in ten has a gold node outside its pool. rrf is the query's cosine
    to node text; twin0 scores rrf plus a bonus on most gold nodes, and q_metrics holds its metrics."""
    ids, E, trip, M, kinds = RT.toy_kb(rng)
    dim = E.shape[1]
    by_pair = {}
    for h, t, r in trip:
        by_pair.setdefault((h, t), []).append(r)
    out_adj, in_adj = {}, {}
    for (h, t), rs in by_pair.items():
        for r in rs:
            out_adj.setdefault(h, []).append((t, r))
            in_adj.setdefault(t, []).append((h, r))
    directors = sorted({t for (h, t), rs in by_pair.items() if 0 in rs})
    actors = sorted({t for (h, t), rs in by_pair.items() if 1 in rs})

    def step(nodes, r, d):
        out = set()
        for x in nodes:
            if d == 0:
                out |= {t for t, r_ in out_adj.get(x, []) if r_ == r}
            else:
                out |= {h for h, r_ in in_adj.get(x, []) if r_ == r}
        return out
    asks = [("m", ((0, 0),)), ("m", ((1, 0),)), ("m", ((2, 0),)), ("m", ((3, 0),)), ("dir", ((0, 1),)),
            ("dir", ((0, 1), (2, 0))), ("act", ((1, 1), (3, 0))), ("m", ((0, 0), (0, 1)))]
    end = {((0, 0),): "p", ((1, 0),): "p", ((2, 0),): "y", ((3, 0),): "g", ((0, 1),): "m", ((0, 1), (2, 0)): "y",
           ((1, 1), (3, 0)): "g", ((0, 0), (0, 1)): "m"}
    tmpl = {ch: rng.standard_normal(dim) + 0.8 * kinds[end[ch]] for _, ch in asks}
    d = Path(root) / "toy" / "read"
    (d / "chunks").mkdir(parents=True)
    truth, R_ = [], []
    while len(R_) < rows:
        kind, ch = asks[int(rng.integers(len(asks)))]
        s = int(rng.choice(M if kind == "m" else directors if kind == "dir" else actors))
        cur = {s}
        for r, dr in ch:
            cur = step(cur, r, dr)
        gold = sorted(cur - {s})
        if not gold:
            continue
        q = unit(tmpl[ch] + 0.25 * E[s] * math.sqrt(dim) + 0.3 * rng.standard_normal(dim))
        R_.append((s, ch, gold, q))
    columns = ["dense_cos", "dense_rr", "splade_rr", "rrf"]
    n_ch = math.ceil(rows / chunk)
    for ci in range(n_ch):
        A = {k: [] for k in KEYS}
        A["score"] = []
        for (s, ch, gold, q) in R_[ci * chunk:(ci + 1) * chunk]:
            ball = {s}
            for _ in range(2):
                ball |= {x for y in list(ball) for x in [t for t, _ in out_adj.get(y, [])] + [h for h, _ in in_adj.get(y, [])]}
            extra = [int(x) for x in rng.choice(len(ids), 6, replace=False)]
            pool = list(dict.fromkeys([s] + sorted(ball - {s}) + extra))
            loc = {g_: i for i, g_ in enumerate(pool)}
            msgs = []
            for (h, t), rs in by_pair.items():
                if h in loc and t in loc:
                    sl = (rs + [-1, -1, -1, -1])[:4]
                    keep_f = rng.random() < 0.8
                    keep_b = rng.random() < 0.8 or not keep_f
                    if keep_f:
                        msgs.append((loc[h], loc[t], 0, 1, 0, sl))
                    if keep_b:
                        msgs.append((loc[t], loc[h], 0, 0, 1, sl))
            msgs.append((0, len(pool) - 1, 1, 0, 0, [-1] * 4))
            seeds, bucket = np.full(10, -1), np.full(10, -1)
            seeds[0], bucket[0] = 0, 0
            seeds[1], bucket[1] = len(pool) - 1, 1
            isg = np.zeros(len(pool), bool)
            isg[[loc[g_] for g_ in gold if g_ in loc]] = True
            gt = int(isg.sum()) + int(rng.random() < 0.1)
            rrf = (unit(E[pool]) @ q).astype(np.float64) + 0.01 * rng.standard_normal(len(pool))
            x = np.zeros((len(pool), 4), np.float16)
            x[:, 0] = rrf
            x[:, 3] = rrf
            tw = rrf + 3.0 * isg * (rng.random(len(pool)) < 0.7)
            qm = np.zeros((1, len(METRIC_NAMES)))
            m3 = lean_metrics(tw.astype(np.float32), isg, gt)
            qm[0, 1], qm[0, 8], qm[0, 4] = m3
            A["q_pool_size"].append(len(pool))
            A["q_edges"].append(len(msgs))
            A["q_emb"].append(q * 3.0)
            A["q_seed_local"].append(seeds)
            A["q_seed_bucket"].append(bucket)
            A["q_gold_total"].append(gt)
            A["q_metrics"].append(qm)
            A["pool"].append(ids[pool].astype(np.int32))
            A["proj"].append((E[pool] @ P).astype(np.float16))
            A["is_gold"].append(isg)
            A["x"].append(x)
            A["score"].append(tw.astype(np.float32)[:, None])
            for k, j, dt in (("e_u", 0, np.int16), ("e_v", 1, np.int16), ("e_fam", 2, np.int8), ("e_fwd", 3, np.int8),
                             ("e_bwd", 4, np.int8)):
                A[k].append(np.asarray([m_[j] for m_ in msgs], dtype=dt))
            A["e_rel"].append(np.asarray([m_[5] for m_ in msgs], dtype=np.int16).reshape(-1, 4))
            truth.append((s, ch, pool, gold))
        arr = {k: (np.asarray(v) if k.startswith("q_") else np.concatenate(v)) for k, v in A.items()}
        np.savez(d / "chunks" / f"c{ci:05d}.npz", **arr)
    rec = {"full": True, "chunks": list(range(n_ch)), "n_chunks": n_ch, "proj": {"dim": RT.PROJ_DIM, "seed": RT.PROJ_SEED},
           "relations": {"typed": True, "offset": 0, "n_relations": 4, "k_rel": 4}, "columns": columns,
           "functions": ["twin0"], "metric_names": METRIC_NAMES}
    (d / "record_0of1.json").write_text(json.dumps(rec), encoding="utf-8")
    names = np.stack([unit(tmpl[asks[r][1]] + 0.5 * rng.standard_normal(dim)) for r in range(4)]).astype(np.float16)
    return truth, names


def brute_chains(n, src, z, dst, Kz, seeds, hops):
    """Dense per-type transition matrices and every type sequence: the chains with mass, by brute force."""
    A = np.zeros((Kz, n, n))
    for s_, z_, d_ in zip(src, z, dst):
        A[z_, s_, d_] = 1.0
    deg = A.sum(2, keepdims=True)
    T = np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)
    m0 = np.zeros(n)
    m0[seeds] = 1.0 / seeds.size
    out = {}
    front = {(): m0}
    for _ in range(hops):
        nxt = {}
        for pre, m in front.items():
            for z_ in range(Kz):
                m2 = m @ T[z_]
                if (m2 > 0).any():
                    nxt[pre + (z_,)] = m2
        out.update(nxt)
        front = nxt
    return out


def selftest():
    import tempfile
    rng = np.random.default_rng(11)
    # top5 is lexsort((rank, -s))[:5] with seeds last
    for _ in range(300):
        n = int(rng.integers(1, 30))
        rrf = rng.integers(0, 4, n).astype(np.float64)
        order, rank = rr_order(rrf)
        seeds = np.unique(rng.integers(0, n, int(rng.integers(0, 3))))
        is_seed = np.zeros(n, bool)
        is_seed[seeds] = True
        k = int(rng.integers(0, n + 1))
        nodes = rng.permutation(np.flatnonzero(~is_seed))[:k]
        vals = rng.integers(1, 4, nodes.size).astype(np.float64)
        s = np.zeros(n)
        s[nodes] = vals
        s[seeds] = -1.0
        want = np.lexsort((rank, -s))[:5]
        got = top5(nodes, vals, rank, order[~is_seed[order]], order[is_seed[order]])
        assert np.array_equal(got, want), (got, want)
    # walk: every chain and its mass, against dense matrices
    for _ in range(40):
        n = int(rng.integers(2, 12))
        m = int(rng.integers(0, 25))
        hl, tl = rng.integers(0, n, m), rng.integers(0, n, m)
        sl = np.where(rng.random((m, 4)) < 0.4, rng.integers(0, 3, (m, 4)), -1)
        lev = Level("exact", lab=np.arange(3))
        seeds = np.unique(rng.integers(0, n, int(rng.integers(1, 3))))
        g = typed_graph(n, hl, tl, sl, lev)
        ch, capped = walk(n, g, seeds, 3, 10 ** 6)
        assert not capped
        got = {t: (nd, ms) for t, nd, ms in ch}
        want = brute_chains(n, g[0], g[1], g[2], lev.Kz, seeds, 3)
        assert set(got) == set(want), (sorted(got), sorted(want))
        for t, (nd, ms) in got.items():
            v = np.zeros(n)
            v[nd] = ms
            assert np.allclose(v, want[t]), t
            assert np.array_equal(nd, np.flatnonzero(want[t] > 0)), t
        for t, nd, ms in candidates(ch, seeds):
            assert not np.isin(nd, seeds).any() and abs(ms.sum() - 1) < 1e-9
    # a level's walk over merged types is the walk over the union of their edges
    lev_none = Level("none")
    n = 9
    hl, tl = rng.integers(0, n, 20), rng.integers(0, n, 20)
    sl = np.where(rng.random((20, 4)) < 0.5, rng.integers(0, 3, (20, 4)), -1)
    src, z, dst, deg, ip = typed_graph(n, hl, tl, sl, lev_none)
    und = {(int(a_), int(b_)) for a_, b_, r_ in zip(hl, tl, sl) if (r_ >= 0).any()}
    und |= {(b_, a_) for a_, b_ in und}
    assert set(zip(src.tolist(), dst.tolist())) == und and (z == 0).all()
    # the real chunk's twin0 metrics, if this machine has one
    real = LOOK / "webqsp" / "selectf" / "chunks" / "c00000.npz"
    if real.exists():
        rrf_col, m_idx = read_record("webqsp", "selectf", LOOK)
        with np.load(real) as zf:
            c = {k: zf[k] for k in ("q_pool_size", "is_gold", "q_gold_total", "q_metrics", "score")}
        off = np.r_[0, np.cumsum(c["q_pool_size"])]
        for i in range(off.size - 1):
            s = c["score"][off[i]:off[i + 1], 0]
            got = lean_metrics(s, c["is_gold"][off[i]:off[i + 1]], int(c["q_gold_total"][i]))
            assert np.allclose(got, c["q_metrics"][i, 0, m_idx]), (i, got, c["q_metrics"][i, 0, m_idx])
        print("the look's twin0 metrics are lean_mlp.row_metrics' on a real webqsp chunk")
    with tempfile.TemporaryDirectory() as td:
        P = RT.proj_matrix()
        truth, names = toy_world(td, rng, P)
        rel_dir = Path(td) / "rel"
        rel_dir.mkdir()
        np.save(rel_dir / "toy_rel_embeddings.npy", names)
        rows_out = Path(td) / "rows.npz"
        a = argparse.Namespace(ds="toy", read="read", hops=2, ks="2", max_chains=MAX_CHAINS, limit=None, out=None,
                               rows_out=str(rows_out))
        res = run(a, root=td, rel_dir=rel_dir)
        # the toy's own twin0 metrics come back through q_metrics
        rrf_col, m_idx = read_record("toy", "read", td)
        paths = RT.chunk_paths("toy", "read", td)[0]
        tw = []
        for p in paths:
            with np.load(p) as zf:
                off = np.r_[0, np.cumsum(zf["q_pool_size"])]
                for i in range(off.size - 1):
                    tw.append(lean_metrics(zf["score"][off[i]:off[i + 1], 0], zf["is_gold"][off[i]:off[i + 1]],
                                           int(zf["q_gold_total"][i])))
        assert np.allclose(np.asarray(tw).mean(0), res["floors"]["twin0"], atol=1e-5)
        names_l = [x["name"] for x in res["levels"]]
        assert names_l == ["none", "dir", "tt2", "exact"], names_l
        for lv, c_ in res["chains"].items():
            assert c_["parent_missing"] == 0 and c_["rows_capped"] == 0, (lv, c_)
        for nm, e in res["arms"].items():
            if "fit" in e:
                assert e["fit"]["monotone"], (nm, e["fit"])
        A_ = res["arms"]
        for lv in names_l:
            assert A_[f"ceiling/{lv}"]["mean"][0] >= max(e["mean"][0] for k, e in A_.items()
                                                          if k.startswith(lv + "/")) - 1e-9, lv
        with np.load(rows_out) as z:
            assert z["arms"].shape[0] == len(z["arm_names"]) and z["arms"].shape[1] == len(truth)
        res2 = run(a, root=td, rel_dir=rel_dir)

        def strip(x):
            if isinstance(x, dict):
                return {k: strip(v) for k, v in x.items() if k not in ("seconds", "script_sha256")}
            return x
        assert json.dumps(strip(res), sort_keys=True) == json.dumps(strip(res2), sort_keys=True), "not deterministic"
        print("toy floors", res["floors"])
        for nm, e in A_.items():
            print(f"  {nm:12s} {e['mean']} ll {e.get('cv_ll')} auc {(e.get('chain_id') or {}).get('auc')}")
        print("toy cv", res["cv"], "\nverdicts", json.dumps(res["verdicts"]))
        # the method on a toy where end text attaches most chains
        cvnn, cvn = res["cv"]["cv_nn"], res["cv"]["cv"]
        assert cvnn.split("/")[0] in ("tt2", "exact") and family(cvnn) == "cv-nn", res["cv"]
        assert cvn.split("/")[1] in NAMED, res["cv"]
        for k in (cvnn, cvn):
            assert A_[k]["mean"][0] > A_["none/pop"]["mean"][0] + 0.1, (k, A_[k]["mean"], A_["none/pop"]["mean"])
            assert A_[k]["chain_id"]["auc"][0] > 0.75, (k, A_[k]["chain_id"])
            assert res["verdicts"][family(k)]["V7"]["verdict"] == "HELPS", res["verdicts"][family(k)]
    print("selftest: top5 is lexsort with seeds last; every chain and its mass match dense per-type walks; candidates "
          "drop the seeds and renormalise; the untyped level is the union graph read both ways; the look's metrics are "
          "lean_mlp's; levels nest and every chain's parent is a candidate one level up; EM never lowers its "
          "objective; each level's ceiling bounds its arms; deterministic; on a toy whose question words lean toward "
          "their answers' kind, cv-nn and cv pick typed levels whose chains beat the untyped walk and identify the "
          "asked chain. all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
