"""Design look (untracked; not a result and not filed): S6 part 2a of the transfer plan, "transfer the estimator, not
the weights". Part 1 (chainpop17, read 10:14 and 10:19 on 4 Oct) found three things. A walk along the right relation
chain matches a trained model on a KB: the exact-chain ceiling is 78.51 R@5 on metaqa x1f against the twin's 73.23.
A question population sharpens an attachment where it is thick (metaqa, +20 points over either start) and collapses
where it is thin (webqsp, 0.003 rows per exact chain). And the held-out likelihood cannot say which chain a group of
questions asks: only text (relation names, end-node text) or a row's reachable chains attaches a cluster to a chain.
Part 2a learns that attachment where gold exists and reads it on a KB it never saw.

A small scorer g rates each (question, candidate chain) pair, at every level of chainpop17's hierarchy at once (none,
dir, tt<k>, exact), from graph-agnostic quantities only. It never sees a raw embedding, a node id or a relation id,
so it cannot memorise a graph's templates or relations:
    text        cos_e    cos(q, the row's own chain end text, sum_v m(v) n(p_v))
                cos_nm   cos(q, the chain's relation-name text, chainpop17's nm), with has_nm
                cos_rd   cos(q, the population's pooled end text of the chain, chainpop17's rd)
    population  lr_<fit> log posterior under each of chainpop17's six fits (pop, pop1, rd, hpop, nm1, nm), floored at
                         -50 for the scorer
                mu_<fit> cos(q, the fitted mu) of pop, hpop and nm1
                lpi_pop, lpi_nm1   log(pi_c K), the fitted prior against uniform
                avail    log(1 + the population's rows that have the chain)
                thick    log(the population's rows with a chain at the level / K), rows per chain
                logN     log(the population's rows)
                ent_pop, ent_nm1   the row's posterior entropy over its candidates, over log(their count)
    structure   hop1..hop3, fwd (share of forward steps; 0.5 at none), k_none/k_dir/k_tt/k_exact (the level's
                kind), reach (log 1 + non-seed nodes reached), ment (the end distribution's entropy over log reach),
                mmax (its peak), minrank (-log(1 + the best rrf rank reached) / log pool size), logC (log of the
                row's candidate count at the level)
none has no hpop: its hpop quantities are its pop's, as in chainpop17. A fit that does not exist (the name fits on
none and dir, or anywhere on a graph without relation-name embeddings) reads as no information: a uniform posterior
over the row's candidates (lr = -logC, normalised entropy 1), mu and lpi 0.

Multiple regimes (Swastik's coarse-to-fine population idea). On the training carve the population is refit at six
sizes: all rows; disjoint random groups of 1/4, 1/20, 1/100 and 1/500 of them (about 1,490, 298, 60 and 12 rows on
metaqa's fit carve); and each row alone. Every row is scored inside its own group, as a read row is inside its own
graph, so g sees estimates from populations as thick as metaqa's and as thin as webqsp's (305 rows over 110,439
exact chains, most seen by one or two rows; the one-row regime is that limit). The thickness quantities let it
learn when to trust a fine chain and when to put its weight on a coarser one: an unseen or thinly asked relation can
back off to its parent cluster, dir or the untyped walk. In each group every level is fitted as in chainpop17 (its
fit_level, unchanged), over the chains its rows have, with the end text pooled over its own rows.

The scorer. g is an MLP (inputs standardised on the training pairs, two hidden layers of 64, ReLU). A pair's logit
is g(x) - logC (the log of the row's candidate count at the pair's level), so before g learns anything every level
holds the same mass whatever its candidate count: webqsp's exact level has about ten times metaqa's candidates per
row, and without the offset an untrained g puts most of its mass there. g is trained listwise: for each
(row, regime), a softmax over all the row's candidate chains at every allowed level, and the loss
-log of the probability on the chains that alone reach the row's best single-chain R@5 among them (rows where no
allowed chain reaches gold are left out). A single chain is graded as the scorer's walk ranks it, on its float32
masses with ties to rrf; chainpop17 graded its ceilings on float64 masses, where near-equal masses order by rounding
noise, so the ceilings reported here are chainpop17's and the labels may differ from them on a few rows. Adam, lr 1e-3, weight decay 1e-4, 12 epochs of batches of 256 (row, regime)
groups, torch deterministic on one thread. An arm trained on one regime repeats its rows once per regime an epoch,
so every arm takes the same number of steps. The epoch is chosen by mean R@5 on metaqa's select carve (its own
population). Score: s(v) = sum over the row's allowed candidates of p_c m_c(v), seeds last, ties by rrf
(chainpop17's top5); a row with no allowed chain is ranked as rrf-s.

Arms:
    full    every feature, every level, all six regimes
    one     full trained on the whole-carve regime only: is the multi-regime training what transfers?
    nopop   no population quantity at all (the row's own end text, the names and the structure)
    noname  no name quantity: the attachment a graph without relation names has
    coarse  full restricted to the none and dir levels (the back-off floor; its labels are recomputed within)
    cf      full with a coarse-to-fine curriculum: epochs 1-2 none and dir, 3-4 add tt, 5-12 add exact; its epoch is
            chosen among 5-12
References, recomputed here from the same walks and fits as chainpop17 (each read reports how closely it reproduces
part 1's arms): every level/fit posterior walk (none/pop and none/rd are the untyped walks), the chain ceilings, rrf,
rrf-s and the twin.

Reads: metaqa x1f (in-domain: the graph g trained on, other rows) and webqsp selectf (zero-shot: a KB that g never
trained or selected on; its population is its own 305 rows).

Verdicts, fixed at 10:55 on 4 Oct before any number from this file. Paired row bootstrap (BOOT 1000) on R@5;
HELPS / HURTS when the interval excludes 0, else SAME:
    W1  webqsp: full minus none/rd (part 1's best label-free arm there): ABOVE / AT / BELOW
    W2  full minus one (the multi-regime training), on webqsp and on metaqa
    W3  full minus nopop (the population), on webqsp and on metaqa
    W4  full minus noname (the names), on webqsp and on metaqa
    W5  cf minus full (the curriculum), on webqsp and on metaqa
    W6  webqsp: full minus coarse (do finer chains add to the back-off floor?)
    M1  metaqa: full against exact/nm1 (part 1's best arm) and against the twin: ABOVE / AT / BELOW; its share of the
        twin's lead over rrf-s: HIGH if the interval lies at or above 0.5, LOW if below 0.25, else MID
    Z1  webqsp: full against l7g-j3a's KB-trained zero-shot band (0.20 to 0.25 R@5): ABOVE / WITHIN / BELOW; its
        share of the twin's lead
If W1 is ABOVE, part 2b feeds g's walk to the lean MLP and the GNN. If not, the result goes to Swastik before
anything else is built. Train-split rows throughout: a look, not a result.

    python outputs/mp_unified/chainscore18.py build --ds metaqa --carve fit --hops 3 --ks 2,4 \
        --regimes 1,0.25,0.05,0.01,0.002,0 --no-nodes --out outputs/mp_unified/lean/cs18-mq-fit.npz
    python outputs/mp_unified/chainscore18.py build --ds metaqa --carve select --hops 3 --ks 2,4 \
        --out outputs/mp_unified/lean/cs18-mq-select.npz
    python outputs/mp_unified/chainscore18.py build --ds metaqa --carve x1f --hops 3 --ks 2,4 \
        --out outputs/mp_unified/lean/cs18-mq-x1f.npz
    python outputs/mp_unified/chainscore18.py build --ds webqsp --carve selectf --hops 2 --ks 4,16,64,256 \
        --out outputs/mp_unified/lean/cs18-wq.npz
    python outputs/mp_unified/chainscore18.py train --train outputs/mp_unified/lean/cs18-mq-fit.npz \
        --select outputs/mp_unified/lean/cs18-mq-select.npz --read metaqa=outputs/mp_unified/lean/cs18-mq-x1f.npz \
        --read webqsp=outputs/mp_unified/lean/cs18-wq.npz --part1 metaqa=outputs/mp_unified/lean/cp17-mq.json \
        --part1 webqsp=outputs/mp_unified/lean/cp17-wq.json --out outputs/mp_unified/lean/cs18.json \
        --rows-out outputs/mp_unified/lean/cs18.rows.npz
    python outputs/mp_unified/chainscore18.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainpop17's count, so the whole-carve fits reproduce its run

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainpop17 as CP  # noqa: E402
import reltype11 as RT  # noqa: E402

LOOK, REL = RT.LOOK, RT.REL
SEED = 20261004
BOOT = 1000
REGIMES = "1,0.25,0.05,0.01,0.002,0"
EPOCHS = 12
BATCH = 256
LR = 1e-3
WD = 1e-4
HID = 64
LR_FLOOR = -50.0
J3A = (0.20, 0.25)
FI = ("cos_e", "cos_nm", "has_nm", "hop1", "hop2", "hop3", "fwd", "k_none", "k_dir", "k_tt", "k_exact", "reach",
      "ment", "mmax", "minrank", "logC")
FR = ("cos_rd", "mu_pop", "mu_hpop", "mu_nm1", "lr_pop", "lr_pop1", "lr_rd", "lr_hpop", "lr_nm1", "lr_nm", "lpi_pop",
      "lpi_nm1", "avail", "thick", "logN", "ent_pop", "ent_nm1")
ALLF = FI + FR
LRC = [ALLF.index("lr_" + f) for f in CP.FITS]
LOGC = FI.index("logC")
NAMEF = ("cos_nm", "has_nm", "mu_nm1", "lr_nm1", "lr_nm", "lpi_nm1", "ent_nm1")
ARMS = ("full", "one", "nopop", "noname", "coarse", "cf")
COARSE = ("none", "dir")
log, unit = RT.log, RT.unit


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kind(name):
    return name if name in ("none", "dir", "exact") else "tt"


# ── build: walks, regimes, features ──────────────────────────────────────────


def fwd_share(types, lev):
    if lev.name == "none":
        return 0.5
    return float(np.mean([int(t) % 2 == 0 for t in types]))


def collect(paths, levels, P, hops, cap, rrf_col, m_idx, keep_nodes):
    """chainpop17.collect's pass (its helpers, unchanged), keeping also each (row, level, chain)'s end text
    e = sum_v m(v) n(p_v) and its walk statistics; node lists and rank data only when keep_nodes (scored carves)."""
    L = len(levels)
    R = {"Q": [], "n": [], "gold": [], "gt": [], "rank": [], "order_ns": [], "seeds_rr": [], "D": [], "twin0": [],
         "rrf": [], "rrf_s": []}
    V = [{"key": {}, "types": []} for _ in range(L)]
    RD = [CP.Grow(RT.PROJ_DIM) for _ in range(L)]
    C = [{"cid": [], "ptr": [], "node": [], "mass": [], "met": [], "met32": [], "e": [], "ws": []} for _ in range(L)]
    st = {"rows": 0, "rows_seed0": 0, "capped": [0] * L, "no_chain": [0] * L}
    for ci, p in enumerate(paths):
        with np.load(p) as zf:
            c = {k: zf[k] for k in CP.KEYS}
        n_off = np.r_[0, np.cumsum(c["q_pool_size"])]
        e_off = np.r_[0, np.cumsum(c["q_edges"])]
        Qc = unit(unit(c["q_emb"]) @ P)
        for i in range(c["q_pool_size"].size):
            a, b = int(n_off[i]), int(n_off[i + 1])
            ea, eb = int(e_off[i]), int(e_off[i + 1])
            n = b - a
            if n >= 1 << 15:
                raise SystemExit(f"a pool of {n} nodes does not fit int16")
            hl, tl, sl, u, v = CP.row_triples(c, a, b, ea, eb)
            gold = c["is_gold"][a:b].astype(bool)
            gt = int(c["q_gold_total"][i])
            rrf = c["x"][a:b, rrf_col].astype(np.float64)
            order, rank = CP.rr_order(rrf)
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
            if keep_nodes:
                R["rank"].append(rank.astype(np.int16))
                R["order_ns"].append(order_ns.astype(np.int16))
                R["seeds_rr"].append(seeds_rr.astype(np.int16))
            R["twin0"].append(c["q_metrics"][i, 0, m_idx])
            R["rrf"].append(CP.metrics(order[:5], gold, gt))
            R["rrf_s"].append(CP.metrics(CP.top5(np.zeros(0, np.int64), np.zeros(0), rank, order_ns, seeds_rr), gold,
                                         gt))
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
                    ws[j] = (len(types), fwd_share(types, lev), math.log1p(nd.size),
                             ent / math.log(nd.size) if nd.size > 1 else 0.0, float(m.max()), float(rank[nd].min()))
                if cand:
                    W = sp.csr_matrix((mass, (np.repeat(np.arange(len(cand)), lens), nodes)), shape=(len(cand), n))
                    E_ = np.asarray(W @ pn)
                    RD[li].ensure(len(Vl["types"]))
                    RD[li].a[cid] += E_
                else:
                    E_ = np.zeros((0, RT.PROJ_DIM))
                C[li]["cid"].append(cid)
                C[li]["met"].append(met)
                C[li]["met32"].append(met32)
                C[li]["e"].append(E_.astype(np.float16))
                C[li]["ws"].append(ws)
                if keep_nodes:
                    C[li]["ptr"].append(np.r_[0, np.cumsum(lens)])
                    C[li]["node"].append(nodes.astype(np.int16))
                    C[li]["mass"].append(mass.astype(np.float32))
            st["rows"] += 1
        if (ci + 1) % 25 == 0 or ci + 1 == len(paths):
            log(f"  chunk {ci + 1}/{len(paths)}: {st['rows']} rows; chains "
                + ", ".join(f"{lev.name} {len(V[li]['types'])}" for li, lev in enumerate(levels)))
    R["Q"] = np.stack(R["Q"]).astype(np.float32)
    for k in ("twin0", "rrf", "rrf_s"):
        R[k] = np.asarray(R[k], np.float64)
    R["D"] = np.asarray(R["D"], np.int64)
    rd = [unit(RD[li].a[:len(V[li]["types"])]) for li in range(L)]
    return R, V, C, rd, st


def layout(C, N, L):
    """Global pair order: row, then level, then the row's candidate order. Returns the per-row offsets, the per
    (row, level) offsets within the row, each pair's level, and the (row, level) counts."""
    cnt = np.asarray([[C[li]["cid"][i].size for li in range(L)] for i in range(N)], np.int64).reshape(N, L)
    rowoff = np.r_[0, np.cumsum(cnt.sum(1))]
    levoff = np.c_[np.zeros(N, np.int64), np.cumsum(cnt, 1)[:, :-1]]
    p_lev = np.repeat(np.tile(np.arange(L), N), cnt.ravel()).astype(np.int8)
    return rowoff, levoff, p_lev, cnt


def fixed_features(R, C, levels, A, rowoff, levoff, cnt):
    N = len(R["n"])
    F = np.zeros((int(rowoff[-1]), len(FI)), np.float32)
    col = {k: j for j, k in enumerate(FI)}
    for li, lev in enumerate(levels):
        nm = A[li]["nm"]
        kc = col["k_" + kind(lev.name)]
        for i in range(N):
            k = int(cnt[i, li])
            if not k:
                continue
            g0 = int(rowoff[i] + levoff[i, li])
            s = slice(g0, g0 + k)
            q = R["Q"][i].astype(np.float64)
            F[s, col["cos_e"]] = unit(C[li]["e"][i].astype(np.float64)) @ q
            if nm is not None:
                F[s, col["cos_nm"]] = nm[C[li]["cid"][i]] @ q
                F[s, col["has_nm"]] = 1.0
            ws = C[li]["ws"][i]
            h = ws[:, 0].astype(np.int64)
            for hh in (1, 2, 3):
                F[s, col[f"hop{hh}"]] = (h == hh)
            F[s, col["fwd"]] = ws[:, 1]
            F[s, kc] = 1.0
            F[s, col["reach"]] = ws[:, 2]
            F[s, col["ment"]] = ws[:, 3]
            F[s, col["mmax"]] = ws[:, 4]
            F[s, col["minrank"]] = -np.log1p(ws[:, 5]) / math.log(max(R["n"][i], 2))
            F[s, col["logC"]] = math.log(k)
    return F


def groups_of(f, N, rng):
    if f >= 1.0:
        return [np.arange(N)]
    if f <= 0.0:
        return [np.asarray([i]) for i in range(N)]
    perm = rng.permutation(N)
    return [np.sort(x) for x in np.array_split(perm, max(1, int(round(1.0 / f))))]


def regime_features(R, C, V, levels, A, rd_all, regs, rowoff, levoff, rng, keep_post):
    """Per regime: the rows' groups (populations) and, per pair, the population quantities from each group's own
    fits. Returns FR (regimes x pairs x len(FR); lr unfloored, -inf where the posterior underflows to 0), gid
    (regimes x rows), the whole-carve posteriors (pairs x fits, float64, NaN where a fit does not exist; only when
    keep_post) and the whole-carve fits' summaries."""
    N = len(R["n"])
    col = {k: j for j, k in enumerate(FR)}
    out = np.zeros((len(regs), int(rowoff[-1]), len(FR)), np.float32)
    gids = np.zeros((len(regs), N), np.int64)
    post1 = np.full((int(rowoff[-1]), len(CP.FITS)), np.nan) if keep_post else None
    summ = {}
    Q = R["Q"]
    for ri, f in enumerate(regs):
        t0 = time.time()
        groups = groups_of(f, N, rng)
        for g, rows in enumerate(groups):
            gids[ri, rows] = g
            hmu, loc_prev = None, None
            for li, lev in enumerate(levels):
                rows_l = [int(i) for i in rows if C[li]["cid"][i].size]
                if not rows_l:
                    break
                allc = np.concatenate([C[li]["cid"][i] for i in rows_l])
                uniq = np.unique(allc)
                K = uniq.size
                loc = np.full(len(V[li]["types"]), -1, np.int64)
                loc[uniq] = np.arange(K)
                Cl = {"cid": {i: loc[C[li]["cid"][i]] for i in rows_l}}
                pr = CP.pairs_of(Cl, rows_l)
                if f >= 1.0:
                    rd = rd_all[li][uniq]
                else:
                    Ecat = np.concatenate([C[li]["e"][i] for i in rows_l]).astype(np.float64)
                    M = sp.csr_matrix((np.ones(allc.size), (loc[allc], np.arange(allc.size))), shape=(K, allc.size))
                    rd = unit(np.asarray(M @ Ecat))
                nm = A[li]["nm"][uniq] if A[li]["nm"] is not None else None
                par = None
                if lev.parent is not None:
                    pg = A[li]["par"][uniq]
                    par = np.where(pg >= 0, loc_prev[np.maximum(pg, 0)], -1)
                fits, hmu = CP.fit_level(Q, pr, lev, {"rd": rd, "nm": nm, "par": par}, K, hmu)
                slot = np.arange(pr["row"].size) - pr["starts"][pr["grp"]]
                gpos = rowoff[pr["row"]] + levoff[pr["row"], li] + slot
                nk = np.diff(np.r_[pr["starts"], pr["row"].size]).astype(np.float64)
                nkp = nk[pr["grp"]]
                for fi_, fn in enumerate(CP.FITS):
                    src = fn if fn in fits else ("pop" if fn == "hpop" else None)
                    if src is None:
                        # the fit does not exist here: no information, a uniform posterior
                        out[ri, gpos, col["lr_" + fn]] = -np.log(nkp)
                        if fn == "nm1":
                            out[ri, gpos, col["ent_nm1"]] = (nkp > 1)
                        continue
                    mu, pi, tr_ = fits[src]
                    r = CP.posterior(Q, pr, mu, pi)
                    if keep_post and f >= 1.0 and fn in fits:
                        post1[gpos, fi_] = r
                    lr = np.full(r.size, -np.inf)
                    np.log(r, out=lr, where=r > 0)
                    out[ri, gpos, col["lr_" + fn]] = lr
                    if fn in ("pop", "hpop", "nm1"):
                        out[ri, gpos, col["mu_" + fn]] = CP.pair_cos(Q, pr, mu)
                    if fn in ("pop", "nm1"):
                        out[ri, gpos, col["lpi_" + fn]] = np.log(pi[pr["c"]] * K)
                        h = np.add.reduceat(-r * np.where(r > 0, lr, 0.0), pr["starts"])
                        hn = np.where(nk > 1, h / np.log(np.maximum(nk, 2)), 0.0)
                        out[ri, gpos, col["ent_" + fn]] = hn[pr["grp"]]
                    if f >= 1.0 and fn in fits:
                        summ[f"{lev.name}/{fn}"] = {
                            "monotone": bool(all(y >= x - 1e-6 * max(1.0, abs(x)) for x, y in zip(tr_, tr_[1:]))),
                            "pi_perplexity": round(float(np.exp(-(pi * np.log(pi)).sum())), 2), "K": int(K)}
                out[ri, gpos, col["cos_rd"]] = CP.pair_cos(Q, pr, rd)
                out[ri, gpos, col["avail"]] = np.log1p(np.bincount(pr["c"], minlength=K)[pr["c"]])
                out[ri, gpos, col["thick"]] = math.log(len(rows_l) / K)
                out[ri, gpos, col["logN"]] = math.log(len(rows))
                loc_prev = loc
        log(f"  regime {f}: {len(groups)} population(s) of {min(len(x) for x in groups)} to "
            f"{max(len(x) for x in groups)} rows, {time.time() - t0:.0f}s")
    return out, gids, post1, summ


def build(a, root=LOOK, rel_dir=REL):
    t0 = time.time()
    P = RT.proj_matrix()
    paths, info = RT.chunk_paths(a.ds, a.carve, root, a.limit)
    rrf_col, m_idx = CP.read_record(a.ds, a.carve, root)
    n_rel = info["n_relations"]
    log(f"{a.ds}={a.carve}: {info['read']} of {info['n_chunks']} chunks, {n_rel} relations")
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
    levels = CP.hierarchy(tt, present, ks, n_rel)
    log(f"union graph {ginfo['triples']} triples, {int(present.sum())} relations present; levels "
        + ", ".join(f"{lev.name} (Kz {lev.Kz})" for lev in levels))
    name_vec, name_sha = None, None
    nv = Path(rel_dir) / f"{a.ds}_rel_embeddings.npy"
    if nv.exists():
        E = np.load(nv).astype(np.float32)
        if E.shape[0] != n_rel:
            raise SystemExit(f"{nv}: {E.shape[0]} names for {n_rel} relations")
        name_vec = unit(unit(E) @ P)
        name_sha = sha(nv)
    keep = not a.no_nodes
    t1 = time.time()
    R, V, C, rd_all, st = collect(paths, levels, P, a.hops, a.max_chains, rrf_col, m_idx, keep)
    N, L = st["rows"], len(levels)
    A = CP.anchors(levels, V, rd_all, name_vec, present)
    log(f"walks in {time.time() - t1:.0f}s")
    rowoff, levoff, p_lev, cnt = layout(C, N, L)
    FIa = fixed_features(R, C, levels, A, rowoff, levoff, cnt)
    regs = [float(x) for x in str(a.regimes).split(",") if x]
    if regs[0] != 1.0:
        raise SystemExit("the first regime must be 1 (the whole carve)")
    t2 = time.time()
    FRa, gids, post1, summ = regime_features(R, C, V, levels, A, rd_all, regs, rowoff, levoff,
                                             np.random.default_rng(SEED), keep)
    log(f"regimes in {time.time() - t2:.0f}s")
    met = np.concatenate([C[li]["met"][i] for i in range(N) for li in range(L)])
    met32 = np.concatenate([C[li]["met32"][i] for i in range(N) for li in range(L)])
    p_cid = np.concatenate([C[li]["cid"][i] for i in range(N) for li in range(L)])
    gold_off = np.r_[0, np.cumsum([x.size for x in R["gold"]])].astype(np.int64)
    meta = {"look": "chainscore18", "args": {k: v for k, v in vars(a).items()},
            "script_sha256": sha(__file__), "chainpop17_sha256": sha(CP.__file__), "reltype11_sha256": sha(RT.__file__),
            "carve": info, "graph": ginfo, "relations_present": int(present.sum()), "name_embeddings_sha256": name_sha,
            "levels": [{"name": lev.name, "Kz": lev.Kz, "vocab": len(V[li]["types"]),
                        "pairs": int(cnt[:, li].sum()), "parent_missing": A[li]["parent_missing"]}
                       for li, lev in enumerate(levels)],
            "rows": N, "rows_seed0": st["rows_seed0"], "capped": st["capped"], "no_chain": st["no_chain"],
            "pairs": int(rowoff[-1]), "regimes": regs, "fits": summ}
    arr = {"FI": FIa, "FR": FRa, "gid": gids, "rowoff": rowoff, "p_lev": p_lev, "p_cid": p_cid, "met": met,
           "met32": met32,
           "D": R["D"], "n": np.asarray(R["n"], np.int32), "gt": np.asarray(R["gt"], np.int32),
           "gold": np.concatenate(R["gold"]).astype(np.int32), "gold_off": gold_off, "twin0": R["twin0"],
           "rrf": R["rrf"], "rrf_s": R["rrf_s"], "kinds": np.asarray([kind(lev.name) for lev in levels]),
           "level_names": np.asarray([lev.name for lev in levels])}
    if keep:
        parts_n, parts_m, ends = [], [], []
        base = 0
        for i in range(N):
            for li in range(L):
                pt = C[li]["ptr"][i]
                parts_n.append(C[li]["node"][i])
                parts_m.append(C[li]["mass"][i])
                ends.append(base + pt[1:])
                base += int(pt[-1])
        arr["node"] = np.concatenate(parts_n).astype(np.int16)
        arr["mass"] = np.concatenate(parts_m).astype(np.float32)
        arr["node_off"] = np.r_[0, np.concatenate(ends)].astype(np.int64)
        for k, ko in (("rank", "rank_off"), ("order_ns", "order_off"), ("seeds_rr", "seeds_off")):
            arr[k] = np.concatenate(R[k]).astype(np.int16)
            arr[ko] = np.r_[0, np.cumsum([x.size for x in R[k]])].astype(np.int64)
        arr["post1"] = post1
    meta["seconds"] = round(time.time() - t0, 1)
    arr["meta"] = np.asarray(json.dumps(meta))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **arr)
    os.replace(tmp, out)
    log(f"built {N} rows, {int(rowoff[-1])} pairs, {len(regs)} regime(s) in {time.time() - t0:.0f}s -> {out}")
    return meta


# ── train and read ───────────────────────────────────────────────────────────


def load(p):
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def arm_cols(arm):
    drop = set()
    if arm == "nopop":
        drop |= set(FR)
    if arm == "noname":
        drop |= set(NAMEF)
    return [j for j, k in enumerate(ALLF) if k not in drop], [k for k in ALLF if k not in drop]


def cf_phases(epochs):
    """The curriculum's first epoch with tt and its first with every level (2 and 4 of 12)."""
    a = max(1, epochs // 6)
    full0 = min(2 * a, epochs - 1)
    return min(a, full0), full0


def allowed_kinds(arm, ep, epochs):
    if arm == "coarse":
        return COARSE
    if arm == "cf":
        tt0, full0 = cf_phases(epochs)
        if ep < tt0:
            return COARSE
        if ep < full0:
            return COARSE + ("tt",)
    return ("none", "dir", "tt", "exact")


def cf_first_full(epochs):
    return cf_phases(epochs)[1]


def pair_mask(d, kinds_ok):
    okl = np.asarray([k in kinds_ok for k in d["kinds"]])
    return okl[d["p_lev"].astype(np.int64)]


def labels(d, allow):
    """Per pair: on = allowed and alone reaching the row's best allowed single-chain R@5 (> 0); per row: has one.
    Single chains are graded as the scorer's walk ranks them (met32: float32 masses, ties by rrf)."""
    r5 = np.where(allow, d["met32"][:, 0], -1.0)
    ro = d["rowoff"]
    N = ro.size - 1
    has = np.diff(ro) > 0
    best = np.full(N, -1.0)
    if has.any():
        best[has] = np.maximum.reduceat(r5, ro[:-1][has])
    on = allow & (r5 == best[np.repeat(np.arange(N), np.diff(ro))]) & (r5 > 0)
    return on, (best > 0)


def features(d, ri, cols):
    X = np.concatenate([d["FI"], d["FR"][ri]], 1)
    X[:, LRC] = np.maximum(X[:, LRC], LR_FLOOR)
    return np.ascontiguousarray(X[:, cols], dtype=np.float32)


def standardise(Xs):
    """Mean and standard deviation over the stacked rows of Xs, in float64 blocks; then Xs normalised in place."""
    n = sum(X.shape[0] for X in Xs)
    s1 = np.zeros(Xs[0].shape[1])
    s2 = np.zeros(Xs[0].shape[1])
    for X in Xs:
        for a in range(0, X.shape[0], 1 << 18):
            B = X[a:a + (1 << 18)].astype(np.float64)
            s1 += B.sum(0)
            s2 += (B * B).sum(0)
    m = s1 / n
    s = np.sqrt(np.maximum(s2 / n - m * m, 0.0))
    s[s < 1e-6] = 1.0
    m, s = m.astype(np.float32), s.astype(np.float32)
    for X in Xs:
        X -= m
        X /= s
    return m, s


def make_model(nf, seed):
    import torch
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(nf, HID), torch.nn.ReLU(), torch.nn.Linear(HID, HID), torch.nn.ReLU(),
                               torch.nn.Linear(HID, 1))


def logits_of(model, X):
    import torch
    out = np.empty(X.shape[0], np.float64)
    with torch.no_grad():
        for a in range(0, X.shape[0], 1 << 17):
            out[a:a + (1 << 17)] = model(torch.from_numpy(X[a:a + (1 << 17)])).squeeze(-1).double().numpy()
    return out


def pair_logits(model, X, d):
    """g(x) - logC per pair (logC unstandardised, from the carve's own fixed features)."""
    return logits_of(model, X) - d["FI"][:, LOGC].astype(np.float64)


def score_rows(d, w, allow):
    """Per row: s(v) = sum over allowed pairs of w m(v) (w normalised within the row), its top 5 and metrics; rows
    with no allowed pair (or no weight) are rrf-s."""
    ro = d["rowoff"]
    N = ro.size - 1
    out = np.array(d["rrf_s"], dtype=np.float64, copy=True)
    no, mass, node = d["node_off"], d["mass"], d["node"]
    for i in range(N):
        a, b = int(ro[i]), int(ro[i + 1])
        idx = np.flatnonzero(allow[a:b]) + a
        if not idx.size:
            continue
        wi = w[idx].astype(np.float64)
        if not np.isfinite(wi).all() or wi.sum() <= 0:
            continue
        wi = wi / wi.sum()
        lens = no[idx + 1] - no[idx]
        tot = int(lens.sum())
        sel = np.repeat(no[idx] - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(tot)
        n = int(d["n"][i])
        s = np.bincount(node[sel].astype(np.int64), weights=np.repeat(wi, lens) * mass[sel], minlength=n)
        sup = np.flatnonzero(s > 0)
        gold = np.zeros(n, bool)
        gold[d["gold"][d["gold_off"][i]:d["gold_off"][i + 1]]] = True
        rank = d["rank"][d["rank_off"][i]:d["rank_off"][i + 1]].astype(np.int64)
        ons = d["order_ns"][d["order_off"][i]:d["order_off"][i + 1]].astype(np.int64)
        srr = d["seeds_rr"][d["seeds_off"][i]:d["seeds_off"][i + 1]].astype(np.int64)
        out[i] = CP.metrics(CP.top5(sup, s[sup], rank, ons, srr), gold, int(d["gt"][i]))
    return out


def softmax_rows(d, z, allow):
    ro = d["rowoff"]
    N = ro.size - 1
    zz = np.where(allow, z, -np.inf)
    has = np.diff(ro) > 0
    st = ro[:-1][has]
    mx = np.full(N, -np.inf)
    if has.any():
        mx[has] = np.maximum.reduceat(zz, st)
    rep = np.repeat(np.arange(N), np.diff(ro))
    e = np.where(allow, np.exp(zz - np.where(np.isfinite(mx), mx, 0.0)[rep]), 0.0)
    s = np.zeros(N)
    if has.any():
        s[has] = np.add.reduceat(e, st)
    return np.where(allow, e / np.where(s > 0, s, 1.0)[rep], 0.0)


def chain_top1(d, p, on, allow):
    """Rows with an on pair: is the scorer's top allowed pair on (ties shared)."""
    ro = d["rowoff"]
    out = []
    for i in range(ro.size - 1):
        a, b = int(ro[i]), int(ro[i + 1])
        o = on[a:b]
        if not o.any():
            continue
        pi = np.where(allow[a:b], p[a:b], -1.0)
        out.append(float(o[pi == pi.max()].mean()))
    return np.asarray(out)


def train_arm(arm, T, Sd, epochs, seed, threads=1):
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    cols, names = arm_cols(arm)
    regs = list(T["meta"]["regimes"])
    use = [0] if arm in ("one", "nopop") else list(range(len(regs)))
    reps = len(regs) if arm in ("one", "nopop") else 1
    Xs = [features(T, ri, cols) for ri in use]
    mean, std = standardise(Xs)
    Xsel = (features(Sd, 0, cols) - mean) / std
    model = make_model(len(cols), seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)
    ro = T["rowoff"]
    logc = T["FI"][:, LOGC]
    hist, best, best_state = [], None, None
    first_ok = cf_first_full(epochs) if arm == "cf" else 0
    for ep in range(epochs):
        ks = allowed_kinds(arm, ep, epochs)
        allow = pair_mask(T, ks)
        on, keep = labels(T, allow)
        rows = np.flatnonzero(keep)
        grp = np.tile(np.c_[np.repeat(np.arange(len(use)), rows.size), np.tile(rows, len(use))], (reps, 1))
        grp = grp[rng.permutation(len(grp))]
        model.train()
        tot, nb = 0.0, 0
        for b0 in range(0, len(grp), BATCH):
            bg = grp[b0:b0 + BATCH]
            idxs = [np.flatnonzero(allow[ro[i]:ro[i + 1]]) + ro[i] for _, i in bg]
            Lm = max(x.size for x in idxs)
            Xb = np.zeros((len(bg), Lm, len(cols)), np.float32)
            Mb = np.zeros((len(bg), Lm), bool)
            Ob = np.zeros((len(bg), Lm), bool)
            Cb = np.zeros((len(bg), Lm), np.float32)
            for j, ((u, _), ix) in enumerate(zip(bg, idxs)):
                Xb[j, :ix.size] = Xs[u][ix]
                Mb[j, :ix.size] = True
                Ob[j, :ix.size] = on[ix]
                Cb[j, :ix.size] = logc[ix]
            z = model(torch.from_numpy(Xb)).squeeze(-1) - torch.from_numpy(Cb)
            z = z.masked_fill(~torch.from_numpy(Mb), float("-inf"))
            lp = torch.log_softmax(z, 1)
            loss = -torch.logsumexp(lp.masked_fill(~torch.from_numpy(Ob), float("-inf")), 1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        model.eval()
        sel_allow = pair_mask(Sd, ks)
        p = softmax_rows(Sd, pair_logits(model, Xsel, Sd), sel_allow)
        sr5 = float(score_rows(Sd, p, sel_allow)[:, 0].mean())
        hist.append({"epoch": ep, "kinds": list(ks), "loss": round(tot / max(nb, 1), 5), "select_r5": round(sr5, 5),
                     "groups": int(len(grp))})
        log(f"    {arm} ep {ep}: loss {tot / max(nb, 1):.4f} select R@5 {sr5:.4f} ({'/'.join(ks)})")
        if ep >= first_ok and (best is None or sr5 > best[1]):
            best = (ep, sr5)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"cols": names, "col_idx": cols, "mean": mean, "std": std, "history": hist, "best_epoch": best[0],
                   "best_select_r5": round(best[1], 5), "kinds_final": list(allowed_kinds(arm, epochs - 1, epochs))}


def reference_arms(d):
    """Every level/fit posterior walk of the whole-carve population (part 1's arms, from the stored float64
    posteriors), the chain ceilings, rrf, rrf-s and the twin, per row (N x 3)."""
    refs = {"rrf": np.asarray(d["rrf"]), "rrf_s": np.asarray(d["rrf_s"]), "twin0": np.asarray(d["twin0"])}
    post1 = d["post1"]
    ro = d["rowoff"]
    met = d["met"]
    for li, name in enumerate(d["level_names"]):
        allow = d["p_lev"].astype(np.int64) == li
        if not allow.any():
            continue
        for fi_, fn in enumerate(CP.FITS):
            w = post1[:, fi_]
            if np.isnan(w[allow]).all():
                continue
            refs[f"{name}/{fn}"] = score_rows(d, np.where(allow, np.nan_to_num(w), 0.0), allow)
        best = np.array(d["rrf_s"], dtype=np.float64, copy=True)
        for i in range(ro.size - 1):
            idx = np.flatnonzero(allow[ro[i]:ro[i + 1]]) + ro[i]
            if idx.size:
                m = met[idx]
                best[i] = m[np.lexsort((-m[:, 2], -m[:, 1], -m[:, 0]))[0]]
        refs[f"ceiling/{name}"] = best
    return refs


def verdict_side(ci, labels_=("ABOVE", "BELOW", "AT")):
    return {"above": labels_[0], "below": labels_[1], "spans": labels_[2]}[CP.side(ci)]


def train(a):
    t0 = time.time()
    T = load(a.train)
    Sd = load(a.select)
    reads = {}
    for x in a.read:
        nm_, p_ = x.split("=", 1)
        reads[nm_] = load(p_)
    part1 = {}
    for x in a.part1 or []:
        nm_, p_ = x.split("=", 1)
        part1[nm_] = json.loads(Path(p_).read_text(encoding="utf-8"))
    res = {"look": "chainscore18", "args": vars(a), "script_sha256": sha(__file__),
           "chainpop17_sha256": sha(CP.__file__),
           "inputs": {"train": T["meta"], "select": Sd["meta"], **{k: v["meta"] for k, v in reads.items()}},
           "inputs_sha256": {p: sha(p) for p in [a.train, a.select] + [x.split("=", 1)[1] for x in a.read]},
           "features": {"fixed": FI, "regime": FR, "names": NAMEF}, "arms": {}, "refs": {}, "repro": {}}
    rows_out = {}
    refs = {}
    for nm_, d in reads.items():
        t1 = time.time()
        refs[nm_] = reference_arms(d)
        res["refs"][nm_] = {k: [round(float(v), 5) for v in x.mean(0)] for k, x in refs[nm_].items()}
        if nm_ in part1:
            rp = {}
            for k, e in part1[nm_]["arms"].items():
                if k in refs[nm_]:
                    mine = refs[nm_][k].mean(0)
                    rp[k] = [round(float(mine[0]), 5), e["mean"][0], round(float(mine[0] - e["mean"][0]), 6)]
            res["repro"][nm_] = {"max_abs_r5_diff": max((abs(v[2]) for v in rp.values()), default=None),
                                 "arms": rp}
        log(f"references for {nm_} in {time.time() - t1:.0f}s; repro "
            f"{res['repro'].get(nm_, {}).get('max_abs_r5_diff')}")
        for k, x in refs[nm_].items():
            rows_out[f"{nm_}|ref|{k}"] = x.astype(np.float32)
    for ai, arm in enumerate(a.arms.split(",")):
        t1 = time.time()
        model, info = train_arm(arm, T, Sd, a.epochs, SEED + 101 * ai)
        e = {"history": info["history"], "best_epoch": info["best_epoch"], "best_select_r5": info["best_select_r5"],
             "cols": info["cols"], "kinds": info["kinds_final"], "reads": {}}
        for nm_, d in reads.items():
            X = (features(d, 0, info["col_idx"]) - info["mean"]) / info["std"]
            allow = pair_mask(d, info["kinds_final"])
            p = softmax_rows(d, pair_logits(model, X, d), allow)
            x = score_rows(d, p, allow)
            rows_out[f"{nm_}|arm|{arm}"] = x.astype(np.float32)
            on, _ = labels(d, allow)
            t1s = chain_top1(d, p, on, allow)
            N = x.shape[0]
            idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
            D = d["D"]
            lv = d["p_lev"].astype(np.int64)
            r = {"mean": [round(float(v), 5) for v in x.mean(0)], "rows": int(N),
                 "chain_top1": round(float(t1s.mean()), 4) if t1s.size else None,
                 "mass_by_level": {str(n_): round(float(p[lv == li].sum() / max(N, 1)), 4)
                                   for li, n_ in enumerate(d["level_names"])},
                 "by_D": {str(int(dd)): [round(float(x[D == dd, 0].mean()), 5), int((D == dd).sum())]
                          for dd in np.unique(D)}}
            for rn in ("none/pop", "none/rd", "exact/pop", "exact/nm1", "twin0", "rrf_s"):
                if rn in refs[nm_]:
                    r[f"minus_{rn}"] = CP.boot_mean(x[:, 0] - refs[nm_][rn][:, 0], idx)
            r["share"] = CP.share(x[:, 0], refs[nm_]["rrf_s"][:, 0], refs[nm_]["twin0"][:, 0], idx)
            e["reads"][nm_] = r
        res["arms"][arm] = e
        log(f"  {arm}: best epoch {info['best_epoch']} ({time.time() - t1:.0f}s) "
            + "; ".join(f"{k} {v['mean']} mass {v['mass_by_level']}" for k, v in e["reads"].items()))
    res["verdicts"] = verdicts(reads, refs, rows_out)
    res["seconds"] = round(time.time() - t0, 1)
    if a.rows_out:
        p = Path(a.rows_out)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, **{k.replace("/", "~").replace("|", "__"): v for k, v in rows_out.items()},
                            **{f"{k}__D": v["D"] for k, v in reads.items()})
        os.replace(tmp, p)
    log(f"verdicts {json.dumps(res['verdicts'])}")
    return res


def verdicts(reads, refs, rows):
    out = {}
    hs = ("HELPS", "HURTS", "SAME")
    for nm_ in reads:
        full = rows.get(f"{nm_}|arm|full")
        if full is None:
            continue
        N = refs[nm_]["rrf_s"].shape[0]
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        v = {}
        f5 = full[:, 0].astype(np.float64)
        for tag, other in (("W2", "one"), ("W3", "nopop"), ("W4", "noname")):
            o = rows.get(f"{nm_}|arm|{other}")
            if o is not None:
                ci = CP.boot_mean(f5 - o[:, 0], idx)
                v[tag] = {"diff": ci, "verdict": verdict_side(ci, hs)}
        cf = rows.get(f"{nm_}|arm|cf")
        if cf is not None:
            ci = CP.boot_mean(cf[:, 0].astype(np.float64) - f5, idx)
            v["W5"] = {"diff": ci, "verdict": verdict_side(ci, hs)}
        sh = CP.share(f5, refs[nm_]["rrf_s"][:, 0], refs[nm_]["twin0"][:, 0], idx)
        shv = "HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID"
        if nm_ == "webqsp":
            if "none/rd" in refs[nm_]:
                ci = CP.boot_mean(f5 - refs[nm_]["none/rd"][:, 0], idx)
                v["W1"] = {"diff": ci, "verdict": verdict_side(ci)}
            co = rows.get(f"{nm_}|arm|coarse")
            if co is not None:
                ci = CP.boot_mean(f5 - co[:, 0], idx)
                v["W6"] = {"diff": ci, "verdict": verdict_side(ci, hs)}
            m = CP.boot_mean(f5, idx)
            v["Z1"] = {"mean": m, "band": list(J3A),
                       "verdict": "ABOVE" if m[1] > J3A[1] else "BELOW" if m[2] < J3A[0] else "WITHIN",
                       "share": sh, "share_verdict": shv}
        if nm_ == "metaqa":
            m1 = {"share": sh, "share_verdict": shv}
            if "exact/nm1" in refs[nm_]:
                ci = CP.boot_mean(f5 - refs[nm_]["exact/nm1"][:, 0], idx)
                m1["vs_exact_nm1"] = {"diff": ci, "verdict": verdict_side(ci)}
            ci = CP.boot_mean(f5 - refs[nm_]["twin0"][:, 0], idx)
            m1["vs_twin"] = {"diff": ci, "verdict": verdict_side(ci)}
            v["M1"] = m1
        out[nm_] = v
    return out


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    P = RT.proj_matrix()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        built = {}
        for nm_, sd, rows in (("train", 11, 128), ("select", 12, 64), ("read", 13, 96)):
            root = td / nm_
            reld = td / f"{nm_}_rel"
            reld.mkdir(parents=True)
            _truth, names = CP.toy_world(root, np.random.default_rng(sd), P, rows=rows)
            np.save(reld / "toy_rel_embeddings.npy", names)
            a = SimpleNamespace(ds="toy", carve="read", hops=2, ks="2", limit=None, max_chains=CP.MAX_CHAINS,
                                regimes="1,0.25,0.05,0" if nm_ == "train" else "1", no_nodes=nm_ == "train",
                                out=str(td / f"{nm_}.npz"))
            meta = build(a, root=root, rel_dir=reld)
            built[nm_] = (root, reld, meta)
            d = load(a.out)
            N = d["n"].size
            # every row sits in exactly one population per regime; the first regime is the whole carve
            assert (d["gid"][0] == 0).all()
            for ri, f in enumerate(meta["regimes"]):
                want = N if f <= 0 else max(1, int(round(1 / f)))
                assert np.array_equal(np.unique(d["gid"][ri]), np.arange(want)), (f, np.unique(d["gid"][ri]))
                X = features(d, ri, list(range(len(ALLF))))
                assert np.isfinite(X).all(), f
            assert d["rowoff"][-1] == d["FI"].shape[0] == d["FR"].shape[1] == d["met"].shape[0]
            assert all(lv["parent_missing"] == 0 for lv in meta["levels"]), meta["levels"]
            # population posteriors are normalised within each (row, level), in every regime
            ro = d["rowoff"]
            for ri in range(len(meta["regimes"])):
                for fn in CP.FITS:
                    w = np.exp(d["FR"][ri, :, FR.index("lr_" + fn)].astype(np.float64))
                    for i in range(ro.size - 1):
                        pl = d["p_lev"][ro[i]:ro[i + 1]]
                        for li in np.unique(pl):
                            assert abs(w[ro[i]:ro[i + 1]][pl == li].sum() - 1.0) < 1e-4, (ri, fn, i, li)
            if nm_ == "train":
                # one-row populations: avail is log 2 and thick is -logC on every pair
                ri = meta["regimes"].index(0.0)
                assert np.allclose(d["FR"][ri, :, FR.index("avail")], math.log(2), atol=1e-6)
                assert np.allclose(d["FR"][ri, :, FR.index("thick")], -d["FI"][:, FI.index("logC")], atol=1e-5)
        # the whole-carve references reproduce chainpop17's run on the same carve
        root, reld, _ = built["read"]
        cp_rows = td / "cp17_rows.npz"
        cp = CP.run(SimpleNamespace(ds="toy", read="read", hops=2, ks="2", max_chains=CP.MAX_CHAINS, limit=None,
                                    out=None, rows_out=str(cp_rows)), root=root, rel_dir=reld)
        d = load(td / "read.npz")
        refs = reference_arms(d)
        missing = sorted(set(cp["arms"]) - set(refs))
        with np.load(cp_rows) as z:
            want = dict(zip(z["arm_names"].tolist(), z["arms"]))
            want.update({f"ceiling/{k}": v for k, v in zip(z["ceiling_names"].tolist(), z["ceilings"])})
            want.update({k: z[k] for k in ("rrf", "rrf_s", "twin0")})
        diffs = {k: float(np.abs(refs[k] - v).max()) for k, v in want.items()}
        print("toy repro: max per-row |diff| against chainpop17's rows", max(diffs.values()), "missing", missing)
        assert not missing and max(diffs.values()) < 1e-12, (missing, diffs)
        # a one-hot weight on each row's ceiling chain reproduces the ceiling
        L = len(d["level_names"])
        allow = d["p_lev"].astype(np.int64) == L - 1
        ro = d["rowoff"]
        w = np.zeros(ro[-1])
        want = np.array(d["rrf_s"], dtype=np.float64, copy=True)
        for i in range(ro.size - 1):
            idx = np.flatnonzero(allow[ro[i]:ro[i + 1]]) + ro[i]
            if idx.size:
                m = d["met32"][idx]
                j = np.lexsort((-m[:, 2], -m[:, 1], -m[:, 0]))[0]
                w[idx[j]] = 1.0
                want[i] = m[j]
        x = score_rows(d, w, allow)
        assert np.abs(x - want).max() < 1e-6, "one-hot parity with met32"
        drift = np.abs(want - refs[f"ceiling/{d['level_names'][-1]}"]).max(1) > 1e-6
        print(f"toy: float32 masses change the ceiling chain's grade on {int(drift.sum())} of {drift.size} rows")
        # training: deterministic, the loss falls within each arm's final level set, and full reads above the
        # untyped walk on a toy KB it never trained on
        outs = []
        for _ in range(2):
            a = SimpleNamespace(train=str(td / "train.npz"), select=str(td / "select.npz"),
                                read=[f"toy={td / 'read.npz'}"], part1=None, arms=",".join(ARMS), epochs=6,
                                out=None, rows_out=None)
            outs.append(train(a))
        r0, r1 = outs
        for arm in ARMS:
            assert r0["arms"][arm]["reads"]["toy"]["mean"] == r1["arms"][arm]["reads"]["toy"]["mean"], arm
            assert r0["arms"][arm]["history"] == r1["arms"][arm]["history"], arm
            h = [x for x in r0["arms"][arm]["history"] if x["kinds"] == r0["arms"][arm]["kinds"]]
            print(f"toy {arm}: loss {h[0]['loss']} -> {h[-1]['loss']}, best epoch {r0['arms'][arm]['best_epoch']}, "
                  f"read {r0['arms'][arm]['reads']['toy']['mean']}")
            assert len(h) < 2 or h[-1]["loss"] < h[0]["loss"], (arm, h)
        print("toy refs", {k: v for k, v in r0["refs"]["toy"].items() if k.startswith(("none/", "exact/", "ceiling",
                                                                                         "rrf", "twin"))})
        full = r0["arms"]["full"]["reads"]["toy"]["mean"][0]
        assert full > r0["refs"]["toy"]["none/pop"][0], (full, r0["refs"]["toy"]["none/pop"])
        assert r0["arms"]["cf"]["best_epoch"] >= cf_first_full(6)
        assert r0["arms"]["coarse"]["kinds"] == list(COARSE)
        assert set(r0["arms"]["nopop"]["cols"]) == set(FI) and not set(NAMEF) & set(r0["arms"]["noname"]["cols"])
    print("selftest: populations partition the rows in every regime; features finite; posteriors normalised per "
          "(row, level) in every regime; one-row populations read avail log 2 and thick -logC; the whole-carve "
          "references reproduce chainpop17's rows exactly; a one-hot weight reproduces its chain's float32 grade; "
          "training deterministic, "
          "its loss falls, and the scorer beats the untyped walk on an unseen toy KB. all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--ds", required=True)
    b.add_argument("--carve", required=True)
    b.add_argument("--hops", type=int, default=3)
    b.add_argument("--ks", default="2,4")
    b.add_argument("--regimes", default="1")
    b.add_argument("--max-chains", type=int, default=CP.MAX_CHAINS)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--no-nodes", action="store_true")
    b.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--select", required=True)
    t.add_argument("--read", action="append", required=True)
    t.add_argument("--part1", action="append")
    t.add_argument("--arms", default=",".join(ARMS))
    t.add_argument("--epochs", type=int, default=EPOCHS)
    t.add_argument("--out")
    t.add_argument("--rows-out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "build":
        if not 1 <= a.hops <= RT.MAXHOP:
            raise SystemExit(f"--hops from 1 to {RT.MAXHOP}")
        build(a)
        return 0
    if a.cmd == "train":
        res = train(a)
        if a.out:
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
