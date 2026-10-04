"""Design look (untracked; not a result and not filed): S5 of the transfer plan, the MLP's label-free alignment.
lean_mlp7g (lean_mlp7 with its step guard) is imported and called unchanged. This file adds three AW arms and qd_gnn15's
population fit, read over the AW walks' typed edges. The fit gives each walk edge a scalar per query. That scalar is
read from the relation's rank and the carve's own unlabeled queries, never from the relation's name, any gold, or any
trained model.

Types. A walk edge's type is z = (d, rank), where d is its famdir (0 S-fwd, 1 S-bwd, 2 S-both) and rank is the label
lean_mlp7 reads for it: a KB relation's rank by count, which is index-time structure, so no name is read. Three kinds
of edge are untyped and get a = 0:
  - OTHER edges: no label, a held label under the mask, or a rank past K_LAB;
  - ner edges;
  - knn edges.

Population fit. One fit per carve and view, over the carve's own rows, with no gold, no name and no model. It is
qd_gnn15's fit, run on the carve's pool edges:
  q_i   = n(n(q_emb_i) R), where R = lean_mlp.projection(), the look's 128-wide projection (= reltype11.proj_matrix())
  d_z   = n(sum of n(p_v) over the population's distinct (u, v, z) typed structural edges), where p_v is the look's
          stored projection of the far endpoint
  C_i   = the types of row i's typed structural edges out of its bucket-0 seeds to a node outside them; with none, out
          of all its seeds to a node outside them
  EM    = reltype11.em over (q_i, C_i), started from d_z (kappa --al-kappa, default 20; 30 rounds). With --al-tau T > 0,
          the M step is anchored to d_z (qd_gnn15.em_tau, copied here).
  pi~_z = (mass_z + 1) / (rows with C + T)
  a(i, z) = clip(log pi~_z + kappa cos(q_i, mu_z) - logsumexp over z' of the same + log T, -10, 10)
Under --hold, the held KB's training and select carves read the held relation as OTHER from the start (lean_mlp7's mask),
so their fits never see it. Each of its read carves gets two fits:
  REV   the held edges typed
  MASK  the held edges untyped
Only KB graphs are fitted (--al-graphs kb). Passage graphs carry anchor labels, but their typing is a later step.

Arms. These are added to lean_mlp7's arms, and every one of them reads the AW block.
  aw0    lean_mlp7's aw with every walk label read OTHER in the logit, so no label text: the label-free twin. Its dropout
         draws are aw's, so it starts from aw's weights and consumes aw's random stream.
  aw0al  aw0 + lam[d] a(i, z_e) in each walk edge's logit. lam starts at 0, so aw0al starts as aw0.
  awal   aw + lam[d] a(i, z_e)
Label dropout hides a label's text and never its type, as in qd_gnn15. NR also keeps a, because types are index-time
structure and only names are hidden.

Reads. Each --read carve gets lean_mlp7's reads: ID, NR, MASK, the held rows, and the edge ranking. The al arms also get
these views:
  ID-A0, ID-AS     the ID read with a off, or with each row's per-type values rotated by one over the row's walk types
                   (AS keeps the values and breaks which type gets which)
  FNR-A0           every label OTHER and a off: the untyped graph
  MASK-A0, MASK-AS (a held graph) the same views under MASK
The rows file also gets a model-free edge ranking: per row, the AUC of a(i, z) over the row's tokens against on-path,
both over all tokens and over typed tokens only. An al arm is paired with its twin (aw0 or aw) whenever the twin is fitted
in the same job.

Every run sets torch.use_deterministic_algorithms(True), so a fit repeats bit for bit at any --threads. lean_mlp7's own
runs do not: two runs of one arm differ by more than the row bootstrap allows. One seed's arm-to-arm difference is
therefore read beside the spread over --seed values, not beside the row bootstrap alone.

    python outputs/mp_unified/lean_host12.py lean_mlp12 --store pca256 --train metaqa=fit --select metaqa=select \\
        --read metaqa=x1f,webqsp=selectf --hold metaqa:mask:has_genre --code-from 2wiki --arms aw0,aw0al --threads 2 \\
        --save-models outputs/mp_unified/lean/l12-mq-genre-a_models.pt --out outputs/mp_unified/lean/l12-mq-genre-a.json
    python outputs/mp_unified/lean_mlp12.py --selftest
"""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import scipy.sparse as sparse  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import lean_mlp7g as LG  # noqa: E402
import reltype11 as RT  # noqa: E402  (S3's EM and pairs, unchanged; its BLAS defaults are setdefault, after numpy)

L7 = LG.L7
LM, L2, L3, L4, L5 = L7.LM, L7.L2, L7.L3, L7.L4, L7.L5
K_LAB, N_DIR = L7.K_LAB, L7.N_DIR
KZ = 3 * (K_LAB + 1)
CLIP = 10.0
NEW_ARMS = ("aw0", "aw0al", "awal")
AL_ARMS = ("aw0al", "awal")
BLIND_ARMS = ("aw0", "aw0al")
TWIN = {"aw0al": "aw0", "awal": "aw"}
POP_KEYS = ("q_pool_size", "q_edges", "q_emb", "pool", "proj", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "q_seed_local",
            "q_seed_bucket")
AL_OPTS = ("--al-kappa", "--al-tau", "--al-iters", "--al-graphs")
AL = {"needed": False, "kappa": 20.0, "tau": 0.0, "iters": RT.EM_ITERS, "graphs": "kb", "log": [], "lam": {}}
PJ = {}
log = L7.log


# ── the population pass ──────────────────────────────────────────────────────


def projection():
    if "R" not in PJ:
        PJ["R"] = LM.projection()
    return PJ["R"]


def qhat(X):
    """q_i = n(n(q_emb_i) R) for each row, (rows, 128) float32."""
    return RT.unit(RT.unit(np.asarray(X, dtype=np.float32)) @ projection())


def chunk_rows12(d, limit):
    """lean_mlp4.chunk_rows4's rows in its order (listed chunks, sorted, up to limit), each as one row's arrays, with the
    pool's stored projections and the query's embedding."""
    recs = sorted(Path(d).glob("record*.json"))
    if not recs:
        raise SystemExit(f"{d}: no look record")
    rec0 = json.loads(recs[0].read_text(encoding="utf-8"))
    if rec0.get("proj") != {"dim": LM.PROJ_DIM, "seed": LM.PROJ_SEED}:
        raise SystemExit(f"{d}: projection {rec0.get('proj')}, not lean_mlp's")
    listed = set()
    for r_ in recs:
        listed |= set(json.loads(r_.read_text(encoding="utf-8"))["chunks"])
    rows = 0
    for ch in sorted((Path(d) / "chunks").glob("c*.npz")):
        if int(ch.stem[1:]) not in listed:
            continue
        with np.load(ch) as z:
            A = {k: z[k] for k in POP_KEYS}
        no = np.concatenate([[0], np.cumsum(A["q_pool_size"])])
        eo = np.concatenate([[0], np.cumsum(A["q_edges"])])
        for i in range(A["q_pool_size"].size):
            if limit is not None and rows >= limit:
                return
            a, b, ea, eb = no[i], no[i + 1], eo[i], eo[i + 1]
            yield {"n": int(A["q_pool_size"][i]), "pool": A["pool"][a:b], "proj": A["proj"][a:b], "q_emb": A["q_emb"][i],
                   "e_u": A["e_u"][ea:eb], "e_v": A["e_v"][ea:eb], "e_fam": A["e_fam"][ea:eb], "e_fwd": A["e_fwd"][ea:eb],
                   "e_bwd": A["e_bwd"][ea:eb], "seeds": A["q_seed_local"][i], "buckets": A["q_seed_bucket"][i]}
            rows += 1


def row_types(r, labels, hold_train=None):
    """A row's typed structural edges: local (u, v), famdir d and label rank as lean_mlp7.walk_entries7 reads them (the
    rank in the direction the edge is read; a miss reads OTHER), with hold_train's ranks read OTHER. Only edges with a
    rank below K_LAB are kept; their global ends come too."""
    efam = np.asarray(r["e_fam"])
    st = efam == 0
    u, v = np.asarray(r["e_u"])[st].astype(np.int64), np.asarray(r["e_v"])[st].astype(np.int64)
    fwd, bwd = np.asarray(r["e_fwd"])[st].astype(bool), np.asarray(r["e_bwd"])[st].astype(bool)
    d = L4.famdir(efam[st], fwd, bwd).astype(np.int64)
    pool = np.asarray(r["pool"]).astype(np.int64)
    gu, gv = pool[u], pool[v]
    if gu.size:
        lf, lb = labels.lookup(gu, gv), labels.lookup(gv, gu)
        a = np.where(d == 1, lb, lf)
        lab = np.where(a >= 0, a, K_LAB).astype(np.int64)
    else:
        lab = np.zeros(0, np.int64)
    if hold_train is not None and lab.size:
        lab = np.where(np.isin(lab, hold_train), K_LAB, lab)
    keep = lab < K_LAB
    return u[keep], v[keep], d[keep], lab[keep], gu[keep], gv[keep]


def seed_levels(n, seeds, buckets, u, v, z):
    """The types of the typed edges out of the bucket-0 seeds to a node outside them, and out of all seeds to a node
    outside them (qd_gnn15.seed_edges' two levels; the fit picks the first that is not empty)."""
    s, b = np.asarray(seeds), np.asarray(buckets)
    ok = s >= 0
    out = []
    for S in (s[ok & (b == 0)], s[ok]):
        if S.size == 0:
            out.append(np.zeros(0, np.int64))
            continue
        ins = np.zeros(n, dtype=bool)
        ins[S] = True
        out.append(z[ins[u] & ~ins[v]])
    return out


def merge_nodes(ids0, P0, bi, bp):
    """qd_gnn15.merge_nodes: (sorted unique ids, the first row of each, the largest gap between a later duplicate and
    its first)."""
    ids = np.concatenate([ids0] + bi)
    P = np.concatenate([P0] + bp)
    order = np.argsort(ids, kind="stable")
    ids, P = ids[order], P[order]
    first = np.r_[True, ids[1:] != ids[:-1]] if ids.size else np.zeros(0, dtype=bool)
    grp = np.cumsum(first) - 1
    gap = float(np.abs(P.astype(np.float32) - P[first][grp].astype(np.float32)).max()) if P.size else 0.0
    return ids[first], P[first], gap


def pop_pass(rows, labels, hold_train, NN, n_check=None):
    """One pass over a carve's rows (as chunk_rows12 yields them): the distinct typed edge keys (gu NN + gv) KZ + z, the
    far endpoints' n(p_v) by global id, each row's candidate types at the two seed levels, and q_i."""
    keys, blk = np.zeros(0, np.int64), []
    ids, P, bi, bp, gap = np.zeros(0, np.int64), np.zeros((0, LM.PROJ_DIM), np.float16), [], [], 0.0
    c0, c1, Q = [], [], []
    k = 0
    for r in rows:
        n = int(r["n"])
        if n_check is not None and n != int(n_check[k]):
            raise SystemExit(f"row {k}: pool {n} in the chunk, {int(n_check[k])} in the carve")
        u, v, d, lab, gu, gv = row_types(r, labels, hold_train)
        z = d * (K_LAB + 1) + lab
        if u.size:
            blk.append((gu * NN + gv) * KZ + z)
            vv = np.unique(v)
            bi.append(np.asarray(r["pool"])[vv].astype(np.int64))
            bp.append(np.asarray(r["proj"])[vv].astype(np.float16))
        lv = seed_levels(n, r["seeds"], r["buckets"], u, v, z)
        c0.append(lv[0])
        c1.append(lv[1])
        Q.append(np.asarray(r["q_emb"], dtype=np.float32))
        k += 1
        if len(blk) >= 256:
            keys = np.unique(np.concatenate([keys] + blk))
            blk = []
        if len(bi) >= 64:
            ids, P, g_ = merge_nodes(ids, P, bi, bp)
            gap, bi, bp = max(gap, g_), [], []
    if blk:
        keys = np.unique(np.concatenate([keys] + blk))
    if bi:
        ids, P, g_ = merge_nodes(ids, P, bi, bp)
        gap = max(gap, g_)
    if gap > 1e-3:
        raise SystemExit(f"one node id carries two projections (gap {gap})")
    return {"keys": keys, "ids": ids, "Pn": RT.unit(P.astype(np.float32)), "c0": c0, "c1": c1, "rows": k, "proj_gap": gap,
            "Qh": qhat(np.stack(Q)) if Q else np.zeros((0, LM.PROJ_DIM), np.float32)}


# ── the population fit ───────────────────────────────────────────────────────


def em_tau(Qh, pr, Kz, kappa, init, tau, seed=RT.SEED, iters=RT.EM_ITERS):
    """qd_gnn15.em_tau (copied): reltype11.em with the M step anchored to the start, mu_z = n(sum_i r_iz q_i + tau d_z);
    tau 0 is reltype11.em."""
    rng = np.random.default_rng(seed)
    mu = np.array(init, dtype=np.float32, copy=True)
    dead = np.linalg.norm(mu, axis=1) < 1e-6
    mu[dead] = RT.unit(rng.standard_normal((int(dead.sum()), Qh.shape[1])))
    anchor = mu.copy()
    rows_n = pr["starts"].size
    pi = np.full(Kz, 1.0 / Kz)
    Qr = Qh[pr["row"]]
    trace = []
    for _ in range(iters):
        logit = np.log(pi[pr["z"]] + 1e-12) + kappa * np.einsum("kd,kd->k", Qr, mu[pr["z"]])
        mx = np.maximum.reduceat(logit, pr["starts"])
        e = np.exp(logit - mx[pr["grp"]])
        s = np.add.reduceat(e, pr["starts"])
        r = e / s[pr["grp"]]
        trace.append(round(float((np.log(s) + mx).mean()), 5))
        W = sparse.csr_matrix((r.astype(np.float32), (pr["z"], pr["grp"])), shape=(Kz, rows_n))
        M = W @ Qh[pr["row"][pr["starts"]]]
        mass = np.bincount(pr["z"], weights=r, minlength=Kz)
        live = mass > 1e-9
        mu[live] = RT.unit(M[live] + tau * anchor[live])
        pi = mass / rows_n
    p_ = pi[pi > 0]
    return mu, pi, {"loglik": trace[::5] + [trace[-1]], "pi_perplexity": round(float(np.exp(-(p_ * np.log(p_)).sum())), 2)}


def align_rows(Qh, mu, logpi, kappa):
    """a(i, z) for every row and present type: clip(s_iz - logsumexp_z' s_iz' + log T, +-10), s = log pi~ + kappa cos."""
    S = logpi[None, :] + kappa * (np.asarray(Qh, dtype=np.float64) @ np.asarray(mu, dtype=np.float32).astype(np.float64).T)
    mx = S.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(S - mx).sum(1))
    return np.clip(S - lse[:, None] + math.log(S.shape[1]), -CLIP, CLIP).astype(np.float32)


def type_namer(labels):
    """A type's name for the log only (a KB's relation name by rank; else the rank): the fit never reads it."""
    voc, rank_of = getattr(labels, "vocab", None), getattr(labels, "rank_of", None)
    by = None if voc is None or rank_of is None else np.argsort(rank_of)

    def name(zk):
        d, lab = int(zk) // (K_LAB + 1), int(zk) % (K_LAB + 1)
        nm = voc[int(by[lab])] if by is not None and lab < by.size else f"rank {lab}"
        return f"{nm}/{('fwd', 'bwd', 'both')[d]}"

    return name


def fit_al(pp, NN, excl=None, tag="", namer=None):
    """The alignment fit of one population (the module docstring): its present types, pi~, mu and a(i, z) for every
    row; excl's ranks are untyped (MASK). None when the population has no typed edge."""
    t0 = time.time()
    keys = pp["keys"]
    ex = None if excl is None else np.asarray(excl, dtype=np.int64)
    if ex is not None and keys.size:
        keys = keys[~np.isin(keys % KZ % (K_LAB + 1), ex)]
    if keys.size == 0:
        AL["log"].append({"tag": tag, "rows": pp["rows"], "types": 0})
        log(f"al fit {tag}: no typed edge; a = 0")
        return None
    z_all = keys % KZ
    Zpop = np.unique(z_all)
    T = int(Zpop.size)
    zmap = np.full(KZ, -1, dtype=np.int64)
    zmap[Zpop] = np.arange(T)
    gv = (keys // KZ) % NN
    j = np.searchsorted(pp["ids"], gv)
    if not (j < pp["ids"].size).all() or not np.array_equal(pp["ids"][np.minimum(j, pp["ids"].size - 1)], gv):
        raise SystemExit(f"{tag}: a far endpoint is missing from the node table")
    D = RT.unit(np.asarray(sparse.csr_matrix((np.ones(keys.size, dtype=np.float32), (zmap[z_all], j)),
                                             shape=(T, pp["ids"].size)) @ pp["Pn"]))
    cr, cz = [], []
    for pos in range(pp["rows"]):
        c = pp["c0"][pos]
        if ex is not None and c.size:
            c = c[~np.isin(c % (K_LAB + 1), ex)]
        if c.size == 0:
            c = pp["c1"][pos]
            if ex is not None and c.size:
                c = c[~np.isin(c % (K_LAB + 1), ex)]
        if c.size:
            cr.append(np.full(c.size, pos, dtype=np.int64))
            cz.append(zmap[c])
    cand_row = np.concatenate(cr) if cr else np.zeros(0, np.int64)
    cand_col = np.concatenate(cz) if cz else np.zeros(0, np.int64)
    if (cand_col < 0).any():
        raise SystemExit(f"{tag}: a candidate type outside the population")
    rows_n = 0
    if cand_row.size:
        pr = RT.pairs(cand_row, cand_col, np.zeros(cand_row.size, dtype=bool), T)
        rows_n = int(pr["starts"].size)
    Qh = pp["Qh"]
    if rows_n == 0:
        mu, pi, info = D.copy(), np.zeros(T), {"loglik": [], "pi_perplexity": None}
    elif AL["tau"] > 0:
        mu, pi, info = em_tau(Qh, pr, T, AL["kappa"], D, AL["tau"], seed=RT.SEED, iters=AL["iters"])
    else:
        mu, pi, info = RT.em(Qh, pr, T, AL["kappa"], D, seed=RT.SEED, iters=AL["iters"])
    mass = pi * rows_n
    pit = (mass + 1.0) / (rows_n + T)
    logpi = np.log(pit)
    mu = np.asarray(mu, dtype=np.float32)
    A = align_rows(Qh, mu, logpi, AL["kappa"])
    top = np.argsort(-pit, kind="stable")[:12]
    nm = namer or (lambda zk: str(int(zk)))
    rec = {"tag": tag, "rows": pp["rows"], "rows_with_candidates": rows_n, "types": T, "edges": int(keys.size),
           "kappa": AL["kappa"], "tau": AL["tau"], "pi_perplexity": info["pi_perplexity"], "loglik": info["loglik"],
           "a_mean": float(A.mean()), "a_sd": float(A.std()), "a_clipped": float((np.abs(A) >= CLIP).mean()),
           "top": [[nm(Zpop[k]), round(float(pit[k]), 4), round(float(mass[k]), 1)] for k in top],
           "seconds": round(time.time() - t0, 1)}
    AL["log"].append(rec)
    log(f"al fit {tag}: {pp['rows']} rows ({rows_n} with seed edges), {T} types, {keys.size} edges, pi perplexity "
        f"{info['pi_perplexity']}, a mean {rec['a_mean']:.3f} sd {rec['a_sd']:.3f}, top {rec['top'][:4]} ({rec['seconds']}s)")
    return {"zmap": zmap, "Zpop": Zpop, "T": T, "mu": mu, "D": D, "logpi": logpi, "A": A, "cand_row": cand_row,
            "cand_col": cand_col, "rec": rec}


def al_cols(fit, d, lab):
    """Each edge's column in the fit (-1: untyped, or a type outside the fit)."""
    d, lab = np.asarray(d, dtype=np.int64), np.asarray(lab, dtype=np.int64)
    typed = (d <= 2) & (lab < K_LAB)
    c = fit["zmap"][np.where(typed, d * (K_LAB + 1) + lab, 0)]
    return np.where(typed, c, -1)


class Carve12(L7.Carve7):
    """lean_mlp7's carve plus the population fits of its rows (a KB's carve, when an al arm is fitted or loaded): rev
    (every typed edge; a held KB's training and select carves already read the held rank OTHER) and, on a held KB's
    read carve, mask (the held rank untyped). al_mode picks on, off (a = 0) or shuf; aw_mask picks the mask fit."""

    def __init__(self, ds, carve, store=None, nodes=None, limit=None, root=LM.LOOK, labels=None, basis=None,
                 struct_only=False, hold=None, hold_mode=None):
        super().__init__(ds, carve, store, nodes, limit, root, labels, basis, struct_only, hold, hold_mode)
        self.al_mode = "on"
        self.al_fits = {"rev": None, "mask": None}
        self.al_info = None
        if AL["needed"] and labels is not None and labels.ok and (AL["graphs"] == "all" or ds in L7.KBS):
            self.fit_al(root, limit, labels)

    def fit_al(self, root, limit, labels):
        t0 = time.time()
        NN = int(labels.n_nodes)
        if NN * NN * KZ >= 2 ** 63:
            raise SystemExit(f"{self.ds}: {NN} nodes overflow the edge key")
        hold_train = self.hold if self.hold_mode == "train" else None
        pp = pop_pass(chunk_rows12(Path(root) / self.ds / self.carve, limit), labels, hold_train, NN, self.n)
        if pp["rows"] != self.rows:
            raise SystemExit(f"{self.ds}={self.carve}: {pp['rows']} rows in the pass, {self.rows} in the carve")
        namer = type_namer(labels)
        tag = f"{self.ds}={self.carve}"
        self.al_fits["rev"] = fit_al(pp, NN, None, tag + ("/REV" if self.hold_mode == "read" else ""), namer)
        if self.hold_mode == "read":
            self.al_fits["mask"] = fit_al(pp, NN, self.hold, tag + "/MASK", namer)
        self.al_info = {"rows": pp["rows"], "far_nodes": int(pp["ids"].size), "proj_gap": pp["proj_gap"],
                        "fits": {k: (None if f is None else f["rec"]) for k, f in self.al_fits.items()},
                        "walk_check": self.al_check(), "seconds": round(time.time() - t0, 1)}
        del pp

    def al_check(self):
        """Every typed walk edge and token maps into the rev fit, and every one but a held rank's into the mask fit."""
        out = {}
        for nm, fit in self.al_fits.items():
            if fit is None:
                continue
            bad, typed = 0, 0
            for d, lab in ((self.aw1["dir"], self.aw1["lab"]), (self.aw2["d1"], self.aw2["l1"]), (self.aw2["d2"], self.aw2["l2"]),
                           (self.tok["d"], self.tok["lab"])):
                t = (d <= 2) & (lab < K_LAB)
                if nm == "mask" and self.hold is not None:
                    t = t & ~np.isin(lab, self.hold)
                typed += int(t.sum())
                bad += int((t & (al_cols(fit, d, lab) < 0)).sum())
            if bad:
                raise SystemExit(f"{self.ds}={self.carve}: {bad} of {typed} typed walk edges / tokens are outside the {nm} fit")
            out[nm] = {"typed": typed}
        return out


# ── batches ──────────────────────────────────────────────────────────────────


def al_batch(carve, qs):
    """a for the batch's one-hop edges (al1), two-hop first and second edges (al2a, al2b) and tokens (tal), from the
    carve's fit for the view (mask under aw_mask, else rev) and its al_mode; None when the view has no fit."""
    fits = getattr(carve, "al_fits", None)
    if not fits:
        return None
    fit = fits.get("mask") if getattr(carve, "aw_mask", False) else fits.get("rev")
    if fit is None:
        return None
    mode = getattr(carve, "al_mode", "on")
    if mode not in ("on", "off", "shuf"):
        raise ValueError(mode)
    sz1 = carve.aw_off1[qs + 1] - carve.aw_off1[qs]
    sz2 = carve.aw_off2[qs + 1] - carve.aw_off2[qs]
    szt = carve.tok_off[qs + 1] - carve.tok_off[qs]
    i1 = np.concatenate([np.arange(carve.aw_off1[q], carve.aw_off1[q + 1]) for q in qs])
    i2 = np.concatenate([np.arange(carve.aw_off2[q], carve.aw_off2[q + 1]) for q in qs])
    it = np.concatenate([np.arange(carve.tok_off[q], carve.tok_off[q + 1]) for q in qs])
    t = torch.from_numpy
    if mode == "off":
        return {"al1": torch.zeros(i1.size), "al2a": torch.zeros(i2.size), "al2b": torch.zeros(i2.size),
                "tal": torch.zeros(it.size)}
    q1, q2, qt = (np.repeat(np.arange(len(qs)), s) for s in (sz1, sz2, szt))
    c1 = al_cols(fit, carve.aw1["dir"][i1], carve.aw1["lab"][i1])
    c2a = al_cols(fit, carve.aw2["d1"][i2], carve.aw2["l1"][i2])
    c2b = al_cols(fit, carve.aw2["d2"][i2], carve.aw2["l2"][i2])
    ct = al_cols(fit, carve.tok["d"][it], carve.tok["lab"][it])
    Ab = fit["A"][qs]
    if mode == "shuf":
        Ab = Ab.copy()
        o1 = np.concatenate([[0], np.cumsum(sz1)])
        o2 = np.concatenate([[0], np.cumsum(sz2)])
        for b in range(len(qs)):
            u = np.unique(np.concatenate([c1[o1[b]:o1[b + 1]], c2a[o2[b]:o2[b + 1]], c2b[o2[b]:o2[b + 1]]]))
            u = u[u >= 0]
            if u.size > 1:
                Ab[b, u] = np.roll(fit["A"][qs[b], u], 1)

    def val(c, qb):
        return t(np.where(c >= 0, Ab[qb, np.maximum(c, 0)], 0.0).astype(np.float32))

    return {"al1": val(c1, q1), "al2a": val(c2a, q2), "al2b": val(c2b, q2), "tal": val(ct, qt)}


def aw_batch12(carve, qs):
    """lean_mlp7.aw_batch7's tensors plus a (al1, al2a, al2b, tal) when the carve has a fit for the view."""
    out = L7.aw_batch7(carve, qs)
    al = al_batch(carve, np.asarray(qs))
    if al is not None:
        out.update(al)
    return out


def batch_of12(carve, qs, blocks):
    feats, nq, base_z, gold, idx = L7._BATCH_OF(carve, qs, [b for b in blocks if b != "AW"])
    if "AW" in blocks:
        feats["AW"] = aw_batch12(carve, qs)
    return feats, nq, base_z, gold, idx


# ── the model ────────────────────────────────────────────────────────────────


class AWNet12(L7.AWNet7):
    """lean_mlp7's AWNet7. The new arms are built as arm aw (the same parameters from the same generator). aw0 and aw0al
    read every walk label as OTHER after the dropout draws; aw0al and awal add lam[d] a to each walk edge's logit. An old
    arm runs AWNet7's forward itself."""

    def __init__(self, r, D=64, k_res=0, p_drop=0.25, seed=0, arm="aw", k_types=8):
        super().__init__(r, D, k_res, p_drop, seed, "aw" if arm in NEW_ARMS else arm, k_types)
        self.arm = arm
        self.blind = arm in BLIND_ARMS
        self.lam = nn.Parameter(torch.zeros(N_DIR)) if arm in AL_ARMS else None

    def logit12(self, phi, qi, d, lab, cos, Z, r, al):
        lg = self.logit7(phi, qi, d, lab, cos, Z, r)
        if self.lam is not None and al is not None:
            lg = lg + self.lam[d] * al
        return lg

    def forward(self, A):
        if self.lam is None and not self.blind:
            return super().forward(A)
        phi = A["qe"] @ self.Wq
        Z, N = A["Z"], A["N"]
        l1, l2a, l2b = A["l1"], A["l2a"], A["l2b"]
        if self.training and self.p_drop > 0:
            l1 = torch.where(torch.rand(l1.shape) < self.p_drop, torch.full_like(l1, K_LAB), l1)
            l2a = torch.where(torch.rand(l2a.shape) < self.p_drop, torch.full_like(l2a, K_LAB), l2a)
            l2b = torch.where(torch.rand(l2b.shape) < self.p_drop, torch.full_like(l2b, K_LAB), l2b)
        if self.blind:
            l1, l2a, l2b = (torch.full_like(x, K_LAB) for x in (l1, l2a, l2b))
        parts = [(A["q1"], A["d1"], l1, A.get("c1")), (A["q2"], A["d2a"], l2a, A.get("c2a")), (A["q2"], A["d2b"], l2b, A.get("c2b"))]
        rs = self.typer_r(A["qe"], Z, parts) if self.typer is not None else [None, None, None]
        als = [A.get(k) for k in ("al1", "al2a", "al2b")]
        la1, la2, lb2 = (self.logit12(phi, qi, d, lab, c, Z, r, al) for (qi, d, lab, c), r, al in zip(parts, rs, als))
        cols = []
        for node, la, lb, bk in ((A["n1"], la1, None, A["b1"]), (A["n2"], la2, lb2, A["b2"])):
            g = torch.sigmoid(la) if lb is None else torch.sigmoid(la) * torch.sigmoid(lb)
            sc = la if lb is None else la + lb
            s = torch.zeros(N).index_add(0, node, g)
            mx = torch.full((N,), float("-inf")).scatter_reduce(0, node, sc, reduce="amax", include_self=True)
            m0 = bk == 0
            s0 = torch.zeros(N).index_add(0, node[m0], g[m0])
            cols += [torch.log1p(s), torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx)), torch.log1p(s0)]
        return torch.stack(cols, 1)

    def unit_logit(self, A, lab):
        if self.blind:
            lab = torch.full_like(lab, K_LAB)
        lg = super().unit_logit(A, lab)
        if self.lam is not None and A.get("tal") is not None:
            lg = lg + self.lam[A["td"]] * A["tal"]
        return lg


class LeanMLP12(L7.LeanMLP7):
    """lean_mlp7's model with an AWNet12 of its arm (LeanMLP7's own construction otherwise, in the same order)."""

    cfg = dict(L7.LeanMLP7.cfg)

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="aw"):
        L3.LeanMLP3.__init__(self, blocks, widths, hidden, dropout, seed)
        self.arm = arm
        if "AW" in self.blocks:
            c = type(self).cfg
            self.awn = AWNet12(c["r"], c["D"], c["k_res"], c["p_drop"], seed, arm, c.get("k_types", 8))


def is_al(m):
    return getattr(getattr(m, "awn", None), "lam", None) is not None


def fit12(train, select, blocks, arm, cfg, seed, hidden, emc=None, aux_omega=1.0):
    """lean_mlp7g.fit7g unchanged; an al arm's lam is read once an epoch (at its select read) into its curve."""
    trace = []
    q0 = LM.quality

    def q_hook(model, carves, blocks_, keep_map):
        if is_al(model):
            trace.append([round(float(v), 4) for v in model.awn.lam.detach()])
        return q0(model, carves, blocks_, keep_map)

    LM.quality = q_hook
    try:
        f = LG.fit7g(train, select, blocks, arm, cfg, seed, hidden, emc, aux_omega)
    finally:
        LM.quality = q0
    if trace:
        for rec, lam in zip(f["curve"], trace):
            rec["lam"] = lam
        AL["lam"][arm] = {"per_epoch": trace, **{k: [round(float(v), 4) for v in f[k]["awn.lam"]] for k in ("best", "swa", "last")}}
        log(f"  {arm}: lam per epoch {trace}; best {AL['lam'][arm]['best']} swa {AL['lam'][arm]['swa']}")
    return f


# ── reads ────────────────────────────────────────────────────────────────────


@torch.no_grad()
def edge_diag12(model, c):
    """lean_mlp7.edge_diag for an old arm. For a new arm: the AUC of its token logit against on-path, per row, over
    aw_batch12's tokens (an al arm's logit with its own a, and with a off: aw_a0)."""
    aw = model.awn
    if not is_al(model) and not getattr(aw, "blind", False):
        return L7.edge_diag(model, c)
    model.eval()
    keys = ["aw"] + (["aw_a0"] if is_al(model) else [])
    sc = {k: [] for k in keys}
    for k in range(0, c.rows, 64):
        qs = np.arange(k, min(k + 64, c.rows))
        A = aw_batch12(c, qs)
        sc["aw"].append(aw.unit_logit(A, A["tl"]).numpy())
        if "aw_a0" in sc:
            A0 = dict(A)
            A0["tal"] = torch.zeros_like(A["tl"], dtype=torch.float32)
            sc["aw_a0"].append(aw.unit_logit(A0, A["tl"]).numpy())
    return auc_summary(sc, c)


def auc_summary(sc, c, typed=None):
    o = c.tok["o"]
    held = getattr(c, "held", None)
    out = {"tokens_mean": float(np.diff(c.tok_off).mean()) if c.rows else None, "on_share": float(o.mean()) if o.size else None}
    for key, parts in sc.items():
        s = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        if typed is None:
            v, ids = L7.row_auc(s, o, c.tok_off)
        else:
            cnt = np.bincount(np.repeat(np.arange(c.rows), np.diff(c.tok_off))[typed], minlength=c.rows)
            v, ids = L7.row_auc(s[typed], o[typed], np.concatenate([[0], np.cumsum(cnt)]))
        out[f"auc_{key}"] = float(v.mean()) if v.size else None
        out[f"rows_both_{key}"] = int(v.size)
        if held is not None:
            h = held[ids]
            out[f"auc_{key}_held"] = float(v[h].mean()) if h.any() else None
            out[f"rows_both_{key}_held"] = int(h.sum())
    return out


def al_diag(c):
    """The alignment's own edge ranking (model-free): per row, the AUC of the tokens' a against on-path, over all tokens
    (untyped at 0) and over the typed tokens alone, for the current view's fit."""
    fit = c.al_fits.get("mask") if c.aw_mask else c.al_fits.get("rev")
    if fit is None:
        return None
    vals = []
    for k in range(0, c.rows, 256):
        qs = np.arange(k, min(k + 256, c.rows))
        vals.append(al_batch(c, qs)["tal"].numpy())
    s = np.concatenate(vals) if vals else np.zeros(0, np.float32)
    typed = al_cols(fit, c.tok["d"], c.tok["lab"]) >= 0
    out = auc_summary({"al": [s]}, c)
    out.update({k.replace("_al", "_al_typed"): v for k, v in auc_summary({"al": [s]}, c, typed).items() if "_al" in k})
    out["typed_share"] = float(typed.mean()) if typed.size else None
    return out


def held_summary12(models, view_of, refs, hm, rng):
    """lean_mlp7.held_summary's entries (REV = the ID read, MASK, against the twin, each other and ctl) plus, for al
    arms, REV-A0, REV-AS, MASK-A0, MASK-AS, NR and FNR-A0 on the held rows, their paired differences, and each al arm
    against its twin arm."""
    out = {"rows": int(hm.sum())}
    if hm.sum() < 2:
        return out
    tw0, g = refs["twin0"][hm], refs["gnn0"][hm]
    t, gm = tw0.mean(0), g.mean(0)
    out["twin0"], out["gnn0"] = t.tolist(), gm.tolist()

    def rho(x):
        return [float((x[j] - t[j]) / (gm[j] - t[j])) if abs(gm[j] - t[j]) > 1e-9 else None for j in range(3)]

    for name in models:
        v = view_of[name]
        e = {}
        for tag, rows in v.items():
            y = rows[hm]
            nm = "REV" if tag == "ID" else tag.replace("ID-", "REV-")
            e[nm] = y.mean(0).tolist()
            e[f"rho_{nm}"] = rho(y.mean(0))
            e[f"{nm}_minus_twin0"] = LM.boot_diff(y, tw0, rng)
        for a_, b_ in (("ID", "MASK"), ("ID", "ID-A0"), ("ID", "ID-AS"), ("MASK", "MASK-A0"), ("MASK", "MASK-AS")):
            if a_ in v and b_ in v:
                na, nb = ("REV" if a_ == "ID" else a_), b_.replace("ID-", "REV-")
                e[f"{na}_minus_{nb}"] = LM.boot_diff(v[a_][hm], v[b_][hm], rng)
        arm, kind = name.split("@")
        ctl = f"ctl@{kind}"
        if ctl in view_of and ctl != name:
            e["REV_minus_ctl"] = LM.boot_diff(v["ID"][hm], view_of[ctl]["ID"][hm], rng)
        tw = f"{TWIN.get(arm)}@{kind}"
        if tw in view_of:
            for tag in ("ID", "MASK"):
                if tag in v and tag in view_of[tw]:
                    e[f"{'REV' if tag == 'ID' else tag}_minus_{TWIN[arm]}"] = LM.boot_diff(v[tag][hm], view_of[tw][tag][hm], rng)
        out[name] = e
    return out


def read12(a, models, out, carve):
    """lean_mlp7.read7's reads (ID, NR, MASK, the held rows, the edge ranking, the rows file) plus the al arms' views,
    their twin pairs and the alignment's own edge ranking."""
    rng = np.random.default_rng(20261003)
    reads, read_costs, read_lean_ms, read_rows, aw_stats, edge, held, al_info = {}, {}, {}, {}, {}, {}, {}, {}
    store = {}
    for ds, cv in LM.parse_sets(a.read):
        key = f"{ds}={cv}"
        c = carve(ds, cv, "read")
        read_rows[key] = {"rows": c.rows, "chunks": c.chunks_read, "n_chunks": c.n_chunks, "carve_queries": c.carve_queries}
        aw_stats[key] = c.aw_stats
        al_info[key] = c.al_info
        (prof, share, lm), = L2.costs_of([c])
        read_lean_ms[key] = c.lean_ms
        read_costs[key] = {name: LM.cost_ms(bl, prof, share, lm) for name, (_m, bl, _k) in models.items()}
        log(f"read {key} ({c.rows} rows); approximate compile ms {({k: round(v, 2) for k, v in read_costs[key].items()})}")
        okm = c.gold_total > 0
        refs = {nm: LM.row_metrics(c.score[:, col], c.gold, c.off, c.gold_total)[okm] for nm, col in LM.SCORE_COL.items()}
        for nm, v in refs.items():
            store[f"{nm}@{key}"] = v.astype(np.float32)
        view_of = {}
        for name, (m, bl, keep) in models.items():
            r, rows = LM.read_one(m, c, bl, keep, rng)
            view_of[name] = {"ID": rows}
            store[f"{name}@{key}"] = rows.astype(np.float32)
            reads[f"{name}@{key}"] = r
        for name in models:
            arm, kind = name.split("@")
            r = reads[f"{name}@{key}"]
            mine = view_of[name]["ID"]
            for ref_arm in ("ctl", "awc"):
                ref = f"{ref_arm}@{kind}"
                if ref in view_of and ref != name and (ref_arm == "ctl" or arm in ("aux", "em", "em0")):
                    r[f"minus_{ref_arm}"] = LM.boot_diff(mine, view_of[ref]["ID"], rng)
            tw = TWIN.get(arm)
            if tw is not None and f"{tw}@{kind}" in view_of:
                r[f"minus_{tw}"] = LM.boot_diff(mine, view_of[f"{tw}@{kind}"]["ID"], rng)
            extra = "".join(f" vs {k[6:]} {L7.pts(r[k])}" for k in r if k.startswith("minus_") and k[6:] in ("ctl", "awc", "aw", "aw0"))
            log(f"  {name}@{key}: {LM.fmt(r)}{extra}")
        aw_models = [name for name, (_m, bl, _k) in models.items() if "AW" in bl]
        al_models = [name for name in aw_models if is_al(models[name][0])]
        has_fit = c.al_fits.get("rev") is not None

        def run_view(tag, names, nr=False, mask=False, mode="on"):
            c.aw_nr, c.aw_mask, c.al_mode = nr, mask, mode
            try:
                for name in names:
                    m, bl, keep = models[name]
                    r, rows = LM.read_one(m, c, bl, keep, rng, view_of[name]["ID"])
                    reads[f"{name}@{key}/{tag}"] = r
                    view_of[name][tag] = rows
                    store[f"{name}@{key}~{tag}"] = rows.astype(np.float32)
                    log(f"  {name}@{key}/{tag}: {LM.fmt(r)} (vs full = vs its own ID read)")
            finally:
                c.aw_nr, c.aw_mask, c.al_mode = False, False, "on"

        run_view("NR", aw_models, nr=True)
        if al_models and has_fit:
            run_view("ID-A0", al_models, mode="off")
            run_view("ID-AS", al_models, mode="shuf")
            run_view("FNR-A0", al_models, nr=True, mode="off")
        if c.hold_mode == "read":
            run_view("MASK", aw_models, mask=True)
            if al_models and c.al_fits.get("mask") is not None:
                run_view("MASK-A0", al_models, mask=True, mode="off")
                run_view("MASK-AS", al_models, mask=True, mode="shuf")
            hm = c.held[okm]
            store[f"held@{key}"] = hm
            held[key] = held_summary12(models, view_of, refs, hm, rng)
            for name, e in held[key].items():
                if isinstance(e, dict):
                    log(f"  held rows ({held[key]['rows']}) {name}@{key}: REV rho {e.get('rho_REV')} vs twin0 {L7.pts(e['REV_minus_twin0'])}"
                        + "".join(f"; {k} {L7.pts(e[k])}" for k in e if "_minus_" in k and k != "REV_minus_twin0"
                                  and not k.endswith("_minus_twin0")))
        for name in aw_models:
            m = models[name][0]
            edge[f"{name}@{key}"] = edge_diag12(m, c)
            log(f"  edge ranking {name}@{key}: {edge[f'{name}@{key}']}")
            if c.hold_mode == "read":
                c.aw_mask = True
                edge[f"{name}@{key}/MASK"] = edge_diag12(m, c)
                c.aw_mask = False
                log(f"  edge ranking {name}@{key}/MASK: {edge[f'{name}@{key}/MASK']}")
        if has_fit:
            edge[f"al@{key}"] = al_diag(c)
            log(f"  alignment edge ranking (model-free) {key}: {edge[f'al@{key}']}")
            if c.hold_mode == "read" and c.al_fits.get("mask") is not None:
                c.aw_mask = True
                edge[f"al@{key}/MASK"] = al_diag(c)
                c.aw_mask = False
                log(f"  alignment edge ranking (model-free) {key}/MASK: {edge[f'al@{key}/MASK']}")
        del c
    out["read_costs_ms"] = read_costs
    out["read_lean_ms"] = read_lean_ms
    out["read_rows"] = read_rows
    out["read_aw_stats"] = aw_stats
    out["read_al"] = al_info
    out["edge_rank"] = edge
    out["held"] = held
    out["reads"] = reads
    if a.out:
        p = Path(a.out).with_suffix(".rows.npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp.npz")
        np.savez_compressed(tmp, **store)
        os.replace(tmp, p)
        out["rows_file"] = {"path": p.name, "keys": len(store), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        log(f"wrote {p} ({len(store)} arrays)")


# ── main ─────────────────────────────────────────────────────────────────────


def strip_al(argv):
    """argv without the --al-* options, and their values."""
    out, vals, i = [], {}, 0
    while i < len(argv):
        x = argv[i]
        hit = next((k for k in AL_OPTS if x == k or x.startswith(k + "=")), None)
        if hit is None:
            out.append(x)
            i += 1
        elif x == hit:
            if i + 1 >= len(argv):
                raise SystemExit(f"{hit} needs a value")
            vals[hit] = argv[i + 1]
            i += 2
        else:
            vals[hit] = x.split("=", 1)[1]
            i += 1
    return out, vals


def arms_of(argv):
    for i, x in enumerate(argv):
        if x == "--arms" and i + 1 < len(argv):
            return [s for s in argv[i + 1].split(",") if s]
        if x.startswith("--arms="):
            return [s for s in x.split("=", 1)[1].split(",") if s]
    return ["ctl", "aw", "awc"]


def install12():
    LG.install()
    if not set(NEW_ARMS) <= set(L7.ARMS):
        L7.ARMS = tuple(L7.ARMS) + tuple(x for x in NEW_ARMS if x not in L7.ARMS)
    L7.Carve7 = Carve12
    L7.batch_of7 = batch_of12
    L7.LeanMLP7 = LeanMLP12
    L7.fit7 = fit12
    L7.read7 = read12


def shas12():
    return {f"{n}_sha256": hashlib.sha256((HERE / f"{n}.py").read_bytes()).hexdigest()
            for n in ("lean_mlp7", "lean_mlp7g", "reltype11", "lean_mlp12")}


def main():
    # CPU index_put(accumulate=True) on more than one thread adds in a thread-dependent order, so lean_mlp7's AW fits are
    # not reproducible (l7g-j3a's and l7g-j3b's awc differ by up to 0.75 R@5 points); the deterministic path adds serially
    torch.use_deterministic_algorithms(True)
    if "--selftest" in sys.argv:
        return selftest()
    argv, vals = strip_al(list(sys.argv))
    AL["kappa"] = float(vals.get("--al-kappa", AL["kappa"]))
    AL["tau"] = float(vals.get("--al-tau", AL["tau"]))
    AL["iters"] = int(vals.get("--al-iters", AL["iters"]))
    AL["graphs"] = vals.get("--al-graphs", AL["graphs"])
    if AL["graphs"] not in ("kb", "all"):
        raise SystemExit(f"--al-graphs {AL['graphs']}: kb or all")
    AL["needed"] = "--load-models" in argv or any(x in AL_ARMS for x in arms_of(argv))
    sys.argv = argv
    install12()
    log(f"lean_mlp12: al {({k: AL[k] for k in ('needed', 'kappa', 'tau', 'iters', 'graphs')})}")
    L7.main()
    out = None
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    if out is not None and out.exists():
        res = json.loads(out.read_text(encoding="utf-8"))
        res["look12"] = {"file": "outputs/mp_unified/lean_mlp12.py", **shas12(),
                         "deterministic": torch.are_deterministic_algorithms_enabled()}
        res["nan_guard"] = {"file": "outputs/mp_unified/lean_mlp7g.py",
                            "sha256": hashlib.sha256((HERE / "lean_mlp7g.py").read_bytes()).hexdigest()}
        res["al"] = {k: AL[k] for k in ("needed", "kappa", "tau", "iters", "graphs")}
        res["al"]["fits"] = AL["log"]
        res["al"]["lam"] = AL["lam"]
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, out)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


class FakeLab12:
    """Every stored pair labelled by a fixed hash of its ends, a few pairs missing; n_nodes for the edge key."""
    ok = True

    def __init__(self, n_nodes, n_lab=6):
        self.n_nodes, self.n_lab = n_nodes, n_lab

    def lookup(self, gu, gv):
        gu, gv = np.asarray(gu, dtype=np.int64), np.asarray(gv, dtype=np.int64)
        lab = ((gu * 7 + gv * 3) % self.n_lab).astype(np.int32)
        return np.where((gu + gv) % 11 == 0, -1, lab)


def toy_row(rng, NN, G):
    n = int(rng.integers(5, 16))
    pool = rng.choice(NN, n, replace=False).astype(np.int32)
    m = int(rng.integers(4, 60))
    eu, ev = rng.integers(0, n, m), rng.integers(0, n, m)
    ok = eu != ev
    eu, ev = eu[ok].astype(np.int16), ev[ok].astype(np.int16)
    fam = rng.choice([0, 0, 0, 1, 2], eu.size).astype(np.int8)
    fwd = ((rng.uniform(size=eu.size) < 0.6) & (fam == 0)).astype(np.int8)
    bwd = (((rng.uniform(size=eu.size) < 0.6) | (fwd == 0)) & (fam == 0)).astype(np.int8)
    S = rng.choice(n, size=min(n, int(rng.integers(1, 5))), replace=False)
    seeds = np.r_[S, -np.ones(10 - S.size, np.int64)]
    buckets = np.r_[rng.integers(0, 2, S.size), -np.ones(10 - S.size, np.int64)]
    return {"n": n, "pool": pool, "proj": G[pool], "q_emb": rng.standard_normal(1536).astype(np.float32), "e_u": eu, "e_v": ev,
            "e_fam": fam, "e_fwd": fwd, "e_bwd": bwd, "seeds": seeds, "buckets": buckets}


def brute_fit_inputs(rows, labels, hold, excl):
    """The distinct typed (gu, gv, z), each row's candidate types and q_i, by loops."""
    R = LM.projection()
    trip, cands, Q = set(), [], []
    for r in rows:
        n, pool = r["n"], r["pool"].astype(np.int64)
        typed = []
        for j in range(r["e_u"].size):
            if int(r["e_fam"][j]) != 0:
                continue
            u, v = int(r["e_u"][j]), int(r["e_v"][j])
            f, b = bool(r["e_fwd"][j]), bool(r["e_bwd"][j])
            d = 2 if (f and b) else (0 if f else 1)
            lab = int(labels.lookup(np.array([pool[v] if d == 1 else pool[u]]), np.array([pool[u] if d == 1 else pool[v]]))[0])
            if lab < 0 or (hold is not None and lab in hold) or (excl is not None and lab in excl):
                continue
            z = d * (K_LAB + 1) + lab
            typed.append((u, v, z))
            trip.add((int(pool[u]), int(pool[v]), z))
        s, bk = r["seeds"], r["buckets"]
        lv = []
        for S in ([int(x) for x, y in zip(s, bk) if x >= 0 and y == 0], [int(x) for x in s if x >= 0]):
            lv.append(sorted({z for (u, v, z) in typed if u in S and v not in S}) if S else [])
        cands.append(lv[0] if lv[0] else lv[1])
        q = r["q_emb"].astype(np.float32)
        q = q / np.linalg.norm(q)
        q = q @ R
        Q.append(q / np.linalg.norm(q))
    return trip, cands, np.stack(Q)


class ToyCarve12(L7.ToyCarve7):
    """lean_mlp7's toy carve with a planted alignment: each row's walk edges that end at a gold node carry label 0 (a
    random structural direction), no other walk edge does, and the fit gives the label-0 types HI and every other
    type a value in LO."""

    HI, LO = 8.0, (-8.0, -2.0)

    def __init__(self, rows, seed, r=8, n_lab=20, plant=True):
        super().__init__(rows, seed, r, n_lab)
        rng = np.random.default_rng(seed + 300)
        self.carve = f"t{seed}"
        self.hold, self.hold_mode, self.aw_mask, self.held = None, None, False, None
        for i in range(self.rows):
            g = self.gold[self.off[i]:self.off[i + 1]]
            s1, s2 = slice(self.aw_off1[i], self.aw_off1[i + 1]), slice(self.aw_off2[i], self.aw_off2[i + 1])
            on1 = (g[self.aw1["node"][s1]] > 0) if plant else np.zeros(s1.stop - s1.start, bool)
            on2 = (g[self.aw2["node"][s2]] > 0) if plant else np.zeros(s2.stop - s2.start, bool)
            self.aw1["dir"][s1] = np.where(on1, rng.integers(0, 3, on1.size), self.aw1["dir"][s1])
            self.aw1["lab"][s1] = np.where(on1, 0, np.where(self.aw1["lab"][s1] == 0, 1, self.aw1["lab"][s1]))
            self.aw2["l1"][s2] = np.where(self.aw2["l1"][s2] == 0, 1, self.aw2["l1"][s2])
            self.aw2["d2"][s2] = np.where(on2, rng.integers(0, 3, on2.size), self.aw2["d2"][s2])
            self.aw2["l2"][s2] = np.where(on2, 0, np.where(self.aw2["l2"][s2] == 0, 1, self.aw2["l2"][s2]))
        self.al_mode = "on"
        self.al_fits = {"rev": self.toy_fit(rng), "mask": None}
        self.al_info = None

    def toy_fit(self, rng, drop=None):
        zk = []
        for d, lab in ((self.aw1["dir"], self.aw1["lab"]), (self.aw2["d1"], self.aw2["l1"]), (self.aw2["d2"], self.aw2["l2"]),
                       (self.tok["d"], self.tok["lab"])):
            t = (d <= 2) & (lab < K_LAB)
            if drop is not None:
                t = t & ~np.isin(lab, drop)
            zk.append((d * (K_LAB + 1) + lab)[t])
        Zpop = np.unique(np.concatenate(zk))
        zmap = np.full(KZ, -1, np.int64)
        zmap[Zpop] = np.arange(Zpop.size)
        A = rng.uniform(self.LO[0], self.LO[1], (self.rows, Zpop.size)).astype(np.float32)
        A[:, (Zpop % (K_LAB + 1)) == 0] = self.HI
        return {"zmap": zmap, "Zpop": Zpop, "T": int(Zpop.size), "A": A}


def selftest():
    import tempfile
    torch.set_num_threads(1)
    rng = np.random.default_rng(12)
    saved = {k: getattr(L7, k) for k in ("ARMS", "Carve7", "batch_of7", "LeanMLP7", "fit7", "read7")}
    saved_obs, saved_batch, saved_cfg = L7.EMState.observe, LM.batch_of, dict(L7.LeanMLP7.cfg)
    try:
        # 1. em_tau at tau 0 = reltype11.em bit for bit; tau > 0 keeps mu nearer its start
        for trial in range(5):
            nr, T = 40, 7
            Qh = RT.unit(rng.standard_normal((nr, 16)).astype(np.float32))
            cr = np.repeat(np.arange(nr), rng.integers(1, 4, nr))
            cz = rng.integers(0, T, cr.size)
            pr = RT.pairs(cr, cz, np.zeros(cr.size, bool), T)
            init = RT.unit(rng.standard_normal((T, 16)).astype(np.float32))
            m1, p1, i1 = RT.em(Qh, pr, T, 20.0, init)
            m2, p2, i2 = em_tau(Qh, pr, T, 20.0, init, 0.0)
            assert np.array_equal(m1, m2) and np.array_equal(p1, p2) and i1 == i2, trial
            m3, _p3, _i3 = em_tau(Qh, pr, T, 20.0, init, 50.0)
            assert float((m3 * init).sum(1).mean()) > float((m1 * init).sum(1).mean()), trial
        assert np.array_equal(projection(), RT.proj_matrix()), "lean_mlp's projection must be reltype11's"
        # 2. the population pass and fit against loops, with no hold, a training hold and the MASK exclusion
        NN = 70
        G = rng.standard_normal((NN, LM.PROJ_DIM)).astype(np.float16)
        labs = FakeLab12(NN)
        rows = [toy_row(rng, NN, G) for _ in range(40)]
        Gn = RT.unit(G.astype(np.float32))
        fits = {}
        for name, hold, excl in (("plain", None, None), ("train", [2], None), ("mask", None, [2])):
            pp = pop_pass(iter(rows), labs, None if hold is None else np.asarray(hold), NN, np.asarray([r["n"] for r in rows]))
            fit = fit_al(pp, NN, excl, tag=f"toy/{name}")
            trip, cands, Qb = brute_fit_inputs(rows, labs, hold, excl)
            assert np.allclose(pp["Qh"], Qb, atol=1e-5), name
            zs = sorted({z for (_u, _v, z) in trip})
            assert fit["Zpop"].tolist() == zs, name
            for z in zs:
                want = RT.unit(sum(Gn[gv] for (_gu, gv, zz) in trip if zz == z)[None, :])[0]
                assert np.allclose(fit["D"][fit["zmap"][z]], want, atol=1e-5), (name, z)
            got = {}
            for rr, cc in zip(fit["cand_row"].tolist(), fit["cand_col"].tolist()):
                got.setdefault(rr, set()).add(int(fit["Zpop"][cc]))
            assert got == {i: set(c) for i, c in enumerate(cands) if c}, name
            T = fit["T"]
            for i in range(len(rows)):
                s = fit["logpi"] + 20.0 * (Qb[i].astype(np.float64) @ fit["mu"].astype(np.float64).T)
                lse = s.max() + np.log(np.exp(s - s.max()).sum())
                assert np.allclose(fit["A"][i], np.clip(s - lse + math.log(T), -10, 10), atol=1e-4), (name, i)
            fits[name] = fit
        for k in ("Zpop", "D", "mu", "A", "logpi", "cand_row", "cand_col"):
            assert np.array_equal(fits["train"][k], fits["mask"][k]), f"the training mask and the MASK exclusion differ on {k}"
        assert not np.isin(fits["mask"]["Zpop"] % (K_LAB + 1), [2]).any() and np.isin(fits["plain"]["Zpop"] % (K_LAB + 1), [2]).any()
        # every typed walk edge of lean_mlp7's walks maps into the plain fit, and every non-held one into the mask fit
        mk = (lambda: {"fwd_flag": 0, "fwd_found": 0, "bwd_flag": 0, "bwd_found": 0, "missing": 0})
        n_typed = 0
        for r in rows:
            one, two, _par, _t = L7.walk_entries7(r["n"], r["pool"], r["e_u"], r["e_v"], r["e_fam"], r["e_fwd"], r["e_bwd"],
                                                  r["seeds"], r["buckets"], labs, mk())
            for d, lab in ((one[1], one[2]), (two[1], two[2]), (two[3], two[4])):
                d, lab = d.astype(np.int64), lab.astype(np.int64)
                t = (d <= 2) & (lab < K_LAB)
                n_typed += int(t.sum())
                assert (al_cols(fits["plain"], d, lab)[t] >= 0).all()
                assert (al_cols(fits["mask"], d, lab)[t & (lab != 2)] >= 0).all()
                assert (al_cols(fits["mask"], d, lab)[t & (lab == 2)] < 0).all()
        assert n_typed > 50, n_typed
        # chunk_rows12 reads a toy look in lean_mlp4.chunk_rows4's order, row for row
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "toy" / "c1"
            (d / "chunks").mkdir(parents=True)
            per = [rows[:7], rows[7:15], rows[15:22]]
            (d / "record_0of1.json").write_text(json.dumps({"chunks": [0, 2], "proj": {"dim": LM.PROJ_DIM, "seed": LM.PROJ_SEED}}),
                                                encoding="utf-8")
            for ci, rs in enumerate(per):
                np.savez(d / "chunks" / f"c{ci:05d}.npz", q_pool_size=np.array([r["n"] for r in rs]),
                         q_edges=np.array([r["e_u"].size for r in rs]), q_emb=np.stack([r["q_emb"] for r in rs]),
                         pool=np.concatenate([r["pool"] for r in rs]), proj=np.concatenate([r["proj"] for r in rs]),
                         e_u=np.concatenate([r["e_u"] for r in rs]), e_v=np.concatenate([r["e_v"] for r in rs]),
                         e_fam=np.concatenate([r["e_fam"] for r in rs]), e_fwd=np.concatenate([r["e_fwd"] for r in rs]),
                         e_bwd=np.concatenate([r["e_bwd"] for r in rs]), q_seed_local=np.stack([r["seeds"] for r in rs]),
                         q_seed_bucket=np.stack([r["buckets"] for r in rs]))
            got = list(chunk_rows12(d, None))
            want = per[0] + per[2]
            assert len(got) == len(want) == len(list(L4.chunk_rows4(d, None)))
            for g_, w_ in zip(got, want):
                assert all(np.array_equal(np.asarray(g_[k]), np.asarray(w_[k])) for k in w_), "chunk_rows12 row"
            assert len(list(chunk_rows12(d, 9))) == 9
            (d / "record_0of1.json").write_text(json.dumps({"chunks": [0], "proj": {"dim": 64, "seed": 1}}), encoding="utf-8")
            try:
                list(chunk_rows12(d, None))
                raise AssertionError("a look with another projection must be refused")
            except SystemExit:
                pass
        # 3. al_batch: on, off, shuf, the mask fit, NR
        c = ToyCarve12(14, 3)
        qs = np.array([0, 2, 5, 11, 13])
        fit = c.al_fits["rev"]
        A = al_batch(c, qs)
        o1 = np.concatenate([np.arange(c.aw_off1[q], c.aw_off1[q + 1]) for q in qs])
        o2 = np.concatenate([np.arange(c.aw_off2[q], c.aw_off2[q + 1]) for q in qs])
        b1 = np.repeat(np.arange(qs.size), c.aw_off1[qs + 1] - c.aw_off1[qs])
        b2 = np.repeat(np.arange(qs.size), c.aw_off2[qs + 1] - c.aw_off2[qs])

        def brute(d, lab, b, Arow):
            out_ = np.zeros(d.size, np.float32)
            for j in range(d.size):
                if d[j] <= 2 and lab[j] < K_LAB:
                    out_[j] = Arow[b[j]][fit["zmap"][d[j] * (K_LAB + 1) + lab[j]]]
            return out_

        Ar = fit["A"][qs]
        assert np.array_equal(A["al1"].numpy(), brute(c.aw1["dir"][o1], c.aw1["lab"][o1], b1, Ar))
        assert np.array_equal(A["al2a"].numpy(), brute(c.aw2["d1"][o2], c.aw2["l1"][o2], b2, Ar))
        assert np.array_equal(A["al2b"].numpy(), brute(c.aw2["d2"][o2], c.aw2["l2"][o2], b2, Ar))
        assert float(A["al1"].abs().sum()) > 0 and bool((A["al1"] == ToyCarve12.HI).any())
        c.al_mode = "off"
        A0 = al_batch(c, qs)
        assert all(float(A0[k].abs().sum()) == 0.0 and A0[k].shape == A[k].shape for k in A)
        c.al_mode = "shuf"
        AS = al_batch(c, qs)
        c.al_mode = "on"
        Ash = Ar.copy()
        for b, q in enumerate(qs):
            cols = []
            for d, lab, oo in ((c.aw1["dir"], c.aw1["lab"], (c.aw_off1[q], c.aw_off1[q + 1])),
                               (c.aw2["d1"], c.aw2["l1"], (c.aw_off2[q], c.aw_off2[q + 1])),
                               (c.aw2["d2"], c.aw2["l2"], (c.aw_off2[q], c.aw_off2[q + 1]))):
                cols += [int(x) for x in al_cols(fit, d[oo[0]:oo[1]], lab[oo[0]:oo[1]]) if x >= 0]
            u = sorted(set(cols))
            if len(u) > 1:
                Ash[b, u] = [Ar[b, u[k - 1]] for k in range(len(u))]
        assert np.array_equal(AS["al1"].numpy(), brute(c.aw1["dir"][o1], c.aw1["lab"][o1], b1, Ash))
        assert np.array_equal(AS["al2b"].numpy(), brute(c.aw2["d2"][o2], c.aw2["l2"][o2], b2, Ash))
        assert not np.array_equal(AS["al1"].numpy(), A["al1"].numpy())
        c.aw_nr = True
        An = al_batch(c, qs)
        c.aw_nr = False
        assert all(torch.equal(An[k], A[k]) for k in A), "NR keeps a"
        l1o, d1o = c.aw1["lab"][o1], c.aw1["dir"][o1]
        hl = int(np.bincount(l1o[(d1o <= 2) & (l1o < K_LAB) & (l1o > 0)]).argmax())
        c.hold = np.array([hl])
        c.al_fits["mask"] = c.toy_fit(np.random.default_rng(1), drop=[hl])
        c.aw_mask = True
        Am = al_batch(c, qs)
        c.aw_mask = False
        held1 = (l1o == hl) & (d1o <= 2)
        assert held1.any() and float(Am["al1"][torch.from_numpy(held1)].abs().sum()) == 0.0
        fit_m = c.al_fits["mask"]
        assert np.array_equal(Am["al1"].numpy()[~held1], np.where(
            al_cols(fit_m, c.aw1["dir"][o1], c.aw1["lab"][o1]) >= 0,
            fit_m["A"][qs][b1, np.maximum(al_cols(fit_m, c.aw1["dir"][o1], c.aw1["lab"][o1]), 0)], 0.0).astype(np.float32)[~held1])
        A7 = L7.aw_batch7(c, qs)
        A12 = aw_batch12(c, qs)
        assert all(torch.equal(A12[k], v) if torch.is_tensor(v) else A12[k] == v for k, v in A7.items()) and "al1" in A12
        # 4. AWNet12: aw = AWNet7 aw op for op; awal (lam 0) = aw; aw0al (lam 0) = aw0; aw0 reads no label; lam moves
        #    the columns and gets a gradient; an old arm's AWNet12 is AWNet7
        A = aw_batch12(c, np.arange(c.rows))
        for train_mode in (True, False):
            nets = {"aw7": L7.AWNet7(8, 16, 0, 0.25, seed=3, arm="aw")}
            nets.update({arm: AWNet12(8, 16, 0, 0.25, seed=3, arm=arm) for arm in ("aw", "awal", "aw0", "aw0al", "awc", "em")})
            nets["awc7"] = L7.AWNet7(8, 16, 0, 0.25, seed=3, arm="awc")
            nets["em7"] = L7.AWNet7(8, 16, 0, 0.25, seed=3, arm="em")
            outs = {}
            for nm_, net in nets.items():
                net.train(train_mode)
                torch.manual_seed(11)
                outs[nm_] = net(A)
            assert torch.equal(outs["aw7"], outs["aw"]) and torch.equal(outs["aw"], outs["awal"]), train_mode
            assert torch.equal(outs["aw0"], outs["aw0al"]) and not torch.equal(outs["aw0"], outs["aw"]), train_mode
            assert torch.equal(outs["awc7"], outs["awc"]) and torch.equal(outs["em7"], outs["em"]), train_mode
            Ap = dict(A)
            Ap["l1"] = torch.where(A["l1"] < K_LAB, (A["l1"] + 5) % 20, A["l1"])
            Ap["l2b"] = torch.where(A["l2b"] < K_LAB, (A["l2b"] + 3) % 20, A["l2b"])
            torch.manual_seed(11)
            assert torch.equal(nets["aw0"](Ap), outs["aw0"]), "aw0 must read no label"
            torch.manual_seed(11)
            assert not torch.equal(nets["aw"](Ap), outs["aw"]), "aw reads its labels"
        assert [n for n, _ in AWNet12(8, 16, 0, 0.25, seed=3, arm="aw").named_parameters()] == \
            [n for n, _ in L7.AWNet7(8, 16, 0, 0.25, seed=3, arm="aw").named_parameters()]
        n_al = AWNet12(8, 16, 0, 0.25, seed=3, arm="aw0al")
        n_al.eval()
        with torch.no_grad():
            n_al.lam.fill_(0.7)
        assert not torch.equal(n_al(A), outs["aw0al"]), "lam must move the columns"
        n_al.train()
        n_al.zero_grad()
        n_al(A).sum().backward()
        assert n_al.lam.grad is not None and float(n_al.lam.grad[:3].abs().sum()) > 0 and float(n_al.lam.grad[3:].abs().sum()) == 0.0
        assert n_al.P.grad is None or float(n_al.P.grad.abs().sum()) == 0.0, "aw0al reads no label code"
        tl_ = n_al.unit_logit(A, A["tl"])
        A0_ = dict(A)
        A0_["tal"] = torch.zeros_like(A["tal"])
        with torch.no_grad():
            want = n_al.unit_logit(A0_, A["tl"]) + n_al.lam[A["td"]] * A["tal"]
        assert torch.allclose(tl_, want), "the token logit adds lam a"
        # 5. the fits: an old arm under LeanMLP12 = under LeanMLP7, bit for bit (fit7g); the new arms fit; the planted
        #    alignment is learned by aw0al (lam > 0 on the structural directions, select above aw0)
        L3.LeanMLP3.dim = LM.PROJ_DIM
        L7.LeanMLP7.cfg = {"r": 8, "D": 16, "k_res": 0, "p_drop": 0.25, "k_types": 4}
        LeanMLP12.cfg = dict(L7.LeanMLP7.cfg)
        base = L5.parse_configs("b=1e-2:1e-4:0.1:4:1")["b"]
        emc = {"lam0": 1.0, "an": 2, "mu": 1.0, "omega": 1.0}
        LM.batch_of = L7.batch_of7
        for arm in ("ctl", "aw", "awc", "aux", "em", "em0"):
            bl = ["rank", "WALK"] if arm == "ctl" else ["rank", "WALK", "AW"]
            tr, se = [L7.ToyCarve7(60, 1), L7.ToyCarve7(40, 4)], [L7.ToyCarve7(30, 2)]
            L7.LeanMLP7 = saved["LeanMLP7"]
            f = LG.fit7g(tr, se, bl, arm, base, 0, 32, emc if arm in L7.EM_ARMS else None)
            L7.LeanMLP7 = LeanMLP12
            g = LG.fit7g(tr, se, bl, arm, base, 0, 32, emc if arm in L7.EM_ARMS else None)
            LG._same(f, g)
        LM.batch_of = batch_of12
        L7.batch_of7 = batch_of12
        L7.LeanMLP7 = LeanMLP12
        # AW alone: with the rank block in, the toy's rank column carries most of the gold and the margin shrinks to a
        # few points; lam's sign is not identifiable (the MLP's weight on the AW columns can carry it), its size is
        tr, se = [ToyCarve12(80, 1), ToyCarve12(60, 4)], [ToyCarve12(50, 2)]
        bl = ["AW"]
        res = {}
        for arm in ("aw0", "aw0al", "awal"):
            f = fit12(tr, se, bl, arm, L5.parse_configs("b=2e-2:1e-4:0.1:8:2")["b"], 0, 32)
            m = L7.build7(bl, f["widths"], 32, f["best"], arm)
            res[arm] = (LM.quality(m, se, bl, {b: 1.0 for b in bl}), f)
        lam = res["aw0al"][1]["best"]["awn.lam"]
        q0_, q1_ = res["aw0"][0], res["aw0al"][0]
        assert all("lam" in rec for rec in res["aw0al"][1]["curve"]) and "lam" not in res["aw0"][1]["curve"][0]
        assert float(lam[:3].abs().mean()) > 0.05 and float(lam[3:].abs().sum()) == 0.0, lam
        assert q1_[0] + q1_[1] > q0_[0] + q0_[1] + 0.2, (q0_, q1_)
        assert set(AL["lam"]) == {"aw0al", "awal"} and len(AL["lam"]["aw0al"]["per_epoch"]) == 8
        # 6. read12 on toy carves (costs stubbed): every view, the held rows, the twin pairs, the edge rankings
        se2 = ToyCarve12(40, 6)
        se2.hold = np.array([3])
        se2.hold_mode = "read"
        se2.held = np.arange(se2.rows) % 3 != 0
        se2.al_fits["mask"] = se2.toy_fit(np.random.default_rng(2), drop=[3])
        for k_, v_ in {"chunks_read": 1, "n_chunks": 1, "carve_queries": se2.rows, "aw_stats": {}, "lean_ms": {},
                       "struct_share": 1.0}.items():
            setattr(se2, k_, v_)
        models = {}
        for arm in ("aw0", "aw0al", "awal"):
            for kind in ("best", "swa"):
                models[f"{arm}@{kind}"] = (L7.build7(bl, res[arm][1]["widths"], 32, res[arm][1][kind], arm), bl, {b: 1.0 for b in bl})

        class _A:
            read = "toy=t6"
            out = None

        old_costs, old_cost = L2.costs_of, LM.cost_ms
        L2.costs_of = lambda cs: [(None, 1.0, {})]
        LM.cost_ms = lambda *a_, **k_: 0.0
        try:
            out = {}
            read12(_A(), models, out, lambda ds, cv, role: se2)
        finally:
            L2.costs_of, LM.cost_ms = old_costs, old_cost
        rk = out["reads"]
        for nm_ in ("aw0al@best", "awal@swa"):
            for tag in ("", "/NR", "/ID-A0", "/ID-AS", "/FNR-A0", "/MASK", "/MASK-A0", "/MASK-AS"):
                assert f"{nm_}@toy=t6{tag}" in rk, (nm_, tag)
        assert "aw0@best@toy=t6/ID-A0" not in rk and "aw0@best@toy=t6/MASK" in rk
        assert "minus_aw0" in rk["aw0al@best@toy=t6"] and "minus_aw" not in rk["aw0al@best@toy=t6"]
        h = out["held"]["toy=t6"]
        assert h["rows"] == int(se2.held.sum())
        e = h["aw0al@best"]
        assert {"REV", "MASK", "REV-A0", "REV-AS", "MASK-A0", "MASK-AS", "NR", "FNR-A0", "REV_minus_REV-A0", "REV_minus_REV-AS",
                "MASK_minus_MASK-A0", "REV_minus_aw0", "MASK_minus_aw0", "REV_minus_MASK"} <= set(e), sorted(e)
        assert "REV-A0" not in h["aw0@best"]
        er = out["edge_rank"]
        assert {"aw0al@best@toy=t6", "aw0al@best@toy=t6/MASK", "al@toy=t6", "al@toy=t6/MASK"} <= set(er)
        assert er["al@toy=t6"]["auc_al_typed"] is not None and 0.0 <= er["al@toy=t6"]["auc_al"] <= 1.0
        assert "auc_aw_a0" in er["aw0al@best@toy=t6"] and "auc_aw_a0" not in er["aw0@best@toy=t6"]
        # the planted types are the on-path ones only where a token's label is 0; the toy's tokens are random, so only
        # the ranges are checked here
        # 7. the options and the bindings
        assert strip_al(["x", "--al-kappa", "5", "--arms", "aw0al", "--al-tau=2", "--out", "o.json"]) == (
            ["x", "--arms", "aw0al", "--out", "o.json"], {"--al-kappa": "5", "--al-tau": "2"})
        assert arms_of(["x", "--arms", "aw0,aw0al"]) == ["aw0", "aw0al"] and arms_of(["x"]) == ["ctl", "aw", "awc"]
        try:
            strip_al(["x", "--al-tau"])
            raise AssertionError("an option with no value must be refused")
        except SystemExit:
            pass
        install12()
        assert L7.Carve7 is Carve12 and L7.batch_of7 is batch_of12 and L7.LeanMLP7 is LeanMLP12 and L7.fit7 is fit12
        assert L7.read7 is read12 and set(NEW_ARMS) <= set(L7.ARMS) and L7.EMState.observe is LG.observe_g
        assert len(L7.ARMS) == len(set(L7.ARMS))
        install12()
        assert len(L7.ARMS) == len(set(L7.ARMS)), "install12 twice adds no arm twice"
    finally:
        for k, v in saved.items():
            setattr(L7, k, v)
        L7.EMState.observe, LM.batch_of = saved_obs, saved_batch
        L7.LeanMLP7.cfg = saved_cfg
    print("selftest ok: em_tau(0) = reltype11.em; the projection is reltype11's; the population pass and fit = loops "
          "(q_i, the typed edge set, descriptors, the two seed levels, a), the training mask = the MASK exclusion, every "
          "typed walk edge maps in; chunk_rows12 = chunk_rows4's rows, a foreign projection refused; al_batch on/off/shuf, "
          "the mask fit, NR keeps a; AWNet12 aw = AWNet7, awal(0) = aw, aw0al(0) = aw0, aw0 reads no label, lam moves "
          "and learns; old arms fit bit for bit under LeanMLP12; aw0al learns the planted alignment (lam "
          f"{[round(float(v), 3) for v in lam[:3]]}, select {[round(v, 3) for v in q1_[:2]]} vs aw0 {[round(v, 3) for v in q0_[:2]]}); "
          "read12's views, held rows, twin pairs and edge rankings; options and bindings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
