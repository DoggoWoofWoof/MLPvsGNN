"""Design look (untracked; not a result and not filed): qd_gnn13.py, pinned and unchanged, with one arm-name option for
QD arms, -al: a label-free alignment channel (S4 of the transfer plan, from S3's reltype11.py). It gives each structural
edge a scalar for each query, read from the relation's id and the graph's own unlabeled queries, never from the
relation's name, any gold, or any trained model.

Types. A structural edge's type is z = 3 min(rank, R_TOP) + d: its relation rank as qd_gnn.edge_labels reads it (the
rank in the direction the edge is read) and its famdir d (0 S-fwd, 1 S-bwd, 2 S-both). A rank is index-time structure
(the KB's relation id ordered by count), so no name is read. qd_gnn6's MASK view shows a held rank as R_TOP, and kb_view
already gives a structural pair with no relation that rank, so those edges share one type. On webqsp, ranks past R_TOP
fold into it as well.

Population fit (per graph and per population: the training carve's rows in training, otherwise the rows being read in
the view being read; no gold anywhere):
  q_i  = n(n(qemb_i) P), P being look_score_kb's 128-wide projection (reltype11.proj_matrix). A node's text p_v is the
         look's stored projection of its unit embedding, normalised: n(p_v) (KB look chunks, deduplicated on the global
         node id, every row's pool checked against its chunk).
  d_z  = n(sum of n(p_v) over the population's distinct (u, v, z) edges): the mean text of a type's far endpoints.
  C_i  = the types of row i's structural edges out of its rank-1 seeds (bucket 0) to a node outside them; with none,
         out of all its seeds; reltype11's radius-1 ball, as messages.
  EM   = reltype11.em over (q_i, C_i), started from d_z, kappa 20, 30 rounds: pi_z and mu_z, the unlabeled query
         distribution's own account of which type a query asks for. --al-tau T > 0 anchors the M-step to the
         descriptor, mu_z = n(sum_i r_iz q_i + T d_z), so a type that few queries support stays near its descriptor
         (webqsp's lesson in S3). The default is 0, which is reltype11's EM exactly.
  pi~_z = (rows_z + 1) / (rows + T) over the population's T present types.
Per edge e of type z in row i:
  a_e(q_i) = clip(s_iz - logsumexp over the population's types of s_iz' + log T, -10, 10),
  s_iz     = log pi~_z + kappa cos(q_i, mu_z),
and 0 on a non-structural edge. The normaliser runs over the population's types and never the row's own, so cutting a
row's edges never changes the value of another edge, and the frontier cut stays exact (--selftest checks it).

Model. QD13's forward with one term per layer: the gate logit adds lam_l a_e and, with -at, the attention logit adds
lam_at_l a_e before the softmax. Both start at zero, so an unfitted -al model scores as the same model without -al. The
draws are identical (the same parameters and generators, drawn in the same order), so -al and its twin start from the
same weights. Label dropout, -ed and -fd cut a_e along with the other edge keys, never changing a type.

Reads. Every qd_gnn2 / qd_gnn6 / qd_gnn7 read fits its own population: ID, NR and SHUF on the read rows, each hold view
on its own rows, and kalpha's pool on its own. NR keeps a_e: names are absent but edges are still typed. Under --hold,
an -al arm adds these views:
  ID-A0                    the held view with a_e = 0 (the alignment's own contribution)
  FULL-A0, FULL-AS         (T0, drop) the held edges back, with a_e off, or with each row's per-type values rotated
                           by one type (a control that keeps the values but breaks which type gets which)
  REV-A0, REV-AS, MASK-A0, MASK-AS   (text arms) the same for REV and MASK
  FNR-A0                   the untyped graph: every label 'other' and a_e = 0
KB graphs only: a passage graph has no relation ids (its label-free types are a later file), so -al refuses one.

    python outputs/mp_unified/qd_gnn15.py --selftest
    python outputs/mp_unified/qd_gnn15.py --train metaqa=fit --hold metaqa:drop:has_genre --rule both --epochs 6 \
        --arms QD-T0-L3-at-al --out outputs/mp_unified/qd/al-mq-genre-t0.json
"""
import hashlib
import json
import math
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn13 as Q13  # noqa: E402  (imports qd_gnn12 ... qd_gnn, which sets the BLAS thread counts before numpy loads)
import reltype11 as RT  # noqa: E402  (S3's EM, pairs, unit and projection, unchanged)

import numpy as np  # noqa: E402
import scipy.sparse as sparse  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

Q12, Q11, Q10, Q7, Q6x, Q5, Q4, Q2, QG = Q13.Q12, Q13.Q11, Q13.Q10, Q13.Q7, Q13.Q6x, Q13.Q5, Q13.Q4, Q13.Q2, Q13.QG
SHAS = {**Q13.SHAS, "reltype11": QG.sha(RT.__file__), "qd_gnn15": QG.sha(__file__)}   # at import: what ran
OPT15 = re.compile(r"-(al)(?=-|$)")
KZ = 3 * (QG.R_TOP + 1)
CLIP = 10.0
FIT_CACHE = 6
EDGE_KEYS = ("src", "dst", "e_row", "d", "a", "fam", "w", "align")
AL = {"needed": False, "kappa": 20.0, "tau": 0.0, "iters": RT.EM_ITERS, "nodes": {}, "node_info": {}, "vocab": {},
      "train_rows": {}, "fit_log": [], "builder": None}
FITS = OrderedDict()   # population key -> {"fit", "refs"}; refs hold the view's lists, so their ids stay unique
PM = {}
log = QG.log
_split13 = Q13.split_opts


# ── arm option ───────────────────────────────────────────────────────────────


def split_opts(nm):
    """(the qd_gnn6 name, the qd_gnn7 + qd_gnn10 + qd_gnn13 + qd_gnn15 options) of an arm name; -al may sit anywhere
    among the options, the others are parsed by qd_gnn13 (and so qd_gnn10 and qd_gnn7) themselves."""
    found = OPT15.findall(nm)
    if len(found) > 1:
        raise SystemExit(f"{nm}: -al given twice")
    base, opts = _split13(OPT15.sub("", nm))
    if not found:
        return base, opts
    if not base.startswith("QD-"):
        raise SystemExit(f"{nm}: -al is for QD arms only")
    if opts.get("cap"):
        raise SystemExit(f"{nm}: -al with -cap is refused (the alignment is computed on the arm's own rows)")
    return base, {**opts, "al": True}


def al_arm(arm):
    return isinstance(arm, QDArm15) and bool((arm.sp.get("t7") or {}).get("al"))


# ── node text, types, queries ────────────────────────────────────────────────


def proj():
    if "P" not in PM:
        PM["P"] = RT.proj_matrix()
    return PM["P"]


def qhat(Qv, rows):
    """q_i = n(n(qemb_i) P) for each row, (len(rows), 128)."""
    X = np.stack([np.asarray(Qv[i]["qemb"], dtype=np.float32) for i in rows])
    return RT.unit(RT.unit(X) @ proj())


def types_of(d, a):
    """z = 3 min(rank, R_TOP) + famdir over structural edges (d in 0..2, rank >= 0: qd_gnn.edge_labels' guarantee)."""
    d = np.asarray(d, dtype=np.int64)
    if d.size and (d.min() < 0 or d.max() > 2):
        raise SystemExit("a structural edge's famdir is not 0, 1 or 2")
    return 3 * np.minimum(np.asarray(a, dtype=np.int64), QG.R_TOP) + d


def merge_nodes(ids0, P0, bi, bp):
    """(sorted unique ids, the first row of each, the largest gap between a later duplicate and its first)."""
    ids = np.concatenate([ids0] + bi)
    P = np.concatenate([P0] + bp)
    order = np.argsort(ids, kind="stable")
    ids, P = ids[order], P[order]
    first = np.r_[True, ids[1:] != ids[:-1]] if ids.size else np.zeros(0, dtype=bool)
    grp = np.cumsum(first) - 1
    gap = float(np.abs(P.astype(np.float32) - P[first][grp].astype(np.float32)).max()) if P.size else 0.0
    return ids[first], P[first], gap


def build_nodes(g, carves, part, Q):
    """The KB's node table over the loaded carves: global id -> n(p_v), from the look chunks qd_gnn2 loaded (each
    row's pool checked against its chunk: row k of a carve is its k-th smallest chunk row, as load_pruned_kb orders)."""
    KB = sys.modules["anchor_kbfit"].KB
    t0 = time.time()
    ids, P = np.zeros(0, dtype=np.int64), np.zeros((0, RT.PROJ_DIM), dtype=np.float16)
    gap, checked, nch, bi, bp = 0.0, 0, 0, [], []
    for carve in carves:
        look = Path(KB.LOOK_ROOT) / g / carve
        rec = json.loads(sorted(look.glob("record*.json"))[0].read_text(encoding="utf-8"))
        if rec.get("proj") != {"dim": RT.PROJ_DIM, "seed": RT.PROJ_SEED}:
            raise SystemExit(f"{look}: projection {rec.get('proj')}, not reltype11's")
        paths = sorted((look / "chunks").glob("c*.npz"))
        crs = []
        for p in paths:
            with np.load(p) as zf:
                crs.append(zf["chunk_rows"].astype(np.int64))
        all_rows = np.sort(np.concatenate(crs))
        rows_c = part[(g, carve)]
        if all_rows.size != len(rows_c):
            raise SystemExit(f"{g}/{carve}: {all_rows.size} chunk rows, {len(rows_c)} loaded rows")
        for p, cr in zip(paths, crs):
            with np.load(p) as zf:
                npool, pool, pj = zf["q_pool_size"], zf["pool"], zf["proj"]
            if pj.shape != (pool.size, RT.PROJ_DIM):
                raise SystemExit(f"{p}: proj {pj.shape} for {pool.size} pool nodes")
            po = np.r_[0, np.cumsum(npool)]
            for k, row in enumerate(cr):
                qi = rows_c[int(np.searchsorted(all_rows, row))]
                if not np.array_equal(np.asarray(Q[qi]["pool"]), pool[po[k]:po[k + 1]]):
                    raise SystemExit(f"{p}: chunk row {row} is not loaded row {qi}'s pool")
                checked += 1
            bi.append(pool.astype(np.int64))
            bp.append(pj.astype(np.float16))
            nch += 1
            if len(bi) >= 8:
                ids, P, gp = merge_nodes(ids, P, bi, bp)
                gap, bi, bp = max(gap, gp), [], []
    if bi:
        ids, P, gp = merge_nodes(ids, P, bi, bp)
        gap = max(gap, gp)
    if gap > 1e-3:
        raise SystemExit(f"{g}: one node id carries two projections (gap {gap})")
    Pn = RT.unit(P.astype(np.float32))
    info = {"carves": list(carves), "chunks": nch, "rows_checked": checked, "nodes": int(ids.size), "proj_max_gap": gap,
            "zero_text_nodes": int((np.linalg.norm(Pn, axis=1) < 1e-6).sum()), "seconds": round(time.time() - t0, 1)}
    return ids, Pn, info


def node_index(g, q):
    ids = AL["nodes"][g][0]
    pool = np.asarray(q["pool"], dtype=np.int64)
    k = np.searchsorted(ids, pool)
    if not (k < ids.size).all() or not np.array_equal(ids[k], pool):
        raise SystemExit(f"{g}: a pool node is missing from the node table")
    return k


def seed_edges(q, u, v):
    """C_i over a row's structural edges (u, v): out of its rank-1 seeds (bucket 0) to a node outside them; with none,
    out of all its seeds to a node outside them; None when neither gives an edge."""
    s, b = np.asarray(q["seeds"]), np.asarray(q["bucket"])
    ok = s >= 0
    for S in (s[ok & (b == 0)], s[ok]):
        if S.size == 0:
            continue
        ins = np.zeros(int(q["n"]), dtype=bool)
        ins[S] = True
        c = ins[u] & ~ins[v]
        if c.any():
            return c
    return None


# ── the population fit ───────────────────────────────────────────────────────


def em_tau(Qh, pr, Kz, kappa, init, tau, seed=RT.SEED, iters=RT.EM_ITERS):
    """reltype11.em with the M-step anchored to the start: mu_z = n(sum_i r_iz q_i + tau d_z)."""
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


def type_name(g, z):
    rank, d = int(z) // 3, int(z) % 3
    voc = AL["vocab"].get(g) or []
    nm = voc[rank] if rank < min(len(voc), QG.R_TOP) else ("other" if rank == QG.R_TOP else f"rank {rank}")
    return f"{nm}/{('fwd', 'bwd', 'both')[d]}"


def fit_pop(g, rows, Qv, Tv, tag):
    """The alignment fit of one population (see the module docstring): its present types, pi~ and mu."""
    t0 = time.time()
    if g not in AL["nodes"]:
        raise SystemExit(f"-al: no node table for {g}")
    ids, Pn = AL["nodes"][g]
    NN = int(ids.size)
    if NN * NN * KZ >= 2 ** 63:
        raise SystemExit(f"{g}: {NN} nodes overflow the edge key")
    keys, blk, cand_row, cand_z = np.zeros(0, dtype=np.int64), [], [], []
    for pos, i in enumerate(rows):
        q, (d, a) = Qv[i], Tv[i]
        m = q["fam"] == 0
        if not m.any():
            continue
        u, v = q["u"][m], q["v"][m]
        z = types_of(d[m], a[m])
        nid = node_index(g, q)
        blk.append((nid[u] * NN + nid[v]) * KZ + z)
        c = seed_edges(q, u, v)
        if c is not None:
            cand_row.append(np.full(int(c.sum()), pos, dtype=np.int64))
            cand_z.append(z[c])
        if len(blk) >= 256:
            keys = np.unique(np.concatenate([keys] + blk))
            blk = []
    if blk:
        keys = np.unique(np.concatenate([keys] + blk))
    z_all = keys % KZ
    Zpop = np.unique(z_all)
    T = int(Zpop.size)
    if T == 0:
        raise SystemExit(f"{g}: a population with no structural edge")
    zmap = np.full(KZ, -1, dtype=np.int64)
    zmap[Zpop] = np.arange(T)
    D = RT.unit(sparse.csr_matrix((np.ones(keys.size, dtype=np.float32), (zmap[z_all], (keys // KZ) % NN)),
                                  shape=(T, NN)) @ Pn)
    Qh = qhat(Qv, rows)
    rows_n = 0
    if cand_row:
        cr = np.concatenate(cand_row)
        pr = RT.pairs(cr, zmap[np.concatenate(cand_z)], np.zeros(cr.size, dtype=bool), T)
        rows_n = int(pr["starts"].size)
    if rows_n == 0:
        mu, pi, info = D.copy(), np.zeros(T), {"loglik": [], "pi_perplexity": None}
    elif AL["tau"] > 0:
        mu, pi, info = em_tau(Qh, pr, T, AL["kappa"], D, AL["tau"], seed=RT.SEED, iters=AL["iters"])
    else:
        mu, pi, info = RT.em(Qh, pr, T, AL["kappa"], D, seed=RT.SEED, iters=AL["iters"])
    mass = pi * rows_n
    pit = (mass + 1.0) / (rows_n + T)
    fit = {"zmap": zmap, "mu": np.asarray(mu, dtype=np.float32), "logpi": np.log(pit), "logT": math.log(T), "T": T,
           "rowset": frozenset(rows)}
    top = np.argsort(-pit, kind="stable")[:12]
    rec = {"graph": g, "tag": tag, "rows": len(rows),
           "rows_sha1": hashlib.sha1(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()[:12],
           "rows_with_candidates": rows_n, "types": T, "edges": int(keys.size), "pi_perplexity": info["pi_perplexity"],
           "loglik": info["loglik"], "top": [[type_name(g, Zpop[k]), round(float(pit[k]), 4), round(float(mass[k]), 1)]
                                             for k in top],
           "seconds": round(time.time() - t0, 1)}
    AL["fit_log"].append(rec)
    log(f"al fit {g} {tag}: {len(rows)} rows ({rows_n} with seed edges), {T} types, {keys.size} edges, pi perplexity "
        f"{info['pi_perplexity']}, top {rec['top'][:4]} ({rec['seconds']}s)")
    return fit


def get_fit(g, rows, Qv, Tv, tag):
    key = (g, id(Qv), id(Tv), len(rows), hashlib.sha1(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest())
    ent = FITS.get(key)
    if ent is not None:
        FITS.move_to_end(key)
        return ent["fit"]
    fit = fit_pop(g, list(rows), Qv, Tv, tag)
    FITS[key] = {"fit": fit, "refs": (Qv, Tv)}
    while len(FITS) > FIT_CACHE:
        FITS.popitem(last=False)
    return fit


def align_batch(fit, rows, Qv, Tv, mode):
    """a_e for every edge of the rows, in qd_gnn.pack_qd's edge order; mode on, off (zeros) or shuf (each row's
    unique types' values rotated by one type)."""
    if mode == "off":
        return np.zeros(sum(int(Qv[i]["u"].size) for i in rows), dtype=np.float32)
    if mode not in ("on", "shuf"):
        raise ValueError(mode)
    Qh = qhat(Qv, rows).astype(np.float64)
    S = fit["logpi"][None, :] + AL["kappa"] * (Qh @ fit["mu"].astype(np.float64).T)      # (B, T)
    mx = S.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(S - mx).sum(1))
    A = np.clip(S - lse[:, None] + fit["logT"], -CLIP, CLIP)
    out = []
    for b, i in enumerate(rows):
        q, (d, a) = Qv[i], Tv[i]
        val = np.zeros(q["u"].size, dtype=np.float64)
        m = q["fam"] == 0
        if m.any():
            zi = fit["zmap"][types_of(d[m], a[m])]
            if (zi < 0).any():
                raise SystemExit("-al: an edge type outside its population's fit")
            if mode == "shuf":
                uz, inv = np.unique(zi, return_inverse=True)
                val[m] = np.roll(A[b, uz], 1)[inv]
            else:
                val[m] = A[b, zi]
        out.append(val)
    return np.concatenate(out).astype(np.float32)


def train_rows(g):
    if AL["train_rows"].get(g) is not None:
        return AL["train_rows"][g]
    part, trains = Q7.STATE7["part"], Q7.STATE7["trains"]
    if not part or not trains or g not in trains:
        raise SystemExit(f"-al: no training rows for {g}")
    return [i for c in trains[g] for i in part[(g, c)]]


# ── the model ────────────────────────────────────────────────────────────────


class QD15(Q13.QD13):
    """qd_gnn13.QD13 plus the alignment terms lam_l a_e (gate) and, with at, lam_at_l a_e (attention), both from
    zero; QD13's forward otherwise, line for line, whichever of its options are on."""

    def __init__(self, kind, d=64, layers=3, n_id=0, g1=False, tb=None, rs=False, lq=False, at=False, rg=0):
        super().__init__(kind, d, layers, n_id, g1=g1, tb=tb, rs=rs, lq=lq, at=at, rg=rg)
        self.lam_g = torch.nn.Parameter(torch.zeros(layers))
        if self.at:
            self.lam_at = torch.nn.Parameter(torch.zeros(layers))

    def attention(self, l, hs, r, qn, P, active, n_in):
        pre = self.Us[l](hs)[P.src] + self.Ud[l](hs)[P.dst] + self.Ur[l](r) + self.Uq[l](qn)[P.e_row]
        e = (F.leaky_relu(pre, 0.2) * self.a_at[l]).sum(-1)
        e = e + self.lam_at[l] * P.align
        on = active[P.src] > 0
        e = torch.where(on, e, torch.full_like(e, -1e30))
        mx = torch.zeros(P.N).scatter_reduce(0, P.dst, e.detach(), "amax", include_self=False)
        ex = torch.exp(e - mx[P.dst]) * on.to(e.dtype)
        den = torch.zeros(P.N).index_add(0, P.dst, ex)
        return ex / den[P.dst].clamp_min(1e-30) * n_in[P.dst]

    def forward(self, P):
        rms = QG.rms
        qn = self.lnq(self.Wq(P.qemb))                                  # (B, d)
        h0 = qn[P.seed_row] * (1.0 + self.E_b[P.seed_b]) + P.z[P.seed_idx, None] * self.e_z
        h = torch.zeros(P.N, self.dim).index_add(0, P.seed_idx, h0)
        r = self.relations(P)
        c = self.label_cos(P) if self.lq else None
        gamma = torch.sigmoid(self.gamma_raw)
        fam = P.fam.clamp(max=2)
        eye = torch.eye(self.dim)
        for l in range(self.L):
            w = self.R[l](r) * (1.0 + torch.tanh(self.G[l](qn)))[P.e_row]
            logit = (self.A[l](qn)[P.e_row] * r).sum(-1) + self.c_fam[l, fam] * P.w + self.b_gate[l]
            if self.lq:
                logit = logit + self.kappa[l] * c
            logit = logit + self.lam_g[l] * P.align
            g = 2.0 * torch.sigmoid(logit)
            hs = rms(h)
            if self.rg:
                coef = torch.softmax(self.C[l](r), -1)                     # (E, B)
                x = coef[:, 0:1] * (hs @ (eye + self.D[l, 0]).T)[P.src]
                for b in range(1, self.rg):
                    x = x + coef[:, b:b + 1] * (hs @ (eye + self.D[l, b]).T)[P.src]
            else:
                x = hs[P.src]
            m = x * w * g[:, None]
            active = (h != 0).any(-1).to(m.dtype)
            n_in = torch.zeros(P.N).index_add(0, P.dst, active[P.src])
            if self.at:
                m = m * self.attention(l, hs, r, qn, P, active, n_in)[:, None]
            if self.g1:
                agg = torch.zeros(P.N, self.dim).index_add(0, P.dst, m) / (1.0 + n_in)[:, None]
            else:
                agg = torch.zeros(P.N, self.dim).index_add(0, P.dst, m) / (1.0 + n_in)[:, None].pow(gamma)
            h = h + torch.relu(self.W[l](agg))
        hn = rms(h)
        reached = (h != 0).any(-1).to(h.dtype)
        f = torch.cat([hn, hn * qn[P.node_row], P.z[:, None], reached[:, None]], 1)
        res = self.wo(torch.relu(self.Wr(f))).squeeze(-1)
        if self.rs:
            cnt = torch.zeros(P.B).index_add(0, P.node_row, torch.ones_like(res))
            mu = torch.zeros(P.B).index_add(0, P.node_row, res) / cnt
            dv = res - mu[P.node_row]
            var = torch.zeros(P.B).index_add(0, P.node_row, dv * dv) / cnt
            res = torch.exp(self.log_c) * dv * torch.rsqrt(var[P.node_row] + 1.0)
        if self.tb is not None:
            res = self.tb * torch.tanh(res / self.tb)
        return P.z + res


def edge_drop(P, rng, top):
    """qd_gnn7.edge_drop, draw for draw, cutting P.align with the other edge keys."""
    u = rng.random(P.B) * top
    keep = rng.random(P.src.numel()) >= u[P.e_row.numpy()]
    kt = torch.from_numpy(keep)
    for k in EDGE_KEYS:
        setattr(P, k, getattr(P, k)[kt])
    return int((~keep).sum())


def family_drop(P, rng, p):
    """qd_gnn13.family_drop, draw for draw, cutting P.align with the other edge keys."""
    fam = np.minimum(P.fam.numpy(), 2)
    row = P.e_row.numpy()
    has = np.zeros((P.B, 3), dtype=bool)
    has[row, fam] = True
    drop = (rng.random((P.B, 3)) < p) & has
    for b in np.flatnonzero(has.any(1) & (drop == has).all(1)):
        drop[b, 0 if has[b, 0] else int(np.flatnonzero(has[b])[0])] = False
    kill = drop[row, fam]
    keep = torch.from_numpy(~kill)
    for k in EDGE_KEYS:
        setattr(P, k, getattr(P, k)[keep])
    return int(kill.sum())


# ── the arms ─────────────────────────────────────────────────────────────────


class QDArm15(Q13.QDArm13):
    """qd_gnn13's arms with -al on QD arms; qd_gnn2.run builds every 'qd'-family arm as QG.QDArm(...)."""

    def __new__(cls, sp, *args, **kwargs):
        if cls is QDArm15:
            if sp.get("w4"):
                return object.__new__(W4Arm15)
            if sp.get("w3"):
                return object.__new__(W3Arm15)
        return object.__new__(cls)

    def __init__(self, sp, Q, TOK, G, z_of, A16, n_id):
        super().__init__(sp, Q, TOK, G, z_of, A16, n_id)
        self.al_mode, self._al_pop = "on", None

    def make(self):
        o = self.sp.get("t7") or {}
        if not o.get("al"):
            return super().make()
        model = QD15(self.sp["kind"], self.sp["d"], self.sp["L"], self.n_id, g1=o.get("g1", False), tb=o.get("tb"),
                     rs=o.get("rs", False), lq=o.get("lq", False), at=o.get("at", False), rg=o.get("rg", 0))
        # qd_gnn13.QDArm13.make's generators, drawn as it draws them (lam is zeros: no draw at init)
        if self.p_row > 0 or self.p_edge > 0:
            self.rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        self.rng_ed = (np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item())) if o.get("ed") else None)
        self.rng_fd = (np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item())) if o.get("fd") else None)
        return model

    def align(self, model, g, rows):
        if model.training:
            pop, mode, tag = (g, train_rows(g), self.Q, self.TOK), "on", "train"
        else:
            pop, mode, tag = self._al_pop, self.al_mode, "read"
            if pop is None or pop[0] != g:
                raise SystemExit("-al: a read with no population (qd_gnn15 sets one around every read)")
        _g, prow, Qv, Tv = pop
        if Qv is not self.Q or Tv is not self.TOK:
            raise SystemExit("-al: the population is not the arm's current view")
        if mode == "off":
            return align_batch(None, rows, Qv, Tv, "off")
        fit = get_fit(g, prow, Qv, Tv, tag)
        if not fit["rowset"].issuperset(rows):
            raise SystemExit("-al: a batch row outside its population")
        return align_batch(fit, rows, Qv, Tv, mode)

    def forward(self, model, g, rows):
        o = self.sp.get("t7") or {}
        if not o.get("al"):
            return super().forward(model, g, rows)
        Qp, Tp = self.rows_for(rows)
        if Qp is not self.Q or Tp is not self.TOK:
            raise SystemExit("-al: rows_for changed the rows")
        P = QG.pack_qd(rows, Qp, Tp, self.z_of, self.sp["K"], self.nr)
        P.align = torch.from_numpy(self.align(model, g, rows))
        if P.align.numel() != P.src.numel():
            raise SystemExit("-al: the alignment does not match the pack's edges")
        if model.training and self.rng is not None:     # qd_gnn3.QDArmDrop.forward's label dropout, draw for draw
            drop = self.rng.random(P.B) < self.p_row
            de = drop[P.e_row.numpy()] | (self.rng.random(P.a.numel()) < self.p_edge)
            P.a = torch.where(torch.from_numpy(de), torch.full_like(P.a, -1), P.a)
        if model.training and self.rng_ed is not None:
            edge_drop(P, self.rng_ed, o["ed"])
        if model.training and o.get("fd"):
            family_drop(P, self.rng_fd, o["fd"])
        return QG.padded(P, model(P))

    def read(self, model, g, rows, rule):
        """qd_gnn.QDArm.read (the fit's select reads) with an -al arm's population set to the rows read."""
        if not al_arm(self):
            return super().read(model, g, rows, rule)
        keep = self._al_pop
        self._al_pop = (g, list(rows), self.Q, self.TOK)
        try:
            return super().read(model, g, rows, rule)
        finally:
            self._al_pop = keep


class W3Arm15(QDArm15, Q13.W3Arm13):
    """qd_gnn13.W3Arm13, unchanged (-al is refused on it), as a QDArm15."""


class W4Arm15(QDArm15, Q13.W4Arm13):
    """qd_gnn13.W4Arm13, unchanged (-al is refused on it), as a QDArm15."""


# ── reads, views and loading ─────────────────────────────────────────────────


_SCORE0 = Q5._score_rows      # qd_gnn2.score_rows, which qd_gnn5 / qd_gnn6 / qd_gnn7 all call through Q5._score_rows
_VIEWS0 = Q6x.views_for
_SVIEW0 = Q6x.score_view


def score_rows(arm, model, g, rows, TY=None):
    """qd_gnn2.score_rows with an -al arm's population set to the rows read, in the arm's current view."""
    if not al_arm(arm):
        return _SCORE0(arm, model, g, rows, TY)
    keep = arm._al_pop
    arm._al_pop = (g, list(rows), arm.Q, arm.TOK)
    try:
        return _SCORE0(arm, model, g, rows, TY)
    finally:
        arm._al_pop = keep


def views_for(arm, rows):
    """qd_gnn6's hold views; an -al arm adds the A0 / AS controls (the nr slot carries (nr, mode))."""
    v = _VIEWS0(arm, rows)
    if not al_arm(arm):
        return v
    out = dict(v)
    for base in ("FULL", "REV", "MASK"):
        if base in v:
            Qv, Tv, nr = v[base]
            out[f"{base}-A0"] = (Qv, Tv, (nr, "off"))
            out[f"{base}-AS"] = (Qv, Tv, (nr, "shuf"))
    if "FNR" in v:
        Qv, Tv, _nr = v["FNR"]
        out["FNR-A0"] = (Qv, Tv, (True, "off"))
    out["ID-A0"] = (arm.Q, arm.TOK, (False, "off"))
    return out


def score_view(arm, model, g, rows, Qv, Tv, nr):
    mode = "on"
    if isinstance(nr, tuple):
        nr, mode = nr
    keep = getattr(arm, "al_mode", "on")
    arm.al_mode = mode
    try:
        return _SVIEW0(arm, model, g, rows, Qv, Tv, nr)
    finally:
        arm.al_mode = keep


def parse_arms(spec, names, rule, AU, AG, G3):
    out = Q7.parse_arms(spec, names, rule, AU, AG, G3)
    AL["needed"] = any((sp.get("t7") or {}).get("al") for sp in out.values())
    return out


def load_all(trains, reads, max_len, t0, AU):
    Q, part, G = Q7.load_all7(trains, reads, max_len, t0, AU)
    if not AL["needed"]:
        return Q, part, G
    build = AL["builder"] or build_nodes
    for g in list(trains) + list(reads):
        if G[g]["kind"] != "kb":
            raise SystemExit(f"-al: {g} is a passage graph; qd_gnn15 aligns a KB's relation ids only")
        carves = (["x1", "select"] + list(trains[g])) if g in trains else [Q2.READ[g]]
        ids, Pn, info = build(g, carves, part, Q)
        AL["nodes"][g] = (ids, Pn)
        AL["node_info"][g] = info
        AL["vocab"][g] = list(G[g]["vocab"])
        log(f"al node table {g}: {info}")
    return Q, part, G


# ── binding and output ───────────────────────────────────────────────────────


_bind13 = Q13.bind
_finish13 = Q13.finish


def bind():
    _bind13()
    QG.QDArm = QDArm15
    Q7.split_opts = split_opts       # qd_gnn7.parse_arms splits names through it (qd_gnn13's bind set its own)
    Q2.parse_arms = parse_arms       # qd_gnn7's, then notes whether any arm aligns
    Q2.load_all = load_all           # qd_gnn7's, then the node tables
    Q5._score_rows = score_rows      # every read scores through it: qd_gnn5.score_rows, qd_gnn6's views, qd_gnn7's kalpha
    Q6x.views_for = views_for        # qd_gnn6.hold_reads calls both by module name
    Q6x.score_view = score_view


def finish(out):
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "qd_gnn15"
    res["pins"]["qd_gnn13"] = SHAS["qd_gnn13"]
    res["pins"]["reltype11"] = SHAS["reltype11"]
    res["qd_gnn15_sha256"] = SHAS["qd_gnn15"]
    res["align"] = ({"kappa": AL["kappa"], "tau": AL["tau"], "iters": AL["iters"], "clip": CLIP,
                     "types": "3 min(rank, R_TOP) + famdir, structural edges", "nodes": AL["node_info"],
                     "fits": AL["fit_log"]} if AL["needed"] else None)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def strip_al(argv):
    """(argv without --al-kappa / --al-tau, after setting them)."""
    out, k = [], 0
    while k < len(argv):
        a = argv[k]
        if a in ("--al-kappa", "--al-tau"):
            x = float(argv[k + 1])
            if (a == "--al-kappa" and not x > 0) or (a == "--al-tau" and x < 0):
                raise SystemExit(f"{a} {x}: kappa > 0, tau >= 0")
            AL["kappa" if a == "--al-kappa" else "tau"] = x
            k += 2
            continue
        out.append(a)
        k += 1
    return out


def main(selftest_run=False):
    if "--selftest" in sys.argv and not selftest_run:
        selftest()
        return
    sys.argv = strip_al(sys.argv)
    Q13.bind = bind                  # qd_gnn13.main hands its module's 'bind' to qd_gnn10 by name (Q10.bind = bind)
    Q13.finish = lambda out: (_finish13(out), finish(out))   # qd_gnn13.main's finish lambda calls Q13.finish by name
    Q13.main(selftest_run=selftest_run)


# ── selftest ─────────────────────────────────────────────────────────────────


ARMS_NEW = "QD-T0-L2-d16-at-al,QD-TXT-L2-d16-K8-pc-al,QD-TXT-L2-d16-K8-dr25e10-ed50-fd50-rg2-al"
ARMS_HOLD = "QD-T0-L2-d16-al,QD-TXT-L2-d16-K8-dr25e10-at-al,QD-T0-L2-d16-at"


def al_reset():
    AL.update(needed=False, kappa=20.0, tau=0.0, iters=RT.EM_ITERS, nodes={}, node_info={}, vocab={}, train_rows={},
              fit_log=[], builder=None)
    FITS.clear()


def synthetic_nodes(g, carves, part, Q):
    """A seeded random unit text per pool node of the loaded rows (the synthetic KB has no look chunks)."""
    rows = sorted({i for c in carves for i in part[(g, c)]})
    ids = np.unique(np.concatenate([np.asarray(Q[i]["pool"], dtype=np.int64) for i in rows]))
    Pn = RT.unit(np.random.default_rng(9).standard_normal((ids.size, RT.PROJ_DIM)).astype(np.float32))
    return ids, Pn, {"synthetic": True, "nodes": int(ids.size)}


def planted_rows(n_rows, rng, n_types=6, per_row=4):
    """Toy KB rows: a rank-1 seed (node 0) with per_row stored triples 0 -r-> k of distinct ranks, one of them asked;
    the query is the asked rank's direction through P's transpose plus noise, each far endpoint's text its rank's
    direction plus noise. Returns rows, the asked rank of each, the node table."""
    C = RT.unit(rng.standard_normal((n_types, RT.PROJ_DIM)))
    Pm = proj()
    Q, asked, ids, Pn = [], [], [], []
    for j in range(n_rows):
        n = 12
        ranks = rng.choice(n_types, per_row, replace=False)
        ask = int(rng.choice(ranks))
        hu, hv = np.zeros(per_row, dtype=np.int64), np.arange(1, per_row + 1)
        xu, xv = rng.integers(1, n, 6), rng.integers(1, n, 6)
        ok = xu != xv
        xu, xv = xu[ok], xv[ok]
        xr = rng.integers(0, n_types, xu.size)
        H, T_, R = np.r_[hu, xu], np.r_[hv, xv], np.r_[ranks, xr]
        u, v = np.r_[H, T_], np.r_[T_, H]
        fwd = np.r_[np.ones(H.size, bool), np.zeros(H.size, bool)]
        bwd = ~fwd
        w2f = np.where(fwd, np.r_[R, R], -1).astype(np.int32)
        w2b = np.where(bwd, np.r_[R, R], -1).astype(np.int32)
        seeds = np.full(10, -1, dtype=np.int64)
        seeds[0] = 0
        bucket = np.zeros(10, dtype=np.int64)
        qv = (C[ask] @ Pm.T) + 0.02 * rng.standard_normal(Pm.shape[0])
        pool = 100000 * j + np.arange(n)
        txt = RT.unit(rng.standard_normal((n, RT.PROJ_DIM)))
        txt[hv] = RT.unit(C[ranks] + 0.3 * rng.standard_normal((per_row, RT.PROJ_DIM)))
        Q.append({"n": n, "u": u, "v": v, "fam": np.zeros(u.size, dtype=np.int64), "fwd": fwd, "bwd": bwd,
                  "w": np.ones(u.size, dtype=np.float32), "w2_f": w2f, "w2_b": w2b, "seeds": seeds, "bucket": bucket,
                  "qemb": qv.astype(np.float32), "pool": pool, "gold": np.zeros(n, dtype=bool), "gt": 0})
        asked.append(ask)
        ids.append(pool)
        Pn.append(txt)
    return Q, asked, np.concatenate(ids).astype(np.int64), np.concatenate(Pn).astype(np.float32)


def selftest():
    import os
    import subprocess
    import tempfile
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    A16 = AU.A16
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    K = 8
    # names
    assert split_opts("QD-TXT-L3-dr25e10-at-al") == ("QD-TXT-L3-dr25e10", {"at": True, "al": True})
    assert split_opts("QD-T0-L3-al-at") == ("QD-T0-L3", {"at": True, "al": True})
    assert split_opts("QD-TXT-L2-dr25e10-tb5-ed50-al-fd25") == ("QD-TXT-L2-dr25e10", {"tb": 0.5, "ed": 0.5, "fd": 0.25,
                                                                                      "al": True})
    assert split_opts("QD-T0-L3-at") == ("QD-T0-L3", {"at": True})
    assert split_opts("W3-TXT-L2-dr25e10") == ("W3-TXT-L2-dr25e10", {})
    for bad in ("QD-T0-L2-al-al", "W3-T0-L2-al", "W4-TXT-L2-al", "QD-T0-L2-cap16-al", "W:T0-al"):
        try:
            split_opts(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    print("selftest: -al parses beside qd_gnn7's, qd_gnn10's and qd_gnn13's options; repeats, -cap and non-QD uses refused")
    # a population fit on synthetic rows, its values and the three modes
    Q = QG.synthetic_rows(24, rng, K)
    for j, q in enumerate(Q):
        q["pool"] = np.arange(q["n"]) + 1000 * j
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    phi = rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    rows = list(range(24))
    al_reset()
    ids_s = np.unique(np.concatenate([q["pool"] for q in Q]))
    AL["nodes"]["g"] = (ids_s, RT.unit(np.random.default_rng(3).standard_normal((ids_s.size, RT.PROJ_DIM))))
    fit = fit_pop("g", rows, Q, TOK, "selftest")
    on, off, sh = (align_batch(fit, rows, Q, TOK, md) for md in ("on", "off", "shuf"))
    assert on.shape == off.shape == sh.shape and not off.any() and np.isfinite(on).all() and on.std() > 0.05
    # each edge's value recomputed here from the fit: s_iz, its softmax over the population's T types, times T
    Xq = np.stack([Q[i]["qemb"] for i in rows]).astype(np.float64)
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
    Qh = Xq @ proj().astype(np.float64)
    Qh /= np.linalg.norm(Qh, axis=1, keepdims=True)
    S = fit["logpi"][None, :] + AL["kappa"] * (Qh @ fit["mu"].astype(np.float64).T)
    post = np.exp(S - S.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    Aref = np.log(post) + fit["logT"]
    assert abs(float(np.exp(fit["logpi"]).sum()) - 1.0) < 1e-9 and (np.abs(on) < CLIP - 1e-3).mean() > 0.05
    k0 = 0
    for b, i in enumerate(rows):
        e = Q[i]["u"].size
        m = Q[i]["fam"] == 0
        assert not on[k0:k0 + e][~m].any() and not sh[k0:k0 + e][~m].any()
        zi = fit["zmap"][types_of(TOK[i][0][m], TOK[i][1][m])]
        assert np.allclose(on[k0:k0 + e][m], np.clip(Aref[b, zi], -CLIP, CLIP), atol=1e-4), i
        for zz in np.unique(zi):     # one value per type in a row, under on and under shuf
            assert np.unique(on[k0:k0 + e][m][zi == zz]).size == 1 and np.unique(sh[k0:k0 + e][m][zi == zz]).size == 1
        assert np.allclose(np.sort(np.unique(on[k0:k0 + e][m])), np.sort(np.unique(sh[k0:k0 + e][m])))
        k0 += e
    print(f"selftest: a population fit ({fit['T']} types) gives one value per (row, type), a softmax over the "
          f"population's types times T, 0 off structural edges; off is zeros, shuf rotates a row's per-type values")
    # the planted toy: the fitted alignment ranks each row's asked type first among its seed edges
    al_reset()
    Qt, asked, ids_t, Pn_t = planted_rows(240, np.random.default_rng(4))
    order = np.argsort(ids_t)
    AL["nodes"]["toy"] = (ids_t[order], RT.unit(Pn_t[order]))
    TOKt = [QG.edge_labels(q) for q in Qt]
    rows_t = list(range(len(Qt)))
    fit_t = fit_pop("toy", rows_t, Qt, TOKt, "selftest")
    at_ = align_batch(fit_t, rows_t, Qt, TOKt, "on")
    hits, k0 = [], 0
    for i in rows_t:
        e = Qt[i]["u"].size
        out_seed = (Qt[i]["u"] == 0) & Qt[i]["fwd"]
        vals, rk = at_[k0:k0 + e][out_seed], Qt[i]["w2_f"][out_seed]
        hits.append(int(rk[int(np.argmax(vals))]) == asked[i])
        k0 += e
    lg = AL["fit_log"][-1]["loglik"]
    assert np.mean(hits) > 0.9 and all(b >= a - 1e-4 for a, b in zip(lg, lg[1:])), (np.mean(hits), lg)
    AL["tau"] = 5.0
    fit_tau = fit_pop("toy", rows_t, Qt, TOKt, "selftest-tau")
    AL["tau"] = 0.0
    assert fit_tau["T"] == fit_t["T"] and not np.allclose(fit_tau["mu"], fit_t["mu"])
    print(f"selftest: on a planted toy the fitted alignment puts the asked type first in {np.mean(hits):.3f} of rows "
          f"(chance 0.25); EM's log-likelihood never falls; --al-tau changes the fit")
    # the model: lam = 0 is QD13 bit for bit; fitted, the frontier cut is exact and gradients reach lam
    al_reset()
    AL["nodes"]["g"] = (ids_s, RT.unit(np.random.default_rng(3).standard_normal((ids_s.size, RT.PROJ_DIM))))
    fit = fit_pop("g", rows, Q, TOK, "selftest")
    P = QG.pack_qd(rows, Q, TOK, z_of, K)
    P.align = torch.from_numpy(align_batch(fit, rows, Q, TOK, "on"))
    for kind in ("T0", "TXT"):
        for L in (1, 2, 3):
            kws = [{}, {"at": True}, {"rg": 2}, {"at": True, "rg": 2}, {"g1": True, "tb": 0.5, "rs": True}]
            if kind == "TXT":
                kws.append({"lq": True, "at": True})
            for kw in kws:
                torch.manual_seed(1)
                m13 = Q13.QD13(kind, 16, L, K, **kw)
                torch.manual_seed(1)
                m15 = QD15(kind, 16, L, K, **kw)
                s13 = m13.state_dict()
                s15 = m15.state_dict()
                assert set(s15) - set(s13) == ({"lam_g", "lam_at"} if kw.get("at") else {"lam_g"}), kw
                assert all(torch.equal(s13[k], s15[k]) for k in s13), f"{kind} L{L} {kw}: the draws differ"
                with torch.no_grad():
                    torch.manual_seed(2)
                    for _n, p in m13.named_parameters():
                        p.add_(0.3 * torch.randn_like(p))
                m15.load_state_dict(m13.state_dict(), strict=False)
                for m in (m13, m15):
                    m.set_phi(phi, K)
                with torch.no_grad():
                    assert torch.equal(m15(P), m13(P)), f"{kind} L{L} {kw}: lam 0 is not QD13"
                    m15.lam_g.normal_(0, 0.5)
                    if kw.get("at"):
                        m15.lam_at.normal_(0, 0.5)
                    assert not torch.equal(m15(P), m13(P)), f"{kind} L{L} {kw}: lam does nothing"
                s_full = m15(P)
                Qc = []
                for q in Q:
                    mk = QG.frontier_mask(q["u"], q["v"], q["seeds"], q["n"], L)
                    qq = {**q, **{k: q[k][mk] for k in ("u", "v", "fam", "fwd", "bwd", "w")}}
                    keep_struct = mk[q["fam"] == 0]
                    qq["w2_f"], qq["w2_b"] = q["w2_f"][keep_struct], q["w2_b"][keep_struct]
                    Qc.append(qq)
                TOKc = [QG.edge_labels(q) for q in Qc]
                Pc = QG.pack_qd(rows, Qc, TOKc, z_of, K)
                Pc.align = torch.from_numpy(align_batch(fit, rows, Qc, TOKc, "on"))
                with torch.no_grad():
                    s_cut = m15(Pc)
                err = float((s_full.detach() - s_cut).abs().max())
                assert err < 1e-5, f"{kind} L{L} {kw}: the frontier cut changes the scores by {err}"
                s_pad, gold, z = QG.padded(P, s_full)
                QG.step_loss(s_pad, gold, z, "np").backward()
                for nm_, p in m15.named_parameters():
                    assert p.grad is None or bool(torch.isfinite(p.grad).all()), f"{kind} L{L} {kw}: {nm_} grad not finite"
                assert float(m15.lam_g.grad.abs().sum()) > 0, "no gradient reaches lam_g"
                if kw.get("at"):
                    assert float(m15.lam_at.grad.abs().sum()) > 0, "no gradient reaches lam_at"
            print(f"selftest QD15 {kind} L{L}: QD13's draws, QD13 bit for bit at lam 0 under {len(kws)} option sets, "
                  f"frontier cut exact ({err:.1e}), gradients reach lam")
    # the drops: the same edges as qd_gnn7's and qd_gnn13's, the alignment cut with them
    for fn15, fn0, arg in ((edge_drop, Q7.edge_drop, 0.6), (family_drop, Q13.family_drop, 0.5)):
        Pa, Pb = QG.pack_qd(rows, Q, TOK, z_of, K), QG.pack_qd(rows, Q, TOK, z_of, K)
        eid = torch.arange(Pa.src.numel(), dtype=torch.float32)
        Pa.w, Pb.align = eid.clone(), eid.clone()
        na, nb = fn0(Pa, np.random.default_rng(5), arg), fn15(Pb, np.random.default_rng(5), arg)
        assert na == nb > 0 and torch.equal(Pa.w, Pb.align)
        assert all(torch.equal(getattr(Pa, k), getattr(Pb, k)) for k in ("src", "dst", "e_row", "d", "a", "fam"))
    print("selftest: the drops keep qd_gnn7's and qd_gnn13's edges draw for draw and cut a_e with them")
    # the arms: no -al = qd_gnn13's arm, draw for draw; -al draws as its twin; an -al arm fits and moves lam
    G = {"g": {"phi": phi, "kind": "passage"}}
    bind()
    al_reset()
    AL["nodes"]["g"] = (ids_s, RT.unit(np.random.default_rng(3).standard_normal((ids_s.size, RT.PROJ_DIM))))
    for nm in ("QD-TXT-L2-d16-K8-dr25e10-tb5-ed50", "QD-T0-L2-d16-at-fd50", "QD-TXT-L2-d16-K8-pc-rg2"):
        sp = Q2.parse_arms(nm, ["g"], "np", AU, AU.AG, AU.G3)[nm]
        a13, a15 = Q13.QDArm13(sp, Q, TOK, G, z_of, A16, 0), QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        assert type(a15) is QDArm15 and not AL["needed"]
        m13_, e13, c13, b13 = Q12._FIT2(a13, ["g"], {"g": rows[:16]}, {"g": rows[16:]}, "both", 2, 2e-3, 1e-4, 0, "graph", AU)
        m15_, e15, c15, b15 = Q12._FIT2(a15, ["g"], {"g": rows[:16]}, {"g": rows[16:]}, "both", 2, 2e-3, 1e-4, 0, "graph", AU)
        assert (c13, e13, b13) == (c15, e15, b15) and type(m13_) is type(m15_), nm
        assert all(torch.equal(x, y) for x, y in zip(m13_.state_dict().values(), m15_.state_dict().values())), nm
    for nm in ("W3-TXT-L2-d16-K8", "W4-T0-L2-d16"):
        spw = Q2.parse_arms(nm, ["g"], "np", AU, AU.AG, AU.G3)[nm]
        assert type(QG.QDArm(spw, Q, TOK, G, z_of, A16, 0)) is (W4Arm15 if spw.get("w4") else W3Arm15), nm
    print("selftest: with no -al QDArm15 fits as qd_gnn13's arm, curve and weights bit for bit; W3 / W4 arms map")
    for twin in ("QD-T0-L2-d16-at-fd50", "QD-TXT-L2-d16-K8-dr25e10-ed50"):
        spec = Q2.parse_arms(f"{twin},{twin}-al", ["g"], "np", AU, AU.AG, AU.G3)
        assert AL["needed"]
        a0, a1 = (QG.QDArm(spec[n], Q, TOK, G, z_of, A16, 0) for n in (twin, f"{twin}-al"))
        torch.manual_seed(5)
        m0 = a0.make()
        torch.manual_seed(5)
        m1 = a1.make()
        s0, s1 = m0.state_dict(), m1.state_dict()
        assert isinstance(m1, QD15) and all(torch.equal(s0[k], s1[k]) for k in s0) and set(s1) - set(s0) <= {"lam_g", "lam_at"}
        for gen in ("rng", "rng_ed", "rng_fd"):
            g0, g1 = getattr(a0, gen), getattr(a1, gen)
            assert (g0 is None) == (g1 is None) and (g0 is None or g0.random() == g1.random()), (twin, gen)
    spa = Q2.parse_arms("QD-TXT-L2-d16-K8-dr25e10-ed50-fd50-at-al", ["g"], "np", AU, AU.AG, AU.G3)
    aa = QG.QDArm(spa["QD-TXT-L2-d16-K8-dr25e10-ed50-fd50-at-al"], Q, TOK, G, z_of, A16, 0)
    AL["train_rows"]["g"] = rows[:16]
    ma, ea, ca, ba = Q12._FIT2(aa, ["g"], {"g": rows[:16]}, {"g": rows[16:]}, "both", 2, 2e-3, 1e-4, 0, "graph", AU)
    assert isinstance(ma, QD15) and hasattr(ma, "lam_at")
    tags = [r_["tag"] for r_ in AL["fit_log"]]
    assert tags.count("train") == 1 and tags.count("read") == 1, tags
    # QD's output head starts at zero, so lam's first gradient comes a step later: one training step from moved weights
    with torch.no_grad():
        torch.manual_seed(4)
        for p in ma.parameters():
            p.add_(0.3 * torch.randn_like(p))
    ma.train()
    aa.prepare(ma, "g")
    ma.zero_grad()
    s_t, gold_t, z_t = aa.forward(ma, "g", rows[:16])
    QG.step_loss(s_t, gold_t, z_t, "np").backward()
    g_lam = (float(ma.lam_g.grad.abs().sum()), float(ma.lam_at.grad.abs().sum()))
    assert min(g_lam) > 0, g_lam
    aa._al_pop = None
    ma.eval()
    try:
        aa.forward(ma, "g", rows[16:])
        raise AssertionError("a read with no population ran")
    except SystemExit:
        pass
    print(f"selftest: -al draws as its twin (weights and generators); an -al arm fits (select {ca}), fits its training "
          f"and select populations once each, trains lam (|grad| lam_g {g_lam[0]:.2e}, lam_at {g_lam[1]:.2e}) and refuses "
          f"a read with no population")
    # end to end, each in a fresh process: no -al runs as qd_gnn13 bit for bit; -al arms run; the hold adds A0 / AS
    with tempfile.TemporaryDirectory() as d:
        res = {}
        for mode, script in (("plain13", Q13.__file__), ("plain", __file__), ("new", __file__), ("hold", __file__)):
            out = Path(d) / f"{mode}.json"
            r = subprocess.run([sys.executable, script, "--selftest-run", "plain" if mode == "plain13" else mode, str(out)],
                               env=dict(os.environ), capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                print(r.stdout[-4000:], r.stderr[-4000:])
                raise AssertionError(f"selftest run {mode} failed")
            res[mode] = json.loads(out.read_text(encoding="utf-8"))

        def reads(r):
            return {a: {s: sv["reads"] for s, sv in v["seeds_read"].items()} for a, v in r["results"].items()}

        def states(mode):
            return [torch.load(f, weights_only=False)["state_dict"] for f in sorted((Path(d) / f"{mode}_models").glob("a*_s*.pt"))]

        def same(a, b):
            return len(a) == len(b) and all(x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x) for x, y in zip(a, b))
        assert reads(res["plain"]) == reads(res["plain13"]), "with no -al the run must read as qd_gnn13's"
        assert same(states("plain"), states("plain13")) and len(states("plain")) == 2
        assert res["plain"]["look"] == "qd_gnn15" and res["plain"]["align"] is None
        assert res["plain"]["pins"]["qd_gnn13"] == SHAS["qd_gnn13"] and res["plain"]["pins"]["reltype11"] == SHAS["reltype11"]
        assert set(res["new"]["results"]) == set(ARMS_NEW.split(",")), set(res["new"]["results"])
        st = states("new")
        assert len(st) == 3 and all("lam_g" in s_ for s_ in st) and any("lam_at" in s_ for s_ in st)
        fits = res["new"]["align"]["fits"]
        tags = {f_["tag"] for f_ in fits}
        assert tags == {"train", "read"} and res["new"]["align"]["nodes"]["metaqa"]["synthetic"]
        assert res["new"]["kalpha"]["reads"] and "_failed" not in res["new"]["kalpha"]["reads"]
        hold = res["hold"]["hold"]["reads"]
        assert "_failed" not in hold and set(hold) == set(ARMS_HOLD.split(",")), set(hold)

        def views(arm):
            return {k.split("/")[0] for k in hold[arm][0]["all"] if " - " not in k}
        a_t0, a_txt, a_plain = ARMS_HOLD.split(",")
        assert views(a_t0) == {"ID", "FULL", "FULL-A0", "FULL-AS", "ID-A0"}, views(a_t0)
        assert views(a_txt) == {"ID", "REV", "MASK", "FNR", "REV-A0", "REV-AS", "MASK-A0", "MASK-AS", "FNR-A0", "ID-A0"}, views(a_txt)
        assert views(a_plain) == {"ID", "FULL"}, views(a_plain)
        assert res["hold"]["align"]["tau"] == 2.0 and res["hold"]["hold"]["ranks"] == [1, 5]
        print(f"selftest e2e: no -al runs as qd_gnn13 bit for bit (reads and saved models); the -al arms run "
              f"({len(fits)} population fits); the hold reads {sorted(views(a_txt))} on the text arm")
    print("selftest ok")


def selftest_run(mode, out):
    """One end-to-end run on qd_gnn11's synthetic KB (a subprocess, so every bind starts fresh)."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    phi = np.random.default_rng(7).standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    names = [f"rel {j}" for j in range(Q11.K_ST)]

    def fake(trains, reads, max_len, t0, AU_):
        Qf, part = Q11.synthetic()
        Gf = {"metaqa": {"kind": "kb", "role": "train", "phi": phi, "phi_sha": "synthetic", "n_rel": Q11.K_ST, "vocab": names,
                         "info": {"pruned_loader": {"synthetic": True}}}}
        return Qf, part, Gf

    Q4._load_all = fake
    AL["builder"] = synthetic_nodes
    arms = {"plain": Q11.ARMS_ST, "new": ARMS_NEW, "hold": ARMS_HOLD}[mode]
    sys.argv = ["qd_gnn15.py", "--train", "metaqa=fit", "--arms", arms, "--rule", "np", "--epochs", "2", "--out", out,
                "--kalpha", "4,8", "--kalpha-draws", "2"]
    if mode == "hold":
        sys.argv += ["--hold", "metaqa:drop:rel 1;rel 5", "--al-tau", "2"]
    main(selftest_run=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--selftest-run":
        selftest_run(sys.argv[2], sys.argv[3])
    else:
        main()
