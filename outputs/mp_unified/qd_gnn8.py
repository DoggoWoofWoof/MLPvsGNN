"""Design look (untracked; not a result and not filed): qd_gnn7.py, pinned and unchanged, with two arm families over
latent edge relations learned by EM, fitted in the same loop as every other arm, on the same rows, batches, loss,
selection and reads.

Latent relations (EdgeTyper). Every pool edge e has a latent type z_e in {0..k-1}. Its prior reads the edge's own
features only, never a node state, a neighbour or the query:
    x_e   = [Px phi[a_e] (zero without a readable label), onehot(d_e), has_label, clip(w_e, -5, 5)]
            (+ log1p of the pool's out-degree of u and in-degree of v with -deg)
    pi_e  = softmax(f2 relu(f1 x_e))
and each type has a query-conditioned relevance
    rho(q, z) = <LN(Wq q), E_r[z]> + b_r[z],        r(q, e) = sum_z pi_e(z) sigmoid(rho(q, z))
so r ranks a row's edges for its query from their features alone. EM fits both on the training rows (online: one E
and one M step per training batch, never at a read). The evidence is a training-only target no score reads:
    o_e = 1 when e lies on a shortest walk of at most L edges from a seed to a gold (per gold; the GMAX golds nearest
          a seed), over the edges whose source is within L - 1 edges of a seed (the ones a walk or a layer can use)
    E-step   q_e(z)  ~  pi_e(z) * theta_g(a_e | z)^lambda_t * Bern(o_e | sigmoid(rho(q, z)))^mu      (no gradient)
    M-step   theta_g: per graph, a decayed count of q over the edges' true labels (closed form; training only)
             loss    omega * mean_e [ -sum_z q_e(z) log pi_e(z)  -  sum_z q_e(z) log Bern_bal(o_e | sigmoid(rho)) ]
Bern_bal weights the on- and off-path edges to half the batch each. lambda_t = lambda0 max(0, 1 - ep / E_an): the
labels supervise the types first (the edges of one label pulled to one type), then are annealed away while the E and
M steps go on, so at the end the types are what the on-path evidence and the features support. A label reaches the
prior only as text: a held, unseen or 'other' label (NR, a dropped label) is no label, and structure alone types the
edge. -dr<R>e<E> hides the text from the prior in training (qd_gnn3's draws); the emission keeps the true label.

W3EM-z<k>[-L<l>][-d<dim>] (no message passing). qd_gnn4's W3 with each step's token (d_e, argmax_z pi_e) and the
chain logit plus the EM edge ranking of its steps' types:
    w_t = <A qn, e'_t> + c[b, len] + eta * sum_j log sigmoid(rho(q, z_j))
The types are a per-edge function of the edge's own features (an index-time table); the walk types and reach sets are
compiled from them and the row's graph, and p reads q and the row's code list: no node's state or score reaches
another node. The typer is fitted by EM only (no score gradient reaches it; with -jt the score's reaches rho).
-lg scores s_v = z_v + kappa log1p(boost_v / 0.05); -ki<x> starts kappa at x / 10.

QDEM-z<k>[-L<l>][-d<dim>] (message passing). A relational GAT over the latent types (the relation acts on the message
as a diagonal map, as in CompGCN's composition), as a residual on the twin's z (qd_gnn.QD's start, state rule and
head; an unfitted QDEM scores z exactly):
    r_e     = E_fd[d_e] + sum_z pi_e(z) E_z[z]
    w_e     = R_l r_e * (1 + tanh(G_l qn))
    l_e     = <A_l qn, r_e> + <Kq_l qn, Kh_l rms(h_u)> / sqrt(dim) + c_l[fam] w + b_l + eta_l log r(q, e)
    alpha_e = softmax of l_e over v's in-edges whose source has a state, times n_v^gamma (n_v: those edges)
    h_v    <- h_v + relu(W_l sum_e alpha_e (V_l rms(h_u)) * w_e)
-noatt: qd_gnn.QD's gate 2 sigmoid(l_e) / (1 + n_v)^gamma (no key term). pi and rho reach the layers detached unless
-jt. qd_gnn7's -tb<T>, -rs, -cap<K> and -ed<P> apply.

Options (both): -lam<x> lambda0 = x/10 (10), -an<E> (default ceil(epochs / 2)), -mu<x> (10), -om<x> (10), -noem (no E
or M step: the typer is fitted by the score's gradient alone; -jt implied; a W3EM's prior then never moves), -jt,
-deg, -noeta (no relevance term in the score), -dr<R>e<E>.

Reads: all of qd_gnn7's (ID / NR / SHUF, every --read graph, --hold, --kalpha) and, after each graph's ID read,
em_diag on up to DIAG_ROWS of its read rows: per row, the AUC of r(q, e) against o_e over the frontier edges, beside
the query-blind sum_z pi_e(z) sigmoid(b_r[z]) and r with every label hidden; the type usage. Per epoch (em trace):
lambda, the posterior and prior entropies, the on-path fraction, how often argmax q and argmax pi agree, and per graph
the NMI between labels and types (decayed posterior counts; the prior's argmax): how far the types moved from labels.

    python outputs/mp_unified/qd_gnn8.py --selftest
    python outputs/mp_unified/qd_gnn8.py --train 2wiki=x4 --read hotpotqa,musique,metaqa,webqsp \
        --arms W3EM-z8-L2,QDEM-z8-L2-tb5-ed50 --rule both --cap-len musique=2 --out outputs/mp_unified/qd/em-2w.json
"""
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn7 as Q7  # noqa: E402  (imports qd_gnn6 ... qd_gnn, which sets the BLAS thread counts before numpy loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

Q6x, Q5 = Q7.Q6x, Q7.Q5
Q4, Q3, Q2, QG = Q7.Q4, Q7.Q3, Q7.Q2, Q7.QG
SHAS = {**Q7.SHAS, "qd_gnn8": QG.sha(__file__)}   # at import: what ran
NAMERE = re.compile(r"(W3EM|QDEM)(?:-z(\d+))?(?:-L([123]))?(?:-d(\d+))?((?:-[a-z0-9]+)*)")
OPTRE = re.compile(r"(lam|an|mu|om|ki|tb|ed|cap)(\d+)|dr(\d+)e(\d+)|(noem|jt|deg|noeta|lg|noatt|rs)")
KLAB = 256        # the label ranks the typer's text reads (qd_gnn's K)
GMAX = 64         # golds per row the on-path target follows (the nearest to a seed)
DIAG_ROWS = 400
EMCFG = {"epochs": 12}
PENDING = {}      # the training batch's M-step loss, added to its step loss
TRACE = {}        # arm name -> one per-epoch list per fit
EMDIAG = {}       # arm name -> one record per (fit, graph) ID read
COST = {}         # arm name -> read-time typer and compile cost (W3EM)
EXTRA = ("a_true", "du", "dv", "em_on", "em_mask")
_STEP2 = Q2.step_loss2
_SCORE = {"f": None}
log = QG.log


# ── arm names ────────────────────────────────────────────────────────────────


def parse_em(nm):
    m = NAMERE.fullmatch(nm)
    if not m:
        raise SystemExit(f"{nm}: (W3EM|QDEM)[-z<types>][-L<1..3>][-d<dim>][-option...]")
    fam = m.group(1)
    k = int(m.group(2) or 8)
    if not 1 <= k <= 64:
        raise SystemExit(f"{nm}: -z between 1 and 64 (z1: one latent type, the no-relation ablation)")
    L = int(m.group(3) or 3)
    o = {"arm": fam.lower(), "k": k, "lam": 1.0, "an": None, "mu": 1.0, "om": 1.0, "em": True, "jt": False,
         "deg": False, "eta": True, "ki": 1.0, "lg": False, "att": True}
    t7, sp = {}, {"family": "qd", "kind": "TXT", "L": L, "d": int(m.group(4) or 64), "K": KLAB, "max_len": L}
    seen = set()
    for tok in [t for t in m.group(5).split("-") if t]:
        mo = OPTRE.fullmatch(tok)
        if not mo:
            raise SystemExit(f"{nm}: unknown option -{tok}")
        key = mo.group(1) or ("dr" if mo.group(3) else mo.group(5))
        if key in seen:
            raise SystemExit(f"{nm}: -{key} given twice")
        seen.add(key)
        if mo.group(1):
            n = int(mo.group(2))
            if key in ("lam", "mu", "om"):
                o[key] = n / 10.0
            elif key == "an":
                o["an"] = n
            elif key == "ki":
                if n <= 0:
                    raise SystemExit(f"{nm}: -ki above 0")
                o["ki"] = n / 10.0
            else:
                if (key == "tb" and n <= 0) or (key == "ed" and not 0 < n < 100) or (key == "cap" and n < 1):
                    raise SystemExit(f"{nm}: -{tok} out of range (tb > 0, 0 < ed < 100, cap >= 1)")
                t7[key] = {"tb": n / 10.0, "ed": n / 100.0, "cap": n}[key]
        elif mo.group(3):
            sp["drop_row"], sp["drop_edge"] = int(mo.group(3)) / 100.0, int(mo.group(4)) / 100.0
        elif key == "noem":
            o["em"], o["jt"] = False, True
        elif key == "noeta":
            o["eta"] = False
        elif key == "noatt":
            o["att"] = False
        elif key == "rs":
            t7["rs"] = True
        else:
            o[key] = True
    if fam == "W3EM":
        bad = sorted(set(t7) | ({"noatt"} & seen))
        if bad:
            raise SystemExit(f"{nm}: {bad} are QDEM options")
        sp["w3"] = True
    elif {"ki", "lg"} & seen:
        raise SystemExit(f"{nm}: -ki and -lg are W3EM options")
    if not o["em"] and seen & {"lam", "an", "mu", "om"}:
        raise SystemExit(f"{nm}: -noem takes no EM option")
    sp["em8"] = o
    if t7:
        sp["t7"] = t7
    return sp


def parse_arms(spec, names, rule, AU, AG, G3):
    full = spec.split(",")
    if len(set(full)) != len(full):
        raise SystemExit(f"--arms {spec}: each arm once")
    em = {nm: parse_em(nm) for nm in full if nm.startswith(("W3EM", "QDEM"))}
    rest = [nm for nm in full if nm not in em]
    base = _PARSE["f"](",".join(rest), names, rule, AU, AG, G3) if rest else {}
    out = {}
    for nm in full:
        out[nm] = em[nm] if nm in em else base[nm]
        Q5.NAMES[id(out[nm])] = nm
    return out


_PARSE = {"f": None}


# ── the typer and its EM ─────────────────────────────────────────────────────


class EdgeTyper(torch.nn.Module):
    """pi_e from the edge's own features; rho(q, z) from the query."""

    def __init__(self, k, deg=False, dx=32, hid=32, dq=32):
        super().__init__()
        self.k, self.deg, self.dx = k, bool(deg), dx
        self.Px = torch.nn.Linear(QG.TDIM, dx, bias=False)
        self.f1 = torch.nn.Linear(dx + 5 + 2 + (2 if self.deg else 0), hid)
        self.f2 = torch.nn.Linear(hid, k)
        self.Wq = torch.nn.Linear(QG.QDIM, dq)
        self.lnq = torch.nn.LayerNorm(dq)
        self.E_r = torch.nn.Parameter(torch.randn(k, dq) * 0.3)
        self.b_r = torch.nn.Parameter(torch.full((k,), -2.0))
        self.register_buffer("phi", torch.zeros(0, QG.TDIM), persistent=False)

    def set_phi(self, phi, K):
        """qd_gnn.QD.set_phi's table: the graph's first K label texts, unit-normalised."""
        t = torch.as_tensor(np.asarray(phi[:K], dtype=np.float32))
        self.phi = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12) * math.sqrt(QG.TDIM)

    def logits(self, d, a, w, du=None, dv=None):
        has = a >= 0
        if self.phi.shape[0] and bool(has.any()):
            T = self.Px(self.phi)
            t = T[a.clamp(0, T.shape[0] - 1)] * has[:, None].to(T.dtype)
        else:
            t = torch.zeros(d.numel(), self.dx)
        x = [t, F.one_hot(d, 5).to(t.dtype), has.to(t.dtype)[:, None], w.clamp(-5.0, 5.0)[:, None]]
        if self.deg:
            x += [torch.log1p(du)[:, None] / 3.0, torch.log1p(dv)[:, None] / 3.0]
        return self.f2(torch.relu(self.f1(torch.cat(x, 1))))

    def rel(self, qemb):
        return self.lnq(self.Wq(qemb)) @ self.E_r.T + self.b_r                    # (B, k)


def nmi(C):
    C = C.double()
    tot = float(C.sum())
    if tot <= 0:
        return None
    P = C / tot
    pz, pa = P.sum(1), P.sum(0)
    nz = P > 0
    I = float((P[nz] * torch.log(P[nz] / (pz[:, None] * pa[None, :])[nz])).sum())
    Hz = float(-(pz[pz > 0] * torch.log(pz[pz > 0])).sum())
    Ha = float(-(pa[pa > 0] * torch.log(pa[pa > 0])).sum())
    return round(I / math.sqrt(Hz * Ha), 4) if Hz > 0 and Ha > 0 else 0.0


class EMCore:
    """The EM state of one fit: the per-graph label emission counts, the epoch, the per-epoch trace."""

    def __init__(self, k):
        self.k = k
        self.reset()

    def reset(self):
        self.ep, self.trained, self.N, self.trace = 0, False, {}, []
        self.acc = {"edges": 0, "on": 0.0, "Hq": 0.0, "Hp": 0.0, "agree": 0.0, "Cp": {}}

    def lam(self, o):
        an = o["an"] if o["an"] is not None else math.ceil(EMCFG["epochs"] / 2)
        return o["lam"] * max(0.0, 1.0 - self.ep / an) if an > 0 else 0.0

    def log_theta(self, g):
        N = self.N.get(g)
        if N is None:
            return torch.zeros(self.k, KLAB)
        T = N + 0.1
        return torch.log(T / T.sum(1, keepdim=True)).float()

    def update(self, g, q, a):
        S = torch.zeros(self.k, KLAB, dtype=torch.float64).index_add_(1, a, q.T.double())
        self.N[g] = S if g not in self.N else 0.9 * self.N[g] + S

    def add(self, g, q, lp, on, at):
        A = self.acc
        A["edges"] += int(q.shape[0])
        A["on"] += float(on.sum())
        A["Hq"] += float(-(q * torch.log(q.clamp_min(1e-12))).sum())
        A["Hp"] += float(-(lp.exp() * lp).sum())
        zp = lp.argmax(1)
        A["agree"] += float((q.argmax(1) == zp).sum())
        has = at >= 0
        if bool(has.any()):
            C = A["Cp"].setdefault(g, torch.zeros(self.k, KLAB, dtype=torch.float64))
            C.index_put_((zp[has], at[has]), torch.ones(int(has.sum()), dtype=torch.float64), accumulate=True)

    def tick(self, o):
        """At the first read after a training pass: close the epoch's trace."""
        if not self.trained:
            return
        A, n = self.acc, max(1, self.acc["edges"])
        if A["edges"]:
            rec = {"epoch": self.ep, "lambda": round(self.lam(o), 4), "edges": A["edges"], "on_path": round(A["on"] / n, 5),
                   "H_post": round(A["Hq"] / n, 4), "H_prior": round(A["Hp"] / n, 4), "agree": round(A["agree"] / n, 4),
                   "nmi_post": {g: nmi(N) for g, N in self.N.items()}, "nmi_prior": {g: nmi(C) for g, C in A["Cp"].items()}}
            self.trace.append(rec)
            log(f"  em ep {self.ep}: lambda {rec['lambda']} on-path {rec['on_path']} H post {rec['H_post']} prior "
                f"{rec['H_prior']} agree {rec['agree']} nmi post {rec['nmi_post']} prior {rec['nmi_prior']}")
        self.ep += 1
        self.trained = False
        self.acc = {"edges": 0, "on": 0.0, "Hq": 0.0, "Hp": 0.0, "agree": 0.0, "Cp": {}}


def onpath_row(q, L):
    """(o, frontier) per pool edge: o, the edge lies on a shortest walk of at most L edges from a seed to a gold (each
    of the GMAX golds nearest a seed); frontier, its source is within L - 1 edges of a seed (qd_gnn.frontier_mask)."""
    n = int(q["n"])
    u, v = q["u"].astype(np.int64), q["v"].astype(np.int64)
    fr = QG.frontier_mask(u, v, q["seeds"], n, L)
    INF = L + 1
    ds = np.full(n, INF, dtype=np.int64)
    ds[np.unique(q["seeds"][q["seeds"] >= 0]).astype(np.int64)] = 0
    for k in range(L):
        src = ds[u] == k
        if not src.any():
            break
        nv = v[src]
        ds[nv] = np.minimum(ds[nv], k + 1)
    G = np.flatnonzero(q["gold"])
    G = G[ds[G] <= L]
    if G.size == 0 or u.size == 0:
        return np.zeros(u.size, dtype=bool), fr
    if G.size > GMAX:
        G = G[np.argsort(ds[G], kind="stable")[:GMAX]]
    dg = np.full((G.size, n), INF, dtype=np.int64)
    dg[np.arange(G.size), G] = 0
    for k in range(L):
        jj, ee = np.nonzero(dg[:, v] == k)
        if jj.size == 0:
            break
        np.minimum.at(dg, (jj, u[ee]), k + 1)
    on = ((ds[u][None, :] + 1 + dg[:, v]) == ds[G][:, None]).any(0)
    return on, fr


def em_targets(arm, rows, Qp):
    on, fr = [], []
    for i in rows:
        q = Qp[i]
        c = arm.onp.get(i)
        if c is None or c[0] is not q["u"]:     # cached per row, compactly (a row's pool can hold 10^4+ edges)
            o_, f_ = onpath_row(q, arm.sp["L"])
            c = arm.onp[i] = (q["u"], np.flatnonzero(o_), np.packbits(f_), o_.size)
        o_ = np.zeros(c[3], dtype=bool)
        o_[c[1]] = True
        on.append(o_)
        fr.append(np.unpackbits(c[2], count=c[3]).astype(bool))
    return torch.from_numpy(np.concatenate(on)), torch.from_numpy(np.concatenate(fr))


def em_step(arm, model, g, E):
    """One E step (no gradient) and the M step's loss on the batch's frontier edges; theta's M step in closed form."""
    o = arm.em
    if E is None or not bool(E.em_mask.any()):
        return None
    ty, m = model.typer, E.em_mask
    du = E.du[m] if E.du is not None else None
    dv = E.dv[m] if E.dv is not None else None
    lp = F.log_softmax(ty.logits(E.d[m], E.a[m], E.w[m], du, dv), 1)
    rl = ty.rel(E.qemb)[E.e_row[m]]
    on = E.em_on[m].to(lp.dtype)
    n, npos = on.numel(), float(on.sum())
    lpos, lneg = on[:, None] * F.logsigmoid(rl), (1.0 - on)[:, None] * F.logsigmoid(-rl)
    if 0 < npos < n:    # the M step's relevance loss: on- and off-path edges at half the batch each
        ll = lpos * (0.5 * n / npos) + lneg * (0.5 * n / (n - npos))
    else:               # a batch with one class only says nothing about which types are relevant
        ll = torch.zeros_like(lpos)
    at = E.a_true[m]
    with torch.no_grad():
        post = lp + o["mu"] * (lpos + lneg)     # the E step's likelihood is the unweighted one
        lam = arm.emc.lam(o)
        has = at >= 0
        if lam > 0 and bool(has.any()):
            post[has] = post[has] + lam * arm.emc.log_theta(g)[:, at[has]].T
        q = torch.softmax(post, 1)
        if bool(has.any()):
            arm.emc.update(g, q[has], at[has])
        arm.emc.add(g, q, lp, on, at)
    return o["om"] * (-(q * lp).sum(1).mean() - (q * ll).sum(1).mean())


def step_loss8(s, gold, z, rule):
    """qd_gnn2.step_loss2 plus the training batch's pending M-step loss."""
    loss = _STEP2(s, gold, z, rule)
    aux = PENDING.pop("aux", None)
    if aux is None:
        return loss
    return aux if loss is None else loss + aux


class EPack:
    pass


def pack_edges(arm, rows, Qp=None, Tp=None, targets=False):
    """A batch's pool edges for the typer, in qd_gnn.pack_qd's order: d, the readable label a (NR: none), the true
    label, w, the row, and with targets the on-path target and the frontier."""
    Qp = arm.Q if Qp is None else Qp
    Tp = arm.TOK if Tp is None else Tp
    E = EPack()
    ne = np.asarray([Tp[i][0].size for i in rows], dtype=np.int64)
    E.B, E.off = len(rows), np.r_[0, np.cumsum(ne)]
    E.d = torch.from_numpy(np.concatenate([Tp[i][0] for i in rows]).astype(np.int64))
    a = np.concatenate([Tp[i][1] for i in rows]).astype(np.int64)
    E.a_true = torch.from_numpy(np.where((a >= 0) & (a < KLAB), a, -1))
    E.a = torch.full_like(E.a_true, -1) if arm.nr else E.a_true.clone()
    E.w = torch.from_numpy(np.concatenate([Qp[i]["w"] for i in rows]).astype(np.float32))
    E.e_row = torch.from_numpy(np.repeat(np.arange(len(rows)), ne))
    E.qemb = torch.from_numpy(np.stack([Qp[i]["qemb"] for i in rows]).astype(np.float32))
    E.du = E.dv = None
    if arm.em["deg"]:
        du, dv = [], []
        for i in rows:
            q, n = Qp[i], int(Qp[i]["n"])
            du.append(np.bincount(q["u"], minlength=n)[q["u"]])
            dv.append(np.bincount(q["v"], minlength=n)[q["v"]])
        E.du = torch.from_numpy(np.concatenate(du).astype(np.float32))
        E.dv = torch.from_numpy(np.concatenate(dv).astype(np.float32))
    if targets:
        E.em_on, E.em_mask = em_targets(arm, rows, Qp)
    return E


def drop_labels(arm, E, rows):
    """qd_gnn4.W3Arm.types' label-dropout draws: a row's labels all hidden with p_row, else each with p_edge."""
    drop = arm.rng.random(len(rows)) < arm.p_row
    ne = np.diff(E.off)
    de = arm.rng.random(int(ne.sum())) < arm.p_edge
    hide = torch.from_numpy(np.where(np.repeat(drop, ne), True, de))
    E.a = torch.where(hide, torch.full_like(E.a, -1), E.a)


def edge_drop8(P, rng, top):
    """qd_gnn7.edge_drop, draw for draw, cutting the EM tensors alike."""
    u = rng.random(P.B) * top
    keep = rng.random(P.src.numel()) >= u[P.e_row.numpy()]
    kt = torch.from_numpy(keep)
    for k in ("src", "dst", "e_row", "d", "a", "fam", "w") + EXTRA:
        x = getattr(P, k, None)
        if x is not None:
            setattr(P, k, x[kt])
    return int((~keep).sum())


# ── the models ───────────────────────────────────────────────────────────────


class W3EM(Q4.W3):
    """qd_gnn4.W3 over (famdir, latent type) tokens, with the EM edge ranking in the chain logit."""

    def __init__(self, k, d=64, L=3, o=None):
        super().__init__("T0", d, L, 0)
        o = o or {}
        self.kind, self.k, self.nt = "EM", k, 5 * k
        self.E_z = torch.nn.Parameter(torch.randn(k, d) * 0.1)
        self.typer = EdgeTyper(k, deg=o.get("deg", False))
        self.eta_on, self.joint, self.lg = bool(o.get("eta", True)), bool(o.get("jt", False)), bool(o.get("lg", False))
        self.eta = torch.nn.Parameter(torch.tensor(1.0 if self.eta_on else 0.0), requires_grad=self.eta_on)
        with torch.no_grad():
            self.log_kappa.fill_(math.log(o.get("ki", 1.0)))
        self.version = 0

    def set_phi(self, phi, K):
        self.typer.set_phi(phi, K)

    def load_state_dict(self, *args, **kwargs):
        """qd_gnn2.fit_shared2 reloads the best epoch's state: no read type cached before it may be used after it."""
        out = super().load_state_dict(*args, **kwargs)
        self.version += 1
        return out

    def token_emb(self, toks):
        return self.E_fd[toks // self.k] + self.E_z[toks % self.k]

    def forward(self, X):
        qn = self.lnq(self.Wq(X.qemb))                                             # (B, dim)
        e = self.chains(X)                                                         # (Tu, dim)
        W = self.A(qn) @ e.T + self.c[X.b_u, (X.len_u - 1).clamp_min(0)][None, :]  # (B, Tu)
        if self.eta_on:
            rl = F.logsigmoid(self.typer.rel(X.qemb))                              # (B, k)
            if not self.joint:
                rl = rl.detach()
            st = X.steps_u
            W = W + self.eta * (rl[:, st.clamp_min(0) % self.k] * (st >= 0).to(rl.dtype)).sum(-1)
        w = torch.gather(W, 1, X.tix.clamp_min(0)).masked_fill(~X.tmask, float("-inf"))
        null = qn @ self.nu + self.c_null
        p = torch.softmax(torch.cat([w, null[:, None]], 1), 1)[:, :-1]           # (B, T)
        kappa = torch.exp(self.log_kappa)
        beta = torch.nn.functional.softplus(self.beta_raw)
        contrib = p[X.eq, X.et] * X.ew.pow(-beta)
        boost = torch.zeros_like(X.z).index_put((X.eq, X.en), contrib, accumulate=True)
        if self.lg:
            boost = torch.log1p(boost / 0.05)
        return torch.where(torch.isfinite(X.z), X.z + kappa * boost, X.z)


class QDEM(torch.nn.Module):
    """An R-GCN over the latent types with query-conditioned attention, a residual on z (qd_gnn.QD's start and head)."""

    def __init__(self, k, d=64, layers=3, o=None, tb=None, rs=False):
        super().__init__()
        o = o or {}
        self.kind, self.k, self.dim, self.L = "EM", k, d, layers
        self.att, self.eta_on, self.joint = bool(o.get("att", True)), bool(o.get("eta", True)), bool(o.get("jt", False))
        self.tb, self.rs = tb, bool(rs)
        self.typer = EdgeTyper(k, deg=o.get("deg", False))
        self.Wq = torch.nn.Linear(QG.QDIM, d)
        self.lnq = torch.nn.LayerNorm(d)
        self.E_b = torch.nn.Parameter(torch.zeros(2, d))
        self.e_z = torch.nn.Parameter(torch.zeros(d))
        self.E_fd = torch.nn.Parameter(torch.randn(5, d))
        self.E_z = torch.nn.Parameter(torch.randn(k, d) * 0.5)
        lin = (lambda bias=False: torch.nn.ModuleList([torch.nn.Linear(d, d, bias=bias) for _ in range(layers)]))
        self.R, self.G, self.A, self.W = lin(), lin(True), lin(), lin()
        self.V, self.Kq, self.Kh = lin(), lin(), lin()
        self.c_fam = torch.nn.Parameter(torch.zeros(layers, 3))
        self.b_gate = torch.nn.Parameter(torch.zeros(layers))
        self.gamma_raw = torch.nn.Parameter(torch.zeros(()))
        self.eta = torch.nn.Parameter(torch.full((layers,), 1.0 if self.eta_on else 0.0), requires_grad=self.eta_on)
        self.Wr = torch.nn.Linear(2 * d + 2, d)
        self.wo = torch.nn.Linear(d, 1)
        torch.nn.init.zeros_(self.wo.weight)
        torch.nn.init.zeros_(self.wo.bias)
        if self.rs:
            self.log_c = torch.nn.Parameter(torch.zeros(()))

    def set_phi(self, phi, K):
        self.typer.set_phi(phi, K)

    def forward(self, P):
        rms = QG.rms
        qn = self.lnq(self.Wq(P.qemb))                                             # (B, d)
        h0 = qn[P.seed_row] * (1.0 + self.E_b[P.seed_b]) + P.z[P.seed_idx, None] * self.e_z
        h = torch.zeros(P.N, self.dim).index_add(0, P.seed_idx, h0)
        pi = torch.softmax(self.typer.logits(P.d, P.a, P.w, getattr(P, "du", None), getattr(P, "dv", None)), 1)
        rl = self.typer.rel(P.qemb)
        if not self.joint:
            pi, rl = pi.detach(), rl.detach()
        r = self.E_fd[P.d] + pi @ self.E_z
        erel = torch.log((pi * torch.sigmoid(rl)[P.e_row]).sum(1).clamp_min(1e-6))
        gamma = torch.sigmoid(self.gamma_raw)
        fam = P.fam.clamp(max=2)
        ninf = float("-inf")
        for l in range(self.L):
            w = self.R[l](r) * (1.0 + torch.tanh(self.G[l](qn)))[P.e_row]
            hs = rms(h)
            asrc = (h != 0).any(-1)[P.src]
            n_in = torch.zeros(P.N).index_add(0, P.dst, asrc.to(h.dtype))
            logit = (self.A[l](qn)[P.e_row] * r).sum(-1) + self.c_fam[l, fam] * P.w + self.b_gate[l] + self.eta[l] * erel
            if self.att:
                logit = logit + (self.Kq[l](qn)[P.e_row] * self.Kh[l](hs)[P.src]).sum(-1) / math.sqrt(self.dim)
                logit = logit.masked_fill(~asrc, ninf)
                with torch.no_grad():
                    mx = torch.full((P.N,), ninf).scatter_reduce(0, P.dst, logit, reduce="amax", include_self=True)
                    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
                ex = torch.exp(logit - mx[P.dst])
                den = torch.zeros(P.N).index_add(0, P.dst, ex)
                den = torch.where(den > 0, den, torch.ones_like(den))    # no active in-edge: every ex is 0
                alpha = ex / den[P.dst] * n_in[P.dst].clamp_min(1.0).pow(gamma)
            else:
                alpha = 2.0 * torch.sigmoid(logit) / (1.0 + n_in[P.dst]).pow(gamma)
            m = self.V[l](hs)[P.src] * w * alpha[:, None]
            agg = torch.zeros(P.N, self.dim).index_add(0, P.dst, m)
            h = h + torch.relu(self.W[l](agg))
        hn = rms(h)
        reached = (h != 0).any(-1).to(h.dtype)
        f = torch.cat([hn, hn * qn[P.node_row], P.z[:, None], reached[:, None]], 1)
        res = self.wo(torch.relu(self.Wr(f))).squeeze(-1)
        if self.rs:     # qd_gnn7.QD7's rs
            cnt = torch.zeros(P.B).index_add(0, P.node_row, torch.ones_like(res))
            mu = torch.zeros(P.B).index_add(0, P.node_row, res) / cnt
            dv = res - mu[P.node_row]
            var = torch.zeros(P.B).index_add(0, P.node_row, dv * dv) / cnt
            res = torch.exp(self.log_c) * dv * torch.rsqrt(var[P.node_row] + 1.0)
        if self.tb is not None:
            res = self.tb * torch.tanh(res / self.tb)
        return P.z + res


# ── the arms ─────────────────────────────────────────────────────────────────


class QDArm8(Q7.QDArm7):
    """qd_gnn7's arms and the two EM families; qd_gnn2.run builds every 'qd'-family arm as QG.QDArm(...)."""

    def __new__(cls, sp, *args, **kwargs):
        if cls is QDArm8:
            em = sp.get("em8")
            if em:
                return object.__new__(W3EMArm if em["arm"] == "w3em" else QDEMArm)
            if sp.get("w4"):
                return object.__new__(W4Arm8)
            if sp.get("w3"):
                return object.__new__(W3Arm8)
        return object.__new__(cls)


class W3Arm8(QDArm8, Q7.W3Arm7):
    """qd_gnn7.W3Arm7, unchanged, as a QDArm8."""


class W4Arm8(QDArm8, Q7.W4Arm7):
    """qd_gnn7.W4Arm7, unchanged, as a QDArm8."""


def note_cost(arm, ty_s, edges, comp_s, rows):
    c = COST.setdefault(Q5.NAMES.get(id(arm.sp), "?"), {"typer_s": 0.0, "edges": 0, "compile_s": 0.0, "rows": 0})
    c["typer_s"] += ty_s
    c["edges"] += edges
    c["compile_s"] += comp_s
    c["rows"] += rows


class W3EMArm(W3Arm8):
    is_em = True

    def __init__(self, sp, Q, TOK, G, z_of, A16, n_id):
        super().__init__(sp, Q, TOK, G, z_of, A16, n_id)
        self.em, self.k = sp["em8"], sp["em8"]["k"]
        self.nt = 5 * self.k
        self.st = Q4.STATS.setdefault(f"W3EM-z{self.k}-L{self.L}", {"rows": 0, "cached_rows": 0, "seconds": 0.0, "types": 0, "pairs": 0})
        self.emc, self.onp = EMCore(self.k), {}

    def make(self):
        model = W3EM(self.k, self.sp["d"], self.L, self.em)
        if self.p_row > 0 or self.p_edge > 0:   # qd_gnn4.W3Arm.make's draw
            self.rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        self.emc.reset()
        TRACE.setdefault(Q5.NAMES.get(id(self.sp), "?"), []).append(self.emc.trace)
        self.cache = {}
        return model

    def read(self, model, g, rows, rule):
        self.emc.tick(self.em)
        return super().read(model, g, rows, rule)

    def compile_em(self, i, tok):
        t = time.perf_counter()
        out = Q4.walk_types3(self.Q[i], tok, self.nt, self.L)
        self.st["rows"] += 1
        self.st["seconds"] += time.perf_counter() - t
        self.st["types"] += int(out[0].size)
        self.st["pairs"] += int(out[2].size)
        return out

    def em_types(self, model, rows):
        """Each row's walk types over its edges' index-time types (argmax pi), and in training the typer's batch."""
        training = model.training
        out = [None] * len(rows)
        cached = not training and not self.nr and not getattr(self, "shuf", False)   # the ID read's types only
        if cached:
            for j, i in enumerate(rows):
                c = self.cache.get(i)
                if c is not None and c[0] == model.version:
                    out[j] = c[1]
        todo = [j for j, x in enumerate(out) if x is None]
        if not todo:
            return out, None
        rr = [rows[j] for j in todo]
        E = pack_edges(self, rr, targets=training and self.em["em"])
        if training and self.rng is not None:
            drop_labels(self, E, rr)
        t = time.perf_counter()
        with torch.no_grad():
            z = model.typer.logits(E.d, E.a, E.w, E.du, E.dv).argmax(1).numpy()
        ty_s = time.perf_counter() - t
        tok = E.d.numpy() * self.k + z
        t = time.perf_counter()
        for j_, (j, i) in enumerate(zip(todo, rr)):
            res = self.compile_em(i, tok[E.off[j_]:E.off[j_ + 1]])
            out[j] = res
            if cached:
                self.cache[i] = (model.version, res)
                self.st["cached_rows"] += 1
        if not training:
            note_cost(self, ty_s, int(E.d.numel()), time.perf_counter() - t, len(rr))
        return out, E

    def forward(self, model, g, rows):
        TYs, E = self.em_types(model, rows)
        X = Q4.pack_w3(rows, self.Q, self.z_of, TYs, self.nt, self.L)
        s = model(X)
        if model.training:
            model.version += 1
            self.emc.trained = True
            if self.em["em"]:
                PENDING["aux"] = em_step(self, model, g, E)
        return s, X.gold, X.z


class QDEMArm(QDArm8):
    is_em = True

    def __init__(self, sp, Q, TOK, G, z_of, A16, n_id):
        super().__init__(sp, Q, TOK, G, z_of, A16, n_id)
        self.em, self.k = sp["em8"], sp["em8"]["k"]
        self.emc, self.onp = EMCore(self.k), {}

    def make(self):
        o = self.sp.get("t7") or {}
        model = QDEM(self.k, self.sp["d"], self.sp["L"], self.em, tb=o.get("tb"), rs=o.get("rs", False))
        # qd_gnn7.QDArm7.make's generators, drawn as it draws them
        if self.p_row > 0 or self.p_edge > 0:
            self.rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        self.rng_ed = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item())) if o.get("ed") else None
        self.emc.reset()
        TRACE.setdefault(Q5.NAMES.get(id(self.sp), "?"), []).append(self.emc.trace)
        return model

    def read(self, model, g, rows, rule):
        self.emc.tick(self.em)
        return super().read(model, g, rows, rule)

    def forward(self, model, g, rows):
        o = self.sp.get("t7") or {}
        Qp, Tp = self.rows_for(rows)
        P = QG.pack_qd(rows, Qp, Tp, self.z_of, self.sp["K"], self.nr)
        P.a_true = P.a.clone()
        P.du = P.dv = None
        if self.em["deg"]:
            P.du = torch.bincount(P.src, minlength=P.N).to(torch.float32)[P.src]
            P.dv = torch.bincount(P.dst, minlength=P.N).to(torch.float32)[P.dst]
        training = model.training
        if training and self.em["em"]:
            P.em_on, P.em_mask = em_targets(self, rows, Qp)
        if training and self.rng is not None:     # qd_gnn7.QDArm7.forward's label dropout, draw for draw
            drop = self.rng.random(P.B) < self.p_row
            de = drop[P.e_row.numpy()] | (self.rng.random(P.a.numel()) < self.p_edge)
            P.a = torch.where(torch.from_numpy(de), torch.full_like(P.a, -1), P.a)
        if training and self.rng_ed is not None:
            edge_drop8(P, self.rng_ed, o["ed"])
        s = model(P)
        if training:
            self.emc.trained = True
            if self.em["em"]:
                PENDING["aux"] = em_step(self, model, g, P)
        return QG.padded(P, s)


# ── the edge-ranking read ────────────────────────────────────────────────────


def auc(score, y):
    pos = int(y.sum())
    neg = int(y.size) - pos
    if pos == 0 or neg == 0:
        return None
    _u, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    rank = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((rank[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def em_diag(arm, model, g, rows):
    name = Q5.NAMES.get(id(arm.sp), "?")
    sub = list(rows[::max(1, len(rows) // DIAG_ROWS)])[:DIAG_ROWS]
    ty = model.typer
    A = {"query": [], "blind": [], "label_hidden": []}
    use = np.zeros(arm.k)
    n_on = n_e = 0
    model.eval()
    with torch.no_grad():
        for s0 in range(0, len(sub), 64):
            rr = sub[s0:s0 + 64]
            Qp, Tp = arm.rows_for(rr)
            E = pack_edges(arm, rr, Qp, Tp, targets=True)
            pi = torch.softmax(ty.logits(E.d, E.a, E.w, E.du, E.dv), 1)
            pin = torch.softmax(ty.logits(E.d, torch.full_like(E.a, -1), E.w, E.du, E.dv), 1)
            rl = torch.sigmoid(ty.rel(E.qemb))[E.e_row]
            sc = {"query": (pi * rl).sum(1).numpy(), "blind": (pi * torch.sigmoid(ty.b_r)).sum(1).numpy(),
                  "label_hidden": (pin * rl).sum(1).numpy()}
            mk, on = E.em_mask.numpy(), E.em_on.numpy()
            use += np.bincount(pi.argmax(1).numpy()[mk], minlength=arm.k)
            n_on += int(on[mk].sum())
            n_e += int(mk.sum())
            for j in range(len(rr)):
                lo, hi = E.off[j], E.off[j + 1]
                mm = mk[lo:hi]
                y = on[lo:hi][mm]
                for key, s in sc.items():
                    x = auc(s[lo:hi][mm], y)
                    if x is not None:
                        A[key].append(x)
    model.train()
    mean = (lambda xs: round(float(np.mean(xs)), 4) if xs else None)
    rec = {"graph": g, "rows": len(sub), "rows_scored": len(A["query"]), "frontier_edges": n_e,
           "on_path": round(n_on / max(1, n_e), 5), **{f"auc_{k}": mean(v) for k, v in A.items()},
           "type_usage": [round(float(x), 4) for x in use / max(1.0, use.sum())]}
    EMDIAG.setdefault(name, []).append(rec)
    log(f"em_diag {name} {g}: AUC query {rec['auc_query']} blind {rec['auc_blind']} labels hidden {rec['auc_label_hidden']}"
        f" (rows {rec['rows_scored']}, on-path {rec['on_path']}, usage {rec['type_usage']})")


def score_rows(arm, model, g, rows, TY=None):
    SZ = _SCORE["f"](arm, model, g, rows, TY)
    if (getattr(arm, "is_em", False) and Q5.PHASE["now"] == "read" and TY is None and not arm.nr
            and not getattr(arm, "shuf", False) and sorted(rows) == sorted(Q7.STATE7["B"].get(g, []))):
        try:
            em_diag(arm, model, g, rows)
        except Exception as ex:  # noqa: BLE001 -- a diagnostic never stops a run
            EMDIAG.setdefault("_failed", []).append({"arm": Q5.NAMES.get(id(arm.sp), "?"), "graph": g, "error": repr(ex)})
            log(f"em_diag failed: {ex!r}")
    return SZ


# ── binding and output ───────────────────────────────────────────────────────


_bind7 = Q7.bind


def bind():
    _bind7()
    QG.QDArm = QDArm8
    _PARSE["f"] = Q2.parse_arms     # qd_gnn7.parse_arms, just bound
    Q2.parse_arms = parse_arms
    _SCORE["f"] = Q2.score_rows     # qd_gnn7.score_rows, just bound
    Q2.score_rows = score_rows
    Q2.step_loss2 = step_loss8


def finish(out):
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "qd_gnn8"
    res["pins"]["qd_gnn7"] = SHAS["qd_gnn7"]
    res["qd_gnn8_sha256"] = SHAS["qd_gnn8"]
    res["em"] = {"trace": TRACE, "diag": EMDIAG,
                 "read_cost": {k: {**v, "typer_ms_per_row": round(1000 * v["typer_s"] / max(1, v["rows"]), 3),
                                   "compile_ms_per_row": round(1000 * v["compile_s"] / max(1, v["rows"]), 3)} for k, v in COST.items()},
                 "klab": KLAB, "gmax": GMAX, "diag_rows": DIAG_ROWS, "epochs": EMCFG["epochs"]}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


# ── selftest (synthetic rows; laptop) ────────────────────────────────────────


def brute_onpath(q, L):
    """The edges on a walk of exactly ds(g) edges from a seed to a gold g with ds(g) <= L (each one a shortest walk),
    by breadth-first distances and depth-first walks."""
    from collections import deque
    adj = {}
    for e, (x, y) in enumerate(zip(q["u"].tolist(), q["v"].tolist())):
        adj.setdefault(x, []).append((y, e))
    S = sorted({int(s) for s in q["seeds"][q["seeds"] >= 0]})
    ds, dq = {s: 0 for s in S}, deque(S)
    while dq:
        x = dq.popleft()
        for y, _e in adj.get(x, []):
            if y not in ds:
                ds[y] = ds[x] + 1
                dq.append(y)
    on = set()
    for g in np.flatnonzero(q["gold"]).tolist():
        D = ds.get(g)
        if D is None or D > L or D == 0:
            continue
        stack = [(s, 0, ()) for s in S]
        while stack:
            x, k, es = stack.pop()
            if k == D:
                if x == g:
                    on.update(es)
                continue
            for y, e in adj.get(x, []):
                stack.append((y, k + 1, es + (e,)))
    return on


def selftest():
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    A16 = AU.A16
    bind()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    K = 8
    # names
    sp = parse_em("QDEM-z8-L2-dr25e10-tb5-ed50-lam5-an3")
    assert sp["em8"]["k"] == 8 and sp["L"] == 2 and sp["t7"] == {"tb": 0.5, "ed": 0.5} and sp["drop_row"] == 0.25
    assert sp["em8"]["lam"] == 0.5 and sp["em8"]["an"] == 3 and sp["kind"] == "TXT" and "w3" not in sp
    sp = parse_em("W3EM-z4-L3-d32-noem-lg-ki30")
    assert sp["w3"] and not sp["em8"]["em"] and sp["em8"]["jt"] and sp["em8"]["lg"] and sp["em8"]["ki"] == 3.0
    for bad in ("W3EM-z8-tb5", "W3EM-noatt", "QDEM-lg", "QDEM-z0", "QDEM-z65", "QDEM-z8-foo", "QDEM-tb5-tb5", "QDEM-noem-lam5", "QDEM-ed0"):
        try:
            parse_em(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    print("selftest: EM arm names parse; misplaced, repeated, unknown and out-of-range options are refused")
    # the on-path target is the brute-force shortest-walk edge set; the frontier is qd_gnn.frontier_mask
    Q = QG.synthetic_rows(60, rng, K)
    for q in Q:
        for L in (1, 2, 3):
            on, fr = onpath_row(q, L)
            assert set(np.flatnonzero(on).tolist()) == brute_onpath(q, L), f"on-path differs at L{L}"
            assert np.array_equal(fr, QG.frontier_mask(q["u"], q["v"], q["seeds"], q["n"], L))
            assert not (on & ~fr).any(), "an on-path edge outside the frontier"
    print("selftest: the on-path target equals the brute-force shortest walks at L 1-3, inside the frontier")
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    phi = rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    G = {"g": {"phi": phi, "kind": "passage"}}
    rows = list(range(60))
    tr, sel = {"g": rows[:40]}, {"g": rows[40:]}
    Q7.STATE7["B"]["g"] = rows[40:]
    # dispatch: every family through QG.QDArm
    for nm, cls in (("QD-T0-L2", QDArm8), ("W3-T0-L2", W3Arm8), ("W4-T0-L2", W4Arm8), ("W3EM-z4-L2-d16", W3EMArm),
                    ("QDEM-z4-L2-d16", QDEMArm)):
        if nm.startswith(("W3EM", "QDEM")):
            s_ = parse_em(nm)
        elif nm.startswith("W4"):
            s_ = Q5.parse_w4(nm)
        elif nm.startswith("W3"):
            s_ = Q4.parse_w3(nm)
        else:
            s_ = QG.parse_qd(nm)
        a_ = QG.QDArm(s_, Q, TOK, G, z_of, A16, 0)
        assert type(a_) is cls and isinstance(a_, QG.QDArm) and isinstance(a_, Q7.QDArm7), nm
    print("selftest: QD, W3, W4, W3EM and QDEM specs build their arms, each a QDArm8 and a qd_gnn7 arm")
    # QDEM: z at init, the frontier cut exact, the score's gradient kept off the typer unless -jt
    for att in (True, False):
        for L in (1, 2, 3):
            torch.manual_seed(1)
            o = {**parse_em("QDEM-z4")["em8"], "att": att}
            model = QDEM(4, 16, L, o)
            model.set_phi(phi, K)
            P = QG.pack_qd(rows, Q, TOK, z_of, K)
            assert torch.equal(model(P), P.z), "an unfitted QDEM does not score z"
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.3 * torch.randn_like(p))
            s_full = model(P)
            Qc = []
            for q in Q:
                mk = QG.frontier_mask(q["u"], q["v"], q["seeds"], q["n"], L)
                qq = {**q, **{k: q[k][mk] for k in ("u", "v", "fam", "fwd", "bwd", "w")}}
                keep_struct = mk[q["fam"] == 0]
                qq["w2_f"], qq["w2_b"] = q["w2_f"][keep_struct], q["w2_b"][keep_struct]
                Qc.append(qq)
            s_cut = model(QG.pack_qd(rows, Qc, [QG.edge_labels(q) for q in Qc], z_of, K))
            err = float((s_full - s_cut).abs().max())
            assert err < 1e-5, f"QDEM att {att} L{L}: the frontier cut moves the scores by {err}"
            s_pad, gold, z = QG.padded(P, s_full)
            Q2.step_loss2(s_pad, gold, z, "np").backward()
            assert model.typer.f2.weight.grad is None, "the score's gradient reached the typer"
            assert model.wo.weight.grad is not None and float(model.wo.weight.grad.abs().sum()) > 0
    model = QDEM(4, 16, 2, {**parse_em("QDEM-z4-jt")["em8"]})
    model.set_phi(phi, K)
    with torch.no_grad():
        model.wo.weight.normal_(0, 0.5)
    P = QG.pack_qd(rows, Q, TOK, z_of, K)
    s_pad, gold, z = QG.padded(P, model(P))
    Q2.step_loss2(s_pad, gold, z, "np").backward()
    assert model.typer.f2.weight.grad is not None and float(model.typer.f2.weight.grad.abs().sum()) > 0
    print("selftest: QDEM scores z at init, the frontier cut is exact (attention and gate, L 1-3), the score's "
          "gradient reaches the typer only with -jt")
    # the EM step: posteriors normalised, theta counted, lambda annealed, its gradient on the typer alone
    arm = QG.QDArm(parse_em("QDEM-z4-L2-d16-an2"), Q, TOK, G, z_of, A16, 0)
    torch.manual_seed(0)
    model = arm.make()
    arm.prepare(model, "g")
    model.train()
    PENDING.clear()
    s_pad, gold, z = arm.forward(model, "g", rows[:16])
    aux = PENDING.pop("aux")
    assert aux is not None and arm.emc.N["g"].sum() > 0 and arm.emc.acc["edges"] > 0
    aux.backward()
    assert float(model.typer.f2.weight.grad.abs().sum()) > 0 and float(model.typer.E_r.grad.abs().sum()) > 0
    assert model.R[0].weight.grad is None and model.wo.weight.grad is None, "the M step reached the scorer"
    lams = []
    for ep in range(4):
        arm.emc.ep = ep
        lams.append(arm.emc.lam(arm.em))
    assert lams == [1.0, 0.5, 0.0, 0.0], lams
    print(f"selftest: the EM step counts theta, its loss {float(aux):.3f} reaches the typer alone; lambda {lams}")
    # fits in qd_gnn2's loop: reproducible, the typer moved by EM, a trace per epoch; NR moves a label arm's read
    for nm in ("QDEM-z4-L2-d16-dr25e10-tb5-ed50", "QDEM-z4-L2-d16-noatt", "W3EM-z4-L2-d16", "W3EM-z4-L3-d16-dr25e10-lg-ki30",
               "W3EM-z4-L2-d16-noem"):
        sp = parse_em(nm)
        Q5.NAMES[id(sp)] = nm
        fits = []
        for _rep in range(2):
            a_ = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
            fits.append((a_,) + tuple(Q2.fit_shared2(a_, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)))
        (a0, m0, e0, c0, _), (a1, m1, e1, c1, _) = fits
        assert c0 == c1 and all(torch.equal(m0.state_dict()[k], m1.state_dict()[k]) for k in m0.state_dict()), f"{nm}: not reproducible"
        torch.manual_seed(0)
        init = (W3EM if sp.get("w3") else QDEM)(4, 16, sp["L"], sp["em8"]).state_dict()
        moved = float((m0.state_dict()["typer.f2.weight"] - init["typer.f2.weight"]).abs().max())
        assert (moved > 0) == sp["em8"]["em"] or not sp.get("w3"), f"{nm}: typer moved {moved}"
        tr_ = TRACE[nm][-1]
        assert (len(tr_) == 3) == sp["em8"]["em"], f"{nm}: trace {len(tr_)}"
        a0.prepare(m0, "g")
        sid = Q2.score_rows(a0, m0, "g", rows[40:])
        a0.nr = True
        snr = Q2.score_rows(a0, m0, "g", rows[40:])
        a0.nr = False
        dnr = max(float(np.abs(x[0] - y[0]).max()) for x, y in zip(sid, snr))
        if sp.get("t7", {}).get("tb"):
            assert all(float(np.abs(s_ - z_).max()) <= 0.5 + 1e-5 for s_, z_ in sid), "tb bound broken"
        print(f"selftest {nm}: reproducible ({c0}), typer moved {moved:.3g}, trace {len(tr_)} epochs, NR moves {dnr:.3g}; "
              f"params {sum(p.numel() for p in m0.parameters())}")
    assert EMDIAG and all(r["rows_scored"] > 0 for k, v in EMDIAG.items() if k != "_failed" for r in v), EMDIAG.get("_failed")
    assert "_failed" not in EMDIAG, EMDIAG["_failed"]
    print(f"selftest: em_diag on every EM arm's ID read ({sum(len(v) for v in EMDIAG.values())} records); "
          f"W3EM read cost {COST}")
    # W3EM's compiled types over EM tokens equal the brute-force walks
    for i, q in enumerate(Q[:20]):
        tok = TOK[i][0].astype(np.int64) * 4 + rng.integers(0, 4, TOK[i][0].size)
        for L in (1, 2, 3):
            ref = Q4.brute_types(q, tok, 20, L)
            codes, ptr, nodes = Q4.walk_types3(q, tok, 20, L)
            assert {int(c): set(nodes[ptr[j]:ptr[j + 1]].tolist()) for j, c in enumerate(codes)} == ref
    print("selftest: W3EM's walk types over (famdir, type) tokens equal the brute-force walks; all checks passed")


def main():
    Q7.bind = bind    # qd_gnn7.main binds through its module's bind, and hands it to qd_gnn5.main
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--epochs" in sys.argv:
        EMCFG["epochs"] = int(sys.argv[sys.argv.index("--epochs") + 1])
    Q7.main()
    if Q6x.STATE["out"] is not None:
        finish(Q6x.STATE["out"])


if __name__ == "__main__":
    main()
