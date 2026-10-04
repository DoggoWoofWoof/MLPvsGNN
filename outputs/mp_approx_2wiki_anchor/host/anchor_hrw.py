"""Design look (untracked; not a result and not filed): a relation model meant for any graph, the hierarchical relation
walk (HRW). T0's relation-free walk prior is the backbone; a learned, text-coded relation operator (anchor_gen's GenMix:
lin, trans, ropet, house, ...) refines it; a per-graph mixing weight decides how much of the refinement is used.

Why: anchor_univ and anchor_few found that the learned relation channels do not carry to a graph outside the fit (2wiki
glinc -> hotpotqa -0.187 R@5 rho; every joint model <= 0 on musique and metaqa), while the relation-free T0 (family x
direction) carries 2wiki -> hotpotqa as well as hotpotqa's own fit does (+0.438). A relation model read with every
relation 'other' (NR) falls far below T0 (ghouse NR -0.73), so for it a graph with no or unknown relations is unsafe.

The model. A walk type tau = (b, a, c): seed bucket b, first edge token a, second c (none for one edge). Its group G(tau)
is tau with every phrase token replaced by its direction's 'other' token: the family x direction walk that T0 reads (the
coarse type), whose reach set R_G is the union of its member types' reach sets (= A16.walk_types on the 'other' tokens,
checked on every graph).
    backbone   w0(G) = <A0 q, bk[b] + t1[f(a)] + t2[f(c)] + ln[L]> + c[b, L], null <A0 q, nu> + c_null: T0's model
               (l16_look_analyze.Mix on the five family x direction codes), drawn first, so its init is a T0 fit's
    relation   w(tau): GenMix's logit (text code h(k) = P phi(k) + D[f(k)], operator op, optional cos term);
               u(tau) = w(tau) - w(G(tau)): what the walk's relations add over the same walk with every relation
               'other' (0 on a type with no phrase)
    posteriors p0 = softmax over the groups and the null of w0 (T0's posterior);
               pf = softmax over the types and the null of w0(G(tau)) - log n_G + u(tau), n_G the group's type count
               (with u = 0 every group keeps T0's mass, split evenly over its types; u moves mass within and between groups)
    bonus      b0(v) = sum_G p0(G) 1[v in R_G] |R_G|^-beta_g,   bf(v) = sum_tau pf(tau) 1[v in R_tau] |R_tau|^-beta_g
               s(v) = z(v) + kappa_g (b0(v) + lam_g (bf(v) - b0(v)))
               kappa_g, beta_g and lam_g (a sigmoid) belong to training graph g; a graph outside the fit reads their means
               (zero-shot) or a (kappa, lam) chosen on N of its labelled rows (few-shot).
Exact by construction, and asserted on every read: with every relation 'other' (NR) each group has one type, itself, so
u = 0, pf = p0, bf = b0, and the read is the backbone's (L0, lam = 0) whatever lam is. A relation table, an unseen
relation or a meaningless relation name therefore moves the model off its own T0 backbone only through lam's share, and
on a new graph the backbone (lam 0) and the twin (bonus off) are points of the calibration grid.

Training (fit): anchor_univ's joint loop (batches of one graph, --balance, the epoch chosen on the mean of the per-graph
select scores under rule p, AdamW with anchor_univ's groups; --patience P stops once P epochs pass without a gain on
that score) with loss CE(s) + alpha CE(s0), s0 the backbone-only score, so the backbone stays a T0 model in its own
right. --lam F fixes lam (not trained); with --lam 0 --alpha 0 on one graph the fit is anchor_univ's T0 fit, draw for
draw (the smoke test checks it bit for bit, both under deterministic algorithms: on 2 threads the CPU index_put that
sums a large index adds in a thread-dependent order, in a backward and in a bonus alike, so anchor_univ's fits and
reads do not repeat bit for bit; every HRW fit and read runs deterministic and does). --hold h/F hides a sha256 fold
of every training graph's phrases or relation names (anchor_gen.hold_map: shown as 'other') in training and selection.
Reads on each training graph's x1 half B, against its twin and GNN:
    ID     as trained;  L0 (lam 0) and NR (every relation 'other'), equal row for row;  L1 (lam 1)
    SWAP   (two-edge) the relation logit read with each walk's two tokens in the other order, groups, reach sets and
           backbone kept: what order the operator uses
    SHUF   (a KB) every relation shown with the next relation's text
    with --hold, on all of B and on its held-reach rows (a gold one structural edge from a seed through a held phrase):
    MASK   as trained (held 'other');  REV  held phrases shown with their own text (unseen relations);
    NRT    held phrases shown with a random vector of the same norm (a relation name that means nothing);
    SHUFH  held phrases shown with another held phrase's text
Reads on a graph outside the fit (xread; its read carve as anchor_few reads it: x1's half B, webqsp's select whole):
    zs, zs L0 (= zs NR, asserted), zs L1, zs SHUF (a KB), and the few-shot grid: draw d takes the first N rows of
    anchor_few's permutation of the pool carve (seed 20261100 + d); (dk, lam) over anchor_kgrid's log-kappa offsets x LAMS
    (and the zero-shot lam; a model fitted with a fixed lam keeps it, so its grid is kappa only) is chosen on them by
    anchor_kgrid's rule (best select score; ties to dk nearest 0, then the smaller dk, then lam nearest the zero-shot
    lam, then the smaller lam) and read on B. The grid's (0, zero-shot lam) point is checked row for row against the zs
    read; the oracle point on B (no rule may use it) and the bonus-off read (the twin: rho 0) are recorded.
Nothing reads a neighbour's score or state: a type's bonus is the same for every node in its reach set.

    python outputs/mp_approx_2wiki_anchor/host/anchor_hrw.py fit --train 2wiki=x4,hotpotqa=x4 --variants A256/ghouse \
        --hold 0/5 --out outputs/mp_approx_2wiki_anchor/host/anchor_hrw_wh_house.json
    python outputs/mp_approx_2wiki_anchor/host/anchor_hrw.py xread --target musique \
        --models outputs/mp_approx_2wiki_anchor/host/anchor_hrw_wh_house_models/v0_s0.pt --out PATH_with_hrw
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_kb_anchor" / "host"))
import anchor_univ as AU  # noqa: E402
import anchor_few as AF  # noqa: E402
import anchor_kgrid as KG  # noqa: E402

AC, G3, AG = AU.AC, AU.G3, AU.AG
AW, AW3, A16, G2 = AU.AW, AU.AW3, AU.A16, AU.G2
R_TOP = AG.R_TOP
PINS = {"anchor_univ": "54da7d219a690fc5cab75ab12fdb769bf2c19564f7eca676c7de07405429a505",
        "anchor_few": "1bcb67a9fffba58923e8589a730ff549f59b0757aee1c599edddfb759587d2bd",
        "anchor_kgrid": "090a6c7f474a604c0962dbaf31ecae19d4161504af529cc84039c663b4a7ae12"}
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
NRT_SEED = 20261003
CHECK_ROWS = 64
OFF = float("-inf")
log = AG.log


def check_pins():
    for mod, key in ((AU, "anchor_univ"), (AF, "anchor_few"), (KG, "anchor_kgrid")):
        if AW.sha(Path(mod.__file__)) != PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    if AW.sha(Path(KG.KC.__file__)) != KG.PINS["kcal"]:
        raise SystemExit(f"{KG.KC.__file__} is not the pinned file")
    AU.check_pins()
    AF.check_pins()


# ── types and groups ────────────────────────────────────────────────────────


def coarse_tokens(nt, K):
    """Token -> its group's token: a structural phrase or 'other' token -> its direction's 'other'; ner, knn unchanged."""
    cm = np.arange(nt, dtype=np.int64)
    s = 3 * (K + 1)
    cm[:s] = (cm[:s] // (K + 1)) * (K + 1) + K
    return cm


def group_codes(codes, nt, K):
    """Each walk code's group code (vectorised)."""
    tb1, cm = nt + 1, coarse_tokens(nt, K)
    codes = np.asarray(codes, dtype=np.int64)
    pre, last = np.divmod(codes, tb1)
    b, first = np.divmod(pre, tb1)
    c2 = np.where(last > 0, cm[np.maximum(last - 1, 0)] + 1, 0)
    return (b * tb1 + cm[first - 1] + 1) * tb1 + c2


def coarse_of(ty, nt, K):
    """A row's coarse types from its types: each code sent to its group, the reach sets united."""
    if not ty:
        return {}
    codes = np.fromiter(ty.keys(), dtype=np.int64, count=len(ty))
    acc = {}
    for c, g in zip(codes.tolist(), group_codes(codes, nt, K).tolist()):
        acc.setdefault(g, []).append(ty[c])
    return {g: (Rs[0] if len(Rs) == 1 else np.unique(np.concatenate(Rs))) for g, Rs in acc.items()}


def same_types(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[c], b[c]) for c in a)


def build_types(Q, sp, rmap, rows, K):
    """(nt, fine types, coarse types) on the rows, as dicts keyed by row."""
    nt, ty = AG.types_for(Q, sp, rmap, rows)
    TYf, TYc = {}, {}
    for i in rows:
        TYf[i] = ty[i]
        TYc[i] = coarse_of(ty[i], nt, K)
    return nt, TYf, TYc


def check_coarse(Q, sp, TYc, rows, K):
    """The united coarse types equal A16.walk_types with every relation 'other' (the NR types), row for row."""
    ck = list(rows[:CHECK_ROWS])
    _nt, tyn = AG.types_for(Q, sp, np.full(R_TOP + 1, K, dtype=np.int64), ck)
    bad = [i for i in ck if not same_types(TYc[i], tyn[i])]
    if bad:
        raise SystemExit(f"the united coarse types differ from the NR walk types on rows {bad[:5]}")
    return len(ck)


class View:
    """One reading of a set of rows: fine types TYf and coarse types TYc per row (dicts keyed by row)."""

    def __init__(self, Q, TYf, TYc, nt, K, z_of):
        self.Q, self.TYf, self.TYc, self.nt, self.K, self.z_of = Q, TYf, TYc, nt, K, z_of
        self.fd = AG.famdir_of_tokens(nt, K)

    def prep(self, i):
        tb1, fd = self.nt + 1, self.fd
        tyc, tyf = self.TYc[i], self.TYf[i]
        cc = np.asarray(sorted(tyc), dtype=np.int64)          # sorted codes, as l16_look_analyze.pack orders them
        pre, lastc = np.divmod(cc, tb1)
        bc, firstc = np.divmod(pre, tb1)
        Rc = [tyc[int(c)] for c in cc]
        szc = np.asarray([R.size for R in Rc], dtype=np.int64)
        ff = np.asarray(sorted(tyf), dtype=np.int64)
        pre, lastf = np.divmod(ff, tb1)
        bf, firstf = np.divmod(pre, tb1)
        Rf = [tyf[int(c)] for c in ff]
        szf = np.asarray([R.size for R in Rf], dtype=np.int64)
        gc = group_codes(ff, self.nt, self.K)
        gix = np.searchsorted(cc, gc)
        if ff.size and (gix.max() >= cc.size or not np.array_equal(cc[gix], gc)):
            raise SystemExit(f"row {i}: a walk type outside every coarse group")
        nG = np.bincount(gix, minlength=cc.size) if cc.size else np.zeros(0, dtype=np.int64)
        if cc.size and (nG == 0).any():
            raise SystemExit(f"row {i}: a coarse group with no walk type")
        e64 = np.zeros(0, dtype=np.int64)
        return {"tbc": bc, "t1c": firstc - 1, "t2c": lastc, "f1c": fd[firstc - 1],
                "f2c": np.where(lastc > 0, fd[np.maximum(lastc - 1, 0)] + 1, 0), "nG": nG,
                "etc": np.repeat(np.arange(cc.size), szc), "enc": np.concatenate(Rc).astype(np.int64) if Rc else e64,
                "ewc": np.repeat(szc, szc).astype(np.float32),
                "tb": bf, "t1": firstf - 1, "t2": lastf, "gix": gix,
                "et": np.repeat(np.arange(ff.size), szf), "en": np.concatenate(Rf).astype(np.int64) if Rf else e64,
                "ew": np.repeat(szf, szf).astype(np.float32)}

    def pack(self, rows):
        Q, z_of = self.Q, self.z_of
        ps = [self.prep(i) for i in rows]
        B = len(rows)
        T = max(1, max(p["tb"].size for p in ps))
        Tc = max(1, max(p["tbc"].size for p in ps))
        N = max(Q[i]["n"] for i in rows)
        L = {k: np.zeros((B, T), dtype=np.int64) for k in ("tb", "t1", "t2", "gix")}
        Lc = {k: np.zeros((B, Tc), dtype=np.int64) for k in ("tbc", "t1c", "t2c", "f1c", "f2c")}
        tmask, tmaskc = np.zeros((B, T), dtype=bool), np.zeros((B, Tc), dtype=bool)
        nG = np.ones((B, Tc), dtype=np.float32)
        z = np.full((B, N), -np.inf, dtype=np.float32)
        gold = np.zeros((B, N), dtype=np.float32)
        ef, ec = [[], [], [], []], [[], [], [], []]
        for bi, (i, p) in enumerate(zip(rows, ps)):
            t, tc = p["tb"].size, p["tbc"].size
            for k in L:
                L[k][bi, :t] = p[k]
            for k in Lc:
                Lc[k][bi, :tc] = p[k]
            tmask[bi, :t], tmaskc[bi, :tc] = True, True
            nG[bi, :tc] = p["nG"]
            for acc, (a, b_, c_) in ((ef, ("et", "en", "ew")), (ec, ("etc", "enc", "ewc"))):
                acc[0].append(np.full(p[a].size, bi, dtype=np.int64))
                acc[1].append(p[a])
                acc[2].append(p[b_])
                acc[3].append(p[c_])
            n = Q[i]["n"]
            z[bi, :n] = np.asarray(z_of[i], dtype=np.float32)
            gold[bi, :n] = np.asarray(Q[i]["gold"], dtype=np.float32)
        X = {k: torch.as_tensor(v) for k, v in {**L, **Lc}.items()}
        X["tmask"], X["tmaskc"] = torch.as_tensor(tmask), torch.as_tensor(tmaskc)
        X["lognG"] = torch.log(torch.as_tensor(nG))
        cat = (lambda xs, dt: torch.as_tensor(np.concatenate(xs) if xs else np.zeros(0), dtype=dt))
        X["ent_f"] = (cat(ef[0], torch.long), cat(ef[1], torch.long), cat(ef[2], torch.long), cat(ef[3], torch.float32))
        X["ent_c"] = (cat(ec[0], torch.long), cat(ec[1], torch.long), cat(ec[2], torch.long), cat(ec[3], torch.float32))
        X["qemb"] = torch.as_tensor(np.stack([Q[i]["qemb"] for i in rows]))
        X["z"], X["gold"] = torch.as_tensor(z), torch.as_tensor(gold)
        return X


# ── the model ───────────────────────────────────────────────────────────────


class HRW(torch.nn.Module):
    def __init__(self, nt, K, phi, op, n_graphs, cos=False, lam_fixed=None, d=64, qdim=1536):
        super().__init__()
        self.t0 = A16.Mix(5, d, qdim)                                 # the T0 backbone, drawn first
        self.gen = AG.GenMix(nt, K, phi, op, cos=cos, d=d, qdim=qdim)  # the relation logit
        for m in (self.t0, self.gen):                                 # the scale, decay and null are the model's own
            del m.log_kappa
            del m.beta_raw
        del self.gen.nu
        del self.gen.c_null
        self.log_kappa_g = torch.nn.Parameter(torch.zeros(n_graphs))
        self.beta_raw_g = torch.nn.Parameter(torch.zeros(n_graphs))
        self.lam_raw_g = torch.nn.Parameter(torch.zeros(n_graphs), requires_grad=lam_fixed is None)
        self.lam_fixed = lam_fixed
        self.nt, self.K, self.op, self.cos = nt, K, op, cos

    def set_phi(self, phi):
        self.gen.set_phi(phi)

    def calib(self, gi):
        """(log_kappa, beta, lam) of training graph gi; gi None: the means over the training graphs (zero-shot)."""
        if gi is None:
            lk, br, lr = self.log_kappa_g.mean(), self.beta_raw_g.mean(), self.lam_raw_g.mean()
        else:
            lk, br, lr = self.log_kappa_g[gi], self.beta_raw_g[gi], self.lam_raw_g[gi]
        lam = torch.sigmoid(lr) if self.lam_fixed is None else torch.tensor(float(self.lam_fixed))
        return lk, torch.nn.functional.softplus(br), lam

    def t0_logit(self, qemb, tbc, f1c, f2c):
        """l16_look_analyze.Mix.forward's logits on the coarse types (in T0's codes) and its null logit."""
        m = self.t0
        aq = m.A(qemb)
        L = (f2c > 0).long()
        e = m.bk[tbc] + m.t1[f1c] + m.t2[f2c] + m.ln[L]
        return (aq[:, None, :] * e).sum(-1) + m.c[tbc, L], aq @ m.nu + m.c_null

    def gen_logit(self, qemb, aq, h, h2, tb, t1, t2, swap=False):
        """anchor_gen.GenMix.forward's logit w before its mask (swap: a two-edge walk's tokens in the other order)."""
        g = self.gen
        if swap:
            two = t2 > 0
            t1, t2 = torch.where(two, t2 - 1, t1), torch.where(two, t1 + 1, t2)
        L = (t2 > 0).long()
        k2 = (t2 - 1).clamp(min=0)
        if g.kind == "lin":
            e = g.bk[tb] + h[t1] + torch.where((t2 > 0)[..., None], h2[k2], torch.zeros_like(h2[k2])) + g.ln[L]
        else:
            x = g.op(g.s0[tb], t1, h)
            x = torch.where((t2 > 0)[..., None], g.op(x, k2, h), x)
            e = x + g.ln[L]
        w = (aq[:, None, :] * e).sum(-1) + g.c[tb, L]
        if g.cos:
            cosv = qemb @ g.Phi.T
            w = w + g.gam[0][g.fdir[t1]] * cosv.gather(1, t1)
            w = w + torch.where(t2 > 0, g.gam[1][g.fdir[k2]] * cosv.gather(1, k2), torch.zeros_like(w))
        return w

    def params_used(self):
        n = sum(p.numel() for p in self.t0.parameters()) + self.gen.params_used()
        n += 2 * self.log_kappa_g.numel() + (self.lam_raw_g.numel() if self.lam_fixed is None else 0)
        return int(n)


def bonus_parts(model, X, gi, swap=False):
    """(log_kappa, lam, b0, bf): the backbone's and the relation term's bonus per pool node."""
    qemb, z = X["qemb"], X["z"]
    wc, null = model.t0_logit(qemb, X["tbc"], X["f1c"], X["f2c"])
    wc = wc.masked_fill(~X["tmaskc"], float("-inf"))
    p0 = torch.softmax(torch.cat([wc, null[:, None]], 1), 1)[:, :-1]
    g = model.gen
    aq = g.A(qemb)
    h, h2 = g.codes()
    uf = model.gen_logit(qemb, aq, h, h2, X["tb"], X["t1"], X["t2"], swap)
    uc = model.gen_logit(qemb, aq, h, h2, X["tbc"], X["t1c"], X["t2c"], swap)
    gix = X["gix"]
    lf = wc.gather(1, gix) - X["lognG"].gather(1, gix) + (uf - uc.gather(1, gix))
    lf = lf.masked_fill(~X["tmask"], float("-inf"))
    pf = torch.softmax(torch.cat([lf, null[:, None]], 1), 1)[:, :-1]
    lk, beta, lam = model.calib(gi)
    eqc, etc_, enc, ewc = X["ent_c"]
    eq, et, en, ew = X["ent_f"]
    b0 = torch.zeros_like(z).index_put((eqc, enc), p0[eqc, etc_] * ewc.pow(-beta), accumulate=True)
    bf = torch.zeros_like(z).index_put((eq, en), pf[eq, et] * ew.pow(-beta), accumulate=True)
    return lk, lam, b0, bf


def mix_score(z, lk, lam, b0, bf):
    """s = z + kappa (b0 + lam (bf - b0)) in l16_look_gate2.scores' form; lam 0 gives the backbone's score exactly,
    and so does any lam when bf == b0."""
    log_kappa = lk + torch.zeros(z.shape[0])
    return torch.where(torch.isfinite(z), z + torch.exp(log_kappa)[:, None] * (b0 + lam * (bf - b0)), z)


def scores(model, X, gi, lam=None, swap=False):
    lk, lam_g, b0, bf = bonus_parts(model, X, gi, swap)
    lam_ = lam_g if lam is None else torch.as_tensor(lam, dtype=torch.float32)
    return mix_score(X["z"], lk, lam_, b0, bf), mix_score(X["z"], lk, torch.zeros(()), b0, bf)


def rule_p_loss(s, gold, z):
    """anchor_univ.fit_joint's loss: the twin's rank-1 node out of the scores and the golds, gold cross-entropy."""
    first = G2.top1(z)
    ar = torch.arange(s.shape[0])
    s = s.clone()
    s[ar, first] = float("-inf")
    gold = gold.clone()
    gold[ar, first] = 0.0
    keep = gold.sum(1) > 0
    if not bool(keep.any()):
        return None
    ls = torch.log_softmax(s[keep], 1)
    gk = gold[keep]
    gw = gk / gk.sum(1, keepdim=True)
    return -(torch.where(gk > 0, ls, torch.zeros_like(ls)) * gw).sum(1).mean()


def read(model, view, rows, gi, lam=None, swap=False):
    """Per-row (R@5, FC@5, hit@1) under rule p (the twin's rank-1 node first), as l16_look_gate2.read_rows reads."""
    model.eval()
    out = []
    Q = view.Q
    with torch.no_grad():
        for s0 in range(0, len(rows), 256):
            rr = list(rows[s0:s0 + 256])
            X = view.pack(rr)
            s, _s0 = scores(model, X, gi, lam, swap)
            s[torch.arange(len(rr)), G2.top1(X["z"])] = float("inf")
            for bi, i in enumerate(rr):
                out.append(A16.metrics_of(s[bi, :Q[i]["n"]].numpy(), Q[i]["gold"], Q[i]["gt"]))
    return np.asarray(out)


def parts_of(model, view, rows, gi):
    """Per row (z, b0, bf) over its pool, and (log_kappa, lam) of gi: the few-shot grid scores from these."""
    model.eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(rows), 256):
            rr = list(rows[s0:s0 + 256])
            X = view.pack(rr)
            lk, lam, b0, bf = bonus_parts(model, X, gi)
            for bi, i in enumerate(rr):
                n = view.Q[i]["n"]
                out.append((X["z"][bi, :n].clone(), b0[bi, :n].clone(), bf[bi, :n].clone()))
    return out, lk.detach(), lam.detach()


def grid_read(Q, rows, parts, lk, dk, lam):
    """Every row's metrics at log_kappa lk + dk (dk -inf: the bonus off) and lam, rule p."""
    out = []
    for i, (z, b0, bf) in zip(rows, parts):
        if dk == OFF:
            s = z.clone()
        else:
            s = torch.where(torch.isfinite(z), z + torch.exp(lk + dk) * (b0 + lam * (bf - b0)), z)
        s[int(torch.argmax(z))] = float("inf")
        out.append(A16.metrics_of(s.numpy(), Q[i]["gold"], Q[i]["gt"]))
    return np.asarray(out)


def pick(scores_, keys, lam0):
    best = max(scores_[k] for k in keys)
    return min((k for k in keys if scores_[k] == best), key=lambda k: (abs(k[0]), k[0], abs(k[1] - lam0), k[1]))


def nrt_phi(phi, ranks, seed):
    """The given ranks' rows replaced by Gaussian vectors of the same norm (a name that means nothing)."""
    out = phi.copy()
    rng = np.random.default_rng(seed)
    for r in ranks:
        x = rng.standard_normal(phi.shape[1]).astype(np.float32)
        out[r] = x * (np.linalg.norm(phi[r]) / max(float(np.linalg.norm(x)), 1e-12))
    return out


def shuf_phi(phi, ranks):
    """Each given rank shown with the next given rank's text (cyclic), as anchor_univ's SHUF shows a KB's relations."""
    out = phi.copy()
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.size >= 2:
        out[ranks] = phi[np.roll(ranks, -1)]
    return out


@torch.no_grad()
def calib_record(model, names):
    out = {}
    for gi, g in enumerate(list(names) + [None]):
        lk, beta, lam = model.calib(None if g is None else gi)
        out["mean (zero-shot)" if g is None else g] = {"kappa": round(math.exp(float(lk)), 4), "beta": round(float(beta), 4),
                                                       "lam": round(float(lam), 4)}
    return out


def parse_variant(v):
    sp = AU.parse_variant(v)
    if sp["family"] != "gen":
        raise SystemExit(f"{v}: HRW's relation logit is a gen model's (A<K>/g<op>[c] or A<K>-1/glin[c])")
    return sp


# ── fit ─────────────────────────────────────────────────────────────────────


def fit(V, G, names, tr, sel, make, epochs, seed, lr, wd, balance, alpha, patience=0):
    torch.manual_seed(seed)
    model = make()
    mats = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": mats, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}], lr=lr)
    rng = np.random.default_rng(seed)
    streams = {g: {"perm": None, "pos": 0} for g in names}
    best, best_state, best_ep, curve, by_graph, n_batches = -1.0, None, -1, [], {g: [] for g in names}, []
    for ep in range(epochs):
        batches = AU.epoch_batches(rng, names, tr, balance, streams)
        n_batches.append({g: sum(1 for b in batches if b[0] == g) for g in names})
        model.train()
        for g, rows in batches:
            model.set_phi(G[g]["phi"])
            X = V.pack(rows)
            s, s0 = scores(model, X, names.index(g))
            loss = rule_p_loss(s, X["gold"], X["z"])
            if loss is None:
                continue
            if alpha > 0:
                loss = loss + alpha * rule_p_loss(s0, X["gold"], X["z"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sc = {}
        for g in names:
            model.set_phi(G[g]["phi"])
            sc[g] = float(read(model, V, sel[g], names.index(g))[:, :2].mean())
            by_graph[g].append(round(sc[g], 4))
        score = sc[names[0]] if len(names) == 1 else float(np.mean([sc[g] for g in names]))
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            lk, _b, lam = model.calib(None)
        log(f"  ep {ep}: select {round(score, 4)} " + " ".join(f"{g} {sc[g]:.4f}" for g in names) +
            f"  (mean kappa {math.exp(float(lk)):.3f}, lam {float(lam):.3f})")
        if patience and ep - best_ep >= patience:
            log(f"  stopped after epoch {ep}: no gain on the select score for {patience} epochs")
            break
    model.load_state_dict(best_state)
    return model, best_ep, curve, by_graph, n_batches


def main_fit(a):
    out_path = Path(a.out)
    if "hrw" not in out_path.name:
        raise SystemExit("--out must carry 'hrw' in its file name")
    if a.balance not in ("graph", "rows"):
        raise SystemExit("--balance graph|rows")
    if a.alpha < 0:
        raise SystemExit("--alpha: non-negative")
    if a.patience < 0 or a.epochs < 1:
        raise SystemExit("--patience: non-negative (0: every epoch); --epochs: at least 1")
    lam_fixed = None if a.lam == "learn" else float(a.lam)
    if lam_fixed is not None and not 0.0 <= lam_fixed <= 1.0:
        raise SystemExit("--lam: learn or a value in [0, 1]")
    trains = AU.parse_train(a.train)
    names = list(trains)
    specs = {v: parse_variant(v) for v in a.variants.split(",")}
    if len({(sp["K"], sp["max_len"]) for sp in specs.values()}) != 1:
        raise SystemExit("--variants: one K and one max_len per run (the walk types are shared)")
    hold = None
    if a.hold:
        h, F = (int(x) for x in a.hold.split("/"))
        if not 0 <= h < F or F < 2:
            raise SystemExit("--hold h/F with 0 <= h < F, F >= 2")
        hold = (h, F)
    torch.set_num_threads(2)
    t0 = time.time()
    sp0 = next(iter(specs.values()))
    K, max_len = sp0["K"], sp0["max_len"]
    Q, part, G = AU.load_graphs(trains, max_len, t0, True)
    for g in names:
        if G[g]["kind"] == "passage" and len(G[g]["vocab"]) < K:
            raise SystemExit(f"{g}: the compact's phrase strings do not cover K")
    tr = {g: [i for lk in trains[g] for i in part[(g, lk)]] for g in names}
    sel = {g: part[(g, "select")] for g in names}
    B = {g: part[(g, "x1")][1::2] for g in names}
    RB = {g: AG.Reader(Q, B[g], 20261002) for g in names}
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    TYf, TYc, TYr, held, checks, nt = {}, {}, {}, {}, {}, None
    for g in names:
        rows_g = sorted({i for lk in ["x1", "select"] + trains[g] for i in part[(g, lk)]})
        if hold:
            rm, held[g] = AG.hold_map(K, G[g]["vocab"], *hold)
        else:
            rm, held[g] = AG.identity_map(K), None
        nt, tf, tc = build_types(Q, sp0, rm, rows_g, K)
        TYf.update(tf)
        TYc.update(tc)
        del tf, tc
        checks[g] = {"coarse_equals_NR_types_rows": check_coarse(Q, sp0, TYc, B[g], K)}
        if hold:
            _nt, tyr = AG.types_for(Q, sp0, AG.identity_map(K), B[g])
            TYr.update({i: tyr[i] for i in B[g]})
            del tyr
        log(f"{g}: walk types on {len(rows_g)} rows; coarse == NR types on {checks[g]['coarse_equals_NR_types_rows']} rows, "
            f"{time.time() - t0:.0f}s")
    V = View(Q, TYf, TYc, nt, K, z_of)
    NRV = View(Q, TYc, TYc, nt, K, z_of)
    REVV = View(Q, TYr, TYc, nt, K, z_of) if hold else None
    res = {"look": "anchor_hrw", "mode": "fit", "train": trains, "script_sha256": AW.sha(Path(__file__)), "pins": PINS,
           "univ_pins": AU.PINS, "balance": a.balance, "epochs": a.epochs, "patience": a.patience, "lr": a.lr, "wd": a.wd, "alpha": a.alpha, "lam": a.lam,
           "hold": a.hold, "deterministic": torch.are_deterministic_algorithms_enabled(), "checks": checks,
           "graphs": {g: {"kind": G[g]["kind"], "train_rows": len(tr[g]), "select_rows": len(sel[g]), "B_rows": len(B[g]),
                          "batches_per_epoch": math.ceil(len(tr[g]) / 128), "info": G[g]["info"], "phi_sha": G[g]["phi_sha"],
                          "vocab_head": [str(x) for x in G[g]["vocab"][:12]], "B": RB[g].base(),
                          **({"held": int(held[g][:K].sum()), "held_head": [str(G[g]["vocab"][r]) for r in np.flatnonzero(held[g][:K])[:12]]}
                             if hold else {})} for g in names},
           "variants": {}}
    per_row = {}
    model_dir = out_path.parent / (out_path.stem + "_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, **{f"B_rows_{g}": np.asarray(B[g]) for g in names})

    for vi, (name, sp) in enumerate(specs.items()):
        t1 = time.time()
        make = (lambda sp_=sp: (lambda: HRW(nt, K, G[names[0]]["phi"], sp_["op"], len(names), cos=sp_["cos"], lam_fixed=lam_fixed)))()
        seeds_out, m0 = {}, None
        for sd in sp["seeds"]:
            t2 = time.time()
            model, best_ep, curve, by_graph, n_batches = fit(V, G, names, tr, sel, make, a.epochs, sd, a.lr, a.wd, a.balance, a.alpha,
                                                             a.patience)
            m0 = m0 or model
            rd = {}
            for gi, g in enumerate(names):
                model.set_phi(G[g]["phi"])
                R = RB[g]
                key = f"{vi}_{sd}_{g}"
                mID = read(model, V, B[g], gi)
                mL0 = read(model, V, B[g], gi, lam=0.0)
                mNR = read(model, NRV, B[g], gi)
                if not np.array_equal(mNR, mL0):
                    raise SystemExit(f"{name}#{sd} {g}: the NR read is not the backbone's (L0)")
                mL1 = read(model, V, B[g], gi, lam=1.0)
                per_row.update({f"{key}_ID": mID, f"{key}_L0": mL0, f"{key}_L1": mL1})
                e = {"ID": R.record(mID), "L0 (= NR)": R.record(mL0, by_type=False), "L1": R.record(mL1, by_type=False),
                     "ID - L0": AW3.boot_pair(mID - mL0, R.W), "L1 - L0": AW3.boot_pair(mL1 - mL0, R.W)}
                if max_len == 2:
                    mS = read(model, V, B[g], gi, swap=True)
                    per_row[f"{key}_SWAP"] = mS
                    e["SWAP"], e["SWAP - ID"] = R.record(mS, by_type=False), AW3.boot_pair(mS - mID, R.W)
                if G[g]["kind"] == "kb":
                    model.set_phi(shuf_phi(G[g]["phi"], np.arange(min(G[g]["n_rel"], R_TOP))))
                    mSh = read(model, V, B[g], gi)
                    model.set_phi(G[g]["phi"])
                    per_row[f"{key}_SHUF"] = mSh
                    e["SHUF"], e["SHUF - ID"] = R.record(mSh, by_type=False), AW3.boot_pair(mSh - mID, R.W)
                if hold:
                    hr = np.flatnonzero(held[g][:K])
                    hrows = AG.held_reach_rows(Q, B[g], held[g], K)
                    pos = {i: j for j, i in enumerate(B[g])}
                    hx = np.asarray([pos[i] for i in hrows], dtype=np.int64)
                    hv = {"REV": read(model, REVV, B[g], gi)}
                    model.set_phi(nrt_phi(G[g]["phi"], hr, NRT_SEED + gi))
                    hv["NRT"] = read(model, REVV, B[g], gi)
                    model.set_phi(shuf_phi(G[g]["phi"], hr))
                    hv["SHUFH"] = read(model, REVV, B[g], gi)
                    model.set_phi(G[g]["phi"])
                    e["held"] = {"phrases": int(hr.size), "of_K": K, "held_reach_rows": len(hrows)}
                    hrr = {"MASK": R.subset(mID, hrows)} if hx.size else {}
                    for nm, m_ in hv.items():
                        per_row[f"{key}_{nm}"] = m_
                        e[nm], e[f"{nm} - MASK"] = R.record(m_, by_type=False), AW3.boot_pair(m_ - mID, R.W)
                        if hx.size:
                            hrr[nm] = R.subset(m_, hrows)
                            hrr[f"{nm} - MASK"] = AW3.boot_pair((m_ - mID)[hx], R.W[:, hx])
                    e["on_held_reach_rows"] = hrr
                rd[g] = e
            pt = model_dir / f"v{vi}_s{sd}.pt"
            calib = calib_record(model, names)
            torch.save({"look": "anchor_hrw", "spec": sp, "name": name, "seed": sd, "nt": nt, "K": K, "names": names,
                        "dataset": "+".join(names), "lam_fixed": lam_fixed, "alpha": a.alpha, "hold": a.hold,
                        "phi_sha": {g: G[g]["phi_sha"] for g in names}, "calib": calib, "state_dict": model.state_dict()}, pt)
            seeds_out[str(sd)] = {"best_epoch": best_ep, "epochs_run": len(curve), "curve": curve, "curve_by_graph": by_graph, "batches_per_epoch": n_batches[0],
                                  "calib": calib, "model_file": str(pt.relative_to(ROOT)) if pt.is_relative_to(ROOT) else str(pt),
                                  "reads": rd, "seconds": round(time.time() - t2, 1)}
            log(f"{name}#{sd}: ep {best_ep}, " + "; ".join(
                f"{g} ID {rd[g]['ID']['rho (R@5, FC@5, hit@1)']} L0 {rd[g]['L0 (= NR)']['rho (R@5, FC@5, hit@1)']} "
                f"L1 {rd[g]['L1']['rho (R@5, FC@5, hit@1)']}" for g in names) + "; calib " + json.dumps(calib))
            res["variants"][name] = {**sp, "tokens": nt, "params": int(sum(p.numel() for p in m0.parameters())),
                                     "params_used": m0.params_used(), "seeds_read": seeds_out, "seconds": round(time.time() - t1, 1)}
            save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


# ── reads on a graph outside the fit ───────────────────────────────────────


def main_xread(a):
    out_path = Path(a.out)
    if "hrw" not in out_path.name:
        raise SystemExit("--out must carry 'hrw' in its file name")
    ns = sorted({int(x) for x in a.n.split(",") if x} - {0})
    if ns and ns[0] < 4:
        raise SystemExit("--n: sizes of at least 4 (or 0 for zero-shot only)")
    deltas = [float(x) for x in a.deltas.split(",")]
    if 0.0 not in deltas or len(set(deltas)) != len(deltas):
        raise SystemExit("--deltas: distinct values, 0 among them")
    paths = [Path(p) for p in a.models.split(",") if p]
    if not paths or len(set(paths)) != len(paths):
        raise SystemExit("--models: one or more distinct HRW checkpoints")
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    for p, ck in zip(paths, cks):
        if ck.get("look") != "anchor_hrw":
            raise SystemExit(f"{p}: not an anchor_hrw checkpoint")
        if a.target in ck["names"]:
            raise SystemExit(f"{p}: fitted on {a.target} itself")
    torch.set_num_threads(2)
    t0 = time.time()
    max_len = max(ck["spec"]["max_len"] for ck in cks)
    loader = AF.load_kb if a.target in AF.KBS else AF.load_passage
    Q, rd_rows, pool, info, kb = loader(a.target, max_len, t0, bool(ns))
    if kb is not None:
        phi, phi_sha, n_rel = kb[0], info["rel_embeddings"], kb[1]
    else:
        (phi, phi_sha), n_rel = G3.phrase_table(a.target), None
    B_rows = list(rd_rows) if a.target in AF.WHOLE else rd_rows[1::2]
    if ns and ns[-1] > len(pool):
        raise SystemExit(f"--n {ns[-1]} is more than the pool's {len(pool)} rows")
    perm = {d: np.random.default_rng(AF.DRAW_SEED + d).permutation(len(pool)) for d in range(a.draws)} if ns else {}
    draws = {d: [pool[i] for i in perm[d][:ns[-1]]] for d in perm}
    L_rows = sorted({r for v in draws.values() for r in v})
    posL = {r: j for j, r in enumerate(L_rows)}
    keep = set(B_rows) | set(L_rows)
    for i in range(len(Q)):            # the pool rows no draw takes are not read: let them go
        if i not in keep:
            Q[i] = None
    RB = AG.Reader(Q, B_rows, 20261002)
    z_of = {i: A16.zscore(Q[i]["score"][:, 0]) for i in sorted(keep)}
    res = {"look": "anchor_hrw", "mode": "xread", "target": a.target, "script_sha256": AW.sha(Path(__file__)), "pins": PINS,
           "few_pins": AF.PINS, "kb_pins": AF.KB_PINS if kb is not None else None, "target_info": info, "phi_sha": phi_sha,
           "read_carve": AF.READ[a.target], "read_rows": "whole" if a.target in AF.WHOLE else "half B",
           "pool": {"carve": AF.POOL[a.target], "rows": len(pool)}, "labelled_rows": len(L_rows), "n": ns, "draws": a.draws,
           "draw_seed": AF.DRAW_SEED, "deltas": [KG.dkey(d) for d in deltas], "lams": list(LAMS), "B": RB.base(), "models": {}}
    per_row = {}

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows), L_rows=np.asarray(L_rows))

    cache = {}
    for jm, (p, ck) in enumerate(zip(paths, cks)):
        t1 = time.time()
        sp, K = ck["spec"], ck["K"]
        tkey = (K, sp["max_len"])
        if tkey not in cache:
            cache.clear()
            nt, TYf, TYc = build_types(Q, sp, AG.identity_map(K), sorted(keep), K)
            ncheck = check_coarse(Q, sp, TYc, B_rows, K)
            cache[tkey] = (nt, View(Q, TYf, TYc, nt, K, z_of), View(Q, TYc, TYc, nt, K, z_of), ncheck)
            log(f"walk types on {len(keep)} rows; coarse == NR types on {ncheck} rows, {time.time() - t0:.0f}s")
        nt, V, NRV, ncheck = cache[tkey]
        if nt != ck["nt"]:
            raise SystemExit(f"{p}: {nt} tokens on the target, {ck['nt']} in the checkpoint")
        model = HRW(nt, K, phi, sp["op"], len(ck["names"]), cos=sp["cos"], lam_fixed=ck["lam_fixed"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        R = RB
        mZ = read(model, V, B_rows, None)
        mZ0 = read(model, V, B_rows, None, lam=0.0)
        mZN = read(model, NRV, B_rows, None)
        if not np.array_equal(mZN, mZ0):
            raise SystemExit(f"{p}: the NR read is not the backbone's (L0)")
        mZ1 = read(model, V, B_rows, None, lam=1.0)
        per_row.update({f"{jm}_zs": mZ, f"{jm}_zsL0": mZ0, f"{jm}_zsL1": mZ1})
        with torch.no_grad():
            lam_zs = float(model.calib(None)[2])
        ent = {"from": ck["dataset"], "name": ck.get("name"), "seed": ck.get("seed"), "spec": sp, "model_file": str(p),
               "lam_fixed": ck["lam_fixed"], "alpha": ck.get("alpha"), "hold": ck.get("hold"), "calib": ck.get("calib"),
               "lam_zs": round(lam_zs, 4), "coarse_equals_NR_types_rows": ncheck,
               "params": int(sum(q.numel() for q in model.parameters())), "params_used": model.params_used(),
               "zs": R.record(mZ), "zs L0 (= NR)": R.record(mZ0, by_type=False), "zs L1": R.record(mZ1, by_type=False),
               "zs - zsL0": AW3.boot_pair(mZ - mZ0, R.W), "zsL1 - zsL0": AW3.boot_pair(mZ1 - mZ0, R.W)}
        if kb is not None:
            model.set_phi(shuf_phi(phi, np.arange(min(n_rel, R_TOP))))
            mSh = read(model, V, B_rows, None)
            model.set_phi(phi)
            per_row[f"{jm}_zsSHUF"] = mSh
            ent["zs SHUF"], ent["zs SHUF - zs"] = R.record(mSh, by_type=False), AW3.boot_pair(mSh - mZ, R.W)
        if ns:
            # a fixed lam is the one the model was trained for (with lam 0 its relation operator never trained): kappa only
            lams = [lam_zs] if ck["lam_fixed"] is not None else sorted(set(LAMS) | {lam_zs})
            ent["grid_lams"] = [round(x, 4) for x in lams]
            keys = [(dk, lm) for dk in deltas for lm in (lams if dk != OFF else [lam_zs])]
            pB, lk, lam_t = parts_of(model, V, B_rows, None)
            lam_of = (lambda lm: lam_t if lm == lam_zs else torch.tensor(lm, dtype=torch.float32))
            gB = {k: grid_read(Q, B_rows, pB, lk, k[0], lam_of(k[1])) for k in keys}
            del pB
            if not np.array_equal(gB[(0.0, lam_zs)], mZ):
                raise SystemExit(f"{p}: the grid's (0, zero-shot lam) read is not the zs read")
            pL, _lk, _lam = parts_of(model, V, L_rows, None)
            gL = {k: grid_read(Q, L_rows, pL, lk, k[0], lam_of(k[1])) for k in keys}
            del pL
            tag = (lambda k: f"{KG.dkey(k[0])}|{k[1]:.4g}")
            b_score = {k: round(float(gB[k][:, :2].mean()), 4) for k in keys}
            k_or = pick(b_score, keys, lam_zs)
            ent["grid_B"] = {tag(k): b_score[k] for k in keys}
            ent["oracle"] = {"at": tag(k_or), "B_score": b_score[k_or], "B": AF.brief(R.record(gB[k_or], by_type=False))}
            if OFF in deltas:
                ent["off"] = AF.brief(R.record(gB[(OFF, lam_zs)], by_type=False))   # the twin under rule p: rho 0 on real looks
            ent["kgrid"] = {}
            for N in ns:
                got, e = [], {"draws": []}
                for d in range(a.draws):
                    idx = [posL[r] for r in draws[d][:N]]
                    sc = {k: round(float(gL[k][idx, :2].mean()), 4) for k in keys}
                    kc = pick(sc, keys, lam_zs)
                    got.append(gB[kc])
                    per_row[f"{jm}_kgrid_{N}_{d}"] = gB[kc]
                    e["draws"].append({"draw": d, "at": tag(kc), "select": sc[kc], "B": AF.brief(R.record(gB[kc], by_type=False))})
                mean = np.mean(np.stack(got), 0)
                e["mean_of_draws"] = AF.brief(R.record(mean, by_type=False))
                e["chosen"] = [x["at"] for x in e["draws"]]
                e["kgrid - zs"] = AW3.boot_pair(mean - mZ, R.W)
                e["kgrid - zsL0"] = AW3.boot_pair(mean - mZ0, R.W)
                if OFF in deltas:
                    e["kgrid - off"] = AW3.boot_pair(mean - gB[(OFF, lam_zs)], R.W)
                ent["kgrid"][str(N)] = e
            del gB, gL
        ent["seconds"] = round(time.time() - t1, 1)
        key = p.name if p.name not in res["models"] else f"{p.parent.name}/{p.name}"
        res["models"][key] = ent
        log(f"{p.parent.name}/{p.name} ({ck['dataset']} -> {a.target}, {sp['model']}): zs {ent['zs']['rho (R@5, FC@5, hit@1)']}, "
            f"zs L0 {ent['zs L0 (= NR)']['rho (R@5, FC@5, hit@1)']}, zs L1 {ent['zs L1']['rho (R@5, FC@5, hit@1)']}"
            + (f"; oracle {ent['oracle']['at']} {ent['oracle']['B']['rho (R@5, FC@5, hit@1)']}; " if ns else "")
            + "; ".join(f"N {N}: {ent['kgrid'][str(N)]['chosen']} rho {ent['kgrid'][str(N)]['mean_of_draws']['rho (R@5, FC@5, hit@1)']}"
                        for N in ns))
        save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--train", required=True)
    f.add_argument("--variants", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--balance", default="graph")
    f.add_argument("--epochs", type=int, default=12)
    f.add_argument("--lr", type=float, default=3e-3)
    f.add_argument("--wd", type=float, default=0.0)
    f.add_argument("--alpha", type=float, default=0.5)
    f.add_argument("--lam", default="learn")
    f.add_argument("--hold", default=None)
    f.add_argument("--patience", type=int, default=0)
    x = sub.add_parser("xread")
    x.add_argument("--target", required=True, choices=tuple(AF.READ))
    x.add_argument("--models", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--n", default="16,64,256,1024")
    x.add_argument("--draws", type=int, default=3)
    x.add_argument("--deltas", default=KG.DELTAS)
    a = ap.parse_args(argv)
    if a.mode == "xread" and a.draws < 1:
        raise SystemExit("--draws: at least 1")
    check_pins()
    AC.bind()
    # on 2 threads the CPU backward of a large index (the backbone's and the operator's embedding tables) adds in a
    # thread-dependent order, so anchor_univ's own fits do not repeat bit for bit; here every fit and read does
    torch.use_deterministic_algorithms(True)
    (main_fit if a.mode == "fit" else main_xread)(a)


if __name__ == "__main__":
    main()
