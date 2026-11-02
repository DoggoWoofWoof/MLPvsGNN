"""LTG's stage (docs/LTG_NEXT_HOP_GNN.md): LTW's learned next-hop score (docs/LTW_LEARNED_NEXT_HOP.md sections 10 and 12,
SIGNAL and STORE_KEEPS) inside the GNN of record. Every pool edge u -> v gets a three-way bilinear score of the question,
the sender u and the question's residual after u, against the receiver v, all on the store's 256 axes (basis_2wiki, the
store block every batch already holds: rows decoded from their int8 codes, the question's projection, the store's own
cosine estimate). Each layer softmaxes its score (plus the sender's state through a learned map) over the sender's
out-edges; the log-ratio to uniform enters the edge's attention logit through a per-family gain and the receiver's update
through two per-family summaries. Every gain starts at 0, so a fit starts as the GNN of record bit for bit; the score's
own weights learn from a next-hop loss in training (each gold row's softmax mass on its gold neighbours, weight LAMBDA),
without which the term would start at a saddle. On PKO's roots, on the host's GPU: the screen at seed 0 on both WQ
splits against the base's seed-0 fit under LTG's in-domain rule (the eight reads of datasets the split trains on and the
WQ-hotpotqa fits' musique 3-hop and 4-hop slices), DEP's sixteen-read rule beside; the full run (seeds 1 and 2) when the
screen asks for it.

The base is the GNN of record after ISO (zdep.iso_call, zdep.base_fit). zml1s.py, zg6r.py, ziso.py and zdep.py are
imported and not edited. For the length of an LTG fit, zg6r's g6r_stack is the term's stack, ZG6R's __init__ adds the
term's parameters after zg6r's own (its own generator, so every other start is zg6r's), ZG6R's forward hands the store
block to the stack, and lean_gpu's listwiseD adds the next-hop loss to the training step whose scores the stack made;
zml1s runs with LTG's names, stamp key, stage and declaration, and on REPLACE with ISO's own-start patches. Every patched
name is put back afterwards, also on a failure.

    python outputs/mp_unified/zltg.py fit --name scr-ltg-nh-s0      (train when needed, then read; kept when read before)
    python outputs/mp_unified/zltg.py compare-all                    (each LTG fit read against the base's fit of its seed)
    python outputs/mp_unified/zltg.py calls                          (outputs/ltg/calls.json and .md)
    python outputs/mp_unified/zltg.py --selftest
"""
import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import ziso as ISO  # noqa: E402
import zdep as DEP  # noqa: E402  (the base after ISO, the slices and DEP's rule; DEP's fits are not touched)
import ml1start as MS  # noqa: E402  (on ISO's KEEP, ML1's zspk start read under ML1's names: amendment A1)

DECLARED = "docs/LTG_NEXT_HOP_GNN.md"
STAGE = "LTG"
KEY = "ltg"                                     # the stamp's key in an LTG fit's screen.json and read.json and LTG's compares
OUT = "outputs/ltg"
ARM = DEP.ARM                                   # zrm's arm: the model is zg6r's with the next-hop term
TAG = "nh"
RANK = 32                                       # LTW's bil rank
SEED_OFF = 4101                                 # the term's own draws (zg6r 3801, G6 3601/3602, U2n 3701)
CLIP = 8.0                                      # the log-ratio to uniform, clipped
LAMBDA = 1.0                                    # the next-hop loss's weight
CHUNK = 1 << 18                                 # edges per chunk of the edge products (a chunk's temporaries ~100 MB)
LOG_EVERY = 500                                 # training steps per line of the loss curve
LAYERS, D, FAMS = 3, 64, 3                      # zg6r's, checked on import
BASIS_NAME = "2wiki"                            # zwq: both WQ splits read basis_2wiki
BASIS_SHA = "0ca4797b2c3cce4244781542b3000f0eac56dbcf1949e26d3fac875dd0963c82"   # LTW A1's (outputs/ltw/store/report.json)
AXES = 256
CODE_SKIP = DEP.CODE_SKIP                       # stage modules, not the trainer's code
R5_TOL = DEP.R5_TOL
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    DEP.ADOPT_FOR_FULL_RUN, DEP.NO_GAIN, DEP.NOT_ADOPTED, DEP.ADOPT, DEP.INCOMPLETE)
ZERO_SHOT = tuple((sp, ds) for sp in M1.SPLITS for ds in M1.WQ.ZERO_SHOT[sp])    # zwq's: datasets a split never trains on
SLICE_SPLIT = "WQ-hotpotqa"                     # the split that trains on musique: its fits' slices decide
assert len(ZERO_SHOT) == 4 and DEP.SLICE_DS in M1.WQ.SPLITS[SLICE_SPLIT]
TERM_LAYER = tuple(tuple(f"ltg_{k}_{p}" for p in ("w", "g", "em", "ea")) for k in range(LAYERS))
TERM_NAMES = ("ltg_V", "ltg_U") + tuple(n for lay in TERM_LAYER for n in lay)


def n_term(axes):
    return 2 * 3 * axes * RANK + LAYERS * (3 * RANK * D + FAMS + 2 * FAMS * D)


N_TERM = n_term(AXES)
assert N_TERM == 68745


def fit_name(seed, split):
    return f"scr-ltg-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-ltg-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zltg: {msg}", flush=True)


def pairs(call):
    return {pair_name(sd, sp): (fit_name(sd, sp), DEP.base_fit(sd, sp, call)) for sd in RULE.SEEDS for sp in M1.SPLITS}


# ── the store's basis ────────────────────────────────────────────────────────

BASIS = {"store": None, "rec": None}


def load_basis(G, root=None, sha=BASIS_SHA, axes=AXES):
    """basis_2wiki (lean_cache.load_basis checks the npz against its record) as lean_mlp3.Store, the record's sha256 pinned
    to LTW A1's and its axes counted (sha and axes None: the self-test's toy basis); held for every ZG6R built after."""
    bz, rec = G.LC.load_basis(BASIS_NAME, root or ROOT / M1.ROOTS["cache"])
    if sha is not None and rec.get("npz_sha256") != sha:
        raise SystemExit(f"zltg: basis_{BASIS_NAME}.npz is {rec.get('npz_sha256')}, not LTW A1's {sha}")
    st = G.LG.L3.Store(bz)
    if axes is not None and int(st.V.shape[1]) != axes:
        raise SystemExit(f"zltg: basis_{BASIS_NAME} holds {st.V.shape[1]} axes, not {axes}")
    BASIS.update(store=st, rec=rec)
    return st, rec


def add_term(G, m, seed):
    """The term's parameters on m (V from its own generator, everything else 0) and the basis as buffers (not saved)."""
    torch, nn = G.torch, G.nn
    st = BASIS["store"]
    if st is None:
        raise RuntimeError("zltg: no basis loaded before a ZG6R was built (zltg.load_basis)")
    A = int(st.V.shape[1])
    g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
    m.ltg_V = nn.Parameter(torch.randn(3, A, RANK, generator=g))
    m.ltg_U = nn.Parameter(torch.zeros(3, A, RANK))
    for wn, gn, en, an in TERM_LAYER:
        setattr(m, wn, nn.Parameter(torch.zeros(3 * RANK, D)))
        setattr(m, gn, nn.Parameter(torch.zeros(FAMS)))
        setattr(m, en, nn.Parameter(torch.zeros(FAMS, D)))
        setattr(m, an, nn.Parameter(torch.zeros(FAMS, D)))
    m.register_buffer("ltg_bV", torch.from_numpy(np.ascontiguousarray(st.V, np.float32)), persistent=False)
    m.register_buffer("ltg_bm", torch.from_numpy(np.ascontiguousarray(st.m, np.float32)), persistent=False)
    m.register_buffer("ltg_bc", torch.from_numpy(np.ascontiguousarray(st.c, np.float32)), persistent=False)
    m.ltg_kappa = float(st.kappa)


# ── the term ─────────────────────────────────────────────────────────────────

_ED = {}
CUR = {}


def edge_dot(G):
    """EdgeDot (built once from zg6r's torch): (E,) P[v] . Q[u], chunk by chunk; its backward adds g Q[u] into P's rows v
    and g P[v] into Q's rows u, chunk by chunk, so no (E, 3 RANK) tensor is kept."""
    fn = _ED.get("fn")
    if fn is not None:
        return fn
    torch = G.torch

    class EdgeDot(torch.autograd.Function):
        @staticmethod
        def forward(ctx, P, Q, v, u, chunk):
            ctx.save_for_backward(P, Q, v, u)
            ctx.chunk = int(chunk)
            out = torch.empty(v.numel(), dtype=P.dtype, device=P.device)
            for a in range(0, v.numel(), ctx.chunk):
                b = a + ctx.chunk
                out[a:b] = (P[v[a:b]] * Q[u[a:b]]).sum(1)
            return out

        @staticmethod
        def backward(ctx, g):
            P, Q, v, u = ctx.saved_tensors
            gP = torch.zeros_like(P) if ctx.needs_input_grad[0] else None
            gQ = torch.zeros_like(Q) if ctx.needs_input_grad[1] else None
            for a in range(0, v.numel(), ctx.chunk):
                b = a + ctx.chunk
                ga = g[a:b].unsqueeze(1)
                if gP is not None:
                    gP.index_add_(0, v[a:b], ga * Q[u[a:b]])
                if gQ is not None:
                    gQ.index_add_(0, u[a:b], ga * P[v[a:b]])
            return gP, gQ, None, None, None

    _ED["fn"] = EdgeDot
    return EdgeDot


def seg_log_softmax(G, s, seg, n):
    """log-softmax of s within each segment seg (n segments); the segment's max is taken without its gradient."""
    torch = G.torch
    mx = torch.full((n,), -math.inf, dtype=s.dtype, device=s.device).scatter_reduce(
        0, seg, s.detach(), reduce="amax", include_self=True)
    sh = s - mx[seg]
    den = torch.zeros(n, dtype=s.dtype, device=s.device).index_add_(0, seg, sh.exp())
    return sh - den.clamp_min(1e-30).log()[seg]


def aux_loss(G, sp, u, v, gold, n):
    """(the next-hop loss, its sender count as a tensor): for each gold sender u with a gold receiver v != u, minus the log
    of the softmax mass (over u's out-edges) sp puts on those receivers, averaged over such senders; 0 without one."""
    torch = G.torch
    gb = gold if gold.dtype == torch.bool else gold > 0
    pos = gb[u] & gb[v] & (u != v)
    if not bool(pos.any()):
        return sp.sum() * 0.0, torch.zeros((), dtype=sp.dtype, device=sp.device)
    lp = seg_log_softmax(G, sp, u, n)
    up, lpp = u[pos], lp[pos]
    mx = torch.full((n,), -math.inf, dtype=lp.dtype, device=lp.device).scatter_reduce(
        0, up, lpp.detach(), reduce="amax", include_self=True)
    se = torch.zeros(n, dtype=lp.dtype, device=lp.device).index_add_(0, up, (lpp - mx[up]).exp())
    cnt = torch.zeros(n, dtype=lp.dtype, device=lp.device).index_add_(0, up, torch.ones_like(lpp))
    has = cnt > 0
    return -(se[has].log() + mx[has]).mean(), has.sum().to(sp.dtype)


def store_terms(G, m, semb, nq):
    """(qk, x, c, nr) with no gradient: qk the question on the store's axes (B, A), x each row's decoded axes (N, A), c the
    store's estimate of cos(question, row) (lean_mlp3.Store's P . Q), nr = |qk - c x| (at least 1e-12)."""
    torch = G.torch
    q, Pz = semb
    A = int(m.ltg_bV.shape[1])
    with torch.no_grad():
        q = q.to(Pz.dtype)
        qk = q @ m.ltg_bV
        ql = (q @ m.ltg_bm - qk @ m.ltg_bc) / m.ltg_kappa
        x = Pz[:, :A]
        c = (x @ qk.T).gather(1, nq.view(-1, 1)).squeeze(1) + Pz[:, A] * ql[nq]
        nr = torch.linalg.vector_norm(qk[nq] - c.unsqueeze(1) * x, dim=1).clamp_min(1e-12)
    return qk, x, c, nr


def term_inputs(G, m, semb, keep, nq, u, N):
    """(P, Q, lnu, kp): P (N, 3 RANK) each row as a receiver, [x U0, x U1, x U2]; Q (N, 3 RANK) each row as a sender,
    [qk V0, x V1, (qk - c x) V2 / nr] (the residual's product taken apart, so no (N, A) tensor is kept); lnu each edge's
    log out-degree of its sender, kp its question's store-block flag."""
    torch = G.torch
    qk, x, c, nr = store_terms(G, m, semb, nq)
    V, U = m.ltg_V, m.ltg_U
    xm = x @ torch.cat([V[1], V[2], U[0], U[1], U[2]], 1)
    Q = torch.cat([(qk @ V[0])[nq], xm[:, :RANK],
                   ((qk @ V[2])[nq] - c.unsqueeze(1) * xm[:, RANK:2 * RANK]) / nr.unsqueeze(1)], 1)
    P = xm[:, 2 * RANK:].contiguous()
    with torch.no_grad():
        deg = torch.zeros(N, dtype=x.dtype, device=x.device).index_add_(
            0, u, torch.ones(u.numel(), dtype=x.dtype, device=x.device))
        lnu = deg.clamp_min(1.0).log()[u]
        kp = keep[nq[u], m.j_semb].to(x.dtype)
    return P, Q, lnu, kp


def term_step(G, m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, te, codes, cte, P, Q, lnu, kp):
    """zg6r.g6r_step, op for op, with the next-hop evidence: ell = clip(log-softmax over the sender's out-edges of
    P[v] . (Q + W_k h)[u], plus the log out-degree) added to the attention logit through the family's gain (before zg6r's
    clamp), and the receiver's per-family mean of sigmoid(logit) ell and attention-weighted ell into its update."""
    torch, Fn = G.torch, G.Fn
    wm, wa, fb, ws, wmean, wagg, wx, b = (getattr(m, n) for n in G.G6.G6_LAYER[k])
    a_k, b_k, d_k, c_k = (getattr(m, n) for n in G.G6R_LAYER[k])
    w_k, g_k, em_k, ea_k = (getattr(m, n) for n in TERM_LAYER[k])
    N = h.shape[0]
    dev, dt = h.device, h.dtype
    c = torch.zeros(B, D, dtype=dt, device=dev).index_add_(0, nq, w * h)
    hm = Fn.linear(h, wm).view(N, FAMS, D)
    msg = hm[u, f]
    lg = (Fn.linear(h, wa)[u] * c[qv]).sum(1) / math.sqrt(D) + fb[f]
    if te.numel():
        rt = codes[cte]
        mt = Fn.linear((rt + d_k) * Fn.linear(h, b_k)[u[te]], a_k)
        msg = msg.index_copy(0, te, mt.to(msg.dtype))
        lg = lg.index_add(0, te, ((rt * Fn.linear(c, c_k)[qv[te]]).sum(1) / math.sqrt(G.K_R)).to(lg.dtype))
    sk = edge_dot(G).apply(P, Q + Fn.linear(h, w_k), v, u, CHUNK)
    ell = ((seg_log_softmax(G, sk, u, N) + lnu).clamp(-CLIP, CLIP) * kp).to(lg.dtype)
    lg = lg + g_k[f] * ell
    lg = lg.clamp(-G.LCLIP, G.LCLIP)
    sg = torch.sigmoid(lg)
    mean = torch.zeros(N * FAMS, D, dtype=dt, device=dev).index_add_(
        0, at, sg.unsqueeze(1) * msg).view(N, FAMS, D) / deg
    ex = torch.exp(lg)
    num = torch.zeros(N, D, dtype=dt, device=dev).index_add_(0, v, ex.unsqueeze(1) * msg)
    den = torch.zeros(N, dtype=dt, device=dev).index_add_(0, v, ex)
    att = num / den.clamp_min(1e-30).unsqueeze(1)
    pre = Fn.linear(h, ws) + Fn.linear(mean.reshape(N, FAMS * D), wmean) + Fn.linear(att, wagg) + \
        Fn.linear(xb, wx) + b
    am = torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, sg * ell).view(N, FAMS) / deg.view(N, FAMS)
    aa = torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, ex * ell).view(N, FAMS) / \
        den.clamp_min(1e-30).unsqueeze(1)
    pre = pre + am @ em_k + aa @ ea_k
    return h + Fn.gelu(Fn.layer_norm(pre, (D,)))


def term_layers(G, m, h, xb, w, u, v, f, nq, B, te, codes, cte, P, Q, lnu, kp):
    """zg6r.g6r_layers with term_step, checkpointed in training as zg6r's."""
    torch = G.torch
    N = h.shape[0]
    dev, dt = h.device, h.dtype
    at = v * FAMS + f
    one = torch.ones(u.numel(), dtype=dt, device=dev)
    deg = torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, one).view(N, FAMS, 1).clamp_min(1.0)
    qv = nq[v]
    ck = G.CKPT["on"] and m.training and torch.is_grad_enabled()
    for k in range(LAYERS):
        args = (G, m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, te, codes, cte, P, Q, lnu, kp)
        h = G.checkpoint(term_step, *args, use_reentrant=False, preserve_rng_state=False) if ck else term_step(*args)
    return h


def term_stack(G, m, H, xb, lk, gd, nq, B, u01, semb, keep):
    """zg6r.g6r_stack, op for op (the family dropout and the typed codes included), with the term's layers. In training
    (with gradients) the scores, the plain edge score P[v] . Q[u] and the kept edges are left for the next-hop loss."""
    torch, Fn = G.torch, G.Fn
    xb = xb.to(H.dtype)
    h0 = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b))
    s0 = Fn.linear(h0, m.g6_ctx_w).squeeze(-1)
    z0 = G.LG.seg_zscore8D(s0.unsqueeze(1), nq, B).squeeze(1).clamp(-G.ZCLIP, G.ZCLIP)
    w = G.G2.seg_softmax_rows(z0, nq, B).unsqueeze(1)
    u, v, f, ci = lk["u"], lk["v"], lk["f"], gd["ci"]
    if u01 is not None:
        ke = G.G2.family_keep(f, nq[v], B, u01)
        u, v, f, ci = u[ke], v[ke], f[ke], ci[ke]
    te = torch.nonzero(ci >= 0).view(-1)
    codes = cte = None
    if te.numel():
        if gd["rel"] is None:
            raise RuntimeError("zg6r: typed edges in the batch without their relation vectors")
        codes = G.rel_codes(m, gd["rel"], gd["crel"], gd["cdir"], H.dtype)
        cte = ci[te]
    N = h0.shape[0]
    P, Q, lnu, kp = term_inputs(G, m, semb, keep, nq, u, N)
    h = term_layers(G, m, h0, xb, w, u, v, f, nq, B, te, codes, cte, P, Q, lnu, kp)
    out = (Fn.linear(h, m.g6_out_w) + Fn.linear(h0, m.g6_skip_w) + m.g6_out_b).squeeze(-1)
    if m.training and torch.is_grad_enabled():
        CUR["aux"] = {"out": out, "sp": edge_dot(G).apply(P, Q, v, u, CHUNK), "u": u, "v": v, "n": N}
    return out


@contextlib.contextmanager
def on_term(G):
    """For the block: zg6r's g6r_stack is term_stack (zg6r's forward calls it by name); a ZG6R built gets the term after
    zg6r's own __init__ (the basis loaded before); ZG6R's forward hands its batch's store block and keep flags to the
    stack. All restored after, also on a failure."""
    orig_init, orig_fwd = G.ZG6R.__init__, G.ZG6R.forward

    def init(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        orig_init(self, blocks, widths, hidden, dropout, seed, arm, ctx)
        add_term(G, self, seed)

    def forward(self, feats, keep, nq, B, base_z):
        if "SEMB" not in feats:
            raise RuntimeError("zltg: no SEMB block in the batch (the term reads the store's rows)")
        CUR["in"] = (feats["SEMB"], keep)
        try:
            return orig_fwd(self, feats, keep, nq, B, base_z)
        finally:
            CUR.pop("in", None)

    def stack(m, H, xb, lk, gd, nq, B, u01=None):
        if "in" not in CUR:
            raise RuntimeError("zltg: g6r_stack called outside ZG6R's forward")
        semb, keep = CUR["in"]
        return term_stack(G, m, H, xb, lk, gd, nq, B, u01, semb, keep)

    try:
        with M1.WQ.patch(G, g6r_stack=stack), M1.WQ.patch(G.ZG6R, __init__=init, forward=forward):
            yield
    finally:
        CUR.clear()


# ── the next-hop loss ────────────────────────────────────────────────────────

AUX = {"steps": 0, "curve": [], "win": None}


def aux_reset():
    AUX.update(steps=0, curve=[], win=None)


def aux_flush():
    w = AUX["win"]
    if not w or not w[3]:
        return
    n = w[3]
    rec = {"step": AUX["steps"], "main": round(float(w[0]) / n, 6), "next_hop": round(float(w[1]) / n, 6),
           "senders": round(float(w[2]) / n, 3)}
    AUX["curve"].append(rec)
    AUX["win"] = None
    log(f"step {rec['step']}: listwise {rec['main']:.4f}, next-hop {rec['next_hop']:.4f}, {rec['senders']:.1f} senders a step")


def aux_add(main, aux, senders):
    """The step's two losses and sender count, summed on the device; a line of the curve every LOG_EVERY steps."""
    parts = (main.detach().double(), aux.detach().double(), senders.detach().double())
    w = AUX["win"]
    if w is None:
        AUX["win"] = [*parts, 1]
    else:
        AUX["win"] = [w[0] + parts[0], w[1] + parts[1], w[2] + parts[2], w[3] + 1]
    AUX["steps"] += 1
    if AUX["win"][3] >= LOG_EVERY:
        aux_flush()


def aux_summary():
    aux_flush()
    return {"steps": AUX["steps"], "lambda": LAMBDA, "every": LOG_EVERY, "curve": list(AUX["curve"])}


@contextlib.contextmanager
def on_aux(G):
    """lean_gpu's listwiseD plus LAMBDA x the next-hop loss when its scores are the ones term_stack left (a training step);
    every other call is listwiseD's. Restored after, also on a failure."""
    orig = G.LG.listwiseD

    def loss(scores, gold, nq, B):
        st = CUR.pop("aux", None)
        base = orig(scores, gold, nq, B)
        if st is None or st["out"] is not scores:
            return base
        a, n_s = aux_loss(G, st["sp"], st["u"], st["v"], gold, st["n"])
        aux_add(base, a, n_s)
        return base + LAMBDA * a

    try:
        with M1.WQ.patch(G.LG, listwiseD=loss):
            yield
    finally:
        CUR.pop("aux", None)


def term_record(G):
    A = int(BASIS["store"].V.shape[1])
    return {"term": "next-hop (LTW's bil on the store's axes)", "rank": RANK, "axes": A, "basis": BASIS_NAME,
            "basis_npz_sha256": (BASIS["rec"] or {}).get("npz_sha256"), "lambda": LAMBDA, "clip": CLIP,
            "term_parameters": n_term(A), "parameters_h128": int(G.N_PARAMS + n_term(A)), "layers": int(G.LAYERS),
            "seed_offset": SEED_OFF}


def term_record_keys():
    return ("term_parameters", "parameters_h128", "rank", "axes", "lambda", "layers")


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """LTG's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def ltg_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the LTG fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def stamp_with_term(orig, call, rec):
    """zml1s.stamp with the term's record, the base and ISO's verdict; a train's stamp also holds the next-hop curve."""
    def stamp(name, env, what, extra=None):
        if name not in FITS:
            return orig(name, env, what, extra)
        _a, sd, sp = FITS[name]
        add = {**rec, "base": DEP.base_fit(sd, sp, call), "iso_verdict": call}
        if what == "screen":
            add["aux"] = aux_summary()
        return orig(name, env, what, {**(extra or {}), **add})
    return stamp


@contextlib.contextmanager
def on_names():
    """zml1s with LTG's names, stamp key, stage and declaration; restored after. Enough for the compares and the calls."""
    sha, dsha, ms = M1.WQ.sha_src(__file__), M1.WQ.sha_src(DEP.__file__), MS.code()
    code = M1.code_shas
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=ltg_fit_name,
                                  code_shas=lambda arm=None: {**code(arm), "zltg_sha256": sha, "zdep_sha256": dsha, **ms}):
        yield


@contextlib.contextmanager
def on_ltg(G, call):
    """An LTG fit's context: LTG's names and stamp, the term, the next-hop loss, and the start: on REPLACE ISO's own (its
    load_encoder in zml1s, its encoder_used in zg6r), on KEEP ML1's zspk start read under ML1's names (ml1start's ml1_load in
    zml1s); restored after."""
    own = call == ISO.REPLACE
    with on_names(), M1.WQ.patch(M1, stamp=stamp_with_term(M1.stamp, call, term_record(G)),
                                 load_encoder=ISO.own_load if own else MS.ml1_load), \
            (M1.WQ.patch(G, encoder_used=ISO.own_used) if own else contextlib.nullcontext()), on_term(G), on_aux(G):
        yield


def import_g():
    G, _U2N = M1.import_arm(ARM)
    got = (G.LAYERS, G.D, G.FAMS, tuple(G.G6R_LAYER[0]))
    if got[:3] != (LAYERS, D, FAMS) or G.LAYERS != DEP.BASE_LAYERS:
        raise SystemExit(f"zltg: zg6r is {got}, not {LAYERS} layers of width {D} over {FAMS} families")
    return G


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zltg: {name} is not one of LTG's fits ({', '.join(FITS)})")
    call = DEP.iso_call()
    G = import_g()
    load_basis(G)
    aux_reset()
    _a, sd, sp = FITS[name]
    log(f"{name}: the next-hop term on {DEP.base_fit(sd, sp, call)}'s start (ISO {call}); {term_record(G)}")
    with on_ltg(G, call):
        return M1.fit(name)


# ── the compares, the slices and the calls ───────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: LTG's, ISO's or ML1's."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    return DEP.fit_block(name, what, base)


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[ARM]:
        raise SystemExit(f"zltg: {name}'s train and read are not on one device with zg6r's rows cap")
    return ks["device"]


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def compare_all(base=ROOT):
    """Each LTG fit that is read against the base's fit of its seed and split, through zwq's run_compare; a compare filed
    before is kept. The full run's fits are compared once they exist."""
    call = DEP.iso_call(base)
    done, ran = [], []
    for cname, (new, bse) in pairs(call).items():
        if not is_read(new, base):
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zltg: {new} and {bse} ran on different devices: {sorted(dev)}")
        with ISO.with_fits(), on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zltg: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "iso_verdict": call,
                    "device": json.loads(dev.pop()), "zltg_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_ltg(call, base=ROOT):
    """({(split, dataset): {seed: R@5}} of LTG's fits, the same of their bases) from LTG's compares."""
    new_r5, base_r5 = {}, {}
    for cname, (new, bse) in pairs(call).items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != DEP.CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        _a, sd, sp = FITS[new]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            new_r5.setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            base_r5.setdefault((sp, r["dataset"]), {})[sd] = r["base"][0]
    return new_r5, base_r5


def slices(call, new_r5, base3, base=ROOT):
    """({(split, slice): {seed: R@5}} of LTG's read fits, the same of the base's three seeds, problems), by zdep.slice_r5.
    Each file's all-question R@5 must be its compare's musique R@5."""
    new_s, base_s, bad = {}, {}, []
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            for f, own, ref in ((fit_name(sd, sp), new_s, new_r5), (DEP.base_fit(sd, sp, call), base_s, base3)):
                if f in FITS and not is_read(f, base):
                    continue
                s = DEP.slice_r5(f, base)
                if s is None:
                    bad.append(f"{f}: no {DEP.SLICE_DS} read file")
                    continue
                want = ref.get((sp, DEP.SLICE_DS), {}).get(sd)
                if want is not None and abs(s["all"] - want) > R5_TOL:
                    bad.append(f"{f}: its read file gives {DEP.SLICE_DS} R@5 {s['all']:.5f}, its compare {want}")
                for sl in DEP.SLICES:
                    if s[sl] is not None:
                        own.setdefault((sp, sl), {})[sd] = s[sl]
    return new_s, base_s, bad


# ── LTG's rule (section 4): in-domain, musique's slices of the split that trains on it ──


def deciding_keys():
    """The eight reads of datasets their split trains on, then the WQ-hotpotqa fits' musique 3-hop and 4-hop."""
    return [rd for rd in RULE.READS if rd not in ZERO_SHOT] + [(SLICE_SPLIT, sl) for sl in DEP.SLICES]


def beside_keys():
    """Reported, not deciding: the four zero-shot reads (zwq's) and the WQ-musique fits' (zero-shot) musique slices."""
    return list(ZERO_SHOT) + [(sp, sl) for sp in M1.SPLITS if sp != SLICE_SPLIT for sl in DEP.SLICES]


def screen_row(rd, b, n):
    """DEP's screen call of one read: floor max(0.0075, 2 x the base's spread), on the seed-0 delta."""
    s = RULE.spread(b)
    floor = RULE.pair_floor([s, s])
    d = n[0] - b[0]
    c = RULE.LOSS if d <= -floor + RULE.EPS else RULE.GAIN if d >= floor - RULE.EPS else RULE.WITHIN
    return {"split": rd[0], "read": rd[1], "base": [round(b[k], 4) for k in RULE.SEEDS], "new": round(n[0], 4),
            "delta": round(d, 4), "floor": round(floor, 4), "call": c}


def full_row(rd, b, n):
    """zml1.call of one read on all three seeds, floor the pair floor of both arms' spreads."""
    d = {k: n[k] - b[k] for k in RULE.SEEDS}
    fl = RULE.pair_floor([RULE.spread(n), RULE.spread(b)])
    return {"split": rd[0], "read": rd[1], "delta": {k: round(d[k], 4) for k in RULE.SEEDS}, "floor": round(fl, 6),
            "call": RULE.call(d, fl)}


def decide_screen(new, base):
    """Section 4, the screen: DEP's per-read call on the ten deciding reads. A LOSS: NOT_ADOPTED; else a GAIN:
    ADOPT_FOR_FULL_RUN; else NO_GAIN. The zero-shot reads and WQ-musique's slices are called beside."""
    rows, problems = [], []
    for rd in deciding_keys():
        b, n = base.get(rd, {}), new.get(rd, {})
        if any(k not in b for k in RULE.SEEDS) or 0 not in n:
            problems.append(f"{rd[0]} {rd[1]}: the base's seeds {sorted(b)}, LTG's {sorted(n)}")
            continue
        rows.append(screen_row(rd, b, n))
    side = [screen_row(rd, base[rd], new[rd]) for rd in beside_keys()
            if all(k in base.get(rd, {}) for k in RULE.SEEDS) and 0 in new.get(rd, {})]
    if problems:
        return {"verdict": INCOMPLETE, "problems": problems, "reads": rows, "beside": side, "counts": {}}
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.WITHIN)}
    v = NOT_ADOPTED if n[RULE.LOSS] else ADOPT_FOR_FULL_RUN if n[RULE.GAIN] else NO_GAIN
    return {"verdict": v, "problems": [], "reads": rows, "beside": side, "counts": n}


def decide_full(new, base):
    """Section 4, the full run: zml1.call on each deciding read. ADOPT: no LOSS among the ten and a GAIN on at least
    one of the eight in-domain dataset reads; else NOT_ADOPTED."""
    rows = []
    for rd in deciding_keys():
        b, n = base.get(rd, {}), new.get(rd, {})
        if any(k not in b or k not in n for k in RULE.SEEDS):
            return {"verdict": INCOMPLETE, "problems": [f"{rd[0]} {rd[1]}: a seed is missing"], "reads": rows,
                    "beside": [], "counts": {}}
        rows.append(full_row(rd, b, n))
    side = [full_row(rd, base[rd], new[rd]) for rd in beside_keys()
            if all(k in base.get(rd, {}) and k in new.get(rd, {}) for k in RULE.SEEDS)]
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.UNCONFIRMED, RULE.WITHIN)}
    gain = any(r["call"] == RULE.GAIN and r["read"] in RULE.DATASETS for r in rows)
    ok = not n[RULE.LOSS] and gain
    return {"verdict": ADOPT if ok else NOT_ADOPTED, "problems": [], "reads": rows, "beside": side, "counts": n}


def fit_problems(call, base=ROOT):
    """Why an LTG fit that is read is not the declared one: its term (parameters, rank, axes, basis, weight), layers, base
    and ISO verdict, its next-hop loss having run, its start, its device, and its trainer's code against the base's fit of
    its seed and split."""
    out = []
    want = (N_TERM, RANK, AXES, BASIS_SHA, LAMBDA, LAYERS)
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = DEP.base_fit(sd, sp, call)
        try:
            ks, kr, kb = fit_block(name, "screen", base), fit_block(name, "read", base), fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        for what, k in (("train", ks), ("read", kr)):
            got = tuple(k.get(x) for x in ("term_parameters", "rank", "axes", "basis_npz_sha256", "lambda", "layers"))
            if got != want or (k.get("base"), k.get("iso_verdict")) != (bse, call):
                out.append(f"{name}: its {what} names the term {got} on {k.get('base')} (ISO {k.get('iso_verdict')}), "
                           f"not {want} on {bse}")
        if int((ks.get("aux") or {}).get("steps") or 0) <= 0:
            out.append(f"{name}: its train ran no next-hop loss step")
        enc, benc = ks.get("encoder") or {}, kb.get("encoder") or {}
        if call == ISO.REPLACE:                     # zg6r-m's own draw: no zspk tensor copied
            ok = enc.get("start") == "own" and enc.get("copies") == 0
        else:                                       # zg6r's start: the zspk encoder its base fit copied
            ok = benc.get("fit") is not None and int(enc.get("copies") or 0) >= 1 and all(
                enc.get(x) == benc.get(x) for x in ("fit", "seed", "sha256"))
        if not ok:
            out.append(f"{name}: its encoder's start ({enc.get('start') or enc.get('fit')}, {enc.get('copies')} copies) "
                       f"is not {bse}'s (ISO {call})")
        if ks.get("device") != kb.get("device"):
            out.append(f"{name}: device {ks.get('device')}, {bse}'s {kb.get('device')}")
        cb, cn = kb.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if k not in CODE_SKIP and cn.get(k) != cb[k])
        if diff:
            out.append(f"{name}: code differs from {bse}'s in {diff}")
    return out


def calls(base=ROOT):
    call = DEP.iso_call(base)
    new_r5, b0 = r5_ltg(call, base)
    base3 = DEP.base_seeds(call, base)
    bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in LTG's compare, {base3.get(rd, {}).get(sd)} in its own stage's"
           for rd, by in sorted(b0.items()) for sd, v in sorted(by.items())
           if base3.get(rd, {}).get(sd) is None or abs(v - base3[rd][sd]) > R5_TOL]
    new_s, base_s, bad_s = slices(call, new_r5, base3, base)
    bad += bad_s
    new_all, base_all = {**new_r5, **new_s}, {**base3, **base_s}
    screen = decide_screen(new_all, base_all)
    full_ready = all(is_read(fit_name(sd, sp), base) for sd in RULE.SEEDS for sp in M1.SPLITS)
    full = decide_full(new_all, base_all) if full_ready else None
    dep_rule = {"screen": DEP.decide_screen(new_all, base_all)["verdict"],
                "full_run": DEP.decide_full(new_r5, base3, new_s, base_s)["verdict"] if full_ready else None}
    bad += fit_problems(call, base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{ARM} with the next-hop term", "iso_verdict": call,
           "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)", "screen": screen, "full_run": full,
           "dep_rule_beside": dep_rule, "problems": bad,
           "parameters": {f: {k: fit_block(f, "screen", base).get(k) for k in term_record_keys()}
                          for f in FITS if is_read(f, base)},
           "aux": {f: fit_block(f, "screen", base).get("aux") for f in FITS if is_read(f, base)},
           "zltg_sha256": M1.WQ.sha_src(__file__), "zdep_sha256": M1.WQ.sha_src(DEP.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# LTG: {ARM} with the next-hop term against {rec['base']}, on PKO's pools", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}",
          f"DEP's sixteen-read rule beside: screen {dep_rule['screen']}, full run {dep_rule['full_run']}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | LTG s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", ""]
        frows = full["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in full.get("beside", [])]
        if frows:
            md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            for r in frows:
                d = r["delta"]
                md.append(f"| {r['split']} | {r['read']} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | {r['floor']:.4f} | "
                          f"{r['call']} |")
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; DEP's rule beside {dep_rule}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def toy_edge_ops(G):
    """EdgeDot against the direct product and by gradcheck; seg_log_softmax against each segment's own log-softmax; the
    next-hop loss by hand."""
    torch = G.torch
    out = {}
    gen = torch.Generator().manual_seed(5)
    N, E, R3 = 9, 23, 6
    P = torch.randn(N, R3, generator=gen, dtype=torch.float64, requires_grad=True)
    Q = torch.randn(N, R3, generator=gen, dtype=torch.float64, requires_grad=True)
    v = torch.randint(0, N, (E,), generator=gen)
    u = torch.randint(0, N, (E,), generator=gen)
    ED = edge_dot(G)
    out["gradcheck"] = bool(torch.autograd.gradcheck(lambda a, b: ED.apply(a, b, v, u, 4), (P, Q)))
    P32, Q32 = P.detach().float().requires_grad_(), Q.detach().float().requires_grad_()
    got = ED.apply(P32, Q32, v, u, 5)
    want = (P32[v] * Q32[u]).sum(1)
    out["edge_dot_max_abs"] = float((got - want).detach().abs().max())
    gw = torch.randn(E, generator=gen)
    gp, gq = torch.autograd.grad(got, (P32, Q32), gw)
    wp, wq = torch.autograd.grad(want, (P32, Q32), gw)
    out["edge_dot_grad_max_abs"] = float(max((gp - wp).abs().max(), (gq - wq).abs().max()))
    s = torch.randn(E, generator=gen)
    lsm = seg_log_softmax(G, s, u, N)
    dense = torch.zeros(E)
    for i in u.unique().tolist():
        k = u == i
        dense[k] = torch.log_softmax(s[k], 0)
    out["seg_lsm_max_abs"] = float((lsm - dense).abs().max())
    gold = torch.zeros(N, dtype=torch.bool)
    gold[[1, 3, 4, 7]] = True
    a, cnt = aux_loss(G, s, u, v, gold, N)
    terms = []
    for i in range(N):
        k = (u == i) & gold[u] & gold[v] & (u != v)
        if bool(k.any()):
            terms.append(-torch.logsumexp(dense[k], 0))
    out["aux_senders"], out["aux_by_hand_senders"] = int(cnt), len(terms)
    out["aux_max_abs"] = float(abs(a - torch.stack(terms).mean())) if terms else float(a)
    z, zc = aux_loss(G, s, u, v, torch.zeros(N, dtype=torch.bool), N)
    out["aux_empty"] = float(z) == 0.0 and float(zc) == 0.0
    return out


def toy_basis(G, root, rng):
    """A toy basis_2wiki in lean_cache's format (npz and its record with the npz's sha256): orthonormal axes, AXES of
    them on the questions' width, so the toy carve's SEMB block (STORE_DIM columns) is read as the real store's."""
    qd = int(G.Q_DIM)
    V, _r = np.linalg.qr(rng.normal(size=(qd, AXES)))
    npz, js = G.LC.basis_paths(BASIS_NAME, root)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, m=(0.01 * rng.normal(size=qd)).astype(np.float32), V=V.astype(np.float32),
             w=np.linspace(1.0, 0.01, AXES).astype(np.float32))
    G.LC.write_json(js, {"graph": BASIS_NAME, "rows": 0, "explained": 0.0, "npz": npz.name,
                         "npz_sha256": G.LC.sha_file(npz)})


def toy_term(G):
    """The term on zg6r's own toy carve (zdep's toy: zg6r's self-test steps 2 and 3), zg6r's parameters drawn at random: at
    the term's start the eval and the training forward are zg6r's bit for bit (typed edges in); c is lean_mlp3.Store's
    estimate; the residual's product taken apart is the direct one; random term parameters move the score and every one
    gets a gradient; the next-hop loss alone moves U at the start (the listwise loss does not); checkpointing on and off
    agree; the parameter count; the strict load with and without the patch; a one-epoch lean_gpu.fit_variant with the
    loss hook firing; the basis pin; the patches' restoring."""
    import tempfile
    import torch
    G6 = G.G6
    G.LG.bind_device_ops()
    out = {}
    with tempfile.TemporaryDirectory(prefix="zltg_") as td:
        tmp = Path(td)
        rng = np.random.default_rng(31)
        G.RM.toy_roots(tmp, rng, chunks=(12, 12))
        G.RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                   rel_dir=tmp / "rel")
        G.ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(look_root=tmp / "look", cache_root=tmp / "cache", links_root=tmp / "lk", ch_root=tmp / "ch")

        class ToyIds(G6.ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        with G.U2N.extra_carves(("metaqa", "toy")):
            G.U2N.build_typed("metaqa", "toy", tmp / "look", tmp / "typed", **kw)
            plain = G.ZL.link_carve(G.RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
            c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            G.Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
            cls = G.g6r_carve(G.Z3.kt_carve(plain, tmp / "kt"), tmp / "typed")
            ce = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        toy_basis(G, tmp / "cache", rng)
        try:
            load_basis(G, tmp / "cache", sha="0" * 64)
            out["pin"] = "passed"
        except SystemExit:
            out["pin"] = "refused"
        st, _rec = load_basis(G, tmp / "cache", sha=None)
        A = int(st.V.shape[1])
        out["axes"] = A
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(ce.rows)[::-1][:7], np.arange(ce.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold0 = ce.batch(qs, blocks)
        keep = torch.ones(B, len(blocks))
        gd, lk = feats[G.G6R_KEY], feats[G.LK_KEY]
        out["typed_edges"] = int((gd["ci"] >= 0).sum())
        out["edges"] = int(lk["u"].numel())
        orig_stack, orig_init, orig_fwd = G.g6r_stack, G.ZG6R.__init__, G.ZG6R.forward
        names0 = tuple(G6.G6_NAMES) + tuple(G.G6R_NAMES)
        gen = torch.Generator().manual_seed(3)
        with on_term(G):
            torch.manual_seed(0)
            m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
            out["term_start"] = bool((m.ltg_V != 0).any()) and all(
                bool((getattr(m, k) == 0).all()) for k in TERM_NAMES if k != "ltg_V")
            out["parameters_term"] = sum(int(getattr(m, k).numel()) for k in TERM_NAMES)
            out["want_term"] = n_term(A)
            out["buffers_not_saved"] = not any(k.startswith("ltg_b") for k in m.state_dict())
            with torch.no_grad():
                for k in names0:
                    getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
            m.eval()
            with torch.no_grad():
                H = m.encode(feats, keep, nq, B)
                xb = torch.cat([feats[G.KT_KEY], m.chain(feats, keep, nq)], 1).to(H.dtype)
                y0 = m(feats, keep, nq, B, bz)
                want0 = orig_stack(m, H, xb, lk, gd, nq, B)
                out["eval_start_is_zg6r"] = bool(torch.equal(y0, want0))
                qk, x, cc, nr = store_terms(G, m, feats["SEMB"], nq)
                q_np, P_np = feats["SEMB"][0].numpy(), feats["SEMB"][1].numpy()
                Qs = np.stack([st.query(q_np[i]) for i in range(B)])
                want_c = (P_np * Qs[nq.numpy()]).sum(1)                 # the toy's vectors are not unit: c is not a cosine
                out["c_max_rel"] = float(np.abs(cc.numpy() - want_c).max() / max(1.0, np.abs(want_c).max()))
                P, Q, lnu, kp = term_inputs(G, m, feats["SEMB"], keep, nq, lk["u"], H.shape[0])
                r = qk[nq] - cc.unsqueeze(1) * x
                direct = (r @ m.ltg_V[2]) / torch.linalg.vector_norm(r, dim=1, keepdim=True)
                out["residual_max_rel"] = float((Q[:, 2 * RANK:] - direct).abs().max() / direct.abs().max().clamp_min(1.0))
                out["lnu_is_log_outdeg"] = bool(torch.allclose(
                    lnu.exp(), torch.bincount(lk["u"], minlength=H.shape[0]).float()[lk["u"]]))
            m.train()
            gs = m.g6_gen.get_state()
            torch.manual_seed(9)
            y1 = m(feats, keep, nq, B, bz)
            CUR.pop("aux", None)
            with M1.WQ.patch(G, g6r_stack=orig_stack), M1.WQ.patch(G.ZG6R, forward=orig_fwd):
                m.g6_gen.set_state(gs)
                torch.manual_seed(9)
                y2 = m(feats, keep, nq, B, bz)
            out["train_start_is_zg6r"] = bool(torch.equal(y1, y2))
            sd0 = {k: v.clone() for k, v in m.state_dict().items()}
            gold = torch.zeros(H.shape[0], dtype=torch.bool)
            pick = torch.randperm(lk["u"].numel(), generator=torch.Generator().manual_seed(4))[:6]
            gold[lk["u"][pick]] = True
            gold[lk["v"][pick]] = True
            aux_reset()
            with on_aux(G):
                m.zero_grad()
                s = m(feats, keep, nq, B, bz)
                G.LG.listwiseD(s, gold, nq, B).backward()
            out["aux_steps"] = AUX["steps"]
            out["aux_U_grad"] = float(m.ltg_U.grad.abs().sum()) if m.ltg_U.grad is not None else 0.0
            m.zero_grad()
            s = m(feats, keep, nq, B, bz)
            CUR.pop("aux", None)
            G.LG.listwiseD(s, gold, nq, B).backward()
            out["listwise_U_grad"] = float(m.ltg_U.grad.abs().sum()) if m.ltg_U.grad is not None else 0.0
            with torch.no_grad():
                for k in TERM_NAMES:
                    getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
            m.eval()
            with torch.no_grad():
                out["term_moves_score"] = not bool(torch.equal(m(feats, keep, nq, B, bz), want0))
            m.train()
            grads = {}
            for ck in (True, False):
                G.CKPT["on"] = ck
                try:
                    m.zero_grad()
                    m.g6_gen.set_state(gs)
                    torch.manual_seed(9)
                    yt = m(feats, keep, nq, B, bz)
                    CUR.pop("aux", None)
                    yt.sum().backward()
                    grads[ck] = (yt.detach().clone(), {k: getattr(m, k).grad.clone() for k in TERM_NAMES
                                                       if getattr(m, k).grad is not None})
                finally:
                    G.CKPT["on"] = True
            out["term_no_grad"] = [k for k in TERM_NAMES if k not in grads[True][1]
                                   or float(grads[True][1][k].abs().sum()) == 0]
            out["ckpt_same_score"] = bool(torch.equal(grads[True][0], grads[False][0]))
            out["ckpt_grad_max_abs"] = max(float((grads[True][1][k] - grads[False][1][k]).abs().max())
                                           for k in grads[True][1])
            sd = m.state_dict()
            G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0).load_state_dict(sd, strict=True)
            out["strict_load_patched"] = "passed"
            m.load_state_dict(sd0, strict=True)
        out["restored"] = G.g6r_stack is orig_stack and G.ZG6R.__init__ is orig_init and G.ZG6R.forward is orig_fwd
        plain_m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
        out["plain_has_term"] = any(hasattr(plain_m, k) for k in TERM_NAMES)
        try:
            plain_m.load_state_dict(sd, strict=True)
            out["strict_load_plain"] = "passed"
        except RuntimeError:
            out["strict_load_plain"] = "refused"
        # a one-epoch lean_gpu.fit_variant: the hook fires inside the trainer's own loop
        cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 1, "swa_from": 0, "cos": 0, "adamw": 0, "drop": 0}
        allq = np.arange(ce.rows, dtype=np.int64)
        _f, nqa, _b, ga = ce.batch(allq, blocks)
        la = _f[G.LK_KEY]
        out["toy_gold_gold_edges"] = int((ga[la["u"]] & ga[la["v"]] & (la["u"] != la["v"])).sum())
        aux_reset()
        with on_term(G), on_aux(G), M1.WQ.patch(G.LG, LeanMLP8D=G.ZG6R):
            res = G.LG.fit_variant([ce], blocks, cfg, 0, 16, "none", "cpu")
        summ = aux_summary()
        out["fit_aux_steps"] = summ["steps"]
        out["fit_curve"] = summ["curve"]
        out["fit_U_moved"] = float(res["swa"]["ltg_U"].abs().sum()) > 0
        out["fit_loss_finite"] = all(math.isfinite(c["loss"]) for c in res["curve"])
        out["fit_restored"] = G.LG.listwiseD.__name__ == "listwiseD" and G.LG.LeanMLP8D is not G.ZG6R
        aux_reset()
    return out


def selftest():
    assert fit_name(0, "WQ-musique") == "scr-ltg-nh-s0" and fit_name(2, "WQ-hotpotqa") == "scr-ltg-nh-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-ltg-nh-vs-base-s1-hp"
    assert pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-ltg-nh-s0", "scr-ml1-zg6r-s0")
    assert pairs(ISO.REPLACE)[pair_name(0, "WQ-hotpotqa")] == ("scr-ltg-nh-s0-hp", "scr-iso-zg6rm-s0-hp")
    import zrtr as RTR
    assert not set(FITS) & set(RTR.FITS)
    assert len(deciding_keys()) == 10 and len(set(deciding_keys())) == 10 and not set(deciding_keys()) & set(beside_keys())
    assert ("WQ-hotpotqa", "musique") in deciding_keys() and ("WQ-musique", "hotpotqa") in deciding_keys()
    assert ("WQ-musique", "metaqa") in beside_keys() and ("WQ-hotpotqa", "metaqa") in beside_keys()
    assert set(deciding_keys()) | set(beside_keys()) == set(DEP.read_keys())
    # 1. the edge operations
    G = import_g()
    e = toy_edge_ops(G)
    assert e["gradcheck"] and e["edge_dot_max_abs"] < 1e-6 and e["edge_dot_grad_max_abs"] < 1e-5, e
    assert e["seg_lsm_max_abs"] < 1e-6 and e["aux_senders"] == e["aux_by_hand_senders"] > 0, e
    assert e["aux_max_abs"] < 1e-5 and e["aux_empty"], e
    # 2. the term on zg6r's toy carve
    t = toy_term(G)
    assert t["pin"] == "refused" and t["typed_edges"] > 0 and t["edges"] > t["typed_edges"], t
    assert t["term_start"] and t["parameters_term"] == t["want_term"] and t["buffers_not_saved"], t
    assert t["eval_start_is_zg6r"] and t["train_start_is_zg6r"], t
    assert t["c_max_rel"] < 1e-5 and t["residual_max_rel"] < 1e-5 and t["lnu_is_log_outdeg"], t
    assert t["aux_steps"] == 1 and t["aux_U_grad"] > 0 and t["listwise_U_grad"] == 0.0, t
    assert t["term_moves_score"] and not t["term_no_grad"], t
    assert t["ckpt_same_score"] and t["ckpt_grad_max_abs"] < 1e-5, t
    assert t["strict_load_patched"] == "passed" and t["restored"] and not t["plain_has_term"], t
    assert t["strict_load_plain"] == "refused", t
    assert t["fit_aux_steps"] > 0 and t["fit_loss_finite"] and t["fit_restored"], t
    assert t["fit_U_moved"] or t["toy_gold_gold_edges"] == 0, t
    BASIS.update(store=None, rec=None)
    # 3. the record at the real basis's size, the patches and their restoring (also on a failure)
    BASIS.update(store=type("S", (), {"V": np.zeros((4, AXES), np.float32)})(), rec={"npz_sha256": BASIS_SHA})
    rec = term_record(G)
    assert rec["term_parameters"] == 68745 and rec["parameters_h128"] == 180218 + 68745 == 248963, rec
    n0 = len(M1.ALL_FITS)
    saved = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder)
    gsaved = (G.encoder_used, G.g6r_stack, G.ZG6R.__init__, G.ZG6R.forward, G.LG.listwiseD)
    seen = []
    orig_stamp = M1.stamp
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        aux_reset()
        with on_ltg(G, ISO.REPLACE):
            assert len(M1.ALL_FITS) == n0 + 6 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
            assert M1.load_encoder is ISO.own_load and G.encoder_used is ISO.own_used
            assert G.g6r_stack is not gsaved[1] and G.ZG6R.__init__ is not gsaved[2] and G.ZG6R.forward is not gsaved[3]
            assert G.LG.listwiseD is not gsaved[4]
            cs = M1.code_shas(ARM)
            assert "zltg_sha256" in cs and "zdep_sha256" in cs and "ml1start_sha256" in cs
            M1._ACTIVE[0] = fit_name(0, "WQ-hotpotqa")
            assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == "scr-ltg-nh-s0-hp"
            assert M1.pko_fit_name(ARM, 1, "WQ-hotpotqa") == "scr-ml1-zg6r-s1-hp"
            M1._ACTIVE[0] = None
            M1.stamp(fit_name(0, "WQ-hotpotqa"), {}, "screen", {"x": 1})
            M1.stamp(fit_name(0, "WQ-hotpotqa"), {}, "read", {"x": 2})
        base_hp = {**rec, "base": "scr-iso-zg6rm-s0-hp", "iso_verdict": ISO.REPLACE}
        assert seen[-2] == ("scr-ltg-nh-s0-hp", "screen", {"x": 1, **base_hp, "aux": aux_summary()}), seen
        assert seen[-1] == ("scr-ltg-nh-s0-hp", "read", {"x": 2, **base_hp}), seen
        assert (G.encoder_used, G.g6r_stack, G.ZG6R.__init__, G.ZG6R.forward, G.LG.listwiseD) == gsaved
        assert len(M1.ALL_FITS) == n0
        with on_ltg(G, ISO.KEEP):
            assert M1.load_encoder is MS.ml1_load and G.encoder_used is gsaved[0] and M1.KEY == KEY
            assert G.g6r_stack is not gsaved[1]
        try:
            with on_ltg(G, ISO.REPLACE):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
    finally:
        M1.stamp = orig_stamp
        M1._ACTIVE[0] = None
        BASIS.update(store=None, rec=None)
    assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder) == saved
    assert (G.encoder_used, G.g6r_stack, G.ZG6R.__init__, G.ZG6R.forward, G.LG.listwiseD) == gsaved
    assert len(M1.ALL_FITS) == n0 and not CUR
    # 4. LTG's rule: ten deciding reads, the zero-shot reads and WQ-musique's slices beside
    bs = {0: 0.50, 1: 0.51, 2: 0.49}                             # spread 0.01, floor 0.02
    base_ = {rd: dict(bs) for rd in DEP.read_keys()}
    new_ = {rd: {0: 0.50} for rd in DEP.read_keys()}
    assert decide_screen(new_, base_)["verdict"] == NO_GAIN
    new_[("WQ-musique", "musique")] = {0: 0.40}                  # zero-shot losses are beside
    new_[("WQ-hotpotqa", "metaqa")] = {0: 0.40}
    new_[("WQ-musique", "musique-4hop")] = {0: 0.40}
    sc = decide_screen(new_, base_)
    assert sc["verdict"] == NO_GAIN and len(sc["beside"]) == 6 and len(sc["reads"]) == 10, sc
    assert sum(r["call"] == RULE.LOSS for r in sc["beside"]) == 3, sc
    assert DEP.decide_screen(new_, base_)["verdict"] == NOT_ADOPTED
    new_[(SLICE_SPLIT, "musique-4hop")] = {0: 0.52}
    assert decide_screen(new_, base_)["verdict"] == ADOPT_FOR_FULL_RUN
    new_[("WQ-hotpotqa", "squad")] = {0: 0.48}
    assert decide_screen(new_, base_)["verdict"] == NOT_ADOPTED
    del new_[("WQ-hotpotqa", "2wiki")]
    assert decide_screen(new_, base_)["verdict"] == INCOMPLETE
    fb = {rd: dict(bs) for rd in DEP.read_keys()}
    fn = {rd: dict(bs) for rd in DEP.read_keys()}
    assert decide_full(fn, fb)["verdict"] == NOT_ADOPTED                                   # no GAIN
    fn[(SLICE_SPLIT, "musique-3hop")] = {0: 0.53, 1: 0.54, 2: 0.52}
    assert decide_full(fn, fb)["verdict"] == NOT_ADOPTED                                   # a slice GAIN alone
    fn[("WQ-hotpotqa", "musique")] = {0: 0.53, 1: 0.54, 2: 0.52}
    f = decide_full(fn, fb)
    assert f["verdict"] == ADOPT and f["counts"][RULE.GAIN] == 2, f
    fn[("WQ-musique", "musique")] = {0: 0.40, 1: 0.41, 2: 0.39}                           # zero-shot: beside
    fn[("WQ-musique", "metaqa")] = {0: 0.45, 1: 0.46, 2: 0.44}
    f = decide_full(fn, fb)
    assert f["verdict"] == ADOPT and sum(r["call"] == RULE.LOSS for r in f["beside"]) == 2, f
    fn[("WQ-hotpotqa", "musique")] = {0: 0.56, 1: 0.505, 2: 0.55}                         # unconfirmed
    f = decide_full(fn, fb)
    assert f["verdict"] == NOT_ADOPTED and f["counts"][RULE.UNCONFIRMED] == 1, f
    fn[("WQ-hotpotqa", "musique")] = {0: 0.53, 1: 0.54, 2: 0.52}
    fn[("WQ-musique", "webqsp")] = {0: 0.45, 1: 0.46, 2: 0.44}                            # a LOSS
    f = decide_full(fn, fb)
    assert f["verdict"] == NOT_ADOPTED and f["counts"][RULE.LOSS] == 1, f
    log("selftest: names and pairs; EdgeDot (gradcheck, the direct product), the segment log-softmax and the next-hop "
        f"loss by hand; the term on zg6r's toy carve ({t['typed_edges']} typed of {t['edges']} edges, {t['axes']} toy "
        f"axes: at its start the eval and training forwards zg6r's bit for bit, c within {t['c_max_rel']:.1e} (relative) "
        f"of lean_mlp3.Store, the residual within {t['residual_max_rel']:.1e} of the direct form, U moved by the next-hop "
        f"loss alone, every term parameter's gradient, checkpointing on and off within {t['ckpt_grad_max_abs']:.1e}, "
        f"{t['parameters_term']} term parameters, the strict load under the patch and its refusal without, the basis "
        f"pin, a one-epoch fit_variant with {t['fit_aux_steps']} next-hop steps ({t['toy_gold_gold_edges']} gold-gold "
        "toy edges)); the patches and their restoring (also on a failure); LTG's rule and DEP's beside pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "calls"))
    ap.add_argument("--name")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "calls":
        return calls()
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
