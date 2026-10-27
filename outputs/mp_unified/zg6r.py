"""G6r (docs/G6R_RELATION_MESSAGES.md): G6's NO_GAIN form, relation-aware messages from text (docs/G6_FREE_GNN.md,
section 5; G6b2's next form, docs/G6B2_ROW_BACKOFF.md, sections 6 and 9), on Kaggle's T4 against KG's base set.

zg6r is zg6 with two changes. On a KB's typed edges (a pool link that carries relations: U2n's typed links) the message
comes from one shared low-rank map conditioned on the edge's relations, in place of zg6's per-family map; and the encoder
(zrm's block encoder and chain match) starts from zspk's trained encoder (KG's zspk fit of the split, its p@swa). Each
relation r is rmatch's centred unit text vector (outputs/m3b/relations); a typed edge's code is
    r_e = the mean over its relations of P_f r on a forward link, P_b r on a backward one, and their mean on both or neither,
P_f and P_b (K_R x 1536, K_R = 16) shared by every relation and KB. In layer k a typed edge u -> v sends
    msg = A_k ((r_e + delta_k) * (B_k h_u)),   its attention logit plus r_e . (C_k ctx_q) / sqrt(K_R),
(A_k D x K_R, B_k K_R x D, delta_k K_R, C_k K_R x D); every other edge is zg6's. P and C start at 0 and delta at 1, so
every logit is zg6's at the start and a typed message is A_k B_k h_u; with no typed edge (the passage graphs) the score is
zg6's bit for bit. The layers are checkpointed in training. 180,218 parameters at hidden 128: zg6's 121,802 and G6r's
58,416.

The fits are zg6r's two on U2d's base build (scr-g6r-zg6r on L-musique, scr-g6r-zg6r-hp on L-hotpotqa, seed 0), trained
and read through zu2d.py's bodies as KG runs them, on KG's device, each read 32,768 rows a batch. In the kernel
(zg6r_session.py runs it, one process per fit):

    python outputs/mp_unified/zg6r.py fit --name N

On the laptop, after `kaggle kernels output` into a new empty folder (KG's BASE_READY and G6b2's verdict filed):

    python outputs/mp_unified/zg6r.py take --from DIR       # each fit's records, reads and relation file, never weights
    python outputs/mp_unified/zg6r.py compare-all           # against KG's zspk (deciding) and KG's zg6 (beside)
    python outputs/mp_unified/zg6r.py verdict               # G6's rule under KG's zspk null, and the beside tables
    python outputs/mp_unified/zg6r.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")         # the kernel's environment sets these per fit; elsewhere two BLAS threads
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zg6b2 as GB2  # noqa: E402  (G6b2's verdict, which names G6r next; through it G6b's and KG's)
import zu2n as U2N  # noqa: E402  (U2n's typed links: their loader, records, batch index and relation table)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

ZB = GB2.ZB
KG = ZB.KG
Z, G6, G2 = KG.Z, KG.G6, KG.G2
Z2, Z3, RM, ZL, SP = KG.Z2, KG.Z3, KG.RM, KG.ZL, KG.SP
S, S2, LG, LC, SR = KG.S, KG.S2, KG.LG, KG.LC, KG.SR
log = KG.log

DECLARED = "docs/G6R_RELATION_MESSAGES.md"
STAGE = "G6r"
KEY = "g6r"                                  # the stamp's key in screen.json, read.json and G6r's compares
ARM = "zg6r"
SEED = 0
BUILD, SPLITS, SUF = KG.BUILD, KG.SPLITS, KG.SUF
FAMS, KT_KEY, LK_KEY, ZCLIP = G6.FAMS, G6.KT_KEY, G6.LK_KEY, G6.ZCLIP
D, LAYERS, LCLIP = G6.D, G6.LAYERS, G6.LCLIP
Q_DIM, SLOTS = RM.Q_DIM, U2N.SLOTS
G6R_KEY = "_g6r"
K_R = 16                                     # the codes' width
SEED_OFF = 3801                              # G6r's own draws (KT 3301, zsp 2301, G2 3501/3502, G6 3601/3602, U2n 3701)
READ_CAP = 32768                             # rows a read batch (whole questions, so the scores do not change)
SCREEN = KG.SCREEN                           # relative: every laptop command runs from the repository's root
OUT = Path("outputs") / "g6r"
ENC_ROOT = ROOT / "outputs" / "g6r" / "enc"  # KG's zspk fit of each split (its models.pt), from the session's dataset
ENC_SHA = {"L-musique": "7998083f15fb57a4333f943881ac8a68433ed7c48591427f89f63b196bc305ab",
           "L-hotpotqa": "52bca67c58fdbd91676493d4ecccb64c62fbeb63d3b34cc19040d1ba02593602"}
ENC_NAMES = ("U", "V", "cm_a", "cm_b", "cm_d", "cm_kappa", "cm_dir", "cm_bias", "cm_pi_w", "cm_pi_b", "cm_beta",
             "l1.weight", "l1.bias", "l2.weight", "l2.bias")            # zrm's block encoder and chain match
ENC_ELEMENTS = 224737                        # their numbers at hidden 128 on the base build's blocks
VERDICT = "screen-verdict"
AGAINST = {0: "zspk", 1: "zg6"}
PAIR = {0: "g6r-zg6r-pair", 1: "g6r-zg6r-vs-zg6-pair"}
RECALL = {k: f"{p}-recall" for k, p in PAIR.items()}
NULL_KIND = {0: "KG's zspk seed null",
             1: "KG's zspk seed null, its seed-0 R@5 replaced by KG's zg6 read (beside)"}
REL_FILE = "g6r_rel.json"
TIMING_FILE = "g6r_timing.json"
SHARE_MIN = G6.SHARE_MIN
REL_SHOW = (("metaqa", None), ("webqsp", 5))  # the codes table: every metaqa relation, webqsp's top five by |P_f r|


def fit_name(split):
    return f"scr-g6r-{ARM}{SUF[split]}"


FITS = {fit_name(sp): sp for sp in SPLITS}
ZSPK, ZG6 = ZB.ZSPK, ZB.ZG6                                         # KG's zspk (deciding) and zg6 (beside), seed 0
CMP = {sp: f"scr-g6r-{ARM}-vs-zspk{SUF[sp]}" for sp in SPLITS}
COMPARES = {CMP[sp]: (fit_name(sp), [ZSPK[sp], ZG6[sp]]) for sp in SPLITS}
KG_ZG6_CMP = ZB.KG_ZG6_CMP                                         # zg6's R@5 as KG filed it
KG_ZRCKP = ZB.KG_ZRCKP                                             # zrckp's seed-0 R@5 (the MLP)
assert all(b in KG.FITS for sp in SPLITS for b in (ZSPK[sp], ZG6[sp]))

RULE = ("ADOPT_FOR_FULL_RUN when zg6r's re-call against KG's zspk under KG's zspk seed null has no LOSS on any of its "
        "twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when KG's base verdict is not "
        "BASE_READY, G6b2's verdict is not NOT_ADOPTED with G6r next, a fit or compare is missing or not G6r's, the "
        "fits are not on KG's device, or a re-call does not hold the twelve reads (G6's rule, docs/G6_FREE_GNN.md "
        "section 4, on the T4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zg6r fitted again on U2s's carried pools, then its full run on all six splits beside "
                              "zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' timing) "
                              "measured again; each its own declared file",
        "GAIN_AND_LOSS": "zg6r under G6b's gate (its edge pass and its no-edge pass mixed per question by the "
                         "question's edge-family shares, as zg6b mixes zg6's; the twin's encoder now starts from "
                         "zspk's); one declared file",
        "NO_GAIN": "the two changes apart: the relation messages with zg6's own encoder (zg6r-m) and zg6 with zspk's "
                   "encoder start (zg6-z), each one fit pair; one declared file",
        "INCOMPLETE": "what is missing is supplied (KG's base set, G6b2's verdict, a fit, a compare or a re-call); "
                      "nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── zspk's encoder: loaded, checked and copied into each zg6r built while it is active ──

ENC = {"state": None, "info": None, "copies": 0}


@contextlib.contextmanager
def using_encoder(state, info):
    """zspk's encoder tensors (ENC_NAMES) active: each zg6r built meanwhile starts from them; ENC['copies'] counts."""
    saved = dict(ENC)
    ENC.update(state=state, info=info, copies=0)
    try:
        yield info
    finally:
        ENC.clear()
        ENC.update(saved)


def rel_path(p):
    p = Path(p)
    return Z.rel(p) if p.is_absolute() and p.is_relative_to(ROOT) else p.as_posix()


def load_encoder(split, root=None, sha=None, elements=ENC_ELEMENTS):
    """(the ENC_NAMES tensors of KG's zspk fit of a split, its p@swa; their record) from root/<split>.pt (that fit's
    models.pt), whose sha256 must be the one pinned here (ENC_SHA)."""
    p = Path(root or ENC_ROOT) / f"{split}.pt"
    if not p.is_file():
        raise SystemExit(f"zg6r: {p} missing (KG's zspk fit of {split}; the session links it from G6r's dataset)")
    h = LC.sha_file(p)
    want = sha or ENC_SHA[split]
    if h != want:
        raise SystemExit(f"zg6r: {p}: sha256 {h}, not KG's zspk fit of {split} ({want})")
    blob = torch.load(p, map_location="cpu", weights_only=True)
    fit_ = KG.fit_name("zspk", 0, split)
    if (blob.get("fit"), blob.get("seed")) != (fit_, 0) or "p@swa" not in blob.get("states", {}):
        raise SystemExit(f"zg6r: {p} is {blob.get('fit')} seed {blob.get('seed')}, not {fit_} seed 0 with a p@swa")
    st = blob["states"]["p@swa"]
    lack = [n for n in ENC_NAMES if n not in st]
    if lack:
        raise SystemExit(f"zg6r: {p}: zspk's p@swa lacks {lack}")
    n = sum(int(st[k].numel()) for k in ENC_NAMES)
    if n != elements:
        raise SystemExit(f"zg6r: {p}: the encoder holds {n} numbers, not {elements}")
    info = {"path": rel_path(p), "sha256": h, "elements": n, "candidate": "p@swa", "fit": blob["fit"],
            "seed": blob["seed"], "hidden": blob.get("hidden"), "basis": blob.get("basis"),
            "basis_sha256": blob.get("basis_sha256")}
    return {k: st[k].detach().clone() for k in ENC_NAMES}, info


@contextlib.contextmanager
def on_encoder(split, root=None):
    st, info = load_encoder(split, root)
    with using_encoder(st, info):
        yield info


def copy_encoder(m, st):
    """The ENC_NAMES tensors of m set to st's (no draw); names or shapes that differ stop."""
    own = m.state_dict(keep_vars=True)
    bad = [n for n in ENC_NAMES if n not in st or n not in own or tuple(st[n].shape) != tuple(own[n].shape)]
    if bad:
        raise SystemExit(f"zg6r: zspk's encoder does not fit zg6r's: {bad}")
    with torch.no_grad():
        for n in ENC_NAMES:
            own[n].copy_(st[n].to(own[n].dtype))


# ── the model ────────────────────────────────────────────────────────────────

G6R_LAYER = tuple(tuple(f"g6r_{k}_{p}" for p in ("a", "b", "d", "c")) for k in range(LAYERS))
G6R_NAMES = ("g6r_pf", "g6r_pb") + tuple(n for lay in G6R_LAYER for n in lay)
N_G6R = 2 * K_R * Q_DIM + LAYERS * (D * K_R + K_R * D + K_R + K_R * D)
N_PARAMS = G6.n_params(128) + N_G6R
CKPT = {"on": True}                          # the layers are checkpointed in training (the selftest also runs without)


def rel_codes(m, rel, crel, cdir, dt):
    """(C, K_R): each combination's code (crel its relation slots, -1 padded; cdir its direction bits): the mean over
    its relations of P_f r (bit 1), P_b r (bit 2), their mean on both bits or none; rel holds the unit relation
    vectors."""
    pf, pb = rel @ m.g6r_pf.T, rel @ m.g6r_pb.T
    valid = crel >= 0
    w = valid.to(dt) / valid.sum(1, keepdim=True).clamp_min(1).to(dt)
    ef = (pf[crel.clamp_min(0)] * w.unsqueeze(2)).sum(1)
    eb = (pb[crel.clamp_min(0)] * w.unsqueeze(2)).sum(1)
    fw = (((cdir & 1) > 0) | (cdir == 0)).to(dt).unsqueeze(1)
    bw = (((cdir & 2) > 0) | (cdir == 0)).to(dt).unsqueeze(1)
    return (fw * ef + bw * eb) / (fw + bw)


def g6r_step(m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, te, codes, cte):
    """Layer k of zg6's stack (zg6.g6_layers' loop, op for op) with each typed edge te's message from the shared
    low-rank map and its attention logit plus its code against the question's context; every other edge is zg6's."""
    wm, wa, fb, ws, wmean, wagg, wx, b = (getattr(m, n) for n in G6.G6_LAYER[k])
    a_k, b_k, d_k, c_k = (getattr(m, n) for n in G6R_LAYER[k])
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
        lg = lg.index_add(0, te, ((rt * Fn.linear(c, c_k)[qv[te]]).sum(1) / math.sqrt(K_R)).to(lg.dtype))
    lg = lg.clamp(-LCLIP, LCLIP)
    mean = torch.zeros(N * FAMS, D, dtype=dt, device=dev).index_add_(
        0, at, torch.sigmoid(lg).unsqueeze(1) * msg).view(N, FAMS, D) / deg
    ex = torch.exp(lg)
    num = torch.zeros(N, D, dtype=dt, device=dev).index_add_(0, v, ex.unsqueeze(1) * msg)
    den = torch.zeros(N, dtype=dt, device=dev).index_add_(0, v, ex)
    att = num / den.clamp_min(1e-30).unsqueeze(1)
    pre = Fn.linear(h, ws) + Fn.linear(mean.reshape(N, FAMS * D), wmean) + Fn.linear(att, wagg) + \
        Fn.linear(xb, wx) + b
    return h + Fn.gelu(Fn.layer_norm(pre, (D,)))


def g6r_layers(m, h, xb, w, u, v, f, nq, B, te, codes, cte):
    """zg6's LAYERS layers with the typed messages (g6r_step), checkpointed in training."""
    N = h.shape[0]
    dev, dt = h.device, h.dtype
    at = v * FAMS + f
    one = torch.ones(u.numel(), dtype=dt, device=dev)
    deg = torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, one).view(N, FAMS, 1).clamp_min(1.0)
    qv = nq[v]
    ck = CKPT["on"] and m.training and torch.is_grad_enabled()
    for k in range(LAYERS):
        args = (m, k, h, xb, w, u, v, f, at, deg, qv, nq, B, te, codes, cte)
        h = checkpoint(g6r_step, *args, use_reentrant=False, preserve_rng_state=False) if ck else g6r_step(*args)
    return h


def g6r_stack(m, H, xb, lk, gd, nq, B, u01=None):
    """zg6r's score (N,): zg6.g6_stack with the typed messages; the family dropout (u01, in training) drops a typed
    edge with its family (0)."""
    xb = xb.to(H.dtype)
    h0 = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b))
    s0 = Fn.linear(h0, m.g6_ctx_w).squeeze(-1)
    z0 = LG.seg_zscore8D(s0.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
    w = G2.seg_softmax_rows(z0, nq, B).unsqueeze(1)
    u, v, f, ci = lk["u"], lk["v"], lk["f"], gd["ci"]
    if u01 is not None:
        ke = G2.family_keep(f, nq[v], B, u01)
        u, v, f, ci = u[ke], v[ke], f[ke], ci[ke]
    te = torch.nonzero(ci >= 0).view(-1)
    codes = cte = None
    if te.numel():
        if gd["rel"] is None:
            raise RuntimeError("zg6r: typed edges in the batch without their relation vectors")
        codes = rel_codes(m, gd["rel"], gd["crel"], gd["cdir"], H.dtype)
        cte = ci[te]
    h = g6r_layers(m, h0, xb, w, u, v, f, nq, B, te, codes, cte)
    return (Fn.linear(h, m.g6_out_w) + Fn.linear(h0, m.g6_skip_w) + m.g6_out_b).squeeze(-1)


class ZG6R(G6.G6Free):
    """zg6r: zg6 with each typed edge's message from its relations' shared low-rank map; its encoder copied from
    zspk's when one is active (using_encoder)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        uni = lambda *shape, fan: (torch.rand(*shape, generator=g) * 2 - 1) / fan ** 0.5  # noqa: E731
        self.g6r_pf = nn.Parameter(torch.zeros(K_R, Q_DIM))
        self.g6r_pb = nn.Parameter(torch.zeros(K_R, Q_DIM))
        for na, nb, nd, nc in G6R_LAYER:
            setattr(self, na, nn.Parameter(uni(D, K_R, fan=K_R)))
            setattr(self, nb, nn.Parameter(uni(K_R, D, fan=D)))
            setattr(self, nd, nn.Parameter(torch.ones(K_R)))
            setattr(self, nc, nn.Parameter(torch.zeros(K_R, D)))
        if ENC["state"] is not None:
            copy_encoder(self, ENC["state"])
            ENC["copies"] += 1

    def forward(self, feats, keep, nq, B, base_z):
        """The score; base_z (rrf's z-score, the MLP's base) is never read."""
        if KT_KEY not in feats:
            raise RuntimeError("zg6r: no KT input in the batch (P1c3's carve adds it)")
        if LK_KEY not in feats:
            raise RuntimeError("zg6r: no pool edges in the batch (zlink's carve adds them)")
        if G6R_KEY not in feats:
            raise RuntimeError("zg6r: no typed links in the batch (G6r's carve adds them)")
        H = self.encode(feats, keep, nq, B)
        xb = torch.cat([feats[KT_KEY].to(H.dtype), self.chain(feats, keep, nq).to(H.dtype)], 1)
        u01 = torch.rand(B, FAMS, generator=self.g6_gen) if self.training else None
        return g6r_stack(self, H, xb, feats[LK_KEY], feats[G6R_KEY], nq, B, u01)


# ── the carve: P1c3's link carve with each batch's typed links (U2n's) ──────


def g6r_carve(base, typed_root=U2N.TYPED_ROOT):
    """base's link carve with G6r's side key: a KB carve's typed links (U2n's, aligned with the carve's link edges and
    checked against U2n's records; None on a passage graph) and the unit centred relation vectors, as zg6e's carve
    holds them. Every block, row, side key and gold is base's."""

    class G6rCarve(base):
        G6R_TYPED_ROOT = typed_root

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.u2n_links = self.links
            self.u2n_typed = U2N.load_typed(self, type(self).G6R_TYPED_ROOT, cache_root, self.u2n_links, verify)
            self.g6r_rel = None if self.u2n_typed is None else Fn.normalize(
                self.chains["rel_centered"].to(torch.float32), dim=1)

        def nbytes(self):
            ts = [] if self.g6r_rel is None else [self.g6r_rel]
            return super().nbytes() + sum(t.numel() * t.element_size() for t in ts)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            idx, _eq, _rseg = U2N.edge_index(self.u2n_links, self.n_np, qs)
            if idx.size != feats[LK_KEY]["u"].numel():
                raise SystemExit(f"zg6r: {self.ds}/{self.carve}: {idx.size} typed-link indices for "
                                 f"{feats[LK_KEY]['u'].numel()} link edges")
            ci, crel, cdir = U2N.typed_batch(self, idx)
            feats[G6R_KEY] = {"ci": ci, "crel": crel, "cdir": cdir, "rel": self.g6r_rel}
            return feats, nq, base_z, gold

    G6rCarve.__name__ = G6rCarve.__qualname__ = f"G6r{base.__name__}"
    return G6rCarve


G6rLinkCarve = g6r_carve(Z3.KTLinkCarveBase)
ARMS = {ARM: (ZG6R, G6rLinkCarve)}
if S.ARMS.get(ARM) not in (None, ARMS[ARM]):
    raise SystemExit(f"zg6r: lean_screen's arm {ARM} is already {S.ARMS[ARM]}")
S.ARMS.update(ARMS)
SETTINGS = {"network": "zg6's (docs/G6_FREE_GNN.md, section 2)", "zg6": G6.SETTINGS,
            "relations": "rmatch's centred unit text vectors of the KB's relations (outputs/m3b/relations), "
                         "normalised again, as zg6e's carve holds them",
            "code": "r_e = the mean over the edge's relation slots of P_f r (forward link), P_b r (backward), their "
                    "mean (both bits or none); P_f and P_b K_R x 1536, shared by every relation and KB",
            "typed_message": "A_k ((r_e + delta_k) * (B_k h_u)) in place of zg6's W_f h_u, per layer",
            "typed_logit": "zg6's plus r_e . (C_k ctx_q) / sqrt(K_R), per layer, before the clip",
            "start": "P and C 0, delta 1, A and B uniform (fan K_R and D) from the seed plus 3801: every logit zg6's, "
                     "a typed message A_k B_k h_u; zg6's score bit for bit with no typed edge",
            "encoder": "zrm's block encoder and chain match (ENC_NAMES, 224,737 numbers) copied from KG's zspk fit of "
                       "the split (p@swa) when zg6r is built, then trained",
            "k_r": K_R, "seed_offset": SEED_OFF, "checkpointed_layers": True, "g6r_parameters": N_G6R,
            "parameters_at_hidden_128": N_PARAMS, "read_rows": READ_CAP,
            "typed_carves": [list(x) for x in U2N.TYPED_CARVES]}


# ── the device, train and read (KG's, with G6r's arm, encoder, names and stamp) ──


def where_ok(env, what):
    if not KG.ok_where(env):
        raise SystemExit(f"zg6r: G6r {what}s in a Kaggle kernel on a {KG.GPU} from the image {KG.IMAGE_TAG} (KG's "
                         f"device), not with {KG.device_of(env)} ({env.get('where')})")


def check_arm():
    if S.ARMS.get(ARM) != ARMS[ARM]:
        raise SystemExit(f"zg6r: the arm {ARM} is {S.ARMS.get(ARM)}, not G6r's {ARMS[ARM]}")


def code_shas():
    return {**GB2.code_shas(), "zu2n_sha256": LC.sha_src(U2N.__file__), "zg6r_sha256": LC.sha_src(__file__)}


def head():
    return {"model": SETTINGS, "kt_root": Z.rel(Z.roots(BUILD)["kt"]), "typed_root": Z.rel(U2N.TYPED_ROOT),
            "enc_root": Z.rel(ENC_ROOT)}


def encoder_used(name, enc, out_root=None):
    """After a fit trained: a zg6r was built from zspk's encoder, and the fit's basis and width are the encoder's."""
    if enc.get("copies", 0) < 1:
        raise SystemExit(f"zg6r: {name}: no zg6r was built while zspk's encoder was active (the fit stays unstamped)")
    p = Path(out_root or S.OUT / "fits") / name / "models.pt"
    blob = torch.load(p, map_location="cpu", weights_only=False)
    if (blob.get("basis_sha256"), blob.get("hidden")) != (enc["basis_sha256"], enc["hidden"]):
        raise SystemExit(f"zg6r: {name} was fitted on the basis {blob.get('basis')} ({blob.get('basis_sha256')}) at "
                         f"hidden {blob.get('hidden')}, zspk's encoder on {enc['basis']} ({enc['basis_sha256']}) at "
                         f"hidden {enc['hidden']} (the fit stays unstamped)")


def train(name, env=None):
    """One of G6r's two fits on U2d's base build: zu2d.train's body as KG runs it, with G6r's arm, zspk's encoder
    active, its name and stamp."""
    if name not in FITS:
        raise SystemExit(f"zg6r: {name} is not one of G6r's fits ({', '.join(FITS)})")
    split = FITS[name]
    check_arm()
    env = KG.device_env() if env is None else env
    where_ok(env, "train")
    argv = ["train", "--name", name, "--arm", ARM, "--device", "cuda", "--cache-root", KG.cache_arg()]
    with Z.on_u2d(BUILD) as r, on_encoder(split) as ei:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv) + ["--seed", str(SEED)]
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.train_carves(split), r["kt"])
        ty = U2N.typed_records(Z3.train_carves(split))
        rc = RM.train(argv, split)
        enc = {**ei, "copies": ENC["copies"]}
        if rc == 0:
            encoder_used(name, enc)
        stamp_fit(name, env, g, kt, ty, enc, r)
    log(f"zg6r {name}: zspk's encoder ({enc['fit']}, {enc['elements']} numbers) copied into {enc['copies']} zg6r")
    return rc


def stamp_fit(name, env, g, kt, ty, enc, r, out_root=None):
    split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    if not sj.exists():
        return
    rec = json.loads(sj.read_text(encoding="utf-8"))
    rec.update({**code_shas(), KEY: {
        "declared_in": DECLARED, "fit": name, "build": BUILD, "arm": ARM, "seed": SEED, "split": split,
        "kg_bases": {"zspk": ZSPK[split], "zg6": ZG6[split]}, "device": KG.device_of(env), "env": env,
        "graph": Z.GRAPH[BUILD], "pools": g.get("construction"), "rule": Z.ZP.RULE[BUILD],
        "cache_root": Z.rel(r["cache"]), "gate": LC.sha_file(r["gate"]), "head": head(), "kt_records": kt,
        "typed_records": ty, "encoder": enc, "rmatch_records": RM.built_records(),
        "zrc_records": Z2._BUILT(r["ch_zrc"]), "zlink_records": ZL.built_records()}})
    LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    """A G6r fit's screen.json and its G6r record, checked against the fit's name."""
    split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("build"), k.get("arm"), k.get("seed"),
            k.get("split")) != (ARM, DECLARED, name, BUILD, ARM, SEED, split):
        raise SystemExit(f"zg6r: {sj} is not G6r's {ARM} fit of {split}")
    return rec, k


def read(name, env=None):
    """The fit read with p@swa on the six s1eval carves of U2d's base build, on the device it was fitted on, READ_CAP
    rows a batch."""
    if name not in FITS:
        raise SystemExit(f"zg6r: {name} is not one of G6r's fits")
    split = FITS[name]
    env = KG.device_env() if env is None else env
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    if KG.device_of(env) != k.get("device"):
        raise SystemExit(f"zg6r: {name} was fitted on {k.get('device')}; this read runs on {KG.device_of(env)}")
    argv = ["--name", name, "--device", "cuda", "--cache-root", KG.cache_arg(), "--rows-cap", str(READ_CAP)]
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.eval_carves(), r["kt"])
        ty = U2N.typed_records(Z3.eval_carves())
        rc = RM.read(argv)
    rj = S.OUT / "fits" / name / "read.json"
    if rj.exists():
        rec = json.loads(rj.read_text(encoding="utf-8"))
        rec.update({**code_shas(), KEY: {
            "declared_in": DECLARED, "fit": name, "build": BUILD, "arm": ARM, "seed": SEED, "split": split,
            "device": KG.device_of(env), "env": env, "gate": LC.sha_file(Z.roots(BUILD)["gate"]),
            "pools": g.get("construction"), "kt_records": kt, "typed_records": ty, "rows_cap": READ_CAP}})
        LC.write_json(rj, rec)
    return rc


def stamped(p, name):
    try:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    k = rec.get(KEY) or {}
    return k.get("declared_in") == DECLARED and k.get("fit") == name


# ── the relation map, beside: P, the codes, metaqa's span, the layers and the encoder's drift ──


def span(C, tol=1e-6):
    """(an orthonormal basis of the span of C's rows, as rows; its rank)."""
    if C.shape[0] == 0:
        return C[:0], 0
    _u, s, vt = np.linalg.svd(C, full_matrices=False)
    r = int((s > tol * max(float(s[0]), 1e-30)).sum())
    return vt[:r], r


def stat3(x):
    x = np.asarray(x, np.float64).reshape(-1)
    return {"mean": round(float(x.mean()), 4), "min": round(float(x.min()), 4), "max": round(float(x.max()), 4)}


def rel_stats(st, rel_dir=None, show=REL_SHOW, enc=None):
    """G6r's relation map of a p@swa state: P_f's and P_b's norms and top singular values; each shown KB's codes
    |P_f r| and |P_b r| (rel_table's unit centred vectors, top ones by |P_f r|); metaqa's centred span (its rank), the
    share of P's norm inside it and the mean share of each later KB's relations inside it; per layer delta's mean, min
    and max and the norms of A, B and C; with enc (zspk's ENC_NAMES tensors), the encoder's drift
    |theta - theta_zspk| / |theta_zspk|."""
    pf, pb = (st[n].detach().double().numpy() for n in ("g6r_pf", "g6r_pb"))
    nf, nb = float(np.linalg.norm(pf)), float(np.linalg.norm(pb))
    out = {"p_f_norm": round(nf, 6), "p_b_norm": round(nb, 6),
           "p_f_sv": [round(float(x), 6) for x in np.linalg.svd(pf, compute_uv=False)[:8]],
           "p_b_sv": [round(float(x), 6) for x in np.linalg.svd(pb, compute_uv=False)[:8]],
           "p_f_in_metaqa_span": None, "p_b_in_metaqa_span": None, "by_dataset": {}}
    Q = None
    for ds, top in show:
        C, names = U2N.rel_table(ds, rel_dir)
        cf, cb = np.linalg.norm(C @ pf.T, axis=1), np.linalg.norm(C @ pb.T, axis=1)
        order = np.argsort(-cf, kind="stable")[:(top or len(names))]
        x = {"relations": len(names), "shown": top or len(names),
             "codes": [[names[i], round(float(cf[i]), 4), round(float(cb[i]), 4)] for i in order]}
        if ds == "metaqa":
            Q, x["span_rank"] = span(C)
            out["p_f_in_metaqa_span"] = round(float(np.linalg.norm(pf @ Q.T)) / max(nf, 1e-30), 4)
            out["p_b_in_metaqa_span"] = round(float(np.linalg.norm(pb @ Q.T)) / max(nb, 1e-30), 4)
        elif Q is not None:
            x["in_metaqa_span"] = round(float(((C @ Q.T) ** 2).sum(1).mean()), 4)
        out["by_dataset"][ds] = x
    out["layers"] = [{"delta": stat3(st[nd].detach().numpy()),
                      **{f"{p}_norm": round(float(st[n].detach().double().norm()), 4)
                         for p, n in (("a", na), ("b", nb_), ("c", nc))}} for na, nb_, nd, nc in G6R_LAYER]
    out["encoder_drift"] = None
    if enc is not None:
        num = sum(float(((st[k].detach().double() - enc[k].double()) ** 2).sum()) for k in ENC_NAMES)
        den = sum(float((enc[k].double() ** 2).sum()) for k in ENC_NAMES)
        out["encoder_drift"] = round(math.sqrt(num / max(den, 1e-30)), 6)
    return out


def rel_beside(name, fits_root=None, rel_dir=None, show=REL_SHOW, enc=None, enc_root=None):
    """g6r_rel.json in the fit's folder: the relation map of its p@swa state (models.pt), with the encoder's drift from
    zspk's when its file is there (or enc is given)."""
    split = FITS[name]
    fd = Path(fits_root or S.OUT / "fits") / name
    blob = torch.load(fd / "models.pt", map_location="cpu", weights_only=False)
    if enc is None and (Path(enc_root or ENC_ROOT) / f"{split}.pt").is_file():
        enc, _info = load_encoder(split, enc_root)
    rec = {"stage": STAGE, "declared_in": DECLARED, "fit": name, "split": split, "candidate": "p@swa",
           **rel_stats(blob["states"]["p@swa"], rel_dir, show, enc)}
    LC.write_json(fd / REL_FILE, rec)
    return rec


def fit(name, rows_cap=0):
    """One fit in the kernel: a read G6r stamped, with its relation file, is kept; a fit trained and stamped but not
    read is only read; otherwise the fit's files are unlinked and it trains and reads. Then the relation file;
    g6r_timing.json holds the seconds and peaks. The read's rows a batch are READ_CAP; another cap stops."""
    if name not in FITS:
        raise SystemExit(f"zg6r: {name} is not one of G6r's fits")
    if rows_cap and int(rows_cap) != READ_CAP:
        raise SystemExit(f"zg6r: {name} reads with {READ_CAP} rows a batch (section 3), not {rows_cap}")
    fd = S.OUT / "fits" / name
    if stamped(fd / "read.json", name) and (fd / REL_FILE).exists():
        log(f"zg6r fit {name}: read before, kept")
        return 0
    env = KG.device_env()
    where_ok(env, "train")
    t0 = time.time()
    trained = stamped(fd / "screen.json", name) and (fd / "train.json").exists() and (fd / "models.pt").exists()
    if not trained:
        KG.clear(fd, "*")
        KG.clear(fd / "reads", "*")
        rc = train(name, env)
        if rc:
            return rc
    t1 = time.time()
    rc = 0
    if not stamped(fd / "read.json", name):
        KG.clear(fd / "reads", "*")
        KG.clear(fd, "read.json")
        rc = read(name, env)
    t2 = time.time()
    if rc == 0:
        rel_beside(name)
    LC.write_json(fd / TIMING_FILE, {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                     "read_seconds": round(t2 - t1, 1), "rel_seconds": round(time.time() - t2, 1),
                                     "peak_gb": KG.peak_gb(), "rows_cap": READ_CAP, "exit": rc,
                                     "declared_in": DECLARED})
    return rc


# ── the laptop: take and the compares ────────────────────────────────────────

NEVER = KG.NEVER
FITS_ROOT = SCREEN / "fits"                  # the laptop's fits, relative: its commands run from the repository's root


def fit_device(name, out_root=None):
    """A G6r fit's device: its screen.json's G6r record, and its read.json's on the same device."""
    out_root = Path(out_root or FITS_ROOT)
    _rec, k = fit_stamp(name, out_root)
    rj = out_root / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    rk = r.get(KEY) or {}
    if rk.get("declared_in") != DECLARED or rk.get("fit") != name or rk.get("device") != k.get("device"):
        raise SystemExit(f"zg6r: {rj} is not G6r's read of {name} on the device it was fitted on")
    return k["device"]


def take(src, out_root=None):
    """Each G6r fit under src (a kernel's output: fits/<name>, or <name>) copied to outputs/screen/fits/<name>: its
    records, reads and relation file, never its weights. A fit must be stamped and read by G6r; a file there already
    must be the same bytes. Other stages' fits in the same output are theirs to take."""
    src = Path(src)
    base = src / "fits" if (src / "fits").is_dir() else src
    dst_root = Path(out_root or FITS_ROOT)
    took, kept = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("scr-g6r-")):
        if d.name not in FITS:
            raise SystemExit(f"zg6r take: {d.name} is not one of G6r's fits")
        fit_device(d.name, base)
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            if f.name in NEVER or f.suffix in (".tmp", ".part"):
                continue
            t = dst_root / d.name / f.relative_to(d)
            if t.exists():
                if LC.sha_file(t) != LC.sha_file(f):
                    raise SystemExit(f"zg6r take: {t} exists with other bytes")
                kept.append(str(t))
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(str(t))
    log(f"zg6r take: {len(took)} files copied, {len(kept)} there before")
    return took


def one_device(new, bases, out_root=None):
    devs = {new: fit_device(new, out_root), **{b: KG.fit_device(b, out_root) for b in bases}}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zg6r: G6r's fit and its bases ran on different devices or images: {devs}")
    return devs[new]


def compare_cmd(cname, out_root=None, screen=None):
    """lean_screen's compare of a G6r fit with KG's zspk (first: it decides) and KG's zg6, one device, the paths
    relative to the repository, stamped with G6r's record."""
    new, bases = COMPARES[cname]
    dev = one_device(new, bases, out_root)
    fits = Path(out_root or FITS_ROOT)
    out = Path(screen or SCREEN) / cname
    with Z.at_root():
        def r_(p):
            p = Path(p)
            return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else str(p)
        rc = S2.main(["compare", "--new", r_(fits / new), "--base", ",".join(r_(fits / b) for b in bases),
                      "--out", r_(out)])
    js = out.with_suffix(".json")
    rec = json.loads(js.read_text(encoding="utf-8"))
    rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                "zg6r_sha256": LC.sha_src(__file__)}
    LC.write_json(js, rec)
    return rc


def compare_all(out_root=None, screen=None):
    done, ran = [], []
    for cname, (new, bases) in COMPARES.items():
        js = Path(screen or SCREEN) / f"{cname}.json"
        if js.exists():
            k = json.loads(js.read_text(encoding="utf-8")).get(KEY) or {}
            if k.get("declared_in") == DECLARED and k.get("new") == new and k.get("bases") == list(bases):
                done.append(cname)
                continue
        compare_cmd(cname, out_root, screen)
        ran.append(cname)
    log(f"zg6r compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """A G6r compare's rows {base fit: {dataset: row}}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.exists():
        raise SystemExit(f"zg6r: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"zg6r: {p} is not G6r's compare {cname} of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"zg6r: {p} was made on {k.get('device')}, not on the fits' device {dev}")
    got = [SR.parts_of(b)[-1:] for b in rec.get("bases") or []]
    if SR.parts_of(rec.get("new", ""))[-1:] != (new,) or got != [(b,) for b in bases]:
        raise SystemExit(f"zg6r: {p} compares {rec.get('new')} with {rec.get('bases')}, not {new} with {bases}")
    split = FITS[new]
    if sorted(rec.get("trained_on", [])) != sorted(LG.FITS[split]):
        raise SystemExit(f"zg6r: {p} trained on {rec.get('trained_on')}, not {split}'s {sorted(LG.FITS[split])}")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"zg6r: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
    out = {}
    for b, bp in zip(bases, rec["bases"]):
        rows = {r["dataset"]: r for r in rec["by_base"][bp]["rows"]}
        if sorted(rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zg6r: {p} reads {sorted(rows)} against {b}, not the six datasets")
        out[b] = rows
    return rec, out


# ── the verdict ──────────────────────────────────────────────────────────────


def g6b2_filed(g6b2_out=None):
    """Problems unless G6b2's filed verdict is NOT_ADOPTED under G6b2's declaration with G6r its next form."""
    p = Path(g6b2_out or GB2.OUT) / f"{GB2.VERDICT}.json"
    if not p.exists():
        return [f"{p} missing (G6b2's verdict)"]
    gv = json.loads(p.read_text(encoding="utf-8"))
    if (gv.get("stage"), gv.get("declared_in"), gv.get("verdict"), gv.get("next_form")) != \
            (GB2.STAGE, GB2.DECLARED, "NOT_ADOPTED", GB2.NEXT["NO_GAIN"]):
        return [f"G6b2's verdict is {gv.get('verdict')} ({gv.get('declared_in')}) with next form "
                f"{str(gv.get('next_form'))[:40]!r}..., not NOT_ADOPTED with G6r next"]
    return []


def restamp(od, which):
    """G6r's fields on the pair and re-call screen_pair and screen_recall wrote; a header on their markdown."""
    for name in (PAIR[which], RECALL[which]):
        js = (od / name).with_suffix(".json")
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "against": AGAINST[which],
                    "null_kind": NULL_KIND[which], "deciding": which == 0, "zg6r_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)
        md = js.with_suffix(".md")
        if md.exists():
            head_ = (f"G6r ({DECLARED}): zg6r against KG's {AGAINST[which]} on the T4"
                     + (", deciding." if which == 0 else ", beside, deciding nothing.") + "\n\n")
            md.write_text(head_ + md.read_text(encoding="utf-8"), encoding="utf-8")


def recall(scr, od, dev, which):
    """screen_pair over G6r's two compares against base `which` (0: KG's zspk, deciding; 1: KG's zg6, beside), then
    screen_recall under KG's zspk null. Beside, the null's seed-0 R@5 is zg6's read as KG filed it (scr-kg-zg6-vs-
    zspk's new column), which the compare's zg6 column must equal; the floors stay zspk's."""
    def load(fn):
        p = Path(fn)
        c = next(c for c in COMPARES if p.stem == c)
        _rec, rows = cmp_rows(p, c, dev)
        return FITS[COMPARES[c][0]], [rows[COMPARES[c][1][which]][ds] for ds in LG.EVAL_ORDER]

    saved = SP.load
    SP.load = load
    try:
        SP.pair([scr / f"{CMP[sp]}.json" for sp in SPLITS], od / PAIR[which])
    finally:
        SP.load = saved
    null, sources = KG.kg_null("zspk", scr, dev)
    if which == 1:
        for sp in SPLITS:
            _r, kz = KG.cmp_rows(scr / f"{KG_ZG6_CMP[sp]}.json", KG_ZG6_CMP[sp], dev)
            null[sp] = {ds: {**v, "seed0": kz[ds]["new"][0]} for ds, v in null[sp].items()}
            sources[f"KG zg6 {sp}"] = str(scr / f"{KG_ZG6_CMP[sp]}.json")
    saved = SR.load_null
    SR.load_null = lambda _files: (null, sources)
    try:
        rec = SR.pair([], (od / PAIR[which]).with_suffix(".json"), od / RECALL[which])
    finally:
        SR.load_null = saved
    restamp(od, which)
    return rec


def tag(r):
    return f"{r['split']} {r['dataset']}"


def timing(name, fits_root):
    d = Path(fits_root) / name
    t = json.loads((d / "train.json").read_text(encoding="utf-8")) if (d / "train.json").exists() else {}
    r = json.loads((d / "read.json").read_text(encoding="utf-8")) if (d / "read.json").exists() else {}
    k = json.loads((d / TIMING_FILE).read_text(encoding="utf-8")) if (d / TIMING_FILE).exists() else {}
    return {"train_seconds": t.get("seconds"), "read_seconds": r.get("seconds"), "train_peak_gpu_gb": t.get("peak_gpu_gb"),
            "read_peak_gpu_gb": r.get("peak_gpu_gb"), "data_gb": t.get("data_gb"), "peak_gb": k.get("peak_gb"),
            "rel_seconds": k.get("rel_seconds")}


fmt = KG.fmt


def calls_of(rc):
    c = {tag(r): r["call"] for r in rc["rows"]}
    return c, [k for k, x in c.items() if x == "GAIN"], [k for k, x in c.items() if x == "LOSS"]


def rel_md(sp, x):
    """The relation file of a split's fit as markdown lines."""
    if x is None:
        return [f"- {sp}: no relation file."]
    out = [f"- {sp}: |P_f| {fmt(x['p_f_norm'])}, |P_b| {fmt(x['p_b_norm'])}; in metaqa's centred span "
           f"{fmt(x['p_f_in_metaqa_span'])} and {fmt(x['p_b_in_metaqa_span'])} of their norms; top singular values "
           f"{x['p_f_sv'][:4]} and {x['p_b_sv'][:4]}; the encoder's drift from zspk's {fmt(x['encoder_drift'])}."]
    for ds, y in x["by_dataset"].items():
        extra = (f", span rank {y['span_rank']}" if "span_rank" in y else "") + \
            (f", {fmt(y['in_metaqa_span'])} of each relation inside metaqa's span on average"
             if "in_metaqa_span" in y else "")
        out.append(f"  - {ds} ({y['relations']} relations{extra}): |P_f r| / |P_b r| " +
                   "; ".join(f"{n} {fmt(a)} / {fmt(b)}" for n, a, b in y["codes"]) + ".")
    out.append("  - per layer: " + "; ".join(
        f"{k}: delta {fmt(ly['delta']['mean'])} ({fmt(ly['delta']['min'])} to {fmt(ly['delta']['max'])}), |A| "
        f"{fmt(ly['a_norm'])}, |B| {fmt(ly['b_norm'])}, |C| {fmt(ly['c_norm'])}" for k, ly in enumerate(x["layers"]))
        + ".")
    return out


def verdict(screen=None, out=None, out_root=None, kg_out=None, g6b2_out=None):
    """G6r's verdict by G6's rule from zg6r's re-call against KG's zspk under KG's zspk null (sections 4 and 6).
    Beside, deciding nothing: zg6r against KG's zg6 under zspk's floors, R@5 per read with zrckp's and the MLP share,
    the relation map of each fit, and seconds and memory beside KG's zg6 and zspk."""
    scr, od = Path(screen or SCREEN), Path(out or OUT)
    fits_root = Path(out_root or FITS_ROOT)
    od.mkdir(parents=True, exist_ok=True)
    kdev, problems = ZB.kg_device(kg_out)
    problems += g6b2_filed(g6b2_out)
    devs = {}
    for n in FITS:
        try:
            devs[n] = fit_device(n, out_root)
        except SystemExit as e:
            problems.append(str(e))
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) > 1:
        problems.append(f"G6r's fits ran on {len(keys)} devices: {sorted(keys)}")
    dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
    if dev is not None and kdev is not None and dev != kdev:
        problems.append(f"G6r's fits ran on {dev}, KG's base set on {kdev}")
    rows = {}
    for c in COMPARES:
        try:
            rows[c] = cmp_rows(scr / f"{c}.json", c, dev)[1]
        except SystemExit as e:
            problems.append(str(e))
    rc = {}
    if not problems:
        try:
            for which in AGAINST:
                rc[which] = recall(scr, od, dev, which)
        except SystemExit as e:
            problems.append(str(e))
    whole = sorted((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER)
    for which, r in rc.items():
        if sorted((x["split"], x["dataset"]) for x in r["rows"]) != whole:
            problems.append(f"the re-call against {AGAINST[which]} holds {len(r['rows'])} reads, not the twelve of "
                            f"{', '.join(SPLITS)}")
    rrows = rc[0]["rows"] if 0 in rc and not problems else []
    gains = [tag(r) for r in rrows if r["call"] == "GAIN"]
    losses = [tag(r) for r in rrows if r["call"] == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "base_arm": "zspk",
           "null_kind": NULL_KIND[0], "device": dev, "next_form": nxt, "gains": gains, "losses": losses,
           "problems": problems, "calls": {tag(r): r["call"] for r in rrows},
           "delta_r5": {tag(r): r["delta"] for r in rrows}, "zg6r_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# G6r's screen verdict: zg6r **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zg6r** (zg6 with each typed KB edge's message from its relations' shared low-rank map, its encoder "
          f"started from zspk's) against KG's zspk on the T4, re-called under KG's zspk seed null: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ".",
          f"- **Next form** (fixed in section 6 before any number): {nxt}.", f"- Device: {json.dumps(dev)}."]
    beside = {}
    if v != "INCOMPLETE":
        zrckp = {}
        for sp in SPLITS:
            _r, zr = KG.cmp_rows(scr / f"{KG_ZRCKP[sp]}.json", KG_ZRCKP[sp], dev)
            zrckp[sp] = {ds: zr[ds]["base"][0] for ds in LG.EVAL_ORDER}
        vz, gz, lz = calls_of(rc[1])
        r5, share = {}, {}
        for r in rrows:
            sp, ds = r["split"], r["dataset"]
            rw = rows[CMP[sp]]
            r5[tag(r)] = {"read": r["read"], "rrf": r["rrf"], "zspk": r["base"], "zg6": rw[ZG6[sp]][ds]["base"][0],
                          "zg6r": r["new"], "zrckp": zrckp[sp][ds], "floor": r["floor"], "call": r["call"],
                          "call_vs_zg6": vz[tag(r)]}
            share[tag(r)] = G6.mlp_share(zrckp[sp][ds], r["new"], r["rrf"])
        rel = {}
        for n, sp in FITS.items():
            p = fits_root / n / REL_FILE
            rel[sp] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        tm = {sp: {"zg6r": timing(n, fits_root), "zg6": KG.timing(ZG6[sp], fits_root),
                   "zspk": KG.timing(ZSPK[sp], fits_root)} for n, sp in FITS.items()}
        beside = {"vs_zg6": {"gains": gz, "losses": lz}, "r5": r5, "mlp_share": share, "rel": rel, "timing": tm,
                  "pair": (od / PAIR[0]).with_suffix(".json").as_posix(),
                  "recall": (od / RECALL[0]).with_suffix(".json").as_posix(),
                  "recall_sha256": LC.sha_file((od / RECALL[0]).with_suffix(".json"))}
        md += ["", "Beside, deciding nothing.", "",
               f"- zg6r against KG's zg6 under zspk's floors: {len(gz)} GAIN ({', '.join(gz) or 'none'}), {len(lz)} "
               f"LOSS ({', '.join(lz) or 'none'}).",
               "", f"R@5 per read on the T4; the MLP share = (zrckp - rrf) / (zg6r - rrf) where zg6r's gain over rrf "
                   f"exceeds {SHARE_MIN}:", "",
               "| read | rrf | zspk | zg6 | zg6r | zrckp (MLP) | MLP share | floor | call | vs zg6 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for k, t in r5.items():
            md.append(f"| {k} ({t['read']}) | {fmt(t['rrf'])} | {fmt(t['zspk'])} | {fmt(t['zg6'])} | {fmt(t['zg6r'])} | "
                      f"{fmt(t['zrckp'])} | {fmt(share[k])} | {fmt(t['floor'])} | {t['call']} | {t['call_vs_zg6']} |")
        md += ["", "The relation map of each fit's p@swa: P's norms and top singular values, the share of their norm in "
                   "metaqa's centred relation span (the only relations a fit trains on), each KB's codes |P_f r| / "
                   "|P_b r|, the layers' delta and norms, and the encoder's drift from zspk's:", ""]
        for sp in SPLITS:
            md += rel_md(sp, rel[sp])
        md += ["", "Seconds and memory: zg6r, and in brackets KG's zg6 and zspk of the same split, all on the T4:", "",
               "| split | train s | read s | peak GPU GB, train / read | peak RAM GB |", "|---|---:|---:|---|---:|"]

        def s0(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
        for sp in SPLITS:
            a_, b_, c_ = tm[sp]["zg6r"], tm[sp]["zg6"], tm[sp]["zspk"]
            md.append(f"| {sp} | {s0(a_['train_seconds'])} ({s0(b_['train_seconds'])}, {s0(c_['train_seconds'])}) | "
                      f"{s0(a_['read_seconds'])} ({s0(b_['read_seconds'])}, {s0(c_['read_seconds'])}) | "
                      f"{fmt(a_['train_peak_gpu_gb'])} / {fmt(a_['read_peak_gpu_gb'])} ({fmt(b_['train_peak_gpu_gb'])} / "
                      f"{fmt(b_['read_peak_gpu_gb'])}, {fmt(c_['train_peak_gpu_gb'])} / {fmt(c_['read_peak_gpu_gb'])}) | "
                      f"{fmt(a_['peak_gb'])} ({fmt(b_['peak_gb'])}, {fmt(c_['peak_gb'])}) |")
    rec["beside"] = beside
    LC.write_json(od / f"{VERDICT}.json", rec)
    (od / f"{VERDICT}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zg6r verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''}); "
        f"next: {nxt}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────

TOY_ENV = KG.TOY_ENV
stops = KG.stops


def fake_rel():
    lay = {"delta": {"mean": 1.0, "min": 0.9, "max": 1.1}, "a_norm": 1.0, "b_norm": 1.0, "c_norm": 0.1}
    return {"p_f_norm": 0.5, "p_b_norm": 0.25, "p_f_sv": [0.4, 0.2], "p_b_sv": [0.2, 0.1], "p_f_in_metaqa_span": 0.9,
            "p_b_in_metaqa_span": 0.8, "encoder_drift": 0.12, "layers": [lay] * LAYERS,
            "by_dataset": {"metaqa": {"relations": 9, "shown": 9, "span_rank": 8, "codes": [["directed by", 0.3, 0.2]]},
                           "webqsp": {"relations": 100, "shown": 5, "in_metaqa_span": 0.3, "codes": [["x", 0.2, 0.1]]}}}


def fake_fit(td, name, dev=None, read=True, stamp=True, rel=True):
    """A G6r fit's screen.json and read.json (the stamps fit_device reads), with train.json, g6r_timing.json, a read,
    models.pt and the relation file."""
    dev = dev or KG.device_of(TOY_ENV)
    split = FITS[name]
    d = Path(td) / "fits" / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    k = {"declared_in": DECLARED, "fit": name, "build": BUILD, "arm": ARM, "seed": SEED, "split": split, "device": dev}
    LC.write_json(d / "screen.json", {"arm": ARM, "split": split, "seed": SEED, **({KEY: k} if stamp else {})})
    LC.write_json(d / "train.json", {"seconds": 160.0, "peak_gpu_gb": 3.6, "data_gb": 2.0})
    LC.write_json(d / TIMING_FILE, {"peak_gb": 9.5, "rel_seconds": 2.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if rel:
        LC.write_json(d / REL_FILE, {"stage": STAGE, "declared_in": DECLARED, "fit": name, **fake_rel()})
    if read:
        LC.write_json(d / "read.json", {"seconds": 13.0, "peak_gpu_gb": 4.6, KEY: dict(k)})
    return d


def fake_cmp(td, cname, deltas=None, zspk_r5=0.6, zg6_r5=None, dev=None, stamp=True, screen=None):
    """One of G6r's compares in lean_screen's form (G6r-stamped), against KG's zspk and zg6 (zg6_r5 {dataset: R@5},
    0.6 where absent); deltas {dataset: zg6r minus zspk}."""
    new, bases = COMPARES[cname]
    split = FITS[new]
    deltas, zg6_r5 = deltas or {}, zg6_r5 or {}
    bp = [str(Path(td) / "fits" / b) for b in bases]
    by = {}
    for b, r5 in zip(bp, ({ds: zspk_r5 for ds in LG.EVAL_ORDER}, {ds: zg6_r5.get(ds, 0.6) for ds in LG.EVAL_ORDER})):
        rows = []
        for ds in LG.EVAL_ORDER:
            nw = round(zspk_r5 + deltas.get(ds, 0.0), 4)
            d = round(nw - r5[ds], 4)
            lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
            rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot", "questions": 100,
                         "base": [r5[ds], 0.5, 0.4], "new": [nw, 0.5, 0.4], "rrf": [0.4, 0.3, 0.2], "delta": [d, 0, 0],
                         "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": S.call(d, lo, hi)})
        by[b] = {"rows": rows, "verdict": S.verdict([r["call"] for r in rows]), "decides": b == bp[0]}
    rec = {"new": str(Path(td) / "fits" / new), "bases": bp, "candidate": "p@swa", "carve": "s1eval",
           "trained_on": sorted(LG.FITS[split]), "by_base": by, "verdict": by[bp[0]]["verdict"]}
    if stamp:
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases),
                    "device": dev or KG.device_of(TOY_ENV)}
    p = Path(screen or td) / f"{cname}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def fake_set(td, deltas=None, kg_g6=None, big=None, kg="BASE_READY", g6b2_v=("NOT_ADOPTED", None), dev=None,
             zg6_r5=None):
    """KG's fake base set (zkg.fake_all) with its base verdict, G6b2's verdict (g6b2_v: verdict and next form, G6r when
    None; None: no file), and G6r's two fits and compares; deltas {(split, dataset): zg6r minus zspk}; kg_g6 {(split,
    dataset): zg6 minus zspk} in KG's compares; big {(split, dataset): (D1, D2)} for zspk's null."""
    td = Path(td)
    deltas, kg_g6 = deltas or {}, kg_g6 or {}
    KG.fake_all(td, g6=kg_g6, big=big)
    if kg:
        LC.write_json(td / "kg" / f"{KG.VERDICT}.json", {"stage": KG.STAGE, "declared_in": KG.DECLARED, "verdict": kg,
                                                         "device": KG.device_of(TOY_ENV)})
    if g6b2_v:
        LC.write_json(td / "g6b2" / f"{GB2.VERDICT}.json", {"stage": GB2.STAGE, "declared_in": GB2.DECLARED,
                                                            "verdict": g6b2_v[0],
                                                            "next_form": g6b2_v[1] or GB2.NEXT["NO_GAIN"]})
    for n, sp in FITS.items():
        fake_fit(td, n, dev=dev)
        z6 = (zg6_r5 or {}).get(sp) or {ds: round(0.6 + kg_g6.get((sp, ds), 0.0), 4) for ds in LG.EVAL_ORDER}
        fake_cmp(td, CMP[sp], {ds: x for (s, ds), x in deltas.items() if s == sp}, zg6_r5=z6, dev=dev)


def real_state():
    """What G6r has under the real folders: its fits, its compares and outputs/g6r."""
    return (sorted(p.name for p in (S.OUT / "fits").glob("scr-g6r-*")),
            sorted(p.name for p in (ROOT / SCREEN).glob("scr-g6r-*")), (ROOT / OUT).exists() and
            sorted(p.name for p in (ROOT / OUT).iterdir()))


def hand_code(m, rel, slots, cdir):
    """rel_codes of one combination by python loops."""
    fw, bw = bool(cdir & 1) or cdir == 0, bool(cdir & 2) or cdir == 0
    rs = [s for s in slots if s >= 0]
    ef, eb = torch.zeros(K_R), torch.zeros(K_R)
    for s in rs:
        ef = ef + m.g6r_pf @ rel[s] / len(rs)
        eb = eb + m.g6r_pb @ rel[s] / len(rs)
    return (ef * float(fw) + eb * float(bw)) / (float(fw) + float(bw))


def by_hand(m, H, xb, lk, gd, nq, B, keep_e=None):
    """g6r_stack's score with python loops over the questions and the edges (the selftest's reference)."""
    N = H.shape[0]
    h0 = Fn.gelu(torch.cat([H, xb], 1) @ m.g6_in_w.T + m.g6_in_b)
    s0 = (h0 @ m.g6_ctx_w.T).squeeze(-1)
    qs_ = nq.tolist()
    rows_of = [[i for i in range(N) if qs_[i] == q] for q in range(B)]
    z0 = torch.zeros(N)
    for rows in rows_of:
        if rows:
            x = s0[rows]
            mu = x.mean()
            sd = ((x - mu) ** 2).mean().sqrt()
            z0[rows] = (x - mu) / sd if float(sd) >= 1e-6 else torch.zeros(len(rows))
    z0 = z0.clamp(-ZCLIP, ZCLIP)
    E = lk["u"].numel()
    edges = [e for e in range(E) if keep_e is None or bool(keep_e[e])]
    U, V_, F, CI = lk["u"].tolist(), lk["v"].tolist(), lk["f"].tolist(), gd["ci"].tolist()
    code = {e: hand_code(m, gd["rel"], gd["crel"][CI[e]].tolist(), int(gd["cdir"][CI[e]])) for e in edges if CI[e] >= 0}
    h = h0
    for k, lay in enumerate(G6.G6_LAYER):
        wm, wa, fb, ws, wmean, wagg, wx, b = (getattr(m, n) for n in lay)
        a_k, b_k, d_k, c_k = (getattr(m, n) for n in G6R_LAYER[k])
        c = torch.zeros(B, D)
        for q, rows in enumerate(rows_of):
            if rows:
                c[q] = (torch.softmax(z0[rows], 0).unsqueeze(1) * h[rows]).sum(0)
        mean, cnt = torch.zeros(N, FAMS, D), torch.zeros(N, FAMS)
        num, den = torch.zeros(N, D), torch.zeros(N)
        for e in edges:
            u_, v_, f_ = U[e], V_[e], F[e]
            q_ = qs_[v_]
            lg = float(((wa @ h[u_]) * c[q_]).sum() / math.sqrt(D) + fb[f_])
            if e in code:
                r = code[e]
                msg = a_k @ ((r + d_k) * (b_k @ h[u_]))
                lg += float((r * (c_k @ c[q_])).sum() / math.sqrt(K_R))
            else:
                msg = wm[f_ * D:(f_ + 1) * D] @ h[u_]
            lg = max(-LCLIP, min(LCLIP, lg))
            mean[v_, f_] += torch.sigmoid(torch.tensor(lg)) * msg
            cnt[v_, f_] += 1
            num[v_] += math.exp(lg) * msg
            den[v_] += math.exp(lg)
        mean = mean / cnt.clamp_min(1).unsqueeze(2)
        att = torch.where(den.unsqueeze(1) > 0, num / den.clamp_min(1e-30).unsqueeze(1), torch.zeros_like(num))
        pre = h @ ws.T + mean.reshape(N, FAMS * D) @ wmean.T + att @ wagg.T + xb @ wx.T + b
        h = h + Fn.gelu(Fn.layer_norm(pre, (D,)))
    return (h @ m.g6_out_w.T + h0 @ m.g6_skip_w.T + m.g6_out_b).squeeze(-1)


def selftest():
    import tempfile
    t0 = time.time()
    before = real_state()
    LG.bind_device_ops()
    # 1. the arm on P1c3's link carve under G6r's (no root of its own), zg6's classes below it; earlier arms unchanged;
    #    the names, bases, sizes and seed offset
    assert S.ARMS[ARM] == (ZG6R, G6rLinkCarve) and ZG6R.__mro__[1:3] == (G6.G6Free, G6.ZM.ZRM) and not ZG6R.TWIN
    assert G6rLinkCarve.__mro__[1] is Z3.KTLinkCarveBase
    assert not [k for k in ("KT_ROOT", "CH_ROOT", "LK_ROOT", "REL_DIR") if k in vars(G6rLinkCarve)]
    assert S.ARMS["zg6"] == G6.ARMS["zg6"] and S.ARMS["zspk"] == Z.ARMS["zspk"] and S.ARMS["zg6b2"] == GB2.ARMS["zg6b2"]
    assert S.ARMS["zg6e"] == U2N.ARMS["zg6e"]
    assert N_G6R == 58416 and N_PARAMS == 180218 and sorted(FITS) == ["scr-g6r-zg6r", "scr-g6r-zg6r-hp"]
    assert COMPARES["scr-g6r-zg6r-vs-zspk-hp"] == ("scr-g6r-zg6r-hp", ["scr-kg-base-zspk-hp", "scr-kg-base-zg6-hp"])
    assert KG_ZG6_CMP["L-musique"] == "scr-kg-zg6-vs-zspk" and KG_ZRCKP["L-hotpotqa"] == "scr-kg-null-zrckp-s1-hp"
    assert SEED_OFF not in (G6.SEED_OFF, G6.SEED_OFF + 1, U2N.HEAD_SEED_OFF, Z3.KT_SEED_OFF)
    assert GB2.NEXT["NO_GAIN"] == G6.NEXT["NO_GAIN"] and G6.NEXT["NO_GAIN"].startswith("G6r, relation-aware messages")
    assert len(ENC_NAMES) == len(set(ENC_NAMES)) == 15 and len(G6R_NAMES) == len(set(G6R_NAMES)) == 2 + 4 * LAYERS
    assert sorted(ENC_SHA) == sorted(SPLITS) and Z.rel(ENC_ROOT) == "outputs/g6r/enc"
    # 2. where: a Kaggle T4 on KG's image only; train and read refuse anything else before touching data; another read
    #    cap stops; a missing encoder file stops
    for bad in ({"where": "elsewhere"}, {"gpu": "Tesla P100-PCIE-16GB"},
                {"image": "kaggle-gpu:py3.13.15:torch2.12.0+cu128:numpy2.1.3"}):
        stops(train, "scr-g6r-zg6r", {**TOY_ENV, **bad}, has="G6r trains in a Kaggle kernel")
        stops(read, "scr-g6r-zg6r-hp", {**TOY_ENV, **bad}, has="G6r reads in a Kaggle kernel")
    stops(train, "scr-g6b2-zg6b2", TOY_ENV, has="not one of G6r's fits")
    stops(read, "scr-g6r-zg6", TOY_ENV, has="not one of G6r's fits")
    stops(fit, "scr-g6r-zg6r", 65536, has="reads with 32768 rows a batch")
    tmp = Path(tempfile.mkdtemp(prefix="zg6r_"))
    stops(load_encoder, "L-musique", tmp / "none", has="missing")
    try:
        rng = np.random.default_rng(31)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(look_root=tmp / "look", cache_root=tmp / "cache", links_root=tmp / "lk", ch_root=tmp / "ch")

        class ToyIds(G6.ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        with U2N.extra_carves(("metaqa", "toy")):
            U2N.build_typed("metaqa", "toy", tmp / "look", tmp / "typed", **kw)
            plain = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
            c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
            cls = g6r_carve(Z3.kt_carve(plain, tmp / "kt"), tmp / "typed")
            ce = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            stops(g6r_carve(Z3.kt_carve(plain, tmp / "kt"), tmp / "typed_none"), "metaqa", "toy", "2wiki", "cpu",
                  tmp / "cache", has="missing")
        stops(cls, "metaqa", "toy", "2wiki", "cpu", tmp / "cache", has="not one of U2n's typed carves")
        # 3. the carve: the typed links are U2n's, aligned with zlink's edges in link_batch's order; each typed edge's
        #    combination holds the build's set and bits; typed edges are family 0; the relation vectors are unit
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(ce.rows)[::-1][:7], np.arange(ce.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = ce.batch(qs, blocks)
        N = nq.numel()
        lk, gd = feats[LK_KEY], feats[G6R_KEY]
        ci, crel, cdir = gd["ci"], gd["crel"], gd["cdir"]
        uz, vz, fz_, idz = U2N.u2n_edges(ce, qs)
        assert torch.equal(uz, lk["u"]) and torch.equal(vz, lk["v"]) and torch.equal(fz_, lk["f"])
        ts_b = ce.u2n_typed["t_set"][idz].astype(np.int64)
        td_b = ce.u2n_typed["t_dir"][idz].astype(np.int64)
        for e in range(ci.numel()):
            if ci[e] >= 0:
                assert crel[ci[e]].tolist() == ce.u2n_typed["sets"][ts_b[e]].astype(np.int64).tolist(), e
                assert int(cdir[ci[e]]) == td_b[e] and int(lk["f"][e]) == 0, e
            else:
                assert ts_b[e] == -1, e
        n_typed = int((ci >= 0).sum())
        assert 0 < n_typed < ci.numel() and crel.shape[0] >= 2 and {1, 2} <= set(cdir.tolist()), set(cdir.tolist())
        assert gd["rel"] is ce.g6r_rel and torch.allclose(ce.g6r_rel.norm(dim=1), torch.ones(5))
        assert ce.nbytes() - super(cls, ce).nbytes() == ce.g6r_rel.numel() * 4 == 5 * Q_DIM * 4
        keep = torch.ones(B, len(blocks))
        # 4. from seed 0: zg6r's construction moves the global generator as zg6's; its state is zg6's plus G6r's (P
        #    and C 0, delta 1, A and B drawn from the seed plus SEED_OFF); the family dropout's generator is zg6's
        nets, after = {}, {}
        for k, cl in (("zg6", G6.ZG6), (ARM, ZG6R)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, ce.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after["zg6"])
        m, z6 = nets[ARM], nets["zg6"]
        sm, s6 = m.state_dict(), z6.state_dict()
        assert sorted(sm) == sorted(set(s6) | set(G6R_NAMES)) and all(torch.equal(sm[k], s6[k]) for k in s6)
        g_ = torch.Generator().manual_seed(SEED_OFF)
        for na, nb, nd, nc in G6R_LAYER:
            assert torch.equal(sm[na], (torch.rand(D, K_R, generator=g_) * 2 - 1) / K_R ** 0.5)
            assert torch.equal(sm[nb], (torch.rand(K_R, D, generator=g_) * 2 - 1) / D ** 0.5)
            assert torch.equal(sm[nd], torch.ones(K_R)) and float(sm[nc].abs().sum()) == 0
        assert float(sm["g6r_pf"].abs().sum()) == float(sm["g6r_pb"].abs().sum()) == 0
        assert sum(int(sm[k].numel()) for k in G6.G6_NAMES + G6R_NAMES) == G6.n_params(16) + N_G6R
        assert torch.equal(m.g6_gen.get_state(), z6.g6_gen.get_state()) and ENC["state"] is None
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for n in G6.G6_NAMES:
                getattr(m, n).copy_(torch.randn(getattr(m, n).shape, generator=gen) * 0.3)
            z6.load_state_dict({k: x for k, x in m.state_dict().items() if k not in G6R_NAMES})
        for x in nets.values():
            x.eval()
        # 5. with no typed edge (a passage graph: the carve's typed links None) zg6r's score is zg6's bit for bit, in
        #    eval and in training with the same draws (checkpointed); at the start on the KB batch every logit is
        #    zg6's and the typed messages are A B h, so the scores differ
        saved = ce.u2n_typed, ce.g6r_rel
        ce.u2n_typed = ce.g6r_rel = None
        fp, nqp, bzp, _gp = ce.batch(qs, blocks)
        ce.u2n_typed, ce.g6r_rel = saved
        assert torch.equal(nqp, nq) and fp[G6R_KEY]["rel"] is None and int((fp[G6R_KEY]["ci"] >= 0).sum()) == 0
        with torch.no_grad():
            assert torch.equal(m(fp, keep, nq, B, bz), z6(fp, keep, nq, B, bz))
            assert not torch.allclose(m(feats, keep, nq, B, bz), z6(feats, keep, nq, B, bz))
            H = m.encode(feats, keep, nq, B)
            xb = torch.cat([feats[KT_KEY], m.chain(feats, keep, nq)], 1)
            y0 = m(feats, keep, nq, B, bz)
            want = by_hand(m, H, xb, lk, gd, nq, B)
            assert torch.allclose(y0, want, atol=1e-4, rtol=1e-4), float((y0 - want).abs().max())
        for x in nets.values():
            x.train()
        ys = {}
        for k in ("zg6", ARM):
            nets[k].g6_gen.manual_seed(77)
            torch.manual_seed(5)
            ys[k] = nets[k](fp, keep, nq, B, bz)
        assert torch.equal(ys[ARM], ys["zg6"]) and ys[ARM].requires_grad
        for x in nets.values():
            x.eval()
        # 6. by hand: the codes (a crafted case with every direction code and empty slots) and, every parameter set,
        #    the layers with the typed messages and logits
        with torch.no_grad():
            for n in G6R_NAMES:
                getattr(m, n).copy_(getattr(m, n) + torch.randn(getattr(m, n).shape, generator=gen) * 0.3)
            rel = Fn.normalize(torch.randn(7, Q_DIM, generator=gen), dim=1)
            cr = torch.tensor([[0, 3, -1, -1, -1, -1, -1, -1], [5, -1, -1, -1, -1, -1, -1, -1],
                               [1, 2, 6, -1, -1, -1, -1, -1], [4, 0, -1, -1, -1, -1, -1, -1]])
            cd = torch.tensor([0, 1, 2, 3])
            rt = rel_codes(m, rel, cr, cd, torch.float32)
            for j in range(4):
                assert torch.allclose(rt[j], hand_code(m, rel, cr[j].tolist(), int(cd[j])), atol=1e-5, rtol=1e-4), j
            assert not torch.allclose(rt[1], rt[2])
            y = m(feats, keep, nq, B, bz)
            want = by_hand(m, H, xb, lk, gd, nq, B)
            assert torch.allclose(y, want, atol=1e-4, rtol=1e-4), float((y - want).abs().max())
            assert not torch.allclose(y, y0)
            # 7. a question without edges: its rows are zg6's (no message reaches them); a batch without G6r's, the
            #    edges' or KT's key stops, and typed edges without their relation vectors stop
            q0_ = int(nq[lk["v"][0]])
            ke = nq[lk["v"]] != q0_
            f3 = dict(feats)
            f3[LK_KEY] = {x: t_[ke] for x, t_ in lk.items()}
            f3[G6R_KEY] = {**gd, "ci": gd["ci"][ke]}
            rows0 = nq == q0_
            assert torch.equal(m(f3, keep, nq, B, bz)[rows0], z6(f3, keep, nq, B, bz)[rows0])
            for bad in ({k: x for k, x in feats.items() if k != G6R_KEY},
                        {k: x for k, x in feats.items() if k != LK_KEY},
                        {k: x for k, x in feats.items() if k != KT_KEY},
                        {**feats, G6R_KEY: {**gd, "rel": None}}):
                try:
                    m(bad, keep, nq, B, bz)
                except RuntimeError:
                    continue
                raise AssertionError("zg6r ran without its typed links, edges, KT input or relation vectors")
        # 8. the family dropout: one B x FAMS draw from zg6's generator; a typed edge goes with its family (the score
        #    by hand over zg2.family_keep's edges); the global generator moves as zg6's
        m.train()
        m.drop.p = 0.0
        with torch.no_grad():
            for _ in range(50):
                g0 = m.g6_gen.get_state()
                u01 = torch.rand(B, FAMS, generator=m.g6_gen)
                kd_ = G2.family_keep(lk["f"], nq[lk["v"]], B, u01)
                if 0 < int(kd_.sum()) < kd_.numel() and 0 < int((kd_ & (ci >= 0)).sum()) < n_typed:
                    break
            assert 0 < int(kd_.sum()) < kd_.numel() and 0 < int((kd_ & (ci >= 0)).sum()) < n_typed
            g1 = m.g6_gen.get_state()
            m.g6_gen.set_state(g0)
            y = m(feats, keep, nq, B, bz)
            assert torch.equal(m.g6_gen.get_state(), g1)
            want = by_hand(m, H, xb, lk, gd, nq, B, kd_)
            assert torch.allclose(y, want, atol=1e-4, rtol=1e-4), float((y - want).abs().max())
        m.drop.p = 0.1
        z6.load_state_dict({k: x for k, x in m.state_dict().items() if k not in G6R_NAMES})
        z6.g6_gen.set_state(m.g6_gen.get_state())
        z6.train()
        with torch.no_grad():
            torch.manual_seed(9)
            m(feats, keep, nq, B, bz)
            ra = torch.rand(4)
            torch.manual_seed(9)
            z6(feats, keep, nq, B, bz)
        assert torch.equal(ra, torch.rand(4)) and torch.equal(m.g6_gen.get_state(), z6.g6_gen.get_state())
        # 9. gradients reach every parameter, checkpointed or not (the same gradients)
        with torch.no_grad():
            m.cm_b.copy_(torch.randn(m.cm_b.shape, generator=gen) * 0.1)
            m.cm_d.copy_(torch.randn(m.cm_d.shape, generator=gen) * 0.1)
        gw = torch.randn(N, generator=torch.Generator().manual_seed(11))
        grads = {}
        for on in (True, False):
            CKPT["on"] = on
            m.g6_gen.manual_seed(13)
            torch.manual_seed(13)
            grads[on] = G6.grads(m, feats, keep, nq, B, bz, gw)
        CKPT["on"] = True
        ga = grads[True]
        assert not [k for k, x in ga.items() if not G6.nonzero(x)], [k for k, x in ga.items() if not G6.nonzero(x)]
        assert all(torch.allclose(ga[k], grads[False][k], atol=1e-6, rtol=1e-5) for k in ga)
        m.eval()
        # 10. zspk's encoder: loaded from its file (sha256, fit, seed, size checked), copied into each zg6r built while
        #     it is active and into nothing else; the draws and every other tensor as without it; a state that does not
        #     fit stops; after a fit, no copy or another basis stops
        torch.manual_seed(1)
        donor = G6.ZG6(blocks, ce.widths, 16, dropout=0.1, seed=1)
        with torch.no_grad():
            dst = {k: x.detach().clone() + 0.01 * torch.randn(x.shape, generator=gen)
                   for k, x in donor.state_dict().items()}
        ne = sum(int(dst[k].numel()) for k in ENC_NAMES)
        (tmp / "enc").mkdir()
        blob = {"fit": KG.fit_name("zspk", 0, "L-musique"), "seed": 0, "hidden": 16, "basis": "2wiki",
                "basis_sha256": "b" * 64, "states": {"p@swa": dst, "p@ep0": dst}, "candidates": ["p@ep0", "p@swa"]}
        torch.save(blob, tmp / "enc" / "L-musique.pt")
        sha_e = LC.sha_file(tmp / "enc" / "L-musique.pt")
        est, ei = load_encoder("L-musique", tmp / "enc", sha=sha_e, elements=ne)
        assert sorted(est) == sorted(ENC_NAMES) and all(torch.equal(est[k], dst[k]) for k in ENC_NAMES)
        assert ei["fit"] == "scr-kg-base-zspk" and ei["elements"] == ne and ei["basis"] == "2wiki" and ei["hidden"] == 16
        stops(load_encoder, "L-musique", tmp / "enc", has="not KG's zspk fit of L-musique")
        stops(load_encoder, "L-musique", tmp / "enc", sha=sha_e, elements=ne + 1, has=f"not {ne + 1}")
        torch.save({**blob, "fit": "scr-kg-base-zg6"}, tmp / "enc" / "L-hotpotqa.pt")
        stops(load_encoder, "L-hotpotqa", tmp / "enc", sha=LC.sha_file(tmp / "enc" / "L-hotpotqa.pt"), elements=ne,
              has="not scr-kg-base-zspk-hp seed 0")
        torch.manual_seed(0)
        plain_m = ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
        r_plain = torch.rand(4)
        with using_encoder(est, ei):
            torch.manual_seed(0)
            with_m = ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
            r_with = torch.rand(4)
            G6.ZG6(blocks, ce.widths, 16, dropout=0.1, seed=0)
            assert ENC["copies"] == 1
        assert ENC == {"state": None, "info": None, "copies": 0}
        with using_encoder({**est, "l2.weight": est["l2.weight"][:, :3]}, ei):
            stops(ZG6R, blocks, ce.widths, 16, has="does not fit")
        assert ENC == {"state": None, "info": None, "copies": 0}
        sp_, sw_ = plain_m.state_dict(), with_m.state_dict()
        assert torch.equal(r_plain, r_with) and torch.equal(plain_m.g6_gen.get_state(), with_m.g6_gen.get_state())
        assert all(torch.equal(sw_[k], est[k]) for k in ENC_NAMES)
        assert all(torch.equal(sw_[k], sp_[k]) for k in sp_ if k not in ENC_NAMES)
        fdx = tmp / "efits" / "scr-g6r-zg6r"
        fdx.mkdir(parents=True)
        torch.save({"basis": "2wiki", "basis_sha256": "b" * 64, "hidden": 16, "states": {}}, fdx / "models.pt")
        encoder_used("scr-g6r-zg6r", {**ei, "copies": 1}, tmp / "efits")
        stops(encoder_used, "scr-g6r-zg6r", {**ei, "copies": 0}, tmp / "efits", has="no zg6r was built")
        stops(encoder_used, "scr-g6r-zg6r", {**ei, "copies": 1, "basis_sha256": "c" * 64}, tmp / "efits",
              has="was fitted on the basis")
        # 11. toy fits through lean_gpu's loop with zspk's encoder active: a repeat is bit for bit; the encoder is
        #     copied; the encoder, the stack, the readout and every G6r parameter move
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        fits_, copies = [], []
        for rep in (0, 1):
            with S.patched(ARM), using_encoder(est, ei):
                fits_.append(LG.fit_variant([ce], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-zg6r {rep}"))
                copies.append(ENC["copies"])
        f1, f2 = fits_
        assert all(not LG.same_state(a, b) for a, b in zip(f1["states"] + [f1["swa"]], f2["states"] + [f2["swa"]]))
        assert copies[0] == copies[1] >= 1, copies
        with using_encoder(est, ei):
            torch.manual_seed(0)
            s0 = ZG6R(blocks, f1["widths"], 16, dropout=0.1, seed=0).state_dict()
        assert all(torch.equal(s0[k], est[k]) for k in ENC_NAMES)
        assert sorted(s0) == sorted(f1["swa"])
        must = ["l1.weight", "l2.weight", "g6_in_w", "g6_out_w", "g6_skip_w", *G6R_NAMES] + \
            [lay[i] for lay in G6.G6_LAYER for i in (0, 1, 3, 4, 5, 6)]
        still = [k for k in must if torch.equal(s0[k], f1["swa"][k])]
        assert not still, still
        # 12. the relation map beside: from a p@swa state, P's norms, singular values and span shares, the codes, the
        #     layers and the encoder's drift, by hand; the file; a vocabulary not its record's stops
        U2N.toy_rel_record(tmp / "rel")
        show = (("metaqa", None),)
        st_ = {k: x.detach().clone() for k, x in m.state_dict().items()}
        fd = tmp / "rfits" / "scr-g6r-zg6r"
        fd.mkdir(parents=True)
        torch.save({"states": {"p@swa": st_}, "candidates": ["p@swa"]}, fd / "models.pt")
        rb = rel_beside("scr-g6r-zg6r", tmp / "rfits", tmp / "rel", show, enc=est)
        assert json.loads((fd / REL_FILE).read_text(encoding="utf-8")) == rb and rb["fit"] == "scr-g6r-zg6r"
        E_ = np.load(tmp / "rel" / "metaqa_rel_embeddings.npy").astype(np.float64)
        U_ = E_ / np.linalg.norm(E_, axis=1, keepdims=True)
        C_ = (U_ - U_.mean(0)).astype(np.float32).astype(np.float64)
        C_ /= np.linalg.norm(C_, axis=1, keepdims=True)
        pf_, pb_ = st_["g6r_pf"].double().numpy(), st_["g6r_pb"].double().numpy()
        assert rb["p_f_norm"] == round(float(np.sqrt((pf_ ** 2).sum())), 6)
        assert rb["by_dataset"]["metaqa"]["span_rank"] == 4 and len(rb["by_dataset"]["metaqa"]["codes"]) == 5
        _u_, _s_, vt_ = np.linalg.svd(C_, full_matrices=False)
        Qm = vt_[:4]
        assert abs(rb["p_b_in_metaqa_span"] - round(float(np.linalg.norm(pb_ @ Qm.T) / np.linalg.norm(pb_)), 4)) < 1e-9
        got = {n_: (a_, b_) for n_, a_, b_ in rb["by_dataset"]["metaqa"]["codes"]}
        for r_ in range(5):
            assert got[f"rel {r_}"] == (round(float(np.linalg.norm(pf_ @ C_[r_])), 4),
                                        round(float(np.linalg.norm(pb_ @ C_[r_])), 4)), r_
        assert [x["shown"] for x in rb["by_dataset"].values()] == [5] and rb["by_dataset"]["metaqa"]["codes"][0][1] == \
            max(a_ for a_, _b in got.values())
        dl = st_["g6r_1_d"].double().numpy()
        assert rb["layers"][1]["delta"] == {"mean": round(float(dl.mean()), 4), "min": round(float(dl.min()), 4),
                                            "max": round(float(dl.max()), 4)}
        assert rb["layers"][2]["c_norm"] == round(float(st_["g6r_2_c"].double().norm()), 4)
        dr = math.sqrt(sum(float(((st_[k].double() - est[k].double()) ** 2).sum()) for k in ENC_NAMES) /
                       sum(float((est[k].double() ** 2).sum()) for k in ENC_NAMES))
        assert rb["encoder_drift"] == round(dr, 6) and rb["encoder_drift"] > 0
        assert rel_beside("scr-g6r-zg6r", tmp / "rfits", tmp / "rel", show, enc_root=tmp / "none")["encoder_drift"] \
            is None
        vj = json.loads((tmp / "rel" / "metaqa_rel_vocab.json").read_text(encoding="utf-8"))
        LC.write_json(tmp / "rel" / "metaqa_rel_vocab.json", {**vj, "vocab": vj["vocab"][::-1]})
        stops(rel_beside, "scr-g6r-zg6r", tmp / "rfits", tmp / "rel", show, est, has="its vocabulary is not the one")
    finally:
        CKPT["on"] = True
        shutil.rmtree(tmp, ignore_errors=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fits = td / "fits"
        # 13. stamps and devices: a fit not G6r's, a read on another device; take copies records, reads and the
        #     relation file, never models.pt, and leaves KG's fits to KG
        fake_fit(td, "scr-g6r-zg6r")
        assert fit_device("scr-g6r-zg6r", fits) == KG.device_of(TOY_ENV)
        fake_fit(td, "scr-g6r-zg6r-hp", stamp=False)
        stops(fit_device, "scr-g6r-zg6r-hp", fits, has="not G6r's zg6r fit")
        d = fake_fit(td, "scr-g6r-zg6r-hp")
        LC.write_json(d / "read.json", {KEY: {"declared_in": DECLARED, "fit": "scr-g6r-zg6r-hp",
                                              "device": {**KG.device_of(TOY_ENV), "gpu": "Tesla P100"}}})
        stops(fit_device, "scr-g6r-zg6r-hp", fits, has="on the device it was fitted on")
        src = td / "out"
        fake_fit(src, "scr-g6r-zg6r")
        KG.fake_fit(src, "scr-kg-base-zg6")
        took = take(src, td / "repo_fits")
        assert sorted(Path(t).name for t in took) == sorted([REL_FILE, TIMING_FILE, "metaqa__s1eval.npz", "read.json",
                                                             "screen.json", "train.json"]), took
        assert not (td / "repo_fits" / "scr-g6r-zg6r" / "models.pt").exists()
        assert not (td / "repo_fits" / "scr-kg-base-zg6").exists() and take(src, td / "repo_fits") == []
        (td / "repo_fits" / "scr-g6r-zg6r" / "train.json").write_text("{}", encoding="utf-8")
        stops(take, src, td / "repo_fits", has="other bytes")
        fake_fit(src, "scr-g6r-zg6r-hp", read=False)
        stops(take, src, td / "d2", has="not G6r's read")
        shutil.rmtree(src / "fits" / "scr-g6r-zg6r-hp")
        (src / "fits" / "scr-g6r-zg6").mkdir()
        stops(take, src, td / "d3", has="not one of G6r's fits")
        # 14. one device across G6r's fit and KG's two bases
        two = ["scr-kg-base-zspk", "scr-kg-base-zg6"]
        KG.fake_fit(td, "scr-kg-base-zspk")
        KG.fake_fit(td, "scr-kg-base-zg6")
        assert one_device("scr-g6r-zg6r", two, fits) == KG.device_of(TOY_ENV)
        KG.fake_fit(td, "scr-kg-base-zg6", dev={**KG.device_of(TOY_ENV), "cudnn": 1})
        stops(one_device, "scr-g6r-zg6r", two, fits, has="different devices")
    # 15. the verdict: a GAIN and no LOSS adopts for the full run; a GAIN and a LOSS gives the gated form; no GAIN (a GAIN
    #     under a floor the null raises, a LOSS alone) gives the two changes apart; INCOMPLETE without KG's BASE_READY,
    #     without G6b2's verdict or with another, without a compare, with the fits on another device than KG's, or with
    #     a zg6 column that is not KG's read; beside: the zg6 re-call, R@5, the MLP share, the relation map and timing
    gm_, lw = ("L-musique", "metaqa"), ("L-hotpotqa", "webqsp")
    cases = (({gm_: 0.02}, None, "ADOPT_FOR_FULL_RUN", NEXT["ADOPT_FOR_FULL_RUN"]),
             ({gm_: 0.02, lw: -0.03}, None, "NOT_ADOPTED", NEXT["GAIN_AND_LOSS"]),
             ({}, None, "NOT_ADOPTED", NEXT["NO_GAIN"]),
             ({("L-musique", "musique"): 0.02}, {("L-musique", "musique"): (0.02, -0.01)}, "NOT_ADOPTED",
              NEXT["NO_GAIN"]),
             ({lw: -0.03}, None, "NOT_ADOPTED", NEXT["NO_GAIN"]))
    for i, (deltas, big, want, nxt) in enumerate(cases):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fake_set(td, deltas, kg_g6={("L-hotpotqa", "2wiki"): 0.01}, big=big)
            v = verdict(td, td / "g6r", td / "fits", td / "kg", td / "g6b2")
            assert v["verdict"] == want and v["next_form"] == nxt, (i, v["verdict"], v["problems"])
            b = v["beside"]
            assert b["r5"]["L-musique metaqa"]["zrckp"] == 0.59 and b["r5"]["L-hotpotqa 2wiki"]["zg6"] == 0.61
            assert b["r5"]["L-musique squad"]["zg6r"] == 0.6 and b["mlp_share"]["L-musique squad"] == round(0.19 / 0.2, 4)
            assert b["rel"]["L-musique"]["encoder_drift"] == 0.12 and b["timing"]["L-hotpotqa"]["zg6r"]["train_seconds"] == 160
            assert b["timing"]["L-hotpotqa"]["zg6"]["train_seconds"] == b["timing"]["L-musique"]["zspk"]["train_seconds"] == 100
            rz = json.loads((td / "g6r" / f"{RECALL[1]}.json").read_text(encoding="utf-8"))
            assert rz["stage"] == STAGE and rz["against"] == "zg6" and not rz["deciding"] and len(rz["rows"]) == 12
            assert rz["null_kind"] == NULL_KIND[1]
            rd = json.loads((td / "g6r" / f"{RECALL[0]}.json").read_text(encoding="utf-8"))
            assert rd["deciding"] and rd["against"] == "zspk" and len(rd["rows"]) == 12
            if i == 0:
                assert v["gains"] == ["L-musique metaqa"] and b["vs_zg6"]["gains"] == ["L-musique metaqa"]
            if i == 3:
                assert rd["changed"] == ["L-musique musique GAIN -> WITHIN"], rd["changed"]
            mdt = (td / "g6r" / f"{VERDICT}.md").read_text(encoding="utf-8")
            assert mdt.startswith("# G6r's screen verdict") and "directed by 0.3000 / 0.2000" in mdt
    for case in ("no_kg", "kg_incomplete", "no_g6b2", "g6b2_adopted", "g6b2_other_next", "no_compare", "other_device",
                 "zg6_column"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            kw = {"no_kg": {"kg": None}, "kg_incomplete": {"kg": "INCOMPLETE"}, "no_g6b2": {"g6b2_v": None},
                  "g6b2_adopted": {"g6b2_v": ("ADOPT_FOR_FULL_RUN", GB2.NEXT["ADOPT_FOR_FULL_RUN"])},
                  "g6b2_other_next": {"g6b2_v": ("NOT_ADOPTED", GB2.NEXT["ADOPT_FOR_FULL_RUN"])},
                  "other_device": {"dev": {**KG.device_of(TOY_ENV), "cudnn": 1}},
                  "zg6_column": {"zg6_r5": {"L-musique": {ds: 0.6 for ds in LG.EVAL_ORDER} | {"squad": 0.61}}}}.get(
                case, {})
            fake_set(td, {gm_: 0.02}, **kw)
            if case == "no_compare":
                (td / f"{CMP['L-hotpotqa']}.json").unlink()
            v = verdict(td, td / "g6r", td / "fits", td / "kg", td / "g6b2")
            assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"] and v["beside"] == {}, \
                (case, v["verdict"], v["problems"])
            has = {"no_kg": "KG's base verdict", "kg_incomplete": "not BASE_READY", "no_g6b2": "G6b2's verdict",
                   "g6b2_adopted": "not NOT_ADOPTED with G6r next", "g6b2_other_next": "not NOT_ADOPTED with G6r next",
                   "no_compare": "missing", "other_device": "KG's base set on",
                   "zg6_column": "compared with seed-0 R@5"}[case]
            assert any(has in p for p in v["problems"]), (case, v["problems"])
    # 16. nothing written under the real folders
    assert real_state() == before, (real_state(), before)
    log(f"zg6r selftest: the arm on P1c3's link carve under G6r's, zg6's classes below it, earlier arms unchanged; "
        f"where-checks and the read cap; the typed links U2n's, aligned with zlink's edges; from seed 0 zg6's draws and "
        f"state plus G6r's ({N_PARAMS} parameters at hidden 128); with no typed edge zg6's score bit for bit (eval and "
        f"training); the codes and the layers by hand; a question without edges; the family dropout's draws; gradients "
        f"reach every parameter, checkpointed or not; zspk's encoder loaded, checked and copied only while active; toy "
        f"fits repeat bit for bit and move every G6r parameter; the relation map by hand; stamps, take and one device; "
        f"the verdict's outcomes, next forms, refusals and beside rows; nothing written under G6r's folders "
        f"({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("fit", "take", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=0)
    ap.add_argument("--from", dest="src")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        if not a.name:
            ap.error("fit takes --name")
        return fit(a.name, a.rows_cap)
    with Z.at_root():
        if a.cmd == "take":
            if not a.src:
                ap.error("take takes --from")
            take(a.src)
            return 0
        if a.cmd == "compare-all":
            return compare_all()
        if a.cmd == "verdict":
            return 1 if verdict()["verdict"] == "INCOMPLETE" else 0
    ap.error("fit, take, compare-all, verdict or --selftest")


if __name__ == "__main__":
    sys.exit(main())
