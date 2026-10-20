"""R2 (docs/R2_BRIDGE_EM.md): learned bridge transforms trained with EM, on both models. P5b and P5c fitted offsets as
fixed columns; R2 learns the transform inside each model's own fit, end to end, and needs no build: the bridges come from
the carve's own first-stage RRF column and its frozen vectors. Per question q, fixed before any row is scored:

    bridges   the K = 5 rows with the highest first-stage RRF (column c_rrf), in descending order, ties by pool position
              (lean_gpu.top_hit's order), NaN last; K_q = min(5, n_q); picked by an input, never by a model score
    prior     pi_k proportional to 1 / (60 + k), k = 1..K_q, renormalised per row over the bridges unmasked for it

and learned (r = 32, R2's own generator at seed + 5502):

    h_q = U_b^T e_q (e_q the 1536-d question vector), h_k = V_b^T p_k, h_v = V_b^T p_v (p the decoded 257-d store vector
    with kappa; V_b shared by bridges and rows)
    t_k = h_q + W_1 h_k + W_2 (h_q * h_k) + o
    s_vk = <t_k, h_v> / sqrt(r), masked where v is bridge k itself or bridge k is absent
    x_v = [a_v, m_v, f_v]: a_v = log sum_k pi~_k(v) exp(s_vk) and m_v = max_k s_vk over the unmasked k, f_v = 1 when one
          is unmasked (a = m = 0 when none)
    head: Linear(3, 16), GELU, Linear(16, 1), its last layer zero, added to the arm's final score

59,537 parameters. In training only, the loss adds LAM = 1 times the batch mean, over the questions with a term, of

    L_br(q) = mean over q's golds v with an unmasked bridge of -log sum_k pi~_k(v) p_k(v),
    p_k(v) = softmax over q's rows v' unmasked for k of s_v'k

whose gradient in s is EM's: -r_k(v) (1[v' = v] - p_k(v')), r_k(v) the posterior that bridge k explains gold v. The
softmax over rows is in the loss only; a row's score reads its own vector, the question's vector and the question's five
bridges' frozen vectors (the same five for every row), never an edge or another row's score.

Two arms, each its base unchanged plus R2's head (the base's carve, settings, seed 0 and batches):
    zrckpb  zrckp's model (outputs/mp_unified/zp1c3p.py) plus R2's head, on zrckp's carve with R2's bridges beside
    zg6m    zg6's model (outputs/mp_unified/zg6.py) plus R2's head on its final score, on zg6's carve with the bridges
One card fit per arm on L-musique and L-hotpotqa (P1c2's roots, P1c3's KT input), read on the six s1eval carves and
compared with its base's card fit of the split (zrckp: scr-p1c3p-zrckp, -hp; zg6: scr-g6-zg6, -hp), which decides; step
1's fit beside. The pair and its re-calls are relz's, stamped as R2's: under N2's fresh null (zrckp's for zrckpb; zspk's,
the GNN family's measured null that G6 uses, moved onto zg6's card fits, for zg6m), which decides, and under the 7 October
null beside. The verdict is this file's.

    python outputs/mp_unified/zr2br.py train --split L-musique --name scr-r2-zrckpb --arm zrckpb --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zr2br.py read --name scr-r2-zrckpb --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zr2br.py compare --new outputs/screen/fits/scr-r2-zrckpb \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-r2-zrckpb
    python outputs/mp_unified/zr2br.py pair --arm zrckpb --screens S1.json,S2.json --out outputs/screen/scr-r2-zrckpb-pair
    python outputs/mp_unified/zr2br.py recall --arm zrckpb --null-kind N2 --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zr2br.py verdict
    python outputs/mp_unified/zr2br.py --selftest
"""
import os
import sys

_NP_BEFORE = "numpy" in sys.modules        # the selftest checks the threads were set before numpy came in
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")         # rx sets these per job; elsewhere two BLAS threads (zn2 would set one)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zg6 as G6  # noqa: E402  (zg6; through it zg2, zp1c3, zp1c2, zprop, zlink, rmatch, relz and screen_recall)
import zp1c3p as Z3P  # noqa: E402  (zrckp)
import zn2 as N2  # noqa: E402  (N2's fresh null of zrckp)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z3, Z2, G2 = G6.Z3, G6.Z2, G6.G2
ZP, ZL, ZM, RM = G6.ZP, G6.ZL, G6.ZM, G6.RM
Z, R, S2, S, LG, LC, SR = G6.Z, G6.R, G6.S2, G6.S, G6.LG, G6.LC, G6.SR
S3, SP = G6.S3, G6.SP
log = S.log

DECLARED = "docs/R2_BRIDGE_EM.md"
STAGE = "R2"
OUT = ROOT / "outputs" / "r2"
SCREEN = ROOT / "outputs" / "screen"
R2_KEY = "_r2"
K = 5                      # bridges per question: the first stage's top five by RRF
R_DIM = 32                 # the bridge space's width r
LAM = 1.0                  # the EM term's weight
PRIOR_C = 60               # pi_k proportional to 1 / (PRIOR_C + k), k = 1..K_q
SEED_OFF = 5502            # R2's own draws (R1 5501, G6 3601 and 3602, G2 3501 and 3502, KT 3301, zsp 2301, rmatch 1201)
HEAD = (3, 16, 1)          # [a, m, f] -> 16 (GELU) -> 1, the last layer zero
W1_SCALE = 0.1             # W_1 ~ N(0, 1) * W1_SCALE / sqrt(r); U_b and V_b ~ N(0, 1); W_2 = o = 0
DECLARED_CONST = {"K": 5, "r": 32, "LAM": 1.0, "prior_c": 60, "seed_offset": 5502, "head": [3, 16, 1], "w1_scale": 0.1}
CHUNK = 1 << 16            # rows per slice of the bridge logits
BRIDGE_ROWS = 1 << 21      # rows per slice of the bridge sort at load
TINY = 1e-30
Q_DIM, P_DIM = RM.Q_DIM, LG.L3.STORE_DIM          # 1536 (e_q), 257 (p with kappa)
R2_PARAMS = ("r2_U", "r2_V", "r2_W1", "r2_W2", "r2_o", "r2_w1", "r2_b1", "r2_w2", "r2_b2")
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpb": "zrckp", "zg6m": "zg6"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-r2-{a}", "L-hotpotqa": f"scr-r2-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpb": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zg6m": {sp: ("screen", "fits", G6.FIT["zg6"][sp]) for sp in SPLITS}}
BASE_SCREENS = {"zrckpb": {sp: f"outputs/l1/base-zrckp-{sp}.json" for sp in SPLITS},      # the card fits against
                "zg6m": {sp: f"outputs/r2/base-zg6-{sp}.json" for sp in SPLITS}}         # step 1's (7 Oct re-base)
G6_SCREENS = {sp: f"outputs/screen/{G6.FIT['zg6'][sp]}.json" for sp in SPLITS}   # G6's: zg6 against zspk's card fits
N2_BASE = {"zrckpb": "zrckp", "zg6m": "zspk"}
N2_NULL = {a: tuple(f"outputs/screen/scr-n2-{N2_BASE[a]}-{s}.json" for s in ("s1", "s2", "s1-hp", "s2-hp"))
           for a in ARM_NAMES}
OCT7_NULL = G6.OCT7_NULL
N2_DESC = {"zrckpb": "zrckp's seeds 1 and 2 against its seed-0 card fit, N2's fresh zrckp null",
           "zg6m": "zspk's seeds 1 and 2 against its seed-0 card fit, N2's fresh zspk null, each read's base moved to "
                   "zg6's card fit through G6's comparisons (the GNN family's measured null, the stand-in G6 uses)"}
OCT7_DESC = "the 7 October null: step 1's seeds 1 and 2 against seed 0 (beside, deciding nothing)"
NULLS = ("N2", "oct7")
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpb": "zrckp plus R2's bridge head", "zg6m": "zg6 plus R2's bridge head on its final score"}
PAIR = {a: SCREEN / f"scr-r2-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-r2-{a}-pair-recall" for a in ARM_NAMES}
RECALL7 = {a: SCREEN / f"scr-r2-{a}-pair-recall-oct7" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
FIXED_TRAIN, FIXED_READ = G6.FIXED_TRAIN, G6.FIXED_READ
TERM = ("lean_gpu's listwise loss + LAM * the batch mean, over its questions with a term, of L_br(q) = the mean over "
        "q's golds with an unmasked bridge of -log sum_k pi~_k(v) p_k(v) (training only)")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under N2's null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when that re-call is missing, not R2's re-call of the arm "
        "under N2's null, not twelve reads, or N2's floors record of its null is missing or refused "
        "(docs/R2_BRIDGE_EM.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools, then its full run on all six splits beside its "
                              "base's, its own declared file (no model change is adopted on today's pools); when only "
                              "one arm passes, the other arm goes to its own next form and is not closed",
        "GAIN_AND_LOSS": "R2g, a learned per-question back-off on R2's head: the head times 2 sigmoid(a . c_q + b), c_q "
                         "the question's own embedding projection and its first-stage top-5 RRF margins, the gate "
                         "starting at 1; one declared file",
        "NO_GAIN": "R2e, the edge form: bridge scores only for bridges linked to the row by zlink's edges (B1d: the edge "
                   "is the bridge), as fixed per-row link flags for the MLP and inside zg6's messages for the GNN; one "
                   "declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def n_params():
    """R2's parameters: U_b, V_b, W_1, W_2, o and the head."""
    h_in, h_mid, h_out = HEAD
    return Q_DIM * R_DIM + P_DIM * R_DIM + 2 * R_DIM * R_DIM + R_DIM + h_mid * h_in + h_mid + h_out * h_mid + h_out


def constants():
    return {"K": K, "r": R_DIM, "LAM": LAM, "prior_c": PRIOR_C, "seed_offset": SEED_OFF, "head": list(HEAD),
            "w1_scale": W1_SCALE}


def check_constants(rec=None, what=None):
    """Stop when this file's constants are not the declared ones, or (rec given) a fit's recorded ones are not."""
    if constants() != DECLARED_CONST:
        raise SystemExit(f"zr2br: R2's constants are {constants()}, not the declared {DECLARED_CONST} ({DECLARED}, "
                         f"section 2)")
    if what is not None and rec != DECLARED_CONST:
        raise SystemExit(f"zr2br: {what} was fitted with R2 constants {rec}, not the declared {DECLARED_CONST}")


SETTINGS = {"bridges": "per question the K rows with the highest first-stage RRF (column c_rrf as stored), descending, "
                       "ties by pool position, NaN last; K_q = min(K, n_q); computed once per carve at load",
            "prior": "pi_k proportional to 1/(prior_c + k), k = 1..K_q, renormalised per row over the bridges unmasked "
                     "for it",
            "space": "h_q = U_b^T e_q, h_v = V_b^T p_v (V_b shared by bridges and rows), t_k = h_q + W_1 h_k + "
                     "W_2 (h_q * h_k) + o",
            "logit": "s_vk = <t_k, h_v> / sqrt(r), masked where v is bridge k or bridge k is absent",
            "row_inputs": ["a = log sum_k pi~_k exp(s_vk)", "m = max_k s_vk", "f = 1 when a bridge is unmasked"],
            "head": "Linear(3, 16), GELU, Linear(16, 1), the last layer zero, added to the arm's final score",
            "init": "U_b, V_b ~ N(0, 1); W_1 ~ N(0, 1) * w1_scale / sqrt(r); W_2 = o = 0; the head's first layer "
                    "uniform(+-1/sqrt(3)); R2's own generator at seed + seed_offset",
            "loss": TERM, "parameters": n_params(), "score_input": False, "edges_read": False}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the bridges ──────────────────────────────────────────────────────────────


def bridges_np(rrf, n, k=K, block=BRIDGE_ROWS):
    """(Q, k) int64: each question's k rows (pool offsets) with the highest rrf, descending, ties by pool position,
    NaN last; -1 past min(k, n_q). rrf per row in pool order, n rows per question."""
    n = np.asarray(n, np.int64)
    rrf = np.asarray(rrf)
    off = np.r_[0, np.cumsum(n)]
    if rrf.ndim != 1 or rrf.size != int(off[-1]):
        raise SystemExit(f"zr2br: {rrf.size} rrf values for {n.size} questions of {int(off[-1])} rows")
    out = np.full((n.size, k), -1, np.int64)
    q0 = 0
    while q0 < n.size:
        q1 = q0 + 1
        while q1 < n.size and off[q1 + 1] - off[q0] <= block:
            q1 += 1
        a, b = int(off[q0]), int(off[q1])
        nn_ = n[q0:q1]
        qid = np.repeat(np.arange(q1 - q0), nn_)
        pos = np.arange(b - a) - (off[q0:q1] - a)[qid]
        x = rrf[a:b].astype(np.float64)
        nan = np.isnan(x)
        key = np.where(nan, 0.0, -x) + 0.0
        o = np.lexsort((pos, key, nan, qid))
        first = off[q0:q1] - a
        for j in range(k):
            has = nn_ > j
            out[q0 + np.flatnonzero(has), j] = pos[o[first[has] + j]]
        q0 = q1
    return out


def rrf_of(c):
    """The carve's first-stage RRF column per row, float32 on the CPU (lean_gpu's c_rrf; the toy's rank column)."""
    col = c.c_rrf if hasattr(c, "c_rrf") else S3.I_RRF
    return c.X[:, col].to("cpu").to(torch.float32).numpy()


def vectors_of(c, qs, idx):
    """e_q (B, 1536) and p (N, 257) of a batch, float32, as lean_gpu.CacheCarve.batch gives SEMB (the toy: its P)."""
    dev = c.device
    eq = c.q_emb[torch.from_numpy(np.asarray(qs, np.int64)).to(dev)].to(torch.float32)
    it = torch.from_numpy(idx).to(dev)
    p = c.decode(c.row[it]) if hasattr(c, "decode") else c.P[it]
    return eq, p.to(torch.float32)


def r2_batch(c, feats, qs):
    """R2's inputs of questions qs: the bridges' batch rows (B, K; -1 absent), e_q and p (SEMB's when the batch has
    it)."""
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    seg = np.cumsum(cnt) - cnt
    br = c.r2_br[qs]
    bri = np.where(br >= 0, br + seg[:, None], -1)
    if "SEMB" in feats:
        eq, p = feats["SEMB"]
    else:
        eq, p = vectors_of(c, qs, np.repeat(c.off_np[qs] - seg, cnt) + np.arange(int(cnt.sum())))
    return {"bri": torch.from_numpy(bri).to(c.device), "eq": eq.to(torch.float32), "p": p.to(torch.float32)}


def r2_carve(base):
    """base's carve class with R2's bridges beside it: batch() adds them under R2_KEY; every block, row, side key and
    gold is base's, unchanged."""

    class R2Carve(base):

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.r2_br = bridges_np(rrf_of(self), self.n_np)

        def nbytes(self):
            return super().nbytes() + int(self.r2_br.nbytes)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[R2_KEY] = r2_batch(self, feats, qs)
            return feats, nq, base_z, gold

    R2Carve.__name__ = R2Carve.__qualname__ = f"R2{base.__name__}"
    return R2Carve


R2KTChainCarveZRC = r2_carve(Z3.KTChainCarveZRC)
R2KTLinkCarveBase = r2_carve(Z3.KTLinkCarveBase)


# ── the head ─────────────────────────────────────────────────────────────────


def r2_init(m, seed):
    g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
    h_in, h_mid, h_out = HEAD
    m.r2_U = nn.Parameter(torch.randn(Q_DIM, R_DIM, generator=g))
    m.r2_V = nn.Parameter(torch.randn(P_DIM, R_DIM, generator=g))
    m.r2_W1 = nn.Parameter(torch.randn(R_DIM, R_DIM, generator=g) * (W1_SCALE / R_DIM ** 0.5))
    m.r2_W2 = nn.Parameter(torch.zeros(R_DIM, R_DIM))
    m.r2_o = nn.Parameter(torch.zeros(R_DIM))
    m.r2_w1 = nn.Parameter((torch.rand(h_mid, h_in, generator=g) * 2 - 1) / h_in ** 0.5)
    m.r2_b1 = nn.Parameter(torch.zeros(h_mid))
    m.r2_w2 = nn.Parameter(torch.zeros(h_out, h_mid))
    m.r2_b2 = nn.Parameter(torch.zeros(h_out))


def r2_logits(m, eq, p, bri, nq):
    """S (N, K) the bridge logits s_vk and M (N, K) the unmasked slots (bridge present and not the row itself)."""
    hq = eq @ m.r2_U
    hv = p @ m.r2_V
    hk = hv[bri.clamp_min(0)]
    hq1 = hq.unsqueeze(1)
    t = hq1 + hk @ m.r2_W1.T + (hq1 * hk) @ m.r2_W2.T + m.r2_o
    N = hv.shape[0]
    parts = [(t[nq[i:i + CHUNK]] * hv[i:i + CHUNK].unsqueeze(1)).sum(-1) for i in range(0, N, CHUNK)]
    S_ = (torch.cat(parts) if parts else hv.new_zeros(0, bri.shape[1])) / math.sqrt(R_DIM)
    rows = torch.arange(N, device=nq.device)
    M = (bri >= 0)[nq] & (bri[nq] != rows.unsqueeze(1))
    return S_, M


def prior_logp(M, dtype):
    """(N, K) log pi~_k(v): log pi_k renormalised over the slots unmasked for row v (0 where masked)."""
    k = torch.arange(1, M.shape[1] + 1, dtype=torch.float64, device=M.device)
    pi = 1.0 / (PRIOR_C + k)
    den = (M.to(torch.float64) * pi).sum(1, keepdim=True)
    lp = torch.log(pi).unsqueeze(0) - torch.log(den.clamp_min(TINY))
    return torch.where(M, lp, torch.zeros_like(lp)).to(dtype)


def row_feats(S_, M, lpt):
    """x (N, 3) = [a, m, f] per row: the prior-weighted log-sum-exp and the max of its unmasked bridge logits, and
    whether it has one (a = m = 0 without)."""
    f = M.any(1, keepdim=True)
    ninf, zero = torch.full_like(S_, -math.inf), torch.zeros_like(S_)
    za = torch.where(f, torch.where(M, lpt + S_, ninf), zero)
    zm = torch.where(f, torch.where(M, S_, ninf), zero)
    fs = f.squeeze(1)
    z1 = torch.zeros(fs.shape[0], dtype=S_.dtype, device=S_.device)
    a = torch.where(fs, torch.logsumexp(za, 1), z1)
    mx = torch.where(fs, zm.amax(1), z1)
    return torch.stack([a, mx, fs.to(S_.dtype)], 1)


def r2_head(m, x):
    return Fn.linear(Fn.gelu(Fn.linear(x, m.r2_w1, m.r2_b1)), m.r2_w2, m.r2_b2).squeeze(-1)


class R2Mix:
    """The arm's final score plus R2's head (zero at the start) over each row's [a, m, f]."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        r2_init(self, seed)

    def r2_parts(self, r2, nq):
        S_, M = r2_logits(self, r2["eq"], r2["p"], r2["bri"], nq)
        return S_, M, prior_logp(M, S_.dtype)

    def r2_feats(self, r2, nq):
        return row_feats(*self.r2_parts(r2, nq))

    r2_head = r2_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if R2_KEY not in feats:
            return s
        return s + self.r2_head(self.r2_feats(feats[R2_KEY], nq))


class ZRCKPB(R2Mix, Z3P.ZRCKP):
    """zrckpb: zrckp's model plus R2's head."""


class ZG6M(R2Mix, G6.ZG6):
    """zg6m: zg6's model plus R2's head on its final score."""


ARMS = {"zrckpb": (ZRCKPB, R2KTChainCarveZRC), "zg6m": (ZG6M, R2KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── the EM term ──────────────────────────────────────────────────────────────


STASH = {"S": None, "M": None, "lpt": None, "N": None}
STATS = {}
COUNTS = ("batches", "batches_with_term", "questions", "questions_with_gold", "questions_with_term")


def reset_stats(lam=LAM):
    STATS.clear()
    STATS.update({"lam": lam, "epochs": [], "listwise_sum": 0.0, "br_sum": 0.0, **{k: 0 for k in COUNTS}})
    STATS["_ep"] = {"listwise_sum": 0.0, "br_sum": 0.0, **{k: 0 for k in COUNTS}}


reset_stats()


def clear_stash():
    STASH.update(S=None, M=None, lpt=None, N=None)


def r2_feats_em(self, r2, nq):
    """R2Mix.r2_feats, its arithmetic unchanged; in training with gradients on, the batch's logits, mask and prior are
    kept for the loss of the same step."""
    S_, M, lpt = self.r2_parts(r2, nq)
    if self.training and torch.is_grad_enabled():
        if STASH["S"] is not None:
            raise SystemExit("zr2br: a batch's bridge logits were kept and never scored")
        STASH.update(S=S_, M=M, lpt=lpt, N=int(nq.numel()))
    return row_feats(S_, M, lpt)


def seg_lse(S_, M, nq, B):
    """(B, K): per question and slot the log-sum-exp of S over the question's rows unmasked for the slot (a finite
    stand-in where it has none); the shift is the detached maximum."""
    N, Kk = S_.shape
    mx = torch.full((B, Kk), -math.inf, dtype=S_.dtype, device=S_.device).scatter_reduce(
        0, nq.unsqueeze(1).expand(N, Kk), torch.where(M, S_, torch.full_like(S_, -math.inf)).detach(), reduce="amax",
        include_self=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    sh = mx[nq]
    e = torch.where(M, torch.exp(torch.where(M, S_, sh) - sh), torch.zeros_like(S_))
    Zs = torch.zeros(B, Kk, dtype=S_.dtype, device=S_.device).index_add_(0, nq, e)
    return mx + torch.log(Zs.clamp_min(TINY))


def br_nll(S_, M, lpt, nq, B, gold):
    """The batch's sum over questions of L_br(q), the questions with a term, and the questions with a gold."""
    g = gold > 0
    ng = torch.zeros(B, dtype=torch.float64, device=S_.device).index_add_(0, nq, g.to(torch.float64))
    nqg = int((ng > 0).sum())
    sel = torch.nonzero(g & M.any(1)).squeeze(1)
    if sel.numel() == 0:
        return S_.sum() * 0.0, 0, nqg
    lse = seg_lse(S_, M, nq, B)
    qv = nq[sel]
    y = torch.where(M[sel], lpt[sel] + S_[sel] - lse[qv], torch.full_like(S_[sel], -math.inf))
    lv = -torch.logsumexp(y, 1)
    cnt = torch.zeros(B, dtype=S_.dtype, device=S_.device).index_add_(0, qv, torch.ones_like(lv))
    tot = torch.zeros(B, dtype=S_.dtype, device=S_.device).index_add_(0, qv, lv)
    has = cnt > 0
    return (tot[has] / cnt[has]).sum(), int(has.sum()), nqg


def make_loss(base, lam):
    def listwise_r2(scores, gold, nq, B):
        loss = base(scores, gold, nq, B)
        ep = STATS["_ep"]
        ep["batches"] += 1
        ep["questions"] += int(B)
        ep["listwise_sum"] += float(loss.detach())
        if STASH["S"] is None:
            return loss
        S_, M, lpt, N = STASH["S"], STASH["M"], STASH["lpt"], STASH["N"]
        clear_stash()
        if N != scores.numel():
            raise SystemExit(f"zr2br: the kept bridge logits are of a batch of {N} rows, the loss's of {scores.numel()}")
        nll, nqs, nqg = br_nll(S_, M, lpt, nq, B, gold)
        ep["questions_with_gold"] += nqg
        if nqs == 0:
            return loss
        ep["batches_with_term"] += 1
        ep["questions_with_term"] += nqs
        ep["br_sum"] += float(nll.detach())
        return loss + (lam * nll / nqs).to(loss.dtype)
    return listwise_r2


def close_epoch():
    ep = STATS["_ep"]
    STATS["epochs"].append({**{k: ep[k] for k in COUNTS}, "listwise_mean": ep["listwise_sum"] / max(ep["batches"], 1),
                            "br_nll_mean": ep["br_sum"] / max(ep["questions_with_term"], 1),
                            "share_with_term": ep["questions_with_term"] / max(ep["questions"], 1)})
    for k in COUNTS + ("listwise_sum", "br_sum"):
        STATS[k] += ep[k]
        ep[k] = 0 if isinstance(ep[k], int) else 0.0


@contextlib.contextmanager
def r2_em(lam=LAM):
    """lean_gpu's loop with the term added to its one loss: R2's r2_feats keeps each training batch's logits, the loss
    adds LAM times the batch mean of L_br, and each epoch's sums close when the loop keeps its state; all restored
    after, an error too."""
    saved = LG.listwiseD, R2Mix.r2_feats, LG.cpu_state
    base_state = saved[2]

    def cpu_state(model):
        close_epoch()
        return base_state(model)

    reset_stats(lam)
    clear_stash()
    LG.listwiseD, R2Mix.r2_feats, LG.cpu_state = make_loss(saved[0], lam), r2_feats_em, cpu_state
    try:
        yield
    finally:
        LG.listwiseD, R2Mix.r2_feats, LG.cpu_state = saved
        clear_stash()


def summary():
    return {"lam": STATS["lam"], **{k: STATS[k] for k in COUNTS}, "epochs": list(STATS["epochs"]),
            "listwise_mean": STATS["listwise_sum"] / max(STATS["batches"], 1),
            "br_nll_mean": STATS["br_sum"] / max(STATS["questions_with_term"], 1),
            "share_with_term": STATS["questions_with_term"] / max(STATS["questions"], 1)}


# ── train, read, compare ─────────────────────────────────────────────────────


def stamp(argv, arm, split, summ):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        g = Z2.gate_passed()
        rec.update({"zr2br_sha256": LC.sha_src(__file__), "zg6_sha256": LC.sha_src(G6.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "r2": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "device": Z2.DEVICE,
                        "constants": constants(), "settings": SETTINGS, "params": list(R2_PARAMS), "term": summ,
                        "graph": "today's",
                        "pools": f"P1e's {g.get('construction')} at each question's frozen size (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zr2br: train takes --arm zrckpb or zg6m, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zr2br: R2's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zr2br: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    bad = G6.fixed_flags(argv, FIXED_TRAIN)
    if bad:
        raise SystemExit(f"zr2br: R2 trains on its base's card settings (seed {LG.SEED}, hidden {LG.HIDDEN}, config "
                         f"{LG.CONFIG}, variant p, the split's fit carves and basis); {bad} would change them")
    check_constants()
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    with Z2.on_p1c2(), r2_em():
        rc = RM.train(argv, split)
    summ = summary()
    if summ["questions_with_term"] == 0:
        raise SystemExit(f"zr2br: no training batch carried the term ({summ['batches']} batches)")
    with Z2.on_p1c2():
        stamp(argv, arm, split, summ)
    ep = summ["epochs"]
    log(f"zr2br {arm} {split}: the term on {summ['questions_with_term']} of {summ['questions']} training questions "
        f"({summ['share_with_term']:.4f}); L_br {ep[0]['br_nll_mean']:.4f} (epoch 0) to {ep[-1]['br_nll_mean']:.4f} "
        f"(last)")
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm = rec.get("arm")
    if arm not in ARMS or "r2" not in rec or (rec.get("r2") or {}).get("arm") != arm:
        raise SystemExit(f"zr2br: {name} is not an R2 fit (arm {arm})")
    if name != FIT[arm].get(rec.get("split")):
        raise SystemExit(f"zr2br: {name} is not the {arm} fit of {rec.get('split')}")
    bad = G6.fixed_flags(argv, FIXED_READ)
    if bad:
        raise SystemExit(f"zr2br: R2 reads p's fits on the six s1eval carves; {bad} would change that")
    check_constants(rec["r2"].get("constants"), name)
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-calls against each base's card fits, under N2's null or the 7 October null ────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"zr2br: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zr2br: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def g6_rebase(null, sources, screens=None):
    """N2's zspk null (G6's loader) moved onto zg6's card fits, as relz.rel_null moves a null onto a base: each read's
    seed 0 must be the zspk R@5 that G6's comparison of zg6 with zspk's card fit holds, and becomes zg6's R@5 there;
    D1, D2 and the floor stay zspk's."""
    screens = screens or G6_SCREENS
    if sorted(screens) != sorted(SPLITS):
        raise SystemExit(f"zr2br: G6's comparisons cover {sorted(screens)}, not {sorted(SPLITS)}")
    for split in SPLITS:
        p = Path(screens[split]) if Path(screens[split]).is_absolute() else ROOT / screens[split]
        if not p.exists():
            raise SystemExit(f"zr2br: {p.name} missing (G6's comparison of zg6 with zspk's card fit of {split})")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if SR.parts_of(rec.get("new", ""))[-3:] != BASE_FITS["zg6m"][split] or \
                SR.parts_of((rec.get("bases") or [""])[0])[-3:] != G6.BASE_FITS[split]:
            raise SystemExit(f"zr2br: {p.name} compares {rec.get('new')} with {(rec.get('bases') or [None])[0]}, not "
                             f"zg6's card fit of {split} with zspk's")
        if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
            raise SystemExit(f"zr2br: {p.name} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
        rows = {r["dataset"]: r for r in rec["by_base"][rec["bases"][0]]["rows"]}
        if sorted(rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zr2br: {p.name} reads {sorted(rows)}, not the six datasets")
        for ds, v in null[split].items():
            if rows[ds]["base"][0] != v["seed0"]:
                raise SystemExit(f"zr2br: {p.name} {ds} is compared with zspk R@5 {rows[ds]['base'][0]}, N2's null "
                                 f"with {v['seed0']}")
            v["zspk_seed0"], v["seed0"] = v["seed0"], rows[ds]["new"][0]
        sources[f"G6 {split}"] = N2.repo_rel(str(p)) or p.name
    return null, sources


def null_loader(arm, w=None, g6_screens=None):
    """The deciding null's loader in screen_recall.load_null's form: N2's zrckp null (zn2's loader) for zrckpb; N2's
    zspk null (G6's loader) moved onto zg6's card fits for zg6m."""
    if arm == "zrckpb":
        ww = w or N2.Where()
        return lambda files: N2.load_null_n2("zrckp", files, ww)
    return lambda files: g6_rebase(*G6.n2_null(files), g6_screens)


@contextlib.contextmanager
def against(loader):
    """relz.against_rel's swap with the deciding null's loader: screen_pair's load decided against the base fits
    (relz.pair_load under on_base) and screen_recall's null the loader's; restored on the way out."""
    saved = SP.load, SR.load_null
    SP.load, SR.load_null = Z.pair_load, loader
    try:
        yield
    finally:
        SP.load, SR.load_null = saved


def fix(arm, null=None):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    sub = []
    if null == "N2":
        sub.append(("step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the tenth round)",
                    f"{N2_DESC[arm]}) (docs/SCREENS.md, section 2, and {DECLARED})"))
    return tuple(sub) + (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp)"),
                         ("(rel's screen fits)", f"({b}'s card fits)"),
                         ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (R2) |"),
                         ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm, null=None):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm, null):
            t = t.replace(a, b)
        head = f"R2 ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}"
        head += f", under {N2_DESC[arm] if null == 'N2' else OCT7_DESC}.\n\n" if null else ".\n\n"
        md.write_text(head + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": STAGE, "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "zr2br_sha256": LC.sha_src(__file__)})
        if null:
            rec.update({"null_kind": null, "null_is": N2_DESC[arm] if null == "N2" else OCT7_DESC,
                        "deciding": null == "N2"})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, null, base_screens=None, w=None, g6_screens=None):
    """relz's re-call of arm's pair against its base's card fits. null 'N2' decides (null_loader's null, no relz
    re-base); 'oct7': the 7 October null re-based to the card fits by relz.rel_null over the base screens
    (BASE_SCREENS), beside only."""
    if null not in NULLS:
        raise SystemExit(f"zr2br: the null is N2 or oct7, not {null}")
    p = Path(pair_file)
    pr = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if pr.get("stage") != STAGE or pr.get("arm") != arm:
        raise SystemExit(f"zr2br: {p.name} is not R2's pair of {arm} ({pr.get('stage')}, {pr.get('arm')})")
    with on_base(arm):
        if null == "N2":
            loader = null_loader(arm, w, g6_screens)
            saved = Z.against_rel
            Z.against_rel = lambda rel_screens=None: against(loader)
            try:
                rec = Z.recall(null_files, pair_file, out)
            finally:
                Z.against_rel = saved
        else:
            rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, arm, null)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r.get('split')} {r.get('dataset')}"


def recall_rows(rp, arm, null, problems):
    rp = Path(rp)
    if not rp.exists():
        problems.append(f"{rp.name} missing")
        return []
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if (rec.get("stage") != STAGE or rec.get("declared_in") != DECLARED or rec.get("arm") != arm
            or rec.get("base_arm") != BASE_OF[arm] or rec.get("null_kind") != null):
        problems.append(f"{rp.name} is not R2's re-call of {arm} under the {null} null")
    rows = rec.get("rows", [])
    if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted((sp, ds) for sp in SPLITS
                                                                         for ds in LG.EVAL_ORDER):
        problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six datasets")
    return rows


def calls_of(rows):
    return {"gains": [tag(r) for r in rows if r.get("call") == "GAIN"],
            "losses": [tag(r) for r in rows if r.get("call") == "LOSS"]}


def floors_problem(arm, root=ROOT):
    """A problem when N2's floors record of the arm's null is missing or not OK (zn2.py floors: REFUSED)."""
    p = Path(root) / "outputs" / "n2" / f"floors-{N2_BASE[arm]}.json"
    if not p.exists():
        return f"N2's floors record of {N2_BASE[arm]} is missing (zn2.py floors --base {N2_BASE[arm]})"
    r = json.loads(p.read_text(encoding="utf-8"))
    if r.get("status") != "OK":
        return f"N2's floors of {N2_BASE[arm]} are {r.get('status')}"
    return None


def fit_terms(arm, root=ROOT):
    """Per split, the EM term from the fit's R2 stamp: L_br's mean per epoch and the share of training questions with
    a term; None where the fit or its stamp is missing."""
    out = {}
    for sp in SPLITS:
        sj = Path(root) / "outputs" / "screen" / "fits" / FIT[arm][sp] / "screen.json"
        t = ((json.loads(sj.read_text(encoding="utf-8")).get("r2") or {}).get("term") if sj.exists() else None)
        out[sp] = None if not t else {"br_nll_by_epoch": [round(e["br_nll_mean"], 4) for e in t.get("epochs", [])],
                                      "share_with_term": round(t["share_with_term"], 4),
                                      "questions_with_term": t["questions_with_term"], "questions": t["questions"]}
    return out


def fmt(x, signed=True):
    if not isinstance(x, (int, float)):
        return "n/a"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def verdict(recalls=None, recalls7=None, compares=None, out=None, root=ROOT):
    """R2's screen verdict per arm from its re-call under N2's null (section 4) and its next form (section 5). Beside,
    deciding nothing: each arm's re-call under the 7 October null, webqsp and metaqa, musique in-domain for both arms,
    the arm minus its base on R@5 / FC@5 / hit@1 per read, and the EM term per epoch with its share of questions."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    recalls7 = recalls7 or {a: RECALL7[a].with_suffix(".json") for a in ARM_NAMES}
    compares = compares or {a: {sp: SCREEN / f"{FIT[a][sp]}.json" for sp in SPLITS} for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        problems, op = [], []
        rows = recall_rows(recalls[a], a, "N2", problems)
        fp = floors_problem(a, root)
        if fp:
            problems.append(fp)
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        orows = recall_rows(recalls7[a], a, "oct7", op)
        rp = Path(recalls[a])
        per[a] = {"verdict": v, "next_form": next_form(v, gains, losses), "gains": gains, "losses": losses,
                  "problems": problems, "calls": {tag(r): r.get("call") for r in rows},
                  "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "recall": rp.relative_to(ROOT).as_posix() if rp.is_relative_to(ROOT) else rp.name,
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
                  "beside": {"oct7_recall": None if op else calls_of(orows), "oct7_problems": op,
                             "reads": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                                       if r.get("dataset") in BESIDE},
                             "minus_base": {sp: G2.beside_rows(compares[a][sp], BASE_FITS[a][sp][2]) for sp in SPLITS},
                             "term": fit_terms(a, root)}}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "zr2br_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def calls_md(c, p):
        if c is None:
            return "not available (" + "; ".join(p) + ")"
        return (f"{len(c['gains'])} GAIN ({', '.join(c['gains']) or 'none'}), {len(c['losses'])} LOSS "
                f"({', '.join(c['losses']) or 'none'})")

    md = ["# R2's screen verdict: " + ", ".join(f"{a} **{per[a]['verdict']}**" for a in ARM_NAMES), "",
          f"Rule: {RULE}.", ""]
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the re-call under {N2_DESC[a]}: "
                  + calls_md(calls_of(recall_rows(recalls[a], a, "N2", [])), [])
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "")
                  + f". Next form (fixed in section 5 before any number): {p['next_form']}.")
    md += ["", "Beside, deciding nothing.", ""]
    for a in ARM_NAMES:
        b = per[a]["beside"]
        md.append(f"- {a} under the 7 October null (step 1's seeds): {calls_md(b['oct7_recall'], b['oct7_problems'])}. "
                  "webqsp and metaqa: " + (", ".join(f"{k} {x['call']} ({fmt(x['delta'])})"
                                                      for k, x in b["reads"].items()) or "none") + ".")
        for sp in SPLITS:
            t = b["term"][sp]
            md.append(f"  - {a} {sp}: " + ("no R2 stamp" if t is None else
                                           f"L_br per epoch {t['br_nll_by_epoch']}; the term on "
                                           f"{t['questions_with_term']} of {t['questions']} training questions "
                                           f"({t['share_with_term']:.4f})") + ".")
    md += [f"- musique in-domain ({ck}): " + ", ".join(f"{a} {side[a]['call']} ({fmt(side[a]['delta'])})"
                                                      for a in ARM_NAMES) + ".", "",
           "Each arm minus its base's card fit: R@5 with its 95% question-bootstrap interval, FC@5 and hit@1:", "",
           "| arm | split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075 floor) |",
           "|---|---|---|---|---|---|---|---|"]
    for a in ARM_NAMES:
        for sp in SPLITS:
            rs = per[a]["beside"]["minus_base"][sp]
            if rs is None:
                md.append(f"| {a} | {sp} | (no comparison with {BASE_FITS[a][sp][2]}) | | | | | |")
                continue
            for r in rs:
                d, lo, hi = r["delta"], r["lo"], r["hi"]
                md.append(f"| {a} | {sp} | {r['dataset']} | {r['read']} | {d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | "
                          f"{d[1]:+.4f} | {d[2]:+.4f} | {r['call']} |")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("zr2br verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def bridges_by_hand(rrf, n, k=K):
    out, o = [], 0
    for nn_ in n:
        v = [float(x) for x in rrf[o:o + nn_]]
        order = sorted(range(nn_), key=lambda i: (math.isnan(v[i]), 0.0 if math.isnan(v[i]) else -v[i], i))
        out.append(order[:k] + [-1] * (k - min(k, nn_)))
        o += nn_
    return np.array(out, np.int64).reshape(len(n), k)


def feats_by_hand(m, eq, p, bri, nq):
    """x = [a, m, f] in float64, question by question, row by row and bridge by bridge."""
    P = {k: getattr(m, k).detach().double().numpy() for k in R2_PARAMS}
    eq, p, bri, nq = eq.double().numpy(), p.double().numpy(), bri.numpy(), nq.numpy()
    out = np.zeros((p.shape[0], 3))
    for v in range(p.shape[0]):
        q = int(nq[v])
        hq, hv = eq[q] @ P["r2_U"], p[v] @ P["r2_V"]
        s, w = [], []
        for k in range(bri.shape[1]):
            b = int(bri[q, k])
            if b < 0 or b == v:
                continue
            hk = p[b] @ P["r2_V"]
            t = hq + P["r2_W1"] @ hk + P["r2_W2"] @ (hq * hk) + P["r2_o"]
            s.append(float(t @ hv) / math.sqrt(R_DIM))
            w.append(1.0 / (PRIOR_C + k + 1))
        if s:
            w = np.array(w) / sum(w)
            out[v] = [math.log(float((w * np.exp(np.array(s))).sum())), max(s), 1.0]
    return out


def nll_by_hand(S_, M, nq, gold, B):
    """The float64 reference: sum over questions of L_br(q), the questions with a term, and d sum / d S by EM's form
    (1/|G_q|) sum_v -r_k(v) (1[v' = v] - p_k(v'))."""
    S_, M, nq, gold = S_.numpy(), M.numpy(), nq.numpy(), gold.numpy() > 0
    N, Kk = S_.shape
    grad, total, nqs = np.zeros((N, Kk)), 0.0, 0
    pi = 1.0 / (PRIOR_C + np.arange(1, Kk + 1))
    for q in range(B):
        rows = np.flatnonzero(nq == q)
        gv = [v for v in rows if gold[v] and M[v].any()]
        if not gv:
            continue
        nqs += 1
        pk = np.zeros((N, Kk))
        for k in range(Kk):
            on = [v for v in rows if M[v, k]]
            if on:
                e = np.exp(S_[on, k] - S_[on, k].max())
                pk[on, k] = e / e.sum()
        lq = 0.0
        for v in gv:
            ks = np.flatnonzero(M[v])
            w = pi[ks] / pi[ks].sum()
            post = w * pk[v, ks]
            lq += -math.log(post.sum())
            r = post / post.sum()
            for j, k in enumerate(ks):
                on = [u for u in rows if M[u, k]]
                for u in on:
                    grad[u, k] += -r[j] * ((u == v) - pk[u, k]) / len(gv)
        total += lq / len(gv)
    return total, nqs, grad


def fake_recall(path, arm, calls, null="N2", over=None):
    rows = [{"split": sp, "dataset": ds, "read": "x", "base": 0.6, "new": 0.6, "delta": 0.0,
             "call": calls.get((sp, ds), "WITHIN")} for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "null_kind": null,
           "verdict": "x", "rows": rows}
    rec.update(over or {})
    LC.write_json(path, rec)
    return path


def snapshot():
    """Every file under R2's roots (outputs/r2; outputs/screen's scr-r2-* screens and fits) and the roots the re-calls
    read (outputs/l1, outputs/n2), with size and mtime; and whether outputs/r2 exists."""
    out = {}
    for base, pat in ((OUT, "**/*"), (SCREEN, "scr-r2-*"), (SCREEN / "fits", "scr-r2-*/**/*"),
                      (ROOT / "outputs" / "l1", "**/*"), (ROOT / "outputs" / "n2", "**/*")):
        if base.exists():
            for p in base.glob(pat):
                if p.is_file():
                    st = p.stat()
                    out[str(p)] = (st.st_size, st.st_mtime_ns)
    return out, OUT.exists()


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    snap0 = snapshot()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    saved_em = LG.listwiseD, R2Mix.r2_feats, LG.cpu_state
    # 1. the arms: zrckp's and zg6's models plus R2's head on their carves with the bridges beside; earlier arms
    #    unchanged; the constants are the declared ones; 59,537 parameters; the threads set before numpy came in
    assert S.ARMS["zrckpb"] == (ZRCKPB, R2KTChainCarveZRC) and S.ARMS["zg6m"] == (ZG6M, R2KTLinkCarveBase)
    assert ZRCKPB.__mro__[1:3] == (R2Mix, Z3P.ZRCKP) and ZG6M.__mro__[1:3] == (R2Mix, G6.ZG6)
    assert R2KTChainCarveZRC.__mro__[1] is Z3.KTChainCarveZRC and R2KTLinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zg6"] == (G6.ZG6, Z3.KTLinkCarveBase)
    assert S.ARMS["zspk"] == Z3.ARMS["zspk"] and S.ARMS["zg6t"] == G6.ARMS["zg6t"]
    assert constants() == DECLARED_CONST and n_params() == 59537 and (Q_DIM, P_DIM) == (1536, 257)
    assert SEED_OFF not in (5501, G6.SEED_OFF, G6.SEED_OFF + 1, G2.SEED_OFF, G2.SEED_OFF + 1, Z3.KT_SEED_OFF, 2301, 1201)
    assert all(os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"))
    if __name__ == "__main__":
        assert not _NP_BEFORE, "numpy came in before the threads were set"
    # 2. the bridges against the order by hand: ties by pool position, NaN last, -0.0 tied with 0, pools under five,
    #    of one and empty, in slices of any size
    rng = np.random.default_rng(61)
    n = np.array([7, 3, 1, 0, 9, 5, 12, 2])
    x = rng.choice(np.array([0.0, -0.0, 0.0161, 0.0164, 0.0328, np.nan, 0.0164, 0.05]), int(n.sum())).astype(np.float16)
    ref = bridges_by_hand(x, n)
    for blk in (BRIDGE_ROWS, 1, 8, 13):
        assert np.array_equal(bridges_np(x, n, block=blk), ref), blk
    assert (ref[2] == [0, -1, -1, -1, -1]).all() and (ref[3] == -1).all() and (ref[1, 3:] == -1).all()
    xs = x.astype(np.float64)
    tie_q = [q for q in range(n.size) if np.unique(xs[n[:q].sum():n[:q + 1].sum()][ref[q][ref[q] >= 0]]).size
             < (ref[q] >= 0).sum()]
    assert tie_q and np.isnan(xs).any() and (np.signbit(xs) & (xs == 0)).any()
    SR.must_stop(bridges_np, x[:-1], n)
    tmp = Path(tempfile.mkdtemp(prefix="zr2br_"))
    try:
        rng = np.random.default_rng(62)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            """G6's toy carve with unit question vectors and store vectors of norm 0.9, the cache's scales."""

            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]
                self.q_emb = self.q_emb / self.q_emb.norm(dim=1, keepdim=True)
                self.P = self.P * (0.9 / self.P.norm(dim=1, keepdim=True))

        ch_cls = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        lk_cls = ZL.link_carve(ch_cls, tmp / "lk")
        c0 = ch_cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
        kt_ch, kt_lk = Z3.kt_carve(ch_cls, tmp / "kt"), Z3.kt_carve(lk_cls, tmp / "kt")
        cm = r2_carve(kt_ch)("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        cg = r2_carve(kt_lk)("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        bare = {"zrckpb": kt_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache"),
                "zg6m": kt_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache")}
        carve = {"zrckpb": cm, "zg6m": cg}
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(cm.rows)[::-1][:7], np.arange(cm.rows)[:5]].astype(np.int64)
        B = qs.size
        keep = torch.ones(B, len(blocks))
        # 3. the carve: the bridges are bridges_np on the carve's own rrf column; a batch's bridge rows are its
        #    questions' own, in its row order; e_q and p are SEMB's, the same without SEMB in the blocks
        assert np.array_equal(cm.r2_br, bridges_np(cm.X[:, S3.I_RRF].numpy(), cm.n_np))
        assert np.array_equal(cm.r2_br, cg.r2_br) and cm.nbytes() == bare["zrckpb"].nbytes() + cm.r2_br.nbytes
        assert (cm.n_np < K).any() and (cm.n_np > K).any(), np.unique(cm.n_np)
        feats, nq, bz, gold = cm.batch(qs, blocks)
        r2 = feats[R2_KEY]
        N = nq.numel()
        cnt = cm.n_np[qs]
        seg = np.cumsum(cnt) - cnt
        for b, q in enumerate(qs):
            want = [seg[b] + j if j >= 0 else -1 for j in cm.r2_br[q]]
            assert r2["bri"][b].tolist() == want
            assert all(int(nq[i]) == b for i in want if i >= 0)
        assert r2["eq"] is feats["SEMB"][0] and r2["p"] is feats["SEMB"][1] and r2["p"].shape[1] == P_DIM
        f_rank, *_ = cm.batch(qs, ["rank"])
        assert "SEMB" not in f_rank and torch.equal(f_rank[R2_KEY]["eq"], r2["eq"])
        assert torch.equal(f_rank[R2_KEY]["p"], r2["p"]) and torch.equal(f_rank[R2_KEY]["bri"], r2["bri"])
        fg, nq_g, bz_g, gold_g = cg.batch(qs, blocks)
        assert torch.equal(fg[R2_KEY]["bri"], r2["bri"]) and ZL.LK_KEY in fg and ZL.LK_KEY not in feats
        # 4. each arm starts as its base bit for bit (state, forward in eval and in training, the global generator);
        #    R2's parameters are its own 59,537 at r 32 from its own generator
        nets = {}
        for arm, base in (("zrckpb", Z3P.ZRCKP), ("zg6m", G6.ZG6)):
            f_ = feats if arm == "zrckpb" else fg
            torch.manual_seed(0)
            m0 = base(blocks, cm.widths, 16, dropout=0.1, seed=0)
            r0 = torch.rand(4)
            torch.manual_seed(0)
            m1 = ARMS[arm][0](blocks, cm.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(r0, torch.rand(4))
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(R2_PARAMS) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k: s1[k] for k in s0})
            assert sum(s1[k].numel() for k in R2_PARAMS) == n_params()
            for mode in (False, True):
                m0.train(mode)
                m1.train(mode)
                with torch.no_grad():
                    torch.manual_seed(5)
                    y0 = m0(f_, keep, nq, B, bz)
                    torch.manual_seed(5)
                    assert torch.equal(y0, m1(f_, keep, nq, B, bz)), (arm, mode)
            m0.eval()
            m1.eval()
            nets[arm] = (m0, m1)
        # 5. the head's inputs are the model by hand (float64); a row is never its own bridge; a pool of one has none
        #    (a = m = 0); the score is the base's plus the head; without R2's input the arm is its base
        m0, m1 = nets["zrckpb"]
        gen = torch.Generator().manual_seed(63)
        scale = {"r2_U": 1.0, "r2_V": 1.0, "r2_W1": 0.3 / R_DIM ** 0.5, "r2_W2": 0.3 / R_DIM ** 0.5, "r2_o": 0.3,
                 "r2_w1": 1.0, "r2_b1": 0.3, "r2_w2": 0.5, "r2_b2": 0.1}
        with torch.no_grad():
            for k in R2_PARAMS:
                getattr(m1, k).copy_(torch.randn(getattr(m1, k).shape, generator=gen) * scale[k])
            xr = m1.r2_feats(r2, nq)
            ref = feats_by_hand(m1, r2["eq"], r2["p"], r2["bri"], nq)
            assert np.abs(xr.double().numpy() - ref).max() < 2e-4, np.abs(xr.double().numpy() - ref).max()
            S_, M = r2_logits(m1, r2["eq"], r2["p"], r2["bri"], nq)
            own = r2["bri"][nq] == torch.arange(N).unsqueeze(1)
            assert own.any() and not (M & own).any()
            eq2, p3 = r2["eq"][:2], r2["p"][:3]
            bri2, nq2 = torch.tensor([[0, -1, -1, -1, -1], [1, 2, -1, -1, -1]]), torch.tensor([0, 1, 1])
            x2 = m1.r2_feats({"eq": eq2, "p": p3, "bri": bri2}, nq2)
            assert (x2[0] == 0).all() and (x2[1:, 2] == 1).all() and torch.allclose(x2[1:, 0], x2[1:, 1], atol=1e-6)
            assert np.abs(x2.double().numpy() - feats_by_hand(m1, eq2, p3, bri2, nq2)).max() < 2e-4
            fz = xr[:, 2] == 0
            assert bool((xr[fz, :2] == 0).all()) and bool(torch.isfinite(xr).all())
            y0, y1 = m0(feats, keep, nq, B, bz), m1(feats, keep, nq, B, bz)
            head = m1.r2_head(xr)
            assert float(head.abs().max()) > 1e-3 and torch.allclose(y1 - y0, head, atol=1e-5)
            assert torch.equal(m1({k: v for k, v in feats.items() if k != R2_KEY}, keep, nq, B, bz), y0)
            # 6. the MLP rule: a row's head reads its own vector, the question's and its bridges', never a score or
            #    base_z; moving a row that is no bridge moves its own inputs only; moving a bridge moves its question
            #    only
            bset = {int(i) for i in r2["bri"].flatten() if i >= 0}
            free = [v for v in range(N) if v not in bset and int(cnt[int(nq[v])]) > K]
            assert free and bset
            for v, is_bridge in ((free[0], False), (min(bset), True)):
                p2 = r2["p"].clone()
                p2[v] += torch.randn(P_DIM, generator=gen)
                x2 = m1.r2_feats({**r2, "p": p2}, nq)
                same = torch.ones(N, dtype=torch.bool)
                same[v] = False
                if is_bridge:
                    same &= nq != nq[v]
                assert torch.equal(x2[same], xr[same]) and not torch.equal(x2[~same], xr[~same]), is_bridge
            bz2 = bz + torch.randn(bz.shape, generator=gen)
            assert torch.allclose(m1(feats, keep, nq, B, bz2) - m0(feats, keep, nq, B, bz2), head, atol=1e-5)
            m1.eval()
        # 7. the EM term: the float64 reference's sum, its questions and its gradient, which is EM's; no gold, no term;
        #    the loss adds LAM times the batch mean; the kept r2_feats is the plain one bit for bit, keeps only in
        #    training with gradients on, and a second kept batch is refused; the swap is restored, an error too
        with r2_em():
            assert R2Mix.r2_feats is r2_feats_em and ZRCKPB.r2_feats is r2_feats_em and ZG6M.r2_feats is r2_feats_em
            assert LG.listwiseD is not saved_em[0] and LG.cpu_state is not saved_em[2]
        assert (LG.listwiseD, R2Mix.r2_feats, LG.cpu_state) == saved_em
        try:
            with r2_em():
                raise ValueError("x")
        except ValueError:
            pass
        assert (LG.listwiseD, R2Mix.r2_feats, LG.cpu_state) == saved_em and STASH["S"] is None
        S64 = S_.detach().double().clone().requires_grad_(True)
        lpt = prior_logp(M, torch.float64)
        tot, nqs, nqg = br_nll(S64, M, lpt, nq, B, gold)
        ref_t, ref_n, ref_g = nll_by_hand(S_.detach().double(), M, nq, gold, B)
        assert nqs == ref_n >= 3 and abs(float(tot) - ref_t) < 1e-9 * max(1.0, abs(ref_t)), (float(tot), ref_t)
        assert nqg == int((torch.zeros(B).index_add_(0, nq, gold.float()) > 0).sum()) >= nqs
        tot.backward()
        assert np.abs(S64.grad.numpy() - ref_g).max() < 1e-9 and np.abs(ref_g).max() > 1e-3
        assert float(S64.grad[~M].abs().max()) == 0.0
        t0z, n0z, _ = br_nll(S64.detach(), M, lpt, nq, B, torch.zeros_like(gold))
        assert n0z == 0 and float(t0z) == 0.0
        m1.train()
        with r2_em():
            a = saved_em[1](m1, r2, nq)
            b = m1.r2_feats(r2, nq)
            assert torch.equal(a, b) and STASH["S"] is not None and STASH["N"] == N
            SR.must_stop(m1.r2_feats, r2, nq)
            clear_stash()
            with torch.no_grad():
                m1.r2_feats(r2, nq)
            assert STASH["S"] is None
            m1.eval()
            m1.r2_feats(r2, nq)
            assert STASH["S"] is None
            m1.train()
            s = m1(feats, keep, nq, B, bz)
            S_k, M_k, l_k = STASH["S"], STASH["M"], STASH["lpt"]
            lst = saved_em[0](s, gold, nq, B)
            tot_l = LG.listwiseD(s, gold, nq, B)
            ref_k, ref_kn, _ = nll_by_hand(S_k.detach().double(), M_k, nq, gold, B)
            assert abs(float(tot_l) - float(lst) - LAM * ref_k / ref_kn) < 1e-5, (float(tot_l), float(lst), ref_k)
            assert torch.equal(l_k, prior_logp(M_k, torch.float32))
            assert STASH["S"] is None and float(LG.listwiseD(s, gold, nq, B)) == float(lst)
            m1(feats, keep, nq, B, bz)
            SR.must_stop(LG.listwiseD, s[:-1], gold[:-1], nq[:-1], B)
            clear_stash()
        m1.eval()
        # 8. toy fits through lean_gpu's loop: with LAM 0 every epoch's state and the SWA state are the plain loop's
        #    bit for bit; with LAM 1 every state moves, L_br falls and the term is counted per epoch; a repeat is
        #    identical; on the carve without R2's input each arm's fit is its base's on every shared key, R2's
        #    parameters at their start
        cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        terms = {}
        for arm in ARM_NAMES:
            cv = carve[arm]
            with S.patched(arm):
                plain = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} plain")
                with r2_em(0.0):
                    zero = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} lam 0")
                    sz = summary()
                with r2_em():
                    fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} lam 1")
                    sf = summary()
                with r2_em():
                    rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} repeat")
                nob = LG.fit_variant([bare[arm]], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} no bridges")
            with S.patched(BASE_OF[arm]):
                base = LG.fit_variant([bare[arm]], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{BASE_OF[arm]}")
            assert (LG.listwiseD, R2Mix.r2_feats, LG.cpu_state) == saved_em
            sts = lambda f: f["states"] + [f["swa"]]  # noqa: E731
            assert all(not LG.same_state(a, b) for a, b in zip(sts(plain), sts(zero))), arm
            assert all(LG.same_state(a, b) for a, b in zip(sts(plain), sts(fit))), arm
            assert all(not LG.same_state(a, b) for a, b in zip(sts(fit), sts(rep))), arm
            assert len(sf["epochs"]) == cfg["epochs"] and sf["questions_with_term"] > 0
            assert sz["questions_with_term"] == sf["questions_with_term"] and 0 < sf["share_with_term"] <= 1
            assert all(e["batches_with_term"] > 0 and e["questions"] >= e["questions_with_gold"]
                       >= e["questions_with_term"] for e in sf["epochs"])
            assert sf["epochs"][-1]["br_nll_mean"] < sf["epochs"][0]["br_nll_mean"], sf["epochs"]
            assert float(fit["swa"]["r2_w2"].abs().max()) > 0 and not torch.equal(fit["swa"]["r2_U"],
                                                                                 plain["swa"]["r2_U"])
            torch.manual_seed(0)
            st0 = ARMS[arm][0](blocks, nob["widths"], 16, dropout=0.1, seed=0).state_dict()
            for a, b in zip(sts(nob), sts(base)):
                assert not LG.same_state({k: a[k] for k in b}, b), arm
                assert all(torch.equal(a[k], st0[k]) for k in R2_PARAMS), arm
            terms[arm] = (sf["epochs"][0]["br_nll_mean"], sf["epochs"][-1]["br_nll_mean"],
                          sz["epochs"][0]["br_nll_mean"], sz["epochs"][-1]["br_nll_mean"])
            log(f"  toy {arm}: L_br {terms[arm][0]:.4f} -> {terms[arm][1]:.4f} with the term, {terms[arm][2]:.4f} -> "
                f"{terms[arm][3]:.4f} at LAM 0; the term on {sf['questions_with_term']} of {sf['questions']} "
                f"training questions")
        # 9. train and read refuse what is not R2's: another arm, split or fit name, the cache root, the device, a card
        #    setting, constants not the declared ones; a fit not R2's, of another split, fitted with other constants,
        #    or a read flag that changes the reads
        cr = ["--cache-root", "outputs/p1c2/cache"]
        for arm in ARM_NAMES:
            good = ["train", "--arm", arm, "--name", FIT[arm]["L-musique"]] + cr
            SR.must_stop(train, ["train", "--arm", BASE_OF[arm], "--name", FIT[arm]["L-musique"]] + cr, "L-musique")
            SR.must_stop(train, good, "L-2wiki")
            SR.must_stop(train, good, "L-hotpotqa")
            SR.must_stop(train, ["train", "--arm", arm, "--name", FIT[arm]["L-musique"]], "L-musique")
            SR.must_stop(train, good[:-1] + ["outputs/p1gb/cache"], "L-musique")
            SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
            for extra in (["--seed", "1"], ["--hid", "64"], ["--config", "1e-3:0:0:4:1"], ["--variants", "q"],
                          ["--train", "metaqa=fit"], ["--basis", "musique"], ["--repeat", "2"], ["--no-verify"]):
                SR.must_stop(train, good + extra + ["--device", "cuda"], "L-musique")
        other = [x for x in ARM_NAMES if x != "zrckpb"][0]
        SR.must_stop(train, ["train", "--arm", "zrckpb", "--name", FIT[other]["L-musique"]] + cr, "L-musique")
        g = globals()
        for k_, v_ in (("K", 4), ("R_DIM", 16), ("LAM", 0.5), ("PRIOR_C", 0)):
            keep_v = g[k_]
            g[k_] = v_
            try:
                SR.must_stop(check_constants)
                SR.must_stop(train, ["train", "--arm", "zrckpb", "--name", FIT["zrckpb"]["L-musique"]] + cr +
                             ["--device", "cuda"], "L-musique")
            finally:
                g[k_] = keep_v
        assert constants() == DECLARED_CONST
        fits = tmp / "fits"
        ok_r2 = {"arm": "zrckpb", "constants": DECLARED_CONST}
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}, "split": "L-musique"}),
                        ("y", {"arm": "zrckpb", "split": "L-musique"}),
                        ("scr-r2-zrckpb", {"arm": "zrckpb", "r2": {**ok_r2, "arm": "zg6m"}, "split": "L-musique"}),
                        ("scr-r2-zrckpb", {"arm": "zrckpb", "r2": ok_r2, "split": "L-hotpotqa"}),
                        ("scr-r2-zrckpb", {"arm": "zg6m", "r2": {**ok_r2, "arm": "zg6m"}, "split": "L-musique"}),
                        ("scr-r2-zrckpb", {"arm": "zrckpb", "r2": {**ok_r2, "constants": {**DECLARED_CONST, "K": 4}},
                                           "split": "L-musique"})):
            (fits / nm).mkdir(parents=True, exist_ok=True)
            LC.write_json(fits / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(fits)] + cr)
        LC.write_json(fits / "scr-r2-zrckpb" / "screen.json", {"arm": "zrckpb", "r2": ok_r2, "split": "L-musique"})
        for extra in (["--read", "metaqa=s1eval"], ["--basis", "musique"], ["--no-verify"], ["--variants", "q"]):
            SR.must_stop(read, ["--name", "scr-r2-zrckpb", "--out-root", str(fits)] + cr + extra)
        SR.must_stop(read, ["--name", "scr-r2-zrckpb", "--out-root", str(fits), "--cache-root", "outputs/p1gb/cache"])
        SR.must_stop(read, ["--name", "none", "--out-root", str(fits)] + cr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert (LG.listwiseD, R2Mix.r2_feats, LG.cpu_state) == saved_em
    # 10. relz's pair and re-calls of each arm against its base's card fits, stamped R2's: under N2's null (zrckp's
    #     through zn2's loader for zrckpb; zspk's through G6's, moved onto zg6's card fits, for zg6m), deciding, and
    #     under the 7 October null re-based by the base screens, beside; the loaders refuse what is not theirs; relz,
    #     screen_pair and screen_recall restored after
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        w = N2.Where(T)
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit, Z.against_rel
        saved_io = SP.load, SR.load_null
        big = {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)}
        null7 = SR.fake_null(T / "null", "n1", big)
        n2 = {"zrckpb": [str(p) for p in N2.toy_null(w, "zrckp", big, r5=0.6)],
              "zg6m": [str(p) for p in G6.fake_n2(T / "n2z", big, base_r5=0.58)]}
        g6s = {sp: str(SR.fake_compare(T / "g6" / "screen", G6.FIT["zg6"][sp], sp, {ds: 0.02 for ds in LG.EVAL_ORDER},
                                       arm="zg6", base_r5=0.58, base="/".join(("outputs",) + G6.BASE_FITS[sp])))
               for sp in SPLITS}
        bad6 = {sp: str(SR.fake_compare(T / "g6b" / "screen", G6.FIT["zg6"][sp], sp, {}, arm="zg6", base_r5=0.59,
                                        base="/".join(("outputs",) + G6.BASE_FITS[sp]))) for sp in SPLITS}
        nl, src = g6_rebase(*G6.n2_null(n2["zg6m"]), g6s)
        assert nl["L-musique"]["metaqa"]["seed0"] == 0.6 and nl["L-musique"]["metaqa"]["zspk_seed0"] == 0.58
        assert abs(nl["L-hotpotqa"]["hotpotqa"]["floor"] - SR.floor_of(-0.010, 0.012)) < 1e-12
        assert sorted(src) == sorted([f"N2 {sp} seed {sd}" for sp in SR.NULL_SPLITS for sd in SR.SEEDS] +
                                     [f"G6 {sp}" for sp in SPLITS])
        SR.must_stop(g6_rebase, *G6.n2_null(n2["zg6m"]), bad6)
        SR.must_stop(g6_rebase, *G6.n2_null(n2["zg6m"]), {"L-musique": g6s["L-musique"]})
        SR.must_stop(G6.n2_null, n2["zrckpb"])
        SR.must_stop(N2.load_null_n2, "zrckp", n2["zg6m"], w)
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            bs = {sp: str(SR.fake_compare(T / f"b-{arm}" / "screen", fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in SPLITS}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in SPLITS}
            za = SR.fake_compare(T / arm, FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm, base_r5=0.6,
                                 base=zb["L-musique"])
            zh = SR.fake_compare(T / arm, FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            zw = SR.fake_compare(T / f"{arm}w", FIT[arm]["L-musique"], "L-musique", {}, arm=arm, base_r5=0.6,
                                 base="outputs/screen/fits/scr-p1c3-zspk")
            SR.must_stop(pair, arm, [zw, zh], T / "bad")
            pr = pair(arm, [za, zh], T / f"p-{arm}")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T / f"p-{arm}.json").read_text(encoding="utf-8"))
            assert pj["stage"] == STAGE and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            assert "null_kind" not in pj and "round" not in pj
            assert (T / f"p-{arm}.md").read_text(encoding="utf-8").startswith(f"R2 ({DECLARED}): {arm}")
            for null, files, extra in (("N2", n2[arm], None), ("oct7", null7, bs)):
                r = recall(arm, files, T / f"p-{arm}.json", T / f"r-{arm}-{null}", null, extra, w, g6s)
                assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], \
                    (arm, null, r["verdict"], r["changed"])
                rj = json.loads((T / f"r-{arm}-{null}.json").read_text(encoding="utf-8"))
                assert rj["stage"] == STAGE and rj["arm"] == arm and rj["base_arm"] == BASE_OF[arm]
                assert rj["null_kind"] == null and rj["deciding"] == (null == "N2") and rj["declared_in"] == DECLARED
                tx = (T / f"r-{arm}-{null}.md").read_text(encoding="utf-8")
                assert tx.startswith(f"R2 ({DECLARED}): {arm}") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (R2) |" in tx
                assert (N2_DESC[arm] in tx) == (null == "N2") and "rel's" not in tx
                assert ("step 1's seeds 1 and 2 against seed 0" in tx) == (null == "oct7")
            SR.must_stop(recall, arm, null7, T / f"p-{arm}.json", T / "x1", "N2", None, w, g6s)
            SR.must_stop(recall, arm, n2[arm], T / f"p-{arm}.json", T / "x2", "oct7", bs, w, g6s)
            SR.must_stop(recall, arm, n2[arm], T / f"p-{arm}.json", T / "x3", "N3", None, w, g6s)
            SR.must_stop(recall, [x for x in ARM_NAMES if x != arm][0], n2[arm], T / f"p-{arm}.json", T / "x4", "N2",
                         None, w, g6s)
            if arm == "zg6m":
                SR.must_stop(recall, arm, n2[arm], T / f"p-{arm}.json", T / "x5", "N2", None, w, bad6)
            assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit, Z.against_rel) == saved_z
            assert (SP.load, SR.load_null) == saved_io
        # 11. the verdict per arm: no LOSS and a GAIN under N2's null adopts for the full run; a LOSS or no GAIN does
        #     not, each with its next form; a missing re-call, one not R2's, not N2's, of another arm or not twelve
        #     reads, or N2's floors missing or refused, is INCOMPLETE; beside: the 7 October re-calls, webqsp and
        #     metaqa, musique in-domain, the arm minus its base per read and the EM term from the fits' stamps
        cmp = {}
        for a in ARM_NAMES:
            cmp[a] = {}
            for sp in SPLITS:
                b0 = f"outputs\\screen\\fits\\{BASE_FITS[a][sp][2]}"
                rows_b = [{"dataset": ds, "read": "x", "delta": [0.01, 0.02, -0.01], "lo": [0.005] * 3,
                           "hi": [0.015] * 3, "call": "GAIN"} for ds in LG.EVAL_ORDER]
                cmp[a][sp] = T / f"cmp-{a}-{sp}.json"
                LC.write_json(cmp[a][sp], {"bases": [b0, "outputs\\step1\\fits\\x"], "by_base": {b0: {"rows": rows_b}}})
            sj = T / "outputs" / "screen" / "fits" / FIT[a]["L-musique"] / "screen.json"
            sj.parent.mkdir(parents=True, exist_ok=True)
            LC.write_json(sj, {"arm": a, "r2": {"term": {"epochs": [{"br_nll_mean": 1.5}, {"br_nll_mean": 1.2}],
                                                         "share_with_term": 0.81, "questions_with_term": 81,
                                                         "questions": 100}}})
        gm, lw, mi = ("L-musique", "musique"), ("L-hotpotqa", "webqsp"), CLAIM1_READ
        o7 = {a: fake_recall(T / f"o7-{a}.json", a, {gm: "GAIN"}, null="oct7") for a in ARM_NAMES}
        for b in ("zrckp", "zspk"):
            LC.write_json(T / "outputs" / "n2" / f"floors-{b}.json", {"status": "OK"})
        cases = ((({gm: "GAIN", mi: "GAIN"}, {}), ({gm: "GAIN", lw: "LOSS"}, {}), "ADOPT_FOR_FULL_RUN", "NOT_ADOPTED",
                  NEXT["ADOPT_FOR_FULL_RUN"], NEXT["GAIN_AND_LOSS"]),
                 (({}, {}), ({lw: "LOSS"}, {}), "NOT_ADOPTED", "NOT_ADOPTED", NEXT["NO_GAIN"], NEXT["NO_GAIN"]),
                 (({gm: "GAIN"}, {"stage": "G6"}), ({gm: "GAIN"}, {"null_kind": "oct7"}), "INCOMPLETE", "INCOMPLETE",
                  NEXT["INCOMPLETE"], NEXT["INCOMPLETE"]),
                 (({gm: "GAIN"}, {"arm": "zg6m"}), ({gm: "GAIN"}, {"rows": [{"split": "L-musique", "dataset": "musique",
                                                                             "call": "GAIN"}]}),
                  "INCOMPLETE", "INCOMPLETE", NEXT["INCOMPLETE"], NEXT["INCOMPLETE"]),
                 (({gm: "GAIN"}, {"base_arm": "zrc"}), ({gm: "GAIN"}, {"declared_in": G6.DECLARED}), "INCOMPLETE",
                  "INCOMPLETE", NEXT["INCOMPLETE"], NEXT["INCOMPLETE"]))
        for i, (ca, cb, va, vb, na, nb) in enumerate(cases):
            rc_ = {"zrckpb": fake_recall(T / f"rc-a{i}.json", "zrckpb", ca[0], over=ca[1]),
                   "zg6m": fake_recall(T / f"rc-b{i}.json", "zg6m", cb[0], over=cb[1])}
            v = verdict(rc_, o7, cmp, T / f"v{i}", root=T)
            got = {a: (v["arms"][a]["verdict"], v["arms"][a]["next_form"]) for a in ARM_NAMES}
            assert got == {"zrckpb": (va, na), "zg6m": (vb, nb)}, (i, got, {a: v["arms"][a]["problems"]
                                                                           for a in ARM_NAMES})
            b = v["arms"]["zrckpb"]["beside"]
            assert b["oct7_recall"] == {"gains": ["L-musique musique"], "losses": []}
            assert all(len(b["minus_base"][sp]) == 6 for sp in SPLITS)
            assert b["minus_base"]["L-musique"][0]["delta"] == [0.01, 0.02, -0.01]
            assert b["term"]["L-musique"] == {"br_nll_by_epoch": [1.5, 1.2], "share_with_term": 0.81,
                                              "questions_with_term": 81, "questions": 100}
            assert b["term"]["L-hotpotqa"] is None
            if i == 0:
                assert v["musique_in_domain_beside"]["zrckpb"]["call"] == "GAIN"
                assert sorted(b["reads"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
            assert "Next form" in (T / f"v{i}.md").read_text(encoding="utf-8")
        LC.write_json(T / "outputs" / "n2" / "floors-zrckp.json", {"status": "REFUSED"})
        rf = {a: fake_recall(T / f"rf-{a}.json", a, {gm: "GAIN"}) for a in ARM_NAMES}
        v = verdict(rf, o7, cmp, T / "vf", root=T)
        assert v["arms"]["zrckpb"]["verdict"] == "INCOMPLETE" and v["arms"]["zg6m"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert v["arms"]["zrckpb"]["problems"] == ["N2's floors of zrckp are REFUSED"], v["arms"]["zrckpb"]["problems"]
        (T / "outputs" / "n2" / "floors-zspk.json").unlink()
        v = verdict(rf, o7, cmp, T / "vm", root=T)
        assert v["arms"]["zg6m"]["verdict"] == "INCOMPLETE" and "zspk is missing" in v["arms"]["zg6m"]["problems"][0]
        v = verdict({a: T / f"none-{a}.json" for a in ARM_NAMES}, {a: T / "n7.json" for a in ARM_NAMES},
                    {a: {sp: T / "x.json" for sp in SPLITS} for a in ARM_NAMES}, T / "vn", root=T)
        assert all(v["arms"][a]["verdict"] == "INCOMPLETE" and v["arms"][a]["beside"]["oct7_recall"] is None
                   and v["arms"][a]["beside"]["minus_base"] == {sp: None for sp in SPLITS} for a in ARM_NAMES)
    # 12. nothing written under R2's roots or the roots its re-calls read
    assert snapshot() == snap0, "the selftest wrote under R2's roots"
    log(f"zr2br selftest: arms zrckpb and zg6m (zrckp's and zg6's models plus R2's head, {n_params()} parameters at r "
        f"{R_DIM}) on their carves with the bridges beside, earlier arms unchanged; the bridges match the order by hand "
        f"(ties by pool position, NaN last, -0.0, pools under five, of one, empty, any slice); a batch's bridges are "
        f"its questions' own and e_q, p are SEMB's; each arm starts as its base bit for bit (state, forward in eval and "
        f"training, the global generator); the head's inputs match float64 by hand, no row is its own bridge; the MLP "
        f"rule (a non-bridge row moves only itself, a bridge only its question); the EM term matches float64 by hand "
        f"and its gradient is EM's to 1e-9; the kept r2_feats is the plain one, keeps in training only, the swap "
        f"restored; toy fits: LAM 0 is the plain loop bit for bit, LAM 1 lowers L_br (zrckpb {terms['zrckpb'][0]:.4f} "
        f"-> {terms['zrckpb'][1]:.4f}, zg6m {terms['zg6m'][0]:.4f} -> {terms['zg6m'][1]:.4f}), a repeat is identical, "
        f"without the bridges each fit is its base's; train and read refusals (constants included); relz's pair and "
        f"re-calls under N2's and the 7 October nulls, stamped R2's, the loaders' refusals, relz restored; the verdict's "
        f"outcomes, next forms and beside rows; nothing written under R2's roots ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        return 0 if all(v["verdict"] != "INCOMPLETE" for v in verdict()["arms"].values()) else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--null-kind", choices=NULLS)
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair or not g.null_kind:
                    gp.error("recall: --pair, --null-kind and the four --null files")
                recall(g.arm, null, g.pair, g.out, g.null_kind)
        except SystemExit as e:
            log(f"zr2br {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zr2br: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
