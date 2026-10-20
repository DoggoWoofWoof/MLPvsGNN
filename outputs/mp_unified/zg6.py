"""G6's screen (docs/G6_FREE_GNN.md): the free GNN, its own model on the lean inputs.

zg2 (G2) kept zspk whole and added a hidden-state stack whose readout was bounded to +-0.5 around zspk's score. The
user's direction of 20 October is that the GNN is not to be built on the MLP. zg6 is its own model, trained from scratch:
- encoder: zrm's block encoder, lean_screen3.ZRet's forward up to its output layer's input (every block's raw, z-scored
  and mask columns through l1 and l2, each with GELU and dropout), hidden 128, its own weights drawn from seed 0. zrm's
  output layer, its rrf skip (base_w) and its chain gate (cm_gate) are deleted, so no MLP score is formed or read;
- boundary: xb = [KT (the row's eight KT columns), fm, fr (rmatch's two chain features of the row, 0 on a graph
  without chains)]; h0 = GELU(W_in [H, xb] + b_in), D wide;
- context: s0 = w0 . h0 per row, z-scored over the question's rows and clipped to +-8; the question's context c is its
  rows' states weighted by the softmax of that, recomputed from the current states at each layer;
- LAYERS layers, each zg2's (zg2.g2_stack's loop, op for op), over zlink's pool edges u -> v of family f:
    l_e   = clip(<W_a h_u, c_q(v)> / sqrt(D) + beta_f, -8, 8)
    m_e   = W_f h_u                                     (one D x D map per family)
    mean  = per family, sum of sigmoid(l_e) m_e over v's edges / v's edge count in that family (0 without one)
    att   = over all of v's edges, sum of exp(l_e) m_e / sum of exp(l_e) (0 without one)
    h     = h + GELU(layer_norm(W_s h + W_mean [mean_0, mean_1, mean_2] + W_att att + W_x xb + b))
  xb enters every layer (NBFNet's boundary condition);
- readout: score = w_o . h_LAYERS + w_s . h0 + b_o, unbounded; w_o and w_s random at the start, not zero;
- edge-family dropout in training (zg2's fd25 rule, zg2.family_keep): each question drops each family it has with
  probability 0.25, keeping its structural family (or its lowest) when all would go. The draws come from the module's
  own generator, so the global generator moves only as the encoder's dropout moves it.
zg6t, the twin, is zg6 with no edges reaching its layers (every mean and att 0): the same parameters, context and
readout, a row-wise model.

Both train with rmatch.py's train exactly as zspk did (lean_gpu's listwise loss, 8 epochs, p@swa, seed 0) on P1c3's
carves (P1c2's pools and roots, the KT input beside), and read the six s1eval carves. No new graph, column, encoder or
text. Index sums only (index_add_), so the deterministic settings hold on the card.

    python outputs/mp_unified/zg6.py train --split L-musique --name scr-g6-zg6 --arm zg6 --device cuda --host
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zg6.py read --name scr-g6-zg6 --device cuda --host --cache-root outputs/p1c2/cache
        --rows-cap 65536
    python outputs/mp_unified/zg6.py compare --new outputs/screen/fits/scr-g6-zg6
        --base outputs/screen/fits/scr-p1c3-zspk,outputs/screen/fits/scr-g6-zg6t,outputs/step1/fits/L-musique
        --out outputs/screen/scr-g6-zg6
    python outputs/mp_unified/zg6.py pair --arm zg6 --screens S1.json,S2.json --out outputs/screen/scr-g6-zg6-pair
    python outputs/mp_unified/zg6.py recall --arm zg6 --null-kind N2 --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zg6.py verdict
    python outputs/mp_unified/zg6.py --selftest
"""
import os
import sys

_NP_BEFORE = "numpy" in sys.modules        # the selftest checks the threads were set before numpy came in
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")         # rx sets these per job; elsewhere two BLAS threads
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

import zg2 as G2  # noqa: E402  (G2's layer, family dropout and screen helpers; through it zp1c3, zp1c2, zprop, rmatch)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z3, Z2 = G2.Z3, G2.Z2
ZP, ZL, ZM, RM = G2.ZP, G2.ZL, G2.ZM, G2.RM
Z, R, S2, S, LG, LC, SR = G2.Z, G2.R, G2.S2, G2.S, G2.LG, G2.LC, G2.SR
S3, SP = ZM.S3, Z.SP
log = S.log

DECLARED = "docs/G6_FREE_GNN.md"
OUT = ROOT / "outputs" / "g6"
SCREEN = ROOT / "outputs" / "screen"
KT_KEY, NKT, LK_KEY, CH_KEY, FAMS, ZCLIP = Z3.KT_KEY, Z3.NKT, ZP.LK_KEY, RM.CH_KEY, ZP.FAMS, ZP.ZCLIP
NB = NKT + 2                       # the boundary fed at every layer: the eight KT columns and the two chain features
D, LAYERS, FD, LCLIP = G2.D, G2.LAYERS, G2.FD, G2.LCLIP            # 64, 3, 0.25, 8.0: G2's
SEED_OFF = 3601                    # G6's own draws (KT 3301, zsp 2301, G2 3501 and 3502, rmatch 1201)
SPLITS = Z3.SPLITS
BASE = "zspk"
ARM_NAMES = ("zg6", "zg6t")
FIT = {"zg6": {"L-musique": "scr-g6-zg6", "L-hotpotqa": "scr-g6-zg6-hp"},
       "zg6t": {"L-musique": "scr-g6-zg6t", "L-hotpotqa": "scr-g6-zg6t-hp"}}
BASE_FITS = G2.BASE_FITS                                                   # scr-p1c3-zspk, scr-p1c3-zspk-hp
BASE_SCREENS = {sp: f"outputs/l1/base-zspk-{sp}.json" for sp in SPLITS}  # zspk's card fits against step 1's
ZRCKP = {"L-musique": "scr-p1c3p-zrckp", "L-hotpotqa": "scr-p1c3p-zrckp-hp"}
N2_NULL = tuple(f"outputs/screen/scr-n2-zspk-{s}.json" for s in ("s1", "s2", "s1-hp", "s2-hp"))
OCT7_NULL = tuple(f"outputs/screen/scr-null-{s}.json" for s in ("s1", "s2", "s1-hp", "s2-hp"))
NULLS = {"N2": "N2's fresh zspk null: zspk's card fit trained again with seeds 1 and 2, each against its seed-0 card fit "
               "(deciding)",
         "oct7": "the 7 October null: step 1's seeds 1 and 2 against seed 0 (beside, deciding nothing)"}
CARD = {"config": LG.CONFIG, "hidden": LG.HIDDEN, "variants": "p"}
FIXED_TRAIN = ("--seed", "--hidden", "--config", "--variants", "--train", "--basis", "--repeat", "--no-verify")
FIXED_READ = ("--read", "--basis", "--variants", "--no-verify")
G6_IN = ("g6_in_w", "g6_in_b")
G6_CTX = ("g6_ctx_w",)
G6_LAYER = tuple(tuple(f"g6_{k}_{p}" for p in ("msg", "att", "fb", "self", "mean", "agg", "x", "b"))
                 for k in range(LAYERS))
G6_OUT = ("g6_out_w", "g6_skip_w", "g6_out_b")
G6_NAMES = G6_IN + G6_CTX + tuple(n for lay in G6_LAYER for n in lay) + G6_OUT
TWIN_ZERO = G6_CTX + tuple(lay[i] for lay in G6_LAYER for i in (0, 1, 2, 4, 5))   # msg, att, fb, mean, agg
DROPPED = ("out.weight", "out.bias", "base_w", "cm_gate")       # zrm's output layer, rrf skip and chain gate: deleted
AGAINST = "zspk's P1c3 card fit of each split (scr-p1c3-zspk, scr-p1c3-zspk-hp), on the same pools and KT input"
BESIDE = {"zg6": "zg6t (the twin, no edges), zrckp (the MLP), G5's zspx and G2's zg2 from their filed reads, and zg6's "
                 "re-call under the 7 October null",
          "zg6t": "zg6, the free GNN it is the twin of"}
WHAT = {"zg6": "the free GNN", "zg6t": "the free GNN's twin, no edges"}
RULE = ("ADOPT_FOR_FULL_RUN when zg6's re-call under N2's fresh zspk seed null has no LOSS on any of its twelve reads "
        "and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when that re-call is missing, not G6's, or not twelve "
        "reads (docs/G6_FREE_GNN.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zg6 fitted again on P1g-b's pools, then its full run on all six splits beside zspk's, "
                              "then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' timing) measured "
                              "again; each its own declared file",
        "GAIN_AND_LOSS": "G6b, a per-question back-off: zg6's score mixed with its twin's through a learned gate on the "
                         "question's edge-family shares, at equal weight at the start; one declared file",
        "NO_GAIN": "G6r, relation-aware messages from text: the relation embeddings "
                   "(outputs/m3b/relations/metaqa_rel_embeddings.npy, webqsp_rel_embeddings.npy) through one shared "
                   "low-rank map in place of the per-family maps on the KB datasets, the encoder started from zspk's "
                   "trained encoder, still unbounded with its own readout; one declared file. No verdict here closes "
                   "the free GNN",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}
PAIR = {a: SCREEN / f"scr-g6-{a}-pair" for a in ARM_NAMES}
RECALL = {"zg6": SCREEN / "scr-g6-zg6-pair-recall", "zg6t": SCREEN / "scr-g6-zg6t-pair-recall",
          "oct7": SCREEN / "scr-g6-zg6-pair-recall-oct7"}
VERDICT = OUT / "screen-verdict"
SHARE_MIN = 0.002                  # the MLP share is shown where zg6's R@5 gain over rrf exceeds this


def n_params(H):
    """G6's own parameters (stack, context and readout) over an encoder of width H."""
    per = FAMS * D * D + D * D + FAMS + D * D + D * FAMS * D + D * D + D * NB + D
    return D * (H + NB) + D + D + LAYERS * per + 2 * D + 1


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the model ────────────────────────────────────────────────────────────────


def edges_of(lk, nq, B, u01, dev):
    """The edges u -> v of family f that reach the layers: none for the twin (lk None); else lk's, less the families
    the dropout drops when u01 (B x FAMS uniform draws, training only) is given (zg2.family_keep's rule)."""
    if lk is None:
        u = v = f = torch.zeros(0, dtype=torch.long, device=dev)
        return u, v, f
    u, v, f = lk["u"], lk["v"], lk["f"]
    if u01 is not None:
        keep = G2.family_keep(f, nq[v], B, u01)
        u, v, f = u[keep], v[keep], f[keep]
    return u, v, f


def g6_layers(m, h, xb, w, u, v, f, nq, B, names=G6_LAYER):
    """LAYERS of zg2's layer (zg2.g2_stack's loop, op for op) from the states h over the edges (u, v, f): the boundary
    xb enters each layer, and the question's context is its rows' current states weighted by w (N x 1, summing to 1 over
    each question's rows)."""
    N = h.shape[0]
    dev, dt = h.device, h.dtype
    E = u.numel()
    at = v * FAMS + f
    one = torch.ones(E, dtype=dt, device=dev)
    deg = torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, one).view(N, FAMS, 1).clamp_min(1.0)
    qv = nq[v]
    for wm, wa, fb, ws, wmean, wagg, wx, b in ((getattr(m, n) for n in lay) for lay in names):
        c = torch.zeros(B, D, dtype=dt, device=dev).index_add_(0, nq, w * h)
        hm = Fn.linear(h, wm).view(N, FAMS, D)
        msg = hm[u, f]
        lg = ((Fn.linear(h, wa)[u] * c[qv]).sum(1) / math.sqrt(D) + fb[f]).clamp(-LCLIP, LCLIP)
        mean = torch.zeros(N * FAMS, D, dtype=dt, device=dev).index_add_(
            0, at, torch.sigmoid(lg).unsqueeze(1) * msg).view(N, FAMS, D) / deg
        ex = torch.exp(lg)
        num = torch.zeros(N, D, dtype=dt, device=dev).index_add_(0, v, ex.unsqueeze(1) * msg)
        den = torch.zeros(N, dtype=dt, device=dev).index_add_(0, v, ex)
        att = num / den.clamp_min(1e-30).unsqueeze(1)
        pre = Fn.linear(h, ws) + Fn.linear(mean.reshape(N, FAMS * D), wmean) + Fn.linear(att, wagg) + \
            Fn.linear(xb, wx) + b
        h = h + Fn.gelu(Fn.layer_norm(pre, (D,)))
    return h


def g6_stack(m, H, xb, lk, nq, B, u01=None):
    """zg6's score (N,) from the encoder's hidden states H and the boundary xb over the edges lk (None: the twin);
    u01: the family dropout's draws (training only)."""
    xb = xb.to(H.dtype)
    h0 = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b))
    s0 = Fn.linear(h0, m.g6_ctx_w).squeeze(-1)
    z0 = LG.seg_zscore8D(s0.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
    w = G2.seg_softmax_rows(z0, nq, B).unsqueeze(1)
    u, v, f = edges_of(lk, nq, B, u01, H.device)
    h = g6_layers(m, h0, xb, w, u, v, f, nq, B)
    return (Fn.linear(h, m.g6_out_w) + Fn.linear(h0, m.g6_skip_w) + m.g6_out_b).squeeze(-1)


class G6Free(ZM.ZRM):
    """zg6: zrm's block encoder (its output layer, rrf skip and chain gate deleted) under G6's stack and readout."""
    TWIN = False

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        H = self.l2.out_features
        del self.out, self.base_w, self.cm_gate
        g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        uni = lambda *shape, fan: (torch.rand(*shape, generator=g) * 2 - 1) / fan ** 0.5  # noqa: E731
        self.g6_in_w = nn.Parameter(uni(D, H + NB, fan=H + NB))
        self.g6_in_b = nn.Parameter(torch.zeros(D))
        self.g6_ctx_w = nn.Parameter(uni(1, D, fan=D))
        for nm, na, nf, ns, nmean, nagg, nx, nb in G6_LAYER:
            setattr(self, nm, nn.Parameter(uni(FAMS * D, D, fan=D)))
            setattr(self, na, nn.Parameter(uni(D, D, fan=D)))
            setattr(self, nf, nn.Parameter(torch.zeros(FAMS)))
            setattr(self, ns, nn.Parameter(uni(D, D, fan=D)))
            setattr(self, nmean, nn.Parameter(uni(D, FAMS * D, fan=FAMS * D)))
            setattr(self, nagg, nn.Parameter(uni(D, D, fan=D)))
            setattr(self, nx, nn.Parameter(uni(D, NB, fan=NB)))
            setattr(self, nb, nn.Parameter(torch.zeros(D)))
        self.g6_out_w = nn.Parameter(uni(1, D, fan=D))
        self.g6_skip_w = nn.Parameter(uni(1, D, fan=D))
        self.g6_out_b = nn.Parameter(torch.zeros(1))
        self.g6_gen = torch.Generator().manual_seed(int(seed) + SEED_OFF + 1)

    def encode(self, feats, keep, nq, B):
        """lean_screen3.ZRet.forward up to its output layer's input, line for line: the hidden state H (N x hidden)."""
        ref = S3.retrieved(feats)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S3.S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        return self.drop(Fn.gelu(self.l2(h)))

    def chain(self, feats, keep, nq):
        """(N, 2): rmatch's chain features fm and fr of each row, masked as rmatch masks them; 0 without chains."""
        N = nq.numel()
        if CH_KEY not in feats:
            return torch.zeros(N, 2, dtype=self.l1.weight.dtype, device=nq.device)
        fm, fr = self.chain_feats(feats, N)
        return torch.stack([fm, fr], 1) * keep[nq, self.j_semb].unsqueeze(1)

    def forward(self, feats, keep, nq, B, base_z):
        """The score; base_z (rrf's z-score, the MLP's base) is never read."""
        if KT_KEY not in feats:
            raise RuntimeError("zg6: no KT input in the batch (P1c3's carve adds it)")
        if not self.TWIN and LK_KEY not in feats:
            raise RuntimeError("zg6: no pool edges in the batch (zlink's carve adds them)")
        H = self.encode(feats, keep, nq, B)
        xb = torch.cat([feats[KT_KEY].to(H.dtype), self.chain(feats, keep, nq).to(H.dtype)], 1)
        lk = None if self.TWIN else feats[LK_KEY]
        u01 = torch.rand(B, FAMS, generator=self.g6_gen) if self.training and not self.TWIN else None
        return g6_stack(self, H, xb, lk, nq, B, u01)


class ZG6(G6Free):
    """zg6: the free GNN over the pool's edges."""


class ZG6T(G6Free):
    """zg6t: zg6 with no edges reaching its layers (the twin)."""
    TWIN = True


ARMS = {"zg6": (ZG6, Z3.KTLinkCarveBase), "zg6t": (ZG6T, Z3.KTLinkCarveBase)}
S.ARMS.update(ARMS)
SETTINGS = {"encoder": "zrm's block encoder (lean_screen3.ZRet's forward up to its output layer's input), hidden 128, "
                       "its own weights from the seed; zrm's output layer, rrf skip and chain gate deleted",
            "mlp_score_read": False, "state": D, "layers": LAYERS, "boundary": [*Z3.KT_COLS, "chain_fm", "chain_fr"],
            "context": "the question's rows' current states, weighted by the softmax of w0 . h0 z-scored over the "
                       "question's rows and clipped to +-8",
            "aggregations": ["gated mean per family", "attention over all"], "logit_clip": LCLIP,
            "readout": "w_o . h_L + w_s . h0 + b, unbounded, w_o and w_s random at the start",
            "family_dropout": FD, "seed_offset": SEED_OFF, "edges": ZP.SETTINGS["edges"],
            "stack_parameters_at_hidden_128": n_params(128)}


# ── train, read, compare ─────────────────────────────────────────────────────


def fixed_flags(argv, fixed):
    """The flags of argv (or argparse abbreviations of them) that would change a setting G6 keeps at zspk's."""
    out = []
    for t in argv:
        k = t.split("=")[0]
        if k.startswith("--") and len(k) > 2 and any(f.startswith(k) for f in fixed):
            out.append(t)
    return out


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        g = Z2.gate_passed()
        rec.update({"zg6_sha256": LC.sha_src(__file__), "zg2_sha256": LC.sha_src(G2.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "g6": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE, "device": Z2.DEVICE, "graph": "today's",
                        "pools": f"P1e's {g.get('construction')} at each question's frozen size (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "model": SETTINGS, "twin": ARMS[arm][0].TWIN, "mlp_score_read": False, "readout": "unbounded",
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zg6: train takes --arm zg6 or zg6t, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zg6: G6's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zg6: the {arm} fit of {split} is {FIT[arm][split]}, not {name}")
    bad = fixed_flags(argv, FIXED_TRAIN)
    if bad:
        raise SystemExit(f"zg6: G6 trains on zspk's card settings (seed {LG.SEED}, hidden {LG.HIDDEN}, config "
                         f"{LG.CONFIG}, variant p, the split's fit carves and basis); {bad} would change them")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm = rec.get("arm")
    if arm not in ARMS or "g6" not in rec:
        raise SystemExit(f"zg6: {name} is not a G6 fit (arm {arm})")
    if name != FIT[arm].get(rec.get("split")):
        raise SystemExit(f"zg6: {name} is not the {arm} fit of {rec.get('split')}")
    bad = fixed_flags(argv, FIXED_READ)
    if bad:
        raise SystemExit(f"zg6: G6 reads p's fits on the six s1eval carves; {bad} would change that")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call against zspk's P1c3 card fits, under N2's null or the 7 October null ─────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"zg6: no arm {arm}")

    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"zg6: no base fit of {BASE} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def n2_null(files):
    """N2's fresh zspk null in screen_recall.load_null's form ({split: {dataset: read}}, sources): zspk's card fit
    trained again with seeds 1 and 2 on L-musique and L-hotpotqa, each compared with its seed-0 card fit (scr-p1c3-zspk,
    -hp), the comparisons the deciding base. seed0 is zspk's seed-0 R@5, the base of every G6 pair row; the floor is
    screen_recall's, max(0.0075, 2 sqrt((D1^2 + D2^2) / 2))."""
    got = {}
    for fn in files:
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zg6: {p} missing (N2's null)")
        rec = json.loads(p.read_text(encoding="utf-8"))
        b0 = SR.parts_of(rec["bases"][0])[-3:]
        split = next((sp for sp, f in BASE_FITS.items() if f == b0), None)
        if split is None:
            raise SystemExit(f"zg6: {p} is decided against {rec['bases'][0]}, not zspk's card fit of "
                             f"{' or '.join(BASE_FITS)} (N2's null)")
        if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
            raise SystemExit(f"zg6: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
        if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
            raise SystemExit(f"zg6: {p} trained on {rec['trained_on']}, not {split}'s {sorted(LG.FITS[split])}")
        s = SR.fit_record(rec["new"])
        if s.get("arm") != BASE or s.get("split") != split or s.get("seed") not in SR.SEEDS:
            raise SystemExit(f"zg6: {p}'s fit is arm {s.get('arm')} with seed {s.get('seed')} on {s.get('split')}, "
                             f"not {BASE} with seed 1 or 2 on {split}")
        off = {k: s[k] for k, want in CARD.items() if k in s and s[k] != want}
        if off:
            raise SystemExit(f"zg6: {p}'s fit is not on zspk's card settings: {off}")
        if (split, s["seed"]) in got:
            raise SystemExit(f"zg6: {split} with seed {s['seed']} appears twice in N2's null")
        rows = rec["by_base"][rec["bases"][0]]["rows"]
        if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zg6: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
        got[(split, s["seed"])] = {"file": str(p), "rows": {r["dataset"]: r for r in rows}}
    want = [(sp, sd) for sp in SR.NULL_SPLITS for sd in SR.SEEDS]
    if sorted(got) != sorted(want):
        raise SystemExit(f"zg6: N2's null holds {sorted(got)}, not {sorted(want)}")
    null = {}
    for split in SR.NULL_SPLITS:
        null[split] = {}
        for ds in LG.EVAL_ORDER:
            r1, r2 = got[(split, 1)]["rows"][ds], got[(split, 2)]["rows"][ds]
            if r1["base"][0] != r2["base"][0]:
                raise SystemExit(f"zg6: {split} {ds}: N2's two seeds are compared with different seed-0 R@5 "
                                 f"({r1['base'][0]}, {r2['base'][0]})")
            d1, d2 = r1["delta"][0], r2["delta"][0]
            null[split][ds] = {"read": r1["read"], "seed0": r1["base"][0], "d1": d1, "d2": d2,
                               "spread": math.sqrt((d1 * d1 + d2 * d2) / 2.0), "floor": SR.floor_of(d1, d2)}
    return null, {f"N2 {sp} seed {sd}": got[(sp, sd)]["file"] for sp, sd in want}


@contextlib.contextmanager
def against_n2():
    """relz.against_rel's swap with N2's null in place of rel's: screen_pair's load decided against the base fits
    (relz.pair_load under on_base) and screen_recall's null N2's; restored on the way out."""
    saved = SP.load, SR.load_null
    SP.load, SR.load_null = Z.pair_load, n2_null
    try:
        yield
    finally:
        SP.load, SR.load_null = saved


def fix(arm, null=None):
    sub = []
    if null == "N2":
        sub.append(("step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the tenth round)",
                    f"zspk's seeds 1 and 2 against its seed-0 card fit, N2's fresh zspk null) (docs/SCREENS.md, "
                    f"section 2, and {DECLARED})"))
    return tuple(sub) + (("(rel's screen fits scr-rel and scr-rel-hp)",
                          "(zspk's P1c3 card fits scr-p1c3-zspk and scr-p1c3-zspk-hp)"),
                         ("(rel's screen fits)", "(zspk's P1c3 card fits)"),
                         ("| rel R@5 | relz R@5 |", f"| zspk R@5 | {arm} R@5 (G6) |"),
                         ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm, null=None):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm, null):
            t = t.replace(a, b)
        head = f"G6 ({DECLARED}): {arm} ({WHAT[arm]}) on KIND_T's pools against {BASE}"
        head += f", under {NULLS[null]}.\n\n" if null else ".\n\n"
        md.write_text(head + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "G6", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "beside": BESIDE[arm], "zg6_sha256": LC.sha_src(__file__)})
        if null:
            rec.update({"null_kind": null, "null_is": NULLS[null], "deciding": arm == "zg6" and null == "N2"})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, null, base_screens=None):
    """relz's re-call of arm's pair against zspk's card fits. null 'N2': N2's fresh zspk null decides the floors
    (zspk's seeds against its card fit, no re-basing); 'oct7': the 7 October null re-based to zspk's card fits by
    relz.rel_null over zspk's base screens (outputs/l1/base-zspk-*), beside only."""
    if null not in NULLS:
        raise SystemExit(f"zg6: the null is N2 or oct7, not {null}")
    p = Path(pair_file)
    pr = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if pr.get("stage") != "G6" or pr.get("arm") != arm:
        raise SystemExit(f"zg6: {p} is not G6's pair of {arm} ({pr.get('stage')}, {pr.get('arm')})")
    with on_base(arm):
        if null == "N2":
            saved = Z.against_rel
            Z.against_rel = lambda rel_screens=None: against_n2()
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
    if (rec.get("stage") != "G6" or rec.get("declared_in") != DECLARED or rec.get("arm") != arm
            or rec.get("base_arm") != BASE or rec.get("null_kind") != null):
        problems.append(f"{rp.name} is not G6's re-call of {arm} under the {null} null")
    rows = rec.get("rows", [])
    got = sorted((r.get("split"), r.get("dataset")) for r in rows)
    if got != sorted((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
        problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six datasets")
    return rows


def calls_of(rows):
    return {"gains": [tag(r) for r in rows if r.get("call") == "GAIN"],
            "losses": [tag(r) for r in rows if r.get("call") == "LOSS"]}


def filed_r5(root=ROOT):
    """R@5 per read (split, dataset) of the models beside G6, from their filed files: zrckp (the MLP, its P1c3p card
    fits through outputs/l1/base-zrckp-*), zspx (G5's re-call) and zg2 (G2's re-call). A file absent, or not the
    expected one, gives nothing."""
    root, out = Path(root), {}
    for sp in SPLITS:
        p = root / "outputs" / "l1" / f"base-zrckp-{sp}.json"
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8"))
            if SR.parts_of(c.get("new", ""))[-1:] == (ZRCKP[sp],):
                for r in c["by_base"][c["bases"][0]]["rows"]:
                    out.setdefault("zrckp", {})[(sp, r["dataset"])] = r["new"][0]
    for k, f, stage in (("zspx", "scr-g5-zspx-pair-recall.json", "G5"), ("zg2", "scr-g2-zg2-pair-recall.json", "G2")):
        p = root / "outputs" / "screen" / f
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec.get("stage") == stage and rec.get("arm") == k:
                for r in rec.get("rows", []):
                    out.setdefault(k, {})[(r["split"], r["dataset"])] = r["new"]
    return out


def mlp_share(mlp, arm, rrf):
    """(zrckp - rrf) / (zg6 - rrf) on R@5; None where a number is missing or zg6's gain over rrf is SHARE_MIN or less."""
    if mlp is None or arm is None or rrf is None or arm - rrf <= SHARE_MIN:
        return None
    return round((mlp - rrf) / (arm - rrf), 4)


def fmt(x, signed=False):
    if not isinstance(x, (int, float)):
        return "n/a"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def verdict(recalls=None, compares=None, out=None, root=ROOT):
    """G6's screen verdict from zg6's re-call under N2's null (section 4) and the next form (section 5). Beside,
    deciding nothing: the twin's re-call, zg6's re-call under the 7 October null, zg6 minus zg6t and zg6 minus zspk on
    R@5 / FC@5 / hit@1 per read, the R@5 of zrckp, zspx and zg2 from their filed reads, and the MLP share."""
    recalls = recalls or {k: p.with_suffix(".json") for k, p in RECALL.items()}
    compares = compares or {sp: SCREEN / f"{FIT['zg6'][sp]}.json" for sp in SPLITS}
    problems, tp, op = [], [], []
    rows = recall_rows(recalls["zg6"], "zg6", "N2", problems)
    trows = recall_rows(recalls["zg6t"], "zg6t", "N2", tp)
    orows = recall_rows(recalls["oct7"], "zg6", "oct7", op)
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    twin = None if tp else calls_of(trows)
    oct7 = None if op else calls_of(orows)
    mp = {sp: G2.beside_rows(compares[sp], FIT["zg6t"][sp]) for sp in SPLITS}
    vs = {sp: G2.beside_rows(compares[sp], BASE_FITS[sp][2]) for sp in SPLITS}
    filed = filed_r5(root)
    tnew = {} if tp else {(r.get("split"), r.get("dataset")): r.get("new") for r in trows}
    r5, share = {}, {}
    for r in rows:
        key = (r.get("split"), r.get("dataset"))
        mlp = filed.get("zrckp", {}).get(key)
        r5[tag(r)] = {"read": r.get("read"), "rrf": r.get("rrf"), "zspk": r.get("base"), "zg6": r.get("new"),
                      "zg6t": tnew.get(key), "zrckp": mlp, "zspx": filed.get("zspx", {}).get(key),
                      "zg2": filed.get("zg2", {}).get(key), "call": r.get("call")}
        share[tag(r)] = mlp_share(mlp, r.get("new"), r.get("rrf"))
    rp = Path(recalls["zg6"])
    rec = {"stage": "G6", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": "zg6", "base_arm": BASE,
           "null_kind": "N2", "next_form": nxt, "gains": gains, "losses": losses, "problems": problems,
           "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
           "recall": rp.relative_to(ROOT).as_posix() if rp.is_relative_to(ROOT) else rp.name,
           "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
           "beside": {"twin_recall": twin, "twin_problems": tp, "oct7_recall": oct7, "oct7_problems": op,
                      "zg6_minus_zg6t": mp, "zg6_minus_zspk": vs, "r5": r5, "mlp_share": share},
           "zg6_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def calls_md(c, p):
        if c is None:
            return "not available (" + "; ".join(p) + ")."
        return (f"{len(c['gains'])} GAIN ({', '.join(c['gains']) or 'none'}), {len(c['losses'])} LOSS "
                f"({', '.join(c['losses']) or 'none'}).")

    md = [f"# G6's screen verdict: zg6 **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zg6** (the free GNN: its own encoder, {LAYERS} layers of {D}-wide hidden-state message passing, an "
          f"unbounded readout, no MLP score) against {AGAINST}, after the re-call under N2's fresh zspk null: "
          f"{len(gains)} GAIN ({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "Beside, deciding nothing.", "",
          "- The twin (zg6t, no edges) against zspk under N2's null: " + calls_md(twin, tp),
          "- zg6 against zspk under the 7 October null (step 1's seeds): " + calls_md(oct7, op), "",
          f"R@5 per read; the MLP share = (zrckp - rrf) / (zg6 - rrf) where zg6's gain over rrf exceeds {SHARE_MIN}:", "",
          "| read | rrf | zspk | zg6 | zg6t | zrckp (MLP) | zspx (G5) | zg2 (G2) | MLP share | call (N2) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for k, t in r5.items():
        md.append(f"| {k} ({t['read']}) | {fmt(t['rrf'])} | {fmt(t['zspk'])} | {fmt(t['zg6'])} | {fmt(t['zg6t'])} | "
                  f"{fmt(t['zrckp'])} | {fmt(t['zspx'])} | {fmt(t['zg2'])} | {fmt(share[k])} | {t['call']} |")
    for title, rs, base in (("zg6 minus zspk", vs, "scr-p1c3-zspk"), ("zg6 minus zg6t (what the edges add)", mp,
                                                                       "scr-g6-zg6t")):
        md += ["", f"{title}: R@5 with its 95% question-bootstrap interval, FC@5 and hit@1:", "",
               "| split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075 floor) |",
               "|---|---|---|---|---|---|---|"]
        for sp in SPLITS:
            if rs[sp] is None:
                md.append(f"| {sp} | (no comparison with {base}) | | | | | |")
                continue
            for r in rs[sp]:
                d, lo, hi = r["delta"], r["lo"], r["hi"]
                md.append(f"| {sp} | {r['dataset']} | {r['read']} | {d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | "
                          f"{d[1]:+.4f} | {d[2]:+.4f} | {r['call']} |")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zg6 verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''}); "
        f"next: {nxt}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, who, calls, null="N2", **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "base": 0.6, "new": 0.65, "rrf": 0.4, "delta": 0.05,
             "call": calls.get((sp, ds), "WITHIN")} for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "G6", "declared_in": DECLARED, "arm": who, "base_arm": BASE, "null_kind": null, "verdict": "x",
           "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def fake_n2(td, big=None, arm=BASE, base_r5=0.6, **fit_over):
    """N2's four comparisons in lean_screen's compare form: zspk with seeds 1 and 2 against its card fit of the split;
    big = {(split, dataset): (D1, D2)}, every other read +-0.002; fit_over replaces fields of the fits' screen.json."""
    big = big or {}
    out = []
    for split, suf in (("L-musique", ""), ("L-hotpotqa", "-hp")):
        for seed in SR.SEEDS:
            dd = {ds: big.get((split, ds), (0.002, -0.002))[seed - 1] for ds in LG.EVAL_ORDER}
            name = f"scr-n2-zspk-s{seed}{suf}"
            p = SR.fake_compare(td, name, split, dd, arm=arm, seed=seed, base_r5=base_r5,
                                base="/".join(("outputs",) + BASE_FITS[split]))
            sj = Path(td) / "fits" / name / "screen.json"
            rec = json.loads(sj.read_text(encoding="utf-8"))
            rec.update(CARD)
            rec.update(fit_over)
            sj.write_text(json.dumps(rec), encoding="utf-8")
            out.append(p)
    return out


def by_hand(m, H, xb, lk, nq, B, twin, keep_e=None):
    """g6_stack with python loops over the questions and the edges (the selftest's reference)."""
    N = H.shape[0]
    h0 = Fn.gelu(torch.cat([H, xb], 1) @ m.g6_in_w.T + m.g6_in_b)
    s0 = (h0 @ m.g6_ctx_w.T).squeeze(-1)
    qs = nq.tolist()
    rows_of = [[i for i in range(N) if qs[i] == q] for q in range(B)]
    z0 = torch.zeros(N)
    for rows in rows_of:
        if rows:
            x = s0[rows]
            mu = x.mean()
            sd = ((x - mu) ** 2).mean().sqrt()
            z0[rows] = (x - mu) / sd if float(sd) >= 1e-6 else torch.zeros(len(rows))
    z0 = z0.clamp(-ZCLIP, ZCLIP)
    edges = [] if twin else list(zip(lk["u"].tolist(), lk["v"].tolist(), lk["f"].tolist()))
    if keep_e is not None and not twin:
        edges = [e for e, k in zip(edges, keep_e.tolist()) if k]
    h = h0
    for lay in G6_LAYER:
        wm, wa, fb, ws, wmean, wagg, wx, b = (getattr(m, n) for n in lay)
        c = torch.zeros(B, D)
        for q, rows in enumerate(rows_of):
            if rows:
                c[q] = (torch.softmax(z0[rows], 0).unsqueeze(1) * h[rows]).sum(0)
        mean = torch.zeros(N, FAMS, D)
        cnt = torch.zeros(N, FAMS)
        num = torch.zeros(N, D)
        den = torch.zeros(N)
        for u_, v_, f_ in edges:
            msg = wm[f_ * D:(f_ + 1) * D] @ h[u_]
            lg = float(((wa @ h[u_]) * c[qs[v_]]).sum() / math.sqrt(D) + fb[f_])
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


def grads(m, feats, keep, nq, B, bz, w):
    """Every parameter's gradient of sum(w * m's score); None where the forward does not reach it."""
    m.zero_grad(set_to_none=True)
    (m(feats, keep, nq, B, bz) * w).sum().backward()
    return {k: (p.grad.detach().clone() if p.grad is not None else None) for k, p in m.named_parameters()}


def nonzero(g):
    return g is not None and float(g.abs().sum()) > 0


def snapshot():
    """Every file under G6's roots (outputs/g6; outputs/screen's scr-g6-* screens and fits) and the beside reads'
    root (outputs/l1), with size and mtime; and whether outputs/g6 exists."""
    out = {}
    for base, pat in ((OUT, "**/*"), (SCREEN, "scr-g6-*"), (SCREEN / "fits", "scr-g6-*/**/*"),
                      (ROOT / "outputs" / "l1", "**/*")):
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
    # 1. the arms on P1c3's link carve; no MLP class in their MRO (no KT head, no zsp step, no G2 base); earlier arms
    #    unchanged; G2's layer constants; the threads set before numpy came in
    assert S.ARMS["zg6"] == (ZG6, Z3.KTLinkCarveBase) and S.ARMS["zg6t"] == (ZG6T, Z3.KTLinkCarveBase)
    assert ZG6.__mro__[1:3] == (G6Free, ZM.ZRM) and ZG6T.TWIN and not ZG6.TWIN
    assert not {Z3.KTMix, ZP.ZProp, Z3.ZSPK, G2.G2Mix} & (set(ZG6.__mro__) | set(ZG6T.__mro__))
    assert S.ARMS["zspk"] == Z3.ARMS["zspk"] and S.ARMS["zrck"] == Z3.ARMS["zrck"]
    assert S.ARMS["zg2"] == G2.ARMS["zg2"] and S.ARMS["zsp"] == Z2.ARMS["zsp"] and S.ARMS["zrc"] == Z2.ARMS["zrc"]
    assert (D, LAYERS, FD, LCLIP, NB, FAMS) == (64, 3, 0.25, 8.0, 10, 3) and n_params(128) == 121802
    assert SEED_OFF not in (Z3.KT_SEED_OFF, G2.SEED_OFF, G2.SEED_OFF + 1, getattr(ZP, "SEED_OFF", None), 1201)
    assert BASE_FITS == {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                         "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}
    assert S3.S is S and CARD == {"config": "2e-3:1e-4:0.1:8:2", "hidden": 128, "variants": "p"}
    assert all(os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"))
    if __name__ == "__main__":
        assert not _NP_BEFORE, "numpy came in before the threads were set"
    tmp = Path(tempfile.mkdtemp(prefix="zg6_"))
    try:
        rng = np.random.default_rng(31)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        plain = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
        cls = Z3.kt_carve(plain, tmp / "kt")
        c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:7], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, _gold = c.batch(qs, blocks)
        N = nq.numel()
        lk = feats[LK_KEY]
        assert lk["u"].numel() > 0 and KT_KEY in feats and CH_KEY in feats and feats[CH_KEY]["eq"].size > 0
        keep = torch.ones(B, len(blocks))
        # 2. from seed 0: zg6's and zg6t's construction moves the global generator as zrm's does; their state is zrm's
        #    less its output layer, rrf skip and chain gate, plus G6's parameters (zrm's own draws where shared), no
        #    KT head, zsp step or G2 stack; the readout is random, not zero; the two arms' states are the same
        nets, after = {}, {}
        for k, cl in (("zrm", ZM.ZRM), ("zg6", ZG6), ("zg6t", ZG6T)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after["zg6"], after["zrm"]) and torch.equal(after["zg6t"], after["zrm"])
        sz = nets["zrm"].state_dict()
        for k in ARM_NAMES:
            sn = nets[k].state_dict()
            assert sorted(sn) == sorted((set(sz) - set(DROPPED)) | set(G6_NAMES)), sorted(set(sn) ^ set(sz))
            assert all(torch.equal(sz[x], sn[x]) for x in sn if x in sz)
            assert not any(x.startswith(("sp_", "kt_", "g2_")) for x in sn)
            assert not any(hasattr(nets[k], a) for a in ("out", "base_w", "cm_gate"))
            assert sum(int(sn[x].numel()) for x in G6_NAMES) == n_params(16)
            assert nonzero(sn["g6_out_w"]) and nonzero(sn["g6_skip_w"]) and nonzero(sn["g6_ctx_w"])
        assert not LG.same_state(nets["zg6"].state_dict(), nets["zg6t"].state_dict())
        assert torch.equal(nets["zg6"].g6_gen.get_state(), nets["zg6t"].g6_gen.get_state())
        m, t, zrm = nets["zg6"], nets["zg6t"], nets["zrm"]
        with torch.no_grad():
            # 3. no MLP score is read: with every ancestor's forward made to raise and base_z NaN, the score is the
            #    same and finite
            for x in nets.values():
                x.eval()
            y0 = m(feats, keep, nq, B, bz)
            fws = {k: vars(k)["forward"] for k in G6Free.__mro__[1:] if k not in (nn.Module, object)
                   and "forward" in vars(k)}
            assert {ZM.ZRM, RM.ChainMatch, S3.ZRet} - set(fws) <= {ZM.ZRM} and RM.ChainMatch in fws

            def boom(self, *a, **kw):
                raise AssertionError(f"{type(self).__name__}: an MLP forward ran")

            try:
                for k in fws:
                    k.forward = boom
                y1 = m(feats, keep, nq, B, torch.full_like(bz, float("nan")))
                y1t = t(feats, keep, nq, B, torch.full_like(bz, float("nan")))
            finally:
                for k, fw in fws.items():
                    k.forward = fw
            assert torch.equal(y0, y1) and bool(torch.isfinite(y1).all()) and bool(torch.isfinite(y1t).all())
            # 4. the encoder is zrm's: its hidden state is the input of zrm's output layer from the same weights, in
            #    eval and in training under the same seed; the chain features are rmatch's, masked as rmatch masks
            #    them, and 0 without chains
            caught = []
            hk = zrm.out.register_forward_pre_hook(lambda mod, inp: caught.append(inp[0]))
            for mode in (False, True):
                zrm.train(mode)
                m.train(mode)
                torch.manual_seed(5)
                zrm(feats, keep, nq, B, bz)
                torch.manual_seed(5)
                assert torch.equal(caught[-1], m.encode(feats, keep, nq, B)), mode
            hk.remove()
            zrm.eval()
            m.eval()
            keep2 = keep.clone()
            keep2[1, blocks.index("SEMB")] = 0.0
            for kp in (keep, keep2):
                fm, fr = zrm.chain_feats(feats, N)
                want = torch.stack([fm, fr], 1) * kp[nq, zrm.j_semb].unsqueeze(1)
                assert torch.equal(m.chain(feats, kp, nq), want)
            assert float(m.chain(feats, keep2, nq)[nq == 1].abs().sum()) == 0.0 and nonzero(m.chain(feats, keep, nq))
            no_ch = {k: x for k, x in feats.items() if k != CH_KEY}
            assert torch.equal(m.chain(no_ch, keep, nq), torch.zeros(N, 2))
            # 5. G6's layers are G2's, op for op: on a zg2 net with G2's names they give zg2.g2_stack's readout bit for
            #    bit (with and without edges, with and without the dropout's draws)
            torch.manual_seed(0)
            g2m = G2.ZG2(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
            gen = torch.Generator().manual_seed(3)
            for n in G2.G2_NAMES:
                getattr(g2m, n).copy_(torch.randn(getattr(g2m, n).shape, generator=gen) * 0.3)
            Hr, sr, kt = torch.randn(N, 16, generator=gen), torch.randn(N, generator=gen), feats[KT_KEY]
            u01 = torch.rand(B, FAMS, generator=gen)
            for twin in (False, True):
                for uu in (None, u01):
                    ref = G2.g2_stack(g2m, Hr, sr, kt, lk, nq, B, twin, uu)
                    z = LG.seg_zscore8D(sr.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
                    xb2 = torch.cat([z.unsqueeze(1), kt], 1)
                    h0 = Fn.gelu(Fn.linear(torch.cat([Hr, xb2], 1), g2m.g2_in_w, g2m.g2_in_b))
                    w2 = G2.seg_softmax_rows(z, nq, B).unsqueeze(1)
                    u, v, f = edges_of(None if twin else lk, nq, B, uu, Hr.device)
                    hh = g6_layers(g2m, h0, xb2, w2, u, v, f, nq, B, G2.G2_LAYER)
                    assert torch.equal(Fn.linear(hh, g2m.g2_out_w, g2m.g2_out_b).squeeze(-1), ref), (twin, uu is None)
            # 6. set: each arm is the model by hand (encoder, boundary, context, three layers, readout), the twin with
            #    no edges; with edges the arm is not its twin
            for n in G6_NAMES:
                getattr(m, n).copy_(torch.randn(getattr(m, n).shape, generator=gen) * 0.3)
            t.load_state_dict(m.state_dict())
            H = m.encode(feats, keep, nq, B)
            xb = torch.cat([feats[KT_KEY], m.chain(feats, keep, nq)], 1)
            for net in (m, t):
                got = net(feats, keep, nq, B, bz)
                want = by_hand(net, H, xb, lk, nq, B, net.TWIN)
                assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (net.TWIN, float((got - want).abs().max()))
            ya, yt = m(feats, keep, nq, B, bz), t(feats, keep, nq, B, bz)
            assert not torch.allclose(ya, yt)
            # 7. the twin is the arm with the edges removed, bit for bit, in eval and in training under the same seed;
            #    it ignores the edges and runs without them; the arm without them stops
            f3 = dict(feats)
            f3[LK_KEY] = {x: y[:0] for x, y in lk.items()}
            no_lk = {k: x for k, x in feats.items() if k != LK_KEY}
            assert torch.equal(m(f3, keep, nq, B, bz), yt) and torch.equal(t(f3, keep, nq, B, bz), yt)
            assert torch.equal(t(no_lk, keep, nq, B, bz), yt)
            for x in (m, t):
                x.train()
            torch.manual_seed(7)
            ytr = t(feats, keep, nq, B, bz)
            torch.manual_seed(7)
            assert torch.equal(m(f3, keep, nq, B, bz), ytr) and not torch.equal(ytr, yt)
            for x in (m, t):
                x.eval()
            for bad in (no_lk, {k: x for k, x in feats.items() if k != KT_KEY}):
                try:
                    m(bad, keep, nq, B, bz)
                except RuntimeError:
                    continue
                raise AssertionError("zg6 ran without its edges or its KT input")
            # 8. the readout is unbounded: scaled by 100 the score is 100 times, far past G2's 0.5 bound
            saved = [getattr(m, n).clone() for n in G6_OUT]
            for n in G6_OUT:
                getattr(m, n).mul_(100.0)
            y100 = m(feats, keep, nq, B, bz)
            for n, x in zip(G6_OUT, saved):
                getattr(m, n).copy_(x)
            assert torch.allclose(y100, 100.0 * ya, rtol=1e-4, atol=1e-3) and float(y100.abs().max()) > 20 * G2.TB
            # 9. the family dropout: at read, and in the twin, no draw; in training the arm draws once (B x FAMS) from
            #    its own generator and is the model by hand over the edges zg2.family_keep keeps; the global generator
            #    moves as the twin's; no draw ever drops all of a question's families (500 draws and the all-drop draw)
            for x, mode in ((m, False), (t, True), (t, False)):
                x.train(mode)
                g0 = x.g6_gen.get_state()
                x(feats, keep, nq, B, bz)
                assert torch.equal(x.g6_gen.get_state(), g0), (x.TWIN, mode)
            m.train()
            m.drop.p = 0.0
            for _ in range(50):              # the generator's next draw that drops some edges, not all
                g0 = m.g6_gen.get_state()
                u01 = torch.rand(B, FAMS, generator=m.g6_gen)
                ke = G2.family_keep(lk["f"], nq[lk["v"]], B, u01)
                if 0 < int(ke.sum()) < ke.numel():
                    break
            assert 0 < int(ke.sum()) < ke.numel()
            g1 = m.g6_gen.get_state()
            m.g6_gen.set_state(g0)
            got = m(feats, keep, nq, B, bz)
            assert torch.equal(m.g6_gen.get_state(), g1)
            want = by_hand(m, H, xb, lk, nq, B, False, ke)
            assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), float((got - want).abs().max())
            assert not torch.allclose(got, ya)
            m.drop.p = 0.1
            t.train()
            torch.manual_seed(9)
            m(feats, keep, nq, B, bz)
            ra = torch.rand(4)
            torch.manual_seed(9)
            t(feats, keep, nq, B, bz)
            assert torch.equal(ra, torch.rand(4))
            m.eval()
            t.eval()
            fv, qv = lk["f"], nq[lk["v"]]
            has = torch.zeros(B).index_add_(0, qv, torch.ones(qv.numel())) > 0
            low = torch.full((B,), FAMS, dtype=torch.long).scatter_reduce(0, qv, fv, reduce="amin", include_self=True)
            gq = torch.Generator().manual_seed(17)
            for i in range(501):
                ud = torch.zeros(B, FAMS) if i == 0 else torch.rand(B, FAMS, generator=gq)
                ke = G2.family_keep(fv, qv, B, ud)
                kept = torch.zeros(B).index_add_(0, qv[ke], torch.ones(int(ke.sum()))) > 0
                assert torch.equal(kept, has), i
                if i == 0:
                    assert torch.equal(ke, fv == low[qv])
        # 10. gradients: in the arm every parameter learns (the encoder, the chain match, each layer, the context and
        #     the readout); in the twin the context and each layer's message, attention, family bias, mean and
        #     attention maps get exactly 0 (no edge reaches them) and every other parameter learns
        with torch.no_grad():
            m.cm_b.copy_(torch.randn(m.cm_b.shape, generator=gen) * 0.1)     # zero at the start: cm_a learns
            m.cm_d.copy_(torch.randn(m.cm_d.shape, generator=gen) * 0.1)     # through them only
            t.load_state_dict(m.state_dict())
        gw = torch.randn(N, generator=torch.Generator().manual_seed(11))
        ga = grads(m, feats, keep, nq, B, bz, gw)
        assert not [k for k, g in ga.items() if not nonzero(g)], [k for k, g in ga.items() if not nonzero(g)]
        assert any(k.startswith("cm_") for k in ga) and "l1.weight" in ga and "U" in ga and "g6_out_b" in ga
        gt = grads(t, feats, keep, nq, B, bz, gw)
        assert all(gt[k] is None or float(gt[k].abs().sum()) == 0 for k in TWIN_ZERO)
        assert not [k for k in gt if k not in TWIN_ZERO and not nonzero(gt[k])], \
            [k for k in gt if k not in TWIN_ZERO and not nonzero(gt[k])]
        # 11. toy fits through lean_gpu's loop (its listwise loss, Adam, SWA): a repeat is bit for bit; the encoder,
        #     the stack and the readout move; in the twin no family bias moves (no edge reached it)
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm in ARM_NAMES:
            with S.patched(arm):
                f1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
                f2 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} repeat")
            assert all(not LG.same_state(a, b) for a, b in zip(f1["states"] + [f1["swa"]], f2["states"] + [f2["swa"]]))
            torch.manual_seed(0)
            s0 = ARMS[arm][0](blocks, f1["widths"], 16, dropout=0.1, seed=0).state_dict()
            assert sorted(s0) == sorted(f1["swa"])
            must = ["l1.weight", "l2.weight", "U", "g6_in_w", "g6_out_w", "g6_skip_w"] + \
                [lay[i] for lay in G6_LAYER for i in (3, 6)] + ([] if arm == "zg6t" else
                                                               [lay[i] for lay in G6_LAYER for i in (0, 1, 2, 4, 5)])
            still = [k for k in must if torch.equal(s0[k], f1["swa"][k])]
            assert not still, (arm, still)
            if arm == "zg6t":
                assert all(float(f1["swa"][lay[2]].abs().sum()) == 0 for lay in G6_LAYER)
        # 12. train and read refuse what is not G6's
        cr = ["--cache-root", "outputs/p1c2/cache"]
        good = ["train", "--arm", "zg6", "--name", FIT["zg6"]["L-musique"]] + cr
        SR.must_stop(train, ["train", "--arm", "zspk", "--name", FIT["zg6"]["L-musique"]] + cr, "L-musique")
        SR.must_stop(train, ["train", "--arm", "zg2", "--name", FIT["zg6"]["L-musique"]] + cr, "L-musique")
        SR.must_stop(train, good, "L-2wiki")
        SR.must_stop(train, ["train", "--arm", "zg6", "--name", FIT["zg6t"]["L-musique"]] + cr, "L-musique")
        SR.must_stop(train, ["train", "--arm", "zg6", "--name", FIT["zg6"]["L-hotpotqa"]] + cr, "L-musique")
        SR.must_stop(train, ["train", "--arm", "zg6", "--name", FIT["zg6"]["L-musique"]], "L-musique")
        SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
        for extra in (["--seed", "1"], ["--seed=2"], ["--hidden", "64"], ["--hid", "64"], ["--config", "1e-3:0:0:4:1"],
                      ["--variants", "q"], ["--train", "metaqa=fit"], ["--basis", "musique"], ["--repeat", "2"],
                      ["--no-verify"]):
            SR.must_stop(train, good + extra + ["--device", "cuda"], "L-musique")
        assert not fixed_flags(good + ["--device", "cuda", "--host", "--threads", "4", "--out-root", "x"], FIXED_TRAIN)
        fits = tmp / "fits"
        for nm, rec in (("y", {"arm": "zspk", "p1c3": {}, "split": "L-musique"}),
                        ("y", {"arm": "zg6", "split": "L-musique"}),
                        ("y", {"arm": "zg2", "g2": {}, "split": "L-musique"}),
                        ("y", {"arm": "zg6", "g6": {}, "split": "L-musique"}),
                        ("scr-g6-zg6", {"arm": "zg6t", "g6": {}, "split": "L-musique"}),
                        ("scr-g6-zg6", {"arm": "zg6", "g6": {}, "split": "L-hotpotqa"})):
            (fits / nm).mkdir(parents=True, exist_ok=True)
            LC.write_json(fits / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(fits)] + cr)
        LC.write_json(fits / "scr-g6-zg6" / "screen.json", {"arm": "zg6", "g6": {}, "split": "L-musique"})
        for extra in (["--read", "metaqa=s1eval"], ["--basis", "musique"], ["--no-verify"], ["--variants", "q"]):
            SR.must_stop(read, ["--name", "scr-g6-zg6", "--out-root", str(fits)] + cr + extra)
        SR.must_stop(read, ["--name", "scr-g6-zg6", "--out-root", str(fits), "--cache-root", "outputs/p1gb/cache"])
        SR.must_stop(read, ["--name", "none", "--out-root", str(fits)] + cr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 13. relz's pair and re-calls against zspk's card fits, stamped G6's: under N2's null (zspk's seeds against its
    #     card fit, the deciding null) and under the 7 October null re-based by zspk's base screens (beside); N2's
    #     loader refuses what is not N2's; relz, screen_pair and screen_recall restored after
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit, Z.against_rel
        saved_io = SP.load, SR.load_null
        assert SR.load_null is Z.NULL_OF
        big = {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)}
        null7 = SR.fake_null(T / "null", "n1", big)
        n2 = fake_n2(T / "n2", big)
        bs = {sp: str(SR.fake_compare(T / BASE_FITS[sp][0], BASE_FITS[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE)) for sp in SPLITS}
        zb = {sp: "/".join(("outputs",) + BASE_FITS[sp]) for sp in SPLITS}
        nl, sources = n2_null(n2)
        assert sorted(sources) == sorted(f"N2 {sp} seed {sd}" for sp in SR.NULL_SPLITS for sd in SR.SEEDS)
        assert abs(nl["L-hotpotqa"]["hotpotqa"]["floor"] - SR.floor_of(-0.010, 0.012)) < 1e-12
        assert nl["L-musique"]["webqsp"]["floor"] == S.FLOOR and nl["L-musique"]["webqsp"]["seed0"] == 0.6
        SR.must_stop(n2_null, null7)                                      # decided against step 1's fits
        SR.must_stop(n2_null, fake_n2(T / "b1", big, arm="base"))         # not zspk's fits
        SR.must_stop(n2_null, fake_n2(T / "b2", big, hidden=64))          # not on the card's settings
        SR.must_stop(n2_null, fake_n2(T / "b3", big, seed=3))             # not seeds 1 and 2
        SR.must_stop(n2_null, n2[:3])
        SR.must_stop(n2_null, [n2[0], n2[0], n2[2], n2[3]])
        for arm in ARM_NAMES:
            za = SR.fake_compare(T / arm, FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm, base_r5=0.6,
                                 base=zb["L-musique"])
            zh = SR.fake_compare(T / arm, FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            zw = SR.fake_compare(T / f"{arm}2", FIT[arm]["L-musique"], "L-musique", {}, arm=arm, base_r5=0.6,
                                 base="outputs/screen/fits/scr-zspg")
            SR.must_stop(pair, arm, [zw, zh], T / "bad")
            pr = pair(arm, [za, zh], T / f"p-{arm}")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE, pr["verdict"]
            pj = json.loads((T / f"p-{arm}.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "G6" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
            assert "null_kind" not in pj and "round" not in pj
            assert (T / f"p-{arm}.md").read_text(encoding="utf-8").startswith(f"G6 ({DECLARED}): {arm}")
            for null, files, extra in (("N2", n2, None), ("oct7", null7, bs)):
                r = recall(arm, files, T / f"p-{arm}.json", T / f"r-{arm}-{null}", null, extra)
                assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], \
                    (arm, null, r["verdict"], r["changed"])
                rj = json.loads((T / f"r-{arm}-{null}.json").read_text(encoding="utf-8"))
                assert rj["stage"] == "G6" and rj["arm"] == arm and rj["base_arm"] == BASE and rj["null_kind"] == null
                assert rj["deciding"] == (arm == "zg6" and null == "N2") and rj["declared_in"] == DECLARED
                assert sorted(rj["null"]) == sorted(
                    [f"N2 {sp} seed {sd}" for sp in SR.NULL_SPLITS for sd in SR.SEEDS] if null == "N2" else
                    [f"{sp} seed {sd}" for sp in SR.NULL_SPLITS for sd in SR.SEEDS] + [f"rel {sp}" for sp in SPLITS])
                tx = (T / f"r-{arm}-{null}.md").read_text(encoding="utf-8")
                assert tx.startswith(f"G6 ({DECLARED}): {arm}") and f"| zspk R@5 | {arm} R@5 (G6) |" in tx
                assert ("N2's fresh zspk null) (docs/SCREENS.md" in tx) == (null == "N2")
                assert ("step 1's seeds 1 and 2 against seed 0" in tx) == (null == "oct7") and "rel's" not in tx
            SR.must_stop(recall, arm, null7, T / f"p-{arm}.json", T / "x1", "N2")         # the 7 Oct files as N2's
            SR.must_stop(recall, arm, n2, T / f"p-{arm}.json", T / "x2", "oct7", bs)      # N2's files as 7 Oct's
            SR.must_stop(recall, arm, fake_n2(T / f"b4{arm}", big, base_r5=0.55), T / f"p-{arm}.json", T / "x3", "N2")
            SR.must_stop(recall, arm, n2, T / f"p-{arm}.json", T / "x4", "N3")
            other = "zg6t" if arm == "zg6" else "zg6"
            SR.must_stop(recall, other, n2, T / f"p-{arm}.json", T / "x5", "N2")          # not other's pair
            assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit, Z.against_rel) == saved_z
            assert (SP.load, SR.load_null) == saved_io
        # 14. the verdict: no LOSS and a GAIN under N2 adopts for the full run; a LOSS or no GAIN does not, each with
        #     its next form; a missing re-call, one not G6's, not N2's, not zg6's or not twelve reads is INCOMPLETE;
        #     beside: the twin's and the 7 October re-calls, zg6 minus zg6t and minus zspk from the comparisons, the
        #     filed R@5 of zrckp, zspx and zg2, and the MLP share
        cmp = {}
        for sp in SPLITS:
            b0, bt = f"outputs\\screen\\fits\\{BASE_FITS[sp][2]}", f"outputs\\screen\\fits\\{FIT['zg6t'][sp]}"
            rows_t = [{"dataset": ds, "read": "x", "delta": [0.004, 0.002, -0.001], "lo": [0.001] * 3,
                       "hi": [0.007] * 3, "call": "WITHIN"} for ds in LG.EVAL_ORDER]
            rows_b = [{"dataset": ds, "read": "x", "delta": [0.05, 0.03, 0.02], "lo": [0.04] * 3,
                       "hi": [0.06] * 3, "call": "GAIN"} for ds in LG.EVAL_ORDER]
            cmp[sp] = T / f"cmp-{sp}.json"
            LC.write_json(cmp[sp], {"bases": [b0, bt], "by_base": {b0: {"rows": rows_b}, bt: {"rows": rows_t}}})
        (T / "outputs" / "l1").mkdir(parents=True)
        for sp in SPLITS:
            rows_m = [{"dataset": ds, "new": [0.6, 0, 0]} for ds in LG.EVAL_ORDER]
            LC.write_json(T / "outputs" / "l1" / f"base-zrckp-{sp}.json",
                          {"new": f"outputs\\screen\\fits\\{ZRCKP[sp]}", "bases": ["b"], "by_base": {"b": {"rows": rows_m}}})
        gm, lw = ("L-musique", "musique"), ("L-hotpotqa", "webqsp")
        tw = fake_recall(T / "tw.json", "zg6t", {("L-hotpotqa", "2wiki"): "GAIN"})
        o7 = fake_recall(T / "o7.json", "zg6", {gm: "GAIN", ("L-musique", "squad"): "GAIN"}, null="oct7")
        rp = T / "rc.json"
        cases = (({gm: "GAIN"}, {}, "ADOPT_FOR_FULL_RUN", NEXT["ADOPT_FOR_FULL_RUN"]),
                 ({gm: "GAIN", lw: "LOSS"}, {}, "NOT_ADOPTED", NEXT["GAIN_AND_LOSS"]),
                 ({}, {}, "NOT_ADOPTED", NEXT["NO_GAIN"]),
                 ({lw: "LOSS"}, {}, "NOT_ADOPTED", NEXT["NO_GAIN"]),
                 ({gm: "GAIN"}, {"stage": "G2"}, "INCOMPLETE", NEXT["INCOMPLETE"]),
                 ({gm: "GAIN"}, {"null_kind": "oct7"}, "INCOMPLETE", NEXT["INCOMPLETE"]),
                 ({gm: "GAIN"}, {"arm": "zg6t"}, "INCOMPLETE", NEXT["INCOMPLETE"]),
                 ({gm: "GAIN"}, {"base_arm": "zsp"}, "INCOMPLETE", NEXT["INCOMPLETE"]),
                 ({gm: "GAIN"}, {"rows": [{"split": "L-musique", "dataset": "musique", "call": "GAIN"}]}, "INCOMPLETE",
                  NEXT["INCOMPLETE"]))
        for i, (calls, over, want, nxt) in enumerate(cases):
            v = verdict({"zg6": fake_recall(rp, "zg6", calls, **over), "zg6t": tw, "oct7": o7}, cmp, T / f"v{i}", root=T)
            assert v["verdict"] == want and v["next_form"] == nxt, (i, v["verdict"], v["problems"])
            b = v["beside"]
            assert b["twin_recall"] == {"gains": ["L-hotpotqa 2wiki"], "losses": []}
            assert b["oct7_recall"] == {"gains": ["L-musique squad", "L-musique musique"], "losses": []}
            assert all(len(b["zg6_minus_zg6t"][sp]) == 6 and len(b["zg6_minus_zspk"][sp]) == 6 for sp in SPLITS)
            assert b["zg6_minus_zspk"]["L-musique"][0]["delta"] == [0.05, 0.03, 0.02]
            if len(v["calls"]) == 12:
                assert b["r5"]["L-musique metaqa"]["zrckp"] == 0.6 and b["r5"]["L-musique metaqa"]["zg6t"] == 0.65
                assert b["mlp_share"]["L-musique metaqa"] == 0.8 and b["r5"]["L-musique metaqa"]["zspx"] is None
            assert "Next form" in (T / f"v{i}.md").read_text(encoding="utf-8")
        v = verdict({"zg6": T / "none.json", "zg6t": T / "none2.json", "oct7": T / "none3.json"}, cmp, T / "vn", root=T)
        assert v["verdict"] == "INCOMPLETE" and v["beside"]["twin_recall"] is None and v["beside"]["oct7_recall"] is None
        v = verdict({"zg6": fake_recall(rp, "zg6", {gm: "GAIN"}), "zg6t": tw, "oct7": tw},
                    {sp: T / "x.json" for sp in SPLITS}, T / "v9", root=T)
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and v["beside"]["oct7_recall"] is None
        assert v["beside"]["zg6_minus_zg6t"] == {sp: None for sp in SPLITS}
        assert mlp_share(0.6, 0.401, 0.4) is None and mlp_share(None, 0.7, 0.4) is None
        # the real filed reads beside (read only): zrckp, zspx and zg2 as filed
        fr = filed_r5(ROOT)
        assert fr["zrckp"][("L-musique", "metaqa")] == 0.8019 and fr["zspx"][("L-hotpotqa", "musique")] == 0.5901
        assert fr["zg2"][("L-musique", "musique")] == 0.511 and all(len(fr[k]) == 12 for k in ("zrckp", "zspx", "zg2"))
    # 15. nothing written under G6's roots or the beside reads' root
    assert snapshot() == snap0, "the selftest wrote under G6's roots"
    log(f"zg6 selftest: arms zg6 and zg6t on P1c3's link carve, no MLP class in their MRO, earlier arms unchanged; "
        f"from seed 0 zrm's encoder and draws less its output layer, rrf skip and chain gate, plus G6's "
        f"{n_params(128)} parameters at hidden 128 (readout random); no MLP score read (every ancestor forward made to "
        f"raise, base_z NaN: the same score); the encoder is zrm's (its output layer's input, eval and training), the "
        f"chain features rmatch's; G6's layers are G2's bit for bit; each arm equals the model by hand; the twin is "
        f"the arm with its edges removed bit for bit (eval and training); the readout unbounded (x100 -> x100); the "
        f"family dropout from the module's own draws, training only, by hand, never all of a question's families "
        f"(501 draws); gradients reach the encoder, the chain match, every layer and the readout (the twin's edge "
        f"maps exactly 0); toy fits repeat bit for bit and move; train and read refusals; relz's pair and re-calls "
        f"under N2's and the 7 October nulls, stamped G6's, N2's loader refusals, relz restored; the verdict's "
        f"outcomes and next forms, its beside rows and the MLP share; the filed beside reads; nothing written under "
        f"G6's roots ({time.time() - t0:.1f}s): ok")
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
        return 1 if verdict()["verdict"] == "INCOMPLETE" else 0
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--null-kind", choices=tuple(NULLS))
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
            log(f"zg6 {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zg6: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
