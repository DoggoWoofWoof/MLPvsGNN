"""G2's screen (docs/G2_HIDDEN_STATE_GNN.md): the lean GNN passes hidden states, not scores.

zspk (zp1c3.ZSPK: zrm's network, the KT head and zsp's propagation step, on P1c3's link carve) passes one number per row
along an edge: the score. zg2 keeps zspk whole and adds a message-passing stack that starts from the shared encoder's
hidden state, the input of zrm's output layer, read from the same forward pass:
- boundary: x = [H (the encoder's hidden state), z (zspk's score z-scored over the question's rows, detached, clipped to
  +-8), KT (the row's eight KT columns)]; h0 = GELU(W_in x + b_in), D wide;
- each of LAYERS layers, for each edge u -> v of family f (zlink's pool edges, both directions, three families):
    c_q   = the question's context: its rows' states weighted by the softmax of the detached z (one vector per question)
    l_e   = clip(<W_a h_u, c_q(v)> / sqrt(D) + beta_f, -8, 8)
    m_e   = (W_f h_u)                                   (one D x D map per family)
    mean  = per family, sum of sigmoid(l_e) m_e over v's edges / v's edge count in that family (0 without one)
    att   = over all of v's edges, sum of exp(l_e) m_e / sum of exp(l_e) (0 without one): a soft pick of one neighbour
    h     = h + GELU(layer_norm(W_s h + W_mean [mean_0, mean_1, mean_2] + W_att att + W_x [z, KT] + b))
  [z, KT] enters at every layer (NBFNet's boundary condition);
- readout: r = w_o h + b_o, zero at the start; zg2's score is zspk's plus TB tanh(r / TB) (the bounded residual of
  qd_gnn13's tb5);
- edge-family dropout in training (qd_gnn13's fd25 rule): each question drops each family it has with probability FD,
  keeping its structural family (or, without one, its lowest family) when every family it has would go. The draws come
  from the module's own generator.
zg2t, the twin, is zg2 with no edges reaching its layers (every mean and att is 0): the same parameters, context and
readout. It is not an MLP-track model (its context pools the question's other rows); it measures what G2's capacity
adds without message passing.

The whole network trains together, with zrm's settings (rmatch.py's train), on P1c3's carves (P1c2's roots, the KT
input beside). From the same seed zg2's forward is zspk's bit for bit until its readout moves (the selftest requires
it), and no draw of zspk's moves. No new graph, column, encoder or text. Index sums only (index_add_), so the
deterministic settings hold on the card.

    python outputs/mp_unified/zg2.py compare --new outputs/screen/fits/scr-p1c3-zspk \\
        --base outputs/step1/fits/L-musique --out outputs/g2/base-zspk-L-musique
    python outputs/mp_unified/zg2.py train --split L-musique --name scr-g2-zg2 --arm zg2 --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zg2.py read --name scr-g2-zg2 --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zg2.py compare --new outputs/screen/fits/scr-g2-zg2 \\
        --base outputs/screen/fits/scr-p1c3-zspk,outputs/screen/fits/scr-g2-zg2t,outputs/screen/fits/scr-zspg,\\
outputs/step1/fits/L-musique --out outputs/screen/scr-g2-zg2
    python outputs/mp_unified/zg2.py pair --arm zg2 --screens S1.json,S2.json --out outputs/screen/scr-g2-zg2-pair
    python outputs/mp_unified/zg2.py recall --arm zg2 --null N1,N2,N3,N4 --pair P.json --out ...-pair-recall
    python outputs/mp_unified/zg2.py verdict
    python outputs/mp_unified/zg2.py --selftest
"""
import os
import sys

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

import zp1c3 as Z3  # noqa: E402  (P1c3's KT input, carves and arms; through it P1c2's roots, gate and card)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/G2_HIDDEN_STATE_GNN.md"
OUT = ROOT / "outputs" / "g2"
SCREEN = ROOT / "outputs" / "screen"
KT_KEY, NKT, LK_KEY, FAMS, ZCLIP = Z3.KT_KEY, Z3.NKT, ZP.LK_KEY, ZP.FAMS, ZP.ZCLIP
NB = 1 + NKT                                    # the boundary fed at every layer: z and the eight KT columns
D, LAYERS, SEED_OFF, TB, FD, LCLIP = 64, 3, 3501, 0.5, 0.25, 8.0
SPLITS = Z3.SPLITS
BASE = "zspk"
ARM_NAMES = ("zg2", "zg2t")
FIT = {"zg2": {"L-musique": "scr-g2-zg2", "L-hotpotqa": "scr-g2-zg2-hp"},
       "zg2t": {"L-musique": "scr-g2-zg2t", "L-hotpotqa": "scr-g2-zg2t-hp"}}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-p1c3-zspk"), "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}
BASE_SCREENS = {sp: f"outputs/g2/base-zspk-{sp}.json" for sp in SPLITS}
AGAINST = "zspk's P1c3 card fit of each split (scr-p1c3-zspk), on the same pools and KT input"
BESIDE = {"zg2": "zg2t (the twin, no edges), zsp's card fit on today's pools (scr-zspg) and step 1's",
          "zg2t": "zsp's card fit on today's pools (scr-zspg) and step 1's"}
G2_IN = ("g2_in_w", "g2_in_b")
G2_LAYER = tuple(tuple(f"g2_{k}_{p}" for p in ("msg", "att", "fb", "self", "mean", "agg", "x", "b"))
                 for k in range(LAYERS))
G2_OUT = ("g2_out_w", "g2_out_b")
G2_NAMES = G2_IN + tuple(n for lay in G2_LAYER for n in lay) + G2_OUT
RULE = ("SCREEN_PASSED when zg2's re-call under the seed null has no LOSS on any of its twelve reads and at least one "
        "GAIN; NOT_ADOPTED otherwise; INCOMPLETE when the re-call is missing, not G2's, or not twelve reads. A pass is "
        "fitted again on P1f's final pools before anything is adopted (docs/G2_HIDDEN_STATE_GNN.md, section 4)")
PAIR = {a: SCREEN / f"scr-g2-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-g2-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"


def n_params(H):
    per = FAMS * D * D + D * D + FAMS + D * D + D * FAMS * D + D * D + D * NB + D
    return D * (H + NB) + D + LAYERS * per + D + 1


# ── the model ────────────────────────────────────────────────────────────────


def _grab(mod, inp):
    """The output layer's input (the encoder's hidden state), kept on the layer for the forward that called it."""
    mod.g2_h = inp[0]
    mod.g2_n = getattr(mod, "g2_n", 0) + 1


def family_keep(f, qv, B, u01):
    """(E,) bool: the edges kept by the family dropout. f: each edge's family; qv: its question; u01: (B, FAMS) uniform
    draws. A question drops each family it has where u01 < FD; when every family it has would go, it keeps its
    structural family (0), or without one its lowest family."""
    dev = f.device
    has = torch.zeros(B * FAMS, dtype=torch.float32, device=dev).index_add_(
        0, qv * FAMS + f, torch.ones(f.numel(), dtype=torch.float32, device=dev)).view(B, FAMS) > 0
    drop = (u01.to(dev) < FD) & has
    left = has & ~drop
    gone = has.any(1) & ~left.any(1)
    first = torch.where(has, torch.arange(FAMS, device=dev).expand(B, FAMS),
                        torch.full_like(drop, FAMS, dtype=torch.long))
    lowest = first.min(1).values
    save = torch.zeros_like(drop)
    rows = torch.nonzero(gone).squeeze(1)
    save[rows, lowest[rows]] = True
    drop = drop & ~save
    return ~drop[qv, f]


def seg_softmax_rows(z, nq, B):
    """Per question, the softmax of z over its rows (z already clipped)."""
    e = torch.exp(z)
    den = torch.zeros(B, dtype=z.dtype, device=z.device).index_add_(0, nq, e)
    return e / den[nq]


def g2_stack(m, H, s, kt, lk, nq, B, twin, u01=None):
    """The stack's readout r (N,) for hidden states H, zspk's score s and KT columns kt over the edges lk (the
    module's docstring); twin: no edges; u01: the family dropout's draws (training only)."""
    N = s.numel()
    dev, dt = s.device, H.dtype
    z = LG.seg_zscore8D(s.detach().unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP).to(dt)
    xb = torch.cat([z.unsqueeze(1), kt.to(dt)], 1)
    h = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g2_in_w, m.g2_in_b))
    w = seg_softmax_rows(z, nq, B).unsqueeze(1)
    if twin:
        u = v = f = torch.zeros(0, dtype=torch.long, device=dev)
    else:
        u, v, f = lk["u"], lk["v"], lk["f"]
        if u01 is not None:
            keep = family_keep(f, nq[v], B, u01)
            u, v, f = u[keep], v[keep], f[keep]
    E = u.numel()
    at = v * FAMS + f
    one = torch.ones(E, dtype=dt, device=dev)
    deg = torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, one).view(N, FAMS, 1).clamp_min(1.0)
    qv = nq[v]
    for wm, wa, fb, ws, wmean, wagg, wx, b in ((getattr(m, n) for n in lay) for lay in G2_LAYER):
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
    return Fn.linear(h, m.g2_out_w, m.g2_out_b).squeeze(-1)


class G2Mix(Z3.ZSPK):
    """zspk plus G2's stack (readout zero at the start) over the encoder's hidden states and the pool's edges."""
    TWIN = False

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        H = self.out.in_features
        g = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        uni = lambda *shape, fan: (torch.rand(*shape, generator=g) * 2 - 1) / fan ** 0.5  # noqa: E731
        self.g2_in_w = nn.Parameter(uni(D, H + NB, fan=H + NB))
        self.g2_in_b = nn.Parameter(torch.zeros(D))
        for nm, na, nf, ns, nmean, nagg, nx, nb in G2_LAYER:
            setattr(self, nm, nn.Parameter(uni(FAMS * D, D, fan=D)))
            setattr(self, na, nn.Parameter(uni(D, D, fan=D)))
            setattr(self, nf, nn.Parameter(torch.zeros(FAMS)))
            setattr(self, ns, nn.Parameter(uni(D, D, fan=D)))
            setattr(self, nmean, nn.Parameter(uni(D, FAMS * D, fan=FAMS * D)))
            setattr(self, nagg, nn.Parameter(uni(D, D, fan=D)))
            setattr(self, nx, nn.Parameter(uni(D, NB, fan=NB)))
            setattr(self, nb, nn.Parameter(torch.zeros(D)))
        self.g2_out_w = nn.Parameter(torch.zeros(1, D))
        self.g2_out_b = nn.Parameter(torch.zeros(1))
        self.g2_gen = torch.Generator().manual_seed(int(seed) + SEED_OFF + 1)
        self.out.register_forward_pre_hook(_grab)

    def forward(self, feats, keep, nq, B, base_z):
        self.out.g2_h, self.out.g2_n = None, 0
        s = super().forward(feats, keep, nq, B, base_z)
        H, n = self.out.g2_h, self.out.g2_n
        self.out.g2_h, self.out.g2_n = None, 0
        if LK_KEY not in feats or KT_KEY not in feats:
            return s
        if n != 1 or H is None or H.shape[0] != s.shape[0]:
            raise RuntimeError(f"zg2: the output layer ran {n} times in one forward, not once")
        u01 = torch.rand(B, FAMS, generator=self.g2_gen) if self.training and not self.TWIN else None
        r = g2_stack(self, H, s, feats[KT_KEY], feats[LK_KEY], nq, B, self.TWIN, u01)
        return s + TB * torch.tanh(r / TB)


class ZG2(G2Mix):
    """zg2: zspk plus G2's stack over the pool's edges."""


class ZG2T(G2Mix):
    """zg2t: zg2 with no edges reaching the stack (the twin)."""
    TWIN = True


ARMS = {"zg2": (ZG2, Z3.KTLinkCarveBase), "zg2t": (ZG2T, Z3.KTLinkCarveBase)}
S.ARMS.update(ARMS)
SETTINGS = {"state": D, "layers": LAYERS, "boundary": ["z", *Z3.KT_COLS], "seed_offset": SEED_OFF, "bound": TB,
            "family_dropout": FD, "logit_clip": LCLIP, "aggregations": ["gated mean per family", "attention over all"],
            "context": "the question's rows' states, softmax of the detached z", "edges": ZP.SETTINGS["edges"],
            "parameters_at_hidden_128": n_params(128)}


# ── train, read, compare ─────────────────────────────────────────────────────


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        g = Z2.gate_passed()
        rec.update({"zg2_sha256": LC.sha_src(__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "g2": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE, "device": Z2.DEVICE, "graph": "today's",
                        "pools": f"P1e's {g.get('construction')} at each question's frozen size (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "stack": SETTINGS, "twin": ARMS[arm][0].TWIN,
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zg2: train takes --arm zg2 or zg2t, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zg2: G2's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zg2: the {arm} fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "g2" not in rec:
        raise SystemExit(f"zg2: {name} is not a G2 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against zspk's P1c3 card fit ──────────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"zg2: no arm {arm}")

    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"zg2: no base fit of {BASE} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    return (("(rel's screen fits scr-rel and scr-rel-hp)",
             "(zspk's P1c3 card fits scr-p1c3-zspk and scr-p1c3-zspk-hp)"),
            ("(rel's screen fits)", "(zspk's P1c3 card fits)"),
            ("| rel R@5 | relz R@5 |", f"| zspk R@5 | {arm} R@5 (G2) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"G2 ({DECLARED}): {arm} on KIND_T's pools against {BASE}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "G2", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "beside": BESIDE[arm], "zg2_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, base_screens=None):
    with on_base(arm):
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, arm)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def recall_rows(rp, arm, problems):
    rp = Path(rp)
    if not rp.exists():
        problems.append(f"{rp.name} missing")
        return []
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec.get("stage") != "G2" or rec.get("declared_in") != DECLARED or rec.get("arm") != arm:
        problems.append(f"{rp.name} is not G2's re-call of {arm}")
    rows = rec.get("rows", [])
    got = sorted((r.get("split"), r.get("dataset")) for r in rows)
    if got != sorted((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
        problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six datasets")
    return rows


def beside_rows(compare_file, base_name):
    """The comparison's rows against base_name (one of its bases); None when absent."""
    p = Path(compare_file)
    if not p.exists():
        return None
    c = json.loads(p.read_text(encoding="utf-8"))
    key = next((b for b in c.get("bases", []) if S2.parts_of(b)[-1] == base_name), None)
    if key is None:
        return None
    return [{"dataset": r["dataset"], "read": r["read"], "delta": r["delta"], "lo": r["lo"], "hi": r["hi"],
             "call": r["call"]} for r in c["by_base"][key]["rows"]]


def verdict(recalls=None, compares=None, out=None):
    """G2's screen verdict from zg2's re-call (section 4). Beside, deciding nothing: zg2 minus its twin per read (the
    stack's message passing) and the twin's own re-call against zspk (the stack's capacity)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    problems = []
    rows = recall_rows(recalls["zg2"], "zg2", problems)
    tp = []
    trows = recall_rows(recalls["zg2t"], "zg2t", tp)
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    if problems:
        v = "INCOMPLETE"
    elif losses or not gains:
        v = "NOT_ADOPTED"
    else:
        v = "SCREEN_PASSED"
    compares = compares or {sp: SCREEN / f"{FIT['zg2'][sp]}.json" for sp in SPLITS}
    mp = {sp: beside_rows(compares[sp], FIT["zg2t"][sp]) for sp in SPLITS}
    twin = None if tp else {"gains": [tag(r) for r in trows if r.get("call") == "GAIN"],
                            "losses": [tag(r) for r in trows if r.get("call") == "LOSS"]}
    rp = Path(recalls["zg2"])
    rec = {"stage": "G2", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": "zg2", "base_arm": BASE,
           "recall": rp.relative_to(ROOT).as_posix() if rp.is_relative_to(ROOT) else rp.name,
           "recall_sha256": LC.sha_file(rp) if rp.exists() else None, "problems": problems, "gains": gains,
           "losses": losses, "calls": {tag(r): r.get("call") for r in rows}, "beside_zg2_minus_zg2t": mp,
           "beside_twin_recall": twin, "beside_twin_problems": tp, "zg2_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# G2's screen verdict: **{v}**", "", f"Rule: {RULE}.", "",
          f"zg2 (zspk plus a {LAYERS}-layer, {D}-wide stack over the encoder's hidden states) against zspk, after the "
          f"seed-null re-call: {len(gains)} GAIN ({', '.join(gains) or 'none'}), {len(losses)} LOSS "
          f"({', '.join(losses) or 'none'})."]
    if problems:
        md += ["", "Problems: " + "; ".join(problems)]
    md += ["", "Beside, deciding nothing.", "",
           "The twin (zg2t, the same stack with no edges) against zspk, after its re-call: " +
           (f"{len(twin['gains'])} GAIN ({', '.join(twin['gains']) or 'none'}), {len(twin['losses'])} LOSS "
            f"({', '.join(twin['losses']) or 'none'})." if twin else "not available (" + "; ".join(tp) + ")."), "",
           "zg2 minus zg2t (what the stack's message passing adds), R@5 / FC@5 / hit@1, with the 95% "
           "question-bootstrap interval of the R@5 difference:", "",
           "| split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075 floor) |",
           "|---|---|---|---|---|---|---|"]
    for sp in SPLITS:
        for r in mp[sp] or []:
            d, lo, hi = r["delta"], r["lo"], r["hi"]
            md.append(f"| {sp} | {r['dataset']} | {r['read']} | {d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | "
                      f"{d[1]:+.4f} | {d[2]:+.4f} | {r['call']} |")
        if mp[sp] is None:
            md.append(f"| {sp} | (no comparison with {FIT['zg2t'][sp]}) | | | | | |")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zg2 verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''})")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, who, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "G2", "declared_in": DECLARED, "arm": who, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def by_hand(m, H, s, kt, lk, nq, B, twin, keep_e=None):
    """g2_stack with python loops over the edges (the selftest's reference)."""
    N = s.numel()
    z = LG.seg_zscore8D(s.detach().unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
    xb = torch.cat([z.unsqueeze(1), kt], 1)
    h = Fn.gelu(torch.cat([H, xb], 1) @ m.g2_in_w.T + m.g2_in_b)
    edges = [] if twin else list(zip(lk["u"].tolist(), lk["v"].tolist(), lk["f"].tolist()))
    if keep_e is not None and not twin:
        edges = [e for e, k in zip(edges, keep_e.tolist()) if k]
    qs = nq.tolist()
    for nm, na, nf, ns, nmean, nagg, nx, nb in G2_LAYER:
        wm, wa, fb, ws, wmean, wagg, wx, b = (getattr(m, n) for n in (nm, na, nf, ns, nmean, nagg, nx, nb))
        c = torch.zeros(B, D)
        for q in range(B):
            rows = [i for i in range(N) if qs[i] == q]
            if rows:
                p = torch.softmax(z[rows], 0)
                c[q] = (p.unsqueeze(1) * h[rows]).sum(0)
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
    return (h @ m.g2_out_w.T + m.g2_out_b).squeeze(-1)


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arms: zspk's model plus the stack on P1c3's link carve; P1c3's, P1c2's and P1c3p's arms unchanged
    assert S.ARMS["zg2"] == (ZG2, Z3.KTLinkCarveBase) and S.ARMS["zg2t"] == (ZG2T, Z3.KTLinkCarveBase)
    assert ZG2.__mro__[1:3] == (G2Mix, Z3.ZSPK) and ZG2T.TWIN and not ZG2.TWIN
    assert S.ARMS["zspk"] == Z3.ARMS["zspk"] and S.ARMS["zrck"] == Z3.ARMS["zrck"]
    assert S.ARMS["zsp"] == Z2.ARMS["zsp"] and S.ARMS["zrc"] == Z2.ARMS["zrc"]
    # 2. the family dropout's rule by hand: drops below FD where the question has the family; all-gone keeps family 0,
    #    or without it the lowest family the question has
    f = torch.tensor([0, 1, 2, 1, 2, 2, 0, 1])
    qv = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3])
    u01 = torch.tensor([[0.1, 0.9, 0.1], [0.9, 0.1, 0.1], [0.9, 0.9, 0.1], [0.9, 0.9, 0.9]])
    assert family_keep(f, qv, 4, u01).tolist() == [False, True, False, True, False, True, True, True]
    u01 = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.9]])
    assert family_keep(f, qv, 4, u01).tolist() == [True, False, False, True, False, True, True, False]
    tmp = Path(tempfile.mkdtemp(prefix="zg2_"))
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
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, _gold = c.batch(qs, blocks)
        lk = feats[LK_KEY]
        assert lk["u"].numel() > 0
        keep = torch.ones(B, len(blocks))
        # 3. from the same seed zg2 and zg2t are zspk's state plus the stack (readout zero), their forward is zspk's
        #    bit for bit in eval and in training, and the global generator is where zspk leaves it
        nets, after = {}, {}
        for k, cl in (("zspk", Z3.ZSPK), ("zg2", ZG2), ("zg2t", ZG2T)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after["zg2"], after["zspk"]) and torch.equal(after["zg2t"], after["zspk"])
        so = nets["zspk"].state_dict()
        for k in ARM_NAMES:
            sn = nets[k].state_dict()
            assert sorted(set(sn) - set(so)) == sorted(G2_NAMES) and all(torch.equal(so[x], sn[x]) for x in so)
            assert sum(int(sn[x].numel()) for x in G2_NAMES) == n_params(16)
            assert float(sn["g2_out_w"].abs().sum()) == 0.0 and float(sn["g2_out_b"].abs().sum()) == 0.0
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out["zg2"], out["zspk"]) and torch.equal(out["zg2t"], out["zspk"]), mode
            # 4. set: zg2 is zspk's score plus TB tanh(r / TB), r the stack by hand from the encoder's hidden state; the
            #    twin by hand with no edges; without the edges or the KT input, zspk's own; the twin ignores the edges
            gen = torch.Generator().manual_seed(3)
            ref = nets["zspk"].eval()
            caught = []
            hk = ref.out.register_forward_pre_hook(lambda mod, inp: caught.append(inp[0]))
            s = ref(feats, keep, nq, B, bz)
            hk.remove()
            assert len(caught) == 1 and caught[0].shape == (s.numel(), 16)
            H = caught[0]
            for k in ARM_NAMES:
                m = nets[k].eval()
                for n in G2_NAMES:
                    getattr(m, n).copy_(torch.randn(getattr(m, n).shape, generator=gen) * 0.3)
                r = by_hand(m, H, s, feats[KT_KEY], lk, nq, B, m.TWIN)
                got = m(feats, keep, nq, B, bz)
                want = s + TB * torch.tanh(r / TB)
                assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (k, float((got - want).abs().max()))
                assert not torch.allclose(got, s)
                for drop in (LK_KEY, KT_KEY):
                    f2 = {x: y for x, y in feats.items() if x != drop}
                    assert torch.equal(m(f2, keep, nq, B, bz), ref(f2, keep, nq, B, bz))
            nets["zg2t"].load_state_dict(nets["zg2"].state_dict())
            g2, g2t = nets["zg2"](feats, keep, nq, B, bz), nets["zg2t"](feats, keep, nq, B, bz)
            assert not torch.allclose(g2, g2t)
            f3 = dict(feats)
            f3[LK_KEY] = {x: y[:0] for x, y in lk.items()}
            assert torch.allclose(nets["zg2"](f3, keep, nq, B, bz), g2t, atol=1e-6)
            # 5. in training the family dropout cuts edges with the module's own draws, by the rule above
            m = nets["zg2"].train()
            m.drop.p = 0.0
            g0 = m.g2_gen.get_state()
            u01 = torch.rand(B, FAMS, generator=m.g2_gen)
            m.g2_gen.set_state(g0)
            ke = family_keep(lk["f"], nq[lk["v"]], B, u01)
            got = m(feats, keep, nq, B, bz)
            m.eval()
            sref = ref(feats, keep, nq, B, bz)
            want = sref + TB * torch.tanh(by_hand(m, H, sref, feats[KT_KEY], lk, nq, B, False, ke) / TB)
            assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), float((got - want).abs().max())
            m.drop.p = 0.1
        # 6. toy fits through lean_gpu's loop: each stack moves, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm in ARM_NAMES:
            with S.patched(arm):
                f1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm}")
                f2_ = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} repeat")
            assert float(f1["swa"]["g2_out_w"].abs().sum()) > 0
            assert all(torch.equal(f1["swa"][x], f2_["swa"][x]) for x in f1["swa"])
        # 7. train and read refuse what is not G2's
        good = ["train", "--arm", "zg2", "--name", FIT["zg2"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
        SR.must_stop(train, ["train", "--arm", "zspk", "--name", FIT["zg2"]["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, good, "L-2wiki")
        SR.must_stop(train, ["train", "--arm", "zg2", "--name", FIT["zg2t"]["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, ["train", "--arm", "zg2", "--name", FIT["zg2"]["L-hotpotqa"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
        SR.must_stop(train, ["train", "--arm", "zg2", "--name", FIT["zg2"]["L-musique"]], "L-musique")
        SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
        (tmp / "fits" / "y").mkdir(parents=True)
        for rec in ({"arm": "zspk", "p1c3": {}}, {"arm": "zg2", "p1c3": {}}, {"arm": "zrckp", "g2": {}}):
            LC.write_json(tmp / "fits" / "y" / "screen.json", rec)
            SR.must_stop(read, ["--name", "y", "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
        SR.must_stop(read, ["--name", "none", "--out-root", str(tmp / "fits"), "--cache-root", "outputs/p1c2/cache"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 8. relz's pair and re-call against zspk's P1c3 card fits, stamped G2's, relz restored after
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        bs = {sp: str(SR.fake_compare(T / BASE_FITS[sp][0], BASE_FITS[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE)) for sp in SPLITS}
        zb = {sp: "/".join(("outputs",) + BASE_FITS[sp]) for sp in SPLITS}
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
            assert pj["stage"] == "G2" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T / f"p-{arm}.json", T / f"r-{arm}", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T / f"r-{arm}.md").read_text(encoding="utf-8")
            assert t.startswith(f"G2 ({DECLARED})") and f"| zspk R@5 | {arm} R@5 (G2) |" in t
            assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
        # 9. the verdict: no LOSS and a GAIN passes the screen; a LOSS, no GAIN, a missing or foreign re-call, or fewer
        #    reads do not; zg2 minus zg2t comes from each comparison's twin base and the twin's re-call is beside
        cmp = {}
        for sp in SPLITS:
            b0, bt = f"outputs\\screen\\fits\\{BASE_FITS[sp][2]}", f"outputs\\screen\\fits\\{FIT['zg2t'][sp]}"
            rows = [{"dataset": ds, "read": "x", "delta": [0.004, 0.002, -0.001], "lo": [0.001] * 3,
                     "hi": [0.007] * 3, "call": "WITHIN"} for ds in LG.EVAL_ORDER]
            cmp[sp] = T / f"cmp-{sp}.json"
            LC.write_json(cmp[sp], {"bases": [b0, bt], "by_base": {b0: {"rows": []}, bt: {"rows": rows}}})
        tw = fake_recall(T / "tw.json", "zg2t", {("L-hotpotqa", "2wiki"): "GAIN"})
        cases = (({("L-musique", "2wiki"): "GAIN"}, {}, "SCREEN_PASSED"),
                 ({("L-musique", "2wiki"): "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"}, {}, "NOT_ADOPTED"),
                 ({}, {}, "NOT_ADOPTED"),
                 ({("L-musique", "2wiki"): "GAIN"}, {"stage": "P1c3"}, "INCOMPLETE"),
                 ({("L-musique", "2wiki"): "GAIN"}, {"arm": "zg2t"}, "INCOMPLETE"),
                 ({("L-musique", "2wiki"): "GAIN"}, {"rows": [{"split": "L-musique", "dataset": "2wiki",
                                                               "call": "GAIN"}]}, "INCOMPLETE"))
        for i, (calls, over, want) in enumerate(cases):
            v = verdict({"zg2": fake_recall(T / f"rc{i}.json", "zg2", calls, **over), "zg2t": tw}, cmp, T / f"v{i}")
            assert v["verdict"] == want, (i, v["verdict"], v["problems"])
            assert all(len(v["beside_zg2_minus_zg2t"][sp]) == 6 for sp in SPLITS)
            assert v["beside_twin_recall"] == {"gains": ["L-hotpotqa 2wiki"], "losses": []}
        v = verdict({"zg2": T / "none.json", "zg2t": T / "none2.json"}, cmp, T / "vn")
        assert v["verdict"] == "INCOMPLETE" and v["beside_twin_recall"] is None
        v = verdict({"zg2": fake_recall(T / "rc9.json", "zg2", {("L-musique", "2wiki"): "GAIN"}), "zg2t": tw},
                    {sp: T / "x.json" for sp in SPLITS}, T / "v9")
        assert v["verdict"] == "SCREEN_PASSED" and v["beside_zg2_minus_zg2t"] == {sp: None for sp in SPLITS}
    log(f"zg2 selftest: arms zg2 and zg2t (zspk plus a {LAYERS}-layer, {D}-wide stack over the encoder's hidden state, "
        f"{n_params(128)} parameters at hidden 128) on P1c3's link carve, earlier arms unchanged; the family dropout's "
        f"rule by hand; at the start both are zspk bit for bit (eval and training draws, the global generator "
        f"unmoved); set, each equals the stack by hand (the twin without edges), in training with the dropout's own "
        f"draws; toy fits move the stack and repeat; train and read refuse what is not G2's; relz's pair and re-call "
        f"against zspk's card fits, stamped G2's; the verdict's cases and its beside rows "
        f"({time.time() - t0:.1f}s): ok")
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
        return 0 if verdict()["verdict"] != "INCOMPLETE" else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"zg2 {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zg2: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
