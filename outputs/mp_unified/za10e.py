"""A10e (docs/A10E_G2_END_TO_END.md): G2's hidden-state layers on G5's zspx, trained end to end; the next form A10's
verdict named for zg2x. A10 put G2's stack, unchanged, over zspx's full score: 0 GAIN, 0 LOSS after the seed-null
re-call (verdict of 18 October). Its stack read zspx's score through a detached z-score, so zspx's scorer and steps were never told
how their score moves the stack. G4's zspkg found that depth pays when the step's inputs keep the gradient (musique in
domain +0.0093 on L-hotpotqa). A10e keeps it in the stack:

    zg2xe   zg2x (za10g2x.ZG2X: zspx's score s3, then s3 + 0.5 tanh(r / 0.5), r = G2's stack reading s3's z-score) with
            that z-score computed without a detach: the boundary [z, KT] fed at every layer and the question context's
            softmax weights now carry the loss back into s3, and through it into zspx's three steps, its KT head and
            its scorer

g2_stack_e is zg2.g2_stack line for line but for that one detach, and ZG2XE.forward is G2Mix.forward line for line but
for the stack it calls (the selftest compares their syntax trees). The forward is zg2x's bit for bit for the same
weights; only the training signal changes. GNN only (the MLP takes no message).

    python outputs/mp_unified/za10e.py train --split L-musique --name scr-a10e-zg2xe --arm zg2xe --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za10e.py read --name scr-a10e-zg2xe --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za10e.py compare --new outputs/screen/fits/scr-a10e-zg2xe \\
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a10e-zg2xe
    python outputs/mp_unified/za10e.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za10e.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za10e.py verdict
    python outputs/mp_unified/za10e.py --selftest
The re-call's base screens are A9's (outputs/a9/base-zspx-<split>.json: zspx's card fits against step 1's).
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

import zg5mix as G5  # noqa: E402  (zspx, its carve, its fits; through it zg3res, zg4prop, zp1c3, zprop, zlink)
import zg2 as G2  # noqa: E402  (G2's stack, G2Mix, ZG2)
import za9fd as A9  # noqa: E402  (zspx's base screens against step 1)
import za10g2x as A10  # noqa: E402  (zg2x, the parent: G2Mix over zspx, its stack's z-score detached)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
ZP, ZL, ZM, RM = G5.ZP, G5.ZL, G5.ZM, G5.RM
Z, R, S2, S, LG, LC, SR = G5.Z, G5.R, G5.S2, G5.S, G5.LG, G5.LC, G5.SR
log = S.log
# zg2's own names, so that g2_stack_e and ZG2XE.forward read as G2's lines
KT_KEY, LK_KEY, FAMS, ZCLIP = G2.KT_KEY, G2.LK_KEY, G2.FAMS, G2.ZCLIP
D, TB, LCLIP, G2_LAYER = G2.D, G2.TB, G2.LCLIP, G2.G2_LAYER
seg_softmax_rows, family_keep = G2.seg_softmax_rows, G2.family_keep

DECLARED = "docs/A10E_G2_END_TO_END.md"
OUT = ROOT / "outputs" / "a10e"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
ARM, BASE_ARM, PARENT_ARM = "zg2xe", "zspx", A10.ARM
FIT = {"L-musique": "scr-a10e-zg2xe", "L-hotpotqa": "scr-a10e-zg2xe-hp"}
BASE_FITS = A9.BASE_FITS
BASE_SCREENS = A9.BASE_SCREENS
AGAINST = "zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp)"
WHAT = "zspx with G2's hidden-state layers over its score, trained end to end"
SETTINGS = {"base": "zspx (outputs/mp_unified/zg5mix.py), every setting, seed, batch and head of it",
            "stack": {**G2.SETTINGS, "context": "the question's rows' states, softmax of z (the gradient kept)"},
            "over": "zspx's full score s3 (after its three steps)",
            "z": "the stack's z-score of s3 over the question's rows, clipped to +-8, with the gradient kept (no "
                 "detach): the boundary [z, KT] at every layer and the context's softmax weights carry the loss back "
                 "into s3",
            "parent": "zg2x (outputs/mp_unified/za10g2x.py): the same model, its stack's z-score of s3 detached",
            "parameters_added_at_hidden_128": G2.n_params(128)}
PAIR = SCREEN / "scr-a10e-zg2xe-pair"
RECALL = SCREEN / "scr-a10e-zg2xe-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zg2xe's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A10e's, or not twelve "
        "reads (docs/A10E_G2_END_TO_END.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zg2xe fitted again on P1g-b's pools, then its full run beside zspx's, its own declared "
                              "file",
        "GAIN_AND_LOSS": "A10eb, the back-off form: zg2xe with the stack's readout bounded at 0.2 instead of 0.5 (the "
                         "MP-unified track's 0.2 tanh), so the layers can move each score less; one declared file",
        "NO_GAIN": "A10p, the stack before the steps: zg2xe's stack (its z-score with the gradient kept) reading "
                   "zspx's pre-step score (zspk's scorer and KT head), its bounded readout added there, so zspx's "
                   "three steps act on the stack's correction (G4: depth pays when the step acts on the score); G2, A10 "
                   "and A10e all put the stack after the steps; one declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the model ────────────────────────────────────────────────────────────────


def g2_stack_e(m, H, s, kt, lk, nq, B, twin, u01=None):
    """zg2.g2_stack line for line, with the gradient kept through its z-score of s (no detach)."""
    N = s.numel()
    dev, dt = s.device, H.dtype
    z = LG.seg_zscore8D(s.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP).to(dt)
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


class ZG2XE(A10.ZG2X):
    """zg2xe: zg2x's model and weights; its stack's z-score of zspx's score keeps the gradient."""

    def forward(self, feats, keep, nq, B, base_z):
        self.out.g2_h, self.out.g2_n = None, 0
        s = super(G2.G2Mix, self).forward(feats, keep, nq, B, base_z)       # what G2Mix.forward's super() returns
        H, n = self.out.g2_h, self.out.g2_n
        self.out.g2_h, self.out.g2_n = None, 0
        if LK_KEY not in feats or KT_KEY not in feats:
            return s
        if n != 1 or H is None or H.shape[0] != s.shape[0]:
            raise RuntimeError(f"zg2: the output layer ran {n} times in one forward, not once")
        u01 = torch.rand(B, FAMS, generator=self.g2_gen) if self.training and not self.TWIN else None
        r = g2_stack_e(self, H, s, feats[KT_KEY], feats[LK_KEY], nq, B, self.TWIN, u01)
        return s + TB * torch.tanh(r / TB)


ARMS = {ARM: (ZG2XE, G3.ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za10e_sha256": LC.sha_src(__file__), "za10g2x_sha256": LC.sha_src(A10.__file__),
                    "zg5mix_sha256": LC.sha_src(G5.__file__), "zg2_sha256": LC.sha_src(G2.__file__),
                    "zg3res_sha256": LC.sha_src(G3.__file__), "zg4prop_sha256": LC.sha_src(G4.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "a10e": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE_ARM, "parent_arm": PARENT_ARM,
                        "device": Z2.DEVICE, "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "res_records": G3.res_records(Z3.train_carves(split)),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM]:
        raise SystemExit(f"za10e: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za10e: A10e's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za10e: zg2xe's fit of {split} is {FIT[split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    G3.res_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") != ARM or "a10e" not in rec:
        raise SystemExit(f"za10e: {name} is not an A10e fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    G3.res_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zg2xe decided against zspx's card fits ──────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za10e: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspx's card fits scr-g5-zspx and scr-g5-zspx-hp)"),
       ("(rel's screen fits)", "(zspx's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspx R@5 | zg2xe R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A10e ({DECLARED}): zg2xe ({WHAT}) against zspx.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A10e", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za10e_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(screens, out):
    with on_base():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, base_screens=None):
    with on_base():
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def parent_beside(root=ROOT):
    """Beside, deciding nothing: zg2x's R@5 difference against the same zspx fits per read, from A10's re-call, so the
    gradient's effect reads directly."""
    p = Path(root) / A10.RECALL.relative_to(ROOT).with_suffix(".json")
    if not p.exists():
        return {}
    rows = json.loads(p.read_text(encoding="utf-8")).get("rows", [])
    return {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows}


def verdict(recall_file=None, out=None, root=ROOT):
    """A10e's screen verdict from zg2xe's re-call (section 4); webqsp and metaqa beside, and zg2x's deltas beside;
    the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A10e" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A10e's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    par = parent_beside(root)
    out_rec = {"stage": "A10e", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "parent_beside": par, "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za10e_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    def less(a, b):
        return fmt(a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "n/a"

    md = [f"# A10e's screen verdict: zg2xe **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zg2xe** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items()) or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspx per read; beside, deciding nothing: zg2x's (A10, its stack's z-score detached) "
          "and the two deltas' difference (rounded deltas):", "",
          "| read | zg2xe | zg2x (A10) | zg2xe - zg2x |", "| --- | --- | --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} | "
           f"{fmt(par.get(tag(r), {}).get('delta'))} {par.get(tag(r), {}).get('call', '')} | "
           f"{less(r.get('delta'), par.get(tag(r), {}).get('delta'))} |" for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za10e verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="A10e", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED if stage == "A10e" else "x", "arm": arm, "verdict": "x",
           "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


STEP_HEADS = A10.G2_HEADS                   # zspx's three step heads' last layers (sp_w2, rs_w2, p2_w2)


def code_of(fn):
    """A function's arguments and its statements after any docstring, unparsed (comments and layout dropped)."""
    import ast
    import inspect
    import textwrap
    f = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    body = f.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value,
                                                                                                str):
        body = body[1:]
    return ast.unparse(f.args), [ast.unparse(x) for x in body]


def grads(m, feats, keep, nq, B, bz, gw):
    """Every parameter's gradient of a fixed linear loss on the forward (None where none reaches)."""
    m.zero_grad(set_to_none=True)
    (m(feats, keep, nq, B, bz) * gw).sum().backward()
    return {k: None if p.grad is None else p.grad.detach().clone() for k, p in m.named_parameters()}


def through_z(m, feats, keep, nq, B, bz, gw):
    """The gradient that reaches zspx's parameters through the stack's z-score alone, built by hand: the stack's part
    of the same linear loss is differentiated by a detached copy of zspx's score that enters the stack only as its
    z-score (the hidden states detached too), and that is carried back through zspx's score."""
    m.zero_grad(set_to_none=True)
    m.out.g2_h, m.out.g2_n = None, 0
    s = super(G2.G2Mix, m).forward(feats, keep, nq, B, bz)
    H = m.out.g2_h
    m.out.g2_h, m.out.g2_n = None, 0
    s_z = s.detach().requires_grad_(True)
    r = g2_stack_e(m, H.detach(), s_z, feats[KT_KEY], feats[LK_KEY], nq, B, False, None)
    g_s, = torch.autograd.grad((TB * torch.tanh(r / TB) * gw).sum(), s_z)
    names = [k for k, _ in m.named_parameters() if not k.startswith("g2_")]
    ps = [dict(m.named_parameters())[k] for k in names]
    gs = torch.autograd.grad(s, ps, grad_outputs=g_s, allow_unused=True)
    return g_s, {k: (torch.zeros_like(p) if g is None else g) for k, p, g in zip(names, ps, gs)}


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm: zg2x's model with its own forward, on G3's carve; zg2x's, zspx's, zg2's and zspk's arms unchanged
    assert S.ARMS[ARM] == (ZG2XE, G3.ResKTLinkCarveBase)
    assert ZG2XE.__mro__[1:4] == (A10.ZG2X, G2.G2Mix, G5.ZSPX) and "__init__" not in vars(ZG2XE)
    assert S.ARMS[PARENT_ARM] == (A10.ZG2X, G3.ResKTLinkCarveBase) and "forward" not in vars(A10.ZG2X)
    assert S.ARMS[BASE_ARM] == (G5.ZSPX, G3.ResKTLinkCarveBase) and S.ARMS["zg2"] == (G2.ZG2, Z3.KTLinkCarveBase)
    assert S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert A9.BASE_ARM == BASE_ARM and BASE_SCREENS == {sp: f"outputs/a9/base-zspx-{sp}.json" for sp in SPLITS}
    assert BASE_FITS == A10.BASE_FITS and AGAINST == A10.AGAINST and LK_KEY == ZL.LK_KEY
    # 2. the code: g2_stack_e is G2's g2_stack but for its one detach; ZG2XE.forward is G2Mix.forward but for the stack
    #    it calls (and its super() spelled out)
    a0, b0 = code_of(G2.g2_stack)
    a1, b1 = code_of(g2_stack_e)
    assert a0 == a1 and sum(x.count("s.detach()") for x in b0) == 1 and not any("detach" in x for x in b1)
    assert [x.replace("s.detach()", "s") for x in b0] == b1 and sum(x != y for x, y in zip(b0, b1)) == 1
    a0, b0 = code_of(G2.G2Mix.forward)
    a1, b1 = code_of(ZG2XE.forward)
    assert a0 == a1 and [x.replace("super().forward(", "super(G2.G2Mix, self).forward(").replace(
        "g2_stack(", "g2_stack_e(") for x in b0] == b1
    tmp = Path(tempfile.mkdtemp(prefix="za10e_"))
    try:
        rng = np.random.default_rng(1015)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, _, _ = G3.toy_embed(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look", lk_root=tmp / "lk",
                  freeze="toy")
        G3.build("metaqa", "toy", 0, nodes=G3.ToyNodes(T), **kw)
        G3.build("metaqa", "toy", 1, nodes=G3.ToyNodes(T), **kw)

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_ch = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        c = G3.res_carve(Z3.kt_carve(cls_lk, tmp / "kt"), tmp / "res")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_ch = cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:7], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        assert feats[LK_KEY]["u"].numel() > 0 and KT_KEY in feats and G3.RES_KEY in feats
        keep = torch.ones(B, len(blocks))
        # 3. from the same seed zg2xe's weights are zg2x's bit for bit, and the global generator is where zg2x leaves it
        ms, after = {}, {}
        for k, cls in (("zspx", G5.ZSPX), ("zg2x", A10.ZG2X), ("zg2xe", ZG2XE)):
            torch.manual_seed(0)
            ms[k] = cls(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
            after[k] = torch.rand(4)
        assert torch.equal(after["zg2xe"], after["zg2x"])
        sd = {k: m.state_dict() for k, m in ms.items()}
        assert not LG.same_state(sd["zg2xe"], sd["zg2x"])
        stack = sorted(k for k in sd["zg2xe"] if k.startswith("g2_"))
        assert stack == sorted(G2.G2_NAMES) == sorted(set(sd["zg2xe"]) - set(sd["zspx"]))
        assert sum(int(sd["zg2xe"][k].numel()) for k in stack) == G2.n_params(16)
        g = torch.Generator().manual_seed(9)
        gw = torch.randn(nq.numel(), generator=torch.Generator().manual_seed(11))
        with torch.no_grad():
            # 4. the forward: zspx's step heads moved so all three steps act; at the start (readout zero) zg2xe is zg2x
            #    and zspx bit for bit; with the readout moved it is zg2x bit for bit and moves off zspx
            for nm in STEP_HEADS:
                w = torch.randn(1, getattr(ms["zspx"], nm).shape[1], generator=g)
                for m in ms.values():
                    getattr(m, nm).copy_(w)
            sx = ms["zspx"](feats, keep, nq, B, bz)
            y0 = ms["zg2xe"](feats, keep, nq, B, bz)
            assert torch.equal(y0, sx) and torch.equal(y0, ms["zg2x"](feats, keep, nq, B, bz))
        # 5. the gradient at the start: with the readout at zero no loss reaches the stack's z-score, so every
        #    parameter's gradient is zg2x's bit for bit
        ge, gx = grads(ms["zg2xe"], feats, keep, nq, B, bz, gw), grads(ms["zg2x"], feats, keep, nq, B, bz, gw)
        assert sorted(ge) == sorted(gx) and all((ge[k] is None) == (gx[k] is None) for k in ge)
        assert all(ge[k] is None or torch.equal(ge[k], gx[k]) for k in ge)
        assert float(ge["g2_out_w"].abs().max()) > 0
        with torch.no_grad():
            w_o = torch.randn(1, D, generator=g)
            for k in ("zg2x", "zg2xe"):
                ms[k].g2_out_w.copy_(w_o)
            y = ms["zg2xe"](feats, keep, nq, B, bz)
            assert torch.equal(y, ms["zg2x"](feats, keep, nq, B, bz))
            assert 1e-4 < float((y - sx).abs().max()) <= TB + 1e-5
        # 6. the gradient with the readout moved: the stack's own gradients are zg2x's bit for bit; zspx's parameters
        #    take zg2x's gradient plus the gradient through the z-score, built by hand (through_z), and nothing else;
        #    that extra gradient reaches zspx's scorer and each of its three step heads
        ge, gx = grads(ms["zg2xe"], feats, keep, nq, B, bz, gw), grads(ms["zg2x"], feats, keep, nq, B, bz, gw)
        assert all(torch.equal(ge[k], gx[k]) for k in stack)
        g_s, extra = through_z(ms["zg2xe"], feats, keep, nq, B, bz, gw)
        assert float(g_s.abs().max()) > 1e-4
        own = sorted(extra)
        assert own == sorted(set(ge) - set(stack)) == sorted(k for k, _ in ms["zspx"].named_parameters())
        for k in own:
            if gx[k] is None:
                assert ge[k] is None and float(extra[k].abs().max()) == 0.0, k
                continue
            assert torch.allclose(ge[k], gx[k] + extra[k], atol=1e-6, rtol=1e-5), (k, float(
                (ge[k] - gx[k] - extra[k]).abs().max()))
        moved = [k for k in own if gx[k] is not None and float((ge[k] - gx[k]).abs().max()) > 1e-6]
        assert all(nm in moved for nm in STEP_HEADS), moved
        assert any(not k.startswith(("sp_", "rs_", "p2_")) for k in moved), moved
        with torch.no_grad():
            # 7. without links it is zspx's scorer, as zg2x is; in training the stack's dropout draws from its own
            #    generator, as zg2x's, and the forward is zg2x's bit for bit under the same global seed
            no_lk = {k: x for k, x in feats.items() if k not in (LK_KEY, G3.RES_KEY)}
            y_n = ms["zg2xe"](no_lk, keep, nq, B, bz)
            assert torch.equal(y_n, ms["zspx"](no_lk, keep, nq, B, bz))
            assert torch.equal(y_n, ms["zg2x"](no_lk, keep, nq, B, bz))
            for k in ("zg2x", "zg2xe"):
                ms[k].train()
            g0 = ms["zg2xe"].g2_gen.get_state()
            assert torch.equal(g0, ms["zg2x"].g2_gen.get_state())
            torch.manual_seed(5)
            y_tr = ms["zg2xe"](feats, keep, nq, B, bz)
            torch.manual_seed(5)
            assert torch.equal(y_tr, ms["zg2x"](feats, keep, nq, B, bz))
            assert not torch.equal(ms["zg2xe"].g2_gen.get_state(), g0)
            assert torch.equal(ms["zg2xe"].g2_gen.get_state(), ms["zg2x"].g2_gen.get_state())
            for k in ("zg2x", "zg2xe"):
                ms[k].eval()
        # 8. toy fits through lean_gpu's loop: a repeat is bit for bit; the stack's readout learns; the fit moves away
        #    from zg2x's on zspx's own weights (the gradient through the z-score); without links the fit is zspx's bit
        #    for bit on every shared key, the stack's readout at zero
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zg2xe")
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zg2xe repeat")
            bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zg2xe, no links")
        with S.patched(PARENT_ARM):
            par = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zg2x")
        with S.patched(BASE_ARM):
            base_bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx, no links")
        assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]], rep["states"] + [rep["swa"]]))
        assert float(fit["swa"]["g2_out_w"].abs().max()) > 0
        assert sorted(fit["swa"]) == sorted(par["swa"])
        assert any(float((fit["swa"][k] - par["swa"][k]).abs().max()) > 0 for k in own)
        for a, b in zip(bare["states"] + [bare["swa"]], base_bare["states"] + [base_bare["swa"]]):
            assert not LG.same_state({k: a[k] for k in b}, b)
            assert float(a["g2_out_w"].abs().max()) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 9. the refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    for wrong in (BASE_ARM, PARENT_ARM, "zg2"):
        SR.must_stop(train, ["train", "--arm", wrong, "--name", FIT["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", A10.FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": BASE_ARM, "g5": {}}), ("w", {"arm": ARM}), ("x", {"arm": PARENT_ARM, "a10": {}}),
                        ("v", {"arm": PARENT_ARM, "a10e": {}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 10. relz's pair and re-call against zspx's card fits, stamped A10e's; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        null1 = SR.fake_null(T_ / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        bs = {sp: str(SR.fake_compare(T_ / BASE_FITS[sp][0], BASE_FITS[sp][2], sp,
                                      {ds: 0.1 for ds in LG.EVAL_ORDER}, arm=BASE_ARM)) for sp in BASE_FITS}
        zb = {sp: "/".join(("outputs",) + BASE_FITS[sp]) for sp in BASE_FITS}
        za = SR.fake_compare(T_ / ARM, FIT["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T_ / ARM, FIT["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T_ / "p")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == BASE_ARM, pr["verdict"]
        pj = json.loads((T_ / "p.json").read_text(encoding="utf-8"))
        assert pj["stage"] == "A10e" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        rj = json.loads((T_ / "r.json").read_text(encoding="utf-8"))
        assert rj["stage"] == "A10e" and rj["arm"] == ARM and rj["declared_in"] == DECLARED
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A10e ({DECLARED})") and f"| zspx R@5 | zg2xe R@5 ({WHAT}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 11. the verdict's outcomes and next forms; A10's own re-call is refused; zg2x's deltas beside
        rp = T_ / "rc.json"
        gm, lw = ("L-musique", "musique"), ("L-hotpotqa", "webqsp")
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v", root=T_)
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and v["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert v["parent_beside"] == {}
        v = verdict(fake_recall(rp, {gm: "GAIN", lw: "LOSS"}), T_ / "v", root=T_)
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["GAIN_AND_LOSS"]
        v = verdict(fake_recall(rp, {}), T_ / "v", root=T_)
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(fake_recall(rp, {lw: "LOSS"}), T_ / "v", root=T_)
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(A10.fake_recall(rp, {gm: "GAIN"}), T_ / "v", root=T_)
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"]
        v = verdict(fake_recall(rp, {gm: "GAIN"}, rows=[{"split": "L-musique", "dataset": "musique", "call": "GAIN"}]),
                    T_ / "v", root=T_)
        assert v["verdict"] == "INCOMPLETE"
        assert verdict(T_ / "none.json", T_ / "v", root=T_)["verdict"] == "INCOMPLETE"
        a10r = T_ / A10.RECALL.relative_to(ROOT).with_suffix(".json")
        a10r.parent.mkdir(parents=True, exist_ok=True)
        A10.fake_recall(a10r, {lw: "LOSS"})
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v", root=T_)
        assert v["parent_beside"]["L-hotpotqa webqsp"]["call"] == "LOSS" and len(v["parent_beside"]) == 12
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        vm = (T_ / "v.md").read_text(encoding="utf-8")
        assert "Next form" in vm and "| read | zg2xe | zg2x (A10) | zg2xe - zg2x |" in vm
    log(f"za10e selftest: zg2xe is zg2x's model with its own forward on G3's carve; g2_stack_e is G2's g2_stack but for "
        f"its one detach and the forward G2Mix's but for that stack (syntax trees compared); its weights and forward "
        f"are zg2x's bit for bit (eval, readout zero or moved, training draws, without links); at the start every "
        f"gradient is zg2x's bit for bit; with the readout moved the stack's gradients are zg2x's bit for bit and "
        f"zspx's are zg2x's plus the gradient through the z-score built by hand, reaching the scorer and all three "
        f"step heads; toy fits: repeats identical, the stack learns, zspx's weights move off zg2x's, without links the "
        f"fit is zspx's bit for bit; train and read refusals; relz's pair and re-call stamped A10e's; the verdict's "
        f"outcomes, next forms and zg2x beside ({time.time() - t0:.1f}s): ok")
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
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(null, g.pair, g.out)
        except SystemExit as e:
            log(f"za10e {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za10e: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
