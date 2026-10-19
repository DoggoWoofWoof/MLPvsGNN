"""A9t (docs/A9T_BOUNDED_DROPOUT.md): the next form A9's verdict named for no GAIN, the MP-unified track's whole form on
G5's zspx. A9's zspxd (zspx trained with qd_gnn13's edge-family dropout) was NOT_ADOPTED with 0 GAIN and 0 LOSS. The
track's arms carried that dropout together with a bound on what message passing adds to a score (qd_gnn13's tb5, the
bound G2 put on its readout). Here both act on the adopted GNN:

    zspxdt  zspxd (outputs/mp_unified/za9fd.py), its dropout unchanged, with the score after zspx's three steps, s3,
            replaced by
                s0 + 0.5 tanh((s3 - s0) / 0.5)
            where s0 is the score before the first step (zspk's scorer and KT head: the value zg5mix.ZSPX.forward
            assigns on its first line) and s3 the score that forward returns. The three increments are bounded
            together, in training and at read; the steps themselves read the unbounded s0, s1 and s2, as in zspx.

No parameter is added and no draw moves: the dropout's draws are zspxd's (its own generator, seed + 9901). At the start
(every step head's last layer zero) s3 = s0, so zspxdt is zspx bit for bit; with the bound off it is zspxd bit for bit.
GNN only; the MLP takes no message.

    python outputs/mp_unified/za9t.py train --split L-musique --name scr-a9t-zspxdt --arm zspxdt --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9t.py read --name scr-a9t-zspxdt --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9t.py compare --new outputs/screen/fits/scr-a9t-zspxdt \\
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a9t-zspxdt
    python outputs/mp_unified/za9t.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za9t.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za9t.py verdict
    python outputs/mp_unified/za9t.py --selftest
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

import za9fd as A9  # noqa: E402  (zspxd and its dropout; through it zg5mix, zg2, zg3res, zg4prop, zp1c3, zprop, zlink)

import numpy as np  # noqa: E402
import torch  # noqa: E402

G5, G2 = A9.G5, A9.G2
G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
ZP, ZL, ZM, RM = G5.ZP, G5.ZL, G5.ZM, G5.RM
Z, R, S2, S, LG, LC, SR = G5.Z, G5.R, G5.S2, G5.S, G5.LG, G5.LC, G5.SR
log = S.log
FAMS = ZL.FAMS

DECLARED = "docs/A9T_BOUNDED_DROPOUT.md"
OUT = ROOT / "outputs" / "a9t"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zspxdt", "zspx"
TB = G2.TB                      # 0.5: qd_gnn13's tb5 (qd_gnn7's -tb<T>, t = T / 10), G2's readout bound
FIT = {"L-musique": "scr-a9t-zspxdt", "L-hotpotqa": "scr-a9t-zspxdt-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-g5-zspx"), "L-hotpotqa": ("screen", "fits", "scr-g5-zspx-hp")}
BASE_SCREENS = {sp: f"outputs/a9/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}     # A9's base compares
AGAINST = "zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp)"
WHAT = "zspx trained with edge-family dropout, its three increments bounded together"
SETTINGS = {"base": "zspxd (outputs/mp_unified/za9fd.py): zspx with A9's edge-family dropout, every setting, seed, "
                    "batch, head and draw of it",
            "dropout": A9.SETTINGS["dropout"],
            "bound": {"form": "s0 + TB tanh((s3 - s0) / TB)", "width": TB,
                      "rule": "qd_gnn13's tb5 (qd_gnn7's -tb rule, t = 5 / 10), G2's readout bound (zg2.TB)",
                      "s0": "the score before zspx's first step: what zg5mix.ZSPX.forward's first line assigns, "
                            "super(ZP.ZProp, self).forward (zspk's scorer and KT head)",
                      "s3": "the score zg5mix.ZSPX.forward returns: after zspk's step (gradient kept), G3's residual "
                            "message and G4's second step",
                      "steps": "each step reads the unbounded score before it (s0, s1, s2), as in zspx",
                      "gradient": "kept through s0 and s3, no detach", "where": "training and read"},
            "parameters_added": 0}
PAIR = SCREEN / "scr-a9t-zspxdt-pair"
RECALL = SCREEN / "scr-a9t-zspxdt-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspxdt's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A9t's, or not twelve reads "
        "(docs/A9T_BOUNDED_DROPOUT.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zspxdt fitted again on P1g-b's pools, then its full run beside zspx's, its own declared "
                              "file",
        "GAIN_AND_LOSS": "A9tq, a bound that backs off per question: the width 0.5 * 2 sigmoid(a . m_q + b), m_q the "
                         "shares of the question's link edges in the three families, a and b zero at the start (so "
                         "A9tq starts as A9t), the dropout kept; one declared file",
        "NO_GAIN": "A9b, the bound alone: zspx's three increments bounded together at 0.5 with no dropout, the one part "
                   "of the track's form not yet on zspx; one declared file. If A9b finds no GAIN either, the MP-unified "
                   "track's form closes on the GNN: its dropout, its bound and both together will have reached it, and "
                   "the MLP takes no message",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the model ────────────────────────────────────────────────────────────────


def bound(s0, s3, tb):
    """s0 + tb tanh((s3 - s0) / tb): the move s3 - s0 kept below tb either way, near the identity where it is small
    (qd_gnn7's -tb rule, here over zspx's three steps together)."""
    return s0 + tb * torch.tanh((s3 - s0) / tb)


class BoundedSteps(G5.ZSPX):
    """zspx's forward (zg5mix.ZSPX.forward) with s0 kept and its returned score s3 bounded around s0; TB None: zspx's."""

    def forward(self, feats, keep, nq, B, base_z):
        s0 = super(ZP.ZProp, self).forward(feats, keep, nq, B, base_z)     # zg5mix's first line: zspk's scorer, KT head
        if ZL.LK_KEY not in feats:
            return s0
        lk = feats[ZL.LK_KEY]
        s = s0 + self.prop_head(G4.prop_inputs_grad(s0, lk, nq, B))       # s1: zspk's step, its gradient kept
        if G3.RES_KEY in feats:
            s = s + self.rs_head(G3.res_inputs(s, lk, feats[G3.RES_KEY], nq, B))      # s2: G3's residual message
        s3 = s + self.p2_head(ZP.prop_inputs(s, lk, nq, B))               # s3: G4's second step, zg5mix's return
        return s3 if TB is None else bound(s0, s3, TB)


class ZSPXDT(A9.ZSPXD, BoundedSteps):
    """zspxdt: zspxd (its dropout, generator and weights unchanged) over the bounded steps. MRO ZSPXD, BoundedSteps,
    ZSPX: zspxd's forward drops the families in training and hands the kept edges to the bounded steps."""


ARMS = {ARM: (ZSPXDT, G3.ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za9t_sha256": LC.sha_src(__file__), "za9fd_sha256": LC.sha_src(A9.__file__),
                    "zg5mix_sha256": LC.sha_src(G5.__file__), "zg2_sha256": LC.sha_src(G2.__file__),
                    "zg3res_sha256": LC.sha_src(G3.__file__), "zg4prop_sha256": LC.sha_src(G4.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "a9t": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE_ARM, "device": Z2.DEVICE,
                        "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "res_records": G3.res_records(Z3.train_carves(split)),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM]:
        raise SystemExit(f"za9t: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za9t: A9t's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za9t: zspxdt's fit of {split} is {FIT[split]}, not {name}")
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
    if rec.get("arm") != ARM or "a9t" not in rec:
        raise SystemExit(f"za9t: {name} is not an A9t fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    G3.res_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspxdt decided against zspx's card fits ─────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za9t: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspx's card fits scr-g5-zspx and scr-g5-zspx-hp)"),
       ("(rel's screen fits)", "(zspx's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspx R@5 | zspxdt R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A9t ({DECLARED}): zspxdt ({WHAT}) against zspx.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A9t", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za9t_sha256": LC.sha_src(__file__)})
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


def a9_beside(root=ROOT):
    """Beside, deciding nothing: zspxd's R@5 difference against zspx per read (the dropout alone), from A9's own
    re-call; empty when that re-call is absent or not A9's."""
    p = Path(root) / A9.RECALL.relative_to(ROOT).with_suffix(".json")
    if not p.exists():
        return {}
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("stage") != "A9" or rec.get("arm") != A9.ARM:
        return {}
    return {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rec.get("rows", [])}


def verdict(recall_file=None, out=None, root=ROOT):
    """A9t's screen verdict from zspxdt's re-call (section 4); webqsp and metaqa beside, and zspxd's reads from A9's
    re-call; the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A9t" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A9t's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    a9 = a9_beside(root)
    out_rec = {"stage": "A9t", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "zspxd_beside": a9, "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za9t_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    md = [f"# A9t's screen verdict: zspxdt **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspxdt** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items()) or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspx per read; zspxd's (A9, the dropout alone) beside, deciding nothing:", "",
          "| read | zspxdt | zspxd (A9) |", "| --- | --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} | {fmt(a9.get(tag(r), {}).get('delta'))} "
           f"{a9.get(tag(r), {}).get('call') or ''} |" for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za9t verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="A9t", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED if stage == "A9t" else "x", "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


@contextlib.contextmanager
def width(tb):
    """The bound's width for a selftest; None takes the bound off (zspxd's forward)."""
    global TB
    saved = TB
    TB = tb
    try:
        yield
    finally:
        TB = saved


def grads(m, feats, keep, nq, B, bz, w):
    """Every parameter's gradient of sum(w * m's score); zero where the forward does not reach it."""
    m.zero_grad(set_to_none=True)
    (m(feats, keep, nq, B, bz) * w).sum().backward()
    return {k: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for k, p in m.named_parameters()}


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm: zspxd's model and dropout over the bounded steps, on G3's carve; zspxd's, zspx's and zspk's arms
    #    unchanged; the width is G2's (qd_gnn13's tb5), the rate A9's; the base fits and base screens are A9's
    assert S.ARMS[ARM] == (ZSPXDT, G3.ResKTLinkCarveBase)
    assert ZSPXDT.__mro__[1:4] == (A9.ZSPXD, BoundedSteps, G5.ZSPX) and BoundedSteps.__mro__[1] is G5.ZSPX
    assert S.ARMS[A9.ARM] == (A9.ZSPXD, G3.ResKTLinkCarveBase) and S.ARMS[BASE_ARM] == (G5.ZSPX, G3.ResKTLinkCarveBase)
    assert S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert TB == G2.TB == SETTINGS["bound"]["width"] == 0.5 and G2.FD == SETTINGS["dropout"]["rate"] == 0.25
    assert FAMS == 3 and BASE_FITS == A9.BASE_FITS and BASE_SCREENS == A9.BASE_SCREENS
    # 2. the bound by hand, and its gradient: d/ds3 = 1 - tanh^2, d/ds0 = tanh^2
    mv = [-3.0, -0.6, -0.01, 0.0, 0.01, 0.6, 3.0]
    s0h = torch.tensor([0.3, -1.2, 2.0, 0.7, -0.4, 1.1, -2.5], dtype=torch.float64, requires_grad=True)
    s3h = (s0h.detach() + torch.tensor(mv, dtype=torch.float64)).requires_grad_(True)
    bh = bound(s0h, s3h, TB)
    want = [a + TB * math.tanh(d / TB) for a, d in zip(s0h.tolist(), mv)]
    assert max(abs(x - y) for x, y in zip(bh.tolist(), want)) < 1e-12
    bh.sum().backward()
    th = [math.tanh(d / TB) for d in mv]
    assert max(abs(g - (1 - t * t)) for g, t in zip(s3h.grad.tolist(), th)) < 1e-12
    assert max(abs(g - t * t) for g, t in zip(s0h.grad.tolist(), th)) < 1e-12
    tmp = Path(tempfile.mkdtemp(prefix="za9t_"))
    try:
        rng = np.random.default_rng(99)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, _, _ = G3.toy_embed(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look", lk_root=tmp / "lk",
                  freeze="toy")
        G3.build("metaqa", "toy", 0, nodes=G3.ToyNodes(T), **kw)
        G3.build("metaqa", "toy", 1, nodes=G3.ToyNodes(T), **kw)
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        c = G3.res_carve(cls_lk, tmp / "res")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_ch = cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:7], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        lk, rs = feats[ZL.LK_KEY], feats[G3.RES_KEY]
        assert lk["u"].numel() > 0 and rs["s"].numel() == lk["u"].numel() and rs["c"].numel() == nq.numel()
        no_lk = {k: x for k, x in feats.items() if k not in (ZL.LK_KEY, G3.RES_KEY)}
        keep = torch.ones(B, len(blocks))
        # 3. zspxdt's state and generator are zspxd's (its weights zspx's) from the same seed: no parameter added
        nets = {}
        for k, cls in (("zspx", G5.ZSPX), ("zspxd", A9.ZSPXD), ("zspxdt", ZSPXDT)):
            torch.manual_seed(0)
            nets[k] = cls(blocks, c.widths, 16, dropout=0.1, seed=0)
        mx, md, mt = nets["zspx"], nets["zspxd"], nets["zspxdt"]
        assert not LG.same_state(mt.state_dict(), mx.state_dict()) and not LG.same_state(mt.state_dict(),
                                                                                         md.state_dict())
        assert torch.equal(mt.a9_gen.get_state(), md.a9_gen.get_state())
        with torch.no_grad():
            for m in nets.values():
                m.eval()
            # at the start every step head's last layer is zero, so s3 = s0: zspxdt is zspx, and zspk's scorer
            y0 = mx(feats, keep, nq, B, bz)
            assert torch.equal(mt(feats, keep, nq, B, bz), y0) and torch.equal(y0, mx(no_lk, keep, nq, B, bz))
            # 4. all three heads moved, small and large: with the bound off zspxdt is zspx bit for bit (the steps as
            #    zg5mix runs them); on, it is s0 + 0.5 tanh((s3 - s0) / 0.5) bit for bit, s0 zspx's scorer and s3
            #    zspx's score; every move shrinks toward s0, below 0.5, by at most |d|^3 / (3 * 0.25)
            g = torch.Generator().manual_seed(9)
            w2 = {nm: torch.randn(1, getattr(mx, nm).shape[1], generator=g) for nm in ("sp_w2", "rs_w2", "p2_w2")}
            for scale in (1e-3, 1.0):
                for nm, w in w2.items():
                    for m in nets.values():
                        getattr(m, nm).copy_(w * scale)
                with width(None):
                    assert torch.equal(mt(feats, keep, nq, B, bz), mx(feats, keep, nq, B, bz))
                s0, s3 = mx(no_lk, keep, nq, B, bz), mx(feats, keep, nq, B, bz)
                y = mt(feats, keep, nq, B, bz)
                assert torch.equal(y, bound(s0, s3, TB))
                d, mo = s3 - s0, y - s0
                assert bool((mo * d >= 0).all()) and bool((mo.abs() <= TB + 1e-4).all())
                assert bool((mo.abs() <= d.abs() + 1e-5).all())
                assert bool((d.abs() - mo.abs() <= d.abs() ** 3 / (3 * TB * TB) + 1e-5).all())
                if scale < 1:
                    assert 0 < float(d.abs().max()) < 0.05, float(d.abs().max())
                    assert float((y - s3).abs().max()) < 0.01 * float(d.abs().max())
                    continue
                big = d.abs() > 1.0
                assert int(big.sum()) > 0 and bool((mo.abs()[big] > TB * math.tanh(2.0) - 1e-5).all())
                assert float((y - s3).abs().max()) > 0.5
                # the three increments are bounded together, not each: the steps read the unbounded s1 and s2
                mx.rs_w2.zero_()
                mx.p2_w2.zero_()
                s1 = mx(feats, keep, nq, B, bz)
                mx.rs_w2.copy_(w2["rs_w2"] * scale)
                s2 = mx(feats, keep, nq, B, bz)
                mx.p2_w2.copy_(w2["p2_w2"] * scale)
                assert torch.equal(mx(feats, keep, nq, B, bz), s3)
                each = s0 + sum(TB * torch.tanh((b - a) / TB) for a, b in ((s0, s1), (s1, s2), (s2, s3)))
                assert float((each - y).abs().max()) > 1e-2
        # the gradient reaches the scorer through s0 in full and through the steps scaled by 1 - tanh^2:
        # d(sum w y) = d(sum w (1 - sech2) s0) + d(sum w sech2 s3), every parameter; the heads learn less than zspx's
        gw = torch.randn(*s3.shape, generator=torch.Generator().manual_seed(11))
        sech2 = 1 - torch.tanh((s3 - s0) / TB) ** 2
        gt = grads(mt, feats, keep, nq, B, bz, gw)
        g0 = grads(mx, no_lk, keep, nq, B, bz, gw * (1 - sech2))
        g3 = grads(mx, feats, keep, nq, B, bz, gw * sech2)
        gx = grads(mx, feats, keep, nq, B, bz, gw)
        assert sorted(gt) == sorted(gx) and all(torch.allclose(gt[k], g0[k] + g3[k], atol=1e-5, rtol=1e-4) for k in gt)
        assert all(float((gt[k] - gx[k]).abs().max()) > 1e-4 for k in ("sp_w2", "rs_w2", "p2_w2"))
        with torch.no_grad():
            # 5. in training: the bound over zspx over the kept edges, the draws from zspxd's own generator, the
            #    global RNG (the scorer's dropout) seeded alike; with the bound off zspxd bit for bit; with FD 0 the
            #    bound over zspx with every edge; at read no draw
            for m in nets.values():
                m.train()
            for _ in range(20):             # the generator's next draw that drops some edges, not all
                g0s = mt.a9_gen.get_state()
                u01 = torch.rand(B, FAMS, generator=mt.a9_gen)
                ke = G2.family_keep(lk["f"], nq[lk["v"]], B, u01)
                if 0 < int(ke.sum()) < ke.numel():
                    break
            assert 0 < int(ke.sum()) < ke.numel()
            mt.a9_gen.set_state(g0s)
            torch.manual_seed(7)
            yt = mt(feats, keep, nq, B, bz)
            torch.manual_seed(7)
            s3k = mx(A9.drop_families(feats, nq, B, u01), keep, nq, B, bz)
            torch.manual_seed(7)
            s0k = mx(no_lk, keep, nq, B, bz)
            torch.manual_seed(7)
            assert torch.equal(yt, bound(s0k, s3k, TB)) and not torch.equal(s3k, mx(feats, keep, nq, B, bz))
            with width(None):
                mt.a9_gen.set_state(g0s)
                md.a9_gen.set_state(g0s)
                torch.manual_seed(7)
                ya = mt(feats, keep, nq, B, bz)
                torch.manual_seed(7)
                assert torch.equal(ya, md(feats, keep, nq, B, bz)) and torch.equal(ya, s3k)
            with A9.rate(0.0):
                torch.manual_seed(7)
                yf = mt(feats, keep, nq, B, bz)
                torch.manual_seed(7)
                s3f = mx(feats, keep, nq, B, bz)
                assert torch.equal(yf, bound(s0k, s3f, TB))
            mt.eval()
            g1 = mt.a9_gen.get_state()
            mt(feats, keep, nq, B, bz)
            assert torch.equal(mt.a9_gen.get_state(), g1)
        # 6. toy fits through lean_gpu's loop: a repeat is bit for bit and the bound moves the fit off zspxd's; with
        #    the bound off the fit is zspxd's bit for bit, with FD 0 too zspx's; without links zspx's
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxdt")
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxdt repeat")
            with width(None):
                off = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxdt, bound off")
                with A9.rate(0.0):
                    off0 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxdt, bound off, FD 0")
            bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxdt, no links")
        with S.patched(A9.ARM):
            dfit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxd")
        with S.patched(BASE_ARM):
            base = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx")
            base_bare = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx, no links")
        pairs = lambda a, b: zip(a["states"] + [a["swa"]], b["states"] + [b["swa"]])  # noqa: E731
        assert all(not LG.same_state(a, b) for a, b in pairs(fit, rep))
        assert all(not LG.same_state(a, b) for a, b in pairs(off, dfit))
        assert all(not LG.same_state(a, b) for a, b in pairs(off0, base))
        assert all(not LG.same_state(a, b) for a, b in pairs(bare, base_bare))
        assert LG.same_state(fit["swa"], dfit["swa"]), "the bound did not move the toy fit"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. the refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", A9.ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": A9.ARM, "a9": {}}), ("w", {"arm": ARM}), ("x", {"arm": ARM, "a9": {}}),
                        ("z", {"arm": A9.ARM, "a9t": {}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 8. relz's pair and re-call against zspx's card fits, stamped A9t's; relz restored after
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
        assert pj["stage"] == "A9t" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A9t ({DECLARED})") and f"| zspx R@5 | zspxdt R@5 ({WHAT}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 9. the verdict's outcomes and next forms; zspxd's reads beside from A9's own re-call
        rp = T_ / "rc.json"
        gm, lw = ("L-musique", "musique"), ("L-hotpotqa", "webqsp")
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v", root=T_)
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and v["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert v["zspxd_beside"] == {}
        v = verdict(fake_recall(rp, {gm: "GAIN", lw: "LOSS"}), T_ / "v", root=T_)
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["GAIN_AND_LOSS"]
        v = verdict(fake_recall(rp, {}), T_ / "v", root=T_)
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(fake_recall(rp, {lw: "LOSS"}), T_ / "v", root=T_)
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(fake_recall(rp, {gm: "GAIN"}, stage="A9", arm=A9.ARM), T_ / "v", root=T_)
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"]
        v = verdict(fake_recall(rp, {gm: "GAIN"}, rows=[{"split": "L-musique", "dataset": "musique", "call": "GAIN"}]),
                    T_ / "v", root=T_)
        assert v["verdict"] == "INCOMPLETE"
        assert verdict(T_ / "none.json", T_ / "v", root=T_)["verdict"] == "INCOMPLETE"
        a9r = T_ / A9.RECALL.relative_to(ROOT).with_suffix(".json")
        a9r.parent.mkdir(parents=True, exist_ok=True)
        A9.fake_recall(a9r, {gm: "GAIN"})
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v", root=T_)
        assert v["zspxd_beside"]["L-musique musique"]["call"] == "GAIN" and len(v["zspxd_beside"]) == 12
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "Next form" in (T_ / "v.md").read_text(encoding="utf-8")
    log(f"za9t selftest: zspxdt is zspxd over the bounded steps (MRO ZSPXD, BoundedSteps, ZSPX), no parameter added; "
        f"the bound and its gradient by hand; at the start zspxdt is zspx bit for bit; with the heads moved (small and "
        f"large) it is zspx with the bound off and s0 + 0.5 tanh((s3 - s0) / 0.5) with it on, bit for bit, every move "
        f"shrunk toward s0 and below 0.5, the three increments bounded together; the gradient through s0 in full and "
        f"the steps scaled by 1 - tanh^2; in training the bound over zspx over the kept edges (zspxd's draws), zspxd "
        f"with the bound off, FD 0 over every edge, no draw at read; toy fits: repeats identical, the bound moves the "
        f"fit, bound off is zspxd's fit and with FD 0 zspx's, no links zspx's; train and read refusals; relz's pair "
        f"and re-call stamped A9t's; the verdict's outcomes and next forms, zspxd beside ({time.time() - t0:.1f}s): ok")
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
            log(f"za9t {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za9t: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
