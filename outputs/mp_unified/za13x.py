"""A13x (docs/A13X_BRIDGE_HEAD_ZSPX.md): zbr's bridge head on zspx, the next form A13's verdict named (NO_GAIN). A13 put
zbr's head on zspk, before its one propagation step; after the seed-null re-call it had no GAIN and no LOSS (its two
filed GAINs, musique and hotpotqa zero-shot, sat inside the null's floors). zspk's step reads the score without the
gradient, so the head never learned through the step. zspx (G5, the adopted GNN) keeps the gradient through its first
step and adds two more (G3's residual message, G4's second step). Here:

    zspxb   zspx (zg5mix.ZSPX on G3's carve) whose score, before zspx's first step reads it, adds zbr's head over zbr's
            13 per-row inputs (zbr.bridge_inputs, unchanged: each row's links to RRF's top 1, 2 and 5 per family,
            two-step paths from the top 5, the bridge flag, the bridge rank by dense_cos among flagged rows, the source
            rank). The head is A13's BrMix, unchanged (zbr's: Linear(13, 32), GELU, Linear(32, 1), the first layer from
            its own generator, seed + 2901, the last at zero). So at the start zspxb is zspx bit for bit; once the head
            moves, all three steps read the bridge-shifted score, and the head learns through the first.

The inputs are fixed before any row is scored (RRF's order, zlink's pool edges, the cached dense_cos column); the head
reads no score. The carve is G3's (zspx's) with each row's dense_cos beside its links (A13's cos_carve, unchanged),
every block, row, edge, KT column, residual and gold unchanged; no build. GNN only: zbr's MLP form closed on zrc
(round 29, NO_GAIN), and the MLP gets the head again only after a gain here (section 5).

    python outputs/mp_unified/za13x.py train --split L-musique --name scr-a13x-zspxb --arm zspxb --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za13x.py read --name scr-a13x-zspxb --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za13x.py compare --new outputs/screen/fits/scr-a13x-zspxb \\
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a13x-zspxb
    python outputs/mp_unified/za13x.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za13x.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za13x.py verdict
    python outputs/mp_unified/za13x.py --selftest

Each read is decided against zspx's card fit of its split (scr-g5-zspx, scr-g5-zspx-hp); step 1's fit beside. The
re-call's base screens are A9's (outputs/a9/base-zspx-<split>.json: zspx's card fits against step 1's).

Speed: per question, two sorts of its pool (RRF's order, the flagged rows' dense_cos) and three passes over its pool's
edges, all from fixed inputs; zspx's steps already make such passes. Any latency figure for zspxb is cold, timed in its
own declared stage.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zg5mix as G5  # noqa: E402  (zspx, its carve, its fits; through it zg3res, zg4prop, zp1c3, zprop, zlink)
import za13br as A13  # noqa: E402  (A13: zbr's head as BrMix, the dense_cos carve; through it zbr)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
ZP, ZL, ZM, RM = G5.ZP, G5.ZL, G5.ZM, G5.RM
Z, R, S2, S, LG, LC, SR = G5.Z, G5.R, G5.S2, G5.S, G5.LG, G5.LC, G5.SR
ZB = A13.ZB
log = S.log

DECLARED = "docs/A13X_BRIDGE_HEAD_ZSPX.md"
OUT = ROOT / "outputs" / "a13x"
SCREEN = ROOT / "outputs" / "screen"
LK_KEY, COS = ZL.LK_KEY, ZB.COS
BR = ZB.BR
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zspxb", "zspx"
FIT = {"L-musique": "scr-a13x-zspxb", "L-hotpotqa": "scr-a13x-zspxb-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-g5-zspx"), "L-hotpotqa": ("screen", "fits", "scr-g5-zspx-hp")}
BASE_SCREENS = {sp: f"outputs/a9/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}
AGAINST = "zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp)"
WHAT = "zspx with zbr's bridge head added to its score before its first step"
SETTINGS = {"base": "zspx (outputs/mp_unified/zg5mix.py), its model, carve, card, seed and batches unchanged",
            "inputs": f"zbr.bridge_inputs, unchanged: {ZB.N_IN} per row from RRF's order (the batch's base_z), zlink's "
                      "pool edges and the row's cached dense_cos",
            "head": f"A13's BrMix, unchanged (zbr's): Linear({ZB.N_IN}, {ZB.HID}), GELU, Linear({ZB.HID}, 1); first "
                    f"layer uniform in +-1/sqrt({ZB.N_IN}) from its own generator (seed + {ZB.SEED_OFF}), last layer "
                    "zero",
            "where": "added to zspx's pre-step score (zrm's plus the KT head) before zspx's first step reads it; the "
                     "first step keeps the gradient, G3's residual message and G4's second step read the shifted score "
                     "without it; MRO ZSPX, ZSPR, ZSPK, ZProp, BrMix, KTMix, ZRM",
            "carve": "G3's carve (P1c3's KT link carve with c and s beside) with each row's dense_cos beside its links "
                     "(A13's cos_carve); no build",
            "parameters_added": ZB.HID * ZB.N_IN + ZB.HID + ZB.HID + 1}
PAIR = SCREEN / "scr-a13x-zspxb-pair"
RECALL = SCREEN / "scr-a13x-zspxb-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspxb's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A13x's, or not twelve "
        "reads (docs/A13X_BRIDGE_HEAD_ZSPX.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zspxb fitted again on P1g-b's pools, then its full run beside zspx's; and A13xm, zbr's "
                              "head on zrckpr (R1's MLP, ADOPT_FOR_FULL_RUN at its screen; zbr's inputs are fixed "
                              "before any row is scored), since zbr's MLP form closed on zrc before the MLP's model "
                              "changed; each its own declared file",
        "GAIN_AND_LOSS": "A13xb, the back-off form: the bridge head bounded (0.2 tanh(h / 0.2), the MP-unified "
                         "track's bound) on zspx, so the steps cannot lean on it where the bridge rule fails (squad, "
                         "comparison questions); one declared file",
        "NO_GAIN": "A13xs, the columns inside the step: zbr's 13 inputs read by zspx's first step head beside "
                   "zprop's 10 (its first layer widened, the new weights zero at the start), so the step weighs a "
                   "row's neighbours by the row's bridge quantities instead of adding a shift (P1g-c's zspxs form for "
                   "R1's columns); one declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the carve: zspx's (G3's), with each row's dense_cos beside its links ─────


CosResKTLinkCarveBase = A13.cos_carve(G3.ResKTLinkCarveBase)


# ── the model ────────────────────────────────────────────────────────────────


class ZSPXB(G5.ZSPX, A13.BrMix):
    """zspxb: zspx plus zbr's bridge head, added before zspx's first step reads the score (MRO ZSPX, ZSPR, ZSPK, ZProp,
    BrMix, KTMix, ZRM: zspx's forward asks for what ZProp's super() returns, which here is BrMix's, the KT score plus
    the bridge head)."""


ARMS = {ARM: (ZSPXB, CosResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za13x_sha256": LC.sha_src(__file__), "za13br_sha256": LC.sha_src(A13.__file__),
                    "zbr_sha256": LC.sha_src(ZB.__file__), "zg5mix_sha256": LC.sha_src(G5.__file__),
                    "zg3res_sha256": LC.sha_src(G3.__file__), "zg4prop_sha256": LC.sha_src(G4.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "a13x": {
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
        raise SystemExit(f"za13x: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za13x: A13x's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za13x: zspxb's fit of {split} is {FIT[split]}, not {name}")
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
    if rec.get("arm") != ARM or "a13x" not in rec:
        raise SystemExit(f"za13x: {name} is not an A13x fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    G3.res_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspxb decided against zspx's card fits ──────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za13x: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspx's card fits scr-g5-zspx and scr-g5-zspx-hp)"),
       ("(rel's screen fits)", "(zspx's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspx R@5 | zspxb R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A13x ({DECLARED}): zspxb ({WHAT}) against zspx.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A13x", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za13x_sha256": LC.sha_src(__file__)})
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


def verdict(recall_file=None, out=None):
    """A13x's screen verdict from zspxb's re-call (section 4); webqsp and metaqa beside; the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A13x" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A13x's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    out_rec = {"stage": "A13x", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za13x_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    md = [f"# A13x's screen verdict: zspxb **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspxb** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items()) or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspx per read:", "", "| read | zspxb |", "| --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} |" for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za13x verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall_rows(calls=None):
    return [{"split": sp, "dataset": ds, "call": (calls or {}).get((sp, ds), "WITHIN"), "delta": 0.0}
            for sp in SPLITS for ds in LG.EVAL_ORDER]


def fake_recall(path, calls, stage="A13x", arm=ARM, **over):
    rec = {"stage": stage, "declared_in": DECLARED if stage == "A13x" else "x", "arm": arm,
           "rows": fake_recall_rows(calls)}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm: zspx plus A13's BrMix (zbr's head) on G3's carve with dense_cos beside the links; zspx's, zspkb's,
    #    zspk's and zbr's own arms unchanged; one zprop, zlink and zp1c3 under both parents; the head's shape and draw
    #    are zbr's, its generator apart from zspx's four heads'
    assert S.ARMS[ARM] == (ZSPXB, CosResKTLinkCarveBase) and CosResKTLinkCarveBase.__mro__[1] is G3.ResKTLinkCarveBase
    assert ZSPXB.__mro__[1:8] == (G5.ZSPX, G3.ZSPR, Z3.ZSPK, ZP.ZProp, A13.BrMix, Z3.KTMix, ZM.ZRM)
    assert S.ARMS[BASE_ARM] == (G5.ZSPX, G3.ResKTLinkCarveBase) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert S.ARMS["zspkb"] == (A13.ZSPKB, A13.CosKTLinkCarveBase) and S.ARMS["zbr"] == (ZB.ZBridge, ZB.BridgeCarveZRC)
    assert A13.Z3 is Z3 and A13.ZP is ZP and ZB.ZL is ZL and A13.S is S and ZB.S is S
    assert issubclass(CosResKTLinkCarveBase, Z3.KTLinkCarveBase)
    assert issubclass(CosResKTLinkCarveBase, RM.ChainCarveBase)
    assert (ZB.N_IN, ZB.HID, ZB.SEED_OFF) == (13, 32, 2901) and SETTINGS["parameters_added"] == 481
    assert len({ZB.SEED_OFF, ZP.SEED_OFF, Z3.KT_SEED_OFF, G3.RS_SEED_OFF, G4.P2_SEED_OFF, ZL.SEED_OFF}) == 6
    assert BASE_SCREENS == {sp: f"outputs/a9/base-zspx-{sp}.json" for sp in SPLITS}
    # P1c2's roots, patched on zlink's and rmatch's carves, reach A13x's carve; the KT and residual roots are zspx's
    with Z2.on_p1c2():
        assert CosResKTLinkCarveBase.LK_ROOT == ZL.LinkCarveBase.LK_ROOT == Z2.LINKS
        assert CosResKTLinkCarveBase.CH_ROOT == RM.ChainCarveBase.CH_ROOT == Z2.CH_RM
    assert CosResKTLinkCarveBase.KT_ROOT == G3.ResKTLinkCarveBase.KT_ROOT == Z3.KT_DIR
    assert CosResKTLinkCarveBase.RES_ROOT == G3.ResKTLinkCarveBase.RES_ROOT == G3.RES_DIR
    tmp = Path(tempfile.mkdtemp(prefix="za13x_"))
    try:
        rng = np.random.default_rng(1313)
        qq = RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, _, _ = G3.toy_embed(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look", lk_root=tmp / "lk",
                  freeze="toy")
        G3.build("metaqa", "toy", 0, nodes=G3.ToyNodes(T), **kw)
        G3.build("metaqa", "toy", 1, nodes=G3.ToyNodes(T), **kw)

        class ToyIds(ZM.ToyZ):
            """zrm's toy carve with question ids and its fourth column named dense_cos (zbr's and A13's toy)."""

            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]
                self.span = {"dense_cos": (3, 4)}

        cls_ch = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        cls_res = G3.res_carve(Z3.kt_carve(cls_lk, tmp / "kt"), tmp / "res")
        cls = A13.cos_carve(cls_res)
        c, cr = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), cls_res("metaqa", "toy", "2wiki", "cpu",
                                                                              tmp / "cache")
        c_ch = cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        # 2. the carve: zspx's batch unchanged (rows, links, KT columns, residuals, base_z, golds), plus each row's
        #    cached dense_cos beside its links
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:6], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        f0, nq0, bz0, g0 = cr.batch(qs, blocks)
        assert torch.equal(nq, nq0) and torch.equal(bz, bz0) and torch.equal(gold, g0)
        assert sorted(feats) == sorted(f0) and sorted(feats[LK_KEY]) == ["cos", "f", "u", "v"]
        assert all(torch.equal(feats[LK_KEY][k], f0[LK_KEY][k]) for k in ("u", "v", "f"))

        def same(a_, b_):
            if isinstance(a_, torch.Tensor):
                return isinstance(b_, torch.Tensor) and a_.dtype == b_.dtype and torch.equal(a_, b_)
            if isinstance(a_, dict):
                return isinstance(b_, dict) and sorted(a_) == sorted(b_) and all(same(a_[k], b_[k]) for k in a_)
            if isinstance(a_, (tuple, list)):
                return type(a_) is type(b_) and len(a_) == len(b_) and all(same(p, q) for p, q in zip(a_, b_))
            if isinstance(a_, np.ndarray):
                return isinstance(b_, np.ndarray) and a_.dtype == b_.dtype and np.array_equal(a_, b_)
            return a_ == b_

        assert all(same(feats[k], f0[k]) for k in f0 if k != LK_KEY)
        lk, rs = feats[LK_KEY], feats[G3.RES_KEY]
        assert lk["u"].numel() > 0 and Z3.KT_KEY in feats and rs["s"].numel() == lk["u"].numel()
        assert rs["c"].numel() == nq.numel()
        Xd = lambda q: c.X[c.off_np[q]:c.off_np[q] + c.n_np[q], 3].to(torch.float32)  # noqa: E731
        assert torch.equal(lk[COS], torch.nan_to_num(torch.cat([Xd(q) for q in qs]), nan=0.0))

        def links_of(q):
            t = qq[q]
            e = {(int(x), int(y), int(f)) for x, y, f in zip(t["e_u"], t["e_v"], t["e_fam"]) if x != y}
            return sorted(e | {(y, x, f) for x, y, f in e})

        # 3. the inputs are zbr's by hand (python sorts and loops over each question's edge list); none reads a score
        x = ZB.bridge_inputs(bz, lk, nq, B)
        xh = ZB.hand_inputs(bz.numpy(), lk[COS].numpy(), c.n_np, qs, links_of)
        assert x.shape == (nq.numel(), ZB.N_IN) and np.allclose(x.numpy(), xh, atol=1e-5)
        assert float(x[:, 0:10].sum()) > 0 and float(x[:, 10].sum()) > 0 and float(x[:, 12].sum()) > 0
        # 4. from one seed zspxb is zspx's state plus zbr's head drawn as zbr and A13's zspkb draw it, the last layer
        #    zero; the global generator is where zspx leaves it
        keep = torch.ones(B, len(blocks))
        nets, after = {}, {}
        for k, cl in ((BASE_ARM, G5.ZSPX), (ARM, ZSPXB), ("zspkb", A13.ZSPKB), ("zbr", ZB.ZBridge)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after[BASE_ARM])
        sx, sb = nets[BASE_ARM].state_dict(), {k: v.clone() for k, v in nets[ARM].state_dict().items()}
        assert sorted(set(sb) - set(sx)) == sorted(BR) and all(torch.equal(sx[k], sb[k]) for k in sx)
        assert all(torch.equal(sb[k], nets[o].state_dict()[k]) for k in BR for o in ("zspkb", "zbr"))
        assert float(sb["br_w2"].abs().sum()) == 0.0 and float(sb["br_w1"].abs().sum()) > 0.0
        g = torch.Generator().manual_seed(13)
        heads = ("sp_w2", "rs_w2", "p2_w2", "kt_w2")
        with torch.no_grad():
            # 5. at the start zspxb is zspx bit for bit, in eval and in training with the same dropout draws, with
            #    zspx's three step heads and its KT head moved (so every step acts)
            for nm in heads:
                w = torch.randn(getattr(nets[BASE_ARM], nm).shape, generator=g)
                for m in (nets[BASE_ARM], nets[ARM]):
                    getattr(m, nm).copy_(w)
            for mode in (False, True):
                out = {}
                for k in (BASE_ARM, ARM):
                    nets[k].train(mode)
                    torch.manual_seed(5)
                    out[k] = nets[k](feats, keep, nq, B, bz)
                assert torch.equal(out[ARM], out[BASE_ARM]), mode
            # 6. the placement: with the head moved, zspxb is zspx's chain over (zspx's pre-step score + zbr's head):
            #    its first step (gradient kept), G3's residual message and G4's second step all read the shifted score
            nb, nx = nets[ARM].eval(), nets[BASE_ARM].eval()
            for k in BR:
                getattr(nb, k).copy_(torch.randn(getattr(nb, k).shape, generator=g))
            s0 = Z3.KTMix.forward(nb, feats, keep, nq, B, bz)

            def chain(first):
                s0b = Z3.KTMix.forward(nb, feats, keep, nq, B, bz) + nb.bridge_head(x)
                s1 = s0b + nb.prop_head(first(s0b, lk, nq, B))
                s2 = s1 + nb.rs_head(G3.res_inputs(s1, lk, rs, nq, B))
                return s2 + nb.p2_head(ZP.prop_inputs(s2, lk, nq, B))

            y, yx = nb(feats, keep, nq, B, bz), nx(feats, keep, nq, B, bz)
            assert torch.equal(y, chain(G4.prop_inputs_grad)) and not torch.equal(y, yx)
            assert torch.equal(Z3.KTMix.forward(nx, feats, keep, nq, B, bz), s0)
            # the steps carry the shift: zspxb minus zspx is more than the head's own term
            assert float((y - yx - nb.bridge_head(x)).abs().max()) > 1e-4
            hd = Fn.gelu(x @ nb.br_w1.T + nb.br_b1) @ nb.br_w2.T + nb.br_b2
            assert torch.allclose(nb.bridge_head(x), hd.squeeze(-1), atol=1e-6)
            # 7. without links it is zspx's scorer (no step, no head); links without dense_cos are refused
            no_lk = {k: v for k, v in feats.items() if k not in (LK_KEY, G3.RES_KEY)}
            assert torch.equal(nb(no_lk, keep, nq, B, bz), s0) and torch.equal(nx(no_lk, keep, nq, B, bz), s0)
            bare = dict(feats, **{LK_KEY: {k: v for k, v in lk.items() if k != COS}})
            try:
                nb(bare, keep, nq, B, bz)
                raise AssertionError("links without dense_cos were not refused")
            except RuntimeError:
                pass
        # 8. the gradient: the head learns through zspx's first step (it differs from a first step that reads the
        #    score detached, zspkb's case), and the model's gradient is the explicit chain's
        gw = torch.randn(nq.numel(), generator=torch.Generator().manual_seed(11))

        def br_grads(fn):
            nb.zero_grad(set_to_none=True)
            (fn() * gw).sum().backward()
            return {k: getattr(nb, k).grad.detach().clone() for k in BR}

        g_model = br_grads(lambda: nb(feats, keep, nq, B, bz))
        g_chain = br_grads(lambda: chain(G4.prop_inputs_grad))
        g_det = br_grads(lambda: chain(ZP.prop_inputs))
        assert all(torch.equal(g_model[k], g_chain[k]) for k in BR)
        assert any(float((g_model[k] - g_det[k]).abs().max()) > 1e-6 for k in BR)
        nb.zero_grad(set_to_none=True)
        # 9. toy fits through lean_gpu's loop: a repeat is bit for bit, zbr's head trains; through the first step the
        #    shared weights match zspx's own fit (the head adds 0 there) and every state after differs; without links
        #    the fit is zspx's bit for bit on every shared key, the head untouched
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            assert LG.LeanMLP8D is ZSPXB and LG.CacheCarve is CosResKTLinkCarveBase
            fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxb")
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxb repeat")
            nolk = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxb, no links")
        with S.patched(BASE_ARM):
            base = LG.fit_variant([cr], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx")
            base_nolk = LG.fit_variant([c_ch], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx, no links")
        pairs = lambda a_, b_: zip(a_["states"] + [a_["swa"]], b_["states"] + [b_["swa"]])  # noqa: E731
        assert all(not LG.same_state(a_, b_) for a_, b_ in pairs(fit, rep))
        assert float(fit["swa"]["br_w2"].abs().sum()) > 0 and not torch.equal(fit["swa"]["br_w1"], sb["br_w1"])
        assert all(r["steps"] == 1 for r in fit["curve"])
        diff = [LG.same_state({k: a_[k] for k in b_}, b_) for a_, b_ in pairs(fit, base)]
        assert not diff[0] and all(diff[1:]), diff
        for a_, b_ in pairs(nolk, base_nolk):
            assert not LG.same_state({k: a_[k] for k in b_}, b_)
            assert torch.equal(a_["br_w1"], sb["br_w1"]) and float(a_["br_w2"].abs().max()) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 10. the refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    for other in (BASE_ARM, "zspkb", "zbr"):
        SR.must_stop(train, ["train", "--arm", other, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                     "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": BASE_ARM, "g5": {}}), ("w", {"arm": ARM}), ("b", {"arm": "zspkb", "a13": {}}),
                        ("a", {"arm": ARM, "a13": {}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 11. relz's pair and re-call against zspx's card fits, stamped A13x's; relz restored after
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
        assert pj["stage"] == "A13x" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A13x ({DECLARED})") and f"| zspx R@5 | zspxb R@5 ({WHAT}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 12. the verdict's outcomes and next forms
        rp = T_ / "rc.json"
        gm, lw = ("L-musique", "musique"), ("L-hotpotqa", "webqsp")
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and v["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        v = verdict(fake_recall(rp, {gm: "GAIN", lw: "LOSS"}), T_ / "v")
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["GAIN_AND_LOSS"]
        v = verdict(fake_recall(rp, {}), T_ / "v")
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(fake_recall(rp, {lw: "LOSS"}), T_ / "v")
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(fake_recall(rp, {gm: "GAIN"}, stage="A13", arm="zspkb"), T_ / "v")
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"]
        v = verdict(fake_recall(rp, {gm: "GAIN"}, rows=fake_recall_rows()[:11]), T_ / "v")
        assert v["verdict"] == "INCOMPLETE"
        assert verdict(T_ / "none.json", T_ / "v")["verdict"] == "INCOMPLETE"
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "Next form" in (T_ / "v.md").read_text(encoding="utf-8")
    log(f"za13x selftest: zspxb is zspx plus zbr's head (zbr's shape and draw, A13's BrMix) on G3's carve with "
        f"dense_cos beside the links, P1c2's roots reaching it; the carve's batch is zspx's plus the cached column; "
        f"the inputs are zbr's by hand; zspxb is zspx bit for bit at the start (eval and training, zspx's heads "
        f"moved); once the head moves, zspx's three steps read the bridge-shifted score and the head's gradient runs "
        f"through the first step; it is zspx's scorer without links and refuses links without dense_cos; toy fits: "
        f"repeats identical, the head trains, zspx's fit on the shared keys through the first step and apart after, "
        f"zspx's bit for bit without links; refusals; relz's pair and re-call stamped A13x's; the verdict's outcomes "
        f"and next forms ({time.time() - t0:.1f}s): ok")
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
            log(f"za13x {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za13x: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
