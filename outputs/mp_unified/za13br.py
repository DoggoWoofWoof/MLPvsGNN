"""A13 (docs/A13_BRIDGE_COLUMNS_GNN.md): zbr's bridge columns on zspk, the audit's thirteenth reopening (its
conditional item: R1, which carried the bridge signal to both models as a residual, was NOT_ADOPTED). G1a's
training-free bridge rule (keep RRF's top 3, fill slots 4-5 with the structural neighbours of RRF's top 2 that have the
highest cos(q)) was ABOVE RRF on four of six datasets. Its quantities reached only the MLP as inputs (round 29, zbr on
zrc: NO_GAIN); they never reached the GNN. Here:

    zspkb   zspk (zp1c3.ZSPK on P1c3's KT link carve) whose score, before zsp's propagation step reads it, adds zbr's
            head over zbr's 13 per-row inputs (zbr.bridge_inputs, unchanged: each row's links to RRF's top 1, 2 and 5
            per family, two-step paths from the top 5, the bridge flag, the bridge rank by dense_cos among flagged rows,
            the source rank). The head is zbr's: Linear(13, 32), GELU, Linear(32, 1), the first layer from its own
            generator (seed + 2901, zbr's draw), the last at zero. So at the start zspkb is zspk bit for bit, and the
            step propagates the bridge-shifted score once the head moves (R1's placement).

The inputs are fixed before any row is scored (RRF's order, zlink's pool edges, the cached dense_cos column); the head
reads no score. The carve is zspk's with each row's dense_cos beside its links (as zbr's carve adds it), every block,
row, edge and gold unchanged; no build. GNN only: zbr's MLP form closed on zrc (NO_GAIN), and the MLP's lean model
(zrckp) gets these columns next only after a gain here (section 5).

    python outputs/mp_unified/za13br.py train --split L-musique --name scr-a13-zspkb --arm zspkb --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za13br.py read --name scr-a13-zspkb --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za13br.py compare --new outputs/screen/fits/scr-a13-zspkb \\
        --base outputs/screen/fits/scr-p1c3-zspk,outputs/step1/fits/L-musique --out outputs/screen/scr-a13-zspkb
    python outputs/mp_unified/za13br.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za13br.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za13br.py verdict
    python outputs/mp_unified/za13br.py --selftest

Each read is decided against zspk's card fit of its split (scr-p1c3-zspk, scr-p1c3-zspk-hp); step 1's fit beside.
The re-call's base screens are L1's (outputs/l1/base-zspk-<split>.json: zspk's card fits against step 1's).

Speed: per question, two sorts of its pool (RRF's order, the flagged rows' dense_cos) and three passes over its pool's
edges, all from fixed inputs; zsp's step already makes three such passes. Any latency figure for zspkb is cold, timed
in its own declared stage.
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

import zp1c3 as Z3  # noqa: E402  (zspk and its KT carve; through it P1c2's roots, gate and card, zprop, zlink, rmatch)
import zbr as ZB  # noqa: E402  (round 29: bridge_inputs, its head's shape and draw, its by-hand inputs)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZL, ZM, RM = Z3.ZP, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A13_BRIDGE_COLUMNS_GNN.md"
OUT = ROOT / "outputs" / "a13"
SCREEN = ROOT / "outputs" / "screen"
LK_KEY, COS = ZL.LK_KEY, ZB.COS
BR = ZB.BR
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zspkb", "zspk"
FIT = {"L-musique": "scr-a13-zspkb", "L-hotpotqa": "scr-a13-zspkb-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-p1c3-zspk"), "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}
BASE_SCREENS = {sp: f"outputs/l1/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}
AGAINST = "zspk's card fit of each split under lean_gpu's loss (scr-p1c3-zspk, scr-p1c3-zspk-hp)"
WHAT = "zspk with zbr's bridge head added to its score before its propagation step"
SETTINGS = {"base": "zspk (outputs/mp_unified/zp1c3.py), its model, carve, card, seed and batches unchanged",
            "inputs": f"zbr.bridge_inputs, unchanged: {ZB.N_IN} per row from RRF's order (the batch's base_z), zlink's "
                      "pool edges and the row's cached dense_cos",
            "head": f"zbr's: Linear({ZB.N_IN}, {ZB.HID}), GELU, Linear({ZB.HID}, 1); first layer uniform in "
                    f"+-1/sqrt({ZB.N_IN}) from its own generator (seed + {ZB.SEED_OFF}), last layer zero",
            "where": "added to zspk's score (zrm's plus the KT head) before zsp's propagation step reads it (R1's "
                     "placement); MRO ZProp, BrMix, KTMix, ZRM",
            "carve": "P1c3's KT link carve with each row's dense_cos beside its links (zbr's carve's column); no build",
            "parameters_added": ZB.HID * ZB.N_IN + ZB.HID + ZB.HID + 1}
PAIR = SCREEN / "scr-a13-zspkb-pair"
RECALL = SCREEN / "scr-a13-zspkb-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspkb's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A13's, or not twelve reads "
        "(docs/A13_BRIDGE_COLUMNS_GNN.md, section 4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zspkb fitted again on P1g-b's pools, then its full run beside zspk's; and A13m, zbr's head "
                              "on zrckp (the MLP's lean model; zbr's inputs are fixed before any row is scored), since "
                              "zbr's MLP form closed on zrc before the MLP's model changed; each its own declared file",
        "GAIN_AND_LOSS": "A13b, the back-off form: the bridge head bounded (0.2 tanh(h / 0.2), the MP-unified track's "
                         "bound), so the step cannot lean on it where the bridge rule fails (squad, comparison "
                         "questions); one declared file",
        "NO_GAIN": "A13x, zbr's bridge head on zspx (the adopted GNN, three steps, where depth paid), before its first "
                   "step; one declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the carve: zspk's, with each row's dense_cos beside its links ────────────


def cos_carve(base):
    """base (a link carve) with each row's dense_cos (the cached column, NaN and infinities to 0, zbr's carve's code)
    under LK_KEY beside u, v and f; every block, row, edge, KT column and gold is base's. A subclass, so P1c2's roots
    (patched on zlink's and rmatch's carves) reach it."""

    class CosCarve(base):
        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            qs_ = np.asarray(qs, np.int64)
            cnt = self.n_np[qs_]
            seg = np.cumsum(cnt) - cnt
            idx = np.repeat(self.off_np[qs_] - seg, cnt) + np.arange(int(cnt.sum()))
            col = self.span["dense_cos"][0]
            cos = self.X[torch.from_numpy(idx).to(self.device), col].to(torch.float32)
            feats[LK_KEY][COS] = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)
            return feats, nq, base_z, gold

    CosCarve.__name__ = CosCarve.__qualname__ = f"Cos{base.__name__}"
    return CosCarve


CosKTLinkCarveBase = cos_carve(Z3.KTLinkCarveBase)


# ── the model ────────────────────────────────────────────────────────────────


class BrMix(Z3.KTMix):
    """zrm's score plus the KT head plus zbr's bridge head (zero at the start)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        g = torch.Generator().manual_seed(int(seed) + ZB.SEED_OFF)
        self.br_w1 = nn.Parameter((torch.rand(ZB.HID, ZB.N_IN, generator=g) * 2 - 1) / ZB.N_IN ** 0.5)
        self.br_b1 = nn.Parameter(torch.zeros(ZB.HID))
        self.br_w2 = nn.Parameter(torch.zeros(1, ZB.HID))
        self.br_b2 = nn.Parameter(torch.zeros(1))

    def bridge_head(self, x):
        return Fn.linear(Fn.gelu(Fn.linear(x, self.br_w1, self.br_b1)), self.br_w2, self.br_b2).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if LK_KEY not in feats:
            return s
        if COS not in feats[LK_KEY]:
            raise RuntimeError("za13br: the batch's links carry no dense_cos (not A13's carve)")
        return s + self.bridge_head(ZB.bridge_inputs(base_z, feats[LK_KEY], nq, B))


class ZSPKB(ZP.ZProp, BrMix):
    """zspkb: zspk plus zbr's bridge head, added before zsp's propagation reads the score (MRO: ZProp, BrMix, KTMix,
    ZRM)."""


ARMS = {ARM: (ZSPKB, CosKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za13br_sha256": LC.sha_src(__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "zbr_sha256": LC.sha_src(ZB.__file__), "a13": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE_ARM, "device": Z2.DEVICE,
                        "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM]:
        raise SystemExit(f"za13br: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za13br: A13's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za13br: zspkb's fit of {split} is {FIT[split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") != ARM or "a13" not in rec:
        raise SystemExit(f"za13br: {name} is not an A13 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspkb decided against zspk's card fits ──────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za13br: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp)"),
       ("(rel's screen fits)", "(zspk's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspk R@5 | zspkb R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A13 ({DECLARED}): zspkb ({WHAT}) against zspk.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A13", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za13br_sha256": LC.sha_src(__file__)})
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
    """A13's screen verdict from zspkb's re-call (section 4); webqsp and metaqa beside; the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A13" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A13's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    out_rec = {"stage": "A13", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za13br_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    md = [f"# A13's screen verdict: zspkb **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspkb** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside: "
          + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items()) or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspk per read:", "", "| read | zspkb |", "| --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} |" for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za13br verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="A13", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "call": calls.get((sp, ds), "WITHIN"), "delta": 0.0}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED, "arm": arm, "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm: zspk plus zbr's head on P1c3's KT link carve with dense_cos beside the links; zspk's, zsp's and zbr's
    #    own arms unchanged; the head's shape and draw are zbr's
    assert S.ARMS[ARM] == (ZSPKB, CosKTLinkCarveBase) and CosKTLinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert [k.__name__ for k in ZSPKB.__mro__[:5]] == ["ZSPKB", "ZProp", "BrMix", "KTMix", "ZRM"]
    assert S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase) and S.ARMS["zsp"] == (ZP.ZProp, ZL.LinkCarveBase)
    assert S.ARMS["zbr"] == (ZB.ZBridge, ZB.BridgeCarveZRC) and ZB.ZL is ZL and ZB.S is S
    assert issubclass(CosKTLinkCarveBase, ZL.LinkCarveBase) and issubclass(CosKTLinkCarveBase, RM.ChainCarveBase)
    assert (ZB.N_IN, ZB.HID, ZB.SEED_OFF) == (13, 32, 2901) and SETTINGS["parameters_added"] == 481
    assert ZB.SEED_OFF not in (ZP.SEED_OFF, Z3.KT_SEED_OFF, ZL.SEED_OFF)
    # P1c2's roots, patched on zlink's and rmatch's carves, reach A13's carve
    with Z2.on_p1c2():
        assert CosKTLinkCarveBase.LK_ROOT == ZL.LinkCarveBase.LK_ROOT == Z2.LINKS
        assert CosKTLinkCarveBase.CH_ROOT == RM.ChainCarveBase.CH_ROOT == Z2.CH_RM
    tmp = Path(tempfile.mkdtemp(prefix="za13br_"))
    try:
        rng = np.random.default_rng(1313)
        qq = RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            """zrm's toy carve with question ids and its fourth column named dense_cos (zbr's toy)."""

            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]
                self.span = {"dense_cos": (3, 4)}

        cls_lk = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        cls_kt = Z3.kt_carve(cls_lk, tmp / "kt")
        cls = cos_carve(cls_kt)
        c, ck = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), cls_kt("metaqa", "toy", "2wiki", "cpu",
                                                                             tmp / "cache")
        # 2. the carve: zspk's batch unchanged, plus each row's cached dense_cos beside its links
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:6], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        f0, nq0, bz0, g0 = ck.batch(qs, blocks)
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
        assert feats[LK_KEY]["u"].numel() > 0 and Z3.KT_KEY in feats
        Xd = lambda q: c.X[c.off_np[q]:c.off_np[q] + c.n_np[q], 3].to(torch.float32)  # noqa: E731
        assert torch.equal(feats[LK_KEY][COS], torch.nan_to_num(torch.cat([Xd(q) for q in qs]), nan=0.0))

        def links_of(q):
            t = qq[q]
            e = {(int(x), int(y), int(f)) for x, y, f in zip(t["e_u"], t["e_v"], t["e_fam"]) if x != y}
            return sorted(e | {(y, x, f) for x, y, f in e})

        # 3. the inputs are zbr's by hand (python sorts and loops over each question's edge list); none reads a score
        x = ZB.bridge_inputs(bz, feats[LK_KEY], nq, B)
        xh = ZB.hand_inputs(bz.numpy(), feats[LK_KEY][COS].numpy(), c.n_np, qs, links_of)
        assert x.shape == (nq.numel(), ZB.N_IN) and np.allclose(x.numpy(), xh, atol=1e-5)
        assert float(x[:, 0:10].sum()) > 0 and float(x[:, 10].sum()) > 0 and float(x[:, 12].sum()) > 0
        # 4. from one seed zspkb is zspk's state plus zbr's head drawn as zbr draws it, the last layer zero; the global
        #    generator is where zspk leaves it
        keep = torch.ones(B, len(blocks))
        nets, after = {}, {}
        for k, cl in ((BASE_ARM, Z3.ZSPK), (ARM, ZSPKB), ("zbr", ZB.ZBridge)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after[BASE_ARM])
        sz, sk, sb = nets[BASE_ARM].state_dict(), nets[ARM].state_dict(), nets["zbr"].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(BR) and all(torch.equal(sz[k], sk[k]) for k in sz)
        assert all(torch.equal(sk[k], sb[k]) for k in BR) and float(sk["br_w2"].abs().sum()) == 0.0
        g = torch.Generator().manual_seed(13)
        with torch.no_grad():
            # 5. at the start zspkb is zspk bit for bit, in eval and in training with the same dropout draws, also once
            #    zspk's step and KT heads have moved
            w = torch.randn(1, ZP.HID, generator=g)
            wk = torch.randn(1, Z3.KT_HID, generator=g)
            for m in (nets[BASE_ARM], nets[ARM]):
                m.sp_w2.copy_(w)
                m.kt_w2.copy_(wk)
            for mode in (False, True):
                out = {}
                for k in (BASE_ARM, ARM):
                    nets[k].train(mode)
                    torch.manual_seed(5)
                    out[k] = nets[k](feats, keep, nq, B, bz)
                assert torch.equal(out[ARM], out[BASE_ARM]), mode
            # 6. with the head moved, zspkb is zspk's propagation over (zspk's pre-step score + zbr's head): the step
            #    reads the bridge-shifted score
            nk = nets[ARM].eval()
            for k in BR:
                getattr(nk, k).copy_(torch.randn(getattr(nk, k).shape, generator=g))
            s_kt = Z3.KTMix.forward(nk, feats, keep, nq, B, bz)
            sb_ = s_kt + nk.bridge_head(x)
            want = sb_ + nk.prop_head(ZP.prop_inputs(sb_, feats[LK_KEY], nq, B))
            y = nk(feats, keep, nq, B, bz)
            assert torch.equal(y, want) and not torch.equal(y, nets[BASE_ARM].eval()(feats, keep, nq, B, bz))
            hd = Fn.gelu(x @ nk.br_w1.T + nk.br_b1) @ nk.br_w2.T + nk.br_b2
            assert torch.allclose(nk.bridge_head(x), hd.squeeze(-1), atol=1e-6)
            # 7. without links it is zspk's (no step, no head); links without dense_cos are refused
            no_lk = {k: v for k, v in feats.items() if k != LK_KEY}
            assert torch.equal(nk(no_lk, keep, nq, B, bz), s_kt)
            bare = dict(feats, **{LK_KEY: {k: v for k, v in feats[LK_KEY].items() if k != COS}})
            try:
                nk(bare, keep, nq, B, bz)
                raise AssertionError("links without dense_cos were not refused")
            except RuntimeError:
                pass
        # 8. toy fits through lean_gpu's loop: a repeat is bit for bit, zbr's head trains; zspk's own fit differs
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            assert LG.LeanMLP8D is ZSPKB and LG.CacheCarve is CosKTLinkCarveBase
            fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspkb")
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspkb repeat")
        with S.patched(BASE_ARM):
            base = LG.fit_variant([ck], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspk")
        pairs = lambda a_, b_: zip(a_["states"] + [a_["swa"]], b_["states"] + [b_["swa"]])  # noqa: E731
        assert all(not LG.same_state(a_, b_) for a_, b_ in pairs(fit, rep))
        assert float(fit["swa"]["br_w2"].abs().sum()) > 0 and not torch.equal(fit["swa"]["br_w1"], sk["br_w1"])
        # one step per toy epoch: through the first step the head adds 0, so the shared keys take zspk's gradient and
        #    match zspk's fit; every state after it differs
        assert all(r["steps"] == 1 for r in fit["curve"])
        diff = [LG.same_state({k: a_[k] for k in b_}, b_) for a_, b_ in pairs(fit, base)]
        assert not diff[0] and all(diff[1:]), diff
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 9. the refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, ["train", "--arm", "zbr", "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": BASE_ARM, "p1c3": {}}), ("w", {"arm": ARM}), ("b", {"arm": "zbr", "zbr": {}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 10. relz's pair and re-call against zspk's card fits, stamped A13's; relz restored after
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
        assert pj["stage"] == "A13" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A13 ({DECLARED})") and f"| zspk R@5 | zspkb R@5 ({WHAT}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 11. the verdict's outcomes and next forms
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
        v = verdict(fake_recall(rp, {gm: "GAIN"}, stage="A12", arm="zspkgs"), T_ / "v")
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"]
        assert verdict(T_ / "none.json", T_ / "v")["verdict"] == "INCOMPLETE"
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "Next form" in (T_ / "v.md").read_text(encoding="utf-8")
    log(f"za13br selftest: zspkb is zspk plus zbr's head (zbr's shape and draw) on P1c3's KT link carve with dense_cos "
        f"beside the links, P1c2's roots reaching it; the carve's batch is zspk's plus the cached column; the inputs "
        f"are zbr's by hand; zspkb is zspk bit for bit at the start (eval and training, zspk's heads moved), its step "
        f"reads the bridge-shifted score once the head moves, it is zspk's without links and refuses links without "
        f"dense_cos; toy fits: repeats identical, the head trains, zspk's fit on the shared keys through the first step "
        f"and apart after; refusals; relz's pair and re-call stamped A13's; "
        f"the verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
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
            log(f"za13br {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za13br: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
