"""A4h (docs/A4H_BLOCK_DROPOUT_HALF.md): A4's block dropout at half the rate, the back-off form A4's verdict named
for zspkd (a GAIN, webqsp zero-shot +0.0134, and a LOSS, musique zero-shot -0.1394). A4 (za4bdrop.py) kept each
(question, block) with probability 0.8 at the top of each of today's models; A4h keeps it with probability 0.9 and
changes nothing else:

    in a training forward, each (question, block) of the model's blocks is kept with probability 0.9, from the model's
    own generator (seed + 101, bdrop20's and A4's), its keep flag times the draw; a read keeps every block

zret's blocks enter as [raw m, z m, m] (a dropped block's columns and flag are 0, lean_mlp's missing-block input), and
rmatch's chain term is gated by SEMB's flag, so it drops with SEMB. The KT head and zsp's propagation read no flag.
No new parameter, no score read; the global stream, dropout's masks included, is the base's.

    zrckph  zrckp's model (outputs/mp_unified/zp1c3p.py) trained with A4h, on zrckp's carve
    zspkh   zspk's model (outputs/mp_unified/zp1c3.py) trained with A4h; zsp's propagation reads the score it changes

    python outputs/mp_unified/za4h.py train --split L-musique --name scr-a4h-zrckph --arm zrckph --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za4h.py read --name scr-a4h-zrckph --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za4h.py compare --new outputs/screen/fits/scr-a4h-zrckph \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a4h-zrckph
    python outputs/mp_unified/za4h.py pair --arm zrckph --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za4h.py recall --arm zrckph --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za4h.py verdict
    python outputs/mp_unified/za4h.py --selftest
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

import zp1c3 as Z3  # noqa: E402  (zspk and its KT carves; through it P1c2's roots, gate and card, zprop, zlink, rmatch)
import zp1c3p as Z3P  # noqa: E402  (zrckp)
import lean_screen3 as S3  # noqa: E402  (zret)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
LM = LG.LM
log = S.log

DECLARED = "docs/A4H_BLOCK_DROPOUT_HALF.md"
OUT = ROOT / "outputs" / "a4h"
SCREEN = ROOT / "outputs" / "screen"
P_DROP = 0.1                                # half of A4's (bdrop20's 0.2, S.BDrop.P)
SEED_OFF = 101                              # bdrop20's mask seed: seed + 101 (the selftest checks lean_screen's)
SETTINGS = {"p_drop": P_DROP, "unit": "(question, block) over the model's blocks", "seed": f"model seed + {SEED_OFF}",
            "where": "the top of the model's forward: every layer reading the keep flags sees the drop",
            "keep": "the incoming keep flag times the draw", "reads": "every block kept", "parameters": 0,
            "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckph": "zrckp", "zspkh": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a4h-{a}", "L-hotpotqa": f"scr-a4h-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckph": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkh": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckph": "zrckp trained with A4h's block dropout (0.1 per question and block)",
        "zspkh": "zspk trained with A4h's block dropout (0.1 per question and block), before its propagation"}
PAIR = {a: SCREEN / f"scr-a4h-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a4h-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("metaqa", "musique", "webqsp")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A4h's re-call of the arm, or "
        "not twelve reads (docs/A4H_BLOCK_DROPOUT_HALF.md, section 4)")
LAST = {"forwards": 0, "cells": 0, "dropped": 0, "questions_all_dropped": 0}   # the last fit's counts


# ── the dropout ──────────────────────────────────────────────────────────────


class A4HTop(S3.ZRet):
    """At the top of the arm's MRO: in a training forward, the keep flags times a draw kept with probability 1 - P_DROP
    from the model's own generator; then the base model's forward, unchanged."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.a4h_gen = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        LAST.update(forwards=0, cells=0, dropped=0, questions_all_dropped=0)

    def a4h_draw(self, shape):
        return torch.rand(shape, generator=self.a4h_gen)

    def forward(self, feats, keep, nq, B, base_z):
        if self.training:
            u = self.a4h_draw(tuple(keep.shape)).to(keep.device)
            k = (u >= P_DROP).to(keep.dtype)
            LAST["forwards"] += 1
            LAST["cells"] += k.numel()
            LAST["dropped"] += int((k == 0).sum())
            LAST["questions_all_dropped"] += int((k.sum(1) == 0).sum())
            keep = keep * k
        return super().forward(feats, keep, nq, B, base_z)


class ZRCKPH(A4HTop, Z3P.ZRCKP):
    """zrckph: zrckp's model trained with A4h (MRO: A4HTop, ZRCKP, KTMixP, ZRM, ChainMatch, ZRet)."""


class ZSPKH(A4HTop, Z3.ZSPK):
    """zspkh: zspk's model trained with A4h; zsp's propagation reads the score it changes (MRO: A4HTop, ZSPK, ZProp, KTMix,
    ZRM, ChainMatch, ZRet)."""


ARMS = {"zrckph": (ZRCKPH, Z3.KTChainCarveZRC), "zspkh": (ZSPKH, Z3.KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def split_of_fit(name):
    for a in ARM_NAMES:
        for sp, n in FIT[a].items():
            if n == name:
                return a, sp
    return None, None


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za4h_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__), "lean_screen_sha256": LC.sha_src(S.__file__),
                    "a4h": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                           "device": Z2.DEVICE, "settings": SETTINGS,
                           "last_fit": {**LAST, "dropped_share": LAST["dropped"] / max(LAST["cells"], 1)},
                           "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                           "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                           "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                           "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)
        log(f"  a4h {arm} {split}: {LAST['dropped']} of {LAST['cells']} (question, block) cells dropped in the last fit "
            f"({LAST['dropped'] / max(LAST['cells'], 1):.4f}), {LAST['questions_all_dropped']} questions with every "
            f"block dropped")


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za4h: train takes --arm zrckph or zspkh, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za4h: A4h's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za4h: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a4h" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za4h: {name} is not an A4h fit (arm {rec.get('arm')})")
    if rec["a4h"].get("split") != split:
        raise SystemExit(f"za4h: {name} is {split}'s fit, its record says {rec['a4h'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za4h: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za4h: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp)"),
            ("(rel's screen fits)", f"({b}'s card fits)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A4h's block dropout) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A4h ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A4h", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za4h_sha256": LC.sha_src(__file__)})
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


NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools, then its full run, its own declared file",
        "GAIN_AND_LOSS": "A4hs, the back-off over blocks: block dropout at 0.1 on the eight fixed blocks only, SEMB and "
                         "with it rmatch's chain term always kept, on both models, its own declared file",
        "NO_GAIN": "the rate closes; the idea goes on in A4f (the KT columns and rmatch's chain term dropped as blocks "
                   "too), declared beside, on both models",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A4h's screen verdict, per arm, from its re-call (section 4); bdrop20's reads beside (musique zero-shot, its gain;
    webqsp zero-shot, its loss; metaqa), musique's in-domain read of both arms side by side, and the next form each
    outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A4h" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A4h's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "next_form": next_form(v, gains, losses),
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A4h", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za4h_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A4h's screen verdict: zrckph **{per['zrckph']['verdict']}**, zspkh **{per['zspkh']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {b['call']} ({num(b['delta'])})" for k, b in p["beside"].items()) or "none")
                  + f". Next: {p['next_form']}.")
    md += ["", f"Beside, deciding nothing: musique in-domain ({ck}): "
           + ", ".join(f"{a} {side[a]['call']} ({num(side[a]['delta'])})" for a in ARM_NAMES) + ".",
           "", "This verdict closes only this form (block dropout 0.1 per question and block over the models' blocks, "
               "at the top of the forward, in training); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za4h verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A4h", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    global P_DROP
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: A4HTop at the top of zrckp's and zspk's models; the base arms unchanged; bdrop20's rate and seed
    assert S.ARMS["zrckph"] == (ZRCKPH, Z3.KTChainCarveZRC) and S.ARMS["zspkh"] == (ZSPKH, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert "zrckpd" not in S.ARMS and "zspkd" not in S.ARMS          # A4's arms are not imported
    assert ZRCKPH.__mro__[1:7] == (A4HTop, Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, S3.ZRet)
    assert ZSPKH.__mro__[1:8] == (A4HTop, Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, S3.ZRet)
    assert P_DROP == 0.1 and S.BDrop.P == 0.2
    rng = np.random.default_rng(84)
    tmp = Path(tempfile.mkdtemp(prefix="za4h_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        c_m = cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_g = cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, len(blocks))
        keep_[3, 1] = 0.0
        assert S.BDrop(blocks, c_m.widths, 16, seed=7).mask_seed == 7 + SEED_OFF
        # 2. the mask: bdrop20's generator at half its rate; a draw of many cells keeps about 0.9; an absent block stays absent
        g = torch.Generator().manual_seed(0 + SEED_OFF)
        u = torch.rand((20000, 2), generator=g)
        assert abs(float((u >= P_DROP).float().mean()) - 0.9) < 0.01
        # 3. each arm against its base: in eval mode bit for bit and no draw; in training at P_DROP 0 bit for bit,
        #    scores and every gradient, and the global stream untouched; with a mask, the arm is the base model's forward
        #    on keep times the mask, bit for bit (rmatch's SEMB-gated chain term and zret's blocks both see it), and a
        #    question whose SEMB is dropped loses the chain term
        for arm, cls, base, cv in (("zrckph", ZRCKPH, Z3P.ZRCKP, c_m), ("zspkh", ZSPKH, Z3.ZSPK, c_g)):
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1_ = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            assert not LG.same_state(m0.state_dict(), m1_.state_dict())
            with torch.no_grad():
                for p_ in list(m0.out.parameters()) + list(m1_.out.parameters()):
                    p_.copy_(torch.linspace(-0.3, 0.3, p_.numel()).reshape(p_.shape))
                for m_ in (m0, m1_):
                    m_.cm_gate.copy_(torch.tensor([0.7, -0.4]))
                    if arm == "zspkh":
                        m_.sp_w2.copy_(torch.randn(m_.sp_w2.shape, generator=torch.Generator().manual_seed(8)))
            m0.eval()
            m1_.eval()
            g_state = m1_.a4h_gen.get_state()
            assert torch.equal(m0(f2, keep_, nq2, B, bz2), m1_(f2, keep_, nq2, B, bz2))
            assert torch.equal(m1_.a4h_gen.get_state(), g_state)
            m0.train()
            m1_.train()
            P_DROP = 0.0
            try:
                for rep in range(2):
                    torch.manual_seed(3 + rep)
                    y0 = m0(f2, keep_, nq2, B, bz2)
                    torch.manual_seed(3 + rep)
                    y1 = m1_(f2, keep_, nq2, B, bz2)
                    assert torch.equal(y0, y1), (arm, rep)
                    m0.zero_grad()
                    m1_.zero_grad()
                    (y0 * torch.linspace(-1, 1, y0.numel())).sum().backward()
                    (y1 * torch.linspace(-1, 1, y1.numel())).sum().backward()
                    for n_, p0 in m0.named_parameters():
                        g0, g1 = p0.grad, dict(m1_.named_parameters())[n_].grad
                        assert (g0 is None and g1 is None) or torch.equal(g0, g1), (arm, rep, n_)
            finally:
                P_DROP = SETTINGS["p_drop"]
            u_ = torch.full((B, len(blocks)), 0.9)
            u_[0, 1] = u_[5, 0] = u_[7, 0] = u_[7, 1] = 0.05                   # SEMB off on 0, rank off on 5, both on 7
            mask = (u_ >= P_DROP).float()
            saved_draw = m1_.a4h_draw
            m1_.a4h_draw = lambda shape: u_
            try:
                torch.manual_seed(5)
                y1 = m1_(f2, keep_, nq2, B, bz2)
            finally:
                m1_.a4h_draw = saved_draw
            torch.manual_seed(5)
            yb = m0(f2, keep_ * mask, nq2, B, bz2)
            assert torch.equal(y1, yb), float((y1 - yb).abs().max())
            torch.manual_seed(5)
            y0 = m0(f2, keep_, nq2, B, bz2)
            moved = torch.zeros(B, dtype=torch.bool).index_fill_(0, torch.tensor([0, 5, 7]), True)
            d = (y1 - y0).abs()
            assert float(d[moved[nq2]].max()) > 1e-4 and float(d[~moved[nq2]].max()) < 1e-5, arm
            with torch.no_grad():
                m0.eval()
                k0 = keep_.clone()
                k0[0, 1] = 0.0
                g_off = m0(f2, k0, nq2, B, bz2)
                m0.cm_gate.zero_()
                g_none = m0(f2, k0, nq2, B, bz2)
                assert torch.equal(g_off[nq2 == 0], g_none[nq2 == 0])             # the chain term drops with SEMB
                m0.train()
        # 4. toy fits through lean_gpu's loop: a repeat is bit-identical; at P_DROP 0 the fit is the base arm's bit for
        #    bit; with A4h it differs, and the last fit's counts are its own (about 0.1 of the cells)
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckph", c_m), ("zspkh", c_g)):
            with S.patched(BASE_OF[arm]):
                bfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                n1 = dict(LAST)
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                assert dict(LAST) == n1 and 0 < n1["dropped"] < n1["cells"], n1
                P_DROP = 0.0
                try:
                    zfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} p0")
                finally:
                    P_DROP = SETTINGS["p_drop"]
            assert not any(LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert not any(LG.same_state(a, b) for a, b in zip(zfit["states"] + [zfit["swa"]],
                                                               bfit["states"] + [bfit["swa"]]))
            assert LG.same_state(fit["swa"], bfit["swa"])
            assert not issubclass(S.ARMS[BASE_OF[arm]][0], A4HTop)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the refusals; the fit names give the split
    assert split_of_fit("scr-a4h-zspkh-hp") == ("zspkh", "L-hotpotqa") and split_of_fit("scr-a4-zspkd") == (None, None)
    good = ["train", "--arm", "zrckph", "--name", FIT["zrckph"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckph"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkh", "--name", FIT["zrckph"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkh", "--name", FIT["zspkh"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a4h-zspkh", {"arm": "zspkd", "a4": {}}),
                        ("scr-a4h-zspkh-hp", {"arm": "zspkh", "a4h": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, stamped A4h's; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T_ / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T_ / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T_ / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T_ / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "A4h" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A4h ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A4h's block dropout) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with the next form of section 5: gains with a loss -> A4h, no gain -> A4f
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckph": fake_recall(T_ / "a.json", "zrckph", {mi: "GAIN"}),
                       "zspkh": fake_recall(T_ / "b.json", "zspkh", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckph"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckph"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspkh"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspkh"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckph": {"call": "GAIN", "delta": 0.0},
                                                   "zspkh": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkh"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckph": fake_recall(T_ / "a.json", "zrckph", {}),
                       "zspkh": fake_recall(T_ / "b.json", "zspkh", {gm: "GAIN"}, stage="A4")}, T_ / "v")
        assert rec["arms"]["zrckph"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkh"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckph"]["next_form"] == NEXT["NO_GAIN"]
        assert rec["arms"]["zspkh"]["next_form"] == NEXT["INCOMPLETE"]
        rec = verdict({"zrckph": T_ / "none.json", "zspkh": fake_recall(T_ / "b.json", "zrckph", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckph"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkh"]["verdict"] == "INCOMPLETE"
    log(f"za4h selftest: A4HTop at the top of zrckp's and zspk's models, half bdrop20's rate, its seed; reads and P_DROP 0 "
        f"are the base bit for bit (scores, gradients, the global stream); with a mask the arm is the base model on "
        f"keep times the mask, bit for bit, the scores moving only on the masked questions, and the chain term drops "
        f"with SEMB; toy fits repeat bit for bit, equal the base's at P_DROP 0 and differ with A4h; train refuses another "
        f"arm, split, fit name, cache root or device; read refuses a fit that is not A4h's or not its split's; relz's "
        f"pair and re-call per arm, stamped A4h's; the verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
    return 0


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
            log(f"za4h {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za4h: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
