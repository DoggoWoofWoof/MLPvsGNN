"""A4fx (docs/A4FX_FAMILY_DROPOUT_ADOPTED.md): A4f's family dropout on the adopted arms, the next form A4f's verdict
named for zrckpf and zspkf (no GAIN; musique zero-shot -0.1147 and -0.2264). A4f (za4f.py) dropped, per question in
training, each of zret's blocks, the KT head's term and rmatch's chain term with probability 0.2, on zrckp's and zspk's
models. G5's zspx and R1's zrckpr have passed their screens since; A4fx moves the family form onto them, and on zrckpr
R1's nine columns become one more family that can drop:

    in a training forward, each question draws len(blocks) + 2 cells (zspx) or len(blocks) + 3 cells (zrckpr) from the
    model's own generator (seed + 101, bdrop20's, A4's and A4f's), each kept with probability 0.8: the first
    len(blocks) multiply its keep flags (A4 exactly), the next drops the KT head's term on the question's rows, the next
    drops rmatch's chain term on them, and on zrckpr the last drops R1's head's term on them; a read keeps everything

As in A4f the chain term stays gated by SEMB's flag as well, a dropped term is 0 on the question's rows and a kept one is
the read's term, unscaled. With A4f's two terms the cells are A4f's bit for bit (the selftest checks it against za4f's
arms). No new parameter, no score read; the global stream, dropout's masks included, is the base's.

    zrckprf  zrckpr's model (outputs/mp_unified/zr1res.py) trained with A4f's cells and the R1 cell, on zrckpr's carve
    zspxf    zspx's model (outputs/mp_unified/zg5mix.py) trained with A4f's cells; zspx's three increments (its step,
             G3's residual message, G4's second step) read the score they change

    python outputs/mp_unified/za4fx.py train --split L-musique --name scr-a4fx-zrckprf --arm zrckprf --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za4fx.py read --name scr-a4fx-zrckprf --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za4fx.py compare --new outputs/screen/fits/scr-a4fx-zrckprf \\
        --base outputs/screen/fits/scr-r1-zrckpr,outputs/step1/fits/L-musique --out outputs/screen/scr-a4fx-zrckprf
    python outputs/mp_unified/za4fx.py pair --arm zrckprf --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za4fx.py recall --arm zrckprf --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za4fx.py verdict
    python outputs/mp_unified/za4fx.py --selftest
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

import zg5mix as G5  # noqa: E402  (zspx and G3's carve; through it zg3res, zg4prop, zp1c3, zprop, zlink, rmatch)
import zr1res as ZR1  # noqa: E402  (zrckpr, its carve and R1's columns; through it zp1c3p)
import za4f as A4F  # noqa: E402  (A4f: its rate and seed; its arms, which the selftest holds A4fx's cells to)
import lean_screen3 as S3  # noqa: E402  (zret)

import numpy as np  # noqa: E402
import torch  # noqa: E402

G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
Z3P = ZR1.Z3P
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
LM = LG.LM
log = S.log

DECLARED = "docs/A4FX_FAMILY_DROPOUT_ADOPTED.md"
OUT = ROOT / "outputs" / "a4fx"
SCREEN = ROOT / "outputs" / "screen"
P_DROP = A4F.P_DROP                         # 0.2, A4f's (A4's and bdrop20's)
SEED_OFF = A4F.SEED_OFF                     # 101, A4f's: the cells come from the model's own generator, seed + 101
TERMS = {"zrckprf": ("kt", "ch", "r1"), "zspxf": ("kt", "ch")}     # the term cells after the blocks', in draw order
TERM_NAME = {"kt": "the KT head's term", "ch": "rmatch's chain term", "r1": "R1's head's term"}
SETTINGS = {"p_drop": P_DROP, "unit": "(question, cell): the model's blocks, then the KT head's term, then rmatch's "
                                      "chain term, then (zrckprf) R1's head's term", "seed": f"model seed + {SEED_OFF}",
            "terms": {a: list(t) for a, t in TERMS.items()},
            "where": "the top of the model's forward: every layer reading the keep flags sees the drop",
            "keep": "the incoming keep flag times the draw; a dropped term is 0 on the question's rows, a kept one "
                    "unscaled; the chain term also keeps SEMB's gate", "reads": "everything kept", "parameters": 0,
            "score_input": False, "parent": "A4f (docs/A4F_FAMILY_DROPOUT.md, outputs/mp_unified/za4f.py)"}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckprf": "zrckpr", "zspxf": "zspx"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a4fx-{a}", "L-hotpotqa": f"scr-a4fx-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckprf": {"L-musique": ("screen", "fits", "scr-r1-zrckpr"),
                         "L-hotpotqa": ("screen", "fits", "scr-r1-zrckpr-hp")},
             "zspxf": {"L-musique": ("screen", "fits", "scr-g5-zspx"),
                       "L-hotpotqa": ("screen", "fits", "scr-g5-zspx-hp")}}
BASE_SCREENS = {"zrckprf": {sp: f"outputs/p1gc/base-pools-zrckpr-{sp}.json" for sp in SPLITS},   # P1g-c's
                "zspxf": {sp: f"outputs/a9/base-zspx-{sp}.json" for sp in SPLITS}}                # A9's
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckprf": "zrckpr trained with A4f's family dropout and the R1 cell (0.2 per question on each block, the KT "
                   "term, the chain term and R1's term)",
        "zspxf": "zspx trained with A4f's family dropout (0.2 per question on each block, the KT term and the chain "
                 "term), before its three increments"}
PAIR = {a: SCREEN / f"scr-a4fx-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a4fx-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("metaqa", "musique", "webqsp")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A4fx's re-call of the arm, "
        "or not twelve reads (docs/A4FX_FAMILY_DROPOUT_ADOPTED.md, section 4)")
LAST = {"forwards": 0, "cells": 0, "dropped": 0, "questions_all_dropped": 0, "kt_dropped": 0, "chain_dropped": 0,
        "r1_dropped": 0, "questions": 0}    # the last fit's counts


# ── the dropout ──────────────────────────────────────────────────────────────


class A4FXTop(S3.ZRet):
    """A4f's A4FTop over the class's term cells: at the top of the arm's MRO, in a training forward, each question's
    len(blocks) + len(TERMS) cells drawn kept with probability 1 - P_DROP from the model's own generator; the keep
    flags times the first len(blocks), and each term times its own cell on the question's rows, in TERMS's order; then
    the base model's forward. With TERMS ("kt", "ch") it is A4f's A4FTop (the selftest holds it to za4f's arms)."""

    TERMS = ("kt", "ch")
    COUNT = {"kt": "kt_dropped", "ch": "chain_dropped", "r1": "r1_dropped"}

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.a4f_gen = torch.Generator().manual_seed(int(seed) + SEED_OFF)
        self.a4f_rows = None
        LAST.update({k: 0 for k in LAST})

    def a4f_draw(self, shape):
        return torch.rand(shape, generator=self.a4f_gen)

    def kt_head(self, x):
        h = super().kt_head(x)
        return h if self.a4f_rows is None else h * self.a4f_rows["kt"]

    def chain_feats(self, feats, n):
        fm, fr = super().chain_feats(feats, n)
        if self.a4f_rows is None:
            return fm, fr
        r = self.a4f_rows["ch"]
        return fm * r, fr * r

    def forward(self, feats, keep, nq, B, base_z):
        if not self.training:
            return super().forward(feats, keep, nq, B, base_z)
        nb = keep.shape[1]
        u = self.a4f_draw((int(keep.shape[0]), nb + len(self.TERMS))).to(keep.device)
        k = (u >= P_DROP).to(keep.dtype)
        LAST["forwards"] += 1
        LAST["cells"] += k.numel()
        LAST["dropped"] += int((k == 0).sum())
        LAST["questions_all_dropped"] += int((k[:, :nb].sum(1) == 0).sum())
        for i, t in enumerate(self.TERMS):
            LAST[self.COUNT[t]] += int((k[:, nb + i] == 0).sum())
        LAST["questions"] += int(k.shape[0])
        self.a4f_rows = {t: k[nq, nb + i] for i, t in enumerate(self.TERMS)}
        try:
            return super().forward(feats, keep * k[:, :nb], nq, B, base_z)
        finally:
            self.a4f_rows = None


class R1Cell(A4FXTop):
    """A4FXTop with R1's head as a third term: the head's output times the question's R1 cell on its rows (the KT
    term's treatment), so a question with that cell off sees zrckpr's score without its columns."""

    TERMS = ("kt", "ch", "r1")

    def r1_head(self, x):
        h = super().r1_head(x)
        return h if self.a4f_rows is None else h * self.a4f_rows["r1"]


class ZRCKPRF(R1Cell, ZR1.ZRCKPR):
    """zrckprf: zrckpr's model trained with A4f's cells and the R1 cell (MRO: R1Cell, A4FXTop, ZRCKPR, ZRCKP, KTMixP, ZRM,
    ChainMatch, ZRet)."""


class ZSPXF(A4FXTop, G5.ZSPX):
    """zspxf: zspx's model trained with A4f's cells; zspx's three increments read the score they change (MRO: A4FXTop,
    ZSPX, ZSPR, ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZRet)."""


ARMS = {"zrckprf": (ZRCKPRF, ZR1.R1KTChainCarveZRC), "zspxf": (ZSPXF, G3.ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


SIDE_KEY = {"zrckprf": "r1_records", "zspxf": "res_records"}


def side_records(arm, carves):
    """The records of the arm's own side input on carves: R1's columns for zrckprf, G3's c and s for zspxf; stops when
    one is missing or not its stage's."""
    return ZR1.r1_records(carves) if arm == "zrckprf" else G3.res_records(carves)


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
        rec.update({"za4fx_sha256": LC.sha_src(__file__), "za4f_sha256": LC.sha_src(A4F.__file__),
                    "zg5mix_sha256": LC.sha_src(G5.__file__), "zg3res_sha256": LC.sha_src(G3.__file__),
                    "zg4prop_sha256": LC.sha_src(G4.__file__), "zr1res_sha256": LC.sha_src(ZR1.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "lean_screen3_sha256": LC.sha_src(S3.__file__),
                    "lean_screen_sha256": LC.sha_src(S.__file__),
                    "a4fx": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                             "device": Z2.DEVICE, "settings": SETTINGS, "terms": list(TERMS[arm]),
                             "last_fit": {**LAST, "dropped_share": LAST["dropped"] / max(LAST["cells"], 1)},
                             "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                             "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                             SIDE_KEY[arm]: side_records(arm, Z3.train_carves(split)),
                             "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                             "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)
        log(f"  a4fx {arm} {split}: {LAST['dropped']} of {LAST['cells']} (question, cell) cells dropped in the last fit "
            f"({LAST['dropped'] / max(LAST['cells'], 1):.4f}), {LAST['questions_all_dropped']} questions with every "
            f"block dropped; "
            + ", ".join(f"{TERM_NAME[t]} dropped on {LAST[A4FXTop.COUNT[t]]}" for t in TERMS[arm])
            + f" of {LAST['questions']} questions")


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za4fx: train takes --arm zrckprf or zspxf, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za4fx: A4fx's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za4fx: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    side_records(arm, Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a4fx" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za4fx: {name} is not an A4fx fit (arm {rec.get('arm')})")
    if rec["a4fx"].get("split") != split:
        raise SystemExit(f"za4fx: {name} is {split}'s fit, its record says {rec['a4fx'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    side_records(arm, Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za4fx: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za4fx: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A4fx's family dropout) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A4fx ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A4fx", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za4fx_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": "A4fxh, the family form at half the rate (0.1 per question on each cell) on the adopted arms, "
                         "its own declared file",
        "NO_GAIN": "A4fxt, the term cells alone on the adopted arms (the KT, chain and, on zrckpr, R1 terms dropped at "
                   "0.2 per question; zret's blocks never), its own declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A4fx's screen verdict, per arm, from its re-call (section 4); bdrop20's reads beside (musique zero-shot, its gain;
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
            if rec.get("stage") != "A4fx" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A4fx's re-call of {a}")
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
    rec = {"stage": "A4fx", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za4fx_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A4fx's screen verdict: zrckprf **{per['zrckprf']['verdict']}**, zspxf **{per['zspxf']['verdict']}**",
          "", f"Rule: {RULE}.", ""]
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
           "", "This verdict closes only this form (family dropout 0.2 per question on each block, the KT term, the chain "
               "term and, on zrckpr, R1's term, at the top of the forward, in training, on the adopted arms); "
               "docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za4fx verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


OFF = {"kt": ("kt_w2", "kt_b2"), "ch": ("cm_gate",), "r1": ("r1_w2", "r1_b2")}     # zeroed, each term is 0 everywhere


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A4fx", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def set_heads(*ms):
    """Every term and head away from zero, the same in each model, so a dropped term moves its rows' scores."""
    with torch.no_grad():
        for m_ in ms:
            for p_ in m_.out.parameters():
                p_.copy_(torch.linspace(-0.3, 0.3, p_.numel()).reshape(p_.shape))
            m_.cm_gate.copy_(torch.tensor([0.7, -0.4]))
            m_.kt_w2.copy_(torch.linspace(-0.5, 0.5, m_.kt_w2.numel()).reshape(m_.kt_w2.shape))
            m_.kt_b2.fill_(0.1)
            if hasattr(m_, "r1_w2"):
                m_.r1_w2.copy_(torch.linspace(0.6, -0.6, m_.r1_w2.numel()).reshape(m_.r1_w2.shape))
                m_.r1_b2.fill_(-0.05)
            for j, w in enumerate(("sp_w2", "rs_w2", "p2_w2")):
                if hasattr(m_, w):
                    getattr(m_, w).copy_(torch.randn(getattr(m_, w).shape, generator=torch.Generator().manual_seed(8 + j)))


def selftest():
    import copy
    import shutil
    import tempfile
    global P_DROP
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: A4FXTop at the top of zrckpr's and zspx's models (R1Cell above it on zrckpr); the base arms and A4f's
    #    unchanged; A4f's rate and seed
    assert S.ARMS["zrckprf"] == (ZRCKPRF, ZR1.R1KTChainCarveZRC) and S.ARMS["zspxf"] == (ZSPXF, G3.ResKTLinkCarveBase)
    assert S.ARMS["zrckpr"] == (ZR1.ZRCKPR, ZR1.R1KTChainCarveZRC) and S.ARMS["zspx"] == (G5.ZSPX, G3.ResKTLinkCarveBase)
    assert S.ARMS["zrckpf"] == (A4F.ZRCKPF, Z3.KTChainCarveZRC) and S.ARMS["zspkf"] == (A4F.ZSPKF, Z3.KTLinkCarveBase)
    assert ZRCKPRF.__mro__[1:9] == (R1Cell, A4FXTop, ZR1.ZRCKPR, Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, S3.ZRet)
    assert ZSPXF.__mro__[1:10] == (A4FXTop, G5.ZSPX, G3.ZSPR, Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch,
                                   S3.ZRet)
    assert all(ARMS[a][0].TERMS == TERMS[a] for a in ARM_NAMES) and not issubclass(A4FXTop, A4F.A4FTop)
    assert P_DROP == 0.2 == A4F.P_DROP == S.BDrop.P and SEED_OFF == 101 == A4F.SEED_OFF
    assert BASE_SCREENS["zspxf"] == {sp: f"outputs/a9/base-zspx-{sp}.json" for sp in SPLITS}
    rng = np.random.default_rng(85)
    tmp = Path(tempfile.mkdtemp(prefix="za4fx_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, _, _ = G3.toy_embed(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        for pt in (0, 1):
            G3.build("metaqa", "toy", pt, out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look",
                     lk_root=tmp / "lk", nodes=G3.ToyNodes(T), freeze="toy")
        ZR1.build("metaqa", "toy", out_root=tmp / "r1", cache_root=tmp / "cache", lk_root=tmp / "lk",
                  res_root=tmp / "res")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_ch = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        kt_ch, kt_lk = Z3.kt_carve(cls_ch, tmp / "kt"), Z3.kt_carve(cls_lk, tmp / "kt")
        c_m = ZR1.r1_carve(kt_ch, tmp / "r1")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")      # zrckpr's, in toy
        c_g = G3.res_carve(kt_lk, tmp / "res")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")     # zspx's, in toy
        c_mk = kt_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")                               # zrckp's (A4f's)
        c_gk = kt_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache")                               # zspk's (A4f's)
        blocks = ["rank", "SEMB"]
        nb = len(blocks)
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, nb)
        keep_[3, 1] = 0.0
        assert S.BDrop(blocks, c_m.widths, 16, seed=7).mask_seed == 7 + SEED_OFF
        # 2. the cells: A4f's generator and rate; a draw of many cells keeps about 0.8
        g = torch.Generator().manual_seed(0 + SEED_OFF)
        u = torch.rand((20000, nb + 3), generator=g)
        assert abs(float((u >= P_DROP).float().mean()) - 0.8) < 0.01
        # 3. A4f's cells exactly: A4FXTop with A4f's two terms over zrckp's and zspk's models is za4f's zrckpf and
        #    zspkf in training, bit for bit (scores, every gradient, the generators' states and the counts)

        class XP(A4FXTop, Z3P.ZRCKP):
            """A4FXTop over zrckp: A4f's zrckpf."""

        class XK(A4FXTop, Z3.ZSPK):
            """A4FXTop over zspk: A4f's zspkf."""

        for mine, theirs, cv in ((XP, A4F.ZRCKPF, c_mk), (XK, A4F.ZSPKF, c_gk)):
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            ma = theirs(blocks, cv.widths, 16, dropout=0.1, seed=0)
            torch.manual_seed(0)
            mb = mine(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert not LG.same_state(ma.state_dict(), mb.state_dict())
            set_heads(ma, mb)
            ma.train()
            mb.train()
            for rep in range(3):
                torch.manual_seed(3 + rep)
                ya = ma(f2, keep_, nq2, B, bz2)
                torch.manual_seed(3 + rep)
                yb = mb(f2, keep_, nq2, B, bz2)
                assert torch.equal(ya, yb), (theirs.__name__, rep)
                ma.zero_grad()
                mb.zero_grad()
                (ya * torch.linspace(-1, 1, ya.numel())).sum().backward()
                (yb * torch.linspace(-1, 1, yb.numel())).sum().backward()
                for n_, pa in ma.named_parameters():
                    ga, gb = pa.grad, dict(mb.named_parameters())[n_].grad
                    assert (ga is None and gb is None) or torch.equal(ga, gb), (theirs.__name__, rep, n_)
            assert torch.equal(ma.a4f_gen.get_state(), mb.a4f_gen.get_state())
            assert {k: A4F.LAST[k] for k in A4F.LAST} == {k: LAST[k] for k in A4F.LAST} and LAST["r1_dropped"] == 0
            assert 0 < LAST["dropped"] < LAST["cells"] == 3 * B * (nb + 2), dict(LAST)
        # 4. each arm against its base: in eval mode bit for bit and no draw; in training at P_DROP 0 bit for bit,
        #    scores and every gradient, and the global stream untouched; with fixed cells, each question's rows are the
        #    base model's on keep times its block cells, with the KT head's last layer at zero where its KT cell is off,
        #    rmatch's gate at zero where its chain cell is off and R1's head's last layer at zero where its R1 cell is
        #    off, bit for bit, the scores moving only on those questions; a question whose SEMB is dropped loses the
        #    chain term
        for arm, cls, base, cv in (("zrckprf", ZRCKPRF, ZR1.ZRCKPR, c_m), ("zspxf", ZSPXF, G5.ZSPX, c_g)):
            terms = TERMS[arm]
            nt = len(terms)
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            assert Z3.KT_KEY in f2 and RM.CH_KEY in f2
            assert (ZR1.R1_KEY in f2) == (arm == "zrckprf") and (G3.RES_KEY in f2) == (arm == "zspxf")
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1_ = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            assert not LG.same_state(m0.state_dict(), m1_.state_dict())
            set_heads(m0, m1_)
            m0.eval()
            m1_.eval()
            g_state = m1_.a4f_gen.get_state()
            assert torch.equal(m0(f2, keep_, nq2, B, bz2), m1_(f2, keep_, nq2, B, bz2))
            assert torch.equal(m1_.a4f_gen.get_state(), g_state) and m1_.a4f_rows is None
            m0.train()
            m1_.train()
            P_DROP = 0.0
            try:
                for rep in range(2):
                    torch.manual_seed(3 + rep)
                    y0 = m0(f2, keep_, nq2, B, bz2)
                    r_0 = torch.get_rng_state()
                    torch.manual_seed(3 + rep)
                    y1 = m1_(f2, keep_, nq2, B, bz2)
                    assert torch.equal(y0, y1) and torch.equal(torch.get_rng_state(), r_0), (arm, rep)
                    m0.zero_grad()
                    m1_.zero_grad()
                    (y0 * torch.linspace(-1, 1, y0.numel())).sum().backward()
                    (y1 * torch.linspace(-1, 1, y1.numel())).sum().backward()
                    for n_, p0 in m0.named_parameters():
                        g0, g1 = p0.grad, dict(m1_.named_parameters())[n_].grad
                        assert (g0 is None and g1 is None) or torch.equal(g0, g1), (arm, rep, n_)
            finally:
                P_DROP = SETTINGS["p_drop"]
            u_ = torch.full((B, nb + nt), 0.9)
            u_[0, 1] = u_[5, 0] = u_[7, 0] = u_[7, 1] = 0.05                   # SEMB off on 0, rank off on 5, both on 7
            u_[2, nb] = u_[4, nb + 1] = u_[6, nb] = u_[6, nb + 1] = 0.05        # KT off on 2, chain on 4, both on 6
            if nt == 3:
                u_[1, nb + 2] = u_[8, nb] = u_[8, nb + 2] = 0.05                # R1 off on 1, KT and R1 on 8
                u_[10, nb:] = 0.05                                              # every term off on 10
            mask = (u_ >= P_DROP).float()
            saved_draw = m1_.a4f_draw
            m1_.a4f_draw = lambda shape: u_
            LAST.update({k: 0 for k in LAST})
            try:
                torch.manual_seed(5)
                y1 = m1_(f2, keep_, nq2, B, bz2)
            finally:
                m1_.a4f_draw = saved_draw
            assert m1_.a4f_rows is None
            want = (B * (nb + nt), 14, 4, 3, 3, 1, B) if nt == 3 else (B * (nb + nt), 8, 2, 2, 0, 1, B)
            assert (LAST["cells"], LAST["dropped"], LAST["kt_dropped"], LAST["chain_dropped"], LAST["r1_dropped"],
                    LAST["questions_all_dropped"], LAST["questions"]) == want, dict(LAST)
            yb = torch.zeros_like(y1)
            for i in range(2 ** nt):
                on = tuple(((i >> j) & 1) == 0 for j in range(nt))
                mv = copy.deepcopy(m0)
                with torch.no_grad():
                    for t, o in zip(terms, on):
                        for w in (() if o else OFF[t]):
                            getattr(mv, w).zero_()
                torch.manual_seed(5)
                y_on = mv(f2, keep_ * mask[:, :nb], nq2, B, bz2)
                sel = torch.ones_like(nq2, dtype=torch.bool)
                for j, o in enumerate(on):
                    r_on = mask[nq2, nb + j].bool()
                    sel = sel & (r_on if o else ~r_on)
                yb = torch.where(sel, y_on, yb)
            assert torch.equal(y1, yb), (arm, float((y1 - yb).abs().max()))
            torch.manual_seed(5)
            y0 = m0(f2, keep_, nq2, B, bz2)
            moved_q = [0, 2, 4, 5, 6, 7] + ([1, 8, 10] if nt == 3 else [])
            moved = torch.zeros(B, dtype=torch.bool).index_fill_(0, torch.tensor(moved_q), True)
            d = (y1 - y0).abs()
            assert float(d[moved[nq2]].max()) > 1e-4 and float(d[~moved[nq2]].max()) < 1e-5, arm
            for j in ((1, 2, 4, 6) if nt == 3 else (2, 4, 6)):
                assert float(d[nq2 == j].max()) > 1e-4, (arm, j)                   # each term's own cell moves its rows
            with torch.no_grad():
                m0.eval()
                k0 = keep_.clone()
                k0[0, 1] = 0.0
                g_off = m0(f2, k0, nq2, B, bz2)
                m0.cm_gate.zero_()
                g_none = m0(f2, k0, nq2, B, bz2)
                assert torch.equal(g_off[nq2 == 0], g_none[nq2 == 0])             # the chain term drops with SEMB
                m0.train()
        # 5. toy fits through lean_gpu's loop: a repeat is bit-identical; at P_DROP 0 the fit is the base arm's bit for
        #    bit; with A4fx it differs, and the last fit's counts are its own (about 0.2 of the cells; R1's cell only
        #    on zrckprf)
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckprf", c_m), ("zspxf", c_g)):
            with S.patched(BASE_OF[arm]):
                bfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                n1 = dict(LAST)
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                assert dict(LAST) == n1 and 0 < n1["dropped"] < n1["cells"], n1
                assert n1["kt_dropped"] > 0 and n1["chain_dropped"] > 0 and (n1["r1_dropped"] > 0) == (arm == "zrckprf")
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
            assert not issubclass(S.ARMS[BASE_OF[arm]][0], A4FXTop)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the refusals; the fit names give the split
    assert split_of_fit("scr-a4fx-zspxf-hp") == ("zspxf", "L-hotpotqa") and split_of_fit("scr-a4f-zspkf") == (None, None)
    good = ["train", "--arm", "zrckprf", "--name", FIT["zrckprf"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckpr", "--name", FIT["zrckprf"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zrckpf", "--name", FIT["zrckprf"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspxf", "--name", FIT["zrckprf"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspxf", "--name", FIT["zspxf"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckpr", "r1": {}}), ("scr-a4fx-zspxf", {"arm": "zspkf", "a4f": {}}),
                        ("scr-a4fx-zspxf-hp", {"arm": "zspxf", "a4fx": {"split": "L-musique"}}),
                        ("scr-a4fx-zrckprf", {"arm": "zrckprf", "a4f": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 7. relz's pair and re-call for each arm against its base arm's card fits, stamped A4fx's; relz restored after
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
            assert pj["stage"] == "A4fx" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A4fx ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A4fx's family dropout) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 8. the verdict, per arm, with the next form of section 5: gains with a loss -> A4fxh, no gain -> A4fxt
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckprf": fake_recall(T_ / "a.json", "zrckprf", {mi: "GAIN"}),
                       "zspxf": fake_recall(T_ / "b.json", "zspxf", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckprf"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckprf"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspxf"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspxf"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckprf": {"call": "GAIN", "delta": 0.0},
                                                   "zspxf": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspxf"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckprf": fake_recall(T_ / "a.json", "zrckprf", {}),
                       "zspxf": fake_recall(T_ / "b.json", "zspxf", {gm: "GAIN"}, stage="A4f",
                                            declared_in=A4F.DECLARED)}, T_ / "v")
        assert rec["arms"]["zrckprf"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspxf"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckprf"]["next_form"] == NEXT["NO_GAIN"]
        assert rec["arms"]["zspxf"]["next_form"] == NEXT["INCOMPLETE"]
        rec = verdict({"zrckprf": T_ / "none.json", "zspxf": fake_recall(T_ / "b.json", "zrckprf", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckprf"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspxf"]["verdict"] == "INCOMPLETE"
        rows11 = fake_recall(T_ / "c.json", "zspxf", {gm: "GAIN"})
        rc11 = json.loads(rows11.read_text(encoding="utf-8"))
        rc11["rows"] = rc11["rows"][:11]
        LC.write_json(rows11, rc11)
        rec = verdict({"zrckprf": fake_recall(T_ / "a.json", "zrckprf", {gm: "GAIN"}), "zspxf": rows11}, T_ / "v")
        assert rec["arms"]["zrckprf"]["verdict"] == "ADOPT_FOR_FULL_RUN" and rec["arms"]["zspxf"]["verdict"] == "INCOMPLETE"
    log(f"za4fx selftest: A4FXTop at the top of zrckpr's and zspx's models (R1Cell above it on zrckpr), A4f's rate and "
        f"seed over the blocks plus the KT term, the chain term and on zrckpr R1's term; with A4f's two terms over zrckp "
        f"and zspk it is za4f's zrckpf and zspkf bit for bit in training (scores, gradients, generators, counts); reads "
        f"and P_DROP 0 are the base bit for bit (scores, gradients, the global stream); with fixed cells each question's "
        f"rows are the base model's on its block cells with the KT head, rmatch's gate or R1's head at zero where their "
        f"cells are off, bit for bit, the scores moving only on those questions, and the chain term drops with SEMB; toy "
        f"fits repeat bit for bit, equal the base's at P_DROP 0 and differ with A4fx; train refuses another arm, split, "
        f"fit name, cache root or device; read refuses a fit that is not A4fx's or not its split's; relz's pair and "
        f"re-call per arm, stamped A4fx's; the verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
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
            log(f"za4fx {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za4fx: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
