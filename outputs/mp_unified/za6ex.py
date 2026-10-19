"""A6ex (docs/A6EX_LINK_WORDS_ADOPTED.md): the words around a link on the adopted arms, the form A6e's verdict named
for no GAIN on both of its arms. A6e (za6e.py) gave D4's link score to zrckp as eight split columns over the first
stage's top ten rows and to zspk as edge weights in zsp's step; both arms were NOT_ADOPTED with no GAIN and no LOSS.
A6 and A6e built on zrckp and zspk; the adopted arms are G5's zspx (zg5mix.py) and R1's zrckpr (zr1res.py), and
A6e's section 5 fixed the next form before any number: the same inputs on those arms, on both models, the idea
closing as a model input only if A6ex finds no GAIN either.

    MLP (zrckprn)  zrckpr's model (zrckp plus R1's head over its nine columns) plus A6e's head over A6e's eight columns,
                   R1's pointwise head at eight inputs (Linear(8, 16), GELU, Linear(16, 1); 161 parameters, the last
                   layer zero at the start, the first drawn from its own generator, seed + 6601)
    GNN (zspxl)    zspx's three steps, each with A6e's six link-weighted inputs (per word family the link-weighted mean
                   of the row's neighbours' z-scores and log1p of its summed weight, at the step's own input score) in
                   the first layer of its head through its own columns, zero at the start (3 x 192 = 576 parameters):

        s0 = zspk's score before its step (zrc's scorer and the KT head)
        s1 = s0 + h_sp(prop_inputs_grad(s0), e(s0))      zspk's step, its ten inputs with the gradient kept   ew_w1
        s2 = s1 + h_rs(res_inputs(s1, c, s), e(s1))      G3's residual message, no gradient                   ew_rs_w1
        s3 = s2 + h_p2(prop_inputs(s2), e(s2))           G4's second step, no gradient                        ew_p2_w1

Every step of zspx reads zlink's edges and none reads a link's words: zspk's step and G4's second step take each
family's links alike, G3's message weighs a link by its parent's score and its residual. A6e's GNN form put the words
in zsp's one step; on zspx each of the three reads them, through its own columns (the heads are separate and each
reads its own score). The six inputs are A6e's (za6e.ew_inputs: the detached score's z), so they carry no gradient to
the scorer; at zero the arm is zspx bit for bit, scores and every shared gradient.

The arrays are A6e's, unchanged (outputs/a6e: the columns and the weights, za6e.py build). No parameter, label or
score of a model enters a column or a weight; zrckprn reads only its own row's columns (no message passing, nothing
normalised over the question's scores); zspxl's steps are message passing, as zspx's are.

    python outputs/mp_unified/za6ex.py train --split L-musique --name scr-a6ex-zspxl --arm zspxl --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za6ex.py read --name scr-a6ex-zspxl --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za6ex.py compare --new outputs/screen/fits/scr-a6ex-zspxl \\
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a6ex-zspxl
    python outputs/mp_unified/za6ex.py compare --new outputs/screen/fits/scr-r1-zrckpr \\
        --base outputs/step1/fits/L-musique --out outputs/a6ex/base-zrckpr-L-musique
    python outputs/mp_unified/za6ex.py pair --arm zspxl --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za6ex.py recall --arm zspxl --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za6ex.py verdict
    python outputs/mp_unified/za6ex.py --selftest
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

import za6e as ZA6E  # noqa: E402  (A6e: the columns and weights, their carves, the head, the weighted inputs, the build)
import zg5mix as G5  # noqa: E402  (zspx; through it G3's residual carve and inputs, G4's steps, zp1c3, zprop, zlink)
import zr1res as R1  # noqa: E402  (zrckpr and its carve, R1's columns)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

G3, G4 = G5.G3, G5.G4
ZA6, LW, Z3, Z3P = ZA6E.ZA6, ZA6E.LW, ZA6E.Z3, ZA6E.Z3P
Z2 = Z3.Z2
ZP, ZL, ZM, RM = Z3.ZP, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log
A6E_KEY, EW_KEY = ZA6E.A6E_KEY, ZA6E.EW_KEY

DECLARED = "docs/A6EX_LINK_WORDS_ADOPTED.md"
OUT = ROOT / "outputs" / "a6ex"
SCREEN = ROOT / "outputs" / "screen"
STEPS = {"ew_w1": "zspk's step, its inputs with the gradient kept (zspkg): s1 = s0 + h_sp(prop_inputs_grad(s0), e(s0))",
         "ew_rs_w1": "G3's residual message (zspr): s2 = s1 + h_rs(res_inputs(s1, c, s), e(s1))",
         "ew_p2_w1": "G4's second step (zspk2): s3 = s2 + h_p2(prop_inputs(s2), e(s2))"}
EWX = tuple(STEPS)
N_EWX = (ZP.HID + G3.RS_HID + G4.P2_HID) * ZA6E.NE
SETTINGS = {"mlp": {"base": "zrckpr (zrckp plus R1's head over its nine columns)", "columns": ZA6E.NC,
                    "hidden": ZA6E.A6E_HID, "seed_offset": ZA6E.A6E_SEED_OFF, "parameters": ZA6E.N_HEAD,
                    "score_input": False, "build": ZA6E.COL_SETTINGS},
            "gnn": {"base": "zspx (G5: zspk's step end to end, G3's residual message, G4's second step)",
                    "inputs": ZA6E.NE, "steps": STEPS,
                    "into": "the first layer of each step's head, through the step's own columns, zero at the start",
                    "parameters": N_EWX,
                    "z": "zsp's: the step's input score, detached, z-scored in its pool, clipped to +-8",
                    "build": ZA6E.EW_SETTINGS},
            "arrays": "A6e's (outputs/a6e, za6e.py build), unchanged", "ents": ZA6E.SETTINGS["ents"]}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckprn": "zrckpr", "zspxl": "zspx"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a6ex-{a}", "L-hotpotqa": f"scr-a6ex-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckprn": {"L-musique": ("screen", "fits", "scr-r1-zrckpr"),
                         "L-hotpotqa": ("screen", "fits", "scr-r1-zrckpr-hp")},
             "zspxl": {"L-musique": ("screen", "fits", "scr-g5-zspx"),
                       "L-hotpotqa": ("screen", "fits", "scr-g5-zspx-hp")}}
BASE_SCREENS = {"zrckprn": {sp: f"outputs/a6ex/base-zrckpr-{sp}.json" for sp in SPLITS},     # this stage's items
                "zspxl": {sp: f"outputs/a9/base-zspx-{sp}.json" for sp in SPLITS}}           # A9's
BASE_KEY = {"zrckprn": "r1_records", "zspxl": "res_records"}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckprn": "zrckpr plus A6e's head over the eight link-word columns",
        "zspxl": "zspx with A6e's link-weighted inputs in each of its three steps"}
PAIR = {a: SCREEN / f"scr-a6ex-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a6ex-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("musique", "webqsp", "hotpotqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A6ex's re-call of the arm, "
        "or not twelve reads (docs/A6EX_LINK_WORDS_ADOPTED.md, section 4)")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── the carves: the base arms' own, with A6e's arrays beside ─────────────────


A6eR1KTChainCarveZRC = ZA6E.cols_carve(R1.R1KTChainCarveZRC)         # zrckpr's (R1's nine columns, no edge) + A6e's
EWResKTLinkCarveBase = ZA6E.ew_carve(G3.ResKTLinkCarveBase)          # zspx's (edges, c and s) + A6e's weights


# ── the models ───────────────────────────────────────────────────────────────


class ZRCKPRN(R1.ZRCKPR):
    """zrckprn: zrckpr's model plus A6e's pointwise head (zero at the start) over each row's own eight columns."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        ZA6E.a6e_init(self, seed)

    a6e_head = ZA6E.a6e_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if A6E_KEY not in feats:
            return s
        return s + self.a6e_head(feats[A6E_KEY])


def step(x, e, w1, b1, ew, w2, b2):
    """One of zspx's heads (Linear, GELU, Linear) with A6e's six weighted inputs e entering its first layer through
    their own columns ew: at ew = 0 the head is zspx's bit for bit."""
    return Fn.linear(Fn.gelu(Fn.linear(x, w1, b1) + Fn.linear(e, ew)), w2, b2).squeeze(-1)


class ZSPXL(G5.ZSPX):
    """zspxl: zspx with the worded links in each of its three steps: each step's head reads its own inputs and, through
    its own columns (zero at the start), A6e's six link-weighted inputs at the score it acts on (MRO: ZSPX, ZSPR, ZSPK,
    ZProp, KTMix, ZRM, ...)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.ew_w1 = nn.Parameter(torch.zeros(ZP.HID, ZA6E.NE))
        self.ew_rs_w1 = nn.Parameter(torch.zeros(G3.RS_HID, ZA6E.NE))
        self.ew_p2_w1 = nn.Parameter(torch.zeros(G4.P2_HID, ZA6E.NE))

    def forward(self, feats, keep, nq, B, base_z):
        if EW_KEY not in feats or ZL.LK_KEY not in feats:
            return super().forward(feats, keep, nq, B, base_z)
        s = super(ZP.ZProp, self).forward(feats, keep, nq, B, base_z)       # zspx's s0, before its steps
        lk, ew = feats[ZL.LK_KEY], feats[EW_KEY]
        s = s + step(G4.prop_inputs_grad(s, lk, nq, B), ZA6E.ew_inputs(s, ew, lk, nq, B), self.sp_w1, self.sp_b1,
                     self.ew_w1, self.sp_w2, self.sp_b2)
        if G3.RES_KEY in feats:
            s = s + step(G3.res_inputs(s, lk, feats[G3.RES_KEY], nq, B), ZA6E.ew_inputs(s, ew, lk, nq, B),
                         self.rs_w1, self.rs_b1, self.ew_rs_w1, self.rs_w2, self.rs_b2)
        return s + step(ZP.prop_inputs(s, lk, nq, B), ZA6E.ew_inputs(s, ew, lk, nq, B), self.p2_w1, self.p2_b1,
                        self.ew_p2_w1, self.p2_w2, self.p2_b2)


ARMS = {"zrckprn": (ZRCKPRN, A6eR1KTChainCarveZRC), "zspxl": (ZSPXL, EWResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def base_records(arm, carves):
    """The base arm's own arrays of each carve part: R1's columns (zrckpr) or G3's residual parts (zspx); stops when
    one is missing or not its stage's."""
    return R1.r1_records(carves) if arm == "zrckprn" else G3.res_records(carves)


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        cv = Z3.train_carves(split)
        rec.update({"za6ex_sha256": LC.sha_src(__file__), "za6e_sha256": LC.sha_src(ZA6E.__file__),
                    "za6lw_sha256": LC.sha_src(ZA6.__file__), "linkwords_sha256": LC.sha_src(LW.__file__),
                    "zg5mix_sha256": LC.sha_src(G5.__file__), "zg3res_sha256": LC.sha_src(G3.__file__),
                    "zg4prop_sha256": LC.sha_src(G4.__file__), "zr1res_sha256": LC.sha_src(R1.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "a6ex": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                        "device": Z2.DEVICE, "added": list(ZA6E.HEAD if arm == "zrckprn" else EWX),
                        "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "a6e_records": ZA6E.a6e_records(cv),
                        "ents_records": ZA6.ents_records(sorted({d for d, _ in cv})),
                        BASE_KEY[arm]: base_records(arm, cv), "kt_records": Z3.kt_records(cv),
                        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def split_of_fit(name):
    for a in ARM_NAMES:
        for sp, n in FIT[a].items():
            if n == name:
                return a, sp
    return None, None


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za6ex: train takes --arm zrckprn or zspxl, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za6ex: A6ex's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za6ex: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    ZA6E.a6e_records(Z3.train_carves(split))
    base_records(arm, Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a6ex" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za6ex: {name} is not an A6ex fit (arm {rec.get('arm')})")
    if rec["a6ex"].get("split") != split:
        raise SystemExit(f"za6ex: {name} is {split}'s fit, its record says {rec['a6ex'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    ZA6E.a6e_records(Z3.eval_carves())
    base_records(arm, Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za6ex: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za6ex: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A6e's link words) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A6ex ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A6ex", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za6ex_sha256": LC.sha_src(__file__)})
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


NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools (A6e's columns and weights built again on P1g-b's "
                              "looks), then its full run, its own declared file",
        "GAIN_AND_LOSS": "A6exp, the passage-graph back-off on the adopted arms: the same arms with the typed graphs' "
                         "relation-name words left out (the relation columns and weights 0), so only title windows and "
                         "shared entities move either model, on both models, its own declared file",
        "NO_GAIN_OTHER_ADOPT": {
            "zspxl": "A6exc: A6e's eight columns on zspx as well, through A6e's pointwise head added to the score "
                     "before zspx's first step reads it (where A6's and R1's GNN arms put their heads), so the GNN "
                     "reads the input the MLP gained with (docs/SCREENS.md 3b), its own declared file",
            "zrckprn": "A6exw: the part of the edge weights the MLP may read without a message: per word family, "
                       "log1p of the row's summed link weight over its zlink edges (A6e's weights, no score of any "
                       "row), three columns beside zrckpr's inputs through A6e's pointwise head at three inputs, its "
                       "own declared file"},
        "NO_GAIN_WAIT": "nothing closes until the other arm's missing re-call is run; then section 5 applies",
        "NO_GAIN": "the link-word idea closes as a model input: A6, A6e and A6ex gave it to both models as learned "
                   "inputs that start at zero (columns on the MLP; columns, then edge weights, then edge weights in "
                   "every step that reads edges on the GNN), on the old bases and on the adopted ones, and no re-call "
                   "found a GAIN; D4 stays a diagnostic",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}
IDEA = {"OPEN": "a GAIN on at least one arm; the next forms above carry the words on",
        "WAITING": "a re-call is incomplete; nothing closes until it is run",
        "CLOSED_AS_A_MODEL_INPUT": "no GAIN on either adopted arm, after none in A6 and A6e: the idea has reached both "
                                   "models as a learned input on both bases (section 5)"}


def next_form(arm, v, gains, other):
    """Section 5: the arm's own outcome; without a GAIN of its own, the other arm's re-call decides: its back-off
    (A6exp, on both models), the input the other model gained with in this model's form, the wait, or the close."""
    if v != "NOT_ADOPTED":
        return NEXT[v]
    if gains:
        return NEXT["GAIN_AND_LOSS"]
    if other["verdict"] == "INCOMPLETE":
        return NEXT["NO_GAIN_WAIT"]
    if not other["gains"]:
        return NEXT["NO_GAIN"]
    return NEXT["GAIN_AND_LOSS"] if other["verdict"] == "NOT_ADOPTED" else NEXT["NO_GAIN_OTHER_ADOPT"][arm]


def idea_of(per):
    """The link-word idea as a model input: OPEN while a complete re-call has a GAIN, WAITING while one is incomplete,
    closed when both are complete without a GAIN."""
    if any(p["gains"] and p["verdict"] != "INCOMPLETE" for p in per.values()):
        return "OPEN"
    if any(p["verdict"] == "INCOMPLETE" for p in per.values()):
        return "WAITING"
    return "CLOSED_AS_A_MODEL_INPUT"


def verdict(recalls=None, out=None):
    """A6ex's screen verdict, per arm, from its re-call (section 4); musique (D4's target), webqsp (A6's loss) and
    hotpotqa beside, musique's in-domain read of both arms side by side, the next form each outcome names and the idea's
    state (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    got = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A6ex" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A6ex's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        got[a] = (rp, problems, rows, gains, losses, v)
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows, gains, losses, v = got[a]
        o = got[next(b for b in ARM_NAMES if b != a)]
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "next_form": next_form(a, v, gains, {"verdict": o[5], "gains": o[3]}),
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    idea = idea_of(per)
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A6ex", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "link_word_idea": idea, "idea_reason": IDEA[idea], "za6ex_sha256": LC.sha_src(__file__), "utc": utc()}
    md = [f"# A6ex's screen verdict: zrckprn **{per['zrckprn']['verdict']}**, zspxl **{per['zspxl']['verdict']}**", "",
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
           "", f"The link-word idea as a model input: **{idea}** ({IDEA[idea]}).",
           "", "This verdict closes only this form (A6e's eight columns beside zrckpr's on the MLP; A6e's link-weighted "
               "inputs in each of zspx's three steps on the GNN); the idea closes only as section 5 fixes it; "
               "docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za6ex verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES) + f"; the idea {idea}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def same(a, b):
    """Two batch values bit for bit: tensors, arrays, dicts and sequences of them, or anything else by equality."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                and np.array_equal(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def chain_by_parts(m, feats, keep, nq, B, bz, on):
    """zspx's three steps rebuilt from G5's parts (zprop's, G3's and G4's own heads and inputs), A6e's six inputs at
    each step's own input score added to the first layer of the steps named in on through that step's columns: what
    zspxl must equal."""
    lk, ew = feats[ZL.LK_KEY], feats[EW_KEY]
    s = super(ZP.ZProp, m).forward(feats, keep, nq, B, bz)
    plan = [("ew_w1", lambda s_: G4.prop_inputs_grad(s_, lk, nq, B), m.prop_head, ZP.SP)]
    if G3.RES_KEY in feats:
        plan.append(("ew_rs_w1", lambda s_: G3.res_inputs(s_, lk, feats[G3.RES_KEY], nq, B), m.rs_head, G3.RS))
    plan.append(("ew_p2_w1", lambda s_: ZP.prop_inputs(s_, lk, nq, B), m.p2_head, G4.P2))
    for key, inputs, head, (w1, b1, w2, b2) in plan:
        x = inputs(s)
        if key in on:
            pre = (Fn.linear(x, getattr(m, w1), getattr(m, b1))
                   + Fn.linear(ZA6E.ew_inputs(s, ew, lk, nq, B), getattr(m, key)))
            s = s + Fn.linear(Fn.gelu(pre), getattr(m, w2), getattr(m, b2)).squeeze(-1)
        else:
            s = s + head(x)
    return s


def fake_recall(path, arm, calls, drop=(), **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER if (sp, ds) not in drop]
    rec = {"stage": "A6ex", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckpr's model plus A6e's head on zrckpr's carve with A6e's columns beside (no edge); zspx's model
    #    with A6e's weighted inputs in its three steps on zspx's carve with A6e's weights beside; the bases and A6e's
    #    arms unchanged
    assert S.ARMS["zrckprn"] == (ZRCKPRN, A6eR1KTChainCarveZRC) and S.ARMS["zspxl"] == (ZSPXL, EWResKTLinkCarveBase)
    assert S.ARMS["zrckpr"] == (R1.ZRCKPR, R1.R1KTChainCarveZRC) and S.ARMS["zspx"] == (G5.ZSPX, G3.ResKTLinkCarveBase)
    assert S.ARMS["zrckpn"] == (ZA6E.ZRCKPN, ZA6E.A6EKTChainCarveZRC)
    assert S.ARMS["zspkl"] == (ZA6E.ZSPKL, ZA6E.EWKTLinkCarveBase)
    assert ZRCKPRN.__mro__[1:3] == (R1.ZRCKPR, Z3P.ZRCKP)
    assert ZSPXL.__mro__[1:5] == (G5.ZSPX, G3.ZSPR, Z3.ZSPK, ZP.ZProp)
    assert A6eR1KTChainCarveZRC.__mro__[1] is R1.R1KTChainCarveZRC
    assert EWResKTLinkCarveBase.__mro__[1] is G3.ResKTLinkCarveBase
    assert not hasattr(A6eR1KTChainCarveZRC, "LK_ROOT") and hasattr(EWResKTLinkCarveBase, "LK_ROOT")
    assert vars(A6eR1KTChainCarveZRC)["A6E_ROOT"] == ZA6E.OUT == vars(EWResKTLinkCarveBase)["A6E_ROOT"]
    assert ZA6E.N_HEAD == 161 and N_EWX == 576 and ZA6E.NE == 6 and ZA6E.NC == 8 and set(EWX) == set(STEPS)
    tmp = Path(tempfile.mkdtemp(prefix="za6ex_"))
    try:
        rng = np.random.default_rng(66)
        nlp = ZA6.FakeNLP()
        nodes = ZA6.toy_corpus(rng)
        N = len(nodes)
        # 2. one toy look for every array the arms read: rmatch's typed toy look; G3's pools, question embeddings and
        #    projection over A6's toy corpus; an rrf column (A6's); then rmatch's chains, zlink's edges, G3's residual
        #    parts, R1's columns, and A6e's columns and weights on the passage and on the KB reading, each by its own
        #    builder
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, _pools, _qembs = G3.toy_embed(tmp, rng, NN=N)
        look = tmp / "look" / "metaqa" / "toy"
        for f in sorted((look / "chunks").glob("c*.npz")):
            with np.load(f) as zf:
                zc = {k_: zf[k_] for k_ in zf.files}
            tot = int(zc["q_pool_size"].sum())
            assert "x" not in zc and int(zc["pool"].max()) < N
            rrf = np.where(rng.random(tot) < 0.75, np.round(rng.random(tot), 1) * 0.03 + 0.001, 0.0)
            np.savez(f, x=rrf.astype(np.float32).reshape(-1, 1), **zc)
        ids = json.loads((look / "ids.json").read_text(encoding="utf-8"))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        for pt in (0, 1):
            G3.build("metaqa", "toy", pt, out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look",
                     lk_root=tmp / "lk", nodes=G3.ToyNodes(T), freeze="toy")
        R1.build("metaqa", "toy", out_root=tmp / "r1", cache_root=tmp / "cache", lk_root=tmp / "lk",
                 res_root=tmp / "res")
        ment_t, _, _ = ZA6.toy_mentions(nodes, nlp)
        qwords = ["alpha bravo", "Who led delta echo golf", "Where is hotel Kay", "kilo lima mike Lux", "oscar papa",
                  "romeo sierra tango victor Moe"]
        qtext = {x: qwords[i % len(qwords)] for i, x in enumerate(ids)}
        kbn = [LW.stems_of([w]) for w in ("alpha", "golf", "kilo", "romeo", "tango")]
        for kb, root in ((None, tmp / "a6e"), (kbn, tmp / "a6ekb")):
            ZA6E.build("metaqa", "toy", ZA6.ToySrc(nodes, ment_t, qtext, kb), out_root=root, cache_root=tmp / "cache",
                       look_root=tmp / "look", lk_root=tmp / "lk")
        # 3. the batches: zrckprn's is zrckpr's with A6e's columns (A6e's own carve's) beside, and holds no edge, no
        #    weight and no residual message; zspxl's is zspx's (edges, c and s) with A6e's weights (A6e's own carve's)
        #    beside its edges, with each edge's word family on either reading
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        cls_res = G3.res_carve(cls_lk, tmp / "res")
        cls_r1 = R1.r1_carve(cls_ch, tmp / "r1")
        at = ("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_m = ZA6E.cols_carve(cls_r1, tmp / "a6e")(*at)
        c_g = ZA6E.ew_carve(cls_res, tmp / "a6e")(*at)
        c_k = ZA6E.ew_carve(cls_res, tmp / "a6ekb")(*at)
        c_r1, c_res = cls_r1(*at), cls_res(*at)
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:7], np.arange(c_m.rows)[:5]].astype(np.int64)
        B = qs.size
        fm, nq, bz, gold = c_m.batch(qs, blocks)
        fr1, nq1, bz1, gold1 = c_r1.batch(qs, blocks)
        fam_, *_ = ZA6E.cols_carve(cls_ch, tmp / "a6e")(*at).batch(qs, blocks)
        assert set(fm) == set(fr1) | {A6E_KEY} and not {ZL.LK_KEY, EW_KEY, G3.RES_KEY} & set(fm)
        assert all(same(fm[k_], fr1[k_]) for k_ in fr1) and same((nq, bz, gold), (nq1, bz1, gold1))
        assert torch.equal(fm[A6E_KEY], fam_[A6E_KEY]) and float(fm[A6E_KEY].abs().max()) > 0.05
        assert float(fm[R1.R1_KEY].abs().max()) > 0.05
        fg, nq_g, bz_g, gold_g = c_g.batch(qs, blocks)
        frs, nq_r, bz_r, gold_r = c_res.batch(qs, blocks)
        fag, *_ = ZA6E.ew_carve(cls_lk, tmp / "a6e")(*at).batch(qs, blocks)
        fk, *_ = c_k.batch(qs, blocks)
        assert set(fg) == set(frs) | {EW_KEY} and all(same(fg[k_], frs[k_]) for k_ in frs)
        assert same((nq_g, bz_g, gold_g), (nq_r, bz_r, gold_r)) and same(fg[EW_KEY], fag[EW_KEY])
        assert fg[EW_KEY]["w"].numel() == fg[ZL.LK_KEY]["u"].numel() and float(fg[EW_KEY]["w"].max()) > 0.1
        wf_g, wf_k = set(fg[EW_KEY]["wf"].tolist()), set(fk[EW_KEY]["wf"].tolist())
        assert {0, 2} <= wf_g <= {0, 2, 3} and {1, 2} <= wf_k <= {1, 2, 3}, (wf_g, wf_k)
        # 4. the models. Each arm from the same seed is its base's state plus its added parameters, no draw moved; with
        #    the base's own heads acting (R1's head on the MLP; zspx's three steps on the GNN) each arm starts as its
        #    base bit for bit, in eval and in training: scores and every shared gradient
        keep_ = torch.ones(B, len(blocks))
        gw = torch.Generator().manual_seed(8)
        w_r1 = torch.randn(1, R1.R1_HID, generator=gw)
        w_sp, w_rs, w_p2 = (torch.randn(1, h, generator=gw) for h in (ZP.HID, G3.RS_HID, G4.P2_HID))
        for arm, cls, base, cv, added, n_added in (("zrckprn", ZRCKPRN, R1.ZRCKPR, c_m, ZA6E.HEAD, ZA6E.N_HEAD),
                                                   ("zspxl", ZSPXL, G5.ZSPX, c_g, EWX, N_EWX)):
            f_, nq_, bz_, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(added) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k_: s1[k_] for k_ in s0})
            assert sum(s1[k_].numel() for k_ in added) == n_added
            with torch.no_grad():
                for m in (m0, m1):
                    if arm == "zrckprn":
                        m.r1_w2.copy_(w_r1)
                    else:
                        m.sp_w2.copy_(w_sp)
                        m.rs_w2.copy_(w_rs)
                        m.p2_w2.copy_(w_p2)
            for train_mode in (False, True):
                m0.train(train_mode)
                m1.train(train_mode)
                for m in (m0, m1):
                    m.zero_grad(set_to_none=True)
                torch.manual_seed(5)
                y0 = m0(f_, keep_, nq_, B, bz_)
                torch.manual_seed(5)
                y1 = m1(f_, keep_, nq_, B, bz_)
                assert torch.equal(y0, y1), (arm, train_mode)
                g_ = torch.randn(y0.shape, generator=torch.Generator().manual_seed(3))
                (y0 * g_).sum().backward()
                (y1 * g_).sum().backward()
                p0_, p1_ = dict(m0.named_parameters()), dict(m1.named_parameters())
                for k_, p in p0_.items():
                    assert (p.grad is None) == (p1_[k_].grad is None), k_
                    assert p.grad is None or torch.equal(p.grad, p1_[k_].grad), (arm, train_mode, k_)
                learn = EWX if arm == "zspxl" else ("a6e_w2",)
                assert all(float(p1_[k_].grad.abs().max()) > 0 for k_ in learn), "the added columns learn at once"
            # with the added parameters moved. zrckprn: the head over the row's own columns moves the score by exactly
            # its term, and one row's columns move that row's score only. zspxl: the six inputs are A6e's (against the
            # brute force) and carry no gradient to the scorer; each step's columns move that step only, zspx's chain
            # rebuilt from G5's parts; all three together, and without G3's c and s. Without its side key each arm is
            # its base
            m0.eval()
            m1.eval()
            if arm == "zspxl":
                lk_, ew_ = f_[ZL.LK_KEY], f_[EW_KEY]
                sg = super(ZP.ZProp, m1).forward(f_, keep_, nq_, B, bz_)
                assert sg.requires_grad and not ZA6E.ew_inputs(sg, ew_, lk_, nq_, B).requires_grad
            with torch.no_grad():
                y0 = m0(f_, keep_, nq_, B, bz_)
                if arm == "zrckprn":
                    m1.a6e_w2.copy_(torch.randn(1, ZA6E.A6E_HID, generator=torch.Generator().manual_seed(9)))
                    y1 = m1(f_, keep_, nq_, B, bz_)
                    head = m1.a6e_head(f_[A6E_KEY])
                    assert float(head.abs().max()) > 1e-3 and torch.allclose(y1 - y0, head, atol=1e-5)
                    x2 = f_[A6E_KEY].clone()
                    x2[3] += 1.0
                    d = m1({**f_, A6E_KEY: x2}, keep_, nq_, B, bz_) - y1
                    assert float(d[3].abs()) > 1e-4 and int(torch.count_nonzero(d)) == 1
                    bare = {k_: x for k_, x in f_.items() if k_ != A6E_KEY}
                else:
                    s0_ = super(ZP.ZProp, m1).forward(f_, keep_, nq_, B, bz_)
                    xe = ZA6E.ew_inputs(s0_, ew_, lk_, nq_, B)
                    xe_ref = ZA6E.ew_inputs_by_hand(s0_, ew_["w"], ew_["wf"], lk_["u"], lk_["v"], nq_, B)
                    assert np.abs(xe.double().numpy() - xe_ref).max() < 1e-5
                    assert float(np.abs(xe_ref[:, :3]).max()) > 0.01 and float(np.abs(xe_ref[:, 3:]).max()) > 0.05
                    gq = torch.Generator().manual_seed(9)
                    W = {k_: torch.randn(getattr(m1, k_).shape, generator=gq) for k_ in EWX}
                    assert torch.equal(chain_by_parts(m1, f_, keep_, nq_, B, bz_, set()), y0)
                    for one in EWX:
                        for k_ in EWX:
                            getattr(m1, k_).copy_(W[k_] if k_ == one else torch.zeros_like(W[k_]))
                        y1 = m1(f_, keep_, nq_, B, bz_)
                        assert torch.allclose(y1, chain_by_parts(m1, f_, keep_, nq_, B, bz_, {one}), atol=1e-5), one
                        assert float((y1 - y0).abs().max()) > 1e-4, one
                    for k_ in EWX:
                        getattr(m1, k_).copy_(W[k_])
                    y1 = m1(f_, keep_, nq_, B, bz_)
                    assert torch.allclose(y1, chain_by_parts(m1, f_, keep_, nq_, B, bz_, set(EWX)), atol=1e-5)
                    no_res = {k_: x for k_, x in f_.items() if k_ != G3.RES_KEY}
                    y_nr = m1(no_res, keep_, nq_, B, bz_)
                    assert torch.allclose(y_nr, chain_by_parts(m1, no_res, keep_, nq_, B, bz_, set(EWX)), atol=1e-5)
                    assert float((y_nr - m0(no_res, keep_, nq_, B, bz_)).abs().max()) > 1e-4
                    no_lk = {k_: x for k_, x in f_.items() if k_ != ZL.LK_KEY}
                    assert torch.equal(m1(no_lk, keep_, nq_, B, bz_), m0(no_lk, keep_, nq_, B, bz_))
                    bare = {k_: x for k_, x in f_.items() if k_ != EW_KEY}
                assert torch.equal(m1(bare, keep_, nq_, B, bz_), m0(bare, keep_, nq_, B, bz_))
        # 5. toy fits through lean_gpu's loop: a repeat is bit-identical and the added parameters learn; on the base
        #    arm's carve (no side key) each arm's fit is its base's bit for bit on every shared key, its additions at
        #    the start
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv, bare_c, keys in (("zrckprn", c_m, c_r1, ("a6e_w2",)), ("zspxl", c_g, c_res, EWX)):
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                bare = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, no side key")
            with S.patched(BASE_OF[arm]):
                base = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            assert all(not LG.same_state(a_, b_) for a_, b_ in zip(fit["states"] + [fit["swa"]],
                                                                   rep["states"] + [rep["swa"]]))
            assert all(float(fit["swa"][k_].abs().max()) > 0 for k_ in keys), arm
            for a_, b_ in zip(bare["states"] + [bare["swa"]], base["states"] + [base["swa"]]):
                assert not LG.same_state({k_: a_[k_] for k_ in b_}, b_), arm
                assert all(float(a_[k_].abs().max()) == 0.0 for k_ in keys)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the refusals
    good = ["train", "--arm", "zrckprn", "--name", FIT["zrckprn"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    for wrong in ("zrckpr", "zrckpn", "zspx"):
        SR.must_stop(train, ["train", "--arm", wrong, "--name", FIT["zrckprn"]["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspxl", "--name", FIT["zrckprn"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspxl", "--name", FIT["zspxl"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    assert split_of_fit("scr-a6ex-zspxl-hp") == ("zspxl", "L-hotpotqa")
    assert split_of_fit("scr-a6e-zspkl") == (None, None) and split_of_fit("scr-r1-zrckpr") == (None, None)
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckpr", "r1": {}}), ("scr-a6ex-zspxl", {"arm": "zspkl", "a6e": {}}),
                        ("scr-a6ex-zrckprn", {"arm": "zspxl", "a6ex": {"split": "L-musique"}}),
                        ("scr-a6ex-zspxl-hp", {"arm": "zspxl", "a6ex": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 7. relz's pair and re-call for each arm against its base arm's card fits, stamped A6ex's; relz restored after
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
            assert pj["stage"] == "A6ex" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A6ex ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A6e's link words) |" in t
            assert json.loads((T_ / arm / "rz.json").read_text(encoding="utf-8"))["stage"] == "A6ex"
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 8. the verdict, per arm, with the next form of section 5 and the idea's state from both arms
        gm, mi, wl = ("L-musique", "musique"), CLAIM1_READ, ("L-hotpotqa", "webqsp")
        a_, b_ = T_ / "a.json", T_ / "b.json"
        rec = verdict({"zrckprn": fake_recall(a_, "zrckprn", {mi: "GAIN"}), "zspxl": fake_recall(b_, "zspxl", {})},
                      T_ / "v")
        assert rec["arms"]["zrckprn"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckprn"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspxl"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspxl"]["next_form"] == NEXT["NO_GAIN_OTHER_ADOPT"]["zspxl"]
        assert rec["link_word_idea"] == "OPEN"
        assert rec["musique_in_domain_beside"] == {"zrckprn": {"call": "GAIN", "delta": 0.0},
                                                   "zspxl": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspxl"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckprn": fake_recall(a_, "zrckprn", {}),
                       "zspxl": fake_recall(b_, "zspxl", {gm: "GAIN", wl: "LOSS"})}, T_ / "v")
        assert rec["arms"]["zspxl"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspxl"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["arms"]["zrckprn"]["next_form"] == NEXT["GAIN_AND_LOSS"] and rec["link_word_idea"] == "OPEN"
        rec = verdict({"zrckprn": fake_recall(a_, "zrckprn", {}), "zspxl": fake_recall(b_, "zspxl", {wl: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zspxl"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckprn"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zrckprn"]["next_form"] == NEXT["NO_GAIN_OTHER_ADOPT"]["zrckprn"]
        assert len({NEXT["NO_GAIN_OTHER_ADOPT"][a] for a in ARM_NAMES}) == 2 and rec["link_word_idea"] == "OPEN"
        rec = verdict({"zrckprn": fake_recall(a_, "zrckprn", {wl: "LOSS"}), "zspxl": fake_recall(b_, "zspxl", {})},
                      T_ / "v")
        assert all(rec["arms"][a]["verdict"] == "NOT_ADOPTED" and rec["arms"][a]["next_form"] == NEXT["NO_GAIN"]
                   for a in ARM_NAMES)
        assert rec["link_word_idea"] == "CLOSED_AS_A_MODEL_INPUT"
        assert "CLOSED_AS_A_MODEL_INPUT" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckprn": T_ / "none.json", "zspxl": fake_recall(b_, "zspxl", {})}, T_ / "v")
        assert rec["arms"]["zrckprn"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckprn"]["next_form"] == NEXT["INCOMPLETE"]
        assert rec["arms"]["zspxl"]["next_form"] == NEXT["NO_GAIN_WAIT"] and rec["link_word_idea"] == "WAITING"
        rec = verdict({"zrckprn": fake_recall(a_, "zrckprn", {mi: "GAIN"}, stage="A6e"),
                       "zspxl": fake_recall(b_, "zrckprn", {gm: "GAIN"})}, T_ / "v")
        assert all(rec["arms"][a]["verdict"] == "INCOMPLETE" for a in ARM_NAMES) and rec["link_word_idea"] == "WAITING"
        rec = verdict({"zrckprn": fake_recall(a_, "zrckprn", {mi: "GAIN"}, drop={gm}),
                       "zspxl": fake_recall(b_, "zspxl", {gm: "GAIN"})}, T_ / "v")
        assert rec["arms"]["zrckprn"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zspxl"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["link_word_idea"] == "OPEN"
    log(f"za6ex selftest: the arms are the adopted ones plus A6e's inputs (zrckpr's carve with the columns and no edge, "
        f"zspx's with the weights beside its edges); every batch value is the base carve's, the columns and weights "
        f"A6e's own carves'; each arm starts as its base bit for bit in eval and training (scores, every shared "
        f"gradient) with the base's heads acting; A6e's head moves zrckprn by exactly its term, row by row; A6e's six "
        f"inputs (against the brute force, no gradient to the scorer) move each of zspx's three steps through its own "
        f"columns and nothing else, all three as zspx's chain rebuilt from G5's parts; without its side key each arm is "
        f"its base; toy fits repeat, learn, and without the side key are the base's; the refusals; relz's pair and "
        f"re-call per arm, stamped A6ex's; the verdict's outcomes, next forms and the idea's state "
        f"({time.time() - t0:.1f}s): ok")
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
            log(f"za6ex {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za6ex: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
