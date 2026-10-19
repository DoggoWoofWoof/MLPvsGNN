"""A8ex (docs/A8EX_QDEPTH_PRIOR_ADOPTED.md): round 11's depth prior on the adopted arms, the next form A8e's verdict
named (both arms NOT_ADOPTED with no GAIN; outputs/a8e/screen-verdict.json). A8e (za8e.py) learned the prior inside the
lean pair (zrckp and zspk). R1's zrckpr and G5's zspx have passed their screens since; A8ex puts the same prior on them,
each against its own base:

    s(v) = the adopted arm's s(v) + l_q[depth(v)],    l_q = q U + P_q W + b    (five values per question)

The prior is A8e's own (za8e.qd_check, qd_init and qd_term, through qdepth.QDepth.depth_logits): the same inputs (the
question's embedding, its pool's depth profile, the row's depth class from WALK's raw columns), the same five classes,
the same shapes (U 1536 x 5, W 6 x 5, b 5: 7,715 parameters) and the same zero start, so each arm starts as its base bit
for bit. The model's listwise loss trains it with its own Adam, learning rate and weight decay.

    zrckprqe  zrckpr's model (outputs/mp_unified/zr1res.py) plus the prior, added to its final score, after the KT head
              and R1's head; on zrckpr's carve
    zspxqe    zspx's model (outputs/mp_unified/zg5mix.py) with the prior added to the KT-mixed score, before zspx's
              step (gradient kept) first reads the score; G3's residual message and G4's second step read it after;
              on zspx's carve

No column and no builder: the prior reads blocks of P1c2's cache (q_emb and WALK) that both adopted arms already read.

    python outputs/mp_unified/za8ex.py train --split L-musique --name scr-a8ex-zrckprqe --arm zrckprqe --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8ex.py read --name scr-a8ex-zrckprqe --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8ex.py compare --new outputs/screen/fits/scr-a8ex-zrckprqe \\
        --base outputs/screen/fits/scr-r1-zrckpr,outputs/step1/fits/L-musique --out outputs/screen/scr-a8ex-zrckprqe
    python outputs/mp_unified/za8ex.py pair --arm zrckprqe --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za8ex.py recall --arm zrckprqe --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za8ex.py verdict
    python outputs/mp_unified/za8ex.py --selftest
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
import za8e as A8E  # noqa: E402  (A8e: the prior, QDMix, its toy walks and hand computation; its arms on the lean pair)
import qdepth as QD  # noqa: E402  (round 11's prior: depth_of, profile, QDepth.depth_logits, head_norms)

import numpy as np  # noqa: E402
import torch  # noqa: E402

G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
Z3P = ZR1.Z3P
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A8EX_QDEPTH_PRIOR_ADOPTED.md"
OUT = ROOT / "outputs" / "a8ex"
SCREEN = ROOT / "outputs" / "screen"
QDP, N_QD = A8E.QDP, A8E.N_QD
SPLITS = Z3.SPLITS
BASE_OF = {"zrckprqe": "zrckpr", "zspxqe": "zspx"}
LEAN_OF = {"zrckprqe": "zrckpqe", "zspxqe": "zspkqe"}             # A8e's arm of the same prior on the lean pair
ARM_NAMES = tuple(BASE_OF)
SETTINGS = {**A8E.SETTINGS,
            "where": {"zrckprqe": "added to zrckpr's final score, after the KT head and R1's head",
                      "zspxqe": "added to the KT-mixed score, before zspx's step (gradient kept), G3's residual message "
                                "and G4's second step read it"},
            "unchanged": "every column, block and head of the adopted arm; R1's columns (zrckprqe) and G3's c and s "
                         "(zspxqe)",
            "parent": "A8e (docs/A8E_QDEPTH_PRIOR.md, outputs/mp_unified/za8e.py)"}
FIT = {a: {"L-musique": f"scr-a8ex-{a}", "L-hotpotqa": f"scr-a8ex-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckprqe": {sp: ("screen", "fits", ZR1.FIT["zrckpr"][sp]) for sp in SPLITS},     # R1's card fits
             "zspxqe": {sp: ("screen", "fits", G5.FIT[sp]) for sp in SPLITS}}                   # G5's card fits
BASE_KEY = {"zrckprqe": "r1", "zspxqe": "g5"}                    # the stage key each base fit's screen.json carries
STEP1_FITS = {sp: ("step1", "fits", sp) for sp in SPLITS}         # step 1's fit, beside in each compare
BASE_SCREENS = {"zrckprqe": {sp: f"outputs/p1gc/base-pools-zrckpr-{sp}.json" for sp in SPLITS},   # P1g-c's
                "zspxqe": {sp: f"outputs/a9/base-zspx-{sp}.json" for sp in SPLITS}}                # A9's
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckprqe": "zrckpr plus round 11's depth prior (A8e's), added to its final score",
        "zspxqe": "zspx plus round 11's depth prior (A8e's), added before its three increments read the score"}
PAIR = {a: SCREEN / f"scr-a8ex-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a8ex-{a}-pair-recall" for a in ARM_NAMES}
A8E_RECALL = {a: SCREEN / f"scr-a8e-{LEAN_OF[a]}-pair-recall.json" for a in ARM_NAMES}   # beside, deciding nothing
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A8ex's re-call of the arm, "
        "or not twelve reads (docs/A8EX_QDEPTH_PRIOR_ADOPTED.md, section 4)")


# ── the arms ─────────────────────────────────────────────────────────────────


class ZRCKPRQE(ZR1.ZRCKPR):
    """zrckprqe: zrckpr's model plus round 11's depth prior (zero at the start), added to its final score, after the KT
    head and R1's head (MRO: ZRCKPRQE, ZRCKPR, ZRCKP, KTMixP, ZRM, ...)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        A8E.qd_check(blocks, widths)
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        A8E.qd_init(self)

    qd_term = A8E.qd_term

    def forward(self, feats, keep, nq, B, base_z):
        return super().forward(feats, keep, nq, B, base_z) + self.qd_term(feats, keep, nq, B)


class ZSPXQE(G5.ZSPX, A8E.QDMix):
    """zspxqe: zspx's model with round 11's depth prior in the score its three increments read (MRO: ZSPXQE, ZSPX, ZSPR,
    ZSPK, ZProp, QDMix, KTMix, ZRM, ...). ZSPX.forward's super(ZProp) call reaches A8e's QDMix, which adds the prior to
    the KT-mixed score before zspx's step reads it. No body."""


ARMS = {"zrckprqe": (ZRCKPRQE, ZR1.R1KTChainCarveZRC), "zspxqe": (ZSPXQE, G3.ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


SIDE_KEY = {"zrckprqe": "r1_records", "zspxqe": "res_records"}


def side_records(arm, carves):
    """The records of the arm's own side input on carves: R1's columns for zrckprqe, G3's c and s for zspxqe; stops when
    one is missing or not its stage's."""
    return ZR1.r1_records(carves) if arm == "zrckprqe" else G3.res_records(carves)


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
        mp = out_root / name / "models.pt"
        rec.update({"za8ex_sha256": LC.sha_src(__file__), "za8e_sha256": LC.sha_src(A8E.__file__),
                    "qdepth_sha256": LC.sha_src(QD.__file__), "zg5mix_sha256": LC.sha_src(G5.__file__),
                    "zg3res_sha256": LC.sha_src(G3.__file__), "zg4prop_sha256": LC.sha_src(G4.__file__),
                    "zr1res_sha256": LC.sha_src(ZR1.__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "a8ex": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                             "device": Z2.DEVICE, "prior": list(QDP), "settings": SETTINGS,
                             "prior_norms": QD.head_norms(mp) if mp.exists() else None,
                             "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                             "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                             SIDE_KEY[arm]: side_records(arm, Z3.train_carves(split)),
                             "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                             "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za8ex: train takes --arm {' or '.join(ARM_NAMES)}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za8ex: A8ex's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za8ex: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a8ex" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za8ex: {name} is not an A8ex fit (arm {rec.get('arm')})")
    if rec["a8ex"].get("split") != split:
        raise SystemExit(f"za8ex: {name} is {split}'s fit, its record says {rec['a8ex'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    side_records(arm, Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    """lean_screen2's compare, after A8ex's refusals: the new fit is an A8ex fit, decided against the adopted arm's card
    fit of the same split with step 1's fit of that split beside (section 4), and its screen keeps the fit's name. A base
    fit's or the new fit's record, where it is on disk, must be that arm's."""
    cp = argparse.ArgumentParser(add_help=False)
    for f in ("--new", "--base", "--out"):
        cp.add_argument(f)
    k, _ = cp.parse_known_args(rest)
    if not (k.new and k.base and k.out):
        raise SystemExit("za8ex: compare takes --new, --base and --out")
    new = Path(k.new)
    arm, split = split_of_fit(new.name)
    if arm is None or new.parts[-3:-1] != ("screen", "fits"):
        raise SystemExit(f"za8ex: compare takes an A8ex fit under outputs/screen/fits, not {k.new}")
    bases = [Path(b) for b in k.base.split(",") if b]
    want = (BASE_FITS[arm][split], STEP1_FITS[split])
    if tuple(b.parts[-3:] for b in bases) != want:
        raise SystemExit(f"za8ex: {new.name} is decided against {'/'.join(want[0])}, with {'/'.join(want[1])} beside, "
                         f"not {k.base}")
    if Path(k.out).parts[-2:] != ("screen", new.name):
        raise SystemExit(f"za8ex: {new.name}'s screen is outputs/screen/{new.name}, not {k.out}")
    for p, a, key in ((new, arm, "a8ex"), (bases[0], BASE_OF[arm], BASE_KEY[arm])):
        sj = (p if p.is_absolute() else ROOT / p) / "screen.json"
        if not sj.exists():
            continue
        rec = json.loads(sj.read_text(encoding="utf-8"))
        if rec.get("arm") != a or key not in rec:
            raise SystemExit(f"za8ex: {p.name}'s record is arm {rec.get('arm')}, not {a}'s fit with its '{key}' stamp")
        if key == "a8ex" and rec[key].get("split") != split:
            raise SystemExit(f"za8ex: {p.name} is {split}'s fit, its record says {rec[key].get('split')}")
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za8ex: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za8ex: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A8ex's depth prior) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A8ex ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A8ex", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za8ex_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": "A8exk, the back-off form on the adopted arms: the prior split by the graph's kind (one U, W and "
                         "b for the KB graphs, metaqa and webqsp, and one for the passage graphs, each at zero), so one "
                         "kind's depth weights cannot move the other kind's scores; on both models, its own declared "
                         "file",
        "NO_GAIN_OTHER_GAINED": "A8exk on this model as well (section 5): the other arm's GAIN keeps the depth prior open "
                                "on both models",
        "NO_GAIN_PENDING": "decided with the other arm's re-call (section 5): the depth prior closes as a model term only "
                           "if neither arm gains",
        "NO_GAIN": "none for the depth prior as a model term, which closes: it has reached both models three ways without "
                   "a GAIN, as a fixed input (A8), learned inside the lean pair (A8e) and learned inside the adopted "
                   "arms (A8ex); no non-model form stays open, since the depth classes are already an input of both "
                   "models, the hop-depth diagnosis's pools already held a gold for 0.972 of metaqa's 3-hop questions "
                   "and 0.921 of webqsp's with 0.001 and 0.010 of their golds unreached, and no parameter-free rule can "
                   "know a question's hop count; the hop-depth oracle's gap goes to the gap ledger as an upper bound "
                   "(section 5)",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, other_gains, pending):
    """An arm's next form (section 5): an arm without a GAIN goes to the back-off when the other arm gains, waits when
    the other arm's re-call is incomplete, and closes the prior as a model term only when neither arm gains."""
    if v != "NOT_ADOPTED":
        return NEXT[v]
    if gains:
        return NEXT["GAIN_AND_LOSS"]
    if other_gains:
        return NEXT["NO_GAIN_OTHER_GAINED"]
    return NEXT["NO_GAIN_PENDING"] if pending else NEXT["NO_GAIN"]


def lean_beside(arm, path=None):
    """A8e's re-call of the same prior on the lean pair (the arm's LEAN_OF twelve reads), beside, deciding nothing; {}
    when its file is missing or is not A8e's re-call of that arm."""
    p = Path(path or A8E_RECALL[arm])
    if not p.exists():
        return {}
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("stage") != "A8e" or rec.get("declared_in") != A8E.DECLARED or rec.get("arm") != LEAN_OF[arm]:
        return {}
    return {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rec.get("rows", [])}


def verdict(recalls=None, out=None, lean=None):
    """A8ex's screen verdict, per arm, from its re-call (section 4); webqsp and metaqa beside, musique's in-domain read of
    both arms side by side, each read beside A8e's on the lean pair, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    lean = lean or {}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A8ex" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A8ex's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "a8e_beside": lean_beside(a, lean.get(a)),
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    pending = any(per[a]["verdict"] == "INCOMPLETE" for a in ARM_NAMES)
    for a in ARM_NAMES:
        other_gains = any(per[b]["gains"] for b in ARM_NAMES if b != a and per[b]["verdict"] != "INCOMPLETE")
        per[a]["next_form"] = next_form(per[a]["verdict"], per[a]["gains"], other_gains, pending)
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A8ex", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za8ex_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# A8ex's screen verdict: " + ", ".join(f"{a} **{per[a]['verdict']}**" for a in ARM_NAMES), "",
          f"Rule: {RULE}.", ""]
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"
    cell = lambda d: f"{d['call']} ({num(d['delta'])})" if d else "-"
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {cell(b)}" for k, b in p["beside"].items()) or "none")
                  + f"; A8e on {LEAN_OF[a]}: " + ("found" if p["a8e_beside"] else "not found")
                  + f". Next: {p['next_form']}.")
    md += ["", f"Beside, deciding nothing: musique in-domain ({ck}): "
           + ", ".join(f"{a} {side[a]['call']} ({num(side[a]['delta'])})" for a in ARM_NAMES) + ".", "",
           "Each read beside A8e's on the lean pair, deciding nothing (call and delta R@5 after each re-call):", "",
           "| read | " + " | ".join(f"{a} | {LEAN_OF[a]} (A8e)" for a in ARM_NAMES) + " |",
           "| --- |" + " --- | --- |" * len(ARM_NAMES)]
    for sp in SPLITS:
        for ds in LG.EVAL_ORDER:
            k_ = f"{sp} {ds}"
            md.append(f"| {k_} | " + " | ".join(
                cell({"call": per[a]["calls"][k_], "delta": per[a]["delta_r5"].get(k_)} if k_ in per[a]["calls"]
                     else None) + " | " + cell(per[a]["a8e_beside"].get(k_)) for a in ARM_NAMES) + " |")
    md += ["", "This verdict closes only this form (round 11's depth prior learned inside the adopted arms, zrckpr and "
               "zspx); the prior closes as a model term only when neither arm gains (section 5); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za8ex verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A8ex", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def set_heads(*ms):
    """Every term and head away from zero, the same in each model (zero at the start, they would hide what the prior
    moves): za2xmx's settings."""
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


@contextlib.contextmanager
def spied():
    """Every z-score lean_gpu.seg_zscore8D takes over a pool's rows (zret's reference z-scores call it too) and every
    read of the score by zspx's three increments, in call order: [(kind, its input, detached)]."""
    got = []
    saved = LG.seg_zscore8D, G4.prop_inputs_grad, G3.res_inputs, ZP.prop_inputs

    def wrap(kind, fn):
        def f(x, *a, **k):
            got.append((kind, x.detach().clone()))
            return fn(x, *a, **k)
        return f

    LG.seg_zscore8D = wrap("z", saved[0])
    G4.prop_inputs_grad, G3.res_inputs, ZP.prop_inputs = wrap("step", saved[1]), wrap("res", saved[2]), wrap("p2",
                                                                                                             saved[3])
    try:
        yield got
    finally:
        LG.seg_zscore8D, G4.prop_inputs_grad, G3.res_inputs, ZP.prop_inputs = saved


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    spied_mods = [m_ for m_ in (sys.modules.get("lean_screen3"), Z3, ZP, G3, G4, QD) if m_ is not None]
    # 1. the arms: zrckpr's and zspx's models plus A8e's prior, on their bases' own carves; the base arms and A8e's arms
    #    unchanged; the prior is A8e's own function; the zspx arm class has no body
    assert S.ARMS["zrckprqe"] == (ZRCKPRQE, ZR1.R1KTChainCarveZRC) and S.ARMS["zspxqe"] == (ZSPXQE, G3.ResKTLinkCarveBase)
    assert S.ARMS["zrckpr"] == (ZR1.ZRCKPR, ZR1.R1KTChainCarveZRC) and S.ARMS["zspx"] == (G5.ZSPX, G3.ResKTLinkCarveBase)
    assert all(S.ARMS[LEAN_OF[a]] == A8E.ARMS[LEAN_OF[a]] for a in ARM_NAMES)
    assert ZRCKPRQE.__mro__[1:6] == (ZR1.ZRCKPR, Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch)
    assert ZSPXQE.__mro__[1:9] == (G5.ZSPX, G3.ZSPR, Z3.ZSPK, ZP.ZProp, A8E.QDMix, Z3.KTMix, ZM.ZRM, RM.ChainMatch)
    assert A8E.QDMix not in G5.ZSPX.__mro__ and all(k_.startswith("__") for k_ in vars(ZSPXQE))
    assert ZRCKPRQE.qd_term is A8E.qd_term and ZSPXQE.qd_term is A8E.qd_term
    assert A8E.Z3 is Z3 and A8E.Z3P is Z3P and A8E.QD is QD and all(m_.LG is LG for m_ in spied_mods)   # spies see all
    # 2. the bases are the adopted arms' card fits (R1's and G5's) and their base screens (P1g-c's and A9's); the prior is
    #    round 11's: qdepth's constants and 7,715 parameters
    assert BASE_FITS == {"zrckprqe": {"L-musique": ("screen", "fits", "scr-r1-zrckpr"),
                                      "L-hotpotqa": ("screen", "fits", "scr-r1-zrckpr-hp")},
                         "zspxqe": {"L-musique": ("screen", "fits", "scr-g5-zspx"),
                                    "L-hotpotqa": ("screen", "fits", "scr-g5-zspx-hp")}}
    assert BASE_SCREENS["zspxqe"] == {sp: f"outputs/a9/base-zspx-{sp}.json" for sp in SPLITS}
    assert BASE_SCREENS["zrckprqe"] == {sp: f"outputs/p1gc/base-pools-zrckpr-{sp}.json" for sp in SPLITS}
    assert (QD.DEPTHS, QD.WALK_W, QD.FIRST_HOP, QD.IS_SEED, QD.Q_DIM, QD.PROF_W) == (5, 16, (12, 13, 14), 15, 1536, 6)
    assert N_QD == 7715 and QDP == ("qd_u", "qd_p", "qd_b")
    rng = np.random.default_rng(61)
    tmp = Path(tempfile.mkdtemp(prefix="za8ex_"))
    neutral = "za8ex-selftest-frozen"
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
        cls_ch = RM.chain_carve(A8E.ToyWalk, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        kt_ch, kt_lk = Z3.kt_carve(cls_ch, tmp / "kt"), Z3.kt_carve(cls_lk, tmp / "kt")
        c_m = ZR1.r1_carve(kt_ch, tmp / "r1")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")      # zrckpr's, in toy
        c_g = G3.res_carve(kt_lk, tmp / "res")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")     # zspx's, in toy
        assert torch.equal(QD.depth_of(c_m.walk), c_m.depth) and QD.depth_of(c_m.walk)[:5].tolist() == [0, 1, 2, 3, 4]
        blocks = ["rank", "SEMB", "WALK"]
        widths = {b: c_m.widths[b] for b in blocks}
        assert all(c_g.widths[b] == widths[b] for b in blocks) and widths["WALK"] == QD.WALK_W
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, len(blocks))
        fm, nqm, _bz, _g = c_m.batch(qs, blocks)
        fg, nqg, _bzg, _gg = c_g.batch(qs, blocks)
        assert Z3.KT_KEY in fm and ZR1.R1_KEY in fm and G3.RES_KEY not in fm and "WALK" in fm
        assert G3.RES_KEY in fg and ZL.LK_KEY in fg and ZR1.R1_KEY not in fg and "WALK" in fg
        assert torch.equal(nqg, nqm) and torch.equal(fg["WALK"], fm["WALK"])     # the two carves share the pools
        for arm, cls, base, cv in (("zrckprqe", ZRCKPRQE, ZR1.ZRCKPR, c_m), ("zspxqe", ZSPXQE, G5.ZSPX, c_g)):
            f_, nq_, bz_, gold_ = cv.batch(qs, blocks)
            # 3. the start: from the same seed each arm is its base's state plus U, W and b at zero, the global stream
            #    where the base leaves it. As built and again with every head's last layer set (alike in both), its
            #    scores in training (the same dropout draws) and in eval are the base's bit for bit, and so is every
            #    gradient of a first listwise step but the prior's own, which are finite and nonzero on U, W and b
            #    (b's summing to zero over the classes: the loss is a softmax over each question's rows)
            nets, after = {}, {}
            for k, cl in (("base", base), ("arm", cls)):
                torch.manual_seed(0)
                nets[k] = cl(blocks, widths, 16, dropout=0.1, seed=0)
                after[k] = torch.rand(4)
            assert torch.equal(after["base"], after["arm"]), arm
            m0, m1 = nets["base"], nets["arm"]
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert set(s0) <= set(s1) and sorted(set(s1) - set(s0)) == sorted(QDP), arm
            assert not LG.same_state(s0, {k: s1[k] for k in s0}), arm
            assert sum(s1[k].numel() for k in QDP) == N_QD and all(float(s1[k].abs().max()) == 0.0 for k in QDP)
            for live in (False, True):
                if live:
                    set_heads(m0, m1)
                ys = {}
                for k, m in nets.items():
                    m.train()
                    m.zero_grad(set_to_none=True)
                    torch.manual_seed(5)
                    ys[k] = m(f_, keep_, nq_, B, bz_)
                    LG.listwiseD(ys[k], gold_, nq_, B).backward()
                assert torch.equal(ys["base"], ys["arm"]), (arm, live)
                p1 = dict(m1.named_parameters())
                for k, p in m0.named_parameters():
                    assert (p.grad is None) == (p1[k].grad is None), (arm, live, k)
                    assert p.grad is None or torch.equal(p.grad, p1[k].grad), (arm, live, k)
                gq = {k: p1[k].grad for k in QDP}
                assert all(v is not None and bool(torch.isfinite(v).all()) and float(v.abs().sum()) > 0
                           for v in gq.values()), (arm, live)
                gb = gq["qd_b"]
                assert abs(float(gb.sum())) <= 1e-5 * max(1.0, float(gb.abs().sum())), (arm, live, float(gb.sum()))
                for m in nets.values():
                    m.eval()
                with torch.no_grad():
                    assert torch.equal(m0(f_, keep_, nq_, B, bz_), m1(f_, keep_, nq_, B, bz_)), (arm, live)
            with torch.no_grad():
                # 4. U, W and b set: the term is l_q at each row's own class against a float64 hand computation from the
                #    raw walks (all five classes in the batch). zrckprqe is zrckpr plus the term, bit for bit, and adds
                #    no z-score and no propagation: the z-scores taken over each pool are the base's, bit for bit. In
                #    zspxqe every z-score before zspx's step is the base's (the KT head reads the base's score); the
                #    step's first read of the score is the KT-mixed score plus the term, then G3's residual and G4's
                #    second step read the score after it; the arm's score is rebuilt from them bit for bit and moves
                gen = torch.Generator().manual_seed(9)
                for k, sc in zip(QDP, (0.02, 0.5, 0.5)):
                    getattr(m1, k).copy_(torch.randn(getattr(m1, k).shape, generator=gen) * sc)
                term = m1.qd_term(f_, keep_, nq_, B)
                d_h, want = A8E.hand_prior(m1, f_, nq_, B)
                assert np.array_equal(d_h, QD.depth_of(f_["WALK"]).numpy()) and len(set(d_h.tolist())) == QD.DEPTHS
                assert np.allclose(term.double().numpy(), want, atol=1e-5, rtol=1e-5) and float(term.abs().max()) > 1e-2
                with spied() as l0:
                    y0 = m0(f_, keep_, nq_, B, bz_)
                with spied() as l1:
                    y1 = m1(f_, keep_, nq_, B, bz_)
                kinds = [k for k, _ in l1]
                assert kinds == [k for k, _ in l0], (arm, kinds)
                if arm == "zrckprqe":
                    assert torch.equal(y1, y0 + term)
                    assert set(kinds) <= {"z"} and all(torch.equal(a, b) for (_, a), (_, b) in zip(l0, l1)), kinds
                else:
                    assert [k for k in kinds if k != "z"] == ["step", "res", "p2"], kinds
                    i = kinds.index("step")
                    assert i > 0 and all(torch.equal(a, b) for (_, a), (_, b) in zip(l0[:i], l1[:i]))
                    lk = f_[ZL.LK_KEY]
                    kts = Z3.KTMix.forward(m1, f_, keep_, nq_, B, bz_)
                    assert torch.equal(kts, Z3.KTMix.forward(m0, f_, keep_, nq_, B, bz_)) and torch.equal(l0[i][1], kts)
                    s_0 = kts + term
                    assert torch.equal(l1[i][1], s_0)
                    s_1 = s_0 + m1.prop_head(G4.prop_inputs_grad(s_0, lk, nq_, B))
                    s_2 = s_1 + m1.rs_head(G3.res_inputs(s_1, lk, f_[G3.RES_KEY], nq_, B))
                    assert torch.equal(l1[kinds.index("res")][1], s_1) and torch.equal(l1[kinds.index("p2")][1], s_2)
                    assert torch.equal(y1, s_2 + m1.p2_head(ZP.prop_inputs(s_2, lk, nq_, B)))
                    assert float((y1 - y0 - term).abs().max()) > 1e-4                # the increments read the prior
                # 5. the prior reads the question's embedding and WALK only, never a row's score: from those two alone
                #    it is the same bit for bit; in zrckprqe so it is without R1's columns (whose head moves the score);
                #    so it is when every other weight moves every row's score; a question whose WALK or SEMB is dropped
                #    takes none, the others keep theirs
                only = {"WALK": f_["WALK"], "SEMB": (f_["SEMB"][0], None)}
                assert torch.equal(m1.qd_term(only, keep_, nq_, B), term)
                if arm == "zrckprqe":
                    no_r1 = {k: v for k, v in f_.items() if k != ZR1.R1_KEY}
                    assert torch.equal(m1.qd_term(no_r1, keep_, nq_, B), term)
                    assert not torch.equal(m1(no_r1, keep_, nq_, B, bz_), y1)
                for k, p in m1.named_parameters():
                    if k not in QDP:
                        p.add_(torch.randn(p.shape, generator=gen) * 0.5)
                assert torch.equal(m1.qd_term(f_, keep_, nq_, B), term)
                assert not torch.equal(m1(f_, keep_, nq_, B, bz_), y1)
                q1 = nq_ == 1
                for j in (m1.j_walk, m1.j_semb):
                    k2 = keep_.clone()
                    k2[1, j] = 0.0
                    t2 = m1.qd_term(f_, k2, nq_, B)
                    assert float(t2[q1].abs().max()) == 0.0 and torch.equal(t2[~q1], term[~q1])
        # 6. toy fits through lean_gpu's loop: a repeat is bit for bit and the prior learns; with the prior frozen at its
        #    zero start, each arm's fit is its base's bit for bit on every shared key
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        states = lambda f: f["states"] + [f["swa"]]
        for arm, cv in (("zrckprqe", c_m), ("zspxqe", c_g)):

            class Frozen(ARMS[arm][0]):
                def __init__(self, *a, **k):
                    super().__init__(*a, **k)
                    for p in (self.qd_u, self.qd_p, self.qd_b):
                        p.requires_grad_(False)

            S.ARMS[neutral] = (Frozen, ARMS[arm][1])
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
            with S.patched(neutral):
                frz = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, prior frozen at zero")
            with S.patched(BASE_OF[arm]):
                bfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            S.ARMS.pop(neutral)
            assert all(not LG.same_state(a, b) for a, b in zip(states(fit), states(rep))), arm
            assert all(float(fit["swa"][k].abs().max()) > 0 for k in QDP), arm
            for a, b in zip(states(frz), states(bfit)):
                assert not LG.same_state({k: a[k] for k in b}, b), arm
                assert all(float(a[k].abs().max()) == 0.0 for k in QDP), arm
            assert any(LG.same_state({k: a[k] for k in b}, b) for a, b in zip(states(fit), states(bfit))), arm
    finally:
        S.ARMS.pop(neutral, None)
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. the refusals: blocks without WALK or SEMB, a WALK of another width; train (another arm, split, fit name, cache
    #    root or device); read (a fit that is not A8ex's, not its arm's or not its split's); compare (a fit that is not
    #    A8ex's, a base that is not the adopted arm's card fit of the split, step 1's of another split, another screen
    #    name, a base record of another arm); pair and re-call of another arm. The fit names give the split
    for bl, wd in ((["rank", "SEMB"], {"rank": 5, "SEMB": 64}), (["rank", "WALK"], {"rank": 5, "WALK": QD.WALK_W}),
                   (["rank", "SEMB", "WALK"], {"rank": 5, "SEMB": 64, "WALK": 15})):
        for cl in (ZRCKPRQE, ZSPXQE):
            SR.must_stop(cl, bl, wd, 16)
    assert split_of_fit("scr-a8ex-zspxqe-hp") == ("zspxqe", "L-hotpotqa") and split_of_fit("scr-a8e-zspkqe") == (None,
                                                                                                                 None)
    good = ["train", "--arm", "zrckprqe", "--name", FIT["zrckprqe"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    for other in ("zrckpr", "zrckpqe", "zspx", "zspkqe"):
        SR.must_stop(train, ["train", "--arm", other, "--name", FIT["zrckprqe"]["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspxqe", "--name", FIT["zrckprqe"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspxqe", "--name", FIT["zspxqe"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckpr", "r1": {}}), ("scr-a8ex-zspxqe", {"arm": "zspkqe", "a8e": {}}),
                        ("scr-a8ex-zrckprqe", {"arm": "zrckprqe", "a8e": {"split": "L-musique"}}),
                        ("scr-a8ex-zrckprqe-hp", {"arm": "zspxqe", "a8ex": {"split": "L-hotpotqa"}}),
                        ("scr-a8ex-zspxqe-hp", {"arm": "zspxqe", "a8ex": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        fits_ = T_ / "outputs" / "screen" / "fits"
        for nm, rec in (("scr-a8ex-zrckprqe", {"arm": "zrckprqe", "a8ex": {"split": "L-musique"}}),
                        ("scr-a8ex-zspxqe-hp", {"arm": "zspxqe", "a8ex": {"split": "L-hotpotqa"}}),
                        ("scr-a8ex-zspxqe", {"arm": "zspxqe", "a8ex": {"split": "L-hotpotqa"}}),
                        ("scr-r1-zrckpr", {"arm": "zrckpr", "r1": {}}), ("scr-g5-zspx-hp", {"arm": "zspx", "g5": {}}),
                        ("scr-r1-zrckpr-hp", {"arm": "zrckp", "p1c3p": {}})):
            (fits_ / nm).mkdir(parents=True)
            LC.write_json(fits_ / nm / "screen.json", rec)
        st1 = T_ / "outputs" / "step1" / "fits"
        cmp_ = lambda nm, base, step1, out=None: ["--new", str(fits_ / nm), "--base", f"{fits_ / base},{st1 / step1}",
                                                  "--out", str(T_ / "outputs" / "screen" / (out or nm))]
        for bad in (cmp_("scr-a8e-zrckpqe", "scr-r1-zrckpr", "L-musique"),            # not an A8ex fit
                    cmp_("scr-a8ex-zrckprqe", "scr-p1c3p-zrckp", "L-musique"),        # the lean pair's base
                    cmp_("scr-a8ex-zrckprqe", "scr-r1-zrckpr-hp", "L-musique"),       # the other split's base
                    cmp_("scr-a8ex-zrckprqe", "scr-g5-zspx", "L-musique"),            # the other arm's base
                    cmp_("scr-a8ex-zrckprqe", "scr-r1-zrckpr", "L-hotpotqa"),         # step 1's of the other split
                    cmp_("scr-a8ex-zrckprqe", "scr-r1-zrckpr", "L-musique", "scr-a8ex-zspxqe"),   # another screen name
                    cmp_("scr-a8ex-zspxqe", "scr-g5-zspx", "L-musique"),              # the fit's record is the other split's
                    ["--new", str(fits_ / "scr-a8ex-zrckprqe"), "--base", str(fits_ / "scr-r1-zrckpr"), "--out",
                     str(T_ / "outputs" / "screen" / "scr-a8ex-zrckprqe")]):           # step 1's fit not beside
            SR.must_stop(compare, bad)
        saved_main, seen = S2.main, []
        S2.main = lambda a: seen.append(list(a)) or 0
        try:
            SR.must_stop(compare, cmp_("scr-a8ex-zrckprqe-hp", "scr-r1-zrckpr-hp", "L-hotpotqa"))   # base record zrckp's
            for nm, base, st in (("scr-a8ex-zrckprqe", "scr-r1-zrckpr", "L-musique"),
                                 ("scr-a8ex-zspxqe-hp", "scr-g5-zspx-hp", "L-hotpotqa")):
                assert compare(cmp_(nm, base, st)) == 0
            assert seen == [["compare"] + cmp_("scr-a8ex-zrckprqe", "scr-r1-zrckpr", "L-musique"),
                            ["compare"] + cmp_("scr-a8ex-zspxqe-hp", "scr-g5-zspx-hp", "L-hotpotqa")], seen
        finally:
            S2.main = saved_main
        SR.must_stop(pair, "zrckpqe", [], T_ / "x")
        SR.must_stop(recall, "zspx", [], T_ / "x.json", T_ / "x")
        # 8. relz's pair and re-call for each arm against its base arm's card fits, stamped A8ex's; relz restored after
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
            assert pj["stage"] == "A8ex" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A8ex ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A8ex's depth prior) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 9. the verdict, per arm, with the next forms of section 5: an adopt, a GAIN with a LOSS (the back-off), no GAIN
        #    beside the other arm's GAIN (the back-off on that model too), no GAIN on both (the prior closes as a model
        #    term), no GAIN beside an incomplete arm (decided with it), five kinds of INCOMPLETE; A8e's re-calls beside
        gm, mi_ = ("L-musique", "musique"), CLAIM1_READ
        lean = {a: fake_recall(T_ / f"lean-{a}.json", LEAN_OF[a], {gm: "GAIN"}, stage="A8e",
                               declared_in=A8E.DECLARED) for a in ARM_NAMES}
        rec = verdict({"zrckprqe": fake_recall(T_ / "a.json", "zrckprqe", {mi_: "GAIN"}),
                       "zspxqe": fake_recall(T_ / "b.json", "zspxqe", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v", lean)
        v = rec["arms"]
        assert v["zrckprqe"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert v["zrckprqe"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert v["zspxqe"]["verdict"] == "NOT_ADOPTED" and v["zspxqe"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckprqe": {"call": "GAIN", "delta": 0.0},
                                                   "zspxqe": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(v["zspxqe"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert len(v["zspxqe"]["a8e_beside"]) == 12 and v["zspxqe"]["a8e_beside"]["L-musique musique"] == {
            "call": "GAIN", "delta": 0.0}
        t = (T_ / "v.md").read_text(encoding="utf-8")
        assert "closes only this form" in t and ("| L-musique musique | WITHIN (+0.0000) | GAIN (+0.0000) | GAIN (+0.0000)"
                                                 " | GAIN (+0.0000) |") in t
        assert ("| L-hotpotqa webqsp | WITHIN (+0.0000) | WITHIN (+0.0000) | LOSS (+0.0000) | WITHIN (+0.0000) |") in t
        rec = verdict({"zrckprqe": fake_recall(T_ / "a.json", "zrckprqe", {("L-musique", "metaqa"): "LOSS"}),
                       "zspxqe": fake_recall(T_ / "b.json", "zspxqe", {gm: "GAIN", mi_: "LOSS"})}, T_ / "v", lean)
        v = rec["arms"]
        assert v["zrckprqe"]["verdict"] == "NOT_ADOPTED" and v["zrckprqe"]["next_form"] == NEXT["NO_GAIN_OTHER_GAINED"]
        assert v["zspxqe"]["verdict"] == "NOT_ADOPTED" and v["zspxqe"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        rec = verdict({"zrckprqe": fake_recall(T_ / "a.json", "zrckprqe", {mi_: "GAIN"}),
                       "zspxqe": fake_recall(T_ / "b.json", "zspxqe", {})}, T_ / "v", lean)
        assert rec["arms"]["zspxqe"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspxqe"]["next_form"] == NEXT["NO_GAIN_OTHER_GAINED"]
        rec = verdict({"zrckprqe": fake_recall(T_ / "a.json", "zrckprqe", {("L-musique", "metaqa"): "LOSS"}),
                       "zspxqe": fake_recall(T_ / "b.json", "zspxqe", {})}, T_ / "v", lean)
        assert all(rec["arms"][a]["verdict"] == "NOT_ADOPTED" and rec["arms"][a]["next_form"] == NEXT["NO_GAIN"]
                   for a in ARM_NAMES)
        t = (T_ / "v.md").read_text(encoding="utf-8")
        assert "three ways" in t and "(A8)" in t and "(A8e)" in t and "(A8ex)" in t and "gap ledger" in t
        rec = verdict({"zrckprqe": fake_recall(T_ / "a.json", "zrckprqe", {}),
                       "zspxqe": fake_recall(T_ / "b.json", "zspxqe", {gm: "GAIN"}, stage="A8e")}, T_ / "v", lean)
        v = rec["arms"]
        assert v["zspxqe"]["verdict"] == "INCOMPLETE" and v["zspxqe"]["next_form"] == NEXT["INCOMPLETE"]
        assert "is not A8ex's re-call" in v["zspxqe"]["problems"][0]
        assert v["zrckprqe"]["verdict"] == "NOT_ADOPTED" and v["zrckprqe"]["next_form"] == NEXT["NO_GAIN_PENDING"]
        rows11 = json.loads(fake_recall(T_ / "c.json", "zspxqe", {gm: "GAIN"}).read_text(encoding="utf-8"))["rows"][1:]
        rec = verdict({"zrckprqe": T_ / "none.json",
                       "zspxqe": fake_recall(T_ / "b.json", "zspxqe", {gm: "GAIN"}, rows=rows11)}, T_ / "v", lean)
        assert all(rec["arms"][a]["verdict"] == "INCOMPLETE" for a in ARM_NAMES)
        rec = verdict({"zrckprqe": fake_recall(T_ / "a.json", "zspxqe", {gm: "GAIN"}),
                       "zspxqe": fake_recall(T_ / "b.json", "zspxqe", {gm: "GAIN"}, declared_in=A8E.DECLARED)},
                      T_ / "v", lean)
        assert all(rec["arms"][a]["verdict"] == "INCOMPLETE" for a in ARM_NAMES)
        rec = verdict({"zrckprqe": fake_recall(T_ / "a.json", "zrckprqe", {mi_: "GAIN"}),
                       "zspxqe": fake_recall(T_ / "b.json", "zspxqe", {})}, T_ / "v",
                      {"zrckprqe": T_ / "none.json", "zspxqe": fake_recall(T_ / "c2.json", "zrckpqe", {}, stage="A8e",
                                                                           declared_in=A8E.DECLARED)})
        assert rec["arms"]["zrckprqe"]["a8e_beside"] == {} and rec["arms"]["zspxqe"]["a8e_beside"] == {}
        assert rec["arms"]["zrckprqe"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        t = (T_ / "v.md").read_text(encoding="utf-8")
        assert "A8e on zrckpqe: not found" in t and "| L-musique metaqa | WITHIN (+0.0000) | - | WITHIN (+0.0000) | - |" in t
    log(f"za8ex selftest: the arms are zrckpr's and zspx's models plus A8e's prior (its own function) on their own "
        f"carves, the base arms and A8e's arms unchanged; each starts as its base bit for bit, as built and with every "
        f"head set (state, global stream, scores in training and eval, every first-step gradient but the prior's, which "
        f"reach U, W and b); with the prior set zrckprqe is zrckpr plus l_q at each row's class (a float64 hand "
        f"computation) with the base's z-scores and no propagation, and zspxqe's step first reads the KT-mixed score "
        f"plus the prior, the z-scores before it the base's, its three increments rebuilt bit for bit; the prior reads "
        f"WALK and the question's embedding only, never a row's score nor R1's columns; toy fits repeat bit for bit, the "
        f"prior learns, and frozen at zero each fit is its base's; the refusals (blocks, train, read, compare's bases, "
        f"pair and re-call); relz's pair and re-call per arm, stamped A8ex's; the verdict's outcomes and next forms, "
        f"A8e's re-calls beside ({time.time() - t0:.1f}s): ok")
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
            log(f"za8ex {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za8ex: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
