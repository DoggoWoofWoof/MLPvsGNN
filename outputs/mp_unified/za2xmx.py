"""A2xmx (docs/A2XMX_ZTOP50_ADOPTED.md): ztop50's replacing form on the adopted arms, the next form A2xm's verdict named
(both arms NOT_ADOPTED with no GAIN; outputs/a2xm/screen-verdict.json). A2xm (za2xm.py) took every z-score zret takes
against the pool's 50 best retrieved rows by rrf, in place of all of them, on zrckp's and zspk's models: musique
zero-shot rose +0.0113 (MLP) and +0.0096 (GNN), within its floor, and the MLP lost webqsp zero-shot on L-hotpotqa
(-0.0351). R1's zrckpr and G5's zspx have passed their screens since; A2xmx moves the replacing form onto them, each
against its own base:

    zret's reference     the pool's retrieved rows (rrf > 0; lean_screen3.retrieved)
    A2xmx's reference    the pool's 50 best of them by rrf (ties by row order; lean_screen4.top_ref), A2xm's

Every z-score zret takes (each block's, fixed and SEMB, and rrf's base z-score) is taken against that reference; a pool
with fewer than two reference rows, or a column constant over them, keeps its whole-pool z-score, as in zret. The arm is
ztop50's forward under the adopted arm's heads (the chain term, the KT head, R1's head on zrckpr, zspx's three
increments on zspx): no parameter, column or random number is added, and where no pool has more than 50 retrieved rows
the arm is its base bit for bit. The reference is chosen by rrf, a first-stage column cached before any row is scored,
so the MLP keeps its rule: no score is read, nothing normalised over the question's rows is computed from a score.

    zrckprxm  zrckpr's model (outputs/mp_unified/zr1res.py) with ztop50's reference; R1's head reads the row's own nine
              columns, as zrckpr's does
    zspxxm    zspx's model (outputs/mp_unified/zg5mix.py) with ztop50's reference, before its three increments (its
              step, G3's residual message, G4's second step), which read the score they change

    python outputs/mp_unified/za2xmx.py train --split L-musique --name scr-a2xmx-zrckprxm --arm zrckprxm --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xmx.py read --name scr-a2xmx-zrckprxm --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xmx.py compare --new outputs/screen/fits/scr-a2xmx-zrckprxm \\
        --base outputs/screen/fits/scr-r1-zrckpr,outputs/step1/fits/L-musique --out outputs/screen/scr-a2xmx-zrckprxm
    python outputs/mp_unified/za2xmx.py pair --arm zrckprxm --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za2xmx.py recall --arm zrckprxm --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za2xmx.py verdict
    python outputs/mp_unified/za2xmx.py --selftest
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
import za2xm as A2XM  # noqa: E402  (A2xm: its reference and settings, its spies; its arms on the lean pair)
import lean_screen3 as S3  # noqa: E402  (zret: seg_zscore_ref, retrieved)
import lean_screen4 as S4  # noqa: E402  (ztop50: ZTop, top_ref)

import numpy as np  # noqa: E402
import torch  # noqa: E402

G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
Z3P = ZR1.Z3P
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A2XMX_ZTOP50_ADOPTED.md"
OUT = ROOT / "outputs" / "a2xmx"
SCREEN = ROOT / "outputs" / "screen"
TOPK = S4.K
REFERENCE = A2XM.REFERENCE                  # the pool's top 50 retrieved rows by rrf, in place of all of them
SETTINGS = {**A2XM.SETTINGS,
            "unchanged": "raw columns, keep flags, the chain term, the KT head, R1's head (zrckprxm), zspx's three "
                         "increments (zspxxm)",
            "parent": "A2xm (docs/A2XM_ZTOP50_REPLACE.md, outputs/mp_unified/za2xm.py)"}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckprxm": "zrckpr", "zspxxm": "zspx"}
LEAN_OF = {"zrckprxm": "zrckpxm", "zspxxm": "zspkxm"}            # A2xm's arm of the same form on the lean pair
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a2xmx-{a}", "L-hotpotqa": f"scr-a2xmx-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckprxm": {"L-musique": ("screen", "fits", "scr-r1-zrckpr"),
                          "L-hotpotqa": ("screen", "fits", "scr-r1-zrckpr-hp")},
             "zspxxm": {"L-musique": ("screen", "fits", "scr-g5-zspx"),
                        "L-hotpotqa": ("screen", "fits", "scr-g5-zspx-hp")}}
BASE_SCREENS = {"zrckprxm": {sp: f"outputs/p1gc/base-pools-zrckpr-{sp}.json" for sp in SPLITS},   # P1g-c's
                "zspxxm": {sp: f"outputs/a9/base-zspx-{sp}.json" for sp in SPLITS}}                # A9's
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckprxm": "zrckpr with zret's reference cut to the pool's top 50 retrieved rows by rrf (ztop50's form)",
        "zspxxm": "zspx with zret's reference cut to the pool's top 50 retrieved rows by rrf (ztop50's form), before "
                  "its three increments"}
PAIR = {a: SCREEN / f"scr-a2xmx-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a2xmx-{a}-pair-recall" for a in ARM_NAMES}
A2XM_RECALL = {a: SCREEN / f"scr-a2xm-{LEAN_OF[a]}-pair-recall.json" for a in ARM_NAMES}   # beside, deciding nothing
VERDICT = OUT / "screen-verdict"
BESIDE = ("musique", "metaqa", "webqsp")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A2xmx's re-call of the arm, "
        "or not twelve reads (docs/A2XMX_ZTOP50_ADOPTED.md, section 4)")


# ── the arms ─────────────────────────────────────────────────────────────────


class ZRCKPRXM(ZR1.ZRCKPR, S4.ZTop):
    """zrckprxm: zrckpr's model with ztop50's forward in place of zret's (MRO: ZRCKPR, ZRCKP, KTMixP, ZRM, ChainMatch,
    ZTop, ZRet): every z-score against the pool's top 50 retrieved rows by rrf; R1's head reads the row's own nine
    columns. Nothing added."""


class ZSPXXM(G5.ZSPX, S4.ZTop):
    """zspxxm: zspx's model with ztop50's forward in place of zret's; zspx's three increments read the score they change
    (MRO: ZSPX, ZSPR, ZSPK, ZProp, KTMix, ZRM, ChainMatch, ZTop, ZRet). Nothing added."""


ARMS = {"zrckprxm": (ZRCKPRXM, ZR1.R1KTChainCarveZRC), "zspxxm": (ZSPXXM, G3.ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


SIDE_KEY = {"zrckprxm": "r1_records", "zspxxm": "res_records"}


def side_records(arm, carves):
    """The records of the arm's own side input on carves: R1's columns for zrckprxm, G3's c and s for zspxxm; stops when
    one is missing or not its stage's."""
    return ZR1.r1_records(carves) if arm == "zrckprxm" else G3.res_records(carves)


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
        rec.update({"za2xmx_sha256": LC.sha_src(__file__), "za2xm_sha256": LC.sha_src(A2XM.__file__),
                    "zg5mix_sha256": LC.sha_src(G5.__file__), "zg3res_sha256": LC.sha_src(G3.__file__),
                    "zg4prop_sha256": LC.sha_src(G4.__file__), "zr1res_sha256": LC.sha_src(ZR1.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "lean_screen3_sha256": LC.sha_src(S3.__file__),
                    "lean_screen4_sha256": LC.sha_src(S4.__file__), "lean_screen_sha256": LC.sha_src(S.__file__),
                    "a2xmx": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                              "device": Z2.DEVICE, "settings": SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                              "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                              "kt_records": Z3.kt_records(Z3.train_carves(split)),
                              SIDE_KEY[arm]: side_records(arm, Z3.train_carves(split)),
                              "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                              "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za2xmx: train takes --arm {' or '.join(ARM_NAMES)}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za2xmx: A2xmx's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za2xmx: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a2xmx" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za2xmx: {name} is not an A2xmx fit (arm {rec.get('arm')})")
    if rec["a2xmx"].get("split") != split:
        raise SystemExit(f"za2xmx: {name} is {split}'s fit, its record says {rec['a2xmx'].get('split')}")
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
        raise SystemExit(f"za2xmx: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za2xmx: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A2xmx's top-50 reference) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A2xmx ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A2xmx", "declared_in": DECLARED, "decided_against": AGAINST[arm], "reference": REFERENCE,
                    "device": Z2.DEVICE, "za2xmx_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": "A2xmg, the learned back-off between the two references on the adopted arms: each z-score "
                         "taken as z_ret + 2 sigmoid(<w, c> + b) (z_top - z_ret), w and b zero so the scale starts at "
                         "1 (A2xmx's form), c the pool's log size and retrieved share (input columns only), on both "
                         "models, its own declared file",
        "NO_GAIN_OTHER_GAINED": "A2xmg on this model as well (the learned back-off between the two references, section "
                                "5): the other arm's GAIN keeps the top-50 idea open on both models",
        "NO_GAIN_PENDING": "decided with the other arm's re-call (section 5): the top-50 idea closes as a model input "
                           "only if neither arm gains",
        "NO_GAIN": "none for the top-50 reference as a model input, which closes: it has met both models beside zret's "
                   "z-scores (A2, A2h) and in their place, on the lean pair (A2xm) and on the adopted arms (A2xmx), "
                   "without a GAIN; ztop50's gain on step 1's model stays in the gap ledger, its shrinking trend beside",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, other_gains, pending):
    """An arm's next form (section 5): an arm without a GAIN goes to the back-off when the other arm gains, waits when
    the other arm's re-call is incomplete, and closes the idea only when neither arm gains."""
    if v != "NOT_ADOPTED":
        return NEXT[v]
    if gains:
        return NEXT["GAIN_AND_LOSS"]
    if other_gains:
        return NEXT["NO_GAIN_OTHER_GAINED"]
    return NEXT["NO_GAIN_PENDING"] if pending else NEXT["NO_GAIN"]


def lean_beside(arm, path=None):
    """A2xm's re-call of the same form on the lean pair (the arm's LEAN_OF twelve reads), beside, deciding nothing; {}
    when its file is missing or is not A2xm's re-call of that arm."""
    p = Path(path or A2XM_RECALL[arm])
    if not p.exists():
        return {}
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("stage") != "A2xm" or rec.get("declared_in") != A2XM.DECLARED or rec.get("arm") != LEAN_OF[arm]:
        return {}
    return {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rec.get("rows", [])}


def verdict(recalls=None, out=None, lean=None):
    """A2xmx's screen verdict, per arm, from its re-call (section 4); musique, metaqa and webqsp beside, musique's
    in-domain read of both arms side by side, A2xm's deltas on the lean pair beside, and the next form each outcome
    names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    lean = lean or {}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A2xmx" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A2xmx's re-call of {a}")
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
                  "a2xm_beside": lean_beside(a, lean.get(a)),
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    pending = any(per[a]["verdict"] == "INCOMPLETE" for a in ARM_NAMES)
    for a in ARM_NAMES:
        other_gains = any(per[b]["gains"] for b in ARM_NAMES if b != a and per[b]["verdict"] != "INCOMPLETE")
        per[a]["next_form"] = next_form(per[a]["verdict"], per[a]["gains"], other_gains, pending)
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A2xmx", "declared_in": DECLARED, "rule": RULE, "reference": REFERENCE, "arms": per,
           "musique_in_domain_beside": side, "za2xmx_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# A2xmx's screen verdict: " + ", ".join(f"{a} **{per[a]['verdict']}**" for a in ARM_NAMES), "",
          f"Rule: {RULE}.", ""]
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"
    for a in ARM_NAMES:
        p = per[a]
        lb = p["a2xm_beside"]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {b['call']} ({num(b['delta'])})" for k, b in p["beside"].items()) or "none")
                  + f"; A2xm on {LEAN_OF[a]}, the same reads: "
                  + (", ".join(f"{k} {lb[k]['call']} ({num(lb[k]['delta'])})" for k in p["beside"] if k in lb)
                     or "not found")
                  + f". Next: {p['next_form']}.")
    md += ["", f"Beside, deciding nothing: musique in-domain ({ck}): "
           + ", ".join(f"{a} {side[a]['call']} ({num(side[a]['delta'])})" for a in ARM_NAMES) + ".",
           "", "This verdict closes only this form (zret's reference cut to the pool's top 50 retrieved rows, in place "
               "of all of them, on the adopted arms); the top-50 idea closes as a model input only when neither arm "
               "gains (section 5); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za2xmx verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A2xmx", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def set_heads(*ms):
    """Every term and head away from zero, the same in each model (zero at the start, they would hide what the reference
    moves)."""
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
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    KC = 3                                                  # a cut the toy pools show (1 to 8 retrieved rows a pool)
    # 1. the arms: zrckpr's and zspx's models with ztop50's forward just above zret's; the base arms, A2xm's, zret's and
    #    ztop50's arms unchanged; ztop50's K and A2xm's reference; the arm classes add nothing of their own
    for a, (cls, cv) in ARMS.items():
        assert S.ARMS[a] == (cls, cv)
    assert S.ARMS["zrckpr"] == (ZR1.ZRCKPR, ZR1.R1KTChainCarveZRC) and S.ARMS["zspx"] == (G5.ZSPX, G3.ResKTLinkCarveBase)
    assert S.ARMS["ztop50"] == (S4.ZTop, S.ARMS["base"][1]) and S.ARMS["zret"] == (S3.ZRet, S.ARMS["base"][1])
    assert all(S.ARMS[LEAN_OF[a]] == A2XM.ARMS[LEAN_OF[a]] for a in ARM_NAMES)
    assert ZRCKPRXM.__mro__[1:8] == (ZR1.ZRCKPR, Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, S4.ZTop, S3.ZRet)
    assert ZSPXXM.__mro__[1:10] == (G5.ZSPX, G3.ZSPR, Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, S4.ZTop,
                                    S3.ZRet)
    assert S4.ZTop not in ZR1.ZRCKPR.__mro__ and S4.ZTop not in G5.ZSPX.__mro__
    assert all(k_.startswith("__") for c, _ in ARMS.values() for k_ in vars(c))
    assert TOPK == S4.K == ZRCKPRXM.K == ZSPXXM.K == 50 and REFERENCE == A2XM.REFERENCE
    assert A2XM.S4 is S4 and A2XM.S3 is S3 and all(m_.LG is LG for m_ in (S3, Z3, ZP, G3, G4))   # the spies see all
    assert BASE_SCREENS["zspxxm"] == {sp: f"outputs/a9/base-zspx-{sp}.json" for sp in SPLITS}
    rng = np.random.default_rng(61)                         # A2xm's toy pools: toy_roots draws first
    tmp = Path(tempfile.mkdtemp(prefix="za2xmx_"))
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
        blocks = ["rank", "SEMB"]
        nb = len(blocks)
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, nb)
        keep_[3, 1] = 0.0                                               # one question without SEMB
        f_, nq_, _bz, _gold = c_m.batch(qs, blocks)
        fg, nqg, _bzg, _goldg = c_g.batch(qs, blocks)
        assert Z3.KT_KEY in f_ and RM.CH_KEY in f_ and ZR1.R1_KEY in f_ and G3.RES_KEY not in f_
        assert G3.RES_KEY in fg and ZL.LK_KEY in fg and ZR1.R1_KEY not in fg
        assert torch.equal(nqg, nq_) and torch.equal(fg["rank"], f_["rank"])     # the two carves share the pools
        rrf = f_["rank"][:, S3.I_RRF]
        nret = torch.zeros(B).index_add_(0, nq_, (rrf > 0).float())
        cut = nret > KC
        # 2. the reference: top_ref against the hand computation; no toy pool has more than 50 retrieved rows, so
        #    A2xmx's reference is zret's there; at KC it keeps the KC best retrieved rows of each pool with more, and
        #    every retrieved row of the others
        assert int(nret.max()) <= TOPK and 0 < int(cut.sum()) < B
        ret = S3.retrieved(f_)
        for k in (TOPK, KC, 1):
            assert np.array_equal(S4.top_ref(rrf, nq_, B, k).numpy(), S4.top_ref_numpy(rrf, nq_, B, k)), k
        assert torch.equal(S4.top_ref(rrf, nq_, B, TOPK), ret)
        tr = S4.top_ref(rrf, nq_, B, KC)
        assert bool((tr <= ret).all())
        assert torch.equal(torch.zeros(B).index_add_(0, nq_, tr), torch.where(cut, torch.full_like(nret, KC), nret))
        assert torch.equal(tr[~cut[nq_]], ret[~cut[nq_]]) and not torch.equal(tr[cut[nq_]], ret[cut[nq_]])
        # 3. the MLP rule (section 2), spied in each arm's forward at KC: top_ref runs once, on the batch's rrf column
        #    itself (an input, with no gradient); every z-score zret took (each block's, then rrf's base) is taken
        #    against that reference, and zret's whole retrieved reference is never formed (replaced, not beside); the
        #    reference is the same whatever the weights. In zrckprxm every within-pool normalisation takes an input
        #    column (the rank block, SEMB's learned map, rrf) and none takes the score; R1's head is the row's own. In
        #    zspxxm the KT head and the three increments z-score the score over the whole pool, as zspx's do: exactly
        #    four, on the chain-matched score, the KT-mixed score, the score after the step and after G3's residual
        for arm, cls, cv in (("zrckprxm", ZRCKPRXM, c_m), ("zspxxm", ZSPXXM, c_g)):
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            r2 = f2["rank"][:, S3.I_RRF]
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            set_heads(m1)
            m1.eval()
            m1.K = KC
            with A2XM.spied({}) as lg, torch.no_grad():
                y1 = m1(f2, keep_, nq2, B, bz2)
            assert len(lg["top"]) == 1 and not lg["ret"], (arm, len(lg["top"]), len(lg["ret"]))
            r_in, k_in, ref = lg["top"][0]
            assert k_in == KC and torch.equal(r_in, r2) and not r_in.requires_grad
            assert torch.equal(ref, S4.top_ref(r2, nq2, B, KC)) and not torch.equal(ref, S3.retrieved(f2))
            assert len(lg["ref"]) == nb + 1 and all(rf is ref for _, rf in lg["ref"])
            with torch.no_grad():
                want = [f2["rank"], S.raw_of(m1, f2, "SEMB", nq2), r2.unsqueeze(1)]
            assert all(torch.equal(x, w_) for (x, _), w_ in zip(lg["ref"], want))
            assert len(lg["z8"]) >= nb + 1 and all(x is x_ for x, (x_, _) in zip(lg["z8"], lg["ref"]))
            extra = lg["z8"][nb + 1:]
            if arm == "zrckprxm":
                assert not extra, len(extra)                            # nothing else is normalised over the rows
            else:
                with torch.no_grad():
                    lk = f2[ZL.LK_KEY]
                    pre = RM.ChainMatch.forward(m1, f2, keep_, nq2, B, bz2)
                    kts = Z3.KTMix.forward(m1, f2, keep_, nq2, B, bz2)
                    s1 = kts + m1.prop_head(G4.prop_inputs_grad(kts, lk, nq2, B))
                    s2 = s1 + m1.rs_head(G3.res_inputs(s1, lk, f2[G3.RES_KEY], nq2, B))
                    y_ = s2 + m1.p2_head(ZP.prop_inputs(s2, lk, nq2, B))
                assert len(extra) == 4, len(extra)
                assert all(torch.equal(x.squeeze(1), w_) for x, w_ in zip(extra, (pre, kts, s1, s2)))
                assert torch.equal(y_, y1)
            m2 = copy.deepcopy(m1)
            with torch.no_grad():
                g_ = torch.Generator().manual_seed(12)
                for p_ in m2.parameters():
                    p_.copy_(torch.randn(p_.shape, generator=g_))
            with A2XM.spied({}) as lg2, torch.no_grad():
                y2 = m2(f2, keep_, nq2, B, bz2)
            assert len(lg2["top"]) == 1 and torch.equal(lg2["top"][0][2], ref)
            assert bool(torch.isfinite(y2).all()) and not torch.allclose(y2, y1)
        # 4. each arm against its base, every head's last layer set. At ztop50's K (no toy pool has more than 50
        #    retrieved rows): the same weights, with no parameter added and no random number drawn, and bit for bit the
        #    base's scores in eval and training and every gradient. At KC: bit for bit the base model with zret's
        #    reference replaced by the top-KC one, scores in eval and training and every gradient, so the reference is
        #    all that differs; l1's raw and keep columns are the base's, and its z-score columns and the scores move
        #    only on the pools with more than KC retrieved rows
        for arm, cls, base, cv in (("zrckprxm", ZRCKPRXM, ZR1.ZRCKPR, c_m), ("zspxxm", ZSPXXM, G5.ZSPX, c_g)):
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            rows_cut = cut[nq2]
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            assert not LG.same_state(m0.state_dict(), m1.state_dict())
            set_heads(m0, m1)
            for K_ in (TOPK, KC):
                m1.K = K_
                for mode in ("eval", "train"):
                    ys = []
                    for m_, k_cut in ((m0, None if K_ == TOPK else K_), (m1, None)):
                        getattr(m_, mode)()
                        m_.zero_grad()
                        torch.manual_seed(3)
                        with A2XM.zret_ref_cut(k_cut, nq2, B):
                            y = m_(f2, keep_, nq2, B, bz2)
                        (y * torch.linspace(-1, 1, y.numel())).sum().backward()
                        ys.append(y)
                    assert torch.equal(ys[0], ys[1]), (arm, K_, mode)
                    p1 = dict(m1.named_parameters())
                    for n_, p0 in m0.named_parameters():
                        g0, g1 = p0.grad, p1[n_].grad
                        assert (g0 is None and g1 is None) or torch.equal(g0, g1), (arm, K_, mode, n_)
            m0.eval()
            m1.eval()
            m1.K = KC
            seen = {}
            hooks = [m_.l1.register_forward_hook(lambda mod, inp, out, key=key: seen.__setitem__(key, inp[0].detach()))
                     for key, m_ in (("base", m0), ("arm", m1))]
            try:
                with torch.no_grad():
                    yb = m0(f2, keep_, nq2, B, bz2)
                    ya = m1(f2, keep_, nq2, B, bz2)
            finally:
                for h_ in hooks:
                    h_.remove()
            zc, rk, o = [], [], 0
            for b in blocks:
                w = int(cv.widths[b])
                rk += list(range(o, o + w)) + [o + 2 * w]
                zc += list(range(o + w, o + 2 * w))
                o += 2 * w + 1
            xb, xa = seen["base"], seen["arm"]
            assert tuple(xa.shape) == tuple(xb.shape) == (nq2.numel(), o)
            assert torch.equal(xa[:, rk], xb[:, rk])
            assert torch.equal(xa[~rows_cut][:, zc], xb[~rows_cut][:, zc]) and torch.equal(ya[~rows_cut], yb[~rows_cut])
            for j in torch.nonzero(cut).flatten().tolist():
                assert float((xa[nq2 == j][:, zc] - xb[nq2 == j][:, zc]).abs().max()) > 1e-4, (arm, j)
                assert float((ya[nq2 == j] - yb[nq2 == j]).abs().max()) > 1e-4, (arm, j)
        # 5. toy fits through lean_gpu's loop: at ztop50's K the fit is its base's bit for bit (no toy pool has more
        #    than 50 retrieved rows); a repeat is bit-identical; at KC the fit differs from the base's
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        states = lambda f: f["states"] + [f["swa"]]
        for arm, cv in (("zrckprxm", c_m), ("zspxxm", c_g)):
            cls = ARMS[arm][0]
            with S.patched(BASE_OF[arm]):
                bfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                cls.K = KC
                try:
                    kfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} k{KC}")
                finally:
                    del cls.K
            assert cls.K == TOPK
            assert not any(LG.same_state(a, b) for a, b in zip(states(fit), states(rep)))
            assert not any(LG.same_state(a, b) for a, b in zip(states(fit), states(bfit)))
            assert LG.same_state(kfit["swa"], bfit["swa"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the refusals; the fit names give the split
    assert split_of_fit("scr-a2xmx-zspxxm-hp") == ("zspxxm", "L-hotpotqa") and split_of_fit("scr-a2xm-zspkxm") == (None,
                                                                                                                 None)
    good = ["train", "--arm", "zrckprxm", "--name", FIT["zrckprxm"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    for other in ("zrckpr", "zrckpxm", "ztop50"):
        SR.must_stop(train, ["train", "--arm", other, "--name", FIT["zrckprxm"]["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspxxm", "--name", FIT["zrckprxm"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspxxm", "--name", FIT["zspxxm"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckpr", "r1": {}}), ("scr-a2xmx-zspxxm", {"arm": "zspkxm", "a2xm": {}}),
                        ("scr-a2xmx-zrckprxm", {"arm": "zrckprxm", "a2xm": {"split": "L-musique"}}),
                        ("scr-a2xmx-zrckprxm-hp", {"arm": "zspxxm", "a2xmx": {"split": "L-hotpotqa"}}),
                        ("scr-a2xmx-zspxxm-hp", {"arm": "zspxxm", "a2xmx": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 7. relz's pair and re-call for each arm against its base arm's card fits, stamped A2xmx's; relz restored after
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
            assert pj["stage"] == "A2xmx" and pj["decided_against"] == AGAINST[arm] and pj["reference"] == REFERENCE
            assert pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A2xmx ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A2xmx's" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 8. the verdict, per arm, with the next forms of section 5: an adopt, a GAIN with a LOSS (the back-off), no
        #    GAIN beside the other arm's GAIN (the back-off on that model too), no GAIN on both (the idea closes), no
        #    GAIN beside an incomplete arm (decided with it), five kinds of INCOMPLETE; A2xm's re-calls beside
        gm, mi_ = ("L-musique", "musique"), CLAIM1_READ
        lean = {a: fake_recall(T_ / f"lean-{a}.json", LEAN_OF[a], {gm: "GAIN"}, stage="A2xm",
                               declared_in=A2XM.DECLARED) for a in ARM_NAMES}
        rec = verdict({"zrckprxm": fake_recall(T_ / "a.json", "zrckprxm", {mi_: "GAIN"}),
                       "zspxxm": fake_recall(T_ / "b.json", "zspxxm", {gm: "GAIN", ("L-hotpotqa", "metaqa"): "LOSS"})},
                      T_ / "v", lean)
        v = rec["arms"]
        assert v["zrckprxm"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert v["zrckprxm"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert v["zspxxm"]["verdict"] == "NOT_ADOPTED" and v["zspxxm"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"]["zrckprxm"] == {"call": "GAIN", "delta": 0.0}
        assert sorted(v["zspxxm"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert len(v["zspxxm"]["a2xm_beside"]) == 12 and v["zspxxm"]["a2xm_beside"]["L-musique musique"] == {
            "call": "GAIN", "delta": 0.0}
        t = (T_ / "v.md").read_text(encoding="utf-8")
        line = next(ln for ln in t.splitlines() if ln.startswith("- **zspxxm**"))
        assert "closes only this form" in t and ("A2xm on zspkxm, the same reads: L-musique metaqa WITHIN (+0.0000), "
                                                 "L-musique musique GAIN (+0.0000)") in line, line
        rec = verdict({"zrckprxm": fake_recall(T_ / "a.json", "zrckprxm", {("L-musique", "metaqa"): "LOSS"}),
                       "zspxxm": fake_recall(T_ / "b.json", "zspxxm", {gm: "GAIN", mi_: "LOSS"})}, T_ / "v", lean)
        v = rec["arms"]
        assert v["zrckprxm"]["verdict"] == "NOT_ADOPTED" and v["zrckprxm"]["next_form"] == NEXT["NO_GAIN_OTHER_GAINED"]
        assert v["zspxxm"]["verdict"] == "NOT_ADOPTED" and v["zspxxm"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        rec = verdict({"zrckprxm": fake_recall(T_ / "a.json", "zrckprxm", {mi_: "GAIN"}),
                       "zspxxm": fake_recall(T_ / "b.json", "zspxxm", {})}, T_ / "v", lean)
        assert rec["arms"]["zspxxm"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspxxm"]["next_form"] == NEXT["NO_GAIN_OTHER_GAINED"]
        rec = verdict({"zrckprxm": fake_recall(T_ / "a.json", "zrckprxm", {("L-musique", "metaqa"): "LOSS"}),
                       "zspxxm": fake_recall(T_ / "b.json", "zspxxm", {})}, T_ / "v", lean)
        assert all(rec["arms"][a]["verdict"] == "NOT_ADOPTED" and rec["arms"][a]["next_form"] == NEXT["NO_GAIN"]
                   for a in ARM_NAMES)
        rec = verdict({"zrckprxm": fake_recall(T_ / "a.json", "zrckprxm", {}),
                       "zspxxm": fake_recall(T_ / "b.json", "zspxxm", {gm: "GAIN"}, stage="A2xm")}, T_ / "v", lean)
        v = rec["arms"]
        assert v["zspxxm"]["verdict"] == "INCOMPLETE" and v["zspxxm"]["next_form"] == NEXT["INCOMPLETE"]
        assert "is not A2xmx's re-call" in v["zspxxm"]["problems"][0]
        assert v["zrckprxm"]["verdict"] == "NOT_ADOPTED" and v["zrckprxm"]["next_form"] == NEXT["NO_GAIN_PENDING"]
        rows11 = json.loads(fake_recall(T_ / "c.json", "zspxxm", {gm: "GAIN"}).read_text(encoding="utf-8"))["rows"][1:]
        rec = verdict({"zrckprxm": T_ / "none.json",
                       "zspxxm": fake_recall(T_ / "b.json", "zspxxm", {gm: "GAIN"}, rows=rows11)}, T_ / "v", lean)
        assert all(rec["arms"][a]["verdict"] == "INCOMPLETE" for a in ARM_NAMES)
        rec = verdict({"zrckprxm": fake_recall(T_ / "a.json", "zspxxm", {gm: "GAIN"}),
                       "zspxxm": fake_recall(T_ / "b.json", "zspxxm", {gm: "GAIN"}, declared_in=A2XM.DECLARED)},
                      T_ / "v", lean)
        assert all(rec["arms"][a]["verdict"] == "INCOMPLETE" for a in ARM_NAMES)
        rec = verdict({"zrckprxm": fake_recall(T_ / "a.json", "zrckprxm", {mi_: "GAIN"}),
                       "zspxxm": fake_recall(T_ / "b.json", "zspxxm", {})}, T_ / "v",
                      {"zrckprxm": T_ / "none.json", "zspxxm": fake_recall(T_ / "c2.json", "zrckpxm", {}, stage="A2xm",
                                                                          declared_in=A2XM.DECLARED)})
        assert rec["arms"]["zrckprxm"]["a2xm_beside"] == {} and rec["arms"]["zspxxm"]["a2xm_beside"] == {}
        assert rec["arms"]["zrckprxm"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert "A2xm on zrckpxm, the same reads: not found" in (T_ / "v.md").read_text(encoding="utf-8")
    log(f"za2xmx selftest: the two arms are zrckpr's and zspx's models with ztop50's forward just above zret's, adding "
        f"nothing; the reference is top_ref's (a hand computation agrees), zret's at K 50 on the toy, the {KC} best "
        f"retrieved rows of the pools with more at K {KC}; MLP rule: top_ref runs once on the batch's rrf column, every "
        f"z-score takes its reference and zret's is never formed, the reference ignores the weights, zrckprxm "
        f"normalises only input columns over a pool, zspxxm's KT head and three increments z-score the score as zspx's "
        f"do (exactly four, each checked); each arm is its base bit for bit at K 50 and, at K {KC}, its base with only "
        f"the reference replaced (scores and every gradient, eval and training), moving l1's z-score columns and the "
        f"scores only on the cut pools; toy fits equal the base's at K 50, repeat bit for bit and differ at K {KC}; "
        f"train refuses another arm, split, fit name, cache root or device; read refuses a fit that is not A2xmx's, not "
        f"its arm's or not its split's; relz's pair and re-call per arm, stamped A2xmx's; the verdict's outcomes and "
        f"next forms, A2xm's re-calls beside ({time.time() - t0:.1f}s): ok")
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
            log(f"za2xmx {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za2xmx: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
