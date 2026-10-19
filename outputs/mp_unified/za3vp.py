"""A3vp (docs/A3VP_P1GB_POOLS.md): A3v's adopted MLP arm fitted again on P1g-b's pools, the next form A3v's verdict
named for zrckpy (outputs/a3v/screen-verdict.json). A3v (za3v.py) trained the lean pair with vshare's two-sided share
rule in zret's reference; its MLP arm passed on P1c2's pools and its GNN arm did not (A3z is the GNN's next form, not a
refit). No model change is adopted on today's pools (the user's rule of 13 October), so the MLP arm is fitted again on
P1g-b's pools and decided against P1g-b's own card fit of its base arm there:

    zrckpy  A3v's MLP as filed (za3v.ZRCKPY on zp1c3.KTChainCarveZRC), against P1g-b's scr-p1gb-zrckp, -hp

The arm is za3v's own object and its carve is the lean MLP's, so A3vp builds nothing: it reads P1g-b's caches, KT
columns and zrc's chains on P1g-b's roots (zp1gb.on_p1gb), as P1g-c's arms read them, behind P1g-c's gate
(zp1gc.gate_open). Every fit and read refuses while that gate is shut, missing or another stage's, and refuses a P1g-b
record rebuilt after it opened. A3v's rule acts in training only; reads are zret's forward, unchanged.

    python outputs/mp_unified/za3vp.py compare --new outputs/screen/fits/scr-p1gb-zrckp \\
        --base outputs/step1/fits/L-musique --out outputs/a3vp/base-zrckp-L-musique
    python outputs/mp_unified/za3vp.py train --split L-musique --name scr-a3vp-zrckpy --arm zrckpy --device cuda \\
        --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za3vp.py read --name scr-a3vp-zrckpy --device cuda --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za3vp.py compare --new outputs/screen/fits/scr-a3vp-zrckpy \\
        --base outputs/screen/fits/scr-p1gb-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a3vp-zrckpy
    python outputs/mp_unified/za3vp.py pair --arm zrckpy --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za3vp.py recall --arm zrckpy --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za3vp.py verdict
    python outputs/mp_unified/za3vp.py --selftest
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

import za3v as A3V  # noqa: E402  (A3v's arms, rule, counts and settings; through it za3vshare, za3t, zp1c3p, zp1c3)
import zp1gb as GB  # noqa: E402  (P1g-b's roots, gate and lean-pair fits)
import zp1gc as P1GC  # noqa: E402  (P1g-c's gate on P1g-b's records)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Z3, Z3P, Z2, ZA3, ZT = A3V.Z3, A3V.Z3P, A3V.Z2, A3V.ZA3, A3V.ZT
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
S3, S7 = A3V.S3, A3V.S7
log = S.log

DECLARED = "docs/A3VP_P1GB_POOLS.md"
OUT = ROOT / "outputs" / "a3vp"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = A3V.SPLITS
SX = {"L-musique": "", "L-hotpotqa": "-hp"}
ARM_NAMES = ("zrckpy",)                                       # A3v's adopted arm; zspky's next form is A3z
ARMS = {a: A3V.ARMS[a] for a in ARM_NAMES}                    # A3v's (model, carve), the same objects
BASE_OF = {a: A3V.BASE_OF[a] for a in ARM_NAMES}              # zrckpy -> zrckp
FIT = {a: {sp: f"scr-a3vp-{a}{SX[sp]}" for sp in SPLITS} for a in ARM_NAMES}
BASE_FITS = {a: {sp: ("screen", "fits", GB.FIT[BASE_OF[a]][sp]) for sp in SPLITS} for a in ARM_NAMES}
BASE_SCREENS = {a: {sp: f"outputs/a3vp/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split on P1g-b's pools (P1g-b's {GB.FIT[BASE_OF[a]]['L-musique']}, "
              f"{GB.FIT[BASE_OF[a]]['L-hotpotqa']})" for a in ARM_NAMES}
WHAT = {a: A3V.WHAT[a] for a in ARM_NAMES}
PAIR = {a: SCREEN / f"scr-a3vp-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a3vp-{a}-pair-recall" for a in ARM_NAMES}
COMPARE = {a: {sp: SCREEN / f"scr-a3vp-{a}{SX[sp]}" for sp in SPLITS} for a in ARM_NAMES}
A3V_RECALL = {a: A3V.RECALL[a].with_suffix(".json") for a in ARM_NAMES}      # A3v's re-call on P1c2's pools, beside
VERDICT = OUT / "screen-verdict"
BESIDE = A3V.BESIDE                                           # metaqa, musique, squad, webqsp
CLAIM1_READ = A3V.CLAIM1_READ                                 # ("L-hotpotqa", "musique")
POOLS = "P1g-b's (outputs/p1gb/pools)"
RULE = ("ADOPT_FOR_FULL_RUN when the arm's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A3vp's re-call of the arm, or "
        "not twelve reads (docs/A3VP_P1GB_POOLS.md, section 4; A3v's rule)")
NEXT = {"ADOPT_FOR_FULL_RUN": "A3v's two-sided share rule joins the full run's MLP when it runs on P1g-b's pools: one "
                              "file, declared before its numbers, after P1g-c's, A5cp's, A7p's and A3vp's verdicts, "
                              "with the composition P1g-c's section 5 sets; the GNN's share form is A3z's to decide",
        "GAIN_AND_LOSS": "A3vs on P1g-b's pools: A3v's rule with the reference's retrieved share t as an input column "
                         "(1 at every read), the back-off form A3v's section 5 names, its own declared file",
        "NO_GAIN": "the share idea stays open on the MLP (it gained on P1c2's pools; docs/SCREENS.md, 3b): A3z's "
                   "whole-pool z-scores beside zret's on P1g-b's pools, its own declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


# ── train and read, on P1g-b's roots behind P1g-c's gate ─────────────────────


def split_of_fit(name):
    for a in ARM_NAMES:
        for sp, n in FIT[a].items():
            if n == name:
                return a, sp
    return None, None


def stamp(argv, arm, split, g, gate_file=None):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        mt = A3V.last_means()
        rec.update({"za3vp_sha256": LC.sha_src(__file__), "za3v_sha256": LC.sha_src(A3V.__file__),
                    "za3t_sha256": LC.sha_src(ZT.__file__), "za3vshare_sha256": LC.sha_src(ZA3.__file__),
                    "zp1gc_sha256": LC.sha_src(P1GC.__file__), "zp1gb_sha256": LC.sha_src(GB.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "lean_screen3_sha256": LC.sha_src(S3.__file__),
                    "lean_screen7_sha256": LC.sha_src(S7.__file__),
                    "a3vp": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                             "base_fit": GB.FIT[BASE_OF[arm]][split], "device": Z2.DEVICE, "settings": A3V.SETTINGS,
                             "last_fit": {**A3V.LAST, **mt}, "pools": POOLS,
                             "cache_root": GB.CACHE.relative_to(ROOT).as_posix(), "p1gb_gate": LC.sha_file(GB.GATE),
                             "p1gc_gate": LC.sha_file(Path(gate_file or P1GC.GATE).with_suffix(".json")),
                             "snap": g.get("snap_sha256"),
                             "kt_records": Z3.kt_records(Z3.train_carves(split), GB.KT_ROOT),
                             "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(GB.CH_ZRC),
                             "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)
        log(f"  a3vp {arm} {split}: {A3V.LAST['applied']} of {A3V.LAST['questions']} training questions reweighted in "
            f"the last fit ({A3V.LAST['eligible']} eligible, {A3V.LAST['eligible_rich']} of them under {A3V.MID}); "
            f"{A3V.LAST['applied_rich']} on the rich side (mean t {mt['mean_t_rich']:.3f}), "
            f"{A3V.LAST['applied_thin']} on the thin side (mean t {mt['mean_t_thin']:.3f})")


def train(argv, split, gate_file=None, roots=None):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za3vp: train takes --arm {' or '.join(ARM_NAMES)} (A3v's adopted arm as filed), not {arm} "
                         f"({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za3vp: A3vp fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za3vp: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    with GB.on_p1gb():
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = P1GC.gate_open(gate_file, roots)
        GB.gate_passed()
        Z3.kt_records(Z3.train_carves(split), GB.KT_ROOT)
        rc = RM.train(argv, split)
        stamp(argv, arm, split, g, gate_file)
    return rc


def read(argv, gate_file=None, roots=None):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a3vp" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za3vp: {name} is not an A3vp fit (arm {rec.get('arm')})")
    if rec["a3vp"].get("split") != split:
        raise SystemExit(f"za3vp: {name} is {split}'s fit, its record says {rec['a3vp'].get('split')}")
    with GB.on_p1gb():
        Z2.cache_root_of(argv)
        P1GC.gate_open(gate_file, roots)
        Z3.kt_records(Z3.eval_carves(), GB.KT_ROOT)
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, the arm decided against P1g-b's fit of its base arm ──────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za3vp: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za3vp: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp, P1g-b's pools)"),
            ("(rel's screen fits)", f"({b}'s card fits on P1g-b's pools)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 (P1g-b's) | {arm} R@5 (A3v's rule, P1g-b's) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A3vp ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}, both on P1g-b's pools.\n\n" + t,
                      encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A3vp", "declared_in": DECLARED, "decided_against": AGAINST[arm], "pools": POOLS,
                    "device": Z2.DEVICE, "za3vp_sha256": LC.sha_src(__file__)})
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


def recall_rows(path, arm, stage, declared):
    """(rows, problems) of a re-call: its stage, declaration and arm, and the twelve reads."""
    rp, problems, rows = Path(path), [], []
    if not rp.exists():
        return rows, [f"{rp.name} missing"]
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec.get("stage") != stage or rec.get("declared_in") != declared or rec.get("arm") != arm:
        problems.append(f"{rp.name} is not {stage}'s re-call of {arm}")
    rows = rec.get("rows", [])
    if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
            (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
        problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    return rows, problems


def side_of(path):
    """{dataset: [delta R@5, FC@5, hit@1]} of a compare against its first base, or {} when it is not there."""
    p = Path(path).with_suffix(".json")
    if not p.exists():
        return {}
    rec = json.loads(p.read_text(encoding="utf-8"))
    return {r["dataset"]: list(r["delta"]) for r in rec["by_base"][rec["bases"][0]]["rows"]}


def counts_of(arm, split, fits_root=None):
    """The reweighting counts A3vp's fit stamped (A3v's LAST and mean t at the fit's end), or None."""
    sj = Path(fits_root or SCREEN / "fits") / FIT[arm][split] / "screen.json"
    if not sj.exists():
        return None
    return (json.loads(sj.read_text(encoding="utf-8")).get("a3vp") or {}).get("last_fit")


def verdict(recalls=None, compares=None, a3v=None, out=None, fits_root=None):
    """A3vp's verdict on its arm, from its re-call (section 4), with the next form each outcome names (section 5);
    beside, deciding nothing: metaqa, musique, squad and webqsp, musique in domain (Claim 1's read), each read's FC@5 and
    hit@1 from the compares, each read's delta beside A3v's on P1c2's pools, and each fit's reweighting counts."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    compares = compares or COMPARE
    a3v = a3v or A3V_RECALL
    per = {}
    for a in ARM_NAMES:
        rows, problems = recall_rows(recalls[a], a, "A3vp", DECLARED)
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        oc = v if v != "NOT_ADOPTED" else ("GAIN_AND_LOSS" if gains else "NO_GAIN")
        old_rows, old_problems = recall_rows(a3v[a], a, "A3v", A3V.DECLARED)
        old = {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in old_rows}
        side = {sp: side_of(compares[a][sp]) for sp in SPLITS}
        reads = {}
        for r in rows:
            d = side.get(r.get("split"), {}).get(r.get("dataset"))
            reads[tag(r)] = {"read": r.get("read"), "call": r.get("call"), "call_filed": r.get("call_filed"),
                             "base": r.get("base"), "new": r.get("new"), "delta": r.get("delta"), "lo": r.get("lo"),
                             "hi": r.get("hi"), "fc5": d[1] if d else None, "hit1": d[2] if d else None,
                             "a3v_p1c2": old.get(tag(r))}
        per[a] = {"verdict": v, "outcome": oc, "gains": gains, "losses": losses, "problems": problems,
                  "next_form": NEXT[oc], "reads": reads,
                  "beside": {t: x for t, x in reads.items() if t.split()[1] in BESIDE},
                  "training_counts": {sp: counts_of(a, sp, fits_root) for sp in SPLITS},
                  "a3v_recall_problems": old_problems,
                  "recall_sha256": LC.sha_file(recalls[a]) if Path(recalls[a]).exists() else None,
                  "a3v_recall_sha256": LC.sha_file(a3v[a]) if Path(a3v[a]).exists() else None}
    a = ARM_NAMES[0]
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = per[a]["reads"].get(ck)
    vs = {x: per[x]["verdict"] for x in ARM_NAMES}
    if vs[a] == "INCOMPLETE":
        f1 = "not settled while the re-call is INCOMPLETE"
    elif vs[a] == "ADOPT_FOR_FULL_RUN":
        f1 = ("zrckpy adopted: the full run's MLP trains with A3v's two-sided rule. Its inputs are zrckp's, so F1's one "
              "input set is unchanged; the full run's file names the training rule as a difference between its models "
              "unless A3z settles the GNN's share form")
    else:
        f1 = "zrckpy not adopted: the full run's MLP trains as zrckp does"
    rec = {"stage": "A3vp", "declared_in": DECLARED, "rule": RULE, "pools": POOLS, "arms": per,
           "musique_in_domain_beside": side, "f1": f1, "za3vp_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"  # noqa: E731
    p = per[a]
    md = [f"# A3vp's screen verdict: {a} **{vs[a]}**", "", f"Rule: {RULE}.", "",
          f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
          f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
          + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
          + (", ".join(f"{t} {b['call']} ({num(b['delta'])})" for t, b in p["beside"].items()) or "none")
          + f". Next: {p['next_form']}.", "", f"F1: {f1}.", "",
          f"{a}, each read beside, deciding nothing (FC@5 and hit@1 from `{Path(compares[a]['L-musique']).name}` and "
          f"its -hp; A3v's delta and call on P1c2's pools from `{Path(a3v[a]).name}`):", "",
          "| split | dataset | read | delta R@5 [95% CI] | call | FC@5 | hit@1 | A3v on P1c2's pools |",
          "| --- | --- | --- | --- | --- | ---: | ---: | --- |"]
    for t, x in p["reads"].items():
        o = x["a3v_p1c2"] or {}
        ci = f" [{num(x['lo'])}, {num(x['hi'])}]" if x["lo"] is not None and x["hi"] is not None else ""
        filed = f" (filed {x['call_filed']})" if x["call_filed"] not in (None, x["call"]) else ""
        md.append(f"| {t.split()[0]} | {t.split()[1]} | {x['read']} | {num(x['delta'])}{ci} | {x['call']}{filed} | "
                  f"{num(x['fc5'])} | {num(x['hit1'])} | {num(o.get('delta'))} {o.get('call') or ''} |")
    md += ["", f"{a}'s reweighting in training (stamped by each fit): " + "; ".join(
        f"{sp} {c['applied']:,} of {c['questions']:,} question visits, {c['applied_rich']:,} on the rich side and "
        f"{c['applied_thin']:,} on the thin side" if c else f"{sp} not stamped"
        for sp, c in p["training_counts"].items()) + ".", "",
        f"Beside, deciding nothing: musique in domain ({ck}, Claim 1's read): {a} {(side or {}).get('call')} "
        f"({num((side or {}).get('delta'))}).", "",
        "This verdict closes only this form (A3v's two-sided rule on the MLP, on P1g-b's pools); docs/SCREENS.md, 3b. "
        "Which pools the full run uses is P1g-c's call (its section 5): on P1g-b's pools this verdict decides whether "
        "the rule joins the MLP; on P1c2's pools A3v's own verdict decides, and this one is reported beside."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za3vp verdict: {a} {vs[a]}; F1: {f1}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, stage="A3vp", declared=DECLARED, rows=None, filed=None, **over):
    rows = rows if rows is not None else [
        {"split": sp, "dataset": ds, "read": "x", "delta": round(0.001 * j, 4), "lo": round(0.001 * j - 0.0005, 4),
         "hi": round(0.001 * j + 0.0005, 4), "call": calls.get((sp, ds), "WITHIN"),
         "call_filed": (filed or {}).get((sp, ds), calls.get((sp, ds), "WITHIN"))}
        for j, (sp, ds) in enumerate((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER)]
    rec = {"stage": stage, "declared_in": declared, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def fake_side(path, base, fc, hit):
    """A compare in lean_screen's form whose first base is base: each read's delta [R@5, FC@5, hit@1]."""
    rows = [{"dataset": ds, "delta": [0.0, fc, hit]} for ds in LG.EVAL_ORDER]
    LC.write_json(Path(path).with_suffix(".json"), {"new": "x", "bases": [base, "outputs/step1/fits/x"],
                                                   "by_base": {base: {"rows": rows}}})
    return path


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    must = P1GC.must_stop_with
    arm = ARM_NAMES[0]
    # 1. the arm is A3v's, the same object; its carve is the lean MLP's (P1g-b's zrckp's), so A3vp reads what P1g-b's
    #    fit reads; the model is P1g-b's zrckp with A3VMix and A3Mix just above zret; the MLP passes no message; webqsp
    #    is in no training split; the base is P1g-b's own card fit; A3v's GNN arm is not here
    assert ARM_NAMES == ("zrckpy",) and "zspky" not in ARMS and BASE_OF == {"zrckpy": "zrckp"}
    assert ARMS[arm] is A3V.ARMS[arm] and S.ARMS[arm] == A3V.ARMS[arm]
    assert ARMS[arm][1] is GB.ARMS["zrckp"][1] is S.ARMS["zrckp"][1] is Z3.KTChainCarveZRC
    assert GB.ARMS["zrckp"][0] is Z3P.ZRCKP
    assert [k for k in ARMS[arm][0].__mro__[1:] if k not in (A3V.A3VMix, ZA3.A3Mix)] == list(Z3P.ZRCKP.__mro__)
    assert ARMS[arm][0].__mro__[1:8] == (Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, A3V.A3VMix, ZA3.A3Mix, S3.ZRet)
    assert ZP.ZProp not in ARMS[arm][0].__mro__ and issubclass(ARMS[arm][1], ZC.ChainCarveZRC)
    assert BASE_FITS[arm] == {sp: ("screen", "fits", f"scr-p1gb-zrckp{SX[sp]}") for sp in SPLITS}
    assert all(not FIT[arm][sp].startswith(("scr-a3v-", "scr-p1gb-")) for sp in SPLITS)
    assert SPLITS == GB.SPLITS == P1GC.SPLITS and all("webqsp" not in LG.FITS[sp] for sp in SPLITS)
    assert not any(n.startswith("a3vp") for n in S.ARMS) and A3V.SETTINGS["parameters"] == 0
    assert A3V.SETTINGS["reads"].startswith("never reweighted") and not A3V.SETTINGS["score_input"]

    # 2. the roots: inside, P1g-b's (zp1gb.on_p1gb), the same that P1g-c's arms read inside zp1gc.on_p1gc; every root
    #    the arm's carve reads is a family P1g-c's gate checks (zrc's chains, KT, and the links and rmatch's chains the
    #    shared binding names); all restored after
    def roots_now():
        return {"cache": Z2.CACHE, "look": Z2.LOOK, "gate": Z2.GATE, "rmatch": RM.ChainCarveBase.CH_ROOT,
                "zrc": ZC.ChainCarveZRC.CH_ROOT, "links": ZL.LinkCarveBase.LK_ROOT, "lk_chains": ZL.LinkCarveBase.CH_ROOT,
                "kt_mlp": ARMS[arm][1].KT_ROOT, "kt_gnn": Z3.KTLinkCarveBase.KT_ROOT, "declared": Z3.DECLARED,
                "rm_built": RM.built_records.__defaults__, "zl_built": ZL.built_records.__defaults__}

    before = roots_now()
    with GB.on_p1gb():
        inside = roots_now()
    with P1GC.on_p1gc():
        theirs = roots_now()
    assert roots_now() == before and inside == theirs, (inside, theirs)
    assert inside == {"cache": GB.CACHE, "look": GB.LOOK, "gate": GB.GATE, "rmatch": GB.CH_RM, "zrc": GB.CH_ZRC,
                      "links": GB.LINKS, "lk_chains": GB.CH_RM, "kt_mlp": GB.KT_ROOT, "kt_gnn": GB.KT_ROOT,
                      "declared": GB.DECLARED, "rm_built": (GB.CH_RM,), "zl_built": (GB.LINKS,)}, inside
    assert {f: r for f, (r, _) in P1GC.RECS.items()} == {"links": inside["links"], "rmatch": inside["rmatch"],
                                                          "zrc": inside["zrc"], "kt": inside["kt_mlp"]}

    # 3. the arm starts as A3v's model bit for bit and fits as A3v's does: the class A3vp's train builds (S.ARMS) is
    #    A3v's; from the same seed, the same state and the same share draws (A3's generator); with zret's output layer
    #    and the KT head set, the same scores and every gradient in eval and in training, on a batch whose eligible
    #    pools sit on both sides of 0.375; toy fits through lean_gpu's loop inside P1g-b's roots equal the same fits
    #    outside them bit for bit (every state, swa and A3v's counts), with questions reweighted on both sides
    rng = np.random.default_rng(87)
    tmp = Path(tempfile.mkdtemp(prefix="za3vp_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")

        class ToyVIds(A3V.ToyV):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_ch = RM.chain_carve(ToyVIds, tmp / "ch", tmp / "rel")
        Z3.toy_kt(tmp / "kt", cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        cv = Z3.kt_carve(cls_ch, tmp / "kt")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(cv.rows)[::-1][:9], np.arange(cv.rows)[:7]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, len(blocks))
        keep_[3, 1] = 0.0
        f2, nq2, bz2, gold = cv.batch(qs, blocks)
        assert RM.CH_KEY in f2 and Z3.KT_KEY in f2
        r2, u2 = ZA3.counts(S3.retrieved(f2), nq2, B)
        elig = (r2 >= A3V.MIN_R) & (u2 >= 1)
        low = (r2.double() / (r2 + u2).double()) < A3V.MID
        assert bool((elig & low).any()) and bool((elig & ~low).any()), (r2.tolist(), u2.tolist())
        torch.manual_seed(0)
        m_a = A3V.ARMS[arm][0](blocks, cv.widths, 16, dropout=0.1, seed=0)
        r_a = torch.get_rng_state()
        torch.manual_seed(0)
        m_p = S.ARMS[arm][0](blocks, cv.widths, 16, dropout=0.1, seed=0)
        assert torch.equal(torch.get_rng_state(), r_a) and type(m_p) is type(m_a)
        assert not LG.same_state(m_a.state_dict(), m_p.state_dict())
        assert torch.equal(m_a.a3_gen.get_state(), m_p.a3_gen.get_state())
        with torch.no_grad():
            for m_ in (m_a, m_p):
                for p_ in m_.out.parameters():
                    p_.copy_(torch.linspace(-0.3, 0.3, p_.numel()).reshape(p_.shape))
                m_.kt_w2.copy_(torch.linspace(-0.5, 0.5, m_.kt_w2.numel()).reshape(m_.kt_w2.shape))
        for mode in ("eval", "train", "train"):
            got = []
            for m_ in (m_a, m_p):
                getattr(m_, mode)()
                m_.zero_grad()
                torch.manual_seed(3)
                y = m_(f2, keep_, nq2, B, bz2)
                LG.listwiseD(y, gold, nq2, B).backward()
                got.append((y.detach(), {n_: p_.grad for n_, p_ in m_.named_parameters()}))
            (ya, ga), (yp, gp) = got
            assert torch.equal(ya, yp), mode
            assert all((ga[n_] is None and gp[n_] is None) or torch.equal(ga[n_], gp[n_]) for n_ in ga), mode
            assert torch.equal(m_a.a3_gen.get_state(), m_p.a3_gen.get_state()), mode
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(arm):
            f_out = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} (A3v's)")
            n_out = {**A3V.LAST, **A3V.last_means()}
            with GB.on_p1gb():
                f_in = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, P1g-b's roots")
                n_in = {**A3V.LAST, **A3V.last_means()}
        assert n_in == n_out and n_in["applied_rich"] > 0 and n_in["applied_thin"] > 0, (n_in, n_out)
        assert n_in["applied"] == n_in["applied_rich"] + n_in["applied_thin"] <= n_in["eligible"] <= n_in["questions"]
        assert A3V.RICH[0] <= n_in["mean_t_rich"] <= A3V.RICH[1] and A3V.THIN[0] <= n_in["mean_t_thin"] <= A3V.THIN[1]
        assert not any(LG.same_state(a_, b_) for a_, b_ in zip(f_in["states"] + [f_in["swa"]],
                                                               f_out["states"] + [f_out["swa"]]))
        assert roots_now() == before
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        # 4. P1g-c's gate, on toy records of P1g-b's: train and read refuse while it is shut, after a record changed,
        #    when it is another stage's or missing; with it open they refuse a missing or failed P1g-b gate and KT
        #    records that are not P1g-b's; with all in place the fit and the read run on P1g-b's roots and the fit is
        #    stamped A3vp's (both gates, the snapshot, the records it read, its base fit, A3v's settings and counts)
        roots = P1GC.toy_records(T_ / "p1gb")
        sn = P1GC.snap(T_ / "snap", roots, P1GC.toy_stamps(T_, roots))
        g = T_ / "g"
        held = {}
        for f in P1GC.RECS:
            for rel in sn["records"][f]:
                held[(f, rel)] = json.loads((roots[f] / rel / "record.json").read_text(encoding="utf-8"))
                (roots[f] / rel / "record.json").unlink()
        cr = ["--cache-root", "outputs/p1gb/cache"]
        fr = ["--out-root", str(T_ / "fits")]
        good = ["train", "--arm", arm, "--name", FIT[arm]["L-musique"]] + cr + fr
        cuda = good + ["--device", "cuda"]
        assert P1GC.gate(g, T_ / "snap", roots) == 1
        must(train, "the gate is SHUT", cuda, "L-musique", g, roots)
        LC.write_json(T_ / "rf0" / FIT[arm]["L-musique"] / "screen.json", {"arm": arm, "a3vp": {"split": "L-musique"}})
        rd0 = ["--name", FIT[arm]["L-musique"], "--out-root", str(T_ / "rf0")] + cr + ["--device", "cuda"]
        must(read, "the gate is SHUT", rd0, g, roots)
        must(read, "missing (zp1gc.py gate)", rd0, T_ / "nogate", roots)
        for (f, rel), r in held.items():
            LC.write_json(roots[f] / rel / "record.json", dict(r, seconds=2.5, utc="t1"))
        assert P1GC.gate(g, T_ / "snap", roots) == 0
        LC.write_json(T_ / "other.json", dict(json.loads(g.with_suffix(".json").read_text(encoding="utf-8")),
                                              stage="P1gb"))
        must(train, "the gate is OPEN (P1gb)", cuda, "L-musique", T_ / "other", roots)
        must(train, "missing (zp1gc.py gate)", cuda, "L-musique", T_ / "nogate", roots)
        kt = T_ / "kt"
        for ds, cvn in set(Z3.train_carves("L-musique") + Z3.train_carves("L-hotpotqa") + Z3.eval_carves()):
            LC.write_json(kt / ds / cvn / "record.json", {"declared_in": GB.DECLARED})
        p1gb_gate = T_ / "p1gb_gate.json"
        seen = []

        def look():
            return (Z2.CACHE, RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT,
                    Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT, Z3.DECLARED)

        def fake_train(argv, split):
            name, out_root = S2.where(argv)
            LC.write_json(out_root / name / "screen.json", {"arm": R.arm_of(argv), "split": split})
            seen.append(("train", split, list(argv), look()))
            return 0

        def fake_read(argv):
            seen.append(("read", None, list(argv), look()))
            return 0

        saved = GB.GATE, GB.KT_ROOT, RM.train, RM.read
        GB.GATE, GB.KT_ROOT, RM.train, RM.read = p1gb_gate, kt, fake_train, fake_read
        try:
            must(train, "missing (scripts/p1gb_pools.py gate)", cuda, "L-musique", g, roots)
            LC.write_json(p1gb_gate, {"verdict": "FAIL", "declared_in": GB.DECLARED})
            must(train, "not P1g-b's PASS", cuda, "L-musique", g, roots)
            LC.write_json(p1gb_gate, {"verdict": "PASS", "declared_in": GB.DECLARED, "construction": ["toy"]})
            LC.write_json(kt / "squad" / "fit" / "record.json", {"declared_in": Z3.DECLARED})   # P1c2's pools' KT
            must(train, "is not P1c3's", cuda, "L-musique", g, roots)
            LC.write_json(kt / "squad" / "fit" / "record.json", {"declared_in": GB.DECLARED})
            assert not seen
            assert train(cuda, "L-musique", g, roots) == 0
            want = (GB.CACHE, GB.CH_RM, GB.CH_ZRC, GB.LINKS, kt, kt, GB.DECLARED)
            assert seen == [("train", "L-musique", cuda, want)], seen
            st = json.loads((T_ / "fits" / FIT[arm]["L-musique"] / "screen.json").read_text(
                encoding="utf-8"))["a3vp"]
            assert st["declared_in"] == DECLARED and (st["arm"], st["base_arm"], st["split"]) == (
                arm, "zrckp", "L-musique") and st["base_fit"] == "scr-p1gb-zrckp" and st["pools"] == POOLS
            assert st["p1gc_gate"] == LC.sha_file(g.with_suffix(".json")) and st["p1gb_gate"] == LC.sha_file(p1gb_gate)
            assert st["snap"] == LC.sha_file(T_ / "snap" / "snap.json") and st["cache_root"] == "outputs/p1gb/cache"
            assert sorted(st["kt_records"]) == sorted(f"{d}/{c}" for d, c in Z3.train_carves("L-musique"))
            assert st["settings"] == A3V.SETTINGS
            assert sorted(st["last_fit"]) == sorted(list(A3V.LAST) + ["mean_t", "mean_t_rich", "mean_t_thin"])
            rd = ["--name", FIT[arm]["L-musique"]] + fr + cr + ["--device", "cuda"]
            assert read(rd, g, roots) == 0 and seen[-1] == ("read", None, rd, want), seen[-1]
            must(read, "the gate is OPEN (P1gb)", rd, T_ / "other", roots)
            must(read, "take --cache-root outputs/p1gb/cache", rd[:-4] + ["--cache-root", "outputs/p1c2/cache"], g,
                 roots)
            r0 = held[("links", "metaqa/fit/part_1of2")]
            LC.write_json(roots["links"] / "metaqa/fit/part_1of2" / "record.json", dict(r0, utc="t2"))
            must(train, "changed after the gate opened", cuda, "L-musique", g, roots)
            must(read, "changed after the gate opened", rd, g, roots)
            assert len(seen) == 2
        finally:
            GB.GATE, GB.KT_ROOT, RM.train, RM.read = saved
        assert roots_now() == before
        # 5. the other refusals, each for its own reason, before any gate or data: A3v's adopted arm only (not A3v's
        #    GNN arm, the lean pair's, A5c's, nor P1g-c's adopted arms), the two splits, A3vp's fit names, P1g-b's
        #    caches, the card; read refuses A3v's, P1g-b's and the wrong split's fits
        for bad in ("zspky", "zrckp", "zspk", "zrckpr", "zspx", "zspxr", "zrckpc", "zrckpv", "zrckpu"):
            must(train, "train takes --arm", ["train", "--arm", bad, "--name", FIT[arm]["L-musique"]] + cr,
                 "L-musique", g, roots)
        must(train, "not L-2wiki", good, "L-2wiki", g, roots)
        must(train, "zrckpy's fit of L-hotpotqa is scr-a3vp-zrckpy-hp", good, "L-hotpotqa", g, roots)
        must(train, "zrckpy's fit of L-musique is scr-a3vp-zrckpy",
             ["train", "--arm", arm, "--name", A3V.FIT[arm]["L-musique"]] + cr, "L-musique", g, roots)
        must(train, "take --cache-root outputs/p1gb/cache", good[:5] + ["--cache-root", "outputs/p1c2/cache"],
             "L-musique", g, roots)
        must(train, "take --cache-root outputs/p1gb/cache", good[:5], "L-musique", g, roots)
        must(train, "not cpu", good, "L-musique", g, roots)
        must(train, "not cpu", good + ["--device", "cpu"], "L-musique", g, roots)
        for nm, rec_ in (("scr-a3v-zrckpy", {"arm": arm, "a3v": {"split": "L-musique"}}),
                         ("scr-a3vp-zrckpy", {"arm": arm, "a3v": {"split": "L-musique"}}),
                         ("scr-a3vp-zrckpy", {"arm": "zrckp", "p1gb": {}}),
                         ("scr-a3vp-zrckpy", {"arm": "zspky", "a3vp": {"split": "L-musique"}}),
                         ("scr-a3vp-zrckpy-hp", {"arm": arm, "a3vp": {"split": "L-musique"}})):
            LC.write_json(T_ / "rf" / nm / "screen.json", rec_)
            must(read, "is not an A3vp fit" if nm != "scr-a3vp-zrckpy-hp" else "is L-hotpotqa's fit",
                 ["--name", nm, "--out-root", str(T_ / "rf")] + cr + ["--device", "cuda"], g, roots)
        assert split_of_fit("scr-a3vp-zrckpy-hp") == (arm, "L-hotpotqa") and split_of_fit("scr-a3v-zrckpy") == (
            None, None)
        # 6. relz's pair and re-call against P1g-b's fits of the base arm, stamped A3vp's; a screen decided against
        #    A3v's base (P1c2's pools) and base screens of A3v's base are refused; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        fits, d = BASE_FITS[arm], T_ / "calls" / arm
        null1 = SR.fake_null(d / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        bs = {sp: str(SR.fake_compare(d / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_OF[arm])) for sp in fits}
        old = A3V.BASE_FITS[arm]
        bs_old = {sp: str(SR.fake_compare(d / "old" / old[sp][0], old[sp][2], sp, {}, arm=BASE_OF[arm])) for sp in old}
        zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
        za = SR.fake_compare(d / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(d / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                             base_r5=0.6, base=zb["L-hotpotqa"])
        zo = SR.fake_compare(d / "zo", FIT[arm]["L-musique"], "L-musique", {}, arm=arm, base_r5=0.6,
                             base="/".join(("outputs",) + old["L-musique"]))
        must(pair, "is decided against", arm, [zo, zh], d / "bad")
        pr = pair(arm, [za, zh], d / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
        pj = json.loads((d / "pz.json").read_text(encoding="utf-8"))
        assert pj["stage"] == "A3vp" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
        assert pj["pools"] == POOLS and "round" not in pj
        must(recall, "with step 1's", arm, null1, d / "pz.json", d / "rbad", bs_old)
        r = recall(arm, null1, d / "pz.json", d / "rz", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (d / "rz.md").read_text(encoding="utf-8")
        assert t.startswith(f"A3vp ({DECLARED})") and \
            f"| {BASE_OF[arm]} R@5 (P1g-b's) | {arm} R@5 (A3v's rule, P1g-b's) |" in t
        assert json.loads((d / "rz.json").read_text(encoding="utf-8"))["stage"] == "A3vp"
        must(pair, "--arm takes", "zspky", [], T_ / "x")
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, with section 5's next forms; beside: metaqa, musique, squad and webqsp, musique in domain,
        #    FC@5 and hit@1 from the compares, A3v's delta and call on P1c2's pools, each fit's reweighting counts from
        #    its stamp; F1's note
        gm, gw, mi = ("L-musique", "musique"), ("L-musique", "webqsp"), CLAIM1_READ
        cmp_ = {arm: {sp: fake_side(T_ / "cmp" / f"{arm}{SX[sp]}", "/".join(("outputs",) + BASE_FITS[arm][sp]),
                                    0.01 if sp == "L-musique" else -0.02, 0.03) for sp in SPLITS}}
        old = {arm: fake_recall(T_ / f"old-{arm}.json", arm, {gw: "GAIN"}, stage="A3v", declared=A3V.DECLARED)}
        fits_root = T_ / "fits"                         # step 4's stamped toy fit of zrckpy on L-musique, only

        def v(c_, over=None, **kw):
            rc = {arm: fake_recall(T_ / "m.json", arm, c_, **(over or {}))}
            return verdict(kw.get("recalls", rc), cmp_, old, T_ / "v", fits_root)

        rec = v({gw: "GAIN"}, over={"filed": {gm: "LOSS"}})
        am = rec["arms"][arm]
        assert am["verdict"] == "ADOPT_FOR_FULL_RUN" and am["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert am["reads"]["L-musique musique"]["call_filed"] == "LOSS"
        md = (T_ / "v.md").read_text(encoding="utf-8")
        assert "| L-musique | musique | x | +0.0020 [+0.0015, +0.0025] | WITHIN (filed LOSS) |" in md, md
        assert am["training_counts"] == {"L-musique": st["last_fit"], "L-hotpotqa": None}
        assert f"L-musique {st['last_fit']['applied']:,} of {st['last_fit']['questions']:,} question visits" in md
        assert "L-hotpotqa not stamped" in md and rec["f1"].startswith("zrckpy adopted")
        assert sorted(am["beside"]) == sorted(f"{sp} {ds}" for sp in SPLITS for ds in BESIDE)
        assert rec["musique_in_domain_beside"]["call"] == "WITHIN"
        x = am["reads"]["L-hotpotqa webqsp"]
        assert (x["fc5"], x["hit1"]) == (-0.02, 0.03) and am["reads"]["L-musique metaqa"]["fc5"] == 0.01
        assert x["a3v_p1c2"] == {"call": "WITHIN", "delta": 0.011} and am["reads"]["L-musique webqsp"]["a3v_p1c2"] == {
            "call": "GAIN", "delta": 0.005}
        assert not am["a3v_recall_problems"]
        assert "closes only this form" in md and "| L-musique | webqsp |" in md and "P1g-c's call" in md
        rec = v({gw: "GAIN", mi: "LOSS"})
        am = rec["arms"][arm]
        assert am["verdict"] == "NOT_ADOPTED" and am["outcome"] == "GAIN_AND_LOSS" and "A3vs" in am["next_form"]
        assert rec["musique_in_domain_beside"]["call"] == "LOSS" and rec["f1"].startswith("zrckpy not adopted")
        rec = v({})
        am = rec["arms"][arm]
        assert am["verdict"] == "NOT_ADOPTED" and am["outcome"] == "NO_GAIN" and am["next_form"] == NEXT["NO_GAIN"]
        assert "A3z" in NEXT["NO_GAIN"] and "A3vs" in NEXT["GAIN_AND_LOSS"]
        for over in ({"stage": "A3v"}, {"declared_in": A3V.DECLARED}):
            rec = v({gw: "GAIN"}, over=over)
            assert rec["arms"][arm]["verdict"] == "INCOMPLETE", over
            assert "is not A3vp's re-call" in rec["arms"][arm]["problems"][0], over
            assert rec["arms"][arm]["next_form"] == NEXT["INCOMPLETE"] and rec["f1"].startswith("not settled")
        rec = verdict({arm: fake_recall(T_ / "g.json", "zspky", {gw: "GAIN"})}, cmp_, old, T_ / "v", fits_root)
        assert rec["arms"][arm]["verdict"] == "INCOMPLETE" and "is not A3vp's re-call" in rec["arms"][arm]["problems"][0]
        rec = verdict({arm: T_ / "none.json"}, cmp_, old, T_ / "v", fits_root)
        assert rec["arms"][arm]["verdict"] == "INCOMPLETE" and "missing" in rec["arms"][arm]["problems"][0]
        short = fake_recall(T_ / "s.json", arm, {gw: "GAIN"})
        sj = json.loads(short.read_text(encoding="utf-8"))
        LC.write_json(short, dict(sj, rows=sj["rows"][:11]))
        rec = verdict({arm: short}, cmp_, old, T_ / "v", fits_root)
        assert rec["arms"][arm]["verdict"] == "INCOMPLETE" and "11 reads" in rec["arms"][arm]["problems"][0]
        rec = verdict({arm: fake_recall(T_ / "m.json", arm, {gw: "GAIN"})}, cmp_, {arm: T_ / "none.json"}, T_ / "v",
                      fits_root)
        assert rec["arms"][arm]["verdict"] == "ADOPT_FOR_FULL_RUN" and rec["arms"][arm]["a3v_recall_problems"]
    log(f"za3vp selftest: the arm is A3v's object on the lean MLP's carve, P1g-b's zrckp with A3VMix and A3Mix just "
        f"above zret, without a message; P1g-b's roots inside, the same P1g-c's arms read and each a family P1g-c's "
        f"gate checks, all restored after; the arm starts as A3v's model bit for bit (state, share draws, scores and "
        f"every gradient, eval and training) and its toy fits inside P1g-b's roots equal A3v's bit for bit, with "
        f"questions reweighted on both sides of {A3V.MID}; train and read refuse a shut, changed, foreign or missing "
        f"P1g-c gate, a missing or failed P1g-b gate, KT records not P1g-b's, P1c2's caches, the CPU, another arm, "
        f"split or fit name, and A3v's, P1g-b's or the wrong split's fits; with all in place they run on P1g-b's roots "
        f"and the fit is stamped A3vp's; relz's pair and re-call against P1g-b's fit, stamped A3vp's, refusing A3v's "
        f"base; the verdict's outcomes, next forms, beside (with each fit's stamped counts) and F1 note "
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
            log(f"za3vp {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za3vp: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
