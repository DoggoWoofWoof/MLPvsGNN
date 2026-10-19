"""A5cp (docs/A5CP_P1GB_POOLS.md): A5c's adopted arms fitted again on P1g-b's pools, the next form A5c's verdict named
for both arms (outputs/a5c/screen-verdict.json). A5c (za5c.py) gave the lean pair zck's kind offsets in their coupled
form and passed on P1c2's pools; no model change is adopted on today's pools (the user's rule of 13 October), so each
arm is fitted again on P1g-b's pools and decided against P1g-b's own card fit of its base arm there:

    zrckpc  A5c's MLP as filed (za5c.ZRCKPC on zp1c3.KTChainCarveZRC), against P1g-b's scr-p1gb-zrckp, -hp
    zspkc   A5c's GNN as filed (za5c.ZSPKC on zp1c3.KTLinkCarveBase), against P1g-b's scr-p1gb-zspk, -hp

The arms are za5c's own objects and their carves are the lean pair's, so A5cp builds nothing: it reads P1g-b's caches,
KT columns, links and chains (zrc's for the MLP, rmatch's for the GNN: the typed graphs' chain entries the offsets act
on) on P1g-b's roots (zp1gb.on_p1gb), as P1g-c's arms read them, behind P1g-c's gate (zp1gc.gate_open). Every fit and
read refuses while that gate is shut, missing or another stage's, and refuses a P1g-b record rebuilt after it opened.

    python outputs/mp_unified/za5cp.py compare --new outputs/screen/fits/scr-p1gb-zrckp \\
        --base outputs/step1/fits/L-musique --out outputs/a5cp/base-zrckp-L-musique
    python outputs/mp_unified/za5cp.py train --split L-musique --name scr-a5cp-zrckpc --arm zrckpc --device cuda \\
        --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za5cp.py read --name scr-a5cp-zrckpc --device cuda --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za5cp.py compare --new outputs/screen/fits/scr-a5cp-zrckpc \\
        --base outputs/screen/fits/scr-p1gb-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a5cp-zrckpc
    python outputs/mp_unified/za5cp.py pair --arm zrckpc --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za5cp.py recall --arm zrckpc --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za5cp.py verdict
    python outputs/mp_unified/za5cp.py --selftest
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

import za5c as A5C  # noqa: E402  (A5c's arms, offsets, settings; through it zp1c3, zp1c3p, lean_screen3, zkind)
import zp1gb as GB  # noqa: E402  (P1g-b's roots, gate and lean-pair fits)
import zp1gc as P1GC  # noqa: E402  (P1g-c's gate on P1g-b's records)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Z3, Z3P, Z2 = A5C.Z3, A5C.Z3P, A5C.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
S3, ZK = A5C.S3, A5C.ZK
log = S.log

DECLARED = "docs/A5CP_P1GB_POOLS.md"
OUT = ROOT / "outputs" / "a5cp"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = A5C.SPLITS
SX = {"L-musique": "", "L-hotpotqa": "-hp"}
ARM_NAMES = A5C.ARM_NAMES
ARMS = {a: A5C.ARMS[a] for a in ARM_NAMES}                  # A5c's (model, carve), the same objects
BASE_OF = dict(A5C.BASE_OF)                                   # zrckpc -> zrckp, zspkc -> zspk
FIT = {a: {sp: f"scr-a5cp-{a}{SX[sp]}" for sp in SPLITS} for a in ARM_NAMES}
BASE_FITS = {a: {sp: ("screen", "fits", GB.FIT[BASE_OF[a]][sp]) for sp in SPLITS} for a in ARM_NAMES}
BASE_SCREENS = {a: {sp: f"outputs/a5cp/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split on P1g-b's pools (P1g-b's {GB.FIT[BASE_OF[a]]['L-musique']}, "
              f"{GB.FIT[BASE_OF[a]]['L-hotpotqa']})" for a in ARM_NAMES}
WHAT = {a: A5C.WHAT[a] for a in ARM_NAMES}
PAIR = {a: SCREEN / f"scr-a5cp-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a5cp-{a}-pair-recall" for a in ARM_NAMES}
COMPARE = {a: {sp: SCREEN / f"scr-a5cp-{a}{SX[sp]}" for sp in SPLITS} for a in ARM_NAMES}
A5C_RECALL = {a: A5C.RECALL[a].with_suffix(".json") for a in ARM_NAMES}      # A5c's re-calls on P1c2's pools, beside
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
CLAIM1_READ = A5C.CLAIM1_READ                                 # ("L-hotpotqa", "musique")
POOLS = "P1g-b's (outputs/p1gb/pools)"
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A5cp's re-call of the arm, or "
        "not twelve reads (docs/A5CP_P1GB_POOLS.md, section 4; A5c's rule)")
NEXT = {"ADOPT_FOR_FULL_RUN": "A5c's coupled offsets join the full run's models when it runs on P1g-b's pools: one "
                              "file, declared before its numbers, after P1g-c's, A5cp's and A7p's verdicts, with the "
                              "composition P1g-c's section 5 sets",
        "GAIN_AND_LOSS": "A5cb on P1g-b's pools: the coupled offsets bounded (each offset 0.2 tanh of its parameter), "
                         "its own declared file",
        "NO_GAIN": "A5i on P1g-b's pools: the graph's kind as a 0/1 input through R1's pointwise head (the learned "
                   "back-off), on both models, its own declared file; the idea stays open (it gained on P1c2's pools; "
                   "docs/SCREENS.md, 3b)",
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
        rec.update({"za5cp_sha256": LC.sha_src(__file__), "za5c_sha256": LC.sha_src(A5C.__file__),
                    "zp1gc_sha256": LC.sha_src(P1GC.__file__), "zp1gb_sha256": LC.sha_src(GB.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "zkind_sha256": LC.sha_src(ZK.__file__),
                    "a5cp": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                             "base_fit": GB.FIT[BASE_OF[arm]][split], "device": Z2.DEVICE, "settings": A5C.SETTINGS,
                             "last_fit": dict(A5C.LAST), "pools": POOLS,
                             "cache_root": GB.CACHE.relative_to(ROOT).as_posix(), "p1gb_gate": LC.sha_file(GB.GATE),
                             "p1gc_gate": LC.sha_file(Path(gate_file or P1GC.GATE).with_suffix(".json")),
                             "snap": g.get("snap_sha256"),
                             "kt_records": Z3.kt_records(Z3.train_carves(split), GB.KT_ROOT),
                             "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(GB.CH_ZRC),
                             "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)
        log(f"  a5cp {arm} {split}: {A5C.LAST['typed_forwards']} typed and {A5C.LAST['untyped_forwards']} untyped "
            f"training forwards in the last fit")


def train(argv, split, gate_file=None, roots=None):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za5cp: train takes --arm {' or '.join(ARM_NAMES)} (A5c's arms as filed), not {arm} "
                         f"({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za5cp: A5cp fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za5cp: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a5cp" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za5cp: {name} is not an A5cp fit (arm {rec.get('arm')})")
    if rec["a5cp"].get("split") != split:
        raise SystemExit(f"za5cp: {name} is {split}'s fit, its record says {rec['a5cp'].get('split')}")
    with GB.on_p1gb():
        Z2.cache_root_of(argv)
        P1GC.gate_open(gate_file, roots)
        Z3.kt_records(Z3.eval_carves(), GB.KT_ROOT)
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against P1g-b's fit of its base arm ──────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za5cp: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za5cp: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 (P1g-b's) | {arm} R@5 (A5c's offsets, P1g-b's) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A5cp ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}, both on P1g-b's pools.\n\n" + t,
                      encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A5cp", "declared_in": DECLARED, "decided_against": AGAINST[arm], "pools": POOLS,
                    "device": Z2.DEVICE, "za5cp_sha256": LC.sha_src(__file__)})
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


def forwards_of(arm, split, fits_root=None):
    """The typed and untyped training forwards A5cp's fit stamped (A5c's LAST at the fit's end), or None."""
    sj = Path(fits_root or SCREEN / "fits") / FIT[arm][split] / "screen.json"
    if not sj.exists():
        return None
    return (json.loads(sj.read_text(encoding="utf-8")).get("a5cp") or {}).get("last_fit")


def verdict(recalls=None, compares=None, a5c=None, out=None, fits_root=None):
    """A5cp's verdict, per arm, from its re-call (section 4), with the next form each outcome names (section 5); beside,
    deciding nothing: webqsp and metaqa, musique in domain (Claim 1's read) of both arms, each read's FC@5 and hit@1
    from the compares, each read's delta beside A5c's on P1c2's pools, and each fit's typed and untyped forwards."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    compares = compares or COMPARE
    a5c = a5c or A5C_RECALL
    per = {}
    for a in ARM_NAMES:
        rows, problems = recall_rows(recalls[a], a, "A5cp", DECLARED)
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        oc = v if v != "NOT_ADOPTED" else ("GAIN_AND_LOSS" if gains else "NO_GAIN")
        old_rows, old_problems = recall_rows(a5c[a], a, "A5c", A5C.DECLARED)
        old = {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in old_rows}
        side = {sp: side_of(compares[a][sp]) for sp in SPLITS}
        reads = {}
        for r in rows:
            d = side.get(r.get("split"), {}).get(r.get("dataset"))
            reads[tag(r)] = {"read": r.get("read"), "call": r.get("call"), "call_filed": r.get("call_filed"),
                             "base": r.get("base"), "new": r.get("new"), "delta": r.get("delta"), "lo": r.get("lo"),
                             "hi": r.get("hi"), "fc5": d[1] if d else None, "hit1": d[2] if d else None,
                             "a5c_p1c2": old.get(tag(r))}
        per[a] = {"verdict": v, "outcome": oc, "gains": gains, "losses": losses, "problems": problems,
                  "next_form": NEXT[oc], "reads": reads,
                  "beside": {t: x for t, x in reads.items() if t.split()[1] in BESIDE},
                  "training_forwards": {sp: forwards_of(a, sp, fits_root) for sp in SPLITS},
                  "a5c_recall_problems": old_problems,
                  "recall_sha256": LC.sha_file(recalls[a]) if Path(recalls[a]).exists() else None,
                  "a5c_recall_sha256": LC.sha_file(a5c[a]) if Path(a5c[a]).exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: per[a]["reads"].get(ck) for a in ARM_NAMES}
    vs = {a: per[a]["verdict"] for a in ARM_NAMES}
    adopted = [a for a in ARM_NAMES if vs[a] == "ADOPT_FOR_FULL_RUN"]
    if "INCOMPLETE" in vs.values():
        f1 = "not settled while a re-call is INCOMPLETE"
    elif len(adopted) == len(ARM_NAMES):
        f1 = "both arms adopted: the offsets join both models, and F1's one input set holds"
    elif adopted:
        f1 = (f"only {adopted[0]} adopted: the full run's file names the offsets as a difference between its two "
              f"models (F1)")
    else:
        f1 = "neither arm adopted: the full run carries no offsets"
    rec = {"stage": "A5cp", "declared_in": DECLARED, "rule": RULE, "pools": POOLS, "arms": per,
           "musique_in_domain_beside": side, "f1": f1, "za5cp_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"  # noqa: E731
    md = [f"# A5cp's screen verdict: zrckpc **{vs['zrckpc']}**, zspkc **{vs['zspkc']}**", "",
          f"Rule: {RULE}.", ""]
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{t} {b['call']} ({num(b['delta'])})" for t, b in p["beside"].items()) or "none")
                  + f". Next: {p['next_form']}.")
    md += ["", f"F1: {f1}.", ""]
    for a in ARM_NAMES:
        md += [f"{a}, each read beside, deciding nothing (FC@5 and hit@1 from `{Path(compares[a]['L-musique']).name}` "
               f"and its -hp; A5c's delta and call on P1c2's pools from `{Path(a5c[a]).name}`):", "",
               "| split | dataset | read | delta R@5 [95% CI] | call | FC@5 | hit@1 | A5c on P1c2's pools |",
               "| --- | --- | --- | --- | --- | ---: | ---: | --- |"]
        for t, x in per[a]["reads"].items():
            o = x["a5c_p1c2"] or {}
            ci = f" [{num(x['lo'])}, {num(x['hi'])}]" if x["lo"] is not None and x["hi"] is not None else ""
            filed = f" (filed {x['call_filed']})" if x["call_filed"] not in (None, x["call"]) else ""
            md.append(f"| {t.split()[0]} | {t.split()[1]} | {x['read']} | {num(x['delta'])}{ci} | {x['call']}{filed} | "
                      f"{num(x['fc5'])} | {num(x['hit1'])} | {num(o.get('delta'))} {o.get('call') or ''} |")
        fw = per[a]["training_forwards"]
        md += ["", f"{a}'s training forwards (typed / untyped, stamped by each fit): " + "; ".join(
            f"{sp} {f['typed_forwards']:,} / {f['untyped_forwards']:,}" if f else f"{sp} not stamped" for sp, f in
            fw.items()) + ".", ""]
    md += [f"Beside, deciding nothing: musique in domain ({ck}, Claim 1's read): "
           + ", ".join(f"{a} {(side[a] or {}).get('call')} ({num((side[a] or {}).get('delta'))})" for a in ARM_NAMES)
           + ".", "",
           "This verdict closes only this form (A5c's coupled offsets on P1g-b's pools); docs/SCREENS.md, 3b. Which "
           "pools the full run uses is P1g-c's call (its section 5): on P1g-b's pools this verdict decides whether the "
           "offsets join, per arm; on P1c2's pools A5c's own verdict decides, and this one is reported beside."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za5cp verdict: " + ", ".join(f"{a} {vs[a]}" for a in ARM_NAMES) + f"; F1: {f1}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, stage="A5cp", declared=DECLARED, rows=None, filed=None, **over):
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
    # 1. the arms are A5c's, the same objects; their carves are the lean pair's (P1g-b's arms'), so A5cp reads what
    #    P1g-b's fits read; each model is its base arm's with A5CRet just above zret; the MLP passes no message; webqsp
    #    is in no training split; the bases are P1g-b's own card fits
    for a in ARM_NAMES:
        assert ARMS[a] is A5C.ARMS[a] and S.ARMS[a] == A5C.ARMS[a], a
        assert ARMS[a][1] is GB.ARMS[BASE_OF[a]][1] is S.ARMS[BASE_OF[a]][1], a
        assert [k for k in ARMS[a][0].__mro__[1:] if k is not A5C.A5CRet] == list(GB.ARMS[BASE_OF[a]][0].__mro__), a
        assert BASE_FITS[a] == {sp: ("screen", "fits", f"scr-p1gb-{BASE_OF[a]}{SX[sp]}") for sp in SPLITS}
        assert all(not FIT[a][sp].startswith(("scr-a5c-", "scr-p1gb-")) for sp in SPLITS)
    assert ZP.ZProp not in ARMS["zrckpc"][0].__mro__ and ZP.ZProp in ARMS["zspkc"][0].__mro__
    assert issubclass(ARMS["zrckpc"][1], ZC.ChainCarveZRC) and issubclass(ARMS["zspkc"][1], ZL.LinkCarveBase)
    assert issubclass(ZL.LinkCarveBase, RM.ChainCarveBase) and RM.TYPED == ("metaqa", "webqsp")
    assert SPLITS == GB.SPLITS == P1GC.SPLITS and all("webqsp" not in LG.FITS[sp] for sp in SPLITS)
    assert not any(n.startswith("a5cp") for n in S.ARMS)

    # 2. the roots: inside, P1g-b's (zp1gb.on_p1gb), the same that P1g-c's arms read inside zp1gc.on_p1gc; every root
    #    A5c's carves read is a family P1g-c's gate checks (links, rmatch's and zrc's chains, KT); all restored after
    def roots_now():
        return {"cache": Z2.CACHE, "look": Z2.LOOK, "gate": Z2.GATE, "rmatch": RM.ChainCarveBase.CH_ROOT,
                "zrc": ZC.ChainCarveZRC.CH_ROOT, "links": ZL.LinkCarveBase.LK_ROOT, "lk_chains": ZL.LinkCarveBase.CH_ROOT,
                "kt_mlp": ARMS["zrckpc"][1].KT_ROOT, "kt_gnn": ARMS["zspkc"][1].KT_ROOT, "declared": Z3.DECLARED,
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

    # 3. each arm starts as A5c's model bit for bit and fits as A5c's does: the class A5cp's train builds (S.ARMS) is
    #    A5c's; from the same seed, the same state and the same draws; with the offsets and the KT head set, the same
    #    scores and every gradient on a typed batch and on its untyped copy, in eval and in training; toy fits through
    #    lean_gpu's loop inside P1g-b's roots equal the same fits outside them bit for bit (every state, swa and the
    #    typed and untyped forward counts); the offsets move on the typed graph and stay at zero on its untyped copy
    rng = np.random.default_rng(86)
    tmp = Path(tempfile.mkdtemp(prefix="za5cp_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_ch = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        c_m = Z3.kt_carve(cls_ch, tmp / "kt")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_g = Z3.kt_carve(cls_lk, tmp / "kt")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, len(blocks))
        keep_[3, 1] = 0.0
        for arm, cv in (("zrckpc", c_m), ("zspkc", c_g)):
            bt, bu = cv.batch(qs, blocks), ZK.Untyped(cv, "2wiki").batch(qs, blocks)
            assert RM.CH_KEY in bt[0] and RM.CH_KEY not in bu[0] and Z3.KT_KEY in bt[0] and Z3.KT_KEY in bu[0]
            torch.manual_seed(0)
            m_a = A5C.ARMS[arm][0](blocks, cv.widths, 16, dropout=0.1, seed=0)
            r_a = torch.get_rng_state()
            torch.manual_seed(0)
            m_p = S.ARMS[arm][0](blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r_a) and type(m_p) is type(m_a)
            s_a, s_p = m_a.state_dict(), m_p.state_dict()
            assert not LG.same_state(s_a, s_p) and all(float(s_p[k].abs().max()) == 0 for k in A5C.KD)
            with torch.no_grad():
                for m_ in (m_a, m_p):
                    m_.kd_w.copy_(torch.linspace(-0.5, 0.4, m_.kd_w.numel()).reshape(m_.kd_w.shape))
                    m_.kd_b.fill_(0.25)
                    m_.kd_base.fill_(-0.6)
                    m_.kt_w2.copy_(torch.linspace(-0.5, 0.5, m_.kt_w2.numel()).reshape(m_.kt_w2.shape))
            for mode in ("eval", "train"):
                for f2, nq2, bz2, gold in (bt, bu):
                    got = []
                    for m_ in (m_a, m_p):
                        getattr(m_, mode)()
                        m_.zero_grad()
                        torch.manual_seed(3)
                        y = m_(f2, keep_, nq2, B, bz2)
                        LG.listwiseD(y, gold, nq2, B).backward()
                        got.append((y.detach(), {n_: p_.grad for n_, p_ in m_.named_parameters()}))
                    (ya, ga), (yp, gp) = got
                    assert torch.equal(ya, yp), (arm, mode, RM.CH_KEY in f2)
                    assert all((ga[n_] is None and gp[n_] is None) or torch.equal(ga[n_], gp[n_]) for n_ in ga)
                    if RM.CH_KEY in f2:
                        assert float(gp["kd_w"].abs().sum()) > 0, (arm, mode)
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckpc", c_m), ("zspkc", c_g)):
            cu = ZK.Untyped(cv, "2wiki")
            with S.patched(arm):
                f_out = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} (A5c's)")
                n_out = dict(A5C.LAST)
                with GB.on_p1gb():
                    f_in = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, P1g-b's roots")
                    n_in = dict(A5C.LAST)
                    f_un = LG.fit_variant([cu], blocks, cfg, 0, 16, "none", "cpu",
                                          tag=f"toy {arm} untyped, P1g-b's roots")
                    n_un = dict(A5C.LAST)
            assert n_in == n_out and n_in["typed_forwards"] > 0 and n_in["untyped_forwards"] == 0, (n_in, n_out)
            assert n_un["typed_forwards"] == 0 and n_un["untyped_forwards"] > 0, n_un
            assert not any(LG.same_state(a_, b_) for a_, b_ in zip(f_in["states"] + [f_in["swa"]],
                                                                   f_out["states"] + [f_out["swa"]]))
            assert float(f_in["swa"]["kd_w"].abs().sum()) > 0 and float(f_in["swa"]["kd_base"].abs().sum()) > 0
            assert all(float(s_[k].abs().max()) == 0 for s_ in f_un["states"] + [f_un["swa"]] for k in A5C.KD)
        assert roots_now() == before
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        # 4. P1g-c's gate, on toy records of P1g-b's: train and read refuse while it is shut, after a record changed,
        #    when it is another stage's or missing; with it open they refuse a missing or failed P1g-b gate and KT
        #    records that are not P1g-b's; with all in place the fit and the read run on P1g-b's roots and the fit is
        #    stamped A5cp's (both gates, the snapshot, the records it read, its base fit)
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
        good = ["train", "--arm", "zrckpc", "--name", FIT["zrckpc"]["L-musique"]] + cr + fr
        cuda = good + ["--device", "cuda"]
        assert P1GC.gate(g, T_ / "snap", roots) == 1
        must(train, "the gate is SHUT", cuda, "L-musique", g, roots)
        LC.write_json(T_ / "rf0" / FIT["zrckpc"]["L-musique"] / "screen.json",
                      {"arm": "zrckpc", "a5cp": {"split": "L-musique"}})
        rd0 = ["--name", FIT["zrckpc"]["L-musique"], "--out-root", str(T_ / "rf0")] + cr + ["--device", "cuda"]
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
        for ds, cv in set(Z3.train_carves("L-musique") + Z3.train_carves("L-hotpotqa") + Z3.eval_carves()):
            LC.write_json(kt / ds / cv / "record.json", {"declared_in": GB.DECLARED})
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
            st = json.loads((T_ / "fits" / FIT["zrckpc"]["L-musique"] / "screen.json").read_text(
                encoding="utf-8"))["a5cp"]
            assert st["declared_in"] == DECLARED and (st["arm"], st["base_arm"], st["split"]) == (
                "zrckpc", "zrckp", "L-musique") and st["base_fit"] == "scr-p1gb-zrckp" and st["pools"] == POOLS
            assert st["p1gc_gate"] == LC.sha_file(g.with_suffix(".json")) and st["p1gb_gate"] == LC.sha_file(p1gb_gate)
            assert st["snap"] == LC.sha_file(T_ / "snap" / "snap.json") and st["cache_root"] == "outputs/p1gb/cache"
            assert sorted(st["kt_records"]) == sorted(f"{d}/{c}" for d, c in Z3.train_carves("L-musique"))
            assert st["settings"] == A5C.SETTINGS and sorted(st["last_fit"]) == sorted(A5C.LAST)
            rd = ["--name", FIT["zrckpc"]["L-musique"]] + fr + cr + ["--device", "cuda"]
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
        # 5. the other refusals, each for its own reason, before any gate or data: A5c's arms only (not the lean pair's,
        #    nor P1g-c's adopted arms), the two splits, A5cp's fit names, P1g-b's caches, the card; read refuses A5c's,
        #    P1g-b's and the wrong split's fits
        for bad in ("zrckp", "zspk", "zrckpr", "zspx", "zspxr"):
            must(train, "train takes --arm", ["train", "--arm", bad, "--name", FIT["zrckpc"]["L-musique"]] + cr,
                 "L-musique", g, roots)
        must(train, "not L-2wiki", good, "L-2wiki", g, roots)
        must(train, "zrckpc's fit of L-hotpotqa is scr-a5cp-zrckpc-hp", good, "L-hotpotqa", g, roots)
        must(train, "zrckpc's fit of L-musique is scr-a5cp-zrckpc",
             ["train", "--arm", "zrckpc", "--name", A5C.FIT["zrckpc"]["L-musique"]] + cr, "L-musique", g, roots)
        must(train, "zspkc's fit of L-musique", ["train", "--arm", "zspkc", "--name", FIT["zrckpc"]["L-musique"]] + cr,
             "L-musique", g, roots)
        must(train, "take --cache-root outputs/p1gb/cache", good[:5] + ["--cache-root", "outputs/p1c2/cache"],
             "L-musique", g, roots)
        must(train, "take --cache-root outputs/p1gb/cache", good[:5], "L-musique", g, roots)
        must(train, "not cpu", good, "L-musique", g, roots)
        must(train, "not cpu", good + ["--device", "cpu"], "L-musique", g, roots)
        for nm, rec_ in (("scr-a5c-zrckpc", {"arm": "zrckpc", "a5c": {"split": "L-musique"}}),
                         ("scr-a5cp-zrckpc", {"arm": "zrckpc", "a5c": {"split": "L-musique"}}),
                         ("scr-a5cp-zrckpc", {"arm": "zrckp", "p1gb": {}}),
                         ("scr-a5cp-zrckpc", {"arm": "zspkc", "a5cp": {"split": "L-musique"}}),
                         ("scr-a5cp-zspkc-hp", {"arm": "zspkc", "a5cp": {"split": "L-musique"}})):
            LC.write_json(T_ / "rf" / nm / "screen.json", rec_)
            must(read, "is not an A5cp fit" if nm != "scr-a5cp-zspkc-hp" else "is L-hotpotqa's fit",
                 ["--name", nm, "--out-root", str(T_ / "rf")] + cr + ["--device", "cuda"], g, roots)
        assert split_of_fit("scr-a5cp-zspkc-hp") == ("zspkc", "L-hotpotqa") and split_of_fit("scr-a5c-zspkc") == (
            None, None)
        # 6. relz's pair and re-call per arm against P1g-b's fits of its base arm, stamped A5cp's; a screen decided
        #    against A5c's base (P1c2's pools) and base screens of A5c's bases are refused; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits, d = BASE_FITS[arm], T_ / "calls" / arm
            null1 = SR.fake_null(d / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(d / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            old = A5C.BASE_FITS[arm]
            bs_old = {sp: str(SR.fake_compare(d / "old" / old[sp][0], old[sp][2], sp, {}, arm=BASE_OF[arm]))
                      for sp in old}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(d / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(d / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            zo = SR.fake_compare(d / "zo", FIT[arm]["L-musique"], "L-musique", {}, arm=arm, base_r5=0.6,
                                 base="/".join(("outputs",) + old["L-musique"]))
            must(pair, "is decided against", arm, [zo, zh], d / "bad")
            pr = pair(arm, [za, zh], d / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((d / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "A5cp" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            assert pj["pools"] == POOLS and "round" not in pj
            must(recall, "with step 1's", arm, null1, d / "pz.json", d / "rbad", bs_old)
            r = recall(arm, null1, d / "pz.json", d / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (d / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A5cp ({DECLARED})") and \
                f"| {BASE_OF[arm]} R@5 (P1g-b's) | {arm} R@5 (A5c's offsets, P1g-b's) |" in t
            assert json.loads((d / "rz.json").read_text(encoding="utf-8"))["stage"] == "A5cp"
        must(pair, "--arm takes", "zrckp", [], T_ / "x")
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with section 5's next forms; beside: webqsp and metaqa, musique in domain of both
        #    arms, FC@5 and hit@1 from the compares, A5c's delta and call on P1c2's pools, each fit's training
        #    forwards from its stamp; F1's note
        gm, gw, mi = ("L-musique", "musique"), ("L-musique", "webqsp"), CLAIM1_READ
        cmp_ = {a: {sp: fake_side(T_ / "cmp" / f"{a}{SX[sp]}", "/".join(("outputs",) + BASE_FITS[a][sp]),
                                  0.01 if sp == "L-musique" else -0.02, 0.03) for sp in SPLITS} for a in ARM_NAMES}
        old = {a: fake_recall(T_ / f"old-{a}.json", a, {gw: "GAIN"}, stage="A5c", declared=A5C.DECLARED)
               for a in ARM_NAMES}
        fits_root = T_ / "fits"                         # step 4's stamped toy fit of zrckpc on L-musique, only

        def v(c_m_, c_g_, over_m=None, over_g=None, **kw):
            rc = {"zrckpc": fake_recall(T_ / "m.json", "zrckpc", c_m_, **(over_m or {})),
                  "zspkc": fake_recall(T_ / "g.json", "zspkc", c_g_, **(over_g or {}))}
            return verdict(kw.get("recalls", rc), cmp_, old, T_ / "v", fits_root)

        rec = v({gw: "GAIN"}, {mi: "GAIN", gm: "LOSS"}, over_m={"filed": {gm: "GAIN"}})
        assert rec["arms"]["zrckpc"]["reads"]["L-musique musique"]["call_filed"] == "GAIN"
        md = (T_ / "v.md").read_text(encoding="utf-8")
        assert "| L-musique | musique | x | +0.0020 [+0.0015, +0.0025] | WITHIN (filed GAIN) |" in md, md
        assert rec["arms"]["zrckpc"]["training_forwards"] == {"L-musique": st["last_fit"], "L-hotpotqa": None}
        assert rec["arms"]["zspkc"]["training_forwards"] == {sp: None for sp in SPLITS}
        assert f"L-musique {st['last_fit']['typed_forwards']:,} / " in md and "L-hotpotqa not stamped" in md
        am, ag = rec["arms"]["zrckpc"], rec["arms"]["zspkc"]
        assert am["verdict"] == "ADOPT_FOR_FULL_RUN" and am["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert ag["verdict"] == "NOT_ADOPTED" and ag["outcome"] == "GAIN_AND_LOSS"
        assert ag["next_form"] == NEXT["GAIN_AND_LOSS"] and "A5cb" in ag["next_form"]
        assert rec["f1"].startswith("only zrckpc adopted")
        assert sorted(am["beside"]) == sorted(f"{sp} {ds}" for sp in SPLITS for ds in BESIDE)
        assert rec["musique_in_domain_beside"]["zspkc"]["call"] == "GAIN"
        assert rec["musique_in_domain_beside"]["zrckpc"]["call"] == "WITHIN"
        x = am["reads"]["L-hotpotqa webqsp"]
        assert (x["fc5"], x["hit1"]) == (-0.02, 0.03) and am["reads"]["L-musique metaqa"]["fc5"] == 0.01
        assert x["a5c_p1c2"] == {"call": "WITHIN", "delta": 0.011} and am["reads"]["L-musique webqsp"]["a5c_p1c2"] == {
            "call": "GAIN", "delta": 0.005}
        assert not am["a5c_recall_problems"]
        md = (T_ / "v.md").read_text(encoding="utf-8")
        assert "closes only this form" in md and "| L-musique | webqsp |" in md and "P1g-c's call" in md
        rec = v({}, {gw: "GAIN"}, over_g={"stage": "A5c"})
        assert rec["arms"]["zrckpc"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zrckpc"]["outcome"] == "NO_GAIN"
        assert rec["arms"]["zrckpc"]["next_form"] == NEXT["NO_GAIN"] and "A5i" in NEXT["NO_GAIN"]
        assert rec["arms"]["zspkc"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zspkc"]["next_form"] == NEXT["INCOMPLETE"] and rec["f1"].startswith("not settled")
        rec = v({gw: "GAIN"}, {mi: "GAIN"})
        assert all(rec["arms"][a]["verdict"] == "ADOPT_FOR_FULL_RUN" for a in ARM_NAMES)
        assert rec["f1"].startswith("both arms adopted")
        rec = v({}, {})
        assert rec["f1"].startswith("neither arm adopted")
        short = fake_recall(T_ / "s.json", "zspkc", {gw: "GAIN"})
        sj = json.loads(short.read_text(encoding="utf-8"))
        LC.write_json(short, dict(sj, rows=sj["rows"][:11]))
        rec = verdict({"zrckpc": T_ / "none.json", "zspkc": short}, cmp_, old, T_ / "v", fits_root)
        assert rec["arms"]["zrckpc"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkc"]["verdict"] == "INCOMPLETE"
        assert "missing" in rec["arms"]["zrckpc"]["problems"][0] and "11 reads" in rec["arms"]["zspkc"]["problems"][0]
        rec = verdict({"zrckpc": fake_recall(T_ / "m.json", "zspkc", {gw: "GAIN"}), "zspkc": T_ / "g.json"}, cmp_,
                      {"zrckpc": T_ / "none.json", "zspkc": old["zspkc"]}, T_ / "v", fits_root)
        assert rec["arms"]["zrckpc"]["verdict"] == "INCOMPLETE" and rec["arms"]["zrckpc"]["a5c_recall_problems"]
    log(f"za5cp selftest: the arms are A5c's objects on the lean pair's carves, each its base arm's model with A5CRet "
        f"just above zret, the MLP without a message; P1g-b's roots inside, the same P1g-c's arms read and each a "
        f"family P1g-c's gate checks, all restored after; each arm starts as A5c's model bit for bit (state, draws, "
        f"scores and every gradient, typed and untyped, eval and training) and its toy fits inside P1g-b's roots equal "
        f"A5c's bit for bit, the offsets moving on the typed graph only; train and read refuse a shut, changed, foreign "
        f"or missing P1g-c gate, a missing or failed P1g-b gate, KT records not P1g-b's, P1c2's caches, the CPU, another "
        f"arm, split or fit name, and A5c's, P1g-b's or the wrong split's fits; with all in place they run on P1g-b's "
        f"roots and the fit is stamped A5cp's; relz's pair and re-call per arm against P1g-b's fits, stamped A5cp's, "
        f"refusing A5c's bases; the verdict's outcomes, next forms, beside (with each fit's stamped forwards) and F1 "
        f"note ({time.time() - t0:.1f}s): ok")
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
            log(f"za5cp {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za5cp: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
