"""P1g-b's full run (docs/P1GB_FULL_RUN.md): the lean pair, unchanged, on P1g-b's pools (outputs/mp_unified/zp1gb.py,
imported unchanged) on the four splits its screen did not fit, every read decided against the same arm's fit of its
split on P1c2's pools, and one verdict for the pair.

  zrckp  the MLP: four fits in outputs/full_p1gb_zrckp/fits; the screen's scr-p1gb-zrckp and scr-p1gb-zrckp-hp are its
         L-musique and L-hotpotqa fits. Decided against zrckp on P1c2's pools: scr-p1c3p-zrckp, scr-p1c3p-zrckp-hp and
         outputs/full_zrckp/fits/S (P1c3p's full run).
  zspk   the GNN: four fits in outputs/full_p1gb_zspk/fits; the screen's scr-p1gb-zspk and scr-p1gb-zspk-hp are its
         L-musique and L-hotpotqa fits. Decided against zspk on P1c2's pools: scr-p1c3-zspk, scr-p1c3-zspk-hp and
         outputs/full_zspk/fits/S (P1c3's full run).

Nothing of the full run trains or is read unless the gate's record says P1g-b's screen is ADOPT_FOR_FULL_RUN.

    python outputs/mp_unified/zp1gbf.py gate
    python outputs/mp_unified/zp1gbf.py train --arm zspk --split J5 --name J5 --out-root outputs/full_p1gb_zspk/fits \\
        --device cuda --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/zp1gbf.py read --name J5 --out-root outputs/full_p1gb_zspk/fits --device cuda --host \\
        --cache-root outputs/p1gb/cache
    python outputs/mp_unified/zp1gbf.py compare --new outputs/full_p1gb_zspk/fits/J5 \\
        --base outputs/full_zspk/fits/J5,outputs/step1/fits/J5 --out outputs/full_p1gb_zspk/compare-J5
    python outputs/mp_unified/zp1gbf.py grade --arm zspk
    python outputs/mp_unified/nullx.py regrade --null F1,...,F12 --grade outputs/full_p1gb_zspk/grade.json \\
        --base-compares L-musique=outputs/p1gb/base-zspk-L-musique.json,...,J5=outputs/p1gbf/base-zspk-J5.json \\
        --out outputs/full_p1gb_zspk/grade-nullx
    python outputs/mp_unified/zp1gbf.py verdict
    python outputs/mp_unified/zp1gbf.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1gb as GB  # noqa: E402  (P1g-b's arms, roots, gate, stamp and read; registers zrckp and zspk)

Z3, Z2, RM = GB.Z3, GB.Z2, GB.RM
Z, R, S2, S, LG, LC, SR = GB.Z, GB.R, GB.S2, GB.S, GB.LG, GB.LC, GB.SR
log = S.log

DECLARED = "docs/P1GB_FULL_RUN.md"
ARMS = tuple(GB.ARM_NAMES)                               # zrckp (the MLP), zspk (the GNN)
SCREEN_SPLITS = tuple(GB.SPLITS)                         # L-musique, L-hotpotqa: the screen's fits
NEW_SPLITS = ("J5", "L-2wiki", "L-squad", "L-metaqa")    # the four this run fits
OUT = ROOT / "outputs" / "p1gbf"
FULL = {a: ROOT / "outputs" / f"full_p1gb_{a}" for a in ARMS}
BASE_FOLDER = {"zrckp": "full_zrckp", "zspk": "full_zspk"}   # P1c3p's and P1c3's full runs, P1c2's pools
SCREENS = {a: {"L-musique": f"outputs/screen/{GB.FIT[a]['L-musique']}.json",
               "L-hotpotqa": f"outputs/screen/{GB.FIT[a]['L-hotpotqa']}.json"} for a in ARMS}
BASE_COMPARES = {a: {sp: (GB.BASE_SCREENS[a][sp] if sp in SCREEN_SPLITS else f"outputs/p1gbf/base-{a}-{sp}.json")
                     for sp in S2.SPLITS} for a in ARMS}
AGAINST = {"zrckp": "zrckp's fit of each split on P1c2's pools (scr-p1c3p-zrckp, outputs/full_zrckp)",
           "zspk": "zspk's fit of each split on P1c2's pools (scr-p1c3-zspk, outputs/full_zspk)"}
SCREEN_VERDICT = GB.VERDICT                              # outputs/p1gb/screen-verdict
GATE = GB.OUT / "full-gate"
FULL_VERDICT = GB.OUT / "full-verdict"
RULE = ("ADOPT when neither arm's re-grade under the seed null has a LOSS on any of its 36 reads and the two arms "
        "together have at least one primary GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a re-grade is missing or not "
        f"the declared one ({DECLARED}, section 3)")


def rel(p):
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def write(out, rec, md):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def base_fit(arm, split):
    """The base an arm's read of split is decided against: the same arm's fit of the split on P1c2's pools."""
    return GB.BASE_FITS[arm].get(split) or (BASE_FOLDER[arm], "fits", split)


# ── the gate: P1g-b's screen ADOPT_FOR_FULL_RUN ──────────────────────────────


def gate_problems(screen=None):
    p = Path(screen or SCREEN_VERDICT).with_suffix(".json")
    if not p.exists():
        return [f"{rel(p)} missing"], {}
    rec = json.loads(p.read_text(encoding="utf-8"))
    got = {"record": rel(p), "sha256": LC.sha_file(p), "verdict": rec.get("verdict")}
    if rec.get("stage") != "P1gb" or rec.get("declared_in") != GB.DECLARED:
        return [f"{rel(p)} is {rec.get('stage')}'s ({rec.get('declared_in')}), not P1g-b's ({GB.DECLARED})"], got
    if rec.get("verdict") != "ADOPT_FOR_FULL_RUN":
        return [f"P1g-b's screen is {rec.get('verdict')}, not ADOPT_FOR_FULL_RUN"], got
    return [], got


def gate(screen=None, out=None):
    """0 when P1g-b's screen is ADOPT_FOR_FULL_RUN, 1 otherwise (the feeder then drops every train and read of this
    run); the record goes to outputs/p1gb/full-gate.{json,md}."""
    probs, got = gate_problems(screen)
    g = "OPEN" if not probs else "SHUT"
    rec = {"stage": "P1gb full run", "declared_in": DECLARED, "gate": g, "problems": probs, "screen": got,
           "zp1gbf_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# P1g-b's full run ({DECLARED}): gate **{g}**", "",
          "Trains, reads, comparisons and grades wait for P1g-b's screen ADOPT_FOR_FULL_RUN.", ""]
    if got:
        md.append(f"- the screen: {got['record']} says {got['verdict']}")
    if probs:
        md += ["", "Problems:"] + [f"- {p}" for p in probs]
    write(out or GATE, rec, md)
    log(f"zp1gbf gate: {g}" + (f" ({'; '.join(probs)})" if probs else ""))
    return 0 if not probs else 1


def require_gate():
    probs, _ = gate_problems()
    if probs:
        raise SystemExit(f"zp1gbf: {'; '.join(probs)}; the full run neither trains nor reads")


# ── train, read, compare ─────────────────────────────────────────────────────


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = Path(out_root) / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zp1gbf_sha256": LC.sha_src(__file__), "p1gb_full": {
            "declared_in": DECLARED, "arm": arm, "split": split, "zp1gb_sha256": LC.sha_src(GB.__file__),
            "decided_against": "/".join(("outputs",) + tuple(base_fit(arm, split)))}})
        LC.write_json(sj, rec)


def train(argv, split):
    """zp1gb.py's train, unchanged in every step (P1g-b's roots, its pool gate, the KT records, rmatch's train and
    P1g-b's stamp), on the four splits its screen did not fit."""
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != GB.ARMS[arm]:
        raise SystemExit(f"zp1gbf: train takes --arm zrckp or zspk as filed, not {arm} ({S.ARMS.get(arm)})")
    if split not in NEW_SPLITS:
        raise SystemExit(f"zp1gbf: the full run fits {', '.join(NEW_SPLITS)}; {split} is not one of them")
    name, out_root = S2.where(argv)
    if name != split or Path(out_root).resolve() != (FULL[arm] / "fits").resolve():
        raise SystemExit(f"zp1gbf: train takes --name {split} --out-root outputs/full_p1gb_{arm}/fits, not {name}, "
                         f"{out_root}")
    require_gate()
    with GB.on_p1gb():
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        GB.gate_passed()
        Z3.kt_records(Z3.train_carves(split), GB.KT_ROOT)
        rc = RM.train(argv, split)
        GB.stamp(argv, arm, split)
    stamp(argv, arm, split)
    return rc


def read(argv):
    """zp1gb.py's read of one of this run's fits, after the gate opens only."""
    require_gate()
    name, out_root = S2.where(argv)
    sj = Path(out_root) / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if "p1gb_full" not in rec:
        raise SystemExit(f"zp1gbf: {sj} is not a fit of P1g-b's full run")
    return GB.read(argv)


def compare(rest):
    require_gate()
    return S2.main(["compare"] + rest)


# ── each arm's grade against its base, and one verdict for the pair ──────────


@contextlib.contextmanager
def on_base(arm):
    """relz's grade under arm's name, each read decided against base_fit(arm, split); relz restored after."""
    if arm not in ARMS:
        raise SystemExit(f"zp1gbf: --arm takes {' or '.join(ARMS)}, not {arm}")
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM = arm, arm
    Z.REL_FITS = {sp: base_fit(arm, sp) for sp in S2.SPLITS}
    Z.REL_SCREENS = dict(BASE_COMPARES[arm])
    Z.rel_fit = lambda split: base_fit(arm, split)
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def grade(arm, full_root=None, reuse=None, out=None, check_fits=True):
    """relz's grade of arm's six comparisons (the screen's two reused), against its fits on P1c2's pools."""
    full_root = Path(full_root or FULL.get(arm) or ".")
    reuse = SCREENS[arm] if reuse is None else reuse
    out = Path(out or full_root / "grade")
    with on_base(arm):
        rec = Z.grade(full_root, reuse, out, check_fits)
    js = out.with_suffix(".json")
    if js.exists():
        r = json.loads(js.read_text(encoding="utf-8"))
        r.update({"stage": "P1gb full run", "declared_in": DECLARED, "decided_against": AGAINST[arm],
                  "zp1gbf_sha256": LC.sha_src(__file__)})
        LC.write_json(js, r)
    md = out.with_suffix(".md")
    if md.exists():
        md.write_text(md.read_text(encoding="utf-8").replace("(docs/FULL_ROUND10.md)", f"({DECLARED})"), encoding="utf-8")
    return rec


def same_compares(got, want):
    norm = (lambda d: {k: Path(v).as_posix().lstrip("./") for k, v in (d or {}).items()})
    return norm(got) == norm(want)


def tag(r):
    return f"{r['split']} {r['dataset']}"


def beside_shares(grades):
    """Claim 1's share per read on P1g-b's pools, deciding nothing: (zrckp - RRF) / (zspk - RRF) on R@5, point
    estimates from the two grades' rows; None where a grade is missing or the GNN's gain is below 0.01."""
    rows = {}
    for a in ARMS:
        p = Path(grades[a])
        if p.exists():
            rows[a] = {(r["split"], r["dataset"]): r for r in json.loads(p.read_text(encoding="utf-8")).get("rows", [])}
    out = []
    if set(rows) != set(ARMS):
        return out
    for key in sorted(set(rows["zrckp"]) & set(rows["zspk"])):
        m, g = rows["zrckp"][key], rows["zspk"][key]
        gain_g, gain_m = g["new"] - g["rrf"], m["new"] - m["rrf"]
        out.append({"split": key[0], "dataset": key[1], "read": g.get("read"), "rrf": g["rrf"], "zrckp": m["new"],
                    "zspk": g["new"], "share": (gain_m / gain_g) if gain_g >= 0.01 else None})
    return out


def full_verdict(regrades=None, grades=None, out=None):
    """One verdict for the pair from the two re-grades under the seed null (RULE), written to
    outputs/p1gb/full-verdict.{json,md}; each arm's own re-grade verdict and Claim 1's shares are reported beside,
    deciding nothing."""
    regrades = regrades or {a: FULL[a] / "grade-nullx.json" for a in ARMS}
    grades = grades or {a: FULL[a] / "grade.json" for a in ARMS}
    arms, problems = {}, []
    for a in ARMS:
        p = Path(regrades[a])
        if not p.exists():
            problems.append(f"{a}: {rel(p)} missing")
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != a or "null_splits" not in rec:
            problems.append(f"{a}: {rel(p)} is arm {rec.get('arm')}'s, not {a}'s re-grade under the null")
            continue
        if not same_compares(rec.get("base_compares"), BASE_COMPARES[a]):
            problems.append(f"{a}: re-graded on base comparisons {rec.get('base_compares')}, not the declared ones")
        rows = rec.get("rows") or []
        if sorted((r["split"], r["dataset"]) for r in rows) != sorted((sp, ds) for sp in S2.SPLITS
                                                                       for ds in LG.EVAL_ORDER):
            problems.append(f"{a}: {len(rows)} reads, not the 36 of the six splits")
        if sum(bool(r.get("primary")) for r in rows) != 11:
            problems.append(f"{a}: {sum(bool(r.get('primary')) for r in rows)} primary reads, not 11")
        if rec.get("verdict") == "INCOMPLETE" or rec.get("problems"):
            problems.append(f"{a}: the re-grade is {rec.get('verdict')} ({rec.get('problems')})")
        arms[a] = {"regrade": rel(p), "regrade_sha256": LC.sha_file(p), "verdict": rec.get("verdict"),
                   "primary_gains": [tag(r) for r in rows if r.get("primary") and r["call"] == "GAIN"],
                   "gains": [tag(r) for r in rows if r["call"] == "GAIN"],
                   "losses": [tag(r) for r in rows if r["call"] == "LOSS"],
                   "calls": {tag(r): r["call"] for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows}}
    if problems:
        v = "INCOMPLETE"
    elif any(x["losses"] for x in arms.values()) or not sum(len(x["primary_gains"]) for x in arms.values()):
        v = "NOT_ADOPTED"
    else:
        v = "ADOPT"
    shares = beside_shares(grades)
    rec = {"stage": "P1gb full run", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arms": arms,
           "problems": problems, "beside_claim1_shares": shares, "zp1gbf_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# P1g-b's full run ({DECLARED}): **{v}**", "", f"Rule: {RULE}.", ""]
    for a, x in arms.items():
        md.append(f"- {a} against {AGAINST[a]} ({x['regrade']}; its own re-grade {x['verdict']}, deciding nothing): "
                  f"{len(x['primary_gains'])} primary GAIN ({', '.join(x['primary_gains']) or 'none'}), "
                  f"{len(x['gains'])} GAIN in all, {len(x['losses'])} LOSS ({', '.join(x['losses']) or 'none'})")
    if problems:
        md += ["", "Problems:"] + [f"- {p}" for p in problems]
    if shares:
        md += ["", "Beside, deciding nothing: R@5 on P1g-b's pools and Claim 1's share (zrckp − RRF) / (zspk − RRF), "
               "point estimates (— where the GNN gains less than 0.01).", "",
               "| split | dataset | read | RRF | zrckp | zspk | share |", "|---|---|---|---:|---:|---:|---:|"]
        for r in shares:
            sh = "—" if r["share"] is None else f"{r['share']:.3f}"
            md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['rrf']:.4f} | {r['zrckp']:.4f} | "
                      f"{r['zspk']:.4f} | {sh} |")
    write(out or FULL_VERDICT, rec, md)
    log(f"zp1gbf verdict: {v}" + (f"; problems {problems}" if problems else ""))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_screen(path, verdict, **over):
    rec = {"stage": "P1gb", "declared_in": GB.DECLARED, "verdict": verdict}
    rec.update(over)
    LC.write_json(Path(path).with_suffix(".json"), rec)
    return path


def fake_regrade(path, arm, calls=None, over=None):
    """A re-grade in nullx's form for arm on the 36 reads; calls = {(split, dataset): call}."""
    rows = [{"split": sp, "dataset": ds, "primary": S2.primary(sp, ds), "call": (calls or {}).get((sp, ds), "WITHIN")}
            for sp in S2.SPLITS for ds in LG.EVAL_ORDER]
    rec = {"arm": arm, "null_splits": list(S2.SPLITS), "verdict": "x", "problems": [], "rows": rows,
           "base_compares": dict(BASE_COMPARES[arm])}
    rec.update(over or {})
    LC.write_json(path, rec)
    return path


def selftest():
    global SCREEN_VERDICT
    t0 = time.time()
    # 1. the arms, splits, folders and bases
    assert ARMS == ("zrckp", "zspk") and SCREEN_SPLITS == ("L-musique", "L-hotpotqa")
    assert set(NEW_SPLITS) | set(SCREEN_SPLITS) == set(S2.SPLITS) and not set(NEW_SPLITS) & set(SCREEN_SPLITS)
    for a in ARMS:
        assert S.ARMS[a] == GB.ARMS[a], a
    assert base_fit("zrckp", "L-musique") == ("screen", "fits", "scr-p1c3p-zrckp")
    assert base_fit("zrckp", "J5") == ("full_zrckp", "fits", "J5")
    assert base_fit("zspk", "L-hotpotqa") == ("screen", "fits", "scr-p1c3-zspk-hp")
    assert base_fit("zspk", "L-squad") == ("full_zspk", "fits", "L-squad")
    assert all(len(BASE_COMPARES[a]) == 6 for a in ARMS)
    assert BASE_COMPARES["zspk"]["L-musique"] == "outputs/p1gb/base-zspk-L-musique.json"
    assert BASE_COMPARES["zrckp"]["L-metaqa"] == "outputs/p1gbf/base-zrckp-L-metaqa.json"
    assert SCREENS["zrckp"]["L-hotpotqa"] == "outputs/screen/scr-p1gb-zrckp-hp.json"
    assert all("full_p1gb_" in FULL[a].name for a in ARMS)
    tmp = Path(tempfile.mkdtemp(prefix="zp1gbf_"))
    saved_sv = SCREEN_VERDICT
    try:
        # 2. the gate: open on P1g-b's ADOPT_FOR_FULL_RUN only
        for name, verdict, over, want in (("yes", "ADOPT_FOR_FULL_RUN", {}, 0), ("no", "NOT_ADOPTED", {}, 1),
                                          ("inc", "INCOMPLETE", {}, 1),
                                          ("stage", "ADOPT_FOR_FULL_RUN", {"stage": "P1c2"}, 1),
                                          ("decl", "ADOPT_FOR_FULL_RUN", {"declared_in": "x"}, 1)):
            fake_screen(tmp / name, verdict, **over)
            assert gate(tmp / name, tmp / f"gate-{name}") == want, name
            g = json.loads((tmp / f"gate-{name}.json").read_text(encoding="utf-8"))
            assert g["gate"] == ("OPEN" if want == 0 else "SHUT") and g["declared_in"] == DECLARED
        assert gate(tmp / "none", tmp / "gate-none") == 1
        # 3. trains, reads and comparisons wait for the gate
        good = ["train", "--arm", "zspk", "--name", "J5", "--out-root", str(FULL["zspk"] / "fits")]
        for sv in ("none", "no", "stage"):
            SCREEN_VERDICT = tmp / sv
            SR.must_stop(require_gate)
            SR.must_stop(compare, ["--new", "x"])
            SR.must_stop(train, good, "J5")
            SR.must_stop(read, ["--name", "J5", "--out-root", str(tmp / "fits")])
        SCREEN_VERDICT = tmp / "yes"
        require_gate()
        (tmp / "fits" / "y").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "y" / "screen.json", {"arm": "zspk", "p1gb": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(tmp / "fits")])           # P1g-b's screen fit, not this run's
        # 4. train takes only this run's four splits, each arm into its own folder; P1c2's roots restored after
        before = {k: getattr(Z2, k) for k in GB.Z2_ROOTS}
        SR.must_stop(train, ["train", "--arm", "zspk", "--name", "L-musique", "--out-root",
                             str(FULL["zspk"] / "fits")], "L-musique")
        SR.must_stop(train, ["train", "--arm", "zsp", "--name", "J5", "--out-root", str(FULL["zspk"] / "fits")], "J5")
        SR.must_stop(train, ["train", "--arm", "zrckp", "--name", "J5", "--out-root", str(FULL["zspk"] / "fits")], "J5")
        SR.must_stop(train, ["train", "--arm", "zspk", "--name", "J5", "--out-root", str(FULL["zspk"] / "fits")],
                     "L-2wiki")
        SR.must_stop(train, good + ["--cache-root", "outputs/p1c2/cache"], "J5")     # P1c2's caches refused
        SR.must_stop(train, good, "J5")                  # zp1c2's own check stops: no --cache-root
        assert {k: getattr(Z2, k) for k in GB.Z2_ROOTS} == before
        # 5. the grades: each arm against its own fits on P1c2's pools; another arm's comparisons are a problem
        saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm, deltas, want in (("zrckp", {"J5": {"musique": 0.0215}}, "ADOPT"),
                                  ("zspk", {"L-2wiki": {"2wiki": -0.03}}, "NOT_ADOPTED")):
            full = tmp / f"full_{arm}"
            for sp in S2.SPLITS:
                SR.fake_compare(full, f"compare-{sp}", sp, deltas.get(sp, {}), arm=arm, base_r5=0.6,
                                base="/".join(("outputs",) + tuple(base_fit(arm, sp))))
            g = grade(arm, full, {}, full / "grade", check_fits=False)
            assert g["verdict"] == want and g["arm"] == arm and g["base_arm"] == arm, (arm, g["problems"])
            gj = json.loads((full / "grade.json").read_text(encoding="utf-8"))
            assert gj["declared_in"] == DECLARED and gj["decided_against"] == AGAINST[arm]
            assert f"({DECLARED})" in (full / "grade.md").read_text(encoding="utf-8")
            other = "zspk" if arm == "zrckp" else "zrckp"
            assert grade(other, full, {}, full / "grade-x", check_fits=False)["verdict"] == "INCOMPLETE"
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
        # 6. one verdict for the pair; Claim 1's shares beside from the two grades
        rg = {a: tmp / f"rg-{a}.json" for a in ARMS}
        gr = {a: tmp / f"full_{a}" / "grade.json" for a in ARMS}
        for (cm, cg), want in (
                (({("J5", "musique"): "GAIN"}, {}), "ADOPT"),
                (({}, {("L-2wiki", "2wiki"): "GAIN"}), "ADOPT"),
                (({("J5", "musique"): "GAIN"}, {("L-squad", "webqsp"): "LOSS"}), "NOT_ADOPTED"),
                (({("L-squad", "2wiki"): "GAIN"}, {("L-musique", "webqsp"): "GAIN"}), "NOT_ADOPTED"),   # no primary
                (({}, {}), "NOT_ADOPTED")):
            fake_regrade(rg["zrckp"], "zrckp", cm)
            fake_regrade(rg["zspk"], "zspk", cg)
            r = full_verdict(rg, gr, tmp / "fv")
            assert r["verdict"] == want, (cm, cg, r["verdict"], r["problems"])
            assert len(r["beside_claim1_shares"]) == 36
        fake_regrade(rg["zrckp"], "zrckp", {("J5", "musique"): "GAIN"})
        for over in ({"arm": "zrckp"}, {"null_splits": None}, {"base_compares": BASE_COMPARES["zrckp"]},
                     {"verdict": "INCOMPLETE"}, {"rows": []}):
            fake_regrade(rg["zspk"], "zspk", {}, over)
            if over.get("null_splits", 0) is None:
                r_ = json.loads(rg["zspk"].read_text(encoding="utf-8"))
                r_.pop("null_splits")
                LC.write_json(rg["zspk"], r_)
            assert full_verdict(rg, gr, tmp / "fv")["verdict"] == "INCOMPLETE", over
        assert full_verdict({"zrckp": rg["zrckp"], "zspk": tmp / "none.json"}, gr, tmp / "fv")["verdict"] == "INCOMPLETE"
        sh = beside_shares(gr)
        assert len(sh) == 36 and all(x["share"] is None or abs(x["share"]) < 1e9 for x in sh)
        assert beside_shares({"zrckp": gr["zrckp"], "zspk": tmp / "none.json"}) == []
    finally:
        SCREEN_VERDICT = saved_sv
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"zp1gbf selftest: four new splits, each arm against its own fits on P1c2's pools; the gate on P1g-b's screen "
        f"(six cases); trains, reads and comparisons wait for it; train refuses the screen's splits, another arm, "
        f"folder or cache root and restores P1c2's roots; grades against each arm's bases; one verdict for the pair "
        f"(five cases, six INCOMPLETEs) with Claim 1's shares beside ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="J5", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "gate":
        return gate()
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "grade":
        require_gate()
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARMS)
        rec = grade(gp.parse_args(rest).arm)
        return 1 if rec["verdict"] == "INCOMPLETE" else 0
    if k.cmd == "verdict":
        rec = full_verdict()
        return 1 if rec["verdict"] == "INCOMPLETE" else 0
    raise SystemExit("zp1gbf: gate, train, read, compare, grade, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
