"""P1c3p's full run (docs/P1C3P_FULL_RUN.md): zrckp, P1c3p's MLP arm (outputs/mp_unified/zp1c3p.py, imported
unchanged), on the four splits its screen did not fit, every read decided against zrc's card fit of its split on today's
pools: the bases, base comparisons, null and rule of zrck's full run (outputs/mp_unified/zp1c3f.py, imported unchanged).

  zrckp  zrc's model plus the pointwise KT head (the row's own eight KT columns, 161 parameters, no score input): four
         fits in outputs/full_zrckp/fits; the screen's scr-p1c3p-zrckp and scr-p1c3p-zrckp-hp are its L-musique and
         L-hotpotqa fits. Decided against scr-zrct, scr-zrct-hp and outputs/full_zrct/fits/S; zrck's fit of the split
         (scr-p1c3-zrck, scr-p1c3-zrck-hp, outputs/full_zrck/fits/S) and step 1's are reported beside.

The fits may start before the screen's verdict: nothing of the full run is read (read, compare) unless P1c3p's screen
is ADOPT_FOR_FULL_RUN and P1c3's full run is ADOPT.

    python outputs/mp_unified/zp1c3pf.py gate
    python outputs/mp_unified/zp1c3pf.py train --split J5 --name J5 --arm zrckp --out-root outputs/full_zrckp/fits \\
        --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3pf.py read --name J5 --out-root outputs/full_zrckp/fits --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3pf.py compare --new outputs/full_zrckp/fits/J5 \\
        --base outputs/full_zrct/fits/J5,outputs/full_zrck/fits/J5,outputs/step1/fits/J5 --out outputs/full_zrckp/compare-J5
    python outputs/mp_unified/zp1c3pf.py grade
    python outputs/mp_unified/nullx.py regrade --null F1,...,F12 --grade outputs/full_zrckp/grade.json \\
        --base-compares L-musique=outputs/zbase2/base-zrc-L-musique.json,... --out outputs/full_zrckp/grade-nullx
    python outputs/mp_unified/zp1c3pf.py verdict
    python outputs/mp_unified/zp1c3pf.py --selftest
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

import zp1c3p as PP  # noqa: E402  (P1c3p's arm, stamp and read; registers zrckp)
import zp1c3f as F  # noqa: E402  (zrck's full run: its bases, base comparisons and verdict record)

Z3, Z2, RM = PP.Z3, PP.Z2, PP.RM
Z, R, S2, S, LG, LC, SR = PP.Z, PP.R, PP.S2, PP.S, PP.LG, PP.LC, PP.SR
log = S.log

DECLARED = "docs/P1C3P_FULL_RUN.md"
ARM, BASE = PP.ARM, PP.BASE                              # zrckp (the MLP), decided against zrc
SCREEN_SPLITS = tuple(PP.SPLITS)                         # L-musique, L-hotpotqa: the screen's fits
NEW_SPLITS = ("J5", "L-2wiki", "L-squad", "L-metaqa")    # the four this run fits
FULL = ROOT / "outputs" / f"full_{ARM}"
SCREENS = {sp: f"outputs/screen/{PP.FIT[sp]}.json" for sp in SCREEN_SPLITS}
BASE_COMPARES = dict(F.BASE_COMPARES[BASE])              # zrc's, as in zrck's full run
SCREEN_VERDICT = PP.VERDICT                              # outputs/p1c3p/screen-verdict
P1C3_FULL_VERDICT = F.FULL_VERDICT                       # outputs/p1c3/full-verdict
GATE = PP.OUT / "full-gate"
FULL_VERDICT = PP.OUT / "full-verdict"
AGAINST = "zrc's card fit of each split on today's pools (scr-zrct, outputs/full_zrct)"
BESIDE = "zrck's fit of the split (scr-p1c3-zrck, outputs/full_zrck: the head with the z-scored score) and step 1's"
RULE = ("ADOPT when zrckp's re-grade under the seed null has no LOSS on any of its 36 reads and at least one primary "
        "GAIN; NOT_ADOPTED otherwise; INCOMPLETE when the re-grade is missing or not the declared one "
        f"({DECLARED}, section 4)")


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


def base_fit(split):
    """zrc's card fit of split on today's pools: the base zrck's full run decided against."""
    return F.base_fit("zrck", split)


def zrck_fit(split):
    """zrck's fit of split (P1c3's screen or its full run), reported beside."""
    return ("screen", "fits", PP.ZRCK_FIT[split]) if split in SCREEN_SPLITS else ("full_zrck", "fits", split)


# ── the gate: P1c3p's screen ADOPT_FOR_FULL_RUN and P1c3's full run ADOPT ─────


def gate_problems(screen=None, p1c3_full=None):
    probs, recs = [], {}
    for key, path, stage, declared, want in (
            ("screen", screen or SCREEN_VERDICT, "P1c3p", PP.DECLARED, "ADOPT_FOR_FULL_RUN"),
            ("p1c3_full", p1c3_full or P1C3_FULL_VERDICT, "P1c3 full run", F.DECLARED, "ADOPT")):
        p = Path(path).with_suffix(".json")
        if not p.exists():
            probs.append(f"{rel(p)} missing")
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        recs[key] = {"record": rel(p), "sha256": LC.sha_file(p), "verdict": rec.get("verdict")}
        if rec.get("stage") != stage or rec.get("declared_in") != declared:
            probs.append(f"{rel(p)} is {rec.get('stage')}'s ({rec.get('declared_in')}), not {stage}'s ({declared})")
        elif rec.get("verdict") != want:
            probs.append(f"{stage} is {rec.get('verdict')}, not {want}")
    return probs, recs


def gate(screen=None, p1c3_full=None, out=None):
    """0 when P1c3p's screen is ADOPT_FOR_FULL_RUN and P1c3's full run ADOPT, 1 otherwise (the feeder then drops every
    read of this run); the record goes to outputs/p1c3p/full-gate.{json,md}."""
    probs, recs = gate_problems(screen, p1c3_full)
    g = "OPEN" if not probs else "SHUT"
    rec = {"stage": "P1c3p full run", "declared_in": DECLARED, "gate": g, "problems": probs, "records": recs,
           "zp1c3pf_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# P1c3p's full run ({DECLARED}): gate **{g}**", "",
          "Reads, comparisons and the grade wait for P1c3p's screen ADOPT_FOR_FULL_RUN and P1c3's full run ADOPT.", ""]
    md += [f"- {k}: {v['record']} says {v['verdict']}" for k, v in recs.items()]
    if probs:
        md += ["", "Problems:"] + [f"- {p}" for p in probs]
    write(out or GATE, rec, md)
    log(f"zp1c3pf gate: {g}" + (f" ({'; '.join(probs)})" if probs else ""))
    return 0 if not probs else 1


def require_gate():
    probs, _ = gate_problems()
    if probs:
        raise SystemExit(f"zp1c3pf: {'; '.join(probs)}; the full run's fits stay unread")


# ── train, read, compare ─────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = Path(out_root) / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zp1c3pf_sha256": LC.sha_src(__file__), "p1c3p_full": {
            "declared_in": DECLARED, "arm": ARM, "split": split, "zp1c3p_sha256": LC.sha_src(PP.__file__),
            "zp1c3f_sha256": LC.sha_src(F.__file__),
            "decided_against": "/".join(("outputs",) + tuple(base_fit(split)))}})
        LC.write_json(sj, rec)


def train(argv, split):
    """zp1c3p.py's train, unchanged in every step, on the four splits its screen did not fit."""
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != PP.ARMS[ARM]:
        raise SystemExit(f"zp1c3pf: train takes --arm {ARM} ({PP.ARMS[ARM]}), not {arm} ({S.ARMS.get(arm)})")
    if split not in NEW_SPLITS:
        raise SystemExit(f"zp1c3pf: the full run fits {', '.join(NEW_SPLITS)}; {split} is not one of them")
    name, out_root = S2.where(argv)
    if name != split or Path(out_root).resolve() != (FULL / "fits").resolve():
        raise SystemExit(f"zp1c3pf: train takes --name {split} --out-root outputs/full_{ARM}/fits, not {name}, "
                         f"{out_root}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        PP.stamp(argv, split)
    stamp(argv, split)
    return rc


def read(argv):
    """zp1c3p.py's read of one of this run's fits, after the gate opens only."""
    require_gate()
    name, out_root = S2.where(argv)
    sj = Path(out_root) / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if "p1c3p_full" not in rec:
        raise SystemExit(f"zp1c3pf: {sj} is not a fit of P1c3p's full run")
    return PP.read(argv)


def compare(rest):
    require_gate()
    return S2.main(["compare"] + rest)


# ── the grade against zrc's card fits, and the verdict ───────────────────────


@contextlib.contextmanager
def on_base():
    """relz's grade under zrckp's name, each read decided against base_fit(split); relz restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM = ARM, BASE
    Z.REL_FITS = {sp: base_fit(sp) for sp in S2.SPLITS}
    Z.REL_SCREENS = dict(BASE_COMPARES)
    Z.rel_fit = base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def grade(full_root=None, reuse=None, out=None, check_fits=True):
    """relz's grade of zrckp's six comparisons (the screen's two reused), against zrc's card fits."""
    full_root = Path(full_root or FULL)
    reuse = SCREENS if reuse is None else reuse
    out = Path(out or full_root / "grade")
    with on_base():
        rec = Z.grade(full_root, reuse, out, check_fits)
    js = out.with_suffix(".json")
    if js.exists():
        r = json.loads(js.read_text(encoding="utf-8"))
        r.update({"stage": "P1c3p full run", "declared_in": DECLARED, "decided_against": AGAINST,
                  "zp1c3pf_sha256": LC.sha_src(__file__)})
        LC.write_json(js, r)
    md = out.with_suffix(".md")
    if md.exists():
        md.write_text(md.read_text(encoding="utf-8").replace("(docs/FULL_ROUND10.md)", f"({DECLARED})"), encoding="utf-8")
    return rec


def beside_rows(compare_file, split):
    """zrckp minus zrck on the split's six reads, from the split's comparison (its base zrck_fit(split)); None when
    absent."""
    p = Path(compare_file)
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        return None
    c = json.loads(p.read_text(encoding="utf-8"))
    key = next((b for b in c.get("bases", []) if S2.parts_of(b)[-3:] == zrck_fit(split)), None)
    if key is None:
        return None
    return [{"dataset": r["dataset"], "read": r["read"], "delta": r["delta"], "lo": r["lo"], "hi": r["hi"],
             "call": r["call"]} for r in c["by_base"][key]["rows"]]


def tag(r):
    return f"{r['split']} {r['dataset']}"


def full_verdict(regrade=None, compares=None, out=None):
    """zrckp's verdict from its re-grade under the seed null (RULE), written to outputs/p1c3p/full-verdict.{json,md};
    zrckp minus zrck reported beside, deciding nothing."""
    p = Path(regrade or FULL / "grade-nullx.json")
    problems, rows, rec = [], [], {}
    if not p.exists():
        problems.append(f"{rel(p)} missing")
    else:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != ARM or "null_splits" not in rec:
            problems.append(f"{rel(p)} is arm {rec.get('arm')}'s, not {ARM}'s re-grade under the null")
        if not F.same_compares(rec.get("base_compares"), BASE_COMPARES):
            problems.append(f"re-graded on base comparisons {rec.get('base_compares')}, not the declared ones")
        rows = rec.get("rows") or []
        if sorted((r["split"], r["dataset"]) for r in rows) != sorted((sp, ds) for sp in S2.SPLITS
                                                                       for ds in LG.EVAL_ORDER):
            problems.append(f"{len(rows)} reads, not the 36 of the six splits")
        if sum(bool(r.get("primary")) for r in rows) != 11:
            problems.append(f"{sum(bool(r.get('primary')) for r in rows)} primary reads, not 11")
        if rec.get("verdict") == "INCOMPLETE" or rec.get("problems"):
            problems.append(f"the re-grade is {rec.get('verdict')} ({rec.get('problems')})")
    primary_gains = [tag(r) for r in rows if r.get("primary") and r.get("call") == "GAIN"]
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    if problems:
        v = "INCOMPLETE"
    elif losses or not primary_gains:
        v = "NOT_ADOPTED"
    else:
        v = "ADOPT"
    compares = compares or {sp: (SCREENS[sp] if sp in SCREEN_SPLITS else FULL / f"compare-{sp}.json")
                            for sp in S2.SPLITS}
    beside = {sp: beside_rows(compares[sp], sp) for sp in S2.SPLITS}
    out_rec = {"stage": "P1c3p full run", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM,
               "base_arm": BASE, "regrade": rel(p), "regrade_sha256": LC.sha_file(p) if p.exists() else None,
               "regrade_verdict": rec.get("verdict"), "problems": problems, "primary_gains": primary_gains,
               "gains": gains, "losses": losses, "calls": {tag(r): r.get("call") for r in rows},
               "beside_zrckp_minus_zrck": beside, "zp1c3pf_sha256": LC.sha_src(__file__),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# P1c3p's full run ({DECLARED}): **{v}**", "", f"Rule: {RULE}.", "",
          f"zrckp ({PP.N_KTP} head parameters, the row's own eight KT columns) against zrc on today's pools, after the "
          f"re-grade under the seed null: {len(primary_gains)} primary GAIN ({', '.join(primary_gains) or 'none'}), "
          f"{len(gains)} GAIN in all, {len(losses)} LOSS ({', '.join(losses) or 'none'}); the re-grade's own verdict "
          f"{rec.get('verdict')}, deciding nothing."]
    if problems:
        md += ["", "Problems:"] + [f"- {x}" for x in problems]
    md += ["", "Beside, deciding nothing: zrckp minus zrck (P1c3's head with the z-scored score), R@5 / FC@5 / hit@1, "
           "with the 95% question-bootstrap interval of the R@5 difference.", "",
           "| split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075 floor) |",
           "|---|---|---|---|---|---|---|"]
    for sp in S2.SPLITS:
        for r in beside[sp] or []:
            d, lo, hi = r["delta"], r["lo"], r["hi"]
            md.append(f"| {sp} | {r['dataset']} | {r['read']} | {d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | "
                      f"{d[1]:+.4f} | {d[2]:+.4f} | {r['call']} |")
        if beside[sp] is None:
            md.append(f"| {sp} | (no comparison with {'/'.join(zrck_fit(sp))}) | | | | | |")
    write(out or FULL_VERDICT, out_rec, md)
    log(f"zp1c3pf verdict: {v} ({len(primary_gains)} primary GAIN, {len(losses)} LOSS"
        f"{'; ' + '; '.join(problems) if problems else ''})")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_verdict(path, stage, declared, verdict):
    LC.write_json(Path(path).with_suffix(".json"), {"stage": stage, "declared_in": declared, "verdict": verdict})
    return path


def fake_regrade(path, calls=None, over=None):
    """A re-grade in nullx's form for zrckp on the 36 reads; calls = {(split, dataset): call}."""
    rows = [{"split": sp, "dataset": ds, "primary": S2.primary(sp, ds), "call": (calls or {}).get((sp, ds), "WITHIN")}
            for sp in S2.SPLITS for ds in LG.EVAL_ORDER]
    rec = {"arm": ARM, "null_splits": list(S2.SPLITS), "verdict": "x", "problems": [], "rows": rows,
           "base_compares": dict(BASE_COMPARES)}
    rec.update(over or {})
    LC.write_json(path, rec)
    return path


def with_zrck_base(path, split):
    """Adds zrck's fit of split to a fake comparison as its second base (the beside rows)."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    b = "outputs\\" + "\\".join(zrck_fit(split))
    rec["bases"].append(b)
    rec["by_base"][b] = {"rows": rec["by_base"][rec["bases"][0]]["rows"], "verdict": "x", "decides": False}
    LC.write_json(path, rec)
    return path


def selftest():
    global SCREEN_VERDICT, P1C3_FULL_VERDICT
    t0 = time.time()
    # 1. the arm, splits, folders and bases: zrck's full run's, for zrckp
    assert ARM == "zrckp" and BASE == "zrc" and S.ARMS[ARM] == PP.ARMS[ARM] and PP.N_KTP == 161
    assert SCREEN_SPLITS == ("L-musique", "L-hotpotqa") == tuple(Z3.SPLITS)
    assert set(NEW_SPLITS) | set(SCREEN_SPLITS) == set(S2.SPLITS) and not set(NEW_SPLITS) & set(SCREEN_SPLITS)
    assert base_fit("L-musique") == ("screen", "fits", "scr-zrct") and base_fit("J5") == ("full_zrct", "fits", "J5")
    assert base_fit("L-hotpotqa") == ("screen", "fits", "scr-zrct-hp")
    assert zrck_fit("L-hotpotqa") == ("screen", "fits", "scr-p1c3-zrck-hp")
    assert zrck_fit("L-squad") == ("full_zrck", "fits", "L-squad")
    assert BASE_COMPARES == F.BASE_COMPARES["zrc"] and len(BASE_COMPARES) == 6
    assert SCREENS["L-musique"] == "outputs/screen/scr-p1c3p-zrckp.json"
    tmp = Path(tempfile.mkdtemp(prefix="zp1c3pf_"))
    saved_g = SCREEN_VERDICT, P1C3_FULL_VERDICT
    try:
        # 2. the gate: both verdicts, each the declared stage's; five shut cases
        sv, fv = tmp / "sv", tmp / "fv"
        for (s, f), want in ((("ADOPT_FOR_FULL_RUN", "ADOPT"), 0), (("NOT_ADOPTED", "ADOPT"), 1),
                             (("INCOMPLETE", "ADOPT"), 1), (("ADOPT_FOR_FULL_RUN", "NOT_ADOPTED"), 1),
                             (("ADOPT_FOR_FULL_RUN", "INCOMPLETE"), 1)):
            fake_verdict(sv, "P1c3p", PP.DECLARED, s)
            fake_verdict(fv, "P1c3 full run", F.DECLARED, f)
            assert gate(sv, fv, tmp / "g") == want, (s, f)
            g = json.loads((tmp / "g.json").read_text(encoding="utf-8"))
            assert g["gate"] == ("OPEN" if want == 0 else "SHUT") and g["declared_in"] == DECLARED
        fake_verdict(sv, "P1c3", PP.DECLARED, "ADOPT_FOR_FULL_RUN")                  # P1c3's screen, not P1c3p's
        assert gate(sv, fv, tmp / "g") == 1
        fake_verdict(sv, "P1c3p", PP.DECLARED, "ADOPT_FOR_FULL_RUN")
        fake_verdict(fv, "P1c3 full run", "docs/x.md", "ADOPT")
        assert gate(sv, fv, tmp / "g") == 1
        assert gate(sv, tmp / "none", tmp / "g") == 1
        # 3. reads and comparisons wait for the open gate
        SCREEN_VERDICT, P1C3_FULL_VERDICT = tmp / "none-sv", tmp / "none-fv"
        SR.must_stop(require_gate)
        SR.must_stop(compare, ["--new", "x"])
        SR.must_stop(read, ["--name", "J5", "--out-root", str(tmp / "fits")])
        fake_verdict(fv, "P1c3 full run", F.DECLARED, "ADOPT")
        fake_verdict(sv, "P1c3p", PP.DECLARED, "NOT_ADOPTED")
        SCREEN_VERDICT, P1C3_FULL_VERDICT = sv, fv
        SR.must_stop(require_gate)
        fake_verdict(sv, "P1c3p", PP.DECLARED, "ADOPT_FOR_FULL_RUN")
        require_gate()
        (tmp / "fits" / "y").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "y" / "screen.json", {"arm": ARM, "p1c3p": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(tmp / "fits")])          # the screen's, not this run's
        # 4. train takes only zrckp, this run's four splits, the split's name and this run's folder
        fits = str(FULL / "fits")
        SR.must_stop(train, ["train", "--arm", ARM, "--name", "L-musique", "--out-root", fits], "L-musique")
        SR.must_stop(train, ["train", "--arm", "zrck", "--name", "J5", "--out-root", fits], "J5")
        SR.must_stop(train, ["train", "--arm", "zrc", "--name", "J5", "--out-root", fits], "J5")
        SR.must_stop(train, ["train", "--arm", ARM, "--name", "L-2wiki", "--out-root", fits], "J5")
        SR.must_stop(train, ["train", "--arm", ARM, "--name", "J5", "--out-root", str(F.FULL["zrck"] / "fits")], "J5")
        SR.must_stop(train, ["train", "--arm", ARM, "--name", "J5", "--out-root", fits], "J5")    # no --cache-root
        assert tuple(Z3.SPLITS) == SCREEN_SPLITS and tuple(PP.SPLITS) == SCREEN_SPLITS
        # 5. the grade against zrc's card fits; a comparison decided by zrck's fit is a problem
        saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for i, (deltas, want) in enumerate((({"J5": {"musique": 0.0215}}, "ADOPT"),
                                            ({"L-2wiki": {"2wiki": -0.03}}, "NOT_ADOPTED"),
                                            ({"L-2wiki": {"metaqa": 0.03}}, "NOT_ADOPTED"))):   # no primary GAIN
            full = tmp / f"full{i}"
            for sp in S2.SPLITS:
                c = SR.fake_compare(full, f"compare-{sp}", sp, deltas.get(sp, {}), arm=ARM, base_r5=0.6,
                                    base="/".join(("outputs",) + tuple(base_fit(sp))))
                with_zrck_base(c, sp)
            g = grade(full, {}, full / "grade", check_fits=False)
            assert g["verdict"] == want and g["arm"] == ARM and g["base_arm"] == BASE, (deltas, g["problems"])
            gj = json.loads((full / "grade.json").read_text(encoding="utf-8"))
            assert gj["declared_in"] == DECLARED and gj["decided_against"] == AGAINST and gj["stage"] == "P1c3p full run"
            assert f"({DECLARED})" in (full / "grade.md").read_text(encoding="utf-8")
            rows = beside_rows(full / "compare-J5.json", "J5")
            assert rows is not None and len(rows) == len(LG.EVAL_ORDER)
            assert beside_rows(full / "compare-J5.json", "L-squad") is None
        wrong = tmp / "wrong"
        for sp in S2.SPLITS:
            SR.fake_compare(wrong, f"compare-{sp}", sp, {}, arm=ARM, base="/".join(("outputs",) + zrck_fit(sp)))
        assert grade(wrong, {}, wrong / "grade", check_fits=False)["verdict"] == "INCOMPLETE"
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
        # 6. the verdict: four cases and seven INCOMPLETEs; zrckp minus zrck beside
        rg = tmp / "rg.json"
        cmp = {sp: tmp / "full0" / f"compare-{sp}.json" for sp in S2.SPLITS}
        for calls, want in (({("J5", "musique"): "GAIN"}, "ADOPT"),
                            ({("J5", "musique"): "GAIN", ("L-squad", "webqsp"): "LOSS"}, "NOT_ADOPTED"),
                            ({("L-squad", "2wiki"): "GAIN"}, "NOT_ADOPTED"),                  # no primary GAIN
                            ({}, "NOT_ADOPTED")):
            fake_regrade(rg, calls)
            r = full_verdict(rg, cmp, tmp / "v")
            assert r["verdict"] == want, (calls, r["verdict"], r["problems"])
            assert all(r["beside_zrckp_minus_zrck"][sp] for sp in S2.SPLITS)
        assert (tmp / "v.md").read_text(encoding="utf-8").startswith(f"# P1c3p's full run ({DECLARED}): **NOT_ADOPTED**")
        for over in ({"arm": "zrck"}, {"null_splits": None}, {"base_compares": F.BASE_COMPARES["zsp"]},
                     {"verdict": "INCOMPLETE"}, {"problems": ["x"]}, {"rows": []}):
            fake_regrade(rg, {("J5", "musique"): "GAIN"}, over)
            if over.get("null_splits", 0) is None:
                r_ = json.loads(rg.read_text(encoding="utf-8"))
                r_.pop("null_splits")
                LC.write_json(rg, r_)
            assert full_verdict(rg, cmp, tmp / "v")["verdict"] == "INCOMPLETE", over
        assert full_verdict(tmp / "none.json", cmp, tmp / "v")["verdict"] == "INCOMPLETE"
    finally:
        SCREEN_VERDICT, P1C3_FULL_VERDICT = saved_g
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"zp1c3pf selftest: zrckp on four new splits against zrck's full run's bases; the gate on P1c3p's screen and "
        f"P1c3's full run (five cases, three refusals); reads and comparisons wait for the open gate; train refuses "
        f"another arm, split, name or folder and leaves the splits unchanged; the grade against zrc's card fits (three "
        f"cases, one INCOMPLETE); the verdict (four cases, seven INCOMPLETEs) with zrckp minus zrck beside "
        f"({time.time() - t0:.1f}s): ok")
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
        return 1 if grade()["verdict"] == "INCOMPLETE" else 0
    if k.cmd == "verdict":
        return 1 if full_verdict()["verdict"] == "INCOMPLETE" else 0
    raise SystemExit("zp1c3pf: gate, train, read, compare, grade, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
