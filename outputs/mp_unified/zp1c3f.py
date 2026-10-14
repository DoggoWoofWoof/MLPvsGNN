"""P1c3's full run (docs/P1C3_FULL_RUN.md): both of P1c3's arms (outputs/mp_unified/zp1c3.py, imported unchanged) on the
four splits its screen did not fit, every read decided against the same family's card fit of its split on today's pools,
and one verdict for the pair.

  zrck  zrc's model plus the KT head (the MLP): four fits in outputs/full_zrck/fits; the screen's scr-p1c3-zrck and
        scr-p1c3-zrck-hp are its L-musique and L-hotpotqa fits. Decided against zrc's card fits: scr-zrct, scr-zrct-hp and
        outputs/full_zrct/fits/S (round 33's bases).
  zspk  zsp's model plus the KT head (the GNN): four fits in outputs/full_zspk/fits; the screen's scr-p1c3-zspk and
        scr-p1c3-zspk-hp are its L-musique and L-hotpotqa fits. Decided against zsp's card fits: scr-zspg, scr-zspg-hp and
        outputs/full_zspg/fits/S.

The fits may start before the screen's verdict: nothing of the full run is read (read, compare) unless the gate's record
says P1c3's screen is ADOPT_FOR_FULL_RUN.

    python outputs/mp_unified/zp1c3f.py gate
    python outputs/mp_unified/zp1c3f.py train --arm zrck --split L-2wiki --name L-2wiki --out-root outputs/full_zrck/fits \\
        --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3f.py read --name L-2wiki --out-root outputs/full_zrck/fits --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zp1c3f.py compare --new outputs/full_zrck/fits/L-2wiki \\
        --base outputs/full_zrct/fits/L-2wiki,outputs/step1/fits/L-2wiki --out outputs/full_zrck/compare-L-2wiki
    python outputs/mp_unified/zp1c3f.py grade --arm zrck
    python outputs/mp_unified/nullx.py regrade --null F1,...,F12 --grade outputs/full_zrck/grade.json \\
        --base-compares L-musique=outputs/zbase2/base-zrc-L-musique.json,... --out outputs/full_zrck/grade-nullx
    python outputs/mp_unified/zp1c3f.py verdict
    python outputs/mp_unified/zp1c3f.py --selftest
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

import zp1c3 as P3  # noqa: E402  (P1c3's arms, KT carves, train and read; through it zp1c2's roots)

Z2, RM = P3.Z2, P3.RM
Z, R, S2, S, LG, LC, SR = P3.Z, P3.R, P3.S2, P3.S, P3.LG, P3.LC, P3.SR
log = S.log

DECLARED = "docs/P1C3_FULL_RUN.md"
ARMS = tuple(P3.ARMS)                                    # zrck (the MLP), zspk (the GNN)
SCREEN_SPLITS = tuple(P3.SPLITS)                         # L-musique, L-hotpotqa: the screen's fits
NEW_SPLITS = ("L-2wiki", "L-squad", "J5", "L-metaqa")    # the four this run fits
FULL = {a: ROOT / "outputs" / f"full_{a}" for a in ARMS}
BASE_FOLDER = {"zrc": "full_zrct", "zsp": "full_zspg"}
SCREENS = {a: {"L-musique": f"outputs/screen/scr-p1c3-{a}.json", "L-hotpotqa": f"outputs/screen/scr-p1c3-{a}-hp.json"}
           for a in ARMS}
RECALLS = {a: f"outputs/screen/scr-p1c3-{a}-pair-recall.json" for a in ARMS}
BASE_COMPARES = {"zrc": {sp: f"outputs/zbase2/base-zrc-{sp}.json" for sp in S2.SPLITS},
                 "zsp": {sp: (f"outputs/zg1/base-zspg-{sp}.json" if sp in SCREEN_SPLITS else
                              f"outputs/full_zspg/base-zspg-{sp}.json") for sp in S2.SPLITS}}
AGAINST = {"zrck": "zrc's card fit of each split on today's pools (scr-zrct, outputs/full_zrct)",
           "zspk": "zsp's card fit of each split on today's pools (scr-zspg, outputs/full_zspg)"}
SCREEN_VERDICT = ROOT / "outputs" / "p1c3" / "screen-verdict"
FULL_VERDICT = ROOT / "outputs" / "p1c3" / "full-verdict"
SCREEN_RULE = ("ADOPT_FOR_FULL_RUN when neither arm's re-call has a LOSS on any of its twelve reads and the two arms "
               "together have at least one GAIN; NOT_ADOPTED otherwise (docs/P1C3_KT_INPUT.md, section 4)")
FULL_RULE = ("ADOPT when neither arm's re-grade under the seed null has a LOSS on any of its 36 reads and the two arms "
             "together have at least one primary GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a re-grade is missing or "
             f"not the declared one ({DECLARED}, section 4)")


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
    """The base an arm's read of split is decided against: its family's card fit of the split on today's pools."""
    b = P3.BASE_OF[arm]
    return Z2.BASE_FITS[b].get(split) or (BASE_FOLDER[b], "fits", split)


# ── the screen's verdict, which gates every read of the full run ─────────────


def screen_verdict(recalls=None, out=None):
    """P1c3's rule on its two re-calls, written to outputs/p1c3/screen-verdict.{json,md}. A missing re-call, or one that
    is not P1c3's re-call of its arm on the screen's twelve reads, stops."""
    recalls = recalls or RECALLS
    arms, rows = {}, []
    want = sorted((sp, ds) for sp in SCREEN_SPLITS for ds in LG.EVAL_ORDER)
    for a in ARMS:
        p = Path(recalls[a])
        p = p if p.is_absolute() else ROOT / p
        if not p.exists():
            raise SystemExit(f"zp1c3f gate: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("stage") != "P1c3" or rec.get("declared_in") != P3.DECLARED or rec.get("arm") != a:
            raise SystemExit(f"zp1c3f gate: {p} is {rec.get('stage')}'s re-call of {rec.get('arm')}, not P1c3's of {a}")
        rs = rec.get("rows") or []
        if sorted((r["split"], r["dataset"]) for r in rs) != want:
            raise SystemExit(f"zp1c3f gate: {p} has {len(rs)} reads, not the twelve of {', '.join(SCREEN_SPLITS)}")
        tag = [f"{r['split']} {r['dataset']}" for r in rs]
        arms[a] = {"recall": rel(p), "recall_sha256": LC.sha_file(p), "verdict": rec.get("verdict"),
                   "gains": [t for t, r in zip(tag, rs) if r["call"] == "GAIN"],
                   "losses": [t for t, r in zip(tag, rs) if r["call"] == "LOSS"]}
        rows += [{"arm": a, **{k: r.get(k) for k in ("split", "dataset", "read", "base", "new", "delta", "lo", "hi",
                                                     "floor", "call", "call_filed")}} for r in rs]
    gains = sum(len(x["gains"]) for x in arms.values())
    ok = gains > 0 and not any(x["losses"] for x in arms.values())
    v = "ADOPT_FOR_FULL_RUN" if ok else "NOT_ADOPTED"
    rec = {"stage": "P1c3", "declared_in": P3.DECLARED, "full_run": DECLARED, "rule": SCREEN_RULE, "verdict": v,
           "arms": arms, "rows": rows, "zp1c3f_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# P1c3's screen ({P3.DECLARED}): **{v}**", "", f"Rule: {SCREEN_RULE}.", "",
          "R@5 of each arm (KT input, KIND_T's pools) minus its family's card fit of the split on today's pools, on the "
          "six s1eval carves; every read called under the seed null (docs/SCREENS.md). A changed call shows filed -> "
          "re-call.", "", "| arm | split | dataset | read | base R@5 | new R@5 | delta | floor | call |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["call"] if r.get("call_filed") in (None, r["call"]) else f"{r['call_filed']} -> {r['call']}"
        md.append(f"| {r['arm']} | {r['split']} | {r['dataset']} | {r.get('read')} | {r['base']:.4f} | {r['new']:.4f} | "
                  f"{r['delta']:+.4f} | {r['floor']:.4f} | {c} |")
    md += [""] + [f"- {a}: {len(x['gains'])} GAIN ({', '.join(x['gains']) or 'none'}), {len(x['losses'])} LOSS "
                  f"({', '.join(x['losses']) or 'none'}); its re-call {x['verdict']}" for a, x in arms.items()]
    md += ["", "ADOPT_FOR_FULL_RUN starts the reads of the full run (" + DECLARED + "); NOT_ADOPTED leaves its fits "
           "unread."]
    write(out or SCREEN_VERDICT, rec, md)
    log(f"zp1c3f gate: P1c3's screen {v}; " + "; ".join(f"{a} {len(x['gains'])} GAIN {len(x['losses'])} LOSS"
                                                         for a, x in arms.items()))
    return rec


def gate(recalls=None, out=None):
    """0 when P1c3's screen is ADOPT_FOR_FULL_RUN, 1 otherwise (the feeder then drops every read of the full run)."""
    return 0 if screen_verdict(recalls, out)["verdict"] == "ADOPT_FOR_FULL_RUN" else 1


def require_screen(path=None):
    p = Path(path or SCREEN_VERDICT).with_suffix(".json")
    if not p.exists():
        raise SystemExit(f"zp1c3f: {p} missing (zp1c3f.py gate); nothing of the full run is read before the screen's "
                         f"verdict")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("stage") != "P1c3" or rec.get("declared_in") != P3.DECLARED or rec.get("verdict") != "ADOPT_FOR_FULL_RUN":
        raise SystemExit(f"zp1c3f: P1c3's screen is {rec.get('verdict')} ({p}); the full run's fits stay unread")
    return rec


# ── train, read, compare ─────────────────────────────────────────────────────


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = Path(out_root) / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zp1c3f_sha256": LC.sha_src(__file__), "p1c3_full": {
            "declared_in": DECLARED, "arm": arm, "split": split, "zp1c3_sha256": LC.sha_src(P3.__file__),
            "decided_against": "/".join(("outputs",) + tuple(base_fit(arm, split)))}})
        LC.write_json(sj, rec)


def train(argv, split):
    """zp1c3.py's train, unchanged, with the four splits the screen did not fit in place of its two."""
    arm = R.arm_of(argv)
    if split not in NEW_SPLITS:
        raise SystemExit(f"zp1c3f: the full run fits {', '.join(NEW_SPLITS)}; {split} is not one of them")
    name, out_root = S2.where(argv)
    if arm not in ARMS or name != split or Path(out_root).resolve() != (FULL[arm] / "fits").resolve():
        raise SystemExit(f"zp1c3f: train takes --arm zrck|zspk --name {split} --out-root outputs/full_<arm>/fits, not "
                         f"{arm}, {name}, {out_root}")
    saved = P3.SPLITS
    P3.SPLITS = NEW_SPLITS
    try:
        rc = P3.train(list(argv), split)
    finally:
        P3.SPLITS = saved
    stamp(argv, arm, split)
    return rc


def read(argv):
    """zp1c3.py's read of one of this run's fits, after the screen's ADOPT_FOR_FULL_RUN only."""
    require_screen()
    name, out_root = S2.where(argv)
    sj = Path(out_root) / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if "p1c3_full" not in rec:
        raise SystemExit(f"zp1c3f: {sj} is not a fit of P1c3's full run")
    return P3.read(argv)


def compare(rest):
    require_screen()
    return S2.main(["compare"] + rest)


# ── each arm's grade against its base, and one verdict for the pair ──────────


@contextlib.contextmanager
def on_base(arm):
    """relz's grade under arm's name, each read decided against base_fit(arm, split); relz restored after."""
    if arm not in ARMS:
        raise SystemExit(f"zp1c3f: --arm takes {' or '.join(ARMS)}, not {arm}")
    b = P3.BASE_OF[arm]
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM = arm, b
    Z.REL_FITS = {sp: base_fit(arm, sp) for sp in S2.SPLITS}
    Z.REL_SCREENS = dict(BASE_COMPARES[b])
    Z.rel_fit = lambda split: base_fit(arm, split)
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def grade(arm, full_root=None, reuse=None, out=None, check_fits=True):
    """relz's grade of arm's six comparisons (the screen's two reused), against its family's card fits."""
    full_root = Path(full_root or FULL.get(arm) or ".")
    reuse = SCREENS[arm] if reuse is None else reuse
    out = Path(out or full_root / "grade")
    with on_base(arm):
        rec = Z.grade(full_root, reuse, out, check_fits)
    js = out.with_suffix(".json")
    if js.exists():
        r = json.loads(js.read_text(encoding="utf-8"))
        r.update({"stage": "P1c3 full run", "declared_in": DECLARED, "decided_against": AGAINST[arm],
                  "zp1c3f_sha256": LC.sha_src(__file__)})
        LC.write_json(js, r)
    md = out.with_suffix(".md")
    if md.exists():
        md.write_text(md.read_text(encoding="utf-8").replace("(docs/FULL_ROUND10.md)", f"({DECLARED})"), encoding="utf-8")
    return rec


def same_compares(got, want):
    norm = (lambda d: {k: Path(v).as_posix().lstrip("./") for k, v in (d or {}).items()})
    return norm(got) == norm(want)


def full_verdict(regrades=None, out=None):
    """One verdict for the pair from the two re-grades under the seed null (FULL_RULE), written to
    outputs/p1c3/full-verdict.{json,md}; each arm's own re-grade verdict is reported beside, deciding nothing."""
    regrades = regrades or {a: FULL[a] / "grade-nullx.json" for a in ARMS}
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
        if not same_compares(rec.get("base_compares"), BASE_COMPARES[P3.BASE_OF[a]]):
            problems.append(f"{a}: re-graded on base comparisons {rec.get('base_compares')}, not the declared ones")
        rows = rec.get("rows") or []
        if sorted((r["split"], r["dataset"]) for r in rows) != sorted((sp, ds) for sp in S2.SPLITS
                                                                       for ds in LG.EVAL_ORDER):
            problems.append(f"{a}: {len(rows)} reads, not the 36 of the six splits")
        if sum(bool(r.get("primary")) for r in rows) != 11:
            problems.append(f"{a}: {sum(bool(r.get('primary')) for r in rows)} primary reads, not 11")
        if rec.get("verdict") == "INCOMPLETE" or rec.get("problems"):
            problems.append(f"{a}: the re-grade is {rec.get('verdict')} ({rec.get('problems')})")
        tag = (lambda r: f"{r['split']} {r['dataset']}")
        arms[a] = {"regrade": rel(p), "regrade_sha256": LC.sha_file(p), "verdict": rec.get("verdict"),
                   "primary_gains": [tag(r) for r in rows if r.get("primary") and r["call"] == "GAIN"],
                   "gains": [tag(r) for r in rows if r["call"] == "GAIN"],
                   "losses": [tag(r) for r in rows if r["call"] == "LOSS"]}
    if problems:
        v = "INCOMPLETE"
    elif any(x["losses"] for x in arms.values()) or not sum(len(x["primary_gains"]) for x in arms.values()):
        v = "NOT_ADOPTED"
    else:
        v = "ADOPT"
    rec = {"stage": "P1c3 full run", "declared_in": DECLARED, "rule": FULL_RULE, "verdict": v, "arms": arms,
           "problems": problems, "zp1c3f_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# P1c3's full run ({DECLARED}): **{v}**", "", f"Rule: {FULL_RULE}.", ""]
    for a, x in arms.items():
        md.append(f"- {a} ({x['regrade']}; its own re-grade {x['verdict']}, deciding nothing): "
                  f"{len(x['primary_gains'])} primary GAIN ({', '.join(x['primary_gains']) or 'none'}), "
                  f"{len(x['gains'])} GAIN in all, {len(x['losses'])} LOSS ({', '.join(x['losses']) or 'none'})")
    if problems:
        md += ["", "Problems:"] + [f"- {p}" for p in problems]
    write(out or FULL_VERDICT, rec, md)
    log(f"zp1c3f verdict: {v}" + (f"; problems {problems}" if problems else ""))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls=None, **over):
    """A re-call in screen_recall's form for arm on the screen's twelve reads; calls = {(split, dataset): call}."""
    rows = [{"split": sp, "dataset": ds, "read": "in-domain", "base": 0.5, "new": 0.5, "delta": 0.0, "lo": -0.004,
             "hi": 0.004, "floor": 0.0075, "call": (calls or {}).get((sp, ds), "WITHIN"),
             "call_filed": (calls or {}).get((sp, ds), "WITHIN")} for sp in SCREEN_SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "P1c3", "declared_in": P3.DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def fake_regrade(path, arm, calls=None, over=None):
    """A re-grade in nullx's form for arm on the 36 reads; calls = {(split, dataset): call}."""
    rows = [{"split": sp, "dataset": ds, "primary": S2.primary(sp, ds), "call": (calls or {}).get((sp, ds), "WITHIN")}
            for sp in S2.SPLITS for ds in LG.EVAL_ORDER]
    rec = {"arm": arm, "null_splits": list(S2.SPLITS), "verdict": "x", "problems": [], "rows": rows,
           "base_compares": dict(BASE_COMPARES[P3.BASE_OF[arm]])}
    rec.update(over or {})
    LC.write_json(path, rec)
    return path


def selftest():
    global SCREEN_VERDICT
    t0 = time.time()
    # 1. the splits, folders and bases
    assert ARMS == ("zrck", "zspk") and SCREEN_SPLITS == ("L-musique", "L-hotpotqa")
    assert set(NEW_SPLITS) | set(SCREEN_SPLITS) == set(S2.SPLITS) and not set(NEW_SPLITS) & set(SCREEN_SPLITS)
    assert base_fit("zrck", "L-musique") == ("screen", "fits", "scr-zrct")
    assert base_fit("zrck", "J5") == ("full_zrct", "fits", "J5")
    assert base_fit("zspk", "L-hotpotqa") == ("screen", "fits", "scr-zspg-hp")
    assert base_fit("zspk", "L-squad") == ("full_zspg", "fits", "L-squad")
    assert all(len(BASE_COMPARES[b]) == 6 for b in ("zrc", "zsp"))
    assert BASE_COMPARES["zsp"]["L-musique"] == "outputs/zg1/base-zspg-L-musique.json"
    assert BASE_COMPARES["zsp"]["J5"] == "outputs/full_zspg/base-zspg-J5.json"
    tmp = Path(tempfile.mkdtemp(prefix="zp1c3f_"))
    saved_sv = SCREEN_VERDICT
    try:
        # 2. the screen's verdict: P1c3's rule on both re-calls; refusals
        rc = {a: tmp / f"rc-{a}.json" for a in ARMS}
        for (ck, cs), want in ((({("L-hotpotqa", "2wiki"): "GAIN"}, {}), "ADOPT_FOR_FULL_RUN"),
                               (({}, {("L-musique", "musique"): "GAIN"}), "ADOPT_FOR_FULL_RUN"),
                               (({("L-hotpotqa", "2wiki"): "GAIN"}, {("L-musique", "webqsp"): "LOSS"}), "NOT_ADOPTED"),
                               (({("L-musique", "metaqa"): "LOSS"}, {("L-musique", "2wiki"): "GAIN"}), "NOT_ADOPTED"),
                               (({}, {}), "NOT_ADOPTED")):
            fake_recall(rc["zrck"], "zrck", ck)
            fake_recall(rc["zspk"], "zspk", cs)
            r = screen_verdict(rc, tmp / "sv")
            assert r["verdict"] == want, (ck, cs, r["verdict"])
            assert gate(rc, tmp / "sv") == (0 if want == "ADOPT_FOR_FULL_RUN" else 1)
            assert (tmp / "sv.md").read_text(encoding="utf-8").startswith(f"# P1c3's screen ({P3.DECLARED}): **{want}**")
        fake_recall(rc["zspk"], "zspk", {})
        for i, (arm, over) in enumerate((("zrck", {"stage": "P1c2"}), ("zspk", {}), ("zrck", {"declared_in": "x"}))):
            bad = dict(rc)
            bad["zrck"] = fake_recall(tmp / f"bad{i}.json", arm, {}, **over)
            SR.must_stop(screen_verdict, bad, tmp / "x")
        short = fake_recall(tmp / "short.json", "zrck", {})
        r_ = json.loads(short.read_text(encoding="utf-8"))
        r_["rows"] = r_["rows"][:-1]
        LC.write_json(short, r_)
        SR.must_stop(screen_verdict, {"zrck": short, "zspk": rc["zspk"]}, tmp / "x")
        SR.must_stop(screen_verdict, {"zrck": tmp / "none.json", "zspk": rc["zspk"]}, tmp / "x")
        # 3. reads and comparisons wait for ADOPT_FOR_FULL_RUN
        SCREEN_VERDICT = tmp / "sv-none"
        SR.must_stop(require_screen)
        SR.must_stop(compare, ["--new", "x"])
        fake_recall(rc["zrck"], "zrck", {})
        screen_verdict(rc, tmp / "sv-no")
        SCREEN_VERDICT = tmp / "sv-no"
        SR.must_stop(require_screen)
        fake_recall(rc["zrck"], "zrck", {("L-musique", "2wiki"): "GAIN"})
        screen_verdict(rc, tmp / "sv-yes")
        SCREEN_VERDICT = tmp / "sv-yes"
        assert require_screen()["verdict"] == "ADOPT_FOR_FULL_RUN"
        (tmp / "fits" / "y").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "y" / "screen.json", {"arm": "zrck", "p1c3": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(tmp / "fits")])          # not a fit of this run
        # 4. train takes only this run's four splits, each arm into its own folder, and restores zp1c3's splits
        good = ["train", "--arm", "zrck", "--name", "L-2wiki", "--out-root", str(FULL["zrck"] / "fits")]
        SR.must_stop(train, ["train", "--arm", "zrck", "--name", "L-musique", "--out-root",
                             str(FULL["zrck"] / "fits")], "L-musique")
        SR.must_stop(train, ["train", "--arm", "zrc", "--name", "L-2wiki", "--out-root", str(FULL["zrck"] / "fits")],
                     "L-2wiki")
        SR.must_stop(train, ["train", "--arm", "zspk", "--name", "L-2wiki", "--out-root", str(FULL["zrck"] / "fits")],
                     "L-2wiki")
        SR.must_stop(train, ["train", "--arm", "zrck", "--name", "J5", "--out-root", str(FULL["zrck"] / "fits")],
                     "L-2wiki")
        SR.must_stop(train, good, "L-2wiki")              # zp1c3's train itself stops: no --cache-root
        assert P3.SPLITS == SCREEN_SPLITS
        # 5. the grades: each arm against its own family's base fits; a comparison against another base is a problem
        saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm, deltas, want in (("zrck", {"J5": {"musique": 0.0215}}, "ADOPT"),
                                  ("zspk", {"L-2wiki": {"2wiki": -0.03}}, "NOT_ADOPTED")):
            full = tmp / f"full_{arm}"
            for sp in S2.SPLITS:
                SR.fake_compare(full, f"compare-{sp}", sp, deltas.get(sp, {}), arm=arm, base_r5=0.6,
                                base="/".join(("outputs",) + tuple(base_fit(arm, sp))))
            g = grade(arm, full, {}, full / "grade", check_fits=False)
            assert g["verdict"] == want and g["arm"] == arm and g["base_arm"] == P3.BASE_OF[arm], (arm, g["problems"])
            gj = json.loads((full / "grade.json").read_text(encoding="utf-8"))
            assert gj["declared_in"] == DECLARED and gj["decided_against"] == AGAINST[arm]
            assert f"({DECLARED})" in (full / "grade.md").read_text(encoding="utf-8")
            other = "zspk" if arm == "zrck" else "zrck"
            assert grade(other, full, {}, full / "grade-x", check_fits=False)["verdict"] == "INCOMPLETE"
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
        # 6. one verdict for the pair
        rg = {a: tmp / f"rg-{a}.json" for a in ARMS}
        for (ck, cs), want in (
                (({("J5", "musique"): "GAIN"}, {}), "ADOPT"),
                (({}, {("L-2wiki", "2wiki"): "GAIN"}), "ADOPT"),
                (({("J5", "musique"): "GAIN"}, {("L-squad", "webqsp"): "LOSS"}), "NOT_ADOPTED"),
                (({("L-squad", "2wiki"): "GAIN"}, {("L-musique", "webqsp"): "GAIN"}), "NOT_ADOPTED"),   # no primary
                (({}, {}), "NOT_ADOPTED")):
            fake_regrade(rg["zrck"], "zrck", ck)
            fake_regrade(rg["zspk"], "zspk", cs)
            r = full_verdict(rg, tmp / "fv")
            assert r["verdict"] == want, (ck, cs, r["verdict"], r["problems"])
        fake_regrade(rg["zrck"], "zrck", {("J5", "musique"): "GAIN"})
        for over in ({"arm": "zrck"}, {"null_splits": None}, {"base_compares": BASE_COMPARES["zrc"]},
                     {"verdict": "INCOMPLETE"}, {"rows": []}):
            fake_regrade(rg["zspk"], "zspk", {}, over)
            if over.get("null_splits", 0) is None:
                r_ = json.loads(rg["zspk"].read_text(encoding="utf-8"))
                r_.pop("null_splits")
                LC.write_json(rg["zspk"], r_)
            assert full_verdict(rg, tmp / "fv")["verdict"] == "INCOMPLETE", over
        assert full_verdict({"zrck": rg["zrck"], "zspk": tmp / "none.json"}, tmp / "fv")["verdict"] == "INCOMPLETE"
    finally:
        SCREEN_VERDICT = saved_sv
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"zp1c3f selftest: four new splits, each arm against its family's card fits; the screen's gate on P1c3's rule "
        f"(five cases, five refusals); reads and comparisons wait for ADOPT_FOR_FULL_RUN; train refuses the screen's "
        f"splits, another arm or folder and restores zp1c3's splits; grades against each family's bases; one verdict "
        f"for the pair (five cases, six INCOMPLETEs) ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-2wiki", choices=S2.SPLITS)
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
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARMS)
        rec = grade(gp.parse_args(rest).arm)
        return 1 if rec["verdict"] == "INCOMPLETE" else 0
    if k.cmd == "verdict":
        rec = full_verdict()
        return 1 if rec["verdict"] == "INCOMPLETE" else 0
    raise SystemExit("zp1c3f: gate, train, read, compare, grade, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
