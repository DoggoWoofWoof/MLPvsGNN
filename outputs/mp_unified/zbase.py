"""The next base if zrm is ADOPTED too (docs/BASE_ZRM_ZRS.md, declared 8 October about 09:50, before zrm's re-grade).
zrs (docs/FULL_ROUND15.md, ADOPTED 09:38) trains rmatch's match alone on zret's frozen fits; zrm (docs/FULL_ROUND14.md)
trains the match and zret's model together. If both are ADOPTED, zrs is the incumbent: zrm becomes the base when,
against zrs's fit of each split, at least one primary read GAINs and none of the 36 LOSEs under the seed null over every
split (nullx.py); otherwise zrs stays the base.

    python outputs/mp_unified/zbase.py gate --grades outputs/full_zrs/grade-nullx.json,outputs/full_zrm/grade-nullx.json
    python outputs/mp_unified/zbase.py base --split J5 --out outputs/zbase/base-zrs-J5
    python outputs/mp_unified/zbase.py compare --split J5 --out outputs/zbase/compare-J5
    python outputs/mp_unified/zbase.py grade --full-root outputs/zbase --out outputs/zbase/grade
    python outputs/mp_unified/zbase.py regrade --null F1,...,F12 --grade outputs/zbase/grade.json \\
        --out outputs/zbase/grade-nullx
    python outputs/mp_unified/zbase.py --selftest

Fits: zrm's and zrs's screen fits on L-musique and L-hotpotqa (scr-zrm, scr-zrm-hp; scr-zrs, scr-zrs-hp), their full
runs' fits on the other four splits; zret's fits as in zrs's grade (its screen fit on L-musique, its full run's on the
rest). `compare` is lean_screen's: zrm's fit against zrs's (decides), zret's and step 1's (reported only). `base` compares
zrs's fit with step 1's alone; nullx.py takes those as each read's base R@5 (later rounds decided against zrs's fits
reuse them). The grade is relz.py's and the re-grade nullx.py's, unchanged, under zrm's name with zrs's fits in place of
rel's. Nothing trains or reads, and no fit folder is written. Speed: this times nothing; any latency figure for the base
is cold (8216ffe).
"""
import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nullx as NX  # noqa: E402
import relz as Z  # noqa: E402

S2, S, SR = Z.S2, Z.S, Z.SR
LG, LC = S.LG, S.LC
log = S.log

ARM, BASE_ARM = "zrm", "zrs"
DOC = "docs/BASE_ZRM_ZRS.md"
SCREEN = {"L-musique": {"zrm": ("screen", "fits", "scr-zrm"), "zrs": ("screen", "fits", "scr-zrs"),
                        "zret": ("screen", "fits", "scr-zret")},
          "L-hotpotqa": {"zrm": ("screen", "fits", "scr-zrm-hp"), "zrs": ("screen", "fits", "scr-zrs-hp"),
                         "zret": ("full_zret", "fits", "L-hotpotqa")}}


def fit(arm, split):
    """An arm's fit of a split, as parts under outputs/."""
    if split not in S2.SPLITS or arm not in ("zrm", "zrs", "zret", "step1"):
        raise SystemExit(f"zbase: no fit of {arm} on {split}")
    if arm == "step1":
        return ("step1", "fits", split)
    return SCREEN.get(split, {}).get(arm) or (f"full_{arm}", "fits", split)


def path(parts):
    return "/".join(("outputs",) + tuple(parts))


# ── the gate ─────────────────────────────────────────────────────────────────


def gate(grades):
    """0 when zrs's and zrm's re-grades are both ADOPT (the comparison runs), 1 when either is not (zrs is the base,
    or zret stays it); a missing file, or one of another arm, stops (2)."""
    if len(grades) != 2:
        raise SystemExit("zbase gate: two re-grades, zrs's then zrm's")
    got = {}
    for want, fn in zip((BASE_ARM, ARM), grades):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zbase gate: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != want or "null_splits" not in rec:
            raise SystemExit(f"zbase gate: {p} is arm {rec.get('arm')}'s, not {want}'s re-grade under the null")
        got[want] = rec.get("verdict")
    ok = all(v == "ADOPT" for v in got.values())
    log(f"zbase gate: zrs {got[BASE_ARM]}, zrm {got[ARM]} -> {'compare zrm with zrs' if ok else 'no comparison'}")
    return 0 if ok else 1


# ── comparisons ──────────────────────────────────────────────────────────────


def compare(split, out):
    """zrm's fit of split against zrs's (decides), zret's and step 1's (reported only)."""
    return S2.main(["compare", "--new", path(fit(ARM, split)), "--base",
                    ",".join(path(fit(a, split)) for a in (BASE_ARM, "zret", "step1")), "--out", str(out)])


def base(split, out):
    """zrs's fit of split against step 1's alone: each read's base R@5 for nullx.py."""
    return S2.main(["compare", "--new", path(fit(BASE_ARM, split)), "--base", path(fit("step1", split)),
                    "--out", str(out)])


# ── relz.py's grade and nullx.py's re-grade, decided against zrs's fits ──────


ZRS_FITS = {sp: SCREEN[sp][BASE_ARM] for sp in SCREEN}


def zrs_fit(split):
    return fit(BASE_ARM, split)


@contextlib.contextmanager
def on_zrs():
    """relz.py's grade under zrm's name, each read decided against zrs's fit of its split; relz restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit = ARM, BASE_ARM, ZRS_FITS, zrs_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit = saved


def restamp(out, title=None):
    """A record written under zrm's name against zrs's fits: this round's file named, this file's sha added."""
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8").replace("(docs/FULL_ROUND10.md)", f"({DOC})")
        if title:
            t = t.replace(*title, 1)
        md.write_text(t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"decided_against": "zrs's fit of each split", "zbase_sha256": LC.sha_src(__file__), "doc": DOC})
        LC.write_json(js, rec)


def grade(full_root, out, check_fits=True):
    """relz.py's grade: ADOPT (zrm becomes the base) when a primary read GAINs and none of the 36 LOSEs."""
    with on_zrs():
        rec = Z.grade(full_root, {}, out, check_fits)
    restamp(out, ("# Full run of zrm against zrs:", "# zrm's fits against zrs's (the next base):"))
    return rec


def compares(root):
    return {sp: str(Path(root) / f"base-zrs-{sp}.json") for sp in S2.SPLITS}


def regrade(null_files, grade_file, out, base_root):
    """nullx.py's re-grade, each read's base R@5 zrs's (base-zrs-<split>.json under base_root)."""
    gp = Path(grade_file)
    if not gp.exists():
        raise SystemExit(f"zbase: {gp} missing")
    g = json.loads(gp.read_text(encoding="utf-8"))
    if g.get("arm") != ARM or g.get("base_arm") != BASE_ARM or g.get("decided_against") != "zrs's fit of each split":
        raise SystemExit(f"zbase: {gp} is {g.get('arm')} against {g.get('base_arm')}, not zrm against zrs's fits")
    rec = NX.regrade(null_files, grade_file, out, compares(base_root))
    restamp(out, ("# Full run of zrm re-graded under", "# zrm's fits against zrs's (the next base), re-graded under"))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    # 1. the fits: screen fits on the two screen splits, full runs' elsewhere; zret's as in zrs's grade
    assert fit("zrm", "L-musique") == ("screen", "fits", "scr-zrm") and fit("zrs", "L-hotpotqa") == (
        "screen", "fits", "scr-zrs-hp")
    assert fit("zret", "L-musique") == ("screen", "fits", "scr-zret")
    assert fit("zret", "L-hotpotqa") == ("full_zret", "fits", "L-hotpotqa")
    assert fit("zrs", "J5") == ("full_zrs", "fits", "J5") and fit("step1", "L-2wiki") == ("step1", "fits", "L-2wiki")
    assert path(fit("zrm", "L-metaqa")) == "outputs/full_zrm/fits/L-metaqa"
    SR.must_stop(fit, "rmatch", "J5")
    SR.must_stop(fit, "zrm", "L-nq")
    # 2. on_zrs swaps relz's names and restores them, after an error too
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit
    with on_zrs():
        assert Z.ARM == ARM and Z.BASE_ARM == BASE_ARM and Z.rel_fit("J5") == ("full_zrs", "fits", "J5")
        assert Z.rel_fit("L-musique") == ("screen", "fits", "scr-zrs")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit) == saved
    try:
        with on_zrs():
            raise KeyError("x")
    except KeyError:
        pass
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)

        def regr(name, arm, v):
            p = T / f"{name}.json"
            p.write_text(json.dumps({"arm": arm, "verdict": v, "null_splits": list(S2.SPLITS)}), encoding="utf-8")
            return str(p)

        # 3. the gate: both ADOPT -> 0; zrm NOT_ADOPTED or INCOMPLETE -> 1; the wrong arm, order or file -> stop
        zs, zm = regr("zs", "zrs", "ADOPT"), regr("zm", "zrm", "ADOPT")
        assert gate([zs, zm]) == 0
        assert gate([zs, regr("zm2", "zrm", "NOT_ADOPTED")]) == 1
        assert gate([zs, regr("zm3", "zrm", "INCOMPLETE")]) == 1
        assert gate([regr("zs2", "zrs", "NOT_ADOPTED"), zm]) == 1
        SR.must_stop(gate, [zm, zs])
        SR.must_stop(gate, [zs, str(T / "none.json")])
        SR.must_stop(gate, [zs])
        (T / "zm4.json").write_text(json.dumps({"arm": "zrm", "verdict": "ADOPT"}), encoding="utf-8")   # a filed grade
        SR.must_stop(gate, [zs, str(T / "zm4.json")])
        # 4. the grade, decided against zrs's fits: J5 webqsp (primary) GAINs, L-2wiki hotpotqa LOSEs at 0.0075 only
        full = T / "zbase"
        full.mkdir()
        null = NX.fake_null(T / "null", "n", {("L-2wiki", "hotpotqa"): (-0.010, 0.012)})              # floor 0.0221
        for sp in S2.SPLITS:
            d = {"J5": {"webqsp": 0.05}, "L-2wiki": {"hotpotqa": -0.0147}}.get(sp, {})
            SR.fake_compare(full / "src", f"fit-{sp}", sp, d, arm=ARM, base_r5=0.6,
                            base=path(fit(BASE_ARM, sp))).replace(full / f"compare-{sp}.json")
            SR.fake_compare(T / "base", f"zrs-{sp}", sp, {ds: 0.1 for ds in LG.EVAL_ORDER}, arm=BASE_ARM,
                            ).replace(full / f"base-zrs-{sp}.json")                      # zrs: 0.6 against step 1's 0.5
        g = grade(full, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1 and g["primary_gains"] == 1, (g["verdict"], g["losses"])
        assert g["arm"] == ARM and g["base_arm"] == BASE_ARM
        t = (full / "grade.md").read_text(encoding="utf-8")
        assert t.startswith("# zrm's fits against zrs's (the next base): **NOT_ADOPTED**") and DOC in t
        assert "FULL_ROUND10" not in t
        assert json.loads((full / "grade.json").read_text(encoding="utf-8"))["decided_against"] == "zrs's fit of each split"
        # 5. the re-grade: zrs's R@5 as each read's base; the LOSS is within its floor, so zrm is ADOPTED
        r = regrade(null, full / "grade.json", full / "grade-nullx", full)
        assert r["verdict"] == "ADOPT" and r["changed"] == ["L-2wiki hotpotqa LOSS -> WITHIN"], (r["verdict"], r["changed"])
        assert all(x["base"] == 0.6 for x in r["rows"]) and sorted(r["base_compares"]) == sorted(S2.SPLITS)
        t = (full / "grade-nullx.md").read_text(encoding="utf-8")
        assert t.startswith("# zrm's fits against zrs's (the next base), re-graded under the seed null")
        # 6. a comparison decided against zret's fit is INCOMPLETE; a re-grade of another arm's grade stops
        SR.fake_compare(full / "src", "fit-J5", "J5", {"webqsp": 0.05}, arm=ARM, base_r5=0.6,
                        base=path(fit("zret", "J5"))).replace(full / "compare-J5.json")
        g2 = grade(full, full / "grade2", check_fits=False)
        assert g2["verdict"] == "INCOMPLETE" and any("not rel's full_zrs/fits/J5" in p for p in g2["problems"])
        (T / "other.json").write_text(json.dumps({"arm": "zrs", "base_arm": "zret"}), encoding="utf-8")
        SR.must_stop(regrade, null, T / "other.json", T / "x", full)
        # 7. zrs's base R@5 must be the grade's: a base comparison read off another fit stops the re-grade
        SR.fake_compare(T / "base", "zrs-L-squad", "L-squad", {ds: 0.2 for ds in LG.EVAL_ORDER}, arm=BASE_ARM,
                        ).replace(full / "base-zrs-L-squad.json")
        SR.must_stop(regrade, null, full / "grade.json", full / "y", full)
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit) == saved
    log(f"zbase selftest: the fits map as declared; on_zrs swaps relz's names and restores them; the gate passes only "
        f"two ADOPT re-grades of zrs then zrm; the grade is decided against zrs's fits and names this round; the "
        f"re-grade takes zrs's R@5 as each base and calls under the null; a wrong base or arm stops "
        f"({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["gate", "base", "compare", "grade", "regrade"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--grades", default="")
    ap.add_argument("--split", choices=list(S2.SPLITS))
    ap.add_argument("--full-root", default="outputs/zbase")
    ap.add_argument("--null", default="")
    ap.add_argument("--grade")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    try:
        if a.cmd == "gate":
            return gate([x for x in a.grades.split(",") if x])
        if a.cmd in ("base", "compare"):
            if not (a.split and a.out):
                ap.error(f"{a.cmd} needs --split and --out")
            return (base if a.cmd == "base" else compare)(a.split, a.out)
        if a.cmd == "grade":
            v = grade(a.full_root, a.out or str(Path(a.full_root) / "grade"))["verdict"]
            return 1 if v == "INCOMPLETE" else 0
        if a.cmd == "regrade":
            null = [x for x in a.null.split(",") if x]
            if len(null) != 12 or not (a.grade and a.out):
                ap.error("regrade needs the twelve --null files, --grade and --out")
            v = regrade(null, a.grade, a.out, a.full_root)["verdict"]
            return 1 if v == "INCOMPLETE" else 0
    except SystemExit as e:
        if isinstance(e.code, str):
            log(f"zbase {a.cmd}: {e}")
            return 2
        raise
    raise SystemExit("zbase: gate, base, compare, grade, regrade, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
