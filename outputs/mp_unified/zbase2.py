"""The next base now that zrc and zkind are both ADOPTED against zrm (docs/BASE_ZRC_ZKIND.md, declared 9 October about
08:00, before any comparison of the two). zrc (round eighteen, docs/FULL_ROUND18.md, re-graded ADOPT 03:35) keeps the 64
entries w0 ranks highest in each typed row; zkind (round nineteen, docs/FULL_ROUND19.md, re-graded ADOPT 07:48) adds a
head over the typed graph's kinds. zrc is the incumbent: zkind becomes the base when, against zrc's fit of each split, at
least one primary read GAINs and none of the 36 LOSEs under the seed null over every split (nullx.py); otherwise zrc is
the base. zbase.py's rule, unchanged, for this pair (zbase.py is frozen; this is its copy for zrc and zkind).

    python outputs/mp_unified/zbase2.py gate --grades outputs/full_zrct/grade-nullx.json,outputs/full_zkind/grade-nullx.json
    python outputs/mp_unified/zbase2.py base --split J5 --out outputs/zbase2/base-zrc-J5
    python outputs/mp_unified/zbase2.py compare --split J5 --out outputs/zbase2/compare-J5
    python outputs/mp_unified/zbase2.py grade --full-root outputs/zbase2 --out outputs/zbase2/grade
    python outputs/mp_unified/zbase2.py regrade --null F1,...,F12 --grade outputs/zbase2/grade.json \\
        --out outputs/zbase2/grade-nullx
    python outputs/mp_unified/zbase2.py --selftest

Fits: zkind's and zrc's screen fits on L-musique and L-hotpotqa (scr-zkind, scr-zkind-hp; scr-zrct, scr-zrct-hp), their
full runs' fits on the other four splits (outputs/full_zkind/fits, outputs/full_zrct/fits); zrm's as in both rounds'
grades (scr-zrm, scr-zrm-hp, outputs/full_zrm/fits). `compare` is lean_screen's: zkind's fit against zrc's (decides),
zrm's and step 1's (reported only). `base` compares zrc's fit with step 1's alone; nullx.py takes those as each read's
base R@5. The grade is relz.py's and the re-grade nullx.py's, unchanged, under zkind's name with zrc's fits in place of
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

ARM, BASE_ARM = "zkind", "zrc"
DOC = "docs/BASE_ZRC_ZKIND.md"
FOLDER = {"zkind": "full_zkind", "zrc": "full_zrct", "zrm": "full_zrm"}
SCREEN = {"L-musique": {"zkind": ("screen", "fits", "scr-zkind"), "zrc": ("screen", "fits", "scr-zrct"),
                        "zrm": ("screen", "fits", "scr-zrm")},
          "L-hotpotqa": {"zkind": ("screen", "fits", "scr-zkind-hp"), "zrc": ("screen", "fits", "scr-zrct-hp"),
                         "zrm": ("screen", "fits", "scr-zrm-hp")}}
DECIDED = "zrc's fit of each split"


def fit(arm, split):
    """An arm's fit of a split, as parts under outputs/."""
    if split not in S2.SPLITS or arm not in ("zkind", "zrc", "zrm", "step1"):
        raise SystemExit(f"zbase2: no fit of {arm} on {split}")
    if arm == "step1":
        return ("step1", "fits", split)
    return SCREEN.get(split, {}).get(arm) or (FOLDER[arm], "fits", split)


def path(parts):
    return "/".join(("outputs",) + tuple(parts))


# ── the gate ─────────────────────────────────────────────────────────────────


def gate(grades):
    """0 when zrc's and zkind's re-grades are both ADOPT (the comparison runs), 1 when either is not; a missing file,
    or one of another arm, stops (2)."""
    if len(grades) != 2:
        raise SystemExit("zbase2 gate: two re-grades, zrc's then zkind's")
    got = {}
    for want, fn in zip((BASE_ARM, ARM), grades):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zbase2 gate: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != want or "null_splits" not in rec:
            raise SystemExit(f"zbase2 gate: {p} is arm {rec.get('arm')}'s, not {want}'s re-grade under the null")
        got[want] = rec.get("verdict")
    ok = all(v == "ADOPT" for v in got.values())
    log(f"zbase2 gate: zrc {got[BASE_ARM]}, zkind {got[ARM]} -> {'compare zkind with zrc' if ok else 'no comparison'}")
    return 0 if ok else 1


# ── comparisons ──────────────────────────────────────────────────────────────


def compare(split, out):
    """zkind's fit of split against zrc's (decides), zrm's and step 1's (reported only)."""
    return S2.main(["compare", "--new", path(fit(ARM, split)), "--base",
                    ",".join(path(fit(a, split)) for a in (BASE_ARM, "zrm", "step1")), "--out", str(out)])


def base(split, out):
    """zrc's fit of split against step 1's alone: each read's base R@5 for nullx.py."""
    return S2.main(["compare", "--new", path(fit(BASE_ARM, split)), "--base", path(fit("step1", split)),
                    "--out", str(out)])


# ── relz.py's grade and nullx.py's re-grade, decided against zrc's fits ──────


ZRC_FITS = {sp: SCREEN[sp][BASE_ARM] for sp in SCREEN}


def zrc_fit(split):
    return fit(BASE_ARM, split)


@contextlib.contextmanager
def on_zrc():
    """relz.py's grade under zkind's name, each read decided against zrc's fit of its split; relz restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit = ARM, BASE_ARM, ZRC_FITS, zrc_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit = saved


def restamp(out, title=None):
    """A record written under zkind's name against zrc's fits: this round's file named, this file's sha added."""
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
        rec.update({"decided_against": DECIDED, "zbase2_sha256": LC.sha_src(__file__), "doc": DOC})
        LC.write_json(js, rec)


def grade(full_root, out, check_fits=True):
    """relz.py's grade: ADOPT (zkind becomes the base) when a primary read GAINs and none of the 36 LOSEs."""
    with on_zrc():
        rec = Z.grade(full_root, {}, out, check_fits)
    restamp(out, ("# Full run of zkind against zrc:", "# zkind's fits against zrc's (the next base):"))
    return rec


def compares(root):
    return {sp: str(Path(root) / f"base-zrc-{sp}.json") for sp in S2.SPLITS}


def regrade(null_files, grade_file, out, base_root):
    """nullx.py's re-grade, each read's base R@5 zrc's (base-zrc-<split>.json under base_root)."""
    gp = Path(grade_file)
    if not gp.exists():
        raise SystemExit(f"zbase2: {gp} missing")
    g = json.loads(gp.read_text(encoding="utf-8"))
    if g.get("arm") != ARM or g.get("base_arm") != BASE_ARM or g.get("decided_against") != DECIDED:
        raise SystemExit(f"zbase2: {gp} is {g.get('arm')} against {g.get('base_arm')}, not zkind against zrc's fits")
    rec = NX.regrade(null_files, grade_file, out, compares(base_root))
    restamp(out, ("# Full run of zkind re-graded under",
                  "# zkind's fits against zrc's (the next base), re-graded under"))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    # 1. the fits: screen fits on the two screen splits, full runs' elsewhere; zrc's folder is full_zrct
    assert fit("zkind", "L-musique") == ("screen", "fits", "scr-zkind") and fit("zrc", "L-hotpotqa") == (
        "screen", "fits", "scr-zrct-hp")
    assert fit("zrm", "L-musique") == ("screen", "fits", "scr-zrm")
    assert fit("zrm", "L-hotpotqa") == ("screen", "fits", "scr-zrm-hp")
    assert fit("zrc", "J5") == ("full_zrct", "fits", "J5") and fit("step1", "L-2wiki") == ("step1", "fits", "L-2wiki")
    assert path(fit("zkind", "L-metaqa")) == "outputs/full_zkind/fits/L-metaqa"
    assert path(fit("zrm", "L-squad")) == "outputs/full_zrm/fits/L-squad"
    SR.must_stop(fit, "zrs", "J5")
    SR.must_stop(fit, "zkind", "L-nq")
    # 2. on_zrc swaps relz's names and restores them, after an error too
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit
    with on_zrc():
        assert Z.ARM == ARM and Z.BASE_ARM == BASE_ARM and Z.rel_fit("J5") == ("full_zrct", "fits", "J5")
        assert Z.rel_fit("L-musique") == ("screen", "fits", "scr-zrct")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit) == saved
    try:
        with on_zrc():
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

        # 3. the gate: both ADOPT -> 0; zkind NOT_ADOPTED or INCOMPLETE -> 1; the wrong arm, order or file -> stop
        zc, zk = regr("zc", "zrc", "ADOPT"), regr("zk", "zkind", "ADOPT")
        assert gate([zc, zk]) == 0
        assert gate([zc, regr("zk2", "zkind", "NOT_ADOPTED")]) == 1
        assert gate([zc, regr("zk3", "zkind", "INCOMPLETE")]) == 1
        assert gate([regr("zc2", "zrc", "NOT_ADOPTED"), zk]) == 1
        SR.must_stop(gate, [zk, zc])
        SR.must_stop(gate, [zc, str(T / "none.json")])
        SR.must_stop(gate, [zc])
        (T / "zk4.json").write_text(json.dumps({"arm": "zkind", "verdict": "ADOPT"}), encoding="utf-8")  # a filed grade
        SR.must_stop(gate, [zc, str(T / "zk4.json")])
        # 4. the grade, decided against zrc's fits: J5 webqsp (primary) GAINs, L-2wiki hotpotqa LOSEs at 0.0075 only
        full = T / "zbase2"
        full.mkdir()
        null = NX.fake_null(T / "null", "n", {("L-2wiki", "hotpotqa"): (-0.010, 0.012)})              # floor 0.0221
        for sp in S2.SPLITS:
            d = {"J5": {"webqsp": 0.05}, "L-2wiki": {"hotpotqa": -0.0147}}.get(sp, {})
            SR.fake_compare(full / "src", f"fit-{sp}", sp, d, arm=ARM, base_r5=0.6,
                            base=path(fit(BASE_ARM, sp))).replace(full / f"compare-{sp}.json")
            SR.fake_compare(T / "base", f"zrc-{sp}", sp, {ds: 0.1 for ds in LG.EVAL_ORDER}, arm=BASE_ARM,
                            ).replace(full / f"base-zrc-{sp}.json")                      # zrc: 0.6 against step 1's 0.5
        g = grade(full, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1 and g["primary_gains"] == 1, (g["verdict"], g["losses"])
        assert g["arm"] == ARM and g["base_arm"] == BASE_ARM
        t = (full / "grade.md").read_text(encoding="utf-8")
        assert t.startswith("# zkind's fits against zrc's (the next base): **NOT_ADOPTED**") and DOC in t
        assert "FULL_ROUND10" not in t
        assert json.loads((full / "grade.json").read_text(encoding="utf-8"))["decided_against"] == DECIDED
        # 5. the re-grade: zrc's R@5 as each read's base; the LOSS is within its floor, so zkind is ADOPTED
        r = regrade(null, full / "grade.json", full / "grade-nullx", full)
        assert r["verdict"] == "ADOPT" and r["changed"] == ["L-2wiki hotpotqa LOSS -> WITHIN"], (r["verdict"], r["changed"])
        assert all(x["base"] == 0.6 for x in r["rows"]) and sorted(r["base_compares"]) == sorted(S2.SPLITS)
        t = (full / "grade-nullx.md").read_text(encoding="utf-8")
        assert t.startswith("# zkind's fits against zrc's (the next base), re-graded under the seed null")
        # 6. a comparison decided against zrm's fit is INCOMPLETE; a re-grade of another arm's grade stops
        SR.fake_compare(full / "src", "fit-J5", "J5", {"webqsp": 0.05}, arm=ARM, base_r5=0.6,
                        base=path(fit("zrm", "J5"))).replace(full / "compare-J5.json")
        g2 = grade(full, full / "grade2", check_fits=False)
        assert g2["verdict"] == "INCOMPLETE" and any("not rel's full_zrct/fits/J5" in p for p in g2["problems"])
        (T / "other.json").write_text(json.dumps({"arm": "zrc", "base_arm": "zrm"}), encoding="utf-8")
        SR.must_stop(regrade, null, T / "other.json", T / "x", full)
        # 7. zrc's base R@5 must be the grade's: a base comparison read off another fit stops the re-grade
        SR.fake_compare(T / "base", "zrc-L-squad", "L-squad", {ds: 0.2 for ds in LG.EVAL_ORDER}, arm=BASE_ARM,
                        ).replace(full / "base-zrc-L-squad.json")
        SR.must_stop(regrade, null, full / "grade.json", full / "y", full)
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.rel_fit) == saved
    log(f"zbase2 selftest: the fits map as declared; on_zrc swaps relz's names and restores them; the gate passes only "
        f"two ADOPT re-grades of zrc then zkind; the grade is decided against zrc's fits and names this round; the "
        f"re-grade takes zrc's R@5 as each base and calls under the null; a wrong base or arm stops "
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
    ap.add_argument("--full-root", default="outputs/zbase2")
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
            log(f"zbase2 {a.cmd}: {e}")
            return 2
        raise
    raise SystemExit("zbase2: gate, base, compare, grade, regrade, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
