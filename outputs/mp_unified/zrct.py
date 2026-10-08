"""Screens, eighteenth round (docs/SCREENS.md; full run docs/FULL_ROUND18.md): zrc trained on its own entries.

zrc is zrm with rmatch's chain entries kept, per row and seed bucket, by w0(c) m_c(v) in place of m_c(v)
(outputs/mp_unified/zrc.py, the seventeenth round). Round seventeen stopped at its identity gate: on metaqa's fit carve
zrc's build and rmatch's differ in 2 of 45,883,297 entries, so zrm's fits are not zrc's. The diagnosis zrcd read zrm's
screen fits with zrc's entries: its re-call was PROMISING and every read but webqsp's was zrm's bit for bit, so by its
declared rule this round trains zrc: zrm's model, settings and training (zrm.py, rmatch.py's train) over zrc's builds
(outputs/zrc/cache), under the arm zrc that zrc.py registers, (zrm.ZRM, zrc.ChainCarveZRC).

    python outputs/mp_unified/zrct.py train --split L-musique --name scr-zrct --arm zrc --device cuda --host
    python outputs/mp_unified/zrct.py read --name scr-zrct --device cuda --host
    python outputs/mp_unified/zrct.py compare --new outputs/screen/fits/scr-zrct \\
        --base outputs/screen/fits/scr-zrm,outputs/screen/fits/scr-zret,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zrct
    python outputs/mp_unified/zrct.py pair --screens outputs/screen/scr-zrct.json,outputs/screen/scr-zrct-hp.json \\
        --out outputs/screen/scr-zrct-pair
    python outputs/mp_unified/zrct.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zrct-pair.json \\
        --out outputs/screen/scr-zrct-pair-recall
    python outputs/mp_unified/zrct.py gate --recall outputs/screen/scr-zrct-pair-recall.json \\
        --base outputs/zbase/grade-nullx.json
    python outputs/mp_unified/zrct.py grade --full-root outputs/full_zrct \\
        --reuse L-musique=outputs/screen/scr-zrct.json,L-hotpotqa=outputs/screen/scr-zrct-hp.json
    python outputs/mp_unified/zrct.py --selftest

Each read is decided against zrm's fit of its split: zrc.py's pair, re-call and grade (relz.py's, with zrm's fits in place
of rel's; the re-call's and re-grade's base R@5 from outputs/zrc/base-zrm-<split>.json), named for this round. zrc.py's
compare also files each read's arrays against zrm's (-same): here they may differ, since the training carves differ in
two entries, except in L-metaqa's fit, which trains no typed graph and must be zrm's bit for bit. The full run starts
only when the screen's re-call is PROMISING and zrm is the base (docs/BASE_ZRM_ZRS.md re-graded ADOPT): `gate`.

Speed: the kept set is chosen at build time from the question's embedding and its pool's relations. Any latency figure
for this arm is cold (8216ffe): each question from scratch, the walk, w0, the selection and the forward included.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zrc as ZC  # noqa: E402

ZM, RM = ZC.ZM, ZC.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
SR = Z.SR
S3 = ZM.S3
log = S.log

ARM = ZC.ARM                     # "zrc": (zrm.ZRM, zrc.ChainCarveZRC)
BASE_ARM = ZC.BASE_ARM           # "zrm"
NAMES = (("section 2 and the seventeenth round", "section 2 and the eighteenth round"),
         ("docs/FULL_ROUND17.md", "docs/FULL_ROUND18.md"))


# ── train and read ───────────────────────────────────────────────────────────


def train(argv, split):
    """rmatch.py's train (zrm's settings and model) for the arm zrc, over zrc's builds; zrc's records stamped."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zrct: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZM.ZRM, ZC.ChainCarveZRC):
        raise SystemExit(f"zrct: the arm {ARM} is {S.ARMS.get(ARM)}, not zrm's model over zrc's builds")
    rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zrct_sha256": LC.sha_src(__file__), "zrc_sha256": LC.sha_src(ZC.__file__),
                    "zrm_sha256": LC.sha_src(ZM.__file__), "lean_screen3_sha256": LC.sha_src(S3.__file__),
                    "zrc": {"model": "zrm.ZRM (rmatch.ChainMatch over lean_screen3.ZRet)",
                            "carve": "zrc.ChainCarveZRC", "entries": "outputs/zrc/cache", "rule": ZC.RULE},
                    "zrc_records": RM.built_records(ZC.CH_OUT)})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    return ZC.read(argv)


def compare(rest):
    return ZC.compare(rest)


# ── zrc.py's pair, re-call and grade, named for this round ───────────────────


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in NAMES:
            t = t.replace(a, b)
        md.write_text(t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"round": "eighteenth", "zrct_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(screens, out):
    rec = ZC.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zrm_screens=None):
    rec = ZC.recall(null_files, pair_file, out, zrm_screens)
    restamp(out)
    return rec


def grade(full_root, reuse, out, check_fits=True):
    rec = ZC.grade(full_root, reuse, out, check_fits)
    restamp(out)
    return rec


# ── the full run's gate ──────────────────────────────────────────────────────


def gate(recall_file, base_file):
    """0 when the screen's re-call is PROMISING and zrm is the base (docs/BASE_ZRM_ZRS.md's re-grade ADOPT), 1
    otherwise; a missing file, or one that is not the declared record, stops (2)."""
    got = []
    for fn, arm, key, want in ((recall_file, ARM, "decided_against", "zrm's fit of each split"),
                               (base_file, "zrm", "decided_against", "zrs's fit of each split")):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zrct gate: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != arm or rec.get(key) != want:
            raise SystemExit(f"zrct gate: {p} is arm {rec.get('arm')}'s, {key} {rec.get(key)!r}; not the declared "
                             "record")
        got.append(rec.get("verdict"))
    if "null_splits" not in json.loads(Path(base_file).read_text(encoding="utf-8")):
        raise SystemExit(f"zrct gate: {base_file} is not a re-grade under the null")
    ok = got[0] == "PROMISING" and got[1] == "ADOPT"
    log(f"zrct gate: the re-call {got[0]}, zrm against zrs {got[1]} -> {'the full run starts' if ok else 'no full run'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    # 1. the arm: zrm's model over zrc's builds; train takes no other arm
    assert S.ARMS[ARM] == (ZM.ZRM, ZC.ChainCarveZRC) and ZC.ChainCarveZRC.CH_ROOT == ZC.CH_OUT
    assert ARM == "zrc" and BASE_ARM == "zrm"
    SR.must_stop(train, ["--arm", "zrm", "--name", "x"], "L-musique")
    SR.must_stop(train, ["--name", "x"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        # 2. the gate: PROMISING and ADOPT -> 0; anything else -> 1; a wrong or missing record stops
        def rec(name, **kw):
            p = T / f"{name}.json"
            p.write_text(json.dumps(kw), encoding="utf-8")
            return str(p)

        rc_p = rec("rp", arm="zrc", decided_against="zrm's fit of each split", verdict="PROMISING")
        rc_m = rec("rm", arm="zrc", decided_against="zrm's fit of each split", verdict="MIXED")
        b_a = rec("ba", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT", null_splits=["J5"])
        b_n = rec("bn", arm="zrm", decided_against="zrs's fit of each split", verdict="NOT_ADOPTED", null_splits=["J5"])
        b_f = rec("bf", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT")         # filed, not nullx
        b_z = rec("bz", arm="zrm", decided_against="zret's fit of each split", verdict="ADOPT", null_splits=["J5"])
        assert gate(rc_p, b_a) == 0 and gate(rc_m, b_a) == 1 and gate(rc_p, b_n) == 1
        SR.must_stop(gate, rc_p, b_f)
        SR.must_stop(gate, rc_p, b_z)
        SR.must_stop(gate, b_a, rc_p)
        SR.must_stop(gate, rc_p, str(T / "none.json"))
        # 3. zrc.py's pair, re-call and grade, against zrm's fits, named for this round
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        zs = {sp: str(SR.fake_compare(T / ZC.zrm_fit(sp)[0], ZC.zrm_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZC.ZRM_FITS}                                            # zrm's fits against step 1's (0.6)
        zb = {sp: "/".join(("outputs",) + ZC.zrm_fit(sp)) for sp in ZC.ZRM_FITS}
        za = SR.fake_compare(T / "z", "scr-zrct", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zrct-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        assert json.loads((T / "pz.json").read_text(encoding="utf-8"))["round"] == "eighteenth"
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "eighteenth round" in t and "seventeenth" not in t and "| zrm R@5 | zrc R@5 |" in t
        assert json.loads((T / "rz.json").read_text(encoding="utf-8"))["decided_against"] == "zrm's fit of each split"
        assert gate(str(T / "rz.json"), b_a) == 0
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in ZC.ZRM_FITS:
                SR.fake_compare(full / "src", f"fit-{sp}", sp, {"webqsp": 0.03} if sp == "J5" else {}, arm=ARM,
                                base="/".join(("outputs",) + ZC.zrm_fit(sp))).replace(full / f"compare-{sp}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1 and g["arm"] == ARM, (g["verdict"], g["losses"])
        t = (full / "grade.md").read_text(encoding="utf-8")
        assert "docs/FULL_ROUND18.md" in t and "FULL_ROUND17" not in t
    log(f"zrct selftest: the arm is zrm's model over zrc's builds and train takes no other; the gate needs a PROMISING "
        f"re-call and zrm ADOPTED against zrs under the null; zrc.py's pair, re-call and grade run against zrm's fits "
        f"and name this round ({time.time() - t0:.1f}s): ok")
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
        return train([k.cmd] + rest, k.split)          # as zrm.py's main
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "gate":
        gp = argparse.ArgumentParser()
        gp.add_argument("--recall", required=True)
        gp.add_argument("--base", required=True)
        g = gp.parse_args(rest)
        try:
            return gate(g.recall, g.base)
        except SystemExit as e:
            log(f"zrct gate: {e}")
            return 2
    if k.cmd in ("pair", "recall", "grade"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--full-root")
        gp.add_argument("--reuse", default="")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair": g.screens and g.out, "recall": g.pair and len(null) == 4 and g.out, "grade": g.full_root}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair needs --screens A,B and --out; recall --pair, the four --null files and --out; "
                     "grade --full-root")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            elif k.cmd == "recall":
                recall(null, g.pair, g.out)
            else:
                reuse = dict(x.split("=", 1) for x in g.reuse.split(",") if x)
                v = grade(g.full_root, reuse, g.out or str(Path(g.full_root) / "grade"))["verdict"]
                return 1 if v == "INCOMPLETE" else 0
        except SystemExit as e:
            log(f"zrct {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zrct: train, read, compare, pair, recall, gate, grade, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
