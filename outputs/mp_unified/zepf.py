"""Round 33's full run (docs/FULL_ROUND33.md): twice the training steps for both models' bases, each decided against
its own base on every split.

  zep    the MLP's base zrc at twice its epochs (zep.py's arm, unchanged): four more fits (L-2wiki, L-squad, J5,
         L-metaqa) in outputs/full_zep/fits; the screen's scr-zep and scr-zep-hp are its L-musique and L-hotpotqa fits.
         Decided against zrc's fits (outputs/full_zrct/fits/S; scr-zrct, scr-zrct-hp).
  zspx   the GNN track's base zsp (zprop.ZProp over zlink.LinkCarveBase) at twice its epochs, on the card: six fits in
         outputs/full_zspx/fits. Decided against zsp's card fits: scr-zspg, scr-zspg-hp (round 31) and four new ones,
         zsp at its own config (zg1.py's base), in outputs/full_zspg/fits.

    python outputs/mp_unified/zepf.py gate --recall outputs/screen/scr-zep-pair-recall.json
    python outputs/mp_unified/zepf.py train --arm zep --split L-2wiki --name L-2wiki --out-root outputs/full_zep/fits --device cuda --host
    python outputs/mp_unified/zepf.py train --arm zspx --split L-2wiki --name L-2wiki --out-root outputs/full_zspx/fits --device cuda --host
    python outputs/mp_unified/zepf.py base --split L-2wiki --name L-2wiki --out-root outputs/full_zspg/fits --device cuda --host
    python outputs/mp_unified/zepf.py read --name L-2wiki --out-root outputs/full_zep/fits --device cuda --host
    python outputs/mp_unified/zepf.py compare --new ... --base ... --out ...
    python outputs/mp_unified/zepf.py grade --arm zep --full-root outputs/full_zep \\
        --reuse L-musique=outputs/screen/scr-zep.json,L-hotpotqa=outputs/screen/scr-zep-hp.json
    python outputs/mp_unified/zepf.py grade --arm zspx --full-root outputs/full_zspx
    python outputs/mp_unified/zepf.py --selftest

The re-grades are nullx.py's (regrade), with each arm's base R@5 from its base fits compared with step 1's.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zep as ZE  # noqa: E402
import zg1 as G  # noqa: E402

ZO, ZP, ZL, RM = ZE.ZO, G.ZP, G.ZL, ZE.RM
Z, R, S2, S, LG, LC, SR = ZE.Z, ZE.R, ZE.S2, ZE.S, ZE.LG, ZE.LC, ZE.SR
log = S.log

if Z is not G.Z or RM is not G.RM or S is not G.S:
    raise SystemExit("zepf: zep.py and zg1.py do not share relz, rmatch and lean_screen")

ZSPX = "zspx"
S.ARMS.update({ZSPX: (ZP.ZProp, ZL.LinkCarveBase)})
ZSPG_FITS = dict(G.ZSPG_FITS)                         # round 31's card fits of zsp on L-musique and L-hotpotqa


def zspg_fit(split):
    """zsp's card fit of a split: round 31's on L-musique and L-hotpotqa, this run's (outputs/full_zspg) elsewhere."""
    return ZSPG_FITS.get(split, ("full_zspg", "fits", split))


ZSPG_SCREENS = {sp: f"outputs/zg1/base-zspg-{sp}.json" for sp in ZSPG_FITS}
GRADES = {   # arm -> (base arm, base fits (screens), base_fit, decided_against, round label)
    "zep": ("zrc", ZO.ZRC_FITS, ZO.ZRC_SCREENS, ZO.zrc_fit, ZO.DECIDED),
    ZSPX: ("zsp", ZSPG_FITS, ZSPG_SCREENS, zspg_fit, "zsp's card fit of each split"),
}
CONFIG = ZE.config_x()                                # 2e-3:1e-4:0.1:16:4


def check_arms():
    ZE.check_base()
    if S.ARMS.get(ZSPX) != S.ARMS.get("zsp") or S.ARMS.get("zsp") != (ZP.ZProp, ZL.LinkCarveBase):
        raise SystemExit(f"zepf: the arm {ZSPX} is {S.ARMS.get(ZSPX)}, not zsp's model on zlink's carve")


def stamp(argv, body):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zepf_sha256": LC.sha_src(__file__), "zep_sha256": LC.sha_src(ZE.__file__),
                    "zg1_sha256": LC.sha_src(G.__file__), "zprop_sha256": LC.sha_src(ZP.__file__),
                    "zepf": body})
        LC.write_json(sj, rec)


def train(argv, split):
    """zep: zep.py's train, unchanged. zspx: rmatch.py's train for zsp's model and carve at twice its epochs, on the
    card (zg1.py's on_card)."""
    check_arms()
    arm = R.arm_of(argv)
    if arm == "zep":
        return ZE.train(argv, split)
    if arm != ZSPX:
        raise SystemExit(f"zepf: train takes --arm zep or {ZSPX}, not {arm}")
    if "--config" in argv:
        raise SystemExit("zepf: the config is the arm's own; drop --config")
    argv = G.on_card(list(argv)) + ["--config", CONFIG]
    rc = RM.train(argv, split)
    stamp(argv, {"model": "zprop.ZProp over zlink.LinkCarveBase (zsp's arm)", "config": CONFIG,
                 "zsp_config": ZE.ZRC_CONFIG, "device": G.DEVICE, "zlink_records": ZL.built_records()})
    return rc


def base(argv, split):
    """zsp at its own config on the card (zg1.py's base): the base fits of the splits round 31 did not fit."""
    if split in ZSPG_FITS:
        raise SystemExit(f"zepf: zsp's card fit of {split} is round 31's ({'/'.join(ZSPG_FITS[split])})")
    rc = G.base(argv, split)
    stamp(argv, {"base_of": ZSPX, "arm": "zsp", "config": "zsp's own (lean_gpu.CONFIG)", "device": G.DEVICE})
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm == "zep":
        return ZE.read(argv)
    if arm in (ZSPX, "zsp"):
        return RM.read(G.on_card(argv))
    raise SystemExit(f"zepf: {name} was trained as {arm}, not zep, {ZSPX} or zsp")


def compare(rest):
    return S2.main(["compare"] + rest)


@contextlib.contextmanager
def on_base(arm):
    if arm not in GRADES:
        raise SystemExit(f"zepf: --arm takes {' or '.join(GRADES)}, not {arm}")
    b, fits, screens, fit_of, _d = GRADES[arm]
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, b, fits, screens, fit_of
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def grade(arm, full_root, reuse, out, check_fits=True):
    with on_base(arm):
        rec = Z.grade(full_root, reuse, out, check_fits)
    js = Path(out).with_suffix(".json")
    if js.exists():
        r = json.loads(js.read_text(encoding="utf-8"))
        r.update({"decided_against": GRADES[arm][4], "round": "thirty-third (full run)", "config": CONFIG,
                  "declared_in": "docs/FULL_ROUND33.md", "zepf_sha256": LC.sha_src(__file__)})
        LC.write_json(js, r)
    md = Path(out).with_suffix(".md")
    if md.exists():
        md.write_text(md.read_text(encoding="utf-8").replace("(docs/FULL_ROUND10.md)", "(docs/FULL_ROUND33.md)"),
                      encoding="utf-8")
    return rec


def gate(recall_file):
    """0 when round 33's re-call (zep against zrc) is PROMISING, 1 otherwise; a missing or foreign record stops."""
    p = Path(recall_file)
    if not p.exists():
        raise SystemExit(f"zepf gate: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("arm") != "zep" or rec.get("decided_against") != ZO.DECIDED or rec.get("round") != 33:
        raise SystemExit(f"zepf gate: {p} is arm {rec.get('arm')}'s against {rec.get('decided_against')!r}, round "
                         f"{rec.get('round')}; not round 33's re-call")
    ok = rec.get("verdict") == "PROMISING"
    log(f"zepf gate: round 33's re-call {rec.get('verdict')} -> {'the full run starts' if ok else 'no full run'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    check_arms()
    assert CONFIG == "2e-3:1e-4:0.1:16:4" and LG.CONFIG == ZE.ZRC_CONFIG, (CONFIG, LG.CONFIG)
    assert S.ARMS[ZSPX] == S.ARMS["zsp"] == (ZP.ZProp, ZL.LinkCarveBase) and S.ARMS["zep"] == S.ARMS["zrc"]
    assert zspg_fit("L-musique") == ("screen", "fits", "scr-zspg") and zspg_fit("J5") == ("full_zspg", "fits", "J5")
    assert ZO.zrc_fit("J5") == ("full_zrct", "fits", "J5") and ZO.zrc_fit("L-hotpotqa")[-1] == "scr-zrct-hp"
    SR.must_stop(train, ["train", "--arm", "zsp", "--name", "x"], "L-2wiki")
    SR.must_stop(train, ["train", "--arm", ZSPX, "--name", "x", "--config", ZE.ZRC_CONFIG], "L-2wiki")
    SR.must_stop(base, ["train", "--name", "x"], "L-musique")
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    for a in GRADES:
        with on_base(a):
            assert Z.ARM == a and Z.BASE_ARM == GRADES[a][0] and Z.rel_fit("L-squad")[0] in ("full_zrct", "full_zspg")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    tmp = Path(tempfile.mkdtemp(prefix="zepf_"))
    try:
        # the gate: PROMISING -> 0, anything else -> 1, a foreign record stops
        for v, want in (("PROMISING", 0), ("NO_GAIN", 1), ("MIXED", 1)):
            LC.write_json(tmp / f"{v}.json", {"arm": "zep", "decided_against": ZO.DECIDED, "round": 33, "verdict": v})
            assert gate(tmp / f"{v}.json") == want
        LC.write_json(tmp / "f.json", {"arm": "zeh", "decided_against": ZO.DECIDED, "round": 33, "verdict": "PROMISING"})
        SR.must_stop(gate, tmp / "f.json")
        SR.must_stop(gate, tmp / "none.json")
        # the grades: each arm against its own base fits; a comparison against another base is a problem
        for arm, deltas, want in (("zep", {"webqsp": 0.0215}, "ADOPT"), (ZSPX, {"2wiki": -0.03}, "NOT_ADOPTED")):
            full = tmp / arm
            fit_of = GRADES[arm][3]
            for sp in S2.SPLITS:
                SR.fake_compare(full, f"compare-{sp}", sp, deltas, arm=arm, base_r5=0.6,
                                base="/".join(("outputs",) + fit_of(sp)))
            g = grade(arm, full, {}, full / "grade", check_fits=False)
            assert g["verdict"] == want and g["arm"] == arm and g["base_arm"] == GRADES[arm][0], (arm, g["verdict"],
                                                                                                g["problems"])
            other = ZSPX if arm == "zep" else "zep"
            g2 = grade(other, full, {}, full / "grade-x", check_fits=False)
            assert g2["verdict"] == "INCOMPLETE", g2["verdict"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"zepf selftest: config {CONFIG} for zep and {ZSPX}; zsp's card base fits; gate and grades ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "base":
        return base(["train"] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "gate":
        gp = argparse.ArgumentParser()
        gp.add_argument("--recall", required=True)
        return gate(gp.parse_args(rest).recall)
    if k.cmd == "grade":
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=sorted(GRADES))
        gp.add_argument("--full-root", required=True)
        gp.add_argument("--reuse", default="")
        g = gp.parse_args(rest)
        reuse = dict(x.split("=", 1) for x in g.reuse.split(",") if x)
        rec = grade(g.arm, g.full_root, reuse, Path(g.full_root) / "grade")
        return 1 if rec["verdict"] == "INCOMPLETE" else 0
    raise SystemExit("zepf: gate, train, base, read, compare, grade, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
