"""Round 33 (docs/SCREENS.md, zep): does zrc, the MLP's base, gain from twice its training steps?

D3 (docs/DIAG_DATA_SCALE.md) found zrc DATA_LIMITED: refit on every 2nd question of its fit carves it loses on 4 of
12 reads under the seed null. Its half fits ran zrc's epochs, so they also took half the optimiser steps. This round
separates the two:
  zep   zrc on its full fit carves, twice zrc's epochs, SWA from twice zrc's first SWA epoch (the same share of the
        run averaged). The screen: decided against zrc's fits (scr-zrct, scr-zrct-hp) under the screens' rule.
  zeh   zrc on every 2nd question (zds.subset(2), D3's half), twice zrc's epochs: as many steps as zrc. Reported for
        D3's reading only: if it still LOSEs against zrc, the loss is the data's, not the steps'.
Everything else is zrc's: zrm.ZRM over zrc.ChainCarveZRC, rmatch.py's train, zrm's settings, seed 0, p@swa. No row,
column, pool, graph or score is new.

    python outputs/mp_unified/zep.py train --split L-musique --name scr-zep --arm zep --device cuda --host
    python outputs/mp_unified/zep.py train --split L-musique --name dsx-zeh --arm zeh --device cuda --host
    python outputs/mp_unified/zep.py read --name scr-zep --device cuda --host
    python outputs/mp_unified/zep.py compare --new outputs/screen/fits/scr-zep \\
        --base outputs/screen/fits/scr-zrct,outputs/step1/fits/L-musique --out outputs/screen/scr-zep
    python outputs/mp_unified/zep.py pair --arm zep --screens A.json,B.json --out outputs/screen/scr-zep-pair
    python outputs/mp_unified/zep.py recall --arm zep --null N1,N2,N3,N4 --pair P.json --out outputs/screen/scr-zep-pair-recall
    python outputs/mp_unified/zep.py --selftest
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import zds as ZD  # noqa: E402

ZO, ZC, ZM, RM = ZD.ZO, ZD.ZC, ZD.ZM, ZD.RM
Z, R, S2, S, LG, LC, SR = ZD.Z, ZD.R, ZD.S2, ZD.S, ZD.LG, ZD.LC, ZD.SR
log = S.log

BASE_ARM = "zrc"
ARMS = {"zep": 1, "zeh": 2}           # arm -> stride of the fit carves
X = 2                                 # epochs and swa_from, times zrc's
ZRC_CONFIG = "2e-3:1e-4:0.1:8:2"      # lean_gpu.CONFIG, zrc's settings (lr:wd:dropout:epochs:swa_from)
for _a in ARMS:
    S.ARMS.update({_a: (ZM.ZRM, ZC.ChainCarveZRC)})


def config_x(base=ZRC_CONFIG, x=X):
    lr, wd, dr, ep, sw = base.split(":")
    return f"{lr}:{wd}:{dr}:{int(ep) * x}:{int(sw) * x}"


def check_base():
    if LG.CONFIG != ZRC_CONFIG:
        raise SystemExit(f"zep: lean_gpu.CONFIG is {LG.CONFIG}, not zrc's {ZRC_CONFIG}")
    for a in ARMS:
        if S.ARMS.get(a) != (ZM.ZRM, ZC.ChainCarveZRC) or S.ARMS.get(BASE_ARM) != S.ARMS[a]:
            raise SystemExit(f"zep: the arm {a} is {S.ARMS.get(a)}, not zrc's model on zrc's carve")


def train(argv, split):
    """rmatch.py's train for zep / zeh: zrc's model and builds at twice zrc's epochs (zeh on every 2nd question)."""
    check_base()
    arm = R.arm_of(argv)
    if arm not in ARMS:
        raise SystemExit(f"zep: train takes --arm {' or '.join(ARMS)}, not {arm}")
    if "--config" in argv:
        raise SystemExit("zep: the config is the arm's own; drop --config")
    cfg = config_x()
    argv = list(argv) + ["--config", cfg]
    every = ARMS[arm]
    if every == 1:
        rc = RM.train(argv, split)
        last = None
    else:
        with ZD.subset(every) as sb:
            rc = RM.train(argv, split)
        last = sb.last
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zep_sha256": LC.sha_src(__file__), "zds_sha256": LC.sha_src(ZD.__file__),
                    "zrc_sha256": LC.sha_src(ZC.__file__), "zrm_sha256": LC.sha_src(ZM.__file__),
                    "zep": {"model": "zrm.ZRM over zrc.ChainCarveZRC (zrc's arm)", "config": cfg,
                            "zrc_config": ZRC_CONFIG, "every": every, "last_fit": last}})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm not in ARMS:
        raise SystemExit(f"zep: {name} was trained as {arm}, not {' or '.join(ARMS)}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


@contextlib.contextmanager
def on_zrc(arm):
    if arm not in ARMS:
        raise SystemExit(f"zep: --arm takes {' or '.join(ARMS)}, not {arm}")
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_ARM, ZO.ZRC_FITS, ZO.ZRC_SCREENS, ZO.zrc_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrc's screen fits scr-zrct and scr-zrct-hp)"),
       ("(rel's screen fits)", "(zrc's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrc R@5 | new R@5 |"),
       ("section 2 and the tenth round", "section 2 and round 33"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"decided_against": ZO.DECIDED, "round": 33, "arm_every": ARMS[arm], "zep_config": config_x(),
                    "zep_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_zrc(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out):
    with on_zrc(arm):
        rec = Z.recall(null_files, pair_file, out, None)
    restamp(out, arm)
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    check_base()
    assert config_x() == "2e-3:1e-4:0.1:16:4", config_x()
    LG.bind_device_ops()
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zep", "--name", "x", "--config", ZRC_CONFIG], "L-musique")
    tmp = Path(tempfile.mkdtemp(prefix="zep_"))
    try:
        rng = np.random.default_rng(33)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        cls = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        base = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
                "drop": False}
        dbl = dict(base, epochs=6, swa_from=2)
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        saved_arms = dict(S.ARMS)
        S.ARMS.update({"zep": (ZM.ZRM, cls)})
        try:
            with S.patched("zep"):
                ref = LG.fit_variant([c], blocks, base, 0, 16, "none", "cpu", tag="toy-zrc")
                two = LG.fit_variant([c], blocks, dbl, 0, 16, "none", "cpu", tag="toy-zep")
                # 1. the doubled run's first epochs are zrc's bit for bit (same seed, same order draws)
                assert all(torch.equal(a[k], b[k]) for a, b in zip(ref["states"], two["states"][:3]) for k in a)
                assert len(two["states"]) == 6 and two["swa_epochs"] == [2, 3, 4, 5], two["swa_epochs"]
                # 2. its SWA state moves
                assert not all(torch.equal(ref["swa"][k], two["swa"][k]) for k in ref["swa"])
                # 3. a repeat is identical
                rep = LG.fit_variant([c], blocks, dbl, 0, 16, "none", "cpu", tag="toy-zep repeat")
                assert all(torch.equal(a[k], b[k]) for a, b in zip(every(two), every(rep)) for k in a)
        finally:
            S.ARMS.clear()
            S.ARMS.update(saved_arms)
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": BASE_ARM})
        SR.must_stop(read, ["--name", "x", "--out-root", str(tmp / "fits")])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    for a in ARMS:
        with on_zrc(a):
            assert Z.ARM == a and Z.REL_FITS == ZO.ZRC_FITS
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zep selftest: config {config_x()}; the doubled run's first epochs are zrc's bit for bit; its SWA moves; "
        f"repeats are identical ({time.time() - t0:.1f}s): ok")
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
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=sorted(ARMS))
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
                if not g.pair or len(null) != 4:
                    gp.error("recall needs --pair and the four --null files")
                recall(g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"zep {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zep: train, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
