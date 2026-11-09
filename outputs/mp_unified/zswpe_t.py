"""SWP Part E (docs/SWP_S_TO_R_SWAP.md section 5f): the training step of the pair trained with the slot EMPTY, run in the N-world only.

    python outputs/host_ops/nworld.py run --mode train -- python outputs/mp_unified/zswpe_t.py train --name scr-ml1-zspk-s0

zswpt.py (Part C, not edited) with the N-world's checks in place of the S-world's: the same fits (zspk, zg6r, GNX, zrckpn; seed 0; both WQ splits),
trained with the N-world's outputs/pko = arm NKF (N + K, the slot empty) and not read there (zswpt's stand-in read stamp on zspk and zg6r, which
start the fits behind them). zswpt is imported first, before numpy or torch: it puts zltm's 6-thread, GPU-hidden environment in place for a zrckpn
train from sys.argv, which is this process's argv too. Neither file imports zabl.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zswpt as T  # noqa: E402  (first: the zrckpn thread environment; Part C's train step, not edited)

ROOT = T.ROOT
DECLARED = "docs/SWP_S_TO_R_SWAP.md"
STAGE = "SWP-E"
ARM = "NKF"                                                    # zswpe.py's arm: N + K on the training carves, the slot empty
WORLD = "nworld"
T.ARM, T.STAGE = ARM, STAGE


def in_world():
    """The N-world in train mode and nothing else: nworld.py run --mode train set NWORLD and SWORLD_MODE, this file sits in a tree named nworld,
    its outputs/pko is the NKF build, and the fit folders are inside the tree itself (not junctions to the record or the S-world)."""
    if os.environ.get("NWORLD") != "1" or os.environ.get("SWORLD_MODE") != "train" or ROOT.name != WORLD:
        raise SystemExit("SWPE-T: a Part E fit trains only in the N-world, in train mode (python outputs/host_ops/nworld.py run --mode train -- ...)")
    pko = Path(os.path.realpath(ROOT / "outputs" / "pko"))
    if pko.parts[-3:] != ("outputs", "abl", ARM):
        raise SystemExit(f"SWPE-T: the N-world's outputs/pko is {pko}, not {ARM}")
    for d in ("screen", "ml1", "ltm", "gnx"):
        p = ROOT / "outputs" / d
        if os.path.realpath(p) != os.path.join(os.path.realpath(ROOT), "outputs", d):
            raise SystemExit(f"SWPE-T: {p} is not the N-world's own folder")
    return True


def standin_read(name, env):
    """zswpt.standin_read with Part E's note: zspk and zg6r start the fits behind them, and zml1s.load_encoder takes a start only when ML1's read
    stamp is on its read.json; a train-mode fit is not read here (Part E reads it later, in read mode)."""
    import zml1s as M1
    fd = M1.WQ.fit_dir(name)
    M1.WQ.write_json(fd / "read.json", {"swpe_standin": True, "note": "no read: a train-mode fit of SWP part E; the stamp lets the next fit start from it"})
    M1.stamp(name, env, "read")
    return 0


T.in_world = in_world                                          # train() and stamp_start() look both names up at call time
T.standin_read = standin_read


def selftest():
    assert T.in_world is in_world and T.standin_read is standin_read and T.ARM == "NKF"
    assert [T.fit_kind(n) for n in T.FITS] == ["zspk", "zspk", "zg6r", "zg6r", "gnx", "gnx", "zrckpn", "zrckpn"]
    keep = dict(os.environ)
    try:
        for env in ({}, {"SWORLD": "1", "SWORLD_MODE": "train"}, {"NWORLD": "1", "SWORLD_MODE": "read"}, {"NWORLD": "1", "SWORLD_MODE": "train"}):
            os.environ.clear()
            os.environ.update({k: v for k, v in keep.items() if k not in ("NWORLD", "SWORLD", "SWORLD_MODE")}, **env)
            for f in (in_world, lambda: T.train("scr-ml1-zspk-s0"), lambda: T.stamp_start("scr-ml1-zspk-s0")):
                try:
                    f()
                    raise AssertionError(f"must refuse outside the N-world in train mode ({env}, tree {ROOT.name})")
                except SystemExit:
                    pass
    finally:
        os.environ.clear()
        os.environ.update(keep)
    assert "zabl" not in sys.modules and "zswpc" not in sys.modules and "zswpe" not in sys.modules, "zswpe_t must not import zabl (the PKO environment)"
    T.selftest()                                               # Part C's own checks of the fit table, with this file's in_world in place
    print("zswpe_t self-test: OK")
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        return selftest()
    return T.main(a)


if __name__ == "__main__":
    raise SystemExit(main())
