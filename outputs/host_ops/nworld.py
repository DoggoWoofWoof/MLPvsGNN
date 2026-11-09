"""The N-world (docs/SWP_S_TO_R_SWAP.md section 5f, SWP Part E): a third working tree for the fits trained with the structural slot EMPTY.

sworld.py's make / verify / mode / run, on a sibling tree ../nworld whose outputs/pko is outputs/abl/NKF (N + K, the slot empty) in train mode and
the record's PKO base in read mode. sworld.py is not edited (the S-world pins it): this file sets its PKO target and passes its own tree to every
call. A run carries NWORLD=1 beside sworld.py's SWORLD=1 / SWORLD_MODE; the Part E modules (zswpe_t.py, zswpe_r.py) check NWORLD and a tree named
nworld, and the Part C modules (zswpt.py, zswpr.py) refuse a tree that is not named sworld, so neither stage can run in the other's tree.

    python outputs/host_ops/nworld.py make
    python outputs/host_ops/nworld.py verify
    python outputs/host_ops/nworld.py mode train|read
    python outputs/host_ops/nworld.py run [--mode train|read] -- CMD
    python outputs/host_ops/nworld.py selftest
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sworld as S  # noqa: E402  (the S-world's code, not edited)

PKO_TARGET = "outputs/abl/NKF"
WORLD = S.ROOT.parent / "nworld"
S.PKO_TARGET = PKO_TARGET                                   # make, set_mode and run read it at call time


def make(world=WORLD, root=S.ROOT):
    return S.make(world, root)


def verify(world=WORLD, root=S.ROOT, quiet=False):
    return S.verify(world, root, quiet)


def set_mode(mode, world=WORLD, root=S.ROOT):
    return S.set_mode(mode, world, root)


def run(cmd, world=WORLD, root=S.ROOT, mode="train"):
    if Path(world).name != "nworld":
        raise SystemExit(f"nworld: {world} is not the N-world")
    keep = os.environ.get("NWORLD")
    os.environ["NWORLD"] = "1"                              # sworld.run copies os.environ into the step's environment
    try:
        return S.run(cmd, world, root, mode)
    finally:
        if keep is None:
            os.environ.pop("NWORLD", None)
        else:
            os.environ["NWORLD"] = keep


def selftest():
    tmp = Path(tempfile.mkdtemp(prefix="nworld_t_"))
    try:
        root, world = tmp / "ws", tmp / "nworld"
        for d, f in (("src", "a.py"), ("scripts", "b.py"), ("docs", "c.md"), ("configs", "d.json"), ("outputs/mp_unified", "m.py"),
                     ("outputs/host_ops", "h.py"), ("outputs/abl/NKF", "x.txt"), ("outputs/abl/SNKF", "y.txt"), ("outputs/other", "o.txt"),
                     ("outputs/pko", "p.txt"), ("data", "z.txt")):
            (root / d).mkdir(parents=True, exist_ok=True)
            (root / d / f).write_text("x", encoding="utf-8")
        make(world, root)
        assert S.is_link(world / "outputs/pko") and os.path.realpath(world / "outputs/pko") == os.path.realpath(root / "outputs/abl/NKF")
        assert os.path.realpath(world / "outputs/pko") != os.path.realpath(root / "outputs/abl/SNKF")
        verify(world, root, quiet=True)
        r = run([sys.executable, "-c", "import os,sys;sys.exit(0 if os.environ.get('NWORLD')=='1' and os.environ.get('SWORLD_MODE')=='train' else 4)"],
                world, root)
        assert r == 0, r
        assert "NWORLD" not in os.environ or os.environ.get("NWORLD") == "1"
        assert set_mode("read", world, root) == 0 and os.path.realpath(world / "outputs/pko") == os.path.realpath(root / "outputs/pko")
        try:
            run([sys.executable, "-c", "pass"], world, root, mode="train")
            raise AssertionError("a train step must not run in read mode")
        except SystemExit:
            pass
        assert set_mode("train", world, root) == 0 and os.path.realpath(world / "outputs/pko") == os.path.realpath(root / "outputs/abl/NKF")
        try:
            run([sys.executable, "-c", "pass"], tmp / "sworld", root)
            raise AssertionError("run must refuse a tree that is not the N-world")
        except SystemExit:
            pass
        (root / "src/a.py").write_text("changed", encoding="utf-8")
        try:
            run([sys.executable, "-c", "pass"], world, root)
            raise AssertionError("a changed record file must stop a run")
        except SystemExit:
            pass
        assert WORLD.name == "nworld" and S.PKO_TARGET == PKO_TARGET
    finally:
        for p in sorted(tmp.rglob("*"), key=lambda q: -len(q.parts)):
            if S.is_link(p):
                os.rmdir(p) if os.name == "nt" else p.unlink()
        shutil.rmtree(tmp, ignore_errors=True)
    print("nworld self-test: OK")
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "make":
        return make()
    if a[0] == "verify":
        return verify()
    if a[0] == "mode":
        if len(a) != 2:
            raise SystemExit("nworld: mode train|read")
        return set_mode(a[1])
    if a[0] == "run":
        if "--" not in a:
            raise SystemExit("nworld: run [--mode train|read] -- CMD")
        head = a[1:a.index("--")]
        mode = head[head.index("--mode") + 1] if "--mode" in head else "train"
        if mode not in ("train", "read"):
            raise SystemExit("nworld: --mode is train or read")
        return run(a[a.index("--") + 1:], mode=mode)
    if a[0] == "selftest":
        return selftest()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
