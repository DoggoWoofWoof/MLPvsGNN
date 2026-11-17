"""The D91 world (docs/D91_D83_WITHOUT_LABELS.md sections 2 and 4): a working tree of its own for D91's fits, LTM columns and reads.

d83world.py with D91's names: sworld's make / verify / mode / run (SWP's S-world, not edited), on a sibling tree ../d91world whose
outputs/pko is outputs/abl/D91F (the five fit carves: D83's pools, looks and caches with the record's three family links, zd91.py) in
train mode and outputs/abl/D91E (the six s1eval carves) in read mode. A run carries D91WORLD=1 beside sworld's SWORLD=1 / SWORLD_MODE;
D91's train, LTM and read modules check D91WORLD and a tree named d91world, so no fit named like a record fit is written outside it.
The fit, LTM, screen and read folders are the tree's own (sworld's OWN, plus d91 for the fit records and the reads).

    python outputs/host_ops/d91world.py make
    python outputs/host_ops/d91world.py verify
    python outputs/host_ops/d91world.py mode train|read
    python outputs/host_ops/d91world.py run [--mode train|read] -- CMD
    python outputs/host_ops/d91world.py selftest
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

PKO_TARGET = "outputs/abl/D91F"
READ_TARGET = "outputs/abl/D91E"
WORLD_NAME = "d91world"
WORLD = S.ROOT.parent / WORLD_NAME
OWN = tuple(S.OWN) + ("d91",)
S.PKO_TARGET, S.READ_TARGET, S.OWN = PKO_TARGET, READ_TARGET, OWN   # make, set_mode and run read them at call time


def make(world=WORLD, root=S.ROOT):
    return S.make(world, root)


def verify(world=WORLD, root=S.ROOT, quiet=False):
    return S.verify(world, root, quiet)


def set_mode(mode, world=WORLD, root=S.ROOT):
    return S.set_mode(mode, world, root)


def run(cmd, world=WORLD, root=S.ROOT, mode="train"):
    if Path(world).name != WORLD_NAME:
        raise SystemExit(f"d91world: {world} is not the D91 world")
    keep = os.environ.get("D91WORLD")
    os.environ["D91WORLD"] = "1"                            # sworld.run copies os.environ into the step's environment
    try:
        return S.run(cmd, world, root, mode)
    finally:
        if keep is None:
            os.environ.pop("D91WORLD", None)
        else:
            os.environ["D91WORLD"] = keep


def selftest():
    tmp = Path(tempfile.mkdtemp(prefix="d91world_t_"))
    try:
        root, world = tmp / "ws", tmp / WORLD_NAME
        for d, f in (("src", "a.py"), ("scripts", "b.py"), ("docs", "c.md"), ("configs", "d.json"), ("outputs/mp_unified", "m.py"),
                     ("outputs/host_ops", "h.py"), ("outputs/abl/D91F", "x.txt"), ("outputs/abl/D91E", "e.txt"), ("outputs/abl/NKF", "y.txt"),
                     ("outputs/other", "o.txt"), ("outputs/pko", "p.txt"), ("data", "z.txt")):
            (root / d).mkdir(parents=True, exist_ok=True)
            (root / d / f).write_text("x", encoding="utf-8")
        make(world, root)
        real = os.path.realpath
        assert S.is_link(world / "outputs/pko") and real(world / "outputs/pko") == real(root / "outputs/abl/D91F")
        for d in OWN:
            p = world / "outputs" / d
            assert p.is_dir() and not S.is_link(p), d
        verify(world, root, quiet=True)
        code = "import os,sys;sys.exit(0 if os.environ.get('D91WORLD')=='1' and os.environ.get('SWORLD_MODE')=='train' else 4)"
        assert run([sys.executable, "-c", code], world, root) == 0
        assert os.environ.get("D91WORLD") in (None, "1")
        assert set_mode("read", world, root) == 0 and real(world / "outputs/pko") == real(root / "outputs/abl/D91E")
        try:
            run([sys.executable, "-c", "pass"], world, root, mode="train")
            raise AssertionError("a train step must not run in read mode")
        except SystemExit:
            pass
        assert set_mode("train", world, root) == 0 and real(world / "outputs/pko") == real(root / "outputs/abl/D91F")
        try:
            run([sys.executable, "-c", "pass"], tmp / "nworld", root)
            raise AssertionError("run must refuse a tree that is not the D91 world")
        except SystemExit:
            pass
        (root / "src/a.py").write_text("changed", encoding="utf-8")
        try:
            run([sys.executable, "-c", "pass"], world, root)
            raise AssertionError("a changed record file must stop a run")
        except SystemExit:
            pass
        assert WORLD.name == WORLD_NAME and S.PKO_TARGET == PKO_TARGET and S.READ_TARGET == READ_TARGET
    finally:
        for p in sorted(tmp.rglob("*"), key=lambda q: -len(q.parts)):
            if S.is_link(p):
                os.rmdir(p) if os.name == "nt" else p.unlink()
        shutil.rmtree(tmp, ignore_errors=True)
    print("d91world self-test: OK")
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
            raise SystemExit("d91world: mode train|read")
        return set_mode(a[1])
    if a[0] == "run":
        if "--" not in a:
            raise SystemExit("d91world: run [--mode train|read] -- CMD")
        head = a[1:a.index("--")]
        mode = head[head.index("--mode") + 1] if "--mode" in head else "train"
        if mode not in ("train", "read"):
            raise SystemExit("d91world: --mode is train or read")
        return run(a[a.index("--") + 1:], mode=mode)
    if a[0] == "selftest":
        return selftest()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
