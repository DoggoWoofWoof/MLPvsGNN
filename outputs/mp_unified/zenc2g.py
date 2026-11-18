"""ENC E2 (docs/ENC_ENCODER_COMPARISON.md section 12.11's log, 19 Nov): ABL's slot types in an arm's world.

    python outputs/host_ops/pylib_run.py outputs/mp_unified/zenc2g.py all DATASET --host     (in an arm's world)
    python outputs/mp_unified/zenc2g.py selftest

zabl_g.py, imported as a module and run unchanged (its main, its argv). One thing differs: zabl_g records each path
relative to the tree it runs in (rel), and the arm's U2m types folder (the hook's swap, 12.10) lies in the record's
tree, outside the arm's world. Such a path is recorded relative to the world ("../ws/..."); every path inside the world
is recorded as zabl_g records it. Nothing else changes; the record string feeds no number.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def outside_rel(orig, root):
    def rel(p):
        try:
            return orig(p)
        except ValueError:
            return os.path.relpath(Path(p).resolve(), Path(root).resolve()).replace("\\", "/")
    rel._e2 = orig
    return rel


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "e2world_x"
        (w / "outputs").mkdir(parents=True)
        (Path(td) / "ws" / "t").mkdir(parents=True)

        def orig(p):
            return str(Path(p).resolve().relative_to(w.resolve())).replace("\\", "/")
        r = outside_rel(orig, w)
        assert r(w / "outputs") == "outputs" and r(Path(td) / "ws" / "t") == "../ws/t", (r(w / "outputs"), r(Path(td) / "ws" / "t"))
    print("zenc2g self-test: OK")
    return 0


def main():
    if sys.argv[1:2] == ["selftest"]:
        return selftest()
    import zabl_g as G
    G.rel = outside_rel(G.rel, G.ROOT)
    return G.main()


if __name__ == "__main__":
    sys.exit(main())
