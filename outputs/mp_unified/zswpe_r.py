"""SWP Part E (docs/SWP_S_TO_R_SWAP.md section 5f, step 5): the reads of the pair trained with the slot EMPTY, in the N-world only.

zswpr.py (Part C's reads, not edited) with the N-world's checks in place of the S-world's: the same engine (zswp.py's fork, lists, cols, read), the same
four fits (GNX and zrckpn, seed 0, both WQ splits), the same arms (SQuAD, MetaQA, WebQSP on Part B's arms; MuSiQue, 2Wiki, HotpotQA on b1 / b1w / b1t),
run inside the N-world (python outputs/host_ops/nworld.py run --mode ... -- ...), where outputs/screen and outputs/ltm are the N-world's own, so the fits
forked are the N-trained ones and the next-hop columns come from the N-trained scorer's fold fits.

    python outputs/host_ops/nworld.py run --mode train -- python outputs/mp_unified/zswpe_r.py lists --arm A --dataset D --host
    python outputs/host_ops/nworld.py run --mode train -- python outputs/mp_unified/zswpe_r.py cols --arm A --dataset D --split S --host
    python outputs/host_ops/nworld.py run --mode read -- python outputs/mp_unified/zswpe_r.py fork --arm A
    python outputs/host_ops/nworld.py run --mode read -- python outputs/mp_unified/zswpe_r.py read --arm A --name FIT --dataset D
    python outputs/mp_unified/zswpe_r.py selftest
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zswpr as R  # noqa: E402  (first: zswp sets a zrckpn read's thread environment from sys.argv at import; Part C's reads, not edited)

S = R.S
STAGE = "SWP-E"
WORLD = "nworld"
R.STAGE = STAGE
S.STAGE = STAGE


def in_world(mode=None):
    """The N-world (NWORLD=1, a tree named nworld), in the mode asked; the folders the reads write are its own."""
    if os.environ.get("NWORLD") != "1" or S.ROOT.name != WORLD:
        raise SystemExit("SWPE-R: Part E's reads run in the N-world only (python outputs/host_ops/nworld.py run --mode ... -- ...)")
    if mode and os.environ.get("SWORLD_MODE") != mode:
        raise SystemExit(f"SWPE-R: this step runs in the N-world's {mode} mode, not {os.environ.get('SWORLD_MODE')}")
    for d in ("screen", "ml1", "ltm", "gnx", "swp"):
        p = S.ROOT / "outputs" / d
        if os.path.realpath(p) != os.path.join(os.path.realpath(S.ROOT), "outputs", d):
            raise SystemExit(f"SWPE-R: {p} is not the N-world's own folder")
    return True


R.in_world = in_world                                          # lists, cols, fork and read look it up at call time


def selftest():
    assert R.in_world is in_world and S.STAGE == STAGE == R.STAGE
    assert len(S.ALL_FITS) == 4 and set(S.ALL_FITS) == set(R.PAIR)
    assert set(R.SPEC) <= set(S.ARMS) and set(R.SPEC_DS) <= set(S.DATASETS)
    keep = dict(os.environ)
    try:
        for env in ({}, {"SWORLD": "1", "SWORLD_MODE": "read"}, {"NWORLD": "1", "SWORLD_MODE": "read"}):
            os.environ.clear()
            os.environ.update({k: v for k, v in keep.items() if k not in ("NWORLD", "SWORLD", "SWORLD_MODE")}, **env)
            for f in (lambda: R.lists("REC", "squad"), lambda: R.cols("NK", "WQ-musique", "squad"), lambda: R.fork("REC"),
                      lambda: R.read("REC", R.PAIR[0], "squad"), lambda: R.read("B1T", R.PAIR[2], "musique")):
                try:
                    f()
                    raise AssertionError(f"must refuse outside the N-world ({env}, tree {S.ROOT.name})")
                except SystemExit:
                    pass
    finally:
        os.environ.clear()
        os.environ.update(keep)
    print("zswpe_r self-test: OK")
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        return selftest()
    return R.main(a)


if __name__ == "__main__":
    sys.exit(main())
