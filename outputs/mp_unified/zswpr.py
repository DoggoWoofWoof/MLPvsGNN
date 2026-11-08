"""SWP part C, the reads of the S-trained pair on Part B's arms (docs/SWP_S_TO_R_SWAP.md section 5, step 4): SQuAD, MetaQA and WebQSP.

zswp.py's read engine (part B: fork, lists, cols, read) pointed at the four fits Part C trained in the S-world (GNX and zrckpn, seed 0, both WQ
splits), inside the S-world (python outputs/host_ops/sworld.py run --mode read -- ...), where outputs/screen and outputs/ltm are the S-world's own,
so the fits forked are the S-trained ones and the next-hop columns come from the S-trained scorer's fold fits. Nothing in zswp.py is edited. Two
differences from part B: (1) the REC arm (the record's own roots) needs lists and columns too, because the record's LTM columns come from the
record's scorer; they are built from the record's PKO base (outputs/pko_rec) into the S-world's outputs/swp/ltm/REC; (2) four fits, not twelve.

    python outputs/host_ops/sworld.py run --mode train -- python outputs/mp_unified/zswpr.py lists --arm A --dataset D --host   # any mode
    python outputs/host_ops/sworld.py run --mode train -- python outputs/mp_unified/zswpr.py cols --arm A --dataset D --split S --host
    python outputs/host_ops/sworld.py run --mode read -- python outputs/mp_unified/zswpr.py fork --arm A
    python outputs/host_ops/sworld.py run --mode read -- python outputs/mp_unified/zswpr.py read --arm A --name FIT --dataset D
    python outputs/mp_unified/zswpr.py selftest
"""
import os
import sys

import zswp as S                                              # noqa: E402  (a zrckpn read's thread environment is set there, from sys.argv)

STAGE = "SWP-C"
ROOT = S.ROOT
PAIR = ("scr-gnx-nx-s0", "scr-gnx-nx-s0-hp", "scr-ltm-zrckpn-s0", "scr-ltm-zrckpn-s0-hp")
S.GNX_FITS = tuple(f for f in S.GNX_FITS if f in PAIR)
S.MLP_FITS = tuple(f for f in S.MLP_FITS if f in PAIR)
S.ALL_FITS = S.GNX_FITS + S.MLP_FITS
S.STAGE = STAGE
assert (len(S.GNX_FITS), len(S.MLP_FITS), len(S.ALL_FITS)) == (2, 2, 4) and set(S.ALL_FITS) == set(PAIR), S.ALL_FITS


def in_world(mode=None):
    """The S-world (SWORLD=1, a tree named sworld), in the mode asked; the folders the reads write are its own."""
    if os.environ.get("SWORLD") != "1" or S.ROOT.name != "sworld":
        raise SystemExit("SWPR: Part C's reads run in the S-world only (python outputs/host_ops/sworld.py run --mode ... -- ...)")
    if mode and os.environ.get("SWORLD_MODE") != mode:
        raise SystemExit(f"SWPR: this step runs in the S-world's {mode} mode, not {os.environ.get('SWORLD_MODE')}")
    for d in ("screen", "ml1", "ltm", "gnx", "swp"):
        p = S.ROOT / "outputs" / d
        if os.path.realpath(p) != os.path.join(os.path.realpath(S.ROOT), "outputs", d):
            raise SystemExit(f"SWPR: {p} is not the S-world's own folder")
    return True


def cols_out(arm):
    """Where an arm's lists and next-hop columns live: the S-world's own folder for every arm, REC included."""
    return S.OUT / "ltm" / arm


S.cols_out = cols_out


def root_of(arm):
    """An arm's roots: ABL's folder; REC: the record's PKO base (outputs/pko_rec, a junction that is there in either mode)."""
    return ROOT / "outputs" / "pko_rec" if arm == S.REC else S.Z.arm_dir(arm)


def lists(arm, ds, host=False):
    in_world()
    S.check_arm(arm)
    import ltmcols as LTMC
    if host:
        import lean_host as LH
        LH.substitute()
    LTMC.lists(ds, S.EVAL, root=root_of(arm), out=cols_out(arm))
    return 0


def cols(arm, split, ds, host=False):
    """LTM's next-hop columns on the arm's carve from the S-world's fold fits (zswp.on_cols reads LTMC.OUT, which is the S-world's own)."""
    in_world()
    S.check_arm(arm)
    import ltmcols as LTMC
    if host:
        import lean_host as LH
        LH.substitute()
    with S.on_cols():
        LTMC.cols(split, ds, S.EVAL, root=root_of(arm), out=cols_out(arm), host=host)
    return 0


def fork(arm):
    in_world()
    return S.fork(arm)


def read(arm, fit, ds):
    in_world("read")
    return S.read(arm, fit, ds)


def selftest():
    assert S.ALL_FITS == S.GNX_FITS + S.MLP_FITS and len(S.ALL_FITS) == 4
    assert all(f.endswith("-s0") or f.endswith("-s0-hp") for f in S.ALL_FITS)
    assert [S.fit_split(f) for f in S.ALL_FITS] == [S.SPLITS[0], S.SPLITS[1], S.SPLITS[0], S.SPLITS[1]]
    assert S.cols_out is cols_out and cols_out("REC") == S.OUT / "ltm" / "REC" and S.OUT.name == "swp"
    assert root_of("REC") == ROOT / "outputs" / "pko_rec" and root_of("RK").name == "RK"
    keep = dict(os.environ)
    try:
        os.environ.pop("SWORLD", None)
        for f in (lambda: lists("REC", "squad"), lambda: cols("NK", "WQ-musique", "squad"), lambda: fork("REC"), lambda: read("REC", PAIR[0], "squad")):
            try:
                f()
                raise AssertionError("must refuse outside the S-world")
            except SystemExit:
                pass
    finally:
        os.environ.clear()
        os.environ.update(keep)
    print("zswpr self-test: OK")
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fork", "lists", "cols", "read", "selftest"))
    ap.add_argument("--arm")
    ap.add_argument("--dataset", choices=S.DATASETS)
    ap.add_argument("--split")
    ap.add_argument("--name")
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd in ("fork", "lists", "cols", "read") and a.arm is None:
        raise SystemExit(f"{a.cmd}: --arm is required")
    if a.cmd == "fork":
        return fork(a.arm)
    if a.cmd == "lists":
        if a.dataset is None:
            raise SystemExit("lists: --dataset is required")
        return lists(a.arm, a.dataset, a.host)
    if a.cmd == "cols":
        if a.dataset is None or a.split is None:
            raise SystemExit("cols: --dataset and --split are required")
        return cols(a.arm, a.split, a.dataset, a.host)
    if a.cmd == "read":
        if a.dataset is None or a.name is None:
            raise SystemExit("read: --dataset and --name are required")
        return read(a.arm, a.name, a.dataset)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
