"""SWP Part C (docs/SWP_S_TO_R_SWAP.md section 5): the training step of the S-trained pair, run in the S-world only.

    python outputs/host_ops/sworld.py run --mode train -- python outputs/mp_unified/zswpt.py train --name scr-ml1-zspk-s0

One fit trained with S in the slot (the S-world's outputs/pko is arm SNKF) and not read there: the read on the record's s1eval carves is replaced by a
no-op (a stand-in read stamp on zspk and zg6r, which start the fits behind them), because the S-world's roots hold the five training carves only; Part C reads the fit afterwards, in read mode, on the arms' roots (zswp.py, zidpb.py).
This file imports neither zabl nor zswpc: a zrckpn fit runs in zltm's environment (6 BLAS / OpenMP / MKL threads, the GPU hidden), set before numpy or
torch loads, and zabl would put the PKO processes' 2 threads into the environment at import.
"""
import os
import sys

_TRAIN_MLP = sys.argv[1:2] == ["train"] and any(a.startswith("scr-ltm-zrckpn") for a in sys.argv)
if _TRAIN_MLP:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "6"                                  # zltm's THREADS
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                   # a CPU fit never sees the GPU

from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DECLARED = "docs/SWP_S_TO_R_SWAP.md"
STAGE = "SWP-C"
ARM = "SNKF"                                                  # zswpc.py's arm: S + N + K on the training carves
SPLITS = ("WQ-musique", "WQ-hotpotqa")                        # a name without the -hp suffix trains on the first, with it on the second
MODELS = {"zspk": "scr-ml1-zspk-s0", "zg6r": "scr-ml1-zg6r-s0", "gnx": "scr-gnx-nx-s0", "zrckpn": "scr-ltm-zrckpn-s0"}
FITS = tuple(f"{base}{suf}" for base in MODELS.values() for suf in ("", "-hp"))


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] zswpt: {msg}", flush=True)


def fit_kind(name):
    for k, base in MODELS.items():
        if name in (base, base + "-hp"):
            return k
    raise SystemExit(f"SWPT: {name} is not one of Part C's fits ({', '.join(FITS)})")


def split_of(name):
    return SPLITS[1] if name.endswith("-hp") else SPLITS[0]


def in_world():
    """The S-world in train mode and nothing else: sworld.py run --mode train set SWORLD and SWORLD_MODE, this file sits in a tree named sworld, its
    outputs/pko is the SNKF build, and the fit folders are inside the tree itself (not junctions to the record)."""
    if os.environ.get("SWORLD") != "1" or os.environ.get("SWORLD_MODE") != "train" or ROOT.name != "sworld":
        raise SystemExit("SWPT: a Part C fit trains only in the S-world, in train mode (python outputs/host_ops/sworld.py run --mode train -- ...)")
    pko = Path(os.path.realpath(ROOT / "outputs" / "pko"))
    if pko.parts[-3:] != ("outputs", "abl", ARM):
        raise SystemExit(f"SWPT: the S-world's outputs/pko is {pko}, not {ARM}")
    for d in ("screen", "ml1", "ltm", "gnx"):
        p = ROOT / "outputs" / d
        if os.path.realpath(p) != os.path.join(os.path.realpath(ROOT), "outputs", d):
            raise SystemExit(f"SWPT: {p} is not the S-world's own folder")
    return True


def noop_read(_name, _env):
    return 0


def standin_read(name, env):
    """zspk and zg6r are the starts of the fits behind them, and zml1s.load_encoder takes a start only when ML1's read stamp is on its read.json.
    A train-mode fit is not read here (Part C reads it later, in read mode), so the stamp goes on a stand-in read.json that says so: it holds no read."""
    import zml1s as M1
    fd = M1.WQ.fit_dir(name)
    M1.WQ.write_json(fd / "read.json", {"swpc_standin": True, "note": "no read: a train-mode fit of SWP part C; the stamp lets the next fit start from it"})
    M1.stamp(name, env, "read")
    return 0


def stamp_start(name):
    """The stand-in stamp on a zspk or zg6r fit that trained without it (the first Part C fits ran with noop_read)."""
    in_world()
    if fit_kind(name) not in ("zspk", "zg6r"):
        raise SystemExit(f"SWPT: {name} is not the start of another fit")
    import zml1s as M1
    fd = Path(os.path.realpath(M1.WQ.fit_dir(name)))
    if not str(fd).startswith(os.path.realpath(ROOT) + os.sep):
        raise SystemExit(f"SWPT: {name}'s fit folder {fd} is outside the S-world")
    if not (M1.stamped(name, "screen") and (fd / "train.json").is_file() and (fd / "models.pt").is_file()):
        raise SystemExit(f"SWPT: {name} is not a trained and stamped fit")
    log(f"stand-in read stamp on {name}")
    return standin_read(name, M1.device_env())


def train(name):
    """zml1s.fit with its read replaced by a no-op, inside the model's own context (zgnx.on_gnx, zltm.on_ltm), after the checks of the S-world and
    of the columns of the split's training carves."""
    in_world()
    kind = fit_kind(name)
    split = split_of(name)
    import zml1s as M1
    WQ = M1.WQ
    fd = Path(os.path.realpath(WQ.fit_dir(name)))
    if not str(fd).startswith(os.path.realpath(ROOT) + os.sep):
        raise SystemExit(f"SWPT: {name}'s fit folder {fd} is outside the S-world")
    log(f"train {name} ({kind}, {split}) in {ROOT}")
    if kind in ("zspk", "zg6r"):
        with WQ.patch(M1, read=standin_read):
            return M1.fit(name)
    import ltmcols as LTMC
    miss = [f"{ds}/{c}" for ds, c in LTMC.train_carves(split)
            if not (LTMC.cols_dir(LTMC.OUT, split, ds, c) / "record.json").is_file()]
    if miss:
        raise SystemExit(f"SWPT: {name}: no LTM columns of {miss} under {split}")
    if kind == "gnx":
        import zgnx as GX
        if name not in GX.FITS:
            raise SystemExit(f"SWPT: {name} is not a GNX fit")
        G = GX.import_g()
        call = GX.DEP.iso_call()
        _a, sd, sp = GX.FITS[name]
        log(f"{name}: NEXT into {GX.DEP.base_fit(sd, sp, call)}'s model and start (ISO {call})")
        with GX.on_gnx(G, call), WQ.patch(M1, read=noop_read):
            return M1.fit(name)
    import zltm as LTM
    LTM.import_arm()
    with LTM.on_ltm(), WQ.patch(M1, read=noop_read):
        return M1.fit(name)


def selftest():
    assert [fit_kind(n) for n in FITS] == ["zspk", "zspk", "zg6r", "zg6r", "gnx", "gnx", "zrckpn", "zrckpn"]
    assert [split_of(n) for n in FITS] == [SPLITS[0], SPLITS[1]] * 4
    for bad in ("scr-ml1-zrckp-s0", "scr-gnx-nx-s1", "x"):
        try:
            fit_kind(bad)
            raise AssertionError(bad)
        except SystemExit:
            pass
    keep = dict(os.environ)
    try:
        for var in ("SWORLD", "SWORLD_MODE"):
            os.environ.pop(var, None)
        for f in (in_world, lambda: train("scr-ml1-zspk-s0"), lambda: stamp_start("scr-ml1-zspk-s0")):         # the record's tree never trains a Part C fit
            try:
                f()
                raise AssertionError("must refuse outside the S-world")
            except SystemExit:
                pass
        os.environ.update(SWORLD="1", SWORLD_MODE="read")
        try:
            in_world()
            raise AssertionError("must refuse in read mode")
        except SystemExit:
            pass
    finally:
        os.environ.clear()
        os.environ.update(keep)
    assert "zabl" not in sys.modules and "zswpc" not in sys.modules, "zswpt must not import zabl (the PKO environment)"
    import zgnx
    import zltm
    import zml1s
    assert set(FITS) <= set(zml1s.ALL_FITS) | set(zgnx.FITS) | set(zltm.FITS), "a Part C fit is in no module's fit table"
    for n in FITS:
        k = fit_kind(n)
        assert (n in zml1s.ALL_FITS) == (k in ("zspk", "zg6r")), (n, k)
        assert (n in zgnx.FITS) == (k == "gnx") and (n in zltm.FITS) == (k == "zrckpn"), (n, k)
    print("zswpt self-test: OK")
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        return selftest()
    if len(a) == 3 and a[0] == "train" and a[1] == "--name":
        return train(a[2])
    if len(a) == 3 and a[0] == "stamp" and a[1] == "--name":
        return stamp_start(a[2])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
