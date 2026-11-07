"""SWP Part C (docs/SWP_S_TO_R_SWAP.md section 5): the models TRAINED with S in the structural slot, read with S, R and nothing in it.

This file is the build stage: the five training ("fit") carves of the two WQ splits built with the structural slot holding S
(the dataset's own structural graph), N and K in, on ABL's chain (zabl.py, not edited): arm SNKF under outputs/abl/SNKF.
zabl's chain is written for the s1eval carve; here it runs on each dataset's fit carve (squad, hotpotqa, 2wiki: "fit"; musique:
"s1fit"; webqsp: "fit", a chained carve) by setting zabl.CARVE to the dataset's fit carve for the length of a process, and
zabl's chain starts the steps through this file (zabl.__file__ is this file's) so every step process takes the same carve.

    python outputs/mp_unified/zswpc.py chain --dataset squad --lanes 3 --host
    python outputs/mp_unified/zswpc.py gate --host
    python outputs/mp_unified/zswpc.py selftest
"""
import argparse
import contextlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zabl as Z  # noqa: E402  (ABL: the chain, its roots and the gate; its source is not edited)

DECLARED = "docs/SWP_S_TO_R_SWAP.md"
STAGE = "SWP-C"
ARM = "SNKF"                                                 # S in the slot, N and K in, on the training carves
Z.ARM_ALL.setdefault(ARM, ("S", 1, 1))
FIT_CARVE = {"squad": "fit", "hotpotqa": "fit", "2wiki": "fit", "musique": "s1fit", "webqsp": "fit"}   # zwq.FIT_CARVE
assert Z.ARM_ALL[ARM] == ("S", 1, 1)
utc, sha_file, sha_src, rel, read_json, write_json = Z.utc, Z.sha_file, Z.sha_src, Z.rel, Z.read_json, Z.write_json


def log(msg):
    Z.log(f"zswpc: {msg}")


def use(ds):
    """zabl for this dataset's fit carve: its carve, and this file as the program its chain starts."""
    if ds not in FIT_CARVE:
        raise SystemExit(f"SWPC: --dataset is one of {', '.join(FIT_CARVE)}")
    Z.CARVE = FIT_CARVE[ds]
    Z.__file__ = str(Path(__file__).resolve())
    return Z.CARVE


def bases(host=False):
    """The two bases PKO copies once into the arm's cache root (zabl's chain does it before the first cache part)."""
    M = Z.B.mods(host)
    M.PK.copy_bases(Z.roots_of(M, ARM))
    return 0


def built():
    return tuple(ds for ds in FIT_CARVE if (Z.arm_dir(ARM) / "records" / f"{ds}__{FIT_CARVE[ds]}.json").exists())


def gate(datasets=None, host=False):
    M = Z.B.mods(host)
    R = Z.roots_of(M, ARM)
    datasets = tuple(datasets or built())
    if len(datasets) != len(FIT_CARVE) and not datasets:
        raise SystemExit("SWPC: no fit carve is filed")
    with Z.B.checks(M):
        verdict = M.PK.gate(R, carves=[(ds, FIT_CARVE[ds]) for ds in datasets])
    write_json(Z.arm_dir(ARM) / f"swpc_gate_{'-'.join(datasets)}.json",
               {"declared_in": DECLARED, "stage": STAGE, "arm": ARM, "pko_gate": rel(R.gate), "pko_gate_sha256": sha_file(R.gate),
                "verdict": verdict, "carves": [f"{ds}/{FIT_CARVE[ds]}" for ds in datasets], "zswpc_sha256": sha_src(__file__),
                "utc": utc()})
    return 0


# ── the fits, in the S-world only ────────────────────────────────────────────

SPLITS = ("WQ-musique", "WQ-hotpotqa")
MODELS = {"zspk": "scr-ml1-zspk-s0", "zg6r": "scr-ml1-zg6r-s0", "gnx": "scr-gnx-nx-s0", "zrckpn": "scr-ltm-zrckpn-s0"}
FITS = tuple(f"{base}{suf}" for base in MODELS.values() for suf in ("", "-hp"))


def fit_kind(name):
    for k, base in MODELS.items():
        if name in (base, base + "-hp"):
            return k
    raise SystemExit(f"SWPC: {name} is not one of Part C's fits ({', '.join(FITS)})")


def in_world():
    """The S-world and nothing else: SWORLD set by sworld.py run, this file under a tree named sworld, outputs/pko the SNKF build, and
    the fit folders inside the tree itself (not through a junction to the record)."""
    if os.environ.get("SWORLD") != "1" or ROOT.name != "sworld":
        raise SystemExit("SWPC: a Part C fit runs only in the S-world (python outputs/host_ops/sworld.py run -- ...)")
    pko = Path(os.path.realpath(ROOT / "outputs" / "pko"))
    if pko.parts[-3:] != ("outputs", "abl", ARM):
        raise SystemExit(f"SWPC: the S-world's outputs/pko is {pko}, not {ARM}")
    for d in ("screen", "ml1", "ltm", "gnx"):
        p = ROOT / "outputs" / d
        if os.path.realpath(p) != os.path.realpath(ROOT) + os.sep + "outputs" + os.sep + d:
            raise SystemExit(f"SWPC: {p} is not the S-world's own folder")
    return True


def noop_read(_name, _env):
    return 0


def train(name):
    """One fit trained in the S-world and not read there (the read on the record's s1eval carves is not Part C's): zml1s.fit with its read
    replaced by a no-op, inside the model's own context (zgnx.on_gnx, zltm.on_ltm), after the checks of the S-world and of the columns."""
    in_world()
    kind = fit_kind(name)
    split = SPLITS[1] if name.endswith("-hp") else SPLITS[0]
    import zml1s as M1
    WQ = M1.WQ
    fd = Path(os.path.realpath(WQ.fit_dir(name)))
    if not str(fd).startswith(os.path.realpath(ROOT) + os.sep):
        raise SystemExit(f"SWPC: {name}'s fit folder {fd} is outside the S-world")
    log(f"train {name} ({kind}, {split}) in {ROOT}")
    if kind in ("zspk", "zg6r"):
        with WQ.patch(M1, read=noop_read):
            return M1.fit(name)
    import ltmcols as LTMC
    miss = [f"{ds}/{c}" for ds, c in LTMC.train_carves(split)
            if not (LTMC.cols_dir(LTMC.OUT, split, ds, c) / "record.json").is_file()]
    if miss:
        raise SystemExit(f"SWPC: {name}: no LTM columns of {miss} under {split}")
    if kind == "gnx":
        import zgnx as GX
        G = GX.import_g()
        call = GX.DEP.iso_call()
        if name not in GX.FITS:
            raise SystemExit(f"SWPC: {name} is not a GNX fit")
        log(f"{name}: NEXT into {GX.DEP.base_fit(GX.FITS[name][1], split, call)}'s model (ISO {call})")
        with GX.on_gnx(G, call), WQ.patch(M1, read=noop_read):
            return M1.fit(name)
    import zltm as LTM
    LTM.import_arm()
    with LTM.on_ltm(), WQ.patch(M1, read=noop_read):
        return M1.fit(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("kind", nargs="?")
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--dataset")
    ap.add_argument("--datasets")
    ap.add_argument("--index")
    ap.add_argument("--shard")
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--name")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        if not a.name:
            raise SystemExit("train: --name is required")
        return train(a.name)
    if a.cmd == "selftest":
        return selftest()
    if a.arm != ARM:
        raise SystemExit(f"SWPC: the arm is {ARM}")
    if a.cmd in ("chain", "step", "pools"):
        if not a.dataset:
            raise SystemExit(f"{a.cmd}: --dataset is required")
        use(a.dataset)
    if a.cmd == "chain":
        return Z.chain(ARM, a.dataset, a.lanes, a.host)
    if a.cmd == "step":
        return Z.run_step(a.kind, ARM, a.dataset, Z.parse_pair(a.index), Z.parse_pair(a.shard), a.host)
    if a.cmd == "pools":
        return Z.pools(ARM, a.dataset, Z.parse_pair(a.shard) or (0, 1), 0, a.host)
    if a.cmd == "bases":
        return bases(a.host)
    if a.cmd == "gate":
        return gate(tuple(a.datasets.split(",")) if a.datasets else None, a.host)
    raise SystemExit(__doc__)


def selftest():
    assert Z.ARM_ALL[ARM] == ("S", 1, 1) and Z.dropped(ARM) == () and Z.applies(ARM) == Z.DATASETS
    assert set(FIT_CARVE) == {"squad", "hotpotqa", "2wiki", "musique", "webqsp"} and set(FIT_CARVE) <= set(Z.SHARDS)
    import zwq
    assert FIT_CARVE == zwq.FIT_CARVE, (FIT_CARVE, zwq.FIT_CARVE)
    keep = (Z.CARVE, Z.__file__)
    try:
        assert use("musique") == "s1fit" and Z.CARVE == "s1fit" and Path(Z.__file__).name == "zswpc.py"
        assert use("squad") == "fit"
        assert Z.arm_dir(ARM) == Z.OUT / "SNKF"
        try:
            use("metaqa")
            raise AssertionError("metaqa has no fit carve")
        except SystemExit:
            pass
    finally:
        Z.CARVE, Z.__file__ = keep
    assert [fit_kind(n) for n in FITS] == ["zspk", "zspk", "zg6r", "zg6r", "gnx", "gnx", "zrckpn", "zrckpn"]
    for bad in ("scr-ml1-zrckp-s0", "scr-gnx-nx-s1", "x"):
        try:
            fit_kind(bad)
            raise AssertionError(bad)
        except SystemExit:
            pass
    keep = dict(os.environ)
    try:
        os.environ.pop("SWORLD", None)
        for f in (in_world, lambda: train("scr-ml1-zspk-s0")):         # the record's tree never trains a Part C fit
            try:
                f()
                raise AssertionError("must refuse outside the S-world")
            except SystemExit:
                pass
    finally:
        os.environ.clear()
        os.environ.update(keep)
    import zgnx
    import zltm
    import zml1s
    assert set(FITS) <= set(zml1s.ALL_FITS) | set(zgnx.FITS) | set(zltm.FITS), "a Part C fit is in no module's fit table"
    print("zswpc self-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
