"""SWP Part C (docs/SWP_S_TO_R_SWAP.md section 5): the models TRAINED with S in the structural slot, read with S, R and nothing in it.

This file is the build stage: the five training ("fit") carves of the two WQ splits built with the structural slot holding S
(the dataset's own structural graph), N and K in, on ABL's chain (zabl.py, not edited): arm SNKF under outputs/abl/SNKF.
zabl's chain is written for the s1eval carve; here it runs on each dataset's fit carve (squad, hotpotqa, 2wiki: "fit"; musique:
"s1fit"; webqsp: "fit", a chained carve) by setting zabl.CARVE to the dataset's fit carve for the length of a process, and
zabl's chain starts the steps through this file (zabl.__file__ is this file's) so every step process takes the same carve.

    python outputs/mp_unified/zswpc.py chain --dataset squad --lanes 3 --host
    python outputs/mp_unified/zswpc.py gate --host
    python outputs/mp_unified/zswpc.py selftest

The S-trained fits themselves are zswpt.py's (it does not import zabl: zrckpn's fit needs zltm's thread environment).
"""
import argparse
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
    a = ap.parse_args(argv)
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
    print("zswpc self-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
