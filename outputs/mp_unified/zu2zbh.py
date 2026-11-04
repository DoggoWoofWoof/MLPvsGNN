"""U2z stage B1 on the lab host (docs/U2Z_NEIGHBOURHOOD_BUDGET.md, amendment A1): zpkoh.py's mirror around zu2zb.py.
zpkoh.py and zu2zb.py are unchanged; this file only runs zu2zb.py as zpkoh runs zpko.py, and nothing in it changes a
number.

    python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2zbh.py [--host] [--min-gib G] COMMAND ARGS

COMMAND and ARGS are zu2zb.py's (plan, n1, check, pools, check-merge, merge, coverage, gate, report). Exit codes are
zpkoh's: 0 done, 1 a step failed, 2 the dataset's inputs are not all here (nothing started), 3 the disk is below the
floor (nothing started; default 12 GiB). Every process keeps BLAS at 2 threads: OpenBLAS's thread count changes the
features.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zpkoh as ZH  # noqa: E402

ZB = HERE / "zu2zb.py"
READS = ("n1", "check", "pools", "coverage", "golds")        # the commands that open a carve of a dataset
FIT = {"musique": "s1fit"}                                    # the dataset's fit carve, whose inputs a carve shares
B1 = ("b1w",)                                                 # B1's settings: zb1p's own inputs, read by zb1p.load_settings


def carve_key(tail):
    """The input map's key of the dataset the command reads (s1sel shares the dataset's fit inputs); None on B1."""
    d = tail[1]
    c = tail[tail.index("--carve") + 1] if "--carve" in tail else "s1sel"
    if c in B1:
        return None
    keys = {f"{x}/{y}" for x, y in ZH.CARVES}
    return f"{d}/{c}" if f"{d}/{c}" in keys else f"{d}/{FIT.get(d, 'fit')}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    host = "--host" in argv
    argv = [x for x in argv if x != "--host"]
    min_gib = 12.0
    if "--min-gib" in argv:
        i = argv.index("--min-gib")
        min_gib = float(argv[i + 1])
        del argv[i:i + 2]
    if "--dir" in argv:                                       # development runs: outputs/NAME instead of outputs/u2zb
        i = argv.index("--dir")
        os.environ["U2ZB_DIR"] = argv[i + 1]
        del argv[i:i + 2]
    if not argv:
        print(__doc__)
        return 0
    if host:
        os.environ["U2ZB_HOST"] = "1"                         # zu2zb.py: zpkoh's mirror is in place (zb1p must not apply it again)
    if host and argv[0] in READS:
        if ZH.guard_disk(min_gib):
            return 3
        key = carve_key(argv)
        if key and ZH.guard_ready([key]):
            return 2
    ZH.ZPKO = ZB
    return ZH.run_zpko(argv, host)


if __name__ == "__main__":
    sys.exit(main())
