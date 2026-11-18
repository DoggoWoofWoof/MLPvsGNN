"""CCS's read with its read_carve guard fixed (18 Nov 2026, before any CCS number). Everything else is zccs.py, unchanged.

zccs.read checks lean_gpu.read_carve's defaults on the function object, and then raises first_q there. But read_carve is
decorated with torch.no_grad(), so lean_gpu.read_carve is torch's wrapper: its __defaults__ is None, and the real defaults
(READ_ROWS, CHECK_Q) are on the wrapped function (__wrapped__). So the guard stopped the first CCS read at once (host, 18 Nov
20:04, 1 s, nothing written). Had the guard passed, the raise would have set defaults the call never reads, and G5 would
have failed after a full GPU read.

This file runs zccs's own command line with zccs.read replaced by the same steps on inspect.unwrap(lean_gpu.read_carve).
zccs.py itself is not edited (the host code freeze until D83's and D91's runs are done). The read record says which file
read, with both files' sha256.

    python outputs/mp_unified/zccs_read.py read --name FIT     # as zccs.py read; every other command passes through
    python outputs/mp_unified/zccs_read.py selftest-guard      # the laptop or the host: the unwrap and the raise, no read
"""
import importlib
import inspect
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zccs as Z  # noqa: E402

R = Z.R


def inner_read_carve():
    """lean_gpu.read_carve's own function (under torch's no_grad wrapper) and its defaults, checked."""
    import lean_gpu as LG
    f = inspect.unwrap(LG.read_carve)
    if f.__defaults__ != (LG.READ_ROWS, LG.CHECK_Q):
        raise SystemExit(f"CCS: lean_gpu.read_carve's own defaults are {f.__defaults__}, not (READ_ROWS, CHECK_Q)")
    return LG, f


def read(fit):
    """zccs.read, step for step, with the first_q raise on the unwrapped read_carve."""
    kind = Z.kind_of(fit)
    Z.fork([fit])
    fdir = Z.FITS_DIR / fit
    outs = {ds: fdir / "reads" / f"{ds}__{Z.CARVE}.npz" for ds in Z.SETTINGS}
    done = fdir / "ccs_read.json"
    if done.exists() and all(p.exists() for p in outs.values()):
        rec = R.read_json(done)
        if all(rec["reads"][ds]["sha256"] == R.sha_file(p) for ds, p in outs.items()):
            Z.log(f"read {fit}: filed")
            return 0
        raise SystemExit(f"CCS: {R.rel(fdir)}'s reads are not the ones ccs_read.json names")
    LG, f = inner_read_carve()
    old = f.__defaults__
    M = importlib.import_module(Z.READER[kind])
    saved = M.OUT
    t = time.time()
    f.__defaults__ = (LG.READ_ROWS, Z.FULL_Q)
    M.OUT = Z.OUT
    try:
        rc = M.read(fit, Z.V)
    finally:
        M.OUT = saved
        f.__defaults__ = old
    if rc:
        return rc
    reads = {ds: Z.check_full(p) for ds, p in outs.items()}
    R.write_json(done, {"declared_in": Z.DECLARED, "stage": Z.STAGE, "fit": fit, "kind": kind, "reader": Z.READER[kind],
                        "carve": Z.CARVE, "reads": reads, "first_q": Z.FULL_Q, "gate_sha256": Z.gate_copy(),
                        "models_sha256": R.read_json(fdir / "fork.json")["models_sha256"],
                        "seconds": round(time.time() - t, 1), "zccs_sha256": R.sha_src(Z.__file__),
                        "read_by": "outputs/mp_unified/zccs_read.py", "zccs_read_sha256": R.sha_src(__file__),
                        "utc": R.utc()})
    Z.log(f"read {fit}: {time.time() - t:.0f} s, every question's scores kept")
    return 0


def selftest_guard():
    """The unwrap finds the defaults, the raise reaches the function the call runs, and the put-back restores them."""
    LG, f = inner_read_carve()
    assert LG.read_carve is not f and LG.read_carve.__wrapped__ is f, "read_carve is no longer wrapped: recheck the fix"
    assert LG.read_carve.__defaults__ is None
    old = f.__defaults__
    f.__defaults__ = (LG.READ_ROWS, Z.FULL_Q)
    try:
        sig = inspect.signature(f)
        assert sig.parameters["first_q"].default == Z.FULL_Q, sig
        assert inspect.signature(LG.read_carve).parameters["first_q"].default == Z.FULL_Q
    finally:
        f.__defaults__ = old
    assert f.__defaults__ == (LG.READ_ROWS, LG.CHECK_Q)
    Z.log(f"selftest-guard: read_carve's own defaults {old}; raised to first_q={Z.FULL_Q} and put back")
    return 0


if __name__ == "__main__":
    Z.read = read
    if sys.argv[1:2] == ["selftest-guard"]:
        sys.exit(selftest_guard())
    sys.exit(Z.main())
