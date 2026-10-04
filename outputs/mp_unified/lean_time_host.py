"""Run lean_time6 on the host through lean_host.py, imported and called unchanged (lean_host6.py's pattern). lean_host's
look list predates lean_time6, so this process adds the name to it before lean_host.main runs; the substitution (the
verified mirror in place of the package, in memory) and the placement annotation are lean_host's. lean_time6 reads
--threads from the command line at import, before numpy loads. The --out JSON also gets this file's sha256.

    python outputs/mp_unified/lean_time_host.py lean_time6 --threads 1 --dataset 2wiki ... (lean_time6's own arguments)
    python outputs/mp_unified/lean_time_host.py --selftest
"""
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lean_host as LH  # noqa: E402  (imports no numpy, so the thread variables lean_time6 sets still apply)

LOOK = "lean_time6"


def selftest():
    assert LOOK not in LH.LOOKS or LH.LOOKS[-1] == LOOK
    assert LH.out_path(["--threads", "1", "--out", "a/b.json"]) == Path("a/b.json")
    import lean_time6
    assert lean_time6._threads(["x", "lean_time6", "--threads", "4"]) == 4 and lean_time6._threads(["x"]) == 1
    print("selftest: lean_time6 imports and reads --threads from the wrapper's command line; --out parsing is lean_host's. "
          "all checks passed")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) < 2 or sys.argv[1] != LOOK:
        raise SystemExit(f"usage: lean_time_host.py {LOOK} ARGS... | --selftest")
    if "--threads" not in sys.argv and not any(x.startswith("--threads=") for x in sys.argv):
        raise SystemExit("pass --threads: lean_time6 sets the BLAS, numba and torch threads from it")
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, LH._threads(sys.argv))
    rest = list(sys.argv[2:])
    if LOOK not in LH.LOOKS:
        LH.LOOKS = LH.LOOKS + (LOOK,)
    LH.main()
    p = LH.out_path(rest)
    if p is not None and p.exists():
        res = json.loads(p.read_text(encoding="utf-8"))
        res["host_wrapper_time"] = {"file": "outputs/mp_unified/lean_time_host.py", "sha256": LH.sha(__file__)}
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, p)


if __name__ == "__main__":
    main()
