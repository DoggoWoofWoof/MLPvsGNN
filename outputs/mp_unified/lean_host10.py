"""lean_host9.py's pattern for colshift10 (label-free shift statistics of a saved lean ensemble): runs it on the host
through lean_host.py, imported and called unchanged, after adding the name to lean_host's look list. The substitution,
the BLAS thread variables and the placement annotation are lean_host's. The --out JSON also gets this file's sha256.

    python outputs/mp_unified/lean_host10.py colshift10 --load-models ... (colshift10's own arguments)
    python outputs/mp_unified/lean_host10.py --selftest
"""
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lean_host as LH  # noqa: E402  (imports no numpy, so the thread variables below still apply)

LOOK = "colshift10"


def selftest():
    assert LOOK not in LH.LOOKS or LH.LOOKS[-1] == LOOK
    looks = LH.LOOKS + (() if LOOK in LH.LOOKS else (LOOK,))
    assert looks[:len(LH.LOOKS)] == LH.LOOKS and LOOK in looks
    assert LH.out_path(["--keeps", "all;-SEMB", "--out", "a/b.json"]) == Path("a/b.json")
    import colshift10  # noqa: F401
    print("selftest: colshift10 is added to lean_host's look list and imports; --out parsing is lean_host's. all checks passed")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) < 2 or sys.argv[1] != LOOK:
        raise SystemExit(f"usage: lean_host10.py {LOOK} ARGS... | --selftest")
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, LH._threads(sys.argv))
    rest = list(sys.argv[2:])
    if LOOK not in LH.LOOKS:
        LH.LOOKS = LH.LOOKS + (LOOK,)
    LH.main()
    p = LH.out_path(rest)
    if p is not None and p.exists():
        res = json.loads(p.read_text(encoding="utf-8"))
        res["host_wrapper10"] = {"file": "outputs/mp_unified/lean_host10.py", "sha256": LH.sha(__file__)}
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, p)


if __name__ == "__main__":
    main()
