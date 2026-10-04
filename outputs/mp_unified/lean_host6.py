"""Run lean_mlp6 on the host through lean_host.py, imported and called unchanged. lean_host's look list predates
lean_mlp6, so this process adds the name to it before lean_host.main runs; the substitution, the BLAS thread variables
and the placement annotation are lean_host's. The --out JSON also gets this file's sha256.

    python outputs/mp_unified/lean_host6.py lean_mlp6 --inputs z --store pca256 ... (lean_mlp6's own arguments)
    python outputs/mp_unified/lean_host6.py --selftest
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

LOOK = "lean_mlp6"


def selftest():
    assert LOOK not in LH.LOOKS or LH.LOOKS[-1] == LOOK
    looks = LH.LOOKS + (() if LOOK in LH.LOOKS else (LOOK,))
    assert looks[:len(LH.LOOKS)] == LH.LOOKS and LOOK in looks
    assert LH.out_path(["--inputs", "z", "--out", "a/b.json"]) == Path("a/b.json")
    import lean_mlp6  # noqa: F401
    print("selftest: lean_mlp6 is added to lean_host's look list and imports; --out parsing is lean_host's. all checks passed")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) < 2 or sys.argv[1] != LOOK:
        raise SystemExit(f"usage: lean_host6.py {LOOK} ARGS... | --selftest")
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, LH._threads(sys.argv))
    rest = list(sys.argv[2:])
    if LOOK not in LH.LOOKS:
        LH.LOOKS = LH.LOOKS + (LOOK,)
    LH.main()
    p = LH.out_path(rest)
    if p is not None and p.exists():
        res = json.loads(p.read_text(encoding="utf-8"))
        res["host_wrapper6"] = {"file": "outputs/mp_unified/lean_host6.py", "sha256": LH.sha(__file__)}
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, p)


if __name__ == "__main__":
    main()
