"""lean_host7g.py for lean_mlp12 (lean_mlp7g with S5's alignment arms). It runs lean_mlp12 on the host through
lean_host.py, after adding the name to lean_host's look list; lean_host is imported and called unchanged. The
substitution, the BLAS thread variables and the placement annotation all come from lean_host. This file adds its own
sha256 to the --out JSON.

    python outputs/mp_unified/lean_host12.py lean_mlp12 --store pca256 ... (lean_mlp7's arguments and --al-*)
    python outputs/mp_unified/lean_host12.py --selftest
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

LOOK = "lean_mlp12"


def selftest():
    assert LOOK not in LH.LOOKS or LH.LOOKS[-1] == LOOK
    looks = LH.LOOKS + (() if LOOK in LH.LOOKS else (LOOK,))
    assert looks[:len(LH.LOOKS)] == LH.LOOKS and LOOK in looks
    assert LH.out_path(["--arms", "aw0al", "--al-kappa", "20", "--out", "a/b.json"]) == Path("a/b.json")
    import lean_mlp12  # noqa: F401
    print("selftest: lean_mlp12 is added to lean_host's look list and imports; --out parsing is lean_host's. all checks passed")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) < 2 or sys.argv[1] != LOOK:
        raise SystemExit(f"usage: lean_host12.py {LOOK} ARGS... | --selftest")
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, LH._threads(sys.argv))
    rest = list(sys.argv[2:])
    if LOOK not in LH.LOOKS:
        LH.LOOKS = LH.LOOKS + (LOOK,)
    LH.main()
    p = LH.out_path(rest)
    if p is not None and p.exists():
        res = json.loads(p.read_text(encoding="utf-8"))
        res["host_wrapper12"] = {"file": "outputs/mp_unified/lean_host12.py", "sha256": LH.sha(__file__)}
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, p)


if __name__ == "__main__":
    main()
