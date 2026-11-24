"""SEM2 R2a, memory-only wrapper (docs/SEM2_EDGE_OPERATORS.md section 9.11).

    python outputs/mp_unified/zsem2tb.py read --ds webqsp --arm e3v --host --threads 8 --qb 32

zsem2t.py runs unchanged. Its traversal batch (128 questions) builds one (batch, seeds, operators, D') tensor, which for
WebQSP's 791 operators is 5.7 GB per copy and ran the host out of memory. This wrapper sets the batch to --qb
questions and calls zsem2t's own main. Each question's lists do not depend on its batch-mates, so the result is the
same up to floating-point ties; the record gets "traversal_batch" added. zsem2t.py's sha256 is the record's
script_sha256, so zsem2t's report reads this row like any other. Outputs under outputs/sem2/r2/ as zsem2t's.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zsem2t as T  # noqa: E402


def main(argv):
    if "--qb" not in argv:
        raise SystemExit("zsem2tb: STOP: --qb N is required")
    i = argv.index("--qb")
    qb = int(argv[i + 1])
    rest = argv[:i] + argv[i + 2:]
    if qb < 1 or qb > T.QB:
        raise SystemExit(f"zsem2tb: STOP: --qb must be 1..{T.QB}")
    if not rest or rest[0] != "read":
        raise SystemExit("zsem2tb: STOP: only the read command is wrapped")
    d = list(T.traverse.__defaults__)
    assert d[-2] == T.QB and len(d) == 6, d
    d[-2] = qb
    T.traverse.__defaults__ = tuple(d)
    rc = T.main(rest)
    ds, arm = rest[rest.index("--ds") + 1], rest[rest.index("--arm") + 1]
    p = T.OUT / "read" / arm / f"{ds}.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["traversal_batch"] = qb
    rec["wrapper"] = "outputs/mp_unified/zsem2tb.py"
    T.Z.wjson(p, rec)
    T.log(f"{ds} {arm}: traversal batch {qb} recorded")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
