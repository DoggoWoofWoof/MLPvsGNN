"""SEM2 R1k with zsem2.rank_read's own block size (docs/SEM2_EDGE_OPERATORS.md section 8.13).

    python outputs/mp_unified/zsem2hkb.py selftest [--threads 2]
    python outputs/mp_unified/zsem2hkb.py r1k --ds D --arm e3v [--host] [--threads 8]

zsem2hk.rank_ceiling blocks the nodes by 5e7 // Q; zsem2.rank_read by 1e8 // Q. On WebQSP (2.6M nodes) the different
block shapes rounded a few near-ties the other way, and 8.11's gate G-B (raw ranks equal to rank_read's exactly)
stopped the row. This wrapper replaces only the block size, so the raw ranks are rank_read's own arithmetic; everything
else is zsem2hk's (imported, never edited). Records carry wrapper_sha256 and block_rule.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2hk as K  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

Z = K.Z
BLOCK_RULE = "zsem2.rank_read's: max(1024, min(65536, 1e8 // Q))"


def rank_ceiling(torch, sc, X, smp, excl, op_of, deg):
    """zsem2hk.rank_ceiling with zsem2.rank_read's block size."""
    s, d, lab = smp["s"], smp["d"], smp["lab"]
    Q = s.size
    opn = np.maximum(op_of[lab], 0)
    rows_t = {L: torch.from_numpy(np.flatnonzero(lab == L)) for L in np.unique(lab).tolist()}
    dt = torch.from_numpy(deg.of(lab, d).astype(np.float32))
    en, eq = excl
    with torch.no_grad():
        q = sc.query(sc.embed(torch.from_numpy(X.rows(s))), torch.from_numpy(opn), torch.from_numpy(lab >= smp["nR"]))
        t = (q * sc.embed(torch.from_numpy(X.rows(d)))).sum(1)
        hi_t, lo_t = (t + Z.EPS)[:, None], (t - Z.EPS)[:, None]
        NB = max(1024, min(65536, int(1e8 // max(Q, 1))))
        acc = {k: torch.zeros(Q, dtype=torch.float64) for k in ("g", "t", "gc", "tc")}
        for b0 in range(0, X.n, NB):
            b1 = min(X.n, b0 + NB)
            S = q @ sc.embed(torch.from_numpy(X.rows(slice(b0, b1)))).T
            lo, hi = np.searchsorted(en, [b0, b1])
            if hi > lo:
                S[torch.from_numpy(eq[lo:hi]), torch.from_numpy(en[lo:hi] - b0)] = float("-inf")
            keep = K.deg_block(torch, deg, rows_t, Q, b0, b1) <= dt[:, None]
            g = S > hi_t
            tie = (S >= lo_t) & ~g
            acc["g"] += g.sum(1)
            acc["t"] += tie.sum(1)
            acc["gc"] += (g & keep).sum(1)
            acc["tc"] += (tie & keep).sum(1)
    a = {k: v.numpy() for k, v in acc.items()}
    return 1.0 + a["g"] + 0.5 * a["t"], 1.0 + a["gc"] + 0.5 * a["tc"]


K.rank_ceiling = rank_ceiling


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rc = K.main(argv)
    if argv and argv[0] == "r1k" and rc == 0:
        a = K.argparse.ArgumentParser()
        a.add_argument("cmd")
        a.add_argument("--ds")
        a.add_argument("--arm")
        a.add_argument("--host", action="store_true")
        a.add_argument("--threads", type=int)
        a = a.parse_args(argv)
        p = K.OUT / a.ds / f"{a.arm}.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["wrapper_sha256"] = Z.sha_file(Path(__file__))
        rec["block_rule"] = BLOCK_RULE
        rec["declared_in"] = K.DECLARED + " (run through zsem2hkb.py: section 8.13)"
        Z.wjson(p, rec)
        Z.log(f"{a.ds} {a.arm}: record tagged with the wrapper")
    return rc


if __name__ == "__main__":
    sys.exit(main())
