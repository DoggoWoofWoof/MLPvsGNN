"""Design look (untracked; not a result and not filed), on carve x1's half B: how far can the typed-walk bonus go?

For each row, the bonus of one walk type alone, s = z + kappa 1[v in R_tau] / |R_tau|^beta, with the protect rule
(the twin's rank-1 node first), at kappa in a small grid and beta fixed; the per-row best (tau, kappa) by recall@5 +
full_coverage@5 is an oracle choice of the type posterior restricted to one type (a mixture can only do at least as
well on a row). Its rho is a lower bound on the ceiling of the bonus's function class with the type posterior chosen per
row, against the fitted A256-1's rho. Also the share of rows where some type improves on the twin, by 2wiki type.
Schemes: T0 (family x direction, walks of 1 and 2 edges) and A256-1 (anchor phrases, walks of one edge).

    python outputs/mp_approx_2wiki_anchor/host/diag_oracle.py
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_walk as AW  # noqa: E402

A16 = AW.A16
KAPPAS = (0.5, 1.0, 2.0, 4.0, 8.0)
BETA = 0.28


def row_metrics(z, gold, gt, top1):
    """metrics_of over a batch of score vectors (k, n), with the protected node first."""
    s = z.copy()
    s[:, top1] = np.inf
    order = np.argsort(-s, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(s.shape[0])[:, None], order] = np.arange(1, s.shape[1] + 1)[None, :]
    g = np.flatnonzero(gold)
    gr = np.sort(rank[:, g], axis=1)
    total = max(gt, 1)
    r5 = (gr <= 5).sum(1) / total
    fc5 = ((gr <= 5).sum(1) == total).astype(float)
    return r5, fc5


def main():
    if AW.sha(Path(A16.__file__)) != AW.A16_SHA:
        raise SystemExit("l16_look_analyze.py is not the pinned file")
    t0 = time.time()
    _ids, Q = A16.load(AW.ROOT / "outputs" / "mp_approx_l16_design" / "look" / "x1")
    checks, _vocab = AW.anchor_tables(Q)
    B = list(range(1, len(Q), 2))
    M = np.stack([q["metrics"] for q in Q])[:, :, A16.RI]
    tB, gB = M[B, 0], M[B, 3]
    gain = gB.mean(0) - tB.mean(0)
    out = {"rows_B": len(B), "script_sha256": AW.sha(Path(__file__)), "kappas": KAPPAS, "beta": BETA, "twin0": tB.mean(0).round(4).tolist(),
           "gnn0": gB.mean(0).round(4).tolist(), "schemes": {}}
    ty = np.asarray([Q[i]["type"] for i in B])
    for scheme in ("T0", "A256-1"):
        if scheme == "T0":
            nt, tk = 5, [A16.famdir(q) for q in Q]
            max_len = 2
        else:
            nt, tk = AW.anchor_tokens(Q, 256, "w2")
            max_len = 1
        best = np.zeros((len(B), 2))
        base = np.zeros((len(B), 2))
        n_types = np.zeros(len(B))
        for bi, i in enumerate(B):
            q = Q[i]
            z = A16.zscore(q["score"][:, 0])
            top1 = int(np.argmax(z))
            TY = A16.walk_types(q, tk[i], nt, max_len)
            n_types[bi] = len(TY)
            r0, f0 = row_metrics(z[None, :], q["gold"], q["gt"], top1)
            base[bi] = (r0[0], f0[0])
            if not TY:
                best[bi] = base[bi]
                continue
            S = []
            for R in TY.values():
                ind = np.zeros(q["n"])
                ind[R] = 1.0 / (R.size ** BETA)
                for k in KAPPAS:
                    S.append(z + k * ind)
            r, f = row_metrics(np.stack(S), q["gold"], q["gt"], top1)
            j = int(np.argmax(r + f))
            if r[j] + f[j] > base[bi].sum():
                best[bi] = (r[j], f[j])
            else:
                best[bi] = base[bi]
        rho = (best.mean(0) - tB[:, :2].mean(0)) / gain[:2]
        better = (best.sum(1) > base.sum(1) + 1e-12)
        by_type = {t: {"rows": int((ty == t).sum()), "oracle": best[ty == t].mean(0).round(4).tolist(), "twin0": tB[ty == t, :2].mean(0).round(4).tolist(),
                       "gnn0": gB[ty == t, :2].mean(0).round(4).tolist(), "share_improvable": round(float(better[ty == t].mean()), 3)}
                   for t in A16.TYPES2W if (ty == t).any()}
        out["schemes"][scheme] = {"oracle_B": best.mean(0).round(4).tolist(), "rule_alone_B": base.mean(0).round(4).tolist(),
                                  "oracle_rho (R@5, FC@5)": rho.round(3).tolist(), "share_improvable": round(float(better.mean()), 3),
                                  "types_per_row": round(float(n_types.mean()), 1), "by_type": by_type}
        print(scheme, json.dumps(out["schemes"][scheme]), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (HERE / "diag_oracle.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
