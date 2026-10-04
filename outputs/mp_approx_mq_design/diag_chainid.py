"""Design look after level 15's read (2c246c1), on r2: the 11,920 metaqa train-split rows level 15 read once, now
spent, so this is design only. It is not a result and is not filed. Level 15's modules are imported unchanged. r2's
sidecar and level 15's units are read in place and never written. This look writes only to its own directory.

diag_within.json: counts do not separate golds inside the true chain's reach set. What is left on hop 3 is choosing the
chain (level 15's wrong-chain gap against NB-oracle-b0 is 0.055, 0.041 of it on hop 3) and how hard to rank its reach
set (0.030). This look breaks the chain choice down by qtype, from level 15's units' kept-round argmax:
  per qtype  rows, agreement (argmax = the true chain), null share, the (unit - NB-oracle-b0) and (unit - GNN) means on
             recall@5, full_coverage@5 and hit@1, and each qtype's share of the hop's (NB-oracle-b0 - unit) recall@5
  wrong rows the most common argmax sequences, and the argmax's reach set against R0 (the true chain's bucket-0 reach
             set): Jaccard, and each set's share of the in-pool golds
Units: FZ full TW-1x (b1d metrics) and FZ b0 TW-1x (b0 metrics), seeds 0 to 2.

    python outputs/mp_approx_mq_design/diag_chainid.py
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import collections  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l15_fit as F15  # noqa: E402  (level 15's fit module, imported unchanged)

P, L0, L8, L9 = F15.P, F15.L0, F15.L8, F15.L9
SEEDS, FUNCS, METRIC_NAMES, RETRIEVAL = F15.SEEDS, F15.FUNCS, F15.METRIC_NAMES, F15.RETRIEVAL
TBL = F15.TBL
UNITS = {"FZ-TW-1x-b1d": ("fz", "full", "TW-1x", "b1d"), "FZ-TW-1x-b0": ("fz", "b0", "TW-1x", "dsh")}
OUT_JSON = HERE / "diag_chainid.json"
ARROW = {0: ">", 1: "<", 2: "~"}


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


def tok_str(seq) -> str:
    if seq == ("null",):
        return "null"
    return " ".join(f"{L8.REL_ORDER[t // 3]}{ARROW[t % 3]}" for t in seq)


def code_str(code: int) -> str:
    if code < 0:
        return "null"
    return f"b{code // TBL}: " + tok_str(L8.token_sequence(code))


def main():
    t0 = time.time()
    P.HARD_STOP_DIR[0] = HERE   # a hard stop of this look lands here, never in level 15's outputs
    P.route_stops()
    vs = L9.views(F15.DATA)
    base, nb = vs["std"], vs["nb"]
    n = base.n_q
    ri = [METRIC_NAMES.index(m) for m in RETRIEVAL]
    qm = base.q_metrics
    T = qm[:, [FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, ri]
    G = qm[:, [FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, ri]
    hop = np.asarray(base.q_hop)
    qts = base.meta["qtypes"]
    qt = [qts[i] for i in base.q_qtype]
    chains = {x: L8.true_chain(x) for x in qts}
    M, A = {}, {}
    for arm, (norm, view, fit, sc) in UNITS.items():
        M[arm] = np.zeros((n, len(SEEDS), len(RETRIEVAL)))
        A[arm] = np.zeros((n, len(SEEDS)), dtype=np.int64)
        for i, k in enumerate(SEEDS):
            npz, js = F15.unit_paths(norm, view, fit, k)
            flog = L8.read_json(js)
            if L0.sha256_file(npz) != flog["arrays_sha256"]:
                raise SystemExit(f"{npz}: not the arrays its log records")
            with np.load(npz) as z:
                if not np.array_equal(z["q"], np.arange(n)):
                    raise SystemExit(f"{npz}: not r2's rows, each once")
                M[arm][:, i] = z[f"metrics_{sc}"][:, ri]
                A[arm][:, i] = z["argmax"]
    O0 = np.zeros((n, len(SEEDS), len(RETRIEVAL)))
    truth, jac, cov_mode, cov0, r0n = [], {a: np.full((n, len(SEEDS)), np.nan) for a in UNITS}, \
        {a: np.full((n, len(SEEDS)), np.nan) for a in UNITS}, np.full(n, np.nan), np.zeros(n)
    for q in range(n):
        steps = chains[qt[q]]
        truth.append(tuple(L8.chain_tokens(steps)))
        r0 = L8.chain_reach(nb, q, steps)[0]
        r0n[q] = r0.size
        g, gt = nb.gold_local(q), int(nb.q_gold_total[q])
        isg = np.zeros(int(nb.q_pool_size[q]), dtype=bool)
        isg[g] = True
        if g.size:
            cov0[q] = isg[r0].sum() / g.size
        bonus = np.zeros(isg.size)
        bonus[r0] = L8.ORACLE_BONUS
        for i, k in enumerate(SEEDS):
            m = F15.rank_metrics(nb.z(q, k) + bonus, g, gt)
            O0[q, i] = [m[x] for x in RETRIEVAL]
        for a in UNITS:
            for i in range(len(SEEDS)):
                c = int(A[a][q, i])
                if c < 0 or L8.token_sequence(c) == truth[q]:
                    continue
                rm = nb.reach(q, c)
                jac[a][q, i] = L8.jaccard(rm, r0)
                if g.size:
                    cov_mode[a][q, i] = isg[rm].sum() / g.size
        if q % 3000 == 0:
            log(f"row {q} of {n}")
    out = {"look": "diag_chainid", "after": "2c246c1", "rows": n, "script_sha256": L0.sha256_file(Path(__file__)), "arms": {}}
    for a in UNITS:
        right = np.asarray([[L8.token_sequence(int(A[a][q, i])) == truth[q] for i in range(len(SEEDS))] for q in range(n)])
        null = A[a] < 0
        per_hop = {}
        for h in (1, 2, 3):
            sh = hop == h
            gap_h = (O0[sh, :, 0] - M[a][sh, :, 0]).mean(1).sum()
            rows = {}
            for x in sorted({qt[q] for q in np.flatnonzero(sh)}):
                sel = np.asarray([qt[q] == x for q in range(n)]) & sh
                wrong = ~right[sel] & ~null[sel]
                conf = collections.Counter(code_str(int(c)) for c, w in zip(A[a][sel].ravel(), wrong.ravel()) if w)
                d_o = (M[a][sel] - O0[sel]).mean((0, 1))
                d_g = (M[a][sel] - G[sel]).mean((0, 1))
                rows[x] = {"rows": int(sel.sum()), "agreement": round(float(right[sel].mean()), 3), "null": round(float(null[sel].mean()), 3),
                           "minus_O0": d_o.round(4).tolist(), "minus_gnn": d_g.round(4).tolist(),
                           "O0_minus_gnn": (O0[sel] - G[sel]).mean((0, 1)).round(4).tolist(),
                           "share_of_hop_O0_gap_recall@5": round(float((O0[sel, :, 0] - M[a][sel, :, 0]).mean(1).sum() / gap_h), 3) if gap_h else None,
                           "r0_mean": round(float(r0n[sel].mean()), 2), "cov0": round(float(np.nanmean(cov0[sel])), 3) if np.isfinite(cov0[sel]).any() else None,
                           "wrong_rows_jaccard_mode_r0": round(float(np.nanmean(jac[a][sel])), 3) if np.isfinite(jac[a][sel]).any() else None,
                           "wrong_rows_cov_mode": round(float(np.nanmean(cov_mode[a][sel])), 3) if np.isfinite(cov_mode[a][sel]).any() else None,
                           "top_wrong": conf.most_common(4), "truth": tok_str(truth[int(np.flatnonzero(sel)[0])])}
            per_hop[f"hop={h}"] = {"rows": int(sh.sum()), "agreement": round(float(right[sh].mean()), 3),
                                   "O0_minus_unit_recall@5_sum_per_row": round(float(gap_h / sh.sum()), 4), "qtypes": rows}
        out["arms"][a] = per_hop
        for h, e in per_hop.items():
            log(f"{a} {h}: {e['rows']} rows, agreement {e['agreement']}, O0 - unit R@5 per row {e['O0_minus_unit_recall@5_sum_per_row']}")
            for x, v in sorted(e["qtypes"].items(), key=lambda kv: -(kv[1]["share_of_hop_O0_gap_recall@5"] or 0)):
                log(f"   {x:38s} {v['rows']:5d} agree {v['agreement']:.3f} null {v['null']:.3f} share {v['share_of_hop_O0_gap_recall@5']} "
                    f"-O0 {v['minus_O0']} -gnn {v['minus_gnn']} O0-gnn {v['O0_minus_gnn']} cov0 {v['cov0']} jac {v['wrong_rows_jaccard_mode_r0']} "
                    f"covm {v['wrong_rows_cov_mode']} | {v['truth']} | {v['top_wrong'][:2]}")
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(OUT_JSON, out)
    log(f"done in {out['seconds']} s")


if __name__ == "__main__":
    main()
