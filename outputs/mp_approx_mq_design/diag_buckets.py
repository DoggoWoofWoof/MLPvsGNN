"""Design look after level 15's read (2c246c1), on r2: the 11,920 metaqa train-split rows level 15 read once, now
spent, so this is design only. It is not a result and is not filed. Level 15's modules are imported unchanged. r2's
sidecar is read in place and never written. This look writes only to its own directory.

Level 15's read: on hop 3 at 4x labels FZ-TW-4x-b0 is within 0.003 recall@5 of NB-oracle-b0 (0.591 against 0.594),
and even NB-oracle (the true chain's reach set from the bucket whose set best matches the golds) is below the GNN on
recall@5 and full_coverage@5 (0.602 / 0.390 against 0.603 / 0.399). Ranking one bucket's reach set first is capped
there. This look asks what the true chain's two buckets give together. R0 and R1 are the true chain's reach sets from
the bucket-0 and the bucket-1 seeds (level 8's chain_reach), each a bound, not a rule:

  O0        R0 first, then z (NB-oracle-b0)          O1        R1 first, then z
  Ostar     the better-matching bucket (NB-oracle)   Ounion    R0 and R1 together first, then z
  O0>1      R0 first, then R1, then z                O1>0      R1 first, then R0, then z
  O0+h      R0 first, then z + h 1[R1] for h in (0.5, 1, 2): R1 lifted inside the rest, softly

by hop, and hop 3 split at gold_total 5, against the GNN and against O0; also each set's gold coverage and non-gold
share.

    python outputs/mp_approx_mq_design/diag_buckets.py
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
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l15_fit as F15  # noqa: E402  (level 15's fit module, imported unchanged)

P, L0, L8, L9 = F15.P, F15.L0, F15.L8, F15.L9
SEEDS, FUNCS, METRIC_NAMES, RETRIEVAL, HOPS = F15.SEEDS, F15.FUNCS, F15.METRIC_NAMES, F15.RETRIEVAL, F15.HOPS
B = L8.ORACLE_BONUS
SOFT = (0.5, 1.0, 2.0)
OUT_JSON = HERE / "diag_buckets.json"


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


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
    chains = {qt: L8.true_chain(qt) for qt in qts}
    rules = ["O0", "O1", "Ostar", "Ounion", "O0>1", "O1>0", *[f"O0+{h}" for h in SOFT]]
    V = {r: np.zeros((n, len(SEEDS), len(RETRIEVAL))) for r in rules}
    row = {key: np.full(n, np.nan) for key in ("r0", "r1", "gt", "cov0", "cov1", "covu", "ng0", "ng1", "ngu", "star_b")}
    for q in range(n):
        steps = chains[qts[base.q_qtype[q]]]
        r0, r1 = L8.chain_reach(nb, q, steps)
        rs, b = L8.r_star(nb, q, steps)
        ru = np.union1d(r0, r1)
        g, gt = nb.gold_local(q), int(nb.q_gold_total[q])
        npool = int(nb.q_pool_size[q])
        isg = np.zeros(npool, dtype=bool)
        isg[g] = True
        row["r0"][q], row["r1"][q], row["gt"][q], row["star_b"][q] = r0.size, r1.size, gt, b
        for key, s in (("0", r0), ("1", r1), ("u", ru)):
            if g.size:
                row[f"cov{key}"][q] = float(isg[s].sum()) / g.size
            if s.size:
                row[f"ng{key}"][q] = float((~isg[s]).mean())
        bon = {}
        for name, first, second in (("O0", r0, None), ("O1", r1, None), ("Ostar", rs, None), ("Ounion", ru, None), ("O0>1", r0, r1),
                                    ("O1>0", r1, r0)):
            x = np.zeros(npool)
            if second is not None:
                x[second] = B / 2
            x[first] = B
            bon[name] = x
        for h in SOFT:
            x = np.zeros(npool)
            x[r1] = h
            x[r0] = B
            bon[f"O0+{h}"] = x
        for i, k in enumerate(SEEDS):
            z = nb.z(q, k)
            for r in rules:
                m = F15.rank_metrics(z + bon[r], g, gt)
                V[r][q, i] = [m[x] for x in RETRIEVAL]
        if q % 3000 == 0:
            log(f"row {q} of {n}")
    W = L0.boot_weights(n)
    out = {"look": "diag_buckets", "after": "2c246c1", "rows": n, "script_sha256": L0.sha256_file(Path(__file__)), "strata": {}}
    sels = {"all": np.ones(n, dtype=bool), **{f"hop={h}": hop == h for h in HOPS},
            "hop=3, gold_total<=5": (hop == 3) & (row["gt"] <= 5), "hop=3, gold_total>5": (hop == 3) & (row["gt"] > 5)}
    for name, sel in sels.items():
        s_den, s_dens, s_read = L8.denominators(T, G, W, sel)
        arms = {r: F15.read_arm(V[r], T, G, s_dens, s_read, W, sel) for r in rules}
        e = {"rows": int(sel.sum()), "readable": s_read, "twin_mean": T[sel].mean((0, 1)).round(4).tolist(),
             "gnn_mean": G[sel].mean((0, 1)).round(4).tolist(),
             "rules": {r: {"rho_bar": arms[r]["rho_bar"]["point"], "mean": V[r][sel].mean((0, 1)).round(4).tolist(),
                           "minus_gnn": {m: [arms[r]["gap_to_gnn"][m]["point"], arms[r]["gap_to_gnn"][m]["ci"]] for m in RETRIEVAL}}
                       for r in rules},
             "pairs": {f"{r} - O0": F15.paired(arms, r, "O0", s_read) for r in rules if r != "O0"},
             "rowstats": {key: (float(np.nanmean(v[sel])) if np.isfinite(v[sel]).any() else None) for key, v in row.items()}}
        out["strata"][name] = e
        log(f"{name}: {e['rows']} rows, twin {e['twin_mean']}, gnn {e['gnn_mean']}; rows {json.dumps({k: (round(v, 3) if v is not None else None) for k, v in e['rowstats'].items()})}")
        for r in rules:
            pr = e["pairs"].get(f"{r} - O0")
            rb = e["rules"][r]["rho_bar"]
            log(f"   {r:8s} rho_bar {'n/a' if rb is None else f'{rb:.3f}'}  mean {e['rules'][r]['mean']}  -gnn "
                + " ".join(f"{v[0]:+.4f}" for v in e["rules"][r]["minus_gnn"].values())
                + (f"  vs O0 {pr['point']:+.4f} [{pr['ci'][0]:+.4f}, {pr['ci'][1]:+.4f}]" if pr and pr["ci"] else ""))
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(OUT_JSON, out)
    log(f"done in {out['seconds']} s")


if __name__ == "__main__":
    main()
