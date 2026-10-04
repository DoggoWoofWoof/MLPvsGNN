"""Design look after level 15's read (2c246c1), on r2: the 11,920 metaqa train-split rows level 15 read once, now
spent, so this is design only. It is not a result and is not filed. Level 15's modules are imported unchanged. r2's
sidecar is read in place and never written. This look writes only to its own directory.

Hop 3 is where level 15's primary is furthest below the GNN. There even NB-oracle-b0, which puts R0 (the true chain's
bucket-0 reach set) first and then follows the twin's order, is below the GNN on recall@5 (0.594 against 0.603) and
full_coverage@5 (0.386 against 0.399). Ranking the chain's reach set first is not enough, so this look asks what is
inside R0. It uses the true chain (the qtype's), so it bounds rules and is not one:

  coverage  the golds' share in R0, R0's non-gold share, |R0| and the gold count, by hop
  counts    e_count, the number of walks of the chain's type that reach a node (compiled before any fit). Does it
            separate golds from non-golds inside R0 (AUC per row), and is it better there than the twin's z?
  rules     R0 first, then by z (NB-oracle-b0, reproduced as a check); by count, then z; by z + lam log count.
            The same three for R* (NB-oracle's bucket, chosen by gold).
  split     the per-row (NB-oracle-b0 - GNN) on recall@5, by whether R0 misses an in-pool gold and holds a non-gold

    python outputs/mp_approx_mq_design/diag_within.py --host
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
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
BONUS = L8.ORACLE_BONUS
LAMS = (-0.5, 0.25, 0.5, 1.0, 2.0)
OUT_JSON = HERE / "diag_within.json"


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


def reach_count(d, q: int, code: int) -> tuple[np.ndarray, np.ndarray]:
    """level 8's Data.reach, with the entries' counts over the same slice."""
    rows = d.type_rows(q)
    tc = d.t_code[rows]
    i = int(np.searchsorted(tc, code))
    if i >= tc.size or int(tc[i]) != code:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    r = rows.start + i
    s = slice(int(d.t_first[r]), int(d.t_first[r] + d.t_size[r]))
    return np.asarray(d.e_node[s], dtype=np.int64), np.asarray(d.e_count[s], dtype=np.int64)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a gold's value > a non-gold's), ties at one half."""
    if not pos.size or not neg.size:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="store_true")   # r2's sidecar is read in place; no package is opened
    a = ap.parse_args(argv)
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
    rules = ["O0", "O0-cnt", *[f"O0-z+{lam}logc" for lam in LAMS], "Ostar", "Ostar-cnt", "Ostar-z+1.0logc"]
    V = {r: np.zeros((n, len(SEEDS), len(RETRIEVAL))) for r in rules}
    row = {key: np.full(n, np.nan) for key in ("r0", "r1", "gold_total", "gold_in_pool", "cov0", "nongold0", "auc_cnt0", "auc_z0",
                                               "auc_cnt_star", "star_bucket")}
    for q in range(n):
        steps = chains[qts[base.q_qtype[q]]]
        toks = L8.chain_tokens(steps)
        r0, c0 = reach_count(nb, q, L8.type_code(0, toks))
        r1, c1 = reach_count(nb, q, L8.type_code(1, toks))
        chk0, chk1 = L8.chain_reach(nb, q, steps)
        if not (np.array_equal(r0, chk0) and np.array_equal(r1, chk1)):
            raise SystemExit(f"row {q}: the reach sets with counts are not level 8's")
        rs, b = L8.r_star(nb, q, steps)
        cs = c1 if b == 1 else c0
        if not np.array_equal(rs, r1 if b == 1 else r0):
            raise SystemExit(f"row {q}: R* is not the bucket's reach set")
        g, gt = nb.gold_local(q), int(nb.q_gold_total[q])
        npool = int(nb.q_pool_size[q])
        isg = np.zeros(npool, dtype=bool)
        isg[g] = True
        row["r0"][q], row["r1"][q], row["gold_total"][q], row["gold_in_pool"][q] = r0.size, r1.size, gt, g.size
        row["star_bucket"][q] = b
        if g.size:
            row["cov0"][q] = float(isg[r0].sum()) / g.size
        if r0.size:
            row["nongold0"][q] = float((~isg[r0]).mean())
            z0 = nb.z(q, 0)
            row["auc_cnt0"][q] = auc(np.log(c0[isg[r0]]), np.log(c0[~isg[r0]]))
            row["auc_z0"][q] = auc(z0[r0][isg[r0]], z0[r0][~isg[r0]])
        if rs.size:
            row["auc_cnt_star"][q] = auc(np.log(cs[isg[rs]]), np.log(cs[~isg[rs]]))
        bon = {}
        for name, rset, cnt in (("O0", r0, c0), ("Ostar", rs, cs)):
            x = np.zeros(npool)
            x[rset] = BONUS
            bon[name] = x
            y = x.copy()
            y[rset] += 1e3 * np.log(cnt)
            bon[f"{name}-cnt"] = y
            for lam in (LAMS if name == "O0" else (1.0,)):
                y = x.copy()
                y[rset] += lam * np.log(cnt)
                bon[f"{name}-z+{lam}logc"] = y
        for i, k in enumerate(SEEDS):
            z = nb.z(q, k)
            for r in rules:
                m = F15.rank_metrics(z + bon[r], g, gt)
                V[r][q, i] = [m[x] for x in RETRIEVAL]
        if q % 2000 == 0:
            log(f"row {q} of {n}")
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    out = {"look": "diag_within", "after": "2c246c1", "rows": n, "lams": list(LAMS), "readable": readable,
           "script_sha256": L0.sha256_file(Path(__file__)), "strata": {}}
    sel_all = {f"hop={h}": hop == h for h in HOPS}
    sel_all["hop=3, gold_total<=5"] = (hop == 3) & (row["gold_total"] <= 5)
    sel_all["hop=3, gold_total>5"] = (hop == 3) & (row["gold_total"] > 5)
    for name, sel in sel_all.items():
        if not sel.any():
            continue
        s_den, s_dens, s_read = L8.denominators(T, G, W, sel)
        arms = {r: F15.read_arm(V[r], T, G, s_dens, s_read, W, sel) for r in rules}
        e = {"rows": int(sel.sum()), "readable": s_read,
             "twin_mean": T[sel].mean((0, 1)).round(4).tolist(), "gnn_mean": G[sel].mean((0, 1)).round(4).tolist(),
             "rules": {r: {"rho_bar": round(arms[r]["rho_bar"]["point"], 4) if arms[r]["rho_bar"]["point"] is not None else None,
                           "mean": V[r][sel].mean((0, 1)).round(4).tolist()} for r in rules},
             "pairs": {f"{r} - O0": F15.paired(arms, r, "O0", s_read) for r in rules if r != "O0"},
             "rowstats": {key: (float(np.nanmean(v[sel])) if np.isfinite(v[sel]).any() else None) for key, v in row.items()},
             "share_r0_empty": float((row["r0"][sel] == 0).mean()),
             "share_r0_all_gold": float(np.nanmean(np.where(row["r0"][sel] > 0, row["nongold0"][sel] == 0, np.nan))),
             "share_cov0_full": float(np.nanmean(np.where(row["gold_in_pool"][sel] > 0, row["cov0"][sel] == 1, np.nan)))}
        if name.startswith("hop=3"):
            d = (V["O0"][:, :, 0] - G[:, :, 0]).mean(1)
            miss = row["cov0"] < 1
            nong = row["nongold0"] > 0
            cats = {"r0_empty": row["r0"] == 0, "covers_all_no_nongold": (row["r0"] > 0) & ~miss & ~nong,
                    "covers_all_with_nongold": (row["r0"] > 0) & ~miss & nong, "misses_no_nongold": (row["r0"] > 0) & miss & ~nong,
                    "misses_with_nongold": (row["r0"] > 0) & miss & nong}
            e["O0_minus_gnn_recall@5_by_category"] = {c: {"rows": int((sel & m).sum()), "mean": float(d[sel & m].mean()) if (sel & m).any() else None,
                                                          "sum_over_stratum_rows": float(d[sel & m].sum() / sel.sum())}
                                                      for c, m in cats.items()}
        out["strata"][name] = e
        log(f"{name}: {e['rows']} rows, twin {e['twin_mean']}, gnn {e['gnn_mean']}")
        for r in rules:
            pr = e["pairs"].get(f"{r} - O0")
            log(f"   {r:18s} rho_bar {e['rules'][r]['rho_bar']}  mean {e['rules'][r]['mean']}"
                + (f"  vs O0 {pr['point']:+.4f} [{pr['ci'][0]:+.4f}, {pr['ci'][1]:+.4f}]" if pr and pr["ci"] else ""))
        log("   rows: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e["rowstats"].items()}))
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(OUT_JSON, out)
    log(f"done in {out['seconds']} s")


if __name__ == "__main__":
    main()
