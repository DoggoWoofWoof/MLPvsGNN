"""Design look after the ULA look, on r2 (the 11,920 metaqa train-split rows level 15 read once, now spent, so this
is design only; not a result, not filed). Level 15's modules are imported unchanged; r2's sidecar is read in place.

diag_buckets.py found the headroom left inside the true chain is per-row bucket choice: on hop 1, O0 (the bucket-0
reach set first) reads 0.914 and Ostar (the better bucket, chosen with the golds) 1.200, while no fixed combination
of the two buckets helps. This look asks whether that choice can be predicted from the row without golds: seed
trust. Per row, y = 1 when O1 (the bucket-1 reach set of the true chain first) beats O0 on the mean of the three
metrics over the twin's seeds. Features (no gold, no edge, no neighbour):
  chain-free   n_b0 (1 when the dense and the splade rank-1 node agree), n_b1, the twin's z of the best bucket-0 and
               bucket-1 seed and their gap
  with chain   the above plus log(1 + |R0|), log(1 + |R1|), R0 empty, R1 empty (the true chain's reach sizes: an
               upper bound for a rule that would use the decoded chain)
A logistic rule is cross-fitted over two halves (even and odd rows); Otrust ranks the predicted bucket's reach set
first, then z. Read by hop, against O0, Ostar, the twin and the GNN. Also the topic entity's place (diagnostic only:
in bucket 0, in bucket 1, not a seed) and how y splits by it.

    python outputs/mp_approx_mq_design/diag_trust.py
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
OUT_JSON = HERE / "diag_trust.json"


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


def logistic(X, y, w=None, l2=1e-2, iters=200):
    """Newton's method for L2-regularised logistic regression (bias in column 0, unpenalised)."""
    n, d = X.shape
    w = np.zeros(d) if w is None else w
    pen = np.full(d, l2)
    pen[0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (p - y) / n + pen * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X / n + np.diag(pen)
        step = np.linalg.solve(Hm + 1e-9 * np.eye(d), g)
        w = w - step
        if np.abs(step).max() < 1e-8:
            break
    return w


def auc(score, y):
    o = np.argsort(score, kind="stable")
    r = np.empty(score.size)
    r[o] = np.arange(1, score.size + 1)
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None


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
    V = {r: np.zeros((n, len(SEEDS), len(RETRIEVAL))) for r in ("O0", "O1", "Ostar")}
    feat = np.zeros((n, 9))
    te_place = np.zeros(n, dtype=np.int64)    # 0 bucket 0, 1 bucket 1, 2 in pool not a seed, 3 not in pool
    for q in range(n):
        steps = chains[qts[base.q_qtype[q]]]
        r0, r1 = L8.chain_reach(nb, q, steps)
        rs, _b = L8.r_star(nb, q, steps)
        g, gt = nb.gold_local(q), int(nb.q_gold_total[q])
        npool = int(nb.q_pool_size[q])
        seeds, buckets = np.asarray(nb.q_seed_local[q]), np.asarray(nb.q_seed_bucket[q])
        s0, s1 = seeds[(buckets == 0) & (seeds >= 0)], seeds[(buckets == 1) & (seeds >= 0)]
        z0 = nb.z(q, SEEDS[0])
        zb0 = float(z0[s0].max()) if s0.size else -5.0
        zb1 = float(z0[s1].max()) if s1.size else -5.0
        feat[q] = [s0.size, s1.size, zb0, zb1, zb0 - zb1, np.log1p(r0.size), np.log1p(r1.size), float(r0.size == 0), float(r1.size == 0)]
        te = int(nb.q_te_local[q])
        te_place[q] = 3 if te < 0 else (0 if np.isin(te, s0) else (1 if np.isin(te, s1) else 2))
        for name, first in (("O0", r0), ("O1", r1), ("Ostar", rs)):
            x = np.zeros(npool)
            x[first] = B
            for i, k in enumerate(SEEDS):
                m = F15.rank_metrics(nb.z(q, k) + x, g, gt)
                V[name][q, i] = [m[r] for r in RETRIEVAL]
        if q % 3000 == 0:
            log(f"row {q} of {n}")
    d01 = V["O1"].mean((1, 2)) - V["O0"].mean((1, 2))
    y = (d01 > 0).astype(np.float64)
    out = {"look": "diag_trust", "after": "look_ula", "rows": n, "script_sha256": L0.sha256_file(Path(__file__)), "features": {}}
    names = ["n_b0", "n_b1", "z_b0", "z_b1", "z_gap", "log_r0", "log_r1", "r0_empty", "r1_empty"]
    sets = {"chain_free": [0, 1, 2, 3, 4], "with_chain": list(range(9))}
    half = (np.arange(n) % 2).astype(bool)
    for sname, cols in sets.items():
        X = np.c_[np.ones(n), feat[:, cols]]
        mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
        X[:, 1:] = (X[:, 1:] - mu) / sd
        prob = np.zeros(n)
        weights = {}
        for h in (0, 1):
            tr, te_ = half == bool(h), half != bool(h)
            w = logistic(X[tr], y[tr])
            prob[te_] = 1.0 / (1.0 + np.exp(-(X[te_] @ w)))
            weights[f"fit_on_half{h}"] = dict(zip(["bias", *[names[c] for c in cols]], np.round(w, 3).tolist()))
        pick = prob > 0.5
        Vt = np.where(pick[:, None, None], V["O1"], V["O0"])
        V[f"Otrust_{sname}"] = Vt
        out["features"][sname] = {"cols": [names[c] for c in cols], "weights": weights, "auc": auc(prob, y),
                                  "auc_by_hop": {f"hop={h}": auc(prob[hop == h], y[hop == h]) for h in HOPS},
                                  "pick_b1_share": float(pick.mean()), "pick_b1_by_hop": {f"hop={h}": float(pick[hop == h].mean()) for h in HOPS}}
        log(f"{sname}: auc {out['features'][sname]['auc']:.3f}, by hop {out['features'][sname]['auc_by_hop']}, pick b1 {pick.mean():.3f}")
    W = L0.boot_weights(n)
    rules = ["O0", "O1", "Ostar", "Otrust_chain_free", "Otrust_with_chain"]
    out["y_share"] = {f"hop={h}": float(y[hop == h].mean()) for h in HOPS}
    out["y_share"]["all"] = float(y.mean())
    out["te_place"] = {f"hop={h}": {str(c): int(((hop == h) & (te_place == c)).sum()) for c in range(4)} for h in HOPS}
    out["y_by_te_place"] = {str(c): (float(y[te_place == c].mean()) if (te_place == c).any() else None) for c in range(4)}
    out["strata"] = {}
    for name, sel in [("all", np.ones(n, dtype=bool))] + [(f"hop={h}", hop == h) for h in HOPS]:
        s_den, s_dens, s_read = L8.denominators(T, G, W, sel)
        arms = {r: F15.read_arm(V[r], T, G, s_dens, s_read, W, sel) for r in rules}
        out["strata"][name] = {"rows": int(sel.sum()),
                               "rules": {r: {"rho_bar": arms[r]["rho_bar"]["point"], "ci": arms[r]["rho_bar"].get("ci"),
                                             "mean": V[r][sel].mean((0, 1)).round(4).tolist()} for r in rules},
                               "pairs": {f"{r} - O0": F15.paired(arms, r, "O0", s_read) for r in rules if r != "O0"}}
        e = out["strata"][name]
        log(f"{name}: " + "; ".join(f"{r} {e['rules'][r]['rho_bar']:.3f}" for r in rules if e["rules"][r]["rho_bar"] is not None))
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(OUT_JSON, out)
    log(f"y share {json.dumps(out['y_share'])}; te place {json.dumps(out['te_place'])}; y by te place {json.dumps(out['y_by_te_place'])}")
    log(f"done in {out['seconds']} s")


if __name__ == "__main__":
    main()
