"""Descriptive, read-only (not a result, not filed): on hop-3 movie_to_X_to_movie_to_Y queries, is the topic movie's own
Y attribute (reached by walking back through the topic movie) gold, and what does removing it from R* do?"""
import collections, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import mp_approx_l8 as L8
data = L8.Data(L8.DATA)
qt = [data.meta["qtypes"][i] for i in data.q_qtype]
chains = {t: L8.true_chain(t) for t in data.meta["qtypes"]}
res = collections.defaultdict(lambda: collections.defaultdict(list))
for q in range(data.n_q):
    t = qt[q]
    steps = chains[t]
    if len(steps) != 3 or not t.startswith("movie_to_"):
        continue
    rs, b = L8.r_star(data, q, steps)
    last = steps[-1]
    own = data.reach(q, L8.type_code(b, [3 * last[0] + last[1]]))   # the topic movie's own Y from the same bucket
    g = data.gold_local(q); gt = int(data.q_gold_total[q])
    own_in = np.intersect1d(own, rs)
    res[t]["own_in_rstar"].append(float(own_in.size > 0))
    if own_in.size:
        res[t]["own_gold_share"].append(float(np.isin(own_in, g).mean()))
    mem = np.zeros(int(data.q_pool_size[q]), bool); mem[rs] = True
    mem2 = mem.copy(); mem2[own_in] = False
    for k in L8.SEEDS:
        z = data.z(q, k)
        for a, m in (("oracle", mem), ("oracle_minus_own", mem2)):
            r = L8.rank_metrics(z + 1e6 * m, g, gt)
            res[t][a].append([r[x] for x in L8.RETRIEVAL])
        res[t]["gnn"].append([float(data.q_metrics[q, L8.FUNCS.index(f"gnn{k}"), L8.METRIC_NAMES.index(x)]) for x in L8.RETRIEVAL])
out = {t: {a: (np.round(np.mean(v, 0), 3).tolist()) for a, v in d.items()} for t, d in res.items()}
tot = collections.defaultdict(list)
for d in res.values():
    for a in ("oracle", "oracle_minus_own", "gnn"):
        tot[a] += d[a]
out["_all_movie_hop3"] = {a: np.round(np.mean(v, 0), 4).tolist() for a, v in tot.items()}
(ROOT / "outputs" / "mp_approx_l8_diag" / "diag3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["_all_movie_hop3"]))
for t, v in sorted(out.items()):
    if t.startswith("_"): continue
    print(t, "own_in_R*", v["own_in_rstar"], "own_gold", v.get("own_gold_share"), "oracle", v["oracle"], "minus_own", v["oracle_minus_own"], "gnn", v["gnn"])
