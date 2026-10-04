"""Descriptive, read-only (not a result, not filed): self-reach through the topic entity, and whether walk counts order
the true chain's reach set, on L8's sidecar."""
import collections, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import mp_approx_l8 as L8
rank_metrics = L8.rank_metrics
data = L8.Data(L8.DATA)
qt = [data.meta["qtypes"][i] for i in data.q_qtype]
chains = {t: L8.true_chain(t) for t in data.meta["qtypes"]}
arms = {"oracle": [], "oracle_minus_seeds": [], "oracle_plus_logc": [], "oracle_logc_only": [], "twin": [], "gnn": []}
per_qt = collections.defaultdict(lambda: collections.defaultdict(list))
self_reach = collections.defaultdict(lambda: [0, 0, 0])   # queries, a b0 seed in R*, that seed is gold
for q in range(data.n_q):
    steps = chains[qt[q]]
    rs, b = L8.r_star(data, q, steps)
    code = L8.type_code(b, L8.chain_tokens(steps))
    node, count, tl = data.entries(q)
    rows = data.type_rows(q)
    tc = np.asarray(data.t_code[rows])
    i = int(np.searchsorted(tc, code))
    cnt = np.zeros(int(data.q_pool_size[q]))
    if i < tc.size and int(tc[i]) == code:
        sel = tl == i
        cnt[node[sel]] = count[sel]
    g = data.gold_local(q); gt = int(data.q_gold_total[q])
    seeds = data.q_seed_local[q][(data.q_seed_bucket[q] == 0) & (data.q_seed_local[q] >= 0)]
    s_in = np.intersect1d(seeds, rs)
    self_reach[qt[q]][0] += 1
    self_reach[qt[q]][1] += int(s_in.size > 0)
    self_reach[qt[q]][2] += int(np.isin(s_in, g).any())
    mem = np.zeros(cnt.size, bool); mem[rs] = True
    mem_ns = mem.copy(); mem_ns[seeds[seeds < mem.size]] = False
    for k in L8.SEEDS:
        z = data.z(q, k)
        vals = {"oracle": z + 1e6 * mem, "oracle_minus_seeds": z + 1e6 * mem_ns,
                "oracle_plus_logc": z + 1e6 * mem + np.log1p(cnt), "oracle_logc_only": 1e6 * mem + np.log1p(cnt) + 1e-6 * z}
        for a, s in vals.items():
            r = rank_metrics(s, g, gt)
            arms[a].append([r[m] for m in L8.RETRIEVAL]); per_qt[qt[q]][a].append(r["recall@5"])
        for a, f in (("twin", "twin"), ("gnn", "gnn")):
            arms[a].append([float(data.q_metrics[q, L8.FUNCS.index(f"{f}{k}"), L8.METRIC_NAMES.index(m)]) for m in L8.RETRIEVAL])
            per_qt[qt[q]][a].append(float(data.q_metrics[q, L8.FUNCS.index(f"{f}{k}"), L8.METRIC_NAMES.index("recall@5")]))
hop = np.repeat(data.q_hop, len(L8.SEEDS))
out = {"by_hop": {f"hop={h}": {a: [round(float(x), 4) for x in np.asarray(v)[hop == h].mean(0)] for a, v in arms.items()} for h in (1, 2, 3)},
       "all": {a: [round(float(x), 4) for x in np.asarray(v).mean(0)] for a, v in arms.items()},
       "metrics": list(L8.RETRIEVAL),
       "self_reach_by_qtype": {t: v for t, v in sorted(self_reach.items()) if v[1]},
       "per_qtype_recall5": {t: {a: round(float(np.mean(v)), 3) for a, v in d.items()} for t, d in per_qt.items()}}
(ROOT / "outputs" / "mp_approx_l8_diag" / "diag2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("all", "by_hop", "metrics")}, indent=1))
print(json.dumps(out["self_reach_by_qtype"]))
