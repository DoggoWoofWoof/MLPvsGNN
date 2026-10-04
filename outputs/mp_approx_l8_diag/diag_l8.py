"""Descriptive, read-only look at L8's filed units (not a result, not filed): where the chain identification fails."""
import collections, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import mp_approx_l8 as L8
METRIC_NAMES, rank_metrics = L8.METRIC_NAMES, L8.rank_metrics

data = L8.Data(L8.DATA)
qt = [data.meta["qtypes"][i] for i in data.q_qtype]
chains = {t: L8.true_chain(t) for t in data.meta["qtypes"]}
truth = [tuple(L8.chain_tokens(chains[t])) for t in qt]
R5 = METRIC_NAMES.index("recall@5")
names = [f"{L8.REL_ORDER[t // 3]}:{('fwd', 'bwd', 'both')[t % 3]}" for t in range(L8.N_TOK)]
def show(toks):
    return "null" if toks == ("null",) else " > ".join(names[t] for t in toks)
out = {}
for arm in ("TP", "TP-hash", "TP-bag"):
    cats = {h: collections.Counter() for h in (1, 2, 3)}
    pairs = {h: collections.Counter() for h in (1, 2, 3)}
    filt = {h: [] for h in (1, 2, 3)}
    per_qt = collections.defaultdict(list)
    for k in L8.SEEDS:
        for fold in range(L8.FOLDS):
            npz, _js = L8.unit_paths(arm, k, fold)
            with np.load(npz) as z:
                qs, am, met = z["q"], z["argmax"], z["metrics"]
            for q, a, m in zip(qs, am, met):
                q = int(q); h = int(data.q_hop[q]); a = int(a)
                tru = truth[q]
                g = data.gold_local(q)
                if a < 0:
                    cat = "null_or_uniform"; toks = ("null",); ra = np.zeros(0, np.int64)
                else:
                    toks = L8.token_sequence(a); b = a // L8.TB ** L8.MAX_L
                    ra = data.reach(q, a)
                    rt = data.reach(q, L8.type_code(b, tru))
                    if toks == tru:
                        cat = "same_chain"
                    elif ra.size and np.array_equal(np.sort(ra), np.sort(rt)):
                        cat = "other_chain_same_reach"
                    elif g.size and np.isin(g, ra).all():
                        cat = "other_chain_covers_golds"
                    elif g.size and np.isin(g, ra).any():
                        cat = "other_chain_some_golds"
                    else:
                        cat = "other_chain_no_gold"
                cats[h][cat] += 1
                if cat not in ("same_chain", "other_chain_same_reach"):
                    pairs[h][(show(tru), show(toks))] += 1
                # a hard filter by the argmax type's reach set, ordered by the twin
                zq = data.z(q, k)
                s = zq + (1e6 * np.isin(np.arange(zq.size), ra) if ra.size else 0.0)
                filt[h].append(rank_metrics(s, g, int(data.q_gold_total[q]))["recall@5"])
                per_qt[qt[q]].append((float(m[R5]), float(data.q_metrics[q, L8.FUNCS.index(f"twin{k}"), R5]),
                                      float(data.q_metrics[q, L8.FUNCS.index(f"gnn{k}"), R5]), cat in ("same_chain", "other_chain_same_reach")))
    out[arm] = {
        "categories": {f"hop={h}": {c: round(v / sum(cats[h].values()), 4) for c, v in cats[h].most_common()} for h in (1, 2, 3)},
        "argmax_hard_filter_recall@5": {f"hop={h}": float(np.mean(filt[h])) for h in (1, 2, 3)},
        "top_confusions": {f"hop={h}": [[a, b, c] for (a, b), c in pairs[h].most_common(8)] for h in (1, 2, 3)},
        "per_qtype": sorted([[t, len(v), round(float(np.mean([x[0] for x in v])), 3), round(float(np.mean([x[1] for x in v])), 3),
                              round(float(np.mean([x[2] for x in v])), 3), round(float(np.mean([x[3] for x in v])), 3)]
                             for t, v in per_qt.items()], key=lambda r: r[2] - r[4])}
# equivalence classes: how many of the query's types share R* (b0)
eq = collections.Counter()
for q in range(data.n_q):
    rs, b = L8.r_star(data, q, chains[qt[q]])
    if not rs.size:
        eq["empty"] += 1; continue
    rows = data.type_rows(q)
    same = 0
    for r in range(rows.start, rows.stop):
        nodes = np.asarray(data.e_node[int(data.t_first[r]):int(data.t_first[r] + data.t_size[r])])
        if nodes.size == rs.size and np.array_equal(np.sort(nodes), np.sort(rs)):
            same += 1
    eq[min(same, 6)] += 1
out["types_sharing_R_star"] = {str(k): v for k, v in sorted(eq.items(), key=lambda kv: str(kv[0]))}
dst = ROOT / "outputs" / "mp_approx_l8_diag" / "diag.json"
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({a: {"categories": v["categories"], "filter": v["argmax_hard_filter_recall@5"]} for a, v in out.items() if a != "types_sharing_R_star"}, indent=1))
print(out["types_sharing_R_star"])
