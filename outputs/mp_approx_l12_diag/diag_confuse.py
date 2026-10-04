"""Descriptive look, after level 13's declaration (b587d92), at where level 12's TW-1x posterior picks the wrong chain on
level 12's 2,221 dev rows, which no later population holds. Not a result and not filed. diag.py, which level 13's file
pins by raw sha256, is imported and not edited; this file reads its saved units (units/k*.npz) and nothing else it wrote.

For every (dev row, seed) pair and each of the posteriors dsh and b1drop (diag.py's transforms, unchanged), the argmax
type is compared with the true chain:
- right: the argmax's token sequence is the true chain's (either bucket);
- null: the argmax is the null type;
- absent: the query has no type of the true chain's sequence in either bucket (the pool holds no such walk);
- length: the argmax has another number of steps;
- pos<i..>: same length, the steps that differ (relation or direction), 0-based from the topic side.
Each cell carries its pairs, its share of the hop's pairs, and its part of 1 - rho_bar (the mean over the readable
metrics of sum_cell (G - arm) / den_m / pairs), the quantity diag.json's cells report. For the wrong pairs it adds the
Jaccard of the argmax's reach set with R* (the true chain's reach set from its better bucket) and the share of the
in-pool golds each reaches, and the most frequent (true qtype -> argmax sequence) confusions by their part of the gap.

    python outputs/mp_approx_l12_diag/diag_confuse.py --host     # -> outputs/mp_approx_l12_diag/diag_confuse.json
"""
import importlib.util
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("l12_diag", HERE / "diag.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
L8, L0, F12 = D.L8, D.L0, D.F12
SEEDS, RET, TBL = D.SEEDS, D.RET, D.TBL
POSTS = ("dsh", "b1drop")


def name_of(toks) -> str:
    out = []
    for t in toks:
        r, d = divmod(int(t), 3)
        out.append(L8.REL_ORDER[r] + ("" if d == 0 else "^-1" if d == 1 else "^+-"))
    return ".".join(out) or "null"


def category(toks_a, true_toks, absent: bool, null: bool) -> str:
    if not null and tuple(toks_a) == tuple(true_toks):
        return "right"
    if null:
        return "null"
    if absent:
        return "absent"
    if len(toks_a) != len(true_toks):
        return "length"
    diff = [i for i, (a, b) in enumerate(zip(toks_a, true_toks)) if a != b]
    return "pos" + "".join(str(i) for i in diff)


def main() -> int:
    host = "--host" in sys.argv
    t0 = time.time()
    _decl, fx, chains = D.setup(host, L8.log_utc)   # host mode is entered here, once
    data = fx.data
    base = D.L9.views(F12.DATA)["std"]               # diag.py's dev_frame, without entering host mode a second time
    qm = base.q_metrics
    T = qm[:, [F12.FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, D.RI]
    G = qm[:, [F12.FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, D.RI]
    _den, _dens, readable = L8.denominators(T, G, L0.boot_weights(base.n_q))
    hop = np.asarray(base.q_hop)
    n, S = base.n_q, len(SEEDS)
    den = {m: float(np.mean(G[:, :, j] - T[:, :, j])) for j, m in enumerate(RET)}
    ri = [RET.index(m) for m in readable]
    qt_names = [base.meta["qtypes"][i] for i in base.q_qtype]
    lps, V = {}, {p: np.zeros((n, S, len(RET))) for p in POSTS}
    for i, k in enumerate(SEEDS):
        with np.load(HERE / "units" / f"k{k}.npz") as zf:
            if not np.array_equal(zf["q"], np.arange(n)):
                raise SystemExit(f"k{k}: not the dev rows, each once")
            lps[k] = D.unpack(zf["lp_score"], zf["lp_score_ptr"])
            for p in POSTS:
                V[p][:, i] = zf[f"V_{p}"]
    cells = {p: defaultdict(lambda: {"pairs": 0, "gap": 0.0, "jac": [], "a_gold": [], "r_gold": []}) for p in POSTS}
    conf = {p: defaultdict(lambda: {"pairs": 0, "gap": 0.0}) for p in POSTS}
    hop_pairs = defaultdict(int)
    mismatched = 0
    for q in range(n):
        steps = chains[qt_names[q]]
        true_toks = tuple(L8.chain_tokens(steps))
        true_seq = L8.type_code(0, true_toks)
        codes = np.asarray(data.t_code[data.type_rows(q)], dtype=np.int64)
        absent = not bool(np.any(codes % TBL == true_seq))
        rs, bstar = L8.r_star(data, q, steps)
        g = data.gold_local(q)
        r_gold = float(np.intersect1d(rs, g).size / g.size) if g.size else float("nan")
        h = int(hop[q])
        for i, k in enumerate(SEEDS):
            lp = lps[k][q]
            if lp.size != codes.size + 1:
                mismatched += 1
                continue
            hop_pairs[h] += 1
            facts = {"codes": codes, "true_seq": true_seq, "bstar": bstar, "te_bucket": -2}
            for p in POSTS:
                lpt, _ch = D.transform(p, lp, codes, facts["true_seq"], facts["bstar"], facts["te_bucket"])
                a = int(np.argmax(lpt))
                null = a == codes.size
                toks_a = () if null else L8.token_sequence(int(codes[a]))
                cat = category(toks_a, true_toks, absent, null)
                gap = float(np.mean([(G[q, i, j] - V[p][q, i, j]) / den[RET[j]] for j in ri]))
                c = cells[p][(h, cat)]
                c["pairs"] += 1
                c["gap"] += gap
                if cat != "right":
                    ra = np.zeros(0, dtype=np.int64) if null else data.reach(q, int(codes[a]))
                    c["jac"].append(L8.jaccard(ra, rs))
                    c["a_gold"].append(float(np.intersect1d(ra, g).size / g.size) if g.size else float("nan"))
                    c["r_gold"].append(r_gold)
                    cf = conf[p][(qt_names[q], name_of(toks_a) if not null else "null", "b1" if (not null and int(codes[a]) // TBL == 1) else "b0")]
                    cf["pairs"] += 1
                    cf["gap"] += gap
    total = n * S
    out = {"queries": n, "pairs": total, "lp_size_mismatches": mismatched, "readable": readable, "hop_pairs": dict(hop_pairs),
           "cells": {}, "confusions": {}}
    for p in POSTS:
        rows = []
        for (h, cat), c in sorted(cells[p].items()):
            mean = lambda xs: float(np.nanmean(xs)) if xs and not np.all(np.isnan(xs)) else None  # noqa: E731
            rows.append({"hop": h, "cat": cat, "pairs": c["pairs"], "share_of_hop": c["pairs"] / hop_pairs[h],
                         "gap_rho": c["gap"] / total, "jac_rstar": mean(c["jac"]), "argmax_gold_share": mean(c["a_gold"]),
                         "rstar_gold_share": mean(c["r_gold"])})
        out["cells"][p] = rows
        top = sorted(conf[p].items(), key=lambda kv: -kv[1]["gap"])[:30]
        out["confusions"][p] = [{"qtype": qt, "argmax": nm, "bucket": b, "pairs": v["pairs"], "gap_rho": v["gap"] / total}
                                for (qt, nm, b), v in top]
        out[f"gap_total_{p}"] = float(sum(r["gap_rho"] for r in rows))
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(HERE / "diag_confuse.json", out)
    for p in POSTS:
        print(p, "total", round(out[f"gap_total_{p}"], 4))
        for r in out["cells"][p]:
            print(f"  hop {r['hop']} {r['cat']:8s} pairs {r['pairs']:5d} ({r['share_of_hop']:.3f})  gap {r['gap_rho']:+.4f}  "
                  f"jac {r['jac_rstar']}  a_gold {r['argmax_gold_share']}  r_gold {r['rstar_gold_share']}")
        for c in out["confusions"][p][:15]:
            print(f"    {c['qtype']} -> {c['argmax']} [{c['bucket']}]  pairs {c['pairs']}  gap {c['gap_rho']:+.4f}")
    print("seconds", out["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
