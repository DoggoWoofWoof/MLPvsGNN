"""MuSiQue diagnosis (docs/MUSIQUE_DIAGNOSIS.md): slice M3B's filed musique eval by hop count and composition type.

Read-only over the sha-pinned M3B eval arrays; trains nothing, selects nothing. Writes
outputs/musique_diag/diagnosis.json and prints the tables.

    python scripts/musique_diagnosis.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

NPZ = "outputs/m3b/eval/musique.npz"
IDS = "outputs/m3b/eval/musique_query_ids.json"
PINS = {
    NPZ: "554b7a20605b6d965f3af62477103dc7c66e05079b716d781d2fa3d3408ee06c",
    IDS: "4412638ce9a31e412545fb750130f843f5abbdc8e52849cfe6dea840acbd1024",
}
OUT = "outputs/musique_diag/diagnosis.json"
ARMS = {
    "rrf": ["fixed:rrf"],
    "QLS-U": [f"qls_u_sota_v1__H128__s{s}" for s in range(3)],
    "GAT-NO-MP": [f"gat_no_mp_v1__H128_L2__s{s}" for s in range(3)],
    "GAT": [f"gat_universal_v1__H128_L2__s{s}" for s in range(3)],
}
METRICS = ["recall@5", "full_coverage@5", "hit@1"]
B = 2000


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hops(qid):
    m = re.match(r"(\d)hop(\d?)__", qid)
    assert m, qid
    return int(m.group(1)), m.group(0)[:-2]


def boot(d, rng):
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    for p, h in PINS.items():
        got = sha(p)
        if got != h:
            raise SystemExit(f"refusing: {p} sha256 {got} != pinned {h}")
    z = np.load(NPZ)
    ids = json.load(open(IDS, encoding="utf-8"))
    n = len(ids)
    hop = np.array([hops(q)[0] for q in ids])
    typ = np.array([hops(q)[1] for q in ids])
    g0 = "gat_universal_v1__H128_L2__s0"
    gold = z[f"{g0}/gold_total"]
    gin = z[f"{g0}/gold_in_pool"]
    for k in ARMS["QLS-U"] + ARMS["GAT-NO-MP"] + ARMS["GAT"]:
        assert np.array_equal(z[f"{k}/gold_total"], gold) and np.array_equal(z[f"{k}/gold_in_pool"], gin), k
    ceil5 = np.minimum(5, gin) / gold
    slices = [("all", np.ones(n, bool))]
    slices += [(f"{h}-hop", hop == h) for h in sorted(set(hop))]
    slices += [(t, typ == t) for t in sorted(set(typ))]
    rng = np.random.default_rng(0)
    rec = {"inputs": PINS, "n": n, "slices": {}}
    for name, m in slices:
        row = {
            "n": int(m.sum()),
            "gold_mean": float(gold[m].mean()),
            "all_gold_in_pool": float((gin[m] == gold[m]).mean()),
            "pool_ceiling@5": float(ceil5[m].mean()),
            "arms": {},
        }
        for arm, keys in ARMS.items():
            row["arms"][arm] = {
                met: {"s0": float(z[f"{keys[0]}/{met}"][m].mean()),
                      "seed_mean": float(np.mean([z[f"{k}/{met}"][m].mean() for k in keys]))}
                for met in METRICS
            }
        row["delta_mp"] = {}
        for met in ("recall@5", "full_coverage@5"):
            d = z[f"{g0}/{met}"][m] - z[f"gat_no_mp_v1__H128_L2__s0/{met}"][m]
            row["delta_mp"][met] = {"s0": float(d.mean()), "ci95": boot(d, rng)}
        rec["slices"][name] = row
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, OUT)

    print("slice | n | golds | all golds in pool | ceiling@5 | R@5 rrf / QLS-U / NO-MP / GAT | FC@5 rrf / QLS-U / NO-MP / GAT | hit@1 GAT | dMP R@5 | dMP FC@5")
    for name, row in rec["slices"].items():
        a = row["arms"]
        r5 = " / ".join(f"{a[k]['recall@5']['s0']:.3f}" for k in ARMS)
        f5 = " / ".join(f"{a[k]['full_coverage@5']['s0']:.3f}" for k in ARMS)
        dr, df = row["delta_mp"]["recall@5"], row["delta_mp"]["full_coverage@5"]
        print(f"{name} | {row['n']} | {row['gold_mean']:.2f} | {row['all_gold_in_pool']:.3f} | {row['pool_ceiling@5']:.3f} | "
              f"{r5} | {f5} | {a['GAT']['hit@1']['s0']:.3f} | "
              f"{dr['s0']:+.3f} [{dr['ci95'][0]:+.3f}, {dr['ci95'][1]:+.3f}] | {df['s0']:+.3f} [{df['ci95'][0]:+.3f}, {df['ci95'][1]:+.3f}]")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
