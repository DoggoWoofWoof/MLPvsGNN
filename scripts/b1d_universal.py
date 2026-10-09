"""B1d addendum: one arm for all settings, chosen by mean even-half R@5 (docs/B1D_BRIDGE_SLOTS.md, Addendum).

    python scripts/b1d_universal.py   -> outputs/bench/hipporag2/b1d_universal.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1d_bridge_slots as D  # noqa: E402
import b1c_walks as W  # noqa: E402

OUT = D.BENCH / "b1d_universal.json"


def arm_lists(name, arm):
    d = D.BENCH / name
    fs = np.load(d / "first_stage.npz")
    nodes = json.loads((d / "nodes.json").read_text(encoding="utf-8"))
    qrec = json.loads((d / "queries.json").read_text(encoding="utf-8"))
    golds = [np.asarray(q["golds"], np.int64) for q in qrec]
    base = fs["rrf"].astype(np.int64)
    if arm == "R0":
        return base, golds
    s, m, f, c = D.parse(arm)
    X = D.unit(D.B.dense_rows(name, "docs", np.asarray([x["package_row"] for x in nodes], np.int64)))
    Q = D.unit(D.B.dense_rows(name, "queries", fs["query_rows"].astype(np.int64)))
    adj = D.adjacency(d, D.FAMSETS[f], len(nodes))
    qs = Q @ X.T
    sr = np.einsum("jsd,nd->jsn", D.unit(Q[:, None, :] + X[base[:, :s]]), X) if c == "QR" else None
    L = np.array([D.slot_list(base[j], s, m, adj, qs[j],
                              None if sr is None else {int(r): sr[j, i] for i, r in enumerate(base[j, :s])})
                  for j in range(len(qrec))])
    return L, golds


def main():
    rec = json.loads((D.OUT_JSON).read_text(encoding="utf-8"))
    S = rec["settings"]
    mean_even = {a: float(np.mean([S[n]["arms"][a]["R@5"]["even"] for n in D.SETTINGS])) for a in D.ARMS}
    chosen = max(D.ARMS, key=lambda a: (mean_even[a], -D.ARMS.index(a)))
    out = {"declared_in": "docs/B1D_BRIDGE_SLOTS.md#addendum", "script_sha256": W.sha(__file__),
           "b1d_json_sha256": W.sha(D.OUT_JSON), "chosen": chosen, "mean_even_R5": round(mean_even[chosen], 4),
           "top5_by_mean_even": sorted(mean_even.items(), key=lambda x: -x[1])[:5], "settings": {}}
    for name in D.SETTINGS:
        L, golds = arm_lists(name, chosen)
        L0, _ = arm_lists(name, "R0")
        r, r0 = W.recall(L, golds, 5), W.recall(L0, golds, 5)
        lo, hi = W.boot_ci(r[1::2] - r0[1::2])
        pub = W.PUBLISHED_R5[name] / 100
        ca, co = W.boot_ci(r), W.boot_ci(r[1::2])
        cl = lambda c: "ABOVE" if c[0] > pub else "BELOW" if c[1] < pub else "AT"
        out["settings"][name] = {
            "R0_R5": round(float(r0.mean()), 4), "R5_all": round(float(r.mean()), 4),
            "R5_all_ci": [round(x, 4) for x in ca], "R5_odd": round(float(r[1::2].mean()), 4),
            "R5_odd_ci": [round(x, 4) for x in co], "odd_gain": round(float((r - r0)[1::2].mean()), 4),
            "odd_gain_ci": [round(lo, 4), round(hi, 4)],
            "call_vs_R0": "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "SAME",
            "published": pub, "call_vs_published_all": cl(ca), "call_vs_published_odd": cl(co),
            "per_setting_best": {"arm": S[name]["chosen"], "R5_all": S[name]["arms"][S[name]["chosen"]]["R@5"]["all"]}}
        print(name, out["settings"][name], flush=True)
    W.write_json(OUT, out)
    print("chosen", chosen, out["top5_by_mean_even"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
