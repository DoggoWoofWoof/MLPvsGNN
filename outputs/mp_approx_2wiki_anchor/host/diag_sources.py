"""Design look (untracked; not a result and not filed), on carve x1's look: where do the golds the GNN lifts come from?

For each row, the golds GNN seed 0 ranks in its top 5 that twin seed 0 does not ("lifted"), and for each the shortest
way the typed walk could reach it: a seed itself; one edge (any family, any direction) from a seed; one edge from a
twin top-k node that is not a seed (k = 5, 10); two edges from a seed; none of these. The same for every gold the twin
misses from its top 5. By 2wiki type. l16_look_analyze.py is imported unchanged (sha256 pinned in anchor_walk.py).

    python outputs/mp_approx_2wiki_anchor/host/diag_sources.py
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
sys.path.insert(0, str(HERE))
import anchor_walk as AW  # noqa: E402

A16 = AW.A16


def main():
    if AW.sha(Path(A16.__file__)) != AW.A16_SHA:
        raise SystemExit("l16_look_analyze.py is not the pinned file")
    t0 = time.time()
    ids, Q = A16.load(AW.ROOT / "outputs" / "mp_approx_l16_design" / "look" / "x1")
    cats = ("seed", "1e_seed_struct", "1e_seed_other", "1e_twintop5", "1e_twintop10", "2e_seed", "none")
    out = {"rows": len(Q), "script_sha256": AW.sha(Path(__file__)), "lifted": {}, "missed": {}}
    for kind in ("lifted", "missed"):
        tot = {t: dict.fromkeys(cats, 0) for t in (*A16.TYPES2W, "all")}
        for q in Q:
            n = q["n"]
            zt = q["score"][:, 0].astype(np.float64)
            zg = q["score"][:, 3].astype(np.float64)
            rt = np.argsort(-zt, kind="stable")
            rg = np.argsort(-zg, kind="stable")
            gold = np.flatnonzero(q["gold"])
            t5, g5 = set(rt[:5].tolist()), set(rg[:5].tolist())
            if kind == "lifted":
                tgt = [g for g in gold if g in g5 and g not in t5]
            else:
                tgt = [g for g in gold if g not in t5]
            if not tgt:
                continue
            S = set(q["seeds"][q["seeds"] >= 0].tolist())
            T5 = set(rt[:5].tolist()) - S
            T10 = set(rt[:10].tolist()) - S
            u, v, fam = q["u"], q["v"], q["fam"]
            adj = [[] for _ in range(n)]
            adjs = [[] for _ in range(n)]
            for a, b, f in zip(u.tolist(), v.tolist(), fam.tolist()):
                adj[a].append(b)
                if f == 0:
                    adjs[a].append(b)
            r1s = set()
            r1 = set()
            for s in S:
                r1.update(adj[s])
                r1s.update(adjs[s])
            r2 = set()
            for x in r1:
                r2.update(adj[x])
            rt5 = set()
            for s in T5:
                rt5.update(adj[s])
            rt10 = set()
            for s in T10:
                rt10.update(adj[s])
            for g in tgt:
                if g in S:
                    c = "seed"
                elif g in r1s:
                    c = "1e_seed_struct"
                elif g in r1:
                    c = "1e_seed_other"
                elif g in rt5:
                    c = "1e_twintop5"
                elif g in rt10:
                    c = "1e_twintop10"
                elif g in r2:
                    c = "2e_seed"
                else:
                    c = "none"
                tot[q["type"]][c] += 1
                tot["all"][c] += 1
        out[kind] = tot
        print(kind, json.dumps(tot), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (HERE / "diag_sources.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
