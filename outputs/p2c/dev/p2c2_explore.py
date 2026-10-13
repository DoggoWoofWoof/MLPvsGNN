"""P2c part 2 exploration on a fit carve (the models' training questions; no choice or call reads them). The 'all'
graph with our triples' P form as a fourth family (all4) against the three-family 'all': P1e's five rules and the
walks on 'all', recomputed with all4. Pools only, nothing fit.

    python p2c2_explore.py DATASET [Q] [THREADS] [--host]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import p2c2_proto as X  # noqa: E402

K, D, B, E, S4, G, P = X.K, X.D, X.B, X.E, X.S4, X.G, X.K.P
NMAX = X.NMAX
OUTD = Path(__file__).with_name("p2c2_explore_out")
SHOW = ("A1all", "Kall", "KTall", "A1Tall", "RRF5", "RRF5T", "RRF7T", "KINDcut", "KIND_T", "KINDcut_T", "KINDcut_7T",
        "A1st", "R0")


def run(name, Q, threads):
    op = S4.Opened()
    carve = "s1fit" if name == "musique" else "fit"
    ds = op.canonical.Dataset(name, root=str(op.served))
    n = int(ds.n_nodes)
    nonly = K.name_only(ds.nodes(), n)
    t0 = time.time()
    H, Pm = X.mentions(name, n)
    ip4, col4, counts = X.p_form(H, Pm, n)
    del H, Pm
    X.log(f"{name}: fourth family {counts} ({time.time() - t0:.0f}s)")
    del ds
    cache = {}
    res = {}

    def sink(name, carve_, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t1 = time.time()
        regime = G.regime_of(families, construction)
        _stores, unions = G.open_unions(cache, op, name, union)
        U4 = X.UnionF.from_union(unions["all"], ((ip4, col4),))
        reg = {"S": "st", "SNK": "all"}.get(regime)
        nq = min(Q, len(U))
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds[:nq]]
        sizes = np.asarray([np.asarray(inc[j]).size for j in range(nq)])
        n_st = unions["st"].deg.size
        pos, lens = {}, {}
        for lo in range(0, nq, S4.BATCH):
            js = list(range(lo, min(nq, lo + S4.BATCH)))
            srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
            st3 = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
            k0, a1 = K.restarts_batch(prep, js, linked, unions, srcs, constant)
            stz = [K.steer(prep, j, constant) for j in js]
            a1_4, k_4 = X.walkF(U4, a1, threads), X.walkF(U4, k0, threads)
            at4, kt4 = X.walkF(U4, a1, threads, stz), X.walkF(U4, k0, threads, stz)
            lk = B.walk_mass(unions[reg], [B.uniform(np.asarray(linked[j], np.int64)) for j in js], threads) if reg else None
            a2 = (B.walk_mass(unions[reg], [E.restart("A2", prep.seeds[j], prep.dense_ids[j], prep.splade_ids[j],
                                                      linked[j], unions[reg].deg, constant) for j in js], threads)
                  if reg else None)
            for i, j in enumerate(js):
                src, _nn, (L, Ls) = srcs[i]
                route = D.routes(L, Ls)
                knd = K.kind(Ls, prep.seeds[j], nonly)
                o3, _p1d = K.p1e_orders(src, st3[i], route, knd, n_st, constant)
                src4 = (src[0], a1_4[i], src[2], k_4[i], src[4])
                st4 = dict(st3[i], KTall=kt4[i][0], A1Tall=at4[i][0])
                o4, _ = K.p1e_orders(src4, st4, route, knd, n_st, constant)
                o = {f"{k}|3": v for k, v in o3.items() if k in SHOW}
                o.update({f"{k}|4": v for k, v in o4.items() if k in SHOW})
                o["A1all|3"], o["A1all|4"] = src[1][0][:NMAX], a1_4[i][0][:NMAX]
                o["Kall|3"], o["Kall|4"] = src[3][0][:NMAX], k_4[i][0][:NMAX]
                o["A1st"], o["R0"] = src[0][0][:NMAX], src[4][0][:NMAX]
                o["DENSE"] = np.asarray(prep.dense_ids[j], np.int64)
                o["SPLADE"] = np.asarray(prep.splade_ids[j], np.int64)
                o["LKreg"] = lk[i][0][:NMAX] if reg else np.empty(0, np.int64)
                o["A2reg"] = a2[i][0][:NMAX] if reg else np.empty(0, np.int64)
                for k, v in o.items():
                    pos.setdefault(k, []).append(P.positions(v, golds[j]))
                    lens.setdefault(k, []).append(int(np.asarray(v).size))
        res.update({"dataset": name, "carve": carve_, "questions": nq, "regime": regime, "fourth": counts,
                    "curves": {k: P.curve(pos[k], lens[k]) for k in pos},
                    "at_I_q": {k: P.at_sizes(pos[k], sizes) for k in pos}, "I_q_mean": float(sizes.mean()),
                    "seconds": round(time.time() - t1, 1)})

    E.run_carves(op, name, {carve}, ("A3",), threads, sink)
    OUTD.mkdir(exist_ok=True)
    (OUTD / f"{name}.json").write_text(json.dumps(res), encoding="utf-8")
    X.log(f"{name}: {res['questions']} q in {res['seconds']}s")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if "--host" in sys.argv:
        G.on_the_mirror()
    run(a[0], int(a[1]) if len(a) > 1 else 600, int(a[2]) if len(a) > 2 else 4)
