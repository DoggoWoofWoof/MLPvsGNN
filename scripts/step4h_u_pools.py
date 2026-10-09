"""Step 4h: precise pools on the universal graphs (docs/STEP4H_U_POOLS.md).

    python scripts/step4h_u_pools.py coverage --datasets D [--threads 5] [--host] [--graphs GU,G0]
                                                                        -> outputs/step4h/coverage_<D>.json
    python scripts/step4h_u_pools.py choose                             -> outputs/step4h/choice.json, report.md
    python scripts/step4h_u_pools.py --selftest

On the host it runs as `python outputs/host_ops/pylib_run.py scripts/step4h_u_pools.py ...` (numba from pylib).

Every pool is base and seeds (bs) plus walk-ranked nodes outside bs. The walk is step 4e's chosen arm (A3: the RRF
lists and linked seeds, specificity weights, restart 0.5) over the regime's families, on one of two graphs:
    GU  `structural` = U1d's structural_U (outputs/u1d/<D>/graph_structural_u.npz; sha256 from its build.json)
    G0  `structural` = today's (M3B's store), reported beside
Two arms, each at m x B_q nodes past bs (B_q = step 4d's budget of the question):
    W(m)  the walk's first m x B_q nodes
    B(m)  the walk's first (m - 1) x B_q nodes, then B_q bridge nodes: nodes outside that pool adjacent (over the
          regime's families) to at least one of the question's top nodes (its seeds and the walk's first TOP nodes),
          ranked by how many top nodes they touch, then by walk rank, then by position; fewer than B_q are topped up
          from the walk.
squad (no graph): every arm is the RRF order of the two top-1000 lists, as step 4e.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src", ROOT / "outputs" / "step4f"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import step4e_paper_pools as E  # noqa: E402

C4, D4, S4 = E.C4, E.D4, E.S4
OUT = ROOT / "outputs" / "step4h"
U1D = ROOT / "outputs" / "u1d"
ARM = "A3"                     # step 4e's choice (outputs/step4e/choice.json)
MS = (1, 2, 3, 4, 6)
TOP = 10
ALL_TIE = 0.002
TRAIN5, SIX, log, utc, write_json = E.TRAIN5, E.SIX, E.log, E.utc, E.write_json


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def u_store(name, n):
    from mp_retrieval.m3b_pools import FamilyStore
    b = json.loads((U1D / name / "build.json").read_text(encoding="utf-8"))
    p = U1D / name / "graph_structural_u.npz"
    if hashlib.sha256(p.read_bytes()).hexdigest() != b["npz_sha256"] or int(b["n_nodes"]) != n:
        raise SystemExit(f"{p}: not U1d's build ({b.get('variant')})")
    z = np.load(p)
    g = SimpleNamespace(n_nodes=n, src=z["src"], dst=z["dst"], rel=z["rel"], weight=None)
    st = FamilyStore.from_graph(g, "structural")
    return st, {"variant": b["variant"], "npz_sha256": b["npz_sha256"], "edges": int(z["src"].size)}


def bridge_order(fams, top, pool, walk_rank, want):
    """Nodes outside pool adjacent to the top nodes, by (touching top nodes desc, walk rank, position)."""
    nb = []
    for t in top:
        cols = [f.col[f.indptr[t]:f.indptr[t + 1]] for f in fams]
        c = np.unique(np.concatenate(cols).astype(np.int64)) if cols else np.empty(0, np.int64)
        nb.append(c)
    if not nb:
        return np.empty(0, np.int64)
    allc = np.concatenate(nb)
    u, cnt = np.unique(allc, return_counts=True)
    keep = ~np.isin(u, pool)
    u, cnt = u[keep], cnt[keep]
    wr = np.array([walk_rank.get(int(x), 1 << 40) for x in u], dtype=np.int64)
    o = np.lexsort((u, wr, -cnt))
    return u[o][:want]


def pools_for(order, bs, seeds, BQ, m, arm, fams):
    k = int(m * BQ)
    if arm == "W" or fams is None:
        return np.union1d(bs, order[:k])
    first = np.union1d(bs, order[:max(0, k - BQ)])
    top = np.union1d(seeds, order[:TOP])
    walk_rank = {int(x): r for r, x in enumerate(order.tolist())}
    br = bridge_order(fams, top, first, walk_rank, BQ)
    p = np.union1d(first, br)
    if br.size < BQ:
        rest = order[~np.isin(order, p)][:BQ - br.size]
        p = np.union1d(p, rest)
    return p


def coverage_stage(names, threads, graphs, host):
    import numba
    numba.set_num_threads(threads)
    op = S4.Opened()
    for name in names:
        out = OUT / f"coverage_{name}.json"
        rec = {"declared_in": "docs/STEP4H_U_POOLS.md", "dataset": name, "arm": ARM, "ms": list(MS), "top": TOP,
               "script_sha256": script_sha256(), "step4e_sha256": E.script_sha256(), "host": bool(host),
               "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "carves": {}}
        t0 = time.time()
        ds, construction, families, stores, union0 = E.open_dataset(op, name)
        n = int(ds.n_nodes)
        unions = {}
        if "G0" in graphs:
            unions["G0"] = union0
        if "GU" in graphs:
            if name == "squad" or not families:
                unions["GU"] = union0
                rec["GU"] = {"note": "no graph family is read (step 4e's RRF order)"}
            else:
                import step4b_pool_reach as S4B
                st, info = u_store(name, n)
                s2 = dict(stores)
                s2["structural"] = st
                if "structural" not in families:
                    raise SystemExit(f"{name}: the regime has no structural family")
                unions["GU"] = S4B.RegimeUnion(s2, families, n)
                rec["GU"] = info
        wanted = {"s1eval"} | ({"s1sel"} if name in TRAIN5 else set())
        pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c in wanted]
        index = texts = None
        if name != "squad":
            index = E.build_index(E.node_names(ds), E.stopwords())
            texts = E.query_texts(ds)
        preps = op.m3b_compile.prepare(ds, [p for _c, p in pops], construction, op.cfg_h, stores, op.m3a,
                                       op.m3b_contract)
        constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
        cov4e = json.loads(E.COVERAGE.read_text(encoding="utf-8"))["carves"]
        for (carve, pop), prep in zip(pops, preps):
            bases = op.m3b_compile.base_rows(construction, prep.dense_ids, prep.splade_ids, op.m3a, op.m3b_contract,
                                             constant)
            U, BQ, bs, inc, _f = E.union_pools(name, carve, pop, prep, bases)
            del bases, U
            linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [np.empty(0, np.int64)] * len(bs)
            golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
            nq = len(bs)
            bmax = max(1, int(math.ceil(max(MS) * int(BQ.max())))) if nq else 1
            ent = {"questions": nq, "B_q_mean": round(float(BQ.mean()), 1),
                   "bs_mean": round(float(np.mean([b.size for b in bs])), 1),
                   "step4e": cov4e.get(f"{name}__{carve}", {}).get("arms", {}).get("A3", {}).get("2"),
                   "graphs": {}}
            for g, un in unions.items():
                tg = time.time()
                if name == "squad":
                    orders = [E.rrf_order(prep.dense_ids[j], prep.splade_ids[j], bs[j], constant, bmax) for j in range(nq)]
                    fams = None
                else:
                    orders = []
                    for lo in range(0, nq, S4.BATCH):
                        hi = min(nq, lo + S4.BATCH)
                        rs = [E.restart(ARM, prep.seeds[j], prep.dense_ids[j], prep.splade_ids[j], linked[j], un.deg,
                                        constant) for j in range(lo, hi)]
                        orders += E.walk(un, rs, bs[lo:hi], bmax, threads)
                    fams = [f for f in un.fams if f.indptr[-1] > 0]
                ge = {}
                for armname in ("W", "B"):
                    if armname == "B" and fams is None:
                        continue
                    for m in MS:
                        if armname == "B" and m < 2:
                            continue
                        a = rec_ = size = 0.0
                        for j in range(nq):
                            p = pools_for(orders[j], bs[j], np.asarray(prep.seeds[j], np.int64), int(BQ[j]), m,
                                          armname, fams)
                            hit = np.isin(golds[j], p)
                            a += hit.all()
                            rec_ += hit.mean()
                            size += p.size
                        ge[f"{armname}{m}"] = {"ALL": round(a / nq, 4), "recall": round(rec_ / nq, 4),
                                               "pool": round(size / nq, 1)}
                ge["seconds"] = round(time.time() - tg, 1)
                ent["graphs"][g] = ge
                log(f"{name}/{carve} {g}: " + " ".join(f"{k}={v['ALL']}/{v['pool']:.0f}" for k, v in ge.items()
                                                      if isinstance(v, dict)) + f" ({ge['seconds']:.0f}s)")
                del orders
            rec["carves"][carve] = ent
            rec["utc"] = utc()
            write_json(out, rec)
            gc.collect()
        log(f"{name}: done ({time.time() - t0:.0f}s)")
    return 0


def choose():
    recs = {d: json.loads((OUT / f"coverage_{d}.json").read_text(encoding="utf-8")) for d in SIX}
    reads = [(d, "s1sel") for d in TRAIN5] + [("webqsp", "s1eval")]
    cfgs = [k for k in recs["metaqa"]["carves"]["s1sel"]["graphs"]["GU"] if k != "seconds"]
    rows = {}
    for c in cfgs:
        alls, ratios = [], []
        for d, carve in reads:
            ent = recs[d]["carves"][carve]
            v = ent["graphs"]["GU"].get(c) or ent["graphs"]["GU"][c.replace("B", "W")]   # squad: B is W
            alls.append(v["ALL"])
            ratios.append(v["pool"] / ent["step4e"]["pool"])
        rows[c] = {"mean_ALL": round(float(np.mean(alls)), 4), "size_ratio_to_4e": round(float(np.mean(ratios)), 3)}
    ok = {c: r for c, r in rows.items() if r["size_ratio_to_4e"] <= 1.0}
    best = max(r["mean_ALL"] for r in ok.values())
    near = [c for c, r in ok.items() if r["mean_ALL"] >= best - ALL_TIE]
    chosen = min(near, key=lambda c: (ok[c]["size_ratio_to_4e"], c))
    out = {"rule": "highest mean ALL over the five s1sel carves and webqsp's s1eval on GU, among configurations whose "
                   "mean pool size is at most step 4e's (ratio <= 1); within 0.002 the smaller pools",
           "configs": rows, "chosen": chosen, "utc": utc()}
    write_json(OUT / "choice.json", out)
    md = ["# Step 4h: pools on the universal graphs", "", f"Chosen: **{chosen}**", "",
          "| config | mean ALL (choice reads) | pool / step 4e |", "| --- | ---: | ---: |"]
    md += [f"| {c} | {r['mean_ALL']} | {r['size_ratio_to_4e']} |" for c, r in rows.items()]
    md += ["", "## Every carve (ALL / mean pool)", "",
           "| dataset | carve | step 4e (G0, A3 k=2) | G0 " + chosen + " | GU " + chosen + " | GU W2 | GU W4 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for d in SIX:
        for carve, ent in recs[d]["carves"].items():
            def f(g, c):
                v = ent["graphs"].get(g, {}).get(c) or ent["graphs"].get(g, {}).get(c.replace("B", "W"))
                return f"{v['ALL']} / {v['pool']:.0f}" if v else "-"
            e4 = ent["step4e"]
            md.append(f"| {d} | {carve} | {e4['ALL']} / {e4['pool']:.0f} | {f('G0', chosen)} | {f('GU', chosen)} | "
                      f"{f('GU', 'W2')} | {f('GU', 'W4')} |")
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


def selftest():
    fams = [SimpleNamespace(indptr=np.array([0, 2, 3, 5, 6, 6]), col=np.array([1, 2, 0, 0, 4, 2]))]
    # node 0 - {1, 2}; 1 - {0}; 2 - {0, 4}; 3 - {2}; 4 - {}
    br = bridge_order(fams, np.array([0, 2]), np.array([0, 2]), {1: 5, 4: 1}, 5)
    assert br.tolist() == [4, 1], br                  # 4 and 1 touch one top node each; 4 has the better walk rank
    order = np.array([1, 4, 3])
    p = pools_for(order, np.array([0, 2]), np.array([0]), 1, 2, "B", fams)
    assert p.tolist() == [0, 1, 2, 4], p
    assert pools_for(order, np.array([0, 2]), np.array([0]), 1, 2, "W", fams).tolist() == [0, 1, 2, 4]
    print("selftest passed")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("coverage", "choose"))
    ap.add_argument("--datasets", nargs="+", default=list(SIX))
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--graphs", default="GU,G0")
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args()
    if a.stage == "choose":
        return choose()
    if a.host:
        import run_one
        run_one.on_the_mirror(SimpleNamespace(E=E))
    OUT.mkdir(parents=True, exist_ok=True)
    return coverage_stage(a.datasets, a.threads, a.graphs.split(","), a.host)


if __name__ == "__main__":
    sys.exit(main())
