"""Step 4h: precise pools on the universal graphs (docs/STEP4H_U_POOLS.md).

    python scripts/step4h_u_pools.py coverage --datasets D [--threads 5] [--host] [--graphs GU,G0]
                                                                        -> outputs/step4h/coverage_<D>.json
    python scripts/step4h_u_pools.py choose                             -> outputs/step4h/choice.json, report.md
    python scripts/step4h_u_pools.py --selftest

On the host it runs as `python outputs/host_ops/pylib_run.py scripts/step4h_u_pools.py ...` (numba from pylib).

Amended 10 October about 01:45 (docs/STEP4H_U_POOLS.md, "Amendment"), before any number on GU: every pool has step
4e's shape, each stage run on the graph G:
    I^G  the frozen construction (m3b_compile.prepare unchanged) with `structural` = G's
    U^G  I^G plus step 4c's walk on G (seeds, restart 0.5) first B_q nodes          (step 4d's union pool)
    W(k) U^G plus the A3 walk on G's first k x B_q nodes outside U^G, k in KS        (step 4e's pools)
    B(k) U^G plus the A3 walk's first (k - 1) x B_q nodes, then B_q bridge nodes: nodes outside that pool adjacent
         (over the regime's families) to the question's top nodes (its seeds and the A3 walk's first TOP nodes), ranked
         by how many top nodes they touch, then by walk rank, then by position; topped up from the walk.
G is GU (`structural` = U1d's structural_U; outputs/u1d/<D>/graph_structural_u.npz, sha256 from its build.json) or G0
(today's). B_q is step 4d's filed budget on both. On G0, I^G, U^G and every W(k) must reproduce steps 4c-4e (checked).
squad (no graph): I, U and every arm are step 4e's RRF pools on both graphs.
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
KS = (0.0, 0.5, 1.0, 2.0, 3.0)    # step 4e's
BK = (1, 2, 3)
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


def bridge_pool(U, order, seeds, BQ, k, fams):
    first = np.union1d(U, order[:max(0, (k - 1) * BQ)])
    top = np.union1d(seeds, order[:TOP])
    walk_rank = {int(x): r for r, x in enumerate(order.tolist())}
    br = bridge_order(fams, top, first, walk_rank, BQ)
    p = np.union1d(first, br)
    if br.size < BQ:
        rest = order[~np.isin(order, p)][:BQ - br.size]
        p = np.union1d(p, rest)
    return p


def tally(pools, golds):
    a = rec = size = 0.0
    for p, g in zip(pools, golds):
        hit = np.isin(g, p)
        a += hit.all()
        rec += hit.mean()
        size += p.size
    n = len(golds)
    return {"ALL": round(a / n, 4), "recall": round(rec / n, 4), "pool": round(size / n, 1)}


def coverage_stage(names, threads, graphs, host):
    import numba
    import step4b_pool_reach as S4B
    numba.set_num_threads(threads)
    alpha_path = C4.STEP4B / "alpha.json"
    if float(json.loads(alpha_path.read_text(encoding="utf-8"))["alpha"]) != C4.ALPHA:
        raise SystemExit(f"{alpha_path}: not the restart value {C4.ALPHA}")
    op = S4.Opened()
    cov4e = json.loads(E.COVERAGE.read_text(encoding="utf-8"))["carves"]
    for name in names:
        out = OUT / f"coverage_{name}.json"
        rec = {"declared_in": "docs/STEP4H_U_POOLS.md", "dataset": name, "arm": ARM, "ks": [f"{k:g}" for k in KS],
               "bridge_ks": list(BK), "top": TOP, "script_sha256": script_sha256(), "step4e_sha256": E.script_sha256(),
               "host": bool(host), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "carves": {}}
        t0 = time.time()
        ds, construction, families, stores, union0 = E.open_dataset(op, name)
        n = int(ds.n_nodes)
        nograph = name == "squad" or not families
        G = {}
        if "G0" in graphs or "GU" in graphs:
            G["G0"] = (stores, union0)          # always run: its identity to steps 4c-4e checks the code
        if "GU" in graphs and not nograph:
            if "structural" not in families:
                raise SystemExit(f"{name}: the regime has no structural family")
            st, info = u_store(name, n)
            s2 = dict(stores)
            s2["structural"] = st
            G["GU"] = (s2, S4B.RegimeUnion(s2, families, n))
            rec["GU"] = info
        elif "GU" in graphs:
            rec["GU"] = {"note": "no graph family is read: GU's pools are G0's (step 4e's RRF pools)"}
        wanted = {"s1eval"} | ({"s1sel"} if name in TRAIN5 else set())
        pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c in wanted]
        index = texts = None
        if not nograph:
            index = E.build_index(E.node_names(ds), E.stopwords())
            texts = E.query_texts(ds)
        constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
        preps = {g: op.m3b_compile.prepare(ds, [p for _c, p in pops], construction, op.cfg_h, st_, op.m3a,
                                           op.m3b_contract) for g, (st_, _u) in G.items()}
        log(f"{name}: prepared {', '.join(G)} ({time.time() - t0:.0f}s)")
        for ci, (carve, pop) in enumerate(pops):
            prep0 = preps["G0"][ci]
            bases = op.m3b_compile.base_rows(construction, prep0.dense_ids, prep0.splade_ids, op.m3a, op.m3b_contract,
                                             constant)
            U0, BQ, bs, inc0, _f = E.union_pools(name, carve, pop, prep0, bases)
            del bases
            linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [np.empty(0, np.int64)] * len(bs)
            golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
            nq = len(bs)
            filed = cov4e.get(f"{name}__{carve}", {}).get("arms", {}).get("A3", {})
            ent = {"questions": nq, "B_q_mean": round(float(BQ.mean()), 1),
                   "bs_mean": round(float(np.mean([b.size for b in bs])), 1),
                   "step4e": filed.get("2"), "graphs": {}}
            for g, (_st, un) in G.items():
                tg = time.time()
                prep = preps[g][ci]
                seeds = [np.asarray(s, np.int64) for s in prep.seeds]
                if any(not np.array_equal(a, b) for a, b in zip(seeds, prep0.seeds)):
                    raise SystemExit(f"{name}/{carve}/{g}: seeds differ from G0's")
                inc = [np.asarray(p, np.int64) for p in prep.pools]
                if nograph:
                    U = U0
                else:
                    porder = []
                    for lo in range(0, nq, S4.BATCH):
                        hi = min(nq, lo + S4.BATCH)
                        porder += un.ppr(seeds[lo:hi], bs[lo:hi], C4.ALPHA, S4.EPS, S4.BMAX, threads)[0]
                    U = [np.union1d(inc[j], np.asarray(porder[j][:int(BQ[j])], np.int64)) for j in range(nq)]
                    del porder
                ge = {"I": tally(inc, golds), "U": tally(U, golds)}
                orders = E.arm_orders(ARM, name, un, prep, U, BQ, linked, constant, threads)
                ge.update({f"W{k}": v for k, v in E.coverage_of(orders, U, BQ, golds).items()})
                if not nograph:
                    fams = [f for f in un.fams if f.indptr[-1] > 0]
                    for k in BK:
                        ge[f"B{k}"] = tally([bridge_pool(U[j], orders[j], seeds[j], int(BQ[j]), k, fams)
                                             for j in range(nq)], golds)
                if g == "G0":
                    same = {"I": all(np.array_equal(a, b) for a, b in zip(inc, inc0)),
                            "U": all(np.array_equal(a, b) for a, b in zip(U, U0)),
                            "W": bool(filed) and all(ge[f"W{k}"] == v for k, v in filed.items() if isinstance(v, dict))}
                    ge["identity_to_step4e"] = same
                    if not all(same.values()):
                        raise SystemExit(f"{name}/{carve}: G0 does not reproduce steps 4c-4e: {same}")
                ge["seconds"] = round(time.time() - tg, 1)
                ent["graphs"][g] = ge
                log(f"{name}/{carve} {g}: " + " ".join(f"{k}={v['ALL']}/{v['pool']:.0f}" for k, v in ge.items()
                                                      if isinstance(v, dict) and "ALL" in v) + f" ({ge['seconds']:.0f}s)")
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
    gu = lambda ent: ent["graphs"].get("GU") or ent["graphs"]["G0"]   # noqa: E731  (squad: no graph, GU is G0)
    cfgs = [k for k, v in recs["metaqa"]["carves"]["s1sel"]["graphs"]["GU"].items()
            if isinstance(v, dict) and k[0] in "WB"]
    rows = {}
    for c in cfgs:
        alls, ratios = [], []
        for d, carve in reads:
            ent = recs[d]["carves"][carve]
            v = gu(ent).get(c) or gu(ent)[c.replace("B", "W")]   # squad: B(k) is W(k)
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
           "| dataset | carve | step 4e (G0, A3 k=2) | G0 " + chosen + " | GU " + chosen + " | GU W2 | GU W3 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for d in SIX:
        for carve, ent in recs[d]["carves"].items():
            def f(g, c):
                gg = ent["graphs"].get(g) or (ent["graphs"]["G0"] if g == "GU" else {})
                v = gg.get(c) or gg.get(c.replace("B", "W"))
                return f"{v['ALL']} / {v['pool']:.0f}" if v else "-"
            e4 = ent["step4e"]
            md.append(f"| {d} | {carve} | {e4['ALL']} / {e4['pool']:.0f} | {f('G0', chosen)} | {f('GU', chosen)} | "
                      f"{f('GU', 'W2')} | {f('GU', 'W3')} |")
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


def selftest():
    fams = [SimpleNamespace(indptr=np.array([0, 2, 3, 5, 6, 6]), col=np.array([1, 2, 0, 0, 4, 2]))]
    # node 0 - {1, 2}; 1 - {0}; 2 - {0, 4}; 3 - {2}; 4 - {}
    br = bridge_order(fams, np.array([0, 2]), np.array([0, 2]), {1: 5, 4: 1}, 5)
    assert br.tolist() == [4, 1], br                  # 4 and 1 touch one top node each; 4 has the better walk rank
    order = np.array([1, 4, 3])
    p = bridge_pool(np.array([0, 2]), order, np.array([0]), 1, 2, fams)
    assert p.tolist() == [0, 1, 2, 4], p             # U and the walk's first node; no bridge left, topped up by 4
    p = bridge_pool(np.array([0, 2]), order, np.array([0]), 1, 1, fams)
    assert p.tolist() == [0, 1, 2], p                # top = {0, 1, 3, 4}: only 1 lies next to them outside U
    t = tally([np.array([0, 1]), np.array([2])], [np.array([0, 1]), np.array([2, 3])])
    assert t == {"ALL": 0.5, "recall": 0.75, "pool": 1.5}, t
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
