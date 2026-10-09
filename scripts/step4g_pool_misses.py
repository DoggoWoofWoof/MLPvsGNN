"""Step 4g: where the golds that step 4e's chosen pools miss are (docs/STEP4G_POOL_MISSES.md).

    python scripts/step4g_pool_misses.py miss DATASET [--threads 5] [--host]   -> outputs/step4g/misses_<dataset>.json
    python scripts/step4g_pool_misses.py report                                -> outputs/step4g/report.json, report.md
    python scripts/step4g_pool_misses.py --selftest

The pools are step 4e's (A3, k = 2), rebuilt by step 4e's own code and gated on its filed coverage.json. For every gold
outside its pool: first-stage ranks, A3 walk rank, linking, degree, hop distance from the pool and from the found golds
(breadth-first, up to 3 hops, over the regime's families as the walk reads them), first-hop families, bucket.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import step4e_paper_pools as E  # noqa: E402

OUT = ROOT / "outputs" / "step4g"
REPORT = OUT / "report.json"
REPORT_MD = OUT / "report.md"
ARM, K = "A3", 2.0
HMAX = 3
BUCKETS = ("F", "W", "H1", "H2", "H3", "X")
log, utc, write_json = E.log, E.utc, E.write_json


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ── breadth-first hops over the union's three CSR families ───────────────────


@njit
def _hops(sources, targets, hmax, ip0, c0, ip1, c1, ip2, c2, dist, fam1, touched):
    """dist[t] for every target (-1 beyond hmax), and fam1[t] = bit mask of the families carrying an edge from a
    source into t. dist and fam1 come in all -1 / 0 and are reset before returning."""
    nt = 0
    for k in range(sources.size):
        s = sources[k]
        if dist[s] < 0:
            dist[s] = 0
            touched[nt] = s
            nt += 1
    left = 0
    for k in range(targets.size):
        if dist[targets[k]] < 0:
            left += 1
    lo, hi = 0, nt
    level = 0
    while level < hmax and left > 0 and hi > lo:
        level += 1
        for i in range(lo, hi):
            u = touched[i]
            for f in range(3):
                if f == 0:
                    ip, cc = ip0, c0
                elif f == 1:
                    ip, cc = ip1, c1
                else:
                    ip, cc = ip2, c2
                for e in range(ip[u], ip[u + 1]):
                    v = cc[e]
                    if level == 1 and dist[v] != 0:
                        fam1[v] |= 1 << f
                    if dist[v] < 0:
                        dist[v] = level
                        touched[nt] = v
                        nt += 1
        lo, hi = hi, nt
        left = 0
        for k in range(targets.size):
            if dist[targets[k]] < 0:
                left += 1
    out_d = np.empty(targets.size, dtype=np.int64)
    out_f = np.empty(targets.size, dtype=np.int64)
    for k in range(targets.size):
        out_d[k] = dist[targets[k]]
        out_f[k] = fam1[targets[k]]
    for i in range(nt):
        dist[touched[i]] = -1
        fam1[touched[i]] = 0
    return out_d, out_f


class Hopper:
    def __init__(self, union):
        self.args = union.args[:6]
        n = int(union.deg.size)
        self.dist = np.full(n, -1, dtype=np.int8)
        self.fam1 = np.zeros(n, dtype=np.int8)
        self.touched = np.empty(n, dtype=np.int64)

    def __call__(self, sources, targets):
        if sources.size == 0:
            return np.full(targets.size, -1, np.int64), np.zeros(targets.size, np.int64)
        return _hops(np.asarray(sources, np.int64), np.asarray(targets, np.int64), HMAX, *self.args, self.dist,
                     self.fam1, self.touched)


def rank_in(arr, g):
    w = np.flatnonzero(np.asarray(arr) == g)
    return int(w[0]) + 1 if w.size else 0


def bucket(m):
    if m["dense_rank"] or m["splade_rank"]:
        return "F"
    if m["a3_rank"]:
        return "W"
    return {1: "H1", 2: "H2", 3: "H3"}.get(m["hops_from_pool"], "X")


def kind_of(rec):
    return str(rec.get("hop") or rec.get("type") or "")


# ── the miss stage ───────────────────────────────────────────────────────────


def miss_stage(name, threads):
    op = E.S4.Opened()
    cov = json.loads(E.COVERAGE.read_text(encoding="utf-8"))
    path = OUT / f"misses_{name}.json"
    rec = {"declared_in": "docs/STEP4G_POOL_MISSES.md", "arm": ARM, "k": K, "hmax": HMAX,
           "script_sha256": script_sha256(), "step4e_script_sha256": E.script_sha256(),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "carves": {}}
    kinds = {}

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        key = f"{name}__{carve}"
        want = cov["carves"][key]["arms"][ARM][f"{K:g}"]["ALL"]
        orders = E.arm_orders(ARM, name, union, prep, U, BQ, linked, constant, threads)
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
        pools = [np.union1d(U[j], orders[j][:int(round(K * BQ[j]))]) for j in range(len(U))]
        all_ok = float(np.mean([np.isin(golds[j], pools[j]).all() for j in range(len(U))]))
        if round(all_ok, 4) != want:
            raise SystemExit(f"{key}: rebuilt ALL {all_ok:.4f} is not step 4e's {want}")
        if not kinds:
            kinds.update({i: kind_of(r) for i, r in enumerate(op_queries(op, name))})
        hop = Hopper(union) if families else None
        deg = union.deg
        misses = []
        for j in range(len(U)):
            inp = np.isin(golds[j], pools[j])
            if inp.all():
                continue
            miss = golds[j][~inp]
            found = golds[j][inp]
            if hop is not None:
                dp, f1 = hop(pools[j], miss)
                dg, _ = hop(found, miss)
            else:
                dp = dg = np.full(miss.size, -1, np.int64)
                f1 = np.zeros(miss.size, np.int64)
            for i, g in enumerate(miss.tolist()):
                m = {"q": str(pop.ids[j]), "kind": kinds.get(int(pop.idx[j]), ""), "gold": int(g),
                     "golds": int(golds[j].size), "found": int(found.size),
                     "dense_rank": rank_in(prep.dense_ids[j], g), "splade_rank": rank_in(prep.splade_ids[j], g),
                     "a3_rank": rank_in(orders[j], g), "B_q": int(BQ[j]), "pool": int(pools[j].size),
                     "linked": bool(np.isin(g, linked[j])), "degree": int(deg[g]) if families else 0,
                     "hops_from_pool": int(dp[i]), "hops_from_found": int(dg[i]),
                     "first_hop_families": [f for b, f in enumerate(E.S4.FAMILIES) if int(f1[i]) >> b & 1]}
                m["a3_rank_over_B"] = round(m["a3_rank"] / max(1, m["B_q"]), 3) if m["a3_rank"] else 0
                m["bucket"] = bucket(m)
                misses.append(m)
        nq = len(U)
        rec["carves"][key] = {"questions": nq, "ALL": round(all_ok, 4), "questions_missing": nq - int(round(all_ok * nq)),
                              "misses": len(misses), "pool_mean": round(float(np.mean([p.size for p in pools])), 1),
                              "families": list(families), "rows": misses, "seconds": round(time.time() - t, 1)}
        b = Counter(m["bucket"] for m in misses)
        log(f"{key}: ALL {all_ok:.4f} (gate ok), {len(misses)} misses: " + " ".join(f"{k}={b[k]}" for k in BUCKETS)
            + f" ({time.time() - t:.0f}s)")
        rec["utc"] = utc()
        write_json(path, rec)

    wanted = {"s1eval"} | ({"s1sel"} if name in E.TRAIN5 else set())
    E.run_carves(op, name, wanted, (ARM,), threads, sink)
    return 0


def op_queries(op, name):
    ds = op.canonical.Dataset(name, root=str(op.served))
    return list(ds.queries())


# ── report ───────────────────────────────────────────────────────────────────


def summarize(ent):
    rows = ent["rows"]
    n = max(1, len(rows))
    b = Counter(m["bucket"] for m in rows)
    fam = Counter(f for m in rows if m["bucket"] == "H1" for f in m["first_hop_families"])
    by_kind = defaultdict(Counter)
    for m in rows:
        by_kind[m["kind"]][m["bucket"]] += 1
    return {"questions": ent["questions"], "ALL": ent["ALL"], "questions_missing": ent["questions_missing"],
            "misses": len(rows), "pool_mean": ent["pool_mean"],
            "share": {k: round(b[k] / n, 3) for k in BUCKETS},
            "F_dense_rank_median": float(np.median([m["dense_rank"] for m in rows if m["dense_rank"]] or [0])),
            "W_rank_over_B_median": float(np.median([m["a3_rank_over_B"] for m in rows if m["bucket"] == "W"] or [0])),
            "linked_share": round(sum(m["linked"] for m in rows) / n, 3),
            "degree0_share": round(sum(m["degree"] == 0 for m in rows) / n, 3),
            "within1_of_found_share": round(sum(0 < m["hops_from_found"] <= 1 for m in rows) / n, 3),
            "within2_of_found_share": round(sum(0 < m["hops_from_found"] <= 2 for m in rows) / n, 3),
            "H1_families": dict(fam),
            "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())}}


def report_stage():
    rep = {"declared_in": "docs/STEP4G_POOL_MISSES.md", "script_sha256": script_sha256(), "carves": {}}
    for name in E.SIX:
        p = OUT / f"misses_{name}.json"
        if not p.exists():
            log(f"{name}: no misses file yet")
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["script_sha256"] != rep["script_sha256"]:
            raise SystemExit(f"{p}: another script")
        for key, ent in r["carves"].items():
            rep["carves"][key] = summarize(ent)
    rep["utc"] = utc()
    write_json(REPORT, rep)
    lines = ["# Step 4g: where step 4e's pools miss golds", "",
             "| carve | questions missing | misses | pool | " + " | ".join(BUCKETS) + " | linked | deg 0 | <=2 hops of a found gold |",
             "| --- | --- | --- | --- | " + " | ".join("---" for _ in BUCKETS) + " | --- | --- | --- |"]
    for key, s in rep["carves"].items():
        lines.append(f"| {key} | {s['questions_missing']} / {s['questions']} | {s['misses']} | {s['pool_mean']:,.0f} | "
                     + " | ".join(f"{s['share'][k]:.3f}" for k in BUCKETS)
                     + f" | {s['linked_share']:.3f} | {s['degree0_share']:.3f} | {s['within2_of_found_share']:.3f} |")
    lines += ["", "H1 first-hop families and buckets by question kind:", ""]
    for key, s in rep["carves"].items():
        lines.append(f"- {key}: H1 via {s['H1_families']}; F median dense rank {s['F_dense_rank_median']:g}; "
                     f"W median rank/B {s['W_rank_over_B_median']:g}; by kind {s['by_kind']}")
    tmp = REPORT_MD.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, REPORT_MD)
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    # 0 -> 1 -> 2 -> 3 -> 4 on family 0; 0 -> 2 on family 2; 5 isolated
    n = 6

    def csr(edges):
        ip = np.zeros(n + 1, np.int64)
        for u, _v in edges:
            ip[u + 1] += 1
        ip = np.cumsum(ip)
        col = np.empty(len(edges), np.int32)
        fill = ip[:-1].copy()
        for u, v in sorted(edges):
            col[fill[u]] = v
            fill[u] += 1
        return ip, col

    a = csr([(0, 1), (1, 2), (2, 3), (3, 4)])
    b = csr([])
    c = csr([(0, 2)])

    class U:
        args = (*a, *b, *c, np.zeros(n, np.int64))
        deg = np.zeros(n, np.int64)
    h = Hopper(U)
    d, f = h(np.array([0]), np.array([1, 2, 3, 4, 5, 0]))
    assert d.tolist() == [1, 1, 2, 3, -1, 0], d
    assert f.tolist() == [1, 4, 0, 0, 0, 0], f
    d2, _ = h(np.array([3]), np.array([4, 0]))
    assert d2.tolist() == [1, -1], d2            # direction as stored; state reset between calls
    assert bucket({"dense_rank": 0, "splade_rank": 7, "a3_rank": 0, "hops_from_pool": 1}) == "F"
    assert bucket({"dense_rank": 0, "splade_rank": 0, "a3_rank": 9, "hops_from_pool": 1}) == "W"
    assert bucket({"dense_rank": 0, "splade_rank": 0, "a3_rank": 0, "hops_from_pool": 2}) == "H2"
    assert bucket({"dense_rank": 0, "splade_rank": 0, "a3_rank": 0, "hops_from_pool": -1}) == "X"
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("miss", "report"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "report":
        return report_stage()
    if a.stage != "miss" or a.dataset not in E.SIX:
        ap.error("miss DATASET | report | --selftest")
    if a.host:
        sys.path.insert(0, str(ROOT / "outputs" / "step4f"))
        import run_one
        run_one.on_the_mirror(sys.modules[__name__])
    return miss_stage(a.dataset, a.threads)


if __name__ == "__main__":
    sys.exit(main())
