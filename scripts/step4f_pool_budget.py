"""Step 4f: smaller pools at step 4e's coverage, a diagnostic (docs/STEP4F_POOL_BUDGET.md).

    python scripts/step4f_pool_budget.py coverage [--datasets ...] [--threads 5]     (laptop: numba)
    python scripts/step4f_pool_budget.py verdict
    python scripts/step4f_pool_budget.py --selftest

Two single orders over every candidate, cut at N rows:
    O1 walk   step 4e's A3 walk (restart, kernel, alpha, eps) with nothing excluded, up to NMAX rows
    O2 fused  RRF of O1, the dense top-1000 and the SPLADE top-1000 by 1/(c + rank); ties by node id
squad (no graph): both are the RRF order of its two lists.
coverage  ALL and recall at every N on the five s1sel carves and the six s1eval carves -> outputs/step4f/budget.json
verdict   section 3's rule -> outputs/step4f/verdict.json and verdict.md
Step 4e's prepared inputs are read through its run_carves (its checks against step 4d's pools included); nothing of
step 4e is written.
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import step4e_paper_pools as E  # noqa: E402

OUT = ROOT / "outputs" / "step4f"
BUDGET = OUT / "budget.json"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
NS = (25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7000)
NMAX = NS[-1]
ORDERS = ("O1", "O2")
TOL = E.K_TOL                       # 0.005, step 4e's
BYTES3 = ("metaqa", "musique", "webqsp")
RATIO = 0.5
CHOICE_CARVES = {n: ("s1eval" if n == "webqsp" else "s1sel") for n in E.SIX}
# step 4e's look bytes on the host, measured 9 Oct 12:27 (GiB; metaqa and musique's looks nearly all built)
LOOK_GIB_4E = {"metaqa": 23.0, "musique": 15.9, "webqsp": 3.5, "2wiki": 0.9, "squad": 0.6, "hotpotqa": 0.5}
log, utc, write_json, sha_file = E.log, E.utc, E.write_json, E.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def fused(o1, d_ids, s_ids, constant):
    """RRF of three ranked lists (absent lists add 0); ties by node id."""
    ids = np.concatenate([np.asarray(o1, np.int64), np.asarray(d_ids, np.int64), np.asarray(s_ids, np.int64)])
    w = np.concatenate([1.0 / (constant + np.arange(1, x.size + 1, dtype=np.float64)) for x in (o1, d_ids, s_ids)])
    u, inv = np.unique(ids, return_inverse=True)
    s = np.bincount(inv, weights=w, minlength=u.size)
    return u[np.lexsort((u, -s))][:NMAX]


def gold_last(order, golds):
    """(position of the last gold in the order, or -1 if a gold is missing; positions of the golds found)."""
    pos = np.full(order.size and int(order.max()) + 1 or 1, -1, dtype=np.int64)
    pos[order] = np.arange(order.size)
    g = golds[golds < pos.size]
    p = pos[g] if g.size else np.empty(0, np.int64)
    found = p[p >= 0]
    last = int(p.max()) if g.size == golds.size and golds.size and (p >= 0).all() else -1
    return last, found


def curve(orders, golds):
    """{N: ALL, recall, mean pool size} for the pools order[:N]."""
    nq = len(orders)
    lasts, founds = zip(*[gold_last(o, g) for o, g in zip(orders, golds)]) if nq else ((), ())
    sizes = np.asarray([o.size for o in orders])
    out = {}
    for n in NS:
        a = np.mean([0 <= x < n for x in lasts])
        r = np.mean([(f < n).sum() / max(1, g.size) for f, g in zip(founds, golds)])
        out[str(n)] = {"ALL": round(float(a), 4), "recall": round(float(r), 4),
                       "pool": round(float(np.minimum(sizes, n).mean()), 1)}
    return out


def coverage_stage(names, threads):
    op = E.S4.Opened()
    rec = json.loads(BUDGET.read_text(encoding="utf-8")) if BUDGET.exists() else {"carves": {}}
    rec.update({"declared_in": "docs/STEP4F_POOL_BUDGET.md", "orders": list(ORDERS), "ns": list(NS), "alpha": E.ALPHA,
                "eps": E.S4.EPS, "script_sha256": script_sha256(), "step4e_script_sha256": E.script_sha256(),
                "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"]})

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
        nq = len(U)
        none = [np.empty(0, np.int64)] * nq
        if name == "squad":
            o1 = [E.rrf_order(prep.dense_ids[j], prep.splade_ids[j], none[j], constant, NMAX) for j in range(nq)]
            o2 = o1
        else:
            o1 = []
            for lo in range(0, nq, E.S4.BATCH):
                hi = min(nq, lo + E.S4.BATCH)
                rs = [E.restart("A3", prep.seeds[j], prep.dense_ids[j], prep.splade_ids[j], linked[j], union.deg,
                                constant) for j in range(lo, hi)]
                o1 += E.walk(union, rs, none[lo:hi], NMAX, threads)
            o2 = [fused(o1[j], prep.dense_ids[j], prep.splade_ids[j], constant) for j in range(nq)]
        ent = {"questions": nq, "U_mean": round(float(np.mean([u.size for u in U])), 1),
               "O1_len_mean": round(float(np.mean([o.size for o in o1])), 1),
               "O1_short": int(sum(o.size < NMAX for o in o1)), "orders": {"O1": curve(o1, golds),
                                                                          "O2": curve(o2, golds)},
               "seconds": round(time.time() - t, 1)}
        rec["carves"][f"{name}__{carve}"] = ent
        rec["utc"] = utc()
        write_json(BUDGET, rec)
        log(f"{name}/{carve}: " + " | ".join(
            f"{o} " + " ".join(f"{n}:{ent['orders'][o][str(n)]['ALL']}" for n in (100, 500, 1000, 2000, 3000, 7000))
            for o in ORDERS) + f" ({ent['seconds']:.0f}s)")

    for name in names:
        wanted = {"s1eval"} | ({"s1sel"} if name in E.TRAIN5 else set())
        E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


def decide(bud, cov4e):
    """Section 3: per dataset the target, N*, the order, the ratio; the verdict."""
    per = {}
    for n in E.SIX:
        c = f"{n}__{CHOICE_CARVES[n]}"
        ch = cov4e["carves"][c]["arms"]["A3"]["2"]
        target = ch["ALL"] - TOL
        b = bud["carves"][c]["orders"]
        star = None
        for N in NS:
            hit = [o for o in ORDERS if b[o][str(N)]["ALL"] >= target]
            if hit:
                star = (N, hit[0])
                break
        best = max((b[o][str(N)]["ALL"], -NS.index(N), o) for o in ORDERS for N in NS)
        ev = bud["carves"][f"{n}__s1eval"]["orders"]
        ev4e = cov4e["carves"][f"{n}__s1eval"]["arms"]["A3"]["2"]
        per[n] = {"carve": c, "step4e_ALL": ch["ALL"], "step4e_pool": ch["pool"], "target": round(target, 4),
                  "N_star": star[0] if star else None, "order": star[1] if star else None,
                  "pool_at_N_star": b[star[1]][str(star[0])]["pool"] if star else None,
                  "ratio": round(b[star[1]][str(star[0])]["pool"] / ch["pool"], 3) if star else None,
                  "best_ALL": best[0], "best_at": [NS[-best[1]], best[2]],
                  "s1eval_ALL_at_N_star": ev[star[1]][str(star[0])]["ALL"] if star else None,
                  "s1eval_step4e_ALL": ev4e["ALL"], "s1eval_step4e_pool": ev4e["pool"],
                  "est_look_GiB": round(LOOK_GIB_4E[n] * b[star[1]][str(star[0])]["pool"] / ch["pool"], 1)
                  if star else None, "step4e_look_GiB": LOOK_GIB_4E[n]}
    all_star = all(per[n]["N_star"] is not None for n in E.SIX)
    small = [n for n in BYTES3 if per[n]["ratio"] is not None and per[n]["ratio"] <= RATIO]
    verdict = "SHRINKS" if all_star and len(small) == len(BYTES3) else ("PARTIAL" if small else "NO_SHRINK")
    return per, verdict


def verdict_stage():
    bud = json.loads(BUDGET.read_text(encoding="utf-8"))
    cov4e = json.loads(E.COVERAGE.read_text(encoding="utf-8"))
    need = [f"{n}__{CHOICE_CARVES[n]}" for n in E.SIX] + [f"{n}__s1eval" for n in E.SIX]
    miss = [c for c in need if c not in bud["carves"]]
    if miss:
        raise SystemExit(f"budget.json is missing {miss}")
    per, verdict = decide(bud, cov4e)
    rec = {"declared_in": "docs/STEP4F_POOL_BUDGET.md", "verdict": verdict, "per_dataset": per,
           "budget_sha256": sha_file(BUDGET), "step4e_coverage_sha256": sha_file(E.COVERAGE),
           "script_sha256": script_sha256(), "utc": utc()}
    write_json(VERDICT, rec)
    rows = ["| dataset | step 4e ALL / pool | target | N* (order) | pool at N* | ratio | s1eval ALL at N* / step 4e | "
            "est. looks GiB (4e) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n in E.SIX:
        p = per[n]
        rows.append(f"| {n} | {p['step4e_ALL']} / {p['step4e_pool']:,.0f} | {p['target']} | "
                    f"{p['N_star'] if p['N_star'] else 'none'} ({p['order'] or '-'}) | "
                    f"{p['pool_at_N_star'] if p['pool_at_N_star'] is not None else '-'} | {p['ratio'] or '-'} | "
                    f"{p['s1eval_ALL_at_N_star'] if p['s1eval_ALL_at_N_star'] is not None else '-'} / "
                    f"{p['s1eval_step4e_ALL']} | {p['est_look_GiB'] if p['est_look_GiB'] is not None else '-'} "
                    f"({p['step4e_look_GiB']}) |")
    md = f"# Step 4f verdict: {verdict}\n\n" + "\n".join(rows) + "\n"
    tmp = VERDICT_MD.with_suffix(".tmp")
    tmp.write_text(md, encoding="utf-8", newline="\n")
    os.replace(tmp, VERDICT_MD)
    print(md)
    return 0


def selftest():
    o = fused(np.array([3, 1, 2]), np.array([2, 9]), np.array([2]), 60)
    assert o[0] == 2 and set(o.tolist()) == {1, 2, 3, 9}, o
    last, found = gold_last(np.array([5, 3, 8, 1]), np.array([3, 1]))
    assert last == 3 and sorted(found.tolist()) == [1, 3]
    last, _ = gold_last(np.array([5, 3]), np.array([3, 7]))
    assert last == -1
    c = curve([np.array([5, 3, 8, 1])], [np.array([3, 1])])
    assert c["25"]["ALL"] == 1.0 and c["25"]["pool"] == 4.0
    bud = {"carves": {}}
    cov = {"carves": {}}
    for n in E.SIX:
        for cv in {CHOICE_CARVES[n], "s1eval"}:
            cov["carves"][f"{n}__{cv}"] = {"arms": {"A3": {"2": {"ALL": 0.95, "pool": 7000.0}}}}
            bud["carves"][f"{n}__{cv}"] = {"orders": {o: {str(N): {"ALL": 0.95 if N >= (3000 if o == "O1" else 2000)
                                                                     else 0.5, "pool": float(N)} for N in NS}
                                                      for o in ORDERS}}
    per, v = decide(bud, cov)
    assert v == "SHRINKS" and per["metaqa"]["N_star"] == 2000 and per["metaqa"]["order"] == "O2", (v, per["metaqa"])
    print("selftest: the fused order, gold positions, the curve, N* and the verdict")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("coverage", "verdict"))
    ap.add_argument("--datasets", default=",".join(E.SIX))
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "coverage":
        return coverage_stage([d for d in a.datasets.split(",") if d], a.threads)
    if a.stage == "verdict":
        return verdict_stage()
    ap.error("a stage: coverage or verdict")


if __name__ == "__main__":
    sys.exit(main())
