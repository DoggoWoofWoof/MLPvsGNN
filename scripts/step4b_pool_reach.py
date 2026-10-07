"""Step 4b: step 4's walk on each dataset's own expansion graph (docs/STEP4B_POOL_REACH_FROZEN_GRAPH.md).

    python scripts/step4b_pool_reach.py --stage select [--datasets metaqa,squad] [--threads 5]
    python scripts/step4b_pool_reach.py --stage eval   [--datasets ...] [--threads 5]
    python scripts/step4b_pool_reach.py --stage grade

P_F is step 4's P with the walk's families set to the frozen regime's (m3b_compile.regime_families): a family outside
the regime enters step 4's kernel with no entries, so the walk never picks it and deg counts only the regime's entries.
I is the frozen construction, built by m3b_compile.prepare unchanged. Kernel, populations, reads and bootstrap are
scripts/step4_pool_reach.py's, imported unchanged; its files under outputs/step4 are only read.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import step4_pool_reach as S4  # noqa: E402

m3b_pools = S4.m3b_pools

OUT = ROOT / "outputs" / "step4b"
STEP4_EVAL = S4.OUT / "eval"
SIX, TRAIN, ALPHAS, ALPHA_TIE, EPS, BMAX, BUDGETS = S4.SIX, S4.TRAIN, S4.ALPHAS, S4.ALPHA_TIE, S4.EPS, S4.BMAX, S4.BUDGETS
log = S4.log


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class Empty:
    """A family with no entries, in the store layout the kernel reads."""

    def __init__(self, n):
        self.n_nodes = n
        self.indptr = np.zeros(n + 1, dtype=np.int64)
        self.col = np.empty(0, dtype=np.int32)


class RegimeUnion(S4.Union):
    """Step 4's Union with every family outside the regime empty."""

    def __init__(self, stores, families, n):
        super().__init__({f: stores[f] if f in families else Empty(n) for f in S4.FAMILIES})
        self.families = tuple(families)


def run_dataset(op, name, kind, alphas, threads):
    t0 = time.time()
    ds = op.canonical.Dataset(name, root=str(op.served))
    pop, strata = S4.population(op, ds, name, kind)
    log(f"{name}/{pop.kind}: {len(pop.ids)} questions, golds resolved ({time.time() - t0:.0f}s)")
    construction = op.construction[name]
    regime = construction["regime"]
    families = op.m3b_compile.regime_families(op.cfg_h, regime) if regime != "RETRIEVAL" else []
    stores = {}
    for f in families:
        path = op.m3b_compile.CSR_CACHE / f"{name}_{f}.npz"
        if not path.exists():
            raise SystemExit(f"{path} is missing: the step reads M3B's stores, it does not build them")
        stores[f] = m3b_pools.load_or_build_store(ds, f, op.m3b_compile.CSR_CACHE)
    t1 = time.time()
    prep = op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores, op.m3a, op.m3b_contract)[0]
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    bases = op.m3b_compile.base_rows(construction, prep.dense_ids, prep.splade_ids, op.m3a, op.m3b_contract, constant)
    nq = len(pop.ids)
    seeds, bs, inc = [], [], []
    base_size = np.zeros(nq, dtype=np.int32)
    for j in range(nq):
        s = np.asarray(prep.seeds[j], dtype=np.int64)
        if not np.array_equal(s, m3b_pools.seeds_of(np.asarray(prep.dense_ids[j]), np.asarray(prep.splade_ids[j]))):
            raise SystemExit(f"{pop.ids[j]}: the prepared seeds are not seeds_of(dense top-5, splade top-5)")
        b = np.asarray(bases[j], dtype=np.int64)
        u = np.union1d(b, s)
        pool = np.asarray(prep.pools[j], dtype=np.int64)
        if not np.isin(u, pool).all():
            raise SystemExit(f"{pop.ids[j]}: the incumbent pool does not hold base and seeds")
        seeds.append(s)
        bs.append(u)
        inc.append(pool)
        base_size[j] = np.unique(b).size
    del prep, bases
    gc.collect()
    log(f"{name}/{pop.kind}: incumbent pools built ({time.time() - t1:.0f}s; {construction['base_pool']}, {regime}, "
        f"{(construction.get('setting') or {}).get('name')}; P_F walks {families or 'nothing'})")
    golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
    n_gold = np.asarray([g.size for g in golds], dtype=np.int32)
    gold_ptr = np.zeros(nq + 1, dtype=np.int64)
    np.cumsum(n_gold, out=gold_ptr[1:])
    rec = {"ids": pop.ids, "strata": strata, "n_gold": n_gold, "gold_ptr": gold_ptr,
           "bs_size": np.asarray([u.size for u in bs], dtype=np.int32), "base_size": base_size,
           "seeds_n": np.asarray([s.size for s in seeds], dtype=np.int32),
           "inc_size": np.asarray([p.size for p in inc], dtype=np.int32),
           "gold_in_bs": np.concatenate([np.isin(g, u) for g, u in zip(golds, bs)]),
           "gold_in_I": np.concatenate([np.isin(g, p) for g, p in zip(golds, inc)])}
    rec["B_q"] = (rec["inc_size"] - rec["bs_size"]).astype(np.int32)
    del inc
    union = RegimeUnion(stores, families, int(stores[families[0]].n_nodes) if families else int(ds.n_nodes))
    numba.set_num_threads(threads)
    meta = {"P": {}}
    for alpha in alphas:
        t2 = time.time()
        ranks, avail, work, touched, count = [], [], [], [], []
        for lo in range(0, nq, S4.BATCH):
            hi = min(nq, lo + S4.BATCH)
            orders, av, wk, tc = union.ppr(seeds[lo:hi], bs[lo:hi], alpha, EPS, BMAX, threads)
            ranks += [S4.ranks_in(o, golds[j]) for o, j in zip(orders, range(lo, hi))]
            count += [o.size for o in orders]
            avail.append(av)
            work.append(wk)
            touched.append(tc)
            log(f"  P_F alpha {alpha}: {hi} of {nq} ({time.time() - t2:.0f}s)")
        key = f"{alpha:g}"
        rec[f"gold_rank_P_{key}"] = np.concatenate(ranks)
        rec[f"P_count_{key}"] = np.asarray(count, dtype=np.int32)
        rec[f"P_avail_{key}"] = np.concatenate(avail)
        rec[f"P_work_{key}"] = np.concatenate(work)
        rec[f"P_touched_{key}"] = np.concatenate(touched)
        meta["P"][key] = {"seconds": round(time.time() - t2, 1), "threads": threads}
    if kind == "eval":
        k = min(S4.CONV_N, nq)
        alpha = alphas[0]
        a, *_ = union.ppr(seeds[:k], bs[:k], alpha, EPS, BMAX, threads)
        b, *_ = union.ppr(seeds[:k], bs[:k], alpha, EPS / 10, BMAX, threads)
        ov = [len(set(x.tolist()) & set(y.tolist())) / max(y.size, 1) for x, y in zip(a, b)]
        meta["convergence"] = {"questions": k, "alpha": alpha, "eps": EPS, "against": EPS / 10, "top": BMAX,
                               "overlap_mean": round(float(np.mean(ov)), 4), "overlap_min": round(float(np.min(ov)), 4)}
        log(f"{name}: convergence {meta['convergence']}")
    meta.update({"dataset": name, "carve": pop.kind, "questions": nq, "carve_sha256": S4.digest(pop.ids),
                 "construction": construction, "walk_families": list(families), "frozen_contract": op.frozen_key,
                 "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "carves_sha256": S4.CARVES_SHA256,
                 "script_sha256": script_sha256(), "step4_script_sha256": S4.script_sha256(), "eps": EPS, "bmax": BMAX,
                 "alphas": list(alphas), "nodes": union.n, "walk_entries": int(union.deg.sum()),
                 "seconds": round(time.time() - t0, 1), "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    d = OUT / kind
    d.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in rec.items() if k not in ("ids", "strata")}
    tmp = d / f"{name}.tmp.npz"
    np.savez(tmp, ids=np.asarray(pop.ids), strata=np.asarray(strata), **arrays)
    os.replace(tmp, d / f"{name}.npz")
    S4.write_json(d / f"{name}.json", meta)
    log(f"{name}/{pop.kind}: filed ({time.time() - t0:.0f}s)")
    del stores, union, rec
    gc.collect()


# ── grading ──────────────────────────────────────────────────────────────────


def choose_alpha():
    per = {}
    for name in TRAIN:
        path = OUT / "select" / f"{name}.npz"
        if not path.exists():
            return None
        rd = S4.Read(path)
        i_all = rd.metrics(rd.hits("I"))["ALL"]
        per[name] = {f"{a:g}": round(float((rd.metrics(rd.hits("P", f"{a:g}"))["ALL"] - i_all).mean()), 5) for a in ALPHAS}
    pooled = {f"{a:g}": round(float(np.mean([per[n][f"{a:g}"] for n in TRAIN])), 5) for a in ALPHAS}
    best = max(pooled.values())
    tied = [a for a in ALPHAS if abs(pooled[f"{a:g}"] - best) <= 1e-12]
    alpha = ALPHA_TIE if len(tied) > 1 and ALPHA_TIE in tied else tied[0]
    rec = {"rule": "the restart value with the larger mean over the five datasets of dALL(P_F - I) at matched size on s1sel; a tie goes to 0.5",
           "per_dataset_dALL": per, "pooled_mean_dALL": pooled, "alpha": alpha, "script_sha256": script_sha256(),
           "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    S4.write_json(OUT / "alpha.json", rec)
    log(f"alpha filed: {alpha} (pooled dALL {pooled})")
    return alpha


def step4_union_all(name, a, ids):
    """Step 4's P over the union at this restart value, read at matched size, when step 4 filed it on the same carve."""
    path = STEP4_EVAL / f"{name}.npz"
    if not path.exists():
        return None
    rd = S4.Read(path)
    if f"gold_rank_P_{a}" not in rd.z or rd.z["ids"].tolist() != ids:
        return None
    return rd.metrics(rd.hits("P", a))["ALL"]


def grade():
    alpha_rec = json.loads((OUT / "alpha.json").read_text(encoding="utf-8"))
    a = f"{alpha_rec['alpha']:g}"
    res = {"declared_in": "docs/STEP4B_POOL_REACH_FROZEN_GRAPH.md", "alpha": alpha_rec, "eps": EPS, "per_dataset": {},
           "script_sha256": script_sha256()}
    for name in SIX:
        rd = S4.Read(OUT / "eval" / f"{name}.npz")
        meta = json.loads((OUT / "eval" / f"{name}.json").read_text(encoding="utf-8"))
        z = rd.z
        m = {arm: rd.metrics(rd.hits(arm, a)) for arm in ("I", "P")}
        d_pi = m["P"]["ALL"] - m["I"]["ALL"]
        e = {"questions": rd.nq, "walk_families": meta["walk_families"], "golds_mean": round(float(z["n_gold"].mean()), 3),
             "matched": {arm: {k: round(float(v.mean()), 4) for k, v in m[arm].items()} for arm in m},
             "size": {arm: S4.size_stats(rd.size(arm, a)) for arm in ("I", "P")},
             "short": {"P": rd.short("P", a)},
             "dALL_P_I": S4.boot(d_pi), "drecall_P_I": S4.boot(m["P"]["recall"] - m["I"]["recall"]),
             "B_q": S4.size_stats(z["B_q"].astype(np.int64)), "base_and_seeds": S4.size_stats(z["bs_size"].astype(np.int64))}
        e["label"] = S4.label(e["dALL_P_I"])
        union_all = step4_union_all(name, a, z["ids"].tolist())
        e["step4_P_union_ALL"] = None if union_all is None else round(float(union_all.mean()), 4)
        slices = {}
        for kind, keys in (("stratum", z["strata"]), ("golds", S4.gold_bin(z["n_gold"]))):
            for s in sorted(set(keys.tolist())):
                w = keys == s
                slices[f"{kind}:{s}"] = {"n": int(w.sum()), **{arm: round(float(m[arm]["ALL"][w].mean()), 4) for arm in m},
                                         "step4_P_union": None if union_all is None else round(float(union_all[w].mean()), 4),
                                         "dALL_P_I": S4.boot(d_pi[w])}
        e["slices"] = slices
        frontier = {"I": {"size": round(float(rd.size("I").mean()), 1), "ALL": e["matched"]["I"]["ALL"],
                          "recall": e["matched"]["I"]["recall"]}}
        for b in BUDGETS:
            mm = rd.metrics(rd.hits("P", a, b))
            frontier[f"P{b}"] = {"size": round(float(rd.size("P", a, b).mean()), 1), "ALL": round(float(mm["ALL"].mean()), 4),
                                 "recall": round(float(mm["recall"].mean()), 4)}
        e["frontier"] = frontier
        e["cost"] = {"P_entries_read": S4.size_stats(z[f"P_work_{a}"].astype(np.float64)),
                     "P_nodes_touched_p50": float(np.percentile(z[f"P_touched_{a}"], 50)),
                     "P_thread_ms_per_question": round(meta["P"][a]["seconds"] * meta["P"][a]["threads"] * 1e3 / rd.nq, 1)}
        e["convergence"] = meta.get("convergence")
        res["per_dataset"][name] = e
        log(f"{name}: ALL I {e['matched']['I']['ALL']} P_F {e['matched']['P']['ALL']} (step 4's union walk {e['step4_P_union_ALL']}); "
            f"dALL(P_F-I) {e['dALL_P_I']} {e['label']}")
    labels = {n: res["per_dataset"][n]["label"] for n in SIX}
    res["labels"] = labels
    res["verdict"] = ("NOT_ADOPTED" if "BELOW" in labels.values() else "ADOPT" if "ABOVE" in labels.values() else "NO_EFFECT")
    res["utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    S4.write_json(OUT / "result.json", res)
    (OUT / "result.md").write_text(render(res), encoding="utf-8")
    log(f"verdict {res['verdict']}: {labels}")
    return res


def render(res):
    a = f"{res['alpha']['alpha']:g}"
    ci = S4.ci_text
    fmt = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    lines = [f"# Step 4b result: {res['verdict']}", "",
             f"Restart value {a} (chosen on the select carves: pooled dALL {res['alpha']['pooled_mean_dALL']}), eps {res['eps']:g}.", "",
             "## Matched size (each question's pool has its incumbent's size)", "",
             "| dataset | walk | questions | mean pool | ALL: I / P_F | dALL P_F - I | label | step 4's union walk ALL | recall: I / P_F | short |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, e in res["per_dataset"].items():
        mt = e["matched"]
        lines.append(f"| {n} | {', '.join(e['walk_families']) or 'none'} | {e['questions']:,} | {e['size']['I']['mean']:,.0f} "
                     f"| {mt['I']['ALL']:.3f} / {mt['P']['ALL']:.3f} | {ci(e['dALL_P_I'])} | {e['label']} | {fmt(e['step4_P_union_ALL'])} "
                     f"| {mt['I']['recall']:.3f} / {mt['P']['recall']:.3f} | {e['short']['P']} |")
    lines += ["", "## Slices (ALL at matched size)", "", "| dataset | slice | n | I | P_F | step 4's union walk | dALL P_F - I |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, e in res["per_dataset"].items():
        for s, v in e["slices"].items():
            lines.append(f"| {n} | {s} | {v['n']:,} | {v['I']:.3f} | {v['P']:.3f} | {fmt(v['step4_P_union'])} | {ci(v['dALL_P_I'])} |")
    lines += ["", "## Frontier (mean pool size: ALL)", "", "| dataset | I | " + " | ".join(f"B {b}" for b in BUDGETS) + " |",
              "| --- | --- | " + " | ".join("---" for _ in BUDGETS) + " |"]
    for n, e in res["per_dataset"].items():
        f = e["frontier"]
        cells = " | ".join(f"{f[f'P{b}']['size']:,.0f}: {f[f'P{b}']['ALL']:.3f}" for b in BUDGETS)
        lines.append(f"| {n} P_F | {f['I']['size']:,.0f}: {f['I']['ALL']:.3f} | {cells} |")
    lines += ["", "## Cost and convergence", "",
              "| dataset | P_F entries read p50 / p95 | P_F thread-ms per question | top-2,000 overlap with eps/10 (mean / min) |",
              "| --- | --- | --- | --- |"]
    for n, e in res["per_dataset"].items():
        c, cv = e["cost"], e["convergence"] or {}
        lines.append(f"| {n} | {c['P_entries_read']['p50']:,.0f} / {c['P_entries_read']['p95']:,.0f} | {c['P_thread_ms_per_question']} "
                     f"| {cv.get('overlap_mean')} / {cv.get('overlap_min')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("select", "eval", "grade"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args(argv)
    if a.stage == "grade":
        grade()
        return 0
    names = [n for n in (a.datasets.split(",") if a.datasets else (TRAIN if a.stage == "select" else SIX)) if n]
    allowed = TRAIN if a.stage == "select" else SIX
    if any(n not in allowed for n in names):
        raise SystemExit(f"--datasets: the {a.stage} stage reads {allowed}")
    if a.stage == "eval":
        path = OUT / "alpha.json"
        if not path.exists():
            raise SystemExit("no alpha.json: run the select stage on the five training datasets first")
        alphas = (float(json.loads(path.read_text(encoding="utf-8"))["alpha"]),)
    else:
        alphas = ALPHAS
    op = S4.Opened()
    for name in names:
        run_dataset(op, name, a.stage, alphas, a.threads)
    if a.stage == "select":
        choose_alpha()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
