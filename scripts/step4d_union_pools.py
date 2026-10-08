"""Step 4d: union pools, I_q and P_F together (docs/STEP4D_UNION_POOLS.md).

    python scripts/step4d_union_pools.py pools [--datasets metaqa,musique] [--threads 5]      (laptop: numba)
    python scripts/step4d_union_pools.py --selftest

pools   Section 2. Per carve, each question's frozen pool I_q (m3b_compile.prepare, unchanged), its base and seeds, and
        step 4c's walk order (step 4b's RegimeUnion over the frozen regime's families, step 4's kernel, restart 0.5,
        eps 1e-7, the first 2,000 nodes). The union expansion is I_q's own expansion (I_q minus base and seeds, in
        node order) followed by the walk order's first B_q nodes that I_q does not hold, B_q = |I_q| - |base and
        seeds| (step 4c's budget). The union pool U_q = base and seeds and that expansion holds I_q, so no question
        loses a node, and so no gold, that its frozen pool holds. Written to outputs/step4d/pools/<dataset>__<carve>.npz
        (step 4c's layout: the ids, |I_q|, sha256 of I_q, base and seeds, B_q, the expansion) with the manifest
        pools.json (each file's sha256, the pool sizes, every gold in the pool and the share of golds in the pool on
        I_q, step 4c's P_F and U_q). On the s1sel and s1eval carves the walk order must equal step 4b's filed arrays,
        as in step 4c.

Everything but the expansion is step 4c's (scripts/step4c_walk_pools.py, imported unchanged): the populations, the
frozen pools, the walk, the checks against step 1's cache and step 4b.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import step4c_walk_pools as C4  # noqa: E402

OUT = ROOT / "outputs" / "step4d"
POOLS = OUT / "pools"
MANIFEST = POOLS / "pools.json"
LOOK = OUT / "look"
SIX, ALPHA, log, utc, sha_file, write_json = C4.SIX, C4.ALPHA, C4.log, C4.utc, C4.sha_file, C4.write_json


def script_sha256():
    return C4.hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def union_expansion(bs, inc, order, b_q):
    """I_q's expansion (inc minus bs, sorted), then order's first b_q nodes outside inc, in order. Returns
    (expansion, walk nodes taken, walk nodes new to I_q)."""
    own = np.setdiff1d(np.asarray(inc, np.int64), np.asarray(bs, np.int64))
    walk = np.asarray(order[:int(b_q)], np.int64)
    new = walk[~np.isin(walk, inc)]
    return np.concatenate([own, new]), int(walk.size), int(new.size)


def filed_arrays(ids, bs, inc, orders):
    """step 4c's file layout with the union expansion; count is the walk nodes taken (count < B_q: short)."""
    inc_size = np.asarray([p.size for p in inc], dtype=np.int32)
    bs_size = np.asarray([u.size for u in bs], dtype=np.int32)
    b_q = (inc_size - bs_size).astype(np.int32)
    exps, count, new = [], [], []
    for u, p, o, b in zip(bs, inc, orders, b_q):
        e, c, n = union_expansion(u, p, o, b)
        exps.append(e)
        count.append(c)
        new.append(n)
    ptr, flat = C4.flat_of(exps)
    bs_ptr, bs_flat = C4.flat_of(bs)
    if max([int(flat.max()) if flat.size else 0, int(bs_flat.max()) if bs_flat.size else 0]) >= 2 ** 31:
        raise SystemExit("a node position does not fit int32")
    new = np.asarray(new, dtype=np.int32)
    return {"ids": np.asarray(list(ids)), "inc_size": inc_size, "bs_size": bs_size, "B_q": b_q,
            "count": np.asarray(count, dtype=np.int32), "new": new, "ptr": ptr, "nodes": flat.astype(np.int32),
            "bs_ptr": bs_ptr, "bs_nodes": bs_flat.astype(np.int32),
            "inc_sha": np.asarray([C4.pool_sha(p) for p in inc], dtype="S64"), "changed": new > 0}


def file_carve(S4, op, name, carve, pop, prep, union, construction, families, threads):
    t0 = time.time()
    m3c = op.m3b_compile
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    bases = m3c.base_rows(construction, prep.dense_ids, prep.splade_ids, op.m3a, op.m3b_contract, constant)
    nq = len(pop.ids)
    seeds, bs, inc = [], [], []
    for j in range(nq):
        s = np.asarray(prep.seeds[j], dtype=np.int64)
        if not np.array_equal(s, S4.m3b_pools.seeds_of(np.asarray(prep.dense_ids[j]), np.asarray(prep.splade_ids[j]))):
            raise SystemExit(f"{pop.ids[j]}: the prepared seeds are not seeds_of(dense top-5, splade top-5)")
        u = np.union1d(np.asarray(bases[j], dtype=np.int64), s)
        pool = np.asarray(prep.pools[j], dtype=np.int64)
        if not np.isin(u, pool).all():
            raise SystemExit(f"{pop.ids[j]}: the frozen pool does not hold base and seeds")
        seeds.append(s)
        bs.append(u)
        inc.append(pool)
    del bases
    inc_size = np.asarray([p.size for p in inc], dtype=np.int32)
    host_rows = C4.step1_rows(name, carve, list(pop.ids), inc_size)
    orders = []
    for lo in range(0, nq, S4.BATCH):
        hi = min(nq, lo + S4.BATCH)
        o, _av, _wk, _tc = union.ppr(seeds[lo:hi], bs[lo:hi], ALPHA, S4.EPS, S4.BMAX, threads)
        orders += o
    arrays = filed_arrays(pop.ids, bs, inc, orders)
    ptr, flat = arrays["ptr"], arrays["nodes"]
    golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
    step4b = None
    if carve in C4.STEP4B_KIND:
        path = C4.STEP4B / C4.STEP4B_KIND[carve] / f"{name}.npz"
        with np.load(path) as z:
            ref = {k: z[k] for k in ("ids", "inc_size", "bs_size", "B_q", "P_count_0.5", "gold_rank_P_0.5")}
        ranks = np.concatenate([S4.ranks_in(o, g) for o, g in zip(orders, golds)])
        same = {"ids": ref["ids"].tolist() == list(pop.ids),
                "inc_size": np.array_equal(ref["inc_size"], arrays["inc_size"]),
                "bs_size": np.array_equal(ref["bs_size"], arrays["bs_size"]), "B_q": np.array_equal(ref["B_q"], arrays["B_q"]),
                "order_length": np.array_equal(ref["P_count_0.5"], np.asarray([o.size for o in orders], dtype=np.int32)),
                "gold_ranks": np.array_equal(ref["gold_rank_P_0.5"], ranks)}
        if not all(same.values()):
            raise SystemExit(f"{name}/{carve}: not step 4b's filed arrays: {same}")
        step4b = {"file": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(path), "equal": same}
    cov = {k: np.zeros(nq) for k in ("all_I", "all_P", "all_U", "rec_I", "rec_P", "rec_U")}
    u_size = np.zeros(nq, dtype=np.int64)
    for j in range(nq):
        upool = np.union1d(bs[j], flat[ptr[j]:ptr[j + 1]].astype(np.int64))
        if not np.isin(inc[j], upool).all():
            raise SystemExit(f"{pop.ids[j]}: the union pool does not hold the frozen pool")
        u_size[j] = upool.size
        pf = np.union1d(bs[j], np.asarray(orders[j][:int(arrays["B_q"][j])], np.int64))
        for key, p in (("I", inc[j]), ("P", pf), ("U", upool)):
            g = np.isin(golds[j], p)
            cov[f"all_{key}"][j], cov[f"rec_{key}"][j] = g.all(), g.mean()
    POOLS.mkdir(parents=True, exist_ok=True)
    fname = f"{name}__{carve}.npz"
    tmp = POOLS / f"{name}__{carve}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, POOLS / fname)
    short = int((arrays["count"] < arrays["B_q"]).sum())
    r4 = lambda x: round(float(x), 4)  # noqa: E731
    ent = {"file": fname, "sha256": sha_file(POOLS / fname), "dataset": name, "carve": carve, "questions": nq,
           "carve_ids_sha256": pop.digest, "zero_gold_excluded": int(pop.zero_gold_excluded),
           "construction": construction, "walk_families": list(families), "changed": int(arrays["changed"].sum()),
           "short": short, "B_q": {"mean": round(float(arrays["B_q"].mean()), 1) if nq else None},
           "pool_mean": {"I": round(float(inc_size.mean()), 1), "U": round(float(u_size.mean()), 1)},
           "pool_max": {"I": int(inc_size.max()), "U": int(u_size.max())},
           "new_nodes_mean": round(float(arrays["new"].mean()), 1),
           "ALL": {k: r4(cov[f"all_{k[0]}"].mean()) for k in ("I", "P_F", "U")},
           "recall": {k: r4(cov[f"rec_{k[0]}"].mean()) for k in ("I", "P_F", "U")},
           "step4b": step4b, "step1_cache_rows": host_rows, "seconds": round(time.time() - t0, 1), "utc": utc()}
    log(f"{name}/{carve}: {nq} questions; pool I {ent['pool_mean']['I']} -> U {ent['pool_mean']['U']}; ALL {ent['ALL']}; "
        f"golds in pool {ent['recall']}; step 4b {'equal' if step4b else '-'} ({ent['seconds']:.0f}s)")
    return ent


def pools_stage(names, threads):
    import numba
    import step4b_pool_reach as S4B
    S4 = S4B.S4
    alpha_path = C4.STEP4B / "alpha.json"
    if float(json.loads(alpha_path.read_text(encoding="utf-8"))["alpha"]) != ALPHA:
        raise SystemExit(f"{alpha_path}: not the restart value {ALPHA}")
    op = S4.Opened()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"files": {}}
    manifest.update({"declared_in": "docs/STEP4D_UNION_POOLS.md", "alpha": ALPHA, "eps": S4.EPS, "bmax": S4.BMAX,
                     "alpha_file_sha256": sha_file(alpha_path), "carves_sha256": S4.CARVES_SHA256,
                     "frozen_contract": op.frozen_key, "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
                     "script_sha256": script_sha256(), "step4c_script_sha256": C4.script_sha256(),
                     "step4b_script_sha256": S4B.script_sha256(), "step4_script_sha256": S4.script_sha256()})
    numba.set_num_threads(threads)
    for name in names:
        t0 = time.time()
        ds = op.canonical.Dataset(name, root=str(op.served))
        pops = C4.populations(op, S4, ds, name)
        log(f"{name}: {', '.join(f'{c} {len(p.ids)}' for c, p in pops)} questions ({time.time() - t0:.0f}s)")
        construction = op.construction[name]
        regime = construction["regime"]
        families = op.m3b_compile.regime_families(op.cfg_h, regime) if regime != "RETRIEVAL" else []
        stores = {}
        for f in families:
            path = op.m3b_compile.CSR_CACHE / f"{name}_{f}.npz"
            if not path.exists():
                raise SystemExit(f"{path} is missing: the step reads M3B's stores, it does not build them")
            stores[f] = S4.m3b_pools.load_or_build_store(ds, f, op.m3b_compile.CSR_CACHE)
        preps = op.m3b_compile.prepare(ds, [p for _c, p in pops], construction, op.cfg_h, stores, op.m3a, op.m3b_contract)
        union = S4B.RegimeUnion(stores, families, int(stores[families[0]].n_nodes) if families else int(ds.n_nodes))
        for (carve, pop), prep in zip(pops, preps):
            manifest["files"][f"{name}__{carve}"] = file_carve(S4, op, name, carve, pop, prep, union, construction,
                                                               families, threads)
            manifest["utc"] = utc()
            write_json(MANIFEST, manifest)
        del preps, pops, stores, union
        gc.collect()
        log(f"{name}: filed ({time.time() - t0:.0f}s)")
    want = [f"{n}__{c}" for n in SIX for c in C4.carves_of(n)]
    log(f"manifest: {sum(k in manifest['files'] for k in want)} of {len(want)} carves filed; {MANIFEST}")
    return 0


def selftest():
    bs = np.array([1, 5])
    inc = np.array([1, 2, 5, 9])
    order = np.array([9, 7, 2, 3, 4])
    e, c, n = union_expansion(bs, inc, order, 2)
    assert e.tolist() == [2, 9, 7] and c == 2 and n == 1                    # I's own, then the walk's new node 7
    arr = filed_arrays(["q"], [bs], [inc], [order])
    up = np.union1d(bs, arr["nodes"][arr["ptr"][0]:arr["ptr"][1]])
    assert np.isin(inc, up).all() and up.tolist() == [1, 2, 5, 7, 9] and bool(arr["changed"][0])
    e2, _c, n2 = union_expansion(bs, inc, np.array([9, 2]), 2)              # the walk adds nothing: U = I
    assert e2.tolist() == [2, 9] and n2 == 0
    print("selftest: the union holds I_q, adds the walk's first B_q nodes outside it, and equals I_q when it adds none")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?")
    ap.add_argument("--datasets", default=",".join(SIX))
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "pools":
        return pools_stage([d for d in a.datasets.split(",") if d], a.threads)
    ap.error("step4d: pools (later stages are declared in docs/STEP4D_UNION_POOLS.md and added before they run)")


if __name__ == "__main__":
    sys.exit(main())
