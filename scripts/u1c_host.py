"""U1c's host side (docs/U1C_RETRAIN_ON_U.md): step 4h's chosen pools on U1d's graphs, filed in step 4e's format for
outputs/mp_unified/look_u1c.py, the bases copied into U1c's cache root, and the identity gate.

    python outputs/host_ops/pylib_run.py scripts/u1c_host.py pools --dataset D [--carves screen|rest] --threads 5 --host
    python scripts/u1c_host.py copy-bases
    python scripts/u1c_host.py gate [--screen]
    python scripts/u1c_host.py --selftest

pools: step 4h's coverage stage on GU only (scripts/step4h_u_pools.py, G = GU: `structural` = U1d's structural_U; squad,
with no graph, keeps step 4e's RRF pools), for every carve the look pipeline reads (step 4c's carves_of); each question's
pool is step 4h's chosen configuration (outputs/step4h/choice.json): W(k) = U^G plus the A3 walk's first k x B_q nodes
outside U^G, or B(k) = step4h_u_pools.bridge_pool. Filed per dataset (outputs/u1c/pools/<D>/pools.json and
<D>__<carve>.npz), so the six datasets' jobs never share a manifest. Each file's ALL and pool size on s1sel and s1eval
must equal step 4h's coverage record for the chosen configuration (the same code on the same graph).
gate: every look shard ran on its dataset's filed pools, and every cache's row counts are the filed pools' sizes (base
and seeds plus the expansion). Step 1's frozen sizes are not checked: on GU the frozen construction itself differs.
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
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src", ROOT / "outputs" / "step4f"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import step4h_u_pools as H  # noqa: E402
import step4e_host as S4E  # noqa: E402

E, C4, S4 = H.E, H.C4, H.S4
SIX, TRAIN5, log, utc, write_json, sha_file = H.SIX, H.TRAIN5, H.log, H.utc, H.write_json, C4.sha_file
OUT = ROOT / "outputs" / "u1c"
POOLS = OUT / "pools"
LOOK = OUT / "look"
CACHE = OUT / "cache"
CHOICE = H.OUT / "choice.json"
FIT_CARVE = S4E.FIT_CARVE
SCREEN = {(d, c) for d, c in FIT_CARVE.items()} | {(d, "s1eval") for d in SIX}


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def carves_for(name, which):
    allc = list(C4.carves_of(name))
    scr = [c for c in allc if (name, c) in SCREEN]
    if which == "screen":
        return scr
    if which == "rest":
        return [c for c in allc if c not in scr]
    return allc


def load_filed(name, carve):
    return C4.load_filed(name, carve, POOLS / name)


def chosen_config():
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))
    c = ch["chosen"]
    if c[0] not in "WB":
        raise SystemExit(f"{CHOICE}: chosen {c!r} is not a W(k) or B(k) configuration")
    return c, sha_file(CHOICE)


def pool_of(cfg, U, order, seeds, BQ, fams):
    """One question's chosen pool on G (step 4h's definitions)."""
    k = float(cfg[1:])
    if cfg[0] == "W" or not fams:                              # squad: B(k) is W(k) (step4h_u_pools.choose)
        return np.union1d(U, order[:int(round(k * BQ))])
    return H.bridge_pool(U, order, seeds, int(BQ), int(k), fams)


def pools_stage(name, which, threads, host):
    import numba
    import step4b_pool_reach as S4B
    numba.set_num_threads(threads)
    cfg, choice_sha = chosen_config()
    cov = json.loads((H.OUT / f"coverage_{name}.json").read_text(encoding="utf-8"))
    op = S4.Opened()
    ds, construction, families, stores, union0 = E.open_dataset(op, name)
    n = int(ds.n_nodes)
    nograph = name == "squad" or not families
    rec_g = {"note": "no graph family is read: step 4e's RRF pools"}
    if nograph:
        st_, un = stores, union0
    else:
        st, rec_g = H.u_store(name, n)
        if cov.get("GU", {}).get("npz_sha256") != rec_g["npz_sha256"]:
            raise SystemExit(f"{name}: step 4h's coverage ran on another structural_U than U1d's build")
        st_ = dict(stores)
        st_["structural"] = st
        un = S4B.RegimeUnion(st_, families, n)
    wanted = carves_for(name, which)
    pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c in wanted]
    index = texts = None
    if not nograph:
        index = E.build_index(E.node_names(ds), E.stopwords())
        texts = E.query_texts(ds)
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    preps0 = op.m3b_compile.prepare(ds, [p for _c, p in pops], construction, op.cfg_h, stores, op.m3a,
                                    op.m3b_contract)                 # G0: step 4d's U_q and B_q are checked on it
    preps = preps0 if nograph else op.m3b_compile.prepare(ds, [p for _c, p in pops], construction, op.cfg_h, st_,
                                                          op.m3a, op.m3b_contract)
    d = POOLS / name
    d.mkdir(parents=True, exist_ok=True)
    mpath = d / "pools.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"files": {}}
    manifest.update({"declared_in": "docs/U1C_RETRAIN_ON_U.md", "dataset": name, "config": cfg,
                     "choice_sha256": choice_sha, "graph": rec_g, "script_sha256": script_sha256(),
                     "step4h_sha256": H.script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"]})
    fams = [] if nograph else [f for f in un.fams if f.indptr[-1] > 0]
    for ci, (carve, pop) in enumerate(pops):
        t = time.time()
        prep, prep0 = preps[ci], preps0[ci]
        bases = op.m3b_compile.base_rows(construction, prep0.dense_ids, prep0.splade_ids, op.m3a, op.m3b_contract,
                                         constant)
        U0, BQ, bs, inc0, _f = E.union_pools(name, carve, pop, prep0, bases)
        del bases
        seeds = [np.asarray(s, np.int64) for s in prep.seeds]
        if any(not np.array_equal(a, np.asarray(b, np.int64)) for a, b in zip(seeds, prep0.seeds)):
            raise SystemExit(f"{name}/{carve}: GU's seeds differ from G0's")
        inc = [np.asarray(p, np.int64) for p in prep.pools]
        nq = len(bs)
        if nograph:
            U = U0
        else:
            porder = []
            for lo in range(0, nq, S4.BATCH):
                hi = min(nq, lo + S4.BATCH)
                porder += un.ppr(seeds[lo:hi], bs[lo:hi], C4.ALPHA, S4.EPS, S4.BMAX, threads)[0]
            U = [np.union1d(inc[j], np.asarray(porder[j][:int(BQ[j])], np.int64)) for j in range(nq)]
            del porder
        linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [np.empty(0, np.int64)] * nq
        orders = E.arm_orders(H.ARM, name, un, prep, U, BQ, linked, constant, threads)
        pools = [pool_of(cfg, U[j], np.asarray(orders[j], np.int64), seeds[j], BQ[j], fams) for j in range(nq)]
        del orders
        exps, new = [], []
        for j in range(nq):
            if not np.isin(U[j], pools[j]).all() or not np.isin(inc[j], pools[j]).all():
                raise SystemExit(f"{pop.ids[j]}: the chosen pool loses a node of U_q or I_q")
            exps.append(np.setdiff1d(pools[j], bs[j]))
            new.append(int(np.setdiff1d(pools[j], U[j]).size))
        ptr, flat = C4.flat_of(exps)
        bs_ptr, bs_flat = C4.flat_of(bs)
        arrays = {"ids": np.asarray(list(pop.ids)), "inc_size": np.asarray([p.size for p in inc], dtype=np.int32),
                  "bs_size": np.asarray([b.size for b in bs], dtype=np.int32), "B_q": BQ.astype(np.int32),
                  "count": np.asarray(new, dtype=np.int32), "new": np.asarray(new, dtype=np.int32), "ptr": ptr,
                  "nodes": flat.astype(np.int32), "bs_ptr": bs_ptr, "bs_nodes": bs_flat.astype(np.int32),
                  "inc_sha": np.asarray([C4.pool_sha(p) for p in inc], dtype="S64"),
                  "changed": np.ones(nq, dtype=bool)}
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
        tl = H.tally(pools, golds)
        if carve in cov["carves"]:
            g = cov["carves"][carve]["graphs"]
            want = (g.get("GU") or g["G0"])
            want = want.get(cfg) or want[cfg.replace("B", "W")]
            if (tl["ALL"], tl["pool"]) != (want["ALL"], want["pool"]):
                raise SystemExit(f"{name}/{carve}: the filed pools ({tl}) are not step 4h's {cfg} ({want})")
        fname = f"{name}__{carve}.npz"
        tmp = d / f"{name}__{carve}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, d / fname)
        manifest["files"][f"{name}__{carve}"] = {
            "file": fname, "sha256": sha_file(d / fname), "dataset": name, "carve": carve, "questions": nq,
            "carve_ids_sha256": pop.digest, "construction": construction, "walk_families": list(families),
            "pool_mean": tl["pool"], "pool_max": int(max(p.size for p in pools)) if pools else 0,
            "U_mean": round(float(np.mean([u.size for u in U])), 1),
            "I_mean": round(float(np.mean([p.size for p in inc])), 1), "ALL": tl["ALL"], "recall": tl["recall"],
            "matches_step4h": carve in cov["carves"], "seconds": round(time.time() - t, 1), "utc": utc()}
        manifest["utc"] = utc()
        write_json(mpath, manifest)
        log(f"{name}/{carve}: {cfg} pool {tl['pool']}, ALL {tl['ALL']} ({time.time() - t:.0f}s)")
        del pools, exps, U, inc
        gc.collect()
    return 0


# ── copy-bases and the gate ──────────────────────────────────────────────────


def copy_bases():
    saved = S4E.CACHE
    S4E.CACHE = CACHE
    try:
        return S4E.copy_bases()
    finally:
        S4E.CACHE = saved


def look_pools_records(ds, carve, ent, want_rows):
    saved = S4E.LOOK
    S4E.LOOK = LOOK
    try:
        return S4E.look_pools_records(ds, carve, ent, want_rows)
    finally:
        S4E.LOOK = saved


def gate(screen=False):
    t0 = time.time()
    rec = {"declared_in": "docs/U1C_RETRAIN_ON_U.md", "scope": "screen" if screen else "all", "rows": {}, "looks": {},
           "manifests": {}, "script_sha256": script_sha256()}
    fails = []
    for ds in SIX:
        rec["manifests"][ds] = sha_file(POOLS / ds / "pools.json")
        for cv in C4.carves_of(ds):
            if screen and (ds, cv) not in SCREEN:
                continue
            filed, ent, msha = load_filed(ds, cv)
            if msha != rec["manifests"][ds]:
                raise SystemExit(f"{ds}: the manifest changed while the gate ran")
            want = S4E.expected_rows(filed)
            bad = look_pools_records(ds, cv, ent, int(want.sum()))
            rec["looks"][f"{ds}={cv}"] = bad or "PASS"
            fails += [f"{ds}/{cv}: {b}" for b in bad]
            ids4, n4, _p4, _r4 = C4.carve_rows(CACHE, ds, cv)
            at = {str(q): k for k, q in enumerate(filed["ids"].tolist())}
            row = {"questions": len(ids4), "pool_mean": ent["pool_mean"], "I_mean": ent["I_mean"]}
            if any(q not in at for q in ids4) or len(ids4) != len(at):
                row["verdict"] = "FAIL: the cache's questions are not the pools file's"
            else:
                k = np.asarray([at[q] for q in ids4], dtype=np.int64)
                row["rows_are_chosen_sizes"] = bool(np.array_equal(n4, want[k]))
                row["verdict"] = "PASS" if row["rows_are_chosen_sizes"] else "FAIL"
            rec["rows"][f"{ds}={cv}"] = row
            if row["verdict"] != "PASS":
                fails.append(f"{ds}/{cv}: rows {row['verdict']}")
    rec["failures"] = fails
    rec["verdict"] = "PASS" if not fails else "FAIL"
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = utc()
    write_json(OUT / ("gate-screen.json" if screen else "gate.json"), rec)
    log(f"gate ({rec['scope']}) {rec['verdict']}: {fails or 'every row count as filed, every look on the filed pools'}")
    return 0 if not fails else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    fams = [SimpleNamespace(indptr=np.array([0, 2, 3, 5, 6, 6]), col=np.array([1, 2, 0, 0, 4, 2]))]
    U, order, seeds = np.array([0]), np.array([3, 1, 2, 4]), np.array([0])
    assert pool_of("W1", U, order, seeds, 2, fams).tolist() == [0, 1, 3]
    assert pool_of("W0.5", U, order, seeds, 2, fams).tolist() == [0, 3]
    assert pool_of("B1", U, order, seeds, 2, fams).tolist() == H.bridge_pool(U, order, seeds, 2, 1, fams).tolist()
    assert pool_of("B2", U, order, seeds, 2, []).tolist() == pool_of("W2", U, order, seeds, 2, []).tolist()
    assert carves_for("webqsp", "screen") == ["s1eval"] and carves_for("webqsp", "rest") == []
    assert carves_for("musique", "screen") == ["s1fit", "s1eval"]
    assert carves_for("2wiki", "rest") == ["select", "s1sel"]
    assert len(SCREEN) == 11
    print("u1c_host selftest ok")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("pools", "copy-bases", "gate"))
    ap.add_argument("--dataset")
    ap.add_argument("--carves", default="screen", choices=("screen", "rest", "all"))
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args()
    if a.stage == "copy-bases":
        return copy_bases()
    if a.stage == "gate":
        return gate(a.screen)
    if a.dataset not in SIX:
        ap.error("pools: --dataset, one of the six")
    if a.host:
        import run_one
        run_one.on_the_mirror(SimpleNamespace(E=E))
    return pools_stage(a.dataset, a.carves, a.threads, a.host)


if __name__ == "__main__":
    sys.exit(main())
