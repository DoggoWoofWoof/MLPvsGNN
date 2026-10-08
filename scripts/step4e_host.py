"""Step 4e's host side (docs/STEP4E_PAPER_POOLS.md section 5): the chosen pools in step 1's look pipeline, its caches
and the identity gate. numba-free (the host has no numba); the pools themselves are scripts/step4e_paper_pools.py's.

    python scripts/step4e_host.py copy-bases
    python scripts/step4e_host.py gate [--screen]
    python scripts/step4e_host.py --selftest

replace_pools runs inside outputs/mp_unified/look_step4e.py, after m3b_compile.prepare: each question's pool becomes
base and seeds and its filed expansion (U_q's expansion, then the chosen arm's new nodes). The gate passes when every
look shard ran on the manifest's pools file, every cache row count is the filed pool's size, and step 1's cache rows are
the frozen pools' sizes. --screen gates the eleven carves the screen reads (the five fit carves it trains on and the six
s1eval carves) into gate-screen.json; the plain gate covers all 21 (gate.json), for the full runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import step4c_walk_pools as C4  # noqa: E402  (numba only inside its pools stage)

OUT = ROOT / "outputs" / "step4e"
POOLS = OUT / "pools"
MANIFEST = POOLS / "pools.json"
LOOK = OUT / "look"
CACHE = OUT / "cache"
CHAINS = OUT / "chains"
SIX, log, utc, sha_file, write_json = C4.SIX, C4.log, C4.utc, C4.sha_file, C4.write_json
carves_of, carve_rows, pool_sha = C4.carves_of, C4.carve_rows, C4.pool_sha
FIT_CARVE = {"metaqa": "fit", "squad": "fit", "musique": "s1fit", "hotpotqa": "fit", "2wiki": "fit"}
SCREEN = tuple(f"{d}={c}" for d, c in FIT_CARVE.items()) + tuple(f"{d}=s1eval" for d in SIX)


def script_sha256():
    return C4.hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_filed(name, carve):
    """One carve's chosen pools file, its sha256 checked against the manifest; (arrays, entry, manifest sha256)."""
    return C4.load_filed(name, carve, POOLS)


def expected_rows(filed):
    """Each question's pool size: base and seeds plus the expansion (disjoint from them by construction)."""
    return filed["bs_size"].astype(np.int64) + np.diff(filed["ptr"]).astype(np.int64)


def replace_pools(prep, filed, m3c, construction, cfg_h, m3a, m3b_contract):
    """Each prepared question's pool becomes base and seeds and its filed expansion (m3c.build_pool), in place. Stops
    on a missing question, base and seeds or a frozen pool that is not the filed one, an expansion that repeats a node,
    holds a node of base and seeds or is shorter than its new nodes, or a pool that is not base and seeds plus the
    expansion or loses a node of the frozen pool."""
    constant = int(cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    bases = m3c.base_rows(construction, prep.dense_ids, prep.splade_ids, m3a, m3b_contract, constant)
    at = {str(q): k for k, q in enumerate(filed["ids"].tolist())}
    rows = new = 0
    for j, q in enumerate(prep.pop.ids):
        k = at.get(str(q))
        if k is None:
            raise SystemExit(f"{q}: not in the pools file")
        base = np.asarray(bases[j], dtype=np.int64)
        seeds = np.asarray(prep.seeds[j], dtype=np.int64)
        bs = np.union1d(base, seeds)
        if bs.size != int(filed["bs_size"][k]) or not np.array_equal(
                bs, filed["bs_nodes"][filed["bs_ptr"][k]:filed["bs_ptr"][k + 1]].astype(np.int64)):
            raise SystemExit(f"{q}: base and seeds are not the filed ones")
        inc = np.asarray(prep.pools[j], dtype=np.int64)
        sha = filed["inc_sha"][k]
        sha = sha.decode() if isinstance(sha, bytes) else str(sha)
        if inc.size != int(filed["inc_size"][k]) or pool_sha(inc) != sha:
            raise SystemExit(f"{q}: the frozen pool is not the filed one")
        exp = filed["nodes"][filed["ptr"][k]:filed["ptr"][k + 1]].astype(np.int64)
        if exp.size < int(filed["count"][k]):
            raise SystemExit(f"{q}: the expansion is shorter than its new nodes")
        if np.unique(exp).size != exp.size or np.isin(exp, bs).any():
            raise SystemExit(f"{q}: the expansion repeats a node or holds a node of base and seeds")
        pool, added = m3c.build_pool(base, seeds, exp)
        if pool.size != bs.size + exp.size or not np.isin(inc, pool).all():
            raise SystemExit(f"{q}: the pool has {pool.size} nodes, not {bs.size + exp.size}, or loses a frozen node")
        rows += int(pool.size)
        new += int(filed["count"][k])
        prep.pools[j] = pool
        prep.seeds_added[j] = added
    return {"questions": len(prep.pop.ids), "rows": rows, "new": new}


# ── host stages ──────────────────────────────────────────────────────────────


def copy_bases():
    """Step 1's two basis files (2wiki, hotpotqa) into the step's cache root, sha-checked (step 4c's copy-bases)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    rec = {"source": str(C4.STEP1_CACHE.relative_to(ROOT)).replace("\\", "/"), "files": {}, "script_sha256": script_sha256()}
    for g in C4.BASES:
        npz, js = C4.STEP1_CACHE / f"basis_{g}.npz", C4.STEP1_CACHE / f"basis_{g}.json"
        brec = json.loads(js.read_text(encoding="utf-8"))
        if sha_file(npz) != brec["npz_sha256"]:
            raise SystemExit(f"{npz}: sha256 is not its record's")
        for src in (npz, js):
            dst = CACHE / src.name
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            a, b = sha_file(src), sha_file(dst)
            if a != b:
                raise SystemExit(f"{dst}: the copy is not its source")
            rec["files"][src.name] = a
    rec["utc"] = utc()
    write_json(CACHE / "bases_copied.json", rec)
    log(f"bases copied: {rec['files']}")
    return 0


def look_pools_records(ds, carve, ent, want_rows):
    """Every look shard of the carve filed a pools record naming the manifest's file and its row total; problems."""
    d = LOOK / ds / carve
    looks = sorted(p.name[len("record"):-len(".json")] for p in d.glob("record*.json"))
    pools = sorted(p.name[len("pools"):-len(".json")] for p in d.glob("pools*.json"))
    bad = []
    if not looks or looks != pools:
        bad.append(f"look records {looks} and pools records {pools} differ")
    for tag in pools:
        r = json.loads((d / f"pools{tag}.json").read_text(encoding="utf-8"))
        if r.get("pools_sha256") != ent["sha256"]:
            bad.append(f"pools{tag}.json names another pools file")
        if r.get("rows") != want_rows or r.get("questions") != ent["questions"]:
            bad.append(f"pools{tag}.json holds {r.get('questions')} questions and {r.get('rows')} rows, the file "
                       f"{ent['questions']} and {want_rows}")
    return bad


def gate(screen=False):
    t0 = time.time()
    manifest_sha = sha_file(MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rec = {"declared_in": "docs/STEP4E_PAPER_POOLS.md", "arm": manifest["arm"], "k": manifest["k"],
           "manifest_sha256": manifest_sha, "scope": "screen" if screen else "all", "rows": {}, "looks": {},
           "script_sha256": script_sha256()}
    fails = []
    for ds in SIX:
        for cv in carves_of(ds):
            if screen and f"{ds}={cv}" not in SCREEN:
                continue
            filed, ent, msha = load_filed(ds, cv)
            if msha != manifest_sha:
                raise SystemExit("the manifest changed while the gate ran")
            want = expected_rows(filed)
            bad = look_pools_records(ds, cv, ent, int(want.sum()))
            rec["looks"][f"{ds}={cv}"] = bad or "PASS"
            fails += [f"{ds}/{cv}: {b}" for b in bad]
            ids1, n1, _p1, _r1 = carve_rows(C4.STEP1_CACHE, ds, cv)
            ids4, n4, _p4, _r4 = carve_rows(CACHE, ds, cv)
            at = {str(q): k for k, q in enumerate(filed["ids"].tolist())}
            row = {"questions": len(ids4), "pool_mean": ent["pool_mean"], "I_mean": ent["I_mean"]}
            if ids1 != ids4 or any(q not in at for q in ids4):
                row["verdict"] = "FAIL: the cache's questions are not step 1's or not the pools file's"
            else:
                k = np.asarray([at[q] for q in ids4], dtype=np.int64)
                ok1 = np.array_equal(n1, filed["inc_size"][k].astype(np.int64))
                ok4 = np.array_equal(n4, want[k])
                row["step1_rows_are_frozen_sizes"] = ok1
                row["rows_are_chosen_sizes"] = ok4
                row["verdict"] = "PASS" if ok1 and ok4 else "FAIL"
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
    class Pop:
        ids = ["q0", "q1"]

    class Prep:
        pop = Pop()
        dense_ids = splade_ids = None
        seeds = [np.array([3]), np.array([7])]
        pools = [np.array([1, 2, 3, 5]), np.array([6, 7, 8])]
        seeds_added = [0, 0]

    class M3C:
        @staticmethod
        def base_rows(*_a):
            return [np.array([1, 2]), np.array([6])]

        @staticmethod
        def build_pool(base, seeds, exp):
            measured = np.union1d(base, exp)
            return np.union1d(measured, seeds), int(np.isin(seeds, measured, invert=True).sum())

    cfg = {"retrieval_pools": {"equal_rrf": {"constant": 60}}}
    bs = [np.array([1, 2, 3]), np.array([6, 7])]
    exps = [np.array([5, 9, 4]), np.array([8, 11])]           # U_q's expansion first, then the arm's new nodes
    ptr, flat = C4.flat_of(exps)
    bptr, bflat = C4.flat_of(bs)
    filed = {"ids": np.array(["q0", "q1"]), "inc_size": np.array([4, 3]), "bs_size": np.array([3, 2]),
             "count": np.array([2, 1]), "ptr": ptr, "nodes": flat.astype(np.int32), "bs_ptr": bptr,
             "bs_nodes": bflat.astype(np.int32),
             "inc_sha": np.array([pool_sha(p) for p in Prep.pools], dtype="S64")}
    assert expected_rows(filed).tolist() == [6, 4]
    assert len(SCREEN) == 11 and all(c in carves_of(d) for d, c in (x.split("=") for x in SCREEN))
    p = Prep()
    p.pools = [x.copy() for x in Prep.pools]
    st = replace_pools(p, filed, M3C, None, cfg, None, None)
    assert st == {"questions": 2, "rows": 10, "new": 3}, st
    assert p.pools[0].tolist() == [1, 2, 3, 4, 5, 9] and p.pools[1].tolist() == [6, 7, 8, 11]
    for bad in ("dup", "bs", "lose"):
        f2 = dict(filed)
        e = [x.copy() for x in exps]
        if bad == "dup":
            e[0] = np.array([5, 5, 4])
        elif bad == "bs":
            e[0] = np.array([5, 3, 4])
        else:
            e[0] = np.array([9, 4, 10])                        # drops frozen node 5
        f2["ptr"], fl = C4.flat_of(e)
        f2["nodes"] = fl.astype(np.int32)
        p = Prep()
        p.pools = [x.copy() for x in Prep.pools]
        try:
            replace_pools(p, f2, M3C, None, cfg, None, None)
        except SystemExit:
            continue
        raise AssertionError(f"replace_pools took a bad expansion ({bad})")
    print("selftest: replace_pools builds base and seeds plus the expansion and stops on a repeated node, a node of "
          "base and seeds and a lost frozen node; expected_rows")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("copy-bases", "gate"))
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "copy-bases":
        return copy_bases()
    if a.stage == "gate":
        return gate(a.screen)
    ap.error("a stage: copy-bases or gate")


if __name__ == "__main__":
    sys.exit(main())
