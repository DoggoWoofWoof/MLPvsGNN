"""P1c2's pools and host stages (docs/P1C2_POOLS_KIND_T.md): P1c (docs/P1C_POOLS_RETRAINED.md, held and never read)
re-declared on the final pool, P1e's KIND_T (docs/P1E_ANCHOR_KIND.md, section 8). The models' pools are rebuilt from
KIND_T's order at each question's own frozen size, in step 4c's file format, for outputs/mp_unified/look_p1c2.py; the
bases are copied into P1c2's cache root; and P1c's identity gate runs on P1c2's roots. Nothing is fit: KIND_T reads no
gold.

    python outputs/host_ops/pylib_run.py scripts/p1c2_pools.py pools DATASET [--threads 5] [--limit N] --host
    python scripts/p1c2_pools.py copy-bases
    python scripts/p1c2_pools.py gate
    python scripts/p1c2_pools.py --selftest

pools       The screen's carves of one dataset (its fit carve, if it trains, and s1eval). Per question, P1d's sources,
            P1e's steered walks, ROUTE's condition and the question's kind (scripts/p1e_anchor_kind.py, imported
            unchanged), and KIND_T's order: ROUTEst for an entity question, RRF5T for a passage question, cut at 7,000.
            The pool is step 4c's: base and seeds, then the order's first B_q nodes outside them,
            B_q = |I_q| - |base and seeds| (m3b_compile.build_pool), so every pool keeps its frozen size; a question
            whose order offers fewer keeps them all and is short. Checks, any failure stops the stage:
              - P1e's verdict chose KIND_T (outputs/p1e/verdict.json), the rule this file declares;
              - P1e's part for the dataset was written by P1e's filed script, on this freeze, with no --limit, and its
                name-only count is the one found here;
              - step 1's host cache rows are the frozen pool sizes built here (step 4c's step1_rows);
              - on s1eval, KIND_T puts every gold at the position P1e filed (outputs/p1e/pos_<D>__s1eval.npz), question
                for question, and its ALL and recall at |I_q| equal P1e's filed record.
            Written to outputs/p1c2/pools/<D>/<D>__<carve>.npz (step 4c's arrays, and per question whether it is an
            entity question and whether it takes the route) and outputs/p1c2/pools/<D>/pools.json (each file's sha256
            and counts, the coverage of I_q and of P1c2's pools).
            --limit N: a development run on the first N questions of each carve. It writes nothing and logs no number,
            only that every step ran and that the checks held (on s1eval, the positions of those questions' golds).
copy-bases  step 1's cache bases (basis_2wiki, basis_hotpotqa) into outputs/p1c2/cache (step 4c's copy_bases).
gate        P1c's gate on P1c2's roots: squad's carves: every cache array equals step 1's (its pools cannot change:
            |I_q| is |base and seeds|), score2 within 1e-4; every carve: each question's rows equal its filed pool size,
            and step 1's rows its |I_q|; every look shard filed a pools record naming the file. Writes
            outputs/p1c2/gate.json; exits 1 on a failure.
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import p1c_pools as C1  # noqa: E402  (P1c's helpers; numba-free at import)

C4S = C1.C4S
DECLARED = "docs/P1C2_POOLS_KIND_T.md"
OUT = ROOT / "outputs" / "p1c2"
POOLS = OUT / "pools"
LOOK = OUT / "look"
CACHE = OUT / "cache"
GATE = OUT / "gate.json"
P1E = ROOT / "outputs" / "p1e"
SIX, TRAIN5, FIT_CARVE, SCREEN = C1.SIX, C1.TRAIN5, C1.FIT_CARVE, C1.SCREEN
CONSTRUCTION = "KIND_T"    # declared in docs/P1C2_POOLS_KIND_T.md, after P1e's verdict and before any number of P1c2
log, utc, write_json, sha_file = C4S.log, C4S.utc, C4S.write_json, C4S.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def carves_for(name):
    return [c for d, c in SCREEN if d == name]


def picked(verdict_path=P1E / "verdict.json"):
    """The rule P1e's verdict chose (section 8 of docs/P1E_ANCHOR_KIND.md)."""
    v = json.loads(Path(verdict_path).read_text(encoding="utf-8"))
    c = v.get("chosen")
    if not c:
        raise SystemExit(f"{verdict_path}: no chosen rule")
    return c, sha_file(verdict_path)


def filed_positions(name, carve, part, path=None):
    """P1e's filed gold positions under KIND_T on one carve: (ptr, positions), after the file's sha256 is checked
    against the part that names it."""
    path = Path(path) if path else P1E / f"pos_{name}__{carve}.npz"
    want = part["carves"][f"{name}__{carve}"]["positions_npz_sha256"]
    if sha_file(path) != want:
        raise SystemExit(f"{path}: not the file P1e's part names")
    with np.load(path) as z:
        return z["ptr"].astype(np.int64), z[CONSTRUCTION].astype(np.int64)


def same_positions(pos, ptr, filed, nq):
    """The first nq questions' gold positions equal the filed ones, gold for gold."""
    got = np.concatenate(pos).astype(np.int64) if pos else np.empty(0, np.int64)
    want = filed[:ptr[nq]]
    return bool(got.size == want.size and np.array_equal(got, want))


def kind_t(K, src, steered, route, knd, n_st, constant):
    """KIND_T's order for one question, by P1e's p1e_orders, unchanged."""
    return np.asarray(K.p1e_orders(src, steered, route, knd, n_st, constant)[0][CONSTRUCTION], np.int64)


# ── pools ────────────────────────────────────────────────────────────────────


def pools_stage(name, threads, limit=0):
    import p1e_anchor_kind as K         # P1e's walks, kind and orders, through it P1d's sources (imports numba)
    D, G, P, E, S4 = K.D, K.G, K.P, K.E, K.S4
    pick, verdict_sha = picked()
    if pick != CONSTRUCTION:
        raise SystemExit(f"P1e's verdict chose {pick}; this script declares {CONSTRUCTION}")
    op = S4.Opened()
    part = json.loads(K.part_path(name).read_text(encoding="utf-8"))
    if part.get("script_sha256") != K.script_sha256() or part.get("limit"):
        raise SystemExit(f"{K.part_path(name)}: not written by P1e's filed script on every question")
    if part.get("freeze_RECORD_SHA256") != op.freeze["RECORD_SHA256"]:
        raise SystemExit(f"{K.part_path(name)}: written on another freeze")
    ds = op.canonical.Dataset(name, root=str(op.served))
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    del ds
    if int(nonly.sum()) != int(part["name_only_nodes"]) or int(nonly.size) != int(part["nodes"]):
        raise SystemExit(f"{name}: {int(nonly.sum())} name-only nodes of {nonly.size}, P1e's part "
                         f"{part['name_only_nodes']} of {part['nodes']}")
    d = POOLS / name
    mpath = d / "pools.json"
    manifest = {"files": {}}
    if not limit and mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    manifest.update({"declared_in": DECLARED, "dataset": name, "construction": CONSTRUCTION,
                     "p1e_verdict_sha256": verdict_sha, "p1e_part_sha256": sha_file(K.part_path(name)),
                     "p1e_script_sha256": K.script_sha256(), "p1d_script_sha256": D.script_sha256(),
                     "p1c_script_sha256": C1.script_sha256(), "step4c_script_sha256": C4S.script_sha256(),
                     "script_sha256": script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
                     "name_only_nodes": int(nonly.sum()), "nodes": int(nonly.size)})
    cache = {}

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        _stores, unions = G.open_unions(cache, op, name, union)
        n_st = int(unions["st"].deg.size)
        nq = min(limit, len(U)) if limit else len(U)
        ids = list(pop.ids)[:nq]
        bs, inc = bs[:nq], inc[:nq]
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds[:nq]]
        inc_size = np.asarray([np.asarray(p).size for p in inc], dtype=np.int32)
        host_rows = None if limit else C4S.step1_rows(name, carve, ids, inc_size)
        orders, pos = [], []
        entity = np.zeros(nq, dtype=bool)
        route = np.zeros(nq, dtype=bool)
        for lo in range(0, nq, S4.BATCH):
            js = list(range(lo, min(nq, lo + S4.BATCH)))
            srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
            steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
            for i, j in enumerate(js):
                src, _nn, (L, Ls) = srcs[i]
                route[j] = D.routes(L, Ls)
                knd = K.kind(Ls, prep.seeds[j], nonly)
                entity[j] = knd == "entity"
                o = kind_t(K, src, steered[i], bool(route[j]), knd, n_st, constant)
                pos.append(P.positions(o, golds[j]))
                # the file keeps the first B_q nodes outside base and seeds (step 4c's expansions); hold no more
                orders.append(C1.outside(o, bs[j])[:max(0, int(inc_size[j]) - int(np.asarray(bs[j]).size))])
        checks = {}
        if carve == "s1eval":
            ent_e = part["carves"].get(f"{name}__{carve}")
            if ent_e is None or ent_e["carve_sha256"] != pop.digest or int(ent_e["questions"]) != len(U):
                raise SystemExit(f"{name}/{carve}: P1e's part holds other questions")
            ptr, filed = filed_positions(name, carve, part)
            checks["positions_equal_p1e"] = same_positions(pos, ptr, filed, nq)
            checks["at_I_q_equals_p1e"] = True if limit else P.at_sizes(pos, inc_size) == ent_e["orders"][CONSTRUCTION]["at_I_q"]
        if not all(checks.values()):
            raise SystemExit(f"{name}/{carve}: {checks}")
        arrays = C4S.filed_arrays(ids, bs, inc, orders)
        arrays["entity"] = entity
        arrays["route"] = route
        ptr_, flat, count, b_q = arrays["ptr"], arrays["nodes"], arrays["count"], arrays["B_q"]
        pools = [np.union1d(np.asarray(bs[j], np.int64), flat[ptr_[j]:ptr_[j + 1]].astype(np.int64)) for j in range(nq)]
        expect = C4S.expected_rows(arrays)
        if any(p.size != int(e) for p, e in zip(pools, expect)):
            raise SystemExit(f"{name}/{carve}: a pool is not its expected size")
        short = int((count < b_q).sum())
        if limit:
            log(f"{name}/{carve} development run: {nq} questions, the sources, steered walks, kinds, KIND_T's order, "
                f"checks {sorted(checks)} and the pools ran ({time.time() - t:.0f}s); nothing written")
            return
        d.mkdir(parents=True, exist_ok=True)
        fname = f"{name}__{carve}.npz"
        tmp = d / f"{name}__{carve}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, d / fname)
        cov_i, cov_p = C1.coverage(inc, golds), C1.coverage(pools, golds)
        manifest["files"][f"{name}__{carve}"] = {
            "file": fname, "sha256": sha_file(d / fname), "dataset": name, "carve": carve, "questions": nq,
            "carve_ids_sha256": pop.digest, "construction": CONSTRUCTION, "walk_families": list(families),
            "changed": int(arrays["changed"].sum()), "short": short,
            "B_q": {"mean": round(float(b_q.mean()), 1) if nq else None, "max": int(b_q.max()) if nq else None},
            "pool_mean": round(float(inc_size.mean()), 1) if nq else None,
            "shares": {"entity": round(float(entity.mean()), 4) if nq else None,
                       "route": round(float(route.mean()), 4) if nq else None,
                       "entity_route": round(float((entity & route).mean()), 4) if nq else None},
            "ALL": {"I": cov_i["ALL"], "P1c2": cov_p["ALL"]}, "recall": {"I": cov_i["recall"], "P1c2": cov_p["recall"]},
            "order_at_I_q": P.at_sizes(pos, inc_size), "checks": checks, "step1_cache_rows": host_rows,
            "seconds": round(time.time() - t, 1), "utc": utc()}
        manifest["utc"] = utc()
        write_json(mpath, manifest)
        e = manifest["files"][f"{name}__{carve}"]
        log(f"{name}/{carve}: {nq} questions, {e['changed']} pools changed, {short} short; ALL I {e['ALL']['I']} "
            f"P1c2 {e['ALL']['P1c2']} ({e['seconds']:.0f}s)")
        del pools, orders, pos
        gc.collect()

    E.run_carves(op, name, set(carves_for(name)), ("A3",), threads, sink)
    return 0


# ── copy-bases and the gate ──────────────────────────────────────────────────


def copy_bases():
    saved = C4S.CACHE
    C4S.CACHE = CACHE
    try:
        return C4S.copy_bases()
    finally:
        C4S.CACHE = saved


def gate():
    t0 = time.time()
    rec = {"declared_in": DECLARED, "construction": CONSTRUCTION, "manifests": {}, "rows": {}, "looks": {},
           "squad": {}, "script_sha256": script_sha256()}
    fails = []
    for ds, cv in SCREEN:
        mp = POOLS / ds / "pools.json"
        rec["manifests"].setdefault(ds, sha_file(mp))
        filed, ent, msha = C4S.load_filed(ds, cv, POOLS / ds)
        if msha != rec["manifests"][ds]:
            raise SystemExit(f"{mp}: changed while the gate ran")
        if ent.get("construction") != CONSTRUCTION:
            fails.append(f"{ds}/{cv}: the pools file is {ent.get('construction')}'s, not {CONSTRUCTION}'s")
        bad = C1.look_pools_records(ds, cv, ent, LOOK)
        rec["looks"][f"{ds}={cv}"] = bad or "PASS"
        fails += [f"{ds}/{cv}: {b}" for b in bad]
        ids1, n1, p1, r1 = C4S.carve_rows(C4S.STEP1_CACHE, ds, cv)
        ids4, n4, p4, r4 = C4S.carve_rows(CACHE, ds, cv)
        row = {"questions": len(ids4), "short": ent["short"], "changed": ent["changed"],
               **C1.rows_check(filed, ids1, n1, ids4, n4)}
        rec["rows"][f"{ds}={cv}"] = row
        if row["verdict"] != "PASS":
            fails.append(f"{ds}/{cv}: rows {row['verdict']}")
        if ds == "squad":
            sq = C1.squad_identity(p1, p4, r1, r4)
            rec["squad"][cv] = sq
            if sq["verdict"] != "PASS":
                fails.append(f"squad/{cv}: identity {sq['verdict']} {sq['arrays']} score2 {sq['score2_max_abs_diff']}")
    rec["failures"] = fails
    rec["verdict"] = "PASS" if not fails else "FAIL"
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = utc()
    write_json(GATE, rec)
    log(f"gate {rec['verdict']}: {fails or 'squad identical, every row count as filed, every look on the filed pools'}")
    return 0 if not fails else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    from types import SimpleNamespace
    t0 = time.time()
    C1.selftest()                       # P1c's helpers: outside, step 4c's file and replacement, coverage, the gate's pieces
    import p1e_anchor_kind as K
    B = K.B
    assert len(SCREEN) == 11 and carves_for("musique") == ["s1fit", "s1eval"] and carves_for("webqsp") == ["s1eval"]
    assert CONSTRUCTION in K.RULES and (POOLS, LOOK, CACHE) != (C1.POOLS, C1.LOOK, C1.CACHE)
    # KIND_T by P1e's orders: ROUTEst for an entity question on the route, A1(st) off it, RRF5T for a passage question
    a1 = B.by_mass(np.asarray([5, 6, 7, 8]), np.asarray([0.4, 0.3, 0.2, 0.1]))
    kst = B.by_mass(np.asarray([1, 2, 3]), np.asarray([0.7, 0.2, 0.1]))
    src = (a1, a1, kst, B.EMPTY, B.by_mass(np.asarray([9, 5]), np.asarray([0.6, 0.4])))
    st = {"KTst": np.asarray([1, 3, 2]), "KTall": np.asarray([1, 2]), "A1Tst": np.asarray([6, 5, 8]),
          "A1Tall": np.asarray([5])}
    assert kind_t(K, src, st, True, "entity", 4, 60).tolist() == [1, 2, 3, 5, 6, 7, 8]
    assert kind_t(K, src, st, False, "entity", 4, 60).tolist() == [5, 6, 7, 8]
    assert kind_t(K, src, st, True, "passage", 4, 60).tolist() == K.D.rrf(
        [a1[0], a1[0], st["KTst"], st["KTall"], src[4][0]], 60).tolist()
    # the pools file from KIND_T's order: base and seeds, then the order's first B_q nodes outside them
    o = kind_t(K, src, st, True, "entity", 4, 60)
    bs = [np.asarray([2, 9]), np.asarray([5])]
    inc = [np.asarray([2, 9, 10, 11, 12]), np.asarray([5, 30, 31])]
    ords = [C1.outside(o, bs[0])[:3], C1.outside(kind_t(K, src, st, False, "entity", 4, 60), bs[1])[:2]]
    assert ords[0].tolist() == [1, 3, 5] and ords[1].tolist() == [6, 7]
    arr = C4S.filed_arrays(["a", "b"], bs, inc, ords)
    assert arr["B_q"].tolist() == [3, 2] and arr["count"].tolist() == [3, 2] and arr["nodes"].tolist() == [1, 3, 5, 6, 7]
    assert C4S.expected_rows(arr).tolist() == [5, 3]
    # the positions check: equal on the first nq questions' golds, unequal on any moved gold or another count
    pos = [np.asarray([0, 4]), np.asarray([-1]), np.asarray([2, 3])]
    ptr = np.asarray([0, 2, 3, 5])
    filed = np.asarray([0, 4, -1, 2, 3])
    assert same_positions(pos, ptr, filed, 3) and same_positions(pos[:2], ptr, filed, 2)
    assert not same_positions(pos[:2], ptr, filed, 3) and not same_positions([pos[0], np.asarray([5])], ptr, filed, 2)
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        np.savez(T / "pos.npz", ptr=ptr, KIND_T=filed.astype(np.int32), RRF5=np.zeros(5, np.int32))
        part = {"carves": {"ds__s1eval": {"positions_npz_sha256": sha_file(T / "pos.npz")}}}
        p_, f_ = filed_positions("ds", "s1eval", part, T / "pos.npz")
        assert p_.tolist() == ptr.tolist() and f_.tolist() == filed.tolist()
        part["carves"]["ds__s1eval"]["positions_npz_sha256"] = "0" * 64
        try:
            filed_positions("ds", "s1eval", part, T / "pos.npz")
            raise AssertionError("a positions file its part does not name passed")
        except SystemExit:
            pass
        # the verdict's choice
        write_json(T / "v.json", {"chosen": "KIND_T"})
        assert picked(T / "v.json")[0] == "KIND_T"
        write_json(T / "v.json", {"call": "NOT_ALL_SIX"})
        try:
            picked(T / "v.json")
            raise AssertionError("a verdict without a choice passed")
        except SystemExit:
            pass
    # replace_pools takes the file: every pool keeps its frozen size, a short question keeps what its order offers
    prep = SimpleNamespace(pop=SimpleNamespace(ids=["a", "b"]), dense_ids=None, splade_ids=None,
                           seeds=[np.asarray([9]), np.asarray([5])], pools=[inc[0].copy(), inc[1].copy()],
                           seeds_added=np.zeros(2, np.int64))

    class FakeM3C:
        @staticmethod
        def base_rows(construction, d, s, m3a, m3b_contract, constant):
            return [np.asarray([2]), np.asarray([5])]

        @staticmethod
        def build_pool(base, seeds, exp):
            measured = np.union1d(base, exp)
            return np.union1d(measured, seeds), int(np.isin(seeds, measured, invert=True).sum())

    cfg = {"retrieval_pools": {"equal_rrf": {"constant": 60}}}
    stt = C4S.replace_pools(prep, arr, FakeM3C, None, cfg, None, None)
    assert stt == {"questions": 2, "changed": 2, "short": 0}, stt
    assert prep.pools[0].tolist() == [1, 2, 3, 5, 9] and prep.pools[1].tolist() == [5, 6, 7]
    log(f"p1c2 selftest: P1c's helpers; KIND_T is P1e's ROUTEst on the route, A1(st) off it and RRF5T for a passage "
        f"question; the pools file keeps base and seeds and KIND_T's first B_q nodes outside them, at each frozen "
        f"size; the positions check and its file's sha256; the verdict's choice; P1c2's roots are its own "
        f"({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("pools", "copy-bases", "gate"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "pools":
        if a.dataset not in SIX:
            ap.error("pools DATASET: one of " + ", ".join(SIX))
        if a.host:                          # P1e's host mode: the frozen package on the mirror
            import p1e_anchor_kind as K
            K.G.on_the_mirror()
        return pools_stage(a.dataset, a.threads, a.limit)
    if a.cmd == "copy-bases":
        return copy_bases()
    if a.cmd == "gate":
        return gate()
    ap.error("pools, copy-bases, gate or --selftest")


if __name__ == "__main__":
    sys.exit(main())
