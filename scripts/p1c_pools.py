"""P1c's pools and host stages (docs/P1C_POOLS_RETRAINED.md): the models' pools rebuilt from P1b2's chosen construction
at each question's own frozen size, in step 4c's file format, for outputs/mp_unified/look_p1c.py; the bases copied
into P1c's cache root; and the identity gate.

    python outputs/host_ops/pylib_run.py scripts/p1c_pools.py pools DATASET [--threads 5] [--limit N] --host
    python scripts/p1c_pools.py copy-bases
    python scripts/p1c_pools.py gate
    python scripts/p1c_pools.py --selftest

pools       The screen's carves of one dataset (its fit carve, if it trains, and s1eval). Per question, P1b2's seven
            sources and P1b's fifteen features (scripts/p1b2_rank_gate.py, unchanged), P1b's gate (class MX's weights)
            and, for RC, P1b2's gate. The construction's order is the class P1b2's hard read takes (RC) or class MX
            (MX), cut at 7,000 as P1b2 cut it. The pool is step 4c's: base and seeds, then the order's first B_q nodes
            outside them, B_q = |I_q| - |base and seeds| (m3b_compile.build_pool), so every pool keeps its frozen size;
            a question whose order offers fewer keeps them all and is short. Checks, any failure stops the stage:
              - the construction is the one P1b2's verdict picked (outputs/p1b2/verdict.json) and the one declared;
              - step 1's host cache rows are the frozen pool sizes built here (step 4c's step1_rows);
              - on a fit carve, the features equal P1b2's gather, and the chosen class's area equals the gather's;
              - on s1eval, the order's ALL and recall at |I_q| equal P1b2's filed record for the construction.
            Written to outputs/p1c/pools/<D>/<D>__<carve>.npz (step 4c's arrays and the chosen class per question) and
            outputs/p1c/pools/<D>/pools.json (each file's sha256 and counts, the coverage of I_q and of P1c's pools).
            --limit N: a development run on the first N questions of each carve; it writes nothing and logs no
            number, only that every step and check ran.
copy-bases  step 1's cache bases (basis_2wiki, basis_hotpotqa) into outputs/p1c/cache (step 4c's copy_bases).
gate        squad's carves: every cache array equals step 1's (its pools cannot change: |I_q| is |base and seeds|),
            score2 within 1e-4; every carve: each question's rows equal its filed pool size, and step 1's rows its
            |I_q|; every look shard filed a pools record naming the file. Writes outputs/p1c/gate.json; exits 1 on a
            failure.
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
import step4c_walk_pools as C4S  # noqa: E402  (numba-free at import)

OUT = ROOT / "outputs" / "p1c"
POOLS = OUT / "pools"
LOOK = OUT / "look"
CACHE = OUT / "cache"
GATE = OUT / "gate.json"
P1B2 = ROOT / "outputs" / "p1b2"
SIX = C4S.SIX
TRAIN5 = SIX[:5]
FIT_CARVE = {"metaqa": "fit", "squad": "fit", "musique": "s1fit", "hotpotqa": "fit", "2wiki": "fit"}
SCREEN = tuple((d, FIT_CARVE[d]) for d in TRAIN5) + tuple((d, "s1eval") for d in SIX)
CONSTRUCTIONS = ("RC", "MX")
CONSTRUCTION = "MX"        # declared in docs/P1C_POOLS_RETRAINED.md, after P1b2's verdict and before any number of P1c
log, utc, write_json, sha_file = C4S.log, C4S.utc, C4S.write_json, C4S.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def carves_for(name):
    return [c for d, c in SCREEN if d == name]


def outside(order, bs):
    """The order without base and seeds, its order kept."""
    order = np.asarray(order, np.int64)
    return order[~np.isin(order, np.asarray(bs, np.int64))]


def chosen_class(construction, pi, orders, classes, hard_choice):
    """RC: the class P1b2's hard read takes; MX: class MX."""
    if construction == "RC":
        return hard_choice(pi, orders)
    if construction == "MX":
        return classes.index("MX")
    raise ValueError(construction)


def picked(verdict_path=P1B2 / "verdict.json"):
    """The construction P1b2's verdict picked for P1c (section 5 of docs/P1B2_RANK_GATE.md)."""
    v = json.loads(Path(verdict_path).read_text(encoding="utf-8"))
    c = (v.get("p1c") or {}).get("construction")
    if c not in CONSTRUCTIONS:
        raise SystemExit(f"{verdict_path}: no P1c pick ({v.get('p1c')})")
    return c, sha_file(verdict_path)


def coverage(sets, golds):
    a = r = 0.0
    for s, g in zip(sets, golds):
        hit = np.isin(g, s)
        a += bool(hit.all()) if g.size else 0.0
        r += float(hit.mean()) if g.size else 0.0
    nq = max(1, len(sets))
    return {"ALL": round(a / nq, 4), "recall": round(r / nq, 4)}


# ── pools ────────────────────────────────────────────────────────────────────


def pools_stage(name, threads, limit=None):
    import p1b2_rank_gate as R          # P1b2's sources, classes and gates (imports numba)
    B, P, E, S4 = R.B, R.P, R.E, R.S4
    if CONSTRUCTION not in CONSTRUCTIONS:
        raise SystemExit("P1c's construction is not declared yet (CONSTRUCTION in this script)")
    pick, verdict_sha = picked()
    if pick != CONSTRUCTION:
        raise SystemExit(f"P1b2's verdict picked {pick}; this script declares {CONSTRUCTION}")
    op = S4.Opened()
    gate1 = R.p1b_gate()
    gate2 = json.loads(R.GATE.read_text(encoding="utf-8"))
    if gate2["script_sha256"] != R.script_sha256() or gate2["p1b_gate_sha256"] != sha_file(B.GATE):
        raise SystemExit("P1b2's gate is not its filed script's, or was fit on another P1b gate")
    if gate2["freeze_RECORD_SHA256"] != op.freeze["RECORD_SHA256"]:
        raise SystemExit("P1b2's gate was fit on another freeze")
    rc_rec = json.loads(R.part_path(name).read_text(encoding="utf-8"))
    if rc_rec.get("gate_sha256") != sha_file(R.GATE) or rc_rec.get("script_sha256") != R.script_sha256():
        raise SystemExit(f"{R.part_path(name)}: not applied with P1b2's filed gate and script")
    ref = None
    if name in TRAIN5:
        grec = json.loads((P1B2 / f"gather_{name}.json").read_text(encoding="utf-8"))
        if sha_file(R.gather_path(name)) != grec["npz_sha256"]:
            raise SystemExit(f"{R.gather_path(name)}: not its record's file")
        with np.load(R.gather_path(name)) as z:
            ref = {k: z[k] for k in ("X", "A", "ids")}
    classes = R.CLASSES
    d = POOLS / name
    mpath = d / "pools.json"
    manifest = {"files": {}}
    if not limit and mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    manifest.update({"declared_in": "docs/P1C_POOLS_RETRAINED.md", "dataset": name, "construction": CONSTRUCTION,
                     "p1b2_verdict_sha256": verdict_sha, "p1b2_gate_sha256": sha_file(R.GATE),
                     "p1b_gate_sha256": sha_file(B.GATE), "p1b2_script_sha256": R.script_sha256(),
                     "step4c_script_sha256": C4S.script_sha256(), "script_sha256": script_sha256(),
                     "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"]})

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        nq = len(pop.ids) if not limit else min(len(pop.ids), limit)
        ids = list(pop.ids)[:nq]
        bs, inc = bs[:nq], inc[:nq]
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds[:nq]]
        inc_size = np.asarray([np.asarray(p).size for p in inc], dtype=np.int32)
        host_rows = None if limit else C4S.step1_rows(name, carve, ids, inc_size)
        fit = carve == FIT_CARVE.get(name)
        if fit and (ref is None or ref["ids"].tolist()[:nq] != ids):
            raise SystemExit(f"{name}/{carve}: P1b2's gather does not hold the carve's questions in order")
        orders, chosen, pos, X = [], [], [], []
        bad_area = 0
        for lo in range(0, nq, S4.BATCH):
            js = range(lo, min(nq, lo + S4.BATCH))
            srcs = R.sources7(name, prep, js, linked, union, constant, threads)
            xb = np.vstack([B.features(prep, j, s[:len(B.SOURCES)]) for j, s in zip(js, srcs)])
            wmx = B.gate_weights(gate1, xb)
            pis = B.gate_weights(gate2, xb)
            X.append(xb)
            for i, (j, src) in enumerate(zip(js, srcs)):
                co = R.class_orders(src, wmx[i])
                k = chosen_class(CONSTRUCTION, pis[i], co, classes, R.hard_choice)
                o = np.asarray(co[classes[k]], np.int64)
                chosen.append(k)
                pos.append(P.positions(o, golds[j]))
                if fit and R.area(o, golds[j]) != float(ref["A"][j, k]):
                    bad_area += 1
                # the file keeps the first B_q nodes outside base and seeds (step 4c's expansions); hold no more
                orders.append(outside(o, bs[j])[:max(0, int(inc_size[j]) - int(np.asarray(bs[j]).size))])
        X = np.vstack(X) if X else np.empty((0, len(B.FEATURES)))
        checks = {}
        if fit:
            checks["features_equal_gather"] = bool(X.shape == ref["X"][:nq].shape and np.array_equal(X, ref["X"][:nq]))
            checks["areas_equal_gather"] = bad_area == 0
        if carve == "s1eval":
            got = P.at_sizes(pos, inc_size)
            want = rc_rec["carves"][f"{name}__s1eval"]["orders"][CONSTRUCTION]["at_I_q"]
            checks["at_I_q_equals_p1b2"] = True if limit else got == want
        if not all(checks.values()):
            raise SystemExit(f"{name}/{carve}: {checks} (areas off on {bad_area} questions)")
        arrays = C4S.filed_arrays(ids, bs, inc, orders)
        arrays["class"] = np.asarray(chosen, dtype=np.int8)
        ptr, flat, count, b_q = arrays["ptr"], arrays["nodes"], arrays["count"], arrays["B_q"]
        pools = [np.union1d(np.asarray(bs[j], np.int64), flat[ptr[j]:ptr[j + 1]].astype(np.int64)) for j in range(nq)]
        expect = C4S.expected_rows(arrays)
        if any(p.size != int(e) for p, e in zip(pools, expect)):
            raise SystemExit(f"{name}/{carve}: a pool is not its expected size")
        short = int((count < b_q).sum())
        if limit:
            log(f"{name}/{carve} development run: {nq} questions, the sources, classes, checks {sorted(checks)} and "
                f"the pools ran ({time.time() - t:.0f}s); nothing written")
            return
        d.mkdir(parents=True, exist_ok=True)
        fname = f"{name}__{carve}.npz"
        tmp = d / f"{name}__{carve}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, d / fname)
        manifest["files"][f"{name}__{carve}"] = {
            "file": fname, "sha256": sha_file(d / fname), "dataset": name, "carve": carve, "questions": nq,
            "carve_ids_sha256": pop.digest, "construction": CONSTRUCTION, "walk_families": list(families),
            "changed": int(arrays["changed"].sum()), "short": short,
            "B_q": {"mean": round(float(b_q.mean()), 1) if nq else None, "max": int(b_q.max()) if nq else None},
            "pool_mean": round(float(inc_size.mean()), 1) if nq else None,
            "class_share": {c: round(float(np.mean(np.asarray(chosen) == k)), 4) for k, c in enumerate(classes)},
            "ALL": {"I": coverage(inc, golds)["ALL"], "P1c": coverage(pools, golds)["ALL"]},
            "recall": {"I": coverage(inc, golds)["recall"], "P1c": coverage(pools, golds)["recall"]},
            "order_at_I_q": P.at_sizes(pos, inc_size), "checks": checks, "step1_cache_rows": host_rows,
            "seconds": round(time.time() - t, 1), "utc": utc()}
        manifest["utc"] = utc()
        write_json(mpath, manifest)
        e = manifest["files"][f"{name}__{carve}"]
        log(f"{name}/{carve}: {nq} questions, {e['changed']} pools changed, {short} short; ALL I {e['ALL']['I']} "
            f"P1c {e['ALL']['P1c']} ({e['seconds']:.0f}s)")
        del pools, orders
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


def look_pools_records(ds, carve, ent, look=LOOK):
    """Every look shard of the carve filed a pools record naming the file and its counts."""
    d = Path(look) / ds / carve
    looks = sorted(p.name[len("record"):-len(".json")] for p in d.glob("record*.json"))
    pools = sorted(p.name[len("pools"):-len(".json")] for p in d.glob("pools*.json"))
    bad = []
    if not looks or looks != pools:
        bad.append(f"look records {looks} and pools records {pools} differ")
    for tag in pools:
        r = json.loads((d / f"pools{tag}.json").read_text(encoding="utf-8"))
        if r.get("pools_sha256") != ent["sha256"]:
            bad.append(f"pools{tag}.json names another pools file")
        if r.get("changed") != ent["changed"] or r.get("short") != ent["short"]:
            bad.append(f"pools{tag}.json counts {r.get('changed')}/{r.get('short')} changed/short, the manifest "
                       f"{ent['changed']}/{ent['short']}")
    return bad


def rows_check(filed, ids1, n1, ids4, n4):
    """Step 1's rows are the filed |I_q| and P1c's are the filed pool sizes, question by question."""
    at = {str(q): k for k, q in enumerate(filed["ids"].tolist())}
    if ids1 != ids4 or any(q not in at for q in ids4):
        return {"verdict": "FAIL: the cache's questions are not step 1's or not the pools file's"}
    k = np.asarray([at[q] for q in ids4], dtype=np.int64)
    ok1 = bool(np.array_equal(n1, filed["inc_size"][k].astype(np.int64)))
    ok4 = bool(np.array_equal(n4, C4S.expected_rows(filed)[k]))
    return {"step1_rows_are_frozen_sizes": ok1, "rows_are_p1c_sizes": ok4, "verdict": "PASS" if ok1 and ok4 else "FAIL"}


def squad_identity(p1, p4, r1, r4):
    sq = {"parts": [len(p1), len(p4)], "arrays": {}, "score2_max_abs_diff": 0.0}
    if len(p1) != len(p4) or [r["query_range"] for r in r1] != [r["query_range"] for r in r4]:
        sq["verdict"] = "FAIL: not step 1's parts"
        return sq
    for a, b, ra, rb in zip(p1, p4, r1, r4):
        if sorted(ra["arrays"]) != sorted(rb["arrays"]):
            sq["arrays"][a.name] = "other arrays"
            continue
        for key in sorted(ra["arrays"]):
            if ra["arrays"][key]["sha256"] == rb["arrays"][key]["sha256"]:
                continue
            if key != "score2":
                sq["arrays"][f"{a.name}/{key}"] = "different"
                continue
            dd = np.abs(np.load(a / "score2.npy").astype(np.float64) - np.load(b / "score2.npy").astype(np.float64))
            sq["score2_max_abs_diff"] = max(sq["score2_max_abs_diff"], float(dd.max()) if dd.size else 0.0)
    sq["verdict"] = "FAIL" if sq["arrays"] or sq["score2_max_abs_diff"] > C4S.SCORE2_TOL else "PASS"
    return sq


def gate():
    t0 = time.time()
    rec = {"declared_in": "docs/P1C_POOLS_RETRAINED.md", "construction": CONSTRUCTION, "manifests": {}, "rows": {},
           "looks": {}, "squad": {}, "script_sha256": script_sha256()}
    fails = []
    for ds, cv in SCREEN:
        mp = POOLS / ds / "pools.json"
        rec["manifests"].setdefault(ds, sha_file(mp))
        filed, ent, msha = C4S.load_filed(ds, cv, POOLS / ds)
        if msha != rec["manifests"][ds]:
            raise SystemExit(f"{mp}: changed while the gate ran")
        bad = look_pools_records(ds, cv, ent)
        rec["looks"][f"{ds}={cv}"] = bad or "PASS"
        fails += [f"{ds}/{cv}: {b}" for b in bad]
        ids1, n1, p1, r1 = C4S.carve_rows(C4S.STEP1_CACHE, ds, cv)
        ids4, n4, p4, r4 = C4S.carve_rows(CACHE, ds, cv)
        row = {"questions": len(ids4), "short": ent["short"], "changed": ent["changed"], **rows_check(filed, ids1, n1, ids4, n4)}
        rec["rows"][f"{ds}={cv}"] = row
        if row["verdict"] != "PASS":
            fails.append(f"{ds}/{cv}: rows {row['verdict']}")
        if ds == "squad":
            sq = squad_identity(p1, p4, r1, r4)
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
    t0 = time.time()
    assert len(SCREEN) == 11 and carves_for("musique") == ["s1fit", "s1eval"] and carves_for("webqsp") == ["s1eval"]
    # outside: base and seeds dropped, the order kept
    o = np.asarray([9, 4, 7, 1, 3, 8], np.int64)
    assert outside(o, np.asarray([1, 7])).tolist() == [9, 4, 3, 8]
    # the chosen class: RC is the hard read, MX is class MX
    classes = ("S", "L", "R", "A1", "LR", "SL", "SLR", "MX")
    orders = {c: np.asarray([1, 2]) for c in classes}
    orders["L"] = np.empty(0, np.int64)

    def hard(pi, ords):
        avail = np.asarray([ords[c].size > 0 for c in classes])
        return int(np.argmax(np.where(avail, pi, -1.0)))

    pi = np.asarray([0.1, 0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    assert chosen_class("RC", pi, orders, classes, hard) == 0          # L is empty: the next best, S
    assert chosen_class("MX", pi, orders, classes, hard) == 7
    try:
        chosen_class("A3", pi, orders, classes, hard)
        raise AssertionError("an undeclared construction ran")
    except ValueError:
        pass
    # step 4c's format: base and seeds, then the order's first B_q nodes outside them; the frozen size kept; short
    bs = [np.asarray([1, 2, 3]), np.asarray([5])]
    inc = [np.asarray([1, 2, 3, 10, 11]), np.asarray([5, 6, 7, 8])]
    ords = [outside([4, 2, 9, 12, 13], bs[0]), outside([5, 20], bs[1])]
    arr = C4S.filed_arrays(["a", "b"], bs, inc, ords)
    assert arr["B_q"].tolist() == [2, 3] and arr["count"].tolist() == [2, 1]
    assert arr["nodes"].tolist() == [4, 9, 20] and arr["changed"].tolist() == [True, True]
    assert C4S.expected_rows(arr).tolist() == [5, 2]                 # b is short: |bs| + its one node

    class FakeM3C:
        @staticmethod
        def base_rows(construction, d, s, m3a, m3b_contract, constant):
            return [np.asarray([1, 2]), np.asarray([5])]

        @staticmethod
        def build_pool(base, seeds, exp):
            measured = np.union1d(base, exp)
            return np.union1d(measured, seeds), int(np.isin(seeds, measured, invert=True).sum())

    class Pop:
        ids = ["a", "b"]

    class Prep:
        pop = Pop()
        dense_ids = splade_ids = None
        seeds = [np.asarray([3]), np.asarray([5])]
        pools = [inc[0].copy(), inc[1].copy()]
        seeds_added = np.zeros(2, np.int64)

    prep = Prep()
    cfg = {"retrieval_pools": {"equal_rrf": {"constant": 60}}}
    st = C4S.replace_pools(prep, arr, FakeM3C, None, cfg, None, None)
    assert st == {"questions": 2, "changed": 2, "short": 1}, st
    assert prep.pools[0].tolist() == [1, 2, 3, 4, 9] and prep.pools[1].tolist() == [5, 20]
    # coverage
    cv = coverage([np.asarray([1, 2]), np.asarray([3])], [np.asarray([1, 2]), np.asarray([3, 4])])
    assert cv == {"ALL": 0.5, "recall": 0.75}, cv
    # the gate's pieces
    filed = {"ids": np.asarray(["a", "b"]), "inc_size": np.asarray([5, 4], np.int32), "bs_size": np.asarray([3, 1]),
             "B_q": np.asarray([2, 3]), "count": np.asarray([2, 1])}
    assert rows_check(filed, ["a", "b"], np.asarray([5, 4]), ["a", "b"], np.asarray([5, 2]))["verdict"] == "PASS"
    assert rows_check(filed, ["a", "b"], np.asarray([5, 4]), ["a", "b"], np.asarray([5, 4]))["verdict"] == "FAIL"
    assert rows_check(filed, ["a", "b"], np.asarray([5, 4]), ["b", "a"], np.asarray([2, 5]))["verdict"].startswith("FAIL")
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        (T / "ds" / "s1eval").mkdir(parents=True)
        ent = {"sha256": "x", "changed": 2, "short": 1}
        assert look_pools_records("ds", "s1eval", ent, T)
        write_json(T / "ds" / "s1eval" / "record_0of1.json", {})
        write_json(T / "ds" / "s1eval" / "pools_0of1.json", {"pools_sha256": "x", "changed": 2, "short": 1})
        assert look_pools_records("ds", "s1eval", ent, T) == []
        write_json(T / "ds" / "s1eval" / "pools_0of1.json", {"pools_sha256": "y", "changed": 2, "short": 1})
        assert look_pools_records("ds", "s1eval", ent, T)
        # the verdict's pick
        write_json(T / "v.json", {"p1c": {"construction": "MX"}})
        assert picked(T / "v.json")[0] == "MX"
        write_json(T / "v.json", {"p1c": {}})
        try:
            picked(T / "v.json")
            raise AssertionError("a verdict without a pick passed")
        except SystemExit:
            pass
    log(f"p1c selftest: the screen's eleven carves; the order outside base and seeds keeps its order; RC takes the "
        f"hard read, MX class MX, nothing else runs; step 4c's file and replacement keep each frozen size and count "
        f"short pools; coverage; the gate's row, look-record and pick checks ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("pools", "copy-bases", "gate"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "pools":
        if a.dataset not in SIX:
            ap.error("pools DATASET: one of " + ", ".join(SIX))
        if a.host:                          # P1b2's host mode: step 4f's run_one points the frozen package at the mirror
            import importlib.util
            import p1b2_rank_gate as R
            spec = importlib.util.spec_from_file_location("s4f_run_one", ROOT / "outputs" / "step4f" / "run_one.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.on_the_mirror(R.F)
        return pools_stage(a.dataset, a.threads, a.limit)
    if a.cmd == "copy-bases":
        return copy_bases()
    if a.cmd == "gate":
        return gate()
    ap.error("pools, copy-bases, gate or --selftest")


if __name__ == "__main__":
    sys.exit(main())
