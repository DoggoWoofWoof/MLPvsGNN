"""P1g-b's pools, KT input and gate (docs/P1GB_POOLS_REFIT.md): the pool rule P1g2 chose (outputs/p1g2/verdict.json:
P1g's candidates and their gated forms, scripts/p1g2_gated.py; P1g's extra sets from scripts/p1g_pool_beam.py,
imported unchanged) written for P1c2's eleven carves (each training dataset's fit carve and
the six s1eval carves), the KT input on their caches, and the gate before any fit. No gold is read to build a pool
and nothing is fit here.

    python outputs/host_ops/pylib_run.py scripts/p1gb_pools.py pools DATASET --host [--threads T] [--limit N]
    python scripts/p1gb_pools.py copy-bases
    python outputs/host_ops/pylib_run.py scripts/p1gb_pools.py kt DATASET [--carve C] --host [--threads T]
    python scripts/p1gb_pools.py gate
    python scripts/p1gb_pools.py --selftest

pools       Per question of each of the dataset's P1c2 carves: base and seeds B and KIND_T's order (P1c2's kind_t, through
            P1e's orders, unchanged); full = the order's nodes outside B; today's fill = full's first |I_q| - |B| nodes
            (P1c2's); and the chosen rule (f, X, gated): fill_f = full's first min(|full|, round(f x |today's fill|))
            nodes, X = P1g's extra set (p1g_pool_beam.extra_sets), or nothing when the rule is gated and today's fill
            is empty. The pool is B, then fill_f, then X's nodes outside fill_f (the expansion). Checks, any failure
            stops the stage:
              - P1g2's verdict is TARGET_MET or CEILING_MET and was written by P1g2's filed script over P1g's;
              - on every question, B is P1c2's filed base and seeds, |I_q| is P1c2's, and today's fill recomputed here
                is P1c2's filed expansion;
              - on s1eval, the pools' ALL, recall and mean rows equal P1g2's record of the chosen rule on that carve
                (its eval record for the five training datasets; for webqsp, its verdict's, from P1g's stats).
            Written to outputs/p1gb/pools/<D>/<D>__<carve>.npz (step 4c's arrays where they apply: ids, inc_size,
            inc_sha, bs_size, bs_ptr, bs_nodes, ptr, nodes, count; and today_count, fill_want, fill_k, x_count) and
            outputs/p1gb/pools/<D>/pools.json (each file's sha256, counts, mean rows and the coverage of P1c2's pools
            and of P1g-b's). --limit N: a development run on the first N questions of each carve; it writes nothing and
            logs no number.
copy-bases  step 1's cache bases (basis_2wiki, basis_hotpotqa) into P1g-b's cache root (step 4c's copy_bases).
kt          P1c3's eight KT columns (scripts/p1c3_kt.py's columns, imported unchanged) for the rows of P1g-b's caches:
            the same orders, recomputed per question; the fill column marks fill_f. Checks: base and seeds and fill_f
            recomputed here are the filed ones, and each question's cache rows are its pool (as sets), each node once.
            Written to outputs/p1gb/kt/<D>/<carve>/ (kt.npy float16, n.npy, ids.json, record.json).
gate        Before any fit, on every carve: the pools file is the manifest's; the cache's questions are step 1's and the
            file's, step 1's rows are the filed |I_q| and P1g-b's the filed pool sizes (|B| + the expansion); every look
            shard filed a pools record naming the file and its counts; the KT record is on the cache's parts and root.
            squad's caches must equal P1c2's array by array when the chosen rule leaves its pools unchanged. Writes
            outputs/p1gb/gate.json; exits 1 on a failure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import p1c2_pools as PC2  # noqa: E402  (P1c2's pools, roots and checks; numba-free at import)
import p1c3_kt as KT3  # noqa: E402  (P1c3's KT columns and cache reader)
import p1g_pool_beam as PG  # noqa: E402  (P1g's candidates and extra sets)
import p1g2_gated as PG2  # noqa: E402  (P1g2's candidates, verdict and s1eval records)

C1, C4S = PC2.C1, PC2.C4S
log, utc, write_json, sha_file = C4S.log, C4S.utc, C4S.write_json, C4S.sha_file
DECLARED = "docs/P1GB_POOLS_REFIT.md"
OUT = ROOT / "outputs" / "p1gb"
POOLS, LOOK, CACHE, KT = OUT / "pools", OUT / "look", OUT / "cache", OUT / "kt"
GATE = OUT / "gate.json"
SIX, TRAIN5, SCREEN = PC2.SIX, PC2.TRAIN5, PC2.SCREEN
MEETS = ("TARGET_MET", "CEILING_MET")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def carves_for(name):
    return PC2.carves_for(name)


def construction_of(chosen):
    return f"P1G:{chosen}"


# ── the chosen rule ──────────────────────────────────────────────────────────


def chosen_rule(verdict_path=None):
    """(rule name, f, extra name, gated, verdict sha256) of P1g2's verdict; stops unless it met its targets and was
    written by P1g2's filed script over P1g's."""
    verdict_path = Path(verdict_path or PG2.VERDICT)
    v = json.loads(verdict_path.read_text(encoding="utf-8"))
    if v.get("script_sha256") != PG2.script_sha256() or v.get("p1g_script_sha256") != PG.script_sha256():
        raise SystemExit(f"{verdict_path}: not written by P1g2's filed script over P1g's")
    if v.get("verdict") not in MEETS:
        raise SystemExit(f"{verdict_path}: P1g2's verdict is {v.get('verdict')}; P1g-b runs only on {MEETS}")
    by = {c[0]: c for c in PG2.candidates()}
    if v.get("chosen") not in by:
        raise SystemExit(f"{verdict_path}: {v.get('chosen')} is not one of P1g2's candidates")
    _n, src, gated = by[v["chosen"]]
    _s, f, ex = {c[0]: c for c in PG.candidates()}[src]
    return v["chosen"], f, ex, bool(gated), sha_file(verdict_path)


def p1g_record(name, carve, chosen):
    """P1g2's filed summary of the chosen rule on an s1eval carve (its eval record; webqsp's from its verdict), or
    None on another carve. A training dataset's missing eval record stops the stage."""
    if carve != "s1eval":
        return None
    if name == "webqsp":
        v = json.loads(PG2.VERDICT.read_text(encoding="utf-8"))
        if v.get("chosen") != chosen:
            raise SystemExit(f"{PG2.VERDICT}: chose {v.get('chosen')}, not {chosen}")
        return v["beside_webqsp_s1eval"]["chosen"]
    if not (PG2.OUT / f"eval_{name}.json").exists():
        raise SystemExit(f"{name}: P1g2's s1eval record is missing (run p1g2_gated.py eval {name} first)")
    s, _v = PG2.eval_summary(name)
    return s[chosen]


def extra_applies(gated, today_size):
    """P1g2's gate: a gated rule's extra set applies only to a question KIND_T expands (today's fill above 0)."""
    return not gated or int(today_size) > 0


def expansion(full, today_size, f, X):
    """(expansion, fill_k, fill_want, x_count): full's first min(|full|, round(f x today_size)) nodes, then X's
    nodes outside them in X's order."""
    full = np.asarray(full, np.int64)
    want = int(np.rint(f * today_size))
    k = min(full.size, want)
    fill = full[:k]
    X = np.asarray(X, np.int64)
    xs = X[~np.isin(X, fill)]
    return np.concatenate([fill, xs]), k, want, int(xs.size)


def pool_arrays(ids, bs, inc, exps, today, wants, ks, xcs):
    """One carve's pools file (step 4c's arrays where they apply)."""
    bs_ptr, bs_flat = C4S.flat_of([np.asarray(u, np.int64) for u in bs])
    ptr, flat = C4S.flat_of([np.asarray(e, np.int64) for e in exps])
    if max([int(flat.max()) if flat.size else 0, int(bs_flat.max()) if bs_flat.size else 0]) >= 2 ** 31:
        raise SystemExit("a node position does not fit int32")
    return {"ids": np.asarray(list(ids)), "inc_size": np.asarray([np.asarray(p).size for p in inc], np.int32),
            "inc_sha": np.asarray([C4S.pool_sha(p) for p in inc], dtype="S64"),
            "bs_size": np.asarray([np.asarray(u).size for u in bs], np.int32), "bs_ptr": bs_ptr,
            "bs_nodes": bs_flat.astype(np.int32), "ptr": ptr, "nodes": flat.astype(np.int32),
            "count": np.asarray([e.size for e in exps], np.int32), "today_count": np.asarray(today, np.int32),
            "fill_want": np.asarray(wants, np.int32), "fill_k": np.asarray(ks, np.int32),
            "x_count": np.asarray(xcs, np.int32)}


def expected_rows(filed):
    """Each question's pool size: base and seeds plus its expansion."""
    return filed["bs_size"].astype(np.int64) + filed["count"].astype(np.int64)


def pools_of(filed):
    return [np.union1d(filed["bs_nodes"][filed["bs_ptr"][j]:filed["bs_ptr"][j + 1]].astype(np.int64),
                       filed["nodes"][filed["ptr"][j]:filed["ptr"][j + 1]].astype(np.int64))
            for j in range(filed["ids"].size)]


# ── pools (the host) ─────────────────────────────────────────────────────────


def pools_stage(name, threads, limit=0):
    import p1e_anchor_kind as K         # P1e's walks, kind and orders, through it P1d's sources (imports numba)
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    D5, D5B = PG.D5, PG.D5B
    chosen, f, ex, gated, vsha = chosen_rule()
    op = S4.Opened()
    ds = op.canonical.Dataset(name, root=str(op.served))
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes_e = DenseNodes(ds.embeddings("dense", "docs"))
    d = POOLS / name
    mpath = d / "pools.json"
    manifest = {"files": {}}
    if not limit and mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    manifest.update({"declared_in": DECLARED, "dataset": name, "construction": construction_of(chosen),
                     "chosen": chosen, "fill_fraction": f, "extra": ex, "gated": gated, "p1g2_verdict_sha256": vsha,
                     "script_sha256": script_sha256(), "p1g_script_sha256": PG.script_sha256(),
                     "p1g2_script_sha256": PG2.script_sha256(),
                     "p1c2_script_sha256": PC2.script_sha256(), "p1e_script_sha256": K.script_sha256(),
                     "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"]})
    cache = {}

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        stores, unions = G.open_unions(cache, op, name, union)
        adj = [(np.ascontiguousarray(stores[fm].indptr, np.int64), np.ascontiguousarray(stores[fm].col))
               for fm in S4.FAMILIES]
        n_st = int(unions["st"].deg.size)
        f2, e2, _m2 = C4S.load_filed(name, carve, PC2.POOLS / name)
        if e2.get("construction") != PC2.CONSTRUCTION:
            raise SystemExit(f"{name}/{carve}: P1c2's pools file is {e2.get('construction')}'s")
        nq = min(limit, len(U)) if limit else len(U)
        ids = [str(q) for q in list(pop.ids)[:nq]]
        if [str(q) for q in f2["ids"].tolist()[:nq]] != ids:
            raise SystemExit(f"{name}/{carve}: P1c2's pools file holds other questions than the carve")
        golds = [np.unique(np.asarray(g, np.int64)) for g in pop.golds[:nq]]
        Bs, incs, exps, today, wants, ks, xcs, todays = [], [], [], [], [], [], [], []
        for lo in range(0, nq, S4.BATCH):
            js = list(range(lo, min(nq, lo + S4.BATCH)))
            srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
            steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
            for i, j in enumerate(js):
                src, _nn, (L, Ls) = srcs[i]
                knd = K.kind(Ls, prep.seeds[j], nonly)
                o = PC2.kind_t(K, src, steered[i], bool(D.routes(L, Ls)), knd, n_st, constant)
                B = np.unique(np.asarray(bs[j], np.int64))
                fb = f2["bs_nodes"][f2["bs_ptr"][j]:f2["bs_ptr"][j + 1]].astype(np.int64)
                if not np.array_equal(B, fb):
                    raise SystemExit(f"{name}/{carve} question {j} ({ids[j]}): base and seeds are not P1c2's")
                isz = int(np.asarray(inc[j]).size)
                if isz != int(f2["inc_size"][j]):
                    raise SystemExit(f"{name}/{carve} question {j} ({ids[j]}): |I_q| is not P1c2's")
                full = C1.outside(o, B)
                tf = full[:max(0, isz - B.size)]
                if not np.array_equal(tf, f2["nodes"][f2["ptr"][j]:f2["ptr"][j + 1]].astype(np.int64)):
                    raise SystemExit(f"{name}/{carve} question {j} ({ids[j]}): today's fill is not P1c2's")
                w_B = D5.weights_of(K.steer(prep, j, constant), B)
                q = D5B.unit(np.asarray(prep.qemb[j], np.float64))
                X = PG.extra_sets(adj, B, w_B, q, nodes_e.read, constant, {ex})[ex] if extra_applies(gated, tf.size)                     else PG.EMPTY
                e, k, want, xc = expansion(full, tf.size, f, X)
                Bs.append(B)
                incs.append(np.asarray(inc[j], np.int64))
                exps.append(e)
                today.append(tf.size)
                wants.append(want)
                ks.append(k)
                xcs.append(xc)
                todays.append(np.union1d(B, tf))
            if (lo // S4.BATCH) % 10 == 9:
                log(f"{name}/{carve}: {js[-1] + 1}/{nq} ({time.time() - t:.0f}s)")
        arrays = pool_arrays(ids, Bs, incs, exps, today, wants, ks, xcs)
        pools = pools_of(arrays)
        if any(p.size != int(r) for p, r in zip(pools, expected_rows(arrays))):
            raise SystemExit(f"{name}/{carve}: a pool repeats a node or holds base and seeds in its expansion")
        cov_t, cov_g = C1.coverage(todays, golds), C1.coverage(pools, golds)
        rows = expected_rows(arrays)
        checks = {"bs_equal_p1c2": True, "inc_equal_p1c2": True, "today_fill_equal_p1c2": True}
        filed_g = None if limit else p1g_record(name, carve, chosen)
        if filed_g is not None:
            checks["equals_p1g_record"] = bool(cov_g["ALL"] == filed_g["ALL"] and cov_g["recall"] == filed_g["recall"]
                                               and round(float(rows.mean()), 1) == filed_g["rows"])
            if not checks["equals_p1g_record"]:
                raise SystemExit(f"{name}/{carve}: ALL {cov_g}, rows {rows.mean():.1f}; P1g filed {filed_g}")
        if limit:
            log(f"{name}/{carve} development run: {nq} questions, KIND_T's order, today's fill against P1c2's, the "
                f"extra set and the pools ran ({time.time() - t:.0f}s); nothing written")
            return
        d.mkdir(parents=True, exist_ok=True)
        fname = f"{name}__{carve}.npz"
        tmp = d / f"{name}__{carve}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, d / fname)
        changed = int(sum(not np.array_equal(p, np.sort(i)) for p, i in zip(pools, incs)))
        manifest["files"][f"{name}__{carve}"] = {
            "file": fname, "sha256": sha_file(d / fname), "dataset": name, "carve": carve, "questions": nq,
            "carve_ids_sha256": pop.digest, "construction": construction_of(chosen), "changed": changed,
            "short": int((arrays["fill_k"] < arrays["fill_want"]).sum()),
            "rows": {"mean": round(float(rows.mean()), 1), "p90": round(float(np.quantile(rows, 0.9)), 1),
                     "max": int(rows.max())},
            "today_rows_mean": round(float(np.mean([p.size for p in todays])), 1),
            "x_count_mean": round(float(arrays["x_count"].mean()), 1),
            "ALL": {"P1c2": cov_t["ALL"], "P1gb": cov_g["ALL"]}, "recall": {"P1c2": cov_t["recall"], "P1gb": cov_g["recall"]},
            "checks": checks, "seconds": round(time.time() - t, 1), "utc": utc()}
        manifest["utc"] = utc()
        write_json(mpath, manifest)
        e = manifest["files"][f"{name}__{carve}"]
        log(f"{name}/{carve}: {nq} questions, rows {e['rows']['mean']} (today {e['today_rows_mean']}); ALL P1c2 "
            f"{cov_t['ALL']} P1g-b {cov_g['ALL']} ({e['seconds']:.0f}s)")
        del pools, todays, exps
        gc.collect()

    E.run_carves(op, name, set(carves_for(name)), ("A3",), threads, sink)
    return 0


# ── the pool replacement (outputs/mp_unified/look_p1gb.py runs it after m3b_compile.prepare) ──────────────────


def replace_pools(prep, filed, m3c, construction, cfg_h, m3a, m3b_contract):
    """Each prepared question's pool becomes base and seeds and its filed P1g-b expansion (m3c.build_pool), in place.
    Stops on a missing question, a base and seed set or a frozen pool that is not the filed one, or an expansion that
    repeats a node, holds a node of base and seeds, or is not the filed fill plus extra nodes."""
    constant = int(cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    bases = m3c.base_rows(construction, prep.dense_ids, prep.splade_ids, m3a, m3b_contract, constant)
    at = {str(q): k for k, q in enumerate(filed["ids"].tolist())}
    changed = short = 0
    for j, q in enumerate(prep.pop.ids):
        k = at.get(str(q))
        if k is None:
            raise SystemExit(f"{q}: not in the pools file")
        base = np.asarray(bases[j], dtype=np.int64)
        seeds = np.asarray(prep.seeds[j], dtype=np.int64)
        bs = np.union1d(base, seeds)
        if not np.array_equal(bs, filed["bs_nodes"][filed["bs_ptr"][k]:filed["bs_ptr"][k + 1]].astype(np.int64)):
            raise SystemExit(f"{q}: base and seeds are not the filed ones")
        inc = np.asarray(prep.pools[j], dtype=np.int64)
        sha = filed["inc_sha"][k]
        sha = sha.decode() if isinstance(sha, bytes) else str(sha)
        if inc.size != int(filed["inc_size"][k]) or C4S.pool_sha(inc) != sha:
            raise SystemExit(f"{q}: the frozen pool is not the filed one")
        exp = filed["nodes"][filed["ptr"][k]:filed["ptr"][k + 1]].astype(np.int64)
        if exp.size != int(filed["count"][k]) or exp.size != int(filed["fill_k"][k]) + int(filed["x_count"][k]):
            raise SystemExit(f"{q}: the expansion is not the filed length")
        if np.unique(exp).size != exp.size or np.isin(exp, bs).any():
            raise SystemExit(f"{q}: the expansion repeats a node or holds a node of base and seeds")
        pool, added = m3c.build_pool(base, seeds, exp)
        if pool.size != bs.size + exp.size:
            raise SystemExit(f"{q}: the P1g-b pool has {pool.size} nodes, not {bs.size + exp.size}")
        changed += int(not np.array_equal(pool, inc))
        short += int(filed["fill_k"][k] < filed["fill_want"][k])
        prep.pools[j] = pool
        prep.seeds_added[j] = added
    return {"questions": len(prep.pop.ids), "changed": changed, "short": short}


def load_filed(name, carve):
    filed, ent, msha = C4S.load_filed(name, carve, POOLS / name)
    if not str(ent.get("construction", "")).startswith("P1G:"):
        raise SystemExit(f"{POOLS / name}: the pools file is {ent.get('construction')}'s, not P1g-b's")
    return filed, ent, msha


# ── copy-bases ───────────────────────────────────────────────────────────────


def copy_bases():
    saved = C4S.CACHE
    C4S.CACHE = CACHE
    try:
        return C4S.copy_bases()
    finally:
        C4S.CACHE = saved


# ── kt (the host) ────────────────────────────────────────────────────────────


def question_kt(orders, bs_j, f_bs, fill_k, f_fill, f_exp, rows, entity):
    """One question: fill_f recomputed (KIND_T's first fill_k nodes outside base and seeds), the three checks (fill,
    base and seeds, the cache's rows are the pool) and P1c3's columns with fill_f as the fill."""
    bs_j = np.asarray(bs_j, np.int64)
    fill = C1.outside(np.asarray(orders["KIND_T"], np.int64), bs_j)[:int(fill_k)]
    rows = np.asarray(rows, np.int64)
    pool = np.union1d(bs_j, np.asarray(f_exp, np.int64))
    ok = {"fill_filed": bool(np.array_equal(fill, np.asarray(f_fill, np.int64))),
          "bs_filed": bool(np.array_equal(bs_j, np.asarray(f_bs, np.int64))),
          "rows_are_pool": bool(rows.size == pool.size and np.unique(rows).size == rows.size
                                and np.array_equal(np.sort(rows), pool))}
    return (KT3.columns(orders, rows, fill, entity) if all(ok.values()) else None), ok


def side_rrf5(order, rows, nmax=KT3.NMAX):
    """The side column (rows x 1, float32): each row's place in P1d's RRF5 order, scaled as P1c3's columns
    (log1p(place) / log1p(NMAX), 1.0 when the row is not in the order). Written beside the KT input for ledger item L2
    (RRF5's position as a ninth column); no P1g-b fit reads it."""
    o = np.asarray(order, np.int64)
    if o.size > nmax:
        raise SystemExit(f"the RRF5 order holds {o.size} nodes, more than {nmax}")
    p_ = KT3.place(o, np.asarray(rows, np.int64))
    return np.where(p_ >= 0, np.log1p(np.maximum(p_, 0)) / np.log1p(float(nmax)), 1.0).astype(np.float32)[:, None]


def kt_stage(name, threads, limit=0, carve=None):
    wanted = set(carves_for(name))
    if carve is not None:
        if carve not in wanted:
            raise SystemExit(f"{name}: {carve} is not one of P1c2's carves {sorted(wanted)}")
        wanted = {carve}
    import p1e_anchor_kind as K
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    if int(K.NMAX) != KT3.NMAX:
        raise SystemExit(f"P1d's cut is {K.NMAX}, not {KT3.NMAX}")
    op = S4.Opened()
    ds = op.canonical.Dataset(name, root=str(op.served))
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    del ds
    cache = {}

    def sink(name_, carve_, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        _stores, unions = G.open_unions(cache, op, name, union)
        n_st = int(unions["st"].deg.size)
        filed, ent, manifest_sha = load_filed(name, carve_)
        ids = [str(q) for q in pop.ids]
        if [str(q) for q in filed["ids"].tolist()] != ids:
            raise SystemExit(f"{name}/{carve_}: P1g-b's pools file holds other questions than the carve")
        c_ids, c_n, c_node, part_shas = KT3.cache_rows(CACHE, name, carve_)
        at = {q: j for j, q in enumerate(ids)}
        if any(q not in at for q in c_ids):
            raise SystemExit(f"{name}/{carve_}: a question of the cache is not the carve's")
        nc = min(limit, len(c_ids)) if limit else len(c_ids)
        js_all = np.asarray([at[q] for q in c_ids[:nc]], np.int64)
        off = np.concatenate([[0], np.cumsum(c_n)])
        kt = np.empty((int(off[nc]), KT3.NKT), np.float16)
        r5 = np.empty((int(off[nc]), 1), np.float16)
        ptr, bptr = filed["ptr"], filed["bs_ptr"]
        checks = {"fill_filed": 0, "bs_filed": 0, "rows_are_pool": 0}
        order_ = np.argsort(js_all, kind="stable")
        for lo in range(0, nc, S4.BATCH):
            cs = order_[lo:lo + S4.BATCH]
            js = [int(js_all[c]) for c in cs]
            srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
            steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
            for i, (c, j) in enumerate(zip(cs, js)):
                src, _nn, (L, Ls) = srcs[i]
                knd = K.kind(Ls, prep.seeds[j], nonly)
                out, p1d = K.p1e_orders(src, steered[i], bool(D.routes(L, Ls)), knd, n_st, constant)
                orders = {"KIND_T": out["KIND_T"], "A1st": p1d["A1st"], "A1all": p1d["A1all"], "Kst": p1d["Kst"],
                          "KTst": out["KTst"], "KTall": out["KTall"]}
                fk = int(filed["fill_k"][j])
                exp = filed["nodes"][ptr[j]:ptr[j + 1]]
                x, ok = question_kt(orders, np.unique(np.asarray(bs[j], np.int64)),
                                    filed["bs_nodes"][bptr[j]:bptr[j + 1]], fk, exp[:fk], exp,
                                    c_node[off[c]:off[c + 1]], knd == "entity")
                if x is None:
                    raise SystemExit(f"{name}/{carve_} question {j} ({ids[j]}): {ok}")
                for k in checks:
                    checks[k] += 1
                kt[off[c]:off[c + 1]] = x
                r5[off[c]:off[c + 1]] = side_rrf5(out["RRF5"], c_node[off[c]:off[c + 1]])
        if limit:
            log(f"p1gb kt {name}/{carve_} development run: {nc} questions, the orders, the checks {sorted(checks)} and "
                f"the columns ran ({time.time() - t:.0f}s); nothing written")
            return
        d = KT / name / carve_
        d.mkdir(parents=True, exist_ok=True)
        arrays = {"kt": kt, "n": c_n.astype(np.int32), "side_rrf5": r5}
        shas = {}
        for k, a in arrays.items():
            tmp = d / f"{k}.tmp.npy"
            np.save(tmp, np.ascontiguousarray(a))
            os.replace(tmp, d / f"{k}.npy")
            shas[k] = sha_file(d / f"{k}.npy")
        write_json(d / "ids.json", c_ids)
        rec = {"declared_in": DECLARED, "columns_declared_in": KT3.DECLARED,
               "side": {"side_rrf5": "each row's place in P1d's RRF5 order, scaled as P1c3's columns; for ledger item "
                                     "L2, read by no P1g-b fit"}, "dataset": name, "carve": carve_,
               "questions": nc, "rows": int(kt.shape[0]), "columns": list(KT3.COLS), "nmax": KT3.NMAX,
               "dtype": "float16", "arrays": {k: {"sha256": v, "shape": list(arrays[k].shape)} for k, v in shas.items()},
               "ids_sha256": sha_file(d / "ids.json"), "cache_root": CACHE.relative_to(ROOT).as_posix(),
               "cache_parts": part_shas, "pools_manifest_sha256": manifest_sha, "pools_file_sha256": ent.get("sha256"),
               "construction": ent.get("construction"), "checks": checks, "script_sha256": script_sha256(),
               "p1c3_script_sha256": KT3.script_sha256(), "p1e_script_sha256": K.script_sha256(),
               "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "threads": threads,
               "seconds": round(time.time() - t, 1), "utc": utc()}
        write_json(d / "record.json", rec)
        log(f"p1gb kt {name}/{carve_}: {nc} questions, {kt.shape[0]} rows, every check held ({rec['seconds']:.0f}s)")
        del kt, c_node
        gc.collect()

    E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


# ── the gate ─────────────────────────────────────────────────────────────────


def rows_check(filed, ids1, n1, ids4, n4):
    """Step 1's rows are the filed |I_q| and P1g-b's are the filed pool sizes, question by question."""
    at = {str(q): k for k, q in enumerate(filed["ids"].tolist())}
    if ids1 != ids4 or any(q not in at for q in ids4):
        return {"verdict": "FAIL: the cache's questions are not step 1's or not the pools file's"}
    k = np.asarray([at[q] for q in ids4], dtype=np.int64)
    ok1 = bool(np.array_equal(n1, filed["inc_size"][k].astype(np.int64)))
    ok4 = bool(np.array_equal(n4, expected_rows(filed)[k]))
    return {"step1_rows_are_frozen_sizes": ok1, "rows_are_p1gb_sizes": ok4, "verdict": "PASS" if ok1 and ok4 else "FAIL"}


def kt_check(ds, cv, root=KT, cache=CACHE):
    rp = Path(root) / ds / cv / "record.json"
    if not rp.exists():
        return "KT record missing"
    r = json.loads(rp.read_text(encoding="utf-8"))
    parts = {p.name: sha_file(p / "record.json") for p in C4S.parts_of(cache, ds, cv)[0]}
    if r.get("declared_in") != DECLARED or r.get("cache_parts") != parts or r.get("cache_root") != Path(cache).relative_to(ROOT).as_posix():
        return "the KT record is not on P1g-b's cache parts and root"
    return None


def gate():
    t0 = time.time()
    rec = {"declared_in": DECLARED, "manifests": {}, "rows": {}, "looks": {}, "kt": {}, "squad": {},
           "script_sha256": script_sha256()}
    fails = []
    constructions = set()
    for ds, cv in SCREEN:
        mp = POOLS / ds / "pools.json"
        rec["manifests"].setdefault(ds, sha_file(mp))
        filed, ent, msha = load_filed(ds, cv)
        if msha != rec["manifests"][ds]:
            raise SystemExit(f"{mp}: changed while the gate ran")
        constructions.add(ent["construction"])
        bad = C1.look_pools_records(ds, cv, ent, LOOK)
        rec["looks"][f"{ds}={cv}"] = bad or "PASS"
        fails += [f"{ds}/{cv}: {b}" for b in bad]
        ids1, n1, p1, r1 = C4S.carve_rows(C4S.STEP1_CACHE, ds, cv)
        ids4, n4, p4, r4 = C4S.carve_rows(CACHE, ds, cv)
        row = {"questions": len(ids4), "short": ent["short"], "changed": ent["changed"],
               **rows_check(filed, ids1, n1, ids4, n4)}
        rec["rows"][f"{ds}={cv}"] = row
        if row["verdict"] != "PASS":
            fails.append(f"{ds}/{cv}: rows {row['verdict']}")
        k = kt_check(ds, cv)
        rec["kt"][f"{ds}={cv}"] = k or "PASS"
        if k:
            fails.append(f"{ds}/{cv}: {k}")
        if ds == "squad":
            f2, _e2, _m2 = C4S.load_filed(ds, cv, PC2.POOLS / ds)
            same = all(np.array_equal(a, b) for a, b in zip(pools_of(filed), p1c2_pools_of(f2)))
            if same:
                _i2, _n2, p2, r2 = C4S.carve_rows(PC2.CACHE, ds, cv)
                sq = C1.squad_identity(p2, p4, r2, r4)
                rec["squad"][cv] = sq
                if sq["verdict"] != "PASS":
                    fails.append(f"squad/{cv}: identity with P1c2's caches {sq['verdict']} {sq['arrays']}")
            else:
                rec["squad"][cv] = "the chosen rule changes squad's pools; no identity check"
    if len(constructions) != 1:
        fails.append(f"the carves' pools come from {sorted(constructions)}, not one rule")
    rec["construction"] = sorted(constructions)
    rec["failures"] = fails
    rec["verdict"] = "PASS" if not fails else "FAIL"
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = utc()
    write_json(GATE, rec)
    log(f"p1gb gate {rec['verdict']}: {fails or 'every row count as filed, every look on the filed pools, the KT input on the caches'}")
    return 0 if not fails else 1


def p1c2_pools_of(f2):
    """P1c2's pools from its file: base and seeds and the filed expansion."""
    return [np.union1d(f2["bs_nodes"][f2["bs_ptr"][j]:f2["bs_ptr"][j + 1]].astype(np.int64),
                       f2["nodes"][f2["ptr"][j]:f2["ptr"][j + 1]].astype(np.int64)) for j in range(f2["ids"].size)]


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    from types import SimpleNamespace
    t0 = time.time()
    assert (POOLS, LOOK, CACHE, KT) != (PC2.POOLS, PC2.LOOK, PC2.CACHE, KT3.KT) and len(SCREEN) == 11
    # 1. the expansion: fill_f then X's nodes outside it; the fill is cut at the order's end
    full = np.asarray([5, 3, 7, 6, 4, 9])
    e, k, want, xc = expansion(full, 4, 1.0, np.asarray([2, 3]))
    assert e.tolist() == [5, 3, 7, 6, 2] and (k, want, xc) == (4, 4, 1)
    e, k, want, xc = expansion(full, 4, 0.5, np.asarray([2, 3, 8]))
    assert e.tolist() == [5, 3, 2, 8] and (k, want, xc) == (2, 2, 2)
    e, k, want, xc = expansion(full, 4, 4.0, np.asarray([9, 1]))
    assert e.tolist() == [5, 3, 7, 6, 4, 9, 1] and (k, want, xc) == (6, 16, 1)
    e, k, want, xc = expansion(np.empty(0, np.int64), 0, 2.0, np.empty(0, np.int64))
    assert e.size == 0 and (k, want, xc) == (0, 0, 0)
    # 2. the file and its pools; rows are |B| + the expansion
    bs = [np.asarray([2, 9]), np.asarray([5])]
    inc = [np.asarray([2, 9, 10, 11, 12]), np.asarray([5, 30, 31])]
    exps = [np.asarray([1, 3, 4]), np.asarray([6, 7, 8, 11])]
    arr = pool_arrays(["a", "b"], bs, inc, exps, [3, 2], [3, 4], [2, 4], [1, 0])
    assert expected_rows(arr).tolist() == [5, 5] and arr["nodes"].tolist() == [1, 3, 4, 6, 7, 8, 11]
    assert [p.tolist() for p in pools_of(arr)] == [[1, 2, 3, 4, 9], [5, 6, 7, 8, 11]]
    # 3. the replacement: every pool is base, seeds and the filed expansion, of any length; a bad file stops it
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
    stt = replace_pools(prep, arr, FakeM3C, None, cfg, None, None)
    assert stt == {"questions": 2, "changed": 2, "short": 1}, stt
    assert prep.pools[0].tolist() == [1, 2, 3, 4, 9] and prep.pools[1].tolist() == [5, 6, 7, 8, 11]
    for key, val in (("count", np.asarray([3, 5], np.int32)), ("x_count", np.asarray([0, 0], np.int32)),
                     ("inc_size", np.asarray([5, 4], np.int32))):
        bad = dict(arr)
        bad[key] = val
        prep.pools = [inc[0].copy(), inc[1].copy()]
        try:
            replace_pools(prep, bad, FakeM3C, None, cfg, None, None)
            raise AssertionError(f"a pools file with a bad {key} passed")
        except SystemExit:
            pass
    bad = pool_arrays(["a", "b"], bs, inc, [np.asarray([1, 9]), exps[1]], [3, 2], [3, 4], [1, 4], [1, 0])
    prep.pools = [inc[0].copy(), inc[1].copy()]
    try:
        replace_pools(prep, bad, FakeM3C, None, cfg, None, None)
        raise AssertionError("an expansion holding a base node passed")
    except SystemExit:
        pass
    # 4. the KT columns: fill_f recomputed and checked, P1c3's columns with fill_f as the fill; bad rows refused
    orders = {"KIND_T": np.asarray([2, 1, 3, 4, 6]), "A1st": np.asarray([], np.int64), "A1all": np.asarray([3]),
              "Kst": np.asarray([4]), "KTst": np.asarray([1]), "KTall": np.asarray([1, 3])}
    rows = np.asarray([9, 1, 3, 2, 4])
    x, ok = question_kt(orders, np.asarray([2, 9]), np.asarray([2, 9]), 2, np.asarray([1, 3]), exps[0], rows, True)
    assert all(ok.values()) and x is not None
    assert x[:, 6].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0] and x[:, 7].tolist() == [1.0] * 5
    assert np.allclose(x, KT3.columns(orders, rows, np.asarray([1, 3]), True))
    x, ok = question_kt(orders, np.asarray([2, 9]), np.asarray([2, 9]), 2, np.asarray([1, 3]), exps[0],
                        np.asarray([9, 1, 3, 2]), False)
    assert x is None and not ok["rows_are_pool"]
    x, ok = question_kt(orders, np.asarray([2, 9]), np.asarray([2, 9]), 3, np.asarray([1, 3]), exps[0], rows, False)
    assert x is None and not ok["fill_filed"]
    # the side column: RRF5's place, scaled as P1c3's columns, 1.0 off the order; an order over NMAX refused
    r5 = side_rrf5(np.asarray([4, 9, 1]), rows)
    assert r5.shape == (5, 1) and r5[:, 0].tolist() == [np.float32(np.log1p(1) / np.log1p(KT3.NMAX)),
                                                         np.float32(np.log1p(2) / np.log1p(KT3.NMAX)), 1.0, 1.0, 0.0]
    try:
        side_rrf5(np.arange(KT3.NMAX + 1), rows)
        raise AssertionError("an RRF5 order over NMAX passed")
    except SystemExit:
        pass
    # 5. the rows check and the chosen rule
    ok = rows_check(arr, ["a", "b"], np.asarray([5, 3]), ["a", "b"], np.asarray([5, 5]))
    assert ok["verdict"] == "PASS"
    assert rows_check(arr, ["a", "b"], np.asarray([5, 3]), ["a", "b"], np.asarray([5, 4]))["verdict"] == "FAIL"
    assert extra_applies(False, 0) and extra_applies(True, 3) and not extra_applies(True, 0)
    e, k, want, xc = expansion(full, 0, 1.5, np.asarray([2, 3]) if extra_applies(False, 0) else PG.EMPTY)
    assert e.tolist() == [2, 3] and (k, want, xc) == (0, 0, 2)          # ungated: X on a question with no fill
    e, k, want, xc = expansion(full, 0, 1.5, np.asarray([2, 3]) if extra_applies(True, 0) else PG.EMPTY)
    assert e.size == 0 and (k, want, xc) == (0, 0, 0)                   # gated: base and seeds alone
    sig = {"script_sha256": PG2.script_sha256(), "p1g_script_sha256": PG.script_sha256()}
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        c = PG.candidates()[1]
        write_json(T / "v.json", {**sig, "verdict": "CEILING_MET", "chosen": c[0]})
        assert chosen_rule(T / "v.json")[:4] == (*c, False)
        write_json(T / "v.json", {**sig, "verdict": "TARGET_MET", "chosen": PG2.GATE + c[0]})
        assert chosen_rule(T / "v.json")[:4] == (PG2.GATE + c[0], c[1], c[2], True)
        for v in ({**sig, "verdict": "NONE_MEETS", "chosen": c[0]},
                  {**sig, "script_sha256": "0" * 64, "verdict": "TARGET_MET", "chosen": c[0]},
                  {**sig, "p1g_script_sha256": "0" * 64, "verdict": "TARGET_MET", "chosen": c[0]},
                  {"script_sha256": PG.script_sha256(), "verdict": "TARGET_MET", "chosen": c[0]},
                  {**sig, "verdict": "TARGET_MET", "chosen": "f9+none"},
                  {**sig, "verdict": "TARGET_MET", "chosen": PG2.GATE + "f1+none"}):
            write_json(T / "v.json", v)
            try:
                chosen_rule(T / "v.json")
                raise AssertionError(f"{v} passed")
            except SystemExit:
                pass
    log(f"p1gb selftest: the expansion (fill_f, then X outside it, cut at the order's end); the pools file and its "
        f"rows; the replacement of any length and its refusals; the KT columns with fill_f; the rows check; P1g2's "
        f"gate; the chosen rule (P1g2's, gated or not) and its refusals ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("pools", "copy-bases", "kt", "gate"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--carve")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd in ("pools", "kt"):
        if a.dataset not in SIX:
            ap.error(f"{a.cmd} DATASET: one of " + ", ".join(SIX))
        if a.host:                          # P1e's host mode: the frozen package on the mirror
            import p1e_anchor_kind as K
            K.G.on_the_mirror()
        if a.cmd == "pools":
            return pools_stage(a.dataset, a.threads, a.limit)
        return kt_stage(a.dataset, a.threads, a.limit, a.carve)
    if a.cmd == "copy-bases":
        return copy_bases()
    if a.cmd == "gate":
        return gate()
    ap.error("pools, copy-bases, kt, gate or --selftest")


if __name__ == "__main__":
    sys.exit(main())
