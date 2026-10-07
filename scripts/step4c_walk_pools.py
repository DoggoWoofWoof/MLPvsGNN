"""Step 4c: step 4b's walk pools in the lean MLP, retrained on all six datasets (docs/STEP4C_WALK_POOLS_RETRAINED.md).

    python scripts/step4c_walk_pools.py pools [--datasets metaqa,squad] [--threads 5]      (laptop: numba)
    python scripts/step4c_walk_pools.py copy-bases                                          (host)
    python scripts/step4c_walk_pools.py gate                                                (host)
    python scripts/step4c_walk_pools.py screen-prep --fit J5                                (host)
    python scripts/step4c_walk_pools.py screen-grade [--host]                               (host)
    python scripts/step4c_walk_pools.py check --fit J5 --host                               (host, CPU)
    python scripts/step4c_walk_pools.py grade [--host]                                      (host)

pools         Section 2. Per carve, each question's frozen pool I_q (m3b_compile.prepare, unchanged), its base and seeds,
              and P_F's order: step 4b's RegimeUnion over the frozen regime's families, step 4's kernel, the restart
              value step 4b filed (outputs/step4b/alpha.json, 0.5), eps 1e-7, the first 2,000 nodes. The expansion is
              the order's first B_q nodes, B_q = |I_q| - |base_q and seeds_q|; a question whose walk offers fewer keeps
              them all and is short. Written to outputs/step4c/pools/<dataset>__<carve>.npz (the ids, |I_q|, sha256 of
              I_q, base and seeds, B_q, the expansion), with the manifest pools.json (each file's sha256, the pool
              counts). On the s1sel and s1eval carves, |I_q|, |base and seeds|, B_q, the order's length and every gold's
              rank in the order must equal step 4b's filed arrays, or the stage stops.
copy-bases    step 1's cache bases (basis_2wiki, basis_hotpotqa: npz and record) into outputs/step4c/cache, each npz
              checked against its record's sha256 and each copy against its source.
gate          Section 3's identity gate. squad's four carves: every cache array equals step 1's, sha256 by sha256; the
              six pair's stored reference scores (score2, never an MLP input) must be identical or within 1e-4, and
              their largest difference is filed. Every other carve: the per-question row counts equal step 1's where
              the question is not short, and |base and seeds| plus its expansion where it is; step 1's equal the filed
              |I_q|. Every look shard filed a pools record naming the manifest's file. Writes outputs/step4c/gate.json;
              exits 1 on a failure.
screen-prep   step 1's models.pt of a fit into outputs/step4c/screen/<fit>, checked against step 1's train record.
screen-grade  Section 5, reported and not graded: per fit and eval read, step 1's pick on the P_F read minus the same
              candidate on the frozen read (both pick rules), with step 1's bootstrap and labels. Writes
              outputs/step4c/screen.json and screen.md.
check         lean_gpu's CPU check of a 4c fit (outputs/step4c/fits), unchanged, with lean_mlp3.Carve3's default look
              root bound to outputs/step4c/look: its reads are checked against the looks they were cached from.
grade         Sections 4 and 6: the 4c fits against step 1's on the eval reads under the pick rule step 1's verdict
              adopts, the eleven primary reads, the verdict and the diagnostics. Writes outputs/step4c/grade.json and
              grade.md. It stops unless gate.json is PASS.

Everything but the pools is step 1's: look_x_six's scoring pass (through outputs/mp_unified/look_step4c.py),
lean_cache.py, lean_gpu.py and step1_grade.py's picks, bootstrap and labels, imported unchanged.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # as lean_gpu.py, which the check runs in this process

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "outputs" / "step4c"
POOLS = OUT / "pools"
MANIFEST = POOLS / "pools.json"
LOOK = OUT / "look"
CACHE = OUT / "cache"
FITS_ROOT = OUT / "fits"
SCREEN = OUT / "screen"
STEP1 = ROOT / "outputs" / "step1"
STEP1_CACHE = STEP1 / "cache"
STEP1_FITS = STEP1 / "fits"
STEP1_GRADE = STEP1 / "grade.json"
STEP4B = ROOT / "outputs" / "step4b"
SIX = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
TRAIN = SIX[:5]
ALPHA = 0.5                       # step 4b's chosen restart value (its alpha.json, checked)
BASES = ("2wiki", "hotpotqa")
SCORE2_TOL = 1e-4                 # lean_gpu's check tolerance
STEP4B_KIND = {"s1sel": "select", "s1eval": "eval"}
CARVES_KEY = {"fit": "fit", "s1fit": "fit", "select": "mselect", "s1sel": "dselect", "s1eval": "eval"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def write_text(path, text):
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def carves_of(name):
    """The carves this step looks, caches and reads on a dataset (section 2)."""
    if name == "webqsp":
        return ("s1eval",)
    return ("s1fit" if name == "musique" else "fit", "select", "s1sel", "s1eval")


def pool_sha(pool):
    """sha256 of a pool as sorted int64 node positions (prepare's dtype)."""
    return hashlib.sha256(np.ascontiguousarray(pool, dtype=np.int64).tobytes()).hexdigest()


def flat_of(arrays):
    ptr = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum([a.size for a in arrays], out=ptr[1:])
    flat = np.concatenate(arrays).astype(np.int64) if arrays else np.empty(0, dtype=np.int64)
    return ptr, flat


def expansions(orders, b_q):
    """Each order's first B_q nodes: (ptr, flat nodes, count). count < B_q marks a short question."""
    count = np.asarray([min(int(b), int(o.size)) for o, b in zip(orders, b_q)], dtype=np.int32)
    ptr, flat = flat_of([np.asarray(o[:c], dtype=np.int64) for o, c in zip(orders, count)])
    return ptr, flat, count


def filed_arrays(ids, bs, inc, orders):
    """One carve's pools file: per question |I_q|, |base and seeds|, B_q, the expansion (the order's first B_q nodes)
    and its length, base and seeds, sha256 of I_q, and whether the P_F pool differs from I_q."""
    inc_size = np.asarray([p.size for p in inc], dtype=np.int32)
    bs_size = np.asarray([u.size for u in bs], dtype=np.int32)
    b_q = (inc_size - bs_size).astype(np.int32)
    ptr, flat, count = expansions(orders, b_q)
    bs_ptr, bs_flat = flat_of(bs)
    if max([int(flat.max()) if flat.size else 0, int(bs_flat.max()) if bs_flat.size else 0]) >= 2 ** 31:
        raise SystemExit("a node position does not fit int32")
    changed = np.asarray([not np.array_equal(np.union1d(u, flat[ptr[j]:ptr[j + 1]]), np.asarray(p, dtype=np.int64))
                          for j, (u, p) in enumerate(zip(bs, inc))], dtype=bool)
    return {"ids": np.asarray(list(ids)), "inc_size": inc_size, "bs_size": bs_size, "B_q": b_q, "count": count, "ptr": ptr,
            "nodes": flat.astype(np.int32), "bs_ptr": bs_ptr, "bs_nodes": bs_flat.astype(np.int32),
            "inc_sha": np.asarray([pool_sha(p) for p in inc], dtype="S64"), "changed": changed}


def expected_rows(filed):
    """Each question's pool size on P_F: |I_q| unless short, else |base and seeds| plus its expansion."""
    short = filed["count"] < filed["B_q"]
    return np.where(short, filed["bs_size"].astype(np.int64) + filed["count"], filed["inc_size"].astype(np.int64))


# ── the pool replacement (outputs/mp_unified/look_step4c.py runs it after m3b_compile.prepare) ─────────────────


def load_filed(name, carve, pools_dir=POOLS):
    """One carve's pools file, after its sha256 is checked against the manifest."""
    manifest_path = Path(pools_dir) / "pools.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ent = manifest["files"].get(f"{name}__{carve}")
    if ent is None:
        raise SystemExit(f"{manifest_path}: no pools file for {name}/{carve}")
    p = Path(pools_dir) / ent["file"]
    got = sha_file(p)
    if got != ent["sha256"]:
        raise SystemExit(f"{p}: sha256 {got} is not the manifest's {ent['sha256']}")
    with np.load(p) as z:
        filed = {k: z[k] for k in z.files}
    return filed, ent, sha_file(manifest_path)


def replace_pools(prep, filed, m3c, construction, cfg_h, m3a, m3b_contract):
    """Each prepared question's pool becomes base and seeds and its filed P_F expansion (m3c.build_pool), in place.
    Stops on a missing question, a base or seed set or a frozen pool that is not the filed one, or an expansion that
    repeats a node, holds a node of base and seeds, or is not the filed length."""
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
        if inc.size != int(filed["inc_size"][k]) or pool_sha(inc) != sha:
            raise SystemExit(f"{q}: the frozen pool is not the filed one")
        exp = filed["nodes"][filed["ptr"][k]:filed["ptr"][k + 1]].astype(np.int64)
        b_q = int(filed["B_q"][k])
        if b_q != inc.size - bs.size or exp.size != int(filed["count"][k]) or exp.size > b_q:
            raise SystemExit(f"{q}: the expansion is not the filed length")
        if np.unique(exp).size != exp.size or np.isin(exp, bs).any():
            raise SystemExit(f"{q}: the expansion repeats a node or holds a node of base and seeds")
        pool, added = m3c.build_pool(base, seeds, exp)
        if pool.size != bs.size + exp.size:
            raise SystemExit(f"{q}: the P_F pool has {pool.size} nodes, not {bs.size + exp.size}")
        changed += int(not np.array_equal(pool, inc))
        short += int(exp.size < b_q)
        prep.pools[j] = pool
        prep.seeds_added[j] = added
    return {"questions": len(prep.pop.ids), "changed": changed, "short": short}


# ── pools (laptop) ───────────────────────────────────────────────────────────


def populations(op, S4, ds, name):
    """(carve, Population) for the dataset's carves, in carves_of's order, each checked against carves.json:
    fit and select are m3b_compile.population's (as look_x_six checks them); s1fit is look_x_six's carve_population
    on carves.json's ids; s1sel and s1eval are step 4's (and step 4b's) populations."""
    m3c = op.m3b_compile
    out = []
    positions = None
    for carve in carves_of(name):
        ent = op.carves["per_dataset"][name][CARVES_KEY[carve]]
        if ent.get("name") != carve:
            raise SystemExit(f"{name}: carves.json's {CARVES_KEY[carve]} carve is {ent.get('name')}, not {carve}")
        if carve in ("s1sel", "s1eval"):
            pop, _strata = S4.population(op, ds, name, STEP4B_KIND[carve])
        else:
            if positions is None:
                positions = op.m3a.node_position_map(ds)
            if carve in ("fit", "select"):
                pop = m3c.population(ds, name, carve, op.cfg, op.cfg_h, op.m3a, positions)
            else:
                ids = list(ent["ids"])
                if S4.digest(ids) != ent["sha256"] or len(ids) != ent["n"]:
                    raise SystemExit(f"{name}/{carve}: the ids do not hash to the record's sha256")
                row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
                keep = set(ids)
                by_id = {r["query_id"]: r for r in ds.queries("train") if r["query_id"] in keep}
                golds = op.m3a.resolve_gold([by_id[q] for q in ids], positions, name)
                zero = np.asarray([g.size == 0 for g in golds], dtype=bool)
                kept = [q for q, z in zip(ids, zero) if not z]
                idx = np.asarray([row_of[q] for q in ids], dtype=np.int64)
                pop = m3c.Population(name, carve, kept, idx[~zero], [g for g, z in zip(golds, zero) if not z], len(ids),
                                     int(zero.sum()), S4.digest(kept))
        if pop.digest != ent["sha256"] or len(pop.ids) != int(ent["n"]):
            raise SystemExit(f"{name}/{carve}: {len(pop.ids)} questions hashing to {pop.digest[:12]}, carves.json "
                             f"{ent['n']} hashing to {ent['sha256'][:12]}")
        pop.kind = carve
        out.append((carve, pop))
    del positions
    gc.collect()
    return out


def step1_rows(name, carve, ids, inc_size):
    """Each of step 1's host cache parts (its record.json, fetched to the laptop) holds as many rows as the frozen pools
    built here for its questions, or the stage stops: the laptop's frozen pools are the ones the host's looks build."""
    paths = sorted((STEP1_CACHE / name / carve).glob("part_*of*/record.json"))
    if not paths:
        return None
    at = {q: k for k, q in enumerate(ids)}
    seen, bad = 0, []
    for p in paths:
        r = json.loads(p.read_text(encoding="utf-8"))
        if any(q not in at for q in r["ids"]):
            raise SystemExit(f"{p}: a question is not in the carve")
        k = np.asarray([at[q] for q in r["ids"]], dtype=np.int64)
        seen += k.size
        if int(inc_size[k].sum()) != int(r["rows"]):
            bad.append(f"{p.parent.name}: {int(inc_size[k].sum())} rows here, {r['rows']} in step 1's cache")
    if bad or seen != len(ids):
        raise SystemExit(f"{name}/{carve}: the frozen pools are not step 1's: {bad or f'{seen} of {len(ids)} questions'}")
    return {"parts": len(paths), "questions": seen, "rows": int(inc_size.sum()), "equal": True}


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
    host_rows = step1_rows(name, carve, list(pop.ids), inc_size)
    orders = []
    for lo in range(0, nq, S4.BATCH):
        hi = min(nq, lo + S4.BATCH)
        o, _av, _wk, _tc = union.ppr(seeds[lo:hi], bs[lo:hi], ALPHA, S4.EPS, S4.BMAX, threads)
        orders += o
    arrays = filed_arrays(pop.ids, bs, inc, orders)
    ptr, flat, count, b_q, changed = arrays["ptr"], arrays["nodes"], arrays["count"], arrays["B_q"], arrays["changed"]
    golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds]
    step4b = None
    if carve in STEP4B_KIND:
        path = STEP4B / STEP4B_KIND[carve] / f"{name}.npz"
        with np.load(path) as z:
            ref = {k: z[k] for k in ("ids", "inc_size", "bs_size", "B_q", "P_count_0.5", "gold_rank_P_0.5")}
        ranks = np.concatenate([S4.ranks_in(o, g) for o, g in zip(orders, golds)])
        same = {"ids": ref["ids"].tolist() == list(pop.ids),
                "inc_size": np.array_equal(ref["inc_size"], arrays["inc_size"]),
                "bs_size": np.array_equal(ref["bs_size"], arrays["bs_size"]), "B_q": np.array_equal(ref["B_q"], b_q),
                "order_length": np.array_equal(ref["P_count_0.5"], np.asarray([o.size for o in orders], dtype=np.int32)),
                "gold_ranks": np.array_equal(ref["gold_rank_P_0.5"], ranks)}
        if not all(same.values()):
            raise SystemExit(f"{name}/{carve}: not step 4b's filed arrays: {same}")
        step4b = {"file": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(path), "equal": same}
    all_i, all_p, rec_i, rec_p = (np.zeros(nq) for _ in range(4))
    for j in range(nq):
        pf = np.union1d(bs[j], flat[ptr[j]:ptr[j + 1]].astype(np.int64))
        gi, gp = np.isin(golds[j], inc[j]), np.isin(golds[j], pf)
        all_i[j], all_p[j], rec_i[j], rec_p[j] = gi.all(), gp.all(), gi.mean(), gp.mean()
    POOLS.mkdir(parents=True, exist_ok=True)
    fname = f"{name}__{carve}.npz"
    tmp = POOLS / f"{name}__{carve}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, POOLS / fname)
    short = int((count < b_q).sum())
    ent = {"file": fname, "sha256": sha_file(POOLS / fname), "dataset": name, "carve": carve, "questions": nq,
           "carve_ids_sha256": pop.digest, "zero_gold_excluded": int(pop.zero_gold_excluded),
           "construction": construction, "walk_families": list(families), "changed": int(changed.sum()), "short": short,
           "B_q": {"mean": round(float(b_q.mean()), 1) if nq else None, "max": int(b_q.max()) if nq else None},
           "pool_mean": round(float(inc_size.mean()), 1) if nq else None,
           "ALL": {"I": round(float(all_i.mean()), 4), "P_F": round(float(all_p.mean()), 4)},
           "recall": {"I": round(float(rec_i.mean()), 4), "P_F": round(float(rec_p.mean()), 4)},
           "step4b": step4b, "step1_cache_rows": host_rows, "seconds": round(time.time() - t0, 1), "utc": utc()}
    log(f"{name}/{carve}: {nq} questions, {ent['changed']} pools changed, {short} short; ALL I {ent['ALL']['I']} "
        f"P_F {ent['ALL']['P_F']}; step 4b {'equal' if step4b else '-'} ({ent['seconds']:.0f}s)")
    return ent


def pools_stage(names, threads):
    import numba
    import step4b_pool_reach as S4B
    S4 = S4B.S4
    alpha_path = STEP4B / "alpha.json"
    alpha_rec = json.loads(alpha_path.read_text(encoding="utf-8"))
    if float(alpha_rec["alpha"]) != ALPHA:
        raise SystemExit(f"{alpha_path}: restart {alpha_rec['alpha']}, the declaration's is {ALPHA}")
    op = S4.Opened()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"files": {}}
    manifest.update({"declared_in": "docs/STEP4C_WALK_POOLS_RETRAINED.md", "alpha": ALPHA, "eps": S4.EPS, "bmax": S4.BMAX,
                     "alpha_file_sha256": sha_file(alpha_path), "carves_sha256": S4.CARVES_SHA256,
                     "frozen_contract": op.frozen_key, "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
                     "script_sha256": script_sha256(), "step4b_script_sha256": S4B.script_sha256(),
                     "step4_script_sha256": S4.script_sha256()})
    numba.set_num_threads(threads)
    for name in names:
        t0 = time.time()
        ds = op.canonical.Dataset(name, root=str(op.served))
        pops = populations(op, S4, ds, name)
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
        t1 = time.time()
        preps = op.m3b_compile.prepare(ds, [p for _c, p in pops], construction, op.cfg_h, stores, op.m3a, op.m3b_contract)
        log(f"{name}: frozen pools built ({time.time() - t1:.0f}s; {regime}, {(construction.get('setting') or {}).get('name')}; "
            f"P_F walks {families or 'nothing'})")
        union = S4B.RegimeUnion(stores, families, int(stores[families[0]].n_nodes) if families else int(ds.n_nodes))
        for (carve, pop), prep in zip(pops, preps):
            manifest["files"][f"{name}__{carve}"] = file_carve(S4, op, name, carve, pop, prep, union, construction,
                                                               families, threads)
            manifest["utc"] = utc()
            write_json(MANIFEST, manifest)
        del preps, pops, stores, union
        gc.collect()
        log(f"{name}: filed ({time.time() - t0:.0f}s)")
    want = [f"{n}__{c}" for n in SIX for c in carves_of(n)]
    have = [k for k in want if k in manifest["files"]]
    log(f"manifest: {len(have)} of {len(want)} carves filed; {MANIFEST}")
    return 0


# ── host stages ──────────────────────────────────────────────────────────────


def copy_bases():
    CACHE.mkdir(parents=True, exist_ok=True)
    rec = {"source": str(STEP1_CACHE.relative_to(ROOT)).replace("\\", "/"), "files": {}, "script_sha256": script_sha256()}
    for g in BASES:
        npz, js = STEP1_CACHE / f"basis_{g}.npz", STEP1_CACHE / f"basis_{g}.json"
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


def parts_of(root, ds, carve):
    """A carve's cache parts in order with their records (lean_cache.part_dirs' tiling rule)."""
    d = Path(root) / ds / carve
    parts = sorted(d.glob("part_*of*"), key=lambda p: int(p.name.split("_")[1].split("of")[0]))
    if not parts:
        raise SystemExit(f"{d}: no cache part")
    nparts = {int(p.name.split("of")[1]) for p in parts}
    if len(nparts) != 1 or len(parts) != nparts.pop():
        raise SystemExit(f"{d}: parts {[p.name for p in parts]} do not make one whole split")
    recs = [json.loads((p / "record.json").read_text(encoding="utf-8")) for p in parts]
    q = 0
    for r in recs:
        if r["query_range"][0] != q:
            raise SystemExit(f"{d}: the parts do not tile the carve's queries")
        q = r["query_range"][1]
    if q != recs[0]["look"]["carve_queries"]:
        raise SystemExit(f"{d}: the parts hold {q} of {recs[0]['look']['carve_queries']} queries")
    return parts, recs


def carve_rows(root, ds, carve):
    parts, recs = parts_of(root, ds, carve)
    ids = [q for r in recs for q in r["ids"]]
    n = np.concatenate([np.load(p / "n.npy").astype(np.int64) for p in parts])
    return ids, n, parts, recs


def look_pools_records(ds, carve, ent):
    """Every look shard of the carve filed a pools record naming the manifest's file (by its sha256; the manifest
    grows as datasets are filed, so a look may have run under an earlier copy of it); returns problems."""
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
        if r.get("changed") != ent["changed"] or r.get("short") != ent["short"]:
            bad.append(f"pools{tag}.json counts {r.get('changed')}/{r.get('short')} changed/short, the manifest "
                       f"{ent['changed']}/{ent['short']}")
    return bad


def gate():
    t0 = time.time()
    manifest_sha = sha_file(MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rec = {"declared_in": "docs/STEP4C_WALK_POOLS_RETRAINED.md", "manifest_sha256": manifest_sha, "squad": {},
           "rows": {}, "looks": {}, "script_sha256": script_sha256()}
    fails = []
    for ds in SIX:
        for cv in carves_of(ds):
            filed, ent, msha = load_filed(ds, cv)
            if msha != manifest_sha:
                raise SystemExit("the manifest changed while the gate ran")
            bad = look_pools_records(ds, cv, ent)
            rec["looks"][f"{ds}={cv}"] = bad or "PASS"
            fails += [f"{ds}/{cv}: {b}" for b in bad]
            ids1, n1, p1, r1 = carve_rows(STEP1_CACHE, ds, cv)
            ids4, n4, p4, r4 = carve_rows(CACHE, ds, cv)
            want = expected_rows(filed)
            at = {str(q): k for k, q in enumerate(filed["ids"].tolist())}
            row = {"questions": len(ids4), "short": ent["short"], "changed": ent["changed"]}
            if ids1 != ids4 or any(q not in at for q in ids4):
                row["verdict"] = "FAIL: the cache's questions are not step 1's or not the pools file's"
            else:
                k = np.asarray([at[q] for q in ids4], dtype=np.int64)
                ok1 = np.array_equal(n1, filed["inc_size"][k].astype(np.int64))
                ok4 = np.array_equal(n4, want[k])
                row["step1_rows_are_frozen_sizes"] = ok1
                row["rows_are_pf_sizes"] = ok4
                row["verdict"] = "PASS" if ok1 and ok4 else "FAIL"
            rec["rows"][f"{ds}={cv}"] = row
            if row["verdict"] != "PASS":
                fails.append(f"{ds}/{cv}: rows {row['verdict']}")
            if ds != "squad":
                continue
            sq = {"parts": [len(p1), len(p4)], "arrays": {}, "score2_max_abs_diff": 0.0}
            if len(p1) != len(p4) or [r["query_range"] for r in r1] != [r["query_range"] for r in r4]:
                sq["verdict"] = "FAIL: not step 1's parts"
            else:
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
                        d = np.abs(np.load(a / "score2.npy").astype(np.float64) - np.load(b / "score2.npy").astype(np.float64))
                        sq["score2_max_abs_diff"] = max(sq["score2_max_abs_diff"], float(d.max()) if d.size else 0.0)
                bad_arrays = bool(sq["arrays"])
                sq["verdict"] = ("FAIL" if bad_arrays or sq["score2_max_abs_diff"] > SCORE2_TOL else "PASS")
            rec["squad"][cv] = sq
            if sq["verdict"] != "PASS":
                fails.append(f"squad/{cv}: identity {sq['verdict']} {sq['arrays']} score2 {sq['score2_max_abs_diff']}")
    rec["failures"] = fails
    rec["verdict"] = "PASS" if not fails else "FAIL"
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = utc()
    write_json(OUT / "gate.json", rec)
    log(f"gate {rec['verdict']}: {fails or 'squad identical, every row count as filed, every look on the filed pools'}")
    return 0 if not fails else 1


def screen_prep(fit):
    src = STEP1_FITS / fit / "models.pt"
    train = json.loads((STEP1_FITS / fit / "train.json").read_text(encoding="utf-8"))
    got = sha_file(src)
    if got != train["models_sha256"]:
        raise SystemExit(f"{src}: sha256 is not step 1's train record's")
    dst = SCREEN / fit / "models.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    if sha_file(dst) != got:
        raise SystemExit(f"{dst}: the copy is not its source")
    write_json(SCREEN / fit / "screen_prep.json", {"source": str(src.relative_to(ROOT)).replace("\\", "/"),
                                                     "models_sha256": got, "script_sha256": script_sha256(), "utc": utc()})
    log(f"{fit}: step 1's models copied ({got[:12]})")
    return 0


def check(fit, host):
    sys.path.insert(0, str(ROOT / "outputs" / "mp_unified"))
    import lean_gpu as LG
    init = LG.L3.Carve3.__init__
    d = init.__defaults__
    if d is None or len(d) != 4 or Path(d[3]) != Path(LG.LM.LOOK):
        raise SystemExit("lean_mlp3.Carve3's defaults are not (store, nodes, limit, root=lean_mlp.LOOK)")
    init.__defaults__ = d[:3] + (LOOK,)
    try:
        return LG.main(["check", "--fit", fit, "--cache-root", str(CACHE), "--out-root", str(FITS_ROOT)]
                       + (["--host"] if host else []))
    finally:
        init.__defaults__ = d


# ── grading ──────────────────────────────────────────────────────────────────


def g1():
    import step1_grade as G1
    return G1


def pick_rule():
    """'D' (s1sel) when step 1's verdict is ADOPT, else 'M' (select); None while step 1's grade is not filed."""
    if not STEP1_GRADE.exists():
        return None, None
    rec = json.loads(STEP1_GRADE.read_text(encoding="utf-8"))
    return ("D" if rec["verdict"] == "ADOPT" else "M"), rec["verdict"]


RULE_CARVE = {"M": "select", "D": "s1sel"}


def compare(G1, a, b):
    """Paired a - b over questions (step 1's bootstrap) and the R@5 label; a and b are (Q, 3) metric rows."""
    d = G1.boot_diff(a, b)
    return {nm: {"mean": c[0], "ci": c[1]} for nm, c in zip(G1.METRICS, d)}, G1.label_of(False, d[0][1])


def means(m):
    return [round(float(x), 6) for x in m.mean(0)] if m.shape[0] else [None, None, None]


class ScreenRead:
    """Step 1's models read on the P_F s1eval caches (outputs/step4c/screen/<fit>), after the gates."""

    def __init__(self, G1, fit1):
        name = fit1.name
        fdir = SCREEN / name
        self.read = json.loads((fdir / "read.json").read_text(encoding="utf-8"))
        cands = G1.candidate_names()
        if self.read["models_sha256"] != fit1.train["models_sha256"] or sha_file(fdir / "models.pt") != fit1.train["models_sha256"]:
            raise SystemExit(f"{name}: the screen's models are not step 1's")
        if self.read["candidates"] != cands:
            raise SystemExit(f"{name}: the screen read lists other candidates")
        want = [f"{d}=s1eval" for d in G1.EVAL_ORDER]
        if sorted(self.read["carves"]) != sorted(want):
            raise SystemExit(f"{name}: the screen reads {sorted(self.read['carves'])}, declared {sorted(want)}")
        self.m, self.refs = {}, {}
        for key in want:
            ds = key.split("=")[0]
            ent = self.read["carves"][key]
            p = fdir / "reads" / ent["file"]
            if sha_file(p) != ent["sha256"]:
                raise SystemExit(f"{p}: sha256 is not the read record's")
            z = np.load(p)
            ids = [str(x) for x in z["ids"]]
            gt = z["gold_total"].astype(np.int64)
            if ids != fit1.ids[(ds, "s1eval")] or not np.array_equal(gt, fit1.gt[(ds, "s1eval")]):
                raise SystemExit(f"{name}: the screen's {ds} read is not on step 1's questions")
            self.m[ds] = np.stack([G1.metrics_of(z["top"][k], z["hit"][k], gt) for k in range(len(cands))])
            self.refs[ds] = np.stack([G1.metrics_of(z["ref_top"][j], z["ref_hit"][j], gt) for j in range(len(G1.REFS))])


def pool_counts(manifest, ds, cv):
    ent = manifest["files"][f"{ds}__{cv}"]
    return {"changed": ent["changed"], "short": ent["short"], "questions": ent["questions"], "ALL": ent["ALL"],
            "recall": ent["recall"]}


def screen_grade(host):
    t0 = time.time()
    G1 = g1()
    carves = G1.load_carves_record()
    rule, verdict1 = pick_rule()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cands = G1.candidate_names()
    rec = {"declared_in": "docs/STEP4C_WALK_POOLS_RETRAINED.md, section 5 (reported, not graded)",
           "step1_verdict": verdict1, "adopted_rule": rule, "reads": [], "script_sha256": script_sha256()}
    for f in G1.FITS:
        fit1 = G1.Fit(f, STEP1_FITS, carves)
        scr = ScreenRead(G1, fit1)
        picks = {r: G1.first_best(fit1.select_scores(RULE_CARVE[r])) for r in ("M", "D")}
        for ds in G1.EVAL_ORDER:
            ok = fit1.gt[(ds, "s1eval")] > 0
            m1, r1 = fit1.m[(ds, "s1eval")][:, ok], fit1.refs[(ds, "s1eval")][:, ok]
            m4, r4 = scr.m[ds][:, ok], scr.refs[ds][:, ok]
            row = {"fit": f, "dataset": ds, "role": "in-domain" if ds in G1.FITS[f] else "zero-shot",
                   "primary": bool(f == "J5" or ds == f[2:]), "questions": int(ok.sum()),
                   "pools": pool_counts(manifest, ds, "s1eval"), "picks": {}, "refs": {}}
            for r, pk in picks.items():
                diff, lab = compare(G1, m4[pk], m1[pk])
                row["picks"][r] = {"candidate": cands[pk], "frozen": means(m1[pk]), "P_F": means(m4[pk]), "diff": diff,
                                   "label": lab}
            for j, nm in enumerate(G1.REFS):
                diff, lab = compare(G1, r4[j], r1[j])
                row["refs"][nm] = {"frozen": means(r1[j]), "P_F": means(r4[j]), "diff": diff, "label": lab}
            rec["reads"].append(row)
            show = row["picks"][rule or "M"]
            log(f"screen {f} {ds}: {show['candidate']} R@5 frozen {show['frozen'][0]:.4f} P_F {show['P_F'][0]:.4f} "
                f"diff {show['diff']['R@5']['mean']:+.4f} {show['label']}")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = utc()
    write_json(OUT / "screen.json", rec)
    write_text(OUT / "screen.md", render_screen(rec))
    log(f"screen filed: {OUT / 'screen.json'}")
    return 0


def fmt_diff(d):
    return f"{d['mean']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"


def render_screen(rec):
    rule = rec["adopted_rule"]
    lines = ["# Step 4c early read (reported, not graded)", "",
             "Step 1's six fits, unchanged, read on the P_F s1eval caches; per question the pick's R@5 on the P_F read "
             "minus the same candidate on the frozen read.",
             f"Step 1's verdict: {rec['step1_verdict'] or 'not filed yet'}; adopted pick rule: {rule or 'unknown (both shown)'}.", ""]
    for r in ("M", "D"):
        lines += [f"## {r}-pick{' (adopted)' if r == rule else ''}", "",
                  "| fit | dataset | role | primary | pick | R@5 frozen | R@5 P_F | diff R@5 [95%] | label | FC@5 diff | hit@1 diff | pools changed |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rec["reads"]:
            p = row["picks"][r]
            lines.append(f"| {row['fit']} | {row['dataset']} | {row['role']} | {'yes' if row['primary'] else ''} | {p['candidate']} "
                         f"| {p['frozen'][0]:.4f} | {p['P_F'][0]:.4f} | {fmt_diff(p['diff']['R@5'])} | {p['label']} "
                         f"| {p['diff']['FC@5']['mean']:+.4f} | {p['diff']['hit@1']['mean']:+.4f} "
                         f"| {row['pools']['changed']:,} of {row['pools']['questions']:,} |")
        lines.append("")
    lines += ["## References on the same rows (J5's reads)", "", "| dataset | ref | R@5 frozen | R@5 P_F | diff R@5 [95%] |",
              "| --- | --- | --- | --- | --- |"]
    for row in rec["reads"]:
        if row["fit"] != "J5":
            continue
        for nm, v in row["refs"].items():
            lines.append(f"| {row['dataset']} | {nm} | {v['frozen'][0]:.4f} | {v['P_F'][0]:.4f} | {fmt_diff(v['diff']['R@5'])} |")
    return "\n".join(lines) + "\n"


def grade(host):
    t0 = time.time()
    G1 = g1()
    carves = G1.load_carves_record()
    rule, verdict1 = pick_rule()
    if rule is None:
        raise SystemExit(f"{STEP1_GRADE} is not filed: the pick rule is the one step 1's verdict adopts")
    gate_path = OUT / "gate.json"
    if not gate_path.exists() or json.loads(gate_path.read_text(encoding="utf-8")).get("verdict") != "PASS":
        raise SystemExit(f"{gate_path} is not PASS: the identity gate stops the grade")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fits1 = {f: G1.Fit(f, STEP1_FITS, carves) for f in G1.FITS}
    fits4 = {f: G1.Fit(f, FITS_ROOT, carves) for f in G1.FITS}
    strata, placement, freeze = G1.strata_rows(carves, host)
    cands = G1.candidate_names()
    other = "M" if rule == "D" else "D"
    rec = {"declared_in": "docs/STEP4C_WALK_POOLS_RETRAINED.md", "script": "scripts/step4c_walk_pools.py",
           "script_sha256": script_sha256(), "carves_sha256": G1.CARVES_SHA256, "freeze_RECORD_SHA256": freeze,
           "placement": placement, "step1_verdict": verdict1, "rule": rule,
           "gate_sha256": sha_file(gate_path), "manifest_sha256": sha_file(MANIFEST),
           "bootstrap": {"resamples": G1.BOOT, "seed": G1.BOOT_SEED, "interval": [2.5, 97.5], "per": "read and stratum"},
           "fits": {}, "picks": {}, "reads": [],
           "diagnostics": {"strata": [], "other_rule": [], "per_variant_picks": {}, "oracle": {}, "candidates_eval_r5": {},
                           "pools": {f"{ds}={cv}": pool_counts(manifest, ds, cv) for ds in SIX for cv in carves_of(ds)}}}
    for f in G1.FITS:
        f1, f4 = fits1[f], fits4[f]
        for fit in (f1, f4):
            if fit.train["config"] != G1.CONFIG:
                raise SystemExit(f"{f}: not step 1's config")
        rec["fits"][f] = {"step1_models_sha256": f1.train["models_sha256"], "step4c_models_sha256": f4.train["models_sha256"],
                          "step4c_check": {"verdict": f4.check["verdict"],
                                           "max_abs_diff": max(c["max_abs_diff"] for c in f4.check["carves"].values())},
                          "step4c_train_seconds": f4.train.get("seconds"), "dead": [f1.train.get("dead"), f4.train.get("dead")]}
        sc = {(arm, r): fit.select_scores(RULE_CARVE[r]) for arm, fit in (("step1", f1), ("step4c", f4)) for r in ("M", "D")}
        pk = {k: G1.first_best(v) for k, v in sc.items()}
        rec["picks"][f] = {f"{arm}:{r}": {"candidate": cands[pk[(arm, r)]], "score": sc[(arm, r)][pk[(arm, r)]]}
                           for arm, r in sc}
        rec["diagnostics"]["per_variant_picks"][f] = {
            arm: {v: {r: cands[9 * i + G1.first_best(sc[(arm, r)][9 * i:9 * i + 9])] for r in ("M", "D")}
                  for i, v in enumerate(G1.VARIANTS)} for arm in ("step1", "step4c")}
        for ds in G1.EVAL_ORDER:
            m1, refs1, ok1 = G1.eval_rows(f1, ds)
            m4, refs4, ok4 = G1.eval_rows(f4, ds)
            if f1.ids[(ds, "s1eval")] != f4.ids[(ds, "s1eval")] or not np.array_equal(ok1, ok4):
                raise SystemExit(f"{f} {ds}: the two arms' eval reads are not on the same questions")
            a, b = pk[("step4c", rule)], pk[("step1", rule)]
            diff, lab = compare(G1, m4[a], m1[b])
            primary = f == "J5" or ds == f[2:]
            role = "in-domain" if ds in G1.FITS[f] else "zero-shot"
            row = {"fit": f, "dataset": ds, "role": role, "primary": bool(primary), "questions": int(ok1.sum()),
                   "step1_pick": cands[b], "step4c_pick": cands[a], "step1": means(m1[b]), "step4c": means(m4[a]),
                   "diff": diff, "label": lab,
                   "refs": {nm: {"frozen": means(refs1[j]), "P_F": means(refs4[j]), "diff": compare(G1, refs4[j], refs1[j])[0]}
                            for j, nm in enumerate(G1.REFS)},
                   "pools": pool_counts(manifest, ds, "s1eval")}
            rec["reads"].append(row)
            a2, b2 = pk[("step4c", other)], pk[("step1", other)]
            d2, l2 = compare(G1, m4[a2], m1[b2])
            rec["diagnostics"]["other_rule"].append({"fit": f, "dataset": ds, "rule": other, "step1_pick": cands[b2],
                                                     "step4c_pick": cands[a2], "step1": means(m1[b2]),
                                                     "step4c": means(m4[a2]), "diff": d2, "label": l2})
            ev1 = [float(m1[k][:, 0].mean()) for k in range(len(cands))]
            ev4 = [float(m4[k][:, 0].mean()) for k in range(len(cands))]
            rec["diagnostics"]["candidates_eval_r5"].setdefault(f, {})[ds] = {"step1": ev1, "step4c": ev4}
            o1, o4 = G1.first_best(ev1), G1.first_best(ev4)
            rec["diagnostics"]["oracle"].setdefault(f, {})[ds] = {
                "step1": {"candidate": cands[o1], "R@5": ev1[o1]}, "step4c": {"candidate": cands[o4], "R@5": ev4[o4]},
                "note": "the eval-best candidate is an oracle, never a result"}
            if primary and ds in strata:
                ids = np.asarray(f1.ids[(ds, "s1eval")])[ok1]
                st = np.asarray([strata[ds][q] for q in ids])
                for s in sorted(set(st)):
                    sel = st == s
                    d3, l3 = compare(G1, m4[a][sel], m1[b][sel])
                    rec["diagnostics"]["strata"].append({
                        "fit": f, "dataset": ds, "stratum": str(s), "questions": int(sel.sum()), "step1": means(m1[b][sel]),
                        "step4c": means(m4[a][sel]), "diff": d3, "label": l3,
                        "refs": {nm: {"frozen": means(refs1[j][sel]), "P_F": means(refs4[j][sel])} for j, nm in enumerate(G1.REFS)}})
            log(f"{f} {ds} ({role}{', primary' if primary else ''}): step 1 {cands[b]} {row['step1'][0]:.4f}  "
                f"4c {cands[a]} {row['step4c'][0]:.4f}  diff {fmt_diff(diff['R@5'])} {lab}")
    prim = [r for r in rec["reads"] if r["primary"]]
    if len(prim) != 11:
        raise SystemExit(f"{len(prim)} primary reads, declared 11")
    rec["verdict"] = G1.verdict_of([r["label"] for r in prim])
    rec["primary_labels"] = {f"{r['fit']}:{r['dataset']}": r["label"] for r in prim}
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = utc()
    write_json(OUT / "grade.json", rec)
    write_text(OUT / "grade.md", render_grade(rec))
    log(f"verdict {rec['verdict']}: {rec['primary_labels']} ({rec['seconds']:.0f}s); {OUT / 'grade.json'}")
    return 0


def render_grade(rec):
    lines = [f"# Step 4c result: {rec['verdict']}", "",
             f"Pick rule: {rec['rule']} (step 1's verdict {rec['step1_verdict']}). Per question with gold: R@5 of the 4c fit's "
             "pick on the P_F read minus step 1's fit's pick on the frozen read.", "",
             "## Reads", "",
             "| fit | dataset | role | primary | step 1 pick | 4c pick | R@5 step 1 | R@5 4c | diff R@5 [95%] | label | FC@5 diff | hit@1 diff |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rec["reads"]:
        lines.append(f"| {r['fit']} | {r['dataset']} | {r['role']} | {'yes' if r['primary'] else ''} | {r['step1_pick']} "
                     f"| {r['step4c_pick']} | {r['step1'][0]:.4f} | {r['step4c'][0]:.4f} | {fmt_diff(r['diff']['R@5'])} "
                     f"| {r['label']} | {r['diff']['FC@5']['mean']:+.4f} | {r['diff']['hit@1']['mean']:+.4f} |")
    lines += ["", "## References on the same rows (J5's reads): frozen / P_F R@5", "", "| dataset | rrf | twin0 | gnn0 |",
              "| --- | --- | --- | --- |"]
    for r in rec["reads"]:
        if r["fit"] == "J5":
            lines.append(f"| {r['dataset']} | " + " | ".join(f"{v['frozen'][0]:.4f} / {v['P_F'][0]:.4f}" for v in r["refs"].values()) + " |")
    lines += ["", "## Primary reads by stratum", "", "| fit | dataset | stratum | n | step 1 | 4c | diff R@5 [95%] | label |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in rec["diagnostics"]["strata"]:
        lines.append(f"| {s['fit']} | {s['dataset']} | {s['stratum']} | {s['questions']:,} | {s['step1'][0]:.4f} "
                     f"| {s['step4c'][0]:.4f} | {fmt_diff(s['diff']['R@5'])} | {s['label']} |")
    lines += ["", "## Pools (s1eval)", "", "| dataset | questions | pools changed | short | ALL frozen / P_F | recall frozen / P_F |",
              "| --- | --- | --- | --- | --- | --- |"]
    for k, v in rec["diagnostics"]["pools"].items():
        if k.endswith("=s1eval"):
            lines.append(f"| {k.split('=')[0]} | {v['questions']:,} | {v['changed']:,} | {v['short']} "
                         f"| {v['ALL']['I']:.3f} / {v['ALL']['P_F']:.3f} | {v['recall']['I']:.3f} / {v['recall']['P_F']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pools", "copy-bases", "gate", "screen-prep", "screen-grade", "check", "grade"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--fit", default=None)
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "pools":
        names = [n for n in (a.datasets.split(",") if a.datasets else SIX) if n]
        if any(n not in SIX for n in names):
            raise SystemExit(f"--datasets: of {SIX}")
        return pools_stage(names, a.threads)
    if a.cmd == "copy-bases":
        return copy_bases()
    if a.cmd == "gate":
        return gate()
    if a.cmd in ("screen-prep", "check"):
        if a.fit not in ("J5",) + tuple(f"L-{d}" for d in TRAIN):
            raise SystemExit("--fit: J5 or L-<dataset>")
        return screen_prep(a.fit) if a.cmd == "screen-prep" else check(a.fit, a.host)
    if a.cmd == "screen-grade":
        return screen_grade(a.host)
    return grade(a.host)


if __name__ == "__main__":
    raise SystemExit(main())
