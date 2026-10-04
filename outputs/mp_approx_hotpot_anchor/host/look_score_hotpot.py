"""Design look for hotpotqa (untracked; not a result and not filed): outputs/mp_approx_l16_design/l16_look_score.py with
the dataset named hotpotqa and nothing else changed (its sha256 is pinned in the anchor look that reads this one). It scores the hotpotqa train-split rows
of one carve of the M3B fit stride (x1 by default: remaining[1::s_fit], rows the twin and the GNN never trained on or
were selected on) with the pilot's six functions, and keeps, per query, what an offline comparison of relation-token
schemes needs: the pool, every packed message edge of every family (local endpoints, family, direction flags,
weight), the seeds and their buckets, the six functions' per-node scores and per-query metrics, the gold flags, the
query embedding, a fixed random projection of each pool node's dense embedding (128 dims, seeded, no data dependence)
and the row's hotpotqa type and evidence relations (label-only, read only to describe what the clusters align with).

Nothing here enters a level. scripts/ and src/ are imported and not edited; nothing is written outside this
directory. Host CPU, the verified mirror in memory (configs/host_mirror_six.yaml, verify_hotpotqa.json).
Level 0's pins are checked without eval arrays (level 0 declared none for hotpotqa): the contract, the selection, the
six weights and records, and the frozen code.

    python outputs/mp_approx_hotpot_anchor/host/look_score_hotpot.py --carve x1 --host [--shard i/n] [--limit N]
"""
import os
import sys

THREADS = 6
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import mp_approx_l0 as L0  # noqa: E402
import mp_approx_l3 as L3  # noqa: E402
import mp_approx_l8 as L8  # noqa: E402
import mp_approx_l12 as P12  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.m3b_features import EDGE_ATTR  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES, rank_metrics  # noqa: E402

NAME = "hotpotqa"
VERIFY = ROOT / "outputs" / "host_mirror_six" / "verify_hotpotqa.json"
VERIFY_SHA = "7bf639021d627bee5c036d12c8c5e09d681d887831334a8707a454a04935d335"
FREEZE = "58958f33a3af74c21fa625ffd79da16c573cb5d2ba7e9745da0056b568591ab0"
MIRROR_ROOT = "C:/Users/Student2/rx/projects/mpr/mirror/CRAG"
A0 = L8.A0
COL_W = A0 + EDGE_ATTR.index("weight")
COL_FWD, COL_BWD = L8.COL_FWD, L8.COL_BWD
PROJ_DIM, PROJ_SEED = 128, 20261001
for _d in (L0.HARD_STOP_DIR, L8.HARD_STOP_DIR, P12.HARD_STOP_DIR):
    _d[0] = HERE


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


def host_mode() -> dict:
    import universal_v2_run as U
    root = yaml.safe_load((ROOT / "configs" / "host_mirror_six.yaml").read_text(encoding="utf-8"))["host"]["mirror_root"]
    if root != MIRROR_ROOT:
        raise SystemExit(f"the mirror root {root} is not {MIRROR_ROOT}")
    rec = json.loads(VERIFY.read_text(encoding="utf-8"))
    served = (Path(root) / "data" / "final_canonical").as_posix()
    if not (rec.get("status") == "VERIFIED" and rec.get("freeze_matches_declared") is True and rec.get("loader_imported_from_mirror") is True
            and Path(rec.get("mirror", "")).as_posix() == served and rec.get("freeze_RECORD_SHA256") == FREEZE and NAME in rec.get("datasets", [])):
        raise SystemExit("verify_hotpotqa.json is not VERIFIED at the mirror for hotpotqa")
    original = U.load_configs

    def on_the_mirror(*args, **kwargs):
        cfg, cfg_m3b, cfg_h = original(*args, **kwargs)
        cfg_m3b["substrate"]["package_root"] = str(Path(root))   # in memory only
        return cfg, cfg_m3b, cfg_h

    U.load_configs = on_the_mirror
    L8.PLACEMENT.clear()
    L8.PLACEMENT.update({"where": "host", "node": platform.node(), "mirror_root": root})
    log(f"host: the mirror at {root} in place of the package, in memory; verify_hotpotqa.json is VERIFIED")
    return dict(L8.PLACEMENT)


def carve_population(m3b_compile, ds, ids, kind, m3a, positions):
    """L12.carve_population's lines with the dataset named hotpotqa: the dataset row of each id, its train-split row, the
    golds by m3a.resolve_gold, and zero-gold ids out. Also returns each kept row's type and evidences."""
    row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
    keep = set(ids)
    by_id = {row["query_id"]: row for row in ds.queries("train") if row["query_id"] in keep}
    rows = [by_id[q] for q in ids]
    idx = np.asarray([row_of[q] for q in ids], dtype=np.int64)
    golds = m3a.resolve_gold(rows, positions, NAME)
    zero = np.asarray([g.size == 0 for g in golds], dtype=bool)
    kept = [q for q, z in zip(ids, zero) if not z]
    pop = m3b_compile.Population(NAME, kind, kept, idx[~zero], [g for g, z in zip(golds, zero) if not z], len(ids), int(zero.sum()),
                                 m3b_pools.ids_digest(kept))
    info = {q: {"type": str(by_id[q].get("type")), "evidences": by_id[q].get("evidences") or [], "gold_refs": by_id[q].get("gold_refs") or []}
            for q in kept}
    return pop, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--carve", default="x1")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(THREADS)
    t_start = time.time()
    if L0.sha256_file(VERIFY) != VERIFY_SHA:
        raise SystemExit("verify_hotpotqa.json is not the pinned bytes")
    placement = host_mode() if a.host else {"where": "laptop"}
    import universal_v2_run as U
    decl0 = L0.load_declaration()
    L0.verify_pins(decl0, ())   # no eval arrays: level 0 declared none for hotpotqa
    cfg, cfg_m3b, cfg_h = U.load_configs()
    inputs = U.model_inputs(cfg, cfg_m3b)
    n_sc = int(inputs["n_scalars"])
    selection = U.read_json(ROOT / decl0["inputs"]["selection"]["path"])
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    m3b_contract = U.M3B_RUN.load_script("m3b_contract")
    contexts, handles, pkg, bank = U.open_contexts_v2(cfg_m3b, [NAME], m3b_compile)
    context, ds = contexts[NAME], handles[NAME]
    models = L0.load_models(decl0, U, inputs, bank, selection)
    for k in L0.SEEDS:
        twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
        if twin.input.input_width != 2 * n_sc + L0.N_VECTOR or gnn.steps != L0.STEPS or gnn.message_passing is not True:
            raise SystemExit(f"seed {k}: not the pilot's architecture")
    m3a = pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][NAME]["construction"]
    positions = m3a.node_position_map(ds)
    source, fit, select = m3b_compile.carves_for(ds, NAME, cfg_m3b)
    fit_cap = int(cfg_m3b["populations"]["training_carves"]["fit"]["size_cap"])
    rule = P12.carve_rule(source, m3b_compile.SELECT_CAP, m3b_compile.SELECT_FRACTION, fit_cap)
    if rule["select"] != select or P12.carve_ids_of("fit", rule) != fit:
        raise SystemExit("the recomputed carve rule is not carves_for's fit and select")
    tc = cfg["m3b_incumbents"]["training_carves_reused_here"][NAME]
    if (int(tc["N"]), int(tc["fit"]), int(tc["select"]), tc["fit_sha256"], tc["select_sha256"]) != \
            (len(source), len(fit), len(select), m3b_pools.ids_digest(fit), m3b_pools.ids_digest(select)):
        raise SystemExit("the fit or select carve differs from configs/universal_v2.yaml's digests")
    ids_carve = P12.carve_ids_of(a.carve, rule)
    log(f"{NAME}: N {len(source)}, s_sel {rule['s_sel']}, s_fit {rule['s_fit']}, remaining {len(rule['remaining'])}; "
        f"carve {a.carve}: {len(ids_carve)} ids")
    pop, info = carve_population(m3b_compile, ds, ids_carve, a.carve, m3a, positions)
    if a.carve in ("fit", "select"):
        ref = m3b_compile.population(ds, NAME, a.carve, cfg_m3b, cfg_h, m3a, positions)
        if list(ref.ids) != list(pop.ids) or not np.array_equal(ref.idx, pop.idx):
            raise SystemExit(f"carve {a.carve} is not m3b_compile.population's")
        del ref
    del positions
    gc.collect()
    rows = np.arange(len(pop.ids), dtype=np.int64)
    if a.limit is not None:
        rows = rows[np.unique(np.linspace(0, rows.size - 1, min(a.limit, rows.size)).round().astype(np.int64))]
    ids = [pop.ids[i] for i in rows]
    carve_queries, carve_digest, zero_excl = len(pop.ids), pop.digest, int(pop.zero_gold_excluded)
    pop.ids, pop.idx, pop.golds = ids, pop.idx[rows], [pop.golds[i] for i in rows]
    t_prep = time.time()
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    n = len(rows)
    sizes = np.asarray([p.size for p in prep.pools], dtype=np.int64)
    chunk = max(1, int(L0.CHUNK_NODES // max(sizes.mean(), 1)))
    n_chunks = math.ceil(n / chunk)
    shard = L8.parse_shard(a.shard)
    mine = L8.shard_chunks(n_chunks, shard)
    out_dir = Path(a.out) if a.out else HERE / "look" / a.carve
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    log(f"carve {a.carve}: {n} of {carve_queries} queries prepared in {time.time() - t_prep:.0f}s, pools mean {sizes.mean():.0f}, "
        f"chunk {chunk}, {n_chunks} chunks ({len(mine)} here), {torch.get_num_threads()} threads")
    proj = (np.random.default_rng(PROJ_SEED).standard_normal((1536, PROJ_DIM)) / math.sqrt(PROJ_DIM)).astype(np.float32)
    columns = inputs["column_indices"]
    funcs = L8.FUNCS
    t0, done = time.time(), 0
    with torch.no_grad():
        for pos, ci in enumerate(mine):
            idx = np.arange(ci * chunk, min((ci + 1) * chunk, n))
            path = chunks_dir / f"c{ci:05d}.npz"
            if path.exists():
                continue
            qds, golds_local, seed_info, embs = [], [], [], []
            for j in idx:
                seeds = np.asarray(prep.seeds[j], dtype=np.int64)
                if not np.array_equal(seeds, m3b_pools.seeds_of(np.asarray(prep.dense_ids[j]), np.asarray(prep.splade_ids[j]))):
                    raise SystemExit(f"{ids[j]}: the prepared seeds are not seeds_of(dense top-5, splade top-5)")
                E = context.nodes.read(prep.pools[j])
                inp = U.QueryInputs(prep.qemb[j], prep.dense_ids[j], prep.dense_scores[j], prep.splade_ids[j], prep.splade_scores[j])
                compiled = U.compile_query_v2(inp, prep.pools[j], prep.seeds[j], context.stores, context.nodes, context.rel_table, embeddings=E)
                pool = np.asarray(compiled.pool, dtype=np.int64)
                sl = np.asarray(compiled.seeds_local, dtype=np.int64)
                if sl.size != seeds.size or not np.array_equal(pool[sl], seeds):
                    raise SystemExit(f"{ids[j]}: a seed is not a pool member")
                top = {int(prep.dense_ids[j][0]), int(prep.splade_ids[j][0])}
                bucket = np.asarray([0 if int(s) in top else 1 for s in seeds], dtype=np.int64)
                gl = m3b_compile.gold_local_of(prep.pools[j], pop.golds[j])
                qds.append({"pool": compiled.pool, "x": compiled.scalars[:, columns], "seedw": compiled.seedw, "qemb": prep.qemb[j],
                            "seeds": compiled.seeds_local, "gold": gl, "gold_total": int(pop.golds[j].size), "emb": E})
                golds_local.append(gl)
                seed_info.append((sl, bucket, pool))
                En = np.asarray(E, dtype=np.float32)
                En = En / np.maximum(np.linalg.norm(En, axis=1, keepdims=True), 1e-12)
                embs.append((En @ proj).astype(np.float16))
                del compiled
            batch = U.pack_queries_v2(qds, context)
            del qds
            ptr = batch.qptr.numpy()
            scores = {}
            for k in L0.SEEDS:
                twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
                scores[f"twin{k}"] = twin(U.arm_view(twin, batch, inputs)).numpy()
                scores[f"gnn{k}"] = gnn(U.arm_view(gnn, batch, inputs)).numpy()
            ei = batch.edge_index.numpy()
            ea = batch.edge_attr.numpy()
            fam_all = L3.family_of(ea[:, :A0])
            P = {k: [] for k in ("q_pool_size", "q_gold_total", "q_gold_in_pool", "q_metrics", "q_emb", "q_seed_local", "q_seed_bucket",
                                 "q_edges", "pool", "score", "is_gold", "proj", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "e_w")}
            for jj, j in enumerate(idx):
                a_, b_ = int(ptr[jj]), int(ptr[jj + 1])
                nq = b_ - a_
                if nq != sizes[j]:
                    raise SystemExit(f"{ids[j]}: packed rows {nq} != pool size {sizes[j]}")
                gl, gt = golds_local[jj], int(pop.golds[j].size)
                full = {f: rank_metrics(np.asarray(scores[f][a_:b_], dtype=np.float64), gl, gt) for f in funcs}
                sel = (ei[1] >= a_) & (ei[1] < b_)
                u, v = ei[0, sel] - a_, ei[1, sel] - a_
                if u.size and (u.min() < 0 or u.max() >= nq):
                    raise SystemExit(f"{ids[j]}: an edge leaves its query")
                sl, bucket, pool = seed_info[jj]
                seed_row, bucket_row = np.full(L8.MAX_SEEDS, -1, dtype=np.int64), np.full(L8.MAX_SEEDS, -1, dtype=np.int64)
                seed_row[:sl.size], bucket_row[:sl.size] = sl, bucket
                is_gold = np.zeros(nq, dtype=bool)
                is_gold[gl] = True
                P["q_pool_size"].append(nq)
                P["q_gold_total"].append(gt)
                P["q_gold_in_pool"].append(int(gl.size))
                P["q_metrics"].append([[full[f][m] for m in METRIC_NAMES] for f in funcs])
                P["q_emb"].append(np.asarray(prep.qemb[j], dtype=np.float32))
                P["q_seed_local"].append(seed_row)
                P["q_seed_bucket"].append(bucket_row)
                P["q_edges"].append(int(u.size))
                P["pool"].append(pool)
                P["score"].append(np.stack([np.asarray(scores[f][a_:b_], dtype=np.float32) for f in funcs], 1))
                P["is_gold"].append(is_gold)
                P["proj"].append(embs[jj])
                P["e_u"].append(u.astype(np.int16))
                P["e_v"].append(v.astype(np.int16))
                P["e_fam"].append(fam_all[sel].astype(np.int8))
                P["e_fwd"].append((ea[sel, COL_FWD] > 0.5).astype(np.int8))
                P["e_bwd"].append((ea[sel, COL_BWD] > 0.5).astype(np.int8))
                P["e_w"].append(ea[sel, COL_W].astype(np.float32))
            arrays = {}
            for key in ("q_pool_size", "q_gold_total", "q_gold_in_pool", "q_edges"):
                arrays[key] = np.asarray(P[key], dtype=np.int64)
            arrays["q_metrics"] = np.asarray(P["q_metrics"], dtype=np.float64)
            arrays["q_emb"] = np.asarray(P["q_emb"], dtype=np.float32)
            arrays["q_seed_local"] = np.asarray(P["q_seed_local"], dtype=np.int64)
            arrays["q_seed_bucket"] = np.asarray(P["q_seed_bucket"], dtype=np.int64)
            for key in ("pool", "score", "is_gold", "proj", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "e_w"):
                arrays[key] = np.concatenate(P[key])
            arrays["chunk_rows"] = rows[idx]
            tmp = chunks_dir / f"c{ci:05d}.tmp.npz"
            np.savez(tmp, **arrays)
            for attempt in range(8):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    time.sleep(5)
            else:
                raise SystemExit(f"{path}: os.replace failed 8 times")
            del batch, scores, arrays, P, ei, ea, embs
            gc.collect()
            done += idx.size
            rate = (time.time() - t0) / done
            left = sum(min((cj + 1) * chunk, n) - cj * chunk for cj in mine[pos + 1:])
            log(f"   chunk {ci + 1}/{n_chunks} ({pos + 1}/{len(mine)} here), {rate * 1000:.0f} ms/query, about {left * rate / 60:.0f} min left")
    rec = {"carve": a.carve, "shard": list(shard) if shard else None, "chunks": mine, "n_chunks": n_chunks, "chunk_queries": chunk,
           "queries": n, "carve_queries": carve_queries, "carve_ids_sha256": carve_digest, "zero_gold_excluded": zero_excl,
           "functions": list(funcs), "metric_names": list(METRIC_NAMES), "proj": {"dim": PROJ_DIM, "seed": PROJ_SEED},
           "rule": {"N": len(source), "s_sel": rule["s_sel"], "s_fit": rule["s_fit"], "remaining": len(rule["remaining"])},
           "limit": a.limit, "placement": placement, "script_sha256": L0.sha256_file(Path(__file__)), "seconds": round(time.time() - t_start, 1),
           "peak_rss_bytes": L8.peak_rss_bytes(), "utc": L0.utc()}
    tag = f"_{shard[0]}of{shard[1]}" if shard else ""
    (out_dir / f"record{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if shard is None or shard[0] == 0:
        (out_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (out_dir / "info.json").write_text(json.dumps([info[q] for q in ids]), encoding="utf-8")
    log(f"done: {done} queries here in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
