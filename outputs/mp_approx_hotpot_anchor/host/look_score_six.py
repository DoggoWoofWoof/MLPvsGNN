"""Design look (untracked; not a result and not filed): l16_look_score.py's scoring pass with the six-dataset pair.

outputs/mp_approx_l16_design/l16_look_score.py (sha256 57974347...) scores one carve of 2wiki's M3B fit stride with the
pilot trio's six functions. The pilot never trained on hotpotqa, musique or webqsp. configs/mp_approx_six_base.yaml
(status RUN, 84855f8) built the six-dataset pair: T_k = u_mlp_v2_mix__H128__six__s{k} (its work 1) and G_k =
u_gnn_v2_ef__H128__six__s{k} (universal-v2 stage 2). Both were fitted on the six M3B fit carves and early-stopped on the
six select carves. This look is l16_look_score.py's main with that pair. scripts/mp_approx_six_base_score.py's
host_mode, pair_verify, pair_open and pair_models are imported unchanged, so every pin of the six-base's two works is
checked before a row is scored. The contexts, the served six-dataset relation bank and every offset are those the GNN
was scored with, at the six-base's scoring threads (torch 4, BLAS 2). The carve rule, the population, the per-query
arrays and the chunk layout are l16_look_score.py's, line for line, so l16_look_analyze.load reads the look.

Nothing here enters a level. scripts/ and src/ are imported and not edited. Nothing is written outside this directory:
every imported helper's hard stop is routed here.

    python outputs/mp_approx_hotpot_anchor/host/look_score_six.py --dataset hotpotqa --carve x1 --host [--shard i/n] [--limit N]
"""
import os
import sys

TORCH_THREADS = 4
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import mp_approx_l0 as L0  # noqa: E402
import mp_approx_l3 as L3  # noqa: E402
import mp_approx_l8 as L8  # noqa: E402
import mp_approx_l12 as P12  # noqa: E402
import mp_approx_six_base as SB  # noqa: E402
import mp_approx_six_base_score as S6  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.m3b_features import EDGE_ATTR  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES, rank_metrics  # noqa: E402

V2, U6 = SB.V2, SB.U6
A0 = L8.A0
COL_W = A0 + EDGE_ATTR.index("weight")
COL_FWD, COL_BWD = L8.COL_FWD, L8.COL_BWD
PROJ_DIM, PROJ_SEED = 128, 20261001
LOOK_SCORE_SHA = "579743477fc04a68bff321d6d6a8834c1c328f57f74225908e3ad055cc2ad7fd"   # the pass copied here
for _d in (L0.HARD_STOP_DIR, L8.HARD_STOP_DIR, P12.HARD_STOP_DIR):
    _d[0] = HERE   # after S6's import, which pointed level 0's at outputs/mp_approx_six_base
SB.OUT = HERE      # SB.hard_stop (and S6.hard_stop through it) writes HERE/hard_stop.json; every other path was bound at import


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


def carve_population(m3b_compile, ds, ids, kind, m3a, positions, name):
    """l16_look_score.carve_population with the dataset passed in. The kept rows' info keeps each row's own type, level
    and evidence fields where it has them (label-only, never an input)."""
    row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
    keep = set(ids)
    by_id = {row["query_id"]: row for row in ds.queries("train") if row["query_id"] in keep}
    rows = [by_id[q] for q in ids]
    idx = np.asarray([row_of[q] for q in ids], dtype=np.int64)
    golds = m3a.resolve_gold(rows, positions, name)
    zero = np.asarray([g.size == 0 for g in golds], dtype=bool)
    kept = [q for q, z in zip(ids, zero) if not z]
    pop = m3b_compile.Population(name, kind, kept, idx[~zero], [g for g, z in zip(golds, zero) if not z], len(ids), int(zero.sum()),
                                 m3b_pools.ids_digest(kept))
    info = {q: {"type": str(by_id[q].get("type")), "level": str(by_id[q].get("level")), "evidences": by_id[q].get("evidences") or [],
                "gold_refs": by_id[q].get("gold_refs") or []} for q in kept}
    return pop, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--carve", default="x1")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    name = a.dataset
    if name not in S6.DATASETS:
        raise SystemExit(f"{name} is not one of the six")
    torch.set_num_threads(TORCH_THREADS)
    t_start = time.time()
    placement = dict(S6.host_mode(SB.load_declaration(), log)) if a.host else {"where": "laptop"}
    S6.pair_verify(S6.SIX)
    cfg, cfg_m3b, cfg_h = V2.load_configs()
    op = S6.pair_open(S6.SIX, cfg, cfg_m3b, name)
    inputs, m3b_compile = op.inputs, op.m3b_compile
    n_sc = int(inputs["n_scalars"])
    m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
    context, ds = op.contexts[name], op.handles[name]
    models = S6.pair_models(S6.SIX, inputs, op.bank)
    for k in L0.SEEDS:
        twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
        if twin.input.input_width != 2 * n_sc + L0.N_VECTOR or gnn.steps != L0.STEPS or gnn.message_passing is not True:
            raise SystemExit(f"seed {k}: not the six pair's architecture")
    m3a = op.pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][name]["construction"]
    positions = m3a.node_position_map(ds)
    source, fit, select = m3b_compile.carves_for(ds, name, cfg_m3b)
    fit_cap = int(cfg_m3b["populations"]["training_carves"]["fit"]["size_cap"])
    rule = P12.carve_rule(source, m3b_compile.SELECT_CAP, m3b_compile.SELECT_FRACTION, fit_cap)
    if rule["select"] != select or P12.carve_ids_of("fit", rule) != fit:
        raise SystemExit("the recomputed carve rule is not carves_for's fit and select")
    tc = U6.stage2_block(cfg)["training_carves"][name]   # the carves the six pair trained and early-stopped on
    got = {"N": len(source), "fit": len(fit), "select": len(select), "fit_sha256": m3b_pools.ids_digest(fit),
           "select_sha256": m3b_pools.ids_digest(select)}
    bad = [k for k in got if k in tc and (int(tc[k]) if k in ("N", "fit", "select") else tc[k]) != got[k]]
    if bad or "fit_sha256" not in tc:
        raise SystemExit(f"the fit or select carve differs from stage 2's filed carve: {bad}")
    ids_carve = P12.carve_ids_of(a.carve, rule)
    log(f"{name}: N {len(source)}, s_sel {rule['s_sel']}, s_fit {rule['s_fit']}, remaining {len(rule['remaining'])}; "
        f"carve {a.carve}: {len(ids_carve)} ids; stage 2's carve checks: {sorted(k for k in got if k in tc)}")
    pop, info = carve_population(m3b_compile, ds, ids_carve, a.carve, m3a, positions, name)
    if a.carve in ("fit", "select"):
        ref = m3b_compile.population(ds, name, a.carve, cfg_m3b, cfg_h, m3a, positions)
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
    out_dir = Path(a.out) if a.out else HERE / "look" / name / a.carve
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
                inp = V2.QueryInputs(prep.qemb[j], prep.dense_ids[j], prep.dense_scores[j], prep.splade_ids[j], prep.splade_scores[j])
                compiled = V2.compile_query_v2(inp, prep.pools[j], prep.seeds[j], context.stores, context.nodes, context.rel_table, embeddings=E)
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
            batch = V2.pack_queries_v2(qds, context)
            del qds
            ptr = batch.qptr.numpy()
            scores = {}
            for k in L0.SEEDS:
                twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
                scores[f"twin{k}"] = twin(V2.arm_view(twin, batch, inputs)).numpy()
                scores[f"gnn{k}"] = gnn(V2.arm_view(gnn, batch, inputs)).numpy()
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
    rec = {"dataset": name, "pair": S6.SIX.name, "pair_keys": S6.SIX.keys(), "carve": a.carve, "shard": list(shard) if shard else None,
           "chunks": mine, "n_chunks": n_chunks, "chunk_queries": chunk, "queries": n, "carve_queries": carve_queries,
           "carve_ids_sha256": carve_digest, "zero_gold_excluded": zero_excl, "functions": list(funcs), "metric_names": list(METRIC_NAMES),
           "proj": {"dim": PROJ_DIM, "seed": PROJ_SEED}, "threads": {"torch": TORCH_THREADS, "blas": os.environ.get("OPENBLAS_NUM_THREADS")},
           "rule": {"N": len(source), "s_sel": rule["s_sel"], "s_fit": rule["s_fit"], "remaining": len(rule["remaining"])},
           "limit": a.limit, "placement": placement, "copied_from_sha256": LOOK_SCORE_SHA, "script_sha256": L0.sha256_file(Path(__file__)),
           "seconds": round(time.time() - t_start, 1), "peak_rss_bytes": L8.peak_rss_bytes(), "utc": L0.utc()}
    tag = f"_{shard[0]}of{shard[1]}" if shard else ""
    (out_dir / f"record{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if shard is None or shard[0] == 0:
        (out_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (out_dir / "info.json").write_text(json.dumps([info[q] for q in ids]), encoding="utf-8")
    log(f"done: {done} queries here in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
