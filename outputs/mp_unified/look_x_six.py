"""Design look (untracked; not a result and not filed): look_score_kb.py's scoring pass for any of the six datasets,
keeping each pool node's 129 compiled input columns, plus a frozen-model feature audit on a subset of chunks.

outputs/mp_approx_kb_anchor/host/look_score_kb.py (sha256 pinned below) scores one carve with the six-dataset pair
(T_k = u_mlp_v2_mix__H128__six__s{k}, G_k = u_gnn_v2_ef__H128__six__s{k}) and keeps per row the pool, the six scores,
the gold flags, the seeds and (KBs) each structural edge's relation ids. It does not keep the inputs the two models read.
This copy keeps them:
    x       (nodes, 129) float16: compiled.scalars[:, inputs['column_indices']], the columns both models read, in the
            frozen contract's order (inputs['columns'] in the record)
    seedw   (nodes, MAX_SEEDS) float16: the compiled reach weights the twin's reach prototype and the GNN read
and, with --full, every array look_score_kb.py writes (q_emb, edges, proj, e_rel on a typed KB). Without --full the
edges and projections are not written again: a carve that already has a look (outputs/mp_approx_l16_design/look,
outputs/mp_approx_hotpot_anchor/host/look, outputs/mp_approx_kb_anchor/host/look) is joined to it row by row on the
pool, which both passes compile the same way (same contexts, same threads, same chunk layout).

Audit (--audit-every N > 0): on every N-th chunk the twin and the GNN of seed --audit-seed are scored again with one
input channel destroyed at a time, and each row's rank metrics are kept (q_audit: rows x perturbations x 2 x metrics):
    a column block   the block's columns permuted across the row's pool nodes (one permutation per row, shared by the
                     block's columns): its marginal over the pool kept, its tie to each node removed. Both models
                     z-score inside the pool, so the raw and the z-scored copies move together.
    semantic         the node embeddings permuted the same way (the twin's semantic channel, its neighbour and reach
                     prototypes, and the GNN's node input all read them)
    no_struct / no_ner / no_knn / no_edges   that edge family (or every edge) removed from the packed batch: the GNN's
                     messages and the twin's per-family neighbour prototypes both lose it
A drop in a row's metric under a perturbation is how much that model leans on that channel on this graph. It is an
importance read on frozen models, not a refit: redundant channels each read low. Nothing here enters a level, a stage
or a selection. scripts/ and src/ are imported and not edited; every helper's hard stop is routed here.

    python outputs/mp_unified/look_x_six.py --dataset musique --carve x1 --host [--shard i/n] [--audit-every 10] [--full]
"""
import os
import sys

TORCH_THREADS = 4
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

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
K_REL = V2.K_REL
LOOK_KB = ROOT / "outputs" / "mp_approx_kb_anchor" / "host" / "look_score_kb.py"
AUDIT_RNG = 20261003
for _d in (L0.HARD_STOP_DIR, L8.HARD_STOP_DIR, P12.HARD_STOP_DIR):
    _d[0] = HERE   # after S6's import, which pointed level 0's at outputs/mp_approx_six_base
SB.OUT = HERE      # SB.hard_stop (and S6.hard_stop through it) writes HERE/hard_stop.json


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


RETRIEVAL = ("dense_cos", "dense_rr", "splade_rr", "rrf", "agreement", "is_seed")
SEEDCOND = ("cos_v_seedproto", "max_cos_v_seed", "cos_v_reachproto", "has_reach_seed", "cos_v_seedproto_h2", "has_seed_h2")
TYPED_REL = ("relmax_seed", "relmax_in", "relmean_in", "has_typed_edge", "rel_ief", "rel_div", "dir_in_frac", "seed_edges_out",
             "seed_edges_in", "relchain2_max")
TYPED_V2 = ("qsupport_h2", "qsupport_h3", "typed_walks_h3", "relpath_max_h3", "relpath_mean_h3", "relpath_min_h3")
EDGE_DROPS = {"no_struct": (0,), "no_ner": (1,), "no_knn": (2,), "no_edges": (0, 1, 2)}


def column_blocks(names):
    """Each of the 129 columns in exactly one block, by its name; plus global_stats, an overlapping read of the
    corpus-level columns (global degree, component size, seed component) across the four topology views."""
    blocks = {}
    for i, n in enumerate(names):
        if n in RETRIEVAL:
            b = "retrieval"
        elif re.search(r"_h[123]_STRUCT$", n):
            b = "depth_STRUCT"
        elif re.search(r"_h[123]_FULL$", n):
            b = "depth_FULL"
        elif n.startswith("opath_"):
            b = "ordered"
        elif n in TYPED_V2:
            b = "typed_v2"
        elif n in TYPED_REL:
            b = "typed_rel"
        elif n in SEEDCOND:
            b = "seedcond"
        elif n.startswith("gcs_"):
            b = "gcs"
        elif n.startswith(("cos_q_proto_", "max_q_nbr_", "cohesion_", "has_nbr_")):
            b = "nbr_agg"
        elif n.endswith("_STRUCT"):
            b = "topo_STRUCT"
        elif n.endswith("_NER"):
            b = "topo_NER"
        elif n.endswith("_KNN"):
            b = "topo_KNN"
        elif n.endswith("_FULL"):
            b = "topo_FULL"
        else:
            raise SystemExit(f"column {n} falls in no block")
        blocks.setdefault(b, []).append(i)
    if sorted(i for v in blocks.values() for i in v) != list(range(len(names))):
        raise SystemExit("the blocks do not partition the columns")
    blocks["global_stats"] = [i for i, n in enumerate(names) if n.startswith(("deg_global_", "component_size_", "seed_component_"))]
    return blocks


def perturbations(names):
    blocks = column_blocks(names)
    out = [(f"cols:{b}", ("cols", idx)) for b, idx in blocks.items()]
    out.append(("semantic", ("emb", None)))
    out += [(k, ("edges", fams)) for k, fams in EDGE_DROPS.items()]
    return out, blocks


def row_permutation(qptr, rng):
    """One permutation of each row's pool nodes, as a global gather index."""
    idx = np.empty(int(qptr[-1]), dtype=np.int64)
    for a, b in zip(qptr[:-1], qptr[1:]):
        idx[a:b] = a + rng.permutation(b - a)
    return torch.from_numpy(idx)


def perturbed(batch, kind, arg, perm):
    if kind == "cols":
        x = batch.x.clone()
        cols = torch.as_tensor(arg, dtype=torch.long)
        x[:, cols] = batch.x[perm][:, cols]
        return dataclasses.replace(batch, x=x)
    if kind == "emb":
        return dataclasses.replace(batch, emb=batch.emb[perm])
    if kind == "edges":
        if batch.edge_attr.shape[0] == 0:
            return batch
        fam = batch.edge_attr[:, :A0].argmax(dim=1)
        keep = torch.ones(fam.shape[0], dtype=torch.bool)
        for f in arg:
            keep &= fam != f
        return dataclasses.replace(batch, edge_index=batch.edge_index[:, keep], edge_attr=batch.edge_attr[keep])
    raise ValueError(kind)


def carve_population(m3b_compile, ds, ids, kind, m3a, positions, name):
    """look_score_six.carve_population, line for line."""
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
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--carve", required=True)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--full", action="store_true", help="also write look_score_kb's edges, projections and query embeddings")
    ap.add_argument("--audit-every", type=int, default=0)
    ap.add_argument("--audit-seed", type=int, default=0)
    a = ap.parse_args(argv)
    name = a.dataset
    if name not in S6.DATASETS:
        raise SystemExit(f"{name} is not one of the six")
    if (name, a.carve) in (("2wiki", "x2"), ("2wiki", "x3"), ("musique", "x3")):
        raise SystemExit(f"{name}/{a.carve} is reserved for a declared read")
    if a.audit_every < 0 or a.audit_seed not in L0.SEEDS:
        raise SystemExit("--audit-every: non-negative; --audit-seed: one of the pair's seeds")
    torch.set_num_threads(TORCH_THREADS)
    t_start = time.time()
    placement = dict(S6.host_mode(SB.load_declaration(), log)) if a.host else {"where": "laptop"}
    S6.pair_verify(S6.SIX)
    cfg, cfg_m3b, cfg_h = V2.load_configs()
    op = S6.pair_open(S6.SIX, cfg, cfg_m3b, name)
    inputs, m3b_compile = op.inputs, op.m3b_compile
    n_sc = int(inputs["n_scalars"])
    names = list(inputs["columns"])
    if len(names) != n_sc or n_sc != 129:
        raise SystemExit("the pair does not read the 129-column contract")
    pert, blocks = perturbations(names)
    m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
    context, ds = op.contexts[name], op.handles[name]
    models = S6.pair_models(S6.SIX, inputs, op.bank)
    for k in L0.SEEDS:
        twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
        if twin.input.input_width != 2 * n_sc + L0.N_VECTOR or gnn.steps != L0.STEPS or gnn.message_passing is not True:
            raise SystemExit(f"seed {k}: not the six pair's architecture")
    offset = int(getattr(context, "rel_offset", -1))
    n_rel = 0 if context.rel_table is None else int(np.asarray(context.rel_table.embeddings).shape[0])
    typed = offset >= 0 and n_rel > 0
    m3a = op.pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][name]["construction"]
    positions = m3a.node_position_map(ds)
    source, fit, select = m3b_compile.carves_for(ds, name, cfg_m3b)
    fit_cap = int(cfg_m3b["populations"]["training_carves"]["fit"]["size_cap"])
    rule = P12.carve_rule(source, m3b_compile.SELECT_CAP, m3b_compile.SELECT_FRACTION, fit_cap)
    if rule["select"] != select or P12.carve_ids_of("fit", rule) != fit:
        raise SystemExit("the recomputed carve rule is not carves_for's fit and select")
    tc = U6.stage2_block(cfg)["training_carves"][name]
    got = {"N": len(source), "fit": len(fit), "select": len(select), "fit_sha256": m3b_pools.ids_digest(fit),
           "select_sha256": m3b_pools.ids_digest(select)}
    bad = [k for k in got if k in tc and (int(tc[k]) if k in ("N", "fit", "select") else tc[k]) != got[k]]
    if bad or "fit_sha256" not in tc:
        raise SystemExit(f"the fit or select carve differs from stage 2's filed carve: {bad}")
    ids_carve = P12.carve_ids_of(a.carve, rule)
    log(f"{name}: N {len(source)}, carve {a.carve}: {len(ids_carve)} ids; typed relations: {typed}")
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
    audited = [ci for ci in mine if a.audit_every and ci % a.audit_every == 0]
    log(f"carve {a.carve}: {n} of {carve_queries} queries prepared in {time.time() - t_prep:.0f}s, pools mean {sizes.mean():.0f}, "
        f"chunk {chunk}, {n_chunks} chunks ({len(mine)} here, {len(audited)} audited x {len(pert)} perturbations), "
        f"{torch.get_num_threads()} threads")
    proj = (np.random.default_rng(PROJ_SEED).standard_normal((1536, PROJ_DIM)) / math.sqrt(PROJ_DIM)).astype(np.float32)
    columns = inputs["column_indices"]
    funcs = L8.FUNCS
    t0, done, audit_s = time.time(), 0, 0.0
    with torch.no_grad():
        for pos, ci in enumerate(mine):
            idx = np.arange(ci * chunk, min((ci + 1) * chunk, n))
            path = chunks_dir / f"c{ci:05d}.npz"
            if path.exists():
                continue
            qds, golds_local, seed_info, embs, xs, sws = [], [], [], [], [], []
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
                xq = compiled.scalars[:, columns]
                qds.append({"pool": compiled.pool, "x": xq, "seedw": compiled.seedw, "qemb": prep.qemb[j],
                            "seeds": compiled.seeds_local, "gold": gl, "gold_total": int(pop.golds[j].size), "emb": E})
                xs.append(np.asarray(xq, dtype=np.float16))
                sws.append(np.asarray(compiled.seedw, dtype=np.float16))
                golds_local.append(gl)
                seed_info.append((sl, bucket, pool))
                if a.full:
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
            audit = None
            if ci in audited:
                t_a = time.time()
                rng = np.random.default_rng(AUDIT_RNG + ci)
                perm = row_permutation(ptr, rng)
                audit = np.zeros((idx.size, len(pert), 2, len(METRIC_NAMES)), dtype=np.float64)
                pair = (models[f"twin{a.audit_seed}"], models[f"gnn{a.audit_seed}"])
                for pi, (_pname, (kind, arg)) in enumerate(pert):
                    pb = perturbed(batch, kind, arg, perm)
                    for mi, mdl in enumerate(pair):
                        sc = mdl(V2.arm_view(mdl, pb, inputs)).numpy()
                        for jj in range(idx.size):
                            a_, b_ = int(ptr[jj]), int(ptr[jj + 1])
                            r = rank_metrics(np.asarray(sc[a_:b_], dtype=np.float64), golds_local[jj], int(pop.golds[idx[jj]].size))
                            audit[jj, pi, mi] = [r[m] for m in METRIC_NAMES]
                    del pb
                audit_s += time.time() - t_a
            ei = batch.edge_index.numpy()
            ea = batch.edge_attr.numpy()
            fam_all = L3.family_of(ea[:, :A0])
            keys = ["q_pool_size", "q_gold_total", "q_gold_in_pool", "q_metrics", "q_seed_local", "q_seed_bucket", "pool", "score", "is_gold"]
            if a.full:
                keys += ["q_emb", "q_edges", "proj", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "e_w"] + (["e_rel"] if typed else [])
            P = {k: [] for k in keys}
            for jj, j in enumerate(idx):
                a_, b_ = int(ptr[jj]), int(ptr[jj + 1])
                nq = b_ - a_
                if nq != sizes[j] or xs[jj].shape[0] != nq:
                    raise SystemExit(f"{ids[j]}: packed rows {nq} != pool size {sizes[j]}")
                gl, gt = golds_local[jj], int(pop.golds[j].size)
                full = {f: rank_metrics(np.asarray(scores[f][a_:b_], dtype=np.float64), gl, gt) for f in funcs}
                sl, bucket, pool = seed_info[jj]
                seed_row, bucket_row = np.full(L8.MAX_SEEDS, -1, dtype=np.int64), np.full(L8.MAX_SEEDS, -1, dtype=np.int64)
                seed_row[:sl.size], bucket_row[:sl.size] = sl, bucket
                is_gold = np.zeros(nq, dtype=bool)
                is_gold[gl] = True
                P["q_pool_size"].append(nq)
                P["q_gold_total"].append(gt)
                P["q_gold_in_pool"].append(int(gl.size))
                P["q_metrics"].append([[full[f][m] for m in METRIC_NAMES] for f in funcs])
                P["q_seed_local"].append(seed_row)
                P["q_seed_bucket"].append(bucket_row)
                P["pool"].append(pool)
                P["score"].append(np.stack([np.asarray(scores[f][a_:b_], dtype=np.float32) for f in funcs], 1))
                P["is_gold"].append(is_gold)
                if a.full:
                    sel = (ei[1] >= a_) & (ei[1] < b_)
                    u, v = ei[0, sel] - a_, ei[1, sel] - a_
                    if u.size and (u.min() < 0 or u.max() >= nq):
                        raise SystemExit(f"{ids[j]}: an edge leaves its query")
                    P["q_emb"].append(np.asarray(prep.qemb[j], dtype=np.float32))
                    P["q_edges"].append(int(u.size))
                    P["proj"].append(embs[jj])
                    P["e_u"].append(u.astype(np.int16))
                    P["e_v"].append(v.astype(np.int16))
                    P["e_fam"].append(fam_all[sel].astype(np.int8))
                    P["e_fwd"].append((ea[sel, COL_FWD] > 0.5).astype(np.int8))
                    P["e_bwd"].append((ea[sel, COL_BWD] > 0.5).astype(np.int8))
                    P["e_w"].append(ea[sel, COL_W].astype(np.float32))
                    if typed:
                        slots = ea[sel][:, ea.shape[1] - K_REL:]
                        if (slots[(slots >= 0)] < offset).any() or (slots[(slots >= 0)] >= offset + n_rel).any():
                            raise SystemExit(f"{ids[j]}: a relation slot is outside {name}'s rows of the bank")
                        P["e_rel"].append(np.where(slots >= 0, slots - offset, -1).astype(np.int16))
            arrays = {}
            for key in ("q_pool_size", "q_gold_total", "q_gold_in_pool") + (("q_edges",) if a.full else ()):
                arrays[key] = np.asarray(P[key], dtype=np.int64)
            arrays["q_metrics"] = np.asarray(P["q_metrics"], dtype=np.float64)
            arrays["q_seed_local"] = np.asarray(P["q_seed_local"], dtype=np.int64)
            arrays["q_seed_bucket"] = np.asarray(P["q_seed_bucket"], dtype=np.int64)
            if a.full:
                arrays["q_emb"] = np.asarray(P["q_emb"], dtype=np.float32)
            for key in [k for k in keys if k in ("pool", "score", "is_gold", "proj", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd", "e_w", "e_rel")]:
                arrays[key] = np.concatenate(P[key])
            arrays["pool"] = arrays["pool"].astype(np.int32)
            arrays["x"] = np.concatenate(xs)
            arrays["seedw"] = np.concatenate(sws)
            if audit is not None:
                arrays["q_audit"] = audit
            arrays["chunk_rows"] = rows[idx]
            tmp = chunks_dir / f"c{ci:05d}.tmp.npz"
            np.savez_compressed(tmp, **arrays)
            for attempt in range(8):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    time.sleep(5)
            else:
                raise SystemExit(f"{path}: os.replace failed 8 times")
            del batch, scores, arrays, P, ei, ea, embs, xs, sws, audit
            gc.collect()
            done += idx.size
            rate = (time.time() - t0) / done
            left = sum(min((cj + 1) * chunk, n) - cj * chunk for cj in mine[pos + 1:])
            log(f"   chunk {ci + 1}/{n_chunks} ({pos + 1}/{len(mine)} here), {rate * 1000:.0f} ms/query, audit {audit_s:.0f}s so far, "
                f"about {left * rate / 60:.0f} min left")
    rec = {"look": "look_x_six", "dataset": name, "pair": S6.SIX.name, "pair_keys": S6.SIX.keys(), "carve": a.carve,
           "shard": list(shard) if shard else None, "chunks": mine, "n_chunks": n_chunks, "chunk_queries": chunk, "queries": n,
           "carve_queries": carve_queries, "carve_ids_sha256": carve_digest, "zero_gold_excluded": zero_excl, "functions": list(funcs),
           "metric_names": list(METRIC_NAMES), "columns": names, "column_blocks": blocks, "full": a.full,
           "proj": {"dim": PROJ_DIM, "seed": PROJ_SEED} if a.full else None,
           "audit": {"every": a.audit_every, "seed": a.audit_seed, "chunks": audited, "perturbations": [p[0] for p in pert],
                     "models": [f"twin{a.audit_seed}", f"gnn{a.audit_seed}"], "rng": AUDIT_RNG, "seconds": round(audit_s, 1)},
           "threads": {"torch": TORCH_THREADS, "blas": os.environ.get("OPENBLAS_NUM_THREADS")},
           "rule": {"N": len(source), "s_sel": rule["s_sel"], "s_fit": rule["s_fit"], "remaining": len(rule["remaining"])},
           "relations": {"typed": typed, "offset": offset, "n_relations": n_rel, "k_rel": K_REL},
           "limit": a.limit, "placement": placement, "copied_from": str(LOOK_KB.relative_to(ROOT)), "copied_from_sha256": L0.sha256_file(LOOK_KB),
           "script_sha256": L0.sha256_file(Path(__file__)), "seconds": round(time.time() - t_start, 1), "peak_rss_bytes": L8.peak_rss_bytes(),
           "utc": L0.utc()}
    tag = f"_{shard[0]}of{shard[1]}" if shard else ""
    (out_dir / f"record{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if shard is None or shard[0] == 0:
        (out_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (out_dir / "info.json").write_text(json.dumps([info[q] for q in ids]), encoding="utf-8")
    log(f"done: {done} queries here in {time.time() - t_start:.0f}s (audit {audit_s:.0f}s)")


if __name__ == "__main__":
    main()
