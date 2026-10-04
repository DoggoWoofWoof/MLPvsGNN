"""Design look (untracked; not a result and not filed): batch-1 serving cost of the lean sets whose compiled blocks
run the need-only compile (need_compile.py) instead of the whole fast compile, beside lean_time3's fast forms, in one
run on the same pools.

lean_time3 (2wiki select, one thread) served pick at 0.228 of the GNN (gnn+fk), 3.3 ms of its 6.6 being the whole
fast compile run for its three compiled blocks (dense_cos, topo_STRUCT, depth_STRUCT). The need-only compile runs
only the groups those blocks read and shares the pool edges with the lean blocks; every column it fills is the whole
compile's bit for bit (need_check.py), so the scores are the same scores.

Paths per query, the order rotating query by query:
  twin+fk, gnn+fk     lean_time3's: fast compile (merged structural edge kernel) -> pack -> torch forward
  <tag>:<set>+fast    lean_time3's: lean path (merged edge kernel), the whole fast compile when the set has compiled
                      blocks, the folded forward
  <tag>:<set>+need    one pass: node embeddings and dense_cos when a compiled group reads them, the rank lists (with
                      dense_cos when asked), the pool edges once (structural; NER and kNN when a lean block or a group
                      reads them), the asked compiled groups, the store gather, the lean blocks, the folded forward.
                      Only sets with compiled blocks have this path (for the others it is the +fast path).

Checks (every query): the twin's and the GNN's scores against the look; each +need path's lean blocks and compiled
columns equal its +fast path's exactly, and its scores equal them exactly (torch.equal); the +fast scores against the
torch model on lean_mlp.batch_of's inputs from the look.

    LEAN_TIME_THREADS=1 python outputs/mp_unified/lean_time4.py --dataset 2wiki --queries 300 \
        --models l3=outputs/mp_unified/lean/l3-2w_models.pt:pick,drop/pick,full,lean2,lean2s,lean --out scratch.json
"""
import os
import sys

THREADS = int(os.environ.get("LEAN_TIME_THREADS", "1"))
os.environ["LEAN_TIME_THREADS"] = str(THREADS)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(THREADS)
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import edges_fast as EF  # noqa: E402
import lean_fuse as LFU  # noqa: E402
import lean_time2 as T2  # noqa: E402
import lean_time3 as T3  # noqa: E402
import need_compile as NC  # noqa: E402

LM, L2, L3, LT, FF = T2.LM, T2.L2, T2.L3, T2.LT, T2.FF
S6, V2 = T2.S6, T2.V2
FWD, BWD = T2.FWD, T2.BWD
NK, EDGE_USERS, STORE_USERS = T2.NK, T2.EDGE_USERS, T2.STORE_USERS
QueryInputs = T2.QueryInputs
log, pct = T2.log, T2.pct


class Lean4:
    """lean_time2.Lean's read path with the need-only compile folded in: one lookup, one set of pool edges."""

    def __init__(self, fc, store16, R, codes, pstore):
        self.fc, self.store16, self.R, self.codes, self.pstore = fc, store16, R, codes, pstore
        self.nc = NC.NeedCompiler(fc)
        self.rank_idx = np.asarray([V2.IDX[c] for c in LM.SPLIT["rank"]], dtype=np.int64)
        self.rrf_col, self.dr_col = V2.IDX["rrf"], V2.IDX["dense_rr"]

    def __call__(self, inp, pool, seeds, buckets, sv, plan, columns, embeddings=None):
        nc, need = self.nc, sv.need
        clock = time.perf_counter
        T = {}
        t = [clock()]

        def lap(g):
            now = clock()
            T[g] = T.get(g, 0.0) + now - t[0]
            t[0] = now

        n = int(pool.size)
        xs = None
        st = nc.begin(pool, seeds)
        try:
            sl = st["seeds_local"]
            X = np.zeros((n, V2.N_COLUMNS), dtype=np.float32)
            if plan.need_E:
                nc.embeddings(st, inp, embeddings)
                lap("E")
            nc.retrieval(st, inp, X)
            out = {"rank": X[:, self.rank_idx].astype(np.float16).astype(np.float32),
                   "_rrf": X[:, self.rrf_col].astype(np.float16).astype(np.float32),
                   "_dscore": np.where(st["d_in"], st["d_score"], 0.0).astype(np.float32), "_din": st["d_in"].copy()}
            lap("rank")
            eu = ev = np.empty(0, np.int64)
            efam = np.empty(0, np.int8)
            efwd = ebwd = np.empty(0, np.bool_)
            if need & EDGE_USERS or plan.any:
                lean_nk = bool(need & NK)
                edges = nc.edges(st, lean_nk or plan.nk)
                pu, pv, attr = edges["structural"]
                us, vs, fs = [pu], [pv], [np.zeros(pu.size, np.int8)]
                fws, bws = [attr[:, FWD] > 0.5], [attr[:, BWD] > 0.5]
                if lean_nk:
                    for f_i, fam in ((1, "ner"), (2, "knn")):
                        u_, v_, _ = edges[fam]
                        us.append(u_), vs.append(v_), fs.append(np.full(u_.size, f_i, np.int8))
                        fws.append(np.zeros(u_.size, np.bool_)), bws.append(np.zeros(u_.size, np.bool_))
                eu, ev = np.concatenate(us).astype(np.int64), np.concatenate(vs).astype(np.int64)
                efam, efwd, ebwd = np.concatenate(fs), np.concatenate(fws), np.concatenate(bws)
                lap("edges")
            if plan.any:
                nc.groups(st, plan, X, lap)
                xs = X[:, columns].astype(np.float16).astype(np.float32)
                lap("xs")
        finally:
            nc.end(st)
        t2 = clock()
        unit_store = sv.store != "pca256"
        if need & STORE_USERS:
            G = self.store16[pool] if unit_store else self.codes[pool]
            t3 = clock()
            T["store_gather"] = t3 - t2
            if unit_store:
                Pf = G.astype(np.float32)
                P = Pf / np.maximum(np.linalg.norm(Pf, axis=1), 1e-12)[:, None]
                q = np.asarray(inp.q, np.float32) @ self.R
                q /= max(float(np.linalg.norm(q)), 1e-12)
            else:
                P = self.pstore.decode(G)
                q = self.pstore.query(np.asarray(inp.q))
            cos = P @ q
        else:
            t3 = clock()
            P, q, cos = np.zeros((n, sv.dim), np.float32), np.zeros(sv.dim, np.float32), np.zeros(n, np.float32)
        t4 = clock()
        T["blk_SEM"] = t4 - t3
        valid = sl >= 0
        S, Bk = sl[valid], buckets[valid]
        st_ = efam == 0
        u_s, v_s = eu[st_], ev[st_]
        lap_t = [clock()]

        def done(b):
            now = clock()
            T[f"blk_{b}"] = T.get(f"blk_{b}", 0.0) + now - lap_t[0]
            lap_t[0] = now

        if "SEED" in need:
            out["SEED"] = T2.seed_blk(P, S, Bk, unit_store)
            done("SEED")
        if "WALK" in need:
            out["WALK"] = T2.walk_blk(n, S, Bk, u_s, v_s, efwd[st_], ebwd[st_])
            done("WALK")
        if "NBR" in need:
            out["NBR"] = L3.nbr_of(n, P, q, cos, u_s, v_s)
            done("NBR")
        if "DLIST" in need:
            listed = X[:, self.dr_col] > 0
            out["DLIST"] = np.stack([np.where(listed, out["_dscore"], 0.0), listed, np.where(listed, 0.0, cos)], 1).astype(np.float32)
            done("DLIST")
        if need & {"DISTS", "DISTF", "NBR2S", "NBR2F"}:
            rrf16 = out["_rrf"].astype(np.float64)
            s_seed = rrf16[S] / max(float(rrf16.max()), 1e-12) if S.size else np.zeros(0)
            for sel, d_name, n2_name in ((st_, "DISTS", "NBR2S"), (np.ones_like(st_), "DISTF", "NBR2F")):
                if not need & {d_name, n2_name}:
                    continue
                pu_, pv_ = L2.pairs(n, eu[sel], ev[sel])
                done(f"pairs_{d_name[-1]}")
                if d_name in need:
                    out[d_name] = L2.dist_fast(n, P, S, s_seed, pu_, pv_)
                    done(d_name)
                if n2_name in need:
                    out[n2_name] = L2.nbr2_fast(n, cos, pu_, pv_)
                    done(n2_name)
        if "WALKF" in need or "NBRF" in need:
            sym = efam > 0
            if "WALKF" in need:
                s0 = np.zeros(n)
                s0[S] = 1.0
                wx = np.zeros((n, 2), np.float32)
                for j, f in enumerate((1, 2)):
                    m = efam == f
                    wx[:, j] = np.log1p(np.bincount(ev[m], weights=s0[eu[m]], minlength=n))
                out["WALKF"] = np.concatenate([T2.walk_blk(n, S, Bk, eu, ev, efwd | sym, ebwd | sym), wx], 1)
                done("WALKF")
            if "NBRF" in need:
                nx = np.zeros((n, 2), np.float32)
                for j, f in enumerate((1, 2)):
                    nx[:, j] = np.bincount(ev[efam == f], minlength=n) > 0
                out["NBRF"] = np.concatenate([L3.nbr_of(n, P, q, cos, eu, ev), nx], 1)
                done("NBRF")
        for b in [k for k in out if not k.startswith("_") and k != "rank"]:
            out[b] = out[b].astype(np.float16).astype(np.float32)
        P16 = P.astype(np.float16).astype(np.float32) if unit_store else P
        return out, q, P16, T, int(u_s.size), int(eu.size), xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--models", nargs="+", required=True, help="tag=FILE.pt:setA,setB (lean_mlp, lean_mlp2 or lean_mlp3 saves)")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--warm", type=int, default=10, help="the first queries, excluded from the summary (numba and torch warm-up)")
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(THREADS)
    t_start = time.time()
    name = a.dataset
    S6.pair_verify(S6.SIX)
    cfg, cfg_m3b, cfg_h = V2.load_configs()
    op = S6.pair_open(S6.SIX, cfg, cfg_m3b, name)
    inputs, m3b_compile = op.inputs, op.m3b_compile
    context, ds = op.contexts[name], op.handles[name]
    if context.rel_table is not None:
        raise SystemExit(f"{name}: typed relations; this look packs untyped graphs only")
    models = S6.pair_models(S6.SIX, inputs, op.bank)
    twin, gnn = models["twin0"], models["gnn0"]
    del models
    m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
    m3a = op.pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][name]["construction"]
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, name, "select", cfg_m3b, cfg_h, m3a, positions)
    del positions
    gc.collect()
    rows = np.arange(min(a.queries, len(pop.ids)), dtype=np.int64)
    pop.ids, pop.idx, pop.golds = [pop.ids[i] for i in rows], pop.idx[rows], [pop.golds[i] for i in rows]
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    builder = LT.PoolBuilder(construction, cfg_h, context.stores, m3a, m3b_contract, m3b_compile)
    columns = inputs["column_indices"]
    nq_all = len(prep.pools)
    log(f"{name}: {nq_all} select queries prepared ({time.time() - t_start:.0f}s); pools mean {np.mean([p.size for p in prep.pools]):.0f}; "
        f"threads {THREADS}")
    fc = FF.compiler_for(context.stores, context.nodes, context.rel_table)
    if fc is None:
        raise SystemExit("no fast compiler for this context")
    kern = T3.Kernels(fc)
    own_gather = fc._h16 is not None or fc._shards is not None
    served, blobs = [], {}
    pca_basis = None
    for tag, path, sets in T2.parse_models(a.models):
        blob = torch.load(path, weights_only=False)
        blobs[tag] = (path, blob)
        kind = "lean_mlp3" if blob.get("store") else ("lean_mlp2" if any(b in L2.NEW for b in blob.get("present", [])) else "lean_mlp")
        if blob.get("store") == "pca256":
            if pca_basis is not None and not np.array_equal(pca_basis["V"], blob["basis"]["V"]):
                raise SystemExit("two pca256 model files with different bases")
            pca_basis = blob["basis"]
        for s in sets:
            sv = T2.Served(tag, path, s, blob, kind)
            sv.folded = T3.Folded(sv)
            sv.plan = NC.Plan(sv.extra)
            if sv.plan.unsupported:
                raise SystemExit(f"{sv.name}: compiled blocks {sv.plan.unsupported} are not mirrored")
            served.append(sv)
            log(f"{sv.name} ({kind}): {sv.active}; {sv.plan}")
    R = LM.projection()
    nodes = np.unique(np.concatenate(prep.pools))
    store16 = np.zeros((fc.n_nodes, LM.PROJ_DIM), dtype=np.float16)
    pstore = L3.Store(pca_basis) if pca_basis is not None else None
    codes = np.zeros((fc.n_nodes, L3.STORE_K), dtype=np.int8) if pstore is not None else None
    for s0 in range(0, nodes.size, 4096):
        rr = nodes[s0:s0 + 4096]
        En = np.asarray(context.nodes.read(rr), dtype=np.float32)
        En = En / np.maximum(np.linalg.norm(En, axis=1, keepdims=True), 1e-12)
        store16[rr] = (En @ R).astype(np.float16)
        if pstore is not None:
            codes[rr] = pstore.codes(En)
    lean = T2.Lean(fc, store16, R, codes, pstore)
    lean4 = Lean4(fc, store16, R, codes, pstore)
    refs = {}
    for tag, (path, blob) in blobs.items():
        if blob.get("store") == "pca256":
            refs[tag] = L3.Carve3(name, "select", pstore, context.nodes, limit=nq_all)
        elif any(b in L2.NEW for b in blob.get("present", [])):
            refs[tag] = L2.Carve2(name, "select", limit=nq_all)
        else:
            refs[tag] = LM.Carve(name, "select", limit=nq_all)
        if refs[tag].rows != nq_all:
            raise SystemExit(f"the select look holds {refs[tag].rows} of the {nq_all} rows")
    look0 = next(iter(refs.values()))
    by_name = {sv.name: sv for sv in served}
    need_sets = [sv for sv in served if sv.plan.any]
    paths = ["twin+fk", "gnn+fk"] + [sv.name + "+fast" for sv in served] + [sv.name + "+need" for sv in need_sets]
    clock = time.perf_counter
    checks = {"pool": 0, "score_max_abs": {p: 0.0 for p in ("twin+fk", "gnn+fk")}, "score_off": {p: 0 for p in ("twin+fk", "gnn+fk")},
              "need_blocks_differ": {sv.name: 0 for sv in need_sets}, "need_extra_differ": {sv.name: 0 for sv in need_sets},
              "need_scores_differ": {sv.name: 0 for sv in need_sets}, "need_top5_same": {sv.name: 0 for sv in need_sets},
              "fast_score_max_abs": {sv.name: 0.0 for sv in served}, "fast_score_off": {sv.name: 0 for sv in served}}
    recs = []
    gc.collect()
    gc.disable()
    kern.fast(True)
    try:
        for i in range(nq_all):
            d_ids, s_ids = np.asarray(prep.dense_ids[i]), np.asarray(prep.splade_ids[i])
            t0 = clock()
            pool, seeds = builder(d_ids, s_ids)
            t_pool = clock() - t0
            if not (np.array_equal(pool, prep.pools[i]) and np.array_equal(seeds, np.asarray(prep.seeds[i], dtype=np.int64))):
                checks["pool"] += 1
            pool = np.asarray(pool, dtype=np.int64)
            seeds = np.asarray(seeds, dtype=np.int64)
            inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
            top = {int(d_ids[0]), int(s_ids[0])}
            buckets = np.asarray([0 if int(x) in top else 1 for x in seeds], dtype=np.int64)
            qemb16 = np.asarray(prep.qemb[i]).astype(np.float16)
            rec = {"i": i, "n": int(pool.size), "pool_ms": 1e3 * t_pool}
            out = {}

            def compile_(laps):
                if own_gather:
                    return fc.compile(inp, pool, seeds, timings=laps, v2=True), None
                E = context.nodes.read(pool)
                return fc.compile(inp, pool, seeds, embeddings=E, timings=laps, v2=True), E

            def full_path(model, tag):
                laps = {}
                c0 = clock()
                comp, E = compile_(laps)
                if E is None:
                    E = fc._E[:pool.size]
                c1 = clock()
                b = LT.fast_pack(comp, E, prep.qemb[i], columns, True)
                c2 = clock()
                with torch.no_grad():
                    sc = model(V2.arm_view(model, b, inputs))
                    torch.topk(sc, min(5, int(sc.shape[0])))
                c3 = clock()
                rec[tag] = {"compile": c1 - c0, "pack": c2 - c1, "forward": c3 - c2, "total": c3 - c0,
                            **{f"lap_{k}": v for k, v in laps.items()}}
                return sc

            def fast_path(sv):
                c0 = clock()
                feats, q, P, T, m_struct, m_all = lean(inp, pool, seeds, buckets, sv)
                xs = None
                if sv.extra:
                    cc = clock()
                    comp, _E = compile_({})
                    xs = comp.scalars[:, columns].astype(np.float16).astype(np.float32)
                    T["compile_for_extra"] = clock() - cc
                c1 = clock()
                sc = sv.folded(feats, q, P, qemb16, xs, refs[sv.tag].blocks_idx)
                c2 = clock()
                rec[sv.name + "+fast"] = {**T, "forward": c2 - c1, "total": c2 - c0}
                return feats, xs, sc

            def need_path(sv):
                c0 = clock()
                emb = None if own_gather else context.nodes.read(pool)
                feats, q, P, T, m_struct, m_all, xs = lean4(inp, pool, seeds, buckets, sv, sv.plan, columns, emb)
                c1 = clock()
                sc = sv.folded(feats, q, P, qemb16, xs, refs[sv.tag].blocks_idx)
                c2 = clock()
                rec[sv.name + "+need"] = {**T, "forward": c2 - c1, "total": c2 - c0}
                fu = sv.folded.fu
                if "SEMB" in fu.active:
                    # SEMB's share of the forward, timed apart and outside the total (lean_price's forward term)
                    cs = clock()
                    _semb = (qemb16.astype(np.float32) @ fu.U)[None, :] * (P @ fu.V)
                    rec[sv.name + "+need"]["semb_side"] = clock() - cs
                return feats, xs, sc

            k = i % len(paths)
            for p in paths[k:] + paths[:k]:
                if p in ("twin+fk", "gnn+fk"):
                    out[p] = full_path(twin if p.startswith("twin") else gnn, p)
                elif p.endswith("+fast"):
                    out[p] = fast_path(by_name[p[:-5]])
                else:
                    out[p] = need_path(by_name[p[:-5]])
            a0, b0 = int(look0.off[i]), int(look0.off[i + 1])
            if b0 - a0 != pool.size:
                raise SystemExit(f"row {i}: the look's pool has {b0 - a0} nodes, not {pool.size}")
            st_sc = look0.score[a0:b0]
            for tag in ("twin+fk", "gnn+fk"):
                col = 0 if tag.startswith("twin") else 3
                dmax = float(np.abs(out[tag].numpy() - st_sc[:, col]).max())
                checks["score_max_abs"][tag] = max(checks["score_max_abs"][tag], dmax)
                checks["score_off"][tag] += int(dmax > 1e-3)
            for sv in served:
                f_fast, xs_fast, sc_fast = out[sv.name + "+fast"]
                look = refs[sv.tag]
                fl, nq_, bz, _g, _ix = LM.batch_of(look, np.asarray([i]), sv.blocks)
                with torch.no_grad():
                    ref_sc = sv.model(fl, sv.keep_t, nq_, 1, bz)
                dmax = float((sc_fast - ref_sc).abs().max())
                checks["fast_score_max_abs"][sv.name] = max(checks["fast_score_max_abs"][sv.name], dmax)
                checks["fast_score_off"][sv.name] += int(dmax > 1e-3)
                if not sv.plan.any:
                    continue
                f_need, xs_need, sc_need = out[sv.name + "+need"]
                if set(f_fast) != set(f_need) or any(not T3.same_arr(f_fast[b], f_need[b]) for b in f_fast):
                    checks["need_blocks_differ"][sv.name] += 1
                cols = np.concatenate([refs[sv.tag].blocks_idx[b] for b in sv.extra])
                if not T3.same_arr(xs_fast[:, cols], xs_need[:, cols]):
                    checks["need_extra_differ"][sv.name] += 1
                checks["need_scores_differ"][sv.name] += int(not torch.equal(sc_fast, sc_need))
                kk = min(5, int(sc_fast.shape[0]))
                checks["need_top5_same"][sv.name] += int(torch.equal(torch.topk(sc_fast, kk).indices, torch.topk(sc_need, kk).indices))
            recs.append(rec)
            if i % 50 == 0:
                log(f"  q{i}: n {pool.size}; ms " + ", ".join(f"{p} {1e3 * rec[p]['total']:.2f}" for p in paths))
    finally:
        gc.enable()
        kern.fast(False)
    warm = recs[a.warm:]
    summ = {}
    for p in paths:
        stages = sorted({k for r in warm for k in r[p]})
        summ[p] = {k: {"p50": 1e3 * pct([r[p].get(k, 0.0) for r in warm], 50), "p95": 1e3 * pct([r[p].get(k, 0.0) for r in warm], 95),
                       "mean": 1e3 * float(np.mean([r[p].get(k, 0.0) for r in warm]))} for k in stages}
    summ["pool"] = {"total": {"p50": pct([r["pool_ms"] for r in warm], 50), "p95": pct([r["pool_ms"] for r in warm], 95),
                              "mean": float(np.mean([r["pool_ms"] for r in warm]))}}
    ratio = {}
    for p in paths:
        if p != "gnn+fk":
            ratio[f"{p}/gnn+fk"] = pct([r[p]["total"] / r["gnn+fk"]["total"] for r in warm], 50)
            ratio[f"{p}/gnn+fk (+pool)"] = pct([(r[p]["total"] + 1e-3 * r["pool_ms"]) / (r["gnn+fk"]["total"] + 1e-3 * r["pool_ms"])
                                                for r in warm], 50)
    for sv in need_sets:
        ratio[f"{sv.name}+need/{sv.name}+fast"] = pct([r[sv.name + "+need"]["total"] / r[sv.name + "+fast"]["total"] for r in warm], 50)
    log(f"{name}: {len(warm)} warm queries, ms p50 / p95 (stage p50s); threads {THREADS}, own gather {own_gather}")
    for p, v in summ.items():
        log(f"  {p:22s} total {v['total']['p50']:7.2f} / {v['total']['p95']:7.2f}   " +
            ", ".join(f"{k} {x['p50']:.3f}" for k, x in v.items() if k != "total" and not k.startswith("lap_")))
    log("  ratios p50: " + ", ".join(f"{k} {v:.3f}" for k, v in ratio.items() if "(+pool)" not in k))
    log(f"  checks: {checks}")
    res = {"look": "lean_time4", "dataset": name, "carve": "select", "queries": len(recs), "warm_excluded": a.warm, "threads": THREADS,
           "numba_threads": FF.numba.get_num_threads(), "own_gather": own_gather, "summary_ms": summ, "ratios_p50": ratio,
           "sets": {sv.name: {"blocks": sv.blocks, "active": sv.active, "compiled_extra": sv.extra, "plan": repr(sv.plan), "store": sv.store,
                              "kind": sv.kind, "folded_in_w": sv.folded.fu.in_w} for sv in served},
           "checks": checks, "models": {tag: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                                        for tag, (path, _b) in blobs.items()},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "imports_sha256": {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (T2, T3, EF, LFU, NC, FF)},
           "rows": recs, "seconds": time.time() - t_start}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
