"""Design look (untracked; not a result and not filed): batch-1 serving cost of the lean sets beside the six GNN's
exact fast serving form (gnn_fast.py), in one run on the same pools.

lean_time4 (2wiki select, one thread) timed the GNN in its training form: its forward (25.0 of 30.1 ms) projects the
whole relation bank of the six, 7,067 rows x 1536 -> 32, on every query, though no 2wiki edge reads it, and computes
the shared cell's step-invariant edge terms at every step. gnn_fast.FastGNN computes those once (the bank at load,
the edge terms once per query), keeps every floating-point operation's operands and order, and scores bit for bit as
the training form. The lean MLP's sets are timed against that form, so the ratio no longer rests on the GNN's waste.

Paths per query, the order rotating query by query:
  twin+fk, gnn+fk     lean_time4's: fast compile -> pack -> the model's own forward (training form)
  gnn+fk:fast         the same compile and pack -> FastGNN (bit for bit gnn+fk's scores)
  gnn+fk:fast+ix      the same, node_projection(e_v) read from an index-time store (64 floats per node) instead of
                      projecting the 1536-wide rows per query; checked to tolerance and top-5
  gnn:floor           compile -> pack -> the GNN's input block alone: what the GNN pays before its cell and evidence
                      flow run (a reference line, not a model; it scores nothing)
  <tag>:<set>+fast, <tag>:<set>+need   lean_time4's lean paths

Checks (every query): the twin's and the GNN's scores against the look; gnn+fk:fast equals gnn+fk exactly
(torch.equal); gnn+fk:fast+ix's largest difference and top-5; lean_time4's lean checks. On the first 20 queries the
pack's node rows equal the rows the index-time store was built from.

    LEAN_TIME_THREADS=1 python outputs/mp_unified/lean_time5.py --dataset 2wiki --queries 300 \\
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
import gnn_fast as GF  # noqa: E402
import lean_fuse as LFU  # noqa: E402
import lean_time2 as T2  # noqa: E402
import lean_time3 as T3  # noqa: E402
import lean_time4 as T4  # noqa: E402
import need_compile as NC  # noqa: E402

LM, L2, L3, LT, FF = T2.LM, T2.L2, T2.L3, T2.LT, T2.FF
S6, V2 = T2.S6, T2.V2
QueryInputs = T2.QueryInputs
log, pct = T2.log, T2.pct
Lean4 = T4.Lean4
GNN_PATHS = ("gnn+fk", "gnn+fk:fast", "gnn+fk:fast+ix")


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
    fgnn = GF.FastGNN(gnn)
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
    # the GNN's index-time store: node_projection of the served rows, built over the same pool nodes as the lean store
    gstore = torch.zeros(fc.n_nodes, gnn.input.semantic.node_projection.out_features, dtype=torch.float32)
    for s0 in range(0, nodes.size, 4096):
        rr = nodes[s0:s0 + 4096]
        E_rows = np.asarray(context.nodes.read(rr))
        with torch.inference_mode():
            gstore[torch.from_numpy(rr)] = gnn.input.semantic.node_projection(torch.from_numpy(E_rows).to(torch.float32))
        En = np.asarray(E_rows, dtype=np.float32)
        En = En / np.maximum(np.linalg.norm(En, axis=1, keepdims=True), 1e-12)
        store16[rr] = (En @ R).astype(np.float16)
        if pstore is not None:
            codes[rr] = pstore.codes(En)
    fgnn_ix = GF.FastGNN(gnn, node_store=gstore)
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
    paths = ["twin+fk", "gnn+fk", "gnn+fk:fast", "gnn+fk:fast+ix", "gnn:floor"] + [sv.name + "+fast" for sv in served] + \
        [sv.name + "+need" for sv in need_sets]
    clock = time.perf_counter
    checks = {"pool": 0, "score_max_abs": {p: 0.0 for p in ("twin+fk", "gnn+fk")}, "score_off": {p: 0 for p in ("twin+fk", "gnn+fk")},
              "need_blocks_differ": {sv.name: 0 for sv in need_sets}, "need_extra_differ": {sv.name: 0 for sv in need_sets},
              "need_scores_differ": {sv.name: 0 for sv in need_sets}, "need_top5_same": {sv.name: 0 for sv in need_sets},
              "fast_score_max_abs": {sv.name: 0.0 for sv in served}, "fast_score_off": {sv.name: 0 for sv in served},
              "gnn_fast_differ": 0, "gnn_fast_top5_same": 0, "gnn_ix_max_abs": 0.0, "gnn_ix_top5_same": 0, "emb_rows_differ": 0,
              "emb_rows_checked": 0}
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
                if tag in ("twin+fk", "gnn+fk"):
                    with torch.no_grad():
                        sc = model(V2.arm_view(model, b, inputs))
                        torch.topk(sc, min(5, int(sc.shape[0])))
                elif tag == "gnn+fk:fast":
                    sc = fgnn(b)
                    torch.topk(sc, min(5, int(sc.shape[0])))
                elif tag == "gnn+fk:fast+ix":
                    sc = fgnn_ix(b, torch.from_numpy(pool))
                    torch.topk(sc, min(5, int(sc.shape[0])))
                else:
                    with torch.inference_mode():
                        fgnn._input(b)
                    sc = None
                c3 = clock()
                if tag == "gnn+fk" and checks["emb_rows_checked"] < 20:
                    checks["emb_rows_checked"] += 1
                    checks["emb_rows_differ"] += int(not np.array_equal(np.asarray(E), np.asarray(context.nodes.read(pool))))
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
                if p.startswith("twin") or p.startswith("gnn"):
                    out[p] = full_path(twin if p.startswith("twin") else gnn, p)
                elif p.endswith("+fast"):
                    out[p] = fast_path(by_name[p[:-5]])
                else:
                    out[p] = need_path(by_name[p[:-5]])
            a0, b0 = int(look0.off[i]), int(look0.off[i + 1])
            if b0 - a0 != pool.size:
                raise SystemExit(f"row {i}: the look's pool has {b0 - a0} nodes, not {pool.size}")
            st_sc = look0.score[a0:b0]
            kk = min(5, int(out["gnn+fk"].shape[0]))
            top_ref = torch.topk(out["gnn+fk"], kk).indices
            checks["gnn_fast_differ"] += int(not torch.equal(out["gnn+fk"], out["gnn+fk:fast"]))
            checks["gnn_fast_top5_same"] += int(torch.equal(top_ref, torch.topk(out["gnn+fk:fast"], kk).indices))
            checks["gnn_ix_max_abs"] = max(checks["gnn_ix_max_abs"], float((out["gnn+fk"] - out["gnn+fk:fast+ix"]).abs().max()))
            checks["gnn_ix_top5_same"] += int(torch.equal(top_ref, torch.topk(out["gnn+fk:fast+ix"], kk).indices))
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
    for den in GNN_PATHS:
        for p in paths:
            if p != den:
                ratio[f"{p}/{den}"] = pct([r[p]["total"] / r[den]["total"] for r in warm], 50)
                ratio[f"{p}/{den} (+pool)"] = pct([(r[p]["total"] + 1e-3 * r["pool_ms"]) / (r[den]["total"] + 1e-3 * r["pool_ms"])
                                                   for r in warm], 50)
    for sv in need_sets:
        ratio[f"{sv.name}+need/{sv.name}+fast"] = pct([r[sv.name + "+need"]["total"] / r[sv.name + "+fast"]["total"] for r in warm], 50)
    log(f"{name}: {len(warm)} warm queries, ms p50 / p95 (stage p50s); threads {THREADS}, own gather {own_gather}")
    for p, v in summ.items():
        log(f"  {p:22s} total {v['total']['p50']:7.2f} / {v['total']['p95']:7.2f}   " +
            ", ".join(f"{k} {x['p50']:.3f}" for k, x in v.items() if k != "total" and not k.startswith("lap_")))
    for den in GNN_PATHS:
        log(f"  ratios p50 against {den}: " + ", ".join(f"{k[:-len(den) - 1]} {v:.3f}" for k, v in ratio.items()
                                                       if "(+pool)" not in k and k.endswith("/" + den)))
    log(f"  checks: {checks}")
    res = {"look": "lean_time5", "dataset": name, "carve": "select", "queries": len(recs), "warm_excluded": a.warm, "threads": THREADS,
           "numba_threads": FF.numba.get_num_threads(), "own_gather": own_gather, "summary_ms": summ, "ratios_p50": ratio,
           "sets": {sv.name: {"blocks": sv.blocks, "active": sv.active, "compiled_extra": sv.extra, "plan": repr(sv.plan), "store": sv.store,
                              "kind": sv.kind, "folded_in_w": sv.folded.fu.in_w} for sv in served},
           "checks": checks, "models": {tag: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                                        for tag, (path, _b) in blobs.items()},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "imports_sha256": {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (T2, T3, T4, EF, LFU, NC, FF, GF)},
           "relation_bank_rows": int(gnn.relation_bank.shape[0]),
           "rows": recs, "seconds": time.time() - t_start}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
