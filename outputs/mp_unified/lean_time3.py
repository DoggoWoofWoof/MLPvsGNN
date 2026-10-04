"""Design look (untracked; not a result and not filed): batch-1 serving cost with every path in its exact fast serving
form, beside its lean_time2 form, in one run on the same pools.

lean_time2 (2wiki select, one thread) put the GNN at 88.5 ms p50, the twin at 30.4 and the lean sets at 17 to 49, and
found two avoidable costs:
  - the structural pool-edge kernel probes a 6 M-entry lookup once per stored entry of every pool row (2.2 ms p50,
    hubs to 158 k entries); edges_fast.typed_edges_gallop merges the pool with each id-sorted row instead and returns
    the same arrays (0.33 ms);
  - the lean forward runs about a dozen torch ops per block (8 to 24 ms); lean_fuse's form folds the keeps and mask
    columns into one l1 matrix and runs [X, z(X)] -> addmm -> GELU -> addmm -> GELU -> addmv (0.6 to 1.1 ms), torch's
    own GELU, the same weights.
Both are serving forms: they change no feature and no weight. The edge kernel is shared, so the twin's and the GNN's
compile take it too (paths 'twin+fk', 'gnn+fk'); the forward form is the lean model's only (the twin's and the GNN's
forwards are their torch modules, as in lean_time2). The NER and kNN kernels stay as they are: their rows are short and
the merge does not pay there (edges_fast, 3 Oct).

Paths per query, the order rotating query by query:
  twin, gnn                  lean_time2's: fast compile (old edge kernel) -> pack -> torch forward
  twin+fk, gnn+fk            the same with the merged structural edge kernel
  <tag>:<set>                lean_time2's lean path: rank lists, edges (old kernel), store gather, blocks, torch forward
  <tag>:<set>+fast           the same blocks with the merged edge kernel, then the folded forward
A set with compiled blocks also runs the fast compile (with its path's edge kernel), measured, as lean_time2 does.

Checks (every query): the twin's and the GNN's scores in both forms against the look; each fast lean path's blocks equal
its old path's blocks exactly; both lean forms' scores against the torch model on lean_mlp.batch_of's inputs from the
look; the fast form's top 5 against the old form's.

    LEAN_TIME_THREADS=1 python outputs/mp_unified/lean_time3.py --dataset 2wiki --queries 300 \
        --models l3=outputs/mp_unified/lean/l3-2w_models.pt:pick,lean,lean2s --out scratch.json
"""
import os
import sys

THREADS = int(os.environ.get("LEAN_TIME_THREADS", "1"))
os.environ["LEAN_TIME_THREADS"] = str(THREADS)              # lean_time2 and edges_fast read it at import
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
import torch.nn.functional as TF  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import edges_fast as EF  # noqa: E402
import lean_fuse as LFU  # noqa: E402
import lean_time2 as T2  # noqa: E402

LM, L2, L3, LT, FF = T2.LM, T2.L2, T2.L3, T2.LT, T2.FF
S6, V2 = T2.S6, T2.V2
FAMILIES, QueryInputs, pack_queries_v2 = T2.FAMILIES, T2.QueryInputs, T2.pack_queries_v2
log, pct = T2.log, T2.pct


def fast_typed(tindptr, tcol, trel, tdir, pool, lookup, cap):
    """fast_features' typed_edges signature, every row merged with the pool (edges_fast: same arrays)."""
    return EF.typed_edges_gallop(tindptr, tcol, trel, tdir, pool, lookup, cap, -1)


class Kernels:
    """Switches the structural edge kernel of the compiler's kernel sets (both the serial and the parallel set)."""

    def __init__(self, fc):
        self.Ks = [fc.K] if fc._K_small is fc.K else [fc.K, fc._K_small]
        self.orig = [K.typed_edges for K in self.Ks]

    def fast(self, on):
        for K, o in zip(self.Ks, self.orig):
            K.typed_edges = fast_typed if on else o


class Folded:
    """lean_fuse.Fused's folded weights, run as numpy [X, z(X)] then torch addmm -> GELU -> addmm -> GELU -> addmv."""

    def __init__(self, sv):
        self.fu = LFU.Fused(sv.model, sv.keep)
        self.A = torch.from_numpy(self.fu.A)
        self.c1 = torch.from_numpy(self.fu.c1)
        self.W2 = torch.from_numpy(self.fu.W2)
        self.b2 = torch.from_numpy(self.fu.b2)
        self.wo = torch.from_numpy(self.fu.wo)

    def __call__(self, feats, q, P, qemb16, extra_x, blocks_idx):
        fu = self.fu
        cols = []
        for b in fu.active:
            if b == "SEMB":
                cols.append((qemb16.astype(np.float32) @ fu.U)[None, :] * (P @ fu.V))
            elif b == "SEM":
                prod = P * q[None, :]
                cols.append(np.concatenate([prod, prod.sum(1, keepdims=True)], 1))
            elif b in feats:
                cols.append(feats[b])
            else:
                cols.append(extra_x[:, blocks_idx[b]])
        X = np.nan_to_num(np.concatenate(cols, 1).astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        XZ = np.concatenate([X, LFU.zscore(X)], 1)
        base = LFU.zscore(feats["_rrf"][:, None].astype(np.float32))[:, 0] * fu.base_w + fu.bo
        with torch.no_grad():
            H = TF.gelu(torch.addmm(self.c1, torch.from_numpy(XZ), self.A))
            H = TF.gelu(torch.addmm(self.b2, H, self.W2))
            sc = torch.addmv(torch.from_numpy(base), H, self.wo)
            torch.topk(sc, min(5, int(sc.shape[0])))
        return sc


def same_arr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if np.issubdtype(x.dtype, np.floating):
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--models", nargs="+", required=True, help="tag=FILE.pt:setA,setB (lean_mlp, lean_mlp2 or lean_mlp3 saves)")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--warm", type=int, default=10, help="the first queries, excluded from the summary (numba and torch warm-up)")
    ap.add_argument("--check", type=int, default=20, help="queries whose fast pack is compared with pack_queries_v2")
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
        f"threads {THREADS}, OMP_WAIT_POLICY {os.environ.get('OMP_WAIT_POLICY', '(unset)')}")
    fc = FF.compiler_for(context.stores, context.nodes, context.rel_table)
    if fc is None:
        raise SystemExit("no fast compiler for this context")
    kern = Kernels(fc)
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
            sv.folded = Folded(sv)
            served.append(sv)
            log(f"{sv.name} ({kind}, store {sv.store}): {sv.active}" + (f"; compiled blocks {sv.extra}: also runs the fast compile" if sv.extra else ""))
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
    paths = ["twin", "gnn", "twin+fk", "gnn+fk"] + [sv.name for sv in served] + [sv.name + "+fast" for sv in served]
    clock = time.perf_counter
    checks = {"pool": 0, "pack": [], "score_max_abs": {p: 0.0 for p in ("twin", "gnn", "twin+fk", "gnn+fk")},
              "score_off": {p: 0 for p in ("twin", "gnn", "twin+fk", "gnn+fk")},
              "fast_blocks_differ": {sv.name: 0 for sv in served},
              "lean_score_max_abs": {p: 0.0 for p in paths[4:]}, "lean_score_off": {p: 0 for p in paths[4:]},
              "fast_top5_same": {sv.name: 0 for sv in served}}
    recs = []
    gc.collect()
    gc.disable()
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

            def full_path(model, tag, fast):
                laps = {}
                kern.fast(fast)
                c0 = clock()
                comp, E = compile_(laps)
                if E is None:
                    E = fc._E[:pool.size]
                c1 = clock()
                kern.fast(False)
                b = LT.fast_pack(comp, E, prep.qemb[i], columns, True)
                c2 = clock()
                with torch.no_grad():
                    sc = model(V2.arm_view(model, b, inputs))
                    torch.topk(sc, min(5, int(sc.shape[0])))
                c3 = clock()
                rec[tag] = {"compile": c1 - c0, "pack": c2 - c1, "forward": c3 - c2, "total": c3 - c0,
                            **{f"lap_{k}": v for k, v in laps.items()}}
                return comp, np.array(E), sc

            def lean_path(sv, fast):
                kern.fast(fast)
                c0 = clock()
                feats, q, P, T, m_struct, m_all = lean(inp, pool, seeds, buckets, sv)
                xs = None
                if sv.extra:
                    laps = {}
                    cc = clock()
                    comp, _E = compile_(laps)
                    xs = comp.scalars[:, columns].astype(np.float16).astype(np.float32)
                    T["compile_for_extra"] = clock() - cc
                kern.fast(False)
                c1 = clock()
                if fast:
                    sc = sv.folded(feats, q, P, qemb16, xs, refs[sv.tag].blocks_idx)
                else:
                    f, nq_, bz = T2.lean_inputs(sv, feats, q, P, qemb16, xs, refs[sv.tag].blocks_idx)
                    with torch.no_grad():
                        sc = sv.model(f, sv.keep_t, nq_, 1, bz)
                        torch.topk(sc, min(5, int(sc.shape[0])))
                c2 = clock()
                rec[sv.name + ("+fast" if fast else "")] = {**T, "forward": c2 - c1, "total": c2 - c0, "edges_struct": m_struct,
                                                            "edges_all": m_all}
                return feats, sc

            k = i % len(paths)
            for p in paths[k:] + paths[:k]:
                if p in ("twin", "gnn", "twin+fk", "gnn+fk"):
                    out[p] = full_path(twin if p.startswith("twin") else gnn, p, p.endswith("+fk"))
                elif p.endswith("+fast"):
                    out[p] = lean_path(by_name[p[:-5]], True)
                else:
                    out[p] = lean_path(by_name[p], False)
            a0, b0 = int(look0.off[i]), int(look0.off[i + 1])
            if b0 - a0 != pool.size:
                raise SystemExit(f"row {i}: the look's pool has {b0 - a0} nodes, not {pool.size}")
            st_sc = look0.score[a0:b0]
            for tag in ("twin", "gnn", "twin+fk", "gnn+fk"):
                col = 0 if tag.startswith("twin") else 3
                dmax = float(np.abs(out[tag][2].numpy() - st_sc[:, col]).max())
                checks["score_max_abs"][tag] = max(checks["score_max_abs"][tag], dmax)
                checks["score_off"][tag] += int(dmax > 1e-3)
            comp_g, E_g = out["gnn+fk"][0], out["gnn+fk"][1]
            if i < a.check:
                ref = pack_queries_v2([{"pool": comp_g.pool, "x": comp_g.scalars[:, columns], "seedw": comp_g.seedw, "qemb": prep.qemb[i],
                                        "seeds": comp_g.seeds_local, "gold": LT.EMPTY_GOLD, "emb": E_g}], context)
                bad = LT.batch_problems(LT.fast_pack(comp_g, E_g, prep.qemb[i], columns, True), ref, True)
                if bad:
                    checks["pack"].append({"row": i, "fields": bad})
            for sv in served:
                f_old, sc_old = out[sv.name]
                f_new, sc_new = out[sv.name + "+fast"]
                if any(not same_arr(f_old[b], f_new[b]) for b in f_old):
                    checks["fast_blocks_differ"][sv.name] += 1
                look = refs[sv.tag]
                fl, nq_, bz, _g, _ix = LM.batch_of(look, np.asarray([i]), sv.blocks)
                with torch.no_grad():
                    ref_sc = sv.model(fl, sv.keep_t, nq_, 1, bz)
                for p, sc in ((sv.name, sc_old), (sv.name + "+fast", sc_new)):
                    dmax = float((sc - ref_sc).abs().max())
                    checks["lean_score_max_abs"][p] = max(checks["lean_score_max_abs"][p], dmax)
                    checks["lean_score_off"][p] += int(dmax > 1e-3)
                kk = min(5, int(sc_old.shape[0]))
                same = torch.equal(torch.topk(sc_old, kk).indices, torch.topk(sc_new, kk).indices)
                checks["fast_top5_same"][sv.name] += int(same)
            recs.append(rec)
            if i % 50 == 0:
                log(f"  q{i}: n {pool.size}; ms " + ", ".join(f"{p} {1e3 * rec[p]['total']:.2f}" for p in paths))
    finally:
        gc.enable()
        kern.fast(False)
    warm = recs[a.warm:]
    summ = {}
    for p in paths:
        stages = sorted({k for r in warm for k in r[p] if not k.startswith("edges_")})
        summ[p] = {k: {"p50": 1e3 * pct([r[p].get(k, 0.0) for r in warm], 50), "p95": 1e3 * pct([r[p].get(k, 0.0) for r in warm], 95),
                       "mean": 1e3 * float(np.mean([r[p].get(k, 0.0) for r in warm]))} for k in stages}
    summ["pool"] = {"total": {"p50": pct([r["pool_ms"] for r in warm], 50), "p95": pct([r["pool_ms"] for r in warm], 95),
                              "mean": float(np.mean([r["pool_ms"] for r in warm]))}}
    ratio = {}
    for p in paths:
        for ref_p in ("gnn", "gnn+fk"):
            if p != ref_p:
                ratio[f"{p}/{ref_p}"] = pct([r[p]["total"] / r[ref_p]["total"] for r in warm], 50)
        ratio[f"{p}/gnn+fk (+pool)"] = pct([(r[p]["total"] + 1e-3 * r["pool_ms"]) / (r["gnn+fk"]["total"] + 1e-3 * r["pool_ms"])
                                            for r in warm], 50)
    log(f"{name}: {len(warm)} warm queries, ms p50 / p95 (stage p50s); threads {THREADS}, own gather {own_gather}")
    for p, v in summ.items():
        log(f"  {p:20s} total {v['total']['p50']:7.2f} / {v['total']['p95']:7.2f}   " +
            ", ".join(f"{k} {x['p50']:.2f}" for k, x in v.items() if k != "total" and not k.startswith("lap_")))
    log("  ratios p50 (fast forms): " + ", ".join(f"{k} {v:.3f}" for k, v in ratio.items() if "+fast/gnn+fk" in k or k.startswith("twin+fk/")))
    log(f"  checks: {checks}")
    res = {"look": "lean_time3", "dataset": name, "carve": "select", "queries": len(recs), "warm_excluded": a.warm, "threads": THREADS,
           "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"), "numba_threads": FF.numba.get_num_threads(), "own_gather": own_gather,
           "summary_ms": summ, "ratios_p50": ratio,
           "sets": {sv.name: {"blocks": sv.blocks, "active": sv.active, "compiled_extra": sv.extra, "store": sv.store, "kind": sv.kind,
                              "folded_in_w": sv.folded.fu.in_w} for sv in served},
           "checks": checks, "models": {tag: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                                        for tag, (path, _b) in blobs.items()},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "imports_sha256": {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (T2, EF, LFU, FF)},
           "rows": recs, "seconds": time.time() - t_start}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
