"""Design look (untracked; not a result and not filed): AW's lean sets fused (lean_fast9), beside lean_fast8's fused
sets and the six GNN's exact fast serving form (gnn_fast.FastGNN), at batch 1, on the same pools, under three cache
regimes. lean_time8 with AW: a set whose blocks include AW is served by lean_fast9.FusedLean9, any other set by
lean_fast8.FusedLean8.

Per query, the path order rotating query by query (lean_time7's setup; batch 1 only):
  cold      each path once, in the rotated order ("first" summaries keep the queries on which the path ran first)
  resident  each path again, a --sweep-mb buffer (default 48 MB, above the last-level cache) rewritten before each
  warm      each path twice more back to back, the second timed
Paths:
  gnn:fast        fast compile -> pack -> FastGNN (the six GNN's forward bit for bit)
  gnn:fast+ix     the same, node_projection(e_v) read from an index-time store (64 floats per node)
  <tag>:<set>+f8  lean_fast8.FusedLean8 -> lean_time6.fold_forward (sets without AW)
  <tag>:<set>+f9  lean_fast9.FusedLean9 -> lean_time6.fold_forward (sets with AW)

Index time, untimed per query and reported: the graph's anchor ranks per typed-CSR entry (lean_fast9.build_tlab, from
lean_mlp4.Labels) and each AW model's label table (lean_mlp4.codes_of with the save's code basis, through the model's
P, its OTHER vector and its free rows).

Checks, every query, untimed: the rebuilt pool equals the prepared pool; gnn:fast's scores against the look's stored
scores (1e-3) and gnn:fast+ix's largest difference from them; per set, the lean_time6 form (lean_time2's lean path ->
fold_rows -> fold_forward) as the reference:
  without AW: its scores against the model on lean_mlp.batch_of's inputs from the look (1e-3); the +f8 path's XZ, base
              and scores equal to it bit for bit; FusedLean9 (run untimed) equal to FusedLean8 bit for bit;
  with AW:    AW's training form (lean_mlp4.Carve4's walks of the look's stored edges and lean_mlp4.Labels' lookups,
              through the model's AWNet) against the fused AW columns (relative to max(1, |x|)); the reference with the
              training form's AW against the model on lean_mlp4.batch_of4's inputs (1e-3); the reference with the
              fused AW columns equal to the +f9 path's XZ, base and scores bit for bit; the +f9 path's scores against
              the reference with the training form's AW (largest difference, top 5 the same);
every path's scores equal across its four runs (torch.equal).

    python outputs/mp_unified/lean_time9.py --threads 1 --dataset 2wiki --queries 300 \\
        --models st=outputs/mp_unified/lean/l4-st_models.pt:p5,p5A,n5A l3=outputs/mp_unified/lean/l3-2w_models.pt:lean2s \\
        --out outputs/mp_unified/lean/time9-2w.json
"""
import os
import sys


def _threads(argv):
    for i, x in enumerate(argv):
        if x == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if x.startswith("--threads="):
            return int(x.split("=", 1)[1])
    return 1


THREADS = _threads(sys.argv)
os.environ["LEAN_TIME_THREADS"] = str(THREADS)        # lean_time, lean_time2, lean_time4 and lean_fast7 read it at import
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
import lean_fast7 as F7  # noqa: E402
import lean_fast8 as F8  # noqa: E402
import lean_fast9 as F9  # noqa: E402
import lean_fuse as LFU  # noqa: E402
import lean_mlp4 as L4  # noqa: E402
import lean_time2 as T2  # noqa: E402
import lean_time3 as T3  # noqa: E402
import lean_time6 as T6  # noqa: E402

LM, L2, L3, LT, FF = T2.LM, T2.L2, T2.L3, T2.LT, T2.FF
S6, V2 = T2.S6, T2.V2
QueryInputs = T2.QueryInputs
log, pct = T2.log, T2.pct
REF = "gnn:fast"
GNN_FAMILY = ("gnn:fast", "gnn:fast+ix")
REGIMES = ("cold", "resident", "warm")
PARTS = ("features", "forward", "total")
AW_TOL = 1e-4


def bits_equal(x, y):
    return x.shape == y.shape and x.dtype == y.dtype and np.array_equal(np.ascontiguousarray(x).view(np.uint32),
                                                                        np.ascontiguousarray(y).view(np.uint32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1, help="BLAS, numba and torch threads (read at import, before numpy loads)")
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--models", nargs="+", required=True, help="tag=FILE.pt:setA,setB (int8 PCA store saves)")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--warm", type=int, default=10, help="the first queries: excluded from every summary")
    ap.add_argument("--sweep-mb", type=float, default=48.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.threads != THREADS:
        raise SystemExit(f"--threads {a.threads}, but the thread variables were set to {THREADS} at import")
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
    gnn = models["gnn0"]
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
    if nq_all <= a.warm:
        raise SystemExit(f"{nq_all} queries, {a.warm} of them warm-up")
    log(f"{name}: {nq_all} select queries prepared ({time.time() - t_start:.0f}s); pools mean {np.mean([p.size for p in prep.pools]):.0f}; "
        f"threads {THREADS}")
    fc = FF.compiler_for(context.stores, context.nodes, context.rel_table)
    if fc is None:
        raise SystemExit("no fast compiler for this context")
    kern = T3.Kernels(fc)
    own_gather = fc._h16 is not None or fc._shards is not None

    # ── the lean models; one int8 store per PCA basis ──
    served, blobs, bases = [], {}, {}
    for tag, path, sets in T2.parse_models(a.models):
        blob = torch.load(path, weights_only=False)
        if blob.get("store") != "pca256":
            raise SystemExit(f"{tag}: store {blob.get('store')}; the fused path serves the int8 PCA store only")
        blobs[tag] = (path, blob)
        bb = blob["basis"]
        key = hashlib.sha256(b"".join(np.ascontiguousarray(bb[k], dtype=np.float32).tobytes() for k in ("m", "V", "w"))).hexdigest()[:12]
        bd = bases.setdefault(key, {"basis": bb, "tags": [], "aw_tag": None})
        bd["tags"].append(tag)
        is4 = "aw_cfg" in blob
        for s in sets:
            sv = F9.Served4(tag, path, s, blob) if is4 else T2.Served(tag, path, s, blob, "lean_mlp3")
            if sv.extra:
                raise SystemExit(f"{sv.name}: compiled blocks {sv.extra}; the fused path computes none")
            sv.folded = T3.Folded(sv)
            sv.key = key
            sv.aw_set = "AW" in sv.need
            if sv.aw_set:
                if bd["aw_tag"] not in (None, tag):
                    raise SystemExit(f"{sv.name}: AW sets from two saves on one PCA basis ({bd['aw_tag']}, {tag})")
                bd["aw_tag"] = tag
            served.append(sv)
            log(f"{sv.name} (basis {key}): {sv.active}")

    # ── index time: the stores, over the nodes the pools touch; AW's label tables ──
    nodes = np.unique(np.concatenate(prep.pools))
    for bd in bases.values():
        bd["store"] = L3.Store(bd["basis"])
        bd["codes"] = np.zeros((fc.n_nodes, L3.STORE_K), dtype=np.int8)
    gstore = torch.zeros(nodes.size, gnn.input.semantic.node_projection.out_features, dtype=torch.float32)
    clock = time.perf_counter
    for s0 in range(0, nodes.size, 4096):
        rr = nodes[s0:s0 + 4096]
        E_rows = np.asarray(context.nodes.read(rr))
        with torch.inference_mode():
            gstore[s0:s0 + rr.size] = gnn.input.semantic.node_projection(torch.from_numpy(E_rows).to(torch.float32))
        En = np.asarray(E_rows, dtype=np.float32)
        En = En / np.maximum(np.linalg.norm(En, axis=1, keepdims=True), 1e-12)
        for bd in bases.values():
            bd["codes"][rr] = bd["store"].codes(En)
    fgnn_ix = GF.FastGNN(gnn, node_store=gstore)
    leans = {k: T2.Lean(fc, None, None, bd["codes"], bd["store"]) for k, bd in bases.items()}
    tlab, tlab_stats, label_info = None, None, None
    if any(sv.aw_set for sv in served):
        t = time.time()
        labels = L4.Labels(name, Path(op.pkg[2]), op.pkg[1])
        if not labels.ok:
            raise SystemExit(f"{name}: no anchor table; this look times AW on a labelled graph")
        label_info = {"structural_edges": int(labels.skey.size), "load_s": labels.seconds,
                      "phrase_sha256": labels.phrase_sha256, "anchors": str(L4.ANCHORS[name])}
        tlab, tlab_stats = F9.build_tlab(fc._s["structural"], labels)
        log(f"tlab: {tlab_stats}")
        if tlab_stats["fw_found"] != tlab_stats["fw"] or tlab_stats["bw_found"] != tlab_stats["bw"]:
            raise SystemExit(f"{name}: typed entries without their stored edge: {tlab_stats}")
        for tag, (_path, blob) in blobs.items():
            if "aw_cfg" not in blob:
                continue
            Z = L4.codes_of(labels, blob["code_basis"])
            t_h = time.perf_counter()
            for sv in served:
                if sv.tag == tag and sv.aw_set:
                    sv.aws = F9.AWServe(sv.model.awn, Z)
            label_info[f"{tag}_label_tables_s"] = time.perf_counter() - t_h
        refs = {}
        for k, bd in bases.items():
            if bd["aw_tag"] is not None:
                cb = blobs[bd["aw_tag"]][1]["code_basis"]
                refs[k] = L4.Carve4(name, "select", bd["store"], context.nodes, limit=nq_all, labels=labels, basis=cb)
                label_info["look_checks"] = refs[k].label_checks
                label_info["look_aw_stats"] = refs[k].aw_stats
            else:
                refs[k] = L3.Carve3(name, "select", bd["store"], context.nodes, limit=nq_all)
        del labels
        gc.collect()
        label_info["seconds"] = time.time() - t
    else:
        refs = {k: L3.Carve3(name, "select", bd["store"], context.nodes, limit=nq_all) for k, bd in bases.items()}
    for k, r_ in refs.items():
        if r_.rows != nq_all:
            raise SystemExit(f"the select look holds {r_.rows} of the {nq_all} rows")
    for sv in served:
        bd = bases[sv.key]
        if sv.aw_set:
            sv.fz = F9.FusedLean9(fc, bd["codes"], bd["store"], sv, sv.aws, tlab)
            sv.f9chk = None
        else:
            sv.fz = F8.FusedLean8(fc, bd["codes"], bd["store"], sv)
            sv.f9chk = F9.FusedLean9(fc, bd["codes"], bd["store"], sv)
    look0 = next(iter(refs.values()))
    paths = list(GNN_FAMILY) + [sv.fz.name for sv in served]
    of = {sv.fz.name: sv for sv in served}
    twins = {}
    for sv in served:
        if sv.aw_set:
            for tw in served:
                if tw.tag == sv.tag and not tw.aw_set and set(tw.active) == set(sv.active) - {"AW"}:
                    twins[sv.fz.name] = tw.fz.name
    sweep = np.ones(int(a.sweep_mb * 2 ** 20) // 8)
    log(f"index built over {nodes.size} nodes ({time.time() - t_start:.0f}s); paths {paths}; AW twins {twins}")

    # ── per query: cold, resident, warm ──
    checks = {"pool": 0, "gnn_score_max_abs": 0.0, "gnn_score_off": 0, "ix_max_abs": 0.0, "ix_top5_same": 0,
              "lean_score_max_abs": {sv.name: 0.0 for sv in served}, "lean_score_off": {sv.name: 0 for sv in served},
              "fused_equal": {sv.name: {"XZ": 0, "base": 0, "scores": 0} for sv in served},
              "f9_equals_f8": {sv.name: {"XZ": 0, "base": 0} for sv in served if not sv.aw_set},
              "aw": {sv.name: {"rel_max": 0.0, "abs_max": 0.0, "off": 0, "score_max_abs": 0.0, "top5_same": 0, "top5_same_model": 0}
                     for sv in served if sv.aw_set},
              "runs_equal": {p: 0 for p in paths}, "queries": 0}
    recs = []
    gc.collect()
    gc.disable()
    kern.fast(True)
    try:
        for i in range(nq_all):
            d_ids, s_ids = np.asarray(prep.dense_ids[i]), np.asarray(prep.splade_ids[i])
            pool, seeds = builder(d_ids, s_ids)
            if not (np.array_equal(pool, prep.pools[i]) and np.array_equal(seeds, np.asarray(prep.seeds[i], dtype=np.int64))):
                checks["pool"] += 1
            pool = np.asarray(pool, dtype=np.int64)
            seeds = np.asarray(seeds, dtype=np.int64)
            pool_t = torch.from_numpy(np.searchsorted(nodes, pool))
            if not np.array_equal(nodes[pool_t.numpy()], pool):
                raise SystemExit(f"row {i}: a pool node is missing from the index-time store")
            inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
            top = {int(d_ids[0]), int(s_ids[0])}
            buckets = np.asarray([0 if int(x) in top else 1 for x in seeds], dtype=np.int64)
            qemb16 = np.asarray(prep.qemb[i]).astype(np.float16)
            kk = min(5, int(pool.size))

            def run(p):
                c0 = clock()
                if p in GNN_FAMILY:
                    if own_gather:
                        comp = fc.compile(inp, pool, seeds, timings={}, v2=True)
                        E = fc._E[:pool.size]
                    else:
                        E = context.nodes.read(pool)
                        comp = fc.compile(inp, pool, seeds, embeddings=E, timings={}, v2=True)
                    c1 = clock()
                    b = LT.fast_pack(comp, E, prep.qemb[i], columns, True)
                    c2 = clock()
                    sc = fgnn(b) if p == "gnn:fast" else fgnn_ix(b, pool_t)
                    torch.topk(sc, kk)
                    c3 = clock()
                    return sc, {"compile": c1 - c0, "pack": c2 - c1, "features": c2 - c0, "forward": c3 - c2, "total": c3 - c0}, None
                sv = of[p]
                T = {}
                XZ, base = sv.fz(inp, pool, seeds, buckets, qemb16, T)
                c2 = clock()
                sc = T6.fold_forward(sv.folded, XZ, base)
                torch.topk(sc, kk)
                c3 = clock()
                return sc, {"st": dict(T), "features": c2 - c0, "forward": c3 - c2, "total": c3 - c0}, (XZ, base)

            k = i % len(paths)
            order = paths[k:] + paths[:k]
            rec = {"i": i, "n": int(pool.size), "first": order[0]}
            out = {}
            rec["cold"] = {}
            for p in order:
                sc, r, xb = run(p)
                rec["cold"][p] = r
                out[p] = [sc, xb]
            rec["resident"] = {}
            for p in order:
                sweep += 1.0
                sc, r, _xb = run(p)
                rec["resident"][p] = r
                out[p].append(sc)
            rec["warm"] = {}
            for p in order:
                sc1, _r, _xb = run(p)
                sc, r, _xb = run(p)
                rec["warm"][p] = r
                out[p] += [sc1, sc]
            # ── checks, untimed ──
            checks["queries"] += 1
            for p in paths:
                s0_ = out[p][0]
                checks["runs_equal"][p] += int(all(torch.equal(s0_, s) for s in out[p][2:]))
            a0, b0 = int(look0.off[i]), int(look0.off[i + 1])
            if b0 - a0 != pool.size:
                raise SystemExit(f"row {i}: the look's pool has {b0 - a0} nodes, not {pool.size}")
            dmax = float(np.abs(out[REF][0].numpy() - look0.score[a0:b0][:, 3]).max())
            checks["gnn_score_max_abs"] = max(checks["gnn_score_max_abs"], dmax)
            checks["gnn_score_off"] += int(dmax > 1e-3)
            checks["ix_max_abs"] = max(checks["ix_max_abs"], float((out[REF][0] - out["gnn:fast+ix"][0]).abs().max()))
            checks["ix_top5_same"] += int(torch.equal(torch.topk(out[REF][0], kk).indices, torch.topk(out["gnn:fast+ix"][0], kk).indices))
            qi = np.asarray([i])
            for sv in served:
                look = refs[sv.key]
                p = sv.fz.name
                XZf, bf = out[p][1]
                scf = out[p][0]
                feats, q, P, _T, _ms, _ma = leans[sv.key](inp, pool, seeds, buckets, sv)
                c = checks["fused_equal"][sv.name]
                if sv.aw_set:
                    fl, nq_, bz, _g, _ix = L4.batch_of4(look, qi, sv.blocks)
                    with torch.no_grad():
                        ref_sc = sv.model(fl, sv.keep_t, nq_, 1, bz)
                        aw_tr = sv.model.awn(L4.aw_batch(look, qi)).numpy()
                    xa, xb = sv.fz.x_slot["AW"]
                    aw_f = XZf[:, xa:xb].copy()
                    XZt, bt = T6.fold_rows(sv.folded, {**feats, "AW": aw_tr}, q, P, qemb16, None, None)
                    sct = T6.fold_forward(sv.folded, XZt, bt)
                    dmax = float((sct - ref_sc).abs().max())
                    XZr, br = T6.fold_rows(sv.folded, {**feats, "AW": aw_f}, q, P, qemb16, None, None)
                    scr = T6.fold_forward(sv.folded, XZr, br)
                    ca = checks["aw"][sv.name]
                    diff = np.abs(aw_f - aw_tr)
                    rel = float((diff / np.maximum(1.0, np.abs(aw_tr))).max()) if diff.size else 0.0
                    ca["rel_max"] = max(ca["rel_max"], rel)
                    ca["abs_max"] = max(ca["abs_max"], float(diff.max()) if diff.size else 0.0)
                    ca["off"] += int(rel > AW_TOL)
                    ca["score_max_abs"] = max(ca["score_max_abs"], float((scf - sct).abs().max()))
                    ca["top5_same"] += int(torch.equal(torch.topk(scf, kk).indices, torch.topk(sct, kk).indices))
                    ca["top5_same_model"] += int(torch.equal(torch.topk(scf, kk).indices, torch.topk(ref_sc, kk).indices))
                else:
                    fl, nq_, bz, _g, _ix = LM.batch_of(look, qi, sv.blocks)
                    with torch.no_grad():
                        ref_sc = sv.model(fl, sv.keep_t, nq_, 1, bz)
                    XZr, br = T6.fold_rows(sv.folded, feats, q, P, qemb16, None, None)
                    scr = T6.fold_forward(sv.folded, XZr, br)
                    dmax = float((scr - ref_sc).abs().max())
                    XZ9, b9 = sv.f9chk(inp, pool, seeds, buckets, qemb16)
                    c9 = checks["f9_equals_f8"][sv.name]
                    c9["XZ"] += int(bits_equal(XZ9, XZf))
                    c9["base"] += int(bits_equal(b9, bf))
                checks["lean_score_max_abs"][sv.name] = max(checks["lean_score_max_abs"][sv.name], dmax)
                checks["lean_score_off"][sv.name] += int(dmax > 1e-3)
                c["XZ"] += int(bits_equal(XZr, XZf))
                c["base"] += int(bits_equal(br, bf))
                c["scores"] += int(torch.equal(scr, scf))
            recs.append(rec)
            if i % 50 == 0:
                log(f"  q{i}: n {pool.size}; resident ms " + ", ".join(f"{p} {1e3 * rec['resident'][p]['total']:.2f}" for p in paths))
    finally:
        gc.enable()
        kern.fast(False)
    log(f"done ({time.time() - t_start:.0f}s); checks: {checks}")

    # ── summary (warm-up queries excluded) ──
    wq = recs[a.warm:]
    summary = {}
    for reg in REGIMES:
        summary[reg] = {}
        for p in paths:
            d = {k: {"p50": 1e3 * pct([r[reg][p][k] for r in wq], 50), "p95": 1e3 * pct([r[reg][p][k] for r in wq], 95),
                     "mean": 1e3 * float(np.mean([r[reg][p][k] for r in wq]))} for k in PARTS}
            d["ratio_total_p50"] = pct([r[reg][p]["total"] / r[reg][REF]["total"] for r in wq], 50)
            d["ratio_features_p50"] = pct([r[reg][p]["features"] / r[reg][REF]["features"] for r in wq], 50)
            if reg == "cold":
                fr = [r for r in wq if r["first"] == p]
                d["first"] = {"queries": len(fr), **{k: {"p50": 1e3 * pct([r["cold"][p][k] for r in fr], 50),
                                                          "mean": 1e3 * float(np.mean([r["cold"][p][k] for r in fr]))} for k in PARTS}} if fr else None
            summary[reg][p] = d
        summary[reg]["aw_over_twin"] = {p: {k: pct([r[reg][p][k] / r[reg][tw][k] for r in wq], 50) for k in PARTS}
                                        for p, tw in twins.items()}
    stages = {}
    for sv in served:
        p = sv.fz.name
        ks = sorted({k for r in wq for k in r["resident"][p]["st"]})
        stages[p] = {k: 1e3 * pct([r["resident"][p]["st"].get(k, 0.0) for r in wq], 50) for k in ks}
    log(f"{name}: {len(wq)} queries after {a.warm} warm-up; threads {THREADS}, own gather {own_gather}")
    for reg in REGIMES:
        log(f"  {reg}: ms p50 total / features / forward; paired ratio to gnn:fast, total | features")
        for p in paths:
            v = summary[reg][p]
            extra = ""
            if reg == "cold" and v.get("first"):
                extra = f"   first ({v['first']['queries']}) total {v['first']['total']['p50']:.2f} features {v['first']['features']['p50']:.2f}"
            log(f"    {p:16s} {v['total']['p50']:7.2f} / {v['features']['p50']:6.2f} / {v['forward']['p50']:6.3f}   "
                f"{v['ratio_total_p50']:.3f} | {v['ratio_features_p50']:.3f}{extra}")
        if twins:
            log("    AW set over its twin without AW, paired p50 (features, total): "
                + ", ".join(f"{p} {v['features']:.3f}, {v['total']:.3f}" for p, v in summary[reg]["aw_over_twin"].items()))
    log("  resident stage p50 (ms): " + "; ".join(f"{p}: " + ", ".join(f"{k} {v:.3f}" for k, v in st.items()) for p, st in stages.items()))
    res = {"look": "lean_time9", "dataset": name, "carve": "select", "queries": len(recs), "warm_excluded": a.warm, "threads": THREADS,
           "torch_threads": torch.get_num_threads(), "numba_threads": FF.numba.get_num_threads(), "own_gather": own_gather,
           "sweep_mb": a.sweep_mb, "summary": summary, "stages_resident_p50_ms": stages, "checks": checks, "aw_tol": AW_TOL,
           "tlab": tlab_stats, "labels": label_info, "twins": twins,
           "sets": {sv.name: {"blocks": sv.blocks, "active": sv.active, "store": sv.store, "basis": sv.key, "path": sv.fz.name,
                              "folded_in_w": sv.folded.fu.in_w, "fused_width": sv.fz.W} for sv in served},
           "models": {tag: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for tag, (path, _b) in blobs.items()},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "imports_sha256": {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                              for m in (F9, F8, F7, L4, T2, T3, T6, EF, LFU, FF, GF)},
           "rows": [{"i": r["i"], "n": r["n"], "first": r["first"],
                     **{reg: {p: {k: r[reg][p][k] for k in PARTS} for p in paths} for reg in REGIMES},
                     "st_resident": {sv.fz.name: r["resident"][sv.fz.name]["st"] for sv in served}} for r in recs],
           "seconds": time.time() - t_start}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(a.out).with_name(Path(a.out).name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, a.out)
        log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
