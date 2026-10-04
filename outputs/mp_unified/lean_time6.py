"""Design look (untracked; not a result and not filed): serving cost per query at batch heights B (--batches, default
1, 8, 32, 128), the lean sets beside the six GNN's exact fast serving form (gnn_fast.py), in one run on the same pools.

lean_time5 timed batch 1 alone. A server that batches B queries pays, per query,
    features(q) + block(B) / B,
features being each query's own work and block the work done once per block of B queries:
  - the GNN family: features = the fast compile and the pack; block = merging the B packs in pack_queries_v2's layout,
    the forward, and every query's top-5;
  - a lean set: features = its blocks (lean_time4's need-only path when it has compiled blocks, lean_time2's lean path
    when it has none) and the folded MLP's input rows [X, z(X)] and base, which are z-scored per query (T3.Folded's
    column part, unchanged); block = stacking the rows, the folded MLP (addmm, GELU, addmm, GELU, addmv) once, and
    every query's top-5.
The block's share falls as B grows, so at large B (or with the forward on an accelerator) the comparison is between
features: the GNN needs its whole compile, a lean set only what its blocks read.

Phase 1, batch 1 (the path order rotating query by query): each path's features and forward, timed apart. Each
query's pack (copied: the compile reuses its buffers), its lean rows and its batch-1 scores are kept, untimed.
Phase 2, for each B: one untimed check pass over the warm queries in blocks of B, then --rounds timed rounds, each a
fresh random order of the warm queries cut into blocks of B; every block runs every path (the order rotating block by
block), its merge, forward and top-5 timed apart.

Paths:
  gnn:fast       fast compile -> pack -> FastGNN (the six GNN's forward bit for bit)
  gnn:fast+ix    the same, node_projection(e_v) read from an index-time store (64 floats per node)
  gnn:floor      fast compile -> pack -> the GNN's input block alone: what the GNN pays before its cell and evidence
                 flow run (a reference line, not a model; it scores nothing)
  twin           fast compile -> pack -> the twin's own forward (its training form). The twin is not edge-free: its
                 input block takes, per edge family, the mean of a learned projection of the neighbours' embeddings
  <tag>:<set>    a lean set (no message passing)

Checks:
  - every query: the rebuilt pool equals the prepared pool; gnn:fast's and the twin's scores against the look's stored
    scores; each lean set's scores against its model on lean_mlp.batch_of's inputs from the look; gnn:fast+ix's
    largest difference from gnn:fast and its top-5;
  - the first --check queries: gnn:fast equals the GNN's training-form forward (torch.equal); each lean set's split
    forward (rows, then the MLP) equals T3.Folded's (torch.equal);
  - each B's check pass: every block's scores, query by query, against the query's batch-1 scores (largest |diff| and
    top-5 identity); on its first block, FastGNN on the merged batch equals the training form on it (torch.equal).

Summary for each B (warm queries only):
  per_query_ms   mean over timed blocks of (the block's queries' features + the block's merge, forward, top-5) / B
  ratio          paired: median over timed blocks of that block cost, the path's over gnn:fast's (p25 and p75 too)
  features_only  the warm queries' summed features, the path's over gnn:fast's: the limit as the block's share vanishes

    python outputs/mp_unified/lean_time6.py --threads 1 --dataset 2wiki --queries 300 --batches 1,8,32,128 \\
        --models l3=outputs/mp_unified/lean/l3-2w_models.pt:pick,drop/pick,lean2s,lean \\
                 lp=outputs/mp_unified/lean/lp-2w_models.pt:p5,p6u,p7,p8,p16,pick --out scratch.json
    python outputs/mp_unified/lean_time6.py --selftest
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
os.environ["LEAN_TIME_THREADS"] = str(THREADS)        # lean_time, lean_time2 and lean_time4 read it at import
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
REF = "gnn:fast"
GNN_FAMILY = ("gnn:fast", "gnn:fast+ix", "gnn:floor", "twin")


# ── the batched serving pieces ───────────────────────────────────────────────


def own(b):
    """A packed batch whose tensors own their memory (the compile reuses its buffers for the next query)."""
    return type(b)(**{k: v.clone() for k, v in vars(b).items()})


def merge(bs):
    """Single-query packed batches as one, in pack_queries_v2's layout: rows stacked, seeds and edges offset."""
    sizes = [int(b.x.shape[0]) for b in bs]
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    ei, sn = [], []
    for b, o in zip(bs, offs.tolist()):
        ei.append(b.edge_index + o)
        s = b.seed_nodes.clone()
        s[s >= 0] += o
        sn.append(s)
    return type(bs[0])(x=torch.cat([b.x for b in bs]), qptr=torch.from_numpy(offs),
                       node_query=torch.repeat_interleave(torch.arange(len(bs)), torch.tensor(sizes)),
                       emb=torch.cat([b.emb for b in bs]), qemb=torch.cat([b.qemb for b in bs]),
                       seedw=torch.cat([b.seedw for b in bs]), seed_nodes=torch.cat(sn), edge_index=torch.cat(ei, 1),
                       edge_attr=torch.cat([b.edge_attr for b in bs]), gold=torch.cat([b.gold for b in bs]))


def top5_rows(sc, sizes):
    """Every query's top-5 node indices from a block's stacked scores: one torch.topk over rows padded with -inf."""
    if len(sizes) == 1:
        return torch.topk(sc, min(5, int(sc.shape[0]))).indices[None, :]
    sz = torch.tensor(sizes)
    q = torch.repeat_interleave(torch.arange(len(sizes)), sz)
    start = torch.cumsum(sz, 0) - sz
    loc = torch.arange(int(sc.shape[0])) - start.index_select(0, q)
    pad = torch.full((len(sizes), max(sizes)), -float("inf"), dtype=sc.dtype)
    pad[q, loc] = sc
    return torch.topk(pad, min(5, max(sizes)), dim=1).indices


def fold_rows(fd, feats, q, P, qemb16, extra_x, blocks_idx):
    """T3.Folded.__call__'s per-query part, unchanged: the active blocks' columns, [X, z(X)] and the base."""
    fu = fd.fu
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
    return XZ, base


def fold_forward(fd, XZ, base):
    """T3.Folded.__call__'s MLP, on one query's rows or a block's stacked rows."""
    with torch.no_grad():
        H = TF.gelu(torch.addmm(fd.c1, torch.from_numpy(XZ), fd.A))
        H = TF.gelu(torch.addmm(fd.b2, H, fd.W2))
        return torch.addmv(torch.from_numpy(base), H, fd.wo)


def peak_rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9     # KiB on Linux
    except Exception:
        return None


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1, help="BLAS, numba and torch threads (read at import, before numpy loads)")
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--models", nargs="+", required=True, help="tag=FILE.pt:setA,setB (lean_mlp, lean_mlp2 or lean_mlp3 saves)")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--warm", type=int, default=10, help="the first queries: excluded from every summary and from phase 2")
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--check", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.threads != THREADS:
        raise SystemExit(f"--threads {a.threads}, but the thread variables were set to {THREADS} at import")
    torch.set_num_threads(THREADS)
    batches = [int(x) for x in a.batches.split(",")]
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
    if nq_all - a.warm < max(batches):
        raise SystemExit(f"{nq_all - a.warm} warm queries cannot fill a block of {max(batches)}")
    log(f"{name}: {nq_all} select queries prepared ({time.time() - t_start:.0f}s); pools mean {np.mean([p.size for p in prep.pools]):.0f}; "
        f"threads {THREADS}; batches {batches}")
    fc = FF.compiler_for(context.stores, context.nodes, context.rel_table)
    if fc is None:
        raise SystemExit("no fast compiler for this context")
    kern = T3.Kernels(fc)
    own_gather = fc._h16 is not None or fc._shards is not None

    # ── the lean models; one int8 store per PCA basis ──
    served, blobs, bases, tag_key = [], {}, {}, {}
    for tag, path, sets in T2.parse_models(a.models):
        blob = torch.load(path, weights_only=False)
        blobs[tag] = (path, blob)
        kind = "lean_mlp3" if blob.get("store") else ("lean_mlp2" if any(b in L2.NEW for b in blob.get("present", [])) else "lean_mlp")
        key = None
        if blob.get("store") == "pca256":
            bb = blob["basis"]
            key = hashlib.sha256(b"".join(np.ascontiguousarray(bb[k], dtype=np.float32).tobytes() for k in ("m", "V", "w"))).hexdigest()[:12]
            bases.setdefault(key, {"basis": bb, "tags": []})["tags"].append(tag)
        if key is not None:
            ref_kind = "pca"
        elif any(b in L2.NEW for b in blob.get("present", [])):
            ref_kind = "lean_mlp2"
        else:
            ref_kind = "lean_mlp"
        tag_key[tag] = (ref_kind, key)
        for s in sets:
            sv = T2.Served(tag, path, s, blob, kind)
            sv.folded = T3.Folded(sv)
            sv.plan = NC.Plan(sv.extra)
            sv.key = key
            if sv.plan.unsupported:
                raise SystemExit(f"{sv.name}: compiled blocks {sv.plan.unsupported} are not mirrored")
            served.append(sv)
            log(f"{sv.name} ({kind}, basis {key}): {sv.active}; {sv.plan}")
    if len(bases) > 1:
        log(f"{len(bases)} PCA bases: " + "; ".join(f"{k} <- {v['tags']}" for k, v in bases.items()))

    # ── index time: the stores, over the nodes the pools touch ──
    R = LM.projection()
    nodes = np.unique(np.concatenate(prep.pools))
    store16 = np.zeros((fc.n_nodes, LM.PROJ_DIM), dtype=np.float16)
    for bd in bases.values():
        bd["store"] = L3.Store(bd["basis"])
        bd["codes"] = np.zeros((fc.n_nodes, L3.STORE_K), dtype=np.int8)
    # the GNN's index-time store holds the pool nodes' rows only (a torch.zeros over every node commits 64 floats a node
    # of RAM; the lean stores are numpy zeros, committed only where written); gnn:fast+ix gathers by position in it
    gstore = torch.zeros(nodes.size, gnn.input.semantic.node_projection.out_features, dtype=torch.float32)
    ix_s = {"read": 0.0, "gnn_ix": 0.0, "rand128": 0.0, "pca256": 0.0}
    clock = time.perf_counter
    for s0 in range(0, nodes.size, 4096):
        rr = nodes[s0:s0 + 4096]
        c0 = clock()
        E_rows = np.asarray(context.nodes.read(rr))
        c1 = clock()
        with torch.inference_mode():
            gstore[s0:s0 + rr.size] = gnn.input.semantic.node_projection(torch.from_numpy(E_rows).to(torch.float32))
        c2 = clock()
        En = np.asarray(E_rows, dtype=np.float32)
        En = En / np.maximum(np.linalg.norm(En, axis=1, keepdims=True), 1e-12)
        store16[rr] = (En @ R).astype(np.float16)
        c3 = clock()
        for bd in bases.values():
            bd["codes"][rr] = bd["store"].codes(En)
        c4 = clock()
        ix_s["read"] += c1 - c0
        ix_s["gnn_ix"] += c2 - c1
        ix_s["rand128"] += c3 - c2
        ix_s["pca256"] += (c4 - c3) / max(len(bases), 1)
    index = {"nodes_built": int(nodes.size), "graph_nodes": int(fc.n_nodes),
             "ms_per_1k_nodes": {k: 1e3 * v / nodes.size * 1e3 for k, v in ix_s.items()},
             "bytes_per_node": {"embedding_fp16": 2 * int(context.nodes.read(nodes[:1]).shape[1]), "gnn_ix": 4 * int(gstore.shape[1]),
                                "rand128": 2 * LM.PROJ_DIM, "pca256": L3.STORE_K}}
    fgnn_ix = GF.FastGNN(gnn, node_store=gstore)
    leans = {None: (T2.Lean(fc, store16, R, None, None), T4.Lean4(fc, store16, R, None, None))}
    for k, bd in bases.items():
        leans[k] = (T2.Lean(fc, store16, R, bd["codes"], bd["store"]), T4.Lean4(fc, store16, R, bd["codes"], bd["store"]))
    refs, ref_of = {}, {}
    for tag in blobs:
        rk = tag_key[tag]
        if rk not in refs:
            if rk[0] == "pca":
                refs[rk] = L3.Carve3(name, "select", bases[rk[1]]["store"], context.nodes, limit=nq_all)
            elif rk[0] == "lean_mlp2":
                refs[rk] = L2.Carve2(name, "select", limit=nq_all)
            else:
                refs[rk] = LM.Carve(name, "select", limit=nq_all)
            if refs[rk].rows != nq_all:
                raise SystemExit(f"the select look holds {refs[rk].rows} of the {nq_all} rows")
        ref_of[tag] = refs[rk]
    look0 = next(iter(refs.values()))
    by_name = {sv.name: sv for sv in served}
    paths = list(GNN_FAMILY) + [sv.name for sv in served]
    scored = [p for p in paths if p != "gnn:floor"]

    # ── phase 1: batch 1 ──
    checks = {"pool": 0, "score_max_abs": {"twin": 0.0, REF: 0.0}, "score_off": {"twin": 0, REF: 0},
              "gnn_fast_equal_training_form": [0, 0], "ix_max_abs": 0.0, "ix_top5_same": 0,
              "lean_score_max_abs": {sv.name: 0.0 for sv in served}, "lean_score_off": {sv.name: 0 for sv in served},
              "lean_split_equal_folded": {sv.name: [0, 0] for sv in served}, "batched": {}}
    packs, pools_t, sizes, recs = [], [], [], []
    rows_of = {sv.name: [] for sv in served}
    s1 = {p: [] for p in scored}
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
            pool_t = torch.from_numpy(np.searchsorted(nodes, pool))         # the rows of gstore, in pool order
            if not np.array_equal(nodes[pool_t.numpy()], pool):
                raise SystemExit(f"row {i}: a pool node is missing from the index-time store")
            inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
            top = {int(d_ids[0]), int(s_ids[0])}
            buckets = np.asarray([0 if int(x) in top else 1 for x in seeds], dtype=np.int64)
            qemb16 = np.asarray(prep.qemb[i]).astype(np.float16)
            rec = {"i": i, "n": int(pool.size), "pool_ms": 1e3 * t_pool}
            kept, out = {}, {}

            def compile_(laps):
                if own_gather:
                    return fc.compile(inp, pool, seeds, timings=laps, v2=True), None
                E = context.nodes.read(pool)
                return fc.compile(inp, pool, seeds, embeddings=E, timings=laps, v2=True), E

            def gnn_path(p):
                c0 = clock()
                comp, E = compile_({})
                if E is None:
                    E = fc._E[:pool.size]
                c1 = clock()
                b = LT.fast_pack(comp, E, prep.qemb[i], columns, True)
                c2 = clock()
                if p == "gnn:fast":
                    sc = fgnn(b)
                elif p == "gnn:fast+ix":
                    sc = fgnn_ix(b, pool_t)
                elif p == "gnn:floor":
                    with torch.inference_mode():
                        fgnn._input(b)
                    sc = None
                else:
                    with torch.no_grad():
                        sc = twin(V2.arm_view(twin, b, inputs))
                if sc is not None:
                    torch.topk(sc, min(5, int(sc.shape[0])))
                c3 = clock()
                rec[p] = {"compile": c1 - c0, "pack": c2 - c1, "features": c2 - c0, "forward": c3 - c2, "total": c3 - c0}
                if "pack" not in kept:
                    kept["pack"] = own(b)
                return sc

            def lean_path(sv):
                lean2, lean4 = leans[sv.key]
                c0 = clock()
                if sv.plan.any:
                    emb = None if own_gather else context.nodes.read(pool)
                    feats, q, P, T, _ms, _ma, xs = lean4(inp, pool, seeds, buckets, sv, sv.plan, columns, emb)
                else:
                    feats, q, P, T, _ms, _ma = lean2(inp, pool, seeds, buckets, sv)
                    xs = None
                c1 = clock()
                XZ, base = fold_rows(sv.folded, feats, q, P, qemb16, xs, ref_of[sv.tag].blocks_idx)
                c2 = clock()
                sc = fold_forward(sv.folded, XZ, base)
                torch.topk(sc, min(5, int(sc.shape[0])))
                c3 = clock()
                rec[sv.name] = {**{f"st_{k}": v for k, v in T.items()}, "blocks": c1 - c0, "rows": c2 - c1, "features": c2 - c0,
                                "forward": c3 - c2, "total": c3 - c0}
                kept[sv.name] = (feats, q, P, xs, XZ, base)
                return sc

            k = i % len(paths)
            for p in paths[k:] + paths[:k]:
                out[p] = gnn_path(p) if p in GNN_FAMILY else lean_path(by_name[p])
            # ── checks, untimed ──
            a0, b0 = int(look0.off[i]), int(look0.off[i + 1])
            if b0 - a0 != pool.size:
                raise SystemExit(f"row {i}: the look's pool has {b0 - a0} nodes, not {pool.size}")
            st_sc = look0.score[a0:b0]
            for p, col in (("twin", 0), (REF, 3)):
                dmax = float(np.abs(out[p].numpy() - st_sc[:, col]).max())
                checks["score_max_abs"][p] = max(checks["score_max_abs"][p], dmax)
                checks["score_off"][p] += int(dmax > 1e-3)
            kk = min(5, int(pool.size))
            checks["ix_max_abs"] = max(checks["ix_max_abs"], float((out[REF] - out["gnn:fast+ix"]).abs().max()))
            checks["ix_top5_same"] += int(torch.equal(torch.topk(out[REF], kk).indices, torch.topk(out["gnn:fast+ix"], kk).indices))
            if i < a.check:
                with torch.no_grad():
                    ref_tf = gnn(V2.arm_view(gnn, kept["pack"], inputs))
                checks["gnn_fast_equal_training_form"][0] += 1
                checks["gnn_fast_equal_training_form"][1] += int(torch.equal(ref_tf, out[REF]))
            for sv in served:
                feats, q, P, xs, XZ, base = kept[sv.name]
                look = ref_of[sv.tag]
                fl, nq_, bz, _g, _ix = LM.batch_of(look, np.asarray([i]), sv.blocks)
                with torch.no_grad():
                    ref_sc = sv.model(fl, sv.keep_t, nq_, 1, bz)
                dmax = float((out[sv.name] - ref_sc).abs().max())
                checks["lean_score_max_abs"][sv.name] = max(checks["lean_score_max_abs"][sv.name], dmax)
                checks["lean_score_off"][sv.name] += int(dmax > 1e-3)
                if i < a.check:
                    c = checks["lean_split_equal_folded"][sv.name]
                    c[0] += 1
                    c[1] += int(torch.equal(out[sv.name], sv.folded(feats, q, P, qemb16, xs, look.blocks_idx)))
                rows_of[sv.name].append((XZ, base))
            packs.append(kept["pack"])
            pools_t.append(pool_t)
            sizes.append(int(pool.size))
            for p in scored:
                s1[p].append(out[p])
            recs.append(rec)
            if i % 50 == 0:
                log(f"  q{i}: n {pool.size}; ms " + ", ".join(f"{p} {1e3 * rec[p]['total']:.2f}" for p in paths))
    finally:
        gc.enable()
        kern.fast(False)
    log(f"phase 1 done ({time.time() - t_start:.0f}s); checks so far: {checks}")

    # ── phase 2: blocks of B ──
    def run_block(p, blk):
        sz = [sizes[i] for i in blk]
        mb = None
        if p in GNN_FAMILY:
            c0 = clock()
            mb = packs[blk[0]] if len(blk) == 1 else merge([packs[i] for i in blk])
            pt = None
            if p == "gnn:fast+ix":
                pt = pools_t[blk[0]] if len(blk) == 1 else torch.cat([pools_t[i] for i in blk])
            c1 = clock()
            if p == "gnn:fast":
                sc = fgnn(mb)
            elif p == "gnn:fast+ix":
                sc = fgnn_ix(mb, pt)
            elif p == "gnn:floor":
                with torch.inference_mode():
                    fgnn._input(mb)
                sc = None
            else:
                with torch.no_grad():
                    sc = twin(V2.arm_view(twin, mb, inputs))
            c2 = clock()
        else:
            rr = rows_of[p]
            c0 = clock()
            if len(blk) == 1:
                XZ, base = rr[blk[0]]
            else:
                XZ = np.concatenate([rr[i][0] for i in blk])
                base = np.concatenate([rr[i][1] for i in blk])
            c1 = clock()
            sc = fold_forward(by_name[p].folded, XZ, base)
            c2 = clock()
        if sc is not None:
            top5_rows(sc, sz)
        c3 = clock()
        return {"merge": c1 - c0, "forward": c2 - c1, "top5": c3 - c2, "total": c3 - c0}, sc, mb

    warm_ids = np.arange(a.warm, nq_all)
    rng = np.random.default_rng(a.seed)
    p2 = {}
    gc.collect()
    gc.disable()
    try:
        for B in batches:
            nb = warm_ids.size // B
            chk = {p: {"max_abs": 0.0, "top5_same": 0, "queries": 0} for p in scored}
            eq_tf = None
            perm = rng.permutation(warm_ids)
            for bi in range(nb):                     # the check pass, untimed (it also warms each block height)
                blk = perm[bi * B:(bi + 1) * B].tolist()
                for p in paths:
                    _t, sc, mb = run_block(p, blk)
                    if sc is None:
                        continue
                    off = 0
                    for i in blk:
                        n = sizes[i]
                        seg, ref = sc[off:off + n], s1[p][i]
                        chk[p]["max_abs"] = max(chk[p]["max_abs"], float((seg - ref).abs().max()))
                        kk = min(5, n)
                        chk[p]["top5_same"] += int(torch.equal(torch.topk(seg, kk).indices, torch.topk(ref, kk).indices))
                        chk[p]["queries"] += 1
                        off += n
                    if bi == 0 and p == REF:
                        with torch.no_grad():
                            eq_tf = bool(torch.equal(gnn(V2.arm_view(gnn, mb, inputs)), sc))
            checks["batched"][str(B)] = {"vs_batch1": chk, "fast_equals_training_form_on_merged": eq_tf}
            brecs = []
            for r in range(a.rounds):
                perm = rng.permutation(warm_ids)
                for bi in range(nb):
                    blk = perm[bi * B:(bi + 1) * B].tolist()
                    k = (bi + r) % len(paths)
                    brec = {"round": r, "q": blk}
                    for p in paths[k:] + paths[:k]:
                        brec[p] = run_block(p, blk)[0]
                    brecs.append(brec)
            p2[B] = brecs
            log(f"  B={B}: {nb} blocks a round x {a.rounds} rounds ({time.time() - t_start:.0f}s); "
                + ", ".join(f"{p} {1e3 * np.mean([b[p]['total'] for b in brecs]) / B:.3f}" for p in paths) + " ms a query (block only)")
    finally:
        gc.enable()

    # ── summary ──
    warm = recs[a.warm:]
    feat = {p: np.asarray([r[p]["features"] for r in recs]) for p in paths}
    batch1 = {}
    for p in paths:
        stages = sorted({k for r in warm for k in r[p]})
        batch1[p] = {k: {"p50": 1e3 * pct([r[p].get(k, 0.0) for r in warm], 50), "p95": 1e3 * pct([r[p].get(k, 0.0) for r in warm], 95),
                         "mean": 1e3 * float(np.mean([r[p].get(k, 0.0) for r in warm]))} for k in stages}
        batch1[p]["ratio_total_p50"] = pct([r[p]["total"] / r[REF]["total"] for r in warm], 50)
    feats_only = {p: float(feat[p][a.warm:].sum() / feat[REF][a.warm:].sum()) for p in paths}
    feats_only_p50 = {p: pct(feat[p][a.warm:] / feat[REF][a.warm:], 50) for p in paths}
    batched = {}
    for B, brecs in p2.items():
        cost = {p: np.asarray([feat[p][b["q"]].sum() + b[p]["total"] for b in brecs]) for p in paths}
        batched[str(B)] = {p: {"per_query_ms": 1e3 * float(cost[p].mean()) / B, "per_query_ms_p50": 1e3 * pct(cost[p], 50) / B,
                               "features_ms": 1e3 * float(np.mean([feat[p][b["q"]].mean() for b in brecs])),
                               "block_per_query_ms": 1e3 * float(np.mean([b[p]["total"] for b in brecs])) / B,
                               "block_ms": {k: 1e3 * float(np.mean([b[p][k] for b in brecs])) for k in ("merge", "forward", "top5", "total")},
                               "block_ms_p50": 1e3 * pct([b[p]["total"] for b in brecs], 50),
                               "ratio": {"p50": pct(cost[p] / cost[REF], 50), "p25": pct(cost[p] / cost[REF], 25),
                                         "p75": pct(cost[p] / cost[REF], 75)},
                               "blocks": len(brecs)} for p in paths}
    log(f"{name}: {len(warm)} warm queries; threads {THREADS}, own gather {own_gather}")
    log("  batch 1 (phase 1), ms p50 total / features / forward; paired ratio to gnn:fast:")
    for p in paths:
        v = batch1[p]
        log(f"    {p:16s} {v['total']['p50']:7.2f} / {v['features']['p50']:6.2f} / {v['forward']['p50']:6.3f}   {v['ratio_total_p50']:.3f}")
    for B in batched:
        log(f"  B={B}: per-query ms, mean (features + block / B) and p50 over blocks; paired block ratio p50 [p25, p75] to gnn:fast:")
        for p in paths:
            v = batched[B][p]
            log(f"    {p:16s} {v['per_query_ms']:7.3f} = {v['features_ms']:6.3f} + {v['block_per_query_ms']:6.3f}  p50 {v['per_query_ms_p50']:7.3f}   "
                f"{v['ratio']['p50']:.3f} [{v['ratio']['p25']:.3f}, {v['ratio']['p75']:.3f}]")
    log("  features only (the limit as B grows) to gnn:fast, summed | per-query p50: "
        + ", ".join(f"{p} {feats_only[p]:.3f} | {feats_only_p50[p]:.3f}" for p in paths))
    log(f"  checks: {checks}")
    res = {"look": "lean_time6", "dataset": name, "carve": "select", "queries": len(recs), "warm_excluded": a.warm, "threads": THREADS,
           "torch_threads": torch.get_num_threads(), "numba_threads": FF.numba.get_num_threads(), "own_gather": own_gather,
           "batches": batches, "rounds": a.rounds, "batch1": batch1, "batched": batched, "features_only_ratio": feats_only,
           "features_only_ratio_p50": feats_only_p50,
           "index": index, "params": {"gnn0": n_params(gnn), "twin0": n_params(twin),
                                      **{sv.name: n_params(sv.model) for sv in served}},
           "folded_params": {sv.name: int(sum(np.asarray(getattr(sv.folded.fu, k)).size for k in ("A", "c1", "W2", "b2", "wo"))
                                          + sum(np.asarray(getattr(sv.folded.fu, k)).size for k in ("U", "V") if "SEMB" in sv.folded.fu.active))
                             for sv in served},
           "sets": {sv.name: {"blocks": sv.blocks, "active": sv.active, "compiled_extra": sv.extra, "plan": repr(sv.plan), "store": sv.store,
                              "basis": sv.key, "kind": sv.kind, "folded_in_w": sv.folded.fu.in_w} for sv in served},
           "checks": checks, "peak_rss_gb": peak_rss_gb(),
           "models": {tag: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for tag, (path, _b) in blobs.items()},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "imports_sha256": {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (T2, T3, T4, EF, LFU, NC, FF, GF)},
           "rows": [{"i": r["i"], "n": r["n"], "pool_ms": r["pool_ms"],
                     **{p: {k: r[p][k] for k in ("features", "forward", "total")} for p in paths}} for r in recs],
           "blocks": {str(B): [{"round": b["round"], "q": b["q"], **{p: b[p]["total"] for p in paths}} for b in brecs] for B, brecs in p2.items()},
           "seconds": time.time() - t_start}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(a.out).with_name(Path(a.out).name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, a.out)
        log(f"wrote {a.out}")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def _split(b):
    """A multi-query packed batch as single-query batches (the inverse of merge, for the self-test)."""
    out = []
    qp = b.qptr.tolist()
    for j in range(len(qp) - 1):
        lo, hi = qp[j], qp[j + 1]
        sel = (b.edge_index[1] >= lo) & (b.edge_index[1] < hi)
        s = b.seed_nodes[j:j + 1].clone()
        s[s >= 0] -= lo
        out.append(type(b)(x=b.x[lo:hi], qptr=torch.tensor([0, hi - lo]), node_query=torch.zeros(hi - lo, dtype=torch.long),
                           emb=b.emb[lo:hi], qemb=b.qemb[j:j + 1], seedw=b.seedw[lo:hi], seed_nodes=s,
                           edge_index=b.edge_index[:, sel] - lo, edge_attr=b.edge_attr[sel], gold=b.gold[lo:hi]))
    return out


def selftest():
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261003)
    n = 0
    for sizes_ in ((40, 1, 75), (12, 30), (5, 9, 2, 60)):
        for typed in (False, True):
            b = GF._batch(rng, sizes_, typed, typed, 50)
            m = merge(_split(b))
            for k, v in vars(b).items():
                w = getattr(m, k)
                assert v.shape == w.shape and torch.equal(v.to(w.dtype), w), k
            assert m.qptr.dtype == m.node_query.dtype == torch.int64
            mdl = GF._model(n, 50)
            assert torch.equal(GF.FastGNN(mdl)(m), GF.FastGNN(mdl)(b))
            n += 1
    for sizes_ in ((3, 7, 1, 12), (9,), (6, 6)):
        sc = torch.from_numpy(rng.standard_normal(sum(sizes_)).astype(np.float32))
        got = top5_rows(sc, list(sizes_))
        off = 0
        for j, s in enumerate(sizes_):
            kk = min(5, s)
            assert torch.equal(got[j, :kk], torch.topk(sc[off:off + s], kk).indices), (sizes_, j)
            off += s
    try:
        main_args = ["--threads", str(THREADS + 1)]
        sys.argv = [__file__] + main_args + ["--models", "x=y.pt:z"]
        main()
        raise AssertionError("a --threads other than the import-time one must be refused")
    except SystemExit as e:
        assert "thread variables" in str(e)
    print(f"selftest: merge inverts a split on {n} random batches (untyped and typed, 1 to 4 queries) field for field, "
          "and FastGNN scores the merged batch as the original; top5_rows equals per-query topk; a mismatched --threads "
          "is refused. all checks passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        raise SystemExit(main())
