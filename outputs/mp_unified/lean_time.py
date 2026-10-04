"""Design look (untracked; not a result and not filed): what the lean MLP (outputs/mp_unified/lean_mlp.py) costs to
serve against the six pair's twin and GNN, one query at a time (batch 1), laptop CPU, on the first --queries rows of
<ds>'s M3B select carve (train-derived). No metric is read here: the scores are checked against the look's stored
scores and lean_mlp's own inputs, so the quality lean_mlp reads on the look rows is the quality of the timed path.

Per query, each path timed with time.perf_counter, the path order rotated query by query:
  pool   the query's pool and seeds rebuilt from its cached first-stage lists under the frozen construction
         (deploy_ck_full_2wiki.PoolBuilder's steps: seeds_of, base_rows, expand_hops, build_pool), checked equal to
         m3b_compile.prepare's; it is the same for every path and reported once, apart
  twin   fast compile (FastCompiler.compile, every group; it gathers the 1536-wide rows itself) -> pack without edges
         (the twin reads none) -> forward
  gnn    fast compile -> pack with the compile's own pool edges (pool_edges' arrays) -> forward
  lean   rank lists (list_ranks, fill_retrieval with dense_cos left 0; no rank column reads it) -> structural pool
         edges (typed_edges, typed_attr; only when the set reads WALK or NBR) -> projected-store gather -> lean blocks
         (lean_mlp.lean_query, numpy) -> LeanMLP forward. A set with compiled blocks other than rank also runs the fast
         compile and is reported two ways: measured (lean + the full compile) and assembled (lean_mlp.cost_ms on this
         query's own compile laps: approximate, flagged)
Index time, once per graph and apart: the projected store, float16(normalize(E) @ R) per node (the look's proj), built
here for the nodes the timed pools touch and extrapolated to the graph (microseconds per node, MiB at 256 B a node).
Checks (all queries): rebuilt pool == prepared pool; twin and GNN scores of the timed paths match the look's stored
scores (max abs difference: the look scored in batches on the host); the lean rank block, edges-derived blocks and
projection equal the look's (counted, not required: BLAS order can flip a float16 rounding); the lean scores match
LeanMLP on lean_mlp.batch_of's inputs. First --check queries: the fast pack equals pack_queries_v2 field for field.

    python outputs/mp_unified/lean_time.py --dataset 2wiki --models outputs/mp_unified/lean/ln-2w_models.pt \
        --sets lean,pick --queries 300 --out outputs/mp_unified/lean/time-2w.json
"""
import os
import sys

THREADS = int(os.environ.get("LEAN_TIME_THREADS", "8"))   # deploy_ck_full_2wiki's laptop latency setting
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import lean_mlp as LM  # noqa: E402
import look_x_six as LX  # noqa: E402
from mp_retrieval import fast_features as FF  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402
from mp_retrieval.m3b_features import EDGE_ATTR, FAMILIES, MAX_SEEDS, QueryInputs  # noqa: E402
from mp_retrieval.m3b_models import PackedBatch  # noqa: E402
from mp_retrieval.universal_v2_models import K_REL, N_EDGE_FEATURES_V2, pack_queries_v2  # noqa: E402

S6, V2 = LX.S6, LX.V2
FWD, BWD = EDGE_ATTR.index("dir_fwd"), EDGE_ATTR.index("dir_bwd")
EMPTY_GOLD = np.zeros(0, dtype=np.int64)
RET_COLS = ("dense_cos", "dense_rr", "dense_in", "splade_rr", "splade_in", "splade_score_norm", "rrf", "agreement", "is_seed",
            "seed_rank")
LEAN_TIMED = ("SEM", "SEED", "WALK", "NBR")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


class PoolBuilder:
    """deploy_ck_full_2wiki.PoolBuilder's steps (that module sets its own thread environment at import, so it is not imported)."""

    def __init__(self, construction, cfg_h, stores, m3a, m3b_contract, m3b_compile):
        self.construction, self.m3a, self.m3b_contract, self.m3b_compile = construction, m3a, m3b_contract, m3b_compile
        self.constant = int(cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
        regime = construction["regime"]
        self.fams = [stores[f] for f in m3b_compile.regime_families(cfg_h, regime)] if regime != "RETRIEVAL" else None
        self.setting = construction.get("setting")

    def __call__(self, d_ids, s_ids):
        seeds = m3b_pools.seeds_of(d_ids, s_ids)
        base = self.m3b_compile.base_rows(self.construction, d_ids[None, :], s_ids[None, :], self.m3a, self.m3b_contract, self.constant)[0]
        expansion = m3b_pools.expand_hops(seeds, self.fams, self.setting) if self.fams is not None else None
        pool, _ = self.m3b_compile.build_pool(np.asarray(base, dtype=np.int64), seeds, expansion)
        return pool, seeds


# ── the pack, from the compile's own edges ───────────────────────────────────


def fast_pack(comp, E, qemb, columns, with_edges):
    """pack_queries_v2 for one query of an untyped graph, its edges taken from the compile (pool_edges' arrays, which the
    fast compile reproduces) instead of being built again; every relation slot -1."""
    n = int(comp.pool.size)
    row = np.full(MAX_SEEDS, -1, dtype=np.int64)
    sl = np.asarray(comp.seeds_local, dtype=np.int64)
    row[:sl.size] = sl[:MAX_SEEDS]
    eis, eas = [], []
    if with_edges:
        for f_i, fam in enumerate(FAMILIES):
            u, v, attr = comp.edges[fam]
            if u.size == 0:
                continue
            eis.append(np.stack((u.astype(np.int64), v.astype(np.int64))))
            onehot = np.zeros((u.size, len(FAMILIES)), dtype=np.float32)
            onehot[:, f_i] = 1.0
            eas.append(np.concatenate((onehot, attr, np.full((u.size, K_REL), -1.0, dtype=np.float32)), axis=1))
    ei = np.concatenate(eis, axis=1) if eis else np.empty((2, 0), dtype=np.int64)
    ea = np.concatenate(eas, axis=0) if eas else np.empty((0, N_EDGE_FEATURES_V2), dtype=np.float32)
    return PackedBatch(x=torch.from_numpy(np.ascontiguousarray(comp.scalars[:, columns])).to(torch.float32),
                       qptr=torch.tensor([0, n], dtype=torch.long), node_query=torch.zeros(n, dtype=torch.long),
                       emb=torch.from_numpy(E), qemb=torch.from_numpy(qemb[None, :]), seedw=torch.from_numpy(comp.seedw),
                       seed_nodes=torch.from_numpy(row[None, :]), edge_index=torch.from_numpy(ei), edge_attr=torch.from_numpy(ea),
                       gold=torch.zeros(n, dtype=torch.bool))


def batch_problems(a, b, with_edges):
    bad = []
    keys = ["x", "qptr", "node_query", "emb", "qemb", "seedw", "seed_nodes", "gold"] + (["edge_index", "edge_attr"] if with_edges else [])
    for k in keys:
        x, y = getattr(a, k), getattr(b, k)
        if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
            bad.append(f"{k}: {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}")
    return bad


# ── the lean path ────────────────────────────────────────────────────────────


class Lean:
    """The lean read path over a FastCompiler's kernels and arrays and an index-time projected store."""

    def __init__(self, fc, store, R):
        self.fc, self.store, self.R = fc, store, R
        self.s = fc._s["structural"]
        self.cols = np.asarray([V2.IDX[c] for c in RET_COLS], dtype=np.int64)
        self.rank_idx = np.asarray([V2.IDX[c] for c in LM.SPLIT["rank"]], dtype=np.int64)
        self.rrf_col = V2.IDX["rrf"]

    def __call__(self, inp, pool, seeds, buckets, need):
        """The set's lean inputs (float32 values rounded through float16, as the look stores them), the projected query,
        the normalised pool projection and the seconds per stage."""
        fc = self.fc
        clock = time.perf_counter
        T = {}
        t = clock()
        n = int(pool.size)
        K = fc._K_small if n < FF.SERIAL_BELOW else fc.K
        lookup = fc._lookup
        lookup[pool] = np.arange(n, dtype=np.int32)
        try:
            sl = lookup[seeds].astype(np.int64)
            X = np.zeros((n, V2.N_COLUMNS), dtype=np.float32)
            d_rank, d_score, d_in = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.bool_)
            s_rank, s_score, s_in = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.bool_)
            K.list_ranks(np.asarray(inp.dense_ids, dtype=np.int64), np.asarray(inp.dense_scores).astype(np.float32), lookup, fc.n_nodes,
                         d_rank, d_score, d_in)
            K.list_ranks(np.asarray(inp.splade_ids, dtype=np.int64), np.asarray(inp.splade_scores).astype(np.float32), lookup, fc.n_nodes,
                         s_rank, s_score, s_in)
            top = float(inp.splade_scores[0]) if inp.splade_scores.size else 0.0
            rrf = np.empty(n, np.float32)
            K.fill_retrieval(X, np.zeros(n, np.float32), d_in, d_rank, s_in, s_rank, s_score, top > 0, np.float32(max(top, 1e-12)), sl,
                             self.cols, rrf)
            out = {"rank": X[:, self.rank_idx].astype(np.float16).astype(np.float32),
                   "_rrf": X[:, self.rrf_col].astype(np.float16).astype(np.float32)}
            t1 = clock()
            T["rank"] = t1 - t
            if need & {"WALK", "NBR"}:
                s = self.s
                _ev, _eu, erel, edir, pair_id, pu, pv = K.typed_edges(s["tindptr"], s["tcol"], s["trel"], s["tdir"], pool, lookup, fc.cap)
                attr = K.typed_attr(pair_id, erel, edir, np.zeros(1, np.float32), False, int(pu.size))
                efwd, ebwd = attr[:, FWD] > 0.5, attr[:, BWD] > 0.5
            else:
                pu = pv = np.empty(0, np.int32)
                efwd = ebwd = np.empty(0, np.bool_)
            t2 = clock()
            T["edges"] = t2 - t1
        finally:
            lookup[pool] = -1
        P = self.store[pool]
        t3 = clock()
        T["proj_gather"] = t3 - t2
        timer = {b: 0.0 for b in LEAN_TIMED}
        L, q, Pn = LM.lean_query(n, P, inp.q, self.R, sl, buckets, pu, pv, np.zeros(pu.size, np.int8), efwd, ebwd, timer)
        for k in LM.STORED_LEAN:
            out[k] = L[k].astype(np.float16).astype(np.float32)
        Pn16 = Pn.astype(np.float16).astype(np.float32)
        T["lean_blocks"] = clock() - t3
        T.update({f"lean_{k}": v for k, v in timer.items()})
        return out, q, Pn16, T, int(pu.size)


def lean_inputs(blocks, feats, q, Pn, qemb16, extra_x=None, blocks_idx=None):
    """LeanMLP's inputs for one query, as lean_mlp.batch_of assembles them."""
    n = Pn.shape[0]
    f = {}
    for b in blocks:
        if b == "SEM":
            prod = Pn * q[None, :]
            f[b] = torch.from_numpy(np.concatenate([prod, prod.sum(1, keepdims=True)], axis=1))
        elif b == "SEMB":
            f[b] = (torch.from_numpy(qemb16.astype(np.float32)[None, :]), torch.from_numpy(Pn))
        elif b in feats:
            f[b] = LM.clean(feats[b])
        else:
            f[b] = LM.clean(extra_x[:, blocks_idx[b]])
    nq = torch.zeros(n, dtype=torch.long)
    base_z = LM.seg_zscore(torch.from_numpy(feats["_rrf"]).unsqueeze(1), nq, 1).squeeze(1)
    return f, nq, base_z


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--models", required=True, help="a lean_mlp --save-models file")
    ap.add_argument("--sets", default="lean,pick", help="the saved fixed-set models to time")
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
    builder = PoolBuilder(construction, cfg_h, context.stores, m3a, m3b_contract, m3b_compile)
    columns = inputs["column_indices"]
    nq_all = len(prep.pools)
    log(f"{name}: {nq_all} select queries prepared ({time.time() - t_start:.0f}s); pools mean {np.mean([p.size for p in prep.pools]):.0f}")
    fc = FF.compiler_for(context.stores, context.nodes, context.rel_table)
    if fc is None:
        raise SystemExit("no fast compiler for this context")
    own_gather = fc._h16 is not None or fc._shards is not None
    look = LM.Carve(name, "select", limit=nq_all)
    if look.rows != nq_all:
        raise SystemExit(f"the select look holds {look.rows} of the {nq_all} rows")
    R = LM.projection()
    nodes = np.unique(np.concatenate(prep.pools))
    t = time.time()
    store = np.zeros((fc.n_nodes, LM.PROJ_DIM), dtype=np.float16)
    for s0 in range(0, nodes.size, 4096):
        rr = nodes[s0:s0 + 4096]
        En = np.asarray(context.nodes.read(rr), dtype=np.float32)
        En = En / np.maximum(np.linalg.norm(En, axis=1, keepdims=True), 1e-12)
        store[rr] = (En @ R).astype(np.float16)
    index_s = time.time() - t
    index = {"projected_nodes": int(nodes.size), "graph_nodes": int(fc.n_nodes), "seconds": index_s,
             "us_per_node": 1e6 * index_s / max(1, nodes.size), "graph_seconds_extrapolated": index_s * fc.n_nodes / max(1, nodes.size),
             "store_mib_graph": fc.n_nodes * LM.PROJ_DIM * 2 / 2 ** 20, "embedding_mib_graph": fc.n_nodes * 1536 * 2 / 2 ** 20}
    log(f"projected store: {nodes.size} nodes in {index_s:.1f}s ({index['us_per_node']:.1f} us/node; the graph's {fc.n_nodes} in "
        f"~{index['graph_seconds_extrapolated']:.0f}s); {index['store_mib_graph']:.0f} MiB float16 against "
        f"{index['embedding_mib_graph']:.0f} MiB of 1536-wide float16 rows")
    lean = Lean(fc, store, R)
    blob = torch.load(a.models, weights_only=False)
    sets = {}
    for s in a.sets.split(","):
        d = blob["models"][s]
        m = LM.LeanMLP(d["blocks"], d["widths"], d["hidden"])
        m.load_state_dict(d["state"])
        m.eval()
        extra = [b for b in d["blocks"] if b not in LM.LEAN and b != "rank"]
        sets[s] = (m, list(d["blocks"]), extra)
        log(f"set {s}: {d['blocks']}" + (f"; compiled blocks {extra}: this set also runs the fast compile" if extra else ""))
    paths = ["twin", "gnn"] + [f"lean:{s}" for s in sets]
    clock = time.perf_counter
    checks = {"pool": 0, "pack": [], "twin_max_abs": 0.0, "gnn_max_abs": 0.0, "twin_off": 0, "gnn_off": 0,
              "lean_rank_off": 0, "lean_walk_off": 0, "lean_seed_off": 0, "lean_nbr_off": 0, "lean_proj_off": 0,
              "lean_proj_values_off": 0, "lean_proj_values": 0, "lean_score_max_abs": {s: 0.0 for s in sets},
              "lean_score_off": {s: 0 for s in sets}}
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

            def full_path(model, with_edges, tag):
                laps = {}
                c0 = clock()
                if own_gather:
                    comp = fc.compile(inp, pool, seeds, timings=laps, v2=True)
                    E = fc._E[:pool.size]
                else:
                    E = context.nodes.read(pool)
                    comp = fc.compile(inp, pool, seeds, embeddings=E, timings=laps, v2=True)
                c1 = clock()
                b = fast_pack(comp, E, prep.qemb[i], columns, with_edges)
                c2 = clock()
                with torch.no_grad():
                    sc = model(V2.arm_view(model, b, inputs))
                    torch.topk(sc, min(5, int(sc.shape[0])))
                c3 = clock()
                rec[tag] = {"compile": c1 - c0, "pack": c2 - c1, "forward": c3 - c2, "total": c3 - c0,
                            **{f"lap_{k}": v for k, v in laps.items()}}
                return comp, np.array(E), b, sc

            def lean_path(s):
                m, bl, extra = sets[s]
                c0 = clock()
                feats, q, Pn, T, m_struct = lean(inp, pool, seeds, buckets, set(bl))
                xs = None
                if extra:
                    cc = clock()
                    comp = fc.compile(inp, pool, seeds, v2=True) if own_gather else \
                        fc.compile(inp, pool, seeds, embeddings=context.nodes.read(pool), v2=True)
                    xs = comp.scalars[:, columns].astype(np.float16).astype(np.float32)
                    T["compile_for_extra"] = clock() - cc
                c1 = clock()
                f, nq_, bz = lean_inputs(bl, feats, q, Pn, qemb16, xs, look.blocks_idx)
                with torch.no_grad():
                    sc = m(f, torch.ones(1, len(bl)), nq_, 1, bz)
                    torch.topk(sc, min(5, int(sc.shape[0])))
                c2 = clock()
                rec[f"lean:{s}"] = {**T, "forward": c2 - c1, "total": c2 - c0}
                return feats, q, Pn, sc, m_struct

            k = i % len(paths)
            for p in paths[k:] + paths[:k]:
                if p == "twin":
                    out[p] = full_path(twin, False, "twin")
                elif p == "gnn":
                    out[p] = full_path(gnn, True, "gnn")
                else:
                    out[p] = lean_path(p.split(":", 1)[1])
            # checks against the look's row i
            a0, b0 = int(look.off[i]), int(look.off[i + 1])
            if b0 - a0 != pool.size:
                raise SystemExit(f"row {i}: the look's pool has {b0 - a0} nodes, not {pool.size}")
            st = look.score[a0:b0]
            for tag, col in (("twin", 0), ("gnn", 3)):
                dmax = float(np.abs(out[tag][3].numpy() - st[:, col]).max())
                checks[f"{tag}_max_abs"] = max(checks[f"{tag}_max_abs"], dmax)
                checks[f"{tag}_off"] += int(dmax > 1e-3)
            comp_g, E_g = out["gnn"][0], out["gnn"][1]
            fams = {f: int(comp_g.edges[f][0].size) for f in FAMILIES}
            rec["edges"] = fams
            if i < a.check:
                ref = pack_queries_v2([{"pool": comp_g.pool, "x": comp_g.scalars[:, columns], "seedw": comp_g.seedw, "qemb": prep.qemb[i],
                                        "seeds": comp_g.seeds_local, "gold": EMPTY_GOLD, "emb": E_g}], context)
                bad = batch_problems(fast_pack(comp_g, E_g, prep.qemb[i], columns, True), ref, True)
                if bad:
                    checks["pack"].append({"row": i, "fields": bad})
            idx = np.arange(a0, b0)
            for s in sets:
                feats, q, Pn, sc, m_struct = out[f"lean:{s}"]
                m, bl, extra = sets[s]
                checks["lean_rank_off"] += int(not np.array_equal(feats["rank"], look.block("rank", idx).astype(np.float32)))
                if {"WALK", "NBR"} & set(bl):
                    for b in LM.STORED_LEAN:
                        checks[f"lean_{b.lower()}_off"] += int(not np.array_equal(feats[b], look.lean[b][idx].astype(np.float32)))
                lp = look.proj[idx].astype(np.float32)
                checks["lean_proj_off"] += int(not np.array_equal(Pn, lp))
                checks["lean_proj_values_off"] += int((Pn != lp).sum())
                checks["lean_proj_values"] += int(Pn.size)
                fl, nq_, bz, _g, _ix = LM.batch_of(look, np.asarray([i]), bl)
                with torch.no_grad():
                    ref_sc = m(fl, torch.ones(1, len(bl)), nq_, 1, bz)
                dmax = float((sc - ref_sc).abs().max())
                checks["lean_score_max_abs"][s] = max(checks["lean_score_max_abs"][s], dmax)
                checks["lean_score_off"][s] += int(dmax > 1e-4)
                if extra:
                    laps_ms = {kk[4:]: 1e3 * v for kk, v in rec["twin"].items() if kk.startswith("lap_")}
                    share = fams["structural"] / max(1, sum(fams.values()))
                    lean_ms = {b: {"p50": 1e3 * rec[f"lean:{s}"][f"lean_{b}"]} for b in LEAN_TIMED if b in bl}
                    rr_ = rec[f"lean:{s}"]
                    rr_["assembled"] = 1e-3 * LM.cost_ms(bl, laps_ms, share, lean_ms) + rr_["proj_gather"] + rr_["forward"]
            recs.append(rec)
            if i % 50 == 0:
                log(f"  q{i}: n {pool.size}; ms " + ", ".join(f"{p} {1e3 * rec[p]['total']:.2f}" for p in paths))
    finally:
        gc.enable()
    warm = recs[a.warm:]
    summ = {}
    for p in paths:
        stages = sorted({k for r in warm for k in r[p]})
        summ[p] = {k: {"p50": 1e3 * pct([r[p].get(k, 0.0) for r in warm], 50), "p95": 1e3 * pct([r[p].get(k, 0.0) for r in warm], 95),
                       "mean": 1e3 * float(np.mean([r[p].get(k, 0.0) for r in warm]))} for k in stages}
    summ["pool"] = {"total": {"p50": pct([r["pool_ms"] for r in warm], 50), "p95": pct([r["pool_ms"] for r in warm], 95),
                              "mean": float(np.mean([r["pool_ms"] for r in warm]))}}
    ratio = {}
    for p in paths[:1] + paths[2:]:
        ratio[f"{p}/gnn"] = pct([r[p]["total"] / r["gnn"]["total"] for r in warm], 50)
        ratio[f"{p}/gnn+pool"] = pct([(r[p]["total"] + 1e-3 * r["pool_ms"]) / (r["gnn"]["total"] + 1e-3 * r["pool_ms"]) for r in warm], 50)
    for p in paths[2:]:
        ratio[f"{p}/twin"] = pct([r[p]["total"] / r["twin"]["total"] for r in warm], 50)
    log(f"{name}: {len(warm)} warm queries, ms p50 / p95 (stage p50s); threads {THREADS}, own gather {own_gather}")
    for p, v in summ.items():
        log(f"  {p:12s} total {v['total']['p50']:7.2f} / {v['total']['p95']:7.2f}   " +
            ", ".join(f"{k} {x['p50']:.2f}" for k, x in v.items() if k != "total" and not k.startswith("lap_")))
    log(f"  ratios p50: {({k: round(v, 3) for k, v in ratio.items()})}")
    log(f"  checks: {checks}")
    res = {"look": "lean_time", "dataset": name, "carve": "select", "queries": len(recs), "warm_excluded": a.warm, "threads": THREADS,
           "numba_threads": FF.numba.get_num_threads(), "own_gather": own_gather, "summary_ms": summ, "ratios_p50": ratio, "index": index,
           "sets": {s: {"blocks": bl, "compiled_extra": ex} for s, (_m, bl, ex) in sets.items()}, "checks": checks,
           "models": a.models, "models_sha256": hashlib.sha256(Path(a.models).read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": recs, "seconds": time.time() - t_start}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
