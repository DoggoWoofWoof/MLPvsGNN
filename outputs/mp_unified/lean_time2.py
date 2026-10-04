"""Design look (untracked; not a result and not filed): batch-1 serving cost of the lean MLPs, all three kinds:
  - lean_mlp: the random store;
  - lean_mlp2: the seed-centric stand-ins;
  - lean_mlp3: the int8 PCA-256 store.
They are served against the six pair's twin and GNN, one query at a time, on laptop CPU, on the first --queries rows of
<ds>'s M3B select carve (train-derived). No metric is read here. Every lean block and score is checked against the
look the model was fitted on, so the quality those looks read is the quality of the timed path.

Per query, each path is timed with time.perf_counter, and the path order rotates query by query:
  pool   The query's pool and seeds, rebuilt from its cached first-stage lists under the frozen construction (as
         outputs/mp_unified/lean_time.py does). It is the same for every path and is reported once, apart.
  twin   fast compile (every group) -> pack with the compile's own pool edges -> forward. The twin is not edge-free:
         its QLS-U input block takes, per edge family, the mean of a learned projection of the neighbours' embeddings.
  gnn    fast compile -> pack with the compile's own pool edges -> forward.
  <tag>:<set>  one saved lean model, served:
         1. rank lists: list_ranks and fill_retrieval, with dense_cos 0. DLIST takes the shipped dense scores of listed
            nodes;
         2. pool edges: the structural family, plus NER and kNN only when a block reads them, by the fast compile's
            own kernels;
         3. the store gather: 256 B a node in either store;
         4. the set's blocks only, each lapped;
         5. the LeanMLP forward.
         A set with compiled blocks also runs the fast compile, measured.
Index time, once per graph and apart: each store is built here for the nodes the timed pools touch and extrapolated
to the graph.
Checks (every query):
  - the rebuilt pool equals the prepared pool;
  - the twin and GNN scores match the look's stored scores;
  - each lean block equals the reference look's stored block, rows counted (float16 rounding can flip);
  - the lean scores match LeanMLP on lean_mlp.batch_of's inputs from that look;
  - the shipped dense scores equal the look's dense_cos on listed nodes.

    python outputs/mp_unified/lean_time2.py --dataset 2wiki --queries 300 \
        --models ln=outputs/mp_unified/lean/ln-2w_models.pt:lean,pick l2=outputs/mp_unified/lean/l2-2w_models.pt:lean2,pick \
                 l3=outputs/mp_unified/lean/l3-2w_models.pt:lean2,pick --out outputs/mp_unified/lean/time-2w.json
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
import lean_mlp2 as L2  # noqa: E402
import lean_mlp3 as L3  # noqa: E402
import lean_time as LT  # noqa: E402
from mp_retrieval import fast_features as FF  # noqa: E402
from mp_retrieval.m3b_features import FAMILIES, QueryInputs  # noqa: E402
from mp_retrieval.universal_v2_models import pack_queries_v2  # noqa: E402

S6, V2 = LT.S6, LT.V2
FWD, BWD = LT.FWD, LT.BWD
NK = set(L2.FULL_NEW)                                    # blocks that read the NER and kNN edges
EDGE_USERS = {"WALK", "NBR"} | set(L2.NEW) - {"DLIST"}
STORE_USERS = {"SEM", "SEMB", "SEED", "NBR", "NBRF", "DLIST", "DISTS", "DISTF", "NBR2S", "NBR2F"}
log = LT.log
pct = LT.pct


# ── the lean blocks, one at a time (lean_mlp's, lean_mlp2's and lean_mlp3's code, split per block) ──


def walk_blk(n, S, Bk, u, v, fw, bw):
    """lean_mlp.lean_query's WALK on the edges u -> v with direction flags fw, bw."""
    walk = np.zeros((n, 16), np.float32)
    s0 = np.zeros(n, np.float64)
    s0[S] = 1.0
    sb = np.zeros(n, np.float64)
    sb[S[Bk == 0]] = 1.0
    col = 0
    for mask in (None, fw, bw):
        uu, vv = (u, v) if mask is None else (u[mask], v[mask])
        c = s0
        for _h in range(3):
            c = np.bincount(vv, weights=c[uu], minlength=n)
            walk[:, col] = np.log1p(c)
            col += 1
    c = sb
    for _h in range(2):
        c = np.bincount(v, weights=c[u], minlength=n)
        walk[:, col] = np.log1p(c)
        col += 1
    deg = np.bincount(v, minlength=n).astype(np.float32)
    walk[:, col] = np.log1p(deg)
    col += 1
    first = np.full(n, 3, np.int64)
    for h in (2, 1, 0):
        first[walk[:, h] > 0] = h
    first[S] = -1
    for h in range(3):
        walk[:, col + h] = first == h
    walk[:, col + 3] = first == -1
    return walk


def seed_blk(P, S, Bk, unit_store):
    """SEED: lean_mlp's (cosines to the normalised seed mean, the seeds, the bucket-0 seeds) on the random store,
    lean_mlp3's (mean, max and bucket-0 max of the inner products) on the PCA store."""
    n = P.shape[0]
    seed = np.zeros((n, 3), np.float32)
    if S.size:
        G = P @ P[S].T
        if unit_store:
            m = P[S].mean(0)
            m /= max(float(np.linalg.norm(m)), 1e-12)
            seed[:, 0] = P @ m
        else:
            seed[:, 0] = G.mean(1)
        seed[:, 1] = G.max(1)
        if bool((Bk == 0).any()):
            seed[:, 2] = G[:, Bk == 0].max(1)
    return seed


class Served:
    """One saved lean model and the serving path its blocks need. A dropout model read on a pick (drop/pick) keeps its
    full block list; its masked blocks (keep 0) are not computed and enter the forward as zeros, which the mask zeroes
    anyway, so its scores equal the read's."""

    def __init__(self, tag, path, set_name, blob, kind):
        d = blob["models"][set_name]
        self.tag, self.path, self.set, self.kind = tag, path, set_name, kind
        self.blocks = list(d["blocks"])
        self.keep = {b: float(d["keep"].get(b, 1.0)) for b in self.blocks}
        self.active = [b for b in self.blocks if self.keep[b] > 0]
        self.keep_t = torch.tensor([[self.keep[b] for b in self.blocks]], dtype=torch.float32)
        self.store = blob.get("store") or "rand128"
        self.dim = L3.STORE_DIM if self.store == "pca256" else LM.PROJ_DIM
        L3.LeanMLP3.dim = self.dim
        self.model = L3.LeanMLP3(d["blocks"], d["widths"], d["hidden"])
        self.model.load_state_dict(d["state"])
        self.model.eval()
        self.extra = [b for b in self.active if b in LM.COMPILED and b != "rank"]
        self.need = set(self.active)
        self.name = f"{tag}:{set_name}"


class Lean:
    """The lean read path over a FastCompiler's kernels and arrays and the index-time stores."""

    def __init__(self, fc, store16, R, codes, pstore):
        self.fc, self.store16, self.R, self.codes, self.pstore = fc, store16, R, codes, pstore
        self.cols = np.asarray([V2.IDX[c] for c in LT.RET_COLS], dtype=np.int64)
        self.rank_idx = np.asarray([V2.IDX[c] for c in LM.SPLIT["rank"]], dtype=np.int64)
        self.rrf_col, self.dr_col = V2.IDX["rrf"], V2.IDX["dense_rr"]

    def __call__(self, inp, pool, seeds, buckets, sv):
        """The set's blocks (float32 values rounded through float16, as the looks store them), the store's query side
        and node side, and the seconds per stage."""
        fc, need = self.fc, sv.need
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
                   "_rrf": X[:, self.rrf_col].astype(np.float16).astype(np.float32),
                   "_dscore": np.where(d_in, d_score, 0.0).astype(np.float32), "_din": d_in.copy()}
            t1 = clock()
            T["rank"] = t1 - t
            eu = ev = np.empty(0, np.int64)
            efam = np.empty(0, np.int8)
            efwd = ebwd = np.empty(0, np.bool_)
            if need & EDGE_USERS:
                s = fc._s["structural"]
                _ev, _eu, erel, edir, pair_id, pu, pv = K.typed_edges(s["tindptr"], s["tcol"], s["trel"], s["tdir"], pool, lookup, fc.cap)
                attr = K.typed_attr(pair_id, erel, edir, np.zeros(1, np.float32), False, int(pu.size))
                us, vs, fs = [pu], [pv], [np.zeros(pu.size, np.int8)]
                fws, bws = [attr[:, FWD] > 0.5], [attr[:, BWD] > 0.5]
                if need & NK:
                    for f_i, fam in ((1, "ner"), (2, "knn")):
                        a_ = fc._s[fam]
                        u_, v_, _w = K.weighted_edges(a_["indptr"], a_["col"], a_["wbits"], fc._lut16, pool, lookup, fc.cap)
                        us.append(u_), vs.append(v_), fs.append(np.full(u_.size, f_i, np.int8))
                        fws.append(np.zeros(u_.size, np.bool_)), bws.append(np.zeros(u_.size, np.bool_))
                eu, ev = np.concatenate(us).astype(np.int64), np.concatenate(vs).astype(np.int64)
                efam, efwd, ebwd = np.concatenate(fs), np.concatenate(fws), np.concatenate(bws)
            t2 = clock()
            T["edges"] = t2 - t1
        finally:
            lookup[pool] = -1
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
        T["blk_SEM"] = t4 - t3                                   # the store's decode, query side and inner products
        valid = sl >= 0
        S, Bk = sl[valid], buckets[valid]
        st = efam == 0
        u_s, v_s = eu[st], ev[st]
        lap = clock()

        def done(b):
            nonlocal lap
            now = clock()
            T[f"blk_{b}"] = T.get(f"blk_{b}", 0.0) + now - lap
            lap = now

        if "SEED" in need:
            out["SEED"] = seed_blk(P, S, Bk, unit_store)
            done("SEED")
        if "WALK" in need:
            out["WALK"] = walk_blk(n, S, Bk, u_s, v_s, efwd[st], ebwd[st])
            done("WALK")
        if "NBR" in need:
            out["NBR"] = L3.nbr_of(n, P, q, cos, u_s, v_s)
            done("NBR")
        if "DLIST" in need:
            listed = X[:, self.dr_col] > 0
            out["DLIST"] = np.stack([np.where(listed, out["_dscore"], 0.0), listed, np.where(listed, 0.0, cos)], 1).astype(np.float32)
            done("DLIST")
        if need & {"DISTS", "DISTF", "NBR2S", "NBR2F"}:
            rrf16 = out["_rrf"].astype(np.float64)                 # the look's float16 rrf column, as lean_mlp2 reads it
            s_seed = rrf16[S] / max(float(rrf16.max()), 1e-12) if S.size else np.zeros(0)
            for sel, d_name, n2_name in ((st, "DISTS", "NBR2S"), (np.ones_like(st), "DISTF", "NBR2F")):
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
                out["WALKF"] = np.concatenate([walk_blk(n, S, Bk, eu, ev, efwd | sym, ebwd | sym), wx], 1)
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
        return out, q, P16, T, int(u_s.size), int(eu.size)


def lean_inputs(sv, feats, q, P, qemb16, extra_x=None, blocks_idx=None):
    """LeanMLP's inputs for one query, as lean_mlp.batch_of assembles them; masked blocks as zeros."""
    n = P.shape[0]
    f = {}
    for b in sv.blocks:
        if b not in sv.need:
            f[b] = (torch.zeros(1, 1536), torch.zeros(n, sv.dim)) if b == "SEMB" else torch.zeros(n, sv.model.widths[b])
        elif b == "SEM":
            prod = P * q[None, :]
            f[b] = torch.from_numpy(np.concatenate([prod, prod.sum(1, keepdims=True)], axis=1))
        elif b == "SEMB":
            f[b] = (torch.from_numpy(qemb16.astype(np.float32)[None, :]), torch.from_numpy(P))
        elif b in feats:
            f[b] = LM.clean(feats[b])
        else:
            f[b] = LM.clean(extra_x[:, blocks_idx[b]])
    nq = torch.zeros(n, dtype=torch.long)
    base_z = LM.seg_zscore(torch.from_numpy(feats["_rrf"]).unsqueeze(1), nq, 1).squeeze(1)
    return f, nq, base_z


# ── main ─────────────────────────────────────────────────────────────────────


def parse_models(specs):
    """tag=file.pt:setA,setB ... -> [(tag, path, [sets])]."""
    out = []
    for spec in specs:
        tag, rest = spec.split("=", 1)
        path, sets = rest.rsplit(":", 1)
        out.append((tag, path, sets.split(",")))
    return out


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
    log(f"{name}: {nq_all} select queries prepared ({time.time() - t_start:.0f}s); pools mean {np.mean([p.size for p in prep.pools]):.0f}")
    fc = FF.compiler_for(context.stores, context.nodes, context.rel_table)
    if fc is None:
        raise SystemExit("no fast compiler for this context")
    own_gather = fc._h16 is not None or fc._shards is not None
    # the saved models and their stores
    served, blobs = [], {}
    pca_basis = None
    for tag, path, sets in parse_models(a.models):
        blob = torch.load(path, weights_only=False)
        blobs[tag] = (path, blob)
        kind = "lean_mlp3" if blob.get("store") else ("lean_mlp2" if any(b in L2.NEW for b in blob.get("present", [])) else "lean_mlp")
        if blob.get("store") == "pca256":
            if pca_basis is not None and not np.array_equal(pca_basis["V"], blob["basis"]["V"]):
                raise SystemExit("two pca256 model files with different bases")
            pca_basis = blob["basis"]
        for s in sets:
            sv = Served(tag, path, s, blob, kind)
            served.append(sv)
            log(f"{sv.name} ({kind}, store {sv.store}): {sv.blocks}" + (f"; compiled blocks {sv.extra}: also runs the fast compile" if sv.extra else ""))
    # the index-time stores, for the nodes the timed pools touch
    R = LM.projection()
    nodes = np.unique(np.concatenate(prep.pools))
    index = {"touched_nodes": int(nodes.size), "graph_nodes": int(fc.n_nodes), "embedding_mib_graph": fc.n_nodes * 1536 * 2 / 2 ** 20}
    store16 = np.zeros((fc.n_nodes, LM.PROJ_DIM), dtype=np.float16)
    pstore = L3.Store(pca_basis) if pca_basis is not None else None
    codes = np.zeros((fc.n_nodes, L3.STORE_K), dtype=np.int8) if pstore is not None else None
    t_read = t_r = t_p = 0.0
    for s0 in range(0, nodes.size, 4096):
        rr = nodes[s0:s0 + 4096]
        t = time.time()
        En = np.asarray(context.nodes.read(rr), dtype=np.float32)
        En = En / np.maximum(np.linalg.norm(En, axis=1, keepdims=True), 1e-12)
        t_read += time.time() - t
        t = time.time()
        store16[rr] = (En @ R).astype(np.float16)
        t_r += time.time() - t
        if pstore is not None:
            t = time.time()
            codes[rr] = pstore.codes(En)
            t_p += time.time() - t
    for key, sec, width in (("rand128", t_r, LM.PROJ_DIM * 2), ("pca256", t_p, L3.STORE_K)):
        if key == "pca256" and pstore is None:
            continue
        index[key] = {"seconds": sec, "us_per_node": 1e6 * sec / max(1, nodes.size), "graph_seconds": sec * fc.n_nodes / max(1, nodes.size),
                      "bytes_per_node": width, "mib_graph": fc.n_nodes * width / 2 ** 20}
    index["row_read_seconds"] = t_read
    log(f"stores over {nodes.size} touched nodes (row reads {t_read:.1f}s): " +
        ", ".join(f"{k} {v['us_per_node']:.2f} us/node, ~{v['graph_seconds']:.0f}s and {v['mib_graph']:.0f} MiB for the graph's {fc.n_nodes}"
                  for k, v in index.items() if isinstance(v, dict)))
    lean = Lean(fc, store16, R, codes, pstore)
    # the reference looks, one per model file (the model's own carve type and store)
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
    ci = {c: i for i, c in enumerate(look0.columns)}
    paths = ["twin", "gnn"] + [sv.name for sv in served]
    by_name = {sv.name: sv for sv in served}
    clock = time.perf_counter
    checks = {"pool": 0, "pack": [], "twin_max_abs": 0.0, "gnn_max_abs": 0.0, "twin_off": 0, "gnn_off": 0,
              "dscore_vs_dense_cos_max_abs": 0.0, "dscore_rows_off": 0,
              "blocks_off": {sv.name: {} for sv in served}, "lean_score_max_abs": {sv.name: 0.0 for sv in served},
              "lean_score_off": {sv.name: 0 for sv in served}}
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
                b = LT.fast_pack(comp, E, prep.qemb[i], columns, with_edges)
                c2 = clock()
                with torch.no_grad():
                    sc = model(V2.arm_view(model, b, inputs))
                    torch.topk(sc, min(5, int(sc.shape[0])))
                c3 = clock()
                rec[tag] = {"compile": c1 - c0, "pack": c2 - c1, "forward": c3 - c2, "total": c3 - c0,
                            **{f"lap_{k}": v for k, v in laps.items()}}
                return comp, np.array(E), b, sc

            def lean_path(sv):
                c0 = clock()
                feats, q, P, T, m_struct, m_all = lean(inp, pool, seeds, buckets, sv)
                xs = None
                est = 0.0
                if sv.extra:
                    # the fast compile has no partial mode, so a set with compiled blocks runs all of it (timed, an
                    # upper bound); compile_need_est prices only the stages those blocks read, by lean_mlp.cost_ms
                    # on this query's own laps
                    laps = {}
                    cc = clock()
                    comp = fc.compile(inp, pool, seeds, timings=laps, v2=True) if own_gather else \
                        fc.compile(inp, pool, seeds, embeddings=context.nodes.read(pool), timings=laps, v2=True)
                    xs = comp.scalars[:, columns].astype(np.float16).astype(np.float32)
                    T["compile_for_extra"] = clock() - cc
                    ne = {f: int(comp.edges[f][0].size) for f in FAMILIES}
                    share = ne["structural"] / max(1, sum(ne.values()))
                    est = 1e-3 * LM.cost_ms(sv.extra, {k: 1e3 * v for k, v in laps.items()}, share, {})
                    T["compile_need_est"] = est
                c1 = clock()
                f, nq_, bz = lean_inputs(sv, feats, q, P, qemb16, xs, refs[sv.tag].blocks_idx)
                with torch.no_grad():
                    sc = sv.model(f, sv.keep_t, nq_, 1, bz)
                    torch.topk(sc, min(5, int(sc.shape[0])))
                c2 = clock()
                rec[sv.name] = {**T, "forward": c2 - c1, "total": c2 - c0, "edges_struct": m_struct, "edges_all": m_all}
                if sv.extra:
                    rec[sv.name]["total_need_est"] = c2 - c0 - T["compile_for_extra"] + est
                return feats, sc

            k = i % len(paths)
            for p in paths[k:] + paths[:k]:
                if p == "twin":
                    out[p] = full_path(twin, True, "twin")
                elif p == "gnn":
                    out[p] = full_path(gnn, True, "gnn")
                else:
                    out[p] = lean_path(by_name[p])
            # checks against the looks' row i
            a0, b0 = int(look0.off[i]), int(look0.off[i + 1])
            if b0 - a0 != pool.size:
                raise SystemExit(f"row {i}: the look's pool has {b0 - a0} nodes, not {pool.size}")
            st_sc = look0.score[a0:b0]
            for tag, col in (("twin", 0), ("gnn", 3)):
                dmax = float(np.abs(out[tag][3].numpy() - st_sc[:, col]).max())
                checks[f"{tag}_max_abs"] = max(checks[f"{tag}_max_abs"], dmax)
                checks[f"{tag}_off"] += int(dmax > 1e-3)
            comp_g, E_g = out["gnn"][0], out["gnn"][1]
            rec["edges"] = {f: int(comp_g.edges[f][0].size) for f in FAMILIES}
            if i < a.check:
                ref = pack_queries_v2([{"pool": comp_g.pool, "x": comp_g.scalars[:, columns], "seedw": comp_g.seedw, "qemb": prep.qemb[i],
                                        "seeds": comp_g.seeds_local, "gold": LT.EMPTY_GOLD, "emb": E_g}], context)
                bad = LT.batch_problems(LT.fast_pack(comp_g, E_g, prep.qemb[i], columns, True), ref, True)
                if bad:
                    checks["pack"].append({"row": i, "fields": bad})
            idx = np.arange(a0, b0)
            # the shipped dense scores against the look's dense_cos, on listed nodes
            sv0 = served[0]
            f0 = out[sv0.name][0]
            din = f0["_din"]
            if din.any():
                dd = np.abs(f0["_dscore"][din] - look0.x[idx][din, ci["dense_cos"]].astype(np.float32))
                checks["dscore_vs_dense_cos_max_abs"] = max(checks["dscore_vs_dense_cos_max_abs"], float(dd.max()))
                checks["dscore_rows_off"] += int(float(dd.max()) > 2e-3)
            for sv in served:
                feats, sc = out[sv.name]
                look = refs[sv.tag]
                offs = checks["blocks_off"][sv.name]
                for b in sv.active:
                    if b in ("SEM", "SEMB") or b in sv.extra:
                        continue
                    mine = feats[b]
                    theirs = look.block(b, idx).astype(np.float32)
                    if mine.shape != theirs.shape:
                        raise SystemExit(f"{sv.name} {b}: {mine.shape} against the look's {theirs.shape}")
                    rows_off = int((~((mine == theirs) | (np.isnan(mine) & np.isnan(theirs)))).any(1).sum())
                    offs[b] = offs.get(b, 0) + rows_off
                fl, nq_, bz, _g, _ix = LM.batch_of(look, np.asarray([i]), sv.blocks)
                with torch.no_grad():
                    ref_sc = sv.model(fl, sv.keep_t, nq_, 1, bz)
                dmax = float((sc - ref_sc).abs().max())
                checks["lean_score_max_abs"][sv.name] = max(checks["lean_score_max_abs"][sv.name], dmax)
                checks["lean_score_off"][sv.name] += int(dmax > 1e-3)
            recs.append(rec)
            if i % 50 == 0:
                log(f"  q{i}: n {pool.size}; ms " + ", ".join(f"{p} {1e3 * rec[p]['total']:.2f}" for p in paths))
    finally:
        gc.enable()
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
        if p == "gnn":
            continue
        ratio[f"{p}/gnn"] = pct([r[p]["total"] / r["gnn"]["total"] for r in warm], 50)
        ratio[f"{p}/gnn+pool"] = pct([(r[p]["total"] + 1e-3 * r["pool_ms"]) / (r["gnn"]["total"] + 1e-3 * r["pool_ms"]) for r in warm], 50)
        if p != "twin":
            ratio[f"{p}/twin"] = pct([r[p]["total"] / r["twin"]["total"] for r in warm], 50)
        if warm and "total_need_est" in warm[0][p]:
            ratio[f"{p}(need_est)/gnn"] = pct([r[p]["total_need_est"] / r["gnn"]["total"] for r in warm], 50)
    log(f"{name}: {len(warm)} warm queries, ms p50 / p95 (stage p50s); threads {THREADS}, own gather {own_gather}")
    for p, v in summ.items():
        log(f"  {p:16s} total {v['total']['p50']:7.2f} / {v['total']['p95']:7.2f}   " +
            ", ".join(f"{k} {x['p50']:.2f}" for k, x in v.items() if k != "total" and not k.startswith("lap_")))
    log(f"  ratios p50: {({k: round(v, 3) for k, v in ratio.items()})}")
    log(f"  checks: {checks}")
    res = {"look": "lean_time2", "dataset": name, "carve": "select", "queries": len(recs), "warm_excluded": a.warm, "threads": THREADS,
           "numba_threads": FF.numba.get_num_threads(), "own_gather": own_gather, "summary_ms": summ, "ratios_p50": ratio, "index": index,
           "sets": {sv.name: {"blocks": sv.blocks, "compiled_extra": sv.extra, "store": sv.store, "kind": sv.kind} for sv in served},
           "checks": checks, "models": {tag: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                                        for tag, (path, _b) in blobs.items()},
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": recs, "seconds": time.time() - t_start}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
