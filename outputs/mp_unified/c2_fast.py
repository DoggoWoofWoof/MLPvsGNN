"""C2 (docs/C2_MLP_COLD_FAST.md): the MLP's exact cold serving path, zrc's rows from the first-stage lists computing
only what zrc reads. C1 (docs/C1_COLD_COST.md, outputs/mp_unified/c1_cold.py, frozen) served zrc through the six GNN's
whole compile and lean_cache's line functions, which compute blocks zrc never reads. Per question C2 computes:

  compile   fast_features.FastCompiler's own kernels and buffers, in its order, for the columns zrc reads only:
            retrieval (rank, dense_cos, rrf), the three families' pool edges, topology on the STRUCT view, the seed
            products E @ E_S.T, and the depth basis on the STRUCT view. Skipped: the NER, KNN and FULL topology views,
            the edge weights, C (neighbour aggregation), GCS, D's seed prototypes, B and the typed basis, and the FULL
            depth view; none of them feeds a column zrc reads (the check below holds every column bit for bit).
  lean      WALK and WALKF by lean_mlp.lean_query's WALK lines alone (the same numpy operations in the same order; its
            SEM, SEED and NBR blocks, and the node projection they read, are not used by zrc), SEED and DISTS by
            lean_cache.store_seed_dists unchanged.
The forward is zrc's (zrm.ZRM) on lean_gpu.CacheCarve.batch's inputs, as in C1.

    python outputs/mp_unified/c2_fast.py check --dataset musique --queries 50       (dev: bit-for-bit against C1's path)
    python outputs/mp_unified/c2_fast.py run --dataset musique --queries 200        (the timed run, pinned)
    python outputs/mp_unified/c2_fast.py report
    python outputs/mp_unified/c2_fast.py --selftest
"""
import os
import sys

THREADS = 1
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
           "LEAN_TIME_THREADS"):
    os.environ[_v] = str(THREADS)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c1_cold as C1  # noqa: E402

FF, LM, LC, ML8, T2 = C1.FF, C1.LM, C1.LC, C1.ML8, C1.T2
V2, IDX = FF.V2, FF.IDX
MLP_VIEW = "STRUCT"
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "c2"
PATHS = ("zrc2", "zsp2", "zrc1", "gnn6")
clock = time.perf_counter


# ── the partial compile ──────────────────────────────────────────────────────


def compile_mlp(fc, inp, pool, seeds, embeddings=None, timings=None):
    """FastCompiler.compile (v2) for the columns zrc reads; every other column is 0. Same kernels, buffers, operands
    and order as fast_features.FastCompiler._compile for what it computes."""
    if fc.threads is not None and fc.threads > 1:
        FF.numba.set_num_threads(fc.threads)
    lap = FF._Lap(timings)
    pool = np.asarray(pool, dtype=np.int64)
    n = int(pool.size)
    if n > 1 and not bool(np.all(pool[1:] > pool[:-1])):
        raise ValueError("the pool must be sorted ascending without repeats")
    small = n < FF.SERIAL_BELOW
    fc._k = fc._K_small if small else fc.K
    fc._par = fc.parallel and not small
    lookup = fc._lookup
    lookup[pool] = np.arange(n, dtype=np.int32)
    try:
        return _compile_mlp(fc, inp, pool, n, seeds, embeddings, lap)
    finally:
        lookup[pool] = -1


def _compile_mlp(fc, inp, pool, n, seeds, embeddings, lap):
    K = fc._k
    lookup = fc._lookup
    X = np.zeros((n, V2.N_COLUMNS), dtype=np.float32)
    q = np.asarray(inp.q, dtype=np.float32)
    seeds = np.asarray(seeds, dtype=np.int64)
    if seeds.size and (seeds.min() < 0 or seeds.max() >= fc.n_nodes or (lookup[seeds] < 0).any()):
        raise ValueError("every seed must be a pool member")
    seeds_local = lookup[seeds].astype(np.int64)
    fc._buffers(n)
    E, E_norm = fc._embeddings(pool, embeddings, n)
    dense_cos = (E @ q).astype(np.float32)

    # retrieval (FastCompiler._compile's lines)
    d_rank = np.zeros(n, np.float32)
    d_score = np.zeros(n, np.float32)
    d_in = np.zeros(n, np.bool_)
    s_rank = np.zeros(n, np.float32)
    s_score = np.zeros(n, np.float32)
    s_in = np.zeros(n, np.bool_)
    K.list_ranks(np.asarray(inp.dense_ids, dtype=np.int64), np.asarray(inp.dense_scores).astype(np.float32), lookup,
                 fc.n_nodes, d_rank, d_score, d_in)
    K.list_ranks(np.asarray(inp.splade_ids, dtype=np.int64), np.asarray(inp.splade_scores).astype(np.float32), lookup,
                 fc.n_nodes, s_rank, s_score, s_in)
    top_splade = float(inp.splade_scores[0]) if inp.splade_scores.size else 0.0
    rrf = np.empty(n, np.float32)
    cols = np.asarray([IDX[c] for c in ("dense_cos", "dense_rr", "dense_in", "splade_rr", "splade_in",
                                        "splade_score_norm", "rrf", "agreement", "is_seed", "seed_rank")], dtype=np.int64)
    K.fill_retrieval(X, dense_cos, d_in, d_rank, s_in, s_rank, s_score, top_splade > 0,
                     np.float32(max(top_splade, 1e-12)), seeds_local, cols, rrf)
    lap("retrieval")

    # pool graph edges, all three families (WALKF and DISTS read them)
    relcos = (fc._rel_emb @ q).astype(np.float32) if fc._rel_emb is not None else None
    edges = {}
    s_arr = fc._s["structural"]
    _ev, _eu, erel, edir, pair_id, pu, pv = K.typed_edges(s_arr["tindptr"], s_arr["tcol"], s_arr["trel"],
                                                          s_arr["tdir"], pool, lookup, fc.cap)
    attr = K.typed_attr(pair_id, erel, edir, relcos if relcos is not None else np.zeros(1, np.float32),
                        relcos is not None, int(pu.size))
    edges["structural"] = (pu, pv, attr)
    for fam in ("ner", "knn"):
        a = fc._s[fam]
        u, v, w = K.weighted_edges(a["indptr"], a["col"], a["wbits"], fc._lut16, pool, lookup, fc.cap)
        edges[fam] = (u, v, K.weight_attr(w))
    lap("edges")

    # topology, the STRUCT view only
    view = MLP_VIEW
    fams = FF.VIEW_FAMILIES[view]
    S = int(seeds_local.size)
    u = np.concatenate([edges[f][0] for f in fams])
    v = np.concatenate([edges[f][1] for f in fams])
    ptr, idx = K.sym_csr(n, u, v)
    csr = {view: (ptr, idx)}
    ptrs = [fc._s[f]["indptr"] for f in fams] + [fc._s[fams[0]]["indptr"]] * (3 - len(fams))
    deg_global = K.deg_global(pool, ptrs[0], ptrs[1], ptrs[2], len(fams))
    r1 = np.zeros((n, S), np.bool_)
    r2 = np.zeros((n, S), np.bool_)
    c = np.asarray([IDX[name.format(view=view)] for name in FF._TOPO_NAMES], dtype=np.int64)
    K.topology(ptr, idx, seeds_local, n, X, c, deg_global, r1, r2)
    log_cols = [IDX[name.format(view=view)] for name in FF._TOPO_LOG]
    lap("topology")

    # D's seed products (the depth basis reads them)
    E_S = E[seeds_local]
    EST = E @ E_S.T
    lap("D")
    if log_cols:
        X[:, log_cols] = np.log1p(X[:, log_cols])

    # the depth basis (FastCompiler._depth_basis), the STRUCT view only
    s = np.zeros(n, dtype=np.float32)
    s[seeds_local] = rrf[seeds_local] / max(float(rrf.max()), 1e-12)
    s_seed = s[seeds_local].astype(np.float64)
    cos32 = dense_cos.astype(np.float32)
    shifted = cos32 + 2.0
    G = EST.astype(np.float64)
    gram = (E_S @ E_S.T).astype(np.float64)
    E_norm64 = E_norm.astype(np.float64)
    qsum = np.empty(n, np.float32)
    rn = np.empty(n, np.float32)
    qmax = np.empty(n, np.float32)
    blocks = fc.ring.blocks(n, FF.blas_threads())
    log32 = []
    saved = V2.VIEWS
    V2.VIEWS = (view,)
    try:
        fc._depth_views(X, n, csr, seeds_local, s, s_seed, cos32, shifted, G, gram, E_norm64, S, qsum, rn, qmax, blocks,
                        log32)
    finally:
        V2.VIEWS = saved
    X[:, log32] = np.log1p(X[:, log32])
    lap("depth_basis")
    return FF.Compiled(pool=pool, seeds_local=seeds_local, scalars=X, edges=edges, seedw=None)


# ── the lean inputs ──────────────────────────────────────────────────────────


def walk_block(n, seeds, buckets, eu, ev, efam, efwd, ebwd):
    """lean_mlp.lean_query's WALK block, its lines alone."""
    valid = seeds >= 0
    S, Bk = seeds[valid].astype(np.int64), buckets[valid]
    st = efam == 0
    u, v = eu[st].astype(np.int64), ev[st].astype(np.int64)
    fw, bw = efwd[st].astype(bool), ebwd[st].astype(bool)
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


def walkf_block(n, seeds, buckets, eu, ev, efam, efwd, ebwd):
    """lean_cache.walkf_of's lines with lean_query's WALK block alone."""
    valid = seeds >= 0
    S = seeds[valid].astype(np.int64)
    eu64, ev64 = eu.astype(np.int64), ev.astype(np.int64)
    fam0 = np.zeros_like(efam)
    sym = efam > 0
    fw = np.where(sym, 1, efwd).astype(efwd.dtype)
    bw = np.where(sym, 1, ebwd).astype(ebwd.dtype)
    W = walk_block(n, seeds, buckets, eu, ev, fam0, fw, bw)
    s0 = np.zeros(n)
    s0[S] = 1.0
    wx = np.zeros((n, 2), np.float32)
    for j, f in enumerate((1, 2)):
        m = efam == f
        wx[:, j] = np.log1p(np.bincount(ev64[m], weights=s0[eu64[m]], minlength=n))
    return np.concatenate([W, wx], 1)


def shared_mlp(su, i, T):
    """C1.shared with the partial compile: pool, read, compile; times into T."""
    p = su.prep
    d_ids, s_ids = np.asarray(p.dense_ids[i]), np.asarray(p.splade_ids[i])
    c0 = clock()
    pool, seeds = su.builder(d_ids, s_ids)
    pool, seeds = np.asarray(pool, np.int64), np.asarray(seeds, np.int64)
    c1 = clock()
    E = su.context.nodes.read(pool)
    c2 = clock()
    inp = T2.QueryInputs(p.qemb[i], p.dense_ids[i], p.dense_scores[i], p.splade_ids[i], p.splade_scores[i])
    comp = compile_mlp(su.fc, inp, pool, seeds, embeddings=E)
    c3 = clock()
    T["pool"], T["read"], T["compile"] = c1 - c0, c2 - c1, c3 - c2
    top = {int(d_ids[0]), int(s_ids[0])}
    bucket = np.asarray([0 if int(s) in top else 1 for s in seeds], dtype=np.int64)
    return {"pool": pool, "seeds": seeds, "bucket": bucket, "E": E, "comp": comp, "qemb": np.asarray(p.qemb[i])}


def lean_rows_mlp(su, sh, T):
    """C1.lean_rows for zrc's columns: edges, then the float16 rows."""
    c0 = clock()
    comp = sh["comp"]
    n = int(comp.pool.size)
    eu, ev, ef, efw, ebw = C1.edges_of(comp)
    c1 = clock()
    sl = np.full(ML8.MAX_SEEDS, -1, np.int64)
    sb = np.full(ML8.MAX_SEEDS, -1, np.int64)
    s_loc = np.asarray(comp.seeds_local, np.int64)
    sl[:s_loc.size], sb[:s_loc.size] = s_loc, sh["bucket"]
    x16 = np.asarray(comp.scalars[:, su.columns], dtype=np.float16)
    walk = walk_block(n, sl, sb, eu, ev, ef, efw, ebw)
    walkf = walkf_block(n, sl, sb, eu, ev, ef, efw, ebw)
    codes = su.codes[sh["pool"]]
    seed, dists = LC.store_seed_dists(n, codes, su.store, sl, sb, x16[:, su.c_rrf_x], eu, ev, ef)
    X = np.empty((n, su.W), np.float16)
    X[:, :su.span[LC.XC_BLOCKS[-1]][1]] = x16[:, su.xcols]
    for b, arr in (("WALK", walk), ("WALKF", walkf), ("SEED", seed), ("DISTS", dists)):
        a, e = su.span[b]
        X[:, a:e] = arr.astype(np.float16)
    c2 = clock()
    T["edges"], T["lean"] = c1 - c0, c2 - c1
    return {"n": n, "X": X, "codes": codes, "q_emb16": np.asarray(sh["qemb"], np.float32).astype(np.float16),
            "e": (eu, ev, ef, efw, ebw), "sl": sl, "sb": sb}


# ── the development check ────────────────────────────────────────────────────


def same(a, b):
    """Equal shape, dtype and bits."""
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(C1.bits(a), C1.bits(b))


def check(name, nq):
    """Every question: the partial compile's zrc columns, the edges and the float16 rows equal C1's path bit for bit,
    and zrc's scores equal; the per-stage times beside (development, not a filed number)."""
    C1.pin(2)
    su = C1.Setup(name, nq)
    acc, bad = {}, []
    for i in [nq] + list(range(nq)):        # the carve's next question first: numba's compile, untimed
        T1, T2_ = {}, {}
        ref_sh = C1.shared(su, i, T1)
        ref = C1.lean_rows(su, ref_sh, T1)
        sh = shared_mlp(su, i, T2_)
        r = lean_rows_mlp(su, sh, T2_)
        cols = np.asarray(su.columns)[np.r_[su.xcols, su.c_rrf_x]]
        ok = {"pool": np.array_equal(sh["pool"], ref_sh["pool"]),
                "cols": same(sh["comp"].scalars[:, cols], ref_sh["comp"].scalars[:, cols]),
                "edges": all(np.array_equal(a, b) for a, b in zip(r["e"], ref["e"])),
                "X": same(r["X"], ref["X"]), "codes": np.array_equal(r["codes"], ref["codes"])}
        rows_a = C1.Rows([r], su.span, su.W, su.c_rrf, su.store)
        rows_b = C1.Rows([ref], su.span, su.W, su.c_rrf, su.store)
        sa, sb_ = C1.forward(su, "zrc", rows_a, 1, False), C1.forward(su, "zrc", rows_b, 1, False)
        ok["scores"] = bool((sa == sb_).all())
        if not all(ok.values()):
            bad.append((i, {k: v for k, v in ok.items() if not v}))
        if i == nq:
            continue
        for k in ("compile", "edges", "lean"):
            acc.setdefault(f"C1 {k}", []).append(T1[k])
            acc.setdefault(f"C2 {k}", []).append(T2_[k])
    C1.log(f"c2 check {name}: {nq} questions, {len(bad)} differ from C1's path {bad[:5]}")
    for k, v in acc.items():
        C1.log(f"  {k:12s} p50 {np.median(v) * 1e3:8.2f} ms")
    return 0 if not bad else 1


# ── the timed run (docs/C2_MLP_COLD_FAST.md) ─────────────────────────────────


def run_path(su, k, i, T):
    """One path for question i from the first-stage lists, cold, every stage of its own timed into T; returns
    (scores, shared dict, row dict or None)."""
    if k == "gnn6":
        sh = C1.shared(su, i, T)
        sc, _ = C1.run_path(su, "gnn6", sh, T)
        return sc, sh, None
    if k == "zrc1":
        sh = C1.shared(su, i, T)
        sc, r = C1.run_path(su, "zrc", sh, T)
        return sc, sh, r
    sh = shared_mlp(su, i, T)
    r = lean_rows_mlp(su, sh, T)
    c0 = clock()
    if k == "zsp2":
        eu, ev, ef = r["e"][:3]
        r["links"] = C1.links_of(r["n"], eu, ev, ef)
    rows = C1.Rows([r], su.span, su.W, su.c_rrf, su.store)
    c1 = clock()
    model = "zsp" if k == "zsp2" else "zrc"
    sc = C1.forward(su, model, rows, 1, links=(model == "zsp"))
    torch.topk(sc, min(5, r["n"]))
    c2 = clock()
    T["links" if k == "zsp2" else "rows"], T["forward"] = c1 - c0, c2 - c1
    return sc, sh, r


def check_c2(su, i, out, cache):
    """Untimed. C2's rows and scores against C1's form on the same question (bit for bit), the pools and seeds against
    the look's, the six GNN against the look's stored scores, and zrc's and zsp's scores against the step-1 cache's."""
    ref = su.ref[i]
    sc, sh, r = {k: v[0] for k, v in out.items()}, {k: v[1] for k, v in out.items()}, {k: v[2] for k, v in out.items()}
    c = {"pool": all(bool(np.array_equal(sh[k]["pool"], ref["pool"])) for k in PATHS)}
    c["seeds"] = all(bool(np.array_equal(r[k]["sl"], ref["sl"]) and np.array_equal(r[k]["sb"], ref["sb"]))
                     for k in ("zrc2", "zsp2", "zrc1"))
    c["edges"] = all(bool(np.array_equal(a, b)) for k in ("zrc2", "zsp2") for a, b in zip(r[k]["e"], r["zrc1"]["e"]))
    c["rows_bits"] = all(same(r[k]["X"], r["zrc1"]["X"]) and np.array_equal(r[k]["codes"], r["zrc1"]["codes"])
                         for k in ("zrc2", "zsp2"))
    c["zrc2_eq_zrc1"] = bool(torch.equal(sc["zrc2"], sc["zrc1"]))
    # zsp in C1's form on C1's rows of this question
    ra = dict(r["zrc1"])
    eu, ev, ef = ra["e"][:3]
    ra["links"] = C1.links_of(ra["n"], eu, ev, ef)
    s1 = C1.forward(su, "zsp", C1.Rows([ra], su.span, su.W, su.c_rrf, su.store), 1, links=True)
    c["zsp2_eq_zsp1"] = bool(torch.equal(sc["zsp2"], s1))
    g = sc["gnn6"].numpy()
    c["gnn6_vs_look"] = float(np.abs(g - ref["gnn0"].astype(np.float32)).max()) if g.size else 0.0
    c["gnn6_top5_same"] = bool(np.array_equal(C1.top5(g), C1.top5(ref["gnn0"].astype(np.float32))))
    if cache is not None:
        c.update(C1.check_scores_on_cache(su, cache, i, {"zrc": sc["zrc2"], "zsp": sc["zsp2"]}, ref["e"][:3]))
    return c


def batch_group(su, group, group_sh):
    """C1's batch-16 forward over a group (zrc and zsp on C2's rows, the six GNN on its own compile)."""
    out = C1.batch_group(su, group, group_sh)
    return {"size": out["size"], "zrc2": out["zrc"], "zsp2": out["zsp"], "gnn6": out["gnn6"]}


def run(a):
    t_start = time.time()
    pinned = C1.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    torch.set_num_threads(THREADS)
    C1.log(f"{a.dataset}: pin {pinned}")
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    if a.dataset in C1.TYPED:
        raise SystemExit(f"{a.dataset}: typed graphs follow the four untyped datasets (docs/C2_MLP_COLD_FAST.md)")
    su = C1.Setup(a.dataset, a.queries)
    nq = len(su.prep.pools) - 1
    C1.log(f"{a.dataset}: {nq} measured questions, index {su.index}")
    cache = C1.LG.CacheCarve(a.dataset, C1.CARVE, su.basis, "cpu")       # untimed; the checks' reference
    if list(cache.ids[:nq]) != list(su.ids[:nq]) or cache.span != su.span or cache.W != su.W:
        raise SystemExit("the cache's questions or columns are not the look's")
    t = time.time()
    for k in PATHS:                       # numba's compile: the carve's next question, once, index time
        run_path(su, k, nq, {})
    su.index["jit_question_s"] = time.time() - t
    recs, checks, b16 = [], [], []
    group, group_sh = [], []
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            order = PATHS[i % len(PATHS):] + PATHS[:i % len(PATHS)]
            T, out = {"order": list(order)}, {}
            for k in order:
                T[k] = {}
                out[k] = run_path(su, k, i, T[k])
            T["n"] = int(out["gnn6"][1]["pool"].size)
            recs.append(T)
            checks.append(check_c2(su, i, out, cache))
            group.append(out["zsp2"][2])
            group_sh.append(out["gnn6"][1])
            if len(group) == C1.B16 or i == nq - 1:
                b16.append(batch_group(su, group, group_sh))
                group, group_sh = [], []
            del out
            if (i + 1) % 25 == 0:
                gc.collect()
                C1.log(f"  {i + 1}/{nq}")
    finally:
        gc.enable()
    rec = {"dataset": a.dataset, "carve": C1.CARVE, "queries": nq, "declared_in": "docs/C2_MLP_COLD_FAST.md",
           "threads": THREADS, "pin": pinned, "index": su.index, "paths": list(PATHS), "per_question": recs,
           "batch16": b16, "checks": checks, "check_summary": summarize(checks),
           "fits": {k: {"dir": str(C1.FITS[k].relative_to(ROOT)).replace("\\", "/"),
                        "models_sha256": C1.sha_file(C1.FITS[k] / "models.pt"), "candidate": C1.CAND} for k in C1.FITS},
           "basis": su.basis, "basis_sha256": su.basis_sha256, "look_records": su.look_head["records"],
           "peak_rss_bytes": LC.peak_rss(), "script_sha256": C1.sha_file(__file__),
           "c1_script_sha256": C1.sha_file(C1.__file__), "numpy": np.__version__, "torch": torch.__version__,
           "seconds": round(time.time() - t_start, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out or OUT / f"{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out, rec)
    C1.log(f"{a.dataset}: done in {rec['seconds']}s; checks {rec['check_summary']}; -> {out}")
    return 0


GOOD_KEYS = ("pool", "seeds", "edges", "rows_bits", "zrc2_eq_zrc1", "zsp2_eq_zsp1", "gnn6_top5_same", "zrc_top5_same",
             "zsp_top5_same")


def summarize(checks):
    s = {"questions": len(checks)}
    for key in GOOD_KEYS:
        v = [c[key] for c in checks if key in c]
        if v:
            s[f"{key}_ok"] = int(sum(v))
    for key in ("gnn6_vs_look", "zrc_vs_cache", "zsp_vs_cache"):
        v = [c[key] for c in checks if key in c]
        if v:
            s[f"{key}_max"] = max(v)
    return s


def totals(T):
    """Per path: (total with the read, total without it), in seconds."""
    return {k: (sum(T[k].values()), sum(T[k].values()) - T[k]["read"]) for k in PATHS}


def ratio_ci(num, den, rng, n=2000):
    r = float(np.percentile(num, 50) / np.percentile(den, 50))
    v = []
    for _ in range(n):
        s = rng.integers(0, num.size, num.size)
        v.append(np.percentile(num[s], 50) / np.percentile(den[s], 50))
    return {"ratio": r, "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}


def report():
    out = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name.startswith("report"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        ds = r["dataset"]
        tot = [totals(T) for T in r["per_question"]]
        good = [j for j, c in enumerate(r["checks"]) if all(c.get(k, True) for k in GOOD_KEYS)]
        bad_share = 1 - len(good) / max(len(tot), 1)
        d = {"questions": len(tot), "kept": len(good), "failing_share": bad_share, "stopped": bad_share > 0.02}
        rng = np.random.default_rng(0)
        for w, wi in (("with_read", 0), ("without_read", 1)):
            for k in PATHS:
                x = np.asarray([tot[j][k][wi] for j in good]) * 1e3
                d[f"{k}_{w}_ms"] = {p: float(np.percentile(x, p)) for p in (50, 95, 99)}
                d[f"{k}_{w}_ms"]["p50_ci"] = C1.boot_ci(x, lambda v: np.percentile(v, 50), rng)
            col = lambda k: np.asarray([tot[j][k][wi] for j in good])  # noqa: E731
            d[f"ratio_gnn6_over_zrc2_{w}_p50"] = ratio_ci(col("gnn6"), col("zrc2"), rng)
            d[f"ratio_zsp2_over_zrc2_{w}_p50"] = ratio_ci(col("zsp2"), col("zrc2"), rng)
            d[f"ratio_gnn6_over_zsp2_{w}_p50"] = ratio_ci(col("gnn6"), col("zsp2"), rng)
            d[f"ratio_zrc1_over_zrc2_{w}_p50"] = ratio_ci(col("zrc1"), col("zrc2"), rng)
        st = {}
        for k in PATHS:
            parts = {}
            for j in good:
                for s_, v in r["per_question"][j][k].items():
                    parts.setdefault(s_, []).append(v * 1e3)
            st[k] = {s_: float(np.percentile(v, 50)) for s_, v in parts.items()}
        d["stage_p50_ms"] = st
        b16 = {}
        for g in r["batch16"]:
            for k in ("zrc2", "zsp2", "gnn6"):
                b16.setdefault(k, []).append(sum(g[k].values()) / g["size"])
        d["batch16_forward_per_q_ms_p50"] = {k: float(np.percentile(v, 50)) * 1e3 for k, v in b16.items()}
        d["index"] = r["index"]
        d["checks"] = r["check_summary"]
        d["pin"] = r["pin"]
        d["peak_rss_gb"] = r["peak_rss_bytes"] / 1e9 if r.get("peak_rss_bytes") else None
        out[ds] = d
    LC.write_json(OUT / "report.json", out)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
    lines = ["| dataset | zrc2 (MLP) | zsp2 (GNN) | six GNN | zrc1 (C1 form) | six GNN / zrc2 | zsp2 / zrc2 | "
             "zrc1 / zrc2 | six GNN / zrc2 (no read) |", "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for ds, d in out.items():
        lines.append(f"| {ds} | {d['zrc2_with_read_ms'][50]:.1f} | {d['zsp2_with_read_ms'][50]:.1f} | "
                     f"{d['gnn6_with_read_ms'][50]:.1f} | {d['zrc1_with_read_ms'][50]:.1f} | "
                     f"{f(d['ratio_gnn6_over_zrc2_with_read_p50'])} | {f(d['ratio_zsp2_over_zrc2_with_read_p50'])} | "
                     f"{f(d['ratio_zrc1_over_zrc2_with_read_p50'])} | {f(d['ratio_gnn6_over_zrc2_without_read_p50'])} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def selftest():
    rng = np.random.default_rng(2)
    for _ in range(20):
        n = int(rng.integers(5, 60))
        m = int(rng.integers(0, 200))
        eu = rng.integers(0, n, m).astype(np.int16)
        ev = rng.integers(0, n, m).astype(np.int16)
        ef = rng.integers(0, 3, m).astype(np.int8)
        efw = rng.integers(0, 2, m).astype(np.int8)
        ebw = rng.integers(0, 2, m).astype(np.int8)
        k = int(rng.integers(0, min(n, ML8.MAX_SEEDS) + 1))
        sl = np.full(ML8.MAX_SEEDS, -1, np.int64)
        sb = np.full(ML8.MAX_SEEDS, -1, np.int64)
        sl[:k] = rng.choice(n, k, replace=False)
        sb[:k] = rng.integers(0, 2, k)
        proj = rng.standard_normal((n, LM.projection().shape[1])).astype(np.float16)
        q = rng.standard_normal(LM.projection().shape[0]).astype(np.float32)
        scratch = {b: 0.0 for b in ("SEM", "SEED", "WALK", "NBR")}
        L, _q, _p = LM.lean_query(n, proj, q, LM.projection(), sl, sb, eu, ev, ef, efw, ebw, scratch)
        assert same(walk_block(n, sl, sb, eu, ev, ef, efw, ebw), L["WALK"])
        wf = LC.walkf_of(n, proj, q, LM.projection(), sl, sb, eu, ev, ef, efw, ebw)
        assert same(walkf_block(n, sl, sb, eu, ev, ef, efw, ebw), wf)
    C1.log("c2_fast selftest: WALK and WALKF equal lean_query's and walkf_of's bit for bit on 20 random graphs: ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "run", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--out")
    ap.add_argument("--no-pin", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        return report()
    if not a.dataset:
        ap.error("--dataset")
    if a.cmd == "run":
        return run(a)
    return check(a.dataset, a.queries)


if __name__ == "__main__":
    sys.exit(main())
