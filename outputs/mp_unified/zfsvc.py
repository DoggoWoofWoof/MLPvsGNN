"""FSV's cold per-question path of both models on pick O's pools, every stage timed, and its development check
(docs/FSV_FAST_VERSIONS.md, amendment A2). The reference is zfsvt.py's read of each fit (outputs/fsv/timing/ref).

  stage (a)  the shared pool, from the first-stage lists: today's pool I (c10_zn.FastPool), base and seeds B, the text links,
             and PKO's rule in its lean form (zpko.rule_batch's calls on one question, without its statistics and
             self-checks): the walks, the orders, the expansion and the KT columns; pool = B plus the expansion
  stage (b)  each model's own inputs from the pool: the compile (c11_fast.compile_serial on c12_fast's table), the lean rows
             (c10_zn.lean_rows10), the links (c3_fast.links_fast), zrckpn's NEXT columns (LTM's fold scorer on the
             question's top five rows), the batch's features
  stage (c)  the forward (eager torch: no fused form exists for zrckpn or zg6r) and the top five

  dev    N questions: every stage, every check against PKO's rule, the reference carve's batch([q]) and the reference read
  run    the 200 timed questions and the JIT question (the carve's 201st): the timed cold pass, the warm forwards at b1
         and b16, then the untimed check pass; writes outputs/fsv/timing/raw/<dataset>.json

    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvc.py dev --dataset musique --questions 40 --blas 2
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvc.py run --dataset musique --blas 2 --lps 2,4 --require-pin
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvc.py threads --blas 2     (the pools after the imports)

OpenBLAS runs on --blas threads (A2: 2, where the filed inputs reproduce); torch, numba and every OpenMP runtime on 1.
"""
import os
import sys


def _opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


BLAS = _opt("--blas", "1")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = BLAS
os.environ.setdefault("NUMBA_NUM_THREADS", "1")      # rx sets it to --cpus; pylib_run has imported numba already
os.environ["LEAN_TIME_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zfsvt as T  # noqa: E402  (the reference read: constants, fits, the capture of a fit's model and carve)

OUT = T.OUT
RAW = OUT / "raw"
clock = time.perf_counter
EMPTY = np.empty(0, np.int64)
KEYS = ("mlp", "gnn")
B16 = 16
TOP = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zfsvc: {msg}", flush=True)


def _load():
    """The heavy imports, after the BLAS variables above are set and numpy is loaded (the C-stage modules set their own)."""
    if "C12" in globals():
        return
    nb_env = os.environ.get("NUMBA_NUM_THREADS")          # rx sets it to --cpus and numba was imported with it; the
    import torch                                          # C-stage modules force it to 1 on import: put it back
    import numba
    import c12_fast as C12
    import c11_fast as C11
    import c10_zn as C10
    import c3_fast as C3
    import c4_fast as C4
    import c1_cold as C1
    import lean_gpu as LG
    import lean_cache as LC
    import lean_time2 as T2
    import zpko as Z
    import zpkoh as ZH
    import zu2n as U2N
    import ltmcols as LTMC
    import ltwdiag as LTW
    import p1e_anchor_kind as Kx
    if nb_env is not None:
        os.environ["NUMBA_NUM_THREADS"] = nb_env
    torch.set_num_threads(1)
    numba.set_num_threads(1)
    globals().update(torch=torch, numba=numba, C12=C12, C11=C11, C10=C10, C3=C3, C4=C4, C1=C1, LG=LG, LC=LC, T2=T2, Z=Z,
                     ZH=ZH, U2N=U2N, LTMC=LTMC, LTW=LTW, Kx=Kx, E=Kx.E, FF=T2.FF, L3=T2.L3, S6=T2.S6, V2=T2.V2,
                     U2C=Z.ZP.U2C)


# ── the reference's model and carve ─────────────────────────────────────────


_TPL = []


def set_threads(blas):
    """One thread in the pools the stages use (torch, numba) and BLAS at the declared count, after the last module that
    resets them (bind_device_ops sets torch's to the config's); returns the effective counts."""
    torch.set_num_threads(1)
    numba.set_num_threads(1)
    out = {"blas_declared": int(blas), "torch": torch.get_num_threads(), "numba": numba.get_num_threads()}
    try:
        import threadpoolctl
        _TPL[:] = [threadpoolctl.threadpool_limits(limits=int(blas), user_api="blas"),
                   threadpoolctl.threadpool_limits(limits=1, user_api="openmp")]
        out["pools"] = [[x.get("internal_api"), x.get("num_threads"), os.path.basename(x.get("filepath") or "")]
                        for x in threadpoolctl.threadpool_info()]
    except ImportError:
        out["pools"] = "threadpoolctl is not installed"
    return out


def grab(key, ds, nq):
    """The fit's p@swa model and carve as its own read chain builds them (zfsvt.read_ref up to its batches: the same
    chain, flags and class), without its reads; the reference arrays are zfsvt's filed ones."""
    fit = T.FITS[(key, ds)]
    got, box = [], {}

    def fake(LGm, a, ds_, nq_):
        flags = LGm.set_flags("cpu")
        LGm.bind_device_ops()
        if a.host:
            import lean_host as LH
            box["placement"] = LH.substitute()
        fdir = Path(a.out_root or LGm.OUT) / a.name
        blob = torch.load(fdir / "models.pt", weights_only=False)
        blob["candidates"] = [T.CAND]
        models = LGm.load_models(blob, "cpu")
        _nm, m, bl = models[0]
        cache_root = Path(a.cache_root) if a.cache_root else LGm.LC.OUT
        c = LGm.CacheCarve(ds_, T.CARVE, blob["basis"], "cpu", cache_root, True, score2=True)
        blocks = [b for b in LGm.SETS["pick"] if b in bl]
        if blocks != list(bl):
            raise SystemExit(f"zfsvc: the pick orders {list(bl)} as {blocks}")
        box.update(model=m, blocks=blocks, carve=c, cache_root=cache_root, basis=blob["basis"], flags=flags)
        return {"ids": [str(x) for x in list(c.ids)[:nq_]]}, {}

    orig = T.read_ref
    T.read_ref = fake
    try:
        _rc, info = (T.ref_mlp if key == "mlp" else T.ref_gnn)(fit, ds, nq, got)
    finally:
        T.read_ref = orig
    if "model" not in box:
        raise SystemExit(f"zfsvc: {fit}: the read chain never reached read_ref")
    box["model"].eval()
    box.update(fit=fit, info=info)
    z = np.load(T.ref_dir(fit) / f"{ds}.npz")
    box["ref"] = {k: z[k] for k in z.files}
    return box


# ── stage (a): the shared pool ──────────────────────────────────────────────


def union_pools_n(name, carve, pop, prep, bases):
    """step 4e's union_pools (its manifest, base-and-seeds and frozen-pool checks, line for line) on the first
    len(pop.ids) questions of step 4d's filed carve: the filed ids' prefix must be the population's ids."""
    path = E.D4.POOLS / f"{name}__{carve}.npz"
    ent = json.loads(E.D4.MANIFEST.read_text(encoding="utf-8"))["files"][f"{name}__{carve}"]
    if E.sha_file(path) != ent["sha256"]:
        raise SystemExit(f"{path}: not step 4d's manifest's file")
    with np.load(path) as z:
        f = {k: z[k] for k in z.files}
    n = len(pop.ids)
    if f["ids"].tolist()[:n] != list(pop.ids):
        raise SystemExit(f"{name}/{carve}: step 4d's first {n} ids are not the population's")
    U, BQ, bs_all, inc_all = [], [], [], []
    for j in range(n):
        s = np.asarray(prep.seeds[j], np.int64)
        bs = np.union1d(np.asarray(bases[j], np.int64), s)
        if not np.array_equal(bs, f["bs_nodes"][f["bs_ptr"][j]:f["bs_ptr"][j + 1]].astype(np.int64)):
            raise SystemExit(f"{pop.ids[j]}: base and seeds are not step 4d's")
        inc = np.asarray(prep.pools[j], np.int64)
        sha = f["inc_sha"][j]
        if E.C4.pool_sha(inc) != (sha.decode() if isinstance(sha, bytes) else str(sha)):
            raise SystemExit(f"{pop.ids[j]}: the frozen pool is not step 4d's")
        u = np.union1d(bs, f["nodes"][f["ptr"][j]:f["ptr"][j + 1]].astype(np.int64))
        U.append(u)
        bs_all.append(bs)
        inc_all.append(inc)
        BQ.append(inc.size if name == "squad" else int(f["B_q"][j]))
    return U, np.asarray(BQ, dtype=np.int64), bs_all, inc_all, f


def open_a(ds_name, nq, full_rule=False):
    """PKO's opening on a population cut to the first nq + 1 questions of the carve (the nq timed and the JIT question),
    and the pool's own objects on W2_S: the stores, FastPool, the name index and the question texts."""
    t0 = time.time()
    placement = ZH.apply_mirror()                                          # CHECK: signature and return
    n = nq + 1
    orig = E.C4.populations

    def populations(op, S4, ds, name):
        out = []
        for c, pop in orig(op, S4, ds, name):
            if c == T.CARVE:
                row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
                ids = [str(q) for q in list(pop.ids)[:n]]
                idx = np.asarray([row_of[q] for q in ids], np.int64)
                pop = op.m3b_compile.Population(name, c, ids, idx, [np.zeros(0, np.int64) for _ in ids], len(ids), 0,
                                                S4.m3b_pools.ids_digest(list(ids)))   # CHECK: fields, as P4a's Bench
            out.append((c, pop))
        return out

    E.C4.populations = populations
    u0 = E.union_pools
    m3b = Kx.S4.m3b_pools
    lb0 = m3b.load_or_build_store                                           # open_carve leaves its W2_S loader in place
    E.union_pools = union_pools_n
    try:
        cx, ctx = Z.open_pko(ds_name, T.CARVE, 1)
    finally:
        E.C4.populations = orig
        E.union_pools = u0
        m3b.load_or_build_store = lb0
    log(f"{ds_name}: PKO opened on {len(cx.bs)} questions ({time.time() - t0:.0f} s)")
    op = cx.op
    t1 = time.time()
    dsh, construction, families, stores0, _union0 = E.open_dataset(op, ds_name)
    n_nodes = int(dsh.n_nodes)
    stores, swap = dict(stores0), None
    g = U2C.graph_csr(ds_name, "chosen", n_nodes)
    if g is not None and "structural" in families:
        ip, col, desc = g
        if json.loads(json.dumps(desc, default=str)) != json.loads(json.dumps(cx.desc, default=str)):
            raise SystemExit(f"{ds_name}: the graph is not PKO's")
        swap = U2C.csr_store(n_nodes, ip, col)
        stores["structural"] = swap
    import lean_time as LT
    builder = LT.PoolBuilder(construction, op.cfg_h, stores, op.m3a, op.m3b_contract, op.m3b_compile)
    fpool = C10.FastPool(builder, n_nodes)
    index = texts = None
    if ds_name != "squad":
        index = E.build_index(E.node_names(dsh), E.stopwords())
        qt = E.query_texts(dsh)
        row_of = {qid: j for j, qid in enumerate(dsh.query_ids)}
        texts = [qt[row_of[str(q)]] for q in cx.pop.ids]
        del qt
    ids = [str(q) for q in cx.pop.ids]
    p = cx.prep
    d = [np.ascontiguousarray(p.dense_ids[i], np.int64) for i in range(len(ids))]
    s = [np.ascontiguousarray(p.splade_ids[i], np.int64) for i in range(len(ids))]
    log(f"{ds_name}: pool objects ({time.time() - t1:.0f} s)")
    return SimpleNamespace(name=ds_name, cx=cx, ctx=ctx, ids=ids, d=d, s=s, fpool=fpool, index=index, texts=texts,
                           swap=swap, n_nodes=n_nodes, nq=nq, full_rule=full_rule, placement=placement, ds=dsh)


def base_of(S, d, s):
    """FastPool's base rule: equal_rrf_budget_* (c10_zn._rrf_base) or dense_top*."""
    fp = S.fpool
    if fp.kind == 0:
        b, err = C10._rrf_base(d, s, fp.budget, fp.constant)
        if err:
            raise ValueError(f"the base: {fp.WHY[err]}")
        return b
    return d[:min(fp.budget, d.size)]


def lean_rule(S, j, linked, I, Bq, T_):
    """zpko.rule_batch's calls on question j (step 4d's B and today's pool computed by the caller), without mass_counts,
    orders_exact, the kind check and the statistics dict: (pool, KT columns, expansion)."""
    cx, ctx = S.cx, S.ctx
    Kc, Dc, prep, constant, threads = cx.K, cx.D, cx.prep, cx.constant, cx.threads
    c0 = clock()
    js, lk = [j], {j: linked}
    srcs = Dc.sources_batch(prep, js, lk, cx.unions, constant, threads)
    steered = Kc.steered_batch(prep, js, lk, cx.unions, srcs, constant, threads)
    k0, a1 = Kc.restarts_batch(prep, js, lk, cx.unions, srcs, constant)
    stz = [Kc.steer(prep, j, constant)]
    q = Z.D5B.unit(np.asarray(prep.qemb[j], np.float64))
    wa = [Z.A.omega(q, ctx.mu, Z.Q.A_BETA, True)[0]]
    wqs = [Z.P.pair_weights(x, ctx.ws) for x in wa] if ctx.passage else wa
    wal = {w: Z.A.rwalk(cx.unions[Z.Q.WALK_UNION[w]], a1 if w.startswith("A1") else k0, stz if w.startswith("KT") else None,
                        wqs, 1, ctx.ty, ctx.alpha, ctx.eps, ctx.bmax, threads) for w in Z.Q.WALKS}
    c1 = clock()
    src, _nn, (L, Ls_) = srcs[0]
    route = Dc.routes(L, Ls_)
    knd = Kc.kind(Ls_, prep.seeds[j], cx.nonly)
    sc = (Z.A.scaled(wal["A1st"][0]), Z.A.scaled(wal["A1all"][0]), Z.A.scaled(wal["Kst"][0]), src[3], src[4])
    stc = dict(steered[0])
    stc["KTst"], stc["KTall"] = wal["KTst"][0][0], wal["KTall"][0][0]
    o1, _p1 = Kc.p1e_orders(sc, stc, route, knd, cx.n_st, constant)
    kt = np.asarray(o1["KIND_T"], np.int64)
    n1 = Z.D5B.n1_table(cx.adj, Bq, Z.D5.weights_of(stz[0], Bq))[0]
    rows = Z.M.Rows(cx.nodes, verify=False)
    rows.add(n1)
    cos1 = (Z.D5B.unit(rows.read(n1)) @ q) if n1.size else np.zeros(0, np.float32)
    cr = Z.D5B.ranks_desc(cos1, n1) if n1.size else Z.S.EMPTY
    n2, first, _taken, _hubs = Z.U2E.second_link(cx.adj, n1, cr, Bq)
    r0src = np.asarray(src[4][0], np.int64)
    rows.add(np.concatenate([kt, r0src, I, np.asarray(n2, np.int64)]))
    full = Z.F.outside(kt, Bq)
    fill = int(Z.M.fill_of(full, I.size, Bq.size))
    r0 = Z.F.outside(r0src, Bq)
    iout = Z.F.outside(I, Bq)
    ords, _sizes, _U, _mem = Z.U2E.question_orders(full, n1, cr, cos1, r0, iout, n2, first, rows.read, q, constant, Dc.rrf)
    kt_cos = np.asarray(ords[Z.M.NAMES[ctx.plan.o]], np.int64)
    _kq, _k, keep = (int(x[0]) for x in Z.kept_counts([kt_cos.size], [fill], [Bq.size], ctx.plan))
    exp = kt_cos[:keep]
    pool = np.union1d(Bq, exp)
    c2 = clock()
    oe, pe = Kc.p1e_orders(src, steered[0], bool(route), knd, cx.n_st, constant)
    kt16 = Z.KT3.columns(Z.ZP.kt_orders(oe, pe), pool, exp, knd == "entity").astype(Z.ZP.KT_DT)
    c3 = clock()
    T_["a_walk"], T_["a_order"], T_["a_kt"] = c1 - c0, c2 - c1, c3 - c2
    return pool, kt16, exp


def stage_a(S, j, T_):
    """Question j's shared pool, cold: today's pool, base and seeds, the text links, the rule. Times into T_."""
    d, s = S.d[j], S.s[j]
    c0 = clock()
    I, seeds, bucket = S.fpool(d, s)
    c1 = clock()
    Bq = np.union1d(base_of(S, d, s), seeds)
    c2 = clock()
    linked = E.link(S.texts[j], S.index) if S.index is not None else EMPTY
    c3 = clock()
    I = np.asarray(I, np.int64)
    if S.full_rule:                                                         # the fallback: PKO's own rule, as filed
        f = Z.rule_batch(S.cx, S.ctx, [j])[0]
        pool, kt16, exp = np.union1d(f["B"], f["exp"]), f["kt"], f["exp"]
    else:
        pool, kt16, exp = lean_rule(S, j, np.asarray(linked, np.int64), I, Bq, T_)
    T_["a_pool"], T_["a_base"], T_["a_link"], T_["a_rule"] = c1 - c0, c2 - c1, c3 - c2, clock() - c3
    return SimpleNamespace(pool=pool, seeds=np.asarray(seeds, np.int64), bucket=np.asarray(bucket), kt=kt16, I=I, B=Bq,
                           exp=exp, linked=np.asarray(linked, np.int64))


# ── stage (b): the compile context and each model's inputs ──────────────────


def open_b(S):
    """The compile on W2_S (look_u2d's swap on S6.pair_open's context), the reused buffers sized to the largest pool, the
    int8 codes of the nodes the pools touch (index time), the decoder and both models' fold scorer."""
    name = S.name
    S6.pair_verify(S6.SIX)
    cfg, cfg_m3b, _cfg_h = V2.load_configs()
    op = S6.pair_open(S6.SIX, cfg, cfg_m3b, name)
    context = op.contexts[name]
    if context.rel_table is not None:
        raise SystemExit(f"{name}: typed relations; the passage graphs only")
    if S.swap is not None:
        context.stores["structural"] = S.swap
    fc = FF.compiler_for(context.stores, context.nodes, context.rel_table)
    if fc is None:
        raise SystemExit("no fast compiler for this context")
    cm = S.boxes["mlp"]["carve"]
    xcols, _spans, ci = LC.xc_columns(cm.look)
    bz, _brec = LC.load_basis(cm.basis, S.boxes["mlp"]["cache_root"])
    su = SimpleNamespace(fc=fc, columns=op.inputs["column_indices"], xcols=xcols, c_rrf_x=ci["rrf"], W=cm.W,
                         span=dict(cm.span), prep=S.cx.prep, context=context, store=L3.Store(bz), codes=None)
    if dict(S.boxes["gnn"]["carve"].span) != su.span or S.boxes["gnn"]["basis"] != S.boxes["mlp"]["basis"]:
        raise SystemExit("the two carves hold different spans or bases")
    S.su, S.c_rrf, S.op_b = su, int(cm.c_rrf), op
    S.take = list(range(su.span["rank"][0], su.span["rank"][1])) + [su.span["dense_cos"][0]]
    return S


def fill_codes(S, j, pool):
    """Index time, untimed, before each question: the int8 codes of its pool's nodes as the cache the filed read was built on
    holds them. The cache's table is not one code per node (a node's code can differ by a step between questions, from the
    build's blocks), so a node-level table built once cannot reproduce every filed read; the timed stages read the same
    table slots either way. Returns the rows whose code differs from the one an earlier question left."""
    cm = S.boxes["mlp"]["carve"]
    lo, hi = int(cm.off_np[j]), int(cm.off_np[j + 1])
    if hi - lo != pool.size:
        raise SystemExit(f"question {j}: the pool has {pool.size} rows, the cache's {hi - lo}")
    c = cm.tab[cm.row[lo:hi]].numpy()
    su = S.su
    bad = int(((su.codes[pool] != c).any(1) & S.code_seen[pool]).sum())
    su.codes[pool] = c
    S.code_seen[pool] = True
    return bad


def prepass(S, qs):
    """Index time, untimed: stage (a) on every question once (numba's compile, the pools' pages, the pools' nodes), the
    buffers sized to the largest pool, the codes of every node the pools touch, the compile's kernel table."""
    su = S.su
    pools = [stage_a(S, j, {}).pool for j in qs]
    nmax = max(int(p.size) for p in pools)
    t = time.time()
    su.codes = np.zeros((int(su.fc.n_nodes), L3.STORE_K), np.int8)
    S.code_seen = np.zeros(int(su.fc.n_nodes), bool)
    nodes = np.unique(np.concatenate(pools))
    S.code_conflicts = sum(fill_codes(S, j, p) for j, p in zip(qs, pools))
    su.fc._buffers(nmax)
    for b in (su.fc._E, su.fc._P, su.fc._P1, su.fc._P2):
        b.fill(0.0)
    ks = C4.Kernels(su.fc)
    k12 = C12.C12K(ks.gallop[1], nmax, int(su.fc._E.shape[1]))
    S.bn = SimpleNamespace(su=su, k_ser=(k12, k12))
    S.dec = C10.Decoder(su)
    S.dec.reserve(nmax)
    S.nmax = nmax
    log(f"index time: {nodes.size} nodes' codes ({S.code_conflicts} rows whose cached code differs from an earlier "
        f"question's), pools up to {nmax} rows ({time.time() - t:.0f} s)")
    return pools


def next_cold(S, j, r, A, link, T_):
    """zrckpn's NEXT columns of question j: LTM's lists of the question's five leading rows and their link neighbours,
    scored by the fold model of the question's hfold, as ltmcols.cols writes them (log-ratio to uniform, clipped)."""
    n, X = r["n"], r["X"]
    _ne, la, lb, lf = link
    u, v, f = (np.asarray(x, np.int64) for x in (la, lb, lf))
    x6 = X[:, S.take].astype(np.float32)
    o = LTW.CH.order_of(x6[:, LTW.J_RRF])                                   # CHECK: question_cases' own order
    rk = np.empty(n, np.int64)
    rk[o] = np.arange(1, n + 1)
    top = o[:LTW.TOP]
    in_top = np.zeros(n, bool)
    in_top[top] = True
    lists = []
    for k, a in enumerate(top):
        a = int(a)
        bits = LTW.fam_bits(n, a, u, v, f)
        cand = np.flatnonzero((bits > 0) & ~in_top)
        if cand.size:
            lists.append((k, a, cand, bits[cand]))
    X5 = np.full((n, LTMC.TOP), LTMC.FILL, np.float32)
    cnt = np.zeros(n, np.int64)
    if lists:
        c_ = np.concatenate([x[2] for x in lists])
        z = {"qe": r["q_emb16"][None, :].astype(np.float16), "T_q": np.zeros(len(lists), np.int64),
             "T_anode": np.asarray([A.pool[x[1]] for x in lists], np.int64), "T_cnode": A.pool[c_],
             "T_c": c_, "T_ct": np.zeros(c_.size, bool), "T_cfam": np.concatenate([x[3] for x in lists]).astype(np.int8),
             "T_crank": rk[c_], "T_cxc": x6[c_], "T_axc": x6[[x[1] for x in lists]],
             "T_off": np.r_[0, np.cumsum([x[2].size for x in lists])].astype(np.int64)}
        sel = np.arange(len(lists))
        idx = LTW.lists_of(z, "T", sel)
        b, _chk = LTW.read_chunk(z, "T", S.cx.nodes, sel, idx)
        its = LTW.items_of(b, z, "T", LTMC.DS_K[S.name])
        m_, mu, sd = S.folds[int(S.hfold[j])]
        for it, sc in zip(its, LTW.score_items(m_, its, mu, sd)):
            rows = it["loc"]
            X5[rows, lists[it["list"]][0]] = LTMC.log_ratio(sc)
            cnt[rows] += 1
    C = np.empty((n, LTMC.WIDTH), np.float16)
    C[:, :LTMC.TOP] = X5
    C[:, LTMC.TOP] = X5.max(1)
    C[:, LTMC.TOP + 1] = cnt
    return C


def build(S, key, packs):
    """The batch's features as lean_gpu.CacheCarve.batch and the arm carves' side keys give them, from per-question packs."""
    Bn = len(packs)
    ns = [p["n"] for p in packs]
    X = torch.from_numpy(np.concatenate([p["X"] for p in packs])).to(torch.float32)
    Xc = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    nq = torch.from_numpy(np.repeat(np.arange(Bn), ns))
    feats = {}
    for b in S.blocks[key]:
        if b == "SEMB":
            feats[b] = (torch.from_numpy(np.stack([p["q16"] for p in packs])).to(torch.float32),
                        torch.from_numpy(np.concatenate([p["P"] for p in packs])))
        elif b == "NEXT":
            feats[b] = torch.nan_to_num(torch.from_numpy(np.concatenate([p["nxt"] for p in packs])).to(torch.float32),
                                        nan=0.0, posinf=0.0, neginf=0.0)
        else:
            a, e = S.su.span[b]
            feats[b] = Xc[:, a:e].contiguous()
    feats["_kt"] = torch.from_numpy(np.concatenate([p["kt"] for p in packs])).to(torch.float32)
    if key == "gnn":
        rseg = np.cumsum(ns) - ns
        ne = [int(p["ne"]) for p in packs]
        eq = np.repeat(np.arange(Bn), ne)
        u = np.concatenate([p["lu"] for p in packs]).astype(np.int64) + rseg[eq]
        v = np.concatenate([p["lv"] for p in packs]).astype(np.int64) + rseg[eq]
        f = np.concatenate([p["lf"] for p in packs]).astype(np.int64)
        feats["_link"] = {"u": torch.from_numpy(u), "v": torch.from_numpy(v), "f": torch.from_numpy(f)}
        feats["_g6r"] = {"ci": torch.full((u.size,), -1, dtype=torch.long),
                         "crel": torch.zeros(0, U2N.SLOTS, dtype=torch.long), "cdir": torch.zeros(0, dtype=torch.long),
                         "rel": None}
    rrf = X[:, S.c_rrf].contiguous()
    base_z = LG.seg_zscore8D(rrf.unsqueeze(1), nq, Bn).squeeze(1)
    return feats, nq, base_z


def forward(S, key, feats, nq, base_z, Bn):
    with torch.no_grad():
        return S.models[key](feats, torch.ones((Bn, len(S.blocks[key])), dtype=torch.float32), nq, Bn, base_z)


def run_model(S, key, j, A, T_):
    """One model on question j's pool (stage (a)'s A), cold: (b) its inputs, (c) its forward and top five. Times into T_.
    Returns (scores, pack, (feats, nq, base_z))."""
    c0 = clock()
    comp = C11.compile_serial(S.bn, C10.qinputs(S.su, j), A.pool, A.seeds, False)
    c1 = clock()
    sh = {"pool": A.pool, "seeds": A.seeds, "bucket": A.bucket, "comp": comp, "qemb": np.asarray(S.cx.prep.qemb[j])}
    r = C10.lean_rows10(S.su, sh, T_, S.dec)                                # T_: edges, lean
    c2 = clock()
    eu, ev, ef = r["e"][0], r["e"][1], r["e"][2]
    link = C3.links_fast(r["n"], eu, ev, ef)
    c3 = clock()
    pack = {"n": r["n"], "X": r["X"], "P": r["P"], "q16": r["q_emb16"], "kt": A.kt, "ne": link[0], "lu": link[1],
            "lv": link[2], "lf": link[3]}
    if key == "mlp":
        pack["nxt"] = next_cold(S, j, r, A, link, T_)
    c4 = clock()
    feats, nq, base_z = build(S, key, [pack])
    c5 = clock()
    s = forward(S, key, feats, nq, base_z, 1)
    torch.topk(s, min(TOP, r["n"]))
    c6 = clock()
    T_.update(compile=c1 - c0, links=c3 - c2, next=c4 - c3, assemble=c5 - c4, forward=c6 - c5, b_total=c5 - c0,
              total=c6 - c0)
    pack["P"] = np.array(pack["P"])                                         # the decoder's buffer is reused
    return s.numpy().copy(), pack, (feats, nq, base_z)


def setup(ds_name, nq, full_rule=False):
    _load()
    t0 = time.time()
    S = open_a(ds_name, nq, full_rule)
    S.boxes = {k: grab(k, ds_name, nq + 1) for k in KEYS}
    S.models = {k: S.boxes[k]["model"] for k in KEYS}
    S.blocks = {k: list(S.boxes[k]["blocks"]) for k in KEYS}
    if [str(x) for x in S.boxes["mlp"]["carve"].ids[:nq + 1]] != S.ids:
        raise SystemExit("the carve's first questions are not the population's")
    open_b(S)
    split = S.boxes["mlp"]["info"]["split"]
    S.folds, _fit_sha, _shas = LTMC.load_folds(LTMC.OUT, split)
    S.hfold = np.random.default_rng([LTMC.SEED, LTMC.DS_K[ds_name], 7]).integers(
        0, LTMC.K, int(S.boxes["mlp"]["carve"].rows))
    S.threads = set_threads(BLAS)
    log(f"{ds_name}: set up ({time.time() - t0:.0f} s); split {split}; threads {S.threads}")
    return S


# ── the checks (untimed) ────────────────────────────────────────────────────


def diff(a, b):
    """(identical, largest absolute difference, differing values) of two feature values (tensors, tuples, dicts, None)."""
    if a is None or b is None:
        return (a is None and b is None), 0.0, 0
    if isinstance(a, dict):
        out = [diff(a.get(k), b.get(k)) for k in sorted(set(a) | set(b))]
        return all(x[0] for x in out), max([x[1] for x in out] + [0.0]), sum(x[2] for x in out)
    if isinstance(a, (tuple, list)):
        out = [diff(x, y) for x, y in zip(a, b)]
        return len(a) == len(b) and all(x[0] for x in out), max([x[1] for x in out] + [0.0]), sum(x[2] for x in out)
    if a.shape != b.shape:
        return False, float("inf"), -1
    if a.numel() == 0:
        return True, 0.0, 0
    d = (a.to(torch.float64) - b.to(torch.float64)).abs()
    return bool((d == 0).all()), float(torch.nan_to_num(d, nan=0.0).max()), int((d != 0).sum())


def check_scores(S, key, j, s):
    """Scores against the reference read at batch 1 (zfsvt's scores_b1): the largest difference and the top five."""
    ref = S.boxes[key]["ref"]
    off = np.r_[0, np.cumsum(ref["n"])]
    want = ref["scores_b1"][off[j]:off[j + 1]]
    if want.size != s.size:
        return {"rows": int(s.size), "ref_rows": int(want.size), "max_diff": None, "top5_same": False, "ok": False}
    md = float(np.abs(s.astype(np.float64) - want.astype(np.float64)).max())
    same = bool(np.array_equal(T.top5(s), T.top5(want)))
    return {"rows": int(s.size), "ref_rows": int(want.size), "max_diff": md, "top5_same": same,
            "ok": bool(md <= T.TOL and same)}


def dev_checks(S, j, A, outs):
    """Everything development compares for question j; returns a flat dict and prints the failures."""
    res = {}
    f = Z.rule_batch(S.cx, S.ctx, [j])[0]
    res["a_B"] = bool(np.array_equal(A.B, f["B"]))
    res["a_I"] = bool(np.array_equal(A.I, f["I"])) and bool(np.array_equal(A.I, np.asarray(S.cx.inc[j], np.int64)))
    res["a_bs"] = bool(np.array_equal(A.B, np.unique(np.asarray(S.cx.bs[j], np.int64))))
    res["a_exp"] = bool(np.array_equal(A.exp, f["exp"]))
    res["a_pool"] = bool(np.array_equal(A.pool, np.union1d(f["B"], f["exp"])))
    res["a_kt"] = bool(np.array_equal(A.kt, f["kt"]))
    res["a_linked"] = bool(np.array_equal(A.linked, np.asarray(S.cx.linked[j], np.int64)))
    res["a_seeds"] = bool(np.array_equal(A.seeds, np.asarray(S.cx.prep.seeds[j], np.int64)))
    for key in KEYS:
        s, pack, (feats, nq, base_z) = outs[key]
        c = S.boxes[key]["carve"]
        rf, rnq, rbz, _g = c.batch(np.asarray([j]), S.blocks[key])
        ok, md, nd = diff(feats, rf)
        res[f"{key}_feats"] = {"identical": ok, "max_diff": md, "differing": nd,
                               "per_key": {k: diff(feats.get(k), rf.get(k))[:3] for k in rf if not diff(feats.get(k), rf.get(k))[0]}}
        res[f"{key}_base_z"] = dict(zip(("identical", "max_diff", "differing"), diff(base_z, rbz)))
        res[f"{key}_scores"] = check_scores(S, key, j, s)
        if key == "mlp":
            nx = c.next_X[int(c.off_np[j]):int(c.off_np[j + 1])].numpy()
            res["next_flips"] = int((nx.view(np.uint16) != pack["nxt"].view(np.uint16)).sum())
            res["next_max_diff"] = float(np.abs(nx.astype(np.float64) - pack["nxt"].astype(np.float64)).max()) if nx.size else 0.0
            res["next_values"] = int(nx.size)
    return res


def dev(a):
    S = setup(a.dataset, a.questions, a.full_rule)
    nq = a.questions
    prepass(S, range(nq + 1))
    S.threads = set_threads(BLAS)
    log(f"threads {S.threads}")
    for key in KEYS:                                                        # the JIT question, both models
        A = stage_a(S, nq, {})
        fill_codes(S, nq, A.pool)
        run_model(S, key, nq, A, {})
    allok = True
    for j in range(nq):
        Ta = {}
        A = stage_a(S, j, Ta)
        fill_codes(S, j, A.pool)
        outs, Tm = {}, {}
        for key in KEYS:
            Tm[key] = {}
            outs[key] = run_model(S, key, j, A, Tm[key])
        res = dev_checks(S, j, A, outs)
        bad = {k: v for k, v in res.items() if v is False or (isinstance(v, dict) and (v.get("ok") is False or v.get("identical") is False))}
        if set(bad) == {"mlp_feats"} and set(bad["mlp_feats"]["per_key"]) == {"NEXT"}:
            bad = {}                                                        # float16 NEXT values a step off are counted (next_flips), not failed
        allok &= not bad
        log(f"q{j} n={A.pool.size} a: " + " ".join(f"{k}={v * 1000:.1f}" for k, v in Ta.items()))
        for key in KEYS:
            log(f"q{j} {key}: " + " ".join(f"{k}={v * 1000:.1f}" for k, v in Tm[key].items()))
        log(f"q{j} checks: next_flips={res['next_flips']}/{res['next_values']} (max {res['next_max_diff']:.4g}); "
            f"mlp_scores={res['mlp_scores']}; gnn_scores={res['gnn_scores']}")
        if bad:
            log(f"q{j} FAILED: {json.dumps(bad, default=str)[:1500]}")
    log("DEV " + ("PASS" if allok else "FAIL"))
    return 0 if allok else 1


# ── the timed run ───────────────────────────────────────────────────────────


def pin(lps):
    """c1_cold.pin's calls for this process only, with a mask of several logical processors: affinity, EcoQoS opted out,
    above-normal priority."""
    if os.name != "nt":
        return {"pinned": False, "why": "not Windows"}
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    HANDLE, BOOL, DWORD = ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong
    k32.GetCurrentProcess.restype = HANDLE
    k32.GetCurrentProcess.argtypes = []
    k32.SetProcessAffinityMask.argtypes = [HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = BOOL
    k32.SetProcessInformation.argtypes = [HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD]
    k32.SetProcessInformation.restype = BOOL
    k32.SetPriorityClass.argtypes = [HANDLE, DWORD]
    k32.SetPriorityClass.restype = BOOL
    h = k32.GetCurrentProcess()
    out = {"affinity_lps": [int(x) for x in lps]}
    out["affinity_ok"] = bool(k32.SetProcessAffinityMask(h, sum(1 << int(x) for x in lps)))
    st = C1._PTS(1, 0x1, 0)
    out["ecoqos_off_ok"] = bool(k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))
    out["above_normal_ok"] = bool(k32.SetPriorityClass(h, 0x8000))
    out["pinned"] = out["affinity_ok"] and out["ecoqos_off_ok"] and out["above_normal_ok"]
    return out


def cpu_load():
    try:
        import psutil
        return psutil.cpu_percent(interval=1.0, percpu=True)
    except Exception:                                                       # noqa: BLE001
        return None


def pin_and_load(lps):
    """The host's per-processor load (1 s sample), then the pin; the load is also sampled after the timed pass."""
    info = {"cpu_percent_before": cpu_load()}
    try:
        info["pin"] = pin(lps)
    except Exception as e:                                                  # noqa: BLE001
        info["pin"] = {"pinned": False, "error": repr(e)[:200]}
    return info


def run(a):
    nq = a.nq or T.NQ
    out_dir = RAW if nq == T.NQ else OUT / "raw_smoke"
    out_file = out_dir / f"{a.dataset}__b{BLAS}.json"                       # one raw file per dataset and BLAS pass
    if nq == T.NQ and out_file.exists() and not a.overwrite:                # A2: a finished pass is not repeated
        raise SystemExit(f"{out_file.name} exists: a finished pass is not repeated for its numbers (--overwrite)")
    beside = int(BLAS) != 2                                                 # A2: the BLAS-1 pass beside the record
    S = setup(a.dataset, nq, False)
    prepass(S, range(nq + 1))
    S.threads = set_threads(BLAS)
    head = {"dataset": a.dataset, "questions": nq, "blas": int(BLAS), "overwrote": bool(a.overwrite and out_file.exists()),
            "pass": "beside: BLAS 1, every question timed, checks counted" if beside else "record: BLAS 2",
            "threads": S.threads, "fits": {k: S.boxes[k]["fit"] for k in KEYS},
            "code_conflicts": S.code_conflicts, "load_and_pin": pin_and_load([int(x) for x in a.lps.split(",")]),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script_sha256": {"zfsvc": T.sha_src(__file__), "zfsvt": T.sha_src(T.__file__)}}
    if a.require_pin and not head["load_and_pin"]["pin"].get("pinned"):
        raise SystemExit(f"{a.dataset}: not pinned: {head['load_and_pin']}")
    for key in KEYS:                                                        # the JIT question
        A = stage_a(S, nq, {})
        fill_codes(S, nq, A.pool)
        run_model(S, key, nq, A, {})
    gc.collect()
    rows, As, packs, scores = [], [], {k: [] for k in KEYS}, {k: [] for k in KEYS}
    for j in range(nq):
        Ta = {}
        A = stage_a(S, j, Ta)
        fill_codes(S, j, A.pool)
        row = {"q": j, "n": int(A.pool.size), "a": Ta}
        for key in (KEYS if j % 2 == 0 else KEYS[::-1]):                    # interleaved: each model leads in turn
            Tm = {}
            s, pack, _f = run_model(S, key, j, A, Tm)
            row[key] = Tm
            packs[key].append(pack)
            scores[key].append(s)
        rows.append(row)
        As.append(A)
    # warm: the forward only, on features built beforehand, at b1 and b16
    for key in KEYS:
        b1 = [build(S, key, [p]) for p in packs[key]]
        for j, (feats, nqt, base_z) in enumerate(b1):
            c0 = clock()
            forward(S, key, feats, nqt, base_z, 1)
            rows[j][key]["warm_b1"] = clock() - c0
        for g0 in range(0, nq - B16 + 1, B16):
            feats, nqt, base_z = build(S, key, packs[key][g0:g0 + B16])
            c0 = clock()
            forward(S, key, feats, nqt, base_z, B16)
            dt = clock() - c0
            for j in range(g0, g0 + B16):
                rows[j][key]["warm_b16"] = dt / B16
    head["load_and_pin"]["cpu_percent_after"] = cpu_load()
    # the checks (section 5), untimed: each model's scores against its fit's filed read, the pool against PKO's own rule,
    # and the NEXT columns against the cache's (float16 values that differ)
    checks = {k: [check_scores(S, k, j, scores[k][j]) for j in range(nq)] for k in KEYS}
    cm = S.boxes["mlp"]["carve"]
    pool_ok, flips = [], []
    for j in range(nq):
        f = Z.rule_batch(S.cx, S.ctx, [j])[0]
        pool_ok.append(bool(np.array_equal(As[j].pool, np.union1d(f["B"], f["exp"])) and np.array_equal(As[j].kt, f["kt"])
                            and np.array_equal(As[j].I, np.asarray(S.cx.inc[j], np.int64))))
        nx = cm.next_X[int(cm.off_np[j]):int(cm.off_np[j + 1])].numpy()
        flips.append(int((nx.view(np.uint16) != packs["mlp"][j]["nxt"].view(np.uint16)).sum()))
    bad = [not (pool_ok[j] and all(checks[k][j]["ok"] for k in KEYS)) for j in range(nq)]
    stop = sum(bad) / nq > 0.02 and not beside                              # the beside pass counts, it does not stop
    head["checks"] = {"questions": nq, "failing": int(sum(bad)), "share": sum(bad) / nq, "stop": stop,
                      "pool_failing": int(sum(not x for x in pool_ok)), "next_flip_values": int(sum(flips)),
                      "next_flip_questions": int(sum(x > 0 for x in flips)),
                      **{k: {"failing": int(sum(not c["ok"] for c in v)),
                             "max_diff": max((c["max_diff"] or 0.0) for c in v)} for k, v in checks.items()}}
    T.write_json(out_file, {"head": head, "rows": rows, "checks": checks, "pool_ok": pool_ok, "next_flips": flips,
                            "kept": [True] * nq if beside else [not b for b in bad]})
    log(f"{a.dataset}: raw written to {out_dir.name}/{out_file.name}; checks {head['checks']}")
    return 1 if stop else 0


# ── the report: A2's "Reported", reduced from the raw files (no choice is made here) ─────────────────────────────────

NBOOT = 2000
PASSES = (("b2", "BLAS 2 (the record)"), ("b1", "BLAS 1 (beside the record: every question timed, checks counted)"))
A_KEYS = ("a_pool", "a_base", "a_link", "a_rule", "a_walk", "a_order", "a_kt")


def _series(rows, kept):
    """Per-question times in seconds over the kept questions, in question order: (a), and per model (b)'s parts, (b), (c),
    model-only (b) + (c), end to end (a) + (b) + (c), and the warm forwards (NaN where a question has none)."""
    rows = [r for r, k in zip(rows, kept) if k]
    s = {"q": np.array([r["q"] for r in rows]), "n": np.array([r["n"] for r in rows])}
    a = {k: np.array([r["a"][k] for r in rows]) for k in A_KEYS}
    s["a"] = {"total": a["a_pool"] + a["a_base"] + a["a_link"] + a["a_rule"], "pool": a["a_pool"], "base": a["a_base"],
              "links": a["a_link"], "rule": a["a_rule"], "walks": a["a_walk"], "orders": a["a_order"], "kt": a["a_kt"],
              "rule_other": a["a_rule"] - a["a_walk"] - a["a_order"] - a["a_kt"]}
    for key in KEYS:
        m = {k: np.array([r[key][k] for r in rows]) for k in
             ("compile", "edges", "lean", "links", "next", "assemble", "b_total", "forward", "total")}
        s[key] = {"compile": m["compile"], "rows": m["edges"] + m["lean"], "links": m["links"], "next": m["next"],
                  "inputs": m["assemble"], "b": m["b_total"], "c": m["forward"], "model_only": m["total"],
                  "end_to_end": s["a"]["total"] + m["total"],
                  "warm_b1": np.array([r[key].get("warm_b1", np.nan) for r in rows]),
                  "warm_b16": np.array([r[key].get("warm_b16", np.nan) for r in rows])}
    return s


def _pq(x):
    x = x[~np.isnan(x)]
    return {"p50": float(np.percentile(x, 50) * 1000), "p95": float(np.percentile(x, 95) * 1000), "n": int(x.size)}


def _ratio(g, m):
    """GNN / MLP of the p50s over the questions both have, and the 2.5th and 97.5th percentiles of that ratio over NBOOT
    resamples of the questions with replacement (numpy default_rng(0): the same resampled questions for every metric)."""
    sel = ~(np.isnan(g) | np.isnan(m))
    g, m = g[sel], m[sel]
    idx = np.random.default_rng(0).integers(0, g.size, (NBOOT, g.size))
    bs = np.median(g[idx], axis=1) / np.median(m[idx], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"ratio": float(np.median(g) / np.median(m)), "lo": float(lo), "hi": float(hi), "n": int(g.size)}


def _noise(s):
    """Beside, deciding nothing: the first half of the questions against the second (a drift in the host's load shows
    here), each model led first against second, and the tail of the model-only times."""
    h = s["q"].size // 2
    ms = lambda x: float(np.median(x) * 1000)                               # noqa: E731
    even = s["q"] % 2 == 0                                                  # the MLP led on even questions
    mo = {k: s[k]["model_only"] for k in KEYS}
    return {"p50_ms_first_half_second_half": {"a": [ms(s["a"]["total"][:h]), ms(s["a"]["total"][h:])],
                                              **{k: [ms(mo[k][:h]), ms(mo[k][h:])] for k in KEYS}},
            "p50_ms_led_first_second": {"mlp": [ms(mo["mlp"][even]), ms(mo["mlp"][~even])],
                                        "gnn": [ms(mo["gnn"][~even]), ms(mo["gnn"][even])]},
            "p95_over_p50_model_only": {k: float(np.percentile(mo[k], 95) / np.median(mo[k])) for k in KEYS},
            "p95_over_p50_a": float(np.percentile(s["a"]["total"], 95) / np.median(s["a"]["total"]))}


def _load_summary(h):
    lp = h["load_and_pin"]
    pin = lp.get("pin") or {}
    pl = [int(x) for x in pin.get("affinity_lps", [])]
    out = {"pinned": bool(pin.get("pinned")), "lps": pl}
    for k, name in (("cpu_percent_before", "before"), ("cpu_percent_after", "after")):
        v = lp.get(k)
        if v is None:
            out[name] = None
            continue
        v = np.asarray(v, float)
        out[name] = {"pinned_lps_mean": float(v[pl].mean()) if pl else None, "all_lps_mean": float(v.mean()),
                     "all_lps_max": float(v.max()), "lps_over_50": int((v > 50).sum()), "n_lps": int(v.size),
                     "per_lp": [float(x) for x in v]}
    return out


def _reduce(raw):
    s = _series(raw["rows"], raw["kept"])
    h = raw["head"]
    st = {"a": {k: _pq(v) for k, v in s["a"].items()}, **{key: {k: _pq(v) for k, v in s[key].items()} for key in KEYS}}
    ratios = {k: _ratio(s["gnn"][k], s["mlp"][k]) for k in ("b", "c", "model_only", "end_to_end", "warm_b1", "warm_b16")}
    return {"questions": h["questions"], "kept": int(s["q"].size), "blas": h.get("blas"), "pass": h.get("pass"),
            "utc": h["utc"], "fits": h["fits"], "threads": h["threads"], "code_conflicts": h["code_conflicts"],
            "script_sha256": h["script_sha256"], "pool_rows": {"p50": float(np.median(s["n"])), "max": int(s["n"].max())},
            "checks": h["checks"], "load": _load_summary(h), "stages_ms": st, "ratios_gnn_over_mlp": ratios,
            "a_share_of_end_to_end": {key: st["a"]["total"]["p50"] / st[key]["end_to_end"]["p50"] for key in KEYS},
            "noise": _noise(s)}


def _md(rep):
    L = []
    w = L.append
    pq = lambda d: f"{d['p50']:.1f} / {d['p95']:.1f}"                       # noqa: E731
    rt = lambda d: f"{d['ratio']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"       # noqa: E731
    w("# FSV timing, Family 1, A2 forms")
    w("")
    w("Family 1 is zrckpn (the MLP) against zg6r (the GNN), every input kept, each at the stage forms of amendment A2 "
      "(`docs/FSV_FAST_VERSIONS.md` section 10). Times are ms per question at batch 1 on the host's CPU, cold unless "
      "labelled warm; each cell is p50 / p95 over the kept questions. Stage (a), the shared pool, is charged in full to "
      "both models: model-only = (b) + (c), end to end = (a) + (b) + (c). A faster exact form found later is an "
      "amendment and a re-time of these numbers. Family 2 (the fast versions) is not in this file.")
    w("")
    w("Ratios are GNN / MLP of the p50s, paired by question, with the 2.5th and 97.5th percentiles over "
      f"{NBOOT:,} resamples of the questions with replacement (`numpy.random.default_rng(0)`).")
    for tag, P in rep["passes"].items():
        D = P["datasets"]
        w("")
        w(f"## {P['label']}")
        w("")
        w("### Cold, per stage (ms, p50 / p95)")
        w("")
        w("| dataset | kept | pool rows p50 | (a) | MLP (b) | MLP (c) | MLP model-only | MLP end to end | GNN (b) | "
          "GNN (c) | GNN model-only | GNN end to end |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ds, d in D.items():
            st = d["stages_ms"]
            w(f"| {ds} | {d['kept']} / {d['questions']} | {d['pool_rows']['p50']:.0f} | {pq(st['a']['total'])} | "
              + " | ".join(pq(st[k][f]) for k in KEYS for f in ("b", "c", "model_only", "end_to_end")) + " |")
        w("")
        w("### Ratio GNN / MLP, and stage (a)'s share of each end-to-end time")
        w("")
        w("| dataset | (b) | (c) | model-only | end to end | (a) p50 ms | (a) share of MLP end to end | "
          "(a) share of GNN end to end |")
        w("|---|---|---|---|---|---|---|---|")
        for ds, d in D.items():
            r = d["ratios_gnn_over_mlp"]
            w(f"| {ds} | {rt(r['b'])} | {rt(r['c'])} | {rt(r['model_only'])} | {rt(r['end_to_end'])} | "
              f"{d['stages_ms']['a']['total']['p50']:.1f} | {100 * d['a_share_of_end_to_end']['mlp']:.0f}% | "
              f"{100 * d['a_share_of_end_to_end']['gnn']:.0f}% |")
        w("")
        w("### Parts of (a) and (b) (p50 ms)")
        w("")
        w("| dataset | (a) pool | base | links | rule: walks | orders | KT | other | MLP compile | rows | links | NEXT | "
          "inputs | GNN compile | rows | links | inputs |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ds, d in D.items():
            st = d["stages_ms"]
            ca = [f"{st['a'][k]['p50']:.1f}" for k in ("pool", "base", "links", "walks", "orders", "kt", "rule_other")]
            cm = [f"{st['mlp'][k]['p50']:.1f}" for k in ("compile", "rows", "links", "next", "inputs")]
            cg = [f"{st['gnn'][k]['p50']:.1f}" for k in ("compile", "rows", "links", "inputs")]
            w(f"| {ds} | " + " | ".join(ca + cm + cg) + " |")
        w("")
        w("### Warm (the forward alone, per question; labelled warm)")
        w("")
        w("| dataset | MLP b1 | GNN b1 | ratio b1 | MLP b16 | GNN b16 | ratio b16 |")
        w("|---|---|---|---|---|---|---|")
        for ds, d in D.items():
            st, r = d["stages_ms"], d["ratios_gnn_over_mlp"]
            w(f"| {ds} | {pq(st['mlp']['warm_b1'])} | {pq(st['gnn']['warm_b1'])} | {rt(r['warm_b1'])} | "
              f"{pq(st['mlp']['warm_b16'])} | {pq(st['gnn']['warm_b16'])} | {rt(r['warm_b16'])} |")
        w("")
        w("### Checks, pin and the host's load (1 s samples before and after the timed pass)")
        w("")
        w("| dataset | questions failing a check | pool failing | score checks failing MLP / GNN | max score diff MLP / GNN | "
          "NEXT values differing | code conflicts | pinned | load before: pinned LPs / all mean / max | "
          "load after: pinned LPs / all mean / max |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for ds, d in D.items():
            c, ld = d["checks"], d["load"]
            f = lambda x: "n/a" if x is None else (                           # noqa: E731
                f"{x['pinned_lps_mean']:.0f}% / {x['all_lps_mean']:.0f}% / {x['all_lps_max']:.0f}%")
            w(f"| {ds} | {c['failing']} of {c['questions']} | {c['pool_failing']} | "
              f"{c['mlp']['failing']} / {c['gnn']['failing']} | "
              f"{c['mlp']['max_diff']:.2g} / {c['gnn']['max_diff']:.2g} | {c['next_flip_values']} | "
              f"{d['code_conflicts']} | {ld['pinned']} | {f(ld['before'])} | {f(ld['after'])} |")
        w("")
        w("### Noise, beside (p50 ms; deciding nothing)")
        w("")
        w("| dataset | (a): first / second half | MLP model-only: first / second half | GNN model-only: first / second half | "
          "MLP led first / second | GNN led first / second | p95 / p50 of (a), MLP, GNN |")
        w("|---|---|---|---|---|---|---|")
        for ds, d in D.items():
            n = d["noise"]
            h2 = n["p50_ms_first_half_second_half"]
            lf = n["p50_ms_led_first_second"]
            w(f"| {ds} | {h2['a'][0]:.0f} / {h2['a'][1]:.0f} | {h2['mlp'][0]:.0f} / {h2['mlp'][1]:.0f} | "
              f"{h2['gnn'][0]:.0f} / {h2['gnn'][1]:.0f} | {lf['mlp'][0]:.0f} / {lf['mlp'][1]:.0f} | "
              f"{lf['gnn'][0]:.0f} / {lf['gnn'][1]:.0f} | {n['p95_over_p50_a']:.2f}, "
              f"{n['p95_over_p50_model_only']['mlp']:.2f}, {n['p95_over_p50_model_only']['gnn']:.2f} |")
    w("")
    w("## Caveats")
    w("")
    w("- The host is shared: other jobs run on it while a timed job does, so every time carries that noise. The process "
      "is pinned to two logical processors (its own affinity, EcoQoS opted out, above-normal priority), the models "
      "alternate in leading each question, and the load of every logical processor is sampled for 1 s before and "
      "after the pass; the other jobs are not pinned and may share those processors.")
    w("- The first-stage retrieval (the question's encoding, the dense and SPLADE lists) is not timed; PKO's prepared "
      "lists are read before timing as index-time data.")
    w("- Stage (a) is PKO's own rule on one question at batch 1 on one thread; no fused forward exists for either "
      "model (eager torch, one thread); OpenBLAS runs on 2 threads in the record and on 1 in the beside pass.")
    w("- Warm times are the forward alone on inputs built beforehand; b16 is the time of 12 batches of the first 192 "
      "questions over 16, so it has no values for the last 8 questions.")
    w("")
    return "\n".join(L)


def report(a):
    raw_dir = Path(a.raw_dir) if a.raw_dir else RAW
    out_dir = Path(a.out_dir) if a.out_dir else OUT
    rep = {"label": "A2 forms", "family": "1: zrckpn (MLP) against zg6r (GNN), every input kept",
           "amendment": "docs/FSV_FAST_VERSIONS.md section 10 (A2)",
           "recipe": {"cells": "p50 and p95 in ms over the kept questions (numpy.percentile, linear)",
                      "end_to_end": "(a) + (b) + (c), stage (a) charged in full to both models",
                      "ratio": f"GNN / MLP of the p50s; 2.5th and 97.5th percentiles over {NBOOT} question resamples "
                               "(numpy default_rng(0))",
                      "warm": "the forward alone; b16 = the batch's time / 16 over 12 batches of the first 192 questions"},
           "passes": {}}
    for tag, label in PASSES:
        ds = {}
        for name in T.DATASETS:
            f = raw_dir / f"{name}__{tag}.json"
            if f.exists():
                ds[name] = _reduce(json.loads(f.read_text(encoding="utf-8")))
        if ds:
            rep["passes"][tag] = {"label": label, "datasets": ds}
    if not rep["passes"]:
        raise SystemExit(f"zfsvc report: no <dataset>__b2.json or __b1.json in {raw_dir}")
    T.write_json(out_dir / "report.json", rep)
    (out_dir / "report.md").write_text(_md(rep), encoding="utf-8", newline="\n")
    log(f"report written to {out_dir.name}: passes {list(rep['passes'])}, datasets "
        f"{ {t: list(p['datasets']) for t, p in rep['passes'].items()} }")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("dev", "run", "report", "threads"))
    p.add_argument("--dataset", choices=T.DATASETS)
    p.add_argument("--questions", type=int, default=2)
    p.add_argument("--blas", default="1")
    p.add_argument("--nq", type=int, default=0, help="run: a smoke run of this many questions (raw_smoke), 0 = the 200")
    p.add_argument("--lps", default="2,4", help="run: the logical processors the process is pinned to")
    p.add_argument("--require-pin", dest="require_pin", action="store_true")
    p.add_argument("--full-rule", dest="full_rule", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="run: replace an existing raw file (a failed pass's)")
    p.add_argument("--raw-dir", dest="raw_dir", default="", help="report: the raw files (default outputs/fsv/timing/raw)")
    p.add_argument("--out-dir", dest="out_dir", default="", help="report: where report.json and report.md go")
    a = p.parse_args(argv)
    if a.cmd == "report":
        return report(a)
    _load()
    if a.cmd == "threads":                                                  # the pools after the imports, no data opened
        log(f"threads {json.dumps(set_threads(BLAS))}")
        return 0
    return {"dev": dev, "run": run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
