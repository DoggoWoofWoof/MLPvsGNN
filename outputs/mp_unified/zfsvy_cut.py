"""FSV amendment A3, correction 2 (docs/FSV_FAST_VERSIONS.md section 15): the cut GNN's path. Family 2 is the GNN's fast version,
zg6r without topo_STRUCT, depth_STRUCT, WALK and WALKF (level L2, ADMITTED in 634a8ad), against the full zrckpn. This file holds
the forms of that path, the check without data (`selftest`), the reference read of the cut fits (`ref`) and the development job
(`devy`). zfsvc.py, zfsvx.py, zfsvy.py and zfsvy_b16.py are imported and not edited, as zfsvy_b16.py did; no timed number is made
here (M4's `run` and `report` come by a later correction, before any M4 pass).

The forms, put in force in this process only around the cut GNN's own run and put back after it (also after an error):
- compile (G_struct): c2_fast.compile_mlp is replaced by a copy (compile_cut) of its lines up to and including its `edges` lap, then
  its return, so the laps `topology`, `D` and `depth_basis` do not run and the columns they fill stay zero; c11_fast.compile_serial,
  the module's own, calls it. The retrieval and edge laps are the module's lines (selftest compares them, line for line);
- rows (G_walk): c2_fast.walk_block and walkf_block, the two calls c10_zn.lean_rows10 makes for the walk columns, are replaced by
  functions that return zeros of their shapes (16 and 18 columns); every other line of lean_rows10 is the module's own;
- assemble and forward: zfsvc.run_model, the module's, with the GNN's entries of S.models, S.blocks and S.boxes swapped for the cut
  fit's (the cut view): zfsvc.build over the cut pick's blocks, and the forward of the cut fit's own models.pt.

    python outputs/mp_unified/zfsvy_cut.py selftest
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy_cut.py ref --dataset musique --blas 6
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy_cut.py devy --dataset musique --questions 40 --blas 2 --lp 8
"""
import sys

import zfsvy_b16 as B                       # first: it imports zfsvy, zfsvx and zfsvc, which set the thread variables before numpy loads

import argparse
import contextlib
import gc
import inspect
import json
import time
import traceback
from types import SimpleNamespace

Y = B.Y
X = Y.X
C = X.C
T = C.T
np = C.np
clock = C.clock

LEVEL = "L2"
CUT = ("topo_STRUCT", "depth_STRUCT", "WALK", "WALKF")      # zfsv.cut_of("L2"): the blocks the cut leaves out
KEPT = ("rank", "dense_cos", "SEMB", "SEED", "DISTS")       # the cut pick, in DOB's order
VERSIONS = ("mlp", "gnn", "cutref", "cut")                  # the full MLP; the full GNN; the full path with the cut pick; the cut path
SEGMENTS = Y.SEGMENTS
PACK_KEYS = ("n", "P", "q16", "kt", "ne", "lu", "lv", "lf")
STATS = {"compile_cut": 0, "walk_zero": 0, "walkf_zero": 0}

FF = V2 = IDX = C2 = None                   # c2_fast's names, bound by bind() once zfsvc._load has imported the C-stage modules
_ORIG = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zfsvy_cut: {msg}", flush=True)


def bind():
    """The names the copy of c2_fast's compile uses, and the module's own functions the forms replace."""
    global FF, V2, IDX, C2
    C._load()
    C2 = C.C11.C2
    if C.C10.C2 is not C2:
        raise SystemExit("c10_zn and c11_fast hold different c2_fast modules")
    FF, V2, IDX = C2.FF, C2.V2, C2.IDX
    for k in ("compile_mlp", "walk_block", "walkf_block"):
        _ORIG.setdefault(k, getattr(C2, k))


def _zfsvg():
    """zfsvg (the cut fits' names, pick and read context), imported on first use with the thread variables kept."""
    with X.thread_env_kept():
        import zfsvg as G
    return G


# ── the forms ────────────────────────────────────────────────────────────────


def compile_cut(fc, inp, pool, seeds, embeddings=None, timings=None):
    """c2_fast.compile_mlp, line for line, calling _compile_cut."""
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
        STATS["compile_cut"] += 1
        return _compile_cut(fc, inp, pool, n, seeds, embeddings, lap)
    finally:
        lookup[pool] = -1


def _compile_cut(fc, inp, pool, n, seeds, embeddings, lap):
    """c2_fast._compile_mlp up to and including its `edges` lap, then its return: no topology, D or depth basis."""
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
    return FF.Compiled(pool=pool, seeds_local=seeds_local, scalars=X, edges=edges, seedw=None)


def walk_zeros(n, seeds, buckets, eu, ev, efam, efwd, ebwd):
    """c2_fast.walk_block's place in the cut path: the WALK columns are not in the cut pick, so zeros of its shape."""
    STATS["walk_zero"] += 1
    return np.zeros((n, 16), np.float32)


def walkf_zeros(n, seeds, buckets, eu, ev, efam, efwd, ebwd):
    """c2_fast.walkf_block's place in the cut path: zeros of its shape (the WALK columns and two more)."""
    STATS["walkf_zero"] += 1
    return np.zeros((n, 18), np.float32)


def put_back_ok():
    return all(getattr(C2, k) is _ORIG[k] for k in ("compile_mlp", "walk_block", "walkf_block"))


@contextlib.contextmanager
def cut_forms():
    """The three replacements in force for the length of the block; put back after, also on an error."""
    if not put_back_ok():
        raise SystemExit("the cut forms are already in force")
    C2.compile_mlp, C2.walk_block, C2.walkf_block = compile_cut, walk_zeros, walkf_zeros
    try:
        yield
    finally:
        for k in ("compile_mlp", "walk_block", "walkf_block"):
            setattr(C2, k, _ORIG[k])
        if not put_back_ok():
            raise SystemExit("the cut forms were not put back")


@contextlib.contextmanager
def cut_view(S):
    """The GNN's entries of S.models, S.blocks and S.boxes swapped for the cut fit's, so that zfsvc.run_model, build, forward,
    check_scores and the model line read the cut model; put back after, also on an error."""
    saved = (S.models["gnn"], S.blocks["gnn"], S.boxes["gnn"])
    S.models["gnn"], S.blocks["gnn"], S.boxes["gnn"] = S.cut.model, S.cut.blocks, S.cut.box
    try:
        yield S
    finally:
        S.models["gnn"], S.blocks["gnn"], S.boxes["gnn"] = saved


def run_version(S, v, j, A, T_):
    """Question j on the pool A as version v does it: zfsvc.run_model's result (scores, pack, (features, question index, base z))."""
    if v in ("mlp", "gnn"):
        return C.run_model(S, v, j, A, T_)
    with cut_view(S):
        if v == "cutref":                                   # the module's own compile and rows, the cut pick selected from them
            return C.run_model(S, "gnn", j, A, T_)
        with cut_forms():                                   # the cut path
            return C.run_model(S, "gnn", j, A, T_)


# ── the cut fit: its model, its pick and its reference read ──────────────────


def cut_fit_of(G, ds):
    """The cut zg6r fit (seed 0) of the split the full GNN is timed on for ds: the full fit's (musique on WQ-hotpotqa's, the others on
    WQ-musique's)."""
    full = T.FITS[("gnn", ds)]
    split = G.M1.ALL_FITS[full][2]
    fit = G.fit_name(LEVEL, 0, split)
    if G.full_of(fit) != full:
        raise SystemExit(f"zfsvy_cut: {fit}'s full fit is {G.full_of(fit)}, not {full}")
    return fit


def ref_gnn_cut(fit, ds, nq, got):
    """zfsvt.ref_gnn for a cut zg6r fit: zml1s.run_arm's read under zfsvg's context for the fit (FSV's names, the level's pick,
    p@swa alone), without zml1s.read's stamp: zrm's read reaches zfsvt.read_ref."""
    import lean_gpu as LG
    G = _zfsvg()
    M1 = G.M1
    if fit not in G.G6_FITS:
        raise SystemExit(f"zfsvy_cut: {fit} is not one of FSV's zg6r fits")
    _arm, _seed, split = G.FITS[fit]
    M1.import_arm(G.GNN)                                    # outside zwq.on_wq, as zfsvg.fit does
    with G.on_fit(fit, LG):
        M1.fit_stamp(fit, what="read")
        with T.capturing(LG, ds, nq, got):
            rc, extra = M1.run_arm(fit, None, "read")
    return rc, {"split": split, "records": {k: v for k, v in extra.items() if k != "encoder"}}


@contextlib.contextmanager
def cut_fits():
    """zfsvt's fit table and GNN reference read pointing at the cut fits for the length of the block; put back after."""
    G = _zfsvg()
    saved_fits, saved_ref = dict(T.FITS), T.ref_gnn
    for ds in T.DATASETS:
        T.FITS[("gnn", ds)] = cut_fit_of(G, ds)
    T.ref_gnn = ref_gnn_cut
    try:
        yield
    finally:
        T.FITS.clear()
        T.FITS.update(saved_fits)
        T.ref_gnn = saved_ref


def ref(a):
    """The cut fit's reference read of the dataset's first 201 questions on the CPU, through zfsvt.ref_cmd (the same chain, flags, class,
    roots and read cap as the fit's read, read_carve's batches, then every question at batch 1), checked against the cut fit's filed read
    file; writes outputs/fsv/timing/ref/<cut fit>/<dataset>.npz and .json, nothing under outputs/screen/fits, and touches no stamp."""
    with cut_fits():
        fit = T.FITS[("gnn", a.dataset)]
        log(f"reference read of {fit} on {a.dataset}")
        return T.ref_cmd("gnn", a.dataset, int(a.questions or T.NREF))


def attach_cut(S, ds, nq):
    """The cut fit's model, blocks and reference arrays on S.cut, built through the cut fit's own read chain (zfsvc.grab's, with the
    fit table and the chain pointing at the cut fit). The carve is the full GNN's, which is the cut fit's: the same cache, split and basis
    (checked here), so no second carve is held."""
    G = _zfsvg()
    fit = cut_fit_of(G, ds)
    full = S.boxes["gnn"]
    got, box = [], {}

    def fake(LGm, a, ds_, nq_):
        flags = LGm.set_flags("cpu")
        LGm.bind_device_ops()
        if a.host:
            import lean_host as LH
            box["placement"] = LH.substitute()
        fdir = T.Path(a.out_root or LGm.OUT) / a.name
        blob = C.torch.load(fdir / "models.pt", weights_only=False)
        blob["candidates"] = [T.CAND]
        _nm, m, bl = LGm.load_models(blob, "cpu")[0]
        c = full["carve"]
        if blob["basis"] != full["basis"] or blob["basis_sha256"] != c.basis_sha256:
            raise SystemExit(f"zfsvy_cut: {a.name}'s basis is not the full GNN carve's ({blob['basis']}, {blob['basis_sha256']})")
        box.update(model=m, blocks=list(bl), basis=blob["basis"], flags=flags)
        return {"ids": [str(x) for x in list(c.ids)[:nq_]]}, {}

    orig = T.read_ref
    T.read_ref = fake
    try:
        rc, info = ref_gnn_cut(fit, ds, nq + 1, got)
    finally:
        T.read_ref = orig
    if rc != 0 or "model" not in box:
        raise SystemExit(f"zfsvy_cut: the read chain of {fit} exited {rc} and {'did' if 'model' in box else 'never'} reached read_ref")
    box["model"].eval()
    if box["blocks"] != list(G.pick_of(LEVEL)) or box["blocks"] != [b for b in S.blocks["gnn"] if b not in CUT]:
        raise SystemExit(f"zfsvy_cut: {fit} holds blocks {box['blocks']}, not the cut pick {list(G.pick_of(LEVEL))}")
    p = T.ref_dir(fit) / f"{ds}.npz"
    if not p.is_file():
        raise SystemExit(f"zfsvy_cut: {p} is missing: run `zfsvy_cut.py ref --dataset {ds}` first")
    z = np.load(p)
    ref_ = {k: z[k] for k in z.files}
    if [str(x) for x in ref_["ids"][:nq + 1]] != list(S.ids):
        raise SystemExit(f"zfsvy_cut: {p.name}'s first questions are not the population's")
    cutbox = {"model": box["model"], "blocks": box["blocks"], "carve": full["carve"], "cache_root": full["cache_root"],
              "basis": box["basis"], "flags": box["flags"], "fit": fit, "info": info, "ref": ref_}
    S.cut = SimpleNamespace(model=box["model"], blocks=box["blocks"], box=cutbox, fit=fit)
    log(f"cut fit {fit}: blocks {box['blocks']}, {sum(int(p_.numel()) for p_ in box['model'].parameters())} parameters")
    return S


# ── the checks ───────────────────────────────────────────────────────────────


def same_kept(xa, xb, spans):
    """The two feature matrices byte for byte in the given column spans."""
    return all(X.same_bytes(xa[:, a:e], xb[:, a:e]) for a, e in spans)


def same_val(a, b):
    return bool(a == b) if np.isscalar(a) else X.same_bytes(a, b)


def bad_of(res):
    """The failing entries of a checks dict (zfsvc.dev_checks' rule)."""
    return {k: v for k, v in res.items()
            if v is False or (isinstance(v, dict) and (v.get("ok") is False or v.get("identical") is False))}


def cut_checks(S, j, ref, cut):
    """Question j's checks of the cut path (inside the cut view): against the full path with the cut pick selected from its inputs
    (ref), byte for byte: each kept column of X, the pack's inputs, the features, the question index, the base z and the scores; and
    A2's against the reference: the features and base z against the cut fit's carve batch([j]) and both paths' scores against the cut
    fit's filed read (within 1e-4, the same top five). Returns (checks, information)."""
    (s_r, p_r, f_r), (s_c, p_c, f_c) = ref, cut
    spans = [S.su.span[b] for b in S.blocks["gnn"] if b != "SEMB"]
    res = {"x_kept": same_kept(p_r["X"], p_c["X"], spans)}
    per = {k: same_val(p_r[k], p_c[k]) for k in PACK_KEYS}
    res["pack"] = {"identical": all(per.values()), "per_key": [k for k, v in per.items() if not v]}
    res["feats_same"] = Y.same_tree(f_r[0], f_c[0])
    res["nq_same"] = Y.same_tree(f_r[1], f_c[1])
    res["base_z_same"] = Y.same_tree(f_r[2], f_c[2])
    res["scores_same"] = X.same_bytes(s_r, s_c)
    carve = S.boxes["gnn"]["carve"]
    rf, _rnq, rbz, _g = carve.batch(np.asarray([j]), S.blocks["gnn"])
    ok, md, nd = C.diff(f_c[0], rf)
    res["feats_carve"] = {"identical": ok, "max_diff": md, "differing": nd,
                          "per_key": {k: C.diff(f_c[0].get(k), rf.get(k))[:3] for k in rf if not C.diff(f_c[0].get(k), rf.get(k))[0]}}
    res["base_z_carve"] = dict(zip(("identical", "max_diff", "differing"), C.diff(f_c[2], rbz)))
    res["cutref_scores"] = C.check_scores(S, "gnn", j, s_r)
    res["cut_scores"] = C.check_scores(S, "gnn", j, s_c)
    cut_cols = [S.su.span[b] for b in CUT]
    info = {"cut_columns_zero": all(not np.asarray(p_c["X"][:, a:e]).any() for a, e in cut_cols)}
    return res, info


# ── the development job ──────────────────────────────────────────────────────


def devy(a):
    """The cut GNN's development check. Per question, in a rotating order, four versions: the full MLP, the full GNN, the full path with the
    cut pick selected from its inputs (cutref) and the cut path (cut). Checks, each question: A2's (zfsvc.dev_checks) on the full MLP and
    GNN with A4's and A6's forms installed; the cut path against cutref byte for byte (kept columns of X, the pack's inputs, features, base
    z, scores) and against the cut fit's carve batch and filed read (within 1e-4, the same top five). Then the warm batch of 16 on the cut
    packs with A7's step against the module's step, byte for byte. Development numbers: no timed pass."""
    C._load()
    bind()
    log(f"pin: {B.pin_two(a.lp)}")
    nq = int(a.questions or 40)
    with Y.installed():
        S = C.setup(a.dataset, nq)
        attach_cut(S, a.dataset, nq)
        C.prepass(S, range(nq + 1))
        S.threads = C.set_threads(C.BLAS)
        log(f"threads {S.threads}")
        log(X.model_line(S, "mlp"))
        log(X.model_line(S, "gnn"))
        with cut_view(S):
            log("cut " + X.model_line(S, "gnn"))
        A = C.stage_a(S, nq, {})                            # the JIT question, every version
        C.fill_codes(S, nq, A.pool)
        for v in VERSIONS:
            run_version(S, v, nq, A, {})
        STATS.update({k: 0 for k in STATS})
        gc.collect()
        seg = {v: {} for v in VERSIONS}
        failing, fail_names, cut_zero = [], {}, 0
        md = {"mlp": 0.0, "gnn": 0.0, "cutref": 0.0, "cut": 0.0}
        top5 = {k: 0 for k in md}
        flips = vals = fq = 0
        sizes, cut_packs, cut_scores = [], [], []
        for j in range(nq):
            A = C.stage_a(S, j, {})
            C.fill_codes(S, j, A.pool)
            outs = {}
            k0 = j % len(VERSIONS)
            for v in VERSIONS[k0:] + VERSIONS[:k0]:
                T_ = {}
                outs[v] = run_version(S, v, j, A, T_)
                for k in SEGMENTS:
                    if k in T_:
                        seg[v].setdefault(k, []).append(float(T_[k]))
            res = C.dev_checks(S, j, A, {"mlp": outs["mlp"], "gnn": outs["gnn"]})
            bad = bad_of(res)
            if set(bad) == {"mlp_feats"} and set(bad["mlp_feats"]["per_key"]) == {"NEXT"}:
                bad = {}                                    # float16 NEXT values a step off are counted
            with cut_view(S):
                cres, info = cut_checks(S, j, outs["cutref"], outs["cut"])
            cbad = bad_of(cres)
            cut_zero += bool(info["cut_columns_zero"])
            allbad = {**bad, **{f"cut:{k}": v for k, v in cbad.items()}}
            if allbad:
                failing.append(j)
                for k in allbad:
                    fail_names.setdefault(k, []).append(j)
                log(f"q{j} FAILED: {json.dumps(allbad, default=str)[:1200]}")
            flips += res["next_flips"]
            vals += res["next_values"]
            fq += res["next_flips"] > 0
            for key in ("mlp", "gnn"):
                md[key] = max(md[key], res[f"{key}_scores"]["max_diff"] or 0.0)
                top5[key] += bool(res[f"{key}_scores"]["top5_same"])
            for key in ("cutref", "cut"):
                md[key] = max(md[key], cres[f"{key}_scores"]["max_diff"] or 0.0)
                top5[key] += bool(cres[f"{key}_scores"]["top5_same"])
            sizes.append((int(A.pool.size), int(outs["gnn"][1]["ne"])))
            cut_packs.append(outs["cut"][1])
            cut_scores.append(outs["cut"][0])
            tot = {v: seg[v]["total"][-1] * 1000 for v in VERSIONS}
            log(f"q{j} n={A.pool.size} edges={sizes[-1][1]} (b)+(c) ms: mlp {tot['mlp']:.1f} gnn {tot['gnn']:.1f} cutref {tot['cutref']:.1f} "
                f"cut {tot['cut']:.1f}; gnn vs cut compile {seg['gnn']['compile'][-1] * 1000:.1f} -> {seg['cut']['compile'][-1] * 1000:.1f}, "
                f"lean {seg['gnn']['lean'][-1] * 1000:.1f} -> {seg['cut']['lean'][-1] * 1000:.1f}, forward "
                f"{seg['gnn']['forward'][-1] * 1000:.1f} -> {seg['cut']['forward'][-1] * 1000:.1f}; scores max diff mlp {res['mlp_scores']['max_diff']} "
                f"gnn {res['gnn_scores']['max_diff']} cut {cres['cut_scores']['max_diff']}")
            del outs, res, cres
        ran = Y.forms_ran(nq)                               # before the batches of 16, whose steps are not counted by zfsvx
        ran_cut = (STATS["compile_cut"] == nq and STATS["walk_zero"] == nq and STATS["walkf_zero"] == nq and put_back_ok())
        b16 = batch16(S, cut_packs, cut_scores)
        stats, xstats = dict(STATS), dict(X.STATS)
        ystats = dict(Y.STATS)
        fast_index = dict(S.fast_index)
        fits = {"mlp": S.boxes["mlp"]["fit"], "gnn": S.boxes["gnn"]["fit"], "cut": S.cut.fit}
    sz = np.array(sizes)
    log(f"pool rows p50 {int(np.median(sz[:, 0]))} (max {int(sz[:, 0].max())}), link edges p50 {int(np.median(sz[:, 1]))} (max "
        f"{int(sz[:, 1].max())})")
    out = {v: {k: [Y.p50(x), Y.mean(x)] for k, x in seg[v].items()} for v in VERSIONS}
    for k in SEGMENTS:
        if k in seg["gnn"]:
            log(f"  {k:<9} p50 ms " + "  ".join(f"{v} {out[v][k][0]:7.2f}" for v in VERSIONS if k in seg[v]))
    t = {v: out[v]["total"][0] for v in VERSIONS}
    log(f"(b)+(c) p50: mlp {t['mlp']:.2f}, gnn {t['gnn']:.2f}, cutref {t['cutref']:.2f}, cut {t['cut']:.2f} ms; cut vs gnn x{t['gnn'] / t['cut']:.2f}, "
        f"cut/mlp {t['cut'] / t['mlp']:.2f}, gnn/mlp {t['gnn'] / t['mlp']:.2f}")
    log("scores vs the filed reads: max diff " + ", ".join(f"{k} {md[k]:.3g}" for k in md) + "; same top five on "
        + ", ".join(f"{k} {top5[k]}" for k in top5) + f" of {nq}")
    log(f"NEXT: {flips} float16 values of {vals} differ from the filed read, in {fq} questions; the cut columns of the cut path's X are zero "
        f"in {cut_zero} of {nq}; forms ran: {ran}, cut forms ran: {ran_cut} {json.dumps(stats)}; batches of 16: {json.dumps(b16)}")
    ok = not failing and ran and ran_cut and b16["same_bytes"] and b16["batches"] > 0
    log("DEVY_JSON " + json.dumps({"dataset": a.dataset, "questions": nq, "blas": C.BLAS, "fits": fits, "fast_index": fast_index, "seg": out,
                                   "max_diff": md, "top5_same": top5, "next_flips": flips, "next_values": vals, "sizes": sizes,
                                   "stats": stats, "stats_x": xstats, "stats_y": ystats, "failing": failing, "failing_by_check": fail_names,
                                   "cut_columns_zero": cut_zero, "ran": bool(ran), "ran_cut": bool(ran_cut), "b16": b16}))
    log("DEVY " + ("PASS" if ok else "FAIL"))
    return 0                                                # the verdict is in the log: a chain of jobs goes on


def batch16(S, packs, scores):
    """The warm batches of 16 on the cut path's packs (inside the cut view), as the timed pass's (zfsvy_b16.forward_t): A4's and A6's
    forms in force, the module's layer step (m5) against A7's batched step (a7), byte for byte, and the batched step's scores against
    the cut path's batch-1 scores (information; the filed read's check is the batch-1 one)."""
    nb = len(packs) // C.B16
    same, diffs = [], []
    with cut_view(S):
        for g in range(nb):
            lo, hi = g * C.B16, (g + 1) * C.B16
            feats, nqt, base_z = C.build(S, "gnn", packs[lo:hi])
            _dt, s_m5 = B.forward_t(S, "gnn", "m5", feats, nqt, base_z)
            s_m5 = s_m5.copy()
            _dt, s_a7 = B.forward_t(S, "gnn", "a7", feats, nqt, base_z)
            same.append(B.same32(s_m5, s_a7))
            want = np.concatenate(scores[lo:hi])
            diffs.append(float(np.abs(s_a7.astype(np.float64) - want.astype(np.float64)).max()))
    return {"batches": nb, "same_bytes": bool(all(same)), "max_diff_vs_b1": max(diffs) if diffs else None}


# ── the check without data ───────────────────────────────────────────────────


def body(fn):
    """The lines of fn's body after its signature and docstring, trailing blanks stripped, blank lines dropped."""
    src = inspect.getsource(fn).splitlines()
    i = 0
    while not src[i].rstrip().endswith(":"):               # the signature
        i += 1
    lines = src[i + 1:]
    first = lines[0].strip()
    if first.startswith('"""'):
        j = 0 if (first.count('"""') >= 2 and len(first) > 3) else 1
        while j and '"""' not in lines[j]:
            j += 1
        lines = lines[j + 1:]
    return [x.rstrip() for x in lines if x.strip()]


def selftest_source():
    """compile_cut is c2_fast.compile_mlp's lines (calling _compile_cut), and _compile_cut is c2_fast._compile_mlp's lines up to and
    including its `edges` lap followed by its return: the retrieval and edge laps are the module's own."""
    bad = 0
    orig = body(_ORIG["compile_mlp"])
    want = [x.replace("_compile_mlp", "_compile_cut") for x in orig]
    want.insert(want.index("        return _compile_cut(fc, inp, pool, n, seeds, embeddings, lap)"),
                '        STATS["compile_cut"] += 1')
    if body(compile_cut) != want:
        bad += 1
        log("compile_cut's lines are not compile_mlp's")
    full = body(C2._compile_mlp)
    k = [i for i, x in enumerate(full) if x.strip() == 'lap("edges")']
    if len(k) != 1:
        raise SystemExit(f"_compile_mlp has {len(k)} edges laps")
    want = full[:k[0] + 1] + [full[-1]]
    got = body(_compile_cut)
    if got != want:
        bad += 1
        log(f"_compile_cut's lines are not _compile_mlp's up to its edges lap and its return ({len(got)} against {len(want)} lines)")
        for i, (x, y) in enumerate(zip(got, want)):
            if x != y:
                log(f"  first difference at line {i}: {x!r} against {y!r}")
                break
    log(f"source: _compile_cut {len(got)} lines of _compile_mlp's {len(full)} (the laps after `edges` and the original's return left out); "
        f"{bad} differ")
    return bad


def selftest_walks(rng):
    """The zero functions return the shapes and dtype of the walks they stand for (checked on a random graph against the module's own),
    are in force only inside cut_forms, and are put back after an error."""
    bad = 0
    n, m = 40, 150
    eu, ev = (rng.integers(0, n, m).astype(np.int16) for _ in range(2))
    efam = rng.integers(0, 3, m).astype(np.int8)
    efwd, ebwd = (rng.integers(0, 2, m).astype(np.int8) for _ in range(2))
    seeds = np.asarray([3, 7, -1, 12], np.int64)
    buckets = np.asarray([0, 1, 0, 1], np.int64)
    args = (n, seeds, buckets, eu, ev, efam, efwd, ebwd)
    for real, zero in ((_ORIG["walk_block"], walk_zeros), (_ORIG["walkf_block"], walkf_zeros)):
        a, z = real(*args), zero(*args)
        if a.shape != z.shape or a.dtype != z.dtype or z.any() or not a.any():
            bad += 1
            log(f"{zero.__name__}: shape {z.shape} {z.dtype} against the walk's {a.shape} {a.dtype}")
    if not put_back_ok():
        bad += 1
        log("the module's functions are not in place before cut_forms")
    with cut_forms():
        in_force = (C.C11.C2.compile_mlp is compile_cut and C.C10.C2.walk_block is walk_zeros and C.C10.C2.walkf_block is walkf_zeros)
        again = False
        try:
            with cut_forms():
                again = True
        except SystemExit:
            pass
    if not in_force or again or not put_back_ok():
        bad += 1
        log(f"cut_forms: in force {in_force}, entered twice {again}, put back {put_back_ok()}")
    try:
        with cut_forms():
            raise RuntimeError("an error inside the block")
    except RuntimeError:
        pass
    if not put_back_ok():
        bad += 1
        log("cut_forms did not put the functions back after an error")
    log(f"walks: shapes, dtype, scope and restore; {bad} differ")
    return bad


def selftest_checks(rng):
    """same_kept sees a changed byte in a kept column and not one in a cut column; the cut view swaps the GNN's entries and puts them back;
    bad_of reads zfsvc's rule."""
    bad = 0
    xa = rng.integers(0, 2 ** 15, (30, 40)).astype(np.uint16).view(np.float16)
    xb = xa.copy()
    kept = [(0, 10), (20, 30)]
    xb[:, 12] = np.float16(1.0)                             # a cut column
    if not same_kept(xa, xb, kept):
        bad += 1
        log("same_kept: a change in a cut column was seen")
    xb[3, 25] = np.float16(2.0) if xa[3, 25] != np.float16(2.0) else np.float16(3.0)
    if same_kept(xa, xb, kept):
        bad += 1
        log("same_kept: a change in a kept column was missed")
    if not same_val(np.arange(3, dtype=np.int64), np.arange(3, dtype=np.int64)) or same_val(np.arange(3), np.arange(3)[::-1].copy()):
        bad += 1
        log("same_val")
    if bad_of({"a": True, "b": {"ok": True}, "c": {"identical": True}}) or set(bad_of({"a": False, "b": {"ok": False}, "c": {"identical": False},
                                                                                    "d": {"max_diff": 1.0}})) != {"a", "b", "c"}:
        bad += 1
        log("bad_of")
    S = SimpleNamespace(models={"mlp": "m", "gnn": "full"}, blocks={"mlp": [1], "gnn": ["f"]}, boxes={"mlp": 1, "gnn": "fullbox"},
                        cut=SimpleNamespace(model="cut", blocks=["c"], box="cutbox"))
    try:
        with cut_view(S):
            inside = (S.models["gnn"], S.blocks["gnn"], S.boxes["gnn"]) == ("cut", ["c"], "cutbox") and S.models["mlp"] == "m"
            raise RuntimeError("an error inside the view")
    except RuntimeError:
        pass
    if not inside or (S.models["gnn"], S.blocks["gnn"], S.boxes["gnn"]) != ("full", ["f"], "fullbox"):
        bad += 1
        log("cut_view: the swap or the restore")
    log(f"checks: same_kept, same_val, bad_of, cut_view; {bad} differ")
    return bad


def selftest_names():
    """The level, its cut, its pick and its fits as zfsvg and zfsv name them: the cut is this file's, the pick is the cut pick in DOB's
    order, and each dataset's cut fit starts from the full fit the full GNN is timed on, on the same split. Needs no data."""
    bad = 0
    G = _zfsvg()
    if set(G.FSV.cut_of(LEVEL)) != set(CUT) or len(G.FSV.cut_of(LEVEL)) != len(CUT):
        bad += 1
        log(f"cut_of({LEVEL}) is {G.FSV.cut_of(LEVEL)}, not {CUT}")
    if tuple(G.pick_of(LEVEL)) != KEPT or KEPT != tuple(b for b in G.DOB.PICK if b not in CUT):
        bad += 1
        log(f"the pick of {LEVEL} is {G.pick_of(LEVEL)}, not {KEPT}")
    names = {}
    for ds in T.DATASETS:
        fit = cut_fit_of(G, ds)
        names[ds] = fit
        sp = G.FITS[fit][2]
        if G.M1.ALL_FITS[T.FITS[("gnn", ds)]][2] != sp or G.LEVEL_OF[fit] != LEVEL or G.FITS[fit][:2] != (G.GNN, 0):
            bad += 1
            log(f"{ds}: {fit} is not the {LEVEL} seed-0 zg6r fit of the full fit's split")
    if names != {"musique": "scr-fsv-zg6r-L2-s0-hp", "hotpotqa": "scr-fsv-zg6r-L2-s0", "2wiki": "scr-fsv-zg6r-L2-s0"}:
        bad += 1
        log(f"the cut fits are {names}")
    before = (dict(T.FITS), T.ref_gnn)
    with cut_fits():
        inside = {ds: T.FITS[("gnn", ds)] for ds in T.DATASETS}
        swapped = T.ref_gnn is ref_gnn_cut
    if inside != names or not swapped or (dict(T.FITS), T.ref_gnn) != before:
        bad += 1
        log("cut_fits: the table or the read function was not swapped or not put back")
    log(f"names: {names}; {bad} differ")
    return bad


def selftest(a):
    C._load()
    bind()
    log(f"threads {C.set_threads(C.BLAS)}")
    rng = np.random.default_rng(23)
    bad = {"source": selftest_source(), "walks": selftest_walks(rng), "checks": selftest_checks(rng), "names": selftest_names()}
    log("SELFTEST " + ("PASS" if not any(bad.values()) else f"FAIL {bad}"))
    return 0 if not any(bad.values()) else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "ref", "devy"))
    ap.add_argument("--dataset", choices=T.DATASETS, default="musique")
    ap.add_argument("--questions", type=int, default=0, help="ref: the questions read (default 201); devy: the questions checked (default 40)")
    ap.add_argument("--blas", default=C.BLAS, help="OpenBLAS threads; zfsvc reads it from argv too (default there: 1)")
    ap.add_argument("--lp", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "devy":                                     # a development job: the verdict is a line of the log, so a chain goes on
        try:
            return devy(a)
        except (Exception, SystemExit) as e:  # noqa: BLE001
            log(f"DEVY STOPPED: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1 if "[WinError 5]" in str(e) or "[WinError 32]" in str(e) else 0
    return {"selftest": selftest, "ref": ref}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
