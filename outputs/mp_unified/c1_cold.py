"""C1 (docs/C1_COLD_COST.md): the cold cost table, zrc (the MLP base) against zsp (the GNN track's base) and the six GNN
(u_gnn_v2_ef in gnn_fast.FastGNN's exact form), per question from its first-stage lists, with no cache and no warm-up.

Per question (the first --queries of the dataset's s1eval look, in carve order), batch 1:
  shared   pool      LT.PoolBuilder on the first-stage dense and SPLADE lists (m3b's pool, as lean_time9)
           read      the pool's rows of the node table (context.nodes.read)
           compile   fast_features.FastCompiler, v2, on those rows (compile_query_v2 bit for bit at the same BLAS setting)
  zrc      edges     the compiled pool edges as the look keeps them (e_u, e_v, e_fam, e_fwd, e_bwd)
           lean      projection, lean_mlp.lean_query (WALK), lean_cache.walkf_of (WALKF), lean_cache.store_seed_dists
                     (SEED, DISTS) on the index-time int8 store codes, and the float16 rows lean_cache writes
           forward   zrm.ZRM (zrc's fit, p@swa) on lean_gpu.CacheCarve.batch's inputs, then top 5
  zsp      the same edges and lean stages (zsp reads zrm's inputs), plus
           links     zlink.chunk_edges and zlink.link_batch on the question's edges
           forward   zprop.ZProp (zsp's fit, p@swa), then top 5
  gnn6     pack      lean_time.fast_pack (pack_queries_v2 for one query, from the compile's own edges)
           forward   gnn_fast.FastGNN of the six pair's gnn0, then top 5
zrc's and zsp's edge and lean stages are computed and timed once for each path (each path is cold on its own); the
shared stages run once per question and count in every path's total. The paths' own stages run in an order that
rotates question by question.

Batch 16: groups of 16 questions in carve order; each question's inputs as at batch 1 (their times are the batch-1
stage times), then one forward per path over the group, timed; per question: the inputs plus the forward over 16.
For gnn6 the group is packed by universal_v2_features.pack_queries_v2 (timed with its forward).

Index time (timed, its own rows): the six pair and the fits loaded, the fast compiler built, the int8 store codes of
every node the measured pools touch (lean_mlp3.Store.codes on unit rows, in blocks of 4096), the numba compile
(one question outside the measured set, the carve's next one, run once).

Checks (untimed): pool, seeds and seed buckets against the look; the edges against the look's; the compiled columns,
the projection and the float16 rows against the look and the step-1 cache (bits; a difference is counted with its
largest size, since the look was built on the host at two BLAS threads); the codes against the cache's; every path's
scores against the same model on the step-1 cache's rows (zrc, zsp: lean_gpu.CacheCarve, with zsp's links from the
look's edges; largest difference, top 5 the same) and gnn6's against the look's stored gnn0 scores.

The process pins itself (laptop): affinity to logical processor 2, power throttling (EcoQoS) off for itself,
ABOVE_NORMAL priority; written beside the output. One BLAS, numba and torch thread.

    python outputs/mp_unified/c1_cold.py run --dataset 2wiki --queries 200 --out outputs/c1/2wiki.json
    python outputs/mp_unified/c1_cold.py report
    python outputs/mp_unified/c1_cold.py --selftest
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
import ctypes  # noqa: E402
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

import gnn_fast as GF  # noqa: E402
import lean_time2 as T2  # noqa: E402
import lean_gpu as LG  # noqa: E402
import lean_cache as LC  # noqa: E402
import zrm as ZM  # noqa: E402
import zlink as ZL  # noqa: E402
import zprop as ZP  # noqa: E402
import mp_approx_l3 as ML3  # noqa: E402
import mp_approx_l8 as ML8  # noqa: E402
from mp_retrieval import m3b_pools  # noqa: E402

LM, L2, L3, LT, FF, S6, V2 = T2.LM, T2.L2, T2.L3, T2.LT, T2.FF, T2.S6, T2.V2
OUT = ROOT / "outputs" / "c1"
FITS = {"zrc": ROOT / "outputs" / "b1b" / "fits" / "zrc" / "J5", "zsp": ROOT / "outputs" / "b1b" / "fits" / "zsp" / "J5"}
CAND = "p@swa"
CARVE = "s1eval"
PATHS = ("zrc", "zsp", "gnn6")
B16 = 16
TYPED = ("metaqa", "webqsp")
clock = time.perf_counter


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bits(a):
    a = np.ascontiguousarray(a)
    return a.view({2: np.uint16, 4: np.uint32, 8: np.uint64, 1: np.uint8}[a.dtype.itemsize])


def diff(a, b):
    """(elements differing as bits, largest absolute difference) of two equal-shape arrays."""
    if a.shape != b.shape:
        return -1, float("inf")
    if a.size == 0:
        return 0, 0.0
    nd = int((bits(a) != bits(b)).sum()) if a.dtype == b.dtype else int((a != b).sum())
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    d = d[np.isfinite(d)]
    return nd, float(d.max()) if d.size else 0.0


# ── pinning (own process only) ───────────────────────────────────────────────


class _PTS(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def pin(lp=2):
    if os.name != "nt":
        return {"pinned": False, "why": "not Windows"}
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    h = k32.GetCurrentProcess()
    out = {"affinity_lp": lp}
    out["affinity_ok"] = bool(k32.SetProcessAffinityMask(h, ctypes.c_size_t(1 << lp)))
    st = _PTS(1, 0x1, 0)                                  # EXECUTION_SPEED controlled, state off: EcoQoS opted out
    out["ecoqos_off_ok"] = bool(k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))
    out["above_normal_ok"] = bool(k32.SetPriorityClass(h, 0x8000))
    out["pinned"] = out["affinity_ok"] and out["ecoqos_off_ok"] and out["above_normal_ok"]
    return out


# ── the one-question carve: lean_gpu.CacheCarve's batch over arrays built here ─


class Rows:
    """CacheCarve's attributes for questions whose rows were built in this process (float16 rows, int8 codes)."""

    def __init__(self, qrows, span, W, c_rrf, store):
        self.device = torch.device("cpu")
        self.span, self.W, self.c_rrf = span, W, c_rrf
        self.n_np = np.asarray([r["n"] for r in qrows], np.int64)
        self.off_np = np.r_[0, np.cumsum(self.n_np)]
        self.rows = len(qrows)
        self.X = torch.from_numpy(np.concatenate([r["X"] for r in qrows]))
        self.gold = torch.zeros(int(self.off_np[-1]), dtype=torch.bool)
        self.tab = torch.from_numpy(np.concatenate([r["codes"] for r in qrows]))
        self.row = torch.arange(int(self.off_np[-1]), dtype=torch.int64)
        self.q_emb = torch.from_numpy(np.stack([r["q_emb16"] for r in qrows]))
        self.s = torch.from_numpy(store.s)
        self.c = torch.from_numpy(np.ascontiguousarray(store.c, np.float32))
        self.kappa = torch.tensor([[store.kappa]], dtype=torch.float32)
        self.chains = None
        if all("links" in r for r in qrows):
            ne = np.asarray([r["links"][0] for r in qrows], np.int64)
            self.links = {"q_ne": ne, "e_off": np.r_[0, np.cumsum(ne)],
                          "e_u": np.concatenate([r["links"][1] for r in qrows]),
                          "e_v": np.concatenate([r["links"][2] for r in qrows]),
                          "e_fam": np.concatenate([r["links"][3] for r in qrows])}
        else:
            self.links = None

    decode = LG.CacheCarve.decode

    def batch(self, qs, blocks, links=False):
        feats, nq, base_z, gold = LG.CacheCarve.batch(self, qs, blocks)
        if links:
            feats[ZL.LK_KEY] = ZL.link_batch(self, qs)
        return feats, nq, base_z, gold


def links_of(n, eu, ev, ef):
    ne, a, b, f, _st = ZL.chunk_edges({"q_pool_size": np.asarray([n]), "q_edges": np.asarray([eu.size]),
                                       "e_u": eu, "e_v": ev, "e_fam": ef}, "c1")
    return int(ne[0]), a, b, f


# ── the cold stages ──────────────────────────────────────────────────────────


class Setup:
    def __init__(self, name, nq):
        t0 = time.time()
        self.name = name
        self.index = {}
        S6.pair_verify(S6.SIX)
        cfg, cfg_m3b, cfg_h = V2.load_configs()
        op = S6.pair_open(S6.SIX, cfg, cfg_m3b, name)
        self.inputs, self.m3b_compile = op.inputs, op.m3b_compile
        self.context, dsh = op.contexts[name], op.handles[name]
        if self.context.rel_table is not None:
            raise SystemExit(f"{name}: typed relations; this harness serves the untyped graphs first")
        self.gnn = S6.pair_models(S6.SIX, self.inputs, op.bank)["gnn0"]
        self.fgnn = GF.FastGNN(self.gnn)
        m3b_contract = V2.M3B_RUN.load_script("m3b_contract")
        m3a = op.pkg[0]
        _k, frozen = self.m3b_compile.frozen_contract(cfg_m3b)
        construction = frozen["per_dataset"][name]["construction"]
        positions = m3a.node_position_map(dsh)
        # the look: its ids in carve order, its chunks (for the checks)
        files, ids, head = look_first(name, nq + 1)
        self.look_head = head
        self.ref = self.look_rows(files, nq + 1)
        ids = ids[:nq + 1]
        self.ids = ids
        pop = carve_population(self.m3b_compile, dsh, ids, CARVE, m3a, positions, name)
        if list(pop.ids) != ids:
            raise SystemExit(f"{name}: a question of the look's first {len(ids)} has no gold in its pool's dataset")
        del positions
        self.prep = self.m3b_compile.prepare(dsh, [pop], construction, cfg_h, self.context.stores, m3a, m3b_contract)[0]
        self.builder = LT.PoolBuilder(construction, cfg_h, self.context.stores, m3a, m3b_contract, self.m3b_compile)
        self.columns = self.inputs["column_indices"]
        self.xcols, spans, ci = LC.xc_columns(head)
        self.c_rrf_x = ci["rrf"]
        self.index["pair_and_lists_s"] = time.time() - t0
        t = time.time()
        self.fc = FF.compiler_for(self.context.stores, self.context.nodes, self.context.rel_table)
        if self.fc is None:
            raise SystemExit("no fast compiler for this context")
        self.index["fast_compiler_s"] = time.time() - t
        # the models
        t = time.time()
        LG.bind_device_ops()
        LG.set_flags("cpu")
        self.models, self.basis = {}, None
        for k, cls in (("zrc", ZM.ZRM), ("zsp", ZP.ZProp)):
            blob = torch.load(FITS[k] / "models.pt", weights_only=False)
            if self.basis not in (None, blob["basis"]):
                raise SystemExit("the fits are on different bases")
            self.basis = blob["basis"]
            ms = {nm: (m, bl) for nm, m, bl in LG.load_models(blob, "cpu", cls=cls)}
            self.models[k] = ms[CAND]
        self.blocks = list(self.models["zrc"][1])
        if list(self.models["zsp"][1]) != self.blocks:
            raise SystemExit("zrc and zsp read different blocks")
        self.index["models_s"] = time.time() - t
        bz, brec = LC.load_basis(self.basis)
        self.store = L3.Store(bz)
        self.basis_sha256 = brec["npz_sha256"]
        self.R = LM.projection()
        # the cache's spans (lean_gpu.CacheCarve's order)
        self.span, at = {}, 0
        for b in LC.XC_BLOCKS:
            self.span[b] = tuple(spans[b])
            at = spans[b][1]
        for b in ("WALK", "WALKF", "SEED", "DISTS"):
            w = LM.LEAN_W.get(b, L2.NEW_W.get(b))
            self.span[b] = (at, at + w)
            at += w
        self.W = at
        self.c_rrf = self.span["rank"][0] + LM.SPLIT["rank"].index("rrf")
        # index time: the int8 codes of every node the measured pools touch
        t = time.time()
        nodes = np.unique(np.concatenate([np.asarray(p, np.int64) for p in self.prep.pools]))
        self.codes = np.zeros((int(self.fc.n_nodes), L3.STORE_K), np.int8)
        for s0 in range(0, nodes.size, 4096):
            rr = nodes[s0:s0 + 4096]
            self.codes[rr] = self.store.codes(L3.unit(np.asarray(self.context.nodes.read(rr), np.float32)))
        self.index["store_codes_s"] = time.time() - t
        self.index["store_codes_nodes"] = int(nodes.size)
        self.index["setup_total_s"] = time.time() - t0

    @staticmethod
    def look_rows(files, nq):
        """The look's arrays per question, for the checks: pool, seeds, buckets, edges, projection, x, gnn0's score."""
        out = []
        for f in files:
            z = np.load(f)
            qps, qed = z["q_pool_size"], z["q_edges"]
            no, eo = np.r_[0, np.cumsum(qps)], np.r_[0, np.cumsum(qed)]
            for i in range(qps.size):
                a, b, ea, eb = no[i], no[i + 1], eo[i], eo[i + 1]
                out.append({"pool": z["pool"][a:b].astype(np.int64), "x": z["x"][a:b], "proj": z["proj"][a:b],
                            "sl": z["q_seed_local"][i], "sb": z["q_seed_bucket"][i],
                            "e": tuple(z[k][ea:eb] for k in ("e_u", "e_v", "e_fam", "e_fwd", "e_bwd")),
                            "gnn0": z["score"][a:b][:, LM.SCORE_COL["gnn0"]]})
                if len(out) >= nq:
                    return out
        return out


def look_first(ds, nq):
    """The look's records and ids (as lean_cache.look_chunks checks them) and only the chunks holding its first nq
    questions; the chunks are numbered in carve order."""
    d = LM.LOOK / ds / CARVE
    recs = sorted(d.glob("record*.json"))
    if not recs:
        raise SystemExit(f"{d}: no look record")
    rr = [json.loads(r.read_text(encoding="utf-8")) for r in recs]
    if any(r["carve_ids_sha256"] != rr[0]["carve_ids_sha256"] or not r.get("full") or r.get("limit") is not None
           for r in rr):
        raise SystemExit(f"{d}: the records disagree, are not --full or carry a --limit")
    ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
    if len(ids) != int(rr[0]["carve_queries"]):
        raise SystemExit(f"{d}: ids.json has {len(ids)} ids, the records {rr[0]['carve_queries']}")
    cq = int(rr[0]["chunk_queries"])
    need = -(-min(nq, len(ids)) // cq)
    files = [d / "chunks" / f"c{i:05d}.npz" for i in range(need)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{d}: chunks {missing[:5]} are not on disk (rx fetch --glob)")
    head = {"records": {r_.name: sha_file(r_) for r_ in recs}, "carve_ids_sha256": rr[0]["carve_ids_sha256"],
            "chunks_read": {f.name: sha_file(f) for f in files}, "blas_threads": rr[0].get("threads"),
            "columns": rr[0]["columns"], "column_blocks": rr[0]["column_blocks"]}
    return files, ids, head


def carve_population(m3b_compile, ds, ids, kind, m3a, positions, name):
    """look_x_six.carve_population (look_score_six's), without the info records. The laptop's copy files the s1eval
    questions under dev where the host's files them under train; the rows are looked up in train, then dev (never
    test), and the pool check against the look holds the two the same."""
    row_of = {qid: j for j, qid in enumerate(ds.query_ids)}
    keep = set(ids)
    by_id = {}
    for split in ("train", "dev"):
        by_id.update({row["query_id"]: row for row in ds.queries(split) if row["query_id"] in keep and row["query_id"] not in by_id})
        if len(by_id) == len(keep):
            break
    rows = [by_id[q] for q in ids]
    idx = np.asarray([row_of[q] for q in ids], dtype=np.int64)
    golds = m3a.resolve_gold(rows, positions, name)
    zero = np.asarray([g.size == 0 for g in golds], dtype=bool)
    kept = [q for q, z in zip(ids, zero) if not z]
    return m3b_compile.Population(name, kind, kept, idx[~zero], [g for g, z in zip(golds, zero) if not z], len(ids),
                                  int(zero.sum()), m3b_pools.ids_digest(kept))


def shared(su, i, T):
    """pool, read and compile for question i of the prepared lists; times into T."""
    p = su.prep
    d_ids, s_ids = np.asarray(p.dense_ids[i]), np.asarray(p.splade_ids[i])
    c0 = clock()
    pool, seeds = su.builder(d_ids, s_ids)
    pool, seeds = np.asarray(pool, np.int64), np.asarray(seeds, np.int64)
    c1 = clock()
    E = su.context.nodes.read(pool)
    c2 = clock()
    inp = T2.QueryInputs(p.qemb[i], p.dense_ids[i], p.dense_scores[i], p.splade_ids[i], p.splade_scores[i])
    comp = su.fc.compile(inp, pool, seeds, embeddings=E, timings=None, v2=True)
    c3 = clock()
    T["pool"], T["read"], T["compile"] = c1 - c0, c2 - c1, c3 - c2
    top = {int(d_ids[0]), int(s_ids[0])}
    bucket = np.asarray([0 if int(s) in top else 1 for s in seeds], dtype=np.int64)
    return {"pool": pool, "seeds": seeds, "bucket": bucket, "E": E, "comp": comp, "qemb": np.asarray(p.qemb[i])}


def edges_of(comp):
    """The compiled pool edges as the look keeps them (pack_queries_v2's order: family by family)."""
    us, vs, fs, fw, bw = [], [], [], [], []
    for f_i, fam in enumerate(ML8.FAMILIES):
        u, v, attr = comp.edges[fam]
        if u.size == 0:
            continue
        us.append(u.astype(np.int16))
        vs.append(v.astype(np.int16))
        fs.append(np.full(u.size, f_i, np.int8))
        fw.append((attr[:, ML8.COL_FWD - ML8.A0] > 0.5).astype(np.int8))
        bw.append((attr[:, ML8.COL_BWD - ML8.A0] > 0.5).astype(np.int8))
    cat = (lambda L, dt: np.concatenate(L) if L else np.zeros(0, dt))
    return cat(us, np.int16), cat(vs, np.int16), cat(fs, np.int8), cat(fw, np.int8), cat(bw, np.int8)


def lean_rows(su, sh, T):
    """zrc's / zsp's per-row inputs for one question (lean_cache's lines): edges, then the float16 rows."""
    c0 = clock()
    comp = sh["comp"]
    n = int(comp.pool.size)
    eu, ev, ef, efw, ebw = edges_of(comp)
    c1 = clock()
    En = np.asarray(sh["E"], dtype=np.float32)
    En = En / np.maximum(np.linalg.norm(En, axis=1, keepdims=True), 1e-12)
    proj = (En @ su.R).astype(np.float16)
    sl = np.full(ML8.MAX_SEEDS, -1, np.int64)
    sb = np.full(ML8.MAX_SEEDS, -1, np.int64)
    s_loc = np.asarray(comp.seeds_local, np.int64)
    sl[:s_loc.size], sb[:s_loc.size] = s_loc, sh["bucket"]
    x16 = np.asarray(comp.scalars[:, su.columns], dtype=np.float16)
    q_emb = np.asarray(sh["qemb"], np.float32)
    scratch = {b: 0.0 for b in LM.LEAN if b != "SEMB"}
    L, _q, _Pn = LM.lean_query(n, proj, q_emb, su.R, sl, sb, eu, ev, ef, efw, ebw, scratch)
    walkf = LC.walkf_of(n, proj, q_emb, su.R, sl, sb, eu, ev, ef, efw, ebw)
    codes = su.codes[sh["pool"]]
    seed, dists = LC.store_seed_dists(n, codes, su.store, sl, sb, x16[:, su.c_rrf_x], eu, ev, ef)
    X = np.empty((n, su.W), np.float16)
    X[:, :su.span[LC.XC_BLOCKS[-1]][1]] = x16[:, su.xcols]
    for b, arr in (("WALK", L["WALK"]), ("WALKF", walkf), ("SEED", seed), ("DISTS", dists)):
        a, e = su.span[b]
        X[:, a:e] = arr.astype(np.float16)
    c2 = clock()
    T["edges"], T["lean"] = c1 - c0, c2 - c1
    return {"n": n, "X": X, "codes": codes, "q_emb16": q_emb.astype(np.float16), "e": (eu, ev, ef, efw, ebw),
            "x16": x16, "proj": proj, "sl": sl, "sb": sb}


@torch.no_grad()
def forward(su, k, rows, B, links):
    m, bl = su.models[k]
    feats, nq, base_z, _g = rows.batch(np.arange(B), bl, links=links)
    keep = torch.ones((B, len(bl)), dtype=torch.float32)
    return m(feats, keep, nq, B, base_z)


def topk_per_q(s, n_np):
    off = np.r_[0, np.cumsum(n_np)]
    for j in range(n_np.size):
        torch.topk(s[off[j]:off[j + 1]], min(5, int(n_np[j])))


def run_path(su, k, sh, T):
    """One path's own stages for one question, cold; returns (scores, row dict or None)."""
    if k == "gnn6":
        c0 = clock()
        b = LT.fast_pack(sh["comp"], sh["E"], sh["qemb"], su.columns, True)
        c1 = clock()
        with torch.no_grad():
            sc = su.fgnn(b)
        torch.topk(sc, min(5, int(sh["pool"].size)))
        c2 = clock()
        T["pack"], T["forward"] = c1 - c0, c2 - c1
        return sc, None
    r = lean_rows(su, sh, T)
    c0 = clock()
    if k == "zsp":
        eu, ev, ef = r["e"][:3]
        r["links"] = links_of(r["n"], eu, ev, ef)
    rows = Rows([r], su.span, su.W, su.c_rrf, su.store)
    c1 = clock()
    sc = forward(su, k, rows, 1, links=(k == "zsp"))
    torch.topk(sc, min(5, r["n"]))
    c2 = clock()
    T["links" if k == "zsp" else "rows"], T["forward"] = c1 - c0, c2 - c1
    return sc, r


# ── the checks (untimed) ─────────────────────────────────────────────────────


def check_question(su, i, sh, r, scores, cache, qi_cache):
    ref = su.ref[i]
    c = {}
    c["pool"] = bool(np.array_equal(sh["pool"], ref["pool"]))
    if r is not None:
        c["seeds"] = bool(np.array_equal(r["sl"], ref["sl"]) and np.array_equal(r["sb"], ref["sb"]))
        c["edges"] = bool(all(np.array_equal(a, b) for a, b in zip(r["e"], ref["e"])))
        c["x"] = diff(r["x16"], ref["x"])
        c["proj"] = diff(r["proj"], ref["proj"])
        if cache is not None and qi_cache is not None:
            a, b = int(cache.off_np[qi_cache]), int(cache.off_np[qi_cache + 1])
            c["rows"] = diff(r["X"], cache.X[a:b].numpy())
            c["codes"] = int((cache.tab[cache.row[a:b]].numpy() != r["codes"]).sum())
    g = scores["gnn6"].numpy()
    c["gnn6_vs_look"] = float(np.abs(g - ref["gnn0"].astype(np.float32)).max()) if g.size else 0.0
    c["gnn6_top5_same"] = bool(np.array_equal(top5(g), top5(ref["gnn0"].astype(np.float32))))
    return c


def top5(s):
    return np.sort(np.argsort(-s, kind="stable")[:5])


@torch.no_grad()
def check_scores_on_cache(su, cache, qi, scores, ref_edges):
    """zrc and zsp on the step-1 cache's rows of question qi (zsp's links from the look's edges), through the same
    one-question carve, against the harness's scores."""
    a, b = int(cache.off_np[qi]), int(cache.off_np[qi + 1])
    eu, ev, ef = ref_edges
    r = {"n": b - a, "X": cache.X[a:b].numpy(), "codes": cache.tab[cache.row[a:b]].numpy(),
         "q_emb16": cache.q_emb[qi].numpy(), "links": links_of(b - a, eu, ev, ef)}
    rows = Rows([r], cache.span, cache.W, cache.c_rrf, su.store)
    out = {}
    for k in ("zrc", "zsp"):
        s = forward(su, k, rows, 1, links=(k == "zsp")).numpy()
        h = scores[k].numpy()
        out[f"{k}_vs_cache"] = float(np.abs(s - h).max()) if s.size else 0.0
        out[f"{k}_top5_same"] = bool(np.array_equal(top5(s), top5(h)))
    return out


# ── run ──────────────────────────────────────────────────────────────────────


def run(a):
    t_start = time.time()
    pinned = pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    torch.set_num_threads(THREADS)
    log(f"{a.dataset}: pin {pinned}")
    if a.dataset in TYPED:
        raise SystemExit(f"{a.dataset}: typed graphs come after the four untyped datasets (docs/C1_COLD_COST.md)")
    su = Setup(a.dataset, a.queries)
    nq = len(su.prep.pools) - 1
    log(f"{a.dataset}: {nq} measured questions, index {su.index}")
    cache = LG.CacheCarve(a.dataset, CARVE, su.basis, "cpu")      # untimed; the checks' reference
    if list(cache.ids[:nq]) != list(su.ids[:nq]) or cache.span != su.span or cache.W != su.W:
        raise SystemExit("the cache's questions or columns are not the look's")
    # numba's compile: the carve's next question, once, timed as index time
    t = time.time()
    T0 = {}
    sh0 = shared(su, nq, T0)
    for k in PATHS:
        run_path(su, k, sh0, {})
    su.index["jit_question_s"] = time.time() - t
    del sh0
    recs, checks = [], []
    group, group_sh = [], []
    b16 = []
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            T = {"shared": {}}
            sh = shared(su, i, T["shared"])
            order = PATHS[i % 3:] + PATHS[:i % 3]
            scores, rows_of = {}, {}
            for k in order:
                T[k] = {}
                scores[k], rows_of[k] = run_path(su, k, sh, T[k])
            T["order"] = list(order)
            T["n"] = int(sh["pool"].size)
            recs.append(T)
            c = check_question(su, i, sh, rows_of["zrc"], scores, cache, i if cache is not None else None)
            if cache is not None:
                c.update(check_scores_on_cache(su, cache, i, scores, su.ref[i]["e"][:3]))
            checks.append(c)
            group.append(rows_of["zsp"])
            group_sh.append(sh)
            if len(group) == B16 or i == nq - 1:
                b16.append(batch_group(su, group, group_sh))
                group, group_sh = [], []
            if (i + 1) % 25 == 0:
                gc.collect()
                log(f"  {i + 1}/{nq}")
    finally:
        gc.enable()
    rec = {"dataset": a.dataset, "carve": CARVE, "queries": nq, "declared_in": "docs/C1_COLD_COST.md",
           "threads": THREADS, "pin": pinned, "index": su.index, "per_question": recs, "batch16": b16,
           "checks": checks, "check_summary": summarize_checks(checks),
           "fits": {k: {"dir": str(FITS[k].relative_to(ROOT)), "models_sha256": sha_file(FITS[k] / "models.pt"),
                        "candidate": CAND} for k in FITS},
           "basis": su.basis, "basis_sha256": su.basis_sha256, "look_records": su.look_head["records"],
           "cache_checked": cache is not None, "peak_rss_bytes": LC.peak_rss(),
           "script_sha256": sha_file(__file__), "numpy": np.__version__, "torch": torch.__version__,
           "seconds": round(time.time() - t_start, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out or OUT / f"{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out, rec)
    log(f"{a.dataset}: done in {rec['seconds']}s; checks {rec['check_summary']}; -> {out}")


def batch_group(su, group, group_sh):
    """One forward per path over a group's questions (inputs built at batch 1)."""
    out = {"size": len(group)}
    for k in PATHS:
        if k == "gnn6":
            c0 = clock()
            qds = [{"pool": sh["comp"].pool, "x": sh["comp"].scalars[:, su.columns], "seedw": sh["comp"].seedw,
                    "qemb": sh["qemb"], "seeds": sh["comp"].seeds_local, "gold": np.zeros(0, np.int64),
                    "gold_total": 0, "emb": sh["E"]} for sh in group_sh]
            b = V2.pack_queries_v2(qds, su.context)
            c1 = clock()
            with torch.no_grad():
                sc = su.fgnn(b)
            topk_per_q(sc, np.asarray([int(sh["pool"].size) for sh in group_sh]))
            c2 = clock()
            out[k] = {"pack": c1 - c0, "forward": c2 - c1}
            continue
        c0 = clock()
        rows = Rows(group, su.span, su.W, su.c_rrf, su.store)
        c1 = clock()
        sc = forward(su, k, rows, len(group), links=(k == "zsp"))
        topk_per_q(sc, rows.n_np)
        c2 = clock()
        out[k] = {"rows": c1 - c0, "forward": c2 - c1}
    return out


def summarize_checks(checks):
    s = {"questions": len(checks)}
    for key in ("pool", "seeds", "edges", "gnn6_top5_same", "zrc_top5_same", "zsp_top5_same"):
        v = [c[key] for c in checks if key in c]
        if v:
            s[f"{key}_ok"] = int(sum(v))
    for key in ("x", "proj", "rows"):
        v = [c[key] for c in checks if key in c]
        if v:
            s[f"{key}_questions_differing"] = int(sum(1 for d in v if d[0] != 0))
            s[f"{key}_max_abs"] = max(d[1] for d in v)
    for key in ("gnn6_vs_look", "zrc_vs_cache", "zsp_vs_cache"):
        v = [c[key] for c in checks if key in c]
        if v:
            s[f"{key}_max"] = max(v)
    v = [c["codes"] for c in checks if "codes" in c]
    if v:
        s["codes_rows_differing"] = int(sum(v))
    return s


# ── report ───────────────────────────────────────────────────────────────────


def stage_totals(T):
    """Per path: total (with the read) and total without the read, in seconds."""
    sh = T["shared"]
    base = sh["pool"] + sh["read"] + sh["compile"]
    out = {}
    for k in PATHS:
        own = sum(T[k].values())
        out[k] = (base + own, base - sh["read"] + own)
    return out


def boot_ci(x, stat, rng, n=2000):
    x = np.asarray(x, float)
    v = [stat(x[rng.integers(0, x.size, x.size)]) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def report(a):
    rows, out = [], {}
    for f in sorted(OUT.glob("*.json")):
        if f.name.startswith("report"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        ds = r["dataset"]
        tot = [stage_totals(T) for T in r["per_question"]]
        good = [j for j, c in enumerate(r["checks"]) if c.get("pool") and c.get("gnn6_top5_same", True)
                and c.get("zrc_top5_same", True) and c.get("zsp_top5_same", True)]
        bad_share = 1 - len(good) / max(len(tot), 1)
        d = {"questions": len(tot), "kept": len(good), "failing_share": bad_share, "stopped": bad_share > 0.02}
        rng = np.random.default_rng(0)
        for w, wi in (("with_read", 0), ("without_read", 1)):
            for k in PATHS:
                x = np.asarray([tot[j][k][wi] for j in good]) * 1e3
                d[f"{k}_{w}_ms"] = {p: float(np.percentile(x, p)) for p in (50, 95, 99)}
                d[f"{k}_{w}_ms"]["p50_ci"] = boot_ci(x, lambda v: np.percentile(v, 50), rng)
            for k in ("zsp", "gnn6"):
                num = np.asarray([tot[j][k][wi] for j in good])
                den = np.asarray([tot[j]["zrc"][wi] for j in good])
                idx = np.arange(num.size)
                ratio = float(np.percentile(num, 50) / np.percentile(den, 50))
                ci = []
                for _ in range(2000):
                    s = rng.integers(0, idx.size, idx.size)
                    ci.append(np.percentile(num[s], 50) / np.percentile(den[s], 50))
                d[f"ratio_{k}_over_zrc_{w}_p50"] = {"ratio": ratio, "ci": [float(np.percentile(ci, 2.5)),
                                                                           float(np.percentile(ci, 97.5))]}
        st = {}
        for k in PATHS:
            parts = {}
            for j in good:
                T = r["per_question"][j]
                for s_, v in list(T["shared"].items()) + list(T[k].items()):
                    parts.setdefault(s_, []).append(v * 1e3)
            st[k] = {s_: float(np.percentile(v, 50)) for s_, v in parts.items()}
        d["stage_p50_ms"] = st
        b16 = {}
        for g in r["batch16"]:
            for k in PATHS:
                b16.setdefault(k, []).append(sum(g[k].values()) / g["size"])
        d["batch16_forward_per_q_ms_p50"] = {k: float(np.percentile(v, 50)) * 1e3 for k, v in b16.items()}
        d["index"] = r["index"]
        d["checks"] = r["check_summary"]
        d["pin"] = r["pin"]
        d["peak_rss_gb"] = r["peak_rss_bytes"] / 1e9 if r.get("peak_rss_bytes") else None
        out[ds] = d
    LC.write_json(OUT / "report.json", out)
    lines = ["| dataset | zrc p50 ms | zsp p50 ms | gnn6 p50 ms | zsp/zrc | gnn6/zrc | gnn6/zrc (no read) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for ds, d in out.items():
        f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
        lines.append(f"| {ds} | {d['zrc_with_read_ms'][50]:.1f} | {d['zsp_with_read_ms'][50]:.1f} | "
                     f"{d['gnn6_with_read_ms'][50]:.1f} | {f(d['ratio_zsp_over_zrc_with_read_p50'])} | "
                     f"{f(d['ratio_gnn6_over_zrc_with_read_p50'])} | {f(d['ratio_gnn6_over_zrc_without_read_p50'])} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    a = np.asarray([1.0, 2.0], np.float16)
    assert diff(a, a.copy()) == (0, 0.0)
    b = a.copy()
    b[1] = 2.5
    assert diff(a, b)[0] == 1 and abs(diff(a, b)[1] - 0.5) < 1e-6
    ne, u, v, f = links_of(4, np.asarray([0, 1, 1, 2], np.int16), np.asarray([1, 0, 1, 3], np.int16),
                           np.asarray([0, 0, 0, 1], np.int8))
    assert ne == 4 and set(zip(u.tolist(), v.tolist(), f.tolist())) == {(0, 1, 0), (1, 0, 0), (2, 3, 1), (3, 2, 1)}
    T = {"shared": {"pool": 1.0, "read": 2.0, "compile": 3.0}, "zrc": {"edges": 1.0, "lean": 1.0, "rows": 0.5, "forward": 0.5},
         "zsp": {"edges": 1.0, "lean": 1.0, "links": 0.5, "forward": 1.0}, "gnn6": {"pack": 1.0, "forward": 4.0}}
    tt = stage_totals(T)
    assert tt["zrc"] == (9.0, 7.0) and tt["gnn6"] == (11.0, 9.0) and tt["zsp"] == (9.5, 7.5)
    assert top5(np.asarray([0.1, 0.9, 0.5, 0.7, 0.3, 0.8], np.float32)).tolist() == [1, 2, 3, 4, 5]
    print("selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run", "report"))
    ap.add_argument("--dataset")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run(a)
    if a.cmd == "report":
        return report(a)
    ap.error("run, report or --selftest")


if __name__ == "__main__":
    main()
