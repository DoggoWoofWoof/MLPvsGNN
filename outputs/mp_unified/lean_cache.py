"""Step 1 feature cache (docs/STEP1_MATCHED_SELECTION.md, section 4): the arrays lean_mlp8's batches read for the step's
blocks, computed once per carve by the same per-query functions lean_mlp, lean_mlp2 and lean_mlp3 call, so the GPU
trainer (lean_gpu.py) reads them instead of rebuilding lean_mlp3.Carve3 in every fit.

    python outputs/mp_unified/lean_cache.py --make-basis 2wiki --host
    python outputs/mp_unified/lean_cache.py --dataset musique --carve s1eval --part 0/4 --host [--check-rows 64]
    python outputs/mp_unified/lean_cache.py --selftest

--make-basis G   lean_mlp3.fit_basis on G's served node rows (60,000 nodes, seed 0) with two BLAS threads, saved to
                 outputs/step1/cache/basis_G.npz (m, V, w as fit_basis returns them) and a record with its sha256.
--dataset/--carve/--part i/n
                 part i of n of a carve's look (outputs/mp_unified/look/<ds>/<carve>): the look's chunks, listed by its
                 records and sorted (lean_mlp.Carve's order), cut into n contiguous runs of whole chunks. A look whose
                 records do not list every chunk is refused. Per query of the part, in Carve's order:
                     xc      the look's float16 columns of rank, dense_cos, topo_STRUCT and depth_STRUCT (Carve.block)
                     walk    lean_mlp.lean_query's WALK, as Carve builds it, float16
                     walkf   lean_mlp2.new_query's WALKF lines (lean_query on every family as one structural multigraph,
                             and log1p of the hop-1 seed walks over NER and kNN edges), float16
                     seed_B  lean_mlp3.store_query's SEED lines on basis B's decoded store, float16
                     dists_B lean_mlp3.store_query's DISTS lines (lean_mlp2.pairs and dist_fast on the decoded store), float16
                     tab_B / row_B   the int8 store codes, store.codes(unit(nodes.read(chunk pool))) per chunk as Carve3 makes
                             them, as a table and a row index: one entry per node, plus one per row whose codes differ from
                             its node's first row (tab_B[row_B] equals the per-row codes; checked)
                     gold    the look's is_gold; score2 its stored six-pair scores twin0 and gnn0 (lean_mlp.SCORE_COL);
                             per query: n (pool size), gold_total, q_emb (float16)
                 written as .npy files under outputs/step1/cache/<ds>/<carve>/part_<i>of<n>/ with a record (shas of every
                 array, of the look records and of the bases).
--check-rows R   on the part holding the carve's first rows: lean_mlp3.Carve3 (limit R, built in this process on each
                 basis) against the cache, block by block, codes, gold, gold totals and query embeddings included, bit for
                 bit (float16 compared as bits). Any difference refuses the part.
The BLAS thread variables are set to 2 before numpy loads (run as a script, over rx's per-job value; imported, by
default), as lean_host's --threads 2 sets them for the lean runs: the store codes are a BLAS product.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    if __name__ == "__main__":
        os.environ[_v] = "2"            # rx sets these to the job's CPUs; the bases and caches are declared at two
    else:
        os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import lean_mlp as LM  # noqa: E402
import lean_mlp2 as L2  # noqa: E402
import lean_mlp3 as L3  # noqa: E402

OUT = ROOT / "outputs" / "step1" / "cache"
XC_BLOCKS = ("rank", "dense_cos", "topo_STRUCT", "depth_STRUCT")
FIT_NODES = 60000
CHUNK_KEYS = ("q_pool_size", "q_edges", "q_gold_total", "q_emb", "pool", "proj", "is_gold", "score", "x", "e_u", "e_v",
              "e_fam", "e_fwd", "e_bwd", "q_seed_local", "q_seed_bucket")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def peak_rss():
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_info()
        return int(getattr(mi, "peak_wset", 0) or getattr(mi, "rss", 0))
    except Exception:
        return None


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def save_npy(d, name, a):
    tmp = d / f"{name}.tmp.npy"
    np.save(tmp, np.ascontiguousarray(a))
    os.replace(tmp, d / f"{name}.npy")
    return sha_file(d / f"{name}.npy")


# ── bases ────────────────────────────────────────────────────────────────────


def basis_paths(graph, root=OUT):
    return Path(root) / f"basis_{graph}.npz", Path(root) / f"basis_{graph}.json"


def make_basis(graph, root=OUT, n_fit=FIT_NODES, placement=None):
    nodes, freeze = L3.open_nodes([graph])
    basis = L3.fit_basis(nodes[graph], n_fit)
    npz, js = basis_paths(graph, root)
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    np.savez(tmp, m=basis["m"], V=basis["V"], w=basis["w"])
    os.replace(tmp, npz)
    rec = {"graph": graph, "fit_nodes": n_fit, "rows": basis["rows"], "explained": basis["explained"],
           "seconds": basis["seconds"], "freeze_RECORD_SHA256": freeze, "npz": npz.name, "npz_sha256": sha_file(npz),
           "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "placement": placement or {"where": "laptop"},
           "script_sha256": sha_src(__file__), "lean_mlp3_sha256": sha_src(L3.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(js, rec)
    log(f"basis {graph}: {basis['rows']} nodes, {basis['explained']:.3f} of the variance ({basis['seconds']:.0f}s); "
        f"{npz} sha256 {rec['npz_sha256'][:12]}")
    return rec


def load_basis(graph, root=OUT):
    npz, js = basis_paths(graph, root)
    rec = json.loads(js.read_text(encoding="utf-8"))
    if sha_file(npz) != rec["npz_sha256"]:
        raise SystemExit(f"{npz}: sha256 is not its record's")
    z = np.load(npz)
    basis = {"m": z["m"], "V": z["V"], "w": z["w"], "rows": rec["rows"], "explained": rec["explained"], "from": graph}
    return basis, rec


# ── the per-query arrays (the frozen modules' lines) ─────────────────────────


def walkf_of(n, proj, q_emb, R, seeds, buckets, eu, ev, efam, efwd, ebwd):
    """lean_mlp2.new_query's WALKF lines."""
    valid = seeds >= 0
    S = seeds[valid].astype(np.int64)
    eu64, ev64 = eu.astype(np.int64), ev.astype(np.int64)
    scratch = {b: 0.0 for b in ("SEM", "SEED", "WALK", "NBR")}
    fam0 = np.zeros_like(efam)
    sym = efam > 0
    fw = np.where(sym, 1, efwd).astype(efwd.dtype)
    bw = np.where(sym, 1, ebwd).astype(ebwd.dtype)
    L, _q, _Pn = LM.lean_query(n, proj, q_emb, R, seeds, buckets, eu, ev, fam0, fw, bw, scratch)
    s0 = np.zeros(n)
    s0[S] = 1.0
    wx = np.zeros((n, 2), np.float32)
    for j, f in enumerate((1, 2)):
        m = efam == f
        wx[:, j] = np.log1p(np.bincount(ev64[m], weights=s0[eu64[m]], minlength=n))
    return np.concatenate([L["WALK"], wx], 1)


def store_seed_dists(n, codes, store, seeds, buckets, x_rrf, eu, ev, efam):
    """lean_mlp3.store_query's SEED and DISTS lines."""
    P = store.decode(codes)
    valid = seeds >= 0
    S, Bk = seeds[valid].astype(np.int64), buckets[valid]
    seed = np.zeros((n, 3), np.float32)
    if S.size:
        G = P @ P[S].T
        seed[:, 0] = G.mean(1)
        seed[:, 1] = G.max(1)
        if bool((Bk == 0).any()):
            seed[:, 2] = G[:, Bk == 0].max(1)
    eu64, ev64 = eu.astype(np.int64), ev.astype(np.int64)
    st = efam == 0
    rrf = x_rrf.astype(np.float64)
    s_seed = rrf[S] / max(float(rrf.max()), 1e-12) if S.size else np.zeros(0)
    u, v = L2.pairs(n, eu64[st], ev64[st])
    return seed, L2.dist_fast(n, P, S, s_seed, u, v)


def code_table(node, codes):
    """(tab, row): one entry per node (its first row's codes), plus one per row whose codes differ from them."""
    order = np.argsort(node, kind="stable")
    ns = node[order]
    start = np.r_[True, ns[1:] != ns[:-1]] if ns.size else np.zeros(0, bool)
    grp = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    cs = codes[order]
    same = (cs == cs[first[grp]]).all(1)
    tab = cs[first]
    at = grp.astype(np.int64)
    odd = np.flatnonzero(~same)
    if odd.size:
        tab = np.concatenate([tab, cs[odd]])
        at[odd] = first.size + np.arange(odd.size)
    row = np.empty_like(at)
    row[order] = at
    if not np.array_equal(tab[row], codes):
        raise SystemExit("the code table does not give back the per-row codes")
    return np.ascontiguousarray(tab), row.astype(np.int32), int(odd.size)


# ── the look's chunks ────────────────────────────────────────────────────────


def look_chunks(ds, carve, root=LM.LOOK):
    d = Path(root) / ds / carve
    recs = sorted(d.glob("record*.json"))
    if not recs:
        raise SystemExit(f"{d}: no look record")
    rr = [json.loads(r.read_text(encoding="utf-8")) for r in recs]
    listed = set()
    for r in rr:
        listed |= set(r["chunks"])
    n_chunks = int(rr[0]["n_chunks"])
    files = {int(p.stem[1:]): p for p in (d / "chunks").glob("c*.npz")}
    missing = sorted(set(range(n_chunks)) - (listed & set(files)))
    if missing:
        raise SystemExit(f"{d}: {len(missing)} of {n_chunks} chunks are not listed or not on disk (first {missing[:5]}); "
                         "the cache covers whole carves only")
    if any(int(r["n_chunks"]) != n_chunks or r["carve_ids_sha256"] != rr[0]["carve_ids_sha256"] or not r.get("full")
           or r.get("limit") is not None for r in rr):
        raise SystemExit(f"{d}: the records disagree, are not --full or carry a --limit")
    ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
    if len(ids) != int(rr[0]["carve_queries"]):
        raise SystemExit(f"{d}: ids.json has {len(ids)} ids, the records {rr[0]['carve_queries']}")
    head = {"records": {r_.name: sha_file(r_) for r_ in recs}, "carve_ids_sha256": rr[0]["carve_ids_sha256"],
            "carve_queries": int(rr[0]["carve_queries"]), "n_chunks": n_chunks, "chunk_queries": int(rr[0]["chunk_queries"]),
            "columns": rr[0]["columns"], "column_blocks": rr[0]["column_blocks"]}
    return d, [files[i] for i in range(n_chunks)], ids, head


def xc_columns(head):
    ci = {c: i for i, c in enumerate(head["columns"])}
    cols, spans, at = [], {}, 0
    for b in XC_BLOCKS:
        idx = [ci[c] for c in LM.SPLIT[b]] if b in LM.SPLIT else list(head["column_blocks"][b])
        cols += idx
        spans[b] = [at, at + len(idx)]
        at += len(idx)
    return np.asarray(cols, np.int64), spans, ci


# ── a part ───────────────────────────────────────────────────────────────────


def build_part(ds, carve, part, nparts, bases, stores, nodes, check_rows=64, root=LM.LOOK, out_root=OUT, placement=None):
    t0 = time.time()
    d, files, ids, head = look_chunks(ds, carve, root)
    K = len(files)
    lo, hi = part * K // nparts, (part + 1) * K // nparts
    mine = list(range(lo, hi))
    xcols, spans, ci = xc_columns(head)
    c_rrf = ci["rrf"]
    R = LM.projection()
    acc = {k: [] for k in ("xc", "walk", "walkf", "gold", "score2", "node")}
    for b in bases:
        acc[f"seed_{b}"], acc[f"dists_{b}"], acc[f"codes_{b}"] = [], [], []
    qn, qgt, qe = [], [], []
    rows_before = 0
    for c in range(lo):                                   # rows of the chunks before this part (for the row range)
        rows_before += int(np.load(files[c])["q_pool_size"].sum())
    q_before = sum(int(np.load(files[c])["q_pool_size"].size) for c in range(lo))
    gather_s = 0.0
    for k, ch in enumerate(mine):
        z = np.load(files[ch])
        A = {key: z[key] for key in CHUNK_KEYS}
        qps, qed = A["q_pool_size"], A["q_edges"]
        no = np.concatenate([[0], np.cumsum(qps)])
        eo = np.concatenate([[0], np.cumsum(qed)])
        t = time.time()
        E = L3.unit(np.asarray(nodes.read(A["pool"].astype(np.int64)), np.float32))
        gather_s += time.time() - t
        codes = {b: stores[b].codes(E) for b in bases}
        del E
        for i in range(qps.size):
            a, b_, ea, eb = no[i], no[i + 1], eo[i], eo[i + 1]
            n = int(qps[i])
            X = A["x"][a:b_]
            acc["xc"].append(X[:, xcols])
            scratch = {b: 0.0 for b in LM.LEAN if b != "SEMB"}
            L, _q, _Pn = LM.lean_query(n, A["proj"][a:b_], A["q_emb"][i], R, A["q_seed_local"][i], A["q_seed_bucket"][i],
                                       A["e_u"][ea:eb], A["e_v"][ea:eb], A["e_fam"][ea:eb], A["e_fwd"][ea:eb],
                                       A["e_bwd"][ea:eb], scratch)
            acc["walk"].append(L["WALK"].astype(np.float16))
            acc["walkf"].append(walkf_of(n, A["proj"][a:b_], A["q_emb"][i], R, A["q_seed_local"][i], A["q_seed_bucket"][i],
                                         A["e_u"][ea:eb], A["e_v"][ea:eb], A["e_fam"][ea:eb], A["e_fwd"][ea:eb],
                                         A["e_bwd"][ea:eb]).astype(np.float16))
            for b in bases:
                seed, dists = store_seed_dists(n, codes[b][a:b_], stores[b], A["q_seed_local"][i], A["q_seed_bucket"][i],
                                               X[:, c_rrf], A["e_u"][ea:eb], A["e_v"][ea:eb], A["e_fam"][ea:eb])
                acc[f"seed_{b}"].append(seed.astype(np.float16))
                acc[f"dists_{b}"].append(dists.astype(np.float16))
                acc[f"codes_{b}"].append(codes[b][a:b_])
            acc["gold"].append(A["is_gold"][a:b_])
            acc["score2"].append(A["score"][a:b_][:, [LM.SCORE_COL["twin0"], LM.SCORE_COL["gnn0"]]])
            acc["node"].append(A["pool"][a:b_].astype(np.int64))
            qn.append(n)
            qgt.append(int(A["q_gold_total"][i]))
            qe.append(A["q_emb"][i].astype(np.float16))
        if (k + 1) % max(1, len(mine) // 10) == 0 or k + 1 == len(mine):
            log(f"  {ds}/{carve} part {part}/{nparts}: chunk {k + 1}/{len(mine)}, {len(qn)} queries "
                f"({time.time() - t0:.0f}s, row gather {gather_s:.0f}s)")
    arrays = {k: np.concatenate(v) for k, v in acc.items() if k not in ("node",) and not k.startswith("codes_")}
    node = np.concatenate(acc["node"])
    odd = {}
    for b in bases:
        codes_rows = np.concatenate(acc[f"codes_{b}"])
        arrays[f"tab_{b}"], arrays[f"row_{b}"], odd[b] = code_table(node, codes_rows)
        del codes_rows
    arrays["node"] = node.astype(np.int32)
    arrays["n"] = np.asarray(qn, np.int32)
    arrays["gold_total"] = np.asarray(qgt, np.int32)
    arrays["q_emb"] = np.stack(qe) if qe else np.zeros((0, 1536), np.float16)
    if int(arrays["n"].sum()) != arrays["xc"].shape[0]:
        raise SystemExit("row count mismatch")
    check = None
    if check_rows and q_before == 0 and len(qn):
        check = check_part(ds, carve, arrays, spans, bases, stores, nodes, min(check_rows, len(qn)), root)
    out = Path(out_root) / ds / carve / f"part_{part}of{nparts}"
    out.mkdir(parents=True, exist_ok=True)
    shas = {k: save_npy(out, k, v) for k, v in sorted(arrays.items())}
    rec = {"dataset": ds, "carve": carve, "part": part, "nparts": nparts, "chunks": [lo, hi], "queries": len(qn),
           "query_range": [q_before, q_before + len(qn)], "rows": int(arrays["n"].sum()), "row_range": [rows_before, rows_before + int(arrays["n"].sum())],
           "ids": ids[q_before:q_before + len(qn)], "look": head, "xc_spans": spans, "bases": {b: stores[b].rec for b in bases},
           "odd_code_rows": odd, "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": shas[k]}
                                           for k, v in arrays.items()},
           "check": check, "numba": L2.njit is not None, "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
           "placement": placement or {"where": "laptop"}, "seconds": time.time() - t0, "row_gather_s": gather_s,
           "peak_rss_bytes": peak_rss(), "script_sha256": sha_src(__file__),
           "module_sha256": {m.__name__: sha_src(m.__file__) for m in (LM, L2, L3)},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out / "record.json", rec)
    log(f"{ds}/{carve} part {part}/{nparts}: {len(qn)} queries, {rec['rows']} rows in {rec['seconds']:.0f}s "
        f"(odd code rows {odd}; check {'-' if check is None else check['verdict']}); {out}")
    return rec


def bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint16) if a.dtype == np.float16 else a


def check_part(ds, carve, arrays, spans, bases, stores, nodes, nq, root):
    """lean_mlp3.Carve3 (limit nq) against the cache's first nq queries, bit for bit."""
    rows = int(arrays["n"][:nq].sum())
    bad, done = [], []
    for b in bases:
        c3 = L3.Carve3(ds, carve, stores[b], nodes, nq, root)
        pairs = [("n", c3.n, arrays["n"][:nq].astype(np.int64)), ("gold_total", c3.gold_total, arrays["gold_total"][:nq]),
                 ("q_emb", c3.q_emb, arrays["q_emb"][:nq]), ("gold", c3.gold, arrays["gold"][:rows]),
                 ("score2", c3.score[:, [LM.SCORE_COL["twin0"], LM.SCORE_COL["gnn0"]]], arrays["score2"][:rows])]
        for blk in XC_BLOCKS:
            s0, s1 = spans[blk]
            pairs.append((blk, c3.block(blk, np.arange(rows)), arrays["xc"][:rows, s0:s1]))
        pairs += [("WALK", c3.block("WALK", np.arange(rows)), arrays["walk"][:rows]),
                  ("WALKF", c3.block("WALKF", np.arange(rows)), arrays["walkf"][:rows]),
                  ("SEED", c3.block("SEED", np.arange(rows)), arrays[f"seed_{b}"][:rows]),
                  ("DISTS", c3.block("DISTS", np.arange(rows)), arrays[f"dists_{b}"][:rows]),
                  ("codes", c3.proj.codes, arrays[f"tab_{b}"][arrays[f"row_{b}"][:rows]]),
                  ("decoded", c3.proj[np.arange(rows)], stores[b].decode(arrays[f"tab_{b}"][arrays[f"row_{b}"][:rows]]))]
        for name, ref, got in pairs:
            ref, got = np.asarray(ref), np.asarray(got)
            if name in ("n", "gold_total"):
                same = ref.shape == got.shape and np.array_equal(ref.astype(np.int64), got.astype(np.int64))
            else:
                same = ref.shape == got.shape and ref.dtype == got.dtype and np.array_equal(bits(ref), bits(got))
            (done if same else bad).append(f"{b}:{name}")
        del c3
    verdict = "IDENTICAL" if not bad else "DIFFERENT"
    log(f"  check against Carve3 on the first {nq} queries ({rows} rows): {verdict} {bad or ''}")
    if bad:
        raise SystemExit(f"{ds}/{carve}: the cache differs from lean_mlp3.Carve3 on {bad}")
    return {"queries": nq, "rows": rows, "verdict": verdict, "compared": done}


# ── loading (the trainer's side) ─────────────────────────────────────────────


def part_dirs(ds, carve, out_root=OUT):
    d = Path(out_root) / ds / carve
    parts = sorted(d.glob("part_*of*"), key=lambda p: int(p.name.split("_")[1].split("of")[0]))
    if not parts:
        raise SystemExit(f"{d}: no cache part")
    nparts = {int(p.name.split("of")[1]) for p in parts}
    if len(nparts) != 1 or len(parts) != nparts.pop():
        raise SystemExit(f"{d}: parts {[p.name for p in parts]} do not make one whole split")
    recs = [json.loads((p / "record.json").read_text(encoding="utf-8")) for p in parts]
    q = 0
    for r in recs:
        if r["query_range"][0] != q:
            raise SystemExit(f"{d}: the parts do not tile the carve's queries")
        q = r["query_range"][1]
    if q != recs[0]["look"]["carve_queries"]:
        raise SystemExit(f"{d}: the parts hold {q} of {recs[0]['look']['carve_queries']} queries")
    return parts, recs


def load_array(p, name, rec, verify=True):
    f = Path(p) / f"{name}.npy"
    if verify and sha_file(f) != rec["arrays"][name]["sha256"]:
        raise SystemExit(f"{f}: sha256 is not its record's")
    return np.load(f, mmap_mode="r")


# ── main ─────────────────────────────────────────────────────────────────────


def open_stores(bases, root=OUT):
    stores = {}
    for b in bases:
        basis, rec = load_basis(b, root)
        s = L3.Store(basis)
        s.rec = {"npz_sha256": rec["npz_sha256"], "rows": rec["rows"], "explained": rec["explained"]}
        stores[b] = s
    return stores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-basis")
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--bases", default="2wiki,hotpotqa")
    ap.add_argument("--check-rows", type=int, default=64)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--look-root", default=None, help="look root (default lean_mlp.LOOK)")
    ap.add_argument("--out-root", default=None, help="cache root (default outputs/step1/cache)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    placement = None
    if a.host:
        import lean_host as LH
        placement = LH.substitute()
    out_root = Path(a.out_root) if a.out_root else OUT
    if a.make_basis:
        make_basis(a.make_basis, out_root, placement=placement)
        return 0
    if not (a.dataset and a.carve):
        raise SystemExit("--dataset and --carve, or --make-basis")
    part, nparts = (int(x) for x in a.part.split("/"))
    if not 0 <= part < nparts:
        raise SystemExit("--part i/n with 0 <= i < n")
    bases = [b for b in a.bases.split(",") if b]
    stores = open_stores(bases, out_root)
    nodes, freeze = L3.open_nodes([a.dataset])
    root = Path(a.look_root) if a.look_root else LM.LOOK
    log(f"cache {a.dataset}/{a.carve} part {part}/{nparts}, bases {bases}, numba {L2.njit is not None}, BLAS threads "
        f"{os.environ.get('OPENBLAS_NUM_THREADS')}, freeze {freeze[:12]}")
    build_part(a.dataset, a.carve, part, nparts, bases, stores, nodes[a.dataset], a.check_rows, root, out_root, placement)
    return 0


def selftest():
    rng = np.random.default_rng(0)
    node = rng.integers(0, 50, 400)
    base = rng.integers(-127, 128, (50, 8)).astype(np.int8)
    codes = base[node].copy()
    codes[[3, 77]] += 1                        # two rows whose codes differ from their node's other rows
    tab, row, odd = code_table(node, codes)
    assert np.array_equal(tab[row], codes) and odd >= 1 and tab.shape[0] == np.unique(node).size + odd
    tab2, row2, odd2 = code_table(node, base[node])
    assert odd2 == 0 and tab2.shape[0] == np.unique(node).size and np.array_equal(tab2[row2], base[node])
    h = np.asarray([np.nan, 1.0, -0.0], np.float16)
    assert np.array_equal(bits(h), bits(h.copy())) and not np.array_equal(bits(h), bits(np.asarray([np.nan, 1.0, 0.0], np.float16)))
    print("selftest: the code table gives back per-row codes (odd rows kept), float16 compared as bits. all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
