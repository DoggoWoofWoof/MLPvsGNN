"""Design look (untracked; not a result and not filed): exact, faster pool-edge kernels, for every path alike.

lean_time2 (2wiki select, batch 1) puts the pool-edge stage at 4.4 to 6.1 ms p50 of the lean paths, and the same
kernels run inside the fast compile the twin and the GNN are served with. fast_features' kernels find a pool node's
in-pool neighbours by walking its whole stored row and probing the global -> local lookup (6 M int32 on 2wiki) for
every entry, so a query costs one random probe per stored entry of every pool row, hubs included.

Both stores are sorted per row (m3b_pools.FamilyStore.from_graph):
  structural  rows by (tail, rel), so a row's typed entries are in ascending neighbour id and the entries of one
              neighbour are one run;
  ner / knn   rows by (-weight, tail): stored order is not id order.
The pool is sorted ascending (FastCompiler.compile refuses anything else), so local index order is global id order.

typed_edges_gallop: a row longer than scan_max is merged with the pool instead of scanned: for each pool member in
ascending order, a galloping lower bound from the previous position finds its run. The in-pool runs come out in
stored order, so the first ``cap`` pairs, the entry order, the pair ids and every output array are the kernel's own.
A row of scan_max entries or fewer is scanned as before.

weighted_edges_sorted: an index-time copy of each row sorted by neighbour id (scol, with sidx = the entry's offset in
the stored row; 8 B a stored entry) lets a long row be merged the same way; the in-pool offsets are then sorted back
into stored order and the first ``cap`` kept, as the stored-order scan keeps them. The copy is built here for the rows
the timed pools touch and its cost extrapolated to the graph, as lean_time2 does for the node stores.

Both are serving forms of the same features (the check compares every output array, and the whole compile, exactly);
neither changes what any model reads.

    python outputs/mp_unified/edges_fast.py --selftest
    LEAN_TIME_THREADS=8 python outputs/mp_unified/edges_fast.py --dataset 2wiki --queries 300 --out scratch.json
"""
import os
import sys

THREADS = int(os.environ.get("LEAN_TIME_THREADS", "8"))   # lean_time2's laptop latency setting
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(THREADS)
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mp_retrieval import fast_features as FF  # noqa: E402
from mp_retrieval.m3b_pools import FamilyStore  # noqa: E402


# ── kernels ──────────────────────────────────────────────────────────────────


@njit(nogil=True)
def _lower_bound(a, lo, hi, x):
    """The first index in [lo, hi) with a[index] >= x (hi if none), galloping from lo: O(log of the distance)."""
    if lo >= hi or a[lo] >= x:
        return lo
    prev = lo
    step = 1
    cur = lo + 1
    while cur < hi and a[cur] < x:
        prev = cur
        step *= 2
        cur = prev + step
    if cur > hi:
        cur = hi
    left = prev + 1
    right = cur
    while left < right:
        mid = (left + right) >> 1
        if a[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left


@njit(nogil=True)
def typed_edges_gallop(tindptr, tcol, trel, tdir, pool, lookup, cap, scan_max):
    """fast_features._typed_edges, exactly, with rows longer than scan_max merged with the (ascending) pool."""
    n = pool.shape[0]
    n_ent = 0
    n_pair = 0
    for v in range(n):
        g = pool[v]
        lo = tindptr[g]
        hi = tindptr[g + 1]
        pairs_v = 0
        if hi - lo <= scan_max:
            prev = -1
            for idx in range(lo, hi):
                loc = lookup[tcol[idx]]
                if loc < 0:
                    continue
                if loc != prev:
                    pairs_v += 1
                    prev = loc
                if pairs_v > cap:
                    break
                n_ent += 1
        else:
            pos = lo
            for j in range(n):
                t = pool[j]
                pos = _lower_bound(tcol, pos, hi, t)
                if pos >= hi:
                    break
                if tcol[pos] != t:
                    continue
                pairs_v += 1
                if pairs_v > cap:
                    break
                e = pos
                while e < hi and tcol[e] == t:
                    e += 1
                n_ent += e - pos
                pos = e
        n_pair += min(pairs_v, cap)
    ev = np.empty(n_ent, np.int32)
    eu = np.empty(n_ent, np.int32)
    erel = np.empty(n_ent, np.int32)
    edir = np.empty(n_ent, np.int32)
    pair_id = np.empty(n_ent, np.int64)
    pu = np.empty(n_pair, np.int32)
    pv = np.empty(n_pair, np.int32)
    k = 0
    p = -1
    for v in range(n):
        g = pool[v]
        lo = tindptr[g]
        hi = tindptr[g + 1]
        pairs_v = 0
        if hi - lo <= scan_max:
            prev = -1
            for idx in range(lo, hi):
                loc = lookup[tcol[idx]]
                if loc < 0:
                    continue
                if loc != prev:
                    pairs_v += 1
                    prev = loc
                    if pairs_v > cap:
                        break
                    p += 1
                    pu[p] = loc
                    pv[p] = v
                ev[k] = v
                eu[k] = loc
                erel[k] = trel[idx]
                edir[k] = tdir[idx]
                pair_id[k] = p
                k += 1
        else:
            pos = lo
            for j in range(n):
                t = pool[j]
                pos = _lower_bound(tcol, pos, hi, t)
                if pos >= hi:
                    break
                if tcol[pos] != t:
                    continue
                pairs_v += 1
                if pairs_v > cap:
                    break
                p += 1
                pu[p] = j
                pv[p] = v
                while pos < hi and tcol[pos] == t:
                    ev[k] = v
                    eu[k] = j
                    erel[k] = trel[pos]
                    edir[k] = tdir[pos]
                    pair_id[k] = p
                    k += 1
                    pos += 1
    return ev, eu, erel, edir, pair_id, pu, pv


@njit(nogil=True)
def _row_hits(indptr, col, pool, lookup, g, scan_max, sstart, scol, sidx, n, buf):
    """The stored offsets (from the row start) of row g's in-pool entries, ascending, into buf; returns their count
    (buf grows by reallocation, returned with the count)."""
    lo = indptr[g]
    hi = indptr[g + 1]
    m = 0
    if hi - lo <= scan_max:
        for idx in range(lo, hi):
            if lookup[col[idx]] >= 0:
                if m == buf.shape[0]:
                    nb = np.empty(2 * buf.shape[0] + 8, np.int64)
                    nb[:m] = buf[:m]
                    buf = nb
                buf[m] = idx - lo
                m += 1
        return m, buf
    s0 = sstart[g]
    deg = hi - lo
    pos = s0
    end = s0 + deg
    for j in range(n):
        t = pool[j]
        pos = _lower_bound(scol, pos, end, t)
        if pos >= end:
            break
        while pos < end and scol[pos] == t:
            if m == buf.shape[0]:
                nb = np.empty(2 * buf.shape[0] + 8, np.int64)
                nb[:m] = buf[:m]
                buf = nb
            buf[m] = sidx[pos]
            m += 1
            pos += 1
    if m > 1:
        buf[:m] = np.sort(buf[:m])
    return m, buf


@njit(nogil=True)
def weighted_edges_sorted(indptr, col, wbits, lut, pool, lookup, cap, scan_max, sstart, scol, sidx):
    """fast_features._weighted_edges, exactly, with rows longer than scan_max merged through the row's id-sorted copy."""
    n = pool.shape[0]
    buf = np.empty(64, np.int64)
    kept = np.zeros(n, np.int64)
    total = 0
    if cap > 0:
        for v in range(n):
            m, buf = _row_hits(indptr, col, pool, lookup, pool[v], scan_max, sstart, scol, sidx, n, buf)
            kept[v] = min(m, cap)
            total += kept[v]
    u = np.empty(total, np.int32)
    vv = np.empty(total, np.int32)
    w = np.empty(total, np.float32)
    k = 0
    if cap > 0:
        for v in range(n):
            if kept[v] == 0:
                continue
            g = pool[v]
            lo = indptr[g]
            m, buf = _row_hits(indptr, col, pool, lookup, g, scan_max, sstart, scol, sidx, n, buf)
            for r in range(kept[v]):
                idx = lo + buf[r]
                u[k] = lookup[col[idx]]
                vv[k] = v
                w[k] = lut[wbits[idx]]
                k += 1
    return u, vv, w


@njit(nogil=True)
def build_sorted_rows(indptr, col, rows, sstart, scol, sidx):
    """The index-time id-sorted copy of each listed row: scol (neighbour ids ascending, stable) and sidx (stored offset),
    written contiguously; sstart[g] = the row's start in scol/sidx."""
    off = 0
    for r in range(rows.shape[0]):
        g = rows[r]
        lo = indptr[g]
        hi = indptr[g + 1]
        seg = col[lo:hi]
        o = np.argsort(seg, kind="mergesort")
        sstart[g] = off
        for k in range(hi - lo):
            scol[off + k] = seg[o[k]]
            sidx[off + k] = o[k]
        off += hi - lo
    return off


@njit(nogil=True)
def rows_unsorted(indptr, col, rows):
    """How many of the listed rows are not in ascending neighbour id (the structural kernel's precondition)."""
    bad = 0
    for r in range(rows.shape[0]):
        g = rows[r]
        for idx in range(indptr[g] + 1, indptr[g + 1]):
            if col[idx] < col[idx - 1]:
                bad += 1
                break
    return bad


def sorted_index(store_arrays, rows):
    """(sstart, scol, sidx, seconds, entries) for the listed rows of one weighted family."""
    indptr, col = store_arrays["indptr"], store_arrays["col"]
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    entries = int((indptr[rows + 1] - indptr[rows]).sum())
    sstart = np.full(indptr.shape[0] - 1, -1, dtype=np.int64)
    scol = np.empty(entries, dtype=col.dtype)
    sidx = np.empty(entries, dtype=np.int32)
    build_sorted_rows(indptr, col, rows[:2], sstart, scol, sidx)   # compile outside the clock
    sstart[:] = -1
    t = time.perf_counter()
    build_sorted_rows(indptr, col, rows, sstart, scol, sidx)
    return sstart, scol, sidx, time.perf_counter() - t, entries


# ── selftest ─────────────────────────────────────────────────────────────────


def _random_store(rng, n, m, family, hubs, weighted):
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    for h in hubs:                                  # hub rows, self-loops and repeated pairs
        k = m // 6
        src[:k] = h
        rng.shuffle(src)
    src[:5] = dst[:5]
    src[5:12] = src[12:19]
    dst[5:12] = dst[12:19]
    g = SimpleNamespace(n_nodes=n, src=src.astype(np.int32), dst=dst.astype(np.int32),
                        rel=rng.integers(0, 7, m).astype(np.int16),
                        weight=(rng.integers(0, 5, m).astype(np.float32) / 4 if weighted else None))   # many ties
    return FF.FastCompiler._store_arrays(FamilyStore.from_graph(g, family))


def selftest():
    rng = np.random.default_rng(0)
    KK = FF._kernels(False)
    lut = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    checks = 0
    for trial in range(40):
        n = int(rng.integers(20, 400))
        m = int(rng.integers(50, 4000))
        hubs = rng.integers(0, n, int(rng.integers(0, 4)))
        st = _random_store(rng, n, m, "structural", hubs, False)
        wt = _random_store(rng, n, m, "ner", hubs, True)
        rows_all = np.arange(n, dtype=np.int64)
        sstart, scol, sidx, _sec, _e = sorted_index(wt, rows_all)
        lookup = np.full(n, -1, np.int32)
        for _p in range(6):
            size = int(rng.integers(1, min(n, 150) + 1))
            pool = np.sort(rng.choice(n, size, replace=False)).astype(np.int64)
            if _p == 0 and hubs.size:
                pool = np.unique(np.concatenate([pool, hubs.astype(np.int64)]))
            lookup[pool] = np.arange(pool.size, dtype=np.int32)
            for cap in (1, 2, 5, 64, 10 ** 9):
                ref = KK.typed_edges(st["tindptr"], st["tcol"], st["trel"], st["tdir"], pool, lookup, cap)
                for scan_max in (-1, 0, 3, pool.size, 10 ** 12):
                    got = typed_edges_gallop(st["tindptr"], st["tcol"], st["trel"], st["tdir"], pool, lookup, cap, scan_max)
                    for a, b in zip(ref, got):
                        assert a.dtype == b.dtype and np.array_equal(a, b), ("typed", trial, cap, scan_max)
                    checks += 1
                ref = KK.weighted_edges(wt["indptr"], wt["col"], wt["wbits"], lut, pool, lookup, cap)
                for scan_max in (-1, 0, 3, pool.size, 10 ** 12):
                    got = weighted_edges_sorted(wt["indptr"], wt["col"], wt["wbits"], lut, pool, lookup, cap, scan_max,
                                                sstart, scol, sidx)
                    for a, b in zip(ref, got):
                        assert a.dtype == b.dtype and np.array_equal(a, b), ("weighted", trial, cap, scan_max)
                    checks += 1
            lookup[pool] = -1
    print(f"selftest: typed_edges_gallop and weighted_edges_sorted equal fast_features' kernels on every output array "
          f"({checks} cases: 40 random graphs with hubs, self-loops and repeated pairs, pools of 1-400 nodes, caps 1 to "
          f"unbounded, scan thresholds from merge-every-row to scan-every-row). all checks passed")


# ── the timing look ──────────────────────────────────────────────────────────


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=np.float64), q)) if len(v) else float("nan")


def same_compiled(a, b):
    if not np.array_equal(a.scalars, b.scalars, equal_nan=True):
        return False
    if not (np.array_equal(a.pool, b.pool) and np.array_equal(a.seeds_local, b.seeds_local)):
        return False
    for f in a.edges:
        for x, y in zip(a.edges[f], b.edges[f]):
            if not np.array_equal(x, y, equal_nan=True):
                return False
    return np.array_equal(a.seedw, b.seedw, equal_nan=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import lean_time as LT
    from mp_retrieval.m3b_features import QueryInputs
    S6, V2 = LT.S6, LT.V2
    t_start = time.time()
    name = a.dataset
    S6.pair_verify(S6.SIX)
    cfg, cfg_m3b, cfg_h = V2.load_configs()
    op = S6.pair_open(S6.SIX, cfg, cfg_m3b, name)
    m3b_compile = op.m3b_compile
    context, ds = op.contexts[name], op.handles[name]
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
    nq = len(prep.pools)
    fc = FF.compiler_for(context.stores, context.nodes, context.rel_table)
    if fc is None:
        raise SystemExit("no fast compiler for this context")
    LT.log(f"{name}: {nq} select queries prepared ({time.time() - t_start:.0f}s); pools mean "
           f"{np.mean([p.size for p in prep.pools]):.0f}; threads {THREADS}")
    touched = np.unique(np.concatenate(prep.pools))
    index = {"touched_rows": int(touched.size), "graph_rows": int(fc.n_nodes)}
    sidx_of = {}
    for fam in ("ner", "knn"):
        arr = fc._s[fam]
        sstart, scol, sidx, sec, entries = sorted_index(arr, touched)
        sidx_of[id(arr["col"])] = (sstart, scol, sidx)
        graph_entries = int(arr["col"].shape[0])
        index[fam] = {"touched_entries": entries, "seconds": sec, "graph_entries": graph_entries,
                      "graph_seconds_est": sec * graph_entries / max(1, entries), "bytes_per_entry": 8,
                      "graph_mib": graph_entries * 8 / 2 ** 20}
    LT.log("sorted ner/knn rows for the touched nodes: " + ", ".join(
        f"{f} {index[f]['touched_entries']} entries in {index[f]['seconds']:.2f}s (graph ~{index[f]['graph_seconds_est']:.0f}s, "
        f"{index[f]['graph_mib']:.0f} MiB)" for f in ("ner", "knn")))
    st = fc._s["structural"]
    unsorted = {"structural": int(rows_unsorted(st["tindptr"], st["tcol"], touched))}
    index["rows_unsorted_by_id"] = unsorted
    if unsorted["structural"]:
        raise SystemExit(f"{unsorted['structural']} touched structural rows are not in ascending neighbour id")
    KS = [fc.K] if fc._K_small is fc.K else [fc.K, fc._K_small]
    orig_k = {id(K): {"typed_edges": K.typed_edges, "weighted_edges": K.weighted_edges} for K in KS}

    def typed_new(tindptr, tcol, trel, tdir, pool, lookup, cap):
        return typed_edges_gallop(tindptr, tcol, trel, tdir, pool, lookup, cap, pool.shape[0])

    def weighted_new(indptr, col, wbits, lut, pool, lookup, cap):
        sstart, scol, sidx = sidx_of[id(col)]
        return weighted_edges_sorted(indptr, col, wbits, lut, pool, lookup, cap, pool.shape[0], sstart, scol, sidx)

    def use(new):
        for K in KS:
            K.typed_edges = typed_new if new else orig_k[id(K)]["typed_edges"]
            K.weighted_edges = weighted_new if new else orig_k[id(K)]["weighted_edges"]

    clock = time.perf_counter
    rec = {k: [] for k in ("typed_old", "typed_new", "typed_merge_all", "ner_old", "ner_new", "knn_old", "knn_new",
                           "compile_old", "compile_new", "edges_lap_old", "edges_lap_new")}
    stats = {k: [] for k in ("n", "deg_structural", "max_structural", "deg_ner", "max_ner", "deg_knn", "max_knn",
                             "rows_merged_structural", "rows_merged_ner", "rows_merged_knn")}
    off = {"typed": 0, "typed_merge_all": 0, "ner": 0, "knn": 0, "compile": 0}
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            pool = np.asarray(prep.pools[i], dtype=np.int64)
            seeds = np.asarray(prep.seeds[i], dtype=np.int64)
            n = int(pool.size)
            K = fc._K_small if n < FF.SERIAL_BELOW else fc.K
            inp = QueryInputs(prep.qemb[i], prep.dense_ids[i], prep.dense_scores[i], prep.splade_ids[i], prep.splade_scores[i])
            stats["n"].append(n)
            for fam, ptr in (("structural", st["tindptr"]), ("ner", fc._s["ner"]["indptr"]), ("knn", fc._s["knn"]["indptr"])):
                d = ptr[pool + 1] - ptr[pool]
                stats[f"deg_{fam}"].append(int(d.sum()))
                stats[f"max_{fam}"].append(int(d.max()))
                stats[f"rows_merged_{fam}"].append(int((d > n).sum()))
            lookup = fc._lookup
            lookup[pool] = np.arange(n, dtype=np.int32)
            try:
                flip = i % 2 == 1
                # structural
                res = {}
                for tag in (("new", "old") if flip else ("old", "new")):
                    c0 = clock()
                    if tag == "old":
                        res[tag] = orig_k[id(K)]["typed_edges"](st["tindptr"], st["tcol"], st["trel"], st["tdir"], pool, lookup, fc.cap)
                    else:
                        res[tag] = typed_edges_gallop(st["tindptr"], st["tcol"], st["trel"], st["tdir"], pool, lookup, fc.cap, n)
                    dt = clock() - c0
                    if i >= a.warm:
                        rec[f"typed_{tag}"].append(1e3 * dt)
                c0 = clock()
                r_all = typed_edges_gallop(st["tindptr"], st["tcol"], st["trel"], st["tdir"], pool, lookup, fc.cap, -1)
                if i >= a.warm:
                    rec["typed_merge_all"].append(1e3 * (clock() - c0))
                off["typed"] += int(not all(np.array_equal(x, y) for x, y in zip(res["old"], res["new"])))
                off["typed_merge_all"] += int(not all(np.array_equal(x, y) for x, y in zip(res["old"], r_all)))
                for fam in ("ner", "knn"):
                    arr = fc._s[fam]
                    sstart, scol, sidx = sidx_of[id(arr["col"])]
                    res = {}
                    for tag in (("new", "old") if flip else ("old", "new")):
                        c0 = clock()
                        if tag == "old":
                            res[tag] = orig_k[id(K)]["weighted_edges"](arr["indptr"], arr["col"], arr["wbits"], fc._lut16, pool,
                                                                      lookup, fc.cap)
                        else:
                            res[tag] = weighted_edges_sorted(arr["indptr"], arr["col"], arr["wbits"], fc._lut16, pool, lookup,
                                                             fc.cap, n, sstart, scol, sidx)
                        dt = clock() - c0
                        if i >= a.warm:
                            rec[f"{fam}_{tag}"].append(1e3 * dt)
                    off[fam] += int(not all(np.array_equal(x, y) for x, y in zip(res["old"], res["new"])))
            finally:
                lookup[pool] = -1
            # the whole fast compile (the twin's and the GNN's compile), old kernels against new
            comp = {}
            for tag in (("new", "old") if i % 2 == 1 else ("old", "new")):
                use(tag == "new")
                T = {}
                c0 = clock()
                comp[tag] = fc.compile(inp, pool, seeds, timings=T)
                dt = clock() - c0
                use(False)
                if i >= a.warm:
                    rec[f"compile_{tag}"].append(1e3 * dt)
                    rec[f"edges_lap_{tag}"].append(1e3 * T.get("edges", 0.0))
            off["compile"] += int(not same_compiled(comp["old"], comp["new"]))
            if i % 50 == 0:
                LT.log(f"  q{i}: n {n}; structural entries {stats['deg_structural'][-1]} (max row {stats['max_structural'][-1]}), "
                       f"ner {stats['deg_ner'][-1]}, knn {stats['deg_knn'][-1]}")
    finally:
        gc.enable()
        use(False)
    summ = {k: {"p50": pct(v, 50), "p95": pct(v, 95), "mean": float(np.mean(v))} for k, v in rec.items()}
    sstats = {k: {"p50": pct(v, 50), "p95": pct(v, 95), "max": float(np.max(v)), "mean": float(np.mean(v))} for k, v in stats.items()}
    res = {"look": "edges_fast", "dataset": name, "rows": nq, "warm_excluded": a.warm, "threads": THREADS,
           "script_sha256": sha(__file__), "fast_features_sha256": sha(FF.__file__), "ms": summ, "pool_stats": sstats,
           "rows_off": off, "index": index}
    LT.log(f"{name}: {nq - a.warm} warm queries, ms p50 / p95; threads {THREADS}")
    for k in rec:
        LT.log(f"  {k:16s} {summ[k]['p50']:8.3f} / {summ[k]['p95']:8.3f}   mean {summ[k]['mean']:8.3f}")
    LT.log("  pool rows p50/p95/max: " + ", ".join(f"{k} {v['p50']:.0f}/{v['p95']:.0f}/{v['max']:.0f}" for k, v in sstats.items()))
    LT.log(f"  rows off (must all be 0): {off}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        LT.log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
