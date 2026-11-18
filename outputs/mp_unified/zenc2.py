"""ENC E2 (docs/ENC_ENCODER_COMPARISON.md section 12): the arm's inputs for the models, on the small and hard datasets.

  python outputs/mp_unified/zenc2.py first --enc mpnet --ds squad [--threads 8]      # the arm's dense top-1000, every query
  python outputs/mp_unified/zenc2.py knn   --enc mpnet --ds squad [--host] [--threads 8]   # the arm's K (SQuAD; MuSiQue and
                                                                                    # MetaQA reuse E1b part 2's K files)
  python outputs/mp_unified/zenc2.py selftest

first: the record's dense first stage (exact float32 inner product, top 1,000, ties to the lower id: zenc.search,
unchanged) over EVERY query of the dataset, stored as the package's own cache is (ids int32, scores float16), so a view
can serve a complete cache to any step. Gate GE0: on E0's s1eval rows the lists equal E0's (zenc first's) up to the
float order of a different GEMM blocking: GE0_ROWS of the rows with equal 1,000-id lists, every top-5 set equal on
GE0_TOP5 of the rows, scores within GE0_SCORE.

knn: the record's kNN rule, zenc1's pieces unchanged (zenc1b2's cmd_knn, widened to SQuAD). GK on the frozen arm: the
rebuilt edge set equals the package's graph/knn.npz.
Outputs only under outputs/enc/e2/.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import zenc as E  # noqa: E402  (first: sets the BLAS threads from --threads before numpy loads)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 12"
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "enc" / "e2"
E1B2 = ROOT / "outputs" / "enc" / "e1b2"
CARRY = ("mpnet", "qwen3e06")                            # section 12.1's arms
SMALL = ("squad", "musique", "metaqa")                   # the small and hard datasets (section 12.8)
GE0_ROWS = 0.999
GE0_TOP5 = 0.9995
GE0_SCORE = 2e-3
GK_W_TOL = 1e-3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zenc2: {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZENC2: {msg}")


def sha_norm(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_shas(*mods):
    return {"code_sha256": {Path(f).name: sha_norm(f) for f in (__file__, E.__file__) + tuple(m.__file__ for m in mods)}}


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def first_path(enc, d):
    return OUT / "first" / enc / f"{d}.npz"


def knn_path(enc, d):
    """SQuAD's K is this file's; MuSiQue's and MetaQA's are E1b part 2's (GK passed there on the frozen arm)."""
    return (OUT if d == "squad" else E1B2) / "knn" / enc / f"{d}.npz"


def save_npz(p, **arrs):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrs)
    os.replace(tmp, p)
    return E.sha256(p)


# ── first: the arm's dense top-1000 over every query ─────────────────────────


def all_queries(root, enc, d):
    paths = E.vec_shards(root, enc, d, "queries")
    Q = np.concatenate([np.asarray(np.load(p, mmap_mode="r"), np.float32) for p in paths])
    nq = int(E.manifest(root, d, "queries")["n_rows"])
    if Q.shape != (nq, E.ENCODERS[enc]["dim"]):
        stop(f"{enc} {d}: query vectors {Q.shape}, the record has {nq} queries")
    return Q


def ge0(ids, sc, e0p):
    """GE0 against E0's lists on its s1eval rows (ids int32, scores float16 on both sides)."""
    with np.load(e0p, allow_pickle=False) as z:
        rows = z["rows"].astype(np.int64)
        e_ids, e_sc = z["top_ids"].astype(np.int64), z["top_scores"].astype(np.float32)
    a_ids, a_sc = ids[rows].astype(np.int64), sc[rows].astype(np.float32)
    same = np.all(a_ids == e_ids, axis=1)
    t5 = np.mean([set(a_ids[i, :5].tolist()) == set(e_ids[i, :5].tolist()) for i in range(rows.size)])
    dif = float(np.abs(a_sc - e_sc).max())
    g = {"rows": int(rows.size), "lists_equal_share": round(float(same.mean()), 6), "top5_set_equal_share": round(float(t5), 6),
         "score_max_abs_diff": round(dif, 6), "e0": {"file": rel(e0p), "sha256": E.sha256(e0p)}}
    g["verdict"] = "PASS" if (g["lists_equal_share"] >= GE0_ROWS and g["top5_set_equal_share"] >= GE0_TOP5
                              and dif <= GE0_SCORE) else "FAIL"
    return g


def cmd_first(a):
    d, enc, t0 = a.ds, a.enc, time.time()
    if d not in SMALL or enc not in CARRY:
        stop(f"first runs on {SMALL} and {CARRY}")
    root = E.served_root()
    Q = all_queries(root, enc, d)
    log(f"{d} {enc}: {Q.shape[0]:,} queries x {Q.shape[1]} dims")
    t1 = time.time()
    ids, sc, _ = E.search(Q, E.vec_shards(root, enc, d, "docs"), E.K)
    secs = round(time.time() - t1, 1)
    del Q
    ids, sc = ids.astype(np.int32), sc.astype(np.float16)
    g = ge0(ids, sc, E.OUT / "e0" / enc / f"{d}.npz")
    log(f"{d} {enc}: GE0 {g['verdict']}: {json.dumps(g)}")
    p = first_path(enc, d)
    sha = save_npz(p, ids=ids, scores=sc)
    rec = {"stage": "ENC E2: first", "declared_in": DECLARED, "dataset": d, "enc": enc, "queries": int(ids.shape[0]),
           "k": E.K, "threads": E.THREADS, "search_seconds": secs, "npz_sha256": sha, "ge0": g,
           "vectors": {"docs_manifest_sha256": E.sha256(E.vec_dir(root, enc, d, "docs") / "manifest.json"),
                       "queries_manifest_sha256": E.sha256(E.vec_dir(root, enc, d, "queries") / "manifest.json"),
                       "revision": E.pinned(enc)},
           "seconds": round(time.time() - t0, 1), **code_shas()}
    E.wjson(p.with_suffix(".json"), rec)
    log(f"{d} {enc}: {ids.shape} -> {rel(p)} ({rec['seconds']} s)")
    return 0 if g["verdict"] == "PASS" else 3


# ── knn: the arm's K on SQuAD (zenc1b2's cmd_knn, widened) ───────────────────


def cmd_knn(a):
    import zenc1 as Z1                                   # the K rule's pieces, unchanged
    import zenc1b2 as B2                                 # k_edges and scores_at, unchanged
    d, enc, t0 = a.ds, a.enc, time.time()
    if d != "squad" or enc not in ("frozen",) + CARRY:
        stop("knn here builds SQuAD's K (MuSiQue's and MetaQA's are E1b part 2's)")
    served, freeze = Z1.package(a.host)
    n, nsha = Z1.dataset_nodes(served, d)
    pos2g, g_first, gsha = Z1.text_groups(served, d, n, nsha)
    paths = E.vec_shards(served, enc, d, "docs")
    offs = Z1.shard_offsets(paths)
    if int(offs[-1]) != n:
        stop(f"{enc} {d}: {int(offs[-1])} vector rows for {n} nodes")
    eg = np.arange(g_first.size, dtype=np.int64)
    log(f"{d} {enc}: {n:,} nodes in {g_first.size:,} text groups; {len(paths)} shards")
    t1 = time.time()
    nbr, sc = Z1.nn_top(paths, offs, pos2g, g_first, eg)
    secs = round(time.time() - t1, 1)
    outp = Z1.out_positions(eg, nbr, g_first)
    osc = B2.scores_at(eg, nbr, sc, g_first, Z1.SEARCH_K)
    src, dst, w = B2.k_edges(pos2g, outp, osc)
    p = knn_path(enc, d)
    sha = save_npz(p, src=src, dst=dst, weight=w)
    rec = {"stage": "ENC E2: K", "declared_in": DECLARED, "dataset": d, "enc": enc, "nodes": n, "nodes_sha256": nsha,
           "groups": int(g_first.size), "groups_sha256": gsha, "freeze_RECORD_SHA256": freeze, "edges": int(src.size),
           "tie_rows": int(Z1.tie_rows(sc).sum()), "threads": E.THREADS, "search_seconds": secs, "npz_sha256": sha,
           "vectors": {"record_sha256": E.RECORD_SHA} if enc == "frozen" else {
               "manifest_sha256": E.sha256(E.vec_dir(served, enc, d, "docs") / "manifest.json"), "revision": E.pinned(enc)},
           **code_shas(Z1, B2)}
    if enc == "frozen":
        with np.load(served / d / "graph" / "knn.npz") as z:
            rs, rd, rw = (z["src"].astype(np.int64), z["dst"].astype(np.int64), z["weight"].astype(np.float32))
        rk = np.minimum(rs, rd) * n + np.maximum(rs, rd)
        mk = src.astype(np.int64) * n + dst.astype(np.int64)
        common, i_m, i_r = np.intersect1d(mk, rk, return_indices=True)
        wdiff = float(np.abs(w[i_m] - rw[i_r]).max()) if common.size else None
        gk = {"record_edges": int(rk.size), "record_unique_pairs": int(np.unique(rk).size), "rebuilt_edges": int(mk.size),
              "common": int(common.size), "rebuilt_only": int(mk.size - common.size),
              "record_only": int(np.unique(rk).size - common.size), "weight_max_abs_diff": wdiff}
        gk["verdict"] = "PASS" if (gk["rebuilt_only"] == 0 and gk["record_only"] == 0
                                   and wdiff is not None and wdiff <= GK_W_TOL) else "FAIL"
        rec["gk"] = gk
        log(f"{d}: GK {gk['verdict']}: {json.dumps(gk)}")
    rec["seconds"] = round(time.time() - t0, 1)
    E.wjson(p.with_suffix(".json"), rec)
    log(f"{d} {enc}: {src.size:,} edges -> {rel(p)} ({rec['seconds']} s)")
    return 0 if rec.get("gk", {}).get("verdict", "PASS") == "PASS" else 3


# ── selftest ─────────────────────────────────────────────────────────────────


def cmd_selftest(_a):
    rng = np.random.default_rng(0)
    nq, n, dim = 300, 2500, 16
    Q = rng.standard_normal((nq, dim)).astype(np.float32)
    D = rng.standard_normal((n, dim)).astype(np.float32)
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        ps = []
        for s, (lo, hi) in enumerate(((0, 1000), (1000, 2500))):
            p = Path(t) / f"shard_{s:05d}.npy"
            np.save(p, D[lo:hi])
            ps.append(p)
        ids, sc, _ = E.search(Q, ps, 1000)
        S = Q @ D.T
        for q in range(0, nq, 37):
            o = np.lexsort((np.arange(n), -S[q]))[:1000]
            if not np.array_equal(o, ids[q]):
                stop(f"selftest: search row {q} is not brute force")
        ids32, sc16 = ids.astype(np.int32), sc.astype(np.float16)
        rows = np.arange(0, nq, 3)
        e0p = Path(t) / "e0.npz"
        np.savez(e0p, rows=rows, top_ids=ids32[rows], top_scores=sc16[rows])
        g = ge0(ids32, sc16, e0p)
        if g["verdict"] != "PASS" or g["lists_equal_share"] != 1.0:
            stop(f"selftest: GE0 on its own lists: {g}")
        bad = ids32.copy()
        bad[rows[:5], 0], bad[rows[:5], 1] = ids32[rows[:5], 1], ids32[rows[:5], 0]
        g = ge0(bad, sc16, e0p)
        if g["verdict"] != "FAIL":
            stop("selftest: GE0 passed swapped lists")
    log("selftest PASS (search = brute force on 9 rows; GE0 passes itself, fails swapped lists)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("first", "knn"):
        s = sub.add_parser(name)
        s.add_argument("--enc", required=True)
        s.add_argument("--ds", required=True)
        s.add_argument("--host", action="store_true")
        s.add_argument("--threads", default="8")
    s = sub.add_parser("selftest")
    s.add_argument("--threads", default="8")
    a = ap.parse_args(argv)
    return {"first": cmd_first, "knn": cmd_knn, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
