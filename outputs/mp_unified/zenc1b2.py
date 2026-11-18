"""ENC E1b part 2 (docs/ENC_ENCODER_COMPARISON.md section 11): each encoder's real pools under PKO's rule.

  python outputs/mp_unified/zenc1b2.py knn --enc mpnet --ds musique [--host] [--threads 8]   # the arm's K over the whole corpus
  python outputs/mp_unified/zenc1b2.py selftest

Step 1 here is the K family: the record's kNN rule (zenc1's, unchanged: exact fp32 top 4 with self over the text groups,
self dropped, each position inheriting its group's neighbours at their lowest positions, undirected) over EVERY group, not
only E1a's slot ends. GK: on the frozen arm the rebuilt edge set must equal the package's graph/knn.npz.
Outputs only under outputs/enc/e1b2/.
"""
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

import zenc1 as Z1  # noqa: E402  (the K rule's pieces, unchanged)

DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 11"
ROOT = HERE.parents[1]
OUT2 = ROOT / "outputs" / "enc" / "e1b2"
ARMS = ("frozen",) + tuple(E.NEW)
SETS = ("musique", "metaqa")
GK_W_TOL = 1e-3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZENC1B2: {msg}")


def sha_norm(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_shas():
    return {"code_sha256": {Path(f).name: sha_norm(f) for f in (__file__, E.__file__, Z1.__file__)}}


# ── the K family over the whole corpus ───────────────────────────────────────


def k_edges(pos2g, outp, osc):
    """Undirected edges (src < dst, sorted) and weights from each group's neighbour positions (G, 4) and scores (G, 4):
    every position inherits its group's neighbours. A pair seen from both ends keeps the larger score."""
    n = pos2g.size
    P = outp[pos2g]
    S = osc[pos2g]
    x = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], P.shape)
    v = P >= 0
    a, b, w = np.minimum(x, P)[v], np.maximum(x, P)[v], S[v]
    if (a == b).any():
        stop("a self edge survived")
    key = a * n + b
    o = np.lexsort((-w, key))
    key, w = key[o], w[o]
    first = np.r_[True, key[1:] != key[:-1]]
    key, w = key[first], w[first]
    return (key // n).astype(np.int32), (key % n).astype(np.int32), w.astype(np.float32)


def scores_at(eg, nbr, sc, g_first):
    """The scores aligned with zenc1.out_positions' columns (the top 4), -inf where the entry is dropped."""
    top = nbr[:, :Z1.SEARCH_K]
    ok = (top != eg[:, None]) & (top < g_first.size)
    return np.where(ok, sc[:, :Z1.SEARCH_K], -np.inf).astype(np.float32)


def cmd_knn(a):
    d, enc, t0 = a.ds, a.enc, time.time()
    if d not in SETS or enc not in ARMS:
        stop(f"part 2 reads {SETS} and {ARMS}")
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
    osc = scores_at(eg, nbr, sc, g_first)
    src, dst, w = k_edges(pos2g, outp, osc)
    tie = int(Z1.tie_rows(sc).sum())
    base = OUT2 / "knn" / enc
    base.mkdir(parents=True, exist_ok=True)
    sha = Z1.save_npz(base / f"{d}.npz", src=src, dst=dst, weight=w)
    rec = {"stage": "ENC E1b part 2: K", "declared_in": DECLARED, "dataset": d, "enc": enc, "nodes": n,
           "nodes_sha256": nsha, "groups": int(g_first.size), "groups_sha256": gsha, "freeze_RECORD_SHA256": freeze,
           "edges": int(src.size), "tie_rows": tie, "threads": E.THREADS, "search_seconds": secs, "npz_sha256": sha,
           "vectors": {"record_sha256": E.RECORD_SHA} if enc == "frozen" else {
               "manifest_sha256": E.sha256(E.vec_dir(served, enc, d, "docs") / "manifest.json"), "revision": E.pinned(enc)},
           **code_shas()}
    if enc == "frozen":
        with np.load(served / d / "graph" / "knn.npz") as z:
            rs, rd, rw = (z["src"].astype(np.int64), z["dst"].astype(np.int64), z["weight"].astype(np.float32))
        ra, rb = np.minimum(rs, rd), np.maximum(rs, rd)
        rk = ra * n + rb
        mk = src.astype(np.int64) * n + dst.astype(np.int64)
        common, i_m, i_r = np.intersect1d(mk, rk, assume_unique=False, return_indices=True)
        wdiff = float(np.abs(w[i_m] - rw[i_r]).max()) if common.size else None
        gk = {"record_edges": int(rk.size), "record_unique_pairs": int(np.unique(rk).size), "rebuilt_edges": int(mk.size),
              "common": int(common.size), "rebuilt_only": int(mk.size - common.size),
              "record_only": int(np.unique(rk).size - common.size), "weight_max_abs_diff": wdiff,
              "record_sorted": bool(np.all(np.diff(rk) > 0))}
        gk["verdict"] = "PASS" if (gk["rebuilt_only"] == 0 and gk["record_only"] == 0
                                   and wdiff is not None and wdiff <= GK_W_TOL) else "FAIL"
        rec["gk"] = gk
        log(f"{d}: GK {gk['verdict']}: {json.dumps(gk)}")
    rec["seconds"] = round(time.time() - t0, 1)
    E.wjson(base / f"{d}.json", rec)
    log(f"{d} {enc}: {src.size:,} edges -> {base / (d + '.npz')} ({rec['seconds']} s)")
    return 0 if rec.get("gk", {}).get("verdict", "PASS") == "PASS" else 3


# ── self-test ────────────────────────────────────────────────────────────────


def cmd_selftest(_a):
    rng = np.random.default_rng(0)
    ok = True
    # k_edges against a brute force on a toy corpus with one duplicated text
    n, dim = 60, 8
    X = rng.standard_normal((n, dim)).astype(np.float32)
    X[17] = X[3]                                            # positions 3 and 17 share a text group
    dig = np.zeros((n, 16), np.uint8)
    dig[:, 0] = np.arange(n) % 256
    dig[17] = dig[3]
    pos2g, g_first = Z1.group_digests(dig)
    G = g_first.size
    Xg = X[g_first] / np.linalg.norm(X[g_first], axis=1, keepdims=True)
    S = Xg @ Xg.T
    nbr = np.argsort(-S, axis=1, kind="stable")[:, :Z1.TOPN]
    sc = np.take_along_axis(S, nbr, 1).astype(np.float32)
    eg = np.arange(G)
    outp = Z1.out_positions(eg, nbr, g_first)
    src, dst, w = k_edges(pos2g, outp, scores_at(eg, nbr, sc, g_first))
    want = set()
    for x in range(n):
        g = pos2g[x]
        top = [t for t in nbr[g, :Z1.SEARCH_K] if t != g]
        for t in top:
            p = int(g_first[t])
            want.add((min(x, p), max(x, p)))
    got = set(zip(src.tolist(), dst.tolist()))
    ok = ok and got == want and bool(np.all(src < dst))
    print(f"selftest k_edges: {len(got)} edges, equal to brute force: {got == want}")
    wok = all(abs(float(w[i]) - float(S[pos2g[s_], pos2g[d_]])) < 1e-5 for i, (s_, d_) in enumerate(zip(src, dst)))
    ok = ok and wok
    print(f"selftest weights equal the pair's cosine: {wok}")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("knn")
    k.add_argument("--enc", required=True)
    k.add_argument("--ds", required=True)
    k.add_argument("--host", action="store_true")
    k.add_argument("--threads", default="8")
    t = sub.add_parser("selftest")
    t.add_argument("--threads", default="2")
    a = ap.parse_args()
    return {"knn": cmd_knn, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
