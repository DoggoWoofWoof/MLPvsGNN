"""ENC E4, section 20.9: the record's kNN rule on the GPU, for the corpora where the CPU search takes days (webqsp,
hotpotqa, 2wiki). zenc1's text groups, unit rows, out_positions and tie rows and zenc1b2's scores_at and k_edges are used
unedited; only the all-pairs top-5 search (zenc1.nn_top) runs on the GPU: exact float32 (TF32 off), every group against
every group, order = score descending then group ascending, as zenc1.row_top orders.

    python outputs/mp_unified/zenc4g.py gk --ds DS [--seg-gb G]     frozen on the GPU against the package's graph/knn.npz
    python outputs/mp_unified/zenc4g.py knn --ds DS [--seg-gb G]    e3v -> outputs/enc/e4/knn/e3v/DS.npz (needs GK PASS)
    python outputs/mp_unified/zenc4g.py selftest                   CPU nn_top vs this search on a toy (CPU torch)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import zenc as E  # noqa: E402
import zenc3v as V  # noqa: E402  (registers e3v and its pin)
import zenc1 as Z1  # noqa: E402
import zenc1b2 as B2  # noqa: E402
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

GPU_DS = ("webqsp", "hotpotqa", "2wiki", "musique", "metaqa")   # musique and metaqa: GK checks of this code only
OUT = E.OUT / "e4"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 20.9"
R = 8                       # candidates kept per row (> zenc1.TOPN = 5)
BLOCK = 4096                # endpoint rows per GEMM
CHUNK = 65536               # candidate columns per GEMM


def stop(msg):
    E.stop(f"E4 GPU K: {msg}")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def code_sha():
    return {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (__file__, Z1.__file__, B2.__file__)}


def keep_top(ts, tg, cs, cg, r):
    """Merge candidate rows (scores, groups) into the top r by score descending, then group ascending."""
    import torch
    s = torch.cat([ts, cs], 1)
    g = torch.cat([tg, cg], 1)
    o = torch.argsort(g, dim=1, stable=True)
    s, g = torch.gather(s, 1, o), torch.gather(g, 1, o)
    o = torch.argsort(-s, dim=1, stable=True)
    s, g = torch.gather(s, 1, o), torch.gather(g, 1, o)
    return s[:, :r].contiguous(), g[:, :r].contiguous()


def nn_top_torch(C, dev, seg_rows, block=BLOCK, chunk=CHUNK, topn=Z1.TOPN):
    """C: (G, d) float32 unit rows in group order (row g = group g's lowest position). Every group's top `topn` groups."""
    import torch
    G = C.shape[0]
    big = torch.iinfo(torch.int64).max
    bs = torch.full((G, R), float("-inf"), dtype=torch.float32, device=dev)
    bg = torch.full((G, R), big, dtype=torch.int64, device=dev)
    t0 = time.time()
    for s0 in range(0, G, seg_rows):
        s1 = min(G, s0 + seg_rows)
        Cs = torch.from_numpy(C[s0:s1]).to(dev)
        for b0 in range(0, G, block):
            b1 = min(G, b0 + block)
            Xb = torch.from_numpy(C[b0:b1]).to(dev)
            ts, tg = bs[b0:b1], bg[b0:b1]
            for c0 in range(0, s1 - s0, chunk):
                c1 = min(s1 - s0, c0 + chunk)
                S = Xb @ Cs[c0:c1].T
                k = min(R, c1 - c0)
                v, i = torch.topk(S, k, dim=1)
                ts, tg = keep_top(ts, tg, v, i + (s0 + c0), R)
            bs[b0:b1], bg[b0:b1] = ts, tg
        del Cs
        log(f"  segment {s1:,}/{G:,} ({time.time() - t0:.0f} s)")
    sc, nb = bs[:, :topn].cpu().numpy(), bg[:, :topn].cpu().numpy()
    nb[~np.isfinite(sc)] = Z1.BIG
    return nb, sc


def group_rows(served, enc, d):
    n, nsha = Z1.dataset_nodes(served, d)
    pos2g, g_first, gsha = Z1.text_groups(served, d, n, nsha)
    paths = E.vec_shards(served, enc, d, "docs")
    offs = Z1.shard_offsets(paths)
    if int(offs[-1]) != n:
        stop(f"{enc} {d}: {int(offs[-1])} vector rows for {n} nodes")
    C = Z1.unit(Z1.gather(paths, offs, g_first))
    return n, nsha, pos2g, g_first, gsha, C


def device(a):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        stop("no GPU")
    return torch.device("cuda"), torch.cuda.get_device_name(0)


def build(a, enc):
    import torch
    t0 = time.time()
    served, freeze = Z1.package(True)
    dev, name = device(a)
    n, nsha, pos2g, g_first, gsha, C = group_rows(served, enc, a.ds)
    seg_rows = max(CHUNK, int(a.seg_gb * 2 ** 30 // (C.shape[1] * 4)))
    log(f"{a.ds} {enc}: {n:,} nodes, {g_first.size:,} groups x {C.shape[1]}; segments of {seg_rows:,} rows on {name}")
    t1 = time.time()
    nbr, sc = nn_top_torch(C, dev, seg_rows)
    secs = round(time.time() - t1, 1)
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    eg = np.arange(g_first.size, dtype=np.int64)
    outp = Z1.out_positions(eg, nbr, g_first)
    osc = B2.scores_at(eg, nbr, sc, g_first, Z1.SEARCH_K)
    src, dst, w = B2.k_edges(pos2g, outp, osc)
    tie = Z1.tie_rows(sc)
    rec = {"declared_in": DECLARED, "dataset": a.ds, "enc": enc, "nodes": n, "nodes_sha256": nsha,
           "groups": int(g_first.size), "groups_sha256": gsha, "freeze_RECORD_SHA256": freeze, "edges": int(src.size),
           "tie_rows": int(tie.sum()), "device": name, "tf32": False, "seg_rows": seg_rows, "gpu_peak_gb": peak,
           "search_seconds": secs, "code_sha256": code_sha(), "threads": E.THREADS}
    return rec, (src, dst, w), (pos2g, g_first, tie, n), served, t0


def cmd_gk(a):
    """20.9 GK-G: frozen's K by this search against the package's graph/knn.npz. A pair in one set and not the other
    passes only when one of its ends' groups is a tie row (4th and 5th scores within zenc1.TIE); common pairs' weights
    within zenc1b2.GK_W_TOL."""
    if a.ds not in GPU_DS:
        stop(f"{a.ds} is not one of {GPU_DS}")
    rec, (src, dst, w), (pos2g, g_first, tie, n), served, t0 = build(a, "frozen")
    with np.load(served / a.ds / "graph" / "knn.npz") as z:
        rs, rd, rw = z["src"].astype(np.int64), z["dst"].astype(np.int64), z["weight"].astype(np.float32)
    rk = np.minimum(rs, rd) * n + np.maximum(rs, rd)
    o = np.argsort(rk, kind="stable")
    rk, rw = rk[o], rw[o]
    first = np.r_[True, rk[1:] != rk[:-1]]
    rk, rw = rk[first], rw[first]
    mk = src.astype(np.int64) * n + dst.astype(np.int64)
    common, i_m, i_r = np.intersect1d(mk, rk, assume_unique=True, return_indices=True)
    only_m = np.setdiff1d(mk, rk, assume_unique=True)
    only_r = np.setdiff1d(rk, mk, assume_unique=True)
    tie_pos = tie[pos2g]

    def explained(keys):
        return tie_pos[keys // n] | tie_pos[keys % n] if keys.size else np.zeros(0, bool)

    em, er = explained(only_m), explained(only_r)
    wd = float(np.abs(w[i_m] - rw[i_r]).max()) if common.size else None
    gk = {"record_pairs": int(rk.size), "rebuilt_pairs": int(mk.size), "common": int(common.size),
          "rebuilt_only": int(only_m.size), "record_only": int(only_r.size),
          "rebuilt_only_unexplained": int((~em).sum()), "record_only_unexplained": int((~er).sum()),
          "weight_max_abs_diff": wd}
    gk["verdict"] = "PASS" if (gk["rebuilt_only_unexplained"] == 0 and gk["record_only_unexplained"] == 0
                               and wd is not None and wd <= B2.GK_W_TOL) else "FAIL"
    rec.update({"stage": "ENC E4: GK-G (frozen on the GPU)", "gk": gk, "seconds": round(time.time() - t0, 1),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    OUT.mkdir(parents=True, exist_ok=True)
    E.wjson(OUT / f"gkg_{a.ds}.json", rec)
    log(f"GK-G {a.ds}: {gk['verdict']} {json.dumps(gk)} ({rec['search_seconds']} s search, {rec['gpu_peak_gb']} GB)")
    return 0 if gk["verdict"] == "PASS" else 3


def cmd_knn(a):
    if a.ds not in ("webqsp", "hotpotqa", "2wiki"):
        stop("e3v's K on the GPU is for webqsp, hotpotqa and 2wiki (20.9)")
    g = OUT / f"gkg_{a.ds}.json"
    if not g.is_file() or json.loads(g.read_text(encoding="utf-8"))["gk"]["verdict"] != "PASS":
        stop(f"GK-G has not passed on {a.ds}")
    fc = OUT / f"fc_{a.ds}.json"
    if not fc.is_file() or json.loads(fc.read_text(encoding="utf-8")).get("verdict") != "PASS":
        stop(f"E4's FC has not passed on {a.ds}")
    rec, (src, dst, w), _, served, t0 = build(a, V.ENC)
    p = OUT / "knn" / V.ENC / f"{a.ds}.npz"
    sha = Z1.save_npz(p, src=src, dst=dst, weight=w)
    if sha is None:
        sha = E.sha256(p)
    rec.update({"stage": "ENC E4: K (e3v, the GPU search)", "npz_sha256": sha, "gk_g": E.sha256(g),
                "vectors": {"manifest_sha256": E.sha256(E.vec_dir(served, V.ENC, a.ds, "docs") / "manifest.json"),
                            "revision": E.pinned(V.ENC)},
                "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    E.wjson(p.with_suffix(".json"), rec)
    log(f"{a.ds} e3v: {src.size:,} edges -> {p} ({rec['seconds']} s)")
    return 0


def cmd_selftest(a):
    """The torch search against zenc1.nn_top on toy shards, on the CPU (exact ties included)."""
    import tempfile
    import torch
    rng = np.random.default_rng(0)
    G, d = 3000, 24
    X = rng.standard_normal((G, d)).astype(np.float32)
    X[17] = X[5]                                       # an exact tie
    X[2900] = X[5]
    with tempfile.TemporaryDirectory() as t:
        paths = []
        for s, (lo, hi) in enumerate(((0, 1000), (1000, 2200), (2200, G))):
            p = Path(t) / f"shard_{s:05d}.npy"
            np.save(p, X[lo:hi].astype(np.float16))
            paths.append(p)
        offs = Z1.shard_offsets(paths)
        pos2g = np.arange(G, dtype=np.int64)
        g_first = np.arange(G, dtype=np.int64)
        eg = np.arange(G, dtype=np.int64)
        nb_ref, sc_ref = Z1.nn_top(paths, offs, pos2g, g_first, eg, every=0)
        C = Z1.unit(Z1.gather(paths, offs, g_first))
        torch.set_num_threads(2)
        nb, sc = nn_top_torch(C, torch.device("cpu"), seg_rows=1100, block=700, chunk=500)
    same = (nb == nb_ref).all(1)
    tie = Z1.tie_rows(sc_ref)
    assert (same | tie).all(), int((~(same | tie)).sum())
    assert np.abs(sc - sc_ref)[np.isfinite(sc_ref)].max() < 1e-5
    assert nb[5, 0] == 5 and set(nb[5, :3].tolist()) == {5, 17, 2900} and list(nb[5, :3]) == [5, 17, 2900]
    print(f"zenc4g selftest: ok ({int(same.sum())}/{G} rows identical, the rest tie rows)")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="zenc4g.py")
    ap.add_argument("cmd", choices=["gk", "knn", "selftest"])
    ap.add_argument("--ds")
    ap.add_argument("--seg-gb", type=float, default=7.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.cmd in ("gk", "knn") and not a.ds:
        stop(f"{a.cmd} needs --ds")
    return {"gk": cmd_gk, "knn": cmd_knn, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
