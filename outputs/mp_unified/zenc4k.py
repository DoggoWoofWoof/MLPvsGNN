"""ENC E4, section 20.8: e3v's K on the four other datasets (hotpotqa, 2wiki, webqsp, metaqa) by the record's kNN rule
(zenc1b2's K code, unedited; its arm and dataset lists and its output folder are widened in this process only).

    python outputs/mp_unified/zenc4k.py gks --ds DS [--n 20000] [--threads N]   frozen, sampled: the rule reproduces the
                                                                                 package's K on this dataset (20.8)
    python outputs/mp_unified/zenc4k.py knn --ds DS [--threads N]               e3v's K -> outputs/enc/e4/knn/e3v/DS.npz
    python outputs/mp_unified/zenc4k.py selftest
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import zenc as E  # noqa: E402  (reads --threads from sys.argv before numpy loads)
import zenc3v as V  # noqa: E402  (registers e3v and its pin)
import zenc1 as Z1  # noqa: E402
import zenc1b2 as B2  # noqa: E402
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

NEW4 = ("hotpotqa", "2wiki", "webqsp", "metaqa")
OUT = E.OUT / "e4"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 20.8"


def stop(msg):
    E.stop(f"E4 K: {msg}")


def widen():
    if V.ENC not in B2.ARMS:
        B2.ARMS = tuple(B2.ARMS) + (V.ENC,)
    B2.SETS = NEW4
    B2.OUT2 = OUT
    B2.DECLARED = DECLARED


def cmd_knn(a):
    if a.ds not in NEW4:
        stop(f"{a.ds} is not one of {NEW4}")
    g = OUT / f"gks_{a.ds}.json"
    if not g.is_file() or json.loads(g.read_text(encoding="utf-8")).get("verdict") != "PASS":
        stop(f"GKs has not passed on {a.ds}")
    widen()
    sys.argv = ["zenc1b2.py", "knn", "--enc", V.ENC, "--ds", a.ds, "--host", "--threads", str(a.threads)]
    return B2.main()


def cmd_gks(a):
    """20.8: on a seeded sample of text groups, frozen's out-pairs by the rule's code are all in the package's K, with
    weights within GK_W_TOL. A full rebuild (GK) passed on SQuAD, MuSiQue and MetaQA; this samples the four others."""
    if a.ds not in NEW4:
        stop(f"{a.ds} is not one of {NEW4}")
    t0 = time.time()
    served, freeze = Z1.package(True)
    n, nsha = Z1.dataset_nodes(served, a.ds)
    pos2g, g_first, gsha = Z1.text_groups(served, a.ds, n, nsha)
    paths = E.vec_shards(served, "frozen", a.ds, "docs")
    offs = Z1.shard_offsets(paths)
    if int(offs[-1]) != n:
        stop(f"frozen {a.ds}: {int(offs[-1])} rows for {n} nodes")
    rng = np.random.default_rng(0)
    m = min(int(a.n), int(g_first.size))
    eg = np.sort(rng.choice(g_first.size, size=m, replace=False)).astype(np.int64)
    nbr, sc = Z1.nn_top(paths, offs, pos2g, g_first, eg)
    outp = Z1.out_positions(eg, nbr, g_first)
    osc = B2.scores_at(eg, nbr, sc, g_first, Z1.SEARCH_K)
    tie = Z1.tie_rows(sc)
    x = np.broadcast_to(g_first[eg][:, None], outp.shape)
    v = outp >= 0
    ka = np.minimum(x, outp)[v].astype(np.int64) * n + np.maximum(x, outp)[v].astype(np.int64)
    kw = osc[v]
    krow = np.broadcast_to(np.arange(m)[:, None], outp.shape)[v]
    with np.load(served / a.ds / "graph" / "knn.npz") as z:
        rs, rd, rw = z["src"].astype(np.int64), z["dst"].astype(np.int64), z["weight"].astype(np.float32)
    rk = np.minimum(rs, rd) * n + np.maximum(rs, rd)
    o = np.argsort(rk, kind="stable")
    rk, rw = rk[o], rw[o]
    pos = np.searchsorted(rk, ka)
    hit = (pos < rk.size) & (rk[np.minimum(pos, rk.size - 1)] == ka)
    wd = np.abs(rw[np.minimum(pos, rk.size - 1)][hit] - kw[hit])
    miss_rows = np.unique(krow[~hit])
    rec = {"stage": "ENC E4: GKs", "declared_in": DECLARED, "dataset": a.ds, "nodes": n, "nodes_sha256": nsha,
           "groups": int(g_first.size), "groups_sha256": gsha, "freeze_RECORD_SHA256": freeze, "sample": m, "seed": 0,
           "out_pairs": int(ka.size), "in_package": int(hit.sum()), "missing": int((~hit).sum()),
           "missing_rows": int(miss_rows.size), "missing_rows_at_ties": int(tie[miss_rows].sum()) if miss_rows.size else 0,
           "tie_rows": int(tie.sum()), "weight_max_abs_diff": float(wd.max()) if wd.size else None,
           "threads": E.THREADS, "seconds": None, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # a miss only at a tie row (4th and 5th scores within zenc1.TIE) may come from a GEMM block of another shape (HFO G4)
    rec["verdict"] = "PASS" if (rec["missing_rows"] == rec["missing_rows_at_ties"] and wd.size
                                and rec["weight_max_abs_diff"] <= B2.GK_W_TOL) else "FAIL"
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    E.wjson(OUT / f"gks_{a.ds}.json", rec)
    print(f"GKs {a.ds}: {rec['verdict']} " + json.dumps({k: rec[k] for k in ("sample", "out_pairs", "missing",
                                                                            "missing_rows_at_ties", "weight_max_abs_diff")}),
          flush=True)
    return 0 if rec["verdict"] == "PASS" else 3


def cmd_selftest(a):
    widen()
    assert V.ENC in B2.ARMS and B2.SETS == NEW4 and B2.OUT2 == OUT
    assert B2.knn_path(V.ENC, "2wiki") == OUT / "knn" / V.ENC / "2wiki.npz"
    assert E.pinned(V.ENC).startswith("gtemodern@") and "+granitesm@" in E.pinned(V.ENC)
    # k_edges on a toy: three groups, positions 0..4 (positions 3 and 4 share group 2)
    pos2g = np.array([0, 1, 2, 2, 0][:4] + [2], np.int64)
    outp = np.array([[1, 2, -1, -1], [0, -1, -1, -1], [0, 1, -1, -1]], np.int64)
    osc = np.array([[0.9, 0.8, -np.inf, -np.inf], [0.9, -np.inf, -np.inf, -np.inf], [0.8, 0.7, -np.inf, -np.inf]],
                   np.float32)
    s, d, w = B2.k_edges(pos2g, outp, osc)
    assert (s < d).all() and len(set(zip(s.tolist(), d.tolist()))) == s.size
    print("zenc4k selftest: ok")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="zenc4k.py")
    ap.add_argument("cmd", choices=["gks", "knn", "selftest"])
    ap.add_argument("--ds")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    return {"gks": cmd_gks, "knn": cmd_knn, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
