"""ENC E4, section 20.10: K's groups as the record's own source rows. HotpotQA's GK-G failed (20.9) because zenc1's
stand-in for a source row, a text group (positions with byte-identical `text`), merges source rows that the record keeps
apart. The record's source rows are its pointer index: frozen's vector manifest names it (materialised.pointer_index, with
its sha256) and the package keeps the file under _history/materialize/, which the host mirror does not copy. A source-row
group is a set of positions with the same (src, row) in that index. The groups are built on the laptop, where the index
is, and pushed. The search is zenc4g's, unedited (nn_top_torch). out_positions, scores_at, k_edges and tie_rows are
zenc1's and zenc1b2's, unedited.

    python outputs/mp_unified/zenc4s.py groups --ds DS     laptop: source-row groups -> outputs/enc/e4/srg/DS.npz + json
    python outputs/mp_unified/zenc4s.py gk --ds DS         host: GK-S, frozen on the GPU with them against graph/knn.npz
    python outputs/mp_unified/zenc4s.py knn --ds DS        host: e3v's K with them -> outputs/enc/e4/knn/e3v/DS.npz
    python outputs/mp_unified/zenc4s.py selftest          a toy whose one shared text holds two source rows
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import zenc as E  # noqa: E402  (sets the BLAS threads from --threads before numpy loads)
import zenc3v as V  # noqa: E402  (registers e3v and its pin)
import zenc1 as Z1  # noqa: E402
import zenc1b2 as B2  # noqa: E402
import zenc4g as G  # noqa: E402
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

OUT = G.OUT
SRG = OUT / "srg"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 20.10"
SETS = ("hotpotqa", "2wiki", "webqsp", "musique", "metaqa", "squad")   # groups: every dataset (a count check)
GPU_SETS = ("hotpotqa", "2wiki")                                       # gk and knn


def stop(msg):
    E.stop(f"E4 source-row groups: {msg}")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def code_sha():
    fs = (__file__, G.__file__, Z1.__file__, B2.__file__)
    return {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in fs}


def pointer_groups(src, row):
    """(pos2g, g_first): a group = one (src, row) of the pointer index; ids ranked by lowest position (zenc1's ranking)."""
    key = np.asarray(src, np.int64) * (1 << 32) + np.asarray(row, np.int64)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(order.size, np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank[np.asarray(inv).ravel()], first[order].astype(np.int64)


def pins(served, d):
    """The pointer index frozen's vectors were materialised from, and the two source-row counts the package records."""
    vm = json.loads((E.vec_dir(served, "frozen", d, "docs") / "manifest.json").read_text(encoding="utf-8"))
    mt = vm.get("materialised") or {}
    gm = served / d / "graph" / "GRAPH_MANIFEST.json"
    k = json.loads(gm.read_text(encoding="utf-8")).get("families", {}).get("knn", {}) if gm.is_file() else {}
    kn = ((k.get("source") or {}).get("source_manifest") or {}).get("n_distinct_source_rows")
    return {"pointer_index": mt.get("pointer_index"), "pointer_index_sha256": mt.get("pointer_index_sha256"),
            "vectors_distinct_source_rows": mt.get("distinct_source_rows"),
            "knn_distinct_source_rows": None if kn is None else int(kn)}


def cmd_groups(a):
    if a.ds not in SETS:
        stop(f"groups reads {SETS}")
    t0 = time.time()
    d = a.ds
    served, freeze = Z1.package(False)
    n, nsha = Z1.dataset_nodes(served, d)
    pn = pins(served, d)
    p_idx = served / "_history" / "materialize" / f"{d}__pointer_index__dense.npz"
    if not p_idx.is_file() or not pn["pointer_index_sha256"]:
        stop(f"{d}: no pointer index ({p_idx.name}) or no pin in frozen's vector manifest")
    psha = E.sha256(p_idx)
    if psha != pn["pointer_index_sha256"]:
        stop(f"{d}: {p_idx.name} is {psha[:12]}, frozen's vectors were materialised from {pn['pointer_index_sha256'][:12]}")
    with np.load(p_idx) as z:
        src, row = z["src"], z["row"]
    if src.shape != (n,) or row.shape != (n,):
        stop(f"{d}: the pointer index has {row.shape} rows for {n} nodes")
    pos2g, g_first = pointer_groups(src, row)
    tpos2g, tfirst, tsha = Z1.text_groups(served, d, n, nsha)
    span = int((tpos2g[g_first][pos2g] != tpos2g).sum())          # positions whose source row's lowest has other text
    pairs = np.unique(tpos2g * np.int64(g_first.size) + pos2g)
    per_text = np.bincount(pairs // g_first.size, minlength=tfirst.size)
    split = np.flatnonzero(per_text > 1)
    want = {pn["vectors_distinct_source_rows"], pn["knn_distinct_source_rows"]} - {None}
    g_sr = "PASS" if want and want == {int(g_first.size)} else "FAIL"
    g_t = "PASS" if span == 0 else "FAIL"
    p = SRG / f"{d}.npz"
    sha = Z1.save_npz(p, n=np.int64(n), nodes_sha256=np.asarray(nsha), pos2g=pos2g, g_first=g_first,
                      pointer_sha256=np.asarray(psha))
    rec = {"declared_in": DECLARED, "dataset": d, "nodes": n, "nodes_sha256": nsha, "freeze_RECORD_SHA256": freeze,
           "pointer_index_file": f"_history/materialize/{p_idx.name}", "pointer_index_sha256": psha, **pn,
           "source_row_groups": int(g_first.size), "text_groups": int(tfirst.size), "text_groups_sha256": tsha,
           "text_groups_split": int(split.size),
           "split_text_groups": [{"lowest_position": int(tfirst[g]), "source_rows": int(per_text[g])} for g in split[:200]],
           "positions_whose_group_spans_texts": span, "G_SR": g_sr, "G_T": g_t, "npz_sha256": sha,
           "code_sha256": code_sha(), "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    E.wjson(p.with_suffix(".json"), rec)
    log(f"{d}: {g_first.size:,} source-row groups, {tfirst.size:,} text groups ({split.size} split); G-SR {g_sr}, "
        f"G-T {g_t} ({rec['seconds']} s)")
    return 0 if g_sr == g_t == "PASS" else 3


def load_groups(served, d, n, nsha):
    p = SRG / f"{d}.npz"
    j = p.with_suffix(".json")
    if not p.is_file() or not j.is_file():
        stop(f"{d}: no source-row groups (built on the laptop by `groups`, then pushed)")
    r = json.loads(j.read_text(encoding="utf-8"))
    if r.get("G_SR") != "PASS" or r.get("G_T") != "PASS":
        stop(f"{d}: G-SR {r.get('G_SR')}, G-T {r.get('G_T')}")
    sha = E.sha256(p)
    if sha != r["npz_sha256"]:
        stop(f"{d}: the groups file is not its record's")
    pn = pins(served, d)
    if pn["pointer_index_sha256"] != r["pointer_index_sha256"]:
        stop(f"{d}: the groups were built from another pointer index than this package's vectors")
    with np.load(p) as z:
        if int(z["n"]) != n or str(z["nodes_sha256"]) != nsha or str(z["pointer_sha256"]) != r["pointer_index_sha256"]:
            stop(f"{d}: the groups are for other nodes")
        pos2g, g_first = z["pos2g"], z["g_first"]
    if {pn["vectors_distinct_source_rows"], pn["knn_distinct_source_rows"]} - {None} != {int(g_first.size)}:
        stop(f"{d}: {g_first.size} groups against the package's source-row counts")
    return pos2g, g_first, sha


def edges(pos2g, g_first, nbr, sc):
    eg = np.arange(g_first.size, dtype=np.int64)
    outp = Z1.out_positions(eg, nbr, g_first)
    osc = B2.scores_at(eg, nbr, sc, g_first, Z1.SEARCH_K)
    src, dst, w = B2.k_edges(pos2g, outp, osc)
    return src, dst, w, Z1.tie_rows(sc)


def build(a, enc):
    """zenc4g.build with the source-row groups in place of the text groups; nothing else differs."""
    import torch
    t0 = time.time()
    served, freeze = Z1.package(True)
    dev, name = G.device(a)
    n, nsha = Z1.dataset_nodes(served, a.ds)
    pos2g, g_first, gsha = load_groups(served, a.ds, n, nsha)
    paths = E.vec_shards(served, enc, a.ds, "docs")
    offs = Z1.shard_offsets(paths)
    if int(offs[-1]) != n:
        stop(f"{enc} {a.ds}: {int(offs[-1])} vector rows for {n} nodes")
    C = Z1.unit(Z1.gather(paths, offs, g_first))
    seg_rows = max(G.CHUNK, int(a.seg_gb * 2 ** 30 // (C.shape[1] * 4)))
    log(f"{a.ds} {enc}: {n:,} nodes, {g_first.size:,} source-row groups x {C.shape[1]}; segments of {seg_rows:,} rows "
        f"on {name}")
    t1 = time.time()
    nbr, sc = G.nn_top_torch(C, dev, seg_rows)
    del C
    secs = round(time.time() - t1, 1)
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    src, dst, w, tie = edges(pos2g, g_first, nbr, sc)
    rec = {"declared_in": DECLARED, "dataset": a.ds, "enc": enc, "nodes": n, "nodes_sha256": nsha,
           "groups": int(g_first.size), "groups_kind": "the record's source rows (pointer index, 20.10)",
           "groups_sha256": gsha, "freeze_RECORD_SHA256": freeze, "edges": int(src.size), "tie_rows": int(tie.sum()),
           "device": name, "tf32": False, "seg_rows": seg_rows, "gpu_peak_gb": peak, "search_seconds": secs,
           "code_sha256": code_sha(), "threads": E.THREADS}
    return rec, (src, dst, w), (pos2g, g_first, tie, n), served, t0


def compare(src, dst, w, pos2g, tie, n, rs, rd, rw):
    """zenc4g.cmd_gk's comparison, unchanged, as a function; also returns the unexplained one-pass-only pairs."""
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
    return gk, only_m[~em], only_r[~er]


def cmd_gk(a):
    if a.ds not in GPU_SETS:
        stop(f"gk is for {GPU_SETS}")
    rec, (src, dst, w), (pos2g, g_first, tie, n), served, t0 = build(a, "frozen")
    with np.load(served / a.ds / "graph" / "knn.npz") as z:
        rs, rd, rw = z["src"].astype(np.int64), z["dst"].astype(np.int64), z["weight"].astype(np.float32)
    gk, um, ur = compare(src, dst, w, pos2g, tie, n, rs, rd, rw)
    gk["rebuilt_only_unexplained_pairs"] = [[int(k // n), int(k % n)] for k in um[:500]]
    gk["record_only_unexplained_pairs"] = [[int(k // n), int(k % n)] for k in ur[:500]]
    rec.update({"stage": "ENC E4: GK-S (frozen on the GPU, the record's source rows)", "gk": gk,
                "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    E.wjson(OUT / f"gks_{a.ds}.json", rec)
    log(f"GK-S {a.ds}: {gk['verdict']} " + json.dumps({k: v for k, v in gk.items() if not k.endswith("_pairs")})
        + f" ({rec['search_seconds']} s search, {rec['gpu_peak_gb']} GB)")
    return 0 if gk["verdict"] == "PASS" else 3


def cmd_knn(a):
    if a.ds not in GPU_SETS:
        stop(f"knn is for {GPU_SETS}")
    g = OUT / f"gks_{a.ds}.json"
    if not g.is_file() or json.loads(g.read_text(encoding="utf-8"))["gk"]["verdict"] != "PASS":
        stop(f"GK-S has not passed on {a.ds}")
    fc = OUT / f"fc_{a.ds}.json"
    if not fc.is_file() or json.loads(fc.read_text(encoding="utf-8")).get("verdict") != "PASS":
        stop(f"E4's FC has not passed on {a.ds}")
    rec, (src, dst, w), _, served, t0 = build(a, V.ENC)
    p = OUT / "knn" / V.ENC / f"{a.ds}.npz"
    sha = Z1.save_npz(p, src=src, dst=dst, weight=w)
    if sha is None:
        sha = E.sha256(p)
    rec.update({"stage": "ENC E4: K (e3v, the GPU search, the record's source rows)", "npz_sha256": sha,
                "gk_s": E.sha256(g),
                "vectors": {"manifest_sha256": E.sha256(E.vec_dir(served, V.ENC, a.ds, "docs") / "manifest.json"),
                            "revision": E.pinned(V.ENC)},
                "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    E.wjson(p.with_suffix(".json"), rec)
    log(f"{a.ds} e3v: {src.size:,} edges -> {p} ({rec['seconds']} s)")
    return 0


def cmd_selftest(a):
    """A shared text holding two source rows with identical stored rows (HotpotQA's case): the text groups miss the
    record's pairs between them, the source-row groups reproduce the record exactly."""
    import torch
    rng = np.random.default_rng(0)
    n, d = 2400, 16
    X = rng.standard_normal((n, d)).astype(np.float16).astype(np.float32)
    row = np.arange(n, dtype=np.int64)
    row[[400, 1500]] = 10                         # true duplicates: one source row, one text
    X[400] = X[10]
    X[1500] = X[10]
    X[700] = X[20]                                # one text, two source rows {20, 21} and {700}: identical stored rows
    X[21] = X[20]
    row[21] = 20
    tkey = row.copy()
    tkey[700] = 20                                # the text groups merge 700 into 20's
    dig = np.zeros((n, 16), np.uint8)
    dig[:, :8] = tkey.view(np.uint8).reshape(n, 8)
    tpos2g, tfirst = Z1.group_digests(dig)
    pos2g, g_first = pointer_groups(np.zeros(n, np.int8), row)
    assert tfirst.size == n - 4 and g_first.size == n - 3, (tfirst.size, g_first.size)
    assert (tpos2g[g_first][pos2g] == tpos2g).all()                       # G-T holds on the toy
    assert pos2g[10] == pos2g[400] == pos2g[1500] and pos2g[20] == pos2g[21] != pos2g[700]
    C = Z1.unit(X[g_first])                                               # the record: brute force over source rows
    ts, tg = Z1.row_top(C @ C.T, np.arange(g_first.size, dtype=np.int64), Z1.TOPN)
    rsrc, rdst, rw, _ = edges(pos2g, g_first, tg, ts)
    torch.set_num_threads(2)
    res = {}
    for kind, (p2g, gf) in (("text", (tpos2g, tfirst)), ("source", (pos2g, g_first))):
        nb, sc = G.nn_top_torch(Z1.unit(X[gf]), torch.device("cpu"), seg_rows=900, block=500, chunk=400)
        src, dst, w, tie = edges(p2g, gf, nb, sc)
        res[kind] = compare(src, dst, w, p2g, tie, n, rsrc.astype(np.int64), rdst.astype(np.int64), rw)[0]
    assert res["source"]["verdict"] == "PASS" and res["source"]["rebuilt_only"] == res["source"]["record_only"] == 0
    assert res["text"]["record_only_unexplained"] > 0, res["text"]
    print(f"zenc4s selftest: ok (text groups: {res['text']['rebuilt_only']} rebuilt-only, {res['text']['record_only']} "
          f"record-only pairs, {res['text']['verdict']}; source-row groups: the record exactly)")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="zenc4s.py")
    ap.add_argument("cmd", choices=["groups", "gk", "knn", "selftest"])
    ap.add_argument("--ds")
    ap.add_argument("--seg-gb", type=float, default=7.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.cmd != "selftest" and not a.ds:
        stop(f"{a.cmd} needs --ds")
    return {"groups": cmd_groups, "gk": cmd_gk, "knn": cmd_knn, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
