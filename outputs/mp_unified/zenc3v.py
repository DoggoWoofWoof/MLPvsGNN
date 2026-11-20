"""ENC E3W (docs/ENC_ENCODER_COMPARISON.md section 19): the fused one-vector arm e3v (gte-modernbert-base fused with
granite-embedding-small-english-r2), its first-stage read and GV.

    python outputs/mp_unified/zenc3v.py vectors --ds squad|musique    e3v's float16 shards from E0N's two encodes
    python outputs/mp_unified/zenc3v.py first --ds squad|musique      zenc's first-stage read on e3v (outputs/enc/e0/e3v/)
    python outputs/mp_unified/zenc3v.py gv                            GV (19.3): that read against E3B's V-no-gamma read
    python outputs/mp_unified/zenc3v.py selftest [--real]

e3v is 16.3's one vector at E3B's fusion weight, with no q-map (19.2): [sqrt(1 - w) v_gtemodern ; sqrt(w) v_granitesm],
w = 0.5, for documents and questions alike, stored float16. Importing this module registers e3v in zenc's ENCODERS and
pins (after zenc3's registration of the two encoders); zenc3w.py runs E2's modules with it. Nothing here trains.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import zenc as E  # noqa: E402  (first: sets the BLAS threads before numpy loads)
import zenc3 as Z3  # noqa: E402  (registers gtemodern, granite, granitesm and pins them to section 17.2)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

STAGE = "ENC E3W"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 19"
ENC = "e3v"
PARTS = ("gtemodern", "granitesm")                    # E3B's arm gtemodern+granitesm: (first, second)
W = 0.5                                               # E3B's chosen fusion weight (18.9)
SCALE = {"gtemodern": float(np.sqrt(1.0 - W)), "granitesm": float(np.sqrt(W))}
DIM = sum(int(Z3.NEW3[e]["dim"]) for e in PARTS)      # 768 + 384 = 1152
DS = ("squad", "musique")
OUT = E.OUT / "e3w"
E3B = E.OUT / "e3b"
REF_ARM = "gtemodern+granitesm"
REF_READ = "gtemodern+granitesm-V-no-gamma"           # E3B's V without its q-map: gamma 0, w 0.5, no windows, no prf
REF_SETTING = {"gamma": 0.0, "beta": 0.0, "W": None, "w": W, "prf": None}
GV_SHARE, GV_PTS = 0.995, 0.10
FUSION = (f"[sqrt({1.0 - W:g}) v_{PARTS[0]} ; sqrt({W:g}) v_{PARTS[1]}] (section 16.3's one vector, E3B's w); "
          "no q-map (section 19.2)")
REPO = " + ".join(Z3.NEW3[e]["repo"] for e in PARTS)

E.ENCODERS[ENC] = {"repo": REPO, "dim": DIM, "query": {}, "budget": min(int(Z3.NEW3[e]["budget"]) for e in PARTS),
                   "parts": PARTS}
_pinned = E.pinned                                    # zenc3's (the parts' pins come from outputs/enc/e0n/models.json)


def pinned(enc):
    if enc != ENC:
        return _pinned(enc)
    return "+".join(f"{e}@{_pinned(e)}" for e in PARTS)


E.pinned = pinned


def log(msg):
    print(f"[zenc3v {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc3v: {msg}")


def code_sha():
    return {"zenc3v.py": E.sha256(__file__)}


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(E.WS)).replace("\\", "/")
    except ValueError:
        return str(p)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fuse(blocks):
    """[sqrt(1 - w) a ; sqrt(w) b] in float32, stored float16 (blocks in PARTS order, float16 unit rows)."""
    if len(blocks) != len(PARTS):
        stop(f"{len(blocks)} blocks for {len(PARTS)} parts")
    n = {int(b.shape[0]) for b in blocks}
    if len(n) != 1:
        stop(f"the parts' blocks have {sorted(n)} rows")
    for e, b in zip(PARTS, blocks):
        if b.ndim != 2 or b.shape[1] != int(Z3.NEW3[e]["dim"]):
            stop(f"{e}: a block {b.shape}, not (rows, {Z3.NEW3[e]['dim']})")
    return np.concatenate([np.asarray(b, np.float32) * np.float32(SCALE[e]) for e, b in zip(PARTS, blocks)],
                          1).astype(np.float16)


# ── vectors ──────────────────────────────────────────────────────────────────

def part_manifest(root, e, ds, kind):
    mp = E.vec_dir(root, e, ds, kind) / "manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    enc = m.get("encoder", {})
    if enc.get("revision") != Z3.NEW3[e]["rev"] or int(enc.get("dim", -1)) != int(Z3.NEW3[e]["dim"]):
        stop(f"{e} {ds} {kind}: its manifest names {enc.get('revision')} at {enc.get('dim')}, not section 17.2's pin")
    return {"manifest": rel(mp), "manifest_sha256": E.sha256(mp), "revision": enc.get("revision"),
            "device": enc.get("device"), "dtype": enc.get("dtype"), "max_seq_length": enc.get("max_seq_length")}


def cmd_vectors(a):
    root = E.served_root()
    t0 = time.time()
    for kind in ("docs", "queries"):
        rman = E.manifest(root, a.ds, kind)
        srcs = {e: E.vec_shards(root, e, a.ds, kind) for e in PARTS}      # shard count and rows checked by zenc
        parts = {e: part_manifest(root, e, a.ds, kind) for e in PARTS}
        base = E.OUT / ENC / a.ds / kind
        base.mkdir(parents=True, exist_ok=True)
        shards = []
        for s in range(int(rman["n_shards"])):
            want = int(rman["shards"][s]["rows"])
            blocks = [np.load(srcs[e][s], mmap_mode="r") for e in PARTS]
            if any(int(b.shape[0]) != want for b in blocks):
                stop(f"{a.ds} {kind} shard {s}: {[b.shape for b in blocks]}, the package's has {want} rows")
            X = fuse(blocks)
            bad = E.damaged(X)
            if bad:
                stop(f"{a.ds} {kind} shard {s}: {bad} damaged rows")
            npy = base / (rman["format"]["file"] % s)
            tmp = npy.with_name(npy.stem + ".tmp.npy")
            np.save(tmp, X)
            os.replace(tmp, npy)
            r = {"shard": s, "rows": int(X.shape[0]), "file": npy.name, "bytes": npy.stat().st_size, "sha256": E.sha256(npy),
                 "sources": {e: {"file": rel(srcs[e][s]), "sha256": E.sha256(srcs[e][s])} for e in PARTS}}
            E.wjson(npy.with_suffix(".json"), r)
            shards.append(r)
            del X, blocks
        rows = sum(r["rows"] for r in shards)
        if rows != int(rman["n_rows"]):
            stop(f"{a.ds} {kind}: {rows} rows, the record has {rman['n_rows']}")
        E.wjson(base / "manifest.json", {
            "dataset": a.ds, "kind": kind, "model": ENC, "n_rows": rows, "shard_size": E.SHARD, "n_shards": len(shards),
            "row_semantics": rman["row_semantics"],
            "format": {"file": rman["format"]["file"], "dtype": "float16", "shape": f"(rows, {DIM})",
                       "normalised": "unit L2 (|norm-1| <= 0.01)"},
            "encoder": {"encoder": REPO, "revision": pinned(ENC), "dim": DIM, "dtype": "float16", "fusion": FUSION,
                        "parts": parts, "query_prompt": "", "document_prompt": ""},
            "shards": [{k: r[k] for k in ("shard", "rows", "file", "bytes", "sha256")} for r in shards],
            "stage": STAGE, "page": DECLARED, "record_sha256": E.RECORD_SHA, "code_sha256": code_sha(),
            "seconds": round(time.time() - t0, 1), "utc": utc()})
        log(f"{a.ds} {kind}: {rows:,} rows x {DIM} in {len(shards)} shards -> {rel(base)}")
    return 0


# ── first stage and GV ───────────────────────────────────────────────────────

def cmd_first(a):
    pinned(ENC)
    E.cmd_first(argparse.Namespace(enc=ENC, ds=a.ds))
    return 0


def recall(h, gt):
    return float(np.mean(np.minimum(h, gt) / gt.astype(np.float64)))


def gv_check(ha, hb, gt):
    """GV's rule on two reads of the same questions: hits at 5 equal on GV_SHARE of them, R@5 within GV_PTS points."""
    share = float(np.mean(ha["d5"] == hb["d5"]))
    ra, rb = recall(ha["d5"], gt), recall(hb["d5"], gt)
    g = {"questions": int(gt.size), "h_d5_equal_share": round(share, 6),
         "r5_pts": {"e3v": round(100 * ra, 2), "e3b": round(100 * rb, 2), "diff": round(100 * (ra - rb), 3)},
         "beside": {f"h_{k}_equal_share": round(float(np.mean(ha[k] == hb[k])), 6) for k in ("d20", "d1000", "rrf5")}}
    g["verdict"] = "PASS" if (share >= GV_SHARE and abs(100 * (ra - rb)) <= GV_PTS) else "FAIL"
    return g


def ref_read(ds):
    grid = E3B / f"grid-{REF_ARM}.json"
    ent = json.loads(grid.read_text(encoding="utf-8"))["reads"][f"{REF_READ}__{ds}"]
    if ent.get("setting") != REF_SETTING:
        stop(f"{REF_READ} {ds}: setting {ent.get('setting')}, not {REF_SETTING}")
    p = E3B / "reads" / ent["npz"]
    if E.sha256(p) != ent["sha256"]:
        stop(f"{rel(p)}: sha256 differs from its grid record")
    return p, {"file": rel(p), "sha256": ent["sha256"], "grid": rel(grid), "grid_sha256": E.sha256(grid)}


def gv_ds(ds):
    zp = E.OUT / "e0" / ENC / f"{ds}.npz"
    rec = json.loads(zp.with_suffix(".json").read_text(encoding="utf-8"))
    if E.sha256(zp) != rec["npz_sha256"]:
        stop(f"{rel(zp)} is not the file its record names")
    rp, rinfo = ref_read(ds)
    with np.load(zp, allow_pickle=False) as z, np.load(rp, allow_pickle=False) as r:
        if [str(x) for x in z["ids"]] != [str(x) for x in r["ids"]]:
            stop(f"{ds}: the two reads hold other questions or another order")
        gt = z["gold_total"].astype(np.int64)
        if not np.array_equal(gt, r["gold_total"].astype(np.int64)):
            stop(f"{ds}: the two reads' gold totals differ")
        if (gt <= 0).any():
            stop(f"{ds}: a read question has no gold")
        ha = {k: z[f"h_{k}"].astype(np.int64) for k in ("d5", "d20", "d1000", "rrf5")}
        hb = {k: r[f"h_{k}"].astype(np.int64) for k in ("d5", "d20", "d1000", "rrf5")}
        t5 = float(np.mean([set(a.tolist()) == set(b.tolist()) for a, b in zip(z["top_ids"][:, :5], r["top20"][:, :5])]))
    g = gv_check(ha, hb, gt)
    g["beside"]["top5_set_equal_share"] = round(t5, 6)
    g["e0"] = {"file": rel(zp), "sha256": rec["npz_sha256"]}
    g["ref"] = rinfo
    return g


def cmd_gv(a):
    out = {"stage": STAGE + ": GV", "declared_in": DECLARED + ".3",
           "rule": f"hits at 5 equal on >= {GV_SHARE} of the s1eval questions and R@5 within {GV_PTS} points, per carve",
           "datasets": {ds: gv_ds(ds) for ds in DS}, "code_sha256": code_sha(), "utc": utc()}
    out["verdict"] = "PASS" if all(v["verdict"] == "PASS" for v in out["datasets"].values()) else "FAIL"
    E.wjson(OUT / "gv.json", out)
    log(f"GV {out['verdict']}: {json.dumps({d: (v['h_d5_equal_share'], v['r5_pts']) for d, v in out['datasets'].items()})}")
    return 0 if out["verdict"] == "PASS" else 3


# ── self-test ────────────────────────────────────────────────────────────────

def unit16(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def selftest_toy():
    rng = np.random.default_rng(0)
    da, db = (int(Z3.NEW3[e]["dim"]) for e in PARTS)
    A, B, Qa, Qb = unit16(rng, 300, da), unit16(rng, 300, db), unit16(rng, 40, da), unit16(rng, 40, db)
    X, Q = fuse([A, B]), fuse([Qa, Qb])
    assert X.shape == (300, DIM) and X.dtype == np.float16 and DIM == 1152
    assert E.damaged(X) == 0 and E.damaged(Q) == 0
    want = (1 - W) * (Qa.astype(np.float32) @ A.astype(np.float32).T) + W * (Qb.astype(np.float32) @ B.astype(np.float32).T)
    got = Q.astype(np.float32) @ X.astype(np.float32).T
    assert float(np.abs(got - want).max()) < 2e-3, float(np.abs(got - want).max())
    assert np.array_equal(X[:, :da].astype(np.float32), (A.astype(np.float32) * np.float32(SCALE[PARTS[0]])).astype(np.float16).astype(np.float32))
    try:
        fuse([A, B[:10]])
        raise AssertionError("fuse took blocks of unequal rows")
    except SystemExit:
        pass
    gt = rng.integers(1, 4, 1000)
    h = {k: rng.integers(0, 3, 1000) for k in ("d5", "d20", "d1000", "rrf5")}
    assert gv_check(h, {k: v.copy() for k, v in h.items()}, gt)["verdict"] == "PASS"
    h2 = {k: v.copy() for k, v in h.items()}
    h2["d5"][:4] = (h2["d5"][:4] + 1) % 3                                  # 0.4% of questions differ
    assert gv_check(h, h2, gt)["h_d5_equal_share"] == 0.996
    h3 = {k: v.copy() for k, v in h.items()}
    h3["d5"][:6] = (h3["d5"][:6] + 1) % 3                                  # 0.6%: under the share
    assert gv_check(h, h3, gt)["verdict"] == "FAIL"
    assert pinned(ENC) == f"gtemodern@{Z3.NEW3['gtemodern']['rev']}+granitesm@{Z3.NEW3['granitesm']['rev']}" or \
        not (E.OUT / "e0n" / "models.json").is_file()
    assert E.ENCODERS[ENC]["dim"] == DIM and E.pinned is pinned
    print("zenc3v toy self-test: OK")


def selftest_real():
    """The host: the parts' manifests are section 17.2's pins over the record's rows; E3B's reference reads are their
    records'; on 300 s1eval questions of each carve, e3v's float16 vectors give E3B's top 5 (as sets) on >= 99%."""
    root = E.served_root()
    for ds in DS:
        for kind in ("docs", "queries"):
            for e in PARTS:
                part_manifest(root, e, ds, kind)
                E.vec_shards(root, e, ds, kind)
        rp, _ = ref_read(ds)
        with np.load(rp, allow_pickle=False) as r:
            qrows, top = r["rows"][:300].astype(np.int64), r["top20"][:300, :5]
        Dp = {e: np.concatenate([np.load(p, mmap_mode="r") for p in E.vec_shards(root, e, ds, "docs")]) for e in PARTS}
        Qp = {e: np.concatenate([np.load(p, mmap_mode="r") for p in E.vec_shards(root, e, ds, "queries")])[qrows] for e in PARTS}
        X, Q = fuse([Dp[e] for e in PARTS]), fuse([Qp[e] for e in PARTS])
        S = Q.astype(np.float32) @ X.astype(np.float32).T
        ours = np.argsort(-S, axis=1, kind="stable")[:, :5]
        same = float(np.mean([set(a.tolist()) == set(b.tolist()) for a, b in zip(ours, top)]))
        log(f"real {ds}: e3v's top 5 equals E3B's V-no-gamma top 5 on {same:.4f} of 300 questions")
        assert same >= 0.99, (ds, same)
        del Dp, Qp, X, Q, S
    print("zenc3v real self-test: OK")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zenc3v.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("vectors", "first"):
        p = sub.add_parser(name)
        p.add_argument("--ds", required=True, choices=DS)
        p.add_argument("--threads", type=int, default=0)
    p = sub.add_parser("gv")
    p.add_argument("--threads", type=int, default=0)
    p = sub.add_parser("selftest")
    p.add_argument("--real", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "vectors":
        return cmd_vectors(a)
    if a.cmd == "first":
        return cmd_first(a)
    if a.cmd == "gv":
        return cmd_gv(a)
    selftest_toy()
    if a.real:
        selftest_real()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
