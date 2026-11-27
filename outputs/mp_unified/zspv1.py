"""SPV1 (docs/SPV1_SPLADE_VECTORS.md): the best SPLADE leg's vectors of every node and every question of every dataset, stored sparse.

    python -E -P -B outputs/mp_unified/zspv1.py selftest                         # laptop, no model
    python -I -B    outputs/mp_unified/zspv1.py encode --ds D [--smoke N] [--min-free-gb 25]   # host card: gate G-V, questions, then node shards (resumable)
    python -I -B    outputs/mp_unified/zspv1.py check --ds D                     # shards against the manifest (no model)
    python -I -B    outputs/mp_unified/zspv1.py replay --ds stark_prime|stark_mag|stark_amazon [--questions 64]   # information: stored vectors against the filed list

The leg is C0 (naver/splade-cocondenser-ensembledistil, outputs/spl1/models/c0) under SPL1 A11's pick l5 (half-precision weights, forward passes of 32 rows,
log1p(relu(max over positions of the raw logits))), the same numbers as the record's pooling in real arithmetic (zspl1l.py; STK1 A8's leg). Texts come from the
dataset's world (zworld: nodes.jsonl line i is node i; queries in row order over every split). Batches are the record's 64 rows from index 0 (padding is pooled, so a
vector depends on its batch): a shard holds 65,536 rows, a multiple of 64 and of the 2,048-row block. Storage per shard: indptr int64, indices uint16 (the
vocabulary is 30,522), data float16. Edits to this module only through an amendment on its page.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import zspl1 as S  # noqa: E402
import zspl1c as C  # noqa: E402
import zspl1l as L  # noqa: E402

log, stop = S.log, S.stop
WS = S.WS
OUT = WS / "outputs" / "spv1"
DOC = "docs/SPV1_SPLADE_VECTORS.md"
ARM = "l5"
SHARD = 65536
GATE_N = 2048
MAX_COL = 65535
DATASETS = ("multihoprag", "squad", "metaqa", "musique", "stark_prime", "stark_amazon", "stark_mag", "webqsp", "hotpotqa", "2wiki")
STARK = {"stark_prime": "prime", "stark_mag": "mag", "stark_amazon": "amazon"}


# ── pure helpers (self-tested) ───────────────────────────────────────────────

def pack(m):
    """A scipy CSR of float32 -> the stored arrays."""
    m = m.tocsr()
    m.sort_indices()
    if m.shape[1] - 1 > MAX_COL:
        stop(f"vocabulary {m.shape[1]} does not fit uint16")
    return {"indptr": m.indptr.astype(np.int64), "indices": m.indices.astype(np.uint16), "data": m.data.astype(np.float16), "n_cols": np.int64(m.shape[1])}


def unpack(z):
    import scipy.sparse as sp
    return sp.csr_matrix((np.asarray(z["data"], np.float32), np.asarray(z["indices"], np.int32), np.asarray(z["indptr"], np.int64)),
                         shape=(len(z["indptr"]) - 1, int(z["n_cols"])))


def shard_ranges(n, shard=SHARD):
    return [(i * shard, min(n, (i + 1) * shard)) for i in range((n + shard - 1) // shard)]


def gate_pass(m):
    """SPL1 A11's E2 / STK1 A8's G-L on one set: the first percentile of the per-vector cosine at least 0.999, the non-zeros within 1.0 percent."""
    return bool(m["cos_p1"] >= L.COS_P1_MIN and abs(m["nnz_ratio"] - 1.0) <= L.NNZ_TOL)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def valid_shard(path, rows):
    if not Path(path).is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return len(z["indptr"]) - 1 == rows and int(z["indptr"][-1]) == len(z["indices"]) == len(z["data"])
    except Exception:
        return False


def set_agree(a, b, k):
    a, b = np.asarray(a)[:, :k], np.asarray(b)[:, :k]
    return float(np.mean([set(x.tolist()) == set(y.tolist()) for x, y in zip(a, b)]))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_npz(path, **arrays):
    """Atomic: write beside, hash the bytes, replace. Returns the sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    sha = sha_file(tmp)
    S.replace_retry(tmp, path)
    return sha


# ── encode ───────────────────────────────────────────────────────────────────

def encode_texts(torch, tok, models, device, texts, arm=ARM):
    """The vectors of `texts` in blocks of C.BLOCK rows (batches of C.BATCH from the first row), as one CSR (float32)."""
    import scipy.sparse as sp
    parts = []
    for i in range(0, len(texts), C.BLOCK):
        parts.append(L.run_encode(torch, tok, models, device, texts[i:i + C.BLOCK], arm)[0])
    return sp.vstack(parts).tocsr() if parts else sp.csr_matrix((0, 30522), dtype=np.float32)


def run_gate(torch, tok, models, device, ds, ntexts, qtexts, d):
    gp = d / "gate.json"
    if gp.is_file():
        g = json.loads(gp.read_text(encoding="utf-8"))
        if g.get("verdict") == "PASS":
            return g
    rec = {}
    ok = True
    for name, xs in (("nodes", ntexts[:GATE_N]), ("questions", qtexts[:GATE_N])):
        A = encode_texts(torch, tok, models, device, xs, ARM)
        z = pack(A)
        A16 = unpack(z)                                   # the stored round trip (uint16 ids, float16 values)
        B = encode_texts(torch, tok, models, device, xs, "l0")
        m = L.compare(A16, B)
        good = gate_pass(m)
        ok = ok and good
        rec[name] = {k: round(float(v), 6) for k, v in m.items()} | {"texts": len(xs), "pass": good}
        log(f"G-V {ds} {name}: {len(xs)} texts, cosine p1 {m['cos_p1']:.6f}, non-zeros {m['nnz_ratio']:.5f} -> {'PASS' if good else 'FAIL'}")
    g = {"stage": "SPV1 gate G-V", "declared_in": DOC, "dataset": ds, "arm": ARM, "reads": rec, "verdict": "PASS" if ok else "FAIL", "utc": S.utc() if hasattr(S, "utc") else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    S.wjson(gp, g)
    return g


def cmd_encode(a):
    import zworld
    t0 = time.time()
    ds = a.ds
    w = zworld.load(ds)
    n = int(w.n)
    d = OUT / ds if not a.smoke else OUT / f"{ds}_smoke"
    d.mkdir(parents=True, exist_ok=True)
    device = a.device
    torch, tok, models, tfv = L.load_models(device, ["l0", ARM])
    qtexts = [q["question"] for q in w.iter_queries(resolve=False)]
    if a.smoke:
        qtexts = qtexts[:a.smoke]
    it = (t for _, t in w.iter_nodes())
    first = list(islice(it, GATE_N if not a.smoke else min(GATE_N, a.smoke)))
    g = run_gate(torch, tok, models, device, ds, first, qtexts, d)
    if g["verdict"] != "PASS":
        log(f"{ds}: G-V FAIL: nothing is stored")
        return 3
    it = (t for _, t in w.iter_nodes())                     # the stream again, from line 0
    nn = a.smoke if a.smoke else n
    qp = d / "queries.npz"
    qrec = None
    if not valid_shard(qp, len(qtexts)):
        t1 = time.time()
        Q = encode_texts(torch, tok, models, device, qtexts)
        z = pack(Q)
        sha = write_npz(qp, **z)
        qrec = {"file": qp.name, "rows": len(qtexts), "nnz_mean": round(float(Q.nnz / max(len(qtexts), 1)), 2), "sha256": sha, "seconds": round(time.time() - t1, 1)}
        log(f"{ds}: questions {len(qtexts):,} in {qrec['seconds']} s, nnz mean {qrec['nnz_mean']}")
    else:
        with np.load(qp, allow_pickle=False) as z:
            qrec = {"file": qp.name, "rows": len(qtexts), "nnz_mean": round(float(len(z["indices"]) / max(len(qtexts), 1)), 2), "sha256": sha_file(qp), "seconds": None}
        log(f"{ds}: questions already stored ({len(qtexts):,})")
    rngs = shard_ranges(nn)
    shards, done_now, nnz_all, enc_s = [], 0, 0, 0.0
    h = hashlib.sha256()
    for k, (lo, hi) in enumerate(rngs):
        texts = list(islice(it, hi - lo))
        if len(texts) != hi - lo:
            stop(f"{ds}: the node stream ended at {lo + len(texts)} of {nn}")
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\n")
        sp_ = d / f"shard_{k:05d}.npz"
        if valid_shard(sp_, hi - lo):
            with np.load(sp_, allow_pickle=False) as z:
                nz = int(len(z["indices"]))
            shards.append({"file": sp_.name, "rows": hi - lo, "nnz": nz, "sha256": sha_file(sp_)})
            nnz_all += nz
            continue
        free = shutil.disk_usage(d).free / 1e9
        if free < a.min_free_gb:
            log(f"{ds}: {free:.1f} GB free is below {a.min_free_gb} GB: stopped before shard {k} (resumable)")
            return 4
        t1 = time.time()
        M = encode_texts(torch, tok, models, device, texts)
        z = pack(M)
        sha = write_npz(sp_, **z)
        el = time.time() - t1
        enc_s += el
        done_now += hi - lo
        shards.append({"file": sp_.name, "rows": hi - lo, "nnz": int(M.nnz), "sha256": sha})
        nnz_all += int(M.nnz)
        log(f"{ds}: shard {k + 1}/{len(rngs)} ({hi:,}/{nn:,} nodes) {el:.0f} s, {(hi - lo) / max(el, 1e-9):.0f} nodes/s, nnz mean {M.nnz / (hi - lo):.1f}")
    man = {"stage": "SPV1 encode", "declared_in": DOC, "dataset": ds, "smoke": a.smoke or None, "arm": ARM, "encoder": "naver/splade-cocondenser-ensembledistil (C0)",
           "weights_sha256": S.sha256(C.DEST / "model.safetensors"), "n_nodes": nn, "world_nodes_sha256": w.nsha, "texts_sha256": h.hexdigest(), "shard_rows": SHARD,
           "shards": shards, "nnz_total": nnz_all, "nnz_mean": round(nnz_all / max(nn, 1), 2), "bytes_stored": int(sum((Path(d) / s["file"]).stat().st_size for s in shards)),
           "queries": qrec, "gate": {k: v for k, v in g["reads"].items()}, "nodes_per_s_this_run": round(done_now / max(enc_s, 1e-9), 1) if done_now else None,
           "seconds_this_run": round(time.time() - t0, 1), "versions": {"torch": torch.__version__, "transformers": tfv},
           "code_sha256": {"outputs/mp_unified/zspv1.py": sha_file(Path(__file__)), "outputs/mp_unified/zspl1l.py": sha_file(Path(L.__file__))},
           "device": device, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    S.wjson(d / "manifest.json", man)
    log(f"{ds}: done: {nn:,} nodes, nnz mean {man['nnz_mean']}, {man['bytes_stored'] / 1e9:.2f} GB stored, {man['seconds_this_run']} s this run")
    return 0


# ── check and replay ─────────────────────────────────────────────────────────

def cmd_check(a):
    d = OUT / a.ds
    mp = d / "manifest.json"
    if not mp.is_file():
        stop(f"{a.ds}: no manifest")
    man = json.loads(mp.read_text(encoding="utf-8"))
    rows = 0
    for s in man["shards"]:
        p = d / s["file"]
        if sha_file(p) != s["sha256"]:
            stop(f"{a.ds}: {s['file']} is not its record's")
        with np.load(p, allow_pickle=False) as z:
            if len(z["indptr"]) - 1 != s["rows"] or (len(z["indices"]) and int(z["indices"].max()) >= int(z["n_cols"])):
                stop(f"{a.ds}: {s['file']} is damaged")
        rows += s["rows"]
    if rows != man["n_nodes"]:
        stop(f"{a.ds}: shards hold {rows} rows, the manifest {man['n_nodes']}")
    if sha_file(d / man["queries"]["file"]) != man["queries"]["sha256"]:
        stop(f"{a.ds}: queries.npz is not its record's")
    log(f"{a.ds}: check PASS ({len(man['shards'])} shards, {rows:,} nodes, {man['bytes_stored'] / 1e9:.2f} GB)")
    return 0


def cmd_replay(a):
    """Information only (no gate): the first N test questions' top lists from the stored vectors against the filed SPLADE list (first.npz top_splade)."""
    import scipy.sparse as sp
    d = OUT / a.ds
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    kb = STARK[a.ds]
    with np.load(WS / "outputs" / "stark" / kb / "first.npz") as z:
        rows, filed = z["rows"].astype(np.int64)[:a.questions], z["top_splade"][:a.questions].astype(np.int64)
    with np.load(d / man["queries"]["file"], allow_pickle=False) as z:
        Q = unpack(z)[rows]
    K = filed.shape[1]
    top_s = np.full((len(rows), K), -np.inf, np.float32)
    top_i = np.full((len(rows), K), -1, np.int64)
    off = 0
    QT = Q.T.tocsc()
    for s in man["shards"]:
        with np.load(d / s["file"], allow_pickle=False) as z:
            D = unpack(z)
        top_s, top_i = S.merge_top(top_s, top_i, np.asarray((D @ QT).T.todense()), off, K)
        off += s["rows"]
    got = np.where(np.isfinite(top_s), top_i, -1)
    rec = {"stage": "SPV1 replay (information)", "declared_in": DOC, "dataset": a.ds, "questions": int(len(rows)), "top": int(K),
           "set_agreement": {str(k): round(set_agree(got, filed, k), 4) for k in (5, 20, 100, 1000)},
           "overlap_at_20_mean": round(float(np.mean([len(set(x[:20].tolist()) & set(y[:20].tolist())) / 20 for x, y in zip(got, filed)])), 4),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    S.wjson(d / "replay.json", rec)
    log(f"{a.ds}: replay {rec['questions']} questions: set agreement {rec['set_agreement']}, mean overlap at 20 {rec['overlap_at_20_mean']}")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def selftest():
    import scipy.sparse as sp
    ok = 0

    def chk(name, cond):
        nonlocal ok
        if not cond:
            raise SystemExit(f"zspv1: selftest FAILED ({name})")
        ok += 1
    chk("shard is a multiple of the batch and the block", SHARD % C.BATCH == 0 and SHARD % C.BLOCK == 0)
    r = shard_ranges(150000)
    chk("shard ranges cover once", r[0] == (0, 65536) and r[-1] == (131072, 150000) and sum(h - l for l, h in r) == 150000 and shard_ranges(0) == [])
    rng = np.random.default_rng(3)
    A = sp.random(37, 30522, density=0.006, format="csr", random_state=4, dtype=np.float32)
    A.data = (np.abs(A.data) * 5 + 0.01).astype(np.float32)
    z = pack(A)
    chk("dtypes", z["indices"].dtype == np.uint16 and z["data"].dtype == np.float16 and z["indptr"].dtype == np.int64)
    B = unpack(z)
    chk("round trip shape and pattern", B.shape == A.shape and np.array_equal(B.indptr, A.indptr) and np.array_equal(B.indices, A.indices))
    chk("round trip values within fp16", float(np.max(np.abs(B.data - A.data) / np.maximum(A.data, 1e-6))) < 1.0e-3)
    m = L.compare(B, A)
    chk("gate passes the fp16 round trip", gate_pass(m) and m["cos_p1"] > 0.9999)
    C2 = A.copy()
    C2.data = C2.data * rng.uniform(0.2, 2.0, C2.data.shape).astype(np.float32)
    chk("gate fails a damaged copy", not gate_pass(L.compare(C2, A)))
    chk("gate fails a non-zero drift", not gate_pass({"cos_p1": 1.0, "nnz_ratio": 0.97}) and gate_pass({"cos_p1": 0.9995, "nnz_ratio": 1.005}))
    import tempfile
    t = Path(tempfile.mkdtemp())
    sha = write_npz(t / "s.npz", **z)
    chk("write is atomic and hashed", (t / "s.npz").is_file() and not (t / "s.npz.part").exists() and sha == sha_file(t / "s.npz"))
    chk("valid shard by rows", valid_shard(t / "s.npz", 37) and not valid_shard(t / "s.npz", 38) and not valid_shard(t / "none.npz", 1))
    chk("set agreement", set_agree([[1, 2, 3], [4, 5, 6]], [[3, 2, 1], [4, 5, 7]], 3) == 0.5)
    chk("sha helper", sha_bytes(b"a") == hashlib.sha256(b"a").hexdigest())
    chk("datasets are the ten of the registry", len(DATASETS) == 10 and len(set(DATASETS)) == 10)
    log(f"zspv1 selftest PASS ({ok} checks)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    e = sub.add_parser("encode")
    e.add_argument("--ds", required=True, choices=DATASETS)
    e.add_argument("--smoke", type=int, default=0)
    e.add_argument("--device", default="cuda")
    e.add_argument("--min-free-gb", type=float, default=25.0)
    c = sub.add_parser("check")
    c.add_argument("--ds", required=True, choices=DATASETS)
    r = sub.add_parser("replay")
    r.add_argument("--ds", required=True, choices=sorted(STARK))
    r.add_argument("--questions", type=int, default=64)
    a = ap.parse_args()
    return {"selftest": lambda: selftest(), "encode": lambda: cmd_encode(a), "check": lambda: cmd_check(a), "replay": lambda: cmd_replay(a)}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
