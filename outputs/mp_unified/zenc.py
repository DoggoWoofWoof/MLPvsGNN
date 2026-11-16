"""ENC (docs/ENC_ENCODER_COMPARISON.md, D93): smaller encoders against the frozen gte-Qwen2-1.5B. Stage E0, the first stage.

    python outputs/mp_unified/zenc.py models                          # host: download and pin the new models
    python outputs/mp_unified/zenc.py g0 [--sample 32]                # laptop: the code reproduces the record's vectors
    python outputs/mp_unified/zenc.py encode --enc mpnet --ds squad   # host GPU: docs and queries, resumes by shard
    python outputs/mp_unified/zenc.py first --enc frozen --ds squad   # host CPU: s1eval top 1000 and hits (G0b on frozen)
    python outputs/mp_unified/zenc.py qtime --enc mpnet               # laptop CPU: one-query encode time (LP 2)
    python outputs/mp_unified/zenc.py report
    python outputs/mp_unified/zenc.py selftest

The text contract is the record's (page section 2): document row i = line i of nodes.jsonl, `text` verbatim; query row j =
query_ids.json[j], the j-th line of the split files in the queries manifest's order; each model's own query prompt, no
document prompt; L2-normalised float16 rows, 40,000 a shard. New vectors live under outputs/enc/<enc>/<ds>/{docs,queries};
the frozen arm reads the record's vectors in place.
"""
import os
import sys


def _threads():
    """The BLAS thread count, set before numpy loads (the search's sums depend on it): --threads N, default 8."""
    n = "8"
    if "--threads" in sys.argv:
        n = sys.argv[sys.argv.index("--threads") + 1]
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = n
    return int(n)


THREADS = _threads()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
WS = Path.cwd().resolve()
OUT = WS / "outputs" / "enc"
RECORD_SHA = "58958f33a3af74c21fa625ffd79da16c573cb5d2ba7e9745da0056b568591ab0"
SIX = ("squad", "musique", "hotpotqa", "2wiki", "metaqa", "webqsp")
PASSAGE = ("squad", "hotpotqa", "2wiki", "musique")
MULTI = ("2wiki", "hotpotqa", "musique")
SHARD = 40000
K = 1000
RRF_C = np.float32(60.0)
F1 = np.float32(1.0)
QB = 1024                                   # queries per GEMM block in the search
READ_FIT = "scr-gnx-nx-s0"                  # the record read whose s1eval ids and gold counts E0 uses
GTE_QINSTR = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
ENCODERS = {
    "frozen": {"repo": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "dim": 1536, "query": {"prompt": GTE_QINSTR},
               "trust": True, "max_seq_length": 32768, "budget": 32768,
               # its remote code calls DynamicCache.get_usable_length, gone from transformers 4.57; an encoder never
               # reads a cache, so the cache is off (one forward per batch, past length 0, as before)
               "config": {"use_cache": False}},
    "qwen3e06": {"repo": "Qwen/Qwen3-Embedding-0.6B", "dim": 1024, "query": {"prompt_name": "query"}, "budget": 49152},
    "mpnet": {"repo": "sentence-transformers/all-mpnet-base-v2", "dim": 768, "query": {}, "budget": 131072},
    "mpnetqa": {"repo": "sentence-transformers/multi-qa-mpnet-base-cos-v1", "dim": 768, "query": {}, "budget": 131072},
}
NEW = ("qwen3e06", "mpnet", "mpnetqa")
ALLOW = ["*.json", "*.txt", "*.safetensors", "*.model", "*.py", "1_Pooling/*", "2_Normalize/*", "2_Dense/*"]
IGNORE = ["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.ot", "*.msgpack"]
G0_MIN_COS = 0.99
G0B_SCORE_TOL = 0.002
G0B_SHARE = 0.995


def stop(msg):
    raise SystemExit(f"ZENC: {msg}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wjson(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(json.dumps(obj, indent=1).encode("utf-8"))
    os.replace(tmp, p)


# ── the substrate ─────────────────────────────────────────────────────────────

def served_root():
    """The CRAG package: ENC_SERVED, else the host mirror beside the workspace, else the laptop's Desktop copy."""
    cands = [os.environ.get("ENC_SERVED"), WS.parent / "mirror" / "CRAG" / "data" / "final_canonical",
             Path.home() / "Desktop" / "CRAG" / "data" / "final_canonical"]
    for c in cands:
        if c and (Path(c) / "CANONICAL_FREEZE.json").is_file():
            root = Path(c)
            got = json.loads((root / "CANONICAL_FREEZE.json").read_text(encoding="utf-8")).get("RECORD_SHA256")
            if got != RECORD_SHA:
                stop(f"the package at {root} is frozen as {got}, not {RECORD_SHA[:8]}...")
            return root
    stop("no CRAG package found (set ENC_SERVED)")


def manifest(root, ds, kind):
    return json.loads((root / ds / "embeddings" / "dense" / kind / "manifest.json").read_text(encoding="utf-8"))


def query_splits(man):
    """The split files' order: the bracketed list of quoted names in row_semantics (the text also holds query_ids.json[j])."""
    m = [x for x in re.findall(r"\[([^\]]*)\]", man["row_semantics"]) if "'" in x or '"' in x]
    if len(m) != 1:
        stop(f"no single split order in {man['row_semantics']!r}")
    return [s.strip().strip("'\"") for s in m[0].split(",") if s.strip()]


def query_texts(root, ds, rows=None):
    """{row: question} for the wanted rows (all by default); every line's query_id is checked against query_ids.json."""
    man = manifest(root, ds, "queries")
    qids = json.loads((root / ds / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    want = None if rows is None else {int(r) for r in rows}
    out, j = {}, 0
    for sp in query_splits(man):
        with open(root / ds / "queries" / f"{sp}.jsonl", encoding="utf-8") as f:
            for ln in f:
                if want is None or j in want:
                    r = json.loads(ln)
                    if r.get("query_id") != qids[j]:
                        stop(f"{ds} query row {j}: {r.get('query_id')!r} is not query_ids.json's {qids[j]!r}")
                    out[j] = r["question"]
                j += 1
    if j != len(qids) or j != man["n_rows"]:
        stop(f"{ds}: {j} query lines, {len(qids)} ids, manifest {man['n_rows']}")
    return out


def doc_chunks(root, ds, want_shard=lambda s: True):
    """(shard, texts or None) per 40,000 lines of nodes.jsonl; lines of unwanted shards are counted, not parsed."""
    buf, s = [], 0
    with open(root / ds / "nodes.jsonl", encoding="utf-8") as f:
        for ln in f:
            buf.append(ln)
            if len(buf) == SHARD:
                yield s, ([json.loads(x)["text"] for x in buf] if want_shard(s) else None)
                buf, s = [], s + 1
    if buf:
        yield s, ([json.loads(x)["text"] for x in buf] if want_shard(s) else None)


def doc_texts(root, ds, rows):
    want, out = {int(r) for r in rows}, {}
    with open(root / ds / "nodes.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i in want:
                out[i] = json.loads(ln)["text"]
    return out


def record_rows(root, ds, kind, rows):
    """The record's float16 rows at the given positions, read shard by shard (memory-mapped)."""
    base = root / ds / "embeddings" / "dense" / kind
    rows = np.asarray(rows, np.int64)
    out = np.empty((rows.size, ENCODERS["frozen"]["dim"]), np.float16)
    for s in np.unique(rows // SHARD):
        a = np.load(base / f"shard_{int(s):05d}.npy", mmap_mode="r")
        m = rows // SHARD == s
        out[m] = a[rows[m] - s * SHARD]
    return out


def vec_dir(root, enc, ds, kind):
    return (root / ds / "embeddings" / "dense" / kind) if enc == "frozen" else (OUT / enc / ds / kind)


def vec_shards(root, enc, ds, kind):
    d = vec_dir(root, enc, ds, kind)
    n = manifest(root, ds, kind)["n_shards"]
    paths = [d / f"shard_{s:05d}.npy" for s in range(n)]
    miss = [p.name for p in paths if not p.is_file()]
    if miss:
        stop(f"{enc} {ds} {kind}: {len(miss)} shards missing ({miss[:3]})")
    if enc != "frozen":
        mf = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if mf.get("n_rows") != manifest(root, ds, kind)["n_rows"]:
            stop(f"{enc} {ds} {kind}: manifest rows {mf.get('n_rows')} differ from the record's")
    return paths


# ── encoding ──────────────────────────────────────────────────────────────────

def pinned(enc):
    if enc == "frozen":
        return None
    p = OUT / "models.json"
    if not p.is_file():
        stop("outputs/enc/models.json missing: run `zenc.py models` first")
    return json.loads(p.read_text(encoding="utf-8"))[enc]["revision"]


def load_model(enc, device, revision=None):
    import torch
    from sentence_transformers import SentenceTransformer
    s = ENCODERS[enc]
    kw = {"device": device, "trust_remote_code": s.get("trust", False), "local_files_only": True}
    if revision:
        kw["revision"] = revision
    if device == "cuda":
        kw["model_kwargs"] = {"torch_dtype": torch.float16}
    if s.get("config"):
        kw["config_kwargs"] = dict(s["config"])
    m = SentenceTransformer(s["repo"], **kw)
    if s.get("max_seq_length"):
        m.max_seq_length = s["max_seq_length"]
    return m


def query_prompt(m, enc):
    q = ENCODERS[enc]["query"]
    if "prompt" in q:
        return q["prompt"]
    if "prompt_name" in q:
        if q["prompt_name"] not in (m.prompts or {}):
            stop(f"{enc}: no prompt named {q['prompt_name']!r} in its config ({sorted(m.prompts or {})})")
        return m.prompts[q["prompt_name"]]
    return ""


def encode_texts(m, enc, texts, kind, budget=None):
    """(rows x dim float16, stats): longest first, batches under a token budget; an out-of-memory halves the budget."""
    import torch
    n = len(texts)
    dim = ENCODERS[enc]["dim"]
    kw = dict(ENCODERS[enc]["query"]) if kind == "queries" else {}
    pre = query_prompt(m, enc) if kind == "queries" else ""
    lens = np.fromiter((len(x) for x in m.tokenizer([pre + t for t in texts], add_special_tokens=True,
                                                     truncation=False)["input_ids"]), np.int64, n)
    lim = int(m.max_seq_length)
    eff = np.minimum(lens, lim)
    order = np.argsort(-eff, kind="stable")
    out = np.empty((n, dim), np.float16)
    budget = int(budget or ENCODERS[enc]["budget"])
    i, ooms = 0, 0
    while i < n:
        b = max(1, min(512, budget // max(int(eff[order[i]]), 1)))
        idx = order[i:i + b]
        try:
            v = m.encode([texts[t] for t in idx], batch_size=len(idx), normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False, **kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            ooms += 1
            budget //= 2
            if budget < 256:
                stop(f"{enc}: out of memory even at a 256-token budget")
            continue
        if v.shape != (len(idx), dim):
            stop(f"{enc}: encode gave {v.shape}, want ({len(idx)}, {dim})")
        out[idx] = v.astype(np.float16)
        i += len(idx)
    return out, {"tokens": int(eff.sum()), "cut": int((lens > lim).sum()), "max_len": int(lens.max(initial=0)),
                 "limit": lim, "ooms": ooms, "budget_end": budget}


def damaged(v):
    """The record's dense_integrity rule: all-zero, non-finite, or |norm - 1| > 0.01."""
    x = np.asarray(v, np.float32)
    fin = np.isfinite(x).all(1)
    nrm = np.linalg.norm(np.where(np.isfinite(x), x, 0), axis=1)
    return int((~fin | (np.abs(x).max(1) == 0) | (np.abs(nrm - 1) > 0.01)).sum())


def cmd_models(a):
    from huggingface_hub import HfApi, snapshot_download
    api, rec = HfApi(), {}
    p = OUT / "models.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    for enc in NEW:
        repo = ENCODERS[enc]["repo"]
        rev = old.get(enc, {}).get("revision") or api.model_info(repo).sha
        names = [x.rfilename for x in (api.model_info(repo, revision=rev).siblings or [])]
        allow = ALLOW + ([] if "model.safetensors" in names else ["pytorch_model.bin"])
        t0 = time.time()
        path = Path(snapshot_download(repo, revision=rev, allow_patterns=allow, ignore_patterns=IGNORE))
        files = {str(f.relative_to(path)).replace("\\", "/"): {"bytes": f.stat().st_size, "sha256": sha256(f)}
                 for f in sorted(path.rglob("*")) if f.is_file()}
        rec[enc] = {"repo": repo, "revision": rev, "seconds": round(time.time() - t0, 1),
                    "bytes": sum(v["bytes"] for v in files.values()), "files": files}
        print(enc, repo, rev, rec[enc]["bytes"], "bytes", flush=True)
    for enc in NEW:          # each loads offline at its pin; parameters counted
        m = load_model(enc, "cpu", rec[enc]["revision"])
        rec[enc]["params"] = int(sum(t.numel() for t in m.parameters()))
        rec[enc]["max_seq_length"] = int(m.max_seq_length)
        rec[enc]["query_prompt"] = query_prompt(m, enc)
        rec[enc]["prompts"] = dict(m.prompts or {})
        rec[enc]["default_prompt_name"] = m.default_prompt_name
        if m.default_prompt_name:
            stop(f"{enc}: a default prompt ({m.default_prompt_name}) would prefix the documents")
        del m
    wjson(p, rec)
    print("wrote", p)


def cmd_g0(a):
    """The frozen model through encode_texts reproduces the record's rows (CPU; the laptop)."""
    import sentence_transformers
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    root = served_root()
    rev = Path(snapshot_download(ENCODERS["frozen"]["repo"], local_files_only=True)).name
    t0 = time.time()
    m = load_model("frozen", "cpu")
    rec = {"stage": "ENC G0", "page": "docs/ENC_ENCODER_COMPARISON.md section 3", "revision": rev,
           "params": int(sum(t.numel() for t in m.parameters())), "load_s": round(time.time() - t0, 1),
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "sentence_transformers": sentence_transformers.__version__},
           "sample": a.sample, "min_cos": G0_MIN_COS, "datasets": {}}
    ok = True
    for ds in SIX:
        rng = np.random.default_rng(0)
        r = {}
        for kind in ("docs", "queries"):
            n = manifest(root, ds, kind)["n_rows"]
            rows = np.sort(rng.choice(n, size=min(a.sample, n), replace=False))
            got = doc_texts(root, ds, rows) if kind == "docs" else query_texts(root, ds, rows)
            texts = [got[int(i)] for i in rows]
            t1 = time.time()
            v, st = encode_texts(m, "frozen", texts, kind)
            ref = record_rows(root, ds, kind, rows).astype(np.float32)
            cos = (v.astype(np.float32) * ref).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(v.astype(np.float32), axis=1))
            r[kind] = {"rows": rows.tolist(), "min_cos": round(float(cos.min()), 6), "mean_cos": round(float(cos.mean()), 6),
                       "seconds": round(time.time() - t1, 1), "max_len": st["max_len"], "cut": st["cut"]}
            ok &= bool(cos.min() >= G0_MIN_COS)
            print(ds, kind, r[kind]["min_cos"], r[kind]["mean_cos"], r[kind]["seconds"], "s", flush=True)
        rec["datasets"][ds] = r
    rec["call"] = "PASS" if ok else "FAIL"
    wjson(OUT / "g0.json", rec)
    print("G0", rec["call"])


def cmd_encode(a):
    import torch
    root = served_root()
    g0 = OUT / "g0.json"
    if not g0.is_file() or json.loads(g0.read_text(encoding="utf-8")).get("call") != "PASS":
        stop("G0 has not passed (outputs/enc/g0.json)")
    if a.enc == "frozen":
        stop("the frozen arm is the record's vectors; nothing to encode")
    rev = pinned(a.enc)
    dev = "cuda" if torch.cuda.is_available() else ("cpu" if a.cpu else None)
    if dev is None:
        stop("no GPU (pass --cpu to encode on the CPU)")
    t0 = time.time()
    m = load_model(a.enc, dev, rev)
    load_s = round(time.time() - t0, 1)
    if m.default_prompt_name:
        stop(f"{a.enc}: a default prompt would prefix the documents")
    for kind in a.kinds.split(","):
        rman = manifest(root, a.ds, kind)
        base = OUT / a.enc / a.ds / kind
        base.mkdir(parents=True, exist_ok=True)

        def done(s):
            jp, npy = base / f"shard_{s:05d}.json", base / f"shard_{s:05d}.npy"
            if not (jp.is_file() and npy.is_file()):
                return False
            return json.loads(jp.read_text(encoding="utf-8")).get("sha256") == sha256(npy)

        if kind == "docs":
            chunks = doc_chunks(root, a.ds, lambda s: not done(s))
        else:
            allq = query_texts(root, a.ds)
            qs = [allq[j] for j in range(len(allq))]
            chunks = ((s, None if done(s) else qs[s * SHARD:(s + 1) * SHARD]) for s in range(rman["n_shards"]))
        shards, rows = [], 0
        for s, texts in chunks:
            npy, jp = base / f"shard_{s:05d}.npy", base / f"shard_{s:05d}.json"
            if texts is None:
                r = json.loads(jp.read_text(encoding="utf-8"))
            else:
                t1 = time.time()
                v, st = encode_texts(m, a.enc, texts, kind)
                bad = damaged(v)
                if bad:
                    stop(f"{a.enc} {a.ds} {kind} shard {s}: {bad} damaged rows (G0c)")
                tmp = base / f"shard_{s:05d}.tmp.npy"
                np.save(tmp, v)
                os.replace(tmp, npy)
                r = {"shard": s, "rows": int(v.shape[0]), "file": npy.name, "bytes": npy.stat().st_size,
                     "sha256": sha256(npy), "seconds": round(time.time() - t1, 2), "device": dev, **st}
                wjson(jp, r)
                print(a.enc, a.ds, kind, s, r["rows"], r["seconds"], "s", "cut", r["cut"], flush=True)
            shards.append(r)
            rows += r["rows"]
        if rows != rman["n_rows"] or len(shards) != rman["n_shards"]:
            stop(f"{a.enc} {a.ds} {kind}: {rows} rows in {len(shards)} shards, the record has {rman['n_rows']} in "
                 f"{rman['n_shards']}")
        wjson(base / "manifest.json", {
            "dataset": a.ds, "kind": kind, "model": a.enc, "n_rows": rows, "shard_size": SHARD, "n_shards": len(shards),
            "row_semantics": rman["row_semantics"],
            "format": {"file": "shard_%05d.npy", "dtype": "float16", "shape": f"(rows, {ENCODERS[a.enc]['dim']})",
                       "normalised": "unit L2 (|norm-1| <= 0.01)"},
            "encoder": {"encoder": ENCODERS[a.enc]["repo"], "revision": rev, "dim": ENCODERS[a.enc]["dim"],
                        "dtype": "float16", "device": dev, "max_seq_length": int(m.max_seq_length),
                        "query_prompt": query_prompt(m, a.enc) if kind == "queries" else "",
                        "document_prompt": "", "load_s": load_s},
            "seconds": round(sum(r["seconds"] for r in shards), 1), "cut": int(sum(r["cut"] for r in shards)),
            "shards": [{k: r[k] for k in ("shard", "rows", "file", "bytes", "sha256")} for r in shards],
            "page": "docs/ENC_ENCODER_COMPARISON.md", "record_sha256": RECORD_SHA})
        print("manifest", base / "manifest.json", flush=True)


# ── the first stage ───────────────────────────────────────────────────────────

def read_path(ds):
    for base in (WS / "outputs" / "screen" / "fits", WS / "outputs" / "d83" / "rec" / "outputs" / "screen" / "fits"):
        p = base / READ_FIT / "reads" / f"{ds}__s1eval.npz"
        if p.is_file():
            return p
    stop(f"no record read {READ_FIT}/reads/{ds}__s1eval.npz")


def s1eval(ds):
    p = read_path(ds)
    z = np.load(p, allow_pickle=False)
    refs = [str(x) for x in z["refs"]]
    return {"path": str(p.relative_to(WS)).replace("\\", "/"), "sha256": sha256(p), "ids": [str(x) for x in z["ids"]],
            "gold_total": z["gold_total"].astype(np.int64), "rrf_top": z["ref_top"][refs.index("rrf")].astype(np.int64)}


def gold_sets(root, ds, ids, gold_total):
    """(flat positions, offsets) of each question's gold_node_ids; cached; the counts must equal the read's."""
    cp = OUT / "e0" / "gold" / f"{ds}.npz"
    if cp.is_file():
        z = np.load(cp, allow_pickle=False)
        if [str(x) for x in z["ids"]] == list(ids):
            return z["flat"], z["off"]
    want = set(ids)
    gid = {}
    man = manifest(root, ds, "queries")
    for sp in query_splits(man):
        with open(root / ds / "queries" / f"{sp}.jsonl", encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                if r["query_id"] in want:
                    g = r["gold_node_ids"]
                    gid[r["query_id"]] = list(g if isinstance(g, list) else json.loads(g))
    if len(gid) != len(want):
        stop(f"{ds}: {len(want) - len(gid)} s1eval ids not in the query files")
    need = {n for v in gid.values() for n in v}
    pos, pat = {}, re.compile(r'"node_id":\s*"([^"]+)"')
    with open(root / ds / "nodes.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            m = pat.search(ln)
            if m and m.group(1) in need:
                pos[m.group(1)] = i
    flat, off = [], [0]
    for q, t in zip(ids, gold_total):
        g = sorted({pos[n] for n in gid[q] if n in pos})
        if len(g) != int(t):
            stop(f"{ds} {q}: {len(g)} resolved golds, the read counts {int(t)}")
        flat += g
        off.append(len(flat))
    flat, off = np.asarray(flat, np.int64), np.asarray(off, np.int64)
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cp, ids=np.asarray(ids), flat=flat, off=off)
    return flat, off


def search(Q, shard_paths, k=K, probe=None):
    """Exact float32 inner product top k over the shards; ties to the lower id. probe (nq, m) ids: their scores too."""
    nq = Q.shape[0]
    bs = np.full((nq, k), -np.inf, np.float32)
    bi = np.full((nq, k), np.iinfo(np.int64).max, np.int64)
    ps = None if probe is None else np.full(probe.shape, np.nan, np.float32)
    off = 0
    for p in shard_paths:
        D = np.asarray(np.load(p, mmap_mode="r"), np.float32)
        n = D.shape[0]
        for q0 in range(0, nq, QB):
            q1 = min(nq, q0 + QB)
            S = Q[q0:q1] @ D.T
            if probe is not None:
                pr = probe[q0:q1]
                hit = (pr >= off) & (pr < off + n)
                r, c = np.nonzero(hit)
                ps[q0 + r, c] = S[r, pr[r, c] - off]
            kk = min(k, n)
            part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            cs = np.concatenate([bs[q0:q1], np.take_along_axis(S, part, 1)], 1)
            ci = np.concatenate([bi[q0:q1], part.astype(np.int64) + off], 1)
            keep = np.argpartition(-cs, k - 1, axis=1)[:, :k]
            bs[q0:q1] = np.take_along_axis(cs, keep, 1)
            bi[q0:q1] = np.take_along_axis(ci, keep, 1)
        off += n
    o = np.lexsort((bi, -bs), axis=1)
    return np.take_along_axis(bi, o, 1), np.take_along_axis(bs, o, 1), ps


def rrf_top(dense_ids, splade_ids, k=5):
    """The record's first stage per question: float32 1/(60 + rank) over each list (first occurrence), summed; by score,
    then node id."""
    nq = dense_ids.shape[0]
    out = np.full((nq, k), -1, np.int64)
    wd = F1 / (RRF_C + np.arange(1, dense_ids.shape[1] + 1, dtype=np.float32))
    wsp = F1 / (RRF_C + np.arange(1, splade_ids.shape[1] + 1, dtype=np.float32))
    for q in range(nq):
        sc = {}
        for ids, w in ((dense_ids[q], wd), (splade_ids[q], wsp)):
            seen = set()
            for i, g in enumerate(ids.tolist()):
                if g < 0 or g in seen:
                    continue
                seen.add(g)
                sc[g] = np.float32(sc.get(g, np.float32(0.0)) + w[i])
        top = sorted(sc.items(), key=lambda t: (-float(t[1]), t[0]))[:k]
        out[q, :len(top)] = [g for g, _ in top]
    return out


def hits(top, flat, off):
    """Golds of each question inside each row of top (nq, k)."""
    h = np.zeros(top.shape[0], np.int64)
    for q in range(top.shape[0]):
        g = flat[off[q]:off[q + 1]]
        if g.size:
            h[q] = np.isin(top[q], g).sum()
    return h


def cmd_first(a):
    root = served_root()
    t0 = time.time()
    s1 = s1eval(a.ds)
    ids, gt = s1["ids"], s1["gold_total"]
    qids = json.loads((root / a.ds / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    qmap = {q: j for j, q in enumerate(qids)}
    rows = np.asarray([qmap[q] for q in ids], np.int64)
    flat, off = gold_sets(root, a.ds, ids, gt)
    qpaths = vec_shards(root, a.enc, a.ds, "queries")
    Q = np.empty((rows.size, ENCODERS[a.enc]["dim"]), np.float32)
    for s in np.unique(rows // SHARD):
        A = np.load(qpaths[int(s)], mmap_mode="r")
        m = rows // SHARD == s
        Q[m] = A[rows[m] - s * SHARD]
    rc = root / a.ds / "retrieval_cache"
    rec_d = np.load(rc / "dense_top1000.npz", mmap_mode="r")
    rec_ids = np.asarray(rec_d["ids"][rows], np.int64)
    rec_sc = np.asarray(rec_d["scores"][rows], np.float32)
    spl = np.asarray(np.load(rc / "splade_top1000.npz", mmap_mode="r")["ids"][rows], np.int64)
    t1 = time.time()
    top_i, top_s, probe = search(Q, vec_shards(root, a.enc, a.ds, "docs"), K, rec_ids if a.enc == "frozen" else None)
    search_s = round(time.time() - t1, 1)
    r5 = rrf_top(top_i, spl, 5)
    h = {"d5": hits(top_i[:, :5], flat, off), "d20": hits(top_i[:, :20], flat, off), "d1000": hits(top_i, flat, off),
         "rrf5": hits(r5, flat, off)}
    gtf = gt.astype(np.float64)
    rec = {"stage": "ENC E0", "page": "docs/ENC_ENCODER_COMPARISON.md section 4", "enc": a.enc, "dataset": a.ds,
           "questions": len(ids), "read": {"path": s1["path"], "sha256": s1["sha256"]}, "threads": THREADS,
           "search_s": search_s, "seconds": None,
           "recall": {k: round(float(np.mean(np.minimum(v, gt) / gtf)), 6) for k, v in h.items()}}
    if a.enc == "frozen":
        rr5 = rrf_top(rec_ids, spl, 5)
        rh = hits(rr5, flat, off)
        diff = np.abs(probe - rec_sc)
        t5 = np.mean([set(rec_ids[q, :5]) <= set(top_i[q, :10]) and set(top_i[q, :5]) <= set(rec_ids[q, :10])
                      for q in range(len(ids))])
        same = float(np.mean(rh == s1["rrf_top"]))
        g = {"i_score_max_abs": round(float(np.nanmax(diff)), 6), "i_score_nan": int(np.isnan(probe).sum()),
             "i_top5_in_top10_share": round(float(t5), 6), "ii_rrf_hits_equal_share": round(same, 6),
             "ii_rrf_r5_record": round(float(np.mean(np.minimum(s1["rrf_top"], gt) / gtf)), 6),
             "ii_rrf_r5_from_cache": round(float(np.mean(np.minimum(rh, gt) / gtf)), 6),
             "iii_rrf_r5_own_minus_record": round(float(np.mean((np.minimum(h["rrf5"], gt) - np.minimum(s1["rrf_top"], gt)) / gtf)), 6)}
        g["call"] = "PASS" if (g["i_score_nan"] == 0 and g["i_score_max_abs"] <= G0B_SCORE_TOL
                               and t5 >= G0B_SHARE and same >= G0B_SHARE) else "FAIL"
        rec["g0b"] = g
        print("G0b", a.ds, json.dumps(g), flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    d = OUT / "e0" / a.enc
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{a.ds}.npz", ids=np.asarray(ids), rows=rows, top_ids=top_i.astype(np.int32),
             top_scores=top_s.astype(np.float16), gold_total=gt, **{f"h_{k}": v.astype(np.int16) for k, v in h.items()})
    rec["npz_sha256"] = sha256(d / f"{a.ds}.npz")
    wjson(d / f"{a.ds}.json", rec)
    print(a.enc, a.ds, json.dumps(rec["recall"]), rec["seconds"], "s", flush=True)
    if a.enc == "frozen" and rec["g0b"]["call"] != "PASS":
        stop(f"G0b FAILED on {a.ds}")


# ── timing and report ─────────────────────────────────────────────────────────

def cmd_qtime(a):
    """Median one-query encode time on the laptop CPU, one thread, pinned to LP 2 where psutil allows."""
    import torch
    torch.set_num_threads(1)
    try:
        import psutil
        psutil.Process().cpu_affinity([2])
        pin = "LP 2"
    except Exception as e:  # noqa: BLE001
        pin = f"not pinned ({type(e).__name__})"
    root = served_root()
    s1 = s1eval("hotpotqa")
    qids = json.loads((root / "hotpotqa" / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    qmap = {q: j for j, q in enumerate(qids)}
    rows = [qmap[q] for q in s1["ids"][:a.n]]
    qt = query_texts(root, "hotpotqa", rows)
    rev = None
    if a.enc != "frozen" and (OUT / "models.json").is_file():
        rev = pinned(a.enc)
    m = load_model(a.enc, "cpu", rev)
    kw = dict(ENCODERS[a.enc]["query"])
    for r in rows[:5]:
        m.encode([qt[r]], normalize_embeddings=True, show_progress_bar=False, **kw)
    ts = []
    for r in rows:
        t = time.perf_counter()
        m.encode([qt[r]], normalize_embeddings=True, show_progress_bar=False, **kw)
        ts.append(time.perf_counter() - t)
    rec = {"enc": a.enc, "n": len(ts), "median_ms": round(1000 * float(np.median(ts)), 2),
           "p90_ms": round(1000 * float(np.percentile(ts, 90)), 2), "pin": pin, "threads": 1, "revision": rev}
    wjson(OUT / "qtime" / f"{a.enc}.json", rec)
    print(json.dumps(rec))


def cmd_report(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import zap1
    rows, calls, miss = [], {}, []
    base = {}
    for ds in SIX:
        p = OUT / "e0" / "frozen" / f"{ds}.npz"
        if p.is_file():
            base[ds] = np.load(p, allow_pickle=False)
        else:
            miss.append(f"frozen/{ds}")
    gap = {}
    for enc in NEW:
        for ds in SIX:
            p = OUT / "e0" / enc / f"{ds}.npz"
            if not p.is_file() or ds not in base:
                miss.append(f"{enc}/{ds}")
                continue
            z, b = np.load(p, allow_pickle=False), base[ds]
            if list(z["ids"]) != list(b["ids"]):
                stop(f"{enc} {ds}: question order differs from the frozen arm's")
            gt = b["gold_total"].astype(np.float64)
            nq = gt.size
            for col in ("d5", "d20", "d1000", "rrf5"):
                x = np.minimum(z[f"h_{col}"], b["gold_total"]) / gt
                y = np.minimum(b[f"h_{col}"], b["gold_total"]) / gt
                pt, lo, hi = zap1.boot_diff(x, y, np.arange(nq), nq)
                gap[(enc, ds, col)] = pt
                rows.append({"enc": enc, "dataset": ds, "column": col, "frozen": round(float(y.mean()), 4),
                             "new": round(float(x.mean()), 4), "diff": round(pt, 4), "ci": [round(lo, 4), round(hi, 4)],
                             "call": zap1.call(lo, hi)})
    call_of = {(r["enc"], r["dataset"], r["column"]): r["call"] for r in rows}
    if not miss:
        x1 = [call_of[(e, d, "d5")] == "BELOW" for e in NEW for d in MULTI]
        x2 = [abs(gap[(e, d, "rrf5")]) < abs(gap[(e, d, "d5")]) for e in NEW for d in PASSAGE]
        x3 = [abs(gap[(e, d, "d1000")]) < abs(gap[(e, d, "d5")]) for e in NEW for d in PASSAGE]
        for name, v in (("X1", x1), ("X2", x2), ("X3", x3)):
            calls[name] = {"held": v, "call": "HELD" if all(v) else "NOT_HELD"}
        d94 = {e: ("OUT" if sum(call_of[(e, d, "rrf5")] == "BELOW" and gap[(e, d, "rrf5")] < -0.10 for d in SIX) >= 3
                   else "E1") for e in NEW}
        calls["D94_default"] = d94
    g0b = {ds: json.loads((OUT / "e0" / "frozen" / f"{ds}.json").read_text(encoding="utf-8")).get("g0b", {}).get("call")
           for ds in SIX if (OUT / "e0" / "frozen" / f"{ds}.json").is_file()}
    cost = {}
    for enc in NEW:
        secs, docs = 0.0, 0
        for ds in SIX:
            mp = OUT / enc / ds / "docs" / "manifest.json"
            if mp.is_file():
                mf = json.loads(mp.read_text(encoding="utf-8"))
                secs, docs = secs + mf["seconds"], docs + mf["n_rows"]
        qt = OUT / "qtime" / f"{enc}.json"
        mj = json.loads((OUT / "models.json").read_text(encoding="utf-8")) if (OUT / "models.json").is_file() else {}
        cost[enc] = {"params": mj.get(enc, {}).get("params"), "bytes_per_row": 2 * ENCODERS[enc]["dim"],
                     "gpu_s_per_million_docs": round(1e6 * secs / docs, 1) if docs else None,
                     "query_ms_cpu": json.loads(qt.read_text(encoding="utf-8"))["median_ms"] if qt.is_file() else None}
    rep = {"stage": "ENC E0", "page": "docs/ENC_ENCODER_COMPARISON.md", "missing": miss, "g0b": g0b,
           "rows": rows, "expectations": calls, "cost": cost}
    wjson(OUT / "e0_report.json", rep)
    lines = ["# ENC E0 report", "", f"missing: {miss}", f"G0b: {g0b}", "",
             "| enc | dataset | column | frozen | new | diff [95%] | call |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['enc']} | {r['dataset']} | {r['column']} | {r['frozen']} | {r['new']} | "
                     f"{r['diff']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['call']} |")
    lines += ["", "expectations: " + json.dumps(calls), "cost: " + json.dumps(cost)]
    (OUT / "e0_report.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print("\n".join(lines[:4]), "\n", json.dumps(calls))


# ── self-test ─────────────────────────────────────────────────────────────────

def cmd_selftest(a):
    rs = ("row j is the dense vector of query j == queries/query_ids.json[j] == the j-th line of the split files "
          "['train', 'dev', 'test'] concatenated in that order")              # the package's own wording (musique)
    assert query_splits({"row_semantics": rs}) == ["train", "dev", "test"]
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        n, dim, nq = 2500, 16, 7
        D = rng.standard_normal((n, dim)).astype(np.float32)
        D[100] = D[7]                                   # a duplicate: ties go to the lower id
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        paths = []
        for s in range(0, n, 1000):
            p = Path(td) / f"shard_{s // 1000:05d}.npy"
            np.save(p, D[s:s + 1000].astype(np.float16))
            paths.append(p)
        Q = rng.standard_normal((nq, dim)).astype(np.float32)
        Q[0] = D[7]
        Df = np.concatenate([np.load(p).astype(np.float32) for p in paths])
        probe = rng.integers(0, n, size=(nq, 30))
        ti, ts, ps = search(Q, paths, 50, probe)
        S = Q @ Df.T
        for q in range(nq):
            ref = np.lexsort((np.arange(n), -S[q]))[:50]
            assert list(ti[q]) == list(ref), (q, ti[q][:5], ref[:5])
        assert np.allclose(ps, np.take_along_axis(S, probe, 1))
        assert list(ti[0][:2]) == [7, 100], ti[0][:3]
        d_ids = np.array([[5, 3, 9, 1], [2, 4, 6, 8]])
        s_ids = np.array([[3, 5, 1, -1], [8, 7, 7, 2]])
        r = rrf_top(d_ids, s_ids, 3)
        assert list(r[0]) == [3, 5, 1], r[0]            # 3 and 5 tie at 1/61 + 1/62: the lower id first
        assert list(r[1]) == [2, 8, 4], r[1]            # 2: 1/61 + 1/64; 8: 1/64 + 1/61 (tie, 2 first); 4: 1/62
        flat, off = np.array([3, 9, 8]), np.array([0, 2, 3])
        assert list(hits(r, flat, off)) == [1, 1]
        v = np.zeros((3, 4), np.float16)
        v[0, 0] = 1
        v[1, :] = 0.4                                   # norm 0.8
        v[2, 0] = np.nan
        assert damaged(v) == 2
        assert query_splits({"row_semantics": "row j ... the split files ['train', 'dev'] concatenated"}) == ["train", "dev"]
    print("zenc selftest: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["models", "g0", "encode", "first", "qtime", "report", "selftest"])
    ap.add_argument("--enc", choices=sorted(ENCODERS))
    ap.add_argument("--ds", choices=SIX)
    ap.add_argument("--kinds", default="queries,docs")
    ap.add_argument("--sample", type=int, default=32)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.cmd in ("encode", "first", "qtime") and not a.enc:
        stop(f"{a.cmd} needs --enc")
    if a.cmd in ("encode", "first") and not a.ds:
        stop(f"{a.cmd} needs --ds")
    {"models": cmd_models, "g0": cmd_g0, "encode": cmd_encode, "first": cmd_first, "qtime": cmd_qtime,
     "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
