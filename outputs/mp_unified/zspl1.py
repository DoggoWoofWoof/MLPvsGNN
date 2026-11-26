"""SPL1 (docs/SPL1_SPARSE_LEG_SCREEN.md): which smaller sparse encoder may replace the record's SPLADE leg.

    python -B -E -P outputs/mp_unified/zspl1.py selftest                     # laptop: -E -P (transformers needs the user site's `packaging`)
    python -I -B   outputs/mp_unified/zspl1.py fetch  --key c1|c2|c3         # public file list at a pinned revision, allow-listed, own empty folder
    python -I -B   outputs/mp_unified/zspl1.py check  --key c1               # the library's own recipe: batch-invariance, the 256 cut, the query route
    python -I -B   outputs/mp_unified/zspl1.py encode --key c1 --ds squad    # nodes in blocks of 2,048, checkpoint after each, top-1000 per question
    python -I -B   outputs/mp_unified/zspl1.py read   [--ds squad musique multihoprag]   # E0's measures + paired bootstrap against C0, the call, the cost line

Run from the workspace root. The model's own recipe is the sentence-transformers SparseEncoder of the fetched folder (activation, pooling over the
attention-masked positions, the inference-free query route of C3), cut at 256 tokens like the record. Nothing is tuned. C0 is the record's own
`retrieval_cache/splade_top1000.npz` for the same questions, not re-run. Reads: zenc's reader (E0: rrf_top, hits, recall = min(hits, gold_total) /
gold_total), zworld's loader. No file is written outside outputs/spl1/. Fetched files are untrusted: safetensors and small json/txt only, never a
pickle or a .py, each model in its own new empty folder, loaded with trust_remote_code=False and local_files_only.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
OUT = WS / "outputs" / "spl1"
DOC = "docs/SPL1_SPARSE_LEG_SCREEN.md"
TOPK, MAXLEN, BLOCK, BATCH, QBATCH = 1000, 256, 2048, 16, 64   # A3: micro-batch 16 (the card has about 6 GB free beside the other account's job)
NB = 1000                                             # bootstrap resamples, seed 0 (the screen's rule)
LEVEL_PTS = -1.0                                      # LEVEL: the interval's low end is at least this many points
SCREEN = ("squad", "musique", "multihoprag")
ORDER = ("c3", "c1", "c2")                            # fewest parameters first: the pick is the first that passes
MODELS = {
    "c1": {"repo": "ibm-granite/granite-embedding-30m-sparse", "rev": "ad82b1fd09541c998c8d45045d601c51fdb8a9b7", "params_m": 30, "cap_mb": 150},
    "c2": {"repo": "opensearch-project/opensearch-neural-sparse-encoding-v2-distill", "rev": "269e6638b2c4f648996691f6d751495285d8f330",
           "params_m": 67, "cap_mb": 400},
    "c3": {"repo": "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini", "rev": "4af867a426867dfdd744097531046f4289a32fdd",
           "params_m": 23, "cap_mb": 200},
}
OK_EXT = {".json", ".txt", ".safetensors", ".md"}
BAD_DIRS = {"onnx", "openvino"}
NODES = {"squad": 20233, "musique": 117534, "multihoprag": 609, "metaqa": 43234, "webqsp": 2592894, "hotpotqa": 5233329, "2wiki": 5989847,
         "stark_prime": 129375, "stark_mag": 1872968}      # package line counts (outputs/ing1/gw_*.json W1); Amazon has no package yet


def log(msg):
    print(f"[zspl1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zspl1: {msg}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wjson(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + ".tmp")
    t.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(t, p)


def allowed_file(path):
    """The fetch rule: small text and safetensors only; no pickle, no .py, no onnx/openvino folders, no hidden files."""
    parts = path.split("/")
    if any(p.startswith(".") for p in parts) or (set(parts[:-1]) & BAD_DIRS):
        return False
    return os.path.splitext(parts[-1])[1].lower() in OK_EXT


# ── fetch ───────────────────────────────────────────────────────────────────

def cmd_fetch(a):
    from huggingface_hub import HfApi, hf_hub_download
    m = MODELS[a.key]
    dest = OUT / "models" / a.key
    if dest.exists() and any(dest.iterdir()):
        stop(f"{dest} is not empty: each model goes into its own new empty folder")
    dest.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = [(x.path, int(x.size or 0)) for x in api.list_repo_tree(m["repo"], revision=m["rev"], recursive=True, token=False) if hasattr(x, "size")]
    take = [(p, s) for p, s in files if allowed_file(p)]
    skip = [(p, s) for p, s in files if not allowed_file(p)]
    total = sum(s for _, s in take)
    log(f"{m['repo']} @ {m['rev'][:8]}: {len(files)} files, {len(take)} allowed, {total / 1e6:.1f} MB; skipped {[p for p, _ in skip]}")
    for p, s in take:
        log(f"  {p}  {s / 1e6:.2f} MB")
    if total > m["cap_mb"] * 1e6:
        stop(f"the allowed files are {total / 1e6:.0f} MB, over this model's cap of {m['cap_mb']} MB: nothing fetched")
    if not any(p.endswith(".safetensors") for p, _ in take):
        stop("no safetensors weights in the allowed list: nothing fetched")
    rec = {"stage": "SPL1 fetch", "declared_in": DOC, "key": a.key, "repo": m["repo"], "revision": m["rev"], "files": {}, "skipped": [p for p, _ in skip]}
    for p, s in take:
        got = hf_hub_download(repo_id=m["repo"], filename=p, revision=m["rev"], local_dir=str(dest), token=False)
        rec["files"][p] = {"bytes": os.path.getsize(got), "listed_bytes": s, "sha256": sha256(got)}
    rec["total_bytes"] = sum(v["bytes"] for v in rec["files"].values())
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wjson(OUT / f"fetch_{a.key}.json", rec)
    log(f"fetched {len(take)} files, {rec['total_bytes'] / 1e6:.1f} MB into {dest}")


# ── the model ───────────────────────────────────────────────────────────────

def load_model(key, device=None):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SparseEncoder
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = OUT / "models" / key
    if not (path / "modules.json").is_file():
        stop(f"{path} has no modules.json: run fetch first")
    m = SparseEncoder(str(path), device=device, trust_remote_code=False, local_files_only=True)
    for mod in m.modules():                                                         # every module that cuts (a routed model has two sides)
        if hasattr(mod, "max_seq_length"):
            mod.max_seq_length = MAXLEN
    return m, device


def get_tok(m):
    try:
        return m.tokenizer
    except Exception:
        return None


def to_csr(t):
    """A (b, V) sparse or dense torch tensor -> CSR float32 (non-negative)."""
    import scipy.sparse as sp
    if getattr(t, "is_sparse", False):
        t = t.coalesce()
        idx, val = t.indices().cpu().numpy(), t.values().cpu().float().numpy()
        return sp.csr_matrix((val.astype(np.float32), (idx[0], idx[1])), shape=tuple(t.shape))
    return sp.csr_matrix(t.detach().cpu().float().numpy().astype(np.float32))


def enc_docs(model, texts, bs=BATCH):
    return to_csr(model.encode_document(texts, batch_size=bs, convert_to_sparse_tensor=True, show_progress_bar=False))


def enc_queries(model, texts, bs=QBATCH):
    return to_csr(model.encode_query(texts, batch_size=bs, convert_to_sparse_tensor=True, show_progress_bar=False))


def cmd_check(a):
    """What the screen relies on, checked on the fetched model: a row's vector does not depend on its batch, text past 256 tokens
    changes nothing, the question route gives a vector, every value is non-negative."""
    import zworld
    w = zworld.load("squad")
    texts = []
    for _, t in w.iter_nodes():
        texts.append(t)
        if len(texts) == 400:
            break
    m, device = load_model(a.key, a.device)
    tok = get_tok(m)
    lens = [len(x) for x in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]] if tok is not None else None
    longs = [t for t, n in zip(texts, lens) if n > MAXLEN + 20][:4] if lens else texts[:4]
    sample = texts[:16]
    Db = enc_docs(m, sample, 16)
    solo = [enc_docs(m, [t], 1) for t in sample]
    d_batch = max(abs(Db[i] - solo[i]).max() for i in range(len(sample)))
    same_support = float(np.mean([set(Db[i].indices) == set(solo[i].indices) for i in range(len(sample))]))
    d_cut = None
    if longs:
        A = enc_docs(m, longs, 4)
        B = enc_docs(m, [t + " zebra quartz nonsense words appended past the cut" for t in longs], 4)
        d_cut = float(abs(A - B).max())
    q = enc_queries(m, ["who wrote the book about the long journey?"])
    rec = {"stage": "SPL1 check", "key": a.key, "device": device, "vocabulary": int(Db.shape[1]),
           "max_abs_diff_batch_vs_alone": float(d_batch), "same_support_share": same_support,
           "max_abs_diff_text_past_the_cut": d_cut, "long_texts_used": len(longs), "min_value": float(min(Db.data.min(), q.data.min())),
           "doc_nnz_mean": float(np.diff(Db.indptr).mean()), "query_nnz": int(q.getnnz()), "tokenizer": tok is not None}
    rec["call"] = "PASS" if (d_batch <= 1e-3 and same_support >= 0.9 and (d_cut is None or d_cut <= 1e-4) and rec["min_value"] >= 0 and rec["query_nnz"] > 0) else "FAIL"
    wjson(OUT / f"check_{a.key}.json", rec)
    log(json.dumps(rec))
    if rec["call"] != "PASS":
        stop(f"check {a.key} FAILED")


# ── data of a screen dataset ────────────────────────────────────────────────

def eval_rows(ds, w):
    if ds in ("squad", "musique"):
        z = np.load(WS / "outputs" / "enc" / "e0" / "e3v" / f"{ds}.npz", allow_pickle=False)
        return z["rows"].astype(np.int64), z
    return np.arange(w.nq, dtype=np.int64), None


def question_texts(w, rows):
    want = {int(r) for r in rows}
    got = {q["row"]: q["question"] for q in w.iter_queries(resolve=False) if q["row"] in want}
    if len(got) != len(want):
        stop(f"{w.name}: {len(want) - len(got)} wanted questions not found")
    return [got[int(r)] for r in rows]


def golds_of(w, rows):
    want = {int(r): j for j, r in enumerate(rows)}
    g = [None] * len(rows)
    for q in w.iter_queries(resolve=True):
        if q["row"] in want:
            g[want[q["row"]]] = np.unique(q["gold_pos"])
    flat = np.concatenate(g) if g else np.zeros(0, np.int64)
    off = np.concatenate([[0], np.cumsum([x.size for x in g])]).astype(np.int64)
    return flat, off


# ── encode ──────────────────────────────────────────────────────────────────

def merge_top(top_s, top_i, sc, i0, k):
    """zstk1s.merge_top: the running best per question; scores that are not positive are not candidates; ties keep the lower id."""
    nq, b = sc.shape
    sc = np.where(sc > 0, sc, -np.inf).astype(np.float32, copy=False)
    cs = np.concatenate([top_s, sc], axis=1)
    ci = np.concatenate([top_i, np.broadcast_to(np.arange(i0, i0 + b, dtype=np.int64), (nq, b))], axis=1)
    order = np.argsort(-cs, axis=1, kind="stable")[:, :k]
    ns, ni = np.take_along_axis(cs, order, 1), np.take_along_axis(ci, order, 1)
    return ns, np.where(np.isfinite(ns), ni, -1)


def block_rates(lines):
    """A4: nodes per second of each block's encode call (the models' forward passes only). Returns (rates, median)."""
    rates = [x["nodes"] / x["seconds"] for x in lines if x.get("seconds", 0) > 0]
    return rates, (float(np.median(rates)) if rates else None)


def cmd_encode(a):
    import zworld
    t0 = time.time()
    w = zworld.load(a.ds)
    rows, _ = eval_rows(a.ds, w)
    qt = question_texts(w, rows)
    texts = [t for _, t in w.iter_nodes()]
    if a.smoke:
        texts = texts[:a.smoke]
    n, k = len(texts), min(TOPK, len(texts))
    m, device = load_model(a.key, a.device)
    if device == "cpu" and a.threads:
        import torch
        torch.set_num_threads(a.threads)
    d = OUT / a.key / (a.ds + (f"_smoke_{device}" if a.smoke else ""))
    d.mkdir(parents=True, exist_ok=True)
    gpu = None
    if device == "cuda":
        import torch
        fm, tm = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "free_mb_at_start": int(fm / 2**20), "total_mb": int(tm / 2**20)}
    t1 = time.time()
    Qm = enc_queries(m, qt)
    log(f"{a.key} {a.ds}: {len(qt)} questions encoded in {time.time() - t1:.1f} s on {device}, query nnz mean {np.diff(Qm.indptr).mean():.1f}, {n:,} nodes")
    st = d / "state.npz"
    top_s, top_i, done = np.full((len(qt), k), -np.inf, np.float32), np.full((len(qt), k), -1, np.int64), 0
    if st.is_file():
        z = np.load(st, allow_pickle=False)
        if int(z["n"]) == n and int(z["k"]) == k and int(z["nq"]) == len(qt):
            top_s, top_i, done = z["top_s"], z["top_i"], int(z["done"])
            log(f"resume at {done:,}")
    tok = get_tok(m)
    bl = d / "blocks.jsonl"
    if done == 0 and bl.is_file():
        bl.unlink()
    cut = nnz = 0
    t2 = time.time()
    n0 = done
    while done < n:
        end = min(n, done + BLOCK)
        blk = texts[done:end]
        if tok is not None:
            cut += sum(len(x) > MAXLEN for x in tok(blk, add_special_tokens=True, truncation=False)["input_ids"])
        tb = time.time()
        Db = enc_docs(m, blk, a.batch)
        dt = time.time() - tb
        nnz += int(Db.nnz)
        S = (Qm @ Db.T).toarray()
        top_s, top_i = merge_top(top_s, top_i, S, done, k)
        with open(bl, "a", encoding="utf-8") as f:
            f.write(json.dumps({"end": end, "nodes": end - done, "seconds": round(dt, 3), "t": round(time.time(), 1)}) + "\n")
        done = end
        tmp = st.with_name("state.tmp.npz")
        np.savez(tmp, top_s=top_s, top_i=top_i, done=done, n=n, k=k, nq=len(qt))
        os.replace(tmp, st)
        el = time.time() - t2
        log(f"  {done:,}/{n:,} nodes, {(done - n0) / max(el, 1e-9):.0f} nodes/s")
    el = time.time() - t2
    ids = np.where(np.isfinite(top_s), top_i, -1).astype(np.int32)
    np.savez(d / "top1000.npz", ids=ids, scores=np.where(np.isfinite(top_s), top_s, 0).astype(np.float32), rows=rows)
    fr = json.loads((OUT / f"fetch_{a.key}.json").read_text(encoding="utf-8")) if (OUT / f"fetch_{a.key}.json").is_file() else {}
    import torch, sentence_transformers, transformers
    brs, brm = block_rates([json.loads(x) for x in bl.read_text(encoding="utf-8").splitlines() if x.strip()]) if bl.is_file() else ([], None)
    rec = {"stage": "SPL1 encode", "declared_in": DOC, "key": a.key, "repo": MODELS[a.key]["repo"], "revision": MODELS[a.key]["rev"],
           "dataset": a.ds, "smoke": a.smoke, "nodes": n, "questions": len(qt), "k": k, "device": device, "threads": (a.threads or None) if device == "cpu" else None, "max_len": MAXLEN, "batch": a.batch,
           "nodes_per_s": round((n - n0) / max(el, 1e-9), 2), "block_rate_median": None if brm is None else round(brm, 2),
           "block_rates": [round(x, 1) for x in brs], "encode_seconds": round(el, 1), "seconds": round(time.time() - t0, 1),
           "doc_nnz_mean": round(nnz / max(n - n0, 1), 1), "doc_bytes_per_node_at_8": round(8 * nnz / max(n - n0, 1), 1),
           "query_nnz_mean": round(float(np.diff(Qm.indptr).mean()), 1), "gpu": gpu, "cut_at_256": None if tok is None else cut, "cut_counted_on_nodes": n - n0,
           "positive_slots_filled_mean": round(float((ids >= 0).sum(1).mean()), 1),
           "model_files": {p: v["sha256"][:16] for p, v in fr.get("files", {}).items()},
           "top1000_sha256": sha256(d / "top1000.npz"), "versions": {"torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
                                                                    "transformers": transformers.__version__}}
    wjson(d / "encode.json", rec)
    log(f"done {a.key} {a.ds}: {rec['nodes_per_s']} nodes/s, {rec['seconds']} s")


# ── read ────────────────────────────────────────────────────────────────────

def boot(diff, seed=0):
    """Mean of the paired per-question differences and its 95% bootstrap interval, in points."""
    rng = np.random.default_rng(seed)
    n = diff.size
    means = np.empty(NB)
    for i in range(0, NB, 100):
        idx = rng.integers(0, n, (min(100, NB - i), n))
        means[i:i + 100] = diff[idx].mean(1)
    return float(diff.mean() * 100), float(np.percentile(means, 2.5) * 100), float(np.percentile(means, 97.5) * 100)


def call_of(lo):
    return "BETTER" if lo > 0 else ("LEVEL" if lo >= LEVEL_PTS else "WORSE")


def hits_in(sets, flat, off):
    """Distinct golds of each question inside each row's set (a list of 1-d arrays)."""
    h = np.zeros(len(sets), np.int64)
    for q, s in enumerate(sets):
        g = flat[off[q]:off[q + 1]]
        if g.size:
            h[q] = np.isin(g, s).sum()
    return h


def reads_of(E, dense, leg, flat, off, gt, n):
    """Per-question recall vectors: rrf5, rrf20, seeds (dense top 5 with the leg's top 5), the leg's R@20 and R@1000 (not where n < 1000)."""
    gtf = gt.astype(np.float64)

    def rec(h):
        return np.minimum(h, gt) / gtf

    r20 = E.rrf_top(dense, leg, 20)
    out = {"rrf5": rec(E.hits(r20[:, :5], flat, off)), "rrf20": rec(E.hits(r20, flat, off)),
           "seeds": rec(hits_in([np.unique(np.concatenate([dense[q, :5], leg[q, :5]])) for q in range(dense.shape[0])], flat, off)),
           "leg20": rec(E.hits(leg[:, :20], flat, off))}
    if n >= 1000:
        out["leg1000"] = rec(E.hits(leg, flat, off))
    return out, r20


def context(E, ds):
    import zworld
    w = zworld.load(ds)
    rows, z = eval_rows(ds, w)
    flat, off = golds_of(w, rows)
    gt = np.diff(off).astype(np.int64)
    c0 = np.asarray(np.load(w.root / "retrieval_cache" / "splade_top1000.npz", mmap_mode="r")["ids"][rows], np.int64)
    if z is not None:
        dense = z["top_ids"].astype(np.int64)
        if not np.array_equal(z["gold_total"], gt):
            stop(f"{ds}: the gold counts of zworld differ from E0's")
        e0 = json.loads((WS / "outputs" / "enc" / "e0" / "e3v" / f"{ds}.json").read_text(encoding="utf-8"))["recall"]["rrf5"]
    else:
        paths_q, paths_d = w.vector_shards("e3v", "queries")[0], w.vector_shards("e3v", "docs")[0]
        if paths_q is None or paths_d is None:
            stop(f"{ds}: e3v vectors are not on this machine")
        Q = np.concatenate([np.asarray(np.load(p, mmap_mode="r"), np.float32) for p in paths_q])[rows]
        dense = E.search(Q, paths_d, k=min(E.K, w.n))[0]
        e0 = None
    return {"w": w, "rows": rows, "flat": flat, "off": off, "gt": gt, "c0": c0, "dense": dense, "e0_rrf5": e0}


def rank_of(vals, high_is_better=False):
    """1 = best; equal values share the lower rank."""
    srt = sorted(vals.values(), reverse=high_is_better)
    return {k: srt.index(v) + 1 for k, v in vals.items()}


def pick_of(passers, params, nnz, speed):
    """A2: among the passing candidates, the lowest sum of ranks over parameters (fewer), non-zeros per node (fewer) and nodes per
    second on SQuAD (more); a tie goes to the fewer parameters. Returns (pick, table)."""
    if not passers:
        return None, {}
    rp = rank_of({k: params[k] for k in passers})
    rn = rank_of({k: nnz[k] for k in passers})
    rs = rank_of({k: speed[k] for k in passers}, True)
    tab = {k: {"params_m": params[k], "nnz_mean": nnz[k], "nodes_per_s": speed[k], "rank_params": rp[k], "rank_nnz": rn[k], "rank_speed": rs[k],
               "rank_sum": rp[k] + rn[k] + rs[k]} for k in passers}
    return min(passers, key=lambda k: (tab[k]["rank_sum"], params[k])), tab


def c0_vectors(w):
    """C0's own stored SPLADE vectors of the package (only MultiHop-RAG keeps them): mean non-zeros per node, read from the shard."""
    import scipy.sparse as sp
    p = w.root / "embeddings" / "splade" / "docs" / "shard_00000.npz"
    if not p.is_file():
        return None
    M = sp.load_npz(p).tocsr()
    return {"rows": int(M.shape[0]), "vocabulary": int(M.shape[1]), "nnz_mean": round(float(np.diff(M.indptr).mean()), 1)}


def cmd_read(a):
    import zenc as E
    t0 = time.time()
    res = {"stage": "SPL1 read", "declared_in": DOC, "datasets": {}, "seconds": None}
    for ds in a.ds or SCREEN:
        c = context(E, ds)
        n = c["w"].n
        base, r20_c0 = reads_of(E, c["dense"], c["c0"], c["flat"], c["off"], c["gt"], n)
        gate = {}
        if c["e0_rrf5"] is not None:
            gate = {"c0_rrf5": round(float(base["rrf5"].mean()), 6), "e0_filed_rrf5": c["e0_rrf5"],
                    "pass": abs(float(base["rrf5"].mean()) - c["e0_rrf5"]) <= 5e-6}
        else:
            lp = WS / "outputs" / "mhr1" / "first" / "lists.npz"
            L = np.load(lp, allow_pickle=False)
            share = float(np.mean(np.all(L["e3v_splade"][:, :5].astype(np.int64) == r20_c0[:, :5], axis=1)))
            gl, go = L["golds_flat"], L["golds_off"]
            same_g = all(set(c["flat"][c["off"][i]:c["off"][i + 1]].tolist()) == set(gl[go[i]:go[i + 1]].tolist()) for i in range(len(c["rows"])))
            gate = {"c0_rrf5_top5_equals_mhr1_lists_share": round(share, 6), "golds_equal_mhr1": same_g, "pass": share >= 0.999 and same_g}
        log(f"{ds}: G-S1 {json.dumps(gate)}")
        if not gate["pass"]:
            stop(f"G-S1 failed on {ds}: the reader does not reproduce the filed first stage; nothing is read")
        dres = {"questions": int(len(c["rows"])), "nodes": n, "gate_gs1": gate, "c0": {k: round(float(v.mean() * 100), 3) for k, v in base.items()},
                "c0_vectors": c0_vectors(c["w"]), "models": {}}
        for key in ORDER:
            p = OUT / key / ds / "top1000.npz"
            if not p.is_file():
                continue
            z = np.load(p, allow_pickle=False)
            if not np.array_equal(z["rows"], c["rows"]):
                stop(f"{key} {ds}: the rows of the list are not the screen's rows")
            leg = z["ids"].astype(np.int64)
            if leg.shape[1] < c["dense"].shape[1]:
                leg = np.pad(leg, ((0, 0), (0, c["dense"].shape[1] - leg.shape[1])), constant_values=-1)
            r, _ = reads_of(E, c["dense"], leg, c["flat"], c["off"], c["gt"], n)
            mr = {"level": {k: round(float(v.mean() * 100), 3) for k, v in r.items()}, "diff_vs_c0": {}}
            for cell in ("rrf5", "seeds", "rrf20", "leg20", "leg1000"):
                if cell in r and cell in base:
                    dm, lo, hi = boot(r[cell] - base[cell])
                    mr["diff_vs_c0"][cell] = {"diff": round(dm, 3), "lo": round(lo, 3), "hi": round(hi, 3), "call": call_of(lo)}
            enc = json.loads((OUT / key / ds / "encode.json").read_text(encoding="utf-8")) if (OUT / key / ds / "encode.json").is_file() else {}
            mr["nodes_per_s"] = enc.get("block_rate_median") or enc.get("nodes_per_s")
            mr["nodes_per_s_wall"] = enc.get("nodes_per_s")
            mr["doc_nnz_mean"] = enc.get("doc_nnz_mean")
            mr["query_nnz_mean"] = enc.get("query_nnz_mean")
            dres["models"][key] = mr
        res["datasets"][ds] = dres
    verdict = {}
    for key in ORDER:
        cells = {ds: {c: res["datasets"][ds]["models"][key]["diff_vs_c0"][c]["call"] for c in ("rrf5", "seeds")}
                 for ds in res["datasets"] if key in res["datasets"][ds]["models"]}
        if len(cells) == len(SCREEN):
            ok = all(v != "WORSE" for d in cells.values() for v in d.values())
            better_all = all(v == "BETTER" for d in cells.values() for v in d.values())
            verdict[key] = {"cells": cells, "passes": ok, "better_on_all_six": better_all, "params_m": MODELS[key]["params_m"]}
    res["verdict"] = verdict
    passing = [k for k in ORDER if verdict.get(k, {}).get("passes")]
    def mean_nnz(k):
        v = [res["datasets"][ds]["models"][k]["doc_nnz_mean"] for ds in SCREEN if k in res["datasets"].get(ds, {}).get("models", {})]
        return round(float(np.mean(v)), 1) if v and all(x is not None for x in v) else None
    speed = {k: (res["datasets"].get("squad", {}).get("models", {}).get(k) or {}).get("nodes_per_s") for k in ORDER}
    nz = {k: mean_nnz(k) for k in ORDER}
    cost_ok = [k for k in passing if nz[k] is not None and speed[k]]
    res["pick"], res["rank_table"] = pick_of(cost_ok, {k: MODELS[k]["params_m"] for k in ORDER}, nz, speed)
    res["cpu_nodes_per_s_smoke"] = {}
    for k in ORDER:
        sp_ = OUT / k / "squad_smoke_cpu" / "encode.json"
        if sp_.is_file():
            e_ = json.loads(sp_.read_text(encoding="utf-8"))
            res["cpu_nodes_per_s_smoke"][k] = e_.get("block_rate_median") or e_.get("nodes_per_s")
    if res["pick"]:
        res["named_beside"] = [k for k in passing if k != res["pick"] and verdict[k]["better_on_all_six"] and not verdict[res["pick"]]["better_on_all_six"]]
    res["cost_hours_at_squad_rate"] = {}
    for key in ORDER:
        r = (res["datasets"].get("squad", {}).get("models", {}).get(key) or {}).get("nodes_per_s")
        if r:
            res["cost_hours_at_squad_rate"][key] = {ds: round(nn / r / 3600, 2) for ds, nn in NODES.items()}
    res["seconds"] = round(time.time() - t0, 1)
    wjson(OUT / "read.json", res)
    print(json.dumps({k: res[k] for k in ("verdict", "pick") if k in res}, indent=1))


# ── selftest ────────────────────────────────────────────────────────────────

def selftest():
    import scipy.sparse as sp
    ok = []

    def chk(name, c):
        ok.append(bool(c))
        log(f"  {'ok  ' if c else 'FAIL'} {name}")

    chk("fetch rule: safetensors, json, txt, md in; pickle, .py, onnx, hidden out",
        allowed_file("model.safetensors") and allowed_file("1_SpladePooling/config.json") and allowed_file("idf.json") and allowed_file("README.md")
        and not allowed_file("pytorch_model.bin") and not allowed_file("modeling.py") and not allowed_file("onnx/model.safetensors")
        and not allowed_file(".cache/x.json") and not allowed_file("model.pkl"))
    # merge: two blocks, a tie keeps the lower id, non-positive scores are not candidates
    ts, ti = np.full((1, 3), -np.inf, np.float32), np.full((1, 3), -1, np.int64)
    ts, ti = merge_top(ts, ti, np.array([[0.5, 0.0, 0.9]], np.float32), 0, 3)
    ts, ti = merge_top(ts, ti, np.array([[0.9, 0.2]], np.float32), 3, 3)
    chk("merge_top order and tie to the lower id", ti.tolist() == [[2, 3, 0]] and ts.tolist()[0][:3] == [np.float32(0.9), np.float32(0.9), np.float32(0.5)])
    chk("merge_top leaves empty slots at -1", merge_top(np.full((1, 4), -np.inf, np.float32), np.full((1, 4), -1, np.int64),
                                                     np.array([[0.0, 0.3]], np.float32), 0, 4)[1].tolist() == [[1, -1, -1, -1]])
    # to_csr round trip on a dense and a sparse tensor
    import torch
    dn = torch.tensor([[0.0, 1.5, 0.0], [2.0, 0.0, 0.25]])
    chk("to_csr dense", np.array_equal(to_csr(dn).toarray(), dn.numpy()) and np.array_equal(to_csr(dn.to_sparse()).toarray(), dn.numpy()))
    # scoring equals a dense dot product
    rng = np.random.default_rng(1)
    A, B = sp.random(5, 40, 0.2, "csr", np.float32, rng), sp.random(7, 40, 0.2, "csr", np.float32, rng)
    chk("sparse dot product equals dense", np.allclose((A @ B.T).toarray(), A.toarray() @ B.toarray().T, atol=1e-6))
    # bootstrap: a constant difference collapses the interval; a null difference straddles 0; the calls follow the declared lines
    m, lo, hi = boot(np.full(500, 0.02))
    chk("boot constant", abs(m - 2.0) < 1e-9 and abs(lo - 2.0) < 1e-9 and abs(hi - 2.0) < 1e-9)
    d = np.concatenate([np.full(250, 0.5), np.full(250, -0.5)])
    m, lo, hi = boot(d)
    chk("boot null straddles zero", lo < 0 < hi and abs(m) < 1e-9)
    chk("calls", call_of(0.01) == "BETTER" and call_of(0.0) == "LEVEL" and call_of(-1.0) == "LEVEL" and call_of(-1.01) == "WORSE")
    # A4: the speed is the median of the blocks' own encode rates, so a suspended or time-sliced block does not move it
    br, bm = block_rates([{"nodes": 100, "seconds": 1.0}, {"nodes": 100, "seconds": 100.0}, {"nodes": 100, "seconds": 2.0}, {"nodes": 5, "seconds": 0}])
    chk("block_rates median ignores a stalled block and a zero-second row", len(br) == 3 and abs(bm - 50.0) < 1e-9)
    # the A2 pick: lowest rank sum over parameters, non-zeros per node and speed; a tie goes to the fewer parameters
    pk, tb = pick_of(["c3", "c1", "c2"], {"c1": 30, "c2": 67, "c3": 23}, {"c1": 262.0, "c2": 252.0, "c3": 339.0}, {"c1": 900.0, "c2": 500.0, "c3": 1500.0})
    chk("pick_of rank sums (c1: 2+2+2, c2: 3+1+3, c3: 1+3+1)", pk == "c3" and tb["c1"]["rank_sum"] == 6 and tb["c2"]["rank_sum"] == 7 and tb["c3"]["rank_sum"] == 5)
    pk, _ = pick_of(["c1", "c2"], {"c1": 30, "c2": 67}, {"c1": 250.0, "c2": 250.0}, {"c1": 500.0, "c2": 900.0})
    chk("pick_of tie goes to the fewer parameters", pk == "c1" and pick_of([], {}, {}, {}) == (None, {}))
    # hits and seeds on a toy: golds {1, 4}, {2}
    flat, off = np.array([1, 4, 2], np.int64), np.array([0, 2, 3], np.int64)
    h = hits_in([np.array([1, 7, 4]), np.array([0, 9])], flat, off)
    chk("hits_in", h.tolist() == [2, 0])
    # the reader's recall arithmetic with the zenc reader on a tiny case: rrf of a dense list and a leg list
    import zenc as E
    dense = np.array([[5, 1, 2, 3, 4, 6]], np.int64)
    leg = np.array([[1, 9, 5, 7, 8, 2]], np.int64)
    r = E.rrf_top(dense, leg, 3)
    chk("rrf_top (the record's first stage) puts the node both lists rank high first", r[0, 0] in (1, 5) and set(r[0, :2]) == {1, 5})
    if not all(ok):
        stop(f"selftest FAILED ({ok.count(False)} of {len(ok)})")
    log(f"selftest PASS ({len(ok)} checks)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.set_defaults(fn=lambda a: selftest())
    s = sub.add_parser("fetch")
    s.add_argument("--key", required=True, choices=sorted(MODELS))
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("check")
    s.add_argument("--key", required=True, choices=sorted(MODELS))
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("encode")
    s.add_argument("--key", required=True, choices=sorted(MODELS))
    s.add_argument("--ds", required=True, choices=SCREEN)
    s.add_argument("--device", default=None)
    s.add_argument("--smoke", type=int, default=0)
    s.add_argument("--batch", type=int, default=BATCH)
    s.add_argument("--threads", type=int, default=0, help="CPU only: torch threads (0 = the library's default)")
    s.set_defaults(fn=cmd_encode)
    s = sub.add_parser("read")
    s.add_argument("--ds", nargs="*", choices=SCREEN)
    s.set_defaults(fn=cmd_read)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
