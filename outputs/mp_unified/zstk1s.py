"""STK1 amendment A3 (docs/STK1_STARK_PACKAGE.md section 5.3): the SPLADE leg of the first stage for the three STaRK KBs.

Run as:
    python -I outputs/mp_unified/zstk1s.py selftest
    python -I outputs/mp_unified/zstk1s.py splade --kb prime|mag|amazon [--threads 6] [--smoke N]

The record's contract (zmhr.py, the seven's first stage): naver/splade-cocondenser-ensembledistil at its pinned revision,
256 tokens, batches of 64 in file order padded to the longest in the batch, log1p(relu(logits)) max-pooled over every position
(padding included, as the record does). The model is read from the local cache only (local_files_only), never downloaded and never
copied. Nothing is stored per node: the node blocks are scored against the test questions' vectors as they are made and only the
running top 2,500 per question is kept (ties to the lower id, positive scores only), checkpointed after every block.
Output: outputs/stark/<kb>/splade_top.npz with `ids` (all questions of query_ids.json, test rows filled, -1 elsewhere and for
unfilled ranks), the file `zstk1.py first --splade-top` reads, and splade_top.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import zstk1 as Z  # noqa: E402  (hides the card before torch loads)

E = Z.E
SPL, SPL_REV, SPL_LEN, SPL_BATCH = "naver/splade-cocondenser-ensembledistil", "49cf4c7b0db5b870a401ddf5e2669993ef3699c7", 256, 64
BLOCK = 32 * SPL_BATCH                      # documents scored per step (2,048)
SAVE_EVERY = 1                              # blocks between checkpoints (a loss = one block, ~25 min on the laptop)
K = Z.TOPK


def merge_top(top_s, top_i, sc, i0, k):
    """Running best per question. top_s/top_i (nq, k): scores (-inf = empty) and ids (-1 = empty), best first, ties in id order.
    sc (nq, b): the scores of nodes i0..i0+b-1. Scores that are not positive are not candidates."""
    nq, b = sc.shape
    sc = np.where(sc > 0, sc, -np.inf).astype(np.float32, copy=False)
    cs = np.concatenate([top_s, sc], axis=1)
    ci = np.concatenate([top_i, np.broadcast_to(np.arange(i0, i0 + b, dtype=np.int64), (nq, b))], axis=1)
    order = np.argsort(-cs, axis=1, kind="stable")[:, :k]            # stable: the older (lower-id) entries first among equals
    ns, ni = np.take_along_axis(cs, order, 1), np.take_along_axis(ci, order, 1)
    return ns, np.where(np.isfinite(ns), ni, -1)


def csr_rows(v, sp):
    """(b, V) dense non-negative pooled vectors -> CSR float32."""
    r, c = np.nonzero(v)
    ind = np.zeros(v.shape[0] + 1, np.int64)
    np.add.at(ind, r + 1, 1)
    return sp.csr_matrix((v[r, c].astype(np.float32), c.astype(np.int32), np.cumsum(ind)), shape=v.shape)


def load_model(threads):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    torch.set_num_threads(int(threads))
    path = snapshot_download(SPL, revision=SPL_REV, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mdl = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True).eval()
    return torch, tok, mdl


def pooled(torch, tok, mdl, batch):
    enc = tok([str(t) for t in batch], return_tensors="pt", truncation=True, max_length=SPL_LEN, padding=True)
    with torch.no_grad():
        v = torch.log1p(torch.relu(mdl(**enc).logits)).max(dim=1).values.numpy()
    return v, int(enc["attention_mask"].sum()), int((enc["attention_mask"].sum(dim=1) >= SPL_LEN).sum())


def lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass


def cmd_splade(a):
    import scipy.sparse as sp
    ds = Z.KB[a.kb]
    root = Z.PKG / ds
    lower_priority()
    torch, tok, mdl = load_model(a.threads)
    V = int(mdl.config.vocab_size)
    texts = [json.loads(x)["text"] for x in open(root / "nodes.jsonl", encoding="utf-8")]
    qids = json.loads((root / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    pos0 = sum(1 for s in Z.SPLITS[:2] for _ in open(root / "queries" / f"{s}.jsonl", encoding="utf-8"))
    qtexts, test_rows = [], []
    with open(root / "queries" / "test.jsonl", encoding="utf-8") as f:
        for j, ln in enumerate(f):
            r = json.loads(ln)
            if qids[pos0 + j] != r["query_id"]:
                Z.stop(f"query row {pos0 + j}: {r['query_id']} is not query_ids.json's {qids[pos0 + j]}")
            qtexts.append(r["question"])
            test_rows.append(pos0 + j)
    nq = len(qtexts)
    if a.smoke:
        t0, toks, lim = time.time(), 0, 0
        n = int(a.smoke)
        for i in range(0, n, SPL_BATCH):
            _, t, c = pooled(torch, tok, mdl, texts[i:min(i + SPL_BATCH, n)])
            toks += t
            lim += c
        s = time.time() - t0
        Z.log(f"smoke splade {a.kb}: {n} nodes in {s:.1f} s = {n / s:.1f}/s, {toks:,} real tokens ({toks / s:.0f}/s), "
              f"{lim} at the 256 limit, threads {a.threads}")
        return 0
    out = Z.ROOT / a.kb
    out.mkdir(parents=True, exist_ok=True)
    state = out / "splade_state.npz"
    t0 = time.time()
    # question vectors (dense (V, nq): the scoring is document-block csr times this)
    QT = np.zeros((V, nq), np.float32)
    for i in range(0, nq, SPL_BATCH):
        v, _, _ = pooled(torch, tok, mdl, qtexts[i:i + SPL_BATCH])
        QT[:, i:i + v.shape[0]] = v.T
    if not np.isfinite(QT).all() or (QT.sum(axis=0) <= 0).any():
        Z.stop("splade: a damaged or empty question vector")
    Z.log(f"splade questions: {nq} ({time.time() - t0:.0f} s)")
    top_s = np.full((nq, K), -np.inf, np.float32)
    top_i = np.full((nq, K), -1, np.int64)
    done, toks, lim = 0, 0, 0
    if state.exists():
        z = np.load(state)
        if int(z["n_nodes"]) != len(texts) or int(z["nq"]) != nq:
            Z.stop("splade state belongs to another package")
        top_s, top_i, done = z["top_s"], z["top_i"].astype(np.int64), int(z["done"])
        toks, lim = int(z["tokens"]), int(z["at_limit"])
        Z.log(f"splade resumes at node {done:,}")
    t1, nb = time.time(), 0
    while done < len(texts):
        end = min(done + BLOCK, len(texts))
        vs = []
        for i in range(done, end, SPL_BATCH):
            v, t, c = pooled(torch, tok, mdl, texts[i:min(i + SPL_BATCH, end)])
            if not np.isfinite(v).all():
                Z.stop(f"splade: a damaged vector near node {i}")
            vs.append(csr_rows(v, sp))
            toks += t
            lim += c
        D = sp.vstack(vs, format="csr")
        S = np.asarray(D.dot(QT)).T                                 # (nq, b)
        top_s, top_i = merge_top(top_s, top_i, S, done, K)
        done, nb = end, nb + 1
        if nb % SAVE_EVERY == 0 or done == len(texts):
            tmp = state.with_name("splade_state.tmp.npz")
            np.savez(tmp, top_s=top_s, top_i=top_i.astype(np.int32), done=done, n_nodes=len(texts), nq=nq, tokens=toks, at_limit=lim)
            os.replace(tmp, state)
            el = time.time() - t1
            Z.log(f"splade nodes {done:,}/{len(texts):,} ({el:.0f} s this run, {toks:,} tokens, {lim} at the limit)")
    ids = np.full((len(qids), K), -1, np.int32)
    ids[np.asarray(test_rows)] = top_i.astype(np.int32)
    npz = out / "splade_top.npz"
    tmp = out / "splade_top.tmp.npz"
    np.savez(tmp, ids=ids, rows=np.asarray(test_rows))
    os.replace(tmp, npz)
    E.wjson(out / "splade_top.json", {
        "stage": "STK1 splade (A3)", "declared_in": Z.DOC, "dataset": ds, "questions": nq, "nodes": len(texts), "top": K,
        "encoder": {"encoder": SPL, "revision": SPL_REV, "max_length": SPL_LEN, "batch": SPL_BATCH, "device": "cpu", "dtype": "float32",
                    "threads": int(a.threads),
                    "pooling": "log1p(relu(logits)).max(dim=1) over every position, padding included (the record's)"},
        "tokens": toks, "at_limit_256": lim, "filled_ranks_mean": round(float((top_i >= 0).sum(axis=1).mean()), 1),
        "npz_sha256": E.sha256(npz), "code_sha256": {"outputs/mp_unified/zstk1s.py": E.sha256(Path(__file__))},
        "seconds_total": round(time.time() - t0, 1), "utc": Z.utc()})
    Z.log(f"splade {a.kb}: done, {npz.name} written")
    return 0


def selftest():
    import scipy.sparse as sp
    rng = np.random.default_rng(3)
    nq, n, k, b = 5, 300, 20, 64
    S = rng.integers(0, 4, size=(nq, n)).astype(np.float32)          # many ties and many zeros
    ts, ti = np.full((nq, k), -np.inf, np.float32), np.full((nq, k), -1, np.int64)
    for i0 in range(0, n, b):
        ts, ti = merge_top(ts, ti, S[:, i0:i0 + b], i0, k)
    for q in range(nq):
        order = sorted((i for i in range(n) if S[q, i] > 0), key=lambda i: (-S[q, i], i))[:k]
        got = [int(x) for x in ti[q] if x >= 0]
        assert got == order, ("merge", q, got[:6], order[:6])
    ts2, ti2 = merge_top(np.full((1, 5), -np.inf, np.float32), np.full((1, 5), -1, np.int64), np.asarray([[0.0, 2.0, 0.0]], np.float32), 10, 5)
    assert ti2[0].tolist() == [11, -1, -1, -1, -1], ti2
    v = np.asarray([[0, 1.5, 0, 2.0], [0, 0, 0, 0], [3.0, 0, 0, 0]], np.float32)
    m = csr_rows(v, sp)
    assert np.array_equal(m.toarray(), v) and m.indptr.tolist() == [0, 2, 2, 3], m.indptr
    try:
        from zmhr import SPL as s0, SPL_REV as r0, SPL_LEN as l0, SPL_BATCH as b0
        assert (s0, r0, l0, b0) == (SPL, SPL_REV, SPL_LEN, SPL_BATCH), "contract differs from zmhr"
        zm = "zmhr agrees"
    except ImportError:
        zm = "zmhr not importable here (constants not cross-checked)"
    Z.log(f"zstk1s selftest PASS (merge with ties, empty ranks, csr rows; {zm})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("splade")
    s.add_argument("--kb", choices=sorted(Z.KB), required=True)
    s.add_argument("--threads", type=int, default=6)
    s.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(argv)
    return selftest() if a.cmd == "selftest" else cmd_splade(a)


if __name__ == "__main__":
    sys.exit(main())
