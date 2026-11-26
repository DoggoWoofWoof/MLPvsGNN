"""STK1 amendment A6 (docs/STK1_STARK_PACKAGE.md section 5.6): the SPLADE leg of the first stage for the three STaRK KBs on the host card.

Run as:
    python -B -E -P outputs/mp_unified/zstk1g.py selftest                    # laptop (torch needs the user site's packages: -E -P, not -I)
    python -I -B   outputs/mp_unified/zstk1g.py splade --kb prime|mag|amazon [--threads 2] [--chunk 16] [--smoke N] [--device cuda|cpu]

The same leg as zstk1s.py (A3) and the same contract, the record's (zmhr.py): naver/splade-cocondenser-ensembledistil at its pinned revision, 256 tokens,
batches of 64 in file order padded to the longest in the batch, log1p(relu(logits)) max-pooled over every position (padding included), the running top
2,500 per question kept (ties to the lower id, positive scores only), checkpointed after every block of 2,048 nodes, `splade_top.npz` of the same shape
for `zstk1.py first --splade-top`. What differs, and only that: the weights are read from the host's copy of the pinned model (the folder SPL1's A7
fetched and converted: `outputs/spl1/models/c0`, `model.safetensors`, never the pickle), the forward pass runs on the card in chunks of 16 rows of the
same padded batch (the vectors are the unsplit batch's), and the module does not hide the card the way zstk1.py does. zstk1s.py is not edited; its
merge, sparse-row and constant definitions are imported.
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

import zstk1 as Z  # noqa: E402  (sets CUDA_VISIBLE_DEVICES to "" at import: undone in main for --device cuda, before any card call)
import zstk1s as B  # noqa: E402

E = Z.E
K, BLOCK, SAVE_EVERY = B.K, B.BLOCK, B.SAVE_EVERY
SPL, SPL_REV, SPL_LEN, SPL_BATCH = B.SPL, B.SPL_REV, B.SPL_LEN, B.SPL_BATCH
MODEL_DIR = Z.ROOT.parent / "spl1" / "models" / "c0"          # outputs/spl1/models/c0 (the host's copy, SPL1 A7)


def pooled(torch, tok, mdl, batch, chunk=16):
    """The record's vector for one batch (padded to the longest in the batch, max-pooled over every position), the forward pass in chunks of `chunk`
    rows of the same padded tensor. Returns (vectors (b, V) float32, real tokens, texts at the limit)."""
    enc = tok([str(t) for t in batch], return_tensors="pt", truncation=True, max_length=SPL_LEN, padding=True)
    dev = next(mdl.parameters()).device
    parts = []
    with torch.no_grad():
        for i in range(0, len(batch), chunk):
            x = {k: v[i:i + chunk].to(dev) for k, v in enc.items()}
            parts.append(torch.log1p(torch.relu(mdl(**x).logits)).max(dim=1).values.float().cpu())
    am = enc["attention_mask"]
    return torch.cat(parts).numpy(), int(am.sum()), int((am.sum(dim=1) >= SPL_LEN).sum())


def load_model(device, threads):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    torch.set_num_threads(int(threads))
    if not (MODEL_DIR / "model.safetensors").is_file():
        Z.stop(f"{MODEL_DIR} has no model.safetensors: SPL1's fetch and convert rows first")
    if device == "cuda" and not torch.cuda.is_available():
        Z.stop("--device cuda but no card is visible")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    mdl = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR), local_files_only=True, use_safetensors=True).eval().to(device)
    return torch, tok, mdl


def cmd_splade(a):
    import scipy.sparse as sp
    ds = Z.KB[a.kb]
    root = Z.PKG / ds
    B.lower_priority()
    torch, tok, mdl = load_model(a.device, a.threads)
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
        t0, toks, lim, nnz = time.time(), 0, 0, 0
        n = int(a.smoke)
        for i in range(0, n, SPL_BATCH):
            v, t, c = pooled(torch, tok, mdl, texts[i:min(i + SPL_BATCH, n)], a.chunk)
            toks += t
            lim += c
            nnz += int(np.count_nonzero(v))
        s = time.time() - t0
        Z.log(f"smoke splade {a.kb} ({a.device}): {n} nodes in {s:.1f} s = {n / s:.1f}/s, {toks:,} real tokens ({toks / s:.0f}/s), {lim} at the 256 limit, "
              f"{nnz / n:.1f} non-zeros per node")
        return 0
    out = Z.ROOT / a.kb
    out.mkdir(parents=True, exist_ok=True)
    state = out / "splade_state.npz"
    t0 = time.time()
    QT = np.zeros((V, nq), np.float32)
    for i in range(0, nq, SPL_BATCH):
        v, _, _ = pooled(torch, tok, mdl, qtexts[i:i + SPL_BATCH], a.chunk)
        QT[:, i:i + v.shape[0]] = v.T
    if not np.isfinite(QT).all() or (QT.sum(axis=0) <= 0).any():
        Z.stop("splade: a damaged or empty question vector")
    Z.log(f"splade questions: {nq} ({time.time() - t0:.0f} s)")
    top_s = np.full((nq, K), -np.inf, np.float32)
    top_i = np.full((nq, K), -1, np.int64)
    done, toks, lim = 0, 0, 0
    if state.exists():
        with np.load(state) as z:
            if int(z["n_nodes"]) != len(texts) or int(z["nq"]) != nq:
                Z.stop("splade state belongs to another package")
            top_s, top_i, done = np.array(z["top_s"]), z["top_i"].astype(np.int64), int(z["done"])
            toks, lim = int(z["tokens"]), int(z["at_limit"])
        Z.log(f"splade resumes at node {done:,}")
    t1, nb = time.time(), 0
    bl = out / "splade_blocks.jsonl"
    if done == 0 and bl.exists():
        bl.unlink()
    while done < len(texts):
        end = min(done + BLOCK, len(texts))
        tb = time.time()
        vs = []
        for i in range(done, end, SPL_BATCH):
            v, t, c = pooled(torch, tok, mdl, texts[i:min(i + SPL_BATCH, end)], a.chunk)
            if not np.isfinite(v).all():
                Z.stop(f"splade: a damaged vector near node {i}")
            vs.append(B.csr_rows(v, sp))
            toks += t
            lim += c
        D = sp.vstack(vs, format="csr")
        sec_enc = time.time() - tb
        S = np.asarray(D.dot(QT)).T                                 # (nq, b)
        top_s, top_i = B.merge_top(top_s, top_i, S, done, K)
        with open(bl, "a", encoding="utf-8") as f:
            f.write(json.dumps({"end": end, "nodes": end - done, "encode_seconds": round(sec_enc, 3), "seconds": round(time.time() - tb, 3),
                                "nnz": int(D.nnz), "t": round(time.time(), 1)}) + "\n")
        done, nb = end, nb + 1
        if nb % SAVE_EVERY == 0 or done == len(texts):
            tmp = state.with_name("splade_state.tmp.npz")
            np.savez(tmp, top_s=top_s, top_i=top_i.astype(np.int32), done=done, n_nodes=len(texts), nq=nq, tokens=toks, at_limit=lim)
            for k in range(40):                                     # Windows: a replace can meet a short lock (SPL1 A5)
                try:
                    os.replace(tmp, state)
                    break
                except PermissionError:
                    if k == 39:
                        raise
                    time.sleep(3)
            Z.log(f"splade nodes {done:,}/{len(texts):,} ({time.time() - t1:.0f} s this run, {toks:,} tokens, {lim} at the limit)")
    ids = np.full((len(qids), K), -1, np.int32)
    ids[np.asarray(test_rows)] = top_i.astype(np.int32)
    npz = out / "splade_top.npz"
    tmp = out / "splade_top.tmp.npz"
    np.savez(tmp, ids=ids, rows=np.asarray(test_rows))
    os.replace(tmp, npz)
    rates = {}
    for ln in bl.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            x = json.loads(ln)
            rates[x["end"]] = x                                    # a block redone after a resume counts once (the last record)
    rr = [x["nodes"] / x["encode_seconds"] for x in rates.values() if x["encode_seconds"] > 0]
    E.wjson(out / "splade_top.json", {
        "stage": "STK1 splade (A6)", "declared_in": Z.DOC, "dataset": ds, "questions": nq, "nodes": len(texts), "top": K,
        "encoder": {"encoder": SPL, "revision": SPL_REV, "max_length": SPL_LEN, "batch": SPL_BATCH, "device": a.device, "dtype": "float32",
                    "chunk_rows": a.chunk, "weights": "outputs/spl1/models/c0/model.safetensors",
                    "weights_sha256": E.sha256(MODEL_DIR / "model.safetensors"),
                    "pooling": "log1p(relu(logits)).max(dim=1) over every position, padding included (the record's)"},
        "tokens": toks, "at_limit_256": lim, "filled_ranks_mean": round(float((top_i >= 0).sum(axis=1).mean()), 1),
        "block_encode_rate_median": round(float(np.median(rr)), 2) if rr else None,
        "npz_sha256": E.sha256(npz), "code_sha256": {"outputs/mp_unified/zstk1g.py": E.sha256(Path(__file__)),
                                                     "outputs/mp_unified/zstk1s.py": E.sha256(Path(B.__file__))},
        "seconds_total": round(time.time() - t0, 1), "utc": Z.utc()})
    Z.log(f"splade {a.kb}: done, {npz.name} written")
    return 0


def selftest():
    import torch
    B.selftest()
    bad = 0

    def chk(name, ok):
        nonlocal bad
        Z.log(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += 0 if ok else 1

    class Tok:
        def __call__(self, texts, return_tensors, truncation, max_length, padding):
            ids = [[(ord(c) % 7) + 1 for c in t][:max_length] for t in texts]
            L = max(len(x) for x in ids)
            return {"input_ids": torch.tensor([x + [0] * (L - len(x)) for x in ids]),
                    "attention_mask": torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])}

    class Mdl:
        def __init__(self):
            self.W = torch.randn(8, 12, generator=torch.Generator().manual_seed(0))
            self.p = [torch.zeros(1)]

        def parameters(self):
            return iter(self.p)

        def __call__(self, input_ids, attention_mask):
            return type("O", (), {"logits": self.W[input_ids] - 0.5})()

    texts = ["abc", "abcdefghij", "xyz", "q", "hello world", "zz", "mm mm", "the end"]
    v1, t1, c1 = pooled(torch, Tok(), Mdl(), texts, chunk=8)
    v2, t2, c2 = pooled(torch, Tok(), Mdl(), texts, chunk=3)
    chk("chunks of any size give the unsplit batch's vectors, tokens and cut count", np.array_equal(v1, v2) and t1 == t2 and c1 == c2)
    chk("this module's constants are zstk1s's (and so zmhr's)", (SPL, SPL_REV, SPL_LEN, SPL_BATCH, K) == (B.SPL, B.SPL_REV, B.SPL_LEN, B.SPL_BATCH, B.K))
    chk("the weights folder is the host copy SPL1 A7 fetched", MODEL_DIR.parts[-3:] == ("spl1", "models", "c0"))
    if bad:
        raise SystemExit(f"zstk1g: selftest FAILED ({bad})")
    Z.log("zstk1g selftest PASS (3 checks beside zstk1s's)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("splade")
    s.add_argument("--kb", choices=sorted(Z.KB), required=True)
    s.add_argument("--threads", type=int, default=2)
    s.add_argument("--chunk", type=int, default=16)
    s.add_argument("--smoke", type=int, default=0)
    s.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = ap.parse_args(argv)
    if a.cmd == "splade" and a.device == "cuda":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)                # zstk1.py hid the card at import; this module's rows are the card's (27 Nov: "we can run whatever is required")
    return selftest() if a.cmd == "selftest" else cmd_splade(a)


if __name__ == "__main__":
    sys.exit(main())
