"""STK1 amendment A6 (docs/STK1_STARK_PACKAGE.md section 5.6): the SPLADE leg of the first stage for the three STaRK KBs on the host card.

Run as:
    python -B -E -P outputs/mp_unified/zstk1g.py selftest                    # laptop (torch needs the user site's packages: -E -P, not -I)
    python -I -B   outputs/mp_unified/zstk1g.py splade --kb prime|mag|amazon [--threads 2] [--chunk 16] [--smoke N] [--device cuda|cpu] [--lever none|l5] [--gate N]

A8 (docs/STK1_STARK_PACKAGE.md section 5.8) adds `--lever l5` (SPL1 A11's pick: half-precision weights, forward passes of 32 rows, the next batch tokenized in a
second thread; default `none` = A6's path below, unchanged) and `--gate N` (gate G-L: A6's path against the lever on the first N nodes and questions).

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
from concurrent.futures import ThreadPoolExecutor
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


LEVERS = ("none", "l5")        # A8 (section 5.8): "none" is A6's path unchanged; "l5" is SPL1 A11's pick (zspl1l.py's arm l5)


class Pooler:
    """The vectors of one batch. lever "none": pooled() above, A6's path, byte for byte. lever "l5" (A8): zspl1l.py's arm l5 on a half-precision model: the
    batch padded to the longest as before, the forward pass in chunks of 32 rows, the maximum over positions of the raw logits and then log1p(relu(.)) (the
    same numbers as A6's pooling in real arithmetic), and the NEXT batch's tokenization runs in a second thread while this one is on the card (`nxt`).
    A tokenized next batch is used only when the batch asked for next is exactly the list it was made from. Returns (vectors (b, V) float32, real tokens,
    texts at the limit), as pooled() does."""

    def __init__(self, torch, tok, mdl, chunk, lever="none"):
        if lever not in LEVERS:
            Z.stop(f"unknown lever {lever}")
        self.torch, self.tok, self.mdl, self.chunk, self.lever, self.pend = torch, tok, mdl, chunk, lever, None
        if lever != "none":
            import zspl1l as L
            self.L, self.arm, self.ex = L, dict(L.ARMS[lever]), ThreadPoolExecutor(1)

    def __call__(self, batch, nxt=None):
        if self.lever == "none":
            return pooled(self.torch, self.tok, self.mdl, batch, self.chunk)
        batch = list(batch)
        enc = self.pend[1].result() if (self.pend is not None and self.pend[0] == batch) else self.L.tokenize(self.tok, batch)
        self.pend = None
        if nxt:
            nxt = list(nxt)
            self.pend = (nxt, self.ex.submit(self.L.tokenize, self.tok, nxt))
        dev = next(self.mdl.parameters()).device
        v = self.L.pool_batch(self.torch, self.mdl, enc, dev, self.arm)
        am = enc["attention_mask"]
        return v, int(am.sum()), int((am.sum(dim=1) >= SPL_LEN).sum())


def load_model(device, threads, lever="none"):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    torch.set_num_threads(int(threads))
    if not (MODEL_DIR / "model.safetensors").is_file():
        Z.stop(f"{MODEL_DIR} has no model.safetensors: SPL1's fetch and convert rows first")
    if device == "cuda" and not torch.cuda.is_available():
        Z.stop("--device cuda but no card is visible")
    if lever != "none" and device != "cuda":
        Z.stop(f"--lever {lever} is half precision and is the card's: --device cuda")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    mdl = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR), local_files_only=True, use_safetensors=True).eval().to(device)
    if lever != "none":
        mdl = mdl.half()
    return torch, tok, mdl


def gate_lever(a, torch, tok, mdl, texts, qtexts):
    """Gate G-L (STK1 A8): the lever's vectors against A6's on this KB's own texts (the first N nodes and the first N test questions), A11's E2 on each set:
    the first percentile of the per-vector cosine at least 0.999 and the non-zeros within 1.0 percent. `mdl` is the float32 model; the lever runs on a
    half-precision copy. Returns 0 if both sets pass, else 1."""
    import copy
    import scipy.sparse as sp
    import zspl1l as L
    m16 = copy.deepcopy(mdl).half()
    p0, p1 = Pooler(torch, tok, mdl, a.chunk, "none"), Pooler(torch, tok, m16, a.chunk, a.lever)

    def run(p, xs):
        t0, out = time.time(), []
        for i in range(0, len(xs), SPL_BATCH):
            out.append(sp.csr_matrix(p(xs[i:i + SPL_BATCH], xs[i + SPL_BATCH:i + 2 * SPL_BATCH])[0]))
        return sp.vstack(out).tocsr(), time.time() - t0

    rec, ok = {}, True
    for name, xs in (("nodes", texts), ("questions", qtexts)):
        A0, s0 = run(p0, xs)
        A1, s1 = run(p1, xs)
        m = L.compare(A1, A0)
        good = m["cos_p1"] >= L.COS_P1_MIN and abs(m["nnz_ratio"] - 1.0) <= L.NNZ_TOL
        ok = ok and good
        rec[name] = {k: round(float(v), 6) for k, v in m.items()} | {"texts": len(xs), "seconds_A6": round(s0, 2), "seconds_lever": round(s1, 2), "pass": bool(good)}
        Z.log(f"G-L {a.kb} {name}: {len(xs)} texts, cosine p1 {m['cos_p1']:.6f}, non-zeros {m['nnz_ratio']:.5f}, bit-equal {m['bit_equal_share']:.3f}, "
              f"{s0:.1f} s (A6) vs {s1:.1f} s ({a.lever}) -> {'PASS' if good else 'FAIL'}")
    Z.log(f"G-L {a.kb}: {'PASS' if ok else 'FAIL'}")
    out = Z.ROOT / a.kb
    out.mkdir(parents=True, exist_ok=True)
    E.wjson(out / f"gate_lever_{a.lever}.json", {"stage": "STK1 A8 gate G-L", "declared_in": Z.DOC + " section 5.8", "kb": a.kb, "lever": a.lever,
                                                 "reads": rec, "pass": bool(ok), "zspl1l_sha256": E.sha256(Path(L.__file__)), "utc": Z.utc()})
    return 0 if ok else 1


def cmd_splade(a):
    import scipy.sparse as sp
    ds = Z.KB[a.kb]
    root = Z.PKG / ds
    B.lower_priority()
    torch, tok, mdl = load_model(a.device, a.threads, "none" if a.gate else a.lever)    # the gate holds the float32 model and halves a copy itself
    P = Pooler(torch, tok, mdl, a.chunk, "none" if a.gate else a.lever)
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
    if a.gate:
        if a.lever == "none":
            Z.stop("--gate N needs --lever l5 (the lever it checks)")
        n = min(int(a.gate), len(texts), nq)
        return gate_lever(a, torch, tok, mdl, texts[:n], qtexts[:n])
    if a.smoke:
        t0, toks, lim, nnz = time.time(), 0, 0, 0
        n = int(a.smoke)
        for i in range(0, n, SPL_BATCH):
            v, t, c = P(texts[i:min(i + SPL_BATCH, n)], texts[i + SPL_BATCH:min(i + 2 * SPL_BATCH, n)])
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
        v, _, _ = P(qtexts[i:i + SPL_BATCH], qtexts[i + SPL_BATCH:i + 2 * SPL_BATCH])
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
            had = str(z["lever"]) if "lever" in z.files else "none"            # a state made before A8 has no lever: A6's path
            if had != a.lever:
                Z.stop(f"splade state was made under lever {had}, this run is {a.lever}: one lever per leg")
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
            v, t, c = P(texts[i:min(i + SPL_BATCH, end)], texts[i + SPL_BATCH:min(i + 2 * SPL_BATCH, len(texts))])
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
            np.savez(tmp, top_s=top_s, top_i=top_i.astype(np.int32), done=done, n_nodes=len(texts), nq=nq, tokens=toks, at_limit=lim, lever=np.str_(a.lever))
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
        "stage": "STK1 splade (A6)" if a.lever == "none" else "STK1 splade (A6, lever " + a.lever + ", A8)", "declared_in": Z.DOC, "dataset": ds, "questions": nq,
        "nodes": len(texts), "top": K, "lever": a.lever,
        "encoder": {"encoder": SPL, "revision": SPL_REV, "max_length": SPL_LEN, "batch": SPL_BATCH, "device": a.device,
                    "dtype": "float32" if a.lever == "none" else "float16",
                    "chunk_rows": a.chunk if a.lever == "none" else P.arm["chunk"], "weights": "outputs/spl1/models/c0/model.safetensors",
                    "weights_sha256": E.sha256(MODEL_DIR / "model.safetensors"),
                    "pooling": ("log1p(relu(logits)).max(dim=1) over every position, padding included (the record's)" if a.lever == "none" else
                                "log1p(relu(max over every position of the raw logits)), padding included (A11's arm " + a.lever + ": the same numbers in real arithmetic)")},
        "tokens": toks, "at_limit_256": lim, "filled_ranks_mean": round(float((top_i >= 0).sum(axis=1).mean()), 1),
        "block_encode_rate_median": round(float(np.median(rr)), 2) if rr else None,
        "npz_sha256": E.sha256(npz), "code_sha256": {"outputs/mp_unified/zstk1g.py": E.sha256(Path(__file__)),
                                                     "outputs/mp_unified/zstk1s.py": E.sha256(Path(B.__file__))}
        | ({} if a.lever == "none" else {"outputs/mp_unified/zspl1l.py": E.sha256(Path(P.L.__file__))}),
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

    import threading
    seen = []

    class Tok:
        def __call__(self, texts, return_tensors, truncation, max_length, padding):
            seen.append(threading.current_thread().name)
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
    bs = [texts[:3], texts[3:6], texts[6:]]
    r0 = [Pooler(torch, Tok(), Mdl(), 8, "none")(b) for b in bs]
    chk("lever none is pooled(): A6's path", all(np.array_equal(x[0], y[0]) and x[1:] == (y[1], y[2]) for x, y in zip(r0, [pooled(torch, Tok(), Mdl(), b, 8) for b in bs])))
    del seen[:]
    p5 = Pooler(torch, Tok(), Mdl(), 8, "l5")
    r5 = [p5(b, bs[i + 1] if i + 1 < len(bs) else None) for i, b in enumerate(bs)]
    chk("lever l5 gives A6's vectors, tokens and cut counts (the rewrite is an identity in real arithmetic; the toy is float32)",
        all(np.array_equal(x[0], y[0]) and x[1:] == y[1:] for x, y in zip(r0, r5)))
    chk("the next batch is tokenized off the main thread and used", seen[0] == "MainThread" and all(s != "MainThread" for s in seen[1:]) and len(seen) == 3)
    del seen[:]
    p5b = Pooler(torch, Tok(), Mdl(), 8, "l5")
    p5b(bs[0], bs[2])                                               # a tokenized batch that is not the one asked for next
    v = p5b(bs[1])
    chk("a tokenized batch that is not the one asked for is not used", np.array_equal(v[0], r0[1][0]) and len(seen) == 3)
    if bad:
        raise SystemExit(f"zstk1g: selftest FAILED ({bad})")
    Z.log("zstk1g selftest PASS (7 checks beside zstk1s's)")
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
    s.add_argument("--lever", choices=LEVERS, default="none", help="A8: l5 = SPL1 A11's pick (half precision, 32-row chunks, prefetch); none = A6's path")
    s.add_argument("--gate", type=int, default=0, help="A8 gate G-L on the first N nodes and N test questions: A6's path against --lever, no leg is run")
    a = ap.parse_args(argv)
    if a.cmd == "splade" and a.lever != "none" and a.device != "cuda":
        ap.error("--lever l5 is half precision and runs on the card: --device cuda")
    if a.cmd == "splade" and a.device == "cuda":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)                # zstk1.py hid the card at import; this module's rows are the card's (27 Nov: "we can run whatever is required")
    return selftest() if a.cmd == "selftest" else cmd_splade(a)


if __name__ == "__main__":
    sys.exit(main())
