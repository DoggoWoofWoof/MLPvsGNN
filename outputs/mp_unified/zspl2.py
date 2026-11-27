"""SPL2: the sparse leg (one model, C3 or the model SPL1 A8's rule D1 names) over the nodes of every dataset, one rule, resumable, for the first stage's lists.

For a dataset: the model encodes the document side of every node in blocks (the card does this), the question side is the model's own inference-free query
encoding (a bag of weighted tokens), a block's scores are the sparse product of the two, and each question keeps its best K = 2,500 nodes over all blocks (the
running top-K is checkpointed, a restart resumes at the last block). The search of a block runs in a second thread while the next block is encoded.

Questions (one rule for the ten): the seven datasets' dev and eval carves (outputs/pl2/<dataset>/pop_dev.npz and pop_eval.npz; MultiHop-RAG: all its questions),
the three STaRK KBs' test questions. Train-question lists are not built here.

  selftest                          the running top-K against a one-shot search, a restart in the middle, the rows rule (toy model, no files)
  encode --ds D [--smoke N] [--device cuda|cpu] [--threads T] [--batch 16] [--dtype fp32|fp16]
                                    -> outputs/spl2/D/{state.npz, blocks.jsonl, top.npz, top.json}   (--smoke N: first N nodes, written under D_smoke_<device>)

Usage on the host: python -I -B outputs/mp_unified/zspl2.py encode --ds multihoprag
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import zspl1 as S1  # noqa: E402  (imported only: the model loading, encoders, merge, block rates)

DOC = "docs/SPL2_SPARSE_LEG_REBUILD.md"
OUT = WS / "outputs" / "spl2"
PL2 = WS / "outputs" / "pl2"
KEY = "c3"
K, BLOCK, SAVE_SECONDS = 2500, 2048, 150.0
WORLD = ("2wiki", "hotpotqa", "metaqa", "musique", "squad", "webqsp", "multihoprag")
STARK = {"stark_prime": "prime", "stark_mag": "mag", "stark_amazon": "amazon"}
log, stop = S1.log, S1.stop


# ── what is encoded ────────────────────────────────────────────────────────────────────────────────────────────────

def carve_rows(ds, nq_all=None):
    """The question rows of the dataset: dev and eval carves (union, ascending); MultiHop-RAG: every question."""
    if ds == "multihoprag":
        return np.arange(int(nq_all), dtype=np.int64)
    parts = []
    for c in ("dev", "eval"):
        p = PL2 / ds / f"pop_{c}.npz"
        if not p.is_file():
            stop(f"{ds}: {p.name} is not here")
        with np.load(p) as z:
            parts.append(z["idx"].astype(np.int64))
    return np.unique(np.concatenate(parts))


class Stream:
    """The node texts in line order, handed out in blocks (a 6 M-node dataset is never held in memory). A block that does not follow the last one
    (a resume) reopens the source and skips to it."""

    def __init__(self, make, n):
        self.make, self.n, self.it, self.pos = make, int(n), None, 0

    def block(self, i0, i1):
        if self.it is None or i0 != self.pos:
            self.it = self.make()
            for _ in itertools.islice(self.it, i0):
                pass
        out = list(itertools.islice(self.it, i1 - i0))
        if len(out) != i1 - i0:
            stop(f"the node source ended at {i0 + len(out):,}, {i1:,} wanted")
        self.pos = i1
        return out


def _stark_texts(path):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            yield json.loads(ln)["text"]


def load_inputs(ds):
    """(node stream, question texts, rows, meta): rows are the question rows the lists are filed under."""
    if ds in WORLD:
        import zworld
        w = zworld.load(ds)
        rows = carve_rows(ds, w.nq)
        return (Stream(lambda: (t for _, t in w.iter_nodes()), w.n), S1.question_texts(w, rows), rows,
                {"source": "zworld", "nodes": w.n})
    if ds in STARK:
        import zstk1 as Z
        root = Z.PKG / Z.KB[STARK[ds]]
        with open(root / "nodes.jsonl", encoding="utf-8") as f:
            n_nodes = sum(1 for _ in f)
        stream = Stream(lambda: _stark_texts(root / "nodes.jsonl"), n_nodes)
        qids = json.loads((root / "queries" / "query_ids.json").read_text(encoding="utf-8"))
        pos0 = sum(1 for s in Z.SPLITS[:2] for _ in open(root / "queries" / f"{s}.jsonl", encoding="utf-8"))
        qtexts, rows = [], []
        with open(root / "queries" / "test.jsonl", encoding="utf-8") as f:
            for j, ln in enumerate(f):
                r = json.loads(ln)
                if qids[pos0 + j] != r["query_id"]:
                    stop(f"query row {pos0 + j}: {r['query_id']} is not query_ids.json's {qids[pos0 + j]}")
                qtexts.append(r["question"])
                rows.append(pos0 + j)
        return stream, qtexts, np.asarray(rows, dtype=np.int64), {"source": "stark package", "nodes": n_nodes}
    stop(f"unknown dataset {ds}")


# ── the search: blocks in, a running top-K per question out ────────────────────────────────────────────────────────

def run_search(encode_block, Qm, n, k, block, state, blocks_log, save_seconds=SAVE_SECONDS, tag=None, log_every=10):
    """encode_block(i0, i1) -> (b, V) CSR of the nodes i0..i1. Returns (top_s, top_i, stats). The encode of block t+1 overlaps the search of block t."""
    nq = Qm.shape[0]
    top_s, top_i, done = np.full((nq, k), -np.inf, np.float32), np.full((nq, k), -1, np.int64), 0
    if state.is_file():
        with np.load(state, allow_pickle=False) as z:                  # closed before the next checkpoint replaces the file (Windows)
            if int(z["n"]) == n and int(z["k"]) == k and int(z["nq"]) == nq and str(z["tag"]) == str(tag):
                top_s, top_i, done = np.array(z["top_s"]), np.array(z["top_i"], dtype=np.int64), int(z["done"])
                log(f"resume at {done:,}")
    if done == 0 and blocks_log.is_file():
        blocks_log.unlink()
    lock = threading.Lock()
    last_save = [time.time()]
    cur = {"s": top_s, "i": top_i}

    def search(Db, i0, i1, enc_seconds, t_start):
        Sb = (Qm @ Db.T).toarray()
        s, i = S1.merge_top(cur["s"], cur["i"], Sb, i0, k)
        with lock:
            cur["s"], cur["i"] = s, i
        with open(blocks_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"end": i1, "nodes": i1 - i0, "seconds": round(enc_seconds, 3), "nnz": int(Db.nnz), "t": round(time.time(), 1)}) + "\n")
        if time.time() - last_save[0] > save_seconds or i1 == n:
            tmp = state.with_name("state.tmp.npz")
            np.savez(tmp, top_s=s, top_i=i.astype(np.int32), done=i1, n=n, k=k, nq=nq, tag=str(tag))
            S1.replace_retry(tmp, state)
            last_save[0] = time.time()
            log(f"  {i1:,}/{n:,} nodes saved ({(time.time() - t_start):.0f} s this run)")

    t0, nb, fut, nnz = time.time(), 0, None, 0
    with ThreadPoolExecutor(max_workers=1) as ex:
        pos = done
        while pos < n:
            end = min(n, pos + block)
            tb = time.time()
            Db = encode_block(pos, end)
            enc = time.time() - tb
            nnz += int(Db.nnz)
            if fut is not None:
                fut.result()                                            # the previous block's search must have finished (and may raise)
            fut = ex.submit(search, Db, pos, end, enc, t0)
            pos = end
            nb += 1
            if nb % log_every == 0:
                log(f"  encoded to {pos:,}/{n:,}, {(pos - done) / max(time.time() - t0, 1e-9):.0f} nodes/s")
        if fut is not None:
            fut.result()
    return cur["s"], cur["i"], {"start": done, "nnz": nnz, "seconds": time.time() - t0}


# ── encode ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def cmd_encode(a):
    t0 = time.time()
    stream, qtexts, rows, meta = load_inputs(a.ds)
    n = min(stream.n, a.smoke) if a.smoke else stream.n
    k = min(K, n)
    m, device = S1.load_model(KEY, a.device)
    if device == "cpu" and a.threads:
        import torch
        torch.set_num_threads(a.threads)
    if a.dtype == "fp16":
        if device != "cuda":
            stop("--dtype fp16 needs the card")
        m = m.half()
    d = OUT / (a.ds + (f"_smoke_{device}" if a.smoke else ""))
    d.mkdir(parents=True, exist_ok=True)
    Qm = S1.enc_queries(m, qtexts)
    log(f"{a.ds}: {len(qtexts)} questions, query non-zeros {np.diff(Qm.indptr).mean():.1f}, {n:,} nodes, {device}, {a.dtype}")
    tok = S1.get_tok(m)
    cut = [0]

    def encode_block(i0, i1):
        blk = stream.block(i0, i1)
        if tok is not None:
            cut[0] += sum(len(x) > S1.MAXLEN for x in tok(blk, add_special_tokens=True, truncation=False)["input_ids"])
        return S1.enc_docs(m, blk, a.batch)

    tag = f"{KEY}|{a.dtype}|{a.batch}|{len(qtexts)}|{int(rows.sum())}"
    top_s, top_i, st = run_search(encode_block, Qm, n, k, BLOCK, d / "state.npz", d / "blocks.jsonl", tag=tag)
    ids = np.where(np.isfinite(top_s), top_i, -1).astype(np.int32)
    tmp = d / "top.tmp.npz"
    np.savez(tmp, ids=ids, scores=np.where(np.isfinite(top_s), top_s, 0).astype(np.float32), rows=rows)
    S1.replace_retry(tmp, d / "top.npz")
    brs, brm = S1.block_rates([json.loads(x) for x in (d / "blocks.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()])
    fr = json.loads((S1.OUT / f"fetch_{KEY}.json").read_text(encoding="utf-8")) if (S1.OUT / f"fetch_{KEY}.json").is_file() else {}
    import sentence_transformers
    import torch
    import transformers
    S1.wjson(d / "top.json", {
        "stage": "SPL2 encode", "declared_in": DOC, "key": KEY, "repo": S1.MODELS[KEY]["repo"], "revision": S1.MODELS[KEY]["rev"], "dataset": a.ds,
        "source": meta["source"], "nodes": n, "questions": len(qtexts), "k": k, "device": device, "dtype": a.dtype, "batch": a.batch, "max_len": S1.MAXLEN,
        "block": BLOCK, "block_rate_median": None if brm is None else round(brm, 2), "encode_seconds_this_run": round(st["seconds"], 1),
        "resumed_from": st["start"], "doc_nnz_mean_this_run": round(st["nnz"] / max(n - st["start"], 1), 1),
        "query_nnz_mean": round(float(np.diff(Qm.indptr).mean()), 1), "cut_at_256_this_run": cut[0],
        "positive_slots_filled_mean": round(float((ids >= 0).sum(1).mean()), 1), "rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "model_files": {p: v["sha256"][:16] for p, v in fr.get("files", {}).items()}, "top_sha256": S1.sha256(d / "top.npz"),
        "code_sha256": S1.sha256(Path(__file__)), "versions": {"torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
                                                               "transformers": transformers.__version__},
        "seconds_total": round(time.time() - t0, 1)})
    log(f"done {a.ds}: {n:,} nodes, median block rate {brm and round(brm, 1)} nodes/s, {time.time() - t0:.0f} s")
    return 0


# ── self-test (toy model: no files, no card) ────────────────────────────────────────────────────────────────────────

def selftest():
    import tempfile

    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    V, n, nq, k = 40, 23, 6, 7
    D = sp.csr_matrix((rng.random((n, V)) * (rng.random((n, V)) < 0.2)).astype(np.float32))
    Q = sp.csr_matrix((rng.random((nq, V)) * (rng.random((nq, V)) < 0.3)).astype(np.float32))
    full = np.asarray((Q @ D.T).todense())
    ok = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        enc = lambda i0, i1: D[i0:i1]  # noqa: E731
        s1, i1, st = run_search(enc, Q, n, k, 4, tmp / "a.npz", tmp / "a.jsonl", save_seconds=0.0, tag="x")
        ref_i = np.argsort(-np.where(full > 0, full, -np.inf), axis=1, kind="stable")[:, :k]
        ref_s = np.take_along_axis(np.where(full > 0, full, -np.inf), ref_i, 1)
        ref_i = np.where(np.isfinite(ref_s), ref_i, -1)
        ok.append(("the running top-K over blocks equals a one-shot search (scores and ids, ties to the lower id)", np.allclose(s1, ref_s, equal_nan=False) and np.array_equal(i1, ref_i)))
        calls = []

        def enc_cut(i0, i1):
            calls.append(i0)
            if i0 >= 12 and len(calls) < 99 and not (tmp / "go").exists():
                raise RuntimeError("stop")
            return D[i0:i1]
        try:
            run_search(enc_cut, Q, n, k, 4, tmp / "b.npz", tmp / "b.jsonl", save_seconds=0.0, tag="x")
            died = False
        except RuntimeError:
            died = True
        (tmp / "go").write_text("1")
        calls.clear()
        s2, i2, st2 = run_search(enc_cut, Q, n, k, 4, tmp / "b.npz", tmp / "b.jsonl", save_seconds=0.0, tag="x")
        ok.append(("a restart after a stop in the middle resumes at the last saved block and gives the same result", died and st2["start"] > 0 and calls[0] == st2["start"]
                   and np.array_equal(i2, i1) and np.allclose(s2, s1)))
        s3, i3, st3 = run_search(enc, Q, n, k, 4, tmp / "b.npz", tmp / "b.jsonl", save_seconds=0.0, tag="y")
        ok.append(("a state of another run (tag) is not resumed", st3["start"] == 0 and np.array_equal(i3, i1)))
    mk = lambda: iter([f"t{j}" for j in range(11)])  # noqa: E731
    st = Stream(mk, 11)
    got = st.block(0, 4) + st.block(4, 8) + st.block(8, 11)
    again = st.block(5, 9)
    ok.append(("the node stream gives the texts in order, a block out of order reopens and skips", got == [f"t{j}" for j in range(11)]
               and again == ["t5", "t6", "t7", "t8"]))
    ok.append(("the rows rule: MultiHop-RAG takes every question", np.array_equal(carve_rows("multihoprag", 5), np.arange(5))))
    for nme, v in ok:
        print(("PASS " if v else "FAIL ") + nme)
    print(f"{sum(v for _, v in ok)}/{len(ok)} checks pass")
    return 0 if all(v for _, v in ok) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    e = sub.add_parser("encode")
    e.add_argument("--ds", required=True, choices=sorted(set(WORLD) | set(STARK)))
    e.add_argument("--device", default=None)
    e.add_argument("--smoke", type=int, default=0)
    e.add_argument("--batch", type=int, default=S1.BATCH)
    e.add_argument("--threads", type=int, default=0)
    e.add_argument("--dtype", default="fp32", choices=("fp32", "fp16"))
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    return cmd_encode(a)


if __name__ == "__main__":
    sys.exit(main())
