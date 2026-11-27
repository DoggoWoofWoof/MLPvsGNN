"""LEG1 part B (docs/LEG1_LEG_NECESSITY.md section 3): the stored C0 vectors the modifications act on.

    python -E -P -B outputs/mp_unified/zleg1b.py selftest
    python -I -B    outputs/mp_unified/zleg1b.py vectors --ds multihoprag|squad [--device cpu] [--threads N]
        # the record's pooling (arm l0 of zspl1l.py: fp32, chunks of 16, batches of 64 in file order from each block of 2,048) of every node and every eval question,
        # as CSR shards under outputs/leg1/vec/<ds>/ (a shard per block, resumable), then the gate G-B0 against the record's C0 top 5
Run from the workspace root. zspl1.py, zspl1c.py and zspl1l.py are imported and edited nowhere. The vectors are made from the texts of the dataset's world; no label is read.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import zspl1 as S  # noqa: E402
import zspl1l as L  # noqa: E402

DOC = "docs/LEG1_LEG_NECESSITY.md section 3"
OUT = S.WS / "outputs" / "leg1" / "vec"
QBLOCK = 1024
GATE_SHARE = 0.99
log, stop = S.log, S.stop


def shard_path(d, kind, i):
    return d / f"{kind}_{i:06d}.npz"


def encode_shards(sp, torch, tok, models, device, texts, d, kind, block):
    """Encode `texts` in blocks of `block`; a finished shard is kept (resume). Returns the seconds of the work done in this run."""
    t0 = time.time()
    for i in range(0, len(texts), block):
        p = shard_path(d, kind, i // block)
        if p.is_file():
            continue
        csr, secs, _ = L.run_encode(torch, tok, models, device, texts[i:i + block], "l0")
        tmp = p.with_name(p.stem + ".tmp.npz")
        sp.save_npz(tmp, csr.astype(np.float32).tocsr(), compressed=False)
        S.replace_retry(tmp, p)
        done = min(i + block, len(texts))
        log(f"  {kind} {done:,}/{len(texts):,}  ({(time.time() - t0):.0f} s this run)")
    return time.time() - t0


def load_shards(sp, d, kind, n_expected):
    fs = sorted(d.glob(f"{kind}_*.npz"))
    fs = [f for f in fs if ".tmp" not in f.name]
    m = sp.vstack([sp.load_npz(f) for f in fs]).tocsr()
    if m.shape[0] != n_expected:
        stop(f"{kind}: {m.shape[0]} rows stored, {n_expected} expected")
    return m


def top5(sp, Q, Dm, block=512):
    """The top 5 of Q @ Dm.T per question (score descending, node id ascending on ties), by blocks."""
    out = np.full((Q.shape[0], 5), -1, np.int64)
    DT = Dm.T.tocsr()
    for i in range(0, Q.shape[0], block):
        sc = (Q[i:i + block] @ DT).toarray()
        idx = np.argpartition(-sc, 5, axis=1)[:, :5] if sc.shape[1] > 5 else np.tile(np.arange(sc.shape[1]), (sc.shape[0], 1))[:, :5]
        for r in range(idx.shape[0]):
            c = idx[r]
            o = np.lexsort((c, -sc[r, c]))
            out[i + r] = c[o]
    return out


def cmd_vectors(a):
    import scipy.sparse as sp
    import zworld
    t0 = time.time()
    if a.ds not in ("multihoprag", "squad"):
        stop("part B reads MultiHop-RAG and SQuAD first (section 3); others by an amendment")
    device = a.device or "cpu"
    torch, tok, models, tfv = L.load_models(device, ["l0"])
    if a.threads:
        torch.set_num_threads(a.threads)
    w = zworld.load(a.ds)
    rows, _ = S.eval_rows(a.ds, w)
    qt = S.question_texts(w, rows)
    texts = [t for _, t in w.iter_nodes()]
    d = OUT / a.ds
    d.mkdir(parents=True, exist_ok=True)
    log(f"{a.ds}: {len(texts):,} nodes, {len(qt):,} questions, device {device}, {torch.get_num_threads()} threads")
    tq = encode_shards(sp, torch, tok, models, device, qt, d, "q", QBLOCK)
    td = encode_shards(sp, torch, tok, models, device, texts, d, "d", S.BLOCK)
    Qm, Dm = load_shards(sp, d, "q", len(qt)), load_shards(sp, d, "d", len(texts))
    rec = np.asarray(np.load(w.root / "retrieval_cache" / "splade_top1000.npz", mmap_mode="r")["ids"][rows][:, :5], np.int64)
    got = top5(sp, Qm, Dm)
    share = float(np.mean(np.all(got == rec, axis=1)))
    set_share = float(np.mean(np.all(np.sort(got, axis=1) == np.sort(rec, axis=1), axis=1)))
    meta = {"stage": "LEG1 part B vectors", "declared_in": DOC, "ds": a.ds, "nodes": len(texts), "questions": len(qt), "device": device, "threads": torch.get_num_threads(),
            "doc_nnz_total": int(Dm.nnz), "doc_nnz_mean": round(Dm.nnz / len(texts), 2), "query_nnz_mean": round(Qm.nnz / len(qt), 2),
            "gate_gb0": {"top5_same_sequence_share": round(share, 5), "top5_same_set_share": round(set_share, 5), "min_share": GATE_SHARE, "pass": share >= GATE_SHARE},
            "encode_seconds_this_run": {"questions": round(tq, 1), "nodes": round(td, 1)}, "seconds": round(time.time() - t0, 1), "versions": {"torch": torch.__version__, "transformers": tfv},
            "sha256": {"doc_shard0": S.sha256(shard_path(d, "d", 0)), "q_shard0": S.sha256(shard_path(d, "q", 0))}}
    S.wjson(d / "meta.json", meta)
    log(f"G-B0 {json.dumps(meta['gate_gb0'])}")
    if not meta["gate_gb0"]["pass"]:
        stop(f"G-B0 failed on {a.ds}: the stored vectors do not reproduce the record's top 5 for {GATE_SHARE:.0%} of the questions; the part is stopped here")
    log(f"done {a.ds}: doc nnz mean {meta['doc_nnz_mean']}, query nnz mean {meta['query_nnz_mean']}, {meta['seconds']} s")


def selftest():
    import scipy.sparse as sp
    bad = 0

    def chk(name, ok):
        nonlocal bad
        log(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += 0 if ok else 1

    rng = np.random.default_rng(0)
    Dm = sp.random(40, 60, density=0.2, random_state=1, format="csr", dtype=np.float32)
    Qm = sp.random(7, 60, density=0.3, random_state=2, format="csr", dtype=np.float32)
    t = top5(sp, Qm, Dm, block=3)
    sc = (Qm @ Dm.T).toarray()
    ref = np.array([np.lexsort((np.arange(40), -sc[q]))[:5] for q in range(7)])
    chk("top5 by blocks equals the full sort (score descending, id ascending)", bool(np.array_equal(t, ref)))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for i in range(3):
            sp.save_npz(shard_path(d, "d", i), Dm[i * 10:(i + 1) * 10], compressed=False)
        (d / "d_000099.tmp.npz").write_bytes(b"x")
        m = load_shards(sp, d, "d", 30)
        chk("shards stack in order and a leftover .tmp file is ignored", m.shape == (30, 60) and bool(np.array_equal(m.toarray(), Dm[:30].toarray())))
    chk("the shard names sort numerically for 99,999 shards", shard_path(Path("."), "d", 7).name == "d_000007.npz")
    chk("the gate share is the declared 99%", GATE_SHARE == 0.99)
    if bad:
        raise SystemExit(f"zleg1b: selftest FAILED ({bad})")
    log("selftest PASS (4 checks)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=lambda a: selftest())
    v = sub.add_parser("vectors")
    v.add_argument("--ds", required=True)
    v.add_argument("--device", default=None)
    v.add_argument("--threads", type=int, default=0)
    v.set_defaults(fn=cmd_vectors)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
