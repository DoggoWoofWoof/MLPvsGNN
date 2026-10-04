"""Design look (untracked; not a result and not filed): how familiar each query is to a model's training queries, from
the query's embedding alone (no label, no candidate, no edge, no score).

Why. On squad x1 every lean read with SEMB in it is below rrf alone, and a lean model fitted without SEMB reaches it;
on the training graph SEMB is what carries the lean MLP to the GNN. lean_read9 reads block-dropout models with SEMB on
and off on the same rows. A switch between the two paths needs a label-free signal that a query is unlike the queries
the model trained on. This look computes one per row, in the row order and filter of the lean reads, so it pairs row by
row with lean_read9's and lean_mlp8's rows files (keys fam@ds=carve beside their NAME@ds=carve).

rows         lean_mlp.Carve's: the carve's chunk files whose shards have filed their records, in name order, each
             chunk's rows in order; the query embedding as the carve stores it (float16). The rows files keep rows with
             gold (gold_total > 0), and so does fam@.
familiarity  fam(q) = the mean cosine of q's embedding with its k nearest bank queries (k = 10; --k), every embedding
             at unit length.
bank         every row of the --bank carves (the models' training carves), gold or not. A --read or --held carve may
             not be a bank carve (its rows would find themselves).
threshold    tau_p = the p-th percentile of fam over the --held carves' gold rows (the training graphs' select carves:
             queries of the training graphs that no fit trained on), p in --pcts. A query with fam < tau_p would take
             the off path. tau_p reads no read carve and no label.
reads        per --read carve: fam's percentiles, and the share of its gold rows below each tau_p.

A carve whose filed records do not list all its chunks is refused (its rows would not be the full read's), unless
--partial.

    python outputs/mp_unified/qfam9.py --bank 2wiki=fit --held 2wiki=select \\
        --read 2wiki=x1,hotpotqa=x1,squad=x1,metaqa=x1f,webqsp=selectf --out outputs/mp_unified/lean/fam9-2w.json
    python outputs/mp_unified/qfam9.py --selftest
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOOK = HERE / "look"

import numpy as np  # noqa: E402


def log(msg):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def parse_sets(spec):
    out = []
    for part in [p for p in spec.split(",") if p.strip()]:
        ds, cv = part.split("=")
        out.append((ds.strip(), cv.strip()))
    return out


def carve_rows(ds, cv, root=LOOK, partial=False):
    """(q_emb (rows, d) float32 from the stored float16, gold_total (rows,), info): lean_mlp.Carve's rows and order."""
    d = Path(root) / ds / cv
    recs = sorted(d.glob("record*.json"))
    if not recs:
        raise SystemExit(f"{d}: no look record")
    r0 = json.loads(recs[0].read_text(encoding="utf-8"))
    listed = set()
    for r_ in recs:
        listed |= set(json.loads(r_.read_text(encoding="utf-8"))["chunks"])
    n_chunks = int(r0["n_chunks"])
    if len(listed) != n_chunks and not partial:
        raise SystemExit(f"{ds}={cv}: records list {len(listed)} of {n_chunks} chunks (--partial to read them anyway)")
    qe, gt, read = [], [], 0
    for ch in sorted((d / "chunks").glob("c*.npz")):
        if int(ch.stem[1:]) not in listed:
            continue
        with np.load(ch) as z:
            qe.append(z["q_emb"].astype(np.float16).astype(np.float32))
            gt.append(z["q_gold_total"].astype(np.int64))
        read += 1
    if read != len(listed):
        raise SystemExit(f"{ds}={cv}: {len(listed)} chunks listed, {read} chunk files found")
    q = np.concatenate(qe)
    return q, np.concatenate(gt), {"chunks_read": read, "n_chunks": n_chunks, "carve_queries": int(r0["carve_queries"]),
                                   "rows": int(q.shape[0])}


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def knn_fam(q, bank, k, step=512):
    """Mean cosine of each row of q with its k nearest rows of bank (both unit length), float64."""
    k = min(k, bank.shape[0])
    out = np.empty(q.shape[0], np.float64)
    for s in range(0, q.shape[0], step):
        sim = q[s:s + step] @ bank.T
        top = np.partition(sim, sim.shape[1] - k, axis=1)[:, sim.shape[1] - k:]
        out[s:s + step] = top.astype(np.float64).mean(1)
    return out


def summary(f):
    return {f"p{p}": float(np.percentile(f, p)) for p in (1, 5, 10, 25, 50, 75, 90)} | {"mean": float(f.mean()), "rows": int(f.size)}


def run(a, root=LOOK):
    bank_sets, held_sets, read_sets = parse_sets(a.bank), parse_sets(a.held), parse_sets(a.read)
    clash = (set(held_sets) | set(read_sets)) & set(bank_sets)
    if clash:
        raise SystemExit(f"a held or read carve is a bank carve: {sorted(clash)}")
    pcts = [float(p) for p in a.pcts.split(",") if p]
    t0 = time.time()
    banks, info = [], {}
    for ds, cv in bank_sets:
        q, _gt, inf = carve_rows(ds, cv, root, a.partial)
        banks.append(unit(q))
        info[f"bank {ds}={cv}"] = inf
    bank = np.concatenate(banks)
    log(f"bank {bank_sets}: {bank.shape[0]} queries ({time.time() - t0:.0f}s)")
    fams, out_reads = {}, {}
    held = []
    for ds, cv in held_sets:
        q, gt, inf = carve_rows(ds, cv, root, a.partial)
        f = knn_fam(unit(q), bank, a.k)[gt > 0]
        fams[f"fam@{ds}={cv}"] = f.astype(np.float32)
        info[f"held {ds}={cv}"] = inf
        held.append(f)
        log(f"held {ds}={cv}: {summary(f)}")
    hf = np.concatenate(held)
    tau = {f"p{p:g}": float(np.percentile(hf, p)) for p in pcts}
    log(f"thresholds from {hf.size} held rows: {tau}")
    for ds, cv in read_sets:
        q, gt, inf = carve_rows(ds, cv, root, a.partial)
        f = knn_fam(unit(q), bank, a.k)[gt > 0]
        fams[f"fam@{ds}={cv}"] = f.astype(np.float32)
        info[f"read {ds}={cv}"] = inf
        out_reads[f"{ds}={cv}"] = {"fam": summary(f), "below": {t: float((f < v).mean()) for t, v in tau.items()}}
        log(f"read {ds}={cv}: {out_reads[f'{ds}={cv}']}")
    res = {"look": "qfam9", "args": vars(a), "k": a.k, "bank_rows": int(bank.shape[0]), "tau": tau,
           "held": {k_: summary(v.astype(np.float64)) for k_, v in fams.items() if k_.split("@")[1] in {f"{d}={c}" for d, c in held_sets}},
           "reads": out_reads, "carves": info, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        npz = p.with_suffix(".fam.npz")
        tmp = npz.with_name(npz.stem + ".tmp.npz")
        np.savez_compressed(tmp, **fams)
        os.replace(tmp, npz)
        res["fam_file"] = {"path": npz.name, "keys": sorted(fams), "sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}
        tmpj = p.with_name(p.name + ".tmp")
        tmpj.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmpj, p)
        log(f"wrote {p} and {npz}")
    res["seconds"] = time.time() - t0
    return res, fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="2wiki=fit")
    ap.add_argument("--held", default="2wiki=select")
    ap.add_argument("--read", default="2wiki=x1")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pcts", default="1,5,10,25")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    run(a)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def _toy_look(root, ds, cv, rows_per_chunk, seed, listed=None, d=8):
    rng = np.random.default_rng(seed)
    dd = Path(root) / ds / cv / "chunks"
    dd.mkdir(parents=True, exist_ok=True)
    qs, gts = [], []
    for i, r in enumerate(rows_per_chunk):
        q = rng.standard_normal((r, d)).astype(np.float32)
        gt = rng.integers(0, 3, size=r)
        np.savez(dd / f"c{i:05d}.npz", q_emb=q, q_gold_total=gt, q_pool_size=np.full(r, 5))
        qs.append(q)
        gts.append(gt)
    chunks = list(range(len(rows_per_chunk))) if listed is None else listed
    (Path(root) / ds / cv / "record_0of1.json").write_text(json.dumps(
        {"chunks": chunks, "n_chunks": len(rows_per_chunk), "carve_queries": int(sum(rows_per_chunk))}), encoding="utf-8")
    return np.concatenate(qs), np.concatenate(gts)


def selftest():
    import tempfile
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    # 1. knn_fam: the mean of the k largest cosines, by brute force; a bank row read back has top-1 cosine 1
    B = unit(rng.standard_normal((300, 16)))
    Q = unit(np.concatenate([B[:7], rng.standard_normal((50, 16))]))
    for k in (1, 3, 10):
        want = np.sort(Q @ B.T, axis=1)[:, -k:].mean(1)
        got = knn_fam(Q, B, k, step=13)
        assert np.allclose(got, want, atol=1e-6), k
    assert np.allclose(knn_fam(Q[:7], B, 1), 1.0, atol=1e-6)
    with tempfile.TemporaryDirectory() as td:
        # 2. carve_rows: listed chunks only, in name order, float16 as stored; an unlisted chunk is refused unless
        #    --partial (then it is skipped, as lean_mlp.Carve skips it)
        q, gt = _toy_look(td, "g", "fit", [5, 7, 4], 1)
        q2, gt2, inf = carve_rows("g", "fit", td)
        assert np.array_equal(q2, q.astype(np.float16).astype(np.float32)) and np.array_equal(gt2, gt) and inf["rows"] == 16
        qa, gta = _toy_look(td, "g", "x1", [6, 6, 6], 2, listed=[0, 2])
        try:
            carve_rows("g", "x1", td)
            raise AssertionError("incomplete carve")
        except SystemExit:
            pass
        qp, gtp, infp = carve_rows("g", "x1", td, partial=True)
        assert np.array_equal(gtp, np.concatenate([gta[:6], gta[12:]])) and infp["chunks_read"] == 2
        _toy_look(td, "g", "select", [9, 3], 3)
        _toy_look(td, "h", "x1", [8], 4)
        # 3. a whole run: thresholds from the held gold rows, shares below them, fam@ keys for gold rows in carve order
        a = SimpleNamespace(bank="g=fit", held="g=select", read="h=x1", k=3, pcts="5,50", partial=False,
                            out=str(Path(td) / "o" / "f.json"))
        res, fams = run(a, td)
        qs_, gs_, _ = carve_rows("g", "select", td)
        fh = knn_fam(unit(qs_), unit(q.astype(np.float16).astype(np.float32)), 3)[gs_ > 0]
        assert np.allclose(fams["fam@g=select"], fh.astype(np.float32))
        assert abs(res["tau"]["p50"] - float(np.percentile(fh, 50))) < 1e-12
        qh, gh, _ = carve_rows("h", "x1", td)
        fr = knn_fam(unit(qh), unit(q.astype(np.float16).astype(np.float32)), 3)[gh > 0]
        assert abs(res["reads"]["h=x1"]["below"]["p50"] - float((fr < res["tau"]["p50"]).mean())) < 1e-12
        z = np.load(Path(td) / "o" / "f.fam.npz")
        assert sorted(z.files) == ["fam@g=select", "fam@h=x1"] and z["fam@h=x1"].shape == (int((gh > 0).sum()),)
        z.close()
        # 4. a read carve that is a bank carve is refused
        try:
            run(SimpleNamespace(**{**vars(a), "read": "g=fit", "out": None}), td)
            raise AssertionError("bank clash")
        except SystemExit:
            pass
    # 5. on a real local look carve (when present), the rows and filter are lean_mlp.Carve's
    if (LOOK / "2wiki" / "select" / "chunks").exists():
        sys.path.insert(0, str(HERE))
        import lean_mlp as LM
        c = LM.Carve("2wiki", "select")
        q, gt, _ = carve_rows("2wiki", "select")
        assert np.array_equal(gt, c.gold_total) and np.array_equal(q, c.q_emb.astype(np.float32)), "rows must be lean_mlp.Carve's"
        print(f"  checked against lean_mlp.Carve on 2wiki=select ({c.rows} rows)")
    print("selftest: knn familiarity is the brute-force mean of the k largest cosines; rows are the filed chunks' in "
          "name order (float16 as stored), incomplete carves refused unless --partial; thresholds come from the held "
          "gold rows and shares below them are computed on read gold rows; a bank carve cannot be read. all checks passed")


if __name__ == "__main__":
    main()
