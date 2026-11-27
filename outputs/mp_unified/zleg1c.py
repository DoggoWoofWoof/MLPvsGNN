"""LEG1 part B reads (docs/LEG1_LEG_NECESSITY.md section 3): the stored C0 vectors with a cap on each node's or each question's non-zeros (B1, B2).

    python -E -P -B outputs/mp_unified/zleg1c.py selftest
    python -E -P -B outputs/mp_unified/zleg1c.py caps --ds multihoprag|squad     # laptop; reads outputs/leg1/vec/<ds> (fetched from the host) and the record's e3v list; writes outputs/leg1/caps_<ds>.json

Run from the workspace root. The link-mixed variant (B3) is not here: it is declared by the amendment LEG1-A1 first. zspl1.py, zspl1l.py, zleg1b.py and zenc.py are imported and edited nowhere.
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
import zleg1b as B  # noqa: E402
import zspl1 as S  # noqa: E402

DOC = "docs/LEG1_LEG_NECESSITY.md section 3"
OUT = S.WS / "outputs" / "leg1"
DOC_CAPS = (128, 64, 32)
Q_CAPS = (32, 16, 8)
FREE_PTS = -1.0                    # FREE if the lower end of the difference is at least this on every read
log, stop = S.log, S.stop


def cap_rows(sp, M, k):
    """Keep each row's k largest weights (ties by column index, smaller first); rows with fewer non-zeros are unchanged."""
    M = M.tocsr()
    ip, ix, dt = M.indptr, M.indices, M.data
    keep_i, keep_d, new_ip = [], [], [0]
    for r in range(M.shape[0]):
        a, b = ip[r], ip[r + 1]
        c, v = ix[a:b], dt[a:b]
        if c.size > k:
            o = np.lexsort((c, -v))[:k]
            o.sort()
            c, v = c[o], v[o]
        keep_i.append(c)
        keep_d.append(v)
        new_ip.append(new_ip[-1] + c.size)
    out = sp.csr_matrix((np.concatenate(keep_d) if keep_d else np.zeros(0, M.dtype), np.concatenate(keep_i) if keep_i else np.zeros(0, np.int32), np.array(new_ip)), shape=M.shape)
    out.sort_indices()
    return out


def topk(sp, Q, Dm, k, block=256):
    """Top k nodes of Q @ Dm.T per question (score descending, node id ascending on ties); -1 pads where fewer than k exist."""
    n = Dm.shape[0]
    k = min(k, n)
    out = np.full((Q.shape[0], k), -1, np.int64)
    DT = Dm.T.tocsr()
    for i in range(0, Q.shape[0], block):
        sc = (Q[i:i + block] @ DT).toarray()
        for r in range(sc.shape[0]):
            s = sc[r]
            if n > 4 * k:
                c = np.argpartition(-s, k - 1)[:k]
                thr = s[c].min()
                c = np.flatnonzero(s >= thr)                      # every node tied with the k-th score is kept before the id order cuts
            else:
                c = np.arange(n)
            o = np.lexsort((c, -s[c]))[:k]
            out[i + r, :o.size] = c[o]
    return out


def reads_of_list(E, dense, leg, flat, off, gt, n):
    """SPLADE leg's R@20, R@1000 and the fusion's R@5, R@20 and seeds (as zspl1.reads_of)."""
    r, _ = S.reads_of(E, dense, leg, flat, off, gt, n)
    return r


def cmd_caps(a):
    import scipy.sparse as sp
    import zenc as E
    t0 = time.time()
    d = OUT / "vec" / a.ds
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").is_file() else None
    if meta is None or not meta["gate_gb0"]["pass"]:
        stop(f"{a.ds}: no vectors or G-B0 did not pass: nothing is read")
    c = S.context(E, a.ds)
    n, dense = c["w"].n, c["dense"]
    Qm, Dm = B.load_shards(sp, d, "q", len(c["rows"])), B.load_shards(sp, d, "d", n)
    ref_leg = topk(sp, Qm, Dm, 1000)
    ref_r = reads_of_list(E, dense, ref_leg, c["flat"], c["off"], c["gt"], n)
    log(f"{a.ds}: reference read in {time.time() - t0:.0f} s: " + json.dumps({k: round(float(v.mean() * 100), 3) for k, v in ref_r.items()}))
    res = {"stage": "LEG1 part B caps", "declared_in": DOC, "ds": a.ds, "questions": int(len(c["rows"])), "nodes": int(n), "gate_gb0": meta["gate_gb0"],
           "reference": {"level": {k: round(float(v.mean() * 100), 3) for k, v in ref_r.items()}, "doc_nnz_mean": round(Dm.nnz / n, 2), "query_nnz_mean": round(Qm.nnz / Qm.shape[0], 2)}, "arms": {}}
    arms = [(f"doc{k}", k, None) for k in DOC_CAPS] + [(f"query{k}", None, k) for k in Q_CAPS]
    for name, kd, kq in arms:
        D2 = cap_rows(sp, Dm, kd) if kd else Dm
        Q2 = cap_rows(sp, Qm, kq) if kq else Qm
        leg = topk(sp, Q2, D2, 1000)
        r = reads_of_list(E, dense, leg, c["flat"], c["off"], c["gt"], n)
        diffs, worst = {}, 1e9
        for cell in ref_r:
            dm, lo, hi = S.boot(r[cell] - ref_r[cell])
            diffs[cell] = {"level": round(float(r[cell].mean() * 100), 3), "diff": round(dm, 3), "lo": round(lo, 3), "hi": round(hi, 3), "call": S.call_of(lo)}
            worst = min(worst, lo)
        res["arms"][name] = {"doc_nnz_mean": round(D2.nnz / n, 2), "query_nnz_mean": round(Q2.nnz / Q2.shape[0], 2), "reads": diffs, "call": "FREE" if worst >= FREE_PTS else "COSTS", "worst_lo": round(worst, 3)}
        log(f"  {name}: nnz doc {res['arms'][name]['doc_nnz_mean']} query {res['arms'][name]['query_nnz_mean']} -> {res['arms'][name]['call']} (worst lower end {worst:.2f}); " +
            "; ".join(f"{k} {v['diff']:+.2f}" for k, v in diffs.items()))
    res["seconds"] = round(time.time() - t0, 1)
    S.wjson(OUT / f"caps_{a.ds}.json", res)


def selftest():
    import scipy.sparse as sp
    bad = 0

    def chk(name, ok):
        nonlocal bad
        log(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += 0 if ok else 1

    M = sp.csr_matrix(np.array([[0, 5, 3, 0, 4, 1], [2, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], np.float32))
    C = cap_rows(sp, M, 2)
    chk("a cap keeps the k largest weights per row and leaves short rows alone", C.getrow(0).toarray().tolist() == [[0, 5, 0, 0, 4, 0]] and C.getrow(1).nnz == 1)
    chk("ties go to the smaller column", C.getrow(2).indices.tolist() == [0, 1])
    rng = np.random.default_rng(3)
    Dm = sp.random(30, 50, density=0.3, random_state=4, format="csr", dtype=np.float32)
    Qm = sp.random(5, 50, density=0.4, random_state=5, format="csr", dtype=np.float32)
    t = topk(sp, Qm, Dm, 10, block=2)
    sc = (Qm @ Dm.T).toarray()
    ref = np.array([np.lexsort((np.arange(30), -sc[q]))[:10] for q in range(5)])
    chk("topk equals the full sort on a toy (score descending, id ascending)", bool(np.array_equal(t, ref)))
    Dm2 = sp.csr_matrix(np.ones((50, 4), np.float32))
    Q2 = sp.csr_matrix(np.ones((2, 4), np.float32))
    t2 = topk(sp, Q2, Dm2, 3, block=1)
    chk("topk breaks an all-tie by node id (the argpartition shortcut keeps every tied node)", t2.tolist() == [[0, 1, 2], [0, 1, 2]])
    chk("FREE needs a lower end of -1.0 or better", FREE_PTS == -1.0)
    if bad:
        raise SystemExit(f"zleg1c: selftest FAILED ({bad})")
    log("selftest PASS (5 checks)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=lambda a: selftest())
    g = sub.add_parser("caps")
    g.add_argument("--ds", required=True, choices=("multihoprag", "squad"))
    g.set_defaults(fn=cmd_caps)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
