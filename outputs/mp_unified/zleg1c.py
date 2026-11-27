"""LEG1 part B reads (docs/LEG1_LEG_NECESSITY.md section 3): the stored C0 vectors with a cap on each node's or each question's non-zeros (B1, B2).

    python -E -P -B outputs/mp_unified/zleg1c.py selftest
    python -E -P -B outputs/mp_unified/zleg1c.py caps --ds multihoprag|squad     # laptop; reads outputs/leg1/vec/<ds> (fetched from the host) and the record's e3v list; writes outputs/leg1/caps_<ds>.json
    python -E -P -B outputs/mp_unified/zleg1c.py mix --ds multihoprag            # LEG1-A1 (section 7): the link-mixed vectors, 18 arms, the dev half picks, the held-out half calls; writes outputs/leg1/mix_<ds>.json

Run from the workspace root. zspl1.py, zspl1l.py, zleg1b.py, zul2p.py (load_mhr) and zenc.py are imported and edited nowhere.
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
ALPHAS = (0.25, 0.5, 1.0)          # LEG1-A1: the weight of the neighbours' mean
GRAPHS = ("R", "K", "RK")          # LEG1-A1: the rule links CN_all, the e3v kNN pairs Ke, their union
MIX_CAP = 128                      # LEG1-A1: the document cap of the capped arms
NOT_WORSE = -1.0                   # LEG1-A1: not WORSE = the lower end at least this
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


def adjacency(sp, keys, n):
    """The undirected 0/1 adjacency of pair keys a * n + b (a < b)."""
    keys = np.unique(np.asarray(keys, np.int64))
    a, c = keys // n, keys % n
    A = sp.csr_matrix((np.ones(2 * a.size, np.float32), (np.concatenate([a, c]), np.concatenate([c, a]))), shape=(n, n))
    A.data[:] = 1.0
    return A


def mix(sp, Dm, A, alpha):
    """v' = v + alpha * mean of the neighbours' vectors; a node with no neighbour is unchanged (LEG1-A1)."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0).astype(np.float32)
    out = (Dm + alpha * (sp.diags(inv) @ A @ Dm)).tocsr()
    out.sort_indices()
    return out


def pick_arm(dev_seeds, nnz):
    """The arm with the largest dev seeds mean; ties: the smaller number of non-zeros, then the name."""
    return sorted(dev_seeds, key=lambda k: (-round(dev_seeds[k], 12), nnz[k], k))[0]


def mix_call(lo_seeds, lo_others):
    """LEG1-A1: GAINS if BETTER on the seeds and not WORSE on the others; LEVEL if not BETTER but not WORSE anywhere; COSTS otherwise."""
    if min([lo_seeds] + list(lo_others)) < NOT_WORSE:
        return "MIXING_COSTS"
    return "MIXING_GAINS" if lo_seeds > 0 else "MIXING_LEVEL"


def enter_leave(ref_leg, new_leg, flat, off, depth=20):
    """Per question: golds outside the reference's top `depth` that enter the new top `depth` (recovered) and golds inside the reference's that leave it (lost)."""
    rec, lost = np.zeros(ref_leg.shape[0]), np.zeros(ref_leg.shape[0])
    for q in range(ref_leg.shape[0]):
        g = flat[off[q]:off[q + 1]]
        if g.size:
            a, b2 = np.isin(g, ref_leg[q, :depth]), np.isin(g, new_leg[q, :depth])
            rec[q], lost[q] = float(np.sum(b2 & ~a)), float(np.sum(a & ~b2))
    return rec, lost


def cmd_mix(a):
    import scipy.sparse as sp
    import zenc as E
    import zul2p
    t0 = time.time()
    if a.ds != "multihoprag":
        stop("LEG1-A1 declares MultiHop-RAG's graph only; the other datasets by their own amendment")
    d = OUT / "vec" / a.ds
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").is_file() else None
    if meta is None or not meta["gate_gb0"]["pass"]:
        stop(f"{a.ds}: no vectors or G-B0 did not pass: nothing is read")
    c = S.context(E, a.ds)
    n, dense, nq = c["w"].n, c["dense"], len(c["rows"])
    Dl = zul2p.load_mhr()
    if Dl.n != n:
        stop(f"the links' node count {Dl.n} is not the vectors' {n}")
    Qm, Dm = B.load_shards(sp, d, "q", nq), B.load_shards(sp, d, "d", n)
    keys = {"R": Dl.F["CN_all"], "K": Dl.F["Ke"], "RK": np.union1d(Dl.F["CN_all"], Dl.F["Ke"])}
    half = nq // 2
    dev, held = np.arange(nq) < half, np.arange(nq) >= half
    ref_leg = topk(sp, Qm, Dm, 1000)
    ref_r = reads_of_list(E, dense, ref_leg, c["flat"], c["off"], c["gt"], n)
    cells = [k for k in ("rrf5", "rrf20", "seeds", "leg20") if k in ref_r]
    res = {"stage": "LEG1 part B3 mix", "declared_in": DOC + " and section 7 (LEG1-A1)", "ds": a.ds, "questions": nq, "dev": int(dev.sum()), "held": int(held.sum()), "nodes": int(n),
           "links": {k: int(np.unique(v).size) for k, v in keys.items()}, "gate_gb0": meta["gate_gb0"],
           "reference": {"dev": {k: round(float(ref_r[k][dev].mean() * 100), 3) for k in cells}, "held": {k: round(float(ref_r[k][held].mean() * 100), 3) for k in cells}, "doc_nnz_mean": round(Dm.nnz / n, 2)},
           "arms": {}}
    dev_seeds, nnzs, store = {}, {}, {}
    for g in GRAPHS:
        A = adjacency(sp, keys[g], n)
        for al in ALPHAS:
            Dx = mix(sp, Dm, A, al)
            for size in ("full", f"cap{MIX_CAP}"):
                name = f"{g}_a{al}_{size}"
                D2 = Dx if size == "full" else cap_rows(sp, Dx, MIX_CAP)
                leg = topk(sp, Qm, D2, 1000)
                r = reads_of_list(E, dense, leg, c["flat"], c["off"], c["gt"], n)
                rc, ls = enter_leave(ref_leg, leg, c["flat"], c["off"])
                store[name] = r
                dev_seeds[name], nnzs[name] = float(r["seeds"][dev].mean()), D2.nnz / n
                res["arms"][name] = {"graph": g, "alpha": al, "size": size, "doc_nnz_mean": round(D2.nnz / n, 2), "nodes_changed": int(np.sum(np.asarray(A.sum(axis=1)).ravel() > 0)),
                                     "dev": {k: round(float(r[k][dev].mean() * 100), 3) for k in cells}, "held": {k: round(float(r[k][held].mean() * 100), 3) for k in cells},
                                     "leg20_golds_recovered_per_question": round(float(rc.mean()), 4), "leg20_golds_lost_per_question": round(float(ls.mean()), 4)}
                log(f"  {name}: nnz {D2.nnz / n:.0f}, dev seeds {dev_seeds[name] * 100:.2f}, held seeds {res['arms'][name]['held']['seeds']:.2f} (ref {res['reference']['held']['seeds']:.2f}), leg20 recovered {rc.mean():.3f} lost {ls.mean():.3f}")
    pk = pick_arm(dev_seeds, nnzs)
    diffs = {}
    for k in cells:
        dm, lo, hi = S.boot((store[pk][k] - ref_r[k])[held])
        diffs[k] = {"diff": round(dm, 3), "lo": round(lo, 3), "hi": round(hi, 3), "call": S.call_of(lo)}
    call = mix_call(diffs["seeds"]["lo"], [diffs[k]["lo"] for k in cells if k != "seeds"])
    res["pick"] = {"arm": pk, "dev_seeds": round(dev_seeds[pk] * 100, 3), "held_vs_reference": diffs, "call": call, "doc_nnz_ratio_to_reference": round(nnzs[pk] / (Dm.nnz / n), 3)}
    res["seconds"] = round(time.time() - t0, 1)
    S.wjson(OUT / f"mix_{a.ds}.json", res)
    log(f"pick {pk}: held-out seeds {diffs['seeds']['diff']:+.2f} [{diffs['seeds']['lo']:+.2f}, {diffs['seeds']['hi']:+.2f}] -> {call}")


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
    n = 4
    keys = np.array([0 * n + 1, 1 * n + 2], np.int64)                 # pairs (0,1) and (1,2); node 3 is alone
    A = adjacency(sp, keys, n)
    chk("the adjacency is undirected and 0/1", A.toarray().tolist() == [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]] and float(A.data.max()) == 1.0)
    Dv = sp.csr_matrix(np.array([[1, 0, 0], [0, 2, 0], [0, 0, 4], [5, 5, 5]], np.float32))
    M = mix(sp, Dv, A, 0.5).toarray()
    chk("mixing adds alpha times the mean of the neighbours; an isolated node is unchanged", np.allclose(M[0], [1, 1, 0]) and np.allclose(M[1], [0.25, 2, 1]) and np.allclose(M[2], [0, 1, 4]) and np.allclose(M[3], [5, 5, 5]))
    chk("the pick: the largest dev seeds, ties to fewer non-zeros", pick_arm({"a": 0.5, "b": 0.6, "c": 0.6}, {"a": 1, "b": 9, "c": 3}) == "c")
    chk("the call: gains need BETTER seeds and nothing WORSE; level and costs otherwise", (mix_call(0.2, [-0.5, 0.0]), mix_call(-0.3, [-0.5]), mix_call(0.4, [-1.2])) == ("MIXING_GAINS", "MIXING_LEVEL", "MIXING_COSTS"))
    flat, off = np.array([1, 2, 3], np.int64), np.array([0, 3], np.int64)
    rc, ls = enter_leave(np.array([[1, 9, 8, 7]]), np.array([[2, 3, 9, 8]]), flat, off, depth=2)
    chk("recovered and lost golds at a depth: golds 2 and 3 enter, gold 1 leaves", rc.tolist() == [2.0] and ls.tolist() == [1.0])
    chk("the arms: 3 graphs x 3 weights x 2 sizes = 18", len(GRAPHS) * len(ALPHAS) * 2 == 18)
    if bad:
        raise SystemExit(f"zleg1c: selftest FAILED ({bad})")
    log("selftest PASS (11 checks)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=lambda a: selftest())
    g = sub.add_parser("caps")
    g.add_argument("--ds", required=True, choices=("multihoprag", "squad"))
    g.set_defaults(fn=cmd_caps)
    m = sub.add_parser("mix")
    m.add_argument("--ds", required=True)
    m.set_defaults(fn=cmd_mix)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
