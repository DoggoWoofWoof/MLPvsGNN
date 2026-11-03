"""PFS amendment A4 (docs/PKO_FEATURE_SCREEN.md section 13): section 11's six pair columns against other anchors, on
section 11's questions and rows, read with the same probe.

Pass 1 is zpfp's read, unchanged (zpfp.read): the lean columns raw and z-scored, A3's twelve (P5), the labels and rrf's
rank. Then, from the frozen encoder's full vectors of the same rows, the same six columns (raw, then z-scored within the
question) against three other anchor sets:

    P10  rrf's top ten rows
    R2   the top five of pass 1's probe on the lean columns and P5 (scored crossfit by question parity; ties by position;
         its first row stands for rrf's first in pair_1 and res_1)
    R3   the top five of the R2 probe's scores, the same way (beside, deciding nothing)

Probes (zpfs.lr_crossfit, section 9's): lean, lean + P5, lean + P10, lean + P5 + R2, lean + P5 + R3, and beside,
lean + P5 + P10. Two checks refuse the read: the columns' code, given rrf's top five, must give A3's twelve on the first
batch; and lean and lean + P5 must reproduce A3's recall@5 (outputs/pfs/pair_<dataset>.json, else section 12's printed
numbers) within 0.002.

zpfp, zpfs, ltwdiag, lean_gpu, lean_cache and lean_mlp3 are imported and not edited; PKO's cache and the node vectors are
read only; this writes only outputs/pfs/anchors_<dataset>.json.

    python outputs/mp_unified/zpfq.py anchors --dataset musique --threads 2 --host
    python outputs/mp_unified/zpfq.py --selftest
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import zpfp as PP  # noqa: E402  (A3's read, columns and carve loading; not edited)
import zpfs as PF  # noqa: E402  (the probe and metrics; not edited)
import ltwdiag as LTW  # noqa: E402  (the free scores; not edited)

DECLARED = PF.DECLARED
AMENDMENT = "A4"
NAMES = PP.NAMES
TOP, TEN = PP.TOP, 10
BATCH = PP.BATCH
ADDS, NOTHING = PP.ADDS, PP.NOTHING              # section 13's thresholds, section 11's
REPRO = 0.002                                     # pass 1's probes against A3's recall@5
MACHINE = 1e-6                                    # the columns' code against A3's twelve on the first batch
SEC12 = {"musique": (0.6500, 0.6663), "hotpotqa": (0.8053, 0.8223), "2wiki": (0.8085, 0.8203),
         "squad": (0.9518, 0.9558), "webqsp": (0.2919, 0.3300)}        # section 12's lean and lean + pair recall@5
PROBES = (("lean", ("XN",)), ("lean+P5", ("XN", "P5")), ("lean+P10", ("XN", "P10")), ("lean+P5+R2", ("XN", "P5", "R2")),
          ("lean+P5+R3", ("XN", "P5", "R3")), ("lean+P5+P10", ("XN", "P5", "P10")))


def log(m):
    PF.log(m)


def call_of(d, form):
    """Section 13's call on d, for form TOP10 or R2."""
    if d >= ADDS:
        return f"{form}_ADDS"
    if d <= -ADDS:
        return f"{form}_HURTS"
    if abs(d) < NOTHING:
        return f"{form}_NOTHING"
    return "MIXED"


def cols_k(X, qv, top):
    """zpfp.pair_cols over any anchors: X (n, d) unit rows of one question, qv (d,) its unit vector, top (k,) anchor rows
    in order. (n, 6) in NAMES order; an anchor is not paired with itself; pair_1 and res_1 read top[0], top[1] for top[0]."""
    n, k = X.shape[0], len(top)
    P = np.empty((n, k), np.float64)
    R = np.empty((n, k), np.float64)
    for j in range(k):
        _cv, _s, _ca, res, pair = LTW.pair_feats(qv, X[top[j]], X)
        P[:, j], R[:, j] = pair, res
    P[top, np.arange(k)] = np.nan
    R[top, np.arange(k)] = np.nan
    k1 = np.zeros(n, np.int64)
    k1[top[0]] = 1
    ar = np.arange(n)
    return np.stack([np.nanmax(P, 1), np.nanmax(R, 1), np.nanmean(P, 1), np.nanmean(R, 1), P[ar, k1], R[ar, k1]], 1)


def rrf_top(rank, k):
    """The rows of rrf's first k (at most the rows less one) in rrf's order; refuses ranks that do not start 0..k-1."""
    k = min(k, rank.size - 1)
    top = np.argsort(rank, kind="stable")[:k]
    if not np.array_equal(rank[top], np.arange(k)):
        raise SystemExit(f"zpfq: rrf's ranks do not start 0..{k - 1}")
    return top


def score_top(s, k=TOP):
    """The rows of the k highest scores, ties by position."""
    return np.argsort(-np.asarray(s, np.float64), kind="stable")[:k]


def anchor_block(c, qs_all, nd, node_of, pick):
    """The six columns raw then z-scored over every row of questions qs_all (consecutive from 0, the table's order: a
    table row is its carve row), BATCH at a time, against pick(rows) for each question's table rows. Also how many
    questions had fewer anchors than asked."""
    import lean_gpu as LG
    out, short = [], 0
    qs_all = np.asarray(qs_all, np.int64)
    for k in range(0, qs_all.size, BATCH):
        qs = qs_all[k:k + BATCH]
        cnt = c.n_np[qs]
        seg = np.cumsum(cnt) - cnt
        N = int(cnt.sum())
        idx = np.repeat(c.off_np[qs] - seg, cnt) + np.arange(N)
        uq, inv = np.unique(node_of(idx), return_inverse=True)
        V = LTW.unit(np.asarray(nd.read(uq), np.float32))
        Q = LTW.unit(c.q_emb[torch.from_numpy(qs)].to(torch.float32).numpy())
        D = np.empty((N, len(NAMES)), np.float64)
        for b in range(qs.size):
            s, e = int(seg[b]), int(seg[b] + cnt[b])
            top, want = pick(idx[s:e])
            short += int(len(top) < want)
            D[s:e] = cols_k(V[inv[s:e]], Q[b], top)
        Dt = torch.from_numpy(D.astype(np.float32))
        nqt = torch.from_numpy(np.repeat(np.arange(qs.size), cnt))
        out.append(torch.cat([Dt, LG.seg_zscore8D(Dt, nqt, qs.size)], 1).numpy())
    return np.concatenate(out), short


def metrics(s, y, q):
    return {"auc": PF.auc(s, y), **PF.hit_at(s, y, q), **{k: v for k, v in PF.recall_at(s, y, q).items() if k != "questions"}}


def by_hop(scores, y, q, ids):
    hop = np.asarray([int(str(x)[0]) if str(x)[:1] in "234" else 0 for x in ids], np.int64)[q]
    out = {}
    for h in (2, 3, 4):
        m = hop == h
        if m.any():
            out[f"{h}hop"] = {nm: PF.recall_at(s[m], y[m], q[m])["recall@5"] for nm, s in scores.items()}
    return out


def a3_reference(ds, out_dir):
    """A3's lean and lean + pair recall@5 for ds: from its JSON when it is here, else section 12's printed numbers."""
    p = Path(out_dir) / f"pair_{ds}.json"
    if p.is_file():
        r = json.loads(p.read_text(encoding="utf-8"))
        return {"from": str(p.as_posix()), "lean": r["lr_lean"]["recall@5"], "lean+P5": r["lr_lean_plus_pair"]["recall@5"]}
    lean, plus = SEC12[ds]
    return {"from": "section 12", "lean": lean, "lean+P5": plus}


def repro_check(ref, got):
    gaps = {k: abs(got[k] - ref[k]) for k in ("lean", "lean+P5")}
    return {"reference": ref, "got": {k: got[k] for k in ("lean", "lean+P5")}, "gaps": gaps,
            "pass": all(v <= REPRO for v in gaps.values())}


def read_all(c, nq, nd, node_of, ref=None):
    """Pass 1 (zpfp.read), the machinery check, the anchor sets in order (R2 after lean + P5's scores, R3 after R2's,
    P10), every probe and the calls. ref: A3's numbers for the reproduction check (None: not checked)."""
    t = PP.read(c, nq, nd, node_of)
    y, q, rank = t["y"], t["q"], t["rank"]
    blocks = {"XN": t["XN"], "P5": t["D"]}
    qs = np.arange(nq)
    n0 = min(BATCH, nq)
    first, _ = anchor_block(c, qs[:n0], nd, node_of, lambda r: (rrf_top(rank[r], TOP), TOP))
    gap = float(np.abs(first - blocks["P5"][:first.shape[0]]).max())
    machinery = {"rows": int(first.shape[0]), "max_abs": gap, "pass": gap <= MACHINE}
    if not machinery["pass"]:
        raise SystemExit(f"zpfq: the columns' code does not give A3's twelve on the first batch ({gap})")
    scores, short = {}, {}

    def probe(nm, keys):
        X = np.concatenate([blocks[k] for k in keys], 1) if len(keys) > 1 else blocks[keys[0]]
        scores[nm], _w = PF.lr_crossfit(X, y, q)
        log(f"   probe {nm}: {X.shape[1]} columns, recall@5 {PF.recall_at(scores[nm], y, q)['recall@5']:.4f}")

    keys = dict(PROBES)
    probe("lean", keys["lean"])
    probe("lean+P5", keys["lean+P5"])
    got = {nm: PF.recall_at(scores[nm], y, q)["recall@5"] for nm in ("lean", "lean+P5")}
    repro = repro_check(ref, got) if ref is not None else None
    if repro is not None and not repro["pass"]:
        raise SystemExit(f"zpfq: pass 1's probes do not reproduce A3's recall@5: {repro}")
    for nm, prev in (("R2", "lean+P5"), ("R3", "lean+P5+R2")):
        s_prev = scores[prev]
        blocks[nm], short[nm] = anchor_block(c, qs, nd, node_of, lambda r, s=s_prev: (score_top(s[r]), TOP))
        probe(f"lean+P5+{nm}", keys[f"lean+P5+{nm}"])
    blocks["P10"], short["P10"] = anchor_block(c, qs, nd, node_of, lambda r: (rrf_top(rank[r], TEN), TEN))
    probe("lean+P10", keys["lean+P10"])
    probe("lean+P5+P10", keys["lean+P5+P10"])
    out = {"rows": int(y.size), "gold_rows": int(y.sum()), "identity": t["identity"], "machinery": machinery,
           "repro": repro, "short_anchor_questions": short, "probes": {nm: metrics(scores[nm], y, q) for nm, _k in PROBES}}
    r5 = {nm: out["probes"][nm]["recall@5"] for nm, _k in PROBES}
    out["d10"] = r5["lean+P10"] - r5["lean+P5"]
    out["d2"] = r5["lean+P5+R2"] - r5["lean+P5"]
    out["d3"] = r5["lean+P5+R3"] - r5["lean+P5"]
    out["d5_10"] = r5["lean+P5+P10"] - r5["lean+P5"]
    out["d_pair"] = r5["lean+P5"] - r5["lean"]
    out["call_top10"] = call_of(out["d10"], "TOP10")
    out["call_r2"] = call_of(out["d2"], "R2")
    out["anchor_overlap"] = overlap(rank, scores, q)
    return out, scores, t


def overlap(rank, scores, q):
    """Mean share of rrf's top five among each probe round's top five (how far the re-anchoring moves the anchors)."""
    o = {}
    qq = np.asarray(q, np.int64)
    starts = np.flatnonzero(np.r_[True, qq[1:] != qq[:-1]])
    ends = np.r_[starts[1:], qq.size]
    for nm in ("lean+P5", "lean+P5+R2"):
        s = scores[nm]
        o[nm] = float(np.mean([len(set(rrf_top(rank[a:b], TOP)) & set(score_top(s[a:b]))) / TOP
                               for a, b in zip(starts, ends)]))
    return o


def anchors_cmd(a):
    import lean_cache as LC
    import lean_mlp3 as L3
    t0 = time.time()
    carve = PF.CARVE[a.dataset]
    nq_want = a.nq or PF.NQ[a.dataset]
    with PP.on_prefix(a.dataset, carve, nq_want, PF.CACHE) as (parts, recs):
        c = PP.open_carve(a.dataset, carve)
    nq = min(nq_want, c.rows)
    nodes, freeze = L3.open_nodes([a.dataset])
    nid = PP.node_index(parts, recs, int(c.off_np[nq]))
    ref = a3_reference(a.dataset, PF.OUT) if nq == PF.NQ[a.dataset] else None
    r, scores, t = read_all(c, nq, nodes[a.dataset], lambda i: nid[i], ref)
    if not r["identity"]["pass"]:
        raise SystemExit(f"zpfq {a.dataset}: identity FAIL {r['identity']}")
    out = {"declared_in": DECLARED, "amendment": AMENDMENT, "dataset": a.dataset, "carve": c.carve, "questions": nq,
           "parts_loaded": len(parts), "part_shas": c.part_shas, "nodes_record_sha256": freeze, **r}
    if a.dataset == PP.HOP_DATASET:
        out["by_hop"] = by_hop(scores, t["y"], t["q"], c.ids[:nq])
    out["host"] = bool(a.host)
    out["script_sha256"] = LC.sha_file(Path(__file__))
    out["zpfp_sha256"] = LC.sha_file(HERE / "zpfp.py")
    out["seconds"] = time.time() - t0
    p = Path(PF.OUT) / f"anchors_{a.dataset}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"anchors {a.dataset}: {nq} questions, {out['rows']} rows, machinery {r['machinery']['max_abs']:.1e}, repro "
        f"{None if r['repro'] is None else r['repro']['pass']} ({out['seconds']:.0f}s) -> {p}")
    for nm, _k in PROBES:
        m = out["probes"][nm]
        log(f"   {nm:12s} auc {m['auc']:.4f} recall@5 {m['recall@5']:.4f} recall@20 {m['recall@20']:.4f} hit@5 {m['hit@5']:.3f}")
    log(f"   d10 {out['d10']:+.4f} {out['call_top10']}; d2 {out['d2']:+.4f} {out['call_r2']}; beside d3 {out['d3']:+.4f}, "
        f"d5_10 {out['d5_10']:+.4f}; rrf's top five kept {out['anchor_overlap']}")
    return 0


# ── self-test (CPU, no cache) ────────────────────────────────────────────────


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    # 1. the columns over five rrf anchors are zpfp's exactly; over ten, the same rule anchor by anchor
    X = LTW.unit(rng.standard_normal((40, 16)).astype(np.float32))
    qv = LTW.unit(rng.standard_normal(16).astype(np.float32))
    top = np.array([7, 3, 11, 0, 25])
    assert np.array_equal(cols_k(X, qv, top), PP.pair_cols(X, qv, top)[0])
    top10 = np.array([7, 3, 11, 0, 25, 30, 2, 19, 38, 5])
    C = cols_k(X, qv, top10)
    want_p = np.stack([LTW.pair_feats(qv, X[t], X)[4] for t in top10], 1)
    want_r = np.stack([LTW.pair_feats(qv, X[t], X)[3] for t in top10], 1)
    for i in range(40):
        keep = top10 != i
        assert abs(C[i, 0] - want_p[i, keep].max()) < 1e-12 and abs(C[i, 3] - want_r[i, keep].mean()) < 1e-12
        k1 = 1 if i == top10[0] else 0
        assert C[i, 4] == want_p[i, k1] and C[i, 5] == want_r[i, k1]
    assert not np.isnan(C).any()
    # 2. anchors: rrf's (checked, at most the rows less one) and a score's (ties by position); the calls
    assert list(rrf_top(np.array([2, 0, 1, 4, 3]), 10)) == [1, 2, 0, 4]
    try:
        rrf_top(np.array([1, 2, 3, 4, 5, 6]), 5)
        raise AssertionError("ranks not starting at 0 must refuse")
    except SystemExit:
        pass
    assert list(score_top(np.array([0.1, 0.5, 0.5, 0.9, 0.0, 0.5]), 4)) == [3, 1, 2, 5]
    assert [call_of(x, "R2") for x in (0.012, 0.01, 0.006, 0.004, -0.004, -0.006, -0.011)] == [
        "R2_ADDS", "R2_ADDS", "MIXED", "R2_NOTHING", "R2_NOTHING", "MIXED", "R2_HURTS"]
    assert call_of(0.02, "TOP10") == "TOP10_ADDS"
    # 3. the read on zpfp's three-question carve: the machinery check, every block, the probes, the calls
    c = PP.FakeCarve()
    nd = PP.FakeNodes(c.V)
    r, scores, t = read_all(c, 3, nd, lambda i: c.node[i])
    assert r["machinery"]["pass"] and r["machinery"]["rows"] == 180 and r["machinery"]["max_abs"] == 0.0, r["machinery"]
    assert r["identity"]["pass"] and r["repro"] is None and set(r["probes"]) == {nm for nm, _k in PROBES}
    assert set(r["short_anchor_questions"]) == {"R2", "R3", "P10"} and not any(r["short_anchor_questions"].values())
    assert r["call_top10"] == call_of(r["d10"], "TOP10") and r["call_r2"] == call_of(r["d2"], "R2")
    assert 0.0 <= r["anchor_overlap"]["lean+P5"] <= 1.0 and set(scores) == {nm for nm, _k in PROBES}
    assert set(by_hop(scores, t["y"], t["q"], c.ids)) == {"2hop", "3hop", "4hop"}
    # R2's anchors are the probe's top five: rebuilt here by hand for question 1
    a, b = 60, 120
    top_s = score_top(scores["lean+P5"][a:b])
    V = LTW.unit(c.V[c.node[a:b]])
    Q = LTW.unit(c.q_emb[1].to(torch.float32).numpy())
    R2_raw, _ = anchor_block(c, np.arange(3), nd, lambda i: c.node[i], lambda rr: (score_top(scores["lean+P5"][rr]), TOP))
    assert np.allclose(R2_raw[a:b, :6], cols_k(V, Q, top_s).astype(np.float32), atol=1e-6)
    # 4. the machinery check refuses columns that are not A3's; the reproduction check refuses a drift over 0.002
    orig = PP.read

    def off_read(*x, **k):
        tt = orig(*x, **k)
        tt["D"] = tt["D"] + np.float32(1e-3)
        return tt
    PP.read = off_read
    try:
        read_all(c, 3, nd, lambda i: c.node[i])
        raise AssertionError("columns off A3's must refuse")
    except SystemExit:
        pass
    finally:
        PP.read = orig
    assert PP.read is orig
    got = {"lean": r["probes"]["lean"]["recall@5"], "lean+P5": r["probes"]["lean+P5"]["recall@5"]}
    assert read_all(c, 3, nd, lambda i: c.node[i], dict(got, **{"from": "made"}))[0]["repro"]["pass"]
    try:
        read_all(c, 3, nd, lambda i: c.node[i], {"from": "made", "lean": got["lean"] + 0.003, "lean+P5": got["lean+P5"]})
        raise AssertionError("a reproduction off by more than 0.002 must refuse")
    except SystemExit:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "pair_musique.json").write_text(json.dumps(
            {"lr_lean": {"recall@5": 0.61}, "lr_lean_plus_pair": {"recall@5": 0.62}}), encoding="utf-8")
        assert a3_reference("musique", tmp)["lean"] == 0.61 and a3_reference("hotpotqa", tmp)["lean+P5"] == 0.8223
    log("selftest: the columns over rrf's five are zpfp's exactly and over ten follow the same rule; rrf's anchors are "
        "checked and a score's are ties-by-position; the calls; the read on zpfp's made carve (machinery exact, every "
        "block and probe, R2's anchors the probe's own top five); the machinery and reproduction refusals all pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("anchors")
    p.add_argument("--dataset", required=True, choices=sorted(PF.CARVE))
    p.add_argument("--nq", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.error("a command or --selftest")
    torch.set_num_threads(a.threads)
    if a.host:
        import lean_host as LH
        LH.substitute()                              # the verified mirror in place of the package, as zpfp --host
    return anchors_cmd(a)


if __name__ == "__main__":
    raise SystemExit(main())
