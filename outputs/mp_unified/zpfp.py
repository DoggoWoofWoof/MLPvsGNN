"""PFS amendment A3 (docs/PKO_FEATURE_SCREEN.md section 11): the question-conditional pair scores against the first
stage's top five rows, over every pool row, from the frozen encoder's full vectors, read with amendment A2's probe (the
172 lean columns, then those plus the new ones) on A2's questions.

For a row c of question q and each of rrf's top five rows a_k (unit vectors of the frozen encoder; a top row is not paired
with itself), ltwdiag's own free scores (pair_feats, the ones LTA's A cases rank an unlinked target by):

    res_k  = c.q - (a_k.q)(a_k.c)          q's part along c once a_k's direction is taken out
    pair_k = (a_k.q + c.q) / |a_k + c|     q's cosine to the pair's normalised sum

The six columns are pair_max, res_max, pair_mean and res_mean over the anchors, and pair_1, res_1 against rrf's first row
(its second row for the first row itself), each raw and as a within-question z-score. Each row's c.q is checked against
the cache's dense_cos (the identity), so the rows, the node vectors and the question vectors are the cache's own.

Only the cache parts that hold the first --nq questions are loaded (lean_cache.part_dirs patched to that prefix inside
the carve's load, put back after), without the pool's links (the probe reads none). zpfs, ltwdiag, lean_gpu, lean_cache
and lean_mlp3 are imported and not edited; PKO's cache and the node vectors are read only; this writes only
outputs/pfs/pair_<dataset>.json.

    python outputs/mp_unified/zpfp.py pair --dataset musique --host
    python outputs/mp_unified/zpfp.py --selftest
"""
import argparse
import contextlib
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

import zpfs as PF  # noqa: E402  (A2's carve settings, probe, metrics and questions; not edited)
import ltwdiag as LTW  # noqa: E402  (the free scores; not edited)

CH = LTW.CH
DECLARED = PF.DECLARED
AMENDMENT = "A3"
TOP = 5
NAMES = ["pair_max", "res_max", "pair_mean", "res_mean", "pair_1", "res_1"]
ADDS, NOTHING = 0.01, 0.005                       # section 11's rule on the recall@5 difference
BATCH = 16
HOP_DATASET = "musique"                           # the one dataset whose ids carry the hop


def log(m):
    PF.log(m)


def call_of(d):
    """Section 11's call on d = recall@5 with the six columns minus recall@5 of the lean probe."""
    if d >= ADDS:
        return "PAIR_ADDS"
    if d <= -ADDS:
        return "PAIR_HURTS"
    if abs(d) < NOTHING:
        return "PAIR_NOTHING"
    return "MIXED"


def pair_cols(X, qv, top):
    """X (n, d) unit rows of one question, qv (d,) its unit vector, top (5,) its rows of rrf's top five in rrf order:
    (n, 6) columns in NAMES order and each row's cosine to the question."""
    n = X.shape[0]
    P = np.empty((n, TOP), np.float64)
    R = np.empty((n, TOP), np.float64)
    cv = None
    for k in range(TOP):
        cv, _s, _ca, res, pair = LTW.pair_feats(qv, X[top[k]], X)
        P[:, k], R[:, k] = pair, res
    P[top, np.arange(TOP)] = np.nan                  # a top row is not paired with itself
    R[top, np.arange(TOP)] = np.nan
    k1 = np.zeros(n, np.int64)
    k1[top[0]] = 1                                   # rrf's first row reads its second
    ar = np.arange(n)
    return np.stack([np.nanmax(P, 1), np.nanmax(R, 1), np.nanmean(P, 1), np.nanmean(R, 1), P[ar, k1], R[ar, k1]], 1), cv


def pair_batch(c, qs, nd, node_of, blocks=PF.FULL_BLOCKS):
    """For questions qs, every pool row: the lean pick's columns (raw, then z-scored over the pool) as A2's probe reads
    them, the six pair columns raw then z-scored, the labels, rrf's rank, and the identity counts."""
    import lean_gpu as LG
    import zlink as ZL
    qs = np.asarray(qs, np.int64)
    feats, nq, base_z, gold = c.batch(qs, blocks)
    B, N = qs.size, nq.numel()
    cnt = c.n_np[qs]
    seg = np.cumsum(cnt) - cnt
    idx = np.repeat(c.off_np[qs] - seg, cnt) + np.arange(N)
    rank = ZL.pool_rank(base_z, nq, B).numpy()
    uq, inv = np.unique(node_of(idx), return_inverse=True)
    V = LTW.unit(np.asarray(nd.read(uq), np.float32))
    Q = LTW.unit(c.q_emb[torch.from_numpy(qs)].to(torch.float32).numpy())
    D = np.empty((N, len(NAMES)), np.float64)
    cv = np.empty(N, np.float64)
    for b in range(B):
        s, e = int(seg[b]), int(seg[b] + cnt[b])
        if e - s <= TOP:
            raise SystemExit(f"zpfp: question {int(qs[b])} has {e - s} rows, not more than {TOP}")
        r = rank[s:e]
        top = np.argsort(r, kind="stable")[:TOP]
        if not np.array_equal(r[top], np.arange(TOP)):
            raise SystemExit(f"zpfp: question {int(qs[b])}: rrf's ranks do not start 0..{TOP - 1}")
        D[s:e], cv[s:e] = pair_cols(V[inv[s:e]], Q[b], top)
    dif = np.abs(cv - feats["dense_cos"][:, 0].numpy().astype(np.float64))
    raw = torch.cat([feats[b] for b in blocks], 1)
    XN = torch.cat([raw, LG.seg_zscore8D(raw, nq, B)], 1)
    Dt = torch.from_numpy(D.astype(np.float32))
    DN = torch.cat([Dt, LG.seg_zscore8D(Dt, nq, B)], 1)
    return {"XN": XN.numpy(), "D": DN.numpy(), "y": gold.bool().numpy(), "q": qs[nq.numpy()], "rank": rank,
            "chk": (int(N), int((dif > CH.TOL_COS).sum()), float(dif.max()))}


def parts_for(recs, nq):
    """How many leading cache parts hold the first nq questions."""
    at = 0
    for m, r in enumerate(recs, 1):
        at += int(r["queries"])
        if at >= nq:
            return m
    raise SystemExit(f"zpfp: the carve holds {at} questions, fewer than {nq}")


@contextlib.contextmanager
def on_prefix(ds, carve, nq, cache_root):
    """lean_cache.part_dirs returns only the leading parts that hold the first nq questions of ds/carve inside the block
    (the carve then loads those parts only); put back after, also on a failure. Yields those parts and records."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, Path(cache_root))
    m = parts_for(recs, nq)
    orig = LC.part_dirs

    def prefix(d, cv, *a, **k):
        p, r = orig(d, cv, *a, **k)
        return (p[:m], r[:m]) if (d, cv) == (ds, carve) else (p, r)
    LC.part_dirs = prefix
    try:
        yield parts[:m], recs[:m]
    finally:
        LC.part_dirs = orig


def open_carve(ds, carve):
    """zpfs's carve (same basis, cache and checks) without the pool's links, on the CPU."""
    import lean_gpu as LG
    LG.bind_device_ops()
    return LG.CacheCarve(ds, carve, PF.BASIS, "cpu", Path(PF.CACHE), True)


def node_index(parts, recs, rows, verify=True):
    """Each carve row's node id for rows [0, rows), from the parts' node arrays in the carve's order (read only, each
    array checked against its record's sha256)."""
    import lean_cache as LC
    got, at = [], 0
    for p, r in zip(parts, recs):
        if at >= rows:
            break
        if int(r["row_range"][0]) != at:
            raise SystemExit(f"zpfp: {p}: its row range {r['row_range']} does not follow {at}")
        a = LC.load_array(p, "node", r, verify)
        if a.shape[0] != int(r["rows"]):
            raise SystemExit(f"zpfp: {p}: node.npy has {a.shape[0]} rows, its record {r['rows']}")
        k = min(int(r["rows"]), rows - at)
        got.append(np.asarray(a[:k], np.int64))
        at += k
    if at != rows:
        raise SystemExit(f"zpfp: the parts hold {at} rows, the questions {rows}")
    return np.concatenate(got)


def read(c, nq, nd, node_of):
    """The probe's tables over the first nq questions, BATCH at a time."""
    parts = [pair_batch(c, np.arange(k, min(k + BATCH, nq)), nd, node_of) for k in range(0, nq, BATCH)]
    t = {k: np.concatenate([p[k] for p in parts]) for k in ("XN", "D", "y", "q", "rank")}
    n, bad, worst = sum(p["chk"][0] for p in parts), sum(p["chk"][1] for p in parts), max(p["chk"][2] for p in parts)
    t["identity"] = {"checked": n, "bad": bad, "max_abs": worst, "pass": n > 0 and bad <= CH.CHECK_BAD * n}
    return t


def report(t, ids=None):
    """Each column's AUC (z-scored) over every row and beyond rrf's top five; the two probes; the call; by hop when the
    question ids are given (musique's carry the hop)."""
    y, q, rank = t["y"], t["q"], t["rank"]
    far = rank >= TOP
    out = {"rows": int(y.size), "gold_rows": int(y.sum()), "golds_beyond_top5": int(y[far].sum()), "columns": {}}
    for j, nm in enumerate(NAMES):
        v = t["D"][:, len(NAMES) + j]
        out["columns"][nm] = {"auc_z": PF.auc(v, y), "auc_z_beyond_top5": PF.auc(v[far], y[far])}
    s0, _w0 = PF.lr_crossfit(t["XN"], y, q)
    s1, w1 = PF.lr_crossfit(np.concatenate([t["XN"], t["D"]], 1), y, q)
    for nm, s in (("lr_lean", s0), ("lr_lean_plus_pair", s1)):
        out[nm] = {"auc": PF.auc(s, y), **PF.hit_at(s, y, q),
                   **{k: v for k, v in PF.recall_at(s, y, q).items() if k != "questions"}}
    d = out["lr_lean_plus_pair"]["recall@5"] - out["lr_lean"]["recall@5"]
    out["d_recall@5"] = d
    out["d_auc"] = out["lr_lean_plus_pair"]["auc"] - out["lr_lean"]["auc"]
    out["call"] = call_of(d)
    out["pair_weights"] = dict(zip(NAMES + [n + "_z" for n in NAMES], [round(float(w), 3) for w in w1[-2 * len(NAMES):]]))
    if ids is not None:
        hop = np.asarray([int(str(x)[0]) if str(x)[:1] in "234" else 0 for x in ids], np.int64)[q]
        out["by_hop"] = {}
        for h in (2, 3, 4):
            m = hop == h
            if m.any():
                out["by_hop"][f"{h}hop"] = {nm: PF.recall_at(s[m], y[m], q[m])["recall@5"]
                                            for nm, s in (("lr_lean", s0), ("lr_lean_plus_pair", s1))}
    return out


def pair_cmd(a):
    import lean_cache as LC
    import lean_mlp3 as L3
    t0 = time.time()
    carve = PF.CARVE[a.dataset]
    nq_want = a.nq or PF.NQ[a.dataset]
    with on_prefix(a.dataset, carve, nq_want, PF.CACHE) as (parts, recs):
        c = open_carve(a.dataset, carve)
    nq = min(nq_want, c.rows)
    nodes, freeze = L3.open_nodes([a.dataset])
    nid = node_index(parts, recs, int(c.off_np[nq]))
    t = read(c, nq, nodes[a.dataset], lambda i: nid[i])
    if not t["identity"]["pass"]:
        raise SystemExit(f"zpfp {a.dataset}: identity FAIL {t['identity']}")
    out = {"declared_in": DECLARED, "amendment": AMENDMENT, "dataset": a.dataset, "carve": c.carve, "questions": nq,
           "parts_loaded": len(parts), "part_shas": c.part_shas, "nodes_record_sha256": freeze, "identity": t["identity"],
           **report(t, c.ids[:nq] if a.dataset == HOP_DATASET else None)}
    a2 = Path(PF.OUT) / f"prf_{a.dataset}.json"
    if a2.is_file():
        r2 = json.loads(a2.read_text(encoding="utf-8"))
        out["a2_beside"] = {"questions": r2.get("questions"), "rows": r2.get("rows"),
                            "lr_lean_recall@5": (r2.get("lr_lean") or {}).get("recall@5"),
                            "lr_lean_auc": (r2.get("lr_lean") or {}).get("auc")}
    out["host"] = bool(a.host)
    out["script_sha256"] = LC.sha_file(Path(__file__))
    out["seconds"] = time.time() - t0
    p = Path(PF.OUT) / f"pair_{a.dataset}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"pair {a.dataset}: {nq} questions, {out['rows']} rows, golds beyond top 5 {out['golds_beyond_top5']}, identity "
        f"{t['identity']['pass']} ({out['seconds']:.0f}s) -> {p}")
    for nm in NAMES:
        cc = out["columns"][nm]
        log(f"   {nm}: auc {cc['auc_z']:.3f} beyond top 5 {cc['auc_z_beyond_top5']:.3f}")
    for nm in ("lr_lean", "lr_lean_plus_pair"):
        r = out[nm]
        log(f"   {nm}: auc {r['auc']:.4f} recall@5 {r['recall@5']:.4f} recall@20 {r['recall@20']:.4f} hit@5 {r['hit@5']:.3f}")
    log(f"   d recall@5 {out['d_recall@5']:+.4f}, d auc {out['d_auc']:+.4f}: {out['call']}"
        + (f" (A2's lean recall@5 {out['a2_beside']['lr_lean_recall@5']})" if "a2_beside" in out else ""))
    return 0


# ── self-test (CPU, no cache) ────────────────────────────────────────────────


class FakeNodes:
    def __init__(self, V):
        self.V = V

    def read(self, ids):
        ids = np.asarray(ids, np.int64)
        assert np.all(np.diff(ids) > 0), "nodes are read sorted and unique"
        return self.V[ids]


class FakeCarve(PF.FakeCarve):
    """zpfs's three-question carve, with question vectors, node ids and full node vectors made so the missing gold
    (row 30) completes the question with the first stage's top row (row 0) while it is far from both, and dense_cos the
    rows' true cosine to the question (rolled by shift to put the rows out of line)."""

    D = 24

    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(11)
        self.node = np.arange(180, dtype=np.int64)[::-1].copy()             # row r holds node 179 - r
        V = LTW.unit(rng.standard_normal((180, self.D)).astype(np.float32))
        qv = []
        for q in range(3):
            o = q * 60
            a, g = V[self.node[o]], rng.standard_normal(self.D).astype(np.float32)
            g = LTW.unit(g - (g @ a) * a)                                   # the missing gold: orthogonal to the top row
            V[self.node[o + 30]] = g
            qv.append(LTW.unit(a + g + 0.05 * rng.standard_normal(self.D).astype(np.float32)))
        self.V = V
        self.q_emb = torch.from_numpy(np.stack(qv).astype(np.float16))
        self.ids = ["2hop__a", "3hop1__b", "4hop2__c"]
        self.true_cos = np.concatenate([LTW.unit(V[self.node[q * 60:(q + 1) * 60]]) @ LTW.unit(
            self.q_emb[q].to(torch.float32).numpy()) for q in range(3)])
        self.shift = 0

    def batch(self, qs, blocks):
        feats, nq, bz, gold = super().batch(qs, blocks)
        idx = np.concatenate([np.arange(self.off_np[q], self.off_np[q + 1]) for q in np.asarray(qs, np.int64)])
        feats["dense_cos"] = torch.from_numpy(np.roll(self.true_cos, self.shift)[idx].astype(np.float32))[:, None]
        return feats, nq, bz, gold


def selftest():
    import tempfile
    import lean_cache as LC
    rng = np.random.default_rng(0)
    # 1. the columns are ltwdiag's free scores, anchor by anchor, without a top row's pair with itself
    X = LTW.unit(rng.standard_normal((40, 16)).astype(np.float32))
    qv = LTW.unit(rng.standard_normal(16).astype(np.float32))
    top = np.array([7, 3, 11, 0, 25])
    C, cv = pair_cols(X, qv, top)
    assert C.shape == (40, 6) and np.allclose(cv, (X @ qv).astype(np.float64))
    want_p = np.stack([LTW.pair_feats(qv, X[t], X)[4] for t in top], 1)
    want_r = np.stack([LTW.pair_feats(qv, X[t], X)[3] for t in top], 1)
    for i in range(40):
        keep = np.array([top[k] != i for k in range(TOP)])
        assert abs(C[i, 0] - want_p[i, keep].max()) < 1e-12 and abs(C[i, 1] - want_r[i, keep].max()) < 1e-12
        assert abs(C[i, 2] - want_p[i, keep].mean()) < 1e-12 and abs(C[i, 3] - want_r[i, keep].mean()) < 1e-12
        k1 = 1 if i == top[0] else 0
        assert C[i, 4] == want_p[i, k1] and C[i, 5] == want_r[i, k1]
    assert not np.isnan(C).any()
    # 2. the call
    assert [call_of(x) for x in (0.012, 0.01, 0.006, 0.004, -0.004, -0.006, -0.011)] == [
        "PAIR_ADDS", "PAIR_ADDS", "MIXED", "PAIR_NOTHING", "PAIR_NOTHING", "MIXED", "PAIR_HURTS"]
    # 3. the prefix of parts (put back after, also on a failure) and the node index (sha-checked, in row order)
    recs = [{"queries": 10, "rows": 100, "row_range": [0, 100]}, {"queries": 10, "rows": 120, "row_range": [100, 220]},
            {"queries": 5, "rows": 50, "row_range": [220, 270]}]
    assert [parts_for(recs, n) for n in (1, 10, 11, 20, 25)] == [1, 1, 2, 2, 3]
    try:
        parts_for(recs, 26)
        raise AssertionError("more questions than the carve holds must refuse")
    except SystemExit:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        ps = []
        for j, r in enumerate(recs):
            p = Path(tmp) / f"part_{j}"
            p.mkdir()
            np.save(p / "node.npy", np.arange(r["row_range"][0], r["row_range"][1], dtype=np.int64) * 3)
            r["arrays"] = {"node": {"sha256": LC.sha_file(p / "node.npy")}}
            ps.append(p)
        assert np.array_equal(node_index(ps, recs, 150), np.arange(150) * 3)
        assert np.array_equal(node_index(ps, recs, 270), np.arange(270) * 3)
        for bad in ([dict(recs[0]), dict(recs[1], row_range=[90, 210])],
                    [dict(recs[0], arrays={"node": {"sha256": "0" * 64}})]):
            try:
                node_index(ps[:len(bad)], bad, 150 if len(bad) == 2 else 50)
                raise AssertionError("a gap in the row ranges, or a node array off its sha, must refuse")
            except SystemExit:
                pass
    orig = LC.part_dirs
    LC.part_dirs = lambda d, cv, root=None: ([Path("a"), Path("b"), Path("c")], recs)
    try:
        with on_prefix("musique", "s1fit", 11, "x") as (pp, rr):
            assert [p.name for p in pp] == ["a", "b"] and len(rr) == 2
            assert [p.name for p in LC.part_dirs("musique", "s1fit", Path("x"))[0]] == ["a", "b"]
            assert [p.name for p in LC.part_dirs("2wiki", "fit", Path("x"))[0]] == ["a", "b", "c"]
        assert len(LC.part_dirs("musique", "s1fit", Path("x"))[0]) == 3
        try:
            with on_prefix("musique", "s1fit", 11, "x"):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
        assert len(LC.part_dirs("musique", "s1fit", Path("x"))[0]) == 3
    finally:
        LC.part_dirs = orig
    assert LC.part_dirs is orig
    # 4. the read on a three-question carve: the identity, the missing gold's pair column, the probe and its call
    c = FakeCarve()
    nd = FakeNodes(c.V)
    t = read(c, 3, nd, lambda i: c.node[i])
    assert t["identity"]["pass"] and t["identity"]["checked"] == 180 and t["identity"]["max_abs"] < 1e-3, t["identity"]
    assert t["XN"].shape == (180, 52) and t["D"].shape == (180, 12) and int(t["y"].sum()) == 6
    pm = t["D"][:, 0]
    for q in range(3):
        o = q * 60
        far = np.arange(o + 5, o + 60)
        assert int(np.argmax(pm[far])) + o + 5 == o + 30, (q, np.argsort(-pm[far])[:3] + o + 5)
    c.shift = 7
    assert not read(c, 3, nd, lambda i: c.node[i])["identity"]["pass"], "rows out of line with dense_cos must fail"
    c.shift = 0
    r = report(t, c.ids)
    assert r["golds_beyond_top5"] == 3 and r["columns"]["pair_max"]["auc_z_beyond_top5"] > 0.9, r["columns"]["pair_max"]
    assert set(r) >= {"lr_lean", "lr_lean_plus_pair", "d_recall@5", "call", "pair_weights", "by_hop"}
    assert set(r["by_hop"]) == {"2hop", "3hop", "4hop"} and r["call"] == call_of(r["d_recall@5"])
    assert "by_hop" not in report(t, None) and len(r["pair_weights"]) == 12
    log("selftest: the six columns are ltwdiag's free scores anchor by anchor (a top row not paired with itself, rrf's "
        "first row reading its second), the call, the part prefix (put back after, also on a failure), the node index "
        "(row order, sha-checked), the identity (passes in line, fails out of line), the missing gold first by pair_max "
        "beyond the top five, and the report all pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("pair")
    p.add_argument("--dataset", required=True, choices=sorted(PF.CARVE))
    p.add_argument("--nq", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.error("a command or --selftest")
    torch.set_num_threads(a.threads)
    if a.host:
        import lean_host as LH
        LH.substitute()                              # the verified mirror in place of the package, as ltacols --host
    return pair_cmd(a)


if __name__ == "__main__":
    raise SystemExit(main())
