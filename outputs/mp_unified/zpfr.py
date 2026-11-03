"""PFS amendment A5 (docs/PKO_FEATURE_SCREEN.md section 14): section 13's read on musique's 3-hop and 4-hop questions.

Sections 11 and 13 read musique's fit carve on its first 400 questions, which are all 2-hop. Its 3-hop and 4-hop
questions sit in the carve's last cache parts. This reads three of them alone (musique/s1fit parts 20, 25 and 26: 242
3-hop and 271 4-hop questions) with zpfq's read unchanged: pass 1 (zpfp.read), P10, R2, R3, the same probes and the
same calls, and each by hop.

The three parts are loaded alone. Inside the carve's load and the node index, lean_cache.part_dirs returns those parts
with their records' row ranges moved to start at 0; the arrays and their shas are unchanged. It is put back after,
also on a failure. The read refuses parts whose questions are not the 242 and 271 named here. There is no A3 read of
these questions, so no reproduction check; the machinery check (rrf's top five give A3's twelve) still refuses.

zpfq, zpfp, zpfs, ltwdiag, lean_gpu, lean_cache and lean_mlp3 are imported and not edited; PKO's cache and the node
vectors are read only; this writes only outputs/pfs/anchors_musique_h34.json.

    python outputs/mp_unified/zpfr.py h34 --threads 2 --host
    python outputs/mp_unified/zpfr.py --selftest
"""
import argparse
import collections
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

import zpfq as PQ  # noqa: E402  (A4's read, probes and calls; not edited)
import zpfp as PP  # noqa: E402  (A3's carve loading and node index; not edited)
import zpfs as PF  # noqa: E402  (the probe and metrics; not edited)

DECLARED = PF.DECLARED
AMENDMENT = "A5"
DS, CARVE = "musique", "s1fit"
PARTS = (20, 25, 26)
WANT = {"3hop": 242, "4hop": 271}
OUT_NAME = "anchors_musique_h34.json"


def log(m):
    PF.log(m)


@contextlib.contextmanager
def on_parts(ds, carve, sel, cache_root):
    """lean_cache.part_dirs returns only parts sel of ds/carve, their records' row ranges moved to start at 0 in sel's
    order (each record keeps its old range as moved_from); put back after, also on a failure. Yields those parts and
    records."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, Path(cache_root))
    if not sel or len(set(sel)) != len(sel) or min(sel) < 0 or max(sel) >= len(parts):
        raise SystemExit(f"zpfr: parts {sel} are not distinct parts of the carve's {len(parts)}")
    pp, rr, at = [parts[i] for i in sel], [], 0
    for i in sel:
        r = dict(recs[i])
        k = int(r["rows"])
        r["moved_from"] = list(recs[i]["row_range"])
        r["row_range"] = [at, at + k]
        rr.append(r)
        at += k
    orig = LC.part_dirs

    def chosen(d, cv, *a, **k):
        p, r = orig(d, cv, *a, **k)
        return (list(pp), [dict(x) for x in rr]) if (d, cv) == (ds, carve) else (p, r)
    LC.part_dirs = chosen
    try:
        yield pp, rr
    finally:
        LC.part_dirs = orig


def hop_counts(ids):
    return dict(sorted(collections.Counter(str(x)[:4] for x in ids).items()))


def hop_calls(by_hop):
    """Per hop, deciding nothing: d2, d10 and d_pair on that hop's questions, and section 13's calls on them."""
    out = {}
    for h, r5 in by_hop.items():
        d2, d10 = r5["lean+P5+R2"] - r5["lean+P5"], r5["lean+P10"] - r5["lean+P5"]
        out[h] = {"d2": d2, "d10": d10, "d_pair": r5["lean+P5"] - r5["lean"],
                  "call_r2": PQ.call_of(d2, "R2"), "call_top10": PQ.call_of(d10, "TOP10")}
    return out


def assemble(r, scores, t, ids):
    """The read's record beside A4's: the pooled calls (section 14 reads these), the pair call on d_pair, by hop."""
    bh = PQ.by_hop(scores, t["y"], t["q"], ids)
    return {**r, "call_pair": PP.call_of(r["d_pair"]), "by_hop": bh, "by_hop_calls": hop_calls(bh)}


def h34_cmd(a):
    import lean_cache as LC
    import lean_mlp3 as L3
    t0 = time.time()
    with on_parts(DS, CARVE, PARTS, PF.CACHE) as (parts, recs):
        c = PP.open_carve(DS, CARVE)
    nq = c.rows
    hops = hop_counts(c.ids)
    if hops != WANT:
        raise SystemExit(f"zpfr: parts {PARTS} hold {hops}, not {WANT}")
    nodes, freeze = L3.open_nodes([DS])
    nid = PP.node_index(parts, recs, int(c.off_np[nq]))
    r, scores, t = PQ.read_all(c, nq, nodes[DS], lambda i: nid[i], None)
    if not r["identity"]["pass"]:
        raise SystemExit(f"zpfr: identity FAIL {r['identity']}")
    out = {"declared_in": DECLARED, "amendment": AMENDMENT, "dataset": DS, "carve": c.carve, "parts": list(PARTS),
           "moved_from": [x["moved_from"] for x in recs], "questions": nq, "hops": hops, "part_shas": c.part_shas,
           "nodes_record_sha256": freeze, **assemble(r, scores, t, c.ids)}
    out["host"] = bool(a.host)
    out["script_sha256"] = LC.sha_file(Path(__file__))
    out["zpfq_sha256"] = LC.sha_file(HERE / "zpfq.py")
    out["zpfp_sha256"] = LC.sha_file(HERE / "zpfp.py")
    out["seconds"] = time.time() - t0
    p = Path(PF.OUT) / OUT_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"h34 musique: {nq} questions {hops}, {out['rows']} rows, machinery {r['machinery']['max_abs']:.1e} "
        f"({out['seconds']:.0f}s) -> {p}")
    for nm, _k in PQ.PROBES:
        m = out["probes"][nm]
        log(f"   {nm:12s} auc {m['auc']:.4f} recall@5 {m['recall@5']:.4f} recall@20 {m['recall@20']:.4f} hit@5 {m['hit@5']:.3f}")
    log(f"   d_pair {out['d_pair']:+.4f} {out['call_pair']}; d10 {out['d10']:+.4f} {out['call_top10']}; d2 {out['d2']:+.4f} "
        f"{out['call_r2']}; beside d3 {out['d3']:+.4f}, d5_10 {out['d5_10']:+.4f}")
    for h, hc in out["by_hop_calls"].items():
        log(f"   {h}: d_pair {hc['d_pair']:+.4f}, d10 {hc['d10']:+.4f}, d2 {hc['d2']:+.4f}")
    return 0


# ── self-test (CPU, no cache) ────────────────────────────────────────────────


def selftest():
    import tempfile
    import lean_cache as LC
    # 1. the chosen parts: moved to start at 0 in order, each keeping its old range; put back after, also on a failure
    recs = [{"queries": 10, "rows": 100, "row_range": [0, 100]}, {"queries": 10, "rows": 120, "row_range": [100, 220]},
            {"queries": 5, "rows": 50, "row_range": [220, 270]}, {"queries": 8, "rows": 70, "row_range": [270, 340]}]
    orig = LC.part_dirs
    LC.part_dirs = lambda d, cv, root=None: ([Path("a"), Path("b"), Path("c"), Path("d")], [dict(x) for x in recs])
    try:
        with on_parts(DS, CARVE, (1, 3), "x") as (pp, rr):
            assert [p.name for p in pp] == ["b", "d"] and [x["row_range"] for x in rr] == [[0, 120], [120, 190]]
            assert [x["moved_from"] for x in rr] == [[100, 220], [270, 340]]
            p2, r2 = LC.part_dirs(DS, CARVE, Path("x"))
            assert [p.name for p in p2] == ["b", "d"] and [x["row_range"] for x in r2] == [[0, 120], [120, 190]]
            assert [p.name for p in LC.part_dirs("2wiki", "fit", Path("x"))[0]] == ["a", "b", "c", "d"]
        assert len(LC.part_dirs(DS, CARVE, Path("x"))[0]) == 4
        for bad in ((), (1, 1), (4,), (-1,)):
            try:
                with on_parts(DS, CARVE, bad, "x"):
                    pass
                raise AssertionError(f"parts {bad} must refuse")
            except SystemExit:
                pass
        try:
            with on_parts(DS, CARVE, (2,), "x"):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
        assert len(LC.part_dirs(DS, CARVE, Path("x"))[0]) == 4
    finally:
        LC.part_dirs = orig
    assert LC.part_dirs is orig
    # 2. zpfp's node index reads the moved records in their new order (sha-checked)
    with tempfile.TemporaryDirectory() as tmp:
        ps, moved, at = [], [], 0
        for j, r in enumerate(recs):
            p = Path(tmp) / f"part_{j}"
            p.mkdir()
            np.save(p / "node.npy", np.arange(r["row_range"][0], r["row_range"][1], dtype=np.int64) * 3)
            r["arrays"] = {"node": {"sha256": LC.sha_file(p / "node.npy")}}
            ps.append(p)
        for j in (1, 3):
            moved.append(dict(recs[j], row_range=[at, at + recs[j]["rows"]]))
            at += recs[j]["rows"]
        got = PP.node_index([ps[1], ps[3]], moved, 190)
        assert np.array_equal(got, np.r_[np.arange(100, 220), np.arange(270, 340)] * 3)
    # 3. the hop counts, and the record on zpfp's made carve through zpfq's read (its ids carry 2, 3 and 4 hops)
    assert hop_counts(["3hop1__a", "4hop2__b", "3hop2__c"]) == {"3hop": 2, "4hop": 1}
    c = PP.FakeCarve()
    nd = PP.FakeNodes(c.V)
    r, scores, t = PQ.read_all(c, 3, nd, lambda i: c.node[i], None)
    o = assemble(r, scores, t, c.ids)
    assert o["call_pair"] == PP.call_of(o["d_pair"]) and set(o["by_hop"]) == {"2hop", "3hop", "4hop"}
    for h, hc in o["by_hop_calls"].items():
        r5 = o["by_hop"][h]
        assert abs(hc["d2"] - (r5["lean+P5+R2"] - r5["lean+P5"])) < 1e-12 and hc["call_r2"] == PQ.call_of(hc["d2"], "R2")
        assert abs(hc["d_pair"] - (r5["lean+P5"] - r5["lean"])) < 1e-12
    assert sum(WANT.values()) == 513 and PARTS == (20, 25, 26)
    log("selftest: the chosen parts (moved to start at 0, old ranges kept, put back after, also on a failure, bad "
        "choices refused), zpfp's node index on the moved records, the hop counts, and the record through zpfq's read "
        "with the pair call and the calls by hop all pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("h34")
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
    return h34_cmd(a)


if __name__ == "__main__":
    raise SystemExit(main())
