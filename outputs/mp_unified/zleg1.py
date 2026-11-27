"""LEG1 part A (docs/LEG1_LEG_NECESSITY.md section 2): the two legs of the first stage alone against their fusion, at equal seed budget, on the record's lists.

    python -E -P -B outputs/mp_unified/zleg1.py selftest
    python -E -P -B outputs/mp_unified/zleg1.py legs --ds squad|musique|multihoprag|...   # reads the record's two top-1000 lists; writes outputs/leg1/legs_<ds>.json
    python -E -P -B outputs/mp_unified/zleg1.py verdict                                    # the call per leg over the datasets read -> outputs/leg1/partA.json

Run from the workspace root. zspl1.py (its context, boot, hits_in) and zenc.py (rrf_top, hits) are imported and edited nowhere. No encode, no card, no label beyond the golds the
record's reads already use.
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

DOC = "docs/LEG1_LEG_NECESSITY.md section 2"
OUT = S.WS / "outputs" / "leg1"
NEED_PTS = 1.0                      # NEEDED if the lower end of the loss is at least this; DISPENSABLE if the upper end is below it
DEPTHS = (5, 20, 100, 1000)
log = S.log


def stop(msg):
    raise SystemExit(f"[zleg1] STOP: {msg}")


def cover(h, gt):
    """The set coverage of one read, as zspl1.reads_of: distinct golds found over golds."""
    return np.minimum(h, gt) / gt.astype(np.float64)


def seed_sets(dense, spl, m=5):
    """The record's seeds (dense top m with SPLADE top m, distinct nodes) and each question's budget u."""
    uni = [np.unique(np.concatenate([dense[q, :m], spl[q, :m]])) for q in range(dense.shape[0])]
    return uni, np.array([u.size for u in uni], np.int64)


def single_leg_hits(lst, u, flat, off):
    """Golds of each question inside the leg's top u[q] (the leg's own first u nodes: equal budget to the union)."""
    h = np.zeros(lst.shape[0], np.int64)
    for q in range(lst.shape[0]):
        g = flat[off[q]:off[q + 1]]
        if g.size:
            h[q] = np.isin(g, lst[q, :u[q]]).sum()
    return h


def loss_call(lo, hi):
    return "NEEDED" if lo >= NEED_PTS else ("DISPENSABLE" if hi < NEED_PTS else "UNCLEAR")


def loss_of(union, other):
    """What is lost by dropping a leg: the union's coverage minus the other leg's alone (paired per question); mean, 95% interval, call."""
    dm, lo, hi = S.boot(union - other)
    return {"loss": round(dm, 3), "lo": round(lo, 3), "hi": round(hi, 3), "call": loss_call(lo, hi)}


def reads(dense, spl, flat, off, gt, n, fuse):
    """The three lists' coverage at the declared depths and at the seeds' equal budget. fuse(dense, spl, k) = the record's fusion's top k."""
    out = {"seeds": {}, "depth": {}}
    uni, u = seed_sets(dense, spl)
    out["seed_budget_mean"] = round(float(u.mean()), 4)
    cu = cover(S.hits_in(uni, flat, off), gt)
    cd = cover(single_leg_hits(dense, u, flat, off), gt)
    cs = cover(single_leg_hits(spl, u, flat, off), gt)
    out["seeds"] = {"union": round(float(cu.mean() * 100), 3), "dense_alone": round(float(cd.mean() * 100), 3), "splade_alone": round(float(cs.mean() * 100), 3),
                    "dropping_splade": loss_of(cu, cd), "dropping_dense": loss_of(cu, cs)}
    kmax = min(1000, dense.shape[1], spl.shape[1])
    fused = fuse(dense, spl, kmax)
    for d in DEPTHS:
        if d > kmax or (d == 1000 and n < 1000):
            continue
        cf = cover(_hits(fused[:, :d], flat, off), gt)
        cdd = cover(_hits(dense[:, :d], flat, off), gt)
        css = cover(_hits(spl[:, :d], flat, off), gt)
        out["depth"][str(d)] = {"fusion": round(float(cf.mean() * 100), 3), "dense": round(float(cdd.mean() * 100), 3), "splade": round(float(css.mean() * 100), 3),
                                "dropping_splade": loss_of(cf, cdd), "dropping_dense": loss_of(cf, css)}
    return out


def _hits(top, flat, off):
    h = np.zeros(top.shape[0], np.int64)
    for q in range(top.shape[0]):
        g = flat[off[q]:off[q + 1]]
        if g.size:
            h[q] = np.isin(g, top[q]).sum()
    return h


def cmd_legs(a):
    import zenc as E
    t0 = time.time()
    c = S.context(E, a.ds)
    n = c["w"].n
    dense, spl = c["dense"], c["c0"]
    if dense.shape[0] != spl.shape[0]:
        stop(f"{a.ds}: the two lists have different numbers of questions")
    log(f"{a.ds}: {dense.shape[0]} questions, {n} nodes, lists {dense.shape[1]} and {spl.shape[1]} deep")
    base, _ = S.reads_of(E, dense, spl, c["flat"], c["off"], c["gt"], n)
    gate = {"rrf5_now": round(float(base["rrf5"].mean() * 100), 3)}
    if c["e0_rrf5"] is not None:
        gate["e0_filed_rrf5"] = c["e0_rrf5"]
        gate["pass"] = abs(float(base["rrf5"].mean()) - c["e0_rrf5"]) <= 5e-6
    else:
        L = np.load(S.WS / "outputs" / "mhr1" / "first" / "lists.npz", allow_pickle=False)
        r5 = E.rrf_top(dense, spl, 5)
        share = float(np.mean(np.all(L["e3v_splade"][:, :5].astype(np.int64) == r5, axis=1)))
        gate["top5_equals_mhr1_lists_share"] = round(share, 6)
        gate["pass"] = share >= 0.999
    log(f"{a.ds}: G-S1 {json.dumps(gate)}")
    if not gate["pass"]:
        stop(f"G-S1 failed on {a.ds}: the reader does not reproduce the filed first stage; nothing is read")
    res = {"stage": "LEG1 part A", "declared_in": DOC, "ds": a.ds, "questions": int(dense.shape[0]), "nodes": int(n), "gate_gs1": gate}
    res.update(reads(dense, spl, c["flat"], c["off"], c["gt"], n, lambda d, s, k: E.rrf_top(d, s, k)))
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    S.wjson(OUT / f"legs_{a.ds}.json", res)
    sd = res["seeds"]
    log(f"{a.ds} seeds: union {sd['union']}  dense alone {sd['dense_alone']}  SPLADE alone {sd['splade_alone']}  | dropping SPLADE {sd['dropping_splade']}  dropping dense {sd['dropping_dense']}")
    print(json.dumps(res, indent=1))


def cmd_verdict(a):
    files = sorted(OUT.glob("legs_*.json"))
    if not files:
        stop("no legs_<ds>.json: run legs first")
    per, calls = {}, {"dense": [], "splade": []}
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        sd = r["seeds"]
        per[r["ds"]] = {"seeds": sd, "R@20": r["depth"].get("20")}
        calls["splade"].append(sd["dropping_splade"]["call"])      # losing SPLADE = SPLADE's worth
        calls["dense"].append(sd["dropping_dense"]["call"])

    def verdict(cs):
        return "NEEDED" if "NEEDED" in cs else ("DISPENSABLE" if all(c == "DISPENSABLE" for c in cs) else "UNCLEAR")

    res = {"stage": "LEG1 part A verdict", "declared_in": DOC, "datasets": sorted(per), "calls": calls, "SPLADE": verdict(calls["splade"]), "dense": verdict(calls["dense"]), "per_dataset": per}
    S.wjson(OUT / "partA.json", res)
    log(f"datasets {sorted(per)}: SPLADE {res['SPLADE']} {calls['splade']}; dense {res['dense']} {calls['dense']}")
    print(json.dumps({k: res[k] for k in ("datasets", "calls", "SPLADE", "dense")}, indent=1))


def selftest():
    bad = 0

    def chk(name, ok):
        nonlocal bad
        log(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += 0 if ok else 1

    # toy: 4 questions, 12 nodes, lists 12 deep; golds: q0 {1,2}; q1 {3}; q2 {4,5,6}; q3 {7}
    dense = np.array([[1, 9, 2, 8, 0, 3, 4, 5, 6, 7, 10, 11], [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 10, 11], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], np.int64)
    spl = np.array([[2, 8, 9, 1, 0, 3, 4, 5, 6, 7, 10, 11], [0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11], [4, 5, 6, 9, 8, 7, 3, 2, 1, 0, 10, 11], [7, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11]], np.int64)
    flat = np.array([1, 2, 3, 4, 5, 6, 7], np.int64)
    off = np.array([0, 2, 3, 6, 7], np.int64)
    gt = np.diff(off)
    uni, u = seed_sets(dense, spl, m=2)
    chk("the seeds are the distinct union and the budget is its size", all(sorted(uni[q].tolist()) == sorted(set(dense[q, :2].tolist()) | set(spl[q, :2].tolist())) for q in range(4)) and u.tolist() == [len(x) for x in uni])
    h = single_leg_hits(dense, u, flat, off)
    chk("a single leg gets the question's own budget: q3's dense top u has no gold 7", int(h[3]) == 0 and u[3] == len(set(dense[3, :2].tolist()) | set(spl[3, :2].tolist())))
    chk("the union's golds on the toy are 2, 1, 2, 1", S.hits_in(uni, flat, off).tolist() == [2, 1, 2, 1])
    cu = cover(S.hits_in(uni, flat, off), gt)
    cd = cover(single_leg_hits(dense, u, flat, off), gt)
    cs = cover(single_leg_hits(spl, u, flat, off), gt)
    chk("at equal budget a single leg can beat the union (q2: SPLADE's own top 4 holds all three golds, the union two)", float(cs[2]) == 1.0 and float(cu[2]) < 1.0)
    chk("coverage is distinct golds over golds, at most 1", float(cu.max()) <= 1.0 and float(cu.min()) >= 0.0)
    chk("the call: lower end 1.0 -> NEEDED; upper end 0.99 -> DISPENSABLE; straddling -> UNCLEAR", (loss_call(1.0, 3.0), loss_call(-2.0, 0.99), loss_call(0.2, 1.5)) == ("NEEDED", "DISPENSABLE", "UNCLEAR"))
    z = np.zeros(300)
    r = loss_of(z, z)
    chk("a zero loss is DISPENSABLE with a degenerate interval", r["loss"] == 0.0 and r["call"] == "DISPENSABLE")
    big = np.ones(300)
    chk("a loss of a full question is NEEDED", loss_of(big, np.zeros(300))["call"] == "NEEDED")
    chk("the bootstrap is deterministic (seed 0)", S.boot(cu - cd) == S.boot(cu - cd))
    if bad:
        raise SystemExit(f"zleg1: selftest FAILED ({bad})")
    log("selftest PASS (9 checks)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=lambda a: selftest())
    g = sub.add_parser("legs")
    g.add_argument("--ds", required=True)
    g.set_defaults(fn=cmd_legs)
    sub.add_parser("verdict").set_defaults(fn=cmd_verdict)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
