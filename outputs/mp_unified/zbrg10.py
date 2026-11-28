"""BRG10 (docs/BRG10_ROW_LENGTH_SURFACE.md): the (L, H) surface of the row-length cap on the per-row-cap cells of MetaQA, HotpotQA and SQuAD, whole split.

Imports zbrg9.py (frozen; not edited) and rebinds its variant list, output folder and earlier-rows reader for the process: the walk, the replacement cap functions, the gates G-S0 and G-S1 and the per-variant
aggregate are BRG9's own. Variants are named L<L>H<H>: a row of at most L entries is kept whole, a longer row is cut at H entries; `eng` is the engine unpatched, `same` the length-rule functions with
the cell's own cap. Nothing is fitted by a model; the only choice is among the grid points, by the rule of section 4 of the page, on one half of the chunks and checked on the other.

    python -E -P -B outputs/mp_unified/zbrg10.py selftest
    python -E -P -B outputs/mp_unified/zbrg10.py screen --ds metaqa|hotpotqa|squad [--chunks N] [--smoke M] [--host]     # N = 0: every chunk; default 3
    python -E -P -B outputs/mp_unified/zbrg10.py read --ds metaqa|hotpotqa|squad                                           # outputs/brg10/read_<set>.json
    python -E -P -B outputs/mp_unified/zbrg10.py surface --ds metaqa|hotpotqa|squad                                        # the table of the read (whole split)
    python -E -P -B outputs/mp_unified/zbrg10.py common                                                                    # the pairs that are grid points of all three sets

Run from the workspace root.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import zbrg9 as Z9  # noqa: E402

DOC = "docs/BRG10_ROW_LENGTH_SURFACE.md"
OUT = ROOT / "outputs" / "brg10"
OUT9 = ROOT / "outputs" / "brg9"
SETS = ("metaqa", "hotpotqa", "squad")
GRIDS = {
    "metaqa": tuple((L, H) for L in (25, 40, 50, 75, 100, 150) for H in (3, 5, 8, 10, 15, 25) if H <= L and (L, H) != (25, 25)),
    "hotpotqa": tuple((L, H) for L in (50, 100, 200, 400) for H in (3, 5, 10, 25, 50, 100) if H <= L and (L, H) != (100, 100)),
}
GRIDS["squad"] = GRIDS["hotpotqa"]
B9 = {(50, 25): "V1", (100, 25): "V2", (100, 10): "V3", (200, 10): "V4", (200, 25): "V5", (100, 50): "V6", (50, 10): "V7"}      # BRG9's variants by (L, H)
FILED = {"metaqa": (9785, 0.933), "hotpotqa": (7405, 0.985), "squad": (11873, 0.996)}   # whole-split questions and DRP1 section 17's ALL of the cell (G-S4)
FLOOR, SLACK_FIT, SLACK_CHECK, POOL_DRIFT = 0.95, 0.005, 0.01, 0.03     # section 4 of the page


def name_of(L, H):
    return f"L{L}H{H}"


def variants(name):
    return (("eng", None, None), ("same", None, None)) + tuple((name_of(L, H), L, H) for L, H in GRIDS[name])


def old_rows9(name, c):
    """BRG9's rows of the same chunk: `eng` and every grid point that is one of its seven variants, to compare per question: (quest, column index, (variants, wanted pairs)); (None, None, None) when there are none."""
    p = OUT9 / name / f"scr_c{c:04d}.npz"
    if not p.is_file():
        return None, None, None
    z = np.load(p, allow_pickle=False)
    try:
        quest, cols, var = z["quest"], [str(x) for x in z["qcols"]], [str(x) for x in z["variants"]]
    finally:
        z.close()
    want = {"eng": "eng"}
    for L, H in GRIDS[name]:
        if (L, H) in B9:
            want[name_of(L, H)] = B9[(L, H)]
    return quest, {k: i for i, k in enumerate(cols)}, (var, want)


def use_grid(name):
    """Rebind BRG9's module-level variant list, column layout, output folder, page and earlier-rows reader for this process."""
    vs = variants(name)
    Z9.VARS = vs
    Z9.VN = tuple(v[0] for v in vs)
    Z9.QCOLS = ("q", "ng") + tuple(f"p{i}" for i in range(len(vs))) + tuple(f"s{i}" for i in range(len(vs)))
    Z9.QI = {c: i for i, c in enumerate(Z9.QCOLS)}
    Z9.OUT = OUT
    Z9.DOC = DOC
    Z9.old_rows = old_rows9


def choose(fit, check, names):
    """The declared rule. On `fit`: eligible = grid points whose ALL is at least max(0.95, eng's ALL - 0.005); the point is the eligible one with the smallest mean pool (ties: smaller H, then smaller L).
    On `check`: it holds when its ALL is at least 0.95, its paired call against `eng` is not BELOW, and its pool ratio is at most its pool ratio on `fit` plus 0.03."""
    ev = fit["variants"]
    thr = max(FLOOR, ev["eng"]["ALL"] - SLACK_FIT)
    ok = [nm for nm in names if ev[nm]["ALL"] >= thr]
    out = {"threshold": thr, "eligible": len(ok), "point": None}
    if not ok:
        return out
    LH = {nm: (int(nm[1:].split("H")[0]), int(nm.split("H")[1])) for nm in names}
    pt = min(ok, key=lambda nm: (ev[nm]["mean_nodes"], LH[nm][1], LH[nm][0]))
    cv = check["variants"]
    out.update({"point": pt, "L": LH[pt][0], "H": LH[pt][1], "fit": ev[pt], "check": cv[pt], "check_eng": cv["eng"]["ALL"],
                "holds": bool(cv[pt]["ALL"] >= FLOOR and cv[pt]["vs_eng"]["call"] != "BELOW" and cv[pt]["nodes_over_eng"] <= ev[pt]["nodes_over_eng"] + POOL_DRIFT)})
    return out


def cmd_read(a):
    use_grid(a.ds)
    name = a.ds
    folder = OUT / name
    gate = Z9.rjson(folder / "gate_s.json") if (folder / "gate_s.json").is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        Z9.stop(f"{name}: no PASS gate_s.json: no number is read")
    files = sorted(folder.glob("scr_c*.npz"))
    if len(files) != len(gate["chunks"]) or len(files) != len(gate["order"]):
        Z9.stop(f"{name}: {len(files)} chunk files, {len(gate['chunks'])} gate entries, {len(gate['order'])} chunks")
    Qs, pos = [], []
    for p in files:
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["qcols"]] != list(Z9.QCOLS) or [str(x) for x in z["variants"]] != list(Z9.VN) or str(z["cell"]) != Z9.CELL9[name]:
                Z9.stop(f"{p.name}: not this module's columns, variants or cell")
            Qs.append(z["quest"])
            pos.append(np.full(z["quest"].shape[0], int(z["pos"])))
        finally:
            z.close()
    Q, P = np.concatenate(Qs), np.concatenate(pos)
    agg = Z9.aggregate
    allp = agg(Q)
    nq, fil = FILED[name]
    g4 = {"questions": int(Q.shape[0]), "filed_questions": nq, "eng_ALL": allp["variants"]["eng"]["ALL"], "filed_ALL": fil}
    g4["ok"] = bool(Q.shape[0] == nq and abs(g4["eng_ALL"] - fil) <= 0.0015)
    if not g4["ok"]:
        Z9.stop(f"G-S4 failed: {g4}")
    fit, check = agg(Q[P % 2 == 0]), agg(Q[P % 2 == 1])
    names = [v[0] for v in Z9.VARS[2:]]
    rec = {"stage": "BRG10 read", "declared_in": DOC, "set": name, "cell": Z9.CELL9[name], "utc": Z9.utc(), "variants": [list(v) for v in Z9.VARS], "chunks_read": len(files), "G-S4": g4,
           "all": allp, "fit": fit, "check": check, "choice": choose(fit, check, names)}
    Z9.wjson(OUT / f"read_{name}.json", rec)
    c = rec["choice"]
    Z9.log(f"{name}: whole split {Q.shape[0]} questions, eng {100 * allp['variants']['eng']['ALL']:.1f} @{allp['variants']['eng']['mean_nodes']:.0f}; rule: threshold {100 * c['threshold']:.1f}, eligible {c['eligible']}, "
           f"point {c['point']}" + (f" fit {100 * c['fit']['ALL']:.1f} @{c['fit']['mean_nodes']:.0f}; check {100 * c['check']['ALL']:.1f} @{c['check']['mean_nodes']:.0f} ({100 * c['check']['vs_eng']['diff']:+.2f} {c['check']['vs_eng']['call']}), holds {c['holds']}" if c["point"] else ""))
    return 0


def cmd_surface(a):
    import json
    rec = json.loads((OUT / f"read_{a.ds}.json").read_text(encoding="utf-8"))
    x = rec["all"]
    e = x["variants"]["eng"]
    print(f"{a.ds}: {x['questions']} questions (whole split); eng ALL {100 * e['ALL']:.1f} @ {e['mean_nodes']:.0f}")
    print("variant      ALL   nodes  ratio  diff[ci] call")
    for nm, v in sorted(x["variants"].items(), key=lambda kv: kv[1]["mean_nodes"]):
        if nm in ("eng", "same"):
            continue
        d = v["vs_eng"]
        print(f"{nm:10s} {100 * v['ALL']:5.1f} {v['mean_nodes']:6.0f} {v['nodes_over_eng']:6.3f}  {100 * d['diff']:+6.2f} [{100 * d['ci95'][0]:+.2f},{100 * d['ci95'][1]:+.2f}] {d['call']}")
    return 0


def cmd_common(_a):
    import json
    recs = {n: json.loads((OUT / f"read_{n}.json").read_text(encoding="utf-8")) for n in SETS}
    com = [(L, H) for (L, H) in GRIDS["hotpotqa"] if (L, H) in GRIDS["metaqa"]]
    print("pair       " + "  ".join(f"{n:>26s}" for n in SETS) + "   eligible on fit (E) and holding on check (H) in all three; cell = fit ALL / check ALL @ whole-split pool ratio")
    for L, H in com:
        nm, cells, el = name_of(L, H), [], True
        for n in SETS:
            r = recs[n]
            thr = max(FLOOR, r["fit"]["variants"]["eng"]["ALL"] - SLACK_FIT)
            f, c, a = r["fit"]["variants"][nm], r["check"]["variants"][nm], r["all"]["variants"][nm]
            ok_fit = f["ALL"] >= thr
            ok_chk = c["ALL"] >= FLOOR and c["vs_eng"]["call"] != "BELOW"
            el &= ok_fit and ok_chk
            cells.append(f"{100 * f['ALL']:5.1f}/{100 * c['ALL']:5.1f} @{a['nodes_over_eng']:.3f} {'E' if ok_fit else '-'}{'H' if ok_chk else '-'}")
        print(f"{nm:10s} " + "  ".join(f"{c:>26s}" for c in cells) + f"   {el}")
    return 0


def synth(names, ng, hit, nq=600, seed=5):
    """A synthetic Q for the selftest: `hit` maps a variant to the share of questions whose golds are all in the pool; the pool size of variant v is 1000 - 10 v."""
    k = np.arange(nq)
    Q = np.zeros((nq, len(Z9.QCOLS)), np.int64)
    Q[:, Z9.QI["q"]] = k
    Q[:, Z9.QI["ng"]] = ng
    for v, nm in enumerate(Z9.VN):
        u = (k * 0.6180339887498949 + 0.137 * v + seed * 0.01) % 1.0      # equidistributed, no random noise in the share
        Q[:, Z9.QI[f"p{v}"]] = np.where(u < hit[nm], ng, ng - 1)
        Q[:, Z9.QI[f"s{v}"]] = 1000 - 10 * v
    return Q


def cmd_selftest(_a):
    rc = Z9.cmd_selftest(_a)
    assert rc == 0
    for nm in SETS:
        vs = variants(nm)
        names = [v[0] for v in vs]
        assert len(set(names)) == len(names), "variant names repeat"
        assert all(1 <= H <= L for _n, L, H in vs[2:]), "H above L"
        assert len(vs) == 2 + len(GRIDS[nm])
    assert len(GRIDS["metaqa"]) == 35 and len(GRIDS["hotpotqa"]) == 22 and GRIDS["squad"] == GRIDS["hotpotqa"]
    assert all(k in GRIDS["hotpotqa"] for k in B9), "BRG9's variants are all grid points of the c100 sets"
    assert {k for k in B9 if k in GRIDS["metaqa"]} == {(50, 25), (100, 25), (100, 10), (50, 10)}
    com = [(L, H) for (L, H) in GRIDS["hotpotqa"] if (L, H) in GRIDS["metaqa"]]
    assert com == [(50, 3), (50, 5), (50, 10), (50, 25), (100, 3), (100, 5), (100, 10), (100, 25)], com
    use_grid("hotpotqa")
    assert Z9.VN[0] == "eng" and Z9.VN[1] == "same" and Z9.OUT == OUT and len(Z9.QCOLS) == 2 + 2 * len(Z9.VN)
    names = list(Z9.VN[2:])
    hit = {nm: 0.90 for nm in Z9.VN}
    hit["eng"] = hit["same"] = 1.0
    hit[names[3]] = 1.0          # eligible, the pool of variant index 5 is 950
    hit[names[10]] = 0.80        # the smaller pool (index 12: 880) but not eligible
    hit[names[7]] = 1.0          # eligible, pool 910 (index 9): the smallest eligible
    Q = synth(names, 3, hit, nq=4000)
    P = np.arange(Q.shape[0]) // 100
    fit, check = Z9.aggregate(Q[P % 2 == 0]), Z9.aggregate(Q[P % 2 == 1])
    c = choose(fit, check, names)
    assert c["point"] == names[7] and c["eligible"] == 2 and abs(c["threshold"] - 0.995) < 1e-9, c
    assert c["holds"] is True and abs(c["check"]["nodes_over_eng"] - 0.91) < 1e-9, c
    # a point whose check half is below the floor does not hold
    Q2 = Q.copy()
    j = Z9.QI[f"p{Z9.VN.index(names[7])}"]
    Q2[(P % 2 == 1), j] = 3 - 1
    f2, c2 = Z9.aggregate(Q2[P % 2 == 0]), Z9.aggregate(Q2[P % 2 == 1])
    assert choose(f2, c2, names)["holds"] is False
    # nothing eligible
    hit2 = {nm: 0.5 for nm in Z9.VN}
    hit2["eng"] = hit2["same"] = 0.97
    Q3 = synth(names, 3, hit2, nq=4000)
    f3, c3 = Z9.aggregate(Q3[P % 2 == 0]), Z9.aggregate(Q3[P % 2 == 1])
    r3 = choose(f3, c3, names)
    assert r3["point"] is None and r3["eligible"] == 0, r3
    q, oi, vw = old_rows9("metaqa", 70)
    if q is not None:
        var, want = vw
        assert all(w in var for w in want.values())
    print("zbrg10 selftest: BRG9's selftest passes; the grids are 35 (MetaQA) and 22 (HotpotQA, SQuAD) distinct (L, H) pairs with H at most L; BRG9's seven variants are grid points; "
          "eight pairs are common to the three; the selection rule picks the smallest eligible pool, rejects an ineligible smaller one, holds or fails on the check half, and returns none when nothing is eligible: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("common")
    for c in ("screen", "read", "surface"):
        p = sub.add_parser(c)
        p.add_argument("--ds", choices=SETS, required=True)
        p.add_argument("--host", action="store_true")
        if c == "screen":
            p.add_argument("--smoke", type=int, default=0)
            p.add_argument("--chunks", type=int, default=3)
    a = ap.parse_args()
    a.kb = None
    if a.cmd == "selftest":
        return cmd_selftest(a)
    if a.cmd == "common":
        return cmd_common(a)
    if a.cmd == "read":
        return cmd_read(a)
    if a.cmd == "surface":
        return cmd_surface(a)
    use_grid(a.ds)
    return Z9.cmd_screen(a)


if __name__ == "__main__":
    sys.exit(main())
