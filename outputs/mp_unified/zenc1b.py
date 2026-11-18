"""ENC E1b part 1 (docs/ENC_ENCODER_COMPARISON.md section 10): each encoder's base pool B under the record's own rule
over its first-stage lists, and B's gold share against E0's frozen arm. CPU only; nothing trains; outputs under
outputs/enc/e1b/ only.

  python outputs/mp_unified/zenc1b.py base --dataset musique --threads 4
  python outputs/mp_unified/zenc1b.py report
  python outputs/mp_unified/zenc1b.py selftest

B = the dataset's own construction's base rows (m3b_compile.base_rows) united with the seeds (m3b_pools.seeds_of), as
zpko.rule_batch forms it. Every callee is the record's, loaded unchanged.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zenc as E  # noqa: E402  (first: it sets the BLAS threads from --threads before numpy loads)

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import zap1 as AP1  # noqa: E402  (boot_diff and call, unchanged)

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mp_retrieval import m3b_pools  # noqa: E402

DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 10"
SETS = ("musique", "squad", "metaqa")
ARMS = ("frozen", "qwen3e06", "mpnet", "mpnetqa")
NEW = ARMS[1:]
BASE = {"musique": "equal_rrf_budget_200", "squad": "equal_rrf_budget_50", "metaqa": "equal_rrf_budget_50"}
RRF_CONSTANT = 60
PASS_LO = -1.0                      # points: the gap's lower bound must be above this
G3_SHARE, G3_PTS = 0.99, 0.1
OUT1B = E.OUT / "e1b"
PARTS = ROOT / "outputs" / "pko" / "pools" / "parts"
CODE = ("outputs/mp_unified/zenc1b.py", "outputs/mp_unified/zenc.py", "outputs/mp_unified/zap1.py",
        "scripts/m3b_compile.py", "scripts/m3b_contract.py", "scripts/m3a_headroom.py", "src/mp_retrieval/m3b_pools.py",
        "src/mp_retrieval/rank_fusion.py")


def stop(msg):
    raise SystemExit(f"ZENC1B: {msg}")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def pct(x, k=4):
    return round(100.0 * float(x), k)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def code_shas():
    return {"code_sha256": {c: E.sha256(ROOT / c) for c in CODE}}


# ── the record's rule, loaded unchanged ─────────────────────────────────────


def load_script(name):
    """As m3b_compile.load_script: the module registered under its own name before it runs."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def record_rule():
    mc = load_script("m3b_compile")
    ct = load_script("m3b_contract")
    return mc, ct, ct.load_headroom_module()


def bases(dense, splade, name, rule):
    """Each question's B: the own construction's base rows united with the seeds, sorted unique (zpko.rule_batch's B)."""
    mc, ct, m3a = rule
    dense, splade = np.ascontiguousarray(dense, np.int64), np.ascontiguousarray(splade, np.int64)
    base = mc.base_rows({"base_pool": name}, dense, splade, m3a, ct, RRF_CONSTANT)
    if len(base) != dense.shape[0]:
        stop("base_rows gave a different number of rows")
    return [np.unique(np.union1d(np.asarray(b, np.int64), m3b_pools.seeds_of(d, s))) for b, d, s in zip(base, dense, splade)]


# ── inputs ───────────────────────────────────────────────────────────────────


def e0_lists(d):
    out = {}
    for a in ARMS:
        p = E.OUT / "e0" / a / f"{d}.npz"
        if not p.is_file():
            stop(f"G1: no E0 list {rel(p)}")
        with np.load(p, allow_pickle=False) as z:
            out[a] = {"ids": [str(x) for x in z["ids"]], "rows": z["rows"].astype(np.int64),
                      "top": z["top_ids"].astype(np.int64), "gt": z["gold_total"].astype(np.int64),
                      "file": rel(p), "sha256": E.sha256(p)}
    f = out["frozen"]
    for a in NEW:
        o = out[a]
        if o["ids"] != f["ids"] or not np.array_equal(o["rows"], f["rows"]) or not np.array_equal(o["gt"], f["gt"]):
            stop(f"G1: {a}'s E0 ids, rows or gold totals on {d} are not the frozen arm's")
        if o["top"].shape != f["top"].shape:
            stop(f"G1: {a}'s E0 list on {d} has another shape")
    return out


def golds(d, ids, gt):
    p = E.OUT / "e0" / "gold" / f"{d}.npz"
    with np.load(p, allow_pickle=False) as z:
        if [str(x) for x in z["ids"]] != list(ids):
            stop(f"G1: the gold cache's ids on {d} are not E0's")
        flat, off = z["flat"].astype(np.int64), z["off"].astype(np.int64)
    if not np.array_equal(np.diff(off), gt):
        stop(f"G1: the gold cache's counts on {d} are not E0's gold totals")
    return flat, off, {"file": rel(p), "sha256": E.sha256(p)}


def filed_pools(d):
    """PKO's filed s1eval parts: each question's B (bs_nodes) and pool (B and the kept expansion), by id."""
    parts = sorted(PARTS.glob(f"{d}__s1eval__*of*.npz"))
    if not parts:
        stop(f"G1: no filed s1eval pools for {d}")
    nof = {int(p.stem.rsplit("of", 1)[1]) for p in parts}
    if len(nof) != 1 or len(parts) != nof.pop():
        stop(f"G1: {d}'s filed parts are not one complete set")
    ids, B, P, files = [], [], [], []
    for p in parts:
        rec = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        if rec.get("dataset") != d or rec.get("carve") != "s1eval":
            stop(f"G1: {p.name}'s record is not {d}/s1eval")
        sha = E.sha256(p)
        if sha != rec.get("file_sha256"):
            stop(f"G1: {p.name}'s sha256 is not its record's")
        with np.load(p, allow_pickle=False) as z:
            q = [str(x) for x in z["ids"]]
            bp, bn = z["bs_ptr"].astype(np.int64), z["bs_nodes"].astype(np.int64)
            ep, en = z["ptr"].astype(np.int64), z["nodes"].astype(np.int64)
            bsz, cnt = z["bs_size"].astype(np.int64), z["count"].astype(np.int64)
        n = len(q)
        if not (bp.size == ep.size == n + 1 and int(bp[-1]) == bn.size and int(ep[-1]) == en.size
                and np.array_equal(np.diff(bp), bsz) and np.array_equal(np.diff(ep), cnt)):
            stop(f"G1: {p.name} breaks its own pointers")
        for j in range(n):
            b, e = bn[bp[j]:bp[j + 1]], en[ep[j]:ep[j + 1]]
            ub = np.unique(b)
            if ub.size != b.size or np.isin(e, ub).any():
                stop(f"G1: {p.name} question {j}: B repeats a row or meets the expansion")
            B.append(ub)
            P.append(np.union1d(ub, e))
        ids += q
        files.append({"file": rel(p), "sha256": sha})
    if len(set(ids)) != len(ids):
        stop(f"G1: {d}'s filed pools repeat a question")
    return ids, B, P, files


# ── measures ─────────────────────────────────────────────────────────────────


def held_of(sets, flat, off):
    """Per gold slot: is it in its question's set; with the slots' question index."""
    nq = off.size - 1
    held = np.zeros(flat.size, bool)
    for q in range(nq):
        lo, hi = int(off[q]), int(off[q + 1])
        if hi > lo:
            held[lo:hi] = np.isin(flat[lo:hi], sets[q])
    return held, np.repeat(np.arange(nq), np.diff(off))


def shares(sets, flat, off):
    held, qidx = held_of(sets, flat, off)
    nq = off.size - 1
    gt = np.diff(off)
    per = np.bincount(qidx, weights=held.astype(float), minlength=nq)
    has = gt > 0
    return held, qidx, {"gold_share": pct(held.mean()) if held.size else None,
                        "all_golds_share": pct(np.mean(per[has] == gt[has])) if has.any() else None,
                        "mean_size": round(float(np.mean([s.size for s in sets])), 2)}


def gate_of(pt, lo, x):
    """Section 10.4, per dataset (points)."""
    if lo > PASS_LO:
        return "PASS"
    if x > 0 and pt > PASS_LO - x:                    # above -(1.0 + X)
        return "NEAR"
    return "FAIL"


def arm_call(gates):
    if any(g is None for g in gates):
        return "INCOMPLETE"
    if all(g == "PASS" for g in gates):
        return "CARRY"
    if "FAIL" in gates:
        return "HOLD"
    return "NEAR"


# ── base ─────────────────────────────────────────────────────────────────────


def cmd_base(a):
    d, t0 = a.dataset, time.time()
    if d not in SETS:
        stop(f"E1b part 1 reads {SETS}")
    if E.WS != ROOT:
        stop(f"run from the workspace root ({ROOT}), not {E.WS}")
    served = E.served_root()
    lists = e0_lists(d)
    f0 = lists["frozen"]
    ids, rows = f0["ids"], f0["rows"]
    flat, off, gfile = golds(d, ids, f0["gt"])
    pids, pB, pP, pfiles = filed_pools(d)
    at = {q: i for i, q in enumerate(pids)}
    miss = [q for q in ids if q not in at]
    if miss:
        stop(f"G1: {len(miss)} of E0's {d} ids are not in the filed pools")
    fB = [pB[at[q]] for q in ids]
    fP = [pP[at[q]] for q in ids]
    rc = served / d / "retrieval_cache"
    rec_dense = np.asarray(np.load(rc / "dense_top1000.npz", mmap_mode="r")["ids"][rows], np.int64)
    spl = np.asarray(np.load(rc / "splade_top1000.npz", mmap_mode="r")["ids"][rows], np.int64)
    rule = record_rule()
    log(f"{d}: {len(ids):,} questions, {flat.size:,} golds, base {BASE[d]}; {len(pfiles)} filed parts")

    bad = [i for i, b in enumerate(bases(rec_dense, spl, BASE[d], rule)) if not np.array_equal(b, fB[i])]
    g2 = {"questions": len(ids), "differ": len(bad), "first": [ids[i] for i in bad[:5]]}
    if bad:
        stop(f"G2 FAILED on {d}: {len(bad)} questions' rebuilt B differ from the filed B (first {g2['first']})")
    log(f"{d}: G2 passed (the rebuilt B equals the filed B on every question)")

    arms = {a_: bases(lists[a_]["top"], spl, BASE[d], rule) for a_ in ARMS}
    same = float(np.mean([np.array_equal(x, y) for x, y in zip(arms["frozen"], fB)]))
    hf, qidx, sf = shares(arms["frozen"], flat, off)
    hB, _, sB = shares(fB, flat, off)
    hP, _, sP = shares(fP, flat, off)
    g3 = {"equal_share": round(same, 6), "gold_share_diff_pts": round(sf["gold_share"] - sB["gold_share"], 4)}
    if same < G3_SHARE or abs(g3["gold_share_diff_pts"]) > G3_PTS:
        stop(f"G3 FAILED on {d}: {g3}")
    log(f"{d}: G3 passed ({g3})")
    x = round(sP["gold_share"] - sB["gold_share"], 4)
    nq = len(ids)
    res = {}
    for a_ in NEW:
        h, _, s = shares(arms[a_], flat, off)
        pt, lo, hi = AP1.boot_diff(h, hf, qidx, nq)
        pt, lo, hi = pct(pt), pct(lo), pct(hi)
        res[a_] = {**s, "gap_pts": [pt, lo, hi], "call": AP1.call(lo, hi), "gate": gate_of(pt, lo, x)}
    rec = {"stage": "ENC E1b part 1", "declared_in": DECLARED, "dataset": d, "base_pool": BASE[d],
           "rrf_constant": RRF_CONSTANT, "questions": nq, "golds": int(flat.size),
           "freeze_RECORD_SHA256": E.RECORD_SHA, "threads": E.THREADS,
           "inputs": {"e0": {a_: {"file": lists[a_]["file"], "sha256": lists[a_]["sha256"]} for a_ in ARMS},
                      "gold": gfile, "filed_pools": pfiles},
           "gates": {"G1": "PASS", "G2": g2, "G3": g3},
           "frozen": {"e0_B": sf, "filed_B": sB, "filed_pool": sP, "X_expansion_pts": x},
           "arms": res, "seconds": round(time.time() - t0, 1), **code_shas()}
    E.wjson(OUT1B / f"{d}.json", rec)
    log(f"{d}: written {rel(OUT1B / (d + '.json'))} ({rec['seconds']} s)")
    return 0


# ── report ───────────────────────────────────────────────────────────────────


def cmd_report(_a):
    recs = {}
    for d in SETS:
        p = OUT1B / f"{d}.json"
        if p.is_file():
            recs[d] = json.loads(p.read_text(encoding="utf-8"))
    calls = {a_: arm_call([recs[d]["arms"][a_]["gate"] if d in recs else None for d in SETS]) for a_ in NEW}
    rep = {"stage": "ENC E1b part 1", "declared_in": DECLARED, "sets": list(SETS), "read": sorted(recs),
           "calls": calls, "datasets": recs, **code_shas()}
    E.wjson(OUT1B / "report.json", rep)
    L = ["# ENC E1b part 1: each encoder's base pool B", "", f"Declared in {DECLARED}. Shares are % of golds in B; "
         "gaps are points against E0's frozen arm (95%, 2,000 resamples).", ""]
    for d in SETS:
        if d not in recs:
            L += [f"## {d}: not read", ""]
            continue
        r = recs[d]
        fz = r["frozen"]
        L += [f"## {d} ({r['questions']:,} questions, {r['golds']:,} golds, {r['base_pool']})", "",
              f"Frozen: E0 B {fz['e0_B']['gold_share']:.2f}; filed B {fz['filed_B']['gold_share']:.2f}; filed pool "
              f"{fz['filed_pool']['gold_share']:.2f}; X (expansion) {fz['X_expansion_pts']:+.2f}. G3 equal share "
              f"{r['gates']['G3']['equal_share']}.", "",
              "| arm | gold share | all golds | mean B | gap (pts, 95%) | call | gate |", "|---|---|---|---|---|---|---|"]
        for a_ in NEW:
            x = r["arms"][a_]
            pt, lo, hi = x["gap_pts"]
            L.append(f"| {a_} | {x['gold_share']:.2f} | {x['all_golds_share']:.2f} | {x['mean_size']:.1f} | "
                     f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}] | {x['call']} | {x['gate']} |")
        L.append("")
    L += ["## The calls", ""] + [f"- **{a_}: {c}**" for a_, c in calls.items()] + [""]
    (OUT1B / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def brute_b(d, s, budget, c=RRF_CONSTANT):
    """Equal RRF of the top 200 (1-based ranks, weight 0.5 each), ties by ascending id, the first `budget`, plus seeds."""
    sc = {}
    for lst in (d[:200], s[:200]):
        for r, x in enumerate(lst):
            sc[int(x)] = sc.get(int(x), 0.0) + 0.5 / (c + r + 1)
    order = sorted(sc, key=lambda x: (-sc[x], x))[:budget]
    seeds = list(d[:5]) + list(s[:5])
    return np.unique(np.asarray(order + [int(x) for x in seeds], np.int64))


def cmd_selftest(_a):
    rng = np.random.default_rng(0)
    rule = record_rule()
    n, N = 64, 3000
    dense = np.stack([rng.permutation(N)[:1000] for _ in range(n)])
    splade = np.stack([np.asarray(list(dict.fromkeys(np.concatenate([dense[i, :rng.integers(0, 40)],
                                                                     rng.permutation(N)]).tolist())))[:1000]
                       for i in range(n)])
    ok = True
    for name, bud in (("equal_rrf_budget_50", 50), ("equal_rrf_budget_200", 200)):
        got = bases(dense, splade, name, rule)
        want = [brute_b(dense[i], splade[i], bud) for i in range(n)]
        agree = sum(np.array_equal(g, w) for g, w in zip(got, want))
        seeded = all(np.isin(dense[i, :5], got[i]).all() and np.isin(splade[i, :5], got[i]).all() for i in range(n))
        print(f"selftest {name}: record rule == brute force on {agree}/{n}; seeds inside: {seeded}")
        ok = ok and agree == n and seeded
    # measures and gates on a toy
    flat, off = np.asarray([1, 2, 3, 4, 5], np.int64), np.asarray([0, 2, 5], np.int64)
    _, _, s = shares([np.asarray([1, 9]), np.asarray([3, 4, 5])], flat, off)
    print("selftest shares:", s)
    ok = ok and s["gold_share"] == 80.0 and s["all_golds_share"] == 50.0
    cases = [((-0.5, -0.9, 0.0), 0.0, "PASS"), ((-0.5, -1.2, 0.0), 0.0, "FAIL"), ((-1.5, -2.0, -1.0), 1.0, "NEAR"),
             ((-2.5, -3.0, -2.0), 1.0, "FAIL"), ((-1.5, -2.0, -1.0), 0.0, "FAIL")]
    for (pt, lo, _hi), x, want in cases:
        g = gate_of(pt, lo, x)
        ok = ok and g == want
        print(f"selftest gate pt {pt} lo {lo} X {x}: {g} (want {want})")
    for gs, want in ((["PASS"] * 3, "CARRY"), (["PASS", "NEAR", "PASS"], "NEAR"), (["PASS", "NEAR", "FAIL"], "HOLD"),
                     (["PASS", None, "PASS"], "INCOMPLETE")):
        ok = ok and arm_call(gs) == want
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base")
    b.add_argument("--dataset", required=True)
    b.add_argument("--threads", default="4")
    r = sub.add_parser("report")
    r.add_argument("--threads", default="1")
    t = sub.add_parser("selftest")
    t.add_argument("--threads", default="2")
    a = ap.parse_args()
    return {"base": cmd_base, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
