"""B1d: training-free chain slots with query offsets on HippoRAG 2's settings (docs/B1D_BRIDGE_SLOTS.md).

    python scripts/b1d_bridge_slots.py run [--datasets musique 2wiki hotpotqa]   -> outputs/bench/hipporag2/b1d.json, b1d.md
    python scripts/b1d_bridge_slots.py --selftest

Nothing is trained or re-encoded: the package's dense vectors (read through b1_build.dense_rows) and B1a's graphs and
first-stage lists only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_build as B  # noqa: E402
import b1c_walks as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
OUT_JSON, OUT_MD = BENCH / "b1d.json", BENCH / "b1d.md"
SETTINGS = ("musique", "2wiki", "hotpotqa")
S_CHOICES, M_CHOICES = (1, 2), (3, 4)
FAMSETS = {"all": ("structural", "ner", "knn"), "structural": ("structural",), "ner": ("ner",), "knn": ("knn",),
           "none": ()}
SCORES = ("Q", "QR")
NONE_TOP = 100
ARMS = ["R0"] + [f"s{s}m{m}/{f}/{c}" for s in S_CHOICES for m in M_CHOICES for f in FAMSETS for c in SCORES]
KS = (2, 5, 10)
LIST_LEN = 20


def parse(arm):
    sm, f, c = arm.split("/")
    return int(sm[1]), int(sm[3]), f, c


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def adjacency(d, fams, n):
    if not fams:
        return None
    a = sum(W.family_matrix(d, f, n) for f in fams)
    return (a > 0).astype(np.int8).tocsr()


def slot_list(base, s, m, adj, qscore, srcscore, length=LIST_LEN):
    """base: the first-stage list (node ids). qscore: cos(q, .) over all nodes (score Q) or None.
    srcscore: dict source -> score over all nodes (score QR) or None. Returns the new head of the list."""
    kept = [int(x) for x in base[:m]]
    keptset = set(kept)
    pos = {int(x): i for i, x in enumerate(base)}
    best = {}
    for r in (int(x) for x in base[:s]):
        if adj is None:
            cands = [int(x) for x in base[:NONE_TOP]]
        else:
            cands = adj.indices[adj.indptr[r]:adj.indptr[r + 1]].tolist()
        sc = qscore if srcscore is None else srcscore[r]
        for p in cands:
            if p in keptset or p == r:
                continue
            v = float(sc[p])
            if p not in best or v > best[p]:
                best[p] = v
    order = sorted(best, key=lambda p: (-best[p], pos.get(p, 10 ** 9), p))
    picked = order[:5 - m]
    head = kept + picked
    used = set(head)
    for x in base:
        if len(head) >= length:
            break
        if int(x) not in used:
            head.append(int(x))
            used.add(int(x))
    return head


def run_lists(base_lists, adjs, X, Q):
    """All arms for one first-stage list matrix. X, Q are unit vectors (nodes, questions)."""
    nq = base_lists.shape[0]
    qs_all = Q @ X.T                                            # (nq, n): score Q
    out = {"R0": base_lists[:, :LIST_LEN].astype(np.int64)}
    for s in S_CHOICES:
        src = base_lists[:, :s]
        U = unit(Q[:, None, :] + X[src])                        # (nq, s, dim): q + r
        sr = np.einsum("jsd,nd->jsn", U, X, optimize=True)      # (nq, s, n): score QR per source
        for m in M_CHOICES:
            for f, adj in adjs.items():
                for c in SCORES:
                    L = np.empty((nq, LIST_LEN), dtype=np.int64)
                    for j in range(nq):
                        srcscore = None if c == "Q" else {int(r): sr[j, i] for i, r in enumerate(src[j])}
                        L[j] = slot_list(base_lists[j], s, m, adj, qs_all[j], srcscore)
                    out[f"s{s}m{m}/{f}/{c}"] = L
    return out


def score_arms(lists, golds):
    nq = len(golds)
    even, odd = np.arange(0, nq, 2), np.arange(1, nq, 2)
    res = {}
    for arm, L in lists.items():
        ent = {}
        for k in KS:
            r = W.recall(L, golds, k)
            ent[f"R@{k}"] = {"even": round(float(r[even].mean()), 4), "odd": round(float(r[odd].mean()), 4),
                             "all": round(float(r.mean()), 4)}
        res[arm] = ent
    return res, even, odd


def run_setting(name):
    t0 = time.time()
    d = BENCH / name
    build = json.loads((d / "build.json").read_text(encoding="utf-8"))
    fs = np.load(d / "first_stage.npz")
    nodes = json.loads((d / "nodes.json").read_text(encoding="utf-8"))
    qrec = json.loads((d / "queries.json").read_text(encoding="utf-8"))
    golds = [np.asarray(q["golds"], np.int64) for q in qrec]
    n = int(build["passages"])
    rows = np.asarray([x["package_row"] for x in nodes], np.int64)
    X = unit(B.dense_rows(name, "docs", rows))
    Q = unit(B.dense_rows(name, "queries", fs["query_rows"].astype(np.int64)))
    adjs = {f: adjacency(d, fl, n) for f, fl in FAMSETS.items()}
    lists = run_lists(fs["rrf"].astype(np.int64), adjs, X, Q)
    log(f"{name}: {len(lists)} arms ({time.time() - t0:.0f}s)")
    res, even, odd = score_arms(lists, golds)
    chosen = max(ARMS, key=lambda a: (res[a]["R@5"]["even"], -ARMS.index(a)))
    dlt = W.recall(lists[chosen], golds, 5)[odd] - W.recall(lists["R0"], golds, 5)[odd]
    lo, hi = W.boot_ci(dlt)
    call = "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "SAME"
    offset = {f"s{s}m{m}/{f}": round(res[f"s{s}m{m}/{f}/QR"]["R@5"]["all"] - res[f"s{s}m{m}/{f}/Q"]["R@5"]["all"], 4)
              for s in S_CHOICES for m in M_CHOICES for f in FAMSETS}
    # E2: the chosen rule on SPLADE's list (reported, never used to choose)
    e2 = None
    if chosen != "R0":
        s, m, f, c = parse(chosen)
        sl = fs["splade"].astype(np.int64)
        src = sl[:, :s]
        sr = np.einsum("jsd,nd->jsn", unit(Q[:, None, :] + X[src]), X, optimize=True)
        qs_all = Q @ X.T
        L = np.empty((sl.shape[0], LIST_LEN), dtype=np.int64)
        for j in range(sl.shape[0]):
            srcscore = None if c == "Q" else {int(r): sr[j, i] for i, r in enumerate(src[j])}
            L[j] = slot_list(sl[j], s, m, adjs[f], qs_all[j], srcscore)
        r_new, r_base = W.recall(L, golds, 5), W.recall(sl, golds, 5)
        e2 = {"splade_R5": round(float(r_base.mean()), 4), "rule_on_splade_R5": round(float(r_new.mean()), 4),
              "gain": round(float((r_new - r_base).mean()), 4),
              "gain_on_rrf": round(res[chosen]["R@5"]["all"] - res["R0"]["R@5"]["all"], 4)}
    kinds = sorted({x["kind"] for x in qrec})
    kind_R5 = {k: {"R0": round(float(W.recall(lists["R0"], golds, 5)[[i for i in range(len(qrec)) if qrec[i]["kind"] == k]].mean()), 4),
                   "chosen": round(float(W.recall(lists[chosen], golds, 5)[[i for i in range(len(qrec)) if qrec[i]["kind"] == k]].mean()), 4)}
               for k in kinds}
    log(f"{name}: chosen {chosen}: R@5 {res[chosen]['R@5']['all']} vs R0 {res['R0']['R@5']['all']}; odd "
        f"{dlt.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] {call}; E2 {e2} ({time.time() - t0:.0f}s)")
    return {"arms": res, "chosen": chosen, "odd_delta_R5": round(float(dlt.mean()), 4),
            "odd_ci": [round(lo, 4), round(hi, 4)], "call": call, "offset_QR_minus_Q_R5": offset, "E2": e2,
            "kind_R5": kind_R5, "published_R5": W.PUBLISHED_R5[name], "build_sha256": W.sha(d / "build.json")}


def write_md(rec):
    S = rec["settings"]
    lines = ["# B1d: training-free chain slots with query offsets", "",
             "| setting | R0 (RRF) R@5 | chosen arm | its R@5 (all) | odd-half delta [95% CI] | call | HippoRAG 2 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in S.items():
        lines.append(f"| {name} | {100 * r['arms']['R0']['R@5']['all']:.1f} | {r['chosen']} | "
                     f"{100 * r['arms'][r['chosen']]['R@5']['all']:.1f} | {100 * r['odd_delta_R5']:+.1f} "
                     f"[{100 * r['odd_ci'][0]:+.1f}, {100 * r['odd_ci'][1]:+.1f}] | {r['call']} | {r['published_R5']} |")
    lines += ["", "R@5 (all 1,000) of every arm:", "", "| arm | " + " | ".join(S) + " |",
              "| --- | " + " | ".join("---:" for _ in S) + " |"]
    for a in ARMS:
        lines.append(f"| {a} | " + " | ".join(f"{100 * r['arms'][a]['R@5']['all']:.1f}" for r in S.values()) + " |")
    lines += ["", "QR minus Q, R@5 points (all):", "", "| rule | " + " | ".join(S) + " |",
              "| --- | " + " | ".join("---:" for _ in S) + " |"]
    for k in next(iter(S.values()))["offset_QR_minus_Q_R5"]:
        lines.append(f"| {k} | " + " | ".join(f"{100 * r['offset_QR_minus_Q_R5'][k]:+.1f}" for r in S.values()) + " |")
    lines += ["", "E2 (the chosen rule on SPLADE's list; reported, never used to choose):", ""]
    for name, r in S.items():
        lines.append(f"- {name}: {r['E2']}")
    lines += ["", "R@5 by question kind (R0 -> chosen):", ""]
    for name, r in S.items():
        lines.append(f"- {name}: " + "; ".join(f"{k} {v['R0']:.3f} -> {v['chosen']:.3f}" for k, v in r["kind_R5"].items()))
    tmp = Path(str(OUT_MD) + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, OUT_MD)
    print("\n".join(lines))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(names):
    rec = {"declared_in": "docs/B1D_BRIDGE_SLOTS.md", "script_sha256": W.sha(__file__), "arms": ARMS, "settings": {}}
    for name in names:
        rec["settings"][name] = run_setting(name)
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        W.write_json(OUT_JSON, rec)
    write_md(rec)
    return 0


def selftest():
    # 5 nodes; base list 0..4 then; node 4 linked to 0 and scores highest
    adj = sp.csr_matrix(np.array([[0, 0, 0, 0, 1, 1], [0] * 6, [0] * 6, [0] * 6, [1, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0]], np.int8))
    base = np.array([0, 1, 2, 3, 4, 5])
    qs = np.array([0, 0, 0, 0, 0.2, 0.9])
    h = slot_list(base, 1, 4, adj, qs, None, length=6)
    assert h == [0, 1, 2, 3, 5, 4], h
    h = slot_list(base, 1, 3, adj, qs, None, length=6)
    assert h == [0, 1, 2, 5, 4, 3], h
    h = slot_list(base, 1, 4, None, qs, None, length=6)
    assert h[:5] == [0, 1, 2, 3, 5] and len(set(h)) == 6
    sr = {0: np.array([0, 0, 0, 0, 0.8, 0.1])}
    assert slot_list(base, 1, 4, adj, qs, sr, length=6)[:5] == [0, 1, 2, 3, 4]
    assert parse("s2m3/ner/QR") == (2, 3, "ner", "QR") and len(ARMS) == 41
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("run",))
    ap.add_argument("--datasets", nargs="+", default=list(SETTINGS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.stage is None:
        return selftest()
    return run(a.datasets)


if __name__ == "__main__":
    sys.exit(main())
