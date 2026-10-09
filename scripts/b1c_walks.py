"""B1c: training-free personalised PageRank on the B1 settings' graphs, and a ledger of the misses
(docs/B1C_WALKS_AND_LEDGER.md).

    python scripts/b1c_walks.py run [--datasets musique 2wiki hotpotqa]   -> outputs/bench/hipporag2/b1c.json, b1c.md
    python scripts/b1c_walks.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
OUT_JSON, OUT_MD = BENCH / "b1c.json", BENCH / "b1c.md"
RAW = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}
PUBLISHED_R5 = {"musique": 74.7, "2wiki": 90.4, "hotpotqa": 96.3}      # HippoRAG 2, arXiv 2502.14802, Table 3
ALPHA, ITERS, RRF_C = 0.5, 50, 60
RESTARTS = ("T5", "RW")
FAMSETS = {"all": ("structural", "ner", "knn"), "structural": ("structural",), "ner": ("ner",), "knn": ("knn",),
           "ner+structural": ("ner", "structural")}
ARMS = ["R0"] + [f"{r}/{f}" for r in RESTARTS for f in FAMSETS]
KS = (2, 5, 10)
BOOT, HMAX = 2000, 3
ENT_LABELS = {"PERSON", "GPE", "ORG", "LOC", "FAC", "WORK_OF_ART", "EVENT", "PRODUCT", "NORP"}
BUCKETS = ("N1", "C1", "T1", "R10", "FAR")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, obj):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


# ── graph and walk ───────────────────────────────────────────────────────────


def family_matrix(d, fam, n):
    z = np.load(d / f"graph_{fam}.npz")
    s, t = z["src"].astype(np.int64), z["dst"].astype(np.int64)
    w = np.ones(s.size) if fam == "structural" else z["weight"].astype(np.float64)
    a = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([s, t]), np.concatenate([t, s]))), shape=(n, n)).tocsr()
    a.sum_duplicates()
    return a


def transition_T(a):
    """T = P^T for P = D^-1 A (rows normalised); nodes with no edge keep their mass (a self loop)."""
    n = a.shape[0]
    deg = np.asarray(a.sum(1)).ravel()
    iso = deg == 0
    a = a + sp.diags(iso.astype(np.float64))
    deg = np.where(iso, 1.0, deg)
    return (sp.diags(1.0 / deg) @ a).T.tocsr()


def restart_matrix(kind, rrf, n):
    q = rrf.shape[0]
    r = np.zeros((n, q))
    for j in range(q):
        lst = rrf[j][rrf[j] >= 0]
        if kind == "T5":
            r[lst[:5], j] = 1.0 / 5
        else:
            w = 1.0 / (RRF_C + np.arange(1, lst.size + 1))
            r[lst, j] = w / w.sum()
    return r


def ppr(T, r, alpha=ALPHA, iters=ITERS):
    x = r.copy()
    for _ in range(iters):
        x = alpha * r + (1 - alpha) * (T @ x)
    return x


def rank_by(scores, rrf, n, top=1000):
    """Per question: nodes by score desc, ties by RRF rank, then node index."""
    q = scores.shape[1]
    out = np.empty((q, min(top, n)), dtype=np.int64)
    for j in range(q):
        rr = np.full(n, 10 ** 9, dtype=np.int64)
        lst = rrf[j][rrf[j] >= 0]
        rr[lst] = np.arange(lst.size)
        out[j] = np.lexsort((np.arange(n), rr, -scores[:, j]))[:out.shape[1]]
    return out


def recall(lists, golds, k):
    return np.array([np.isin(g, lists[i, :k]).mean() for i, g in enumerate(golds)])


def boot_ci(d, seed=0, n=BOOT):
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n)])
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# ── hops for the ledger ──────────────────────────────────────────────────────


def hops_from(a_bool, sources, targets, hmax=HMAX):
    n = a_bool.shape[0]
    seen = np.zeros(n, bool)
    seen[sources] = True
    dist = np.full(targets.size, -1)
    dist[seen[targets]] = 0
    front = seen.copy()
    for h in range(1, hmax + 1):
        nxt = (a_bool @ front.astype(np.float64)) > 0
        new = nxt & ~seen
        hit = new[targets] & (dist < 0)
        dist[hit] = h
        seen |= new
        front = new
        if not new.any() or (dist >= 0).all():
            break
    return dist


# ── run ──────────────────────────────────────────────────────────────────────


def run_setting(name, nlp):
    t0 = time.time()
    d = BENCH / name
    build = json.loads((d / "build.json").read_text(encoding="utf-8"))
    fs = np.load(d / "first_stage.npz")
    rrf = fs["rrf"].astype(np.int64)
    qrec = json.loads((d / "queries.json").read_text(encoding="utf-8"))
    golds = [np.asarray(q["golds"], np.int64) for q in qrec]
    n = int(build["passages"])
    fams = {f: family_matrix(d, f, n) for f in ("structural", "ner", "knn")}
    lists = {"R0": rrf}
    for fk, fl in FAMSETS.items():
        T = transition_T(sum(fams[f] for f in fl))
        for rk in RESTARTS:
            lists[f"{rk}/{fk}"] = rank_by(ppr(T, restart_matrix(rk, rrf, n)), rrf, n)
    log(f"{name}: {len(lists)} arms ranked ({time.time() - t0:.0f}s)")
    q = len(qrec)
    even, odd = np.arange(0, q, 2), np.arange(1, q, 2)
    res = {}
    for arm, L in lists.items():
        ent = {}
        for k in KS:
            r = recall(L, golds, k)
            ent[f"R@{k}"] = {"even": round(float(r[even].mean()), 4), "odd": round(float(r[odd].mean()), 4),
                             "all": round(float(r.mean()), 4)}
        res[arm] = ent
    chosen = max(ARMS, key=lambda a: (res[a]["R@5"]["even"], -ARMS.index(a)))
    dlt = recall(lists[chosen], golds, 5)[odd] - recall(rrf, golds, 5)[odd]
    lo, hi = boot_ci(dlt)
    call = "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "SAME"
    log(f"{name}: chosen {chosen}: R@5 all {res[chosen]['R@5']['all']} vs R0 {res['R0']['R@5']['all']}; "
        f"odd delta {dlt.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] {call}")
    # ledger
    L = lists[chosen]
    a_all = sum(fams.values()).astype(bool).astype(np.float64)
    a_fam = {f: m.astype(bool).astype(np.float64) for f, m in fams.items()}
    corpus = json.loads((ROOT / "outputs" / "bench" / "hipporag2" / "raw" / f"{RAW[name]}_corpus.json").read_text(encoding="utf-8"))
    qraw = json.loads((ROOT / "outputs" / "bench" / "hipporag2" / "raw" / f"{RAW[name]}.json").read_text(encoding="utf-8"))
    qents = [{e.text.lower().strip() for e in doc.ents if e.label_ in ENT_LABELS and len(e.text) > 2}
             for doc in nlp.pipe([x["question"] for x in qraw], batch_size=256)]
    rows = []
    for j in range(q):
        top5 = L[j, :5]
        g = golds[j]
        miss = g[~np.isin(g, top5)]
        if not miss.size:
            continue
        found = g[np.isin(g, top5)]
        h_all = hops_from(a_all, top5, miss)
        h_found = hops_from(a_all, found, miss) if found.size else np.full(miss.size, -1)
        h_f = {f: hops_from(m, top5, miss) for f, m in a_fam.items()}
        qn = norm(qraw[j]["question"])
        for i, x in enumerate(miss.tolist()):
            r_arm = np.flatnonzero(L[j] == x)
            r_r0 = np.flatnonzero(rrf[j] == x)
            t = norm(corpus[x]["title"])
            pe = {e.text.lower().strip() for e in nlp(corpus[x]["text"]).ents
                  if e.label_ in ENT_LABELS and len(e.text) > 2}
            m = {"q": j, "kind": qrec[j]["kind"], "gold": x, "golds": int(g.size), "found": int(found.size),
                 "rank": int(r_arm[0]) + 1 if r_arm.size else 0, "rank_r0": int(r_r0[0]) + 1 if r_r0.size else 0,
                 "hops_top5": int(h_all[i]), "hops_found": int(h_found[i]),
                 "hops_family": {f: int(v[i]) for f, v in h_f.items()},
                 "title_in_question": bool(len(t) >= 3 and f" {t} " in f" {qn} "),
                 "shared_entity": bool(qents[j] & pe)}
            m["named"] = m["title_in_question"] or m["shared_entity"]
            m["bucket"] = ("N1" if m["named"] else "C1" if m["hops_found"] == 1 else "T1" if m["hops_top5"] == 1
                           else "R10" if 0 < m["rank"] <= 10 else "FAR")
            rows.append(m)
    b = Counter(m["bucket"] for m in rows)
    nm = max(1, len(rows))
    summary = {
        "misses": len(rows), "golds": int(sum(x.size for x in golds)),
        "share": {k: round(b[k] / nm, 3) for k in BUCKETS},
        "named_share": round(sum(m["named"] for m in rows) / nm, 3),
        "title_share": round(sum(m["title_in_question"] for m in rows) / nm, 3),
        "second_of_chain_share": round(sum(m["found"] > 0 for m in rows) / nm, 3),
        "hops_top5": dict(Counter(str(m["hops_top5"]) for m in rows)),
        "one_hop_by_family": {f: round(sum(m["hops_family"][f] == 1 for m in rows) / nm, 3) for f in a_fam},
        "rank_median": float(np.median([m["rank"] if m["rank"] else 1001 for m in rows])),
        "by_kind": {k: {"misses": c, "share_of_misses": round(c / nm, 3)}
                    for k, c in sorted(Counter(m["kind"] for m in rows).items())},
        "kind_R5": {k: round(float(recall(L, golds, 5)[[i for i in range(q) if qrec[i]["kind"] == k]].mean()), 4)
                    for k in sorted({x["kind"] for x in qrec})},
    }
    log(f"{name}: {len(rows)} misses: " + " ".join(f"{k}={b[k]}" for k in BUCKETS) + f" ({time.time() - t0:.0f}s)")
    return {"arms": res, "chosen": chosen, "odd_delta_R5": round(float(dlt.mean()), 4), "odd_ci": [round(lo, 4), round(hi, 4)],
            "call": call, "published_R5": PUBLISHED_R5[name], "ledger_summary": summary, "ledger": rows,
            "build_sha256": sha(d / "build.json")}


def write_md(rec):
    lines = ["# B1c: training-free walks on HippoRAG 2's settings", "",
             "| setting | R0 (RRF) R@5 | chosen arm | its R@5 (all) | odd-half delta [95% CI] | call | HippoRAG 2 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in rec["settings"].items():
        lines.append(f"| {name} | {100 * r['arms']['R0']['R@5']['all']:.1f} | {r['chosen']} | "
                     f"{100 * r['arms'][r['chosen']]['R@5']['all']:.1f} | {100 * r['odd_delta_R5']:+.1f} "
                     f"[{100 * r['odd_ci'][0]:+.1f}, {100 * r['odd_ci'][1]:+.1f}] | {r['call']} | {r['published_R5']} |")
    lines += ["", "R@5 (all 1,000) of every arm:", "", "| arm | " + " | ".join(rec["settings"]) + " |",
              "| --- | " + " | ".join("---" for _ in rec["settings"]) + " |"]
    for a in ARMS:
        lines.append(f"| {a} | " + " | ".join(f"{100 * r['arms'][a]['R@5']['all']:.1f}" for r in rec["settings"].values()) + " |")
    lines += ["", "Ledger of the chosen arm's misses:", "",
              "| setting | misses / golds | " + " | ".join(BUCKETS) + " | named | second of a chain | median rank |",
              "| --- | --- | " + " | ".join("---" for _ in BUCKETS) + " | --- | --- | --- |"]
    for name, r in rec["settings"].items():
        s = r["ledger_summary"]
        lines.append(f"| {name} | {s['misses']} / {s['golds']} | " + " | ".join(f"{s['share'][k]:.3f}" for k in BUCKETS)
                     + f" | {s['named_share']:.3f} | {s['second_of_chain_share']:.3f} | {s['rank_median']:g} |")
    lines += [""]
    for name, r in rec["settings"].items():
        s = r["ledger_summary"]
        lines.append(f"- {name}: hops from top 5 {s['hops_top5']}; one hop by family {s['one_hop_by_family']}; "
                     f"R@5 by kind {s['kind_R5']}")
    tmp = Path(str(OUT_MD) + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, OUT_MD)
    print("\n".join(lines))


def run(names):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    rec = {"declared_in": "docs/B1C_WALKS_AND_LEDGER.md", "script_sha256": sha(__file__), "alpha": ALPHA,
           "iters": ITERS, "arms": ARMS, "settings": {}}
    for name in names:
        rec["settings"][name] = run_setting(name, nlp)
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(OUT_JSON, rec)
    write_md(rec)
    return 0


def selftest():
    a = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], float))
    T = transition_T(a)
    x = ppr(T, np.array([[1.0], [0], [0]]))
    assert x[0, 0] > x[1, 0] > 0 and x[2, 0] == 0
    rrf = np.array([[2, 1, 0]])
    assert rank_by(np.array([[0.0], [0.0], [0.0]]), rrf, 3).tolist() == [[2, 1, 0]]
    ab = (a > 0).astype(np.float64)
    assert hops_from(ab, np.array([0]), np.array([1, 2])).tolist() == [1, -1]
    assert norm("The Collegian (Houston)") == "the collegian houston"
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("run",))
    ap.add_argument("--datasets", nargs="+", default=list(RAW))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.stage is None:
        return selftest()
    return run(a.datasets)


if __name__ == "__main__":
    sys.exit(main())
