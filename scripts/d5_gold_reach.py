"""D5: where the golds today's pools miss sit in the graph (docs/D5_GOLD_REACH.md). A diagnostic: it reads golds,
decides nothing, and no model trains or reads.

    python scripts/d5_gold_reach.py run DATASET [--host] [--limit Q]   s1sel on the five training datasets;
                                                                        webqsp's s1eval beside
    python scripts/d5_gold_reach.py table
    python scripts/d5_gold_reach.py --selftest

Per question: B, its base and seeds (step 4d's, as every pool stage opens them); its golds; P1f's filed KIND_T depth
of each gold (outputs/p1f/stats_{ds}__{carve}.npz, checked question by question). Per gold:
    hop     links from B over the union of the three families (0 in B; 1, 2; 3 = not within two links, or the
            two-link neighbourhood was over the budget and is not expanded)
    fam1    for a gold one link from B: which families carry a link from B to it (bits structural 1, ner 2, knn 4)
    cnt     for a gold one link from B: how many nodes of B link to it
    r_cnt, r_w  its rank among N1 (B's one-link neighbours outside B), 1 = first, ordered by the count of B-nodes
            linking to it (r_cnt) or by the sum of those B-nodes' first-stage weights w = 1/(c + dense rank) +
            1/(c + splade rank) (r_w, P1e's steer), ties by node id
    today   whether today's pool (P1c2's, KIND_T at |I_q|) holds it
Per question: |N1| and, when expanded, |N2| (the nodes two links from B).
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

DECLARED = "docs/D5_GOLD_REACH.md"
OUT = ROOT / "outputs" / "d5"
P1F = ROOT / "outputs" / "p1f"
TRAIN5 = ("metaqa", "squad", "musique", "hotpotqa", "2wiki")
NO_DEPTH = np.int64(1) << 40                 # P1f's: a gold KIND_T's 7,000 nodes do not hold
FAMS = ("structural", "ner", "knn")
BUDGET2 = 5_000_000                          # neighbour entries read to expand N1 into N2; over it, hop 2 is not taken
NOT_N1 = np.int64(1) << 40                   # r_cnt / r_w for a gold not one link from B
DEPTH_BUCKETS = (("bs", 0, 0), ("1-100", 1, 100), ("101-1000", 101, 1000), ("1001-7000", 1001, 7000),
                 ("unheld", NO_DEPTH, NO_DEPTH))
TAKE = (100, 300, 1000, 3000)                # B plus N1's first k, taken from the orders themselves: the check
GRID = (50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 7000)   # the same coverage from the ranks, beside


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


# ── the per-question measures (pure numpy; the selftest runs them on a toy graph) ──


def neighbours(adj, nodes):
    """(ids, parent index) of every neighbour entry of nodes over the families in adj [(indptr, col)]: one entry per
    (family, link); fam code per entry."""
    ids, par, fam = [], [], []
    nodes = np.asarray(nodes, np.int64)
    for k, (ip, col) in enumerate(adj):
        a, b = ip[nodes], ip[nodes + 1]
        cnt = (b - a).astype(np.int64)
        if cnt.sum() == 0:
            continue
        idx = np.repeat(a - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(cnt.sum())
        ids.append(col[idx].astype(np.int64))
        par.append(np.repeat(np.arange(nodes.size), cnt))
        fam.append(np.full(int(cnt.sum()), 1 << k, np.int64))
    if not ids:
        e = np.empty(0, np.int64)
        return e, e, e
    return np.concatenate(ids), np.concatenate(par), np.concatenate(fam)


def measure(adj, B, golds, w_B):
    """One question. B sorted unique; golds unique; w_B: each B node's first-stage weight. Returns the per-gold
    arrays (hop, fam1, cnt, r_cnt, r_w), |N1|, |N2| (-1 when not expanded) and N1's two orders."""
    B = np.asarray(B, np.int64)
    g = np.asarray(golds, np.int64)
    ids, par, fam = neighbours(adj, B)
    out1 = ~np.isin(ids, B)
    ids, par, fam = ids[out1], par[out1], fam[out1]
    # one link: per distinct node, the count of distinct B-parents, their weight sum, the families
    key = ids * (B.size + 1) + par
    _, first = np.unique(key, return_index=True)
    u_ids, u_par = ids[first], par[first]
    n1, inv = np.unique(u_ids, return_inverse=True)
    cnt = np.bincount(inv, minlength=n1.size).astype(np.int64)
    wsum = np.bincount(inv, weights=np.asarray(w_B, np.float64)[u_par], minlength=n1.size)
    fams = np.zeros(n1.size, np.int64)
    np.bitwise_or.at(fams, np.searchsorted(n1, ids), fam)
    o_cnt = np.lexsort((n1, -cnt))           # by count desc, ties by node id
    o_w = np.lexsort((n1, -wsum))
    r_cnt = np.empty(n1.size, np.int64)
    r_cnt[o_cnt] = np.arange(1, n1.size + 1)
    r_w = np.empty(n1.size, np.int64)
    r_w[o_w] = np.arange(1, n1.size + 1)
    hop = np.full(g.size, 3, np.int64)
    hop[np.isin(g, B)] = 0
    p = np.searchsorted(n1, g)
    in1 = (p < n1.size) & (n1[np.minimum(p, max(n1.size - 1, 0))] == g) if n1.size else np.zeros(g.size, bool)
    hop[in1 & (hop != 0)] = 1
    G_fam = np.where(in1, fams[np.minimum(p, max(n1.size - 1, 0))] if n1.size else 0, 0)
    G_cnt = np.where(in1, cnt[np.minimum(p, max(n1.size - 1, 0))] if n1.size else 0, 0)
    G_rc = np.where(in1, r_cnt[np.minimum(p, max(n1.size - 1, 0))] if n1.size else NOT_N1, NOT_N1)
    G_rw = np.where(in1, r_w[np.minimum(p, max(n1.size - 1, 0))] if n1.size else NOT_N1, NOT_N1)
    n2 = -1
    rest = hop == 3
    entries = int(sum(int((ip[n1 + 1] - ip[n1]).sum()) for ip, _c in adj)) if n1.size else 0
    if rest.any() and entries <= BUDGET2:
        ids2, _p2, _f2 = neighbours(adj, n1)
        N2 = np.setdiff1d(np.unique(ids2), np.union1d(B, n1), assume_unique=False)
        n2 = int(N2.size)
        hop[rest & np.isin(g, N2)] = 2
    elif not rest.any():
        n2 = -2                              # nothing to look for two links out
    return {"hop": hop, "fam1": G_fam, "cnt": G_cnt, "r_cnt": G_rc, "r_w": G_rw}, int(n1.size), n2, \
        (n1[o_cnt], n1[o_w])


def weights_of(steer, B):
    """P1e's steer (K.steer's (nodes, w - floor, floor)) read at each node of B: w(v) = 1/(c + rank_dense) +
    1/(c + rank_splade), a list's rank its depth + 1 where it does not hold v."""
    u, ex, floor = steer
    w = np.full(B.size, float(floor))
    if u.size:
        k = np.minimum(np.searchsorted(u, B), u.size - 1)
        hit = u[k] == B
        w[hit] += ex[k[hit]]
    return w


# ── the stage (the host) ─────────────────────────────────────────────────────


def carves_of(name):
    return ("s1eval",) if name == "webqsp" else ("s1sel",)


def run(name, limit=0):
    import p1e_anchor_kind as K                     # step 4e's carves and P2b's stores, through P1e and P1d
    D, G, S4, E = K.D, K.G, K.S4, K.E
    op = S4.Opened()
    cache = {}
    rec = {"declared_in": DECLARED, "dataset": name, "script_sha256": script_sha256(), "budget2": BUDGET2,
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "carves": {}}

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        stores, _unions = G.open_unions(cache, op, name, union)
        adj = [(np.ascontiguousarray(stores[f].indptr, np.int64), np.ascontiguousarray(stores[f].col))
               for f in S4.FAMILIES]
        with np.load(P1F / f"stats_{name}__{carve}.npz") as z:
            st = {k: z[k] for k in ("bs", "inc", "ptr", "depth")}
        nq = min(limit, len(U)) if limit else len(U)
        golds = [np.unique(np.asarray(gg, np.int64)) for gg in pop.golds[:nq]]
        ptr = np.concatenate([[0], np.cumsum([gg.size for gg in golds])]).astype(np.int64)
        if not np.array_equal(st["ptr"][:nq + 1], ptr):
            raise SystemExit(f"{name}/{carve}: P1f's golds are not the carve's")
        cols = {k: [] for k in ("hop", "fam1", "cnt", "r_cnt", "r_w")}
        n1s, n2s, today, cov = [], [], [], {(o, k): [] for o in ("cnt", "w") for k in TAKE}
        for j in range(nq):
            B = np.unique(np.asarray(bs[j], np.int64))
            if B.size != int(st["bs"][j]):
                raise SystemExit(f"{name}/{carve} question {j}: |B| {B.size}, P1f's {int(st['bs'][j])}")
            dep = st["depth"][ptr[j]:ptr[j + 1]]
            take = max(0, int(st["inc"][j]) - B.size)
            today.append(dep <= take)
            m, n1, n2, (o_cnt, o_w) = measure(adj, B, golds[j], weights_of(K.steer(prep, j, constant), B))
            for k in cols:
                cols[k].append(m[k])
            n1s.append(n1)
            n2s.append(n2)
            for o, order in (("cnt", o_cnt), ("w", o_w)):
                for k in TAKE:
                    cov[(o, k)].append(bool(golds[j].size) and bool(np.isin(golds[j], np.union1d(B, order[:k])).all()))
            if (j + 1) % 250 == 0:
                log(f"{name}/{carve}: {j + 1}/{nq} ({time.time() - t:.0f}s)")
        a = {k: np.concatenate(v) if v else np.empty(0, np.int64) for k, v in cols.items()}
        a.update(ptr=ptr, depth=st["depth"][:ptr[-1]], today=np.concatenate(today) if today else np.empty(0, bool),
                 n1=np.asarray(n1s, np.int64), n2=np.asarray(n2s, np.int64))
        a.update({f"cov_{o}_{k}": np.asarray(v, bool) for (o, k), v in cov.items()})
        a["bs"] = st["bs"][:nq].astype(np.int64)
        bad = [f"{o}_{k}" for o in ("cnt", "w") for k in TAKE
               if not np.array_equal(q_all(a, (a["hop"] == 0) | (a[f"r_{o}"] <= k)), a[f"cov_{o}_{k}"])]
        if bad:
            raise SystemExit(f"{name}/{carve}: the ranks do not give the orders' coverage at {bad}")
        if limit:
            log(f"{name}/{carve}: development run on {nq} questions ({time.time() - t:.0f}s); nothing written: "
                f"{json.dumps(summary(a))}")
            return
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"reach_{name}__{carve}.npz"
        np.savez_compressed(OUT / f"reach_{name}__{carve}.tmp.npz", **a)
        os.replace(OUT / f"reach_{name}__{carve}.tmp.npz", p)
        rec["carves"][f"{name}__{carve}"] = {"questions": nq, "file_sha256": sha_file(p),
                                             "summary": summary(a), "seconds": round(time.time() - t, 1),
                                             "utc": utc()}
        write_json(OUT / f"reach_{name}.json", rec)
        log(f"{name}/{carve} ({nq} q, {time.time() - t:.0f}s): written")

    E.run_carves(op, name, set(carves_of(name)), ("A3",), 2, sink)
    return 0


# ── summaries ────────────────────────────────────────────────────────────────


def bucket_of(depth):
    out = np.empty(depth.size, object)
    for nm, lo, hi in DEPTH_BUCKETS:
        out[(depth >= lo) & (depth <= hi)] = nm
    return out


def q_all(a, ok):
    """Per question: it has golds and ok holds for every one (P1f's ALL: a question with no golds counts 0)."""
    ng = np.diff(a["ptr"])
    nq = ng.size
    qi = np.repeat(np.arange(nq), ng)
    good = np.bincount(qi, weights=np.asarray(ok, float), minlength=nq)
    return (ng > 0) & (good == ng)


def summary(a):
    hop, dep, today = a["hop"], a["depth"], a["today"]
    nq = a["n1"].size
    ng = np.diff(a["ptr"])
    qi = np.repeat(np.arange(nq), ng)
    s = {"questions": int(nq), "golds": int(hop.size)}
    b = bucket_of(dep)
    s["by_depth"] = {}
    for nm, _lo, _hi in DEPTH_BUCKETS:
        m = b == nm
        s["by_depth"][nm] = {"golds": int(m.sum()), **{f"hop{h}": int((m & (hop == h)).sum()) for h in range(4)}}
    miss = ~today
    s["missed_today"] = {"golds": int(miss.sum()), **{f"hop{h}": int((miss & (hop == h)).sum()) for h in range(4)}}
    m1 = miss & (hop == 1)
    if m1.any():
        s["missed_today"]["hop1_fam"] = {f: int(((a["fam1"][m1] >> k) & 1).sum()) for k, f in enumerate(FAMS)}
        for o in ("r_cnt", "r_w"):
            r = a[o][m1]
            s["missed_today"][f"hop1_{o}_quantiles"] = {q: int(np.quantile(r, q)) for q in (0.5, 0.8, 0.9, 0.95)}
        s["missed_today"]["hop1_cnt_quantiles"] = {q: int(np.quantile(a["cnt"][m1], q)) for q in (0.5, 0.9)}
    s["n1_quantiles"] = {q: int(np.quantile(a["n1"], q)) for q in (0.5, 0.9)} if nq else {}
    n2 = a["n2"][a["n2"] >= 0]
    s["n2_expanded"] = int(n2.size)
    s["n2_quantiles"] = {q: int(np.quantile(n2, q)) for q in (0.5, 0.9)} if n2.size else {}
    allq = np.zeros(nq, bool)
    if nq:
        good = np.bincount(qi, weights=today.astype(float), minlength=nq)
        allq = (ng > 0) & (good == ng)
    s["ALL_today"] = round(float(allq.mean()), 4) if nq else None
    hmax = np.zeros(nq, np.int64)
    if hop.size:
        np.maximum.at(hmax, qi, hop)
    s["ALL_within"] = {f"hop{h}": round(float(((ng > 0) & (hmax <= h)).mean()), 4) for h in range(3)} if nq else {}
    s["ALL_B_plus_N1_top"] = {f"{o}_{k}": round(float(a[f"cov_{o}_{k}"].mean()), 4)
                             for o in ("cnt", "w") for k in TAKE} if nq else {}
    s["n1_mean"] = round(float(a["n1"].mean()), 1) if nq else None
    if nq:
        h0 = hop == 0
        s["B_rows_mean"] = round(float(a["bs"].mean()), 1)
        s["grid"] = {o: {k: {"ALL": round(float(q_all(a, h0 | (a[f"r_{o}"] <= k)).mean()), 4),
                            "rows": round(float((a["bs"] + np.minimum(k, a["n1"])).mean()), 1)} for k in GRID}
                     for o in ("cnt", "w")}
        s["grid_today_or_w"] = {k: round(float(q_all(a, today | (a["r_w"] <= k)).mean()), 4) for k in GRID}
    return s


def table():
    rows = []
    for name in TRAIN5 + ("webqsp",):
        p = OUT / f"reach_{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        for c, e in r["carves"].items():
            rows.append((c, e["summary"]))
    lines = ["# D5: where the missed golds sit", "",
             "| carve | ALL today | ALL if every gold within 0/1/2 links of B | missed golds: 1 / 2 / 3+ links | |N1| p50 / "
             "p90 | missed 1-link golds' rank in N1 by count / by weight (p50, p90) |", "|---|---|---|---|---|---|"]
    for c, s in rows:
        mt = s["missed_today"]
        rc = mt.get("hop1_r_cnt_quantiles", {})
        rw = mt.get("hop1_r_w_quantiles", {})
        lines.append(f"| {c} | {s['ALL_today']} | {s['ALL_within'].get('hop0')} / {s['ALL_within'].get('hop1')} / "
                     f"{s['ALL_within'].get('hop2')} | {mt['hop1']} / {mt['hop2']} / {mt['hop3']} of {mt['golds']} | "
                     f"{s['n1_quantiles'].get(0.5, s['n1_quantiles'].get('0.5'))} / "
                     f"{s['n1_quantiles'].get(0.9, s['n1_quantiles'].get('0.9'))} | "
                     f"{rc.get('0.5', rc.get(0.5))}, {rc.get('0.9', rc.get(0.9))} / "
                     f"{rw.get('0.5', rw.get(0.5))}, {rw.get('0.9', rw.get(0.9))} |")
    lines += ["", "B plus the first k of N1 (ALL):", "", "| carve | " + " | ".join(
        f"{o} {k}" for o in ("cnt", "w") for k in TAKE) + " |", "|---|" + "---|" * (2 * len(TAKE))]
    for c, s in rows:
        lines.append(f"| {c} | " + " | ".join(str(s["ALL_B_plus_N1_top"][f"{o}_{k}"]) for o in ("cnt", "w")
                                              for k in TAKE) + " |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def csr(n, edges):
    indptr = np.zeros(n + 1, np.int64)
    src = np.asarray([a for a, _b in edges] + [b for _a, b in edges], np.int64)
    dst = np.asarray([b for _a, b in edges] + [a for a, _b in edges], np.int64)
    o = np.lexsort((dst, src))
    src, dst = src[o], dst[o]
    np.add.at(indptr, src + 1, 1)
    return np.cumsum(indptr), dst.astype(np.int32)


def selftest():
    # nodes 0..9; B = {0, 1}; structural 0-2, 1-2, 2-5; ner 1-3; knn 0-4, 4-6; 7 isolated
    st = csr(10, [(0, 2), (1, 2), (2, 5)])
    ner = csr(10, [(1, 3)])
    knn = csr(10, [(0, 4), (4, 6)])
    adj = [st, ner, knn]
    m, n1, n2, (o_cnt, o_w) = measure(adj, np.array([0, 1]), np.array([1, 2, 3, 5, 6, 7]), np.array([1.0, 0.25]))
    assert m["hop"].tolist() == [0, 1, 1, 2, 2, 3], m["hop"]
    assert n1 == 3 and n2 == 2                                   # N1 = {2, 3, 4}; N2 = {5, 6}
    assert m["cnt"].tolist() == [0, 2, 1, 0, 0, 0]               # 2 is linked from both B nodes
    assert m["fam1"].tolist() == [0, 1, 2, 0, 0, 0]
    assert o_cnt.tolist() == [2, 3, 4]                           # count 2 first, then ties by id
    assert o_w.tolist() == [2, 4, 3]                             # weights 1.25, 1.0 (from 0), 0.25 (from 1)
    assert m["r_cnt"].tolist()[1:3] == [1, 2] and m["r_w"].tolist()[1:3] == [1, 3]
    assert m["r_cnt"][0] == NOT_N1 and m["r_w"][4] == NOT_N1
    # a question with every gold one link out does not expand N2
    m, n1, n2, _o = measure(adj, np.array([0, 1]), np.array([2, 4]), np.array([1.0, 0.25]))
    assert m["hop"].tolist() == [1, 1] and n2 == -2
    # the steer read at B: a node in neither list gets the floor
    w = weights_of((np.array([0, 4]), np.array([0.5, 0.2]), 0.1), np.array([0, 1, 4]))
    assert np.allclose(w, [0.6, 0.1, 0.3]), w
    # the summary on two questions
    a = {"hop": np.array([0, 1, 3, 1]), "fam1": np.array([0, 4, 0, 1]), "cnt": np.array([0, 3, 0, 1]),
         "r_cnt": np.array([NOT_N1, 2, NOT_N1, 40]), "r_w": np.array([NOT_N1, 1, NOT_N1, 30]),
         "ptr": np.array([0, 3, 4]), "depth": np.array([0, 50, NO_DEPTH, 900]),
         "today": np.array([True, True, False, False]), "n1": np.array([100, 300]), "n2": np.array([5000, -2]),
         "bs": np.array([10, 20])}
    a.update({f"cov_{o}_{k}": np.array([False, k >= 1000]) for o in ("cnt", "w") for k in TAKE})
    s = summary(a)
    assert s["ALL_today"] == 0.0 and s["missed_today"] == {**s["missed_today"], "golds": 2, "hop1": 1, "hop3": 1}
    assert s["ALL_within"] == {"hop0": 0.0, "hop1": 0.5, "hop2": 0.5}
    assert s["by_depth"]["unheld"]["hop3"] == 1 and s["by_depth"]["101-1000"]["hop1"] == 1
    assert s["ALL_B_plus_N1_top"]["cnt_1000"] == 0.5 and s["ALL_B_plus_N1_top"]["w_100"] == 0.0
    assert s["grid"]["w"][50] == {"ALL": 0.5, "rows": 65.0} and s["grid"]["cnt"][50]["ALL"] == 0.5
    assert s["grid"]["cnt"][1]["ALL"] == 0.0 if 1 in GRID else True
    assert s["grid"]["w"][50]["ALL"] == 0.5 and s["grid_today_or_w"][50] == 0.5 and s["B_rows_mean"] == 15.0
    print("d5_gold_reach selftest: ok", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run", "table"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run(a.dataset, a.limit)
    if a.cmd == "table":
        return table()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
