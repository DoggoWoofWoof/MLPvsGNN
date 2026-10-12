"""P1a: our pools on the full corpora beside the state-of-the-art pools run on the same corpora (docs/P1A_POOL_SCOREBOARD.md).
A diagnostic: nothing is adopted.

    python scripts/p1a_pool_scoreboard.py coverage DATASET [--carves s1eval s1sel] [--threads 5] [--host] [--limit N]
    python scripts/p1a_pool_scoreboard.py verdict
    python scripts/p1a_pool_scoreboard.py --selftest

Every order is cut at NMAX rows (step 4f's), with nothing excluded, on the dataset's regime union graph and step 4e's
kernel (its alpha and eps). Only the restart differs between the walks:
    A3      ours: step 4e's chosen restart (seeds and linked names at half the mass, RRF-weighted lists at the other half,
            HippoRAG's specificity on both); step 4f's O1
    A0      ours before step 4e: today's seeds only
    A1      HippoRAG 2's restart rule: seeds at half the mass, the RRF-weighted dense and SPLADE top-1000 lists at half
    A2      A1 with HippoRAG's node specificity
    LINKED  the KB-QA line from our linked entities: uniform over the exact-match linked nodes (step 4e's linking) only
    ORACLE  the KB-QA line's exposure (metaqa, webqsp only): uniform over the ASSIGNED topic entities. A diagnostic
            column, never a seed of ours
and the first-stage lists:
    DENSE, SPLADE   the top-1000 lists
    RRF     the equal-RRF order of the two lists
squad has no graph: every walk order is its RRF order (as in steps 4e and 4f), and LINKED and ORACLE are empty there.
A question with no linked node (LINKED) or no assigned topic entity (ORACLE) gets an empty order: those systems cannot
enter the graph without one.
Sets, each at its own size: I_q (the frozen pools today's models read), U_q (step 4d's), P4e (step 4e's chosen pools).
Each order is also cut at every question's own |I_q| and |P4e| (matched size).
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
import step4e_paper_pools as E  # noqa: E402
import step4f_pool_budget as F  # noqa: E402

OUT = ROOT / "outputs" / "p1a"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
NS = F.NS
NMAX = F.NMAX
WALKS = ("A3", "A0", "A1", "A2", "LINKED", "ORACLE")
LISTS = ("DENSE", "SPLADE", "RRF")
ORDERS = WALKS + LISTS
OURS = "A3"
SOTA = ("A1", "A2", "LINKED", "DENSE", "SPLADE", "RRF")
KB = ("metaqa", "webqsp")
M_KBQA = {"metaqa": 500, "webqsp": 2000}     # ReaRev's PageRank-Nibble top m (MetaQA, WebQSP)
TIE = 0.002                                  # step 4e's tie
SETS = ("I_q", "U_q", "P4e")
SIX = E.SIX
log, utc, write_json, sha_file = E.log, E.utc, E.write_json, E.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def part_path(name):
    return OUT / f"scoreboard_{name}.json"


# ── per question ─────────────────────────────────────────────────────────────


def positions(order, golds):
    """Each gold's position in the order, -1 where the order does not hold it."""
    order = np.asarray(order, np.int64)
    if order.size == 0 or golds.size == 0:
        return np.full(golds.size, -1, dtype=np.int64)
    srt = np.argsort(order, kind="stable")
    so = order[srt]
    k = np.minimum(np.searchsorted(so, golds), so.size - 1)
    return np.where(so[k] == golds, srt[k], -1).astype(np.int64)


def curve(pos, lens):
    """{N: ALL, recall, mean pool} over questions, the orders cut at N. pos: per-question gold positions."""
    out = {}
    for n in NS:
        a = r = 0.0
        for p in pos:
            hit = (p >= 0) & (p < n)
            a += bool(hit.all()) if p.size else 0.0
            r += float(hit.mean()) if p.size else 0.0
        nq = max(1, len(pos))
        out[str(n)] = {"ALL": round(a / nq, 4), "recall": round(r / nq, 4),
                       "pool": round(float(np.minimum(np.asarray(lens), n).mean()) if len(lens) else 0.0, 1)}
    return out


def at_sizes(pos, sizes):
    """ALL and recall with each question's order cut at its own size."""
    a = r = 0.0
    for p, s in zip(pos, sizes):
        hit = (p >= 0) & (p < s)
        a += bool(hit.all()) if p.size else 0.0
        r += float(hit.mean()) if p.size else 0.0
    nq = max(1, len(pos))
    return {"ALL": round(a / nq, 4), "recall": round(r / nq, 4), "size": round(float(np.mean(sizes)), 1)}


def set_cov(sets, golds):
    a = r = 0.0
    for s, g in zip(sets, golds):
        hit = np.isin(g, s)
        a += bool(hit.all()) if g.size else 0.0
        r += float(hit.mean()) if g.size else 0.0
    nq = max(1, len(sets))
    return {"ALL": round(a / nq, 4), "recall": round(r / nq, 4), "size": round(float(np.mean([s.size for s in sets])), 1)}


# ── topic entities (ORACLE; a diagnostic, read only here) ───────────────────


def assigned_topics(op, ds, name, pop):
    if name not in KB:
        return None
    recs = {}
    want = set(int(i) for i in pop.idx)
    for k, r in enumerate(ds.queries()):
        if k in want:
            recs[k] = r
    out = []
    if name == "metaqa":
        pmap = op.m3a.node_position_map(ds)
        for i in pop.idx:
            tid = recs[int(i)].get("topic_entity_node_id")
            out.append(np.asarray([pmap[tid]] if tid in pmap else [], dtype=np.int64))
    else:
        for i in pop.idx:
            v = recs[int(i)].get("topic_positions") or []
            if isinstance(v, str):
                v = json.loads(v)
            out.append(np.unique(np.asarray(v, dtype=np.int64)))
    return out


def p4e_pools(name, carve, pop):
    man = json.loads((E.MANIFEST).read_text(encoding="utf-8"))
    ent = man["files"][f"{name}__{carve}"]
    path = E.POOLS / ent["file"]
    if sha_file(path) != ent["sha256"]:
        raise SystemExit(f"{path}: not step 4e's manifest's file")
    with np.load(path) as z:
        ids, ptr, nodes, bp, bn = z["ids"], z["ptr"], z["nodes"], z["bs_ptr"], z["bs_nodes"]
        if ids.tolist() != list(pop.ids):
            raise SystemExit(f"{name}/{carve}: step 4e's ids are not the carve's")
        # step 4d's layout: the pool is base and seeds plus the expansion
        pools = [np.union1d(bn[bp[j]:bp[j + 1]], nodes[ptr[j]:ptr[j + 1]]).astype(np.int64) for j in range(len(pop.ids))]
    if abs(float(np.mean([p.size for p in pools])) - float(ent["pool_mean"])) > 0.05:
        raise SystemExit(f"{name}/{carve}: step 4e's pools average {np.mean([p.size for p in pools]):.1f} rows, "
                         f"its manifest {ent['pool_mean']}")
    return pools


# ── the stage ────────────────────────────────────────────────────────────────


def coverage_stage(name, carves, threads, limit):
    op = E.S4.Opened()
    path = part_path(name)
    rec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"carves": {}}
    rec.update({"declared_in": "docs/P1A_POOL_SCOREBOARD.md", "dataset": name, "orders": list(ORDERS),
                "ns": list(NS), "alpha": E.ALPHA, "eps": E.S4.EPS, "script_sha256": script_sha256(),
                "step4e_script_sha256": E.script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
                "limit": limit})

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        nq = len(U) if not limit else min(limit, len(U))
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds[:nq]]
        topics = assigned_topics(op, op.canonical.Dataset(name, root=str(op.served)), name, pop) if name in KB else None
        p4e = p4e_pools(name, carve, pop)
        none = [np.empty(0, np.int64)] * nq
        pos = {o: [] for o in ORDERS}
        lens = {o: [] for o in ORDERS}
        t_order = {o: 0.0 for o in ORDERS}
        for lo in range(0, nq, E.S4.BATCH):
            hi = min(nq, lo + E.S4.BATCH)
            js = range(lo, hi)
            batch = {}
            t0 = time.time()
            for o in ("DENSE", "SPLADE"):
                ids = prep.dense_ids if o == "DENSE" else prep.splade_ids
                batch[o] = [np.asarray(ids[j], np.int64) for j in js]
            batch["RRF"] = [E.rrf_order(prep.dense_ids[j], prep.splade_ids[j], none[j], constant, NMAX) for j in js]
            t_lists = time.time() - t0
            for o in LISTS:
                t_order[o] += t_lists / len(LISTS)
            for o in WALKS:
                t0 = time.time()
                if name == "squad":
                    batch[o] = batch["RRF"] if o in ("A3", "A0", "A1", "A2") else [np.empty(0, np.int64) for _ in js]
                elif o in ("A3", "A0", "A1", "A2"):
                    rs = [E.restart(o, prep.seeds[j], prep.dense_ids[j], prep.splade_ids[j], linked[j], union.deg,
                                    constant) for j in js]
                    batch[o] = E.walk(union, rs, none[lo:hi], NMAX, threads)
                else:
                    src = (linked if o == "LINKED" else topics) if (o == "LINKED" or name in KB) else None
                    got = [np.empty(0, np.int64) for _ in js]
                    if src is not None:
                        sel = [j for j in js if np.asarray(src[j]).size]
                        if sel:
                            rs = []
                            for j in sel:
                                u = np.unique(np.asarray(src[j], np.int64))
                                rs.append((u, np.full(u.size, 1.0 / u.size)))
                            walked = E.walk(union, rs, [none[0]] * len(sel), NMAX, threads)
                            for j, w in zip(sel, walked):
                                got[j - lo] = w
                    batch[o] = got
                t_order[o] += time.time() - t0
            for o in ORDERS:
                for j, order in zip(js, batch[o]):
                    pos[o].append(positions(order, golds[j]))
                    lens[o].append(int(np.asarray(order).size))
            del batch
        sizes = {"I_q": np.asarray([np.asarray(inc[j]).size for j in range(nq)]),
                 "P4e": np.asarray([p4e[j].size for j in range(nq)])}
        sets = {"I_q": set_cov([np.asarray(inc[j], np.int64) for j in range(nq)], golds),
                "U_q": set_cov([np.asarray(U[j], np.int64) for j in range(nq)], golds),
                "P4e": set_cov(p4e[:nq], golds)}
        ent = {"questions": nq, "of": len(U),
               "linked_share": round(float(np.mean([np.asarray(linked[j]).size > 0 for j in range(nq)])), 4),
               "oracle_share": (round(float(np.mean([topics[j].size > 0 for j in range(nq)])), 4)
                                if topics is not None else None),
               "orders": {o: {"curve": curve(pos[o], lens[o]), "len_mean": round(float(np.mean(lens[o])), 1),
                              "short": int(sum(x < NMAX for x in lens[o])),
                              "at_I_q": at_sizes(pos[o], sizes["I_q"]), "at_P4e": at_sizes(pos[o], sizes["P4e"]),
                              "seconds": round(t_order[o], 1)} for o in ORDERS},
               "sets": sets, "seconds": round(time.time() - t, 1)}
        rec["carves"][f"{name}__{carve}"] = ent
        rec["utc"] = utc()
        write_json(path, rec)
        log(f"{name}/{carve} ({nq} q, {ent['seconds']:.0f}s): " + " | ".join(
            f"{o} " + " ".join(f"{n}:{ent['orders'][o]['curve'][str(n)]['ALL']}" for n in (100, 1000, 7000))
            for o in ("A3", "A1", "LINKED", "RRF") + (("ORACLE",) if name in KB else ())))

    wanted = set(carves) & ({"s1eval"} | ({"s1sel"} if name in E.TRAIN5 else set()))
    E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


# ── the verdict (section 4 of the declared file) ─────────────────────────────


def call(ent, name):
    """AT_OR_ABOVE / BELOW for one carve: A3 against the best LLM-free SOTA order at every N."""
    below = []
    for n in NS:
        for m in ("ALL", "recall"):
            ours = ent["orders"][OURS]["curve"][str(n)][m]
            best_o = max(SOTA, key=lambda o: ent["orders"][o]["curve"][str(n)][m])
            best = ent["orders"][best_o]["curve"][str(n)][m]
            if ours < best - TIE:
                below.append({"N": n, "measure": m, "ours": ours, "sota": best, "order": best_o,
                              "gap": round(ours - best, 4)})
    out = {"call": "BELOW" if below else "AT_OR_ABOVE", "below": below,
           "largest_gap": min((b["gap"] for b in below), default=0.0)}
    for size in ("at_I_q", "at_P4e"):
        ours = ent["orders"][OURS][size]
        best_o = max(SOTA, key=lambda o: ent["orders"][o][size]["ALL"])
        out[size] = {"A3": ours, "best_sota": best_o, "sota": ent["orders"][best_o][size],
                     "set": ent["sets"]["I_q" if size == "at_I_q" else "P4e"]}
    if name in KB:
        m = M_KBQA[name]
        ora = ent["orders"]["ORACLE"]["curve"][str(m)]
        reach = next((n for n in NS if ent["orders"][OURS]["curve"][str(n)]["ALL"] >= ora["ALL"]), None)
        out["oracle"] = {"m": m, "ORACLE_at_m": ora, "A3_at_m": ent["orders"][OURS]["curve"][str(m)],
                         "LINKED_at_m": ent["orders"]["LINKED"]["curve"][str(m)],
                         "A3_reaches_oracle_ALL_at_N": reach, "oracle_share": ent["oracle_share"]}
    return out


def verdict_stage():
    parts = {}
    for name in SIX:
        p = part_path(name)
        if not p.exists():
            raise SystemExit(f"{p} is missing")
        parts[name] = json.loads(p.read_text(encoding="utf-8"))
    shas = {(r["script_sha256"], r["freeze_RECORD_SHA256"]) for r in parts.values()}
    if len(shas) != 1:
        raise SystemExit(f"the parts ran on {len(shas)} scripts or freezes")
    if any(r.get("limit") for r in parts.values()):
        raise SystemExit("a part ran with --limit")
    calls = {}
    for name, r in parts.items():
        for c, ent in r["carves"].items():
            calls[c] = call(ent, name)
    head = {n: calls[f"{n}__s1eval"]["call"] for n in SIX}
    rec = {"declared_in": "docs/P1A_POOL_SCOREBOARD.md", "calls": calls, "s1eval": head,
           "all_six_at_or_above": all(v == "AT_OR_ABOVE" for v in head.values()), "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(parts, calls), encoding="utf-8", newline="\n")
    log("calls on s1eval: " + ", ".join(f"{n} {v}" for n, v in head.items()))
    return 0


def report_md(parts, calls):
    L = ["# P1a: our pools against the SOTA pools on the full corpora (s1eval)", "",
         "ALL (every gold in the pool) at N rows; recall in brackets. Records: outputs/p1a/scoreboard_<dataset>.json.", ""]
    show = (100, 500, 1000, 2000, 7000)
    for name in SIX:
        ent = parts[name]["carves"][f"{name}__s1eval"]
        cl = calls[f"{name}__s1eval"]
        L += [f"## {name}: {cl['call']}", "", "| order | " + " | ".join(f"N = {n:,}" for n in show) + " |",
              "| --- |" + " ---: |" * len(show)]
        for o in ORDERS:
            if o == "ORACLE" and name not in KB:
                continue
            cv = ent["orders"][o]["curve"]
            L.append(f"| {o} | " + " | ".join(f"{cv[str(n)]['ALL']:.3f} ({cv[str(n)]['recall']:.3f})" for n in show)
                     + " |")
        L.append("")
        for size, label in (("at_I_q", "today's pools (I_q)"), ("at_P4e", "step 4e's pools")):
            s = cl[size]
            L.append(f"- At each question's size of {label} ({s['set']['size']:,.0f} rows mean): the set "
                     f"{s['set']['ALL']:.3f}, A3 {s['A3']['ALL']:.3f}, best SOTA order {s['best_sota']} "
                     f"{s['sota']['ALL']:.3f}.")
        if cl["below"]:
            L.append(f"- Below at {len(cl['below'])} points; the largest gap {cl['largest_gap']:+.4f}.")
        if name in KB:
            o = cl["oracle"]
            L.append(f"- Assigned topic entities (diagnostic), m = {o['m']:,}: ORACLE {o['ORACLE_at_m']['ALL']:.3f}, "
                     f"LINKED {o['LINKED_at_m']['ALL']:.3f}, A3 {o['A3_at_m']['ALL']:.3f}; A3 reaches ORACLE's ALL at "
                     f"N = {o['A3_reaches_oracle_ALL_at_N'] or 'beyond 7,000'}.")
        L.append("")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    g = np.asarray([3, 7, 9], np.int64)
    assert positions([9, 1, 3, 5], g).tolist() == [2, -1, 0]
    assert positions([], g).tolist() == [-1, -1, -1]
    pos = [positions([9, 1, 3, 5, 7], g), positions([3, 7, 9], g)]
    cv = curve(pos, [5, 3])
    assert cv["25"] == {"ALL": 1.0, "recall": 1.0, "pool": 4.0}, cv["25"]
    s = at_sizes(pos, np.asarray([3, 3]))
    assert s["ALL"] == 0.5 and abs(s["recall"] - (2 / 3 + 1) / 2) < 1e-4, s
    assert set_cov([np.asarray([3, 7]), np.asarray([3, 7, 9])], [g, g]) == {"ALL": 0.5, "recall": 0.8333, "size": 2.5}
    ent = {"orders": {o: {"curve": {str(n): {"ALL": 0.9, "recall": 0.95} for n in NS},
                          "at_I_q": {"ALL": 0.9, "recall": 0.95, "size": 5.0},
                          "at_P4e": {"ALL": 0.9, "recall": 0.95, "size": 9.0}} for o in ORDERS},
           "sets": {s_: {"ALL": 0.9, "recall": 0.95, "size": 5.0} for s_ in SETS}, "oracle_share": 1.0}
    assert call(ent, "2wiki")["call"] == "AT_OR_ABOVE"
    ent["orders"]["A1"]["curve"]["500"]["ALL"] = 0.9025
    assert call(ent, "2wiki")["call"] == "BELOW" and call(ent, "2wiki")["below"][0]["order"] == "A1"
    ent["orders"]["A1"]["curve"]["500"]["ALL"] = 0.9015
    assert call(ent, "2wiki")["call"] == "AT_OR_ABOVE"
    ent["orders"]["ORACLE"]["curve"]["2000"]["ALL"] = 0.95
    assert call(ent, "webqsp")["oracle"]["A3_reaches_oracle_ALL_at_N"] is None
    print("selftest: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("coverage", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--carves", nargs="+", default=["s1eval", "s1sel"])
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="development only: the first N questions of each carve")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "coverage":
        if a.dataset not in SIX:
            raise SystemExit(f"dataset must be one of {SIX}")
        if a.host:
            import importlib.util
            spec = importlib.util.spec_from_file_location("s4f_run_one", ROOT / "outputs" / "step4f" / "run_one.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.on_the_mirror(F)
        return coverage_stage(a.dataset, a.carves, a.threads, a.limit)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
