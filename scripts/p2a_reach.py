"""P2a, reach: what each link family is worth to the pools, on all six full corpora (docs/P2A_LINK_FAMILIES.md, part 1).
A diagnostic: nothing is adopted, and no model trains or reads.

    python scripts/p2a_reach.py reach DATASET [--threads 5] [--host] [--limit N] [--out DIR] [--only-regime]
    python scripts/p2a_reach.py verdict
    python scripts/p2a_reach.py --selftest

Graphs: the three link families of M3B's stores (structural S, ner N, knn K), every non-empty subset of them (S, N, K,
SN, SK, NK, SNK), and none. Each walk is step 4e's kernel (its alpha and eps, cut at NMAX rows, nothing excluded) on the
subset's union: from a node, the walk picks a family uniformly among those in which the node has neighbours (step 4's
rule), and deg counts only the subset's entries, so A3's specificity follows the subset. On none, an order is its
restart's own nodes by weight, ties by node position.
Restarts: A3 (ours, step 4e's) and A1 (HippoRAG 2's rule) on all six; LINKED (uniform over our linked entities) on
metaqa and webqsp. Each order is cut at N = 25 to 7,000 and at every question's own |I_q| and |P4e|, as in P1a.
The dataset's regime subset (STRUCT: S, FULL: SNK) must reproduce P1a's A3, A1 and LINKED records on the five graph
datasets; squad's regime is none, where P1a walked nothing.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import p1a_pool_scoreboard as P  # noqa: E402
import step4b_pool_reach as S4B  # noqa: E402

E, F = P.E, P.F
S4 = E.S4
OUT = ROOT / "outputs" / "p2a"
VERDICT = OUT / "reach_verdict.json"
VERDICT_MD = OUT / "reach_verdict.md"
NS, NMAX, TIE, KB, SIX = P.NS, P.NMAX, P.TIE, P.KB, P.SIX
FAM = {"S": "structural", "N": "ner", "K": "knn"}
SUBSETS = ("none", "S", "N", "K", "SN", "SK", "NK", "SNK")
REGIME = {"STRUCT": "S", "FULL": "SNK", "RETRIEVAL": "none"}
EMPTY = (np.empty(0, np.int64), np.empty(0, np.float64))
log, utc, write_json, sha_file = P.log, P.utc, P.write_json, P.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def part_path(name):
    return OUT / f"reach_{name}.json"


def restarts_of(name):
    return ("A3", "A1") + (("LINKED",) if name in KB else ())


def canon(s):
    """A subset's name with its families in S, N, K order; the empty subset is none."""
    s = "".join(c for c in "SNK" if c in s)
    return s or "none"


def families_of(sub):
    return [] if sub == "none" else [FAM[c] for c in sub]


# ── per question ─────────────────────────────────────────────────────────────


def restart(o, prep, j, linked_j, deg, constant):
    """(nodes, weights) of restart o for question j on a graph with degrees deg."""
    if o == "LINKED":
        u = np.unique(np.asarray(linked_j, np.int64))
        return (u, np.full(u.size, 1.0 / u.size)) if u.size else EMPTY
    return E.restart(o, prep.seeds[j], prep.dense_ids[j], prep.splade_ids[j], linked_j, deg, constant)


def zero_order(r, nmax=NMAX):
    """No graph: the restart's own nodes by weight, ties by node position."""
    nodes, w = np.asarray(r[0], np.int64), np.asarray(r[1], np.float64)
    if nodes.size == 0:
        return np.empty(0, np.int64)
    return nodes[np.lexsort((nodes, -w))][:nmax]


def orders_on(un, rs, threads):
    """Each restart's order on the subset union un (None: no graph); an empty restart gives an empty order."""
    if un is None:
        return [zero_order(r) for r in rs]
    out = [np.empty(0, np.int64) for _ in rs]
    sel = [i for i, r in enumerate(rs) if np.asarray(r[0]).size]
    if sel:
        walked = E.walk(un, [rs[i] for i in sel], [np.empty(0, np.int64)] * len(sel), NMAX, threads)
        for i, w in zip(sel, walked):
            out[i] = w
    return out


def family_stores(op, name, union):
    """All three families' stores: the regime's from the union already open, the others from M3B's cache (refusing to
    build any)."""
    ds = op.canonical.Dataset(name, root=str(op.served))
    out = {}
    for f, s in zip(S4.FAMILIES, union.fams):
        if isinstance(s, S4B.Empty):
            path = op.m3b_compile.CSR_CACHE / f"{name}_{f}.npz"
            if not path.exists():
                raise SystemExit(f"{path} is missing: the stage reads M3B's stores, it does not build them")
            s = S4.m3b_pools.load_or_build_store(ds, f, op.m3b_compile.CSR_CACHE)
        out[f] = s
    n = {int(s.n_nodes) for s in out.values()}
    if len(n) != 1:
        raise SystemExit(f"{name}: the three families disagree on the node count ({sorted(n)})")
    return out, n.pop()


# ── the stage ────────────────────────────────────────────────────────────────


def reach_stage(name, threads, limit, out=None, only_regime=False):
    op = S4.Opened()
    out = Path(out) if out else OUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / part_path(name).name
    rec = {"declared_in": "docs/P2A_LINK_FAMILIES.md", "dataset": name, "subsets": "regime" if only_regime else list(SUBSETS),
           "restarts": list(restarts_of(name)), "ns": list(NS), "alpha": E.ALPHA, "eps": S4.EPS,
           "script_sha256": script_sha256(), "p1a_script_sha256": P.script_sha256(),
           "step4e_script_sha256": E.script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
           "limit": limit, "carves": {}}
    cache = {}

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        nq = len(U) if not limit else min(limit, len(U))
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds[:nq]]
        p4e = P.p4e_pools(name, carve, pop)
        sizes = {"I_q": np.asarray([np.asarray(inc[j]).size for j in range(nq)]),
                 "P4e": np.asarray([p4e[j].size for j in range(nq)])}
        if "stores" not in cache:
            cache["stores"], cache["n"] = family_stores(op, name, union)
        stores, n = cache["stores"], cache["n"]
        regime = REGIME[construction["regime"]]
        if canon("".join(c for c in "SNK" if FAM[c] in families)) != regime:
            raise SystemExit(f"{name}: the regime's families {families} are not {regime}")
        ent = {"questions": nq, "of": len(U), "regime": regime,
               "linked_share": round(float(np.mean([np.asarray(linked[j]).size > 0 for j in range(nq)])), 4),
               "sets": {"I_q": P.set_cov([np.asarray(inc[j], np.int64) for j in range(nq)], golds),
                        "P4e": P.set_cov(p4e[:nq], golds)},
               "subsets": {}}
        for sub in ((regime,) if only_regime else SUBSETS):
            t1 = time.time()
            un = None if sub == "none" else S4B.RegimeUnion(stores, families_of(sub), n)
            deg = np.zeros(n, dtype=np.int64) if un is None else un.deg
            res = {}
            for o in restarts_of(name):
                t0 = time.time()
                pos, lens = [], []
                for lo in range(0, nq, S4.BATCH):
                    js = range(lo, min(nq, lo + S4.BATCH))
                    rs = [restart(o, prep, j, linked[j], deg, constant) for j in js]
                    for j, od in zip(js, orders_on(un, rs, threads)):
                        pos.append(P.positions(od, golds[j]))
                        lens.append(int(np.asarray(od).size))
                res[o] = {"curve": P.curve(pos, lens), "len_mean": round(float(np.mean(lens)), 1),
                          "short": int(sum(x < NMAX for x in lens)),
                          "at_I_q": P.at_sizes(pos, sizes["I_q"]), "at_P4e": P.at_sizes(pos, sizes["P4e"]),
                          "seconds": round(time.time() - t0, 1)}
            ent["subsets"][sub] = {"orders": res, "entries": int(deg.sum()), "nodes_with_edges": int((deg > 0).sum()),
                                   "seconds": round(time.time() - t1, 1)}
            del un, deg
            gc.collect()
            log(f"{name}/{carve} {sub:4s} ({time.time() - t1:.0f}s): " + " | ".join(
                f"{o} " + " ".join(f"{k}:{res[o]['curve'][str(k)]['ALL']}" for k in (100, 1000, 7000))
                + f" @I_q {res[o]['at_I_q']['ALL']}" for o in res))
        ent["seconds"] = round(time.time() - t, 1)
        rec["carves"][f"{name}__{carve}"] = ent
        rec["utc"] = utc()
        write_json(path, rec)
        log(f"{name}/{carve}: {nq} q, {ent['seconds']:.0f}s, written")

    wanted = {"s1eval"} | ({"s1sel"} if name in E.TRAIN5 else set())
    E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


# ── the verdict (section 4 of the declared file) ─────────────────────────────


def value_points(ent, o, with_f, without_f):
    """with_f minus without_f at every N and both measures, and at the matched sizes."""
    a, b = ent["subsets"][with_f]["orders"][o], ent["subsets"][without_f]["orders"][o]
    pts = [{"N": k, "measure": m, "value": round(a["curve"][str(k)][m] - b["curve"][str(k)][m], 4)}
           for k in NS for m in ("ALL", "recall")]
    matched = {s: {m: round(a[s][m] - b[s][m], 4) for m in ("ALL", "recall")} for s in ("at_I_q", "at_P4e")}
    return pts, matched


def label(pts):
    up = any(p["value"] > TIE for p in pts)
    down = any(p["value"] < -TIE for p in pts)
    return "MIXED" if up and down else "CARRIES" if up else "HARMS" if down else "NOTHING"


def family_calls(ent, o):
    """Each family against the regime: left out of it (LOO) when it is in the regime, added to it (ADD) otherwise."""
    regime = ent["regime"]
    out = {}
    for c in "SNK":
        if c in regime:
            kind, with_f, without_f = "LOO", regime, canon(regime.replace(c, ""))
        else:
            kind, with_f, without_f = "ADD", canon(("" if regime == "none" else regime) + c), regime
        pts, matched = value_points(ent, o, with_f, without_f)
        above = [p for p in pts if p["value"] > TIE]
        below = [p for p in pts if p["value"] < -TIE]
        out[FAM[c]] = {"kind": kind, "with": with_f, "without": without_f, "label": label(pts),
                       "largest": max((p["value"] for p in pts), default=0.0),
                       "smallest": min((p["value"] for p in pts), default=0.0),
                       "above": above, "below": below, "matched": matched}
    return out


def subset_vs_regime(ent, o):
    """Every other subset against the dataset's regime: its label and extremes over every N and both measures."""
    out = {}
    for s in SUBSETS:
        if s != ent["regime"]:
            pts, matched = value_points(ent, o, s, ent["regime"])
            out[s] = {"label": label(pts), "largest": max(p["value"] for p in pts),
                      "smallest": min(p["value"] for p in pts), "matched": matched}
    return out


def universal(calls):
    """One subset for all six: a candidate when, under A3 on s1eval, it is never below a dataset's regime by more than the
    tie (CARRIES, NOTHING, or the regime itself) and above it on at least one dataset."""
    out = {}
    for s in SUBSETS:
        labs = {n: calls[f"{n}__s1eval"]["A3"]["vs_regime"].get(s, {"label": "REGIME"})["label"] for n in SIX}
        out[s] = {"labels": labs, "candidate": all(v in ("CARRIES", "NOTHING", "REGIME") for v in labs.values())
                  and any(v == "CARRIES" for v in labs.values())}
    return out


def candidates(calls, name):
    """A regime family is a drop candidate when leaving it out is NOTHING or HARMS under every restart on every carve;
    a family outside the regime is an add candidate when adding it CARRIES under some restart on s1eval and does not
    HARM or MIX under that restart on s1sel (where the dataset has one)."""
    carves = [calls[k] for k in (f"{name}__s1eval", f"{name}__s1sel") if k in calls]
    ev, sel = calls[f"{name}__s1eval"], calls.get(f"{name}__s1sel")
    out = {}
    for f in FAM.values():
        if ev["A3"]["families"][f]["kind"] == "LOO":
            drop = all(c[o]["families"][f]["label"] in ("NOTHING", "HARMS") for c in carves for o in restarts_of(name))
            out[f] = "DROP_CANDIDATE" if drop else "KEEP"
        else:
            add = any(ev[o]["families"][f]["label"] == "CARRIES"
                      and (sel is None or sel[o]["families"][f]["label"] in ("CARRIES", "NOTHING"))
                      for o in restarts_of(name))
            out[f] = "ADD_CANDIDATE" if add else "LEAVE_OUT"
    return out


def best_subsets(ent, o):
    """The subset with the most ALL at each N (ties to the regime, then to fewer families), and at |I_q|."""
    regime = ent["regime"]

    def key(sub, val):
        return (val, sub == regime, -len(families_of(sub)))
    best = {}
    for k in NS:
        vals = {s: ent["subsets"][s]["orders"][o]["curve"][str(k)]["ALL"] for s in SUBSETS}
        b = max(SUBSETS, key=lambda s: key(s, vals[s]))
        best[str(k)] = {"subset": b, "ALL": vals[b], "regime_ALL": vals[regime]}
    vals = {s: ent["subsets"][s]["orders"][o]["at_I_q"]["ALL"] for s in SUBSETS}
    b = max(SUBSETS, key=lambda s: key(s, vals[s]))
    best["at_I_q"] = {"subset": b, "ALL": vals[b], "regime_ALL": vals[regime]}
    return best


def p1a_check(ent, p1a_ent, name):
    """The regime subset must reproduce P1a's A3, A1 and LINKED records (curves and matched sizes) exactly."""
    bad = []
    if name == "squad":
        return bad
    reg = ent["subsets"][ent["regime"]]["orders"]
    for o in restarts_of(name):
        mine, theirs = reg[o], p1a_ent["orders"][o]
        for k in NS:
            for m in ("ALL", "recall"):
                if mine["curve"][str(k)][m] != theirs["curve"][str(k)][m]:
                    bad.append(f"{o} N={k} {m}: {mine['curve'][str(k)][m]} vs P1a {theirs['curve'][str(k)][m]}")
        for s in ("at_I_q", "at_P4e"):
            if mine[s] != theirs[s]:
                bad.append(f"{o} {s}: {mine[s]} vs P1a {theirs[s]}")
    return bad


def verdict_stage():
    parts, p1a = {}, {}
    for name in SIX:
        p = part_path(name)
        if not p.exists():
            raise SystemExit(f"{p} is missing")
        parts[name] = json.loads(p.read_text(encoding="utf-8"))
        q = P.part_path(name)
        if not q.exists():
            raise SystemExit(f"{q} is missing: the check reads P1a's records")
        p1a[name] = json.loads(q.read_text(encoding="utf-8"))
    shas = {(r["script_sha256"], r["freeze_RECORD_SHA256"]) for r in parts.values()}
    if len(shas) != 1:
        raise SystemExit(f"the parts ran on {len(shas)} scripts or freezes")
    if any(r.get("limit") or r.get("subsets") != list(SUBSETS) for r in parts.values()):
        raise SystemExit("a part ran with --limit or --only-regime")
    calls, checks = {}, {}
    for name in SIX:
        for key, ent in parts[name]["carves"].items():
            if ent["questions"] != p1a[name]["carves"][key]["questions"]:
                raise SystemExit(f"{key}: {ent['questions']} questions, P1a {p1a[name]['carves'][key]['questions']}")
            checks[key] = p1a_check(ent, p1a[name]["carves"][key], name)
            calls[key] = {o: {"families": family_calls(ent, o), "vs_regime": subset_vs_regime(ent, o),
                              "best": best_subsets(ent, o)} for o in restarts_of(name)}
    if any(checks.values()):
        write_json(OUT / "reach_check_failed.json", checks)
        raise SystemExit("the regime subsets do not reproduce P1a's records: see outputs/p2a/reach_check_failed.json")
    rec = {"declared_in": "docs/P2A_LINK_FAMILIES.md", "calls": calls, "universal": universal(calls),
           "candidates": {n: candidates(calls, n) for n in SIX}, "checks": "regime subsets equal P1a's records",
           "script_sha256": next(iter(shas))[0], "freeze_RECORD_SHA256": next(iter(shas))[1],
           "parts": {n: sha_file(part_path(n)) for n in SIX}, "p1a_parts": {n: sha_file(P.part_path(n)) for n in SIX},
           "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(parts, calls, rec["universal"], rec["candidates"]), encoding="utf-8")
    log("reach verdict written: " + "; ".join(
        f"{n} " + ", ".join(f"{f} {calls[f'{n}__s1eval']['A3']['families'][f]['kind']} "
                            f"{calls[f'{n}__s1eval']['A3']['families'][f]['label']} ({rec['candidates'][n][f]})"
                            for f in FAM.values()) for n in SIX)
        + "; universal candidates: " + (", ".join(s for s, v in rec["universal"].items() if v["candidate"]) or "none"))
    return 0


def report_md(parts, calls, uni, cands):
    lines = ["# P2a, reach: what each link family is worth to the pools (s1eval)", "",
             "With minus without the family, at every N and both measures; the regime is the dataset's frozen graph. "
             f"Labels use the tie {TIE}: CARRIES (above it somewhere, never below), HARMS, MIXED, NOTHING.", ""]
    for name in SIX:
        ent = parts[name]["carves"][f"{name}__s1eval"]
        lines += [f"## {name} (regime {ent['regime']}, {ent['questions']} questions)", "",
                  "| restart | family | kind | label | largest | smallest | at I_q (ALL) |", "| --- | --- | --- | --- | ---: | ---: | ---: |"]
        for o in restarts_of(name):
            for f, c in calls[f"{name}__s1eval"][o]["families"].items():
                lines.append(f"| {o} | {f} | {c['kind']} ({c['with']} vs {c['without']}) | {c['label']} | "
                             f"{c['largest']:+.4f} | {c['smallest']:+.4f} | {c['matched']['at_I_q']['ALL']:+.4f} |")
        lines += ["", "ALL by subset, A3:", "", "| subset | " + " | ".join(str(k) for k in (25, 100, 500, 1000, 2000, 7000))
                  + " | at I_q |", "| --- |" + " ---: |" * 7]
        for sub in SUBSETS:
            o = ent["subsets"][sub]["orders"]["A3"]
            lines.append(f"| {sub}{' (regime)' if sub == ent['regime'] else ''} | " + " | ".join(
                f"{o['curve'][str(k)]['ALL']:.3f}" for k in (25, 100, 500, 1000, 2000, 7000)) + f" | {o['at_I_q']['ALL']:.3f} |")
        lines += ["", "Candidates: " + ", ".join(f"{f} {v}" for f, v in cands[name].items()), ""]
    lines += ["## One subset for all six (A3, s1eval, against each dataset's regime)", "",
              "| subset | " + " | ".join(SIX) + " | candidate |", "| --- |" + " --- |" * (len(SIX) + 1)]
    for s, v in uni.items():
        lines.append(f"| {s} | " + " | ".join(v["labels"][n] for n in SIX) + f" | {'yes' if v['candidate'] else 'no'} |")
    return "\n".join(lines) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


class _Store:
    def __init__(self, n, edges):
        self.n_nodes = n
        src = np.asarray([a for a, _b in edges] + [b for _a, b in edges], np.int64)
        dst = np.asarray([b for _a, b in edges] + [a for a, _b in edges], np.int64)
        o = np.lexsort((dst, src))
        src, dst = src[o], dst[o]
        self.indptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(src, minlength=n), out=self.indptr[1:])
        self.col = dst.astype(np.int32)


def selftest():
    n = 12
    stores = {S4.FAMILIES[0]: _Store(n, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7)]),
              S4.FAMILIES[1]: _Store(n, [(1, 8), (8, 9), (4, 10)]),
              S4.FAMILIES[2]: _Store(n, [(10, 11), (2, 9)])}
    rs = [(np.asarray([0, 3]), np.asarray([0.75, 0.25])), (np.asarray([9]), np.asarray([1.0])), EMPTY]
    # a subset union equals a union built from those stores alone (the others empty), and deg counts the subset only
    for sub in SUBSETS[1:]:
        un = S4B.RegimeUnion(stores, families_of(sub), n)
        ref = S4.Union({f: (stores[f] if f in families_of(sub) else S4B.Empty(n)) for f in S4.FAMILIES})
        assert un.deg.tolist() == ref.deg.tolist(), sub
        assert un.deg.sum() == sum(int(stores[f].indptr[-1]) for f in families_of(sub)), sub
        got = orders_on(un, rs, 2)
        want = E.walk(ref, rs[:2], [np.empty(0, np.int64)] * 2, NMAX, 2)
        assert got[0].tolist() == want[0].tolist() and got[1].tolist() == want[1].tolist() and got[2].size == 0, sub
    # the regime subset is the regime union step 4e opens
    full = S4B.RegimeUnion(stores, list(S4.FAMILIES), n)
    assert orders_on(full, rs, 2)[0].tolist() == E.walk(S4.Union(stores), rs[:1], [np.empty(0, np.int64)], NMAX, 2)[0].tolist()
    # a walk leaves the seeds' component only through the subset's families
    s_only = orders_on(S4B.RegimeUnion(stores, ["structural"], n), [(np.asarray([0]), np.asarray([1.0]))], 1)[0]
    assert set(s_only.tolist()) <= {0, 1, 2, 3, 4, 5, 6, 7}, s_only
    # no graph: the restart's own nodes by weight, ties by node position
    assert zero_order((np.asarray([5, 2, 9, 1]), np.asarray([0.1, 0.4, 0.1, 0.4]))).tolist() == [1, 2, 5, 9]
    assert zero_order(EMPTY).size == 0
    # names and labels
    assert canon("NS") == "SN" and canon("") == "none" and canon("KNS") == "SNK"
    assert label([{"value": 0.001}, {"value": -0.001}]) == "NOTHING"
    assert label([{"value": 0.003}, {"value": -0.001}]) == "CARRIES"
    assert label([{"value": -0.003}]) == "HARMS" and label([{"value": 0.01}, {"value": -0.01}]) == "MIXED"
    # LOO / ADD against the regime, on a toy record
    def toy(vals):
        return {"orders": {"A3": {"curve": {str(k): {"ALL": vals, "recall": vals} for k in NS},
                                  "at_I_q": {"ALL": vals, "recall": vals}, "at_P4e": {"ALL": vals, "recall": vals}}}}
    ent = {"regime": "S", "subsets": {s: toy(0.5) for s in SUBSETS}}
    ent["subsets"]["S"] = toy(0.6)
    ent["subsets"]["SN"] = toy(0.55)
    fc = family_calls(ent, "A3")
    assert fc["structural"]["kind"] == "LOO" and fc["structural"]["without"] == "none" and fc["structural"]["label"] == "CARRIES"
    assert fc["ner"]["kind"] == "ADD" and fc["ner"]["with"] == "SN" and fc["ner"]["label"] == "HARMS"
    assert fc["knn"]["with"] == "SK" and fc["knn"]["label"] == "HARMS"
    ent2 = {"regime": "none", "subsets": {s: toy(0.5) for s in SUBSETS}}
    fc2 = family_calls(ent2, "A3")
    assert fc2["ner"]["with"] == "N" and fc2["ner"]["without"] == "none" and fc2["ner"]["label"] == "NOTHING"
    assert best_subsets(ent, "A3")["100"]["subset"] == "S"
    # drop and add candidates: every restart on every carve for a drop; some restart, not undone on s1sel, for an add
    def toy_calls(loo, add, rs=("A3", "A1")):
        fam = {f: {"kind": "LOO" if f == "structural" else "ADD", "label": loo if f == "structural" else add}
               for f in FAM.values()}
        return {o: {"families": fam} for o in rs}
    c = {"2wiki__s1eval": toy_calls("NOTHING", "CARRIES"), "2wiki__s1sel": toy_calls("HARMS", "NOTHING")}
    assert candidates(c, "2wiki") == {"structural": "DROP_CANDIDATE", "ner": "ADD_CANDIDATE", "knn": "ADD_CANDIDATE"}
    c["2wiki__s1sel"] = toy_calls("CARRIES", "MIXED")
    assert candidates(c, "2wiki") == {"structural": "KEEP", "ner": "LEAVE_OUT", "knn": "LEAVE_OUT"}
    assert candidates({"webqsp__s1eval": toy_calls("NOTHING", "CARRIES", restarts_of("webqsp"))}, "webqsp")["ner"] == "ADD_CANDIDATE"
    # one subset for all six: never below a regime by more than the tie, above one somewhere
    uc = {f"{nm}__s1eval": {"A3": {"vs_regime": {s: {"label": "NOTHING"} for s in SUBSETS}}} for nm in SIX}
    uc["musique__s1eval"]["A3"]["vs_regime"]["SN"] = {"label": "CARRIES"}
    del uc["2wiki__s1eval"]["A3"]["vs_regime"]["SN"]
    u = universal(uc)
    assert u["SN"]["candidate"] and u["SN"]["labels"]["2wiki"] == "REGIME" and not u["S"]["candidate"]
    uc["webqsp__s1eval"]["A3"]["vs_regime"]["SN"] = {"label": "MIXED"}
    assert not universal(uc)["SN"]["candidate"]
    print("selftest: ok (subset unions equal unions of those stores alone, deg counts the subset, zero-step orders, "
          "LOO/ADD against the regime, labels, best subsets, drop/add candidates, one subset for all six)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("reach", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="", help="write the part here instead (a development check)")
    ap.add_argument("--only-regime", action="store_true", help="walk the regime subset only (a development check)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "reach":
        if a.dataset not in SIX:
            raise SystemExit(f"dataset must be one of {SIX}")
        if a.host:
            import importlib.util
            spec = importlib.util.spec_from_file_location("s4f_run_one", ROOT / "outputs" / "step4f" / "run_one.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.on_the_mirror(F)
        return reach_stage(a.dataset, a.threads, a.limit, a.out, a.only_regime)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
