"""P1d: one parameter-free pool rule for all six datasets (docs/P1D_FREE_POOL.md). Pools only: no model trains or
reads, and nothing is fit.

    python scripts/p1d_free_pool.py pools DATASET [--threads 5] [--host] [--limit Q]   s1sel (training) and s1eval
    python scripts/p1d_free_pool.py verdict
    python scripts/p1d_free_pool.py --selftest

Five sources per question, on every dataset (squad's families walked as P2b walks them), over M3B's stores and P2b's
two unions with step 4e's kernel (its alpha and eps, nothing excluded); a walk keeps its top NMAX nodes, its masses
scaled to sum to 1 (P1b's walk with masses):
    A1st, A1all   HippoRAG 2's restart (P1a's A1) walked on the structural family alone (st) and on all three (all)
    Kst, Kall     the corroborated anchors walked on st and on all: uniform over L*, the question's linked nodes that
                  the dense and SPLADE top HEAD also hold; empty when L* is
    R0            the RRF lists, unwalked: P1a's RRF order, with P1b's zero-step masses
Six rules, fixed before any number (RULES):
    A1st       A1(st)
    A1both     CombSUM of A1(st) and A1(all)
    ROUTEst    when the question has linked nodes and L* holds all of them: K(st), then A1(st)'s other nodes in
               A1(st)'s order; A1(st) otherwise
    ROUTEboth  the same with CombSUM of K(st) and K(all), then A1both's other nodes; A1both otherwise
    SUM5       CombSUM of the five sources
    RRF5       RRF over the five orders, with the first stage's constant
CombSUM is P1b's mixture with equal weights. Empty sources are skipped; every order is cut at NMAX, ties by node
position. One rule is chosen on the five training datasets' s1sel carves (the fewest points more than TIE below P1a's
bar, then the smaller largest gap, then the higher mean ALL at |I_q|, then RULES' order); the call is on s1eval,
against P1a's bar.
--limit Q (pools): a development run on the first Q questions of each carve. It writes nothing and logs no number,
only that every step ran, that the walks equal step 4e's own walk on the same restarts (and on the regime's own union
where the regime walks), and that R0 equals P1a's RRF order, order for order.
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
import p2b_family_gate as G  # noqa: E402

B, R2, P, E, F, S4, S4B = G.B, G.R2, G.P, G.E, G.F, G.S4, G.S4B
OUT = ROOT / "outputs" / "p1d"
P1B2 = ROOT / "outputs" / "p1b2"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
NS, NMAX, TIE, KB, SIX, TRAIN5 = B.NS, B.NMAX, B.TIE, B.KB, B.SIX, B.TRAIN5
HEAD = 100                  # P1b's head of the first-stage lists (its linked_in_lists feature)
UNIONS = G.UNIONS           # st: the structural family alone; all: the three families
SOURCES = ("A1st", "A1all", "Kst", "Kall", "R0")
RULES = ("A1st", "A1both", "ROUTEst", "ROUTEboth", "SUM5", "RRF5")
ORDERS = RULES + tuple(s for s in SOURCES if s not in RULES)
MEASURES = ("ALL", "recall")
# the declared checks: (our order, the filed record, P2a's subset, its order)
CHECKS = (("A1st", "p2a", "S", "A1"), ("A1all", "p2a", "SNK", "A1"), ("R0", "p1a", None, "RRF"))
LFL = ("A1st", "A1all", "L_st", "L_all", "DENSE", "SPLADE", "RRF")     # the like-for-like bar's orders
assert len(set(ORDERS)) == len(ORDERS) == 10 and len(set(LFL)) == len(LFL)
log, utc, write_json, sha_file = B.log, B.utc, B.write_json, B.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def part_path(name):
    return OUT / f"pools_{name}.json"


# ── the sources and the rules ────────────────────────────────────────────────


def anchors(prep, j, linked_j):
    """(L, L*): the question's linked nodes, and those the dense and SPLADE top HEAD also hold."""
    L = np.unique(np.asarray(linked_j, np.int64))
    head = np.union1d(np.asarray(prep.dense_ids[j][:HEAD], np.int64), np.asarray(prep.splade_ids[j][:HEAD], np.int64))
    return L, L[np.isin(L, head)]


def routes(L, Ls):
    """ROUTE's condition: the question has linked nodes and the lists' head holds every one."""
    return bool(L.size) and Ls.size == L.size


def a1_restart(prep, j, linked_j, deg, constant):
    return E.restart("A1", prep.seeds[j], prep.dense_ids[j], prep.splade_ids[j], linked_j, deg, constant)


def r0_source(prep, j, constant):
    """R0: the RRF lists unwalked, in P1a's RRF order (ties by node position on the raw scores; P1b's zero step sorts
    the scaled scores, which can tie two nodes the raw scores do not), with P1b's masses (the scores scaled to sum
    to 1)."""
    u, w = E.rrf_weights(prep.dense_ids[j], prep.splade_ids[j], constant)
    if u.size > NMAX:
        raise SystemExit(f"the RRF lists hold {u.size} nodes, more than {NMAX}")
    o = np.lexsort((u, -w))
    return u[o], w[o] / w.sum()


def sources_batch(prep, js, linked, unions, constant, threads):
    """Per question of js: the five SOURCES, each (nodes, masses) by mass, and (|L|, |L*|)."""
    anc = [anchors(prep, j, linked[j]) for j in js]
    a1 = [a1_restart(prep, j, linked[j], unions["st"].deg, constant) for j in js]
    k0 = [B.uniform(ls) for _l, ls in anc]
    walked = {}
    for u, _fams in UNIONS:
        walked[f"A1{u}"] = B.walk_mass(unions[u], a1, threads)
        walked[f"K{u}"] = B.walk_mass(unions[u], k0, threads)
    return [(tuple(walked[s][i] for s in SOURCES[:4]) + (r0_source(prep, j, constant),),
             (anc[i][0].size, anc[i][1].size), anc[i]) for i, j in enumerate(js)]


def csum(srcs, nmax=NMAX):
    """CombSUM: P1b's mixture with equal weights over the non-empty sources (each a distribution summing to 1)."""
    keep = [s for s in srcs if s[0].size]
    return B.mix_order(keep, np.ones(len(keep)), nmax) if keep else np.empty(0, np.int64)


def rrf(orders, c, nmax=NMAX):
    """RRF: a node scores sum over the non-empty orders of 1 / (c + its rank), ranks from 1; ties by node position."""
    keep = [np.asarray(o, np.int64) for o in orders if np.asarray(o).size]
    if not keep:
        return np.empty(0, np.int64)
    ids = np.concatenate(keep)
    sc = np.concatenate([1.0 / (c + np.arange(1, o.size + 1, dtype=np.float64)) for o in keep])
    u, inv = np.unique(ids, return_inverse=True)
    m = np.bincount(inv, weights=sc, minlength=u.size)
    return u[np.lexsort((u, -m))][:nmax]


def backfill(head, rest, nmax=NMAX):
    """head's order, then rest's nodes that head does not hold, in rest's order; cut at nmax."""
    head = np.asarray(head, np.int64)[:nmax]
    if head.size >= nmax:
        return head
    rest = np.asarray(rest, np.int64)
    return np.concatenate([head, rest[~np.isin(rest, head)]])[:nmax]


def rule_orders(src, route, constant):
    """Every order of ORDERS for one question, from its five sources and ROUTE's condition."""
    a1st, a1all, kst, kall, r0 = src
    o = {"A1st": a1st[0][:NMAX], "A1all": a1all[0][:NMAX], "Kst": kst[0][:NMAX], "Kall": kall[0][:NMAX],
         "R0": r0[0][:NMAX]}
    o["A1both"] = csum([a1st, a1all])
    o["ROUTEst"] = backfill(o["Kst"], o["A1st"]) if route else o["A1st"]
    o["ROUTEboth"] = backfill(csum([kst, kall]), o["A1both"]) if route else o["A1both"]
    o["SUM5"] = csum(list(src))
    o["RRF5"] = rrf([o[s] for s in SOURCES], constant)
    return o


def dev_check(name, prep, js, linked, unions, union, regime, constant, threads, srcs):
    """A development run's check: every walk equals step 4e's own walk on the same restart and union (and on the
    regime's own union where the regime walks), and R0 equals P1a's RRF order."""
    none = np.empty(0, np.int64)
    a1 = [a1_restart(prep, j, linked[j], unions["st"].deg, constant) for j in js]
    k0 = [B.uniform(anchors(prep, j, linked[j])[1]) for j in js]
    regs = {"S": "st", "SNK": "all"}
    for u, _fams in UNIONS:
        pairs = [(unions[u], "P2b's union")] + ([(union, "the regime's union")] if regs.get(regime) == u else [])
        for s, rs in ((f"A1{u}", a1), (f"K{u}", k0)):
            k = SOURCES.index(s)
            sel = [i for i, r in enumerate(rs) if r[0].size]
            got = [srcs[i][0][k][0] for i in sel]
            if any(srcs[i][0][k][0].size for i in range(len(rs)) if i not in set(sel)):
                raise SystemExit(f"{name}: {s} walked an empty restart to a non-empty order")
            for un, what in pairs:
                want = E.walk(un, [rs[i] for i in sel], [none] * len(sel), NMAX, threads) if sel else []
                bad = sum(not np.array_equal(g, w) for g, w in zip(got, want))
                if bad:
                    raise SystemExit(f"{name}: {s} differs from step 4e's walk on {what} on {bad} questions")
    for i, j in enumerate(js):
        if not np.array_equal(srcs[i][0][4][0], E.rrf_order(prep.dense_ids[j], prep.splade_ids[j], none, constant, NMAX)):
            raise SystemExit(f"{name}: R0 differs from P1a's RRF order")


# ── the stage (s1sel, s1eval) ────────────────────────────────────────────────


def pools_stage(name, threads, limit=0):
    op = S4.Opened()
    rec = {"declared_in": "docs/P1D_FREE_POOL.md", "dataset": name, "orders": list(ORDERS), "ns": list(NS),
           "alpha": E.ALPHA, "eps": S4.EPS, "head": HEAD, "script_sha256": script_sha256(),
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "limit": limit, "carves": {}}
    cache = {}

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        regime = G.regime_of(families, construction)
        _stores, unions = G.open_unions(cache, op, name, union)
        nq = min(limit, len(U)) if limit else len(U)
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds[:nq]]
        p4e = P.p4e_pools(name, carve, pop)
        pos = {o: [] for o in ORDERS}
        lens = {o: [] for o in ORDERS}
        n_linked = n_corr = n_route = 0
        for lo in range(0, nq, S4.BATCH):
            js = range(lo, min(nq, lo + S4.BATCH))
            srcs = sources_batch(prep, js, linked, unions, constant, threads)
            if limit:
                dev_check(name, prep, js, linked, unions, union, regime, constant, threads, srcs)
            for j, (src, (nl, nc), (L, Ls)) in zip(js, srcs):
                route = routes(L, Ls)
                n_linked += nl > 0
                n_corr += nc > 0
                n_route += route
                orders = rule_orders(src, route, constant)
                for o in ORDERS:
                    pos[o].append(P.positions(orders[o], golds[j]))
                    lens[o].append(int(orders[o].size))
        sizes = {"I_q": np.asarray([np.asarray(inc[j]).size for j in range(nq)]),
                 "P4e": np.asarray([p4e[j].size for j in range(nq)])}
        ent = {"questions": nq, "carve_sha256": pop.digest, "regime": regime,
               "shares": {"linked": round(n_linked / max(1, nq), 4), "corroborated": round(n_corr / max(1, nq), 4),
                          "route": round(n_route / max(1, nq), 4)},
               "orders": {o: {"curve": P.curve(pos[o], lens[o]), "len_mean": round(float(np.mean(lens[o])), 1),
                              "short": int(sum(x < NMAX for x in lens[o])),
                              "at_I_q": P.at_sizes(pos[o], sizes["I_q"]), "at_P4e": P.at_sizes(pos[o], sizes["P4e"])}
                          for o in ORDERS},
               "sets": {"I_q": P.set_cov([np.asarray(inc[j], np.int64) for j in range(nq)], golds),
                        "P4e": P.set_cov(p4e[:nq], golds)},
               "seconds": round(time.time() - t, 1)}
        if limit:
            log(f"{name}/{carve}: development run on {nq} questions ({time.time() - t:.0f}s): every step ran, the walks "
                "equal step 4e's on the same restarts and unions, and R0 equals P1a's RRF order; nothing written")
            return
        OUT.mkdir(parents=True, exist_ok=True)
        ppath = OUT / f"pos_{name}__{carve}.npz"
        tmp = OUT / f"pos_{name}__{carve}.tmp.npz"
        np.savez(tmp, ptr=np.concatenate([[0], np.cumsum([g.size for g in golds])]).astype(np.int64),
                 **{o: np.concatenate(pos[o]).astype(np.int32) for o in ORDERS})
        os.replace(tmp, ppath)
        ent["positions_npz_sha256"] = sha_file(ppath)
        rec["carves"][f"{name}__{carve}"] = ent
        rec["utc"] = utc()
        write_json(part_path(name), rec)
        log(f"{name}/{carve} ({nq} q, {ent['seconds']:.0f}s): written; shares linked {ent['shares']['linked']}, "
            f"corroborated {ent['shares']['corroborated']}, route {ent['shares']['route']}")

    wanted = {"s1eval"} | ({"s1sel"} if name in TRAIN5 else set())
    E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


# ── the verdict (section 5 of the declared file) ─────────────────────────────


def rc_path(name):
    return P1B2 / f"rc_{name}.json"


def check_against_filed(ent, p2a_ent, p1a_ent):
    """The declared checks: A1st and A1all equal P2a's A1 on S and SNK, and R0 equals P1a's RRF order, at every N on
    both measures and at both matched sizes."""
    bad = []
    for ours, src, sub, theirs in CHECKS:
        ref = p2a_ent["subsets"][sub]["orders"][theirs] if src == "p2a" else p1a_ent["orders"][theirs]
        a = ent["orders"][ours]
        for n in NS:
            for m in MEASURES:
                if a["curve"][str(n)][m] != ref["curve"][str(n)][m]:
                    bad.append({"order": ours, src: f"{sub or ''} {theirs}".strip(), "N": n, "measure": m,
                                "p1d": a["curve"][str(n)][m], "filed": ref["curve"][str(n)][m]})
        for s in ("at_I_q", "at_P4e"):
            if a[s] != ref[s]:
                bad.append({"order": ours, src: f"{sub or ''} {theirs}".strip(), "size": s, "p1d": a[s], "filed": ref[s]})
    return bad


def best_of(curves, n, m):
    """(the order, its value) leading at N on measure m among curves {order: curve}; ties to the first listed."""
    o = max(curves, key=lambda k: curves[k][str(n)][m])
    return o, curves[o][str(n)][m]


def against(curve, bar):
    """One order's curve against a bar {order: curve}: BELOW where it is more than TIE under the bar's best at some N
    and measure, AT_OR_ABOVE otherwise."""
    below, above = [], []
    for n in NS:
        for m in MEASURES:
            o, best = best_of(bar, n, m)
            ours = curve[str(n)][m]
            row = {"N": n, "measure": m, "rule": ours, "sota": best, "order": o, "gap": round(ours - best, 4)}
            if ours < best - TIE:
                below.append(row)
            elif ours > best + TIE:
                above.append(row)
    return {"call": "BELOW" if below else "AT_OR_ABOVE", "below": below, "above": above,
            "largest_gap": min((b["gap"] for b in below), default=0.0),
            "largest_lead": max((a["gap"] for a in above), default=0.0)}


def p1a_bar(p1a_ent):
    return {o: p1a_ent["orders"][o]["curve"] for o in P.SOTA}


def lfl_bar(ent, p2b_ent, p1a_ent):
    """The like-for-like bar: A1 on st and all (ours, checked against P2a's), LINKED on st and all (P2b's L_st and
    L_all), and the lists (P1a's)."""
    out = {o: ent["orders"][o]["curve"] for o in ("A1st", "A1all")}
    out.update({o: p2b_ent["orders"][o]["curve"] for o in ("L_st", "L_all")})
    out.update({o: p1a_ent["orders"][o]["curve"] for o in ("DENSE", "SPLADE", "RRF")})
    return out


def choose(parts, p1a):
    """Section 5: one rule for all six, on the five training datasets' s1sel carves."""
    rows = {}
    for r in RULES:
        per = {n: against(parts[n]["carves"][f"{n}__s1sel"]["orders"][r]["curve"],
                          p1a_bar(p1a[n]["carves"][f"{n}__s1sel"])) for n in TRAIN5}
        rows[r] = {"below_points": sum(len(c["below"]) for c in per.values()),
                   "largest_gap": min(c["largest_gap"] for c in per.values()),
                   "mean_ALL_at_I_q": round(float(np.mean([parts[n]["carves"][f"{n}__s1sel"]["orders"][r]["at_I_q"]["ALL"]
                                                           for n in TRAIN5])), 4),
                   "per_dataset": {n: {"call": c["call"], "below_points": len(c["below"]),
                                       "largest_gap": c["largest_gap"]} for n, c in per.items()}}
    chosen = min(RULES, key=lambda r: (rows[r]["below_points"], -rows[r]["largest_gap"], -rows[r]["mean_ALL_at_I_q"],
                                       RULES.index(r)))
    return chosen, rows


def reach_n(values, target):
    """The smallest N of the grid whose value is within TIE of target, None beyond NMAX."""
    return next((n for n in NS if values[str(n)] >= target - TIE), None)


def beside(ent, p1a_ent, p2b_ent, rc_ent, chosen):
    """Section 5's lines beside, deciding nothing."""
    bar = p1a_bar(p1a_ent)
    sizes = {}
    for s, st in (("at_I_q", "I_q"), ("at_P4e", "P4e")):
        best_o = max(P.SOTA, key=lambda o: p1a_ent["orders"][o][s]["ALL"])
        sizes[s] = {"set": ent["sets"][st], "rule": ent["orders"][chosen][s], "best_sota": best_o,
                    "sota": p1a_ent["orders"][best_o][s], "MX": rc_ent["orders"]["MX"][s],
                    "RC": rc_ent["orders"]["RC"][s], "MXU": p2b_ent["orders"]["MXU"][s]}
    target = ent["sets"]["I_q"]["ALL"]
    bar_all = {str(n): best_of(bar, n, "ALL")[1] for n in NS}
    return {"rules": {r: {k: v for k, v in against(ent["orders"][r]["curve"], bar).items() if k != "above"}
                      for r in RULES},
            "like_for_like": {r: {k: v for k, v in against(ent["orders"][r]["curve"], lfl_bar(ent, p2b_ent, p1a_ent)).items()
                                  if k != "above"} for r in RULES},
            "at_sizes": sizes,
            "golds_per_node": {"target_set_I_q_ALL": target, "set_I_q_mean_size": ent["sets"]["I_q"]["size"],
                               "rule_reaches_at_N": reach_n({str(n): ent["orders"][chosen]["curve"][str(n)]["ALL"]
                                                             for n in NS}, target),
                               "bar_reaches_at_N": reach_n(bar_all, target)},
            "shares": ent["shares"]}


def verdict_stage():
    parts, p1a, p2a, p2b, rc = {}, {}, {}, {}, {}
    for name in SIX:
        for d, path in ((parts, part_path(name)), (p1a, P.part_path(name)), (p2a, R2.part_path(name)),
                        (p2b, G.part_path(name)), (rc, rc_path(name))):
            if not path.exists():
                raise SystemExit(f"{path} is missing")
            d[name] = json.loads(path.read_text(encoding="utf-8"))
    keys = {(r["script_sha256"], r["freeze_RECORD_SHA256"]) for r in parts.values()}
    if len(keys) != 1:
        raise SystemExit("the parts ran on different scripts or freezes")
    script, freeze = next(iter(keys))
    if any(r.get("limit") for r in parts.values()):
        raise SystemExit("a part ran with --limit")
    if any(r["freeze_RECORD_SHA256"] != freeze for d in (p1a, p2a, p2b, rc) for r in d.values()):
        raise SystemExit("a filed record ran on another freeze")
    checks = {}
    for name in SIX:
        want = {f"{name}__s1eval"} | ({f"{name}__s1sel"} if name in TRAIN5 else set())
        if set(parts[name]["carves"]) != want:
            raise SystemExit(f"{name}: carves {sorted(parts[name]['carves'])}, not {sorted(want)}")
        for c, ent in parts[name]["carves"].items():
            for what, d in (("P1a", p1a), ("P2a", p2a), ("P2b", p2b), ("P1b2", rc)):
                other = d[name]["carves"].get(c)
                if other is None or other["questions"] != ent["questions"]:
                    raise SystemExit(f"{what} has no {c} on these questions")
                if "carve_sha256" in other and other["carve_sha256"] != ent["carve_sha256"]:
                    raise SystemExit(f"{c}: not {what}'s questions")
            bad = check_against_filed(ent, p2a[name]["carves"][c], p1a[name]["carves"][c])
            if bad:
                write_json(OUT / "check_failed.json", {"carve": c, "points": bad})
                raise SystemExit(f"{c}: the sources differ from the filed records at {len(bad)} points "
                                 f"(outputs/p1d/check_failed.json), first {bad[0]}")
            checks[c] = ("A1st and A1all equal P2a's A1 on S and SNK, and R0 equals P1a's RRF order, at every N on "
                         "both measures and at both matched sizes")
    chosen, rows = choose(parts, p1a)
    calls = {n: against(parts[n]["carves"][f"{n}__s1eval"]["orders"][chosen]["curve"],
                        p1a_bar(p1a[n]["carves"][f"{n}__s1eval"])) for n in SIX}
    call = "ALL_SIX" if all(c["call"] == "AT_OR_ABOVE" for c in calls.values()) else "NOT_ALL_SIX"
    side = {c: beside(parts[n]["carves"][c], p1a[n]["carves"][c], p2b[n]["carves"][c], rc[n]["carves"][c], chosen)
            for n in SIX for c in parts[n]["carves"]}
    rec = {"declared_in": "docs/P1D_FREE_POOL.md", "call": call, "chosen": chosen, "choice_s1sel": rows,
           "s1eval": calls, "beside": side, "checks": checks, "verdict_script_sha256": script_sha256(),
           "script_sha256": script, "freeze_RECORD_SHA256": freeze,
           "parts": {n: sha_file(part_path(n)) for n in SIX}, "p1a_parts": {n: sha_file(P.part_path(n)) for n in SIX},
           "p2a_parts": {n: sha_file(R2.part_path(n)) for n in SIX}, "p2b_parts": {n: sha_file(G.part_path(n)) for n in SIX},
           "p1b2_parts": {n: sha_file(rc_path(n)) for n in SIX}, "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(parts, p1a, rec), encoding="utf-8", newline="\n")
    log(f"P1d: chosen {chosen} on s1sel; {call} on s1eval: " + ", ".join(
        f"{n} {c['call']}" + (f" ({len(c['below'])} points, largest gap {c['largest_gap']:+.4f})" if c["below"] else "")
        for n, c in calls.items()))
    return 0


def report_md(parts, p1a, rec):
    show = (25, 100, 500, 1000, 2000, 7000)
    chosen = rec["chosen"]
    L = [f"# P1d: one parameter-free pool rule for all six datasets: {chosen}, {rec['call']}", "",
         "ALL (every gold in the pool) at N, gold recall in brackets, on s1eval. The rules and sources from "
         "outputs/p1d/pools_<dataset>.json; the bar is the best of P1a's SOTA orders at each N and measure.", ""]
    for name in SIX:
        c = f"{name}__s1eval"
        ent, pa, cl, sd = parts[name]["carves"][c], p1a[name]["carves"][c], rec["s1eval"][name], rec["beside"][c]
        L += [f"## {name} ({ent['questions']:,} questions, regime {ent['regime']}): {chosen} {cl['call']}", "",
              "| order | " + " | ".join(f"N = {n:,}" for n in show) + " | at I_q |", "| --- |" + " ---: |" * (len(show) + 1)]
        for o in ORDERS:
            od = ent["orders"][o]
            L.append(f"| {o}{' (chosen)' if o == chosen else ''} | " + " | ".join(
                f"{od['curve'][str(n)]['ALL']:.3f} ({od['curve'][str(n)]['recall']:.3f})" for n in show)
                + f" | {od['at_I_q']['ALL']:.3f} |")
        bar = p1a_bar(pa)
        L.append("| the bar (P1a) | " + " | ".join(
            "{:.3f} ({:.3f}) {}".format(best_of(bar, n, "ALL")[1], best_of(bar, n, "recall")[1], best_of(bar, n, "ALL")[0])
            for n in show) + f" | {sd['at_sizes']['at_I_q']['sota']['ALL']:.3f} |")
        z = sd["at_sizes"]
        L += ["", "- " + "; ".join(
            f"at {lab} ({z[s]['set']['size']:,.0f} rows mean): today's pools {z[s]['set']['ALL']:.3f}, {chosen} "
            f"{z[s]['rule']['ALL']:.3f}, best SOTA {z[s]['best_sota']} {z[s]['sota']['ALL']:.3f}; learned, beside: MX "
            f"{z[s]['MX']['ALL']:.3f}, RC {z[s]['RC']['ALL']:.3f}, MXU {z[s]['MXU']['ALL']:.3f}"
            for s, lab in (("at_I_q", "|I_q|"), ("at_P4e", "|P4e|"))) + ".",
            f"- Golds per node: today's pools' ALL ({sd['golds_per_node']['target_set_I_q_ALL']:.3f} at "
            f"{sd['golds_per_node']['set_I_q_mean_size']:,.0f} rows) is reached by {chosen} at N = "
            f"{sd['golds_per_node']['rule_reaches_at_N'] or 'beyond 7,000'}, by the bar at N = "
            f"{sd['golds_per_node']['bar_reaches_at_N'] or 'beyond 7,000'}.",
            f"- Shares of questions: with a linked node {sd['shares']['linked']:.3f}, with a corroborated one "
            f"{sd['shares']['corroborated']:.3f}, on ROUTE's K branch {sd['shares']['route']:.3f}."]
        if cl["below"]:
            L.append(f"- Below the bar at {len(cl['below'])} points, the largest gap {cl['largest_gap']:+.4f}: " + ", ".join(
                f"N={b['N']} {b['measure']} {b['gap']:+.4f} ({b['order']})" for b in cl["below"][:12])
                + (" ..." if len(cl["below"]) > 12 else "") + ".")
        lf = sd["like_for_like"][chosen]
        L += [f"- Against the like-for-like bar (beside): {lf['call']}"
              + (f", {len(lf['below'])} points, the largest gap {lf['largest_gap']:+.4f}" if lf["below"] else "") + ".", ""]
    L += ["## The choice (s1sel, the five training datasets)", "",
          "| rule | points below the bar | largest gap | mean ALL at I_q | " + " | ".join(TRAIN5) + " |",
          "| --- | ---: | ---: | ---: |" + " --- |" * len(TRAIN5)]
    for r in RULES:
        w = rec["choice_s1sel"][r]
        L.append(f"| {r}{' (chosen)' if r == chosen else ''} | {w['below_points']} | {w['largest_gap']:+.4f} | "
                 f"{w['mean_ALL_at_I_q']:.4f} | " + " | ".join(
                     f"{w['per_dataset'][n]['call']} ({w['per_dataset'][n]['below_points']})" for n in TRAIN5) + " |")
    L += ["", "## Every rule against the bar on s1eval (beside)", "",
          "| rule | " + " | ".join(SIX) + " |", "| --- |" + " --- |" * len(SIX)]
    for r in RULES:
        L.append(f"| {r} | " + " | ".join(
            "{} ({}, {:+.4f})".format(rec["beside"][f"{n}__s1eval"]["rules"][r]["call"],
                                      len(rec["beside"][f"{n}__s1eval"]["rules"][r]["below"]),
                                      rec["beside"][f"{n}__s1eval"]["rules"][r]["largest_gap"]) for n in SIX) + " |")
    L += ["", f"**{rec['call']}**: {chosen} on s1eval, " + ", ".join(f"{n} {c['call']}" for n, c in rec["s1eval"].items())
          + "."]
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _fake_curve(v):
    return {str(n): {"ALL": v, "recall": v, "pool": float(n)} for n in NS}


def _fake_order(v, size=100.0):
    return {"curve": _fake_curve(v), "at_I_q": {"ALL": v, "recall": v, "size": size},
            "at_P4e": {"ALL": v, "recall": v, "size": size}}


def verdict_selftest():
    """The verdict's checks, choice, tie-breaks and call on made-up records, in a temporary folder."""
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="p1d_selftest_"))
    g = globals()
    saved = (g["OUT"], g["P1B2"], g["VERDICT"], g["VERDICT_MD"], P.OUT, R2.OUT, G.OUT)
    try:
        g.update(OUT=tmp / "p1d", P1B2=tmp / "p1b2", VERDICT=tmp / "p1d" / "verdict.json",
                 VERDICT_MD=tmp / "p1d" / "verdict.md")
        P.OUT, R2.OUT, G.OUT = tmp / "p1a", tmp / "p2a", tmp / "p2b"
        for d in ("p1d", "p1b2", "p1a", "p2a", "p2b"):
            (tmp / d).mkdir(parents=True)

        def write_all(sel=None, ev=None, r0_off=None, a1all_off=None, limit=0):
            """sel: {rule: (ALL at |I_q|, {N: ALL there})} on every s1sel carve, every other point at the bar's 0.5;
            ev: {(dataset, rule): {N: ALL there}} on s1eval."""
            sel, ev = sel or {}, ev or {}
            for n in SIX:
                carves = [f"{n}__s1eval"] + ([f"{n}__s1sel"] if n in TRAIN5 else [])
                ours = {"script_sha256": "s", "freeze_RECORD_SHA256": "f", "limit": limit, "carves": {}}
                pa, p2, pb, r = ({"freeze_RECORD_SHA256": "f", "carves": {}} for _ in range(4))
                for c in carves:
                    ent = {"questions": 10, "carve_sha256": "c", "regime": "S",
                           "shares": {"linked": 0.5, "corroborated": 0.4, "route": 0.3},
                           "orders": {o: _fake_order(0.5) for o in ORDERS},
                           "sets": {"I_q": {"ALL": 0.5, "recall": 0.5, "size": 100.0},
                                    "P4e": {"ALL": 0.5, "recall": 0.5, "size": 200.0}}}
                    if c.endswith("s1sel"):
                        for o, (at, pts) in sel.items():
                            ent["orders"][o]["at_I_q"]["ALL"] = at
                            for k, v in pts.items():
                                ent["orders"][o]["curve"][str(k)]["ALL"] = v
                    for (dn, o), pts in ev.items():
                        if dn == n and c.endswith("s1eval"):
                            for k, v in pts.items():
                                ent["orders"][o]["curve"][str(k)]["ALL"] = v
                    ours["carves"][c] = ent
                    same = lambda o: json.loads(json.dumps(ent["orders"][o]))      # noqa: E731  the filed twin
                    po = {o: _fake_order(0.5) for o in ("A3", "A0", "ORACLE") + P.SOTA}
                    po["RRF"] = same("R0")
                    if n == r0_off and c.endswith("s1eval"):
                        po["RRF"]["curve"]["100"]["recall"] = 0.51
                    pa["carves"][c] = {"questions": 10, "orders": po}
                    subs = {s: {"orders": {"A1": same(o), "A3": _fake_order(0.5)}} for s, o in (("S", "A1st"),
                                                                                                ("SNK", "A1all"))}
                    if n == a1all_off and c.endswith("s1sel"):
                        subs["SNK"]["orders"]["A1"]["at_P4e"]["ALL"] = 0.49
                    p2["carves"][c] = {"questions": 10, "subsets": subs}
                    pb["carves"][c] = {"questions": 10, "carve_sha256": "c",
                                       "orders": {o: _fake_order(0.4) for o in ("MXU", "L_st", "L_all")}}
                    r["carves"][c] = {"questions": 10, "carve_sha256": "c",
                                      "orders": {o: _fake_order(0.45) for o in ("RC", "MX")}}
                write_json(part_path(n), ours)
                write_json(P.part_path(n), pa)
                write_json(R2.part_path(n), p2)
                write_json(G.part_path(n), pb)
                write_json(rc_path(n), r)

        def run():
            assert verdict_stage() == 0
            return json.loads(VERDICT.read_text(encoding="utf-8"))

        # every rule at the bar: no point below anywhere, the first rule in RULES' order, ALL_SIX
        write_all()
        v = run()
        assert v["chosen"] == "A1st" and v["call"] == "ALL_SIX", (v["chosen"], v["call"])
        assert VERDICT_MD.read_text(encoding="utf-8").startswith("# P1d: one parameter-free pool rule")
        assert v["beside"]["metaqa__s1eval"]["golds_per_node"]["rule_reaches_at_N"] == NS[0]
        # the higher mean ALL at I_q breaks a tie in points and gaps
        write_all({"ROUTEst": (0.52, {})})
        assert run()["chosen"] == "ROUTEst"
        # fewer points below beats a smaller largest gap and a higher mean
        two = {25: 0.497, 100: 0.497}
        write_all({"A1st": (0.6, two), "A1both": (0.5, {25: 0.49}), "ROUTEst": (0.5, two), "ROUTEboth": (0.5, two),
                   "SUM5": (0.5, two), "RRF5": (0.5, two)})
        v = run()
        assert v["chosen"] == "A1both" and v["choice_s1sel"]["A1both"]["below_points"] == 5, v["choice_s1sel"]["A1both"]
        assert v["choice_s1sel"]["A1st"]["below_points"] == 10 and v["choice_s1sel"]["A1st"]["largest_gap"] == -0.003
        # on equal points, the smaller largest gap beats a higher mean
        write_all({"A1st": (0.6, {25: 0.49}), "A1both": (0.5, {25: 0.49}), "ROUTEst": (0.5, {25: 0.495}),
                   "ROUTEboth": (0.5, {25: 0.49}), "SUM5": (0.5, {25: 0.49}), "RRF5": (0.5, {25: 0.49})})
        assert run()["chosen"] == "ROUTEst"
        # the call on s1eval: at the tie is AT_OR_ABOVE, just past it is BELOW, named with its point and the leader
        write_all(ev={("musique", "A1st"): {500: 0.4985}})
        v = run()
        assert v["call"] == "ALL_SIX" and v["s1eval"]["musique"]["call"] == "AT_OR_ABOVE"
        write_all(ev={("musique", "A1st"): {500: 0.4975}})
        v = run()
        assert v["call"] == "NOT_ALL_SIX" and v["s1eval"]["musique"]["call"] == "BELOW"
        b = v["s1eval"]["musique"]["below"]
        assert len(b) == 1 and b[0]["N"] == 500 and b[0]["measure"] == "ALL" and b[0]["order"] == P.SOTA[0], b
        # the declared checks refuse records that differ from the filed ones
        for kw, word in (({"r0_off": "webqsp"}, "webqsp__s1eval"), ({"a1all_off": "2wiki"}, "2wiki__s1sel")):
            write_all(**kw)
            try:
                verdict_stage()
                raise AssertionError(f"a source that differs from the filed records was not refused ({kw})")
            except SystemExit as e:
                assert word in str(e) and "differ from the filed records" in str(e), e
        write_all(limit=64)
        try:
            verdict_stage()
            raise AssertionError("a part run with --limit was not refused")
        except SystemExit as e:
            assert "--limit" in str(e), e
    finally:
        (g["OUT"], g["P1B2"], g["VERDICT"], g["VERDICT_MD"], P.OUT, R2.OUT, G.OUT) = saved
        shutil.rmtree(tmp, ignore_errors=True)


def selftest():
    from types import SimpleNamespace
    EMPTY = B.EMPTY
    # CombSUM: a lone source keeps its order, empty sources are skipped, ties go by node position
    s1 = B.by_mass(np.asarray([7, 3, 9, 1]), np.asarray([0.4, 0.3, 0.2, 0.1]))
    s2 = B.by_mass(np.asarray([3, 7]), np.asarray([0.5, 0.5]))
    assert csum([s1]).tolist() == [7, 3, 9, 1] and csum([EMPTY, s1, EMPTY]).tolist() == [7, 3, 9, 1]
    assert csum([EMPTY]).size == 0 and csum([]).size == 0
    assert csum([s1, s2]).tolist() == [7, 3, 9, 1]                 # 0.9 and 0.8, then 0.2 and 0.1
    assert csum([B.uniform([5, 2]), B.uniform([8])]).tolist() == [8, 2, 5]
    # RRF: a lone order keeps itself, empty orders are skipped, ties go by node position
    assert rrf([np.asarray([4, 1, 6])], 60).tolist() == [4, 1, 6]
    assert rrf([np.empty(0, np.int64), np.asarray([4, 1, 6]), np.empty(0, np.int64)], 60).tolist() == [4, 1, 6]
    assert rrf([np.asarray([5, 2]), np.asarray([2, 5])], 60).tolist() == [2, 5]
    assert rrf([np.asarray([5, 2, 9]), np.asarray([9])], 60).tolist() == [9, 5, 2]
    assert rrf([], 60).size == 0 and rrf([np.asarray([3, 1])], 60, nmax=1).tolist() == [3]
    # the backfill: the head first, then the rest's other nodes in its order, cut at nmax
    assert backfill([4, 9], [9, 1, 4, 7]).tolist() == [4, 9, 1, 7]
    assert backfill([], [9, 1]).tolist() == [9, 1] and backfill([4, 9, 2], [1], nmax=2).tolist() == [4, 9]
    # the corroborated links and ROUTE's branches
    prep = SimpleNamespace(dense_ids=np.asarray([[2, 5, 8] + list(range(10, 200))]),
                           splade_ids=np.asarray([[3, 5] + list(range(300, 500))]))
    for linked, ls, route in (([3, 2], [2, 3], True), ([2, 400], [2], False), ([400, 401], [], False), ([], [], False),
                              ([250], [], False)):
        L, Ls = anchors(prep, 0, np.asarray(linked, np.int64))
        assert Ls.tolist() == ls and routes(L, Ls) is route, (linked, Ls, route)
    # the sources on toy questions: the walks are step 4e's, on each union and on the regime's own union
    n = 30
    edges = {"structural": [(i, i + 1) for i in range(12)] + [(0, 20), (20, 21)],
             "ner": [(1, 13), (13, 14), (5, 15), (2, 22)],
             "knn": [(15, 16), (3, 17), (17, 18), (25, 26), (22, 23)]}
    stores = {f: B._Store(n, e) for f, e in edges.items()}
    unions = {u: S4B.RegimeUnion(stores, list(fams), n) for u, fams in UNIONS}
    rng = np.random.default_rng(0)
    nq = 4
    dense = np.stack([rng.permutation(n - 1)[:12] for _ in range(nq)]).astype(np.int64)      # node 29: in no list
    splade = np.stack([rng.permutation(n - 1)[:12] for _ in range(nq)]).astype(np.int64)
    prep = SimpleNamespace(seeds=[np.union1d(dense[q, :5], splade[q, :5]) for q in range(nq)], dense_ids=dense,
                           splade_ids=splade)
    linked = [dense[0, :2].copy(), np.empty(0, np.int64), np.asarray([dense[2, 0], 29]), np.asarray([29])]
    js = range(nq)
    srcs = sources_batch(prep, js, linked, unions, 60, 2)
    for regime, un in (("S", unions["st"]), ("SNK", unions["all"])):
        dev_check("toy", prep, js, linked, unions, un, regime, 60, 2, srcs)
    assert [routes(*s[2]) for s in srcs] == [True, False, False, False]
    assert srcs[1][0][2][0].size == 0 and srcs[3][0][3][0].size == 0 and srcs[2][0][2][0].size > 0
    for s, _c, _a in srcs:
        for nodes, mass in s:
            assert np.unique(nodes).size == nodes.size and (mass.size == 0 or abs(mass.sum() - 1) < 1e-9)
    assert not np.array_equal(srcs[0][0][0][0], srcs[0][0][1][0])
    # the rules: A1st is its source, ROUTE takes K's head only on the route, every order is unique and cut at NMAX
    for (s, _c, (L, Ls)) in srcs:
        o = rule_orders(s, routes(L, Ls), 60)
        assert set(o) == set(ORDERS) and all(np.unique(v).size == v.size <= NMAX for v in o.values())
        assert o["A1st"].tolist() == s[0][0].tolist() and o["A1both"].tolist() == csum([s[0], s[1]]).tolist()
        if routes(L, Ls):
            k = o["Kst"].size
            assert o["ROUTEst"][:k].tolist() == o["Kst"].tolist() and set(o["ROUTEst"].tolist()) >= set(o["A1st"].tolist())
        else:
            assert o["ROUTEst"].tolist() == o["A1st"].tolist() and o["ROUTEboth"].tolist() == o["A1both"].tolist()
        assert set(o["SUM5"].tolist()) == set().union(*[set(x[0].tolist()) for x in s])
        assert o["RRF5"].tolist() == rrf([o[x] for x in SOURCES], 60).tolist()
    verdict_selftest()
    print("selftest: ok (CombSUM, RRF and the backfill on toys, the corroborated links and ROUTE's branches, the sources "
          "equal step 4e's walks on both unions and the regime's own, the rules, the verdict's checks, choice, "
          "tie-breaks and call)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("pools", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="a development run on the first Q questions; writes nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "pools":
        if a.dataset not in SIX:
            raise SystemExit(f"dataset must be one of {SIX}")
        if a.host:
            G.on_the_mirror()
        return pools_stage(a.dataset, a.threads, a.limit)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
