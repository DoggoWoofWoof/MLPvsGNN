"""P1e: what the links landed on decides how they are used (docs/P1E_ANCHOR_KIND.md). Pools only: no model trains or
reads, and nothing is fit.

    python scripts/p1e_anchor_kind.py pools DATASET [--threads 5] [--host] [--limit Q]   s1sel (training) and s1eval
    python scripts/p1e_anchor_kind.py verdict
    python scripts/p1e_anchor_kind.py --selftest

P1d's five sources, imported unchanged (A1st, A1all, Kst, Kall, R0), and five more:
    KTst, KTall     P1d's K restart (uniform over L*) walked with the first stage's steer: within a family, a node's
                    mass goes to its neighbours in proportion to w(v) = 1/(c + rank_dense(v)) + 1/(c + rank_splade(v)),
                    a list's rank taken as its depth + 1 where it does not hold v; with every w equal it is step 4e's
                    walk, bit for bit
    A1Tst, A1Tall   P1d's A1 restart (HippoRAG 2's) walked with the same steer
    KCUTst          K(st)'s nodes whose mass is at least 1/n, n the graph's nodes: what the walk holds above the level a
                    walker with no information gives every node
A question's kind: an entity question when the nodes it starts from are all name-only (an entity record, whose text is
one of its names: its title, display name or disambiguated display name): its corroborated anchors L*, or its seeds
(the first stage's top five) when L* is empty. A passage question otherwise. ROUTE's condition is P1d's (linked nodes,
and L* holds every one).
    ROUTEcut        on the route, KCUTst then A1(st)'s other nodes in A1(st)'s order; A1(st) otherwise
    RRF5T           RRF over A1st, A1all, KTst, KTall and R0 (P1d's RRF5 with K steered)
    RRF7T           RRF over A1st, A1all, A1Tst, A1Tall, KTst, KTall and R0 (RRF5T with the steered A1 walks added)
The rules (RULES), fixed before any number:
    RRF5            P1d's chosen rule, on every question
    KINDcut         entity: ROUTEcut; passage: RRF5
    KIND_T          entity: ROUTEst (P1d's, K(st) uncut); passage: RRF5T
    KINDcut_T       entity: ROUTEcut; passage: RRF5T
    KINDcut_7T      entity: ROUTEcut; passage: RRF7T
The checks: P1d's ten orders, recomputed here, put every gold at the position P1d filed, question for question, on
every carve; the steered walk with every weight equal is step 4e's walk on K's and A1's restarts and both unions
(development runs). One rule is chosen on the five training datasets' s1sel carves by P1d's order (the fewest points
more than TIE below P1a's bar, then the smaller largest gap, then the higher mean ALL at |I_q|, then RULES' order); the
call is on s1eval, against P1a's bar.
--limit Q (pools): a development run on the first Q questions of each carve. It writes nothing and logs no number,
only that every step ran and that the checks held.
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
from numba import njit, prange  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import p1d_free_pool as D  # noqa: E402

G, B, R2, P, E, S4, S4B = D.G, D.B, D.R2, D.P, D.E, D.S4, D.S4B
OUT = ROOT / "outputs" / "p1e"
VERDICT = OUT / "verdict.json"
VERDICT_MD = OUT / "verdict.md"
NS, NMAX, TIE, SIX, TRAIN5, MEASURES = D.NS, D.NMAX, D.TIE, D.SIX, D.TRAIN5, D.MEASURES
UNIONS = D.UNIONS
RULES = ("RRF5", "KINDcut", "KIND_T", "KINDcut_T", "KINDcut_7T")
NEW = ("KTst", "KTall", "A1Tst", "A1Tall", "KCUTst", "ROUTEcut", "RRF5T", "RRF7T")    # recorded beside the rules
STEERED = ("KTst", "KTall", "A1Tst", "A1Tall")
ORDERS = RULES + NEW
P1D_ORDERS = D.ORDERS                                       # P1d's ten orders, checked against its filed positions
assert len(set(ORDERS)) == len(ORDERS) == 13 and not set(NEW) & set(P1D_ORDERS) and set(STEERED) <= set(NEW)
log, utc, write_json, sha_file = D.log, D.utc, D.write_json, D.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def part_path(name):
    return OUT / f"pools_{name}.json"


def p1d_pos_path(name, carve):
    return D.OUT / f"pos_{name}__{carve}.npz"


# ── the steered walk (KT) ────────────────────────────────────────────────────


@njit(inline="always")
def _tspread(ip, cc, u, m, nf, floor, ex, r, flag, touched, queue, deg, eps, n, nt, tail):
    """u's mass m over its neighbours in one family: 1/nf of it, in proportion to w = floor + ex within the family."""
    a = ip[u]
    b = ip[u + 1]
    if b <= a:
        return nt, tail
    tot = 0.0
    for k in range(a, b):
        tot += floor + ex[cc[k]]
    base = m / (nf * tot)
    for k in range(a, b):
        v = cc[k]
        if flag[v] & 2 == 0:
            flag[v] |= 2
            touched[nt] = v
            nt += 1
        r[v] += base * (floor + ex[v])
        if flag[v] & 1 == 0 and r[v] >= eps * max(deg[v], 1):
            flag[v] |= 1
            queue[tail % n] = v
            tail += 1
    return nt, tail


@njit
def _tpush(seeds, wts, ip0, c0, ip1, c1, ip2, c2, deg, floor, ex, alpha, eps, p, r, flag, touched, queue):
    """step 4e's _wpush with nothing excluded, spreading by _tspread."""
    n = deg.size
    nt = 0
    tail = 0
    for k in range(seeds.size):
        s = seeds[k]
        if flag[s] & 2 == 0:
            flag[s] |= 2
            touched[nt] = s
            nt += 1
        r[s] += wts[k]
    for k in range(seeds.size):
        s = seeds[k]
        if flag[s] & 1 == 0:
            flag[s] |= 1
            queue[tail % n] = s
            tail += 1
    head = 0
    while head < tail:
        u = queue[head % n]
        head += 1
        flag[u] &= 254
        du = deg[u]
        ru = r[u]
        if ru < eps * max(du, 1):
            continue
        r[u] = 0.0
        p[u] += alpha * ru
        if du == 0:
            continue
        m = (1.0 - alpha) * ru
        nf = 0
        if ip0[u + 1] > ip0[u]:
            nf += 1
        if ip1[u + 1] > ip1[u]:
            nf += 1
        if ip2[u + 1] > ip2[u]:
            nf += 1
        nt, tail = _tspread(ip0, c0, u, m, nf, floor, ex, r, flag, touched, queue, deg, eps, n, nt, tail)
        nt, tail = _tspread(ip1, c1, u, m, nf, floor, ex, r, flag, touched, queue, deg, eps, n, nt, tail)
        nt, tail = _tspread(ip2, c2, u, m, nf, floor, ex, r, flag, touched, queue, deg, eps, n, nt, tail)
    return nt


@njit(parallel=True)
def tppr_batch(seeds_ptr, seeds_flat, w_flat, st_ptr, st_nodes, st_ex, floors, ip0, c0, ip1, c1, ip2, c2, deg, alpha,
               eps, bmax, n_chunks, out_nodes, out_count):
    nq = seeds_ptr.size - 1
    n = deg.size
    for t in prange(n_chunks):
        lo = (nq * t) // n_chunks
        hi = (nq * (t + 1)) // n_chunks
        if lo < hi:
            p = np.zeros(n)
            r = np.zeros(n)
            flag = np.zeros(n, dtype=np.uint8)
            touched = np.empty(n, dtype=np.int32)
            queue = np.empty(n, dtype=np.int32)
            ex = np.zeros(n)
            for q in range(lo, hi):
                seeds = seeds_flat[seeds_ptr[q]:seeds_ptr[q + 1]]
                if seeds.size == 0:
                    out_count[q] = 0
                    continue
                for k in range(st_ptr[q], st_ptr[q + 1]):
                    ex[st_nodes[k]] = st_ex[k]
                nt = _tpush(seeds, w_flat[seeds_ptr[q]:seeds_ptr[q + 1]], ip0, c0, ip1, c1, ip2, c2, deg, floors[q], ex,
                            alpha, eps, p, r, flag, touched, queue)
                _avail, cnt = S4._top(p, flag, touched, nt, bmax, out_nodes[q])
                out_count[q] = cnt
                for k in range(nt):
                    v = touched[k]
                    p[v] = 0.0
                    r[v] = 0.0
                    flag[v] = 0
                for k in range(st_ptr[q], st_ptr[q + 1]):
                    ex[st_nodes[k]] = 0.0


def steer(prep, j, c):
    """(the lists' nodes, w - floor for each, floor): w(v) = 1/(c + rank_dense) + 1/(c + rank_splade), a list's rank
    its depth + 1 where it does not hold v, so a node in neither list has w = floor."""
    d = np.asarray(prep.dense_ids[j], np.int64)
    s = np.asarray(prep.splade_ids[j], np.int64)
    fd, fs = 1.0 / (c + d.size + 1), 1.0 / (c + s.size + 1)
    u = np.union1d(d, s)
    ex = np.zeros(u.size)
    ex[np.searchsorted(u, d)] += 1.0 / (c + np.arange(1, d.size + 1)) - fd
    ex[np.searchsorted(u, s)] += 1.0 / (c + np.arange(1, s.size + 1)) - fs
    return u, ex, fd + fs


def twalk(union, restarts, steers, threads):
    """[nodes by mass descending then position] per restart, the walk steered by steers [(nodes, ex, floor)]."""
    nq = len(restarts)
    if nq == 0:
        return []
    s_ptr, s_flat = S4.flat([r[0] for r in restarts])
    w_flat = np.concatenate([r[1] for r in restarts]).astype(np.float64)
    t_ptr, t_nodes = S4.flat([st[0] for st in steers])
    t_ex = np.concatenate([st[1] for st in steers]).astype(np.float64)
    floors = np.asarray([st[2] for st in steers], np.float64)
    out = np.full((nq, NMAX), -1, dtype=np.int64)
    cnt = np.zeros(nq, dtype=np.int64)
    tppr_batch(s_ptr, s_flat, w_flat, t_ptr, t_nodes, t_ex, floors, *union.args, float(E.ALPHA), float(S4.EPS),
               int(NMAX), max(1, min(nq, 4 * threads)), out, cnt)
    return [out[q, :cnt[q]].copy() for q in range(nq)]


def unit_steers(prep, js):
    """Every weight 1: the check that the steered walk is step 4e's walk."""
    return [(np.empty(0, np.int64), np.empty(0, np.float64), 1.0) for _j in js]


def restarts_batch(prep, js, linked, unions, srcs, constant):
    """Per question of js: (K's restart, uniform over L*; A1's restart, HippoRAG 2's, as P1d walks it)."""
    k0 = [B.uniform(a[1]) for _s, _n, a in srcs]
    a1 = [D.a1_restart(prep, j, linked[j], unions["st"].deg, constant) for j in js]
    return k0, a1


def steered_batch(prep, js, linked, unions, srcs, constant, threads):
    """Per question of js: {KTst, KTall, A1Tst, A1Tall}, P1d's K and A1 restarts walked with the first stage's steer."""
    k0, a1 = restarts_batch(prep, js, linked, unions, srcs, constant)
    stz = [steer(prep, j, constant) for j in js]
    w = {}
    for u, _fams in UNIONS:
        w[f"KT{u}"] = twalk(unions[u], k0, stz, threads)
        w[f"A1T{u}"] = twalk(unions[u], a1, stz, threads)
    return [{k: v[i] for k, v in w.items()} for i in range(len(js))]


# ── the question's kind, the cut and the rules ───────────────────────────────


def name_only(records, n):
    """Per node: True when its text is one of its names (its title, display name or disambiguated display name): an
    entity record, whose text says nothing beyond its name. A node without text or names is not."""
    out = np.zeros(int(n), dtype=bool)
    for k, rec in enumerate(records):
        tx = str(rec.get("text") or "").strip()
        names = {str(rec.get(f) or "").strip() for f in ("title", "display_name", "display_name_disambiguated")}
        out[k] = bool(tx) and tx in names
    return out


def kind(Ls, seeds, nonly):
    """'entity' when the nodes the question starts from are all name-only: L*, or the seeds when L* is empty."""
    start = np.asarray(Ls, np.int64) if np.asarray(Ls).size else np.asarray(seeds, np.int64)
    return "entity" if start.size and bool(nonly[start].all()) else "passage"


def kcut(kst, n):
    """K(st)'s nodes whose (scaled) mass is at least 1/n."""
    nodes, mass = kst
    return nodes[mass >= 1.0 / n] if nodes.size else nodes


def p1e_orders(src, st, route, knd, n_st, constant):
    """Every order of ORDERS for one question, and P1d's ten, from P1d's five sources, the four steered walks
    (st = {KTst, KTall, A1Tst, A1Tall}: orders), ROUTE's condition, the question's kind and the st graph's node count."""
    o = D.rule_orders(src, route, constant)
    p1d = {k: o[k] for k in P1D_ORDERS}
    a1st = o["A1st"]
    out = {k: np.asarray(st[k], np.int64)[:NMAX] for k in STEERED}
    out["KCUTst"] = kcut(src[2], n_st)
    out["ROUTEcut"] = D.backfill(out["KCUTst"], a1st) if route else a1st
    out["RRF5"] = o["RRF5"]
    out["RRF5T"] = D.rrf([o["A1st"], o["A1all"], out["KTst"], out["KTall"], o["R0"]], constant)
    out["RRF7T"] = D.rrf([o["A1st"], o["A1all"], out["A1Tst"], out["A1Tall"], out["KTst"], out["KTall"], o["R0"]],
                         constant)
    ent = knd == "entity"
    out["KINDcut"] = out["ROUTEcut"] if ent else out["RRF5"]
    out["KIND_T"] = o["ROUTEst"] if ent else out["RRF5T"]
    out["KINDcut_T"] = out["ROUTEcut"] if ent else out["RRF5T"]
    out["KINDcut_7T"] = out["ROUTEcut"] if ent else out["RRF7T"]
    return {k: out[k] for k in ORDERS}, p1d


# ── the stage (s1sel, s1eval) ────────────────────────────────────────────────


def dev_check(name, prep, js, linked, unions, constant, threads, srcs):
    """A development run's check: the steered walk with every weight 1 equals step 4e's walk on K's and A1's restarts,
    on both unions."""
    none = np.empty(0, np.int64)
    k0, a1 = restarts_batch(prep, js, linked, unions, srcs, constant)
    for what, rs in (("K", k0), ("A1", a1)):
        sel = [i for i, r in enumerate(rs) if r[0].size]
        for u, _fams in UNIONS:
            got = twalk(unions[u], [rs[i] for i in sel], unit_steers(prep, sel), threads)
            want = E.walk(unions[u], [rs[i] for i in sel], [none] * len(sel), NMAX, threads) if sel else []
            bad = sum(not np.array_equal(g, w) for g, w in zip(got, want))
            if bad:
                raise SystemExit(f"{name}: {what}T{u} with every weight 1 differs from step 4e's walk on {bad} questions")


def check_p1d_positions(name, carve, pos_p1d, nq):
    """P1d's ten orders put every gold where P1d filed it, question for question."""
    path = p1d_pos_path(name, carve)
    if not path.exists():
        raise SystemExit(f"{path} is missing")
    z = np.load(path)
    ptr = z["ptr"]
    for o in P1D_ORDERS:
        got = np.concatenate(pos_p1d[o]).astype(np.int64) if pos_p1d[o] else np.empty(0, np.int64)
        want = z[o].astype(np.int64)[:ptr[nq]]
        if got.size != want.size or not np.array_equal(got, want):
            raise SystemExit(f"{name}/{carve}: P1d's {o} puts the golds elsewhere than P1d's filed positions")


def pools_stage(name, threads, limit=0):
    op = S4.Opened()
    ds = op.canonical.Dataset(name, root=str(op.served))
    nonly = name_only(ds.nodes(), ds.n_nodes)
    del ds
    rec = {"declared_in": "docs/P1E_ANCHOR_KIND.md", "dataset": name, "orders": list(ORDERS), "ns": list(NS),
           "alpha": E.ALPHA, "eps": S4.EPS, "head": D.HEAD, "script_sha256": script_sha256(),
           "p1d_script_sha256": D.script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
           "limit": limit, "name_only_nodes": int(nonly.sum()), "nodes": int(nonly.size), "carves": {}}
    cache = {}

    def sink(name, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        regime = G.regime_of(families, construction)
        _stores, unions = G.open_unions(cache, op, name, union)
        n_st = int(unions["st"].deg.size)
        nq = min(limit, len(U)) if limit else len(U)
        golds = [np.unique(np.asarray(g, dtype=np.int64)) for g in pop.golds[:nq]]
        p4e = P.p4e_pools(name, carve, pop)
        pos = {o: [] for o in ORDERS}
        lens = {o: [] for o in ORDERS}
        pos_p1d = {o: [] for o in P1D_ORDERS}
        counts = {"entity": 0, "passage": 0, "entity_route": 0, "route": 0}
        cut_len = []
        for lo in range(0, nq, S4.BATCH):
            js = list(range(lo, min(nq, lo + S4.BATCH)))
            srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
            if limit:
                D.dev_check(name, prep, js, linked, unions, union, regime, constant, threads, srcs)
                dev_check(name, prep, js, linked, unions, constant, threads, srcs)
            steered = steered_batch(prep, js, linked, unions, srcs, constant, threads)
            for i, j in enumerate(js):
                src, _nn, (L, Ls) = srcs[i]
                route = D.routes(L, Ls)
                knd = kind(Ls, prep.seeds[j], nonly)
                counts[knd] += 1
                counts["route"] += route
                counts["entity_route"] += route and knd == "entity"
                orders, p1d = p1e_orders(src, steered[i], route, knd, n_st, constant)
                cut_len.append(orders["KCUTst"].size)
                for o in ORDERS:
                    pos[o].append(P.positions(orders[o], golds[j]))
                    lens[o].append(int(orders[o].size))
                for o in P1D_ORDERS:
                    pos_p1d[o].append(P.positions(p1d[o], golds[j]))
        check_p1d_positions(name, carve, pos_p1d, nq)
        if limit:
            log(f"{name}/{carve}: development run on {nq} questions ({time.time() - t:.0f}s): every step ran; P1d's "
                "orders put every gold at P1d's filed positions; P1d's sources equal step 4e's walks; the steered walk "
                "with every weight 1 equals step 4e's walk on K's and A1's restarts on both unions; nothing written")
            return
        sizes = {"I_q": np.asarray([np.asarray(inc[j]).size for j in range(nq)]),
                 "P4e": np.asarray([p4e[j].size for j in range(nq)])}
        ent = {"questions": nq, "carve_sha256": pop.digest, "regime": regime,
               "shares": {k: round(v / max(1, nq), 4) for k, v in counts.items()},
               "kcut_len_mean": round(float(np.mean(cut_len)) if cut_len else 0.0, 1),
               "orders": {o: {"curve": P.curve(pos[o], lens[o]), "len_mean": round(float(np.mean(lens[o])), 1),
                              "short": int(sum(x < NMAX for x in lens[o])),
                              "at_I_q": P.at_sizes(pos[o], sizes["I_q"]), "at_P4e": P.at_sizes(pos[o], sizes["P4e"])}
                          for o in ORDERS},
               "sets": {"I_q": P.set_cov([np.asarray(inc[j], np.int64) for j in range(nq)], golds),
                        "P4e": P.set_cov(p4e[:nq], golds)},
               "checks": "P1d's ten orders put every gold at P1d's filed positions, question for question",
               "seconds": round(time.time() - t, 1)}
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
        log(f"{name}/{carve} ({nq} q, {ent['seconds']:.0f}s): written; P1d's positions held; shares {ent['shares']}")

    wanted = {"s1eval"} | ({"s1sel"} if name in TRAIN5 else set())
    E.run_carves(op, name, wanted, ("A3",), threads, sink)
    return 0


# ── the verdict (section 5 of the declared file) ─────────────────────────────


def choose(parts, p1a):
    """One rule for all six, on the five training datasets' s1sel carves, by P1d's order."""
    rows = {}
    for r in RULES:
        per = {n: D.against(parts[n]["carves"][f"{n}__s1sel"]["orders"][r]["curve"],
                            D.p1a_bar(p1a[n]["carves"][f"{n}__s1sel"])) for n in TRAIN5}
        rows[r] = {"below_points": sum(len(c["below"]) for c in per.values()),
                   "largest_gap": min(c["largest_gap"] for c in per.values()),
                   "mean_ALL_at_I_q": round(float(np.mean([parts[n]["carves"][f"{n}__s1sel"]["orders"][r]["at_I_q"]["ALL"]
                                                           for n in TRAIN5])), 4),
                   "per_dataset": {n: {"call": c["call"], "below_points": len(c["below"]),
                                       "largest_gap": c["largest_gap"]} for n, c in per.items()}}
    chosen = min(RULES, key=lambda r: (rows[r]["below_points"], -rows[r]["largest_gap"], -rows[r]["mean_ALL_at_I_q"],
                                       RULES.index(r)))
    return chosen, rows


def beside(ent, p1d_ent, p1a_ent, p2b_ent, rc_ent, chosen):
    """Section 5's lines beside, deciding nothing. The like-for-like bar is P1d's (A1 and LINKED on st and on all, and
    the lists), with A1st and A1all from P1d's filed part."""
    bar = D.p1a_bar(p1a_ent)
    lfl = D.lfl_bar(p1d_ent, p2b_ent, p1a_ent)
    sizes = {}
    for s, st in (("at_I_q", "I_q"), ("at_P4e", "P4e")):
        best_o = max(P.SOTA, key=lambda o: p1a_ent["orders"][o][s]["ALL"])
        sizes[s] = {"set": ent["sets"][st], "rule": ent["orders"][chosen][s], "best_sota": best_o,
                    "sota": p1a_ent["orders"][best_o][s], "MX": rc_ent["orders"]["MX"][s],
                    "RC": rc_ent["orders"]["RC"][s], "MXU": p2b_ent["orders"]["MXU"][s]}
    target = ent["sets"]["I_q"]["ALL"]
    bar_all = {str(n): D.best_of(bar, n, "ALL")[1] for n in NS}
    return {"rules": {r: {k: v for k, v in D.against(ent["orders"][r]["curve"], bar).items() if k != "above"}
                      for r in RULES},
            "like_for_like": {r: {k: v for k, v in D.against(ent["orders"][r]["curve"], lfl).items() if k != "above"}
                              for r in RULES},
            "at_sizes": sizes,
            "golds_per_node": {"target_set_I_q_ALL": target, "set_I_q_mean_size": ent["sets"]["I_q"]["size"],
                               "rule_reaches_at_N": D.reach_n({str(n): ent["orders"][chosen]["curve"][str(n)]["ALL"]
                                                               for n in NS}, target),
                               "bar_reaches_at_N": D.reach_n(bar_all, target),
                               "rules_reach_at_N": {r: D.reach_n({str(n): ent["orders"][r]["curve"][str(n)]["ALL"]
                                                                  for n in NS}, target) for r in RULES}},
            "shares": ent["shares"]}


def verdict_stage():
    parts, p1d, p1a, p2b, rc = {}, {}, {}, {}, {}
    for name in SIX:
        for d, path in ((parts, part_path(name)), (p1d, D.part_path(name)), (p1a, P.part_path(name)),
                        (p2b, G.part_path(name)), (rc, D.rc_path(name))):
            if not path.exists():
                raise SystemExit(f"{path} is missing")
            d[name] = json.loads(path.read_text(encoding="utf-8"))
    keys = {(r["script_sha256"], r["freeze_RECORD_SHA256"]) for r in parts.values()}
    if len(keys) != 1:
        raise SystemExit("the parts ran on different scripts or freezes")
    script, freeze = next(iter(keys))
    if any(r.get("limit") for r in parts.values()):
        raise SystemExit("a part ran with --limit")
    if any(r["freeze_RECORD_SHA256"] != freeze for d in (p1d, p1a, p2b, rc) for r in d.values()):
        raise SystemExit("a filed record ran on another freeze")
    if {r["p1d_script_sha256"] for r in parts.values()} != {r["script_sha256"] for r in p1d.values()}:
        raise SystemExit("the parts imported another P1d script than the one P1d's filed parts ran")
    for name in SIX:
        want = {f"{name}__s1eval"} | ({f"{name}__s1sel"} if name in TRAIN5 else set())
        if set(parts[name]["carves"]) != want:
            raise SystemExit(f"{name}: carves {sorted(parts[name]['carves'])}, not {sorted(want)}")
        for c, ent in parts[name]["carves"].items():
            for what, d in (("P1d", p1d), ("P1a", p1a), ("P2b", p2b), ("P1b2", rc)):
                other = d[name]["carves"].get(c)
                if other is None or other["questions"] != ent["questions"]:
                    raise SystemExit(f"{what} has no {c} on these questions")
                if "carve_sha256" in other and other["carve_sha256"] != ent["carve_sha256"]:
                    raise SystemExit(f"{c}: not {what}'s questions")
            if ent["orders"]["RRF5"] != p1d[name]["carves"][c]["orders"]["RRF5"]:
                raise SystemExit(f"{c}: RRF5 differs from P1d's filed RRF5")
    chosen, rows = choose(parts, p1a)
    calls = {n: D.against(parts[n]["carves"][f"{n}__s1eval"]["orders"][chosen]["curve"],
                          D.p1a_bar(p1a[n]["carves"][f"{n}__s1eval"])) for n in SIX}
    call = "ALL_SIX" if all(c["call"] == "AT_OR_ABOVE" for c in calls.values()) else "NOT_ALL_SIX"
    side = {c: beside(parts[n]["carves"][c], p1d[n]["carves"][c], p1a[n]["carves"][c], p2b[n]["carves"][c],
                      rc[n]["carves"][c], chosen)
            for n in SIX for c in parts[n]["carves"]}
    rec = {"declared_in": "docs/P1E_ANCHOR_KIND.md", "call": call, "chosen": chosen, "choice_s1sel": rows,
           "s1eval": calls, "beside": side,
           "checks": "every part's P1d orders put every gold at P1d's filed positions (checked in the pools stage); "
                     "RRF5's curves and matched sizes equal P1d's filed RRF5 on every carve",
           "verdict_script_sha256": script_sha256(), "script_sha256": script, "freeze_RECORD_SHA256": freeze,
           "parts": {n: sha_file(part_path(n)) for n in SIX}, "p1d_parts": {n: sha_file(D.part_path(n)) for n in SIX},
           "p1a_parts": {n: sha_file(P.part_path(n)) for n in SIX}, "p2b_parts": {n: sha_file(G.part_path(n)) for n in SIX},
           "p1b2_parts": {n: sha_file(D.rc_path(n)) for n in SIX}, "utc": utc()}
    write_json(VERDICT, rec)
    VERDICT_MD.write_text(report_md(parts, p1a, rec), encoding="utf-8", newline="\n")
    log(f"P1e: chosen {chosen} on s1sel; {call} on s1eval: " + ", ".join(
        f"{n} {c['call']}" + (f" ({len(c['below'])} points, largest gap {c['largest_gap']:+.4f})" if c["below"] else "")
        for n, c in calls.items()))
    return 0


def report_md(parts, p1a, rec):
    show = (25, 50, 100, 500, 1000, 2000, 7000)
    chosen = rec["chosen"]
    L = [f"# P1e: what the links landed on decides how they are used: {chosen}, {rec['call']}", "",
         "ALL (every gold in the pool) at N, gold recall in brackets, on s1eval. The rules and sources from "
         "outputs/p1e/pools_<dataset>.json; the bar is the best of P1a's SOTA orders at each N and measure.", ""]
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
        bar = D.p1a_bar(pa)
        L.append("| the bar (P1a) | " + " | ".join(
            "{:.3f} ({:.3f}) {}".format(D.best_of(bar, n, "ALL")[1], D.best_of(bar, n, "recall")[1],
                                        D.best_of(bar, n, "ALL")[0]) for n in show)
            + f" | {sd['at_sizes']['at_I_q']['sota']['ALL']:.3f} |")
        z = sd["at_sizes"]
        g = sd["golds_per_node"]
        L += ["", "- " + "; ".join(
            f"at {lab} ({z[s]['set']['size']:,.0f} rows mean): today's pools {z[s]['set']['ALL']:.3f}, {chosen} "
            f"{z[s]['rule']['ALL']:.3f}, best SOTA {z[s]['best_sota']} {z[s]['sota']['ALL']:.3f}; learned, beside: MX "
            f"{z[s]['MX']['ALL']:.3f}, RC {z[s]['RC']['ALL']:.3f}, MXU {z[s]['MXU']['ALL']:.3f}"
            for s, lab in (("at_I_q", "|I_q|"), ("at_P4e", "|P4e|"))) + ".",
            f"- Golds per node: today's pools' ALL ({g['target_set_I_q_ALL']:.3f} at {g['set_I_q_mean_size']:,.0f} rows) "
            f"is reached by {chosen} at N = {g['rule_reaches_at_N'] or 'beyond 7,000'}, by the bar at N = "
            f"{g['bar_reaches_at_N'] or 'beyond 7,000'}; by each rule: " + ", ".join(
                f"{r} {g['rules_reach_at_N'][r] or '>7,000'}" for r in RULES) + ".",
            "- Shares of questions: " + ", ".join(f"{k} {v:.3f}" for k, v in sd["shares"].items()) + "."]
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


def verdict_selftest():
    """The verdict's checks, choice, tie-breaks and call on made-up records, in a temporary folder."""
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="p1e_selftest_"))
    g = globals()
    saved = (g["OUT"], g["VERDICT"], g["VERDICT_MD"], D.OUT, D.P1B2, P.OUT, G.OUT)
    try:
        g.update(OUT=tmp / "p1e", VERDICT=tmp / "p1e" / "verdict.json", VERDICT_MD=tmp / "p1e" / "verdict.md")
        D.OUT, D.P1B2, P.OUT, G.OUT = tmp / "p1d", tmp / "p1b2", tmp / "p1a", tmp / "p2b"
        for d in ("p1e", "p1d", "p1b2", "p1a", "p2b"):
            (tmp / d).mkdir(parents=True)

        def write_all(sel=None, ev=None, rrf5_off=None, limit=0, p1d_script="d"):
            """sel: {rule: (ALL at |I_q|, {N: ALL there})} on every s1sel carve, every other point at the bar's 0.5;
            ev: {(dataset, rule): {N: ALL there}} on s1eval."""
            sel, ev = sel or {}, ev or {}
            for n in SIX:
                carves = [f"{n}__s1eval"] + ([f"{n}__s1sel"] if n in TRAIN5 else [])
                ours = {"script_sha256": "s", "p1d_script_sha256": p1d_script, "freeze_RECORD_SHA256": "f",
                        "limit": limit, "carves": {}}
                pd = {"script_sha256": "d", "freeze_RECORD_SHA256": "f", "carves": {}}
                pa, pb, r = ({"freeze_RECORD_SHA256": "f", "carves": {}} for _ in range(3))
                for c in carves:
                    ent = {"questions": 10, "carve_sha256": "c", "regime": "S",
                           "shares": {"entity": 0.5, "passage": 0.5, "entity_route": 0.3, "route": 0.3},
                           "orders": {o: D._fake_order(0.5) for o in ORDERS},
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
                    dent = {"questions": 10, "carve_sha256": "c",
                            "orders": {o: json.loads(json.dumps(ent["orders"]["RRF5"])) if o == "RRF5"
                                       else D._fake_order(0.5) for o in P1D_ORDERS}}
                    if n == rrf5_off and c.endswith("s1eval"):
                        dent["orders"]["RRF5"]["curve"]["100"]["recall"] = 0.51
                    pd["carves"][c] = dent
                    pa["carves"][c] = {"questions": 10,
                                       "orders": {o: D._fake_order(0.5) for o in ("A3", "A0", "ORACLE") + P.SOTA}}
                    pb["carves"][c] = {"questions": 10, "carve_sha256": "c",
                                       "orders": {o: D._fake_order(0.4) for o in ("MXU", "L_st", "L_all")}}
                    r["carves"][c] = {"questions": 10, "carve_sha256": "c",
                                      "orders": {o: D._fake_order(0.45) for o in ("RC", "MX")}}
                write_json(part_path(n), ours)
                write_json(D.part_path(n), pd)
                write_json(P.part_path(n), pa)
                write_json(G.part_path(n), pb)
                write_json(D.rc_path(n), r)

        def run():
            assert verdict_stage() == 0
            return json.loads(VERDICT.read_text(encoding="utf-8"))

        # every rule at the bar: no point below anywhere, the first rule in RULES' order, ALL_SIX
        write_all()
        v = run()
        assert v["chosen"] == RULES[0] and v["call"] == "ALL_SIX", (v["chosen"], v["call"])
        assert VERDICT_MD.read_text(encoding="utf-8").startswith("# P1e: what the links landed on")
        assert v["beside"]["metaqa__s1eval"]["golds_per_node"]["rule_reaches_at_N"] == NS[0]
        assert set(v["beside"]["webqsp__s1eval"]["like_for_like"]) == set(RULES)
        # the higher mean ALL at I_q breaks a tie in points and gaps
        write_all({RULES[3]: (0.52, {})})
        assert run()["chosen"] == RULES[3]
        # fewer points below beats a smaller largest gap and a higher mean
        two = {25: 0.497, 100: 0.497}
        write_all({r: ((0.6, two) if k == 0 else (0.5, {25: 0.49}) if k == 1 else (0.5, two))
                   for k, r in enumerate(RULES)})
        v = run()
        assert v["chosen"] == RULES[1] and v["choice_s1sel"][RULES[1]]["below_points"] == 5, v["choice_s1sel"][RULES[1]]
        assert v["choice_s1sel"][RULES[0]]["below_points"] == 10 and v["choice_s1sel"][RULES[0]]["largest_gap"] == -0.003
        # on equal points, the smaller largest gap beats a higher mean
        write_all({r: ((0.6, {25: 0.49}) if k == 0 else (0.5, {25: 0.495}) if k == 2 else (0.5, {25: 0.49}))
                   for k, r in enumerate(RULES)})
        assert run()["chosen"] == RULES[2]
        # the call on s1eval: at the tie is AT_OR_ABOVE, just past it is BELOW, named with its point and the leader
        write_all(ev={("musique", RULES[0]): {500: 0.4985}})
        v = run()
        assert v["call"] == "ALL_SIX" and v["s1eval"]["musique"]["call"] == "AT_OR_ABOVE"
        write_all(ev={("musique", RULES[0]): {500: 0.4975}})
        v = run()
        assert v["call"] == "NOT_ALL_SIX" and v["s1eval"]["musique"]["call"] == "BELOW"
        b = v["s1eval"]["musique"]["below"]
        assert len(b) == 1 and b[0]["N"] == 500 and b[0]["measure"] == "ALL" and b[0]["order"] == P.SOTA[0], b
        # refusals: RRF5 not P1d's, a part run with --limit, a part that imported another P1d script
        for kw, word in (({"rrf5_off": "webqsp"}, "differs from P1d's filed RRF5"), ({"limit": 64}, "--limit"),
                         ({"p1d_script": "x"}, "another P1d script")):
            write_all(**kw)
            try:
                verdict_stage()
                raise AssertionError(f"not refused: {kw}")
            except SystemExit as e:
                assert word in str(e), (kw, e)
    finally:
        (g["OUT"], g["VERDICT"], g["VERDICT_MD"], D.OUT, D.P1B2, P.OUT, G.OUT) = saved
        shutil.rmtree(tmp, ignore_errors=True)


def selftest():
    from types import SimpleNamespace
    assert all(n in NS for n in (25, 50, 100, 500, 1000, 2000, 7000)), NS
    # the kind: name-only records, then the question's start nodes
    recs = [{"title": "Paris", "text": "Paris is the capital of France."}, {"display_name": "Kismet", "text": "Kismet"},
            {"display_name": "unnamed entity", "display_name_disambiguated": "unnamed entity #3", "text": "unnamed entity"},
            {"display_name_disambiguated": "topic record #7", "text": "topic record. topic notable for: X."},
            {"text": ""}, {"title": "Green (band)", "text": "Green (band)"}]
    nonly = name_only(recs, len(recs))
    assert nonly.tolist() == [False, True, True, False, False, True], nonly
    assert kind([1, 2], [0], nonly) == "entity" and kind([1, 3], [1], nonly) == "passage"
    assert kind([], [1, 5], nonly) == "entity" and kind([], [1, 0], nonly) == "passage" and kind([], [], nonly) == "passage"
    # the cut keeps K's nodes at or above 1/n, in K's order
    kst = (np.asarray([4, 2, 9, 7]), np.asarray([0.5, 0.3, 0.15, 0.05]))
    assert kcut(kst, 10).tolist() == [4, 2, 9] and kcut(kst, 2).tolist() == [4] and kcut(B.EMPTY, 5).size == 0
    # the steered walk on a toy graph: every weight 1 is step 4e's walk, bit for bit; steering moves mass to listed
    # neighbours; an empty restart walks to nothing
    n = 30
    edges = {"structural": [(i, i + 1) for i in range(12)] + [(0, 20), (20, 21), (0, 25), (0, 26)],
             "ner": [(1, 13), (13, 14), (5, 15), (2, 22)],
             "knn": [(15, 16), (3, 17), (17, 18), (25, 26), (22, 23)]}
    stores = {f: B._Store(n, e) for f, e in edges.items()}
    unions = {u: S4B.RegimeUnion(stores, list(fams), n) for u, fams in UNIONS}
    none = np.empty(0, np.int64)
    rs = [B.uniform([0]), B.uniform([3, 17]), (np.asarray([0, 5, 17]), np.asarray([0.5, 0.3, 0.2])), B.EMPTY]
    for u, _f in UNIONS:
        got = twalk(unions[u], rs, unit_steers(None, range(4)), 2)
        want = E.walk(unions[u], rs[:3], [none] * 3, NMAX, 2)
        assert all(np.array_equal(a, b) for a, b in zip(got[:3], want)) and got[3].size == 0, u
    # on toy questions (P1d's self-test's): P1d's sources are step 4e's walks, the steered walks with every weight 1
    # are too, on K's and A1's restarts; the four steered walks come back for every question
    rng = np.random.default_rng(0)
    dense = np.stack([rng.permutation(n - 1)[:12] for _ in range(4)]).astype(np.int64)
    splade = np.stack([rng.permutation(n - 1)[:12] for _ in range(4)]).astype(np.int64)
    tq = SimpleNamespace(seeds=[np.union1d(dense[q, :5], splade[q, :5]) for q in range(4)], dense_ids=dense,
                         splade_ids=splade)
    linked = [dense[0, :2].copy(), np.empty(0, np.int64), np.asarray([dense[2, 0], 29]), np.asarray([29])]
    srcs = D.sources_batch(tq, range(4), linked, unions, 60, 2)
    D.dev_check("toy", tq, range(4), linked, unions, unions["st"], "S", 60, 2, srcs)
    dev_check("toy", tq, list(range(4)), linked, unions, 60, 2, srcs)
    sw = steered_batch(tq, list(range(4)), linked, unions, srcs, 60, 2)
    assert [set(s) for s in sw] == [set(STEERED)] * 4 and sw[1]["KTst"].size == sw[3]["KTall"].size == 0
    assert all(s["A1Tst"].size and np.unique(s["A1Tst"]).size == s["A1Tst"].size for s in sw)
    prep = SimpleNamespace(dense_ids=np.asarray([[26, 3, 8]]), splade_ids=np.asarray([[26, 4]]))
    u0, ex0, fl0 = steer(prep, 0, 60)
    assert u0.tolist() == [3, 4, 8, 26] and abs(fl0 - (1 / 64 + 1 / 63)) < 1e-15
    assert abs((fl0 + ex0[3]) - 2 / 61) < 1e-15 and abs((fl0 + ex0[0]) - (1 / 62 + 1 / 63)) < 1e-15
    plain = twalk(unions["st"], [B.uniform([0])], unit_steers(None, [0]), 2)[0]
    steered = twalk(unions["st"], [B.uniform([0])], [steer(prep, 0, 60)], 2)[0]
    assert plain[0] == steered[0] == 0 and steered.tolist().index(26) < plain.tolist().index(26), (plain, steered)
    assert set(steered.tolist()) == set(plain.tolist())
    # the orders on toy sources: RRF5 is P1d's, the kind's branches, the cut route and its backfill, the fusions
    a1 = B.by_mass(np.asarray([5, 6, 7, 8]), np.asarray([0.4, 0.3, 0.2, 0.1]))
    kst = B.by_mass(np.asarray([1, 2, 3]), np.asarray([0.7, 0.2, 0.1]))
    src = (a1, a1, kst, B.EMPTY, B.by_mass(np.asarray([9, 5]), np.asarray([0.6, 0.4])))
    st = {"KTst": np.asarray([1, 3, 2]), "KTall": np.asarray([1, 2]), "A1Tst": np.asarray([6, 5, 8]),
          "A1Tall": np.asarray([5])}
    o_e, p1d = p1e_orders(src, st, True, "entity", 4, 60)
    o_p, _ = p1e_orders(src, st, True, "passage", 4, 60)
    o_n, _ = p1e_orders(src, st, False, "entity", 4, 60)
    assert set(o_e) == set(ORDERS) and set(p1d) == set(P1D_ORDERS)
    assert all(np.unique(v).size == v.size <= NMAX for o in (o_e, o_p, o_n) for v in o.values())
    assert o_e["RRF5"].tolist() == p1d["RRF5"].tolist() == D.rule_orders(src, True, 60)["RRF5"].tolist()
    assert all(o_e[k].tolist() == st[k].tolist() for k in STEERED)
    assert o_e["KCUTst"].tolist() == [1] and o_e["ROUTEcut"].tolist() == [1, 5, 6, 7, 8]
    assert o_e["KIND_T"].tolist() == p1d["ROUTEst"].tolist() == [1, 2, 3, 5, 6, 7, 8]
    assert o_e["KINDcut"].tolist() == o_e["KINDcut_T"].tolist() == o_e["KINDcut_7T"].tolist() == [1, 5, 6, 7, 8]
    assert o_p["KINDcut"].tolist() == o_p["RRF5"].tolist() and o_p["KINDcut_7T"].tolist() == o_p["RRF7T"].tolist()
    assert o_p["KIND_T"].tolist() == o_p["KINDcut_T"].tolist() == o_p["RRF5T"].tolist()
    assert o_p["RRF5T"].tolist() == D.rrf([a1[0], a1[0], st["KTst"], st["KTall"], src[4][0]], 60).tolist()
    assert o_p["RRF7T"].tolist() == D.rrf([a1[0], a1[0], st["A1Tst"], st["A1Tall"], st["KTst"], st["KTall"],
                                           src[4][0]], 60).tolist() == [5, 6, 8, 1, 2, 7, 9, 3]
    assert o_n["ROUTEcut"].tolist() == o_n["KIND_T"].tolist() == o_n["KINDcut"].tolist() == a1[0].tolist()
    verdict_selftest()
    print("selftest: ok (the name-only test and the question's kind, the cut, the steered walk equals step 4e's walk "
          "with every weight 1 on K's and A1's restarts and moves mass to listed nodes when steered, the orders and "
          "branches, the verdict's checks, choice, tie-breaks and call)")
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
