"""D5b: three more orders of B's one-link neighbours, from the question's own similarity to each (docs/D5_GOLD_REACH.md
section 6). A diagnostic beside D5: it reads golds and decides nothing; no model trains or reads.

    python scripts/d5b_neighbour_orders.py run DATASET [--limit Q]   D5's carves (s1sel; webqsp s1eval beside), on the host
    python scripts/d5b_neighbour_orders.py check DATASET             after the fetch: D5b's hops etc. equal D5's files
    python scripts/d5b_neighbour_orders.py --selftest

Per question, B and N1 exactly as D5 has them (the union of the three families; N1 = B's one-link neighbours outside B),
and with ê the frozen encoder's dense embeddings made unit length (the ones the first stage's dense list is ranked by):
    cos   N1 by ê_q · ê_v
    rrf   N1 by 1/(c + rank by cos) + 1/(c + rank by D5's weight sum), c the stage's constant (P1e's)
    res   N1 by the best parent's residual: max over v's B-parents p of cos(q - (q·p̂)p̂, v), the question with what
          p already says taken out (one parameter-free offset)
ties by node id. Per gold: its rank in N1 under each (NOT_N1 outside N1); for a gold two links out, the best rank of
an N1 node linking to it (its bridge) under each of D5's and these orders; its own rank in the first stage's dense
and splade lists (0 = not in the 1,000).
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import d5_gold_reach as D5  # noqa: E402

DECLARED = "docs/D5_GOLD_REACH.md"
OUT = D5.OUT
NOT_N1 = D5.NOT_N1
ORDERS = ("cnt", "w", "cos", "rrf", "res")
GRID = D5.GRID
log, utc, write_json, sha_file = D5.log, D5.utc, D5.write_json, D5.sha_file


def script_sha256():
    import hashlib
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def unit(x):
    x = np.asarray(x, np.float32)
    n = np.sqrt(np.einsum("...i,...i->...", x, x, dtype=np.float64))[..., None].astype(np.float32)
    return x / np.where(n > 0, n, np.float32(1.0))


def pair_dots(A, ia, C, ic, block=4096):
    """A[ia[k]] · C[ic[k]] for every k, in blocks (no full copy of the pairs' rows)."""
    out = np.empty(ia.size, np.float32)
    for i in range(0, ia.size, block):
        out[i:i + block] = np.einsum("ij,ij->i", A[ia[i:i + block]], C[ic[i:i + block]])
    return out


def ranks_desc(score, ids):
    """1-based rank of each entry under score descending, ties by id ascending."""
    o = np.lexsort((ids, -np.asarray(score, np.float64)))
    r = np.empty(ids.size, np.int64)
    r[o] = np.arange(1, ids.size + 1)
    return r


def n1_table(adj, B, w_B):
    """N1 (sorted) with, per node, its B-parent count and weight sum, and the distinct (parent index, N1 index) pairs."""
    ids, par, _fam = D5.neighbours(adj, B)
    out1 = ~np.isin(ids, B)
    ids, par = ids[out1], par[out1]
    key = ids * (B.size + 1) + par
    _, first = np.unique(key, return_index=True)
    u_ids, u_par = ids[first], par[first]
    n1, inv = np.unique(u_ids, return_inverse=True)
    cnt = np.bincount(inv, minlength=n1.size).astype(np.int64)
    wsum = np.bincount(inv, weights=np.asarray(w_B, np.float64)[u_par], minlength=n1.size)
    return n1, cnt, wsum, u_par, inv


def orders_of(n1, cnt, wsum, u_par, inv, q, EB, EN, c):
    """{order: rank of each N1 node}. q: the question's unit embedding; EB, EN: unit embeddings of B and N1."""
    if n1.size == 0:
        return {o: np.empty(0, np.int64) for o in ORDERS}
    r = {"cnt": ranks_desc(cnt, n1), "w": ranks_desc(wsum, n1)}
    qv = EN @ q
    r["cos"] = ranks_desc(qv, n1)
    r["rrf"] = ranks_desc(1.0 / (c + r["cos"]) + 1.0 / (c + r["w"]), n1)
    qp = EB @ q                                         # q·p̂ per B node
    pv = pair_dots(EB, u_par, EN, inv)                  # p̂·v̂ per (parent, node) pair
    den = np.sqrt(np.maximum(1.0 - qp[u_par] ** 2, 1e-12))
    s = (qv[inv] - qp[u_par] * pv) / den
    best = np.full(n1.size, -np.inf)
    np.maximum.at(best, inv, s)
    r["res"] = ranks_desc(best, n1)
    return r


def gold_ranks(adj, n1, r, golds, hop):
    """Per gold: its rank under each order (NOT_N1 outside N1), and for a hop-2 gold the best rank of an N1 node
    linking to it (NOT_N1 when none, or the gold is not two links out)."""
    out = {f"r_{o}": np.full(golds.size, NOT_N1, np.int64) for o in ORDERS}
    out.update({f"b_{o}": np.full(golds.size, NOT_N1, np.int64) for o in ORDERS})
    if n1.size == 0:
        return out
    p = np.minimum(np.searchsorted(n1, golds), n1.size - 1)
    in1 = n1[p] == golds
    for o in ORDERS:
        out[f"r_{o}"][in1] = r[o][p[in1]]
    for i in np.flatnonzero(hop == 2):
        nb, _par, _f = D5.neighbours(adj, golds[i:i + 1])
        k = np.minimum(np.searchsorted(n1, nb), n1.size - 1)
        k = np.unique(k[n1[k] == nb])
        if k.size:
            for o in ORDERS:
                out[f"b_{o}"][i] = int(r[o][k].min())
    return out


def list_rank(lst, golds):
    lst = np.asarray(lst, np.int64)
    pos = {int(x): i + 1 for i, x in enumerate(lst)}
    return np.asarray([pos.get(int(g), 0) for g in golds], np.int64)


# ── per-parent beams ─────────────────────────────────────────────────────────

P_GRID = (10, 20, 50, 100, 200)             # B's first P nodes by first-stage weight
M_GRID = (1, 2, 3, 5, 10)                   # each one's first m neighbours outside B
P2_MAX, M2_MAX = 50, 5                      # the second step only up to these (its cost grows as P x m x degree)
BEAM_SCORES = ("cos", "res")


def top_children(adj, parents, exclude, q, Ep, read, mmax):
    """For each parent (in the given order) its first mmax neighbours outside exclude, by cos(q, v) and by
    res_p(v) = cos(q - (q·p̂)p̂, v), ties by id: {score: (parent index, node, rank within the parent)}."""
    ids, par, _f = D5.neighbours(adj, np.asarray(parents, np.int64))
    keep = ~np.isin(ids, exclude)
    ids, par = ids[keep], par[keep]
    if ids.size == 0:
        e = np.empty(0, np.int64)
        return {sc: (e, e, e) for sc in BEAM_SCORES}
    _, first = np.unique(par * (int(ids.max()) + 1) + ids, return_index=True)
    ids, par = ids[first], par[first]
    u, inv = np.unique(ids, return_inverse=True)
    E = unit(read(u))
    qv = (E @ q)[inv]
    qp = Ep @ q
    pv = pair_dots(Ep, par, E, inv)
    res = (qv - qp[par] * pv) / np.sqrt(np.maximum(1.0 - qp[par] ** 2, 1e-12))
    out = {}
    for sc, score in (("cos", qv), ("res", res)):
        o = np.lexsort((ids, -score, par))
        ps, vs = par[o], ids[o]
        start = np.searchsorted(ps, ps, side="left")
        rk = np.arange(ps.size) - start
        k = rk < mmax
        out[sc] = (ps[k], vs[k], rk[k])
    return out


def cat(parts):
    parts = [x for x in parts if x.size]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)


def beams(adj, B, w_B, q, read, golds):
    """Per gold, whether B plus the beam holds it, and per question the rows: one step (beam1: the first m neighbours
    of each of B's first P nodes) and two (beam2: beam1 plus the first m neighbours of each beam1 node, P <= P2_MAX and
    m <= M2_MAX; elsewhere False and -1)."""
    shape = (len(BEAM_SCORES), len(P_GRID), len(M_GRID))
    c1, c2 = np.zeros((golds.size,) + shape, bool), np.zeros((golds.size,) + shape, bool)
    r1, r2 = np.full(shape, -1, np.int64), np.full(shape, -1, np.int64)
    inB = np.isin(golds, B)
    Bord = B[np.lexsort((B, -np.asarray(w_B, np.float64)))][:max(P_GRID)]
    top1 = top_children(adj, Bord, B, q, unit(read(Bord)), read, max(M_GRID))
    b1max = cat([top1[sc][1][(top1[sc][0] < P2_MAX) & (top1[sc][2] < M2_MAX)] for sc in BEAM_SCORES])
    top2 = top_children(adj, b1max, B, q, unit(read(b1max)), read, M2_MAX) if b1max.size else None
    for si, sc in enumerate(BEAM_SCORES):
        ps, vs, rk = top1[sc]
        for pi, P in enumerate(P_GRID):
            for mi, m in enumerate(M_GRID):
                b1 = np.unique(vs[(ps < P) & (rk < m)])
                c1[:, si, pi, mi] = inB | np.isin(golds, b1)
                r1[si, pi, mi] = B.size + b1.size
                if P > P2_MAX or m > M2_MAX:
                    continue
                b2 = b1
                if top2 is not None and b1.size:
                    ps2, vs2, rk2 = top2[sc]
                    b2 = cat([b1, vs2[np.isin(b1max[ps2], b1) & (rk2 < m)]])
                c2[:, si, pi, mi] = inB | np.isin(golds, b2)
                r2[si, pi, mi] = B.size + b2.size
    return c1, c2, r1, r2

# ── the stage ────────────────────────────────────────────────────────────────


def run(name, limit=0):
    import p1e_anchor_kind as K
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from mp_retrieval.m3b_features import DenseNodes
    G, S4, E = K.G, K.S4, K.E
    op = S4.Opened()
    ds = op.canonical.Dataset(name, root=str(op.served))
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    cache = {}
    rec = {"declared_in": DECLARED, "stage": "D5b", "dataset": name, "script_sha256": script_sha256(),
           "d5_script_sha256": D5.script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "carves": {}}

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        stores, _unions = G.open_unions(cache, op, name, union)
        adj = [(np.ascontiguousarray(stores[f].indptr, np.int64), np.ascontiguousarray(stores[f].col))
               for f in S4.FAMILIES]
        with np.load(D5.P1F / f"stats_{name}__{carve}.npz") as z:
            st = {k: z[k] for k in ("bs", "inc", "ptr", "depth")}
        nq = min(limit, len(U)) if limit else len(U)
        golds = [np.unique(np.asarray(gg, np.int64)) for gg in pop.golds[:nq]]
        ptr = np.concatenate([[0], np.cumsum([gg.size for gg in golds])]).astype(np.int64)
        if not np.array_equal(st["ptr"][:nq + 1], ptr):
            raise SystemExit(f"{name}/{carve}: P1f's golds are not the carve's")
        cols = {k: [] for k in [f"r_{o}" for o in ORDERS] + [f"b_{o}" for o in ORDERS] + ["dense", "splade", "hop",
                                                                                         "today"]}
        n1s, mism = [], 0
        bc1, bc2, br1, br2 = [], [], [], []
        for j in range(nq):
            B = np.unique(np.asarray(bs[j], np.int64))
            if B.size != int(st["bs"][j]):
                raise SystemExit(f"{name}/{carve} question {j}: |B| {B.size}, P1f's {int(st['bs'][j])}")
            g = golds[j]
            w_B = D5.weights_of(K.steer(prep, j, constant), B)
            m5, n1_5, _n2, _o = D5.measure(adj, B, g, w_B)             # D5's own measures, as D5 filed them
            hop = m5["hop"]
            take = max(0, int(st["inc"][j]) - B.size)
            cols["today"].append(st["depth"][ptr[j]:ptr[j + 1]] <= take)
            cols["hop"].append(hop)
            n1, cnt, wsum, u_par, inv = n1_table(adj, B, w_B)
            q = unit(np.asarray(prep.qemb[j], np.float64))
            EB, EN = unit(nodes.read(B)), unit(nodes.read(n1))
            r = orders_of(n1, cnt, wsum, u_par, inv, q, EB, EN, constant)
            gr = gold_ranks(adj, n1, r, g, hop)
            mism += int(not (np.array_equal(gr["r_cnt"], m5["r_cnt"]) and np.array_equal(gr["r_w"], m5["r_w"])
                             and n1.size == n1_5))
            for k in gr:
                cols[k].append(gr[k])
            cols["dense"].append(list_rank(prep.dense_ids[j], g))
            cols["splade"].append(list_rank(prep.splade_ids[j], g))
            n1s.append(n1.size)
            c1, c2, rr1, rr2 = beams(adj, B, w_B, q, nodes.read, g)
            bc1.append(c1)
            bc2.append(c2)
            br1.append(rr1)
            br2.append(rr2)
            if (j + 1) % 250 == 0:
                log(f"{name}/{carve}: {j + 1}/{nq} ({time.time() - t:.0f}s)")
        if mism:
            raise SystemExit(f"{name}/{carve}: {mism} questions' N1, count or weight ranks are not D5's measure's")
        a = {k: np.concatenate(v) if v else np.empty(0, np.int64) for k, v in cols.items()}
        a.update(beam1=np.concatenate(bc1), beam2=np.concatenate(bc2), rows1=np.stack(br1), rows2=np.stack(br2),
                 ptr=ptr, bs=st["bs"][:nq].astype(np.int64), n1=np.asarray(n1s, np.int64))
        s = summary(a)
        if limit:
            log(f"{name}/{carve}: development run on {nq} questions ({time.time() - t:.0f}s); nothing written: "
                f"{json.dumps(s)}")
            return
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"orders_{name}__{carve}.npz"
        np.savez_compressed(OUT / f"orders_{name}__{carve}.tmp.npz", **a)
        os.replace(OUT / f"orders_{name}__{carve}.tmp.npz", p)
        rec["carves"][f"{name}__{carve}"] = {"questions": nq, "file_sha256": sha_file(p), "summary": s,
                                             "checks": {"n1_and_ranks_equal_d5_measure": True},
                                             "seconds": round(time.time() - t, 1), "utc": utc()}
        write_json(OUT / f"orders_{name}.json", rec)
        log(f"{name}/{carve} ({nq} q, {time.time() - t:.0f}s): written")

    E.run_carves(op, name, set(D5.carves_of(name)), ("A3",), 2, sink)
    return 0


def check(name):
    """After the fetch: D5b's hops, today, N1 sizes and D5's two ranks equal D5's filed arrays (outputs/d5)."""
    d5 = json.loads((OUT / f"reach_{name}.json").read_text(encoding="utf-8"))
    bad = []
    for c5, e in d5["carves"].items():
        p5 = OUT / f"reach_{c5}.npz"
        if sha_file(p5) != e["file_sha256"]:
            raise SystemExit(f"{p5}: not the file D5 filed")
        with np.load(p5) as z5, np.load(OUT / f"orders_{c5}.npz") as zb:
            for k5, kb in (("ptr", "ptr"), ("hop", "hop"), ("today", "today"), ("n1", "n1"), ("r_cnt", "r_cnt"),
                           ("r_w", "r_w"), ("bs", "bs")):
                if not np.array_equal(z5[k5], zb[kb]):
                    bad.append(f"{c5}:{k5}")
    print(f"{name}: D5b equals D5's filed arrays" if not bad else f"{name}: NOT EQUAL {bad}", flush=True)
    return 1 if bad else 0


def summary(a):
    hop, today = a["hop"], a["today"]
    nq = a["n1"].size
    s = {"questions": int(nq)}
    h0 = hop == 0
    s["grid"] = {o: {k: {"ALL": round(float(D5.q_all(a, h0 | (a[f"r_{o}"] <= k)).mean()), 4),
                         "rows": round(float((a["bs"] + np.minimum(k, a["n1"])).mean()), 1)} for k in GRID}
                 for o in ORDERS}
    s["grid_today_or"] = {o: {k: round(float(D5.q_all(a, today | (a[f"r_{o}"] <= k)).mean()), 4) for k in GRID}
                          for o in ORDERS}
    miss = ~today
    m1, m2 = miss & (hop == 1), miss & (hop == 2)
    qs = (0.25, 0.5, 0.75, 0.9)
    s["missed_hop1"] = {"golds": int(m1.sum()), **({o: {q: int(np.quantile(a[f"r_{o}"][m1], q)) for q in qs}
                                                    for o in ORDERS} if m1.any() else {})}
    b2 = {o: a[f"b_{o}"][m2] for o in ORDERS}
    s["missed_hop2_bridge"] = {"golds": int(m2.sum()), **({o: {q: int(np.quantile(b2[o], q)) for q in qs}
                                                           for o in ORDERS} if m2.any() else {})}
    if "beam1" in a:
        s["beam"] = {}
        for si, sc in enumerate(BEAM_SCORES):
            for pi, P in enumerate(P_GRID):
                for mi, m in enumerate(M_GRID):
                    e = {"ALL1": round(float(D5.q_all(a, a["beam1"][:, si, pi, mi]).mean()), 4),
                         "rows1": round(float(a["rows1"][:, si, pi, mi].mean()), 1),
                         "ALL_today_or1": round(float(D5.q_all(a, today | a["beam1"][:, si, pi, mi]).mean()), 4)}
                    if P <= P2_MAX and m <= M2_MAX:
                        e.update(ALL2=round(float(D5.q_all(a, a["beam2"][:, si, pi, mi]).mean()), 4),
                                 rows2=round(float(a["rows2"][:, si, pi, mi].mean()), 1),
                                 ALL_today_or2=round(float(D5.q_all(a, today | a["beam2"][:, si, pi, mi]).mean()), 4))
                    s["beam"][f"{sc}_P{P}_m{m}"] = e
    for nm, m in (("missed", miss), ("missed_hop1", m1), ("missed_hop2", m2), ("held_today", today)):
        if m.any():
            s[f"first_stage_{nm}"] = {"in_dense_1000": round(float((a["dense"][m] > 0).mean()), 4),
                                      "in_splade_1000": round(float((a["splade"][m] > 0).mean()), 4)}
    return s


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    # nodes 0..7; B = {0, 1}; links 0-2, 0-3, 1-3, 1-4, 3-5, 4-6
    st = D5.csr(8, [(0, 2), (0, 3), (1, 3), (1, 4), (3, 5), (4, 6)])
    adj = [st, D5.csr(8, []), D5.csr(8, [])]
    B = np.array([0, 1])
    n1, cnt, wsum, u_par, inv = n1_table(adj, B, np.array([1.0, 0.5]))
    assert n1.tolist() == [2, 3, 4] and cnt.tolist() == [1, 2, 1] and np.allclose(wsum, [1.0, 1.5, 0.5])
    # embeddings in 3-d: q = x; p0 = x (B node 0 says all of q), p1 = y; node 2 = x, 3 = (x+z), 4 = z
    E = unit(np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
                      np.float64))
    q = unit(np.array([1.0, 0, 0]))
    r = orders_of(n1, cnt, wsum, u_par, inv, q, E[B], E[n1], 60.0)
    assert r["cnt"].tolist() == [2, 1, 3] and r["w"].tolist() == [2, 1, 3]
    assert r["cos"].tolist() == [1, 2, 3]                        # cos 1, 0.707, 0
    # res: node 2's only parent is 0 (q·p0 = 1: the residual is ~0 length, score (1 - 1)/~0 = 0); node 3 via parent 1
    # (q·p1 = 0): cos 0.707; node 4 via parent 1: 0
    assert r["res"][1] == 1, r["res"]
    rrf = 1 / (60 + np.array([1, 2, 3])) + 1 / (60 + np.array([2, 1, 3]))
    assert r["rrf"].tolist() == ranks_desc(rrf, n1).tolist()
    golds = np.array([0, 3, 5, 6, 7])
    hop = np.array([0, 1, 2, 2, 3])
    gr = gold_ranks(adj, n1, r, golds, hop)
    assert gr["r_cos"].tolist() == [NOT_N1, 2, NOT_N1, NOT_N1, NOT_N1]
    assert gr["b_cnt"].tolist() == [NOT_N1, NOT_N1, 1, 3, NOT_N1]  # 5's bridge is 3 (rank 1 by count); 6's is 4
    assert list_rank([9, 3, 5], np.array([3, 4])).tolist() == [2, 0]
    a = {"hop": hop, "today": np.array([True, False, False, True, False]), "ptr": np.array([0, 2, 5]),
         "bs": np.array([2, 2]), "n1": np.array([3, 3]), "dense": np.array([1, 0, 0, 5, 0]),
         "splade": np.array([0, 0, 0, 0, 0]), **gr}
    s = summary(a)
    assert s["grid"]["cos"][50]["ALL"] == 0.5 and s["grid"]["cos"][50]["rows"] == 5.0    # q0 = {0, 3}: both held
    assert s["missed_hop1"]["golds"] == 1 and s["missed_hop2_bridge"]["golds"] == 1
    assert s["first_stage_missed"]["in_dense_1000"] == 0.0 and s["first_stage_held_today"]["in_dense_1000"] == 1.0
    # beams: B = {0, 1}, weights put 1 first; m = 1 from each parent by cos: 1 -> 3 (cos .707) over 4 (0); 0 -> 2
    def read(rows):
        return E[np.asarray(rows, np.int64)]
    g = np.array([3, 5, 6])
    c1, c2, r1, r2 = beams(adj, B, np.array([0.5, 1.0]), q, read, g)
    si, pi, mi = BEAM_SCORES.index("cos"), P_GRID.index(10), M_GRID.index(1)
    assert c1[:, si, pi, mi].tolist() == [True, False, False] and r1[si, pi, mi] == 4      # B + {2, 3}
    # step 2 from {2, 3}: 2 has no neighbour outside B; 3's are 0, 1 (B) and 5: beam2 = {2, 3, 5}
    assert c2[:, si, pi, mi].tolist() == [True, True, False] and r2[si, pi, mi] == 5
    assert r2[si, P_GRID.index(100), mi] == -1                                                # not computed past P2_MAX
    mi2 = M_GRID.index(2)
    assert c1[:, si, pi, mi2].tolist() == [True, False, False] and r1[si, pi, mi2] == 5      # B + {2, 3, 4}
    assert c2[:, si, pi, mi2].tolist() == [True, True, True] and r2[si, pi, mi2] == 7        # + 5, 6
    print("d5b_neighbour_orders selftest: ok", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run", "check"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run(a.dataset, a.limit)
    if a.cmd == "check":
        return check(a.dataset)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
