"""U2c: today's pool rules on U2b's chosen graph, each also cut to fewer rows, with the upstream-failure ledger
(docs/U2C_POOLS_ON_CHOSEN.md). Pools only: nothing trains, nothing is fit, and no cut reads a gold.

    python outputs/mp_unified/zu2c.py stats DATASET --graph today|chosen [--threads 2] [--limit Q]
    python outputs/mp_unified/zu2c.py verdict
    python outputs/mp_unified/zu2c.py --selftest

stats    One dataset's selection carve (s1sel; webqsp's s1eval, beside), on one graph:
           today   the four passage datasets' structural graph read back from outputs/u2b/<D>/pkg_structural.npz and
                   swapped in (the swap U2c makes for the chosen graph); metaqa and webqsp as cached (their graph is
                   their KB under both). Every question must reproduce P1g's filed rows and held golds for all 805
                   candidates, P1f's |B|, |I_q| and depths, and the frozen I of the cached graph.
           chosen  U2b's chosen graph (outputs/u2b/choice.json, status CHOSEN) swapped in for `structural` on musique,
                   squad, hotpotqa and 2wiki: base and seeds are step 4d's, the frozen I is rebuilt on the graph, and
                   KIND_T, the beams and the cuts run on it. Not run on metaqa or webqsp.
         Per question and candidate (P1g's 805): rows, and each gold's place in the candidate's build order (P1h's: 0 in
         base and seeds, p >= 1 the expansion's p-th node, -1 outside). Per gold, for the ledger: in the first stage's
         lists, one link from base and seeds, in KIND_T's order, in the frozen I. --limit Q: the first Q questions,
         every check, nothing written and no number logged.
verdict  P1g2's 1,603 candidates, and each one without a beam at P1h's nine taus below 1, on both graphs; P1g's choice
         rule on each; the ledger of the chosen graph's misses. Writes outputs/u2c/verdict.{json,md}.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import p1f_pool_size as F  # noqa: E402  (P1f's stats, cost and weights)
import p1g_pool_beam as PG  # noqa: E402  (P1g's candidates, extra sets, evaluation, summary and choice)
import p1g2_gated as PG2  # noqa: E402  (P1g2's candidates and P1g's filed arrays)
import zp1h as P1H  # noqa: E402  (P1h's grid and cut)

D5, D5B = PG.D5, PG.D5B
log, utc, write_json, sha_file = PG.log, PG.utc, PG.write_json, PG.sha_file
DECLARED = "docs/U2C_POOLS_ON_CHOSEN.md"
OUT = ROOT / "outputs" / "u2c"
U2B = ROOT / "outputs" / "u2b"
SIX, TRAIN5 = PG.SIX, PG.TRAIN5
PASSAGE = ("musique", "squad", "hotpotqa", "2wiki")      # U2b's four: the graph is swapped on these only
GRAPHS = ("today", "chosen")
TAUS = P1H.TAUS
UNORDERED = np.int32(1 << 30)     # a gold only a beam adds: held, at no single place (beams enter at tau = 1 only)
CAUSES = ("TAU_CUT", "BUDGET", "UNREACHED")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def carve_of(name):
    if name not in SIX:
        raise SystemExit(f"dataset must be one of {SIX}")
    return D5.carves_of(name)[0]


def tkey(t):
    return P1H.key(t)


# ── the graph ────────────────────────────────────────────────────────────────


def csr_store(n, indptr, col):
    """FamilyStore.from_graph's structural store for an undirected edge set given as a symmetric CSR (rows sorted, no
    self-loop): each pair stored once from its lower id, relation 0. Built from the CSR without from_graph's sort; the
    self-test checks it equal to from_graph field for field."""
    from mp_retrieval.m3b_pools import FamilyStore
    indptr = np.ascontiguousarray(indptr, np.int64)
    col = np.ascontiguousarray(col, np.int32)
    if indptr.size != n + 1 or int(indptr[-1]) != col.size:
        raise SystemExit("the CSR is not one of n nodes")
    deg = np.diff(indptr)
    tdir = np.empty(col.size, np.int8)
    step = 1 << 24
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        a, b = int(indptr[lo]), int(indptr[hi])
        heads = np.repeat(np.arange(lo, hi, dtype=np.int64), deg[lo:hi])
        tdir[a:b] = np.where(heads < col[a:b], 1, 2)
        del heads
    if (tdir == 1).sum() * 2 != col.size:
        raise SystemExit("the CSR is not symmetric without self-loops")
    m = col.size // 2
    rel_count = np.asarray([m], np.int64) if m else np.zeros(0, np.int64)
    return FamilyStore("structural", int(n), indptr, col, typed_indptr=indptr.copy(), typed_col=col,
                       typed_rel=np.zeros(col.size, np.int16), typed_dir=tdir, rel_count=rel_count, edges_stored=m)


def chosen_graph():
    """U2b's chosen graph name, from its choice record (status CHOSEN, written by zu2b.py as committed)."""
    import u1b
    p = U2B / "choice.json"
    if not p.exists():
        raise SystemExit(f"{p}: U2b has not chosen")
    ch = json.loads(p.read_text(encoding="utf-8"))
    if ch.get("status") != "CHOSEN" or not ch.get("chosen"):
        raise SystemExit(f"{p}: U2b's status is {ch.get('status')}, not CHOSEN")
    if ch.get("zu2b_sha256") != u1b.sha_src(HERE / "zu2b.py"):
        raise SystemExit(f"{p}: not written by the committed zu2b.py")
    return ch["chosen"], sha_file(p)


def graph_csr(name, graph, n):
    """(indptr, col, description) of the graph swapped in on a passage dataset; None on metaqa and webqsp."""
    if name not in PASSAGE:
        return None
    if graph == "today":
        q = json.loads((U2B / name / "questions.json").read_text(encoding="utf-8"))
        p = U2B / name / "pkg_structural.npz"
        if sha_file(p) != q["pkg_structural_npz_sha256"]:
            raise SystemExit(f"{p}: not the file U2b's question record names")
        desc = {"graph": "PKG", "npz": str(p.relative_to(ROOT)).replace("\\", "/"), "npz_sha256": sha_file(p)}
    else:
        g, csha = chosen_graph()
        rule = "u1d" if g.startswith("V") else "new"
        b = json.loads((U2B / name / "graphs" / f"build_{rule}.json").read_text(encoding="utf-8"))
        e = b["graphs"][g]
        p = U2B / name / "graphs" / e["file"]
        if not p.exists():
            raise SystemExit(f"{p}: fetch it from the U2b volume first")
        if sha_file(p) != e["npz_sha256"]:
            raise SystemExit(f"{p}: not the file U2b's build record names")
        desc = {"graph": g, "npz": str(p.relative_to(ROOT)).replace("\\", "/"), "npz_sha256": e["npz_sha256"],
                "choice_sha256": csha, "build": {k: e[k] for k in ("directed_edges", "undirected_neighbours",
                                                                     "mean_degree", "max_degree") if k in e}}
    with np.load(p) as z:
        ip, col = z["indptr"], z["col"]
    if ip.size != n + 1:
        raise SystemExit(f"{p}: {ip.size - 1} nodes, not {n}")
    return ip, col, desc


# ── one question ─────────────────────────────────────────────────────────────


def extra_sets_ranked(adj, B, w_B, q, read, c, want):
    """P1g's extra sets (the beams through p1g_pool_beam.extra_sets, unchanged; each cut as it computes it: the first
    k of B's one-link neighbours under D5b's order), each cut node's rank under its order, and B's one-link
    neighbours n1."""
    fam = {nm: (f, p) for nm, f, p in PG.extras()}
    other = {nm for nm in want if fam[nm][0] != "CUT"}
    sets = PG.extra_sets(adj, B, w_B, q, read, c, other) if other else {}
    ranks = {}
    n1, cnt, wsum, u_par, inv = D5B.n1_table(adj, B, w_B)
    cuts = [(nm, fam[nm][1]) for nm in want if fam[nm][0] == "CUT"]
    r = D5B.orders_of(n1, cnt, wsum, u_par, inv, q, D5B.unit(read(B)), D5B.unit(read(n1)), c) \
        if cuts and n1.size else None
    for nm, (o, k) in cuts:
        if r is None:
            sets[nm], ranks[nm] = PG.EMPTY, PG.EMPTY
        else:
            keep = r[o] <= k
            sets[nm], ranks[nm] = n1[keep], np.asarray(r[o][keep], np.int64)
    return sets, ranks, n1


def place_question(B, full, base, golds, sets, ranks, cands, fams):
    """P1g's evaluate_question (rows and held, unchanged), and each gold's place in each candidate's build order:
    0 in B; p >= 1 the p-th node of the expansion, which is KIND_T's first n_f nodes outside B, then the extra set's
    nodes outside them by their rank under the extra's order (P1h's build order); -1 outside the pool. A gold only a
    beam adds is UNORDERED."""
    rows, held = PG.evaluate_question(B, full, base, golds, sets, cands)
    nf = {f: min(full.size, int(np.rint(f * base))) for f in PG.FILL_F}
    inB = np.isin(golds, B)
    gpos = PG.positions_in(golds, full)
    lay = {}
    for ex, X in sets.items():
        X = np.asarray(X, np.int64)
        Xs = X[np.argsort(ranks[ex], kind="stable")] if fams[ex] == "CUT" else X
        lay[ex] = (Xs, PG.positions_in(Xs, full), PG.positions_in(golds, Xs))
    pos = np.full((golds.size, len(cands)), -1, np.int32)
    for i, (_n, f, ex) in enumerate(cands):
        k = nf[f]
        p = np.full(golds.size, -1, np.int64)
        inf = gpos < k
        p[inf] = gpos[inf] + 1
        Xs, xp, gi = lay[ex]
        out = xp >= k
        if Xs.size:
            inx = (gi < Xs.size) & ~inf & ~inB
            if fams[ex] == "CUT":
                p[inx] = k + np.cumsum(out)[gi[inx]]
            else:
                p[inx] = UNORDERED
        p[inB] = 0
        if int(rows[i]) != B.size + k + int(out.sum()) or not np.array_equal(p >= 0, held[:, i]):
            raise SystemExit("a build order disagrees with P1g's rows or held golds")
        pos[:, i] = p
    return rows, held, pos


# ── stats (the laptop, or the host) ──────────────────────────────────────────


def stats_stage(name, graph, threads=2, limit=0):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    carve = carve_of(name)
    if graph not in GRAPHS:
        raise SystemExit(f"graph must be one of {GRAPHS}")
    if graph == "chosen" and name not in PASSAGE:
        raise SystemExit(f"{name}: its graph is its KB under both; the today run serves both")
    numba.set_num_threads(threads)
    t0 = time.time()
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    n = int(ds.n_nodes)
    g = graph_csr(name, graph, n)
    stores, union, desc = stores0, union0, {"graph": "cached (the KB)"}
    if g is not None:
        ip, col, desc = g
        swap = csr_store(n, ip, col)
        del ip, col
        stores = dict(stores0)
        if "structural" in families:
            stores["structural"] = swap
        union = S4B.RegimeUnion(stores, families, n) if families else union0
        load0 = S4.m3b_pools.load_or_build_store

        def load(ds_, family, cache_dir):
            if family == "structural":
                if ds_.name != name:
                    raise SystemExit(f"{ds_.name}: U2c swaps {name}'s graph only")
                return swap
            return load0(ds_, family, cache_dir)

        S4.m3b_pools.load_or_build_store = load
    pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c == carve]
    (carve_, pop), = pops
    index = texts = None
    if name != "squad":
        index = E.build_index(E.node_names(ds), E.stopwords())
        texts = E.query_texts(ds)
        if len(texts) != int(ds.n_queries):
            raise SystemExit(f"{name}: {len(texts)} query records, not {ds.n_queries}")
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    prep0 = op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores0, op.m3a, op.m3b_contract)[0]
    prep = prep0 if stores is stores0 else \
        op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores, op.m3a, op.m3b_contract)[0]
    if prep.dense_ids.shape[1] < 1000 or prep.splade_ids.shape[1] < 1000:
        raise SystemExit(f"{name}/{carve}: retrieval lists are not 1000 deep")
    bases = op.m3b_compile.base_rows(construction, prep0.dense_ids, prep0.splade_ids, op.m3a, op.m3b_contract,
                                     constant)
    _U0, _BQ, bs, inc0, _f = E.union_pools(name, carve, pop, prep0, bases)     # base, seeds and G0's I: step 4d's
    del bases, _U0
    if any(not np.array_equal(np.asarray(a, np.int64), np.asarray(b, np.int64)) for a, b in zip(prep.seeds, prep0.seeds)):
        raise SystemExit(f"{name}/{carve}: the seeds on the graph are not G0's")
    inc = [np.asarray(p, np.int64) for p in prep.pools]
    if graph == "today" and any(not np.array_equal(a, b) for a, b in zip(inc, inc0)):
        raise SystemExit(f"{name}/{carve}: today's graph through the swap does not rebuild the frozen I")
    linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [PG.EMPTY] * len(bs)
    log(f"{name}/{carve} {graph}: prepared ({time.time() - t0:.0f}s)")

    cands = PG.candidates()
    fams = {nm: f for nm, f, _p in PG.extras()}
    want = {ex for _n, _f, ex in cands}
    cache = {}
    stores_all, unions = G.open_unions(cache, op, name, union)
    adj = [(np.ascontiguousarray(stores_all[f].indptr, np.int64), np.ascontiguousarray(stores_all[f].col))
           for f in S4.FAMILIES]
    n_st = int(unions["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    st = F.load_stats(name, carve)
    nq = min(limit, len(bs)) if limit else len(bs)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(st["ptr"][:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: P1f's golds are not the carve's")
    ref = None
    if graph == "today":
        a, names, _rec = PG2.load_arrays(name, carve)
        if list(names) != [c[0] for c in cands]:
            raise SystemExit(f"{name}: P1g's arrays are not P1g's 805 candidates in order")
        ref = a
    ti = [c[0] for c in cands].index(PG.TODAY)
    rows, pos, fills, ledger = [], [], np.zeros(nq, np.int64), {k: [] for k in ("listed", "one_link", "kt_pos", "in_I")}
    t = time.time()
    for lo in range(0, nq, S4.BATCH):
        js = list(range(lo, min(nq, lo + S4.BATCH)))
        srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
        steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
        for i, j in enumerate(js):
            src, _nn, (L, Ls) = srcs[i]
            route = D.routes(L, Ls)
            knd = K.kind(Ls, prep.seeds[j], nonly)
            o = np.asarray(K.p1e_orders(src, steered[i], route, knd, n_st, constant)[0][F.CONSTRUCTION], np.int64)
            B = np.unique(np.asarray(bs[j], np.int64))
            if B.size != int(st["bs"][j]):
                raise SystemExit(f"{name}/{carve} question {j}: |B| is not P1f's")
            if graph == "today" and inc[j].size != int(st["inc"][j]):
                raise SystemExit(f"{name}/{carve} question {j}: |I| is not P1f's")
            take = max(0, int(inc[j].size) - B.size)
            full = F.outside(o, B)
            fill = full[:take]
            fills[j] = fill.size
            w_B = D5.weights_of(K.steer(prep, j, constant), B)
            q = D5B.unit(np.asarray(prep.qemb[j], np.float64))
            sets, ranks, n1 = extra_sets_ranked(adj, B, w_B, q, nodes.read, constant, want)
            r, h, p = place_question(B, full, fill.size, golds[j], sets, ranks, cands, fams)
            if ref is not None:
                today = st["depth"][ptr[j]:ptr[j + 1]] <= take
                if not (np.array_equal(h[:, ti], today) and int(r[ti]) == B.size + fill.size
                        and np.array_equal(r, ref["rows"][j]) and np.array_equal(h, ref["held"][ptr[j]:ptr[j + 1]])
                        and int(ref["fill"][j]) == fill.size):
                    raise SystemExit(f"{name}/{carve} question {j}: today's graph does not reproduce P1g's arrays")
            gj = golds[j]
            ledger["listed"].append(np.isin(gj, prep.dense_ids[j]) | np.isin(gj, prep.splade_ids[j]))
            ledger["one_link"].append(np.isin(gj, n1))
            kp = PG.positions_in(gj, full)
            ledger["kt_pos"].append(np.where(kp < full.size, kp, -1).astype(np.int64))
            ledger["in_I"].append(np.isin(gj, inc[j]))
            rows.append(r)
            pos.append(p)
        if (lo // S4.BATCH) % 10 == 9:
            log(f"{name}/{carve} {graph}: {js[-1] + 1}/{nq} ({time.time() - t:.0f}s)")
    arr = {"rows": np.stack(rows).astype(np.int32) if rows else np.zeros((0, len(cands)), np.int32),
           "pos": np.concatenate(pos) if pos else np.zeros((0, len(cands)), np.int32),
           "ptr": ptr, "bs": st["bs"][:nq].astype(np.int64), "fill": fills,
           "inc": np.asarray([x.size for x in inc[:nq]], np.int64)}
    for k, v in ledger.items():
        arr[k] = np.concatenate(v) if v else np.zeros(0, np.int64 if k == "kt_pos" else bool)
    held = arr["pos"] >= 0
    s = PG.summarize({"rows": arr["rows"], "held": held, "ptr": ptr}, cands)
    if limit:
        log(f"{name}/{carve} {graph}: development run on {nq} questions ({time.time() - t:.0f}s): every check "
            "passed; nothing written")
        return 0
    od = OUT / "stats" / graph
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{name}__{carve}.npz"
    tmp = od / f"{name}__{carve}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, p)
    rec = {"declared_in": DECLARED, "stage": "U2c", "dataset": name, "carve": carve, "graph": graph,
           "graph_detail": desc, "script_sha256": script_sha256(), "p1g_script_sha256": PG.script_sha256(),
           "p1g2_script_sha256": PG2.script_sha256(), "zp1h_sha256": P1H.script_sha256(),
           "p1e_script_sha256": K.script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
           "threads": threads, "questions": nq, "golds": int(ptr[-1]), "candidates": [c[0] for c in cands],
           "file_sha256": sha_file(p), "summary": s,
           "checks": {"base_seeds_step4d": True, "seeds_equal_g0": True, "b_equal_p1f": True,
                      **({"i_equal_g0_and_p1f": True, "rows_held_fill_equal_p1g": True, "today_equal_p1f": True}
                         if graph == "today" else {})},
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(od / f"{name}__{carve}.json", rec)
    log(f"{name}/{carve} {graph} ({nq} q, {time.time() - t0:.0f}s): written")
    return 0


# ── the verdict ──────────────────────────────────────────────────────────────


def load_stats(name, graph):
    """One dataset's arrays on a graph (metaqa and webqsp: today's under both), checked against their record."""
    g = graph if name in PASSAGE else "today"
    carve = carve_of(name)
    rp = OUT / "stats" / g / f"{name}__{carve}.json"
    if not rp.exists():
        return None
    rec = json.loads(rp.read_text(encoding="utf-8"))
    p = OUT / "stats" / g / f"{name}__{carve}.npz"
    if rec.get("script_sha256") != script_sha256() or sha_file(p) != rec["file_sha256"] or not all(rec["checks"].values()):
        raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
    with np.load(p) as z:
        a = {k: z[k] for k in z.files}
    if rec["candidates"] != [c[0] for c in PG.candidates()]:
        raise SystemExit(f"{rp}: not P1g's 805 candidates in order")
    if PG.summarize({"rows": a["rows"], "held": a["pos"] >= 0, "ptr": a["ptr"]}, PG.candidates()) != rec["summary"]:
        raise SystemExit(f"{p}: the arrays do not give the record's summary")
    return a, rec


def ordered(src):
    """Whether P1g's candidate src has a build order P1h's cut can read (no beam)."""
    return not src.split("+", 1)[1].startswith("beam")


def grid_candidates():
    """[(name, P1g's candidate it reads, gated, tau)]: P1g2's 1,603 in order, each followed by its taus below 1 from
    0.99 down when it has no beam."""
    out = []
    for nm, src, gated in PG2.candidates():
        out.append((nm, src, gated, 1.0))
        if ordered(src):
            out += [(f"{nm}@{tkey(t)}", src, gated, t) for t in sorted(TAUS, reverse=True) if t < 1.0]
    return out


def column(a, col, src, gated):
    """(rows, pos) of one P1g2 candidate: P1g's column, or the same fraction's none where the gate is shut."""
    r, p = a["rows"][:, col[src]], a["pos"][:, col[src]]
    if gated:
        open_ = np.asarray(a["fill"]) > 0
        gq = np.repeat(open_, np.diff(a["ptr"]))
        nn = PG2.none_of(src)
        r = np.where(open_, r, a["rows"][:, col[nn]])
        p = np.where(gq, p, a["pos"][:, col[nn]])
    return r.astype(np.int64), p.astype(np.int64)


def cut_column(a, rows, pos, tau):
    """P1h's cut at tau (it reads |B| and the pool's length only): (rows, held)."""
    b = np.asarray(a["bs"], np.int64)
    e = rows - b
    keep = P1H.keep_counts(b, e, tau)
    qi = np.repeat(np.arange(b.size), np.diff(a["ptr"]))
    held = (pos == 0) | ((pos >= 1) & (pos <= keep[qi])) if tau < 1.0 else pos >= 0
    return b + keep, held


def table_of(arrays, grid):
    """{candidate: {dataset: P1g's summary}} over grid, from each dataset's arrays."""
    out = {nm: {} for nm, *_ in grid}
    names = [c[0] for c in PG.candidates()]
    col = {nm: i for i, nm in enumerate(names)}
    for d, a in arrays.items():
        for nm, src, gated, tau in grid:
            r, p = column(a, col, src, gated)
            rows, held = cut_column(a, r, p, tau)
            out[nm][d] = PG.summarize({"rows": rows[:, None], "held": held[:, None], "ptr": a["ptr"]},
                                      [(nm, None, None)])[nm]
    return out


def ledger_of(a, a_today, src, gated, tau):
    """Each gold the chosen pool misses, by cause: TAU_CUT (the rule at tau = 1 holds it), BUDGET (some candidate on
    the graph holds it), UNREACHED (none does); the unreached split by the first stage's lists, one link and KIND_T."""
    names = [c[0] for c in PG.candidates()]
    col = {nm: i for i, nm in enumerate(names)}
    r, p = column(a, col, src, gated)
    _rows, held = cut_column(a, r, p, tau)
    full = p >= 0
    anyc = (a["pos"] >= 0).any(axis=1)
    miss = ~held
    cause = np.where(full, "TAU_CUT", np.where(anyc, "BUDGET", "UNREACHED"))
    ng = np.diff(a["ptr"])
    qi = np.repeat(np.arange(ng.size), ng)
    out = {"golds": int(held.size), "missed": int(miss.sum()),
           "questions": int(ng.size), "questions_missing_a_gold": int(np.unique(qi[miss]).size),
           "by_cause": {c: int((miss & (cause == c)).sum()) for c in CAUSES}}
    un = miss & (cause == "UNREACHED")
    out["unreached"] = {"listed": int((un & a["listed"]).sum()), "unlisted": int((un & ~a["listed"]).sum()),
                        "one_link": int((un & a["one_link"]).sum()), "kind_t_holds": int((un & (a["kt_pos"] >= 0)).sum()),
                        "in_frozen_I": int((un & a["in_I"]).sum())}
    if a_today is not None:
        reach0 = (a_today["pos"] >= 0).any(axis=1)
        out["unreached"]["today_graph_reaches"] = int((un & reach0).sum())
    worst = {}
    for c in CAUSES:
        worst[c] = int(np.unique(qi[miss & (cause == c)]).size)
    out["questions_by_cause"] = worst
    return out


def verdict_stage():
    grid = grid_candidates()
    order = [g[0] for g in grid]
    by = {g[0]: g for g in grid}
    arrays, recs, missing = {g: {} for g in GRAPHS}, {g: {} for g in GRAPHS}, []
    for g in GRAPHS:
        for d in SIX:
            x = load_stats(d, g)
            if x is None:
                missing.append(f"{d}/{g}")
            else:
                arrays[g][d], recs[g][d] = x
    if missing:
        raise SystemExit(f"INCOMPLETE: no stats for {missing}")
    gname, csha = chosen_graph()
    for d in PASSAGE:
        if recs["chosen"][d]["graph_detail"]["graph"] != gname:
            raise SystemExit(f"{d}: the chosen run is on {recs['chosen'][d]['graph_detail']['graph']}, not {gname}")
    w = F.weights()
    res = {}
    for g in GRAPHS:
        tab = table_of({d: arrays[g][d] for d in TRAIN5}, grid)
        tab = {c: tab[c] for c in order}
        verdict, pick, ceil, tgt, summ = PG.choose(tab, w)
        beside = table_of({"webqsp": arrays[g]["webqsp"]}, [by[pick], by[PG.TODAY]])
        _nm, src, gated, tau = by[pick]
        res[g] = {"verdict": verdict, "chosen": pick, "reads": {"candidate": src, "gated": gated, "tau": tau},
                  "ceilings": ceil, "targets": tgt, "cost": summ[pick]["cost"], "today_cost": summ[PG.TODAY]["cost"],
                  "chosen_s1sel": {d: tab[pick][d] for d in TRAIN5}, "today_s1sel": {d: tab[PG.TODAY][d] for d in TRAIN5},
                  "beside_webqsp_s1eval": {"chosen": beside[pick]["webqsp"], "today": beside[PG.TODAY]["webqsp"]},
                  "cheapest_at_tau1": min((summ[c]["cost"], i, c) for i, c in enumerate(order)
                                          if summ[c]["meets"] and by[c][3] == 1.0)[2]
                  if any(summ[c]["meets"] and by[c][3] == 1.0 for c in order) else None,
                  "summary": summ}
    _nm, src, gated, tau = by[res["chosen"]["chosen"]]
    ledger = {d: ledger_of(arrays["chosen"][d], arrays["today"][d], src, gated, tau) for d in SIX}
    p1g2 = json.loads(PG2.VERDICT.read_text(encoding="utf-8"))
    holds = {d: bool(res["chosen"]["verdict"] != "NONE_MEETS" and res["chosen"]["targets"][d] >= PG.TARGET
                     and res["chosen"]["chosen_s1sel"][d]["ALL"] >= PG.TARGET - 1e-9) for d in ("hotpotqa", "2wiki")}
    rec = {"declared_in": DECLARED, "stage": "U2c", "script_sha256": script_sha256(), "graph": gname,
           "u2b_choice_sha256": csha, "verdict": res["chosen"]["verdict"], "chosen": res["chosen"]["chosen"],
           "headline_graph": {"holds": holds, "graph": gname if all(holds.values()) else
                              {d: (gname if holds.get(d, True) else "the hyperlinks (named)") for d in PASSAGE}},
           "reads": res["chosen"]["reads"], "weights": w, "graphs": res, "ledger": ledger,
           "beside_p1g2": {"verdict": p1g2["verdict"], "chosen": p1g2["chosen"], "cost": p1g2["cost"]["chosen"]},
           "stats": {g: {d: {"file_sha256": recs[g][d]["file_sha256"], "graph": recs[g][d]["graph_detail"].get("graph")}
                         for d in SIX} for g in GRAPHS},
           "utc": utc()}
    write_json(OUT / "verdict.json", rec)
    (OUT / "verdict.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"U2c: {rec['verdict']} on {gname}: {rec['chosen']} (cost {res['chosen']['cost']}; today's graph "
        f"{res['today']['chosen']} {res['today']['cost']}; P1g2 {rec['beside_p1g2']['cost']})")
    return 0


def report_md(rec):
    L = [f"# U2c: {rec['verdict']} on {rec['graph']}", "",
         f"Chosen: `{rec['chosen']}` (P1g's `{rec['reads']['candidate']}`, gated {rec['reads']['gated']}, "
         f"tau {tkey(rec['reads']['tau'])}). Declared in {rec['declared_in']}.", "",
         f"Headline graph: {rec['headline_graph']['graph']} (holds hotpotqa and 2wiki at 0.95: "
         f"{rec['headline_graph']['holds']}).", ""]
    for g in GRAPHS:
        e = rec["graphs"][g]
        L += [f"## {g} graph: {e['verdict']}, `{e['chosen']}`, cost {e['cost']} (today's pool {e['today_cost']})", "",
              "| dataset | target | ALL | rows | rows p90 | today ALL | today rows |", "|---|---:|---:|---:|---:|---:|---:|"]
        for d in TRAIN5:
            c, t = e["chosen_s1sel"][d], e["today_s1sel"][d]
            L.append(f"| {d} | {e['targets'][d]} | {c['ALL']} | {c['rows']} | {c['rows_p90']} | {t['ALL']} | {t['rows']} |")
        c, t = e["beside_webqsp_s1eval"]["chosen"], e["beside_webqsp_s1eval"]["today"]
        L += [f"| webqsp (s1eval, beside) | - | {c['ALL']} | {c['rows']} | {c['rows_p90']} | {t['ALL']} | {t['rows']} |", ""]
    L += ["## Ledger: the chosen pool's misses on the chosen graph", "",
          "| dataset | golds | missed | TAU_CUT | BUDGET | UNREACHED | unreached listed | one link | KIND_T holds | "
          "today's graph reaches |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d, e in rec["ledger"].items():
        u = e["unreached"]
        L.append(f"| {d} | {e['golds']} | {e['missed']} | {e['by_cause']['TAU_CUT']} | {e['by_cause']['BUDGET']} | "
                 f"{e['by_cause']['UNREACHED']} | {u['listed']} | {u['one_link']} | {u['kind_t_holds']} | "
                 f"{u.get('today_graph_reaches', '-')} |")
    b = rec["beside_p1g2"]
    L += ["", f"Beside: P1g2 on today's graph without a cut: {b['verdict']} `{b['chosen']}`, cost {b['cost']}.", ""]
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _selftest():
    rng = np.random.default_rng(0)
    from mp_retrieval.m3b_pools import FamilyStore
    import zu2b
    # 1. the CSR store is from_graph's, field for field
    for n, m in ((1, 0), (7, 0), (40, 120), (300, 2000)):
        src, dst = rng.integers(0, n, m), rng.integers(0, n, m)
        ip, col = zu2b.undirected_csr(n, src[src != dst], dst[src != dst])
        hd = np.repeat(np.arange(n), np.diff(ip))
        up = hd < col
        a = FamilyStore.from_graph(SimpleNamespace(n_nodes=n, src=hd[up], dst=col[up].astype(np.int64), rel=None,
                                                   weight=None), "structural")
        b = csr_store(n, ip, col)
        for f in ("family", "n_nodes", "edges_stored"):
            assert getattr(a, f) == getattr(b, f), f
        for f in ("indptr", "col", "typed_indptr", "typed_col", "typed_rel", "typed_dir", "rel_count"):
            x, y = getattr(a, f), getattr(b, f)
            assert x.dtype == y.dtype and np.array_equal(x, y), (n, m, f)
        assert a.weight is None and b.weight is None
    # 2. the extra sets are P1g's, set for set; their ranks order them as D5b does
    n, dim = 400, 16
    V = rng.standard_normal((n, dim)).astype(np.float32)
    read = lambda r: V[np.asarray(r, np.int64)]  # noqa: E731
    adj = []
    for m in (900, 600, 0):
        ip, col = zu2b.undirected_csr(n, rng.integers(0, n, m), rng.integers(0, n, m))
        adj.append((ip.astype(np.int64), col))
    want = {nm for nm, _f, _p in PG.extras()}
    for trial in range(12):
        B = np.unique(rng.choice(n, int(rng.integers(1, 30)), replace=False)).astype(np.int64)
        w_B = rng.random(B.size)
        q = D5B.unit(rng.standard_normal(dim))
        sets, ranks, n1 = extra_sets_ranked(adj, B, w_B, q, read, 60, want)
        ref = PG.extra_sets(adj, B, w_B, q, read, 60, want)
        assert sets.keys() == ref.keys()
        for k in ref:
            assert np.array_equal(np.sort(sets[k]), np.sort(ref[k])), (trial, k)
            if k.startswith("cut_"):
                assert ranks[k].size == sets[k].size and np.unique(ranks[k]).size == ranks[k].size
        assert not np.isin(n1, B).any()
    # 3. the build-order places, against a direct build of every candidate's order
    cands = PG.candidates()
    fams = {nm: f for nm, f, _p in PG.extras()}
    for trial in range(25):
        B = np.unique(rng.choice(n, int(rng.integers(0, 20)), replace=False)).astype(np.int64)
        rest = np.setdiff1d(np.arange(n), B)
        full = rng.permutation(rest)[:int(rng.integers(0, rest.size))]
        base = int(rng.integers(0, full.size + 1))
        sets, ranks = {}, {}
        for nm, f, _p in PG.extras():
            X = np.unique(rng.choice(rest, int(rng.integers(0, 40)), replace=False)) if f != "NONE" else PG.EMPTY
            sets[nm] = X.astype(np.int64)
            if f == "CUT":
                ranks[nm] = rng.permutation(X.size).astype(np.int64) + 1
        golds = np.unique(rng.choice(n, int(rng.integers(1, 6)), replace=False)).astype(np.int64)
        rows, held, pos = place_question(B, full, base, golds, sets, ranks, cands, fams)
        for i, (_nm, f, ex) in enumerate(cands):
            k = min(full.size, int(np.rint(f * base)))
            fill = full[:k]
            X = sets[ex]
            outx = X[~np.isin(X, fill)]
            if fams[ex] == "CUT":
                outx = outx[np.argsort(ranks[ex][~np.isin(X, fill)], kind="stable")]
            build = list(fill) + list(outx)
            assert rows[i] == B.size + len(build)
            for gi, gd in enumerate(golds):
                want_p = 0 if gd in B else (build.index(gd) + 1 if gd in build else -1)
                if fams[ex] == "BEAM" and want_p > k:
                    want_p = int(UNORDERED)
                assert pos[gi, i] == want_p, (trial, i, gd, pos[gi, i], want_p)
    # 4. the grid, the gate, the cut and the summary against a direct count
    grid = grid_candidates()
    assert len([g for g in grid if g[3] == 1.0]) == len(PG2.candidates()) == 1603
    assert all(ordered(src) for _n, src, _g, t in grid if t < 1.0)
    assert not any(ordered(src) for src in [c[0] for c in PG.candidates()] if "+beam" in src)
    nq = 30
    ptr = np.concatenate([[0], np.cumsum(rng.integers(1, 4, nq))]).astype(np.int64)
    ng = int(ptr[-1])
    C = len(cands)
    bsz = rng.integers(0, 6, nq)
    e = rng.integers(0, 50, (nq, C))
    a = {"rows": (bsz[:, None] + e).astype(np.int32), "ptr": ptr, "bs": bsz.astype(np.int64),
         "fill": rng.integers(0, 3, nq)}
    qi = np.repeat(np.arange(nq), np.diff(ptr))
    pos = rng.integers(-1, 60, (ng, C))
    pos = np.where(pos > e[qi], -1, pos)
    pos[(pos == 0) & (bsz[qi] == 0)[:, None]] = -1
    a["pos"] = pos.astype(np.int32)
    names = [c[0] for c in cands]
    col = {nm: i for i, nm in enumerate(names)}
    pick = [g for g in grid if g[0] in ("g:f1.5+cut_cos_5000@0.9", "f1+none", "f2+none@0.5", "f4+cut_res_50@0.99")]
    assert len(pick) == 4
    tab = table_of({"x": a}, pick)
    for nm, src, gated, tau in pick:
        allc, rec_, rws = 0, 0.0, []
        for j in range(nq):
            s = src if (not gated or a["fill"][j] > 0) else PG2.none_of(src)
            c = col[s]
            b, ee = int(bsz[j]), int(a["rows"][j, c]) - int(bsz[j])
            kk = int(P1H.keep_counts(np.asarray([b]), np.asarray([ee]), tau)[0])
            gp = a["pos"][ptr[j]:ptr[j + 1], c]
            h = (gp == 0) | ((gp >= 1) & (gp <= kk)) if tau < 1.0 else gp >= 0
            allc += int(h.all())
            rec_ += float(h.mean())
            rws.append(b + kk)
        assert tab[nm]["x"]["ALL"] == round(allc / nq, 4), nm
        assert tab[nm]["x"]["recall"] == round(rec_ / nq, 4), nm
        assert tab[nm]["x"]["rows"] == round(float(np.mean(rws)), 1), nm
    # 5. the ledger's causes on a toy
    led = ledger_of({**a, "listed": np.ones(ng, bool), "one_link": np.zeros(ng, bool), "kt_pos": np.full(ng, -1),
                     "in_I": np.zeros(ng, bool)}, None, "f1+none", False, 0.5)
    assert sum(led["by_cause"].values()) == led["missed"]
    print("zu2c selftest: OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("stats", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--graph", choices=GRAPHS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return 0
    if a.stage == "stats":
        if not a.dataset or not a.graph:
            raise SystemExit("stats needs DATASET and --graph")
        return stats_stage(a.dataset, a.graph, a.threads, a.limit)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
