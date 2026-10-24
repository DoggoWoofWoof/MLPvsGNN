"""U2m (c) and (e): one construction and one walk recipe for all six (docs/U2M_POOLS_THREE_FAMILIES.md). Pools only:
nothing trains, nothing is fit, and no construction, recipe, order, budget or cut reads a gold.

    python outputs/mp_unified/zu2m.py stats DATASET [--threads 2] [--limit Q] [--shard I/K]
    python outputs/mp_unified/zu2m.py merge DATASET --parts K
    python outputs/mp_unified/zu2m.py verdict
    python outputs/mp_unified/zu2m.py --selftest

stats    One dataset's selection carve (s1sel; webqsp's s1eval, beside) on U2e's graph. Each question walks once (P1d's
         five sources and P1e's steered walks). Then, for each of the 12 constructions (3 bases x 4 expansions over the
         three families) and the dataset's own 13 Sep construction, and for each of the 3 recipes (kind_t, rrf5t,
         both), it builds U2e's 14 orders and records their lengths and every gold's place. The own construction with
         kind_t is checked against U2e's filed arrays, question for question (section 8). --limit Q: the first Q
         questions, every check, nothing written and no number logged. --shard I/K: the I-th of K contiguous question
         ranges, into outputs/u2m/stats/parts/ (outputs/mp_unified/zu2m_modal.py runs them on Modal).
merge    Join a dataset's K parts into the file one run over every question writes.
verdict  Frontiers C (kind_t) and E (the kind-free recipes) over the 498,960 universal candidates, the verdicts (c) and
         (e), and what is carried forward (section 7). Writes outputs/u2m/verdict_ce.json and verdict_ce.md.
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

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zu2e as U2E  # noqa: E402  (U2e's orders, places, budgets, cut, frontier and ledger, unchanged)

U2C, PG, F, P1H, D5, D5B = U2E.U2C, U2E.PG, U2E.F, U2E.P1H, U2E.D5, U2E.D5B
log, utc, write_json, sha_file = U2E.log, U2E.utc, U2E.write_json, U2E.sha_file
DECLARED = "docs/U2M_POOLS_THREE_FAMILIES.md"
OUT = ROOT / "outputs" / "u2m"
SIX, TRAIN5 = U2E.SIX, U2E.TRAIN5
NAMES, MS, M_GRID, TAUS, LEVELS = U2E.NAMES, U2E.MS, U2E.M_GRID, U2E.TAUS, U2E.LEVELS
NO = len(NAMES)                                    # U2e's 14 orders
EMPTY = PG.EMPTY
BASES = (("rrf50", "equal_rrf_budget_50"), ("rrf200", "equal_rrf_budget_200"), ("dense200", "dense_top200"))
H1 = {"name": "h1_c25", "hops": 1, "per_seed_cap": 25}
H2 = {"name": "h2_c25", "hops": 2, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 2000}
H3 = {"name": "h3_c25_v2000", "hops": 3, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 2000}
EXPANSIONS = (("none", None), ("FULL:h1_c25", H1), ("FULL:h2_c25", H2), ("FULL:h3_c25_v2000", H3))
GRID = tuple((f"{bs}+{en}", bi, ei) for bi, (bs, _bp) in enumerate(BASES) for ei, (en, _st) in enumerate(EXPANSIONS))
RECIPES = ("kind_t", "rrf5t", "both")
R = len(RECIPES)
KIND_FREE = (1, 2)                                 # rrf5t, both
U2E_PICK = "kt_cos:ada_f12_k0@0.85"                # U2e's filed pick (outputs/u2e/verdict.json), on the own construction
VERDICTS_C = ("UNIFIED_RAISED", "UNIFIED_HOLDS", "UNIFIED_AT_A_PRICE", "NOT_UNIFIED")
VERDICTS_E = ("KIND_FREE_NO_PRICE", "KIND_FREE_AT_A_PRICE", "KIND_FREE_TOO_DEAR")
FAMILIES_OF_ORDERS = ("kt", "kt_n1", "cos", "kt_cos", "kt_cos_r0")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def own_name(construction):
    """A 13 Sep construction's name in section 4's terms, e.g. rrf50+STRUCT:h1_c25."""
    short = {bp: bs for bs, bp in BASES}[construction["base_pool"]]
    if construction["regime"] == "RETRIEVAL":
        return f"{short}+none"
    return f"{short}+{construction['regime']}:{construction['setting']['name']}"


def cells_of(construction):
    """(the cells' names: the 12 grid constructions, then the own one when it is not among them; the own one's index)."""
    names = [g[0] for g in GRID]
    own = own_name(construction)
    if own in names:
        return names, names.index(own)
    return names + [own], len(names)


def column(k, r, o):
    """The column of cell k, recipe r and U2e order o in the per-gold and per-question arrays."""
    return (k * R + r) * NO + o


def fill_of(order_out, n_i, n_b):
    """Today's fill (U2e's definition with the recipe's order): its first max(0, |I| - |B|) nodes outside B."""
    return int(np.asarray(order_out)[:max(0, int(n_i) - int(n_b))].size)


class Rows:
    """One question's node rows, read once from DenseNodes and served to every order (section 8). verify: compare
    every served read with DenseNodes.read (development runs)."""

    def __init__(self, nodes, verify=False):
        self.nodes, self.verify = nodes, verify
        self.blocks = []                            # [(sorted ids, their rows)], disjoint

    @property
    def ids(self):
        return np.sort(np.concatenate([b[0] for b in self.blocks])) if self.blocks else EMPTY

    def add(self, ids):
        new = np.setdiff1d(np.asarray(ids, np.int64), self.ids)
        if new.size:
            self.blocks.append((new, np.asarray(self.nodes.read(new))))

    def read(self, rows):
        rows = np.asarray(rows, np.int64)
        if not rows.size or not self.blocks:
            out = self.nodes.read(rows)
            if rows.size:
                raise SystemExit("a node row was asked for that this question did not read")
            return out
        out = np.empty((rows.size,) + self.blocks[0][1].shape[1:], self.blocks[0][1].dtype)
        found = np.zeros(rows.size, bool)
        for ids, mat in self.blocks:
            k = np.minimum(np.searchsorted(ids, rows), ids.size - 1)
            hit = ids[k] == rows
            out[hit] = mat[k[hit]]
            found |= hit
        if not found.all():
            raise SystemExit("a node row was asked for that this question did not read")
        if self.verify:
            want = self.nodes.read(rows)
            if out.dtype != want.dtype or not out.flags.c_contiguous or not np.array_equal(out, want):
                raise SystemExit("a served row is not DenseNodes' bytes")
        return out


# ── stats (the laptop, or Modal in shards) ───────────────────────────────────

PER_QUESTION = ("bs", "inc", "n_i", "fill", "lens", "n_u", "n_n1", "n_r0", "n_n2", "taken", "hubs")
PER_GOLD = ("pos", "in_I", "one_link")
ARRAYS = PER_QUESTION + PER_GOLD + ("ptr",)
HEAD_KEYS = ("declared_in", "stage", "dataset", "carve", "graph_detail", "script_sha256", "zu2e_sha256", "zu2c_sha256",
             "u2e_stats_sha256", "p1e_script_sha256", "freeze_RECORD_SHA256", "threads", "cells", "own_cell",
             "recipes", "orders", "bases", "expansions")


def stats_stage(name, threads=2, limit=0, shard=None):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if D.SOURCES[4] != "R0" or F.CONSTRUCTION != "KIND_T":
        raise SystemExit("P1d's fifth source is not R0, or P1f's construction is not KIND_T")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    x2e = U2E.load_u2e(name)
    if x2e is None:
        raise SystemExit(f"{name}: U2e's filed arrays are not under outputs/u2e/stats")
    e2, r2e = x2e
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    cfg_set = {s["name"]: s for s in op.cfg_h["graph_regimes"]["expansion_settings"]}
    h3s = [c["setting"] for c in op.construction.values() if c.get("setting") and c["setting"]["name"] == H3["name"]]
    if cfg_set.get("h1_c25") != H1 or cfg_set.get("h2_c25") != H2 or not h3s or any(s != H3 for s in h3s):
        raise SystemExit("section 4's hop settings are not the filed ones")
    full_fams = list(op.cfg_h["graph_regimes"]["FULL"])
    if full_fams != list(S4.FAMILIES):
        raise SystemExit("FULL is not the three families in their order")
    cells, own_k = cells_of(construction)
    n = int(ds.n_nodes)
    g = U2C.graph_csr(name, "chosen", n)
    stores, union, desc = stores0, union0, {"graph": "cached (the KB)"}
    if g is not None:
        ip, col, desc = g
        swap = U2C.csr_store(n, ip, col)
        del ip, col
        stores = dict(stores0)
        if "structural" in families:
            stores["structural"] = swap
        union = S4B.RegimeUnion(stores, families, n) if families else union0
        load0 = S4.m3b_pools.load_or_build_store

        def load(ds_, family, cache_dir):
            if family == "structural":
                if ds_.name != name:
                    raise SystemExit(f"{ds_.name}: U2m swaps {name}'s graph only")
                return swap
            return load0(ds_, family, cache_dir)

        S4.m3b_pools.load_or_build_store = load
    if desc != r2e["graph_detail"]:
        raise SystemExit(f"{name}: the graph is not the one U2e's filed run read")
    pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c == carve]
    (_carve, pop), = pops
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
    own_bases = op.m3b_compile.base_rows(construction, prep0.dense_ids, prep0.splade_ids, op.m3a, op.m3b_contract,
                                         constant)
    _U0, _BQ, bs_own, _inc0, _f = E.union_pools(name, carve, pop, prep0, own_bases)     # base, seeds: step 4d's
    del _U0
    if any(not np.array_equal(np.asarray(a, np.int64), np.asarray(b, np.int64)) for a, b in zip(prep.seeds, prep0.seeds)):
        raise SystemExit(f"{name}/{carve}: the seeds on the graph are not G0's")
    base_rows = [op.m3b_compile.base_rows({"base_pool": bp}, prep0.dense_ids, prep0.splade_ids, op.m3a,
                                          op.m3b_contract, constant) for _bs, bp in BASES]
    own_b = [bp for _bs, bp in BASES].index(construction["base_pool"])
    if any(not np.array_equal(np.union1d(np.asarray(x, np.int64), np.asarray(s, np.int64)), np.asarray(y, np.int64))
           for x, s, y in zip(base_rows[own_b], prep0.seeds, bs_own)):
        raise SystemExit(f"{name}/{carve}: the own base, built by name, is not step 4d's")
    del own_bases, bs_own
    inc_own = [np.asarray(p, np.int64) for p in prep.pools]
    linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [EMPTY] * len(prep.seeds)
    log(f"{name}/{carve}: prepared; cells {cells} (own {cells[own_k]}) ({time.time() - t0:.0f}s)")

    cache = {}
    stores_all, unions = G.open_unions(cache, op, name, union)
    adj = [(np.ascontiguousarray(stores_all[f].indptr, np.int64), np.ascontiguousarray(stores_all[f].col))
           for f in S4.FAMILIES]
    fams_full = [stores_all[f] for f in full_fams]
    n_st = int(unions["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    nq = min(limit, len(prep.seeds)) if limit else len(prep.seeds)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(e2["ptr"][:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: U2e's golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    own_cols = [column(own_k, 0, o) for o in range(NO)]
    pq = {k: [] for k in PER_QUESTION}
    pg = {k: [] for k in PER_GOLD}
    t = time.time()
    done = 0
    for lo in range(q0, q1, S4.BATCH):
        js = list(range(lo, min(q1, lo + S4.BATCH)))
        srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
        steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
        for i, j in enumerate(js):
            src, _nn, (L, Ls) = srcs[i]
            route = D.routes(L, Ls)
            knd = K.kind(Ls, prep.seeds[j], nonly)
            out, p1d = K.p1e_orders(src, steered[i], route, knd, n_st, constant)
            kt = np.asarray(out[F.CONSTRUCTION], np.int64)
            want_kt = p1d["ROUTEst"] if knd == "entity" else out["RRF5T"]
            if not np.array_equal(kt, np.asarray(want_kt, np.int64)):
                raise SystemExit(f"{name}/{carve} question {j}: KIND_T is not ROUTEst or RRF5T by the kind")
            rec = [kt, np.asarray(out["RRF5T"], np.int64),
                   np.asarray(D.rrf([p1d["ROUTEst"], out["RRF5T"]], constant), np.int64)]
            seeds = np.asarray(prep.seeds[j], np.int64)
            q = D5B.unit(np.asarray(prep.qemb[j], np.float64))
            w_all = K.steer(prep, j, constant)
            rows = Rows(nodes, verify=bool(limit))
            gj = golds[j]
            g0, g1 = int(ptr[j]), int(ptr[j + 1])
            per_b = []
            for bi in range(len(BASES)):
                B = np.unique(np.union1d(np.asarray(base_rows[bi][j], np.int64), seeds))
                n1, _cnt, _ws, _up, _inv = D5B.n1_table(adj, B, D5.weights_of(w_all, B))
                per_b.append({"B": B, "n1": n1, "base": np.asarray(base_rows[bi][j], np.int64),
                              "r0": F.outside(src[4][0], B)})
            rows.add(np.concatenate([x["n1"] for x in per_b]))
            for x in per_b:
                n1 = x["n1"]
                x["cos1"] = D5B.unit(rows.read(n1)) @ q if n1.size else np.empty(0, np.float32)
                x["cr"] = D5B.ranks_desc(x["cos1"], n1) if n1.size else EMPTY
                x["n2"], x["first"], x["taken"], x["hubs"] = U2E.second_link(adj, n1, x["cr"], x["B"])
            exp = [None] + [S4.m3b_pools.expand_hops(seeds, fams_full, st) for _en, st in EXPANSIONS[1:]]
            inc = []
            for k in range(len(cells)):
                if k < len(GRID):
                    _nm, bi, ei = GRID[k]
                    pool, _added = op.m3b_compile.build_pool(per_b[bi]["base"], seeds, exp[ei])
                    inc.append((bi, np.asarray(pool, np.int64)))
                else:
                    inc.append((own_b, inc_own[j]))
            if not np.array_equal(inc[own_k][1], inc_own[j]):
                raise SystemExit(f"{name}/{carve} question {j}: the own construction's I is not U2e's")
            for k, (bi, I) in enumerate(inc):
                if not np.isin(seeds, per_b[bi]["B"]).all() or not np.isin(per_b[bi]["B"], I).all() or \
                        (k < len(GRID) and GRID[k][2] == 0 and not np.array_equal(I, per_b[bi]["B"])):
                    raise SystemExit(f"{name}/{carve} question {j}: cell {cells[k]}'s B or I is malformed")
            rows.add(np.concatenate(rec + [src[4][0]] + [I for _bi, I in inc] + [x["n2"] for x in per_b]))
            lens = np.zeros(len(cells) * R * NO, np.int32)
            n_u = np.zeros(len(cells) * R * len(MS), np.int64)
            fill = np.zeros(len(cells) * R, np.int64)
            pos = np.zeros((gj.size, len(cells) * R * NO), np.int32)
            for k, (bi, I) in enumerate(inc):
                x = per_b[bi]
                B = x["B"]
                iout = F.outside(I, B)
                in_b = np.isin(gj, B)
                for r in range(R):
                    full = F.outside(rec[r], B)
                    fill[k * R + r] = fill_of(full, I.size, B.size)
                    ords, sizes, U, mem = U2E.question_orders(full, x["n1"], x["cr"], x["cos1"], x["r0"], iout,
                                                              x["n2"], x["first"], rows.read, q, constant, D.rrf)
                    if not U2E.orders_exact(ords, B, full, x["n1"], U, mem):
                        raise SystemExit(f"{name}/{carve} question {j}: an order of {cells[k]}/{RECIPES[r]} is not "
                                         "exactly its sources' nodes")
                    for o, nm in enumerate(NAMES):
                        c = column(k, r, o)
                        lens[c] = ords[nm].size
                        pos[:, c] = U2E.places(gj, in_b, ords[nm])
                    n_u[(k * R + r) * len(MS):(k * R + r + 1) * len(MS)] = [sizes[m] for m in MS]
            in_I = np.stack([np.isin(gj, I) for _bi, I in inc], axis=1)
            one = np.stack([np.isin(gj, x["n1"]) for x in per_b], axis=1)
            xo = per_b[own_b]
            n2o = [int((xo["first"] < m).sum()) for m in M_GRID]
            if not (xo["B"].size == int(e2["bs"][j]) and fill[own_k * R] == int(e2["fill"][j])
                    and inc_own[j].size == int(e2["inc"][j]) and np.array_equal(lens[own_cols], e2["lens"][j])
                    and np.array_equal(pos[:, own_cols], e2["pos"][g0:g1])
                    and xo["n1"].size == int(e2["n_n1"][j]) and xo["r0"].size == int(e2["n_r0"][j])
                    and F.outside(inc_own[j], xo["B"]).size == int(e2["n_i"][j])
                    and np.array_equal(n2o, e2["n_n2"][j])
                    and np.array_equal(n_u[own_k * R * len(MS):(own_k * R + 1) * len(MS)], e2["n_u"][j])
                    and xo["taken"] == int(e2["taken"][j]) and xo["hubs"] == int(e2["hubs"][j])
                    and np.array_equal(in_I[:, own_k], e2["in_I"][g0:g1])
                    and np.array_equal(one[:, own_b], e2["one_link"][g0:g1])):
                raise SystemExit(f"{name}/{carve} question {j}: the own construction with kind_t is not U2e's")
            pq["bs"].append([per_b[bi]["B"].size for bi, _I in inc])
            pq["inc"].append([I.size for _bi, I in inc])
            pq["n_i"].append([F.outside(I, per_b[bi]["B"]).size for bi, I in inc])
            pq["fill"].append(fill)
            pq["lens"].append(lens)
            pq["n_u"].append(n_u)
            pq["n_n1"].append([x["n1"].size for x in per_b])
            pq["n_r0"].append([x["r0"].size for x in per_b])
            pq["n_n2"].append([int((x["first"] < m).sum()) for x in per_b for m in M_GRID])
            pq["taken"].append([x["taken"] for x in per_b])
            pq["hubs"].append([x["hubs"] for x in per_b])
            pg["pos"].append(pos)
            pg["in_I"].append(in_I)
            pg["one_link"].append(one)
            done += 1
            if done % 100 == 0:
                log(f"{name}/{carve}: {done}/{q1 - q0} ({time.time() - t:.0f}s)")
    arr = arrays_of(pq, pg, ptr[q0:q1 + 1], len(cells))
    head = {"declared_in": DECLARED, "stage": "U2m", "dataset": name, "carve": carve, "graph_detail": desc,
            "script_sha256": script_sha256(), "zu2e_sha256": U2E.script_sha256(), "zu2c_sha256": U2C.script_sha256(),
            "u2e_stats_sha256": r2e["file_sha256"], "p1e_script_sha256": K.script_sha256(),
            "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "threads": threads, "cells": cells,
            "own_cell": own_k, "recipes": list(RECIPES), "orders": list(NAMES),
            "bases": [list(b) for b in BASES], "expansions": [[en, st] for en, st in EXPANSIONS]}
    checks = {"graph_equal_u2e": True, "seeds_equal_g0": True, "own_base_equal_step4d": True,
              "kind_t_by_kind": True, "own_I_equal_u2e": True, "cells_well_formed": True, "orders_exact": True,
              "own_kind_t_equal_u2e": True, "rows_served_verified": bool(limit)}
    if limit:
        log(f"{name}/{carve}: development run on questions {q0}-{q1 - 1} of {nq} ({time.time() - t:.0f}s): "
            "every check passed, every served row verified; nothing written")
        return 0
    if shard:
        od = OUT / "stats" / "parts"
        od.mkdir(parents=True, exist_ok=True)
        stem = f"{name}__{carve}__{shard[0]}of{shard[1]}"
        p = od / f"{stem}.npz"
        tmp = od / f"{stem}.tmp.npz"
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, p)
        write_json(od / f"{stem}.json", {**head, "part": shard[0], "parts": shard[1], "q0": q0, "q1": q1,
                                         "questions_all": nq, "file_sha256": sha_file(p), "checks": checks,
                                         "seconds": round(time.time() - t0, 1), "utc": utc()})
        log(f"{name}/{carve} part {shard[0]}/{shard[1]} (questions {q0}-{q1 - 1}, {time.time() - t0:.0f}s): written")
        return 0
    return write_stats(name, carve, arr, head, checks, round(time.time() - t0, 1))


def arrays_of(pq, pg, ptr, nk):
    nc = nk * R * NO
    a = {"bs": np.asarray(pq["bs"], np.int64).reshape(-1, nk), "inc": np.asarray(pq["inc"], np.int64).reshape(-1, nk),
         "n_i": np.asarray(pq["n_i"], np.int64).reshape(-1, nk),
         "fill": np.asarray(pq["fill"], np.int64).reshape(-1, nk * R),
         "lens": np.asarray(pq["lens"], np.int32).reshape(-1, nc),
         "n_u": np.asarray(pq["n_u"], np.int64).reshape(-1, nk * R * len(MS)),
         "n_n1": np.asarray(pq["n_n1"], np.int64).reshape(-1, len(BASES)),
         "n_r0": np.asarray(pq["n_r0"], np.int64).reshape(-1, len(BASES)),
         "n_n2": np.asarray(pq["n_n2"], np.int64).reshape(-1, len(BASES) * len(M_GRID)),
         "taken": np.asarray(pq["taken"], np.int64).reshape(-1, len(BASES)),
         "hubs": np.asarray(pq["hubs"], np.int64).reshape(-1, len(BASES)),
         "pos": np.concatenate(pg["pos"]) if pg["pos"] else np.zeros((0, nc), np.int32),
         "in_I": np.concatenate(pg["in_I"]) if pg["in_I"] else np.zeros((0, nk), bool),
         "one_link": np.concatenate(pg["one_link"]) if pg["one_link"] else np.zeros((0, len(BASES)), bool),
         "ptr": np.asarray(ptr, np.int64)}
    return {k: a[k] for k in ARRAYS}


def summary_of(a, cells):
    """Per cell and recipe: each order's whole reach (ALL at full length, tau = 1) and mean length; per cell |B|, |I|
    and today's fill; per base the sources' mean sizes."""
    ptr = a["ptr"]
    ng = np.diff(ptr)
    qi = np.repeat(np.arange(ng.size), ng)
    has = ng > 0
    out = {"cells": {}}
    for k, cn in enumerate(cells):
        e = {"B": round(float(a["bs"][:, k].mean()), 1), "I": round(float(a["inc"][:, k].mean()), 1), "recipes": {}}
        for r, rn in enumerate(RECIPES):
            o_ = {}
            for o, nm in enumerate(NAMES):
                c = column(k, r, o)
                hits = np.bincount(qi, weights=(a["pos"][:, c] >= 0).astype(np.float64), minlength=ng.size)
                o_[nm] = {"ALL": round(float((has & (hits == ng)).mean()), 4),
                          "length": round(float(a["lens"][:, c].mean()), 1)}
            e["recipes"][rn] = {"fill": round(float(a["fill"][:, k * R + r].mean()), 1), "orders": o_}
        out["cells"][cn] = e
    out["bases"] = {bs: {"N1": round(float(a["n_n1"][:, bi].mean()), 1), "R0": round(float(a["n_r0"][:, bi].mean()), 1),
                         **{f"N2_{m}": round(float(a["n_n2"][:, bi * len(M_GRID) + mi].mean()), 1)
                            for mi, m in enumerate(M_GRID)},
                         "parents_taken": round(float(a["taken"][:, bi].mean()), 2),
                         "hubs": round(float(a["hubs"][:, bi].mean()), 3)}
                    for bi, (bs, _bp) in enumerate(BASES)}
    return out


def write_stats(name, carve, arr, head, checks, seconds, extra=None):
    od = OUT / "stats"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{name}__{carve}.npz"
    tmp = od / f"{name}__{carve}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, p)
    nq = int(arr["bs"].shape[0])
    rec = {**head, "questions": nq, "golds": int(arr["ptr"][-1]), "file_sha256": sha_file(p),
           "summary": summary_of(arr, head["cells"]), "checks": checks, "seconds": seconds, **(extra or {}),
           "utc": utc()}
    write_json(od / f"{name}__{carve}.json", rec)
    log(f"{name}/{carve} ({nq} q, {seconds:.0f}s): written")
    return 0


def merge_arrays(parts):
    """The arrays of one run over every question, from its shards' arrays in order."""
    out = {k: np.concatenate([p[k] for p in parts]) for k in PER_QUESTION + PER_GOLD}
    ptr = [np.asarray([0], np.int64)]
    for a, p in enumerate(parts):
        q = np.asarray(p["ptr"], np.int64)
        if q.size != p["bs"].shape[0] + 1 or int(q[0]) != int(ptr[-1][-1]):
            raise SystemExit(f"part {a}: its ptr does not continue the previous part's")
        ptr.append(q[1:])
    out["ptr"] = np.concatenate(ptr)
    return {k: out[k] for k in ARRAYS}


def merge_stage(name, k):
    carve = U2C.carve_of(name)
    od = OUT / "stats" / "parts"
    recs, parts = [], []
    for i in range(k):
        stem = f"{name}__{carve}__{i}of{k}"
        rp = od / f"{stem}.json"
        if not rp.exists():
            raise SystemExit(f"{rp}: missing")
        r = json.loads(rp.read_text(encoding="utf-8"))
        p = od / f"{stem}.npz"
        if r.get("script_sha256") != script_sha256() or sha_file(p) != r["file_sha256"] or not all(
                v for c, v in r["checks"].items() if c != "rows_served_verified"):
            raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
        with np.load(p) as z:
            parts.append({a: z[a] for a in z.files})
        recs.append(r)
    head = {a: recs[0][a] for a in HEAD_KEYS}
    nq = recs[0]["questions_all"]
    for i, r in enumerate(recs):
        if {a: r[a] for a in HEAD_KEYS} != head or r["checks"] != recs[0]["checks"] or r["questions_all"] != nq:
            raise SystemExit(f"part {i}: another script, freeze, graph, thread count, check set, cell set or carve")
        if (r["part"], r["parts"]) != (i, k) or (r["q0"], r["q1"]) != U2C.shard_range(nq, i, k):
            raise SystemExit(f"part {i}: not the {i}-th of {k} question ranges")
    arr = merge_arrays(parts)
    e2 = U2E.load_u2e(name)[0]
    if arr["bs"].shape[0] != nq or not np.array_equal(e2["ptr"][:nq + 1], arr["ptr"]):
        raise SystemExit(f"{name}/{carve}: the merged parts do not cover U2e's questions and golds")
    return write_stats(name, carve, arr, head, recs[0]["checks"], round(sum(r["seconds"] for r in recs), 1),
                       {"shards": k, "part_sha256": [r["file_sha256"] for r in recs],
                        "seconds_longest_part": max(r["seconds"] for r in recs)})


# ── the verdict (the laptop) ─────────────────────────────────────────────────


def load_u2m(name):
    carve = U2C.carve_of(name)
    rp = OUT / "stats" / f"{name}__{carve}.json"
    if not rp.exists():
        return None
    rec = json.loads(rp.read_text(encoding="utf-8"))
    p = OUT / "stats" / f"{name}__{carve}.npz"
    if rec.get("script_sha256") != script_sha256() or sha_file(p) != rec["file_sha256"] or not all(
            v for c, v in rec["checks"].items() if c != "rows_served_verified"):
        raise SystemExit(f"{rp}: not this script's, not its file, or a check failed")
    with np.load(p) as z:
        a = {k: z[k] for k in z.files}
    if rec["cells"][:len(GRID)] != [g[0] for g in GRID] or rec["recipes"] != list(RECIPES) or \
            rec["orders"] != list(NAMES) or summary_of(a, rec["cells"]) != rec["summary"]:
        raise SystemExit(f"{p}: not this script's cells, recipes or orders, or the arrays do not give its summary")
    return a, rec


def candidate_name(cell, r, o, bn, t):
    return f"{cell}/{RECIPES[r]}/{NAMES[o]}:{bn}@{P1H.key(t)}"


def index_grid(ks, rs):
    """[(k, r, o, budget index, tau)] of cells ks x recipes rs in section 6's order (U2e's order within a recipe)."""
    nb = len(U2E.budgets())
    taus = sorted(TAUS, reverse=True)
    return [(k, r, o, b, t) for k in ks for r in rs for o in range(NO) for b in range(nb) for t in taus]


def table_arrays(a, ks, rs):
    """(ALL, rows) of every candidate of cells ks x recipes rs on one dataset's arrays, in index_grid's order, each
    rounded as P1g's summarize rounds it, from U2e's budget, cut and evaluation (section 6)."""
    ptr = a["ptr"]
    ng = np.diff(ptr)
    nq = ng.size
    qi = np.repeat(np.arange(nq), ng)
    has = ng > 0
    bud = U2E.budgets()
    taus = sorted(TAUS, reverse=True)
    n = len(ks) * len(rs) * NO * len(bud) * len(taus)
    ALL, ROWS = np.empty(n), np.empty(n)
    i = 0
    for k in ks:
        b = np.asarray(a["bs"][:, k], np.int64)
        for r in rs:
            fl = a["fill"][:, k * R + r]
            kq = [U2E.budget_k(kind, prm, fl) for _bn, kind, prm in bud]
            for o in range(NO):
                c = column(k, r, o)
                ln = np.asarray(a["lens"][:, c], np.int64)
                p = a["pos"][:, c]
                for kb in kq:
                    kk = np.minimum(ln, kb)
                    for t in taus:
                        keep = P1H.keep_counts(b, kk, t)
                        rows = b + keep
                        held = (p == 0) | ((p >= 1) & (p <= keep[qi]))
                        hits = np.bincount(qi, weights=held.astype(np.float64), minlength=nq)
                        ALL[i] = round(float((has & (hits == ng)).mean()), 4)
                        ROWS[i] = round(float(rows.astype(np.float64).mean()), 1)
                        i += 1
    return ALL, ROWS


def held_of(a, k, r, o, b, t):
    """(rows, held) of one candidate on one dataset's arrays: U2e's evaluate on cell k and recipe r's column."""
    bn, kind, prm = U2E.budgets()[b]
    sub = {"bs": a["bs"][:, k], "lens": a["lens"][:, [column(k, r, o)]], "pos": a["pos"][:, [column(k, r, o)]],
           "ptr": a["ptr"]}
    return U2E.evaluate(sub, 0, U2E.budget_k(kind, prm, a["fill"][:, k * R + r]), t)


def pick_of(pts, cost_r, cap):
    """(level, index) of the frontier's point at the highest level within the cap, or (None, None)."""
    within = [(t, k) for t, _tgt, k in pts if k is not None and cost_r[k] <= cap + 1e-9]
    if not within:
        return None, None
    return max(within, key=lambda x: x[0])


def decide(l_c, l_e, incumbent_level):
    """Section 7: (verdict (c), verdict (e), what is carried: 'E', 'C' or 'U2e')."""
    if l_c is None:
        vc = "NOT_UNIFIED"
    elif l_c > incumbent_level + 1e-9:
        vc = "UNIFIED_RAISED"
    elif abs(l_c - incumbent_level) < 1e-9:
        vc = "UNIFIED_HOLDS"
    else:
        vc = "UNIFIED_AT_A_PRICE"
    if l_e is not None and (l_c is None or l_e >= l_c - 1e-9):
        ve = "KIND_FREE_NO_PRICE"
    elif l_e is not None and abs(l_e - (l_c - PG.TOL)) < 1e-9:
        ve = "KIND_FREE_AT_A_PRICE"
    else:
        ve = "KIND_FREE_TOO_DEAR"
    carried = "E" if ve != "KIND_FREE_TOO_DEAR" else ("C" if vc != "NOT_UNIFIED" else "U2e")
    return vc, ve, carried


def verdict_stage():
    t0 = time.time()
    am, rm, missing = {}, {}, []
    for d in SIX:
        x = load_u2m(d)
        if x is None:
            missing.append(d)
        else:
            am[d], rm[d] = x
    if missing:
        raise SystemExit(f"INCOMPLETE: no U2m stats for {missing}")
    v2e = json.loads((U2E.OUT / "verdict.json").read_text(encoding="utf-8"))
    if v2e.get("script_sha256") != U2E.script_sha256() or v2e["pick"] != U2E_PICK:
        raise SystemExit("outputs/u2e/verdict.json: not the committed zu2e.py's verdict, or not the pick section 6 names")
    cap, inc_level, inc_cost = float(v2e["cap"]), float(v2e["level"]), float(v2e["pick_cost"])
    a2e = {}
    for d in SIX:
        a2e[d] = U2E.load_u2e(d)[0]
        if not np.array_equal(a2e[d]["ptr"], am[d]["ptr"]):
            raise SystemExit(f"{d}: U2m's questions and golds are not U2e's")
    # the check: U2m's table on the own construction with kind_t is U2e's table on U2e's arrays
    c2e = U2E.u2e_candidates()
    t2e = U2E.u2e_table(a2e, c2e)
    own = {}
    for d in SIX:
        k = rm[d]["own_cell"]
        own[d] = table_arrays(am[d], [k], range(R))
        for i, c in enumerate(c2e):
            if own[d][0][i] != t2e[c[0]][d]["ALL"] or own[d][1][i] != t2e[c[0]][d]["rows"]:
                raise SystemExit(f"{d}: U2m's table is not U2e's at {c[0]}")
    log(f"U2m verdict: U2e's table reproduced on the own construction ({time.time() - t0:.0f}s)")
    univ = index_grid(range(len(GRID)), range(R))
    tabs = {d: table_arrays(am[d], range(len(GRID)), range(R)) for d in SIX}
    log(f"U2m verdict: {len(univ)} universal candidates on six datasets ({time.time() - t0:.0f}s)")
    w = F.weights()
    A = np.stack([tabs[d][0] for d in TRAIN5], axis=1)
    RW = np.stack([tabs[d][1] for d in TRAIN5], axis=1)
    costs = np.asarray([F.cost(dict(zip(TRAIN5, r.tolist())), w) for r in RW], np.float64)
    cost_r = np.asarray([round(float(c), 1) for c in costs])
    ceil = A.max(axis=0)
    rec_of = np.asarray([x[1] for x in univ])
    bud = U2E.budgets()

    def name_of(i):
        k, r, o, b, t = univ[i]
        return candidate_name(GRID[k][0], r, o, bud[b][0], t)

    def point(t, tgt, idx, kk, cr_, tab_of):
        if kk is None:
            return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": None}
        i = int(idx[kk])
        return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": tab_of["name"](i),
                "cost": float(cr_[i]), "within_cap": bool(cr_[i] <= cap + 1e-9),
                "s1sel": {d: {"ALL": float(tab_of["A"][d][i]), "rows": float(tab_of["R"][d][i])} for d in TRAIN5},
                "webqsp_s1eval": {"ALL": float(tab_of["A"]["webqsp"][i]), "rows": float(tab_of["R"]["webqsp"][i])}}

    tab_u = {"name": name_of, "A": {d: tabs[d][0] for d in SIX}, "R": {d: tabs[d][1] for d in SIX}}
    fr = {}
    for key, rs in (("C", (0,)), ("E", KIND_FREE)):
        idx = np.flatnonzero(np.isin(rec_of, rs))
        pts = U2E.frontier(A[idx], costs[idx], ceil)
        lv, kk = pick_of(pts, cost_r[idx], cap)
        fr[key] = {"points": [point(t, tg, idx, k_, cost_r, tab_u) for t, tg, k_ in pts], "level": lv,
                   "index": None if kk is None else int(idx[kk])}
    vc, ve, carried = decide(fr["C"]["level"], fr["E"]["level"], inc_level)

    # beside: the own construction's frontier per recipe, at the universal targets
    own_names = index_grid([0], range(R))
    A_own = np.stack([own[d][0] for d in TRAIN5], axis=1)
    RW_own = np.stack([own[d][1] for d in TRAIN5], axis=1)
    c_own = np.asarray([F.cost(dict(zip(TRAIN5, r.tolist())), w) for r in RW_own], np.float64)
    cr_own = np.asarray([round(float(c), 1) for c in c_own])
    tab_o = {"name": lambda i: candidate_name("own", own_names[i][1], own_names[i][2], bud[own_names[i][3]][0],
                                              own_names[i][4]),
             "A": {d: own[d][0] for d in SIX}, "R": {d: own[d][1] for d in SIX}}
    own_fr = {}
    for r, rn in enumerate(RECIPES):
        idx = np.flatnonzero(np.asarray([x[1] for x in own_names]) == r)
        pts = U2E.frontier(A_own[idx], c_own[idx], ceil)
        own_fr[rn] = [point(t, tg, idx, k_, cr_own, tab_o) for t, tg, k_ in pts]
    i2e = [c[0] for c in c2e].index(U2E_PICK)
    if cr_own[i2e] != inc_cost:
        raise SystemExit("U2e's pick does not cost what U2e filed")

    # beside: the cheapest point of each construction, recipe and order family at every level
    def cheapest(groups):
        out = {}
        for g, idx in groups.items():
            out[g] = [None if k_ is None else {"pick": name_of(int(idx[k_])), "cost": float(cost_r[idx[k_]])}
                      for _t, _tg, k_ in U2E.frontier(A[idx], costs[idx], ceil)]
        return out
    kk_ = np.asarray([x[0] for x in univ])
    oo_ = np.asarray([x[2] for x in univ])
    fam = np.asarray([FAMILIES_OF_ORDERS.index(U2E.COLUMNS[o][0]) for o in oo_])
    by_cell = cheapest({GRID[k][0]: np.flatnonzero(kk_ == k) for k in range(len(GRID))})
    by_recipe = cheapest({rn: np.flatnonzero(rec_of == r) for r, rn in enumerate(RECIPES)})
    by_family = cheapest({f: np.flatnonzero(fam == i) for i, f in enumerate(FAMILIES_OF_ORDERS)})

    per_dataset, union_ceiling = {}, {}
    for d in SIX:
        al, rw = tabs[d]
        cd = float(al.max())
        rows_at = {}
        for t in LEVELS:
            tg = round(min(t, cd - PG.TOL), 4)
            i = np.flatnonzero(al >= tg - 1e-9)
            k_ = int(i[np.argmin(rw[i])])
            rows_at[str(t)] = {"target": tg, "rows": float(rw[k_]), "ALL": float(al[k_]), "candidate": name_of(k_)}
        per_dataset[d] = {"ceiling": cd, "fewest_rows": rows_at}
        reach = (am[d]["pos"] >= 0).any(axis=1)
        union_ceiling[d] = {"ALL": U2E.all_of(reach, am[d]["ptr"]), "golds_outside": int((~reach).sum())}

    # the carried pick: its arrays, its rule at tau = 1, the ledger and its price against U2e's pick
    if carried == "U2e":
        carried_name = f"own/kind_t/{U2E_PICK}"
    else:
        carried_name = name_of(fr[carried]["index"])
    ledger, price = {}, {}
    for d in SIX:
        a = am[d]
        reach = (a["pos"] >= 0).any(axis=1)
        if carried == "U2e":
            k, r, o, b, t = (rm[d]["own_cell"],) + own_names[i2e][1:]
        else:
            k, r, o, b, t = univ[fr[carried]["index"]]
        rows, held = held_of(a, k, r, o, b, t)
        _r1, held1 = held_of(a, k, r, o, b, 1.0)
        ledger[d] = U2E.ledger_of(held, held1 | held, reach | held1 | held, a["ptr"])
        price[d] = {"carried": {"ALL": U2E.all_of(held, a["ptr"]), "rows": round(float(rows.mean()), 1)},
                    "u2e_pick": {"ALL": float(own[d][0][i2e]), "rows": float(own[d][1][i2e])}}
    rec = {"declared_in": DECLARED, "stage": "U2m (c) and (e)", "script_sha256": script_sha256(),
           "zu2e_sha256": U2E.script_sha256(), "verdict_c": vc, "verdict_e": ve, "carried": carried,
           "carried_pick": carried_name, "cap": cap,
           "u2e_pick": {"name": f"own/kind_t/{U2E_PICK}", "level": inc_level, "cost": inc_cost},
           "pick_C": {"level": fr["C"]["level"],
                      "name": None if fr["C"]["index"] is None else name_of(fr["C"]["index"]),
                      "cost": None if fr["C"]["index"] is None else float(cost_r[fr["C"]["index"]])},
           "pick_E": {"level": fr["E"]["level"],
                      "name": None if fr["E"]["index"] is None else name_of(fr["E"]["index"]),
                      "cost": None if fr["E"]["index"] is None else float(cost_r[fr["E"]["index"]])},
           "weights": w, "candidates": {"universal": len(univ), "own": len(own_names)},
           "ceilings": dict(zip(TRAIN5, ceil.tolist())), "levels": list(LEVELS),
           "frontier_C": fr["C"]["points"], "frontier_E": fr["E"]["points"], "own_frontiers": own_fr,
           "by_cell": by_cell, "by_recipe": by_recipe, "by_family": by_family, "per_dataset": per_dataset,
           "union_ceiling": union_ceiling, "ledger": ledger, "price": price,
           "own_cells": {d: rm[d]["cells"][rm[d]["own_cell"]] for d in SIX},
           "stats": {d: {"file_sha256": rm[d]["file_sha256"], "seconds": rm[d]["seconds"]} for d in SIX},
           "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "verdict_ce.json", rec)
    (OUT / "verdict_ce.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"U2m: {vc}, {ve}; carried {carried}: {carried_name} ({time.time() - t0:.0f}s)")
    return 0


def report_md(rec):
    def pk(p):
        return "none within the cap" if p["name"] is None else f"`{p['name']}` at level {p['level']}, cost {p['cost']}"
    L = [f"# U2m (c) and (e): {rec['verdict_c']}, {rec['verdict_e']}", "",
         f"Declared in {rec['declared_in']}. Carried forward: **{rec['carried']}**, `{rec['carried_pick']}`.", "",
         f"- Pick C (kind_t, universal): {pk(rec['pick_C'])}.",
         f"- Pick E (kind-free, universal): {pk(rec['pick_E'])}.",
         f"- U2e's pick (own construction): `{rec['u2e_pick']['name']}` at level {rec['u2e_pick']['level']}, cost "
         f"{rec['u2e_pick']['cost']}. Cap {rec['cap']}.", "",
         "## The carried pick against U2e's pick", "",
         "| dataset | carried ALL | rows | U2e's pick ALL | rows | union ceiling |", "|---|---:|---:|---:|---:|---:|"]
    for d, p in rec["price"].items():
        L.append(f"| {d} | {p['carried']['ALL']} | {p['carried']['rows']} | {p['u2e_pick']['ALL']} | "
                 f"{p['u2e_pick']['rows']} | {rec['union_ceiling'][d]['ALL']} |")
    for key, title in (("frontier_C", "Frontier C (kind_t)"), ("frontier_E", "Frontier E (kind-free)")):
        L += ["", f"## {title}: every training dataset at min(level, its ceiling - 0.005)", "",
              "| level | cost | within cap | pick | " + " | ".join(f"{d} ALL (rows)" for d in TRAIN5)
              + " | webqsp ALL (rows) |", "|---|---:|---|---|" + "---|" * (len(TRAIN5) + 1)]
        for e in rec[key]:
            if e["pick"] is None:
                L.append(f"| {e['level']} | - | - | none meets | " + " | ".join("-" for _ in TRAIN5) + " | - |")
                continue
            L.append(f"| {e['level']} | {e['cost']} | {e['within_cap']} | `{e['pick']}` | "
                     + " | ".join(f"{e['s1sel'][d]['ALL']} ({e['s1sel'][d]['rows']})" for d in TRAIN5)
                     + f" | {e['webqsp_s1eval']['ALL']} ({e['webqsp_s1eval']['rows']}) |")
    for key, title in (("by_cell", "construction"), ("by_recipe", "recipe"), ("by_family", "order family")):
        L += ["", f"## The cheapest point of each {title} (cost; - when none meets)", "",
              "| level | " + " | ".join(rec[key]) + " |", "|---|" + "---:|" * len(rec[key])]
        for i, t in enumerate(rec["levels"]):
            L.append(f"| {t} | " + " | ".join("-" if v[i] is None else str(v[i]["cost"]) for v in rec[key].values())
                     + " |")
    L += ["", "## The own construction's frontier per recipe (cost; beside)", "",
          "| level | " + " | ".join(rec["own_frontiers"]) + " |", "|---|" + "---:|" * len(rec["own_frontiers"])]
    for i, t in enumerate(rec["levels"]):
        L.append(f"| {t} | " + " | ".join("-" if v[i]["pick"] is None else str(v[i]["cost"])
                                          for v in rec["own_frontiers"].values()) + " |")
    L += ["", "## Per dataset: the fewest mean rows reaching each level (an upper bound only)", "",
          "| dataset | ceiling | " + " | ".join(str(t) for t in rec["levels"]) + " |",
          "|---|---:|" + "---:|" * len(rec["levels"])]
    for d, e in rec["per_dataset"].items():
        L.append(f"| {d} | {e['ceiling']} | " + " | ".join(str(e["fewest_rows"][str(t)]["rows"]) for t in rec["levels"])
                 + " |")
    L += ["", "## Ledger: the carried pick's misses", "",
          "| dataset | golds | missed | TAU_CUT | BUDGET | UNREACHED | questions missing a gold |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for d, e in rec["ledger"].items():
        L.append(f"| {d} | {e['golds']} | {e['missed']} | {e['by_cause']['TAU_CUT']} | {e['by_cause']['BUDGET']} | "
                 f"{e['by_cause']['UNREACHED']} | {e['questions_missing_a_gold']} |")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _selftest():
    import p1d_free_pool as P1D
    rng = np.random.default_rng(0)
    # 1. the grid, its names and the own constructions
    assert len(GRID) == 12 and len({g[0] for g in GRID}) == 12 and GRID[0][0] == "rrf50+none"
    assert GRID[-1][0] == "dense200+FULL:h3_c25_v2000" and [g[0] for g in GRID[4:8]][2] == "rrf200+FULL:h2_c25"
    own13 = {"metaqa": ({"base_pool": "equal_rrf_budget_50", "regime": "STRUCT", "setting": H3}, "rrf50+STRUCT:h3_c25_v2000", 12),
             "squad": ({"base_pool": "equal_rrf_budget_50", "regime": "RETRIEVAL", "setting": None}, "rrf50+none", 0),
             "musique": ({"base_pool": "equal_rrf_budget_200", "regime": "FULL", "setting": H2}, "rrf200+FULL:h2_c25", 6),
             "hotpotqa": ({"base_pool": "equal_rrf_budget_50", "regime": "STRUCT", "setting": H1}, "rrf50+STRUCT:h1_c25", 12),
             "webqsp": ({"base_pool": "dense_top200", "regime": "FULL", "setting": H3}, "dense200+FULL:h3_c25_v2000", 11)}
    for d, (c, nm, k) in own13.items():
        cells, own_k = cells_of(c)
        assert own_name(c) == nm and own_k == k and cells[own_k] == nm and len(cells) == (13 if k == 12 else 12), d
    # 2. the column layout is one to one
    cols = {column(k, r, o) for k in range(13) for r in range(R) for o in range(NO)}
    assert cols == set(range(13 * R * NO))
    # 3. the both recipe against a direct RRF, and today's fill
    for _ in range(30):
        a_ = rng.permutation(300)[:int(rng.integers(0, 80))]
        b_ = rng.permutation(300)[:int(rng.integers(0, 80))]
        got = P1D.rrf([a_, b_], 60)
        assert np.array_equal(got, U2E._direct_rrf([x for x in (a_, b_) if x.size], 60)[:P1D.NMAX])
        B = np.unique(rng.choice(300, int(rng.integers(0, 20)), replace=False))
        full = F.outside(got, B)
        n_i = B.size + int(rng.integers(0, 120))
        assert fill_of(full, n_i, B.size) == min(n_i - B.size, full.size)
    assert fill_of(np.arange(5), 3, 7) == 0
    # 4. the node rows: served reads are DenseNodes' bytes
    class _Nodes:
        def __init__(self, m):
            self.m = m

        def read(self, rows):
            return self.m[np.asarray(rows, np.int64)].astype(np.float32)
    nodes = _Nodes(rng.standard_normal((200, 16)).astype(np.float16))
    rows = Rows(nodes, verify=True)
    rows.add(rng.choice(200, 30, replace=False))
    rows.add(rng.choice(200, 60, replace=False))
    for _ in range(20):
        pick = rng.choice(rows.ids, int(rng.integers(1, rows.ids.size)), replace=True)
        assert np.array_equal(rows.read(pick), nodes.read(pick))
    assert rows.read(EMPTY).shape == (0, 16)
    try:
        rows.read(np.setdiff1d(np.arange(200), rows.ids)[:1])
    except SystemExit:
        pass
    else:
        raise AssertionError("an unread row was served")
    # 5. the vectorized table against U2e's evaluate and summ, on toy arrays of two cells
    nq, nk = 30, 2
    ptr = np.concatenate([[0], np.cumsum(rng.integers(1, 4, nq))]).astype(np.int64)
    ng = int(ptr[-1])
    qi = np.repeat(np.arange(nq), np.diff(ptr))
    a = {"bs": rng.integers(0, 6, (nq, nk)).astype(np.int64), "fill": rng.integers(0, 300, (nq, nk * R)),
         "lens": rng.integers(0, 3000, (nq, nk * R * NO)).astype(np.int32), "ptr": ptr}
    pos = rng.integers(-1, 3000, (ng, nk * R * NO))
    pos = np.where(pos > a["lens"][qi], -1, pos)
    for k in range(nk):
        cs = [column(k, r, o) for r in range(R) for o in range(NO)]
        sub = pos[:, cs]
        sub[(sub == 0) & (a["bs"][qi, k] == 0)[:, None]] = -1
        pos[:, cs] = sub
    a["pos"] = pos.astype(np.int32)
    ALL, ROWS = table_arrays(a, range(nk), range(R))
    idx = index_grid(range(nk), range(R))
    assert ALL.size == len(idx) == nk * R * 13860
    bud = U2E.budgets()
    for i in rng.choice(len(idx), 400, replace=False):
        k, r, o, b, t = idx[int(i)]
        rows_, held = held_of(a, k, r, o, b, t)
        s = U2E.summ(rows_, held, ptr, "x")
        assert (ALL[i], ROWS[i]) == (s["ALL"], s["rows"]), (k, r, o, b, t)
    e = {"bs": a["bs"][:, 1], "fill": a["fill"][:, R + 2], "lens": a["lens"][:, [column(1, 2, o) for o in range(NO)]],
         "pos": a["pos"][:, [column(1, 2, o) for o in range(NO)]], "ptr": ptr}
    t2 = U2E.u2e_table({"x": e}, U2E.u2e_candidates())
    A1, R1 = table_arrays(a, [1], [2])
    assert all((A1[i], R1[i]) == (t2[c[0]]["x"]["ALL"], t2[c[0]]["x"]["rows"]) for i, c in enumerate(U2E.u2e_candidates()))
    assert candidate_name("rrf50+none", 2, 6, bud[0][0], 0.85) == "rrf50+none/both/kt_cos:fix_25@0.85"
    # 6. the verdicts and what is carried
    assert decide(0.96, 0.96, 0.955) == ("UNIFIED_RAISED", "KIND_FREE_NO_PRICE", "E")
    assert decide(0.955, 0.95, 0.955) == ("UNIFIED_HOLDS", "KIND_FREE_AT_A_PRICE", "E")
    assert decide(0.955, 0.945, 0.955) == ("UNIFIED_HOLDS", "KIND_FREE_TOO_DEAR", "C")
    assert decide(0.96, 0.95, 0.955) == ("UNIFIED_RAISED", "KIND_FREE_TOO_DEAR", "C")
    assert decide(0.95, None, 0.955) == ("UNIFIED_AT_A_PRICE", "KIND_FREE_TOO_DEAR", "C")
    assert decide(None, 0.95, 0.955) == ("NOT_UNIFIED", "KIND_FREE_NO_PRICE", "E")
    assert decide(None, None, 0.955) == ("NOT_UNIFIED", "KIND_FREE_TOO_DEAR", "U2e")
    A = np.asarray([[0.95, 0.96], [0.97, 0.97], [0.96, 0.99]])
    pts = U2E.frontier(A, np.asarray([100.0, 150.0, 140.0]), A.max(axis=0))
    assert pick_of(pts, np.asarray([100.0, 150.0, 140.0]), 145.0) == (0.96, 2)
    assert pick_of(pts, np.asarray([100.0, 150.0, 140.0]), 50.0) == (None, None)
    # 7. the shards: any split merges back to the whole; parts out of order are refused
    whole = {"bs": a["bs"], "inc": rng.integers(0, 90, (nq, nk)), "n_i": rng.integers(0, 90, (nq, nk)),
             "fill": a["fill"], "lens": a["lens"], "n_u": rng.integers(0, 90, (nq, nk * R * len(MS))),
             "n_n1": rng.integers(0, 50, (nq, 3)), "n_r0": rng.integers(0, 50, (nq, 3)),
             "n_n2": rng.integers(0, 50, (nq, 9)), "taken": rng.integers(0, 100, (nq, 3)),
             "hubs": rng.integers(0, 3, (nq, 3)), "pos": a["pos"], "in_I": rng.random((ng, nk)) < 0.6,
             "one_link": rng.random((ng, 3)) < 0.3, "ptr": ptr}
    for k in (1, 2, 3, 7, nq):
        parts = []
        for i in range(k):
            q0, q1 = U2C.shard_range(nq, i, k)
            g0, g1 = int(ptr[q0]), int(ptr[q1])
            parts.append({**{x: whole[x][q0:q1] for x in PER_QUESTION}, **{x: whole[x][g0:g1] for x in PER_GOLD},
                          "ptr": ptr[q0:q1 + 1]})
        m = merge_arrays(parts)
        assert list(m) == list(ARRAYS) and all(np.array_equal(m[x], whole[x]) for x in ARRAYS), k
    try:
        merge_arrays([parts[1], parts[0]] + parts[2:])
    except SystemExit:
        pass
    else:
        raise AssertionError("parts out of order merged")
    cells2 = ["rrf50+none", "rrf50+FULL:h1_c25"]
    assert summary_of(m, cells2) == summary_of(whole, cells2)
    print("zu2m selftest: OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("stats", "merge", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", help="I/K: the I-th (from 0) of K contiguous question ranges")
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return 0
    if a.stage == "stats":
        if not a.dataset:
            raise SystemExit("stats needs DATASET")
        shard = None
        if a.shard:
            i, k = (int(x) for x in a.shard.split("/"))
            shard = (i, k)
        return stats_stage(a.dataset, a.threads, a.limit, shard)
    if a.stage == "merge":
        if not a.dataset or a.parts < 1:
            raise SystemExit("merge needs DATASET and --parts K")
        return merge_stage(a.dataset, a.parts)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
