"""U2m section 19: E1, U2l's entity match (IS alone), as the pools' entity family (docs/U2M_POOLS_THREE_FAMILIES.md,
section 19). Pools only: nothing trains, and no family, construction, budget or cut reads a question or a gold.

    python outputs/mp_unified/zu2m_e1.py stats DATASET [--threads 2] [--limit Q [--control]] [--shard I/K]
    python outputs/mp_unified/zu2m_e1.py merge DATASET --parts K
    python outputs/mp_unified/zu2m_e1.py verdict
    python outputs/mp_unified/zu2m_e1.py --selftest

stats    One dataset's carve (s1sel; webqsp's s1eval). Each question walks twice, as zu2m.py's stats walk: on the filed
         families (structural, the package's NER, kNN), checked against U2m's filed own kind_t columns, and on E1's
         (the ner slot empty on a passage graph, U2l's IS on a KB). The own cell's kind_t columns (fill, lens, pos)
         of E1's walks, N1, N2 and I are recorded. --limit Q: the first Q questions, every check, nothing written and
         no number logged; with --control the package's NER stands in E1's slot, so every question's E1 side must
         equal its filed side. --shard I/K: the I-th of K contiguous question ranges, into outputs/u2m/e1/stats/parts/.
merge    Join a dataset's K parts into the file one run over every question writes.
verdict  U2e's 13,860 candidates on the own construction with kind_t, on U2m's arrays with E1's columns spliced in on
         all six; section 19's rule and verdict. Writes outputs/u2m/e1/verdict_e1.json and .md.
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

import zu2m as M  # noqa: E402  (U2m's cells, columns, fill, table, held and pick, unchanged)
import zu2o as O  # noqa: E402  (U2o's kind_t columns, splice, misses and the carried pick's check, unchanged)

U2E = M.U2E
U2C, PG, F, P1H, D5, D5B = U2E.U2C, U2E.PG, U2E.F, U2E.P1H, U2E.D5, U2E.D5B
log, utc, write_json, sha_file = U2E.log, U2E.utc, U2E.write_json, U2E.sha_file
DECLARED = "docs/U2M_POOLS_THREE_FAMILIES.md, section 19"
OUT = M.OUT / "e1"
U2L = ROOT / "outputs" / "u2l" / "el"
U2B = ROOT / "outputs" / "u2b"
SIX, TRAIN5 = U2E.SIX, U2E.TRAIN5
NAMES, NO, R, EMPTY = M.NAMES, M.NO, M.R, M.EMPTY
FAMILIES = ("structural", "ner", "knn")
UNIONS = (("st", ("structural",)), ("all", FAMILIES))
ST_SOURCES = (0, 2, 4)                              # P1d's A1st, Kst and R0: no walk of theirs reads ner
ST_STEERED = ("KTst", "A1Tst")
CARRIED = O.CARRIED
VERDICTS = ("E1_RAISED", "E1_CHEAPER", "NO_GAIN")
PER_QUESTION = ("bs", "inc", "fill", "lens", "kt_same")
PER_GOLD = ("pos", "in_I")
ARRAYS = PER_QUESTION + PER_GOLD + ("ptr",)
HEAD_KEYS = ("declared_in", "stage", "dataset", "carve", "graph_detail", "e1", "script_sha256", "zu2m_sha256",
             "zu2o_sha256", "zu2e_sha256", "u2m_stats_sha256", "verdict_ce_sha256", "u2l_choice_sha256",
             "p1e_script_sha256", "freeze_RECORD_SHA256", "threads", "own_cell", "orders", "regime_families", "I")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def el_choice():
    """U2l's filed choice: CHOSEN E1, read on E1 + R + kNN."""
    p = U2L / "choice.json"
    c = json.loads(p.read_text(encoding="utf-8"))
    if c.get("outcome") != "CHOSEN E1" or c.get("form") != "E1" or c.get("graph") != "E1+R_kNN":
        raise SystemExit(f"{p}: U2l's choice is not E1")
    return c, sha_file(p)


def e1_store(name, n, desc):
    """E1's store for the ner slot (section 19).
    A passage graph: empty. E1 with the sibling links is the structural graph already: U2l's gate 1 found it equal,
    array for array, to U2b's filed W2_S, and that file is the one U2m swaps in (its sha256 is the graph_detail's).
    A KB: U2l's IS, undirected and collapsed by zu2b.undirected_csr (rows in node-id order; self-loops kept, as U2l
    graded it), weight 1 on every entry (no pool rule reads a family's weight)."""
    import zu2b
    from mp_retrieval.m3b_pools import FamilyStore
    b = json.loads((U2L / name / "build.json").read_text(encoding="utf-8"))
    p = U2L / name / "el.npz"
    if sha_file(p) != b["el_npz_sha256"] or int(b["nodes"]) != int(n):
        raise SystemExit(f"{name}: U2l's el.npz is not its build record's, or not this dataset's nodes")
    if name in zu2b.PASSAGE:
        u2b = json.loads((U2B / name / "graphs" / "build_new.json").read_text(encoding="utf-8"))["graphs"]["W2_S"]
        if b["gate1"] != "PASS" or b.get("u2i_cue") is not None:
            raise SystemExit(f"{name}: U2l's gate 1 is not PASS on W2_S")
        if desc.get("graph") != "W2_S" or desc.get("npz_sha256") != u2b["npz_sha256"]:
            raise SystemExit(f"{name}: the structural graph U2m swaps in is not U2b's filed W2_S, U2l's gate-1 graph")
        ip, col = np.zeros(n + 1, np.int64), np.empty(0, np.int32)
        info = {"ner": "empty", "e1_with_siblings_is_structural": True, "is_links": int(b["is_links"]),
                "el_npz_sha256": b["el_npz_sha256"]}
        stored = 0
    else:
        with np.load(p) as z:
            s, t = z["is_src"].astype(np.int64), z["is_dst"].astype(np.int64)
        ip, col = zu2b.undirected_csr(n, s, t)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(ip))
        info = {"ner": "IS", "is_links": int(s.size), "entries": int(col.size),
                "self_loops": int((rows == col).sum()), "el_npz_sha256": b["el_npz_sha256"]}
        stored = int(s.size)
        del s, t, rows
    if ip.size != n + 1 or int(ip[-1]) != col.size:
        raise SystemExit(f"{name}: E1's store is not one of {n} nodes")
    return FamilyStore("ner", int(n), ip, col, weight=np.ones(col.size, np.float16), edges_stored=stored), info


def e1_unions(stores_all, unions_w, e1, S4B, n):
    """E1's stores and unions: the filed ones with E1 in the ner slot; the st union is the filed one itself."""
    stores_e = dict(stores_all)
    stores_e["ner"] = e1
    return stores_e, {"st": unions_w["st"], "all": S4B.RegimeUnion(stores_e, list(FAMILIES), n)}


def adjacency(stores):
    return [(np.ascontiguousarray(stores[f].indptr, np.int64), np.ascontiguousarray(stores[f].col)) for f in FAMILIES]


def same_sources(a, b, idx):
    return all(np.array_equal(a[i][0], b[i][0]) and np.array_equal(a[i][1], b[i][1]) for i in idx)


def same_walks(a, b, keys):
    """P1e's steered walks are node orders (twalk's)."""
    return all(np.array_equal(a[k], b[k]) for k in keys)


# ── stats (the laptop, or elsewhere in shards) ───────────────────────────────


def stats_stage(name, threads=2, limit=0, shard=None, control=False):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if control and not limit:
        raise SystemExit("--control is a development run's check: give --limit")
    if D.SOURCES != ("A1st", "A1all", "Kst", "Kall", "R0") or F.CONSTRUCTION != "KIND_T":
        raise SystemExit("P1d's sources are not A1st, A1all, Kst, Kall, R0, or P1f's construction is not KIND_T")
    if tuple(S4.FAMILIES) != FAMILIES or tuple((u, tuple(f)) for u, f in G.UNIONS) != UNIONS:
        raise SystemExit("the families or the walks' unions are not the ones section 19 reads")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    vrec = O.ce_verdict()
    _choice, choice_sha = el_choice()
    x2m = M.load_u2m(name)
    if x2m is None:
        raise SystemExit(f"{name}: U2m's filed arrays are not under outputs/u2m/stats")
    a2m, r2m = x2m
    if vrec["stats"][name]["file_sha256"] != r2m["file_sha256"]:
        raise SystemExit(f"{name}: U2m's filed arrays are not the ones (c) and (e)'s verdict read")
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    cells, own_k = M.cells_of(construction)
    if cells != r2m["cells"] or own_k != r2m["own_cell"]:
        raise SystemExit(f"{name}: the cells are not U2m's")
    reg_f = list(op.m3b_compile.regime_families(op.cfg_h, construction["regime"])) \
        if construction["regime"] != "RETRIEVAL" else []
    n = int(ds.n_nodes)
    g = U2C.graph_csr(name, "chosen", n)
    stores, union, desc = stores0, union0, {"graph": "cached (the KB)"}
    if g is not None:                     # zu2m.py's swap, unchanged
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
    if desc != r2m["graph_detail"]:
        raise SystemExit(f"{name}: the graph is not the one U2m's filed run read")
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
    if any(not np.array_equal(np.asarray(x, np.int64), np.asarray(y, np.int64)) for x, y in zip(prep.seeds, prep0.seeds)):
        raise SystemExit(f"{name}/{carve}: the seeds on the graph are not G0's")
    base_own = op.m3b_compile.base_rows(construction, prep0.dense_ids, prep0.splade_ids, op.m3a, op.m3b_contract,
                                        constant)
    inc_own = [np.asarray(p, np.int64) for p in prep.pools]
    del prep0
    setting = construction.get("setting")
    linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [EMPTY] * len(prep.seeds)
    cache = {}
    stores_all, unions_w = G.open_unions(cache, op, name, union)
    if control:
        e1, e1_info = stores_all["ner"], {"ner": "the package's NER (--control)"}
    else:
        e1, e1_info = e1_store(name, n, desc)
    stores_e, unions_e = e1_unions(stores_all, unions_w, e1, S4B, n)
    adj_w, adj_e = adjacency(stores_all), adjacency(stores_e)
    fam_w = [stores_all[f] for f in reg_f]
    fam_e = [stores_e[f] for f in reg_f]
    reads_ner = "ner" in reg_f
    log(f"{name}/{carve}: prepared; own cell {cells[own_k]}, I reads {reg_f or 'no family'}; E1's ner slot: "
        f"{e1_info['ner']} ({time.time() - t0:.0f}s)")
    n_st = int(unions_w["st"].deg.size)
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    nq = min(limit, len(prep.seeds)) if limit else len(prep.seeds)
    golds = [np.unique(np.asarray(x, np.int64)) for x in pop.golds[:nq]]
    ptr = np.concatenate([[0], np.cumsum([x.size for x in golds])]).astype(np.int64)
    if not np.array_equal(a2m["ptr"][:nq + 1], ptr):
        raise SystemExit(f"{name}/{carve}: U2m's golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    own_cols = [M.column(own_k, 0, o) for o in range(NO)]
    pq = {k: [] for k in PER_QUESTION}
    pg = {k: [] for k in PER_GOLD}
    t = time.time()
    done = 0
    for lo in range(q0, q1, S4.BATCH):
        js = list(range(lo, min(q1, lo + S4.BATCH)))
        sides = {}
        for side, unions in (("w", unions_w), ("e", unions_e)):
            srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
            sides[side] = (srcs, K.steered_batch(prep, js, linked, unions, srcs, constant, threads))
        for i, j in enumerate(js):
            seeds = np.asarray(prep.seeds[j], np.int64)
            q = D5B.unit(np.asarray(prep.qemb[j], np.float64))
            w_all = K.steer(prep, j, constant)
            rows = M.Rows(nodes, verify=bool(limit))
            gj = golds[j]
            g0, g1 = int(ptr[j]), int(ptr[j + 1])
            base = np.asarray(base_own[j], np.int64)
            B = np.unique(np.union1d(base, seeds))
            in_b = np.isin(gj, B)
            got = {}
            for side, adj, fam in (("w", adj_w, fam_w), ("e", adj_e, fam_e)):
                srcs, steered = sides[side]
                src, _nn, (L, Ls) = srcs[i]
                knd = K.kind(Ls, prep.seeds[j], nonly)
                out, p1d = K.p1e_orders(src, steered[i], D.routes(L, Ls), knd, n_st, constant)
                kt = np.asarray(out[F.CONSTRUCTION], np.int64)
                want_kt = p1d["ROUTEst"] if knd == "entity" else out["RRF5T"]
                if not np.array_equal(kt, np.asarray(want_kt, np.int64)):
                    raise SystemExit(f"{name}/{carve} question {j}: KIND_T is not ROUTEst or RRF5T by the kind")
                n1, _cnt, _ws, _up, _inv = D5B.n1_table(adj, B, D5.weights_of(w_all, B))
                x = {"n1": n1, "r0": F.outside(src[4][0], B)}
                rows.add(n1)
                x["cos1"] = D5B.unit(rows.read(n1)) @ q if n1.size else np.empty(0, np.float32)
                x["cr"] = D5B.ranks_desc(x["cos1"], n1) if n1.size else EMPTY
                x["n2"], x["first"], _taken, _hubs = U2E.second_link(adj, n1, x["cr"], B)
                if reads_ner:                 # the I moves only where the construction reads ner
                    I = np.asarray(op.m3b_compile.build_pool(base, seeds, S4.m3b_pools.expand_hops(seeds, fam,
                                                                                                setting))[0], np.int64)
                else:
                    I = inc_own[j]
                if not np.isin(seeds, B).all() or not np.isin(B, I).all():
                    raise SystemExit(f"{name}/{carve} question {j}: B or the {side} I is malformed")
                rows.add(np.concatenate([kt, src[4][0], I, x["n2"]]))
                cols = O.columns_of(F.outside(kt, B), x, F.outside(I, B), gj, in_b, rows, q, constant, D.rrf, B,
                                    I.size)
                got[side] = {"src": src, "steered": steered[i], "kind": knd, "kt": kt, "I": I, "cols": cols}
            w, e = got["w"], got["e"]
            fw, lw, pw = w["cols"]
            if reads_ner and not np.array_equal(w["I"], inc_own[j]):
                raise SystemExit(f"{name}/{carve} question {j}: the I rebuilt on the filed families is not prepare's")
            if not (B.size == int(a2m["bs"][j, own_k]) and w["I"].size == int(a2m["inc"][j, own_k])
                    and fw == int(a2m["fill"][j, own_k * R]) and np.array_equal(lw, a2m["lens"][j, own_cols])
                    and np.array_equal(pw, a2m["pos"][g0:g1, own_cols])
                    and np.array_equal(np.isin(gj, w["I"]), a2m["in_I"][g0:g1, own_k])):
                raise SystemExit(f"{name}/{carve} question {j}: the own kind_t columns on the filed families are not "
                                 "U2m's")
            if e["kind"] != w["kind"] or not same_sources(e["src"], w["src"], ST_SOURCES) or \
                    not same_walks(e["steered"], w["steered"], ST_STEERED):
                raise SystemExit(f"{name}/{carve} question {j}: a walk that reads no ner, or the kind, changed")
            fe, le, pe = e["cols"]
            if control and not (np.array_equal(e["kt"], w["kt"]) and np.array_equal(e["I"], w["I"]) and fe == fw
                                and np.array_equal(le, lw) and np.array_equal(pe, pw)):
                raise SystemExit(f"{name}/{carve} question {j}: with the package's NER in E1's slot, the E1 side is "
                                 "not the filed side")
            pq["bs"].append(B.size)
            pq["inc"].append([w["I"].size, e["I"].size])
            pq["fill"].append(fe)
            pq["lens"].append(le)
            pq["kt_same"].append(bool(np.array_equal(e["kt"], w["kt"])))
            pg["pos"].append(pe)
            pg["in_I"].append(np.stack([np.isin(gj, w["I"]), np.isin(gj, e["I"])], axis=1))
            done += 1
            if done % 100 == 0:
                log(f"{name}/{carve}: {done}/{q1 - q0} ({time.time() - t:.0f}s)")
    arr = arrays_of(pq, pg, ptr[q0:q1 + 1])
    head = {"declared_in": DECLARED, "stage": "U2m E1", "dataset": name, "carve": carve, "graph_detail": desc,
            "e1": e1_info, "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(),
            "zu2o_sha256": O.script_sha256(), "zu2e_sha256": U2E.script_sha256(), "u2m_stats_sha256": r2m["file_sha256"],
            "verdict_ce_sha256": sha_file(M.OUT / "verdict_ce.json"), "u2l_choice_sha256": choice_sha,
            "p1e_script_sha256": K.script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
            "threads": threads, "own_cell": own_k, "orders": list(NAMES), "regime_families": reg_f,
            "I": "rebuilt on both families (the construction reads ner)" if reads_ner else
                 "prepare's pools on both families (the construction reads no ner)"}
    checks = {"graph_equal_u2m": True, "seeds_equal_g0": True, "kind_t_by_kind": True, "I_on_filed_equal_prepare": True,
              "own_kind_t_on_filed_equal_u2m": True, "orders_exact": True, "st_walks_r0_kind_unchanged": True,
              "e1_of_u2l": True, "rows_served_verified": bool(limit)}
    if limit:
        log(f"{name}/{carve}: development run on questions {q0}-{q1 - 1} of {nq} ({time.time() - t:.0f}s): every "
            "check passed, every served row verified" + ("; with the package's NER in E1's slot every question's E1 "
                                                         "side equals its filed side" if control else "")
            + "; nothing written")
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


def arrays_of(pq, pg, ptr):
    a = {"bs": np.asarray(pq["bs"], np.int64), "inc": np.asarray(pq["inc"], np.int64).reshape(-1, 2),
         "fill": np.asarray(pq["fill"], np.int64), "lens": np.asarray(pq["lens"], np.int32).reshape(-1, NO),
         "kt_same": np.asarray(pq["kt_same"], bool),
         "pos": np.concatenate(pg["pos"]) if pg["pos"] else np.zeros((0, NO), np.int32),
         "in_I": np.concatenate(pg["in_I"]) if pg["in_I"] else np.zeros((0, 2), bool),
         "ptr": np.asarray(ptr, np.int64)}
    return {k: a[k] for k in ARRAYS}


def summary_of(a):
    """The I's ALL and mean size on the filed families and on E1's; E1's mean fill, the share of questions whose KT
    order E1 leaves unchanged, and each order's whole reach and length under E1."""
    ptr = a["ptr"] - a["ptr"][0]
    ng = np.diff(ptr)
    qi = np.repeat(np.arange(ng.size), ng)
    has = ng > 0
    out = {"I": {}, "fill": round(float(a["fill"].mean()), 1), "kt_same": round(float(a["kt_same"].mean()), 4),
           "orders": {}}
    for i, g in enumerate(("filed", "E1")):
        out["I"][g] = {"ALL": U2E.all_of(a["in_I"][:, i], ptr), "size": round(float(a["inc"][:, i].mean()), 1)}
    for o, nm in enumerate(NAMES):
        hits = np.bincount(qi, weights=(a["pos"][:, o] >= 0).astype(np.float64), minlength=ng.size)
        out["orders"][nm] = {"ALL": round(float((has & (hits == ng)).mean()), 4),
                             "length": round(float(a["lens"][:, o].mean()), 1)}
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
           "summary": summary_of(arr), "checks": checks, "seconds": seconds, **(extra or {}), "utc": utc()}
    write_json(od / f"{name}__{carve}.json", rec)
    log(f"{name}/{carve} ({nq} q, {seconds:.0f}s): written")
    return 0


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
            raise SystemExit(f"part {i}: another script, graph, thread count, check set or carve")
        if (r["part"], r["parts"]) != (i, k) or (r["q0"], r["q1"]) != U2C.shard_range(nq, i, k):
            raise SystemExit(f"part {i}: not the {i}-th of {k} question ranges")
    out = {a: np.concatenate([p[a] for p in parts]) for a in PER_QUESTION + PER_GOLD}
    ptr = [np.asarray([0], np.int64)]
    for i, p in enumerate(parts):
        q = np.asarray(p["ptr"], np.int64)
        if q.size != p["bs"].shape[0] + 1 or int(q[0]) != int(ptr[-1][-1]):
            raise SystemExit(f"part {i}: its ptr does not continue the previous part's")
        ptr.append(q[1:])
    out["ptr"] = np.concatenate(ptr)
    arr = {a: out[a] for a in ARRAYS}
    a2m = M.load_u2m(name)[0]
    if arr["bs"].shape[0] != nq or not np.array_equal(a2m["ptr"][:nq + 1], arr["ptr"]):
        raise SystemExit(f"{name}/{carve}: the merged parts do not cover U2m's questions and golds")
    return write_stats(name, carve, arr, head, recs[0]["checks"], round(sum(r["seconds"] for r in recs), 1),
                       {"shards": k, "part_sha256": [r["file_sha256"] for r in recs],
                        "seconds_longest_part": max(r["seconds"] for r in recs)})


# ── the verdict (the laptop) ─────────────────────────────────────────────────


def load_e(name):
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
    if rec["orders"] != list(NAMES) or summary_of(a) != rec["summary"]:
        raise SystemExit(f"{p}: not this script's orders, or the arrays do not give its summary")
    return a, rec


def decide(level, cost, car_level, car_cost):
    if level is None:
        return "NO_GAIN"
    if level > car_level + 1e-9:
        return "E1_RAISED"
    if abs(level - car_level) <= 1e-9 and cost < car_cost - 1e-9:
        return "E1_CHEAPER"
    return "NO_GAIN"


def verdict_stage():
    t0 = time.time()
    vrec = O.ce_verdict()
    _choice, choice_sha = el_choice()
    car_name, car_level, car_cost = CARRIED
    cap = float(vrec["cap"])
    w = F.weights()
    if w != vrec["weights"]:
        raise SystemExit("P1f's weights are not the ones (c) and (e)'s verdict used")
    ceil = np.asarray([vrec["ceilings"][d] for d in TRAIN5], np.float64)
    am, rm, ae, re_, missing = {}, {}, {}, {}, []
    for d in SIX:
        a, r = M.load_u2m(d)
        if vrec["stats"][d]["file_sha256"] != r["file_sha256"]:
            raise SystemExit(f"{d}: U2m's arrays are not the ones (c) and (e)'s verdict read")
        am[d], rm[d] = a, r
        x = load_e(d)
        if x is None:
            missing.append(d)
            continue
        ae[d], re_[d] = x
        if re_[d]["u2m_stats_sha256"] != r["file_sha256"] or re_[d]["own_cell"] != r["own_cell"] or \
                re_[d]["u2l_choice_sha256"] != choice_sha or not np.array_equal(ae[d]["ptr"], a["ptr"]) or \
                not np.array_equal(ae[d]["bs"], a["bs"][:, r["own_cell"]]) or \
                not np.array_equal(ae[d]["inc"][:, 0], a["inc"][:, r["own_cell"]]):
            raise SystemExit(f"{d}: E1's arrays are not on U2m's questions, golds, B and I, or not U2l's E1")
    if missing:
        raise SystemExit(f"INCOMPLETE: no E1 stats for {missing}")
    bud = U2E.budgets()
    cands = M.index_grid([0], [0])

    def name_of(i):
        _k, r, o, b, t = cands[i]
        return M.candidate_name("own", r, o, bud[b][0], t)

    names = [name_of(i) for i in range(len(cands))]
    a2e = {d: U2E.load_u2e(d)[0] for d in SIX}
    c2e = U2E.u2e_candidates()
    t2e = U2E.u2e_table(a2e, c2e)
    if [c[0] for c in c2e] != [nm.split("/", 2)[2] for nm in names]:
        raise SystemExit("U2e's candidates are not the own kind_t grid in U2m's order")
    tw, te, ax = {}, {}, {}
    for d in SIX:
        k = rm[d]["own_cell"]
        ax[d] = O.splice(am[d], k, ae[d])
        tw[d] = M.table_arrays(am[d], [k], [0])
        te[d] = M.table_arrays(ax[d], [k], [0])
        for i, c in enumerate(c2e):
            if tw[d][0][i] != t2e[c[0]][d]["ALL"] or tw[d][1][i] != t2e[c[0]][d]["rows"]:
                raise SystemExit(f"{d}: the table on the filed families is not U2e's at {c[0]}")
    log(f"U2m E1 verdict: U2e's table reproduced on the filed families ({time.time() - t0:.0f}s)")

    def frontier_of(tab):
        A = np.stack([tab[d][0] for d in TRAIN5], axis=1)
        RW = np.stack([tab[d][1] for d in TRAIN5], axis=1)
        costs = np.asarray([F.cost(dict(zip(TRAIN5, r_.tolist())), w) for r_ in RW], np.float64)
        cost_r = np.asarray([round(float(c), 1) for c in costs])
        fr = U2E.frontier(A, costs, ceil)
        return fr, cost_r, M.pick_of(fr, cost_r, cap)

    fr_w, cr_w, (lv_w, kk_w) = frontier_of(tw)
    for (t, _tg, k_), x in zip(fr_w, vrec["own_frontiers"]["kind_t"]):
        got = None if k_ is None else names[k_]
        if abs(t - x["level"]) > 1e-9 or got != x["pick"] or (k_ is not None and float(cr_w[k_]) != x["cost"]):
            raise SystemExit(f"level {t}: the frontier on the filed families is {got}, not the filed own kind_t "
                             "frontier's")
    i2e = names.index(car_name)
    if lv_w != car_level or kk_w != i2e or float(cr_w[i2e]) != car_cost:
        raise SystemExit("on the filed families the rule does not return U2e's pick at its level and cost")
    fr, cost_r, (lv, kk) = frontier_of(te)
    v = decide(lv, None if kk is None else float(cost_r[kk]), car_level, car_cost)
    pick = names[kk] if v != "NO_GAIN" else car_name

    def point(t, tgt, k_, tab, cr_):
        if k_ is None:
            return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": None}
        return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": names[k_], "cost": float(cr_[k_]),
                "within_cap": bool(cr_[k_] <= cap + 1e-9),
                "s1sel": {d: {"ALL": float(tab[d][0][k_]), "rows": float(tab[d][1][k_])} for d in TRAIN5},
                "webqsp_s1eval": {"ALL": float(tab["webqsp"][0][k_]), "rows": float(tab["webqsp"][1][k_])}}

    # beside: the I, the share of questions whose KT order E1 leaves, U2e's point under E1, the misses by cause
    the_I = {d: {**re_[d]["summary"]["I"], "kt_same": re_[d]["summary"]["kt_same"]} for d in SIX}
    at_u2e = {d: {"filed": {"ALL": float(tw[d][0][i2e]), "rows": float(tw[d][1][i2e])},
                  "E1": {"ALL": float(te[d][0][i2e]), "rows": float(te[d][1][i2e])}} for d in SIX}
    why = {}
    for d in SIX:
        k = rm[d]["own_cell"]
        cw = (k,) + cands[i2e][1:]
        why[d] = {"u2e_pick_filed": O.misses(am[d], k, cw, car_name), "u2e_pick_E1": O.misses(ax[d], k, cw, car_name)}
        if kk is not None and kk != i2e:
            why[d]["e1_pick"] = O.misses(ax[d], k, (k,) + cands[kk][1:], names[kk])
    rec = {"declared_in": DECLARED, "stage": "U2m E1", "script_sha256": script_sha256(),
           "zu2m_sha256": M.script_sha256(), "zu2o_sha256": O.script_sha256(), "zu2e_sha256": U2E.script_sha256(),
           "verdict_ce_sha256": sha_file(M.OUT / "verdict_ce.json"), "u2l_choice_sha256": choice_sha,
           "verdict": v, "pick": pick, "level": lv, "cost": None if kk is None else float(cost_r[kk]),
           "frontier_pick": None if kk is None else names[kk],
           "carried": {"pick": car_name, "level": car_level, "cost": car_cost}, "cap": cap, "weights": w,
           "ceilings": dict(zip(TRAIN5, ceil.tolist())), "candidates": len(cands),
           "e1": {d: re_[d]["e1"] for d in SIX},
           "frontier_E1": [point(t, tg, k_, te, cost_r) for t, tg, k_ in fr],
           "frontier_filed": [point(t, tg, k_, tw, cr_w) for t, tg, k_ in fr_w],
           "the_I": the_I, "u2e_point": at_u2e, "misses": why,
           "stats": {d: {"file_sha256": re_[d]["file_sha256"], "seconds": re_[d]["seconds"]} for d in SIX},
           "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "verdict_e1.json", rec)
    (OUT / "verdict_e1.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"U2m E1: {v}; pick {pick} at level {lv} ({time.time() - t0:.0f}s)")
    return 0


def report_md(rec):
    c = rec["carried"]
    L = [f"# U2m E1: {rec['verdict']}", "",
         f"Declared in {rec['declared_in']}. Pick: `{rec['pick']}`. The rule's pick under E1: "
         f"`{rec['frontier_pick']}` at level {rec['level']}, cost {rec['cost']}. Carried: `{c['pick']}` at level "
         f"{c['level']}, cost {c['cost']}. Cap {rec['cap']}.", "",
         "## The I and the KT order (beside)", "",
         "| dataset | E1's ner slot | I ALL filed | size | I ALL E1 | size | questions with KT unchanged |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for d, x in rec["the_I"].items():
        L.append(f"| {d} | {rec['e1'][d]['ner']} | {x['filed']['ALL']} | {x['filed']['size']} | {x['E1']['ALL']} | "
                 f"{x['E1']['size']} | {x['kt_same']} |")
    L += ["", "## U2e's point on the filed families and on E1's", "", "| dataset | filed ALL | rows | E1 ALL | rows |",
          "|---|---:|---:|---:|---:|"]
    for d, x in rec["u2e_point"].items():
        L.append(f"| {d} | {x['filed']['ALL']} | {x['filed']['rows']} | {x['E1']['ALL']} | {x['E1']['rows']} |")
    for title, key in (("E1's families", "frontier_E1"), ("the filed families", "frontier_filed")):
        L += ["", f"## The frontier on {title}", "",
              "| level | cost | within cap | pick | " + " | ".join(f"{d} ALL (rows)" for d in TRAIN5) + " | webqsp |",
              "|---|---:|---|---|" + "---:|" * (len(TRAIN5) + 1)]
        for e in rec[key]:
            if e["pick"] is None:
                L.append(f"| {e['level']} | | | none | " + " | ".join("" for _ in TRAIN5) + " | |")
                continue
            L.append(f"| {e['level']} | {e['cost']} | {e['within_cap']} | `{e['pick']}` | "
                     + " | ".join(f"{e['s1sel'][d]['ALL']} ({e['s1sel'][d]['rows']})" for d in TRAIN5)
                     + f" | {e['webqsp_s1eval']['ALL']} ({e['webqsp_s1eval']['rows']}) |")
    L += ["", "## Misses by cause (beside)", "", "| dataset | candidate | families | missed | TAU_CUT | BUDGET | "
          "UNREACHED |", "|---|---|---|---:|---:|---:|---:|"]
    for d, x in rec["misses"].items():
        for tag, m in x.items():
            L.append(f"| {d} | `{m['candidate']}` | {tag} | {m['missed']} | "
                     + " | ".join(str(m["by_cause"][c]) for c in U2E.CAUSES) + " |")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _selftest():
    import zu2b
    import step4_pool_reach as S4
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_pools import FamilyStore
    rng = np.random.default_rng(18)
    if tuple(S4.FAMILIES) != FAMILIES:
        raise AssertionError("step 4's families are not section 19's")
    for trial in range(60):
        n = int(rng.integers(3, 50))
        m = int(rng.integers(0, 4 * n))
        s, t = rng.integers(0, n, m), rng.integers(0, n, m)               # duplicates and self-loops on purpose
        ip, col = zu2b.undirected_csr(n, s, t)
        brute = [sorted({int(b) for a, b in zip(s, t) if a == x} | {int(a) for a, b in zip(s, t) if b == x})
                 for x in range(n)]
        assert np.array_equal(np.diff(ip), [len(r) for r in brute]), trial
        assert np.array_equal(col, np.asarray([v for r in brute for v in r], np.int32)), trial
        # three toy families; E1 in the ner slot changes the all union only, and the package's NER gives it back
        fam = {}
        for f in FAMILIES:
            ms = int(rng.integers(0, 3 * n))
            a_, b_ = zu2b.undirected_csr(n, rng.integers(0, n, ms), rng.integers(0, n, ms))
            fam[f] = FamilyStore(f, n, a_, b_, weight=np.ones(b_.size, np.float16), edges_stored=ms)
        e1 = FamilyStore("ner", n, ip, col, weight=np.ones(col.size, np.float16), edges_stored=m)
        unions_w = {u: S4B.RegimeUnion(fam, list(fs), n) for u, fs in UNIONS}
        stores_e, unions_e = e1_unions(fam, unions_w, e1, S4B, n)
        assert unions_e["st"] is unions_w["st"] and stores_e["ner"] is e1 and fam["ner"] is not e1
        want = np.diff(fam["structural"].indptr) + np.diff(ip) + np.diff(fam["knn"].indptr)
        assert np.array_equal(unions_e["all"].deg, want), trial
        _s2, back = e1_unions(fam, unions_w, fam["ner"], S4B, n)
        assert all(np.array_equal(x, y) for x, y in zip(back["all"].args, unions_w["all"].args)), trial
        assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
                   for x, y in zip(adjacency(stores_e), [(fam["structural"].indptr, fam["structural"].col),
                                                        (ip, col), (fam["knn"].indptr, fam["knn"].col)])), trial
    # the empty slot: every row empty, the walk's degree is the other two families'
    n = 7
    empty = FamilyStore("ner", n, np.zeros(n + 1, np.int64), np.empty(0, np.int32), weight=np.ones(0, np.float16))
    fam = {f: FamilyStore(f, n, *zu2b.undirected_csr(n, np.arange(n - 1), np.arange(1, n))) for f in FAMILIES}
    _s, ue = e1_unions(fam, {u: S4B.RegimeUnion(fam, list(fs), n) for u, fs in UNIONS}, empty, S4B, n)
    assert np.array_equal(ue["all"].deg, 2 * np.diff(fam["structural"].indptr))
    # the summary and the verdicts
    a = {"bs": np.asarray([3, 4]), "inc": np.asarray([[5, 6], [7, 7]]), "fill": np.asarray([2, 3]),
         "lens": np.ones((2, NO), np.int32), "kt_same": np.asarray([True, False]),
         "pos": np.asarray([[0] * NO, [-1] * NO, [3] * NO], np.int32),
         "in_I": np.asarray([[True, True], [False, True], [True, True]]), "ptr": np.asarray([0, 2, 3])}
    sm = summary_of(a)
    assert sm["kt_same"] == 0.5 and sm["I"]["filed"]["ALL"] == 0.5 and sm["I"]["E1"]["ALL"] == 1.0
    assert sm["orders"][NAMES[0]]["ALL"] == 0.5 and sm["fill"] == 2.5
    assert decide(0.96, 2000.0, 0.955, 1812.6) == "E1_RAISED"
    assert decide(0.955, 1800.0, 0.955, 1812.6) == "E1_CHEAPER"
    assert decide(0.955, 1812.6, 0.955, 1812.6) == "NO_GAIN"
    assert decide(0.95, 1000.0, 0.955, 1812.6) == "NO_GAIN"
    assert decide(None, None, 0.955, 1812.6) == "NO_GAIN"
    assert ST_SOURCES == (0, 2, 4) and VERDICTS == ("E1_RAISED", "E1_CHEAPER", "NO_GAIN")
    log("zu2m_e1 self-test: undirected_csr against a brute force (60 toy graphs, duplicates and self-loops), E1 in "
        "the ner slot changes the all union only (the st union is the filed one, the package's NER gives the filed "
        "union back), the empty slot, the summary, the verdicts: passed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("stats", "merge", "verdict"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--shard", help="I/K: the I-th (from 0) of K contiguous question ranges")
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return 0
    if a.stage in ("stats", "merge") and not a.dataset:
        raise SystemExit(f"{a.stage} needs DATASET")
    if a.stage == "stats":
        shard = None
        if a.shard:
            i, k = (int(x) for x in a.shard.split("/"))
            shard = (i, k)
        return stats_stage(a.dataset, a.threads, a.limit, shard, a.control)
    if a.stage == "merge":
        return merge_stage(a.dataset, a.parts)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
