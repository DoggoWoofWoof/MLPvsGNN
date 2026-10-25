"""U2o: U2e's pool rule with K1 as the I's row order (docs/U2O_K1_I_ORDER.md). Pools only: nothing trains, and no
row order, construction, budget or cut reads a question or a gold.

    python outputs/mp_unified/zu2o.py graph DATASET
    python outputs/mp_unified/zu2o.py stats DATASET [--threads 2] [--limit Q] [--shard I/K]
    python outputs/mp_unified/zu2o.py merge DATASET --parts K
    python outputs/mp_unified/zu2o.py verdict
    python outputs/mp_unified/zu2o.py --selftest

graph    musique, hotpotqa or 2wiki: K1's columns of the chosen W2_S (U2k, section 4: each row's entries by s(u, v),
         highest first, sibling-only entries last, ties by node id), every row, with the zero-shot scorer U2k grades
         the dataset with. Writes outputs/u2o/graphs/<D>_K1.npz and .json.
stats    One of those datasets' s1sel. Each question walks once, as zu2m.py's stats walk, on W2_S in node-id order.
         The own construction's I is rebuilt on W2_S's order (checked against prepare's pools and U2m's filed own
         kind_t columns) and on K1's, and the own cell's kind_t columns (fill, lens, pos) of the K1 I are recorded.
         --limit Q: the first Q questions, every check, nothing written and no number logged. --shard I/K: the I-th
         of K contiguous question ranges, into outputs/u2o/stats/parts/.
merge    Join a dataset's K parts into the file one run over every question writes.
verdict  U2e's 13,860 candidates on the own construction with kind_t, on U2m's arrays with the K1 columns spliced in
         on the three datasets; section 3's rule and verdict. Writes outputs/u2o/verdict.json and verdict.md.
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

U2E = M.U2E
U2C, PG, F, P1H, D5, D5B = U2E.U2C, U2E.PG, U2E.F, U2E.P1H, U2E.D5, U2E.D5B
log, utc, write_json, sha_file = U2E.log, U2E.utc, U2E.write_json, U2E.sha_file
DECLARED = "docs/U2O_K1_I_ORDER.md"
OUT = ROOT / "outputs" / "u2o"
SIX, TRAIN5 = U2E.SIX, U2E.TRAIN5
NAMES, NO, R, BASES, EMPTY = M.NAMES, M.NO, M.R, M.BASES, M.EMPTY
K1_SETS = ("musique", "hotpotqa", "2wiki")         # the passage datasets whose construction reads links
SCORER_SHA256 = {"2wiki": "7fdfc8ab64c2cbc33ac2b4d9d6522eb94582fb074e6933fa87d05e8189ed9a0d",
                 "hotpotqa": "c7441681cf1e7746fc0cb250dfa9d36842f6cd9efeb99ec20cf080bd980c0a57"}   # as in 2ee7446
CARRIED = ("own/kind_t/" + M.U2E_PICK, 0.955, 1812.6)  # U2e's pick, level and cost (U2m, sections 12 and 15)
VERDICTS = ("K1_RAISED", "K1_CHEAPER", "NO_GAIN")
ROW_BLOCK = 4_000_000                               # entries per reorder block (whole rows)
O_KT, O_KTN1 = NAMES.index("kt"), NAMES.index("kt_n1")
PER_QUESTION = ("bs", "inc", "fill", "lens")
PER_GOLD = ("pos", "in_I")
ARRAYS = PER_QUESTION + PER_GOLD + ("ptr",)
HEAD_KEYS = ("declared_in", "stage", "dataset", "carve", "graph_detail", "k1_npz_sha256", "script_sha256",
             "zu2m_sha256", "zu2e_sha256", "u2m_stats_sha256", "verdict_ce_sha256", "p1e_script_sha256", "threads",
             "own_cell", "orders")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def ce_verdict():
    """(c) and (e)'s filed verdict, checked to be the committed zu2m.py's, with U2e's pick carried as filed."""
    p = M.OUT / "verdict_ce.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("script_sha256") != M.script_sha256() or rec.get("zu2e_sha256") != U2E.script_sha256():
        raise SystemExit("outputs/u2m/verdict_ce.json is not the committed zu2m.py's verdict")
    u = rec["u2e_pick"]
    if rec["carried"] != "U2e" or (u["name"], float(u["level"]), float(u["cost"])) != CARRIED:
        raise SystemExit("(c) and (e)'s verdict does not carry U2e's pick at the level and cost section 1 names")
    vb = json.loads((M.OUT / "b" / "verdict_b.json").read_text(encoding="utf-8"))
    if vb["verdict_b"] != "NO_GAIN" or vb["pick"] != f"{CARRIED[0]}|G0" or int(vb["G"]) != 0:
        raise SystemExit("(b)'s filed verdict does not keep U2e's pick at G = 0")
    return rec


# ── K1's row order (U2k, section 4), in blocks of whole rows ─────────────────


def row_blocks(ip, maxe):
    """[(a, b)]: consecutive row ranges covering every row, each of at most maxe entries or one row."""
    n = ip.size - 1
    out, a = [], 0
    while a < n:
        b = int(np.searchsorted(ip, ip[a] + maxe, side="right")) - 1
        b = min(max(b, a + 1), n)
        out.append((a, b))
        a = b
    return out


def k1_columns(ip, col, key, s, sibkey, n, pair_scores, block=ROW_BLOCK):
    """K1's col: every row's entries by (row, -s(u, v), node id), zu2j.reorder's sort, built block by block. Checks
    that each row is in node-id order (zero keys return col) and that each K1 row holds the row's own entries.
    Returns (K1's col, rows whose order changed)."""
    new = np.empty_like(col)
    changed = 0
    for a, b in row_blocks(ip, block):
        rows = np.arange(a, b, dtype=np.int64)
        pos, r, v, ps = pair_scores(ip, col, rows, key, s, sibkey, n)
        if v.size and not np.array_equal(v[np.lexsort((v, np.zeros(v.size, np.int64), r))], v):
            raise SystemExit("W2_S's rows are not in node-id order; REFUSED")
        o = np.lexsort((v, -ps, r))
        w = v[o]
        new[pos] = w.astype(col.dtype)
        if not np.array_equal(w[np.lexsort((w, r))], v):
            raise SystemExit("a K1 row does not hold its W2_S row's entries")
        if v.size:
            moved = np.bincount(r, weights=(w != v).astype(np.float64), minlength=rows.size)
            changed += int((moved > 0).sum())
        del pos, r, v, ps, o, w
    return new, changed


def graph_stage(d):
    import zu2k as K2                     # U2k's links, scores and pair scores, unchanged
    if d not in K1_SETS:
        raise SystemExit(f"{d}: K1 is built on {K1_SETS} only")
    t0 = time.time()
    sname = K2.GRADED_WITH[d]
    fit = K2.FIT_ON[sname]
    sp = K2.OUT / f"scorer_{fit}.json"
    if sha_file(sp) != SCORER_SHA256[fit]:
        raise SystemExit(f"{sp}: not U2k's filed scorer")
    sc = json.loads(sp.read_text(encoding="utf-8"))
    r2m = json.loads((M.OUT / "stats" / f"{d}__{U2C.carve_of(d)}.json").read_text(encoding="utf-8"))
    grade = json.loads((K2.OUT / d / "grade_s1sel.json").read_text(encoding="utf-8"))
    L = K2.Links(d)
    n = L.n
    g = U2C.graph_csr(d, "chosen", n)
    if g is None:
        raise SystemExit(f"{d}: no chosen graph")
    ip, col, desc = g
    if desc != r2m["graph_detail"] or desc["graph"] != "W2_S":
        raise SystemExit(f"{d}: the chosen graph is not W2_S, or not the file U2m's filed run read")
    s, stats = L.scores(sc)
    if grade["scorer"] != sname or json.loads(json.dumps(stats)) != grade["standardization"]:
        raise SystemExit(f"{d}: the scores are not the ones U2k graded the dataset with")
    with np.load(K2.U2I / d / "links.npz") as z:
        sib_s, sib_d = z["sib_src"].astype(np.int64), z["sib_dst"].astype(np.int64)
    sibkey = np.unique(np.concatenate([sib_s * n + sib_d, sib_d * n + sib_s]))
    del sib_s, sib_d
    log(f"{d}: scorer {sname} (fitted on {fit}); W2_S {col.size:,} entries ({time.time() - t0:.0f} s)")
    new, changed = k1_columns(ip, col, L.key, s, sibkey, n, K2.pair_scores)
    od = OUT / "graphs"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"{d}_K1.npz"
    tmp = od / f"{d}_K1.tmp.npz"
    np.savez_compressed(tmp, indptr=ip, col=new)
    os.replace(tmp, p)
    import u1b
    write_json(od / f"{d}_K1.json", {
        "declared_in": DECLARED, "stage": "U2o graph", "dataset": d, "source": desc, "scorer": sname,
        "scorer_fit_on": fit, "scorer_sha256": SCORER_SHA256[fit], "standardization_equal_u2k": True,
        "rows": int(n), "entries": int(col.size), "rows_reordered": int(n), "rows_changed": changed,
        "npz": str(p.relative_to(ROOT)).replace("\\", "/"), "npz_sha256": sha_file(p),
        "script_sha256": script_sha256(), "zu2k_sha256": u1b.sha_src(K2.__file__),
        "zu2j_sha256": u1b.sha_src(K2.zu2j.__file__), "seconds": round(time.time() - t0, 1), "utc": utc()})
    log(f"{d}: K1 written ({time.time() - t0:.0f} s)")
    return 0


def load_k1(d):
    """(record, indptr, col) of a dataset's K1 graph, checked to be this script's and its file."""
    rp = OUT / "graphs" / f"{d}_K1.json"
    if not rp.exists():
        raise SystemExit(f"{rp}: run `zu2o.py graph {d}` first")
    rec = json.loads(rp.read_text(encoding="utf-8"))
    p = ROOT / rec["npz"]
    if rec.get("script_sha256") != script_sha256() or sha_file(p) != rec["npz_sha256"]:
        raise SystemExit(f"{rp}: not this script's K1, or not its file")
    with np.load(p) as z:
        return rec, z["indptr"], z["col"]


# ── stats (the laptop, or elsewhere in shards) ───────────────────────────────


def columns_of(full, x, iout, gj, in_b, rows, q, constant, rrf, B, n_i):
    """The own cell's kind_t columns for one I: (fill, lens over U2e's 14 orders, every gold's place in each)."""
    ords, _sizes, U, mem = U2E.question_orders(full, x["n1"], x["cr"], x["cos1"], x["r0"], iout, x["n2"], x["first"],
                                               rows.read, q, constant, rrf)
    if not U2E.orders_exact(ords, B, full, x["n1"], U, mem):
        raise SystemExit("an order of the own cell's kind_t is not exactly its sources' nodes")
    lens = np.asarray([ords[nm].size for nm in NAMES], np.int32)
    pos = np.stack([U2E.places(gj, in_b, ords[nm]) for nm in NAMES], axis=1).astype(np.int32)
    return M.fill_of(full, n_i, B.size), lens, pos


def stats_stage(name, threads=2, limit=0, shard=None):
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    C4 = E.C4
    if name not in K1_SETS:
        raise SystemExit(f"{name}: U2o's stats run on {K1_SETS}; the others are U2m's arrays as filed")
    if D.SOURCES[4] != "R0" or F.CONSTRUCTION != "KIND_T":
        raise SystemExit("P1d's fifth source is not R0, or P1f's construction is not KIND_T")
    carve = U2C.carve_of(name)
    numba.set_num_threads(threads)
    t0 = time.time()
    vrec = ce_verdict()
    x2m = M.load_u2m(name)
    if x2m is None:
        raise SystemExit(f"{name}: U2m's filed arrays are not under outputs/u2m/stats")
    a2m, r2m = x2m
    if vrec["stats"][name]["file_sha256"] != r2m["file_sha256"]:
        raise SystemExit(f"{name}: U2m's filed arrays are not the ones (c) and (e)'s verdict read")
    grec, kip, kcol = load_k1(name)
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    cells, own_k = M.cells_of(construction)
    if cells != r2m["cells"] or own_k != r2m["own_cell"]:
        raise SystemExit(f"{name}: the cells are not U2m's")
    reg_f = list(op.m3b_compile.regime_families(op.cfg_h, construction["regime"])) \
        if construction["regime"] != "RETRIEVAL" else []
    if "structural" not in reg_f or "structural" not in families:
        raise SystemExit(f"{name}: the construction does not read the links")
    n = int(ds.n_nodes)
    ip, col, desc = U2C.graph_csr(name, "chosen", n)
    if desc != r2m["graph_detail"] or grec["source"] != desc:
        raise SystemExit(f"{name}: the graph is not the one U2m's filed run read, or not K1's source")
    if not np.array_equal(kip, ip) or kcol.size != col.size:
        raise SystemExit(f"{name}: K1's rows are not W2_S's")
    swap = U2C.csr_store(n, ip, col)
    k1 = U2C.csr_store(n, kip, kcol)
    del ip, col, kip, kcol
    stores = dict(stores0)
    stores["structural"] = swap
    union = S4B.RegimeUnion(stores, families, n)
    load0 = S4.m3b_pools.load_or_build_store

    def load(ds_, family, cache_dir):
        if family == "structural":
            if ds_.name != name:
                raise SystemExit(f"{ds_.name}: U2o swaps {name}'s graph only")
            return swap
        return load0(ds_, family, cache_dir)

    S4.m3b_pools.load_or_build_store = load
    pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c == carve]
    (_carve, pop), = pops
    index = E.build_index(E.node_names(ds), E.stopwords())
    texts = E.query_texts(ds)
    if len(texts) != int(ds.n_queries):
        raise SystemExit(f"{name}: {len(texts)} query records, not {ds.n_queries}")
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    prep0 = op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores0, op.m3a, op.m3b_contract)[0]
    prep = op.m3b_compile.prepare(ds, [pop], construction, op.cfg_h, stores, op.m3a, op.m3b_contract)[0]
    if any(not np.array_equal(np.asarray(x, np.int64), np.asarray(y, np.int64)) for x, y in zip(prep.seeds, prep0.seeds)):
        raise SystemExit(f"{name}/{carve}: the seeds on the graph are not G0's")
    base_own = op.m3b_compile.base_rows(construction, prep0.dense_ids, prep0.splade_ids, op.m3a, op.m3b_contract,
                                        constant)
    inc_own = [np.asarray(p, np.int64) for p in prep.pools]
    setting = construction["setting"]
    fam_w = [stores[f] for f in reg_f]
    fam_k = [k1 if f == "structural" else stores[f] for f in reg_f]
    linked = E.linked_for(name, ds, pop, index, texts)
    log(f"{name}/{carve}: prepared; own cell {cells[own_k]}, families {reg_f} ({time.time() - t0:.0f}s)")
    cache = {}
    stores_all, unions = G.open_unions(cache, op, name, union)
    adj = [(np.ascontiguousarray(stores_all[f].indptr, np.int64), np.ascontiguousarray(stores_all[f].col))
           for f in S4.FAMILIES]
    n_st = int(unions["st"].deg.size)
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
            seeds = np.asarray(prep.seeds[j], np.int64)
            q = D5B.unit(np.asarray(prep.qemb[j], np.float64))
            w_all = K.steer(prep, j, constant)
            rows = M.Rows(nodes, verify=bool(limit))
            gj = golds[j]
            g0, g1 = int(ptr[j]), int(ptr[j + 1])
            base = np.asarray(base_own[j], np.int64)
            B = np.unique(np.union1d(base, seeds))
            n1, _cnt, _ws, _up, _inv = D5B.n1_table(adj, B, D5.weights_of(w_all, B))
            x = {"n1": n1, "r0": F.outside(src[4][0], B)}
            rows.add(n1)
            x["cos1"] = D5B.unit(rows.read(n1)) @ q if n1.size else np.empty(0, np.float32)
            x["cr"] = D5B.ranks_desc(x["cos1"], n1) if n1.size else EMPTY
            x["n2"], x["first"], _taken, _hubs = U2E.second_link(adj, n1, x["cr"], B)
            I_w = np.asarray(op.m3b_compile.build_pool(base, seeds, S4.m3b_pools.expand_hops(seeds, fam_w, setting))[0],
                             np.int64)
            I_k = np.asarray(op.m3b_compile.build_pool(base, seeds, S4.m3b_pools.expand_hops(seeds, fam_k, setting))[0],
                             np.int64)
            if not np.array_equal(I_w, inc_own[j]):
                raise SystemExit(f"{name}/{carve} question {j}: the I rebuilt on W2_S's order is not prepare's pools")
            if not np.isin(seeds, B).all() or not np.isin(B, I_w).all() or not np.isin(B, I_k).all():
                raise SystemExit(f"{name}/{carve} question {j}: B or an I is malformed")
            rows.add(np.concatenate([kt, src[4][0], I_w, I_k, x["n2"]]))
            full = F.outside(kt, B)
            in_b = np.isin(gj, B)
            fw, lw, pw = columns_of(full, x, F.outside(I_w, B), gj, in_b, rows, q, constant, D.rrf, B, I_w.size)
            if not (B.size == int(a2m["bs"][j, own_k]) and I_w.size == int(a2m["inc"][j, own_k])
                    and fw == int(a2m["fill"][j, own_k * R]) and np.array_equal(lw, a2m["lens"][j, own_cols])
                    and np.array_equal(pw, a2m["pos"][g0:g1, own_cols])
                    and np.array_equal(np.isin(gj, I_w), a2m["in_I"][g0:g1, own_k])):
                raise SystemExit(f"{name}/{carve} question {j}: the own kind_t columns on W2_S's order are not U2m's")
            if np.array_equal(I_k, I_w):
                fk, lk, pk = fw, lw, pw
            else:
                fk, lk, pk = columns_of(full, x, F.outside(I_k, B), gj, in_b, rows, q, constant, D.rrf, B, I_k.size)
            if not (np.array_equal(lk[[O_KT, O_KTN1]], lw[[O_KT, O_KTN1]])
                    and np.array_equal(pk[:, [O_KT, O_KTN1]], pw[:, [O_KT, O_KTN1]])):
                raise SystemExit(f"{name}/{carve} question {j}: kt or kt_n1 changed with the I")
            pq["bs"].append(B.size)
            pq["inc"].append([I_w.size, I_k.size])
            pq["fill"].append(fk)
            pq["lens"].append(lk)
            pg["pos"].append(pk)
            pg["in_I"].append(np.stack([np.isin(gj, I_w), np.isin(gj, I_k)], axis=1))
            done += 1
            if done % 100 == 0:
                log(f"{name}/{carve}: {done}/{q1 - q0} ({time.time() - t:.0f}s)")
    arr = arrays_of(pq, pg, ptr[q0:q1 + 1])
    head = {"declared_in": DECLARED, "stage": "U2o", "dataset": name, "carve": carve, "graph_detail": desc,
            "k1_npz_sha256": grec["npz_sha256"], "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(),
            "zu2e_sha256": U2E.script_sha256(), "u2m_stats_sha256": r2m["file_sha256"],
            "verdict_ce_sha256": sha_file(M.OUT / "verdict_ce.json"), "p1e_script_sha256": K.script_sha256(),
            "threads": threads, "own_cell": own_k, "orders": list(NAMES)}
    checks = {"graph_equal_u2m": True, "k1_rows_are_w2s_rows": True, "seeds_equal_g0": True, "kind_t_by_kind": True,
              "I_on_w2s_equal_prepare": True, "own_kind_t_on_w2s_equal_u2m": True, "orders_exact": True,
              "kt_kt_n1_unchanged": True, "rows_served_verified": bool(limit)}
    if limit:
        log(f"{name}/{carve}: development run on questions {q0}-{q1 - 1} of {nq} ({time.time() - t:.0f}s): every "
            "check passed, every served row verified; nothing written")
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
         "pos": np.concatenate(pg["pos"]) if pg["pos"] else np.zeros((0, NO), np.int32),
         "in_I": np.concatenate(pg["in_I"]) if pg["in_I"] else np.zeros((0, 2), bool),
         "ptr": np.asarray(ptr, np.int64)}
    return {k: a[k] for k in ARRAYS}


def summary_of(a):
    """The I's ALL and mean size on W2_S's order and on K1's; K1's mean fill and each order's whole reach and length."""
    ptr = a["ptr"] - a["ptr"][0]
    ng = np.diff(ptr)
    qi = np.repeat(np.arange(ng.size), ng)
    has = ng > 0
    out = {"I": {}, "fill": round(float(a["fill"].mean()), 1), "orders": {}}
    for i, g in enumerate(("W2_S", "K1")):
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


def load_o(name):
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


def splice(a, k, x):
    """U2m's arrays with cell k's kind_t columns (fill, lens, pos) replaced by x's; every other column as filed."""
    out = {kk: np.array(v, copy=True) for kk, v in a.items()}
    cols = [M.column(k, 0, o) for o in range(NO)]
    out["fill"][:, k * R] = x["fill"]
    out["lens"][:, cols] = x["lens"]
    out["pos"][:, cols] = x["pos"]
    return out


def decide(level, cost, car_level, car_cost):
    if level is None:
        return "NO_GAIN"
    if level > car_level + 1e-9:
        return "K1_RAISED"
    if abs(level - car_level) <= 1e-9 and cost < car_cost - 1e-9:
        return "K1_CHEAPER"
    return "NO_GAIN"


def misses(a, k, cand, t_name):
    """A candidate's misses on one dataset's arrays by U2m's causes (TAU_CUT, BUDGET, UNREACHED)."""
    _k, r, o, b, t = cand
    rows, held = M.held_of(a, k, r, o, b, t)
    _rw, held1 = M.held_of(a, k, r, o, b, 1.0)
    reach = (a["pos"] >= 0).any(axis=1)
    cause = np.where(held1, "TAU_CUT", np.where(reach, "BUDGET", "UNREACHED"))
    return {"candidate": t_name, "missed": int((~held).sum()),
            "by_cause": {c: int((~held & (cause == c)).sum()) for c in U2E.CAUSES}}


def verdict_stage():
    t0 = time.time()
    vrec = ce_verdict()
    car_name, car_level, car_cost = CARRIED
    cap = float(vrec["cap"])
    w = F.weights()
    if w != vrec["weights"]:
        raise SystemExit("P1f's weights are not the ones (c) and (e)'s verdict used")
    ceil = np.asarray([vrec["ceilings"][d] for d in TRAIN5], np.float64)
    am, rm, ao, ro, missing = {}, {}, {}, {}, []
    for d in SIX:
        a, r = M.load_u2m(d)
        if vrec["stats"][d]["file_sha256"] != r["file_sha256"]:
            raise SystemExit(f"{d}: U2m's arrays are not the ones (c) and (e)'s verdict read")
        am[d], rm[d] = a, r
        if d in K1_SETS:
            x = load_o(d)
            if x is None:
                missing.append(d)
                continue
            ao[d], ro[d] = x
            if ro[d]["u2m_stats_sha256"] != r["file_sha256"] or ro[d]["own_cell"] != r["own_cell"] or \
                    not np.array_equal(ao[d]["ptr"], a["ptr"]) or \
                    not np.array_equal(ao[d]["bs"], a["bs"][:, r["own_cell"]]) or \
                    not np.array_equal(ao[d]["inc"][:, 0], a["inc"][:, r["own_cell"]]):
                raise SystemExit(f"{d}: U2o's arrays are not on U2m's questions, golds, B and I")
    if missing:
        raise SystemExit(f"INCOMPLETE: no U2o stats for {missing}")
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
    tw, tk, ak = {}, {}, {}
    for d in SIX:
        k = rm[d]["own_cell"]
        ak[d] = splice(am[d], k, ao[d]) if d in K1_SETS else am[d]
        tw[d] = M.table_arrays(am[d], [k], [0])
        tk[d] = M.table_arrays(ak[d], [k], [0])
        for i, c in enumerate(c2e):
            if tw[d][0][i] != t2e[c[0]][d]["ALL"] or tw[d][1][i] != t2e[c[0]][d]["rows"]:
                raise SystemExit(f"{d}: the table on W2_S's order is not U2e's at {c[0]}")
            if d not in K1_SETS and (tk[d][0][i] != tw[d][0][i] or tk[d][1][i] != tw[d][1][i]):
                raise SystemExit(f"{d}: the table changed on a dataset K1 does not touch")
    log(f"U2o verdict: U2e's table reproduced on W2_S's order ({time.time() - t0:.0f}s)")

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
            raise SystemExit(f"level {t}: the frontier on W2_S's order is {got}, not the filed own kind_t frontier's")
    i2e = names.index(car_name)
    if lv_w != car_level or kk_w != i2e or float(cr_w[i2e]) != car_cost:
        raise SystemExit("on W2_S's order the rule does not return U2e's pick at its level and cost")
    fr, cost_r, (lv, kk) = frontier_of(tk)
    v = decide(lv, None if kk is None else float(cost_r[kk]), car_level, car_cost)
    pick = names[kk] if v != "NO_GAIN" else car_name

    def point(t, tgt, k_, tab, cr_):
        if k_ is None:
            return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": None}
        return {"level": t, "targets": dict(zip(TRAIN5, tgt.tolist())), "pick": names[k_], "cost": float(cr_[k_]),
                "within_cap": bool(cr_[k_] <= cap + 1e-9),
                "s1sel": {d: {"ALL": float(tab[d][0][k_]), "rows": float(tab[d][1][k_])} for d in TRAIN5},
                "webqsp_s1eval": {"ALL": float(tab["webqsp"][0][k_]), "rows": float(tab["webqsp"][1][k_])}}

    # beside: the I, U2e's point under K1, the misses by cause
    u2k = {}
    for d in K1_SETS:
        gp = ROOT / "outputs" / "u2k" / d / "grade_s1sel.json"
        g = json.loads(gp.read_text(encoding="utf-8"))
        u2k[d] = {f: g["forms"][f]["I"] for f in ("R0", "K1")}
    the_I = {d: {**ro[d]["summary"]["I"], "u2k_filed_I": u2k[d]} for d in K1_SETS}
    at_u2e = {d: {"W2_S": {"ALL": float(tw[d][0][i2e]), "rows": float(tw[d][1][i2e])},
                  "K1": {"ALL": float(tk[d][0][i2e]), "rows": float(tk[d][1][i2e])}} for d in SIX}
    why = {}
    for d in SIX:
        k = rm[d]["own_cell"]
        cw = (k,) + cands[i2e][1:]
        why[d] = {"u2e_pick_W2_S": misses(am[d], k, cw, car_name), "u2e_pick_K1": misses(ak[d], k, cw, car_name)}
        if kk is not None and kk != i2e:
            why[d]["k1_pick"] = misses(ak[d], k, (k,) + cands[kk][1:], names[kk])
    rec = {"declared_in": DECLARED, "stage": "U2o", "script_sha256": script_sha256(), "zu2m_sha256": M.script_sha256(),
           "zu2e_sha256": U2E.script_sha256(), "verdict_ce_sha256": sha_file(M.OUT / "verdict_ce.json"),
           "verdict": v, "pick": pick, "level": lv, "cost": None if kk is None else float(cost_r[kk]),
           "frontier_pick": None if kk is None else names[kk],
           "carried": {"pick": car_name, "level": car_level, "cost": car_cost}, "cap": cap, "weights": w,
           "ceilings": dict(zip(TRAIN5, ceil.tolist())), "candidates": len(cands), "k1_datasets": list(K1_SETS),
           "frontier_K1": [point(t, tg, k_, tk, cost_r) for t, tg, k_ in fr],
           "frontier_W2_S": [point(t, tg, k_, tw, cr_w) for t, tg, k_ in fr_w],
           "the_I": the_I, "u2e_point": at_u2e, "misses": why,
           "stats": {d: {"file_sha256": ro[d]["file_sha256"], "seconds": ro[d]["seconds"],
                         "k1_npz_sha256": ro[d]["k1_npz_sha256"]} for d in K1_SETS},
           "utc": utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "verdict.json", rec)
    (OUT / "verdict.md").write_text(report_md(rec), encoding="utf-8", newline="\n")
    log(f"U2o: {v}; pick {pick} at level {lv} ({time.time() - t0:.0f}s)")
    return 0


def report_md(rec):
    c = rec["carried"]
    L = [f"# U2o: {rec['verdict']}", "",
         f"Declared in {rec['declared_in']}. Pick: `{rec['pick']}`. The rule's pick under K1: "
         f"`{rec['frontier_pick']}` at level {rec['level']}, cost {rec['cost']}. Carried: `{c['pick']}` at level "
         f"{c['level']}, cost {c['cost']}. Cap {rec['cap']}.", "",
         "## The I (beside)", "", "| dataset | ALL on W2_S | size | ALL on K1 | size | U2k's filed R0 | U2k's filed K1 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for d, x in rec["the_I"].items():
        L.append(f"| {d} | {x['W2_S']['ALL']} | {x['W2_S']['size']} | {x['K1']['ALL']} | {x['K1']['size']} | "
                 f"{x['u2k_filed_I']['R0']['ALL']} | {x['u2k_filed_I']['K1']['ALL']} |")
    L += ["", "## U2e's point on W2_S's order and on K1's", "", "| dataset | W2_S ALL | rows | K1 ALL | rows |",
          "|---|---:|---:|---:|---:|"]
    for d, x in rec["u2e_point"].items():
        L.append(f"| {d} | {x['W2_S']['ALL']} | {x['W2_S']['rows']} | {x['K1']['ALL']} | {x['K1']['rows']} |")
    for title, key in (("K1", "frontier_K1"), ("W2_S", "frontier_W2_S")):
        L += ["", f"## The frontier on {title}'s order", "",
              "| level | cost | within cap | pick | " + " | ".join(f"{d} ALL (rows)" for d in TRAIN5) + " | webqsp |",
              "|---|---:|---|---|" + "---:|" * (len(TRAIN5) + 1)]
        for e in rec[key]:
            if e["pick"] is None:
                L.append(f"| {e['level']} | | | none | " + " | ".join("" for _ in TRAIN5) + " | |")
                continue
            L.append(f"| {e['level']} | {e['cost']} | {e['within_cap']} | `{e['pick']}` | "
                     + " | ".join(f"{e['s1sel'][d]['ALL']} ({e['s1sel'][d]['rows']})" for d in TRAIN5)
                     + f" | {e['webqsp_s1eval']['ALL']} ({e['webqsp_s1eval']['rows']}) |")
    L += ["", "## Misses by cause (beside)", "", "| dataset | candidate | order | missed | TAU_CUT | BUDGET | UNREACHED |",
          "|---|---|---|---:|---:|---:|---:|"]
    for d, x in rec["misses"].items():
        for tag, m in x.items():
            L.append(f"| {d} | `{m['candidate']}` | {tag} | {m['missed']} | "
                     + " | ".join(str(m["by_cause"][c]) for c in U2E.CAUSES) + " |")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def _selftest():
    import zu2k as K2
    zu2j = K2.zu2j
    rng = np.random.default_rng(23)
    for trial in range(80):
        n = int(rng.integers(5, 60))
        m = int(rng.integers(n, 6 * n))
        ws, wt = rng.integers(0, n, m), rng.integers(0, n, m)
        ok = ws != wt
        key = np.unique(ws[ok].astype(np.int64) * n + wt[ok])
        ws, wt = key // n, key % n
        s = rng.choice(np.asarray([-1.0, 0.0, 0.5, 2.0], np.float32), key.size)           # ties on purpose
        ns = int(rng.integers(0, 2 * n))
        a_, b_ = rng.integers(0, n, ns), rng.integers(0, n, ns)
        ok = a_ != b_
        sib_s, sib_d = a_[ok].astype(np.int64), b_[ok].astype(np.int64)
        sibkey = np.unique(np.concatenate([sib_s * n + sib_d, sib_d * n + sib_s]))
        us = np.concatenate([ws, wt, sib_s, sib_d])
        vs = np.concatenate([wt, ws, sib_d, sib_s])
        und = np.unique(us * n + vs)
        u, v = und // n, und % n
        ip = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(u, minlength=n), out=ip[1:])
        col = v.astype(np.int32)
        rows = np.arange(n, dtype=np.int64)
        pos, r, vv, ps = K2.pair_scores(ip, col, rows, key, s, sibkey, n)
        want = zu2j.reorder(col, pos, r, vv, -ps)                                         # U2k's K1, as graded
        assert np.array_equal(zu2j.reorder(col, pos, r, vv, np.zeros(vv.size, np.int64)), col)
        sd = dict(zip(key.tolist(), s.tolist()))
        brute = []
        for x in range(n):
            ent = col[ip[x]:ip[x + 1]].astype(np.int64).tolist()
            sc = {y: max(sd.get(x * n + y, -np.inf), sd.get(y * n + x, -np.inf)) for y in ent}
            brute += sorted(ent, key=lambda y: (-sc[y], y))
        assert np.array_equal(np.asarray(brute, np.int64), want.astype(np.int64)), trial
        for blk in (1, 3, 17, 10 ** 9):
            got, changed = k1_columns(ip, col, key, s, sibkey, n, K2.pair_scores, block=blk)
            assert np.array_equal(got, want) and got.dtype == col.dtype, (trial, blk)
            per = [not np.array_equal(got[ip[x]:ip[x + 1]], col[ip[x]:ip[x + 1]]) for x in range(n)]
            assert changed == sum(per), (trial, blk)
        for a, b in row_blocks(ip, 3):
            assert b > a and (ip[b] - ip[a] <= 3 or b == a + 1)
        long = np.flatnonzero(np.diff(ip) >= 2)
        if long.size:                                    # a row out of node-id order is refused
            x = int(long[0])
            bad = col.copy()
            bad[ip[x]:ip[x + 1]] = bad[ip[x]:ip[x + 1]][::-1]
            try:
                k1_columns(ip, bad, key, s, sibkey, n, K2.pair_scores)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"trial {trial}: a row out of node-id order was not refused")
    # the splice: only the own cell's kind_t columns change
    nk = 4
    nq, ngl = 7, 15
    a = {"fill": rng.integers(0, 50, (nq, nk * R)), "lens": rng.integers(0, 50, (nq, nk * R * NO)).astype(np.int32),
         "pos": rng.integers(-1, 50, (ngl, nk * R * NO)).astype(np.int32), "bs": rng.integers(1, 9, (nq, nk)),
         "ptr": np.arange(nq + 1)}
    for k in range(nk):
        x = {"fill": rng.integers(0, 50, nq), "lens": rng.integers(0, 50, (nq, NO)).astype(np.int32),
             "pos": rng.integers(-1, 50, (ngl, NO)).astype(np.int32)}
        b = splice(a, k, x)
        cols = [M.column(k, 0, o) for o in range(NO)]
        rest = np.setdiff1d(np.arange(nk * R * NO), cols)
        assert np.array_equal(b["lens"][:, cols], x["lens"]) and np.array_equal(b["pos"][:, cols], x["pos"])
        assert np.array_equal(b["fill"][:, k * R], x["fill"])
        assert np.array_equal(b["lens"][:, rest], a["lens"][:, rest]) and np.array_equal(b["pos"][:, rest],
                                                                                         a["pos"][:, rest])
        assert np.array_equal(np.delete(b["fill"], k * R, axis=1), np.delete(a["fill"], k * R, axis=1))
        assert np.array_equal(b["bs"], a["bs"]) and np.array_equal(b["ptr"], a["ptr"])
    assert decide(0.96, 2000.0, 0.955, 1812.6) == "K1_RAISED"
    assert decide(0.955, 1800.0, 0.955, 1812.6) == "K1_CHEAPER"
    assert decide(0.955, 1812.6, 0.955, 1812.6) == "NO_GAIN"
    assert decide(0.95, 1000.0, 0.955, 1812.6) == "NO_GAIN"
    assert decide(None, None, 0.955, 1812.6) == "NO_GAIN"
    assert NAMES[O_KT] == "kt" and NAMES[O_KTN1] == "kt_n1"
    log("zu2o self-test: K1's blockwise reorder against U2k's reorder and a per-row sort (80 toy graphs, siblings and "
        "ties, 4 block sizes), the splice, the verdicts: passed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=("graph", "stats", "merge", "verdict"))
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
    if a.stage in ("graph", "stats", "merge") and not a.dataset:
        raise SystemExit(f"{a.stage} needs DATASET")
    if a.stage == "graph":
        return graph_stage(a.dataset)
    if a.stage == "stats":
        shard = None
        if a.shard:
            i, k = (int(x) for x in a.shard.split("/"))
            shard = (i, k)
        return stats_stage(a.dataset, a.threads, a.limit, shard)
    if a.stage == "merge":
        return merge_stage(a.dataset, a.parts)
    if a.stage == "verdict":
        return verdict_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
