"""U2d's pools, KT and via inputs, and gate (docs/U2D_REFIT_ON_U2E.md, sections 2 to 4). Two builds on P1c2's eleven
carves (each training dataset's fit carve, musique's s1fit, and the six s1eval carves). No gold is read to build a pool
and nothing is fit here.

    python outputs/mp_unified/zu2d_pools.py pools --build base|new DATASET --carve C [--shard I/K] [--limit Q]
    python outputs/mp_unified/zu2d_pools.py merge --build base|new DATASET
    python outputs/mp_unified/zu2d_pools.py check DATASET [--shard I/K] [--limit Q]
    python outputs/mp_unified/zu2d_pools.py check-merge DATASET
    python outputs/mp_unified/zu2d_pools.py copy-bases --build base|new
    python outputs/mp_unified/zu2d_pools.py kt --build base|new DATASET --carve C
    python outputs/mp_unified/zu2d_pools.py gate --build base|new
    python outputs/mp_unified/zu2d_pools.py --selftest

pools        One carve's questions, or the I-th of K contiguous ranges of them (--shard), on the build's graph.
             base: today's graph; the expansion is KIND_T's order outside B, its first |I_q| - |B| nodes (P1c2's rule).
             new:  W2_S swapped in on the passage datasets as U2e swaps it (zu2c.graph_csr and csr_store, unchanged);
                   kt_cos and cos_0 from zu2e.question_orders with no second link; the expansion is kt_cos's first
                   keep nodes, keep = P1h's cut at 0.85 over B and kt_cos's first min(|kt_cos|, round(12 |fill|)).
             Per question it also writes P1c3's eight KT columns (p1c3_kt.columns, unchanged; the fill column marks
             the expansion) and, on the new build, the seven via columns (section 4), over the pool sorted by node.
             Checks, any failure stops the shard: B is step 4d's (step 4e's union_pools); on the new build the seeds
             are the same on both graphs, the graph is U2e's, every order is exactly its sources' nodes, and on webqsp
             s1eval (U2e's own carve) |B|, |fill|, |I|, |kt_cos|, |cos_0| and every gold's place in kt_cos and cos_0
             are U2e's filed arrays'; on the base build, where P1c2's file is here, B, |I_q|, I_q's sha256, the
             expansion, entity and route are P1c2's. --limit Q: the shard's first Q questions, every check, nothing
             written and no number logged. Parts go to outputs/u2d/<build>/pools/parts/.
merge        Join a dataset's parts, carve by carve, into outputs/u2d/<build>/pools/<D>/<D>__<carve>.npz (P1g-b's
             arrays plus today_inc_size, budget, kt_len, entity, route, changed) and pools.json (each file's sha256,
             its parts, counts, rows, ALL, recall and the R@5 and hit@1 caps). Checks: the base build's counts equal
             P1c2's manifest (questions, changed, short, the budget's mean and maximum, mean |I_q|, ALL and recall of
             I and of the pools, the entity and route shares); the new build's webqsp s1eval equals U2e's verdict.
check        The new build's rule on a training dataset's s1sel (U2e's carve), every question checked against U2e's
             filed arrays as above; writes only its check parts (outputs/u2d/new/check/parts/).
check-merge  The parts joined; the pick's ALL, recall, mean rows and p90 rows must equal U2e's verdict. Writes
             outputs/u2d/new/check/<D>.json.
copy-bases   Step 1's cache bases into the build's cache root (step 4c's copy_bases).
kt           The KT columns (and the via columns on the new build) put in the cache's row order: each cache question's
             rows must be its pool, each node once. Writes outputs/u2d/<build>/kt/<D>/<carve>/ (kt.npy, n.npy, ids.json,
             record.json; via.npy and via.json). On the base build the record says, beside, whether the kt and n
             arrays are P1c3's filed ones (REPRODUCED or DIFFERS).
gate         Before any fit, on every carve: the pools file is the manifest's and one rule built every carve; step 1's
             rows are the filed today_inc_size (each part's n array by its sha256); the cache's questions are step 1's
             and the file's, and its rows the filed pool sizes; every look shard filed a pools record for this file;
             the KT (and via) records sit on the cache's parts and root; on the new build the five check records
             PASS. Writes outputs/u2d/<build>/gate.json; exits 1 on a failure.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
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

import zu2e as U2E  # noqa: E402  (U2e's orders, budget, places and filed arrays; through it U2c's graph swap)
import p1gb_pools as PGB  # noqa: E402  (P1g-b's pools file and replacement; numba-free at import)

U2C = U2E.U2C
PG, F, P1H, D5, D5B = U2E.PG, U2E.F, U2E.P1H, U2E.D5, U2E.D5B
PC2, KT3 = PGB.PC2, PGB.KT3
C1, C4S = PC2.C1, PC2.C4S
log, utc, write_json, sha_file = C4S.log, C4S.utc, C4S.write_json, C4S.sha_file
DECLARED = "docs/U2D_REFIT_ON_U2E.md"
OUT = ROOT / "outputs" / "u2d"
BUILDS = ("base", "new")
SIX, TRAIN5, SCREEN = PC2.SIX, PC2.TRAIN5, PC2.SCREEN
PICK = "kt_cos:ada_f12_k0@0.85"           # U2e's pick, checked against outputs/u2e/verdict.json
PICK_ORDER, PICK_F, PICK_K0, PICK_TAU = "kt_cos", 12.0, 0, 0.85
RULE = {"base": PC2.CONSTRUCTION, "new": PICK}
VIA_COLS = ("ktcos", "cos", "n1", "n1cnt", "n1w", "r0", "inI")
VIA_PMAX = 1 << 20                        # ktcos and cos: log1p(place) / log1p(2^20), at most 1
VIA_CNT = 64.0                            # n1cnt and n1w: log1p(.) / log1p(64), at most 1
VIA_W = 30.5                              # 30.5 x P1e's steer weight is 1 for a node first in both lists (c = 60)
KT_DT = np.float16
EMPTY = PG.EMPTY
PER_Q = ("inc_size", "bs_size", "count", "today_count", "fill_want", "fill_k", "x_count", "today_inc_size", "budget",
         "kt_len", "entity", "route", "changed", "n_gold", "n_held", "n_held_I")
HEAD_KEYS = ("declared_in", "stage", "build", "dataset", "carve", "construction", "graph_detail", "carve_ids_sha256",
             "questions_all", "script_sha256", "zu2e_sha256", "zu2c_sha256", "p1gb_sha256", "p1c2_sha256",
             "p1c3_sha256", "p1e_sha256", "zp1h_sha256", "u2e_verdict_sha256", "p1c2_file_sha256",
             "freeze_RECORD_SHA256", "threads")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def roots(build):
    if build not in BUILDS:
        raise SystemExit(f"--build is one of {BUILDS}, not {build}")
    b = OUT / build
    return {"pools": b / "pools", "parts": b / "pools" / "parts", "look": b / "look", "cache": b / "cache",
            "kt": b / "kt", "gate": b / "gate.json", "check": b / "check"}


def construction_of(build):
    return f"U2D:{build}:{RULE[build]}"


def u2e_pick(path=None):
    """U2e's verdict (RAISED, by the committed zu2e.py) and its sha256; stops unless it picked PICK."""
    p = Path(path or U2E.OUT / "verdict.json")
    v = json.loads(p.read_text(encoding="utf-8"))
    if v.get("script_sha256") != U2E.script_sha256() or v.get("verdict") != "RAISED" or v.get("pick") != PICK:
        raise SystemExit(f"{p}: not U2e's RAISED verdict picking {PICK} by the committed zu2e.py")
    order, rest = PICK.split(":")
    bn, tau = rest.split("@")
    if (order, bn, float(tau)) != (PICK_ORDER, f"ada_f{PICK_F:g}_k{PICK_K0}", PICK_TAU) or order not in U2E.NAMES:
        raise SystemExit(f"{PICK}: not this file's order {PICK_ORDER}, budget and cut")
    return v, sha_file(p)


def u2e_arrays(name):
    """U2e's filed stats arrays of the dataset's carve (zu2e.load_u2e: the committed script's, its file, its checks)."""
    x = U2E.load_u2e(name)
    if x is None:
        raise SystemExit(f"{name}: U2e's filed arrays are not under {U2E.OUT / 'stats'}")
    return x[0]


def u2e_graph(name):
    """The graph U2e's filed stats ran on (its record's graph_detail)."""
    rp = U2E.OUT / "stats" / f"{name}__{U2C.carve_of(name)}.json"
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec.get("script_sha256") != U2E.script_sha256():
        raise SystemExit(f"{rp}: not the committed zu2e.py's record")
    return rec["graph_detail"]


def p1c2_entry(name, carve):
    """P1c2's manifest entry of the carve (outputs/p1c2/pools/<D>/pools.json, in the repository)."""
    m = json.loads((PC2.POOLS / name / "pools.json").read_text(encoding="utf-8"))
    e = m["files"].get(f"{name}__{carve}")
    if e is None or e.get("construction") != PC2.CONSTRUCTION:
        raise SystemExit(f"{name}/{carve}: no P1c2 pools entry of {PC2.CONSTRUCTION}")
    return e


def p1c2_file(name, carve):
    """P1c2's pools file of the carve when it is here (the six s1eval carves and musique's s1fit), else None."""
    e = p1c2_entry(name, carve)
    if not (PC2.POOLS / name / e["file"]).exists():
        return None, None
    f2, e2, _m = C4S.load_filed(name, carve, PC2.POOLS / name)
    return f2, e2["sha256"]


# ── one question ─────────────────────────────────────────────────────────────


def base_expansion(full, inc_size, bs_size):
    """P1c2's rule: KIND_T's order outside B, its first |I_q| - |B| nodes. Returns (expansion, |I_q| - |B|)."""
    want = max(0, int(inc_size) - int(bs_size))
    return np.asarray(full, np.int64)[:want], want


def new_expansion(kt_cos, fill_size, bs_size):
    """U2e's pick: k = min(|kt_cos|, round(12 |fill|)) (zu2e.budget_k), then P1h's cut at 0.85 over B and kt_cos's
    first k nodes (zp1h.keep_counts). Returns (expansion = kt_cos's first keep nodes, k, round(12 |fill|))."""
    want = int(U2E.budget_k("ADA", (PICK_F, PICK_K0), [int(fill_size)])[0])
    kt_cos = np.asarray(kt_cos, np.int64)
    k = min(int(kt_cos.size), want)
    keep = int(P1H.keep_counts(np.asarray([int(bs_size)]), np.asarray([k]), PICK_TAU)[0])
    return kt_cos[:keep], k, want


def scaled_place(order, rows, pmax=VIA_PMAX):
    p = KT3.place(np.asarray(order, np.int64), rows)
    return np.where(p >= 0, np.minimum(1.0, np.log1p(np.maximum(p, 0)) / np.log1p(float(pmax))), 1.0)


def via_columns(rows, kt_cos, cos0, n1, cnt, wsum, r0, iout):
    """The seven via columns (section 4) of a question's rows: its places in kt_cos and cos_0, N1 membership with the
    B-parent count and first-stage weight, R0 membership and I^G membership. Rows of B take 'not from this source'."""
    rows = np.asarray(rows, np.int64)
    n1 = np.asarray(n1, np.int64)
    x = np.zeros((rows.size, len(VIA_COLS)), np.float32)
    x[:, 0] = scaled_place(kt_cos, rows)
    x[:, 1] = scaled_place(cos0, rows)
    if n1.size and rows.size:
        k = np.minimum(np.searchsorted(n1, rows), n1.size - 1)
        hit = n1[k] == rows
        den = np.log1p(VIA_CNT)
        x[:, 2] = hit
        x[:, 3] = np.where(hit, np.minimum(1.0, np.log1p(np.asarray(cnt, np.float64)[k]) / den), 0.0)
        x[:, 4] = np.where(hit, np.minimum(1.0, np.log1p(VIA_W * np.asarray(wsum, np.float64)[k]) / den), 0.0)
    x[:, 5] = np.isin(rows, np.asarray(r0, np.int64))
    x[:, 6] = np.isin(rows, np.asarray(iout, np.int64))
    return x


def kt_orders(out, p1d):
    return {"KIND_T": out["KIND_T"], "A1st": p1d["A1st"], "A1all": p1d["A1all"], "Kst": p1d["Kst"],
            "KTst": out["KTst"], "KTall": out["KTall"]}


def u2e_same(q, a, j, gj, g0, g1):
    """One question against U2e's filed arrays on the same carve: |B|, |fill|, |I|, |kt_cos|, |cos_0| and each gold's
    place (0 in B, p >= 1 the p-th node, -1 outside) in kt_cos and cos_0."""
    kc, c0 = U2E.NAMES.index(PICK_ORDER), U2E.NAMES.index("cos")
    inb = np.isin(gj, q["B"])
    return bool(q["B"].size == int(a["bs"][j]) and q["fill"].size == int(a["fill"][j])
                and q["inc"].size == int(a["inc"][j]) and q["kt_cos"].size == int(a["lens"][j, kc])
                and q["cos0"].size == int(a["lens"][j, c0])
                and np.array_equal(U2E.places(gj, inb, q["kt_cos"]), a["pos"][g0:g1, kc])
                and np.array_equal(U2E.places(gj, inb, q["cos0"]), a["pos"][g0:g1, c0]))


def coverage_counts(n_gold, n_held):
    """p1c_pools.coverage from per-question counts, its arithmetic and order kept (a question with no gold counts 0)."""
    a = r = 0.0
    for g, h in zip(np.asarray(n_gold).tolist(), np.asarray(n_held).tolist()):
        a += (h == g) if g else 0.0
        r += (h / g) if g else 0.0
    nq = max(1, len(n_gold))
    return {"ALL": round(a / nq, 4), "recall": round(r / nq, 4)}


def caps(n_gold, n_held):
    """R@5 cap: mean over questions of min(5, golds held) / golds; hit@1 cap: share of questions holding a gold."""
    g = np.asarray(n_gold, np.float64)
    h = np.asarray(n_held, np.float64)
    r5 = np.where(g > 0, np.minimum(5.0, h) / np.maximum(g, 1.0), 0.0)
    return {"r5": round(float(r5.mean()), 4) if g.size else None, "hit1": round(float((h > 0).mean()), 4) if g.size else None}


# ── opening a carve on a build's graph ───────────────────────────────────────


def open_carve(build, name, carve, threads):
    """The carve prepared on the build's graph (U2e's stats_stage's swap on the new build), with step 4d's B, the
    sources' stores and unions, and the dense node reader."""
    import p1e_anchor_kind as K           # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    import numba
    import step4b_pool_reach as S4B
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    if D.SOURCES[4] != "R0" or int(K.NMAX) != KT3.NMAX or F.CONSTRUCTION != PC2.CONSTRUCTION:
        raise SystemExit("P1d's sources, P1d's cut or KIND_T's name are not the filed ones")
    numba.set_num_threads(threads)
    op = S4.Opened()
    ds, construction, families, stores0, union0 = E.open_dataset(op, name)
    n = int(ds.n_nodes)
    stores, union = stores0, union0
    desc = {"graph": "today's (the cached stores)"}
    if build == "new":
        desc = {"graph": "cached (the KB)"}
        g = U2C.graph_csr(name, "chosen", n)
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
                        raise SystemExit(f"{ds_.name}: U2d swaps {name}'s graph only")
                    return swap
                return load0(ds_, family, cache_dir)

            S4.m3b_pools.load_or_build_store = load
        if desc != u2e_graph(name):
            raise SystemExit(f"{name}: the graph is not the one U2e's pick ran on")
    pops = [(c, p) for c, p in E.C4.populations(op, S4, ds, name) if c == carve]
    if len(pops) != 1:
        raise SystemExit(f"{name}: {len(pops)} populations of {carve}")
    (_c, pop), = pops
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
    _U0, _BQ, bs, inc0, _f = E.union_pools(name, carve, pop, prep0, bases)     # base, seeds: step 4d's
    del bases, _U0, _f
    if any(not np.array_equal(np.asarray(a, np.int64), np.asarray(b, np.int64)) for a, b in zip(prep.seeds, prep0.seeds)):
        raise SystemExit(f"{name}/{carve}: the seeds on the build's graph are not today's")
    inc = [np.asarray(p, np.int64) for p in prep.pools]
    linked = E.linked_for(name, ds, pop, index, texts) if index is not None else [EMPTY] * len(bs)
    stores_all, unions = G.open_unions({}, op, name, union)
    adj = [(np.ascontiguousarray(stores_all[f].indptr, np.int64), np.ascontiguousarray(stores_all[f].col))
           for f in S4.FAMILIES]
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs")) if build == "new" else None
    return SimpleNamespace(K=K, D=D, S4=S4, op=op, build=build, name=name, carve=carve, pop=pop, prep=prep, bs=bs,
                           inc=inc, today_inc=[np.asarray(p).size for p in inc0], linked=linked, unions=unions,
                           adj=adj, n_st=int(unions["st"].deg.size), nonly=nonly, nodes=nodes, constant=constant,
                           desc=desc, threads=threads)


def question(cx, j, src, steered):
    """One question's B, orders and expansion on the build (and, on the new build, the via sources)."""
    K, D = cx.K, cx.D
    s, _nn, (L, Ls) = src
    route = bool(D.routes(L, Ls))
    knd = K.kind(Ls, cx.prep.seeds[j], cx.nonly)
    out, p1d = K.p1e_orders(s, steered, route, knd, cx.n_st, cx.constant)
    o = np.asarray(out[F.CONSTRUCTION], np.int64)
    B = np.unique(np.asarray(cx.bs[j], np.int64))
    full = F.outside(o, B)
    inc = cx.inc[j]
    q = {"B": B, "route": route, "entity": knd == "entity", "inc": inc, "orders": kt_orders(out, p1d)}
    if cx.build == "base":
        exp, want = base_expansion(full, inc.size, B.size)
        q.update(exp=exp, k=int(exp.size), want=want, today=int(exp.size), kt_len=int(full.size))
        return q
    fill = full[:max(0, inc.size - B.size)]
    w_B = D5.weights_of(K.steer(cx.prep, j, cx.constant), B)
    qv = D5B.unit(np.asarray(cx.prep.qemb[j], np.float64))
    n1, cnt, wsum, _up, _inv = D5B.n1_table(cx.adj, B, w_B)
    cos1 = D5B.unit(cx.nodes.read(n1)) @ qv if n1.size else np.empty(0, np.float32)
    cr = D5B.ranks_desc(cos1, n1) if n1.size else EMPTY
    r0 = F.outside(s[4][0], B)
    iout = F.outside(inc, B)
    ords, _sizes, U, mem = U2E.question_orders(full, n1, cr, cos1, r0, iout, EMPTY, EMPTY, cx.nodes.read, qv,
                                               cx.constant, D.rrf)
    if not U2E.orders_exact(ords, B, full, n1, U, mem):
        raise SystemExit(f"{cx.name}/{cx.carve} question {j}: an order is not exactly its sources' nodes")
    kt_cos, cos0 = ords[PICK_ORDER], ords["cos"]
    exp, k, want = new_expansion(kt_cos, fill.size, B.size)
    q.update(exp=exp, k=k, want=want, today=int(fill.size), kt_len=int(kt_cos.size), fill=fill, kt_cos=kt_cos,
             cos0=cos0, n1=n1, cnt=cnt, wsum=wsum, r0=r0, iout=iout)
    return q


def head_of(cx, build, vsha, f2sha):
    import p1e_anchor_kind as K
    return {"declared_in": DECLARED, "stage": "U2d", "build": build, "dataset": cx.name, "carve": cx.carve,
            "construction": construction_of(build), "graph_detail": cx.desc, "carve_ids_sha256": cx.pop.digest,
            "questions_all": len(cx.bs), "script_sha256": script_sha256(), "zu2e_sha256": U2E.script_sha256(),
            "zu2c_sha256": U2C.script_sha256(), "p1gb_sha256": PGB.script_sha256(), "p1c2_sha256": PC2.script_sha256(),
            "p1c3_sha256": KT3.script_sha256(), "p1e_sha256": K.script_sha256(), "zp1h_sha256": P1H.script_sha256(),
            "u2e_verdict_sha256": vsha, "p1c2_file_sha256": f2sha,
            "freeze_RECORD_SHA256": cx.op.freeze["RECORD_SHA256"], "threads": cx.threads}


def stem_of(name, carve, shard):
    i, k = shard or (0, 1)
    return f"{name}__{carve}__{i}of{k}"


def save_npy(path, a):
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, np.ascontiguousarray(a))
    os.replace(tmp, path)
    return sha_file(path)


def save_npz(path, arrays):
    tmp = path.with_name(path.name[:-len(".npz")] + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return sha_file(path)


def part_arrays(ids, Bs, incs, exps, per, gold_ptr, held):
    """A part's pools arrays: P1g-b's (p1gb_pools.pool_arrays) and the added per-question columns; per gold, held."""
    a = PGB.pool_arrays(ids, Bs, incs, exps, per["today_count"], per["fill_want"], per["fill_k"], per["x_count"])
    for k in PER_Q:
        if k not in a:
            a[k] = np.asarray(per[k], bool if k in ("entity", "route", "changed") else np.int32)
    a["gold_ptr"] = np.asarray(gold_ptr, np.int64)
    a["held"] = np.asarray(held, bool)
    return a


# ── pools (Modal, in shards) ─────────────────────────────────────────────────


def pools_stage(build, name, carve, threads=2, limit=0, shard=None):
    R = roots(build)
    if carve not in PC2.carves_for(name):
        raise SystemExit(f"{name}: {carve} is not one of P1c2's carves {PC2.carves_for(name)}")
    vsha = u2e_pick()[1] if build == "new" else None
    t0 = time.time()
    e2 = p1c2_entry(name, carve)
    f2, f2sha = p1c2_file(name, carve) if build == "base" else (None, None)
    cx = open_carve(build, name, carve, threads)
    pop, K, D, S4 = cx.pop, cx.K, cx.D, cx.S4
    nq = len(cx.bs)
    ids = [str(q) for q in pop.ids]
    if e2["carve_ids_sha256"] != pop.digest or int(e2["questions"]) != nq:
        raise SystemExit(f"{name}/{carve}: not P1c2's questions")
    if f2 is not None and [str(q) for q in f2["ids"].tolist()] != ids:
        raise SystemExit(f"{name}/{carve}: P1c2's file holds other questions than the carve")
    golds = [np.unique(np.asarray(g, np.int64)) for g in pop.golds]
    gptr = np.concatenate([[0], np.cumsum([g.size for g in golds])]).astype(np.int64)
    ref = None
    if build == "new" and carve == U2C.carve_of(name):
        ref = u2e_arrays(name)
        if not np.array_equal(ref["ptr"], gptr):
            raise SystemExit(f"{name}/{carve}: U2e's golds are not the carve's")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    if limit:
        q1 = min(q1, q0 + limit)
    log(f"u2d pools {build} {name}/{carve}: prepared ({time.time() - t0:.0f}s); questions {q0}-{q1 - 1} of {nq}")
    Bs, incs, exps, kts, vias, held, gp = [], [], [], [], [], [], [0]
    per = {k: [] for k in PER_Q if k not in ("inc_size", "bs_size", "count")}
    checks = {"bs_step4d": True, "seeds_same_both_graphs": True}
    if build == "new":
        checks.update(graph_u2e=True, orders_exact=True)
    if ref is not None:
        checks["equal_u2e_arrays"] = True
    if f2 is not None:
        checks["equal_p1c2_file"] = True
    t = time.time()
    for lo in range(q0, q1, S4.BATCH):
        js = list(range(lo, min(q1, lo + S4.BATCH)))
        srcs = D.sources_batch(cx.prep, js, cx.linked, cx.unions, cx.constant, threads)
        steered = K.steered_batch(cx.prep, js, cx.linked, cx.unions, srcs, cx.constant, threads)
        for i, j in enumerate(js):
            q = question(cx, j, srcs[i], steered[i])
            B, exp, inc, gj = q["B"], q["exp"], q["inc"], golds[j]
            if ref is not None and not u2e_same(q, ref, j, gj, int(gptr[j]), int(gptr[j + 1])):
                raise SystemExit(f"{name}/{carve} question {j} ({ids[j]}): not U2e's filed arrays")
            if f2 is not None:
                sha = f2["inc_sha"][j]
                sha = sha.decode() if isinstance(sha, bytes) else str(sha)
                if not (np.array_equal(B, f2["bs_nodes"][f2["bs_ptr"][j]:f2["bs_ptr"][j + 1]].astype(np.int64))
                        and inc.size == int(f2["inc_size"][j]) and C4S.pool_sha(inc) == sha
                        and np.array_equal(exp, f2["nodes"][f2["ptr"][j]:f2["ptr"][j + 1]].astype(np.int64))
                        and q["entity"] == bool(f2["entity"][j]) and q["route"] == bool(f2["route"][j])):
                    raise SystemExit(f"{name}/{carve} question {j} ({ids[j]}): not P1c2's filed pool")
            pool = np.union1d(B, exp)
            if pool.size != B.size + exp.size:
                raise SystemExit(f"{name}/{carve} question {j}: the expansion repeats a node or holds one of B")
            kts.append(KT3.columns(q["orders"], pool, exp, q["entity"]).astype(KT_DT))
            if build == "new":
                vias.append(via_columns(pool, q["kt_cos"], q["cos0"], q["n1"], q["cnt"], q["wsum"], q["r0"],
                                        q["iout"]).astype(KT_DT))
            h = np.isin(gj, pool)
            held.append(h)
            gp.append(gp[-1] + gj.size)
            Bs.append(B)
            incs.append(inc)
            exps.append(exp)
            for k, x in (("today_count", q["today"]), ("fill_want", q["want"]), ("fill_k", int(exp.size)),
                         ("x_count", 0), ("today_inc_size", cx.today_inc[j]), ("budget", q["k"]),
                         ("kt_len", q["kt_len"]), ("entity", q["entity"]), ("route", q["route"]),
                         ("changed", not np.array_equal(pool, np.sort(inc))), ("n_gold", int(gj.size)),
                         ("n_held", int(h.sum())), ("n_held_I", int(np.isin(gj, inc).sum()))):
                per[k].append(x)
        if (lo // S4.BATCH) % 10 == 9:
            log(f"u2d pools {build} {name}/{carve}: {js[-1] + 1 - q0}/{q1 - q0} ({time.time() - t:.0f}s)")
    if limit:
        log(f"u2d pools {build} {name}/{carve} development run: questions {q0}-{q1 - 1}, checks {sorted(checks)} "
            f"held ({time.time() - t:.0f}s); nothing written")
        return 0
    arrays = part_arrays(ids[q0:q1], Bs, incs, exps, per, np.asarray(gp, np.int64) + gptr[q0], np.concatenate(held)
                         if held else np.zeros(0, bool))
    kt = np.concatenate(kts) if kts else np.zeros((0, KT3.NKT), KT_DT)
    via = (np.concatenate(vias) if vias else np.zeros((0, len(VIA_COLS)), KT_DT)) if build == "new" else None
    if kt.shape[0] != int(PGB.expected_rows(arrays).sum()) or (via is not None and via.shape[0] != kt.shape[0]):
        raise SystemExit(f"{name}/{carve}: the columns do not cover the pools' rows")
    pd = R["parts"]
    pd.mkdir(parents=True, exist_ok=True)
    stem = stem_of(name, carve, shard)
    files = {"npz": save_npz(pd / f"{stem}.npz", arrays), "kt": save_npy(pd / f"{stem}.kt.npy", kt)}
    if via is not None:
        files["via"] = save_npy(pd / f"{stem}.via.npy", via)
    i, k = shard or (0, 1)
    write_json(pd / f"{stem}.json", {**head_of(cx, build, vsha, f2sha), "part": i, "parts": k, "q0": q0, "q1": q1,
                                     "rows": int(kt.shape[0]), "files": files, "checks": checks,
                                     "seconds": round(time.time() - t0, 1), "utc": utc()})
    log(f"u2d pools {build} {name}/{carve} part {i}/{k} (questions {q0}-{q1 - 1}, {kt.shape[0]} rows, "
        f"{time.time() - t0:.0f}s): written")
    return 0


# ── merge ────────────────────────────────────────────────────────────────────


def find_parts(d, name, carve):
    """The carve's part records in order; stops unless they are one whole set I = 0..K-1 of one K."""
    recs = {}
    for p in Path(d).glob(f"{name}__{carve}__*of*.json"):
        i, k = p.stem.split("__")[-1].split("of")
        recs[(int(i), int(k))] = p
    ks = {k for _i, k in recs}
    if len(ks) != 1 or sorted(i for i, _k in recs) != list(range(next(iter(ks)))):
        raise SystemExit(f"{d}: {name}/{carve}'s parts {sorted(recs)} are not one whole set")
    k = ks.pop()
    return [json.loads(recs[(i, k)].read_text(encoding="utf-8")) for i in range(k)]


def load_parts(d, name, carve, sums=True):
    """The carve's part records and arrays, each part's files checked against its record and the parts against each
    other (one head, contiguous question ranges)."""
    recs = find_parts(d, name, carve)
    head = {k: recs[0].get(k) for k in HEAD_KEYS}
    nq = recs[0]["questions_all"]
    arrs = []
    for i, r in enumerate(recs):
        if {k: r.get(k) for k in HEAD_KEYS} != head or r["checks"] != recs[0]["checks"] or not all(r["checks"].values()):
            raise SystemExit(f"{name}/{carve} part {i}: another head, check set, or a failed check")
        if (r["part"], r["parts"]) != (i, len(recs)) or (r["q0"], r["q1"]) != U2C.shard_range(nq, i, len(recs)):
            raise SystemExit(f"{name}/{carve} part {i}: not the {i}-th of {len(recs)} question ranges")
        if head["script_sha256"] != script_sha256():
            raise SystemExit(f"{name}/{carve}: parts written by another zu2d_pools.py")
        stem = stem_of(name, carve, (i, len(recs)))
        for key, suf in (("npz", ".npz"), ("kt", ".kt.npy"), ("via", ".via.npy")):
            if key in r["files"] and sums and sha_file(Path(d) / f"{stem}{suf}") != r["files"][key]:
                raise SystemExit(f"{stem}{suf}: not its record's file")
        with np.load(Path(d) / f"{stem}.npz") as z:
            arrs.append({k: z[k] for k in z.files})
    return recs, arrs, head


def merge_part_arrays(arrs):
    """One carve's pools arrays from its parts' (each pointer rebuilt from the sizes)."""
    out = {}
    for k in ("ids", "inc_sha") + PER_Q:
        out[k] = np.concatenate([a[k] for a in arrs])
    out["bs_nodes"] = np.concatenate([a["bs_nodes"] for a in arrs])
    out["nodes"] = np.concatenate([a["nodes"] for a in arrs])
    out["held"] = np.concatenate([a["held"] for a in arrs])
    for a in arrs:
        if not (np.array_equal(np.diff(a["bs_ptr"]), a["bs_size"]) and np.array_equal(np.diff(a["ptr"]), a["count"])
                and int(a["gold_ptr"][-1] - a["gold_ptr"][0]) == a["held"].size
                and np.array_equal(np.diff(a["gold_ptr"]), a["n_gold"])):
            raise SystemExit("a part's pointers are not its sizes")
    gp = [arrs[0]["gold_ptr"][:1]]
    for i, a in enumerate(arrs):
        if int(a["gold_ptr"][0]) != int(gp[-1][-1]):
            raise SystemExit(f"part {i}: its golds do not continue the previous part's")
        gp.append(a["gold_ptr"][1:])
    out["gold_ptr"] = np.concatenate(gp).astype(np.int64)
    out["bs_ptr"] = ptr_of(out["bs_size"])
    out["ptr"] = ptr_of(out["count"])
    return out


def ptr_of(sizes):
    """step 4c's flat_of pointer (int64, from 0) of the sizes."""
    ptr = np.zeros(len(sizes) + 1, np.int64)
    np.cumsum(np.asarray(sizes, np.int64), out=ptr[1:])
    return ptr


def carve_summary(a):
    """Counts, rows, coverage and caps of one carve's merged arrays."""
    rows = PGB.expected_rows(a)
    b_q = (a["inc_size"].astype(np.int64) - a["bs_size"].astype(np.int64)).astype(np.int32)
    nq = int(a["ids"].size)
    ent, rte = a["entity"].astype(bool), a["route"].astype(bool)
    return {"questions": nq, "changed": int(a["changed"].sum()), "short": int((a["fill_k"] < a["fill_want"]).sum()),
            "B_q": {"mean": round(float(b_q.mean()), 1) if nq else None, "max": int(b_q.max()) if nq else None},
            "pool_mean": round(float(a["inc_size"].mean()), 1) if nq else None,
            "rows": {"mean": round(float(rows.mean()), 1), "p90": round(float(np.quantile(rows, 0.9)), 1),
                     "max": int(rows.max())},
            "budget_mean": round(float(a["budget"].mean()), 1), "fill_want_mean": round(float(a["fill_want"].mean()), 1),
            "today_inc_mean": round(float(a["today_inc_size"].mean()), 1),
            "shares": {"entity": round(float(ent.mean()), 4) if nq else None,
                       "route": round(float(rte.mean()), 4) if nq else None,
                       "entity_route": round(float((ent & rte).mean()), 4) if nq else None},
            "I": coverage_counts(a["n_gold"], a["n_held_I"]), "pool": coverage_counts(a["n_gold"], a["n_held"]),
            "caps": {"I": caps(a["n_gold"], a["n_held_I"]), "pool": caps(a["n_gold"], a["n_held"])}}


def p1c2_counts_equal(s, e2):
    """The base build's carve summary against P1c2's manifest entry, field by field."""
    got = {"questions": s["questions"], "changed": s["changed"], "short": s["short"], "B_q": s["B_q"],
           "pool_mean": s["pool_mean"], "shares": s["shares"],
           "ALL": {"I": s["I"]["ALL"], "P1c2": s["pool"]["ALL"]}, "recall": {"I": s["I"]["recall"], "P1c2": s["pool"]["recall"]}}
    want = {k: e2.get(k) for k in got}
    return got == want, got, want


def merge_stage(build, name):
    R = roots(build)
    d = R["pools"] / name
    mpath = d / "pools.json"
    if mpath.exists():
        raise SystemExit(f"{mpath} exists; U2d never overwrites a pools manifest")
    t0 = time.time()
    manifest = {"files": {}, "declared_in": DECLARED, "stage": "U2d", "build": build, "dataset": name,
                "construction": construction_of(build), "script_sha256": script_sha256()}
    pending = []
    for carve in PC2.carves_for(name):
        recs, arrs, head = load_parts(R["parts"], name, carve)
        a = merge_part_arrays(arrs)
        del arrs
        nq = int(head["questions_all"])
        if a["ids"].size != nq or head["construction"] != construction_of(build) or head["build"] != build:
            raise SystemExit(f"{name}/{carve}: the parts hold {a['ids'].size} of {nq} questions, or another build")
        s = carve_summary(a)
        checks = dict(recs[0]["checks"])
        e2 = p1c2_entry(name, carve)
        if build == "base":
            ok, got, want = p1c2_counts_equal(s, e2)
            checks["counts_equal_p1c2_manifest"] = ok
            if not ok:
                raise SystemExit(f"{name}/{carve}: {got} is not P1c2's {want}")
        if build == "new" and carve == U2C.carve_of(name):
            v, _vsha = u2e_pick()
            got = U2E.summ(PGB.expected_rows(a).astype(np.int64), a["held"], a["gold_ptr"], PICK)
            checks["equals_u2e_verdict"] = got == v["pick_webqsp_s1eval"]
            if not checks["equals_u2e_verdict"]:
                raise SystemExit(f"{name}/{carve}: {got} is not U2e's verdict's {v['pick_webqsp_s1eval']}")
        d.mkdir(parents=True, exist_ok=True)
        fname = f"{name}__{carve}.npz"
        fsha = save_npz(d / fname, a)
        stems = [stem_of(name, carve, (i, len(recs))) for i in range(len(recs))]
        pending.append((f"{name}__{carve}", {
            "file": fname, "sha256": fsha, "dataset": name, "carve": carve, "construction": construction_of(build),
            "rule": RULE[build], "graph_detail": head["graph_detail"], "carve_ids_sha256": head["carve_ids_sha256"],
            **s, "parts": [{"stem": st, "q0": r["q0"], "q1": r["q1"], "rows": r["rows"], **r["files"]}
                           for st, r in zip(stems, recs)],
            "head": head, "checks": checks, "seconds_parts": round(sum(r["seconds"] for r in recs), 1), "utc": utc()}))
        log(f"u2d merge {build} {name}/{carve}: {nq} questions, rows {s['rows']['mean']}, ALL I {s['I']['ALL']} pool "
            f"{s['pool']['ALL']}, caps R@5 {s['caps']['pool']['r5']} hit@1 {s['caps']['pool']['hit1']}")
        del a
        gc.collect()
    manifest["files"] = dict(pending)
    manifest["seconds"] = round(time.time() - t0, 1)
    manifest["utc"] = utc()
    write_json(mpath, manifest)
    return 0


def load_filed(build, name, carve):
    filed, ent, msha = C4S.load_filed(name, carve, roots(build)["pools"] / name)
    if ent.get("construction") != construction_of(build):
        raise SystemExit(f"{name}/{carve}: the pools file is {ent.get('construction')}'s, not {construction_of(build)}")
    return filed, ent, msha


# ── the s1sel check of the new build (Modal, in shards) ──────────────────────


def check_stage(name, threads=2, limit=0, shard=None):
    if name not in TRAIN5:
        raise SystemExit(f"check runs on the five training datasets' s1sel; webqsp's s1eval is checked by pools")
    carve = U2C.carve_of(name)
    R = roots("new")
    _v, vsha = u2e_pick()
    t0 = time.time()
    a = u2e_arrays(name)
    cx = open_carve("new", name, carve, threads)
    K, D, S4 = cx.K, cx.D, cx.S4
    nq = len(cx.bs)
    golds = [np.unique(np.asarray(g, np.int64)) for g in cx.pop.golds]
    gptr = np.concatenate([[0], np.cumsum([g.size for g in golds])]).astype(np.int64)
    if a["bs"].size != nq or not np.array_equal(a["ptr"], gptr):
        raise SystemExit(f"{name}/{carve}: U2e's arrays are not the carve's questions and golds")
    q0, q1 = U2C.shard_range(nq, *shard) if shard else (0, nq)
    if limit:
        q1 = min(q1, q0 + limit)
    rows, held, gp = [], [], [0]
    for lo in range(q0, q1, S4.BATCH):
        js = list(range(lo, min(q1, lo + S4.BATCH)))
        srcs = D.sources_batch(cx.prep, js, cx.linked, cx.unions, cx.constant, threads)
        steered = K.steered_batch(cx.prep, js, cx.linked, cx.unions, srcs, cx.constant, threads)
        for i, j in enumerate(js):
            q = question(cx, j, srcs[i], steered[i])
            if not u2e_same(q, a, j, golds[j], int(gptr[j]), int(gptr[j + 1])):
                raise SystemExit(f"{name}/{carve} question {j}: not U2e's filed arrays")
            rows.append(q["B"].size + q["exp"].size)
            held.append(np.isin(golds[j], np.union1d(q["B"], q["exp"])))
            gp.append(gp[-1] + golds[j].size)
    if limit:
        log(f"u2d check {name}/{carve} development run: questions {q0}-{q1 - 1}, every question equal to U2e's "
            f"arrays ({time.time() - t0:.0f}s); nothing written")
        return 0
    pd = R["check"] / "parts"
    pd.mkdir(parents=True, exist_ok=True)
    stem = stem_of(name, carve, shard)
    i, k = shard or (0, 1)
    fsha = save_npz(pd / f"{stem}.npz", {"rows": np.asarray(rows, np.int64), "gold_ptr": np.asarray(gp, np.int64) +
                                         gptr[q0], "held": np.concatenate(held) if held else np.zeros(0, bool)})
    write_json(pd / f"{stem}.json", {**head_of(cx, "new", vsha, None), "part": i, "parts": k, "q0": q0, "q1": q1,
                                     "rows": int(sum(rows)), "files": {"npz": fsha},
                                     "checks": {"equal_u2e_arrays": True, "orders_exact": True, "graph_u2e": True,
                                                "seeds_same_both_graphs": True},
                                     "seconds": round(time.time() - t0, 1), "utc": utc()})
    log(f"u2d check {name}/{carve} part {i}/{k}: written ({time.time() - t0:.0f}s)")
    return 0


def check_merge(name):
    carve = U2C.carve_of(name)
    R = roots("new")
    v, vsha = u2e_pick()
    recs, arrs, head = load_parts(R["check"] / "parts", name, carve)
    rows = np.concatenate([x["rows"] for x in arrs])
    held = np.concatenate([x["held"] for x in arrs])
    gp = [arrs[0]["gold_ptr"][:1]]
    for x in arrs:
        if int(x["gold_ptr"][0]) != int(gp[-1][-1]):
            raise SystemExit(f"{name}: a check part's golds do not continue the previous part's")
        gp.append(x["gold_ptr"][1:])
    gptr = np.concatenate(gp).astype(np.int64)
    a = u2e_arrays(name)
    got = U2E.summ(rows, held, gptr, PICK)
    ok = bool(rows.size == a["bs"].size and np.array_equal(gptr, a["ptr"]) and got == v["pick_s1sel"][name])
    rec = {"declared_in": DECLARED, "stage": "U2d", "check": "the new build's rule on s1sel against U2e", "dataset": name,
           "carve": carve, "pick": PICK, "verdict": "PASS" if ok else "FAIL", "got": got, "filed": v["pick_s1sel"][name],
           "u2e_verdict_sha256": vsha, "parts": [r["files"]["npz"] for r in recs], "head": head,
           "script_sha256": script_sha256(), "utc": utc()}
    write_json(R["check"] / f"{name}.json", rec)
    log(f"u2d check {name}: {rec['verdict']} ({got} against U2e's {v['pick_s1sel'][name]})")
    return 0 if ok else 1


# ── copy-bases and kt ────────────────────────────────────────────────────────


def copy_bases(build):
    saved = C4S.CACHE
    C4S.CACHE = roots(build)["cache"]
    try:
        return C4S.copy_bases()
    finally:
        C4S.CACHE = saved


def to_cache_order(filed, flat, c_ids, c_n, c_node):
    """flat (rows x m: each question's pool sorted by node, in the file's question order) read at each cache row.
    Stops unless the cache holds the file's questions and each question's cache rows are its pool, each node once."""
    at = {str(q): j for j, q in enumerate(filed["ids"].tolist())}
    if len(c_ids) != len(at) or any(q not in at for q in c_ids):
        raise SystemExit("the cache's questions are not the pools file's")
    sizes = PGB.expected_rows(filed)
    pp = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    if flat.shape[0] != int(pp[-1]):
        raise SystemExit(f"the columns hold {flat.shape[0]} rows, the pools {int(pp[-1])}")
    off = np.concatenate([[0], np.cumsum(c_n)]).astype(np.int64)
    if int(off[-1]) != np.asarray(c_node).size:
        raise SystemExit("the cache's rows per question do not cover its nodes")
    out = np.empty((int(off[-1]), flat.shape[1]), flat.dtype)
    bp, ep = filed["bs_ptr"], filed["ptr"]
    for c, qid in enumerate(c_ids):
        j = at[qid]
        pool = np.union1d(filed["bs_nodes"][bp[j]:bp[j + 1]].astype(np.int64),
                          filed["nodes"][ep[j]:ep[j + 1]].astype(np.int64))
        r = np.asarray(c_node[off[c]:off[c + 1]], np.int64)
        if r.size != pool.size or np.unique(r).size != r.size:
            raise SystemExit(f"{qid}: the cache holds {r.size} rows ({np.unique(r).size} distinct), its pool {pool.size}")
        if r.size == 0:
            continue
        idx = np.minimum(np.searchsorted(pool, r), pool.size - 1)
        if not np.array_equal(pool[idx], r):
            raise SystemExit(f"{qid}: the cache's rows are not its pool")
        out[off[c]:off[c + 1]] = flat[pp[j] + idx]
    return out


def part_columns(build, name, carve, ent, key):
    """The carve's columns (kt or via) over its pools in the file's question order, from its parts (sha-checked)."""
    d = roots(build)["parts"]
    out = []
    for p in ent["parts"]:
        f = d / f"{p['stem']}.{key}.npy"
        if sha_file(f) != p[key]:
            raise SystemExit(f"{f}: not the manifest's part")
        out.append(np.load(f))
    return np.concatenate(out) if out else np.zeros((0, KT3.NKT if key == "kt" else len(VIA_COLS)), KT_DT)


def p1c3_same(name, carve, shas):
    """Beside: whether the base build's kt and n arrays are P1c3's filed ones."""
    rp = KT3.KT / name / carve / "record.json"
    if not rp.exists():
        return "P1c3's record is not here"
    r = json.loads(rp.read_text(encoding="utf-8"))
    return "REPRODUCED" if all(r["arrays"][k]["sha256"] == shas[k] for k in ("kt", "n")) else "DIFFERS"


def kt_stage(build, name, carve):
    import p1e_anchor_kind as K
    R = roots(build)
    if carve not in PC2.carves_for(name):
        raise SystemExit(f"{name}: {carve} is not one of P1c2's carves")
    t0 = time.time()
    filed, ent, msha = load_filed(build, name, carve)
    c_ids, c_n, c_node, part_shas = KT3.cache_rows(R["cache"], name, carve)
    d = R["kt"] / name / carve
    d.mkdir(parents=True, exist_ok=True)
    kt = to_cache_order(filed, part_columns(build, name, carve, ent, "kt"), c_ids, c_n, c_node)
    shas = {"kt": save_npy(d / "kt.npy", kt), "n": save_npy(d / "n.npy", np.asarray(c_n, np.int32))}
    write_json(d / "ids.json", c_ids)
    common = {"dataset": name, "carve": carve, "questions": len(c_ids), "rows": int(kt.shape[0]),
              "ids_sha256": sha_file(d / "ids.json"), "cache_root": R["cache"].relative_to(ROOT).as_posix(),
              "cache_parts": part_shas, "pools_manifest_sha256": msha, "pools_file_sha256": ent["sha256"],
              "construction": ent["construction"], "build": build, "script_sha256": script_sha256(),
              "p1c3_script_sha256": KT3.script_sha256(), "p1e_script_sha256": K.script_sha256(),
              "checks": {"rows_are_pool": True, "parts_sha256": True}}
    if build == "new":
        via = to_cache_order(filed, part_columns(build, name, carve, ent, "via"), c_ids, c_n, c_node)
        vsha = save_npy(d / "via.npy", via)
        write_json(d / "via.json", {"declared_in": DECLARED, "columns": list(VIA_COLS), "dtype": "float16",
                                    "arrays": {"via": {"sha256": vsha, "shape": list(via.shape)},
                                               "n": {"sha256": shas["n"], "shape": [len(c_ids)]}},
                                    "scales": {"pmax": VIA_PMAX, "cnt": VIA_CNT, "w": VIA_W}, **common, "utc": utc()})
        del via
    rec = {"declared_in": DECLARED, "columns_declared_in": KT3.DECLARED, "columns": list(KT3.COLS), "nmax": KT3.NMAX,
           "dtype": "float16", "arrays": {"kt": {"sha256": shas["kt"], "shape": list(kt.shape)},
                                          "n": {"sha256": shas["n"], "shape": [len(c_ids)]}},
           **common, "beside_p1c3": p1c3_same(name, carve, shas) if build == "base" else None,
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(d / "record.json", rec)
    log(f"u2d kt {build} {name}/{carve}: {len(c_ids)} questions, {kt.shape[0]} rows in the cache's order"
        f"{'; P1c3 ' + rec['beside_p1c3'] if build == 'base' else ''} ({rec['seconds']:.0f}s)")
    return 0


# ── the gate ─────────────────────────────────────────────────────────────────


def npy_sha(a):
    """sha256 of an array saved as np.save writes it (step 1's records name their arrays' file sha256)."""
    b = io.BytesIO()
    np.save(b, np.ascontiguousarray(a))
    return hashlib.sha256(b.getvalue()).hexdigest()


def step1_check(filed, recs):
    """Step 1's rows are the filed today_inc_size: for each part, the sha256 of its questions' sizes as int32 is the
    part record's n array's. Returns (step 1's ids in order, the parts that fail)."""
    at = {str(q): j for j, q in enumerate(filed["ids"].tolist())}
    ids1, bad = [], []
    for r in recs:
        qs = [str(q) for q in r["ids"]]
        if any(q not in at for q in qs):
            bad.append(f"part {r.get('part')}: a question not in the pools file")
            continue
        ids1 += qs
        n = filed["today_inc_size"][np.asarray([at[q] for q in qs], np.int64)].astype(np.int32)
        a = r["arrays"]["n"]
        if a.get("dtype") != "int32" or list(a.get("shape", [])) != [n.size] or npy_sha(n) != a.get("sha256"):
            bad.append(f"part {r.get('part')}: step 1's n is not the filed today_inc_size")
    return ids1, bad


def kt_check(build, ds, cv, key="record"):
    R = roots(build)
    rp = R["kt"] / ds / cv / f"{key}.json"
    if not rp.exists():
        return f"{key}.json missing"
    r = json.loads(rp.read_text(encoding="utf-8"))
    parts = {p.name: sha_file(p / "record.json") for p in C4S.parts_of(R["cache"], ds, cv)[0]}
    if r.get("declared_in") != DECLARED or r.get("cache_parts") != parts or \
            r.get("cache_root") != R["cache"].relative_to(ROOT).as_posix() or r.get("build") != build:
        return f"{key}.json is not on the build's cache parts and root"
    return None


def gate(build):
    R = roots(build)
    t0 = time.time()
    rec = {"declared_in": DECLARED, "build": build, "manifests": {}, "rows": {}, "looks": {}, "kt": {}, "checks": {},
           "script_sha256": script_sha256()}
    fails, constructions = [], set()
    for ds, cv in SCREEN:
        mp = R["pools"] / ds / "pools.json"
        rec["manifests"].setdefault(ds, sha_file(mp))
        filed, ent, msha = load_filed(build, ds, cv)
        if msha != rec["manifests"][ds]:
            raise SystemExit(f"{mp}: changed while the gate ran")
        constructions.add(ent["construction"])
        bad = C1.look_pools_records(ds, cv, ent, R["look"])
        rec["looks"][f"{ds}={cv}"] = bad or "PASS"
        fails += [f"{ds}/{cv}: {b}" for b in bad]
        _p1, recs1 = C4S.parts_of(C4S.STEP1_CACHE, ds, cv)
        ids1, bad1 = step1_check(filed, recs1)
        ids4, n4, _p4, _r4 = C4S.carve_rows(R["cache"], ds, cv)
        at = {str(q): k for k, q in enumerate(filed["ids"].tolist())}
        same_q = ids1 == ids4 and len(ids4) == len(at) and all(q in at for q in ids4)
        ok4 = same_q and bool(np.array_equal(n4, PGB.expected_rows(filed)[np.asarray([at[q] for q in ids4], np.int64)]))
        row = {"questions": len(ids4), "short": ent["short"], "changed": ent["changed"],
               "step1_rows_are_today_inc_size": not bad1, "questions_are_step1s_and_the_files": same_q,
               "rows_are_pool_sizes": ok4, "verdict": "PASS" if not bad1 and ok4 else "FAIL"}
        rec["rows"][f"{ds}={cv}"] = row
        if row["verdict"] != "PASS":
            fails.append(f"{ds}/{cv}: rows {row} {bad1}")
        for key in ("record",) + (("via",) if build == "new" else ()):
            k = kt_check(build, ds, cv, key)
            rec["kt"][f"{ds}={cv}:{key}"] = k or "PASS"
            if k:
                fails.append(f"{ds}/{cv}: {k}")
    if build == "new":
        for ds in TRAIN5:
            p = R["check"] / f"{ds}.json"
            v = json.loads(p.read_text(encoding="utf-8")).get("verdict") if p.exists() else "missing"
            rec["checks"][ds] = v
            if v != "PASS":
                fails.append(f"{ds}: the s1sel check is {v}")
    if len(constructions) != 1:
        fails.append(f"the carves' pools come from {sorted(constructions)}, not one rule")
    rec.update({"construction": sorted(constructions), "failures": fails, "verdict": "PASS" if not fails else "FAIL",
                "seconds": round(time.time() - t0, 1), "utc": utc()})
    write_json(R["gate"], rec)
    log(f"u2d gate {build} {rec['verdict']}: {fails or 'every row count as filed, every look on the filed pools, the KT input on the caches'}")
    return 0 if not fails else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def _direct_keep(b, n, tau):
    """P1h's cut by a direct loop: mass 1/(60 + p) over the build order, shortest prefix reaching tau of the total."""
    if tau == 1.0 or n == 0:
        return n - b
    cs = np.cumsum(1.0 / (60.0 + np.arange(1, n + 1, dtype=np.float64)))
    m = int(np.searchsorted(cs, tau * cs[-1], side="left")) + 1
    return min(max(m, b), n) - b


def _toy_filed(rng, nq, build):
    """A toy carve's pools arrays (as pools_stage files them) and its frozen pools."""
    ids, Bs, incs, exps, per, held, gp = [], [], [], [], {k: [] for k in PER_Q}, [], [0]
    for j in range(nq):
        B = np.unique(rng.integers(0, 60, rng.integers(1, 6)))
        inc = np.unique(np.concatenate([B, rng.integers(60, 120, rng.integers(0, 9))]))
        cand = rng.permutation(np.setdiff1d(np.arange(60, 200), B))[:rng.integers(0, 12)]
        exp = cand.astype(np.int64)
        ids.append(f"q{j}")
        Bs.append(B)
        incs.append(inc)
        exps.append(exp)
        g = np.unique(rng.integers(0, 200, rng.integers(0, 4)))
        h = np.isin(g, np.union1d(B, exp))
        held.append(h)
        gp.append(gp[-1] + g.size)
        for k, x in (("today_count", exp.size), ("fill_want", exp.size + (build == "new")), ("fill_k", exp.size),
                     ("x_count", 0), ("today_inc_size", inc.size), ("budget", exp.size), ("kt_len", exp.size + 3),
                     ("entity", bool(j % 2)), ("route", bool(j % 3 == 0)), ("changed", True), ("n_gold", g.size),
                     ("n_held", int(h.sum())), ("n_held_I", int(np.isin(g, inc).sum()))):
            per[k].append(x)
    return part_arrays(ids, Bs, incs, exps, per, gp, np.concatenate(held)), incs


def selftest():
    import tempfile
    t0 = time.time()
    rng = np.random.default_rng(0)
    assert len(SCREEN) == 11 and all(roots(b)["pools"] != PC2.POOLS for b in BUILDS)
    assert construction_of("base") == "U2D:base:KIND_T" and construction_of("new") == f"U2D:new:{PICK}"
    # 1. the expansions: P1c2's fill, and U2e's pick (budget, then P1h's cut) against a direct loop
    full = np.asarray([9, 4, 7, 1, 8])
    e, want = base_expansion(full, 7, 4)
    assert e.tolist() == [9, 4, 7] and want == 3
    e, want = base_expansion(full, 3, 4)
    assert e.size == 0 and want == 0
    for _ in range(300):
        nb, nk, fs = int(rng.integers(0, 80)), int(rng.integers(0, 3000)), int(rng.integers(0, 400))
        kt_cos = rng.permutation(10000)[:nk]
        e, k, want = new_expansion(kt_cos, fs, nb)
        assert want == round(12 * fs) and k == min(nk, want)
        assert e.size == _direct_keep(nb, nb + k, PICK_TAU) and np.array_equal(e, kt_cos[:e.size])
    e, k, want = new_expansion(np.arange(50), 0, 30)
    assert e.size == 0 and (k, want) == (0, 0)
    # 2. the via columns against a direct computation; B's rows are 'not from this source'
    B = np.asarray([3, 10])
    kt_cos = np.asarray([20, 5, 11, 7])
    cos0 = np.asarray([11, 7, 20, 5, 30])
    n1, cnt, wsum = np.asarray([5, 7, 30]), np.asarray([1, 70, 2]), np.asarray([2 / 61, 0.001, 5.0])
    r0, iout = np.asarray([7, 11]), np.asarray([20, 99])
    rows = np.union1d(B, np.asarray([20, 5, 11, 7]))
    x = via_columns(rows, kt_cos, cos0, n1, cnt, wsum, r0, iout)
    lp = np.log1p(float(VIA_PMAX))
    for r_i, v in enumerate(rows.tolist()):
        want_row = [np.log1p(kt_cos.tolist().index(v)) / lp if v in kt_cos else 1.0,
                    np.log1p(cos0.tolist().index(v)) / lp if v in cos0 else 1.0,
                    1.0 if v in n1 else 0.0,
                    min(1.0, np.log1p(cnt[n1.tolist().index(v)]) / np.log1p(64.0)) if v in n1 else 0.0,
                    min(1.0, np.log1p(30.5 * wsum[n1.tolist().index(v)]) / np.log1p(64.0)) if v in n1 else 0.0,
                    1.0 if v in r0 else 0.0, 1.0 if v in iout else 0.0]
        assert np.allclose(x[r_i], want_row, atol=1e-6), (v, x[r_i], want_row)
    assert x[rows == 3].tolist() == [[1.0, 1.0, 0, 0, 0, 0, 0]] and x[rows == 5][0, 4] == np.float32(np.log1p(1.0) / np.log1p(64.0))
    assert x[rows == 7][0, 3] == 1.0 and x[rows == 30].size == 0
    assert via_columns(np.asarray([4]), EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY).tolist() == [[1, 1, 0, 0, 0, 0, 0]]
    # 3. coverage from counts is p1c_pools.coverage; the caps against a direct count
    sets = [np.unique(rng.integers(0, 30, rng.integers(0, 12))) for _ in range(400)]
    golds = [np.unique(rng.integers(0, 30, rng.integers(0, 6))) for _ in range(400)]
    ng = np.asarray([g.size for g in golds])
    nh = np.asarray([int(np.isin(g, s).sum()) for s, g in zip(sets, golds)])
    assert coverage_counts(ng, nh) == C1.coverage(sets, golds)
    c = caps(ng, nh)
    assert c["r5"] == round(float(np.mean([min(5, h) / g if g else 0.0 for g, h in zip(ng, nh)])), 4)
    assert c["hit1"] == round(float(np.mean([h > 0 for h in nh])), 4)
    # 4. a part's file, the merge of parts, the replacement on both builds
    for build in BUILDS:
        whole, incl = _toy_filed(np.random.default_rng(7), 9, build)
        # split the whole into three parts as shards write them, then merge
        cuts = [U2C.shard_range(9, i, 3) for i in range(3)]
        sub = []
        for a0, a1 in cuts:
            p = {k: whole[k][a0:a1] for k in ("ids", "inc_sha") + PER_Q}
            p["bs_nodes"] = whole["bs_nodes"][whole["bs_ptr"][a0]:whole["bs_ptr"][a1]]
            p["nodes"] = whole["nodes"][whole["ptr"][a0]:whole["ptr"][a1]]
            p["bs_ptr"] = whole["bs_ptr"][a0:a1 + 1]
            p["ptr"] = whole["ptr"][a0:a1 + 1]
            p["gold_ptr"] = whole["gold_ptr"][a0:a1 + 1]
            p["held"] = whole["held"][whole["gold_ptr"][a0]:whole["gold_ptr"][a1]]
            sub.append(p)
        m = merge_part_arrays(sub)
        assert all(np.array_equal(m[k], whole[k]) for k in whole), build
        bad = [dict(s) for s in sub]
        bad[1]["gold_ptr"] = bad[1]["gold_ptr"] + 1
        try:
            merge_part_arrays(bad)
            raise AssertionError("parts whose golds do not continue passed")
        except SystemExit:
            pass
        s = carve_summary(m)
        assert s["questions"] == 9 and s["short"] == (9 if build == "new" else 0)
        assert s["pool"] == coverage_counts(m["n_gold"], m["n_held"])
        pools = PGB.pools_of(m)
        assert [p.size for p in pools] == PGB.expected_rows(m).tolist()
        prep = SimpleNamespace(pop=SimpleNamespace(ids=[str(q) for q in m["ids"].tolist()]), dense_ids=None,
                               splade_ids=None, pools=None, seeds_added=np.zeros(9, np.int64))
        bsl = [m["bs_nodes"][m["bs_ptr"][j]:m["bs_ptr"][j + 1]].astype(np.int64) for j in range(9)]
        prep.pools = [np.asarray(p, np.int64).copy() for p in incl]
        prep.seeds = [b[-1:] for b in bsl]

        class FakeM3C:
            @staticmethod
            def base_rows(construction, d, s_, m3a, m3b_contract, constant):
                return [b[:-1] if b.size > 1 else b for b in bsl]

            @staticmethod
            def build_pool(base, seeds, exp):
                measured = np.union1d(base, exp)
                return np.union1d(measured, seeds), int(np.isin(seeds, measured, invert=True).sum())

        st = PGB.replace_pools(prep, m, FakeM3C, None, {"retrieval_pools": {"equal_rrf": {"constant": 60}}}, None, None)
        assert st["questions"] == 9 and st["short"] == (9 if build == "new" else 0)
        assert all(np.array_equal(p, q) for p, q in zip(prep.pools, pools))
    # 5. the columns in the cache's order, and the refusals
    m, _incl = _toy_filed(np.random.default_rng(3), 5, "base")
    pools = PGB.pools_of(m)
    flat = np.concatenate([np.stack([p.astype(np.float32), -p.astype(np.float32)], 1) for p in pools])
    order = [3, 0, 4, 1, 2]
    c_ids = [str(m["ids"][j]) for j in order]
    c_rows = [rng.permutation(pools[j]) for j in order]
    out = to_cache_order(m, flat, c_ids, np.asarray([r.size for r in c_rows]), np.concatenate(c_rows))
    assert np.array_equal(out[:, 0], np.concatenate(c_rows).astype(np.float32))
    assert np.array_equal(out[:, 1], -np.concatenate(c_rows).astype(np.float32))
    j_big = int(np.argmax([p.size for p in pools]))
    assert pools[j_big].size >= 2
    for mutate in ("drop", "repeat", "other", "question"):
        rows_ = [r.copy() for r in c_rows]
        ids_ = list(c_ids)
        k = order.index(j_big)
        if mutate == "drop":
            rows_[k] = rows_[k][:-1]
        elif mutate == "repeat":
            rows_[k] = np.concatenate([rows_[k][:-1], rows_[k][:1]])
        elif mutate == "other":
            rows_[k] = np.concatenate([rows_[k][:-1], [10 ** 6]])
        else:
            ids_[k] = "nobody"
        try:
            to_cache_order(m, flat, ids_, np.asarray([r.size for r in rows_]), np.concatenate(rows_))
            raise AssertionError(f"a cache with a {mutate} passed")
        except SystemExit:
            pass
    # 6. step 1's rows by the n array's sha256, as step 1 saves it
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        sizes = m["today_inc_size"].astype(np.int32)
        np.save(T / "n.npy", sizes[[0, 1, 2]])
        recs = [{"part": 0, "ids": [str(q) for q in m["ids"][:3]],
                 "arrays": {"n": {"dtype": "int32", "shape": [3], "sha256": sha_file(T / "n.npy")}}},
                {"part": 1, "ids": [str(q) for q in m["ids"][3:]],
                 "arrays": {"n": {"dtype": "int32", "shape": [2], "sha256": npy_sha(sizes[3:])}}}]
        ids1, bad = step1_check(m, recs)
        assert not bad and ids1 == [str(q) for q in m["ids"]]
        m2 = dict(m)
        m2["today_inc_size"] = m["today_inc_size"] + np.asarray([0, 0, 0, 0, 1], np.int32)
        assert step1_check(m2, recs)[1] == ["part 1: step 1's n is not the filed today_inc_size"]
    # 7. shards tile a carve; the parts' set must be whole
    for nq in (1, 7, 1534):
        for k in (1, 2, 5):
            if k <= nq:
                rs = [U2C.shard_range(nq, i, k) for i in range(k)]
                assert rs[0][0] == 0 and rs[-1][1] == nq and all(a[1] == b[0] for a, b in zip(rs, rs[1:]))
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        for i in (0, 2):
            write_json(T / f"x__fit__{i}of3.json", {"part": i})
        try:
            find_parts(T, "x", "fit")
            raise AssertionError("an incomplete part set passed")
        except SystemExit:
            pass
        write_json(T / "x__fit__1of3.json", {"part": 1})
        assert [r["part"] for r in find_parts(T, "x", "fit")] == [0, 1, 2]
        write_json(T / "x__fit__0of1.json", {"part": 0})
        try:
            find_parts(T, "x", "fit")
            raise AssertionError("two part sets passed")
        except SystemExit:
            pass
    log(f"zu2d_pools selftest: both expansions (P1c2's fill; U2e's budget and P1h's cut against a direct loop); the via "
        f"columns against a direct computation; coverage from counts equal to p1c_pools'; the caps; the pools file, "
        f"its merge from parts and its refusals; the replacement on both builds; the columns in the cache's order and "
        f"its refusals; step 1's rows by the n array's sha256; the shards and the part sets ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("pools", "merge", "check", "check-merge", "copy-bases", "kt", "gate"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--build", choices=BUILDS)
    ap.add_argument("--carve")
    ap.add_argument("--shard")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd is None:
        ap.error("a command or --selftest")
    if a.cmd in ("pools", "merge", "check", "check-merge", "kt") and a.dataset not in SIX:
        ap.error(f"{a.cmd} DATASET: one of " + ", ".join(SIX))
    if a.cmd in ("pools", "merge", "copy-bases", "kt", "gate") and a.build is None:
        ap.error(f"{a.cmd} needs --build base or new")
    shard = None
    if a.shard:
        i, k = (int(x) for x in a.shard.split("/"))
        shard = (i, k)
    if a.cmd == "pools":
        if not a.carve:
            ap.error("pools needs --carve")
        return pools_stage(a.build, a.dataset, a.carve, a.threads, a.limit, shard)
    if a.cmd == "merge":
        return merge_stage(a.build, a.dataset)
    if a.cmd == "check":
        return check_stage(a.dataset, a.threads, a.limit, shard)
    if a.cmd == "check-merge":
        return check_merge(a.dataset)
    if a.cmd == "copy-bases":
        return copy_bases(a.build)
    if a.cmd == "kt":
        if not a.carve:
            ap.error("kt needs --carve")
        return kt_stage(a.build, a.dataset, a.carve)
    return gate(a.build)


if __name__ == "__main__":
    sys.exit(main())
