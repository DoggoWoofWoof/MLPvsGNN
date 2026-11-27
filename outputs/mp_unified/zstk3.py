"""STK3 (docs/STK3_STARK_POOLS.md): do the seven's pool cells carry to the STaRK knowledge bases. A transfer read: nothing is selected on a STaRK question.

    python -I -B outputs/mp_unified/zstk3.py selftest
    (on the lab host every command that opens the engine runs as `python outputs/host_ops/pylib_run.py outputs/mp_unified/zstk3.py <cmd> ... --host`: numba comes from pylib_run)
    python -I -B outputs/mp_unified/zstk3.py cells                              # laptop: the declared cell list (from PL6's select.json of the seven): outputs/stk3/cells.json
    python -I -B outputs/mp_unified/zstk3.py layout --kb prime                  # host: the B0 layout of the KB, F0T weights rebuilt in process, G-F and G-L: outputs/stk3/<kb>/layout.*
    python -I -B outputs/mp_unified/zstk3.py probe --kb prime [--questions 2]   # host: seconds per question and peak memory of the whole cell list; no label is read
    python -I -B outputs/mp_unified/zstk3.py run --kb prime [--part I K] [--upto M]   # host: G-S, then chunks of 128 questions in the declared order, each written when done
    python -I -B outputs/mp_unified/zstk3.py read --kb prime                    # laptop or host: the read of the chunks that exist: outputs/stk3/<kb>/read.json

One module, no per-KB branch. It imports the pool engine (zpl2g .. zpl6g), PL1's layout builder (zpl1m), STK2's name helpers (zstk2) and the seven's F0T rule
(zul1, d83stores), and edits none of them. The F0T order weights STK2 did not store are rebuilt here by zpl1h.f0t_in_process's rule and gated against STK2's keys.
Run from the workspace root.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import zpl1m as PM  # noqa: E402  (first: BLAS threads, then zul2c / zenc)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402,F401
import zpl5g as Z5  # noqa: E402
import zpl6g as Z6  # noqa: E402

DOC = "docs/STK3_STARK_POOLS.md"
SEVEN = ("squad", "musique", "hotpotqa", "2wiki", "metaqa", "webqsp", "multihoprag")
KB_ORDER = ("prime", "mag", "amazon")
OUT = ROOT / "outputs" / "stk3"
STARK = ROOT / "outputs" / "stark"
CHUNK = 128                                   # questions per chunk
PERM_SEED = 20261127                          # the chunk order: a seeded permutation of the chunk numbers (zleg1p's seed)
MAX_NODES = 16000.0                           # the cell rule: filed mean nodes of a pick on its own dataset at most this
DEPTH = 1000                                  # the first-stage lists the engine reads (the seven's depth)
REF_DEPTH = 2500                              # the depth of first.npz's lists
PICK_KEYS = ("R95_all", "R95_rung0", "R95_bounded", "R98_all", "R98_rung0")
FLAT_RE = re.compile(r"^(rrf1000_b\d+|dense_top\d+|splade_top200)$")
GATE_CELLS = ("dense_top20", "dense_top100", "splade_top200", "rrf1000_b20", "rrf1000_b100")
RRF_ALL_AGREE = 0.98                          # G-S (amendment 3): the all-gold booleans of rrf1000_b20 and rrf1000_b100 equal those of first.npz's fused top 20 and 100 on this share of the gate chunk
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc


# ── pure helpers (self-tested) ───────────────────────────────────────────────

def chunk_order(n_chunks, k):
    """The declared order of the chunk numbers for the KB with index k: a seeded permutation."""
    return [int(c) for c in np.random.default_rng([PERM_SEED, k]).permutation(n_chunks)]


def chunk_range(c, nq):
    return c * CHUNK, min((c + 1) * CHUNK, nq)


def twin(name):
    """The same walk over the other family label (FULL <-> STRUCT), or None."""
    if "+FULL:" in name:
        return name.replace("+FULL:", "+STRUCT:", 1)
    if "+STRUCT:" in name:
        return name.replace("+STRUCT:", "+FULL:", 1)
    return None


def flat_key(name):
    m = re.search(r"(\d+)$", name)
    return (0 if name.startswith("rrf1000_b") else (1 if name.startswith("dense_top") else 2), int(m.group(1)) if m else 0)


def sha_lf(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def maxrank(lists, golds):
    """Per question the 0-based position of the last gold in the list, or -1 when a gold is not in it (the list's depth is then too short)."""
    out = np.full(len(golds), -1, np.int64)
    for q, g in enumerate(golds):
        m = np.isin(lists[q], g)
        if int(m.sum()) == g.size:
            out[q] = int(np.flatnonzero(m).max())
    return out


def all_at(mr, depth):
    """All golds inside the top `depth` of the list: maxrank in [0, depth)."""
    return (mr >= 0) & (mr < depth)


def set_agree(a, b, n):
    """Share of questions whose first n entries of the two row lists are the same set."""
    ok = [set(map(int, x[:n])) == set(map(int, y[:n])) for x, y in zip(a, b)]
    return float(np.mean(ok)) if ok else float("nan")


def call_lift(ci):
    """LIFTS if the paired interval of (pick - flat) lies above 0, BELOW if below, else LEVEL."""
    if ci[0] > 0:
        return "LIFTS"
    if ci[1] < 0:
        return "BELOW"
    return "LEVEL"


def pick_cells(selects, max_nodes=MAX_NODES):
    """selects: {dataset: select.json dict}. Returns (picks, twins, flat, left_out, why): the pick cells (the five pick keys and the incumbent) whose filed mean
    nodes on the dataset that named them are at most max_nodes, the FULL/STRUCT twins of those that the same table holds under the cap, and the flat ladder."""
    cand, tables = {}, {}
    for ds, sel in selects.items():
        tables[ds] = {r["name"]: r for r in sel["table"]}
        found = [(k, (sel["picks"].get(k) or {}).get("name")) for k in PICK_KEYS] + [("incumbent", (sel.get("incumbent") or {}).get("name"))]
        for key, nm in found:
            if not nm:
                continue
            if nm not in tables[ds]:
                stop(f"{ds}: {key} names {nm}, which is not in its table")
            cand.setdefault(nm, []).append({"dataset": ds, "key": key, "nodes": float(tables[ds][nm]["nodes"])})
    picks = [n for n, v in cand.items() if min(x["nodes"] for x in v) <= max_nodes]
    left = [{"name": n, "from": v} for n, v in cand.items() if n not in picks]
    twins, why = [], {}
    for n in picks:
        t = twin(n)
        if not t or t in picks or t in twins:
            continue
        for ds in {x["dataset"] for x in cand[n]}:
            r = tables[ds].get(t)
            if r is not None and float(r["nodes"]) <= max_nodes:
                twins.append(t)
                why[t] = {"of": n, "dataset": ds, "nodes": float(r["nodes"])}
                break
    flat = sorted({n for t in tables.values() for n in t if FLAT_RE.match(n)} - set(picks) - set(twins), key=flat_key)   # a flat pick (SQuAD's) stays a pick
    return picks, twins, flat, left, {"cand": cand, "twin_of": why}


# ── cells ─────────────────────────────────────────────────────────────

def cells_path():
    return OUT / "cells.json"


def cmd_cells(a):
    Z6.prepare(False)
    selects, sha = {}, {}
    for ds in SEVEN:
        p = Z6.PL6 / ds / "select.json"
        selects[ds] = rjson(p)
        sha[ds] = sha_lf(p)
    picks, twins, flat, left, info = pick_cells(selects)
    names = picks + twins + flat
    if len(set(names)) != len(names):
        stop("a cell name is twice in the list")
    plan = Z5.Plan5(names)
    if plan.C != len(names):
        stop(f"the plan holds {plan.C} cells for {len(names)} names")
    rec = {"stage": "STK3 cells", "declared_in": DOC, "utc": utc(), "max_nodes": MAX_NODES, "pick_keys": list(PICK_KEYS), "names": names,
           "groups": {"picks": picks, "twins": twins, "flat": flat}, "cells_sha256": P2.cells_sha256(names), "select_sha256": sha,
           "picked_by": info["cand"], "twin_of": info["twin_of"], "left_out": left, "chunk": CHUNK, "perm_seed": PERM_SEED, "depth": DEPTH,
           "script_sha256": sha_lf(__file__)}
    wjson(cells_path(), rec)
    log(f"cells: {len(picks)} picks, {len(twins)} twins, {len(flat)} flat = {len(names)}; {len(left)} left out above {MAX_NODES:.0f} nodes: "
        f"{[x['name'] for x in left]}")
    return 0


# ── the layout ───────────────────────────────────────────────────────────

def kb_dir(kb):
    return OUT / kb


def f0t_in_process(titles, texts, n):
    """F0T's pairs with D83's weights, in process (zpl1h.f0t_in_process's rule: zul1.build_entries and forms_from_entries over the nodes' titles and texts, the two
    tables joined by d83stores.form_pairs). Returns ((a, b, w), F0 keys, F0T keys)."""
    import d83stores as DSt
    import zul1 as UL
    passes = lambda: ((t, x) for t, x in zip(titles, texts))  # noqa: E731
    src, grp, tsrc, tgrp, _nv = UL.build_entries(passes, n)
    (fa, fb, fw), (la, lb, lw) = UL.forms_from_entries(src, grp, tsrc, tgrp, n)
    f0 = UL.pair_keys(fa, fb, n)
    lt = UL.pair_keys(la, lb, n)
    if f0.size != fa.size or lt.size != la.size:
        stop("a co-mention table holds a pair twice or a self pair")
    a, b, w = DSt.form_pairs([(fa, fb, fw.astype(np.float32)), (la, lb, lw.astype(np.float32))], n)
    return (a, b, w), f0, np.union1d(f0, lt)


def struct_arrays(keys, n):
    return (keys // n).astype(np.int32), (keys % n).astype(np.int32)


def families_of(M3A, tab, n):
    sa, sb = struct_arrays(tab["structural_keys"], n)
    gs = SimpleNamespace(n_nodes=n, src=sa, dst=sb, weight=None)
    gn = SimpleNamespace(n_nodes=n, src=tab["ner"]["a"], dst=tab["ner"]["b"], weight=tab["ner"]["order_key"])
    gk = SimpleNamespace(n_nodes=n, src=tab["knn"]["a"], dst=tab["knn"]["b"], weight=tab["knn"]["weight"])
    return {"structural": M3A.FamilyCSR(gs, "structural"), "ner": M3A.FamilyCSR(gn, "ner"), "knn": M3A.FamilyCSR(gk, "knn")}


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
    return P2.sha256_file(path)


HOSTMODE = False                              # --host: the lab host's package mirror (zpkoh.apply_mirror), as every host row opens the engine


def plain_op():
    Z6.prepare(HOSTMODE)
    return P2.opened()


def cmd_layout(a):
    import zstk2 as K2
    import zworld as W
    t0 = time.time()
    kb = a.kb
    ds = K2.KBS[kb]
    od = kb_dir(kb)
    if (od / "layout.json").exists() and not a.redo:
        stop(f"{P2.rel(od / 'layout.json')} exists (the layout is filed once; --redo rewrites it)")
    w = W.load(ds)
    n = w.n
    base = STARK / kb
    lk = rjson(base / "links.json")
    nrec = rjson(base / "names.json")
    E = K2.E
    if E.sha256(base / "names.jsonl") != nrec["names_sha256"] or nrec["dataset_json_sha256"] != E.sha256(w.root / "DATASET.json") \
            or lk["dataset_json_sha256"] != E.sha256(w.root / "DATASET.json"):
        stop("the names sidecar, the links record and the package are not one set")
    fl = base / "links"
    sha_ok = {k: E.sha256(fl / f"{k}.npz") == v for k, v in lk["files"].items()}
    if not all(sha_ok.values()):
        stop(f"a links file is not its record's: {sha_ok}")
    ids, _types, texts = K2.read_nodes(w)
    names = []
    with open(base / "names.jsonl", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            r = json.loads(ln)
            if r["node_id"] != ids[i]:
                stop(f"names row {i} is {r['node_id']!r}, the package's node {i} is {ids[i]!r}")
            names.append(r["name"])
    if len(names) != n:
        stop(f"{len(names)} names for {n} nodes")
    titles = [K2.title_of(nm, i) for nm, i in zip(names, ids)]
    f0t, f0_keys, f0t_keys = f0t_in_process(titles, texts, n)
    S = np.load(fl / "S.npz")["keys"].astype(np.int64)
    W2 = np.load(fl / "W2_S.npz")["keys"].astype(np.int64)
    f0_file = np.load(fl / "F0.npz")["keys"].astype(np.int64)
    f0t_file = np.load(fl / "F0T.npz")["keys"].astype(np.int64)
    with np.load(fl / "K.npz") as z:
        K = PM.knn_triple(z["src"], z["dst"], z["weight"], n)
    kkeys = K[0].astype(np.int64) * n + K[1].astype(np.int64)
    gf = {"links_files_equal_record": sha_ok, "F0_keys_equal": bool(np.array_equal(f0_keys, f0_file)), "F0T_keys_equal": bool(np.array_equal(f0t_keys, f0t_file)),
          "F0_pairs": int(f0_keys.size), "F0T_pairs": int(f0t_keys.size), "weights_positive": bool((f0t[2] > 0).all()),
          "triple_keys_equal": bool(np.array_equal(f0t[0].astype(np.int64) * n + f0t[1], f0t_keys)),
          "S_pairs": int(S.size), "S_pairs_record": int(lk["S"]["pairs"]), "S_equal_record": int(S.size) == int(lk["S"]["pairs"])}
    gf["verdict"] = "PASS" if (all(sha_ok.values()) and gf["F0_keys_equal"] and gf["F0T_keys_equal"] and gf["weights_positive"] and gf["triple_keys_equal"]
                               and gf["S_equal_record"]) else "FAIL"
    if gf["verdict"] != "PASS":
        wjson(od / "layout.json", {"stage": "STK3 layout", "kb": kb, "gate_GF": gf, "verdict": "FAIL"})
        stop(f"G-F FAIL {gf}")
    tab, counts = PM.tables(n, "SRK", S, W2, f0t, K, None)
    exp = PM.expected_union(n, "SRK", S, W2, f0t_keys, kkeys, None)
    gl = PM.gates(tab, counts, n, "SRK", exp)
    gl["union_equals_store"] = {"here": int(counts["union"]), "stk2_store": int(lk["store"]), "equal": int(counts["union"]) == int(lk["store"])}
    op = plain_op()
    M3A = op.m3a
    stores = PM.to_stores(tab, n)
    fam = families_of(M3A, tab, n)
    order = {}
    for name, st in (("ner", stores[0]), ("knn", stores[1])):
        c = fam[name]
        order[name] = bool(np.array_equal(np.asarray(c.indptr, np.int64), np.asarray(st.indptr, np.int64))
                           and np.array_equal(np.asarray(c.col, np.int64), np.asarray(st.col, np.int64)))
    gl["row_order_equals_FamilyCSR"] = order
    if not (gl["union_equals_store"]["equal"] and all(order.values())):
        gl["verdict"] = "FAIL"
    flat = {f"ner_{k}": x for k, x in tab["ner"].items()}
    flat.update({f"knn_{k}": x for k, x in tab["knn"].items()})
    flat.update(struct_key=tab["structural_keys"], struct_label=tab["structural_label"])
    sha = save_npz(od / "layout.npz", **flat) if gl["verdict"] == "PASS" else None
    rec = {"stage": "STK3 layout", "declared_in": DOC, "kb": kb, "dataset": ds, "version": "SRK", "layout_kind": "B0", "nodes": n,
           "inputs": {"links_json_sha256": E.sha256(base / "links.json"), "links_files": lk["files"], "names_sha256": nrec["names_sha256"]},
           "counts": counts, "gate_GF": gf, "gate_GL": gl, "layout_npz_sha256": sha, "seconds": round(time.time() - t0, 1), "peak_gb": P2.peak_gb(), "utc": utc(),
           "script_sha256": sha_lf(__file__), "verdict": gl["verdict"]}
    wjson(od / "layout.json", rec)
    log(f"{kb}: layout {gl['verdict']}: structural {counts['structural_pairs']:,} (S {counts['structural_S']:,}, R {counts['structural_R']:,}), ner {counts['ner_pairs_out']:,}, "
        f"knn {counts['knn_pairs_out']:,}, union {counts['union']:,} (STK2 store {int(lk['store']):,}); {rec['seconds']} s, peak {rec['peak_gb']} GB")
    return 0 if gl["verdict"] == "PASS" else 3


def read_layout(kb):
    rec = rjson(kb_dir(kb) / "layout.json")
    if rec.get("verdict") != "PASS":
        stop(f"{kb}: the layout record is not PASS")
    p = kb_dir(kb) / "layout.npz"
    if P2.sha256_file(p) != rec["layout_npz_sha256"]:
        stop(f"{kb}: layout.npz is not its record's")
    z = np.load(p)
    try:
        tab = {"ner": {k[4:]: z[k] for k in z.files if k.startswith("ner_")}, "knn": {k[4:]: z[k] for k in z.files if k.startswith("knn_")},
               "structural_keys": z["struct_key"], "structural_label": z["struct_label"]}
    finally:
        z.close()
    return tab, rec


# ── the source ───────────────────────────────────────────────────────────

def load_stark_source(kb, split="test"):
    """The engine's Source for one KB's questions of a split: the first stage's lists (first.npz, cut to DEPTH), the golds (distinct positions, sorted), e3v's query
    and document vectors, and the three families of the layout. No per-KB branch."""
    import zstk2 as K2
    import zworld as W
    op = plain_op()
    ds = K2.KBS[kb]
    w = W.load(ds)
    qs = list(w.iter_queries(splits=[split]))
    rows = np.asarray([q["row"] for q in qs], np.int64)
    with np.load(STARK / kb / "first.npz") as z:
        if not np.array_equal(z["rows"].astype(np.int64), rows):
            stop(f"{kb}: first.npz's rows are not the {split} split's query rows")
        dense = z["top_dense_e3v"][:, :DEPTH].astype(np.int32)
        splade = z["top_splade"][:, :DEPTH].astype(np.int32)
    golds = [np.unique(q["gold_pos"]).astype(np.int64) for q in qs]
    if any(g.size == 0 for g in golds):
        stop(f"{kb}: a question with no gold node")
    ids = [str(q["query_id"]) for q in qs]
    tab, lay = read_layout(kb)
    src = P2.Source()
    src.name, src.carve, src.first, src.n = ds, split, "e3v", int(w.n)
    pop = {"ids": ids, "idx": rows, "golds": golds, "digest": P2.digest_ids(ids)}
    cache = {}

    def loader(f):
        if not cache:
            cache.update({k: P2.CSR(v.indptr, v.col) for k, v in families_of(op.m3a, tab, int(w.n)).items()})
        return cache[f]
    fams = P2.Fams(loader)
    qdir, ddir = w.vector_dir("e3v", "queries"), w.vector_dir("e3v", "docs")
    src.graph_kind = "layout:B0"
    src.layout_sha256 = lay["layout_npz_sha256"]
    return op, P2.finish_source(src, pop, dense, splade, fams, qdir, ddir)


def ref_path(kb):
    return kb_dir(kb) / "ref.npz"


def make_ref(kb, src):
    """The reference arrays the read needs without the package: qids, gold counts and the position of the last gold in each first-stage list (first.npz, depth 2,500)."""
    p = ref_path(kb)
    if p.is_file():
        return
    with np.load(STARK / kb / "first.npz") as z:
        fused, dn, sp = z["top_rrf_e3v_splade"], z["top_dense_e3v"], z["top_splade"]
        arr = {"maxrank_fused": maxrank(fused, src.golds), "maxrank_dense": maxrank(dn, src.golds), "maxrank_splade": maxrank(sp, src.golds)}
    save_npz(p, qids=np.asarray(src.qids, dtype=str), gold_counts=src.gold_counts.astype(np.int32), depth=np.asarray(REF_DEPTH), **arr)


# ── gates and probe ──────────────────────────────────────────────────────

def gate_s(op, kb, src, lo, hi):
    """G-S: the engine's flat cells equal the lists of first.npz on the gate chunk (exact for the dense and SPLADE tops; the fused tops by their all-gold booleans, amendment 3:
    the engine's fusion (M3A's inherited equal-RRF over lists 1,000 deep) and STK1's (float32 1/(60 + rank), lists 2,500 deep, ties by node id) are two implementations, so
    their top-N sets are reported, not gated)."""
    M3A = op.m3a
    plan = Z5.Plan5(list(GATE_CELLS))
    P, S = Z5.evaluate5(op, src, plan, lo, hi, every=64)
    gc = src.gold_counts[lo:hi]
    with np.load(STARK / kb / "first.npz") as z:
        dn, sp, fu = z["top_dense_e3v"][lo:hi], z["top_splade"][lo:hi], z["top_rrf_e3v_splade"][lo:hi]
    gl = src.golds[lo:hi]
    out = {"questions": int(hi - lo), "exact": {}, "rrf": {}}
    for nm, lst, k in (("dense_top20", dn, 20), ("dense_top100", dn, 100), ("splade_top200", sp, 200)):
        ref = np.asarray([np.isin(g, x[:k]).all() for g, x in zip(gl, lst)])
        got = P[:, plan.names.index(nm)].astype(np.int64) == gc
        out["exact"][nm] = {"mismatches": int((ref != got).sum()), "sizes_ok": bool((S[:, plan.names.index(nm)] == k).all())}
    d = np.asarray(src.dense[lo:hi], dtype=np.int64)
    s = np.asarray(src.splade[lo:hi], dtype=np.int64)
    fused = M3A.rrf_fused_rows(d, s, P2.RRF_CONST)
    for nm, k in (("rrf1000_b20", 20), ("rrf1000_b100", 100)):
        ag = set_agree(fused, fu, k)
        ref = np.asarray([np.isin(g, x[:k]).all() for g, x in zip(gl, fu)])
        got = P[:, plan.names.index(nm)].astype(np.int64) == gc
        out["rrf"][nm] = {"set_agreement": ag, "all_gold_agreement": float((ref == got).mean())}
    out["verdict"] = "PASS" if (all(v["mismatches"] == 0 and v["sizes_ok"] for v in out["exact"].values())
                                and all(v["all_gold_agreement"] >= RRF_ALL_AGREE for v in out["rrf"].values())) else "FAIL"
    return out


def cmd_probe(a):
    cells = rjson(cells_path())
    names = cells["names"]
    t0 = time.time()
    op, src = load_stark_source(a.kb)
    load_s = time.time() - t0
    plan = Z5.Plan5(names)
    nq = int(a.questions)
    empty = np.zeros(0, np.int64)                                           # no label is read: grade3 sees an empty gold list
    P = np.zeros((plan.C,), np.int16)
    S = np.zeros((plan.C,), np.int32)
    per_q, sizes = [], []
    t1 = time.time()
    for q, st in Z5.iter_states5(op, src, plan, 0, nq):
        Z3.grade3(plan, st, empty, P, S)
        per_q.append(round(time.time() - t1, 2))
        t1 = time.time()
        sizes.append([int(x) for x in S])
    sz = np.asarray(sizes, np.int64)
    rec = {"stage": "STK3 probe", "declared_in": DOC, "kb": a.kb, "utc": utc(), "questions": nq, "load_seconds": round(load_s, 1), "seconds_per_question": per_q,
           "cells": plan.C, "cells_sha256": cells["cells_sha256"], "peak_gb": P2.peak_gb(), "layout_sha256": src.layout_sha256,
           "pool_size_by_cell": {nm: [int(x) for x in sz[:, i]] for i, nm in enumerate(plan.names)}, "labels_read": False}
    wjson(kb_dir(a.kb) / "probe.json", rec)
    log(f"{a.kb}: probe {nq} questions: {per_q} s each, load {load_s:.0f} s, peak {rec['peak_gb']} GB; largest pool {int(sz.max())} nodes")
    return 0


# ── run ──────────────────────────────────────────────────────────────────

def chunk_path(kb, c):
    return kb_dir(kb) / f"chunk_{c:04d}.npz"


def cmd_run(a):
    kb = a.kb
    cells = rjson(cells_path())
    names = cells["names"]
    part, nparts = (int(a.part[0]), int(a.part[1])) if a.part else (0, 1)
    op, src = load_stark_source(kb)
    make_ref(kb, src)
    nq = src.nq
    n_chunks = (nq + CHUNK - 1) // CHUNK
    order = chunk_order(n_chunks, KB_ORDER.index(kb))
    gp = kb_dir(kb) / "gate_s.json"
    gate = rjson(gp) if gp.is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        lo, hi = chunk_range(order[0], nq)
        t0 = time.time()
        gate = {"stage": "STK3 gates", "declared_in": DOC, "kb": kb, "utc": utc(), "gate_chunk": order[0], "G-S": gate_s(op, kb, src, lo, hi),
                "seconds": round(time.time() - t0, 1)}
        gate["verdict"] = gate["G-S"]["verdict"]
        wjson(gp, gate)
        log(f"{kb}: G-S {gate['verdict']} {json.dumps(gate['G-S'])[:400]}")
        if gate["verdict"] != "PASS":
            return 3
    plan = Z5.Plan5(names)
    upto = int(a.upto) if a.upto else len(order)
    done = 0
    for pos in range(part, min(upto, len(order)), nparts):
        c = order[pos]
        path = chunk_path(kb, c)
        if path.is_file():
            continue
        lo, hi = chunk_range(c, nq)
        t0 = time.time()
        P, S = Z5.evaluate5(op, src, plan, lo, hi, every=16)
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as f:
            np.savez(f, lo=lo, hi=hi, names=np.asarray(names), present=P, size=S, qids=np.asarray(src.qids[lo:hi], dtype=str),
                     gold_counts=src.gold_counts[lo:hi].astype(np.int32), seconds=round(time.time() - t0, 1), cells_sha256=cells["cells_sha256"],
                     layout_sha256=src.layout_sha256)
        os.replace(tmp, path)
        done += 1
        log(f"{kb}: chunk {c} (questions {lo}-{hi}) done in {time.time() - t0:.0f} s ({done} this run, position {pos + 1} of {len(order)}; {P2.peak_gb()} GB)")
    log(f"{kb}: part {part} of {nparts} finished ({done} chunks made now)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────

def load_chunks(kb, names, cells_sha):
    idx, qids, pres, size, gcs = [], [], [], [], []
    for p in sorted(kb_dir(kb).glob("chunk_*.npz")):
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["names"]] != names or str(z["cells_sha256"]) != cells_sha:
                stop(f"{p.name}: its cells are not the declared list")
            lo, hi = int(z["lo"]), int(z["hi"])
            idx.append(np.arange(lo, hi))
            qids += [str(x) for x in z["qids"]]
            pres.append(z["present"].astype(np.int64))
            size.append(z["size"].astype(np.float64))
            gcs.append(z["gold_counts"].astype(np.int64))
        finally:
            z.close()
    if not idx:
        return None
    return np.concatenate(idx), qids, np.concatenate(pres), np.concatenate(size), np.concatenate(gcs)


def cmd_read(a):
    kb = a.kb
    cells = rjson(cells_path())
    names = cells["names"]
    gp = kb_dir(kb) / "gate_s.json"
    if not gp.is_file() or rjson(gp).get("verdict") != "PASS":
        stop(f"{kb}: no PASS gate file: no number is read")
    got = load_chunks(kb, names, cells["cells_sha256"])
    if got is None:
        stop(f"{kb}: no chunks")
    idx, qids, pres, size, gc = got
    z = np.load(ref_path(kb), allow_pickle=False)
    try:
        ref_q, ref_gc = [str(x) for x in z["qids"]], z["gold_counts"].astype(np.int64)
        mr = {k[8:]: z[k].astype(np.int64) for k in z.files if k.startswith("maxrank_")}
    finally:
        z.close()
    if [ref_q[i] for i in idx] != qids or not np.array_equal(ref_gc[idx], gc):
        stop(f"{kb}: the chunks' questions are not the reference's")
    nq_total = len(ref_q)
    k_kb = KB_ORDER.index(kb)
    rng = np.random.default_rng([P2.BOOT_SEED, k_kb])
    allg = pres == gc[:, None]
    anyg = pres > 0
    fused_mr = mr["fused"][idx]
    rows, at = {}, {n: i for i, n in enumerate(names)}
    for i, n in enumerate(names):
        q = int(idx.size)
        mean_nodes = float(size[:, i].mean())
        ref_n = int(min(REF_DEPTH, math.ceil(mean_nodes)))
        ref_all = all_at(fused_mr, ref_n)
        pr = P2.paired((allg[:, i], size[:, i]), (ref_all, np.full(q, float(ref_n))), rng)
        rows[n] = {"all": float(allg[:, i].mean()), "all_ci": list(P2.wilson(int(allg[:, i].sum()), q)), "any": float(anyg[:, i].mean()), "nodes": mean_nodes,
                   "nodes_p95": float(np.percentile(size[:, i], 95)), "nodes_max": float(size[:, i].max()),
                   "flat_ref": {"list": "first.npz fused top N", "N": ref_n, "all": float(ref_all.mean()), "reference_shorter_than_pool": bool(mean_nodes > REF_DEPTH),
                                "diff": pr["all_diff"], "diff_ci": pr["all_diff_ci"], "call": call_lift(pr["all_diff_ci"])}}
    pairs = {}
    for t, why in cells["twin_of"].items():
        o = why["of"]
        full, stru = (o, t) if "+FULL:" in o else (t, o)
        pr = P2.paired((allg[:, at[full]], size[:, at[full]]), (allg[:, at[stru]], size[:, at[stru]]), rng)
        pairs[f"{full} vs {stru}"] = {"full_all": float(allg[:, at[full]].mean()), "struct_all": float(allg[:, at[stru]].mean()), "diff": pr["all_diff"],
                                      "diff_ci": pr["all_diff_ci"], "nodes_ratio": pr["nodes_ratio"], "nodes_ratio_ci": pr["nodes_ratio_ci"]}
    base = {"fused": {"ALL@%d" % d: float(all_at(fused_mr, d).mean()) for d in (100, 500, 1000, 2000, 2500)},
            "dense": {"ALL@%d" % d: float(all_at(mr["dense"][idx], d).mean()) for d in (100, 500, 1000, 2000, 2500)}}
    reach = sorted([(rows[n]["nodes"], n) for n in names if rows[n]["all"] >= 0.95])
    best = {}
    for lim in (500, 1000, 2500, 3500, 5000, 16000):
        cand = [(rows[n]["all"], n) for n in names if rows[n]["nodes"] <= lim]
        best[str(lim)] = dict(zip(("all", "name"), max(cand))) if cand else {"all": None, "name": None}
    rec = {"stage": "STK3 read", "declared_in": DOC, "kb": kb, "utc": utc(), "questions_covered": int(idx.size), "questions_total": nq_total,
           "status": "COMPLETE" if idx.size == nq_total else "PARTIAL", "chunks": f"{len(list(kb_dir(kb).glob('chunk_*.npz')))} of {(nq_total + CHUNK - 1) // CHUNK}",
           "first_stage_on_covered": base, "cells": rows, "family_pairs": pairs, "reach_95": [{"nodes": a_, "name": n_} for a_, n_ in reach],
           "best_by_node_limit_POST_HOC": best, "cells_sha256": cells["cells_sha256"]}
    wjson(kb_dir(kb) / "read.json", rec)
    log(f"{kb} [{rec['status']} {idx.size}/{nq_total}]: flat fused ALL@2500 {base['fused']['ALL@2500']:.4f}; cells reaching 0.95: {len(reach)}; best by limit {best}")
    return 0


# ── self-test ────────────────────────────────────────────────────────────

def cmd_selftest(a):
    checks, bad = 0, []

    def ok(cond, what):
        nonlocal checks
        checks += 1
        if not cond:
            bad.append(what)
        print(("PASS " if cond else "FAIL ") + what)

    ok(chunk_order(22, 0) == chunk_order(22, 0) and sorted(chunk_order(22, 0)) == list(range(22)) and chunk_order(22, 0) != chunk_order(22, 1), "chunk order: a seeded permutation per KB")
    ok(chunk_range(21, 2801) == (2688, 2801) and chunk_range(0, 2801) == (0, 128), "chunk ranges")
    ok(twin("rrf1000_b400+FULL:h2_bf_v2000~cs") == "rrf1000_b400+STRUCT:h2_bf_v2000~cs" and twin("a+STRUCT:x") == "a+FULL:x" and twin("rrf1000_b15") is None, "twin: FULL <-> STRUCT")
    ok([bool(FLAT_RE.match(x)) for x in ("rrf1000_b15", "dense_top5", "splade_top200", "rrf1000_b50+KNN:h1", "frozen_union", "equal_rrf_budget_50")] == [True, True, True, False, False, False],
       "flat names")
    lists = np.asarray([[5, 1, 9, 3], [2, 4, 6, 8]])
    golds = [np.asarray([1, 3]), np.asarray([4, 7])]
    mr = maxrank(lists, golds)
    ok(list(mr) == [3, -1] and list(all_at(mr, 4)) == [True, False] and list(all_at(mr, 3)) == [False, False], "maxrank and all_at (a gold outside the list is -1, never inside)")
    ok(set_agree([np.asarray([1, 2, 3])], [np.asarray([2, 1, 9])], 2) == 1.0 and set_agree([np.asarray([1, 2, 3])], [np.asarray([3, 2, 9])], 3) == 0.0, "set agreement")
    ok([call_lift([0.01, 0.05]), call_lift([-0.02, 0.01]), call_lift([-0.05, -0.01])] == ["LIFTS", "LEVEL", "BELOW"], "calls: LIFTS, LEVEL, BELOW")
    sel = {"table": [{"name": "rrf1000_b15", "nodes": 15.0}, {"name": "rrf1000_b50+FULL:h3_bf_v20000~cs", "nodes": 9000.0}, {"name": "rrf1000_b50+STRUCT:h3_bf_v20000~cs", "nodes": 8000.0},
                     {"name": "rrf1000_b60+FULL:h3_bf_v40000~cs", "nodes": 39000.0}, {"name": "dense_top5", "nodes": 5.0}, {"name": "frozen_union", "nodes": 300.0}],
           "picks": {"R95_all": {"name": "rrf1000_b60+FULL:h3_bf_v40000~cs"}, "R95_rung0": {"name": "rrf1000_b50+FULL:h3_bf_v20000~cs"},
                     "R95_bounded": {"name": "rrf1000_b15"}, "R98_all": None, "R98_rung0": {"name": "rrf1000_b15"}},
           "incumbent": {"name": "rrf1000_b15"}}
    pk, tw, fl, left, info = pick_cells({"toy": sel})
    ok(set(pk) == {"rrf1000_b50+FULL:h3_bf_v20000~cs", "rrf1000_b15"} and tw == ["rrf1000_b50+STRUCT:h3_bf_v20000~cs"] and fl == ["dense_top5"]
       and [x["name"] for x in left] == ["rrf1000_b60+FULL:h3_bf_v40000~cs"], "cells: the cap leaves out a 39,000-node pick; the twin under the cap joins; the flat ladder leaves out a flat pick")
    # a toy layout through PL1's builder: F0T weights positive, the union as expected, the families disjoint
    n = 12
    rng = np.random.default_rng(0)
    allp = np.unique(np.asarray([a_ * n + b_ for a_ in range(n) for b_ in range(a_ + 1, n)]))
    pick = rng.permutation(allp)
    S, W2 = np.sort(pick[:8]), np.sort(pick[6:12])
    fk = np.sort(pick[10:16])
    f0t = ((fk // n).astype(np.int32), (fk % n).astype(np.int32), rng.uniform(0.5, 2, fk.size).astype(np.float32))
    kk = np.sort(pick[14:24])
    K = ((kk // n).astype(np.int32), (kk % n).astype(np.int32), rng.uniform(0.1, 1, kk.size).astype(np.float32))
    tab, counts = PM.tables(n, "SRK", S, W2, f0t, K, None)
    exp = PM.expected_union(n, "SRK", S, W2, fk, kk, None)
    g = PM.gates(tab, counts, n, "SRK", exp)
    ok(g["verdict"] == "PASS" and counts["union"] == np.union1d(np.union1d(S, W2), np.union1d(fk, kk)).size, "the SRK layout of a toy: gates PASS, the union is the sets' union")
    ok(set(map(int, tab["structural_keys"][tab["structural_label"] == 0])) == set(map(int, S)), "the structural store labels S as 0")
    # Plan5 builds over a name list that has a flat, a walk and a family cell
    Z6.prepare(False)
    try:
        plan = Z5.Plan5(["rrf1000_b15", "dense_top20", "rrf1000_b25+FULL:h1_c100+top25", "rrf1000_b400+STRUCT:h3_c25_v2000"])
        ok(plan.C == 4, "Plan5 parses the walk cells of PL5 and PL6")
    except SystemExit as e:
        ok(False, f"Plan5 parses the walk cells ({e})")
    if os.path.isfile(cells_path()):
        c = rjson(cells_path())
        ok(len(c["names"]) == len(set(c["names"])) and c["cells_sha256"] == P2.cells_sha256(c["names"]), "cells.json: unique names and its hash")
    print(f"{checks - len(bad)}/{checks} checks passed")
    if bad:
        print("FAILED:", bad)
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("cells")
    for c in ("layout", "probe", "run", "read"):
        p = sub.add_parser(c)
        p.add_argument("--kb", required=True, choices=KB_ORDER)
        if c != "read":
            p.add_argument("--host", action="store_true")
        if c == "layout":
            p.add_argument("--redo", action="store_true")
        if c == "probe":
            p.add_argument("--questions", type=int, default=2)
        if c == "run":
            p.add_argument("--part", nargs=2, metavar=("I", "K"))
            p.add_argument("--upto", type=int, default=0)
    a = ap.parse_args(argv)
    global HOSTMODE
    HOSTMODE = bool(getattr(a, "host", False))
    return {"selftest": cmd_selftest, "cells": cmd_cells, "layout": cmd_layout, "probe": cmd_probe, "run": cmd_run, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
