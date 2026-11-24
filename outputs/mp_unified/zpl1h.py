"""PL1 step S0 (docs/PL1_MHR_S0.md): MultiHop-RAG's pool record chain, so its RK arm is graded like the other six.

    python outputs/mp_unified/zpl1h.py first                               # S0a: the e3v lists, the gold table, G-E
    python outputs/mp_unified/zpl1h.py layout                              # S0b: B0 layout, G-F and G-L
    python outputs/mp_unified/zpl1h.py gk                                  # G-K: the knee code on MuSiQue's committed grid
    python outputs/mp_unified/zpl1h.py knee                                # S0c: the own construction on dev (e3v), the frozen read beside
    python outputs/mp_unified/zpl1h.py record                              # S0d: the record and G-R
    python outputs/mp_unified/zpl1h.py pools --carve s1sel|s1eval [--shard I K] [--limit N] [--threads 2]   # S0e / S0f
    python outputs/mp_unified/zpl1h.py gates                               # G-W (ii) and G-N, after S0e
    python outputs/mp_unified/zpl1h.py selftest

MultiHop-RAG is the seventh dataset and is treated as the other six: no new rule, constant or cut. This module imports the pinned
modules and edits none. Everything that is keyed to the six datasets (canonical.DATASETS, the package loader's vectors and caches, step 4's
opened package and its populations, the structural graph swap, the stores' cache, B1P's union_pools, the carve's depth check) is handled by
in-process patches, each put back on exit. The one pinned function that cannot be reached by a patch, zu2d_pools.open_carve (its depth
check needs 1,000 candidates per list, MultiHop-RAG has 609 nodes), is re-compiled from its own source with that one condition read as
min(1000, n); the self-test compares the two sources line by line. The pool rule is zpko.rule_batch as it is (pick O), the places are
zpl1w.make_rule's, the types are 2g's null types in memory (the loaders' own shape, weights of exactly 1), the lists are the full
609-node rankings (no padding). Gold is read only by the places pass.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import zpl1m as PM  # noqa: E402  (first: BLAS threads, then zul2c / zenc)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import difflib  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import zenc as E  # noqa: E402
import zmhr as ZM  # noqa: E402
import zpl1w as PW  # noqa: E402

DOC = "docs/PL1_MHR_S0.md"
DS = "multihoprag"
VERSION = "RK"
OUTD = ROOT / "outputs" / "pl1" / DS / VERSION
S0D = OUTD / "s0"
MHR1 = ROOT / "outputs" / "mhr1"
CARVES = {"s1sel": "dev", "s1eval": "test"}               # the carve names step 4 and PKO know, and MultiHop-RAG's split of each (UL2 13.2)
KNEE, BOUND = 0.90, 2500.0                                # M3B's constants (checked equal to m3b_contract's)
Q_DIM = 1152                                              # e3v
FIRST_EXPECT = "first.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zpl1h: {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZPL1H: {msg}")


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_arr(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wjson(p, obj):
    E.wjson(p, obj)


def rjson(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def peak_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2 ** 30, 2)
    except Exception:
        return None


# ── the package ──────────────────────────────────────────────────────────────

_PKG = {}


def package():
    """MultiHop-RAG's nodes, queries, gold positions and splits, read from the package files (zmhr's readers)."""
    if _PKG:
        return _PKG
    ZM.point_zenc()
    root = ZM.mhr_root()
    nodes, qs = ZM.nodes_of(root), ZM.queries_of(root)
    n = len(nodes)
    pos = {x["node_id"]: i for i, x in enumerate(nodes)}
    if len(pos) != n:
        stop("a node id repeats")
    bad = [q["query_id"] for q in qs for g in q["gold_node_ids"] if g not in pos]
    if bad:
        stop(f"gold node ids that are not nodes: {bad[:3]}")
    ids = [q["query_id"] for q in qs]
    if len(set(ids)) != len(ids):
        stop("a query id repeats")
    golds_raw = [np.asarray(sorted(pos[g] for g in q["gold_node_ids"]), np.int64) for q in qs]     # zmhr.cmd_first's, repeats kept
    nby = rjson(root / DS / "DATASET.json")["queries_by_split"]
    split = np.asarray(sum(([s] * int(nby[s]) for s in ZM.SPLITS), []))
    if split.size != len(qs):
        stop("the split counts do not cover the questions")
    _PKG.update(root=root, nodes=nodes, qs=qs, n=n, nq=len(qs), ids=ids, golds_raw=golds_raw,
                golds=[np.unique(g) for g in golds_raw], split=split, pos=pos)
    return _PKG


def qvec_shards(root, kind):
    return E.vec_shards(root, "e3v", DS, kind)


def gold_flat_off(golds_raw):
    flat = np.concatenate(golds_raw).astype(np.int64)
    off = np.r_[0, np.cumsum([g.size for g in golds_raw])].astype(np.int64)
    return flat, off


def populations(op, P):
    """[(carve, Population)] for s1sel (dev) and s1eval (test): query-file order, zero-gold questions excluded and counted, digest = step 4's
    digest of the kept ids (m3b_compile.Population's fields)."""
    out = []
    for carve, sp in CARVES.items():
        rows = np.flatnonzero(P["split"] == sp)
        keep = [int(j) for j in rows if P["golds"][int(j)].size]
        ids = [P["ids"][j] for j in keep]
        pop = op.m3b_compile.Population(DS, carve, ids, np.asarray(keep, np.int64), [P["golds"][j] for j in keep], int(rows.size),
                                        int(rows.size - len(keep)), op.S4.digest(ids) if hasattr(op, "S4") else _digest(ids))
        out.append((carve, pop))
    return out


def _digest(ids):
    return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()


def gate_p(pops):
    """G-P: dev 224 and test 1,130 questions in query-file order, zero-gold counted, every gold resolves, ids unique."""
    P = package()
    res = {}
    for carve, pop in pops:
        sp = CARVES[carve]
        want = int((P["split"] == sp).sum())
        ok = (pop.n_before == want and len(pop.ids) + pop.zero_gold_excluded == want and len(set(pop.ids)) == len(pop.ids)
              and all(g.size and g.max() < P["n"] for g in pop.golds) and bool(np.all(np.diff(pop.idx) > 0))
              and [P["ids"][int(j)] for j in pop.idx] == list(pop.ids))
        res[carve] = {"split": sp, "questions": len(pop.ids), "n_before": pop.n_before, "zero_gold_excluded": pop.zero_gold_excluded,
                      "digest": pop.digest, "ok": bool(ok)}
    res["verdict"] = "PASS" if all(v["ok"] for v in res.values()) and res["s1sel"]["questions"] == 224 and res["s1eval"]["questions"] == 1130 else "FAIL"
    return res


# ── S0a: the first stage and the gold table ──────────────────────────────────

def first_files():
    return {"e3v": S0D / "first_e3v.npz", "gold": S0D / "gold.npz", "record": S0D / "first.json"}


def cmd_first(a):
    import zap1 as Z
    t0 = time.time()
    P = package()
    root, n, nq = P["root"], P["n"], P["nq"]
    f = first_files()
    if f["e3v"].exists() and not a.redo:
        stop(f"{rel(f['e3v'])} exists (S0a is filed once; --redo rewrites it)")
    Qm = np.concatenate([np.load(p).astype(np.float32) for p in qvec_shards(root, "queries")])
    if Qm.shape != (nq, Q_DIM):
        stop(f"e3v query vectors {Qm.shape}, expected {(nq, Q_DIM)}")
    ids, sc, _ = E.search(Qm, qvec_shards(root, "docs"), k=n)
    if ids.shape != (nq, n) or not (np.sort(ids, axis=1) == np.arange(n)).all():
        stop("the e3v lists are not complete rankings of the 609 nodes")
    rc = root / DS / "retrieval_cache"
    with np.load(rc / "splade_top1000.npz") as z:
        s_ids = z["ids"].astype(np.int64)
    with np.load(rc / "dense_top1000.npz") as z:
        f_ids = z["ids"].astype(np.int64)
    for nm, x in (("splade", s_ids), ("frozen", f_ids)):
        if x.shape != (nq, n) or not (np.sort(x, axis=1) == np.arange(n)).all():
            stop(f"the package's {nm} lists are not complete rankings")
    flat, off = gold_flat_off(P["golds_raw"])
    sha_e = ZM.save(f["e3v"], ids=ids.astype(np.int32), scores=sc.astype(np.float16))
    sha_g = ZM.save(f["gold"], ids=np.asarray(P["ids"], dtype=str), flat=flat, off=off)
    # G-E: the top-20 equal MHR1's, the measures equal first.json's
    flat_f = rjson(MHR1 / FIRST_EXPECT)
    with np.load(MHR1 / "first" / "lists.npz") as z:
        lists = {k: z[k] for k in z.files}
    ge = {"e3v_top20_equal": bool(np.array_equal(lists["e3v"], ids[:, :20])),
          "splade_top20_equal": bool(np.array_equal(lists["splade"], s_ids[:, :20])),
          "frozen_top20_equal": bool(np.array_equal(lists["frozen"], f_ids[:, :20])),
          "gold_flat_equal": bool(np.array_equal(lists["golds_flat"], flat)), "gold_off_equal": bool(np.array_equal(lists["golds_off"], off))}
    reads = flat_f["reads"]["all"]
    meas = {}
    for arm, L in (("e3v", ids), ("splade", s_ids), ("frozen", f_ids)):
        got = {f"d{k}": Z.pct(ZM.coverage(L, P["golds_raw"], k).mean()) for k in ZM.COV_K}
        want = {k: reads[arm][k] for k in got}
        meas[arm] = {"recomputed": got, "filed": want, "equal": got == want}
    ge["measures"] = meas
    ge["verdict"] = "PASS" if all(v for k, v in ge.items() if k.endswith("_equal")) and all(m["equal"] for m in meas.values()) else "FAIL"
    rec = {"stage": "PL1 S0a", "declared_in": DOC, "dataset": DS, "nodes": n, "questions": nq,
           "by_split": {s: int((P["split"] == s).sum()) for s in ZM.SPLITS},
           "depth": int(ids.shape[1]), "lists": "complete rankings of the 609 nodes (no padding)",
           "files": {"first_e3v": {"file": rel(f["e3v"]), "sha256": sha_e}, "gold": {"file": rel(f["gold"]), "sha256": sha_g}},
           "sources": {"e3v_docs_manifest_sha256": E.sha256(E.vec_dir(root, "e3v", DS, "docs") / "manifest.json"),
                       "e3v_queries_manifest_sha256": E.sha256(E.vec_dir(root, "e3v", DS, "queries") / "manifest.json"),
                       "splade_top1000_sha256": E.sha256(rc / "splade_top1000.npz"), "frozen_top1000_sha256": E.sha256(rc / "dense_top1000.npz"),
                       "mhr1_first_json_sha256": E.sha256(MHR1 / FIRST_EXPECT), "mhr1_lists_sha256": E.sha256(MHR1 / "first" / "lists.npz"),
                       "dataset_json_sha256": E.sha256(root / DS / "DATASET.json")},
           "gate_GE": ge, "seconds": round(time.time() - t0, 1), "utc": utc(), "threads": E.THREADS, "script_sha256": script_sha256()}
    wjson(f["record"], rec)
    log(f"S0a: G-E {ge['verdict']} ({json.dumps({k: v['recomputed']['d5'] for k, v in meas.items()})}) in {rec['seconds']} s")
    return 0 if ge["verdict"] == "PASS" else 3


def read_first():
    f = first_files()
    rec = rjson(f["record"])
    if rec["gate_GE"]["verdict"] != "PASS":
        stop("S0a's G-E is not PASS")
    if E.sha256(f["e3v"]) != rec["files"]["first_e3v"]["sha256"] or E.sha256(f["gold"]) != rec["files"]["gold"]["sha256"]:
        stop("S0a's files are not the ones its record names")
    with np.load(f["e3v"]) as z:
        e3v = (z["ids"].astype(np.int32), z["scores"].astype(np.float16))
    with np.load(f["gold"]) as z:
        gold_ids, flat, off = [str(x) for x in z["ids"]], z["flat"].astype(np.int64), z["off"].astype(np.int64)
    return rec, e3v, {q: np.unique(flat[off[j]:off[j + 1]]) for j, q in enumerate(gold_ids)}


# ── S0b: the layout ──────────────────────────────────────────────────────────

def f0t_in_process(P):
    """F0T's pairs with D83's weights, in process: zul1.build_entries and forms_from_entries over the nodes' titles and texts (the build
    zmhr.cmd_links ran, here with the weights kept), the two tables joined by d83stores.form_pairs (zpl1m.f0t_triple's rule)."""
    import d83stores as DSt
    import zul1 as UL
    nodes, n = P["nodes"], P["n"]
    titles, texts = [x["title"] for x in nodes], [x["text"] for x in nodes]
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


def sym_csr(keys, n):
    """The symmetric CSR (rows sorted, no self loop) of undirected pair keys (zu2c.csr_store's input)."""
    a, b = keys // n, keys % n
    h, t = np.concatenate([a, b]), np.concatenate([b, a])
    o = np.lexsort((t, h))
    h, t = h[o], t[o]
    ip = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(h, minlength=n), out=ip[1:])
    return ip, t.astype(np.int32)


def tab_of(z):
    return {"ner": {k[4:]: z[k] for k in z.files if k.startswith("ner_")}, "knn": {k[4:]: z[k] for k in z.files if k.startswith("knn_")},
            "structural_keys": z["struct_key"], "structural_label": z["struct_label"]}


def cmd_layout(a):
    import d83stores as DSt
    t0 = time.time()
    P = package()
    n = P["n"]
    od = OUTD
    if (od / "layout.json").exists() and not a.redo:
        stop(f"{rel(od / 'layout.json')} exists (the layout is filed once; --redo rewrites it)")
    lk = rjson(MHR1 / "links.json")
    W2 = np.load(MHR1 / "links" / "rule" / "W2_S.npz")["keys"].astype(np.int64)
    f0t_file = np.load(MHR1 / "links" / "rule" / "F0T.npz")["keys"].astype(np.int64)
    f0_file = np.load(MHR1 / "links" / "rule" / "F0.npz")["keys"].astype(np.int64)
    with np.load(MHR1 / "links" / "e3v" / "K.npz") as z:
        K = PM.knn_triple(z["src"], z["dst"], z["weight"], n)
    f0t, f0_keys, f0t_keys = f0t_in_process(P)
    kkeys = K[0].astype(np.int64) * n + K[1].astype(np.int64)
    gf = {"F0T_keys_equal": bool(np.array_equal(f0t_keys, f0t_file)), "CN_all_keys_equal": bool(np.array_equal(f0_keys, f0_file)),
          "F0T_pairs": int(f0t_keys.size), "CN_all_pairs": int(f0_keys.size),
          "weights_positive": bool((f0t[2] > 0).all()), "triple_keys_equal": bool(np.array_equal(f0t[0].astype(np.int64) * n + f0t[1], f0t_keys))}
    gf["verdict"] = "PASS" if gf["F0T_keys_equal"] and gf["CN_all_keys_equal"] and gf["weights_positive"] and gf["triple_keys_equal"] else "FAIL"
    if gf["verdict"] != "PASS":
        wjson(od / "layout.json", {"stage": "PL1 S0b", "gate_GF": gf, "verdict": "FAIL"})
        stop(f"G-F FAIL {gf}")
    tab, counts = PM.tables(n, VERSION, None, W2, f0t, K, None)
    exp = PM.expected_union(n, VERSION, None, W2, f0t_keys, kkeys, None)
    gl = PM.gates(tab, counts, n, VERSION, exp)
    # the counts equal MHR1's links.json (W2_S 2, R, K store, RK) and each store's row order equals FamilyCSR's on the same pairs
    arm = lk["arms"]["e3v"]
    nm = {"W2_S_pairs": (counts["W2_S_pairs"], int(lk["W2_S"]["pairs"])), "R": (int(np.union1d(W2, f0t_keys).size), int(arm["R"])),
          "K_store": (counts["knn_pairs_out"], int(arm["K_store"])), "RK": (counts["union"], int(arm["RK"]))}
    gl["counts_vs_links_json"] = {k: {"here": v[0], "mhr1": v[1], "equal": v[0] == v[1]} for k, v in nm.items()}
    op = _plain_opened()
    M3A = op.m3a
    stores = PM.to_stores(tab, n)
    order = {}
    for fam, st, w in (("ner", stores[0], tab["ner"]["order_key"]), ("knn", stores[1], tab["knn"]["weight"])):
        g = SimpleNamespace(n_nodes=n, src=tab["ner" if fam == "ner" else "knn"]["a"], dst=tab["ner" if fam == "ner" else "knn"]["b"], weight=w)
        c = M3A.FamilyCSR(g, fam)
        order[fam] = bool(np.array_equal(c.indptr, np.asarray(st.indptr, np.int64)) and np.array_equal(np.asarray(c.col, np.int64), np.asarray(st.col, np.int64)))
    gl["row_order_equals_FamilyCSR"] = order
    if not (all(v["equal"] for v in gl["counts_vs_links_json"].values()) and all(order.values())):
        gl["verdict"] = "FAIL"
    flat = {f"ner_{k}": x for k, x in tab["ner"].items()}
    flat.update({f"knn_{k}": x for k, x in tab["knn"].items()})
    flat.update(struct_key=tab["structural_keys"], struct_label=tab["structural_label"])
    sha = PM.L2.save_npz(od / "layout.npz", **flat) if gl["verdict"] == "PASS" else None
    rec = {"stage": "PL1 S0b", "declared_in": DOC, "dataset": DS, "version": VERSION, "layout_kind": "B0", "nodes": n,
           "nodes_sha256": E.sha256(P["root"] / DS / "nodes.jsonl"),
           "inputs": {"links_json_sha256": E.sha256(MHR1 / "links.json"),
                      "W2_S_sha256": E.sha256(MHR1 / "links" / "rule" / "W2_S.npz"), "F0T_sha256": E.sha256(MHR1 / "links" / "rule" / "F0T.npz"),
                      "F0_sha256": E.sha256(MHR1 / "links" / "rule" / "F0.npz"), "K_e3v_sha256": E.sha256(MHR1 / "links" / "e3v" / "K.npz")},
           "note": "B0 only: no carried forms (UL2's call for MultiHop-RAG is not filed); the layout and the pools are rebuilt once as a new declared block when it is",
           "counts": counts, "gate_GF": gf, "gate_GL": gl, "layout_npz_sha256": sha, "seconds": round(time.time() - t0, 1), "utc": utc(),
           "script_sha256": script_sha256()}
    wjson(od / "layout.json", rec)
    log(f"S0b: G-F {gf['verdict']}, G-L {gl['verdict']} {json.dumps(counts)[:300]}")
    return 0 if gl["verdict"] == "PASS" else 3


def read_layout():
    rec = rjson(OUTD / "layout.json")
    if rec["gate_GL"]["verdict"] != "PASS" or rec["gate_GF"]["verdict"] != "PASS":
        stop("the layout's G-F or G-L is not PASS")
    if E.sha256(OUTD / "layout.npz") != rec["layout_npz_sha256"]:
        stop("layout.npz is not the file its record names")
    with np.load(OUTD / "layout.npz") as z:
        return tab_of(z), rec


# ── S0c and G-K: the knee grid ───────────────────────────────────────────────

_OP = {}


def _plain_opened():
    """step 4's opened package, unpatched: the headroom module, the two configs and the contract module."""
    if "op" not in _OP:
        import step4_pool_reach as S4
        _OP["op"] = S4.Opened()
        _OP["op"].S4 = S4
    return _OP["op"]


def grid_rows(op, dense, splade, golds, n, fam, name):
    """m3a_headroom.run_dataset's per-population loop on arrays, in its order and with its calls (prefix_rows, rrf_fused_rows,
    stable_union_rows, FamilyCSR's expand, pool_union_rows, cell): the 7 retrieval pools and the graph regimes x settings x bases."""
    M3A, cfg_h = op.m3a, op.cfg_h
    rp = cfg_h["retrieval_pools"]
    ks = tuple(int(k) for k in rp["ks"])
    constant = int(rp["equal_rrf"]["constant"])
    budgets = [int(b) for b in rp["equal_rrf"]["budgets"]]
    golds_r = M3A.golds_ragged(golds)
    label = {"dataset": name, "population": "eval"}
    d200, s200 = dense[:, :200], splade[:, :200]
    named = {"dense_top200": M3A.prefix_rows(d200, 200), "splade_top200": M3A.prefix_rows(s200, 200),
             "frozen_union": [np.union1d(x, y) for x, y in zip(d200, s200)]}
    fused = M3A.rrf_fused_rows(d200, s200, constant)
    for b in budgets:
        named[f"equal_rrf_budget_{b}"] = [r[:b] for r in fused]
    rows = []
    for pool_name, prows in named.items():
        row, _present = M3A.cell(prows, golds_r, n, ks, **label, regime="RETRIEVAL", setting="-", base_pool="-", pool=pool_name)
        rows.append(row)
    seeds = M3A.stable_union_rows(dense[:, :5], splade[:, :5])
    gr = cfg_h["graph_regimes"]
    regimes = {k: v for k, v in gr.items() if k.isupper() and isinstance(v, list) and v}
    for regime, fam_names in regimes.items():
        fams = [fam[f] for f in fam_names]
        for setting in gr["expansion_settings"]:
            expansions = [M3A.expand(s, fams, setting) for s in seeds]
            for base_name in gr["base_pools_for_exposure"]:
                union = M3A.pool_union_rows(named[base_name], expansions)
                row, _present = M3A.cell(union, golds_r, n, ks, **label, regime=regime, setting=setting["name"], base_pool=base_name,
                                         pool=f"{base_name}+{regime}:{setting['name']}")
                rows.append(row)
            del expansions
    return rows


def cells_of_rows(op, rows, source):
    """m3b_contract.committed_cells' cell dicts, from rows."""
    KEEP = op.m3b_contract.KEEP
    return [{"pool": r["pool"], "regime": r["regime"], "setting": r["setting"], "base_pool": r["base_pool"], "source": source,
             **{k: r[k] for k in KEEP if k in r}} for r in rows if "fraction_of_attainable@5" in r]


def layout_families(op, tab, n):
    M3A = op.m3a
    sa, sb = struct_arrays(tab["structural_keys"], n)
    gs = SimpleNamespace(n_nodes=n, src=sa, dst=sb, weight=None)
    gn = SimpleNamespace(n_nodes=n, src=tab["ner"]["a"], dst=tab["ner"]["b"], weight=tab["ner"]["order_key"])
    gk = SimpleNamespace(n_nodes=n, src=tab["knn"]["a"], dst=tab["knn"]["b"], weight=tab["knn"]["weight"])
    return {"structural": M3A.FamilyCSR(gs, "structural"), "ner": M3A.FamilyCSR(gn, "ner"), "knn": M3A.FamilyCSR(gk, "knn")}


def cmd_gk(a):
    """G-K: the knee code reproduces MuSiQue's committed grid and construction (the same loop and finish_record on MuSiQue's own data)."""
    t0 = time.time()
    op = _plain_opened()
    M3A, MC, cfg_h = op.m3a, op.m3b_contract, op.cfg_h
    if (MC.KNEE, MC.BOUND) != (KNEE, BOUND):
        stop("m3b_contract's knee and bound are not the declared 0.90 and 2500")
    name = a.ds
    ds = op.canonical.Dataset(name, root=str(op.served))
    n = int(ds.n_nodes)
    positions = M3A.node_position_map(ds)
    split = cfg_h["populations"]["eval_splits"][name]
    idx, rows_ = M3A.population_rows(ds, split, cfg_h)
    golds = M3A.resolve_gold(rows_, positions, name)
    zero = np.asarray([g.size == 0 for g in golds])
    idx, golds = idx[~zero], [g for g, z in zip(golds, zero) if not z]
    dense_ids, _ = ds.cache("dense")
    dense = dense_ids[idx].astype(np.int64)
    del dense_ids
    splade_ids, _ = ds.cache("splade")
    splade = splade_ids[idx].astype(np.int64)
    del splade_ids
    log(f"G-K {name}: {len(golds)} questions read ({peak_gb()} GB)")
    fam = {f: M3A.FamilyCSR(ds.graph(f), f) for f in ("structural", "ner", "knn")}
    log(f"G-K {name}: the three families' CSRs built ({peak_gb()} GB)")
    rows = grid_rows(op, dense, splade, golds, n, fam, name)
    mine = cells_of_rows(op, rows, f"committed:outputs/m3a/headroom/{name}.json")
    filed = MC.committed_cells(name)
    key = lambda c: (c["pool"], c["regime"], c["setting"], c["base_pool"])  # noqa: E731
    fm = {key(c): c for c in filed}
    mm = {key(c): c for c in mine}
    worst, miss = 0.0, [k for k in fm if k not in mm] + [k for k in mm if k not in fm]
    per = []
    for k, c in fm.items():
        if k in mm:
            d = max((abs(float(mm[k][x]) - float(c[x])) for x in c if x not in ("pool", "regime", "setting", "base_pool", "source")), default=0.0)
            worst = max(worst, d)
            per.append({"pool": k[0], "candidates_mean": [mm[k]["candidates_mean"], c["candidates_mean"]],
                        "fraction_of_attainable@5": [mm[k]["fraction_of_attainable@5"], c["fraction_of_attainable@5"]], "max_abs_diff": d})
    cfg = op.cfg
    rec_mine = MC.finish_record({"dataset": name}, mine, name, cfg, cfg_h)
    filed_rec = rjson(ROOT / "outputs" / "m3b" / "contract" / f"{name}.json")
    same_construction = rec_mine["construction"] == filed_rec["construction"]
    verdict = "PASS" if (not miss and len(fm) == len(mm) and worst <= 1e-9 and same_construction) else "FAIL"
    res = {"stage": "PL1 G-K", "declared_in": DOC, "dataset": name, "questions": len(golds), "cells_filed": len(fm), "cells_here": len(mm),
           "missing_or_extra": [list(k) for k in miss], "max_abs_diff": worst, "tolerance": 1e-9,
           "construction_here": rec_mine["construction"], "construction_filed": filed_rec["construction"], "construction_equal": same_construction,
           "per_cell": sorted(per, key=lambda r: r["candidates_mean"][1]), "verdict": verdict, "seconds": round(time.time() - t0, 1),
           "peak_gb": peak_gb(), "utc": utc(), "script_sha256": script_sha256()}
    wjson(S0D / f"gk_{name}.json", res)
    log(f"G-K {name}: {verdict}, {len(mm)}/{len(fm)} cells, max |diff| {worst:.3g}, construction equal {same_construction} ({res['seconds']} s)")
    return 0 if verdict == "PASS" else 3


def gate_gk_passed():
    p = S0D / "gk_musique.json"
    if not p.is_file() or rjson(p)["verdict"] != "PASS":
        stop("G-K (the knee code on MuSiQue) is not PASS: run `zpl1h.py gk --ds musique` first")
    return rjson(p)


def cmd_knee(a):
    """S0c: M3B's knee rule on dev, over the 31 cells of the committed grid computed on MultiHop-RAG's own B0 stores. The e3v read is the
    construction; the frozen read is filed beside and decides nothing."""
    t0 = time.time()
    gk = gate_gk_passed()
    P = package()
    n = P["n"]
    op = _plain_opened()
    MC, cfg, cfg_h = op.m3b_contract, op.cfg, op.cfg_h
    first, (e3v_ids, _sc), _g = read_first()
    tab, lay = read_layout()
    fam = layout_families(op, tab, n)
    pops = populations(op, P)
    gp = gate_p(pops)
    if gp["verdict"] != "PASS":
        stop(f"G-P {gp}")
    dev = dict(pops)["s1sel"]
    rc = P["root"] / DS / "retrieval_cache"
    with np.load(rc / "splade_top1000.npz") as z:
        splade = z["ids"].astype(np.int64)[dev.idx]
    with np.load(rc / "dense_top1000.npz") as z:
        frozen = z["ids"].astype(np.int64)[dev.idx]
    out = {}
    for arm, dense in (("e3v", e3v_ids.astype(np.int64)[dev.idx]), ("frozen", frozen)):
        rows = grid_rows(op, dense, splade, dev.golds, n, fam, DS)
        cells = cells_of_rows(op, rows, f"computed:{rel(S0D / f'knee_{arm}.json')}")
        rec = {"stage": "PL1 S0c" if arm == "e3v" else "PL1 S0c (frozen, beside)", "declared_in": DOC, "dataset": DS, "eval_split": "dev",
               "arm": arm, "knee": KNEE, "bound": BOUND, "carve": "s1sel", "questions": len(dev.ids), "population_digest": dev.digest,
               "layout_npz_sha256": lay["layout_npz_sha256"], "first_e3v_sha256": first["files"]["first_e3v"]["sha256"],
               "gate_GK": {"file": rel(S0D / "gk_musique.json"), "sha256": E.sha256(S0D / "gk_musique.json"), "verdict": gk["verdict"]},
               "utc": utc(), "script_sha256": script_sha256(), "seconds": round(time.time() - t0, 1)}
        rec = MC.finish_record(rec, cells, DS, cfg, cfg_h)
        wjson(S0D / f"knee_{arm}.json", rec)
        out[arm] = rec
        c = rec["chosen"]
        log(f"S0c {arm}: {rec['reading']} -> {c['pool']} (candidates {c['candidates_mean']:.1f}, fraction {c['fraction_of_attainable@5']:.4f}); "
            f"{rec['verdict']}")
    same = out["e3v"]["construction"] == out["frozen"]["construction"]
    log(f"S0c: e3v and frozen constructions {'agree' if same else 'DIFFER'}")
    return 0


# ── S0d: the record ──────────────────────────────────────────────────────────

def null_types(entries):
    """2g's null types in the loaders' shape (zpl1w.null_type_arrays): zero centres, the null class on every entry."""
    import zu2m_a_types as AT
    k = int(AT.K_EM)
    return PW.null_type_arrays(int(entries), Q_DIM, k), k


def cmd_record(a):
    import zpko as PK
    import p1e_anchor_kind as K
    t0 = time.time()
    if (S0D / "record.json").exists() and not a.redo:
        stop(f"{rel(S0D / 'record.json')} exists (the record is filed once; --redo rewrites it)")
    knee = rjson(S0D / "knee_e3v.json")
    tab, lay = read_layout()
    n = package()["n"]
    construction = knee["construction"]
    import zu2m as M2
    bases = [bp for _bs, bp in M2.BASES]
    in_bases = construction["base_pool"] in bases
    cells = own_k = own_b = None
    if in_bases:
        cells, own_k = M2.cells_of(construction)
        own_b = bases.index(construction["base_pool"])
    Q, S, A = PK.Q, PK.S, PK.A
    cw = Q.carried_walk()
    beta = float(cw["beta"])
    walk = f"{cw['a_cfg']} x LW@{beta:g}"
    entries = int(2 * tab["structural_keys"].size)
    arr, k_a = null_types(entries)
    plan = PK.pick_plan()
    consts = {"alpha": float(K.E.ALPHA), "eps": float(K.S4.EPS), "nmax": int(K.D.NMAX), "G": int(cw["G"]), "pick": cw["pick"], "beta": beta, "walk": walk}
    v = PK.u2s_verdict()
    gr = {}
    for d in ("musique", "squad", "hotpotqa", "2wiki"):
        rec_d, _npz = PK.u2s_record(d, v)
        gr[d] = {k: (rec_d[k] == consts[k] if k not in ("alpha", "eps", "beta") else float(rec_d[k]) == consts[k]) for k in consts}
    gr_ok = all(all(x.values()) for x in gr.values()) and cw["pick"] == PK.u2s_record("musique", v)[0]["pick"]
    rec = {"stage": "PL1 S0d", "declared_in": DOC, "dataset": DS, "version": VERSION, "pick": PK.PICK,
           "pick_plan": {"r": plan.r, "o": plan.o, "b": plan.b, "kind": plan.kind, "prm": list(plan.prm), "tau": plan.tau},
           "construction": construction, "construction_from": {"file": rel(S0D / "knee_e3v.json"), "sha256": E.sha256(S0D / "knee_e3v.json")},
           "base_in_u2m_bases": bool(in_bases), "cells": cells, "own_cell": None if own_k is None else int(own_k),
           "cell_base": None if own_b is None else int(own_b),
           "graph_detail": {"graph": "MultiHop-RAG B0 (W2_S typed store, F0T ner store, K_e3v knn store after the S > R > K dedup)",
                            "layout": rel(OUTD / "layout.npz"), "layout_npz_sha256": lay["layout_npz_sha256"]},
           "walk": walk, "carried": consts, "types": {"weights": "off (2g)", "k": k_a, "entries": entries, "em_entries_non_null": int((arr["cls"] != k_a).sum()),
                                                      "score_entries_non_null": 0, "mu_sha256": sha_arr(arr["mu"]), "cls_sha256": sha_arr(arr["cls"])},
           "gate_GR": {"against": "the U2s records of the four passage datasets", "per_dataset": gr, "verdict": "PASS" if gr_ok else "FAIL"},
           "no_u2s_arrays": "U2s's stats stage is not re-run: the pick is carried (U2s's verdict, pick O) and the pool rule needs the own cell only",
           "utc": utc(), "seconds": round(time.time() - t0, 1), "script_sha256": script_sha256()}
    if not in_bases:
        rec["flag"] = ("the knee's base pool is outside zu2m.BASES (rrf50, rrf200, dense200): U2m's own_name and cells_of do not apply; "
                       "the pool rule reads the base through base_rows and needs neither")
    wjson(S0D / "record.json", rec)
    log(f"S0d: G-R {rec['gate_GR']['verdict']}, construction {json.dumps(construction)[:200]}, base in BASES: {in_bases}")
    return 0 if gr_ok else 3


def read_record():
    rec = rjson(S0D / "record.json")
    if rec["gate_GR"]["verdict"] != "PASS":
        stop("the record's G-R is not PASS")
    return rec


# ── the world: the six-keyed pieces patched for MultiHop-RAG ─────────────────

def relaxed_open_carve(ZP):
    """zu2d_pools.open_carve compiled from its own source with its one depth condition read as min(1000, n). Returns (function, old source,
    new source)."""
    src = inspect.getsource(ZP.open_carve)
    old = "    if prep.dense_ids.shape[1] < 1000 or prep.splade_ids.shape[1] < 1000:\n"
    new = "    if prep.dense_ids.shape[1] < min(1000, n) or prep.splade_ids.shape[1] < min(1000, n):\n"
    if src.count(old) != 1:
        stop("zu2d_pools.open_carve's depth check is not the text this module was written against")
    src2 = src.replace(old, new)
    ns = {}
    exec(compile(src2, "<zpl1h: zu2d_pools.open_carve with the depth check read as min(1000, n)>", "exec"), ZP.__dict__, ns)
    return ns["open_carve"], src, src2


def source_diff(a, b):
    return [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]


@contextlib.contextmanager
def world(M, spec, stats):
    """The swaps of the pools passes (every patch put back on exit). spec: tab (the layout's tables), construction, pops [(carve, pop)],
    lists (the e3v lists) and the record. M: zb1p.mods."""
    P = package()
    n = P["n"]
    Pt = PW.Patches()
    sv = stats.setdefault("served", {k: 0 for k in ("canonical", "embeddings", "cache", "opened", "open_dataset", "populations", "union_pools",
                                                    "family_stores", "graph_csr", "stores", "open_carve")})
    S4, E_, ZP, U2C = M.S4, M.E, M.ZP, M.U2C
    import step4b_pool_reach as S4B
    G = M.K.G
    R2 = G.R2
    C4 = E_.C4
    st = {"canon": None, "restore": []}
    e3v_ids, e3v_sc = spec.lists
    cache_d = {"ids": e3v_ids, "sc": e3v_sc}
    ip, col = sym_csr(spec.tab["structural_keys"], n)
    desc = {"graph": "MultiHop-RAG W2_S (RK's typed store)", "pairs": int(spec.tab["structural_keys"].size), "layout_npz_sha256": spec.layout_sha}
    made = {}

    def stores():
        if not made:
            ner, knn = PM.to_stores(spec.tab, n)
            made["s"] = {"structural": U2C.csr_store(n, ip, col), "ner": ner, "knn": knn}
        return made["s"]

    def patch_canonical(canonical):
        if st["canon"] is canonical:
            return
        if st["canon"] is not None:
            stop("two package loaders in one process")
        st["canon"] = canonical
        canonical.DATASETS.append(DS)
        canonical.SPLITS[DS] = list(ZM.SPLITS)
        st["restore"].append(lambda: (canonical.DATASETS.remove(DS), canonical.SPLITS.pop(DS)))
        sv["canonical"] += 1
        D = canonical.Dataset
        o_emb, o_cache = D.embeddings, D.cache

        def embeddings(self, model, kind="docs"):
            if self.name == DS and model == "dense":
                sv["embeddings"] += 1
                return canonical.Embeddings(str(E.vec_dir(None, "e3v", DS, kind)), "dense")
            return o_emb(self, model, kind)

        def cache(self, model):
            if self.name == DS and model == "dense":
                sv["cache"] += 1
                return cache_d["ids"], cache_d["sc"]
            return o_cache(self, model)

        Pt.set(D, embeddings=embeddings, cache=cache)

    base_cls = S4.Opened

    class OpenedMHR(base_cls):
        def __init__(self):
            super().__init__()
            patch_canonical(self.canonical)
            self.served = ZM.PKG                         # Dataset(name, root=op.served) reads MultiHop-RAG's package
            self.construction = dict(self.construction)
            self.construction[DS] = spec.construction
            sv["opened"] += 1

    Pt.set(S4, Opened=OpenedMHR)

    def open_dataset(op, name):
        if name != DS:
            stop(f"open_dataset({name}) under MultiHop-RAG's world")
        sv["open_dataset"] += 1
        ds = op.canonical.Dataset(name, root=str(ZM.PKG))
        c = op.construction[name]
        regime = c["regime"]
        fams = op.m3b_compile.regime_families(op.cfg_h, regime) if regime != "RETRIEVAL" else []
        stt = {f: stores()[f] for f in fams}
        union = S4B.RegimeUnion(stt, fams, int(stt[fams[0]].n_nodes) if fams else int(ds.n_nodes))
        return ds, c, fams, stt, union

    def populations_(op, S4_, ds, name):
        if name != DS:
            stop(f"populations({name}) under MultiHop-RAG's world")
        sv["populations"] += 1
        return list(spec.pops)

    def union_pools(name_, carve_, pop, prep, bases):
        """B1P's form (zpl1w's): B = base united with the seeds, I = the prepared pools."""
        if name_ != DS:
            stop(f"union_pools({name_}) under MultiHop-RAG's world")
        sv["union_pools"] += 1
        bs = [np.union1d(np.asarray(b, np.int64), np.asarray(s, np.int64)) for b, s in zip(bases, prep.seeds)]
        return None, None, bs, [np.asarray(p, np.int64) for p in prep.pools], None

    def family_stores(op, name, union):
        if name != DS:
            stop(f"family_stores({name}) under MultiHop-RAG's world")
        sv["family_stores"] += 1
        s = stores()
        return {f: s[f] for f in S4.FAMILIES}, n

    o_graph_csr, o_u2e_graph = U2C.graph_csr, ZP.u2e_graph

    def graph_csr(name, graph, n_):
        if name != DS:
            return o_graph_csr(name, graph, n_)
        if int(n_) != n:
            stop(f"graph_csr: {n_} nodes, not {n}")
        sv["graph_csr"] += 1
        return ip, col, desc

    def u2e_graph(name):
        return desc if name == DS else o_u2e_graph(name)

    mp_mod = S4.m3b_pools
    o_store = mp_mod.load_or_build_store

    def load_store(ds_, family, cache_dir):
        if ds_.name != DS or family not in ("structural", "ner", "knn"):
            return o_store(ds_, family, cache_dir)
        sv["stores"] += 1
        return stores()[family]

    Pt.set(E_, open_dataset=open_dataset, union_pools=union_pools)
    Pt.set(C4, populations=populations_)
    Pt.set(R2, family_stores=family_stores)
    Pt.set(U2C, graph_csr=graph_csr)
    Pt.set(ZP, u2e_graph=u2e_graph)
    Pt.set(mp_mod, load_or_build_store=load_store)
    fn, src_old, src_new = relaxed_open_carve(ZP)

    def open_short(build, name, carve, threads):
        sv["open_carve"] += 1
        return fn(build, name, carve, threads)

    try:
        yield SimpleNamespace(served=sv, open_short=open_short, desc=desc, ip=ip, col=col, made=made,
                              source_diff=source_diff(src_old, src_new))
    finally:
        Pt.restore()
        while st["restore"]:
            st["restore"].pop()()


def build_ctx(M, cx, W):
    """open_pko's context for a passage dataset, with 2g's null types in memory and no U2s record: every constant is the module's own
    (G-R compared them with the four U2s records at S0d)."""
    PK = M.PK
    Q, A, P, AT = PK.Q, PK.A, PK.P, PK.AT
    plan = PK.pick_plan()
    cw = Q.carried_walk()
    beta = float(cw["beta"])
    entries = int(np.asarray(cx.adj[0][0])[-1])
    if entries != int(W.col.size):
        stop(f"the structural adjacency holds {entries} entries, the layout's W2_S {int(W.col.size)}")
    k_a = int(AT.K_EM)
    dim = int(np.asarray(cx.nodes.read(np.zeros(1, np.int64))).shape[1])
    arr = PW.null_type_arrays(entries, dim, k_a)
    mu = np.asarray(arr["mu"], np.float64)
    cls = arr["cls"]
    cls_s = np.full(entries, P.NULL, np.int8)
    ty = (np.ascontiguousarray(P.pair_types(cls, cls_s, k_a), np.int16),) + A.no_types()[1:]
    ws = P.weights(np.zeros(P.NT), beta)
    walk = f"{cw['a_cfg']} x LW@{beta:g}"
    rec = read_record()
    if walk != rec["walk"]:
        stop(f"the walk {walk} is not the record's {rec['walk']}")
    alpha, eps, bmax = float(cx.K.E.ALPHA), float(cx.S4.EPS), int(cx.D.NMAX)
    for k, v in (("alpha", alpha), ("eps", eps), ("nmax", bmax)):
        if rec["carried"][k] != v:
            stop(f"{k} is {v}, the record's {rec['carried'][k]}")
    # G-W (i): the context's types are the null type on every entry and the weights are exactly 1
    q0 = np.zeros(dim)
    q0[0] = 1.0
    wa = A.omega(q0, mu, Q.A_BETA, True)[0]
    pw = P.pair_weights(wa, ws)
    ok = (not mu.any() and bool((cls == k_a).all()) and bool((cls_s == P.NULL).all()) and np.unique(ty[0]).size == 1 and bool((wa == 1.0).all())
          and bool((pw == 1.0).all()))
    if not ok:
        stop("G-W (i): the context's types are not the null type with weights of exactly 1")
    construction = cx.op.construction[DS]
    base_own = cx.op.m3b_compile.base_rows({"base_pool": construction["base_pool"]}, cx.prep.dense_ids, cx.prep.splade_ids, cx.op.m3a,
                                           cx.op.m3b_contract, cx.constant)
    recf = S0D / "record.json"
    ctx = SimpleNamespace(
        name=DS, carve=cx.carve, rec=rec, plan=plan, cw=cw, passage=True, mu=mu, ty=ty, ws=ws, alpha=alpha, eps=eps, bmax=bmax, walk=walk,
        base_own=base_own, desc=cx.desc, u2s=lambda: None,
        u2s_names={"verdict": None, "record": rel(recf), "record_sha256": E.sha256(recf), "file_sha256": None, "freeze_RECORD_SHA256": None,
                   "note": "MultiHop-RAG has no U2s stats stage (docs/PL1_MHR_S0.md section 3, S0d)"},
        types={"em_types_sha256": sha_arr(cls), "score_types_sha256": sha_arr(cls_s), "weights": "off (2g)", "entries": entries,
               "em_entries_non_null": int((cls != k_a).sum()), "score_entries_non_null": 0})
    return ctx


def pools_run(M, spec, carve, base, shard, limit, threads, plain, stats):
    """One pools pass on MultiHop-RAG's world: zpko.pools_pass with the opener below and, unless plain, zpl1w.make_rule's places."""
    PK = M.PK
    R = PK.roots(base)
    store = {}
    gold = read_gold_table()
    rule = None if plain else PW.make_rule(PK, store, gold)
    with world(M, spec, stats) as W:
        def opener(name, carve_, threads_):
            if name != DS:
                stop(f"the opener serves {DS} only")
            cx = W.open_short("new", name, carve_, threads_)
            return cx, build_ctx(M, cx, W)

        out = PK.pools_pass(DS, carve, threads=int(threads), limit=int(limit or 0), shard=shard, R=R, opener=opener, rule=rule)
        stats["source_diff"] = W.source_diff
    sv = stats["served"]
    need = ["canonical", "embeddings", "cache", "opened", "open_dataset", "populations", "union_pools", "family_stores", "graph_csr", "open_carve"]
    if spec.construction["regime"] != "RETRIEVAL":      # a RETRIEVAL construction opens no regime store (step4e.open_dataset's own branch)
        need.append("stores")
    miss = [k for k in need if not sv[k]]
    if miss:
        stop(f"the swaps {miss} never served (the world was not in force)")
    return out, store, R


def read_gold_table():
    _rec, _e, g = read_first()
    return g


def make_spec(carve_only=None):
    P = package()
    op = _plain_opened()
    tab, lay = read_layout()
    rec = read_record()
    first, lists, _g = read_first()
    pops = populations(op, P)
    gp = gate_p(pops)
    if gp["verdict"] != "PASS":
        stop(f"G-P {gp}")
    return SimpleNamespace(tab=tab, layout_sha=lay["layout_npz_sha256"], construction=rec["construction"], pops=pops, lists=lists, gate_p=gp,
                           first=first, layout=lay, record=rec)


def places_dir(base):
    return Path(base) / "places"


def cmd_pools(a):
    t0 = time.time()
    carve = a.carve
    if carve not in CARVES:
        stop(f"the carves are {tuple(CARVES)}")
    i, k = int(a.shard[0]), int(a.shard[1])
    base = (OUTD / "dev") if carve == "s1sel" else OUTD
    if a.tag:
        base = OUTD / "gn" / a.tag
    if carve == "s1eval" and not a.tag and not a.limit:
        gw = S0D / "gates_dev.json"
        if not gw.is_file() or rjson(gw)["verdict"] != "PASS":
            stop("S0f (the test pools) waits for S0e's gates: run `zpl1h.py gates` and see PASS first")
    M = PW.mods_of(False)
    spec = make_spec()
    stats = {"carve": carve, "dataset": DS}
    out, store, R = pools_run(M, spec, carve, base, (i, k), a.limit, a.threads, a.plain, stats)
    PK = M.PK
    st = PK.stem(DS, carve, (i, k))
    rec = {"stage": "PL1 S0e" if carve == "s1sel" else "PL1 S0f", "declared_in": DOC, "dataset": DS, "version": VERSION, "carve": carve,
           "shard": [i, k], "limit": int(a.limit or 0), "plain": bool(a.plain), "threads": int(a.threads), "weights": "off (2g)",
           "result": out, "stats": stats, "gate_GP": spec.gate_p, "seconds": round(time.time() - t0, 1), "script_sha256": script_sha256(),
           "peak_gb": peak_gb(), "utc": utc()}
    if a.limit:
        pos = np.concatenate([s["pos"] for s in store.values()] or [np.zeros(0, np.int32)])
        keep_ = np.asarray([s["keep"] for s in store.values()], np.int32)
        gp = np.concatenate([[0], np.cumsum([s["g"].size for s in store.values()])]).astype(np.int64)
        cl = PW.classes(pos, keep_, gp) if store else np.zeros(0, np.int8)
        log(f"{carve} {i}of{k}: {out['questions']} questions in {out['seconds']:.1f} s ({out['seconds'] / max(1, out['questions']):.3f} s per question, "
            f"open {out['open']:.1f} s); nothing written; places {len(store)} ({pos.size} golds)")
        return 0
    if not a.plain:
        ids = [str(x) for x in np.load(R.parts / f"{st}.npz", allow_pickle=False)["ids"]]
        sha, arr = PW.write_places(places_dir(base) / f"{st}.npz", ids, store)
        rec["places"] = {"file": rel(places_dir(base) / f"{st}.npz"), "sha256": sha, "questions": len(ids), "golds": int(arr["g_node"].size),
                         "classes": {n_: int((PW.classes(arr["pos"], arr["keep"], arr["g_ptr"]) == c).sum())
                                     for c, n_ in enumerate(("IN_B", "KEPT", "BEYOND", "OUTSIDE"))}}
    wjson(R.parts / f"{st}.view.json", rec)
    log(f"{carve} {i}of{k}: done in {rec['seconds']} s; {rec.get('places', {}).get('classes')}; record {rel(R.parts / f'{st}.view.json')}")
    return 0


# ── G-W (ii) and G-N ─────────────────────────────────────────────────────────

def part_files(base, carve, k):
    PK = _pk()
    R = PK.roots(base)
    arrs, kts = [], []
    for i in range(k):
        st = PK.stem(DS, carve, (i, k))
        with np.load(R.parts / f"{st}.npz", allow_pickle=False) as z:
            arrs.append({x: z[x] for x in z.files})
        kts.append(np.load(R.parts / f"{st}.kt.npy", allow_pickle=False))
    return arrs, np.concatenate(kts)


def _pk():
    import zpko
    return zpko


def cmd_gates(a):
    """G-W (ii): zu2m_a's own self-test. G-N: the dev pools with the plain rule_batch and with make_rule's places are bit-identical on two other
    shard layouts and on 1 and 2 numba threads, no POOL_GOLD field is in any pool file, and the places are separate files."""
    t0 = time.time()
    PK = _pk()
    import zu2m_a as A2
    base_main = OUTD / "dev"
    k0 = PK.stem(DS, "s1sel", (0, 1))
    if not (base_main / "pools" / "parts" / f"{k0}.json").is_file():
        stop("S0e (the dev pools, one shard) is not filed: run `zpl1h.py pools --carve s1sel` first")
    res = {"stage": "PL1 gates", "declared_in": DOC, "dataset": DS, "carve": "s1sel", "utc": utc()}
    rc = A2._selftest()
    res["G_W_ii"] = {"zu2m_a_selftest_rc": rc, "verdict": "PASS" if rc in (0, None) else "FAIL"}
    view = rjson(base_main / "pools" / "parts" / f"{k0}.view.json")
    partj = rjson(base_main / "pools" / "parts" / f"{k0}.json")
    ty = partj["types"]
    res["G_W_iii"] = {"weights": view["weights"], "types_in_part_record": ty,
                      "verdict": "PASS" if (view["weights"] == "off (2g)" and not view["plain"] and ty["weights"] == "off (2g)"
                                            and ty["em_entries_non_null"] == 0 and ty["score_entries_non_null"] == 0) else "FAIL"}
    base_arrs, base_kt = part_files(base_main, "s1sel", 1)
    runs = {}
    for tag, plain, k, th in (("plain2_t1", True, 2, 1), ("rule3_t2", False, 3, 2)):
        for i in range(k):
            ns = argparse.Namespace(carve="s1sel", shard=[str(i), str(k)], limit=0, threads=th, tag=tag, plain=plain)
            if not (OUTD / "gn" / tag / "pools" / "parts" / f"{PK.stem(DS, 's1sel', (i, k))}.json").exists():
                cmd_pools(ns)
        arrs, kt = part_files(OUTD / "gn" / tag, "s1sel", k)
        merged = PK.merge_arrays(arrs) if k > 1 else arrs[0]
        same = {x: bool(np.array_equal(np.asarray(merged[x]), np.asarray(base_arrs[0][x])) and np.asarray(merged[x]).dtype == np.asarray(base_arrs[0][x]).dtype)
                for x in base_arrs[0]}
        same["kt"] = bool(np.array_equal(kt, base_kt) and kt.dtype == base_kt.dtype)
        same["keys_equal"] = sorted(merged) == sorted(base_arrs[0])
        runs[tag] = {"shards": k, "plain": plain, "threads": th, "identical": all(same.values()), "per_array": same}
    gold_fields = sorted(set(base_arrs[0]) & set(PK.POOL_GOLD))
    # the places file is a separate file; the two keys it shares with the pools file (the question ids and the KT row lengths, for alignment)
    # carry no gold, every other key of it is absent from the pools file
    pf = places_dir(base_main) / f"{k0}.npz"
    shared = sorted(set(np.load(pf).files) & set(base_arrs[0])) if pf.is_file() else None
    places_sep = pf.is_file() and set(shared) <= {"ids", "kt_len"}
    res["G_N"] = {"runs": runs, "pool_gold_fields": gold_fields, "places_are_a_separate_file": bool(places_sep), "places_keys_shared": shared,
                  "verdict": "PASS" if all(r["identical"] for r in runs.values()) and not gold_fields and places_sep else "FAIL"}
    res["verdict"] = "PASS" if all(res[k]["verdict"] == "PASS" for k in ("G_W_ii", "G_W_iii", "G_N")) else "FAIL"
    res["seconds"] = round(time.time() - t0, 1)
    res["script_sha256"] = script_sha256()
    wjson(S0D / "gates_dev.json", res)
    log(f"gates: G-W ii {res['G_W_ii']['verdict']}, G-W iii {res['G_W_iii']['verdict']}, G-N {res['G_N']['verdict']} -> {res['verdict']}")
    return 0 if res["verdict"] == "PASS" else 3


# ── self-test ────────────────────────────────────────────────────────────────

def selftest():
    import zpko as PK
    import zu2d_pools as ZP
    import zu2m_a as A2
    import zu2p as P2
    # 1. the relaxed open_carve differs from the pinned one in the one depth line and nothing else
    fn, old, new = relaxed_open_carve(ZP)
    d = source_diff(old, new)
    assert len(d) == 2 and d[0].startswith("-") and d[1].startswith("+") and "1000" in d[0] and "min(1000, n)" in d[1], d
    assert fn.__globals__ is ZP.__dict__ and fn.__name__ == "open_carve" and fn is not ZP.open_carve
    # 2. 2g's null context: the weights through the walk's own functions are exactly 1, in this module's own shapes
    arr, k = null_types(7)
    assert arr["mu"].shape == (k, Q_DIM) and not arr["mu"].any() and (arr["cls"] == k).all() and arr["cls"].dtype == np.int8
    q = np.random.default_rng(1).normal(size=Q_DIM)
    wa = A2.omega(q, np.asarray(arr["mu"], np.float64), PK.Q.A_BETA, True)[0]
    ws = P2.weights(np.zeros(P2.NT), float(PK.Q.carried_walk()["beta"]))
    assert (wa == 1.0).all() and (P2.pair_weights(wa, ws) == 1.0).all()
    # 3. pure helpers
    keys = np.asarray([1 * 6 + 3, 2 * 6 + 5, 0 * 6 + 1], np.int64)
    keys.sort()
    ip, col = sym_csr(keys, 6)
    assert ip.tolist() == [0, 1, 3, 4, 5, 6, 7][:0] or ip[-1] == 6 and col.size == 6 and np.all(np.diff(ip) >= 0)
    sa, sb = struct_arrays(keys, 6)
    assert (sa.astype(np.int64) * 6 + sb == keys).all()
    # 4. the grid on a toy corpus: 31 cells, unique names, the rule picks one of them
    op = _plain_opened()
    rng = np.random.default_rng(4)
    n, nq = 40, 9
    dense = np.stack([rng.permutation(n) for _ in range(nq)]).astype(np.int64)
    splade = np.stack([rng.permutation(n) for _ in range(nq)]).astype(np.int64)
    golds = [np.unique(rng.choice(n, 3, replace=False)) for _ in range(nq)]
    pairs = np.unique(rng.integers(0, n * n, 80))
    pairs = pairs[(pairs // n) < (pairs % n)]
    ga = (pairs // n).astype(np.int32)
    gb = (pairs % n).astype(np.int32)
    M3A = op.m3a
    fam = {"structural": M3A.FamilyCSR(SimpleNamespace(n_nodes=n, src=ga[:10], dst=gb[:10], weight=None), "structural"),
           "ner": M3A.FamilyCSR(SimpleNamespace(n_nodes=n, src=ga, dst=gb, weight=rng.random(ga.size).astype(np.float32)), "ner"),
           "knn": M3A.FamilyCSR(SimpleNamespace(n_nodes=n, src=ga[10:], dst=gb[10:], weight=rng.random(ga.size - 10).astype(np.float32)), "knn")}
    rows = grid_rows(op, dense, splade, golds, n, fam, "toy")
    cells = cells_of_rows(op, rows, "toy")
    assert len(cells) == 31 and len({(c["pool"], c["regime"], c["setting"], c["base_pool"]) for c in cells}) == 31, len(cells)
    rec = op.m3b_contract.finish_record({"dataset": "toy"}, cells, "toy", op.cfg, op.cfg_h)
    assert rec["reading"] == "rule_as_filed" and rec["chosen"] in cells and rec["construction"]["regime"] in ("RETRIEVAL", "STRUCT", "NER", "KNN", "FULL")
    assert all(c["candidates_mean"] <= n for c in cells)
    # 5. the world: every patch is in force inside and put back outside (the package is read, nothing is run)
    import step4_pool_reach as S4
    M = PW.mods_of(False)
    keep = (S4.Opened, M.E.open_dataset, M.E.union_pools, M.E.C4.populations, M.U2C.graph_csr, M.ZP.u2e_graph, M.K.G.R2.family_stores,
            S4.m3b_pools.load_or_build_store)
    spec = SimpleNamespace(tab={"structural_keys": keys, "ner": {}, "knn": {}}, layout_sha="x", construction={"base_pool": "equal_rrf_budget_50",
                           "regime": "RETRIEVAL", "setting": None}, pops=[], lists=(np.zeros((1, 1), np.int32), np.zeros((1, 1), np.float16)))
    # the structural CSR is built from the keys over the package's 609 nodes
    spec.tab["structural_keys"] = np.asarray([1 * 609 + 3, 2 * 609 + 5], np.int64)
    stats = {}
    with world(M, spec, stats) as W:
        assert S4.Opened is not keep[0] and M.E.open_dataset is not keep[1] and M.ZP.u2e_graph is not keep[5]
        op2 = S4.Opened()
        assert op2.construction[DS]["base_pool"] == "equal_rrf_budget_50" and str(op2.served) == str(ZM.PKG)
        ds = op2.canonical.Dataset(DS, root=str(op2.served))
        assert ds.n_nodes == 609 and ds.n_queries == 2255 and DS in op2.canonical.DATASETS
        assert W.desc["pairs"] == 2 and W.ip[-1] == 4 and W.source_diff
        assert M.U2C.graph_csr(DS, "chosen", 609)[2] is W.desc
    assert (S4.Opened, M.E.open_dataset, M.E.union_pools, M.E.C4.populations, M.U2C.graph_csr, M.ZP.u2e_graph, M.K.G.R2.family_stores,
            S4.m3b_pools.load_or_build_store) == keep
    assert DS not in op2.canonical.DATASETS and DS not in op2.canonical.SPLITS
    try:
        op2.canonical.Dataset(DS, root=str(ZM.PKG))
        raise AssertionError("the loader still knows the dataset after the world")
    except ValueError:
        pass
    # 6. the populations (laptop: reads the package's query files only)
    P = package()
    pops = populations(_plain_opened(), P)
    gp = gate_p(pops)
    assert gp["s1sel"]["questions"] == 224 and gp["s1eval"]["questions"] == 1130 and gp["verdict"] == "PASS", gp
    print("selftest ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for nm in ("first", "layout", "record"):
        s = sub.add_parser(nm)
        s.add_argument("--redo", action="store_true")
    g = sub.add_parser("gk")
    g.add_argument("--ds", default="musique")
    sub.add_parser("knee")
    pl = sub.add_parser("pools")
    pl.add_argument("--carve", required=True)
    pl.add_argument("--shard", nargs=2, default=["0", "1"])
    pl.add_argument("--limit", type=int, default=0)
    pl.add_argument("--threads", type=int, default=2)
    pl.add_argument("--tag", default="")
    pl.add_argument("--plain", action="store_true")
    sub.add_parser("gates")
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    return {"first": cmd_first, "layout": cmd_layout, "gk": cmd_gk, "knee": cmd_knee, "record": cmd_record, "pools": cmd_pools,
            "gates": cmd_gates}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
