"""STK1 (docs/STK1_STARK_PACKAGE.md): the three STaRK knowledge bases (Prime, MAG, Amazon) as the eighth to tenth datasets,
packaged in the CRAG datasets' layout by ONE builder, encoded by e3v on the host CPU, and read at the first stage.
Nothing trains. No filed module is edited: zenc, zenc3 and zenc3v are imported and, in this process only, zenc's
`served_root` is pointed at the STaRK package (as zmhr does for MultiHop-RAG).

    python -I outputs/mp_unified/zstk1.py selftest
    python -I outputs/mp_unified/zstk1.py package --kb prime|mag|amazon --proc DIR --dl DIR     section 4 and 5 (P1-P5)
    python -I outputs/mp_unified/zstk1.py encode --kb K [--threads 8] [--smoke N]               host CPU: section 6
    python -I outputs/mp_unified/zstk1.py vectors --kb K                                        fuse e3v from its parts
    python -I outputs/mp_unified/zstk1.py first --kb K [--threads 8] [--limit-q N]              section 7: the legs and the metrics

DIR arguments are folders of downloaded (untrusted) files; every node_info.pkl is read through `Safe` only, every .pt
through torch.load(weights_only=True). Run with `python -I`.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # CPU by design: the card is the other user's (section 6)
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import zenc as E  # noqa: E402  (first: reads --threads from sys.argv before numpy loads)

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DOC = "docs/STK1_STARK_PACKAGE.md"
WS = HERE.parents[1]
ROOT = WS / "outputs" / "stark"
PKG = ROOT / "package"
FREEZE = PKG / "PACKAGE_FREEZE.json"
KB = {"prime": "stark_prime", "mag": "stark_mag", "amazon": "stark_amazon"}
CAP = 1200                                          # section 4: characters of a node's text
SPLITS = ("train", "dev", "test")                   # the benchmark's own train, val, test index files
INDEX = {"train": "train.index", "dev": "val.index", "test": "test.index"}
CPU_BUDGET = 8192                                   # section 6: tokens per batch on the CPU
BLK = 4096                                          # section 6: rows per checkpointed block
TOPK = 2500                                         # section 7: the depth every leg is read to
KS = (1, 5, 20, 100, 1000, 2500)
BOOT, SEED = 1000, 0
PUBLISHED = {                                       # section 7: the benchmark paper's Table 6, full test (v3, as printed)
    "prime": {"BM25": (12.75, 27.92, 31.25, 19.84), "ada-002": (12.63, 31.49, 36.00, 21.41),
              "multi-ada-002": (15.10, 33.56, 38.05, 23.49)},
    "mag": {"BM25": (25.85, 45.25, 45.69, 34.91), "ada-002": (29.08, 49.61, 48.36, 38.62),
            "multi-ada-002": (25.92, 50.43, 50.80, 36.94)},
    "amazon": {"BM25": (44.94, 67.42, 53.77, 55.30), "ada-002": (39.16, 62.73, 53.29, 50.35),
               "multi-ada-002": (40.07, 64.98, 55.12, 51.55)},
}
DQ = ("row j is the dense vector of query j == queries/query_ids.json[j] == the j-th line of the split files "
      "['train', 'dev', 'test'] concatenated in that order")


def log(msg):
    print(f"[zstk1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zstk1: {msg}")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(WS)).replace("\\", "/")
    except ValueError:
        return str(p)


def code_sha():
    return {"outputs/mp_unified/zstk1.py": E.sha256(Path(__file__))}


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── the safe readers ─────────────────────────────────────────────────────────

ALLOWED = {("builtins", n) for n in ("dict", "list", "set", "tuple", "str", "int", "float", "bool", "frozenset")} | {
    ("collections", "OrderedDict"),
    # Amazon's node_info.pkl holds numpy scalars: the dtype constructor and the scalar rebuild, nothing else of numpy
    ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}


class Safe(pickle.Unpickler):
    """Builtin containers, OrderedDict and numpy's dtype and scalar rebuild only; any other global raises."""

    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def safe_load(path):
    with open(path, "rb") as f:
        return Safe(f).load()


def load_pt(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def id_name(d):
    """A type dictionary as {int id: name}, whichever way the file maps."""
    if not isinstance(d, dict) or not d:
        stop(f"a type dictionary that is not a dict: {type(d).__name__}")
    k0 = next(iter(d))
    out = {int(v): str(k) for k, v in d.items()} if isinstance(k0, str) else {int(k): str(v) for k, v in d.items()}
    if len(out) != len(d):
        stop("a type dictionary with repeated ids")
    return out


# ── section 4: the text rule T1 ──────────────────────────────────────────────

_LETTER = re.compile(r"[^\W\d_]")


def flat(v, seen):
    """One value as text. A string is kept if it has a letter, is not 'nan' and is not already seen; a list keeps its
    order; a dict's children are deduplicated in stored order, then ordered by length, shortest first (ties stored)."""
    if isinstance(v, str):
        s = " ".join(v.split())
        if not s or s.lower() == "nan" or not _LETTER.search(s) or s in seen:
            return ""
        seen.add(s)
        return s
    if isinstance(v, (list, tuple)):
        parts = [flat(x, seen) for x in v]
        return "; ".join(p for p in parts if p)
    if isinstance(v, dict):
        kids = [flat(x, seen) for x in v.values()]
        kids = [k for k in kids if k]
        kids.sort(key=len)
        return "; ".join(kids)
    return ""


def node_text(rec, tname):
    """T1: the head `<type>:` then the record's other fields flattened (its own 'type' key left out), cut at CAP."""
    body = flat({k: v for k, v in rec.items() if k != "type"}, set())
    return (f"{tname}: {body}" if body else f"{tname}:")[:CAP].rstrip()


# ── section 5: the package ───────────────────────────────────────────────────

def rows_manifest(ds, kind, n):
    ns = (n + E.SHARD - 1) // E.SHARD
    return {"dataset": ds, "kind": kind, "model": "dense", "n_rows": n, "shard_size": E.SHARD, "n_shards": ns,
            "row_semantics": ("row i is the dense vector of position i == line i of nodes.jsonl" if kind == "docs" else DQ),
            "format": {"file": "shard_%05d.npy", "dtype": "float16", "shape": "(rows, dim)",
                       "normalised": "unit L2 (|norm-1| <= 0.01)"},
            "encoder": {"encoder": None, "note": "STaRK has no frozen-arm vectors; e3v's parts live in outputs/enc/"},
            "shards": [{"shard": s, "rows": min(E.SHARD, n - s * E.SHARD), "file": "shard_%05d.npy" % s, "bytes": None,
                        "sha256": None} for s in range(ns)],
            "status": "rows only (no frozen vectors for STaRK)", "page": DOC}


def write_lines(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return {"file": rel(p), "bytes": p.stat().st_size, "sha256": E.sha256(p), "n": len(rows)}


def read_qa(dl):
    csv.field_size_limit(1 << 28)
    with open(Path(dl) / "stark_qa.csv", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        if head != ["id", "query", "answer_ids"]:
            stop(f"stark_qa.csv columns {head}")
        rows, reps = {}, {}
        for r in rd:
            qid = int(r[0])
            if qid in rows:
                stop(f"stark_qa.csv repeats id {qid}")
            ans = json.loads(r[2])
            if not isinstance(ans, list) or not ans or not all(isinstance(x, int) for x in ans):
                stop(f"id {qid}: answer_ids {r[2][:60]!r}")
            dist = list(dict.fromkeys(ans))                     # amendment A1: the gold set is the distinct nodes, first-seen order
            if len(dist) != len(ans):
                reps[qid] = (len(ans), len(dist))
            rows[qid] = (r[1], dist)
    idx = {}
    for sp, fn in INDEX.items():
        idx[sp] = [int(x) for x in (Path(dl) / fn).read_text(encoding="utf-8").split()]
    t01 = [int(x) for x in (Path(dl) / "test-0.1.index").read_text(encoding="utf-8").split()]
    return rows, idx, t01, reps


def cmd_package(a):
    t0 = time.time()
    ds = KB[a.kb]
    proc, dl = Path(a.proc), Path(a.dl)
    root = PKG / ds
    if (root / "DATASET.json").is_file():
        stop(f"{rel(root)} exists: the package is built once")
    info = safe_load(proc / "node_info.pkl")
    n = len(info)
    if sorted(info) != list(range(n)):
        stop("P1: node_info.pkl keys are not 0..N-1")
    ntype = load_pt(proc / "node_types.pt")
    tdict = id_name(safe_load(proc / "node_type_dict.pkl"))
    if int(ntype.shape[0]) != n:
        stop(f"P1: node_types has {int(ntype.shape[0])} rows, node_info {n}")
    ntype = ntype.numpy().astype(np.int64)
    if not set(np.unique(ntype).tolist()) <= set(tdict):
        stop("P1: a node type id has no name")
    ei = load_pt(proc / "edge_index.pt").numpy().astype(np.int64)
    et = load_pt(proc / "edge_types.pt").numpy().astype(np.int64)
    rdict = id_name(safe_load(proc / "edge_type_dict.pkl"))
    if ei.ndim != 2 or ei.shape[0] != 2 or ei.shape[1] != et.shape[0]:
        stop(f"P1: edge_index {ei.shape}, edge_types {et.shape}")
    if int(ei.min()) < 0 or int(ei.max()) >= n or not set(np.unique(et).tolist()) <= set(rdict):
        stop("P1: an edge endpoint out of range or a relation id with no name")
    rows, idx, t01, reps = read_qa(dl)
    rep_by = {sp: {"questions": sum(1 for q in idx[sp] if q in reps),
                   "answer_ids_dropped": sum(reps[q][0] - reps[q][1] for q in idx[sp] if q in reps)} for sp in SPLITS}
    allids = [q for sp in SPLITS for q in idx[sp]]
    if len(set(allids)) != len(allids) or set(allids) != set(rows):
        stop(f"P1: splits {[len(idx[s]) for s in SPLITS]} do not partition stark_qa.csv ({len(rows)} rows)")
    if not set(t01) <= set(idx["test"]):
        stop("P1: test-0.1 is not inside test")
    # nodes (T1)
    nodes, lens, tcount = [], [], {}
    for i in range(n):
        tn = tdict[int(ntype[i])]
        tcount[tn] = tcount.get(tn, 0) + 1
        t = node_text(info[i], tn)
        nodes.append({"node_id": f"{ds}:{i}", "text": t, "type": tn, "kb_index": i, "content_hash": sha_text(t)})
        lens.append(len(t))
    del info
    # queries
    qrows, qorder, bad_gold, gtypes = [], [], 0, {}
    for sp in SPLITS:
        part = []
        for q in idx[sp]:
            text, ans = rows[q]
            if len(set(ans)) != len(ans) or min(ans) < 0 or max(ans) >= n:
                bad_gold += 1
                continue
            for x in ans:
                gtypes[nodes[x]["type"]] = gtypes.get(nodes[x]["type"], 0) + 1
            part.append({"query_id": f"{ds}_q{q:05d}", "question": text, "gold_node_ids": [nodes[x]["node_id"] for x in ans],
                         "n_golds": len(ans), "split": sp, "source_position": q, "dataset": ds})
        qrows.append((sp, part))
        qorder += [r["query_id"] for r in part]
    if bad_gold:
        stop(f"P1: {bad_gold} questions with an answer out of range or repeated")
    files = {"nodes": write_lines(root / "nodes.jsonl", nodes)}
    for sp, part in qrows:
        files[f"queries/{sp}"] = write_lines(root / "queries" / f"{sp}.jsonl", part)
    E.wjson(root / "queries" / "query_ids.json", qorder)
    files["query_ids"] = {"file": rel(root / "queries" / "query_ids.json"), "sha256": E.sha256(root / "queries" / "query_ids.json"),
                          "n": len(qorder)}
    for kind, m in (("docs", n), ("queries", len(qorder))):
        E.wjson(root / "embeddings" / "dense" / kind / "manifest.json", rows_manifest(ds, kind, m))
    # the S store: the benchmark's edges, labelled by relation (section 5); STK2 reads it
    g = root / "graph"
    g.mkdir(parents=True, exist_ok=True)
    np.savez(g / "s_edges.npz", src=ei[0].astype(np.int32), dst=ei[1].astype(np.int32), rel=et.astype(np.int16))
    E.wjson(g / "s_rel.json", {str(k): v for k, v in sorted(rdict.items())})
    key = ei[0] * n + ei[1]
    uk = np.unique(key)
    rec_share = float(np.isin(ei[1] * n + ei[0], uk).mean())
    s_stats = {"edges": int(ei.shape[1]), "distinct_pairs": int(uk.size), "self_loops": int((ei[0] == ei[1]).sum()),
               "reciprocal_share": round(rec_share, 6), "relations": len(rdict),
               "by_relation": {rdict[int(r)]: int(c) for r, c in zip(*np.unique(et, return_counts=True))}}
    ds_json = {"RECORD": "STK1_PACKAGE", "dataset": ds, "kb": a.kb, "declared_in": DOC, "n_nodes": n, "n_queries": len(qorder),
               "queries_by_split": {sp: len(p) for sp, p in qrows}, "node_types": dict(sorted(tcount.items())),
               "layout": {"nodes": "nodes.jsonl", "queries": "queries/<split>.jsonl + queries/query_ids.json",
                          "embeddings": "embeddings/dense/<docs|queries>/manifest.json (rows only) + outputs/enc/<arm>/<ds>/",
                          "graph": "graph/s_edges.npz (src, dst, rel) + graph/s_rel.json"},
               "nodes": {**files["nodes"], "fields": sorted(nodes[0]),
                         "position_semantics": "line i of nodes.jsonl is the benchmark's node i: embeddings row i, graph endpoint i"},
               "queries": {"n": len(qorder), "splits_in_row_order": list(SPLITS),
                           "by_split": {sp: files[f"queries/{sp}"] for sp in SPLITS}, "query_ids": files["query_ids"],
                           "test_0_1": [f"{ds}_q{q:05d}" for q in t01]},
               "split_rule": "the benchmark's own index files: train.index, val.index (named dev here), test.index, each in file order",
               "answer_repeats_a1": rep_by,
               "text_rule": f"T1 (section 4), cap {CAP} characters", "s_edges": s_stats,
               "raw": {k: E.sha256(dl / k) for k in ("stark_qa.csv", "train.index", "val.index", "test.index", "test-0.1.index")}
               | {k: E.sha256(proc / k) for k in ("node_info.pkl", "edge_index.pt", "edge_types.pt", "node_types.pt")},
               "licence": "CC-BY-4.0 (the dataset card's)"}
    E.wjson(root / "DATASET.json", ds_json)
    fr = json.loads(FREEZE.read_text(encoding="utf-8")) if FREEZE.is_file() else {"RECORD": "STK1_PACKAGE", "datasets": {}}
    fr["datasets"][ds] = {"dataset_json_sha256": E.sha256(root / "DATASET.json"), "utc": utc()}
    fr["declared_in"] = DOC
    E.wjson(FREEZE, fr)
    # P4: zenc's readers accept it
    E.served_root = lambda: PKG
    qt = E.query_texts(PKG, ds)
    ch = list(E.doc_chunks(PKG, ds))
    p4 = {"query_rows": len(qt), "doc_rows": sum(len(c[1]) for c in ch), "doc_shards": len(ch),
          "texts_equal": [t for c in ch for t in c[1]] == [x["text"] for x in nodes],
          "first_question_equal": qt[0] == rows[idx["train"][0]][0]}
    p4["pass"] = p4["query_rows"] == len(qorder) and p4["doc_rows"] == n and p4["texts_equal"] and p4["first_question_equal"]
    la = np.asarray(lens)
    rec = {"stage": "STK1 package", "declared_in": DOC, "dataset": ds, "nodes": n, "queries": len(qorder),
           "by_split": ds_json["queries_by_split"], "node_types": ds_json["node_types"], "gold_node_types": gtypes,
           "answer_repeats_a1": rep_by,
           "P1": "pass", "P4": p4, "P5": s_stats, "text_chars": {"mean": round(float(la.mean()), 1), "median": int(np.median(la)),
                                                                  "p90": int(np.percentile(la, 90)), "max": int(la.max()),
                                                                  "at_cap": int((la >= CAP).sum()),
                                                                  "head_only": int(sum(1 for x in nodes if x["text"].endswith(":")))},
           "files": files, "dataset_json_sha256": E.sha256(root / "DATASET.json"), "seconds": round(time.time() - t0, 1),
           "utc": utc(), **code_sha()}
    E.wjson(ROOT / a.kb / "package.json", rec)
    log(f"package {ds}: {n:,} nodes, {len(qorder):,} queries {rec['by_split']}, {s_stats['edges']:,} edges; "
        f"P4 {p4['pass']}; text mean {rec['text_chars']['mean']} chars")
    if not p4["pass"]:
        stop(f"P4 failed: {p4}")
    return 0


# ── section 6: the encode ────────────────────────────────────────────────────

def zmods():
    import zenc3 as Z3
    import zenc3v as V
    return Z3, V


def cmd_encode(a):
    import torch
    Z3, V = zmods()
    torch.set_num_threads(int(a.threads))
    if torch.cuda.is_available():
        stop("torch still sees a GPU")
    ds = KB[a.kb]
    E.served_root = lambda: PKG
    root = PKG
    g0 = E.OUT / "g0.json"
    if not g0.is_file() or json.loads(g0.read_text(encoding="utf-8")).get("call") != "PASS":
        stop("G0 has not passed (outputs/enc/g0.json)")
    if a.smoke:
        _, ts0 = next(iter(E.doc_chunks(root, ds, lambda s: s == 0)))
        texts = ts0[:a.smoke]
        for e in V.PARTS:
            E.ENCODERS[e]["budget"] = CPU_BUDGET
            m = E.load_model(e, "cpu", Z3.pinned(e))
            t1 = time.time()
            v, st = E.encode_texts(m, e, texts, "docs")
            dt = time.time() - t1
            log(f"smoke {e}: {len(texts)} nodes in {dt:.1f} s = {len(texts) / dt:.1f}/s, {st['tokens']:,} tokens "
                f"({st['tokens'] / dt:.0f}/s), cut {st['cut']}, max_len {st['max_len']}, threads {a.threads}")
            del m
        return 0
    t0 = time.time()
    for e in V.PARTS:
        rev = Z3.pinned(e)
        E.ENCODERS[e]["budget"] = CPU_BUDGET
        m = E.load_model(e, "cpu", rev)
        if m.default_prompt_name:
            stop(f"{e}: a default prompt would prefix the documents")
        for kind in ("queries", "docs"):
            rman = E.manifest(root, ds, kind)
            base = E.OUT / e / ds / kind
            base.mkdir(parents=True, exist_ok=True)

            def done(s):
                jp, npy = base / f"shard_{s:05d}.json", base / f"shard_{s:05d}.npy"
                return jp.is_file() and npy.is_file() and json.loads(jp.read_text(encoding="utf-8")).get("sha256") == E.sha256(npy)

            if kind == "docs":
                chunks = E.doc_chunks(root, ds, lambda s: not done(s))
            else:
                allq = E.query_texts(root, ds)
                qs = [allq[j] for j in range(len(allq))]
                chunks = ((s, None if done(s) else qs[s * E.SHARD:(s + 1) * E.SHARD]) for s in range(rman["n_shards"]))
            shards = []
            for s, texts in chunks:
                npy, jp = base / f"shard_{s:05d}.npy", base / f"shard_{s:05d}.json"
                if texts is None:
                    shards.append(json.loads(jp.read_text(encoding="utf-8")))
                    continue
                t1, parts, toks = time.time(), [], 0
                for b in range(0, len(texts), BLK):
                    pp = base / f"shard_{s:05d}.blk{b // BLK:04d}.npy"
                    if pp.is_file():
                        parts.append(pp)
                        continue
                    v, st = E.encode_texts(m, e, texts[b:b + BLK], kind)
                    toks += st["tokens"]
                    bad = E.damaged(v)
                    if bad or st["cut"]:
                        stop(f"{e} {ds} {kind} shard {s} block {b // BLK}: {bad} damaged rows, {st['cut']} texts cut at the window")
                    tmp = pp.with_name(pp.stem + ".tmp.npy")
                    np.save(tmp, v)
                    os.replace(tmp, pp)
                    parts.append(pp)
                    done_rows = min(b + BLK, len(texts))
                    log(f"{e} {kind} shard {s}: {done_rows:,}/{len(texts):,} rows ({time.time() - t1:.0f} s)")
                v = np.concatenate([np.load(pp) for pp in parts], 0)
                if v.shape[0] != len(texts):
                    stop(f"{e} {ds} {kind} shard {s}: {v.shape[0]} rows for {len(texts)} texts")
                tmp = base / f"shard_{s:05d}.tmp.npy"
                np.save(tmp, v)
                os.replace(tmp, npy)
                r = {"shard": s, "rows": int(v.shape[0]), "file": npy.name, "bytes": npy.stat().st_size, "sha256": E.sha256(npy),
                     "seconds": round(time.time() - t1, 2), "device": "cpu", "tokens_this_run": int(toks), "cut": 0,
                     "limit": int(m.max_seq_length), "budget": CPU_BUDGET}
                E.wjson(jp, r)
                for pp in parts:                      # this process's own scratch blocks, assembled and checksummed above
                    pp.unlink()
                shards.append(r)
                log(f"{e} {ds} {kind} shard {s}: {r['rows']:,} rows assembled ({r['seconds']} s)")
            rows = sum(r["rows"] for r in shards)
            if rows != rman["n_rows"] or len(shards) != rman["n_shards"]:
                stop(f"{e} {ds} {kind}: {rows} rows in {len(shards)} shards, the package has {rman['n_rows']} in {rman['n_shards']}")
            E.wjson(base / "manifest.json", {
                "dataset": ds, "kind": kind, "model": e, "n_rows": rows, "shard_size": E.SHARD, "n_shards": len(shards),
                "row_semantics": rman["row_semantics"],
                "format": {"file": "shard_%05d.npy", "dtype": "float16", "shape": f"(rows, {E.ENCODERS[e]['dim']})",
                           "normalised": "unit L2 (|norm-1| <= 0.01)"},
                "encoder": {"encoder": E.ENCODERS[e]["repo"], "revision": rev, "dim": E.ENCODERS[e]["dim"], "dtype": "float16",
                            "device": "cpu", "max_seq_length": int(m.max_seq_length), "weights": "float32",
                            "query_prompt": E.query_prompt(m, e) if kind == "queries" else "", "document_prompt": ""},
                "seconds": round(sum(r["seconds"] for r in shards), 1), "cut": int(sum(r["cut"] for r in shards)),
                "shards": [{k: r[k] for k in ("shard", "rows", "file", "bytes", "sha256")} for r in shards],
                "stage": "STK1 encode", "page": DOC, "threads": int(a.threads), "budget": CPU_BUDGET, "utc": utc(),
                "code_sha256": code_sha()})
        del m
    log(f"{a.kb}: both parts encoded in {time.time() - t0:.0f} s")
    return 0


def cmd_vectors(a):
    Z3, V = zmods()
    E.served_root = lambda: PKG
    return V.cmd_vectors(argparse.Namespace(ds=KB[a.kb]))


# ── section 7: the first stage ───────────────────────────────────────────────

def metrics(top, ans):
    """top (nq, TOPK) node ids, ans list of int arrays: per question hit1, hit5, r20, mrr (within TOPK) and, per k in KS,
    recall and all-golds."""
    nq = top.shape[0]
    out = {"hit1": np.zeros(nq), "hit5": np.zeros(nq), "r20": np.zeros(nq), "mrr": np.zeros(nq)}
    for k in KS:
        out[f"R@{k}"] = np.zeros(nq)
        out[f"ALL@{k}"] = np.zeros(nq)
    for q in range(nq):
        a_ = ans[q]
        hit = np.isin(top[q], a_)
        pos = np.flatnonzero(hit)
        out["hit1"][q] = float(hit[0])
        out["hit5"][q] = float(hit[:5].any())
        out["r20"][q] = hit[:20].sum() / a_.size
        out["mrr"][q] = 1.0 / (pos[0] + 1) if pos.size else 0.0
        for k in KS:
            c = int(hit[:k].sum())
            out[f"R@{k}"][q] = c / a_.size
            out[f"ALL@{k}"][q] = float(c == a_.size)
    return out


def boot(x, B=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    m = x.shape[0]
    mu = np.asarray([x[rng.integers(0, m, m)].mean() for _ in range(B)])
    return float(x.mean()), float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))


def call(lo, hi, ref):
    return "ABOVE" if lo > ref else ("BELOW" if hi < ref else "LEVEL")


def bm25_top(texts, qtexts, k=TOPK, k1=1.5, b=0.75, chunk=64):
    """Okapi BM25 over the node texts (lowercase [a-z0-9] tokens, Lucene-style idf); ties to the lower id; positive scores only."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+", dtype=np.float32)
    X = cv.fit_transform(texts).tocsr()
    n = X.shape[0]
    dl = np.asarray(X.sum(1)).ravel()
    avg = float(dl.mean()) or 1.0
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float64)
    idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(n), np.diff(X.indptr))
    tf = X.data
    X.data = (idf[X.indices] * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl[rows] / avg))).astype(np.float32)
    XT = X.T.tocsr()
    Qc = cv.transform(qtexts).tocsr()
    top = np.full((len(qtexts), k), -1, np.int64)
    for q0 in range(0, len(qtexts), chunk):
        S = (Qc[q0:q0 + chunk] @ XT).toarray()
        for r in range(S.shape[0]):
            s = S[r]
            kk = min(k, int((s > 0).sum()))
            if kk == 0:
                continue
            cand = np.argpartition(-s, kk - 1)[:kk]
            o = np.lexsort((cand, -s[cand]))
            top[q0 + r, :kk] = cand[o]
    return top


def cmd_first(a):
    ds = KB[a.kb]
    E.served_root = lambda: PKG
    root = PKG
    t0 = time.time()
    qids = json.loads((root / ds / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    test_rows, ans, qtexts = [], [], []
    j = 0
    pos0 = sum(1 for sp in SPLITS[:2] for _ in open(root / ds / "queries" / f"{sp}.jsonl", encoding="utf-8"))
    with open(root / ds / "queries" / "test.jsonl", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            test_rows.append(pos0 + j)
            ans.append(np.asarray([int(x.rsplit(":", 1)[1]) for x in r["gold_node_ids"]], np.int64))
            qtexts.append(r["question"])
            if qids[pos0 + j] != r["query_id"]:
                stop(f"query row {pos0 + j}: {r['query_id']} is not query_ids.json's {qids[pos0 + j]}")
            j += 1
    if a.limit_q:
        test_rows, ans, qtexts = test_rows[:a.limit_q], ans[:a.limit_q], qtexts[:a.limit_q]
    qp = E.vec_shards(root, "e3v", ds, "queries")
    Qall = np.concatenate([np.load(p).astype(np.float32) for p in qp])
    Q = Qall[np.asarray(test_rows)]
    top_d, _, _ = E.search(Q, E.vec_shards(root, "e3v", ds, "docs"), TOPK)
    log(f"dense search: {Q.shape[0]} questions ({time.time() - t0:.0f} s)")
    texts = [json.loads(x)["text"] for x in open(root / ds / "nodes.jsonl", encoding="utf-8")]
    top_b = bm25_top(texts, qtexts)
    log(f"BM25: {len(texts):,} nodes ({time.time() - t0:.0f} s)")
    legs = {"dense_e3v": top_d, "bm25": top_b}
    if a.splade_top:
        z = np.load(a.splade_top)["ids"][np.asarray(test_rows)]
        legs["splade"] = z.astype(np.int64)
        rr = np.full((len(qtexts), TOPK), -1, np.int64)
        for q in range(len(qtexts)):
            sc = {}
            for lst in (top_d[q], z[q]):
                seen = set()
                for i, g in enumerate(lst.tolist()):
                    if g < 0 or g in seen:
                        continue
                    seen.add(g)
                    sc[g] = np.float32(sc.get(g, np.float32(0.0)) + np.float32(1.0) / (E.RRF_C + np.float32(i + 1)))
            top = sorted(sc.items(), key=lambda t: (-float(t[1]), t[0]))[:TOPK]
            rr[q, :len(top)] = [g for g, _ in top]
        legs["rrf_e3v_splade"] = rr
    rec = {"stage": "STK1 first", "declared_in": DOC, "dataset": ds, "questions": len(qtexts), "legs": {}, "published": PUBLISHED[a.kb],
           "limit_q": a.limit_q or None, "threads": E.THREADS, "code_sha256": code_sha(), "utc": utc()}
    for name, top in legs.items():
        mt = metrics(top, ans)
        rec["legs"][name] = {k: dict(zip(("mean", "lo", "hi"), (round(x * 100, 2) for x in boot(v)))) for k, v in mt.items()}
    for name, pub in PUBLISHED[a.kb].items():
        for leg in legs:
            cl = {}
            for k, ref in zip(("hit1", "hit5", "r20", "mrr"), pub):
                s = rec["legs"][leg][k]
                cl[k] = {"ours": s["mean"], "published": ref, "diff": round(s["mean"] - ref, 2), "call": call(s["lo"], s["hi"], ref)}
            rec.setdefault("calls", {}).setdefault(name, {})[leg] = cl
    pb, sb = PUBLISHED[a.kb]["BM25"], rec["legs"]["bm25"]
    gb = max(abs(sb["hit1"]["mean"] - pb[0]), abs(sb["r20"]["mean"] - pb[2]))
    rec["G_B"] = {"hit1_diff": round(sb["hit1"]["mean"] - pb[0], 2), "r20_diff": round(sb["r20"]["mean"] - pb[2], 2),
                  "max_abs": round(gb, 2), "call": "REPRODUCES" if gb <= 3 else ("CLOSE" if gb <= 8 else "DIFFERENT")}
    d = ROOT / a.kb
    d.mkdir(parents=True, exist_ok=True)
    tag = "first" if not a.limit_q else f"first-smoke{a.limit_q}"
    np.savez(d / f"{tag}.npz", rows=np.asarray(test_rows), **{f"top_{k}": v.astype(np.int32) for k, v in legs.items()})
    rec["npz_sha256"] = E.sha256(d / f"{tag}.npz")
    rec["seconds"] = round(time.time() - t0, 1)
    E.wjson(d / f"{tag}.json", rec)
    for leg, s in rec["legs"].items():
        log(f"{a.kb} {leg}: Hit@1 {s['hit1']['mean']} Hit@5 {s['hit5']['mean']} R@20 {s['r20']['mean']} MRR {s['mrr']['mean']} "
            f"ALL@2500 {s['ALL@2500']['mean']}")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

def selftest():
    ok = 0

    def chk(name, cond):
        nonlocal ok
        if not cond:
            stop(f"selftest: {name}")
        ok += 1

    # T1: head, type key left out, numeric/nan/no-letter leaves dropped, duplicates dropped, shortest first, cap
    rec = {"id": 7, "type": "x", "name": "Alpha", "source": "S1", "details": {"b": "long text about alpha beta", "a": "C7orf69",
           "n": float("nan"), "z": "nan", "p": "12.5", "d": "Alpha", "l": ["tiny", "also tiny"], "num": 5}}
    t = node_text(rec, "gene/protein")
    chk("head and shortest first", t.startswith("gene/protein: S1; Alpha; "))
    chk("dedupe and drops", "nan" not in t.lower() and "12.5" not in t and t.count("Alpha") == 1)
    chk("shortest first", t.index("C7orf69") < t.index("long text about"))
    chk("list order", t.index("tiny; also tiny") > 0)
    long = {"name": "N", "review": [{"reviewText": "word " * 400}], "title": "T"}
    t2 = node_text(long, "product")
    chk("cap", len(t2) <= CAP and t2.startswith("product: N; T"))
    chk("head only", node_text({"id": 1, "type": "t", "n": 3}, "t") == "t:")
    # Safe: numpy scalar allowed, a hostile global blocked
    import numpy as _np
    buf = io.BytesIO()
    pickle.dump({"a": [_np.float64(1.5), "x"]}, buf)
    chk("numpy scalar loads", Safe(io.BytesIO(buf.getvalue())).load()["a"][1] == "x")

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo no",))
    buf2 = io.BytesIO()
    pickle.dump(Evil(), buf2)
    try:
        Safe(io.BytesIO(buf2.getvalue())).load()
        chk("hostile blocked", False)
    except pickle.UnpicklingError:
        chk("hostile blocked", True)
    # metrics
    top = np.asarray([[5, 1, 2, 9, 3], [7, 8, 0, 4, 6]], np.int64)
    old = KS
    globals()["KS"] = (1, 2, 5)
    try:
        m = metrics(top, [np.asarray([1, 3]), np.asarray([9])])
    finally:
        globals()["KS"] = old
    chk("hit1", m["hit1"].tolist() == [0.0, 0.0])
    chk("hit5/r20", m["hit5"].tolist() == [1.0, 0.0] and abs(m["r20"][0] - 1.0) < 1e-9 and m["r20"][1] == 0.0)
    chk("mrr", abs(m["mrr"][0] - 0.5) < 1e-9 and m["mrr"][1] == 0.0)
    chk("all@5", m["ALL@5"].tolist() == [1.0, 0.0] and m["ALL@2"].tolist() == [0.0, 0.0])
    # BM25: the document sharing the query's rare terms ranks first, ties to the lower id
    docs = ["red apple pie", "green apple", "blue sky over the sea", "apple pie recipe with red apples"]
    tb = bm25_top(docs, ["red apple pie", "sky sea"], k=4)
    chk("bm25 order", tb[0, 0] == 0 and tb[1, 0] == 2 and (tb[1, 1] == -1))
    # id_name accepts both directions
    chk("id_name", id_name({"a": 0, "b": 1}) == {0: "a", 1: "b"} and id_name({0: "a", 1: "b"}) == {0: "a", 1: "b"})
    log(f"selftest PASS ({ok} checks)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for c in ("package", "encode", "vectors", "first"):
        s = sub.add_parser(c)
        s.add_argument("--kb", required=True, choices=sorted(KB))
        s.add_argument("--threads", type=int, default=8)
        if c == "package":
            s.add_argument("--proc", required=True)
            s.add_argument("--dl", required=True)
        if c == "encode":
            s.add_argument("--smoke", type=int, default=0)
        if c == "first":
            s.add_argument("--limit-q", type=int, default=0)
            s.add_argument("--splade-top", default=None)
    a = p.parse_args(argv)
    return {"selftest": lambda: selftest(), "package": lambda: cmd_package(a), "encode": lambda: cmd_encode(a),
            "vectors": lambda: cmd_vectors(a), "first": lambda: cmd_first(a)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
