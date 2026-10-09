"""B1a: HippoRAG 2's three benchmark settings on our substrate and graph method (docs/B1_HIPPORAG2_ALIGNMENT.md).

    python scripts/b1_build.py build [--datasets musique 2wiki hotpotqa]   -> outputs/bench/hipporag2/<dataset>/
    python scripts/b1_build.py --selftest

Nothing is re-encoded: every passage and question maps to a row of the frozen CRAG package (read only), whose dense and
SPLADE vectors it takes. structural = the package family induced on the setting's rows; ner and knn are rebuilt on the
setting by the package's rules (CRAG build_ner_edges; exact 3-NN cosine).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
PKG = Path("C:/Users/Swastik/Desktop/CRAG/data/final_canonical")
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
RAW = BENCH / "raw"
SETTINGS = {"musique": ("musique", "dev"), "2wiki": ("2wikimultihopqa", "dev"), "hotpotqa": ("hotpotqa", "validation")}
SPLIT_ORDER = {"musique": ("train", "dev", "test"), "2wiki": ("train", "dev", "test"),
               "hotpotqa": ("train", "validation", "test")}
ENT_LABELS = {"PERSON", "GPE", "ORG", "LOC", "FAC", "WORK_OF_ART", "EVENT", "PRODUCT", "NORP"}
DF_MIN, DF_MAX = 2, 25
KNN = 3
TOP = 1000
RRF_C = 60
KS = (2, 5, 10)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nows(s):
    return "".join(unicodedata.normalize("NFC", str(s)).split())


def write_json(p, obj):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def save_npz(p, **arrays):
    tmp = Path(str(p) + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, p)


# ── mapping ──────────────────────────────────────────────────────────────────


def map_passages(ds, corpus):
    """Setting node i -> package row: equal title, equal text with whitespace removed (lowest row on ties)."""
    want = defaultdict(list)
    for i, c in enumerate(corpus):
        want[c["title"]].append(i)
    row = np.full(len(corpus), -1, dtype=np.int64)
    exact = np.zeros(len(corpus), dtype=bool)
    with open(PKG / ds / "nodes.jsonl", encoding="utf-8") as f:
        for r, line in enumerate(f):
            if not want:
                break
            d = json.loads(line)
            hits = want.get(d.get("title"))
            if not hits:
                continue
            t = d.get("text", "")
            for i in hits:
                if row[i] < 0 and nows(t) == nows(corpus[i]["text"]):
                    row[i] = r
                    exact[i] = " ".join(t.split()) == " ".join(corpus[i]["text"].split())
    if (row < 0).any():
        raise SystemExit(f"{ds}: {int((row < 0).sum())} passages match no package row")
    return row, exact


def map_questions(ds, split, qs):
    ids = json.loads((PKG / ds / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    row_of = {q: j for j, q in enumerate(ids)}
    text = {}
    with open(PKG / ds / "queries" / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text[d["query_id"]] = d["question"]
    rows = []
    for q in qs:
        qid = q.get("id") or q.get("_id")
        if qid not in text or text[qid] != q["question"] or qid not in row_of:
            raise SystemExit(f"{ds}: question {qid} does not match the package's {split} split")
        rows.append(row_of[qid])
    return np.asarray(rows, dtype=np.int64), len(ids)


def golds_of(name, qs, corpus):
    by_tt = defaultdict(list)
    by_title = defaultdict(list)
    for i, c in enumerate(corpus):
        by_tt[(c["title"], nows(c["text"]))].append(i)
        by_title[c["title"]].append(i)
    out, unresolved, multi = [], 0, 0
    for q in qs:
        g = set()
        if name == "musique":
            for p in q["paragraphs"]:
                if p.get("is_supporting"):
                    hit = by_tt.get((p["title"], nows(p["paragraph_text"])), [])
                    unresolved += not hit
                    g.update(hit)
        else:
            for sf in q["supporting_facts"]:
                t = sf["title"] if isinstance(sf, dict) else sf[0]
                hit = by_title.get(t, [])
                unresolved += not hit
                multi += len(hit) > 1
                g.update(hit)
        out.append(np.asarray(sorted(g), dtype=np.int64))
    return out, unresolved, multi


# ── vectors ──────────────────────────────────────────────────────────────────


def dense_rows(ds, kind, rows):
    d = PKG / ds / "embeddings" / "dense" / kind
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    size = int(m["shard_size"])
    dim = np.load(d / "shard_00000.npy", mmap_mode="r").shape[1]
    out = np.empty((rows.size, dim), dtype=np.float32)
    for s in np.unique(rows // size):
        a = np.load(d / f"shard_{int(s):05d}.npy", mmap_mode="r")
        sel = np.flatnonzero(rows // size == s)
        out[sel] = np.asarray(a[rows[sel] - s * size], dtype=np.float32)
        del a
    return out


def splade_rows(ds, kind, rows):
    d = PKG / ds / "embeddings" / "splade" / kind
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    size = int(m["shard_size"])
    parts, order = [], []
    for s in np.unique(rows // size):
        with np.load(d / f"shard_{int(s):05d}.npz") as z:
            a = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(int(x) for x in z["shape"]))
        sel = np.flatnonzero(rows // size == s)
        parts.append(a[rows[sel] - s * size])
        order.append(sel)
        del a
    m_ = sp.vstack(parts).tocsr()
    inv = np.empty(rows.size, dtype=np.int64)
    inv[np.concatenate(order)] = np.arange(rows.size)
    return m_[inv]


def top_lists(scores, k):
    """Top-k columns per row, by score descending, ties by column ascending."""
    n = scores.shape[1]
    k = min(k, n)
    out = np.empty((scores.shape[0], k), dtype=np.int32)
    cols = np.arange(n)
    for i in range(scores.shape[0]):
        out[i] = np.lexsort((cols, -scores[i]))[:k]
    return out


# ── graphs ───────────────────────────────────────────────────────────────────


def induced_structural(ds, rows):
    z = np.load(PKG / ds / "graph" / "structural.npz")
    pos = np.full(int(max(rows.max(), max(z["src"].max(), z["dst"].max()))) + 1, -1, dtype=np.int64)
    pos[rows] = np.arange(rows.size)
    s, d = pos[z["src"]], pos[z["dst"]]
    keep = (s >= 0) & (d >= 0) & (s != d)
    rel = z["rel"][keep] if "rel" in z.files else np.zeros(int(keep.sum()), np.int16)
    e = np.unique(np.stack([s[keep], d[keep], rel.astype(np.int64)], 1), axis=0)
    return e[:, 0].astype(np.int32), e[:, 1].astype(np.int32), e[:, 2].astype(np.int16)


def ner_edges(texts):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    ent_docs = defaultdict(set)
    for i, doc in enumerate(nlp.pipe([str(t) for t in texts], batch_size=256)):
        for e in doc.ents:
            if e.label_ in ENT_LABELS and len(e.text) > 2:
                ent_docs[e.text.lower().strip()].add(i)
    w = defaultdict(float)
    for _e, docs in ent_docs.items():
        df = len(docs)
        if DF_MIN <= df <= DF_MAX:
            docs = sorted(docs)
            for a in range(len(docs)):
                for b in range(a + 1, len(docs)):
                    w[(docs[a], docs[b])] += 1.0 / df
    keys = sorted(w)
    src = np.asarray([k[0] for k in keys], dtype=np.int32)
    dst = np.asarray([k[1] for k in keys], dtype=np.int32)
    return src, dst, np.asarray([w[k] for k in keys], dtype=np.float32), len(ent_docs)


def knn_edges(x, k=KNN, block=2048):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n = x.shape[0]
    pairs = {}
    for lo in range(0, n, block):
        s = x[lo:lo + block] @ x.T
        for i in range(s.shape[0]):
            s[i, lo + i] = -np.inf
        nb = top_lists(s, k)
        for i in range(s.shape[0]):
            for j in nb[i]:
                a, b = sorted((lo + i, int(j)))
                pairs[(a, b)] = float(s[i, j])
    keys = sorted(pairs)
    return (np.asarray([p[0] for p in keys], np.int32), np.asarray([p[1] for p in keys], np.int32),
            np.asarray([pairs[p] for p in keys], np.float32))


def undirected_csr(n, fams):
    r, c = [], []
    for s, d in fams:
        r += [s, d]
        c += [d, s]
    a = sp.csr_matrix((np.ones(sum(x.size for x in r)), (np.concatenate(r), np.concatenate(c))), shape=(n, n))
    a.data[:] = 1
    return a


def stats(n, s, d, undirected):
    deg = np.bincount(np.concatenate([s, d]) if undirected else s, minlength=n) if s.size else np.zeros(n, int)
    touched = np.zeros(n, bool)
    touched[s] = touched[d] = True
    return {"edges": int(s.size), "mean_degree": round(float(np.bincount(np.concatenate([s, d]), minlength=n).mean()), 3),
            "isolated": int((~touched).sum()), "max_out_degree": int(deg.max()) if n else 0}


def recall_at(lists, golds, k):
    v = [np.isin(g, lists[i, :k]).mean() for i, g in enumerate(golds) if g.size]
    return round(float(np.mean(v)), 4)


def rrf(dl, sl, c=RRF_C, k=TOP):
    out = np.empty((dl.shape[0], k), dtype=np.int32)
    for i in range(dl.shape[0]):
        w = defaultdict(float)
        for r, j in enumerate(dl[i]):
            w[int(j)] += 1.0 / (c + r + 1)
        for r, j in enumerate(sl[i]):
            w[int(j)] += 1.0 / (c + r + 1)
        o = sorted(w, key=lambda j: (-w[j], j))[:k]
        out[i, :len(o)] = o
        out[i, len(o):] = -1
    return out


# ── build ────────────────────────────────────────────────────────────────────


def build(name):
    t0 = time.time()
    hp, split = SETTINGS[name]
    out = BENCH / name
    out.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((RAW / f"{hp}_corpus.json").read_text(encoding="utf-8"))
    qs = json.loads((RAW / f"{hp}.json").read_text(encoding="utf-8"))
    rows, exact = map_passages(name, corpus)
    log(f"{name}: {len(corpus)} passages mapped ({int(exact.sum())} exact, {int((~exact).sum())} whitespace-only)")
    qrows, nq_pkg = map_questions(name, split, qs)
    golds, unresolved, multi = golds_of(name, qs, corpus)
    log(f"{name}: {len(qs)} questions mapped; golds {sum(g.size for g in golds)} ({unresolved} unresolved refs)")
    # agreement with the package's golds
    pkg_gold = {}
    with open(PKG / name / "queries" / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pkg_gold[d["query_id"]] = d.get("gold_node_ids", [])
    nodeid_row = {}
    want_ids = {g for q in qs for g in pkg_gold.get(q.get("id") or q.get("_id"), [])}
    with open(PKG / name / "nodes.jsonl", encoding="utf-8") as f:
        for r, line in enumerate(f):
            if want_ids and '"node_id"' in line:
                d = json.loads(line)
                if d["node_id"] in want_ids:
                    nodeid_row[d["node_id"]] = r
    setting_of_row = {int(r): i for i, r in enumerate(rows)}
    agree = []
    for q, g in zip(qs, golds):
        pg = {setting_of_row.get(nodeid_row.get(x, -1), -1) for x in pkg_gold.get(q.get("id") or q.get("_id"), [])}
        agree.append(pg == set(g.tolist()))
    # vectors and first stage
    xd = dense_rows(name, "docs", rows)
    qd = dense_rows(name, "queries", qrows)
    dl = top_lists(qd @ xd.T, TOP)
    xs = splade_rows(name, "docs", rows)
    qsp = splade_rows(name, "queries", qrows)
    sl = top_lists(np.asarray((qsp @ xs.T).todense()), TOP)
    fl = rrf(dl, sl)
    log(f"{name}: first stage done ({time.time() - t0:.0f}s)")
    # graphs
    n = len(corpus)
    ss, sd, srel = induced_structural(name, rows)
    ns, nd, nw, n_ents = ner_edges([corpus[i]["text"] for i in range(n)])
    log(f"{name}: ner {ns.size} edges ({time.time() - t0:.0f}s)")
    ks, kd, kw = knn_edges(xd)
    save_npz(out / "graph_structural.npz", src=ss, dst=sd, rel=srel)
    save_npz(out / "graph_ner.npz", src=ns, dst=nd, weight=nw)
    save_npz(out / "graph_knn.npz", src=ks, dst=kd, weight=kw)
    save_npz(out / "first_stage.npz", dense=dl, splade=sl, rrf=fl, query_rows=qrows)
    write_json(out / "nodes.json", [{"package_row": int(r), "title": corpus[i]["title"], "exact": bool(exact[i])}
                                    for i, r in enumerate(rows)])
    write_json(out / "queries.json", [{"id": q.get("id") or q.get("_id"), "package_query_row": int(r),
                                       "golds": g.tolist(), "kind": str(q.get("type") or q.get("id", "").split("__")[0])}
                                      for q, r, g in zip(qs, qrows, golds)])
    a = undirected_csr(n, [(ss, sd), (ns, nd), (ks, kd)])
    within = {1: [], 2: []}
    for i, g in enumerate(golds):
        if not g.size:
            continue
        f0 = np.zeros(n, bool)
        f0[dl[i, :5]] = True
        f1 = f0 | (a @ f0.astype(np.float64) > 0)
        f2 = f1 | (a @ f1.astype(np.float64) > 0)
        within[1].append(f1[g].mean())
        within[2].append(f2[g].mean())
    rec = {"declared_in": "docs/B1_HIPPORAG2_ALIGNMENT.md", "script_sha256": sha(Path(__file__)),
           "setting": name, "source_files": {f: json.loads((RAW / "fetch.json").read_text())["files"][f]
                                             for f in (f"{hp}.json", f"{hp}_corpus.json")},
           "package": {"dataset": name, "split": split, "nodes_jsonl_sha256": json.loads((PKG / name / "build_info.json").read_text())["nodes_jsonl_sha256"],
                       "freeze_RECORD_SHA256": json.loads((PKG / "CANONICAL_FREEZE.json").read_text())["RECORD_SHA256"]},
           "passages": n, "passages_exact": int(exact.sum()), "passages_whitespace_only": int((~exact).sum()),
           "distinct_package_rows": int(np.unique(rows).size),
           "questions": len(qs), "questions_without_gold": int(sum(g.size == 0 for g in golds)),
           "gold_refs_unresolved": unresolved, "gold_titles_multi": multi,
           "gold_mean": round(float(np.mean([g.size for g in golds])), 3),
           "gold_agrees_with_package": round(float(np.mean(agree)), 4),
           "graph": {"structural": stats(n, ss, sd, False), "ner": stats(n, ns, nd, True) | {"entities": n_ents},
                     "knn": stats(n, ks, kd, True)},
           "structural_relations": dict(Counter(srel.tolist())),
           "first_stage_R": {lst: {f"R@{k}": recall_at(m, golds, k) for k in KS}
                             for lst, m in (("dense", dl), ("splade", sl), ("rrf", fl))},
           "gold_within_hops_of_dense_top5": {f"{h}": round(float(np.mean(v)), 4) for h, v in within.items()},
           "files": {p.name: sha(p) for p in sorted(out.glob("*")) if p.suffix in (".npz", ".json")
                     and p.name != "build.json"},
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out / "build.json", rec)
    log(f"{name}: R@5 dense {rec['first_stage_R']['dense']['R@5']} splade {rec['first_stage_R']['splade']['R@5']} "
        f"rrf {rec['first_stage_R']['rrf']['R@5']}; gold agreement {rec['gold_agrees_with_package']}; "
        f"edges s/n/k {ss.size}/{ns.size}/{ks.size} ({time.time() - t0:.0f}s)")
    return rec


def selftest():
    s = np.array([[1.0, 3.0, 3.0, 0.0]])
    assert top_lists(s, 3).tolist() == [[1, 2, 0]]
    x = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    a, b, w = knn_edges(x, k=1)
    assert sorted(zip(a.tolist(), b.tolist())) == [(0, 1), (2, 3)], (a, b)
    dl = np.array([[0, 1]]); sl = np.array([[1, 2]])
    assert rrf(dl, sl, k=3).tolist() == [[1, 0, 2]]
    assert recall_at(np.array([[3, 1, 2]]), [np.array([1, 9])], 2) == 0.5
    assert nows(' a  "b c" ') == nows('a "bc"')
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("build",))
    ap.add_argument("--datasets", nargs="+", default=list(SETTINGS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.stage is None:
        return selftest()
    for name in a.datasets:
        build(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
