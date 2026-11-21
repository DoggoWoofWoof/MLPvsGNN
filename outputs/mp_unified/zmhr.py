"""MHR1 (docs/MHR1_MULTIHOPRAG_PACKAGE.md): MultiHop-RAG packaged like the six CRAG datasets, encoded on e3v (frozen beside),
read at the first stage and at the graph level. Nothing trains; no filed module is edited (zenc, zenc3, zenc3v, zenc4, zenc1,
zenc1b2, zd84, zul1, zap1 and zu2b are imported and, in this process only, pointed at this package).

    python outputs/mp_unified/zmhr.py selftest                     laptop
    python outputs/mp_unified/zmhr.py package                      laptop: section 2 with P1-P4
    python outputs/mp_unified/zmhr.py splade [--threads 4]         laptop CPU: the record's SPLADE contract
    python outputs/mp_unified/zmhr.py encode --enc gtemodern|granitesm [--threads 4] [--cpu]   host GPU: GP, then zenc's encode
    python outputs/mp_unified/zmhr.py frozen [--threads 4] [--cpu] host GPU: the frozen arm into the package
                                       (--cpu: section 3.1, the host CPU, float32 weights, 8,192 tokens a batch)
    python outputs/mp_unified/zmhr.py vectors [--threads 1]        host CPU: zenc3v's fusion
    python outputs/mp_unified/zmhr.py fc [--threads 1]             host CPU: FC on every row
    python outputs/mp_unified/zmhr.py cuts [--threads 1]           host CPU: which articles each window cut
    python outputs/mp_unified/zmhr.py first [--threads 4]          laptop: section 4
    python outputs/mp_unified/zmhr.py links [--threads 4]          laptop: section 5's families and stores
    python outputs/mp_unified/zmhr.py grade                        laptop: section 5's measures and call
    python outputs/mp_unified/zmhr.py report                       laptop: outputs/mhr1/report.md
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
if "--cpu" in sys.argv:                          # section 3.1: hide the card before anything imports torch
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = ""
import zenc as E  # noqa: E402  (reads --threads from sys.argv before numpy loads)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DOC = "docs/MHR1_MULTIHOPRAG_PACKAGE.md"
DS = "multihoprag"
ROOTWS = HERE.parents[1]
PKG = ROOTWS / "outputs" / "mhr1" / "package"
FREEZE = PKG / "PACKAGE_FREEZE.json"
OUT = ROOTWS / "outputs" / "mhr1"
GP_DIR = OUT / "gp"
SPLITS = ("train", "dev", "test")
FR_TRAIN, FR_DEV = (4, 10), (1, 10)                 # floor(40%), floor(10%) of each type, the rest test
SEED, BOOT = 0, 2000
FROZEN_REV = "a9af15a6372d7d6b25e9fb07c2ccb9e1fe645644"
SPL, SPL_REV, SPL_LEN, SPL_BATCH = "naver/splade-cocondenser-ensembledistil", "49cf4c7b0db5b870a401ddf5e2669993ef3699c7", 256, 64
COV_K = (2, 4, 5, 10, 20)
ARMS = ("e3v", "frozen")
LABELS = ("S", "R.mention", "R.name_comention", "R.title_comention", "K.knn")    # D83's vocabulary, ids 0-4
DQ = "row j is the dense vector of query j == queries/query_ids.json[j] == the j-th line of the split files " \
     "['train', 'dev', 'test'] concatenated in that order"


def log(msg):
    print(f"[zmhr {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zmhr: {msg}")


def rel(p):
    try:
        return Path(p).resolve().relative_to(ROOTWS).as_posix()
    except ValueError:
        return str(p)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_sha():
    return {"zmhr.py": E.sha256(__file__)}


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def strip_ws(t):
    """The benchmark's normalisation (retrieval_evaluate.py): spaces and newlines removed."""
    return str(t).replace(" ", "").replace("\n", "")


# ── the package root, for zenc's readers in this process ─────────────────────

def mhr_root():
    if not FREEZE.is_file():
        stop(f"{rel(FREEZE)} missing: run `zmhr.py package` first")
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    got = E.sha256(PKG / DS / "DATASET.json")
    if f.get("RECORD") != "MHR1_PACKAGE" or f.get("dataset_json_sha256") != got:
        stop(f"the package's DATASET.json hashes to {got[:12]}, its freeze names {str(f.get('dataset_json_sha256'))[:12]}")
    return PKG


def point_zenc():
    E.served_root = mhr_root                     # zenc's encode, zenc3v's vectors and zenc4's FC look it up at call time


def nodes_of(root=None):
    root = root or mhr_root()
    with open(root / DS / "nodes.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def queries_of(root=None):
    root = root or mhr_root()
    out = []
    for sp in SPLITS:
        with open(root / DS / "queries" / f"{sp}.jsonl", encoding="utf-8") as f:
            out += [json.loads(x) for x in f]
    ids = json.loads((root / DS / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    if [q["query_id"] for q in out] != ids:
        stop("the split files' ids are not query_ids.json")
    return out


# ── section 2: the package ───────────────────────────────────────────────────

def split_rule(types, seed=SEED):
    """By question, stratified by type (sorted); one generator drawn type after type; floor 40% train, floor 10% dev."""
    rng = np.random.default_rng(seed)
    out = [None] * len(types)
    for t in sorted(set(types)):
        idx = np.asarray([k for k, x in enumerate(types) if x == t], np.int64)
        p = idx[rng.permutation(idx.size)]
        ntr = idx.size * FR_TRAIN[0] // FR_TRAIN[1]
        ndv = idx.size * FR_DEV[0] // FR_DEV[1]
        for k in p[:ntr]:
            out[int(k)] = "train"
        for k in p[ntr:ntr + ndv]:
            out[int(k)] = "dev"
        for k in p[ntr + ndv:]:
            out[int(k)] = "test"
    return out


def write_lines(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return {"file": rel(p), "bytes": p.stat().st_size, "sha256": E.sha256(p), "n": len(rows)}


def dense_rows_manifest(kind, n):
    return {"dataset": DS, "kind": kind, "model": "dense", "n_rows": n, "shard_size": E.SHARD, "n_shards": 1,
            "row_semantics": ("row i is the dense vector of position i == line i of nodes.jsonl" if kind == "docs" else DQ),
            "format": {"file": "shard_%05d.npy", "dtype": "float16", "shape": "(rows, 1536)",
                       "normalised": "unit L2 (|norm-1| <= 0.01)"},
            "encoder": {"encoder": E.ENCODERS["frozen"]["repo"], "revision": FROZEN_REV, "dim": 1536, "dtype": "float16",
                        "query_instruction": E.GTE_QINSTR},
            "shards": [{"shard": 0, "rows": n, "file": "shard_00000.npy", "bytes": None, "sha256": None}],
            "status": "rows only: `zmhr.py frozen` writes the vectors and their sha256", "page": DOC}


def cmd_package(a):
    import zd84 as D
    t0 = time.time()
    arts, qs, raw_sha = D.load()                                      # P1: D84's G1
    ids, golds, types, items = D.golds_of(arts, qs)                   # P1: D84's G2
    n = len(arts)
    at = {D.norm(x["url"]): i for i, x in enumerate(arts)}
    nodes = []
    for x in arts:
        ch = sha_text(x["body"])
        nodes.append({"node_id": f"{DS}:{ch[:24]}", "title": x["title"], "text": x["body"], "content_hash": ch,
                      "source": x.get("source"), "category": x.get("category"), "published_at": x.get("published_at"),
                      "url": x.get("url"), "author": x.get("author")})
    nid = [x["node_id"] for x in nodes]
    if len(set(nid)) != n:
        stop("two articles share a node id (identical bodies)")
    bodies = [strip_ws(x["body"]) for x in arts]
    p3_bad, p3_multi, facts = [], 0, 0
    rows = []
    for j, g, t in zip(ids, golds, types):
        q = qs[j]
        ev, order = [], []
        for e in q["evidence_list"]:
            i = at[D.norm(e.get("url"))]
            f = strip_ws(e.get("fact", ""))
            facts += 1
            if not f or f not in bodies[i]:
                p3_bad.append((j, i))
            p3_multi += int(sum(f in b for b in bodies) > 1)
            if i not in order:
                order.append(i)
            ev.append({"node_id": nid[i], "fact": e.get("fact"), "title": e.get("title"), "source": e.get("source"),
                       "url": e.get("url"), "published_at": e.get("published_at")})
        if sorted(order) != g.tolist():
            stop(f"query {j}: its gold order {order} is not D84's set {g.tolist()}")
        rows.append({"query_id": f"mhr_q{j:04d}", "question": q["query"], "answer": q.get("answer"), "question_type": t,
                     "gold_node_ids": [nid[i] for i in order], "evidence": ev, "n_golds": len(order),
                     "dataset": DS, "source_position": j})
    if p3_bad:
        stop(f"P3: {len(p3_bad)} facts are not inside their named article (e.g. {p3_bad[:5]})")
    sp = split_rule(types)
    root = PKG / DS
    files = {"nodes": write_lines(root / "nodes.jsonl", nodes)}
    qorder = []
    for s in SPLITS:
        part = [r for r, x in zip(rows, sp) if x == s]
        files[f"queries/{s}"] = write_lines(root / "queries" / f"{s}.jsonl", part)
        qorder += [r["query_id"] for r in part]
    E.wjson(root / "queries" / "query_ids.json", qorder)
    files["query_ids"] = {"file": rel(root / "queries" / "query_ids.json"),
                          "sha256": E.sha256(root / "queries" / "query_ids.json"), "n": len(qorder)}
    for kind, m in (("docs", n), ("queries", len(rows))):
        mp = root / "embeddings" / "dense" / kind / "manifest.json"
        if mp.is_file() and json.loads(mp.read_text(encoding="utf-8")).get("shards", [{}])[0].get("sha256"):
            stop(f"{rel(mp)} already names frozen vectors: the package is built once")
        E.wjson(mp, dense_rows_manifest(kind, m))
    by = {s: {t: sum(1 for x, y in zip(sp, types) if x == s and y == t) for t in sorted(set(types))} for s in SPLITS}
    ds_json = {"RECORD": "MHR1_PACKAGE", "dataset": DS, "declared_in": DOC, "n_nodes": n,
               "n_queries": len(rows), "queries_by_split": {s: sum(by[s].values()) for s in SPLITS},
               "queries_by_split_and_type": by,
               "layout": {"nodes": "nodes.jsonl", "queries": "queries/<split>.jsonl + queries/query_ids.json",
                          "embeddings": "embeddings/<dense|splade>/<docs|queries>/shard_XXXXX.<npy|npz> + manifest.json",
                          "graph": "graph/knn.npz", "retrieval_cache": "retrieval_cache/<dense|splade>_top1000.npz"},
               "nodes": {**files["nodes"], "fields": sorted(nodes[0]),
                         "position_semantics": "line i of nodes.jsonl is position i: embeddings row i, graph endpoint i, "
                                               "cache id i; position i is the corpus file's article i (D84's node i)"},
               "queries": {"n": len(rows), "splits_in_row_order": list(SPLITS),
                           "by_split": {s: files[f"queries/{s}"] for s in SPLITS}, "query_ids": files["query_ids"],
                           "fields": sorted(rows[0]), "dropped_null": len(qs) - len(rows)},
               "split_rule": "by question, stratified by question_type (sorted); numpy default_rng(0) permutes each "
                             "type's questions (file order), drawn type after type; floor 40% train, floor 10% dev, the "
                             "rest test; rows train, dev, test, each in file order",
               "raw": {"repo": "yixuantt/MultiHopRAG", "revision": D.REVISION, "files_sha256": raw_sha,
                       "licence": "ODC-By"}}
    E.wjson(root / "DATASET.json", ds_json)
    E.wjson(FREEZE, {"RECORD": "MHR1_PACKAGE", "dataset": DS, "dataset_json_sha256": E.sha256(root / "DATASET.json"),
                     "declared_in": DOC, "utc": utc()})
    # P2: D84 reproduced from the package
    pn = nodes_of()
    F, st = D.rule_forms([{"title": x["title"], "body": x["text"]} for x in pn])
    d84 = json.loads((ROOTWS / "outputs" / "d84" / "grade.json").read_text(encoding="utf-8"))
    p2 = {"F0T": int(F["F0T"].size), "CN_all": int(F["CN_all"].size), "d84_F0T": d84["pairs"]["F0T"],
          "d84_CN_all": d84["pairs"]["CN_all"], "kept": len(rows), "d84_kept": d84["kept_queries"],
          "gold_counts": {str(k): sum(1 for r in rows if r["n_golds"] == k) for k in (1, 2, 3, 4)},
          "d84_gold_counts": d84["gold_counts"]}
    p2["pass"] = (p2["F0T"] == p2["d84_F0T"] and p2["CN_all"] == p2["d84_CN_all"] and p2["kept"] == p2["d84_kept"]
                  and all(p2["gold_counts"].get(k, 0) == v for k, v in d84["gold_counts"].items()))
    # P4: zenc's readers
    point_zenc()
    qt = E.query_texts(mhr_root(), DS)
    ch = list(E.doc_chunks(mhr_root(), DS))
    p4 = {"query_rows": len(qt), "doc_shards": len(ch), "doc_rows": sum(len(c[1]) for c in ch),
          "first_question_equal": qt[0] == queries_of()[0]["question"],
          "texts_equal": [t for c in ch for t in c[1]] == [x["text"] for x in pn]}
    p4["pass"] = p4["query_rows"] == len(rows) and p4["doc_shards"] == 1 and p4["doc_rows"] == n and \
        p4["first_question_equal"] and p4["texts_equal"]
    rec = {"stage": "MHR1 package", "declared_in": DOC, "dataset": DS, "nodes": n, "queries": len(rows),
           "dropped_null": len(qs) - len(rows), "by_split_and_type": by,
           "P1": "pass", "P2": p2, "P3": {"facts": facts, "inside_named_article": facts - len(p3_bad),
                                          "facts_inside_more_than_one_article": p3_multi, "pass": not p3_bad},
           "P4": p4, "rule_build": st, "files": files, "dataset_json_sha256": E.sha256(root / "DATASET.json"),
           "body_chars": {"min": min(len(x["text"]) for x in pn), "median": int(np.median([len(x["text"]) for x in pn])),
                          "max": max(len(x["text"]) for x in pn)},
           "seconds": round(time.time() - t0, 1), "utc": utc(), **code_sha()}
    E.wjson(OUT / "package.json", rec)
    log(f"package: {n} nodes, {len(rows)} queries {rec['by_split_and_type']}; P2 {p2['pass']} P3 {not p3_bad} "
        f"P4 {p4['pass']} -> {rel(root)}")
    if not (p2["pass"] and p4["pass"]):
        stop(f"P2 or P4 failed: {json.dumps({'P2': p2, 'P4': p4})}")
    return 0


# ── section 3: the encodes ───────────────────────────────────────────────────

def cmd_splade(a):
    import scipy.sparse as sp
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    torch.set_num_threads(int(a.threads))
    root = mhr_root()
    path = snapshot_download(SPL, revision=SPL_REV, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mdl = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True).eval()
    texts = {"docs": [x["text"] for x in nodes_of(root)], "queries": [q["question"] for q in queries_of(root)]}
    for kind in ("docs", "queries"):
        t0 = time.time()
        X = texts[kind]
        r_, c_, d_ = [], [], []
        for i in range(0, len(X), SPL_BATCH):
            b = [str(t) for t in X[i:i + SPL_BATCH]]
            enc = tok(b, return_tensors="pt", truncation=True, max_length=SPL_LEN, padding=True)
            with torch.no_grad():
                v = torch.log1p(torch.relu(mdl(**enc).logits)).max(dim=1).values.numpy()
            for j in range(v.shape[0]):
                nz = np.nonzero(v[j])[0]
                r_ += [i + j] * len(nz)
                c_ += nz.tolist()
                d_ += v[j][nz].tolist()
        mat = sp.csr_matrix((np.asarray(d_, np.float32), (r_, c_)), shape=(len(X), mdl.config.vocab_size), dtype=np.float32)
        if mat.shape[0] != len(X) or not np.isfinite(mat.data).all() or (np.diff(mat.indptr) == 0).any():
            stop(f"splade {kind}: a damaged or empty row")
        cut = int(sum(len(x) > SPL_LEN for x in tok(X, add_special_tokens=True, truncation=False)["input_ids"]))
        base = root / DS / "embeddings" / "splade" / kind
        base.mkdir(parents=True, exist_ok=True)
        npz = base / "shard_00000.npz"
        tmp = base / "shard_00000.tmp.npz"
        sp.save_npz(tmp, mat)
        os.replace(tmp, npz)
        E.wjson(base / "manifest.json", {
            "dataset": DS, "kind": kind, "model": "splade", "n_rows": len(X), "shard_size": E.SHARD, "n_shards": 1,
            "row_semantics": (f"row i is the splade vector of position i == line i of nodes.jsonl" if kind == "docs"
                              else DQ.replace("dense", "splade")),
            "format": {"file": "shard_%05d.npz", "keys": "scipy.sparse.save_npz CSR float32", "shape": list(mat.shape)},
            "encoder": {"encoder": SPL, "revision": SPL_REV, "max_length": SPL_LEN, "batch": SPL_BATCH, "device": "cpu",
                        "dtype": "float32", "threads": int(a.threads),
                        "pooling": "log1p(relu(logits)).max(dim=1) over every position, padding included (the record's)"},
            "shards": [{"shard": 0, "rows": len(X), "file": npz.name, "bytes": npz.stat().st_size, "sha256": E.sha256(npz)}],
            "cut_at_256": cut, "nnz_mean": round(float(np.diff(mat.indptr).mean()), 1),
            "seconds": round(time.time() - t0, 1), "utc": utc(), "page": DOC, **code_sha()})
        log(f"splade {kind}: {len(X)} rows, {cut} cut at {SPL_LEN}, nnz {np.diff(mat.indptr).mean():.0f} "
            f"({time.time() - t0:.0f} s)")
    return 0


def zmods():
    import zenc3 as Z3
    import zenc3v as V
    return Z3, V


CPU_BUDGET = 8192                                 # section 3.1: tokens per batch on the CPU (one long article alone)


def peak_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", getattr(mi, "rss", 0)) / 2 ** 30, 2)
    except Exception:  # noqa: BLE001
        return None


def cpu_guard(a):
    """Section 3.1: with --cpu the card is hidden (CUDA_VISIBLE_DEVICES set before torch loads); refuse if it shows."""
    import torch
    if not a.cpu:
        return "cuda"
    if torch.cuda.is_available():
        stop("--cpu, but torch still sees a GPU")
    torch.set_num_threads(int(a.threads))
    return "cpu"


def cmd_encode(a):
    Z3, V = zmods()
    point_zenc()
    if a.enc not in V.PARTS:
        stop(f"--enc must be one of e3v's parts {V.PARTS}")
    dev = cpu_guard(a)
    rev = Z3.pinned(a.enc)
    t0 = time.time()
    if dev == "cuda":
        Z3.DS = (DS,)
        old = Z3.OUT
        Z3.OUT = GP_DIR                           # GP's record goes to outputs/mhr1/gp/, not over E0N's
        try:
            Z3.gp_check(a.enc, rev)
        finally:
            Z3.OUT = old
        log(f"GP passed for {a.enc}")
    else:
        E.wjson(GP_DIR / f"gp-{a.enc}.json", {"stage": "MHR1 GP", "declared_in": DOC + " section 3.1", "enc": a.enc,
                                              "revision": rev, "pass": None,
                                              "not_run": "CPU float32 (section 3.1): float32 is GP's reference",
                                              "utc": utc(), **code_sha()})
        E.ENCODERS[a.enc]["budget"] = CPU_BUDGET  # this process only
        log(f"{a.enc} on the CPU ({a.threads} threads, budget {CPU_BUDGET}); GP not run (section 3.1)")
    E.cmd_encode(argparse.Namespace(enc=a.enc, ds=DS, kinds="queries,docs", cpu=dev == "cpu"))
    E.wjson(OUT / f"run-{a.enc}.json", {"enc": a.enc, "device": dev, "threads": int(a.threads),
                                        "budget": int(E.ENCODERS[a.enc]["budget"]), "seconds": round(time.time() - t0, 1),
                                        "peak_working_set_gb": peak_gb(), "utc": utc(), **code_sha()})
    return 0


def cmd_frozen(a):
    import torch
    from huggingface_hub import snapshot_download
    point_zenc()
    root = mhr_root()
    dev = cpu_guard(a)
    if dev == "cuda" and not torch.cuda.is_available():
        stop("no GPU")
    rev = Path(snapshot_download(E.ENCODERS["frozen"]["repo"], local_files_only=True)).name
    if rev != FROZEN_REV:
        stop(f"the cached frozen snapshot is {rev}, not {FROZEN_REV}")
    t0 = time.time()
    m = E.load_model("frozen", dev)
    load_s = round(time.time() - t0, 1)
    budget = CPU_BUDGET if dev == "cpu" else None
    texts = {"docs": [x["text"] for x in nodes_of(root)], "queries": [q["question"] for q in queries_of(root)]}
    for kind in ("queries", "docs"):
        t1 = time.time()
        v, st = E.encode_texts(m, "frozen", texts[kind], kind, budget=budget)
        bad = E.damaged(v)
        if bad:
            stop(f"frozen {kind}: {bad} damaged rows")
        base = root / DS / "embeddings" / "dense" / kind
        npy, tmp = base / "shard_00000.npy", base / "shard_00000.tmp.npy"
        np.save(tmp, v)
        os.replace(tmp, npy)
        man = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        man["shards"] = [{"shard": 0, "rows": int(v.shape[0]), "file": npy.name, "bytes": npy.stat().st_size,
                          "sha256": E.sha256(npy)}]
        man["encoder"].update({"device": dev, "weights_dtype": "float32" if dev == "cpu" else "float16",
                               "threads": int(a.threads) if dev == "cpu" else None,
                               "max_seq_length": int(m.max_seq_length), "load_s": load_s,
                               "query_prompt": E.query_prompt(m, "frozen") if kind == "queries" else "",
                               "document_prompt": "", "path": "zenc.load_model + zenc.encode_texts (G0's)"})
        man.update({"status": "vectors written", "seconds": round(time.time() - t1, 1), **st,
                    "peak_working_set_gb": peak_gb(), "utc": utc(), **code_sha()})
        E.wjson(base / "manifest.json", man)
        log(f"frozen {kind}: {v.shape[0]} rows, cut {st['cut']}, max_len {st['max_len']} ({time.time() - t1:.0f} s)")
    return 0


def cmd_vectors(a):
    Z3, V = zmods()
    point_zenc()
    V.DS = V.DS + (DS,)
    return V.cmd_vectors(argparse.Namespace(ds=DS))


def cmd_fc(a):
    import zenc4 as Z4
    point_zenc()
    t0 = time.time()
    r = Z4.fc_ds(DS, rows_per_shard=10 ** 9)
    r.update({"stage": "MHR1: FC (every row)", "declared_in": DOC + " section 3", "seconds": round(time.time() - t0, 1),
              "utc": utc(), **code_sha()})
    E.wjson(OUT / "fc.json", r)
    log(f"FC {r['verdict']}")
    return 0 if r["verdict"] == "PASS" else 3


def cmd_cuts(a):
    """Section 3's windows, per article: which articles each encoder cut. Token counts as zenc.encode_texts counts them
    (the model's tokenizer, special tokens added, no truncation, against its max_seq_length); each count must equal the
    encode's recorded `cut`. Host CPU (the parts' tokenizers are cached on the host only)."""
    from transformers import AutoTokenizer
    Z3, V = zmods()
    point_zenc()
    root = mhr_root()
    texts = [x["text"] for x in nodes_of(root)]
    rec = {"stage": "MHR1 cut articles", "declared_in": DOC + " section 3", "dataset": DS, "nodes": len(texts),
           "cut_positions": {}, "limit": {}, "max_tokens": {}, "encode_cut": {}}
    for enc in ("gtemodern", "granitesm", "frozen"):
        if enc == "frozen":
            from huggingface_hub import snapshot_download
            rev = Path(snapshot_download(E.ENCODERS["frozen"]["repo"], local_files_only=True)).name
            if rev != FROZEN_REV:
                stop(f"the cached frozen snapshot is {rev}, not {FROZEN_REV}")
            man = json.loads((root / DS / "embeddings" / "dense" / "docs" / "manifest.json").read_text(encoding="utf-8"))
        else:
            rev = Z3.pinned(enc)
            man = json.loads((E.OUT / enc / DS / "docs" / "manifest.json").read_text(encoding="utf-8"))
        tok = AutoTokenizer.from_pretrained(E.ENCODERS[enc]["repo"], revision=rev, local_files_only=True,
                                            trust_remote_code=E.ENCODERS[enc].get("trust", False))
        lim = int(man["encoder"]["max_seq_length"])
        lens = np.asarray([len(x) for x in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]], np.int64)
        cp = np.flatnonzero(lens > lim)
        if int(cp.size) != int(man["cut"]):
            stop(f"{enc}: {cp.size} articles over {lim} tokens, its encode recorded {man['cut']} cut")
        rec["cut_positions"][enc] = cp.tolist()
        rec["limit"][enc], rec["max_tokens"][enc], rec["encode_cut"][enc] = lim, int(lens.max()), int(man["cut"])
        log(f"{enc}: {cp.size} of {len(texts)} articles over {lim} tokens (max {lens.max()})")
    rec.update({"utc": utc(), **code_sha()})
    E.wjson(OUT / "cuts.json", rec)
    return 0


# ── section 4: the first stage ───────────────────────────────────────────────

def coverage(top, golds, k):
    return np.asarray([np.isin(top[q, :k], g).sum() / g.size for q, g in enumerate(golds)], np.float64)


def bench_measures(ranked, facts, holds):
    """The benchmark's loop (retrieval_evaluate.py) on article lists: holds(q, doc, fact) says the doc's text holds the
    (normalised) fact. Per question: hits@4, hits@10, MAP@10 (its form), MRR@10."""
    nq = len(facts)
    h4, h10, mp, mr = (np.zeros(nq) for _ in range(4))
    for q in range(nq):
        gold = facts[q]
        found, ap, first = [], 0.0, None
        for rank, d in enumerate(ranked[q][:11], start=1):
            if any(holds(d, g) for g in gold):
                if rank <= 10:
                    h10[q] = 1
                    if first is None:
                        first = rank
                    if rank <= 4:
                        h4[q] = 1
                    c = 0
                    for g in gold:
                        if holds(d, g) and g not in found:
                            c += 1
                            found.append(g)
                    ap += c / rank
        mp[q] = ap / min(len(gold), 10)
        mr[q] = 1.0 / first if first else 0.0
    return {"hits4": h4, "hits10": h10, "map10": mp, "mrr10": mr}


def paired(x, y, boot=BOOT, seed=SEED):
    """Mean of x - y (points) and its paired 95% interval over questions, with zap1's call."""
    import zap1 as Z
    d, lo, hi = Z.boot_diff(x, y, np.arange(len(x)), len(x), boot, seed)
    return {"diff": Z.pct(d), "lo": Z.pct(lo), "hi": Z.pct(hi), "call": Z.call(lo, hi)}


def rank_all(S):
    """Every row's columns by score descending, then column ascending."""
    cols = np.broadcast_to(np.arange(S.shape[1]), S.shape)
    return np.lexsort((cols, -S), axis=1)


def cmd_first(a):
    import scipy.sparse as sp
    point_zenc()
    root = mhr_root()
    t0 = time.time()
    nodes, qs = nodes_of(root), queries_of(root)
    n = len(nodes)
    pos = {x["node_id"]: i for i, x in enumerate(nodes)}
    golds = [np.asarray(sorted(pos[g] for g in q["gold_node_ids"]), np.int64) for q in qs]
    flat = np.concatenate(golds)
    off = np.r_[0, np.cumsum([g.size for g in golds])].astype(np.int64)
    gt = np.asarray([g.size for g in golds], np.int64)
    bodies = [strip_ws(x["text"]) for x in nodes]
    facts = [[strip_ws(e["fact"]) for e in q["evidence"]] for q in qs]
    hold = {}

    def holds(d, f):
        k = (d, f)
        if k not in hold:
            hold[k] = f in bodies[d]
        return hold[k]

    def spl(kind):
        return sp.load_npz(root / DS / "embeddings" / "splade" / kind / "shard_00000.npz").astype(np.float32)

    Ssp = np.asarray((spl("queries") @ spl("docs").T).todense(), np.float32)
    lists = {"splade": rank_all(Ssp)}
    scores = {}
    for arm in ARMS:
        Q = np.concatenate([np.load(p).astype(np.float32) for p in E.vec_shards(root, arm, DS, "queries")])
        ids, sc, _ = E.search(Q, E.vec_shards(root, arm, DS, "docs"), k=n)
        lists[arm], scores[arm] = ids, sc
        lists[f"{arm}+splade"] = E.rrf_top(ids, lists["splade"], max(COV_K))      # every list n deep; RRF top 20
    import zap1 as Z
    types = np.asarray([q["question_type"] for q in qs])
    nby = json.loads((root / DS / "DATASET.json").read_text(encoding="utf-8"))["queries_by_split"]
    split = np.asarray(sum(([s] * int(nby[s]) for s in SPLITS), []))
    if split.size != len(qs):
        stop("the split counts do not cover the questions")
    per = {}
    for name, L in lists.items():
        m = {f"d{k}": coverage(L, golds, k) for k in COV_K}
        m.update(bench_measures(L, facts, holds))
        per[name] = m
    for arm in ARMS:
        per[arm]["rrf5"] = per[f"{arm}+splade"]["d5"]
    # cut articles (section 3's windows): outputs/mhr1/cuts.json, `zmhr.py cuts` (its counts equal the encodes')
    cr = json.loads((OUT / "cuts.json").read_text(encoding="utf-8"))
    cpos = {"e3v": sorted(set(cr["cut_positions"]["gtemodern"]) | set(cr["cut_positions"]["granitesm"])),
            "frozen": cr["cut_positions"]["frozen"], "splade": list(range(n))}
    cut = {}
    on_cut = {}
    for arm, cp in cpos.items():
        isc = np.zeros(n, bool)
        isc[np.asarray(cp, np.int64)] = True
        slots = isc[flat]
        on_cut[arm] = np.asarray([isc[g].any() for g in golds])
        cut[arm] = {"articles": int(isc.sum()), "gold_slots": int(slots.sum()), "gold_slots_pct": Z.pct(slots.mean()),
                    "questions_with_a_cut_gold": int(on_cut[arm].sum())}
    cut["cuts_json_sha256"] = E.sha256(OUT / "cuts.json")
    sel = {"all": np.ones(len(qs), bool), "test": split == "test",
           **{t: types == t for t in sorted(set(types.tolist()))}, **{f"golds{k}": gt == k for k in (2, 3, 4)},
           "a_gold_cut_by_e3v": on_cut["e3v"], "no_gold_cut_by_e3v": ~on_cut["e3v"]}
    sel = {k: v for k, v in sel.items() if v.sum() > 0}

    def tab(mask):
        out = {}
        for name, m in per.items():
            out[name] = {k: Z.pct(v[mask].mean()) for k, v in m.items()}
        out["e3v_minus_frozen"] = {k: paired(per["e3v"][k][mask], per["frozen"][k][mask])
                                   for k in per["e3v"] if k in per["frozen"]}
        return out

    res = {s: {"questions": int(m.sum()), **tab(m)} for s, m in sel.items()}
    head = res["all"]["e3v_minus_frozen"]
    fails = [{"cell": c, **head[c]} for c in ("d5", "rrf5") if head[c]["call"] == "BELOW"]
    call = "e3v HOLDS ON MULTIHOP-RAG" if not fails else "e3v LOSES ON MULTIHOP-RAG"
    d = OUT / "first"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "lists.npz", **{k.replace("+", "_"): v[:, :20].astype(np.int32) for k, v in lists.items()},
             golds_flat=flat, golds_off=off)
    rc = root / DS / "retrieval_cache"
    rc.mkdir(parents=True, exist_ok=True)
    np.savez(rc / "dense_top1000.npz", ids=lists["frozen"].astype(np.int32), scores=scores["frozen"].astype(np.float16))
    np.savez(rc / "splade_top1000.npz", ids=lists["splade"].astype(np.int32),
             scores=np.take_along_axis(Ssp, lists["splade"], 1).astype(np.float32))
    rec = {"stage": "MHR1 first stage", "declared_in": DOC + " section 4", "dataset": DS, "nodes": n,
           "questions": len(qs), "call": call, "fails": fails, "cut": cut, "threads": E.THREADS,
           "vectors": {arm: E.sha256(E.vec_dir(root, arm, DS, "docs") / "manifest.json") for arm in ARMS},
           "splade_manifest_sha256": E.sha256(root / DS / "embeddings" / "splade" / "docs" / "manifest.json"),
           "dataset_json_sha256": E.sha256(root / DS / "DATASET.json"), "lists_sha256": E.sha256(d / "lists.npz"),
           "cache_sha256": {f: E.sha256(rc / f) for f in ("dense_top1000.npz", "splade_top1000.npz")},
           "reads": res, "seconds": round(time.time() - t0, 1), "utc": utc(), **code_sha()}
    E.wjson(OUT / "first.json", rec)
    a5 = res["all"]
    log(f"{call}: d5 e3v {a5['e3v']['d5']} frozen {a5['frozen']['d5']} gap {head['d5']}; rrf5 gap {head['rrf5']}")
    return 0


# ── section 5: the links ─────────────────────────────────────────────────────

def w2s_keys(titles, texts, X):
    """U2b's rule `new`, variant W2, form S, with the arm's float32 rows for the choice among same-named targets."""
    import zu2b as U
    import zul1 as UL
    n = len(titles)
    rule = U.NEW_RULE
    S = U.Surfaces(titles, rule)
    cap, low = U.stat_pass(S, texts, titles, rule)
    common = low > cap
    desc = np.asarray([U.is_descriptive(t, x) for t, x in zip(titles, texts)], bool)
    edges, stats = U.link_pass(S, texts, titles, rule, common, lambda p: X[np.asarray(p, np.int64)],
                               {"W2": U.VARIANTS["W2"]})
    ms, md = (x.astype(np.int64) for x in edges["W2"])
    g = {}
    group = np.asarray([g.setdefault(U.norm_title(t, rule).strip(), len(g)) for t in titles], np.int64)
    ss, sd = U.sibling_pairs(group, desc)
    keys = UL.pair_keys(np.concatenate([ms, ss]), np.concatenate([md, sd]), n)
    return keys, {"mention_links": int(ms.size), "sibling_links": int(ss.size), "pairs": int(keys.size),
                  "descriptive_nodes": int(desc.sum()), "surfaces": len(S.surf), "common_forms": int(common.sum()),
                  **{f"W2_{k}": v for k, v in stats["W2"].items()}}


def k_family(root, arm, texts):
    """The record's K on the arm's document rows (zenc1's and zenc1b2's pieces, unedited)."""
    import zenc1 as Z1
    import zenc1b2 as B2
    n = len(texts)
    dig = np.frombuffer(b"".join(Z1.text_digest(t) for t in texts), np.uint8).reshape(n, 16)
    pos2g, g_first = Z1.group_digests(dig)
    paths = E.vec_shards(root, arm, DS, "docs")
    offs = Z1.shard_offsets(paths)
    if int(offs[-1]) != n:
        stop(f"{arm}: {int(offs[-1])} rows for {n} nodes")
    eg = np.arange(g_first.size, dtype=np.int64)
    nbr, sc = Z1.nn_top(paths, offs, pos2g, g_first, eg, every=0)
    outp = Z1.out_positions(eg, nbr, g_first)
    osc = B2.scores_at(eg, nbr, sc, g_first, Z1.SEARCH_K)
    src, dst, w = B2.k_edges(pos2g, outp, osc)
    return src, dst, w, {"groups": int(g_first.size), "edges": int(src.size), "tie_rows": int(Z1.tie_rows(sc).sum())}


def save(p, **arrs):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, p)
    return E.sha256(p)


def cmd_links(a):
    import zd84 as D
    import zul1 as UL
    point_zenc()
    root = mhr_root()
    t0 = time.time()
    nodes = nodes_of(root)
    n = len(nodes)
    titles, texts = [x["title"] for x in nodes], [x["text"] for x in nodes]
    F, st = D.rule_forms([{"title": t, "body": x} for t, x in zip(titles, texts)])
    f0, f0t = F["CN_all"], F["F0T"]
    lt_only = np.setdiff1d(f0t, f0)
    rec = {"stage": "MHR1 links", "declared_in": DOC + " section 5", "dataset": DS, "nodes": n,
           "dataset_json_sha256": E.sha256(root / DS / "DATASET.json"), "rule_build": st,
           "F0": int(f0.size), "F0T": int(f0t.size), "title_only": int(lt_only.size), "arms": {}}
    rec["files"] = {"F0": save(OUT / "links" / "rule" / "F0.npz", keys=f0),
                    "F0T": save(OUT / "links" / "rule" / "F0T.npz", keys=f0t)}
    for arm in ARMS:
        X = np.concatenate([np.load(p).astype(np.float32) for p in E.vec_shards(root, arm, DS, "docs")])
        w2, w2st = w2s_keys(titles, texts, X)
        ks, kd, kw, kst = k_family(root, arm, texts)
        kk = ks.astype(np.int64) * n + kd.astype(np.int64)
        r = np.union1d(w2, f0t)
        k_store = np.setdiff1d(kk, r)
        allk = np.union1d(r, kk)
        lab = np.zeros(allk.size, np.uint8)
        for bit, keys in ((1, w2), (2, f0), (3, lt_only), (4, kk)):
            lab[UL.member(allk, np.unique(keys))] |= np.uint8(1 << bit)
        fam = np.where(UL.member(allk, r), 0, 1).astype(np.uint8)          # 0 = R's store, 1 = K's store (R > K)
        base = OUT / "links" / arm
        files = {"W2_S": save(base / "W2_S.npz", keys=w2), "K": save(base / "K.npz", src=ks, dst=kd, weight=kw),
                 "store": save(base / "store.npz", keys=allk, labels=lab, family=fam,
                               label_names=np.asarray(LABELS), family_names=np.asarray(["R", "K"]))}
        per_label = {LABELS[b]: int(((lab >> b) & 1).sum()) for b in range(len(LABELS))}
        rec["arms"][arm] = {"W2_S": w2st, "K": kst, "R": int(r.size), "K_pairs": int(np.unique(kk).size),
                            "K_in_R": int(np.unique(kk).size - k_store.size), "K_store": int(k_store.size),
                            "RK": int(allk.size), "per_label": per_label, "files": files,
                            "vectors_manifest_sha256": E.sha256(E.vec_dir(root, arm, DS, "docs") / "manifest.json")}
        if arm == "frozen":
            g = root / DS / "graph"
            rec["package_knn_sha256"] = save(g / "knn.npz", src=ks, dst=kd, weight=kw)
        log(f"{arm}: W2_S {w2.size} pairs, K {kk.size} ({rec['arms'][arm]['K_in_R']} in R), R {r.size}, RK {allk.size}")
    rec.update({"seconds": round(time.time() - t0, 1), "utc": utc(), "threads": E.THREADS, **code_sha()})
    E.wjson(OUT / "links.json", rec)
    return 0


def forms_of():
    n = len(nodes_of())
    F = {"F0T": np.load(OUT / "links" / "rule" / "F0T.npz")["keys"]}
    for arm in ARMS:
        b = OUT / "links" / arm
        w2 = np.load(b / "W2_S.npz")["keys"]
        with np.load(b / "K.npz") as z:
            kk = np.unique(z["src"].astype(np.int64) * n + z["dst"].astype(np.int64))
        r = np.union1d(w2, F["F0T"])
        F.update({f"W2_S@{arm}": w2, f"R@{arm}": r, f"K@{arm}": kk, f"RK@{arm}": np.union1d(r, kk)})
    return F, n


DIFFS = (("R@e3v", "F0T"), ("RK@e3v", "R@e3v"), ("RK@frozen", "R@frozen"), ("RK@e3v", "RK@frozen"), ("K@e3v", "K@frozen"))


def gp_rows(F, golds, n, sel=None):
    import zd84 as D
    import zul1 as UL
    gl = [g for j, g in enumerate(golds) if sel is None or sel[j]]
    qidx, sk = UL.gp_slots(gl, n)
    if sk.size == 0:
        return None
    pairs = n * (n - 1) / 2
    cov = {f: UL.member(sk, F[f]) for f in F}
    rows = {"slots": int(sk.size), "questions": int(np.unique(qidx).size)}
    for f in F:
        rows[f] = {"linked": int(cov[f].sum()), **D.share_call(cov[f], qidx, len(gl), F[f].size / pairs)}
    for x, r in DIFFS:
        rows[f"{x} - {r}"] = UL.grade_diff(cov[x], cov[r], qidx, len(gl))
    return rows


def gs_rows(F, golds, n, sel=None):
    import zap1 as Z
    import zd84 as D
    import zul1 as UL
    has = np.array([g.size >= 2 and (sel is None or sel[j]) for j, g in enumerate(golds)])
    q = np.flatnonzero(has)
    unit = np.arange(q.size)
    flags = {f: D.gs_flags(F[f], golds, n)[q] for f in F}
    rows = {"questions": int(q.size), **{f: {"connected": int(v.sum()), "share": Z.pct(v.mean())} for f, v in flags.items()}}
    for x, r in DIFFS:
        rows[f"{x} - {r}"] = UL.grade_diff(flags[x], flags[r], unit, q.size)
    return rows


def cmd_grade(a):
    import zap1 as Z
    t0 = time.time()
    F, n = forms_of()
    nodes, qs = nodes_of(), queries_of()
    pos = {x["node_id"]: i for i, x in enumerate(nodes)}
    golds = [np.asarray(sorted(pos[g] for g in q["gold_node_ids"]), np.int64) for q in qs]
    types = np.asarray([q["question_type"] for q in qs])
    gc = np.asarray([g.size for g in golds])
    pairs = n * (n - 1) / 2
    res = {"pairs": {f: int(v.size) for f, v in F.items()}, "density_pct": {f: Z.pct(v.size / pairs) for f, v in F.items()},
           "GP": gp_rows(F, golds, n), "GS": gs_rows(F, golds, n)}
    res["price_pairs_per_linked_slot"] = {f: (round(F[f].size / res["GP"][f]["linked"], 1) if res["GP"][f]["linked"] else None)
                                          for f in F}
    res["by_type"] = {t: {"GP": gp_rows(F, golds, n, types == t), "GS": gs_rows(F, golds, n, types == t)}
                      for t in sorted(set(types.tolist()))}
    res["by_gold_count"] = {str(k): {"GP": gp_rows(F, golds, n, gc == k), "GS": gs_rows(F, golds, n, gc == k)}
                            for k in (2, 3, 4)}
    c = res["GP"]["RK@e3v - RK@frozen"]
    call = "e3v's LINKS CARRY ON MULTIHOP-RAG" if c["call"] != "BELOW" else "NOT_CARRIED"
    rec = {"stage": "MHR1 link grade", "declared_in": DOC + " section 5", "dataset": DS, "nodes": n, "questions": len(qs),
           "call": call, "deciding_cell": {"RK@e3v - RK@frozen (GP)": c},
           "links_sha256": E.sha256(OUT / "links.json"), **res, "seconds": round(time.time() - t0, 1), "utc": utc(),
           **code_sha()}
    E.wjson(OUT / "grade.json", rec)
    log(f"{call}: {c}")
    return 0


# ── report ───────────────────────────────────────────────────────────────────

def cmd_report(a):
    fr = json.loads((OUT / "first.json").read_text(encoding="utf-8"))
    gr = json.loads((OUT / "grade.json").read_text(encoding="utf-8"))
    lk = json.loads((OUT / "links.json").read_text(encoding="utf-8"))
    pk = json.loads((OUT / "package.json").read_text(encoding="utf-8"))
    L = [f"# MHR1 report ({utc()})", "", f"Declared in {DOC}. Package: {pk['nodes']} articles, {pk['queries']} kept questions "
         f"({json.dumps(pk['by_split_and_type'])}).", "", f"## First stage: **{fr['call']}**", "",
         "| list | " + " | ".join(["d2", "d4", "d5", "d10", "d20", "rrf5", "hits4", "hits10", "map10", "mrr10"]) + " |",
         "|---" * 11 + "|"]
    A = fr["reads"]["all"]
    for name in ("e3v", "frozen", "splade", "e3v+splade", "frozen+splade"):
        m = A[name]
        L.append(f"| {name} | " + " | ".join(str(m.get(k, "")) for k in ("d2", "d4", "d5", "d10", "d20", "rrf5", "hits4",
                                                                           "hits10", "map10", "mrr10")) + " |")
    L += ["", "| e3v - frozen (all) | points [95%] | call |", "|---|---|---|"]
    for k, v in A["e3v_minus_frozen"].items():
        L.append(f"| {k} | {v['diff']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] | {v['call']} |")
    L += ["", "| subset | n | e3v d5 | frozen d5 | gap d5 [95%] | gap rrf5 [95%] |", "|---|---|---|---|---|---|"]
    for s, r in fr["reads"].items():
        g5, gr5 = r["e3v_minus_frozen"]["d5"], r["e3v_minus_frozen"]["rrf5"]
        L.append(f"| {s} | {r['questions']} | {r['e3v']['d5']} | {r['frozen']['d5']} | {g5['diff']:+.2f} [{g5['lo']:+.2f}, "
                 f"{g5['hi']:+.2f}] {g5['call']} | {gr5['diff']:+.2f} [{gr5['lo']:+.2f}, {gr5['hi']:+.2f}] {gr5['call']} |")
    L += ["", f"Cut at the window: {json.dumps(fr['cut'])}.", "", f"## Links: **{gr['call']}**", "",
          "| form | pairs | density % | GP linked | GP share % [95%] | lift | call | GS % | pairs per linked slot |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f in gr["pairs"]:
        g, s = gr["GP"][f], gr["GS"][f]
        L.append(f"| {f} | {gr['pairs'][f]:,} | {gr['density_pct'][f]} | {g['linked']:,} | {g['share']} [{g['lo']}, {g['hi']}] | "
                 f"{g['lift']} | {g['call']} | {s['share']} | {gr['price_pairs_per_linked_slot'][f]} |")
    L += ["", "| difference | GP points [95%] | GS points [95%] |", "|---|---|---|"]
    for x, r in DIFFS:
        k = f"{x} - {r}"
        g, s = gr["GP"][k], gr["GS"][k]
        L.append(f"| {k} | {g['diff']:+.2f} [{g['lo']:+.2f}, {g['hi']:+.2f}] {g['call']} | {s['diff']:+.2f} "
                 f"[{s['lo']:+.2f}, {s['hi']:+.2f}] {s['call']} |")
    L += ["", "Stores (R > K):"]
    for arm, v in lk["arms"].items():
        L.append(f"- {arm}: R {v['R']:,}; K {v['K_pairs']:,} ({v['K_in_R']:,} already in R, {v['K_store']:,} in K's store); "
                 f"RK {v['RK']:,}; labels {json.dumps(v['per_label'])}; W2_S {json.dumps(v['W2_S'])}; K {json.dumps(v['K'])}")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    log(f"wrote {rel(OUT / 'report.md')}")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

def selftest():
    rng = np.random.default_rng(1)
    types = [str(x) for x in rng.choice(["a", "b", "c"], size=503)]
    s1, s2 = split_rule(types), split_rule(types)
    assert s1 == s2 and None not in s1
    for t in set(types):
        m = [s for s, y in zip(s1, types) if y == t]
        assert m.count("train") == len(m) * 4 // 10 and m.count("dev") == len(m) // 10, t
    # the benchmark's loop, literally, on strings
    docs = ["xx alpha yy", "beta here", "nothing", "alpha and beta", "gamma"]
    facts = [["alpha", "beta"], ["gamma"], ["zeta"]]
    ranked = [[2, 0, 1, 3, 4], [4, 0, 1, 2, 3], [0, 1, 2, 3, 4]]

    def lit(retrieved, gold):
        h4 = h10 = False
        find, aps, first = [], 0.0, None
        for rank, item in enumerate(retrieved[:11], start=1):
            if any(g in item for g in gold):
                if rank <= 10:
                    h10 = True
                    if first is None:
                        first = rank
                    if rank <= 4:
                        h4 = True
                    c = 0
                    for g in gold:
                        if g in item and g not in find:
                            c += 1
                            find.append(g)
                    aps += c / rank
        return h4, h10, aps / min(len(gold), 10), (1 / first if first else 0)

    got = bench_measures(ranked, facts, lambda d, f: f in docs[d])
    for q in range(3):
        want = lit([docs[d] for d in ranked[q]], facts[q])
        assert (got["hits4"][q], got["hits10"][q], got["map10"][q], got["mrr10"][q]) == \
            (float(want[0]), float(want[1]), want[2], want[3]), (q, want)
    assert abs(got["map10"][0] - (1 / 2 + 1 / 3) / 2) < 1e-12 and got["mrr10"][0] == 0.5 and got["hits4"][2] == 0
    top = np.asarray([[3, 1, 0], [2, 0, 1]])
    cv = coverage(top, [np.asarray([0, 3]), np.asarray([1])], 2)
    assert cv.tolist() == [0.5, 0.0]
    S = np.asarray([[0.5, 0.9, 0.9], [0.1, 0.1, 0.2]], np.float32)
    assert rank_all(S).tolist() == [[1, 2, 0], [2, 0, 1]]
    # W2_S on a toy: one exact title mention, one sibling pair
    titles = ["Big Apple Corp", "River Town", "River Town", "Other"]
    texts = ["It is based in River Town now.", "A town on a river; Big Apple Corp has offices here.",
             "River Town, second text with no title of another.", "Nothing to see."]
    X = np.eye(4, dtype=np.float32)
    keys, st = w2s_keys(titles, texts, X)
    n = 4
    got = sorted((int(k // n), int(k % n)) for k in keys)
    assert (1, 2) in got and (0, 1) in got and st["sibling_links"] == 2, (got, st)
    log("selftest passed")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "package", "splade", "encode", "frozen", "vectors", "fc", "cuts", "first", "links",
                                   "grade", "report"])
    p.add_argument("--enc")
    p.add_argument("--cpu", action="store_true", help="section 3.1: encode on the CPU, float32 weights")
    p.add_argument("--threads", type=int, default=E.THREADS)
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    return {"package": cmd_package, "splade": cmd_splade, "encode": cmd_encode, "frozen": cmd_frozen,
            "vectors": cmd_vectors, "fc": cmd_fc, "cuts": cmd_cuts, "first": cmd_first, "links": cmd_links, "grade": cmd_grade,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
