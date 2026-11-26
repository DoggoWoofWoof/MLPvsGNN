"""ING1 phase B (docs/ING1_INGESTION_AND_STAGE_REGISTRY.md, section 3.7 and amendment A1): one read-only world loader for the ten datasets.

    python -I -B outputs/mp_unified/zworld.py selftest
    python -I -B outputs/mp_unified/zworld.py gate [--ds NAME ...] [--no-legacy]     # G-W, run from the workspace root

`load(name)` opens a dataset's package through configs/datasets.json and gives one interface for every dataset: node ids and texts
(line i of nodes.jsonl is node i), the queries in row order (row j = query_ids.json[j], the split files in the record's order) with
their DISTINCT gold nodes as ids and positions, the S edge file, and the vector shards of each arm. It reads, never writes.
No dataset name appears here: every name, root and arm comes from the registry. G-W compares this loader with the legacy readers
(zenc, zul1, zul2m) on counts and content hashes and writes outputs/ing1/gw_<dataset>.json.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
sys.dont_write_bytecode = True
ENC = WS / "outputs" / "enc"
OUT = WS / "outputs" / "ing1"
DOC = "docs/ING1_INGESTION_AND_STAGE_REGISTRY.md"
_NID = re.compile(r'"node_id": "([^"]+)"')


def stop(msg):
    raise SystemExit(f"zworld: {msg}")


def log(msg):
    print(f"[zworld {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ingest():
    """The registry tool's own resolver (one rule for 'where is the package')."""
    spec = importlib.util.spec_from_file_location("ing1_ingest", WS / "scripts" / "ingest" / "ingest.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class World:
    def __init__(self, cfg, row, root):
        self.cfg, self.row, self.name, self.kind = cfg, row, row["name"], row["kind"]
        self.root = Path(root)
        self.served, self.dsdir = self.root.parent, self.root.name       # zenc's (root, ds) pair
        self.rec = json.loads((self.root / "DATASET.json").read_text(encoding="utf-8"))
        self.n = int(self.rec["n_nodes"])
        self.nsha = self.rec["nodes"]["sha256"]
        self.nq = int(self.rec["n_queries"])
        q = self.rec["queries"]
        self.splits = list(q["splits_in_row_order"])
        self.by_split = {s: int(v["n"]) for s, v in q["by_split"].items()}
        self.arms = list(row.get("arms", []))
        self.s_source = row["s_source"]
        self._pos = None
        if sum(self.by_split.values()) != self.nq or set(self.splits) != set(self.by_split):
            stop(f"{self.name}: the record's splits do not add up to its query count")

    # nodes: line i of nodes.jsonl is node i
    def iter_nodes(self):
        """(node_id, text) in line order."""
        with open(self.root / "nodes.jsonl", encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                yield r["node_id"], r["text"]

    def node_ids(self):
        out = []
        with open(self.root / "nodes.jsonl", encoding="utf-8") as f:
            for ln in f:
                m = _NID.search(ln, 0, 4000)
                out.append(m.group(1) if m else json.loads(ln)["node_id"])
        if len(out) != self.n:
            stop(f"{self.name}: {len(out)} node lines, the record says {self.n}")
        return out

    def positions(self):
        if self._pos is None:
            ids = self.node_ids()
            self._pos = {x: i for i, x in enumerate(ids)}
            if len(self._pos) != len(ids):
                stop(f"{self.name}: node ids are not unique")
        return self._pos

    # queries: row j = query_ids.json[j]
    def query_row_ids(self):
        return json.loads((self.root / "queries" / "query_ids.json").read_text(encoding="utf-8"))

    def iter_queries(self, splits=None, resolve=True):
        """dict per question in row order: row, query_id, question, split, gold_ids (distinct, first seen), gold_pos (int64, same order),
        n_listed (the list's length before collapsing). A gold id that is not a node stops the read (resolve=True)."""
        qids = self.query_row_ids()
        if len(qids) != self.nq:
            stop(f"{self.name}: query_ids.json has {len(qids)} ids, the record {self.nq}")
        pos = self.positions() if resolve else None
        j = 0
        for sp in self.splits:
            with open(self.root / "queries" / f"{sp}.jsonl", encoding="utf-8") as f:
                for ln in f:
                    r = json.loads(ln)
                    if r.get("query_id") != qids[j]:
                        stop(f"{self.name}: query row {j} is {r.get('query_id')!r}, query_ids.json has {qids[j]!r}")
                    listed = list(r.get("gold_node_ids") or [])
                    dist = list(dict.fromkeys(listed))
                    if sp in splits_or(splits, self.splits):
                        gp = np.asarray([pos[g] for g in dist], np.int64) if resolve else None
                        yield {"row": j, "query_id": qids[j], "question": r["question"], "split": sp, "gold_ids": dist, "gold_pos": gp,
                               "n_listed": len(listed)}
                    j += 1
        if j != self.nq:
            stop(f"{self.name}: {j} query lines, the record says {self.nq}")

    # edges: S is the structural family; a package keeps it in graph/structural.npz or graph/s_edges.npz
    def edge_file(self, family="structural"):
        g = self.root / "graph"
        cands = {"structural": ("structural.npz", "s_edges.npz"), "ner": ("ner.npz",), "knn": ("knn.npz",)}[family]
        for c in cands:
            if (g / c).is_file():
                return g / c
        return None

    def rel_vocab(self):
        fam = ((self.rec.get("graph") or {}).get("families") or {}).get("structural") or {}
        if fam.get("relation_vocabulary"):
            return list(fam["relation_vocabulary"])
        p = self.root / "graph" / "s_rel.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    # vectors: frozen arm in the package; every other arm under outputs/enc/<arm>/<dataset>/
    def vector_dir(self, arm, kind):
        if arm not in self.arms:
            stop(f"{self.name}: no arm {arm!r} is registered (arms {self.arms})")
        return (self.root / "embeddings" / "dense" / kind) if arm == "frozen" else (ENC / arm / self.name / kind)

    def vector_shards(self, arm, kind):
        man = json.loads((self.root / "embeddings" / "dense" / kind / "manifest.json").read_text(encoding="utf-8"))
        d = self.vector_dir(arm, kind)
        paths = [d / f"shard_{s:05d}.npy" for s in range(int(man["n_shards"]))]
        miss = [p.name for p in paths if not p.is_file()]
        if miss:
            return None, f"{len(miss)} of {len(paths)} shards are not on this machine"
        return paths, None


def splits_or(want, have):
    return have if want is None else list(want)


def load(name, cfg=None):
    I = _ingest()
    cfg = cfg or json.loads((WS / "configs" / "datasets.json").read_text(encoding="utf-8"))
    row = next((r for r in cfg["datasets"] if r["name"] == name), None)
    if row is None:
        stop(f"{name} is not in configs/datasets.json (add its row first; ingest.py check must pass)")
    root = I.package_dir(cfg, row)
    if root is None:
        stop(f"{name}: no package on this machine")
    return World(cfg, row, root)


def selftest():
    import os
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="zworld_"))
    try:
        pk = tmp / "pk" / "toy"
        (pk / "queries").mkdir(parents=True)
        nodes = [{"node_id": f"toy:{i}", "text": f"text {i}"} for i in range(5)]
        (pk / "nodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in nodes), encoding="utf-8")
        qa = [{"query_id": "a", "question": "qa?", "gold_node_ids": ["toy:3", "toy:1", "toy:3"]},
              {"query_id": "b", "question": "qb?", "gold_node_ids": []}]
        qb = [{"query_id": "c", "question": "qc?", "gold_node_ids": ["toy:4"]}]
        for sp, rows in (("train", qa), ("dev", qb)):
            (pk / "queries" / f"{sp}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        (pk / "queries" / "query_ids.json").write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
        rec = {"n_nodes": 5, "n_queries": 3, "nodes": {"sha256": "x", "n": 5},
               "queries": {"splits_in_row_order": ["train", "dev"], "by_split": {"train": {"n": 2}, "dev": {"n": 1}}}}
        (pk / "DATASET.json").write_text(json.dumps(rec), encoding="utf-8")
        row = {"name": "toy", "kind": "passage", "arms": ["e3v"], "s_source": "none"}
        w = World({}, row, pk)
        assert [i for i, _ in w.iter_nodes()] == [f"toy:{i}" for i in range(5)]
        assert w.node_ids() == [f"toy:{i}" for i in range(5)]
        qs = list(w.iter_queries())
        assert [q["row"] for q in qs] == [0, 1, 2]
        assert qs[0]["gold_ids"] == ["toy:3", "toy:1"] and qs[0]["gold_pos"].tolist() == [3, 1] and qs[0]["n_listed"] == 3
        assert qs[1]["gold_ids"] == [] and qs[1]["gold_pos"].size == 0
        assert [q["row"] for q in w.iter_queries(["dev"])] == [2]
        (pk / "queries" / "query_ids.json").write_text(json.dumps(["a", "x", "c"]), encoding="utf-8")
        try:
            list(w.iter_queries())
            raise AssertionError("a wrong query id must stop")
        except SystemExit as e:
            assert "query row 1" in str(e)
        (pk / "queries" / "query_ids.json").write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
        (pk / "queries" / "dev.jsonl").write_text(json.dumps({"query_id": "c", "question": "qc?", "gold_node_ids": ["toy:99"]}) + "\n",
                                                  encoding="utf-8")
        try:
            list(w.iter_queries())
            raise AssertionError("an unknown gold must stop")
        except KeyError:
            pass
        assert w.edge_file() is None and w.rel_vocab() is None
        assert sha256_file(pk / "nodes.jsonl") == hashlib.sha256((pk / "nodes.jsonl").read_bytes()).hexdigest()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log("zworld selftest PASS (order, distinct golds, positions, split filter, wrong id stops, unknown gold stops)")
    return 0


# ── G-W: the loader against the legacy readers ──────────────────────────────────────────────────────────────────────────────

def _legacy_modules():
    sys.path.insert(0, str(HERE))
    import zenc as E  # noqa: E402  (reads sys.argv --threads; its WS is the working directory: run from the root)
    return E


def gate(w, legacy=True):
    g, t0 = {}, time.time()

    def put(name, ok, detail):
        g[name] = {"verdict": "PASS" if ok is True else ("FAIL" if ok is False else "N/A"), "detail": detail}
        log(f"{w.name} {name}: {g[name]['verdict']} ({detail})")

    # W1: the file is the record's
    sha = sha256_file(w.root / "nodes.jsonl")
    nid = w.node_ids()
    put("W1 nodes.jsonl is the record's", sha == w.nsha and len(nid) == w.n, f"{len(nid):,} lines, sha256 {sha[:12]}")
    # W2: the world's counts and ids against the pinned stage records (the layout grade, where one exists)
    pins = []
    for tpl in ("outputs/ul2c/{ds}/grade.json", "outputs/ul2m/{ds}/grade.json"):
        p = WS / tpl.format(ds=w.name)
        if p.is_file():
            r = json.loads(p.read_text(encoding="utf-8"))
            pins.append((tpl.split("/")[1], r["nodes"] == w.n and r["nodes_sha256"] == w.nsha))
    put("W2 the pinned layout grade names this node file", all(ok for _, ok in pins) if pins else None,
        ", ".join(f"{a}: {'equal' if ok else 'DIFFERENT'}" for a, ok in pins) or "no legacy grade for this dataset")
    # W3: queries, rows and splits
    qs = list(w.iter_queries())
    reps = sum(1 for q in qs if q["n_listed"] != len(q["gold_ids"]))
    empty = sum(1 for q in qs if not q["gold_ids"])
    put("W3 queries read in row order, every gold resolves", len(qs) == w.nq,
        f"{len(qs):,} questions, {empty} without a node gold, {reps} listing a node twice (collapsed)")
    if not legacy:
        return g
    E = _legacy_modules()
    # W4: node texts against zenc.doc_chunks (the encoder's own reader)
    h1, h2 = hashlib.sha256(), hashlib.sha256()
    n1 = n2 = 0
    for _i, tx in w.iter_nodes():
        h1.update(tx.encode("utf-8") + b"\n")
        n1 += 1
    for _s, texts in E.doc_chunks(w.served, w.dsdir):
        for tx in texts:
            h2.update(tx.encode("utf-8") + b"\n")
            n2 += 1
    put("W4 node texts equal zenc.doc_chunks", n1 == n2 == w.n and h1.hexdigest() == h2.hexdigest(), f"{n1:,} texts, {h1.hexdigest()[:12]}")
    # W5: questions against zenc.query_texts
    old = E.query_texts(w.served, w.dsdir)
    same = len(old) == len(qs) and all(old[q["row"]] == q["question"] for q in qs)
    put("W5 questions equal zenc.query_texts", same, f"{len(old):,} rows")
    # W6: golds against the legacy gold readers
    by_id = {q["query_id"]: q for q in qs}
    if w.name == "multihoprag":
        import zul2m as M2
        root, nodes, _n, _s = M2.head()
        golds, _split, _types = M2.golds_splits(root, nodes)
        ok = len(golds) == len(qs) and all(np.array_equal(np.unique(a), np.unique(q["gold_pos"])) for a, q in zip(golds, qs))
        put("W6 golds equal zul2m.golds_splits", ok, f"{len(golds):,} questions")
    elif w.name in ("squad", "musique", "hotpotqa", "2wiki", "metaqa", "webqsp"):
        import zul1 as UL
        tot, bad = 0, []
        for cv in UL.CARVES[w.name]:
            ids, golds, _sha = UL.carve_questions(w.name, cv)
            for i, a in zip(ids, golds):
                q = by_id.get(i)
                tot += 1
                if q is None or not np.array_equal(np.unique(a), np.unique(q["gold_pos"])):
                    bad.append(i)
        put("W6 golds equal zul1.carve_questions (both carves)", not bad, f"{tot:,} questions, {len(bad)} differ {bad[:3]}")
    else:
        put("W6 golds against a legacy reader", None, "no legacy reader exists for this dataset (it is new)")
    # W7: vector shards against zenc.vec_shards
    for arm in w.arms:
        enc = "frozen" if arm == "frozen" else arm
        for kind in ("docs", "queries"):
            mine, why = w.vector_shards(arm, kind)
            if mine is None:
                put(f"W7 {arm} {kind} shards", None, why)
                continue
            try:
                old_p = E.vec_shards(w.served, enc, w.dsdir, kind)
            except SystemExit as e:
                put(f"W7 {arm} {kind} shards", False, f"zenc stops: {e}")
                continue
            put(f"W7 {arm} {kind} shards equal zenc.vec_shards", [str(a) for a in mine] == [str(b) for b in old_p], f"{len(mine)} shards")
    g["_seconds"] = round(time.time() - t0, 1)
    return g


def cmd_gate(a):
    cfg = json.loads((WS / "configs" / "datasets.json").read_text(encoding="utf-8"))
    names = a.ds or [r["name"] for r in cfg["datasets"]]
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for nme in names:
        try:
            w = load(nme, cfg)
        except SystemExit as e:
            log(f"{nme}: {e}")
            continue
        g = gate(w, legacy=not a.no_legacy)
        verdict = "FAIL" if any(v["verdict"] == "FAIL" for k, v in g.items() if not k.startswith("_")) else "PASS"
        bad += verdict == "FAIL"
        rec = {"stage": "ING1 G-W", "declared_in": DOC, "dataset": nme, "verdict": verdict, "gates": {k: v for k, v in g.items() if not k.startswith("_")},
               "legacy_compared": not a.no_legacy, "seconds": g.get("_seconds"), "record_sha256": sha256_file(w.root / "DATASET.json"),
               "code_sha256": {"outputs/mp_unified/zworld.py": sha256_file(Path(__file__))}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        tmp = OUT / f"gw_{nme}.tmp"
        tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        tmp.replace(OUT / f"gw_{nme}.json")
        log(f"{nme}: G-W {verdict}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    g = sub.add_parser("gate")
    g.add_argument("--ds", nargs="*")
    g.add_argument("--no-legacy", action="store_true")
    g.add_argument("--threads", default="8")
    a = ap.parse_args()
    return selftest() if a.cmd == "selftest" else cmd_gate(a)


if __name__ == "__main__":
    sys.exit(main())
