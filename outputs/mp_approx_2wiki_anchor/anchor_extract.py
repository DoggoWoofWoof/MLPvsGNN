"""Design look (untracked; not a result and not filed): anchor phrases for 2wiki's structural (hyperlink) edges.

2wiki's structural family has one relation, 'hyperlink' (rel is 0 on every row), so a typed walk on it can only use
family x direction. Each hyperlink comes from a mention in the source article, and the words just before the mention
("directed by [Y]", "son of [Y]", "born in [Y]") name the relation the link carries. This file recovers them: a fixed,
parameter-free attribute of each stored edge, computed from the corpus text alone (no query, no gold, no fit).

It reads the CRAG package in place, read-only (files opened 'rb' only, nothing under the root written, moved, renamed
or hashed):
  data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip   the raw records (id, title, sentences, mentions)
  data/final_canonical/2wiki/nodes.jsonl                            node position i = line i, source_id = curid
  data/final_canonical/2wiki/graph/structural.npz                   src, dst (positions), rel
It regenerates the structural edge sequence with the builder's pass-2 rule (CRAG scratchpad/build_2wiki_universe.py:
per record, mentions in stored order, each ref_id parsed as an int, kept when it is a corpus id, not the record's own
id and not already kept for this record), maps curids to positions through nodes.jsonl, and checks the sequence
against every row of structural.npz (in order, else as a permutation). Only then are the phrases attached, by row.

Per edge, from sentences[sent_idx][:start] (the text before the mention): the last three tokens (words, numbers,
punctuation), normalised (a lowercase word kept, a capitalised word -> <C>, a number -> <N>, punctuation kept, <s> at
the sentence start); w1 is the last token, w2 the last two. Also the sentence index and the edge's rank among its
record's kept edges.

Writes only to its own directory: anchors.npz (w1, w2 int32 ids; sent int8; kidx int16; aligned to structural.npz
rows), vocab.json (w1 and w2 strings in id order with counts), anchor_extract.json (counts, checks, timing).

    python outputs/mp_approx_2wiki_anchor/anchor_extract.py [--workers 5]
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from collections import deque  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CRAG = Path("C:/Users/Swastik/Desktop/CRAG")
ZIP = CRAG / "data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip"
MEMBER = "para_with_hyperlink.jsonl"
NODES = CRAG / "data/final_canonical/2wiki/nodes.jsonl"
STRUCT = CRAG / "data/final_canonical/2wiki/graph/structural.npz"
N_NODES = 5989847
N_EDGES = 28963600
CHUNK = 20000
TOK = re.compile(r"\w[\w'\-]*|[^\w\s]")
WINDOW = 80

_W = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(w):
    if w[0].isdigit():
        return "<N>"
    if w[0].isalpha() or w[0] == "_":
        if w.islower() or not any(c.isupper() for c in w):
            return w
        return "<C>" if w[0].isupper() else w.lower()
    return w


def phrase(sentence, start):
    toks = [norm(t) for t in TOK.findall(sentence[max(0, start - WINDOW):start])]
    cut = start - WINDOW > 0
    if cut and toks:
        toks = toks[1:]   # the window may have split the first word
    toks = ["<s>", "<s>", "<s>"][:max(0, 3 - len(toks))] + toks[-3:]
    return toks[-1], toks[-2] + " " + toks[-1]


def init(sorted_cid, order):
    _W["cid"] = sorted_cid
    _W["pos"] = order


def positions(cids):
    """curid -> node position, -1 when the curid is not a node."""
    c = _W["cid"]
    j = np.searchsorted(c, cids)
    jc = np.minimum(j, c.size - 1)
    hit = (j < c.size) & (c[jc] == cids)
    return np.where(hit, _W["pos"][jc], -1), hit


def work(lines):
    rec_ids, fails = [], 0
    cand_rec, cand_m, cand_t = [], [], []
    recs = []
    for raw in lines:
        try:
            r = json.loads(raw)
        except Exception:
            fails += 1
            continue
        ri = len(recs)
        cid = int(r["id"])
        recs.append((cid, r.get("sentences") or [], r.get("mentions") or []))
        rec_ids.append(cid)
        for mi, m in enumerate(r.get("mentions", [])):
            rids = m.get("ref_ids") or []
            for tid in rids:
                try:
                    t = int(tid)
                except Exception:
                    continue
                cand_rec.append(ri)
                cand_m.append(mi)
                cand_t.append(t)
    rec_ids = np.asarray(rec_ids, dtype=np.int64)
    cr = np.asarray(cand_rec, dtype=np.int64)
    cm = np.asarray(cand_m, dtype=np.int64)
    ct = np.asarray(cand_t, dtype=np.int64)
    out = {"records": len(recs), "fails": fails, "rec_ids": rec_ids, "raw_targets": int(ct.size)}
    if ct.size == 0:
        out.update(src=np.zeros(0, np.int32), dst=np.zeros(0, np.int32), sent=np.zeros(0, np.int8), kidx=np.zeros(0, np.int16),
                   w1=[], w2=[], tgt_out=0, self_loops=0, bad=0)
        return out
    tpos, hit = positions(ct)
    out["tgt_out"] = int((~hit).sum())
    own = rec_ids[cr]
    selfl = hit & (ct == own)
    out["self_loops"] = int(selfl.sum())
    keep = hit & ~selfl
    idx = np.flatnonzero(keep)
    # the first candidate per (record, target), in candidate order (records then mentions then ref_ids)
    key = cr[idx] * (1 << 33) + ct[idx]
    _, first = np.unique(key, return_index=True)
    idx = idx[np.sort(first)]
    spos, shit = positions(rec_ids[cr[idx]])
    if not shit.all():
        raise RuntimeError("a record with kept edges is not a node")
    w1, w2, sent, bad = [], [], np.zeros(idx.size, np.int8), 0
    kidx = np.zeros(idx.size, np.int16)
    prev, k = -1, 0
    for e, c in enumerate(idx):
        ri, mi = int(cr[c]), int(cm[c])
        k = k + 1 if ri == prev else 0
        prev = ri
        kidx[e] = min(k, 32767)
        _cid, sents, ments = recs[ri]
        m = ments[mi]
        si, st = m.get("sent_idx"), m.get("start")
        if isinstance(si, int) and isinstance(st, int) and 0 <= si < len(sents) and 0 <= st <= len(sents[si]):
            a, b = phrase(sents[si], st)
            sent[e] = min(si, 127)
        else:
            a, b, bad = "<bad>", "<bad> <bad>", bad + 1
            sent[e] = -1
        w1.append(a)
        w2.append(b)
    out.update(src=spos.astype(np.int32), dst=tpos[idx].astype(np.int32), sent=sent, kidx=kidx, w1=w1, w2=w2, bad=bad)
    return out


def node_curids():
    """curid per node position, from nodes.jsonl's source_id (bytes search, no full parse)."""
    out = np.empty(N_NODES, dtype=np.int64)
    pat = re.compile(rb'"source_id": "(\d+)"')
    n = 0
    with open(NODES, "rb") as f:
        for line in f:
            m = pat.search(line)
            if m is None:
                raise SystemExit(f"nodes.jsonl line {n}: no source_id")
            out[n] = int(m.group(1))
            n += 1
    if n != N_NODES:
        raise SystemExit(f"nodes.jsonl has {n} lines, expected {N_NODES}")
    return out


def chunks():
    z = zipfile.ZipFile(ZIP)
    with z.open(MEMBER) as f:
        buf = []
        for raw in f:
            buf.append(raw)
            if len(buf) == CHUNK:
                yield buf
                buf = []
        if buf:
            yield buf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    t0 = time.time()
    cur = node_curids()
    if np.unique(cur).size != N_NODES:
        raise SystemExit("node curids are not unique")
    order = np.argsort(cur, kind="stable")
    sorted_cid = cur[order]
    log(f"{N_NODES} node curids read in {time.time() - t0:.0f}s")
    d1, d2 = {}, {}
    c1, c2 = [], []
    parts = {k: [] for k in ("src", "dst", "sent", "kidx", "w1", "w2")}
    rec_all = []
    tot = {"records": 0, "fails": 0, "raw_targets": 0, "tgt_out": 0, "self_loops": 0, "bad": 0}
    with Pool(a.workers, initializer=init, initargs=(sorted_cid, order.astype(np.int64))) as pool:
        it, pending = chunks(), deque()

        def submit():
            buf = next(it, None)
            if buf is not None:
                pending.append(pool.apply_async(work, (buf,)))   # at most 2 x workers chunks in flight, in order

        for _ in range(2 * a.workers):
            submit()
        ci = -1
        while pending:
            o = pending.popleft().get()
            submit()
            ci += 1
            for k in tot:
                tot[k] += o[k]
            rec_all.append(o["rec_ids"])
            for k in ("src", "dst", "sent", "kidx"):
                parts[k].append(o[k])
            for name, d, c in (("w1", d1, c1), ("w2", d2, c2)):
                ids = np.empty(len(o[name]), dtype=np.int32)
                for e, s in enumerate(o[name]):
                    j = d.get(s)
                    if j is None:
                        j = d[s] = len(c)
                        c.append(0)
                    c[j] += 1
                    ids[e] = j
                parts[name].append(ids)
            if ci % 25 == 0:
                log(f"chunk {ci}: {tot['records']} records, {sum(x.size for x in parts['src'])} edges, "
                    f"{len(d1)} w1, {len(d2)} w2, {time.time() - t0:.0f}s")
    arr = {k: np.concatenate(v) for k, v in parts.items()}
    n = arr["src"].size
    log(f"stream done: {tot}, {n} edges, {time.time() - t0:.0f}s")
    res = {"look": "anchor_extract", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "counts": tot,
           "edges_regenerated": int(n), "inputs": {str(p.relative_to(CRAG)): {"bytes": p.stat().st_size, "mtime": p.stat().st_mtime}
                                                  for p in (ZIP, NODES, STRUCT)}}
    rec_all = np.concatenate(rec_all)
    res["zip_records_parsed"] = int(rec_all.size)
    res["zip_ids_unique"] = int(np.unique(rec_all).size)
    res["zip_ids_equal_node_curids"] = bool(np.array_equal(np.unique(rec_all), sorted_cid))
    if not res["zip_ids_equal_node_curids"]:
        raise SystemExit(f"the zip's record ids are not the node curids: {res}")
    with np.load(STRUCT) as zf:
        s_src, s_dst, s_rel = zf["src"], zf["dst"], zf["rel"]
    res["structural_rows"] = int(s_src.size)
    res["structural_rel_values"] = np.unique(s_rel).tolist()
    if s_src.size != N_EDGES or n != N_EDGES:
        raise SystemExit(f"edge counts differ: structural {s_src.size}, regenerated {n}")
    if np.array_equal(s_src, arr["src"]) and np.array_equal(s_dst, arr["dst"]):
        res["alignment"] = "in_order"
        perm = None
    else:
        ks = s_src.astype(np.int64) * N_NODES + s_dst
        kr = arr["src"].astype(np.int64) * N_NODES + arr["dst"]
        os_, or_ = np.argsort(ks, kind="stable"), np.argsort(kr, kind="stable")
        if not np.array_equal(ks[os_], kr[or_]):
            raise SystemExit("the regenerated edges are not the structural rows, even as a set")
        if np.unique(ks).size != ks.size:
            raise SystemExit("structural rows repeat a (src, dst) pair; a permutation alignment is ambiguous")
        perm = np.empty(N_EDGES, dtype=np.int64)
        perm[os_] = or_            # structural row i <- regenerated edge perm[i]
        res["alignment"] = "permutation"
        res["in_order_prefix"] = int(np.argmax((s_src != arr["src"]) | (s_dst != arr["dst"])))
    if perm is not None:
        for k in ("sent", "kidx", "w1", "w2", "src", "dst"):
            arr[k] = arr[k][perm]
        if not (np.array_equal(s_src, arr["src"]) and np.array_equal(s_dst, arr["dst"])):
            raise SystemExit("the permuted edges do not match the structural rows")
    res["checked_rows"] = int(N_EDGES)
    res["vocab"] = {"w1": len(c1), "w2": len(c2)}
    for name, c in (("w1", c1), ("w2", c2)):
        cc = np.asarray(c)
        top = np.sort(cc)[::-1]
        res[f"{name}_coverage_of_top"] = {str(k): round(float(top[:k].sum() / cc.sum()), 4) for k in (16, 64, 256, 1024, 4096, 16384, 65535)}
    inv1 = [None] * len(d1)
    for s, j in d1.items():
        inv1[j] = s
    inv2 = [None] * len(d2)
    for s, j in d2.items():
        inv2[j] = s
    o1, o2 = np.argsort(-np.asarray(c1), kind="stable"), np.argsort(-np.asarray(c2), kind="stable")
    res["w1_top"] = [[inv1[j], int(c1[j])] for j in o1[:60]]
    res["w2_top"] = [[inv2[j], int(c2[j])] for j in o2[:120]]
    np.savez(HERE / "anchors.tmp.npz", w1=arr["w1"], w2=arr["w2"], sent=arr["sent"], kidx=arr["kidx"])
    os.replace(HERE / "anchors.tmp.npz", HERE / "anchors.npz")
    (HERE / "vocab.json").write_text(json.dumps({"w1": [[s, int(c1[j])] for j, s in enumerate(inv1)],
                                                 "w2": [[s, int(c2[j])] for j, s in enumerate(inv2)]}, ensure_ascii=False), encoding="utf-8")
    res["seconds"] = round(time.time() - t0, 1)
    res["anchors_npz_sha256"] = hashlib.sha256((HERE / "anchors.npz").read_bytes()).hexdigest()
    (HERE / "anchor_extract.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done: {res['alignment']}, vocab {res['vocab']}, {res['seconds']}s")
    log("w2 top: " + ", ".join(f"{s}:{c}" for s, c in res["w2_top"][:40]))


if __name__ == "__main__":
    main()
