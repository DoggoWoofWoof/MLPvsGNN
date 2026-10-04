"""Design look (untracked; not a result and not filed): anchor phrases for hotpotqa's structural (hyperlink) edges.

hotpotqa's structural family has one relation, 'hyperlink', so a typed walk on it can only use family x direction, as
on 2wiki before outputs/mp_approx_2wiki_anchor/anchor_extract.py gave each hyperlink the words its source article puts
before the link. This file does the same for hotpotqa: a fixed, parameter-free attribute of each stored edge, from the
corpus text alone (no query, no gold, no fit), normalised by anchor_extract.py's phrase() (imported unchanged, sha256
pinned below), so the two datasets' phrases are one definition.

It reads the CRAG package in place, read-only (files opened 'rb' only; nothing under the root written, moved, renamed
or hashed):
  data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
  data/final_canonical/hotpotqa/nodes.jsonl                  node position i = line i, with its title
  data/final_canonical/hotpotqa/graph/structural.npz         src, dst (positions), rel
It regenerates the structural edge sequence with CRAG's builder rule (scratchpad/c3_hotpot_graph.py): tar members in
order, lines in order, each record's text_with_links flattened in order, each <a href> unquoted with '_' -> ' ' and
stripped, kept when it names a corpus title, is not the record's own and (record, target) was not kept before anywhere
in the stream. A title maps to the last nodes.jsonl line that carries it (the builder's dict kept the last). Titles are
looked up by a 64-bit blake2b hash (checked collision-free over the node titles). The sequence is checked against
every row of structural.npz (in order, else as a permutation) before any phrase is attached.

Per edge, from the text before the link in its sentence with earlier links' tags removed: w1 (the last normalised
token) and w2 (the last two), the sentence index and the edge's rank among its record's kept edges.

Writes only to its own directory: anchors.npz, vocab.json, anchor_extract_hotpot.json, and host/anchors_compact.npz
(make_compact.py's format: ranks by count as uint16, the top 4096 strings).

    python outputs/mp_approx_hotpot_anchor/anchor_extract_hotpot.py [--workers 4]
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import bz2  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.parse import unquote  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_2wiki_anchor"))
import anchor_extract as AE  # noqa: E402  (2wiki's extractor; only norm/phrase are used, unchanged)

AE_SHA_PIN = "5efb986683fce4190a106e19c2df918da39803df074852f10fb1b8ebce99640d"
CRAG = Path("C:/Users/Swastik/Desktop/CRAG")
TB = CRAG / "data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
NODES = CRAG / "data/final_canonical/hotpotqa/nodes.jsonl"
STRUCT = CRAG / "data/final_canonical/hotpotqa/graph/structural.npz"
N_NODES = 5233329
N_EDGES = 15367541
AHREF = re.compile(r'<a href="([^"]*)">')
TAG = re.compile(r'<a href="[^"]*">|</a>')
BATCH = 64          # tar members per task
TOP = 4096

_W = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


def init(keys, pos):
    _W["k"], _W["p"] = keys, pos


def lookup(hs):
    k = _W["k"]
    j = np.searchsorted(k, hs)
    jc = np.minimum(j, k.size - 1)
    hit = (j < k.size) & (k[jc] == hs)
    return np.where(hit, _W["p"][jc], -1)


def flat(t):
    """c3_hotpot_graph.py's flattening order, yielding each string."""
    if isinstance(t, str):
        yield t
    elif isinstance(t, list):
        for x in t:
            yield from flat(x)


def work(blobs):
    """Per member blob: every record's candidates in stream order (record title position, target position, the phrase,
    the sentence index); misses are -1 and are dropped in the main process, after the counts."""
    rec_src, c_rec, c_t, c_sent, w1, w2 = [], [], [], [], [], []
    stats = {"members": 0, "fails": 0, "records": 0, "hrefs": 0}
    for blob in blobs:
        stats["members"] += 1
        try:
            text = bz2.decompress(blob).decode("utf-8")
        except Exception:
            stats["fails"] += 1
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stats["records"] += 1
            ri = len(rec_src)
            rec_src.append(h64(r.get("title")) if isinstance(r.get("title"), str) else 0)
            for si, sent in enumerate(flat(r.get("text_with_links") or [])):
                for m in AHREF.finditer(sent):
                    tt = unquote(m.group(1)).replace("_", " ").strip()
                    plain = TAG.sub("", sent[:m.start()])
                    a, b = AE.phrase(plain, len(plain))
                    c_rec.append(ri)
                    c_t.append(h64(tt))
                    c_sent.append(min(si, 127))
                    w1.append(a)
                    w2.append(b)
                    stats["hrefs"] += 1
    rs = lookup(np.asarray(rec_src, dtype=np.int64)) if rec_src else np.zeros(0, np.int64)
    ts = lookup(np.asarray(c_t, dtype=np.int64)) if c_t else np.zeros(0, np.int64)
    return {"stats": stats, "rec_pos": rs.astype(np.int64), "c_rec": np.asarray(c_rec, dtype=np.int64), "t_pos": ts.astype(np.int64),
            "sent": np.asarray(c_sent, dtype=np.int8), "w1": w1, "w2": w2}


def node_titles():
    """title hash per node position and the hash -> last position table."""
    hs = np.empty(N_NODES, dtype=np.int64)
    n = 0
    with open(NODES, "rb") as f:
        for line in f:
            hs[n] = h64(json.loads(line)["title"])
            n += 1
    if n != N_NODES:
        raise SystemExit(f"nodes.jsonl has {n} lines, expected {N_NODES}")
    return hs


def members():
    with tarfile.open(TB, "r:bz2") as tar:
        buf = []
        for m in tar:
            if not m.isfile():
                continue
            buf.append(tar.extractfile(m).read())
            if len(buf) == BATCH:
                yield buf
                buf = []
        if buf:
            yield buf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    ae_sha = hashlib.sha256(Path(AE.__file__).read_bytes()).hexdigest()
    if ae_sha != AE_SHA_PIN:
        raise SystemExit("anchor_extract.py is not the pinned file")
    t0 = time.time()
    hs = node_titles()
    order = np.argsort(hs, kind="stable")
    sh = hs[order]
    dup = np.r_[sh[1:] == sh[:-1], False]
    # the last position per hash: within a run of equal hashes (stable order = ascending position) keep the run's end
    last = ~dup
    keys, pos = sh[last], order[last]
    n_dup_titles = int(N_NODES - keys.size)
    log(f"{N_NODES} node titles hashed in {time.time() - t0:.0f}s, {n_dup_titles} positions share a title hash with a later one")
    # a hash shared by different titles would merge them; check on the titles that repeat a hash
    if n_dup_titles:
        rep = set(sh[dup].tolist())   # every hash that more than one position carries
        seen = {}
        with open(NODES, "rb") as f:
            for i, line in enumerate(f):
                if int(hs[i]) in rep:
                    t = json.loads(line)["title"]
                    if seen.setdefault(int(hs[i]), t) != t:
                        raise SystemExit(f"a title hash collision at line {i}")
    parts = {k: [] for k in ("rec_pos", "c_rec", "t_pos", "sent", "w1", "w2")}
    d1, d2, c1, c2 = {}, {}, [], []
    tot = {"members": 0, "fails": 0, "records": 0, "hrefs": 0}
    rec_base = 0
    with Pool(a.workers, initializer=init, initargs=(keys, pos.astype(np.int64))) as pool:
        it, pending = members(), deque()

        def submit():
            buf = next(it, None)
            if buf is not None:
                pending.append(pool.apply_async(work, (buf,)))   # at most 2 x workers tasks in flight, in order

        for _ in range(2 * a.workers):
            submit()
        ti = -1
        while pending:
            o = pending.popleft().get()
            submit()
            ti += 1
            for k in tot:
                tot[k] += o["stats"][k]
            parts["rec_pos"].append(o["rec_pos"])
            parts["c_rec"].append(o["c_rec"] + rec_base)
            rec_base += o["rec_pos"].size
            parts["t_pos"].append(o["t_pos"])
            parts["sent"].append(o["sent"])
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
            if ti % 50 == 0:
                log(f"task {ti}: {tot}, {len(d1)} w1, {len(d2)} w2, {time.time() - t0:.0f}s")
    rec_pos = np.concatenate(parts["rec_pos"])
    c_rec = np.concatenate(parts["c_rec"])
    t_pos = np.concatenate(parts["t_pos"])
    sent = np.concatenate(parts["sent"])
    w1 = np.concatenate(parts["w1"])
    w2 = np.concatenate(parts["w2"])
    log(f"stream done: {tot}, {t_pos.size} candidates, {time.time() - t0:.0f}s")
    s_pos = rec_pos[c_rec]
    keep = (s_pos >= 0) & (t_pos >= 0) & (t_pos != s_pos)
    counts = {"candidates": int(t_pos.size), "record_not_node": int((s_pos < 0).sum()), "target_not_node": int(((s_pos >= 0) & (t_pos < 0)).sum()),
              "self": int(((s_pos >= 0) & (t_pos == s_pos)).sum())}
    idx = np.flatnonzero(keep)
    key = s_pos[idx] * N_NODES + t_pos[idx]
    _, first = np.unique(key, return_index=True)
    idx = idx[np.sort(first)]
    counts["kept"] = int(idx.size)
    src, dst = s_pos[idx], t_pos[idx]
    rr = c_rec[idx]
    starts = np.r_[True, rr[1:] != rr[:-1]]
    run_start = np.maximum.accumulate(np.where(starts, np.arange(rr.size), 0))
    kidx = np.minimum(np.arange(rr.size) - run_start, 32767).astype(np.int16)
    arr = {"src": src, "dst": dst, "sent": sent[idx], "kidx": kidx, "w1": w1[idx], "w2": w2[idx]}
    # the phrase counts of the kept edges (the stream counted every candidate)
    c1k = np.bincount(arr["w1"], minlength=len(c1))
    c2k = np.bincount(arr["w2"], minlength=len(c2))
    res = {"look": "anchor_extract_hotpot", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "anchor_extract_sha256": ae_sha,
           "stream": tot, "counts": counts, "duplicate_title_positions": n_dup_titles,
           "inputs": {str(p.relative_to(CRAG)): {"bytes": p.stat().st_size, "mtime": p.stat().st_mtime} for p in (TB, NODES, STRUCT)}}
    with np.load(STRUCT) as zf:
        s_src, s_dst, s_rel = zf["src"], zf["dst"], zf["rel"]
    res["structural_rows"] = int(s_src.size)
    res["structural_rel_values"] = np.unique(s_rel).tolist()
    n = src.size
    if s_src.size != N_EDGES or n != N_EDGES:
        (HERE / "anchor_extract_hotpot.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        raise SystemExit(f"edge counts differ: structural {s_src.size}, regenerated {n}")
    if np.array_equal(s_src, src) and np.array_equal(s_dst, dst):
        res["alignment"] = "in_order"
        perm = None
    else:
        ks = s_src.astype(np.int64) * N_NODES + s_dst
        kr = src.astype(np.int64) * N_NODES + dst
        os_, or_ = np.argsort(ks, kind="stable"), np.argsort(kr, kind="stable")
        if not np.array_equal(ks[os_], kr[or_]):
            (HERE / "anchor_extract_hotpot.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            raise SystemExit("the regenerated edges are not the structural rows, even as a set")
        if np.unique(ks).size != ks.size:
            raise SystemExit("structural rows repeat a (src, dst) pair; a permutation alignment is ambiguous")
        perm = np.empty(N_EDGES, dtype=np.int64)
        perm[os_] = or_
        res["alignment"] = "permutation"
        res["in_order_prefix"] = int(np.argmax((s_src != src) | (s_dst != dst)))
        for k in arr:
            arr[k] = arr[k][perm]
        if not (np.array_equal(s_src, arr["src"]) and np.array_equal(s_dst, arr["dst"])):
            raise SystemExit("the permuted edges do not match the structural rows")
    res["checked_rows"] = int(N_EDGES)
    inv1 = [None] * len(d1)
    for s, j in d1.items():
        inv1[j] = s
    inv2 = [None] * len(d2)
    for s, j in d2.items():
        inv2[j] = s
    # vocab ids re-numbered over the kept edges only, by first appearance in structural row order
    out = {}
    for name, inv, ck in (("w1", inv1, c1k), ("w2", inv2, c2k)):
        ids = arr[name]
        used, first_row = np.unique(ids, return_index=True)
        used = used[np.argsort(first_row, kind="stable")]
        remap = np.full(len(inv), -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        arr[name] = remap[ids].astype(np.int32)
        strings = [inv[j] for j in used]
        cnt = ck[used].astype(np.int64)
        if np.bincount(arr[name], minlength=used.size).tolist() != cnt.tolist():
            raise SystemExit(f"{name}: the re-numbered ids do not reproduce the counts")
        res[f"{name}_vocab"] = int(used.size)
        top = np.sort(cnt)[::-1]
        res[f"{name}_coverage_of_top"] = {str(k): round(float(top[:k].sum() / cnt.sum()), 4) for k in (16, 64, 256, 1024, 4096, 16384, 65535)}
        o = np.argsort(-cnt, kind="stable")
        rank = np.empty(o.size, dtype=np.int64)
        rank[o] = np.arange(o.size)
        out[f"{name}r"] = np.minimum(rank[arr[name]], 65535).astype(np.uint16)
        out[f"{name}_top"] = np.asarray([strings[j] for j in o[:TOP]])
        out[f"{name}_top_count"] = cnt[o[:TOP]]
        res[f"{name}_top"] = [[strings[j], int(cnt[j])] for j in o[:(60 if name == "w1" else 120)]]
        out[f"_strings_{name}"] = (strings, cnt)
    np.savez(HERE / "anchors.tmp.npz", w1=arr["w1"], w2=arr["w2"], sent=arr["sent"], kidx=arr["kidx"])
    os.replace(HERE / "anchors.tmp.npz", HERE / "anchors.npz")
    (HERE / "vocab.json").write_text(json.dumps({name: [[s, int(c)] for s, c in zip(*out.pop(f"_strings_{name}"))] for name in ("w1", "w2")},
                                                ensure_ascii=False), encoding="utf-8")
    out["sent"], out["kidx"] = arr["sent"], arr["kidx"]
    (HERE / "host").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "host" / "anchors_compact.tmp.npz", **out)
    os.replace(HERE / "host" / "anchors_compact.tmp.npz", HERE / "host" / "anchors_compact.npz")
    res["anchors_npz_sha256"] = hashlib.sha256((HERE / "anchors.npz").read_bytes()).hexdigest()
    res["compact_sha256"] = hashlib.sha256((HERE / "host" / "anchors_compact.npz").read_bytes()).hexdigest()
    res["compact_bytes"] = (HERE / "host" / "anchors_compact.npz").stat().st_size
    res["seconds"] = round(time.time() - t0, 1)
    (HERE / "anchor_extract_hotpot.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done: {res['alignment']}, counts {counts}, vocab w1 {res['w1_vocab']} w2 {res['w2_vocab']}, {res['seconds']}s")
    log("w2 top: " + ", ".join(f"{s}:{c}" for s, c in res["w2_top"][:40]))


if __name__ == "__main__":
    main()
