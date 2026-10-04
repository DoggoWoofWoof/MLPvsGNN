"""Design look (untracked; not a result and not filed): anchor phrases for musique's structural (title-mention) edges.

musique's structural family is derived (CRAG scratchpad/c3_derived.py): doc A -> doc B iff A's lowercased text holds
B's stripped, lowercased title (4 characters or more) as a word-bounded phrase. Its one relation, 'title_mention',
leaves a typed walk only family x direction. This file gives each stored edge the words A puts before the first such
mention of B's title: a fixed, parameter-free attribute from the corpus text alone (no query, no gold, no fit),
normalised by outputs/mp_approx_2wiki_anchor/anchor_extract.py's phrase() (imported unchanged, sha256 pinned below), so
2wiki's, hotpotqa's and musique's phrases are one definition.

It reads the CRAG package in place, read-only (files opened for reading only; nothing under the root written, moved,
renamed or hashed):
  data/final_canonical/musique/nodes.jsonl            position i = line i: title, text
  data/final_canonical/musique/graph/structural.npz   src, dst (positions), rel
Checks, before any phrase is kept:
  rule   every stored row (A, B) satisfies the builder's rule: B's title passes the length test and
         r'\\b' + re.escape(title) + r'\\b' matches A's lowercased text
  fwd    on 4,000 source positions drawn with a fixed seed, the builder's full rule (its selective-token index, its
         stop list and word pattern, restated here) regenerates exactly the stored destination set
The mention is the first match; when lowercasing changes a text's length the match is redone on the original text with
re.IGNORECASE (counted). A paragraph is one 'sentence' for phrase(): <s> pads only at the paragraph start.

Writes only to its own directory: anchors.npz (w1, w2 int32 ids aligned to structural.npz rows; sent, all 0; kidx, the
row's rank among its source's stored rows), vocab.json, anchor_extract_musique.json (counts, checks, timing) and
host/anchors_compact.npz (make_compact.py's format: ranks by count as uint16, the top 4096 strings).

    python outputs/mp_approx_musique_anchor/anchor_extract_musique.py
"""
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_2wiki_anchor"))
sys.dont_write_bytecode = True
import anchor_extract as AE  # noqa: E402

AE_SHA = "5efb986683fce4190a106e19c2df918da39803df074852f10fb1b8ebce99640d"
CRAG = Path("C:/Users/Swastik/Desktop/CRAG")
NODES = CRAG / "data/final_canonical/musique/nodes.jsonl"
STRUCT = CRAG / "data/final_canonical/musique/graph/structural.npz"
N_NODES = 117534
N_EDGES = 2744076
MIN_TITLE_LEN = 4
# c3_derived.py's index rule, restated (STOP, WORD, the selective token)
STOP = set("the a an of and or to in on at for with by from as is are was were be this that these those "
           "he she it they we you i his her its their our your".split())
WORD = re.compile(r"[A-Za-z0-9]+")
SAMPLE, SAMPLE_SEED = 4000, 20261002
TOP = 4096


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if sha_file(Path(AE.__file__)) != AE_SHA:
        raise SystemExit("anchor_extract.py is not the pinned file")
    t0 = time.time()
    titles, texts = [], []
    with open(NODES, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            titles.append(d["title"])
            texts.append(d["text"])
    if len(titles) != N_NODES:
        raise SystemExit(f"{len(titles)} nodes, not {N_NODES}")
    with np.load(STRUCT) as zf:
        src, dst = zf["src"].astype(np.int64), zf["dst"].astype(np.int64)
        rel = zf["rel"]
    if src.size != N_EDGES or int(rel.max()) != 0 or int(rel.min()) != 0:
        raise SystemExit("structural.npz is not the frozen layout")
    log(f"{N_NODES} nodes, {N_EDGES} edges read, {time.time() - t0:.0f}s")

    tl = [t.strip() for t in titles]
    ok_title = np.asarray([len(t) >= MIN_TITLE_LEN for t in tl])
    pat = {}

    def pattern(j):
        p = pat.get(j)
        if p is None:
            p = pat[j] = re.compile(r"\b" + re.escape(tl[j].lower()) + r"\b")
        return p

    # rule check + phrases, in storage order
    lower = {}
    w1_id, w2_id, w1_v, w2_v = {}, {}, [], []
    W1 = np.empty(N_EDGES, dtype=np.int32)
    W2 = np.empty(N_EDGES, dtype=np.int32)
    kidx = np.empty(N_EDGES, dtype=np.int16)
    seen_k = defaultdict(int)
    counts = {"rows": N_EDGES, "short_title": 0, "no_match": 0, "case_fallback": 0, "paragraph_start_pad": 0}
    bad = []
    for r in range(N_EDGES):
        a, b = int(src[r]), int(dst[r])
        k = seen_k[a]
        seen_k[a] = k + 1
        kidx[r] = min(k, 32767)
        if not ok_title[b]:
            counts["short_title"] += 1
            bad.append((r, a, b, "short"))
            W1[r] = W2[r] = -1
            continue
        low = lower.get(a)
        if low is None:
            if len(lower) > 4096:
                lower.clear()
            low = lower[a] = texts[a].lower()
        m = pattern(b).search(low)
        if m is None:
            counts["no_match"] += 1
            bad.append((r, a, b, "nomatch"))
            W1[r] = W2[r] = -1
            continue
        start = m.start()
        if len(low) != len(texts[a]):
            counts["case_fallback"] += 1
            m2 = re.compile(r"\b" + re.escape(tl[b]) + r"\b", re.IGNORECASE).search(texts[a])
            start = m2.start() if m2 is not None else start
        one, two = AE.phrase(texts[a], start)
        if two.startswith("<s>"):
            counts["paragraph_start_pad"] += 1
        if one not in w1_id:
            w1_id[one] = len(w1_v)
            w1_v.append([one, 0])
        if two not in w2_id:
            w2_id[two] = len(w2_v)
            w2_v.append([two, 0])
        W1[r], W2[r] = w1_id[one], w2_id[two]
        w1_v[W1[r]][1] += 1
        w2_v[W2[r]][1] += 1
        if (r + 1) % 500000 == 0:
            log(f"  {r + 1} rows, {len(w2_v)} w2 phrases, {time.time() - t0:.0f}s")
    log(f"rule check: {counts}, {time.time() - t0:.0f}s")
    if counts["short_title"] or counts["no_match"]:
        (HERE / "anchor_extract_musique.bad.json").write_text(json.dumps({"counts": counts, "first": bad[:200]}, indent=1), encoding="utf-8")
        raise SystemExit(f"{len(bad)} stored rows do not satisfy the builder's rule (anchor_extract_musique.bad.json)")

    # forward check: the builder's full rule on a sample of sources regenerates the stored destination sets
    tokidx = defaultdict(list)
    for j, t in enumerate(tl):
        if len(t) < MIN_TITLE_LEN:
            continue
        toks = [w.lower() for w in WORD.findall(t)]
        cand = [w for w in toks if w not in STOP and len(w) > 2] or toks
        if not cand:
            continue
        tokidx[max(cand, key=len)].append(j)
    order = np.argsort(src, kind="stable")
    ss = src[order]
    rng = np.random.default_rng(SAMPLE_SEED)
    sample = np.sort(rng.choice(N_NODES, SAMPLE, replace=False))
    fwd = {"sources": int(SAMPLE), "equal": 0, "differ": 0, "stored_edges": 0}
    diffs = []
    for a in sample.tolist():
        low = texts[a].lower()
        toks = set(WORD.findall(low))
        got = set()
        for tok in toks:
            for j in tokidx.get(tok, ()):
                if j == a or j in got:
                    continue
                if pattern(j).search(low):
                    got.add(j)
        lo, hi = np.searchsorted(ss, a, "left"), np.searchsorted(ss, a, "right")
        stored = set(dst[order[lo:hi]].tolist())
        fwd["stored_edges"] += len(stored)
        if got == stored:
            fwd["equal"] += 1
        else:
            fwd["differ"] += 1
            if len(diffs) < 50:
                diffs.append({"src": a, "missing": sorted(stored - got)[:10], "extra": sorted(got - stored)[:10]})
    log(f"forward check: {fwd}, {time.time() - t0:.0f}s")

    np.savez(HERE / "anchors.npz", w1=W1, w2=W2, sent=np.zeros(N_EDGES, dtype=np.int8), kidx=kidx)
    voc = {"w1": w1_v, "w2": w2_v}
    (HERE / "vocab.json").write_text(json.dumps(voc, ensure_ascii=False), encoding="utf-8")
    out = {}
    for name, ids in (("w1", W1), ("w2", W2)):
        c = np.asarray([n for _s, n in voc[name]], dtype=np.int64)
        o = np.argsort(-c, kind="stable")
        rank = np.empty(o.size, dtype=np.int64)
        rank[o] = np.arange(o.size)
        if np.bincount(ids, minlength=c.size).tolist() != c.tolist():
            raise SystemExit(f"{name}: the stored ids do not reproduce the vocabulary counts")
        out[f"{name}r"] = np.minimum(rank[ids], 65535).astype(np.uint16)
        out[f"{name}_top"] = np.asarray([voc[name][j][0] for j in o[:TOP]])
        out[f"{name}_top_count"] = c[o[:TOP]]
    out["sent"] = np.zeros(N_EDGES, dtype=np.int8)
    out["kidx"] = kidx
    (HERE / "host").mkdir(exist_ok=True)
    dst_c = HERE / "host" / "anchors_compact.npz"
    tmp = HERE / "host" / "anchors_compact.tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, dst_c)
    top = out["w2_top"][:40].tolist()
    cover = {k: round(float(out["w2_top_count"][:k].sum()) / N_EDGES, 4) for k in (64, 256, 1024, 4096)}
    res = {"script_sha256": sha_file(Path(__file__)), "anchor_extract_sha256": AE_SHA, "counts": counts, "forward_check": fwd,
           "forward_diffs": diffs, "w1_distinct": len(w1_v), "w2_distinct": len(w2_v), "w2_top40": top, "w2_cover": cover,
           "compact_sha256": sha_file(dst_c), "anchors_sha256": sha_file(HERE / "anchors.npz"), "seconds": round(time.time() - t0, 1)}
    (HERE / "anchor_extract_musique.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(json.dumps({k: v for k, v in res.items() if k != "forward_diffs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
