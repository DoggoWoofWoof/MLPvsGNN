"""Diagnostic D4, the words of a link (docs/DIAG_LINK_WORDS.md): do the words around the mention that makes a link say
which neighbour a question follows?

D1 (bridgediag.py) found the golds zrm misses next to golds it finds, ranked 5th to 9th among about 45 such neighbours
on musique by its score. This asks whether the words of the link between a found gold and a candidate rank the missed
gold higher. No training: zrc's fitted weights (the MLP's base) are read on CPU through zrct's read (zrc's), on a copy
of each fit's models.pt and screen.json in a work folder (the sealed fit folders are never written), with bridgediag's
Capture keeping p@swa's score per row. The edges come from the look's chunks (ids, pool sizes, gold flags and each
question's top-five gold count checked against the read), the texts from the served package (u1b.package_root, its
freeze and each nodes.jsonl's sha256 checked), the questions from each dataset's eval split file (never a test split),
the NER mentions from spaCy with the graph's own settings (`ents`; on the host from the laptop's installed files).

A link's words, one window per mention, the mention's own tokens left out:
  structural, passage graphs   the WIN tokens on each side of each occurrence of one end's title in the other's text
  structural, KB graphs        the names of the relations in the edge's slots (outputs/m3b/relations/<ds>_rel_vocab.json)
  NER (musique, squad)         the WIN tokens on each side of each mention, in both passages, of each entity both
                               mention whose corpus df is within the graph's [DF_MIN, DF_MAX]
  kNN                          none
The link score of a candidate c given a found row a: the largest, over the links between them and their windows, of the
summed IDF (log((N + 1) / (df + 1)) over every node's title and text) of the question's stems in the window, the stems
of a's and c's titles left out.

Per read (a fit and a carve), over questions with a gold: zrc's R@5 (as filed: within TOL_FILED of read.json, or exit 1),
D1's C1 and O1 and the same with the fused order (C1w, O1w); over questions zrc finds partly, each missed gold next to a
found gold ranked among the found golds' neighbours outside the top five by zrc's score (base), by the link score then
the score (words) and by the sum of the two ranks then the score (fused).

    python outputs/mp_unified/linkwords.py ents --dataset musique --procs 8 --host
    python outputs/mp_unified/linkwords.py run --fits scr-zrct,scr-zrct-hp --fit-root outputs/screen/fits \\
        --out-dir outputs/diag --threads 8 --host
    python outputs/mp_unified/linkwords.py verdict --runs outputs/diag/linkwords-scr-zrct.json,outputs/diag/linkwords-scr-zrct-hp.json \\
        --out outputs/diag/linkwords-zrc
    python outputs/mp_unified/linkwords.py dev --dataset musique --chunks 3      (a local code check, first-stage score)
    python outputs/mp_unified/linkwords.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import bridgediag as BD  # noqa: E402

ROOT = HERE.parents[1]
WORK = ROOT / "outputs" / "diag" / "linkwords-work"
ENTS = ROOT / "outputs" / "linkwords"
CFG = ROOT / "configs" / "m3b_controlled_comparison.yaml"
REL_DIR = ROOT / "outputs" / "m3b" / "relations"
PYLIB = ROOT / "outputs" / "host_ops" / "pylib"
SPACY_ZIP = PYLIB / "spacy3815_cp313_win_amd64.zip"
SPACY_ZIP_SHA256 = "2eda757009a6685d920b6f07852e24a6b1c6d8438b010977611a76bebc31ce76"
SPACY_DEST = PYLIB / "spacy3815_cp313"
SPACY_VERSION, MODEL_VERSION = "3.8.15", "3.8.0"
CAND, TOL_FILED = BD.CAND, BD.TOL_FILED
order_of, nbrs, pick, log = BD.order_of, BD.nbrs, BD.pick, BD.log

# the words (docs/DIAG_LINK_WORDS.md), fixed before any number
WIN = 10                      # tokens on each side of a mention
STEM = 5                      # a token's stem: its first five characters, lowercased
TOKEN = re.compile(r"[^\W_]+")
STOP = frozenset("""a about above after again against all also am an and any are aren as at be because been before being
below between both but by can cannot could couldn did didn do does doesn doing down during each few for from further had
hadn has hasn have haven having he her here hers herself him himself his how i if in into is isn it its itself just me
more most mustn my myself no nor not now of off on once only or other ought our ours ourselves out over own same shan
she should shouldn so some such than that the their theirs them themselves then there these they this those through to
too under until up upon very was wasn we were weren what when where which while who whom whose why will with would
wouldn you your yours yourself yourselves""".split())
KB = ("metaqa", "webqsp")
NER_WORDS = ("musique", "squad")
ENT_LABELS = ("PERSON", "GPE", "ORG", "LOC", "FAC", "WORK_OF_ART", "EVENT", "PRODUCT", "NORP")   # CRAG ner_edges.py
DF_MIN, DF_MAX = 2, 25
MIN_ENT_CHARS = 3             # len(e.text) > 2
# the rule, fixed before any number
DECIDE = "musique"
LE2_ABS, LE2_REL = 0.05, 1.25
REPRO_MIN = 0.9
EPS = 1e-12


# ── words ────────────────────────────────────────────────────────────────────


def words_of(text):
    """A text's word tokens (runs of letters and digits, lowercased) with their character spans."""
    low, st, en = [], [], []
    for m in TOKEN.finditer(text):
        low.append(m.group().lower())
        st.append(m.start())
        en.append(m.end())
    return low, np.asarray(st, np.int64), np.asarray(en, np.int64)


def stems_of(low):
    return [w[:STEM] for w in low]


def question_stems(question):
    """The question's stems: its tokens less the stop words and single characters, cut to STEM characters."""
    return frozenset(w[:STEM] for w in (x.lower() for x in TOKEN.findall(question)) if len(w) > 1 and w not in STOP)


class Doc:
    """One node: its title's tokens and stems, its text's tokens and stems, its NER mentions as token ranges."""
    __slots__ = ("title_low", "title_stems", "low", "stem", "ments")

    def __init__(self, title, text, mentions=()):
        self.title_low = [x.lower() for x in TOKEN.findall(title or "")]
        self.title_stems = frozenset(stems_of(self.title_low))
        self.low, st, en = words_of(text or "")
        self.stem = stems_of(self.low)
        self.ments = []
        for key, s0, e0 in mentions:
            i0 = int(np.searchsorted(en, s0, side="right"))        # the first token ending after the mention starts
            i1 = int(np.searchsorted(st, e0, side="left"))         # past the last token starting before it ends
            if i1 > i0:
                self.ments.append((int(key), i0, i1))


def window(stem, i0, i1, win=WIN):
    """The stems of up to win tokens on each side of tokens [i0, i1), those tokens left out."""
    return stem[max(0, i0 - win):i0] + stem[i1:i1 + win]


def occurrences(hay, needle):
    """Where needle (a token list) starts as consecutive tokens of hay."""
    m = len(needle)
    if m == 0 or m > len(hay):
        return []
    first = needle[0]
    return [i for i, w in enumerate(hay) if w == first and hay[i:i + m] == needle]


def title_windows(dx, dy):
    """Windows around each occurrence of dy's title in dx's text."""
    m = len(dy.title_low)
    return [window(dx.stem, i, i + m) for i in occurrences(dx.low, dy.title_low)]


def ner_windows(dx, dy):
    """Windows around each mention, in both texts, of each entity both mention."""
    if not dx.ments or not dy.ments:
        return []
    shared = {k for k, _, _ in dx.ments} & {k for k, _, _ in dy.ments}
    return [window(d.stem, i0, i1) for d in (dx, dy) for k, i0, i1 in d.ments if k in shared]


def link_score(da, dc, links, q, idf, kb_names=None):
    """(score, family, windows): the largest summed IDF of the question's stems (less a's and c's title stems) in one
    window of the links between a and c. links: (family, relation slots) per edge."""
    qq = q - da.title_stems - dc.title_stems
    best, fam_best, nwin = 0.0, -1, 0
    for fam, rels in links:
        if fam == 0:
            if kb_names is not None:
                wins = [kb_names[r] for r in rels if r >= 0]
            else:
                wins = title_windows(da, dc) + title_windows(dc, da)
        elif fam == 1:
            wins = ner_windows(da, dc)
        else:
            wins = []
        nwin += len(wins)
        if not qq:
            continue
        for w in wins:
            sc = sum(idf[x] for x in qq.intersection(w))
            if sc > best:
                best, fam_best = sc, fam
    return best, fam_best, nwin


# ── one question ─────────────────────────────────────────────────────────────


class QRec:
    """One question's read: pool scores, gold flags, gold total, node ids, and the edges at its anchors."""
    __slots__ = ("s", "g", "gt", "pool", "u", "v", "fam", "rel")

    def __init__(self, s, g, gt, pool, u, v, fam, rel):
        self.s, self.g, self.gt, self.pool = s, g, gt, pool
        self.u, self.v, self.fam, self.rel = u, v, fam, rel


def anchors_of(s, g):
    """The rows whose neighbourhoods the reads need: the top-1 (C1), the highest-ranked found gold (O1) and, when the
    question is partly found, every found gold (the nb ranks); and the count of golds in the top five."""
    o = order_of(s)
    top = o[:5]
    found = [int(r) for r in top if g[r]]
    in_top = np.zeros(s.size, bool)
    in_top[top] = True
    a = {int(o[0])} if s.size else set()
    if found:
        a.add(found[0])
        if (g & ~in_top).any():
            a.update(found)
    return sorted(a), len(found)


def keep_edges(qr_s, g, gt, pool, u, v, fam, rel):
    """A QRec with only the edges at the anchors (every read below uses no other)."""
    a, n_found = anchors_of(qr_s, g)
    A = np.asarray(a, np.int64)
    m = np.isin(u, A) | np.isin(v, A)
    return QRec(qr_s, g, int(gt), pool, u[m].astype(np.int32), v[m].astype(np.int32), fam[m].astype(np.int8),
                None if rel is None else rel[m].astype(np.int16)), n_found


def ranks(idx, s, w):
    """Base, words and fused ranks (1 = first) of rows idx: base by score (bridgediag's order); words by link score,
    then base; fused by the sum of the two ranks, then base."""
    m = idx.size
    pos = np.empty(m, np.int64)
    pos[order_of(s[idx])] = np.arange(m)
    w = np.asarray(w, np.float64)
    rw = np.empty(m, np.int64)
    rw[np.lexsort((pos, -w))] = np.arange(1, m + 1)
    rb = pos + 1
    rf = np.empty(m, np.int64)
    rf[np.lexsort((pos, rb + rw))] = np.arange(1, m + 1)
    return rb, rw, rf


def pick_fused(order, keep, anchor, cnt, s, u, v, n, lw, k=5):
    """bridgediag.pick with one anchor whose neighbours not yet chosen are taken in the fused order."""
    chosen = [int(x) for x in keep]
    taken = np.zeros(n, bool)
    taken[chosen] = True
    idx = np.flatnonzero(nbrs(n, [anchor], u, v) & ~taken)
    if idx.size:
        _, _, rf = ranks(idx, s, [lw(anchor, int(c))[0] for c in idx])
        sel = idx[np.argsort(rf, kind="stable")][:cnt]
        chosen += [int(x) for x in sel]
        taken[sel] = True
    for r in order:
        if len(chosen) >= k:
            break
        if not taken[r]:
            chosen.append(int(r))
            taken[r] = True
    return np.asarray(chosen[:k], np.int64)


def words_question(qr, score):
    """The counts of one question. score(node_a, node_c, links) -> (link score, family, windows)."""
    s, g, u, v = qr.s, qr.g, qr.u, qr.v
    n = s.size
    o = order_of(s)
    top = o[:5]
    in_top = np.zeros(n, bool)
    in_top[top] = True
    found = [int(r) for r in top if g[r]]
    missed = np.flatnonzero(g & ~in_top)
    links = {}
    for e in range(u.size):
        a, b = int(u[e]), int(v[e])
        links.setdefault((a, b) if a < b else (b, a), []).append(
            (int(qr.fam[e]), tuple(int(x) for x in qr.rel[e]) if qr.rel is not None else ()))
    memo = {}

    def lw(a, c):
        key = (a, c) if a < c else (c, a)
        if key not in memo:
            ls = links.get(key)
            memo[key] = score(int(qr.pool[a]), int(qr.pool[c]), ls) if ls else (0.0, -1, 0)
        return memo[key]

    def hits(sel):
        return int(g[sel].sum())
    out = {"gt": int(qr.gt), "zrc": len(found)}
    out["C1"] = hits(pick(o, o[:3], [(int(o[0]), 2)], s, u, v, n)) if n else 0
    out["C1w"] = hits(pick_fused(o, o[:3], int(o[0]), 2, s, u, v, n, lw)) if n else 0
    out["O1"] = hits(pick(o, o[:3], [(found[0], 2)], s, u, v, n)) if found else 0
    out["O1w"] = hits(pick_fused(o, o[:3], found[0], 2, s, u, v, n, lw)) if found else 0
    if found and missed.size:
        adj = nbrs(n, found, u, v)
        out["partial"] = 1
        out["missed"] = int(missed.size)
        out["missed_adj"] = int(adj[missed].sum())
        cand = np.flatnonzero(adj & ~in_top)
        if cand.size:
            sc = []
            for c in cand:
                c = int(c)
                best, nw = (0.0, -1, 0), 0
                for a in found:
                    if ((a, c) if a < c else (c, a)) in links:
                        r = lw(a, c)
                        nw += r[2]
                        if r[0] > best[0]:
                            best = r
                sc.append((best[0], best[1], nw))
            w = np.asarray([x[0] for x in sc], np.float64)
            rb, rw, rf = ranks(cand, s, w)
            at = {int(c): i for i, c in enumerate(cand)}
            out["nb"] = [[int(rb[at[int(m)]]), int(rw[at[int(m)]]), int(rf[at[int(m)]]), float(w[at[int(m)]]),
                          int(sc[at[int(m)]][1]), int(sc[at[int(m)]][2])] for m in missed if adj[m]]
            out["nb_size"] = int(cand.size)
            out["cand_words"] = int((w > 0).sum())
            out["cand_windows"] = int(sum(1 for x in sc if x[2] > 0))
    return out


def summarise(rows):
    """Means over questions with golds, and the ranks over the partly found."""
    w = [r for r in rows if r["gt"] > 0]
    res = {"questions": len(w)}
    for k in ("zrc", "C1", "C1w", "O1", "O1w"):
        res[k] = {"R@5": float(np.mean([min(r[k], r["gt"]) / r["gt"] for r in w])) if w else None,
                  "FC@5": float(np.mean([r[k] >= r["gt"] for r in w])) if w else None}
    if w:
        for k in ("C1", "C1w", "O1", "O1w"):
            res[k]["dR@5"] = res[k]["R@5"] - res["zrc"]["R@5"]
        res["C1w"]["vs_C1"] = res["C1w"]["R@5"] - res["C1"]["R@5"]
        res["O1w"]["vs_O1"] = res["O1w"]["R@5"] - res["O1"]["R@5"]
    p = [r for r in w if r.get("partial")]
    res["partial_questions"] = len(p)
    res["missed_in_pool"] = sum(r["missed"] for r in p)
    res["missed_adj"] = sum(r["missed_adj"] for r in p)
    nb = [x for r in p for x in r.get("nb", [])]
    res["ranked"] = len(nb)
    res["le2"] = res["le1"] = res["median_rank"] = None
    if nb:
        a = np.asarray([x[:3] for x in nb], np.int64)
        names = ("base", "words", "fused")
        res["le2"] = {k: float(np.mean(a[:, i] <= 2)) for i, k in enumerate(names)}
        res["le1"] = {k: float(np.mean(a[:, i] <= 1)) for i, k in enumerate(names)}
        res["median_rank"] = {k: float(np.median(a[:, i])) for i, k in enumerate(names)}
        res["missed_with_window"] = float(np.mean([x[5] > 0 for x in nb]))
        res["missed_with_words"] = float(np.mean([x[3] > 0 for x in nb]))
        fam = [x[4] for x in nb if x[3] > 0]
        res["missed_words_family"] = {str(f): fam.count(f) for f in sorted(set(fam))}
        sz = [r["nb_size"] for r in p if "nb_size" in r]
        res["nb_size_median"] = float(np.median(sz)) if sz else None
        tot = sum(sz)
        res["cand_with_window"] = sum(r.get("cand_windows", 0) for r in p) / tot if tot else None
        res["cand_with_words"] = sum(r.get("cand_words", 0) for r in p) / tot if tot else None
    return res


# ── texts, questions, mentions, relations ────────────────────────────────────


def corpus_pass(served, ds, need, qset):
    """One pass over the dataset's nodes.jsonl (sha256 checked against DATASET.json): each question stem's df over the
    nodes' titles and texts, and the needed nodes' titles and texts."""
    import u1b as U1B
    man = json.loads((served / ds / "DATASET.json").read_text(encoding="utf-8"))
    want, n = man["nodes"]["sha256"], int(man["nodes"]["n"])
    h = hashlib.sha256()
    df = dict.fromkeys(qset, 0)
    texts = {}
    i = -1
    with open(served / ds / "nodes.jsonl", "rb") as f:
        for i, raw in enumerate(f):
            h.update(raw)
            rec = json.loads(raw)
            title, text = U1B.title_of(rec), rec["text"]
            for w in {x.lower()[:STEM] for x in TOKEN.findall(f"{title} {text}")} & qset:
                df[w] += 1
            if i in need:
                texts[i] = (title, text)
    if i + 1 != n or h.hexdigest() != want:
        raise SystemExit(f"linkwords: {ds}/nodes.jsonl has {i + 1} lines and sha256 {h.hexdigest()}; DATASET.json "
                         f"names {n} and {want}")
    if len(texts) != len(need):
        raise SystemExit(f"linkwords: {ds}: {len(need) - len(texts)} needed nodes are not in nodes.jsonl")
    idf = {w: math.log((n + 1) / (df[w] + 1)) for w in qset}
    return texts, idf, {"nodes": n, "nodes_sha256": want, "need": len(need)}


def questions(served, ds, split):
    """id -> question from the dataset's eval split file (webqsp's train_holdout from train); never a test split."""
    fn = "train" if split == "train_holdout" else split
    if fn == "test":
        raise SystemExit(f"linkwords: {ds}: the eval split is test; refusing")
    out = {}
    with open(served / ds / "queries" / f"{fn}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["query_id"]] = r["question"]
    return out


def kb_names(ds, rel_dir=REL_DIR):
    v = json.loads((Path(rel_dir) / f"{ds}_rel_vocab.json").read_text(encoding="utf-8"))["vocab"]
    return [stems_of([x.lower() for x in TOKEN.findall(name)]) for name in v]


def arrays_sha(z, names):
    h = hashlib.sha256()
    for k in names:
        a = np.ascontiguousarray(z[k])
        h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


ENT_ARRAYS = ("node", "key", "start", "end", "label", "df")


def load_ents(ds, nodes_sha, root=ENTS):
    """mentions(node) -> [(key, start, end)] with the key's df within [DF_MIN, DF_MAX], and the ents record; or
    (None, None) when the dataset has none."""
    p, rj = Path(root) / f"{ds}_ents.npz", Path(root) / f"{ds}_ents.json"
    if not p.exists():
        return None, None
    rec = json.loads(rj.read_text(encoding="utf-8"))
    z = np.load(p)
    if arrays_sha(z, ENT_ARRAYS) != rec["arrays_sha256"] or rec["nodes_sha256"] != nodes_sha:
        raise SystemExit(f"linkwords: {p} is not its record's, or was made on other nodes")
    node, key, df = z["node"].astype(np.int64), z["key"].astype(np.int64), z["df"].astype(np.int64)
    keep = (df[key] >= DF_MIN) & (df[key] <= DF_MAX)
    node, key = node[keep], key[keep]
    start, end = z["start"][keep].astype(np.int64), z["end"][keep].astype(np.int64)
    if node.size and np.any(np.diff(node) < 0):
        raise SystemExit(f"linkwords: {p}: mentions are not in node order")
    off = np.searchsorted(node, np.arange(int(rec["nodes"]) + 1))

    def mentions(i):
        a, b = int(off[i]), int(off[i + 1])
        return list(zip(key[a:b].tolist(), start[a:b].tolist(), end[a:b].tolist()))
    return mentions, rec


# ── NER mentions (spaCy, the graph's settings) ───────────────────────────────


def spacy_path(host):
    """On the host: the laptop's installed spaCy closure, unpacked once from the pushed zip (sha256 checked), for the
    END of sys.path so the env's own packages come first. On the laptop: None (the installed spaCy)."""
    if not host:
        return None
    sha = hashlib.sha256(SPACY_ZIP.read_bytes()).hexdigest()
    if sha != SPACY_ZIP_SHA256:
        raise SystemExit(f"linkwords: {SPACY_ZIP.name} sha256 {sha} != {SPACY_ZIP_SHA256}")
    mark = SPACY_DEST / "unpacked_from.sha256"
    if mark.exists() and mark.read_text().strip() == sha:
        return str(SPACY_DEST)
    if SPACY_DEST.exists():
        raise SystemExit(f"{SPACY_DEST} holds another unpack: move it aside first")
    tmp = SPACY_DEST.with_name(SPACY_DEST.name + f".tmp{os.getpid()}")
    with zipfile.ZipFile(SPACY_ZIP) as z:
        z.extractall(tmp)
    (tmp / "unpacked_from.sha256").write_text(sha)
    os.replace(tmp, SPACY_DEST)
    return str(SPACY_DEST)


_NLP = None


def _spacy_init(path):
    global _NLP
    if path and path not in sys.path:
        sys.path.append(path)
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])


def _ents_batch(task):
    lo, texts = task
    out = []
    for j, doc in enumerate(_NLP.pipe(texts, batch_size=256)):
        for e in doc.ents:
            if e.label_ in ENT_LABELS and len(e.text) >= MIN_ENT_CHARS:
                out.append((lo + j, e.text.lower().strip(), e.start_char, e.end_char, ENT_LABELS.index(e.label_)))
    return out


def aggregate(rows, n):
    """Arrays from (node, key text, start, end, label) rows in node order: keys sorted, df = distinct nodes per key."""
    keys = sorted({r[1] for r in rows})
    kid = {k: i for i, k in enumerate(keys)}
    node = np.asarray([r[0] for r in rows], np.int32)
    key = np.asarray([kid[r[1]] for r in rows], np.int32)
    arr = {"node": node, "key": key, "start": np.asarray([r[2] for r in rows], np.int32),
           "end": np.asarray([r[3] for r in rows], np.int32), "label": np.asarray([r[4] for r in rows], np.int8)}
    pairs = np.unique(key.astype(np.int64) * n + node) if rows else np.zeros(0, np.int64)
    arr["df"] = np.bincount(pairs // n, minlength=len(keys)).astype(np.int32)
    return arr, keys


def rebuild_ner(key, node, df, n):
    """The graph's NER edges from the mentions (CRAG ner_edges.py): node pairs sharing a key with DF_MIN <= df <= DF_MAX,
    weight the sum of 1/df. Returns sorted pair codes a * n + b (a < b) and weights."""
    keep = (df[key] >= DF_MIN) & (df[key] <= DF_MAX)
    u = np.unique(key[keep].astype(np.int64) * n + node[keep])
    kk, nn = u // n, u % n
    cuts = np.flatnonzero(np.diff(kk)) + 1
    codes, ws = [], []
    for grp_k, grp in zip(np.split(kk, cuts), np.split(nn, cuts)):
        if grp.size < 2:
            continue
        i, j = np.triu_indices(grp.size, 1)
        codes.append(grp[i] * n + grp[j])
        ws.append(np.full(i.size, 1.0 / float(df[grp_k[0]])))
    if not codes:
        return np.zeros(0, np.int64), np.zeros(0)
    c, w = np.concatenate(codes), np.concatenate(ws)
    uc, inv = np.unique(c, return_inverse=True)
    return uc, np.bincount(inv, weights=w)


def repro(arr, n, src, dst, weight):
    """How much of the graph's NER family the mentions rebuild."""
    rc, rw = rebuild_ner(arr["key"].astype(np.int64), arr["node"].astype(np.int64), arr["df"].astype(np.int64), n)
    a, b = np.minimum(src, dst).astype(np.int64), np.maximum(src, dst).astype(np.int64)
    gc, gi = np.unique(a * n + b, return_index=True)
    common, ri, gj = np.intersect1d(rc, gc, return_indices=True)
    diff = np.abs(rw[ri] - weight[gi][gj]) if common.size else np.zeros(0)
    return {"graph_edges": int(gc.size), "rebuilt_edges": int(rc.size), "common": int(common.size),
            "reproduced": common.size / gc.size if gc.size else None,
            "extra_share": (rc.size - common.size) / rc.size if rc.size else None,
            "weight_max_abs_diff": float(diff.max()) if diff.size else None,
            "weight_within_1e-4": float(np.mean(diff <= 1e-4)) if diff.size else None}


def ents_cmd(a):
    import multiprocessing as mp
    import u1b as U1B
    t0 = time.time()
    path = spacy_path(a.host)
    if path:
        sys.path.append(path)
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    if spacy.__version__ != SPACY_VERSION or nlp.meta.get("version") != MODEL_VERSION:
        raise SystemExit(f"linkwords: spaCy {spacy.__version__} / model {nlp.meta.get('version')}, declared "
                         f"{SPACY_VERSION} / {MODEL_VERSION}")
    del nlp
    served, freeze = U1B.package_root(a.host)
    _titles, texts, n, nodes_sha = U1B.read_nodes(served, a.dataset)
    log(f"ents {a.dataset}: {n} nodes; spaCy {SPACY_VERSION} from {path or 'the installed packages'}, {a.procs} procs")
    tasks = [(lo, texts[lo:lo + a.batch]) for lo in range(0, n, a.batch)]
    rows = []
    with mp.get_context("spawn").Pool(a.procs, initializer=_spacy_init, initargs=(path,)) as pool:
        for i, out in enumerate(pool.imap(_ents_batch, tasks)):
            rows.extend(out)
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                log(f"  {min((i + 1) * a.batch, n)}/{n} nodes, {len(rows)} mentions, {time.time() - t0:.0f}s")
    arr, keys = aggregate(rows, n)
    gz = np.load(served / a.dataset / "graph" / "ner.npz")
    rp = repro(arr, n, gz["src"], gz["dst"], gz["weight"].astype(np.float64))
    ENTS.mkdir(parents=True, exist_ok=True)
    out = ENTS / f"{a.dataset}_ents.npz"
    tmp = out.with_name(out.stem + ".tmp.npz")
    blob = np.frombuffer("\n".join(k.replace("\n", " ") for k in keys).encode("utf-8"), np.uint8)
    np.savez_compressed(tmp, keys_blob=blob, **arr)
    os.replace(tmp, out)
    rec = {"dataset": a.dataset, "nodes": n, "nodes_sha256": nodes_sha, "freeze_RECORD_SHA256": freeze,
           "spacy": SPACY_VERSION, "model": MODEL_VERSION, "from": "host zip " + SPACY_ZIP_SHA256 if path else "laptop",
           "settings": {"disable": ["parser", "lemmatizer"], "labels": list(ENT_LABELS), "key": "e.text.lower().strip()",
                        "min_chars": MIN_ENT_CHARS, "df": [DF_MIN, DF_MAX]},
           "mentions": int(arr["node"].size), "keys": len(keys),
           "keys_in_df_range": int(((arr["df"] >= DF_MIN) & (arr["df"] <= DF_MAX)).sum()),
           "arrays_sha256": arrays_sha(arr, ENT_ARRAYS), "graph": rp, "procs": a.procs,
           "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    BD.write(ENTS / f"{a.dataset}_ents", rec, f"# NER mentions, {a.dataset}\n\n```\n{json.dumps(rec, indent=1)}\n```\n")
    log(f"ents {a.dataset}: {rec['mentions']} mentions, {rec['keys']} keys; the graph's NER edges rebuilt "
        f"{rp['reproduced']}; {rec['seconds']:.0f}s")
    return 0


# ── the reads ────────────────────────────────────────────────────────────────


def plan_carve(c, s, top_filed, look_root=None):
    """Each question's QRec from the look's chunks, checked against the read row for row."""
    import lean_cache as LC
    _d, files, ids, _head = LC.look_chunks(c.ds, c.carve, look_root or LC.LM.LOOK)
    if list(ids) != list(c.ids):
        raise SystemExit(f"linkwords: {c.ds}/{c.carve}: the look's ids are not the cache's")
    gold_c = c.gold.cpu().numpy() if hasattr(c.gold, "cpu") else np.asarray(c.gold)
    recs, qi, r_at = [], 0, 0
    for f in files:
        z = np.load(f)
        qps, qed = z["q_pool_size"].astype(np.int64), z["q_edges"].astype(np.int64)
        isg, pool = z["is_gold"].astype(bool), z["pool"].astype(np.int32)
        eu, ev, ef = z["e_u"].astype(np.int64), z["e_v"].astype(np.int64), z["e_fam"].astype(np.int64)
        er = z["e_rel"] if "e_rel" in z.files else None
        no = np.concatenate([[0], np.cumsum(qps)])
        eo = np.concatenate([[0], np.cumsum(qed)])
        for i in range(qps.size):
            n = int(qps[i])
            if n != int(c.n_np[qi]):
                raise SystemExit(f"linkwords: {c.ds}/{c.carve} question {qi}: pool {n} in the look, {c.n_np[qi]} in the cache")
            g = isg[no[i]:no[i + 1]]
            if not np.array_equal(g, gold_c[r_at:r_at + n]):
                raise SystemExit(f"linkwords: {c.ds}/{c.carve} question {qi}: gold flags differ between look and cache")
            e0, e1 = eo[i], eo[i + 1]
            qr, n_found = keep_edges(np.asarray(s[r_at:r_at + n], np.float32).copy(), g.copy(), c.gt_np[qi],
                                     pool[no[i]:no[i + 1]].copy(), eu[e0:e1], ev[e0:e1], ef[e0:e1],
                                     None if er is None else er[e0:e1])
            if top_filed is not None and n_found != int(top_filed[qi]):
                raise SystemExit(f"linkwords: {c.ds}/{c.carve} question {qi}: {n_found} golds in the top five, the read "
                                 f"counted {int(top_filed[qi])}")
            recs.append(qr)
            qi += 1
            r_at += n
    if qi != c.n_np.size or r_at != int(c.off_np[-1]):
        raise SystemExit(f"linkwords: {c.ds}/{c.carve}: the look covers {qi} questions and {r_at} rows")
    return recs


def r_at5(recs):
    v = []
    for qr in recs:
        if qr.gt > 0:
            top = order_of(qr.s)[:5]
            v.append(min(int(qr.g[top].sum()), qr.gt) / qr.gt)
    return float(np.mean(v)) if v else None


def words_dataset(served, ds, split, items, rel_dir=REL_DIR, ents_root=ENTS):
    """The words of every question of one dataset, for every read of it (items: (fit, key, ids, QRecs)); one corpus
    pass. Returns {(fit, key): summary} and the dataset's record."""
    t0 = time.time()
    qt = questions(served, ds, split)
    qs_of = {}
    for _fit, key, ids, _recs in items:
        miss = [q for q in ids if q not in qt]
        if miss:
            raise SystemExit(f"linkwords: {ds}: {len(miss)} questions are not in the {split} file (first {miss[:3]})")
        qs_of[key] = [question_stems(qt[q]) for q in ids]
    qset = frozenset().union(*(x for v in qs_of.values() for x in v))
    need = set()
    for it in items:
        for qr in it[3]:
            if qr.u.size:
                need.update(qr.pool[np.unique(np.concatenate([qr.u, qr.v]))].tolist())
    texts, idf, info = corpus_pass(served, ds, need, qset)
    mentions, erec = load_ents(ds, info["nodes_sha256"], ents_root) if ds in NER_WORDS else (None, None)
    if ds == DECIDE and mentions is None:
        raise SystemExit(f"linkwords: {ds}'s NER mentions are missing; run `ents --dataset {ds}` first")
    kbn = kb_names(ds, rel_dir) if ds in KB else None

    @functools.lru_cache(maxsize=40000)
    def doc(i):
        t = texts[i]
        return Doc(t[0], t[1], mentions(i) if mentions else ())

    out = {}
    for fit, key, _ids, recs in items:
        if kbn is not None:
            mx = max((int(qr.rel.max()) for qr in recs if qr.rel is not None and qr.rel.size), default=-1)
            if mx >= len(kbn):
                raise SystemExit(f"linkwords: {ds}: relation slot {mx}, the vocabulary has {len(kbn)} names")
        rows = []
        qs = qs_of[key]
        if len(qs) != len(recs):
            raise SystemExit(f"linkwords: {ds}/{key}: {len(recs)} questions read, {len(qs)} ids")
        for j, qr in enumerate(recs):
            q = qs[j]
            rows.append(words_question(qr, lambda x, y, links, q=q: link_score(doc(x), doc(y), links, q, idf, kbn)))
        sm = summarise(rows)
        sm["words"] = {"structural": "relation names" if kbn is not None else "title occurrences",
                       "ner": mentions is not None}
        out[(fit, key)] = sm
    info.update({"split": split, "question_stems": len(qset), "seconds": time.time() - t0,
                 "ents": None if erec is None else {k: erec[k] for k in ("arrays_sha256", "mentions", "keys", "graph",
                                                                         "from")}})
    doc.cache_clear()
    return out, info


def run_cmd(a):
    import torch
    import yaml
    import u1b as U1B
    torch.set_num_threads(a.threads)
    import zrct as ZT
    LG, LC = ZT.LG, ZT.LC
    t0 = time.time()
    fits = [f for f in a.fits.split(",") if f]
    served, freeze = U1B.package_root(a.host)
    splits = yaml.safe_load(CFG.read_text(encoding="utf-8"))["populations"]["eval_splits"]
    cap = BD.Capture(LG)
    plans, recs = {}, {}
    for fit in fits:
        src = Path(a.fit_root) / fit
        work = WORK / fit
        work.mkdir(parents=True, exist_ok=True)
        for fn in ("models.pt", "screen.json"):
            shutil.copyfile(src / fn, work / fn)
            if LC.sha_file(src / fn) != LC.sha_file(work / fn):
                raise SystemExit(f"linkwords: the copy of {fn} differs")
        filed = json.loads((src / "read.json").read_text(encoding="utf-8"))
        recs[fit] = {"fit": fit, "fit_root": Path(a.fit_root).as_posix(), "candidate": CAND,
                     "arm": json.loads((src / "screen.json").read_text(encoding="utf-8"))["arm"],
                     "models_sha256": LC.sha_file(src / "models.pt"), "script_sha256": LC.sha_src(__file__),
                     "freeze_RECORD_SHA256": freeze, "reads": {}, "identity_reads": {}}

        def on_carve(c, s, base, top_k, fit=fit, filed=filed):
            key = f"{c.ds}={c.carve}"
            qrs = plan_carve(c, s, top_k)
            if c.carve == "s1eval":                    # the words are read on the s1eval carves only (as declared)
                plans[(fit, key)] = (c.ds, list(c.ids), qrs)
            fr = filed["carves"].get(key, {}).get("swa_means", {}).get(CAND)
            r5 = r_at5(qrs)
            ok = fr is not None and r5 is not None and abs(r5 - fr[0]) <= TOL_FILED
            recs[fit]["identity_reads"][key] = {"R@5": r5, "filed_R@5": fr[0] if fr else None, "ok": ok}
            log(f"  {fit} {key}: zrc R@5 {r5} (filed {fr[0] if fr else None}) {'ok' if ok else 'DIFFERS'}")
        cap.on_carve = on_carve
        argv = ["read", "--name", fit, "--out-root", str(WORK), "--device", a.device, "--threads", str(a.threads)]
        if a.host:
            argv.append("--host")
        rc = ZT.main(argv)
        if rc not in (0, None):
            raise SystemExit(f"linkwords: zrc's read of {fit} exited {rc}")
    bad = {f: [k for k, v in r["identity_reads"].items() if not v["ok"]] for f, r in recs.items()}
    for f, r in recs.items():
        r["identity"] = "PASS" if not bad[f] and r["identity_reads"] else "FAIL"
        r["identity_fail"] = bad[f]
    if any(r["identity"] != "PASS" for r in recs.values()):
        for f, r in recs.items():
            write_run(Path(a.out_dir), r)
        log(f"linkwords: identity failed {bad}; no words are read")
        return 1
    by_ds = {}
    for (fit, key), (ds, ids, qrs) in plans.items():
        by_ds.setdefault(ds, []).append((fit, key, ids, qrs))
    texts_rec = {}
    for ds in sorted(by_ds, key=lambda d: (d != DECIDE, d)):
        out, info = words_dataset(served, ds, splits[ds], by_ds[ds])
        for (fit, key), sm in out.items():
            sm.update(recs[fit]["identity_reads"][key])
            recs[fit]["reads"][key] = sm
        texts_rec[ds] = info
        for it in by_ds[ds]:
            plans.pop((it[0], it[1]))
        m = out.get((fits[0], f"{ds}=s1eval"), {})
        log(f"  {ds}: {info['need']} nodes read in {info['seconds']:.0f}s; {fits[0]} le2 {m.get('le2')}")
        for f, r in recs.items():                      # kept as each dataset is done (a later failure keeps these)
            r["texts"] = texts_rec
            r["ents"] = {d: v["ents"] for d, v in texts_rec.items() if v.get("ents")}
            r["complete"] = False
            write_run(Path(a.out_dir), r)
    for f, r in recs.items():
        r["complete"] = True
        r["texts"] = texts_rec
        r["ents"] = {ds: v["ents"] for ds, v in texts_rec.items() if v.get("ents")}
        r["seconds"] = time.time() - t0
        r["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_run(Path(a.out_dir), r)
    v = verdict([recs[f] for f in fits])
    if a.verdict_out:
        write_verdict(Path(a.verdict_out), v, [recs[f] for f in fits])
    log(f"linkwords: {v['verdict']} in {time.time() - t0:.0f}s")
    return 0


# ── the verdict and the tables ───────────────────────────────────────────────


def verdict(runs):
    """The declared rule on musique's reads, one run per fit."""
    if len(runs) != 2:
        return {"verdict": "INCOMPLETE", "why": f"{len(runs)} runs, the rule reads two fits"}
    passes = {}
    for r in runs:
        if r.get("identity") != "PASS":
            return {"verdict": "INCOMPLETE", "why": f"{r['fit']}: identity {r.get('identity')}"}
        rd = {k.split("=")[0]: v for k, v in r["reads"].items()}
        m = rd.get(DECIDE)
        rep = ((r.get("ents") or {}).get(DECIDE) or {}).get("graph", {}).get("reproduced")
        if rep is None or rep < REPRO_MIN:
            return {"verdict": "INCOMPLETE", "why": f"{r['fit']}: the mentions rebuild {rep} of {DECIDE}'s NER edges"}
        if not m or not m.get("le2"):
            return {"verdict": "INCOMPLETE", "why": f"{r['fit']}: {DECIDE} ranks no missed gold"}
        b, f = m["le2"]["base"], m["le2"]["fused"]
        passes[r["fit"]] = {"base": b, "fused": f,                     # "at least": equality passes
                            "pass": bool(f - b >= LE2_ABS - EPS and f >= LE2_REL * b - EPS)}
    k = sum(p["pass"] for p in passes.values())
    return {"verdict": "SIGNAL" if k == 2 else "WEAK" if k == 1 else "NO_SIGNAL", "fits": passes,
            "rule": {"dataset": DECIDE, "le2_abs": LE2_ABS, "le2_rel": LE2_REL, "repro_min": REPRO_MIN}}


def f_(x, p=4):
    return "" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


def table(r):
    lines = ["| read | questions | zrc R@5 | filed | partly found | missed in pool | next to a found gold | ranked | "
             "nb median size | le2 base | le2 words | le2 fused | median rank b / w / f | missed with window | "
             "missed with words | candidates with words | C1 | C1w | O1 | O1w |",
             "| --- |" + " ---: |" * 19]
    for k, v in r["reads"].items():
        le2, mr = v.get("le2") or {}, v.get("median_rank") or {}
        lines.append(
            f"| {k} | {v['questions']} | {f_(v['zrc']['R@5'])} | {f_(v.get('filed_R@5'))} | {v['partial_questions']} | "
            f"{v['missed_in_pool']} | {v['missed_adj']} | {v['ranked']} | {f_(v.get('nb_size_median'), 1)} | "
            f"{f_(le2.get('base'), 3)} | {f_(le2.get('words'), 3)} | {f_(le2.get('fused'), 3)} | "
            f"{f_(mr.get('base'), 1)} / {f_(mr.get('words'), 1)} / {f_(mr.get('fused'), 1)} | "
            f"{f_(v.get('missed_with_window'), 3)} | {f_(v.get('missed_with_words'), 3)} | "
            f"{f_(v.get('cand_with_words'), 3)} | {v['C1']['dR@5']:+.4f} | {v['C1w']['dR@5']:+.4f} | "
            f"{v['O1']['dR@5']:+.4f} | {v['O1w']['dR@5']:+.4f} |")
    return "\n".join(lines) + "\n"


def write_run(out_dir, r):
    md = f"# D4 link words: {r['fit']} (identity {r.get('identity')})\n\n" + (table(r) if r.get("reads") else "")
    BD.write(Path(out_dir) / f"linkwords-{r['fit']}", r, md)


def write_verdict(out, v, runs):
    v = dict(v)
    v["runs"] = {r["fit"]: {"identity": r.get("identity"), "complete": r.get("complete"),
                            "models_sha256": r["models_sha256"]} for r in runs}
    md = f"# D4 link words: {v['verdict']}\n\n" + "".join(f"## {r['fit']}\n\n" + table(r) + "\n" for r in runs
                                                           if r.get("reads"))
    BD.write(out, v, md)


def verdict_cmd(a):
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.runs.split(",") if p]
    v = verdict(runs)
    write_verdict(Path(a.out), v, runs)
    log(f"linkwords verdict: {v['verdict']}")
    return 0


# ── a local code check ───────────────────────────────────────────────────────


def dev_cmd(a):
    """The plan, texts and words on the first chunks of a local s1eval look, with the look's first score column in
    place of zrc's (a code check: no read, no identity, nothing decided)."""
    import yaml
    import u1b as U1B
    import lean_cache as LC
    d = Path(LC.LM.LOOK) / a.dataset / "s1eval"
    ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
    files = sorted((d / "chunks").glob("c*.npz"), key=lambda p: int(p.stem[1:]))[:a.chunks]
    if [int(p.stem[1:]) for p in files] != list(range(len(files))):
        raise SystemExit("dev: the first chunks are not on disk")
    recs = []
    for f in files:
        z = np.load(f)
        qps, qed = z["q_pool_size"].astype(np.int64), z["q_edges"].astype(np.int64)
        no, eo = np.concatenate([[0], np.cumsum(qps)]), np.concatenate([[0], np.cumsum(qed)])
        er = z["e_rel"] if "e_rel" in z.files else None
        for i in range(qps.size):
            r0, r1, e0, e1 = no[i], no[i + 1], eo[i], eo[i + 1]
            qr, _ = keep_edges(z["score"][r0:r1, 0].astype(np.float32), z["is_gold"][r0:r1].astype(bool),
                               int(z["q_gold_total"][i]), z["pool"][r0:r1].astype(np.int32),
                               z["e_u"][e0:e1].astype(np.int64), z["e_v"][e0:e1].astype(np.int64),
                               z["e_fam"][e0:e1].astype(np.int64), None if er is None else er[e0:e1])
            recs.append(qr)
    served, _ = U1B.package_root(False)
    split = yaml.safe_load(CFG.read_text(encoding="utf-8"))["populations"]["eval_splits"][a.dataset]
    out, info = words_dataset(served, a.dataset, split, [("dev", "dev", ids[:len(recs)], recs)])
    sm = out[("dev", "dev")]
    print(json.dumps({"questions": len(recs), "need": info["need"], "seconds": round(info["seconds"], 1),
                      "ents": bool(info["ents"]),
                      "counts": {k: sm.get(k) for k in ("partial_questions", "missed_adj", "ranked", "nb_size_median",
                                                        "missed_with_window", "cand_with_window")}}, indent=1))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    assert question_stems("Who is the spouse of the Green performer?") == {"spous", "green", "perfo"}
    assert question_stems("In what US state is it, A?") == {"us", "state"}
    ta = "Green is the fourth studio album by English musician Steve Hillage. Written in spring 1977, it was released."
    tc = "Steve Hillage is an English musician. His partner is Miquette Giraudy."
    tm = "Miquette Giraudy is a French keyboard player, the partner of Steve Hillage."
    da = Doc("Green (Steve Hillage album)", ta)
    dc = Doc("Steve Hillage", tc, [(7, tc.index("Miquette"), tc.index("Miquette") + 16)])
    dm = Doc("Miquette Giraudy", tm, [(7, 0, 16), (9, tm.index("Steve"), tm.index("Steve") + 13)])
    assert dc.ments == [(7, 9, 11)] and dm.ments == [(7, 0, 2), (9, 10, 12)], (dc.ments, dm.ments)
    w = title_windows(da, dc)
    assert w == [["green", "is", "the", "fourt", "studi", "album", "by", "engli", "music",
                  "writt", "in", "sprin", "1977", "it", "was", "relea"]], w
    assert title_windows(dc, da) == []                          # the album's full title is not in the musician's text
    assert occurrences(["a", "b", "a", "b"], ["a", "b"]) == [0, 2] and occurrences(["a"], ["a", "b"]) == []
    assert window(list("abcdefghijklmnopqrstuvwxyz"), 12, 14, 3) == list("jklopq")
    nw = ner_windows(dc, dm)                                    # key 7 shared; 9 only in dm
    assert nw == [["steve", "hilla", "is", "an", "engli", "music", "his", "partn", "is"],
                  ["is", "a", "frenc", "keybo", "playe", "the", "partn", "of", "steve", "hilla"]], nw
    idf = {"music": 2.0, "green": 1.0, "partn": 3.0, "spous": 5.0, "actor": 4.0, "film": 1.5}
    q = question_stems("Who was the musician on Green?")
    assert q == {"music", "green"}
    assert link_score(da, dc, [(0, ())], q, idf) == (2.0, 0, 1)  # 'green' is a's title; 'music' around the mention
    assert link_score(da, dc, [(2, ())], q, idf) == (0.0, -1, 0)  # kNN: no words
    qp = frozenset({"partn", "spous"})
    assert link_score(dc, dm, [(1, ()), (2, ())], qp, idf) == (3.0, 1, 2)
    assert link_score(dc, dm, [(0, ()), (1, ())], qp, idf)[0] == 3.0
    kbn = [stems_of(["directed", "by"]), stems_of(["starred", "actors"])]
    dk1, dk2 = Doc(None, "Temuera Morrison"), Doc(None, "Once Were Warriors")
    assert link_score(dk1, dk2, [(0, (1, -1, -1, -1))], frozenset({"actor", "film"}), idf, kbn) == (4.0, 0, 1)
    assert link_score(dk1, dk2, [(0, (0, -1, -1, -1))], frozenset({"actor", "film"}), idf, kbn) == (0.0, -1, 1)
    # ranks: base 0,1,2,3; words put row 2 first
    rb, rw, rf = ranks(np.arange(4), np.array([4.0, 3, 2, 1]), [0, 0, 5, 0])
    assert list(rb) == [1, 2, 3, 4] and list(rw) == [2, 3, 1, 4] and list(rf) == [1, 3, 2, 4], (rb, rw, rf)
    rb, rw, rf = ranks(np.array([1, 3]), np.array([0.0, 1, 0, 2]), [0.0, 0.0])
    assert list(rb) == [2, 1] and list(rw) == [2, 1] and list(rf) == [2, 1]
    # a question: golds 0 (found, top) and 7 (missed); 0's neighbours 5, 6, 7; only the 0-7 link has words
    s = np.array([9, 8, 7, 6, 5, 3, 2, 1], np.float32)
    g = np.zeros(8, bool)
    g[[0, 7]] = True
    u = np.array([0, 0, 0, 1, 2])
    v = np.array([5, 6, 7, 3, 4])
    fam = np.array([0, 1, 1, 0, 2])
    qr, nf = keep_edges(s, g, 2, np.arange(100, 108, dtype=np.int32), u, v, fam, None)
    assert nf == 1 and list(qr.u) == [0, 0, 0] and list(qr.v) == [5, 6, 7], (qr.u, qr.v)   # anchors {0}

    def fake(x, y, links):
        return (2.0, 1, 1) if {x, y} == {100, 107} else (0.0, -1, 1 if links[0][0] != 2 else 0)
    out = words_question(qr, fake)
    # base among {5,6,7}: 5,6,7; words: 7 first; fused: 5 (1+2), 7 (3+1), 6 (2+3)
    assert out["zrc"] == 1 and out["C1"] == 1 and out["C1w"] == 2 and out["O1"] == 1 and out["O1w"] == 2, out
    assert out["nb"] == [[3, 1, 2, 2.0, 1, 1]] and out["nb_size"] == 3 and out["cand_words"] == 1, out
    sm = summarise([out])
    assert sm["le2"] == {"base": 0.0, "words": 1.0, "fused": 1.0} and sm["missed_with_words"] == 1.0, sm
    assert sm["zrc"]["R@5"] == 0.5 and sm["C1w"]["R@5"] == 1.0 and sm["C1w"]["vs_C1"] == 0.5, sm
    assert sm["missed_words_family"] == {"1": 1}, sm
    # no gold in the top five: nothing partial, the oracle keeps zrc's count
    g2 = np.zeros(8, bool)
    g2[6] = True
    qr2, nf2 = keep_edges(s, g2, 1, np.arange(8, dtype=np.int32), u, v, fam, None)
    out2 = words_question(qr2, fake)
    assert nf2 == 0 and out2["O1"] == 0 and out2["O1w"] == 0 and "partial" not in out2, out2
    # mentions: df, the graph's edges rebuilt
    n = 30
    rows = [(0, "a", 0, 3, 0), (1, "a", 4, 7, 0), (1, "b", 0, 3, 1), (2, "b", 1, 4, 1), (3, "b", 2, 5, 1), (3, "b", 9, 12, 1),
            (4, "c", 0, 3, 2)] + [(i, "z", 0, 3, 3) for i in range(5, 30)] + [(0, "z", 9, 12, 3)]
    rows.sort(key=lambda r: r[0])
    arr, keys = aggregate(rows, n)
    assert keys == ["a", "b", "c", "z"] and list(arr["df"]) == [2, 3, 1, 26], (keys, arr["df"])
    rc, rw_ = rebuild_ner(arr["key"].astype(np.int64), arr["node"].astype(np.int64), arr["df"].astype(np.int64), n)
    assert list(rc) == [0 * n + 1, 1 * n + 2, 1 * n + 3, 2 * n + 3] and np.allclose(rw_, [0.5, 1 / 3, 1 / 3, 1 / 3]), (rc, rw_)
    rp = repro(arr, n, np.array([1, 2, 1, 3, 0]), np.array([0, 1, 3, 2, 4]), np.array([0.5, 1 / 3, 1 / 3, 1 / 3, 1.0]))
    assert rp["graph_edges"] == 5 and rp["common"] == 4 and abs(rp["reproduced"] - 0.8) < 1e-12, rp
    assert rp["weight_max_abs_diff"] < 1e-12 and rp["extra_share"] == 0.0, rp
    with tempfile.TemporaryDirectory() as t:
        np.savez_compressed(Path(t) / "x_ents.npz", **arr)
        BD.write(Path(t) / "x_ents", {"nodes": n, "nodes_sha256": "abc", "arrays_sha256": arrays_sha(arr, ENT_ARRAYS)}, "")
        mentions, _ = load_ents("x", "abc", t)
        assert mentions(1) == [(0, 4, 7), (1, 0, 3)] and mentions(4) == [] and mentions(5) == [], mentions(1)
        try:
            load_ents("x", "other", t)
            raise AssertionError("load_ents took another corpus's mentions")
        except SystemExit:
            pass
        rl = Path(t) / "r_rel_vocab.json"
        rl.write_text(json.dumps({"vocab": ["directed_by", "film.film.starring"]}), encoding="utf-8")
        assert kb_names("r", t) == [["direc", "by"], ["film", "film", "starr"]]

    # the verdict
    def run(fit, b, f, rep=0.97, ident="PASS"):
        return {"fit": fit, "identity": ident, "models_sha256": "x", "ents": {DECIDE: {"graph": {"reproduced": rep}}},
                "reads": {f"{DECIDE}=s1eval": {"le2": {"base": b, "words": 0.0, "fused": f}}}}
    assert verdict([run("a", 0.25, 0.32), run("b", 0.25, 0.33)])["verdict"] == "SIGNAL"
    assert verdict([run("a", 0.25, 0.32), run("b", 0.25, 0.31)])["verdict"] == "WEAK"       # 1.25x needs 0.3125
    assert verdict([run("a", 0.25, 0.30), run("b", 0.10, 0.12)])["verdict"] == "NO_SIGNAL"
    assert verdict([run("a", 0.02, 0.07), run("b", 0.0, 0.05)])["verdict"] == "SIGNAL"      # the bounds pass
    assert verdict([run("a", 0.04, 0.0899), run("b", 0.0, 0.06)])["verdict"] == "WEAK"
    assert verdict([run("a", 0.4, 0.5), run("b", 0.4, 0.5)])["verdict"] == "SIGNAL"         # 0.5 = 1.25 x 0.4
    assert verdict([run("a", 0.25, 0.4, rep=0.85), run("b", 0.25, 0.4)])["verdict"] == "INCOMPLETE"
    assert verdict([run("a", 0.25, 0.4, ident="FAIL"), run("b", 0.25, 0.4)])["verdict"] == "INCOMPLETE"
    assert verdict([run("a", 0.25, 0.4)])["verdict"] == "INCOMPLETE"
    nr = run("b", 0.25, 0.4)
    nr["reads"][f"{DECIDE}=s1eval"]["le2"] = None
    assert verdict([run("a", 0.25, 0.4), nr])["verdict"] == "INCOMPLETE"
    print("linkwords selftest: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("ents", "run", "verdict", "dev"))
    ap.add_argument("--dataset")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--fits", default="scr-zrct,scr-zrct-hp")
    ap.add_argument("--fit-root", default="outputs/screen/fits")
    ap.add_argument("--out-dir", default="outputs/diag")
    ap.add_argument("--verdict-out", default="outputs/diag/linkwords-zrc")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--runs", default="")
    ap.add_argument("--out")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "ents":
        return ents_cmd(a)
    if a.cmd == "run":
        return run_cmd(a)
    if a.cmd == "verdict":
        return verdict_cmd(a)
    if a.cmd == "dev":
        return dev_cmd(a)
    ap.error("ents, run, verdict, dev or --selftest")


if __name__ == "__main__":
    sys.exit(main())
