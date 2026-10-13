"""P2c part 1: LLM-free triples in place of an LLM-built graph, calibrated on B1 (docs/P2C_LLM_FREE_TRIPLES.md).

    python scripts/p2c_triples.py parse musique|2wiki [--limit N]          -> outputs/p2c/parse_b1_<setting>.{npz,json}
    python scripts/p2c_triples.py parse DATASET --part k/n [--host] [--limit N]
                                                                          -> outputs/p2c/parse_<dataset>_<k>of<n>.{npz,json}
    python scripts/p2c_triples.py refs musique|2wiki [--limit N]           -> outputs/p2c/ref_<setting>_<ref>.npz
    python scripts/p2c_triples.py run musique|2wiki [--limit Q]            -> outputs/p2c/b1_<setting>.json
    python scripts/p2c_triples.py verdict                                  -> outputs/p2c/verdict.{json,md}
    python scripts/p2c_triples.py --selftest

Our construction (version 1): spaCy 3.8.15 with en_core_web_sm 3.8.0 (a CNN pipeline, no transformer), fixed phrase,
string and dependency rules. The references (HippoRAG 2's OpenIE with gpt-4o-mini and Llama-3.3-70B, SiReRAG's
propositions) are read as released from the CRAG repo (read-only) and never committed. B1c's walk functions and B1's
vector readers are imported unchanged. --limit runs write nothing and log no number.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs" / "p2c"
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
RAW = BENCH / "raw"
PKG = Path("C:/Users/Swastik/Desktop/CRAG/data/final_canonical")
REPOS = Path("C:/Users/Swastik/Desktop/CRAG/baselines/repos")
SPACY_ZIP = ROOT / "outputs" / "host_ops" / "pylib" / "spacy3815_cp313_win_amd64.zip"
SPACY_DEST = ROOT / "outputs" / "host_ops" / "pylib" / "spacy_cp313"

SETTINGS = {"musique": "musique", "2wiki": "2wikimultihopqa"}          # B1 setting -> its raw file stem
REFS = {"musique": {"H4o": ("hipporag", "HippoRAG/outputs/musique/openie_results_ner_gpt-4o-mini.json"),
                    "H70": ("hipporag", "HippoRAG/outputs/musique/openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json"),
                    "SR": ("sirerag", "SiReRAG/musique_kg.json")},
        "2wiki": {"SR": ("sirerag", "SiReRAG/2wiki_kg.json")}}
FULL_PARTS = {"musique": 2, "squad": 1, "hotpotqa": 40, "2wiki": 60}   # part 2's input (the host's parse items)

NUMERIC = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY"}
ENT_LABELS = {"PERSON", "GPE", "ORG", "LOC", "FAC", "WORK_OF_ART", "EVENT", "PRODUCT", "NORP"}   # the ner family's
PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their"}
RELATIVE = {"who", "whom", "which", "that"}
ARTICLES = ("the ", "a ", "an ")
MAX_CHUNK = 8
DF_MIN, DF_MAX = 2, 25                         # the P rule (b1_build.ner_edges)
ALPHA, STEPS = 0.5, 50                         # restart share and steps (B1c's; HippoRAG 2's damping 0.5)
PASSAGE_W = 0.05                               # HippoRAG 2's passage_node_weight
LINK_TOP = 5                                   # HippoRAG 2's linking_top_k
NGRAM = 8
KS = (2, 5, 10)
BOOT, TIE = 2000, 0.002
QBATCH = 100
WALKS = ("WP-T5", "WP-RW", "WH")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def peak_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 30, 2)
    except Exception:                                  # noqa: BLE001 (no psutil: no figure)
        return None


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def save_npz(p, **arrays):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p)[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    for i in range(6):                         # a scanner may hold the new file for a moment (WinError 32)
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if i == 5:
                raise
            time.sleep(5)


# ── normal form and packing ───────────────────────────────────────────────────


def nf(s):
    """Lower case, "'s" removed, non-word characters as spaces, collapsed, a leading article removed; '' if < 2 chars."""
    s = re.sub(r"['\u2019]s\b", "", str(s).lower())
    s = " ".join(re.sub(r"[^\w\s]", " ", s).split())
    for a in ARTICLES:
        if s.startswith(a):
            s = s[len(a):]
            break
    return s if len(s) >= 2 else ""


def title_nf(title):
    return nf(re.sub(r"\s*\([^()]*\)\s*$", "", str(title)))


def pack(strings):
    b = [s.encode("utf-8") for s in strings]
    off = np.zeros(len(b) + 1, np.int64)
    off[1:] = np.cumsum([len(x) for x in b])
    return np.frombuffer(b"".join(b), np.uint8).copy(), off


def unpack(blob, off):
    raw = np.asarray(blob, np.uint8).tobytes()
    return [raw[off[i]:off[i + 1]].decode("utf-8") for i in range(off.size - 1)]


def to_csr(lists, dtype=np.int32, width=None):
    ptr = np.zeros(len(lists) + 1, np.int64)
    ptr[1:] = np.cumsum([len(x) for x in lists])
    if width is None:
        flat = np.asarray([v for x in lists for v in x], dtype)
    else:
        flat = np.asarray([v for x in lists for v in x], dtype).reshape(-1, width)
    return ptr, flat


def from_csr(ptr, flat):
    return [flat[ptr[i]:ptr[i + 1]] for i in range(ptr.size - 1)]


class Vocab:
    def __init__(self):
        self.ids, self.words = {}, []

    def __call__(self, s):
        i = self.ids.get(s)
        if i is None:
            i = self.ids[s] = len(self.words)
            self.words.append(s)
        return i


# ── our construction (version 1) ──────────────────────────────────────────────


def conj(t):
    out, stack = [], [t]
    while stack:
        x = stack.pop()
        out.append(x)
        stack.extend(c for c in x.children if c.dep_ == "conj")
    return out


def extract(doc, title):
    """One passage: (phrases, triples) in normal form, by the rules of section 3."""
    tnf = title_nf(title)
    tok, span_text, persons, phrases = {}, {}, set(), set()
    for e in doc.ents:
        if e.label_ in NUMERIC:
            continue
        p = nf(e.text)
        if not p:
            continue
        phrases.add(p)
        span_text.setdefault(p, e.text)
        if e.label_ == "PERSON":
            persons.add(p)
        for t in e:
            tok[t.i] = p
    for ch in doc.noun_chunks:
        if ch.root.pos_ not in ("NOUN", "PROPN"):
            continue
        toks = list(ch)
        while toks and (toks[0].dep_ == "det" or (toks[0].dep_ == "poss" and toks[0].pos_ == "PRON")):
            toks = toks[1:]
        if not toks or len(toks) > MAX_CHUNK:
            continue
        sp = doc[toks[0].i: toks[-1].i + 1]
        p = nf(sp.text)
        if not p:
            continue
        phrases.add(p)
        span_text.setdefault(p, sp.text)
        for t in toks:
            tok.setdefault(t.i, p)                 # entities before chunks
    if tnf:
        phrases.add(tnf)
    # string rules within the passage
    canon = {}
    longp = [p for p in persons if len(p.split()) >= 2]
    for p in phrases:
        if " " not in p:
            c = [q for q in longp if q.split()[-1] == p]
            if len(c) == 1:
                canon[p] = c[0]
    for t in doc:
        if (len(t.text) >= 2 and t.text.isalpha() and t.text.isupper() and t.i >= 2 and t.i + 1 < len(doc)
                and doc[t.i - 1].text == "(" and doc[t.i + 1].text == ")"):
            p = tok.get(t.i - 2)
            if p and p in span_text:
                ws = span_text[p].split()
                if ws and ws[0].lower() in ("the", "a", "an"):
                    ws = ws[1:]
                init = "".join(w[0] for w in ws if w[:1].isupper())
                a = nf(t.text)
                if a and init == t.text and a != p:
                    canon[a] = canon.get(p, p)
    phrases = {canon.get(p, p) for p in phrases}

    def arg(t):
        if t.pos_ == "PRON" and t.lower_ in PRONOUNS:
            return tnf or None
        p = tok.get(t.i)
        return canon.get(p, p) if p else None

    trip = set()

    def add(s, r, o):
        if s and o and s != o and r:
            trip.add((s, r, o))

    for t in doc:
        if t.pos_ in ("VERB", "AUX"):
            subs = [c for c in t.children if c.dep_ in ("nsubj", "nsubjpass")]
            if t.dep_ in ("relcl", "acl") and t.head.pos_ in ("NOUN", "PROPN") and (
                    not subs or all(s.lower_ in RELATIVE for s in subs)):
                subs = [t.head]
            if subs:
                subs = [x for s in subs for x in conj(s)]
                rel = " ".join([t.lemma_.lower()] + [c.lower_ for c in t.children if c.dep_ == "prt"])
                objs = [x for c in t.children if c.dep_ in ("dobj", "attr", "oprd", "dative") for x in conj(c)]
                preps = [(c.lower_, x) for c in t.children if c.dep_ in ("prep", "agent")
                         for p in c.children if p.dep_ == "pobj" for x in conj(p)]
                for s in subs:
                    sa = arg(s)
                    for o in objs:
                        add(sa, rel, arg(o))
                    for pw, o in preps:
                        add(sa, f"{rel} {pw}", arg(o))
        if t.dep_ == "appos":
            add(arg(t.head), "be", arg(t))
        if t.pos_ in ("NOUN", "PROPN"):
            for c in t.children:
                if c.dep_ == "prep":
                    for p in c.children:
                        if p.dep_ == "pobj":
                            for x in conj(p):
                                add(arg(t), c.lower_, arg(x))
        if t.dep_ == "poss" and t.pos_ != "PRON":
            add(arg(t), "poss", arg(t.head))
    return phrases, trip


def load_spacy(host=False, disable=()):
    if host:
        sha = sha_file(SPACY_ZIP)
        mark = SPACY_DEST / "unpacked_from.sha256"
        if not (mark.exists() and mark.read_text().strip() == sha):
            raise SystemExit(f"spaCy is not unpacked from this zip on the host: run `unpack-spacy` first")
        sys.path.append(str(SPACY_DEST))
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=list(disable))
    return nlp, spacy.__version__, nlp.meta["version"]


def unpack_spacy():
    """The host has no spaCy: the laptop's installed files (the pushed zip, never committed) are unpacked once, by one
    item that the parse items wait for, beside the numba bundle. load_spacy puts them at the END of sys.path, so the rx
    env's own packages win where both exist."""
    sha = sha_file(SPACY_ZIP)
    mark = SPACY_DEST / "unpacked_from.sha256"
    if mark.exists() and mark.read_text().strip() == sha:
        log(f"spaCy already unpacked from {sha[:12]}")
        return 0
    if SPACY_DEST.exists():
        raise SystemExit(f"{SPACY_DEST.name} holds another unpack: it must be moved aside first")
    tmp = SPACY_DEST.with_name(SPACY_DEST.name + ".tmp")
    if tmp.exists():
        raise SystemExit(f"{tmp.name} is left from an earlier unpack: it must be moved aside first")
    with zipfile.ZipFile(SPACY_ZIP) as z:
        z.extractall(tmp)
    (tmp / "unpacked_from.sha256").write_text(sha)
    os.replace(tmp, SPACY_DEST)
    sys.path.append(str(SPACY_DEST))
    import spacy
    nlp = spacy.load("en_core_web_sm")
    ph, tr = extract(nlp("Lionel Messi joined FC Barcelona in 2004."), "Lionel Messi")
    assert ("lionel messi", "join", "barcelona") in tr, tr
    log(f"spaCy {spacy.__version__} with {nlp.meta['version']} unpacked from {sha[:12]} and parsed a sentence")
    return 0


def ner_keys_of(texts):
    """Today's ner family's keys per passage, exactly as b1_build.ner_edges makes them (parser and lemmatizer off)."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    out = []
    for doc in nlp.pipe([str(t) for t in texts], batch_size=256):
        out.append({e.text.lower().strip() for e in doc.ents if e.label_ in ENT_LABELS and len(e.text) > 2})
    return out


def parse_texts(nlp, titles, texts, every=None):
    """Each passage's phrase ids and (subject, relation, object) ids, mapped as they come (one vocabulary for phrases,
    one for relations); returns the packed arrays and the number of words."""
    V, R = Vocab(), Vocab()
    ids, trips, words = [], [], 0
    t0 = time.time()
    for i, doc in enumerate(nlp.pipe((str(t) for t in texts), batch_size=256)):
        p, tr = extract(doc, titles[i])
        ids.append(np.asarray(sorted({V(x) for x in p}), np.int32))
        trips.append(np.asarray(sorted((V(a), R(r), V(b)) for a, r, b in tr), np.int32).reshape(-1, 3))
        words += len(doc)
        if every and (i + 1) % every == 0:
            log(f"  {i + 1} passages ({time.time() - t0:.0f}s)")
    return pack_ids(V, R, ids, trips), words


def pack_ids(V, R, ids, trips):
    ptr = np.zeros(len(ids) + 1, np.int64)
    ptr[1:] = np.cumsum([x.size for x in ids])
    tptr = np.zeros(len(trips) + 1, np.int64)
    tptr[1:] = np.cumsum([x.shape[0] for x in trips])
    pid = np.concatenate(ids) if ids else np.zeros(0, np.int32)
    tri = np.concatenate(trips) if trips else np.zeros((0, 3), np.int32)
    vb, vo = pack(V.words)
    rb, ro = pack(R.words)
    return dict(vocab_blob=vb, vocab_off=vo, ptr=ptr, pid=pid.astype(np.int32), rel_blob=rb, rel_off=ro, tptr=tptr,
                trip=tri.astype(np.int32).reshape(-1, 3))


def pack_parse(phr, trip):
    """From phrase sets and triple sets (the self-test's)."""
    V, R = Vocab(), Vocab()
    ids = [np.asarray(sorted({V(x) for x in ps}), np.int32) for ps in phr]
    tr = [np.asarray(sorted((V(a), R(r), V(b)) for a, r, b in ts), np.int32).reshape(-1, 3) for ts in trip]
    return pack_ids(V, R, ids, tr)


def b1_corpus(setting):
    stem = SETTINGS[setting]
    corpus = json.loads((RAW / f"{stem}_corpus.json").read_text(encoding="utf-8"))
    return corpus


def parse_b1(setting, limit=None, write=True):
    t0 = time.time()
    corpus = b1_corpus(setting)
    if limit:
        corpus = corpus[:limit]
    nlp, sv, mv = load_spacy()
    arrays, words = parse_texts(nlp, [c["title"] for c in corpus], [c["text"] for c in corpus], every=2000)
    keys = ner_keys_of([c["text"] for c in corpus])
    K = Vocab()
    nptr, nid = to_csr([sorted({K(k) for k in ks}) for ks in keys])
    kb, ko = pack(K.words)
    arrays |= dict(ner_blob=kb, ner_off=ko, nptr=nptr, nid=nid)
    rec = {"declared_in": "docs/P2C_LLM_FREE_TRIPLES.md", "script_sha256": script_sha256(), "setting": setting,
           "spacy": sv, "model": mv, "passages": len(corpus), "words": int(words),
           "phrases": int(arrays["vocab_off"].size - 1), "mentions": int(arrays["pid"].size),
           "triples": int(arrays["trip"].shape[0]), "relations": int(arrays["rel_off"].size - 1),
           "ner_keys": int(ko.size - 1), "corpus_sha256": sha_file(RAW / f"{SETTINGS[setting]}_corpus.json"),
           "limit": limit, "seconds": round(time.time() - t0, 1), "utc": utc()}
    if limit or not write:
        log(f"parse {setting}" + (f" --limit {limit}" if limit else "") + f": every step ran ({rec['seconds']}s); "
            "nothing written")
        return arrays, rec
    save_npz(OUT / f"parse_b1_{setting}.npz", **arrays)
    write_json(OUT / f"parse_b1_{setting}.json", rec)
    log(f"parse {setting}: {rec['passages']} passages, {rec['words']} words ({rec['seconds']}s)")
    return arrays, rec


def corpus_root(host):
    if not host:
        return PKG
    import mp_approx_six_base as SB
    import mp_approx_six_base_score as S6
    return Path(S6.host_mode(SB.load_declaration(), log)["mirror_root"]) / "data" / "final_canonical"


def parse_part(ds, k, n, host, limit=None):
    """One part of a full corpus (part 2's input): lines [k*N//n, (k+1)*N//n) of the served nodes.jsonl."""
    t0 = time.time()
    path = corpus_root(host) / ds / "nodes.jsonl"
    with open(path, "rb") as f:
        N = sum(b.count(b"\n") for b in iter(lambda: f.read(1 << 24), b""))
    lo, hi = k * N // n, (k + 1) * N // n
    titles, texts = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < lo:
                continue
            if i >= hi or (limit and len(texts) >= limit):
                break
            d = json.loads(line)
            titles.append(d.get("title", ""))
            texts.append(d.get("text", ""))
    nlp, sv, mv = load_spacy(host)
    arrays, words = parse_texts(nlp, titles, texts, every=20000)
    rec = {"declared_in": "docs/P2C_LLM_FREE_TRIPLES.md", "script_sha256": script_sha256(), "dataset": ds,
           "part": [k, n], "lines": [lo, lo + len(texts)], "corpus_lines": N, "where": "host" if host else "laptop",
           "nodes_jsonl_bytes": os.path.getsize(path), "spacy": sv, "model": mv, "passages": len(texts),
           "words": int(words), "phrases": int(arrays["vocab_off"].size - 1), "mentions": int(arrays["pid"].size),
           "triples": int(arrays["trip"].shape[0]), "limit": limit, "seconds": round(time.time() - t0, 1),
           "peak_gb": peak_gb(), "utc": utc()}
    if limit:
        log(f"parse {ds} part {k}/{n} --limit {limit}: every step ran ({rec['seconds']}s, {len(texts)} passages, "
            f"peak {rec['peak_gb']} GB); nothing written")
        return rec
    stem = OUT / f"parse_{ds}_{k}of{n}"
    save_npz(stem.with_suffix(".npz"), **arrays)
    write_json(stem.with_suffix(".json"), rec)
    log(f"parse {ds} part {k}/{n}: {rec['passages']} passages, {rec['words']} words ({rec['seconds']}s)")
    return rec


# ── the references ─────────────────────────────────────────────────────────────


def as_list(x):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return None
    return x if isinstance(x, list) else None


def read_hipporag(docs):
    """Per passage: (title, text, phrases = triple ends, entity list, relation pairs one per triple); skipped counts."""
    rows, skipped, bad, kept, loops = [], 0, 0, 0, 0
    for d in docs:
        tr = as_list(d.get("extracted_triples"))
        en = as_list(d.get("extracted_entities"))
        if tr is None or en is None:
            bad += 1
            tr, en = tr or [], en or []
        ph, pairs = set(), []
        for t in tr:
            if not isinstance(t, (list, tuple)) or len(t) != 3:
                skipped += 1
                continue
            s, o = nf(t[0]), nf(t[2])
            if not s or not o or not str(t[1]).strip():
                skipped += 1
                continue
            ph.update((s, o))
            kept += 1
            if s != o:
                pairs.append((s, o))
            else:
                loops += 1                         # one phrase at both ends: the phrase, no edge
        ents = {nf(e) for e in en if nf(e)}
        rows.append((d.get("title", ""), d.get("text", ""), ph, ents, pairs))
    return rows, {"triples": kept, "triples_skipped": skipped, "triples_one_phrase": loops, "unreadable_passages": bad}


def read_sirerag(items):
    rows, props = [], 0
    for d in items:
        title, _, text = str(d.get("text", "")).partition("\n")
        ph, pairs = set(), []
        for f in d.get("facts", []) or []:
            props += 1
            es = sorted({nf(e) for e in (f.get("entities") or []) if nf(e)})
            ph.update(es)
            pairs += [(es[a], es[b]) for a in range(len(es)) for b in range(a + 1, len(es))]
        rows.append((title, text, ph, set(ph), pairs))
    return rows, {"propositions": props}


def aligned(rows, corpus):
    bad = [i for i, (t, x, *_r) in enumerate(rows)
           if nf(t) != nf(corpus[i]["title"]) or nf(x) != nf(corpus[i]["text"])]
    return len(rows) == len(corpus) and not bad, bad[:5]


def refs_stage(setting, limit=None, write=True):
    corpus = b1_corpus(setting)
    out = {}
    for name, (kind, rel) in REFS[setting].items():
        t0 = time.time()
        src = REPOS / rel
        raw = json.loads(src.read_text(encoding="utf-8"))
        items = raw["docs"] if kind == "hipporag" else raw
        n_items = len(items)
        if limit:
            items = items[:limit]
        rows, extra = (read_hipporag if kind == "hipporag" else read_sirerag)(items)
        ok, bad = aligned(rows, corpus[:len(rows)] if limit else corpus)
        if not ok:
            raise SystemExit(f"{setting} {name}: the reference does not match B1's passages (first: {bad})")
        V = Vocab()
        phr = [sorted({V(p) for p in r[2]}) for r in rows]
        ent = [sorted({V(p) for p in r[3]}) for r in rows]
        prs = [[(V(a), V(b)) for a, b in r[4]] for r in rows]
        ptr, pid = to_csr(phr)
        eptr, eid = to_csr(ent)
        rptr, rpair = to_csr(prs, width=2)
        vb, vo = pack(V.words)
        meta = {"script_sha256": script_sha256(), "reference": name, "kind": kind, "file": rel, "bytes": src.stat().st_size, "sha256": sha_file(src),
                "items": n_items, "passages": len(rows), "phrases": len(V.words), "pairs": int(rpair.shape[0]),
                "mentions": int(pid.size), **extra, "seconds": round(time.time() - t0, 1)}
        out[name] = (dict(vocab_blob=vb, vocab_off=vo, ptr=ptr, pid=pid, eptr=eptr, eid=eid, rptr=rptr, rpair=rpair),
                     meta)
        if write and not limit:
            save_npz(OUT / f"ref_{setting}_{name}.npz", **out[name][0], meta=np.asarray(json.dumps(meta)))
        log(f"refs {setting} {name}: read and matched ({meta['seconds']}s)"
            + ("" if write and not limit else "; nothing written"))
    return out


# ── graphs: the P and H forms ─────────────────────────────────────────────────


def p_edges(sets):
    """The P rule (b1_build.ner_edges): passages sharing a key in DF_MIN..DF_MAX passages, weight sum 1/df; a < b."""
    docs_of = defaultdict(set)
    for i, ks in enumerate(sets):
        for k in ks:
            docs_of[k].add(i)
    w = defaultdict(float)
    for _k, docs in docs_of.items():
        df = len(docs)
        if DF_MIN <= df <= DF_MAX:
            docs = sorted(docs)
            for a in range(len(docs)):
                for b in range(a + 1, len(docs)):
                    w[(docs[a], docs[b])] += 1.0 / df
    keys = sorted(w)
    return (np.asarray([k[0] for k in keys], np.int64), np.asarray([k[1] for k in keys], np.int64),
            np.asarray([w[k] for k in keys], np.float64))


def sym(n, s, t, w):
    import scipy.sparse as sp
    a = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([s, t]), np.concatenate([t, s]))), shape=(n, n)).tocsr()
    a.sum_duplicates()
    return a


def h_form(n, phr_ids, pairs, V, with_rel=True):
    """Passages 0..n-1, phrases n..n+V-1, as HippoRAG 2's undirected graph holds them (add_passage_edges, add_fact_edges,
    add_new_edges): a mention is one edge of weight 1; a triple adds 1 to its (s, o) entry and 1 to its (o, s) entry, and
    each entry is an edge, so two phrases are joined with twice their number of triples. Self-loops are not edges."""
    s = np.concatenate([np.full(len(x), i, np.int64) for i, x in enumerate(phr_ids)] or [np.zeros(0, np.int64)])
    t = n + np.concatenate([np.asarray(x, np.int64) for x in phr_ids] or [np.zeros(0, np.int64)])
    w = np.ones(s.size)
    if with_rel and len(pairs):
        p = np.asarray(pairs, np.int64).reshape(-1, 2)
        s = np.concatenate([s, n + p[:, 0]])
        t = np.concatenate([t, n + p[:, 1]])
        w = np.concatenate([w, np.full(p.shape[0], 2.0)])
    return sym(n + V, s, t, w)


class Graph:
    """One graph X on a setting: its phrases per passage (strings and ids), its relation pairs and its vocabulary."""

    def __init__(self, name, words, phr_ids, pairs_per_passage, ents=None):
        self.name, self.words = name, list(words)
        self.phr_ids = [np.asarray(x, np.int64) for x in phr_ids]
        self.pairs_pp = [np.asarray(x, np.int64).reshape(-1, 2) for x in pairs_per_passage]
        self.ents = None if ents is None else [np.asarray(x, np.int64) for x in ents]
        self.id_of = {w: i for i, w in enumerate(self.words)}
        df = np.zeros(len(self.words), np.int64)
        for x in self.phr_ids:
            df[x] += 1
        self.df = df

    def phrase_sets(self):
        return [{self.words[i] for i in x} for x in self.phr_ids]

    def pair_sets(self):
        return [{tuple(sorted((self.words[a], self.words[b]))) for a, b in x} for x in self.pairs_pp]

    def all_pairs(self):
        return np.concatenate(self.pairs_pp) if self.pairs_pp else np.zeros((0, 2), np.int64)


def graph_from_parse(z, which="ours"):
    if which == "ner":
        words = [nf(k) for k in unpack(z["ner_blob"], z["ner_off"])]
        V = Vocab()
        ids = [sorted({V(words[i]) for i in x if words[i]}) for x in from_csr(z["nptr"], z["nid"])]
        return Graph("NER", V.words, ids, [[] for _ in ids])
    words = unpack(z["vocab_blob"], z["vocab_off"])
    ids = from_csr(z["ptr"], z["pid"])
    tr = from_csr(z["tptr"], z["trip"])
    return Graph("ours", words, ids, [x[:, [0, 2]] for x in tr])


def graph_from_ref(name, z):
    return Graph(name, unpack(z["vocab_blob"], z["vocab_off"]), from_csr(z["ptr"], z["pid"]),
                 from_csr(z["rptr"], z["rpair"]), from_csr(z["eptr"], z["eid"]))


# ── agreement ─────────────────────────────────────────────────────────────────


def micro(ref_sets, our_sets, ref_extra=None):
    """(recall, precision): the reference's items found among ours; ours found among the reference's (and its extra)."""
    hit_r = sum(len(r & o) for r, o in zip(ref_sets, our_sets))
    n_r = sum(len(r) for r in ref_sets)
    if ref_extra is None:
        ref_extra = ref_sets
    hit_p = sum(len(o & (r | e)) for r, o, e in zip(ref_sets, our_sets, ref_extra))
    n_o = sum(len(o) for o in our_sets)
    return (round(hit_r / n_r, 4) if n_r else None, round(hit_p / n_o, 4) if n_o else None)


def agreement(ref, g):
    rp, gp = ref.phrase_sets(), g.phrase_sets()
    ents = [{ref.words[i] for i in x} for x in ref.ents] if ref.ents is not None else None
    phr = micro(rp, gp, ents)
    pr = micro(ref.pair_sets(), g.pair_sets()) if any(len(x) for x in g.pairs_pp) else (0.0, None)
    es = set(zip(*p_edges(rp)[:2]))
    eo = set(zip(*p_edges(gp)[:2]))
    edge = (round(len(es & eo) / len(es), 4) if es else None, round(len(es & eo) / len(eo), 4) if eo else None)
    return {"phrases": {"recall": phr[0], "precision": phr[1]}, "pairs": {"recall": pr[0], "precision": pr[1]},
            "P_edges": {"recall": edge[0], "precision": edge[1], "reference": len(es), "ours": len(eo)}}


# ── walks ─────────────────────────────────────────────────────────────────────


def wp_lists(BC, fams, rrf, n, extra=None):
    a = sum(fams.values())
    if extra is not None:
        a = a + extra
    T = BC.transition_T(a)
    out = {}
    for rk in ("T5", "RW"):
        r = BC.restart_matrix(rk, rrf, n)
        assert np.allclose(r.sum(0), 1.0), "a WP reset does not sum to 1"
        out[f"WP-{rk}"] = BC.rank_by(BC.ppr(T, r), rrf, n)
    return out


def match_phrases(question, g):
    """X's phrases that are word n-grams (n <= 8) of the question's normal form: the five largest 1/df, ties to the
    longer phrase, then the earlier one."""
    w = nf(question).split()
    seen = {}
    for i in range(len(w)):
        for k in range(1, NGRAM + 1):
            if i + k > len(w):
                break
            j = g.id_of.get(" ".join(w[i:i + k]))
            if j is not None and g.df[j] > 0 and j not in seen:
                seen[j] = (1.0 / g.df[j], k, i)
    best = sorted(seen.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[1][2]))[:LINK_TOP]
    return [(j, v[0]) for j, v in best]


def wh_lists(BC, g, questions, dense_scores, dense_list, rrf, n, with_rel=True):
    """HippoRAG 2's search on X's H form with the string link (section 5); no match -> the dense list."""
    V = len(g.words)
    T = BC.transition_T(h_form(n, g.phr_ids, g.all_pairs(), V, with_rel))
    q = len(questions)
    lists = np.array(dense_list[:q], dtype=np.int64, copy=True)
    matched = np.zeros(q, bool)
    lo, hi = dense_scores.min(1, keepdims=True), dense_scores.max(1, keepdims=True)
    pw = PASSAGE_W * (dense_scores - lo) / np.where(hi > lo, hi - lo, 1.0)
    links = [match_phrases(x, g) for x in questions]
    for b0 in range(0, q, QBATCH):
        js = [j for j in range(b0, min(q, b0 + QBATCH)) if links[j]]
        if not js:
            continue
        R = np.zeros((n + V, len(js)))
        for c, j in enumerate(js):
            R[:n, c] = pw[j]
            for pid, wt in links[j]:
                R[n + pid, c] += wt
        R /= R.sum(0, keepdims=True)
        assert np.allclose(R.sum(0), 1.0), "a WH reset does not sum to 1"
        x = BC.ppr(T, R)[:n]
        lists[js] = BC.rank_by(x, rrf[js], n)
        matched[js] = True
    return lists, matched


# ── the share ─────────────────────────────────────────────────────────────────


def boot_idx(q, seed=0, n=BOOT):
    return np.random.default_rng(seed).integers(0, q, (n, q))


def mean_ci(d, idx):
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def share_call(d_ref, d_ours, idx):
    """d_ref, d_ours: per-question R@5 gains over the walk's base. Returns the share, its interval and the call."""
    g_ref, lo_r, hi_r = mean_ci(d_ref, idx)
    g_our, lo_o, hi_o = mean_ci(d_ours, idx)
    diff, lo_d, hi_d = mean_ci(d_ours - d_ref, idx)
    out = {"gain_ref": [round(g_ref, 4), round(lo_r, 4), round(hi_r, 4)],
           "gain_ours": [round(g_our, 4), round(lo_o, 4), round(hi_o, 4)],
           "ours_minus_ref": [round(diff, 4), round(lo_d, 4), round(hi_d, 4)]}
    if not (g_ref > TIE and lo_r > 0):
        return out | {"share": None, "share_ci": None, "call": "NO_REFERENCE_GAIN"}
    num, den = d_ours[idx].mean(1), d_ref[idx].mean(1)
    ok = den > 0
    r = num[ok] / den[ok]
    out |= {"share": round(g_our / g_ref, 4), "share_ci": [round(float(np.quantile(r, 0.025)), 4),
                                                           round(float(np.quantile(r, 0.975)), 4)],
            "share_resamples": int(ok.sum())}
    if not (diff < -TIE and hi_d < 0):
        call = "MATCHED"
    elif g_our > TIE and lo_o > 0:
        call = "PARTIAL"
    else:
        call = "NONE"
    return out | {"call": call}


# ── run ───────────────────────────────────────────────────────────────────────


def load_b1(setting):
    d = BENCH / setting
    build = json.loads((d / "build.json").read_text(encoding="utf-8"))
    fs = np.load(d / "first_stage.npz")
    qrec = json.loads((d / "queries.json").read_text(encoding="utf-8"))
    nodes = json.loads((d / "nodes.json").read_text(encoding="utf-8"))
    qraw = json.loads((RAW / f"{SETTINGS[setting]}.json").read_text(encoding="utf-8"))
    if [str(q.get("id") or q.get("_id")) for q in qraw] != [str(q["id"]) for q in qrec]:
        raise SystemExit(f"{setting}: the raw questions are not queries.json's, in order")
    return build, fs, qrec, nodes, [q["question"] for q in qraw]


def run_stage(setting, limit=None):
    import b1_build as B
    import b1c_walks as BC
    t0 = time.time()
    build, fs, qrec, nodes, questions = load_b1(setting)
    n = int(build["passages"])
    rrf = fs["rrf"].astype(np.int64)
    golds = [np.asarray(q["golds"], np.int64) for q in qrec]
    checks = {}
    # the dense scores are B1's
    xd = B.dense_rows(setting, "docs", np.asarray([x["package_row"] for x in nodes], np.int64))
    qd = B.dense_rows(setting, "queries", fs["query_rows"].astype(np.int64))
    S = qd @ xd.T
    if not np.array_equal(B.top_lists(S, B.TOP), fs["dense"]):
        raise SystemExit(f"{setting}: the package's vectors do not give B1's dense list")
    checks["dense_list_equal"] = True
    checks["vector_norm_range"] = [round(float(np.linalg.norm(xd, axis=1).min()), 4),
                                   round(float(np.linalg.norm(xd, axis=1).max()), 4)]
    # inputs: our parse and the references
    if limit:
        arrays, prec = parse_b1(setting, write=False)
        z = arrays
        refs = {k: v[0] for k, v in refs_stage(setting, write=False).items()}
        rmeta = {}
    else:
        z = dict(np.load(OUT / f"parse_b1_{setting}.npz"))
        prec = json.loads((OUT / f"parse_b1_{setting}.json").read_text(encoding="utf-8"))
        refs, rmeta = {}, {}
        for name in REFS[setting]:
            zz = np.load(OUT / f"ref_{setting}_{name}.npz")
            refs[name] = {k: zz[k] for k in zz.files if k != "meta"}
            rmeta[name] = json.loads(str(zz["meta"]))
        if prec["script_sha256"] != script_sha256() or prec.get("limit"):
            raise SystemExit(f"{setting}: the parse ran on another script or with --limit")
        if any(m.get("script_sha256") != script_sha256() for m in rmeta.values()):
            raise SystemExit(f"{setting}: a reference was read by another script")
    ours = graph_from_parse(z)
    ner = graph_from_parse(z, "ner")
    R = {name: graph_from_ref(name, refs[name]) for name in REFS[setting]}
    for g in [ours, ner, *R.values()]:
        if len(g.phr_ids) != n:
            raise SystemExit(f"{setting}: {g.name} covers {len(g.phr_ids)} passages, not {n}")
    # the P rule is the ner family's
    kw = unpack(z["ner_blob"], z["ner_off"])
    raw_keys = [{kw[i] for i in x} for x in from_csr(z["nptr"], z["nid"])]
    s_, t_, w_ = p_edges(raw_keys)
    gz = np.load(BENCH / setting / "graph_ner.npz")
    if not (np.array_equal(s_, gz["src"].astype(np.int64)) and np.array_equal(t_, gz["dst"].astype(np.int64))
            and np.abs(w_ - gz["weight"].astype(np.float64)).max() <= 1e-6):
        raise SystemExit(f"{setting}: NER's keys through the P rule do not give graph_ner.npz")
    checks["P_rule_equals_graph_ner"] = True
    log(f"{setting}: dense list and the P rule checked ({time.time() - t0:.0f}s)")
    # agreement
    agree = {}
    for name, ref in R.items():
        noREL = Graph("ours-noREL", ours.words, ours.phr_ids, [[] for _ in ours.phr_ids])
        agree[name] = {"ours": agreement(ref, ours), "NER": agreement(ref, ner), "ours-noREL": agreement(ref, noREL)}
    log(f"{setting}: agreement done ({time.time() - t0:.0f}s)")
    # WP
    fams = {f: BC.family_matrix(BENCH / setting, f, n) for f in ("structural", "ner", "knn")}
    lists = {"today": wp_lists(BC, fams, rrf, n)}
    b1c = json.loads((BENCH / "b1c.json").read_text(encoding="utf-8"))["settings"][setting]["arms"]
    for rk in ("T5", "RW"):
        got = round(float(BC.recall(lists["today"][f"WP-{rk}"], golds, 5).mean()), 4)
        if got != b1c[f"{rk}/all"]["R@5"]["all"]:
            raise SystemExit(f"{setting}: WP {rk} over today's families gives R@5 {got}, B1c filed "
                             f"{b1c[f'{rk}/all']['R@5']['all']}")
    checks["WP_today_equals_B1c"] = True
    for g in [ours, *R.values()]:
        s_, t_, w_ = p_edges(g.phrase_sets())
        lists[g.name] = wp_lists(BC, fams, rrf, n, sym(n, s_, t_, w_))
    for name, d in lists.items():
        for w, L in d.items():
            if any(np.unique(L[j, :10]).size < 10 for j in range(L.shape[0])):
                raise SystemExit(f"{setting}: a {w} ranking of {name} holds fewer than 10 passages")
    log(f"{setting}: WP walks done ({time.time() - t0:.0f}s)")
    # WH
    q = limit or len(questions)
    wh = {}
    for g in [ner, ours, *R.values()]:
        for rel in (True, False):
            if g.name == "NER" and not rel:
                continue
            key = g.name if rel else f"{g.name}-noREL"
            L, m = wh_lists(BC, g, questions[:q], S[:q], fs["dense"].astype(np.int64), rrf[:q], n, rel)
            if any(np.unique(L[j, :10]).size < 10 for j in range(q)):
                raise SystemExit(f"{setting}: a WH ranking of {key} holds fewer than 10 passages")
            wh[key] = (L, m)
    log(f"{setting}: WH walks done ({time.time() - t0:.0f}s)")
    checks["resets_sum_to_one"] = True
    if limit:
        log(f"run {setting} --limit {limit}: every step ran and the checks held ({time.time() - t0:.0f}s); "
            "nothing written")
        return None
    # measures, gains and shares
    gq = golds[:q]
    rec_q = lambda L: {k: BC.recall(L, gq, k) for k in KS}                           # noqa: E731
    walks = {}
    for name, d in lists.items():
        for w, L in d.items():
            walks.setdefault(w, {})["today" if name == "today" else f"today+{name}"] = rec_q(L)
    walks["WH"] = {k: rec_q(L) for k, (L, _m) in wh.items()}
    walks["WH"]["dense"] = rec_q(fs["dense"].astype(np.int64))
    idx = boot_idx(q)
    base = {"WP-T5": "today", "WP-RW": "today", "WH": "NER"}
    key = {"WP-T5": lambda x: f"today+{x}", "WP-RW": lambda x: f"today+{x}", "WH": lambda x: x}
    shares = {}
    for w in WALKS:
        b = walks[w][base[w]][5]
        d_ours = walks[w][key[w]("ours")][5] - b
        shares[w] = {name: share_call(walks[w][key[w](name)][5] - b, d_ours, idx) for name in R}
    rec = {"declared_in": "docs/P2C_LLM_FREE_TRIPLES.md", "script_sha256": script_sha256(), "setting": setting,
           "questions": q, "passages": n, "checks": checks, "parse": prec, "references": rmeta,
           "agreement": agree,
           "walks": {w: {k: {f"R@{kk}": round(float(v[kk].mean()), 4) for kk in KS} for k, v in d.items()}
                     for w, d in walks.items()},
           "wh_matched": {k: round(float(m.mean()), 4) for k, (_L, m) in wh.items()},
           "shares": shares, "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(OUT / f"b1_{setting}.json", rec)
    np.savez_compressed(OUT / f"b1_{setting}_r5.npz", **{f"{w}|{k}": v[5].astype(np.float32)
                                                          for w, d in walks.items() for k, v in d.items()})
    log(f"{setting}: written ({rec['seconds']}s)")
    return rec


# ── verdict ───────────────────────────────────────────────────────────────────


def verdict_stage():
    recs = {}
    for s in SETTINGS:
        p = OUT / f"b1_{s}.json"
        if not p.exists():
            raise SystemExit(f"{p.name} is missing: run `run {s}` first")
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["script_sha256"] != script_sha256():
            raise SystemExit(f"{p.name} ran on another script")
        recs[s] = r
    calls = {s: {w: {k: v["call"] for k, v in r["shares"][w].items()} for w in WALKS} for s, r in recs.items()}
    out = {"declared_in": "docs/P2C_LLM_FREE_TRIPLES.md", "script_sha256": script_sha256(), "calls": calls,
           "settings": {s: {"shares": r["shares"], "agreement": r["agreement"], "walks": r["walks"],
                            "wh_matched": r["wh_matched"]} for s, r in recs.items()}, "utc": utc()}
    write_json(OUT / "verdict.json", out)
    md = report_md(recs)
    tmp = OUT / "verdict.md.tmp"
    tmp.write_text(md, encoding="utf-8")
    os.replace(tmp, OUT / "verdict.md")
    print(md)
    return 0


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def report_md(recs):
    L = ["# P2c part 1: LLM-free triples against LLM-built graphs on B1", ""]
    for s, r in recs.items():
        L += [f"## {s} ({r['questions']} questions, {r['passages']} passages)", "",
              "Agreement (recall / precision, %):", "",
              "| reference | graph | phrases | pairs | P edges |", "| --- | --- | --- | --- | --- |"]
        for ref, d in r["agreement"].items():
            for g, a in d.items():
                L.append(f"| {ref} | {g} | {pct(a['phrases']['recall'])} / {pct(a['phrases']['precision'])} | "
                         f"{pct(a['pairs']['recall'])} / {pct(a['pairs']['precision'])} | "
                         f"{pct(a['P_edges']['recall'])} / {pct(a['P_edges']['precision'])} |")
        L += ["", "R@5 (%) in each walk:", "", "| walk | " + " | ".join(r["walks"]["WH"]) + " |",
              "| --- | " + " | ".join("---" for _ in r["walks"]["WH"]) + " |"]
        L.append("| WH | " + " | ".join(pct(v["R@5"]) for v in r["walks"]["WH"].values()) + " |")
        for w in ("WP-T5", "WP-RW"):
            L += ["", f"| {w} | " + " | ".join(r["walks"][w]) + " |", "| --- | " + " | ".join("---" for _ in r["walks"][w]) + " |",
                  "| R@5 | " + " | ".join(pct(v["R@5"]) for v in r["walks"][w].values()) + " |"]
        L += ["", "The share of each reference's gain that ours recovers:", "",
              "| walk | reference | reference's gain [95%] | our gain [95%] | ours − reference | share [95%] | call |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for w in WALKS:
            for ref, c in r["shares"][w].items():
                g, o, d = c["gain_ref"], c["gain_ours"], c["ours_minus_ref"]
                sh = "-" if c["share"] is None else f"{c['share']:.2f} [{c['share_ci'][0]:.2f}, {c['share_ci'][1]:.2f}]"
                L.append(f"| {w} | {ref} | {100 * g[0]:+.2f} [{100 * g[1]:+.2f}, {100 * g[2]:+.2f}] | "
                         f"{100 * o[0]:+.2f} [{100 * o[1]:+.2f}, {100 * o[2]:+.2f}] | {100 * d[0]:+.2f} | {sh} | "
                         f"{c['call']} |")
        L += ["", "WH questions with a matched phrase: " + ", ".join(f"{k} {v:.3f}" for k, v in r["wh_matched"].items()),
              ""]
    return "\n".join(L) + "\n"


# ── self-test ─────────────────────────────────────────────────────────────────


def selftest():
    import spacy
    import scipy.sparse as sp
    import b1_build as B
    import b1c_walks as BC
    # the normal form
    assert nf("The Collegian (Houston)") == "collegian houston"
    assert nf("Barcelona's youth academy") == "barcelona youth academy"
    assert nf("Cesc F\ufffdbregas") == "cesc f bregas" and nf("a") == "" and nf("An Apple") == "apple"
    assert title_nf("Green (band)") == "green" and title_nf("Lionel Messi") == "lionel messi"
    # every pattern on parsed sentences (the parse is spaCy's; the rules are ours)
    nlp = spacy.load("en_core_web_sm")
    ph, tr = extract(nlp("Lionel Messi joined FC Barcelona in 2004. He scored against Real Madrid."), "Lionel Messi")
    assert {"lionel messi", "fc barcelona", "real madrid"} <= ph, ph
    assert ("lionel messi", "join", "barcelona") in tr, tr                         # the entity holds the head token
    assert ("lionel messi", "score against", "real madrid") in tr, tr            # pronoun -> title, verb + prep
    ph, tr = extract(nlp("Kerry won the Munster Championship. Colin O'Mahony, a footballer, plays for Kerry."), "Colin O'Mahony")
    assert ("kerry", "win", "munster championship") in tr, tr
    assert ("colin o mahony", "be", "footballer") in tr, tr                       # apposition
    ph, tr = extract(nlp("The film was directed by Steven Spielberg."), "Jaws (film)")
    assert ("film", "direct by", "steven spielberg") in tr, tr                    # agent
    ph, tr = extract(nlp("Teutberga was a daughter of Boso the Elder."), "Teutberga")
    assert ("teutberga", "be", "daughter") in tr and ("daughter", "of", "boso the elder") in tr, tr
    ph, tr = extract(nlp("Barcelona's youth academy trained Messi."), "x")
    assert ("barcelona", "poss", "barcelona youth academy") in tr, tr             # possessive
    ph, tr = extract(nlp("Lionel Messi was born in Rosario. Messi moved to Spain."), "x")
    assert "messi" not in ph and ("lionel messi", "move to", "spain") in tr, (ph, tr)   # surname rule
    ph, tr = extract(nlp("The National Basketball Association (NBA) was founded in New York."), "x")
    assert "nba" not in ph and "national basketball association" in ph, ph       # acronym rule
    ph, tr = extract(nlp("It is the team that won the cup."), "x")
    assert ("team", "win", "cup") in tr, tr                                       # relative clause
    for _p, t in [extract(nlp("He met himself."), "")]:
        assert all(s != o for s, _r, o in t)
    # the P rule against b1_build.ner_edges on made-up passages, and by hand
    texts = ["Paris is in France.", "Paris has the Louvre.", "France borders Spain.", "Spain and France.", "Nothing."]
    keys = ner_keys_of(texts)
    s1, d1, w1, _ = B.ner_edges(texts)
    s2, d2, w2 = p_edges(keys)
    assert np.array_equal(s1, s2) and np.array_equal(d1, d2) and np.allclose(w1, w2, atol=1e-6)
    s3, d3, w3 = p_edges([{"a", "b"}, {"a"}, {"a", "b"}, {"c"}])
    assert list(zip(s3.tolist(), d3.tolist())) == [(0, 1), (0, 2), (1, 2)] and np.allclose(w3, [1 / 3, 1 / 3 + 1 / 2, 1 / 3])
    # the H walk against a dense power iteration
    g = Graph("x", ["a", "b", "c"], [[0], [0, 1], [2], []], [[(0, 1)], [], [], []])
    A = h_form(4, g.phr_ids, g.all_pairs(), 3).toarray()
    assert A[0, 4] == 1 and A[4, 5] == 2 and A[5, 4] == 2 and A[3].sum() == 0 and A.shape == (7, 7)
    T = BC.transition_T(sp.csr_matrix(A))
    deg = A.sum(1)
    P = np.where(deg[:, None] > 0, A / np.where(deg[:, None] > 0, deg[:, None], 1), np.eye(7))
    r = np.array([[0.2], [0.1], [0], [0.1], [0.6], [0], [0]])
    x = r.copy()
    for _ in range(200):
        x = ALPHA * r + (1 - ALPHA) * P.T @ x
    assert np.allclose(BC.ppr(T, r, iters=200), x, atol=1e-12)
    # the reset: top five, ties, no match
    g2 = Graph("y", ["red", "big red", "dog", "big", "blue", "cat", "the"], [[0, 1, 2], [0, 3], [4, 5], [2]],
               [[], [], [], []])
    m = match_phrases("Is the big red dog blue, or a cat?", g2)
    assert [j for j, _w in m] == [1, 3, 4, 5, 0], m                               # 1/df, then longer, then earlier
    assert match_phrases("nothing here", g2) == []
    rrf = np.tile(np.arange(4), (2, 1))
    L, mt = wh_lists(BC, g2, ["nothing here", "the dog"], np.array([[0.1, 0.4, 0.3, 0.2], [0.1, 0.4, 0.3, 0.2]]),
                     np.array([[1, 2, 3, 0], [1, 2, 3, 0]]), rrf, 4)
    assert L[0].tolist() == [1, 2, 3, 0] and not mt[0] and mt[1]
    # the reference readers
    rows, ex = read_hipporag([{"title": "T", "text": "x", "extracted_entities": "['Ann', 'Bob', 'Cy']",
                               "extracted_triples": "[['Ann', 'r', 'Bob'], ['Ann', 'r'], ['', 'r', 'Bob'], ['Bob', 'r', 'Bob']]"}])
    assert rows[0][2] == {"ann", "bob"} and rows[0][3] == {"ann", "bob", "cy"} and rows[0][4] == [("ann", "bob")], rows
    assert ex == {"triples": 2, "triples_skipped": 2, "triples_one_phrase": 1, "unreadable_passages": 0}, ex
    rows, ex = read_sirerag([{"text": "T\nx", "facts": [{"fact": "f", "entities": ["Ann", "Bob", "Cy"]},
                                                         {"fact": "g", "entities": ["Ann"]}]}])
    assert rows[0][0] == "T" and sorted(rows[0][4]) == [("ann", "bob"), ("ann", "cy"), ("bob", "cy")], rows
    assert ex == {"propositions": 2}
    assert aligned([("T", "x  y", set(), set(), [])], [{"title": "t", "text": "X y"}])[0]
    # agreement
    ref = Graph("r", ["a", "b", "c"], [[0, 1], [2]], [[(0, 1)], []], ents=[[0, 1, 2], [2]])
    ours = Graph("o", ["a", "c", "d"], [[0, 1], [2]], [[(0, 1)], []])
    a = agreement(ref, ours)
    # ours {a, c} and {d} against the reference's {a, b} and {c} (entities {a, b, c} and {c}): 1 of 3 found; 2 of ours in
    assert a["phrases"] == {"recall": 0.3333, "precision": 0.6667} and a["pairs"] == {"recall": 0.0, "precision": 0.0}
    ref = Graph("r", ["a", "b"], [[0], [0], [1], [1]], [[], [], [], []])         # P edges (0,1) and (2,3)
    ours = Graph("o", ["a", "c", "d"], [[0], [0], [1], [2]], [[], [], [], []])   # P edge (0,1) only
    a = agreement(ref, ours)["P_edges"]
    assert a == {"recall": 0.5, "precision": 1.0, "reference": 2, "ours": 1}, a
    # the share's calls
    idx = boot_idx(400)
    z = np.zeros(400)
    one = np.r_[np.ones(200), np.zeros(200)]
    assert share_call(z, one, idx)["call"] == "NO_REFERENCE_GAIN"
    assert share_call(one, one, idx)["call"] == "MATCHED" and share_call(one, one, idx)["share"] == 1.0
    half = np.r_[np.ones(100), np.zeros(300)]
    assert share_call(one, half, idx)["call"] == "PARTIAL"
    assert share_call(one, z, idx)["call"] == "NONE"
    # a parse part written and read back
    arr = pack_parse([{"a b", "c"}, set()], [{("a b", "r", "c")}, set()])
    p = OUT / "selftest_parse.npz"
    save_npz(p, **arr)
    zz = dict(np.load(p))
    os.remove(p)
    gg = graph_from_parse(zz)
    assert gg.phrase_sets() == [{"a b", "c"}, set()] and gg.pair_sets() == [{("a b", "c")}, set()]
    assert unpack(*pack(["\u00e9t\u00e9", "", "x"])) == ["\u00e9t\u00e9", "", "x"]
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("parse", "refs", "run", "verdict", "unpack-spacy"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("--part")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.stage is None:
        return selftest()
    if a.stage == "unpack-spacy":
        return unpack_spacy()
    if a.stage == "parse":
        if a.part:
            k, n = (int(x) for x in a.part.split("/"))
            if a.name not in FULL_PARTS or n != FULL_PARTS[a.name] or not 0 <= k < n:
                raise SystemExit(f"parse: {a.name} {a.part} is not one of the declared parts {FULL_PARTS}")
            parse_part(a.name, k, n, a.host, a.limit)
        else:
            if a.name not in SETTINGS:
                raise SystemExit(f"parse: B1 settings are {list(SETTINGS)}")
            parse_b1(a.name, a.limit)
        return 0
    if a.stage == "refs":
        refs_stage(a.name, a.limit)
        return 0
    if a.stage == "run":
        run_stage(a.name, a.limit)
        return 0
    return verdict_stage()


if __name__ == "__main__":
    sys.exit(main())
