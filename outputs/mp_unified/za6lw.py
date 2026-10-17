"""A6 (docs/A6_LINK_WORDS_INPUT.md): the words around a link as a fixed input to both of the lean pair's models.
D4 (docs/DIAG_LINK_WORDS.md) scored a link between two passages by the words around it: the largest summed IDF of the
question's stems (less both ends' title stems) in one window of the link (structural: the 10 tokens on each side of each
occurrence of one end's title in the other's text, or the relation names on a KB graph; NER: the 10 tokens on each
side of each mention of an entity both passages mention; kNN: none). Fused with zrc's score it ranked musique's missed
golds 1st or 2nd +0.036 and +0.018 more often, short of D4's bar for a fused score; it never reached a model. A6 gives
D4's link score to both models as six columns per row, computed once from the first stage's list, before either model
scores a row:

    anchors   the question's top five rows by rrf among the retrieved rows (rrf > 0), ties by row
    w_f(v)    over the anchors a other than v linked to v in family f (structural, NER), the largest link score of the
              links between a and v (D4's link_score), 0 without one
    columns   log1p w_s, log1p w_n, w_s / Q, w_n / Q (Q the question's total stem IDF; 0 when it is 0), and a flag per
              family that v has a window with an anchor

No parameter, label or score of a model; nothing is normalised over the question's scores. NER mentions come from
spaCy 3.8.15 with en_core_web_sm 3.8.0 run as the graph's NER family was built (D4's settings), on every node of all
six corpora, in parts, merged and kept only when they rebuild at least 0.9 of the graph's NER edges. Both models read
the columns through R1's pointwise head, Linear(6, 16), GELU, Linear(16, 1) (129 parameters, the last layer zero at
the start, the first drawn from its own generator, seed + 6601):

    zrckpw  zrckp's model plus the head at the top, on zrckp's carve with the columns beside
    zspkw   zspk's model with the head added with the KT head before zsp's propagation reads the score

    python outputs/mp_unified/za6lw.py spacy --host
    python outputs/mp_unified/za6lw.py ents --dataset 2wiki --part 0/60 --host
    python outputs/mp_unified/za6lw.py merge --dataset 2wiki --host
    python outputs/mp_unified/za6lw.py build --dataset 2wiki --carve fit --host
    python outputs/mp_unified/za6lw.py train --split L-musique --name scr-a6-zrckpw --arm zrckpw --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za6lw.py read --name scr-a6-zrckpw --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za6lw.py compare --new outputs/screen/fits/scr-a6-zrckpw \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a6-zrckpw
    python outputs/mp_unified/za6lw.py pair --arm zrckpw --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za6lw.py recall --arm zrckpw --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za6lw.py verdict
    python outputs/mp_unified/za6lw.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if sys.argv[1:2] in (["spacy"], ["ents"], ["merge"], ["build"]):     # one CPU per build item
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1c3 as Z3  # noqa: E402  (zspk and its KT carves; through it P1c2's roots, gate and card, zprop, zlink, rmatch)
import zp1c3p as Z3P  # noqa: E402  (zrckp)
import linkwords as LW  # noqa: E402  (D4: tokens, stems, windows, the link score, spaCy's settings, the NER rebuild)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A6_LINK_WORDS_INPUT.md"
OUT = ROOT / "outputs" / "a6"
ENTS_DIR = OUT / "ents"
PARTS_DIR = ENTS_DIR / "parts"
COLS_DIR = OUT / "cols"
SCREEN = ROOT / "outputs" / "screen"
A6_KEY = "_a6"
N_ANCHOR = 5
COL_NAMES = ("log1p_w_struct", "log1p_w_ner", "share_struct", "share_ner", "has_struct", "has_ner")
NC = len(COL_NAMES)
A6_HID, A6_SEED_OFF = 16, 6601
A6 = ("a6_w1", "a6_b1", "a6_w2", "a6_b2")
N_A6 = A6_HID * NC + A6_HID + A6_HID + 1
ENT_PARTS = {"metaqa": 1, "squad": 1, "musique": 2, "webqsp": 10, "hotpotqa": 40, "2wiki": 60}
PART_ARRAYS = ("node", "lkey", "start", "end", "label")
QUERY_FILES = ("train", "dev", "validation")                   # never test
ENT_SETTINGS = {"spacy": LW.SPACY_VERSION, "model": LW.MODEL_VERSION, "disable": ["parser", "lemmatizer"],
                "labels": list(LW.ENT_LABELS), "key": "e.text.lower().strip()", "min_chars": LW.MIN_ENT_CHARS,
                "df": [LW.DF_MIN, LW.DF_MAX], "text": "the node's text field", "batch_size": 256}
BUILD_SETTINGS = {"columns": list(COL_NAMES), "anchors": f"the first stage's top {N_ANCHOR} rows by rrf among the "
                                                         "retrieved rows (rrf > 0), ties by row",
                  "score": "D4's link score (linkwords.link_score) of the links between the row and an anchor, per "
                           "family; the largest over the anchors other than the row",
                  "windows": "structural: title occurrences, relation names on KB graphs; ner: shared-entity mentions "
                             "(df within [2, 25]); knn: none",
                  "share": "the score over the question's total stem IDF, 0 when that is 0",
                  "has": "1 when the row has a window with an anchor in the family", "window": LW.WIN,
                  "stem": LW.STEM, "idf": "log((N + 1) / (df + 1)) over every node's title and text",
                  "stored": "float16"}
SETTINGS = {"columns": NC, "hidden": A6_HID, "seed_offset": A6_SEED_OFF, "parameters": N_A6, "score_input": False,
            "build": BUILD_SETTINGS, "ents": ENT_SETTINGS}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpw": "zrckp", "zspkw": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a6-{a}", "L-hotpotqa": f"scr-a6-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpw": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkw": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpw": "zrckp plus the A6 head over the link-word columns",
        "zspkw": "zspk plus the A6 head over the link-word columns, before its propagation"}
PAIR = {a: SCREEN / f"scr-a6-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a6-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("musique", "2wiki", "hotpotqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A6's re-call of the arm, or "
        "not twelve reads (docs/A6_LINK_WORDS_INPUT.md, section 4)")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(json.dumps(rec, indent=1, ensure_ascii=False).encode("utf-8"))
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def manifest(served, ds):
    man = json.loads((Path(served) / ds / "DATASET.json").read_text(encoding="utf-8"))
    return int(man["nodes"]["n"]), man["nodes"]["sha256"]


# ── the NER mentions: spaCy in parts, merged ─────────────────────────────────


def load_nlp(host):
    """spaCy as D4 loads it (the host: the laptop's installed files, unpacked once by `spacy`), versions checked."""
    path = LW.spacy_path(host)
    if path and path not in sys.path:
        sys.path.append(path)
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    if spacy.__version__ != LW.SPACY_VERSION or nlp.meta.get("version") != LW.MODEL_VERSION:
        raise SystemExit(f"za6lw: spaCy {spacy.__version__} / model {nlp.meta.get('version')}, declared "
                         f"{LW.SPACY_VERSION} / {LW.MODEL_VERSION}")
    return nlp, path


def part_lines(N, k, n):
    if not 0 <= k < n:
        raise SystemExit(f"za6lw: part {k}/{n}")
    return k * N // n, (k + 1) * N // n


def mention_rows(nlp, texts, lo):
    """(node, key, start, end, label) per kept entity, as linkwords._ents_batch keeps them."""
    rows = []
    for j, doc in enumerate(nlp.pipe(texts, batch_size=ENT_SETTINGS["batch_size"])):
        for e in doc.ents:
            if e.label_ in LW.ENT_LABELS and len(e.text) >= LW.MIN_ENT_CHARS:
                rows.append((lo + j, e.text.lower().strip(), e.start_char, e.end_char, LW.ENT_LABELS.index(e.label_)))
    return rows


def ents_part(ds, k, n, served, nlp, out_root=PARTS_DIR, where="laptop", spacy_from=None):
    """The mentions of lines [k N // n, (k + 1) N // n) of the dataset's nodes.jsonl, with the part's own key list."""
    t0 = time.time()
    N, want = manifest(served, ds)
    lo, hi = part_lines(N, k, n)
    texts = []
    with open(Path(served) / ds / "nodes.jsonl", "rb") as f:
        for i, raw in enumerate(f):
            if i >= hi:
                break
            if i >= lo:
                texts.append(json.loads(raw)["text"])
    if len(texts) != hi - lo:
        raise SystemExit(f"za6lw: {ds} part {k}/{n}: {len(texts)} lines read, not {hi - lo}")
    rows = mention_rows(nlp, texts, lo)
    keys = sorted({r[1] for r in rows})
    kid = {x: i for i, x in enumerate(keys)}
    arr = {"node": np.asarray([r[0] for r in rows], np.int32), "lkey": np.asarray([kid[r[1]] for r in rows], np.int32),
           "start": np.asarray([r[2] for r in rows], np.int32), "end": np.asarray([r[3] for r in rows], np.int32),
           "label": np.asarray([r[4] for r in rows], np.int8)}
    kj = np.frombuffer(json.dumps(keys, ensure_ascii=False).encode("utf-8"), np.uint8)
    stem = Path(out_root) / ds / f"part_{k}of{n}"
    save_npz(stem.with_suffix(".npz"), keys_json=kj, **arr)
    rec = {"declared_in": DECLARED, "stage": "A6", "dataset": ds, "part": [k, n], "lines": [lo, hi], "nodes": N,
           "nodes_sha256_declared": want, "settings": ENT_SETTINGS, "mentions": int(arr["node"].size),
           "keys": len(keys), "arrays_sha256": LW.arrays_sha(arr, PART_ARRAYS),
           "keys_sha256": hashlib.sha256(kj.tobytes()).hexdigest(), "where": where, "spacy_from": spacy_from,
           "seconds": round(time.time() - t0, 1), "script_sha256": LC.sha_src(__file__), "utc": utc()}
    write_json(stem.with_suffix(".json"), rec)
    log(f"za6lw ents {ds} part {k}/{n}: lines {lo}..{hi}, {rec['mentions']} mentions, {rec['keys']} keys "
        f"({rec['seconds']}s)")
    return rec


def load_part(ds, k, n, N, want, root=PARTS_DIR):
    stem = Path(root) / ds / f"part_{k}of{n}"
    rp = stem.with_suffix(".json")
    if not rp.exists():
        raise SystemExit(f"za6lw: {rp} missing (za6lw.py ents --dataset {ds} --part {k}/{n})")
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if (rec.get("declared_in") != DECLARED or rec.get("settings") != ENT_SETTINGS or rec.get("dataset") != ds
            or rec.get("part") != [k, n] or rec.get("lines") != list(part_lines(N, k, n)) or rec.get("nodes") != N
            or rec.get("nodes_sha256_declared") != want):
        raise SystemExit(f"za6lw: {rp} is not A6's part {k}/{n} of {ds} on these nodes")
    z = np.load(stem.with_suffix(".npz"))
    arr = {x: z[x] for x in PART_ARRAYS}
    kj = z["keys_json"]
    if LW.arrays_sha(arr, PART_ARRAYS) != rec["arrays_sha256"] or hashlib.sha256(kj.tobytes()).hexdigest() != rec[
            "keys_sha256"]:
        raise SystemExit(f"za6lw: {stem}.npz is not its record's")
    return json.loads(bytes(kj).decode("utf-8")), arr, LC.sha_file(rp)


def nodes_sha(served, ds):
    h, cnt = hashlib.sha256(), 0
    with open(Path(served) / ds / "nodes.jsonl", "rb") as f:
        for raw in f:
            h.update(raw)
            cnt += 1
    return cnt, h.hexdigest()


def merge(ds, served, n, parts_root=PARTS_DIR, out_root=ENTS_DIR, d4_root=LW.ENTS, freeze=None, where="laptop"):
    """The parts' mentions as one file in D4's format (keys sorted, df the distinct nodes per key, rows in node order),
    the nodes' sha256 checked, kept only when they rebuild at least REPRO_MIN of the graph's NER edges."""
    t0 = time.time()
    N, want = manifest(served, ds)
    parts = [load_part(ds, k, n, N, want, parts_root) for k in range(n)]
    allk = sorted(set().union(*(set(p[0]) for p in parts)))
    kid = {x: i for i, x in enumerate(allk)}
    cols = {x: [] for x in ("node", "key", "start", "end", "label")}
    for keys, arr, _sha in parts:
        g = np.asarray([kid[x] for x in keys], np.int64)
        cols["node"].append(arr["node"].astype(np.int64))
        cols["key"].append(g[arr["lkey"].astype(np.int64)] if keys else np.zeros(0, np.int64))
        for x in ("start", "end", "label"):
            cols[x].append(arr[x])
    node = np.concatenate(cols["node"]) if cols["node"] else np.zeros(0, np.int64)
    key = np.concatenate(cols["key"]) if cols["key"] else np.zeros(0, np.int64)
    if node.size and (np.any(np.diff(node) < 0) or node.min() < 0 or node.max() >= N):
        raise SystemExit(f"za6lw: {ds}: the parts' mentions are not in node order within the corpus")
    pairs = np.unique(key * N + node) if node.size else np.zeros(0, np.int64)
    arr = {"node": node.astype(np.int32), "key": key.astype(np.int32),
           "start": np.concatenate(cols["start"]).astype(np.int32), "end": np.concatenate(cols["end"]).astype(np.int32),
           "label": np.concatenate(cols["label"]).astype(np.int8),
           "df": np.bincount(pairs // N, minlength=len(allk)).astype(np.int32)}
    cnt, got = nodes_sha(served, ds)
    if cnt != N or got != want:
        raise SystemExit(f"za6lw: {ds}/nodes.jsonl has {cnt} lines and sha256 {got}; DATASET.json names {N} and {want}")
    gz = np.load(Path(served) / ds / "graph" / "ner.npz")
    rp = LW.repro(arr, N, gz["src"], gz["dst"], gz["weight"].astype(np.float64))
    ok = rp["reproduced"] is not None and rp["reproduced"] >= LW.REPRO_MIN
    sha = LW.arrays_sha(arr, LW.ENT_ARRAYS)
    d4j = Path(d4_root) / f"{ds}_ents.json"
    d4 = json.loads(d4j.read_text(encoding="utf-8")) if d4j.exists() else None
    rec = {"declared_in": DECLARED, "stage": "A6", "dataset": ds, "nodes": N, "nodes_sha256": want,
           "freeze_RECORD_SHA256": freeze, "spacy": LW.SPACY_VERSION, "model": LW.MODEL_VERSION,
           "from": f"A6's {n} parts ({where})", "settings": {"disable": ENT_SETTINGS["disable"],
                                                              "labels": ENT_SETTINGS["labels"],
                                                              "key": ENT_SETTINGS["key"],
                                                              "min_chars": ENT_SETTINGS["min_chars"],
                                                              "df": ENT_SETTINGS["df"]},
           "mentions": int(arr["node"].size), "keys": len(allk),
           "keys_in_df_range": int(((arr["df"] >= LW.DF_MIN) & (arr["df"] <= LW.DF_MAX)).sum()),
           "arrays_sha256": sha, "graph": rp, "gate": {"repro_min": LW.REPRO_MIN, "passed": bool(ok)},
           "parts": {f"part_{k}of{n}": parts[k][2] for k in range(n)},
           "d4": None if d4 is None else {"arrays_sha256": d4["arrays_sha256"], "identical": d4["arrays_sha256"] == sha},
           "seconds": round(time.time() - t0, 1), "script_sha256": LC.sha_src(__file__), "utc": utc()}
    stem = Path(out_root) / f"{ds}_ents"
    if not ok:
        write_json(stem.with_name(f"{ds}_ents.gate_failed.json"), rec)
        raise SystemExit(f"za6lw: {ds}: the mentions rebuild {rp['reproduced']} of the graph's NER edges, below "
                         f"{LW.REPRO_MIN}; nothing kept")
    blob = np.frombuffer("\n".join(k.replace("\n", " ") for k in allk).encode("utf-8"), np.uint8)
    save_npz(stem.with_suffix(".npz"), keys_blob=blob, **arr)
    write_json(stem.with_suffix(".json"), rec)
    log(f"za6lw merge {ds}: {rec['mentions']} mentions, {rec['keys']} keys; the graph's NER edges rebuilt "
        f"{rp['reproduced']} (extra {rp['extra_share']}); D4's file "
        f"{'identical' if d4 and rec['d4']['identical'] else 'differs' if d4 else 'absent'} ({rec['seconds']}s)")
    return rec


# ── the columns ──────────────────────────────────────────────────────────────


class Served:
    """The served package's questions (never a test split), one corpus pass (linkwords.corpus_pass: the question stems'
    IDF and the needed nodes' titles and texts, the nodes' sha256 checked), A6's mentions and the KB relation names."""

    def __init__(self, host, ents_root=ENTS_DIR):
        import u1b as U1B
        self.served, self.freeze = U1B.package_root(host)
        self.ents_root = ents_root

    def questions(self, ds):
        out = {}
        for fn in QUERY_FILES:
            p = self.served / ds / "queries" / f"{fn}.jsonl"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        out.setdefault(r["query_id"], r["question"])
        return out

    def corpus(self, ds, need, qset):
        return LW.corpus_pass(self.served, ds, need, qset)

    def mentions(self, ds, nodes_sha256):
        m, rec = LW.load_ents(ds, nodes_sha256, self.ents_root)
        if m is None:
            raise SystemExit(f"za6lw: {ds}'s mentions are missing (za6lw.py merge --dataset {ds})")
        return m, LC.sha_file(Path(self.ents_root) / f"{ds}_ents.json")

    def kb_names(self, ds):
        return LW.kb_names(ds) if ds in LW.KB else None


def anchors_of(rrf, k=N_ANCHOR):
    """The first stage's top k rows: the retrieved rows (rrf > 0) by rrf, ties by row."""
    ret = np.flatnonzero(rrf > 0)
    return ret[np.argsort(-rrf[ret], kind="stable")][:k]


def anchor_edges(n, A, eu, ev, ef):
    """The edges of the worded families (structural 0, NER 1) with an anchor at one end, self-loops left out."""
    isA = np.zeros(n, bool)
    isA[A] = True
    return np.flatnonzero((eu != ev) & (ef < 2) & (isA[eu] | isA[ev]))


def question_cols(n, A, eu, ev, ef, er, pool, q, idf, doc, kbn):
    """One question's six columns, float64 (n, NC)."""
    out = np.zeros((n, NC))
    isA = np.zeros(n, bool)
    isA[A] = True
    links = {}
    for i in range(eu.size):
        u, v, f = int(eu[i]), int(ev[i]), int(ef[i])
        for a, c in ((u, v), (v, u)):
            if isA[a]:
                links.setdefault((a, c, f), []).append(None if er is None else er[i])
    best, has = np.zeros((n, 2)), np.zeros((n, 2), bool)
    for (a, c, f), rels in links.items():
        lk = [(f, r) for r in rels] if (f == 0 and kbn is not None) else [(f, None)]
        sc, _fam, nw = LW.link_score(doc(int(pool[a])), doc(int(pool[c])), lk, q, idf, kbn)
        if sc > best[c, f]:
            best[c, f] = sc
        if nw:
            has[c, f] = True
    qm = float(sum(idf[w] for w in q))
    out[:, 0:2] = np.log1p(best)
    if qm > 0:
        out[:, 2:4] = best / qm
    out[:, 4:6] = has
    return out


def build(ds, carve, src, part=None, out_root=COLS_DIR, cache_root=None, look_root=None, placement=None):
    """The six columns of one step-1 cache part of a carve (every part when part is None), from P1c2's look (pools, rrf
    and edges), one corpus pass and A6's mentions."""
    cache_root, look_root = cache_root or Z2.CACHE, look_root or Z2.LOOK
    t0 = time.time()
    d, files, ids, head = LC.look_chunks(ds, carve, look_root)
    if "rrf" not in head["columns"]:
        raise SystemExit(f"za6lw: {d}: the look has no rrf column")
    j_rrf = head["columns"].index("rrf")
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    if part is not None and not 0 <= part < len(parts):
        raise SystemExit(f"za6lw: {ds}/{carve} has {len(parts)} parts, not a part {part}")
    qtext = src.questions(ds)
    kbn = src.kb_names(ds)
    plan, need = [], set()
    for pi, (p, r) in enumerate(zip(parts, recs)):
        if part is not None and pi != part:
            continue
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"za6lw: {p}: built from another look than {d}")
        qa, qb = r["query_range"]
        if qb - qa != r["queries"] or not 0 <= qa <= qb <= len(ids):
            raise SystemExit(f"za6lw: {p}: query range {qa}..{qb} for {r['queries']} queries of {len(ids)}")
        miss = [x for x in ids[qa:qb] if x not in qtext]
        if miss:
            raise SystemExit(f"za6lw: {ds}/{carve}: {len(miss)} questions are in no {'/'.join(QUERY_FILES)} file "
                             f"(first {miss[:3]})")
        qi = qa
        lo, hi = r["chunks"]
        qs, n_all = [], []
        for ch in range(lo, hi):
            with np.load(files[ch]) as zf:
                c = {k: zf[k] for k in ("q_pool_size", "q_edges", "pool", "x", "e_u", "e_v", "e_fam")}
                er_all = zf["e_rel"] if "e_rel" in zf.files else None
            if c["x"].shape[1] != len(head["columns"]):
                raise SystemExit(f"za6lw: {files[ch]}: {c['x'].shape[1]} columns, the look names {len(head['columns'])}")
            qps, qed = c["q_pool_size"].astype(np.int64), c["q_edges"].astype(np.int64)
            no, eo = np.r_[0, np.cumsum(qps)], np.r_[0, np.cumsum(qed)]
            if int(no[-1]) != c["pool"].size or int(eo[-1]) != c["e_u"].size:
                raise SystemExit(f"za6lw: {files[ch]}: pools or edges disagree with q_pool_size and q_edges")
            for i in range(qps.size):
                n = int(qps[i])
                pool = c["pool"][no[i]:no[i + 1]].astype(np.int64)
                eu, ev = c["e_u"][eo[i]:eo[i + 1]].astype(np.int64), c["e_v"][eo[i]:eo[i + 1]].astype(np.int64)
                ef = c["e_fam"][eo[i]:eo[i + 1]].astype(np.int64)
                if eu.size and (min(eu.min(), ev.min()) < 0 or max(eu.max(), ev.max()) >= n):
                    raise SystemExit(f"za6lw: {files[ch]} question {i}: an edge leaves its pool")
                A = anchors_of(c["x"][no[i]:no[i + 1], j_rrf].astype(np.float64))
                sel = anchor_edges(n, A, eu, ev, ef)
                er = None if er_all is None else er_all[eo[i]:eo[i + 1]][sel]
                if er is not None and kbn is not None and er.size and int(er.max()) >= len(kbn):
                    raise SystemExit(f"za6lw: {ds}: relation slot {int(er.max())}, the vocabulary has {len(kbn)} names")
                qs.append((ids[qi], n, A, eu[sel], ev[sel], ef[sel], er, pool))
                need.update(pool[A].tolist())
                need.update(pool[eu[sel]].tolist())
                need.update(pool[ev[sel]].tolist())
                n_all.append(n)
                qi += 1
        n_all = np.asarray(n_all, np.int64)
        if n_all.size != r["queries"] or int(n_all.sum()) != r["rows"]:
            raise SystemExit(f"za6lw: {p}: {n_all.size} queries and {int(n_all.sum())} rows against {r['queries']} and "
                             f"{r['rows']}")
        if (p / "n.npy").exists() and not np.array_equal(
                np.asarray(LC.load_array(p, "n", r, verify=True)).astype(np.int64), n_all):
            raise SystemExit(f"za6lw: {p}: the chunks' pool sizes are not step 1's cached n")
        if qi != qb:
            raise SystemExit(f"za6lw: {p}: its chunks hold questions {qa}..{qi}, its record names {qa}..{qb}")
        plan.append((p, r, qs))
    qst = {q_[0]: LW.question_stems(qtext[q_[0]]) for _p, _r, qs in plan for q_ in qs}
    texts, idf, info = src.corpus(ds, need, frozenset().union(*qst.values()) if qst else frozenset())
    mentions, ents_sha = src.mentions(ds, info["nodes_sha256"])

    @functools.lru_cache(maxsize=50000)
    def doc(i):
        t = texts[i]
        return LW.Doc(t[0], t[1], mentions(i))

    done = {}
    for p, r, qs in plan:
        tp = time.time()
        x = np.concatenate([question_cols(n, A, eu, ev, ef, er, pool, qst[qid], idf, doc, kbn)
                            for qid, n, A, eu, ev, ef, er, pool in qs]) if qs else np.zeros((0, NC))
        if x.shape[0] != r["rows"] or not np.isfinite(x).all():
            raise SystemExit(f"za6lw: {p.name}: {x.shape[0]} rows of columns, or a column is not finite")
        x16 = x.astype(np.float16)
        out = Path(out_root) / ds / carve / p.name
        out.mkdir(parents=True, exist_ok=True)
        sha = R.save_npy(out, "cols", x16)
        qs_ = lambda y: [round(float(v), 4) for v in np.quantile(y, [0.05, 0.5, 0.95])] if y.size else []
        rec = {"declared_in": DECLARED, "stage": "A6", "dataset": ds, "carve": carve, "part": p.name,
               "step1_record_sha256": LC.sha_file(p / "record.json"), "look": head["records"],
               "carve_ids_sha256": head["carve_ids_sha256"], "ents_record_sha256": ents_sha,
               "nodes_sha256": info["nodes_sha256"], "queries": len(qs), "rows": int(x.shape[0]),
               "anchors_per_question": round(float(np.mean([q_[2].size for q_ in qs])), 3) if qs else 0.0,
               "anchor_edges": int(sum(q_[3].size for q_ in qs)), "kb_names": kbn is not None,
               "cache_root": Path(cache_root).resolve().relative_to(ROOT).as_posix()
               if Path(cache_root).resolve().is_relative_to(ROOT) else "toy",
               "settings": BUILD_SETTINGS,
               "rows_with_a_window": {f: round(float(x[:, 4 + j].mean()), 4) if x.size else 0.0
                                      for j, f in enumerate(("struct", "ner"))},
               "rows_with_a_word": {f: round(float((x[:, j] > 0).mean()), 4) if x.size else 0.0
                                    for j, f in enumerate(("struct", "ner"))},
               "quantiles_5_50_95_log1p_where_a_word": {f: qs_(x[x[:, j] > 0, j]) for j, f in
                                                        enumerate(("struct", "ner"))},
               "max_abs_float16_error": float(np.abs(x16.astype(np.float64) - x).max()) if x.size else 0.0,
               "arrays": {"cols": {"dtype": x16.dtype.str, "shape": list(x16.shape), "sha256": sha}},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "seconds": round(time.time() - tp, 1), "utc": utc()}
        R.write_json(out / "record.json", rec)
        done[p.name] = {"queries": rec["queries"], "rows": rec["rows"], "windows": rec["rows_with_a_window"]}
        log(f"  {ds}/{carve} {p.name}: {rec['queries']} queries, {rec['rows']} rows; rows with a window "
            f"{rec['rows_with_a_window']}, with a word {rec['rows_with_a_word']} ({time.time() - t0:.0f}s)")
    doc.cache_clear()
    log(f"za6lw build {ds}/{carve}: {len(done)} part(s), {len(need)} nodes read ({time.time() - t0:.0f}s)")
    return done


# ── the carve ────────────────────────────────────────────────────────────────


def load_a6(c, root, cache_root, verify=True):
    """A carve's six columns on its device (float16), each part tied to its step-1 record."""
    parts, _recs = LC.part_dirs(c.ds, c.carve, cache_root)
    xs, shas = [], {}
    for p in parts:
        d = Path(root) / c.ds / c.carve / p.name
        rp = d / "record.json"
        if not rp.exists():
            raise SystemExit(f"za6lw: {rp} missing (za6lw.py build --dataset {c.ds} --carve {c.carve})")
        h = json.loads(rp.read_text(encoding="utf-8"))
        if h.get("declared_in") != DECLARED or h.get("settings") != BUILD_SETTINGS:
            raise SystemExit(f"za6lw: {rp} is not A6's build")
        if h["step1_record_sha256"] != c.part_shas[p.name]:
            raise SystemExit(f"za6lw: {rp} was built on another step-1 part")
        fn = d / "cols.npy"
        if verify and LC.sha_file(fn) != h["arrays"]["cols"]["sha256"]:
            raise SystemExit(f"za6lw: {fn}: sha256 is not its record's")
        x = np.load(fn)
        if x.dtype != np.float16 or list(x.shape) != h["arrays"]["cols"]["shape"] or x.shape[1:] != (NC,):
            raise SystemExit(f"za6lw: {fn}: {x.dtype} {x.shape}, not its record's float16 (rows, {NC})")
        xs.append(x)
        shas[p.name] = LC.sha_file(rp)
    x = np.concatenate(xs)
    if x.shape[0] != int(c.off_np[-1]):
        raise SystemExit(f"za6lw: {c.ds}/{c.carve}: {x.shape[0]} rows of columns against the cache's "
                         f"{int(c.off_np[-1])}")
    return torch.from_numpy(x).to(c.device), shas


def a6_batch(c, qs):
    """The columns of questions qs, in lean_gpu.CacheCarve.batch's row order, float32 on c's device."""
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    idx = np.repeat(c.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    return c.a6[torch.from_numpy(idx).to(c.device)].to(torch.float32)


def a6_carve(base, a6_root=COLS_DIR):
    """base's carve class with A6's columns beside it: batch() adds them under A6_KEY; every block, row, side key and
    gold is base's, unchanged."""

    class A6Carve(base):
        A6_ROOT = a6_root

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.a6, self.a6_records = load_a6(self, type(self).A6_ROOT, cache_root, verify)

        def nbytes(self):
            return super().nbytes() + self.a6.numel() * self.a6.element_size()

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[A6_KEY] = a6_batch(self, qs)
            return feats, nq, base_z, gold

    A6Carve.__name__ = A6Carve.__qualname__ = f"A6{base.__name__}"
    return A6Carve


A6KTChainCarveZRC = a6_carve(Z3.KTChainCarveZRC)
A6KTLinkCarveBase = a6_carve(Z3.KTLinkCarveBase)


# ── the head ─────────────────────────────────────────────────────────────────


def a6_init(m, seed):
    g = torch.Generator().manual_seed(int(seed) + A6_SEED_OFF)
    m.a6_w1 = nn.Parameter((torch.rand(A6_HID, NC, generator=g) * 2 - 1) / NC ** 0.5)
    m.a6_b1 = nn.Parameter(torch.zeros(A6_HID))
    m.a6_w2 = nn.Parameter(torch.zeros(1, A6_HID))
    m.a6_b2 = nn.Parameter(torch.zeros(1))


def a6_head(m, x):
    return Fn.linear(Fn.gelu(Fn.linear(x, m.a6_w1, m.a6_b1)), m.a6_w2, m.a6_b2).squeeze(-1)


class ZRCKPW(Z3P.ZRCKP):
    """zrckpw: zrckp's model plus the A6 head (zero at the start) over each row's own six columns."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        a6_init(self, seed)

    a6_head = a6_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if A6_KEY not in feats:
            return s
        return s + self.a6_head(feats[A6_KEY])


class A6Mix(Z3.KTMix):
    """zrm's score, the KT head, then the A6 head: in zspkw, the score zsp's propagation reads."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        a6_init(self, seed)

    a6_head = a6_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if A6_KEY not in feats:
            return s
        return s + self.a6_head(feats[A6_KEY])


class ZSPKW(ZP.ZProp, A6Mix):
    """zspkw: zspk's model with the A6 head added before zsp's propagation reads the score (MRO: ZProp, A6Mix, KTMix,
    ZRM)."""


ARMS = {"zrckpw": (ZRCKPW, A6KTChainCarveZRC), "zspkw": (ZSPKW, A6KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def a6_records(carves, root=COLS_DIR, cache_root=None):
    """Each carve part's A6 record sha256; stops when one is missing or not A6's."""
    out = {}
    for ds, cv in carves:
        parts, _ = LC.part_dirs(ds, cv, cache_root or Z2.CACHE)
        for p in parts:
            rp = Path(root) / ds / cv / p.name / "record.json"
            if not rp.exists():
                raise SystemExit(f"za6lw: {rp} missing (za6lw.py build --dataset {ds} --carve {cv})")
            if json.loads(rp.read_text(encoding="utf-8")).get("declared_in") != DECLARED:
                raise SystemExit(f"za6lw: {rp} is not A6's")
            out[f"{ds}/{cv}/{p.name}"] = LC.sha_file(rp)
    return out


def ents_records(datasets, root=ENTS_DIR):
    out = {}
    for ds in datasets:
        rp = Path(root) / f"{ds}_ents.json"
        if not rp.exists():
            raise SystemExit(f"za6lw: {rp} missing (za6lw.py merge --dataset {ds})")
        out[ds] = LC.sha_file(rp)
    return out


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        cv = Z3.train_carves(split)
        rec.update({"za6lw_sha256": LC.sha_src(__file__), "linkwords_sha256": LC.sha_src(LW.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "a6": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                        "device": Z2.DEVICE, "head": list(A6), "settings": SETTINGS,
                        "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                        "gate": LC.sha_file(Z2.GATE), "a6_records": a6_records(cv),
                        "ents_records": ents_records(sorted({d for d, _ in cv})),
                        "kt_records": Z3.kt_records(cv), "rmatch_records": RM.built_records(),
                        "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def split_of_fit(name):
    for a in ARM_NAMES:
        for sp, n in FIT[a].items():
            if n == name:
                return a, sp
    return None, None


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za6lw: train takes --arm zrckpw or zspkw, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za6lw: A6's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za6lw: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    a6_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a6" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za6lw: {name} is not an A6 fit (arm {rec.get('arm')})")
    if rec["a6"].get("split") != split:
        raise SystemExit(f"za6lw: {name} is {split}'s fit, its record says {rec['a6'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    a6_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za6lw: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za6lw: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp)"),
            ("(rel's screen fits)", f"({b}'s card fits)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A6's link words) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A6 ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A6", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za6lw_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, base_screens=None):
    with on_base(arm):
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, arm)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools (the columns built again on P1g-b's looks), then "
                              "its full run, its own declared file",
        "GAIN_AND_LOSS": "A6k, the back-off form: the structural columns split by their reading (relation names on the "
                         "typed graphs, title windows on the others; eight columns), so the KB graphs' words, which D4 "
                         "found lead away from the gold, cannot move the passage graphs' weights; on both models, its "
                         "own declared file",
        "NO_GAIN": "A6e: on zspk, each pool edge's own link score (between its two ends) as a weight in zsp's "
                   "propagation, a message along the worded link; on zrckp, the anchors widened to the first stage's "
                   "top ten rows; one declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A6's screen verdict, per arm, from its re-call (section 4); musique, 2wiki and hotpotqa beside (D4's reads),
    musique's in-domain read of both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A6" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A6's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "next_form": next_form(v, gains, losses),
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A6", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za6lw_sha256": LC.sha_src(__file__), "utc": utc()}
    md = [f"# A6's screen verdict: zrckpw **{per['zrckpw']['verdict']}**, zspkw **{per['zspkw']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {b['call']} ({num(b['delta'])})" for k, b in p["beside"].items()) or "none")
                  + f". Next: {p['next_form']}.")
    md += ["", f"Beside, deciding nothing: musique in-domain ({ck}): "
           + ", ".join(f"{a} {side[a]['call']} ({num(side[a]['delta'])})" for a in ARM_NAMES) + ".",
           "", "This verdict closes only this form (D4's link score to the first stage's top five rows, six columns "
               "through R1's pointwise head); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za6lw verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


class FakeEnt:
    def __init__(self, text, label, s, e):
        self.text, self.label_, self.start_char, self.end_char = text, label, s, e


class FakeNLP:
    """spaCy's pipe on the toy texts: the entities are the capitalised word pairs, a PERSON when the pair starts with
    'Ann', an ORG otherwise; 'Cardinal' words come out as CARDINAL (kept out by the label rule)."""

    def pipe(self, texts, batch_size=256):
        import re
        for t in texts:
            ents = []
            for m in re.finditer(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b", t):
                lab = "CARDINAL" if m.group(1) == "Cardinal" else "PERSON" if m.group(1) == "Ann" else "ORG"
                ents.append(FakeEnt(m.group(0), lab, m.start(), m.end()))

            class D:
                pass
            d = D()
            d.ents = ents
            yield d


class ToySrc:
    """The served interface on toy nodes: titles and texts, mentions as built, questions, KB names when given."""

    def __init__(self, nodes, mentions, questions, kbn=None):
        self.nodes, self.ments, self.qs, self.kbn = nodes, mentions, questions, kbn
        self.read = set()

    def questions(self, ds):
        return dict(self.qs)

    def corpus(self, ds, need, qset):
        df = dict.fromkeys(qset, 0)
        for t in self.nodes.values():
            for w in {x.lower()[:LW.STEM] for x in LW.TOKEN.findall(f"{t[0]} {t[1]}")} & qset:
                df[w] += 1
        N = len(self.nodes)
        self.read = set(need)
        return ({i: self.nodes[i] for i in need}, {w: math.log((N + 1) / (df[w] + 1)) for w in qset},
                {"nodes": N, "nodes_sha256": "toy", "need": len(need)})

    def mentions(self, ds, nodes_sha256):
        return (lambda i: list(self.ments.get(i, ()))), "toy"

    def kb_names(self, ds):
        return self.kbn


def toy_corpus(rng, N=40):
    """Toy nodes: two-word titles (the capitalised pairs FakeNLP calls entities), texts that mention other nodes'
    titles and shared entities among filler words and the question words."""
    words = ["alpha", "bravo", "delta", "echo", "golf", "hotel", "india", "kilo", "lima", "mike", "oscar", "papa",
             "romeo", "sierra", "tango", "victor"]
    names = [f"{a} {b}" for a in ("Ann", "Bob", "Cid", "Dee", "Eve") for b in ("Kay", "Lux", "Moe", "Nan", "Ola",
                                                                               "Pim", "Qua", "Rio")]
    titles = [names[i % len(names)] for i in range(N)]
    nodes = {}
    for i in range(N):
        parts = []
        for _ in range(int(rng.integers(4, 9))):
            parts += list(rng.choice(words, int(rng.integers(2, 6))))
            r = rng.random()
            if r < 0.4:
                parts.append(titles[int(rng.integers(N))])
            elif r < 0.6:
                parts.append(names[int(rng.integers(8))])
            elif r < 0.7:
                parts.append("Cardinal Two")
        nodes[i] = (titles[i], " ".join(parts))
    return nodes


def toy_mentions(nodes, nlp):
    """The toy mentions as merge keeps them (df within [2, 25]), (key id, start, end) per node."""
    rows = mention_rows(nlp, [nodes[i][1] for i in range(len(nodes))], 0)
    arr, keys = LW.aggregate(rows, len(nodes))
    keep = (arr["df"][arr["key"]] >= LW.DF_MIN) & (arr["df"][arr["key"]] <= LW.DF_MAX)
    out = {}
    for nd, k, s0, e0 in zip(arr["node"][keep], arr["key"][keep], arr["start"][keep], arr["end"][keep]):
        out.setdefault(int(nd), []).append((int(k), int(s0), int(e0)))
    return out, arr, keys


def cols_by_hand(n, rrf, eu, ev, ef, er, pool, q, idf, docs, kbn):
    """The six columns by brute force over every (row, anchor, family): float64 (n, NC)."""
    ret = [i for i in range(n) if rrf[i] > 0]
    A = sorted(ret, key=lambda i: (-rrf[i], i))[:N_ANCHOR]
    out = np.zeros((n, NC))
    qm = sum(idf[w] for w in q)
    for c in range(n):
        for f in (0, 1):
            best, has = 0.0, False
            for a in A:
                if a == c:
                    continue
                E = [k for k in range(eu.size) if ef[k] == f and {int(eu[k]), int(ev[k])} == {a, c}]
                if not E:
                    continue
                da, dc = docs[int(pool[a])], docs[int(pool[c])]
                if f == 0 and kbn is not None:
                    wins = [kbn[r] for k in E for r in er[k] if r >= 0]
                elif f == 0:
                    wins = LW.title_windows(da, dc) + LW.title_windows(dc, da)
                else:
                    wins = LW.ner_windows(da, dc)
                has = has or bool(wins)
                qq = q - da.title_stems - dc.title_stems
                for w in wins:
                    best = max(best, sum(idf[x] for x in qq.intersection(w)))
            out[c, f], out[c, 2 + f], out[c, 4 + f] = math.log1p(best), best / qm if qm > 0 else 0.0, float(has)
    return out


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A6", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's and zspk's models plus the head, on their carves with the columns beside; the bases unchanged
    assert S.ARMS["zrckpw"] == (ZRCKPW, A6KTChainCarveZRC) and S.ARMS["zspkw"] == (ZSPKW, A6KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZRCKPW.__mro__[1] is Z3P.ZRCKP and ZSPKW.__mro__[1:5] == (ZP.ZProp, A6Mix, Z3.KTMix, ZM.ZRM)
    assert A6KTChainCarveZRC.__mro__[1] is Z3.KTChainCarveZRC and A6KTLinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert N_A6 == 129 and ZL.FAMS == 3
    tmp = Path(tempfile.mkdtemp(prefix="za6_"))
    try:
        rng = np.random.default_rng(61)
        nlp = FakeNLP()
        nodes = toy_corpus(rng)
        N = len(nodes)
        # 2. the mentions in parts, merged, are a one-shot build's (linkwords.aggregate) array for array; the merge
        #    checks the nodes' sha256, keeps the file only when it rebuilds the graph's NER edges, and D4's loader reads it
        served = tmp / "served"
        (served / "toyds" / "graph").mkdir(parents=True)
        lines = b"".join(json.dumps({"title": nodes[i][0], "text": nodes[i][1]}).encode() + b"\n" for i in range(N))
        (served / "toyds" / "nodes.jsonl").write_bytes(lines)
        write_json(served / "toyds" / "DATASET.json", {"nodes": {"n": N, "sha256": hashlib.sha256(lines).hexdigest()}})
        rows = mention_rows(nlp, [nodes[i][1] for i in range(N)], 0)
        one, one_keys = LW.aggregate(rows, N)
        rc, rw = LW.rebuild_ner(one["key"].astype(np.int64), one["node"].astype(np.int64),
                                one["df"].astype(np.int64), N)
        assert rc.size > 5
        np.savez(served / "toyds" / "graph" / "ner.npz", src=(rc // N).astype(np.int32), dst=(rc % N).astype(np.int32),
                 weight=rw.astype(np.float32))
        for k in range(3):
            ents_part("toyds", k, 3, served, nlp, out_root=tmp / "parts")
        rec = merge("toyds", served, 3, parts_root=tmp / "parts", out_root=tmp / "ents", d4_root=tmp / "none")
        z = np.load(tmp / "ents" / "toyds_ents.npz")
        assert rec["arrays_sha256"] == LW.arrays_sha(one, LW.ENT_ARRAYS) and rec["graph"]["reproduced"] == 1.0
        assert all(np.array_equal(z[k], one[k]) for k in LW.ENT_ARRAYS) and rec["gate"]["passed"]
        assert bytes(z["keys_blob"]).decode("utf-8").split("\n") == one_keys and rec["d4"] is None
        ments, mrec = LW.load_ents("toyds", rec["nodes_sha256"], tmp / "ents")
        assert mrec["arrays_sha256"] == rec["arrays_sha256"] and any(ments(i) for i in range(N))
        assert 0 < rec["keys_in_df_range"] < rec["keys"] and "CARDINAL" not in str(rec["settings"]["labels"])
        # a part of another split, a changed part, a node file that is not the manifest's, a graph not rebuilt: refused
        SR.must_stop(merge, "toyds", served, 2, tmp / "parts", tmp / "ents", tmp / "none")
        pj = tmp / "parts" / "toyds" / "part_1of3.json"
        keep_b = pj.read_bytes()
        h = json.loads(keep_b)
        h["arrays_sha256"] = "x"
        write_json(pj, h)
        SR.must_stop(merge, "toyds", served, 3, tmp / "parts", tmp / "ents", tmp / "none")
        pj.write_bytes(keep_b)
        (served / "toyds" / "nodes.jsonl").write_bytes(lines + b"{}\n")
        SR.must_stop(merge, "toyds", served, 3, tmp / "parts", tmp / "ents2", tmp / "none")
        (served / "toyds" / "nodes.jsonl").write_bytes(lines)
        np.savez(served / "toyds" / "graph" / "ner.npz", src=np.arange(30, dtype=np.int32) % N,
                 dst=(np.arange(30, dtype=np.int32) * 7 + 1) % N, weight=np.ones(30, np.float32))
        SR.must_stop(merge, "toyds", served, 3, tmp / "parts", tmp / "ents2", tmp / "none")
        assert not (tmp / "ents2" / "toyds_ents.npz").exists() and (tmp / "ents2" / "toyds_ents.gate_failed.json").exists()
        # 3. the columns: rmatch's typed toy look with pools and an rrf column added; each question's columns against
        #    the brute force, on the passage reading (title windows) and the KB reading (relation names); the anchors
        #    are the first stage's top five; a row with no anchor link is 0
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        look = tmp / "look" / "metaqa" / "toy"
        rr = json.loads((look / "record_0of1.json").read_text(encoding="utf-8"))
        assert rr["columns"] == ["rrf"]
        ids = json.loads((look / "ids.json").read_text(encoding="utf-8"))
        for f in sorted((look / "chunks").glob("c*.npz")):
            with np.load(f) as zf:
                zc = {k_: zf[k_] for k_ in zf.files}
            tot = int(zc["q_pool_size"].sum())
            pool = np.concatenate([rng.choice(N, int(n), replace=False) for n in zc["q_pool_size"]]).astype(np.int32)
            rrf = np.where(rng.random(tot) < 0.75, np.round(rng.random(tot), 1) * 0.03 + 0.001, 0.0)
            np.savez(f, pool=pool, x=rrf.astype(np.float32).reshape(-1, 1), **zc)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        ment_t, _, _ = toy_mentions(nodes, nlp)
        qwords = ["alpha bravo", "Who led delta echo golf", "Where is hotel Kay", "kilo lima mike Lux", "oscar papa",
                  "romeo sierra tango victor Moe"]
        qtext = {x: qwords[i % len(qwords)] for i, x in enumerate(ids)}
        kbn = [LW.stems_of([w]) for w in ("alpha", "golf", "kilo", "romeo", "tango")]
        for kb in (None, kbn):
            src = ToySrc(nodes, ment_t, qtext, kb)
            out_r = tmp / ("a6kb" if kb else "a6")
            build("metaqa", "toy", src, out_root=out_r, cache_root=tmp / "cache", look_root=tmp / "look")
            read_build, need_ref = set(src.read), set()
            _d, files, _ids, head = LC.look_chunks("metaqa", "toy", tmp / "look")
            got = np.concatenate([np.load(out_r / "metaqa" / "toy" / f"part_{p}of2" / "cols.npy") for p in (0, 1)])
            _t, idf, _i = src.corpus("metaqa", set(range(N)), frozenset().union(*(LW.question_stems(t) for t in
                                                                                 qtext.values())))
            docs = {i: LW.Doc(nodes[i][0], nodes[i][1], ment_t.get(i, ())) for i in range(N)}
            refs, at, qi, n_word, n_none = [], 0, 0, 0, 0
            for f in files:
                zc = np.load(f)
                no, eo = np.r_[0, np.cumsum(zc["q_pool_size"])], np.r_[0, np.cumsum(zc["q_edges"])]
                for i in range(zc["q_pool_size"].size):
                    n = int(zc["q_pool_size"][i])
                    ref = cols_by_hand(n, zc["x"][no[i]:no[i + 1], 0].astype(np.float64), zc["e_u"][eo[i]:eo[i + 1]],
                                       zc["e_v"][eo[i]:eo[i + 1]], zc["e_fam"][eo[i]:eo[i + 1]],
                                       zc["e_rel"][eo[i]:eo[i + 1]], zc["pool"][no[i]:no[i + 1]],
                                       LW.question_stems(qtext[ids[qi]]), idf, docs, kb)
                    refs.append(ref)
                    pl = zc["pool"][no[i]:no[i + 1]].astype(np.int64)
                    A_ = anchors_of(zc["x"][no[i]:no[i + 1], 0].astype(np.float64))
                    eu_, ev_ = zc["e_u"][eo[i]:eo[i + 1]].astype(np.int64), zc["e_v"][eo[i]:eo[i + 1]].astype(np.int64)
                    hit = [k_ for k_ in range(eu_.size) if eu_[k_] != ev_[k_] and zc["e_fam"][eo[i] + k_] < 2
                           and (eu_[k_] in A_ or ev_[k_] in A_)]
                    need_ref |= set(pl[A_].tolist()) | {int(pl[eu_[k_]]) for k_ in hit} | {int(pl[ev_[k_]]) for k_ in hit}
                    n_word += int((ref[:, :2] > 0).any(1).sum())
                    n_none += int((ref[:, 4:] == 0).all(1).sum())
                    qi += 1
            ref = np.concatenate(refs)
            assert got.shape == ref.shape == (int(sum(np.load(f)["q_pool_size"].sum() for f in files)), NC)
            assert np.abs(got.astype(np.float64) - ref).max() < 4e-3, np.abs(got.astype(np.float64) - ref).max()
            assert n_word > 3 and n_none > 3, (kb is not None, n_word, n_none)
            assert read_build == need_ref, (len(read_build), len(need_ref))
            rec0 = json.loads((out_r / "metaqa" / "toy" / "part_0of2" / "record.json").read_text(encoding="utf-8"))
            assert rec0["stage"] == "A6" and rec0["kb_names"] == (kb is not None) and rec0["max_abs_float16_error"] < 4e-3
        # one part at a time is the whole build's arrays byte for byte
        for pt in (1, 0):
            build("metaqa", "toy", ToySrc(nodes, ment_t, qtext), pt, out_root=tmp / "a6p", cache_root=tmp / "cache",
                  look_root=tmp / "look")
        for pt in (0, 1):
            fa, fb = (r_ / "metaqa" / "toy" / f"part_{pt}of2" / "cols.npy" for r_ in (tmp / "a6", tmp / "a6p"))
            assert fa.read_bytes() == fb.read_bytes(), pt
        # the KB reading reads the relation names: its structural columns differ from the passage reading's
        a = np.load(tmp / "a6" / "metaqa" / "toy" / "part_0of2" / "cols.npy")
        b = np.load(tmp / "a6kb" / "metaqa" / "toy" / "part_0of2" / "cols.npy")
        assert not np.array_equal(a[:, [0, 2, 4]], b[:, [0, 2, 4]]) and np.array_equal(a[:, [1, 3, 5]], b[:, [1, 3, 5]])
        # the anchors: top five by rrf among rrf > 0, ties by row; none when nothing is retrieved
        assert anchors_of(np.array([0.0, 0.2, 0.5, 0.2, 0.0, 0.1, 0.3, 0.05])).tolist() == [2, 6, 1, 3, 5]
        assert anchors_of(np.zeros(4)).size == 0
        # the build refuses a missing question, a part from another look, and a relation slot past the names
        SR.must_stop(build, "metaqa", "toy", ToySrc(nodes, ment_t, dict(list(qtext.items())[1:])), None, tmp / "x",
                     tmp / "cache", tmp / "look")
        SR.must_stop(build, "metaqa", "toy", ToySrc(nodes, ment_t, qtext, kbn[:2]), None, tmp / "x", tmp / "cache",
                     tmp / "look")
        p0 = tmp / "cache" / "metaqa" / "toy" / "part_0of2" / "record.json"
        keep_b = p0.read_bytes()
        h = json.loads(keep_b)
        h["look"]["carve_ids_sha256"] = "x"
        R.write_json(p0, h)
        SR.must_stop(build, "metaqa", "toy", ToySrc(nodes, ment_t, qtext), None, tmp / "x", tmp / "cache", tmp / "look")
        SR.must_stop(build, "metaqa", "toy", ToySrc(nodes, ment_t, qtext), 2, tmp / "x", tmp / "cache", tmp / "look")
        p0.write_bytes(keep_b)
        # 4. a batch's columns are its rows' own, on both carves; the MLP's carve holds no edge
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        c_m = a6_carve(cls_ch, tmp / "a6")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_g = a6_carve(cls_lk, tmp / "a6")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:7], np.arange(c_m.rows)[:5]].astype(np.int64)
        B = qs.size
        fm, nq, bz, _gold = c_m.batch(qs, blocks)
        fg, nq_g, bz_g, _ = c_g.batch(qs, blocks)
        assert ZL.LK_KEY not in fm and ZL.LK_KEY in fg and fm[A6_KEY].dtype == torch.float32
        assert torch.equal(fm[A6_KEY], fg[A6_KEY]) and torch.equal(nq, nq_g)
        allc = np.concatenate([np.load(tmp / "a6" / "metaqa" / "toy" / f"part_{p}of2" / "cols.npy") for p in (0, 1)])
        want = np.concatenate([allc[c_m.off_np[q]:c_m.off_np[q + 1]] for q in qs]).astype(np.float32)
        assert np.array_equal(fm[A6_KEY].numpy(), want) and float(fm[A6_KEY].abs().max()) > 0.05
        # 5. the models: each arm starts as its base bit for bit; with the head moved, zrckpw adds the head over the
        #    columns and zspkw's propagation reads the score with the head added; without the columns each is its base
        keep_ = torch.ones(B, len(blocks))
        for arm, cls, base, cv in (("zrckpw", ZRCKPW, Z3P.ZRCKP, c_m), ("zspkw", ZSPKW, Z3.ZSPK, c_g)):
            f_, nq_, bz_, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(A6) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k: s1[k] for k in s0})
            assert sum(s1[k].numel() for k in A6) == N_A6
            m0.eval()
            m1.eval()
            with torch.no_grad():
                if arm == "zspkw":
                    w = torch.randn(1, ZP.HID, generator=torch.Generator().manual_seed(8))
                    m0.sp_w2.copy_(w)
                    m1.sp_w2.copy_(w)
                y0 = m0(f_, keep_, nq_, B, bz_)
                assert torch.equal(y0, m1(f_, keep_, nq_, B, bz_))
                m1.a6_w2.copy_(torch.randn(1, A6_HID, generator=torch.Generator().manual_seed(9)))
                y1 = m1(f_, keep_, nq_, B, bz_)
                head = m1.a6_head(f_[A6_KEY])
                assert float(head.abs().max()) > 1e-3
                if arm == "zrckpw":
                    assert torch.allclose(y1 - y0, head, atol=1e-5)
                else:
                    pre = super(ZP.ZProp, m1).forward(f_, keep_, nq_, B, bz_)
                    kt_only = Z3.KTMix.forward(m1, f_, keep_, nq_, B, bz_)
                    assert torch.allclose(pre - kt_only, head, atol=1e-5)
                    assert torch.allclose(y1, pre + m1.prop_head(ZP.prop_inputs(pre, f_[ZL.LK_KEY], nq_, B)),
                                          atol=1e-5)
                    assert float((y1 - y0 - head).abs().max()) > 1e-4
                bare = {k: x for k, x in f_.items() if k != A6_KEY}
                assert torch.equal(m1(bare, keep_, nq_, B, bz_), m0(bare, keep_, nq_, B, bz_))
        # 6. toy fits through lean_gpu's loop: a repeat is bit-identical and the head learns; on the carve without the
        #    columns each arm's fit is its base's bit for bit on every shared key, its head at the start
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv, bare_c in (("zrckpw", c_m, cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")),
                                ("zspkw", c_g, cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"))):
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                bare = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, no columns")
            with S.patched(BASE_OF[arm]):
                base = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            assert all(not LG.same_state(a_, b_) for a_, b_ in zip(fit["states"] + [fit["swa"]],
                                                                   rep["states"] + [rep["swa"]]))
            assert float(fit["swa"]["a6_w2"].abs().max()) > 0
            for a_, b_ in zip(bare["states"] + [bare["swa"]], base["states"] + [base["swa"]]):
                assert not LG.same_state({k: a_[k] for k in b_}, b_), arm
                assert float(a_["a6_w2"].abs().max()) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. the refusals
    good = ["train", "--arm", "zrckpw", "--name", FIT["zrckpw"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpw"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkw", "--name", FIT["zrckpw"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkw", "--name", FIT["zspkw"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    assert split_of_fit("scr-a6-zspkw-hp") == ("zspkw", "L-hotpotqa") and split_of_fit("scr-a5-zspkk") == (None, None)
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a6-zspkw", {"arm": "zspkr", "r1": {}}),
                        ("scr-a6-zspkw-hp", {"arm": "zspkw", "a6": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 8. relz's pair and re-call for each arm against its base arm's card fits, stamped A6's; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T_ / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T_ / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T_ / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T_ / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "A6" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A6 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A6's link words) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 9. the verdict, per arm, with the next form of section 5
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpw": fake_recall(T_ / "a.json", "zrckpw", {mi: "GAIN"}),
                       "zspkw": fake_recall(T_ / "b.json", "zspkw", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpw"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckpw"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspkw"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspkw"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckpw": {"call": "GAIN", "delta": 0.0},
                                                   "zspkw": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkw"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpw": fake_recall(T_ / "a.json", "zrckpw", {}),
                       "zspkw": fake_recall(T_ / "b.json", "zspkw", {gm: "GAIN"}, stage="R1")}, T_ / "v")
        assert rec["arms"]["zrckpw"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkw"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckpw"]["next_form"] == NEXT["NO_GAIN"]
        rec = verdict({"zrckpw": T_ / "none.json", "zspkw": fake_recall(T_ / "b.json", "zrckpw", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpw"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkw"]["verdict"] == "INCOMPLETE"
    log(f"za6lw selftest: the mentions in parts merge to a one-shot build's arrays, D4's loader reads them, and the "
        f"merge refuses another split, a changed part, other nodes and a graph it does not rebuild; the columns match "
        f"the brute force on the passage and KB readings, the KB reading reads relation names, the anchors are the "
        f"first stage's top five; the build refuses a missing question, another look and a slot past the names; a "
        f"batch's columns are its rows' own; each arm starts as its base bit for bit, zspkw's propagation reads the "
        f"head, and without the columns each is its base; toy fits repeat, the head learns, and without the columns "
        f"each fit is its base's; the refusals; relz's pair and re-call per arm, stamped A6's; the verdict's outcomes "
        f"and next forms ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd in ("spacy", "ents", "merge", "build"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--dataset")
        gp.add_argument("--part")
        gp.add_argument("--carve")
        gp.add_argument("--host", action="store_true")
        gp.add_argument("--cache-root")
        gp.add_argument("--look-root")
        gp.add_argument("--out-root")
        g = gp.parse_args(rest)
        where = "host" if g.host else "laptop"
        if k.cmd == "spacy":
            nlp, path = load_nlp(g.host)
            ents = [(e.text, e.label_) for e in nlp("Ann Lux met Bob Kay of the Acme Corporation in Paris.").ents]
            log(f"za6lw spacy: {LW.SPACY_VERSION} with {LW.MODEL_VERSION} from {path or 'the installed packages'}; "
                f"entities {ents}")
            return 0
        if g.dataset not in ENT_PARTS:
            gp.error(f"--dataset: one of {', '.join(ENT_PARTS)}")
        import u1b as U1B
        if k.cmd == "ents":
            kk, n = (int(x) for x in g.part.split("/"))
            if n != ENT_PARTS[g.dataset]:
                gp.error(f"{g.dataset} runs in {ENT_PARTS[g.dataset]} parts, not {n}")
            served, _f = U1B.package_root(g.host)
            nlp, path = load_nlp(g.host)
            ents_part(g.dataset, kk, n, served, nlp, where=where, spacy_from="host zip" if path else "laptop")
            return 0
        if k.cmd == "merge":
            served, freeze = U1B.package_root(g.host)
            merge(g.dataset, served, ENT_PARTS[g.dataset], freeze=freeze, where=where)
            return 0
        if not g.carve:
            gp.error("build: --carve")
        build(g.dataset, g.carve, Served(g.host), part=None if g.part is None else int(g.part), out_root=Path(g.out_root) if g.out_root else COLS_DIR,
              cache_root=Path(g.cache_root) if g.cache_root else None,
              look_root=Path(g.look_root) if g.look_root else None, placement={"where": where})
        return 0
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        return 0 if all(v["verdict"] != "INCOMPLETE" for v in verdict()["arms"].values()) else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"za6lw {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za6lw: spacy, ents, merge, build, train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
