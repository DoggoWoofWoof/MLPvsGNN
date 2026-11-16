"""UL1: one rule-built NER store on the six training corpora, graded against the oracle pairs before any refit
(docs/UL1_UNIFIED_LINKS_ORACLE_CHECK.md).

    python outputs/mp_unified/zul1.py selftest
    python outputs/mp_unified/zul1.py build --dataset D [--host]   # F0T's title pairs; F0 checked against U2r's CN_all (G2)
    python outputs/mp_unified/zul1.py grade --dataset D [--host]   # GP (and MuSiQue's CH) reach of each union, per carve
    python outputs/mp_unified/zul1.py report                       # the rule of section 6 over the six grades

Graph level only: nothing trains, no model reads anything, no filed graph, pool, fit or record changes. The forms in R's NER slot:
none, N_sp (the package's spaCy store), CN_all (U2r's rule names, df [2, 25]) and F0T (CN_all's names plus each read passage's own
title runs, AP1b's level-0 title form). Labels (gold positions, MuSiQue's decompositions) are the grading only; no rule reads one.
Outputs only under outputs/ul1/. Arrays (npz) stay uncommitted.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from array import array  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import oraclepairs as O  # noqa: E402
import u1b  # noqa: E402
import zap1 as Z  # noqa: E402
import zap1b as AP  # noqa: E402
import zu2b  # noqa: E402
import zu2i  # noqa: E402
import zu2l_el as EL  # noqa: E402
import zu2r as U  # noqa: E402

DECLARED = "docs/UL1_UNIFIED_LINKS_ORACLE_CHECK.md"
OUT = ROOT / "outputs" / "ul1"
U2B = ROOT / "outputs" / "u2b"
U2R = ROOT / "outputs" / "u2r"
ABLU = ROOT / "outputs" / "abl" / "u"            # ABL-U: R's rule part on the KBs (layout section 1)
CARVES_JSON = ROOT / "outputs" / "step1" / "carves.json"
SIX = ("musique", "squad", "hotpotqa", "2wiki", "metaqa", "webqsp")
PASSAGE = ("musique", "squad", "hotpotqa", "2wiki")
KB = ("metaqa", "webqsp")
CARVES = {d: (("s1eval",) if d == "webqsp" else ("s1sel", "s1eval")) for d in SIX}
CARVE_KEY = {"s1sel": "dselect", "s1eval": "eval"}
DOMAINS = {"2wiki": ("RK", "SRK"), "hotpotqa": ("RK", "SRK"), "musique": ("RK",), "squad": ("RK",), "metaqa": ("SRK",), "webqsp": ("SRK",)}   # A1
HAS_S = ("2wiki", "hotpotqa", "metaqa", "webqsp")   # MuSiQue's and SQuAD's title mentions are not S since A1, and are not read
FORMS = ("none", "N_sp", "CN_all", "F0T")
TRY = ("F0T", "CN_all")                    # section 6's order
BOUND = 1.30                               # a form's union at most 1.30 times N_sp's union
KB_GOLDS = (2, 10)                         # a KB question's golds are its answer set: GP read for 2 to 10 golds only (beside)
LO, CAP = Z.LO, Z.CAP                      # U2r's df cap [2, 25]
BOOT, SEED = Z.BOOT, Z.SEED


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def codes():
    return {"script_sha256": script_sha256(), "zap1_sha256": Z.script_sha256(), "zap1b_sha256": AP.script_sha256(),
            "zu2r_sha256": U.script_sha256(), "zu2l_el_sha256": EL.script_sha256()}


# ── the build: F0 (U2r's CN_all, rebuilt) and the level-0 title pairs ────────


def build_entries(passes, n):
    """Two streamed passes: U2r's capitalization statistic, then each read passage's names (U2r's names_of) and the runs of its own stripped title
    that are not already among its names (zap1b's runs_of on title_stripped). Returns the (passage, key) entries of the names and of the added title
    runs, over one vocabulary."""
    cap, low = {}, {}
    for _t, x in passes():
        if x is not None:
            U.count_stat(U.tokens(x), cap, low)

    def usual(k):
        return not (low.get(k, 0) > cap.get(k, 0))

    vocab, src, grp, tsrc, tgrp = {}, array("i"), array("i"), array("i"), array("i")
    seen = 0
    for i, (t, x) in enumerate(passes()):
        seen += 1
        if x is None:
            continue
        names = U.names_of(U.tokens(x), usual)
        for k in names:
            src.append(i)
            grp.append(vocab.setdefault(k, len(vocab)))
        for k in set(AP.runs_of(U.tokens(AP.title_stripped(t)), usual)) - names:
            tsrc.append(i)
            tgrp.append(vocab.setdefault(k, len(vocab)))
    if seen != n:
        raise SystemExit(f"the second pass read {seen} nodes, expected {n}")
    nv = len(vocab)
    del vocab
    as64 = lambda a: np.asarray(a).astype(np.int64)  # noqa: E731
    return as64(src), as64(grp), as64(tsrc), as64(tgrp), nv


def forms_from_entries(src, grp, tsrc, tgrp, n):
    """F0 = co-mention pairs of the names; LT = co-mention pairs of names plus title runs (zap1b's level (0, True)); F0T = F0 | LT, as pair keys."""
    f0 = EL.co_mentions(src, grp, n, lo=LO, hi=CAP)
    lt = EL.co_mentions(np.concatenate([src, tsrc]), np.concatenate([grp, tgrp]), n, lo=LO, hi=CAP)
    return f0, lt


def names_record(d):
    r = json.loads((U2R / d / "names.json").read_text(encoding="utf-8"))
    if Z.sha_file(U2R / d / "names.npz") != r["names_npz_sha256"]:
        raise SystemExit(f"gate G1: {d}: U2r's names.npz is not its record's")
    return r


def cmd_build(a):
    d, t0 = a.dataset, time.time()
    served, freeze = u1b.package_root(a.host)
    passes, n, nsha, n_cvt = U.corpus(served, d)
    nr = names_record(d)
    if nr["nodes"] != n or nr["nodes_sha256"] != nsha or nr["freeze_RECORD_SHA256"] != freeze:
        raise SystemExit(f"gate G1: {d}: U2r's names were built on other nodes or another freeze")
    src, grp, tsrc, tgrp, nv = build_entries(passes, n)
    t1 = time.time()
    log(f"{d}: {src.size:,} name entries, {tsrc.size:,} title-run entries added, {nv:,} keys ({t1 - t0:.0f}s)")
    (fa, fb, fw), (la, lb, lw) = forms_from_entries(src, grp, tsrc, tgrp, n)
    del src, grp, tsrc, tgrp
    with np.load(U2R / d / "names.npz") as z:                                             # G2: F0 is U2r's CN_all, pairs and weights
        g2 = (np.array_equal(fa.astype(np.int32), z["cn_all_src"]) and np.array_equal(fb.astype(np.int32), z["cn_all_dst"])
              and np.array_equal(fw.astype(np.float32), z["cn_all_w"]))
    if not g2:
        raise SystemExit(f"gate G2: {d}: the rebuilt F0 ({fa.size:,} pairs) is not U2r's CN_all ({nr['cn_all_pairs']:,})")
    f0t = np.union1d(fa * n + fb, la * n + lb)
    od = OUT / d
    od.mkdir(parents=True, exist_ok=True)
    sha = zu2b.save_npz(od / "f0t.npz", lt_src=la.astype(np.int32), lt_dst=lb.astype(np.int32), lt_w=lw.astype(np.float32))
    rec = {"dataset": d, "declared_in": DECLARED, "freeze_RECORD_SHA256": freeze, "nodes": int(n), "nodes_sha256": nsha, "cvt_mediators": int(n_cvt),
           "keys": int(nv), "f0_pairs": int(fa.size), "lt_pairs": int(la.size), "f0t_pairs": int(f0t.size), "f0t_over_f0": round(f0t.size / max(fa.size, 1), 4),
           "gate_G2": "PASS", "names_npz_sha256": nr["names_npz_sha256"], "f0t_npz_sha256": sha,
           "seconds": round(time.time() - t0, 1), **codes()}
    EL.write_json(od / "build.json", rec)
    log(f"{d}: F0 {fa.size:,} pairs (= U2r's CN_all), F0T {f0t.size:,} pairs; written {od / 'build.json'} ({rec['seconds']}s)")
    return 0


# ── pair keys and membership ────────────────────────────────────────────────


def pair_keys(a, b, n):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    m = a != b
    a, b = a[m], b[m]
    return np.unique(np.minimum(a, b) * n + np.maximum(a, b))


def csr_keys(indptr, col, n, rows_per=1 << 18):
    """The pairs of an undirected CSR (both directions stored), one key per pair, built block by block (WebQSP's rule graph holds about
    10^9 entries). Stops unless the CSR is undirected in count (as many entries above the diagonal as below, none on it)."""
    indptr = np.asarray(indptr, np.int64)
    parts, up, down, last, ordered = [], 0, 0, -1, True
    for r0 in range(0, n, rows_per):
        r1 = min(n, r0 + rows_per)
        c = np.asarray(col[indptr[r0]:indptr[r1]], np.int64)
        rows = np.repeat(np.arange(r0, r1, dtype=np.int64), np.diff(indptr[r0:r1 + 1]))
        m = c > rows
        up += int(m.sum())
        down += int((c < rows).sum())
        k = rows[m] * n + c[m]
        del c, rows, m
        if k.size:
            ordered = ordered and k[0] > last and bool((k[1:] > k[:-1]).all())
            last = int(k[-1])
        parts.append(k)
    if up != down or up + down != int(indptr[-1]):
        raise SystemExit(f"gate G1: a CSR read as undirected has {up} entries above the diagonal, {down} below, {int(indptr[-1])} in all")
    k = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    del parts
    return k if ordered else np.unique(k)


def member(sk, keys):
    """Which of sk are in the sorted unique keys."""
    sk = np.asarray(sk, np.int64)
    if keys.size == 0 or sk.size == 0:
        return np.zeros(sk.size, bool)
    i = np.searchsorted(keys, sk)
    i[i == keys.size] = keys.size - 1
    return keys[i] == sk


def union_size(base, form):
    """|base | form| for sorted unique key arrays, without building the union."""
    return int(base.size + form.size - member(form, base).sum())


# ── the families, the questions, the slots ──────────────────────────────────


def rule_part(d, n, nsha, freeze):
    """R's rule part (W2_S): U2b's on the passage corpora, ABL-U's on the KBs, each checked against its build record. Returns (path, sha256)."""
    if d in PASSAGE:
        p, e = zu2i.ref_graph(d)
        if p is None:
            raise SystemExit(f"gate G1: {d}: U2b's W2_S is not in outputs/u2i/{d}/ref/")
        return p, e["npz_sha256"]
    b = json.loads((ABLU / d / "build.json").read_text(encoding="utf-8"))
    p = ABLU / d / "W2_S.npz"
    if b["nodes"] != n or b["nodes_sha256"] != nsha or b["freeze_record_sha256"] != freeze or not p.exists() or Z.sha_file(p) != b["npz_sha256"]:
        raise SystemExit(f"gate G1: {d}: ABL-U's W2_S is missing or is not its build record's (nodes, freeze, sha256)")
    return p, b["npz_sha256"]


def families(served, d, n, nsha, freeze):
    """{family: sorted unique pair keys}: S (where A1 gives the dataset S), N_sp and K from the package (each checked against GRAPH_MANIFEST),
    W2_S (R's rule part), CN_all (U2r's) and F0T (CN_all | the build's title pairs)."""
    F, sh, slab = {}, {}, None
    for fam, key in (("ner", "N_sp"), ("knn", "K")):
        s_, t_, sh[f"{fam}_npz_sha256"] = EL.pkg_family(served, d, fam)
        F[key] = pair_keys(s_, t_, n)
        del s_, t_
    if d in HAS_S:
        slab, sh["structural_npz_sha256"] = s_labels(served, d, n)
        F["S"] = np.unique(slab[0])
    p, sh["w2_s_npz_sha256"] = rule_part(d, n, nsha, freeze)
    with np.load(p) as z:
        ip = z["indptr"]
        if ip.size != n + 1:
            raise SystemExit(f"gate G1: {d}: W2_S has {ip.size - 1} rows, the package {n}")
        F["W2_S"] = csr_keys(ip, z["col"], n)
    log(f"{d}: W2_S {F['W2_S'].size:,} pairs")
    names_record(d)
    with np.load(U2R / d / "names.npz") as z:
        F["CN_all"] = pair_keys(z["cn_all_src"], z["cn_all_dst"], n)
    b = json.loads((OUT / d / "build.json").read_text(encoding="utf-8"))
    if b["script_sha256"] != script_sha256() or Z.sha_file(OUT / d / "f0t.npz") != b["f0t_npz_sha256"] or b["nodes"] != n:
        raise SystemExit(f"gate G1: {d}: the build is not this code's, or f0t.npz is not its record's")
    with np.load(OUT / d / "f0t.npz") as z:
        F["F0T"] = np.union1d(F["CN_all"], pair_keys(z["lt_src"], z["lt_dst"], n))
    if F["F0T"].size != b["f0t_pairs"]:
        raise SystemExit(f"gate G1: {d}: F0T has {F['F0T'].size} pairs, the build {b['f0t_pairs']}")
    sh["f0t_npz_sha256"] = b["f0t_npz_sha256"]
    sh["names_npz_sha256"] = b["names_npz_sha256"]
    return F, sh, slab


def s_labels(served, d, n):
    """S's pairs with the store's own relation label (rel and the manifest's relation vocabulary): (keys ascending, rel, vocab), one row per distinct
    (pair, relation); a pair with two relations keeps both. Checked against GRAPH_MANIFEST's sha256."""
    f = json.loads((served / d / "graph" / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))["families"]["structural"]
    p = served / d / "graph" / Path(f["file"]).name
    if Z.sha_file(p) != f["npz_sha256"]:
        raise SystemExit(f"gate G1: {d}: {p.name} is not GRAPH_MANIFEST's")
    vocab = [str(v) for v in (f.get("relation_vocabulary") or [])]
    with np.load(p) as z:
        a, b, r = z["src"].astype(np.int64), z["dst"].astype(np.int64), z["rel"].astype(np.int64)
    m = a != b
    k = np.minimum(a[m], b[m]) * n + np.maximum(a[m], b[m])
    r = r[m]
    del a, b, m
    if r.size and (r.min() < 0 or r.max() >= len(vocab)):
        raise SystemExit(f"gate G1: {d}: a structural relation id is outside the manifest's vocabulary of {len(vocab)}")
    R = max(len(vocab), 1)
    kr = np.unique(k * R + r)
    return (kr // R, kr % R, vocab), f["npz_sha256"]


# ── edge labels: how each edge was made, one vocabulary with the KB relations ─


def label_sets(sk, F, slab, dom, form):
    """{label: bool per slot}: which creation labels the slot's pair carries in the union of `form` (several at once is allowed). S's labels are
    the store's relations (prefixed S.); R.mention = R's rule links (W2_S); R's NER store as R.ner_spacy, or R.name_comention / R.title_comention
    (F0T's pairs not in CN_all); K.knn."""
    L = {}
    if dom == "SRK" and slab is not None:
        keys, rel, vocab = slab
        lo, hi = np.searchsorted(keys, sk, "left"), np.searchsorted(keys, sk, "right")
        for j in np.flatnonzero(hi > lo):
            for x in rel[lo[j]:hi[j]].tolist():
                L.setdefault("S." + vocab[x], np.zeros(sk.size, bool))[j] = True
    L["R.mention"] = member(sk, F["W2_S"])
    if form == "N_sp":
        L["R.ner_spacy"] = member(sk, F["N_sp"])
    elif form in ("CN_all", "F0T"):
        cn = member(sk, F["CN_all"])
        L["R.name_comention"] = cn
        if form == "F0T":
            L["R.title_comention"] = member(sk, F["F0T"]) & ~cn
    L["K.knn"] = member(sk, F["K"])
    return L


def label_pairs(F, slab, dom, form):
    """Pairs of the union carrying each label (a pair is counted under every label it carries)."""
    out = {}
    if dom == "SRK" and slab is not None:
        keys, rel, vocab = slab
        for x, c in enumerate(np.bincount(rel, minlength=len(vocab)).tolist()):
            if c:
                out["S." + vocab[x]] = c
    out["R.mention"] = int(F["W2_S"].size)
    if form == "N_sp":
        out["R.ner_spacy"] = int(F["N_sp"].size)
    elif form in ("CN_all", "F0T"):
        out["R.name_comention"] = int(F["CN_all"].size)
        if form == "F0T":
            out["R.title_comention"] = int(F["F0T"].size - F["CN_all"].size)
    out["K.knn"] = int(F["K"].size)
    return out


def label_table(L, pairs, linked):
    """Per label: the linked slots carrying it, those carrying it alone, and slots per million of its pairs; sorted by slots carrying."""
    if not L:
        return []
    M = np.vstack([L[k] for k in L])
    alone = M.sum(0) == 1
    rows = []
    for i, k in enumerate(L):
        c = int((M[i] & linked).sum())
        rows.append({"label": k, "pairs": pairs.get(k, 0), "slots": c, "alone": int((M[i] & alone).sum()),
                     "per_million_pairs": round(1e6 * c / pairs[k], 2) if pairs.get(k) else None})
    return sorted(rows, key=lambda r: (-r["slots"], r["label"]))


def carve_questions(d, carve):
    """The carve's ids and gold positions (U2b's questions file, checked against its record and against carves.json's ids in order)."""
    qrec = json.loads((U2B / d / "questions.json").read_text(encoding="utf-8"))
    qp = U2B / d / f"questions_{carve}.npz"
    sha = Z.sha_file(qp)
    if sha != qrec["carves"][carve]["npz_sha256"]:
        raise SystemExit(f"gate G3: {d}/{carve}: the questions file is not its record's")
    ids = json.loads((U2B / d / f"questions_{carve}.json").read_text(encoding="utf-8"))["ids"]
    want = json.loads(CARVES_JSON.read_text(encoding="utf-8"))["per_dataset"][d][CARVE_KEY[carve]]["ids"]
    if [str(x) for x in ids] != [str(x) for x in want]:
        raise SystemExit(f"gate G3: {d}/{carve}: the questions' ids are not carves.json's, in order")
    with np.load(qp) as z:
        golds = [np.unique(g.astype(np.int64)) for g in zu2b.unragged(z["gold_ptr"], z["gold_col"])]
    if len(golds) != len(ids):
        raise SystemExit(f"gate G3: {d}/{carve}: {len(golds)} gold rows for {len(ids)} ids")
    return [str(x) for x in ids], golds, sha


def gp_slots(golds, n, lim=None):
    """GP: (question index, key) for every unordered pair of distinct golds; lim = (lo, hi) gold counts read (the KBs)."""
    qi, keys = [], []
    for j, g in enumerate(golds):
        if g.size and (g.min() < 0 or g.max() >= n):
            raise SystemExit(f"gate G3: question {j}: a gold position is out of range")
        if g.size < 2 or (lim and not (lim[0] <= g.size <= lim[1])):
            continue
        a, b = np.triu_indices(g.size, 1)
        keys.append(g[a] * n + g[b])
        qi.append(np.full(a.size, j, np.int64))
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(qi), np.concatenate(keys)


def field(rec, k):
    v = rec[k]
    return ast.literal_eval(v) if isinstance(v, str) and k != "hop" else v


def musique_queries(served, ids):
    """The package's query records for the carve's ids (any split file), by id."""
    need, out = set(ids), {}
    for f in sorted((served / "musique" / "queries").glob("*.jsonl")):
        with open(f, "rb") as h:
            for raw in h:
                r = json.loads(raw)
                q = str(r.get("query_id"))
                if q in need:
                    if q in out:
                        raise SystemExit(f"gate G3: musique: query {q} is in two query files")
                    out[q] = r
    miss = need - set(out)
    if miss:
        raise SystemExit(f"gate G3: musique: {len(miss)} carve ids have no query record (e.g. {sorted(miss)[:3]})")
    return out


def node_positions(served, d):
    pos = {}
    with open(served / d / "nodes.jsonl", "rb") as f:
        for i, raw in enumerate(f):
            pos[json.loads(raw)["node_id"]] = i
    return pos


def chain_slots(recs, ids, golds, posmap, n):
    """CH: oraclepairs' ORC rule on each carve question (hop j's support paragraph joined to that of each earlier hop its text names, '#k'); the hop
    nodes are context_node_ids[paragraph_support_idx]. G3: the hop nodes are the question's golds, and the hops form one connected chain."""
    qi, keys, hops = [], [], []
    for j, (qid, g) in enumerate(zip(ids, golds)):
        r = recs[qid]
        ctx, dec = field(r, "context_node_ids"), field(r, "question_decomposition")
        pos, deps = [], []
        for k, sub in enumerate(dec):
            pos.append(posmap[ctx[int(sub["paragraph_support_idx"])]])
            refs = sorted({int(m) - 1 for m in O.REF_RE.findall(sub["question"])})
            deps += [(k, x) for x in refs if 0 <= x < k]
        gl = {posmap[x] for x in field(r, "gold_node_ids")}
        if set(pos) != set(g.tolist()) or gl != set(g.tolist()):
            raise SystemExit(f"gate G3: musique {qid}: hop nodes {sorted(set(pos))}, gold_node_ids {sorted(gl)}, carve golds {g.tolist()}")
        if not O.connected(len(pos), deps):
            raise SystemExit(f"gate G3: musique {qid}: the hops do not form one connected chain ({deps})")
        h = int(str(r["hop"])[0])
        if h != len(dec):
            raise SystemExit(f"gate G3: musique {qid}: hop {r['hop']} with {len(dec)} sub-questions")
        hops.append(h)
        for a, b in sorted({(min(pos[k], pos[x]), max(pos[k], pos[x])) for k, x in deps if pos[k] != pos[x]}):
            qi.append(j)
            keys.append(a * n + b)
    return np.asarray(qi, np.int64), np.asarray(keys, np.int64), np.asarray(hops, np.int64)


# ── grading ─────────────────────────────────────────────────────────────────


def grade_diff(cx, cr, qidx, nq):
    d, lo, hi = Z.boot_diff(cx, cr, qidx, nq)
    return {"diff": Z.pct(d), "lo": Z.pct(lo), "hi": Z.pct(hi), "call": Z.call(lo, hi)}


def slot_rows(cov, qidx, nq):
    """Per form: linked, share, minus N_sp and minus none (paired intervals); F0T minus CN_all."""
    rows = {}
    for f in FORMS:
        c = cov[f]
        rows[f] = {"linked": int(c.sum()), "share": Z.pct(c.mean()), "vs_N_sp": grade_diff(c, cov["N_sp"], qidx, nq),
                   "vs_none": grade_diff(c, cov["none"], qidx, nq)}
    rows["F0T_vs_CN_all"] = grade_diff(cov["F0T"], cov["CN_all"], qidx, nq)
    return rows


def price(sizes, cov):
    """New union pairs per slot added over the union without a NER store."""
    out = {}
    for f in FORMS[1:]:
        add = int(cov[f].sum() - cov["none"].sum())
        out[f] = round((sizes[f] - sizes["none"]) / add, 1) if add > 0 else None
    return out


def grade_dataset(F, n, carves, domains, kb=False, ch=None, slab=None):
    """carves = {carve: (ids, golds, sha)}; ch = {carve: (qidx, keys, hops)} (MuSiQue). Returns the domain sizes and the per-carve rows."""
    res = {"family_pairs": {k: int(v.size) for k, v in F.items()}, "domains": {}, "carves": {}}
    bases = {}
    for dom in domains:
        base = np.union1d(F["S"], F["W2_S"]) if dom == "SRK" else F["W2_S"]           # S and R's rule part: the pairs above K
        bases[dom] = base
        s_n = int(F["S"].size) if dom == "SRK" else 0
        k_in = member(F["K"], base)
        sizes, held = {}, {}
        for f in FORMS:
            sr = int(base.size) if f == "none" else union_size(base, F[f])
            k_net = int(F["K"].size - (k_in if f == "none" else (k_in | member(F["K"], F[f]))).sum())
            sizes[f] = sr + k_net
            held[f] = {"S": s_n, "R": sr - s_n, "K": k_net}                              # S > R > K: each pair once, in its highest family
        del k_in
        res["domains"][dom] = {"families": ["S", "R", "K"] if dom == "SRK" else ["R", "K"], "union_pairs": sizes, "held_S_R_K": held,
                               "ratio_to_N_sp": {f: round(sizes[f] / sizes["N_sp"], 4) for f in FORMS}}
    for cv, (ids, golds, sha) in carves.items():
        nq = len(ids)
        kinds = {"GP": gp_slots(golds, n, KB_GOLDS if kb else None)}
        if ch is not None:
            kinds["CH"] = ch[cv][:2]
        rc = {"questions": nq, "questions_npz_sha256": sha, "slots": {k: int(v[1].size) for k, v in kinds.items()}}
        for dom in domains:
            rd = {}
            for kind, (qidx, sk) in kinds.items():
                if sk.size == 0:
                    rd[kind] = None
                    continue
                inbase = member(sk, bases[dom]) | member(sk, F["K"])
                cov = {"none": inbase}
                for f in FORMS[1:]:
                    cov[f] = inbase | member(sk, F[f])
                rd[kind] = {"forms": slot_rows(cov, qidx, nq), "price": price(res["domains"][dom]["union_pairs"], cov),
                            "labels": {f: label_table(label_sets(sk, F, slab, dom, f), label_pairs(F, slab, dom, f), cov[f]) for f in ("N_sp", "F0T")}}
                if kind == "CH":
                    rd["CC"] = chain_connected(cov, qidx, ch[cv][2], nq)
            rc[dom] = rd
        res["carves"][cv] = rc
    return res, bases


def chain_connected(cov, qidx, hops, nq):
    """CC: a question whose chain slots are all linked, by hops (questions with at least one chain slot)."""
    has = np.bincount(qidx, minlength=nq) > 0
    out = {}
    for f in FORMS:
        miss = np.bincount(qidx, weights=(~cov[f]).astype(float), minlength=nq)
        cc = has & (miss == 0)
        out[f] = {str(h): {"questions": int((has & (hops == h)).sum()), "connected": int((cc & (hops == h)).sum()),
                           "share": Z.pct(cc[has & (hops == h)].mean()) if (has & (hops == h)).any() else None} for h in (2, 3, 4)}
        out[f]["all"] = {"questions": int(has.sum()), "connected": int(cc.sum()), "share": Z.pct(cc[has].mean())}
    q = np.flatnonzero(has)
    unit = np.arange(q.size)
    for f in ("F0T", "CN_all"):
        a = (has & (np.bincount(qidx, weights=(~cov[f]).astype(float), minlength=nq) == 0))[q]
        r = (has & (np.bincount(qidx, weights=(~cov["N_sp"]).astype(float), minlength=nq) == 0))[q]
        out[f]["all"]["vs_N_sp"] = grade_diff(a, r, unit, q.size)
    return out


def s_held(F, form_keys):
    """Beside (2Wiki, HotpotQA): the share of S's pairs the R + K union holds with each form."""
    if F["S"].size == 0:
        return None
    held = member(F["S"], F["W2_S"]) | member(F["S"], F["K"])
    out = {"none": Z.pct(held.mean())}
    for f in FORMS[1:]:
        out[f] = Z.pct((held | member(F["S"], form_keys[f])).mean())
    return out


def cmd_grade(a):
    d, t0 = a.dataset, time.time()
    served, freeze = u1b.package_root(a.host)
    man = json.loads((served / d / "DATASET.json").read_text(encoding="utf-8"))
    n = int(man["nodes"]["n"])
    F, shas, slab = families(served, d, n, man["nodes"]["sha256"], freeze)
    log(f"{d}: families " + ", ".join(f"{k} {v.size:,}" for k, v in F.items()) + f" ({time.time() - t0:.0f}s)")
    carves = {cv: carve_questions(d, cv) for cv in CARVES[d]}
    ch = None
    if d == "musique":
        recs = musique_queries(served, [i for cv in carves for i in carves[cv][0]])
        posmap = node_positions(served, d)
        if len(posmap) != n:
            raise SystemExit(f"gate G3: musique: {len(posmap)} distinct node ids for {n} nodes")
        ch = {cv: chain_slots(recs, carves[cv][0], carves[cv][1], posmap, n) for cv in carves}
        del recs, posmap
    res, _bases = grade_dataset(F, n, carves, DOMAINS[d], kb=d in KB, ch=ch, slab=slab)
    if slab is not None:
        res["S_relations"] = len(slab[2])
    if d in ("2wiki", "hotpotqa"):
        res["S_held_by_RK"] = s_held(F, F)
    out = {"dataset": d, "declared_in": DECLARED, "freeze_RECORD_SHA256": freeze, "nodes": n, "nodes_sha256": man["nodes"]["sha256"],
           "kb": d in KB, **shas, **res, "seconds": round(time.time() - t0, 1), **codes()}
    EL.write_json(OUT / d / "grade.json", out)
    log(f"{d}: graded in {out['seconds']}s; written {OUT / d / 'grade.json'}")
    return 0


# ── the rule (section 6) and the report ─────────────────────────────────────


def decide(G):
    """G = {dataset: grade}. The first form of TRY that has GP (and MuSiQue's CH) minus N_sp not BELOW on every passage dataset, domain and carve,
    and a union at most BOUND times N_sp's on all six datasets and domains (one construction), is CARRIED; else NOT_CARRIED with every failure
    named. A KB's GP rows never decide (its golds are an answer set)."""
    fails = {}
    for x in TRY:
        fx = []
        for d in SIX:
            g = G[d]
            for dom in DOMAINS[d]:
                r = g["domains"][dom]["ratio_to_N_sp"][x]
                if r > BOUND:
                    fx.append({"dataset": d, "domain": dom, "condition": "size", "ratio_to_N_sp": r})
                if d not in PASSAGE:
                    continue
                for cv in CARVES[d]:
                    for kind in ("GP", "CH"):
                        row = g["carves"][cv][dom].get(kind)
                        if row is None:
                            continue
                        c = row["forms"][x]["vs_N_sp"]
                        if c["call"] == "BELOW":
                            fx.append({"dataset": d, "domain": dom, "carve": cv, "condition": kind, **c})
        fails[x] = fx
        if not fx:
            return f"CARRIED {x}", x, fails
    return "NOT_CARRIED", None, fails


def load_grades():
    G, bad = {}, []
    for d in SIX:
        p = OUT / d / "grade.json"
        if not p.exists():
            bad.append(d)
            continue
        g = json.loads(p.read_text(encoding="utf-8"))
        if g["script_sha256"] != script_sha256():
            bad.append(d)
            continue
        G[d] = g
    return G, bad


def cmd_report(_a):
    G, bad = load_grades()
    if bad:
        rec = {"declared_in": DECLARED, "verdict": "REFUSED", "missing_or_other_code": bad, **codes()}
        EL.write_json(OUT / "report.json", rec)
        log(f"REFUSED: no grade of this code for {bad}")
        return 1
    verdict, carried, fails = decide(G)
    rec = {"declared_in": DECLARED, "verdict": verdict, "carried": carried, "failures": fails, "bound": BOUND, "boot": BOOT, "seed": SEED,
           "kb_golds_read": list(KB_GOLDS), "grades_sha256": {d: Z.sha_file(OUT / d / "grade.json") for d in G}, "datasets": G, **codes()}
    EL.write_json(OUT / "report.json", rec)
    (OUT / "report.md").write_text(render(rec), encoding="utf-8", newline="\n")
    log(f"{verdict}; written {OUT / 'report.json'} and report.md")
    return 0


def _cell(c):
    return f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}"


def render(rec):
    L = [f"# UL1 report ({DECLARED}): {rec['verdict']}", "",
         "Graph level, no model. A slot is linked when its pair is an edge of the union. Differences are paired over questions (ratio estimator), "
         f"95% interval of {rec['boot']} question resamples (seed {rec['seed']}). Domains (A1): RK = R + K, SRK = S + R + K, with R = W2_S + "
         "the form (ABL-U's W2_S on the KBs); pairs held per family under S > R > K. KB GP rows (questions with "
         f"{rec['kb_golds_read'][0]}-{rec['kb_golds_read'][1]} golds) are beside.", ""]
    for d, g in rec["datasets"].items():
        L += [f"## {d}{' (KB, beside)' if g['kb'] else ''}", "", f"nodes {g['nodes']:,}; family pairs: " +
              ", ".join(f"{k} {v:,}" for k, v in g["family_pairs"].items()), ""]
        L += ["| domain | " + " | ".join(f"{f} union" for f in FORMS) + " | F0T / N_sp | CN_all / N_sp |", "|---|" + "---:|" * (len(FORMS) + 2)]
        for dom, x in g["domains"].items():
            L.append(f"| {dom} | " + " | ".join(f"{x['union_pairs'][f]:,}" for f in FORMS) +
                     f" | {x['ratio_to_N_sp']['F0T']:.4f} | {x['ratio_to_N_sp']['CN_all']:.4f} |")
        L.append("")
        for dom, x in g["domains"].items():
            L.append(f"- {dom}, pairs held under S > R > K: " + "; ".join(
                f"{f} " + " / ".join(f"{k} {v:,}" for k, v in x["held_S_R_K"][f].items() if dom == "SRK" or k != "S") for f in FORMS))
        L.append("")
        L += ["| carve | domain | kind | slots | " + " | ".join(f"{f} share" for f in FORMS) + " | F0T - N_sp | CN_all - N_sp | N_sp - none | F0T - CN_all | price (F0T / CN_all / N_sp) |",
              "|---|---|---|---:|" + "---:|" * len(FORMS) + ":--|:--|:--|:--|:--|"]
        for cv, rc in g["carves"].items():
            for dom in g["domains"]:
                for kind in ("GP", "CH"):
                    row = rc[dom].get(kind)
                    if kind not in rc["slots"]:
                        continue
                    if row is None:
                        L.append(f"| {cv} | {dom} | {kind} | 0 | " + " | ".join("-" for _ in FORMS) + " | - | - | - | - | - |")
                        continue
                    fr, pr = row["forms"], row["price"]
                    L.append(f"| {cv} | {dom} | {kind} | {rc['slots'][kind]:,} | " + " | ".join(f"{fr[f]['share']:.2f}%" for f in FORMS) +
                             f" | {_cell(fr['F0T']['vs_N_sp'])} | {_cell(fr['CN_all']['vs_N_sp'])} | {_cell(fr['N_sp']['vs_none'])} | "
                             f"{_cell(fr['F0T_vs_CN_all'])} | {pr['F0T']} / {pr['CN_all']} / {pr['N_sp']} |")
        L.append("")
        for cv, rc in g["carves"].items():
            for dom in g["domains"]:
                for kind in ("GP", "CH"):
                    row = rc[dom].get(kind)
                    if not row:
                        continue
                    for f in ("F0T", "N_sp"):
                        t = row["labels"][f]
                        L.append(f"- labels, {cv} {dom} {kind}, {f}'s union (slots carrying / alone / per 10^6 pairs): " + "; ".join(
                            f"{r['label']} {r['slots']}/{r['alone']}/{r['per_million_pairs']}" for r in t[:12]) +
                            (f"; and {len(t) - 12} more labels" if len(t) > 12 else ""))
        L.append("")
        for cv, rc in g["carves"].items():
            cc = rc.get("RK", {}).get("CC")
            if cc:
                L += [f"MuSiQue {cv}, chain connected (every chain slot of the question linked, RK):", "",
                      "| form | 2 hops | 3 hops | 4 hops | all | minus N_sp |", "|---|---:|---:|---:|---:|:--|"]
                for f in FORMS:
                    x = cc[f]
                    L.append(f"| {f} | " + " | ".join(f"{x[h]['connected']}/{x[h]['questions']} ({x[h]['share']}%)" for h in ("2", "3", "4", "all")) +
                             f" | {_cell(x['all']['vs_N_sp']) if 'vs_N_sp' in x['all'] else ''} |")
                L.append("")
        if g.get("S_held_by_RK"):
            L += ["Beside: share of S's pairs held by the RK union: " + ", ".join(f"{f} {v}%" for f, v in g["S_held_by_RK"].items()), ""]
    L += ["## The rule (section 6)", "", f"Bound: a form's union at most {rec['bound']} times N_sp's. **{rec['verdict']}.**", ""]
    for x, fx in rec["failures"].items():
        L.append(f"- {x}: " + ("holds everywhere" if not fx else "; ".join(
            f"{f['dataset']} {f['domain']} {f.get('carve', '')} {f['condition']} " +
            (f"ratio {f['ratio_to_N_sp']}" if f["condition"] == "size" else f"{f['diff']:+.2f} [{f['lo']:+.2f}, {f['hi']:+.2f}]") for f in fx)))
    return "\n".join(L) + "\n"


# ── self-test ───────────────────────────────────────────────────────────────


def selftest():
    # 1. the streamed build equals zap1b's level-0 forms (F0 and F0T) on toy corpora, and F0 equals U2r's CN_all
    corpora = [
        (["Tox0", "Tox1", "Delta State", "Tox3", "Tox4"],
         ["She was born in Delta State in 1990.", "The governor of Delta State resigned.", "It is a state in Nigeria, west of Benin.",
          "Benin and Nigeria are neighbours.", "Tox4"]),
        (["Tox0", "Tox1 (toy)", "Tox2", "Prince of Delta State"],
         ["she was named Chief Justice of Delta State in 2003.", "an organization for women of Delta State, Nigeria, was founded.",
          "the Chief Justice of Delta State retired.", "unrelated words here about Nigeria."]),
    ]
    for titles, texts in corpora:
        n = len(texts)
        passes = lambda: ((t, x if zu2b.is_descriptive(t, x) else None) for t, x in zip(titles, texts))  # noqa: E731,B023
        src, grp, tsrc, tgrp, _nv = build_entries(passes, n)
        (fa, fb, fw), (la, lb, _lw) = forms_from_entries(src, grp, tsrc, tgrp, n)
        f0t = np.union1d(fa * n + fb, la * n + lb)
        runs, subs, truns, tsubs = AP.name_tables(passes, n)
        ref = AP.build_forms(AP.level_pairs(runs, subs, truns, tsubs, n))
        assert np.array_equal(fa * n + fb, ref["F0"]), (fa, fb, ref["F0"])
        assert np.array_equal(f0t, ref["F0T"]), (f0t, ref["F0T"])
        s2, g2, ti, _ = U.build_names(passes, n)
        (ua, ub, uw), _u = U.edges(s2, g2, ti, n)
        assert np.array_equal(fa, ua) and np.array_equal(fb, ub) and np.allclose(fw, uw)
    # the titled hub: passage 2 (title "Delta State") joins the two passages that name it only in F0T
    titles, texts = corpora[0]
    n = len(texts)
    passes = lambda: ((t, x if zu2b.is_descriptive(t, x) else None) for t, x in zip(titles, texts))  # noqa: E731
    src, grp, tsrc, tgrp, _nv = build_entries(passes, n)
    (fa, fb, _), (la, lb, _) = forms_from_entries(src, grp, tsrc, tgrp, n)
    f0 = set(zip(fa.tolist(), fb.tolist()))
    f0t = f0 | set(zip(la.tolist(), lb.tolist()))
    assert (0, 1) in f0 and (0, 2) not in f0 and (1, 2) not in f0, f0
    assert {(0, 1), (0, 2), (1, 2)} <= f0t, f0t
    assert (2, 3) in f0 and 4 not in {x for p in f0t for x in p}, f0                      # 'nigeria' and 'benin' in 2 and 3; node 4 is not read
    # 2. keys, membership, union sizes against brute force
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(5, 60))
        a, b = rng.integers(0, n, 300), rng.integers(0, n, 300)
        k = pair_keys(a, b, n)
        want = sorted({(min(x, y), max(x, y)) for x, y in zip(a.tolist(), b.tolist()) if x != y})
        assert k.tolist() == [x * n + y for x, y in want]
        ip, col = zu2b.undirected_csr(n, a[a != b], b[a != b])
        assert np.array_equal(csr_keys(ip, col, n), k)
        k2 = pair_keys(rng.integers(0, n, 200), rng.integers(0, n, 200), n)
        assert union_size(k, k2) == np.union1d(k, k2).size
        sk = rng.integers(0, n * n, 50)
        assert np.array_equal(member(sk, k), np.isin(sk, k))
    assert member(np.asarray([3]), np.zeros(0, np.int64)).tolist() == [False]
    # 3. GP slots, the KB cap, and the chain slots of a toy MuSiQue question
    golds = [np.asarray([1, 4, 7]), np.asarray([2]), np.asarray(list(range(12)))]
    qi, sk = gp_slots(golds, 20)
    assert qi.tolist() == [0, 0, 0] + [2] * 66 and sk[:3].tolist() == [1 * 20 + 4, 1 * 20 + 7, 4 * 20 + 7]
    qi, sk = gp_slots(golds, 20, KB_GOLDS)
    assert qi.tolist() == [0, 0, 0]
    posmap = {f"m:{i}": i for i in range(10)}
    recs = {"q1": {"context_node_ids": [f"m:{i}" for i in range(10)], "gold_node_ids": "['m:5', 'm:2', 'm:8']", "hop": "3hop1",
                   "question_decomposition": [{"question": "A >> b", "paragraph_support_idx": 5}, {"question": "#1 >> c", "paragraph_support_idx": 2},
                                              {"question": "#2 >> d", "paragraph_support_idx": 8}]},
            "q2": {"context_node_ids": [f"m:{i}" for i in range(10)], "gold_node_ids": ["m:1", "m:3", "m:9"], "hop": "3hop2",
                   "question_decomposition": [{"question": "A >> b", "paragraph_support_idx": 1}, {"question": "C >> d", "paragraph_support_idx": 3},
                                              {"question": "#1 and #2 >> e", "paragraph_support_idx": 9}]}}
    qi, sk, hops = chain_slots(recs, ["q1", "q2"], [np.asarray([2, 5, 8]), np.asarray([1, 3, 9])], posmap, 10)
    assert qi.tolist() == [0, 0, 1, 1] and sk.tolist() == [2 * 10 + 5, 2 * 10 + 8, 1 * 10 + 9, 3 * 10 + 9] and hops.tolist() == [3, 3], (qi, sk)
    bad = dict(recs)
    bad["q1"] = dict(recs["q1"], question_decomposition=[dict(x) for x in recs["q1"]["question_decomposition"]])
    bad["q1"]["question_decomposition"][2]["question"] = "D >> d"                        # hop 3 names no earlier hop: not one chain
    try:
        chain_slots(bad, ["q1"], [np.asarray([2, 5, 8])], posmap, 10)
        raise AssertionError("an unconnected chain passed G3")
    except SystemExit as e:
        assert "connected" in str(e)
    # 4. grading on a toy graph: A1's domains, union sizes and S > R > K counts against brute force, coverage, price, chain connectivity, labels
    n = 10
    slab = (np.asarray([9, 9, 25]), np.asarray([0, 1, 0]), ["a", "b"])                 # S: (0, 9) under two relations, (2, 5) under one
    F = {"S": np.asarray([9, 25]), "W2_S": pair_keys([2, 3], [5, 4], n), "K": pair_keys([4, 2, 2], [6, 5, 8], n), "N_sp": pair_keys([2, 1], [8, 9], n),
         "CN_all": pair_keys([2], [8], n), "F0T": pair_keys([2, 1, 3, 0], [8, 9, 9, 1], n)}
    carves = {"s1eval": (["q1", "q2"], [np.asarray([2, 5, 8]), np.asarray([1, 3, 9])], "x")}
    ch = {"s1eval": (qi, sk, hops)}
    res, _ = grade_dataset(F, n, carves, ("RK", "SRK"), ch=ch, slab=slab)
    for dm, fams in (("RK", ("W2_S", "K")), ("SRK", ("S", "W2_S", "K"))):
        for f in FORMS:
            parts = [F[x] for x in fams] + ([F[f]] if f != "none" else [])
            tot = np.unique(np.concatenate(parts)).size
            above_k = np.unique(np.concatenate([F[x] for x in fams if x != "K"] + ([F[f]] if f != "none" else []))).size
            h = res["domains"][dm]["held_S_R_K"][f]
            assert res["domains"][dm]["union_pairs"][f] == tot and h["S"] + h["R"] == above_k and sum(h.values()) == tot, (dm, f, h, tot)
            assert h["S"] == (2 if dm == "SRK" else 0)
    assert res["domains"]["RK"]["union_pairs"] == {"none": 4, "N_sp": 5, "CN_all": 4, "F0T": 7}, res["domains"]["RK"]
    assert res["domains"]["SRK"]["held_S_R_K"]["F0T"] == {"S": 2, "R": 5, "K": 1}
    r = res["carves"]["s1eval"]["RK"]
    assert [r["CH"]["forms"][f]["linked"] for f in FORMS] == [2, 3, 2, 4], r["CH"]["forms"]
    assert r["CH"]["price"]["F0T"] == 1.5 and r["CH"]["price"]["N_sp"] == 1.0, r["CH"]["price"]          # (7 - 4) / (4 - 2); (5 - 4) / (3 - 2)
    assert r["CC"]["F0T"]["all"]["connected"] == 2 and r["CC"]["N_sp"]["all"]["connected"] == 1 and r["CC"]["none"]["all"]["connected"] == 1
    assert r["GP"]["forms"]["F0T"]["linked"] == 4 and res["carves"]["s1eval"]["slots"] == {"GP": 6, "CH": 4}
    t = {x["label"]: (x["slots"], x["alone"], x["pairs"]) for x in res["carves"]["s1eval"]["SRK"]["CH"]["labels"]["F0T"]}
    assert t == {"S.a": (1, 0, 2), "R.mention": (1, 0, 2), "R.name_comention": (1, 0, 1), "R.title_comention": (2, 2, 3), "K.knn": (2, 0, 3)}, t
    t = {x["label"] for x in res["carves"]["s1eval"]["RK"]["CH"]["labels"]["N_sp"]}
    assert t == {"R.mention", "R.ner_spacy", "K.knn"}, t                                   # no S label in R + K
    # an undirected CSR read in blocks; a one-way CSR stops
    ip, col = zu2b.undirected_csr(n, np.asarray([0, 2, 3, 7]), np.asarray([9, 5, 4, 1]))
    assert np.array_equal(csr_keys(ip, col, n, rows_per=3), pair_keys([0, 2, 3, 7], [9, 5, 4, 1], n))
    try:
        csr_keys(np.asarray([0, 1] + [1] * 9), np.asarray([5]), n)
        raise AssertionError("a one-way CSR passed")
    except SystemExit as e:
        assert "diagonal" in str(e)
    # 5. the rule on synthetic grades
    def fake(call_f0t="LEVEL", call_cn="LEVEL", r_f0t=1.1, r_cn=1.05, gp=True):
        G = {}
        for d in SIX:
            g = {"domains": {}, "carves": {}}
            for dm in DOMAINS[d]:
                g["domains"][dm] = {"ratio_to_N_sp": {"none": 0.9, "N_sp": 1.0, "CN_all": r_cn, "F0T": r_f0t}}
            for cv in CARVES[d]:
                g["carves"][cv] = {}
                for dm in DOMAINS[d]:
                    row = {"forms": {"F0T": {"vs_N_sp": {"diff": 0.1, "lo": -0.1, "hi": 0.3, "call": call_f0t}},
                                     "CN_all": {"vs_N_sp": {"diff": 0.0, "lo": -0.2, "hi": 0.2, "call": call_cn}}}}
                    g["carves"][cv][dm] = {"GP": row if (gp and d != "squad") else None}
                    if d == "musique":
                        g["carves"][cv][dm]["CH"] = row
            G[d] = g
        return G
    assert decide(fake())[0] == "CARRIED F0T"
    v, x, fails = decide(fake(r_f0t=1.31))
    assert v == "CARRIED CN_all" and all(f["condition"] == "size" for f in fails["F0T"]) and len(fails["F0T"]) == 8, fails
    v, x, fails = decide(fake(call_f0t="BELOW", call_cn="BELOW"))
    assert v == "NOT_CARRIED" and x is None and fails["CN_all"], v
    G = fake()
    G["metaqa"]["carves"]["s1sel"]["SRK"]["GP"]["forms"]["F0T"]["vs_N_sp"]["call"] = "BELOW"   # a KB's GP never decides
    assert decide(G)[0] == "CARRIED F0T"
    G["webqsp"]["domains"]["SRK"]["ratio_to_N_sp"]["F0T"] = 1.4                              # the size bound holds on all six
    v, x, fails = decide(G)
    assert v == "CARRIED CN_all" and fails["F0T"] == [{"dataset": "webqsp", "domain": "SRK", "condition": "size", "ratio_to_N_sp": 1.4}], fails
    print("selftest ok")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for c in ("build", "grade"):
        p = sub.add_parser(c)
        p.add_argument("--dataset", required=True, choices=SIX)
        p.add_argument("--host", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    selftest()
    return {"build": cmd_build, "grade": cmd_grade, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
