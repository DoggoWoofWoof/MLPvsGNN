"""A6e (docs/A6E_LINK_WORDS_EDGES.md): the words around a link, in the form A6's verdict named for no GAIN, on both of
the lean pair's models. A6 (za6lw.py) gave D4's link score to the first stage's top five rows as six columns through
R1's pointwise head; both arms were NOT_ADOPTED with no GAIN, their only losses on webqsp zero-shot, where the
structural words are the relation names in an edge's slots. A6e, fixed in A6's section 5 and its section 7 before any
number:

    MLP (zrckpn)  the anchors widened to the first stage's top ten rows by rrf, and the structural columns split by their
                  reading: on a typed graph (a KB: relation names) the relation columns, on the others (title windows)
                  the title columns, so the KB's words cannot move the passage graphs' weights; eight columns, R1's
                  pointwise head (Linear(8, 16), GELU, Linear(16, 1); 161 parameters, the last layer zero at the start)
    GNN (zspkl)   each pool edge's own link score (D4's, between its two ends) as a weight in zsp's propagation: per
                  word family (title windows, relation names, shared entities) the link-weighted mean of the row's
                  neighbours' z-scores and log1p of its summed weight, six inputs beside zsp's ten into the first layer
                  of zsp's head through their own columns (zero at the start, 192 parameters), so the step reads the
                  worded links as messages and can learn to leave them

A link's weight is log1p of D4's link score (linkwords.link_score: the largest summed IDF of the question's stems, less
both ends' title stems, in one window of the links between the two rows); the kNN family has no words and weighs 0.
Both outputs come from one pass per step-1 cache part over P1c2's look: every worded pair (structural or NER, both ends
in the pool, no self-loop) is scored once; the MLP's columns are the anchors' pairs, the GNN's weights every pool edge
zlink stores, checked edge for edge against P1c2's zlink build. No parameter, label or score of a model enters a
column or a weight; the MLP reads only its own row's columns (no message passing, nothing normalised over the
question's scores); the GNN's step is message passing, as zsp's is. The mentions are A6's (outputs/a6/ents).

    python outputs/mp_unified/za6e.py build --dataset 2wiki --carve fit --part 0 --host
    python outputs/mp_unified/za6e.py train --split L-musique --name scr-a6e-zrckpn --arm zrckpn --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za6e.py read --name scr-a6e-zrckpn --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za6e.py compare --new outputs/screen/fits/scr-a6e-zrckpn \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a6e-zrckpn
    python outputs/mp_unified/za6e.py pair --arm zrckpn --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za6e.py recall --arm zrckpn --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za6e.py verdict
    python outputs/mp_unified/za6e.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if sys.argv[1:2] == ["build"]:                                  # one CPU per build item
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import za6lw as ZA6  # noqa: E402  (A6: the served interface, the anchors, A6's mentions, the toys)
import zp1c3 as Z3  # noqa: E402  (zspk and its KT carves; through it P1c2's roots, gate and card, zprop, zlink, rmatch)
import zp1c3p as Z3P  # noqa: E402  (zrckp)
import linkwords as LW  # noqa: E402  (D4: stems, windows, the link score)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A6E_LINK_WORDS_EDGES.md"
OUT = ROOT / "outputs" / "a6e"
SCREEN = ROOT / "outputs" / "screen"
COLS, EWS = "cols", "ew"                                        # the two outputs, under OUT
A6E_KEY, EW_KEY = "_a6e", "_a6e_ew"
N_ANCHOR = 10
COL_NAMES = ("log1p_w_title", "log1p_w_rel", "log1p_w_ner", "share_title", "share_rel", "share_ner", "has_struct",
             "has_ner")
NC = len(COL_NAMES)
WF = ("title", "rel", "ner")                                    # the word families; 3 is none (kNN)
NE = 2 * len(WF)
A6E_HID, A6E_SEED_OFF = 16, 6601                                # A6's head at eight inputs
HEAD = ("a6e_w1", "a6e_b1", "a6e_w2", "a6e_b2")
N_HEAD = A6E_HID * NC + A6E_HID + A6E_HID + 1
EW = ("ew_w1",)
N_EW = ZP.HID * NE
LRU = 50000
COL_SETTINGS = {"columns": list(COL_NAMES), "anchors": f"the first stage's top {N_ANCHOR} rows by rrf among the "
                                                       "retrieved rows (rrf > 0), ties by row",
                "score": "D4's link score (linkwords.link_score) of the links between the row and an anchor, per "
                         "family; the largest over the anchors other than the row",
                "split": "the structural family's score goes to the relation columns on a typed graph (KB relation "
                         "names) and to the title columns otherwise; has_struct is one column (a window exists)",
                "windows": "structural: title occurrences, relation names on KB graphs; ner: shared-entity mentions "
                           "(A6's, df within [2, 25]); knn: none",
                "share": "the score over the question's total stem IDF, 0 when that is 0",
                "has": "1 when the row has a window with an anchor in the family", "window": LW.WIN,
                "stem": LW.STEM, "idf": "log((N + 1) / (df + 1)) over every node's title and text",
                "stored": "float16"}
EW_SETTINGS = {"weight": "log1p of D4's link score (linkwords.link_score) of the links between the edge's two ends in "
                         "its family (all raw edges of that family between them); 0 on the kNN family",
               "edges": "zlink's stored edges (chunk_edges: undirected, both directions, unique, no self-loops), "
                        "checked edge for edge", "symmetric": True,
               "word_family": "title windows (structural, untyped graph), relation names (structural, typed graph), "
                              "shared entities (NER), none (kNN)", "stored": "float16"}
SETTINGS = {"mlp": {"columns": NC, "hidden": A6E_HID, "seed_offset": A6E_SEED_OFF, "parameters": N_HEAD,
                    "score_input": False, "build": COL_SETTINGS},
            "gnn": {"inputs": NE, "into": "zsp's head's first layer, own columns, zero at the start",
                    "parameters": N_EW, "z": "zsp's: the detached score's z-score in its pool, clipped to +-8",
                    "build": EW_SETTINGS},
            "ents": "A6's (outputs/a6/ents, za6lw.py merge)"}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpn": "zrckp", "zspkl": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a6e-{a}", "L-hotpotqa": f"scr-a6e-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpn": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkl": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpn": "zrckp plus the head over eight link-word columns to the first stage's top ten rows",
        "zspkl": "zspk with each pool edge's link score as a weight in its propagation"}
PAIR = {a: SCREEN / f"scr-a6e-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a6e-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("musique", "webqsp", "hotpotqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A6e's re-call of the arm, "
        "or not twelve reads (docs/A6E_LINK_WORDS_EDGES.md, section 4)")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── the build: one pass, both outputs ────────────────────────────────────────


def question_scores(n, A, eu, ev, ef, er, pool, q, idf, doc, kbn, lu, lv, lf):
    """One question's worded pairs scored once: (the eight columns (n, NC), each zlink edge's weight (lu.size,), counts).
    A pair is a family (0 structural, 1 NER) and two pool rows joined by at least one raw edge of it; its score is D4's
    over all of them (on a typed graph each raw edge's relation slots)."""
    m = (eu != ev) & (ef < 2)
    a_, b_, f_ = np.minimum(eu[m], ev[m]), np.maximum(eu[m], ev[m]), ef[m]
    uk, inv = np.unique((a_ * n + b_) * 2 + f_, return_inverse=True)
    G = uk.size
    ga, gb, gf = uk // 2 // n, uk // 2 % n, uk % 2
    sc, hw = np.zeros(G), np.zeros(G, bool)
    if G and kbn is not None:
        erm = er[m]
        order = np.argsort(inv, kind="stable")
        bnd = np.r_[0, np.cumsum(np.bincount(inv, minlength=G))]
    for g in range(G):
        f = int(gf[g])
        lk = [(0, erm[k]) for k in order[bnd[g]:bnd[g + 1]]] if (f == 0 and kbn is not None) else [(f, None)]
        s_, _fam, nw = LW.link_score(doc(int(pool[ga[g]])), doc(int(pool[gb[g]])), lk, q, idf, kbn)
        sc[g], hw[g] = s_, nw > 0
    # the MLP's columns: per row, the best pair with an anchor at its other end, per family
    isA = np.zeros(n, bool)
    isA[A] = True
    ka, kb_ = isA[ga], isA[gb]
    c = np.r_[gb[ka], ga[kb_]]
    ff = np.r_[gf[ka], gf[kb_]]
    best, has = np.zeros((n, 2)), np.zeros((n, 2), bool)
    np.maximum.at(best, (c, ff), np.r_[sc[ka], sc[kb_]])
    np.logical_or.at(has, (c, ff), np.r_[hw[ka], hw[kb_]])
    j0 = 1 if kbn is not None else 0
    out = np.zeros((n, NC))
    qm = float(sum(idf[w] for w in q))
    out[:, j0], out[:, 2] = np.log1p(best[:, 0]), np.log1p(best[:, 1])
    if qm > 0:
        out[:, 3 + j0], out[:, 5] = best[:, 0] / qm, best[:, 1] / qm
    out[:, 6:8] = has
    # the GNN's weights: each zlink edge of a worded family takes its pair's score
    w = np.zeros(lu.size)
    wd = lf < 2
    if wd.any():
        key = (np.minimum(lu[wd], lv[wd]) * n + np.maximum(lu[wd], lv[wd])) * 2 + lf[wd]
        pos = np.searchsorted(uk, key)
        if G == 0 or bool((pos >= G).any()) or not np.array_equal(uk[np.minimum(pos, G - 1)], key):
            raise SystemExit("za6e: a zlink edge has no raw edge of its family in the look")
        w[wd] = np.log1p(sc[pos])
    return out, w, {"pairs": int(G), "pairs_with_a_word": int((sc > 0).sum()), "anchor_pairs": int((ka | kb_).sum())}


def stored_links(ds, carve, p, lk_root):
    """zlink's stored edges of one step-1 part, each array checked against its record, and the record's sha256."""
    cp = Path(lk_root) / ds / carve / p.name
    rp = cp / "record.json"
    if not rp.exists():
        raise SystemExit(f"za6e: {rp} missing (zlink's build of {ds}/{carve})")
    h = json.loads(rp.read_text(encoding="utf-8"))
    if h.get("settings") != ZL.SETTINGS or h["step1_record_sha256"] != LC.sha_file(p / "record.json"):
        raise SystemExit(f"za6e: {rp} is not zlink's build of {p}")
    arr = {}
    for k in ("q_ne", "e_u", "e_v", "e_fam"):
        fn = cp / f"{k}.npy"
        if LC.sha_file(fn) != h["arrays"][k]["sha256"]:
            raise SystemExit(f"za6e: {fn}: sha256 is not its record's")
        arr[k] = np.load(fn)
    return h, LC.sha_file(rp), arr


def build(ds, carve, src, part=None, out_root=OUT, cache_root=None, look_root=None, lk_root=None, placement=None):
    """The eight columns and the edge weights of one step-1 cache part of a carve (every part when part is None), from
    P1c2's look (pools, rrf and raw edges), zlink's stored edges, one corpus pass and A6's mentions."""
    cache_root, look_root, lk_root = cache_root or Z2.CACHE, look_root or Z2.LOOK, lk_root or Z2.LINKS
    t0 = time.time()
    d, files, ids, head = LC.look_chunks(ds, carve, look_root)
    if "rrf" not in head["columns"]:
        raise SystemExit(f"za6e: {d}: the look has no rrf column")
    j_rrf = head["columns"].index("rrf")
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    if part is not None and not 0 <= part < len(parts):
        raise SystemExit(f"za6e: {ds}/{carve} has {len(parts)} parts, not a part {part}")
    qtext = src.questions(ds)
    kbn = src.kb_names(ds)
    plan, need = [], []
    for pi, (p, r) in enumerate(zip(parts, recs)):
        if part is not None and pi != part:
            continue
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"za6e: {p}: built from another look than {d}")
        qa, qb = r["query_range"]
        if qb - qa != r["queries"] or not 0 <= qa <= qb <= len(ids):
            raise SystemExit(f"za6e: {p}: query range {qa}..{qb} for {r['queries']} queries of {len(ids)}")
        miss = [x for x in ids[qa:qb] if x not in qtext]
        if miss:
            raise SystemExit(f"za6e: {ds}/{carve}: {len(miss)} questions are in no {'/'.join(ZA6.QUERY_FILES)} file "
                             f"(first {miss[:3]})")
        lh, lsha, larr = stored_links(ds, carve, p, lk_root)
        lo, hi = r["chunks"]
        for ch in range(lo, hi):
            with np.load(files[ch]) as zf:
                qps, qed = zf["q_pool_size"].astype(np.int64), zf["q_edges"].astype(np.int64)
                pool, eu, ev, ef = zf["pool"], zf["e_u"].astype(np.int64), zf["e_v"].astype(np.int64), zf["e_fam"]
            no = np.r_[0, np.cumsum(qps)]
            qe = np.repeat(np.arange(qps.size), qed)
            if qe.size != eu.size or int(no[-1]) != pool.size:
                raise SystemExit(f"za6e: {files[ch]}: pools or edges disagree with q_pool_size and q_edges")
            m = (eu != ev) & (ef < 2)
            need += [pool[no[qe[m]] + eu[m]], pool[no[qe[m]] + ev[m]]]
        plan.append((p, r, lh, lsha, larr))
    need = set(np.unique(np.concatenate(need)).astype(np.int64).tolist()) if need else set()
    qst = {x: LW.question_stems(qtext[x]) for _p, r, *_ in plan for x in ids[r["query_range"][0]:r["query_range"][1]]}
    texts, idf, info = src.corpus(ds, need, frozenset().union(*qst.values()) if qst else frozenset())
    mentions, ents_sha = src.mentions(ds, info["nodes_sha256"])

    @functools.lru_cache(maxsize=LRU)
    def doc(i):
        t = texts[i]
        return LW.Doc(t[0], t[1], mentions(i))

    done = {}
    for p, r, lh, lsha, larr in plan:
        tp = time.time()
        qi, qb = r["query_range"]
        lo, hi = r["chunks"]
        xs, ws, n_all = [], [], []
        acc = {"q_ne": [], "e_u": [], "e_v": [], "e_fam": []}
        st = {"pairs": 0, "pairs_with_a_word": 0, "anchor_pairs": 0}
        for ch in range(lo, hi):
            with np.load(files[ch]) as zf:
                c = {k: zf[k] for k in ("q_pool_size", "q_edges", "pool", "x", "e_u", "e_v", "e_fam")}
                er_all = zf["e_rel"] if "e_rel" in zf.files else None
            if c["x"].shape[1] != len(head["columns"]):
                raise SystemExit(f"za6e: {files[ch]}: {c['x'].shape[1]} columns, the look names {len(head['columns'])}")
            if kbn is not None and er_all is None:
                raise SystemExit(f"za6e: {files[ch]}: a typed graph's chunk without relation slots")
            ne, la, lb, lfam, _st = ZL.chunk_edges(c, files[ch])
            for k, x in zip(("q_ne", "e_u", "e_v", "e_fam"), (ne, la, lb, lfam)):
                acc[k].append(x)
            qps, qed = c["q_pool_size"].astype(np.int64), c["q_edges"].astype(np.int64)
            no, eo, lo_ = np.r_[0, np.cumsum(qps)], np.r_[0, np.cumsum(qed)], np.r_[0, np.cumsum(ne)]
            for i in range(qps.size):
                n = int(qps[i])
                pool = c["pool"][no[i]:no[i + 1]].astype(np.int64)
                eu, ev = c["e_u"][eo[i]:eo[i + 1]].astype(np.int64), c["e_v"][eo[i]:eo[i + 1]].astype(np.int64)
                ef = c["e_fam"][eo[i]:eo[i + 1]].astype(np.int64)
                er = None if (er_all is None or kbn is None) else er_all[eo[i]:eo[i + 1]]
                if er is not None and er.size and int(er.max()) >= len(kbn):
                    raise SystemExit(f"za6e: {ds}: relation slot {int(er.max())}, the vocabulary has {len(kbn)} names")
                A = ZA6.anchors_of(c["x"][no[i]:no[i + 1], j_rrf].astype(np.float64), N_ANCHOR)
                x_, w_, s_ = question_scores(n, A, eu, ev, ef, er, pool, qst[ids[qi]], idf, doc, kbn,
                                             la[lo_[i]:lo_[i + 1]].astype(np.int64),
                                             lb[lo_[i]:lo_[i + 1]].astype(np.int64),
                                             lfam[lo_[i]:lo_[i + 1]].astype(np.int64))
                xs.append(x_)
                ws.append(w_)
                for k in st:
                    st[k] += s_[k]
                n_all.append(n)
                qi += 1
                if qi % 500 == 0:
                    log(f"    {ds}/{carve} {p.name}: question {qi - r['query_range'][0]}/{r['queries']}, "
                        f"{st['pairs']} pairs ({time.time() - tp:.0f}s)")
        arrays = {k: np.concatenate(v) for k, v in acc.items()}
        for k in arrays:
            if not np.array_equal(arrays[k], larr[k]) or arrays[k].dtype != larr[k].dtype:
                raise SystemExit(f"za6e: {p.name}: the look's edges ({k}) are not zlink's stored edges")
        n_all = np.asarray(n_all, np.int64)
        if n_all.size != r["queries"] or int(n_all.sum()) != r["rows"] or qi != qb:
            raise SystemExit(f"za6e: {p}: {n_all.size} queries and {int(n_all.sum())} rows against {r['queries']} and "
                             f"{r['rows']}")
        if (p / "n.npy").exists() and not np.array_equal(
                np.asarray(LC.load_array(p, "n", r, verify=True)).astype(np.int64), n_all):
            raise SystemExit(f"za6e: {p}: the chunks' pool sizes are not step 1's cached n")
        x = np.concatenate(xs) if xs else np.zeros((0, NC))
        w = np.concatenate(ws) if ws else np.zeros(0)
        if x.shape[0] != r["rows"] or w.size != lh["edges"] or not (np.isfinite(x).all() and np.isfinite(w).all()):
            raise SystemExit(f"za6e: {p.name}: {x.shape[0]} rows and {w.size} weights against {r['rows']} and "
                             f"{lh['edges']}, or a value is not finite")
        x16, w16 = x.astype(np.float16), w.astype(np.float16)
        qs_ = lambda y: [round(float(v), 4) for v in np.quantile(y, [0.05, 0.5, 0.95])] if y.size else []
        common = {"declared_in": DECLARED, "stage": "A6e", "dataset": ds, "carve": carve, "part": p.name,
                  "step1_record_sha256": LC.sha_file(p / "record.json"), "look": head["records"],
                  "carve_ids_sha256": head["carve_ids_sha256"], "ents_record_sha256": ents_sha,
                  "nodes_sha256": info["nodes_sha256"], "queries": int(n_all.size), "rows": int(x.shape[0]),
                  "kb_names": kbn is not None, "pairs": st,
                  "cache_root": Path(cache_root).resolve().relative_to(ROOT).as_posix()
                  if Path(cache_root).resolve().is_relative_to(ROOT) else "toy",
                  "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__)}
        oc = Path(out_root) / COLS / ds / carve / p.name
        oc.mkdir(parents=True, exist_ok=True)
        sha_c = R.save_npy(oc, "cols", x16)
        R.write_json(oc / "record.json", {
            **common, "settings": COL_SETTINGS,
            "rows_with_a_window": {f: round(float(x[:, 6 + j].mean()), 4) if x.size else 0.0
                                   for j, f in enumerate(("struct", "ner"))},
            "rows_with_a_word": {f: round(float((x[:, j] > 0).mean()), 4) if x.size else 0.0
                                 for j, f in enumerate(("title", "rel", "ner"))},
            "max_abs_float16_error": float(np.abs(x16.astype(np.float64) - x).max()) if x.size else 0.0,
            "arrays": {"cols": {"dtype": x16.dtype.str, "shape": list(x16.shape), "sha256": sha_c}},
            "seconds": round(time.time() - tp, 1), "utc": utc()})
        oe = Path(out_root) / EWS / ds / carve / p.name
        oe.mkdir(parents=True, exist_ok=True)
        sha_w = R.save_npy(oe, "w", w16)
        wd = arrays["e_fam"] < 2
        R.write_json(oe / "record.json", {
            **common, "settings": EW_SETTINGS, "zlink_record_sha256": lsha, "edges": int(w.size),
            "worded_edges": int(wd.sum()),
            "worded_edges_with_a_word": round(float((w[wd] > 0).mean()), 4) if wd.any() else 0.0,
            "quantiles_5_50_95_where_a_word": qs_(w[w > 0]),
            "max_abs_float16_error": float(np.abs(w16.astype(np.float64) - w).max()) if w.size else 0.0,
            "arrays": {"w": {"dtype": w16.dtype.str, "shape": list(w16.shape), "sha256": sha_w}},
            "seconds": round(time.time() - tp, 1), "utc": utc()})
        done[p.name] = {"queries": int(n_all.size), "rows": int(x.shape[0]), "edges": int(w.size), "pairs": st}
        log(f"  {ds}/{carve} {p.name}: {n_all.size} queries, {x.shape[0]} rows, {w.size} edges; {st['pairs']} worded "
            f"pairs, {st['pairs_with_a_word']} with a word ({time.time() - t0:.0f}s)")
    doc.cache_clear()
    log(f"za6e build {ds}/{carve}: {len(done)} part(s), {len(need)} nodes read ({time.time() - t0:.0f}s)")
    return done


# ── the carves ───────────────────────────────────────────────────────────────


def part_record(root, c, p, settings):
    rp = Path(root) / c.ds / c.carve / p.name / "record.json"
    if not rp.exists():
        raise SystemExit(f"za6e: {rp} missing (za6e.py build --dataset {c.ds} --carve {c.carve})")
    h = json.loads(rp.read_text(encoding="utf-8"))
    if h.get("declared_in") != DECLARED or h.get("settings") != settings:
        raise SystemExit(f"za6e: {rp} is not A6e's build")
    if h["step1_record_sha256"] != c.part_shas[p.name]:
        raise SystemExit(f"za6e: {rp} was built on another step-1 part")
    return rp, h


def load_array(fn, h, key, verify):
    if verify and LC.sha_file(fn) != h["arrays"][key]["sha256"]:
        raise SystemExit(f"za6e: {fn}: sha256 is not its record's")
    x = np.load(fn)
    if x.dtype != np.float16 or list(x.shape) != h["arrays"][key]["shape"]:
        raise SystemExit(f"za6e: {fn}: {x.dtype} {x.shape}, not its record's float16 {h['arrays'][key]['shape']}")
    return x


def load_cols(c, root, cache_root, verify=True):
    """A carve's eight columns on its device (float16), each part tied to its step-1 record."""
    parts, _recs = LC.part_dirs(c.ds, c.carve, cache_root)
    xs, shas = [], {}
    for p in parts:
        rp, h = part_record(Path(root) / COLS, c, p, COL_SETTINGS)
        x = load_array(rp.parent / "cols.npy", h, "cols", verify)
        if x.shape[1:] != (NC,):
            raise SystemExit(f"za6e: {rp.parent}: {x.shape[1:]} columns, not {NC}")
        xs.append(x)
        shas[p.name] = LC.sha_file(rp)
    x = np.concatenate(xs)
    if x.shape[0] != int(c.off_np[-1]):
        raise SystemExit(f"za6e: {c.ds}/{c.carve}: {x.shape[0]} rows of columns against the cache's "
                         f"{int(c.off_np[-1])}")
    return torch.from_numpy(x).to(c.device), shas


def load_ew(c, root, cache_root, verify=True):
    """A carve's edge weights in host memory (float16), aligned with its zlink edges: each part tied to its step-1
    record and to the zlink record the carve loaded; and whether the graph is typed."""
    parts, _recs = LC.part_dirs(c.ds, c.carve, cache_root)
    ws, shas, kb = [], {}, set()
    for p in parts:
        rp, h = part_record(Path(root) / EWS, c, p, EW_SETTINGS)
        if h["zlink_record_sha256"] != c.links["records"].get(p.name):
            raise SystemExit(f"za6e: {rp} was built on other zlink edges than the carve's")
        x = load_array(rp.parent / "w.npy", h, "w", verify)
        if x.shape != (h["edges"],):
            raise SystemExit(f"za6e: {rp.parent}: {x.shape}, its record has {h['edges']} edges")
        ws.append(x)
        shas[p.name] = LC.sha_file(rp)
        kb.add(bool(h["kb_names"]))
    w = np.concatenate(ws)
    if w.size != c.links["e_u"].size or len(kb) != 1:
        raise SystemExit(f"za6e: {c.ds}/{c.carve}: {w.size} weights against {c.links['e_u'].size} edges, or its parts "
                         f"disagree on the graph's kind")
    return w, kb.pop(), shas


def cols_batch(c, qs):
    """The columns of questions qs, in lean_gpu.CacheCarve.batch's row order, float32 on c's device."""
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    idx = np.repeat(c.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    return c.a6e[torch.from_numpy(idx).to(c.device)].to(torch.float32)


def word_family(fam, kb):
    """Each edge's word family: 0 title windows, 1 relation names, 2 shared entities, 3 none (kNN)."""
    fam = np.asarray(fam, np.int64)
    return np.where(fam == 1, 2, np.where(fam == 0, 1 if kb else 0, 3))


def ew_batch(c, qs):
    """The weights and word families of questions qs's edges, in zlink.link_batch's order, on c's device."""
    lk = c.links
    qs = np.asarray(qs, np.int64)
    ne = lk["q_ne"][qs]
    idx = np.repeat(lk["e_off"][qs] - (np.cumsum(ne) - ne), ne) + np.arange(int(ne.sum()))
    return {"w": torch.from_numpy(c.ew[idx].astype(np.float32)).to(c.device),
            "wf": torch.from_numpy(word_family(lk["e_fam"][idx], c.ew_kb).astype(np.int8)).to(c.device)}


def cols_carve(base, root=OUT):
    """base's carve class with A6e's eight columns beside it under A6E_KEY; every block, row, side key and gold is
    base's, unchanged."""

    class ColsCarve(base):
        A6E_ROOT = root

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.a6e, self.a6e_records = load_cols(self, type(self).A6E_ROOT, cache_root, verify)

        def nbytes(self):
            return super().nbytes() + self.a6e.numel() * self.a6e.element_size()

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[A6E_KEY] = cols_batch(self, qs)
            return feats, nq, base_z, gold

    ColsCarve.__name__ = ColsCarve.__qualname__ = f"A6e{base.__name__}"
    return ColsCarve


def ew_carve(base, root=OUT):
    """base's carve class (one with zlink's edges) with A6e's edge weights beside them under EW_KEY; every block, row,
    edge and gold is base's, unchanged."""

    class EWCarve(base):
        A6E_ROOT = root

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.ew, self.ew_kb, self.ew_records = load_ew(self, type(self).A6E_ROOT, cache_root, verify)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[EW_KEY] = ew_batch(self, qs)
            if feats[EW_KEY]["w"].numel() != feats[ZL.LK_KEY]["u"].numel():
                raise SystemExit(f"za6e: {self.ds}/{self.carve}: the batch's weights are not its edges")
            return feats, nq, base_z, gold

    EWCarve.__name__ = EWCarve.__qualname__ = f"EW{base.__name__}"
    return EWCarve


A6EKTChainCarveZRC = cols_carve(Z3.KTChainCarveZRC)
EWKTLinkCarveBase = ew_carve(Z3.KTLinkCarveBase)


# ── the models ───────────────────────────────────────────────────────────────


def a6e_init(m, seed):
    g = torch.Generator().manual_seed(int(seed) + A6E_SEED_OFF)
    m.a6e_w1 = nn.Parameter((torch.rand(A6E_HID, NC, generator=g) * 2 - 1) / NC ** 0.5)
    m.a6e_b1 = nn.Parameter(torch.zeros(A6E_HID))
    m.a6e_w2 = nn.Parameter(torch.zeros(1, A6E_HID))
    m.a6e_b2 = nn.Parameter(torch.zeros(1))


def a6e_head(m, x):
    return Fn.linear(Fn.gelu(Fn.linear(x, m.a6e_w1, m.a6e_b1)), m.a6e_w2, m.a6e_b2).squeeze(-1)


class ZRCKPN(Z3P.ZRCKP):
    """zrckpn: zrckp's model plus R1's pointwise head (zero at the start) over each row's own eight columns."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        a6e_init(self, seed)

    a6e_head = a6e_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if A6E_KEY not in feats:
            return s
        return s + self.a6e_head(feats[A6E_KEY])


def ew_inputs(s, ew, lk, nq, B):
    """(N, 6): per word family (title windows, relation names, shared entities) the link-weighted mean of the row's
    in-neighbours' z-scores (zsp's: the detached score's z in its pool, clipped) and log1p of the summed weight; 0
    where the row has no worded neighbour of the family."""
    N = nq.numel()
    dt = s.dtype
    z = LG.seg_zscore8D(s.detach().unsqueeze(1), nq, B).squeeze(1)
    zc = z.clamp(-ZP.ZCLIP, ZP.ZCLIP)
    u, v = lk["u"], lk["v"]
    w = ew["w"].to(dt)
    at = v * 4 + ew["wf"].long()
    W = torch.zeros(N * 4, dtype=dt, device=s.device).index_add_(0, at, w).view(N, 4)[:, :3]
    M = torch.zeros(N * 4, dtype=dt, device=s.device).index_add_(0, at, w * zc[u]).view(N, 4)[:, :3]
    has = W > 0
    mean = torch.where(has, M / W.clamp_min(1e-12), torch.zeros_like(W))
    return torch.cat([mean, torch.log1p(W)], 1)


class ZSPKL(Z3.ZSPK):
    """zspkl: zspk with the worded links in its propagation: zsp's head's first layer reads its ten inputs and, through
    its own columns (zero at the start), the six link-weighted ones (MRO: ZSPK, ZProp, KTMix, ZRM, ...)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.ew_w1 = nn.Parameter(torch.zeros(ZP.HID, NE))

    def forward(self, feats, keep, nq, B, base_z):
        if EW_KEY not in feats or ZP.LK_KEY not in feats:
            return super().forward(feats, keep, nq, B, base_z)
        s = super(ZP.ZProp, self).forward(feats, keep, nq, B, base_z)
        lk = feats[ZP.LK_KEY]
        pre = (Fn.linear(ZP.prop_inputs(s, lk, nq, B), self.sp_w1, self.sp_b1)
               + Fn.linear(ew_inputs(s, feats[EW_KEY], lk, nq, B), self.ew_w1))
        return s + Fn.linear(Fn.gelu(pre), self.sp_w2, self.sp_b2).squeeze(-1)


ARMS = {"zrckpn": (ZRCKPN, A6EKTChainCarveZRC), "zspkl": (ZSPKL, EWKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def a6e_records(carves, root=OUT, cache_root=None):
    """Each carve part's A6e records' sha256 (columns and weights); stops when one is missing or not A6e's."""
    out = {}
    for ds, cv in carves:
        parts, _ = LC.part_dirs(ds, cv, cache_root or Z2.CACHE)
        for p in parts:
            for sub in (COLS, EWS):
                rp = Path(root) / sub / ds / cv / p.name / "record.json"
                if not rp.exists():
                    raise SystemExit(f"za6e: {rp} missing (za6e.py build --dataset {ds} --carve {cv})")
                if json.loads(rp.read_text(encoding="utf-8")).get("declared_in") != DECLARED:
                    raise SystemExit(f"za6e: {rp} is not A6e's")
                out[f"{sub}/{ds}/{cv}/{p.name}"] = LC.sha_file(rp)
    return out


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        cv = Z3.train_carves(split)
        rec.update({"za6e_sha256": LC.sha_src(__file__), "za6lw_sha256": LC.sha_src(ZA6.__file__),
                    "linkwords_sha256": LC.sha_src(LW.__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "a6e": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                        "device": Z2.DEVICE, "added": list(HEAD if arm == "zrckpn" else EW), "settings": SETTINGS,
                        "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                        "gate": LC.sha_file(Z2.GATE), "a6e_records": a6e_records(cv),
                        "ents_records": ZA6.ents_records(sorted({d for d, _ in cv})),
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
        raise SystemExit(f"za6e: train takes --arm zrckpn or zspkl, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za6e: A6e's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za6e: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    a6e_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a6e" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za6e: {name} is not an A6e fit (arm {rec.get('arm')})")
    if rec["a6e"].get("split") != split:
        raise SystemExit(f"za6e: {name} is {split}'s fit, its record says {rec['a6e'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    a6e_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za6e: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za6e: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A6e's link words) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A6e ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A6e", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za6e_sha256": LC.sha_src(__file__)})
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


NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools (the columns and weights built again on P1g-b's "
                              "looks), then its full run, its own declared file",
        "GAIN_AND_LOSS": "A6ep, the passage-graph back-off: the same arm with the typed graphs' relation-name words "
                         "left out (the relation columns and weights 0), so only title windows and shared entities "
                         "move either model, on both models, its own declared file",
        "NO_GAIN": "A6ex: the link words on the adopted arms (G5's zspx with the edge weights, R1's zrckpr with the "
                   "eight columns), on both models, its own declared file; the link-word idea closes as a model input "
                   "only if A6ex finds no GAIN either",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A6e's screen verdict, per arm, from its re-call (section 4); musique (D4's target), webqsp (A6's loss) and
    hotpotqa beside, musique's in-domain read of both arms side by side, and the next form each outcome names
    (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A6e" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A6e's re-call of {a}")
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
    rec = {"stage": "A6e", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za6e_sha256": LC.sha_src(__file__), "utc": utc()}
    md = [f"# A6e's screen verdict: zrckpn **{per['zrckpn']['verdict']}**, zspkl **{per['zspkl']['verdict']}**", "",
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
           "", "This verdict closes only this form (eight split columns to the first stage's top ten rows on the MLP; "
               "each pool edge's link score as a weighted message in zsp's step on the GNN); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za6e verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def pair_by_hand(a, c, f, eu, ev, ef, er, pool, q, idf, docs, kbn):
    """(score, has a window, linked): D4's score of the links between rows a and c in family f, by brute force."""
    E = [k for k in range(eu.size) if ef[k] == f and eu[k] != ev[k] and {int(eu[k]), int(ev[k])} == {a, c}]
    if not E:
        return 0.0, False, False
    da, dc = docs[int(pool[a])], docs[int(pool[c])]
    if f == 0 and kbn is not None:
        wins = [kbn[r] for k in E for r in er[k] if r >= 0]
    elif f == 0:
        wins = LW.title_windows(da, dc) + LW.title_windows(dc, da)
    else:
        wins = LW.ner_windows(da, dc)
    qq = q - da.title_stems - dc.title_stems
    return max([sum(idf[x] for x in qq.intersection(w)) for w in wins], default=0.0), bool(wins), True


def cols_by_hand(n, rrf, eu, ev, ef, er, pool, q, idf, docs, kbn, k=N_ANCHOR):
    """The eight columns by brute force over every (row, anchor, family): float64 (n, NC)."""
    ret = [i for i in range(n) if rrf[i] > 0]
    A = sorted(ret, key=lambda i: (-rrf[i], i))[:k]
    out = np.zeros((n, NC))
    qm = sum(idf[w] for w in q)
    j0 = 1 if kbn is not None else 0
    for c in range(n):
        for f in (0, 1):
            best, has = 0.0, False
            for a in A:
                if a != c:
                    sc, hw, linked = pair_by_hand(a, c, f, eu, ev, ef, er, pool, q, idf, docs, kbn)
                    if linked:
                        best, has = max(best, sc), has or hw
            j = j0 if f == 0 else 2
            out[c, j], out[c, 3 + j], out[c, 6 + f] = math.log1p(best), best / qm if qm > 0 else 0.0, float(has)
    return out


def ew_inputs_by_hand(s, w, wf, u, v, nq, B):
    """ew_inputs by loops (float64)."""
    z = LG.seg_zscore8D(s.detach().unsqueeze(1), nq, B).squeeze(1).double().clamp(-ZP.ZCLIP, ZP.ZCLIP)
    N = nq.numel()
    W, M = np.zeros((N, 3)), np.zeros((N, 3))
    for e in range(u.numel()):
        k = int(wf[e])
        if k < 3:
            W[int(v[e]), k] += float(w[e])
            M[int(v[e]), k] += float(w[e]) * float(z[int(u[e])])
    mean = np.where(W > 0, M / np.maximum(W, 1e-12), 0.0)
    return np.concatenate([mean, np.log1p(W)], 1)


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A6e", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's model plus the head on its carve with the eight columns beside; zspk's model with the
    #    weighted inputs on its carve with the weights beside; the bases unchanged
    assert S.ARMS["zrckpn"] == (ZRCKPN, A6EKTChainCarveZRC) and S.ARMS["zspkl"] == (ZSPKL, EWKTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZRCKPN.__mro__[1] is Z3P.ZRCKP and ZSPKL.__mro__[1:5] == (Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM)
    assert A6EKTChainCarveZRC.__mro__[1] is Z3.KTChainCarveZRC and EWKTLinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert N_HEAD == 161 and N_EW == 192 and ZL.FAMS == 3 and NC == 8 and N_ANCHOR == 10
    assert word_family([0, 1, 2, 0], False).tolist() == [0, 2, 3, 0] and word_family([0, 1, 2], True).tolist() == [1, 2, 3]
    tmp = Path(tempfile.mkdtemp(prefix="za6e_"))
    try:
        rng = np.random.default_rng(61)
        nlp = ZA6.FakeNLP()
        nodes = ZA6.toy_corpus(rng)
        N = len(nodes)
        # 2. the build: rmatch's typed toy look with pools and an rrf column (A6's); each question's eight columns and
        #    each zlink edge's weight against the brute force, on the passage reading and the KB reading
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        look = tmp / "look" / "metaqa" / "toy"
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
        ment_t, _, _ = ZA6.toy_mentions(nodes, nlp)
        qwords = ["alpha bravo", "Who led delta echo golf", "Where is hotel Kay", "kilo lima mike Lux", "oscar papa",
                  "romeo sierra tango victor Moe"]
        qtext = {x: qwords[i % len(qwords)] for i, x in enumerate(ids)}
        kbn = [LW.stems_of([w]) for w in ("alpha", "golf", "kilo", "romeo", "tango")]
        lkd = tmp / "lk" / "metaqa" / "toy"
        lks = {k_: np.concatenate([np.load(lkd / f"part_{p}of2" / f"{k_}.npy") for p in (0, 1)])
               for k_ in ("q_ne", "e_u", "e_v", "e_fam")}
        loff = np.r_[0, np.cumsum(lks["q_ne"])]
        many_anchors = wider = 0
        for kb in (None, kbn):
            src = ZA6.ToySrc(nodes, ment_t, qtext, kb)
            out_r = tmp / ("a6ekb" if kb else "a6e")
            build("metaqa", "toy", src, out_root=out_r, cache_root=tmp / "cache", look_root=tmp / "look",
                  lk_root=tmp / "lk")
            read_build, need_ref = set(src.read), set()
            _d, files, _ids, head = LC.look_chunks("metaqa", "toy", tmp / "look")
            got = np.concatenate([np.load(out_r / COLS / "metaqa" / "toy" / f"part_{p}of2" / "cols.npy") for p in (0, 1)])
            got_w = np.concatenate([np.load(out_r / EWS / "metaqa" / "toy" / f"part_{p}of2" / "w.npy") for p in (0, 1)])
            _t, idf, _i = src.corpus("metaqa", set(range(N)), frozenset().union(*(LW.question_stems(t) for t in
                                                                                 qtext.values())))
            docs = {i: LW.Doc(nodes[i][0], nodes[i][1], ment_t.get(i, ())) for i in range(N)}
            refs, ref_w, qi, n_word, n_none = [], [], 0, 0, 0
            for f in files:
                zc = np.load(f)
                no, eo = np.r_[0, np.cumsum(zc["q_pool_size"])], np.r_[0, np.cumsum(zc["q_edges"])]
                for i in range(zc["q_pool_size"].size):
                    n = int(zc["q_pool_size"][i])
                    sl, el = slice(no[i], no[i + 1]), slice(eo[i], eo[i + 1])
                    eu_, ev_, ef_ = (zc["e_u"][el].astype(np.int64), zc["e_v"][el].astype(np.int64),
                                     zc["e_fam"][el].astype(np.int64))
                    er_, pl, q_ = zc["e_rel"][el], zc["pool"][sl].astype(np.int64), LW.question_stems(qtext[ids[qi]])
                    rr = zc["x"][sl, 0].astype(np.float64)
                    ref = cols_by_hand(n, rr, eu_, ev_, ef_, er_, pl, q_, idf, docs, kb)
                    refs.append(ref)
                    if (rr > 0).sum() > 5:
                        many_anchors += 1
                        wider += int(not np.array_equal(
                            ref, cols_by_hand(n, rr, eu_, ev_, ef_, er_, pl, q_, idf, docs, kb, k=5)))
                    for e in range(loff[qi], loff[qi + 1]):
                        a_, b_, f_ = int(lks["e_u"][e]), int(lks["e_v"][e]), int(lks["e_fam"][e])
                        ref_w.append(math.log1p(pair_by_hand(a_, b_, f_, eu_, ev_, ef_, er_, pl, q_, idf, docs, kb)[0])
                                     if f_ < 2 else 0.0)
                    m_ = (eu_ != ev_) & (ef_ < 2)
                    need_ref |= set(pl[eu_[m_]].tolist()) | set(pl[ev_[m_]].tolist())
                    n_word += int((ref[:, :3] > 0).any(1).sum())
                    n_none += int((ref[:, 6:] == 0).all(1).sum())
                    qi += 1
            ref, ref_w = np.concatenate(refs), np.asarray(ref_w)
            assert got.shape == ref.shape == (int(sum(np.load(f)["q_pool_size"].sum() for f in files)), NC)
            assert np.abs(got.astype(np.float64) - ref).max() < 4e-3, np.abs(got.astype(np.float64) - ref).max()
            assert got_w.shape == ref_w.shape == lks["e_u"].shape
            assert np.abs(got_w.astype(np.float64) - ref_w).max() < 4e-3, np.abs(got_w.astype(np.float64) - ref_w).max()
            assert n_word > 3 and n_none > 3 and float((got_w > 0).mean()) > 0.05, (kb is not None, n_word, n_none)
            assert not got_w[lks["e_fam"] == 2].any()
            assert read_build == need_ref, (len(read_build), len(need_ref))
            # the split: the passage reading fills only the title columns, the KB reading only the relation columns
            unused = (1, 4) if kb is None else (0, 3)
            assert not got[:, list(unused)].any() and got[:, 0 if kb is None else 1].any() and got[:, 2].any()
            for sub in (COLS, EWS):
                rec0 = json.loads((out_r / sub / "metaqa" / "toy" / "part_0of2" / "record.json").read_text(
                    encoding="utf-8"))
                assert rec0["stage"] == "A6e" and rec0["kb_names"] == (kb is not None)
                assert rec0["max_abs_float16_error"] < 4e-3
        assert many_anchors > 3 and wider > 0, (many_anchors, wider)
        # the KB reading reads the relation names: the structural values move, the NER ones do not
        a = np.load(tmp / "a6e" / COLS / "metaqa" / "toy" / "part_0of2" / "cols.npy")
        b = np.load(tmp / "a6ekb" / COLS / "metaqa" / "toy" / "part_0of2" / "cols.npy")
        assert np.array_equal(a[:, [2, 5, 7]], b[:, [2, 5, 7]]) and not np.array_equal(a[:, [0, 1]], b[:, [0, 1]])
        wa, wb = (np.load(r_ / EWS / "metaqa" / "toy" / "part_0of2" / "w.npy") for r_ in (tmp / "a6e", tmp / "a6ekb"))
        f0 = np.load(lkd / "part_0of2" / "e_fam.npy")
        assert np.array_equal(wa[f0 == 1], wb[f0 == 1]) and not np.array_equal(wa[f0 == 0], wb[f0 == 0])
        # one part at a time is the whole build's arrays byte for byte
        for pt in (1, 0):
            build("metaqa", "toy", ZA6.ToySrc(nodes, ment_t, qtext), pt, out_root=tmp / "a6ep",
                  cache_root=tmp / "cache", look_root=tmp / "look", lk_root=tmp / "lk")
        for pt in (0, 1):
            for sub, fn in ((COLS, "cols.npy"), (EWS, "w.npy")):
                fa, fb = (r_ / sub / "metaqa" / "toy" / f"part_{pt}of2" / fn for r_ in (tmp / "a6e", tmp / "a6ep"))
                assert fa.read_bytes() == fb.read_bytes(), (pt, sub)
        # the build refuses a missing question, a slot past the names, a part from another look, a part out of range,
        # zlink edges built on another part and zlink edges that are not the look's
        args = (tmp / "x", tmp / "cache", tmp / "look", tmp / "lk")
        SR.must_stop(build, "metaqa", "toy", ZA6.ToySrc(nodes, ment_t, dict(list(qtext.items())[1:])), None, *args)
        SR.must_stop(build, "metaqa", "toy", ZA6.ToySrc(nodes, ment_t, qtext, kbn[:2]), None, *args)
        SR.must_stop(build, "metaqa", "toy", ZA6.ToySrc(nodes, ment_t, qtext), 2, *args)
        p0 = tmp / "cache" / "metaqa" / "toy" / "part_0of2" / "record.json"
        keep_b = p0.read_bytes()
        h = json.loads(keep_b)
        h["look"]["carve_ids_sha256"] = "x"
        R.write_json(p0, h)
        SR.must_stop(build, "metaqa", "toy", ZA6.ToySrc(nodes, ment_t, qtext), None, *args)
        p0.write_bytes(keep_b)
        lr = lkd / "part_1of2" / "record.json"
        keep_l = lr.read_bytes()
        h = json.loads(keep_l)
        h["step1_record_sha256"] = "x"
        R.write_json(lr, h)
        SR.must_stop(build, "metaqa", "toy", ZA6.ToySrc(nodes, ment_t, qtext), 1, *args)
        lr.write_bytes(keep_l)
        fe = lkd / "part_1of2" / "e_fam.npy"
        keep_e = fe.read_bytes()
        ff_ = np.load(fe)
        ff_[0] = (ff_[0] + 1) % 3
        h = json.loads(keep_l)
        h["arrays"]["e_fam"]["sha256"] = R.save_npy(lkd / "part_1of2", "e_fam", ff_)
        R.write_json(lr, h)
        SR.must_stop(build, "metaqa", "toy", ZA6.ToySrc(nodes, ment_t, qtext), 1, *args)
        fe.write_bytes(keep_e)
        lr.write_bytes(keep_l)
        # 3. a batch's columns are its rows' own; a batch's weights are its edges' own, in zlink's order, with each
        #    edge's word family; the base carves' feats are unchanged
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        c_m = cols_carve(cls_ch, tmp / "a6e")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_g = ew_carve(cls_lk, tmp / "a6e")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_k = ew_carve(cls_lk, tmp / "a6ekb")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_b = cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert c_g.ew_kb is False and c_k.ew_kb is True
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:7], np.arange(c_m.rows)[:5]].astype(np.int64)
        B = qs.size
        fm, nq, bz, _gold = c_m.batch(qs, blocks)
        fg, nq_g, bz_g, _ = c_g.batch(qs, blocks)
        fb_, _nq, _bz, _ = c_b.batch(qs, blocks)
        assert ZL.LK_KEY not in fm and EW_KEY not in fm and A6E_KEY not in fg and fm[A6E_KEY].dtype == torch.float32
        assert torch.equal(nq, nq_g) and all(torch.equal(fg[ZL.LK_KEY][k_], fb_[ZL.LK_KEY][k_]) for k_ in "uvf")
        allc = np.concatenate([np.load(tmp / "a6e" / COLS / "metaqa" / "toy" / f"part_{p}of2" / "cols.npy")
                               for p in (0, 1)])
        want = np.concatenate([allc[c_m.off_np[q]:c_m.off_np[q + 1]] for q in qs]).astype(np.float32)
        assert np.array_equal(fm[A6E_KEY].numpy(), want) and float(fm[A6E_KEY].abs().max()) > 0.05
        allw = np.concatenate([np.load(tmp / "a6e" / EWS / "metaqa" / "toy" / f"part_{p}of2" / "w.npy") for p in (0, 1)])
        want_w = np.concatenate([allw[loff[q]:loff[q + 1]] for q in qs]).astype(np.float32)
        want_f = np.concatenate([lks["e_fam"][loff[q]:loff[q + 1]] for q in qs]).astype(np.int64)
        assert np.array_equal(fg[EW_KEY]["w"].numpy(), want_w) and float(want_w.max()) > 0.1
        assert np.array_equal(fg[EW_KEY]["wf"].numpy(), word_family(want_f, False))
        assert np.array_equal(fg[EW_KEY]["wf"].numpy(), fg[ZL.LK_KEY]["f"].numpy() * 0 + word_family(
            fg[ZL.LK_KEY]["f"].numpy(), False))
        fk, _nq, _bz, _ = c_k.batch(qs, blocks)
        assert np.array_equal(fk[EW_KEY]["wf"].numpy(), word_family(want_f, True))
        # a weights part built on other zlink edges is refused by the carve
        er_ = tmp / "a6e" / EWS / "metaqa" / "toy" / "part_0of2" / "record.json"
        keep_r = er_.read_bytes()
        h = json.loads(keep_r)
        h["zlink_record_sha256"] = "x"
        R.write_json(er_, h)
        SR.must_stop(ew_carve(cls_lk, tmp / "a6e"), "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        er_.write_bytes(keep_r)
        # 4. the models. Each arm from the same seed is its base's state plus its added parameters, no draw moved; at
        #    the start each is its base bit for bit, in eval and in training (scores and every shared gradient)
        keep_ = torch.ones(B, len(blocks))
        for arm, cls, base, cv, added in (("zrckpn", ZRCKPN, Z3P.ZRCKP, c_m, HEAD), ("zspkl", ZSPKL, Z3.ZSPK, c_g, EW)):
            f_, nq_, bz_, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(added) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k_: s1[k_] for k_ in s0})
            assert sum(s1[k_].numel() for k_ in added) == (N_HEAD if arm == "zrckpn" else N_EW)
            with torch.no_grad():
                if arm == "zspkl":
                    w = torch.randn(1, ZP.HID, generator=torch.Generator().manual_seed(8))
                    m0.sp_w2.copy_(w)
                    m1.sp_w2.copy_(w)
            for train_mode in (False, True):
                m0.train(train_mode)
                m1.train(train_mode)
                for m in (m0, m1):
                    m.zero_grad(set_to_none=True)
                torch.manual_seed(5)
                y0 = m0(f_, keep_, nq_, B, bz_)
                torch.manual_seed(5)
                y1 = m1(f_, keep_, nq_, B, bz_)
                assert torch.equal(y0, y1), (arm, train_mode)
                g_ = torch.randn(y0.shape, generator=torch.Generator().manual_seed(3))
                (y0 * g_).sum().backward()
                (y1 * g_).sum().backward()
                p0_, p1_ = dict(m0.named_parameters()), dict(m1.named_parameters())
                for k_, p in p0_.items():
                    assert (p.grad is None) == (p1_[k_].grad is None), k_
                    assert p.grad is None or torch.equal(p.grad, p1_[k_].grad), (arm, train_mode, k_)
                ga_ = [p1_[k_].grad for k_ in added]
                if arm == "zspkl":
                    assert float(ga_[0].abs().max()) > 0, "the weighted columns learn from the start"
            # with the added parameters moved: zrckpn adds the head over its columns; zspkl's head reads the weighted
            # inputs (against the brute force) through their own columns; without its side key each arm is its base
            m0.eval()
            m1.eval()
            with torch.no_grad():
                y0 = m0(f_, keep_, nq_, B, bz_)
                if arm == "zrckpn":
                    m1.a6e_w2.copy_(torch.randn(1, A6E_HID, generator=torch.Generator().manual_seed(9)))
                    y1 = m1(f_, keep_, nq_, B, bz_)
                    head = m1.a6e_head(f_[A6E_KEY])
                    assert float(head.abs().max()) > 1e-3 and torch.allclose(y1 - y0, head, atol=1e-5)
                    bare = {k_: x for k_, x in f_.items() if k_ != A6E_KEY}
                else:
                    m1.ew_w1.copy_(torch.randn(ZP.HID, NE, generator=torch.Generator().manual_seed(9)))
                    y1 = m1(f_, keep_, nq_, B, bz_)
                    pre = super(ZP.ZProp, m1).forward(f_, keep_, nq_, B, bz_)
                    lk_ = f_[ZL.LK_KEY]
                    xe = ew_inputs(pre, f_[EW_KEY], lk_, nq_, B)
                    xe_ref = ew_inputs_by_hand(pre, f_[EW_KEY]["w"], f_[EW_KEY]["wf"], lk_["u"], lk_["v"], nq_, B)
                    assert np.abs(xe.double().numpy() - xe_ref).max() < 1e-5 and float(np.abs(xe_ref[:, :3]).max()) > 0.01
                    assert float(np.abs(xe_ref[:, 3:]).max()) > 0.05
                    hid = Fn.gelu(Fn.linear(ZP.prop_inputs(pre, lk_, nq_, B), m1.sp_w1, m1.sp_b1)
                                  + Fn.linear(xe, m1.ew_w1))
                    assert torch.allclose(y1, pre + Fn.linear(hid, m1.sp_w2, m1.sp_b2).squeeze(-1), atol=1e-5)
                    assert float((y1 - y0).abs().max()) > 1e-4
                    bare = {k_: x for k_, x in f_.items() if k_ != EW_KEY}
                assert torch.equal(m1(bare, keep_, nq_, B, bz_), m0(bare, keep_, nq_, B, bz_))
        # 5. toy fits through lean_gpu's loop: a repeat is bit-identical and the added parameters learn; on the carve
        #    without the side key each arm's fit is its base's bit for bit on every shared key, its additions at the
        #    start
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv, bare_c, key in (("zrckpn", c_m, cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), "a6e_w2"),
                                     ("zspkl", c_g, cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), "ew_w1")):
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                bare = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, no side key")
            with S.patched(BASE_OF[arm]):
                base = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            assert all(not LG.same_state(a_, b_) for a_, b_ in zip(fit["states"] + [fit["swa"]],
                                                                   rep["states"] + [rep["swa"]]))
            assert float(fit["swa"][key].abs().max()) > 0
            for a_, b_ in zip(bare["states"] + [bare["swa"]], base["states"] + [base["swa"]]):
                assert not LG.same_state({k_: a_[k_] for k_ in b_}, b_), arm
                assert float(a_[key].abs().max()) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the refusals
    good = ["train", "--arm", "zrckpn", "--name", FIT["zrckpn"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpn"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zrckpw", "--name", FIT["zrckpn"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkl", "--name", FIT["zrckpn"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkl", "--name", FIT["zspkl"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    assert split_of_fit("scr-a6e-zspkl-hp") == ("zspkl", "L-hotpotqa") and split_of_fit("scr-a6-zspkw") == (None, None)
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a6e-zspkl", {"arm": "zspkw", "a6": {}}),
                        ("scr-a6e-zspkl-hp", {"arm": "zspkl", "a6e": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 7. relz's pair and re-call for each arm against its base arm's card fits, stamped A6e's; relz restored after
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
            assert pj["stage"] == "A6e" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A6e ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A6e's link words) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 8. the verdict, per arm, with the next form of section 5
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpn": fake_recall(T_ / "a.json", "zrckpn", {mi: "GAIN"}),
                       "zspkl": fake_recall(T_ / "b.json", "zspkl", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpn"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckpn"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspkl"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspkl"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckpn": {"call": "GAIN", "delta": 0.0},
                                                   "zspkl": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkl"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpn": fake_recall(T_ / "a.json", "zrckpn", {}),
                       "zspkl": fake_recall(T_ / "b.json", "zspkl", {gm: "GAIN"}, stage="A6")}, T_ / "v")
        assert rec["arms"]["zrckpn"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkl"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckpn"]["next_form"] == NEXT["NO_GAIN"]
        rec = verdict({"zrckpn": T_ / "none.json", "zspkl": fake_recall(T_ / "b.json", "zrckpn", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpn"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkl"]["verdict"] == "INCOMPLETE"
    log(f"za6e selftest: the columns and the edge weights match the brute force on the passage and KB readings (the "
        f"split fills only its reading's columns, the KB reading moves only the structural values), ten anchors reach "
        f"rows five do not, one part at a time is the whole build byte for byte, and the build refuses a missing "
        f"question, a slot past the names, another look, a part out of range and zlink edges not the look's; a "
        f"batch's columns and weights are its rows' and edges' own, with their word families; each arm starts as its "
        f"base bit for bit in eval and training (scores, every shared gradient), the head and the weighted inputs "
        f"(against the brute force) move it, and without its side key each is its base; toy fits repeat, learn, and "
        f"without the side key are the base's; the refusals; relz's pair and re-call per arm, stamped A6e's; the "
        f"verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "build":
        gp = argparse.ArgumentParser()
        gp.add_argument("--dataset", required=True, choices=sorted(ZA6.ENT_PARTS))
        gp.add_argument("--carve", required=True)
        gp.add_argument("--part", type=int)
        gp.add_argument("--host", action="store_true")
        g = gp.parse_args(rest)
        build(g.dataset, g.carve, ZA6.Served(g.host), part=g.part,
              placement={"where": "host" if g.host else "laptop"})
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
            log(f"za6e {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za6e: build, train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
