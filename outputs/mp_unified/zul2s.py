"""UL2S (docs/UL2S_SEED_ANCHORED_LABEL_LEDGER.md): the seed-anchored label ledger on the e3v layout, all seven datasets. An anchor is one of the
first stage's top five rows (the rrf of the e3v dense list and the SPLADE list decides; the dense top five is read beside it with the core
measures); a target is a gold the first stage missed; a candidate is a unit's pair at an anchor whose other end is outside the question's five.
Whole graph, no pool. Nothing trains and no model reads anything; no filed graph, pool, fit or record changes. Gold positions are the grading
only. zul2p, zul2c, zul2u, zul2, zul1, zul2m, zap4 and zenc are imported, never edited. Outputs only under outputs/ul2p/seed/. Arrays (npz)
are never written. UL2S decides nothing: it measures and names.

    python outputs/mp_unified/zul2s.py selftest
    python outputs/mp_unified/zul2s.py ledger --ds D [--host] [--threads 1]
    python outputs/mp_unified/zul2s.py report                                  # E1-E7, the lists section 7 names, the tables
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zul2p as P  # noqa: E402  (imports zul2c -> zul2u -> zul2 -> zmhb -> zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import zap4 as AP  # noqa: E402
import zenc as E  # noqa: E402

C, L2, UL, Z = P.C, P.L2, P.UL, P.Z
DECLARED = "docs/UL2S_SEED_ANCHORED_LABEL_LEDGER.md"
OUT = ROOT / "outputs" / "ul2p" / "seed"
DS7, FORMS, BINS, MIN_T, HOPS, MOST, KEEP = P.DS7, P.FORMS, P.BINS, P.MIN_T, P.HOPS, P.MOST, P.KEEP
DECIDE, BESIDE = P.DECIDE, P.BESIDE
TOP = AP.TOP                                      # 5: the first stage's top rows, the anchors
LISTS = ("rrf", "dense")                          # rrf decides (D-S1); dense is read beside it, core measures
KINDS = ("seed", "top1", "found")
CLASSES = ("SEED_B0", "SEED_FORM", "GOLD_LINKED", "LINKED_ELSE", "ISOLATED")
MIN_SHARE = 0.05                                  # section 7's cost flag: a label's share of the any-link candidates at the seeds
GAP_LIMIT = 0.05                                  # E7
EMPTY = np.zeros(0, np.int64)
OTHERS = np.array([[k for k in range(TOP) if k != j] for j in range(TOP)])
log = P.log


def stop(msg):
    raise SystemExit(f"[zul2s] STOP: {msg}")


def rel(p):
    return P.rel(p)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pair_key(a, b, n):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    return np.minimum(a, b) * n + np.maximum(a, b)


def in_keys(x, keys):
    """Membership of x (any shape) in the sorted unique keys, same shape."""
    x = np.asarray(x, np.int64)
    if keys.size == 0 or x.size == 0:
        return np.zeros(x.shape, bool)
    return UL.member(x.ravel(), keys).reshape(x.shape)


# ── the engine: seeds, one unit's candidates and targets, the kinds ──────────


class Seeds:
    """One list's anchors over one cell. golds: sorted unique arrays, one per question; seeds: (nq, TOP) distinct rows, best first. qs: the
    questions with a gold. Anchor occurrences (oq, oj, on): the TOP rows of each question in qs. Missed golds (mq, mg): golds outside the five.
    Targets (t_occ, t_mid, t_key): every anchor occurrence against every missed gold of its question."""

    def __init__(self, n, golds, seeds, hop=None):
        seeds = np.asarray(seeds, np.int64)
        nq = len(golds)
        if seeds.shape != (nq, TOP):
            stop(f"seeds {seeds.shape}, expected {(nq, TOP)}")
        self.n, self.nq, self.seeds = n, nq, seeds
        self.hop = np.zeros(nq, np.int64) if hop is None else np.asarray(hop, np.int64)
        qs = np.array([q for q in range(nq) if golds[q].size], np.int64)
        self.qs = qs
        self.oq = np.repeat(qs, TOP)
        self.oj = np.tile(np.arange(TOP), qs.size)
        self.on = seeds[qs].ravel()
        self.nodes = np.unique(self.on)
        self.ocid = np.searchsorted(self.nodes, self.on)
        gk = np.unique(np.concatenate([np.full(golds[q].size, q, np.int64) * n + golds[q] for q in qs])) if qs.size else EMPTY
        okey = self.oq * n + self.on
        found = in_keys(okey, gk)
        self.kmask = {"seed": np.ones(okey.size, bool), "top1": self.oj == 0, "found": found}
        self.anch = {k: np.bincount(self.oq[m], minlength=nq)[qs].astype(float) for k, m in self.kmask.items()}
        mk = gk[~in_keys(gk, np.unique(okey))]
        self.mq, self.mg = mk // n, mk % n
        mc = np.bincount(self.mq, minlength=nq)
        self.miss = mc[qs].astype(float)
        ptr = np.r_[0, np.cumsum(mc)]
        rep = mc[self.oq]
        self.t_occ = np.repeat(np.arange(okey.size), rep)
        start = np.repeat(np.cumsum(rep) - rep, rep)
        self.t_mid = ptr[self.oq[self.t_occ]] + (np.arange(self.t_occ.size) - start)
        self.t_key = pair_key(self.on[self.t_occ], self.mg[self.t_mid], n)
        sq = seeds[self.oq]
        self.w_key = pair_key(self.on[:, None], np.take_along_axis(sq, OTHERS[self.oj], axis=1), n)
        a_l, b_l = [], []
        for q in np.flatnonzero(mc >= 2):
            i, j = np.triu_indices(int(mc[q]), 1)
            a_l.append(ptr[q] + i)
            b_l.append(ptr[q] + j)
        self.mp_a = np.concatenate(a_l) if a_l else EMPTY
        self.mp_b = np.concatenate(b_l) if b_l else EMPTY
        self.mp_key = pair_key(self.mg[self.mp_a], self.mg[self.mp_b], n) if a_l else EMPTY

    def first_counts(self):
        found = self.anch["found"].sum()
        miss = self.miss
        return {"questions": int(self.qs.size), "golds": int(found + miss.sum()), "found_golds": int(found), "missed_golds": int(miss.sum()),
                "questions_with_missed": int((miss > 0).sum()), "questions_all_found": int((miss == 0).sum())}


def unit_eval(S, keys):
    """One unit's pairs (sorted unique; incident to the anchor nodes at least): per anchor occurrence its candidates (the unit's pairs at the
    row, either direction, whose other end is outside the question's five), and which anchor-by-missed-gold pairs are the unit's."""
    deg = P.node_deg(types.SimpleNamespace(nodes=S.nodes, n=S.n), keys)
    cand = deg[S.ocid] - in_keys(S.w_key, keys).sum(1)
    return cand, in_keys(S.t_key, keys)


def kind_cols(S, ev, kind):
    """Per-question sums over qs for one kind: candidates, targets (candidate occurrences that are missed golds), missed golds reached by at
    least one anchor of the kind; and the reached flags over the missed golds."""
    cand_occ, hit = ev
    m = S.kmask[kind]
    cand = np.bincount(S.oq[m], weights=cand_occ[m], minlength=S.nq)[S.qs]
    tm = m[S.t_occ] & hit
    tgt = np.bincount(S.oq[S.t_occ[tm]], minlength=S.nq)[S.qs].astype(float)
    vec = np.zeros(S.mq.size, bool)
    vec[S.t_mid[tm]] = True
    reach = np.bincount(S.mq[vec], minlength=S.nq)[S.qs].astype(float)
    return cand, tgt, reach, vec


def unit_measure(S, kind, pairs, cols, anycols, core, qm=None):
    """One unit's measures (section 4's table) over the questions that have an anchor of the kind (qm: a boolean over qs, a stratum)."""
    sel = S.anch[kind] > 0
    if qm is not None:
        sel = sel & qm
    c, t, r = cols[0][sel], cols[1][sel], cols[2][sel]
    a, m = S.anch[kind][sel], S.miss[sel]
    rec = {"questions": int(sel.sum()), "anchors": int(a.sum()), "pairs": int(pairs), "candidates": int(c.sum()), "targets": int(t.sum()),
           "missed": int(m.sum()), "reached": int(r.sum()),
           "price_per_anchor": round(float(AP.ratio(c.sum(), a.sum())), 4) if a.sum() else None,
           "precision": AP.boot(lambda s: AP.ratio(s["t"], s["c"]), {"t": t, "c": c}),
           "reach": AP.boot(lambda s: AP.ratio(s["r"], s["m"]), {"r": r, "m": m}),
           "graph_price": round(pairs / float(r.sum()), 1) if r.sum() else None}
    if core:
        return rec
    ex = c * m / (S.n - TOP)
    rec["base"] = float(AP.ratio(ex.sum(), c.sum()))
    rec["lift"] = AP.boot(lambda s: AP.ratio(s["t"], s["e"]), {"t": t, "e": ex})
    rec["lift_call"] = AP.lift_call(rec["lift"])
    qa = m > 0
    one = np.ones(int(qa.sum()))
    rec["question_reach"] = {"questions_with_missed": int(qa.sum()),
                             "any": AP.boot(lambda s: AP.ratio(s["x"], s["o"]), {"x": (r[qa] > 0).astype(float), "o": one}),
                             "all": AP.boot(lambda s: AP.ratio(s["x"], s["o"]), {"x": (r[qa] == m[qa]).astype(float), "o": one})}
    if anycols is not None:
        rec["vs_any"] = P.paired((c, t), (anycols[0][sel], anycols[1][sel]))
    return rec


def classes_of(S, rch_any, rch_all, all_keys):
    """The class (index into CLASSES) of each missed gold: reached through B0 from a seed, or only through a form, else joined to another
    missed gold of the question by a B0 or form pair, else with a B0 or form pair at all, else none."""
    gl = np.zeros(S.mq.size, bool)
    if S.mp_key.size:
        h = in_keys(S.mp_key, all_keys)
        gl[S.mp_a[h]] = True
        gl[S.mp_b[h]] = True
    un = np.unique(S.mg)
    deg = P.node_deg(types.SimpleNamespace(nodes=un, n=S.n), all_keys)
    has = deg[np.searchsorted(un, S.mg)] > 0 if un.size else np.zeros(0, bool)
    cls = np.full(S.mg.size, 4, np.int64)
    cls[has] = 3
    cls[gl] = 2
    cls[rch_all] = 1
    cls[rch_any] = 0
    return cls


def class_record(S, rch_any, rch_all, all_keys, multi):
    cls = classes_of(S, rch_any, rch_all, all_keys)
    qa = S.miss > 0
    out = {"missed_golds": int(cls.size), "questions_with_missed": int(qa.sum())}
    for i, name in enumerate(CLASSES):
        col = np.bincount(S.mq[cls == i], minlength=S.nq)[S.qs].astype(float)
        rec = {"count": int((cls == i).sum()), "na": bool(name == "GOLD_LINKED" and not multi),
               "share": AP.boot(lambda s: AP.ratio(s["c"], s["m"]), {"c": col[qa], "m": S.miss[qa]})}
        out[name] = rec
    return out, cls


def exact_rows(S, any_k, bits, names, kind="seed"):
    """The any-link pairs by exact label set (disjoint): each set's columns at the kind's anchors, and the reached flags."""
    rows = []
    for b in np.unique(bits):
        cols = kind_cols(S, unit_eval(S, any_k[bits == b]), kind)
        rows.append({"bits": int(b), "labels": [names[i] for i in range(len(names)) if int(b) >> i & 1], "cols": cols})
    return rows


def frontier_of(S, rows, kind="seed"):
    """The cut frontier: the exact label sets in descending precision, the cumulative candidates, targets and reach (as sets). Points only."""
    live = [r for r in rows if r["cols"][0].sum() > 0]
    live.sort(key=lambda r: (-float(AP.ratio(r["cols"][1].sum(), r["cols"][0].sum())), -float(r["cols"][0].sum()), r["bits"]))
    tot_c = sum(float(r["cols"][0].sum()) for r in live)
    sel = S.anch[kind] > 0
    miss_total = float(S.miss[sel].sum())
    cum_c = cum_t = 0.0
    vec = np.zeros(S.mq.size, bool)
    out = []
    for r in live:
        c, t = float(r["cols"][0].sum()), float(r["cols"][1].sum())
        cum_c, cum_t = cum_c + c, cum_t + t
        vec |= r["cols"][3]
        out.append({"labels": r["labels"], "candidates": int(c), "targets": int(t), "precision": round(t / c, 6),
                    "share_of_candidates": round(c / tot_c, 5) if tot_c else None, "cum_candidates": int(cum_c), "cum_targets": int(cum_t),
                    "cum_precision": round(cum_t / cum_c, 6), "cum_reach": round(float(vec.sum()) / miss_total, 5) if miss_total else None})
    return out


def exact_record(S, rows, anyc):
    """The most frequent exact label sets (until KEEP of the candidates, at most MOST), each measured; the sum is checked against the any unit."""
    tot = sum(float(r["cols"][0].sum()) for r in rows)
    if abs(tot - float(anyc[0].sum())) > 1e-6:
        stop(f"the exact sets hold {tot} candidates, the any-link unit {anyc[0].sum()}")
    rs = sorted(rows, key=lambda r: (-float(r["cols"][0].sum()), r["bits"]))
    out, seen = [], 0.0
    for r in rs[:MOST]:
        rec = unit_measure(S, "seed", 0, r["cols"], anyc, False)
        rec.pop("pairs")
        rec["labels"] = r["labels"]
        rec["share_of_candidates"] = round(float(AP.ratio(r["cols"][0].sum(), tot)), 5)
        out.append(rec)
        seen += float(r["cols"][0].sum())
        if tot and seen >= KEEP * tot:
            break
    return out


# ── the units: P's, plus 'all' = B0 and every form beyond it ─────────────────


def add_all_unit(U, D, v, sizes):
    parts = []
    for f in FORMS:
        sel = D.z[f"rank__{f}"] >= 0
        if v == "SRK" and D.z["inS"].size:
            sel &= ~D.z["inS"]
        parts.append(D.z["allF"][sel])
    pairs = int(sizes["B0"]) + C.union_count(sorted(parts, key=lambda p: -p.size))
    keys = np.unique(np.concatenate([U["any"][1]] + [U[f"form:{f}"][1] for f in FORMS]))
    U["all"] = (pairs, keys)


def list_record(S, U, bi, full, multi, d):
    """One list's record: per kind every unit's measures (core only for the dense list); for the seed kind of the deciding list, the exact
    label sets, the frontier, multi minus single, the first against the last rank quintile, MuSiQue's hops; for both lists the classes."""
    names, bits = bi
    any_k = U["any"][1]
    ev = {k: unit_eval(S, u[1]) for k, u in U.items()}
    kinds = KINDS if multi else ("seed", "top1")
    rec = {"first_stage": S.first_counts(), "kinds": {}}
    if not multi:
        rec["kinds_na"] = {"found": "one gold per question: a found anchor has no partner and the question has no target"}
    seed_cols = None
    for kind in kinds:
        cols = {k: kind_cols(S, ev[k], kind) for k in U}
        anyc = cols["any"]
        units = {k: unit_measure(S, kind, U[k][0], cols[k], None if k == "any" else anyc, not full) for k in U}
        rec["kinds"][kind] = {"units": units}
        if kind == "seed":
            seed_cols = cols
    rec["classes"], _cls = class_record(S, seed_cols["any"][3], seed_cols["all"][3], U["all"][1], multi)
    if not full:
        return rec
    rows = exact_rows(S, any_k, bits, names)
    anyc = seed_cols["any"]
    rec["exact"] = exact_record(S, rows, anyc)
    rec["frontier"] = frontier_of(S, rows)
    pop = np.array([bin(int(b)).count("1") for b in range(1 << len(names))])
    multi_c = kind_cols(S, unit_eval(S, any_k[pop[bits] >= 2]), "seed")
    single_c = kind_cols(S, unit_eval(S, any_k[pop[bits] == 1]), "seed")
    sel = S.anch["seed"] > 0
    rec["multi_minus_single"] = P.paired((multi_c[0][sel], multi_c[1][sel]), (single_c[0][sel], single_c[1][sel]))
    rec["q1_vs_q5"] = {f: P.paired((seed_cols[f"form:{f}:q1"][0][sel], seed_cols[f"form:{f}:q1"][1][sel]),
                                   (seed_cols[f"form:{f}:q{BINS}"][0][sel], seed_cols[f"form:{f}:q{BINS}"][1][sel])) for f in FORMS}
    if d == "musique":
        rec["hops"] = {}
        for h in HOPS:
            qm = S.hop[S.qs] == h
            if qm.any():
                rec["hops"][str(h)] = {"questions": int(qm.sum()),
                                       "units": {k: unit_measure(S, "seed", U[k][0], seed_cols[k], None, True, qm) for k in U}}
    return rec


# ── the first stage: the six from E0's lists, MultiHop-RAG from PL1 S0a's ────


def seeds_rrf(dense, splade):
    return E.rrf_top(np.asarray(dense, np.int64), np.asarray(splade, np.int64), TOP)


def hits_of(golds, top):
    return np.array([np.isin(golds[q], top[q]).sum() for q in range(len(golds))], np.int64)


def check_seeds(seeds, n, what):
    """G-S5: every question has five distinct rows in [0, n)."""
    s = np.asarray(seeds)
    ok = s.shape[1] == TOP and bool((s >= 0).all()) and bool((s < n).all())
    if ok:
        ss = np.sort(s, axis=1)
        ok = bool((ss[:, 1:] != ss[:, :-1]).all())
    if not ok:
        stop(f"G-S5: {what}: a question's five rows are not five distinct rows in [0, {n})")


def first_host(d, D):
    """The six: the carve's ids and golds (UL2's s1eval), E0's e3v lists and gold cache, the served SPLADE list. Gates G-S1 to G-S5."""
    ids, golds, qsha = UL.carve_questions(d, DECIDE)
    pe, pg = E.OUT / "e0" / "e3v" / f"{d}.npz", E.OUT / "e0" / "gold" / f"{d}.npz"
    for p in (pe, pg):
        if not p.is_file():
            stop(f"{rel(p)} is missing (E0's {'lists' if p is pe else 'gold cache'} of {d})")
    with np.load(pe, allow_pickle=False) as z:
        e0 = {k: z[k] for k in ("ids", "rows", "top_ids", "gold_total", "h_d5", "h_rrf5")}
    with np.load(pg, allow_pickle=False) as z:
        gids, flat, off = [str(x) for x in z["ids"]], z["flat"], z["off"]
    gc = [np.unique(flat[off[q]:off[q + 1]]) for q in range(len(gids))]
    gs = {}
    gs["G-S1"] = {"ids_equal_e0": [str(x) for x in e0["ids"]] == ids, "ids_equal_gold_cache": gids == ids,
                  "golds_equal_gold_cache": len(gc) == len(golds) and all(np.array_equal(a, b) for a, b in zip(golds, gc)),
                  "gold_total_equals_sizes": bool(np.array_equal(e0["gold_total"], np.array([g.size for g in golds], np.int64)))}
    if not all(gs["G-S1"].values()):
        stop(f"G-S1: {d}: {gs['G-S1']}")
    cells = [c for (cv, _k), c in D.cells.items() if cv == DECIDE]
    gs["G-S4"] = {"questions": len(ids), "cells_nq": sorted({int(c["nq"]) for c in cells}), "ids_sha256": hashlib.sha256(",".join(ids).encode()).hexdigest()}
    if any(int(c["nq"]) != len(ids) for c in cells):
        stop(f"G-S4: {d}: the population ({len(ids)}) is not UL2's s1eval cell ({gs['G-S4']['cells_nq']})")
    root = E.served_root()
    qids = json.loads((root / d / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    qmap = {q: j for j, q in enumerate(qids)}
    rows = np.asarray([qmap[q] for q in ids], np.int64)
    if not np.array_equal(rows, e0["rows"]):
        stop(f"G-S1: {d}: E0's rows are not the query file's positions of the carve's ids")
    spl = np.asarray(np.load(root / d / "retrieval_cache" / "splade_top1000.npz", mmap_mode="r")["ids"][rows], np.int64)
    dense = e0["top_ids"].astype(np.int64)
    seeds = {"dense": dense[:, :TOP], "rrf": seeds_rrf(dense, spl)}
    for k, v in seeds.items():
        check_seeds(v, D.n, f"{d}/{k}")
    gs["G-S5"] = "PASS"
    gs["G-S2"] = {"dense_top5_hits_equal_h_d5": bool(np.array_equal(hits_of(golds, seeds["dense"]), e0["h_d5"].astype(np.int64)))}
    gs["G-S3"] = {"rrf_top5_hits_equal_h_rrf5": bool(np.array_equal(hits_of(golds, seeds["rrf"]), e0["h_rrf5"].astype(np.int64)))}
    if not gs["G-S2"]["dense_top5_hits_equal_h_d5"]:
        stop(f"G-S2: {d}: the dense top five's gold hits are not E0's h_d5")
    if not gs["G-S3"]["rrf_top5_hits_equal_h_rrf5"]:
        stop(f"G-S3: {d}: the rrf top five's gold hits are not E0's h_rrf5")
    hop = np.array([AP.hop_of(d, x) for x in ids], np.int64) if d == "musique" else np.zeros(len(ids), np.int64)
    pop = types.SimpleNamespace(cv=DECIDE, ids=ids, golds=golds, seeds=seeds, hop=hop, zero_gold=0)
    src = {"e0_e3v": {"file": rel(pe), "sha256": E.sha256(pe)}, "e0_gold": {"file": rel(pg), "sha256": E.sha256(pg)},
           "questions_npz_sha256": qsha, "splade": {"file": f"{d}/retrieval_cache/splade_top1000.npz", "rows": int(rows.size)}}
    return [pop], gs, src


def first_mhr(D):
    """MultiHop-RAG: PL1 S0a's complete lists and gold table (their record's shas), the package's SPLADE list; test decides, dev beside."""
    import zul2m as M2
    s0 = ROOT / "outputs" / "pl1" / "multihoprag" / "RK" / "s0"
    rec = json.loads((s0 / "first.json").read_text(encoding="utf-8"))
    if rec["gate_GE"]["verdict"] != "PASS":
        stop("G-S3: PL1 S0a's G-E is not PASS")
    pe, pg = s0 / "first_e3v.npz", s0 / "gold.npz"
    if E.sha256(pe) != rec["files"]["first_e3v"]["sha256"] or E.sha256(pg) != rec["files"]["gold"]["sha256"]:
        stop("G-S3: S0a's lists or gold table are not the ones first.json names")
    with np.load(pe) as z:
        dense = z["ids"].astype(np.int64)
    with np.load(pg) as z:
        gids, flat, off = [str(x) for x in z["ids"]], z["flat"].astype(np.int64), z["off"].astype(np.int64)
    root, nodes, n, _nsha = M2.head()
    raw, split, _t = M2.golds_splits(root, nodes)
    nq = len(gids)
    if len(raw) != nq or dense.shape[0] != nq:
        stop(f"G-S1: {nq} gold rows, {len(raw)} questions, {dense.shape[0]} lists")
    gs = {"G-S1": {"gold_table_equals_golds_splits": all(np.array_equal(np.sort(flat[off[q]:off[q + 1]]), np.sort(raw[q])) for q in range(nq)),
                   "lists_sha_equals_first_json": True}}
    if not gs["G-S1"]["gold_table_equals_golds_splits"]:
        stop("G-S3: S0a's gold table is not zul2m.golds_splits'")
    sp = root / "multihoprag" / "retrieval_cache" / "splade_top1000.npz"
    if E.sha256(sp) != rec["sources"]["splade_top1000_sha256"]:
        stop("G-S3: the package's SPLADE list is not the one first.json names")
    with np.load(sp) as z:
        spl = z["ids"].astype(np.int64)
    if spl.shape != dense.shape:
        stop("G-S3: the SPLADE list and the dense list differ in shape")
    d5 = Z_pct(M2.M.coverage(dense, raw, 5).mean())
    want = rec["gate_GE"]["measures"]["e3v"]["filed"]["d5"]
    gs["G-S2"] = {"dense_d5_recomputed": d5, "first_json_d5": want, "equal": bool(d5 == want)}
    if not gs["G-S2"]["equal"]:
        stop(f"G-S2: the dense list's d5 is {d5}, first.json's {want}")
    seeds = {"dense": dense[:, :TOP], "rrf": seeds_rrf(dense, spl)}
    for k, v in seeds.items():
        check_seeds(v, n, f"multihoprag/{k}")
    gs["G-S5"] = "PASS"
    gs["G-S3"] = {"lists_and_gold_sha256_equal_first_json": True, "golds_equal_golds_splits": True}
    golds_all = [np.unique(g) for g in raw]
    pops = []
    for cv, spn in M2.CARVE_SPLIT.items():
        idx = np.flatnonzero(split == spn)
        keep = np.array([j for j in idx if golds_all[j].size], np.int64)
        cell = D.cells[(cv, "GP")]
        if int(cell["nq"]) != int(idx.size):
            stop(f"G-S4: {cv}: the split has {idx.size} questions, UL2's cell {cell['nq']}")
        pops.append(types.SimpleNamespace(cv=cv, ids=[gids[j] for j in keep], golds=[golds_all[j] for j in keep],
                                          seeds={k: v[keep] for k, v in seeds.items()}, hop=np.zeros(keep.size, np.int64),
                                          zero_gold=int(idx.size - keep.size)))
    gs["G-S4"] = {cp.cv: {"questions": len(cp.ids), "zero_gold_excluded": cp.zero_gold} for cp in pops}
    src = {"first_e3v": rec["files"]["first_e3v"], "gold": rec["files"]["gold"], "splade_top1000_sha256": rec["sources"]["splade_top1000_sha256"],
           "pl1_first_json_sha256": E.sha256(s0 / "first.json")}
    return pops, gs, src


def Z_pct(x):
    return round(100 * float(x), 2)


def anchor_nodes(D, pops):
    am = np.zeros(D.n, bool)
    for p in pops:
        am[np.asarray(p.seeds["rrf"]).ravel()] = True
        am[np.asarray(p.seeds["dense"]).ravel()] = True
        for g in p.golds:
            am[g] = True
    return am


# ── the ledger ───────────────────────────────────────────────────────────────


def cmd_ledger(a):
    d, t0 = a.ds, time.time()
    D = P.load_mhr() if d == "multihoprag" else P.load_host(d, a.host)
    gt = P.gates(D, D.g)
    pops, gs, src = first_mhr(D) if d == "multihoprag" else first_host(d, D)
    log(f"{d}: gates G-P1..G-P3 and G-S1..G-S5 pass; {sum(len(p.ids) for p in pops)} questions, versions {D.vers} ({time.time() - t0:.0f} s)")
    am = anchor_nodes(D, pops)
    rec = {"dataset": d, "declared_in": DECLARED, "status": "read", "nodes": D.n, "nodes_sha256": D.nsha, "sources": {**D.src, **src},
           "gates": {**gt, **gs}, "anchor_nodes": int(am.sum()), "multi_gold": any(g.size > 1 for p in pops for g in p.golds),
           "population": {p.cv: {"questions": len(p.ids), "zero_gold_excluded": p.zero_gold, "golds": int(sum(g.size for g in p.golds))} for p in pops},
           "versions": {}}
    for v in D.vers:
        U, bi, sizes = P.units_of(D, v, am)
        add_all_unit(U, D, v, sizes)
        log(f"{d}/{v}: {len(U)} units over {int(am.sum()):,} anchor nodes ({time.time() - t0:.0f} s)")
        cells = {}
        for p in pops:
            lists = {}
            for lst in LISTS:
                S = Seeds(D.n, p.golds, p.seeds[lst], p.hop)
                lists[lst] = list_record(S, U, bi, lst == "rrf", rec["multi_gold"], d)
                log(f"{d}/{v}/{p.cv}/{lst}: written into the record ({time.time() - t0:.0f} s)")
            cells[p.cv] = {"questions": len(p.ids), "lists": lists}
        rec["versions"][v] = {"b0": sizes, "cells": cells}
    rec["memory"] = C.peak_gb()
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zul2s_sha256=script_sha256(),
               zul2p_sha256=P.script_sha256(), **{k: v for k, v in C.codes().items() if k in ("zul2c_sha256", "zul2u_sha256", "zul2_sha256")})
    L2.wjson(OUT / d / "ledger.json", rec)
    log(f"{d}: ledger {rel(OUT / d / 'ledger.json')} ({rec['seconds']} s)")
    return 0


# ── the report: E1-E7, section 7's lists, the tables ─────────────────────────


def load_ledgers():
    out = {}
    for d in DS7:
        p = OUT / d / "ledger.json"
        if p.is_file():
            out[d] = json.loads(p.read_text(encoding="utf-8"))
    return out


def cells_of(L, carve=DECIDE):
    """(dataset, version, cell record, multi_gold) for every read deciding cell."""
    for d, led in L.items():
        if led.get("status") != "read":
            continue
        for v, vr in led["versions"].items():
            c = vr["cells"].get(carve)
            if c:
                yield d, v, c, led["multi_gold"]


def rr(c, kind="seed"):
    return c["lists"]["rrf"]["kinds"][kind]["units"]


def pt(x):
    return None if not x else x["pt"]


def verdict(held, reads, detail):
    return {"call": "NOT_READ" if not reads else ("HELD" if held == reads else "NOT_HELD"), "held": held, "reads": reads, "detail": detail}


def expectations(L):
    E_ = {}
    held = reads = 0
    det = []
    for d, v, c, _m in cells_of(L):
        u = rr(c)
        chk = ["any"] + [k for k, r in u.items() if k.startswith("label:") and r["targets"] >= MIN_T]
        bad = [k for k in chk if u[k]["lift_call"] != "POINTS"]
        reads += 1
        held += not bad
        det += [f"{d}/{v}: {k} {u[k]['lift_call']}" for k in bad]
    E_["E1"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, c, _m in cells_of(L):
        call = c["lists"]["rrf"]["multi_minus_single"]["call"]
        want = call != "ABOVE" if d == "metaqa" else call == "ABOVE"
        reads += 1
        held += want
        det.append(f"{d}/{v}: multi minus single {call}: {'held' if want else 'not'}")
    E_["E2"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, c, _m in cells_of(L):
        u = rr(c)
        for f in FORMS:
            r = u[f"form:{f}"]
            if r["targets"] >= MIN_T:
                reads += 1
                ok = r["vs_any"]["call"] == "BELOW"
                held += ok
                if not ok:
                    det.append(f"{d}/{v}: {f} vs any {r['vs_any']['call']}")
    E_["E3"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, c, _m in cells_of(L):
        if d in ("metaqa", "webqsp"):
            u = rr(c)
            s = pt(u["group:S"]["precision"])
            others = [pt(u[k]["precision"]) for k in ("label:R.mention", "label:R.name_comention", "label:R.title_comention", "label:K.knn")
                      if k in u and pt(u[k]["precision"]) is not None]
            labs = [pt(r["precision"]) for k, r in u.items() if k.startswith("label:S.") and pt(r["precision"]) is not None]
            reads += 1
            ok = s is not None and all(s > o for o in others)
            held += ok
            det.append(f"{d}/{v}: S group {s}, the other labels {others}, S relations {labs}: {'held' if ok else 'not'}")
    E_["E4"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, c, _m in cells_of(L):
        if d == "musique":
            hp = c["lists"]["rrf"].get("hops", {})
            r = [pt(hp[h]["units"]["any"]["reach"]) if h in hp else None for h in map(str, HOPS)]
            reads += 1
            ok = None not in r and r[0] > r[1] > r[2]
            held += ok
            det.append(f"musique/{v}: any-link reach by hops {r}: {'held' if ok else 'not'}")
    E_["E5"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, c, multi in cells_of(L):
        if multi:
            f = c["lists"]["rrf"]["kinds"]["found"]["units"]["any"]["precision"]
            t = c["lists"]["rrf"]["kinds"]["top1"]["units"]["any"]["precision"]
            reads += 1
            ok = bool(f and t and f["ci"][0] > t["ci"][1])
            held += ok
            det.append(f"{d}/{v}: found {pt(f)} {f and f['ci']}, top1 {pt(t)} {t and t['ci']}: {'held' if ok else 'not'}")
    E_["E6"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, c, multi in cells_of(L):
        if multi:
            g = pt(c["lists"]["rrf"]["classes"]["GOLD_LINKED"]["share"])
            reads += 1
            ok = g is not None and g < GAP_LIMIT
            held += ok
            det.append(f"{d}/{v}: GOLD_LINKED {g}: {'held' if ok else 'not'}")
    E_["E7"] = verdict(held, reads, det)
    return E_


def follows(L):
    """Section 7's lists, from every read deciding cell (rrf, seed kind). Names only: they change no rule and decide nothing."""
    out = {"weights": [], "universal": [], "agreement": [], "cuts": [], "flags": [], "gap": []}
    tally = {}
    for d, v, c, multi in cells_of(L):
        u = rr(c)
        tag = f"{d}/{v}"
        anyc = u["any"]["candidates"]
        for k, r in u.items():
            if not k.startswith("label:") or r["targets"] < MIN_T:
                continue
            call = (r.get("vs_any") or {}).get("call")
            t = tally.setdefault(k, [0, 0])
            t[1] += 1
            if r["lift_call"] == "POINTS" and call == "ABOVE":
                out["weights"].append(f"{tag}: {k}")
                t[0] += 1
        for k, r in u.items():
            if not k.startswith("label:") or not anyc or r["candidates"] / anyc < MIN_SHARE:
                continue
            call = (r.get("vs_any") or {}).get("call")
            if call == "BELOW" or r["lift_call"] in ("NO_LIFT", "BELOW_BASE"):
                out["flags"].append(f"{tag}: {k} ({r['candidates'] / anyc:.1%} of the any-link candidates; vs any {call}, {r['lift_call']})")
        for f in FORMS:
            if u[f"form:{f}"]["vs_any"]["call"] == "BELOW" and u[f"form:{f}:q1"]["vs_any"]["call"] == "ABOVE":
                out["cuts"].append(f"{tag}: {f}")
        out["agreement"].append((tag, c["lists"]["rrf"]["multi_minus_single"]["call"]))
        cl = c["lists"]["rrf"]["classes"]
        gap = (pt(cl["ISOLATED"]["share"]) or 0) + (pt(cl["LINKED_ELSE"]["share"]) or 0)
        out["gap"].append({"arm": tag, "no_rule_reaches": round(gap, 5), "ISOLATED": pt(cl["ISOLATED"]["share"]),
                           "LINKED_ELSE": pt(cl["LINKED_ELSE"]["share"]), "GOLD_LINKED": None if not multi else pt(cl["GOLD_LINKED"]["share"]),
                           "SEED_B0": pt(cl["SEED_B0"]["share"]), "SEED_FORM": pt(cl["SEED_FORM"]["share"])})
    out["universal"] = [f"{k}: ABOVE and POINTS on {a} of {b} arms with at least {MIN_T} targets{' (UNIVERSAL)' if a == b else ''}"
                        for k, (a, b) in sorted(tally.items())]
    out["agreement"] = {"above": sum(1 for _t, c in out["agreement"] if c == "ABOVE"), "arms": len(out["agreement"]),
                        "calls": dict(out["agreement"])}
    return out


def ci(x, scale=100.0, nd=2):
    return P.ci(x, scale, nd)


def table(units, keys, core=False):
    if core:
        rows = ["| unit | pairs | price per anchor | precision % | reach % | graph price |", "|---|---|---|---|---|---|"]
    else:
        rows = ["| unit | pairs | price per anchor | precision % | base % | lift call | reach % | graph price | vs any (points) |",
                "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        r = units[k]
        gp = r["graph_price"] if r["graph_price"] is not None else "-"
        if core:
            rows.append(f"| {k} | {r['pairs']:,} | {r['price_per_anchor']} | {ci(r['precision'])} | {ci(r['reach'])} | {gp} |")
        else:
            va = r.get("vs_any")
            base = f"{r['base'] * 100:.3f}" if r.get("base") == r.get("base") and r.get("base") is not None else "-"
            rows.append(f"| {k} | {r['pairs']:,} | {r['price_per_anchor']} | {ci(r['precision'])} | {base} | {r['lift_call'] or '-'} | "
                        f"{ci(r['reach'])} | {gp} | {(ci(va['diff']) + ' ' + va['call']) if va and va.get('diff') else '-'} |")
    return rows


def render(L, E_, W):
    lines = ["# UL2S report: the seed-anchored label ledger on the e3v layout", "",
             f"Declared in {DECLARED}. Written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. Decides nothing.", "",
             "## Expectations (rrf list, seed kind)", "", "| id | call | reads | detail |", "|---|---|---|---|"]
    for k, e in E_.items():
        lines.append(f"| {k} | {e['call']} | {e['held']} of {e['reads']} | {'; '.join(e['detail'][:6])}{' ...' if len(e['detail']) > 6 else ''} |")
    lines += ["", "## What follows (section 7: names only; no rule change, no carry)", ""]
    for key, title in (("weights", "Candidate label weights (POINTS and ABOVE the any-link unit)"),
                       ("universal", "Per label, across the arms where it has enough targets"),
                       ("cuts", "Candidate tighter cuts (a form BELOW the any-link unit whose first quintile is ABOVE it)"),
                       ("flags", f"Cost without precision (a label holding at least {MIN_SHARE:.0%} of the any-link candidates, BELOW or without lift)")):
        lines.append(f"- {title}: " + ("; ".join(W[key]) if W[key] else "none"))
    ag = W["agreement"]
    lines.append(f"- The agreement candidate (multi minus single ABOVE): {ag['above']} of {ag['arms']} arms ({ag['calls']})")
    lines += ["- The rules' gap (shares of the missed golds, rrf seeds):", ""]
    lines += ["| arm | no rule reaches (ISOLATED + LINKED_ELSE) % | ISOLATED % | LINKED_ELSE % | GOLD_LINKED % | SEED_B0 % | SEED_FORM % |", "|---|---|---|---|---|---|---|"]
    f = lambda x: "n/a" if x is None else f"{100 * x:.2f}"          # noqa: E731
    for g in W["gap"]:
        lines.append(f"| {g['arm']} | {f(g['no_rule_reaches'])} | {f(g['ISOLATED'])} | {f(g['LINKED_ELSE'])} | {f(g['GOLD_LINKED'])} | "
                     f"{f(g['SEED_B0'])} | {f(g['SEED_FORM'])} |")
    for d, led in L.items():
        lines += ["", f"## {d}", ""]
        for v, vr in led["versions"].items():
            for cv, c in vr["cells"].items():
                for lst in LISTS:
                    lr = c["lists"][lst]
                    fs = lr["first_stage"]
                    lines += ["", f"### {d} / {v} / {cv} / {lst} list: {fs['questions']:,} questions, {fs['golds']:,} golds, "
                              f"{fs['found_golds']:,} found, {fs['missed_golds']:,} missed ({fs['questions_with_missed']:,} questions)", ""]
                    keys = [k for k in lr["kinds"]["seed"]["units"] if not (k.startswith("form:") and ":q" in k)]
                    if lst == "rrf":
                        for kind, kk in lr["kinds"].items():
                            lines += ["", f"**{kind} anchors**", ""]
                            lines += table(kk["units"], keys if kind == "seed" else [k for k in keys if not k.startswith("form:")])
                    else:
                        for kind, kk in lr["kinds"].items():
                            lines += ["", f"**{kind} anchors (core)**", ""]
                            lines += table(kk["units"], [k for k in keys if not k.startswith("form:")], core=True)
                    cl = lr["classes"]
                    lines += ["", "Classes of the missed golds: " + "; ".join(
                        f"{n} {'n/a' if cl[n]['na'] else ci(cl[n]['share'])} ({cl[n]['count']:,})" for n in CLASSES), ""]
                    if lst != "rrf":
                        continue
                    lines += [f"Multi minus single: {ci(lr['multi_minus_single']['diff'])} {lr['multi_minus_single']['call']}.", "",
                              "Forms, first and last rank quintile (precision %; first against last):", ""]
                    for fm in FORMS:
                        q1, q5 = lr["kinds"]["seed"]["units"][f"form:{fm}:q1"], lr["kinds"]["seed"]["units"][f"form:{fm}:q{BINS}"]
                        lines.append(f"- {fm}: first {ci(q1['precision'])} ({q1['targets']} targets), last {ci(q5['precision'])} "
                                     f"({q5['targets']} targets), {lr['q1_vs_q5'][fm]['call']}")
                    lines += ["", "Cut frontier (exact label sets, descending precision, cumulative):", "",
                              "| labels | candidates | targets | precision % | cum candidates | cum precision % | cum reach % |", "|---|---|---|---|---|---|---|"]
                    for r in lr["frontier"]:
                        lines.append(f"| {'+'.join(r['labels'])} | {r['candidates']:,} | {r['targets']:,} | {100 * r['precision']:.3f} | "
                                     f"{r['cum_candidates']:,} | {100 * r['cum_precision']:.3f} | {f(r['cum_reach'])} |")
                    if "hops" in lr:
                        lines += ["", "By hops (any-link unit and its labels, core):", ""]
                        for h, hr in lr["hops"].items():
                            u = hr["units"]
                            lines.append(f"- {h} hops, {hr['questions']} questions: any reach {ci(u['any']['reach'])}, precision {ci(u['any']['precision'])}")
    return "\n".join(lines) + "\n"


def cmd_report(_a):
    L = load_ledgers()
    if not L:
        stop("no ledger is written")
    E_ = expectations(L)
    W = follows(L)
    rec = {"declared_in": DECLARED, "datasets": sorted(L), "expectations": E_, "follows": W,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "zul2s_sha256": script_sha256()}
    OUT.mkdir(parents=True, exist_ok=True)
    L2.wjson(OUT / "report.json", rec)
    (OUT / "report.md").write_bytes(render(L, E_, W).encode("utf-8"))
    log("report written: " + ", ".join(f"{k} {e['call']}" for k, e in E_.items()))
    return 0


# ── the self-test: brute force on random graphs, wrong engines, then MultiHop-RAG's real shapes ──


def toy(rng, n, nq):
    golds, seeds = [], np.zeros((nq, TOP), np.int64)
    for q in range(nq):
        golds.append(np.sort(rng.choice(n, int(rng.integers(0, 6)), replace=False)))
        g0 = golds[q][0] if golds[q].size and q % 2 == 0 else None
        pool = np.setdiff1d(np.arange(n), [] if g0 is None else [g0])
        row = list(rng.choice(pool, TOP - (g0 is not None), replace=False)) + ([g0] if g0 is not None else [])
        seeds[q] = rng.permutation(row)
    golds[3] = golds[2].copy()
    return golds, seeds


def nbrs_of(keys, n):
    nb = {}
    for k in np.asarray(keys).tolist():
        nb.setdefault(k // n, set()).add(k % n)
        nb.setdefault(k % n, set()).add(k // n)
    return nb


def brute_seed(n, keys, golds, seeds, kind):
    """By loops over sets: per question with a gold, its anchors of the kind, candidates, targets, reached missed golds, missed golds."""
    nb = nbrs_of(keys, n)
    out = []
    for g, s in zip(golds, seeds):
        if not g.size:
            continue
        five, gl = {int(x) for x in s}, {int(x) for x in g}
        missed = gl - five
        anchors = [int(x) for j, x in enumerate(s) if kind == "seed" or (kind == "top1" and j == 0) or (kind == "found" and int(x) in gl)]
        cand = tgt = 0
        reached = set()
        for a in anchors:
            cands = nb.get(a, set()) - five
            cand += len(cands)
            tgt += len(cands & missed)
            reached |= cands & missed
        out.append((len(anchors), cand, tgt, len(reached), len(missed)))
    return np.array(out, float).T


def engine_rows(S, keys, kind, ev=unit_eval):
    cand, tgt, rch, vec = kind_cols(S, ev(S, keys), kind)
    return np.array([S.anch[kind], cand, tgt, rch, S.miss]), vec


def unit_eval_no_within(S, keys):          # a wrong engine: pairs to the question's other seed rows are not taken out
    deg = P.node_deg(types.SimpleNamespace(nodes=S.nodes, n=S.n), keys)
    return deg[S.ocid], in_keys(S.t_key, keys)


def unit_eval_one_way(S, keys):            # a wrong engine: a pair counts at its smaller end only
    lo = keys // S.n
    i = np.minimum(np.searchsorted(S.nodes, lo), S.nodes.size - 1)
    deg = np.bincount(i[S.nodes[i] == lo], minlength=S.nodes.size).astype(float)
    return deg[S.ocid] - in_keys(S.w_key, keys).sum(1), in_keys(S.t_key, keys)


def brute_classes(n, any_k, all_k, golds, seeds):
    na, nl = nbrs_of(any_k, n), nbrs_of(all_k, n)
    out = []
    for g, s in zip(golds, seeds):
        if not g.size:
            continue
        five = {int(x) for x in s}
        missed = sorted({int(x) for x in g} - five)
        for x in missed:
            r_any = any(x in na.get(a, ()) for a in five)
            r_all = any(x in nl.get(a, ()) for a in five)
            glk = any(h in nl.get(x, ()) for h in missed if h != x)
            out.append(0 if r_any else 1 if r_all else 2 if glk else 3 if nl.get(x) else 4)
    return np.array(out, np.int64)


def selftest():
    rng = np.random.default_rng(11)
    n, nq = 40, 30
    golds, seeds = toy(rng, n, nq)
    S = Seeds(n, golds, seeds)
    assert S.qs.size == sum(1 for g in golds if g.size)
    assert np.array_equal(S.nodes, np.unique(np.concatenate([seeds[q] for q in S.qs])))
    bit_wrong = {"within": 0, "one_way": 0}
    am = np.zeros(n, bool)
    am[seeds.ravel()] = True
    for g in golds:
        am[g] = True
    for dens in (0.03, 0.15, 0.6):
        a, b = rng.integers(0, n, 900), rng.integers(0, n, 900)
        m = a != b
        keys = np.unique(np.minimum(a, b)[m] * n + np.maximum(a, b)[m])
        keys = keys[rng.random(keys.size) < dens]
        for kind in KINDS:
            want = brute_seed(n, keys, golds, seeds, kind)
            got, _v = engine_rows(S, keys, kind)
            assert np.array_equal(got, want), (dens, kind, got, want)
            got2, _v = engine_rows(S, P.incident(keys, am, n, block=7), kind)     # incident keys change nothing
            assert np.array_equal(got2, want), (dens, kind, "incident")
            bit_wrong["within"] += not np.array_equal(engine_rows(S, keys, kind, unit_eval_no_within)[0], want)
            bit_wrong["one_way"] += not np.array_equal(engine_rows(S, keys, kind, unit_eval_one_way)[0], want)
        assert np.array(engine_rows(S, EMPTY, "seed")[0])[1:4].sum() == 0
    assert bit_wrong["within"] > 0 and bit_wrong["one_way"] > 0, bit_wrong     # the wrong engines are caught
    # classes: the five-way split against set loops, partition of the missed golds, nested units
    any_k = np.unique(np.concatenate([np.minimum(x, y) * n + np.maximum(x, y) for x, y in
                                      [(rng.integers(0, n, 120), rng.integers(0, n, 120))]]))
    any_k = any_k[(any_k // n) != (any_k % n)]
    extra = np.unique(np.minimum(rng.integers(0, n, 150), rng.integers(0, n, 150)) * n + rng.integers(0, n, 150))
    extra = extra[(extra // n) < (extra % n)]
    all_k = np.union1d(any_k, extra)
    ra, rl = engine_rows(S, any_k, "seed")[1], engine_rows(S, all_k, "seed")[1]
    assert (rl | ~ra).all()                                                           # reach through B0 is reach through all
    cls = classes_of(S, ra, rl, P.incident(all_k, am, n, block=5))
    assert np.array_equal(cls, brute_classes(n, any_k, all_k, golds, seeds)), "classes"
    assert cls.size == int(S.miss.sum()) and set(np.unique(cls)) <= set(range(5))
    rec, _c = class_record(S, ra, rl, all_k, True)
    assert sum(rec[c]["count"] for c in CLASSES) == cls.size
    # the measure: expected targets, the base and the boot's point values
    cols = tuple(engine_rows(S, all_k, "seed")[0][1:4])
    mrec = unit_measure(S, "seed", all_k.size, cols, None, False)
    ex = float((cols[0] * S.miss / (n - TOP)).sum())
    assert abs(mrec["base"] - ex / cols[0].sum()) < 1e-12 and abs(mrec["lift"]["pt"] - round(cols[1].sum() / ex, 5)) < 1e-9
    assert mrec["targets"] == int(cols[1].sum()) and mrec["reached"] == int(cols[2].sum())
    # the cut frontier: the cumulative columns equal the union's, and the exact sets add up to the unit
    names = ["a", "b", "c"]
    bits = rng.integers(1, 8, all_k.size)
    rows = exact_rows(S, all_k, bits, names)
    assert abs(sum(r["cols"][0].sum() for r in rows) - cols[0].sum()) < 1e-6
    fr = frontier_of(S, rows)
    assert [r["precision"] for r in fr] == sorted((r["precision"] for r in fr), reverse=True)
    order = sorted((r for r in rows if r["cols"][0].sum() > 0),
                   key=lambda r: (-float(AP.ratio(r["cols"][1].sum(), r["cols"][0].sum())), -float(r["cols"][0].sum()), r["bits"]))
    for j in range(1, len(order) + 1):
        ks = np.unique(np.concatenate([all_k[bits == r["bits"]] for r in order[:j]]))
        c2, t2, r2, _v = kind_cols(S, unit_eval(S, ks), "seed")
        assert fr[j - 1]["cum_candidates"] == int(c2.sum()) and fr[j - 1]["cum_targets"] == int(t2.sum())
        assert abs(fr[j - 1]["cum_reach"] - r2.sum() / S.miss.sum()) < 1e-5, j
    ex_rec = exact_record(S, rows, tuple(cols))
    assert 1 <= len(ex_rec) <= MOST and sum(r["share_of_candidates"] for r in ex_rec) <= 1.00001
    # the bootstrap and the calls
    x = (np.array([3.0, 0, 2, 5]), np.array([1.0, 0, 1, 2]))
    p = P.paired(x, x)
    assert p["call"] == "LEVEL" and p["diff"]["pt"] == 0.0
    assert all(e["call"] == "NOT_READ" for e in expectations({}).values())
    assert follows({}) == {"weights": [], "universal": [], "agreement": {"above": 0, "arms": 0, "calls": {}}, "cuts": [], "flags": [], "gap": []}
    # section 7's lists and the expectations on a made-up ledger whose answers are known
    def unit(lift, va, cand=10, tgt=30, prec=0.1, lo=0.08, hi=0.12, reach=0.5):
        return {"lift_call": lift, "vs_any": {"call": va}, "candidates": cand, "targets": tgt, "precision": {"pt": prec, "ci": [lo, hi]},
                "reach": {"pt": reach, "ci": [reach - 0.01, reach + 0.01]}}
    fu = {"any": unit("POINTS", None, cand=100), "label:A": unit("POINTS", "ABOVE", cand=50), "label:B": unit("POINTS", "LEVEL", cand=4),
          "label:C": unit("NO_LIFT", "ABOVE", cand=10), "label:D": unit("POINTS", "BELOW", cand=30), "group:S": unit("POINTS", "ABOVE")}
    for f in FORMS:
        fu[f"form:{f}"] = unit("POINTS", "BELOW")
        fu[f"form:{f}:q1"] = unit("POINTS", "ABOVE" if f in ("H5", "NT1") else "LEVEL")
    fu["form:HM"] = unit("POINTS", "LEVEL")
    fu["form:HM:q1"] = unit("POINTS", "ABOVE")
    cl = {c: {"count": 1, "na": False, "share": {"pt": 0.1, "ci": [0.05, 0.15]}} for c in CLASSES}
    cl["GOLD_LINKED"]["share"]["pt"] = 0.02
    fk = {"seed": {"units": fu}, "top1": {"units": {"any": unit("POINTS", None, prec=0.05, lo=0.04, hi=0.06)}},
          "found": {"units": {"any": unit("POINTS", None, prec=0.2, lo=0.18, hi=0.22)}}}
    cell = {"questions": 5, "lists": {"rrf": {"kinds": fk, "classes": cl, "multi_minus_single": {"call": "ABOVE"}, "q1_vs_q5": {}}}}
    fake = {"zz": {"status": "read", "multi_gold": True, "versions": {"RK": {"cells": {DECIDE: cell}}}}}
    w = follows(fake)
    assert w["weights"] == ["zz/RK: label:A"], w
    assert w["cuts"] == ["zz/RK: H5", "zz/RK: NT1"], w
    assert len(w["flags"]) == 2 and any("label:D" in x for x in w["flags"]) and any("label:C" in x for x in w["flags"]), w     # 4% is not flagged
    assert w["universal"] == ["label:A: ABOVE and POINTS on 1 of 1 arms with at least 20 targets (UNIVERSAL)"] + [
        f"label:{x}: ABOVE and POINTS on 0 of 1 arms with at least 20 targets" for x in "BCD"], w
    assert w["agreement"]["above"] == 1 and abs(w["gap"][0]["no_rule_reaches"] - 0.2) < 1e-12
    ex = expectations(fake)
    assert ex["E1"]["call"] == "NOT_HELD" and ex["E2"]["call"] == "HELD" and ex["E6"]["call"] == "HELD" and ex["E7"]["call"] == "HELD", ex
    assert ex["E3"]["call"] == "NOT_HELD" and ex["E4"]["call"] == "NOT_READ" and ex["E5"]["call"] == "NOT_READ"
    # real shapes: MultiHop-RAG (laptop): the gates, then the engine against set loops on its own graph
    D = P.load_mhr()
    P.gates(D, D.g)
    pops, gs, _src = first_mhr(D)
    assert [len(p.ids) for p in pops] == [1130, 224] and [p.cv for p in pops] == [DECIDE, BESIDE], [(p.cv, len(p.ids)) for p in pops]
    am = anchor_nodes(D, pops)
    U, bi, sizes = P.units_of(D, "RK", am)
    add_all_unit(U, D, "RK", sizes)
    p = pops[0]
    k12 = 12
    S12 = Seeds(D.n, p.golds[:k12], p.seeds["rrf"][:k12])
    full = {"any": np.unique(np.concatenate([D.F["W2_S"], D.F["F0T"], D.F["Ke"]])), "label:R.mention": D.F["W2_S"], "label:K.knn": D.F["Ke"]}
    full["form:HK1"] = D.z["allF"][D.z["rank__HK1"] >= 0]
    full["all"] = np.unique(np.concatenate([full["any"]] + [D.z["allF"][D.z[f"rank__{f}"] >= 0] for f in FORMS]))
    for name, keys in full.items():
        for kind in KINDS:
            got, _v = engine_rows(S12, U[name][1], kind)
            assert np.array_equal(got, brute_seed(D.n, keys, p.golds[:k12], p.seeds["rrf"][:k12], kind)), (name, kind)
    assert U["all"][0] == full["all"].size and U["all"][0] >= U["any"][0]
    St = Seeds(D.n, p.golds, p.seeds["rrf"])
    _cols = {k: kind_cols(St, unit_eval(St, U[k][1]), "seed") for k in ("any", "all")}
    rec, _c = class_record(St, _cols["any"][3], _cols["all"][3], U["all"][1], True)
    assert sum(rec[c]["count"] for c in CLASSES) == int(St.miss.sum()) == rec["missed_golds"]
    assert St.first_counts()["questions"] == 1130
    log("selftest: the engine equals set loops on random graphs (kinds seed, top1, found; incident keys) and on MultiHop-RAG's real graph, "
        "two wrong engines are caught, classes equal their loops, the frontier's cumulative columns equal the union's; gates pass on MultiHop-RAG")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zul2s.py")
    ap.add_argument("cmd", choices=["selftest", "ledger", "report"])
    ap.add_argument("--ds", choices=DS7)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "ledger" and not a.ds:
        stop("--ds is needed")
    return {"selftest": lambda _a: selftest(), "ledger": cmd_ledger, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
