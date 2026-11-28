"""DRP1 amendment 3 (docs/DRP1_DROP_SPLADE_AND_K.md section 14): the dense-only, no-K pool on the DEFINED encoder (e3v), and a name-seed arm for the lexical gap SPLADE left.

The pool code is PL2 to PL6's own (zpl2g.py to zpl6g.py, zleg1p.py, zdrp1.py and zjnt2.py are imported and edited nowhere). The cells, the eval questions and the seeded chunk
order are DRP1's own (outputs/drp1/pool/<ds>/cells_<ds>.json), so every number pairs with section 13's on the same questions. Two arms, both with the K store empty:

    both  : the SPLADE slot is the e3v dense list (the seeds are the e3v dense top 5)         [DRP1's `both` arm on the e3v source]
    names : the SPLADE slot is the question's name seeds (zjnt2.qn_extra) followed by the e3v dense list in its own order without them, cut to the dense width

    python -E -P -B outputs/mp_unified/zdrp2.py selftest
    python -E -P -B outputs/mp_unified/zdrp2.py names --ds D [--host] [--threads 1]
        # builds the names slot of the eval carve (outputs/drp1e/pool/<ds>/names_slot.npz) and G-E0 and G-N1 (gate_names.json)
    python -E -P -B outputs/mp_unified/zdrp2.py run --ds D [--arm both|names ...] [--part I K] [--slice N] [--host] [--threads 1]
        # G-E2 and G-D1 once per part, then the first N chunks of DRP1's seeded order (all when --slice is absent): outputs/drp1e/pool/<ds>/<arm>/chunk_<n>.npz
    python -E -P -B outputs/mp_unified/zdrp2.py read [--ds D ...]
        # laptop or host: the paired reads (outputs/drp1e/read_<ds>.json); a slice is named as a slice

Run from the workspace root.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zdrp1 as D1  # noqa: E402
import zjnt2 as J2  # noqa: E402
import zleg1p as L  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl5g as Z5  # noqa: E402
import zpl6g as Z6  # noqa: E402

DOC = "docs/DRP1_DROP_SPLADE_AND_K.md"
DS = ("squad", "musique", "metaqa", "webqsp", "hotpotqa", "2wiki", "multihoprag")   # the six ENC E0 covers, and MultiHop-RAG's own e3v first-stage record
ARMS = ("both", "names")
OUT = ROOT / "outputs" / "drp1e"
POOL = OUT / "pool"
LINE = 0.01                                            # one point of ALL: the ABOVE / BELOW line of the names-minus-both read
LIMS = (500, 1000, 2500, 3500)
SHARE_FULL, SHARE_PART, GAP_MIN = 0.90, 0.25, 0.01     # section 14: RECOVERED / PARTIAL / NO_GAP
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc


# ── pure helpers (self-tested) ───────────────────────────────────────────────

def build_names_slot(dense, qn):
    """The SPLADE slot of the `names` arm. dense: (nq, W) ints; qn: per question the name seeds (node ids). A question's row is its name seeds (those the dense list ranks come first
    in the dense order, the rest by node id) followed by the dense row without them, cut to W. A question without a name seed gets the dense row itself. Returns (slot, n_names)."""
    slot = np.array(dense, copy=True)
    nn = np.zeros(len(dense), dtype=np.int64)
    for i in range(len(dense)):
        q = np.asarray(qn[i], dtype=np.int64).ravel()
        if q.size == 0:
            continue
        row = np.asarray(dense[i], dtype=np.int64)
        pos = {int(v): j for j, v in enumerate(row) if v >= 0}
        names = {int(v) for v in q}
        inlist = sorted((v for v in names if v in pos), key=lambda v: pos[v])
        first = inlist + sorted(v for v in names if v not in pos)
        keep = [int(v) for v in row if int(v) not in names]
        slot[i] = np.asarray(first + keep, dtype=np.int64)[:row.size].astype(slot.dtype)
        nn[i] = len(first)
    return slot, nn


def names_fails(dense, slot, qn):
    """G-N1: returns the list of failed statements (empty = PASS)."""
    bad = []
    dense, slot = np.asarray(dense), np.asarray(slot)
    if dense.shape != slot.shape or dense.dtype != slot.dtype:
        return ["the slot's shape or dtype is not the dense list's"]
    for i in range(len(dense)):
        row, s = dense[i].astype(np.int64), slot[i].astype(np.int64)
        v = s[s >= 0]
        if np.unique(v).size != v.size:
            bad.append(f"q{i}: a repeated id")
            continue
        q = {int(x) for x in np.asarray(qn[i]).ravel()}
        if not q:
            if not np.array_equal(row, s):
                bad.append(f"q{i}: no name seed but the row is not the dense row")
            continue
        k = len(q)
        if {int(x) for x in s[:k]} != q:
            bad.append(f"q{i}: the first {k} entries are not the name seeds")
            continue
        rest = [int(x) for x in row if int(x) not in q]
        if [int(x) for x in s[k:]] != rest[:s.size - k]:
            bad.append(f"q{i}: the rest is not the dense order without the name seeds")
            continue
        seeds = {int(x) for x in row[:5]} | {int(x) for x in s[:5]}
        if not q <= seeds:
            bad.append(f"q{i}: a name seed is not among the seeds")
    return bad


def gain_call(ci):
    """names minus both, a fraction, its paired interval: ABOVE if the lower end is above +1.0 point, BELOW if the upper end is below -1.0 point, else LEVEL."""
    if ci[0] > LINE:
        return "ABOVE"
    if ci[1] < -LINE:
        return "BELOW"
    return "LEVEL"


def best_within(rows, key, lim):
    """The cell with the highest ALL among those whose mean nodes are within lim (ties: fewer nodes), post hoc; (name, all, nodes) or None."""
    c = [(v[f"{key}_all"], -v[f"{key}_nodes"], n) for n, v in rows.items() if v[f"{key}_nodes"] <= lim]
    if not c:
        return None
    a, nn, n = max(c)
    return {"name": n, "all": a, "nodes": -nn}


def gap_call(filed_best, both_best, names_best):
    """Section 14's gap call: gap = filed best minus `both`'s best within the limit; share = (names - both) / gap."""
    gap = filed_best - both_best
    if gap < GAP_MIN:
        return {"call": "NO_GAP", "gap": gap, "share": None}
    share = (names_best - both_best) / gap
    return {"call": "RECOVERED" if share >= SHARE_FULL else "PARTIAL" if share >= SHARE_PART else "NOT_RECOVERED", "gap": gap, "share": share}


def rows_of(names, a, b):
    """name -> {a_all, a_nodes, b_all, b_nodes} on the common questions; a, b: name -> (all bool, size)."""
    return {n: {"a_all": float(a[n][0].mean()), "a_nodes": float(a[n][1].mean()), "b_all": float(b[n][0].mean()), "b_nodes": float(b[n][1].mean())} for n in names}


# ── files ────────────────────────────────────────────────────────────────────

def cells_of(ds):
    p = D1.cells_path(ds)
    if not p.is_file():
        stop(f"{ds}: no DRP1 cell list ({p.name})")
    return rjson(p)


def slot_path(ds):
    return POOL / ds / "names_slot.npz"


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def question_texts(ds, src):
    """The eval carve's question texts in the source's order; every query id is checked."""
    if ds == "multihoprag":
        import zul2m as M2
        root = M2.M.mhr_root()
        by_id = {str(x["query_id"]): x["question"] for x in M2.M.queries_of(root)}
        miss = [i for i in src.qids if str(i) not in by_id]
        if miss:
            stop(f"{ds}: {len(miss)} eval ids are not query ids (first {miss[0]!r})")
        return [str(by_id[str(i)]) for i in src.qids]
    root = J2.E.served_root()
    qids = json.loads((root / ds / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    rows = [int(r) for r in src.idx]
    if [str(qids[r]) for r in rows] != [str(x) for x in src.qids]:
        stop(f"{ds}: the eval carve's rows are not its query ids")
    qt = J2.E.query_texts(root, ds, rows)
    return [str(qt[r]) for r in rows]


def g_e0(ds, src, cells):
    """G-E0: the source's first stage is E0's e3v list (its file is the one E0's record names) and the eval carve is the one the cell list was cut for."""
    bad = []
    if src.first != "e3v":
        bad.append("the source's first stage is not e3v")
    if src.nq != cells["nq"] or L.qids_sha([str(x) for x in src.qids]) != cells["qids_sha256"]:
        bad.append("the eval carve is not the one the cell list was cut for")
    if ds != "multihoprag":
        e0 = P2.E0_DIR / f"{ds}.npz"
        rec = rjson(P2.E0_DIR / f"{ds}.json")
        if rec.get("enc") != "e3v":
            bad.append("E0's record is not e3v's")
        if sha_file(e0) != rec.get("npz_sha256"):
            bad.append("E0's e3v npz is not the file its record names")
    if np.asarray(src.dense).shape[1] < min(1000, int(src.n)):
        bad.append("the dense list is shallower than 1,000 (or than the KB)")
    return bad


# ── names ────────────────────────────────────────────────────────────────────

def cmd_names(a):
    Z6.prepare(bool(a.host))
    ds = a.ds
    cells = cells_of(ds)
    src = P2.load_source(ds, "eval", "e3v")
    bad0 = g_e0(ds, src, cells)
    gate = {"stage": "DRP1 amendment 3 gates", "declared_in": DOC, "dataset": ds, "utc": utc(), "G-E0": {"failed": bad0}}
    if bad0:
        gate["verdict"] = "FAIL"
        wjson(POOL / ds / "gate_names.json", gate)
        stop(f"{ds}: G-E0 failed: {bad0}")
    qt = question_texts(ds, src)
    (POOL / ds).mkdir(parents=True, exist_ok=True)
    sl = {"ids": [str(x) for x in src.qids], "qtext": qt}
    qn, _gr, info = J2.qn_extra(ds, sl, src.n, POOL / ds)
    dense = np.asarray(src.dense)
    slot, nn = build_names_slot(dense, qn)
    bad1 = names_fails(dense, slot, qn)
    gate["G-N1"] = {"failed": bad1[:10], "n_failed": len(bad1)}
    gate["verdict"] = "PASS" if not bad1 else "FAIL"
    gate["names"] = {"questions": int(len(qn)), "with_a_name_seed": int((nn > 0).sum()), "mean_name_seeds": float(nn.mean()), "info": info}
    wjson(POOL / ds / "gate_names.json", gate)
    if bad1:
        stop(f"{ds}: G-N1 failed: {bad1[:3]}")
    with open(slot_path(ds), "wb") as f:
        np.savez(f, slot=slot, n_names=nn, qsha=np.array(cells["qids_sha256"]), dense_sha=np.array(sha_arr(dense)))
    log(f"{ds}: names slot built for {len(qn)} questions: {int((nn > 0).sum())} with a name seed, mean {nn.mean():.2f} per question; G-E0 and G-N1 PASS")
    return 0


def load_slot(ds, src, cells):
    z = np.load(slot_path(ds), allow_pickle=False)
    try:
        if str(z["qsha"]) != cells["qids_sha256"] or str(z["dense_sha"]) != sha_arr(np.asarray(src.dense)):
            stop(f"{ds}: the names slot is not for this eval carve and this dense list")
        return z["slot"]
    finally:
        z.close()


# ── run ──────────────────────────────────────────────────────────────────────

def variant(src, arm, slot):
    d = D1.variant(src, "both")                        # the SPLADE slot is the dense list, K empty
    if arm == "names":
        d = copy.copy(d)
        d.splade = slot
    return d


def arm_fails(src, d, arm):
    """G-D1: the arm's identities; empty = PASS."""
    bad = []
    if int(np.asarray(d.fams["knn"].indptr)[-1]) != 0 or np.asarray(d.fams["knn"].col).size != 0:
        bad.append("the knn store is not empty")
    for f in ("structural", "ner"):
        if d.fams[f] is not src.fams[f]:
            bad.append(f"the {f} store is not the filed one")
    if arm == "both" and d.splade is not d.dense:
        bad.append("both: the SPLADE slot is not the dense list")
    if arm == "names" and (d.splade is d.dense or d.splade is src.splade):
        bad.append("names: the SPLADE slot is the dense list or the older SPLADE list")
    return bad


def cmd_run(a):
    Z6.prepare(bool(a.host))
    ds = a.ds
    cells = cells_of(ds)
    names = cells["names"]
    arms = list(a.arm) if a.arm else list(ARMS)
    if any(x not in ARMS for x in arms):
        stop(f"unknown arm in {arms}")
    part, nparts = (int(a.part[0]), int(a.part[1])) if a.part else (0, 1)
    op = P2.opened()
    src = P2.load_source(ds, "eval", "e3v")
    bad0 = g_e0(ds, src, cells)
    if bad0:
        stop(f"{ds}: G-E0 failed: {bad0}")
    gn = POOL / ds / "gate_names.json"
    if not gn.is_file() or rjson(gn).get("verdict") != "PASS":
        stop(f"{ds}: no PASS gate_names.json: run `names` first")
    slot = load_slot(ds, src, cells) if "names" in arms else None
    gq = int(cells["gate_questions"])
    ge = POOL / ds / f"gate_run_p{part}.json"
    gate = rjson(ge) if ge.is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        t0 = time.time()
        bad1 = []
        for arm in ARMS:
            bad1 += [f"{arm}: {x}" for x in arm_fails(src, variant(src, arm, load_slot(ds, src, cells) if arm == "names" else None), arm)]
        nok = D1.variant(src, "nok")                    # the intact SPLADE slot, K emptied
        kf = [i for i, n in enumerate(names) if not D1.uses_k(n)]
        bad2 = []
        if kf:
            plan = Z5.Plan5([names[i] for i in kf])
            P0, S0 = Z5.evaluate5(op, src, plan, 0, gq, every=16)
            P1, S1 = Z5.evaluate5(op, nok, plan, 0, gq, every=16)
            gc = src.gold_counts[:gq]
            bad2 = L.gate_equal(P1, S1, gc, P0.astype(np.int64) == np.asarray(gc, dtype=np.int64)[:, None], S0)
        gate = {"stage": "DRP1 amendment 3 gates", "declared_in": DOC, "dataset": ds, "part": part, "utc": utc(), "G-D1": {"failed": bad1},
                "G-E2": {"checked": bool(kf), "k_free_cells": len(kf), "questions": gq, "n_mismatches": len(bad2), "mismatches": bad2[:10]},
                "seconds": round(time.time() - t0, 1), "verdict": "PASS" if not (bad1 or bad2) else "FAIL"}
        wjson(ge, gate)
        log(f"{ds}: gates {gate['verdict']} (G-D1 {bad1 or 'ok'}; G-E2 {len(bad2)} mismatches over {len(kf)} K-free cells)")
        if gate["verdict"] != "PASS":
            return 3
    plan = Z5.Plan5(names)
    order = cells["chunk_order"]
    top = len(order) if a.slice is None else min(int(a.slice), len(order))
    done = 0
    for pos in range(part, top, nparts):
        c = order[pos]
        lo, hi = L.chunk_range(c, src.nq)
        for arm in arms:
            path = POOL / ds / arm / f"chunk_{c:04d}.npz"
            if path.is_file():
                continue
            d = variant(src, arm, slot)
            t0 = time.time()
            P, S = Z5.evaluate5(op, d, plan, lo, hi, every=64)
            D1.write_chunk(path, lo=lo, hi=hi, names=np.asarray(names), present=P, size=S, seconds=round(time.time() - t0, 1), arm=arm)
            done += 1
            log(f"{ds} {arm}: chunk {c} (questions {lo}-{hi}) done in {time.time() - t0:.0f} s ({done} this run, position {pos + 1} of {top}; {P2.peak_gb()} GB)")
    log(f"{ds}: part {part} of {nparts} finished ({done} chunks made now)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────────

def read_ds(ds, k):
    cells = cells_of(ds)
    names = cells["names"]
    for p in [POOL / ds / "gate_names.json"] + sorted((POOL / ds).glob("gate_run_p*.json")):
        if not p.is_file() or rjson(p).get("verdict") != "PASS":
            stop(f"{ds}: {p.name} is not a PASS gate file: no arm number is read")
    if not list((POOL / ds).glob("gate_run_p*.json")):
        stop(f"{ds}: no run gate file")
    qids, gc = L.filed_meta(ds)
    if L.qids_sha(qids) != cells["qids_sha256"]:
        stop(f"{ds}: the filed eval grids are not the ones the cell list was cut for")
    got = {}
    for arm in ARMS:
        if (POOL / ds / arm).is_dir():
            g = D1.load_arm_chunks(POOL / ds / arm, names, gc)
            if g is not None:
                got[arm] = g
    if set(got) != set(ARMS):
        return {"dataset": ds, "status": "NO_CHUNKS"}
    old_folder = D1.POOL / ds / "both" if cells["uses_k"] else (L.POOL / ds if ds in D1.A6_DS else D1.POOL / ds / "nospl")
    old = D1.load_arm_chunks(old_folder, names, gc) if old_folder.is_dir() else None
    old_label = "both" if cells["uses_k"] else "nospl (no cell reads K: equal to both)"
    if cells["uses_k"] and ds in ("metaqa",):
        old = None                                      # DRP1 did not run MetaQA's `both` arm
    idx = D1.common_questions([v[0] for v in got.values()])
    arr = Z6.eval_arrays(ds, "frozen")
    filed = {n: (arr[n][0][idx], arr[n][1][idx]) for n in names}
    arms = {}
    for arm, (ii, al, sz) in got.items():
        sel = D1.take(ii, idx)
        arms[arm] = {n: (al[sel, i], sz[sel, i]) for i, n in enumerate(names)}
    if old is not None and set(map(int, idx)) <= set(map(int, old[0])):
        sel = D1.take(old[0], idx)
        arms["older_both"] = {n: (old[1][sel, i], old[2][sel, i]) for i, n in enumerate(names)}
    tab = D1.arm_table(names, filed, arms, k, cells["r95"], cells["bounded"])
    rng = np.random.default_rng([P2.BOOT_SEED, 100 + k])
    nb = L.compare(names, arms["both"], arms["names"], rng)             # loss = both - names: the gain is its negative
    gain = {}
    for n, r in nb.items():
        ci = [-r["loss_ci"][1], -r["loss_ci"][0]]
        gain[n] = {"names_all": r["dense_all"], "both_all": r["filed_all"], "gain": -r["loss"], "gain_ci": ci, "call": gain_call(ci), "nodes_names_over_both": r["nodes_dense_over_filed"]}
    best = {}
    rows_fb, rows_bn = rows_of(names, filed, arms["both"]), rows_of(names, arms["both"], arms["names"])
    for lim in LIMS:
        bf = best_within({n: {"f_all": v["a_all"], "f_nodes": v["a_nodes"]} for n, v in rows_fb.items()}, "f", lim)
        bb = best_within({n: {"b_all": v["b_all"], "b_nodes": v["b_nodes"]} for n, v in rows_fb.items()}, "b", lim)
        bn = best_within({n: {"n_all": v["b_all"], "n_nodes": v["b_nodes"]} for n, v in rows_bn.items()}, "n", lim)
        if not (bf and bb and bn):
            best[str(lim)] = None
            continue
        pr = P2.paired(arms["names"][bn["name"]], arms["both"][bb["name"]], np.random.default_rng([P2.BOOT_SEED, 200 + k, lim]))
        rec = {"filed": bf, "both": bb, "names": bn, "names_minus_both": pr["all_diff"], "names_minus_both_ci": pr["all_diff_ci"], "call": gain_call(pr["all_diff_ci"]),
               "gap": gap_call(bf["all"], bb["all"], bn["all"])}
        if "older_both" in arms:
            ob = best_within({n: {"o_all": float(arms["older_both"][n][0].mean()), "o_nodes": float(arms["older_both"][n][1].mean())} for n in names}, "o", lim)
            if ob:
                pe = P2.paired(arms["both"][bb["name"]], arms["older_both"][ob["name"]], np.random.default_rng([P2.BOOT_SEED, 300 + k, lim]))
                rec["older_both"] = ob
                rec["e3v_minus_older"] = pe["all_diff"]
                rec["e3v_minus_older_ci"] = pe["all_diff_ci"]
        best[str(lim)] = rec
    nq = int(cells["nq"])
    rec = {"stage": "DRP1 amendment 3 read", "declared_in": DOC, "dataset": ds, "utc": utc(), "questions_covered": int(idx.size), "questions_total": nq,
           "status": "COMPLETE" if idx.size == nq else "SLICE", "chunks": int(idx.size // L.CHUNK + (1 if idx.size % L.CHUNK else 0)), "older_both_is": old_label if "older_both" in arms else None,
           "r95": cells["r95"], "arms": tab, "names_minus_both": gain, "best_within": best}
    wjson(OUT / f"read_{ds}.json", rec)
    for lim in (2500,):
        b = best.get(str(lim))
        if b:
            log(f"{ds} [{rec['status']} {idx.size}/{nq}]: best within {lim} nodes: filed {100 * b['filed']['all']:.1f} at {b['filed']['nodes']:.0f}; both(e3v) {100 * b['both']['all']:.1f} at "
                f"{b['both']['nodes']:.0f}; names {100 * b['names']['all']:.1f} at {b['names']['nodes']:.0f}; names-both {100 * b['names_minus_both']:+.2f} "
                f"[{100 * b['names_minus_both_ci'][0]:+.2f}, {100 * b['names_minus_both_ci'][1]:+.2f}] {b['call']}; gap {b['gap']['call']}")
    return rec


def cmd_read(a):
    for k, ds in enumerate(DS):
        if a.ds and ds not in a.ds:
            continue
        if not (POOL / ds).is_dir():
            continue
        read_ds(ds, k)
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def cmd_selftest(a):
    checks, bad = 0, []

    def ok(cond, what):
        nonlocal checks
        checks += 1
        if not cond:
            bad.append(what)

    dense = np.asarray([[10, 11, 12, 13, 14, 15, 16, 17], [20, 21, 22, 23, 24, 25, 26, 27], [30, 31, 32, 33, 34, 35, 36, 37]], dtype=np.int32)
    qn = [np.asarray([12, 99], np.int64), np.zeros(0, np.int64), np.asarray([37, 31, 55], np.int64)]
    slot, nn = build_names_slot(dense, qn)
    ok(slot[0].tolist() == [12, 99, 10, 11, 13, 14, 15, 16], "a name seed in the list comes first, one outside it after, then the dense order without them")
    ok(slot[1].tolist() == dense[1].tolist() and nn.tolist() == [2, 0, 3], "a question without a name seed keeps the dense row")
    ok(slot[2].tolist() == [31, 37, 55, 30, 32, 33, 34, 35] and slot.dtype == dense.dtype, "in-list names in the dense order (31 before 37), then outside ones, dtype kept")
    ok(names_fails(dense, slot, qn) == [], "G-N1 passes on a correct slot")
    broken = slot.copy()
    broken[0, 3] = broken[0, 2]
    ok(names_fails(dense, broken, qn) != [], "G-N1 catches a repeated id")
    broken = slot.copy()
    broken[1, 0], broken[1, 1] = broken[1, 1], broken[1, 0]
    ok(names_fails(dense, broken, qn) != [], "G-N1 catches a changed no-name row")
    broken = slot.copy()
    broken[0, 0], broken[0, 2] = broken[0, 2], broken[0, 0]
    ok(names_fails(dense, broken, qn) != [], "G-N1 catches name seeds that are not first")
    ok(names_fails(dense, slot.astype(np.int64), qn) != [], "G-N1 catches a dtype change")
    padded = np.asarray([[5, 6, 7, -1, -1]], dtype=np.int32)
    s2, n2 = build_names_slot(padded, [np.asarray([7], np.int64)])
    ok(s2[0].tolist() == [7, 5, 6, -1, -1] and n2.tolist() == [1], "a padded row keeps its padding")
    ok(names_fails(padded, s2, [np.asarray([7], np.int64)]) == [], "G-N1 passes on a padded row")
    ok(gain_call([0.012, 0.03]) == "ABOVE" and gain_call([-0.03, -0.011]) == "BELOW" and gain_call([-0.005, 0.02]) == "LEVEL" and gain_call([0.0, 0.01]) == "LEVEL", "the names-minus-both call")
    rows = {"a": {"x_all": 0.5, "x_nodes": 400.0}, "b": {"x_all": 0.9, "x_nodes": 3000.0}, "c": {"x_all": 0.9, "x_nodes": 2000.0}}
    ok(best_within(rows, "x", 500)["name"] == "a" and best_within(rows, "x", 3500)["name"] == "c" and best_within(rows, "x", 100) is None, "best_within takes the highest ALL, then fewer nodes")
    ok(gap_call(0.966, 0.882, 0.93)["call"] == "PARTIAL" and gap_call(0.966, 0.882, 0.96)["call"] == "RECOVERED" and gap_call(0.966, 0.882, 0.89)["call"] == "NOT_RECOVERED"
       and gap_call(0.90, 0.895, 0.95)["call"] == "NO_GAP", "the gap call")
    ok(abs(gap_call(0.966, 0.882, 0.93)["share"] - (0.048 / 0.084)) < 1e-9, "the recovered share")
    ok(DS[-1] == "multihoprag" and len(set(DS)) == 7, "the datasets")

    class Src:
        pass

    s = Src()
    s.dense, s.splade = dense, dense + 1
    ip, col = np.asarray([0, 2, 3, 3], dtype=np.int64), np.asarray([1, 2, 0], dtype=np.int32)
    s.fams = P2.Fams(lambda f: P2.CSR(ip, col))
    for arm in ARMS:
        d = variant(s, arm, slot if arm == "names" else None)
        ok(arm_fails(s, d, arm) == [], f"G-D1 passes on {arm}")
    ok(arm_fails(s, variant(s, "both", None), "names") != [] and arm_fails(s, variant(s, "names", slot), "both") != [], "G-D1 tells the arms apart")
    ok(arm_fails(s, s, "both") != [], "G-D1 catches an intact source")
    print(f"selftest: {checks - len(bad)}/{checks} checks passed" + ("" if not bad else f"; FAILED: {bad}"))
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("names")
    p.add_argument("--ds", required=True, choices=DS)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p = sub.add_parser("run")
    p.add_argument("--ds", required=True, choices=DS)
    p.add_argument("--arm", nargs="*")
    p.add_argument("--part", nargs=2, metavar=("I", "K"))
    p.add_argument("--slice", type=int, default=None)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p = sub.add_parser("read")
    p.add_argument("--ds", nargs="*")
    a = ap.parse_args(argv)
    return {"selftest": cmd_selftest, "names": cmd_names, "run": cmd_run, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
