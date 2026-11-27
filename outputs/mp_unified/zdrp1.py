"""DRP1 (docs/DRP1_DROP_SPLADE_AND_K.md): what the pool loses when the SPLADE leg, the K family, or both are dropped, on all ten datasets under one rule.

The seven's cells are LEG1-A6's rule (PL6's needed cells whose filed eval mean nodes are at most 16,000, plus the R95 pick whatever its size), on the eval carve
with the frozen first stage; STaRK's cells are STK3's 16 pick cells, on the test split with e3v. The pool code is PL2 to PL6's own (zpl2g.py to zpl6g.py, zleg1p.py
and zstk3.py are imported and edited nowhere); the arms change ONE thing each:

    nospl : the SPLADE list is the dense list wherever the pool code reads it (zleg1p.dense_only: the seeds become the dense top 5, rrf1000_bN the dense top N)
    nok   : the K family (the 'knn' store of the walk) is empty; S ('structural') and R ('ner') are the filed CSRs, untouched
    both  : the two changes together

    python -E -P -B outputs/mp_unified/zdrp1.py selftest
    python -E -P -B outputs/mp_unified/zdrp1.py cells [--ds D ...]
        # laptop: the seven's cell lists and gate references (outputs/drp1/pool/<ds>/cells_<ds>.json, gate_ref_<ds>.npz) and STaRK's (outputs/drp1/stark_cells.json); the filed
        # numbers beside them, no arm number
    python -E -P -B outputs/mp_unified/zdrp1.py run --ds D [--arm A ...] [--part I K] [--host]
        # one of the seven: G-D0 (intact reproduces the filed arrays of the first 16 questions), G-D1 (the arm's identities), G-D2 (nok leaves every K-free cell equal to the
        # filed one), then the arm's chunks of 128 questions in the declared order (resumable): outputs/drp1/pool/<ds>/<arm>/chunk_<n>.npz
    python -E -P -B outputs/mp_unified/zdrp1.py runk --kb KB [--arm A ...] [--part I K] [--host]
        # one STaRK KB, the same three gates against STK3's chunk of the gate position: outputs/drp1/pool/<kb>/<arm>/chunk_<n>.npz
    python -E -P -B outputs/mp_unified/zdrp1.py read [--ds D ...] [--kb KB ...]
        # laptop: paired reads against the filed arrays (the seven) or STK3's chunks (STaRK); PARTIAL when not every question is covered: outputs/drp1/read_<name>.json

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
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zleg1p as L  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl5g as Z5  # noqa: E402
import zpl6g as Z6  # noqa: E402
import zstk3 as T3  # noqa: E402

DOC = "docs/DRP1_DROP_SPLADE_AND_K.md"
A6_DS = L.DS                                           # the four LEG1-A6 filed the no-SPLADE arm for: metaqa, musique, squad, multihoprag
SEVEN = A6_DS + ("webqsp", "hotpotqa", "2wiki")        # A6's order first, so chunk_order(n, k) of the four is A6's own
KBS = T3.KB_ORDER                                      # prime, mag, amazon
ARMS = ("nospl", "nok", "both")
OUT = ROOT / "outputs" / "drp1"
POOL = OUT / "pool"
K_RE = re.compile(r"\+(FULL|KNN|FS):")                 # the regimes that read the K store: FULL, KNN, and FS (FULL at hop 1)
LIMITS = (500, 1000, 2500, 3500, 5000, 16000)
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc


# ── pure helpers (self-tested) ───────────────────────────────────────────────

def uses_k(name):
    """Does the cell's walk read the K store (regimes FULL, KNN, FS)? Flat cells and STRUCT / NER walks do not."""
    return bool(K_RE.search(name))


def needed_arms(in_a6, names):
    """The arms a dataset needs run. nospl only where A6 did not file it; nok and both only when some cell reads K (else nok = the filed arrays and both = nospl, by construction)."""
    arms = [] if in_a6 else ["nospl"]
    if any(uses_k(n) for n in names):
        arms += ["nok", "both"]
    return arms


class NoK:
    """A Source's families with the 'knn' store emptied (indptr all zero, no columns). The other stores are the base's own objects. The empty CSR is built once."""

    def __init__(self, base):
        self.base, self._k = base, None

    def __getitem__(self, f):
        g = self.base[f]
        if f != "knn":
            return g
        if self._k is None:
            self._k = P2.CSR(np.zeros(np.asarray(g.indptr).shape, dtype=np.asarray(g.indptr).dtype), np.asarray(g.col)[:0])
        return self._k


def variant(src, arm):
    """A shallow copy of the Source for the arm; the original is untouched."""
    d = src
    if arm in ("nospl", "both"):
        d = L.dense_only(d)
    if arm in ("nok", "both"):
        d = copy.copy(d)
        d.fams = NoK(src.fams)
    return d


def arm_fails(src, d, arm):
    """G-D1: the arm's identities. Returns the list of failed statements (empty = PASS)."""
    bad = []
    if arm in ("nospl", "both") and d.splade is not d.dense:
        bad.append("splade is not the dense list")
    if arm in ("nok", "both"):
        if int(np.asarray(d.fams["knn"].indptr)[-1]) != 0 or np.asarray(d.fams["knn"].col).size != 0:
            bad.append("the knn store is not empty")
        for f in ("structural", "ner"):
            if d.fams[f] is not src.fams[f]:
                bad.append(f"the {f} store is not the filed one")
    if arm == "nok" and d.splade is not src.splade:
        bad.append("nok changed the SPLADE list")
    return bad


def sha_names(names):
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def cells_path(ds):
    return POOL / ds / f"cells_{ds}.json"


def ref_path(ds):
    return POOL / ds / f"gate_ref_{ds}.npz"


def chunk_path(name, arm, c):
    return POOL / name / arm / f"chunk_{c:04d}.npz"


STARK_CELLS = OUT / "stark_cells.json"


def common_questions(parts):
    """parts: list of question-index arrays (one per source). The sorted indices every one of them covers."""
    s = None
    for p in parts:
        s = set(map(int, p)) if s is None else s & set(map(int, p))
    return np.asarray(sorted(s or []), dtype=np.int64)


def take(idx, sel):
    """Row positions of the indices `sel` inside the sorted-or-not array `idx`."""
    pos = {int(v): i for i, v in enumerate(idx)}
    return np.asarray([pos[int(v)] for v in sel], dtype=np.int64)


def best_by_limit(rows, key):
    """Per node limit the cell with the highest ALL among those whose mean nodes are within it (post hoc), or None."""
    out = {}
    for lim in LIMITS:
        c = [(v[f"{key}_all"], -v[f"{key}_nodes"], n) for n, v in rows.items() if v[f"{key}_nodes"] <= lim]
        if c:
            a, nn, n = max(c)
            out[str(lim)] = {"name": n, "all": a, "nodes": -nn}
        else:
            out[str(lim)] = None
    return out


# ── cells ────────────────────────────────────────────────────────────────────

def cells_seven(a):
    for k, ds in enumerate(SEVEN):
        if a.ds and ds not in a.ds:
            continue
        sel_path = Z6.PL6 / ds / "select.json"
        sel = rjson(sel_path)
        arr = Z6.eval_arrays(ds, "frozen")
        need = Z6.needed_names(sel)
        miss = [n for n in need if n not in arr]
        if miss:
            stop(f"{ds}: {len(miss)} of PL6's needed cells have no filed eval array, e.g. {miss[0]}")
        r95 = sel["picks"]["R95_all"]["name"]
        if r95 not in need:
            stop(f"{ds}: the R95 pick {r95} is not among PL6's needed cells")
        names = [n for n in need if float(arr[n][1].mean()) <= L.MAX_NODES or n == r95]
        left = [{"name": n, "filed_nodes": float(arr[n][1].mean())} for n in need if n not in names]
        bnd = (sel["picks"].get("R95_bounded") or {}).get("name")
        qids, gc = L.filed_meta(ds)
        nq = len(qids)
        n_chunks = (nq + L.CHUNK - 1) // L.CHUNK
        gq = min(L.GATE_Q, nq)
        if ds in A6_DS:
            a6 = rjson(L.cells_path(ds))
            if a6["names"] != names or a6["chunk_order"] != L.chunk_order(n_chunks, k):
                stop(f"{ds}: the cell list or chunk order differs from LEG1-A6's")
        rec = {"stage": "DRP1 cells", "declared_in": DOC, "dataset": ds, "utc": utc(), "nq": nq, "qids_sha256": L.qids_sha(qids), "chunk": L.CHUNK, "perm_seed": L.PERM_SEED,
               "n_chunks": n_chunks, "chunk_order": L.chunk_order(n_chunks, k), "gate_questions": gq, "max_nodes": L.MAX_NODES, "r95": r95, "bounded": bnd,
               "r95_over_cap": float(arr[r95][1].mean()) > L.MAX_NODES, "select_sha256": hashlib.sha256(sel_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
               "names": names, "uses_k": [n for n in names if uses_k(n)], "arms_to_run": needed_arms(ds in A6_DS, names),
               "filed": {n: {"all": float(arr[n][0].mean()), "nodes": float(arr[n][1].mean())} for n in names}, "left_out": left}
        (POOL / ds).mkdir(parents=True, exist_ok=True)
        wjson(cells_path(ds), rec)
        with open(ref_path(ds), "wb") as f:
            np.savez(f, names=np.asarray(names), allgold=np.stack([arr[n][0][:gq] for n in names], axis=1), size=np.stack([arr[n][1][:gq] for n in names], axis=1), gold_counts=gc[:gq])
        log(f"cells {ds}: {len(names)} cells ({len(rec['uses_k'])} read K, {len(left)} left out above {L.MAX_NODES:.0f} nodes), {nq} questions in {n_chunks} chunks; "
            f"R95 {r95} ({rec['filed'][r95]['nodes']:.0f} nodes); arms to run {rec['arms_to_run']}")


def cells_stark():
    c = rjson(T3.cells_path())
    picks = set(c["groups"]["picks"])
    names = [n for n in c["names"] if n in picks]
    rec = {"stage": "DRP1 stark cells", "declared_in": DOC, "utc": utc(), "names": names, "cells_sha256": sha_names(names), "stk3_cells_sha256": c["cells_sha256"],
           "uses_k": [n for n in names if uses_k(n)], "arms_to_run": needed_arms(False, names)}
    OUT.mkdir(parents=True, exist_ok=True)
    wjson(STARK_CELLS, rec)
    log(f"cells stark: {len(names)} pick cells of STK3's 63 ({len(rec['uses_k'])} read K); arms {rec['arms_to_run']}")


def cmd_cells(a):
    cells_seven(a)
    if not a.ds:
        cells_stark()
    return 0


# ── run ──────────────────────────────────────────────────────────────────────

def write_chunk(path, **arrays):
    tmp = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def kfree_check(op, d, names, lo, hi, ref_all, ref_size):
    """G-D2 (nok): every cell that does not read K equals the filed arrays on questions lo..hi. ref_*: (hi-lo, len(names)) arrays. Returns mismatches [question, cell]."""
    cols = [i for i, n in enumerate(names) if not uses_k(n)]
    if not cols:
        return []
    plan = Z5.Plan5([names[i] for i in cols])
    P, S = Z5.evaluate5(op, d, plan, lo, hi, every=16)
    gc = d.gold_counts[lo:hi]
    bad = L.gate_equal(P, S, gc, ref_all[:, cols], ref_size[:, cols])
    return bad


def arm_gates(op, src, arm, names, lo, hi, ref_all, ref_size, tag, gate_file):
    """G-D1 and G-D2 for one arm, once per (dataset, arm, part); returns True on PASS."""
    gate = rjson(gate_file) if gate_file.is_file() else None
    if gate is not None and gate.get("verdict") == "PASS":
        return True
    t0 = time.time()
    d = variant(src, arm)
    bad1 = arm_fails(src, d, arm)
    bad2 = kfree_check(op, d, names, lo, hi, ref_all, ref_size) if arm == "nok" else []
    gate = {"stage": "DRP1 gates", "declared_in": DOC, "set": tag, "arm": arm, "utc": utc(), "G-D1": {"failed": bad1}, "G-D2": {"checked": arm == "nok", "mismatches": bad2[:10], "n_mismatches": len(bad2)},
            "seconds": round(time.time() - t0, 1), "verdict": "PASS" if not (bad1 or bad2) else "FAIL"}
    wjson(gate_file, gate)
    log(f"{tag} {arm}: gates {gate['verdict']} (G-D1 {bad1 or 'ok'}; G-D2 {len(bad2)} mismatches)")
    return gate["verdict"] == "PASS"


def cmd_run(a):
    Z6.prepare(bool(a.host))
    ds = a.ds
    cells = rjson(cells_path(ds))
    names = cells["names"]
    arms = list(a.arm) if a.arm else cells["arms_to_run"]
    if any(x not in ARMS for x in arms):
        stop(f"unknown arm in {arms}")
    part, nparts = (int(a.part[0]), int(a.part[1])) if a.part else (0, 1)
    op = P2.opened()
    src = P2.load_source(ds, "eval", "frozen")
    if src.nq != cells["nq"] or L.qids_sha([str(x) for x in src.qids]) != cells["qids_sha256"]:
        stop(f"{ds}: the eval carve here is not the one the cell list was cut for")
    gq = int(cells["gate_questions"])
    ref = np.load(ref_path(ds), allow_pickle=False)
    try:
        if [str(x) for x in ref["names"]] != names:
            stop(f"{ds}: the gate reference is not for this cell list")
        ref_all, ref_size = ref["allgold"], ref["size"]
    finally:
        ref.close()
    g0 = POOL / ds / f"gate0_p{part}.json"
    gate = rjson(g0) if g0.is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        t0 = time.time()
        P, S = Z5.evaluate5(op, src, Z5.Plan5(names), 0, gq, every=16)                       # the INTACT source
        bad = L.gate_equal(P, S, src.gold_counts[:gq], ref_all, ref_size)
        gate = {"stage": "DRP1 gates", "declared_in": DOC, "dataset": ds, "part": part, "utc": utc(), "G-D0": {"questions": gq, "cells": len(names), "mismatches": bad[:10], "n_mismatches": len(bad)},
                "seconds": round(time.time() - t0, 1), "verdict": "PASS" if not bad else "FAIL"}
        wjson(g0, gate)
        log(f"{ds}: G-D0 {gate['verdict']} ({len(bad)} mismatches over {gq} questions x {len(names)} cells)")
        if gate["verdict"] != "PASS":
            return 3
    for arm in arms:
        if not arm_gates(op, src, arm, names, 0, gq, ref_all, ref_size, ds, POOL / ds / f"gate_{arm}_p{part}.json"):
            return 3
    plan = Z5.Plan5(names)
    order = cells["chunk_order"]
    done = 0
    for pos in range(part, len(order), nparts):
        c = order[pos]
        lo, hi = L.chunk_range(c, src.nq)
        for arm in arms:
            path = chunk_path(ds, arm, c)
            if path.is_file():
                continue
            d = variant(src, arm)
            t0 = time.time()
            P, S = Z5.evaluate5(op, d, plan, lo, hi, every=64)
            write_chunk(path, lo=lo, hi=hi, names=np.asarray(names), present=P, size=S, seconds=round(time.time() - t0, 1), arm=arm)
            done += 1
            log(f"{ds} {arm}: chunk {c} (questions {lo}-{hi}) done in {time.time() - t0:.0f} s ({done} this run, position {pos + 1} of {len(order)}; {P2.peak_gb()} GB)")
    log(f"{ds}: part {part} of {nparts} finished ({done} chunks made now)")
    return 0


def cmd_runk(a):
    T3.HOSTMODE = bool(a.host)
    kb = a.kb
    sc = rjson(STARK_CELLS)
    names = sc["names"]
    arms = list(a.arm) if a.arm else sc["arms_to_run"]
    if any(x not in ARMS for x in arms):
        stop(f"unknown arm in {arms}")
    part, nparts = (int(a.part[0]), int(a.part[1])) if a.part else (0, 1)
    op, src = T3.load_stark_source(kb)
    nq = src.nq
    n_chunks = (nq + T3.CHUNK - 1) // T3.CHUNK
    order = T3.chunk_order(n_chunks, KBS.index(kb))
    c0 = order[0]
    lo0, hi0 = T3.chunk_range(c0, nq)
    gq = min(L.GATE_Q, hi0 - lo0)
    z = np.load(T3.chunk_path(kb, c0), allow_pickle=False)
    try:
        at = {str(n): i for i, n in enumerate(z["names"])}
        cols = [at[n] for n in names]
        ref_p, ref_s = z["present"][:gq][:, cols].astype(np.int64), z["size"][:gq][:, cols].astype(np.float64)
        ref_gc = z["gold_counts"][:gq].astype(np.int64)
        if int(z["lo"]) != lo0 or sc["stk3_cells_sha256"] != str(z["cells_sha256"]):
            stop(f"{kb}: STK3's chunk {c0} is not the one the cell list was cut for")
    finally:
        z.close()
    ref_all = ref_p == ref_gc[:, None]
    g0 = POOL / kb / f"gate0_p{part}.json"
    gate = rjson(g0) if g0.is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        t0 = time.time()
        P, S = Z5.evaluate5(op, src, Z5.Plan5(names), lo0, lo0 + gq, every=16)
        bad = L.gate_equal(P, S, src.gold_counts[lo0:lo0 + gq], ref_all, ref_s)
        gate = {"stage": "DRP1 gates", "declared_in": DOC, "kb": kb, "part": part, "gate_chunk": c0, "utc": utc(),
                "G-D0": {"questions": gq, "cells": len(names), "mismatches": bad[:10], "n_mismatches": len(bad)}, "seconds": round(time.time() - t0, 1),
                "verdict": "PASS" if not bad else "FAIL"}
        wjson(g0, gate)
        log(f"{kb}: G-D0 {gate['verdict']} ({len(bad)} mismatches over {gq} questions x {len(names)} cells)")
        if gate["verdict"] != "PASS":
            return 3
    for arm in arms:
        if not arm_gates(op, src, arm, names, lo0, lo0 + gq, ref_all, ref_s, kb, POOL / kb / f"gate_{arm}_p{part}.json"):
            return 3
    plan = Z5.Plan5(names)
    done = 0
    for pos in range(part, len(order), nparts):
        c = order[pos]
        lo, hi = T3.chunk_range(c, nq)
        for arm in arms:
            path = chunk_path(kb, arm, c)
            if path.is_file():
                continue
            d = variant(src, arm)
            t0 = time.time()
            P, S = Z5.evaluate5(op, d, plan, lo, hi, every=16)
            write_chunk(path, lo=lo, hi=hi, names=np.asarray(names), present=P, size=S, qids=np.asarray(src.qids[lo:hi], dtype=str), gold_counts=src.gold_counts[lo:hi].astype(np.int32),
                        seconds=round(time.time() - t0, 1), arm=arm, cells_sha256=sc["cells_sha256"], layout_sha256=src.layout_sha256)
            done += 1
            log(f"{kb} {arm}: chunk {c} (questions {lo}-{hi}) done in {time.time() - t0:.0f} s ({done} this run, position {pos + 1} of {len(order)}; {P2.peak_gb()} GB)")
    log(f"{kb}: part {part} of {nparts} finished ({done} chunks made now)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────────

def load_arm_chunks(folder, names, gc_all):
    """The chunks of one arm that exist: (question indices, all-gold bool (Q, C), sizes (Q, C)), or None."""
    idx, allg, size = [], [], []
    for p in sorted(Path(folder).glob("chunk_*.npz")):
        z = np.load(p, allow_pickle=False)
        try:
            if [str(x) for x in z["names"]] != names:
                stop(f"{p.name}: its cells are not the declared list")
            lo, hi = int(z["lo"]), int(z["hi"])
            q = np.arange(lo, hi)
            idx.append(q)
            allg.append(z["present"].astype(np.int64) == gc_all[q][:, None])
            size.append(z["size"].astype(np.float64))
        finally:
            z.close()
    if not idx:
        return None
    return np.concatenate(idx), np.concatenate(allg), np.concatenate(size)


def load_stk3_chunks(kb, names, wanted):
    """STK3's filed chunks restricted to `names` (columns by name) and to the chunk numbers in `wanted`: (idx, all-gold (Q, C), sizes (Q, C), gold_counts)."""
    idx, allg, size, gcs = [], [], [], []
    for c in sorted(wanted):
        p = T3.chunk_path(kb, c)
        if not p.is_file():
            continue
        z = np.load(p, allow_pickle=False)
        try:
            at = {str(n): i for i, n in enumerate(z["names"])}
            cols = [at[n] for n in names]
            gc = z["gold_counts"].astype(np.int64)
            idx.append(np.arange(int(z["lo"]), int(z["hi"])))
            allg.append(z["present"][:, cols].astype(np.int64) == gc[:, None])
            size.append(z["size"][:, cols].astype(np.float64))
            gcs.append(gc)
        finally:
            z.close()
    if not idx:
        return None
    return np.concatenate(idx), np.concatenate(allg), np.concatenate(size), np.concatenate(gcs)


def arm_table(names, filed, arms, rng_key, r95, bounded):
    """filed: name -> (all bool, size) on the common questions; arms: arm -> the same. Per arm the paired loss at every cell, the two headline cells, the reach and the best by limit."""
    out = {}
    for j, arm in enumerate(arms):
        rows = L.compare(names, filed, arms[arm], np.random.default_rng([P2.BOOT_SEED, rng_key, j]))
        rec = {"cells": rows, "reach_filed": L.reach(rows, "filed"), "reach_arm": L.reach(rows, "dense"), "best_filed": best_by_limit(rows, "filed"), "best_arm": best_by_limit(rows, "dense")}
        for key, nm in (("r95_row", r95), ("bounded_row", bounded)):
            rec[key] = {"name": nm, **rows[nm]} if nm in rows else None
        rec["call_r95"] = rows[r95]["call"] if r95 in rows else None
        out[arm] = rec
    return out


def zero_arm(names, filed, label):
    """An arm equal to the filed arrays by construction (nok on a cell list that reads no K): the loss is 0 and the call DISPENSABLE; nothing is estimated."""
    rows = {}
    for n in names:
        a_, s_ = filed[n]
        w = L.wilson(int(a_.sum()), a_.size)
        rows[n] = {"questions": int(a_.size), "filed_all": float(a_.mean()), "filed_all_ci": w, "filed_nodes": float(s_.mean()), "dense_all": float(a_.mean()), "dense_all_ci": w,
                   "dense_nodes": float(s_.mean()), "loss": 0.0, "loss_ci": [0.0, 0.0], "call": "DISPENSABLE", "nodes_dense_over_filed": 1.0, "nodes_dense_over_filed_ci": [1.0, 1.0]}
    return rows


def read_seven(ds, k):
    cells = rjson(cells_path(ds))
    names = cells["names"]
    qids, gc = L.filed_meta(ds)
    if L.qids_sha(qids) != cells["qids_sha256"]:
        stop(f"{ds}: the filed eval grids are not the ones the cell list was cut for")
    for p in sorted((POOL / ds).glob("gate*_p*.json")):
        if rjson(p)["verdict"] != "PASS":
            stop(f"{ds}: {p.name} is not PASS: no arm number is read")
    if not list((POOL / ds).glob("gate0_p*.json")):
        stop(f"{ds}: no gate file: no arm number is read")
    ak = bool(cells["uses_k"])
    got, folders = {}, {"nospl": (L.POOL / ds) if ds in A6_DS else (POOL / ds / "nospl"), "nok": POOL / ds / "nok", "both": POOL / ds / "both"}
    for arm in ARMS:
        if arm == "nok" and not ak:
            continue
        if arm == "both" and not ak:
            continue
        if folders[arm].is_dir():
            g = load_arm_chunks(folders[arm], names, gc)
            if g is not None:
                got[arm] = g
    if not got and ak:
        return {"dataset": ds, "status": "NO_CHUNKS"}
    if not ak and "nospl" not in got:
        return {"dataset": ds, "status": "NO_CHUNKS"}
    idx = common_questions([v[0] for v in got.values()])
    arr = Z6.eval_arrays(ds, "frozen")
    filed = {n: (arr[n][0][idx], arr[n][1][idx]) for n in names}
    arms = {}
    for arm, (ii, al, sz) in got.items():
        sel = take(ii, idx)
        arms[arm] = {n: (al[sel, i], sz[sel, i]) for i, n in enumerate(names)}
    tab = arm_table(names, filed, arms, k, cells["r95"], cells["bounded"])
    by_construction = {}
    if not ak:
        by_construction = {"nok": "no cell reads K: equal to the filed arrays", "both": "no cell reads K: equal to nospl"}
        tab["nok"] = {"cells": zero_arm(names, filed, ds), "by_construction": True}
        tab["both"] = {"by_construction": True, "equals": "nospl"}
    nq = int(cells["nq"])
    rec = {"stage": "DRP1 read", "declared_in": DOC, "dataset": ds, "utc": utc(), "questions_covered": int(idx.size), "questions_total": nq,
           "status": "COMPLETE" if idx.size == nq else "PARTIAL", "r95": cells["r95"], "bounded": cells["bounded"], "r95_over_cap": cells["r95_over_cap"], "arms": tab,
           "by_construction": by_construction, "nospl_source": "LEG1-A6 chunks" if ds in A6_DS else "DRP1 chunks", "left_out": cells["left_out"]}
    wjson(OUT / f"read_{ds}.json", rec)
    for arm, t in tab.items():
        r = t.get("r95_row")
        if r:
            log(f"{ds} {arm} [{rec['status']} {idx.size}/{nq}]: R95 cell {cells['r95']}: filed ALL {r['filed_all']:.4f}, arm ALL {r['dense_all']:.4f}; loss {100 * r['loss']:+.2f} "
                f"[{100 * r['loss_ci'][0]:+.2f}, {100 * r['loss_ci'][1]:+.2f}] {r['call']}")
    return rec


def read_stark(kb):
    sc = rjson(STARK_CELLS)
    names = sc["names"]
    for p in sorted((POOL / kb).glob("gate*_p*.json")):
        if rjson(p)["verdict"] != "PASS":
            stop(f"{kb}: {p.name} is not PASS: no arm number is read")
    if not list((POOL / kb).glob("gate0_p*.json")):
        stop(f"{kb}: no gate file: no arm number is read")
    got, wanted = {}, None
    for arm in ARMS:
        folder = POOL / kb / arm
        if not folder.is_dir():
            continue
        cs = {int(p.stem.split("_")[1]) for p in folder.glob("chunk_*.npz")}
        if cs:
            got[arm] = cs
            wanted = cs if wanted is None else wanted & cs
    if not got or not wanted:
        return {"kb": kb, "status": "NO_CHUNKS"}
    f = load_stk3_chunks(kb, names, wanted)
    if f is None:
        return {"kb": kb, "status": "NO_STK3_CHUNKS"}
    fidx, fall, fsize, fgc = f
    arms = {}
    for arm in got:
        parts = []
        for c in sorted(wanted):
            z = np.load(chunk_path(kb, arm, c), allow_pickle=False)
            try:
                if [str(x) for x in z["names"]] != names or str(z["cells_sha256"]) != sc["cells_sha256"]:
                    stop(f"{kb} {arm} chunk {c}: its cells are not the declared list")
                parts.append((np.arange(int(z["lo"]), int(z["hi"])), z["present"].astype(np.int64) == z["gold_counts"].astype(np.int64)[:, None], z["size"].astype(np.float64)))
            finally:
                z.close()
        ii = np.concatenate([p[0] for p in parts])
        al, sz = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
        if not np.array_equal(ii, fidx):
            stop(f"{kb} {arm}: the arm's questions are not the filed chunks' questions")
        arms[arm] = {n: (al[:, i], sz[:, i]) for i, n in enumerate(names)}
    filed = {n: (fall[:, i], fsize[:, i]) for i, n in enumerate(names)}
    tab = arm_table(names, filed, arms, 10 + KBS.index(kb), None, None)
    zr = np.load(T3.ref_path(kb), allow_pickle=False)
    try:
        nq = int(len(zr["qids"]))
    finally:
        zr.close()
    rec = {"stage": "DRP1 read", "declared_in": DOC, "kb": kb, "utc": utc(), "questions_covered": int(fidx.size), "chunks": sorted(int(c) for c in wanted), "status": "COMPLETE" if fidx.size == nq else "PARTIAL",
           "arms": tab, "questions_total": nq}
    wjson(OUT / f"read_{kb}.json", rec)
    for arm, t in tab.items():
        rows = t["cells"]
        w = max(rows.items(), key=lambda kv: kv[1]["filed_all"] if kv[1]["filed_nodes"] <= 3500 else -1)
        log(f"{kb} {arm} [{fidx.size} questions]: best filed cell within 3,500 nodes {w[0]}: filed ALL {w[1]['filed_all']:.4f}, arm ALL {w[1]['dense_all']:.4f}; loss {100 * w[1]['loss']:+.2f} {w[1]['call']}")
    return rec


def cmd_read(a):
    for k, ds in enumerate(SEVEN):
        if (a.ds or a.kb) and ds not in (a.ds or []):
            continue
        if not cells_path(ds).is_file():
            continue
        read_seven(ds, k)
    for kb in KBS:
        if (a.ds or a.kb) and kb not in (a.kb or []):
            continue
        if not STARK_CELLS.is_file() or not (POOL / kb).is_dir():
            continue
        read_stark(kb)
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def cmd_selftest(a):
    checks, bad = 0, []

    def ok(cond, what):
        nonlocal checks
        checks += 1
        if not cond:
            bad.append(what)

    ok(uses_k("rrf1000_b25+FULL:h1_c100+top25") and uses_k("rrf1000_b50+KNN:h1_c100+top25") and uses_k("rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50"), "FULL, KNN and FS read K")
    ok(not uses_k("rrf1000_b25+STRUCT:h3_c100_v5000") and not uses_k("rrf1000_b15") and not uses_k("dense_top200") and not uses_k("x+NER:h1_c25"), "STRUCT, NER and flat cells do not read K")
    ok(needed_arms(True, ["a+STRUCT:h1_c25"]) == [] and needed_arms(False, ["a+STRUCT:h1_c25"]) == ["nospl"], "no K cell: no nok or both; nospl only outside A6")
    ok(needed_arms(True, ["a+FULL:h1_c25"]) == ["nok", "both"] and needed_arms(False, ["a+FULL:h1_c25"]) == ["nospl", "nok", "both"], "K cells: nok and both; nospl outside A6")
    ok(A6_DS == ("metaqa", "musique", "squad", "multihoprag") and SEVEN[:4] == A6_DS and len(SEVEN) == 7 and len(set(SEVEN)) == 7, "the seven, A6's four first")
    ok(KBS == ("prime", "mag", "amazon"), "STaRK order is STK3's")
    # NoK on a tiny family set
    ip, col = np.asarray([0, 2, 3, 3], dtype=np.int64), np.asarray([1, 2, 0], dtype=np.int32)
    base = P2.Fams(lambda f: P2.CSR(ip, col))
    nk = NoK(base)
    ok(int(nk["knn"].indptr[-1]) == 0 and nk["knn"].col.size == 0 and nk["knn"].indptr.dtype == ip.dtype and nk["knn"].col.dtype == col.dtype, "knn is empty, dtypes kept")
    ok(nk["structural"] is base["structural"] and nk["ner"] is base["ner"], "S and R are the base's own objects")
    ok(nk["knn"] is nk["knn"] and nk["knn"].n_nodes == 3, "the empty CSR is built once and keeps n_nodes")
    nk2 = NoK({"structural": P2.CSR(ip, col), "knn": P2.CSR(ip, col)})
    ok(nk2["structural"].col.size == 3 and nk2["knn"].col.size == 0, "NoK also wraps a plain dict (MultiHop-RAG's families)")

    class Src:
        pass

    s = Src()
    s.dense, s.splade, s.fams = np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3), base
    for arm, spl, kn in (("nospl", True, False), ("nok", False, True), ("both", True, True)):
        d = variant(s, arm)
        ok((d.splade is d.dense) == spl and (d.fams is not s.fams) == kn and s.splade is not s.dense and s.fams is base, f"variant {arm}")
        ok(arm_fails(s, d, arm) == [], f"arm_fails {arm} passes on its own variant")
    ok(arm_fails(s, variant(s, "nospl"), "nok") != [] and arm_fails(s, s, "nospl") != [], "arm_fails catches the wrong variant")
    ok(arm_fails(s, variant(s, "nok"), "nok") == [], "nok keeps the SPLADE list")
    # reads
    rng = np.random.default_rng(1)
    a_ = rng.random(400) < 0.9
    f = {"x": (a_, np.full(400, 100.0))}
    z = zero_arm(["x"], f, "t")
    ok(z["x"]["loss"] == 0.0 and z["x"]["call"] == "DISPENSABLE" and z["x"]["filed_all"] == z["x"]["dense_all"], "zero_arm is the filed arrays")
    arm_all = a_ & (rng.random(400) < 0.8)
    t = arm_table(["x"], f, {"nospl": {"x": (arm_all, np.full(400, 90.0))}}, 0, "x", None)
    r = t["nospl"]["r95_row"]
    ok(abs(r["loss"] - (a_.mean() - arm_all.mean())) < 1e-12 and r["loss_ci"][0] < r["loss"] < r["loss_ci"][1] and r["call"] in ("NEEDED", "UNCLEAR") and t["nospl"]["bounded_row"] is None, "arm_table: the loss is filed minus arm")
    ok(abs(r["nodes_dense_over_filed"] - 0.9) < 1e-9, "arm_table: nodes ratio is arm over filed")
    ok(list(common_questions([np.arange(0, 10), np.arange(5, 20), np.arange(7, 9)])) == [7, 8] and common_questions([np.arange(3)]).tolist() == [0, 1, 2], "common_questions")
    ok(take(np.asarray([4, 5, 6, 7]), np.asarray([6, 4])).tolist() == [2, 0], "take")
    bl = best_by_limit({"a": {"x_all": 0.5, "x_nodes": 400.0}, "b": {"x_all": 0.9, "x_nodes": 3000.0}}, "x")
    ok(bl["500"]["name"] == "a" and bl["3500"]["name"] == "b" and bl["5000"]["all"] == 0.9, "best_by_limit")
    ok(L.chunk_order(19, 1) == L.chunk_order(19, 1) and sorted(L.chunk_order(19, 1)) == list(range(19)), "the chunk order is A6's seeded permutation")
    ok(sha_names(["a", "b"]) != sha_names(["b", "a"]), "names sha is order sensitive")
    print(f"selftest: {checks - len(bad)}/{checks} checks passed" + ("" if not bad else f"; FAILED: {bad}"))
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("cells")
    p.add_argument("--ds", nargs="*")
    p = sub.add_parser("run")
    p.add_argument("--ds", required=True, choices=SEVEN)
    p.add_argument("--arm", nargs="*", choices=ARMS)
    p.add_argument("--part", nargs=2, metavar=("I", "K"))
    p.add_argument("--host", action="store_true")
    p = sub.add_parser("runk")
    p.add_argument("--kb", required=True, choices=KBS)
    p.add_argument("--arm", nargs="*", choices=ARMS)
    p.add_argument("--part", nargs=2, metavar=("I", "K"))
    p.add_argument("--host", action="store_true")
    p = sub.add_parser("read")
    p.add_argument("--ds", nargs="*")
    p.add_argument("--kb", nargs="*")
    a = ap.parse_args(argv)
    return {"selftest": cmd_selftest, "cells": cmd_cells, "run": cmd_run, "runk": cmd_runk, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
