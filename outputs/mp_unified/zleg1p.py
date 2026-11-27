"""LEG1-A6 (docs/LEG1_LEG_NECESSITY.md section 15): the pool read without the SPLADE leg, on MetaQA, MuSiQue, SQuAD and MultiHop-RAG.

PL6's filed cells of each dataset, on its eval carve with the frozen first stage, built by PL5's and PL6's own code with ONE change: the SPLADE list is the dense
list wherever the pool code reads it (the seeds become the dense top 5; `rrf1000_bN` becomes the dense top N; `splade_*` cells equal their dense cells). zpl2g.py to
zpl6g.py are imported and edited nowhere.

    python -E -P -B outputs/mp_unified/zleg1p.py selftest
    python -E -P -B outputs/mp_unified/zleg1p.py cells [--ds D ...]
        # laptop: the declared cell list per dataset (PL6's needed cells whose filed eval mean nodes are <= 16,000), the filed numbers beside it, the chunk order,
        # and the gate reference (the filed arrays of the first 16 questions): outputs/leg1/pool/<ds>/cells_<ds>.json, gate_ref_<ds>.npz. No dense-only number.
    python -E -P -B outputs/mp_unified/zleg1p.py run --ds D [--part I K] [--host]
        # G-D0 (the intact SPLADE list reproduces the filed arrays of the first 16 questions, a stop otherwise), G-D1, then the dense-only chunks of 128
        # questions in the declared order (parts take every K-th position), each written when done (resumable): outputs/leg1/pool/<ds>/chunk_<n>.npz
    python -E -P -B outputs/mp_unified/zleg1p.py read [--ds D ...]
        # laptop: the paired read of the chunks that exist against the filed arrays; PARTIAL when not every question is covered: read_<ds>.json, pool_dense_only.json

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
import math  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zpl2g as P2  # noqa: E402
import zpl3g as Z3  # noqa: E402,F401
import zpl5g as Z5  # noqa: E402
import zpl6g as Z6  # noqa: E402

DOC = "docs/LEG1_LEG_NECESSITY.md section 15 (LEG1-A6)"
DS = ("metaqa", "musique", "squad", "multihoprag")
POOL = ROOT / "outputs" / "leg1" / "pool"
CHUNK = 128                                   # questions per chunk
PERM_SEED = 20261127                          # the chunk order: a seeded permutation of the chunk numbers
GATE_Q = 16                                   # G-D0: the first questions graded with the intact SPLADE list
MAX_NODES = 16000.0                           # the cell rule: PL6's needed cells whose filed eval mean nodes are at most this
LINE = 0.01                                   # the NEEDED / DISPENSABLE line: 1.0 point of ALL
log, stop, rjson, wjson, utc = P2.log, P2.stop, P2.rjson, P2.wjson, P2.utc


# ── pure helpers (self-tested) ───────────────────────────────────────────────

def chunk_order(n_chunks, k):
    """The declared order of the chunk numbers for the dataset with index k: a seeded permutation (a prefix is then a spread sample, whatever the question order)."""
    return [int(c) for c in np.random.default_rng([PERM_SEED, k]).permutation(n_chunks)]


def chunk_range(c, nq):
    return c * CHUNK, min((c + 1) * CHUNK, nq)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [mid - half, mid + half]


def call_loss(ci):
    """Section 2's convention on the loss (filed ALL minus dense-only ALL, a fraction): NEEDED if the lower end is at least +1.0 point, DISPENSABLE if the upper end is below +1.0."""
    if ci[0] >= LINE:
        return "NEEDED"
    if ci[1] < LINE:
        return "DISPENSABLE"
    return "UNCLEAR"


def dense_only(src):
    """A shallow copy of a Source whose SPLADE list is its dense list (the one change of LEG1-A6); the original is untouched."""
    d = copy.copy(src)
    d.splade = src.dense
    return d


def gate_equal(P, S, gold_counts, ref_allgold, ref_size):
    """G-D0: the graded arrays (present counts, sizes) of the intact source equal the filed ones for every cell. Returns the list of (question, cell) mismatches."""
    allgold = P.astype(np.int64) == np.asarray(gold_counts, dtype=np.int64)[:, None]
    bad_a = np.argwhere(allgold != ref_allgold)
    bad_s = np.argwhere(S.astype(np.float64) != np.asarray(ref_size, dtype=np.float64))
    return [[int(q), int(c), "all"] for q, c in bad_a] + [[int(q), int(c), "size"] for q, c in bad_s]


def compare(names, filed, dense, rng):
    """filed, dense: name -> (all-gold bool, size) on the same questions. Per cell the loss (filed ALL minus dense-only ALL), its paired bootstrap, and both systems' ALL with Wilson."""
    out = {}
    for n in names:
        f, d = filed[n], dense[n]
        q = f[0].size
        pr = P2.paired(f, d, rng)                                   # new = filed, old = dense-only: all_diff is the loss
        lo, hi = pr["nodes_ratio_ci"]
        out[n] = {"questions": int(q),
                  "filed_all": float(f[0].mean()), "filed_all_ci": wilson(int(f[0].sum()), q), "filed_nodes": float(f[1].mean()),
                  "dense_all": float(d[0].mean()), "dense_all_ci": wilson(int(d[0].sum()), q), "dense_nodes": float(d[1].mean()),
                  "loss": pr["all_diff"], "loss_ci": pr["all_diff_ci"], "call": call_loss(pr["all_diff_ci"]),
                  "nodes_dense_over_filed": 1.0 / pr["nodes_ratio"], "nodes_dense_over_filed_ci": [1.0 / hi, 1.0 / lo]}
    return out


def reach(rows, key):
    """Among the cells, the one with the smallest mean nodes whose ALL is at least 0.95 (a post-hoc read, not a selection); None when no cell reaches."""
    ok = [(v[f"{key}_nodes"], n, v[f"{key}_all"]) for n, v in rows.items() if v[f"{key}_all"] >= Z6.TARGET]
    if not ok:
        return None
    nodes, name, al = min(ok)
    return {"name": name, "all": al, "nodes": nodes}


# ── files ────────────────────────────────────────────────────────────────────

def cells_path(ds):
    return POOL / ds / f"cells_{ds}.json"


def ref_path(ds):
    return POOL / ds / f"gate_ref_{ds}.npz"


def chunk_path(ds, c):
    return POOL / ds / f"chunk_{c:04d}.npz"


def filed_meta(ds):
    """qids and gold counts of the filed eval grids (PL5's confirmation grid and PL6's, one question order)."""
    qids = gc = None
    for g in (Z6.pl5_confirm_grid(ds, "frozen"), (Z5.load_grid5(Z6.PL6 / ds / "grid_eval_frozen_confirm")
                                                  if Path(str(Z6.PL6 / ds / "grid_eval_frozen_confirm") + ".npz").is_file() else None)):
        if g is None:
            continue
        if qids is None:
            qids, gc = [str(x) for x in g["qids"]], np.asarray(g["gold_counts"], dtype=np.int64)
        elif [str(x) for x in g["qids"]] != qids or not np.array_equal(gc, g["gold_counts"]):
            stop(f"{ds}: PL5's and PL6's eval grids are not on the same questions")
    if qids is None:
        stop(f"{ds}: no filed eval grid here")
    return qids, gc


def qids_sha(qids):
    return hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest()


# ── cells ────────────────────────────────────────────────────────────────────

def cmd_cells(a):
    for k, ds in enumerate(DS):
        if a.ds and ds not in a.ds:
            continue
        sel_path = Z6.PL6 / ds / "select.json"
        sel = rjson(sel_path)
        arr = Z6.eval_arrays(ds, "frozen")
        need = Z6.needed_names(sel)
        miss = [n for n in need if n not in arr]
        if miss:
            stop(f"{ds}: {len(miss)} of PL6's needed cells have no filed eval array, e.g. {miss[0]}")
        names = [n for n in need if float(arr[n][1].mean()) <= MAX_NODES]
        left = [{"name": n, "filed_nodes": float(arr[n][1].mean())} for n in need if n not in names]
        r95 = sel["picks"]["R95_all"]["name"]
        if r95 not in names:
            stop(f"{ds}: the R95 pick {r95} is outside the node cap")
        qids, gc = filed_meta(ds)
        nq = len(qids)
        n_chunks = (nq + CHUNK - 1) // CHUNK
        gq = min(GATE_Q, nq)
        rec = {"stage": "LEG1-A6 cells", "declared_in": DOC, "dataset": ds, "utc": utc(), "nq": nq, "qids_sha256": qids_sha(qids), "chunk": CHUNK, "perm_seed": PERM_SEED,
               "n_chunks": n_chunks, "chunk_order": chunk_order(n_chunks, k), "gate_questions": gq, "max_nodes": MAX_NODES, "r95": r95,
               "select_sha256": hashlib.sha256(sel_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(), "names": names,
               "filed": {n: {"all": float(arr[n][0].mean()), "nodes": float(arr[n][1].mean())} for n in names}, "left_out": left}
        POOL.joinpath(ds).mkdir(parents=True, exist_ok=True)
        wjson(cells_path(ds), rec)
        with open(ref_path(ds), "wb") as f:
            np.savez(f, names=np.asarray(names), allgold=np.stack([arr[n][0][:gq] for n in names], axis=1), size=np.stack([arr[n][1][:gq] for n in names], axis=1),
                     gold_counts=gc[:gq])
        log(f"cells {ds}: {len(names)} cells kept ({len(left)} left out above {MAX_NODES:.0f} nodes), {nq} questions in {n_chunks} chunks; R95 pick {r95}")
    return 0


# ── run ──────────────────────────────────────────────────────────────────────

def cmd_run(a):
    Z6.prepare(bool(a.host))
    ds = a.ds
    cells = rjson(cells_path(ds))
    names = cells["names"]
    part, nparts = (int(a.part[0]), int(a.part[1])) if a.part else (0, 1)
    op = P2.opened()
    src = P2.load_source(ds, "eval", "frozen")
    if src.nq != cells["nq"] or qids_sha([str(x) for x in src.qids]) != cells["qids_sha256"]:
        stop(f"{ds}: the eval carve here is not the one the cell list was cut for")
    plan = Z5.Plan5(names)
    gp = POOL / ds / f"gate_p{part}.json"
    gate = rjson(gp) if gp.is_file() else None
    if gate is None or gate.get("verdict") != "PASS":
        t0 = time.time()
        ref = np.load(ref_path(ds), allow_pickle=False)
        try:
            if [str(x) for x in ref["names"]] != names:
                stop(f"{ds}: the gate reference is not for this cell list")
            gq = int(cells["gate_questions"])
            P, S = Z5.evaluate5(op, src, plan, 0, gq, every=16)                   # the INTACT source: SPLADE as filed
            bad = gate_equal(P, S, src.gold_counts[:gq], ref["allgold"], ref["size"])
        finally:
            ref.close()
        d = dense_only(src)
        M3A = op.m3a
        seeds = M3A.stable_union_rows(np.asarray(d.dense[:gq, :5], dtype=np.int64), np.asarray(d.splade[:gq, :5], dtype=np.int64))
        g1 = bool(d.splade is d.dense) and all(len(s) == 5 for s in seeds)
        gate = {"stage": "LEG1-A6 gates", "declared_in": DOC, "dataset": ds, "part": part, "utc": utc(), "G-D0": {"questions": gq, "cells": len(names), "mismatches": bad[:10], "n_mismatches": len(bad)},
                "G-D1": {"seeds_per_question": sorted({len(s) for s in seeds}), "splade_is_dense": bool(d.splade is d.dense)}, "seconds": round(time.time() - t0, 1),
                "verdict": "PASS" if (not bad and g1) else "FAIL"}
        wjson(gp, gate)
        log(f"{ds}: gates {gate['verdict']} (G-D0 {len(bad)} mismatches over {gq} questions x {len(names)} cells; G-D1 seeds {gate['G-D1']['seeds_per_question']})")
        if gate["verdict"] != "PASS":
            return 3
    d = dense_only(src)
    order = cells["chunk_order"]
    done = 0
    for pos in range(part, len(order), nparts):
        c = order[pos]
        path = chunk_path(ds, c)
        if path.is_file():
            continue
        lo, hi = chunk_range(c, src.nq)
        t0 = time.time()
        P, S = Z5.evaluate5(op, d, plan, lo, hi, every=64)
        tmp = path.with_name(path.name + ".part")
        with open(tmp, "wb") as f:
            np.savez(f, lo=lo, hi=hi, names=np.asarray(names), present=P, size=S, seconds=round(time.time() - t0, 1))
        os.replace(tmp, path)
        done += 1
        log(f"{ds}: chunk {c} (questions {lo}-{hi}) done in {time.time() - t0:.0f} s ({done} this run, position {pos + 1} of {len(order)}; {P2.peak_gb()} GB)")
    log(f"{ds}: part {part} of {nparts} finished ({done} chunks made now)")
    return 0


# ── read ─────────────────────────────────────────────────────────────────────

def load_chunks(ds, names, gc_all):
    """The chunks that exist: (question indices, dense-only all-gold bool (Q, C), sizes (Q, C))."""
    idx, allg, size = [], [], []
    for p in sorted((POOL / ds).glob("chunk_*.npz")):
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


def read_one(ds, k):
    cells = rjson(cells_path(ds))
    names = cells["names"]
    qids, gc = filed_meta(ds)
    if qids_sha(qids) != cells["qids_sha256"]:
        stop(f"{ds}: the filed eval grids are not the ones the cell list was cut for")
    for p in sorted((POOL / ds).glob("gate_p*.json")):
        if rjson(p)["verdict"] != "PASS":
            stop(f"{ds}: {p.name} is not PASS: no dense-only number is read")
    if not list((POOL / ds).glob("gate_p*.json")):
        stop(f"{ds}: no gate file: no dense-only number is read")
    got = load_chunks(ds, names, gc)
    if got is None:
        return {"dataset": ds, "status": "NO_CHUNKS"}
    idx, allg, size = got
    arr = Z6.eval_arrays(ds, "frozen")
    filed = {n: (arr[n][0][idx], arr[n][1][idx]) for n in names}
    dense = {n: (allg[:, i], size[:, i]) for i, n in enumerate(names)}
    rng = np.random.default_rng([P2.BOOT_SEED, k])
    rows = compare(names, filed, dense, rng)
    r95 = cells["r95"]
    nq = int(cells["nq"])
    rec = {"stage": "LEG1-A6 read", "declared_in": DOC, "dataset": ds, "utc": utc(), "questions_covered": int(idx.size), "questions_total": nq,
           "status": "COMPLETE" if idx.size == nq else "PARTIAL", "chunks": f"{len(list((POOL / ds).glob('chunk_*.npz')))} of {cells['n_chunks']}",
           "r95": r95, "r95_row": rows[r95], "call": rows[r95]["call"], "cells": rows, "reach_filed": reach(rows, "filed"), "reach_dense_only": reach(rows, "dense"),
           "left_out": cells["left_out"]}
    wjson(POOL / ds / f"read_{ds}.json", rec)
    r = rows[r95]
    log(f"{ds} [{rec['status']} {idx.size}/{nq}]: R95 cell {r95}: filed ALL {r['filed_all']:.4f} @ {r['filed_nodes']:.0f}, dense-only ALL {r['dense_all']:.4f} @ {r['dense_nodes']:.0f}; "
        f"loss {100 * r['loss']:+.2f} points [{100 * r['loss_ci'][0]:+.2f}, {100 * r['loss_ci'][1]:+.2f}] {r['call']}")
    return rec


def cmd_read(a):
    out = {"stage": "LEG1-A6 pool read", "declared_in": DOC, "utc": utc(), "datasets": {}}
    for k, ds in enumerate(DS):
        if a.ds and ds not in a.ds:
            continue
        if not cells_path(ds).is_file():
            out["datasets"][ds] = {"status": "NOT_READ", "why": "no cell list"}
            continue
        out["datasets"][ds] = read_one(ds, k)
    wjson(POOL / "pool_dense_only.json", out)
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def cmd_selftest(a):
    checks, bad = 0, []

    def ok(cond, what):
        nonlocal checks
        checks += 1
        if not cond:
            bad.append(what)
    for k in range(4):
        o = chunk_order(19, k)
        ok(sorted(o) == list(range(19)), f"chunk order {k} is a permutation")
        ok(o == chunk_order(19, k), f"chunk order {k} is deterministic")
    ok(chunk_order(19, 0) != chunk_order(19, 1), "orders differ by dataset")
    ok(chunk_range(0, 1130) == (0, 128) and chunk_range(8, 1130) == (1024, 1130), "chunk ranges")
    w = wilson(50, 100)
    ok(abs(w[0] - 0.4038) < 1e-3 and abs(w[1] - 0.5962) < 1e-3, "wilson 50/100")
    ok(call_loss([0.012, 0.03]) == "NEEDED" and call_loss([-0.01, 0.009]) == "DISPENSABLE" and call_loss([0.0, 0.02]) == "UNCLEAR", "call convention")

    class Fake:
        pass
    s = Fake()
    s.dense = np.arange(40, dtype=np.int32).reshape(2, 20)
    s.splade = s.dense[:, ::-1].copy()
    d = dense_only(s)
    ok(d.splade is s.dense and d.dense is s.dense and not np.array_equal(s.splade, s.dense), "dense_only swaps only the SPLADE list")
    import m3a_headroom as M3A
    seeds = M3A.stable_union_rows(d.dense[:, :5], d.splade[:, :5])
    ok(all(len(x) == 5 for x in seeds) and all(np.array_equal(x, d.dense[i, :5]) for i, x in enumerate(seeds)), "dense-only seeds are the dense top 5")
    fused = M3A.rrf_fused_rows(d.dense[:, :20], d.splade[:, :20], P2.RRF_CONST)
    ok(all(np.array_equal(f, d.dense[i, :20]) for i, f in enumerate(fused)), "the fusion of the dense list with itself keeps its order")
    # gate comparison
    gc = np.array([2, 1, 3])
    P = np.array([[2, 1], [1, 0], [3, 3]], dtype=np.int16)
    S = np.array([[5, 7], [6, 8], [9, 4]], dtype=np.int32)
    refa = np.array([[True, False], [True, False], [True, True]])
    ok(gate_equal(P, S, gc, refa, S.astype(np.float64)) == [], "gate equal when the arrays match")
    P2_ = P.copy()
    P2_[0, 1] = 2
    S2 = S.copy()
    S2[2, 0] = 10
    ok(sorted(map(tuple, gate_equal(P2_, S2, gc, refa, S.astype(np.float64)))) == [(0, 1, "all"), (2, 0, "size")], "gate finds the changed cells")
    # compare and reach on synthetic arrays: dense-only loses 10 of 200 questions at cell A, none at B
    rng = np.random.default_rng(1)
    nq = 200
    fa = np.ones(nq, dtype=bool)
    fa[:4] = False
    da = fa.copy()
    da[4:14] = False
    sz = np.full(nq, 50.0)
    filed = {"A": (fa, sz), "B": (fa.copy(), sz * 2)}
    dense = {"A": (da, sz * 0.8), "B": (fa.copy(), sz * 2)}
    rows = compare(["A", "B"], filed, dense, rng)
    ok(abs(rows["A"]["loss"] - 0.05) < 1e-9 and rows["A"]["call"] == "NEEDED", f"loss at A {rows['A']['loss']} call {rows['A']['call']}")
    ok(rows["B"]["loss"] == 0.0 and rows["B"]["call"] == "DISPENSABLE", "no loss at B")
    ok(abs(rows["A"]["nodes_dense_over_filed"] - 0.8) < 1e-9, "nodes ratio dense over filed")
    ok(reach(rows, "filed")["name"] == "A" and reach(rows, "dense")["name"] == "B", f"reach picks the smallest-nodes cell that reaches (dense: {reach(rows, 'dense')})")
    # chunks round trip
    global POOL
    keep = POOL
    with tempfile.TemporaryDirectory() as td:
        POOL = Path(td)
        (POOL / "x").mkdir()
        names = ["A", "B"]
        gca = np.arange(300) % 3 + 1
        for c in (2, 0):
            lo, hi = chunk_range(c, 300)
            pres = np.stack([gca[lo:hi], np.maximum(gca[lo:hi] - 1, 0)], axis=1).astype(np.int16)
            with open(chunk_path("x", c), "wb") as f:
                np.savez(f, lo=lo, hi=hi, names=np.asarray(names), present=pres, size=np.full((hi - lo, 2), 3, dtype=np.int32), seconds=0.0)
        got = load_chunks("x", names, gca)
        ok(got is not None and got[0].size == 128 + 44 and bool(got[1][:, 0].all()) and not bool(got[1][:, 1].any()), "chunks load in order of file and give the all-gold flags")
    POOL = keep
    res = {"stage": "LEG1-A6 selftest", "checks": checks, "problems": bad, "verdict": "PASS" if not bad else "FAIL"}
    print(json.dumps(res))
    return 0 if not bad else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("cells")
    p.add_argument("--ds", nargs="*", choices=DS)
    p = sub.add_parser("run")
    p.add_argument("--ds", required=True, choices=DS)
    p.add_argument("--part", nargs=2, metavar=("I", "K"))
    p.add_argument("--host", action="store_true")
    p = sub.add_parser("read")
    p.add_argument("--ds", nargs="*", choices=DS)
    a = ap.parse_args(argv)
    return {"selftest": cmd_selftest, "cells": cmd_cells, "run": cmd_run, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
