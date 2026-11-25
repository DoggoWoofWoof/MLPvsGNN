"""UL5 (docs/UL5_DEPTH_SIGNAL.md), the extraction half: the depth ladder of each dataset's family and three label-free first-stage statistics per question.

    python outputs/mp_unified/zul5st.py cells                                   # the declared cells of the seven families and their sha256
    python outputs/mp_unified/zul5st.py selftest --ds D [--n 12] [--host]       # G5-3 and G5-4 on the first questions of a dev carve
    python outputs/mp_unified/zul5st.py extract --ds D --carve dev [--part I K] [--host]

It imports zpl6g.py (PL6's engine, which imports PL5's, PL4's, PL3's and PL2's) and edits none of them. The pool code takes and reads no labels; labels enter in zpl3g.grade3
only, as in every PL grid. The statistics (cos1, gap15, agree) are computed from the stored first-stage lists and the stored vectors and never read a label.
Everything is deterministic (integer walks, float64 inner products, ties to the lower node id), and the engine-identity pins of the filed PL6 grids are recorded in each head.
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
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zpl6g as Z6  # noqa: E402

Z5, P2, Z3 = Z6.Z5, Z6.P2, Z6.Z3
DECLARED = "docs/UL5_DEPTH_SIGNAL.md"
OUT = ROOT / "outputs" / "ul5"
SEVEN = ("multihoprag", "squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp")
FAMILY = {                                          # PL6's U95 picks, base+walk without the cut (docs/UL4_UNIVERSAL_LP.md section 4; the same names)
    "multihoprag": "rrf1000_b25+FULL:h1_c100",
    "squad": "rrf1000_b15",
    "hotpotqa": "rrf1000_b25+STRUCT:h1_c100~cs",
    "2wiki": "rrf1000_b400+FULL:h1_c100~s20",
    "musique": "rrf1000_b400+FULL:h3_bf_v20000~cs-fu",
    "metaqa": "rrf1000_b25+STRUCT:h3_c100_v5000",
    "webqsp": "rrf1000_b400+FULL:h3_bf_v40000~cs-vr",
}
LADDER_M = (0, 100, 200, 400, 800, 1200, 1600, 2000, 2500, 3500, 5000, 7500, 10000, 15000, 20000, 30000)   # walk nodes kept beyond the base (M = 0: the base alone); then the uncut pool
STAT_NAMES = ("cos1", "gap15", "agree")
EXPECT_CELLS = 17 * 6 + 1
EXPECT_SHA256 = "37ad6402a67a528051d1866e44ff546f29aa2e4b6f4e475dfce6a2946daa09da"      # fixed at the declaration commit (docs section 6)


def stop(msg):
    raise SystemExit(f"[zul5st] STOP: {msg}")


def log(msg):
    print(f"[zul5st {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the declared cells ───────────────────────────────────────────────────────

def family_cells(d):
    """The ladder of one dataset: the base alone (M = 0), P cut to M walk nodes for each M of LADDER_M above 0, and the uncut P. A family without a walk (SQuAD) is its base alone."""
    p = FAMILY[d]
    b = p.split("+")[0]
    if "+" not in p:
        return [b]
    return [b] + [f"{p}+top{m}" for m in LADDER_M if m > 0] + [p]


def all_cells():
    return {d: family_cells(d) for d in SEVEN}


def cells_digest():
    names = [n for d in SEVEN for n in family_cells(d)]
    return len(names), P2.cells_sha256(names)


def level_m(d, name):
    """The universal level of a cell: M walk nodes kept (0 = the base alone; None = the uncut pool)."""
    if "+" not in name:
        return 0
    if name == FAMILY[d]:
        return None
    return int(name.rsplit("+top", 1)[1])


def require_cells():
    n, sha = cells_digest()
    if n != EXPECT_CELLS:
        stop(f"the cell list has {n} cells, expected {EXPECT_CELLS}")
    if EXPECT_SHA256 is not None and sha != EXPECT_SHA256:
        stop("the cell list is not the declared one (cells_sha256 differs from EXPECT_SHA256)")


# ── the label-free statistics ────────────────────────────────────────────────

def question_stats(src, c0, c1):
    """For rows c0..c1 of a source: cos1 (the question vector's inner product with the dense list's first node), gap15 (cos1 minus the same for its fifth) and agree (how many of
    the dense list's first five nodes are among the splade list's first five). The stored lists and vectors only; no label is read here."""
    d = np.asarray(src.dense[c0:c1], dtype=np.int64)
    s = np.asarray(src.splade[c0:c1], dtype=np.int64)
    qv = np.asarray(src.qvec(c0, c1), dtype=np.float64)
    n = c1 - c0
    top = np.asarray(src.dvec(d[:, :5].reshape(-1)), dtype=np.float64).reshape(n, 5, -1)
    sc = (top * qv[:, None, :]).sum(axis=2)
    out = np.zeros((n, len(STAT_NAMES)), dtype=np.float64)
    out[:, 0] = sc[:, 0]
    out[:, 1] = sc[:, 0] - sc[:, 4]
    for i in range(n):
        out[i, 2] = len(set(d[i, :5].tolist()) & set(s[i, :5].tolist()))
    return out


def stats_rows(src, lo, hi, chunk=256):
    return np.concatenate([question_stats(src, c0, min(c0 + chunk, hi)) for c0 in range(lo, hi, chunk)], axis=0)


# ── engine pins ──────────────────────────────────────────────────────────────

def code_sha256():
    """One hash of this module's code and constants and of PL6's (which holds PL5's, PL4's, PL3's and PL2's): the same on every dataset (G5-1 reads it)."""
    h = hashlib.sha256()
    for fn in (family_cells, level_m, question_stats, stats_rows):
        h.update(inspect.getsource(fn).replace("\r\n", "\n").encode("utf-8"))
    h.update(json.dumps({"family": FAMILY, "ladder_m": list(LADDER_M), "stats": list(STAT_NAMES)}, sort_keys=True).encode("utf-8"))
    h.update(Z6.code_path_sha256().encode("utf-8"))
    return h.hexdigest()


def prepared(host):
    Z6.prepare(host)
    require_cells()


def load(ds, carve):
    op = P2.opened()
    src = P2.load_source(ds, carve, "frozen")
    return op, src


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_cells(a):
    n, sha = cells_digest()
    log(f"{n} cells over {len(SEVEN)} datasets, sha256 {sha}")
    for d in SEVEN:
        c = family_cells(d)
        log(f"{d}: {len(c)} cells; first {c[0]}; last {c[-1]}")
    if EXPECT_SHA256 is not None:
        require_cells()
        log("matches the declared list")
    return 0


def cmd_selftest(a):
    prepared(a.host)
    op, src = load(a.ds, "dev")
    n = min(a.n, src.nq)
    names = family_cells(a.ds)
    plan = Z5.Plan5(names)
    verdict = {}
    t0 = time.time()
    P, S = Z5.evaluate5(op, src, plan, 0, n)
    verdict["questions"] = int(n)
    verdict["cells"] = int(plan.C)
    # G5-4: the ladder is nested: cuts are prefixes of one ranked list, so presence and size do not fall as M grows
    ok_nest = bool((np.diff(P.astype(np.int64), axis=1) >= 0).all() and (np.diff(S.astype(np.int64), axis=1) >= 0).all()) if len(names) > 1 else True
    verdict["G5-4_nested"] = ok_nest
    # G5-3: the statistics read no label (the source runs with its labels gone) and are the same with and without them
    st1 = stats_rows(src, 0, n)
    saved = src.golds
    src.golds = None
    st2 = stats_rows(src, 0, n)
    src.golds = saved
    verdict["G5-3_labels_unread"] = bool(np.array_equal(st1, st2) and "gold" not in inspect.getsource(question_stats) and "gold" not in inspect.getsource(stats_rows))
    verdict["stats_finite"] = bool(np.isfinite(st1).all())
    verdict["agree_range"] = [float(st1[:, 2].min()), float(st1[:, 2].max())]
    verdict["verdict"] = "PASS" if (ok_nest and verdict["G5-3_labels_unread"] and verdict["stats_finite"]) else "FAIL"
    verdict.update({"dataset": a.ds, "host": P2.HOST, "seconds": round(time.time() - t0, 1), "script_sha256": script_sha256(), "code_sha256": code_sha256(), "utc": P2.utc()})
    p = OUT / f"selftest_{a.ds}{'_host' if P2.HOST else ''}.json"
    P2.wjson(p, verdict)
    log(f"{a.ds}: selftest {verdict['verdict']} (nested {ok_nest}, labels unread {verdict['G5-3_labels_unread']}) in {verdict['seconds']} s")
    return 0 if verdict["verdict"] == "PASS" else 1


def cmd_extract(a):
    t0 = time.time()
    prepared(a.host)
    names = family_cells(a.ds)
    plan = Z5.Plan5(names)
    op, src = load(a.ds, a.carve)
    lo, hi, part = 0, src.nq, None
    if a.part:
        i, k = int(a.part[0]), int(a.part[1])
        lo, hi = i * src.nq // k, (i + 1) * src.nq // k
        part = (i, k)
    log(f"{a.ds}/{a.carve}: {src.nq} questions, rows {lo}..{hi}, {plan.C} cells, {len(plan.walks)} walks, graph {src.graph_kind} ({P2.peak_gb()} GB)")
    st = stats_rows(src, lo, hi)
    P, S = Z5.evaluate5(op, src, plan, lo, hi)
    base = OUT / a.ds / f"ext_{a.carve}"
    p = Path(str(base) + (f".part{part[0]}of{part[1]}" if part else "") + ".npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez(tmp, names=np.asarray(names, dtype=str), present=P, size=S, qids=np.asarray(src.qids[lo:hi], dtype=str), gold_counts=src.gold_counts[lo:hi].astype(np.int32),
             lo=np.asarray(lo), hi=np.asarray(hi), stats=st, stat_names=np.asarray(STAT_NAMES, dtype=str))
    os.replace(tmp, p)
    head = {"stage": "UL5 extraction", "declared_in": DECLARED, "dataset": a.ds, "carve": a.carve, "first": "frozen", "questions": int(hi - lo), "rows": [int(lo), int(hi)],
            "part": list(part) if part else None, "cells": plan.C, "cells_sha256": P2.cells_sha256(names), "n_nodes": int(src.n), "population_digest": src.pop_digest,
            "code_sha256": code_sha256(), "pl6_code_path_sha256": Z6.code_path_sha256(), "pinned": dict(Z6._PREPARED), "script_sha256": script_sha256(),
            "npz": P2.rel(p), "npz_sha256": P2.sha256_file(p), "seconds": round(time.time() - t0, 1), "peak_gb": P2.peak_gb(), "host": P2.HOST, "utc": P2.utc()}
    P2.wjson(Path(str(p)[:-4] + ".json"), head)
    log(f"{a.ds}/{a.carve}: filed {P2.rel(p)} in {head['seconds']} s")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cells")
    s = sub.add_parser("selftest")
    s.add_argument("--ds", required=True, choices=SEVEN)
    s.add_argument("--n", type=int, default=12)
    e = sub.add_parser("extract")
    e.add_argument("--ds", required=True, choices=SEVEN)
    e.add_argument("--carve", required=True, choices=("dev", "eval"))
    e.add_argument("--part", nargs=2, type=int, default=None, metavar=("I", "K"))
    for name in ("selftest", "extract"):
        sub.choices[name].add_argument("--host", action="store_true", help="open the package as the lab host's mirror")
    a = ap.parse_args(argv)
    if a.cmd == "cells":
        return cmd_cells(a)
    return {"selftest": cmd_selftest, "extract": cmd_extract}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
