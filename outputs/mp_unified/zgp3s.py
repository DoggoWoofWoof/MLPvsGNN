"""GP3's program (docs/GP3_NODE_COUNT_RULE.md): reads the chunks zgp3.py wrote and grades every triple N_1 >= N_2 >= N_3 of the ladder exactly.

    python outputs/mp_unified/zgp3s.py selftest                  # the grading equals the naive pools; P1/P2 equal a brute-force search (no real data)
    python outputs/mp_unified/zgp3s.py solve                     # the dev quarter: P1 / P2 per score, the null, own optima, the final rule -> outputs/gp3/rule_q.json + report_q.md
    python outputs/mp_unified/zgp3s.py grade                     # the rest of dev, ONCE: the quarter's final rule and per-cell triples -> outputs/gp3/rule_r.json + report_r.md
The chunks (outputs/gp3/<d>/work/<cell>_<part>_<q0>.npz) carry the sha256 of the zgp3.py that wrote them; solve and grade require ONE sha for a whole part and record it.
No label is read here that the chunks do not hold, and the rest-of-dev chunks are never opened by `solve`.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zgp3 as G3  # noqa: E402  (imports zgp1 first)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

LADDER, SCORES, CELLS, OUT = G3.LADDER, G3.SCORES, G3.CELLS, G3.OUT
log, rel, stop, L2 = G3.log, G3.rel, G3.stop, G3.L2
TRIPLES = [(i, j, k) for i in range(len(LADDER)) for j in range(i + 1) for k in range(j + 1)]       # N_1 >= N_2 >= N_3: 165 triples
ALL_MIN = 0.95                                            # the coverage target (the user's "95 plus")
P2_N = (2500, 3500, 5000)                                 # the node ceilings of P2 (the user's 2.5k and two looser)
R_MIN_ALL = 0.92                                          # the rest of dev: every cell's ALL at the quarter's final triple at least this (point value)
CAP_SHARE = 0.02                                          # a cell with more than this share of questions over GATHER_CAP is CAPPED
SIMPLE = ("K4", "K2", "K1", "K3", "PP")                         # the simplicity order of the scores for ties
TIE = 1.05                                                # within this factor of the best P1 the simpler score wins
NULL_MARGIN = 0.9                                         # a score beats the random-order null if its P1 worst-cell nodes are at most this factor of the smaller random P1
UL5_ORACLE = {"2wiki": 541, "musique": 1831, "metaqa": 2745, "webqsp": 9301}      # UL5 section 11.2, per-question least nodes with the golds read, a different universe: context only
UL5_UNIFORM = {"multihoprag": 255, "squad": 15, "hotpotqa": 85, "2wiki": 911, "musique": 20038, "metaqa": 4984, "webqsp": 39572}    # UL5 section 11.2, the uniform level: context only


def cell_name(d, v):
    return f"{d}/{v}"


def script_sha256():
    return G3.script_sha256.__globals__["hashlib"].sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ── loading ──────────────────────────────────────────────────────────────────


# A3 (cost, decided from `time` alone): the quarter of a heavy cell is the chunks whose index (position // CHUNK_Q in the part's question list) mod n is in the subset; the rest of dev is never subsampled.
QUARTER_SHARDS = {"hotpotqa": (19, (1, 4, 7, 10, 13, 16)), "2wiki": (19, (1, 4, 7, 10, 13, 16))}    # A6 adds 2wiki (its time row: heavy, 464 s per question over both cells)


def want_questions(d, nq, part):
    want = G3.part_indices(nq, part)
    if part == "q" and d in QUARTER_SHARDS:
        n, sub = QUARTER_SHARDS[d]
        keep = np.asarray([(i // G3.CHUNK_Q) % n in sub for i in range(want.size)], bool)
        want = want[keep]
    return want


def load_cell_chunks(d, v, part):
    """All chunks of one cell and part, concatenated; the zgp3 sha of the chunks is returned beside."""
    wd = OUT / d / "work"
    files = sorted(wd.glob(f"{v}_{part}_*.npz"))
    if not files:
        return None, None
    parts, nq, shas, rmeta = [], None, set(), set()
    for f in files:
        with np.load(f) as z:
            shas.add(str(z["zsha"]))
            rmeta.add((str(z["rscore"]), tuple(map(tuple, z["rtriples"].tolist())), str(z["zrsha"])) if "rscore" in z.files else None)     # A4: a restricted chunk (zgp3r.py)
            parts.append({k: z[k] for k in z.files if k not in ("zsha", "zrsha", "rscore", "rtriples", "cstats")})
            nq = int(z["nq"])
    if len(rmeta) != 1:
        stop(f"{d}/{v}/{part}: the chunks mix full and restricted walks or sets of triples: {sorted(map(str, rmeta))}")
    want = want_questions(d, nq, part)
    got = np.sort(np.concatenate([p["qidx"] for p in parts]))
    if not np.array_equal(got, want):
        stop(f"{d}/{v}/{part}: the chunks hold {got.size} questions, the part has {want.size}: rows are missing")
    cat, off, gq = {}, 0, []
    for p in parts:
        gq.append(p["gq"] + off)
        off += p["qidx"].size
    for k in parts[0]:
        if k != "nq":
            cat[k] = np.concatenate([p[k] for p in parts])
    cat["gq"] = np.concatenate(gq)
    rm = next(iter(rmeta))
    cat["restricted"] = None if rm is None else {"score": rm[0], "triples": [tuple(t) for t in rm[1]], "zgp3r_sha256": rm[2]}
    return cat, shas


def check_extract(d, part):
    """The gates the extraction recorded: G3-1 and the UL2P gates read, every shard of the part present."""
    fs = sorted((OUT / d).glob(f"extract_{part}*.json"))
    fs = [f for f in fs if not f.name.endswith("_limit.json")]
    if not fs:
        stop(f"{d}: no extract_{part}*.json")
    seen, n_sh = set(), None
    for f in fs:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("status") != "read" or r["gates"].get("G3-1") != "PASS":
            stop(f"{rel(f)}: gates not passed")
        seen.add(r["shard"][0])
        n_sh = r["shard"][1]
    need = set(range(n_sh))
    if part == "q" and d in QUARTER_SHARDS:
        if n_sh != QUARTER_SHARDS[d][0]:
            stop(f"{d}/{part}: the rows are shard x of {n_sh}, the declared quarter is of {QUARTER_SHARDS[d][0]}")
        need = set(QUARTER_SHARDS[d][1])
    if seen != need:
        stop(f"{d}/{part}: shards {sorted(seen)} of {n_sh}, declared {sorted(need)}")
    return {"files": [f.name for f in fs], "gates": r["gates"]}


def triple_eval(c, k, ts):
    """ALL share and mean pool size at the given triples (tuples of ladder indices), score index k of the chunk arrays, and the per-question detail. Entries of triples that a restricted walk
    did not read are -1 in the chunk and stop the run."""
    Q = c["nfree"].size
    t1 = np.array([t[0] for t in ts])
    t2 = np.array([t[1] for t in ts])
    t3 = np.array([t[2] for t in ts])
    s1, s2, s3 = c["sz1"][:, k][:, t1], c["sz2"][:, k][:, t1, t2], c["sz3"][:, k][:, t1, t2, t3]
    if bool((s1 < 0).any() or (s2 < 0).any() or (s3 < 0).any()):                              # each hop's size on its own: a sum of -1 placeholders could still be positive
        stop("a size of an invalid triple (or of a triple the restricted walk did not read) was read")
    size = c["nfree"][:, None].astype(np.int64) + s1 + s2 + s3
    glev, gfree = c["glev"], c["gfree"]
    cov = np.zeros((glev.size, len(ts)), bool)
    cov |= (gfree | (glev == 0))[:, None]
    m1 = glev == 1
    cov[m1] |= c["gb1"][m1, k][:, None] <= t1[None, :]
    m2 = glev == 2
    cov[m2] |= c["gb2"][m2, k][:, t1] <= t2[None, :]
    m3 = glev == 3
    cov[m3] |= c["gb3"][m3, k][:, t1, t2] <= t3[None, :]
    unc = np.zeros((Q, len(ts)), np.int64)
    np.add.at(unc, c["gq"], (~cov).astype(np.int64))
    allq = unc == 0
    return {"all": allq.mean(axis=0), "nodes": size.mean(axis=0), "allq": allq, "size": size, "cov": cov}


SLOT_KINDS = (("GP", 0), ("CH", 1))


def slot_reads(c, cov, S, v):
    """A4 (beside, decides nothing): the slots of the dev slice (both ends gold nodes, outputs/gp3/<d>/slots.npz) against the stored gold coverage `cov` (golds x triples). Per kind: the slots
    of the chunks' questions, those linked in version v (pool-free), those whose two ends are both in Pool(q) at each triple, and those linked AND in the pool (the slot recall)."""
    qidx, Q = c["qidx"], c["qidx"].size
    order = np.argsort(qidx, kind="stable")
    qs = qidx[order]
    pos = np.minimum(np.searchsorted(qs, S["q"]), Q - 1)
    ok = qs[pos] == S["q"]
    row = order[pos]
    goff = np.searchsorted(c["gq"], np.arange(Q))
    glen = np.bincount(c["gq"], minlength=Q)
    ia, ib = S["ia"].astype(np.int64)[ok], S["ib"].astype(np.int64)[ok]
    r = row[ok]
    if bool((ia >= glen[r]).any() or (ib >= glen[r]).any() or (ia == ib).any()):
        stop("slot reads: a slot's gold index is beyond the golds the chunks hold, or its two ends are one gold (slots.npz and the chunks disagree)")
    kept = cov[goff[r] + ia] & cov[goff[r] + ib]
    lk = S[f"linked_{v}"][ok]
    kind = S["kind"][ok]
    sq = S["q"][ok]
    out = {}
    for nm, kd in SLOT_KINDS:
        m = kind == kd
        n = int(m.sum())
        if n == 0:
            continue
        miss = np.zeros((Q, kept.shape[1]), np.int64)                                      # per question and triple: linked slots whose two ends are not both in the pool
        np.add.at(miss, r[m], (lk[m][:, None] & ~kept[m]).astype(np.int64))
        has = np.bincount(r[m], weights=lk[m].astype(float), minlength=Q) > 0               # questions with at least one linked slot of this kind
        out[nm] = {"slots": n, "questions": int(np.unique(sq[m]).size), "linked": int(lk[m].sum()), "kept": kept[m].sum(axis=0).astype(np.int64).tolist(),
                   "kept_linked": (kept[m] & lk[m][:, None]).sum(axis=0).astype(np.int64).tolist(), "questions_linked": int(has.sum()),
                   "questions_all_linked_kept": (has[:, None] & (miss == 0)).sum(axis=0).astype(np.int64).tolist()}
    return out


def slot_triples(rec, name):
    """The triples read for slots in one cell: floor, the final rule's, the cell's own optimum of the final score, the final score's P2 triples, the ladder top."""
    K = rec["final"]["score"]
    sc = rec["scores"][K]
    tr = {"floor": (0, 0, 0), "final": tuple(rec["final"]["triple"])}
    if sc["own"][name]:
        tr["own"] = tuple(sc["own"][name]["triple"])
    for x in P2_N:
        if sc["P2"][str(x)]:
            tr[f"P2_{x}"] = tuple(sc["P2"][str(x)]["triple"])
    tr["top"] = TRIPLES[-1]
    return tr


def rest_triples(q, name):
    """The triples the rest-of-dev grade reads for one cell: the final, the cell's own quarter optimum of the final score, the P2 triples of the final score (None where the quarter had none)."""
    K = q["final"]["score"]
    sc = q["scores"][K]
    out = {"final": tuple(q["final"]["triple"]), "own_quarter": tuple(sc["own"][name]["triple"]) if sc["own"][name] else None}
    for x in P2_N:
        out[f"P2_{x}"] = tuple(sc["P2"][str(x)]["triple"]) if sc["P2"][str(x)] else None
    return out


def rest_spec(q, d):
    """A5: what a restricted extraction of the rest of dev reads for dataset d: the final score and the union over d's cells of the triples the grade reads, and the floor."""
    ts = {(0, 0, 0)}
    for dd, v in CELLS:
        if dd == d:
            ts |= {t for t in rest_triples(q, cell_name(dd, v)).values() if t is not None}
    return q["final"]["score"], sorted(ts)


def cmd_rest_spec(a):
    q = json.loads((OUT / "rule_q.json").read_text(encoding="utf-8"))
    if not q["final"]:
        stop("the quarter has no final rule")
    K, ts = rest_spec(q, a.ds)
    print(f"--score {K} --triples " + ",".join("-".join(str(i) for i in t) for t in ts))
    return 0


def slot_reads_all(rec, chunks):
    """Per cell with a slots file: the slot reads at the triples above, for the final score (a restricted chunk: at the triples it read)."""
    out = {}
    if not rec["final"]:
        return out
    k = SCORES.index(rec["final"]["score"])
    for name, c in chunks.items():
        d, v = name.split("/")
        f = OUT / d / "slots.npz"
        if not f.is_file():
            log(f"{name}: no slots.npz, no slot read")
            continue
        S = dict(np.load(f))
        tr = slot_triples(rec, name)
        kk = k
        if c.get("restricted"):
            tr = {nm: t for nm, t in tr.items() if t in c["restricted"]["triples"]}
            kk = 0
        ts = list(tr.values())
        ev = triple_eval(c, kk, ts)
        rd = slot_reads(c, ev["cov"], S, v)
        for kind, x in rd.items():
            x["triples"] = {nm: {"triple": list(t), "N": [LADDER[i] for i in t], "j": j} for j, (nm, t) in enumerate(tr.items())}
        out[name] = {"reads": rd, "slots_npz_sha256": hashlib.sha256((OUT / d / "slots.npz").read_bytes()).hexdigest()}
    return out


def md_slots(rec):
    sr = rec.get("slot_reads") or {}
    if not sr:
        return ""
    L = ["", f"**Slot reads (A4, beside; decides nothing), final score {rec['final']['score']}.** Slots = pairs of golds (GP) and MuSiQue chain pairs (CH); both ends are gold nodes. Link recall = slots linked in the cell's any-link graph, whatever the pool "
         "(pool-free); pool pairs = slots with both ends in Pool(q); slot recall = slots linked AND with both ends in Pool(q) (the share of slots a model with this pool can use as a direct link). Intervals (Wilson over slots) treat slots as independent.", ""]
    rows = []
    for name, x in sr.items():
        for kind, y in x["reads"].items():
            for nm, tinfo in y["triples"].items():
                j, n = tinfo["j"], y["slots"]
                lo, hi = wilson(y["kept_linked"][j], n)
                rows.append([name, kind, f"{n:,}", nm, " / ".join(fmt(g) for g in tinfo["N"]), f"{100 * y['linked'] / n:.1f}", f"{100 * y['kept'][j] / n:.1f}", f"{100 * y['kept_linked'][j] / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]",
                             f"{100 * y['kept_linked'][j] / y['linked']:.1f}" if y["linked"] else "-",
                             f"{100 * y['questions_all_linked_kept'][j] / y['questions_linked']:.1f}" if y["questions_linked"] else "-"])
    L += [table(["cell", "kind", "slots", "triple", "N_1 / N_2 / N_3", "link recall %", "pool pairs %", "slot recall % [Wilson 95%]", "linked slots kept in the pool %", "questions with every linked slot kept %"], rows), ""]
    return "\n".join(L)


def cell_curves(c):
    """Per score index: ALL share and mean pool size of every triple, and the per-question detail."""
    return {k: triple_eval(c, k, TRIPLES) for k in range(c["sz1"].shape[1])}


def program(cur, cells, score_idx, p1_cells=None):
    """P1 and P2 of one score over the cells (exact enumeration): cur[cell][k]['all'|'nodes'] are arrays over TRIPLES. P1 is over p1_cells (default all): the cells that can reach
    ALL 0.95 at all; P2, the own optima, the floor and the top are over every cell."""
    A = np.stack([cur[c][score_idx]["all"] for c in cells])           # cells x triples
    N = np.stack([cur[c][score_idx]["nodes"] for c in cells])
    p1c = list(cells) if p1_cells is None else [c for c in cells if c in p1_cells]
    ix = [cells.index(c) for c in p1c]
    feas = (A[ix] >= ALL_MIN).all(axis=0) if ix else np.zeros(len(TRIPLES), bool)
    p1 = None
    if feas.any():
        worst = np.where(feas, N[ix].max(axis=0), np.inf)
        t = int(np.lexsort((np.arange(len(TRIPLES)), N[ix].sum(axis=0), worst))[0])
        p1 = {"triple": list(TRIPLES[t]), "N": [LADDER[i] for i in TRIPLES[t]], "worst_nodes": float(worst[t]), "cells": p1c, "nodes": {c: float(N[i, t]) for i, c in enumerate(cells)},
              "all": {c: float(A[i, t]) for i, c in enumerate(cells)}}
    p2 = {}
    for nmax in P2_N:
        ok = N.max(axis=0) <= nmax
        if ok.any():
            worst_all = np.where(ok, A.min(axis=0), -1.0)
            t = int(np.lexsort((np.arange(len(TRIPLES)), N.max(axis=0), -worst_all))[0])
            p2[str(nmax)] = {"triple": list(TRIPLES[t]), "N": [LADDER[i] for i in TRIPLES[t]], "min_all": float(worst_all[t]), "nodes": {c: float(N[i, t]) for i, c in enumerate(cells)},
                             "all": {c: float(A[i, t]) for i, c in enumerate(cells)}}
        else:
            p2[str(nmax)] = None
    own = {}
    for i, c in enumerate(cells):
        f = A[i] >= ALL_MIN
        j = int(np.flatnonzero(f)[np.argmin(N[i][f])]) if f.any() else -1
        own[c] = None if j < 0 else {"nodes": float(N[i][j]), "triple": list(TRIPLES[j]), "N": [LADDER[x] for x in TRIPLES[j]], "all": float(A[i][j])}
    return {"P1": p1, "P2": p2, "own": own, "floor": {c: {"nodes": float(N[i, 0]), "all": float(A[i, 0])} for i, c in enumerate(cells)},
            "top": {c: {"nodes": float(N[i, len(TRIPLES) - 1]), "all": float(A[i, len(TRIPLES) - 1])} for i, c in enumerate(cells)}}


def choose_final(sc):
    """The declared final rule: the simplest K score within TIE of the least P1 worst-cell nodes; with no K feasible, the K with the largest P2 min ALL at the loosest ceiling."""
    feas = {K: sc[K]["P1"]["worst_nodes"] for K in SIMPLE if sc[K]["P1"]}
    if feas:
        m = min(feas.values())
        K = next(k for k in SIMPLE if k in feas and feas[k] <= TIE * m)
        return {"basis": "P1", "score": K, "triple": sc[K]["P1"]["triple"], "N": sc[K]["P1"]["N"], "worst_nodes": feas[K]}
    best = None
    for K in SIMPLE:
        p = sc[K]["P2"][str(P2_N[-1])]
        if p and (best is None or p["min_all"] > best[1] + 1e-12):
            best = (K, p["min_all"], p)
    if best is None:
        return None
    K, _m, p = best
    return {"basis": f"P2@{P2_N[-1]}", "score": K, "triple": p["triple"], "N": p["N"], "worst_nodes": max(p["nodes"].values())}


def null_verdict(sc, K):
    r = [sc[x]["P1"]["worst_nodes"] for x in ("R0", "R1") if sc[x]["P1"]]
    p = sc[K]["P1"]
    if p is None:
        return {"call": "NO_P1", "random": r}
    if not r:
        return {"call": "BEATS_NULL_INFEASIBLE", "random": r}
    return {"call": "BEATS_NULL" if p["worst_nodes"] <= NULL_MARGIN * min(r) else "NOT_BEATING_NULL", "ratio": p["worst_nodes"] / min(r), "random": r}


# ── markdown ─────────────────────────────────────────────────────────────────


def fmt(x):
    return "-" if x is None else f"{x:,.0f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def md_quarter(rec):
    cells, sc = rec["cells"], rec["scores"]
    L = [f"### GP3 dev quarter: P1 / P2 (written by zgp3s.py solve, {rec['utc']})", "",
         f"Chunks written by zgp3.py sha256 {rec['chunk_zgp3_sha256'][:12]}; ladder {rec['ladder']}; {len(rec['triples'])} triples. Mean distinct nodes per question (base, seeds and kept nodes), ALL = share of questions with every gold in the pool.", ""]
    L += ["**P1 per score (least worst-cell mean nodes with every cell at ALL >= 0.95):**", ""]
    rows = []
    for K in SCORES:
        p = sc[K]["P1"]
        if p is None:
            rows.append([K, "infeasible on the ladder", "-", "-", "-"])
        else:
            w = max(p["nodes"], key=lambda c: p["nodes"][c])
            rows.append([K, fmt(p["worst_nodes"]), " / ".join(fmt(x) for x in p["N"]), w, f"{len(p['cells'])} of {len(cells)}" + ("" if len(p["cells"]) == len(cells) else " (left out: " + ", ".join(c for c in cells if c not in p["cells"]) + ")")])
    L += [table(["score", "worst-cell mean nodes", "N_1 / N_2 / N_3", "worst cell", "cells in P1 (ladder top reaches 0.95)"], rows), ""]
    fin = rec["final"]
    L += [f"**Final rule (declared choice): {fin['score'] if fin else 'none'}" + (f" at N = {fin['N']} ({fin['basis']}, worst-cell mean nodes {fmt(fin['worst_nodes'])}).**" if fin else ".**"), ""]
    L += [f"**Null:** {rec['null']['call']}" + (f" (ratio to the smaller random-order P1 {rec['null']['ratio']:.3f})" if "ratio" in rec["null"] else "") + f"; random-order P1s {[fmt(x) for x in rec['null']['random']]}.", ""]
    if fin:
        K = fin["score"]
        p = sc[K]["P1"] or sc[K]["P2"][str(P2_N[-1])]
        rows = []
        for c in cells:
            o = sc[K]["own"][c]
            ro = [sc[x]["own"][c]["nodes"] for x in ("R0", "R1") if sc[x]["own"][c]]
            rows.append([c, f"{p['all'][c]:.4f}", fmt(p["nodes"][c]), fmt(o["nodes"]) if o else "infeasible", (" / ".join(fmt(x) for x in o["N"])) if o else "-", fmt(min(ro)) if ro else "infeasible",
                         f"{rec['cap_share'][c]:.3f}", "CAPPED" if rec["cap_share"][c] > CAP_SHARE else ""])
        L += [f"**Per cell at the final rule ({K}) and at its own optimum (ALL >= 0.95), beside the better random order's own optimum:**", ""]
        L += [table(["cell", "ALL at the final triple", "nodes at the final triple", f"own optimum nodes ({K})", "own N_1 / N_2 / N_3", "random own optimum nodes", "capped share", ""], rows), ""]
    L += ["**Own optimum nodes per cell and score (ALL >= 0.95 in that cell alone):**", ""]
    rows = [[c] + [fmt(sc[K]["own"][c]["nodes"]) if sc[K]["own"][c] else "infeasible" for K in SCORES] for c in cells]
    L += [table(["cell"] + list(SCORES), rows), ""]
    L += ["**P2: the largest worst-cell ALL under a worst-cell mean-node ceiling:**", ""]
    rows = []
    for K in SCORES:
        r = [K]
        for n in P2_N:
            p = sc[K]["P2"][str(n)]
            if p is None:
                r.append("infeasible")
            else:
                w = min(p["all"], key=lambda c: p["all"][c])
                r.append(f"{p['min_all']:.4f} ({w}; N {' / '.join(fmt(x) for x in p['N'])})")
        rows.append(r)
    L += [table(["score"] + [f"ceiling {n:,}" for n in P2_N], rows), ""]
    L += ["**Floor (N = 0: base, seeds and S whole) and top of the ladder, per cell, score K1:**", ""]
    rows = [[c, fmt(sc["K1"]["floor"][c]["nodes"]), f"{sc['K1']['floor'][c]['all']:.4f}", fmt(sc["K1"]["top"][c]["nodes"]), f"{sc['K1']['top'][c]['all']:.4f}", fmt(UL5_UNIFORM.get(c.split('/')[0])),
             fmt(UL5_ORACLE.get(c.split('/')[0]))] for c in cells]
    L += [table(["cell", "floor nodes", "floor ALL", "ladder-top nodes", "ladder-top ALL", "UL5 uniform nodes (context)", "UL5 oracle nodes (context)"], rows), ""]
    L += ["**Golds by distance and the gather cap, per cell:**", ""]
    rows = [[c] + [f"{x:,}" for x in rec["gold_levels"][c]] + [f"{rec['cap_share'][c]:.3f}"] for c in cells]
    L += [table(["cell", "free or level 0", "level 1", "level 2", "level 3", "beyond (never reached)", "share of questions with a capped hop"], rows), ""]
    ju = rec.get("uncapped")
    if ju:
        cu, su = ju["cells"], ju["scores"]
        L += [f"**Reading over the cells that are not CAPPED (A2, descriptive): {', '.join(cu)}.**", ""]
        rows = []
        for K in SCORES:
            p = su[K]["P1"]
            if p is None:
                rows.append([K, "infeasible on the ladder", "-", "-"])
            else:
                rows.append([K, fmt(p["worst_nodes"]), " / ".join(fmt(x) for x in p["N"]), max(p["nodes"], key=lambda c: p["nodes"][c])])
        L += [table(["score", "worst-cell mean nodes (P1)", "N_1 / N_2 / N_3", "worst cell"], rows), ""]
        fu = ju["final"]
        L += [f"Final rule on these cells: {fu['score'] if fu else 'none'}" + (f" at N = {fu['N']} ({fu['basis']}, worst-cell mean nodes {fmt(fu['worst_nodes'])}); null {ju['null']['call']}." if fu else "."), ""]
        rows = []
        for K in SCORES:
            r = [K]
            for n in P2_N:
                p = su[K]["P2"][str(n)]
                r.append("infeasible" if p is None else f"{p['min_all']:.4f} ({min(p['all'], key=lambda c: p['all'][c])}; N {' / '.join(fmt(x) for x in p['N'])})")
            rows.append(r)
        L += [table(["score"] + [f"P2 ceiling {n:,}" for n in P2_N], rows), ""]
    return "\n".join(L) + md_slots(rec)


# ── solve (the quarter) ──────────────────────────────────────────────────────


def cmd_solve(a):
    t0 = time.time()
    cells, cur, shas, gold_levels, cap_share, gates, chunks = [], {}, set(), {}, {}, {}, {}
    for d, v in CELLS:
        c, sh = load_cell_chunks(d, v, "q")
        if c is None:
            stop(f"{d}/{v}: no quarter chunks")
        name = cell_name(d, v)
        cells.append(name)
        shas |= sh
        chunks[name] = c
        cur[name] = cell_curves(c)
        gl, gf = c["glev"], c["gfree"]
        gold_levels[name] = [int((gf | (gl == 0)).sum())] + [int((~gf & (gl == j)).sum()) for j in (1, 2, 3)] + [int((~gf & (gl >= 4)).sum())]
        cap_share[name] = float(c["capped"].any(axis=1).mean())
        if d not in gates:
            gates[d] = check_extract(d, "q")
        log(f"{name}: {c['nfree'].size} questions, {c['glev'].size} golds, capped share {cap_share[name]:.3f} ({time.time() - t0:.0f} s)")
    if len(shas) != 1:
        stop(f"G3-7: the quarter chunks carry {len(shas)} zgp3 shas")
    rec = {"declared_in": G3.DECLARED, "part": "q", "cells": cells, "ladder": list(LADDER), "triples": [list(t) for t in TRIPLES], "scores": {}, "zgp3s_sha256": script_sha256(),
           "chunk_zgp3_sha256": next(iter(shas)), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gold_levels": gold_levels, "cap_share": cap_share, "extract_gates": gates,
           "questions": {c: int(cur[c][0]["allq"].shape[0]) for c in cells}}
    rec["p1_cells"] = {}
    for k, K in enumerate(SCORES):
        reach = [c for c in cells if cur[c][k]["all"][-1] >= ALL_MIN]                  # the cells whose ladder top reaches ALL 0.95 under this score
        rec["p1_cells"][K] = reach
        rec["scores"][K] = program(cur, cells, k, reach)
        p1 = rec["scores"][K]["P1"]
        log(f"{K}: P1 over {len(reach)} of {len(cells)} cells " + ("infeasible on the ladder" if p1 is None else f"worst-cell mean nodes {p1['worst_nodes']:.0f} at N = {p1['N']}"))
    rec["final"] = choose_final(rec["scores"])
    rec["null"] = null_verdict(rec["scores"], rec["final"]["score"]) if rec["final"] else {"call": "NO_FINAL", "random": []}
    rec["curves"] = {c: {K: {"all": [round(float(x), 5) for x in cur[c][k]["all"]], "nodes": [round(float(x), 1) for x in cur[c][k]["nodes"]]} for k, K in enumerate(SCORES)} for c in cells}
    # A2 item 4: the same programs over the cells that are not CAPPED (descriptive; the final rule above is the declared choice on all cells)
    cu = [c for c in cells if cap_share[c] <= CAP_SHARE]
    rec["uncapped"] = None
    if cu and len(cu) < len(cells):
        ju = {"cells": cu, "p1_cells": {}, "scores": {}}
        for k, K in enumerate(SCORES):
            reach = [c for c in cu if cur[c][k]["all"][-1] >= ALL_MIN]
            ju["p1_cells"][K] = reach
            ju["scores"][K] = program(cur, cu, k, reach)
        ju["final"] = choose_final(ju["scores"])
        ju["null"] = null_verdict(ju["scores"], ju["final"]["score"]) if ju["final"] else {"call": "NO_FINAL", "random": []}
        rec["uncapped"] = ju
        log(f"uncapped reading over {len(cu)} of {len(cells)} cells ({', '.join(cu)}): final {ju['final']}")
    rec["slot_reads"] = slot_reads_all(rec, chunks)                                       # A4: beside, decides nothing (reads the stored gold coverage and slots.npz)
    L2.wjson(OUT / "rule_q.json", rec)
    (OUT / "report_q.md").write_text(md_quarter(rec) + "\n", encoding="utf-8", newline="\n")
    log(f"filed {rel(OUT / 'rule_q.json')} and report_q.md ({time.time() - t0:.0f} s); final {rec['final']}; null {rec['null']['call']}")
    return 0


# ── grade (the rest of dev, once) ────────────────────────────────────────────


def cmd_grade(a):
    t0 = time.time()
    q = json.loads((OUT / "rule_q.json").read_text(encoding="utf-8"))
    fin = q["final"]
    if not fin:
        stop("the quarter has no final rule")
    if (OUT / "rule_r.json").is_file() and not a.again:
        stop("rule_r.json exists: the rest of dev is read once (G3-8)")
    K = fin["score"]
    k = SCORES.index(K)
    cells, rows, shas, chunks_r = [], {}, set(), {}
    for d, v in CELLS:
        c, sh = load_cell_chunks(d, v, "r")
        if c is None:
            stop(f"{d}/{v}: no rest-of-dev chunks")
        name = cell_name(d, v)
        cells.append(name)
        shas |= sh
        chunks_r[name] = c
        check_extract(d, "r")
        need = rest_triples(q, name)                                                     # the triples the grade reads: the final, the cell's own, the P2s (None where the quarter had none)
        rs = c["restricted"]
        if rs:                                                                          # A5: a restricted walk of the final score at exactly the triples the grade reads (and the floor)
            if rs["score"] != K:
                stop(f"{name}: the restricted chunks are of score {rs['score']}, the final rule is {K}")
            ts = list(dict.fromkeys([t for t in need.values() if t is not None] + [(0, 0, 0)]))
            lost = [t for t in ts if t not in rs["triples"]]
            if lost:
                stop(f"{name}: the restricted chunks did not read the triples {lost}")
            kk = 0
        else:
            ts, kk = TRIPLES, k
        ev = triple_eval(c, kk, ts)
        col = {t: i for i, t in enumerate(ts)}
        n = int(ev["allq"].shape[0])
        r = {"questions": n, "capped_share": float(c["capped"].any(axis=1).mean()), "restricted": bool(rs)}
        for nm, t in need.items():
            if t is None:
                r[nm] = None
                continue
            ix, qx = col[t], TRIPLES.index(t)
            kq = int(ev["allq"][:, ix].sum())
            lo, hi = wilson(kq, n)
            r[nm] = {"triple": list(t), "N": [LADDER[x] for x in t], "all": kq / n, "all_ci": [lo, hi], "nodes": float(ev["nodes"][ix]),
                     "quarter_all": q["curves"][name][K]["all"][qx], "quarter_nodes": q["curves"][name][K]["nodes"][qx]}
        if rs:
            r["reoptimised_on_rest"] = None                                              # the grid was not walked: no re-optimisation is possible
            r["floor"] = {"nodes": float(ev["nodes"][col[(0, 0, 0)]]), "all": float(ev["all"][col[(0, 0, 0)]])}
        else:
            f = ev["all"] >= ALL_MIN
            r["reoptimised_on_rest"] = None if not f.any() else {"nodes": float(ev["nodes"][f].min()), "N": [LADDER[x] for x in TRIPLES[int(np.flatnonzero(f)[np.argmin(ev['nodes'][f])])]]}
            r["floor"] = {"nodes": float(ev["nodes"][0]), "all": float(ev["all"][0])}
        rows[name] = r
        log(f"{name}: {n} questions{' (restricted walk)' if rs else ''}, ALL {r['final']['all']:.4f} at the final triple, nodes {r['final']['nodes']:.0f} ({time.time() - t0:.0f} s)")
    if len(shas) != 1:
        stop(f"G3-7: the rest-of-dev chunks carry {len(shas)} zgp3 shas")
    low = [c for c in cells if rows[c]["final"]["all"] < R_MIN_ALL]
    capped = [c for c in cells if rows[c]["capped_share"] > CAP_SHARE]
    call = "HELD" if not low and not capped else ("NOT_HELD" if low else "HELD_BUT_CAPPED")
    rec = {"declared_in": G3.DECLARED, "part": "r", "final": fin, "call": call, "cells_below": low, "cells_capped": capped, "cells": rows, "zgp3s_sha256": script_sha256(),
           "chunk_zgp3_sha256": next(iter(shas)), "quarter_chunk_zgp3_sha256": q["chunk_zgp3_sha256"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec["slot_reads"] = slot_reads_all(q, chunks_r)                                       # A4: beside, decides nothing
    L2.wjson(OUT / "rule_r.json", rec)
    (OUT / "report_r.md").write_text(md_rest(rec) + "\n", encoding="utf-8", newline="\n")
    log(f"filed {rel(OUT / 'rule_r.json')} and report_r.md: {call}; below {low}; capped {capped} ({time.time() - t0:.0f} s)")
    return 0


def md_rest(rec):
    fin, rows = rec["final"], rec["cells"]
    L = [f"### GP3 rest of dev: the quarter's final rule graded once (zgp3s.py grade, {rec['utc']})", "",
         f"Final rule {fin['score']} at N = {fin['N']} ({fin['basis']}). **Call: {rec['call']}** (every cell's ALL at least {R_MIN_ALL}); cells below {rec['cells_below']}, capped {rec['cells_capped']}.", ""]
    body = []
    for c, r in rows.items():
        f = r["final"]
        body.append([c, r["questions"], f"{f['all']:.4f} [{f['all_ci'][0]:.3f}, {f['all_ci'][1]:.3f}]", f"{f['quarter_all']:.4f}", fmt(f["nodes"]), fmt(f["quarter_nodes"]),
                     fmt(r["own_quarter"]["nodes"]) if r["own_quarter"] else "-", f"{r['own_quarter']['all']:.4f}" if r["own_quarter"] else "-",
                     fmt(r["reoptimised_on_rest"]["nodes"]) if r["reoptimised_on_rest"] else ("not read (restricted walk)" if r.get("restricted") else "infeasible"), f"{r['capped_share']:.3f}"])
    L += [table(["cell", "questions", "ALL at the final triple [Wilson 95%]", "quarter ALL", "nodes", "quarter nodes", "nodes at the cell's quarter triple", "its ALL",
                 "nodes re-optimised on the rest (optimistic, not a call)", "capped share"], body), ""]
    L += ["**P2 triples of the quarter, read on the rest (worst-cell ALL / worst-cell mean nodes):**", ""]
    body = []
    for x in P2_N:
        al = [r[f"P2_{x}"]["all"] for r in rows.values() if r[f"P2_{x}"]]
        nd = [r[f"P2_{x}"]["nodes"] for r in rows.values() if r[f"P2_{x}"]]
        body.append([f"{x:,}", f"{min(al):.4f}" if al else "-", fmt(max(nd)) if nd else "-"])
    L += [table(["ceiling", "worst-cell ALL on the rest", "worst-cell mean nodes on the rest"], body), ""]
    return "\n".join(L) + md_slots(rec)


# ── self-test ────────────────────────────────────────────────────────────────


def brute(A, N):
    """P1 and P2 by a plain loop over the triples."""
    best1 = None
    for t in range(A.shape[1]):
        if (A[:, t] >= ALL_MIN).all():
            key = (N[:, t].max(), N[:, t].sum(), t)
            if best1 is None or key < best1[0]:
                best1 = (key, t)
    out2 = {}
    for nmax in P2_N:
        best = None
        for t in range(A.shape[1]):
            if N[:, t].max() <= nmax:
                key = (-A[:, t].min(), N[:, t].max(), t)
                if best is None or key < best[0]:
                    best = (key, t)
        out2[str(nmax)] = None if best is None else best[1]
    return (None if best1 is None else best1[1]), out2


def selftest():
    global TRIPLES
    rng = np.random.default_rng(11)
    trials, checked, _cc, packed, naive_rows, toy_triples = G3.toy_runs()
    saved = TRIPLES
    TRIPLES = toy_triples
    try:
        c = G3.pack_chunk(packed, np.arange(len(packed)))
        cur = cell_curves(c)
        for k in range(len(SCORES)):
            want_all = np.stack([r[0][k] for r in naive_rows])
            want_size = np.stack([r[1][k] for r in naive_rows])
            assert np.array_equal(cur[k]["allq"], want_all), k
            assert np.array_equal(cur[k]["size"], want_size), k
        pr = program({"toy": cur}, ["toy"], 0)
        assert pr["floor"]["toy"]["nodes"] == float(cur[0]["nodes"][0])
        # slot reads (A4) equal a loop on the stored coverage; a question whose golds are all in the pool keeps every slot; slots of questions outside the chunks are ignored
        c2 = dict(c)
        c2["qidx"] = (c["qidx"] * 3 + 1).astype(np.int32)
        Qn, glen = c["qidx"].size, np.bincount(c["gq"], minlength=c["qidx"].size)
        goff = np.searchsorted(c["gq"], np.arange(Qn))
        sq, sa, sb, sk_ = [], [], [], []
        for r_ in range(Qn):
            for a_ in range(int(glen[r_])):
                for b_ in range(a_ + 1, int(glen[r_])):
                    sq.append(int(c2["qidx"][r_]))
                    sa.append(a_)
                    sb.append(b_)
                    sk_.append(int(rng.integers(0, 2)))
        sq += [2, 5, 10 ** 6]
        sa += [0, 0, 0]
        sb += [1, 1, 1]
        sk_ += [0, 0, 1]
        S = {"q": np.asarray(sq, np.int64), "ia": np.asarray(sa, np.int16), "ib": np.asarray(sb, np.int16), "kind": np.asarray(sk_, np.int8), "linked_RK": rng.random(len(sq)) < 0.5}
        nslot = 0
        for k in range(len(SCORES)):
            ev = triple_eval(c2, k, TRIPLES)
            rd = slot_reads(c2, ev["cov"], S, "RK")
            for nm, kd in SLOT_KINDS:
                mine = [i for i in range(len(sq)) if sk_[i] == kd and sq[i] in set(c2["qidx"].tolist())]
                if not mine:
                    assert nm not in rd
                    continue
                want_kept = np.zeros(len(TRIPLES), np.int64)
                want_kl = np.zeros(len(TRIPLES), np.int64)
                for i in mine:
                    r_ = int(np.flatnonzero(c2["qidx"] == sq[i])[0])
                    kp = ev["cov"][goff[r_] + sa[i]] & ev["cov"][goff[r_] + sb[i]]
                    want_kept += kp
                    want_kl += kp & bool(S["linked_RK"][i])
                    assert not (ev["allq"][r_] & ~kp).any()
                assert rd[nm]["slots"] == len(mine) and rd[nm]["kept"] == want_kept.tolist() and rd[nm]["kept_linked"] == want_kl.tolist(), (k, nm)
                assert rd[nm]["linked"] == int(sum(bool(S["linked_RK"][i]) for i in mine))
                qs_ = sorted({sq[i] for i in mine if bool(S["linked_RK"][i])})                                  # the questions whose linked slots are all kept, by a loop
                want_qa = np.zeros(len(TRIPLES), np.int64)
                for qq in qs_:
                    r_ = int(np.flatnonzero(c2["qidx"] == qq)[0])
                    okq = np.ones(len(TRIPLES), bool)
                    for i in mine:
                        if sq[i] == qq and bool(S["linked_RK"][i]):
                            okq &= ev["cov"][goff[r_] + sa[i]] & ev["cov"][goff[r_] + sb[i]]
                    want_qa += okq
                assert rd[nm]["questions_linked"] == len(qs_) and rd[nm]["questions_all_linked_kept"] == want_qa.tolist(), (k, nm)
                nslot += len(mine)
        assert nslot > 200, nslot        # A5: a packed restricted chunk grades like the full grid at the triples it read (sizes, ALL, per-gold coverage), and a triple it did not read stops the grade
        import zgp3r as R
        G1_, H_, rr = G3.G1, G3.H, np.random.default_rng(3)
        ladder_t, nrest = (0, 1, 3, 6), 0
        all_t = [(i, j, k_) for i in range(4) for j in range(i + 1) for k_ in range(j + 1)]
        for _t in range(14):
            n = int(rr.integers(14, 46))
            keys = G1_.toy_graph(rr, n, int(rr.integers(2 * n, 5 * n)))
            if keys.size < 8:
                continue
            Gt = G1_.Graph([H_.build_comp(keys, n)], n)
            lev = np.full(n, G1_.DEFAULT_LEV, np.int8)
            mod = int(rr.choice([4, 16, 997]))
            dotf = lambda kk_, mod=mod, n=n: (((np.minimum(kk_ // n, kk_ % n) * 7919 + np.maximum(kk_ // n, kk_ % n) * 104729) % mod).astype(np.float32) / np.float32(mod))  # noqa: E731
            isS = lambda a_, c_: ((np.minimum(a_, c_) * 31 + np.maximum(a_, c_) * 17) % 5 == 0)  # noqa: E731
            ctx = G3.Ctx(Gt, H_.Walker(n), lev, n, dotf, isS, ladder=ladder_t, ubin=lambda c_: ((np.asarray(c_) * 5 + 3) % 7).astype(np.int8), seedw=lambda s_: (np.asarray(s_) * 13 + 5) % 9 + 1)
            Kr = SCORES[int(rr.integers(0, len(SCORES)))]
            pick = sorted({all_t[i] for i in rr.choice(len(all_t), 4, replace=False)} | {(0, 0, 0)})
            fulls, rests = [], []
            for qn in range(5):
                seeds, golds = np.unique(rr.integers(0, n, int(rr.integers(1, 4)))), np.unique(rr.integers(0, n, int(rr.integers(1, 6))))
                ctx.set_question(None, None, np.unique(rr.integers(0, n, int(rr.integers(1, 6)))))
                fulls.append(G3.run_question(ctx, seeds, golds, qn))
                rests.append(R.run_question_r(ctx, seeds, golds, qn, Kr, pick))
            cf, cr = G3.pack_chunk(fulls, np.arange(5)), G3.pack_chunk(rests, np.arange(5))
            ef, er = triple_eval(cf, SCORES.index(Kr), pick), triple_eval(cr, 0, pick)
            assert np.array_equal(ef["allq"], er["allq"]) and np.array_equal(ef["size"], er["size"]) and np.array_equal(ef["cov"], er["cov"]), (_t, Kr, pick)
            nrest += 1
            miss = [t for t in all_t if t not in pick and (bool((cr["sz1"][:, 0][:, t[0]] < 0).any()) or bool((cr["sz2"][:, 0][:, t[0], t[1]] < 0).any())
                                                             or bool((cr["sz3"][:, 0][:, t[0], t[1], t[2]] < 0).any()))]            # a triple whose size the restricted walk did not store
            if miss:
                try:
                    triple_eval(cr, 0, [miss[0]])
                except SystemExit:
                    pass
                else:
                    raise AssertionError("a triple the restricted walk did not read was graded")
        assert nrest > 8, nrest
        # P1 / P2 equal the plain search on random matrices
        for _ in range(300):
            nc = int(rng.integers(1, 6))
            nt = len(TRIPLES)
            A = np.round(rng.random((nc, nt)) ** 0.3, 2)
            A[:, int(rng.integers(0, nt))] = 1.0
            N = np.round(rng.random((nc, nt)) * 9000)
            N[rng.random((nc, nt)) < 0.2] = 2500.0
            fake = {f"c{i}": {0: {"all": A[i], "nodes": N[i]}} for i in range(nc)}
            res = program(fake, list(fake), 0)
            b1, b2 = brute(A, N)
            assert (res["P1"] is None) == (b1 is None)
            if b1 is not None:
                assert res["P1"]["triple"] == list(TRIPLES[b1]), (res["P1"]["triple"], TRIPLES[b1])
            for nmax in P2_N:
                assert (res["P2"][str(nmax)] is None) == (b2[str(nmax)] is None)
                if b2[str(nmax)] is not None:
                    assert res["P2"][str(nmax)]["triple"] == list(TRIPLES[b2[str(nmax)]])
        for _ in range(100):
            nc = int(rng.integers(2, 6))
            nt = len(TRIPLES)
            A = np.round(rng.random((nc, nt)) ** 0.3, 2)
            A[:, int(rng.integers(0, nt))] = 1.0
            N = np.round(rng.random((nc, nt)) * 9000)
            fake = {f"c{i}": {0: {"all": A[i], "nodes": N[i]}} for i in range(nc)}
            sub_cells = [f"c{i}" for i in range(nc) if rng.random() < 0.6] or ["c0"]
            res = program(fake, list(fake), 0, sub_cells)
            ids = [int(c[1:]) for c in sub_cells]
            b1, _b2 = brute(A[ids], N[ids])
            assert (res["P1"] is None) == (b1 is None)
            if b1 is not None:
                assert res["P1"]["triple"] == list(TRIPLES[b1])
        # a wrong objective is caught: the worst cell first, not the sum, on a planted case (the sum would take the second triple)
        nt = len(TRIPLES)
        A = np.ones((2, nt))
        N = np.full((2, nt), 99.0)
        N[:, :2] = [[10.0, 12.0], [10.0, 1.0]]
        res = program({"c0": {0: {"all": A[0], "nodes": N[0]}}, "c1": {0: {"all": A[1], "nodes": N[1]}}}, ["c0", "c1"], 0)
        assert res["P1"]["triple"] == list(TRIPLES[0]) and brute(A, N)[0] == 0
        # the final-rule tie order: the simpler score wins within TIE
        sc = {K: {"P1": {"worst_nodes": w, "triple": [0, 0, 0], "N": [0, 0, 0], "nodes": {}, "all": {}}, "P2": {}} for K, w in (("K1", 100.0), ("K2", 103.0), ("K3", 99.0), ("K4", 130.0), ("PP", 140.0))}
        assert choose_final(sc)["score"] == "K2"
        sc["K4"]["P1"]["worst_nodes"] = 103.0
        assert choose_final(sc)["score"] == "K4"
        sc["K4"]["P1"] = None
        sc["K2"]["P1"]["worst_nodes"] = 120.0
        assert choose_final(sc)["score"] == "K1"
    finally:
        TRIPLES = saved
    log(f"selftest PASS (zgp3s): the chunk grading equals the naive pools of {checked} walks on {trials} toy graphs; P1 and P2 equal a plain search on 300 random matrices; the final-rule tie order holds; the slot reads equal a loop on {nslot} (slot, score) reads of the toy chunks; a restricted chunk grades like the full grid at the triples it read on {nrest} toy sets and a triple it did not read stops")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("solve")
    g = sub.add_parser("grade")
    g.add_argument("--again", action="store_true")
    rs = sub.add_parser("rest-spec")
    rs.add_argument("--ds", required=True)
    a = ap.parse_args(argv)
    return {"selftest": lambda _a: selftest(), "solve": cmd_solve, "grade": cmd_grade, "rest-spec": cmd_rest_spec}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
