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
SIMPLE = ("K4", "K2", "K1", "K3")                         # the simplicity order of the scores for ties
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


def load_cell_chunks(d, v, part):
    """All chunks of one cell and part, concatenated; the zgp3 sha of the chunks is returned beside."""
    wd = OUT / d / "work"
    files = sorted(wd.glob(f"{v}_{part}_*.npz"))
    if not files:
        return None, None
    parts, nq, shas = [], None, set()
    for f in files:
        with np.load(f) as z:
            shas.add(str(z["zsha"]))
            parts.append({k: z[k] for k in z.files if k not in ("zsha", "cstats")})
            nq = int(z["nq"])
    want = G3.part_indices(nq, part)
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
    if seen != set(range(n_sh)):
        stop(f"{d}/{part}: shards {sorted(seen)} of {n_sh}")
    return {"files": [f.name for f in fs], "gates": r["gates"]}


def cell_curves(c):
    """Per score index: ALL share and mean pool size of every triple, and the per-question detail."""
    Q = c["nfree"].size
    NS = c["sz1"].shape[1]
    t1 = np.array([t[0] for t in TRIPLES])
    t2 = np.array([t[1] for t in TRIPLES])
    t3 = np.array([t[2] for t in TRIPLES])
    out = {}
    for k in range(NS):
        size = c["nfree"][:, None].astype(np.int64) + c["sz1"][:, k][:, t1] + c["sz2"][:, k][:, t1, t2] + c["sz3"][:, k][:, t1, t2, t3]
        if bool((size < 0).any()):
            stop("a size of an invalid triple was read")
        glev, gfree = c["glev"], c["gfree"]
        cov = np.zeros((glev.size, len(TRIPLES)), bool)
        cov |= (gfree | (glev == 0))[:, None]
        m1 = glev == 1
        cov[m1] |= c["gb1"][m1, k][:, None] <= t1[None, :]
        m2 = glev == 2
        cov[m2] |= c["gb2"][m2, k][:, t1] <= t2[None, :]
        m3 = glev == 3
        cov[m3] |= c["gb3"][m3, k][:, t1, t2] <= t3[None, :]
        unc = np.zeros((Q, len(TRIPLES)), np.int64)
        np.add.at(unc, c["gq"], (~cov).astype(np.int64))
        allq = unc == 0
        out[k] = {"all": allq.mean(axis=0), "nodes": size.mean(axis=0), "allq": allq, "size": size}
    return out


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
    return "\n".join(L)


# ── solve (the quarter) ──────────────────────────────────────────────────────


def cmd_solve(a):
    t0 = time.time()
    cells, cur, shas, gold_levels, cap_share, gates = [], {}, set(), {}, {}, {}
    for d, v in CELLS:
        c, sh = load_cell_chunks(d, v, "q")
        if c is None:
            stop(f"{d}/{v}: no quarter chunks")
        name = cell_name(d, v)
        cells.append(name)
        shas |= sh
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
    ti = TRIPLES.index(tuple(fin["triple"]))
    cells, rows, shas = [], {}, set()
    for d, v in CELLS:
        c, sh = load_cell_chunks(d, v, "r")
        if c is None:
            stop(f"{d}/{v}: no rest-of-dev chunks")
        name = cell_name(d, v)
        cells.append(name)
        shas |= sh
        check_extract(d, "r")
        cur = cell_curves(c)[k]
        n = int(cur["allq"].shape[0])
        own_q = q["scores"][K]["own"][name]
        oi = TRIPLES.index(tuple(own_q["triple"])) if own_q else None
        r = {"questions": n, "capped_share": float(c["capped"].any(axis=1).mean())}
        for nm, ix in (("final", ti), ("own_quarter", oi)) + tuple((f"P2_{x}", TRIPLES.index(tuple(q["scores"][K]["P2"][str(x)]["triple"])) if q["scores"][K]["P2"][str(x)] else None) for x in P2_N):
            if ix is None:
                r[nm] = None
                continue
            kq = int(cur["allq"][:, ix].sum())
            lo, hi = wilson(kq, n)
            r[nm] = {"triple": list(TRIPLES[ix]), "N": [LADDER[x] for x in TRIPLES[ix]], "all": kq / n, "all_ci": [lo, hi], "nodes": float(cur["nodes"][ix]),
                     "quarter_all": q["curves"][name][K]["all"][ix], "quarter_nodes": q["curves"][name][K]["nodes"][ix]}
        f = cur["all"] >= ALL_MIN
        r["reoptimised_on_rest"] = None if not f.any() else {"nodes": float(cur["nodes"][f].min()), "N": [LADDER[x] for x in TRIPLES[int(np.flatnonzero(f)[np.argmin(cur['nodes'][f])])]]}
        rows[name] = r
        log(f"{name}: {n} questions, ALL {r['final']['all']:.4f} at the final triple, nodes {r['final']['nodes']:.0f} ({time.time() - t0:.0f} s)")
    if len(shas) != 1:
        stop(f"G3-7: the rest-of-dev chunks carry {len(shas)} zgp3 shas")
    low = [c for c in cells if rows[c]["final"]["all"] < R_MIN_ALL]
    capped = [c for c in cells if rows[c]["capped_share"] > CAP_SHARE]
    call = "HELD" if not low and not capped else ("NOT_HELD" if low else "HELD_BUT_CAPPED")
    rec = {"declared_in": G3.DECLARED, "part": "r", "final": fin, "call": call, "cells_below": low, "cells_capped": capped, "cells": rows, "zgp3s_sha256": script_sha256(),
           "chunk_zgp3_sha256": next(iter(shas)), "quarter_chunk_zgp3_sha256": q["chunk_zgp3_sha256"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
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
                     fmt(r["reoptimised_on_rest"]["nodes"]) if r["reoptimised_on_rest"] else "infeasible", f"{r['capped_share']:.3f}"])
    L += [table(["cell", "questions", "ALL at the final triple [Wilson 95%]", "quarter ALL", "nodes", "quarter nodes", "nodes at the cell's quarter triple", "its ALL",
                 "nodes re-optimised on the rest (optimistic, not a call)", "capped share"], body), ""]
    L += ["**P2 triples of the quarter, read on the rest (worst-cell ALL / worst-cell mean nodes):**", ""]
    body = []
    for x in P2_N:
        al = [r[f"P2_{x}"]["all"] for r in rows.values() if r[f"P2_{x}"]]
        nd = [r[f"P2_{x}"]["nodes"] for r in rows.values() if r[f"P2_{x}"]]
        body.append([f"{x:,}", f"{min(al):.4f}" if al else "-", fmt(max(nd)) if nd else "-"])
    L += [table(["ceiling", "worst-cell ALL on the rest", "worst-cell mean nodes on the rest"], body), ""]
    return "\n".join(L)


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
        sc = {K: {"P1": {"worst_nodes": w, "triple": [0, 0, 0], "N": [0, 0, 0], "nodes": {}, "all": {}}, "P2": {}} for K, w in (("K1", 100.0), ("K2", 103.0), ("K3", 99.0), ("K4", 130.0))}
        assert choose_final(sc)["score"] == "K2"
        sc["K4"]["P1"]["worst_nodes"] = 103.0
        assert choose_final(sc)["score"] == "K4"
        sc["K4"]["P1"] = None
        sc["K2"]["P1"]["worst_nodes"] = 120.0
        assert choose_final(sc)["score"] == "K1"
    finally:
        TRIPLES = saved
    log(f"selftest PASS (zgp3s): the chunk grading equals the naive pools of {checked} walks on {trials} toy graphs; P1 and P2 equal a plain search on 300 random matrices; the final-rule tie order holds")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("solve")
    g = sub.add_parser("grade")
    g.add_argument("--again", action="store_true")
    a = ap.parse_args(argv)
    return {"selftest": lambda _a: selftest(), "solve": cmd_solve, "grade": cmd_grade}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
