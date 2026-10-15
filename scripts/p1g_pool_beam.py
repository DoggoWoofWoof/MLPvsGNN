"""P1g: one pool rule for all six datasets, today's pool plus nodes reached through the base's own links
(docs/P1G_POOL_BEAM.md). Pools only: nothing trains or reads, and nothing is fit.

    python outputs/host_ops/pylib_run.py scripts/p1g_pool_beam.py stats DATASET --host [--limit Q]
                                         D5's carves (s1sel; webqsp's s1eval beside), every candidate
    python scripts/p1g_pool_beam.py verdict            section 5's choice on the five s1sel carves
    python outputs/host_ops/pylib_run.py scripts/p1g_pool_beam.py eval DATASET --host
                                         after the verdict: the chosen candidate and today's on s1eval (beside)
    python scripts/p1g_pool_beam.py report             the verdict with s1eval beside
    python scripts/p1g_pool_beam.py --selftest

A candidate is (f, X). The pool is the question's base and seeds B (step 1's, as every pool stage opens them), the
first round(f x |fill|) of KIND_T's nodes outside B (today's fill is P1c2's: those nodes cut at today's size, f = 1;
f > 1 goes on along the same order, up to its end; P1e's walks, kinds and orders, imported unchanged), and an extra
set X of nodes outside B:
    none                         nothing more (f = 1 with none is today's pool)
    beam (score, P, m, steps)    D5b's per-parent beams: B's first P nodes by first-stage weight, each one's first m
                                 neighbours outside B by cos(q, v) or by the parent's residual; steps = 2 adds each
                                 beam node's first m the same way
    cut (order, k)               the first k of B's one-link neighbours under one of D5b's orders (w, rrf, cos, res)
rows = |B u fill_f u X|; a gold is held when it is in that set. --limit Q: a development run on the first Q questions;
it writes nothing and logs no number.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import d5_gold_reach as D5  # noqa: E402
import d5b_neighbour_orders as D5B  # noqa: E402
import p1f_pool_size as F  # noqa: E402

DECLARED = "docs/P1G_POOL_BEAM.md"
OUT = ROOT / "outputs" / "p1g"
VERDICT = OUT / "verdict.json"
TRAIN5 = F.TRAIN5
SIX = F.SIX
TARGET = 0.95                               # the user's: ALL >= 0.95 on every dataset
TOL = 0.005                                 # within this of a dataset's ceiling counts as the ceiling (section 5)
FILL_F = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)   # fractions of today's fill; above 1, KIND_T's order goes on
CUT_ORDERS = ("w", "rrf", "cos", "res")
CUT_K = (50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 7000)
TODAY = "f1+none"
EMPTY = np.empty(0, np.int64)
log, utc, write_json, sha_file = D5.log, D5.utc, D5.write_json, D5.sha_file


def script_sha256():
    import hashlib
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the candidates ───────────────────────────────────────────────────────────


def extras():
    """[(name, family, params)] in section 4's order."""
    out = [("none", "NONE", ())]
    for sc in D5B.BEAM_SCORES:
        for P in D5B.P_GRID:
            for m in D5B.M_GRID:
                out.append((f"beam1_{sc}_P{P}_m{m}", "BEAM", (sc, P, m, 1)))
                if P <= D5B.P2_MAX and m <= D5B.M2_MAX:
                    out.append((f"beam2_{sc}_P{P}_m{m}", "BEAM", (sc, P, m, 2)))
    for o in CUT_ORDERS:
        for k in CUT_K:
            out.append((f"cut_{o}_{k}", "CUT", (o, k)))
    return out


def candidates():
    """[(name, f, extra name)]: every fill fraction with every extra."""
    return [(f"f{f:g}+{nm}", f, nm) for f in FILL_F for nm, _fam, _prm in extras()]


def extra_sets(adj, B, w_B, q, read, c, want):
    """{extra name: its nodes (unique, outside B)} for the extras in want."""
    ex = {nm: (fam, prm) for nm, fam, prm in extras() if nm in want}
    out = {nm: EMPTY for nm, (fam, _p) in ex.items() if fam == "NONE"}
    beams = [(nm, prm) for nm, (fam, prm) in ex.items() if fam == "BEAM"]
    if beams:
        Pmax, mmax = max(p[1] for _n, p in beams), max(p[2] for _n, p in beams)
        Bord = B[np.lexsort((B, -np.asarray(w_B, np.float64)))][:Pmax]
        top1 = D5B.top_children(adj, Bord, B, q, D5B.unit(read(Bord)), read, mmax)
        two = [p for _n, p in beams if p[3] == 2]
        top2, b1max = None, EMPTY
        if two:
            P2, m2 = max(p[1] for p in two), max(p[2] for p in two)
            b1max = D5B.cat([top1[sc][1][(top1[sc][0] < P2) & (top1[sc][2] < m2)] for sc in D5B.BEAM_SCORES])
            if b1max.size:
                top2 = D5B.top_children(adj, b1max, B, q, D5B.unit(read(b1max)), read, m2)
        for nm, (sc, P, m, steps) in beams:
            ps, vs, rk = top1[sc]
            b = np.unique(vs[(ps < P) & (rk < m)])
            if steps == 2 and top2 is not None and b.size:
                ps2, vs2, rk2 = top2[sc]
                b = D5B.cat([b, vs2[np.isin(b1max[ps2], b) & (rk2 < m)]])
            out[nm] = b
    cuts = [(nm, prm) for nm, (fam, prm) in ex.items() if fam == "CUT"]
    if cuts:
        n1, cnt, wsum, u_par, inv = D5B.n1_table(adj, B, w_B)
        if n1.size:
            r = D5B.orders_of(n1, cnt, wsum, u_par, inv, q, D5B.unit(read(B)), D5B.unit(read(n1)), c)
        for nm, (o, k) in cuts:
            out[nm] = n1[r[o] <= k] if n1.size else EMPTY
    return out


def positions_in(X, fill):
    """Each X node's index in fill, or fill.size when fill does not hold it."""
    X = np.asarray(X, np.int64)
    out = np.full(X.size, fill.size, np.int64)
    if fill.size and X.size:
        srt = np.argsort(fill, kind="stable")
        so = fill[srt]
        k = np.minimum(np.searchsorted(so, X), so.size - 1)
        hit = so[k] == X
        out[hit] = srt[k[hit]]
    return out


def evaluate_question(B, full, base, golds, sets, cands):
    """Per candidate: rows |B u full[:n_f] u X| and, per gold, whether that set holds it. full is KIND_T's order
    outside B, base = |today's fill| (its first base nodes), n_f = min(|full|, round(f x base))."""
    nf = {f: min(full.size, int(np.rint(f * base))) for f in FILL_F}
    inB = np.isin(golds, B)
    gpos = positions_in(golds, full)
    gin = {ex: np.isin(golds, X) for ex, X in sets.items()}
    xpos = {ex: positions_in(X, full) for ex, X in sets.items()}
    rows = np.empty(len(cands), np.int32)
    held = np.empty((golds.size, len(cands)), bool)
    for i, (_n, f, ex) in enumerate(cands):
        k = nf[f]
        rows[i] = B.size + k + int((xpos[ex] >= k).sum())
        held[:, i] = inB | (gpos < k) | gin[ex]
    return rows, held


def summarize(a, cands):
    """{candidate: {ALL, recall, rows, rows_p90}} for one carve's arrays."""
    ptr = a["ptr"]
    ng = np.diff(ptr)
    nq = ng.size
    qi = np.repeat(np.arange(nq), ng)
    has = ng > 0
    out = {}
    for i, (nm, _f, _ex) in enumerate(cands):
        hits = np.bincount(qi, weights=a["held"][:, i].astype(np.float64), minlength=nq)
        rows = a["rows"][:, i].astype(np.float64)
        out[nm] = {"ALL": round(float((has & (hits == ng)).mean()), 4),
                   "recall": round(float(np.where(has, hits / np.maximum(ng, 1), 0.0).mean()), 4),
                   "rows": round(float(rows.mean()), 1), "rows_p90": round(float(np.quantile(rows, 0.9)), 1)}
    return out


# ── stats and eval (the host) ────────────────────────────────────────────────


def run_stage(name, carves, cands, out_name, limit=0, threads=2):
    import p1e_anchor_kind as K          # P1e's walks, kinds and orders; through it P1d's sources (imports numba)
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from mp_retrieval.m3b_features import DenseNodes
    D, G, E, S4 = K.D, K.G, K.E, K.S4
    op = S4.Opened()
    ds = op.canonical.Dataset(name, root=str(op.served))
    nonly = K.name_only(ds.nodes(), ds.n_nodes)
    nodes = DenseNodes(ds.embeddings("dense", "docs"))
    want = {ex for _n, _f, ex in cands}
    cache = {}
    rec = {"declared_in": DECLARED, "stage": "P1g", "dataset": name, "script_sha256": script_sha256(),
           "d5b_script_sha256": D5B.script_sha256(), "p1f_script_sha256": F.script_sha256(),
           "p1e_script_sha256": K.script_sha256(), "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"],
           "candidates": [c[0] for c in cands], "carves": {}}

    def sink(name_, carve, pop, prep, U, BQ, bs, inc, linked, union, constant, families, construction):
        t = time.time()
        stores, unions = G.open_unions(cache, op, name, union)
        adj = [(np.ascontiguousarray(stores[f].indptr, np.int64), np.ascontiguousarray(stores[f].col))
               for f in S4.FAMILIES]
        n_st = int(unions["st"].deg.size)
        st = F.load_stats(name, carve)
        nq = min(limit, len(U)) if limit else len(U)
        golds = [np.unique(np.asarray(g, np.int64)) for g in pop.golds[:nq]]
        ptr = np.concatenate([[0], np.cumsum([g.size for g in golds])]).astype(np.int64)
        if not np.array_equal(st["ptr"][:nq + 1], ptr):
            raise SystemExit(f"{name}/{carve}: P1f's golds are not the carve's")
        ti = [c[0] for c in cands].index(TODAY)
        rows, held, fills, bad = [], [], np.zeros(nq, np.int64), 0
        for lo in range(0, nq, S4.BATCH):
            js = list(range(lo, min(nq, lo + S4.BATCH)))
            srcs = D.sources_batch(prep, js, linked, unions, constant, threads)
            steered = K.steered_batch(prep, js, linked, unions, srcs, constant, threads)
            for i, j in enumerate(js):
                src, _nn, (L, Ls) = srcs[i]
                route = D.routes(L, Ls)
                knd = K.kind(Ls, prep.seeds[j], nonly)
                o = np.asarray(K.p1e_orders(src, steered[i], route, knd, n_st, constant)[0][F.CONSTRUCTION], np.int64)
                B = np.unique(np.asarray(bs[j], np.int64))
                if B.size != int(st["bs"][j]) or np.asarray(inc[j]).size != int(st["inc"][j]):
                    raise SystemExit(f"{name}/{carve} question {j}: |B| or |I| is not P1f's")
                take = max(0, int(st["inc"][j]) - B.size)
                full = F.outside(o, B)
                fill = full[:take]
                fills[j] = fill.size
                w_B = D5.weights_of(K.steer(prep, j, constant), B)
                q = D5B.unit(np.asarray(prep.qemb[j], np.float64))
                sets = extra_sets(adj, B, w_B, q, nodes.read, constant, want)
                r, h = evaluate_question(B, full, fill.size, golds[j], sets, cands)
                today = st["depth"][ptr[j]:ptr[j + 1]] <= take
                bad += int(not (np.array_equal(h[:, ti], today) and int(r[ti]) == B.size + fill.size))
                rows.append(r)
                held.append(h)
            if (lo // S4.BATCH) % 10 == 9:
                log(f"{name}/{carve}: {js[-1] + 1}/{nq} ({time.time() - t:.0f}s)")
        if bad:
            raise SystemExit(f"{name}/{carve}: today's pool differs from P1f's filed depths on {bad} questions")
        a = {"rows": np.stack(rows) if rows else np.zeros((0, len(cands)), np.int32),
             "held": np.concatenate(held) if held else np.zeros((0, len(cands)), bool),
             "ptr": ptr, "bs": st["bs"][:nq].astype(np.int64), "fill": fills}
        s = summarize(a, cands)
        if limit:
            log(f"{name}/{carve}: development run on {nq} questions ({time.time() - t:.0f}s): KIND_T's order, the "
                "beams and cuts ran; today's pool equals P1f's; nothing written")
            return
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"{out_name}_{name}__{carve}.npz"
        tmp = OUT / f"{out_name}_{name}__{carve}.tmp.npz"
        np.savez_compressed(tmp, **a)
        os.replace(tmp, p)
        rec["carves"][f"{name}__{carve}"] = {"questions": nq, "file_sha256": sha_file(p), "summary": s,
                                             "checks": {"today_equals_p1f": True, "b_and_i_equal_p1f": True},
                                             "seconds": round(time.time() - t, 1), "utc": utc()}
        write_json(OUT / f"{out_name}_{name}.json", rec)
        log(f"{name}/{carve} ({nq} q, {time.time() - t:.0f}s): written")

    E.run_carves(op, name, set(carves), ("A3",), threads, sink)
    return 0


def stats_stage(name, limit=0, threads=2):
    if name not in SIX:
        raise SystemExit(f"dataset must be one of {SIX}")
    return run_stage(name, D5.carves_of(name), candidates(), "stats", limit, threads)


def eval_stage(name, limit=0, threads=2):
    if name not in TRAIN5:
        raise SystemExit(f"eval reads the five training datasets' s1eval; webqsp's is in its stats ({TRAIN5})")
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    if v.get("script_sha256") != script_sha256():
        raise SystemExit(f"{VERDICT}: not written by this script")
    chosen = v["chosen"]
    cands = [c for c in candidates() if c[0] in {TODAY, chosen}]
    return run_stage(name, ("s1eval",), cands, "eval", limit, threads)


# ── the verdict ──────────────────────────────────────────────────────────────


def load_summary(name, out_name="stats"):
    rec = json.loads((OUT / f"{out_name}_{name}.json").read_text(encoding="utf-8"))
    if rec.get("script_sha256") != script_sha256():
        raise SystemExit(f"{out_name}_{name}.json: not written by this script")
    out = {}
    for key, e in rec["carves"].items():
        p = OUT / f"{out_name}_{key}.npz"
        if p.exists() and sha_file(p) != e["file_sha256"]:
            raise SystemExit(f"{p}: not the file the record names")
        if not all(e["checks"].values()):
            raise SystemExit(f"{p}: a check failed")
        out[key] = e["summary"]
    return out


def choose(table, weights, names=TRAIN5):
    """Section 5. table: {candidate: {ds: {ALL, rows, ...}}} in section 4's order. Returns (verdict, chosen,
    ceilings, targets, per-candidate summary)."""
    cands = list(table)
    ceil = {d: max(table[c][d]["ALL"] for c in cands) for d in names}
    tgt = {d: round(min(TARGET, ceil[d] - TOL), 4) for d in names}
    summ, feas, rest = {}, [], []
    for i, c in enumerate(cands):
        cost = F.cost({d: table[c][d]["rows"] for d in names}, weights)
        gap = min(table[c][d]["ALL"] - tgt[d] for d in names)
        ok = gap >= -1e-9
        summ[c] = {"cost": round(cost, 1), "meets": ok, "worst_gap": round(gap, 4)}
        (feas if ok else rest).append((cost, i, c, gap))
    if feas:
        pick = min(feas)[2]
        verdict = "TARGET_MET" if all(tgt[d] >= TARGET for d in names) else "CEILING_MET"
    else:
        pick = min(rest, key=lambda x: (-round(x[3], 4), x[0], x[1]))[2]
        verdict = "NONE_MEETS"
    return verdict, pick, ceil, tgt, summ


def verdict_stage():
    table = {}
    for d in TRAIN5:
        s = load_summary(d)[f"{d}__s1sel"]
        for c, e in s.items():
            table.setdefault(c, {})[d] = e
    order = [c[0] for c in candidates()]
    table = {c: table[c] for c in order}
    w = F.weights()
    verdict, pick, ceil, tgt, summ = choose(table, w)
    beside = load_summary("webqsp")["webqsp__s1eval"]
    fams = {}
    for c in order:
        fam = c.split("+")[1].split("_")[0] + ("" if "+none" in c else "_" + c.split("_")[1])
        key = c.split("+")[0] + ":" + fam
        if summ[c]["meets"] and (key not in fams or summ[c]["cost"] < summ[fams[key]]["cost"]):
            fams[key] = c
    per_ds = {d: min((table[c][d]["rows"], i, c) for i, c in enumerate(order) if table[c][d]["ALL"] >= tgt[d] - 1e-9)[2]
              for d in TRAIN5}
    rec = {"declared_in": DECLARED, "stage": "P1g", "script_sha256": script_sha256(), "verdict": verdict,
           "chosen": pick, "today": TODAY, "ceilings": ceil, "targets": tgt, "weights": w,
           "chosen_s1sel": {d: table[pick][d] for d in TRAIN5}, "today_s1sel": {d: table[TODAY][d] for d in TRAIN5},
           "cost": {"chosen": summ[pick]["cost"], "today": summ[TODAY]["cost"]},
           "beside_webqsp_s1eval": {"chosen": beside[pick], "today": beside[TODAY]},
           "beside_cheapest_per_family": {k: {"candidate": c, "cost": summ[c]["cost"]} for k, c in fams.items()},
           "beside_per_dataset_cheapest": {d: {"candidate": c, **table[c][d]} for d, c in per_ds.items()},
           "summary": summ, "utc": utc()}
    write_json(VERDICT, rec)
    log(f"P1g: {verdict}, chosen {pick} (cost {summ[pick]['cost']} against today's {summ[TODAY]['cost']})")
    return 0


def report_stage():
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    lines = [f"# P1g's verdict: **{v['verdict']}**", "",
             f"Chosen on s1sel: `{v['chosen']}`; today's pool is `{v['today']}`. Targets: min(0.95, ceiling - 0.005).",
             "", "| dataset | carve | target | today ALL / rows | chosen ALL / rows |", "| --- | --- | ---: | --- | --- |"]
    for d in TRAIN5:
        t, c = v["today_s1sel"][d], v["chosen_s1sel"][d]
        lines.append(f"| {d} | s1sel | {v['targets'][d]:.4f} | {t['ALL']:.4f} / {t['rows']:,.0f} | "
                     f"{c['ALL']:.4f} / {c['rows']:,.0f} |")
    for d in TRAIN5:
        p = OUT / f"eval_{d}.json"
        if p.exists():
            s = load_summary(d, "eval")[f"{d}__s1eval"]
            lines.append(f"| {d} | s1eval (beside) | | {s[v['today']]['ALL']:.4f} / {s[v['today']]['rows']:,.0f} | "
                         f"{s[v['chosen']]['ALL']:.4f} / {s[v['chosen']]['rows']:,.0f} |")
    b = v["beside_webqsp_s1eval"]
    lines.append(f"| webqsp | s1eval (beside) | | {b['today']['ALL']:.4f} / {b['today']['rows']:,.0f} | "
                 f"{b['chosen']['ALL']:.4f} / {b['chosen']['rows']:,.0f} |")
    lines += ["", f"Cost (the question-weighted mean of the five datasets' mean rows): chosen {v['cost']['chosen']:,}, "
              f"today {v['cost']['today']:,}.", "", "Beside: the cheapest candidate meeting every target, per family:",
              "", "| family | candidate | cost |", "| --- | --- | ---: |"]
    for k, e in sorted(v["beside_cheapest_per_family"].items(), key=lambda kv: kv[1]["cost"]):
        lines.append(f"| {k} | `{e['candidate']}` | {e['cost']:,} |")
    lines += ["", "Beside: each dataset's cheapest candidate reaching its own target (a per-dataset pick, an upper "
              "bound only):", "", "| dataset | candidate | ALL | rows |", "| --- | --- | ---: | ---: |"]
    for d, e in v["beside_per_dataset_cheapest"].items():
        lines.append(f"| {d} | `{e['candidate']}` | {e['ALL']:.4f} | {e['rows']:,.0f} |")
    (OUT / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    ex = extras()
    assert len(ex) == 1 + 2 * (25 + 12) + 4 * 10 and len({e[0] for e in ex}) == len(ex)
    cands = candidates()
    assert len(cands) == len(FILL_F) * len(ex) == 805 and TODAY in {c[0] for c in cands} and len({c[0] for c in cands}) == len(cands)
    # D5b's toy graph: nodes 0..7; B = {0, 1}; links 0-2, 0-3, 1-3, 1-4, 3-5, 4-6; q = x
    st = D5.csr(8, [(0, 2), (0, 3), (1, 3), (1, 4), (3, 5), (4, 6)])
    adj = [st, D5.csr(8, []), D5.csr(8, [])]
    B = np.array([0, 1])
    E = D5B.unit(np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
                          np.float64))
    q = D5B.unit(np.array([1.0, 0, 0]))

    def read(rows):
        return E[np.asarray(rows, np.int64)]

    want = {"none", "beam1_cos_P10_m1", "beam2_cos_P10_m1", "beam1_cos_P10_m2", "beam2_cos_P10_m2", "cut_cos_50",
            "cut_w_50"}
    sets = extra_sets(adj, B, np.array([0.5, 1.0]), q, read, 60.0, want)
    assert sets["beam1_cos_P10_m1"].tolist() == [2, 3] and sets["beam2_cos_P10_m1"].tolist() == [2, 3, 5]  # as D5b
    assert sets["beam1_cos_P10_m2"].tolist() == [2, 3, 4] and sets["beam2_cos_P10_m2"].tolist() == [2, 3, 4, 5, 6]
    assert sets["cut_cos_50"].tolist() == [2, 3, 4] and sets["none"].size == 0
    # the same beam alone gives the same set (eval computes only the chosen one)
    one = extra_sets(adj, B, np.array([0.5, 1.0]), q, read, 60.0, {"beam2_cos_P10_m1"})
    assert one["beam2_cos_P10_m1"].tolist() == [2, 3, 5]
    # positions and the union's rows: KIND_T's order outside B = [5, 3, 7, 6, 4, 9]; today's fill its first 4
    full = np.array([5, 3, 7, 6, 4, 9])
    assert positions_in(np.array([3, 8, 6]), full).tolist() == [1, 6, 3]
    assert positions_in(np.array([3]), EMPTY).tolist() == [0]
    golds = np.array([0, 2, 6])
    cs = [("f1+none", 1.0, "none"), ("f0.5+none", 0.5, "none"), ("f0.25+beam1_cos_P10_m1", 0.25, "beam1_cos_P10_m1"),
          ("f0.5+beam2_cos_P10_m2", 0.5, "beam2_cos_P10_m2"), ("f1.5+none", 1.5, "none"),
          ("f1.5+beam1_cos_P10_m1", 1.5, "beam1_cos_P10_m1"), ("f4+none", 4.0, "none")]
    r, h = evaluate_question(B, full, 4, golds, sets, cs)
    # f1: B + 4 = 6 rows, golds 0 (B), 6 (fill 3); f0.5: B + {5, 3} = 4, gold 0 only
    # f0.25 + {2, 3}: B + {5} + {2, 3} = 5 rows, golds 0, 2; f0.5 + {2..6}: B + {5, 3} + {2, 4, 6} = 7, all three
    # f1.5: B + the order's first 6 = 8 rows, golds 0, 6; f1.5 + {2, 3}: 3 is in the 6, so + {2} = 9, all three
    # f4: round(16) is cut at the order's end (6), as f1.5
    assert r.tolist() == [6, 4, 5, 7, 8, 9, 8], r
    assert h.tolist() == [[True] * 7, [False, False, True, True, False, True, False],
                          [True, False, False, True, True, True, True]], h
    # summarize: two questions (golds 2 and 1), and one with none
    a = {"rows": np.array([[6, 4], [3, 3], [2, 2]], np.int32), "ptr": np.array([0, 2, 3, 3]),
         "held": np.array([[True, True], [True, False], [False, False]])}
    s = summarize(a, [("x", 1.0, "none"), ("y", 0.5, "none")])
    assert s["x"]["ALL"] == round(1 / 3, 4) and s["y"]["ALL"] == 0.0 and s["y"]["recall"] == round(0.5 / 3, 4)
    assert s["x"]["rows"] == round(11 / 3, 1)
    # choose: targets min(0.95, ceiling - TOL); the cheapest that meets all; NONE_MEETS by the worst gap
    w = {"a": 1, "b": 1}
    tab = {"t": {"a": {"ALL": 0.96, "rows": 100}, "b": {"ALL": 0.88, "rows": 100}},
           "u": {"a": {"ALL": 0.97, "rows": 300}, "b": {"ALL": 0.935, "rows": 300}},
           "v": {"a": {"ALL": 0.951, "rows": 200}, "b": {"ALL": 0.932, "rows": 200}},
           "x": {"a": {"ALL": 0.99, "rows": 900}, "b": {"ALL": 0.936, "rows": 900}}}
    v, pick, ceil, tgt, summ = choose(tab, w, names=("a", "b"))
    assert ceil == {"a": 0.99, "b": 0.936} and tgt == {"a": 0.95, "b": 0.931}, (ceil, tgt)
    assert (v, pick) == ("CEILING_MET", "v"), (v, pick)
    tab2 = {"t": tab["t"], "y": {"a": {"ALL": 0.96, "rows": 50}, "b": {"ALL": 0.97, "rows": 400}}}
    assert choose(tab2, w, names=("a", "b"))[:2] == ("TARGET_MET", "y")
    tab3 = {"p": {"a": {"ALL": 0.99, "rows": 10}, "b": {"ALL": 0.90, "rows": 10}},
            "q": {"a": {"ALL": 0.94, "rows": 10}, "b": {"ALL": 0.98, "rows": 10}}}
    v3, p3, _c, t3, s3 = choose(tab3, w, names=("a", "b"))
    assert t3 == {"a": 0.95, "b": 0.95} and v3 == "NONE_MEETS" and p3 == "q", (v3, p3, s3)   # gaps -0.05 vs -0.01
    log("p1g selftest: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("stats", "verdict", "eval", "report"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage in ("stats", "eval"):
        if a.host:                          # P1e's host mode: the frozen package on the mirror (as P1f's stats)
            import p1e_anchor_kind as K
            K.G.on_the_mirror()
        return (stats_stage if a.stage == "stats" else eval_stage)(a.dataset, a.limit, a.threads)
    if a.stage == "verdict":
        return verdict_stage()
    if a.stage == "report":
        return report_stage()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
