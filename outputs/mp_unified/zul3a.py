"""UL3 part A (docs/UL3_LINK_TIGHTENING.md): which pairs of the B0 layout does the gold need? A one-hop SCREEN on the gold pairs, pool-free.
Gold-anchored like UL2P (an anchor is a slot endpoint, a candidate is a B0 pair at that node, a target is the slot's other endpoint); every
B0 pair at the anchors falls in exactly one CELL of a partition (its label profile, its S > R > K group, and/or the quintile of the
float32 dot of its two stored e3v rows); a keep-set is the cells of highest gold precision that hold RHO of the any-link graph's covered
slots on the DEV carve (s1sel; WebQSP: PL's dev, the non-holdout half of train); it is graded on s1eval. Nothing trains, no model reads anything, no filed graph,
pool, fit or record changes; the gold is the grading only. zul2p, zul2c, zul2u, zul2, zul1, zul2m and zap4 are imported, never edited.
Outputs only under outputs/ul3/. Arrays are never written.

    python outputs/mp_unified/zul3a.py selftest
    python outputs/mp_unified/zul3a.py sub --ds D [--host] [--threads 1]        # iteration run: a quarter of dev selects, the rest of dev grades, s1eval unread
    python outputs/mp_unified/zul3a.py sub-report                               # its diagnostic tables (no call)
    python outputs/mp_unified/zul3a.py cells --ds D [--host] [--threads 1]      # the confirmation: D's cells and keep-sets, dev selects, s1eval grades
    python outputs/mp_unified/zul3a.py report                                   # the screen's tables and calls
"""
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
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import zap4 as AP  # noqa: E402

C, L2, UL, MB = P.C, P.L2, P.UL, P.MB
DECLARED = "docs/UL3_LINK_TIGHTENING.md"
OUT = ROOT / "outputs" / "ul3"
DS6 = ("multihoprag", "musique", "metaqa", "webqsp", "hotpotqa", "2wiki")
BINS = 5                                            # dot quintiles inside a class (equal count on the selection half)
RHO = 0.95                                          # the coverage budget of the call: the PL target
RHOS = (0.99, 0.95, 0.90, 0.80)                     # the frontier, beside
R_MIN = 0.92                                        # the call: reach retained on the graded half, point value
TAU_MAX = 0.50                                      # the call: candidates retained on the graded half, point value
VARIANTS = ("P", "C", "PC")                         # P: label profile; C: group x dot quintile; PC: profile x dot quintile
BLOCK = 1 << 17
SEL_PAR, EVAL_PAR = 0, 1                            # the self-test's toy carves only: question index parity
SUB_M = 4                                           # the iteration subset: the dev questions whose index mod 4 is 0 (a quarter); the rest of dev checks it
log = MB.log


def stop(msg):
    raise SystemExit(f"[zul3a] STOP: {msg}")


def rel(p):
    return C.rel(p)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the engine: dots, classes, bins, cells ───────────────────────────────────


def pair_dot(rows, keys, n, block=BLOCK):
    """The float32 dot of the two stored rows of every pair (the filed forms' own score), blockwise."""
    out = np.empty(keys.size, np.float32)
    for lo in range(0, keys.size, block):
        k = keys[lo:lo + block]
        out[lo:lo + block] = (rows.take(k // n) * rows.take(k % n)).sum(1)
    return out


def classes(bits, names, variant):
    """The class of each any-link pair: P and PC the exact label profile (the pair's bits), C the S > R > K group (0 = S, 1 = R, 2 = K)."""
    if variant in ("P", "PC"):
        return bits.astype(np.int64)
    r_mask = sum(1 << i for i, nm in enumerate(names) if nm.startswith("R."))
    g = np.full(bits.size, 2, np.int64)
    g[(bits & r_mask) != 0] = 1
    if names[0] == "S":
        g[(bits & 1) != 0] = 0
    return g


def class_name(c, names, variant):
    if variant == "C":
        return ("S" if names[0] == "S" else None, "R", "K")[c] if (names[0] == "S" or c) else "S"
    return "+".join(names[i] for i in range(len(names)) if c >> i & 1)


def edges_for(cls, dot, sel_mask, variant):
    """Per class: the BINS-1 equal-count cut points of the dots of the class's pairs at the selection half's anchors (None when too few).
    P has no bins."""
    ed = {}
    if variant == "P":
        return ed
    for c in np.unique(cls).tolist():
        x = np.sort(dot[sel_mask & (cls == c)])
        ed[c] = x[[x.size * j // BINS for j in range(1, BINS)]] if x.size >= BINS else None
    return ed


def assign_bins(cls, dot, edges, variant):
    b = np.zeros(cls.size, np.int64)
    if variant == "P":
        return b
    for c, e in edges.items():
        if e is not None:
            m = cls == c
            b[m] = np.searchsorted(e, dot[m], side="right")
    return b


def cell_columns(an, any_k, cid):
    """{cell id: (candidates, targets, covered) per question over an.qs}: one pass per cell; a cell's keys stay sorted (stable order)."""
    order = np.argsort(cid, kind="stable")
    cs = cid[order]
    ids, start = np.unique(cs, return_index=True)
    end = np.append(start[1:], cs.size)
    return {int(i): P.unit_cols(an, any_k[order[a:b]]) for i, a, b in zip(ids.tolist(), start.tolist(), end.tolist())}


def s_cells(ids, names, variant):
    """The cell ids whose class is S (the pair is in the structural store): A.1 never removes them. Empty where there is no S (RK versions)."""
    if not names or names[0] != "S":
        return set()
    return {i for i in ids if (i // BINS == 0 if variant == "C" else bool((i // BINS) & 1))}


def keep_set(cols, rho, protect=()):
    """The cells, highest gold precision first (ties: fewer candidates, then the cell id), taken until their covered slots reach rho of all
    cells' covered slots; cells with no candidate are never kept. protect: cell ids kept first whatever their precision (A.1: the S cells), counted
    in the budget and the price. None when nothing is covered."""
    tot = float(sum(c[2].sum() for c in cols.values()))
    if tot <= 0:
        return None
    pro = sorted(i for i in protect if i in cols and cols[i][0].sum() > 0)
    rows = [(i, float(c[0].sum()), float(c[2].sum())) for i, c in cols.items() if c[0].sum() > 0 and i not in set(pro)]
    rows.sort(key=lambda r: (-(r[2] / r[1]), r[1], r[0]))
    kept, acc = list(pro), float(sum(cols[i][2].sum() for i in pro))
    for i, cand, cov in rows:
        if acc >= rho * tot:
            break
        kept.append(i)
        acc += cov
    return kept


def total(cols, ids):
    z = np.zeros_like(next(iter(cols.values()))[0]) if cols else np.zeros(0)
    cand, tgt, cov = z.copy(), z.copy(), z.copy()
    for i in ids:
        cand, tgt, cov = cand + cols[i][0], tgt + cols[i][1], cov + cols[i][2]
    return cand, tgt, cov


def grade(cols, kept):
    """The kept cells against all cells on one half: reach and price retained, precision of both, paired difference, and the call."""
    ka, aa = total(cols, kept), total(cols, list(cols))
    ratio = lambda a, b: AP.boot(lambda s: AP.ratio(s["a"], s["b"]), {"a": a, "b": b})        # noqa: E731
    R, T = ratio(ka[2], aa[2]), ratio(ka[0], aa[0])
    rec = {"questions": int(aa[0].size), "any": {"candidates": int(aa[0].sum()), "covered": int(aa[2].sum())},
           "kept": {"candidates": int(ka[0].sum()), "covered": int(ka[2].sum())}, "reach_retained": R, "price_retained": T,
           "precision_kept": AP.boot(lambda s: AP.ratio(s["t"], s["c"]), {"t": ka[1], "c": ka[0]}),
           "precision_any": AP.boot(lambda s: AP.ratio(s["t"], s["c"]), {"t": aa[1], "c": aa[0]}),
           "vs_any": P.paired((ka[0], ka[1]), (aa[0], aa[1]))}
    rec["call"] = "NOT_READ" if (R is None or T is None) else ("SCREEN_IN" if R["pt"] >= R_MIN and T["pt"] <= TAU_MAX else "SCREEN_OUT")
    return rec


def split_cell(cell, par):
    """The slots of the questions whose index has the parity par (nq, hop unchanged)."""
    m = (cell["qi"] % 2) == par
    return {**cell, "qi": cell["qi"][m], "sk": cell["sk"][m]}


def anchor_nodes(cells, n):
    am = np.zeros(n, bool)
    for c in cells:
        am[c["sk"] // n] = True
        am[c["sk"] % n] = True
    return am


def cells_record(n, any_k, bits, names, dot, sel, ev):
    """One (version, kind): per variant the classes, edges, per-cell columns on the DEV carve (sel), the kept sets at each RHO and their grade
    on s1eval (ev). sel and ev are slot cells (qi, sk, nq, hop)."""
    ams = anchor_nodes([sel], n)
    selmask = ams[any_k // n] | ams[any_k % n]
    an = {"sel": P.Anchors(n, sel["qi"], sel["sk"], sel["nq"], sel["hop"]), "eval": P.Anchors(n, ev["qi"], ev["sk"], ev["nq"], ev["hop"])}
    anyc = {k: P.unit_cols(a, any_k) for k, a in an.items()}
    out = {"any": {k: {"candidates": int(c[0].sum()), "targets": int(c[1].sum()), "covered": int(c[2].sum())} for k, c in anyc.items()}}
    for v in VARIANTS:
        cls = classes(bits, names, v)
        edges = edges_for(cls, dot, selmask, v)
        cid = cls * BINS + assign_bins(cls, dot, edges, v)
        cols = {k: cell_columns(a, any_k, cid) for k, a in an.items()}
        for k, a in an.items():                                                   # G3-1: the cells partition the any-link pairs
            cand = sum(c[0] for c in cols[k].values())
            cov = sum(c[2] for c in cols[k].values())
            if not (np.array_equal(cand, anyc[k][0]) and np.array_equal(cov, anyc[k][2])):
                stop(f"G3-1: {v}/{k}: the cells do not partition the any-link pairs")
        vr = {"cells": {}, "kept": {}, "kept_S": {}}
        prot = s_cells(set(cols["sel"]) | set(cols["eval"]), names, v)
        vr["s_cells"] = sorted(prot)
        for i in sorted(set(cols["sel"]) | set(cols["eval"])):                    # a cell with pairs only at the graded carve's anchors is listed too
            vr["cells"][str(i)] = {"class": class_name(i // BINS, names, v), "bin": int(i % BINS),
                                   **{k: {"candidates": int(cols[k][i][0].sum()), "covered": int(cols[k][i][2].sum())} for k in cols if i in cols[k]}}
        vr["edges"] = {str(c): (None if e is None else [float(x) for x in e]) for c, e in edges.items()}
        for rho in RHOS:
            for key, protect in (("kept_S", prot), ("kept", ())):                    # kept_S: A.1, the call; kept: S not protected, as first declared, beside
                kept = keep_set(cols["sel"], rho, protect)
                if kept is None:
                    vr[key][str(rho)] = {"status": "not read: nothing covered on the selection half"}
                    continue
                vr[key][str(rho)] = {"cells": kept, "sel": grade(cols["sel"], kept), "eval": grade(cols["eval"], [i for i in kept if i in cols["eval"]])}
        out[v] = vr
    return out


# ── the data and the command ─────────────────────────────────────────────────


def rows_of(d, D, host):
    if d == "multihoprag":
        import zul2m as M2
        root, _nodes, n, _nsha = M2.head()
        return C.Rows.of(root, "multihoprag", n)
    served, _freeze, n, _nsha = L2.dataset_head(types.SimpleNamespace(ds=d, host=host))
    return C.Rows.of(served, d, n)


def webqsp_dev(D):
    """WebQSP has no s1sel (it is read zero-shot): the selection carve is PL's dev, the non-holdout half of train (its pop_dev.npz golds, gold pairs
    as the KB datasets read them). G3-4: its question ids are disjoint from s1eval's."""
    p = ROOT / "outputs" / "pl2" / "webqsp" / "pop_dev.npz"
    if not p.is_file():
        stop(f"G3-4: {rel(p)} is not here")
    with np.load(p) as z:
        ids, flat, off, dg = [str(x) for x in z["ids"]], z["flat"].astype(np.int64), z["off"], str(z["digest"])
    golds = [np.unique(flat[off[i]:off[i + 1]]) for i in range(len(ids))]
    eval_ids, _g, _s = UL.carve_questions("webqsp", C.DECIDE)
    if set(ids) & set(eval_ids):
        stop("G3-4: PL's dev carve and s1eval share questions")
    qi, sk = UL.gp_slots(golds, D.n, UL.KB_GOLDS)
    return {"qi": qi, "sk": sk, "nq": len(ids), "hop": np.zeros(len(ids), np.int64), "qsha": dg}


def dev_cell(D, kind):
    """The selection carve's slot cell for one kind: s1sel's (WebQSP: PL's dev, joined to D.cells by webqsp_dev)."""
    return D.cells.get((C.BESIDE, kind))


def ledger_gate(d, v, cv, kind, anyrec):
    """G3-2: the any-link unit's candidates, targets and covered slots on the whole carve equal UL2P's filed ledger (when it is here)."""
    p = P.OUT / d / "ledger.json"
    if not p.is_file():
        return {"checked": False, "why": "no UL2P ledger here"}
    u = json.loads(p.read_text(encoding="utf-8"))["versions"][v]["cells"][cv][kind]["units"]["any"]
    for k in ("candidates", "targets", "covered"):
        if int(u[k]) != int(anyrec[k]):
            stop(f"G3-2: {d}/{v}/{cv}/{kind}: any-link {k} {anyrec[k]}, the filed ledger {u[k]}")
    return {"checked": True, "candidates": int(u["candidates"]), "covered": int(u["covered"])}


def dot_gate(rows, keys, n, dot, k=512):
    """G3-3: the blocked float32 dots equal a float64 dot of the same rows (a fixed sample of the pairs)."""
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(keys.size, size=min(k, keys.size), replace=False))
    a, b = rows.take(keys[idx] // n).astype(np.float64), rows.take(keys[idx] % n).astype(np.float64)
    want = (a * b).sum(1)
    err = float(np.abs(want - dot[idx]).max()) if idx.size else 0.0
    if err > 2e-5:
        stop(f"G3-3: the blocked dots differ from a float64 dot by {err}")
    return {"sample": int(idx.size), "max_abs_err": err, "max_abs_dot": float(np.abs(dot).max()) if dot.size else 0.0,
            "mean_norm": float(np.sqrt((a * a).sum(1)).mean()) if idx.size else None}


def sub_cell(cell, m, inside):
    """The iteration subset of a dev cell: the slots of the questions whose index mod m is 0 (inside) or is not 0 (the rest). nq unchanged."""
    k = (cell["qi"] % m == 0) if inside else (cell["qi"] % m != 0)
    return {**cell, "qi": cell["qi"][k], "sk": cell["sk"][k]}


def load_ds(a):
    d, t0 = a.ds, time.time()
    if d not in DS6:
        stop("--ds is one of the six slot datasets (SQuAD has no gold pair)")
    D = P.load_mhr() if d == "multihoprag" else P.load_host(d, a.host)
    gt = P.gates(D, D.g)
    if d == "webqsp":                                               # PL's dev carve joins the cells so that its anchors are read (G3-4 inside)
        D.cells[(C.BESIDE, "GP")] = webqsp_dev(D)
    rows = rows_of(d, D, a.host)
    if int(rows.starts[-1]) != D.n:
        stop(f"{d}: {int(rows.starts[-1])} vector rows for {D.n} nodes")
    log(f"{d}: gates G-P1..G-P3 pass; {len(D.cells)} cells; versions {D.vers} ({time.time() - t0:.0f} s)")
    return d, t0, D, gt, rows


def cmd_sub(a):
    """The iteration run (user 25 Nov ~23:30: small subsets first, the full datasets once the rule works): a quarter of the dev carve selects, the other
    three quarters grade, the anchors are the dev carve's only, and s1eval is never used. A diagnostic, no call is filed from it."""
    d, t0, D, gt, rows = load_ds(a)
    kinds = sorted({k for (cv, k) in D.cells if cv == C.BESIDE})
    devs = {k: D.cells[(C.BESIDE, k)] for k in kinds}
    if not devs:
        stop(f"{d}: no dev carve (s1sel) to select on")
    am = anchor_nodes(list(devs.values()), D.n)
    rec = {"dataset": d, "declared_in": DECLARED, "mode": "sub", "status": "diagnostic", "sub_m": SUB_M, "reads_s1eval": False, "nodes": D.n,
           "nodes_sha256": D.nsha, "sources": D.src, "gates": {"UL2P": gt}, "anchor_nodes": int(am.sum()),
           "constants": {"BINS": BINS, "RHO": RHO, "RHOS": list(RHOS), "variants": list(VARIANTS),
                         "selection": f"dev questions with index mod {SUB_M} = 0", "graded": f"the other dev questions (index mod {SUB_M} != 0)"},
           "versions": {}, "g3_2": {}}
    for v in D.vers:
        sizes = P.b0_sizes(D.F, v)
        U, (names, bits) = P.base_units(D.F, D.slab, v == "SRK", am, D.n, sizes)
        any_k = U["any"][1]
        del U
        dot = pair_dot(rows, any_k, D.n)
        rec.setdefault("g3_3", {})[v] = dot_gate(rows, any_k, D.n, dot)
        log(f"{d}/{v}: {any_k.size:,} any-link pairs at the dev anchors, dots done ({time.time() - t0:.0f} s)")
        rec["versions"][v] = {"b0": sizes, "kinds": {}}
        for kind, dev in devs.items():
            kr = cells_record(D.n, any_k, bits, names, dot, sub_cell(dev, SUB_M, True), sub_cell(dev, SUB_M, False))
            if d != "webqsp":
                whole = P.unit_cols(P.Anchors(D.n, dev["qi"], dev["sk"], dev["nq"], dev["hop"]), any_k)
                rec["g3_2"][f"{v}/{kind}/{C.BESIDE}"] = ledger_gate(d, v, C.BESIDE, kind, {"candidates": int(whole[0].sum()), "targets": int(whole[1].sum()),
                                                                                         "covered": int(whole[2].sum())})
            rec["versions"][v]["kinds"][kind] = kr
            log(f"{d}/{v}/{kind}: sub cells written into the record ({time.time() - t0:.0f} s)")
    rec["memory"] = C.peak_gb()
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zul3a_sha256=script_sha256(),
               zul2p_sha256=P.script_sha256())
    L2.wjson(OUT / d / "sub.json", rec)
    log(f"{d}: {rel(OUT / d / 'sub.json')} ({rec['seconds']} s)")
    return 0


def cmd_cells(a):
    d, t0, D, gt, rows = load_ds(a)
    am = P.anchor_mask(D)
    rec = {"dataset": d, "declared_in": DECLARED, "status": "read", "nodes": D.n, "nodes_sha256": D.nsha, "sources": D.src, "gates": {"UL2P": gt},
           "anchor_nodes": int(am.sum()), "constants": {"BINS": BINS, "RHO": RHO, "RHOS": list(RHOS), "R_MIN": R_MIN, "TAU_MAX": TAU_MAX,
                                                      "variants": list(VARIANTS), "selection": "dev carve: s1sel; WebQSP: PL's dev (outputs/pl2/webqsp/pop_dev.npz)",
                                                      "graded": "s1eval"},
           "versions": {}, "g3_2": {}}
    for v in D.vers:
        sizes = P.b0_sizes(D.F, v)
        U, (names, bits) = P.base_units(D.F, D.slab, v == "SRK", am, D.n, sizes)
        any_k = U["any"][1]
        del U
        dot = pair_dot(rows, any_k, D.n)
        rec.setdefault("g3_3", {})[v] = dot_gate(rows, any_k, D.n, dot)
        log(f"{d}/{v}: {any_k.size:,} any-link pairs at the anchors, dots done ({time.time() - t0:.0f} s)")
        kinds = sorted({k for (cv, k) in D.cells if cv == C.DECIDE})
        rec["versions"][v] = {"b0": sizes, "kinds": {}}
        for kind in kinds:
            ev_full = D.cells[(C.DECIDE, kind)]
            dev = dev_cell(D, kind)
            if dev is None:
                stop(f"{d}/{kind}: no dev carve (s1sel) to select on")
            kr = cells_record(D.n, any_k, bits, names, dot, dev, ev_full)
            for cv, cl in ((C.DECIDE, ev_full), (C.BESIDE, dev)):
                if cv == C.BESIDE and d == "webqsp":
                    continue                                              # PL's dev carve is not a UL2P carve: no filed ledger to equal
                whole = P.unit_cols(P.Anchors(D.n, cl["qi"], cl["sk"], cl["nq"], cl["hop"]), any_k)
                anyrec = {"candidates": int(whole[0].sum()), "targets": int(whole[1].sum()), "covered": int(whole[2].sum())}
                rec["g3_2"][f"{v}/{kind}/{cv}"] = ledger_gate(d, v, cv, kind, anyrec)
            rec["versions"][v]["kinds"][kind] = kr
            log(f"{d}/{v}/{kind}: cells written into the record ({time.time() - t0:.0f} s)")
    rec["memory"] = C.peak_gb()
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zul3a_sha256=script_sha256(),
               zul2p_sha256=P.script_sha256())
    L2.wjson(OUT / d / "cells.json", rec)
    log(f"{d}: {rel(OUT / d / 'cells.json')} ({rec['seconds']} s)")
    return 0


# ── the report ───────────────────────────────────────────────────────────────


def load_cells():
    out = {}
    for d in DS6:
        p = OUT / d / "cells.json"
        if p.is_file():
            out[d] = json.loads(p.read_text(encoding="utf-8"))
    return out


def judged(L):
    """(dataset, version, kind, kind record) for every judged cell."""
    for d, rec in L.items():
        for v, vr in rec["versions"].items():
            for kind, kr in vr["kinds"].items():
                yield d, v, kind, kr


def screen_calls(L):
    """Per variant: the call of each judged cell (at RHO) and the universal read: passes iff every judged cell is SCREEN_IN (and all six read)."""
    out = {}
    read = all(d in L for d in DS6)
    for v in VARIANTS:
        rec = {"all_six_read": read}
        for key, tag in (("kept_S", ""), ("kept", "_S_unprotected")):          # A.1 (S protected) is the call; S not protected is reported beside
            cells = {}
            for d, ver, kind, kr in judged(L):
                k = kr[v][key].get(str(RHO), {})
                cells[f"{d}/{ver}/{kind}"] = k.get("eval", {}).get("call", "NOT_READ")
            rec["cells" + tag] = cells
            rec["call" + tag] = "NOT_READ" if not read else ("PASSES_SCREEN" if cells and all(c == "SCREEN_IN" for c in cells.values()) else "FAILS_SCREEN")
        out[v] = rec
    return out


def pc(x, nd=1):
    return "n/a" if not x else f"{100 * x['pt']:.{nd}f} [{100 * x['ci'][0]:.{nd}f}, {100 * x['ci'][1]:.{nd}f}]"


def render(L, S):
    w = [f"# UL3 part A: the one-hop screen, report", "",
         f"Declared in `{DECLARED}`. Read: {', '.join(L) or 'none'}. Reach and price retained are shares of the any-link graph's covered slots and candidates "
         f"on s1eval, points with 95% intervals over questions, in percent. The keep-sets are chosen on the dev carve; the call fixes RHO = {RHO}, SCREEN_IN = reach retained >= {100 * R_MIN:.0f} and price retained <= {100 * TAU_MAX:.0f} (point values).", "",
         "## The universal read (a variant passes iff every judged cell is SCREEN_IN)"]
    w += ["", "The call is A.1's (the S cells are never removed; section 12 of the page); the same keep rule with S not protected, as first declared, is beside."]
    w.append("")
    w += ["| variant | call (S protected) | cells SCREEN_IN | cells | beside: S not protected |", "|---|---|---:|---:|---|"]
    for v in VARIANTS:
        s = S[v]
        w.append(f"| {v} | {s['call']} | {sum(c == 'SCREEN_IN' for c in s['cells'].values())} | {len(s['cells'])} | {s['call_S_unprotected']} |")
    for d, ver, kind, kr in judged(L):
        a_e, a_s = kr["any"]["eval"], kr["any"]["sel"]
        w += ["", f"## {d} {ver} {kind}", "",
              f"any-link candidates / covered slots: s1eval {a_e['candidates']:,} / {a_e['covered']:,}; dev {a_s['candidates']:,} / {a_s['covered']:,}.", ""]
        has_s = any(kr[v]["s_cells"] for v in VARIANTS)
        for key, title in (("kept_S", "S protected (A.1, the call)"), ("kept", "S not protected (as first declared), beside")):
            if key == "kept" and not has_s:
                w += ["No S cell in this version: the two rules are identical.", ""]
                continue
            w += [f"**{title}**", "",
                  "| variant | RHO | reach retained (dev) | reach retained (s1eval) | price retained (s1eval) | precision kept minus any (s1eval, pts) | call |",
                  "|---|---:|---|---|---|---|---|"]
            for v in VARIANTS:
                for rho in RHOS:
                    k = kr[v][key].get(str(rho), {})
                    e = k.get("eval")
                    if not e:
                        w.append(f"| {v} | {rho} | not read | | | | |")
                        continue
                    vs = e["vs_any"]["diff"]
                    dv = "n/a" if not vs else "%+.3f" % (100 * vs["pt"])
                    w.append("| %s | %s | %s | %s | %s | %s | %s |" % (v, rho, pc(k["sel"]["reach_retained"]), pc(e["reach_retained"]), pc(e["price_retained"]), dv,
                                                                  e["call"] if rho == RHO else "-"))
            w.append("")
        k = kr["PC"]["kept_S"].get(str(RHO))
        if k and "cells" in k:
            cells = kr["PC"]["cells"]
            w += ["PC's kept cells at RHO %s (S protected; dev; the S cells first, then in precision order): " % RHO +
                  "; ".join("%s q%d" % (cells[str(i)]["class"], cells[str(i)]["bin"] + 1) for i in k["cells"][:14]) + (" ..." if len(k["cells"]) > 14 else "")]
    return "\n".join(w) + "\n"


def cmd_report(_a):
    L = load_cells()
    S = screen_calls(L)
    rec = {"declared_in": DECLARED, "datasets": list(L), "screen": S, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "zul3a_sha256": script_sha256()}
    L2.wjson(OUT / "report.json", rec)
    (OUT / "report.md").write_text(render(L, S), encoding="utf-8", newline="\n")
    log(f"report written: {', '.join(f'{v} {S[v]['call']} (S unprotected {S[v]['call_S_unprotected']})' for v in VARIANTS)}")
    return 0


def load_subs():
    out = {}
    for d in DS6:
        p = OUT / d / "sub.json"
        if p.is_file():
            out[d] = json.loads(p.read_text(encoding="utf-8"))
    return out


def render_sub(L):
    w = ["# UL3 part A: the iteration subset, diagnostic report", "",
         f"Declared in `{DECLARED}`. The keep-sets are chosen on the dev questions with index mod {SUB_M} = 0 and graded on the other dev questions; "
         "s1eval is not read. No call is filed from this report: it shows what each rule keeps and which classes it loses, so that the next declared rule "
         "targets the failures. Shares are points with 95% intervals over questions, in percent."]
    for d, rec in L.items():
        for v, vr in rec["versions"].items():
            for kind, kr in vr["kinds"].items():
                a_e, a_s = kr["any"]["eval"], kr["any"]["sel"]
                w += ["", f"## {d} {v} {kind}", "", f"any-link candidates / covered slots: subset {a_s['candidates']:,} / {a_s['covered']:,}; rest of dev "
                      f"{a_e['candidates']:,} / {a_e['covered']:,}.", "",
                      "S protected (A.1; the S cells are never removed), reach and price retained of the rest of dev:", "",
                      "| variant | RHO | reach retained (subset) | reach retained (rest of dev) | price retained (rest of dev) |", "|---|---:|---|---|---|"]
                for var in VARIANTS:
                    for rho in RHOS:
                        k = kr[var]["kept_S"].get(str(rho), {})
                        e = k.get("eval")
                        if not e:
                            w.append(f"| {var} | {rho} | not read | | |")
                            continue
                        w.append("| %s | %s | %s | %s | %s |" % (var, rho, pc(k["sel"]["reach_retained"]), pc(e["reach_retained"]), pc(e["price_retained"])))
                k = kr["P"]["kept_S"].get(str(RHO))
                if k and "cells" in k:
                    kept = set(k["cells"])
                    lost = sorted(((c["eval"]["covered"], c["eval"]["candidates"], c["class"]) for i, c in kr["P"]["cells"].items()
                                   if int(i) not in kept and "eval" in c and c["eval"]["covered"] > 0), reverse=True)[:6]
                    w += ["", "P (S protected) at RHO %s, the classes it drops that hold the most covered slots on the rest of dev (covered / candidates): " % RHO +
                          ("; ".join("%s %d / %d" % (nm, cv, cd) for cv, cd, nm in lost) if lost else "none")]
    return "\n".join(w) + "\n"


def cmd_sub_report(_a):
    L = load_subs()
    rec = {"declared_in": DECLARED, "mode": "sub", "datasets": list(L), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "zul3a_sha256": script_sha256()}
    L2.wjson(OUT / "sub_report.json", rec)
    (OUT / "sub_report.md").write_text(render_sub(L), encoding="utf-8", newline="\n")
    log(f"sub report written: {', '.join(L) or 'nothing read'}")
    return 0


# ── the self-test: brute force on small random graphs, then MultiHop-RAG's real shapes ──


def naive_dot(V, keys, n):
    return np.array([float(np.float32(V[k // n] @ V[k % n])) for k in keys.tolist()], np.float32)


class ToyRows:
    def __init__(self, V):
        self.V = V.astype(np.float32)
        self.starts = np.array([0, V.shape[0]])
        self.dim = V.shape[1]

    def take(self, idx):
        return self.V[np.asarray(idx, np.int64)]


def selftest():
    rng = np.random.default_rng(11)
    n, nq = 60, 24
    V = rng.normal(size=(n, 8)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    rows = ToyRows(V)
    # blocked dots equal one-by-one dots
    keys = np.unique(np.minimum(*(r := rng.integers(0, n, (2, 500)))) * n + np.maximum(*r))
    keys = keys[(keys // n) != (keys % n)]
    d1 = pair_dot(rows, keys, n, block=13)
    assert np.allclose(d1, naive_dot(V, keys, n), atol=1e-6) and np.array_equal(d1, pair_dot(rows, keys, n, block=1 << 10))
    # the layout: random stores like zul2p's selftest, a real partition of the any-link pairs
    F = {k: np.unique(rng.integers(0, n * n, 220)) for k in ("W2_S", "CN_all", "F0T", "Ke", "S")}
    for k in F:
        F[k] = F[k][(F[k] // n) < (F[k] % n)]
    F["F0T"] = np.union1d(F["F0T"], F["CN_all"])
    rs = rng.integers(0, 4, F["S"].size)
    slab = (F["S"], rs, ["r0", "r1", "r2", "r3"])
    golds = [np.sort(rng.choice(n, size=int(rng.integers(2, 6)), replace=False)) for _ in range(nq)]
    qi, sk = UL.gp_slots(golds, n)
    cell = {"qi": qi, "sk": sk, "nq": nq, "hop": np.zeros(nq, np.int64), "qsha": None}
    sel, ev = split_cell(cell, SEL_PAR), split_cell(cell, EVAL_PAR)
    assert set((sel["qi"] % 2).tolist()) <= {0} and set((ev["qi"] % 2).tolist()) <= {1} and sel["qi"].size + ev["qi"].size == qi.size
    am = np.ones(n, bool)
    for srk in (False, True):
        v = "SRK" if srk else "RK"
        sz = P.b0_sizes(F, v)
        U, (names, bits) = P.base_units(F, slab, srk, am, n, sz)
        any_k = U["any"][1]
        dot = pair_dot(rows, any_k, n, block=17)
        for variant in VARIANTS:
            cls = classes(bits, names, variant)
            if variant == "C":
                grp = np.where(UL.member(any_k, F["S"]) if srk else False, 0,
                               np.where(UL.member(any_k, F["W2_S"]) | UL.member(any_k, F["F0T"]), 1, 2))
                assert np.array_equal(cls, grp), "C groups are S > R > K"
            ams = anchor_nodes([sel], n)
            selmask = ams[any_k // n] | ams[any_k % n]
            edges = edges_for(cls, dot, selmask, variant)
            b = assign_bins(cls, dot, edges, variant)
            if variant != "P":
                for c, e in edges.items():                         # bins: equal count on the selection pairs, ordered by the dot
                    m = selmask & (cls == c)
                    if e is None:
                        assert m.sum() < BINS and (b[cls == c] == 0).all()
                        continue
                    cnt = np.bincount(b[m], minlength=BINS)
                    assert cnt.sum() == m.sum() and cnt.max() - cnt.min() <= max(2, int(np.isin(dot[m], e).sum()) + 1)
                    mm = np.flatnonzero(cls == c)
                    assert (np.diff(b[mm][np.argsort(dot[mm], kind="stable")]) >= 0).all()
            else:
                assert (b == 0).all()
            cid = cls * BINS + b
            an = P.Anchors(n, ev["qi"], ev["sk"], nq, ev["hop"])
            cols = cell_columns(an, any_k, cid)
            tot = P.unit_cols(an, any_k)
            assert np.array_equal(sum(c[0] for c in cols.values()), tot[0]) and np.array_equal(sum(c[2] for c in cols.values()), tot[2])
            for i, c in cols.items():                             # a cell's columns equal the engine's on the cell's keys
                assert np.array_equal(np.array(c), np.array(P.unit_cols(an, any_k[cid == i])))
            prot = s_cells(set(cols), names, variant)                  # A.1: the S cells are exactly the cells whose pairs are in the S store, always kept
            assert (not srk) <= (not prot), "no S cell without S"
            for i in prot:
                m = cid == i
                assert UL.member(any_k[m], F["S"]).all(), "an S cell holds only S pairs"
            if prot:
                cc = keep_set(cols, 0.5, prot)
                assert cc is None or {i for i in prot if cols[i][0].sum() > 0} <= set(cc)
    # the keep rule on a made-up table whose answer is known
    z = lambda *x: (np.array([x[0]], float), np.array([2.0 * x[1]]), np.array([float(x[1])]))   # noqa: E731
    cols = {1: z(100, 5), 2: z(10, 4), 3: z(1000, 1), 4: z(5, 0), 5: z(0, 0), 6: z(20, 4)}
    assert keep_set(cols, 0.95) == [2, 6, 1, 3] or keep_set(cols, 0.95) == [2, 6, 1], keep_set(cols, 0.95)
    assert keep_set(cols, 0.50) == [2, 6], keep_set(cols, 0.50)             # precision 0.40, 0.20, 0.05, 0.001; 14 of 14 covered
    assert keep_set(cols, 1.0) == [2, 6, 1, 3]
    assert keep_set({1: z(7, 0)}, 0.95) is None
    cols_p = {1: z(100, 5), 2: z(10, 4), 3: z(1000, 1)}                      # A.1: the protected cell comes first whatever its precision and counts in the budget
    assert keep_set(cols_p, 0.5) == [2, 1] and keep_set(cols_p, 0.5, {3}) == [3, 2], (keep_set(cols_p, 0.5), keep_set(cols_p, 0.5, {3}))
    assert keep_set(cols_p, 0.5, {9}) == [2, 1], "a protected id that is not a cell is ignored"
    cols_t = {1: z(10, 3), 2: z(10, 3)}                                      # a tie: the lower id first
    assert keep_set(cols_t, 0.4) == [1]
    # a wrong rule is caught: keeping the LOWEST precision first reaches the budget at a higher price
    wrong = sorted(cols, key=lambda i: (cols[i][2][0] / max(cols[i][0][0], 1), i))
    acc, kept = 0.0, []
    for i in wrong:
        if acc >= 0.95 * 14:
            break
        kept.append(i)
        acc += cols[i][2][0]
    assert sum(cols[i][0][0] for i in kept) > sum(cols[i][0][0] for i in keep_set(cols, 0.95))
    g = grade({1: (np.array([4.0, 0.0]), np.array([2.0, 0.0]), np.array([1.0, 0.0])), 2: (np.array([6.0, 6.0]), np.array([0.0, 2.0]), np.array([0.0, 1.0]))}, [1])
    assert g["any"] == {"candidates": 16, "covered": 2} and g["kept"] == {"candidates": 4, "covered": 1}
    assert abs(g["reach_retained"]["pt"] - 0.5) < 1e-12 and abs(g["price_retained"]["pt"] - 0.25) < 1e-12 and g["call"] == "SCREEN_OUT"
    assert grade({1: (np.array([4.0]), np.array([4.0]), np.array([2.0]))}, [1])["call"] == "SCREEN_OUT"          # price retained 1.0
    # real shapes: MultiHop-RAG (laptop): the whole command, then G3-2 against UL2P's filed ledger
    D = P.load_mhr()
    P.gates(D, D.g)
    rows_m = rows_of("multihoprag", D, False)
    assert int(rows_m.starts[-1]) == D.n
    amm = P.anchor_mask(D)
    sizes = P.b0_sizes(D.F, "RK")
    U, (names, bits) = P.base_units(D.F, D.slab, False, amm, D.n, sizes)
    any_k = U["any"][1]
    dot = pair_dot(rows_m, any_k, D.n)
    dg = dot_gate(rows_m, any_k, D.n, dot)
    ev_full = D.cells[(C.DECIDE, "GP")]
    sel_full = D.cells[(C.BESIDE, "GP")]
    kr = cells_record(D.n, any_k, bits, names, dot, sel_full, ev_full)
    whole = P.unit_cols(P.Anchors(D.n, ev_full["qi"], ev_full["sk"], ev_full["nq"], ev_full["hop"]), any_k)
    wsel = P.unit_cols(P.Anchors(D.n, sel_full["qi"], sel_full["sk"], sel_full["nq"], sel_full["hop"]), any_k)
    gl = ledger_gate("multihoprag", "RK", C.DECIDE, "GP", {"candidates": int(whole[0].sum()), "targets": int(whole[1].sum()), "covered": int(whole[2].sum())})
    assert kr["any"]["eval"]["covered"] == int(whole[2].sum()) and kr["any"]["sel"]["covered"] == int(wsel[2].sum())
    for v in VARIANTS:                                                          # every cell is listed, the listed cells hold all covered slots of both carves
        cv = kr[v]["cells"].values()
        assert sum(c.get("eval", {}).get("covered", 0) for c in cv) == kr["any"]["eval"]["covered"]
        assert sum(c.get("sel", {}).get("covered", 0) for c in cv) == kr["any"]["sel"]["covered"]
    s_in, s_rest = sub_cell(sel_full, SUB_M, True), sub_cell(sel_full, SUB_M, False)       # the iteration subset: a partition of the dev slots, no s1eval
    assert s_in["qi"].size + s_rest["qi"].size == sel_full["qi"].size and (s_in["qi"] % SUB_M == 0).all() and (s_rest["qi"] % SUB_M != 0).all()
    ks = cells_record(D.n, any_k, bits, names, dot, s_in, s_rest)
    assert ks["any"]["sel"]["covered"] + ks["any"]["eval"]["covered"] == int(wsel[2].sum()), "subset + rest = the whole dev carve's covered slots"
    Lt = {"multihoprag": {"versions": {"RK": {"kinds": {"GP": kr}}}}}                   # the report code runs on the real record's shape (text only, nothing written)
    txt = render(Lt, screen_calls(Lt))
    assert "## multihoprag RK GP" in txt and "NOT_READ" in txt
    txt = render_sub({"multihoprag": {"versions": {"RK": {"kinds": {"GP": ks}}}}})
    assert "## multihoprag RK GP" in txt and "s1eval is not read" in txt
    for v in VARIANTS:
        assert kr[v]["s_cells"] == [] and all(kr[v]["kept_S"][str(r)]["cells"] == kr[v]["kept"][str(r)]["cells"] for r in RHOS), "no S in an RK version: A.1 = A"
        k = kr[v]["kept"][str(RHO)]
        assert "eval" in k and k["eval"]["any"]["covered"] == kr["any"]["eval"]["covered"]
        assert k["sel"]["reach_retained"]["pt"] >= RHO - 1e-9, "the dev carve holds RHO of its own covered slots"
    log(f"selftest: dots, classes, bins, cell partition and the keep rule equal brute force; MultiHop-RAG: G3-2 {gl}, G3-3 {dg}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zul3a.py")
    ap.add_argument("cmd", choices=["selftest", "cells", "report", "sub", "sub-report"])
    ap.add_argument("--ds", choices=DS6)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd in ("cells", "sub") and not a.ds:
        stop("--ds is needed")
    return {"selftest": lambda _a: selftest(), "cells": cmd_cells, "report": cmd_report, "sub": cmd_sub, "sub-report": cmd_sub_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
