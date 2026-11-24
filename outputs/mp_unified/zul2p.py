"""UL2P (docs/UL2P_LABEL_PRECISION_E3V.md): the label precision ledger on the e3v layout. Gold-anchored and pool-free: an anchor is a slot
endpoint, a candidate is a unit's pair at that node, a target is the slot's other endpoint. Nothing trains and no model reads anything; no filed
graph, pool, fit or record changes. Labels (gold positions, MuSiQue's decompositions) are the grading only. zul2c, zul2u, zul2, zul1, zul2m and
zap4 are imported, never edited. Outputs only under outputs/ul2p/. Arrays (npz) are never written.

    python outputs/mp_unified/zul2p.py selftest
    python outputs/mp_unified/zul2p.py ledger --ds D [--host] [--threads 1]    # D's ledger (squad: its n/a record)
    python outputs/mp_unified/zul2p.py report                                  # the expectations E1-E7 and the tables
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zul2c as C  # noqa: E402  (imports zul2u -> zul2 -> zmhb -> zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import zap4 as AP  # noqa: E402

L2U, L2, Z, MB, UL = C.L2U, C.L2, C.Z, C.MB, C.UL
DECLARED = "docs/UL2P_LABEL_PRECISION_E3V.md"
OUT = ROOT / "outputs" / "ul2p"
DS7 = ("multihoprag", "musique", "squad", "metaqa", "webqsp", "hotpotqa", "2wiki")
FORMS = C.ORDER                                    # H5, HM, HK1, NTc, NT1, HK2, H10
LABEL = C.LABEL
DECIDE, BESIDE = C.DECIDE, C.BESIDE                # s1eval decides; s1sel is beside
TOP_REL = 12                                       # S relations listed (section 5); the rest pooled as S.other
BINS = 5                                           # rank quintiles of each form
MIN_T = 20                                         # a call (E1, E4, E5) needs this many targets
KEEP, MOST = 0.99, 12                              # exact sets: until 99% of the pairs at the anchors, at most 12
HOPS = (2, 3, 4)
BASE_BITS = ("R.mention", "R.name_comention", "R.title_comention", "K.knn")
EMPTY = np.zeros(0, np.int64)
log = MB.log


def stop(msg):
    raise SystemExit(f"[zul2p] STOP: {msg}")


def rel(p):
    return C.rel(p)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the engine: anchors, incident pairs, one unit's columns ──────────────────


class Anchors:
    """One cell's slots (question index qi, key sk, nq questions) and their anchor occurrences: one per (question, node), for each endpoint
    of each of the question's slots. qs: the questions with a slot, in order; every column below is over qs."""

    def __init__(self, n, qi, sk, nq, hop=None):
        qi, sk = np.asarray(qi, np.int64), np.asarray(sk, np.int64)
        self.n, self.nq, self.qi, self.sk = n, nq, qi, sk
        key = np.unique(np.concatenate([qi * n + sk // n, qi * n + sk % n]))
        self.oq, self.on = key // n, key % n
        self.nodes = np.unique(self.on)
        self.ocid = np.searchsorted(self.nodes, self.on)
        self.slots_q = np.bincount(qi, minlength=nq)
        self.anch_q = np.bincount(self.oq, minlength=nq)
        self.qs = np.flatnonzero(self.slots_q)
        self.hop = np.zeros(nq, np.int64) if hop is None else np.asarray(hop, np.int64)


def incident(keys, am, n, block=1 << 24):
    """The keys (order kept) with an endpoint in the boolean node mask am."""
    out = []
    for lo in range(0, keys.size, block):
        k = keys[lo:lo + block]
        out.append(k[am[k // n] | am[k % n]])
    return np.concatenate(out) if out else keys[:0]


def node_deg(an, keys):
    """Each anchor node's number of pairs among keys (sorted unique; either direction)."""
    deg = np.zeros(an.nodes.size)
    if keys.size == 0 or an.nodes.size == 0:
        return deg
    for x in (keys // an.n, keys % an.n):
        i = np.minimum(np.searchsorted(an.nodes, x), an.nodes.size - 1)
        deg += np.bincount(i[an.nodes[i] == x], minlength=an.nodes.size)
    return deg


def unit_cols(an, keys):
    """One unit's per-question sums over qs: candidates (the unit's pairs at the anchors), targets (two per covered slot), covered slots."""
    deg = node_deg(an, keys)
    cand = np.bincount(an.oq, weights=deg[an.ocid], minlength=an.nq)
    cov = np.bincount(an.qi[UL.member(an.sk, keys)], minlength=an.nq).astype(float)
    return cand[an.qs], 2 * cov[an.qs], cov[an.qs]


def paired(a, b):
    """The precision of unit a minus unit b's, paired over the same questions: {diff, call}."""
    d = AP.boot(lambda s: AP.ratio(s["ta"], s["ca"]) - AP.ratio(s["tb"], s["cb"]),
                {"ta": a[1], "ca": a[0], "tb": b[1], "cb": b[0]})
    return {"diff": d, "call": AP.call(d)}


def measure(an, cols, anycols=None, qm=None, core=False):
    """One unit's measures over the cell's questions (qm: a boolean over qs, a stratum): section 4's table."""
    cand, tgt, cov = (np.asarray(x, float) for x in cols)
    anc, sl = an.anch_q[an.qs].astype(float), an.slots_q[an.qs].astype(float)
    if qm is not None:
        cand, tgt, cov, anc, sl = cand[qm], tgt[qm], cov[qm], anc[qm], sl[qm]
    n = an.n
    rec = {"questions": int(cand.size), "anchors": int(anc.sum()), "slots": int(sl.sum()), "candidates": int(cand.sum()),
           "targets": int(tgt.sum()), "covered": int(cov.sum()),
           "price_per_anchor": round(float(AP.ratio(cand.sum(), anc.sum())), 4) if anc.sum() else None,
           "reach": AP.boot(lambda s: AP.ratio(s["cov"], s["sl"]), {"cov": cov, "sl": sl}),
           "precision": AP.boot(lambda s: AP.ratio(s["t"], s["c"]), {"t": tgt, "c": cand})}
    if not core:
        rec["base"] = float(AP.ratio(2 * sl.sum(), anc.sum() * (n - 1)))
        rec["lift"] = AP.boot(lambda s: AP.ratio(s["t"], s["c"]) / AP.ratio(2 * s["s"], s["a"] * (n - 1)),
                              {"t": tgt, "c": cand, "s": sl, "a": anc})
        rec["lift_call"] = AP.lift_call(rec["lift"])
    if anycols is not None:
        ac = np.asarray(anycols[0], float), np.asarray(anycols[1], float)
        if qm is not None:
            ac = ac[0][qm], ac[1][qm]
        rec["vs_any"] = paired((cand, tgt), ac)
    return rec


# ── the units ────────────────────────────────────────────────────────────────


def b0_sizes(F, v):
    """B0's size and its S > R > K parts (section 5's groups), counted pairwise (no union is built)."""
    S = F["S"] if v == "SRK" else EMPTY
    r = C.union_count([F["W2_S"], F["F0T"]])
    r_in_s = int((UL.member(S, F["W2_S"]) | UL.member(S, F["F0T"])).sum()) if S.size else 0
    k_new = int((~(UL.member(F["Ke"], S) | UL.member(F["Ke"], F["W2_S"]) | UL.member(F["Ke"], F["F0T"]))).sum())
    return {"S": int(S.size), "R": int(r - r_in_s), "K": k_new, "B0": int(S.size + r - r_in_s + k_new)}


def quintiles(keys_sorted, ranks, am, n):
    """A form's pairs (sorted keys) with their ranks (unique): the incident part of each of BINS equal-count rank parts, first = highest
    ranked, each sorted; and the parts' sizes."""
    cnt = keys_sorted.size
    if cnt == 0:
        return [EMPTY] * BINS, [0] * BINS
    rs = np.sort(ranks)
    cuts = rs[[cnt * j // BINS for j in range(1, BINS)]]
    bins = np.searchsorted(cuts, ranks, side="right")
    m = am[keys_sorted // n] | am[keys_sorted % n]
    ki, bi = keys_sorted[m], bins[m]
    return [ki[bi == b] for b in range(BINS)], np.bincount(bins, minlength=BINS).astype(int).tolist()


def base_units(F, slab, srk, am, n, sizes):
    """The label, group and any-link units of B0 on one version, over the pairs incident to the anchor mask am: {name: (pairs, sorted keys)},
    and the any-link pairs' label bits ((names, bit array over the any-link keys))."""
    inc = lambda k: incident(k, am, n)                       # noqa: E731
    W, CN, T0, K = inc(F["W2_S"]), inc(F["CN_all"]), inc(F["F0T"]), inc(F["Ke"])
    TI = np.setdiff1d(T0, CN, assume_unique=True)
    S = inc(F["S"]) if srk else EMPTY
    U = {}
    if srk:
        ks, rs, vocab = slab
        m = am[ks // n] | am[ks % n]
        ik, ir = ks[m], rs[m]
        cnt = np.bincount(ir, minlength=len(vocab))
        top = [int(x) for x in np.argsort(-cnt, kind="stable")[:TOP_REL] if cnt[x] > 0]
        full = np.bincount(rs, minlength=len(vocab))
        for x in top:
            U[f"label:S.{vocab[x]}"] = (int(full[x]), ik[ir == x])
        rest = ~np.isin(ir, top)
        if rest.any():
            U["label:S.other"] = (int(np.unique(ks[~np.isin(rs, top)]).size), np.unique(ik[rest]))
    U["label:R.mention"] = (int(F["W2_S"].size), W)
    U["label:R.name_comention"] = (int(F["CN_all"].size), CN)
    U["label:R.title_comention"] = (int(F["F0T"].size - UL.member(F["CN_all"], F["F0T"]).sum()), TI)
    U["label:K.knn"] = (int(F["Ke"].size), K)
    R_g = np.setdiff1d(np.union1d(W, T0), S, assume_unique=True)
    K_g = np.setdiff1d(K, np.union1d(S, np.union1d(W, T0)), assume_unique=True)
    if srk:
        U["group:S"] = (sizes["S"], S)
    U["group:R"] = (sizes["R"], R_g)
    U["group:K"] = (sizes["K"], K_g)
    any_k = np.union1d(np.union1d(S, R_g), K_g)
    U["any"] = (sizes["B0"], any_k)
    names = (["S"] if srk else []) + list(BASE_BITS)
    parts = ([S] if srk else []) + [W, CN, TI, K]
    bits = np.zeros(any_k.size, np.int64)
    for i, p in enumerate(parts):
        bits |= UL.member(any_k, p).astype(np.int64) << i
    return U, (names, bits)


def form_units(z, v, f, am, n):
    """Form f beyond B0 (in SRK, beyond S too): the whole form and its rank quintiles, as {name: (pairs, sorted keys)}."""
    r = z[f"rank__{f}"]
    sel = r >= 0
    if v == "SRK" and z["inS"].size:
        sel &= ~z["inS"]
    idx = np.flatnonzero(sel)
    keys = z["allF"][idx]
    parts, sizes = quintiles(keys, r[idx], am, n)
    out = {f"form:{f}": (int(idx.size), incident(keys, am, n))}
    for b in range(BINS):
        out[f"form:{f}:q{b + 1}"] = (sizes[b], parts[b])
    return out


def units_of(D, v, am):
    sizes = b0_sizes(D.F, v)
    U, bi = base_units(D.F, D.slab, v == "SRK", am, D.n, sizes)
    for f in FORMS:
        U.update(form_units(D.z, v, f, am, D.n))
    return U, bi, sizes


def exact_sets(an, any_k, bits, names, anyc):
    """The most frequent exact label sets of the any-link pairs at the anchors (until KEEP of them, at most MOST), each measured."""
    rows = []
    for b in np.unique(bits):
        cols = unit_cols(an, any_k[bits == b])
        rows.append((int(b), cols))
    tot = sum(c[0].sum() for _b, c in rows)
    if abs(tot - anyc[0].sum()) > 1e-6:
        stop(f"the exact sets hold {tot} candidates, the any-link unit {anyc[0].sum()}")
    rows.sort(key=lambda x: (-x[1][0].sum(), x[0]))
    out, seen = [], 0.0
    for b, cols in rows[:MOST]:
        rec = measure(an, cols, anyc)
        rec["labels"] = [names[i] for i in range(len(names)) if b >> i & 1]
        rec["share_of_candidates"] = round(float(AP.ratio(cols[0].sum(), tot)), 5)
        out.append(rec)
        seen += cols[0].sum()
        if tot and seen >= KEEP * tot:
            break
    return out


def cell_record(D, v, cv, kind, cell, U, bi):
    an = Anchors(D.n, cell["qi"], cell["sk"], cell["nq"], cell["hop"])
    cols = {k: unit_cols(an, u[1]) for k, u in U.items()}
    anyc = cols["any"]
    units = {}
    for k, u in U.items():
        r = measure(an, cols[k], None if k == "any" else anyc)
        r["pairs"] = int(u[0])
        r["graph_price"] = round(u[0] / r["covered"], 1) if r["covered"] else None
        units[k] = r
    names, bits = bi
    pop = np.array([bin(int(b)).count("1") for b in range(1 << len(names))])
    any_k = U["any"][1]
    multi, single = unit_cols(an, any_k[pop[bits] >= 2]), unit_cols(an, any_k[pop[bits] == 1])
    rec = {"questions": int(an.qs.size), "slots": int(cell["sk"].size), "units": units,
           "exact": exact_sets(an, any_k, bits, names, anyc), "multi_minus_single": paired(multi, single),
           "q1_vs_q5": {f: paired(cols[f"form:{f}:q1"], cols[f"form:{f}:q{BINS}"]) for f in FORMS}}
    if D.d == "musique":
        rec["hops"] = {}
        for h in HOPS:
            qm = an.hop[an.qs] == h
            if qm.any():
                rec["hops"][str(h)] = {"questions": int(qm.sum()),
                                       "units": {k: measure(an, cols[k], None if k == "any" else anyc, qm, core=True) for k in U}}
    return rec


# ── the data: one dataset's families, slots and the grade's ranks ─────────────


def grade_check(d, g, zp, bsha):
    if g.get("build_sha256") != bsha:
        stop(f"{d}: grade.json is not this build's (12.6's code changed since it was written)")
    if Z.sha_file(zp) != g["ranks_npz_sha256"]:
        stop(f"{d}: ranks.npz is not grade.json's")


def load_ranks(p):
    with np.load(p) as z:
        keep = [k for k in z.files if k in ("allF", "inS") or k.startswith("rank__") or k.endswith(("__qi", "__nq", "__B0"))]
        return {k: z[k] for k in keep}


def gates(D, g):
    """G-P1: B0's size per version = the grade's; G-P2: each form's pairs beyond B0 = the grade's; G-P3: each cell's slots, questions and
    B0 cover = the grade's ranks.npz (array equality)."""
    out = {"G-P1": {}, "G-P2": {}, "G-P3": {}}
    for v in D.vers:
        s = b0_sizes(D.F, v)
        if s["B0"] != int(g["sizes"]["B0"][v]):
            stop(f"G-P1: {D.d}/{v}: B0 {s['B0']:,}, the grade {g['sizes']['B0'][v]:,}")
        out["G-P1"][v] = s
    for f in FORMS:
        have = int((D.z[f"rank__{f}"] >= 0).sum())
        if have != int(g["pairs_beyond_B0"][f]):
            stop(f"G-P2: {D.d}/{f}: {have:,} pairs beyond B0, the grade {g['pairs_beyond_B0'][f]:,}")
        out["G-P2"][f] = have
    for (cv, kind), c in D.cells.items():
        pre = f"{cv}__{kind}__"
        gc = g["cells"][cv][kind]
        if not np.array_equal(D.z[pre + "qi"], c["qi"]) or int(D.z[pre + "nq"]) != c["nq"] or int(gc["slots"]) != c["sk"].size:
            stop(f"G-P3: {D.d}/{cv}/{kind}: the slots are not the grade's")
        if c.get("qsha") and gc.get("questions_npz_sha256") and c["qsha"] != gc["questions_npz_sha256"]:
            stop(f"G-P3: {D.d}/{cv}/{kind}: the questions' sha256 is not the grade's")
        for v in D.vers:
            cov = UL.member(c["sk"], D.F["W2_S"]) | UL.member(c["sk"], D.F["F0T"]) | UL.member(c["sk"], D.F["Ke"])
            if v == "SRK":
                cov |= UL.member(c["sk"], D.F["S"])
            if not np.array_equal(cov, D.z[f"{pre}{v}__B0"]):
                stop(f"G-P3: {D.d}/{cv}/{kind}/{v}: B0's cover of the slots is not the grade's")
            rec = UL.member(c["sk"], D.F["W2_S"]) | UL.member(c["sk"], D.F["F0T"]) | UL.member(c["sk"], D.F["K"])
            if v == "SRK":
                rec |= UL.member(c["sk"], D.F["S"])
            if int(rec.sum()) != int(gc[v]["linked"]["REC"]):
                stop(f"G-P3: {D.d}/{cv}/{kind}/{v}: REC covers {int(rec.sum())} slots, the grade {gc[v]['linked']['REC']}")
        out["G-P3"][f"{cv}/{kind}"] = {"slots": int(c["sk"].size), "questions": int(c["nq"])}
    return out


def load_host(d, host):
    a = types.SimpleNamespace(ds=d, host=host)
    served, freeze, n, nsha = L2.dataset_head(a)
    gp, zp = C.OUT / d / "grade.json", C.OUT / d / "ranks.npz"
    if not gp.is_file() or not zp.is_file():
        stop(f"{d}: 12.6's grade is not here ({rel(gp)}, ranks.npz): run ul2c-grade-{d} first")
    g = json.loads(gp.read_text(encoding="utf-8"))
    grade_check(d, g, zp, C.build_sha(C.G_FUNCS))
    if g["nodes"] != n or g["nodes_sha256"] != nsha or g["freeze_RECORD_SHA256"] != freeze:
        stop(f"{d}: the package is not the grade's (nodes, nodes sha256 or freeze)")
    F0, shas, slab = UL.families(served, d, n, nsha, freeze)
    Ke, krec = L2.knn_keys(served, d, n, nsha)
    if shas != g["inputs"] or krec != g["knn_e3v"]:
        stop(f"{d}: the families' or K_e3v's inputs are not the grade's")
    F = {"W2_S": F0["W2_S"], "CN_all": F0["CN_all"], "F0T": F0["F0T"], "Ke": Ke, "K": F0["K"], "S": F0.get("S", EMPTY)}
    cells = {}
    for cv in UL.CARVES[d]:
        ids, kinds, _h, qsha = L2.slot_kinds(served, d, n, cv)
        hq = np.array([AP.hop_of(d, x) for x in ids], np.int64)
        for kind, (qi, sk) in kinds.items():
            if sk.size:
                cells[(cv, kind)] = {"qi": qi, "sk": sk, "nq": len(ids), "hop": hq, "qsha": qsha}
    D = types.SimpleNamespace(d=d, n=n, nsha=nsha, vers=tuple(g["versions"]), F=F, slab=slab, cells=cells, z=load_ranks(zp), g=g,
                              src={"grade_json_sha256": Z.sha_file(gp), "ranks_npz_sha256": g["ranks_npz_sha256"]})
    return D


def load_mhr():
    import zul2m as M2
    root, nodes, n, nsha = M2.head()
    F0, gl = M2.families(n)
    lk = json.loads((M2.M.OUT / "links.json").read_text(encoding="utf-8"))
    p0 = M2.M.OUT / "links" / "rule" / "F0.npz"
    if M2.E.sha256(p0) != lk["files"]["F0"]:
        stop("G-L: links/rule/F0.npz is not links.json's")
    cn = np.load(p0)["keys"].astype(np.int64)
    if cn.size != int(lk["F0"]):
        stop("G-L: F0 is not links.json's count")
    gp, zp = M2.DOUT / "grade.json", M2.DOUT / "ranks.npz"
    g = json.loads(gp.read_text(encoding="utf-8"))
    grade_check("multihoprag", g, zp, M2.build_sha(M2.G_FUNCS))
    if g["nodes"] != n or g["nodes_sha256"] != nsha:
        stop("multihoprag: the package is not the grade's")
    golds, split, _types = M2.golds_splits(root, nodes)
    cells = {}
    for cv, sp in M2.CARVE_SPLIT.items():
        idx = np.flatnonzero(split == sp)
        qi, sk = UL.gp_slots([golds[j] for j in idx], n)
        cells[(cv, "GP")] = {"qi": qi, "sk": sk, "nq": int(idx.size), "hop": np.zeros(idx.size, np.int64), "qsha": None}
    F = {"W2_S": F0["W2_S"], "CN_all": cn, "F0T": F0["F0T"], "Ke": F0["K_e3v"], "K": F0["K_frozen"], "S": EMPTY}
    return types.SimpleNamespace(d="multihoprag", n=n, nsha=nsha, vers=("RK",), F=F, slab=None, cells=cells, z=load_ranks(zp), g=g,
                                 src={"grade_json_sha256": Z.sha_file(gp), "ranks_npz_sha256": g["ranks_npz_sha256"],
                                      "links_json_sha256": M2.E.sha256(M2.M.OUT / "links.json")})


def anchor_mask(D):
    am = np.zeros(D.n, bool)
    for c in D.cells.values():
        am[c["sk"] // D.n] = True
        am[c["sk"] % D.n] = True
    return am


# ── the ledger ───────────────────────────────────────────────────────────────


def squad_record():
    """SQuAD has one gold per question and so no slot: no precision (section 9, D-P1). Its label sizes are UL2's filed ones."""
    p = L2.OUT / "squad" / "grade.json"
    rec = {"dataset": "squad", "declared_in": DECLARED, "status": "n/a",
           "reason": "one gold per question: no gold pair, so the gold-anchored ledger has no cell (section 9, D-P1)"}
    if p.is_file():
        j = json.loads(p.read_text(encoding="utf-8"))
        rec["label_pairs_L2e"] = j["labels"]["RK"]["pairs"]
        rec["ul2_grade_sha256"] = Z.sha_file(p)
    return rec


def cmd_ledger(a):
    d, t0 = a.ds, time.time()
    if d == "squad":
        rec = squad_record()
    else:
        D = load_mhr() if d == "multihoprag" else load_host(d, a.host)
        gt = gates(D, D.g)
        log(f"{d}: gates pass; {len(D.cells)} cells, versions {D.vers} ({time.time() - t0:.0f} s)")
        am = anchor_mask(D)
        rec = {"dataset": d, "declared_in": DECLARED, "status": "read", "nodes": D.n, "nodes_sha256": D.nsha, "sources": D.src,
               "gates": gt, "anchor_nodes": int(am.sum()), "versions": {}}
        for v in D.vers:
            U, bi, sizes = units_of(D, v, am)
            log(f"{d}/{v}: {len(U)} units over {int(am.sum()):,} anchor nodes ({time.time() - t0:.0f} s)")
            cells = {}
            for (cv, kind), cell in D.cells.items():
                cells.setdefault(cv, {})[kind] = cell_record(D, v, cv, kind, cell, U, bi)
                log(f"{d}/{v}/{cv}/{kind}: ledger written into the record ({time.time() - t0:.0f} s)")
            rec["versions"][v] = {"b0": sizes, "cells": cells}
        rec["memory"] = C.peak_gb()
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zul2p_sha256=script_sha256(),
               **{k: v for k, v in C.codes().items() if k in ("zul2c_sha256", "zul2u_sha256", "zul2_sha256")})
    L2.wjson(OUT / d / "ledger.json", rec)
    log(f"{d}: ledger {rel(OUT / d / 'ledger.json')} ({rec['seconds']} s)")
    return 0


# ── the report: expectations E1-E7 (section 6) and the tables ────────────────


def load_ledgers():
    out = {}
    for d in DS7:
        p = OUT / d / "ledger.json"
        if p.is_file():
            out[d] = json.loads(p.read_text(encoding="utf-8"))
    return out


def cells_of(L, kinds=("GP", "CH"), carve=DECIDE):
    """(dataset, version, kind, cell record) for every read s1eval cell."""
    for d, led in L.items():
        if led.get("status") != "read":
            continue
        for v, vr in led["versions"].items():
            for kind, c in vr["cells"].get(carve, {}).items():
                if kind in kinds:
                    yield d, v, kind, c


def pt(x):
    return None if not x else x["pt"]


def verdict(held, reads, detail):
    return {"call": "NOT_READ" if not reads else ("HELD" if held == reads else "NOT_HELD"), "held": held, "reads": reads, "detail": detail}


def expectations(L):
    E = {}
    held = reads = 0
    det = []
    for d, v, kind, c in cells_of(L):
        bad = [k for k, r in c["units"].items() if k.startswith("label:") and r["targets"] >= MIN_T and r["lift_call"] != "POINTS"]
        if any(k.startswith("label:") and r["targets"] >= MIN_T for k, r in c["units"].items()):
            reads += 1
            held += not bad
            det += [f"{d}/{v}/{kind}: {k} {c['units'][k]['lift_call']}" for k in bad]
    E["E1"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, kind, c in cells_of(L, ("GP",)):
        if d in ("hotpotqa", "2wiki", "musique"):
            u = c["units"]
            t = pt(u["label:R.title_comention"]["precision"])
            others = [pt(u[k]["precision"]) for k in ("label:R.mention", "label:R.name_comention") if pt(u[k]["precision"]) is not None]
            reads += 1
            ok = t is not None and all(t > o for o in others)
            held += ok
            det.append(f"{d}/{v}: title {t}, others {others}: {'held' if ok else 'not'}")
    E["E2"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, kind, c in cells_of(L, ("GP",)):
        if d in ("metaqa", "webqsp"):
            u = c["units"]
            s = pt(u["group:S"]["precision"])
            others = [pt(u[k]["precision"]) for k in ("label:R.mention", "label:R.name_comention", "label:R.title_comention", "label:K.knn")
                      if k in u and pt(u[k]["precision"]) is not None]
            reads += 1
            ok = s is not None and all(s > o for o in others)
            held += ok
            det.append(f"{d}/{v}: S {s}, others {others}: {'held' if ok else 'not'}")
    E["E3"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, kind, c in cells_of(L):
        for f in FORMS:
            r = c["units"][f"form:{f}"]
            if r["targets"] >= MIN_T:
                reads += 1
                ok = r["vs_any"]["call"] == "BELOW"
                held += ok
                if not ok:
                    det.append(f"{d}/{v}/{kind}: {f} vs any {r['vs_any']['call']}")
    E["E4"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, kind, c in cells_of(L):
        for f in FORMS:
            if c["units"][f"form:{f}:q1"]["targets"] >= MIN_T and c["units"][f"form:{f}:q{BINS}"]["targets"] >= MIN_T:
                reads += 1
                call = c["q1_vs_q5"][f]["call"]
                held += call == "ABOVE"
                if call != "ABOVE":
                    det.append(f"{d}/{v}/{kind}: {f} first vs last quintile {call}")
    E["E5"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, kind, c in cells_of(L, ("GP",)):
        if d in ("multihoprag", "musique", "hotpotqa", "2wiki", "metaqa"):
            call = c["multi_minus_single"]["call"]
            want = call != "ABOVE" if d == "metaqa" else call == "ABOVE"
            reads += 1
            held += want
            det.append(f"{d}/{v}: multi minus single {call}: {'held' if want else 'not'}")
    E["E6"] = verdict(held, reads, det)
    held = reads = 0
    det = []
    for d, v, kind, c in cells_of(L, ("GP",)):
        if d == "musique" and "hops" in c:
            r = [pt(c["hops"][h]["units"]["any"]["reach"]) if h in c["hops"] else None for h in map(str, HOPS)]
            reads += 1
            ok = None not in r and r[0] > r[1] > r[2]
            held += ok
            det.append(f"musique/{v}: reach by hops {r}: {'held' if ok else 'not'}")
    E["E7"] = verdict(held, reads, det)
    return E


MIN_SHARE = 0.05                                   # section 6's flag: a label's share of the any-link candidates at the anchors


def follows(L):
    """Section 6's three named lists, from every read s1eval cell. They are names: they change no rule and decide nothing."""
    out = {"weights": [], "cuts": [], "flags": []}
    for d, v, kind, c in cells_of(L):
        u = c["units"]
        tag = f"{d}/{v}/{kind}"
        anyc = u["any"]["candidates"]
        for k, r in u.items():
            if not k.startswith("label:"):
                continue
            call = (r.get("vs_any") or {}).get("call")
            if r["lift_call"] == "POINTS" and call == "ABOVE":
                out["weights"].append(f"{tag}: {k}")
            if call == "BELOW" and anyc and r["candidates"] / anyc >= MIN_SHARE:
                out["flags"].append(f"{tag}: {k} ({r['candidates'] / anyc:.1%} of the any-link candidates)")
        for f in FORMS:
            if u[f"form:{f}"]["vs_any"]["call"] == "BELOW" and u[f"form:{f}:q1"]["vs_any"]["call"] == "ABOVE":
                out["cuts"].append(f"{tag}: {f}")
    return out


def ci(x, scale=100.0, nd=2):
    if not x:
        return "-"
    return f"{x['pt'] * scale:.{nd}f} [{x['ci'][0] * scale:.{nd}f}, {x['ci'][1] * scale:.{nd}f}]"


def table(c, keys):
    rows = ["| unit | pairs | price per anchor | precision % | reach % | graph price | lift call | vs any (points) |",
            "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        r = c["units"][k]
        va = r.get("vs_any")
        rows.append(f"| {k} | {r['pairs']:,} | {r['price_per_anchor']} | {ci(r['precision'])} | {ci(r['reach'])} | "
                    f"{r['graph_price'] if r['graph_price'] is not None else '-'} | {r['lift_call'] or '-'} | "
                    f"{(ci(va['diff']) + ' ' + va['call']) if va and va.get('diff') else '-'} |")
    return rows


def render(L, E, W):
    lines = ["# UL2P report: the label precision ledger on the e3v layout", "",
             f"Declared in {DECLARED}. Written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. Decides nothing.", "",
             "## Expectations", "", "| id | call | reads | detail |", "|---|---|---|---|"]
    for k, e in E.items():
        lines.append(f"| {k} | {e['call']} | {e['held']} of {e['reads']} | {'; '.join(e['detail'][:6])}{' ...' if len(e['detail']) > 6 else ''} |")
    lines += ["", "## What follows (section 6: names only; no rule changes)", ""]
    for key, title in (("weights", "Candidate label weights (a label that POINTS and is ABOVE any link)"),
                       ("cuts", "Candidate tighter cuts (a form BELOW any link whose first quintile is ABOVE it)"),
                       ("flags", f"Flags (a label BELOW any link holding at least {MIN_SHARE:.0%} of the any-link candidates)")):
        lines.append(f"- {title}: " + ("; ".join(W[key]) if W[key] else "none"))
    for d, led in L.items():
        lines += ["", f"## {d}", ""]
        if led.get("status") != "read":
            lines.append(f"n/a: {led.get('reason')}")
            continue
        for v, vr in led["versions"].items():
            for cv in (DECIDE, BESIDE):
                for kind, c in vr["cells"].get(cv, {}).items():
                    lines += ["", f"### {d} / {v} / {cv} / {kind}: {c['questions']:,} questions, {c['slots']:,} slots", ""]
                    keys = [k for k in c["units"] if not (k.startswith("form:") and ":q" in k)]
                    lines += table(c, keys)
                    lines += ["", f"Multi minus single: {ci(c['multi_minus_single']['diff'])} {c['multi_minus_single']['call']}.", ""]
                    lines += ["Forms, first and last rank quintile (precision %):", ""]
                    for f in FORMS:
                        q1, q5 = c["units"][f"form:{f}:q1"], c["units"][f"form:{f}:q{BINS}"]
                        lines.append(f"- {f}: first {ci(q1['precision'])} ({q1['targets']} targets), last {ci(q5['precision'])} "
                                     f"({q5['targets']} targets), {c['q1_vs_q5'][f]['call']}")
    return "\n".join(lines) + "\n"


def cmd_report(_a):
    L = load_ledgers()
    if not L:
        stop("no ledger is written")
    E = expectations(L)
    W = follows(L)
    rec = {"declared_in": DECLARED, "datasets": sorted(L), "expectations": E, "follows": W,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "zul2p_sha256": script_sha256()}
    OUT.mkdir(parents=True, exist_ok=True)
    L2.wjson(OUT / "report.json", rec)
    (OUT / "report.md").write_bytes(render(L, E, W).encode("utf-8"))
    log("report written: " + ", ".join(f"{k} {e['call']}" for k, e in E.items()))
    return 0


# ── the self-test: brute force on small random graphs, then MultiHop-RAG's real shapes ──


def brute(golds, keys, n):
    """Per question (those with a slot): candidates, targets, covered slots, by loops over sets."""
    ks = set(keys.tolist())
    nbr = {}
    for k in ks:
        nbr.setdefault(k // n, set()).add(k % n)
        nbr.setdefault(k % n, set()).add(k // n)
    out = []
    for g in golds:
        g = [int(x) for x in g]
        if len(g) < 2:
            continue
        cand = sum(len(nbr.get(a, ())) for a in g)
        tgt = sum(len(nbr.get(a, set()) & (set(g) - {a})) for a in g)
        out.append((cand, tgt, tgt // 2))
    return np.array(out, float).T


def selftest():
    rng = np.random.default_rng(7)
    n, nq = 50, 16
    golds = [np.sort(rng.choice(n, size=int(rng.integers(1, 6)), replace=False)) for _ in range(nq)]
    golds[4] = golds[5].copy()
    qi, sk = UL.gp_slots(golds, n)
    an = Anchors(n, qi, sk, nq)
    for dens in (0.02, 0.1, 0.5):
        a, b = rng.integers(0, n, 600), rng.integers(0, n, 600)
        m = a != b
        keys = np.unique(np.minimum(a, b)[m] * n + np.maximum(a, b)[m])
        keys = keys[rng.random(keys.size) < dens]
        got = np.array(unit_cols(an, keys))
        want = brute(golds, keys, n)
        assert np.array_equal(got, want), (dens, got, want)
        am = np.zeros(n, bool)
        am[an.nodes] = True
        assert np.array_equal(np.array(unit_cols(an, incident(keys, am, n, block=7))), want)     # incident keys change nothing
    assert np.array(unit_cols(an, EMPTY)).sum() == 0
    # groups, exact sets, quintiles partition the same any-link pairs
    F = {k: np.unique(rng.integers(0, n * n, 150)) for k in ("W2_S", "CN_all", "F0T", "Ke", "S")}
    for k in F:
        F[k] = F[k][(F[k] // n) < (F[k] % n)]
    F["F0T"] = np.union1d(F["F0T"], F["CN_all"])
    rs = rng.integers(0, 4, F["S"].size)
    slab = (F["S"], rs, ["r0", "r1", "r2", "r3"])
    am = np.ones(n, bool)
    for srk in (False, True):
        v = "SRK" if srk else "RK"
        sz = b0_sizes(F, v)
        U, (names, bits) = base_units(F, slab, srk, am, n, sz)
        anyk = U["any"][1]
        want_any = np.unique(np.concatenate([F["W2_S"], F["F0T"], F["Ke"]] + ([F["S"]] if srk else [])))
        assert np.array_equal(anyk, want_any) and sz["B0"] == want_any.size
        assert sz["S"] + sz["R"] + sz["K"] == sz["B0"]
        assert U["group:R"][1].size + U["group:K"][1].size + (U["group:S"][1].size if srk else 0) == anyk.size
        assert all(np.array_equal(np.sort(U[k][1]), U[k][1]) for k in U)
        assert (bits > 0).all() and bits.max() < 1 << len(names)
    r = rng.permutation(400)[:137].astype(np.int32)
    ks = np.sort(rng.choice(n * n, 137, replace=False))
    parts, sizes = quintiles(ks, r, am, n)
    assert sum(sizes) == 137 and max(sizes) - min(sizes) <= 1 and sum(p.size for p in parts) == 137
    assert np.array_equal(np.sort(np.concatenate(parts)), ks)
    o = np.argsort(r)
    first = ks[o[:sizes[0]]]
    assert np.array_equal(np.sort(parts[0]), np.sort(first))
    # the bootstrap and the calls
    x = (np.array([3.0, 0, 2, 5]), np.array([1.0, 0, 1, 2]))
    p = paired(x, x)
    assert p["call"] == "LEVEL" and p["diff"]["pt"] == 0.0
    ex = expectations({})
    assert all(e["call"] == "NOT_READ" for e in ex.values())
    assert follows({}) == {"weights": [], "cuts": [], "flags": []}
    # section 6's lists on a made-up ledger whose answers are known
    def unit(lift, va, cand=10, tgt=30):
        return {"lift_call": lift, "vs_any": {"call": va}, "candidates": cand, "targets": tgt}
    fu = {"any": {"candidates": 100}, "label:A": unit("POINTS", "ABOVE"), "label:B": unit("POINTS", "LEVEL"),
          "label:C": unit("NO_LIFT", "ABOVE"), "label:D": unit("POINTS", "BELOW", cand=5), "label:E": unit("POINTS", "BELOW", cand=4),
          "group:R": unit("POINTS", "ABOVE")}
    for f in FORMS:
        fu[f"form:{f}"] = unit("POINTS", "BELOW")
        fu[f"form:{f}:q1"] = unit("POINTS", "ABOVE" if f in ("H5", "NT1") else "LEVEL")
    fu["form:HM"] = unit("POINTS", "LEVEL")
    fu["form:HM:q1"] = unit("POINTS", "ABOVE")
    fake = {"zz": {"status": "read", "versions": {"RK": {"cells": {DECIDE: {"GP": {"units": fu}}}}}}}
    w = follows(fake)
    assert w["weights"] == ["zz/RK/GP: label:A"], w
    assert w["cuts"] == ["zz/RK/GP: H5", "zz/RK/GP: NT1"], w
    assert len(w["flags"]) == 1 and w["flags"][0].startswith("zz/RK/GP: label:D"), w     # 5% flags, 4% does not
    # real shapes: MultiHop-RAG (laptop): the gates, then the engine against set loops on its own graph
    D = load_mhr()
    gates(D, D.g)
    am = anchor_mask(D)
    U, bi, _sz = units_of(D, "RK", am)
    cell = D.cells[(DECIDE, "GP")]
    an = Anchors(D.n, cell["qi"], cell["sk"], cell["nq"])
    some = an.qs[:12]
    gold_sets = [np.unique(np.concatenate([cell["sk"][cell["qi"] == q] // D.n, cell["sk"][cell["qi"] == q] % D.n])) for q in some]
    for name, full in (("any", np.unique(np.concatenate([D.F["W2_S"], D.F["F0T"], D.F["Ke"]]))),
                       ("label:R.mention", D.F["W2_S"]), ("label:K.knn", D.F["Ke"])):
        got = np.array(unit_cols(an, U[name][1]))[:, :12]
        assert np.array_equal(got, brute(gold_sets, full, D.n)), name
    f = "HK1"
    kept = D.z["allF"][D.z[f"rank__{f}"] >= 0]
    got = np.array(unit_cols(an, U[f"form:{f}"][1]))[:, :12]
    assert np.array_equal(got, brute(gold_sets, kept, D.n))
    assert sum(U[f"form:{f}:q{b}"][0] for b in range(1, BINS + 1)) == U[f"form:{f}"][0] == kept.size
    log("selftest: the engine equals brute force (random graphs and MultiHop-RAG's real shapes); gates G-P1..G-P3 pass on MultiHop-RAG")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zul2p.py")
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
