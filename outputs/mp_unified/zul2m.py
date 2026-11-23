"""UL2 section 13 (docs/UL2_LAYOUT_ON_E3V.md): MultiHop-RAG joins section 12.6's chain, with section 12.1's forms built by
zul2's code on each article's own text, and the chain run over seven datasets. Nothing trains. No filed module is edited:
zmhr, zul2c, zul2u, zul2, zul1, zmhb, zap1 and zenc are imported; zenc is pointed at MHR1's package in this process only.

    python outputs/mp_unified/zul2m.py selftest
    python outputs/mp_unified/zul2m.py hbuild --enc e3v|frozen   laptop: R.hub's table on MultiHop-RAG
    python outputs/mp_unified/zul2m.py nbuild                    laptop: NTc and NT1 on e3v
    python outputs/mp_unified/zul2m.py ntrank                    laptop: NTc's groups, NT1's scores (gate G-N)
    python outputs/mp_unified/zul2m.py grade                     laptop: ranks beyond B0, the carves' GP slots, sizes
    python outputs/mp_unified/zul2m.py report7                   host: section 12.6.3's chain over the seven grades
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zmhr as M  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import zul2c as C  # noqa: E402

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

L2U, L2, Z, MB, UL = C.L2U, C.L2, C.Z, C.MB, C.UL
E = L2.E
DECLARED = "docs/UL2_LAYOUT_ON_E3V.md, section 13"
DS = M.DS
OUT = ROOT / "outputs" / "ul2m"
DOUT = OUT / DS
CARVE_SPLIT = {"s1eval": "test", "s1sel": "dev"}      # 13.2: MHR1's splits; train is not read
G1_KEYS = 26532                                        # 13.3 G1: MHR1's F0T build (links.json rule_build keys)
VERS = ("RK",)
ORDER = C.ORDER
log = MB.log


def stop(msg):
    raise SystemExit(f"[zul2m] STOP: {msg}")


def codes():
    return {"zul2m_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **C.codes()}


def build_sha(funcs):
    src = "".join(inspect.getsource(globals()[f]) for f in funcs)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


G_FUNCS = ("head", "families", "names_m", "golds_splits", "load_h", "load_nt", "cmd_grade")


def head():
    M.point_zenc()
    root = M.mhr_root()
    nodes = M.nodes_of(root)
    man = json.loads((root / DS / "DATASET.json").read_text(encoding="utf-8"))
    n = len(nodes)
    if n != int(man["nodes"]["n"]) or E.sha256(root / DS / "nodes.jsonl") != man["nodes"]["sha256"]:
        stop("nodes.jsonl is not DATASET.json's")
    return root, nodes, n, man["nodes"]["sha256"]


def families(n):
    """G-L: MHR1's link files by the sha256s in links.json; F0T, W2_S and each K by its count; REC and B0 by MHR1's RK sizes."""
    lk = json.loads((M.OUT / "links.json").read_text(encoding="utf-8"))
    rule = M.OUT / "links" / "rule"
    for f, sha in lk["files"].items():
        if E.sha256(rule / f"{f}.npz") != sha:
            stop(f"G-L: links/rule/{f}.npz is not links.json's")
    F = {"F0T": np.load(rule / "F0T.npz")["keys"].astype(np.int64), "W2_S": np.load(rule / "W2_S.npz")["keys"].astype(np.int64)}
    for arm in M.ARMS:
        p = M.OUT / "links" / arm / "K.npz"
        if E.sha256(p) != lk["arms"][arm]["files"]["K"]:
            stop(f"G-L: links/{arm}/K.npz is not links.json's")
        with np.load(p) as z:
            F[f"K_{arm}"] = L2.ukeys(z["src"], z["dst"], n)
        if F[f"K_{arm}"].size != int(lk["arms"][arm]["K_pairs"]):
            stop(f"G-L: K_{arm} holds {F[f'K_{arm}'].size} pairs, links.json {lk['arms'][arm]['K_pairs']}")
    if F["F0T"].size != int(lk["F0T"]) or F["W2_S"].size != int(lk["W2_S"]["pairs"]):
        stop("G-L: F0T or W2_S is not links.json's count")
    rec = C.union_count([F["F0T"], F["W2_S"], F["K_frozen"]])
    b0 = C.union_count([F["F0T"], F["W2_S"], F["K_e3v"]])
    if rec != int(lk["arms"]["frozen"]["RK"]) or b0 != int(lk["arms"]["e3v"]["RK"]):
        stop(f"G-L: REC {rec} or B0 {b0} is not MHR1's RK@frozen / RK@e3v")
    return F, {"links_json_sha256": E.sha256(M.OUT / "links.json"), "F0T": int(F["F0T"].size), "W2_S": int(F["W2_S"].size),
               "K_e3v": int(F["K_e3v"].size), "K_frozen": int(F["K_frozen"].size), "REC": rec, "B0": b0, "verdict": "PASS"}


def names_m(nodes, n):
    """Section 13.1: zmhb.build_entries_keep over (title, text) in node order; gate G1 against MHR1's F0T build's key count."""
    def passes():
        return ((x["title"], x["text"]) for x in nodes)
    s, g, vocab, _usual = MB.build_entries_keep(passes, n)
    if len(vocab) != G1_KEYS:
        stop(f"G1: {len(vocab):,} keys, MHR1's F0T build {G1_KEYS:,}")
    return MB.Names(s, g, len(vocab), n), vocab, passes


def exclusion(F, enc, n):
    N = L2.h_exclusion({"W2_S": F["W2_S"], "F0T": F["F0T"]}, F[f"K_{enc}"])
    Np, Nc = L2.both_csr(N, n)
    return N, Np, Nc


# ── builds (laptop) ──────────────────────────────────────────────────────────

def cmd_hbuild(a):
    enc, t0 = a.enc, time.time()
    root, nodes, n, nsha = head()
    F, gl = families(n)
    N, Np, Nc = exclusion(F, enc, n)
    names, _v, _p = names_m(nodes, n)
    gc = L2.gate_gc(names, F["F0T"], n)
    if gc["verdict"] != "PASS":
        stop(f"G-C: {gc}")
    V, vrec = L2.load_rows(root, enc, DS, n)
    top, H, st = L2.key_pass(names, V, Np, Nc, N, n, threads=a.threads)
    tab = {"tab_b": top.b, "tab_s": top.s, "tmax": top.tmax}
    del top
    (oa, ov, ob, os_), fl = L2.overrides(names, V, Np, Nc, tab, [("RK", None, None)], n)
    tab.update(ov_a=oa, ov_v=ov, ov_b=ob, ov_s=os_, **H)
    rec = {"dataset": DS, "enc": enc, "declared_in": DECLARED, "nodes": n, "nodes_sha256": nsha, "R": L2.R_DEPTH, "ks": list(L2.KS),
           "eps": L2.EPS, "dfmax": L2.DFMAX, "hub_lo": L2.HUB_LO, "exclusion_pairs": int(N.size), "gate_GL": gl,
           "gate_G1": {"keys": int(names.df.size), "want": G1_KEYS, "verdict": "PASS"}, "gate_GC": gc, "vectors": vrec, "pass": st,
           "flagged_rows": fl, "variants": ["RK"], "sizes": {f"H{k}": int(L2.h_pairs(tab, n, k).size) for k in L2.KS}}
    rec["sizes"].update(HM10=int(L2.mutual_pairs(tab, n, 10).size), HK1=int(L2.hk_pairs(tab, n, 1).size),
                        HK2=int(L2.hk_pairs(tab, n, 2).size))
    rec["npz_sha256"] = L2.save_npz(DOUT / f"h_{enc}.npz", **tab)
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               build_sha256=L2.build_sha(L2.H_FUNCS), **codes())
    L2.wjson(DOUT / f"h_{enc}.json", rec)
    log(f"{DS} {enc}: N {N.size:,}; G-C {gc['verdict']}; sizes {rec['sizes']} ({rec['seconds']} s)")
    return 0


def load_h(enc, n):
    p, j = DOUT / f"h_{enc}.npz", DOUT / f"h_{enc}.json"
    if not p.is_file() or not j.is_file():
        stop(f"h_{enc} is not built")
    r = json.loads(j.read_text(encoding="utf-8"))
    if r["build_sha256"] != L2.build_sha(L2.H_FUNCS) or Z.sha_file(p) != r["npz_sha256"] or int(r["nodes"]) != n:
        stop(f"h_{enc}.npz is not zul2's build, or not its record's")
    with np.load(p) as z:
        return {x: z[x] for x in z.files}, r


def cmd_nbuild(a):
    t0 = time.time()
    root, nodes, n, nsha = head()
    F, gl = families(n)
    N, Np, Nc = exclusion(F, "e3v", n)
    names, vocab, passes = names_m(nodes, n)
    wy, wk = L2.written_entries(passes, vocab, names, n)
    V, vrec = L2.load_rows(root, "e3v", DS, n)
    ntc, nt1, st = L2.nt_forms(names, wy, wk, V, Np, Nc, n)
    sha = L2.save_npz(DOUT / "nt_e3v.npz", ntc=ntc, nt1=nt1)
    rec = {"dataset": DS, "enc": "e3v", "declared_in": DECLARED, "nodes": n, "nodes_sha256": nsha, "win": L2.WIN, "cap": L2.HUB_LO,
           "dfmax": L2.DFMAX, "gate_GL": gl, "vectors": vrec, **st, "ntc_pairs": int(ntc.size), "nt1_pairs": int(nt1.size),
           "npz_sha256": sha, "seconds": round(time.time() - t0, 1), "build_sha256": L2.build_sha(L2.NT_FUNCS), **codes()}
    L2.wjson(DOUT / "nt_e3v.json", rec)
    log(f"{DS}: {wy.size:,} written entries; NTc {ntc.size:,}, NT1 {nt1.size:,} ({rec['seconds']} s)")
    return 0


def nt_built(n):
    p, j = DOUT / "nt_e3v.npz", DOUT / "nt_e3v.json"
    r = json.loads(j.read_text(encoding="utf-8"))
    if r["build_sha256"] != L2.build_sha(L2.NT_FUNCS) or Z.sha_file(p) != r["npz_sha256"] or int(r["nodes"]) != n:
        stop("nt_e3v.npz is not zul2's build, or not its record's")
    with np.load(p) as z:
        return z["ntc"], z["nt1"]


def cmd_ntrank(a):
    t0 = time.time()
    root, nodes, n, nsha = head()
    names, vocab, passes = names_m(nodes, n)
    ntc, nt1 = nt_built(n)
    wy, wk = L2.written_entries(passes, vocab, names, n)
    ck, cs, nkeys = C.nt_groups(names, wy, wk, n)
    gn = {"ntc_pairs": int(ntc.size), "ntc_recomputed": int(ck.size), "ntc_keys": nkeys, "ntc_same": bool(np.array_equal(np.sort(ck), ntc))}
    rows = C.Rows.of(root, DS, n)
    a1, b1 = nt1 // n, nt1 % n
    s1 = (rows.take(a1) * rows.take(b1)).sum(1).astype(np.float32) if nt1.size else np.zeros(0, np.float32)
    k1, s1r = C.ranked(a1, b1, s1, n)
    gn.update(nt1_pairs=int(nt1.size), nt1_scored=int(k1.size), nt1_same=bool(np.array_equal(np.sort(k1), nt1)))
    gn["verdict"] = "PASS" if gn["ntc_same"] and gn["nt1_same"] else "FAIL"
    if gn["verdict"] != "PASS":
        stop(f"G-N: {gn}")
    sha = L2.save_npz(DOUT / "ntrank.npz", ntc_keys=ck, ntc_score=cs, nt1_keys=k1, nt1_score=s1r)
    rec = {"dataset": DS, "declared_in": DECLARED, "nodes": n, "nodes_sha256": nsha, "text": "the article's own text", "gate_GN": gn,
           "npz_sha256": sha, "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **codes()}
    L2.wjson(DOUT / "ntrank.json", rec)
    log(f"{DS}: G-N PASS; NTc {ck.size:,} in {nkeys:,} groups, NT1 {k1.size:,} scored ({rec['seconds']} s)")
    return 0


def load_nt(n):
    p, j = DOUT / "ntrank.npz", DOUT / "ntrank.json"
    r = json.loads(j.read_text(encoding="utf-8"))
    if Z.sha_file(p) != r["npz_sha256"] or r["gate_GN"]["verdict"] != "PASS":
        stop("ntrank.npz is not its record's")
    ntc, nt1 = nt_built(n)
    with np.load(p) as z:
        out = {"NTc": (z["ntc_keys"], z["ntc_score"]), "NT1": (z["nt1_keys"], z["nt1_score"])}
    if not (np.array_equal(np.sort(out["NTc"][0]), ntc) and np.array_equal(np.sort(out["NT1"][0]), nt1)):
        stop("G-N: ntrank's lists are not the built NTc and NT1")
    return out, {"npz_sha256": r["npz_sha256"], "gate_GN": r["gate_GN"]}


# ── grade (laptop) ───────────────────────────────────────────────────────────

def golds_splits(root, nodes):
    qs = M.queries_of(root)
    pos = {x["node_id"]: i for i, x in enumerate(nodes)}
    golds = [np.asarray(sorted(pos[g] for g in q["gold_node_ids"]), np.int64) for q in qs]
    nby = {}
    for s in M.SPLITS:
        with open(root / DS / "queries" / f"{s}.jsonl", encoding="utf-8") as f:
            nby[s] = sum(1 for x in f if x.strip())
    split = np.asarray(sum(([s] * nby[s] for s in M.SPLITS), []))
    if split.size != len(qs):
        stop("the split counts do not cover the questions")
    return golds, split, np.asarray([q["question_type"] for q in qs])


def cmd_grade(a):
    t0 = time.time()
    root, nodes, n, nsha = head()
    F, gl = families(n)
    tab, hrec = load_h("e3v", n)
    forms, g_r = C.table_forms(tab, n)
    if any(v["verdict"] != "PASS" for v in g_r.values()):
        stop(f"G-R: {g_r}")
    nt, nrec = load_nt(n)
    forms.update(nt)
    tabf, hfrec = load_h("frozen", n)
    h5f = L2.h_pairs(tabf, n, L2.K_H)
    del tab, tabf

    def in_base(x):
        return UL.member(x, F["W2_S"]) | UL.member(x, F["F0T"]) | UL.member(x, F["K_e3v"])

    def in_rec(x):
        return UL.member(x, F["W2_S"]) | UL.member(x, F["F0T"]) | UL.member(x, F["K_frozen"])

    golds, split, types = golds_splits(root, nodes)
    cells = {}
    for cv, sp in CARVE_SPLIT.items():
        idx = np.flatnonzero(split == sp)
        qi, sk = UL.gp_slots([golds[j] for j in idx], n)
        cells[(cv, "GP")] = (qi, sk, int(idx.size))
    allF, A = C.assemble(forms, in_base, np.zeros(0, np.int64), VERS, cells, lambda sk, v: in_base(sk))
    sizes = {"REC": {"RK": gl["REC"]}, "B0": {"RK": gl["B0"]}}
    # G-U: over all questions, REC and B0 link MHR1's filed RK@frozen and RK@e3v GP counts
    mg = json.loads((M.OUT / "grade.json").read_text(encoding="utf-8"))
    qa, ska = UL.gp_slots(golds, n)
    gu = {"REC": int(in_rec(ska).sum()), "B0": int(in_base(ska).sum()),
          "want_REC": int(mg["GP"]["RK@frozen"]["linked"]), "want_B0": int(mg["GP"]["RK@e3v"]["linked"])}
    gu["verdict"] = "PASS" if (gu["REC"], gu["B0"]) == (gu["want_REC"], gu["want_B0"]) else "FAIL"
    if gu["verdict"] != "PASS":
        stop(f"G-U: {gu}")
    res = {"dataset": DS, "declared_in": DECLARED, "nodes": n, "nodes_sha256": nsha, "kb": False, "versions": list(VERS),
           "carves": CARVE_SPLIT, "gate_GL": gl, "gate_GR": g_r, "gate_GU": gu, "table": {k: hrec[k] for k in ("npz_sha256", "sizes",
           "flagged_rows", "build_sha256", "exclusion_pairs")}, "ntrank": nrec, "order": C.SCORE,
           "form_pairs": {f: int(forms[f][0].size) for f in ORDER},
           "pairs_beyond_B0": {"any": int(allF.size), **{f: int((A[f"rank__{f}"] >= 0).sum()) for f in ORDER}},
           "in_S": 0, "sizes": sizes, "ratio_to_REC": {"RK": round(gl["B0"] / gl["REC"], 4)},
           "room": {"RK": C.BOUND_NUM * gl["REC"] // C.BOUND_DEN - gl["B0"]}, "cells": {}, "beside": {}}
    for (cv, kind), (qi, sk, nq) in cells.items():
        rc_cov, b0 = in_rec(sk), A[f"{cv}__{kind}__RK__B0"]
        res["cells"].setdefault(cv, {})[kind] = {"questions": nq, "slots": int(sk.size), "split": CARVE_SPLIT[cv],
                                                 "RK": {"linked": {"REC": int(rc_cov.sum()), "B0": int(b0.sum())},
                                                        "B0-REC": L2.diff_row(b0, rc_cov, qi, nq)}}
    # beside: each form alone (uncut) over B0, and L2f (frozen H5 with frozen K) against REC, on s1eval, s1sel, all and each type
    sel = {"s1eval": split == "test", "s1sel": split == "dev", "all": np.ones(split.size, bool),
           **{t: types == t for t in sorted(set(types.tolist()))}}
    for s, m in sel.items():
        idx = np.flatnonzero(m)
        qi, sk = UL.gp_slots([golds[j] for j in idx], n)
        b0, rc = in_base(sk), in_rec(sk)
        row = {"questions": int(idx.size), "slots": int(sk.size), "B0": int(b0.sum()), "REC": int(rc.sum())}
        for f in ORDER:
            cf = b0 | UL.member(sk, np.sort(forms[f][0]))
            row[f"{f} alone over B0"] = {"new": int((cf & ~b0).sum()), **L2.diff_row(cf, b0, qi, int(idx.size))}
        l2f = rc | UL.member(sk, h5f)
        row["L2f - REC"] = {"new": int((l2f & ~rc).sum()), **L2.diff_row(l2f, rc, qi, int(idx.size))}
        res["beside"][s] = row
    res["beside"]["L2f"] = {"H5_frozen_pairs": int(h5f.size), "table": {k: hfrec[k] for k in ("npz_sha256", "sizes")}}
    res["ranks_npz_sha256"] = L2.save_npz(DOUT / "ranks.npz", allF=allF, **A)
    res.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), memory=C.peak_gb(),
               build_sha256=build_sha(G_FUNCS), **codes())
    L2.wjson(DOUT / "grade.json", res)
    log(f"{DS}: graded; {allF.size:,} pairs beyond B0; sizes {sizes}; room {res['room']} ({res['seconds']} s)")
    return 0


# ── report7 (host, after the six zul2c grades) ───────────────────────────────

def load_m():
    j, p = DOUT / "grade.json", DOUT / "ranks.npz"
    if not j.is_file() or not p.is_file():
        return None, None
    g = json.loads(j.read_text(encoding="utf-8"))
    if g.get("build_sha256") != build_sha(G_FUNCS) or Z.sha_file(p) != g["ranks_npz_sha256"]:
        return None, None
    with np.load(p) as z:
        return C.State(DS, g, {k: z[k] for k in z.files if k != "allF"}), g


def carried_of(D, carried):
    """zul2c.write_carried's arrays, written under outputs/ul2m/<d>/ (zul2c's own carried.npz files are not touched)."""
    out = {}
    for d, st in D.items():
        src = (DOUT if d == DS else C.OUT / d) / "ranks.npz"
        with np.load(src) as z:
            allF = z["allF"]
        idx = np.flatnonzero(st.inU)
        code = np.zeros(idx.size, np.uint8)
        for i, f in enumerate(carried):
            r = st.rank[f][idx]
            code |= ((r >= 0) & (r < st.cut[f])).astype(np.uint8) << np.uint8(i)
        out[d] = L2.save_npz(OUT / d / "carried.npz", keys=allF[idx], code=code, forms=np.asarray(carried, dtype="U8"))
    return out


def cmd_report7(_a):
    D, G, bad = C.load_all()
    sm, gm = load_m()
    if sm is None:
        bad.append(DS)
    if bad:
        L2.wjson(OUT / "report.json", {"declared_in": DECLARED, "verdict": "REFUSED", "missing_or_other_code": bad, **codes()})
        log(f"REFUSED: no grade of this code for {bad}")
        return 1
    D[DS], G[DS] = sm, gm
    C.JUDGED[DS] = ("GP",)                                     # section 13.2, this process only
    mp = L2.OUT / "musique" / "grade.json"
    mgr = json.loads(mp.read_text(encoding="utf-8"))
    if mgr["part_b_chain"]["P_star"] != C.P_STAR:
        stop(f"P* {C.P_STAR} is not section 7's filed {mgr['part_b_chain']['P_star']}")
    over = [{"dataset": d, "version": v, "ratio": round(st.base[v] / st.rec[v], 4)} for d, st in D.items() for v in st.vers
            if C.BOUND_DEN * st.base[v] > C.BOUND_NUM * st.rec[v]]
    carried, steps = C.chain(D)
    sha = carried_of(D, carried)
    layout = {d: {"forms": {f: next(s["datasets"][d]["cut"] for s in steps if s["form"] == f) for f in carried},
                  "union": dict(st.size), "ratio_to_REC": {v: round(st.size[v] / st.rec[v], 4) for v in st.vers},
                  "carried_npz_sha256": sha[d]} for d, st in D.items()}
    six = C.OUT / "report.json"
    rec = {"declared_in": DECLARED, "datasets": list(D), "pending": [], "P_star": C.P_STAR, "bound": f"{C.BOUND_NUM}/{C.BOUND_DEN}",
           "boot": L2.BOOT, "seed": L2.SEED, "order": list(ORDER), "scores": C.SCORE, "judged": {d: list(C.JUDGED[d]) for d in D},
           "verdict": "CARRIED" if carried else "NONE_CARRIED", "forms": carried, "base_over_bound": over, "layout": layout,
           "steps": steps, "beside": {"six_dataset_report_sha256": Z.sha_file(six) if six.is_file() else None,
                                      "musique_only_chain_as_filed": {"carried": mgr["part_b_chain"]["carried"],
                                                                      "zul2_grade_sha256": Z.sha_file(mp)},
                                      "multihoprag_alone": gm["beside"]},
           "bases": {d: {"sizes": g["sizes"], "ratio_to_REC": g["ratio_to_REC"]} for d, g in G.items()},
           "grades_sha256": {d: Z.sha_file((DOUT if d == DS else C.OUT / d) / "grade.json") for d in G}, **codes()}
    L2.wjson(OUT / "report.json", rec)
    (OUT / "report.md").write_text(C.render(rec).replace("section 12.6", "section 13 (seven datasets)", 1), encoding="utf-8",
                                   newline="\n")
    log(f"section 13: {rec['verdict']} {carried}; base over the bound: {over}")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────

def selftest():
    """The seven-dataset chain's plumbing on a toy MultiHop-RAG: zul2c.assemble's arrays read back through zul2c.State, JUDGED patched
    in this process, zul2c.chain over it; the carve map; build identities. Reads no data."""
    rng = np.random.default_rng(5)
    n = 40
    base = L2.ukeys(rng.integers(0, n, 60), rng.integers(0, n, 60), n)
    forms = {}
    for f in ORDER:
        k = L2.ukeys(rng.integers(0, n, 30), rng.integers(0, n, 30), n)
        forms[f] = C.ranked(k // n, k % n, rng.random(k.size).astype(np.float32), n)

    def in_base(x):
        return UL.member(x, base)
    golds = [np.sort(rng.choice(n, size=int(rng.integers(2, 5)), replace=False)).astype(np.int64) for _ in range(50)]
    cells = {}
    for cv, idx in (("s1eval", range(0, 30)), ("s1sel", range(30, 50))):
        qi, sk = UL.gp_slots([golds[j] for j in idx], n)
        cells[(cv, "GP")] = (qi, sk, len(idx))
    allF, A = C.assemble(forms, in_base, np.zeros(0, np.int64), VERS, cells, lambda sk, v: in_base(sk))
    rec = int(base.size) + 3
    g = {"versions": ["RK"], "sizes": {"REC": {"RK": rec}, "B0": {"RK": int(base.size)}},
         "form_pairs": {f: int(forms[f][0].size) for f in ORDER}, "cells": {"s1eval": {"GP": {}}, "s1sel": {"GP": {}}}}
    st = C.State(DS, g, A)
    had = C.JUDGED.get(DS)
    C.JUDGED[DS] = ("GP",)
    try:
        carried, steps = C.chain({DS: st})
    finally:
        if had is None:
            C.JUDGED.pop(DS, None)
    assert len(steps) == len(ORDER) and all(s["cells"] for s in steps), "a step read no MultiHop-RAG cell"
    assert C.BOUND_DEN * st.size["RK"] <= C.BOUND_NUM * rec, "(c) broken"
    for s in steps:
        cut = s["datasets"][DS]["cut"]
        assert cut["kept_new_pairs"] <= cut["new_pairs"]
    assert set(CARVE_SPLIT.values()) <= set(M.SPLITS) and "train" not in CARVE_SPLIT.values()
    for f in G_FUNCS:
        assert f in globals(), f
    log(f"selftest passed: toy chain over {len(steps)} forms (carried {carried}); union {st.size['RK']} within 1.30 x {rec}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("selftest", "hbuild", "nbuild", "ntrank", "grade", "report7"))
    p.add_argument("--enc", choices=("e3v", "frozen"))
    p.add_argument("--threads", type=int, default=E.THREADS)
    a = p.parse_args(argv)
    if a.cmd == "hbuild" and not a.enc:
        p.error("--enc is required")
    return {"selftest": lambda _a: selftest(), "hbuild": cmd_hbuild, "nbuild": cmd_nbuild, "ntrank": cmd_ntrank, "grade": cmd_grade,
            "report7": cmd_report7}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
