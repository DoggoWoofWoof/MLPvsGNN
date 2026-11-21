"""UL2 section 12: universal rules (docs/UL2_LAYOUT_ON_E3V.md, section 12). One rule set on every dataset (W2_S, F0T, H_5 and Part B's
forms: the same code and constants, each reading the dataset's rule text: a passage's own text, a KB entity's ABL-U rule text) and one
decision over all datasets: a form is carried on all of them or on none.

    python outputs/mp_unified/zul2u.py selftest
    python outputs/mp_unified/zul2u.py kbtext --ds metaqa|webqsp [--host]                        # gates G-T, G-F; F0T on the rule text
    python outputs/mp_unified/zul2u.py hbuild --ds metaqa|webqsp --enc frozen|e3v [--host] [--threads 8]   # R.hub's table, rule text
    python outputs/mp_unified/zul2u.py nbuild --ds metaqa|webqsp [--host] [--threads 4]          # the written-name forms, rule text
    python outputs/mp_unified/zul2u.py grade --ds D [--host] [--threads 1]                       # bases, forms, slot masks on D
    python outputs/mp_unified/zul2u.py report                                                    # the one decision (section 12.3)

Graph level only: nothing trains, no model reads anything, no filed graph, pool, fit or record changes. Labels (gold positions, MuSiQue's
decompositions) are the grading only; no rule reads one. zul2, zul1, zmhb, zabl_u, zu2r, zap1 and u1b are imported, never edited; the
passage datasets' tables are zul2's (outputs/ul2/<d>/), read by zul2's build identity. Outputs only under outputs/ul2u/. Arrays (npz)
stay uncommitted.
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

import zul2 as L2  # noqa: E402  (imports zmhb, which imports zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import u1b  # noqa: E402
import zabl_u as AU  # noqa: E402
import zap1 as Z  # noqa: E402
import zmhb as MB  # noqa: E402
import zu2r as U  # noqa: E402
import zul1 as UL  # noqa: E402

DECLARED = "docs/UL2_LAYOUT_ON_E3V.md, section 12"
OUT = ROOT / "outputs" / "ul2u"
SIX, PASSAGE, KB = UL.SIX, UL.PASSAGE, UL.KB
VERSIONS = UL.DOMAINS
ORDER = ("H5", "HM", "HK1", "NTc", "NT1", "HK2", "H10")        # the chain's order: H_5 first, then section 7's
BIT = {f: 1 << i for i, f in enumerate(ORDER)}
NCODE = 1 << len(ORDER)
LABEL = {"H5": "R.hub", **L2.B_LABEL}
P_STAR = 4997.1                                                # section 7's P* (H_5e's MuSiQue s1eval CH price), filed in 61fca78
BOUND = L2.BOUND                                               # 1.30 x |REC|
JUDGED = {"musique": ("GP", "CH"), "hotpotqa": ("GP",), "2wiki": ("GP",), "metaqa": ("GP",), "webqsp": ("GP",), "squad": ()}
DECIDE, BESIDE = "s1eval", "s1sel"
BASES = ("U", "N")                                             # U: B0 (decides); N: U-name, the KBs' F0T as UL1 filed it (beside)
log = MB.log


def stop(msg):
    raise SystemExit(f"UL2u: {msg}")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def codes():
    return {"zul2u_sha256": script_sha256(), "zabl_u_sha256": u1b.sha_src(AU.__file__), **L2.codes()}


KT_FUNCS = ("rule_text", "rule_passes", "gate_gt", "f0t_of", "cmd_kbtext")
H_FUNCS = ("rule_text", "rule_passes", "load_f0tu", "names_u", "cmd_hbuild")
NT_FUNCS = ("rule_text", "rule_passes", "load_f0tu", "names_u", "cmd_nbuild")


def build_sha(funcs):
    """A build's identity: the source of the functions it runs here, zul2's build identities (its key pass, picks and written names) and
    the rule text's code (zabl_u) and UL1's."""
    h = hashlib.sha256()
    for f in funcs:
        h.update(inspect.getsource(globals()[f]).replace("\r\n", "\n").encode("utf-8"))
    h.update(json.dumps([L2.build_sha(L2.H_FUNCS), L2.build_sha(L2.NT_FUNCS), u1b.sha_src(AU.__file__), AU.SEP, UL.codes()],
                        sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def rel(p):
    try:
        return str(Path(p).relative_to(ROOT)).replace("\\", "/")
    except ValueError:                                         # a smoke run's copy outside the workspace
        return Path(p).name


# ── the KB rule text (ABL-U's, ABL section 8) and F0T on it ──────────────────


def rule_text(served, d):
    """Every node's rule text: its distinct KB neighbours' names (both directions, node-id order) joined by "; ", its own name left out;
    its own name when it has no neighbour (zabl_u.texts_of over zabl_u.kb_neighbours, as ABL-U's build reads them)."""
    titles, _x, n, nsha = u1b.read_nodes(served, d, 0, 0)
    src, dst, _r, _f = u1b.package_structural(served, d)
    kip, kcol = AU.kb_neighbours(n, src, dst)
    del src, dst, _r, _f
    texts = AU.texts_of(titles, kip, kcol)
    st = {"kb_neighbour_entries": int(kcol.size), "nodes_with_neighbours": int((np.diff(kip) > 0).sum()),
          "text_chars": int(sum(len(x) for x in texts))}
    return titles, texts, n, nsha, st


def rule_passes(titles, texts):
    """A callable giving a fresh pass of (title, text or None), as zu2r.corpus gives the KBs' name text."""
    return lambda: ((t, x if x else None) for t, x in zip(titles, texts))


def gate_gt(d, n, nsha, freeze):
    """G-T: the recipe identity ABL-U's build computes, and the nodes, equal ABL-U's filed record; ABL-U's W2_S is its record's."""
    b = json.loads((UL.ABLU / d / "build.json").read_text(encoding="utf-8"))
    recipe = hashlib.sha256((AU.SEP + "|" + json.dumps(AU.texts_of(["a", "b", "c"], *AU.kb_neighbours(
        3, [0, 1], [1, 2])))).encode()).hexdigest()
    tr = b.get("text_recipe", {})
    ok = (tr.get("sha256") == recipe and tr.get("sep") == AU.SEP and tr.get("own_name") is False and int(b["nodes"]) == n
          and b["nodes_sha256"] == nsha and b["freeze_record_sha256"] == freeze)
    if not ok:
        stop(f"gate G-T: {d}: the rule text's recipe or nodes are not ABL-U's filed build's")
    _p, w2 = UL.rule_part(d, n, nsha, freeze)
    return {"recipe_sha256": recipe, "abl_u_w2_s_npz_sha256": w2, "verdict": "PASS"}


def f0t_of(passes, n):
    """F0T by UL1's build, unedited: the co-mention pairs (df 2..25) of the names, and of the names with each node's own title runs."""
    src, grp, tsrc, tgrp, nv = UL.build_entries(passes, n)
    (fa, fb, _fw), (la, lb, _lw) = UL.forms_from_entries(src, grp, tsrc, tgrp, n)
    k = UL.pair_keys(np.concatenate([fa, la]), np.concatenate([fb, lb]), n)
    return k, {"keys": int(nv), "name_entries": int(src.size), "title_run_entries": int(tsrc.size), "f0_pairs": int(fa.size),
               "lt_pairs": int(la.size)}


def cmd_kbtext(a):
    d, t0 = a.ds, time.time()
    if d not in KB:
        stop("the rule text is ABL-U's on the two KBs; a passage's rule text is its own text")
    served, freeze, n, nsha = L2.dataset_head(a)
    gt = gate_gt(d, n, nsha, freeze)
    F, shas, _slab = UL.families(served, d, n, nsha, freeze)
    filed = F["F0T"]
    del F
    passes0, n0, nsha0, _cvt = U.corpus(served, d)
    if (n0, nsha0) != (n, nsha):
        stop(f"{d}: the name text is not DATASET.json's")
    mine0, st0 = f0t_of(passes0, n)
    b = json.loads((UL.OUT / d / "build.json").read_text(encoding="utf-8"))
    gf = {"keys": st0["keys"], "ul1_keys": int(b["keys"]), "f0t_pairs": int(mine0.size), "ul1_f0t_pairs": int(filed.size),
          "same_pairs": bool(np.array_equal(mine0, filed))}
    gf["verdict"] = "PASS" if gf["same_pairs"] and gf["keys"] == gf["ul1_keys"] else "FAIL"
    log(f"{d}: gate G-F {gf['verdict']} ({gf}) ({time.time() - t0:.0f} s)")
    del mine0, passes0
    if gf["verdict"] != "PASS":
        stop(f"gate G-F: {d}: UL1's build path on the name text does not reproduce UL1's filed F0T")
    titles, texts, n1, nsha1, tst = rule_text(served, d)
    if (n1, nsha1) != (n, nsha):
        stop(f"{d}: the rule text's nodes are not DATASET.json's")
    log(f"{d}: rule text {tst} ({time.time() - t0:.0f} s)")
    f0tu, st = f0t_of(rule_passes(titles, texts), n)
    sha = L2.save_npz(OUT / d / "f0t_u.npz", f0t=f0tu)
    rec = {"dataset": d, "declared_in": DECLARED, "freeze_RECORD_SHA256": freeze, "nodes": n, "nodes_sha256": nsha, "gate_GT": gt,
           "gate_GF": gf, "rule_text": tst, **st, "f0t_u_pairs": int(f0tu.size), "f0t_filed_pairs": int(filed.size),
           "inputs": shas, "npz_sha256": sha, "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_sha256": build_sha(KT_FUNCS), **codes()}
    L2.wjson(OUT / d / "kbtext.json", rec)
    log(f"{d}: F0T_u {f0tu.size:,} pairs ({st['keys']:,} keys) beside the filed F0T's {filed.size:,}; written {rel(OUT / d / 'kbtext.json')} "
        f"({rec['seconds']} s)")
    return 0


def load_f0tu(d, n):
    p, j = OUT / d / "f0t_u.npz", OUT / d / "kbtext.json"
    if not p.is_file() or not j.is_file():
        stop(f"{d}: F0T_u is not built (kbtext first)")
    r = json.loads(j.read_text(encoding="utf-8"))
    if r["build_sha256"] != build_sha(KT_FUNCS) or Z.sha_file(p) != r["npz_sha256"] or int(r["nodes"]) != n:
        stop(f"{d}: f0t_u.npz is not this code's build, or not its record's")
    with np.load(p) as z:
        k = z["f0t"]
    if k.size != int(r["f0t_u_pairs"]):
        stop(f"{d}: F0T_u holds {k.size} pairs, its record {r['f0t_u_pairs']}")
    return k, r


def names_u(served, d, n, nsha, kt):
    """R.hub's names on the rule text (zmhb.build_entries_keep: UL1's entries with the vocabulary), checked against kbtext's key count."""
    titles, texts, n2, nsha2, _st = rule_text(served, d)
    if (n2, nsha2) != (n, nsha):
        stop(f"{d}: the rule text's nodes are not DATASET.json's")
    passes = rule_passes(titles, texts)
    s, g, vocab, _usual = MB.build_entries_keep(passes, n)
    if len(vocab) != int(kt["keys"]):
        stop(f"gate G1: {d}: {len(vocab):,} keys on the rule text, kbtext's build {kt['keys']:,}")
    return MB.Names(s, g, len(vocab), n), vocab, passes


def kb_head(a):
    if a.ds not in KB:
        stop("the passage datasets' tables and written names are zul2's (outputs/ul2/<d>/)")
    served, freeze, n, nsha = L2.dataset_head(a)
    F, shas, _slab = UL.families(served, a.ds, n, nsha, freeze)
    F0Tu, kt = load_f0tu(a.ds, n)
    return served, freeze, n, nsha, F, shas, F0Tu, kt


# ── R.hub's table and the written names on the rule text (the KBs) ────────────


def cmd_hbuild(a):
    d, enc, t0 = a.ds, a.enc, time.time()
    served, freeze, n, nsha, F, shas, F0Tu, kt = kb_head(a)
    if enc == "frozen":
        K, krec = F["K"], {"file": "package graph/knn.npz", "npz_sha256": shas["knn_npz_sha256"]}
    else:
        K, krec = L2.knn_keys(served, d, n, nsha)
    N = L2.h_exclusion({"W2_S": F["W2_S"], "F0T": F0Tu}, K)
    del F, K
    Np, Nc = L2.both_csr(N, n)
    log(f"{d} {enc}: N {N.size:,} pairs ({time.time() - t0:.0f} s)")
    names, _vocab, _passes = names_u(served, d, n, nsha, kt)
    del _vocab, _passes
    gc = L2.gate_gc(names, F0Tu, n)
    log(f"{d}: {names.node_key.size:,} name entries on the rule text; gate G-C {gc['verdict']} ({gc['pairs_checked']:,} pairs of "
        f"{gc['keys']:,} skipped keys, {time.time() - t0:.0f} s)")
    V, vrec = L2.load_rows(served, enc, d, n)
    log(f"{d} {enc}: rows {V.shape} {V.dtype} ({time.time() - t0:.0f} s)")
    top, H, st = L2.key_pass(names, V, Np, Nc, N, n, threads=a.threads)
    tab = {"tab_b": top.b, "tab_s": top.s, "tmax": top.tmax}
    del top
    (oa, ov, ob, os_), fl = L2.overrides(names, V, Np, Nc, tab, [("RK", None, None)], n)
    tab.update(ov_a=oa, ov_v=ov, ov_b=ob, ov_s=os_, **H)
    log(f"{d} {enc}: pass {st['seconds_pass']} s; flagged rows {fl}; {oa.size:,} exact entries ({time.time() - t0:.0f} s)")
    rec = {"dataset": d, "enc": enc, "declared_in": DECLARED, "rule_text": "ABL-U (ABL section 8)", "freeze_RECORD_SHA256": freeze,
           "nodes": n, "nodes_sha256": nsha, "R": L2.R_DEPTH, "ks": list(L2.KS), "eps": L2.EPS, "dfmax": L2.DFMAX, "hub_lo": L2.HUB_LO,
           "exclusion_pairs": int(N.size), "knn": krec, "vectors": vrec, "inputs": shas, "f0t_u_npz_sha256": kt["npz_sha256"],
           "gate_GC": gc, "pass": st, "flagged_rows": fl, "variants": ["RK"],
           "sizes": {f"H{k}": int(L2.h_pairs(tab, n, k).size) for k in L2.KS}}
    rec["sizes"].update(HM10=int(L2.mutual_pairs(tab, n, 10).size), HK1=int(L2.hk_pairs(tab, n, 1).size),
                        HK2=int(L2.hk_pairs(tab, n, 2).size))
    rec["npz_sha256"] = L2.save_npz(OUT / d / f"h_{enc}.npz", **tab)
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), build_sha256=build_sha(H_FUNCS),
               **codes())
    L2.wjson(OUT / d / f"h_{enc}.json", rec)
    log(f"{d} {enc}: written {rel(OUT / d / ('h_' + enc + '.npz'))} ({rec['seconds']} s)")
    return 0


def cmd_nbuild(a):
    d, t0 = a.ds, time.time()
    served, freeze, n, nsha, F, shas, F0Tu, kt = kb_head(a)
    K, krec = L2.knn_keys(served, d, n, nsha)
    N = L2.h_exclusion({"W2_S": F["W2_S"], "F0T": F0Tu}, K)
    del F, K
    Np, Nc = L2.both_csr(N, n)
    del N
    names, vocab, passes = names_u(served, d, n, nsha, kt)
    wy, wk = L2.written_entries(passes, vocab, names, n)
    del vocab
    log(f"{d}: {wy.size:,} written entries on the rule text ({time.time() - t0:.0f} s)")
    V, vrec = L2.load_rows(served, "e3v", d, n)
    ntc, nt1, st = L2.nt_forms(names, wy, wk, V, Np, Nc, n)
    sha = L2.save_npz(OUT / d / "nt_e3v.npz", ntc=ntc, nt1=nt1)
    rec = {"dataset": d, "enc": "e3v", "declared_in": DECLARED, "rule_text": "ABL-U (ABL section 8)", "nodes": n, "nodes_sha256": nsha,
           "freeze_RECORD_SHA256": freeze, "win": L2.WIN, "cap": L2.HUB_LO, "dfmax": L2.DFMAX, "knn": krec, "vectors": vrec, "inputs": shas,
           "f0t_u_npz_sha256": kt["npz_sha256"], **st, "ntc_pairs": int(ntc.size), "nt1_pairs": int(nt1.size), "npz_sha256": sha,
           "seconds": round(time.time() - t0, 1), "build_sha256": build_sha(NT_FUNCS), **codes()}
    L2.wjson(OUT / d / "nt_e3v.json", rec)
    log(f"{d}: NTc {ntc.size:,} pairs, NT1 {nt1.size:,} pairs ({rec['seconds']} s)")
    return 0


def load_kb_table(d, enc, n):
    p, j = OUT / d / f"h_{enc}.npz", OUT / d / f"h_{enc}.json"
    if not p.is_file() or not j.is_file():
        stop(f"{d}: {p.name} is not built")
    r = json.loads(j.read_text(encoding="utf-8"))
    if r["build_sha256"] != build_sha(H_FUNCS) or Z.sha_file(p) != r["npz_sha256"] or int(r["nodes"]) != n:
        stop(f"{d}: {p.name} is not this code's build, or not its record's")
    with np.load(p) as z:
        tab = {x: z[x] for x in z.files}
    return tab, r


def kb_nt(d, n):
    p, j = OUT / d / "nt_e3v.npz", OUT / d / "nt_e3v.json"
    if not p.is_file() or not j.is_file():
        stop(f"{d}: nt_e3v.npz is not built (nbuild first)")
    r = json.loads(j.read_text(encoding="utf-8"))
    if r["build_sha256"] != build_sha(NT_FUNCS) or Z.sha_file(p) != r["npz_sha256"] or int(r["nodes"]) != n:
        stop(f"{d}: nt_e3v.npz is not this code's build, or not its record's")
    with np.load(p) as z:
        return z["ntc"], z["nt1"]


def tables(d, n):
    """The forms from e3v's table and written names (deciding) and frozen's H_5 (L2f, beside): zul2's on the passage datasets, this
    module's on the KBs."""
    if d in PASSAGE:
        tf, rf = L2.load_table(d, "frozen", n)
        if d in L2.GH_SETS and rf.get("gate_GH", {}).get("verdict") != "PASS":
            stop(f"gate G-H has not passed on {d}: no e3v number is read")
        te, re_ = L2.load_table(d, "e3v", n)
        forms = L2.part_b_forms(d, n, te)
        if not {"NTc", "NT1"} <= set(forms):
            stop(f"{d}: zul2's written-name forms (outputs/ul2/{d}/nt_e3v.npz) are not built")
    else:
        tf, rf = load_kb_table(d, "frozen", n)
        te, re_ = load_kb_table(d, "e3v", n)
        forms = {"H10": L2.h_pairs(te, n, 10), "HM": L2.mutual_pairs(te, n, 10), "HK1": L2.hk_pairs(te, n, 1), "HK2": L2.hk_pairs(te, n, 2)}
        forms["NTc"], forms["NT1"] = kb_nt(d, n)
    forms["H5"] = L2.h_pairs(te, n, L2.K_H)
    h5f = L2.h_pairs(tf, n, L2.K_H)
    brief = {e: {k: r[k] for k in ("npz_sha256", "sizes", "flagged_rows", "build_sha256") if k in r} for e, r in (("frozen", rf), ("e3v", re_))}
    return {f: forms[f] for f in ORDER}, h5f, brief


# ── masks: each pair beyond a base, and each slot, as the set of forms holding it ─


def form_codes(forms):
    """Every pair of any form, and its code (bit i: in ORDER[i])."""
    allF = L2.union_of([forms[f] for f in ORDER])
    code = np.zeros(allF.size, np.uint8)
    for i, f in enumerate(ORDER):
        code |= UL.member(allF, forms[f]).astype(np.uint8) << np.uint8(i)
    return allF, code


def hist_of(allF, codeF, base):
    """Count of the pairs beyond base by code: the union of base and any set C of forms holds |base| + the counts of codes meeting C."""
    return np.bincount(codeF[~UL.member(allF, base)], minlength=NCODE).astype(np.int64)


def slot_codes(sk, forms):
    code = np.zeros(sk.size, np.uint8)
    for i, f in enumerate(ORDER):
        code |= UL.member(sk, forms[f]).astype(np.uint8) << np.uint8(i)
    return code


def union_size(G, d, base, dom, carried):
    cm = sum(BIT[x] for x in carried)
    h = np.asarray(G[d]["hist"][base][dom], np.int64)
    c = np.arange(NCODE)
    return int(G[d]["sizes"][base][dom] + h[(c & cm) != 0].sum())


# ── grade ────────────────────────────────────────────────────────────────────


def cmd_grade(a):
    d, t0 = a.ds, time.time()
    served, freeze, n, nsha = L2.dataset_head(a)
    F, shas, _slab = UL.families(served, d, n, nsha, freeze)
    del F["N_sp"], F["CN_all"]
    Ke, krec = L2.knn_keys(served, d, n, nsha)
    kt = None
    if d in KB:
        F0Tu, kt = load_f0tu(d, n)
    else:
        F0Tu = F["F0T"]
    forms, h5f, hrec = tables(d, n)
    allF, codeF = form_codes(forms)
    log(f"{d}: forms " + ", ".join(f"{f} {forms[f].size:,}" for f in ORDER) + f"; {allF.size:,} pairs in any ({time.time() - t0:.0f} s)")
    cells, ul1 = {}, json.loads((UL.OUT / d / "grade.json").read_text(encoding="utf-8"))
    M = {}
    for cv in UL.CARVES[d]:
        ids, kinds, _hops, qsha = L2.slot_kinds(served, d, n, cv)
        for kind, (qi, sk) in kinds.items():
            if sk.size == 0:
                continue
            cells[(cv, kind)] = (qi, sk, len(ids), qsha)
            pre = f"{cv}__{kind}__"
            M[pre + "qi"], M[pre + "nq"], M[pre + "code"] = qi.astype(np.int64), np.asarray(len(ids), np.int64), slot_codes(sk, forms)
    form_pairs = {f: int(forms[f].size) for f in ORDER}
    for f in ORDER:
        del forms[f]
    sizes, hist = {"REC": {}, "U": {}, "N": {}, "L2f": {}}, {"U": {}, "N": {}}
    for dom in VERSIONS[d]:
        S = [F["S"]] if dom == "SRK" else []
        arms = (("REC", lambda: L2.union_of(S + [F["W2_S"], F["F0T"], F["K"]])),
                ("U", lambda: L2.union_of(S + [F["W2_S"], F0Tu, Ke])),
                ("N", lambda: L2.union_of(S + [F["W2_S"], F["F0T"], Ke])),
                ("L2f", lambda: L2.union_of(S + [F["W2_S"], F0Tu, h5f, F["K"]])))
        for arm, build in arms:                                       # one union held at a time (WebQSP's hold ~5e8 pairs)
            u = build()
            sizes[arm][dom] = int(u.size)
            if arm in hist:
                hist[arm][dom] = hist_of(allF, codeF, u).tolist()
            for (cv, kind), (qi, sk, nq, _q) in cells.items():
                M[f"{cv}__{kind}__{dom}__{arm}"] = UL.member(sk, u)
            del u
        log(f"{d}/{dom}: " + ", ".join(f"{arm} {sizes[arm][dom]:,}" for arm in sizes) + f" ({time.time() - t0:.0f} s)")
    del allF, codeF
    filed = {}
    zp = L2.OUT / d / "grade.json"
    if zp.is_file():                                                  # zul2's filed grade: REC (and, on a KB, U-name = its L2e) agree
        zg = json.loads(zp.read_text(encoding="utf-8"))
        for dom in VERSIONS[d]:
            if zg["sizes"]["REC"][dom] != sizes["REC"][dom] or (d in KB and zg["sizes"]["L2e"][dom] != sizes["N"][dom]):
                stop(f"{d}/{dom}: REC or U-name's base is not zul2's filed REC or L2e")
        filed = {"zul2_grade_sha256": Z.sha_file(zp), "REC": "same", **({"N_is_zul2_L2e": True} if d in KB else {})}
    res = {"dataset": d, "declared_in": DECLARED, "nodes": n, "nodes_sha256": nsha, "freeze_RECORD_SHA256": freeze, "kb": d in KB,
           "versions": list(VERSIONS[d]), "inputs": shas, "knn_e3v": krec, "tables": hrec, "kbtext": (
               {k: kt[k] for k in ("npz_sha256", "f0t_u_pairs", "f0t_filed_pairs", "keys", "gate_GF", "gate_GT")} if kt else None),
           "form_pairs": form_pairs, "h5_frozen_pairs": int(h5f.size), "sizes": sizes, "hist": hist, "against_zul2": filed,
           "ratio_to_REC": {arm: {dom: round(sizes[arm][dom] / sizes["REC"][dom], 4) for dom in VERSIONS[d]} for arm in ("U", "N", "L2f")},
           "cells": {}}
    for (cv, kind), (qi, sk, nq, qsha) in cells.items():
        pre = f"{cv}__{kind}__"
        code = M[pre + "code"]
        rc = {"questions": nq, "slots": int(sk.size), "questions_npz_sha256": qsha}
        for dom in VERSIONS[d]:
            cov = {arm: M[f"{pre}{dom}__{arm}"] for arm in ("REC", "U", "N", "L2f")}
            want = ul1["carves"][cv][dom][kind]["forms"]["F0T"]["linked"]
            if int(cov["REC"].sum()) != int(want):
                stop(f"gate G-U: {d}/{cv}/{dom}/{kind}: REC links {int(cov['REC'].sum())}, UL1 filed {want}")
            uh = cov["U"] | ((code & BIT["H5"]) != 0)
            alone = {}
            for f in ORDER:
                cf = cov["U"] | ((code & BIT[f]) != 0)
                new = int((cf & ~cov["U"]).sum())
                add = int(sum(h for c, h in enumerate(hist["U"][dom]) if c & BIT[f]))
                alone[f] = {"new": new, "pairs_added": add, "price": round(add / new, 1) if new else None,
                            "gain": L2.diff_row(cf, cov["U"], qi, nq)}
            rc[dom] = {"linked": {arm: int(c.sum()) for arm, c in cov.items()} | {"U+H5": int(uh.sum())},
                       "share": {arm: Z.pct(c.mean()) for arm, c in cov.items()} | {"U+H5": Z.pct(uh.mean())},
                       "contrasts": {"U-REC": L2.diff_row(cov["U"], cov["REC"], qi, nq), "U+H5-REC": L2.diff_row(uh, cov["REC"], qi, nq),
                                     "L2f-REC": L2.diff_row(cov["L2f"], cov["REC"], qi, nq),
                                     **({"U-N": L2.diff_row(cov["U"], cov["N"], qi, nq)} if d in KB else {})},
                       "each_form_over_U": alone}
        res["cells"].setdefault(cv, {})[kind] = rc
    res["masks_npz_sha256"] = L2.save_npz(OUT / d / "masks.npz", **M)
    res.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **codes())
    L2.wjson(OUT / d / "grade.json", res)
    log(f"{d}: graded in {res['seconds']} s; written {rel(OUT / d / 'grade.json')}")
    return 0


# ── the one decision (section 12.3) ──────────────────────────────────────────


def step(G, M, base, carve, carried, f):
    """Form f over the running base (base and the carried forms): each version's pairs added and union against REC, and each judged cell
    of the carve (new slots, price, scale-free price, gain)."""
    cm, b = sum(BIT[x] for x in carried), BIT[f]
    c = np.arange(NCODE)
    size, cells = {}, []
    for d in G:
        for dom in G[d]["versions"]:
            h = np.asarray(G[d]["hist"][base][dom], np.int64)
            cur = union_size(G, d, base, dom, carried)
            ad = int(h[((c & b) != 0) & ((c & cm) == 0)].sum())
            rec = G[d]["sizes"]["REC"][dom]
            size[f"{d}/{dom}"] = {"pairs_added": ad, "union": cur + ad, "ratio_to_REC": round((cur + ad) / rec, 4)}
            for kind in JUDGED[d]:
                pre = f"{carve}__{kind}__"
                if pre + "qi" not in M[d]:
                    continue
                qi, code, nq = M[d][pre + "qi"], M[d][pre + "code"], int(M[d][pre + "nq"])
                covc = M[d][f"{pre}{dom}__{base}"] | ((code & cm) != 0)
                cf = covc | ((code & b) != 0)
                new = int((cf & ~covc).sum())
                cells.append({"dataset": d, "version": dom, "kind": kind, "slots": int(code.size), "new": new, "pairs_added": ad,
                              "price": round(ad / new, 1) if new else None,
                              "scale_free": round((ad / rec) / (new / code.size), 3) if new else None, "gain": L2.diff_row(cf, covc, qi, nq)})
    return size, cells


def chain(G, M, base):
    """Section 12.3: the forms in ORDER over the running base; carried when (a) ABOVE in a judged s1eval cell and BELOW in none, (b) every
    cell's price at most P* where the form adds slots, (c) every dataset's and version's union at most BOUND x |REC|."""
    carried, steps = [], []
    for f in ORDER:
        size, cells = step(G, M, base, DECIDE, carried, f)
        above = any(x["gain"]["call"] == "ABOVE" for x in cells)
        below = any(x["gain"]["call"] == "BELOW" for x in cells)
        dear = [f"{x['dataset']}/{x['version']}/{x['kind']}" for x in cells if x["new"] and x["price"] > P_STAR]
        over = [k for k, v in size.items() if v["ratio_to_REC"] > BOUND]
        ok = above and not below and not dear and not over
        steps.append({"form": f, "label": LABEL[f], "over": list(carried), "carried": bool(ok),
                      "conditions": {"a_gain": bool(above and not below), "b_price": not dear, "c_size": not over},
                      "price_over_P_star": dear, "over_bound": over, "size": size, "cells": cells,
                      "beside_s1sel": step(G, M, base, BESIDE, carried, f)[1]})
        if ok:
            carried.append(f)
    return carried, steps


def verdict_of(G, base, carried):
    over = [{"dataset": d, "version": dom, "ratio": round(G[d]["sizes"][base][dom] / G[d]["sizes"]["REC"][dom], 4)}
            for d in G for dom in G[d]["versions"] if G[d]["sizes"][base][dom] > BOUND * G[d]["sizes"]["REC"][dom]]
    if over:
        return "BASE_OVER_BOUND", over
    return ("CARRIED" if carried else "NONE_CARRIED"), []


def load_all():
    G, M, bad = {}, {}, []
    for d in SIX:
        j, p = OUT / d / "grade.json", OUT / d / "masks.npz"
        if not j.is_file() or not p.is_file():
            bad.append(d)
            continue
        g = json.loads(j.read_text(encoding="utf-8"))
        if g["zul2u_sha256"] != script_sha256() or Z.sha_file(p) != g["masks_npz_sha256"]:
            bad.append(d)
            continue
        G[d] = g
        with np.load(p) as z:
            M[d] = {k: z[k] for k in z.files}
    return G, M, bad


def cmd_report(_a):
    G, M, bad = load_all()
    if bad:
        L2.wjson(OUT / "report.json", {"declared_in": DECLARED, "verdict": "REFUSED", "missing_or_other_code": bad, **codes()})
        log(f"REFUSED: no grade of this code for {bad}")
        return 1
    mg = json.loads((L2.OUT / "musique" / "grade.json").read_text(encoding="utf-8"))
    if mg["part_b_chain"]["P_star"] != P_STAR:
        stop(f"P* {P_STAR} is not section 7's filed {mg['part_b_chain']['P_star']}")
    out = {}
    for base in BASES:
        carried, steps = chain(G, M, base)
        v, over = verdict_of(G, base, carried)
        out[base] = {"verdict": v, "forms": carried if v == "CARRIED" else [], "base_over_bound": over, "steps": steps}
    rec = {"declared_in": DECLARED, "datasets": list(G), "pending": ["multihop-rag (joins when MHR1 files its package)"], "P_star": P_STAR,
           "bound": BOUND, "boot": L2.BOOT, "seed": L2.SEED, "order": list(ORDER), "decides": "U", "U": out["U"], "beside": {
               "U_name": out["N"], "musique_only_chain_as_filed": {"carried": mg["part_b_chain"]["carried"],
                                                                   "zul2_grade_sha256": Z.sha_file(L2.OUT / "musique" / "grade.json")}},
           "bases": {d: {"sizes": g["sizes"], "ratio_to_REC": g["ratio_to_REC"]} for d, g in G.items()},
           "grades_sha256": {d: Z.sha_file(OUT / d / "grade.json") for d in G}, **codes()}
    L2.wjson(OUT / "report.json", rec)
    (OUT / "report.md").write_text(render(rec, G), encoding="utf-8", newline="\n")
    log(f"universal: {out['U']['verdict']} {out['U']['forms']}; beside, U-name: {out['N']['verdict']} {out['N']['forms']}")
    return 0


def _c(c):
    return f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}"


def render(rec, G):
    u = rec["U"]
    L = [f"# UL2 universal rules: the call ({DECLARED})", "",
         f"**{u['verdict']}** {u['forms']} (P* = {rec['P_star']} pairs per slot; bound {rec['bound']} x |REC|; "
         f"datasets {', '.join(rec['datasets'])}; pending: {', '.join(rec['pending'])}).", "",
         f"Beside: U-name (the KBs' F0T as filed) {rec['beside']['U_name']['verdict']} {rec['beside']['U_name']['forms']}; "
         f"MuSiQue's own chain as filed {rec['beside']['musique_only_chain_as_filed']['carried']}.", "",
         "Graph level, no model. A slot is linked when its pair is in the union. Differences are paired over questions (ratio estimator), "
         f"95% interval of {rec['boot']} question resamples (seed {rec['seed']}), in points. s1eval decides.", ""]
    if u["base_over_bound"]:
        L += ["Base over the bound: " + "; ".join(f"{x['dataset']}/{x['version']} {x['ratio']}" for x in u["base_over_bound"]), ""]
    L += ["| dataset | version | REC | B0 (U) | U / REC | U-name / REC | L2f / REC |", "|---|---|---:|---:|---:|---:|---:|"]
    for d, b in rec["bases"].items():
        for dom in G[d]["versions"]:
            L.append(f"| {d} | {dom} | {b['sizes']['REC'][dom]:,} | {b['sizes']['U'][dom]:,} | {b['ratio_to_REC']['U'][dom]:.4f} | "
                     f"{b['ratio_to_REC']['N'][dom]:.4f} | {b['ratio_to_REC']['L2f'][dom]:.4f} |")
    L.append("")
    for st in u["steps"]:
        cd = st["conditions"]
        L += [f"## {st['form']} ({st['label']}) over B0 + {st['over'] or 'nothing'}: {'CARRIED' if st['carried'] else 'not carried'} "
              f"(a {cd['a_gain']}, b {cd['b_price']}, c {cd['c_size']})", ""]
        if st["price_over_P_star"]:
            L.append(f"- price over P* in: {', '.join(st['price_over_P_star'])}")
        if st["over_bound"]:
            L.append(f"- over the bound in: {', '.join(st['over_bound'])}")
        for x in st["cells"]:
            L.append(f"- {x['dataset']} {x['version']} {x['kind']} ({x['slots']:,} slots): +{x['new']} slots, {x['pairs_added']:,} pairs, "
                     f"price {x['price']}, scale-free {x['scale_free']}; {_c(x['gain'])}")
        L.append("- sizes: " + "; ".join(f"{k} +{v['pairs_added']:,} -> {v['ratio_to_REC']}" for k, v in st["size"].items()))
        L.append("")
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(0)
    # the rule text and F0T on it: f0t_of is every pair sharing a name of df 2..25, or a name or own title run of df 2..25; and the keys
    # R.hub's pass skips (df 2..25 over its entries) hold only those pairs (gate G-C on the rule text)
    words = ["Paris", "London", "Berlin Wall", "Rome", "Oslo", "Lima", "Cairo", "Delhi", "Tokyo", "Quito", "Accra", "Hanoi"]
    n = 120
    titles = [f"{words[i % len(words)]} {['North', 'South', 'East', 'West', 'Upper'][i % 5]}" if i % 3 else words[i % len(words)]
              for i in range(n)]
    titles[7] = ""                                                        # an empty name is left out of its neighbours' texts
    src, dst = rng.integers(0, n, 400), rng.integers(0, n, 400)
    kip, kcol = AU.kb_neighbours(n, src, dst)
    texts = AU.texts_of(titles, kip, kcol)
    passes = rule_passes(titles, texts)
    assert sum(1 for _t, x in passes() if x is None) == sum(1 for x in texts if not x)
    k, st = f0t_of(passes, n)
    s1, g1, t1, tg1, nv = UL.build_entries(passes, n)

    def brute(s_, g_):
        out = set()
        for key in set(g_.tolist()):
            P = sorted(set(s_[g_ == key].tolist()))
            if 2 <= len(P) <= 25:
                out |= {P[i] * n + P[j] for i in range(len(P)) for j in range(i + 1, len(P))}
        return out
    want = brute(s1, g1) | brute(np.r_[s1, t1], np.r_[g1, tg1])
    assert set(k.tolist()) == want and st["keys"] == nv and len(want) > 50, (len(want), k.size)
    s, g, vocab, _u = MB.build_entries_keep(passes, n)
    names = MB.Names(s, g, len(vocab), n)
    assert len(vocab) == nv and L2.gate_gc(names, k, n)["verdict"] == "PASS"
    # masks: the union of a base with any set of forms, and each slot's cover, from the codes
    n2 = 5000
    base = L2.ukeys(rng.integers(0, n2, 3000), rng.integers(0, n2, 3000), n2)
    forms = {f: L2.ukeys(rng.integers(0, n2, 400 + 100 * i), rng.integers(0, n2, 400 + 100 * i), n2) for i, f in enumerate(ORDER)}
    allF, codeF = form_codes(forms)
    h = hist_of(allF, codeF, base)
    sk = L2.ukeys(rng.integers(0, n2, 2000), rng.integers(0, n2, 2000), n2)
    sc = slot_codes(sk, forms)
    G0 = {"musique": {"hist": {"U": {"RK": h.tolist()}}, "sizes": {"U": {"RK": int(base.size)}}}}
    for _ in range(40):
        C = [f for f in ORDER if rng.random() < 0.4]
        u = L2.union_of([base] + [forms[f] for f in C])
        assert union_size(G0, "musique", "U", "RK", C) == u.size, C
        cm = sum(BIT[f] for f in C)
        assert np.array_equal(UL.member(sk, base) | ((sc & cm) != 0), UL.member(sk, u)), C
    # the chain, each dataset with its own forms: H5 cheap and ABOVE (carried); HM dear (b); HK1 dear on a KB only (b: all or none);
    # NTc over the bound on SQuAD only (c); NT1 cheap (carried, over H5); HK2 adds no slot (a); H10 = H5 (nothing new over it)
    N3, nq = 10_000_000, 200
    slot = lambda q, o: (2 * q + o) * N3 + (2 * q + o + 1)                 # noqa: E731
    junk = lambda lo, m: (np.arange(lo, lo + m, dtype=np.int64) + 5_000_000) * N3 + 9_999_999  # noqa: E731
    none = np.zeros(0, np.int64)
    qs = np.arange(nq, dtype=np.int64)
    sm, sq = slot(qs, 0), slot(qs[:100], 1_000)
    h5 = np.union1d(sm[40:50], junk(1_000_000, 40_000))
    spec = {"musique": {"base": np.union1d(sm[:40], junk(0, 400_000)), "sk": sm, "qi": qs, "nq": nq, "forms": {
                "H5": h5, "HM": np.union1d(sm[50:52], junk(1_100_000, 60_000)), "HK1": np.union1d(sm[60:80], junk(1_200_000, 20_000)),
                "NTc": np.union1d(sm[80:110], junk(1_400_000, 40_000)), "NT1": np.union1d(sm[110:125], junk(1_500_000, 5_000)),
                "HK2": junk(1_600_000, 100), "H10": h5}},
            "metaqa": {"base": np.union1d(sq[:10], junk(500_000, 200_000)), "sk": sq, "qi": qs[:100], "nq": 100, "forms": {
                "HK1": np.union1d(sq[10:11], junk(1_300_000, 10_000))}},
            "squad": {"base": junk(800_000, 100_000), "forms": {"NTc": junk(2_000_000, 40_000)}}}

    def pack(dd, sp):
        fm = {f: sp["forms"].get(f, none) for f in ORDER}
        allF_, codeF_ = form_codes(fm)
        dom = "SRK" if dd == "metaqa" else "RK"
        g_ = {"versions": [dom], "sizes": {"REC": {dom: int(sp["base"].size)}}, "hist": {}}
        m_ = {}
        kinds = JUDGED[dd] if "sk" in sp else ()
        for kind in kinds:
            pre = f"{DECIDE}__{kind}__"
            m_[pre + "qi"], m_[pre + "nq"], m_[pre + "code"] = sp["qi"], np.asarray(sp["nq"]), slot_codes(sp["sk"], fm)
        for b_ in BASES:
            g_["sizes"][b_] = {dom: int(sp["base"].size)}
            g_["hist"][b_] = {dom: hist_of(allF_, codeF_, sp["base"]).tolist()}
            for kind in kinds:
                m_[f"{DECIDE}__{kind}__{dom}__{b_}"] = UL.member(sp["sk"], sp["base"])
        return g_, m_
    G, M = {}, {}
    for dd, sp in spec.items():
        G[dd], M[dd] = pack(dd, sp)
    carried, steps = chain(G, M, "U")
    got = {st_["form"]: st_ for st_ in steps}
    assert got["HM"]["conditions"]["b_price"] is False and not got["HM"]["carried"]
    assert got["HK1"]["price_over_P_star"] == ["metaqa/SRK/GP"] and not got["HK1"]["carried"], got["HK1"]["price_over_P_star"]
    assert got["NTc"]["over_bound"] == ["squad/RK"] and not got["NTc"]["carried"], got["NTc"]["over_bound"]
    assert got["HK2"]["conditions"]["a_gain"] is False
    assert carried == ["H5", "NT1"], carried
    assert got["H10"]["cells"][0]["new"] == 0                             # H10 = H5 here: nothing new over the carried H5
    assert verdict_of(G, "U", carried) == ("CARRIED", [])
    G["squad"]["sizes"]["U"]["RK"] = int(1.31 * G["squad"]["sizes"]["REC"]["RK"]) + 1
    v, over = verdict_of(G, "U", carried)
    assert v == "BASE_OVER_BOUND" and over[0]["dataset"] == "squad"
    assert not chain(G, M, "U")[0]                                       # a base over the bound carries nothing anywhere
    assert build_sha(H_FUNCS) != build_sha(NT_FUNCS) != build_sha(KT_FUNCS)
    print("zul2u self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zul2u.py")
    ap.add_argument("cmd", choices=["selftest", "kbtext", "hbuild", "nbuild", "grade", "report"])
    ap.add_argument("--ds", choices=SIX)
    ap.add_argument("--enc", choices=L2.ENCS)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd in ("kbtext", "hbuild", "nbuild", "grade") and not a.ds:
        stop("--ds is needed")
    if a.cmd == "hbuild" and not a.enc:
        stop("--enc is needed")
    return {"selftest": lambda _a: selftest(), "kbtext": cmd_kbtext, "hbuild": cmd_hbuild, "nbuild": cmd_nbuild, "grade": cmd_grade,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
