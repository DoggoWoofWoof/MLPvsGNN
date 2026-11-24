"""PL1 step S1 (docs/PL1_POOLS_ON_UL2_LAYOUT.md sections 2b and 3): the adopted UL2 layout as the walk stores' pair tables.

    python outputs/mp_unified/zpl1m.py build --ds squad --version RK [--host] [--threads 8]
    python outputs/mp_unified/zpl1m.py selftest            # toy arrays and brute-force sets
    python outputs/mp_unified/zpl1m.py selftest-real       # SQuAD's real base (UL1's F0T, W2_S, the package's K as the K stand-in), laptop

One (dataset, version) per call. The structural family holds S where the version has S, else W2_S; the ner family holds the rest
of R (F0T, then W2_S's other pairs where S holds the slot, then the carried forms); the knn family holds K_e3v (section 2b).
Dedup (D83's, S > R > K): where S holds the slot the ner store loses S's pairs; inside R nothing is removed; the knn store loses
every pair the other two hold. The walk reads a row's ORDER, not a weight, so each ner pair carries an order key (float32,
larger first, ties by node id): F0T by its 1/df weight (D83's), then W2_S's other pairs (key 0), then the carried pairs by the
first carried form that keeps them (UL2's chain order) and the rank in that form's list. Writes outputs/pl1/<d>/<v>/layout.npz
and layout.json. Imports zul2c, zul2, zul1, d83stores; edits none of them. Reads no question, no gold, no vector.

The base (F0T's pairs and weights, W2_S, S, K_e3v) is checked against UL2's filed grade (zul2c/<d>/grade.json); the carried
pairs against zul2c's report and carried.npz (written when UL2's call is filed). Gate G-L: the union of the three stores equals
the filed union of the layout, the stores are disjoint where the dedup says, every count is filed.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import zul2c as L2C  # noqa: E402  (first: imports zul2u -> zul2 -> zmhb -> zenc, so the BLAS threads come from --threads before numpy loads)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import d83stores as DS  # noqa: E402
import zap1 as Z  # noqa: E402
import zul1 as UL  # noqa: E402

L2 = L2C.L2
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "pl1"
DECLARED = "docs/PL1_POOLS_ON_UL2_LAYOUT.md"
SIX = UL.SIX
HAS_S = UL.HAS_S


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zpl1m: {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZPL1M: {msg}")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── pure array code ──────────────────────────────────────────────────────────

def isin(x, sorted_keys):
    return DS.isin_sorted(np.asarray(x, np.int64), sorted_keys)


def union_size(parts):
    """|p1 | p2 | ...| of sorted unique key arrays (UL2c's counter: part by part, no union built)."""
    return L2C.union_count(sorted(parts, key=lambda p: -p.size))


def carried_order(fidx, rank, rank_max):
    """The order key of a carried pair: by the first carried form that keeps it, then its rank in that form's list; float32, larger
    first, always below F0T's (positive) and W2_S's (zero) keys. Ties among near ranks fall to the node id (the store's sort)."""
    return (-(1.0 + fidx.astype(np.float64) + rank.astype(np.float64) / (float(rank_max) + 1.0))).astype(np.float32)


def tables(n, version, S, W2, f0t, K, carried):
    """The three walk stores of one (dataset, version) as pair tables (docs/PL1_POOLS_ON_UL2_LAYOUT.md 2b, amended 2e).
    S: sorted keys or None (RK); W2: sorted keys; f0t: (a, b, w) with a < b ascending; K: (a, b, w) unique pairs a < b ascending;
    carried: (keys ascending, fidx, rank) or None. The typed store holds W2_S (RK) or S and W2_S (SRK, ABL's SR union), every pair
    with its label (0 S, 1 R; S wins a pair both hold); the ner store F0T then the carried forms; the knn store K. Returns the
    tables and the counts."""
    sv = version == "SRK"
    if sv != (S is not None):
        stop(f"version {version} and S {'given' if S is not None else 'missing'} disagree")
    fa, fb, fw = (np.asarray(x) for x in f0t)
    fk = fa.astype(np.int64) * n + fb.astype(np.int64)
    if fk.size and (np.any(np.diff(fk) <= 0) or not (fw > 0).all()):
        stop("F0T is not ascending unique with positive weights")
    ka, kb, kw = (np.asarray(x) for x in K)
    kk = ka.astype(np.int64) * n + kb.astype(np.int64)
    if kk.size and np.any(np.diff(kk) <= 0):
        stop("K is not ascending unique")
    counts = {"version": version, "F0T_pairs": int(fk.size), "W2_S_pairs": int(W2.size), "S_pairs": None if S is None else int(S.size),
              "K_pairs_in": int(kk.size)}
    # ner tiers
    keys, okey, w, tier = [fk], [fw.astype(np.float32)], [fw.astype(np.float32)], [np.zeros(fk.size, np.uint8)]
    if carried is not None:
        ck, cf, cr = carried
        ck = np.asarray(ck, np.int64)
        if ck.size and np.any(np.diff(ck) <= 0):
            stop("carried keys are not ascending unique")
        bad = int(isin(ck, fk).sum() + isin(ck, W2).sum() + isin(ck, kk).sum() + (isin(ck, S).sum() if S is not None else 0))
        if bad:
            stop(f"{bad} carried pairs lie inside B0 (the layout's base)")
        rmax = int(np.max(cr)) if cr.size else 0
        keys.append(ck)
        okey.append(carried_order(np.asarray(cf), np.asarray(cr), rmax))
        w.append(np.zeros(ck.size, np.float32))
        tier.append(np.full(ck.size, 2, np.uint8))
        counts["carried_pairs"] = int(ck.size)
    else:
        counts["carried_pairs"] = 0
    k_all = np.concatenate(keys)
    o = np.argsort(k_all, kind="stable")
    k_all, okey, w, tier = k_all[o], np.concatenate(okey)[o], np.concatenate(w)[o], np.concatenate(tier)[o]
    if k_all.size > 1 and np.any(np.diff(k_all) == 0):
        stop("a pair is in two ner tiers")
    counts["ner_pairs_in"] = int(k_all.size)
    if sv:                                                    # S > R: the ner store loses the pairs S holds (an F0T pair W2_S also holds stays: R inside R)
        gone = isin(k_all, S)
        k_all, okey, w, tier = k_all[~gone], okey[~gone], w[~gone], tier[~gone]
        counts["ner_removed_in_S"] = int(gone.sum())
    else:
        counts["ner_removed_in_S"] = 0
    struct = np.union1d(S, W2) if sv else W2                  # the typed store: ABL's SR union under SRK
    slab = np.ones(struct.size, np.uint8)                     # label 1 = R
    if sv:
        slab[isin(struct, S)] = 0                             # label 0 = S (S > R on a pair both hold)
    held = np.union1d(struct, k_all)
    gone = isin(kk, held)                                     # R > K and S > K
    kk2, kw2 = kk[~gone], kw[~gone]
    counts.update(knn_removed_in_SR=int(gone.sum()), knn_pairs_out=int(kk2.size), ner_pairs_out=int(k_all.size),
                  structural_pairs=int(struct.size), structural_S=int((slab == 0).sum()), structural_R=int((slab == 1).sum()),
                  tiers={"F0T": int((tier == 0).sum()), "carried": int((tier == 2).sum())})
    counts["union"] = union_size([struct, np.unique(k_all), kk2])
    ner = {"a": (k_all // n).astype(np.int32), "b": (k_all % n).astype(np.int32), "order_key": okey, "weight": w, "tier": tier}
    knn = {"a": (kk2 // n).astype(np.int32), "b": (kk2 % n).astype(np.int32), "weight": kw2.astype(np.float32)}
    return {"ner": ner, "knn": knn, "structural_keys": struct, "structural_label": slab}, counts


def expected_union(n, version, S, W2, f0t_keys, K_keys, carried_keys):
    """The filed layout's union of one version, from the sets alone: (S +) W2_S + F0T + K_e3v + the carried pairs not in S."""
    parts = [W2, f0t_keys, K_keys]
    if version == "SRK":
        parts.append(S)
    if carried_keys is not None and carried_keys.size:
        parts.append(carried_keys)
    return union_size([np.unique(p) for p in parts])


def gates(tab, counts, n, version, exp_union):
    """G-L: the families are disjoint where the dedup says, and the union is the layout's."""
    ner_k = tab["ner"]["a"].astype(np.int64) * n + tab["ner"]["b"].astype(np.int64)
    knn_k = tab["knn"]["a"].astype(np.int64) * n + tab["knn"]["b"].astype(np.int64)
    st, lab = tab["structural_keys"], tab["structural_label"]
    s_keys = st[lab == 0]
    g = {"knn_disjoint_from_structural": int(isin(knn_k, st).sum()) == 0, "knn_disjoint_from_ner": int(isin(knn_k, ner_k).sum()) == 0,
         "ner_disjoint_from_S": int(isin(ner_k, s_keys).sum()) == 0,
         "labels": {"S": int(s_keys.size), "R": int((lab == 1).sum()), "S_expected": counts["S_pairs"] or 0,
                    "equal": int(s_keys.size) == (counts["S_pairs"] or 0) and bool(np.all(np.diff(st) > 0)) if st.size > 1 else True},
         "union": {"stores": counts["union"], "layout": int(exp_union)}}
    g["union"]["equal"] = g["union"]["stores"] == g["union"]["layout"]
    g["verdict"] = "PASS" if (g["knn_disjoint_from_structural"] and g["knn_disjoint_from_ner"] and g["ner_disjoint_from_S"]
                              and g["labels"]["equal"] and g["union"]["equal"]) else "FAIL"
    return g


def to_stores(tab, n):
    """(ner, knn) FamilyStores of a layout's tables, rows ordered by the order key (the K store by its cosine)."""
    return DS.weighted_store("ner", n, tab["ner"]["a"], tab["ner"]["b"], tab["ner"]["order_key"]), \
        DS.weighted_store("knn", n, tab["knn"]["a"], tab["knn"]["b"], tab["knn"]["weight"])


# ── the real inputs ──────────────────────────────────────────────────────────

def f0t_triple(d, n):
    """UL1's F0T with D83's weights: zd83.f0t_pairs' rule, read here so no D83 module is imported."""
    import json as _j
    U2R, UL1 = UL.U2R, UL.OUT
    nr = _j.loads((U2R / d / "names.json").read_text(encoding="utf-8"))
    br = _j.loads((UL1 / d / "build.json").read_text(encoding="utf-8"))
    if Z.sha_file(U2R / d / "names.npz") != nr["names_npz_sha256"]:
        stop(f"{d}: U2r's names.npz is not its record's")
    if Z.sha_file(UL1 / d / "f0t.npz") != br["f0t_npz_sha256"] or int(br["nodes"]) != int(n) or int(nr["nodes"]) != int(n):
        stop(f"{d}: UL1's f0t.npz is not its record's, or not of the store's {n} nodes")
    if br["names_npz_sha256"] != nr["names_npz_sha256"]:
        stop(f"{d}: UL1 was built on other U2r names")
    with np.load(U2R / d / "names.npz") as z:
        f0 = (z["cn_all_src"], z["cn_all_dst"], z["cn_all_w"])
    with np.load(UL1 / d / "f0t.npz") as z:
        lt = (z["lt_src"], z["lt_dst"], z["lt_w"])
    a, b, w = DS.form_pairs([f0, lt], int(n))
    if a.size != int(br["f0t_pairs"]):
        stop(f"{d}: F0T holds {a.size:,} pairs, UL1 filed {int(br['f0t_pairs']):,}")
    return a, b, w


def knn_triple(src, dst, w, n):
    """A K file's edges as unique pairs a < b ascending; a pair seen twice keeps the larger weight (zenc1b2.k_edges')."""
    s, t = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    a, b = np.minimum(s, t), np.maximum(s, t)
    ok = a != b
    key, w = a[ok] * n + b[ok], np.asarray(w, np.float32)[ok]
    o = np.lexsort((-w, key))
    key, w = key[o], w[o]
    first = np.r_[True, key[1:] != key[:-1]] if key.size else np.zeros(0, bool)
    key, w = key[first], w[first]
    return (key // n).astype(np.int32), (key % n).astype(np.int32), w


def carried_of(d, carried_forms_n=None):
    """UL2c's carried pairs of dataset d: (keys, first carried form's index, its rank), from carried.npz and ranks.npz, both checked
    against zul2c's report. None until the call is filed."""
    rep = L2C.OUT / "report.json"
    cp = L2C.OUT / d / "carried.npz"
    if not rep.is_file() or not cp.is_file():
        stop(f"{d}: zul2c's report or carried.npz is not filed (the layout is adopted when UL2's call is filed)")
    r = json.loads(rep.read_text(encoding="utf-8"))
    if r.get("verdict") == "REFUSED":
        stop("zul2c's report was refused")
    lay = r["layout"][d]
    if Z.sha_file(cp) != lay["carried_npz_sha256"]:
        stop(f"{d}: carried.npz is not the report's")
    g = json.loads((L2C.OUT / d / "grade.json").read_text(encoding="utf-8"))
    rk = L2C.OUT / d / "ranks.npz"
    if Z.sha_file(rk) != g["ranks_npz_sha256"]:
        stop(f"{d}: ranks.npz is not the grade's")
    with np.load(cp) as z:
        keys, code, forms = z["keys"], z["code"], [str(x) for x in z["forms"]]
    with np.load(rk) as z:
        allF = z["allF"]
        ranks = [z[f"rank__{f}"] for f in forms]
    pos = np.searchsorted(allF, keys)
    if keys.size and not np.array_equal(allF[pos], keys):
        stop(f"{d}: a carried pair is not in ranks.npz's allF")
    first = np.full(keys.size, -1, np.int64)
    rank = np.zeros(keys.size, np.int64)
    for i in range(len(forms) - 1, -1, -1):                   # the lowest set bit wins
        has = (code >> np.uint8(i)) & 1 == 1
        first[has] = i
        rank[has] = ranks[i][pos][has]
    if keys.size and (first < 0).any():
        stop(f"{d}: a carried pair has no carried form")
    if (rank < 0).any():
        stop(f"{d}: a carried pair is not in its form's list")
    return keys, first, rank, {"forms": forms, "report_layout": lay, "bits": {f: int(((code >> np.uint8(i)) & 1).sum()) for i, f in enumerate(forms)}}


def cmd_build(a):
    d, v, t0 = a.ds, a.version, time.time()
    if d not in SIX or v not in L2C.VERSIONS[d]:
        stop(f"{d} has the versions {L2C.VERSIONS.get(d)}")
    served, freeze, n, nsha = L2.dataset_head(a)
    F, shas, _slab = UL.families(served, d, n, nsha, freeze)
    Ke_keys, krec = L2.knn_keys(served, d, n, nsha)
    kp = ROOT / krec["file"]
    with np.load(kp) as z:
        K = knn_triple(z["src"], z["dst"], z["weight"], n)
    if (K[0].astype(np.int64) * n + K[1]).size != Ke_keys.size:
        stop(f"{d}: K_e3v's pairs differ from zul2's")
    ck, cf, cr, crec = carried_of(d)
    S = F.get("S") if v == "SRK" else None
    f0 = f0t_triple(d, n)
    tab, counts = tables(n, v, S, F["W2_S"], f0, K, (ck, cf, cr))
    g = json.loads((L2C.OUT / d / "grade.json").read_text(encoding="utf-8"))
    carried_new = ck if S is None else ck[~isin(ck, S)]
    exp = expected_union(n, v, S, F["W2_S"], F["F0T"], Ke_keys, carried_new)
    gate = gates(tab, counts, n, v, exp)
    filed = int(crec["report_layout"]["union"][v])
    gate["union_vs_report"] = {"stores": counts["union"], "report": filed, "equal": counts["union"] == filed}
    gate["base_vs_grade"] = {"B0_in_layout": int(union_size([np.unique(p) for p in ([F["W2_S"], F["F0T"], Ke_keys] + ([S] if S is not None else []))])),
                             "grade_B0": int(g["sizes"]["B0"][v])}
    gate["base_vs_grade"]["equal"] = gate["base_vs_grade"]["B0_in_layout"] == gate["base_vs_grade"]["grade_B0"]
    if not (gate["union_vs_report"]["equal"] and gate["base_vs_grade"]["equal"]):
        gate["verdict"] = "FAIL"
    od = OUT / d / v
    sha = None
    if gate["verdict"] == "PASS":
        flat = {f"ner_{k}": x for k, x in tab["ner"].items()}
        flat.update({f"knn_{k}": x for k, x in tab["knn"].items()})
        flat.update(struct_key=tab["structural_keys"], struct_label=tab["structural_label"])
        sha = L2.save_npz(od / "layout.npz", **flat)
    rec = {"stage": "PL1 S1", "declared_in": DECLARED, "dataset": d, "version": v, "nodes": n, "nodes_sha256": nsha,
           "freeze_RECORD_SHA256": freeze, "inputs": shas, "knn_e3v": krec, "carried": {"forms": crec["forms"], "bits": crec["bits"]},
           "counts": counts, "gate_GL": gate, "layout_npz_sha256": sha, "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": script_sha256()}
    L2.wjson(od / "layout.json", rec)
    log(f"{d}/{v}: G-L {gate['verdict']} {json.dumps(counts)[:400]} ({rec['seconds']} s)")
    return 0 if gate["verdict"] == "PASS" else 3


# ── self-tests ───────────────────────────────────────────────────────────────

def _toy(rng, n, m):
    k = np.unique(rng.integers(0, n * n, m))
    a, b = k // n, k % n
    ok = a < b
    return np.unique(a[ok] * n + b[ok])


def selftest():
    rng = np.random.default_rng(0)
    n = 400
    for version in ("RK", "SRK"):
        W2, Fk, Kk, Sk = _toy(rng, n, 3000), _toy(rng, n, 3000), _toy(rng, n, 3000), _toy(rng, n, 2000)
        fw = (rng.random(Fk.size) + 0.1).astype(np.float32)
        kw = rng.random(Kk.size).astype(np.float32)
        S = Sk if version == "SRK" else None
        base = [W2, Fk, Kk] + ([Sk] if S is not None else [])
        pool = np.setdiff1d(_toy(rng, n, 4000), np.unique(np.concatenate(base)))
        ck = np.sort(rng.choice(pool, size=min(300, pool.size), replace=False))
        cf, cr = rng.integers(0, 3, ck.size), rng.integers(0, 50, ck.size)
        tab, c = tables(n, version, S, W2, (Fk // n, Fk % n, fw), (Kk // n, Kk % n, kw), (ck, cf, cr))
        g = gates(tab, c, n, version, expected_union(n, version, S, W2, Fk, Kk, ck))
        assert g["verdict"] == "PASS", g
        # brute force with python sets
        W, Ff, Kf, Cf = set(W2.tolist()), set(Fk.tolist()), set(Kk.tolist()), set(ck.tolist())
        Sf = set(Sk.tolist()) if S is not None else set()
        ner = (Ff | Cf) - Sf
        struct = (Sf | W) if version == "SRK" else W
        knn = Kf - struct - ner
        nk = set((tab["ner"]["a"].astype(np.int64) * n + tab["ner"]["b"]).tolist())
        kk = set((tab["knn"]["a"].astype(np.int64) * n + tab["knn"]["b"]).tolist())
        assert nk == ner and kk == knn and set(tab["structural_keys"].tolist()) == struct, version
        labof = dict(zip(tab["structural_keys"].tolist(), tab["structural_label"].tolist()))
        assert all(labof[k] == (0 if k in Sf else 1) for k in struct), version      # S wins a pair both hold
        assert int((tab["structural_label"] == 0).sum()) == len(Sf), version
        # the order: F0T first (descending weight), then carried by form then rank
        ner_st, knn_st = to_stores(tab, n)
        ip, col = np.asarray(ner_st.indptr), np.asarray(ner_st.col)
        tierof = dict(zip((tab["ner"]["a"].astype(np.int64) * n + tab["ner"]["b"]).tolist(), tab["ner"]["tier"].tolist()))
        okeyof = dict(zip((tab["ner"]["a"].astype(np.int64) * n + tab["ner"]["b"]).tolist(), tab["ner"]["order_key"].tolist()))
        for u in range(n):
            row = col[ip[u]:ip[u + 1]].astype(np.int64)
            ks = [min(u, x) * n + max(u, x) for x in row.tolist()]
            tr = [tierof[x] for x in ks]
            assert tr == sorted(tr), (version, u, tr)
            ok = [okeyof[x] for x in ks]
            assert all(ok[i] >= ok[i + 1] for i in range(len(ok) - 1)), (version, u)
        assert int(ner_st.edges_stored) == len(ner) and int(knn_st.edges_stored) == len(knn)
        print(version, "ok", {k: c[k] for k in ("ner_pairs_out", "knn_pairs_out", "knn_removed_in_SR", "union")}, c["tiers"])
    # a carried pair inside B0 is refused
    try:
        tables(n, "RK", None, W2, (Fk // n, Fk % n, fw), (Kk // n, Kk % n, kw), (Kk[:3], np.zeros(3, int), np.zeros(3, int)))
        raise AssertionError("a carried pair inside B0 was accepted")
    except SystemExit:
        pass
    print("selftest ok")
    return 0


def selftest_real(a):
    """SQuAD's real base. K = the package's frozen K (K_e3v's stand-in with the same file shape); the carried pairs are synthetic.
    Checks UL2's filed REC union (W2_S + F0T + K_frozen) equals zul2c's grade, and the plumbing on real arrays."""
    d, v = "squad", "RK"
    a.ds = d
    served, freeze, n, nsha = L2.dataset_head(a)
    F, shas, _ = UL.families(served, d, n, nsha, freeze)
    with np.load(served / d / "graph" / "knn.npz") as z:
        K = knn_triple(z["src"], z["dst"], z["weight"], n)
    f0 = f0t_triple(d, n)
    g = json.loads((L2C.OUT / d / "grade.json").read_text(encoding="utf-8"))
    kk = K[0].astype(np.int64) * n + K[1]
    rec = union_size([np.unique(p) for p in (F["W2_S"], F["F0T"], kk)])
    assert rec == g["sizes"]["REC"][v], (rec, g["sizes"]["REC"][v])
    assert not (kk != F["K"]).any() if kk.size == F["K"].size else True
    rng = np.random.default_rng(1)
    base = np.unique(np.concatenate([F["W2_S"], F["F0T"], kk]))
    cand = np.unique(rng.integers(0, n, (2000, 2)).astype(np.int64))
    cand = _toy(rng, n, 3000)
    cand = np.setdiff1d(cand, base)
    ck = np.sort(cand[:1000])
    cf, cr = rng.integers(0, 2, ck.size), rng.integers(0, 100, ck.size)
    t0 = time.time()
    tab, c = tables(n, v, None, F["W2_S"], f0, K, (ck, cf, cr))
    ge = gates(tab, c, n, v, expected_union(n, v, None, F["W2_S"], F["F0T"], kk, ck))
    assert ge["verdict"] == "PASS", ge
    ner, knn = to_stores(tab, n)
    assert int(ner.edges_stored) == c["ner_pairs_out"] and int(knn.edges_stored) == c["knn_pairs_out"]
    print("selftest-real ok:", json.dumps(c)[:500], f"{time.time() - t0:.1f} s")
    return 0


def cmd_smoke(a):
    """Plumbing only, never graded: one (dataset, version)'s layout with the package's frozen K and no carried pair, under
    outputs/pl1/smoke/<d>/<v>/ (zpl1w's --smoke reads it). It stands in for the real layout while UL2's call is unfiled, so the
    world adapter's e3v path can be run end to end; no number read from it enters any record."""
    d, v, t0 = a.ds, a.version, time.time()
    if d not in SIX or v not in L2C.VERSIONS[d]:
        stop(f"{d} has the versions {L2C.VERSIONS.get(d)}")
    served, freeze, n, nsha = L2.dataset_head(a)
    F, shas, _slab = UL.families(served, d, n, nsha, freeze)
    with np.load(served / d / "graph" / "knn.npz") as z:
        K = knn_triple(z["src"], z["dst"], z["weight"], n)
    kk = K[0].astype(np.int64) * n + K[1]
    S = F.get("S") if v == "SRK" else None
    tab, counts = tables(n, v, S, F["W2_S"], f0t_triple(d, n), K, None)
    exp = expected_union(n, v, S, F["W2_S"], F["F0T"], kk, None)
    gate = gates(tab, counts, n, v, exp)
    od = ROOT / "outputs" / "pl1" / "smoke" / d / v
    sha = None
    if gate["verdict"] == "PASS":
        flat = {f"ner_{k}": x for k, x in tab["ner"].items()}
        flat.update({f"knn_{k}": x for k, x in tab["knn"].items()})
        flat.update(struct_key=tab["structural_keys"], struct_label=tab["structural_label"])
        sha = L2.save_npz(od / "layout.npz", **flat)
    rec = {"stage": "PL1 S1 smoke (plumbing only, never graded)", "declared_in": DECLARED, "dataset": d, "version": v, "nodes": n,
           "smoke": True, "k": "the package's frozen K (not K_e3v)", "carried": "none", "counts": counts, "gate_GL": gate,
           "layout_npz_sha256": sha, "seconds": round(time.time() - t0, 1), "script_sha256": script_sha256()}
    L2.wjson(od / "layout.json", rec)
    log(f"smoke {d}/{v}: G-L {gate['verdict']} {json.dumps(counts)[:300]}")
    return 0 if gate["verdict"] == "PASS" else 3


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ds", required=True)
    b.add_argument("--version", required=True)
    sm = sub.add_parser("smoke")
    sm.add_argument("--ds", required=True)
    sm.add_argument("--version", required=True)
    for s in (b, sm, sub.add_parser("selftest-real")):
        s.add_argument("--host", action="store_true")
        s.add_argument("--threads", type=int, default=2)
    sub.add_parser("selftest")
    a = p.parse_args(argv)
    return {"selftest": lambda _a: selftest(), "selftest-real": selftest_real, "build": cmd_build, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
