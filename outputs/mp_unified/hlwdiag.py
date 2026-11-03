"""HLW: what the corpus hyperlinks add to HippoRAG 2's walk, on HippoRAG 2's own settings
(docs/HLW_HYPERLINK_LIFT_OF_PPR.md).

    python outputs/mp_unified/hlwdiag.py selftest
    python outputs/mp_unified/hlwdiag.py run        -> outputs/hlw/report.json, report.md

Part A walks B1's graphs (none, S, NK, SNK) from four restarts (A1 decides) with B1C's kernel; part B reads P3a's filed
full-corpus orders. Imports scripts/b1c_walks.py unchanged; writes only under outputs/hlw.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import b1c_walks as B  # noqa: E402  (imported, not edited)

DECLARED = "docs/HLW_HYPERLINK_LIFT_OF_PPR.md"
OUT = ROOT / "outputs" / "hlw"
SETTINGS = ("2wiki", "hotpotqa", "musique")
HYPERLINKED = ("2wiki", "hotpotqa")          # structural = the corpus's own hyperlinks; musique's = title mentions
GRAPHS = {"none": (), "S": ("structural",), "NK": ("ner", "knn"), "SNK": ("structural", "ner", "knn")}
RESTARTS = ("A1", "A2", "T5", "RW")
DECIDE = "A1"
MARGIN = 0.010
KS = (2, 5, 10)
RRF_C = 60
SEED_K = 5
B1C_SAME = {"T5/S": "T5/structural", "T5/SNK": "T5/all", "RW/S": "RW/structural", "RW/SNK": "RW/all"}
P3A_ROWS = ("RRF", "A1", "A1_all", "A2", "A2_all")


def log(m):
    print(f"[hlw {time.strftime('%H:%M:%S')}] {m}", flush=True)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── restarts (step 4e's A1 and A2, B1C's T5 and RW) ─────────────────────────


def seeds_of(d_row, s_row):
    """m3b's frozen seed rule: the dense top 5, then the SPLADE top 5, first occurrence kept."""
    merged = np.concatenate((d_row[:SEED_K], s_row[:SEED_K])).astype(np.int64)
    _, first = np.unique(merged, return_index=True)
    return merged[np.sort(first)]


def rrf_weights(d_ids, s_ids, constant=RRF_C):
    """step 4e's: rows of the two lists and 1/(c + dense rank) + 1/(c + SPLADE rank)."""
    w = 1.0 / (constant + np.arange(1, d_ids.size + 1, dtype=np.float64))
    w2 = 1.0 / (constant + np.arange(1, s_ids.size + 1, dtype=np.float64))
    ids = np.concatenate([np.asarray(d_ids, np.int64), np.asarray(s_ids, np.int64)])
    u, inv = np.unique(ids, return_inverse=True)
    return u, np.bincount(inv, weights=np.concatenate([w, w2]), minlength=u.size)


def spec(nodes, deg):
    return 1.0 / np.log(math.e + deg[nodes].astype(np.float64))


def hippo_restart(arm, seeds, d_ids, s_ids, deg):
    """(nodes, weights summing to 1): half on the seeds, half on the RRF weights; A2 scales both by specificity."""
    sn = np.unique(np.asarray(seeds, np.int64))
    sw = np.ones(sn.size)
    rn, rw = rrf_weights(d_ids, s_ids)
    if arm == "A2":
        sw = sw * spec(sn, deg)
        rw = rw * spec(rn, deg)
    nodes = np.concatenate([sn, rn])
    wts = np.concatenate([0.5 * sw / sw.sum(), 0.5 * rw / rw.sum()])
    u, inv = np.unique(nodes, return_inverse=True)
    return u, np.bincount(inv, weights=wts, minlength=u.size)


def restart_matrix(arm, fs, n, deg):
    if arm in ("T5", "RW"):
        return B.restart_matrix(arm, fs["rrf"].astype(np.int64), n)
    d, s = fs["dense"].astype(np.int64), fs["splade"].astype(np.int64)
    r = np.zeros((n, d.shape[0]))
    for j in range(d.shape[0]):
        dj, sj = d[j][d[j] >= 0], s[j][s[j] >= 0]
        u, w = hippo_restart(arm, seeds_of(dj, sj), dj, sj, deg)
        r[u, j] = w
    return r


# ── graphs, walks and measures ───────────────────────────────────────────────


def adjacency(fams, fl, n):
    a = sp.csr_matrix((n, n))
    for f in fl:
        a = a + fams[f]
    a = a.tocsr()
    a.sum_duplicates()
    return a


def degree(a):
    """Distinct neighbours of each node in the walked graph (self loops left out)."""
    b = a.tolil()
    b.setdiag(0)
    b = b.tocsr()
    b.eliminate_zeros()
    return np.diff(b.indptr).astype(np.int64)


def measures(L, golds):
    out = {f"R@{k}": B.recall(L, golds, k) for k in KS}
    out["FC@5"] = np.array([float(np.isin(g, L[i, :5]).all()) for i, g in enumerate(golds)])
    return out


def ci_of(d):
    lo, hi = B.boot_ci(np.asarray(d, np.float64))
    return {"R@5": round(float(np.mean(d)), 4), "ci": [round(lo, 4), round(hi, 4)]}


def call_of(add, alone, margin=MARGIN):
    """NO_LIFT / LIFT / SMALL from the two paired R@5 differences ({"R@5", "ci"} each)."""
    if add["ci"][1] < margin and alone["ci"][1] < margin:
        return "NO_LIFT"
    if any(x["ci"][0] > 0 and x["R@5"] >= margin for x in (add, alone)):
        return "LIFT"
    return "SMALL"


def side(d):
    lo, hi = d["ci"]
    return "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "SAME"


def part_a(name):
    t0 = time.time()
    d = B.BENCH / name
    n = int(json.loads((d / "build.json").read_text(encoding="utf-8"))["passages"])
    qrec = json.loads((d / "queries.json").read_text(encoding="utf-8"))
    golds = [np.asarray(q["golds"], np.int64) for q in qrec]
    if not all(g.size and g.min() >= 0 and g.max() < n for g in golds):
        raise SystemExit(f"{name}: a question with no gold or a gold outside the setting")
    with np.load(d / "first_stage.npz") as z:
        fs = {k: z[k] for k in ("dense", "splade", "rrf")}
    fams = {f: B.family_matrix(d, f, n) for f in ("structural", "ner", "knn")}
    lists = {"R0": fs["rrf"].astype(np.int64)}
    for gk, fl in GRAPHS.items():
        a = adjacency(fams, fl, n)
        T, deg = B.transition_T(a), degree(a)
        for rk in RESTARTS:
            lists[f"{rk}/{gk}"] = B.rank_by(B.ppr(T, restart_matrix(rk, fs, n, deg)), lists["R0"], n)
        log(f"{name}: graph {gk} walked ({time.time() - t0:.0f}s)")
    M = {arm: measures(L, golds) for arm, L in lists.items()}
    # checks against B1C's filed arms
    filed = json.loads((B.BENCH / "b1c.json").read_text(encoding="utf-8"))["settings"][name]["arms"]
    for ours, theirs in [("R0", "R0")] + list(B1C_SAME.items()):
        for k in KS:
            got, want = round(float(M[ours][f"R@{k}"].mean()), 4), filed[theirs][f"R@{k}"]["all"]
            if abs(got - want) > 5e-5:
                raise SystemExit(f"{name}: {ours} R@{k} {got} != B1C's {theirs} {want}")
    rec = {"questions": len(golds), "passages": n,
           "nnz": {f: int(fams[f].nnz) for f in fams},
           "arms": {arm: {m: round(float(v.mean()), 4) for m, v in M[arm].items()} for arm in lists},
           "paired": {}, "calls": {}}
    r5 = {arm: M[arm]["R@5"] for arm in lists}
    for rk in RESTARTS:
        p = {"ADD": ci_of(r5[f"{rk}/SNK"] - r5[f"{rk}/NK"]), "ALONE": ci_of(r5[f"{rk}/S"] - r5[f"{rk}/none"]),
             "SNK-R0": ci_of(r5[f"{rk}/SNK"] - r5["R0"]), "S-R0": ci_of(r5[f"{rk}/S"] - r5["R0"]),
             "NK-R0": ci_of(r5[f"{rk}/NK"] - r5["R0"])}
        rec["paired"][rk] = p
        rec["calls"][rk] = {"hyperlinks": call_of(p["ADD"], p["ALONE"]) if name in HYPERLINKED else "BESIDE",
                            "SNK_vs_R0": side(p["SNK-R0"]), "S_vs_R0": side(p["S-R0"])}
    log(f"{name}: done ({time.time() - t0:.0f}s); A1 ADD {rec['paired']['A1']['ADD']} ALONE "
        f"{rec['paired']['A1']['ALONE']}")
    return rec


# ── part B: P3a's filed full-corpus orders ───────────────────────────────────


def part_b(name):
    with np.load(ROOT / "outputs" / "p3a" / f"orders_{name}.npz", allow_pickle=False) as z:
        g = z["gptr"]
        nq = len(z["ids"])
        q = np.repeat(np.arange(nq), np.diff(g))
        r5 = {}
        for row in P3A_ROWS:
            pos = z["pos_" + row]
            hit = ((pos >= 0) & (pos < 5)).astype(np.float64)
            r5[row] = np.bincount(q, weights=hit, minlength=nq) / np.maximum(np.diff(g), 1)
    rec = {"questions": nq, "R@5": {row: round(float(v.mean()), 4) for row, v in r5.items()}, "paired": {}}
    for row in P3A_ROWS[1:]:
        rec["paired"][f"{row}-RRF"] = ci_of(r5[row] - r5["RRF"])
    rec["paired"]["A1_all-A1"] = ci_of(r5["A1_all"] - r5["A1"])
    return rec


# ── beside: the models' lift from the hyperlinks (filed) ─────────────────────


def beside():
    out = {}
    u1a = json.loads((ROOT / "outputs" / "u1a" / "report.json").read_text(encoding="utf-8"))
    b1p = json.loads((ROOT / "outputs" / "b1p" / "report.json").read_text(encoding="utf-8"))
    for name in SETTINGS:
        pu = u1a["settings"][name]["paired"]
        out[name] = {"u1a_b1_minus_b1t": {k.replace("b1t-b1 ", ""): {"R@5": round(-v["R@5"], 4),
                                                                     "ci": [round(-v["ci"][1], 4), round(-v["ci"][0], 4)]}
                                          for k, v in pu.items() if k.startswith("b1t-b1 ")},
                     "b1p_b1w_minus_b1t": b1p["settings"][name].get("paired_w_minus_t")}
    return out


def report_md(rec):
    L = ["# HLW: what the corpus hyperlinks add to HippoRAG 2's walk (part A: B1; part B: full corpora)", "",
         f"Declared in {DECLARED}. Overall call: **{rec['call']}**.", "",
         "## Part A, B1 (1,000 questions each), R@5 paired differences [95% CI]", "",
         "| setting | restart | ADD (SNK - NK) | ALONE (S - none) | SNK - R0 | S - R0 | call |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    f = lambda x: f"{x['R@5']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}]"  # noqa: E731
    for name in SETTINGS:
        a = rec["part_a"][name]
        for rk in RESTARTS:
            p = a["paired"][rk]
            L.append(f"| {name} | {rk} | {f(p['ADD'])} | {f(p['ALONE'])} | {f(p['SNK-R0'])} | {f(p['S-R0'])} | "
                     f"{a['calls'][rk]['hyperlinks']} |")
    L += ["", "R@5 of every arm:", "", "| arm | " + " | ".join(SETTINGS) + " |", "| --- |" + " ---: |" * len(SETTINGS)]
    for arm in rec["part_a"][SETTINGS[0]]["arms"]:
        L.append(f"| {arm} | " + " | ".join(f"{rec['part_a'][s]['arms'][arm]['R@5']:.4f}" for s in SETTINGS) + " |")
    L += ["", "## Part B, full corpora (P3a's filed orders, every s1eval question)", "",
          "| dataset | questions | RRF | A1 (hyperlinks) | A1_all | A1 - RRF | A1_all - RRF | A1_all - A1 |",
          "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |"]
    for name, b in rec["part_b"].items():
        p = b["paired"]
        L.append(f"| {name} | {b['questions']} | {b['R@5']['RRF']:.4f} | {b['R@5']['A1']:.4f} | {b['R@5']['A1_all']:.4f} "
                 f"| {f(p['A1-RRF'])} | {f(p['A1_all-RRF'])} | {f(p['A1_all-A1'])} |")
    L += ["", "No graph without the hyperlinks was walked at full scale (P3a), so part B has no ADD or ALONE.", "",
          "## Beside: the trained models' lift from the hyperlinks (filed, deciding nothing)", ""]
    for name, b in rec["beside"].items():
        L.append(f"- {name}: U1a b1 - b1t " + "; ".join(f"{k} {f(v)}" for k, v in b["u1a_b1_minus_b1t"].items()))
    L.append("")
    return "\n".join(L)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"declared_in": DECLARED, "script_sha256": script_sha256(), "b1c_sha256": B.sha(ROOT / "scripts" / "b1c_walks.py"),
           "margin": MARGIN, "decide": DECIDE, "part_a": {}, "part_b": {}}
    for name in SETTINGS:
        rec["part_a"][name] = part_a(name)
    for name in HYPERLINKED:
        rec["part_b"][name] = part_b(name)
    rec["beside"] = beside()
    calls = [rec["part_a"][s]["calls"][DECIDE]["hyperlinks"] for s in HYPERLINKED]
    rec["call"] = ("PPR_NO_HYPERLINK_LIFT" if all(c == "NO_LIFT" for c in calls)
                   else "PPR_HYPERLINK_LIFT" if "LIFT" in calls else "MIXED")
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "report.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    (OUT / "report.md").write_text(report_md(rec), encoding="utf-8")
    log(f"call {rec['call']}; wrote {OUT / 'report.json'}")


def selftest():
    rng = np.random.default_rng(0)
    n = 40
    d_ids, s_ids = rng.permutation(n)[:12], rng.permutation(n)[:12]
    deg = rng.integers(0, 6, n)
    import importlib
    E = importlib.import_module("step4e_paper_pools")
    for arm in ("A1", "A2"):
        u, w = hippo_restart(arm, seeds_of(d_ids, s_ids), d_ids, s_ids, deg)
        u2, w2 = E.restart(arm, seeds_of(d_ids, s_ids), d_ids, s_ids, np.empty(0, np.int64), deg, RRF_C)
        assert np.array_equal(u, u2) and np.allclose(w, w2, atol=1e-12), arm
        assert abs(w.sum() - 1) < 1e-12
    fs = {"dense": np.stack([d_ids, s_ids]), "splade": np.stack([s_ids, d_ids]), "rrf": np.stack([d_ids, s_ids])}
    for arm in RESTARTS:
        r = restart_matrix(arm, fs, n, deg)
        assert np.allclose(r.sum(0), 1.0), arm
    a0 = adjacency({}, (), n)
    T0 = B.transition_T(a0)
    r = restart_matrix("A1", fs, n, deg)
    assert np.allclose(B.ppr(T0, r), r)                       # the none graph returns its restart
    sc = np.zeros((n, 1))
    sc[[3, 7], 0] = 1.0                                        # a tie in mass goes to the better RRF rank
    L = B.rank_by(sc, np.array([[7, 3] + [i for i in range(n) if i not in (3, 7)]]), n, top=5)
    assert list(L[0, :2]) == [7, 3]
    no = {"R@5": 0.002, "ci": [-0.003, 0.006]}
    yes = {"R@5": 0.02, "ci": [0.01, 0.03]}
    small = {"R@5": 0.004, "ci": [-0.002, 0.012]}
    assert call_of(no, no) == "NO_LIFT" and call_of(yes, no) == "LIFT" and call_of(small, no) == "SMALL"
    assert degree(sp.csr_matrix(np.array([[1.0, 2.0], [2.0, 0.0]]))).tolist() == [1, 1]
    print("hlw selftest: PASS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        selftest()
    elif cmd == "run":
        run()
    else:
        raise SystemExit(__doc__)
