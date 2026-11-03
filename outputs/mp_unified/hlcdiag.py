"""HLC: the hyperlinks' lift to our trained models against their lift to HippoRAG 2's walk, question for question, on
HippoRAG 2's three settings (docs/HLC_HYPERLINK_LIFT_MODELS_VS_WALK.md).

    python outputs/mp_unified/hlcdiag.py selftest
    python outputs/mp_unified/hlcdiag.py run          (laptop, minutes) -> outputs/hlc/report.json, report.md

Per question: D = (model with the links - model without) - (A1 walk with the links - A1 walk without), R@5, with B1b's
paired 2,000-resample interval. Family H: the corpus's own hyperlinks (U1a's b1 and b1t reads, HLW's graphs). Family W:
the recreated links W2_S (B1P's b1w and b1t reads, B1F's W2_S bench). Imports hlwdiag, b1b and b1c_walks unchanged;
reads filed read files and reports; writes only under outputs/hlc.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b1b as BB  # noqa: E402  (B1b's per_question, boot_ci and candidate, unchanged)
import hlwdiag as H  # noqa: E402  (HLW's adjacency, restarts and measures, unchanged)

B = H.B  # b1c_walks: family_matrix, transition_T, ppr, rank_by, recall
DECLARED = "docs/HLC_HYPERLINK_LIFT_MODELS_VS_WALK.md"
OUT = ROOT / "outputs" / "hlc"
BENCH_H = ROOT / "outputs" / "bench" / "hipporag2"
BENCH_W = ROOT / "outputs" / "bench" / "hipporag2_w"
SETTINGS = ("2wiki", "hotpotqa", "musique")
SHARED = ("graph_ner.npz", "graph_knn.npz", "first_stage.npz", "queries.json", "nodes.json")
TOL = 5e-5
WALK = "A1"                                   # HLW's deciding restart (HippoRAG 2's own rule)
# family H: U1a's reads (b1 = B1b's, b1t = U1a's text-only carve); J5 decides, the held fit is beside
H_DECIDE = ("2wiki", "hotpotqa")              # musique's structural family is title mentions, not hyperlinks: beside
H_ARMS = {"zsp": "GNN", "zrc": "MLP"}         # B1b's arms: the GNN track's base and the MLP's base
H_HELD = {"musique": "L-musique", "2wiki": "L-2wiki", "hotpotqa": "L-hotpotqa"}
# family W: B1P's reads (b1w against b1t), each question's R@5 averaged over the seeds and splits, as B1P's report does
W_ARMS = {"zg6r": "GNN", "zrckp": "MLP"}
W_SEEDS = (0, 1, 2)
W_SUF = {"WQ-musique": "", "WQ-hotpotqa": "-hp"}
W_IN = {"musique": ("WQ-hotpotqa",), "hotpotqa": ("WQ-musique",), "2wiki": ("WQ-musique", "WQ-hotpotqa")}


def log(m):
    print(f"[hlc {time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ── calls ────────────────────────────────────────────────────────────────────


def ci_of(d):
    d = np.asarray(d, np.float64)
    return {"R@5": round(float(d.mean()), 4), "ci": BB.boot_ci(d)}


def call_of(c):
    """GREATER when the paired interval's lower end is above 0, LESS when its upper end is below 0, else SAME."""
    lo, hi = c["ci"]
    return "GREATER" if lo > 0 else "LESS" if hi < 0 else "SAME"


def stage_call(calls):
    if calls and all(c == "GREATER" for c in calls):
        return "GREATER_EVERYWHERE"
    return "LESS_SOMEWHERE" if "LESS" in calls else "MIXED"


# ── the walk ─────────────────────────────────────────────────────────────────


def pairs_once(d, n):
    """The structural family with each undirected pair once (B1C's family_matrix on the distinct pairs): W2_S's file
    lists every pair in both directions; B1's hyperlinks list each link once."""
    with np.load(d / "graph_structural.npz") as z:
        s, t = z["src"].astype(np.int64), z["dst"].astype(np.int64)
    lo, hi = np.minimum(s, t), np.maximum(s, t)
    u = np.unique(np.stack([lo, hi], 1), axis=0)
    both = np.unique(np.stack([s, t], 1), axis=0)
    rec = {"rows": int(s.size), "pairs": int(u.shape[0]), "self_loops": int((s == t).sum()),
           "every_pair_twice": bool(both.shape[0] == s.size == 2 * u.shape[0])}
    w = np.ones(u.shape[0])
    a = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([u[:, 0], u[:, 1]]), np.concatenate([u[:, 1], u[:, 0]]))),
                      shape=(n, n)).tocsr()
    a.sum_duplicates()
    return a, rec


def walk_lists(fs, fams, graphs, n, r0):
    """{graph: the A1 walk's ranked lists} with HLW's kernel, restart and tie rule (hlwdiag.part_a, A1 only)."""
    out = {}
    for gk, fl in graphs.items():
        a = H.adjacency(fams, fl, n)
        T, deg = B.transition_T(a), H.degree(a)
        out[gk] = B.rank_by(B.ppr(T, H.restart_matrix(WALK, fs, n, deg)), r0, n)
    return out


def setting_walks(name):
    """Per question R@5 (and the arms' means) of R0 and the A1 walk on NK and SNK of B1's graph (family H), and on
    W2_S + NK, pairs once (family W; as stored beside). NK, the first stage and the questions are B1's, byte for byte."""
    t0 = time.time()
    dh, dw = BENCH_H / name, BENCH_W / name
    same = {f: sha_file(dh / f) == sha_file(dw / f) for f in SHARED}
    if not all(same.values()):
        raise SystemExit(f"{name}: W2_S's bench differs from B1's outside the structural family: {same}")
    n = int(read_json(dh / "build.json")["passages"])
    qs = read_json(dh / "queries.json")
    golds = [np.asarray(q["golds"], np.int64) for q in qs]
    with np.load(dh / "first_stage.npz") as z:
        fs = {k: z[k] for k in ("dense", "splade", "rrf")}
    r0 = fs["rrf"].astype(np.int64)
    fams = {f: B.family_matrix(dh, f, n) for f in ("structural", "ner", "knn")}
    lists = {"R0": r0}
    lists.update({f"{WALK}/{g}": L for g, L in walk_lists(fs, fams, {"NK": ("ner", "knn"),
                                                                      "SNK": ("structural", "ner", "knn")},
                                                          n, r0).items()})
    w_once, w_rec = pairs_once(dw, n)
    famw = {"structural": w_once, "ner": fams["ner"], "knn": fams["knn"]}
    lists[f"{WALK}/WNK"] = walk_lists(fs, famw, {"WNK": ("structural", "ner", "knn")}, n, r0)["WNK"]
    fam_s = {"structural": B.family_matrix(dw, "structural", n), "ner": fams["ner"], "knn": fams["knn"]}
    lists[f"{WALK}/WNK_stored"] = walk_lists(fs, fam_s, {"WNK_stored": ("structural", "ner", "knn")}, n, r0)["WNK_stored"]
    M = {arm: H.measures(L, golds) for arm, L in lists.items()}
    log(f"{name}: walks done ({time.time() - t0:.0f}s)")
    return {"ids": [str(q["id"]) for q in qs], "gold_counts": [len(g) for g in golds],
            "r5": {arm: M[arm]["R@5"] for arm in lists},
            "arms": {arm: {m: round(float(v.mean()), 4) for m, v in M[arm].items()} for arm in lists},
            "w2s_structural": w_rec, "shared_files_equal": same, "passages": n}


# ── the models ───────────────────────────────────────────────────────────────


def model_r5(p, ids, gc):
    with np.load(p) as z:
        if [str(x) for x in z["ids"]] != ids:
            raise SystemExit(f"{p}: the read's questions are not queries.json's, in its order")
        return BB.per_question(z, gc)["R@5"]


def family_h(name, wk, u1a):
    """U1a's model lift (b1 - b1t) against the walk's (SNK - NK), J5 and the held fit, both arms."""
    s = u1a["settings"][name]
    rows = {}
    for arm, role in H_ARMS.items():
        for split in ("J5", H_HELD[name]):
            pw = ROOT / "outputs" / "b1b" / "fits" / arm / split / "reads" / f"{name}__b1.npz"
            pt = ROOT / "outputs" / "u1a" / "fits" / "t" / arm / split / "reads" / f"{name}__b1t.npz"
            for p, carve in ((pw, "b1"), (pt, "b1t")):
                if sha_file(p) != s["carves"][carve]["reads"][f"{arm}/{split}"]["file_sha256"]:
                    raise SystemExit(f"{p}: not the read U1a filed")
            mw, mt = model_r5(pw, wk["ids"], wk["gold_counts"]), model_r5(pt, wk["ids"], wk["gold_counts"])
            rows[f"{arm}/{split}"] = cell(mw, mt, wk["r5"][f"{WALK}/SNK"], wk["r5"][f"{WALK}/NK"], wk["r5"]["R0"],
                                          role, "in-domain" if split == "J5" else "held-out")
            filed = s["paired"][f"b1t-b1 {arm}/{split}"]
            got = rows[f"{arm}/{split}"]["model_lift"]
            if abs(got["R@5"] + filed["R@5"]) > TOL or [round(-x, 4) for x in filed["ci"][::-1]] != got["ci"]:
                raise SystemExit(f"{name} {arm}/{split}: the model lift {got} is not U1a's filed b1t-b1 {filed}")
    return rows


def family_w(name, wk, b1p, which="in_domain"):
    """B1P's model lift (b1w - b1t, each question's mean over the seeds and splits) against the walk's (WNK - NK)."""
    s = b1p["settings"][name]
    splits = W_IN[name] if which == "in_domain" else tuple(x for x in W_SUF if x not in W_IN[name])
    rows = {}
    if not splits:                            # 2wiki is in-domain under both splits: no zero-shot read
        return rows
    for arm, role in W_ARMS.items():
        per = {}
        for v, carve in (("w", "b1w"), ("t", "b1t")):
            got = []
            for sp_ in splits:
                for sd in W_SEEDS:
                    fit = f"scr-ml1-{arm}-s{sd}{W_SUF[sp_]}"
                    p = ROOT / "outputs" / "b1p" / "fits" / v / fit / "reads" / f"{name}__{carve}.npz"
                    if sha_file(p) != s["fits"][f"{carve}/{fit}"]["file_sha256"]:
                        raise SystemExit(f"{p}: not the read B1P filed")
                    got.append(model_r5(p, wk["ids"], wk["gold_counts"]))
            per[v] = np.mean(np.stack(got), axis=0)
            want = s[which][f"{arm}/{carve}"]
            if abs(float(per[v].mean()) - want["R@5"]) > TOL or len(got) != want["fits"]:
                raise SystemExit(f"{name} {arm}/{carve} {which}: {per[v].mean():.4f} over {len(got)} fits, not B1P's {want}")
        rows[f"{arm}/{which}"] = cell(per["w"], per["t"], wk["r5"][f"{WALK}/WNK"], wk["r5"][f"{WALK}/NK"],
                                      wk["r5"]["R0"], role, which.replace("_", "-"))
        rows[f"{arm}/{which}"]["walk_stored"] = ci_of(wk["r5"][f"{WALK}/WNK_stored"] - wk["r5"][f"{WALK}/NK"])
        rows[f"{arm}/{which}"]["difference_vs_stored"] = ci_of((per["w"] - per["t"]) -
                                                               (wk["r5"][f"{WALK}/WNK_stored"] - wk["r5"][f"{WALK}/NK"]))
        rows[f"{arm}/{which}"]["call_vs_stored"] = call_of(rows[f"{arm}/{which}"]["difference_vs_stored"])
        filed = s["paired_w_minus_t"][f"{arm}/{which}"]
        got = rows[f"{arm}/{which}"]["model_lift"]
        if abs(got["R@5"] - filed["R@5"]) > TOL or got["ci"] != filed["ci"]:
            raise SystemExit(f"{name} {arm}/{which}: the model lift {got} is not B1P's filed w-t {filed}")
    return rows


def cell(mw, mt, ww, wt, r0, role, read):
    m, w = mw - mt, ww - wt
    d = ci_of(m - w)
    return {"model": role, "read": read,
            "levels": {"model_with": round(float(mw.mean()), 4), "model_without": round(float(mt.mean()), 4),
                       "walk_with": round(float(ww.mean()), 4), "walk_without": round(float(wt.mean()), 4),
                       "R0": round(float(r0.mean()), 4)},
            "model_lift": ci_of(m), "walk_lift": ci_of(w), "difference": d, "call": call_of(d),
            "model_with_minus_walk_with": ci_of(mw - ww), "model_without_minus_walk_with": ci_of(mt - ww)}


# ── run ──────────────────────────────────────────────────────────────────────


def run():
    t0 = time.time()
    hlw, u1a, b1p = (read_json(ROOT / "outputs" / x / "report.json") for x in ("hlw", "u1a", "b1p"))
    rec = {"declared_in": DECLARED, "script_sha256": script_sha256(), "walk": WALK, "candidate": BB.CAND,
           "boot": {"resamples": BB.BOOT, "seed": BB.SEED}, "hipporag2_published": BB.HIPPORAG2,
           "reports_sha256": {x: sha_file(ROOT / "outputs" / x / "report.json") for x in ("hlw", "u1a", "b1p")},
           "settings": {}}
    for name in SETTINGS:
        wk = setting_walks(name)
        filed = hlw["part_a"][name]["arms"]
        for arm in ("R0", f"{WALK}/NK", f"{WALK}/SNK"):
            for m, v in wk["arms"][arm].items():
                if abs(v - filed[arm][m]) > TOL:
                    raise SystemExit(f"{name}: {arm} {m} {v} is not HLW's filed {filed[arm][m]}")
        hw = ci_of(wk["r5"][f"{WALK}/SNK"] - wk["r5"][f"{WALK}/NK"])
        if hw != hlw["part_a"][name]["paired"][WALK]["ADD"]:
            raise SystemExit(f"{name}: the walk's lift {hw} is not HLW's filed ADD")
        rec["settings"][name] = {
            "questions": len(wk["ids"]), "passages": wk["passages"], "walk_arms": wk["arms"],
            "w2s_structural": wk["w2s_structural"], "shared_files_equal": wk["shared_files_equal"],
            "H": family_h(name, wk, u1a),
            "W": {**family_w(name, wk, b1p, "in_domain"), **family_w(name, wk, b1p, "zero_shot")}}
        log(f"{name}: done ({time.time() - t0:.0f}s)")
    dec_h = [rec["settings"][d]["H"][f"{a}/J5"]["call"] for d in H_DECIDE for a in H_ARMS]
    dec_w = [rec["settings"][d]["W"][f"{a}/in_domain"]["call"] for d in SETTINGS for a in W_ARMS]
    rec["call_H"], rec["call_W"] = stage_call(dec_h), stage_call(dec_w)
    rec["deciding_cells"] = {"H": [f"{d} {a}/J5" for d in H_DECIDE for a in H_ARMS],
                             "W": [f"{d} {a}/in_domain" for d in SETTINGS for a in W_ARMS]}
    rec["checks"] = {"hlw_arms": True, "hlw_add": True, "u1a_paired": True, "b1p_paired": True, "read_sha256": True,
                     "read_ids": True, "w2s_bench_shares_b1": True}
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(report_md(rec), encoding="utf-8", newline="\r\n")
    log(f"H {rec['call_H']}, W {rec['call_W']}; wrote {OUT / 'report.json'} ({rec['seconds']:.0f}s)")
    return rec


def report_md(rec):
    def pc(c):
        return f"{100 * c['R@5']:+.1f} [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}]"

    L = [f"# HLC: the hyperlinks' lift to our models against their lift to HippoRAG 2's walk", "",
         f"Declared in {rec['declared_in']}. Calls: family H **{rec['call_H']}**, family W **{rec['call_W']}**. "
         f"R@5 points, paired by question, 95% intervals ({rec['boot']['resamples']} resamples).", ""]
    for fam, title in (("H", "Family H: the corpus's own hyperlinks (U1a's b1 against b1t; the walk's SNK against NK)"),
                       ("W", "Family W: the recreated links W2_S (B1P's b1w against b1t; the walk's W2_S+NK against NK)")):
        L += [f"## {title}", "", "| setting | model | read | model lift | walk lift | difference | call | model with − walk with |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, s in rec["settings"].items():
            for key, c in s[fam].items():
                L.append(f"| {name} | {key} ({c['model']}) | {c['read']} | {pc(c['model_lift'])} | {pc(c['walk_lift'])} | "
                         f"{pc(c['difference'])} | {c['call']} | {pc(c['model_with_minus_walk_with'])} |")
        L.append("")
    L += ["## Levels (R@5)", "", "| setting | R0 | walk NK | walk SNK | walk W2_S+NK | walk W2_S+NK (stored) |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in rec["settings"].items():
        a = s["walk_arms"]
        L.append(f"| {name} | {a['R0']['R@5']} | {a['A1/NK']['R@5']} | {a['A1/SNK']['R@5']} | {a['A1/WNK']['R@5']} | "
                 f"{a['A1/WNK_stored']['R@5']} |")
    L += ["", f"Checks: {rec['checks']}.", ""]
    return "\n".join(L)


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    # 1. the calls
    assert call_of({"R@5": 0.05, "ci": [0.01, 0.09]}) == "GREATER"
    assert call_of({"R@5": -0.05, "ci": [-0.09, -0.01]}) == "LESS"
    assert call_of({"R@5": 0.01, "ci": [-0.01, 0.03]}) == "SAME"
    assert stage_call(["GREATER", "GREATER"]) == "GREATER_EVERYWHERE"
    assert stage_call(["GREATER", "SAME"]) == "MIXED" and stage_call(["GREATER", "LESS"]) == "LESS_SOMEWHERE"
    assert stage_call([]) == "MIXED"
    # 2. pairs once: a file with every pair in both directions gives family_matrix on the distinct pairs
    n = 30
    u = np.unique(np.sort(rng.integers(0, n, size=(40, 2)), axis=1), axis=0)
    u = u[u[:, 0] != u[:, 1]]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:   # b1c's family_matrix leaves its npz open
        d = Path(td)
        np.savez(d / "graph_structural.npz", src=np.concatenate([u[:, 0], u[:, 1]]), dst=np.concatenate([u[:, 1], u[:, 0]]))
        a, r = pairs_once(d, n)
        assert r["every_pair_twice"] and r["pairs"] == u.shape[0] and r["self_loops"] == 0
        np.savez(d / "graph_structural.npz", src=u[:, 0], dst=u[:, 1])
        assert (abs(a - B.family_matrix(d, "structural", n)) > 0).nnz == 0
        with np.load(d / "graph_structural.npz") as stored:
            assert stored["src"].size == u.shape[0]
        np.savez(d / "graph_structural.npz", src=np.concatenate([u[:, 0], u[:, 1]]), dst=np.concatenate([u[:, 1], u[:, 0]]))
        assert np.allclose(B.family_matrix(d, "structural", n).toarray(), 2 * a.toarray())
    # 3. the cell: lifts, difference and levels from per-question arrays
    mw, mt, ww, wt, r0 = (rng.random(200) for _ in range(5))
    c = cell(mw, mt, ww, wt, r0, "GNN", "in-domain")
    assert c["difference"]["R@5"] == round(float(((mw - mt) - (ww - wt)).mean()), 4)
    assert c["model_lift"]["R@5"] == round(float((mw - mt).mean()), 4) and c["call"] == call_of(c["difference"])
    # 4. B1b's interval is symmetric under negation (U1a files b1t - b1; HLC reads b1 - b1t)
    x = rng.normal(0.02, 0.1, 500)
    lo, hi = BB.boot_ci(x)
    assert BB.boot_ci(-x) == [round(-hi, 4), round(-lo, 4)]
    # 5. a read in another question order is refused
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.npz"
        np.savez(p, ids=np.array(["b", "a"]), gold_total=np.array([1, 1]), candidates=np.array([BB.CAND]),
                 top=np.array([[1, 0]]), hit=np.array([[1, 0]]))
        assert model_r5(p, ["b", "a"], [1, 1]).tolist() == [1.0, 0.0]
        try:
            model_r5(p, ["a", "b"], [1, 1])
            raise AssertionError("a read in another order passed")
        except SystemExit:
            pass
    print("hlc selftest: PASS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        selftest()
    elif cmd == "run":
        run()
    else:
        raise SystemExit(__doc__)
