"""Step 2 grade (docs/STEP2_EVAL_MIX_WEIGHTS.md, sections 5 and 6): each of the six seed-0 weighted fits picked by
step 1's rule, against step 1's fit picked by the same rule, on the eval reads: the paired differences, the labels,
the verdict and the diagnostics. It reads both steps' read arrays and records only, after the gates below; it trains
nothing and loads no model.

    python scripts/step2_grade.py [--host] [--step1-fits DIR] [--fits-root DIR] [--step1-grade FILE] [--out-dir DIR]
    python scripts/step2_grade.py --selftest

Gates (a failure stops before any number is computed):
  - outputs/step1/carves.json has the declared sha256 (step1_grade.load_carves_record). outputs/step2/weights.json
    has the declared sha256; each dataset's ids hash to carves.json's fit carve, each question's weight is its
    stratum's, and each dataset's mean weight is 1.
  - step 1's grade (outputs/step1/grade.json) read the same carves.json, and its verdict is ADOPT, NOT_ADOPTED or
    NO_EFFECT. It fixes the primary pick rule: D after ADOPT, M otherwise.
  - each step-1 and step-2 fit passes step1_grade.Fit's gates (records, CPU check PASS, models and read hashes, the
    declared carves, config, seed, hidden and candidates). Step 1's picks recomputed from its fits equal its grade's.
  - each step-2 fit's train.json carries the step-2 record: weights.json's sha256, the trainer
    outputs/mp_unified/lean_gpu2.py, and for each training carve the weights of all its fit questions.
  - the two steps' fits of one name trained on the same cache parts and basis, and read the same questions with the
    same gold totals on every carve.
Reads (section 5): per question of an eval read with gold, the step-2 pick minus the step-1 pick under the primary
  rule on R@5, FC@5 and hit@1, with step1_grade.boot_diff's paired bootstrap (2,000 resamples, default_rng(20261007)
  drawn afresh for each read and each stratum, percentiles 2.5 and 97.5). ABOVE / BELOW: the R@5 interval lies above /
  below 0. AT: it holds 0. There is no SAME: two trainings never share a pick.
Verdict over the eleven primary reads (J5 on its five training datasets, each leave-out fit on its held-out dataset,
  J5 on webqsp): ADOPT (none BELOW, one or more ABOVE), NOT_ADOPTED (one or more BELOW), NO_EFFECT (all AT). The
  secondary reads (each leave-out fit on its four training datasets and on webqsp) get the same labels, ungraded.
Diagnostics (section 6, never graded): the primary reads by stratum (step1_carves.stratum on the eval population's
  rows; webqsp has none); every read under the other pick rule; each step-2 candidate's eval R@5 and the eval-best
  candidate (an oracle, never a result); both steps' picks (variant and state); the references rrf, twin0 and gnn0 on
  the same rows.
Writes <out-dir>/grade.json and <out-dir>/grade.md (default outputs/step2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import step1_grade as G  # noqa: E402

STEP1, STEP2 = ROOT / "outputs" / "step1", ROOT / "outputs" / "step2"
WEIGHTS = STEP2 / "weights.json"
WEIGHTS_SHA256 = "90d263db6de7d4a921df0db5851abc490a5db0639d9412797dde1cf2d595d564"   # the declaration, section 2
TRAINER = "outputs/mp_unified/lean_gpu2.py"
VERDICTS = ("ADOPT", "NOT_ADOPTED", "NO_EFFECT")
RULE_CARVE = {"M": "select", "D": "s1sel"}
METRICS = G.METRICS
log = G.log


def label_of(ci):
    lo, hi = ci
    return "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "AT"


def primary_rule(verdict):
    """Section 5: the rule step 1 adopts, D after ADOPT and M otherwise."""
    if verdict not in VERDICTS:
        raise SystemExit(f"step 1's verdict {verdict!r} is not one of {VERDICTS}")
    return "D" if verdict == "ADOPT" else "M"


def load_weights(carves, path=WEIGHTS):
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != WEIGHTS_SHA256:
        raise SystemExit(f"{path}: sha256 {got} is not the declared {WEIGHTS_SHA256}")
    w = json.loads(raw.decode("utf-8"))
    if sorted(w["per_dataset"]) != sorted(G.TRAIN_ORDER) or w["carves_sha256"] != G.CARVES_SHA256:
        raise SystemExit(f"{path}: not the five training datasets' weights on the declared carves.json")
    for ds, ent in w["per_dataset"].items():
        n = ent["n"]
        if not (len(ent["ids"]) == len(ent["strata"]) == len(ent["weights"]) == n):
            raise SystemExit(f"{path}: {ds} lists {len(ent['ids'])} ids for n = {n}")
        if G.digest(ent["ids"]) != ent["ids_sha256"] or ent["ids_sha256"] != carves["per_dataset"][ds]["fit"]["sha256"]:
            raise SystemExit(f"{path}: {ds}'s ids are not carves.json's fit carve")
        if any(x != ent["weight_of_stratum"][t] for t, x in zip(ent["strata"], ent["weights"])):
            raise SystemExit(f"{path}: {ds} has a question whose weight is not its stratum's")
        if abs(sum(ent["weights"]) / n - 1.0) > 1e-9:
            raise SystemExit(f"{path}: {ds}'s mean weight is {sum(ent['weights']) / n}, not 1")
    return w, got


def step2_gate(fit, wrec, wsha):
    """The step-2 record lean_gpu2.train_cmd adds to train.json: weights.json, the trainer, every fit question."""
    s2 = fit.train.get("step2") or {}
    if s2.get("weights_sha256") != wsha or s2.get("trainer") != TRAINER:
        raise SystemExit(f"{fit.name}: train.json lacks the step-2 record for weights.json {wsha[:12]}")
    want = {f"{d}={G.FIT_CARVE[d]}" for d in G.FITS[fit.name]}
    got = s2.get("weights") or {}
    if set(got) != want:
        raise SystemExit(f"{fit.name}: weighted carves {sorted(got)}, declared {sorted(want)}")
    for key, v in got.items():
        ent = wrec["per_dataset"][key.split("=")[0]]
        if v.get("unit") or v.get("questions") != ent["n"] or v.get("weight_of_stratum") != ent["weight_of_stratum"]:
            raise SystemExit(f"{fit.name}: {key} did not train on weights.json's weights for all {ent['n']} fit questions")
    return s2


def same_inputs(a, b):
    """The two steps' fits of one name: the same cache parts and basis, the same read questions and gold totals."""
    def parts(f):
        return [(t["dataset"], t["carve"], t["questions"], t["rows"], t["carve_ids_sha256"], t["parts"])
                for t in f.train["train"]]
    if parts(a) != parts(b) or a.train["basis_sha256"] != b.train["basis_sha256"]:
        raise SystemExit(f"{a.name}: step 2 trained on other cache parts or another basis than step 1")
    if sorted(a.m) != sorted(b.m):
        raise SystemExit(f"{a.name}: the two steps read different carves")
    for key in a.m:
        if a.ids[key] != b.ids[key] or not np.array_equal(a.gt[key], b.gt[key]):
            raise SystemExit(f"{a.name}: {key} reads other questions or gold totals in the two steps")


def load_step1_grade(path):
    raw = Path(path).read_bytes()
    g1 = json.loads(raw.decode("utf-8"))
    if g1.get("carves_sha256") != G.CARVES_SHA256:
        raise SystemExit(f"{path} graded other carves than the declared carves.json")
    return g1, hashlib.sha256(raw).hexdigest()


def compare(m2, m1):
    c = G.boot_diff(m2, m1)
    return {nm: {"mean": x[0], "ci": x[1]} for nm, x in zip(METRICS, c)}, label_of(c[0][1])


def grade(step1_fits, fits_root, step1_grade_path, out_dir, host, weights_path=WEIGHTS):
    t0 = time.time()
    carves = G.load_carves_record()
    wrec, wsha = load_weights(carves, weights_path)
    g1, g1sha = load_step1_grade(step1_grade_path)
    rule = primary_rule(g1.get("verdict"))
    other = "M" if rule == "D" else "D"
    cands = G.candidate_names()
    f1 = {f: G.Fit(f, Path(step1_fits), carves) for f in G.FITS}
    f2 = {f: G.Fit(f, Path(fits_root), carves) for f in G.FITS}
    pk, s2rec = {}, {}
    for f in G.FITS:
        s2rec[f] = step2_gate(f2[f], wrec, wsha)
        same_inputs(f1[f], f2[f])
        pk[f] = {}
        for a in ("M", "D"):
            s1, s2 = f1[f].select_scores(RULE_CARVE[a]), f2[f].select_scores(RULE_CARVE[a])
            k1, k2 = G.first_best(s1), G.first_best(s2)
            if g1["picks"][f][a]["candidate"] != cands[k1]:
                raise SystemExit(f"{f}: step 1's {a}-pick from its fits is {cands[k1]}, its grade's is "
                                 f"{g1['picks'][f][a]['candidate']}")
            pk[f][a] = {"step1": k1, "step2": k2, "step1_score": s1[k1], "step2_score": s2[k2], "step2_scores": s2}
    log(f"gates pass: step 1's verdict {g1['verdict']}, so the primary pick rule is {rule}")
    strata, placement, freeze = G.strata_rows(carves, host)
    rec = {"declared_in": "docs/STEP2_EVAL_MIX_WEIGHTS.md", "script": "scripts/step2_grade.py",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
           "step1_grade_script_sha256": hashlib.sha256(Path(G.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
           "carves_sha256": G.CARVES_SHA256, "weights_sha256": wsha, "freeze_RECORD_SHA256": freeze,
           "placement": placement, "step1_grade": {"sha256": g1sha, "verdict": g1["verdict"]},
           "primary_rule": rule, "other_rule": other,
           "bootstrap": {"resamples": G.BOOT, "seed": G.BOOT_SEED, "interval": [2.5, 97.5], "per": "read and stratum"},
           "fits": {}, "picks": {}, "reads": [],
           "diagnostics": {"strata": [], "other_rule_reads": [], "oracle": {}, "candidates_eval_r5": {},
                           "refs_equal": {}}}
    for f in G.FITS:
        rec["fits"][f] = {
            "step1_models_sha256": f1[f].train["models_sha256"], "step2_models_sha256": f2[f].train["models_sha256"],
            "step2_read_sha256": f2[f].sha, "step2_weights": s2rec[f]["weights"],
            "step2_trainer_sha256": s2rec[f].get("trainer_sha256"),
            "step2_check": {"verdict": f2[f].check["verdict"],
                            "max_abs_diff": max(c["max_abs_diff"] for c in f2[f].check["carves"].values())},
            "train_seconds": {"step1": f1[f].train.get("seconds"), "step2": f2[f].train.get("seconds")}}
        rec["picks"][f] = {a: {"step1": cands[p["step1"]], "step1_score": p["step1_score"],
                               "step2": cands[p["step2"]], "step2_score": p["step2_score"],
                               "step2_scores": p["step2_scores"]} for a, p in pk[f].items()}
        for ds in G.EVAL_ORDER:
            m1, refs, ok = G.eval_rows(f1[f], ds)
            m2, refs2, _ = G.eval_rows(f2[f], ds)
            rec["diagnostics"]["refs_equal"].setdefault(f, {})[ds] = bool(np.array_equal(refs, refs2))
            ev_r5 = [float(m2[k][:, 0].mean()) for k in range(len(cands))]
            rec["diagnostics"]["candidates_eval_r5"].setdefault(f, {})[ds] = ev_r5
            orc = G.first_best(ev_r5)
            rec["diagnostics"]["oracle"].setdefault(f, {})[ds] = {
                "candidate": cands[orc], "R@5": ev_r5[orc], "step2_pick_R@5": ev_r5[pk[f][rule]["step2"]],
                "step1_pick_R@5": float(m1[pk[f][rule]["step1"]][:, 0].mean()),
                "note": "the eval-best candidate is an oracle, never a result"}
            primary = f == "J5" or ds == f[2:]          # J5 on all six reads; a leave-out fit on its held-out dataset
            role = "in-domain" if ds in G.FITS[f] else "zero-shot"
            a1, a2 = m1[pk[f][rule]["step1"]], m2[pk[f][rule]["step2"]]
            diff, lab = compare(a2, a1)
            row = {"fit": f, "dataset": ds, "role": role, "primary": bool(primary), "rule": rule,
                   "questions": int(ok.sum()), "without_gold": int((~ok).sum()),
                   "step1_pick": cands[pk[f][rule]["step1"]], "step2_pick": cands[pk[f][rule]["step2"]],
                   "step1": G.means(a1), "step2": G.means(a2), "diff": diff, "label": lab,
                   "refs": {nm: G.means(refs[j]) for j, nm in enumerate(G.REFS)}}
            rec["reads"].append(row)
            b1, b2 = m1[pk[f][other]["step1"]], m2[pk[f][other]["step2"]]
            d_o, l_o = compare(b2, b1)
            rec["diagnostics"]["other_rule_reads"].append({
                "fit": f, "dataset": ds, "role": role, "primary": bool(primary), "rule": other,
                "step1_pick": cands[pk[f][other]["step1"]], "step2_pick": cands[pk[f][other]["step2"]],
                "step1": G.means(b1), "step2": G.means(b2), "diff": d_o, "label": l_o})
            if primary and ds in strata:
                ids = np.asarray(f1[f].ids[(ds, "s1eval")])[ok]
                st = np.asarray([strata[ds][q] for q in ids])
                for s in sorted(set(st)):
                    sel = st == s
                    d_s, l_s = compare(a2[sel], a1[sel])
                    rec["diagnostics"]["strata"].append({
                        "fit": f, "dataset": ds, "stratum": str(s), "questions": int(sel.sum()),
                        "step1": G.means(a1[sel]), "step2": G.means(a2[sel]), "diff": d_s, "label": l_s,
                        "refs": {nm: G.means(refs[j][sel]) for j, nm in enumerate(G.REFS)}})
            log(f"{f} {ds} ({role}{', primary' if primary else ''}, rule {rule}): step 1 {row['step1_pick']} "
                f"{row['step1'][0]:.4f}  step 2 {row['step2_pick']} {row['step2'][0]:.4f}  diff "
                f"{diff['R@5']['mean']:+.4f} [{diff['R@5']['ci'][0]:+.4f}, {diff['R@5']['ci'][1]:+.4f}] {lab}")
    prim = [r for r in rec["reads"] if r["primary"]]
    if len(prim) != 11:
        raise SystemExit(f"{len(prim)} primary reads, declared 11")
    rec["verdict"] = G.verdict_of([r["label"] for r in prim])
    rec["primary_labels"] = {f"{r['fit']}:{r['dataset']}": r["label"] for r in prim}
    rec["seconds"] = time.time() - t0
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("grade.json", json.dumps(rec, indent=1)), ("grade.md", render(rec))):
        tmp = out_dir / (name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, out_dir / name)
    log(f"verdict {rec['verdict']}: {rec['primary_labels']} ({rec['seconds']:.0f}s); {out_dir / 'grade.json'}")
    return 0


def fmt_ci(d):
    return f"{d['mean']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"


def render(rec):
    r_ = rec["primary_rule"]
    L = ["# Step 2 grade (seed 0)", "",
         f"Verdict over the eleven primary reads: **{rec['verdict']}**. Step 1's verdict was {rec['step1_grade']['verdict']}, "
         f"so both steps' fits are picked by rule {r_} ({'the matched select carves' if r_ == 'D' else 'M3B select carves'}). "
         "Development numbers; the paper's numbers come from one declared confirmation run.", "",
         "## Picks", "", "| fit | rule | step 1 pick (score) | step 2 pick (score) |", "| --- | --- | --- | --- |"]
    for f, p in rec["picks"].items():
        for a in (r_, rec["other_rule"]):
            L.append(f"| {f} | {a}{'' if a == r_ else ' (other)'} | {p[a]['step1']} ({p[a]['step1_score']:.4f}) | "
                     f"{p[a]['step2']} ({p[a]['step2_score']:.4f}) |")
    for title, prim in (("Primary reads (graded on R@5)", True), ("Secondary reads (not graded)", False)):
        L += ["", f"## {title}", "",
              "| fit | read | role | R@5 step 1 | R@5 step 2 | step 2 - step 1 R@5 [95%] | label | FC@5 [95%] | hit@1 [95%] | twin0 R@5 | gnn0 R@5 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in rec["reads"]:
            if r["primary"] != prim:
                continue
            L.append(f"| {r['fit']} | {r['dataset']} | {r['role']} | {r['step1'][0]:.4f} | {r['step2'][0]:.4f} | "
                     f"{fmt_ci(r['diff']['R@5'])} | {r['label']} | {fmt_ci(r['diff']['FC@5'])} | "
                     f"{fmt_ci(r['diff']['hit@1'])} | {r['refs']['twin0'][0]:.4f} | {r['refs']['gnn0'][0]:.4f} |")
    L += ["", "## Primary reads by stratum (diagnostic)", "",
          "| fit | read | stratum | questions | R@5 step 1 | R@5 step 2 | step 2 - step 1 R@5 [95%] | label |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in rec["diagnostics"]["strata"]:
        L.append(f"| {s['fit']} | {s['dataset']} | {s['stratum']} | {s['questions']} | {s['step1'][0]:.4f} | "
                 f"{s['step2'][0]:.4f} | {fmt_ci(s['diff']['R@5'])} | {s['label']} |")
    L += ["", f"## Every read under the other pick rule ({rec['other_rule']}; diagnostic)", "",
          "| fit | read | role | primary | step 1 pick | step 2 pick | R@5 step 1 | R@5 step 2 | step 2 - step 1 R@5 [95%] | label |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rec["diagnostics"]["other_rule_reads"]:
        L.append(f"| {r['fit']} | {r['dataset']} | {r['role']} | {'yes' if r['primary'] else 'no'} | {r['step1_pick']} | "
                 f"{r['step2_pick']} | {r['step1'][0]:.4f} | {r['step2'][0]:.4f} | {fmt_ci(r['diff']['R@5'])} | {r['label']} |")
    L += ["", "## Step 2's eval-best candidate (an oracle, never a result)", "",
          "| fit | read | eval-best | R@5 | step 2 pick R@5 | step 1 pick R@5 |", "| --- | --- | --- | --- | --- | --- |"]
    for f, o in rec["diagnostics"]["oracle"].items():
        for ds, v in o.items():
            L.append(f"| {f} | {ds} | {v['candidate']} | {v['R@5']:.4f} | {v['step2_pick_R@5']:.4f} | {v['step1_pick_R@5']:.4f} |")
    L += ["", "## Weights used (per fit and training carve)", "", "| fit | carve | questions | mean | min | max |",
          "| --- | --- | --- | --- | --- | --- |"]
    for f, x in rec["fits"].items():
        for key, v in x["step2_weights"].items():
            L.append(f"| {f} | {key} | {v['questions']} | {v['mean']:.4f} | {v['min']:.4f} | {v['max']:.4f} |")
    L += ["", f"carves.json {rec['carves_sha256'][:12]}..., weights.json {rec['weights_sha256'][:12]}..., step 1 grade "
          f"{rec['step1_grade']['sha256'][:12]}..., freeze {str(rec['freeze_RECORD_SHA256'])[:12]}..., script "
          f"{rec['script_sha256'][:12]}..., {rec['utc']}", ""]
    return "\n".join(L)


# ── selftest ─────────────────────────────────────────────────────────────────

DECLARED = {   # docs/STEP2_EVAL_MIX_WEIGHTS.md section 2, to three decimals
    "metaqa": {"hop1": 0.874, "hop2": 1.052, "hop3": 1.052},
    "squad": {"answerable=False": 1.500, "answerable=True": 0.749},
    "musique": {"hop2": 0.718, "hop3": 1.429, "hop4": 2.846},
    "hotpotqa": {"easy/bridge": 0.279, "easy/comparison": 0.279, "medium/bridge": 0.279, "medium/comparison": 0.279,
                 "hard/bridge": 4.460, "hard/comparison": 4.460},
    "2wiki": {"inference": 4.037, "bridge_comparison": 1.115, "comparison": 0.790, "compositional": 0.904}}


def selftest():
    ok = []
    assert [label_of([0.001, 0.01]), label_of([-0.02, -0.001]), label_of([-0.01, 0.01]), label_of([0.0, 0.01]),
            label_of([-0.01, 0.0]), label_of([0.0, 0.0])] == ["ABOVE", "BELOW", "AT", "AT", "AT", "AT"]
    assert G.verdict_of(["AT", "ABOVE"]) == "ADOPT" and G.verdict_of(["ABOVE", "BELOW"]) == "NOT_ADOPTED"
    assert G.verdict_of(["AT", "AT"]) == "NO_EFFECT"
    ok.append("labels (no SAME) and verdict")
    assert [primary_rule(v) for v in VERDICTS] == ["D", "M", "M"]
    try:
        primary_rule(None)
        raise AssertionError("a missing step-1 verdict must stop the grade")
    except SystemExit:
        pass
    ok.append("the primary rule follows step 1's verdict")
    rng = np.random.default_rng(5)
    a, b = rng.random((301, 3)), rng.random((301, 3))
    d, lab = compare(a, b)
    want = G.boot_diff(a, b)
    assert [d[nm]["mean"] for nm in METRICS] == [x[0] for x in want] and d["R@5"]["ci"] == want[0][1]
    assert lab == label_of(want[0][1])
    ok.append("the comparison is step1_grade's paired bootstrap")
    carves = G.load_carves_record()
    wrec, wsha = load_weights(carves)
    for ds, tab in DECLARED.items():
        got = {t: round(x, 3) for t, x in wrec["per_dataset"][ds]["weight_of_stratum"].items()}
        assert got == tab, (ds, got, tab)
    ok.append(f"weights.json ({wsha[:12]}) passes its gates and matches the declared table")
    fake = type("F", (), {})()
    fake.name = "L-musique"
    fake.train = {"step2": {"weights_sha256": wsha, "trainer": TRAINER, "weights": {
        f"{d}={G.FIT_CARVE[d]}": {"questions": wrec["per_dataset"][d]["n"],
                                  "weight_of_stratum": wrec["per_dataset"][d]["weight_of_stratum"]}
        for d in G.FITS["L-musique"]}}}
    step2_gate(fake, wrec, wsha)
    for bad in ("sha", "unit", "short", "carve"):
        x = json.loads(json.dumps(fake.train))
        if bad == "sha":
            x["step2"]["weights_sha256"] = "0" * 64
        elif bad == "unit":
            x["step2"]["weights"]["2wiki=fit"]["unit"] = True
        elif bad == "short":
            x["step2"]["weights"]["2wiki=fit"]["questions"] -= 1
        else:
            del x["step2"]["weights"]["2wiki=fit"]
        fake2 = type("F", (), {})()
        fake2.name, fake2.train = "L-musique", x
        try:
            step2_gate(fake2, wrec, wsha)
            raise AssertionError(f"the step-2 gate let a {bad} record through")
        except SystemExit:
            pass
    ok.append("the step-2 record gate")
    log("selftest: " + "; ".join(ok) + ". all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", action="store_true", help="the verified mirror in place of the package (lean_host)")
    ap.add_argument("--step1-fits", default=str(STEP1 / "fits"))
    ap.add_argument("--fits-root", default=str(STEP2 / "fits"))
    ap.add_argument("--step1-grade", default=str(STEP1 / "grade.json"))
    ap.add_argument("--out-dir", default=str(STEP2))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    return grade(Path(a.step1_fits), Path(a.fits_root), Path(a.step1_grade), Path(a.out_dir), a.host)


if __name__ == "__main__":
    sys.exit(main())
