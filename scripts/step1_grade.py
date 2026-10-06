"""Step 1 grade (docs/STEP1_MATCHED_SELECTION.md, sections 6 to 8): the M-pick and the D-pick of each of the six
seed-0 fits, their paired differences on the eval reads, the labels and the verdict, and the diagnostics. It reads the
fits' read arrays (outputs/step1/fits/<fit>/reads/*.npz) and records only, after the gates below; it trains nothing
and loads no model.

    python scripts/step1_grade.py [--host] [--fits-root DIR] [--out-dir DIR]
    python scripts/step1_grade.py --selftest

Gates (a failure stops before any number is computed):
  - outputs/step1/carves.json has the declared sha256, and the package's freeze is the one carves.json read.
  - each fit has train.json, read.json and check.json; check.json's verdict is PASS and it covers exactly the read's
    carves; models.pt's sha256 equals the train and read records'; each read array's sha256 equals the read record's.
  - the fit trains on the declared carves (musique's fit carve is s1fit) with config 2e-3:1e-4:0.1:8:2, seed 0 and
    hidden 128, and its candidates are the 36 declared, in the tie order (p, pf, n, nf; epochs 0 to 7, then SWA).
  - it reads exactly the declared carves (its training datasets' select and s1sel carves, the six s1eval carves), and
    each read's ids hash to carves.json's sha256 for that carve (select: M3B's select digest).
Picks (section 6): per candidate, the mean over the fit's pooled select rows (M: its select carves; D: its s1sel
  carves, in training order) of (R@5 + FC@5) / 2, a row without gold counting 0 (lean_mlp.quality, as fit8); the
  highest wins and a tie goes to the candidate first in the declared order (fit8's strict >). The per-variant picks
  take the state only, by the same rule.
Reads (section 7): per question of an eval read with gold (lean_mlp.read_one's rows), D-pick minus M-pick on R@5,
  FC@5 and hit@1 (lean_mlp.row_metrics' rule); the paired bootstrap over questions as lean_mlp.boot_diff (2,000
  resamples, numpy default_rng(20261007) drawn afresh for each read and each stratum, percentiles 2.5 and 97.5).
  SAME: the two picks are one candidate. ABOVE / BELOW: the R@5 interval lies above / below 0. AT: it holds 0.
Verdict over the eleven primary reads (J5 on its five training datasets, each leave-out fit on its held-out dataset,
  J5 on webqsp): ADOPT (none BELOW, one or more ABOVE), NOT_ADOPTED (one or more BELOW), NO_EFFECT (all SAME or AT).
  The secondary reads (each leave-out fit on its four training datasets and on webqsp) get the same labels, ungraded.
Diagnostics (section 8, never graded): per fit, the Spearman correlation over the 36 candidates between select
  quality ((R@5 + FC@5) / 2, as the pick) and eval R@5, for each training dataset's own M and D carves and for the
  pooled carves the picks use; every candidate's eval R@5 and the eval-best candidate (an oracle, never a result); the
  primary reads by stratum (step1_carves.stratum on the eval population's rows; webqsp has none); both picks' variant
  and state and the per-variant picks; the references rrf, twin0 and gnn0 on the same rows.
Writes <out-dir>/grade.json and <out-dir>/grade.md (default outputs/step1).
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
STEP1 = ROOT / "outputs" / "step1"
CARVES = STEP1 / "carves.json"
CARVES_SHA256 = "53cfb89f1e41a76257113719639f96318d14279f86b50a970f9a0bbf6686d746"   # the declaration, section 2
TRAIN_ORDER = ("metaqa", "squad", "musique", "hotpotqa", "2wiki")
EVAL_ORDER = TRAIN_ORDER + ("webqsp",)
FIT_CARVE = {d: ("s1fit" if d == "musique" else "fit") for d in TRAIN_ORDER}
FITS = {"J5": TRAIN_ORDER, **{f"L-{d}": tuple(x for x in TRAIN_ORDER if x != d) for d in TRAIN_ORDER}}
VARIANTS = ("p", "pf", "n", "nf")
EPOCHS = 8
CONFIG = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 8, "swa_from": 2, "cos": False, "adamw": False, "drop": False}
SEED, HIDDEN = 0, 128
BOOT, BOOT_SEED, BOOT_CHUNK = 2000, 20261007, 200
REFS = ("rrf", "twin0", "gnn0")
METRICS = ("R@5", "FC@5", "hit@1")
CARVE_KEY = {"select": "mselect", "s1sel": "dselect", "s1eval": "eval"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest(ids):
    return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()


def candidate_names():
    return [f"{v}@{s}" for v in VARIANTS for s in [f"ep{e}" for e in range(EPOCHS)] + ["swa"]]


def read_carves(train):
    return [(d, "select") for d in train] + [(d, "s1sel") for d in train] + [(d, "s1eval") for d in EVAL_ORDER]


def metrics_of(top, hit, gt):
    """(R@5, FC@5, hit@1) per question from the read's counts, lean_mlp.row_metrics' values (0 without gold)."""
    top, hit, gt = np.asarray(top, np.float64), np.asarray(hit, np.float64), np.asarray(gt, np.int64)
    out = np.zeros((gt.size, 3))
    ok = gt > 0
    out[ok, 0] = top[ok] / gt[ok]
    out[ok, 1] = (top[ok] == gt[ok]).astype(np.float64)
    out[ok, 2] = hit[ok]
    return out


def first_best(scores):
    """Index of the highest score; a tie goes to the first (fit8: `if score > best`)."""
    best, at = -1.0, None
    for k, s in enumerate(scores):
        if s > best:
            best, at = s, k
    return at


def boot_diff(a, b, n=BOOT, seed=BOOT_SEED, chunk=BOOT_CHUNK):
    """lean_mlp.boot_diff(a, b, default_rng(seed)), its resample means taken in chunks of resamples (same draws)."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, d.shape[0], size=(n, d.shape[0]))
    bs = np.empty((n, d.shape[1]))
    for s in range(0, n, chunk):
        bs[s:s + chunk] = d[idx[s:s + chunk]].mean(1)
    return [[float(d.mean(0)[j]), [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]]
            for j in range(d.shape[1])]


def label_of(same, ci):
    if same:
        return "SAME"
    lo, hi = ci
    return "ABOVE" if lo > 0 else "BELOW" if hi < 0 else "AT"


def verdict_of(labels):
    if any(x == "BELOW" for x in labels):
        return "NOT_ADOPTED"
    if any(x == "ABOVE" for x in labels):
        return "ADOPT"
    return "NO_EFFECT"


def rankdata(x):
    """Average ranks (1-based), ties sharing their mean rank."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    r = np.empty(x.size)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ── loading, behind the gates ────────────────────────────────────────────────


def load_carves_record():
    raw = CARVES.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != CARVES_SHA256:
        raise SystemExit(f"{CARVES}: sha256 {got} is not the declared {CARVES_SHA256}")
    return json.loads(raw.decode("utf-8"))


class Fit:
    """One fit's records and read arrays, after the gates: per carve the (K, Q, 3) metrics, ids and gold totals."""

    def __init__(self, name, root, carves):
        self.name, fdir = name, root / name
        need = [fdir / x for x in ("train.json", "read.json", "check.json", "models.pt")]
        miss = [str(p) for p in need if not p.exists()]
        if miss:
            raise SystemExit(f"{name}: missing {miss}")
        self.train = json.loads((fdir / "train.json").read_text(encoding="utf-8"))
        self.read = json.loads((fdir / "read.json").read_text(encoding="utf-8"))
        self.check = json.loads((fdir / "check.json").read_text(encoding="utf-8"))
        cands = candidate_names()
        want_train = [{"dataset": d, "carve": FIT_CARVE[d]} for d in FITS[name]]
        got_train = [{"dataset": t["dataset"], "carve": t["carve"]} for t in self.train["train"]]
        if got_train != want_train:
            raise SystemExit(f"{name}: trains on {got_train}, declared {want_train}")
        for t in self.train["train"]:
            ent = carves["per_dataset"][t["dataset"]]["fit"]
            if t["carve_ids_sha256"] != ent["sha256"]:
                raise SystemExit(f"{name}: {t['dataset']}/{t['carve']} is not carves.json's fit carve")
        if (self.train["config"] != CONFIG or self.train["seed"] != SEED or self.train["hidden"] != HIDDEN
                or self.train["candidates"] != cands or self.read["candidates"] != cands):
            raise SystemExit(f"{name}: config, seed, hidden or candidates are not the declared ones")
        msha = sha_file(fdir / "models.pt")
        if not (msha == self.train["models_sha256"] == self.read["models_sha256"]):
            raise SystemExit(f"{name}: models.pt's sha256 is not the train and read records'")
        want = [f"{d}={cv}" for d, cv in read_carves(FITS[name])]
        if sorted(self.read["carves"]) != sorted(want):
            raise SystemExit(f"{name}: reads {sorted(self.read['carves'])}, declared {sorted(want)}")
        if self.check.get("verdict") != "PASS" or sorted(self.check["carves"]) != sorted(want) or any(
                c.get("verdict") != "PASS" for c in self.check["carves"].values()):
            raise SystemExit(f"{name}: the CPU check is not PASS on every read carve")
        self.m, self.ids, self.gt, self.refs, self.sha = {}, {}, {}, {}, {}
        for key in want:
            ds, cv = key.split("=")
            ent = self.read["carves"][key]
            p = fdir / "reads" / ent["file"]
            if sha_file(p) != ent["sha256"]:
                raise SystemExit(f"{name}: {p} does not hash to the read record's sha256")
            z = np.load(p)
            ids = [str(x) for x in z["ids"]]
            want_sha = carves["per_dataset"][ds][CARVE_KEY[cv]]["sha256"]
            if digest(ids) != want_sha or ent["carve_ids_sha256"] != want_sha:
                raise SystemExit(f"{name}: {key}'s ids are not carves.json's {cv} carve")
            if [str(x) for x in z["candidates"]] != cands or [str(x) for x in z["refs"]] != list(REFS):
                raise SystemExit(f"{name}: {key} lists other candidates or references")
            gt = z["gold_total"].astype(np.int64)
            self.m[(ds, cv)] = np.stack([metrics_of(z["top"][k], z["hit"][k], gt) for k in range(len(cands))])
            self.refs[(ds, cv)] = np.stack([metrics_of(z["ref_top"][j], z["ref_hit"][j], gt) for j in range(len(REFS))])
            self.ids[(ds, cv)], self.gt[(ds, cv)], self.sha[key] = ids, gt, ent["sha256"]
        log(f"{name}: gates pass ({len(want)} read carves, models {msha[:12]})")

    def select_scores(self, cv):
        """Per candidate, (R@5 + FC@5) / 2 over the pooled select rows of carve kind cv, as fit8 (lean_mlp.quality)."""
        out = []
        for k in range(self.m[(FITS[self.name][0], cv)].shape[0]):
            q = np.concatenate([self.m[(d, cv)][k] for d in FITS[self.name]]).mean(0)
            out.append(0.5 * (float(q[0]) + float(q[1])))
        return out

    def own_select_scores(self, ds, cv):
        out = []
        for k in range(self.m[(ds, cv)].shape[0]):
            q = self.m[(ds, cv)][k].mean(0)
            out.append(0.5 * (float(q[0]) + float(q[1])))
        return out


def eval_rows(fit, ds):
    ok = fit.gt[(ds, "s1eval")] > 0
    return fit.m[(ds, "s1eval")][:, ok], fit.refs[(ds, "s1eval")][:, ok], ok


def compare(a, b, same):
    if same:
        return [[0.0, [0.0, 0.0]] for _ in METRICS]
    return boot_diff(a, b)


def strata_rows(carves, host):
    """step1_carves.stratum of every eval question of the five training datasets, from the eval population's rows."""
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "outputs" / "mp_unified"))
    placement = None
    if host:
        import lean_host as LH
        placement = LH.substitute()
    import look_x_six as LX
    import step1_carves as SC
    _cfg, cfg_m3b, cfg_h = LX.V2.load_configs()
    m3b_compile = LX.V2.M3B_RUN.load_script("m3b_compile")
    m3a, canonical, served, freeze = m3b_compile.open_package(cfg_m3b)
    if freeze["RECORD_SHA256"] != carves["freeze_RECORD_SHA256"]:
        raise SystemExit(f"the package's freeze {freeze['RECORD_SHA256']} is not the one carves.json read")
    out = {}
    for name in TRAIN_ORDER:
        ent = carves["per_dataset"][name]["eval"]
        ds = canonical.Dataset(name, root=str(served))
        rows = {r["query_id"]: r for r in m3a.population_rows(ds, ent["split"], cfg_h)[1]}
        out[name] = {q: SC.stratum(name, rows[q]) for q in ent["ids"]}
        log(f"strata {name}: {dict(sorted(__import__('collections').Counter(out[name].values()).items()))}")
    return out, placement, freeze["RECORD_SHA256"]


# ── the grade ────────────────────────────────────────────────────────────────


def means(m):
    return [round(float(x), 6) for x in m.mean(0)] if m.shape[0] else [None, None, None]


def grade(fits_root, out_dir, host):
    t0 = time.time()
    carves = load_carves_record()
    fits = {f: Fit(f, fits_root, carves) for f in FITS}
    strata, placement, freeze = strata_rows(carves, host)
    cands = candidate_names()
    rec = {"declared_in": "docs/STEP1_MATCHED_SELECTION.md", "script": "scripts/step1_grade.py",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
           "carves_sha256": CARVES_SHA256, "freeze_RECORD_SHA256": freeze, "placement": placement,
           "bootstrap": {"resamples": BOOT, "seed": BOOT_SEED, "interval": [2.5, 97.5], "per": "read and stratum"},
           "fits": {}, "picks": {}, "reads": [], "diagnostics": {"spearman": {}, "oracle": {}, "strata": [],
                                                                  "per_variant_picks": {}, "candidates_eval_r5": {}}}
    for f, fit in fits.items():
        rec["fits"][f] = {"models_sha256": fit.train["models_sha256"], "read_sha256": fit.sha,
                          "check": {"verdict": fit.check["verdict"],
                                    "max_abs_diff": max(c["max_abs_diff"] for c in fit.check["carves"].values())},
                          "train_seconds": fit.train.get("seconds"), "dead": fit.train.get("dead")}
        sc = {"M": fit.select_scores("select"), "D": fit.select_scores("s1sel")}
        pk = {a: first_best(sc[a]) for a in sc}
        rec["picks"][f] = {a: {"candidate": cands[pk[a]], "score": sc[a][pk[a]], "scores": sc[a]} for a in sc}
        rec["diagnostics"]["per_variant_picks"][f] = {
            v: {a: cands[9 * i + first_best(sc[a][9 * i:9 * i + 9])] for a in sc} for i, v in enumerate(VARIANTS)}
        same = pk["M"] == pk["D"]
        sp = {}
        for ds in EVAL_ORDER:
            m, refs, ok = eval_rows(fit, ds)
            ev_r5 = [float(m[k][:, 0].mean()) for k in range(len(cands))]
            rec["diagnostics"]["candidates_eval_r5"].setdefault(f, {})[ds] = ev_r5
            orc = first_best(ev_r5)
            rec["diagnostics"]["oracle"].setdefault(f, {})[ds] = {
                "candidate": cands[orc], "R@5": ev_r5[orc], "M_pick_R@5": ev_r5[pk["M"]], "D_pick_R@5": ev_r5[pk["D"]],
                "note": "the eval-best candidate is an oracle, never a result"}
            sp[ds] = {"pooled_M": spearman(sc["M"], ev_r5), "pooled_D": spearman(sc["D"], ev_r5)}
            if ds in FITS[f]:
                sp[ds]["own_M"] = spearman(fit.own_select_scores(ds, "select"), ev_r5)
                sp[ds]["own_D"] = spearman(fit.own_select_scores(ds, "s1sel"), ev_r5)
            primary = f == "J5" or ds == f[2:]          # J5 on all six reads; a leave-out fit on its held-out dataset
            role = "in-domain" if ds in FITS[f] else "zero-shot"
            cmp_ = compare(m[pk["D"]], m[pk["M"]], same)
            lab = label_of(same, cmp_[0][1])
            row = {"fit": f, "dataset": ds, "role": role, "primary": bool(primary), "questions": int(ok.sum()),
                   "without_gold": int((~ok).sum()), "M_pick": cands[pk["M"]], "D_pick": cands[pk["D"]],
                   "M": means(m[pk["M"]]), "D": means(m[pk["D"]]),
                   "diff": {nm: {"mean": c[0], "ci": c[1]} for nm, c in zip(METRICS, cmp_)}, "label": lab,
                   "refs": {nm: means(refs[j]) for j, nm in enumerate(REFS)}}
            rec["reads"].append(row)
            if primary and ds in strata:
                ids = np.asarray(fit.ids[(ds, "s1eval")])[ok]
                st = np.asarray([strata[ds][q] for q in ids])
                for s in sorted(set(st)):
                    sel = st == s
                    c2 = compare(m[pk["D"]][sel], m[pk["M"]][sel], same)
                    rec["diagnostics"]["strata"].append({
                        "fit": f, "dataset": ds, "stratum": str(s), "questions": int(sel.sum()),
                        "M": means(m[pk["M"]][sel]), "D": means(m[pk["D"]][sel]),
                        "diff": {nm: {"mean": c[0], "ci": c[1]} for nm, c in zip(METRICS, c2)},
                        "label": label_of(same, c2[0][1]),
                        "refs": {nm: means(refs[j][sel]) for j, nm in enumerate(REFS)}})
            log(f"{f} {ds} ({role}{', primary' if primary else ''}): M {cands[pk['M']]} {row['M'][0]:.4f}  "
                f"D {cands[pk['D']]} {row['D'][0]:.4f}  diff {cmp_[0][0]:+.4f} [{cmp_[0][1][0]:+.4f}, {cmp_[0][1][1]:+.4f}] {lab}")
        rec["diagnostics"]["spearman"][f] = sp
    prim = [r for r in rec["reads"] if r["primary"]]
    if len(prim) != 11:
        raise SystemExit(f"{len(prim)} primary reads, declared 11")
    rec["verdict"] = verdict_of([r["label"] for r in prim])
    rec["primary_labels"] = {f"{r['fit']}:{r['dataset']}": r["label"] for r in prim}
    rec["seconds"] = time.time() - t0
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "grade.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, out_dir / "grade.json")
    md = render(rec)
    tmp = out_dir / "grade.md.tmp"
    tmp.write_text(md, encoding="utf-8")
    os.replace(tmp, out_dir / "grade.md")
    log(f"verdict {rec['verdict']}: {rec['primary_labels']} ({rec['seconds']:.0f}s); {out_dir / 'grade.json'}")
    return 0


def fmt_ci(d):
    return f"{d['mean']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"


def render(rec):
    L = ["# Step 1 grade (seed 0)", "",
         f"Verdict over the eleven primary reads: **{rec['verdict']}**. Development numbers; the paper's numbers come "
         "from one declared confirmation run.", "",
         "## Picks", "", "| fit | M-pick (score) | D-pick (score) |", "| --- | --- | --- |"]
    for f, p in rec["picks"].items():
        L.append(f"| {f} | {p['M']['candidate']} ({p['M']['score']:.4f}) | {p['D']['candidate']} ({p['D']['score']:.4f}) |")
    for title, prim in (("Primary reads (graded on R@5)", True), ("Secondary reads (not graded)", False)):
        L += ["", f"## {title}", "",
              "| fit | read | role | R@5 M | R@5 D | D - M R@5 [95%] | label | D - M FC@5 [95%] | D - M hit@1 [95%] | twin0 R@5 | gnn0 R@5 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in rec["reads"]:
            if r["primary"] != prim:
                continue
            L.append(f"| {r['fit']} | {r['dataset']} | {r['role']} | {r['M'][0]:.4f} | {r['D'][0]:.4f} | "
                     f"{fmt_ci(r['diff']['R@5'])} | {r['label']} | {fmt_ci(r['diff']['FC@5'])} | "
                     f"{fmt_ci(r['diff']['hit@1'])} | {r['refs']['twin0'][0]:.4f} | {r['refs']['gnn0'][0]:.4f} |")
    L += ["", "## Primary reads by stratum (diagnostic)", "",
          "| fit | read | stratum | questions | R@5 M | R@5 D | D - M R@5 [95%] | label |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in rec["diagnostics"]["strata"]:
        L.append(f"| {s['fit']} | {s['dataset']} | {s['stratum']} | {s['questions']} | {s['M'][0]:.4f} | {s['D'][0]:.4f} | "
                 f"{fmt_ci(s['diff']['R@5'])} | {s['label']} |")
    L += ["", "## Which select carve predicts the eval (Spearman over the 36 candidates; diagnostic)", "",
          "| fit | read | pooled M | pooled D | own M | own D |", "| --- | --- | --- | --- | --- | --- |"]

    def f3(x):
        return "-" if x is None else f"{x:+.3f}"
    for f, sp in rec["diagnostics"]["spearman"].items():
        for ds, v in sp.items():
            L.append(f"| {f} | {ds} | {f3(v['pooled_M'])} | {f3(v['pooled_D'])} | {f3(v.get('own_M'))} | {f3(v.get('own_D'))} |")
    L += ["", "## Eval-best candidate (an oracle, never a result)", "",
          "| fit | read | eval-best | R@5 | M-pick R@5 | D-pick R@5 |", "| --- | --- | --- | --- | --- | --- |"]
    for f, o in rec["diagnostics"]["oracle"].items():
        for ds, v in o.items():
            L.append(f"| {f} | {ds} | {v['candidate']} | {v['R@5']:.4f} | {v['M_pick_R@5']:.4f} | {v['D_pick_R@5']:.4f} |")
    L += ["", "## Per-variant picks (state only; not graded)", "", "| fit | variant | M | D |", "| --- | --- | --- | --- |"]
    for f, pv in rec["diagnostics"]["per_variant_picks"].items():
        for v, x in pv.items():
            L.append(f"| {f} | {v} | {x['M']} | {x['D']} |")
    L += ["", f"carves.json {rec['carves_sha256'][:12]}..., freeze {str(rec['freeze_RECORD_SHA256'])[:12]}..., "
          f"script {rec['script_sha256'][:12]}..., {rec['utc']}", ""]
    return "\n".join(L)


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    ok = []
    assert first_best([0.1, 0.3, 0.3, 0.2]) == 1 and first_best([0.5]) == 0
    ok.append("ties go to the first candidate")
    assert [label_of(True, [0, 0]), label_of(False, [0.001, 0.01]), label_of(False, [-0.02, -0.001]),
            label_of(False, [-0.01, 0.01]), label_of(False, [0.0, 0.01]), label_of(False, [-0.01, 0.0])] == \
        ["SAME", "ABOVE", "BELOW", "AT", "AT", "AT"]
    assert verdict_of(["SAME", "AT", "ABOVE"]) == "ADOPT" and verdict_of(["ABOVE", "BELOW"]) == "NOT_ADOPTED"
    assert verdict_of(["SAME", "AT", "AT"]) == "NO_EFFECT"
    ok.append("labels and verdict")
    rng = np.random.default_rng(3)
    a, b = rng.random((517, 3)), rng.random((517, 3))
    want = None
    sys.path.insert(0, str(ROOT / "outputs" / "mp_unified"))
    try:
        import lean_mlp as LM
        want = LM.boot_diff(a, b, np.random.default_rng(BOOT_SEED), n=BOOT)   # lean_mlp's default is 1,000
    except Exception as e:  # noqa: BLE001
        log(f"lean_mlp not importable here ({e!r}); the bootstrap is checked against its formula")
    if want is None:
        r2 = np.random.default_rng(BOOT_SEED)
        d = a - b
        bs = d[r2.integers(0, d.shape[0], size=(BOOT, d.shape[0]))].mean(1)
        want = [[float(d.mean(0)[j]), [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]]
                for j in range(3)]
    assert boot_diff(a, b) == want, (boot_diff(a, b), want)
    ok.append("the chunked bootstrap equals lean_mlp.boot_diff bit for bit")
    x = np.array([3.0, 1.0, 2.0, 2.0, 5.0])
    assert rankdata(x).tolist() == [4.0, 1.0, 2.5, 2.5, 5.0]
    y = rng.random(36)
    z = y ** 3 + 0.1 * rng.random(36)
    try:
        from scipy.stats import spearmanr
        assert abs(spearman(y, z) - float(spearmanr(y, z).correlation)) < 1e-12
        assert abs(spearman(x, -x) + 1.0) < 1e-12
        ok.append("spearman equals scipy's")
    except ImportError:
        ok.append("spearman (scipy absent)")
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    gt = np.array([2, 0, 3, 1])
    m = metrics_of([1, 0, 3, 1], [1, 0, 0, 1], gt)
    assert m.tolist() == [[0.5, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    ok.append("metrics from counts")
    c = candidate_names()
    assert len(c) == 36 and c[0] == "p@ep0" and c[8] == "p@swa" and c[9] == "pf@ep0" and c[-1] == "nf@swa"
    assert len(read_carves(FITS["J5"])) == 16 and len(read_carves(FITS["L-musique"])) == 14
    prim = [(f, d) for f in FITS for d in EVAL_ORDER if (f == "J5") or (d == f[2:])]
    assert len(prim) == 11, prim
    ok.append("candidates, read carves and the eleven primary reads as declared")
    try:
        import lean_gpu as LG
        assert LG.FITS == FITS and LG.TRAIN_ORDER == TRAIN_ORDER and LG.EVAL_ORDER == EVAL_ORDER
        assert LG.FIT_CARVE == FIT_CARVE and LG.candidate_names(list(VARIANTS)) == c
        assert all(LG.read_carves(t) == read_carves(t) for t in FITS.values())
        assert LG.L5.parse_configs("x=" + LG.CONFIG)["x"] == CONFIG and LG.SEED == SEED and LG.HIDDEN == HIDDEN
        assert [v[0] for v in LG.VARIANTS] == list(VARIANTS)
        ok.append("fits, carves, candidates and config equal lean_gpu's")
    except ImportError as e:
        log(f"lean_gpu not importable here ({e!r}); its constants are not compared")
    log("selftest: " + "; ".join(ok) + ". all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", action="store_true", help="the verified mirror in place of the package (lean_host)")
    ap.add_argument("--fits-root", default=str(STEP1 / "fits"))
    ap.add_argument("--out-dir", default=str(STEP1))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    return grade(Path(a.fits_root), Path(a.out_dir), a.host)


if __name__ == "__main__":
    sys.exit(main())
