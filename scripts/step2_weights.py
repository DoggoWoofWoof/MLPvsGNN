"""Step 2 weights (docs/STEP2_EVAL_MIX_WEIGHTS.md, section 2): every fit question's stratum and loss weight, for the
five training datasets' fit carves (musique: F'). Deterministic. Reads the served package read-only (step1_carves.py's
loader) and outputs/step1/carves.json, and writes outputs/step2/weights.json. No score, model or result is read.

    python scripts/step2_weights.py            # writes outputs/step2/weights.json
    python scripts/step2_weights.py --check    # recomputes and compares with the filed weights.json

The rule:
  p_fit(t)   the fit carve's share of stratum t (step1_carves.stratum), as carves.json records it (mix.fit summed
             over o; four decimals)
  p_eval(t)  the eval population's share of t, as carves.json records it (mix.eval summed over o)
  r(t)       p_eval(t) / p_fit(t), clipped to [1/4, 4]
  w(q)       r(t(q)) / mean of r(t(q')) over the fit carve's questions: each dataset's mean weight is 1
Gates: carves.json's sha256 is the declared one and the package's freeze is the one it read; the fit carve's ids hash
to carves.json's fit sha256 (the M3B fit population, or musique's F' listed in carves.json); the strata recomputed
from the fit questions' rows give carves.json's fit shares to within 2e-4.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m3b_run import load_script, CONFIG, HEADROOM_CONFIG  # noqa: E402
import step1_carves as SC  # noqa: E402

CARVES = ROOT / "outputs" / "step1" / "carves.json"
CARVES_SHA256 = "53cfb89f1e41a76257113719639f96318d14279f86b50a970f9a0bbf6686d746"   # step 1's declaration, section 2
OUT = ROOT / "outputs" / "step2" / "weights.json"
TRAIN = ("metaqa", "squad", "musique", "hotpotqa", "2wiki")
CLIP = (0.25, 4.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shares(labels):
    c = collections.Counter(labels)
    n = sum(c.values())
    return {t: k / n for t, k in sorted(c.items())}


def summed(mix):
    out = collections.defaultdict(float)
    for cell, v in mix.items():
        out[cell.split("|", 1)[1]] += v
    return dict(sorted(out.items()))


def weights_of(fit_strata, p_fit, p_eval):
    r = {t: min(CLIP[1], max(CLIP[0], p_eval.get(t, 0.0) / p_fit[t])) for t in p_fit}
    mean_r = sum(r[t] for t in fit_strata) / len(fit_strata)
    w = {t: r[t] / mean_r for t in r}
    return r, mean_r, w


def build():
    raw = CARVES.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CARVES_SHA256:
        raise SystemExit(f"{CARVES} is not the declared carves.json")
    carves = json.loads(raw.decode("utf-8"))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg_h = yaml.safe_load(HEADROOM_CONFIG.read_text(encoding="utf-8"))
    m3b_compile = load_script("m3b_compile")
    m3a, canonical, served, freeze = m3b_compile.open_package(cfg)
    if freeze["RECORD_SHA256"] != carves["freeze_RECORD_SHA256"]:
        raise SystemExit(f"the package's freeze {freeze['RECORD_SHA256']} is not the one carves.json read")
    rec = {"declared_in": "docs/STEP2_EVAL_MIX_WEIGHTS.md", "script": "scripts/step2_weights.py",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
           "carves_sha256": CARVES_SHA256, "freeze_RECORD_SHA256": freeze["RECORD_SHA256"], "clip": list(CLIP),
           "per_dataset": {}}
    for name in TRAIN:
        t0 = time.time()
        ent = carves["per_dataset"][name]
        ds = canonical.Dataset(name, root=str(served))
        if name == "musique":
            fit_ids = list(ent["fit"]["ids"])
        else:
            positions = m3a.node_position_map(ds)
            fit_ids = list(m3b_compile.population(ds, name, "fit", cfg, cfg_h, m3a, positions).ids)
            del positions
        if SC.digest(fit_ids) != ent["fit"]["sha256"]:
            raise SystemExit(f"{name}: the fit carve's ids are not carves.json's")
        keep = set(fit_ids)
        row_of = {r["query_id"]: r for r in ds.queries("train") if r["query_id"] in keep}
        fit_strata = [SC.stratum(name, row_of[q]) for q in fit_ids]
        p_fit, p_eval = summed(ent["mix"]["fit"]), summed(ent["mix"]["eval"])
        got = shares(fit_strata)
        if set(got) != set(p_fit) or any(abs(got[t] - p_fit[t]) > 2e-4 for t in got):
            raise SystemExit(f"{name}: the fit questions' strata {got} are not carves.json's {p_fit}")
        r, mean_r, w = weights_of(fit_strata, p_fit, p_eval)
        wq = [w[t] for t in fit_strata]
        rec["per_dataset"][name] = {
            "fit_carve": ent["fit"]["name"], "n": len(fit_ids), "ids_sha256": ent["fit"]["sha256"],
            "p_fit": p_fit, "p_eval": p_eval, "fit_shares_recomputed": got, "ratio_clipped": r, "mean_ratio": mean_r,
            "weight_of_stratum": w, "weighted_share": {t: got[t] * w[t] for t in got}, "mean_weight": sum(wq) / len(wq),
            "ids": fit_ids, "strata": fit_strata, "weights": wq}
        log(f"{name}: {len(fit_ids)} fit questions; " + ", ".join(f"{t} {w[t]:.3f}" for t in w) + f" ({time.time() - t0:.0f}s)")
        del row_of
        gc.collect()
    return rec


def strip(rec):
    return {k: v for k, v in rec.items() if k not in ("utc", "seconds")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="recompute and compare with the filed weights.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    rec = build()
    if a.check:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        same = json.dumps(strip(old), sort_keys=True) == json.dumps(strip(rec), sort_keys=True)
        print("check:", "IDENTICAL" if same else "DIFFERS")
        return 0 if same else 1
    rec["utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, OUT)
    log(f"wrote {OUT} ({rec['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
