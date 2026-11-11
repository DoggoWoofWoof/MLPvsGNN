"""AP2 boundary check (post hoc; docs/AP2_RULE_NER_SLOT_ON_B1.md section 9). Decides nothing and is not part of the declared rule.

AP2's report (`zap2.py report`, filed first) gives each in-domain cell's word on b1n - b1w from `b1b.boot_ci` (2,000 resamples of the 1,000 questions, seed 0,
the interval rounded to four decimals). One cell, zg6r on 2Wiki, has its interval's top end at -0.0 after that rounding, so its word is SAME by the pinned code
and the verdict REPLACE. This script reads the same per-question means (the reads the report used) and records, for each of the six in-domain cells:

  * the mean difference and the seed-0 interval before the rounding (the same resampling as `b1b.boot_ci`, `np.quantile` on the resampled means);
  * the same interval for bootstrap seeds 0 to 19 (2,000 resamples each), the interval's top and bottom ends before the rounding, and the word each seed gives
    at the filed rounding (four decimals) and before it.

Run from the repository root:  python -I outputs/ap2/boundary_check.py   (needs the b1n reads fetched under outputs/b1n/fits/h and the report's inputs).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs" / "mp_unified"))
sys.argv = ["zap2.py", "report"]                       # zb1h's header reads sys.argv[1:2]
import numpy as np  # noqa: E402
import zap2 as Z  # noqa: E402
import b1b as BB  # noqa: E402

SEEDS = 20


def word(lo, hi):
    return "GREATER" if lo > 0 else "LESS" if hi < 0 else "SAME"


def main():
    Z.bind()
    rec = {"missing": []}
    cells = {}
    for ds in ("2wiki", "hotpotqa", "musique"):
        _s, means, _qs = Z.read_carve("n", ds, rec)
        for arm in Z.MODELS:
            d = means[(arm, "b1n", "in_domain")] - means[(arm, "b1w", "in_domain")]
            per_seed = []
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                m = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(BB.BOOT)])
                lo, hi = float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
                filed = [round(lo, 4), round(hi, 4)]
                if seed == BB.SEED:
                    assert filed == BB.boot_ci(d), "seed 0 differs from b1b.boot_ci"
                per_seed.append({"seed": seed, "lo": lo, "hi": hi, "word_filed_rounding": word(*filed), "word_unrounded": word(lo, hi)})
            count = {w: sum(1 for p in per_seed if p["word_filed_rounding"] == w) for w in ("GREATER", "SAME", "LESS")}
            count_u = {w: sum(1 for p in per_seed if p["word_unrounded"] == w) for w in ("GREATER", "SAME", "LESS")}
            cells[f"{ds}/{arm}"] = {"mean_diff": float(d.mean()), "seed0_ci_unrounded": [per_seed[0]["lo"], per_seed[0]["hi"]],
                                    "seed0_ci_filed": BB.boot_ci(d), "words_over_seeds_filed_rounding": count,
                                    "words_over_seeds_unrounded": count_u, "per_seed": per_seed}
            print(f"{ds:9s} {arm:7s} mean {d.mean():+.6f}  seed-0 unrounded [{per_seed[0]['lo']:+.6f}, {per_seed[0]['hi']:+.6f}]  "
                  f"words over {SEEDS} seeds: filed rounding {count}, unrounded {count_u}")
    if rec["missing"]:
        raise SystemExit(f"missing reads: {rec['missing']}")
    out = {"declared_in": "docs/AP2_RULE_NER_SLOT_ON_B1.md", "note": "post hoc, after AP2's report; decides nothing", "bootstrap_seeds": SEEDS,
           "resamples": BB.BOOT, "cells": cells, "zap2_sha256": Z.H.sha_src(Z.__file__), "utc": Z.H.utc()}
    Z.H.write_json(ROOT / "outputs" / "ap2" / "boundary_check.json", out)
    print("written outputs/ap2/boundary_check.json")


if __name__ == "__main__":
    main()
