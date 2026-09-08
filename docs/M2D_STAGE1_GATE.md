# M2D Stage 1 — gate

Applied by `scripts/m2d_stage1_gate.py`, committed before any Stage-1 fit existed. Seed 0 only. Blockers against reused S3, controls against reused native S4, both from M2B's immutable baseline table on the same held-out portion. Nothing already fit was refit to produce a comparison row.

**Verdict: STOP_S4_DEVELOPMENT**

0 of 2 arms pass. A pass by one of 2 is a selection over 2 and is reported as such.

## A1

Passes: **no**. a blocker misses the 0.50pp bound by more than its own seed spread.

| blocker | R@5 vs S3 | band | outcome | R@1 vs S4 | MRR vs S4 |
|---|---:|---:|---|---:|---:|
| squad_clean/R1 | -0.403 | 0.345 | PASS | +4.585 | +2.687 |
| musique_clean/R1 | -1.631 | 0.816 | FAIL | +6.650 | +7.867 |

| control | R@5 vs S4 | outcome | R@1 vs S4 | MRR vs S4 |
|---|---:|---|---:|---:|
| hotpotqa_clean/R1 | +1.316 | PASS | +4.497 | +5.205 |
| metaqa/R1 | -0.100 | PASS | -0.070 | -0.251 |

| cell | arm p95 ms | S4 p95 | S3 p95 | below S3 | vs native S4 |
|---|---:|---:|---:|---|---:|
| squad_clean/R1 | 1.420 | 1.050 | 1.839 | yes | +35.2% |
| musique_clean/R1 | 1.390 | 1.055 | 1.843 | yes | +31.7% |
| hotpotqa_clean/R1 | 1.385 | 1.071 | 1.946 | yes | +29.3% |
| metaqa/R1 | 1.368 | 1.063 | 1.865 | yes | +28.8% |

| blocker | S4 top-1 errors | corrected | newly broken | net |
|---|---:|---:|---:|---:|
| squad_clean/R1 | 1,644 | 404 | 165 | +239 |
| musique_clean/R1 | 301 | 140 | 34 | +106 |

## A3_MINIMAL

Passes: **no**. a blocker misses the bound by less than its own seed spread, so seed 0 cannot settle it.

| blocker | R@5 vs S3 | band | outcome | R@1 vs S4 | MRR vs S4 |
|---|---:|---:|---|---:|---:|
| squad_clean/R1 | +0.077 | 0.345 | PASS | +4.182 | +2.546 |
| musique_clean/R1 | -0.565 | 0.816 | RESOLVABLE | +8.595 | +10.195 |

| control | R@5 vs S4 | outcome | R@1 vs S4 | MRR vs S4 |
|---|---:|---|---:|---:|
| hotpotqa_clean/R1 | +1.661 | PASS | +4.280 | +4.995 |
| metaqa/R1 | -0.041 | PASS | -0.258 | -0.162 |

| cell | arm p95 ms | S4 p95 | S3 p95 | below S3 | vs native S4 |
|---|---:|---:|---:|---|---:|
| squad_clean/R1 | 1.289 | 1.050 | 1.839 | yes | +22.7% |
| musique_clean/R1 | 1.313 | 1.055 | 1.843 | yes | +24.4% |
| hotpotqa_clean/R1 | 1.293 | 1.071 | 1.946 | yes | +20.7% |
| metaqa/R1 | 1.306 | 1.063 | 1.865 | yes | +23.0% |

| blocker | S4 top-1 errors | corrected | newly broken | net |
|---|---:|---:|---:|---:|
| squad_clean/R1 | 1,644 | 420 | 202 | +218 |
| musique_clean/R1 | 301 | 160 | 23 | +137 |

## The reading, filed in advance

**case_3** — both fail.

Condition B's sole-ranker evidence does not transfer into the integrated scorer.

Action: STOP_S4_DEVELOPMENT. S3 is retained and M3 proceeds.

## Extra seeds

Extra seeds are proposed only for an arm whose blocker shortfall is inside that cell's own seed spread. Everything else is already settled by seed 0 under the filed rule, and a seed that cannot change the decision is not run.

Could change the decision for: A3_MINIMAL. Authorised by this gate: False.

## STOP_FOR_REVIEW

