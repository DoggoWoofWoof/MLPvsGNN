# M2D Stage 1 — report

**Verdict: STOP_S4_DEVELOPMENT** — case_3, applied by `scripts/m2d_stage1_gate.py`, which was committed at `6f429ee` before any of these eight fits existed.

Eight new fits: two arms (A1, A3-MINIMAL) at seed 0 on four cells. S3 and S4 are reused from M2B's immutable baseline table on the same held-out portion; nothing already fit was refit to produce a comparison row. Every artifact was verified on fetch against its held-out panel digest and its cell's shared structural inputs, and selected by commit `0e498cf0d392`. No test data was read: every fit reports `test_split_read: false`.

## Metrics (section 8)

Mandatory five, on the held-out panel. `reuse` rows are M2B's, not refit here.

### squad_clean/R1 (blocker)

| model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| S3 (reuse) | 0.732400 | 0.917706 | 0.967197 | 0.813783 | 0.967197 |
| S4 (reuse) | 0.684635 | 0.910224 | 0.963936 | 0.784993 | 0.963936 |
| A1 | 0.730481 | 0.913677 | 0.967006 | 0.811862 | 0.967006 |
| A3_MINIMAL | 0.726453 | 0.918473 | 0.967197 | 0.810455 | 0.967197 |

### musique_clean/R1 (blocker)

| model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| S3 (reuse) | 0.444166 | 0.773526 | 0.873275 | 0.928313 | 0.751568 |
| S4 (reuse) | 0.311167 | 0.750941 | 0.867629 | 0.771464 | 0.741531 |
| A1 | 0.377666 | 0.757215 | 0.868256 | 0.850137 | 0.744040 |
| A3_MINIMAL | 0.397114 | 0.767880 | 0.871393 | 0.873413 | 0.749059 |

### hotpotqa_clean/R1 (control)

| model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| S3 (reuse) | 0.430378 | 0.767501 | 0.882984 | 0.908679 | 0.778232 |
| S4 (reuse) | 0.380940 | 0.776955 | 0.888733 | 0.852953 | 0.788452 |
| A1 | 0.425907 | 0.790112 | 0.890010 | 0.905002 | 0.790240 |
| A3_MINIMAL | 0.423735 | 0.793562 | 0.890904 | 0.902905 | 0.792284 |

### metaqa/R1 (control)

| model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| S3 (reuse) | 0.056978 | 0.100193 | 0.138819 | 0.241195 | 0.058125 |
| S4 (reuse) | 0.067793 | 0.114514 | 0.149671 | 0.287126 | 0.061318 |
| A1 | 0.067093 | 0.113517 | 0.149191 | 0.284618 | 0.060807 |
| A3_MINIMAL | 0.065218 | 0.114108 | 0.149200 | 0.285507 | 0.060935 |

## The three per-blocker deltas (section 8)

Percentage points. The gate's blocker rule reads the `A3_MINIMAL − S3` column.

| blocker | metric | A1 − S4 | A3_MINIMAL − S4 | A3_MINIMAL − S3 |
|---|---|---:|---:|---:|
| squad_clean/R1 | recall@1 | +4.585 | +4.182 | -0.595 |
| squad_clean/R1 | recall@5 | +0.345 | +0.825 | +0.077 |
| squad_clean/R1 | recall@20 | +0.307 | +0.326 | +0.000 |
| squad_clean/R1 | mrr | +2.687 | +2.546 | -0.333 |
| squad_clean/R1 | full_coverage@20 | +0.307 | +0.326 | +0.000 |
| musique_clean/R1 | recall@1 | +6.650 | +8.595 | -4.705 |
| musique_clean/R1 | recall@5 | +0.627 | +1.694 | -0.565 |
| musique_clean/R1 | recall@20 | +0.063 | +0.376 | -0.188 |
| musique_clean/R1 | mrr | +7.867 | +10.195 | -5.490 |
| musique_clean/R1 | full_coverage@20 | +0.251 | +0.753 | -0.251 |

Controls, against native S4 — section 7's rule is no R@5 regression beyond 0.50pp.

| control | A1 R@5 − S4 | A3_MINIMAL R@5 − S4 |
|---|---:|---:|
| hotpotqa_clean/R1 | +1.316 | +1.661 |
| metaqa/R1 | -0.100 | -0.041 |

## Error-conditioned top-1 analysis (section 8)

Conditioned on the queries native S4 got wrong at rank 1, measured against a re-score of M2B's own S4 checkpoint in the same container on the same panel — not against the filed row, so the pairing is query-by-query.

| cell | arm | S4 top-1 errors | corrected | fraction of errors | newly broken | fraction of correct | net |
|---|---|---:|---:|---:|---:|---:|---:|
| squad_clean/R1 | A1 | 1,644 | 404 | 0.2457 | 165 | 0.0462 | +239 |
| squad_clean/R1 | A3_MINIMAL | 1,644 | 420 | 0.2555 | 202 | 0.0566 | +218 |
| musique_clean/R1 | A1 | 301 | 140 | 0.4651 | 34 | 0.0685 | +106 |
| musique_clean/R1 | A3_MINIMAL | 301 | 160 | 0.5316 | 23 | 0.0464 | +137 |
| hotpotqa_clean/R1 | A1 | 932 | 478 | 0.5129 | 126 | 0.0423 | +352 |
| hotpotqa_clean/R1 | A3_MINIMAL | 932 | 486 | 0.5215 | 151 | 0.0506 | +335 |
| metaqa/R1 | A1 | 6,079 | 175 | 0.0288 | 203 | 0.1161 | -28 |
| metaqa/R1 | A3_MINIMAL | 6,079 | 180 | 0.0296 | 194 | 0.1109 | -14 |

## Latency (section 10)

Full uncached inference: the structural columns, all of native S4's projection work, the arm's added column and its reduction, and the widened scorer. Nothing precomputed between queries — every fit reports `cached_or_precomputed_semantic_difference: false`. Section 10's standing requirement is that a repaired S4 keep p95 below S3's.

| cell | model | p50 ms | p95 ms | p99 ms | p95 below S3 |
|---|---|---:|---:|---:|---|
| squad_clean/R1 | S3 (reuse) | 1.8246 | 1.8387 | 1.8580 | — |
| squad_clean/R1 | S4 (reuse) | 1.0415 | 1.0505 | 1.0672 | yes |
| squad_clean/R1 | A1 | 1.2950 | 1.4202 | 1.6443 | yes |
| squad_clean/R1 | A3_MINIMAL | 1.2747 | 1.2887 | 1.2981 | yes |
| musique_clean/R1 | S3 (reuse) | 1.8258 | 1.8428 | 1.8501 | — |
| musique_clean/R1 | S4 (reuse) | 1.0300 | 1.0554 | 1.0667 | yes |
| musique_clean/R1 | A1 | 1.3765 | 1.3896 | 1.4334 | yes |
| musique_clean/R1 | A3_MINIMAL | 1.2998 | 1.3125 | 1.3229 | yes |
| hotpotqa_clean/R1 | S3 (reuse) | 1.9328 | 1.9456 | 1.9511 | — |
| hotpotqa_clean/R1 | S4 (reuse) | 1.0627 | 1.0714 | 1.0742 | yes |
| hotpotqa_clean/R1 | A1 | 1.3742 | 1.3854 | 1.3916 | yes |
| hotpotqa_clean/R1 | A3_MINIMAL | 1.2805 | 1.2932 | 1.3056 | yes |
| metaqa/R1 | S3 (reuse) | 1.8482 | 1.8645 | 1.8886 | — |
| metaqa/R1 | S4 (reuse) | 1.0525 | 1.0625 | 1.0717 | yes |
| metaqa/R1 | A1 | 1.3485 | 1.3680 | 1.3858 | yes |
| metaqa/R1 | A3_MINIMAL | 1.2917 | 1.3064 | 1.3208 | yes |

Added semantic time and scorer time, same container, against a native S4 re-score.

| cell | arm | semantic p95 | scorer p95 | native S4 p95 | added p95 | increase |
|---|---|---:|---:|---:|---:|---:|
| squad_clean/R1 | A1 | 0.8039 | 0.3153 | 0.6610 | 0.5154 | 114.85% |
| squad_clean/R1 | A3_MINIMAL | 0.6685 | 0.2903 | 0.6610 | 0.3800 | 94.96% |
| musique_clean/R1 | A1 | 0.7893 | 0.3048 | 0.7060 | 0.4731 | 96.82% |
| musique_clean/R1 | A3_MINIMAL | 0.7108 | 0.3005 | 0.7060 | 0.3946 | 85.90% |
| hotpotqa_clean/R1 | A1 | 0.7860 | 0.2998 | 0.6613 | 0.4952 | 109.50% |
| hotpotqa_clean/R1 | A3_MINIMAL | 0.6997 | 0.2990 | 0.6613 | 0.4089 | 95.56% |
| metaqa/R1 | A1 | 0.7786 | 0.2999 | 0.7001 | 0.4683 | 95.40% |
| metaqa/R1 | A3_MINIMAL | 0.6963 | 0.2976 | 0.7001 | 0.3860 | 86.60% |

> The per-component figures come from a hooked pass whose synchronisations cost time the clean pass does not pay. They do not sum to the total and are not a partition of it; the p95 column above is the number section 10 asks for.

## Parameters (section 11)

At the live structural width of 9, which every one of the eight artifacts reports and which M2B's filed fits pin independently.

| model | semantic | scorer | total | added semantic | added total vs S4 |
|---|---:|---:|---:|---:|---:|
| S3 (reuse) | 3,072 | 513 | 3,585 | — | -201,632 |
| S4 (reuse) | 196,608 | 8,609 | 205,217 | — | +0 |
| A1 | 196,608 | 8,673 | 205,281 | 0 | +64 |
| A3_MINIMAL | 198,144 | 8,641 | 206,785 | 1,536 | +1,568 |

A3-MINIMAL adds exactly the 1,536 semantic parameters section 3 authorised, and nothing else: no `semantic_product`, no `dot_qd_pct`, no structural feature.

## Training time and measured cost (sections 6 and 15)

| cell | arm | training s | peak train VRAM MB | peak inference VRAM MB | peak RSS MB |
|---|---|---:|---:|---:|---:|
| squad_clean/R1 | A1 | 153.31 | 1008.2 | 930.2 | 6497.9 |
| squad_clean/R1 | A3_MINIMAL | 156.43 | 1042.0 | 927.9 | 6511.0 |
| musique_clean/R1 | A1 | 24.29 | 330.6 | 250.1 | 4676.6 |
| musique_clean/R1 | A3_MINIMAL | 24.48 | 366.0 | 247.9 | 4684.5 |
| hotpotqa_clean/R1 | A1 | 119.46 | 3687.4 | 3602.7 | 8975.6 |
| hotpotqa_clean/R1 | A3_MINIMAL | 121.30 | 3723.6 | 3600.4 | 8989.7 |
| metaqa/R1 | A1 | 135.19 | 2767.5 | 2678.1 | 10413.6 |
| metaqa/R1 | A3_MINIMAL | 134.26 | 2804.8 | 2675.5 | 10433.4 |

Total training time across the eight fits: **868.72 s**.

Measured cost re-prices the compute record's own line items with the seconds that happened, so the two are comparable line by line.

| cell | predicted work s | measured work s | ratio | train | scoring | benchmarks |
|---|---:|---:|---:|---:|---:|---:|
| squad_clean/R1 | 409.6 | 331.5 | 0.81 | 309.7 | 18.8 | 2.9 |
| musique_clean/R1 | 66.9 | 54.7 | 0.82 | 48.8 | 3.0 | 3.0 |
| hotpotqa_clean/R1 | 326.1 | 258.3 | 0.79 | 240.8 | 14.6 | 3.0 |
| metaqa/R1 | 359.5 | 301.3 | 0.84 | 269.5 | 28.9 | 3.0 |
| **total** | **1162.0** | **945.9** | **0.81** | | | |

At the record's own divisor (0.4) and rate ($2.2408/h): **$1.4719** compute plus $0.1896 measured container overhead = **$1.66 measured**, against $2.00 predicted and a $5.00 filed ceiling. Within ceiling: true.

The prediction came in high, in the direction the record said it would: its multiplier was measured on a host CPU, and the record stated in advance that a host ratio should be expected to overstate the container ratio because a compute-bound GEMM and a memory-bound elementwise pass do not scale alike.

## Verdict

**STOP_S4_DEVELOPMENT**

Reading, filed in advance: **case_3** (both fail) — Condition B's sole-ranker evidence does not transfer into the integrated scorer.

Action: STOP_S4_DEVELOPMENT. S3 is retained and M3 proceeds.

Neither arm passed the filed rule, so S3 is retained and S4 development stops. One distinction the case label does not carry, recorded here because it changes what a reader should conclude:

* **A1** — squad_clean/R1 PASS, musique_clean/R1 FAIL.
* **A3_MINIMAL** — squad_clean/R1 PASS, musique_clean/R1 RESOLVABLE.

A3-MINIMAL's shortfall on musique_clean/R1 is RESOLVABLE, not FAIL: it misses the 0.50pp bound against S3 by less than that cell's own measured S4 seed spread, so seed 0 provably cannot settle it. RESOLVABLE is a third outcome, not a softer failure, and it never advances the phase on its own. The declaration's mechanism for that state is seeds 1 and 2 on the blockers, which section 14 authorises only on a pass; the gate reports it could change the decision for A3-MINIMAL and authorises it: false.

So the honest summary is not that the added column did nothing. It corrected more top-1 errors than it broke on both blockers, improved R@1 and MRR against native S4 on every cell but metaqa, held both controls, and stayed under S3's p95 everywhere. It did not reach S3's R@5 on musique_clean/R1 by a margin seed 0 cannot resolve. Under the rule filed before the fits, that is not a repair.

## STOP_FOR_REVIEW

