# M2D Stage-0 advance gate

**STOP_PENDING_B** — neither A nor C holds, and B was not measured. A STOP that rests on an unmeasured condition is not a STOP -- measure B, then re-apply this gate unchanged.

every delta is arm minus Z0_S4 within one cell's own panel. Nothing here is compared against M2B's filed table, which reports the holdout portion and is a different panel.

## Panels

| cell | queries | commit |
|---|---:|---|
| hotpotqa_clean/R1 | 15,656 | `677b2fa7ec20` |
| metaqa/R1 | 31,310 | `677b2fa7ec20` |
| musique_clean/R1 | 3,190 | `677b2fa7ec20` |
| squad_clean/R1 | 20,850 | `677b2fa7ec20` |

## A — does a fixed fusion work?

0 of 3 eligible arms pass. A pass by one of 3 is a selection over 3 and is reported as such.

| arm | squad_clean/R1 (blocker) | musique_clean/R1 (blocker) | hotpotqa_clean/R1 (control) | metaqa/R1 (control) | passes |
|---|---:|---:|---:|---:|---|
| Z1_S4_DENSE | -2.561 | -3.932 | -6.771 | -23.652 | no |
| Z2_S4_SPLADE | -1.424 | -6.142 | -6.767 | -22.897 | no |
| Z3_S4_DENSE_SPLADE | -2.508 | -3.958 | -7.112 | -25.655 | no |

Every figure is recall@5 in percentage points against Z0_S4. An arm passes on a gain in both blockers, at least one reaching +0.25pp, and neither control down more than 0.50pp.

## B — does one named primitive reorder?

**UNMEASURED**, and recorded as such before the results existed rather than after: the Stage-0 probe emits rankings for four whole models and no ranking for any single primitive, so no evidence here bears on B either way.

What would settle it: rank each cell's frozen pool by each raw primitive alone -- cosine_qd, the raw query-document dot, its within-query percentile, mean_abs_diff -- and report, on the queries S4 gets top-1 wrong, the share where that primitive alone ranks a relevant candidate above S4's wrong top item.

## C — does S3+S4 repair both blockers?

Holds: **no**. Diagnostic only — the arm runs two semantic models and can never be a result.

| cell | recall@1 | recall@5 | MRR |
|---|---:|---:|---:|
| squad_clean/R1 | +1.477 | +0.398 | +1.080 |
| musique_clean/R1 | +2.670 | -1.134 | +2.995 |

that the two representations carry different information, so an integrated single-model representation is worth trying. Never that two semantic models may be run at inference.

