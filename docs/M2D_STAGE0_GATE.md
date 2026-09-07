# M2D Stage-0 advance gate

**ADVANCE_TARGETED_M2D** — B_a_named_primitive_reorders holds.

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

1 of 5 primitives S4 is missing reach a majority on both failure cells. A pass by one of 5 is a selection over 5 and is reported as such.

| primitive | squad_clean/R1 | musique_clean/R1 | majority on both |
|---|---:|---:|---|
| cosine_qd | 0.2857 | 0.7067 | no |
| dot_qd_pct | 0.2853 | 0.7067 | no |
| mean_abs_diff | 0.3442 | 0.7253 | no |
| semantic_difference | 0.5949 | 0.7482 | yes |
| semantic_product | 0.3543 | 0.7067 | no |

Each figure is the share of the queries S4 gets wrong at rank 1 -- with a relevant candidate in the pool -- on which that primitive ALONE ranks a relevant candidate above S4's wrong top item. The bar is a majority (> 0.50) on both.

Measured but not admitted, because B is about a primitive S4 is MISSING and S4 already computes these: normalized_state_dot.

## C — does S3+S4 repair both blockers?

Holds: **no**. Diagnostic only — the arm runs two semantic models and can never be a result.

| cell | recall@1 | recall@5 | MRR |
|---|---:|---:|---:|
| squad_clean/R1 | +1.477 | +0.398 | +1.080 |
| musique_clean/R1 | +2.670 | -1.134 | +2.995 |

that the two representations carry different information, so an integrated single-model representation is worth trying. Never that two semantic models may be run at inference.
