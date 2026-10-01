# MP-Approx level 13: a typed-walk model without message passing, with its bucket-1 walk types dropped

Declaration: `configs/mp_approx_l13.yaml`. The scripts are `scripts/mp_approx_l13.py` (population, dev scoring pass, check) and `scripts/mp_approx_l13_fit.py` (deploy views, fits, read, doc, file). The record is `outputs/mp_approx_l13/record.json`.

## What was measured

On metaqa, level 12's non-message-passing typed-walk model was fitted on level 12's train-split carves, as level 12 fitted it, and read once on 2,000 fresh V2_GATE dev rows (hop 2 and hop 3; no hop-1 row was left), disjoint from levels 0 to 12. Every unit is scored twice. dsh is level 11's sharpened mixture, as in level 12. b1d is the same score after every bucket-1 walk type's probability is moved to the null type, before the tempering, with its own beta, kappa and eta chosen on the select carve. It reads the posterior and the walk types' bucket digit only, and no gold. Its form was fixed before this file, from a design look on level 12's dev rows. TW-1x fits on the GNN's metaqa fit carve and chooses its settings on the GNN's select carve, so it is label-matched to the GNN. TW-4x adds level 12's x1 to x3 and fits on four times the labels, and the GNN is not refitted on them. rho is level 8's quantity: the share of the GNN's gain over its twin on recall@5, full_coverage@5 and hit@1, and rho_bar is its mean over the readable metrics.

## Reading

- The primary arm is TW-1x-b1d (label-matched, 5,883 fit / 1,482 inner rows), and its band is **L13_HIGH**: rho_bar 0.967 [0.911, 1.031].
- Readable metrics: recall@5, full_coverage@5, hit@1.
- Flags: none.
- The interpretation map entries that apply: l13_high, matched_below_gnn, b1d_adds, b1d_adds_4x, data_adds_4x, data_adds_4x_b1d.
- The first read arm with no readable metric below the GNN (reaches_gnn): TW-4x-dsh, at 23,538 fit rows.

## Arms

Intervals are 95% bootstrap intervals over 1,000 dev-row resamples; they do not resample the training carves or the fits. The gap to the GNN is the mean over seeds and dev rows of M(arm) - M(G).

| arm | labels | rho_bar | band | gap recall@5 | gap full_coverage@5 | gap hit@1 |
|---|---|---|---|---|---|---|
| TW-1x-dsh | 5,883 fit / 1,482 inner | 0.916 [0.860, 0.978] | L13_HIGH | -0.010 [-0.015, -0.004] BELOW_GNN | -0.013 [-0.020, -0.004] BELOW_GNN | 0.001 [-0.011, 0.013] |
| TW-1x-b1d | 5,883 fit / 1,482 inner | 0.967 [0.911, 1.031] | L13_HIGH | -0.006 [-0.011, -0.001] BELOW_GNN | -0.006 [-0.013, 0.002] | 0.006 [-0.005, 0.017] |
| TW-4x-dsh | 23,538 fit / 1,482 inner | 0.982 [0.927, 1.044] | L13_HIGH | -0.005 [-0.010, 0.001] | -0.007 [-0.014, 0.001] | 0.013 [0.001, 0.024] BEATS_GNN |
| TW-4x-b1d | 23,538 fit / 1,482 inner | 1.021 [0.963, 1.088] | L13_HIGH | -0.002 [-0.007, 0.004] | -0.001 [-0.009, 0.007] | 0.016 [0.006, 0.027] BEATS_GNN |
| NB-oracle | none (reference) | 1.114 [1.056, 1.184] | L13_ABOVE_GNN | 0.005 [0.000, 0.010] BEATS_GNN | 0.004 [-0.003, 0.012] | 0.036 [0.026, 0.048] BEATS_GNN |

## Contrasts

Paired differences of rho_bar on the same dev rows.

| contrast | of | paired difference |
|---|---|---|
| b1d_adds | rho_bar(TW-1x-b1d) - rho_bar(TW-1x-dsh) | 0.051 [0.022, 0.080] |
| b1d_adds_4x | rho_bar(TW-4x-b1d) - rho_bar(TW-4x-dsh) | 0.039 [0.013, 0.068] |
| data_4x | rho_bar(TW-4x-dsh) - rho_bar(TW-1x-dsh) | 0.066 [0.035, 0.095] |
| data_4x_b1d | rho_bar(TW-4x-b1d) - rho_bar(TW-1x-b1d) | 0.054 [0.026, 0.083] |
| ceiling_gap | rho_bar(NB-oracle) - rho_bar(TW-1x-b1d) | 0.147 [0.106, 0.190] |
| ceiling_gap_4x | rho_bar(NB-oracle) - rho_bar(TW-4x-b1d) | 0.093 [0.052, 0.135] |

## By hop

| arm | hop=2 (1000 rows) | hop=3 (1000 rows) |
|---|---|---|
| TW-1x-dsh | 0.974 [0.877, 1.082] L13_HIGH | 0.897 [0.824, 0.972] L13_HIGH |
| TW-1x-b1d | 1.062 [0.974, 1.165] L13_HIGH | 0.934 [0.864, 1.010] L13_HIGH |
| TW-4x-dsh | 1.042 [0.939, 1.152] L13_HIGH | 0.962 [0.893, 1.035] L13_HIGH |
| TW-4x-b1d | 1.084 [0.984, 1.195] L13_HIGH | 0.999 [0.930, 1.077] L13_HIGH |
| NB-oracle | 1.154 [1.053, 1.269] L13_ABOVE_GNN | 1.100 [1.030, 1.182] L13_ABOVE_GNN |

## Anchors (descriptive)

- TW-1x: argmax chain right on 0.876 of dev rows (hop=2 0.907, hop=3 0.846; genre-ending 0.610 of 339); b1d moved a mean 0.115 of the scored rows' posterior mass to the null type.
- TW-4x: argmax chain right on 0.890 of dev rows (hop=2 0.904, hop=3 0.877; genre-ending 0.599 of 339); b1d moved a mean 0.120 of the scored rows' posterior mass to the null type.
- Gap split (NB-oracle - TW-1x-b1d): argmax right on 5,258 (row, seed) pairs, 0.079 (hop 2 0.013, hop 3 0.066); wrong on 742, 0.068 (hop 2 0.011, hop 3 0.058).
- Gap split (NB-oracle - TW-4x-b1d): argmax right on 5,342 (row, seed) pairs, 0.066 (hop 2 0.013, hop 3 0.054); wrong on 658, 0.027 (hop 2 0.005, hop 3 0.022).
- beta per unit: TW-1x-dsh 1.5 x1, 2.0 x1, 3.0 x1; TW-1x-b1d 3.0 x1, 4.0 x2; TW-4x-dsh 2.0 x3; TW-4x-b1d 2.0 x2, 4.0 x1.
- Grid edges, TW-1x-dsh: kappa at an end 0, eta at an end 1 of 3 units.
- Grid edges, TW-1x-b1d: kappa at an end 0, eta at an end 1 of 3 units.
- Grid edges, TW-4x-dsh: kappa at an end 0, eta at an end 0 of 3 units.
- Grid edges, TW-4x-b1d: kappa at an end 0, eta at an end 0 of 3 units.

## Checks

- Dev scoring integrity: 0 mismatches against the stored per-query metrics on 2,000 dev rows.
- Repeat unit bit-identical: True.
- Units whose fit-set marginal log-likelihood fell between rounds: 0.
- Units whose kept theta sits at a clip bound: 0.

## What this does not say

Every arm is a measurement model on metaqa, fitted to metaqa train-split gold labels and read on V2_GATE dev rows. It is never a deployable or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature contract, M3, M4 or any selection. TW-1x is the only label-matched fit. TW-4x fits on more metaqa labels than the GNN had, which the GNN was not refitted on. This file's population holds hop-2 and hop-3 rows only, so its rho is not level 12's overall rho, and it says nothing about hop 1. It is also not the within-U_q oracle rho of levels 0 to 7.
