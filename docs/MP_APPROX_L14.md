# MP-Approx level 14: a typed-walk model without message passing, fitted on its bucket-0 walk types

Declaration: `configs/mp_approx_l14.yaml`. The scripts are `scripts/mp_approx_l14.py` (population, r's scoring pass, check) and `scripts/mp_approx_l14_fit.py` (deploy views, b0, fits, read, doc, file). The record is `outputs/mp_approx_l14/record.json`.

## What was measured

On metaqa, level 12's non-message-passing typed-walk model was fitted on level 12's train-split carves and read once on r: 11,920 metaqa train-split rows (3,480 of hop 1, 4,306 of hop 2, 4,134 of hop 3) that the twin, the GNN and every level never trained on, selected on or read. r is the union of the M3B fit stride's offsets 8 and 9, disjoint from level 12's nine carves, so every number here is a reading on train-split rows, not on dev or test rows. Each fit is read in two views. In the full view (every nb walk type, as at levels 12 and 13) its units are scored under dsh, level 11's sharpened mixture, and under b1d, level 13's move of the bucket-1 probability to the null type after the fit. In the b0 view every part's nb types are filtered to bucket 0 before the fit (the walks that start from the dense or the splade rank-1 node), so the model is fitted, its settings chosen and r scored on them only; its dsh score is b0. The filter reads the walk types' bucket digit only, and no gold. TW-1x fits on the GNN's metaqa fit carve and chooses its settings on the GNN's select carve, so it is label-matched to the GNN. TW-4x adds level 12's x1 to x3 and fits on four times the labels, and the GNN is not refitted on them. rho is level 8's quantity: the share of the GNN's gain over its twin on recall@5, full_coverage@5 and hit@1, and rho_bar is its mean over the readable metrics.

## Reading

- The primary arm is TW-1x-b0 (label-matched, b0 view, 5,883 fit / 1,482 inner rows), and its band on r is **L14_HIGH**: rho_bar 0.905 [0.879, 0.933].
- Readable metrics: recall@5, full_coverage@5, hit@1.
- Flags: none.
- The interpretation map entries that apply: l14_high, matched_below_gnn, b1d_adds, b1d_adds_4x, data_adds_4x, data_adds_4x_b1d, data_adds_4x_b0, b0_ceiling_binds.
- The first read arm with no readable metric below the GNN (reaches_gnn): none; every read arm stays below the GNN on a readable metric.

## Arms

Intervals are 95% bootstrap intervals over 1,000 resamples of r's rows; they do not resample the training carves or the fits. The gap to the GNN is the mean over seeds and rows of M(arm) - M(G).

| arm | view | labels | rho_bar | band | gap recall@5 | gap full_coverage@5 | gap hit@1 |
|---|---|---|---|---|---|---|---|
| TW-1x-dsh | full | 5,883 fit / 1,482 inner | 0.868 [0.842, 0.896] | L14_HIGH | -0.008 [-0.010, -0.006] BELOW_GNN | -0.012 [-0.015, -0.010] BELOW_GNN | -0.011 [-0.015, -0.006] BELOW_GNN |
| TW-1x-b1d | full | 5,883 fit / 1,482 inner | 0.895 [0.869, 0.922] | L14_HIGH | -0.008 [-0.010, -0.006] BELOW_GNN | -0.011 [-0.013, -0.008] BELOW_GNN | -0.005 [-0.009, -0.001] BELOW_GNN |
| TW-1x-b0 | b0 | 5,883 fit / 1,482 inner | 0.905 [0.879, 0.933] | L14_HIGH | -0.007 [-0.009, -0.005] BELOW_GNN | -0.008 [-0.011, -0.006] BELOW_GNN | -0.006 [-0.011, -0.002] BELOW_GNN |
| TW-4x-dsh | full | 23,538 fit / 1,482 inner | 0.910 [0.883, 0.937] | L14_HIGH | -0.006 [-0.008, -0.004] BELOW_GNN | -0.010 [-0.013, -0.007] BELOW_GNN | -0.004 [-0.009, 0.001] |
| TW-4x-b1d | full | 23,538 fit / 1,482 inner | 0.937 [0.912, 0.963] | L14_HIGH | -0.005 [-0.007, -0.003] BELOW_GNN | -0.008 [-0.010, -0.005] BELOW_GNN | 0.000 [-0.004, 0.004] |
| TW-4x-b0 | b0 | 23,538 fit / 1,482 inner | 0.947 [0.921, 0.974] | L14_HIGH | -0.005 [-0.007, -0.003] BELOW_GNN | -0.006 [-0.009, -0.004] BELOW_GNN | 0.002 [-0.001, 0.007] |
| NB-oracle | full | none (reference) | 1.081 [1.056, 1.107] | L14_ABOVE_GNN | 0.003 [0.001, 0.004] BEATS_GNN | 0.001 [-0.002, 0.003] | 0.024 [0.020, 0.028] BEATS_GNN |
| NB-oracle-b0 | b0 | none (reference) | 1.004 [0.978, 1.030] | L14_HIGH | -0.002 [-0.004, -0.000] BELOW_GNN | -0.003 [-0.005, 0.000] | 0.011 [0.007, 0.014] BEATS_GNN |

## Contrasts

Paired differences of rho_bar on the same rows of r.

| contrast | of | paired difference |
|---|---|---|
| b0_adds | rho_bar(TW-1x-b0) - rho_bar(TW-1x-b1d) | 0.011 [-0.003, 0.024] |
| b0_adds_4x | rho_bar(TW-4x-b0) - rho_bar(TW-4x-b1d) | 0.010 [-0.002, 0.022] |
| b1d_adds | rho_bar(TW-1x-b1d) - rho_bar(TW-1x-dsh) | 0.026 [0.012, 0.040] |
| b1d_adds_4x | rho_bar(TW-4x-b1d) - rho_bar(TW-4x-dsh) | 0.027 [0.013, 0.041] |
| data_4x | rho_bar(TW-4x-dsh) - rho_bar(TW-1x-dsh) | 0.041 [0.027, 0.056] |
| data_4x_b1d | rho_bar(TW-4x-b1d) - rho_bar(TW-1x-b1d) | 0.043 [0.030, 0.056] |
| data_4x_b0 | rho_bar(TW-4x-b0) - rho_bar(TW-1x-b0) | 0.042 [0.028, 0.056] |
| ceiling_gap | rho_bar(NB-oracle) - rho_bar(TW-1x-b1d) | 0.186 [0.167, 0.207] |
| ceiling_gap_b0 | rho_bar(NB-oracle-b0) - rho_bar(TW-1x-b0) | 0.099 [0.083, 0.115] |
| ceiling_gap_b0_4x | rho_bar(NB-oracle-b0) - rho_bar(TW-4x-b0) | 0.057 [0.046, 0.070] |
| bucket_ceiling | rho_bar(NB-oracle) - rho_bar(NB-oracle-b0) | 0.077 [0.063, 0.093] |

## By hop

| arm | hop=1 (3,480 rows) | hop=2 (4,306 rows) | hop=3 (4,134 rows) |
|---|---|---|---|
| TW-1x-dsh | 0.914 [0.788, 1.073] L14_HIGH | 1.045 [0.986, 1.113] L14_HIGH | 0.817 [0.788, 0.845] L14_HIGH |
| TW-1x-b1d | 0.806 [0.682, 0.941] L14_HIGH | 1.091 [1.029, 1.161] L14_ABOVE_GNN | 0.851 [0.820, 0.879] L14_HIGH |
| TW-1x-b0 | 0.784 [0.664, 0.918] L14_HIGH | 1.097 [1.034, 1.166] L14_ABOVE_GNN | 0.865 [0.835, 0.895] L14_HIGH |
| TW-4x-dsh | 0.947 [0.822, 1.107] L14_HIGH | 1.101 [1.037, 1.171] L14_ABOVE_GNN | 0.855 [0.826, 0.886] L14_HIGH |
| TW-4x-b1d | 0.862 [0.739, 1.001] L14_HIGH | 1.133 [1.071, 1.203] L14_ABOVE_GNN | 0.892 [0.864, 0.921] L14_HIGH |
| TW-4x-b0 | 0.852 [0.731, 0.988] L14_HIGH | 1.145 [1.080, 1.213] L14_ABOVE_GNN | 0.903 [0.873, 0.932] L14_HIGH |
| NB-oracle | 1.361 [1.201, 1.582] L14_ABOVE_GNN | 1.259 [1.193, 1.332] L14_ABOVE_GNN | 1.000 [0.973, 1.029] L14_HIGH |
| NB-oracle-b0 | 0.925 [0.801, 1.074] L14_HIGH | 1.233 [1.169, 1.307] L14_ABOVE_GNN | 0.950 [0.922, 0.978] L14_HIGH |

| contrast by hop | hop=1 | hop=2 | hop=3 |
|---|---|---|---|
| b0_adds | -0.022 [-0.072, 0.022] | 0.006 [-0.021, 0.031] | 0.015 [-0.001, 0.032] |
| b1d_adds | -0.108 [-0.196, -0.036] | 0.046 [0.018, 0.075] | 0.034 [0.018, 0.051] |

## Anchors (descriptive)

- full/TW-1x: argmax chain right on 0.885 of r's rows (hop=1 0.931, hop=2 0.898, hop=3 0.834; genre-ending 0.578 of 1,941).
- full/TW-4x: argmax chain right on 0.897 of r's rows (hop=1 0.944, hop=2 0.891, hop=3 0.863; genre-ending 0.578 of 1,941).
- b0/TW-1x: argmax chain right on 0.867 of r's rows (hop=1 0.904, hop=2 0.880, hop=3 0.822; genre-ending 0.605 of 1,941).
- b0/TW-4x: argmax chain right on 0.868 of r's rows (hop=1 0.904, hop=2 0.885, hop=3 0.821; genre-ending 0.588 of 1,941).
- b1d moved a mean 0.106 of the scored rows' posterior mass to the null type (TW-1x).
- b1d moved a mean 0.113 of the scored rows' posterior mass to the null type (TW-4x).
- Gap split (NB-oracle - TW-1x-b1d): argmax right on 31,656 (row, seed) pairs, 0.094 (hop 1 0.034, hop 2 0.017, hop 3 0.043); wrong on 4,104, 0.092 (hop 1 0.012, hop 2 0.016, hop 3 0.065).
- Gap split (NB-oracle - TW-4x-b1d): argmax right on 32,076 (row, seed) pairs, 0.095 (hop 1 0.034, hop 2 0.017, hop 3 0.044); wrong on 3,684, 0.049 (hop 1 0.007, hop 2 0.007, hop 3 0.034).
- Gap split (NB-oracle-b0 - TW-1x-b0): argmax right on 31,008 (row, seed) pairs, 0.016 (hop 1 0.001, hop 2 0.009, hop 3 0.006); wrong on 4,752, 0.083 (hop 1 0.010, hop 2 0.019, hop 3 0.055).
- Gap split (NB-oracle-b0 - TW-4x-b0): argmax right on 31,054 (row, seed) pairs, 0.007 (hop 1 0.001, hop 2 0.005, hop 3 0.002); wrong on 4,706, 0.051 (hop 1 0.006, hop 2 0.013, hop 3 0.032).
- r's nb view, all: 177.484 types and 5104.155 walk entries per row, of which bucket 0 holds 45.811 and 1080.828; 0.000 of rows have no bucket-0 type.
- r's nb view, hop=1: 171.957 types and 4805.441 walk entries per row, of which bucket 0 holds 45.629 and 1062.061; 0.000 of rows have no bucket-0 type.
- r's nb view, hop=2: 161.546 types and 5054.350 walk entries per row, of which bucket 0 holds 30.838 and 776.520; 0.000 of rows have no bucket-0 type.
- r's nb view, hop=3: 198.739 types and 5407.489 walk entries per row, of which bucket 0 holds 61.559 and 1413.596; 0.000 of rows have no bucket-0 type.
- beta per unit: TW-1x-dsh 1.5 x1, 2.0 x1, 3.0 x1; TW-1x-b1d 3.0 x1, 4.0 x2; TW-1x-b0 3.0 x1, 4.0 x1, 8.0 x1; TW-4x-dsh 2.0 x3; TW-4x-b1d 2.0 x2, 4.0 x1; TW-4x-b0 4.0 x1, 8.0 x1, inf x1.
- Grid edges, TW-1x-dsh: kappa at an end 0, eta at an end 1 of 3 units.
- Grid edges, TW-1x-b1d: kappa at an end 0, eta at an end 1 of 3 units.
- Grid edges, TW-1x-b0: kappa at an end 0, eta at an end 0 of 3 units.
- Grid edges, TW-4x-dsh: kappa at an end 0, eta at an end 0 of 3 units.
- Grid edges, TW-4x-b1d: kappa at an end 0, eta at an end 0 of 3 units.
- Grid edges, TW-4x-b0: kappa at an end 0, eta at an end 1 of 3 units.

Exchangeability (descriptive): the twin's and the GNN's mean over seeds on each population. The twin and the GNN trained on the fit carve and selected on the select carve, and saw none of the other populations.

| population | twin recall@5 | GNN recall@5 | twin full_coverage@5 | GNN full_coverage@5 | twin hit@1 | GNN hit@1 |
|---|---|---|---|---|---|---|
| r (train split) | 0.735 | 0.798 | 0.613 | 0.684 | 0.786 | 0.911 |
| level 13's dev rows | 0.627 | 0.709 | 0.465 | 0.555 | 0.737 | 0.893 |
| level 12's dev rows | 0.674 | 0.748 | 0.532 | 0.612 | 0.758 | 0.910 |
| level 12's fit carve | 0.744 | 0.800 | 0.627 | 0.681 | 0.824 | 0.936 |

## Checks

- r has no stored per-query metrics (the pilot never evaluated on it), so T_k's and G_k's metrics on r are the carve pass's own forwards. Level 12's loopcheck, which checked that pass against stored metrics, is pinned and on file as equal.
- Scoring integrity: every prepared seed set equals seeds_of and is in the pool; every structural edge carries a relation slot and a direction flag; every walk count is below 2^32 and no ceiling was passed.
- Repeat unit (b0, TW-1x, k 0) bit-identical: True.
- Units whose fit-set marginal log-likelihood fell between rounds: 0.
- Units whose kept theta sits at a clip bound: 0.

## What this does not say

Every arm is a measurement model on metaqa, fitted to metaqa train-split gold labels and read on r, train-split rows that neither it, the twin nor the GNN saw. It is never a deployable or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature contract, M3, M4 or any selection. TW-1x is the only label-matched fit. TW-4x fits on more metaqa labels than the GNN had, which the GNN was not refitted on. A reading on r is a reading on train-split rows. It is not a dev-split or a test-split reading, and it is never set beside level 13's dev-row numbers as one quantity. b0 reads the walks from two of the seeds only, so it reads less of the graph than the GNN reads. b0 was chosen from a look at level 12's dev rows, where it read +0.021 [-0.009, 0.051] over b1d with a loss on hop 1. rho here is level 8's quantity. It is also not the within-U_q oracle rho of levels 0 to 7.
