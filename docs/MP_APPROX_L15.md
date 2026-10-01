# MP-Approx level 15: the typed-walk model without message passing, with its type posterior normalised by FZ

Declaration: `configs/mp_approx_l15.yaml`. The scripts are `scripts/mp_approx_l15.py` (population, r2's scoring pass, check) and `scripts/mp_approx_l15_fit.py` (the eval part, FZ, deploy views, fits, read, doc, file). The record is `outputs/mp_approx_l15/record.json`.

## What was measured

On metaqa, level 14's non-message-passing typed-walk model, with its views, scores, selection, carves and fits, was fitted under two norms of its type posterior and read once on r2: 11,920 metaqa train-split rows (3,480 of hop 1, 4,306 of hop 2, 4,134 of hop 3) that the twin, the GNN and every level never trained on, selected on or read. r2 is the union of the M3B fit stride's offsets 10 and 11, disjoint from level 12's nine carves and from level 14's r, so every number here is a reading on train-split rows, not on dev or test rows. Under sm (level 8's softmax) a unit is level 14's unit, refitted here with r2 as its scored set. Under FZ, the one change of this file, in training and scoring alike, the posterior is level 9's NB-hyb logits normalised over U, the walk types of the unit's fit rows (V, built before the fit) united with the row's present types, plus the null type. A chain's mass over both seed buckets then goes to the buckets where the row has its walks, and to the null type when it has none. FZ adds no parameter and reads no edge, neighbour or gold. A full-view unit is read under dsh (level 11's sharpened mixture) and b1d (level 13's move of the bucket-1 probability to the null type after the fit); a b0 unit (level 14's bucket-0 view, fitted, selected and scored on the walks from the dense or the splade rank-1 node only) is read as b0. TW-1x fits on the GNN's metaqa fit carve and chooses its settings on the GNN's select carve, so it is label-matched to the GNN. TW-4x adds level 12's x1 to x3 and fits on four times the labels, and the GNN is not refitted on them. rho is level 8's quantity: the share of the GNN's gain over its twin on recall@5, full_coverage@5 and hit@1, and rho_bar is its mean over the readable metrics.

## Reading

- The primary arm is FZ-TW-1x-b1d (label-matched, FZ, b1d, 5,883 fit / 1,482 inner rows), and its band on r2 is **L15_HIGH**: rho_bar 0.929 [0.902, 0.957].
- Readable metrics: recall@5, full_coverage@5, hit@1.
- Flags: none.
- The interpretation map entries that apply: l15_high, matched_below_gnn, fz_adds, fz_adds_b0, fz_hurts_dsh, fz_adds_4x, fz_adds_4x_b0, fz_hurts_4x_dsh, over_l14_primary, data_adds_4x_fz, data_adds_4x_fz_b0, b0_ceiling_binds, fz_adds_hop1, fz_adds_hop2, fz_adds_hop3.
- The first read arm with no readable metric below the GNN (reaches_gnn): none; every read arm stays below the GNN on a readable metric.

## Arms

Intervals are 95% bootstrap intervals over 1,000 resamples of r2's rows; they do not resample the training carves or the fits. The gap to the GNN is the mean over seeds and rows of M(arm) - M(G).

| arm | norm | view | labels | rho_bar | band | gap recall@5 | gap full_coverage@5 | gap hit@1 |
|---|---|---|---|---|---|---|---|---|
| TW-1x-dsh | sm | full | 5,883 fit / 1,482 inner | 0.858 [0.831, 0.886] | L15_HIGH | -0.010 [-0.012, -0.008] BELOW_GNN | -0.012 [-0.015, -0.009] BELOW_GNN | -0.012 [-0.016, -0.008] BELOW_GNN |
| TW-1x-b1d | sm | full | 5,883 fit / 1,482 inner | 0.894 [0.868, 0.922] | L15_HIGH | -0.008 [-0.010, -0.006] BELOW_GNN | -0.009 [-0.012, -0.007] BELOW_GNN | -0.006 [-0.010, -0.002] BELOW_GNN |
| TW-1x-b0 | sm | b0 | 5,883 fit / 1,482 inner | 0.910 [0.881, 0.938] | L15_HIGH | -0.008 [-0.010, -0.006] BELOW_GNN | -0.008 [-0.010, -0.005] BELOW_GNN | -0.005 [-0.009, -0.001] BELOW_GNN |
| TW-4x-dsh | sm | full | 23,538 fit / 1,482 inner | 0.909 [0.882, 0.938] | L15_HIGH | -0.006 [-0.008, -0.005] BELOW_GNN | -0.009 [-0.012, -0.007] BELOW_GNN | -0.004 [-0.008, 0.000] |
| TW-4x-b1d | sm | full | 23,538 fit / 1,482 inner | 0.936 [0.909, 0.964] | L15_HIGH | -0.006 [-0.007, -0.004] BELOW_GNN | -0.007 [-0.010, -0.005] BELOW_GNN | 0.001 [-0.003, 0.005] |
| TW-4x-b0 | sm | b0 | 23,538 fit / 1,482 inner | 0.961 [0.932, 0.989] | L15_HIGH | -0.005 [-0.007, -0.003] BELOW_GNN | -0.005 [-0.008, -0.002] BELOW_GNN | 0.004 [-0.000, 0.008] |
| FZ-TW-1x-dsh | fz | full | 5,883 fit / 1,482 inner | 0.828 [0.800, 0.855] | L15_HIGH | -0.011 [-0.013, -0.009] BELOW_GNN | -0.014 [-0.017, -0.011] BELOW_GNN | -0.017 [-0.022, -0.013] BELOW_GNN |
| FZ-TW-1x-b1d | fz | full | 5,883 fit / 1,482 inner | 0.929 [0.902, 0.957] | L15_HIGH | -0.006 [-0.008, -0.004] BELOW_GNN | -0.006 [-0.009, -0.004] BELOW_GNN | -0.002 [-0.006, 0.002] |
| FZ-TW-1x-b0 | fz | b0 | 5,883 fit / 1,482 inner | 0.929 [0.902, 0.956] | L15_HIGH | -0.006 [-0.008, -0.005] BELOW_GNN | -0.006 [-0.009, -0.004] BELOW_GNN | -0.002 [-0.006, 0.002] |
| FZ-TW-4x-dsh | fz | full | 23,538 fit / 1,482 inner | 0.862 [0.835, 0.889] | L15_HIGH | -0.010 [-0.012, -0.008] BELOW_GNN | -0.012 [-0.015, -0.009] BELOW_GNN | -0.009 [-0.014, -0.005] BELOW_GNN |
| FZ-TW-4x-b1d | fz | full | 23,538 fit / 1,482 inner | 0.974 [0.946, 1.000] | L15_HIGH | -0.004 [-0.006, -0.002] BELOW_GNN | -0.003 [-0.006, -0.001] BELOW_GNN | 0.004 [-0.000, 0.008] |
| FZ-TW-4x-b0 | fz | b0 | 23,538 fit / 1,482 inner | 0.983 [0.955, 1.009] | L15_HIGH | -0.004 [-0.005, -0.002] BELOW_GNN | -0.002 [-0.005, 0.001] | 0.005 [0.001, 0.009] BEATS_GNN |
| NB-oracle | reference | full | none (reference) | 1.076 [1.049, 1.103] | L15_ABOVE_GNN | 0.002 [0.000, 0.004] BEATS_GNN | 0.001 [-0.002, 0.004] | 0.023 [0.019, 0.027] BEATS_GNN |
| NB-oracle-b0 | reference | b0 | none (reference) | 1.013 [0.986, 1.041] | L15_HIGH | -0.002 [-0.004, -0.000] BELOW_GNN | -0.001 [-0.004, 0.002] | 0.011 [0.007, 0.015] BEATS_GNN |

## Contrasts

Paired differences of rho_bar on the same rows of r2.

| contrast | of | paired difference |
|---|---|---|
| fz_adds | rho_bar(FZ-TW-1x-b1d) - rho_bar(TW-1x-b1d) | 0.035 [0.023, 0.048] |
| fz_adds_b0 | rho_bar(FZ-TW-1x-b0) - rho_bar(TW-1x-b0) | 0.019 [0.006, 0.033] |
| fz_adds_dsh | rho_bar(FZ-TW-1x-dsh) - rho_bar(TW-1x-dsh) | -0.030 [-0.044, -0.016] |
| fz_adds_4x | rho_bar(FZ-TW-4x-b1d) - rho_bar(TW-4x-b1d) | 0.038 [0.026, 0.050] |
| fz_adds_4x_b0 | rho_bar(FZ-TW-4x-b0) - rho_bar(TW-4x-b0) | 0.022 [0.010, 0.035] |
| fz_adds_4x_dsh | rho_bar(FZ-TW-4x-dsh) - rho_bar(TW-4x-dsh) | -0.047 [-0.062, -0.033] |
| over_l14_primary | rho_bar(FZ-TW-1x-b1d) - rho_bar(TW-1x-b0) | 0.020 [0.004, 0.036] |
| b1d_over_b0_fz | rho_bar(FZ-TW-1x-b1d) - rho_bar(FZ-TW-1x-b0) | 0.001 [-0.010, 0.012] |
| b1d_over_b0_fz_4x | rho_bar(FZ-TW-4x-b1d) - rho_bar(FZ-TW-4x-b0) | -0.009 [-0.017, 0.000] |
| data_4x_fz | rho_bar(FZ-TW-4x-b1d) - rho_bar(FZ-TW-1x-b1d) | 0.044 [0.031, 0.056] |
| data_4x_fz_b0 | rho_bar(FZ-TW-4x-b0) - rho_bar(FZ-TW-1x-b0) | 0.054 [0.043, 0.067] |
| ceiling_gap | rho_bar(NB-oracle) - rho_bar(FZ-TW-1x-b1d) | 0.147 [0.128, 0.168] |
| ceiling_gap_4x | rho_bar(NB-oracle) - rho_bar(FZ-TW-4x-b1d) | 0.103 [0.086, 0.120] |
| ceiling_gap_b0 | rho_bar(NB-oracle-b0) - rho_bar(FZ-TW-1x-b0) | 0.085 [0.072, 0.099] |
| bucket_ceiling | rho_bar(NB-oracle) - rho_bar(NB-oracle-b0) | 0.063 [0.049, 0.078] |

## By hop

| arm | hop=1 (3,480 rows) | hop=2 (4,306 rows) | hop=3 (4,134 rows) |
|---|---|---|---|
| TW-1x-dsh | 0.814 [0.732, 0.905] L15_HIGH | 0.965 [0.905, 1.032] L15_HIGH | 0.834 [0.802, 0.868] L15_HIGH |
| TW-1x-b1d | 0.790 [0.704, 0.883] L15_HIGH | 1.073 [1.012, 1.140] L15_ABOVE_GNN | 0.856 [0.826, 0.889] L15_HIGH |
| TW-1x-b0 | 0.759 [0.672, 0.858] L15_HIGH | 1.074 [1.008, 1.146] L15_ABOVE_GNN | 0.881 [0.849, 0.914] L15_HIGH |
| TW-4x-dsh | 0.913 [0.831, 1.005] L15_HIGH | 1.041 [0.980, 1.108] L15_HIGH | 0.872 [0.840, 0.905] L15_HIGH |
| TW-4x-b1d | 0.840 [0.762, 0.928] L15_HIGH | 1.110 [1.049, 1.177] L15_ABOVE_GNN | 0.898 [0.867, 0.930] L15_HIGH |
| TW-4x-b0 | 0.825 [0.739, 0.924] L15_HIGH | 1.142 [1.082, 1.211] L15_ABOVE_GNN | 0.925 [0.892, 0.960] L15_HIGH |
| FZ-TW-1x-dsh | 0.843 [0.761, 0.933] L15_HIGH | 0.935 [0.871, 1.001] L15_HIGH | 0.798 [0.765, 0.832] L15_HIGH |
| FZ-TW-1x-b1d | 0.846 [0.761, 0.943] L15_HIGH | 1.143 [1.077, 1.211] L15_ABOVE_GNN | 0.878 [0.846, 0.910] L15_HIGH |
| FZ-TW-1x-b0 | 0.826 [0.742, 0.921] L15_HIGH | 1.114 [1.053, 1.180] L15_ABOVE_GNN | 0.888 [0.857, 0.917] L15_HIGH |
| FZ-TW-4x-dsh | 0.867 [0.784, 0.965] L15_HIGH | 0.977 [0.916, 1.046] L15_HIGH | 0.829 [0.797, 0.861] L15_HIGH |
| FZ-TW-4x-b1d | 0.901 [0.817, 1.000] L15_HIGH | 1.208 [1.142, 1.274] L15_ABOVE_GNN | 0.915 [0.884, 0.946] L15_HIGH |
| FZ-TW-4x-b0 | 0.843 [0.751, 0.955] L15_HIGH | 1.228 [1.164, 1.296] L15_ABOVE_GNN | 0.929 [0.897, 0.960] L15_HIGH |
| NB-oracle | 1.200 [1.104, 1.331] L15_ABOVE_GNN | 1.242 [1.173, 1.310] L15_ABOVE_GNN | 1.012 [0.981, 1.045] L15_HIGH |
| NB-oracle-b0 | 0.914 [0.829, 1.015] L15_HIGH | 1.240 [1.177, 1.311] L15_ABOVE_GNN | 0.960 [0.928, 0.991] L15_HIGH |

| contrast by hop | hop=1 | hop=2 | hop=3 |
|---|---|---|---|
| fz_adds | 0.056 [0.022, 0.095] | 0.070 [0.044, 0.099] | 0.022 [0.006, 0.038] |
| fz_adds_b0 | 0.067 [0.018, 0.116] | 0.040 [0.015, 0.067] | 0.006 [-0.010, 0.024] |
| over_l14_primary | 0.087 [0.041, 0.140] | 0.069 [0.041, 0.101] | -0.003 [-0.022, 0.016] |

## By where the true chain's walks start (descriptive)

true_from_b0: the true chain has a walk from a bucket-0 seed; true_from_b1_only: from bucket-1 seeds only; true_nowhere: from neither (level 8's chain_reach on r2's nb view, the chain named by the row's qtype). A stratum's rho is read only where its own denominator interval lies above 0.

| arm | where=true_from_b0 (10,652 rows) | where=true_from_b1_only (1,009 rows) | where=true_nowhere (259 rows) |
|---|---|---|---|
| TW-1x-dsh | 0.929 [0.903, 0.956] | -0.690 [-1.157, -0.400] | n/a |
| TW-1x-b1d | 0.959 [0.933, 0.987] | -0.483 [-0.765, -0.296] | n/a |
| TW-1x-b0 | 0.990 [0.963, 1.018] | -0.826 [-1.268, -0.561] | n/a |
| TW-4x-dsh | 0.970 [0.944, 0.997] | -0.455 [-0.848, -0.189] | n/a |
| TW-4x-b1d | 0.995 [0.969, 1.022] | -0.327 [-0.586, -0.149] | n/a |
| TW-4x-b0 | 1.027 [0.999, 1.054] | -0.542 [-0.903, -0.310] | n/a |
| FZ-TW-1x-dsh | 0.919 [0.894, 0.946] | -1.310 [-1.986, -0.910] | n/a |
| FZ-TW-1x-b1d | 0.981 [0.954, 1.009] | -0.226 [-0.399, -0.112] | n/a |
| FZ-TW-1x-b0 | 0.974 [0.948, 1.000] | -0.109 [-0.227, -0.024] | n/a |
| FZ-TW-4x-dsh | 0.945 [0.920, 0.973] | -1.137 [-1.757, -0.734] | n/a |
| FZ-TW-4x-b1d | 1.018 [0.991, 1.047] | -0.016 [-0.084, 0.050] | n/a |
| FZ-TW-4x-b0 | 1.029 [1.002, 1.057] | -0.055 [-0.139, 0.009] | n/a |
| NB-oracle | 1.117 [1.091, 1.144] | 0.172 [-0.106, 0.403] | n/a |
| NB-oracle-b0 | 1.059 [1.030, 1.086] | 0.000 [0.000, 0.000] | n/a |

| contrast by where | where=true_from_b0 | where=true_from_b1_only | where=true_nowhere |
|---|---|---|---|
| fz_adds | 0.022 [0.010, 0.032] | 0.257 [0.101, 0.476] | n/a |
| fz_adds_b0 | -0.016 [-0.027, -0.006] | 0.717 [0.481, 1.081] | n/a |

## Anchors (descriptive)

- sm/full/TW-1x: argmax chain right on 0.882 of r2's rows (hop=1 0.925, hop=2 0.899, hop=3 0.829; true_from_b0 0.949, true_from_b1_only 0.401, true_nowhere 0.000; genre-ending 0.561 of 1,929); null mode on 0.006 of the (row, seed) pairs (true_from_b0 0.000, true_from_b1_only 0.064, true_nowhere 0.032).
- sm/full/TW-4x: argmax chain right on 0.899 of r2's rows (hop=1 0.942, hop=2 0.893, hop=3 0.870; true_from_b0 0.967, true_from_b1_only 0.415, true_nowhere 0.000; genre-ending 0.561 of 1,929); null mode on 0.020 of the (row, seed) pairs (true_from_b0 0.001, true_from_b1_only 0.184, true_nowhere 0.187).
- sm/b0/TW-1x: argmax chain right on 0.868 of r2's rows (hop=1 0.906, hop=2 0.879, hop=3 0.825; true_from_b0 0.972, true_from_b1_only 0.000, true_nowhere 0.000; genre-ending 0.604 of 1,929); null mode on 0.069 of the (row, seed) pairs (true_from_b0 0.008, true_from_b1_only 0.614, true_nowhere 0.431).
- sm/b0/TW-4x: argmax chain right on 0.872 of r2's rows (hop=1 0.906, hop=2 0.886, hop=3 0.830; true_from_b0 0.976, true_from_b1_only 0.000, true_nowhere 0.000; genre-ending 0.589 of 1,929); null mode on 0.085 of the (row, seed) pairs (true_from_b0 0.011, true_from_b1_only 0.738, true_nowhere 0.582).
- fz/full/TW-1x: argmax chain right on 0.894 of r2's rows (hop=1 0.934, hop=2 0.908, hop=3 0.846; true_from_b0 0.931, true_from_b1_only 0.731, true_nowhere 0.000; genre-ending 0.602 of 1,929); null mode on 0.053 of the (row, seed) pairs (true_from_b0 0.024, true_from_b1_only 0.152, true_nowhere 0.875).
- fz/full/TW-4x: argmax chain right on 0.914 of r2's rows (hop=1 0.945, hop=2 0.924, hop=3 0.877; true_from_b0 0.951, true_from_b1_only 0.756, true_nowhere 0.000; genre-ending 0.674 of 1,929); null mode on 0.051 of the (row, seed) pairs (true_from_b0 0.021, true_from_b1_only 0.161, true_nowhere 0.844).
- fz/b0/TW-1x: argmax chain right on 0.843 of r2's rows (hop=1 0.878, hop=2 0.873, hop=3 0.781; true_from_b0 0.943, true_from_b1_only 0.000, true_nowhere 0.000; genre-ending 0.567 of 1,929); null mode on 0.131 of the (row, seed) pairs (true_from_b0 0.033, true_from_b1_only 0.960, true_nowhere 0.910).
- fz/b0/TW-4x: argmax chain right on 0.870 of r2's rows (hop=1 0.897, hop=2 0.896, hop=3 0.820; true_from_b0 0.973, true_from_b1_only 0.000, true_nowhere 0.000; genre-ending 0.601 of 1,929); null mode on 0.112 of the (row, seed) pairs (true_from_b0 0.012, true_from_b1_only 0.966, true_nowhere 0.900).
- b1d moved a mean 0.106 of the scored rows' posterior mass to the null type (sm/TW-1x).
- b1d moved a mean 0.112 of the scored rows' posterior mass to the null type (sm/TW-4x).
- b1d moved a mean 0.143 of the scored rows' posterior mass to the null type (fz/TW-1x).
- b1d moved a mean 0.127 of the scored rows' posterior mass to the null type (fz/TW-4x).
- FZ vocabulary, full/TW-1x: V holds 2,069 walk types, 1,976 with a partner in the other bucket; the scored rows hold a mean 348.000 (row, type) entries outside V, on 81.000 of their 11,920 rows.
- FZ vocabulary, full/TW-4x: V holds 2,350 walk types, 2,072 with a partner in the other bucket; the scored rows hold a mean 108.000 (row, type) entries outside V, on 26.000 of their 11,920 rows.
- FZ vocabulary, b0/TW-1x: V holds 929 walk types, 0 with a partner in the other bucket; the scored rows hold a mean 147.000 (row, type) entries outside V, on 47.000 of their 11,920 rows.
- FZ vocabulary, b0/TW-4x: V holds 1,022 walk types, 0 with a partner in the other bucket; the scored rows hold a mean 37.000 (row, type) entries outside V, on 17.000 of their 11,920 rows.
- Gap split (NB-oracle - FZ-TW-1x-b1d): argmax right on 31,973 (row, seed) pairs, 0.095 (hop 1 0.027, hop 2 0.016, hop 3 0.052); wrong on 3,787, 0.052 (hop 1 0.006, hop 2 0.004, hop 3 0.042).
- Gap split (NB-oracle - TW-1x-b1d): argmax right on 31,555 (row, seed) pairs, 0.095 (hop 1 0.023, hop 2 0.024, hop 3 0.049); wrong on 4,205, 0.087 (hop 1 0.015, hop 2 0.011, hop 3 0.061).
- Gap split (NB-oracle - FZ-TW-4x-b1d): argmax right on 32,677 (row, seed) pairs, 0.075 (hop 1 0.026, hop 2 0.008, hop 3 0.041); wrong on 3,083, 0.028 (hop 1 0.002, hop 2 -0.001, hop 3 0.027).
- Gap split (NB-oracle-b0 - FZ-TW-1x-b0): argmax right on 30,138 (row, seed) pairs, 0.030 (hop 1 0.003, hop 2 0.017, hop 3 0.009); wrong on 5,622, 0.055 (hop 1 0.005, hop 2 0.009, hop 3 0.041).
- Gap split (NB-oracle-b0 - TW-1x-b0): argmax right on 31,052 (row, seed) pairs, 0.025 (hop 1 0.003, hop 2 0.015, hop 3 0.007); wrong on 4,708, 0.079 (hop 1 0.012, hop 2 0.019, hop 3 0.048).
- r2's nb view, all: 177.718 types and 5071.125 walk entries per row, of which bucket 0 holds 45.845 and 1049.957; 0.000 of rows have no bucket-0 type.
- r2's nb view, hop=1: 173.280 types and 4777.210 walk entries per row, of which bucket 0 holds 46.300 and 1046.001; 0.000 of rows have no bucket-0 type.
- r2's nb view, hop=2: 161.811 types and 5035.466 walk entries per row, of which bucket 0 holds 31.523 and 779.664; 0.000 of rows have no bucket-0 type.
- r2's nb view, hop=3: 198.023 types and 5355.684 walk entries per row, of which bucket 0 holds 60.381 and 1334.825; 0.000 of rows have no bucket-0 type.
- beta per unit: TW-1x-dsh 1.5 x1, 2.0 x1, 3.0 x1; TW-1x-b1d 3.0 x1, 4.0 x2; TW-1x-b0 3.0 x1, 4.0 x1, 8.0 x1; TW-4x-dsh 2.0 x3; TW-4x-b1d 2.0 x2, 4.0 x1; TW-4x-b0 4.0 x1, 8.0 x1, inf x1; FZ-TW-1x-dsh 1.0 x1, 2.0 x2; FZ-TW-1x-b1d 1.5 x1, 2.0 x2; FZ-TW-1x-b0 1.5 x1, 3.0 x1, 4.0 x1; FZ-TW-4x-dsh 1.5 x1, 2.0 x1, 8.0 x1; FZ-TW-4x-b1d 3.0 x1, 4.0 x2; FZ-TW-4x-b0 4.0 x1, 8.0 x2.
- Grid edges, TW-1x-dsh: kappa at an end 0, eta at an end 1 of 3 units.
- Grid edges, TW-1x-b1d: kappa at an end 0, eta at an end 1 of 3 units.
- Grid edges, TW-1x-b0: kappa at an end 0, eta at an end 0 of 3 units.
- Grid edges, TW-4x-dsh: kappa at an end 0, eta at an end 0 of 3 units.
- Grid edges, TW-4x-b1d: kappa at an end 0, eta at an end 0 of 3 units.
- Grid edges, TW-4x-b0: kappa at an end 0, eta at an end 1 of 3 units.
- Grid edges, FZ-TW-1x-dsh: kappa at an end 0, eta at an end 0 of 3 units.
- Grid edges, FZ-TW-1x-b1d: kappa at an end 0, eta at an end 1 of 3 units.
- Grid edges, FZ-TW-1x-b0: kappa at an end 0, eta at an end 0 of 3 units.
- Grid edges, FZ-TW-4x-dsh: kappa at an end 0, eta at an end 0 of 3 units.
- Grid edges, FZ-TW-4x-b1d: kappa at an end 0, eta at an end 1 of 3 units.
- Grid edges, FZ-TW-4x-b0: kappa at an end 0, eta at an end 1 of 3 units.

Exchangeability (descriptive): the twin's and the GNN's mean over seeds on each population. The twin and the GNN trained on the fit carve and selected on the select carve, and saw none of the other populations.

| population | twin recall@5 | GNN recall@5 | twin full_coverage@5 | GNN full_coverage@5 | twin hit@1 | GNN hit@1 |
|---|---|---|---|---|---|---|
| r2 (train split) | 0.736 | 0.798 | 0.615 | 0.682 | 0.782 | 0.911 |
| r (train split; level 14's read) | 0.735 | 0.798 | 0.613 | 0.684 | 0.786 | 0.911 |
| level 13's dev rows | 0.627 | 0.709 | 0.465 | 0.555 | 0.737 | 0.893 |
| level 12's dev rows | 0.674 | 0.748 | 0.532 | 0.612 | 0.758 | 0.910 |
| level 12's fit carve | 0.744 | 0.800 | 0.627 | 0.681 | 0.824 | 0.936 |

## Checks

- r2 has no stored per-query metrics (the pilot never evaluated on it), so T_k's and G_k's metrics on r2 are the carve pass's own forwards. Level 12's loopcheck, which checked that pass against stored metrics, is pinned and on file as equal.
- Scoring integrity: every prepared seed set equals seeds_of and is in the pool; every structural edge carries a relation slot and a direction flag; every walk count is below 2^32 and no ceiling was passed.
- Every FZ posterior summed to one within 1e-4 on every row each FZ unit scored, fitted or selected on (a hard stop otherwise).
- Repeat unit (fz, full, TW-1x, k 0) bit-identical: True.
- Units whose fit-set marginal log-likelihood fell between rounds: 0.
- Units whose kept theta sits at a clip bound: 0.

## What this does not say

Every arm is a measurement model on metaqa, fitted to metaqa train-split gold labels and read on r2, train-split rows that neither it, the twin nor the GNN saw. It is never a deployable or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature contract, M3, M4 or any selection. TW-1x and FZ-TW-1x are the only label-matched fits. TW-4x and FZ-TW-4x fit on more metaqa labels than the GNN had, which the GNN was not refitted on. A reading on r2 is a reading on train-split rows. It is not a dev-split or a test-split reading, and it is never set beside a dev-row number as one quantity, nor beside level 14's numbers on r as one quantity. FZ was chosen from design looks at r, where FZ-TW-1x-b1d read +0.043 [0.031, 0.056] over TW-1x-b1d; r2 is read once here. b0 reads the walks from two of the seeds only, so it reads less of the graph than the GNN reads. rho here is level 8's quantity. It is also not the within-U_q oracle rho of levels 0 to 7.
