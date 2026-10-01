# MP-Approx level 12: a typed-walk model without message passing, fitted on the GNN's own metaqa labels

Declaration: `configs/mp_approx_l12.yaml`. The scripts are `scripts/mp_approx_l12.py` (population, dev and carve scoring passes, loopcheck, checks) and `scripts/mp_approx_l12_fit.py` (deploy views, fits, read, doc, file). The record is `outputs/mp_approx_l12/record.json`.

## What was measured

On metaqa, level 11's non-message-passing typed-walk model was fitted on metaqa train-split labels and read once on 2,221 fresh V2_GATE dev rows, disjoint from levels 0 to 11. The model is level 10's NB-set model (level 9's NB-hyb model on non-backtracking walks, with level 10's set likelihood and EM over the latent chain), scored two ways: cov, level 10's coverage, and dsh, level 8's mixture under a sharpened posterior (level 11's). TW-1x fits on the M3B fit carve, the GNN's own metaqa training rows (5,883 of them with an in-pool gold), and chooses its kept round, kappa, eta and beta on the M3B select carve (1,482 rows with an in-pool gold), on which the GNN chose its epoch. TW-1x is label-matched to the GNN on metaqa: its fit rows are the GNN's metaqa fit carve, its settings are chosen on the GNN's select carve, and it is read on dev rows that neither model saw. The GNN also trained on 2wiki's and squad's fit carves, which hold no metaqa label. TW-2x, TW-4x and TW-8x add 1, 3 and 7 further train-split carves of the same stride and fit on 11,777, 23,538 and 47,108 labelled rows. The GNN is not refitted on them, so none is a matched comparison, and each of their readings names its label count. The file measures what non-message-passing typed-walk models recover of the GNN's gain on this population. rho is level 8's quantity on a fresh population. It is not the within-U_q oracle rho of levels 0 to 7, and the two are never one quantity.

## Reading

- The primary arm is TW-1x-dsh (label-matched, 5,883 fit / 1,482 inner rows), and its band is **L12_HIGH**.
- Readable metrics: recall@5, full_coverage@5, hit@1.
- Flags: none.
- The interpretation map entries that apply: l12_high, matched_below_gnn, data_adds_4x, data_adds_8x, last_doubling_flat, dsh_adds.
- The first fit whose dsh arm has no readable metric below the GNN (reaches_gnn): none; every fit's dsh arm stays below the GNN on a readable metric.

## Arms

rho_bar is the mean of rho over the readable metrics. Intervals are 95% bootstrap intervals over 1,000 dev-row resamples; they do not resample the training carves or the fits. An arm's name is its fit, then its score. The label counts are the rows with an in-pool gold the fit trains on and chooses its settings on.

| arm | labels | rho_bar | band | rho recall@5 | rho full_coverage@5 | rho hit@1 |
|---|---|---|---|---|---|---|
| TW-1x-dsh | 5,883 fit / 1,482 inner | 0.878 [0.826, 0.934] | L12_HIGH | 0.860 [0.799, 0.919] | 0.838 [0.762, 0.915] | 0.936 [0.873, 1.005] |
| TW-1x-cov | 5,883 fit / 1,482 inner | 0.842 [0.787, 0.897] | L12_HIGH | 0.819 [0.758, 0.882] | 0.773 [0.696, 0.850] | 0.934 [0.870, 1.003] |
| TW-2x-dsh | 11,777 fit / 1,482 inner | 0.891 [0.831, 0.950] | L12_HIGH | 0.861 [0.794, 0.930] | 0.852 [0.769, 0.933] | 0.960 [0.894, 1.027] |
| TW-2x-cov | 11,777 fit / 1,482 inner | 0.880 [0.825, 0.939] | L12_HIGH | 0.856 [0.792, 0.917] | 0.821 [0.744, 0.904] | 0.962 [0.896, 1.035] |
| TW-4x-dsh | 23,538 fit / 1,482 inner | 0.916 [0.867, 0.971] | L12_HIGH | 0.900 [0.843, 0.958] | 0.868 [0.791, 0.946] | 0.980 [0.916, 1.050] |
| TW-4x-cov | 23,538 fit / 1,482 inner | 0.895 [0.842, 0.953] | L12_HIGH | 0.870 [0.812, 0.931] | 0.835 [0.755, 0.913] | 0.981 [0.914, 1.055] |
| TW-8x-dsh | 47,108 fit / 1,482 inner | 0.915 [0.860, 0.968] | L12_HIGH | 0.899 [0.839, 0.956] | 0.857 [0.780, 0.932] | 0.988 [0.927, 1.056] |
| TW-8x-cov | 47,108 fit / 1,482 inner | 0.899 [0.842, 0.955] | L12_HIGH | 0.886 [0.824, 0.947] | 0.829 [0.750, 0.907] | 0.983 [0.916, 1.053] |
| NB-oracle | none (reference) | 1.042 [0.988, 1.101] | L12_HIGH | 1.001 [0.944, 1.066] | 0.994 [0.919, 1.076] | 1.131 [1.068, 1.195] |

The gap to the GNN is the mean over seeds and dev rows of M(arm) - M(G).

| arm | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| TW-1x-dsh | -0.010 [-0.015, -0.006] BELOW_GNN | -0.013 [-0.020, -0.007] BELOW_GNN | -0.010 [-0.020, 0.001] |
| TW-1x-cov | -0.013 [-0.018, -0.009] BELOW_GNN | -0.018 [-0.025, -0.011] BELOW_GNN | -0.010 [-0.020, 0.000] |
| TW-2x-dsh | -0.010 [-0.015, -0.005] BELOW_GNN | -0.012 [-0.019, -0.005] BELOW_GNN | -0.006 [-0.017, 0.004] |
| TW-2x-cov | -0.011 [-0.015, -0.006] BELOW_GNN | -0.014 [-0.022, -0.008] BELOW_GNN | -0.006 [-0.016, 0.005] |
| TW-4x-dsh | -0.007 [-0.012, -0.003] BELOW_GNN | -0.011 [-0.017, -0.004] BELOW_GNN | -0.003 [-0.013, 0.007] |
| TW-4x-cov | -0.010 [-0.014, -0.005] BELOW_GNN | -0.013 [-0.020, -0.007] BELOW_GNN | -0.003 [-0.014, 0.008] |
| TW-8x-dsh | -0.007 [-0.012, -0.003] BELOW_GNN | -0.011 [-0.018, -0.006] BELOW_GNN | -0.002 [-0.011, 0.008] |
| TW-8x-cov | -0.008 [-0.013, -0.004] BELOW_GNN | -0.014 [-0.021, -0.007] BELOW_GNN | -0.003 [-0.013, 0.008] |
| NB-oracle | 0.000 [-0.004, 0.005] | -0.000 [-0.007, 0.006] | 0.020 [0.011, 0.029] BEATS_GNN |

Descriptive means over seeds and dev rows (the twin and the GNN are the stored values):

| arm | recall@5 (arm / twin / GNN) | full_coverage@5 | hit@1 |
|---|---|---|---|
| TW-1x-dsh | 0.738 / 0.674 / 0.748 | 0.599 / 0.532 / 0.612 | 0.900 / 0.758 / 0.910 |
| TW-1x-cov | 0.735 / 0.674 / 0.748 | 0.593 / 0.532 / 0.612 | 0.900 / 0.758 / 0.910 |
| TW-2x-dsh | 0.738 / 0.674 / 0.748 | 0.600 / 0.532 / 0.612 | 0.904 / 0.758 / 0.910 |
| TW-2x-cov | 0.737 / 0.674 / 0.748 | 0.597 / 0.532 / 0.612 | 0.904 / 0.758 / 0.910 |
| TW-4x-dsh | 0.741 / 0.674 / 0.748 | 0.601 / 0.532 / 0.612 | 0.907 / 0.758 / 0.910 |
| TW-4x-cov | 0.738 / 0.674 / 0.748 | 0.598 / 0.532 / 0.612 | 0.907 / 0.758 / 0.910 |
| TW-8x-dsh | 0.741 / 0.674 / 0.748 | 0.600 / 0.532 / 0.612 | 0.908 / 0.758 / 0.910 |
| TW-8x-cov | 0.740 / 0.674 / 0.748 | 0.598 / 0.532 / 0.612 | 0.908 / 0.758 / 0.910 |
| NB-oracle | 0.748 / 0.674 / 0.748 | 0.611 / 0.532 / 0.612 | 0.930 / 0.758 / 0.910 |

## Denominators

| metric | mean M(G) - M(T) | readable |
|---|---|---|
| recall@5 | 0.074 [0.067, 0.081] | True |
| full_coverage@5 | 0.080 [0.070, 0.090] | True |
| hit@1 | 0.152 [0.138, 0.167] | True |

## Contrasts

Paired differences of rho_bar on the same dev rows.

| contrast | of | paired difference |
|---|---|---|
| data_2x | rho_bar(TW-2x-dsh) - rho_bar(TW-1x-dsh) | 0.013 [-0.021, 0.044] |
| data_4x | rho_bar(TW-4x-dsh) - rho_bar(TW-1x-dsh) | 0.038 [0.011, 0.068] |
| data_8x | rho_bar(TW-8x-dsh) - rho_bar(TW-1x-dsh) | 0.037 [0.005, 0.070] |
| step_4x | rho_bar(TW-4x-dsh) - rho_bar(TW-2x-dsh) | 0.026 [-0.005, 0.059] |
| step_8x | rho_bar(TW-8x-dsh) - rho_bar(TW-4x-dsh) | -0.002 [-0.020, 0.016] |
| dsh_adds | rho_bar(TW-1x-dsh) - rho_bar(TW-1x-cov) | 0.036 [0.012, 0.060] |
| dsh_adds_8x | rho_bar(TW-8x-dsh) - rho_bar(TW-8x-cov) | 0.015 [-0.003, 0.033] |
| ceiling_gap | rho_bar(NB-oracle) - rho_bar(TW-1x-dsh) | 0.164 [0.122, 0.213] |
| ceiling_gap_8x | rho_bar(NB-oracle) - rho_bar(TW-8x-dsh) | 0.127 [0.088, 0.171] |

## By hop

rho_bar and band per hop, each hop read on its own readable metrics.

| arm | hop=1 (221 rows; recall@5, full_coverage@5, hit@1) | hop=2 (1000 rows; recall@5, full_coverage@5, hit@1) | hop=3 (1000 rows; recall@5, full_coverage@5, hit@1) |
|---|---|---|---|
| TW-1x-dsh | 1.024 [0.840, 1.341] L12_HIGH | 1.012 [0.865, 1.192] L12_HIGH | 0.838 [0.781, 0.898] L12_HIGH |
| TW-1x-cov | 0.993 [0.799, 1.299] L12_HIGH | 0.963 [0.814, 1.141] L12_HIGH | 0.805 [0.746, 0.870] L12_HIGH |
| TW-2x-dsh | 1.046 [0.856, 1.402] L12_HIGH | 0.989 [0.834, 1.160] L12_HIGH | 0.859 [0.795, 0.927] L12_HIGH |
| TW-2x-cov | 1.033 [0.849, 1.370] L12_HIGH | 1.005 [0.853, 1.179] L12_HIGH | 0.841 [0.779, 0.904] L12_HIGH |
| TW-4x-dsh | 1.001 [0.791, 1.351] L12_HIGH | 1.052 [0.892, 1.241] L12_HIGH | 0.878 [0.823, 0.935] L12_HIGH |
| TW-4x-cov | 1.042 [0.854, 1.389] L12_HIGH | 1.039 [0.887, 1.217] L12_HIGH | 0.852 [0.793, 0.910] L12_HIGH |
| TW-8x-dsh | 0.992 [0.765, 1.369] L12_HIGH | 1.075 [0.915, 1.268] L12_HIGH | 0.871 [0.815, 0.927] L12_HIGH |
| TW-8x-cov | 0.969 [0.750, 1.311] L12_HIGH | 1.072 [0.909, 1.268] L12_HIGH | 0.852 [0.793, 0.912] L12_HIGH |
| NB-oracle | 1.155 [1.000, 1.520] L12_HIGH | 1.243 [1.090, 1.445] L12_ABOVE_GNN | 0.988 [0.936, 1.047] L12_HIGH |

## Anchors (descriptive)

- Training rows with an in-pool gold per fit, by part (fit rows; inner rows):
  - TW-1x: fit 5,883 (fit 5,883); inner 1,482 (select 1,482)
  - TW-2x: fit 11,777 (fit 5,883, x1 5,894); inner 1,482 (select 1,482)
  - TW-4x: fit 23,538 (fit 5,883, x1 5,894, x2 5,866, x3 5,895); inner 1,482 (select 1,482)
  - TW-8x: fit 47,108 (fit 5,883, x1 5,894, x2 5,866, x3 5,895, x4 5,893, x5 5,885, x6 5,900, x7 5,892); inner 1,482 (select 1,482)
- Walk sequences on the dev rows' types that occur on no fit row (they get no gradient), and the share of dev rows whose true chain's sequence occurs on no fit row:
  - TW-1x: 6 of 929 sequences; true chain unseen on 0.000
  - TW-2x: 6 of 929 sequences; true chain unseen on 0.000
  - TW-4x: 2 of 929 sequences; true chain unseen on 0.000
  - TW-8x: 2 of 929 sequences; true chain unseen on 0.000
- The argmax type has the true chain's tokens on this share of dev rows (the mean over seeds; hop 1, 2, 3; genre-ending questions, 397 of them):
  - TW-1x: 0.865 (0.940, 0.878, 0.835; genre-ending 0.577)
  - TW-2x: 0.876 (0.940, 0.879, 0.859; genre-ending 0.599)
  - TW-4x: 0.870 (0.946, 0.868, 0.856; genre-ending 0.545)
  - TW-8x: 0.873 (0.946, 0.868, 0.862; genre-ending 0.568)
- The ceiling gap (NB-oracle - arm) split by whether the fit's argmax chain is the true chain, in rho_bar units (the hops are contributions to the whole and sum to it):
  - TW-1x-dsh: argmax right on 5762 (row, seed) pairs, 0.101 (by hop 0.004, 0.022, 0.075); wrong on 901, 0.063 (by hop 0.000, 0.022, 0.040)
  - TW-8x-dsh: argmax right on 5815 (row, seed) pairs, 0.093 (by hop 0.006, 0.017, 0.069); wrong on 848, 0.035 (by hop -0.000, 0.015, 0.021)
- The set likelihood's kept theta (rho_1, rho_2, rho_3, eps_s), the mean over units, then the range:
  - TW-1x: mean 0.8864, 0.9451, 0.8743, 0.0001075; min 0.8838, 0.9447, 0.8694, 0.0001052; max 0.8882, 0.9456, 0.8778, 0.0001092
  - TW-2x: mean 0.8799, 0.9557, 0.8842, 0.0001163; min 0.8797, 0.9556, 0.8807, 0.0001144; max 0.8801, 0.9558, 0.8874, 0.0001179
  - TW-4x: mean 0.8731, 0.9584, 0.8832, 0.0001169; min 0.8713, 0.9579, 0.8819, 0.0001162; max 0.874, 0.9588, 0.8851, 0.0001178
  - TW-8x: mean 0.8749, 0.9563, 0.8924, 0.0001216; min 0.8734, 0.9562, 0.8863, 0.0001182; max 0.8763, 0.9564, 0.8972, 0.0001241
- The dsh score's beta per unit (the number of units choosing each value):
  - TW-1x-dsh: 1.0 0, 1.5 1, 2.0 1, 3.0 1, 4.0 0, 8.0 0, inf 0
  - TW-2x-dsh: 1.0 0, 1.5 0, 2.0 0, 3.0 0, 4.0 2, 8.0 1, inf 0
  - TW-4x-dsh: 1.0 0, 1.5 0, 2.0 3, 3.0 0, 4.0 0, 8.0 0, inf 0
  - TW-8x-dsh: 1.0 0, 1.5 2, 2.0 1, 3.0 0, 4.0 0, 8.0 0, inf 0
- Grid edges and kept rounds per arm:
  - TW-1x-dsh (3 units): kappa at an end 0 (low 0, high 0), eta at an end 1 (low 1, high 0), the last round kept 0
  - TW-1x-cov (3 units): kappa at an end 0 (low 0, high 0), eta at an end 1 (low 0, high 1), the last round kept 0
  - TW-2x-dsh (3 units): kappa at an end 0 (low 0, high 0), eta at an end 0 (low 0, high 0), the last round kept 1
  - TW-2x-cov (3 units): kappa at an end 0 (low 0, high 0), eta at an end 1 (low 0, high 1), the last round kept 1
  - TW-4x-dsh (3 units): kappa at an end 0 (low 0, high 0), eta at an end 0 (low 0, high 0), the last round kept 0
  - TW-4x-cov (3 units): kappa at an end 0 (low 0, high 0), eta at an end 1 (low 0, high 1), the last round kept 0
  - TW-8x-dsh (3 units): kappa at an end 0 (low 0, high 0), eta at an end 0 (low 0, high 0), the last round kept 0
  - TW-8x-cov (3 units): kappa at an end 0 (low 0, high 0), eta at an end 0 (low 0, high 0), the last round kept 0

The twin and the GNN on each sidecar (the mean over seeds; on the fit carve both are in-sample, and on the select carve the GNN chose its epoch):

| sidecar | rows | with an in-pool gold | recall@5 twin / GNN | full_coverage@5 twin / GNN | hit@1 twin / GNN |
|---|---|---|---|---|---|
| dev | 2,221 | 2,184 | 0.674 / 0.748 | 0.532 / 0.612 | 0.758 / 0.910 |
| fit | 5,960 | 5,883 | 0.744 / 0.800 | 0.627 / 0.681 | 0.824 / 0.936 |
| select | 1,497 | 1,482 | 0.740 / 0.800 | 0.626 / 0.683 | 0.786 / 0.921 |
| x1 | 5,960 | 5,894 | 0.731 / 0.792 | 0.610 / 0.674 | 0.786 / 0.912 |
| x2 | 5,960 | 5,866 | 0.724 / 0.788 | 0.602 / 0.670 | 0.778 / 0.908 |
| x3 | 5,960 | 5,895 | 0.729 / 0.794 | 0.605 / 0.677 | 0.780 / 0.912 |
| x4 | 5,960 | 5,893 | 0.733 / 0.796 | 0.611 / 0.680 | 0.790 / 0.916 |
| x5 | 5,960 | 5,885 | 0.740 / 0.803 | 0.618 / 0.690 | 0.780 / 0.910 |
| x6 | 5,960 | 5,900 | 0.738 / 0.798 | 0.616 / 0.684 | 0.786 / 0.913 |
| x7 | 5,960 | 5,892 | 0.732 / 0.791 | 0.606 / 0.672 | 0.785 / 0.908 |

Level 9's anchors on each sidecar:

| sidecar | chain reaches a gold, std / nb | swapped, std / nb | NB removes of R* | in-pool golds in no nb reach set | topic entity in the pool | pool mean |
|---|---|---|---|---|---|---|
| dev | 0.919 / 0.900 | 0.000 / 0.000 | 0.125 | 0.001 | 0.998 | 2018.016 |
| fit | 0.933 / 0.919 | 0.000 / 0.000 | 0.095 | 0.001 | 0.997 | 2016.947 |
| select | 0.933 / 0.922 | 0.000 / 0.000 | 0.096 | 0.001 | 0.996 | 2014.672 |
| x1 | 0.934 / 0.921 | 0.000 / 0.000 | 0.092 | 0.001 | 0.997 | 2013.259 |
| x2 | 0.923 / 0.911 | 0.000 / 0.000 | 0.098 | 0.002 | 0.996 | 2017.232 |
| x3 | 0.938 / 0.926 | 0.000 / 0.000 | 0.094 | 0.001 | 0.998 | 2015.815 |
| x4 | 0.936 / 0.925 | 0.000 / 0.000 | 0.094 | 0.001 | 0.998 | 2016.056 |
| x5 | 0.932 / 0.919 | 0.000 / 0.000 | 0.099 | 0.002 | 0.997 | 2015.149 |
| x6 | 0.935 / 0.918 | 0.000 / 0.000 | 0.097 | 0.001 | 0.998 | 2020.204 |
| x7 | 0.934 / 0.923 | 0.000 / 0.000 | 0.092 | 0.001 | 0.998 | 2018.227 |

## Checks

- Dev scoring integrity: 0 mismatches against the stored per-query metrics on 2,221 dev rows.
- Training-cache equality: every scored query's pool, seeds, golds, gold total, edge counts and float16 embedding equal the GNN's training cache on the fit carve (5,960 queries checked in its shards, 0 in its assemble) and on the select carve (1,497 and 0).
- Loopcheck: the carve loop against level 8's scoring pass on 24 dev smoke rows, 26 arrays compared, equal: True.
- Direction checks: passed on the dev rows and on every carve (9 carves).
- Repeat unit bit-identical: True.
- Units whose fit-set marginal log-likelihood fell between rounds: 0.
- Units whose kept theta sits at a clip bound: 0.

## What this does not say

Every arm is a measurement model on metaqa, fitted to metaqa train-split gold labels and read on V2_GATE dev rows. It is never a deployable or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature contract, M3, M4 or any selection. TW-1x is the only label-matched reading. TW-2x, TW-4x and TW-8x fit on more metaqa labels than the GNN had, which the GNN was not refitted on, so the data contrasts measure more labelled training data for this model, not a comparison at matched labels. Level 11's numbers were measured on a different population with dev-row training labels, and they are not one quantity with this file's. A deployable model of this form would need its own declaration, under the QLS-U contract on every dataset, with its compiled form timed.
