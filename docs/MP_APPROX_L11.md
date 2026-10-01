# MP-Approx level 11: more gold-labelled training rows and a sharpened mixture for a typed-walk model without message passing, on metaqa

Declaration: `configs/mp_approx_l11.yaml`. The scripts are `scripts/mp_approx_l11.py` (population, scoring pass, check) and `scripts/mp_approx_l11_fit.py` (fits, read, doc, file). The record is `outputs/mp_approx_l11/record.json`.

## What was measured

On metaqa, level 10's non-message-passing typed-walk model (level 9's NB-hyb model on non-backtracking walks, with level 10's set likelihood and EM over the latent chain) was fitted two ways and scored two ways. NB-set fits on this file's rows only, as level 10 did. NB-setX fits on the same rows plus those of level 9's and level 10's 6,000 compiled rows that have an in-pool gold; they enter as fit and inner-validation rows only and are never scored. NB-set's units fit on 2,153 gold-labelled queries on average, with 225 inner-validation queries. NB-setX's units fit on 7,473 (2,153 of this file's, 2,666 of level 9's, 2,654 of level 10's), with 842 inner-validation queries. The cov score is level 10's coverage. The dsh score is level 8's mixture under the posterior raised to a power beta and renormalised, with beta chosen together with kappa and eta on the inner queries; at beta = 1 it is level 10's dens score. Every arm is cross-fitted on 3000 fresh metaqa V2_GATE queries, disjoint from levels 0 to 10. NB-set-cov is level 10's primary procedure on these queries. rho is level 8's quantity on a fresh population. It is not the within-U_q oracle rho of levels 0 to 7, and the two are never one quantity.

## Reading

- The primary arm is NB-setX-dsh, and its band is **L11_HIGH**.
- Readable metrics: recall@5, full_coverage@5, hit@1.
- Flags: none.
- The interpretation map entries that apply: l11_high, data_adds, above_level10_primary.

## Arms

rho_bar is the mean of rho over the readable metrics. Intervals are 95% bootstrap intervals over 1,000 query resamples. An arm's name is its fit, then its score.

| arm | rho_bar | band | rho recall@5 | rho full_coverage@5 | rho hit@1 |
|---|---|---|---|---|---|
| NB-setX-dsh | 0.879 [0.825, 0.930] | L11_HIGH | 0.877 [0.814, 0.936] | 0.851 [0.779, 0.925] | 0.909 [0.846, 0.978] |
| NB-setX-cov | 0.872 [0.819, 0.928] | L11_HIGH | 0.870 [0.812, 0.928] | 0.845 [0.776, 0.924] | 0.900 [0.835, 0.971] |
| NB-set-dsh | 0.796 [0.744, 0.852] | L11_HIGH | 0.812 [0.752, 0.875] | 0.777 [0.704, 0.856] | 0.798 [0.731, 0.864] |
| NB-set-cov | 0.772 [0.722, 0.829] | L11_HIGH | 0.787 [0.728, 0.848] | 0.731 [0.656, 0.808] | 0.799 [0.736, 0.867] |
| NB-oracle | 1.078 [1.024, 1.132] | L11_ABOVE_GNN | 1.032 [0.978, 1.086] | 1.012 [0.934, 1.094] | 1.189 [1.118, 1.258] |

The gap to the GNN is the mean over seeds and queries of M(arm) - M(G).

| arm | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| NB-setX-dsh | -0.008 [-0.012, -0.004] BELOW_GNN | -0.011 [-0.016, -0.005] BELOW_GNN | -0.012 [-0.020, -0.003] BELOW_GNN |
| NB-setX-cov | -0.008 [-0.013, -0.005] BELOW_GNN | -0.011 [-0.017, -0.005] BELOW_GNN | -0.013 [-0.022, -0.004] BELOW_GNN |
| NB-set-dsh | -0.012 [-0.017, -0.008] BELOW_GNN | -0.016 [-0.022, -0.010] BELOW_GNN | -0.026 [-0.035, -0.017] BELOW_GNN |
| NB-set-cov | -0.014 [-0.018, -0.010] BELOW_GNN | -0.019 [-0.025, -0.013] BELOW_GNN | -0.026 [-0.035, -0.016] BELOW_GNN |
| NB-oracle | 0.002 [-0.001, 0.005] | 0.001 [-0.005, 0.006] | 0.024 [0.016, 0.032] BEATS_GNN |

Descriptive means over seeds and queries (the twin and the GNN are the stored values):

| arm | recall@5 (arm / twin / GNN) | full_coverage@5 | hit@1 |
|---|---|---|---|
| NB-setX-dsh | 0.796 / 0.739 / 0.804 | 0.678 / 0.617 / 0.689 | 0.898 / 0.780 / 0.909 |
| NB-setX-cov | 0.796 / 0.739 / 0.804 | 0.678 / 0.617 / 0.689 | 0.896 / 0.780 / 0.909 |
| NB-set-dsh | 0.792 / 0.739 / 0.804 | 0.673 / 0.617 / 0.689 | 0.883 / 0.780 / 0.909 |
| NB-set-cov | 0.790 / 0.739 / 0.804 | 0.669 / 0.617 / 0.689 | 0.883 / 0.780 / 0.909 |
| NB-oracle | 0.806 / 0.739 / 0.804 | 0.690 / 0.617 / 0.689 | 0.934 / 0.780 / 0.909 |

## Denominators

| metric | mean M(G) - M(T) | readable |
|---|---|---|
| recall@5 | 0.065 [0.059, 0.071] | True |
| full_coverage@5 | 0.072 [0.063, 0.081] | True |
| hit@1 | 0.129 [0.118, 0.141] | True |

## Contrasts

| contrast | of | paired difference |
|---|---|---|
| data_adds | rho_bar(NB-setX-dsh) - rho_bar(NB-set-dsh) | 0.083 [0.049, 0.117] |
| data_adds_cov | rho_bar(NB-setX-cov) - rho_bar(NB-set-cov) | 0.099 [0.067, 0.134] |
| dsh_adds | rho_bar(NB-setX-dsh) - rho_bar(NB-setX-cov) | 0.007 [-0.013, 0.027] |
| dsh_adds_own | rho_bar(NB-set-dsh) - rho_bar(NB-set-cov) | 0.023 [-0.003, 0.049] |
| over_level10_primary | rho_bar(NB-setX-dsh) - rho_bar(NB-set-cov) | 0.107 [0.073, 0.142] |
| ceiling_gap | rho_bar(NB-oracle) - rho_bar(NB-setX-dsh) | 0.199 [0.160, 0.242] |

## By hop

rho_bar and band per hop, each hop read on its own readable metrics.

| arm | hop=1 (1000 queries; recall@5, full_coverage@5, hit@1) | hop=2 (1000 queries; recall@5, full_coverage@5, hit@1) | hop=3 (1000 queries; recall@5, full_coverage@5, hit@1) |
|---|---|---|---|
| NB-setX-dsh | 0.799 [0.684, 0.928] L11_HIGH | 0.938 [0.798, 1.077] L11_HIGH | 0.879 [0.815, 0.940] L11_HIGH |
| NB-setX-cov | 0.822 [0.708, 0.947] L11_HIGH | 0.927 [0.788, 1.067] L11_HIGH | 0.865 [0.798, 0.934] L11_HIGH |
| NB-set-dsh | 0.673 [0.543, 0.806] L11_MID | 0.826 [0.682, 0.978] L11_HIGH | 0.813 [0.750, 0.880] L11_HIGH |
| NB-set-cov | 0.678 [0.546, 0.812] L11_MID | 0.757 [0.598, 0.905] L11_HIGH | 0.795 [0.730, 0.864] L11_HIGH |
| NB-oracle | 1.086 [0.966, 1.231] L11_HIGH | 1.211 [1.061, 1.383] L11_ABOVE_GNN | 1.035 [0.978, 1.100] L11_HIGH |

## Anchors (descriptive)

- Training rows per unit (the mean over units, then the range of the total):
  - NB-set: fit 2,153 (own 2,153; 2,108 to 2,196), inner 225 (211 to 234)
  - NB-setX: fit 7,473 (own 2,153, level9 2,666, level10 2,654; 7,428 to 7,516), inner 842 (828 to 851)
- The argmax type has the true chain's tokens on this share of queries (the mean over seeds; hop 1, 2, 3; genre-ending questions, 464 of them):
  - NB-set: 0.875 (0.928, 0.867, 0.831; genre-ending 0.596)
  - NB-setX: 0.894 (0.946, 0.883, 0.854; genre-ending 0.586)
- The ceiling gap (NB-oracle - arm) split by whether the fit's argmax chain is the true chain, in rho_bar units (the hops are contributions to the whole and sum to it):
  - NB-setX-dsh: argmax right on 8050 (query, seed) pairs, 0.142 (by hop 0.031, 0.031, 0.080); wrong on 950, 0.056 (by hop 0.013, 0.019, 0.024)
  - NB-set-cov: argmax right on 7877 (query, seed) pairs, 0.179 (by hop 0.035, 0.042, 0.102); wrong on 1123, 0.126 (by hop 0.028, 0.042, 0.056)
- The set likelihood's kept theta (rho_1, rho_2, rho_3, eps_s), the mean over units, then the range:
  - NB-set: mean 0.8615, 0.9684, 0.8729, 0.0001026; min 0.8486, 0.9619, 0.847, 9.409e-05; max 0.875, 0.9734, 0.8899, 0.0001121
  - NB-setX: mean 0.8742, 0.9631, 0.8708, 0.000108; min 0.8703, 0.961, 0.8449, 9.621e-05; max 0.877, 0.966, 0.8893, 0.0001173
- The dsh score's beta per unit (the number of units choosing each value):
  - NB-setX-dsh: 1.0 0, 1.5 1, 2.0 2, 3.0 3, 4.0 6, 8.0 3, inf 0
  - NB-set-dsh: 1.0 0, 1.5 0, 2.0 2, 3.0 4, 4.0 4, 8.0 4, inf 1
- NMI between NB-setX-dsh's argmax token sequence and the qtype (k = 0): 0.916.
- Grid edges and kept rounds per arm:
  - NB-setX-dsh (15 units): kappa at an end 0 (low 0, high 0), eta at an end 3 (low 3, high 0), the last round kept 1
  - NB-setX-cov (15 units): kappa at an end 0 (low 0, high 0), eta at an end 8 (low 0, high 8), the last round kept 1
  - NB-set-dsh (15 units): kappa at an end 0 (low 0, high 0), eta at an end 4 (low 4, high 0), the last round kept 0
  - NB-set-cov (15 units): kappa at an end 0 (low 0, high 0), eta at an end 10 (low 0, high 10), the last round kept 0

| family | chain reaches a gold (either bucket) | swapped | R* recall | R* precision | types / query | entries / query |
|---|---|---|---|---|---|---|
| std | 0.936 | 0.000 | 0.913 | 0.819 | 201.837 | 5367.672 |
| nb | 0.920 | 0.000 | 0.890 | 0.865 | 176.815 | 5030.312 |

- The non-backtracking rule removes 0.096 of the standard R* on average and 0.023 of the in-pool golds. In-pool golds that lie in no non-backtracking reach set: 0.001 (a descriptive anchor filed by the check stage beyond the declared list).
- Topic entity: in the pool 0.997, in b0 0.933. Queries with an in-pool gold: 0.991.

## Checks

- Scoring integrity: 0 mismatches against the stored per-query metrics on 3000 queries.
- Direction checks: std 0.936 against swapped 0.000; nb 0.920 against swapped 0.000.
- Repeat unit bit-identical: True.
- Units whose fit-set marginal log-likelihood fell between rounds: 0.
- Set units whose kept theta sits at a clip bound: 0.

## What this does not say

Every arm is a cross-fitted measurement model on metaqa V2_GATE queries with gold labels. It is never a deployable or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature contract, M3, M4 or any selection. NB-setX fits on more gold-labelled queries than NB-set (the training rows above), so data_adds measures more labelled training data for the same model, not a different model. Level 10's numbers were measured on a different population; the paired comparison with level 10's procedure is NB-set-cov on this file's queries.
