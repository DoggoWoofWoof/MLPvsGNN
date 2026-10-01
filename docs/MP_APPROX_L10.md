# MP-Approx level 10: a set likelihood and a coverage score on non-backtracking typed walks, without message passing, on metaqa

Declaration: `configs/mp_approx_l10.yaml`. The scripts are `scripts/mp_approx_l10.py` (population, scoring pass, check) and `scripts/mp_approx_l10_fit.py` (fits, read, doc, file). The record is `outputs/mp_approx_l10/record.json`.

## What was measured

On metaqa, level 9's non-message-passing typed-walk model (non-backtracking walks, the rotary text composition plus an exact per-sequence residual, a linear query map, EM over the latent chain) was fitted two ways and scored two ways. The draw likelihood is level 8's: each gold is a draw from the type's reach set. The set likelihood treats every node of the reach set as gold with one fitted probability rho_L, and every other pool node with another, eps_s. The dens score is level 8's mixture of uniform reach sets. The cov score is the posterior's coverage: the probability that the query's chosen walk type reaches the node. Every arm is cross-fitted on 3000 fresh metaqa V2_GATE queries, disjoint from levels 0 to 9. NB-draw-dens is level 9's primary procedure on these queries. rho is level 8's quantity on a fresh population. It is not the within-U_q oracle rho of levels 0 to 7, and the two are never one quantity.

## Reading

- The primary arm is NB-set-cov, and its band is **L10_MID**.
- Readable metrics: recall@5, full_coverage@5, hit@1.
- Flags: GRID_EDGE.
- The interpretation map entries that apply: l10_mid, set_adds, nb_adds.

## Arms

rho_bar is the mean of rho over the readable metrics. Intervals are 95% bootstrap intervals over 1,000 query resamples. An arm's name is its fit, then its score.

| arm | rho_bar | band | rho recall@5 | rho full_coverage@5 | rho hit@1 |
|---|---|---|---|---|---|
| NB-set-cov | 0.719 [0.665, 0.772] | L10_MID | 0.705 [0.644, 0.762] | 0.662 [0.579, 0.734] | 0.791 [0.729, 0.860] |
| NB-set-dens | 0.718 [0.665, 0.770] | L10_MID | 0.725 [0.667, 0.781] | 0.706 [0.632, 0.777] | 0.722 [0.658, 0.788] |
| NB-draw-cov | 0.681 [0.620, 0.743] | L10_MID | 0.637 [0.570, 0.702] | 0.610 [0.525, 0.689] | 0.795 [0.727, 0.870] |
| NB-draw-dens | 0.699 [0.646, 0.756] | L10_MID | 0.703 [0.644, 0.763] | 0.689 [0.612, 0.766] | 0.707 [0.642, 0.780] |
| STD-set-cov | 0.665 [0.611, 0.720] | L10_MID | 0.672 [0.610, 0.729] | 0.631 [0.547, 0.709] | 0.692 [0.627, 0.758] |
| STD-set-dens | 0.643 [0.591, 0.696] | L10_MID | 0.679 [0.622, 0.736] | 0.680 [0.604, 0.756] | 0.568 [0.504, 0.635] |
| NB-oracle | 1.049 [0.993, 1.102] | L10_HIGH | 0.998 [0.943, 1.054] | 0.979 [0.906, 1.052] | 1.168 [1.097, 1.239] |
| STD-oracle | 0.948 [0.895, 1.002] | L10_HIGH | 0.951 [0.896, 1.003] | 0.904 [0.825, 0.984] | 0.989 [0.920, 1.068] |

The gap to the GNN is the mean over seeds and queries of M(arm) - M(G).

| arm | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| NB-set-cov | -0.018 [-0.022, -0.014] BELOW_GNN | -0.022 [-0.028, -0.017] BELOW_GNN | -0.026 [-0.035, -0.017] BELOW_GNN |
| NB-set-dens | -0.017 [-0.021, -0.013] BELOW_GNN | -0.019 [-0.025, -0.014] BELOW_GNN | -0.035 [-0.044, -0.026] BELOW_GNN |
| NB-draw-cov | -0.022 [-0.026, -0.018] BELOW_GNN | -0.025 [-0.032, -0.020] BELOW_GNN | -0.026 [-0.035, -0.016] BELOW_GNN |
| NB-draw-dens | -0.018 [-0.022, -0.014] BELOW_GNN | -0.020 [-0.026, -0.014] BELOW_GNN | -0.037 [-0.046, -0.027] BELOW_GNN |
| STD-set-cov | -0.020 [-0.024, -0.016] BELOW_GNN | -0.024 [-0.031, -0.018] BELOW_GNN | -0.039 [-0.048, -0.029] BELOW_GNN |
| STD-set-dens | -0.019 [-0.024, -0.015] BELOW_GNN | -0.021 [-0.027, -0.015] BELOW_GNN | -0.054 [-0.064, -0.044] BELOW_GNN |
| NB-oracle | -0.000 [-0.004, 0.003] | -0.001 [-0.006, 0.003] | 0.021 [0.013, 0.029] BEATS_GNN |
| STD-oracle | -0.003 [-0.007, 0.000] | -0.006 [-0.012, -0.001] BELOW_GNN | -0.001 [-0.011, 0.008] |

Descriptive means over seeds and queries (the twin and the GNN are the stored values):

| arm | recall@5 (arm / twin / GNN) | full_coverage@5 | hit@1 |
|---|---|---|---|
| NB-set-cov | 0.778 / 0.736 / 0.796 | 0.663 / 0.620 / 0.685 | 0.887 / 0.788 / 0.914 |
| NB-set-dens | 0.779 / 0.736 / 0.796 | 0.666 / 0.620 / 0.685 | 0.879 / 0.788 / 0.914 |
| NB-draw-cov | 0.774 / 0.736 / 0.796 | 0.660 / 0.620 / 0.685 | 0.888 / 0.788 / 0.914 |
| NB-draw-dens | 0.778 / 0.736 / 0.796 | 0.665 / 0.620 / 0.685 | 0.877 / 0.788 / 0.914 |
| STD-set-cov | 0.776 / 0.736 / 0.796 | 0.661 / 0.620 / 0.685 | 0.875 / 0.788 / 0.914 |
| STD-set-dens | 0.777 / 0.736 / 0.796 | 0.664 / 0.620 / 0.685 | 0.859 / 0.788 / 0.914 |
| NB-oracle | 0.796 / 0.736 / 0.796 | 0.684 / 0.620 / 0.685 | 0.935 / 0.788 / 0.914 |
| STD-oracle | 0.793 / 0.736 / 0.796 | 0.679 / 0.620 / 0.685 | 0.912 / 0.788 / 0.914 |

## Denominators

| metric | mean M(G) - M(T) | readable |
|---|---|---|
| recall@5 | 0.061 [0.055, 0.067] | True |
| full_coverage@5 | 0.065 [0.056, 0.074] | True |
| hit@1 | 0.126 [0.115, 0.138] | True |

## Contrasts

| contrast | of | paired difference |
|---|---|---|
| set_adds | rho_bar(NB-set-cov) - rho_bar(NB-draw-cov) | 0.039 [0.006, 0.074] |
| set_adds_dens | rho_bar(NB-set-dens) - rho_bar(NB-draw-dens) | 0.018 [-0.008, 0.046] |
| cov_adds | rho_bar(NB-set-cov) - rho_bar(NB-set-dens) | 0.002 [-0.025, 0.028] |
| cov_adds_draw | rho_bar(NB-draw-cov) - rho_bar(NB-draw-dens) | -0.019 [-0.052, 0.014] |
| over_level9_primary | rho_bar(NB-set-cov) - rho_bar(NB-draw-dens) | 0.020 [-0.010, 0.048] |
| nb_adds | rho_bar(NB-set-cov) - rho_bar(STD-set-cov) | 0.055 [0.017, 0.090] |
| ceiling_gap | rho_bar(NB-oracle) - rho_bar(NB-set-cov) | 0.329 [0.277, 0.379] |

## By hop

rho_bar and band per hop, each hop read on its own readable metrics.

| arm | hop=1 (1000 queries; recall@5, full_coverage@5, hit@1) | hop=2 (1000 queries; recall@5, full_coverage@5, hit@1) | hop=3 (1000 queries; recall@5, full_coverage@5, hit@1) |
|---|---|---|---|
| NB-set-cov | 0.675 [0.497, 0.898] L10_MID | 0.782 [0.653, 0.919] L10_HIGH | 0.710 [0.648, 0.772] L10_MID |
| NB-set-dens | 0.591 [0.415, 0.807] L10_MID | 0.783 [0.655, 0.922] L10_HIGH | 0.720 [0.657, 0.781] L10_MID |
| NB-draw-cov | 0.619 [0.384, 0.844] L10_MID | 0.859 [0.716, 0.998] L10_HIGH | 0.643 [0.578, 0.710] L10_MID |
| NB-draw-dens | 0.694 [0.527, 0.936] L10_MID | 0.869 [0.738, 1.014] L10_HIGH | 0.656 [0.590, 0.720] L10_MID |
| STD-set-cov | 0.719 [0.530, 0.936] L10_MID | 0.807 [0.682, 0.946] L10_HIGH | 0.619 [0.559, 0.679] L10_MID |
| STD-set-dens | 0.666 [0.496, 0.892] L10_MID | 0.769 [0.649, 0.902] L10_HIGH | 0.605 [0.546, 0.665] L10_MID |
| NB-oracle | 1.200 [0.982, 1.547] L10_HIGH | 1.120 [0.973, 1.275] L10_HIGH | 1.007 [0.950, 1.065] L10_HIGH |
| STD-oracle | 1.200 [0.982, 1.547] L10_HIGH | 1.109 [0.962, 1.263] L10_HIGH | 0.866 [0.804, 0.926] L10_HIGH |

## Anchors (descriptive)

- The argmax type has the true chain's tokens on this share of queries (the mean over seeds; hop 1, 2, 3; genre-ending questions, 474 of them):
  - NB-draw: 0.809 (0.873, 0.827, 0.726; genre-ending 0.262)
  - NB-set: 0.868 (0.914, 0.862, 0.828; genre-ending 0.572)
  - STD-set: 0.870 (0.888, 0.869, 0.852; genre-ending 0.490)
- The ceiling gap (NB-oracle - arm) split by whether the fit's argmax chain is the true chain, in rho_bar units (the hops are contributions to the whole and sum to it):
  - NB-set-cov: argmax right on 7812 (query, seed) pairs, 0.211 (by hop 0.038, 0.040, 0.132); wrong on 1188, 0.119 (by hop 0.015, 0.024, 0.080)
  - NB-draw-dens: argmax right on 7281 (query, seed) pairs, 0.184 (by hop 0.036, 0.035, 0.113); wrong on 1719, 0.165 (by hop 0.015, 0.013, 0.136)
- The set likelihood's kept theta (rho_1, rho_2, rho_3, eps_s), the mean over units, then the range:
  - NB-set: mean 0.873, 0.9708, 0.8454, 9.945e-05; min 0.8585, 0.9667, 0.825, 8.466e-05; max 0.8877, 0.9748, 0.8756, 0.000119
  - STD-set: mean 0.9189, 0.9072, 0.7707, 7.081e-05; min 0.9046, 0.9006, 0.7491, 6.328e-05; max 0.9291, 0.9114, 0.7915, 8.13e-05
- NMI between NB-set-cov's argmax token sequence and the qtype (k = 0): 0.889.
- Grid edges and kept rounds per arm:
  - NB-set-cov (15 units): kappa at an end 0 (low 0, high 0), eta at an end 8 (low 0, high 8), the last round kept 0
  - NB-set-dens (15 units): kappa at an end 0 (low 0, high 0), eta at an end 4 (low 1, high 3), the last round kept 0
  - NB-draw-cov (15 units): kappa at an end 0 (low 0, high 0), eta at an end 5 (low 0, high 5), the last round kept 0
  - NB-draw-dens (15 units): kappa at an end 0 (low 0, high 0), eta at an end 4 (low 0, high 4), the last round kept 0
  - STD-set-cov (15 units): kappa at an end 0 (low 0, high 0), eta at an end 6 (low 0, high 6), the last round kept 0
  - STD-set-dens (15 units): kappa at an end 0 (low 0, high 0), eta at an end 3 (low 0, high 3), the last round kept 0

| family | chain reaches a gold (either bucket) | swapped | R* recall | R* precision | types / query | entries / query |
|---|---|---|---|---|---|---|
| std | 0.938 | 0.000 | 0.913 | 0.820 | 202.181 | 5400.793 |
| nb | 0.925 | 0.000 | 0.895 | 0.869 | 177.057 | 5066.022 |

- The non-backtracking rule removes 0.094 of the standard R* on average and 0.018 of the in-pool golds. In-pool golds that lie in no non-backtracking reach set: 0.001 (a descriptive anchor filed by the check stage beyond the declared list).
- Topic entity: in the pool 0.997, in b0 0.934. Queries with an in-pool gold: 0.990.

## Checks

- Scoring integrity: 0 mismatches against the stored per-query metrics on 3000 queries.
- Direction checks: std 0.938 against swapped 0.000; nb 0.925 against swapped 0.000.
- Repeat unit bit-identical: True.
- Units whose fit-set marginal log-likelihood fell between rounds: 0.
- Set units whose kept theta sits at a clip bound: 0.

## What this does not say

Every arm is a cross-fitted measurement model on metaqa V2_GATE queries with gold labels. It is never a deployable or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature contract, M3, M4 or any selection. Level 9's numbers were measured on a different population; the paired comparison with level 9's procedure is NB-draw-dens on this file's queries.
