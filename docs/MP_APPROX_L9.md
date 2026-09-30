# MP-Approx level 9: non-backtracking typed walks and an exact residual, without message passing, on metaqa

Declaration: `configs/mp_approx_l9.yaml`. The script is `scripts/mp_approx_l9.py`. The record is `outputs/mp_approx_l9/record.json`.

## What was measured

On metaqa, level 8's non-message-passing typed-walk model was pushed at its weak points. The walks are non-backtracking (a walk never steps straight back to the node it came from). The type vector gains an exact learned residual per token sequence beside level 8's rotary text composition. The kappa and eta grid is wider, and EM runs 10 rounds. Every change has a paired contrast on the same queries, and level 8's own protocol is read nested inside every unit as the @L8 arms. Every arm is cross-fitted on 3000 fresh metaqa V2_GATE queries, disjoint from levels 0 to 8. rho is level 8's quantity on a fresh population. It is not the within-U_q oracle rho of levels 0 to 7, and the two are never one quantity.

## Reading

- The primary arm is NB-hyb, and its band is **L9_HIGH**.
- Readable metrics: recall@5, full_coverage@5, hit@1.
- Flags: EM_NOT_MONOTONE.
- The interpretation map entries that apply: l9_high, id_residual_adds, above_level8_primary, nb_raises_ceiling.

## Arms

rho_bar is the mean of rho over the readable metrics. Intervals are 95% bootstrap intervals over 1,000 query resamples. An @L8 arm is the same fit read under level 8's protocol (rounds 0 to 5, level 8's grid).

| arm | rho_bar | band | rho recall@5 | rho full_coverage@5 | rho hit@1 |
|---|---|---|---|---|---|
| NB-hyb | 0.757 [0.691, 0.815] | L9_HIGH | 0.739 [0.670, 0.806] | 0.732 [0.637, 0.818] | 0.800 [0.733, 0.868] |
| NB-hyb-mlp | 0.752 [0.690, 0.808] | L9_HIGH | 0.738 [0.672, 0.797] | 0.727 [0.641, 0.807] | 0.790 [0.724, 0.860] |
| NB-text | 0.631 [0.570, 0.688] | L9_MID | 0.638 [0.572, 0.705] | 0.622 [0.540, 0.706] | 0.635 [0.567, 0.699] |
| NB-id | 0.737 [0.676, 0.795] | L9_MID | 0.735 [0.671, 0.798] | 0.748 [0.657, 0.833] | 0.727 [0.656, 0.797] |
| NB-hash | 0.703 [0.648, 0.761] | L9_MID | 0.720 [0.660, 0.778] | 0.734 [0.657, 0.813] | 0.656 [0.592, 0.723] |
| STD-hyb | 0.724 [0.668, 0.778] | L9_MID | 0.756 [0.696, 0.813] | 0.739 [0.656, 0.827] | 0.677 [0.610, 0.744] |
| STD-text | 0.595 [0.542, 0.651] | L9_MID | 0.651 [0.592, 0.711] | 0.641 [0.560, 0.724] | 0.494 [0.425, 0.560] |
| STD-hash | 0.686 [0.630, 0.737] | L9_MID | 0.722 [0.665, 0.775] | 0.731 [0.650, 0.814] | 0.605 [0.537, 0.667] |
| NB-hyb@L8 | 0.744 [0.675, 0.805] | L9_MID | 0.733 [0.656, 0.802] | 0.715 [0.620, 0.803] | 0.784 [0.717, 0.853] |
| NB-hyb-mlp@L8 | 0.738 [0.677, 0.794] | L9_MID | 0.723 [0.656, 0.787] | 0.720 [0.629, 0.801] | 0.770 [0.703, 0.838] |
| NB-text@L8 | 0.580 [0.515, 0.642] | L9_MID | 0.607 [0.537, 0.679] | 0.604 [0.516, 0.692] | 0.528 [0.451, 0.597] |
| NB-id@L8 | 0.717 [0.654, 0.775] | L9_MID | 0.715 [0.645, 0.778] | 0.724 [0.637, 0.807] | 0.712 [0.638, 0.782] |
| NB-hash@L8 | 0.708 [0.650, 0.767] | L9_MID | 0.731 [0.668, 0.792] | 0.744 [0.664, 0.825] | 0.648 [0.576, 0.717] |
| STD-hyb@L8 | 0.709 [0.648, 0.765] | L9_MID | 0.735 [0.667, 0.801] | 0.727 [0.629, 0.818] | 0.664 [0.596, 0.729] |
| STD-text@L8 | 0.539 [0.480, 0.603] | L9_MID | 0.599 [0.533, 0.670] | 0.599 [0.512, 0.691] | 0.419 [0.343, 0.496] |
| STD-hash@L8 | 0.661 [0.602, 0.718] | L9_MID | 0.710 [0.645, 0.770] | 0.737 [0.652, 0.824] | 0.535 [0.461, 0.603] |
| NB-oracle | 1.069 [1.017, 1.122] | L9_ABOVE_GNN | 1.021 [0.965, 1.075] | 1.009 [0.931, 1.092] | 1.176 [1.112, 1.246] |
| STD-oracle | 0.991 [0.941, 1.045] | L9_HIGH | 0.991 [0.937, 1.045] | 0.993 [0.917, 1.077] | 0.989 [0.923, 1.056] |

The gap to the GNN is the mean over seeds and queries of M(arm) - M(G).

| arm | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| NB-hyb | -0.016 [-0.021, -0.012] BELOW_GNN | -0.017 [-0.024, -0.011] BELOW_GNN | -0.026 [-0.036, -0.017] BELOW_GNN |
| NB-hyb-mlp | -0.016 [-0.020, -0.012] BELOW_GNN | -0.018 [-0.024, -0.012] BELOW_GNN | -0.028 [-0.037, -0.018] BELOW_GNN |
| NB-text | -0.022 [-0.027, -0.018] BELOW_GNN | -0.024 [-0.030, -0.019] BELOW_GNN | -0.048 [-0.058, -0.038] BELOW_GNN |
| NB-id | -0.016 [-0.020, -0.012] BELOW_GNN | -0.016 [-0.022, -0.011] BELOW_GNN | -0.036 [-0.046, -0.026] BELOW_GNN |
| NB-hash | -0.017 [-0.021, -0.014] BELOW_GNN | -0.017 [-0.023, -0.012] BELOW_GNN | -0.045 [-0.055, -0.035] BELOW_GNN |
| STD-hyb | -0.015 [-0.019, -0.011] BELOW_GNN | -0.017 [-0.023, -0.011] BELOW_GNN | -0.042 [-0.052, -0.033] BELOW_GNN |
| STD-text | -0.022 [-0.026, -0.018] BELOW_GNN | -0.023 [-0.029, -0.017] BELOW_GNN | -0.067 [-0.078, -0.057] BELOW_GNN |
| STD-hash | -0.017 [-0.021, -0.013] BELOW_GNN | -0.017 [-0.023, -0.011] BELOW_GNN | -0.052 [-0.062, -0.042] BELOW_GNN |
| NB-hyb@L8 | -0.017 [-0.021, -0.012] BELOW_GNN | -0.018 [-0.025, -0.012] BELOW_GNN | -0.028 [-0.038, -0.019] BELOW_GNN |
| NB-hyb-mlp@L8 | -0.017 [-0.022, -0.013] BELOW_GNN | -0.018 [-0.025, -0.013] BELOW_GNN | -0.030 [-0.040, -0.020] BELOW_GNN |
| NB-text@L8 | -0.024 [-0.029, -0.020] BELOW_GNN | -0.025 [-0.032, -0.019] BELOW_GNN | -0.062 [-0.072, -0.052] BELOW_GNN |
| NB-id@L8 | -0.018 [-0.022, -0.014] BELOW_GNN | -0.018 [-0.024, -0.012] BELOW_GNN | -0.038 [-0.048, -0.028] BELOW_GNN |
| NB-hash@L8 | -0.017 [-0.021, -0.013] BELOW_GNN | -0.016 [-0.022, -0.011] BELOW_GNN | -0.046 [-0.056, -0.036] BELOW_GNN |
| STD-hyb@L8 | -0.016 [-0.021, -0.012] BELOW_GNN | -0.018 [-0.024, -0.011] BELOW_GNN | -0.044 [-0.055, -0.034] BELOW_GNN |
| STD-text@L8 | -0.025 [-0.029, -0.020] BELOW_GNN | -0.026 [-0.032, -0.019] BELOW_GNN | -0.076 [-0.088, -0.066] BELOW_GNN |
| STD-hash@L8 | -0.018 [-0.022, -0.014] BELOW_GNN | -0.017 [-0.023, -0.011] BELOW_GNN | -0.061 [-0.071, -0.051] BELOW_GNN |
| NB-oracle | 0.001 [-0.002, 0.005] | 0.001 [-0.005, 0.006] | 0.023 [0.015, 0.031] BEATS_GNN |
| STD-oracle | -0.001 [-0.004, 0.003] | -0.000 [-0.006, 0.005] | -0.001 [-0.010, 0.007] |

Descriptive means over seeds and queries (the twin and the GNN are the stored values):

| arm | recall@5 (arm / twin / GNN) | full_coverage@5 | hit@1 |
|---|---|---|---|
| NB-hyb | 0.785 / 0.740 / 0.802 | 0.675 / 0.627 / 0.692 | 0.885 / 0.780 / 0.911 |
| NB-hyb-mlp | 0.785 / 0.740 / 0.802 | 0.674 / 0.627 / 0.692 | 0.883 / 0.780 / 0.911 |
| NB-text | 0.779 / 0.740 / 0.802 | 0.667 / 0.627 / 0.692 | 0.863 / 0.780 / 0.911 |
| NB-id | 0.785 / 0.740 / 0.802 | 0.676 / 0.627 / 0.692 | 0.875 / 0.780 / 0.911 |
| NB-hash | 0.784 / 0.740 / 0.802 | 0.675 / 0.627 / 0.692 | 0.866 / 0.780 / 0.911 |
| STD-hyb | 0.786 / 0.740 / 0.802 | 0.675 / 0.627 / 0.692 | 0.869 / 0.780 / 0.911 |
| STD-text | 0.780 / 0.740 / 0.802 | 0.669 / 0.627 / 0.692 | 0.845 / 0.780 / 0.911 |
| STD-hash | 0.784 / 0.740 / 0.802 | 0.674 / 0.627 / 0.692 | 0.859 / 0.780 / 0.911 |
| NB-hyb@L8 | 0.785 / 0.740 / 0.802 | 0.673 / 0.627 / 0.692 | 0.883 / 0.780 / 0.911 |
| NB-hyb-mlp@L8 | 0.784 / 0.740 / 0.802 | 0.674 / 0.627 / 0.692 | 0.881 / 0.780 / 0.911 |
| NB-text@L8 | 0.777 / 0.740 / 0.802 | 0.666 / 0.627 / 0.692 | 0.849 / 0.780 / 0.911 |
| NB-id@L8 | 0.784 / 0.740 / 0.802 | 0.674 / 0.627 / 0.692 | 0.873 / 0.780 / 0.911 |
| NB-hash@L8 | 0.785 / 0.740 / 0.802 | 0.675 / 0.627 / 0.692 | 0.865 / 0.780 / 0.911 |
| STD-hyb@L8 | 0.785 / 0.740 / 0.802 | 0.674 / 0.627 / 0.692 | 0.867 / 0.780 / 0.911 |
| STD-text@L8 | 0.777 / 0.740 / 0.802 | 0.666 / 0.627 / 0.692 | 0.835 / 0.780 / 0.911 |
| STD-hash@L8 | 0.784 / 0.740 / 0.802 | 0.675 / 0.627 / 0.692 | 0.850 / 0.780 / 0.911 |
| NB-oracle | 0.803 / 0.740 / 0.802 | 0.692 / 0.627 / 0.692 | 0.934 / 0.780 / 0.911 |
| STD-oracle | 0.801 / 0.740 / 0.802 | 0.691 / 0.627 / 0.692 | 0.910 / 0.780 / 0.911 |

## Denominators

| metric | mean M(G) - M(T) | readable |
|---|---|---|
| recall@5 | 0.062 [0.056, 0.068] | True |
| full_coverage@5 | 0.064 [0.056, 0.072] | True |
| hit@1 | 0.132 [0.120, 0.144] | True |

## Contrasts

| contrast | of | paired difference |
|---|---|---|
| nb_adds | rho_bar(NB-hyb) - rho_bar(STD-hyb) | 0.033 [-0.006, 0.075] |
| nb_adds_hash | rho_bar(NB-hash) - rho_bar(STD-hash) | 0.018 [-0.016, 0.052] |
| id_residual_adds | rho_bar(NB-hyb) - rho_bar(NB-text) | 0.126 [0.088, 0.163] |
| text_residual_adds | rho_bar(NB-hyb) - rho_bar(NB-id) | 0.020 [-0.018, 0.060] |
| id_vs_hash | rho_bar(NB-id) - rho_bar(NB-hash) | 0.033 [-0.007, 0.070] |
| query_mlp_adds | rho_bar(NB-hyb-mlp) - rho_bar(NB-hyb) | -0.005 [-0.034, 0.021] |
| protocol_adds | rho_bar(NB-hyb) - rho_bar(NB-hyb@L8) | 0.013 [-0.002, 0.031] |
| over_level8_primary | rho_bar(NB-hyb) - rho_bar(STD-text@L8) | 0.218 [0.168, 0.269] |
| ceiling_gap | rho_bar(NB-oracle) - rho_bar(NB-hyb) | 0.312 [0.258, 0.367] |
| nb_ceiling | rho_bar(NB-oracle) - rho_bar(STD-oracle) | 0.078 [0.052, 0.104] |

## By hop

rho_bar and band per hop, each hop read on its own readable metrics.

| arm | hop=1 (1000 queries; recall@5, full_coverage@5, hit@1) | hop=2 (1000 queries; recall@5, full_coverage@5, hit@1) | hop=3 (1000 queries; recall@5, full_coverage@5, hit@1) |
|---|---|---|---|
| NB-hyb | 0.626 [0.334, 0.859] L9_MID | 0.956 [0.819, 1.093] L9_HIGH | 0.725 [0.653, 0.793] L9_MID |
| NB-hyb-mlp | 0.725 [0.493, 0.953] L9_MID | 0.964 [0.835, 1.107] L9_HIGH | 0.703 [0.632, 0.771] L9_MID |
| NB-text | 0.684 [0.464, 0.876] L9_MID | 0.791 [0.671, 0.915] L9_HIGH | 0.583 [0.513, 0.651] L9_MID |
| NB-id | 0.491 [0.131, 0.731] L9_MID | 0.958 [0.815, 1.108] L9_HIGH | 0.716 [0.645, 0.789] L9_MID |
| NB-hash | 0.608 [0.371, 0.795] L9_MID | 0.955 [0.833, 1.097] L9_HIGH | 0.651 [0.592, 0.721] L9_MID |
| STD-hyb | 0.769 [0.566, 0.987] L9_HIGH | 0.947 [0.821, 1.092] L9_HIGH | 0.662 [0.602, 0.728] L9_MID |
| STD-text | 0.741 [0.530, 0.941] L9_MID | 0.726 [0.593, 0.855] L9_MID | 0.538 [0.472, 0.601] L9_MID |
| STD-hash | 0.796 [0.640, 0.988] L9_HIGH | 1.007 [0.895, 1.139] L9_HIGH | 0.586 [0.525, 0.648] L9_MID |
| STD-text@L8 | 0.606 [0.307, 0.834] L9_MID | 0.649 [0.468, 0.804] L9_MID | 0.496 [0.429, 0.566] L9_MID |
| NB-hyb@L8 | 0.572 [0.206, 0.828] L9_MID | 0.937 [0.794, 1.081] L9_HIGH | 0.716 [0.640, 0.783] L9_MID |
| NB-oracle | 1.122 [0.958, 1.464] L9_HIGH | 1.307 [1.181, 1.465] L9_ABOVE_GNN | 1.004 [0.946, 1.067] L9_HIGH |
| STD-oracle | 1.122 [0.958, 1.464] L9_HIGH | 1.305 [1.179, 1.463] L9_ABOVE_GNN | 0.890 [0.833, 0.953] L9_HIGH |

## Anchors (descriptive)

| family | chain reaches a gold (either bucket) | swapped | R* recall | R* precision | b0 seed in R* | gold b0 seed in R* | types / query | entries / query |
|---|---|---|---|---|---|---|---|---|
| std | 0.940 | 0.000 | 0.923 | 0.825 | 0.131 | 0.002 | 202.169 | 5381.634 |
| nb | 0.929 | 0.000 | 0.905 | 0.876 | 0.002 | 0.001 | 177.366 | 5044.591 |

- The non-backtracking rule removes 0.093 of the standard R* on average (by hop: 0.000, 0.101, 0.177), and 0.019 of the in-pool golds (by hop: 0.000, 0.000, 0.056).
- Topic entity: in the pool 0.997, among the seeds 0.983, in b0 0.937. Queries with an in-pool gold: 0.989.
- The argmax type has the true chain's tokens on this share of queries (the mean over seeds; then hop 1, 2, 3):
  - NB-hyb: 0.846 (0.916, 0.864, 0.759)
  - NB-hyb-mlp: 0.845 (0.917, 0.861, 0.757)
  - NB-text: 0.733 (0.824, 0.742, 0.632)
  - NB-id: 0.843 (0.942, 0.858, 0.728)
  - NB-hash: 0.737 (0.836, 0.840, 0.534)
  - STD-hyb: 0.836 (0.819, 0.859, 0.829)
  - STD-text: 0.644 (0.524, 0.734, 0.673)
  - STD-hash: 0.733 (0.819, 0.829, 0.552)
- NMI between NB-hyb's argmax token sequence and the qtype (k = 0): 0.892.
- Grid edges and kept rounds per arm, under the wide protocol / level 8's protocol:
  - NB-hyb (15 units): kappa at an end 0 / 14 (low end 0 / 14), eta at an end 1 / 14 (high end 1 / 14), the last round kept 1 / 5
  - NB-hyb-mlp (15 units): kappa at an end 0 / 15 (low end 0 / 15), eta at an end 2 / 12 (high end 1 / 12), the last round kept 1 / 3
  - NB-text (15 units): kappa at an end 0 / 15 (low end 0 / 15), eta at an end 0 / 12 (high end 0 / 12), the last round kept 0 / 2
  - NB-id (15 units): kappa at an end 0 / 8 (low end 0 / 8), eta at an end 5 / 15 (high end 5 / 15), the last round kept 1 / 15
  - NB-hash (15 units): kappa at an end 0 / 14 (low end 0 / 14), eta at an end 1 / 14 (high end 1 / 14), the last round kept 3 / 15
  - STD-hyb (15 units): kappa at an end 0 / 14 (low end 0 / 14), eta at an end 1 / 13 (high end 1 / 13), the last round kept 1 / 11
  - STD-text (15 units): kappa at an end 0 / 13 (low end 0 / 13), eta at an end 1 / 13 (high end 1 / 13), the last round kept 0 / 8
  - STD-hash (15 units): kappa at an end 0 / 15 (low end 0 / 15), eta at an end 0 / 13 (high end 0 / 13), the last round kept 4 / 15

## Checks

- Scoring integrity: 0 mismatches against the stored per-query metrics on 3000 queries.
- Direction checks: std 0.940 against swapped 0.000; nb 0.929 against swapped 0.000.
- Repeat unit bit-identical: True.
- Units whose fit-set marginal log-likelihood fell between rounds: 7.

## What this does not say

Every arm is a cross-fitted measurement model on metaqa V2_GATE queries with gold labels. It is never a deployable or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature contract, M3, M4 or any selection. Level 8's numbers were measured on a different population, and the paired comparisons with level 8's procedure are the @L8 arms on this file's queries.
