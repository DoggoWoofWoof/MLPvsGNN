# MP-Approx level 8: a typed-walk model without message passing, on metaqa

Declaration: `configs/mp_approx_l8.yaml`. The script is `scripts/mp_approx_l8.py`. The record is `outputs/mp_approx_l8/record.json`.

## What was measured

On metaqa, a model without message passing reads compiled typed walks from the query's seeds as fixed counts. It learns only which relation chain the question asks for, and it is fitted to gold labels. This file measures how much of the pilot GNN's gain over its twin that model recovers. Every arm is cross-fitted on 3000 metaqa V2_GATE queries, disjoint from levels 0 to 7. rho here is not the within-U_q oracle rho of levels 0 to 7, and the two are never one quantity.

## Reading

- The primary arm is TP, and its band is **L8_MID**.
- Readable metrics: recall@5, full_coverage@5, hit@1.
- Flags: none.
- The interpretation map entries that apply: l8_mid, order_matters, latent_beats_direct.

## Arms

rho_bar is the mean of rho over the readable metrics. Intervals are 95% bootstrap intervals over 1,000 query resamples.

| arm | rho_bar | band | rho recall@5 | rho full_coverage@5 | rho hit@1 |
|---|---|---|---|---|---|
| TP | 0.529 [0.471, 0.591] | L8_MID | 0.604 [0.541, 0.668] | 0.627 [0.550, 0.707] | 0.355 [0.272, 0.438] |
| TP-bag | 0.260 [0.185, 0.332] | L8_MID | 0.338 [0.258, 0.409] | 0.464 [0.370, 0.552] | -0.022 [-0.129, 0.080] |
| TP-pos | 0.537 [0.478, 0.594] | L8_MID | 0.599 [0.538, 0.657] | 0.624 [0.548, 0.705] | 0.387 [0.307, 0.468] |
| TP-hash | 0.630 [0.568, 0.688] | L8_MID | 0.683 [0.619, 0.745] | 0.721 [0.639, 0.794] | 0.486 [0.402, 0.563] |
| TP-direct | -0.717 [-0.868, -0.588] | L8_LOW | -0.575 [-0.732, -0.439] | -0.585 [-0.765, -0.426] | -0.991 [-1.208, -0.805] |
| TP-oracle | 1.007 [0.954, 1.063] | L8_HIGH | 1.006 [0.951, 1.065] | 1.008 [0.939, 1.089] | 1.008 [0.939, 1.083] |

The gap to the GNN is the mean over seeds and queries of M(arm) - M(G).

| arm | recall@5 | full_coverage@5 | hit@1 |
|---|---|---|---|
| TP | -0.024 [-0.029, -0.020] BELOW_GNN | -0.026 [-0.033, -0.020] BELOW_GNN | -0.076 [-0.087, -0.066] BELOW_GNN |
| TP-bag | -0.041 [-0.046, -0.036] BELOW_GNN | -0.038 [-0.045, -0.031] BELOW_GNN | -0.120 [-0.133, -0.109] BELOW_GNN |
| TP-pos | -0.025 [-0.029, -0.020] BELOW_GNN | -0.027 [-0.033, -0.020] BELOW_GNN | -0.072 [-0.083, -0.063] BELOW_GNN |
| TP-hash | -0.019 [-0.024, -0.015] BELOW_GNN | -0.020 [-0.026, -0.014] BELOW_GNN | -0.061 [-0.072, -0.050] BELOW_GNN |
| TP-direct | -0.097 [-0.105, -0.089] BELOW_GNN | -0.112 [-0.123, -0.102] BELOW_GNN | -0.235 [-0.251, -0.221] BELOW_GNN |
| TP-oracle | 0.000 [-0.003, 0.004] | 0.001 [-0.005, 0.006] | 0.001 [-0.008, 0.009] |

Descriptive means over seeds and queries (the twin and the GNN are the stored values):

| arm | recall@5 (arm / twin / GNN) | full_coverage@5 | hit@1 |
|---|---|---|---|
| TP | 0.781 / 0.744 / 0.806 | 0.668 / 0.623 / 0.694 | 0.835 / 0.794 / 0.911 |
| TP-bag | 0.765 / 0.744 / 0.806 | 0.656 / 0.623 / 0.694 | 0.791 / 0.794 / 0.911 |
| TP-pos | 0.781 / 0.744 / 0.806 | 0.667 / 0.623 / 0.694 | 0.839 / 0.794 / 0.911 |
| TP-hash | 0.786 / 0.744 / 0.806 | 0.674 / 0.623 / 0.694 | 0.851 / 0.794 / 0.911 |
| TP-direct | 0.709 / 0.744 / 0.806 | 0.582 / 0.623 / 0.694 | 0.677 / 0.794 / 0.911 |
| TP-oracle | 0.806 / 0.744 / 0.806 | 0.695 / 0.623 / 0.694 | 0.912 / 0.794 / 0.911 |

## Denominators

| metric | mean M(G) - M(T) | readable |
|---|---|---|
| recall@5 | 0.061 [0.056, 0.068] | True |
| full_coverage@5 | 0.071 [0.063, 0.080] | True |
| hit@1 | 0.118 [0.106, 0.130] | True |

## Contrasts

| contrast | of | paired difference |
|---|---|---|
| order | rho_bar(TP) - rho_bar(TP-bag) | 0.269 [0.222, 0.320] |
| untied_positions | rho_bar(TP-pos) - rho_bar(TP) | 0.008 [-0.025, 0.042] |
| text_vs_hash | rho_bar(TP) - rho_bar(TP-hash) | -0.101 [-0.150, -0.054] |
| latent_vs_direct | rho_bar(TP) - rho_bar(TP-direct) | 1.246 [1.117, 1.385] |
| ceiling_gap | rho_bar(TP-oracle) - rho_bar(TP) | 0.478 [0.416, 0.543] |

## By hop

| hop | queries | readable | TP | TP-bag | TP-pos | TP-hash | TP-direct | TP-oracle |
|---|---|---|---|---|---|---|---|---|
| hop=1 | 1000 | recall@5, full_coverage@5, hit@1 | 0.838 [0.626, 1.101] L8_HIGH | 0.681 [0.453, 0.913] L8_MID | 0.758 [0.544, 0.994] L8_HIGH | 0.850 [0.647, 1.104] L8_HIGH | -0.979 [-1.979, -0.424] L8_LOW | 1.334 [1.116, 1.728] L8_ABOVE_GNN |
| hop=2 | 1000 | recall@5, full_coverage@5, hit@1 | 0.676 [0.525, 0.815] L8_MID | 0.506 [0.324, 0.662] L8_MID | 0.665 [0.514, 0.801] L8_MID | 0.881 [0.736, 1.036] L8_HIGH | -2.319 [-3.046, -1.774] L8_LOW | 1.253 [1.134, 1.407] L8_ABOVE_GNN |
| hop=3 | 1000 | recall@5, full_coverage@5, hit@1 | 0.442 [0.368, 0.512] L8_MID | 0.127 [0.034, 0.208] L8_LOW | 0.468 [0.399, 0.535] L8_MID | 0.527 [0.462, 0.592] L8_MID | -0.240 [-0.365, -0.145] L8_LOW | 0.892 [0.835, 0.952] L8_HIGH |

## Anchors (descriptive)

- Topic entity: in the pool 0.996, among the seeds 0.981, in b0 0.935.
- The true chain reaches an in-pool gold from b0 0.893, from b1 0.270, from either 0.937. With directions swapped: 0.000.
- R* against the in-pool golds: recall 0.915, precision 0.828.
- Queries with an in-pool gold: 0.988.
- Types per query: mean 204.364, max 367. Walk entries per query: mean 5448.773, max 14970.
- TP's argmax type has the true chain's tokens on 0.591 of queries (the mean over seeds). NMI between the argmax token sequence and the qtype (k = 0): 0.776.

## Checks

- Scoring integrity: 0 mismatches against the stored per-query metrics on 3000 queries.
- Direction check: the declared chain reaches a gold on 0.937 of queries, and the swapped chain on 0.000.
- Repeat unit bit-identical: True.
- Units whose fit-set marginal log-likelihood fell between rounds: 0.

## What this does not say

Every arm is a cross-fitted measurement model on metaqa V2_GATE queries with gold labels. It is never a deployable or selected model, and it says nothing about another dataset. Nothing here enters QLS-U, the twin, a feature contract, M3, M4 or any selection.
