# MP-Approx level 6 (MP-ORACLE): the query-conditioned recurrent typed kernel

Declared in `configs/mp_approx_l6.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l6/record.json` (git-ignored), assembled 2026-09-30T07:00:33Z at `1251387` from the host's per-dataset `read.json` files, without arithmetic.

Registered question: "After matching candidate exposure and inference-time graph information to modern graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?"

Level 6 of the MP-Approx ladder, on the MP-ORACLE track, declared on the user's message of 2026-09-30 with the advisor's L6 analysis. The question, as the advisor put it: "Can the residual MetaQA message-passing advantage be recovered by composing the already-successful one-hop query-conditioned operator across multiple typed steps, without learned neighbor-state propagation?" As at every earlier level, the trained GNN's own outputs are the targets, which the proposal allows "only to measure approximation capacity". No probe or kernel is a retriever, a teacher or a feature.

**Placement: host-native.** Every probe this file compares, B0-mlp and L3-att included, was fitted and read on host_gpu_det as a new draw. No number here is set beside a number made on the laptop.

**Reach is one hop throughout.** Every message carries a neighbour's fixed row, never a neighbour state. The steps re-weight the same one-hop entries, each time conditioned on the receiver's updated state. Nothing here tests neighbour-state propagation.

## What was measured

- **The recurrence**: z_v^(0) = W_in n_v (64). At each of T steps a shared kernel gives the row's one-hop entries and its self entry positive weights summing to 1 per head (4 heads). The receiver side [n_v | z_v^(t) | q_tilde] feeds the kernel, which also reads each entry's family, relation text, relation attributes and direction. The messages M_typed = sum w val and M_STRUCT (the same weights over the structural entries, renormalised; 0 without one) update z^(t+1) = z^(t) + g([z | M_typed | M_STRUCT | q_tilde]). Level 3's readout reads [b_v | z^(T)]. Every parameter is shared across steps; no input names a dataset.
- **KERN form** (L6-1, L6-2, L6-3, L6-fix3, L6-list): level 5's factorised kernel phi(receiver)^T psi(u, e), rank 32 per head. psi reads only the neighbour's row and the edge, so C_v = sum psi val^T and c_v = sum psi are computed once per forward and reused at every step. A laptop test holds the step-by-step message equal to that compiled form. The halo row holds query-dependent columns, so these moments are per query and pool, not per node.
- **L6-fix3**: the kernel held at its step-0 value at every step (same depth, parameters and mixer). **L6-list**: L6-3 under level 3's listwise objective. **ATT form** (L6-att1, L6-att3): the same recurrence with level 3's exact attention score.
- **Refitted from level 3** with its code, unchanged: B0-mlp (node-local) and L3-att (learned query-conditioned one-hop attention, the frozen reference).
- **Targets, recovery, U_q and bands** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k); rho_bar is the mean recovery over the readable metrics among recall@5, full_coverage@5 and hit@1. Metrics within U_q are an **upper bound** on full-pool metrics. L6_HIGH (>= 0.75, interval low >= 0.50), L6_LOW (<= 0.25, interval high <= 0.50), L6_MID otherwise.

## Readings

| dataset | queries | reading (L6-3 on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |
|---|---:|---|---|---|---|---|
| metaqa | 2,400 | **L6_LOW** | -0.129 [-0.233, -0.045] | recall@5, full_coverage@5, hit@1 | FIT_NOT_RANK (r, B0-mlp); FIT_NOT_RANK (r, L3-att); FIT_NOT_RANK (r, L6-1); FIT_NOT_RANK (r, L6-2); FIT_NOT_RANK (r, L6-3); FIT_NOT_RANK (r, L6-fix3); FIT_NOT_RANK (r, L6-att1); FIT_NOT_RANK (r, L6-att3) | composition_recovers, state_matters, l6_below_attention, objective_adds, edge_effect_l6 |
| 2wiki | 6,290 | **L6_HIGH** | 0.821 [0.780, 0.862] | recall@5, full_coverage@5 | none | l6_high, composition_flat, l6_not_below_attention, objective_adds, edge_effect_l6 |
| squad | 5,841 | **NOT_READ** | n/a | none | SEED_BOUND | none |

Across datasets, from the per-dataset entries only (no number pooled): **advisor_ideal** no; **positive_control_held** yes; **negative_control_as_expected** yes.

### rho_bar per probe on r (host_gpu_det; one fit per cell, three GNN seeds x five folds)

| probe | metaqa | 2wiki | squad |
|---|---|---|---|
| B0-mlp | -0.650 [-0.769, -0.549] (L6_LOW) | 0.055 [-0.002, 0.109] (L6_LOW) | n/a (NOT_READ) |
| L3-att | -0.076 [-0.161, 0.002] (L6_LOW) | 0.799 [0.759, 0.838] (L6_HIGH) | n/a (NOT_READ) |
| L6-1 | -0.185 [-0.287, -0.096] (L6_LOW) | 0.804 [0.764, 0.845] (L6_HIGH) | n/a (NOT_READ) |
| L6-2 | -0.133 [-0.233, -0.053] (L6_LOW) | 0.816 [0.776, 0.858] (L6_HIGH) | n/a (NOT_READ) |
| L6-3 | -0.129 [-0.233, -0.045] (L6_LOW) | 0.821 [0.780, 0.862] (L6_HIGH) | n/a (NOT_READ) |
| L6-fix3 | -0.167 [-0.270, -0.083] (L6_LOW) | 0.807 [0.768, 0.846] (L6_HIGH) | n/a (NOT_READ) |
| L6-list | 0.202 [0.133, 0.265] (L6_LOW) | 0.891 [0.851, 0.933] (L6_HIGH) | n/a (NOT_READ) |
| L6-att1 | 0.033 [-0.055, 0.110] (L6_LOW) | 0.832 [0.789, 0.871] (L6_HIGH) | n/a (NOT_READ) |
| L6-att3 | -0.021 [-0.110, 0.055] (L6_LOW) | 0.830 [0.788, 0.867] (L6_HIGH) | n/a (NOT_READ) |
| ref:other_seed | 1.064 [1.044, 1.086] (L6_HIGH) | 0.867 [0.837, 0.893] (L6_HIGH) | n/a (NOT_READ) |

### Contrasts of rho_bar (paired bootstrap, level 0's resample matrix)

| contrast | of | metaqa | 2wiki | squad |
|---|---|---|---|---|
| composition_2 | L6-2 - L6-1, r | 0.051 [0.012, 0.090] | 0.012 [-0.011, 0.034] | not read |
| composition_3 | L6-3 - L6-1, r | 0.055 [0.014, 0.097] | 0.016 [-0.008, 0.039] | not read |
| depth_3_over_2 | L6-3 - L6-2, r | 0.004 [-0.030, 0.037] | 0.004 [-0.013, 0.020] | not read |
| state_kernel | L6-3 - L6-fix3, r | 0.038 [0.010, 0.066] | 0.013 [-0.002, 0.028] | not read |
| l6_vs_attention | L6-3 - L3-att, r | -0.053 [-0.106, -0.002] | 0.022 [-0.006, 0.051] | not read |
| one_step_vs_attention | L6-1 - L3-att, r | -0.109 [-0.163, -0.057] | 0.005 [-0.021, 0.033] | not read |
| att_composition_3 | L6-att3 - L6-att1, r | -0.054 [-0.097, -0.019] | -0.002 [-0.025, 0.022] | not read |
| att_l6_vs_attention | L6-att3 - L3-att, r | 0.055 [0.004, 0.106] | 0.031 [0.005, 0.059] | not read |
| objective | L6-list - L6-3, r | 0.331 [0.268, 0.405] | 0.070 [0.042, 0.100] | not read |
| l6_over_node_local | L6-3 - B0-mlp, r | 0.521 [0.441, 0.604] | 0.765 [0.709, 0.829] | not read |
| edge_l6_vs_attention | L6-3 - L3-att, e | -0.001 [-0.010, 0.008] | 0.014 [0.006, 0.022] | not read |

### Shares (descriptive; read only where the denominator's interval lies above 0)

| share | of | metaqa | 2wiki | squad |
|---|---|---|---|---|
| l6_share | (L6-3 - B0-mlp) / (L3-att - B0-mlp), r | 0.907 [0.819, 0.996] | 1.029 [0.992, 1.070] | not read |
| gap_closed | (L6-3 - L3-att) / (ref:other_seed - L3-att), r | -0.047 [-0.092, -0.002] | 0.318 [-0.137, 0.909] | not read |

What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):

- **composition_flat**: the composition_3 interval contains 0 -- no measurable gain from composing the one-hop kernel
- **composition_recovers**: the composition_3 interval above 0 -- three composed steps recover more than one step of the same form
- **edge_effect_l6**: the band of L6-3 on e is L6_HIGH, or edge_l6_vs_attention's interval reaches 0 or above
- **l6_below_attention**: the l6_vs_attention interval below 0
- **l6_high**: primary L6_HIGH: composing the one-hop typed kernel over three receiver-state steps predicts the GNN's ranking effect within U_q on that dataset (an upper bound, not a full-pool claim)
- **l6_not_below_attention**: the l6_vs_attention interval's upper end at or above 0
- **objective_adds**: the objective interval above 0 -- the objective limits recovery, not only the operator
- **state_matters**: the state_kernel interval above 0 -- re-weighting on the updated state carries recovery that a kernel held at step 0, with the same depth and mixer, does not

### Level 3, as filed (bands and interpretation entries only)

| dataset | level 3 reading | level 3 interpretation |
|---|---|---|
| metaqa | L3_LOW | neighbourhood_adds, objective_adds, edge_effect_weighted |
| 2wiki | L3_HIGH | l3_high, l3_weighting_adds, neighbourhood_adds, objective_adds, edge_effect_weighted |
| squad | NOT_READ | none |

## metaqa

2,400 V2_GATE queries (the declared subsample: 800 per hop of the 19,738), 165,893 U_q rows (mean |U_q| 69.1); 1,967,659 one-hop entries.

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.061 | [0.055, 0.068] | yes |
| full_coverage@5 | 0.065 | [0.055, 0.074] | yes |
| hit@1 | 0.127 | [0.114, 0.140] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.562 | 0.686 | 0.550 | 0.474 | 0.541 | -0.582 | -0.481 | -0.888 | -0.650 [-0.769, -0.549] | L6_LOW |
| L3-att | 0.690 | 0.781 | 0.649 | 0.626 | 0.602 | 0.025 | 0.034 | -0.287 | -0.076 [-0.161, 0.002] | L6_LOW |
| L6-1 | 0.683 | 0.779 | 0.656 | 0.619 | 0.594 | -0.066 | -0.075 | -0.413 | -0.185 [-0.287, -0.096] | L6_LOW |
| L6-2 | 0.693 | 0.784 | 0.659 | 0.623 | 0.601 | -0.036 | -0.030 | -0.334 | -0.133 [-0.233, -0.053] | L6_LOW |
| L6-3 | 0.696 | 0.785 | 0.658 | 0.626 | 0.603 | -0.012 | -0.038 | -0.337 | -0.129 [-0.233, -0.045] | L6_LOW |
| L6-fix3 | 0.692 | 0.782 | 0.657 | 0.612 | 0.598 | -0.044 | -0.053 | -0.403 | -0.167 [-0.270, -0.083] | L6_LOW |
| L6-list | 0.431 | 0.603 | 0.505 | 0.642 | 0.579 | 0.267 | 0.282 | 0.057 | 0.202 [0.133, 0.265] | L6_LOW |
| L6-att1 | 0.707 | 0.791 | 0.661 | 0.642 | 0.612 | 0.137 | 0.160 | -0.198 | 0.033 [-0.055, 0.110] | L6_LOW |
| L6-att3 | 0.708 | 0.792 | 0.666 | 0.640 | 0.610 | 0.085 | 0.079 | -0.228 | -0.021 [-0.110, 0.055] | L6_LOW |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.529 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L6_LOW |
| ref:other_seed | 0.590 | 0.702 | 0.590 | 0.865 | 0.671 | 1.066 | 1.051 | 1.076 | 1.064 [1.044, 1.086] | L6_HIGH |

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.446 | [0.432, 0.459] | yes |
| full_coverage@5 | 0.376 | [0.358, 0.391] | yes |
| hit@1 | 0.697 | [0.682, 0.711] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.790 | 0.856 | 0.750 | 0.893 | 0.520 | 0.733 | 0.680 | 0.600 | 0.671 [0.655, 0.687] | L6_MID |
| L3-att | 0.842 | 0.886 | 0.776 | 0.918 | 0.572 | 0.809 | 0.759 | 0.692 | 0.753 [0.738, 0.769] | L6_HIGH |
| L6-3 | 0.845 | 0.889 | 0.778 | 0.915 | 0.576 | 0.809 | 0.764 | 0.683 | 0.752 [0.737, 0.767] | L6_HIGH |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.246 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L6_LOW |
| ref:other_seed | 0.599 | 0.716 | 0.620 | 0.907 | 0.545 | 0.833 | 0.797 | 0.687 | 0.772 [0.761, 0.783] | L6_HIGH |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.736 [0.731, 0.741]; e 0.732 [0.728, 0.737].

### Strata (descriptive; a stratum's rho and contrasts are shown only where it has a readable metric, and none is a reading)

| stratum | queries | readable | B0-mlp rho_bar | L3-att rho_bar | L6-1 rho_bar | L6-2 rho_bar | L6-3 rho_bar | L6-fix3 rho_bar | L6-list rho_bar | L6-att1 rho_bar | L6-att3 rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| hop=1 | 800 | recall@5, full_coverage@5, hit@1 | -1.039 [-1.842, -0.582] | 0.228 [-0.167, 0.521] | 0.139 [-0.278, 0.425] | -0.007 [-0.453, 0.303] | -0.125 [-0.649, 0.189] | 0.044 [-0.350, 0.304] | 0.427 [0.084, 0.669] | 0.236 [-0.180, 0.512] | 0.273 [-0.106, 0.558] | 1.174 [1.067, 1.375] |
| hop=2 | 800 | recall@5, full_coverage@5, hit@1 | -1.049 [-1.466, -0.759] | -0.650 [-1.022, -0.402] | -0.918 [-1.368, -0.613] | -0.802 [-1.225, -0.517] | -0.756 [-1.164, -0.481] | -0.844 [-1.248, -0.566] | -0.006 [-0.208, 0.152] | -0.511 [-0.862, -0.280] | -0.653 [-1.036, -0.406] | 1.086 [1.039, 1.150] |
| hop=3 | 800 | recall@5, full_coverage@5, hit@1 | -0.489 [-0.602, -0.386] | 0.045 [-0.038, 0.120] | -0.023 [-0.121, 0.061] | 0.038 [-0.049, 0.120] | 0.045 [-0.043, 0.125] | -0.004 [-0.098, 0.076] | 0.230 [0.151, 0.299] | 0.158 [0.077, 0.238] | 0.116 [0.031, 0.192] | 1.044 [1.022, 1.067] |
| first_support_STRUCT=no_gold_in_pool | 24 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,412 | recall@5, full_coverage@5, hit@1 | -0.619 [-0.798, -0.474] | 0.056 [-0.070, 0.156] | -0.047 [-0.177, 0.063] | -0.035 [-0.175, 0.077] | -0.080 [-0.230, 0.037] | -0.074 [-0.211, 0.036] | 0.196 [0.080, 0.292] | 0.100 [-0.022, 0.212] | 0.078 [-0.053, 0.185] | 1.080 [1.047, 1.120] |
| first_support_STRUCT=h2 | 809 | recall@5, full_coverage@5, hit@1 | -0.577 [-0.797, -0.418] | -0.193 [-0.366, -0.065] | -0.309 [-0.517, -0.157] | -0.228 [-0.416, -0.089] | -0.175 [-0.361, -0.036] | -0.245 [-0.433, -0.102] | 0.220 [0.108, 0.314] | -0.039 [-0.207, 0.087] | -0.095 [-0.260, 0.030] | 1.068 [1.035, 1.106] |
| first_support_STRUCT=h3 | 155 | recall@5, full_coverage@5, hit@1 | -0.898 [-1.256, -0.631] | -0.189 [-0.462, 0.006] | -0.296 [-0.605, -0.059] | -0.201 [-0.484, 0.027] | -0.174 [-0.451, 0.037] | -0.262 [-0.556, -0.035] | 0.173 [-0.010, 0.343] | 0.006 [-0.233, 0.201] | -0.133 [-0.393, 0.077] | 1.013 [0.971, 1.057] |

| stratum | composition_3 | l6_vs_attention | att_composition_3 |
|---|---|---|---|
| hop=1 | -0.265 [-0.498, -0.127] | -0.354 [-0.668, -0.167] | 0.037 [-0.133, 0.242] |
| hop=2 | 0.162 [0.036, 0.306] | -0.105 [-0.280, 0.074] | -0.142 [-0.272, -0.034] |
| hop=3 | 0.068 [0.027, 0.112] | -0.000 [-0.054, 0.051] | -0.042 [-0.080, -0.009] |
| first_support_STRUCT=no_gold_in_pool | not read | not read | not read |
| first_support_STRUCT=h1 | -0.033 [-0.091, 0.017] | -0.135 [-0.211, -0.066] | -0.022 [-0.080, 0.030] |
| first_support_STRUCT=h2 | 0.134 [0.055, 0.222] | 0.019 [-0.082, 0.114] | -0.055 [-0.126, 0.009] |
| first_support_STRUCT=h3 | 0.122 [0.033, 0.226] | 0.015 [-0.103, 0.136] | -0.140 [-0.228, -0.061] |

## 2wiki

6,290 V2_GATE queries (all of them), 245,055 U_q rows (mean |U_q| 39.0); 1,580,750 one-hop entries.

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.048 | [0.045, 0.051] | yes |
| full_coverage@5 | 0.104 | [0.097, 0.111] | yes |
| hit@1 | -0.007 | [-0.012, -0.002] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.440 | 0.627 | 0.467 | 0.468 | 0.758 | 0.101 | 0.010 | -2.644 (nr) | 0.055 [-0.002, 0.109] | L6_LOW |
| L3-att | 0.645 | 0.753 | 0.596 | 0.665 | 0.816 | 0.831 | 0.767 | -0.432 (nr) | 0.799 [0.759, 0.838] | L6_HIGH |
| L6-1 | 0.652 | 0.759 | 0.602 | 0.661 | 0.817 | 0.829 | 0.779 | -0.780 (nr) | 0.804 [0.764, 0.845] | L6_HIGH |
| L6-2 | 0.655 | 0.760 | 0.604 | 0.665 | 0.819 | 0.843 | 0.789 | -0.515 (nr) | 0.816 [0.776, 0.858] | L6_HIGH |
| L6-3 | 0.656 | 0.761 | 0.607 | 0.663 | 0.819 | 0.847 | 0.795 | -0.545 (nr) | 0.821 [0.780, 0.862] | L6_HIGH |
| L6-fix3 | 0.655 | 0.761 | 0.606 | 0.664 | 0.819 | 0.835 | 0.779 | -0.538 (nr) | 0.807 [0.768, 0.846] | L6_HIGH |
| L6-list | 0.576 | 0.715 | 0.575 | 0.666 | 0.816 | 0.911 | 0.871 | 0.333 (nr) | 0.891 [0.851, 0.933] | L6_HIGH |
| L6-att1 | 0.655 | 0.761 | 0.608 | 0.667 | 0.818 | 0.864 | 0.799 | -0.924 (nr) | 0.832 [0.789, 0.871] | L6_HIGH |
| L6-att3 | 0.659 | 0.763 | 0.610 | 0.670 | 0.819 | 0.857 | 0.802 | -0.886 (nr) | 0.830 [0.788, 0.867] | L6_HIGH |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.704 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L6_LOW |
| ref:other_seed | 0.365 | 0.545 | 0.460 | 0.641 | 0.792 | 0.871 | 0.863 | 4.591 (nr) | 0.867 [0.837, 0.893] | L6_HIGH |

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.197 | [0.191, 0.202] | yes |
| full_coverage@5 | 0.367 | [0.357, 0.378] | yes |
| hit@1 | -0.034 | [-0.038, -0.029] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.573 | 0.707 | 0.604 | 0.692 | 0.753 | 0.542 | 0.495 | 2.598 (nr) | 0.518 [0.502, 0.534] | L6_MID |
| L3-att | 0.782 | 0.809 | 0.712 | 0.853 | 0.834 | 0.841 | 0.792 | 1.937 (nr) | 0.816 [0.802, 0.829] | L6_HIGH |
| L6-3 | 0.795 | 0.820 | 0.716 | 0.861 | 0.839 | 0.852 | 0.808 | 1.983 (nr) | 0.830 [0.817, 0.843] | L6_HIGH |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.652 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L6_LOW |
| ref:other_seed | 0.777 | 0.748 | 0.663 | 0.898 | 0.850 | 0.928 | 0.904 | 2.211 (nr) | 0.916 [0.907, 0.925] | L6_HIGH |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.577 [0.573, 0.581]; e 0.859 [0.858, 0.861].

### Strata (descriptive; a stratum's rho and contrasts are shown only where it has a readable metric, and none is a reading)

| stratum | queries | readable | B0-mlp rho_bar | L3-att rho_bar | L6-1 rho_bar | L6-2 rho_bar | L6-3 rho_bar | L6-fix3 rho_bar | L6-list rho_bar | L6-att1 rho_bar | L6-att3 rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| gold_total=2 | 4,946 | recall@5, full_coverage@5 | 0.193 [0.128, 0.255] | 0.948 [0.898, 0.997] | 0.938 [0.890, 0.988] | 0.951 [0.903, 1.004] | 0.954 [0.909, 1.007] | 0.937 [0.890, 0.987] | 0.996 [0.946, 1.046] | 0.979 [0.930, 1.032] | 0.961 [0.914, 1.008] | 0.943 [0.907, 0.973] |
| gold_total>=3 | 1,344 | recall@5, full_coverage@5 | -0.232 [-0.343, -0.141] | 0.491 [0.420, 0.562] | 0.525 [0.459, 0.593] | 0.537 [0.469, 0.605] | 0.542 [0.477, 0.611] | 0.538 [0.469, 0.603] | 0.673 [0.610, 0.740] | 0.526 [0.455, 0.593] | 0.557 [0.489, 0.625] | 0.707 [0.660, 0.753] |
| first_support_STRUCT=no_gold_in_pool | 2 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 291 | recall@5, full_coverage@5 | 0.567 [-0.025, 1.209] | 0.238 [-0.548, 0.809] | 0.517 [-0.048, 1.102] | 0.793 [0.305, 1.605] | 0.722 [0.245, 1.473] | 0.712 [0.226, 1.459] | 0.588 [0.066, 1.138] | 0.465 [-0.320, 1.096] | 0.680 [0.072, 1.367] | 0.640 [-0.407, 1.151] |
| first_support_STRUCT=h1 | 4,937 | recall@5, full_coverage@5 | 0.033 [-0.024, 0.087] | 0.805 [0.764, 0.846] | 0.805 [0.766, 0.848] | 0.815 [0.775, 0.859] | 0.820 [0.781, 0.861] | 0.806 [0.766, 0.847] | 0.896 [0.858, 0.937] | 0.835 [0.794, 0.875] | 0.831 [0.791, 0.868] | 0.878 [0.849, 0.905] |
| first_support_STRUCT=h2 | 603 | recall@5, full_coverage@5 | 0.476 [0.267, 0.718] | 0.766 [0.528, 0.989] | 0.784 [0.559, 1.020] | 0.756 [0.518, 0.995] | 0.720 [0.500, 0.949] | 0.720 [0.485, 0.968] | 0.791 [0.573, 1.030] | 0.721 [0.484, 0.944] | 0.766 [0.537, 0.986] | 0.615 [0.322, 0.819] |
| first_support_STRUCT=h3 | 457 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| stratum | composition_3 | l6_vs_attention | att_composition_3 |
|---|---|---|---|
| gold_total=2 | 0.016 [-0.013, 0.045] | 0.007 [-0.026, 0.042] | -0.018 [-0.046, 0.011] |
| gold_total>=3 | 0.017 [-0.026, 0.054] | 0.052 [-0.002, 0.103] | 0.031 [-0.009, 0.072] |
| first_support_STRUCT=no_gold_in_pool | not read | not read | not read |
| first_support_STRUCT=none | 0.205 [-0.144, 0.800] | 0.483 [0.168, 1.338] | 0.215 [-0.167, 0.802] |
| first_support_STRUCT=h1 | 0.014 [-0.011, 0.037] | 0.015 [-0.013, 0.044] | -0.003 [-0.026, 0.020] |
| first_support_STRUCT=h2 | -0.064 [-0.246, 0.098] | -0.046 [-0.261, 0.164] | 0.046 [-0.107, 0.197] |
| first_support_STRUCT=h3 | not read | not read | not read |

## squad

5,841 V2_GATE queries (all of them; the control), 176,880 U_q rows (mean |U_q| 30.3); 1,905,184 one-hop entries.

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | -0.004 | [-0.007, -0.002] | no |
| full_coverage@5 | -0.004 | [-0.007, -0.002] | no |
| hit@1 | -0.009 | [-0.013, -0.005] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.443 | 0.600 | 0.421 | 0.348 | 0.793 | -0.342 (nr) | -0.342 (nr) | -0.252 (nr) | n/a | NOT_READ |
| L3-att | 0.517 | 0.666 | 0.505 | 0.431 | 0.811 | 0.000 (nr) | 0.000 (nr) | -0.031 (nr) | n/a | NOT_READ |
| L6-1 | 0.542 | 0.688 | 0.530 | 0.471 | 0.817 | 0.039 (nr) | 0.039 (nr) | 0.101 (nr) | n/a | NOT_READ |
| L6-2 | 0.543 | 0.689 | 0.529 | 0.477 | 0.817 | 0.066 (nr) | 0.066 (nr) | 0.050 (nr) | n/a | NOT_READ |
| L6-3 | 0.541 | 0.688 | 0.535 | 0.482 | 0.818 | 0.026 (nr) | 0.026 (nr) | 0.151 (nr) | n/a | NOT_READ |
| L6-fix3 | 0.542 | 0.688 | 0.534 | 0.477 | 0.818 | 0.224 (nr) | 0.224 (nr) | 0.126 (nr) | n/a | NOT_READ |
| L6-list | 0.469 | 0.630 | 0.470 | 0.460 | 0.809 | -0.092 (nr) | -0.092 (nr) | 0.308 (nr) | n/a | NOT_READ |
| L6-att1 | 0.539 | 0.686 | 0.530 | 0.472 | 0.817 | 0.197 (nr) | 0.197 (nr) | 0.107 (nr) | n/a | NOT_READ |
| L6-att3 | 0.539 | 0.686 | 0.529 | 0.467 | 0.816 | 0.026 (nr) | 0.026 (nr) | 0.031 (nr) | n/a | NOT_READ |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.746 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | -0.245 | 0.167 | 0.278 | 0.316 | 0.724 | 2.105 (nr) | 2.105 (nr) | 1.836 (nr) | n/a | NOT_READ |

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | -0.001 | [-0.002, 0.001] | no |
| full_coverage@5 | -0.001 | [-0.002, 0.001] | no |
| hit@1 | -0.001 | [-0.002, 0.000] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.387 | 0.612 | 0.507 | 0.470 | 0.949 | 0.222 (nr) | 0.222 (nr) | 1.000 (nr) | n/a | NOT_READ |
| L3-att | 0.569 | 0.725 | 0.612 | 0.566 | 0.957 | 1.444 (nr) | 1.444 (nr) | 1.524 (nr) | n/a | NOT_READ |
| L6-3 | 0.601 | 0.742 | 0.633 | 0.587 | 0.959 | 1.111 (nr) | 1.111 (nr) | 1.381 (nr) | n/a | NOT_READ |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.936 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | 0.219 | 0.530 | 0.495 | 0.499 | 0.944 | 1.000 (nr) | 1.000 (nr) | 1.571 (nr) | n/a | NOT_READ |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.186 [0.181, 0.191]; e 0.527 [0.518, 0.536].

### Strata (descriptive; a stratum's rho and contrasts are shown only where it has a readable metric, and none is a reading)

| stratum | queries | readable | B0-mlp rho_bar | L3-att rho_bar | L6-1 rho_bar | L6-2 rho_bar | L6-3 rho_bar | L6-fix3 rho_bar | L6-list rho_bar | L6-att1 rho_bar | L6-att3 rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| first_support_STRUCT=no_gold_in_pool | 123 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 3,437 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,725 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h2 | 526 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h3 | 30 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| stratum | composition_3 | l6_vs_attention | att_composition_3 |
|---|---|---|---|
| first_support_STRUCT=no_gold_in_pool | not read | not read | not read |
| first_support_STRUCT=none | not read | not read | not read |
| first_support_STRUCT=h1 | not read | not read | not read |
| first_support_STRUCT=h2 | not read | not read | not read |
| first_support_STRUCT=h3 | not read | not read | not read |

## What this does not say

- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery measures capacity within U_q; it never describes a deployable model, and no probe output or kernel enters any retriever, feature, teacher or selection.
- Reach is one hop throughout. A LOW reading does not show that multi-hop reach is needed; it shows only that iterated receiver-state re-weighting of the one-hop neighbourhood does not recover the residual. Neighbour-state propagation is not tested.
- A model that used this recurrence without GNN outputs would be a competitor, and would need its own declaration under the QLS-U contract. This file does not open one, nor the decomposition, the deployable 2wiki kernel or the latency retiming.
- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high recovery are both results (readings.wording).

## Reproducibility, placement and compute

| dataset | probes (min) | read (min) | device | driver | determinism warnings | level 3 mirror.json sha256 (host) | repeat | refits bit-identical to level 3 |
|---|---:|---:|---|---|---:|---|---|---|
| metaqa | 65 | 1.1 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `049d5cd2c3719895` | unit bit-identical | 4 of 4 |
| 2wiki | 92 | 2.7 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `4f2db84a5a1b0d80` | n/a | 4 of 4 |
| squad | 81 | 2.5 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `de7c1917942862d2` | n/a | 4 of 4 |

Parameters per arm: L6-1 253,569, L6-2 253,569, L6-3 253,569, L6-fix3 253,569, L6-list 253,569, L6-att1 208,257, L6-att3 208,257.

Placement: `host_gpu_det` = {"host": "host", "device": "cuda", "threads": 8, "mode": "det", "tf32": false, "env": "mpr-cu128@62fc45e9e1ba"}, applied by level 3's host_placement in every host process, with CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch loads. Every host job verified level 3's mirror.json against every level 0 and level 3 file before reading a byte, and ran from one commit; the LF sha256 of every repository module it imported was filed and checked against the committed files at the file stage. Deviations: a push at 2026-09-30T06:43Z, while the squad and 2wiki jobs ran, added the host-mirror commit (1251387) and the six-base work-2 commit (e5be296) to the workspace: six tracked files and one manifest, none imported or read by level 6 (each read.json's module_sha256 records the modules that ran).
