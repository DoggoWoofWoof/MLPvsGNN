# MP-Approx level 5 (MP-ORACLE): fixed one-hop neighbour-distribution moments, and the compiled-moment arm

Declared in `configs/mp_approx_l5.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l5/record.json` (git-ignored), assembled 2026-09-29T23:46:50Z at `f4fc3c2` from the host's per-dataset `read.json` files, without arithmetic.

Registered question: "After matching candidate exposure and inference-time graph information to modern graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?"

Level 5 of the MP-Approx ladder, on the MP-ORACLE track, declared with level 4 (typed paths) on the user's message of 2026-09-30. It replaces level 3's learned one-hop attention with fixed summaries of the same one-hop neighbourhood, and it carries the compiled-moment arm that level 3's l3_high entry licensed on 2wiki. As at every earlier level, the trained GNN's own outputs are the targets, which the proposal allows "only to measure approximation capacity". No probe, moment or kernel is a retriever, a teacher or a feature.

**Placement: host-native.** The moments were computed on the host CPU inside each dataset's job; every probe this file compares, B0-mlp, L3-mean and L3-att included, was fitted and read on host_gpu_det as a new draw. No number here is set beside a number made on the laptop.

## What was measured

- **Moments** (3366 columns per U_q row, float64 from level 3's float16 halo rows, stored float16): v's own X R (64); the mean, population standard deviation and maximum of the in-neighbours' fixed halo rows (3 x 322); the mean per family (structural, ner, knn; 3 x 322); two query kernels, weights exp((zc_u - max zc) / tau) with zc_u the neighbour's pool z-score of dense_cos (halo column 129), tau 1 and 0.25 (2 x 322); the row of the most query-similar neighbour (322); a relation kernel over structural entries with rel_mask 1, weights exp((rel_compat - max) / 0.1) (322); the mean edge feature eps_e (73); the maximum of the five edge attributes (5); log1p of the entry counts, all and per family (4). One hop only, no self entry; a row without an entry keeps only its own X R.
- **L5-mom**: level 0's MLP on [b_v | M_v], M_v standardised with level 0's rule per fold. No attention and no learned weighting. **L5-list**: the same with level 3's listwise objective.
- **L5-kern** (the compiled-moment arm): level 3's L3-att with the score replaced by a positive factorised kernel kappa = phi(q, v)^T psi(u, e), phi = softplus(W_phi [n_v | q_tilde]), psi = softplus(W_psi [n_u | eps_e]), 4 heads of rank 32. Because psi reads only the neighbour's fixed row and the edge, sum psi val^T and sum psi are node-level moments a deployed system could compile once; phi is the only query-time part. A laptop test holds the on-the-fly message equal to the compiled form.
- **Refitted from level 3** with its code, unchanged: B0-mlp (node-local), L3-mean (uniform weighting of the entries) and L3-att (learned query-conditioned one-hop attention, the ceiling).
- **Targets, recovery, U_q and bands** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k); rho_bar is the mean recovery over the readable metrics among recall@5, full_coverage@5 and hit@1. Metrics within U_q are an **upper bound** on full-pool metrics. L5_HIGH (>= 0.75, interval low >= 0.50), L5_LOW (<= 0.25, interval high <= 0.50), L5_MID otherwise.

## Readings

| dataset | queries | reading (L5-mom on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |
|---|---:|---|---|---|---|---|
| metaqa | 2,400 | **L5_LOW** | -0.551 [-0.670, -0.451] | recall@5, full_coverage@5, hit@1 | FIT_NOT_RANK (r, B0-mlp); FIT_NOT_RANK (r, L3-mean); FIT_NOT_RANK (r, L3-att); FIT_NOT_RANK (r, L5-mom); FIT_NOT_RANK (r, L5-kern) | moments_below_attention, kernel_below_attention, objective_adds |
| 2wiki | 6,290 | **L5_MID** | 0.599 [0.556, 0.642] | recall@5, full_coverage@5 | none | moments_add_over_mean, moments_below_attention, kernel_below_attention, objective_adds |
| squad | 5,841 | **NOT_READ** | n/a | none | SEED_BOUND | none |

### rho_bar per probe on r (host_gpu_det; one fit per cell, three GNN seeds x five folds)

| dataset | B0-mlp | L3-mean | L3-att | L5-mom | L5-list | L5-kern | ref:other_seed |
|---|---|---|---|---|---|---|---|
| metaqa | -0.650 [-0.769, -0.549] (L5_LOW) | -0.518 [-0.633, -0.423] (L5_LOW) | -0.076 [-0.161, 0.002] (L5_LOW) | -0.551 [-0.670, -0.451] (L5_LOW) | -0.188 [-0.274, -0.116] (L5_LOW) | -0.358 [-0.466, -0.268] (L5_LOW) | 1.064 [1.044, 1.086] (L5_HIGH) |
| 2wiki | 0.055 [-0.002, 0.109] (L5_LOW) | 0.562 [0.518, 0.606] (L5_MID) | 0.799 [0.759, 0.838] (L5_HIGH) | 0.599 [0.556, 0.642] (L5_MID) | 0.719 [0.679, 0.766] (L5_MID) | 0.746 [0.706, 0.789] (L5_MID) | 0.867 [0.837, 0.893] (L5_HIGH) |
| squad | n/a (NOT_READ) | n/a (NOT_READ) | n/a (NOT_READ) | n/a (NOT_READ) | n/a (NOT_READ) | n/a (NOT_READ) | n/a (NOT_READ) |

### Contrasts of rho_bar (paired bootstrap, level 0's resample matrix)

| contrast | of | metaqa | 2wiki | squad |
|---|---|---|---|---|
| moments_over_mean | L5-mom - L3-mean, r | -0.033 [-0.091, 0.026] | 0.037 [0.004, 0.070] | n/a |
| moments_vs_attention | L5-mom - L3-att, r | -0.475 [-0.551, -0.405] | -0.199 [-0.236, -0.162] | n/a |
| moments_over_node_local | L5-mom - B0-mlp, r | 0.099 [0.029, 0.170] | 0.544 [0.493, 0.601] | n/a |
| kernel_vs_attention | L5-kern - L3-att, r | -0.282 [-0.347, -0.225] | -0.053 [-0.081, -0.025] | n/a |
| kernel_over_moments | L5-kern - L5-mom, r | 0.193 [0.135, 0.253] | 0.147 [0.110, 0.182] | n/a |
| objective_moments | L5-list - L5-mom, r | 0.363 [0.292, 0.442] | 0.120 [0.089, 0.153] | n/a |
| edge_moments_vs_attention | L5-mom - L3-att, e | -0.059 [-0.069, -0.048] | -0.100 [-0.110, -0.088] | n/a |
| edge_moments_over_mean | L5-mom - L3-mean, e | -0.009 [-0.020, 0.000] | 0.017 [0.005, 0.028] | n/a |

### Shares of attention's gain over node-local (descriptive; read only where the denominator's interval lies above 0)

| share | of | metaqa | 2wiki | squad |
|---|---|---|---|---|
| moment_share | (L5-mom - B0-mlp) / (L3-att - B0-mlp), r | 0.173 [0.054, 0.277] | 0.732 [0.686, 0.779] | not read |
| kernel_share | (L5-kern - B0-mlp) / (L3-att - B0-mlp), r | 0.508 [0.416, 0.596] | 0.929 [0.892, 0.966] | not read |

What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):

- **kernel_below_attention**: the kernel_vs_attention interval below 0 -- compilation to a rank-32 factorised kernel loses part of the ceiling
- **moments_add_over_mean**: the moments_over_mean interval above 0 -- the spread, maxima, family split and fixed kernels add over the mean
- **moments_below_attention**: the moments_vs_attention interval below 0 -- learned query-conditioned weighting carries recovery that fixed moments do not
- **objective_adds**: the objective_moments interval above 0 -- the objective limits recovery, not only the features

### Level 3, as filed (bands and interpretation entries only)

| dataset | level 3 reading | level 3 interpretation |
|---|---|---|
| metaqa | L3_LOW | neighbourhood_adds, objective_adds, edge_effect_weighted |
| 2wiki | L3_HIGH | l3_high, l3_weighting_adds, neighbourhood_adds, objective_adds, edge_effect_weighted |
| squad | NOT_READ | none |

## metaqa

2,400 V2_GATE queries (the declared subsample: 800 per hop of the 19,738), 165,893 U_q rows (mean |U_q| 69.1); 1,967,659 one-hop entries; 449 rows without an entry; moments 165,893 x 3366 in 1.8 min on the host CPU (sha256 `6f69dd852882a7c0`).

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.061 | [0.055, 0.068] | yes |
| full_coverage@5 | 0.065 | [0.055, 0.074] | yes |
| hit@1 | 0.127 | [0.114, 0.140] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.562 | 0.686 | 0.550 | 0.474 | 0.541 | -0.582 | -0.481 | -0.888 | -0.650 [-0.769, -0.549] | L5_LOW |
| L3-mean | 0.611 | 0.722 | 0.591 | 0.516 | 0.561 | -0.452 | -0.387 | -0.716 | -0.518 [-0.633, -0.423] | L5_LOW |
| L3-att | 0.690 | 0.781 | 0.649 | 0.626 | 0.602 | 0.025 | 0.034 | -0.287 | -0.076 [-0.161, 0.002] | L5_LOW |
| L5-mom | 0.622 | 0.731 | 0.612 | 0.546 | 0.559 | -0.468 | -0.436 | -0.749 | -0.551 [-0.670, -0.451] | L5_LOW |
| L5-list | 0.292 | 0.516 | 0.451 | 0.567 | 0.538 | -0.099 | -0.120 | -0.344 | -0.188 [-0.274, -0.116] | L5_LOW |
| L5-kern | 0.660 | 0.762 | 0.638 | 0.587 | 0.579 | -0.250 | -0.235 | -0.590 | -0.358 [-0.466, -0.268] | L5_LOW |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.529 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L5_LOW |
| ref:other_seed | 0.590 | 0.702 | 0.590 | 0.865 | 0.671 | 1.066 | 1.051 | 1.076 | 1.064 [1.044, 1.086] | L5_HIGH |

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.446 | [0.432, 0.459] | yes |
| full_coverage@5 | 0.376 | [0.358, 0.391] | yes |
| hit@1 | 0.697 | [0.682, 0.711] | yes |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.790 | 0.856 | 0.750 | 0.893 | 0.520 | 0.733 | 0.680 | 0.600 | 0.671 [0.655, 0.687] | L5_MID |
| L3-mean | 0.812 | 0.869 | 0.764 | 0.902 | 0.542 | 0.761 | 0.713 | 0.637 | 0.704 [0.687, 0.720] | L5_MID |
| L3-att | 0.842 | 0.886 | 0.776 | 0.918 | 0.572 | 0.809 | 0.759 | 0.692 | 0.753 [0.738, 0.769] | L5_HIGH |
| L5-mom | 0.804 | 0.863 | 0.755 | 0.901 | 0.538 | 0.755 | 0.694 | 0.634 | 0.694 [0.678, 0.712] | L5_MID |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.246 | 0.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | L5_LOW |
| ref:other_seed | 0.599 | 0.716 | 0.620 | 0.907 | 0.545 | 0.833 | 0.797 | 0.687 | 0.772 [0.761, 0.783] | L5_HIGH |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.736 [0.731, 0.741]; e 0.732 [0.728, 0.737].

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | L3-mean rho_bar | L3-att rho_bar | L5-mom rho_bar | L5-list rho_bar | L5-kern rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|---|
| hop=1 | 800 | recall@5, full_coverage@5, hit@1 | -1.039 [-1.842, -0.582] | 0.061 [-0.366, 0.348] | 0.228 [-0.167, 0.521] | 0.113 [-0.256, 0.374] | 0.192 [-0.247, 0.468] | 0.101 [-0.315, 0.409] | 1.174 [1.067, 1.375] |
| hop=2 | 800 | recall@5, full_coverage@5, hit@1 | -1.049 [-1.466, -0.759] | -0.996 [-1.401, -0.703] | -0.650 [-1.022, -0.402] | -1.195 [-1.716, -0.848] | -0.255 [-0.532, -0.050] | -1.046 [-1.540, -0.719] | 1.086 [1.039, 1.150] |
| hop=3 | 800 | recall@5, full_coverage@5, hit@1 | -0.489 [-0.602, -0.386] | -0.459 [-0.580, -0.365] | 0.045 [-0.038, 0.120] | -0.454 [-0.575, -0.356] | -0.217 [-0.309, -0.140] | -0.225 [-0.327, -0.135] | 1.044 [1.022, 1.067] |
| first_support_STRUCT=no_gold_in_pool | 24 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,412 | recall@5, full_coverage@5, hit@1 | -0.619 [-0.798, -0.474] | -0.349 [-0.507, -0.223] | 0.056 [-0.070, 0.156] | -0.302 [-0.458, -0.183] | -0.144 [-0.276, -0.031] | -0.227 [-0.373, -0.112] | 1.080 [1.047, 1.120] |
| first_support_STRUCT=h2 | 809 | recall@5, full_coverage@5, hit@1 | -0.577 [-0.797, -0.418] | -0.570 [-0.787, -0.422] | -0.193 [-0.366, -0.065] | -0.688 [-0.923, -0.504] | -0.141 [-0.286, -0.025] | -0.452 [-0.660, -0.297] | 1.068 [1.035, 1.106] |
| first_support_STRUCT=h3 | 155 | recall@5, full_coverage@5, hit@1 | -0.898 [-1.256, -0.631] | -0.879 [-1.235, -0.610] | -0.189 [-0.462, 0.006] | -0.943 [-1.313, -0.661] | -0.417 [-0.666, -0.217] | -0.517 [-0.826, -0.266] | 1.013 [0.971, 1.057] |

## 2wiki

6,290 V2_GATE queries (all of them), 245,055 U_q rows (mean |U_q| 39.0); 1,580,750 one-hop entries; 31,654 rows without an entry; moments 245,055 x 3366 in 1.5 min on the host CPU (sha256 `c9fb9df6073a84dd`).

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.048 | [0.045, 0.051] | yes |
| full_coverage@5 | 0.104 | [0.097, 0.111] | yes |
| hit@1 | -0.007 | [-0.012, -0.002] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.440 | 0.627 | 0.467 | 0.468 | 0.758 | 0.101 | 0.010 | -2.644 (nr) | 0.055 [-0.002, 0.109] | L5_LOW |
| L3-mean | 0.584 | 0.718 | 0.564 | 0.595 | 0.794 | 0.589 | 0.535 | -1.424 (nr) | 0.562 [0.518, 0.606] | L5_MID |
| L3-att | 0.645 | 0.753 | 0.596 | 0.665 | 0.816 | 0.831 | 0.767 | -0.432 (nr) | 0.799 [0.759, 0.838] | L5_HIGH |
| L5-mom | 0.576 | 0.709 | 0.565 | 0.602 | 0.795 | 0.625 | 0.573 | -0.780 (nr) | 0.599 [0.556, 0.642] | L5_MID |
| L5-list | 0.455 | 0.645 | 0.543 | 0.617 | 0.792 | 0.731 | 0.708 | 1.553 (nr) | 0.719 [0.679, 0.766] | L5_MID |
| L5-kern | 0.636 | 0.748 | 0.594 | 0.662 | 0.813 | 0.772 | 0.720 | -0.735 (nr) | 0.746 [0.706, 0.789] | L5_MID |
| ref:twin | 0.000 | n/a | 0.000 | 0.000 | 0.704 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L5_LOW |
| ref:other_seed | 0.365 | 0.545 | 0.460 | 0.641 | 0.792 | 0.871 | 0.863 | 4.591 (nr) | 0.867 [0.837, 0.893] | L5_HIGH |

### Target e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN

| metric | gap M(G_k) - M(G0_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | 0.197 | [0.191, 0.202] | yes |
| full_coverage@5 | 0.367 | [0.357, 0.378] | yes |
| hit@1 | -0.034 | [-0.038, -0.029] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.573 | 0.707 | 0.604 | 0.692 | 0.753 | 0.542 | 0.495 | 2.598 (nr) | 0.518 [0.502, 0.534] | L5_MID |
| L3-mean | 0.703 | 0.779 | 0.685 | 0.796 | 0.800 | 0.726 | 0.673 | 2.397 (nr) | 0.700 [0.684, 0.715] | L5_MID |
| L3-att | 0.782 | 0.809 | 0.712 | 0.853 | 0.834 | 0.841 | 0.792 | 1.937 (nr) | 0.816 [0.802, 0.829] | L5_HIGH |
| L5-mom | 0.722 | 0.766 | 0.694 | 0.812 | 0.806 | 0.742 | 0.692 | 2.672 (nr) | 0.717 [0.702, 0.731] | L5_MID |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.652 | 0.000 | 0.000 | -0.000 (nr) | 0.000 [0.000, 0.000] | L5_LOW |
| ref:other_seed | 0.777 | 0.748 | 0.663 | 0.898 | 0.850 | 0.928 | 0.904 | 2.211 (nr) | 0.916 [0.907, 0.925] | L5_HIGH |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.577 [0.573, 0.581]; e 0.859 [0.858, 0.861].

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | L3-mean rho_bar | L3-att rho_bar | L5-mom rho_bar | L5-list rho_bar | L5-kern rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|---|
| gold_total=2 | 4,946 | recall@5, full_coverage@5 | 0.193 [0.128, 0.255] | 0.683 [0.629, 0.736] | 0.948 [0.898, 0.997] | 0.703 [0.652, 0.756] | 0.785 [0.732, 0.846] | 0.876 [0.829, 0.925] | 0.943 [0.907, 0.973] |
| gold_total>=3 | 1,344 | recall@5, full_coverage@5 | -0.232 [-0.343, -0.141] | 0.311 [0.233, 0.384] | 0.491 [0.420, 0.562] | 0.382 [0.307, 0.458] | 0.579 [0.509, 0.649] | 0.476 [0.402, 0.549] | 0.707 [0.660, 0.753] |
| first_support_STRUCT=no_gold_in_pool | 2 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 291 | recall@5, full_coverage@5 | 0.567 [-0.025, 1.209] | 0.568 [0.062, 1.236] | 0.238 [-0.548, 0.809] | 0.167 [-0.875, 0.718] | 0.455 [-0.086, 0.982] | 0.147 [-0.733, 0.683] | 0.640 [-0.407, 1.151] |
| first_support_STRUCT=h1 | 4,937 | recall@5, full_coverage@5 | 0.033 [-0.024, 0.087] | 0.559 [0.516, 0.605] | 0.805 [0.764, 0.846] | 0.606 [0.564, 0.649] | 0.729 [0.689, 0.777] | 0.755 [0.713, 0.800] | 0.878 [0.849, 0.905] |
| first_support_STRUCT=h2 | 603 | recall@5, full_coverage@5 | 0.476 [0.267, 0.718] | 0.649 [0.397, 0.931] | 0.766 [0.528, 0.989] | 0.548 [0.287, 0.802] | 0.512 [0.258, 0.741] | 0.633 [0.357, 0.886] | 0.615 [0.322, 0.819] |
| first_support_STRUCT=h3 | 457 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## squad

5,841 V2_GATE queries (all of them; the control), 176,880 U_q rows (mean |U_q| 30.3); 1,905,184 one-hop entries; 20,514 rows without an entry; moments 176,880 x 3366 in 1.4 min on the host CPU (sha256 `8f67e6011dbb181c`).

### Target r_k = z(G_k) - z(T_k), the GNN over its twin

| metric | gap M(G_k) - M(T_k) | 95% CI | readable |
|---|---:|---|---|
| recall@5 | -0.004 | [-0.007, -0.002] | no |
| full_coverage@5 | -0.004 | [-0.007, -0.002] | no |
| hit@1 | -0.009 | [-0.013, -0.005] | no |

| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0-mlp | 0.443 | 0.600 | 0.421 | 0.348 | 0.793 | -0.342 (nr) | -0.342 (nr) | -0.252 (nr) | n/a | NOT_READ |
| L3-mean | 0.489 | 0.639 | 0.471 | 0.393 | 0.803 | -0.066 (nr) | -0.066 (nr) | -0.101 (nr) | n/a | NOT_READ |
| L3-att | 0.517 | 0.666 | 0.505 | 0.431 | 0.811 | 0.000 (nr) | 0.000 (nr) | -0.031 (nr) | n/a | NOT_READ |
| L5-mom | 0.486 | 0.650 | 0.504 | 0.437 | 0.809 | 0.053 (nr) | 0.053 (nr) | 0.220 (nr) | n/a | NOT_READ |
| L5-list | 0.402 | 0.585 | 0.453 | 0.440 | 0.800 | 0.079 (nr) | 0.079 (nr) | -0.025 (nr) | n/a | NOT_READ |
| L5-kern | 0.525 | 0.674 | 0.513 | 0.444 | 0.813 | -0.184 (nr) | -0.184 (nr) | -0.019 (nr) | n/a | NOT_READ |
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
| L3-mean | 0.523 | 0.697 | 0.592 | 0.549 | 0.954 | 1.222 (nr) | 1.222 (nr) | 1.429 (nr) | n/a | NOT_READ |
| L3-att | 0.569 | 0.725 | 0.612 | 0.566 | 0.957 | 1.444 (nr) | 1.444 (nr) | 1.524 (nr) | n/a | NOT_READ |
| L5-mom | 0.530 | 0.679 | 0.614 | 0.548 | 0.955 | 2.000 (nr) | 2.000 (nr) | 1.476 (nr) | n/a | NOT_READ |
| ref:no_edge | 0.000 | n/a | 0.000 | 0.000 | 0.936 | -0.000 (nr) | -0.000 (nr) | -0.000 (nr) | n/a | NOT_READ |
| ref:other_seed | 0.219 | 0.530 | 0.495 | 0.499 | 0.944 | 1.000 (nr) | 1.000 (nr) | 1.571 (nr) | n/a | NOT_READ |

Seed reproducibility of the targets (mean of the three seed pairs): r 0.186 [0.181, 0.191]; e 0.527 [0.518, 0.536].

### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)

| stratum | queries | readable | B0-mlp rho_bar | L3-mean rho_bar | L3-att rho_bar | L5-mom rho_bar | L5-list rho_bar | L5-kern rho_bar | ref:other_seed rho_bar |
|---|---:|---|---|---|---|---|---|---|---|
| first_support_STRUCT=no_gold_in_pool | 123 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=none | 3,437 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h1 | 1,725 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h2 | 526 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| first_support_STRUCT=h3 | 30 | none | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## What this does not say

- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery measures capacity within U_q; it never describes a deployable model, and no probe output, moment or kernel enters any retriever, feature, teacher or selection.
- A model that used these moments or the factorised kernel without GNN outputs would be a competitor, and would need its own declaration under the QLS-U contract. This file does not open one.
- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high recovery are both results (readings.wording).
- No later level or arm is opened by any reading here.

## Reproducibility, placement and compute

| dataset | moments (min) | probes (min) | read (min) | device | driver | determinism warnings | level 3 mirror.json sha256 (host) | repeat | refits bit-identical to level 3 |
|---|---:|---:|---:|---|---|---:|---|---|---|
| metaqa | 1.8 | 36 | 1.1 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `049d5cd2c3719895` | unit bit-identical; moments byte-identical | 6 of 6 |
| 2wiki | 1.5 | 57 | 2.5 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `4f2db84a5a1b0d80` | n/a | 6 of 6 |
| squad | 1.4 | 43 | 2.3 | NVIDIA RTX 4500 Ada Generation | 596.71 | 0 | `de7c1917942862d2` | n/a | 6 of 6 |

Placement: `host_gpu_det` = {"host": "host", "device": "cuda", "threads": 8, "mode": "det", "tf32": false, "env": "mpr-cu128@62fc45e9e1ba"}, applied by level 3's host_placement in every host process, with CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch loads. Every host job verified level 3's mirror.json against every level 0 and level 3 file before reading a byte, and ran from one commit; the LF sha256 of every repository module it imported was filed and checked against the committed files at the file stage. Deviations: none.
